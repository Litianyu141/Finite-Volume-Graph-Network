import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
from utils.utilities import decompose_and_trans_node_attr_to_cell_attr_graph
from torch_geometric.data import Data


class EdgeBlock(nn.Module):

    def __init__(self, custom_func=None, dual_edge=False):

        super(EdgeBlock, self).__init__()
        self.net = custom_func
        self.dual_edge = dual_edge

    def forward(self, graph):

        cell_attr, edge_index, edge_attr, _ = (
            decompose_and_trans_node_attr_to_cell_attr_graph(
                graph, has_changed_node_attr_to_cell_attr=True
            )
        )
        senders_idx = edge_index[0]
        receivers_idx = edge_index[1]

        if self.dual_edge:

            twoway_senders = torch.cat((senders_idx, receivers_idx), dim=0)
            twoway_recivers = torch.cat((receivers_idx, senders_idx), dim=0)

            edges_to_collect = []
            senders_attr = cell_attr[twoway_senders]
            receivers_attr = cell_attr[twoway_recivers]

            edges_to_collect.append(senders_attr)
            edges_to_collect.append(receivers_attr)
            edges_to_collect.append(edge_attr)

            collected_edges = torch.cat(edges_to_collect, dim=1)

        else:
            edges_to_collect = []
            senders_attr = cell_attr[senders_idx]
            receivers_attr = cell_attr[receivers_idx]

            edges_to_collect.append(senders_attr)
            edges_to_collect.append(receivers_attr)
            edges_to_collect.append(edge_attr)

            collected_edges = torch.cat(edges_to_collect, dim=1)

        edge_attr_ = self.net(collected_edges)  # Update

        return Data(x=cell_attr, edge_attr=edge_attr_, edge_index=edge_index)


class CellBlock(nn.Module):

    def __init__(
        self,
        input_size,
        attention_size,
        mp_times=2,
        MultiHead=1,
        custom_func=None,
        dual_edge=False,
    ):

        super(CellBlock, self).__init__()
        if attention_size % MultiHead > 0:
            raise ValueError("MultiHead must be the factor of attention_size")
        # self.Linear = nn.Sequential(nn.LazyLinear(1),nn.LeakyReLU(negative_slope=0.2))
        # self.Linear_projection = nn.ModuleList([self.Linear for i in range(MultiHead)])
        self.net = custom_func
        self.mp_times = mp_times
        self.dual_edge = dual_edge

    def forward(self, graph, graph_node):

        # Decompose graph
        edge_attr = graph.edge_attr
        cells_to_collect = []
        """
        receivers_idx = graph.cell_face[0]
        num_nodes = graph.num_nodes #num_nodes stands for the number of cells
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)
        """
        if self.dual_edge:
            twoway_edge_attr = edge_attr
        else:
            twoway_edge_attr = torch.cat(torch.chunk(edge_attr, 2, dim=-1), dim=0)

        # if self.attention:
        #     senders_idx,receivers_idx = graph.edge_index
        #     twoway_cell_connections = torch.cat([senders_idx,receivers_idx],dim=0)
        #     senders_node_idx,receivers_node_idx = graph_node.edge_index
        #     twoway_node_connections = torch.cat([senders_node_idx,receivers_node_idx],dim=0)
        #     attention_input = []
        #     for index,attention_module in enumerate(self.Linear_projection):
        #         attention_input.append(attention_module(twoway_edge_attr))

        #     attention_factor = F.softmax(torch.cat(attention_input,dim=1), dim=0)
        #     node_agg_received_edges = scatter_mean(torch.mul(twoway_edge_attr,attention_factor),twoway_node_connections, dim=0, dim_size=graph_node.num_nodes)
        #     cell_agg_received_nodes=torch.index_select(node_agg_received_edges,0,graph_node.face[0])
        #     for i in range(1,3):
        #         cell_agg_received_nodes+=torch.index_select(node_agg_received_edges,0,graph_node.face[i])
        #     cell_agg_received_edges = scatter_add(torch.mul(twoway_edge_attr,attention_factor),twoway_cell_connections, dim=0, dim_size=graph.num_nodes)
        # else:

        if self.mp_times >= 2:
            senders_node_idx, receivers_node_idx = graph_node.edge_index
            twoway_node_connections = torch.cat(
                [senders_node_idx, receivers_node_idx], dim=0
            )

            """ first message aggregation to nodes"""
            node_agg_received_edges = scatter_add(
                twoway_edge_attr,
                twoway_node_connections,
                dim=0,
                dim_size=graph_node.num_nodes,
            )

            """ second message aggregation to cells"""
            cell_agg = (
                torch.index_select(node_agg_received_edges, 0, graph_node.face[0])
                + torch.index_select(node_agg_received_edges, 0, graph_node.face[1])
                + torch.index_select(node_agg_received_edges, 0, graph_node.face[2])
            ) / 3.0
        else:
            senders_idx, receivers_idx = graph.edge_index
            twoway_cell_connections = torch.cat([senders_idx, receivers_idx], dim=0)
            cell_agg = scatter_add(
                twoway_edge_attr,
                twoway_cell_connections,
                dim=0,
                dim_size=graph.num_nodes,
            )

        cells_to_collect.append(graph.x)
        cells_to_collect.append(cell_agg)
        collected_nodes = torch.cat(cells_to_collect, dim=-1)
        x = self.net(collected_nodes)

        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)
