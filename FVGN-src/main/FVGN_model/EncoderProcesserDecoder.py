import torch.nn as nn
import torch
from .GN_blocks import EdgeBlock, CellBlock
from utils.utilities import (
    decompose_and_trans_node_attr_to_cell_attr_graph,
    copy_geometric_data,
)
from torch_geometric.data import Data


# from torch_geometric.nn import InstanceNorm
def build_mlp(
    in_size,
    hidden_size,
    out_size,
    drop_out=True,
    lay_norm=True,
    dropout_prob=0.2,
    nn_act="SiLU",
):
    if nn_act == "SiLU":
        nn_act_fn = nn.SiLU()
    elif nn_act == "ReLU":
        nn_act_fn = nn.ReLU()
    else:
        raise NotImplementedError

    if drop_out:
        module = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn_act_fn,
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn_act_fn,
            nn.Linear(hidden_size, out_size),
        )
    else:
        module = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn_act_fn,
            nn.Linear(hidden_size, hidden_size),
            nn_act_fn,
            nn.Linear(hidden_size, out_size),
        )
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Module):

    def __init__(
        self, edge_input_size=128, cell_input_size=128, hidden_size=128, nn_act="SiLU"
    ):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(
            edge_input_size, hidden_size, hidden_size, drop_out=False, nn_act=nn_act
        )
        self.cb_encoder = build_mlp(
            cell_input_size, hidden_size, hidden_size, drop_out=False, nn_act=nn_act
        )

    def forward(self, graph):

        cell_attr, edge_index, edge_attr, _ = (
            decompose_and_trans_node_attr_to_cell_attr_graph(
                graph, has_changed_node_attr_to_cell_attr=False
            )
        )

        cell_ = self.cb_encoder(cell_attr)
        edge_ = self.eb_encoder(edge_attr)

        return Data(x=cell_, edge_attr=edge_, edge_index=edge_index)


class GnBlock(nn.Module):

    def __init__(
        self,
        hidden_size=128,
        drop_out=False,
        mp_times=2,
        MultiHead=1,
        dual_edge=False,
        nn_act="SiLU",
    ):

        super(GnBlock, self).__init__()

        self.dual_edge = dual_edge
        eb_input_dim = 3 * hidden_size
        cb_input_dim = int(hidden_size + 0.5 * hidden_size)
        cb_custom_func = build_mlp(
            cb_input_dim, hidden_size, hidden_size, drop_out=False, nn_act=nn_act
        )
        eb_custom_func = build_mlp(
            eb_input_dim, hidden_size, hidden_size, drop_out=False, nn_act=nn_act
        )
        self.eb_module = EdgeBlock(custom_func=eb_custom_func, dual_edge=dual_edge)
        self.cb_module = CellBlock(
            hidden_size,
            hidden_size,
            mp_times=mp_times,
            MultiHead=MultiHead,
            custom_func=cb_custom_func,
            dual_edge=dual_edge,
        )

    def forward(self, graph, graph_node):

        graph_last = copy_geometric_data(graph, has_changed_node_attr_to_cell_attr=True)

        graph = self.cb_module(graph, graph_node)

        graph = self.eb_module(graph)

        # resdiual connection
        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x

        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)


class Decoder(nn.Module):

    def __init__(
        self,
        edge_hidden_size=128,
        cell_hidden_size=128,
        edge_output_size=7,
        cell_output_size=2,
        cell_input_size=2,
        dual_edge=False,
        nn_act="SiLU",
    ):
        super(Decoder, self).__init__()
        self.dual_edge = dual_edge
        if dual_edge:
            self.edge_decode_module = build_mlp(
                edge_hidden_size * 2,
                edge_hidden_size,
                edge_output_size,
                drop_out=False,
                lay_norm=False,
                nn_act=nn_act,
            )
        else:
            self.edge_decode_module = build_mlp(
                edge_hidden_size,
                edge_hidden_size,
                edge_output_size,
                drop_out=False,
                lay_norm=False,
                nn_act=nn_act,
            )
        # self.cell_decode_module = build_mlp(cell_hidden_size, cell_hidden_size, cell_output_size, drop_out=False,lay_norm=False)

    def forward(self, graph=None, features=None, cell_features=None):

        if self.dual_edge:
            edge_decoded_attr = self.edge_decode_module(
                torch.cat((torch.chunk(graph.edge_attr, 2, 0)), dim=-1)
            )
        else:
            edge_decoded_attr = self.edge_decode_module(graph.edge_attr)
        return (False, edge_decoded_attr)


class EncoderProcesserDecoder(nn.Module):

    def __init__(
        self,
        message_passing_num,
        cell_input_size,
        edge_input_size,
        cell_output_size,
        edge_output_size,
        drop_out,
        hidden_size=128,
        mp_times=2,
        MultiHead=1,
        dual_edge=False,
        nn_act="SiLU",
    ):

        super(EncoderProcesserDecoder, self).__init__()

        self.message_passing_num = message_passing_num
        self.dual_edge = dual_edge

        self.encoder = Encoder(
            edge_input_size=edge_input_size,
            cell_input_size=cell_input_size,
            hidden_size=hidden_size,
            nn_act=nn_act,
        )

        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(
                GnBlock(
                    hidden_size=hidden_size,
                    drop_out=drop_out,
                    mp_times=mp_times,
                    MultiHead=MultiHead,
                    dual_edge=dual_edge,
                    nn_act=nn_act,
                )
            )
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder = Decoder(
            edge_hidden_size=hidden_size,
            cell_hidden_size=hidden_size,
            cell_output_size=cell_output_size,
            edge_output_size=edge_output_size,
            dual_edge=dual_edge,
            nn_act=nn_act,
        )

    def forward(self, graph, graph_node):
        # input graph has cell_attr as x, edge_attr as edge_attr, edge_neighbour_cell_index as edge_index
        graph = self.encoder(graph)
        graph_last = copy_geometric_data(graph, has_changed_node_attr_to_cell_attr=True)
        count = self.message_passing_num
        for model in self.processer_list:
            graph = model(graph, graph_node)
            """add skip connection"""
            if count == int(self.message_passing_num / 2):
                graph.x = graph.x + graph_last.x
                graph.edge_attr = graph.edge_attr + graph_last.edge_attr
            count -= 1
        cell_decoded_attr, edge_decoded_attr = self.decoder(graph=graph)

        return (cell_decoded_attr, edge_decoded_attr)


class Intergrator(nn.Module):
    def __init__(
        self, edge_input_size=7, cell_input_size=2, cell_output_size=2, loss_cont=0
    ):
        super(Intergrator, self).__init__()
        self.edge_input_size = edge_input_size
        self.cell_input_size = cell_input_size
        self.cell_output_size = cell_output_size
        self.loss_cont = loss_cont

    def forward(
        self,
        uv_face=0,
        p_face=0,
        flux_D=0,
        unv=None,
        rho=None,
        rhs_coef=None,
        face_area=None,
        cell_face=None,
        device=None,
    ):

        # prepare face neighbour cell`s index
        e_ED_0 = torch.index_select(face_area, 0, cell_face[0])
        e_ED_1 = torch.index_select(face_area, 0, cell_face[1])
        e_ED_2 = torch.index_select(face_area, 0, cell_face[2])

        """conserved form convection term"""
        uu_vu_face = torch.cat(
            (uv_face[:, 0:1] * uv_face, uv_face[:, 1:2] * uv_face), dim=-1
        )

        # Advection_on_cells_edge_0 = torch.index_select(flux_A,0,cell_face[0])
        # Advection_on_cells_edge_1 = torch.index_select(flux_A,0,cell_face[1])
        # Advection_on_cells_edge_2 = torch.index_select(flux_A,0,cell_face[2])
        # ever cell has 3 edges in trianglaur mesh

        """ intergrate to form continutiy equation"""
        if self.loss_cont > 0:
            loss_continuity = (
                self.chain_dot_product(
                    torch.index_select(uv_face, 0, cell_face[0]), unv[:, 0, :]
                )
                * e_ED_0
                + self.chain_dot_product(
                    torch.index_select(uv_face, 0, cell_face[1]), unv[:, 1, :]
                )
                * e_ED_1
                + self.chain_dot_product(
                    torch.index_select(uv_face, 0, cell_face[2]), unv[:, 2, :]
                )
                * e_ED_2
            )
        else:
            loss_continuity = 0

        """ intergrate to form convection term of momentenm equation"""
        integrate_flux_A_threeedges = (
            self.chain_flux_dot_product(uu_vu_face[cell_face[0]], unv[:, 0, :]) * e_ED_0
            + self.chain_flux_dot_product(uu_vu_face[cell_face[1]], unv[:, 1, :])
            * e_ED_1
            + self.chain_flux_dot_product(uu_vu_face[cell_face[2]], unv[:, 2, :])
            * e_ED_2
        )

        """ intergrate to form diffusion/visicous term of momentenm equation"""
        integrate_flux_D_threeedges = (
            torch.index_select(flux_D, 0, cell_face[0])
            + torch.index_select(flux_D, 0, cell_face[1])
            + torch.index_select(flux_D, 0, cell_face[2])
        )

        """ intergrate to form pressure term of momentenm equation"""
        integrate_flux_P_threeedges = (
            torch.index_select(p_face, 0, cell_face[0]) * unv[:, 0, :] * e_ED_0
            + torch.index_select(p_face, 0, cell_face[1]) * unv[:, 1, :] * e_ED_1
            + torch.index_select(p_face, 0, cell_face[2]) * unv[:, 2, :] * e_ED_2
        )

        return (
            loss_continuity,
            rhs_coef
            * (
                (
                    -integrate_flux_A_threeedges
                    - (1.0 / rho) * (integrate_flux_P_threeedges)
                )
            )
            + integrate_flux_D_threeedges,
        )

    # a and b has to be the same size
    def chain_dot_product(self, a, b, keepdim=True):

        return torch.sum(a * b, dim=-1, keepdim=keepdim)

    # a and b has to be the different size
    def chain_flux_dot_product(self, a, b):
        """
        Compute the dot product in a chained manner for tensor 'a' with varying even dimensions
        and tensor 'b' with 2 dimensions.

        Parameters:
        a (torch.Tensor): A tensor with even number of columns.
        b (torch.Tensor): Another tensor with exactly 2 columns.

        Returns:
        torch.Tensor: The concatenated result of multiple dot products between slices of 'a' and 'b'.

        Raises:
        ValueError: If the input tensors do not meet the dimension requirements.
        """

        # Verify that 'a' has an even number of columns and 'b' has exactly 2 columns
        if a.dim() < 2 or a.size(1) % 2 != 0:
            raise ValueError("Tensor 'a' must have an even number of columns")
        if b.dim() < 2 or b.size(1) != 2:
            raise ValueError("Tensor 'b' must have exactly 2 columns")

        try:
            # Perform the chained dot product
            results = []
            for i in range(0, a.size(1), 2):
                dot_product = self.chain_dot_product(a[:, i : i + 2], b)
                results.append(dot_product)
            return torch.cat(results, dim=-1)
        except RuntimeError as e:
            # Handle potential errors in torch operations
            raise RuntimeError(f"Error in performing chained dot product: {e}")
