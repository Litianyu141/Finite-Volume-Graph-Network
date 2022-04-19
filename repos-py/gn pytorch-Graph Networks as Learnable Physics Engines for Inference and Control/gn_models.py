import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from utils import *


_node_feat_size = 128
_edge_feat_size = 128
_graph_feat_size = 128


class EdgeBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(EdgeBlock, self).__init__()
        self.f_e = nn.Sequential(
            nn.Linear(graph_feat_size + 2 * node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256,256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,edge_feat_size),
        )
    def forward(self, g, ns, nr, e):
        x = torch.cat([g, ns, nr, e], dim = -1)
        return self.f_e(x)
    
class NodeBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(NodeBlock, self).__init__()
        self.f_n = nn.Sequential(
            nn.Linear(graph_feat_size + node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, node_feat_size),
        )
    def forward(self, g, n, e):
        x = torch.cat([g, n, e], dim = -1)
        return self.f_n(x)

    
class GraphBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(GraphBlock, self).__init__()
        self.f_g = nn.Sequential(
            nn.Linear(graph_feat_size + node_feat_size + edge_feat_size, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, graph_feat_size),
        )
    def forward(self, g, n, e):
        x = torch.cat([g, n, e], dim = -1)
        return self.f_g(x)
    
    
class GNBlock(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(GNBlock, self).__init__()
        self.edge_block = EdgeBlock(graph_feat_size, node_feat_size, edge_feat_size)
        self.node_block = NodeBlock(graph_feat_size, node_feat_size, edge_feat_size)
        self.graph_block = GraphBlock(graph_feat_size, node_feat_size, edge_feat_size)
        
        self.graph_feat_size = graph_feat_size
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        
        
    def forward(self, x):
        bs = x.graph['feat'].size(0)
        #edge update
        for u,v in x.edges():
            g = x.graph['feat']
            ns = x.node[u]['feat']
            nr = x.node[v]['feat']
            e = x[u][v]['feat']
            x[u][v]['temp_feat'] = self.edge_block(g, ns, nr, e)
            
        for u,v in x.edges():
            x[u][v]['feat'] = x[u][v]['temp_feat']
        
        #node update
        for u in x.nodes():
            g = x.graph['feat']
            n = x.node[u]['feat']
            pred = list(x.predecessors(u))
            n_e_agg = torch.zeros(bs, self.edge_feat_size)
            if x.graph['feat'].is_cuda:
                n_e_agg = n_e_agg.cuda()
            for v in pred:
                n_e_agg += x[v][u]['feat']
            x.node[u]['temp_feat'] = self.node_block(g, n, n_e_agg)
        
        for u in x.nodes():
            x.node[u]['feat'] = x.node[u]['temp_feat']
         
        #graph update
        e_agg = torch.zeros(bs, self.edge_feat_size)
        n_agg = torch.zeros(bs, self.node_feat_size)
        if x.graph['feat'].is_cuda:
            e_agg = e_agg.cuda()
            n_agg = n_agg.cuda()

        for u,v in x.edges():
            e_agg += x[u][v]['feat']
        for u in x.nodes():
            n_agg += x.node[u]['feat']
        g = x.graph['feat']
        x.graph['feat'] = self.graph_block(g, n_agg, e_agg)
        
        return x

def subtract(G, H):
    G_out = G.copy()
    G_out.graph['feat'] -= H.graph['feat']
    for node in G_out.nodes():
        G_out.nodes[node]['feat'] -= H.nodes[node]['feat']
    for edge in G.edges():
        G_out[edge[0]][edge[1]]['feat'] -= H[edge[0]][edge[1]]['feat']

    return G_out

class Normalizer:
    def __init__(self):
        self.count = 0
        self.momentum = 0.99
        self.G = None
    def input(self, G):
        if self.count == 0:
            self.G = G.copy()

            del self.G.graph['feat']
            for node in self.G.nodes():
                del self.G.nodes[node]['feat']
            for edge in self.G.edges():
                del self.G[edge[0]][edge[1]]['feat']

            self.G.graph['feat_mean'] = torch.mean(G.graph['feat'], dim=0, keepdim=True)
            self.G.graph['feat_var'] = torch.var(G.graph['feat'], dim=0, keepdim=True)

            for node in G.nodes():
                self.G.nodes[node]['feat_mean'] = torch.mean(G.nodes[node]['feat'], dim=0, keepdim=True)
                self.G.nodes[node]['feat_var'] = torch.var(G.nodes[node]['feat'], dim=0, keepdim=True)

            for edge in G.edges():
                self.G[edge[0]][edge[1]]['feat_mean'] = torch.mean(G[edge[0]][edge[1]]['feat'], dim=0, keepdim=True)
                self.G[edge[0]][edge[1]]['feat_var'] = torch.var(G[edge[0]][edge[1]]['feat'], dim=0, keepdim=True)
        else:
            self.G.graph['feat_mean'] = self.momentum * self.G.graph['feat_mean'] + (1-self.momentum) * torch.mean(G.graph['feat'], dim=0, keepdim=True)
            self.G.graph['feat_var'] = self.momentum * self.G.graph['feat_var'] + (1-self.momentum) * torch.var(G.graph['feat'], dim=0, keepdim=True)

            for node in G.nodes():
                self.G.nodes[node]['feat_mean'] = self.momentum * self.G.nodes[node]['feat_mean'] + (1-self.momentum) * torch.mean(G.nodes[node]['feat'], dim=0, keepdim=True)
                self.G.nodes[node]['feat_var'] = self.momentum * self.G.nodes[node]['feat_var'] + (1-self.momentum) * torch.var(G.nodes[node]['feat'], dim=0, keepdim=True)

            for edge in G.edges():
                self.G[edge[0]][edge[1]]['feat_mean'] =  self.momentum * self.G[edge[0]][edge[1]]['feat_mean'] + (1-self.momentum) * torch.mean(G[edge[0]][edge[1]]['feat'], dim=0, keepdim=True)
                self.G[edge[0]][edge[1]]['feat_var'] = self.momentum *  self.G[edge[0]][edge[1]]['feat_var'] + (1-self.momentum) * torch.var(G[edge[0]][edge[1]]['feat'], dim=0, keepdim=True)

        #print(self.G.nodes[0]['feat_var'])
        self.count += 1
        ## accumulate mean and var

    def get(self):
        return self.G

    def normalize(self, H):
        G_out = H.copy()

        G_out.graph['feat'] = (G_out.graph['feat'] - self.G.graph['feat_mean']) / (torch.sqrt(self.G.graph['feat_var']) + 1e-6).detach()
        for node in G_out.nodes():
            G_out.nodes[node]['feat'] = (G_out.nodes[node]['feat'] - self.G.nodes[node]['feat_mean']) / (torch.sqrt(self.G.nodes[node]['feat_var']) + 1e-6).detach()
        for edge in G_out.edges():
            G_out[edge[0]][edge[1]]['feat'] = (G_out[edge[0]][edge[1]]['feat'] - self.G[edge[0]][edge[1]]['feat_mean']) / (torch.sqrt(self.G[edge[0]][edge[1]]['feat_var']) + 1e-6).detach()

        #print(G_out.nodes[0]['feat'])
        return G_out

    def inormalize(self, H):
        G_out = H.copy()
        for node in G_out.nodes():
            G_out.nodes[node]['feat'] = G_out.nodes[node]['feat']  *  (torch.sqrt(self.G.nodes[node]['feat_var']) + 1e-6).detach() +  self.G.nodes[node]['feat_mean']

        return G_out

    def get_std(self):
        std = []
        for node in self.G.nodes():
            std.append(self.G.nodes[node]['feat_var'])

        std = torch.cat(std, 0)
        std = torch.sqrt(std + 1e-6)
        #print(std)
        return std


class FFGN(nn.Module):
    def __init__(self, graph_feat_size, node_feat_size, edge_feat_size):
        super(FFGN, self).__init__()
        self.GN1 = GNBlock(graph_feat_size, node_feat_size, edge_feat_size)
        self.GN2 = GNBlock(graph_feat_size*2, node_feat_size*2, edge_feat_size*2)
        self.linear = nn.Linear(node_feat_size*2, node_feat_size)

    def forward(self, G_in):
        G = G_in.copy()
        G = self.GN1(G)
        #Graph concatenate
        G.graph['feat'] = torch.cat([G.graph['feat'], G_in.graph['feat']], dim=-1)
        for node in G.nodes():
            G.nodes[node]['feat'] = torch.cat([G.nodes[node]['feat'], G_in.nodes[node]['feat']], dim = -1)
        for edge in G.edges():
                G[edge[0]][edge[1]]['feat'] = torch.cat([G[edge[0]][edge[1]]['feat'], G_in[edge[0]][edge[1]]['feat']],
                                                         dim = -1)
        G = self.GN2(G)

        for node in G.nodes():
            G.nodes[node]['feat'] = self.linear(G.nodes[node]['feat'])
        #use a linear layer to change back to original node feature size
        return G

if __name__ == "__main__":
    G1 = nx.erdos_renyi_graph(10,0.3).to_directed()
    #nx.draw(G1)
    #plt.show()
    init_graph_features(G1, _graph_feat_size, _node_feat_size, _edge_feat_size, cuda = True)
    gn = FFGN(_graph_feat_size, _node_feat_size, _edge_feat_size).cuda()
    G_out = gn(G1)
    torch.sum(G_out.graph['feat'] ** 2).backward()
