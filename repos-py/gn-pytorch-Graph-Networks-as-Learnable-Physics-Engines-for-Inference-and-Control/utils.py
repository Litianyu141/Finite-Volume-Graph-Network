import torch
import numpy as np

def load_graph_features(G, action, state, delta_state, bs = 1, norm = True, noise = 0.003, std = None):

    pos = state[:, 5:5+18].view(-1,6,3)
    if noise > 0:
        pos_noise = torch.randn(pos.size()).cuda() * noise * std[:, :3]
    else:
        pos_noise = 0


    pos += pos_noise
    if noise > 0:
        delta_state[:, 5:5+18] -= pos_noise.view(-1, 18)

    joints = pos[:,1:,-1] - pos[:,:-1,-1]
    joints[joints > np.pi] -= np.pi * 2
    joints[joints < -np.pi] += np.pi * 2
    
    if norm:
        center_pos = torch.mean(pos[:,:,:2], dim = 1, keepdim = True)
        pos[:,:,:2] -= center_pos

    vel = state[:, 5+18:5+36].view(-1,6,3)

    if noise > 0:
        vel_noise = torch.randn(vel.size()).cuda() * noise * std[:, 3:]
    else:
        vel_noise = 0
    vel += vel_noise

    if noise > 0:
        delta_state[:, 5+18:5+36] -= vel_noise.view(-1, 18)    

    for node in G.nodes():
        #print(node)
        G.nodes[node]['feat'][:,:3] = pos[:,node]
        G.nodes[node]['feat'][:, 3:] = vel[:, node]

    for edge in G.edges():
        if edge[0] < edge[1]:
            G[edge[0]][edge[1]]['feat'][:,0] = -1
        else:
            G[edge[0]][edge[1]]['feat'][:, 0] = 1

        m = min(edge)
        G[edge[0]][edge[1]]['feat'][:, 1] = joints[:,m]
        G[edge[0]][edge[1]]['feat'][:, 2] = action[:,m]
    return G


def build_graph_loss(G, state):
    loss = 0
    n_nodes = len(G)

    pos = state[:, 5:5 + 18].view(-1, 6, 3)
    pos[:,:,2] -=  (pos[:,:,2] > np.pi).float() * np.pi * 2
    pos[:, :, 2] += (pos[:, :, 2] < -np.pi).float() * np.pi * 2

    vel = state[:, 5 + 18:5 + 36].view(-1, 6, 3)

    for node in G.nodes():
        loss += torch.mean((G.nodes[node]['feat'][:,:3] - pos[:,node]) ** 2)
        loss += torch.mean((G.nodes[node]['feat'][:, 3:] - vel[:, node]) ** 2)

    loss /= n_nodes
    return loss

def build_graph_loss2(G, H):
    loss = 0
    n_nodes = len(G)
    for node in G.nodes():
        loss += torch.mean((G.nodes[node]['feat'][:,:3] - H.nodes[node]['feat'][:,:3]) ** 2)
        loss += torch.mean((G.nodes[node]['feat'][:, 3:] - H.nodes[node]['feat'][:,3:]) ** 2)

    loss /= n_nodes
    return loss

def init_graph_features(G, graph_feat_size, node_feat_size, edge_feat_size, bs=1, cuda=False):
    if cuda:
        G.graph['feat'] = torch.zeros(bs, graph_feat_size).cuda()
        for node in G.nodes():
            G.nodes[node]['feat'] = torch.zeros(bs, node_feat_size).cuda()
        for edge in G.edges():
            G[edge[0]][edge[1]]['feat'] = torch.zeros(bs, edge_feat_size).cuda()
    else:
        G.graph['feat'] = torch.zeros(bs, graph_feat_size)
        for node in G.nodes():
            G.nodes[node]['feat'] = torch.zeros(bs, node_feat_size)
        for edge in G.edges():
            G[edge[0]][edge[1]]['feat'] = torch.zeros(bs, edge_feat_size)


def detach(G):
    G.graph['feat'] = G.graph['feat'].detach()
    for node in G.nodes():
        G.nodes[node]['feat'] = G.nodes[node]['feat'].detach()
    for edge in G.edges():
        G[edge[0]][edge[1]]['feat'] = G[edge[0]][edge[1]]['feat'].detach()
    return G