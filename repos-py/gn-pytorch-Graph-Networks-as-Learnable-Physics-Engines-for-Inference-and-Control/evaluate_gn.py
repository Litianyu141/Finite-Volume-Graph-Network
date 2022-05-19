import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
import torch.optim as optim
import matplotlib.pyplot as plt
from gn_models import init_graph_features, FFGN
import torch
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import sys
from scipy.stats import pearsonr
from train_gn import SwimmerDataset
from utils import *
import argparse


def evaluate_graph_loss(G, state, last_state):
    n_nodes = len(G)

    dpos = state[:, 5:5 + 18].view(-1, 6, 3)
    last_pos = last_state[:, 5:5 + 18].view(-1, 6, 3)
    vel = state[:, 5 + 18:5 + 36].view(-1, 6, 3)
    mean = 0

    true = []
    pred = []

    for node in G.nodes():
        #print(node)
        #loss += torch.mean((G.nodes[node]['feat'][:,:3] - pos[:,node]) ** 2)
        #loss += torch.mean((G.nodes[node]['feat'][:, 3:] - vel[:, node]) ** 2)
        mean += torch.mean(torch.abs((G.nodes[node]['feat'][:,:3] - dpos[:,node]) / dpos[:,node] ))
        pred.append(G.nodes[node]['feat'][:,:3])
        true.append(dpos[:,node])

    pred = torch.stack(pred).view(-1,1)
    true = torch.stack(true).view(-1,1)

    plt.figure()
    for node in G.nodes():
        pos = last_pos[0, node, :3].cpu().data.numpy()

        angle = pos[2]
        x = pos[0]
        y = pos[1]
        r = 0.05
        dy = np.cos(angle) * r
        dx = - np.sin(angle) * r
        # plt.figure()
        plt.plot([x - dx, x + dx], [y - dy, y + dy], 'g', alpha = 0.5)

        pos = G.nodes[node]['feat'][0,:3].cpu().data.numpy() + last_pos[0,node,:3].cpu().data.numpy()

        angle = pos[2]
        x = pos[0]
        y = pos[1]
        r = 0.05
        dy = np.cos(angle) * r
        dx = - np.sin(angle) * r
        # plt.figure()
        plt.plot([x - dx, x + dx], [y - dy, y + dy],'r', alpha = 0.5)

        pos = dpos[0,node].cpu().data.numpy() + last_pos[0, node, :3].cpu().data.numpy()

        angle = pos[2]
        x = pos[0]
        y = pos[1]
        r = 0.05
        dy = np.cos(angle) * r
        dx = - np.sin(angle) * r
        # plt.figure()
        plt.plot([x - dx, x + dx], [y - dy, y + dy],'b', alpha = 0.5)
    plt.axis('equal')
    plt.show()

    mean /= n_nodes

    return mean.data.item(), true, pred

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default = '',  help='model path')
    opt = parser.parse_args()
    print(opt)

    dset = SwimmerDataset('swimmer_test.npy')
    use_cuda = True
    dl = DataLoader(dset, batch_size=200, num_workers=0, drop_last=True)
    G1 = nx.path_graph(6).to_directed()
    #nx.draw(G1)
    #plt.show()
    node_feat_size = 6
    edge_feat_size = 3
    graph_feat_size = 10
    gn = FFGN(graph_feat_size, node_feat_size, edge_feat_size).cuda()
    gn.load_state_dict(torch.load(opt.model))

    normalizers = torch.load('normalize.pth')
    in_normalizer = normalizers['in_normalizer']
    out_normalizer = normalizers['out_normalizer']
    std = in_normalizer.get_std()
    step = 0
    for i,data in enumerate(dl):
        action, delta_state, last_state = data
        action, delta_state, last_state = action.float(), delta_state.float(), last_state.float()
        if use_cuda:
            action, delta_state, last_state = action.cuda(), delta_state.cuda(), last_state.cuda()

        init_graph_features(G1, graph_feat_size, node_feat_size, edge_feat_size, cuda=True, bs = 200)
        load_graph_features(G1, action, last_state, delta_state, bs=200, noise = 0.03, std = std)
        G_out = gn(in_normalizer.normalize(G1))
        G_out = out_normalizer.inormalize(G_out)
        loss, true, pred = evaluate_graph_loss(G_out, delta_state, last_state)
        true = true.data.cpu().numpy()
        pred = pred.data.cpu().numpy()
        plt.scatter(true, pred, s = 2, alpha = 0.7)
        plt.show()

        r = pearsonr(true, pred)[0][0]
        print(loss, r)
        step += 1
        if i > 50:
            break
