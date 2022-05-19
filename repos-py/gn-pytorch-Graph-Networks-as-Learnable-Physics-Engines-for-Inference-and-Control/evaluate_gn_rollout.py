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
from PIL import Image
import imageio
from utils import *
import argparse


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    plt.close()
    return np.array(Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) ) )

def draw_snake(state):

    fig = plt.figure()
    for i in range(6):
        pos = state[i, :3]

        angle = pos[2]
        x = pos[0]
        y = pos[1]
        r = 0.05
        dy = np.cos(angle) * r
        dx = - np.sin(angle) * r
        # plt.figure()
        plt.plot([x - dx, x + dx], [y - dy, y + dy], 'g', alpha = 0.5)
        plt.axis('equal')

    return fig



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default = '',  help='model path')
    opt = parser.parse_args()
    print(opt)

    dset = SwimmerDataset('swimmer_test.npy')
    use_cuda = True
    dl = DataLoader(dset, batch_size=200, num_workers=0, drop_last=True)

    #nx.draw(G1)
    #plt.show()
    node_feat_size = 6
    edge_feat_size = 3
    graph_feat_size = 10
    gn = FFGN(graph_feat_size, node_feat_size, edge_feat_size).cuda()
    gn.load_state_dict(torch.load(opt.model))
    action, state = dset.get_episode(10)

    position = state[:, 5:5 + 18].reshape(-1, 6, 3)
    normalizers = torch.load('normalize.pth')
    in_normalizer = normalizers['in_normalizer']
    out_normalizer = normalizers['out_normalizer']


    """
    writer = imageio.get_writer('test_plt.mp4', fps=30)
    for frame in range(100):
        fig = draw_snake(position[frame])
        img = fig2img(fig)
        writer.append_data(img)
        print(frame)
    writer.close()
    """

    start = 10
    state_tensor = torch.from_numpy(state[start, :].astype(np.float32)).unsqueeze(0).cuda()
    writer = imageio.get_writer('test_pred.mp4', fps=10)

    for frame in range(start + 1,100):
        action_tensor = torch.from_numpy(action[frame, :].astype(np.float32)).unsqueeze(0).cuda()
        #action_tensor.fill_(0)
        #print(state_tensor.size(), action_tensor.size())
        G1 = nx.path_graph(6).to_directed()
        init_graph_features(G1, graph_feat_size, node_feat_size, edge_feat_size, cuda=True, bs=1)
        load_graph_features(G1, action_tensor, state_tensor, bs=1)
        G_out = gn(in_normalizer.normalize(G1))
        G_out = out_normalizer.inormalize(G_out)
        delta_tensor = torch.zeros(state_tensor.size()).cuda()

        for i in range(6):
            delta_tensor[0, 5 + 6 * i:11 + 6 * i] = G_out.nodes[i]['feat']

        state_tensor += delta_tensor
        true_state_tensor = torch.from_numpy(state[frame, :].astype(np.float32)).unsqueeze(0).cuda()
        #state_tensor[0,:5] = true_state_tensor[0,:5]

        if frame % 2 == 0:
            state_tensor = true_state_tensor

        s = state_tensor.cpu().data.numpy()
        position = s[0, 5:5 + 18].reshape(6, 3)

        fig = draw_snake(position)
        img = fig2img(fig)
        writer.append_data(img)

    writer.close()

    """

    step = 0
    for i,data in enumerate(dl):
        action, delta_state, last_state = data
        action, delta_state, last_state = action.float(), delta_state.float(), last_state.float()
        if use_cuda:
            action, delta_state, last_state = action.cuda(), delta_state.cuda(), last_state.cuda()

        init_graph_features(G1, graph_feat_size, node_feat_size, edge_feat_size, cuda=True, bs = 200)
        load_graph_features(G1, action, last_state,bs=200)
        G_out = gn(G1)
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
    """