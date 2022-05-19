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

def get_graph_features(G, bs = 1):
    state = torch.zeros((bs, 41)).cuda()
    
    #joints = state[:,:5]
    pos = torch.zeros((bs, 6, 3)).cuda()
    vel = torch.zeros((bs, 6, 3)).cuda()
    
    # only get node features
    for node in G.nodes():
        #print(node)
        pos[:,node] = G.nodes[node]['feat'][:,:3]
        vel[:, node] = G.nodes[node]['feat'][:, 3:]

        
    state[:, 5:5+18] = pos.view(-1, 18)
    state[:, 5+18:5+36] = pos.view(-1,18)
    return state

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
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def draw_state(state):
    state = state.cpu().data.numpy()[0]
    positions = state[5:5+18].reshape(6,3)
 
    fig = plt.figure()

    for node in range(6):
        pos = positions[node]
        angle = pos[2]
        x = pos[0]
        y = pos[1]
        r = 0.05
        dy = np.cos(angle) * r
        dx = - np.sin(angle) * r
        plt.plot([x - dx, x + dx], [y - dy, y + dy], 'g', alpha = 0.5)

        plt.axis('equal')
        
    
    img = fig2img(fig)
    plt.close()    
    return img


if __name__ == '__main__':
	model_fn = '/home/fei/Development/physics_predmodel/gn/logs/runs/October01_14:59:16/model_1240000.pth'
	dset = SwimmerDataset('swimmer_test.npy')
	use_cuda = True
	dl = DataLoader(dset, batch_size=200, num_workers=0, drop_last=True)
	node_feat_size = 6
	edge_feat_size = 3
	graph_feat_size = 10
	gn = FFGN(graph_feat_size, node_feat_size, edge_feat_size).cuda()
	gn.load_state_dict(torch.load(model_fn))

	normalizers = torch.load('normalize.pth')
	in_normalizer = normalizers['in_normalizer']
	out_normalizer = normalizers['out_normalizer']

	G1 = nx.path_graph(6).to_directed()
	dl_e = enumerate(dl)
	data = dset.__get_episode__(353)
	data = [torch.from_numpy(item) for item in data]

	writer = imageio.get_writer('test_pred.mp4', fps=6)
	action, delta_state, last_state = data
	action, delta_state, last_state = action.float(), delta_state.float(), last_state.float()

	if use_cuda:
	    action, delta_state, last_state = action.cuda(), delta_state.cuda(), last_state.cuda()
	    
	state = last_state[1].unsqueeze(0)
	state_gt = last_state[1].unsqueeze(0).clone()

	for i in range(1, 50):
	    print(i)
	    action_i = action[i].unsqueeze(0)
	    delta_state_i = delta_state[i].unsqueeze(0)
	    last_state_i = last_state[i].unsqueeze(0)
	    
	    init_graph_features(G1, graph_feat_size, node_feat_size, edge_feat_size, cuda=True, bs = 1)
	    load_graph_features(G1, action_i, state, None, bs=1, noise = 0)
	    G_out = gn(in_normalizer.normalize(G1))
	    G_out = out_normalizer.inormalize(G_out)
	    
	    delta_state_pred = get_graph_features(G_out)
	    
	    state_gt += delta_state_i
	    state += delta_state_pred

	    img = draw_state(state_gt)
	    img_pred = draw_state(state)
	    
	    writer.append_data(np.concatenate([img, img_pred], axis = 1))

	writer.close()