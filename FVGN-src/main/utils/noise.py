import torch
from utils.utilities import NodeType
import random


def prob_return(p):
    # p is a probability between 0 and 1
    # returns True with probability p and False with probability 1-p
    assert 0 <= p <= 1, "p must be a valid probability"
    return random.random() < p


def get_v_noise_on_cell(graph, noise_std, outchannel, device):

    velocity_sequence = graph.x[:, 1:3]
    # type = graph.x[:, 0]
    noise_u = torch.normal(
        std=noise_std, mean=0.0, size=(velocity_sequence.shape[0], 1)
    ).to(device)
    noise_v = torch.normal(
        std=noise_std, mean=0.0, size=(velocity_sequence.shape[0], 1)
    ).to(device)
    noise = torch.cat((noise_u, noise_v), dim=1)
    # mask = type!=NodeType.NORMAL
    # noise[mask]=0
    return noise


def get_noise_on_edge(graph, noise_std, outchannel, device):
    velocity_sequence = graph.x.repeat(1, 2)
    type = graph.x[:, 0]
    noise_u = torch.normal(
        std=noise_std, mean=0.0, size=(velocity_sequence.shape[0], 1)
    ).to(device)
    noise_v = torch.normal(
        std=noise_std, mean=0.0, size=(velocity_sequence.shape[0], 1)
    ).to(device)
    noise = torch.cat((noise_u, noise_v), dim=1)
    mask = type != NodeType.NORMAL
    noise[mask] = 0
    return noise.to(device)


def get_v_noise_on_node(graph, noise_std, outchannel, device):
    if noise_std == 0:
        return torch.zeros_like(graph.x[:, 1:3], device=device)
    else:
        velocity_sequence = graph.x[:, 1:3]
        type = graph.x[:, 0]
        noise_v = torch.normal(
            std=noise_std, mean=0.0, size=(velocity_sequence.shape[0], outchannel)
        ).to(device)
        noise = noise_v
        mask = type != NodeType.NORMAL
        noise[mask] = 0
        return noise.to(device)
