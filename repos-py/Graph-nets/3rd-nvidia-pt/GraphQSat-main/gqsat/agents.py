# Copyright 2019-2020 Nvidia Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX


class Agent(object):
    def act(self, state):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class MiniSATAgent(Agent):
    """Use MiniSAT agent to solve the problem"""

    def act(self, observation):
        return -1  # this will make GymSolver use VSIDS to make a decision

    def __str__(self):
        return "<MiniSAT Agent>"


class RandomAgent(Agent):
    """Uniformly sample the action space"""

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

    def __str__(self):
        return "<Random Agent>"


class GraphAgent:
    def __init__(self, net, args):

        self.net = net
        self.device = args.device
        self.debug = args.debug
        self.qs_buffer = []

    def forward(self, hist_buffer):
        self.net.eval()
        with torch.no_grad():
            vdata, edata, conn, udata = hist_buffer[0]
            vdata = torch.tensor(vdata, device=self.device)
            edata = torch.tensor(edata, device=self.device)
            udata = torch.tensor(udata, device=self.device)
            conn = torch.tensor(conn, device=self.device)
            vout, eout, _ = self.net(x=vdata, edge_index=conn, edge_attr=edata, u=udata)
            res = vout[vdata[:, VAR_ID_IDX] == 1]

            if self.debug:
                self.qs_buffer.append(res.flatten().cpu().numpy())
            return res

    def act(self, hist_buffer, eps):
        if np.random.random() < eps:
            vars_to_decide = np.where(hist_buffer[-1][0][:, VAR_ID_IDX] == 1)[0]
            acts = [a for v in vars_to_decide for a in (v * 2, v * 2 + 1)]
            return int(np.random.choice(acts))
        else:
            qs = self.forward(hist_buffer)
            return self.choose_actions(qs)

    def choose_actions(self, qs):
        return qs.flatten().argmax().item()
