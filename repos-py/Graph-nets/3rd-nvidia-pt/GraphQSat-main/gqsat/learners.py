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

from torch import nn
from torch_scatter import scatter_max
import torch
from torch.optim.lr_scheduler import StepLR
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX


class GraphLearner:
    def __init__(self, net, target, buffer, args):
        self.net = net
        self.target = target
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.lr_scheduler = StepLR(
            self.optimizer, args.lr_scheduler_frequency, args.lr_scheduler_gamma
        )

        if args.loss == "mse":
            self.loss = nn.MSELoss()
        elif args.loss == "huber":
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError("Unknown Loss function.")

        self.bsize = args.bsize
        self.gamma = args.gamma
        self.buffer = buffer
        self.target_update_freq = args.target_update_freq
        self.step_ctr = 0
        self.grad_clip = args.grad_clip
        self.grad_clip_norm_type = args.grad_clip_norm_type
        self.device = args.device

    def get_qs(self, states):
        vout, eout, _ = self.net(
            x=states[0],
            edge_index=states[2],
            edge_attr=states[1],
            v_indices=states[4],
            e_indices=states[5],
            u=states[6],
        )
        return vout[states[0][:, VAR_ID_IDX] == 1], states[3]

    def get_target_qs(self, states):
        vout, eout, _ = self.target(
            x=states[0],
            edge_index=states[2],
            edge_attr=states[1],
            v_indices=states[4],
            e_indices=states[5],
            u=states[6],
        )
        return vout[states[0][:, VAR_ID_IDX] == 1].detach(), states[3]

    def step(self):
        s, a, r, s_next, nonterminals = self.buffer.sample(self.bsize)
        # calculate the targets first to optimize the GPU memory

        with torch.no_grad():
            target_qs, target_vertex_sizes = self.get_target_qs(s_next)
            idx_for_scatter = [
                [i] * el.item() * 2 for i, el in enumerate(target_vertex_sizes)
            ]
            idx_for_scatter = torch.tensor(
                [el for subl in idx_for_scatter for el in subl],
                dtype=torch.long,
                device=self.device,
            ).flatten()
            target_qs = scatter_max(target_qs.flatten(), idx_for_scatter, dim=0)[0]
            targets = r + nonterminals * self.gamma * target_qs

        self.net.train()
        qs, var_vertex_sizes = self.get_qs(s)
        # qs.shape[1] values per node (same num of actions per node)
        gather_idx = (var_vertex_sizes * qs.shape[1]).cumsum(0).roll(1)
        gather_idx[0] = 0

        qs = qs.flatten()[gather_idx + a]

        loss = self.loss(qs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.grad_clip, norm_type=self.grad_clip_norm_type
        )
        self.optimizer.step()

        if not self.step_ctr % self.target_update_freq:
            self.target.load_state_dict(self.net.state_dict())

        self.step_ctr += 1

        # I do not know a better solution for getting the lr from the scheduler.
        # This will fail for different lrs for different layers.
        lr_for_the_update = self.lr_scheduler.get_lr()[0]

        self.lr_scheduler.step()
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "lr": lr_for_the_update,
            "average_q": qs.mean(),
        }
