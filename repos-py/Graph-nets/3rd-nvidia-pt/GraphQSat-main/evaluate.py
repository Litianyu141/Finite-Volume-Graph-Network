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

import numpy as np

import torch
import pickle
import yaml

from gqsat.utils import build_eval_argparser, evaluate
from gqsat.models import SatModel
from gqsat.agents import GraphAgent

import os
import time

if __name__ == "__main__":
    parser = build_eval_argparser()
    eval_args = parser.parse_args()
    with open(os.path.join(eval_args.model_dir, "status.yaml"), "r") as f:
        train_status = yaml.load(f, Loader=yaml.Loader)
        args = train_status["args"]

    # use same args used for training and overwrite them with those asked for eval
    for k, v in vars(eval_args).items():
        setattr(args, k, v)

    args.device = (
        torch.device("cpu")
        if args.no_cuda or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    net = SatModel.load_from_yaml(os.path.join(args.model_dir, "model.yaml")).to(
        args.device
    )

    # modify core steps for the eval as requested
    if args.core_steps != -1:
        # -1 if use the same as for training
        net.steps = args.core_steps

    net.load_state_dict(
        torch.load(os.path.join(args.model_dir, args.model_checkpoint)), strict=False
    )

    agent = GraphAgent(net, args)

    st_time = time.time()
    scores, eval_metadata, _ = evaluate(agent, args)
    end_time = time.time()

    print(
        f"Evaluation is over. It took {end_time - st_time} seconds for the whole procedure"
    )

    # with open("../eval_results.pkl", "wb") as f:
    #     pickle.dump(scores, f)

    for pset, pset_res in scores.items():
        res_list = [el for el in pset_res.values()]
        print(f"Results for {pset}")
        print(
            f"median_relative_score: {np.nanmedian(res_list)}, mean_relative_score: {np.mean(res_list)}"
        )
