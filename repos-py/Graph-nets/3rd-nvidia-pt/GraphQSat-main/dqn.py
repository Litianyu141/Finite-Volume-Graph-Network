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

import os
from collections import deque
import pickle
import copy
import yaml

from gqsat.utils import build_argparser, evaluate, make_env
from gqsat.models import EncoderCoreDecoder, SatModel
from gqsat.agents import GraphAgent
from gqsat.learners import GraphLearner
from gqsat.buffer import ReplayGraphBuffer

from tensorboardX import SummaryWriter


def save_training_state(
    model,
    learner,
    episodes_done,
    transitions_seen,
    best_eval_so_far,
    args,
    in_eval_mode=False,
):
    # save the model
    model_path = os.path.join(args.logdir, f"model_{learner.step_ctr}.chkp")
    torch.save(model.state_dict(), model_path)

    # save the experience replay
    buffer_path = os.path.join(args.logdir, "buffer.pkl")

    with open(buffer_path, "wb") as f:
        pickle.dump(learner.buffer, f)

    # save important parameters
    train_status = {
        "step_ctr": learner.step_ctr,
        "latest_model_name": model_path,
        "buffer_path": buffer_path,
        "args": args,
        "episodes_done": episodes_done,
        "logdir": args.logdir,
        "transitions_seen": transitions_seen,
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "optimizer_class": type(learner.optimizer),
        "best_eval_so_far": best_eval_so_far,
        "scheduler_class": type(learner.lr_scheduler),
        "scheduler_state_dict": learner.lr_scheduler.state_dict(),
        "in_eval_mode": in_eval_mode,
    }
    status_path = os.path.join(args.logdir, "status.yaml")

    with open(status_path, "w") as f:
        yaml.dump(train_status, f, default_flow_style=False)

    return status_path


def get_annealed_eps(n_trans, args):
    if n_trans < args.init_exploration_steps:
        return args.eps_init
    if n_trans > args.eps_decay_steps:
        return args.eps_final
    else:
        assert n_trans - args.init_exploration_steps >= 0
        return (args.eps_init - args.eps_final) * (
            1 - (n_trans - args.init_exploration_steps) / args.eps_decay_steps
        ) + args.eps_final


def arg2activation(activ_str):
    if activ_str == "relu":
        return torch.nn.ReLU
    elif activ_str == "tanh":
        return torch.nn.Tanh
    elif activ_str == "leaky_relu":
        return torch.nn.LeakyReLU
    else:
        raise ValueError("Unknown activation function")


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    args.device = (
        torch.device("cpu")
        if args.no_cuda or not torch.cuda.is_available()
        else torch.device("cuda")
    )

    if args.status_dict_path:
        # training mode, resuming from the status dict

        # load the train status dict
        with open(args.status_dict_path, "r") as f:
            train_status = yaml.load(f, Loader=yaml.Loader)
        eval_resume_signal = train_status["in_eval_mode"]
        # swap the args
        args = train_status["args"]

        # load the model
        net = SatModel.load_from_yaml(os.path.join(args.logdir, "model.yaml")).to(
            args.device
        )
        net.load_state_dict(torch.load(train_status["latest_model_name"]))

        target_net = SatModel.load_from_yaml(
            os.path.join(args.logdir, "model.yaml")
        ).to(args.device)
        target_net.load_state_dict(net.state_dict())

        # load the buffer
        with open(train_status["buffer_path"], "rb") as f:
            buffer = pickle.load(f)
        learner = GraphLearner(net, target_net, buffer, args)
        learner.step_ctr = train_status["step_ctr"]

        learner.optimizer = train_status["optimizer_class"](
            net.parameters(), lr=args.lr
        )
        learner.optimizer.load_state_dict(train_status["optimizer_state_dict"])
        learner.lr_scheduler = train_status["scheduler_class"](
            learner.optimizer, args.lr_scheduler_frequency, args.lr_scheduler_gamma
        )
        learner.lr_scheduler.load_state_dict(train_status["scheduler_state_dict"])

        # load misc training status params
        n_trans = train_status["transitions_seen"]
        ep = train_status["episodes_done"]

        env = make_env(args.train_problems_paths, args, test_mode=False)

        agent = GraphAgent(net, args)

        best_eval_so_far = train_status["best_eval_so_far"]

    else:
        # training mode, learning from scratch or continuing learning from some previously trained model
        writer = SummaryWriter()
        args.logdir = writer.logdir

        model_save_path = os.path.join(args.logdir, "model.yaml")
        best_eval_so_far = (
            {args.eval_problems_paths: -1}
            if not args.eval_separately_on_each
            else {k: -1 for k in args.eval_problems_paths.split(":")}
        )

        env = make_env(args.train_problems_paths, args, test_mode=False)
        if args.model_dir is not None:
            # load an existing model and continue training
            net = SatModel.load_from_yaml(
                os.path.join(args.model_dir, "model.yaml")
            ).to(args.device)
            net.load_state_dict(
                torch.load(os.path.join(args.model_dir, args.model_checkpoint))
            )
        else:
            # learning from scratch
            net = EncoderCoreDecoder(
                (env.vertex_in_size, env.edge_in_size, env.global_in_size),
                core_out_dims=(
                    args.core_v_out_size,
                    args.core_e_out_size,
                    args.core_e_out_size,
                ),
                out_dims=(2, None, None),
                core_steps=args.core_steps,
                dec_out_dims=(
                    args.decoder_v_out_size,
                    args.decoder_e_out_size,
                    args.decoder_e_out_size,
                ),
                encoder_out_dims=(
                    args.encoder_v_out_size,
                    args.encoder_e_out_size,
                    args.encoder_e_out_size,
                ),
                save_name=model_save_path,
                e2v_agg=args.e2v_aggregator,
                n_hidden=args.n_hidden,
                hidden_size=args.hidden_size,
                activation=arg2activation(args.activation),
                independent_block_layers=args.independent_block_layers,
            ).to(args.device)
        print(str(net))
        target_net = copy.deepcopy(net)

        buffer = ReplayGraphBuffer(args, args.buffer_size)
        agent = GraphAgent(net, args)

        n_trans = 0
        ep = 0
        learner = GraphLearner(net, target_net, buffer, args)
        eval_resume_signal = False

    print(args.__str__())
    loss = None

    while learner.step_ctr < args.batch_updates:

        ret = 0
        obs = env.reset(args.train_time_max_decisions_allowed)
        done = env.isSolved

        if args.history_len > 1:
            raise NotImplementedError(
                "History len greater than one is not implemented for graph nets."
            )
        hist_buffer = deque(maxlen=args.history_len)
        for _ in range(args.history_len):
            hist_buffer.append(obs)
        ep_step = 0

        save_flag = False

        while not done:
            annealed_eps = get_annealed_eps(n_trans, args)
            action = agent.act(hist_buffer, eps=annealed_eps)
            next_obs, r, done, _ = env.step(action)
            buffer.add_transition(obs, action, r, done)
            obs = next_obs

            hist_buffer.append(obs)
            ret += r

            if (not n_trans % args.step_freq) and (
                buffer.ctr > max(args.init_exploration_steps, args.bsize + 1)
                or buffer.full
            ):
                step_info = learner.step()
                if annealed_eps is not None:
                    step_info["annealed_eps"] = annealed_eps

                # we increment the step_ctr in the learner.step(), that's why we need to do -1 in tensorboarding
                # we do not need to do -1 in checking for frequency since 0 has already passed

                if not learner.step_ctr % args.save_freq:
                    # save the exact model you evaluated and make another save after the episode ends
                    # to have proper transitions in the replay buffer to pickle
                    status_path = save_training_state(
                        net,
                        learner,
                        ep - 1,
                        n_trans,
                        best_eval_so_far,
                        args,
                        in_eval_mode=eval_resume_signal,
                    )
                    save_flag = True
                if (
                    args.env_name == "sat-v0" and not learner.step_ctr % args.eval_freq
                ) or eval_resume_signal:
                    scores, _, eval_resume_signal = evaluate(
                        agent, args, include_train_set=False
                    )

                    for sc_key, sc_val in scores.items():
                        # list can be empty if we hit the time limit for eval
                        if len(sc_val) > 0:
                            res_vals = [el for el in sc_val.values()]
                            median_score = np.nanmedian(res_vals)
                            if (
                                best_eval_so_far[sc_key] < median_score
                                or best_eval_so_far[sc_key] == -1
                            ):
                                best_eval_so_far[sc_key] = median_score
                            writer.add_scalar(
                                f"data/median relative score: {sc_key}",
                                np.nanmedian(res_vals),
                                learner.step_ctr - 1,
                            )
                            writer.add_scalar(
                                f"data/mean relative score: {sc_key}",
                                np.nanmean(res_vals),
                                learner.step_ctr - 1,
                            )
                            writer.add_scalar(
                                f"data/max relative score: {sc_key}",
                                np.nanmax(res_vals),
                                learner.step_ctr - 1,
                            )
                    for k, v in best_eval_so_far.items():
                        writer.add_scalar(k, v, learner.step_ctr - 1)

                for k, v in step_info.items():
                    writer.add_scalar(k, v, learner.step_ctr - 1)

                writer.add_scalar("data/num_episodes", ep, learner.step_ctr - 1)

            n_trans += 1
            ep_step += 1

        writer.add_scalar("data/ep_return", ret, learner.step_ctr - 1)
        writer.add_scalar("data/ep_steps", env.step_ctr, learner.step_ctr - 1)
        writer.add_scalar("data/ep_last_reward", r, learner.step_ctr - 1)
        print(f"Episode {ep + 1}: Return {ret}.")
        ep += 1

        if save_flag:
            status_path = save_training_state(
                net,
                learner,
                ep - 1,
                n_trans,
                best_eval_so_far,
                args,
                in_eval_mode=eval_resume_signal,
            )
            save_flag = False
