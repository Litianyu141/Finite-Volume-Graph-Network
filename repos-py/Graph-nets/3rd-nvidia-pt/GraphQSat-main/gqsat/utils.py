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


import argparse
import torch
import numpy as np
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX
import time
import pickle
from collections import deque
import os
import sys


def add_common_options(parser):
    parser.add_argument(
        "--with_restarts",
        action="store_true",
        help="Do restarts in Minisat if set",
        dest="with_restarts",
    )
    parser.add_argument(
        "--no_restarts",
        action="store_false",
        help="Do not do restarts in Minisat if set",
        dest="with_restarts",
    )
    parser.set_defaults(with_restarts=False)

    parser.add_argument(
        "--compare_with_restarts",
        action="store_true",
        help="Compare to MiniSAT with restarts",
        dest="compare_with_restarts",
    )
    parser.add_argument(
        "--compare_no_restarts",
        action="store_false",
        help="Compare to MiniSAT without restarts",
        dest="compare_with_restarts",
    )
    parser.set_defaults(compare_with_restarts=False)
    parser.add_argument(
        "--test_max_data_limit_per_set",
        type=int,
        help="Max number of problems to load from the dataset for the env. EVAL/TEST mode.",
        default=None,
    )

    parser.add_argument(
        "--test_time_max_decisions_allowed",
        type=int,
        help="Number of steps the agent will act from the beginning of the episode when evaluating. "
        "Otherwise it will return -1 asking minisat to make a decision. "
        "Float because I want infinity by default (no minisat at all)",
    )
    parser.add_argument("--env-name", type=str, default="sat-v0", help="Environment.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Modify the flow of the script, i.e. run for less iterations",
    )

    parser.add_argument(
        "--model-dir",
        help="Path to the folder with checkpoints and model.yaml file",
        type=str,
    )
    parser.add_argument(
        "--model-checkpoint",
        help="Filename of the checkpoint, relative to the --model-dir param.",
        type=str,
    )
    parser.add_argument("--logdir", type=str, help="Dir for writing the logs")
    parser.add_argument(
        "--eps-final", type=float, default=0.1, help="Final epsilon value."
    )
    parser.add_argument(
        "--eval-problems-paths",
        help="Path to the problem dataset for evaluation",
        type=str,
    )
    parser.add_argument(
        "--train_max_data_limit_per_set",
        type=int,
        help="Max number of problems to load from the dataset for the env. TRAIN mode.",
        default=None,
    )
    parser.add_argument("--no-cuda", action="store_true", help="Use the cpu")

    parser.add_argument(
        "--dump_timings_path",
        type=str,
        help="If not empty, defines the directory to save the wallclock time performance",
    )


def build_eval_argparser():
    # for eval we want to use mostly the args used for training these will override those used for training, be careful
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core-steps",
        type=int,
        help="Number of message passing iterations. "
        "\-1 for the same number as used for training",
        default=-1,
    )
    parser.add_argument(
        "--eval-time-limit",
        type=int,
        help="Time limit for evaluation. If it takes more, return what it has and quit eval. In seconds.",
    )

    add_common_options(parser)
    return parser


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--bsize", type=int, default=128, help="Batch size for the learning step"
    )
    parser.add_argument(
        "--eps-init", type=float, default=1.0, help="Exploration epsilon"
    )

    parser.add_argument(
        "--expert-exploration-prob",
        type=float,
        default=0.0,
        help="When do an exploratory action, choose minisat action with this prob. Otherwise choose random.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discounting")
    parser.add_argument(
        "--eps-decay-steps",
        type=int,
        default=5000,
        help="How many transitions to decay for.",
    )
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=1000,
        help="How often to copy the parameters to traget.",
    )
    parser.add_argument(
        "--batch-updates",
        type=int,
        default=1000000000,
        help="num of batch updates to train for",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10000, help="Memory Replay size."
    )
    parser.add_argument(
        "--history-len", type=int, default=1, help="Frames to stack for the input"
    )
    parser.add_argument(
        "--init-exploration-steps",
        type=int,
        default=100,
        help="Start learning after this number of transitions.",
    )
    parser.add_argument(
        "--step-freq", type=int, default=4, help="Step every k-th frame"
    )
    parser.add_argument("--loss", default="mse", help="Loss to use: mse|huber")
    parser.add_argument(
        "--save-freq",
        default=100000,
        type=int,
        help="How often to save the model. Measured in minibatch updates.",
    )

    parser.add_argument("--train-problems-paths", type=str)

    parser.add_argument(
        "--eval-freq",
        default=10000,
        type=int,
        help="How often to evaluate. Measured in minibatch updates.",
    )
    parser.add_argument(
        "--eval-time-limit",
        default=3600,
        type=int,
        help="Time limit for evaluation. If it takes more, return what it has and quit eval. In seconds.",
    )

    parser.add_argument(
        "--status-dict-path", help="Path to the saved status dict", type=str
    )
    parser.add_argument(
        "--core-steps", type=int, default=8, help="Number of message passing iterations"
    )

    parser.add_argument(
        "--priority_alpha", type=float, default=0.5, help="Alpha in the PER"
    )
    parser.add_argument(
        "--priority_beta",
        type=float,
        default=0.5,
        help="Initial value of the IS weight in PER. Annealed to 1 during training.",
    )
    parser.add_argument(
        "--opt", default="sgd", help="Optimizer to use: sgd|adam|rmsprop"
    )
    parser.add_argument(
        "--e2v-aggregator",
        default="sum",
        help="Aggregation to use for e->v. Can be sum|mean",
    )

    parser.add_argument(
        "--n_hidden", type=int, default=1, help="Number of hidden layers for all MLPs."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Number of units in MLP hidden layers.",
    )
    parser.add_argument(
        "--decoder_v_out_size",
        type=int,
        default=32,
        help="Vertex size of the decoder output.",
    )
    parser.add_argument(
        "--decoder_e_out_size",
        type=int,
        default=1,
        help="Edge size of the decoder output.",
    )
    parser.add_argument(
        "--decoder_g_out_size",
        type=int,
        default=1,
        help="Global attr size of the decoder output.",
    )
    parser.add_argument(
        "--encoder_v_out_size",
        type=int,
        default=32,
        help="Vertex size of the decoder output.",
    )
    parser.add_argument(
        "--encoder_e_out_size",
        type=int,
        default=32,
        help="Edge size of the decoder output.",
    )
    parser.add_argument(
        "--encoder_g_out_size",
        type=int,
        default=32,
        help="Global attr size of the decoder output.",
    )
    parser.add_argument(
        "--core_v_out_size",
        type=int,
        default=64,
        help="Vertex size of the decoder output.",
    )
    parser.add_argument(
        "--core_e_out_size",
        type=int,
        default=64,
        help="Edge size of the decoder output.",
    )
    parser.add_argument(
        "--core_g_out_size",
        type=int,
        default=32,
        help="Global attr size of the decoder output.",
    )
    parser.add_argument(
        "--independent_block_layers",
        type=int,
        default=0,
        help="Number of hidden layers in the encoder/decoder",
    )

    # example from https://stackoverflow.com/a/31347222
    parser.add_argument(
        "--eval_separately_on_each",
        dest="eval_separately_on_each",
        help="If you provide multiple eval datasets e.g. path1:path2, it will "
        "evaluate separately on each and tensorboard/metric them seaprately.",
        action="store_true",
    )
    parser.add_argument(
        "--no_eval_separately_on_each",
        dest="eval_separately_on_each",
        help="If you provide multiple eval datasets e.g. path1:path2, it will "
        "evaluate JOINTLY on them",
        action="store_false",
    )
    parser.set_defaults(eval_separately_on_each=True)

    parser.add_argument(
        "--train_time_max_decisions_allowed",
        type=int,
        default=sys.maxsize,
        help="Number of steps the agent will act from the beginning of the episode when training. "
        "Otherwise it will return -1 asking minisat to make a decision. "
        "Float because I want infinity by default (no minisat at all)",
    )

    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "tanh"],
        help="Activation function",
    )

    parser.add_argument(
        "--lr_scheduler_gamma",
        type=float,
        default=1.0,
        help="Scheduler multiplies lr by this number each LR_SCHEDULER_FREQUENCY number of steps",
    )
    parser.add_argument(
        "--lr_scheduler_frequency",
        type=int,
        default=1000,
        help="Every this number of steps, we multiply the lr by LR_SCHEDULER_GAMMA",
    )

    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Clip gradient by norm."
    )
    parser.add_argument(
        "--grad_clip_norm_type",
        type=float,
        default=2,
        help='Which norm to use when clipping. Use float("inf") to use maxnorm.',
    )

    parser.add_argument(
        "--max_cap_fill_buffer",
        dest="max_cap_fill_buffer",
        help="If true, when cap is surpassed, use -1 and return each state",
        action="store_true",
    )
    parser.add_argument(
        "--no_max_cap_fill_buffer",
        dest="max_cap_fill_buffer",
        help="If this is on, when cap is surpassed, play till the end and return last state only",
        action="store_false",
    )
    parser.set_defaults(max_cap_fill_buffer=False)

    parser.add_argument(
        "--penalty_size",
        type=float,
        default=0.0001,
        help="amount of penalty to apply each step",
    )

    add_common_options(parser)

    return parser


def batch_graphs(graphs, device):
    # we treat a batch of graphs as a one mega graph with several components disconnected from each other
    # we can simply concat the data and adjust the connectivity indices

    vertex_sizes = torch.tensor([el[0].shape[0] for el in graphs], dtype=torch.long)
    edge_sizes = torch.tensor([el[1].shape[0] for el in graphs], dtype=torch.long)
    vcumsize = np.cumsum(vertex_sizes)
    variable_nodes_sizes = torch.tensor(
        [el[0][el[0][:, VAR_ID_IDX] == 1].shape[0] for el in graphs],
        dtype=torch.long,
        device=device,
    )

    vbatched = torch.cat([el[0] for el in graphs])
    ebatched = torch.cat([el[1] for el in graphs])
    gbatched = torch.cat([el[3] for el in graphs])  # 3 is for global

    conn = torch.cat([el[2] for el in graphs], dim=1)
    conn_adjuster = vcumsize.roll(1)
    conn_adjuster[0] = 0
    conn_adjuster = torch.tensor(
        np.concatenate(
            [edge_sizes[vidx].item() * [el] for vidx, el in enumerate(conn_adjuster)]
        ),
        dtype=torch.long,
        device=device,
    )
    conn = conn + conn_adjuster.expand(2, -1)

    v_graph_belonging = torch.tensor(
        np.concatenate([el.item() * [gidx] for gidx, el in enumerate(vertex_sizes)]),
        dtype=torch.long,
        device=device,
    )
    e_graph_belonging = torch.tensor(
        np.concatenate([el.item() * [gidx] for gidx, el in enumerate(edge_sizes)]),
        dtype=torch.long,
        device=device,
    )

    return (
        vbatched,
        ebatched,
        conn,
        variable_nodes_sizes,
        v_graph_belonging,
        e_graph_belonging,
        gbatched,
    )


import gym, minisat


def make_env(problems_paths, args, test_mode=False):
    max_data_limit_per_set = None
    if test_mode and hasattr(args, "test_max_data_limit_per_set"):
        max_data_limit_per_set = args.test_max_data_limit_per_set
    if not test_mode and hasattr(args, "train_max_data_limit_per_set"):
        max_data_limit_per_set = args.train_max_data_limit_per_set
    return gym.make(
        args.env_name,
        problems_paths=problems_paths,
        args=args,
        test_mode=test_mode,
        max_cap_fill_buffer=False if test_mode else args.max_cap_fill_buffer,
        penalty_size=args.penalty_size if hasattr(args, "penalty_size") else None,
        with_restarts=args.with_restarts if hasattr(args, "with_restarts") else None,
        compare_with_restarts=args.compare_with_restarts
        if hasattr(args, "compare_with_restarts")
        else None,
        max_data_limit_per_set=max_data_limit_per_set,
    )


def evaluate(agent, args, include_train_set=False):
    agent.net.eval()
    problem_sets = (
        [args.eval_problems_paths]
        if not args.eval_separately_on_each
        else [k for k in args.eval_problems_paths.split(":")]
    )
    if include_train_set:
        problem_sets.extend(
            [args.train_problems_paths]
            if not args.eval_separately_on_each
            else [k for k in args.train_problems_paths.split(":")]
        )

    res = {}

    st_time = time.time()
    print("Starting evaluation. Fasten your seat belts!")

    total_iters_ours = 0
    total_iters_minisat = 0

    for pset in problem_sets:
        eval_env = make_env(pset, args, test_mode=True)
        DEBUG_ROLLOUTS = None
        pr = 0
        walltime = {}
        scores = {}
        with torch.no_grad():
            while eval_env.test_to != 0 or pr == 0:
                p_st_time = time.time()
                obs = eval_env.reset(
                    max_decisions_cap=args.test_time_max_decisions_allowed
                )
                done = eval_env.isSolved

                while not done:
                    # if time.time() - st_time > args.eval_time_limit:
                    #     print(
                    #         "Eval time limit surpassed. Returning what I have, and quitting the evaluation."
                    #     )
                    #     return res, eval_env.metadata, False

                    action = agent.act([obs], eps=0)
                    obs, _, done, _ = eval_env.step(action)

                walltime[eval_env.curr_problem] = time.time() - p_st_time
                print(
                    f"It took {walltime[eval_env.curr_problem]} seconds to solve a problem."
                )
                sctr = 1 if eval_env.step_ctr == 0 else eval_env.step_ctr
                ns = eval_env.normalized_score(sctr, eval_env.curr_problem)
                print(f"Evaluation episode {pr+1} is over. Your score is {ns}.")
                total_iters_ours += sctr
                pdir, pname = os.path.split(eval_env.curr_problem)
                total_iters_minisat += eval_env.metadata[pdir][pname][1]
                scores[eval_env.curr_problem] = ns
                pr += 1
                if DEBUG_ROLLOUTS is not None and pr >= DEBUG_ROLLOUTS:
                    break
        print(
            f"Evaluation is done. Median relative score: {np.nanmedian([el for el in scores.values()]):.2f}, "
            f"mean relative score: {np.mean([el for el in scores.values()]):.2f}, "
            f"iters frac: {total_iters_minisat/total_iters_ours:.2f}"
        )
        res[pset] = scores

    if args.dump_timings_path:
        target_fname = (
            os.path.join(
                args.dump_timings_path,
                args.eval_problems_paths.replace("/", "_")
                + f"_cap_{args.test_time_max_decisions_allowed}",
            )
            + ".pkl"
        )
        with open(target_fname, "wb") as f:
            pickle.dump(walltime, f)
    agent.net.train()
    return (
        res,
        {
            "metadata": eval_env.metadata,
            "iters_frac": total_iters_minisat / total_iters_ours,
            "mean_score": np.mean([el for el in scores.values()]),
            "median_score": np.median([el for el in scores.values()]),
        },
        False,
    )
