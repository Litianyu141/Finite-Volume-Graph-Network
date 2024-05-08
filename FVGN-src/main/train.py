import os
import torch
from torch.optim import AdamW
import numpy as np
from FVGN_model.FVGN import FVGN
from dataset import Load_mesh
from utils import get_param, scheduler
from utils.loss_compute import compute_FVM_loss, direct_decode_loss
from utils.get_param import get_hyperparam
from utils.Logger import Logger
from torch_geometric.loader import DataLoader
import subprocess
from torch_geometric.nn import global_mean_pool
import time

# configurate parameters
params, git_info = get_param.params()

# git information
if git_info is not False:
    git_info = {
        "git_branch": params.git_branch,
        "git_commit_dates": params.git_commit_dates,
    }
else:
    git_info = {"git_branch": " ", "git_commit_dates": " "}

# for saving model
# torch.manual_seed(0)

# check cuda
torch.cuda.set_per_process_memory_fraction(0.99, 0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize flow parameters
rho = params.rho
mu = params.mu
dt = params.dt

# initialize fluid model
model = FVGN(device=device, params=params)

fluid_model = model.to(device)
fluid_model.train()
optimizer = AdamW(fluid_model.parameters(), lr=params.lr)

two_step_scheduler = scheduler.ExpLR(
    optimizer, decay_steps=params.n_epochs - params.before_explr_decay_steps, gamma=1e-4
)

lr_scheduler = scheduler.GradualStepExplrScheduler(
    optimizer,
    multiplier=1.0,
    milestone=[params.lr_milestone],
    gamma=params.lr_milestone_gamma,
    total_epoch=params.before_explr_decay_steps,
    after_scheduler=two_step_scheduler,
    expgamma=params.lr_expgamma,
    decay_steps=params.n_epochs - params.before_explr_decay_steps,
    min_lr=1e-6,
)

# Time step of the training phase, which contains 0-150 for 2 times and 150-450 one time, total 600 steps
trian_time_steps = np.asarray(
    [
        np.random.permutation(params.train_traj_length)
        for i in range(params.dataset_size)
    ]
).astype(np.int64)

# initialize Logger and load model / optimizer if according parameters were given
logger = Logger(
    get_hyperparam(params),
    use_csv=False,
    use_tensorboard=params.log,
    params=params,
    git_info=git_info,
)
if (
    params.load_latest
    or params.load_date_time is not None
    or params.load_index is not None
):
    logger.load_logger(datetime=params.load_date_time, load=False)
    # load_logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=params.log,params=params,git_info=git_info)
    if params.load_optimizer:
        params.load_date_time, params.load_index, trian_time_steps = logger.load_state(
            model=fluid_model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            datetime=params.load_date_time,
            index=params.load_index,
            device=device,
            trian_time_steps=trian_time_steps,
        )
    else:
        params.load_date_time, params.load_index, trian_time_steps = logger.load_state(
            model=fluid_model,
            optimizer=None,
            scheduler=None,
            datetime=params.load_date_time,
            index=params.load_index,
            device=device,
            trian_time_steps=trian_time_steps,
        )
    params.load_index = int(params.load_index)
    print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index

# initialize Training Dataset
start = time.time()
train_datasets = Load_mesh.Data_Pool(params=params, is_traning=True, device=device)
train_datasets._set_status(is_training=True)
host_dataset_dir = train_datasets.load_mesh_to_cpu(
    mode=params.dataset_type, split="train"
)

# train_datasets._set_dataset("full")
train_loader = DataLoader(
    train_datasets,
    batch_size=params.batch_size,
    num_workers=2,
    shuffle=True,
    persistent_workers=True,
)
# noise_std_list=np.array([2e-2,4e-2,6e-2])
end = time.time()
print("Training traj has been loaded time consuming:{0}".format(end - start))

# initialize Validation Dataset
valid_file_path = os.path.split(__file__)[0] + "/validate.py"

# training loss function
loss_function = torch.nn.MSELoss(reduction="mean")

# training loop set
training_round = 0
load_index = params.load_index
if load_index is None or params.start_over:
    load_index = 0
else:
    training_round = load_index
current_num_samples = 0
epoch = 0

while True:
    start = time.time()
    fluid_model.train()

    for batch_index, graph_list in enumerate(train_loader):

        optimizer.zero_grad()
        (
            mbatch_graph_node,
            mbatch_graph_edge,
            graph_old,
            mask_face_interior,
            mask_face_boundary,
        ) = train_datasets.train_datapreprocessing(
            graph_list, dual_edge=params.dual_edge
        )
        current_num_samples += mbatch_graph_node.num_graphs

        """ we define one training_round = dataset_size samples also dataset_size graphs, which means every 1k training sample/graphs counts one epoch"""
        if current_num_samples >= 1000:
            training_round += 1
            current_num_samples = 0
            allow_valid = True

        # stastic for normalizer before back papargation
        if training_round < params.pre_statistics_times:
            with torch.no_grad():
                # accmulate statistics 1 epoch before backpropagation
                _, _, _, _, _ = fluid_model(
                    graph=graph_old,
                    graph_edge=mbatch_graph_edge,
                    graph_node=mbatch_graph_node,
                    rho=rho,
                    mu=mu,
                    dt=dt,
                    edge_one_hot=params.edge_one_hot,
                    cell_one_hot=params.cell_one_hot,
                    device=device,
                )

        else:
            # forwarding the model,graph_old`s cell and edge attr has been normalized but without model update
            (
                loss_continuity,
                predicted_delta_u,
                target_delta_u,
                predicted_edge_attr,
                target_face_uvp_normalized,
            ) = fluid_model(
                graph=graph_old,
                graph_edge=mbatch_graph_edge,
                graph_node=mbatch_graph_node,
                rho=rho,
                mu=mu,
                dt=dt,
                edge_one_hot=params.edge_one_hot,
                cell_one_hot=params.cell_one_hot,
                device=device,
            )

            loss, loss_continuity, loss_mom, loss_face_uv, loss_face_p = (
                compute_FVM_loss(
                    params=params,
                    graph_cell=graph_old,
                    graph_edge=mbatch_graph_edge,
                    mask_face_interior=mask_face_interior,
                    predicted_edge_attr=predicted_edge_attr,
                    target_face_uvp_normalized=target_face_uvp_normalized,
                    predicted_delta_u=predicted_delta_u,
                    target_delta_u=target_delta_u,
                    loss_function=loss_function,
                )
            )

            # compute gradients
            loss.backward()

            # perform optimization step
            learning_rate = optimizer.state_dict()["param_groups"][0]["lr"]
            optimizer.step()

        if training_round > params.pre_statistics_times and (current_num_samples == 0):
            logger.log(f"loss_{params.loss}", loss.item(), training_round)
            logger.log(f"loss_mom", loss_mom.mean().item(), training_round)
            logger.log(f"loss_face_p", loss_face_p.mean().item(), training_round)
            logger.log(f"loss_face_uv", loss_face_uv.mean().item(), training_round)
            logger.log(f"learning_rate", learning_rate, training_round)
            
            print(
                f"{epoch:5}: training_round: {training_round:5}; "
                f"{params.loss:15}: {loss.item():8.2f}; "
                f"loss_mom: {loss_mom.mean().item():8.5f}; "
                f"loss_face_uv: {loss_face_uv.mean().item():8.5f}; "
                f"loss_face_p: {loss_face_p.mean().item():8.5f}; "
                f"ever epoch time costs: {(time.time()-start):6.2f}s; "
                f"git commit datetime: {params.git_commit_dates:20}; "
                f"current branch: {params.git_branch:15}"
            )
            
            print(f"lr: {learning_rate:e}")
            start = time.time()

            lr_scheduler.step()

        # save state after every 20 epoch and start a validation process
        if (
            ((training_round) % 5 == 0)
            and (params.log)
            and (training_round > params.pre_statistics_times)
            and (training_round > 0)
            and (current_num_samples == 0)
        ):
        # if True:
            model_saved_path = logger.save_state(
                model=fluid_model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                index=training_round,
                trian_time_steps=trian_time_steps.copy(),
            )
            training_round = int(training_round)
            try:
                process.wait()
                command_output = process.stdout.read().decode("utf-8")
                print(command_output)
            except:
                pass
            finally:
                process = subprocess.Popen(
                    [
                        "python",
                        f"{logger.target_valid_file_path}",
                        f"--model_dir={model_saved_path}",
                        f"--valid_dataset_dir={host_dataset_dir}",
                        f"--training_round={training_round}",
                        f"--valid_dataset_type={params.dataset_type}",
                        f"--host_device={'cuda:1'}", # we recommand train with 2 cuda device, one is for training and another for real time testing/validating to reach fastest traing speed
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

    if training_round > params.n_epochs:
        break

    epoch += 1

print("training done")
