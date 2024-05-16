import csv
import torch
import argparse
import torch_geometric.transforms as T
import numpy as np
from FVGN_model.FVGN import FVGN
import os
from dataset import Load_mesh
from utils import get_param, utilities
from utils.get_param import get_hyperparam
from utils.Logger import Logger
import time
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
from utils import write_tec
import matplotlib

matplotlib.use("Agg")
from matplotlib import tri as mtri
import matplotlib.pyplot as plt

import gc


def plot_MSE(MSE, RMSE, result_dir):
    MSE = MSE.numpy()
    RMSE = RMSE.numpy()
    pMSE = [MSE[:, 0], MSE[:, 1], MSE[:, 2], RMSE[:, 0], RMSE[:, 1], RMSE[:, 2]]
    pRMSE = [RMSE[:, 0], RMSE[:, 1], RMSE[:, 2]]
    x = np.arange(0, MSE.shape[0])
    """
    plt.subplot(121)
    plt.plot(MSE[0:599,0], linewidth=1, color="red", marker="o",label="U MSE")
    plt.plot(MSE[0:599,1], linewidth=1, color="orange", marker="o",label="V MSE")
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplot(122)
    plt.plot(MSE[0:599,2], linewidth=1, color="orange", marker="o",label="P MSE")
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(result_dir+'test_zhexian.png')
    plt.show()
    """


def rollout_error(predicteds, targets):

    number_len = targets.shape[0]
    squared_diff = np.square(predicteds - targets).reshape(number_len, -1)
    loss = np.sqrt(
        np.cumsum(np.mean(squared_diff, axis=1), axis=0) / np.arange(1, number_len + 1)
    )

    for show_step in range(0, 1000000, 50):
        if show_step < number_len:
            print("testing rmse  @step %d loss: %.2e" % (show_step, loss[show_step]))
        else:
            break

    return loss


def extract_cylinder_boundary(graph_node: Data, graph_edge: Data, graph_cell: Data):

    face_node = graph_node.edge_index
    node_type = graph_node.node_type
    mesh_pos = graph_node.pos
    centroid = graph_cell.pos
    cell_face = graph_edge.face
    cell_type = graph_cell.x[:, 0:1]
    cells_three_pos = [
        torch.index_select(mesh_pos, 0, graph_node.face[0]),
        torch.index_select(mesh_pos, 0, graph_node.face[1]),
        torch.index_select(mesh_pos, 0, graph_node.face[2]),
    ]

    node_topwall = torch.max(mesh_pos[:, 1])
    node_bottomwall = torch.min(mesh_pos[:, 1])
    node_outlet = torch.max(mesh_pos[:, 0])
    node_inlet = torch.min(mesh_pos[:, 0])

    face_type = graph_edge.x[:, 0:1]
    left_face_node_pos = torch.index_select(mesh_pos, 0, face_node[0])
    right_face_node_pos = torch.index_select(mesh_pos, 0, face_node[1])

    left_face_node_type = torch.index_select(node_type, 0, face_node[0])
    right_face_node_type = torch.index_select(node_type, 0, face_node[1])

    face_center_pos = (left_face_node_pos + right_face_node_pos) / 2.0

    face_topwall = torch.max(face_center_pos[:, 1])
    face_bottomwall = torch.min(face_center_pos[:, 1])
    face_outlet = torch.max(face_center_pos[:, 0])
    face_inlet = torch.min(face_center_pos[:, 0])

    centroid_topwall = torch.max(centroid[:, 1])
    centroid_bottomwall = torch.min(centroid[:, 1])
    centroid_outlet = torch.max(centroid[:, 0])
    centroid_inlet = torch.min(centroid[:, 0])

    MasknodeT = torch.full((mesh_pos.shape[0], 1), True).cuda()
    MasknodeF = torch.logical_not(MasknodeT).cuda()

    MaskfaceT = torch.full((face_node.shape[1], 1), True).cuda()
    MaskfaceF = torch.logical_not(MaskfaceT).cuda()

    MaskcellT = torch.full((cell_face.shape[1], 1), True).cuda()
    MaskcellF = torch.logical_not(MaskcellT).cuda()

    cylinder_node_mask = torch.where(
        (
            (node_type == utilities.NodeType.WALL_BOUNDARY)
            & (mesh_pos[:, 1:2] < node_topwall)
            & (mesh_pos[:, 1:2] > node_bottomwall)
            & (mesh_pos[:, 0:1] > node_inlet)
            & (mesh_pos[:, 0:1] < node_outlet)
        ),
        MasknodeT,
        MasknodeF,
    ).squeeze(1)

    cylinder_face_mask = torch.where(
        (
            (face_type == utilities.NodeType.WALL_BOUNDARY)
            & (face_center_pos[:, 1:2] < face_topwall)
            & (face_center_pos[:, 1:2] > face_bottomwall)
            & (face_center_pos[:, 0:1] > face_inlet)
            & (face_center_pos[:, 0:1] < face_outlet)
            & (left_face_node_pos[:, 1:2] < node_topwall)
            & (left_face_node_pos[:, 1:2] > node_bottomwall)
            & (left_face_node_pos[:, 0:1] > node_inlet)
            & (left_face_node_pos[:, 0:1] < node_outlet)
            & (right_face_node_pos[:, 1:2] < node_topwall)
            & (right_face_node_pos[:, 1:2] > node_bottomwall)
            & (right_face_node_pos[:, 0:1] > node_inlet)
            & (right_face_node_pos[:, 0:1] < node_outlet)
            & (left_face_node_type == utilities.NodeType.WALL_BOUNDARY)
            & (right_face_node_type == utilities.NodeType.WALL_BOUNDARY)
        ),
        MaskfaceT,
        MaskfaceF,
    ).squeeze(1)

    cylinder_cell_mask = torch.where(
        (
            (cell_type == utilities.NodeType.WALL_BOUNDARY)
            & (centroid[:, 1:2] < centroid_topwall)
            & (centroid[:, 1:2] > centroid_bottomwall)
            & (centroid[:, 0:1] > centroid_inlet)
            & (centroid[:, 0:1] < centroid_outlet)
            & (cells_three_pos[0][:, 1:2] < centroid_topwall)
            & (cells_three_pos[0][:, 1:2] > centroid_bottomwall)
            & (cells_three_pos[0][:, 0:1] > centroid_inlet)
            & (cells_three_pos[0][:, 0:1] < centroid_outlet)
            & (cells_three_pos[1][:, 1:2] < centroid_topwall)
            & (cells_three_pos[1][:, 1:2] > centroid_bottomwall)
            & (cells_three_pos[1][:, 0:1] > centroid_inlet)
            & (cells_three_pos[1][:, 0:1] < centroid_outlet)
            & (cells_three_pos[2][:, 1:2] < centroid_topwall)
            & (cells_three_pos[2][:, 1:2] > centroid_bottomwall)
            & (cells_three_pos[2][:, 0:1] > centroid_inlet)
            & (cells_three_pos[2][:, 0:1] < centroid_outlet)
        ),
        MaskcellT,
        MaskcellF,
    ).squeeze(1)

    # plt.scatter(face_center_pos[cylinder_face_mask].cpu().numpy()[:,0],face_center_pos[cylinder_face_mask].cpu().numpy()[:,1],edgecolors='red')
    # plt.show()
    return cylinder_node_mask, cylinder_face_mask, cylinder_cell_mask


def extract_relonyds_number(graph_node, graph_edge):
    """prepare data for cal_relonyds_number"""
    target_on_node = graph_node.x[:, 0, 0:2].cpu()
    edge_index = graph_node.edge_index.cpu()
    target_on_edge = (
        torch.index_select(target_on_node, 0, edge_index[0])
        + torch.index_select(target_on_node, 0, edge_index[1])
    ) / 2.0
    face_type = graph_edge.x[:, 0:1].cpu().view(-1)
    node_type = graph_node.node_type.cpu().view(-1)
    Inlet = target_on_edge[face_type == utilities.NodeType.INFLOW][:, 0]
    face_length = graph_edge.x[:, 1:2].cpu()[:, 0][
        face_type == utilities.NodeType.INFLOW
    ]
    total_u = torch.sum(Inlet * face_length)
    mesh_pos = graph_node.pos.cpu()
    top = torch.max(mesh_pos[:, 1]).numpy()
    bottom = torch.min(mesh_pos[:, 1]).numpy()
    left = torch.min(mesh_pos[:, 0]).numpy()
    right = torch.max(mesh_pos[:, 0]).numpy()
    mean_u = total_u / (top - bottom)

    """cal cylinder diameter"""
    boundary_pos = mesh_pos[node_type == utilities.NodeType.WALL_BOUNDARY].numpy()
    cylinder_mask = torch.full((boundary_pos.shape[0], 1), True).view(-1).numpy()
    cylinder_not_mask = np.logical_not(cylinder_mask)
    cylinder_mask = np.where(
        (
            (boundary_pos[:, 1] > bottom)
            & (boundary_pos[:, 1] < top)
            & (boundary_pos[:, 0] < right)
            & (boundary_pos[:, 0] > left)
        ),
        cylinder_mask,
        cylinder_not_mask,
    )
    boundary_pos_obs = torch.from_numpy(boundary_pos[cylinder_mask])
    # _,_,R,_= hyper_fit(np.asarray(cylinder_pos))
    # L0 = R*2.
    _, wingleft_index = torch.max(boundary_pos_obs[:, 0:1], dim=0)
    _, winglright_index = torch.min(boundary_pos_obs[:, 0:1], dim=0)

    L0 = torch.norm(
        boundary_pos_obs[wingleft_index] - boundary_pos_obs[winglright_index]
    )

    rho = params.rho
    mu = params.mu
    Re = (mean_u * L0 * rho) / mu
    mean_u = mean_u
    return Re, mean_u


@torch.no_grad()
def rollout(
    model,
    datasets,
    batch_index=[],
    result_dir=None,
    rollout_start_step=None,
    rollout_time_length=None,
    rho=None,
    mu=None,
    dt=None,
    dual_edge=False,
    saving_tec=None,
    plot_boundary=False,
):

    result = {}
    predicted_cell_UV_list = []
    predicted_cell_P_list = []
    predicted_edge_UVP_list = []

    face_length_list = []
    targets_UVP_on_cell = []

    # retrive data first
    (
        mbatch_graph_node,
        mbatch_graph_edge,
        graph_old,
        mask_face_interior,
        mask_face_boundary,
    ) = datasets.require_minibatch_mesh(
        start_epoch=rollout_start_step,
        batch_index=batch_index,
        is_training=False,
        dual_edge=dual_edge,
    )

    cylinder_node_mask_list = []
    cylinder_face_mask_list = []
    cylinder_cell_mask_list = []
    Re_num_list = []
    mean_u_list = []

    mbatch_graph_node_list = Batch.to_data_list(mbatch_graph_node)
    mbatch_graph_edge_list = Batch.to_data_list(mbatch_graph_edge)
    graph_old_list = Batch.to_data_list(graph_old)

    for index in range(len(graph_old_list)):
        graph_node = mbatch_graph_node_list[index]
        graph_edge = mbatch_graph_edge_list[index]
        graph_cell = graph_old_list[index]
        cylinder_node_mask, cylinder_face_mask, cylinder_cell_mask = (
            extract_cylinder_boundary(
                graph_node.clone(), graph_edge.clone(), graph_cell.clone()
            )
        )
        cylinder_node_mask_list.append(cylinder_node_mask)
        cylinder_face_mask_list.append(cylinder_face_mask)
        cylinder_cell_mask_list.append(cylinder_cell_mask)

        Re_num, mean_u = extract_relonyds_number(graph_node.clone(), graph_edge.clone())
        Re_num_list.append(Re_num)
        mean_u_list.append(mean_u)

    cylinder_node_mask = torch.cat(cylinder_node_mask_list, dim=0)
    cylinder_face_mask = torch.cat(cylinder_face_mask_list, dim=0)
    cylinder_cell_mask = torch.cat(cylinder_cell_mask_list, dim=0)

    Re_num_list = torch.stack(Re_num_list, dim=0)
    mean_u_list = torch.stack(mean_u_list, dim=0)

    current_mesh_pos = mbatch_graph_node.pos.clone()

    target_UVP_on_face = mbatch_graph_edge.y.transpose(0, 1).clone()
    targets_UVP_on_cell = graph_old.y.transpose(0, 1).clone()
    target_cylinder_boundary_edge_UV = target_UVP_on_face[
        :, cylinder_face_mask, 0:2
    ].clone()
    target_cylinder_boundary_edge_P = target_UVP_on_face[
        :, cylinder_face_mask, 2:3
    ].clone()

    cell_type = graph_old.x[:, 0].clone()
    Re = graph_old.x[:, 1:2].clone()
    edge_RMP_EU = graph_old.edge_attr[:, 2:5].clone()
    face_node = mbatch_graph_node.edge_index.T.clone()
    edge_Euclidean_distance = mbatch_graph_edge.x[:, 1:2].clone()

    """plot mesh"""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def colorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.set_aspect("equal")

    triang = mtri.Triangulation(
        mbatch_graph_node.pos[:, 0].cpu().numpy(),
        mbatch_graph_node.pos[:, 1].cpu().numpy(),
        mbatch_graph_node.face.cpu().T.numpy(),
    )

    bb_min = graph_old.y[:, 0, 0:1].cpu().min()
    bb_max = graph_old.y[:, 0, 0:1].cpu().max()

    ax.triplot(triang, "ko-", ms=0.5, lw=0.3, zorder=1)
    cntr = ax.tripcolor(
        triang,
        graph_old.y[:, 0, 0].cpu().numpy(),
        vmin=bb_min,
        vmax=bb_max,
        shading="flat",
    )

    # 使用fig.colorbar并指定ax参数来自适应colorbar的长度
    cbar = colorbar(cntr)

    ticks = np.linspace(bb_min, bb_max, num=4)
    cbar.set_ticks(ticks)

    # 其他代码
    os.makedirs(
        os.path.join(result_dir, "rollout_index" + str(batch_index[0])), exist_ok=True
    )
    plt.savefig(
        os.path.join(
            result_dir, "rollout_index" + str(batch_index[0]), f"{batch_index[0]}.png"
        )
    )
    plt.close()
    """plot mesh"""

    total_time = 0  # 用于累积总时间

    for epoch in range(rollout_start_step, rollout_time_length):

        start_time = time.time()

        # forwarding the model,graph_old`s cell and edge attr has been normalized but without model upadte
        predicted_uvp_on_edge, next_UV_on_cell = model(
            graph=graph_old,
            graph_edge=mbatch_graph_edge,
            graph_node=mbatch_graph_node,
            rho=rho,
            mu=mu,
            dt=dt,
            edge_one_hot=edge_one_hot,
            cell_one_hot=cell_one_hot,
            device=device,
        )

        end_time = time.time()
        epoch_time = (end_time - start_time) * 1000
        total_time += epoch_time

        # Print the time for each iteration in milliseconds, formatted in scientific notation with 3 decimal places
        print(f"Time for epoch {epoch}: {epoch_time:.3e} ms")

        # don`t update the boundary
        predicted_uvp_on_edge[mask_face_boundary, 0:2] = target_UVP_on_face[
            epoch, mask_face_boundary, 0:2
        ]

        # linear interplot
        predicted_cell_P = (
            torch.index_select(
                predicted_uvp_on_edge[:, 2:], 0, mbatch_graph_edge.face[0, :]
            )
            + torch.index_select(
                predicted_uvp_on_edge[:, 2:], 0, mbatch_graph_edge.face[1, :]
            )
            + torch.index_select(
                predicted_uvp_on_edge[:, 2:], 0, mbatch_graph_edge.face[2, :]
            )
        ) / 3.0

        # for calculating MSE & RMSE
        predicted_cell_UV_list.append(next_UV_on_cell[:, 0:2].unsqueeze(0))
        predicted_cell_P_list.append(predicted_cell_P.unsqueeze(0))
        predicted_edge_UVP_list.append(predicted_uvp_on_edge.clone())
        face_length_list.append(edge_Euclidean_distance.unsqueeze(0))

        graph_old = datasets.create_next_graph(
            graph_old,
            graph_edge,
            mask_face_boundary,
            next_UV_on_cell,
            cell_type.unsqueeze(1),
            Re,
            edge_RMP_EU,
            dual_edge=dual_edge,
        )

    datasets.mbatch_graph_node.clear()
    datasets.mbatch_graph_edge.clear()
    datasets.mbatch_graph_cell.clear()
    rollstep = rollout_time_length

    # fluid zone
    predicted_cell_UV = torch.cat(predicted_cell_UV_list, dim=0)
    predicted_cell_P = torch.cat(predicted_cell_P_list, dim=0)
    face_length = torch.cat(face_length_list)
    predicted_edge_UVP = torch.stack(predicted_edge_UVP_list)
    target_edge_UVP = target_UVP_on_face

    # calc MSE & RMSE
    num_steps = predicted_cell_UV.shape[0]
    time_steps = torch.from_numpy(np.arange(1, num_steps + 1, dtype=np.float32)).to(
        predicted_cell_UV.device
    )

    # 计算速度和压力的误差平方
    squared_error_uv = (
        targets_UVP_on_cell[:, :, 0:2] - predicted_cell_UV[:, :, 0:2]
    ) ** 2
    squared_error_p = (target_edge_UVP[:, :, 2:3] - predicted_edge_UVP[:, :, 2:3]) ** 2

    # 计算累积均方误差 (MSE)
    cumulative_mse_uv = torch.cumsum(torch.mean(squared_error_uv, dim=[1, 2]), dim=0)
    cumulative_mse_p = torch.cumsum(torch.mean(squared_error_p, dim=[1, 2]), dim=0)

    # 除以时间步长以得到平均累积 MSE
    average_cumulative_mse_uv = cumulative_mse_uv / time_steps
    average_cumulative_mse_p = cumulative_mse_p / time_steps

    # 计算 u 和 v 的相对均方根误差 (RMSE)
    cumulative_squared_error_u = torch.sum(squared_error_uv[:, :, 0], dim=1)
    cumulative_squared_error_v = torch.sum(squared_error_uv[:, :, 1], dim=1)

    denominator_u = torch.sum(predicted_cell_UV[:, :, 0] ** 2, dim=1)
    denominator_v = torch.sum(predicted_cell_UV[:, :, 1] ** 2, dim=1)

    cumulative_rmse_u = (
        torch.cumsum(cumulative_squared_error_u / denominator_u, dim=0) / time_steps
    )
    cumulative_rmse_v = (
        torch.cumsum(cumulative_squared_error_v / denominator_v, dim=0) / time_steps
    )

    # 计算 uv 和 p 的相对均方根误差 (RMSE)
    cumulative_squared_error_uv = torch.sum(squared_error_uv, dim=[1, 2])
    cumulative_squared_error_p = torch.sum(squared_error_p, dim=[1, 2])

    denominator_uv = torch.sum(predicted_cell_UV[:, :, 0:2] ** 2, axis=[1, 2])
    denominator_p = torch.sum(predicted_edge_UVP[:, :, 2:3] ** 2, axis=[1, 2])

    cumulative_rmse_uv = (
        torch.cumsum(cumulative_squared_error_uv / denominator_uv, axis=0) / time_steps
    )
    cumulative_rmse_p = (
        torch.cumsum(cumulative_squared_error_p / denominator_p, axis=0) / time_steps
    )

    # 将累积误差转换为字典格式
    scalars = {}

    # 更新字典中的值
    scalars["uv_mse"] = average_cumulative_mse_uv.cpu().numpy()
    scalars["p_mse"] = average_cumulative_mse_p.cpu().numpy()
    scalars["u_rmse"] = cumulative_rmse_u.cpu().numpy()
    scalars["v_rmse"] = cumulative_rmse_v.cpu().numpy()
    scalars["uv_rmse"] = cumulative_rmse_uv.cpu().numpy()
    scalars["p_rmse"] = cumulative_rmse_p.cpu().numpy()

    re_num = Re_num_list[0]
    mean_u = mean_u_list[0]
    graph_index = batch_index[0]
    rollout_index = graph_index
    divergence_dection = predicted_cell_UV[:, :, :].cpu()
    print(f"graph_index_{graph_index}_relonyds_number: %.2e" % re_num)
    print(f"graph_index_{graph_index}_max_U: %.2e" % torch.max(divergence_dection))
    print(f"graph_index_{graph_index}_uv_mse: %.2e" % average_cumulative_mse_uv.mean())
    print(f"graph_index_{graph_index}_p_mse: %.2e" % average_cumulative_mse_p.mean())
    print(f"graph_index_{graph_index}_u_rmse: %.2e" % cumulative_rmse_u.mean())
    print(f"graph_index_{graph_index}_v_rmse: %.2e" % cumulative_rmse_v.mean())
    print(f"graph_index_{graph_index}_uv_rmse: %.2e" % cumulative_rmse_uv.mean())
    print(f"graph_index_{graph_index}_p_rmse: %.2e" % cumulative_rmse_p.mean())

    print(
        "-----------------------NO.{0} mesh ROLL OUT DONE--------------------------------\n".format(
            graph_index
        )
    )

    # composing fluid zone and boundary zone for writing tecplot file and plotting CL and CD
    result["mean_u"] = mean_u.cpu().numpy()
    result["reynolds_num"] = re_num.cpu().numpy()
    result["mesh_pos"] = (
        mbatch_graph_node.pos[:, :]
        .to("cpu")
        .unsqueeze(0)
        .repeat(rollout_time_length, 1, 1)
        .numpy()
    )
    result["cells"] = (
        mbatch_graph_node.face.T[:, :]
        .to("cpu")
        .unsqueeze(0)
        .repeat(rollout_time_length, 1, 1)
        .numpy()
    )
    result["cells_face"] = (
        mbatch_graph_edge.face.T[:, :]
        .to("cpu")
        .unsqueeze(0)
        .repeat(rollout_time_length, 1, 1)
        .numpy()
    )
    result["face_node"] = (
        mbatch_graph_node.edge_index.T[:, :]
        .to("cpu")
        .unsqueeze(0)
        .repeat(rollout_time_length, 1, 1)
        .numpy()
    )
    result["node_type"] = (
        mbatch_graph_node.node_type[:, :]
        .unsqueeze(0)
        .repeat(rollout_time_length, 1, 1)
        .to("cpu")
        .numpy()
    )

    # only UVP are cell centered variables
    result["cell_type"] = (
        cell_type[:]
        .to("cpu")
        .unsqueeze(0)
        .unsqueeze(2)
        .repeat(rollout_time_length, 1, 1)
        .numpy()
    )
    result["velocity"] = predicted_cell_UV[:, :, :].to("cpu").numpy()
    result["pressure"] = predicted_cell_P[:, :, :].to("cpu").numpy()
    result["face_length"] = face_length[:, :, :].to("cpu").numpy()
    result["target_cylinder|pressure"] = (
        target_cylinder_boundary_edge_P[:, :, :].to("cpu").numpy()
    )
    result["target_cylinder|velocity"] = (
        target_cylinder_boundary_edge_UV[:, :, :].to("cpu").numpy()
    )
    result["centroid"] = graph_old.centroid[:, :].to("cpu").numpy()
    result["cell_area"] = graph_old.cell_area[:, :].to("cpu").numpy()
    result["target|UVP"] = targets_UVP_on_cell[:, :, :].to("cpu").numpy()

    saving_path = (
        result_dir
        + "/"
        + "rollout_index"
        + str(rollout_index)
        + "/re_"
        + str(re_num.cpu().numpy())
        + "_mu"
        + str(mu)
        + "_"
        + str(rollout_time_length)
        + "steps"
        + "MSE_"
        + str(average_cumulative_mse_uv.mean().item())
        + ".dat"
    )
    os.makedirs(os.path.split(saving_path)[0], exist_ok=True)

    if saving_tec is not None:
        if graph_index in saving_tec:
            write_tec.write_tecplot_ascii_cell_centered(
                raw_data=result,
                saving_path=saving_path,
                save_tec=True,
            )
        else:
            write_tec.write_tecplot_ascii_cell_centered(
                raw_data=result,
                saving_path=saving_path,
                save_tec=False,
            )
    else:
        write_tec.write_tecplot_ascii_cell_centered(
            raw_data=result,
            saving_path=saving_path,
            save_tec=False,
        )

    return scalars


if __name__ == "__main__":

    torch.manual_seed(0)

    # configurate parameters
    def str2bool(v):
        """
        'boolean type variable' for add_argument
        """
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("boolean value expected.")

    parser = argparse.ArgumentParser(description="Implementation of MeshGraphNets")
    parser.add_argument("--gpu", type=int, default=1, help="gpu number: 0 or 1")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="/lvm_data/litianyu/mycode-new/GEP-FVGN/Logger/net GN-Cell; hs 128; mu 0.001; rho 1; dt 0.01;/FVGN-result-collected/Hybriddatset-Re=200-2500-SiLU-global-mean-sum-mp=2-train-length=400--plot-rect--plot-cylinder/states/14240.state",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/home/litianyu/mycode/repos-py/FVM/my_FVNN/rollouts/tecplot/",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="test batch size at once forward"
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--rollout_num", type=int, default=100)
    parser.add_argument(
        "--sample_index",
        nargs="*",
        type=int,
        default=None,
        help="testing samples index",
    )
    parser.add_argument(
        "--write_tec_index",
        nargs="*",
        type=int,
        default=1,
        help="testing samples index",
    )

    args = parser.parse_args()
    params = get_param.params(os.path.split(args.model_dir)[0], parser)

    if args.sample_index is not None:
        args.rollout_num = len(args.sample_index)

    torch.cuda.set_per_process_memory_fraction(0.99, 0)
    # gpu devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_one_hot = params.edge_one_hot
    cell_one_hot = params.cell_one_hot

    # initialize flow parameters
    rollout_time_length = 599
    rho = params.rho
    mu = params.mu
    dt = params.dt
    params.train_traj_length = rollout_time_length - 3
    print("fluid parameters rho:{0} mu:{1} dt:{2}".format(rho, mu, dt))

    # initialize Logger and load model / optimizer if according parameters were given
    logger = Logger(
        get_hyperparam(params), use_csv=False, use_tensorboard=False, copy_code=False
    )
    params.load_index = 0 if params.load_index is None else params.load_index

    start = time.time()
    # validation_datasets = Load_mesh.Data_Pool(params=params,is_traning=False,device=device)
    datasets = Load_mesh.Data_Pool(params=params, is_traning=False, device=device)
    datasets._set_status(is_training=False)
    datasets.load_mesh_to_cpu(mode="h5", split=args.split)
    datasets._set_dataset("full")

    # datasets._set_dataset(is_full=True)
    end = time.time()
    noise_std = 2e-2
    print("traj has been loaded time consuming:{0}".format(end - start))
    rollout_start_step = 0

    simulator = FVGN(device=device, model_dir=args.model_dir, params=params)

    simulator.load_checkpoint(device=device, is_traning=False)
    fluid_model = simulator.to(device)
    fluid_model.eval()

    result_dir = os.path.split(args.model_dir)[0] + "/" + args.split
    os.makedirs(result_dir, exist_ok=True)

    nepochs = os.path.split(args.model_dir)[1].split(".")[0]
    dates = params.git_commit_dates.replace(":", "_").replace("+", "_")

    result_dir = result_dir + "/epochs_" + nepochs

    rollout_results = {
        "uv_mse": [],
        "p_mse": [],
        "u_rmse": [],
        "v_rmse": [],
        "uv_rmse": [],
        "p_rmse": [],
    }

    last_time = time.time()
    roll_outs_MSE_it = []
    roll_outs_edge_MSE_it = []
    roll_outs_RMSE_it = []
    roll_outs_p_MSE_it = []
    roll_outs_p_RMSE_it = []
    result_to_file = []
    batch_size = args.batch_size
    all_rollout_index = np.arange(args.rollout_num)

    Total_MSE = {}
    Total_Relative_MSE = {}
    if args.sample_index is None:
        if args.rollout_num <= 1:
            front_batch_index = all_rollout_index.reshape(1, 1)
        elif args.rollout_num % batch_size == 0:
            all_rollout_index = all_rollout_index.reshape(-1, batch_size)
            front_batch_index = all_rollout_index.tolist()
        else:
            front_batch_index = (
                all_rollout_index[
                    0 : (args.rollout_num - args.rollout_num % batch_size)
                ]
                .reshape(-1, batch_size)
                .tolist()
            )
            front_batch_index.append(
                all_rollout_index[(args.rollout_num - args.rollout_num % batch_size) :]
            )
    else:
        front_batch_index = [[sample_index] for sample_index in args.sample_index]

    for batch_index in front_batch_index:

        write_tecplot_index = args.write_tec_index

        scalar_data = rollout(
            model=fluid_model,
            datasets=datasets,
            batch_index=batch_index,
            result_dir=result_dir,
            rollout_time_length=rollout_time_length,
            rollout_start_step=rollout_start_step,
            rho=rho,
            mu=mu,
            dt=dt,
            dual_edge=params.dual_edge,
            saving_tec=write_tecplot_index,
            plot_boundary=True,
        )

        os.makedirs(
            os.path.join(result_dir + "/" + "rollout_index" + str(batch_index[0])),
            exist_ok=True,
        )

        """write error to file"""
        # 使用 f-string 完整构建文件路径
        file_name = f"UV_RMSE({np.mean(scalar_data['uv_rmse']):2e}).csv"
        file_path = f"{result_dir}/rollout_index{str(batch_index[0])}/{file_name}"

        # 使用 pickle 保存数据
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            for key, value in scalar_data.items():

                # 写入标题
                writer.writerow([f"total_{key}"])
                # 写入数据行
                writer.writerow([np.mean(value)])

                rollout_results[key].append(scalar_data[key])

            # 写入标题
            writer.writerow(scalar_data.keys())
            # 写入数据行
            writer.writerow(scalar_data.values())

        gc.collect()
        torch.cuda.empty_cache()

    """>>>>>>>>>plot error stastics line>>>>>>>>>"""
    print("-----ALL ROLLOUT DONE-----")
    final_file_name = (
        f"UV_RMSE({np.mean(np.stack(rollout_results['uv_rmse'], axis=0)):2e}).csv"
    )
    final_file_path = f"{result_dir}/{args.split}_{final_file_name}.csv"
    with open(final_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        for key, value in rollout_results.items():

            total_frames_error = np.stack(value, axis=0)

            print(f"total_{key}: {np.mean(total_frames_error)}")

            writer.writerow([f"total_{key}"])

            writer.writerow([np.mean(total_frames_error)])

            # 绘图
            plt.figure(figsize=(10, 5))  # 设置图形大小，这个大小需要根据实际需求调整
            mean_error = np.mean(total_frames_error, axis=0)
            rollout_results[key] = mean_error
            plt.plot(
                mean_error, label=key, color="red", linewidth=2
            )  # 指定线颜色和宽度

            # 设置标题和轴标签
            # plt.title('Error over Simulation Rollout Steps', fontsize=14)
            plt.xlabel("Simulation Rollout Step", fontsize=12)
            plt.ylabel("Relative-MSE", fontsize=12)

            # 设置图例
            plt.legend(loc="upper left", frameon=False)  # 关闭图例边框

            # 设置网格
            plt.grid(True, linestyle="--", alpha=0.5)  # 网格线样式

            # 设置轴的范围和刻度标签的字体大小
            plt.xlim(0, len(mean_error))  # 根据数据的长度设置 x 轴范围
            plt.ylim(
                0, np.max(mean_error) * 1.1
            )  # y 轴稍微高于最大误差值，留出一些空间
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # 显示并保存图形
            plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
            # plt.show()

            # 保存图像
            output_path = os.path.join(f"{result_dir}/{args.split}_{key}.png")
            plt.savefig(output_path, dpi=300)
            plt.close()  # 关闭当前窗口

        # 写入标题
        writer.writerow(rollout_results.keys())
        # 写入数据行
        writer.writerow(rollout_results.values())
