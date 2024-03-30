import json
import os
import sys
sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])

import torch
import numpy as np
import pickle
import enum

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import pandas as pd
from circle_fit import hyper_fit
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from scipy.signal import find_peaks

# 先设置全局字体为Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9
    REINFORCED_BOUNDARY = 10


def triangles_to_faces(faces, deform=False):
    """Computes mesh edges from triangles."""
    if not deform:
        # collect edges from triangles
        edges = torch.cat(
            (
                faces[:, 0:2],
                faces[:, 1:3],
                torch.stack((faces[:, 2], faces[:, 0]), dim=1),
            ),
            dim=0,
        )
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(
            packed_edges, return_inverse=False, return_counts=False, dim=0
        )
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (
            torch.cat((senders, receivers), dim=0),
            torch.cat((receivers, senders), dim=0),
        )
        unique_edges = torch.stack((receivers, senders), dim=1)
        return {
            "two_way_connectivity": two_way_connectivity,
            "senders": senders,
            "receivers": receivers,
            "unique_edges": unique_edges,
        }
    else:
        edges = torch.cat(
            (
                faces[:, 0:2],
                faces[:, 1:3],
                faces[:, 2:4],
                torch.stack((faces[:, 3], faces[:, 0]), dim=1),
            ),
            dim=0,
        )
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(
            packed_edges, return_inverse=False, return_counts=False, dim=0
        )
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (
            torch.cat((senders, receivers), dim=0),
            torch.cat((receivers, senders), dim=0),
        )
        return {
            "two_way_connectivity": two_way_connectivity,
            "senders": senders,
            "receivers": receivers,
        }


def formatnp(data):
    """
    Generate appropriate format string for numpy array

    Argument:
        - data: a list of numpy array
    """

    dataForm = []
    for i in range(len(data)):
        if np.issubsctype(data[i], np.integer):
            dataForm.append(" {:d}".format(data[i]))
        else:
            dataForm.append(" {:e}".format(data[i]))
        if (i + 1) % 3 == 0:
            dataForm.append("\n")
        if i == (len(data) - 1) and (i + 1) % 3 > 0:
            dataForm.append("\n")
    return " ".join(dataForm)


def formatnp_c(data):
    """
    Generate appropriate format string for numpy array

    Argument:
        - data: a list of numpy array
    """

    dataForm = []
    for i in range(len(data)):
        if np.issubsctype(data[i], np.integer):
            dataForm.append(" {:d}".format(data[i]))
        else:
            dataForm.append(" {:e}".format(data[i]))
        if (i + 1) % 3 == 0:
            dataForm.append("\n")
    return " ".join(dataForm)


def formatnp_f(data):
    """
    Generate appropriate format string for numpy array

    Argument:
        - data: a list of numpy array
    """

    dataForm = []
    for i in range(len(data)):
        if np.issubsctype(data[i], np.integer):
            dataForm.append(" {:d}".format(data[i]))
        else:
            dataForm.append(" {:e}".format(data[i]))
        if (i + 1) % 2 == 0:
            dataForm.append("\n")
    return " ".join(dataForm)


def write_cell_index(Cells, writer):
    for index in range(Cells.shape[0]):
        writer.write(formatnp_c(Cells[index]))


def write_face_index(faces, writer):
    for index in range(faces.shape[0]):
        writer.write(formatnp_f(faces[index]))


def write_tecplotzone(
    filename="FVGN.dat",
    datasets=None,
    time_step_length=100,
    has_cell_centered=False,
):
    time_avg_start = 100

    # interior zone
    time_sum_velocity = np.zeros_like(datasets[0]["velocity"])[1, :, :]
    time_sum_pressure = np.zeros_like(datasets[0]["pressure"])[1, :, :]
    time_avg_velocity = np.zeros_like(datasets[0]["velocity"])
    time_avg_pressure = np.zeros_like(datasets[0]["pressure"])

    target_time_sum_velocity = np.zeros_like(datasets[0]["velocity"])[1, :, :]
    target_time_sum_pressure = np.zeros_like(datasets[0]["pressure"])[1, :, :]
    target_time_avg_velocity = np.zeros_like(datasets[0]["velocity"])
    target_time_avg_pressure = np.zeros_like(datasets[0]["pressure"])

    for i in range(time_avg_start, datasets[0]["velocity"].shape[0]):
        time_sum_velocity += datasets[0]["velocity"][i]
        time_avg_velocity[i] = time_sum_velocity / (i - time_avg_start + 1)
        time_sum_pressure += datasets[0]["pressure"][i]
        time_avg_pressure[i] = time_sum_pressure / (i - time_avg_start + 1)

        target_time_sum_velocity += datasets[0]["target|UVP"][i, :, 0:2]
        target_time_avg_velocity[i] = target_time_sum_velocity / (
            i - time_avg_start + 1
        )
        target_time_sum_pressure += datasets[0]["target|UVP"][i, :, 2:3]
        target_time_avg_pressure[i] = target_time_sum_pressure / (
            i - time_avg_start + 1
        )

    # boundary zone
    time_sum_velocity_b = np.zeros_like(datasets[1]["velocity"])[1, :, :]
    time_sum_pressure_b = np.zeros_like(datasets[1]["pressure"])[1, :, :]
    time_avg_velocity_b = np.zeros_like(datasets[1]["velocity"])
    time_avg_pressure_b = np.zeros_like(datasets[1]["pressure"])

    target_time_sum_velocity_b = np.zeros_like(datasets[1]["velocity"])[1, :, :]
    target_time_sum_pressure_b = np.zeros_like(datasets[1]["pressure"])[1, :, :]
    target_time_avg_velocity_b = np.zeros_like(datasets[1]["velocity"])
    target_time_avg_pressure_b = np.zeros_like(datasets[1]["pressure"])

    for i in range(time_avg_start, datasets[1]["velocity"].shape[0]):
        time_sum_velocity_b += datasets[1]["velocity"][i]
        time_avg_velocity_b[i] = time_sum_velocity_b / (i - time_avg_start + 1)
        time_sum_pressure_b += datasets[1]["pressure"][i]
        time_avg_pressure_b[i] = time_sum_pressure_b / (i - time_avg_start + 1)

        target_time_sum_velocity_b += datasets[1]["target|velocity"][i, :, 0:2]
        target_time_avg_velocity_b[i] = target_time_sum_velocity_b / (
            i - time_avg_start + 1
        )
        target_time_sum_pressure_b += datasets[1]["target|pressure"][i, :, 0:1]
        target_time_avg_pressure_b[i] = target_time_sum_pressure_b / (
            i - time_avg_start + 1
        )

    with open(filename, "w") as f:

        f.write('TITLE = "Visualization of the volumetric solution"\n')
        f.write(
            'VARIABLES = "X"\n"Y"\n"U"\n"V"\n"P"\n"avgU"\n"avgV"\n"avgP"\n"target|U"\n"target|V"\n"target|P"\n"target|avgU"\n"target|avgV"\n"target|avgP"\n'
        )
        for i in range(time_step_length):
            for zone in datasets:
                zonename = zone["zonename"]
                if zonename == "Fluid":
                    f.write('ZONE T="{0}"\n'.format(zonename))
                    X = zone["mesh_pos"][i, :, 0]
                    Y = zone["mesh_pos"][i, :, 1]
                    U = zone["velocity"][i, :, 0]
                    V = zone["velocity"][i, :, 1]
                    P = zone["pressure"][i, :, 0]
                    avgU = time_avg_velocity[i, :, 0]
                    avgV = time_avg_velocity[i, :, 1]
                    avgP = time_avg_pressure[i, :, 0]
                    target_U = zone["target|UVP"][i, :, 0]
                    target_V = zone["target|UVP"][i, :, 1]
                    target_P = zone["target|UVP"][i, :, 2]
                    avgtarget_U = target_time_avg_velocity[i, :, 0]
                    avgtarget_V = target_time_avg_velocity[i, :, 1]
                    avgtarget_P = target_time_avg_pressure[i, :, 0]
                    field = np.concatenate(
                        (
                            X,
                            Y,
                            U,
                            V,
                            P,
                            avgU,
                            avgV,
                            avgP,
                            target_U,
                            target_V,
                            target_P,
                            avgtarget_U,
                            avgtarget_V,
                            avgtarget_P,
                        ),
                        axis=0,
                    )
                    Cells = zone["cells"][i, :, :] + 1

                    f.write(" STRANDID=1, SOLUTIONTIME={0}\n".format(0.01 * i))
                    f.write(
                        f" Nodes={X.size}, Elements={Cells.shape[0]}, "
                        "ZONETYPE=FETRIANGLE\n"
                    )
                    f.write(" DATAPACKING=BLOCK\n")
                    if has_cell_centered:
                        f.write(
                            " VARLOCATION=([3,4,5,6,7,8,9,10,11,12,13,14]=CELLCENTERED)\n"
                        )
                    else:
                        f.write(" VARLOCATION=([3,4,5,6,7,8]=NODAL)\n")
                    f.write(
                        " DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n"
                    )
                    f.write(formatnp(field))
                    f.write(" ")
                    write_cell_index(Cells, f)

        print("saved tecplot file at " + filename)
        """
        for node, field in zip(x, fields):
            f.write(f'{node[0].item()}\t{node[1].item()}\t0.0\t'f'{field[0].item()}\t{field[1].item()}\t'f'{field[2].item()}\n')

        for elem in elemlist:
            f.write('\t'.join(str(x+1) for x in elem))
            #if len(elem) == 3:
                # repeat last vertex if triangle
            #    f.write(f'\t{elem[-1]+1}')
            f.write('\n')
        """


def write_tecplot_ascii_nodal(raw_data, is_tfrecord, pkl_path, saving_path):
    cylinder_pos = []
    cylinder_velocity = []
    cylinder_pressure = []
    cylinder_index = []
    cylinder = {}
    if is_tfrecord:
        dataset = raw_data
    else:
        with open(pkl_path, "rb") as fp:
            dataset = pickle.load(fp)
    for j in range(600):
        new_pos_dict = {}
        mesh_pos = dataset["mesh_pos"][j]
        coor_y = dataset["mesh_pos"][j, :, 1]
        mask_F = np.full(coor_y.shape, False)
        mask_T = np.full(coor_y.shape, True)
        node_type = dataset["node_type"][j, :, 0]
        mask_of_coor = np.where(
            (node_type == NodeType.WALL_BOUNDARY)
            & (coor_y > np.min(coor_y))
            & (coor_y < np.max(coor_y)),
            mask_T,
            mask_F,
        )
        mask_of_coor_index = np.argwhere(
            (node_type == NodeType.WALL_BOUNDARY)
            & (coor_y > np.min(coor_y))
            & (coor_y < np.max(coor_y))
        )
        cylinder_x = dataset["mesh_pos"][j, :, 0][mask_of_coor]
        cylinder_u = dataset["velocity"][j, :, 0][mask_of_coor]
        cylinder_y = coor_y[mask_of_coor]
        cylinder_v = dataset["velocity"][j, :, 1][mask_of_coor]
        cylinder_p = dataset["pressure"][j, :, 0][mask_of_coor]
        coor = np.stack((cylinder_x, cylinder_y), axis=-1)
        cylinder_speed = np.stack((cylinder_u, cylinder_v), axis=-1)

        cylinder_pos.append(coor)
        cylinder_velocity.append(cylinder_speed)
        cylinder_pressure.append(cylinder_p)

        for index in range(coor.shape[0]):
            new_pos_dict[str(coor[index])] = index
        cells_node = torch.from_numpy(dataset["cells"][j]).to(torch.int32)
        decomposed_cells = triangles_to_faces(cells_node)
        senders = decomposed_cells["senders"]
        receivers = decomposed_cells["receivers"]
        mask_F = np.full(senders.shape, False)
        mask_T = np.full(senders.shape, True)
        mask_index_s = np.isin(senders, mask_of_coor_index)
        mask_index_r = np.isin(receivers, mask_of_coor_index)

        mask_index_of_face = np.where((mask_index_s) & (mask_index_r), mask_T, mask_F)

        senders = senders[mask_index_of_face]
        receivers = receivers[mask_index_of_face]
        senders_f = []
        receivers_f = []
        for i in range(senders.shape[0]):
            senders_f.append(new_pos_dict[str(mesh_pos[senders[i]])])
            receivers_f.append(new_pos_dict[str(mesh_pos[receivers[i]])])
        cylinder_boundary_face = np.stack(
            (np.asarray(senders_f), np.asarray(receivers_f)), axis=-1
        )
        cylinder_index.append(cylinder_boundary_face)
    dataset["zonename"] = "Fluid"
    flow_zone = dataset
    cylinder["zonename"] = "Cylinder_Boundary"
    cylinder["mesh_pos"] = np.asarray(cylinder_pos)
    cylinder["velocity"] = np.asarray(cylinder_velocity)
    cylinder["pressure"] = np.expand_dims(np.asarray(cylinder_pressure), -1)
    cylinder["face"] = np.asarray(cylinder_index)
    tec_saving_path = saving_path
    write_tecplotzone(tec_saving_path, [flow_zone])


def rearrange_dict(zone):
    """transform dict to list, so pandas dataframe can handle it properly"""
    dict_list = []
    build = False
    for k, v in zone.items():
        if k == "zonename" or k == "mean_u" or k == "relonyds_num" or k == "cylinder_D":
            continue
        if v.shape[2] > 1:
            for j in range(v.shape[2]):
                for index in range(zone["mesh_pos"].shape[0]):
                    dict_new = {}
                    if (
                        k == "zonename"
                        or k == "mean_u"
                        or k == "relonyds_num"
                        or k == "cylinder_D"
                    ):
                        continue
                    elif k == "centroid" or k == "cell_area":
                        dict_new[k + str(j)] = v[0][:, j]
                    else:
                        dict_new[k + str(j)] = v[index][:, j]
                    if not build:
                        dict_list.append(dict_new)
                build = True
                for index in range(zone["mesh_pos"].shape[0]):
                    if (
                        k == "zonename"
                        or k == "mean_u"
                        or k == "relonyds_num"
                        or k == "cylinder_D"
                    ):
                        continue
                    elif k == "centroid" or k == "cell_area":
                        dict_list[index][k + str(j)] = v[0][:, j]
                    else:
                        dict_list[index][k + str(j)] = v[index][:, j]
        else:
            for index in range(zone["mesh_pos"].shape[0]):
                if (
                    k == "zonename"
                    or k == "mean_u"
                    or k == "relonyds_num"
                    or k == "cylinder_D"
                ):
                    continue
                elif k == "centroid" or k == "cell_area":
                    dict_list[index][k] = v[0][:, 0]
                else:
                    dict_list[index][k] = v[index][:, 0]
        build = True
    return dict_list


def write_tecplot_ascii_cell_centered(
    raw_data,
    saving_path,
    save_tec=False,
):

    # write interior zone and boundary zone to tecplot file
    raw_data["zonename"] = "Fluid"
    flow_zone = raw_data
    tec_saving_path = saving_path
    if save_tec:
        write_tecplotzone(
            filename=tec_saving_path,
            datasets=[flow_zone],
            time_step_length=raw_data["velocity"].shape[0],
            has_cell_centered=True,
        )

def calc_edge_unv(edge_index, mesh_pos, centroid):

    senders = torch.from_numpy(edge_index[:, 0])
    receivers = torch.from_numpy(edge_index[:, 1])
    # calculate unit norm vector
    # unv = torch.ones((edge_index.shape[0],2),dtype=torch.float32)
    pos_diff = torch.index_select(
        torch.from_numpy(mesh_pos), 0, senders
    ) - torch.index_select(torch.from_numpy(mesh_pos), 0, receivers)
    unv = torch.stack((-pos_diff[:, 1], pos_diff[:, 0]), dim=-1)
    face_length = torch.norm(pos_diff, dim=1)

    # unv[:,1] = -(pos_diff[:,0]/pos_diff[:,1])
    deinf = torch.tensor([0, 1], dtype=torch.float32).repeat(unv.shape[0], 1)
    unv = torch.where((torch.isinf(unv)), deinf, unv)
    unv = unv / torch.norm(unv, dim=1).view(-1, 1)
    face_center_pos = (
        torch.index_select(torch.from_numpy(mesh_pos), 0, senders)
        + torch.index_select(torch.from_numpy(mesh_pos), 0, receivers)
    ) / 2.0

    _, wingleft_index = torch.max(torch.from_numpy(mesh_pos)[:, 0:1], dim=0)
    _, winglright_index = torch.min(torch.from_numpy(mesh_pos)[:, 0:1], dim=0)

    cylinder_center = (
        (
            torch.from_numpy(mesh_pos)[wingleft_index]
            + torch.from_numpy(mesh_pos)[winglright_index]
        )
        / 2.0
    ).view(-1, 2)

    c_f = face_center_pos - cylinder_center
    edge_uv_times_ccv = c_f[:, 0] * unv[:, 0] + c_f[:, 1] * unv[:, 1]
    mask = torch.logical_or(
        (edge_uv_times_ccv) > 0, torch.full(edge_uv_times_ccv.shape, False)
    )
    unv = torch.where(torch.stack((mask, mask), dim=-1), unv, unv * (-1.0))

    return unv, face_length.view(-1, 1)


def polygon_area(coordinates):
    n = len(coordinates)
    if n < 3:
        return 0

    area = 0
    for i in range(n):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)

    return abs(area) / 2
