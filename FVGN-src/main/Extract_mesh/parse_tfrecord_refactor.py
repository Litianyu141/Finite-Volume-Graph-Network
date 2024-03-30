# 2 -*- encoding: utf-8 -*-
"""
@File    :   parse_tfrecord.py
@Author  :   litianyu 
@Version :   1.0
@Contact :   lty1040808318@163.com
"""
# 解析tfrecord数据
import os
import sys
current_file_path = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(current_file_path)
from utils.utilities import NodeType

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import functools
import json
import numpy as np
import h5py
import multiprocessing as mp
import time
import torch
from torch_geometric.data import Data
import numpy as np
from torch_scatter import scatter_add, scatter_mean

import matplotlib.pyplot as plt
from utils.write_tec import write_tecplot_ascii_nodal
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from matplotlib import animation

from circle_fit import hyper_fit

c_NORMAL_max = 0
c_OBSTACLE_max = 0
c_AIRFOIL_max = 0
c_HANDLE_max = 0
c_INFLOW_max = 0
c_OUTFLOW_max = 0
c_WALL_BOUNDARY_max = 0
c_SIZE_max = 0

c_NORMAL_min = 3000
c_OBSTACLE_min = 3000
c_AIRFOIL_min = 3000
c_HANDLE_min = 3000
c_INFLOW_min = 3000
c_OUTFLOW_min = 3000
c_WALL_BOUNDARY_min = 3000
c_SIZE_min = 3000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("done loading packges")
marker_cell_sp1 = []
marker_cell_sp2 = []

mask_of_mesh_pos_sp1 = np.arange(5233)
mask_of_mesh_pos_sp2 = np.arange(5233)

inverse_of_marker_cell_sp1 = np.arange(1)
inverse_of_marker_cell_sp2 = np.arange(1)

new_mesh_pos_iframe_sp1 = np.arange(1)
new_mesh_pos_iframe_sp2 = np.arange(1)

new_node_type_iframe_sp1 = np.arange(1)
new_node_type_iframe_sp2 = np.arange(1)

pos_dict_1 = {}
pos_dict_2 = {}
switch_of_redress = True

def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    """for index,frame in enumerate(ds):
      data = _parse(frame, meta)"""
    ds = ds.prefetch(1)
    return ds


def dividing_line(index, x):
    if index == 0:
        return x
    else:
        return 0.1 * index * x


def stastic_nodeface_type(frame):
    if len(frame.shape) > 1:
        flatten = frame[:, 0]
    else:
        flatten = frame
    c_NORMAL = flatten[flatten == NodeType.NORMAL].shape[0]
    c_OBSTACLE = flatten[flatten == NodeType.OBSTACLE].shape[0]
    c_AIRFOIL = flatten[flatten == NodeType.AIRFOIL].shape[0]
    c_HANDLE = flatten[flatten == NodeType.HANDLE].shape[0]
    c_INFLOW = flatten[flatten == NodeType.INFLOW].shape[0]
    c_OUTFLOW = flatten[flatten == NodeType.OUTFLOW].shape[0]
    c_WALL_BOUNDARY = flatten[flatten == NodeType.WALL_BOUNDARY].shape[0]
    c_SIZE = flatten[flatten == NodeType.SIZE].shape[0]
    c_GHOST_INFLOW = flatten[flatten == NodeType.GHOST_INFLOW].shape[0]
    c_GHOST_OUTFLOW = flatten[flatten == NodeType.GHOST_OUTFLOW].shape[0]
    c_GHOST_WALL_BOUNDARY = flatten[flatten == NodeType.GHOST_WALL].shape[0]
    c_GHOST_AIRFOIL = flatten[flatten == NodeType.GHOST_AIRFOIL].shape[0]
    # for i in range(flatten.shape[0]):
    #       if(flatten[i]==NodeType.NORMAL):
    #             c_NORMAL+=1
    #       elif(flatten[i]==NodeType.OBSTACLE):
    #             c_OBSTACLE+=1
    #       elif(flatten[i]==NodeType.AIRFOIL):
    #             c_AIRFOIL+=1
    #       elif(flatten[i]==NodeType.HANDLE):
    #             c_HANDLE+=1
    #       elif(flatten[i]==NodeType.INFLOW):
    #             c_INFLOW+=1
    #       elif(flatten[i]==NodeType.OUTFLOW):
    #             c_OUTFLOW+=1
    #       elif(flatten[i]==NodeType.WALL_BOUNDARY):
    #             c_WALL_BOUNDARY+=1
    #       elif(flatten[i]==NodeType.SIZE):
    #             c_SIZE+=1
    #       elif(flatten[i]==NodeType.GHOST_INFLOW):
    #             c_GHOST_INFLOW+=1
    #       elif(flatten[i]==NodeType.GHOST_OUTFLOW):
    #             c_GHOST_OUTFLOW+=1
    #       elif(flatten[i]==NodeType.GHOST_WALL):
    #             c_GHOST_WALL+=1
    #       elif(flatten[i]==NodeType.GHOST_AIRFOIL):
    #             c_GHOST_AIRFOIL+=1
    print(
        "NORMAL: {0} OBSTACLE: {1} AIRFOIL: {2} HANDLE: {3} INFLOW: {4} OUTFLOW: {5} WALL_BOUNDARY: {6} SIZE: {7} GHOST_INFLOW: {8} GHOST_OUTFLOW: {9} GHOST_WALL_BOUNDARY: {10} GHOST_AIRFOIL: {11}".format(
            c_NORMAL,
            c_OBSTACLE,
            c_AIRFOIL,
            c_HANDLE,
            c_INFLOW,
            c_OUTFLOW,
            c_WALL_BOUNDARY,
            c_SIZE,
            c_GHOST_INFLOW,
            c_GHOST_OUTFLOW,
            c_GHOST_WALL_BOUNDARY,
            c_GHOST_AIRFOIL,
        )
    )
    rtval = np.zeros(int(max(NodeType)) + 1).astype(np.int32)
    for node_type in enumerate(NodeType):
        try:
            rtval[node_type[0]] = eval(
                "c_" + str(node_type[1]).replace("NodeType.", "")
            )
        except:
            rtval[node_type[0]] = 0
    return rtval


def stastic(frame):
    flatten = frame[:, 0]
    global c_NORMAL_max
    global c_OBSTACLE_max
    global c_AIRFOIL_max
    global c_HANDLE_max
    global c_INFLOW_max
    global c_OUTFLOW_max
    global c_WALL_BOUNDARY_max
    global c_SIZE_max

    global c_NORMAL_min
    global c_OBSTACLE_min
    global c_AIRFOIL_min
    global c_HANDLE_min
    global c_INFLOW_min
    global c_OUTFLOW_min
    global c_WALL_BOUNDARY_min
    global c_SIZE_min

    c_NORMAL = 0
    c_OBSTACLE = 0
    c_AIRFOIL = 0
    c_HANDLE = 0
    c_INFLOW = 0
    c_OUTFLOW = 0
    c_WALL_BOUNDARY = 0
    c_SIZE = 0

    for i in range(flatten.shape[0]):
        if flatten[i] == NodeType.NORMAL:
            c_NORMAL += 1

        elif flatten[i] == NodeType.OBSTACLE:
            c_OBSTACLE += 1

        elif flatten[i] == NodeType.AIRFOIL:
            c_AIRFOIL += 1

        elif flatten[i] == NodeType.HANDLE:
            c_HANDLE += 1

        elif flatten[i] == NodeType.INFLOW:
            c_INFLOW += 1

        elif flatten[i] == NodeType.OUTFLOW:
            c_OUTFLOW += 1

        elif flatten[i] == NodeType.WALL_BOUNDARY:
            c_WALL_BOUNDARY += 1

        elif flatten[i] == NodeType.SIZE:
            c_SIZE += 1

    c_NORMAL_max = max(c_NORMAL_max, c_NORMAL)
    c_NORMAL_min = min(c_NORMAL_min, c_NORMAL)
    c_OBSTACLE_max = max(c_OBSTACLE_max, c_OBSTACLE)
    c_OBSTACLE_min = min(c_OBSTACLE_min, c_OBSTACLE)
    c_AIRFOIL_max = max(c_AIRFOIL_max, c_AIRFOIL)
    c_OBSTACLE_min = min(c_AIRFOIL_min, c_AIRFOIL)
    c_HANDLE_max = max(c_HANDLE_max, c_HANDLE)
    c_HANDLE_min = min(c_HANDLE_min, c_HANDLE)
    c_INFLOW_max = max(c_INFLOW_max, c_INFLOW)
    c_INFLOW_min = min(c_INFLOW_min, c_INFLOW)
    c_OUTFLOW_max = max(c_OUTFLOW_max, c_OUTFLOW)
    c_OUTFLOW_min = min(c_OUTFLOW_min, c_OUTFLOW)
    c_WALL_BOUNDARY_max = max(c_WALL_BOUNDARY_max, c_WALL_BOUNDARY)
    c_WALL_BOUNDARY_min = min(c_WALL_BOUNDARY_min, c_WALL_BOUNDARY)
    c_SIZE_max = max(c_SIZE_max, c_SIZE)
    c_SIZE_min = min(c_SIZE_min, c_SIZE)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(record, mode="airfoil"):

    feature = {}
    for key, value in record.items():
        feature[key] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value.tobytes()])
        )

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()

def write_tfrecord_one(tfrecord_path, records, mode):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        serialized = serialize_example(records, mode=mode)
        writer.write(serialized)


def write_tfrecord_one_with_writer(writer, records, mode):
    serialized = serialize_example(records, mode=mode)
    writer.write(serialized)


def write_tfrecord(tfrecord_path, records, np_index):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for index, frame in enumerate(records):
            serialized = serialize_example(frame)
            writer.write(serialized)
            print("process:{0} is writing traj:{1}".format(np_index, index))


def write_tfrecord_mp(tfrecord_path_1, tfrecord_path_2, records):
    procs = []
    npc = 0
    n_shard = 2
    for shard_id in range(n_shard):
        if shard_id == 0:
            args = (tfrecord_path_1, records[0], npc)
        elif shard_id == 1:
            args = (tfrecord_path_2, records[1], npc + 1)
        p = mp.Process(target=write_tfrecord, args=args)
        p.start()
        procs.append(p)

    for proc in procs:
        proc.join()


def find_pos(mesh_point, mesh_pos_sp1):
    for k in range(mesh_pos_sp1.shape[0]):
        if (mesh_pos_sp1[k] == mesh_point).all():
            print("found{}".format(k))
            return k
    return False


def convert_to_tensors(input_dict):
    # 遍历字典中的所有键
    for key in input_dict.keys():
        # 检查值的类型
        value = input_dict[key]
        if isinstance(value, np.ndarray):
            # 如果值是一个Numpy数组，使用torch.from_numpy进行转换
            input_dict[key] = torch.from_numpy(value)
        elif not isinstance(value, torch.Tensor):
            # 如果值不是一个PyTorch张量，使用torch.tensor进行转换
            input_dict[key] = torch.tensor(value)
        # 如果值已经是一个PyTorch张量，不进行任何操作

    # 返回已更新的字典
    return input_dict

def node_based_WLSQ(phi_node, mesh_pos, edge_index):
    '''
    phi_node: (N, k)
    mesh_pos: (N, 2)
    edge_index: (2,E) bidirectional
    '''
    # node contribute
    outdegree_node_index = edge_index[1]
    indegree_node_index = edge_index[0]

    mesh_pos_diff_on_edge = (
        mesh_pos[outdegree_node_index] - mesh_pos[indegree_node_index]
    ).unsqueeze(2)
    
    mesh_pos_diff_on_edge_T = mesh_pos_diff_on_edge.transpose(1, 2)
    weight_node_to_node = 1.0 / torch.norm(
        mesh_pos_diff_on_edge, dim=1, keepdim=True
    )
    
    left_on_edge = torch.matmul(
        mesh_pos_diff_on_edge * weight_node_to_node,
        mesh_pos_diff_on_edge_T * weight_node_to_node,
    )

    A_node_to_node = scatter_add(
        left_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )

    phi_diff_on_edge = (
        weight_node_to_node
        * (
            (
                phi_node[outdegree_node_index] - phi_node[indegree_node_index]
            ).unsqueeze(1)
        )
        * weight_node_to_node
        * mesh_pos_diff_on_edge
    )

    B_node_to_node = scatter_add(
        phi_diff_on_edge, indegree_node_index, dim=0, dim_size=mesh_pos.shape[0]
    )
    
    A_left = A_node_to_node
    
    B_right = B_node_to_node
    
    nabla_phi_node = []
    
    """ first method"""
    # nabla_phi_node_lst = torch.linalg.lstsq(
    #     A_node_to_node_x, B_phi_node_to_node_x
    # ).solution

    """ second method"""
    # nabla_phi_node_lst = torch.matmul(A_inv_node_to_node_x,B_phi_node_to_node_x)

    """ third method"""
    nabla_phi_node_lst = torch.linalg.solve(A_left, B_right)
    
    """ fourth method"""
    # nabla_phi_node_lst = torch.matmul(R_inv_Q_t,B_phi_node_to_node_x)
    
    nabla_phi_node = torch.cat(
        [nabla_phi_node_lst[:, :, i_dim] for i_dim in range(phi_node.shape[-1])],
        dim=-1,
    )
    

    return nabla_phi_node

# Lmain
def extract_mesh_state(
    dataset,
    tf_writer,
    index,
    origin_writer=None,
    solving_params=None,
    h5_writer=None,
    path=None,
):

    mesh = {}
    mesh["mesh_pos"] = np.expand_dims(dataset["mesh_pos"][0], axis=0)
    mesh["cells_node"] = np.expand_dims(np.sort(dataset["cells"][0], axis=1), axis=0)
    cells_node = torch.from_numpy(mesh["cells_node"][0]).to(torch.int32)

    """>>>compute centriod crds>>>"""
    mesh_pos = mesh["mesh_pos"][0]
    cells_node_double = mesh["cells_node"][0]
    dataset["centroid"] = np.expand_dims(
        (
            mesh_pos[cells_node_double[:, 0]]
            + mesh_pos[cells_node_double[:, 1]]
            + mesh_pos[cells_node_double[:, 2]]
        )
        / 3.0,
        axis=0,
    )
    mesh["centroid"] = dataset["centroid"]
    """<<<compute centriod crds<<<"""


    # compose face
    decomposed_cells = triangles_to_faces(cells_node, mesh_pos)
    face = decomposed_cells["face_with_bias"]
    senders = face[:, 0]
    receivers = face[:, 1]
    edge_with_bias = decomposed_cells["edge_with_bias"]
    mesh["face"] = face.T.numpy().astype(np.int32)

    # compute face length
    mesh["face_length"] = (
        torch.norm(
            torch.from_numpy(mesh_pos)[senders] - torch.from_numpy(mesh_pos)[receivers],
            dim=-1,
            keepdim=True,
        )
        .to(torch.float32)
        .numpy()
    )

    # check-out face_type
    face_type = np.zeros((mesh["face"].shape[1], 1), dtype=np.int32)
    a = torch.index_select(
        torch.from_numpy(dataset["node_type"][0]), 0, torch.from_numpy(mesh["face"][0])
    ).numpy()
    b = torch.index_select(
        torch.from_numpy(dataset["node_type"][0]), 0, torch.from_numpy(mesh["face"][1])
    ).numpy()
    face_center_pos = (
        torch.index_select(
            torch.from_numpy(mesh_pos), 0, torch.from_numpy(mesh["face"][0])
        ).numpy()
        + torch.index_select(
            torch.from_numpy(mesh_pos), 0, torch.from_numpy(mesh["face"][1])
        ).numpy()
    ) / 2.0

    mesh_pos = dataset["mesh_pos"][0]
    node_type = dataset["node_type"][0].reshape(-1)
    # print("After renumed data has node type:")
    # stastic_nodeface_type(node_type)

    if "compressible" in solving_params["name"]:
        face_type = torch.from_numpy(face_type)
        Airfoil = torch.full(face_type.shape, NodeType.AIRFOIL).to(torch.int32)
        Interior = torch.full(face_type.shape, NodeType.NORMAL).to(torch.int32)
        Inlet = torch.full(face_type.shape, NodeType.INFLOW).to(torch.int32)
        a = torch.from_numpy(a).view(-1)
        b = torch.from_numpy(b).view(-1)
        face_type[(a == b) & (a == NodeType.AIRFOIL) & (b == NodeType.AIRFOIL), :] = (
            Airfoil[(a == b) & (a == NodeType.AIRFOIL) & (b == NodeType.AIRFOIL), :]
        )
        face_type[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :] = (
            Interior[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :]
        )
        face_type[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :] = (
            Inlet[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :]
        )

    else:

        topwall = np.max(face_center_pos[:, 1])
        bottomwall = np.min(face_center_pos[:, 1])
        outlet = np.max(face_center_pos[:, 0])
        inlet = np.min(face_center_pos[:, 0])

        """ for more robustness """
        topwall_Upper_limit = topwall + 1e-4
        topwall_Lower_limit = topwall - 1e-4

        bottomwall_Upper_limit = bottomwall + 1e-4
        bottomwall_Lower_limit = bottomwall - 1e-4

        outlet_Upper_limit = outlet + 1e-4
        outlet_Lower_limit = outlet - 1e-4

        inlet_Upper_limit = inlet + 1e-4
        inlet_Lower_limit = inlet - 1e-4

        original_limit = [
            (topwall_Lower_limit, topwall_Upper_limit),
            (bottomwall_Lower_limit, bottomwall_Upper_limit),
            (outlet_Lower_limit, outlet_Upper_limit),
            (inlet_Lower_limit, inlet_Upper_limit),
        ]

        face_type = torch.from_numpy(face_type)
        WALL_BOUNDARY_t = torch.full(face_type.shape, NodeType.WALL_BOUNDARY).to(
            torch.int32
        )
        Interior = torch.full(face_type.shape, NodeType.NORMAL).to(torch.int32)
        Inlet = torch.full(face_type.shape, NodeType.INFLOW).to(torch.int32)
        Outlet = torch.full(face_type.shape, NodeType.OUTFLOW).to(torch.int32)
        a = torch.from_numpy(a).view(-1)
        b = torch.from_numpy(b).view(-1)

        """ Without considering the corner points """
        face_type[
            (a == b) & (a == NodeType.WALL_BOUNDARY) & (b == NodeType.WALL_BOUNDARY), :
        ] = WALL_BOUNDARY_t[
            (a == b) & (a == NodeType.WALL_BOUNDARY) & (b == NodeType.WALL_BOUNDARY), :
        ]
        face_type[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :] = (
            Inlet[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :]
        )
        face_type[(a == b) & (a == NodeType.OUTFLOW) & (b == NodeType.OUTFLOW), :] = (
            Outlet[(a == b) & (a == NodeType.OUTFLOW) & (b == NodeType.OUTFLOW), :]
        )
        face_type[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :] = (
            Interior[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :]
        )

        """ Use position relationship to regulate the corner points """
        face_type[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.INFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.INFLOW))
            )
            & (
                torch.from_numpy(
                    (
                        (face_center_pos[:, 0] < inlet_Upper_limit)
                        & (face_center_pos[:, 0] > inlet_Lower_limit)
                    )
                ).to(torch.bool)
            ),
            :,
        ] = Inlet[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.INFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.INFLOW))
            )
            & (
                torch.from_numpy(
                    (
                        (face_center_pos[:, 0] < inlet_Upper_limit)
                        & (face_center_pos[:, 0] > inlet_Lower_limit)
                    )
                ).to(torch.bool)
            ),
            :,
        ]

        face_type[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.OUTFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.OUTFLOW))
            )
            & (
                (
                    torch.from_numpy(
                        (face_center_pos[:, 0] < outlet_Upper_limit)
                        & (face_center_pos[:, 0] > outlet_Lower_limit)
                    )
                ).to(torch.bool)
            ),
            :,
        ] = Outlet[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.OUTFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.OUTFLOW))
            )
            & (
                (
                    torch.from_numpy(
                        (face_center_pos[:, 0] < outlet_Upper_limit)
                        & (face_center_pos[:, 0] > outlet_Lower_limit)
                    )
                ).to(torch.bool)
            ),
            :,
        ]

    mesh["face_type"] = face_type
    # print("After renumed data has face type:")
    # stastic_nodeface_type(face_type)

    # compute cell_face index and cells_type
    face_list = torch.from_numpy(mesh["face"]).transpose(0, 1).numpy()
    face_index = {}
    for i in range(face_list.shape[0]):
        face_index[str(face_list[i])] = i
    nodes_of_cell = torch.stack(torch.chunk(edge_with_bias, 3, 0), dim=1)

    nodes_of_cell = nodes_of_cell.numpy()
    edges_of_cell = np.ones(
        (nodes_of_cell.shape[0], nodes_of_cell.shape[1]), dtype=np.int32
    )
    cells_type = np.zeros((nodes_of_cell.shape[0], 1), dtype=np.int32)

    for i in range(nodes_of_cell.shape[0]):
        three_face_index = [
            face_index[str(nodes_of_cell[i][0])],
            face_index[str(nodes_of_cell[i][1])],
            face_index[str(nodes_of_cell[i][2])],
        ]
        three_face_type = [
            face_type[three_face_index[0]],
            face_type[three_face_index[1]],
            face_type[three_face_index[2]],
        ]
        INFLOW_t = 0
        WALL_BOUNDARY_t = 0
        OUTFLOW_t = 0
        AIRFOIL_t = 0
        NORMAL_t = 0
        for type in three_face_type:
            if type == NodeType.INFLOW:
                INFLOW_t += 1
            elif type == NodeType.WALL_BOUNDARY:
                WALL_BOUNDARY_t += 1
            elif type == NodeType.OUTFLOW:
                OUTFLOW_t += 1
            elif type == NodeType.AIRFOIL:
                AIRFOIL_t += 1
            else:
                NORMAL_t += 1

        if WALL_BOUNDARY_t > 0 and NORMAL_t > 0 and INFLOW_t == 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t == 0 and INFLOW_t > 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t == 0 and INFLOW_t == 0 and OUTFLOW_t > 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t > 0 and INFLOW_t > 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t > 0 and INFLOW_t == 0 and OUTFLOW_t > 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif AIRFOIL_t > 0 and NORMAL_t > 0 and INFLOW_t == 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.AIRFOIL

        elif INFLOW_t > 0 and NORMAL_t > 0 and WALL_BOUNDARY_t == 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.INFLOW

        elif OUTFLOW_t > 0 and NORMAL_t > 0 and WALL_BOUNDARY_t == 0 and INFLOW_t == 0:
            cells_type[i] = NodeType.OUTFLOW
        else:
            cells_type[i] = NodeType.NORMAL
        for j in range(3):
            single_face_index = face_index[str(nodes_of_cell[i][j])]
            edges_of_cell[i][j] = single_face_index
    mesh["cells_face"] = edges_of_cell
    mesh["cells_type"] = cells_type

    # unit normal vector
    pos_diff = torch.index_select(
        torch.from_numpy(mesh_pos), 0, senders
    ) - torch.index_select(torch.from_numpy(mesh_pos), 0, receivers)
    unv = torch.cat((-pos_diff[:, 1:2], pos_diff[:, 0:1]), dim=1)
    for i in range(unv.shape[0]):
        if torch.isinf(unv[i][1]):
            unv[i] = torch.tensor([0, 1], dtype=torch.float32)
    unv = unv / (torch.norm(unv, dim=1).view(-1, 1))

    # TODO:complete the normal vector calculation
    face_center_pos = (
        torch.index_select(
            torch.from_numpy(mesh_pos), 0, torch.from_numpy(mesh["face"][0])
        ).numpy()
        + torch.index_select(
            torch.from_numpy(mesh_pos), 0, torch.from_numpy(mesh["face"][1])
        ).numpy()
    ) / 2.0
    centroid = torch.from_numpy(mesh["centroid"][0])
    cells_face = torch.from_numpy(mesh["cells_face"]).T

    # calc cell version of unv, and make sure all unv point outside of the cell
    edge_unv_set = []
    for i in range(3):
        edge_1 = cells_face[i]
        edge_1_uv = torch.index_select(unv, 0, edge_1)
        edge_1_center_centroid_vec = torch.from_numpy(
            mesh["centroid"][0]
        ) - torch.index_select(torch.from_numpy(face_center_pos), 0, edge_1)
        edge_uv_times_ccv = (
            edge_1_uv[:, 0] * edge_1_center_centroid_vec[:, 0]
            + edge_1_uv[:, 1] * edge_1_center_centroid_vec[:, 1]
        )
        Edge_op = torch.logical_or(
            edge_uv_times_ccv > 0, torch.full(edge_uv_times_ccv.shape, False)
        )
        Edge_op = torch.stack((Edge_op, Edge_op), dim=-1)
        edge_1_unv = torch.where(Edge_op, edge_1_uv * (-1.0), edge_1_uv)
        edge_unv_set.append(edge_1_unv.unsqueeze(1))
    mesh["unit_norm_v"] = torch.cat(edge_unv_set, dim=1).numpy()

    # compute face`s neighbor cell index or in other words, G_v
    cell_dict = {}
    edge_index = np.expand_dims(mesh["face"], axis=0)

    count_1 = 0
    for i in range(nodes_of_cell.shape[0]):
        edge_1 = str(nodes_of_cell[i, 0])
        edge_2 = str(nodes_of_cell[i, 1])
        edge_3 = str(nodes_of_cell[i, 2])

        if edge_1 in cell_dict:
            cell_dict[edge_1] = [cell_dict[edge_1][0], np.asarray(i, dtype=np.int32)]
            count_1 += 1
        else:
            cell_dict[edge_1] = [np.asarray(i, dtype=np.int32)]

        if edge_2 in cell_dict:
            cell_dict[edge_2] = [cell_dict[edge_2][0], np.asarray(i, dtype=np.int32)]
            count_1 += 1
        else:
            cell_dict[edge_2] = [np.asarray(i, dtype=np.int32)]

        if edge_3 in cell_dict:
            cell_dict[edge_3] = [cell_dict[edge_3][0], np.asarray(i, dtype=np.int32)]
            count_1 += 1
        else:
            cell_dict[edge_3] = [np.asarray(i, dtype=np.int32)]

    edge_index_t = edge_index.transpose(0, 2, 1)
    neighbour_cell = np.zeros_like(edge_index_t)
    face_node_index = edge_index_t
    count = 0
    for i in range(edge_index_t.shape[1]):
        face_str = str(face_node_index[0, i, :])
        cell_index = cell_dict[face_str]
        if len(cell_index) > 1:
            neighbour_cell[0][i] = np.stack((cell_index[0], cell_index[1]), axis=0)
        else:
            neighbour_cell[0][i] = np.stack(
                (cell_index[0], cell_index[0]), axis=0
            )  # adding self-loop instead of ghost cell
            count += 1
            
    # plot_edge_direction(centroid,torch.from_numpy(neighbour_cell[0]))
    neighbour_cell_with_bias = reorder_face(
        centroid, torch.from_numpy(neighbour_cell[0]), plot=False
    )
    mesh["neighbour_cell"] = (
        neighbour_cell_with_bias.unsqueeze(0).numpy().transpose(0, 2, 1)
    )

    # compute cell attribute V_BIC and P_BIC and
    node_index_of_cell = torch.from_numpy(mesh["cells_node"][0]).transpose(1, 0)
    v_target = dataset["velocity"]
    p_target = dataset["pressure"]
    cell_node_dist = (
        torch.sum(
            (
                torch.index_select(
                    torch.from_numpy(dataset["mesh_pos"]), 1, node_index_of_cell[0]
                )[0:1, :, :]
                - centroid.view(1, -1, 2)
            )
            ** 2,
            dim=2,
        ).view(1, -1, 1)
        + torch.sum(
            (
                torch.index_select(
                    torch.from_numpy(dataset["mesh_pos"]), 1, node_index_of_cell[1]
                )[0:1, :, :]
                - centroid.view(1, -1, 2)
            )
            ** 2,
            dim=2,
        ).view(1, -1, 1)
        + torch.sum(
            (
                torch.index_select(
                    torch.from_numpy(dataset["mesh_pos"]), 1, node_index_of_cell[2]
                )[0:1, :, :]
                - centroid.view(1, -1, 2)
            )
            ** 2,
            dim=2,
        ).view(1, -1, 1)
    )
    cell_factor_1 = (
        torch.sum(
            (
                torch.index_select(
                    torch.from_numpy(dataset["mesh_pos"]), 1, node_index_of_cell[0]
                )[0:1, :, :]
                - centroid.view(1, -1, 2)
            )
            ** 2,
            dim=2,
        ).view(1, -1, 1)
        / cell_node_dist
    )
    cell_factor_2 = (
        torch.sum(
            (
                torch.index_select(
                    torch.from_numpy(dataset["mesh_pos"]), 1, node_index_of_cell[1]
                )[0:1, :, :]
                - centroid.view(1, -1, 2)
            )
            ** 2,
            dim=2,
        ).view(1, -1, 1)
        / cell_node_dist
    )
    cell_factor_3 = (
        torch.sum(
            (
                torch.index_select(
                    torch.from_numpy(dataset["mesh_pos"]), 1, node_index_of_cell[2]
                )[0:1, :, :]
                - centroid.view(1, -1, 2)
            )
            ** 2,
            dim=2,
        ).view(1, -1, 1)
        / cell_node_dist
    )
    
    mesh["cell_factor"] = torch.cat(
        (cell_factor_1, cell_factor_2, cell_factor_3), dim=2
    ).numpy()
    mesh["target|velocity_on_node"] = (
        v_target  # obviously, velocity with BC, IC is v_target[0]
    )
    mesh["target|pressure_on_node"] = (
        p_target  # obviously, velocity with BC, IC is v_pressure[1]
    )
    if "compressible" in solving_params["name"]:
        mesh["target|density"] = dataset["density"]

    # compute cell_area
    cells_face = torch.from_numpy(mesh["cells_face"]).T
    face_length = torch.from_numpy(mesh["face_length"])
    len_edge_1 = torch.index_select(face_length, 0, cells_face[0])
    len_edge_2 = torch.index_select(face_length, 0, cells_face[1])
    len_edge_3 = torch.index_select(face_length, 0, cells_face[2])
    p = (1.0 / 2.0) * (len_edge_1 + len_edge_2 + len_edge_3)
    cells_area = torch.sqrt(p * (p - len_edge_1) * (p - len_edge_2) * (p - len_edge_3))
    mesh["cells_area"] = cells_area.numpy()


    # set all attributes to dimension [1, N, ...] except for veloity pressure and density is [600,N,...]
    mesh["node_type"] = np.expand_dims(dataset["node_type"][0], axis=0)
    mesh["mesh_pos"] = np.expand_dims(mesh_pos, axis=0)
    mesh["face"] = np.expand_dims(mesh["face"], axis=0)
    mesh["face_type"] = np.expand_dims(mesh["face_type"], axis=0)
    mesh["face_length"] = np.expand_dims(mesh["face_length"], axis=0)
    mesh["cells_face"] = np.expand_dims(mesh["cells_face"], axis=0)
    mesh["cells_type"] = np.expand_dims(mesh["cells_type"], axis=0)
    mesh["cells_area"] = np.expand_dims(mesh["cells_area"], axis=0)
    mesh["unit_norm_v"] = np.expand_dims(mesh["unit_norm_v"], axis=0)

    mesh = cal_mean_u_and_cd(mesh, path)

    """ >>>         stastic_nodeface_type           >>>"""
    # print("After renumed data has cell type:")
    # stastic_nodeface_type(cells_type)
    """ <<<         stastic_nodeface_type           <<<"""

    """ >>>         plot boundary node pos           >>>"""
    # fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    # ax.cla()
    # ax.set_aspect('equal')
    # #bb_min = mesh['velocity'].min(axis=(0, 1))
    # #bb_max = mesh['velocity'].max(axis=(0, 1))
    # plt.scatter(mesh_pos[node_type==NodeType.NORMAL,0],mesh_pos[node_type==NodeType.NORMAL,1],c='red',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.WALL_BOUNDARY,0],mesh_pos[node_type==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.INFLOW,0],mesh_pos[node_type==NodeType.INFLOW,1],c='blue',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.OUTFLOW,0],mesh_pos[node_type==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=1.5,zorder=5)
    # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node)
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3,zorder=1)

    # plt.savefig(f"{path['base_dir']}/{path['case_number']}_node distribution.png")
    # plt.close()
    """ <<<         plot boundary node pos           <<<"""

    """ >>>         plot boundary face center pos           >>>"""
    # fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    # ax.cla()
    # ax.set_aspect('equal')
    # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node)
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.NORMAL,0],face_center_pos[face_type[:,0]==NodeType.NORMAL,1],c='red',linewidths=1,s=1)

    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.WALL_BOUNDARY,0],face_center_pos[face_type[:,0]==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=1)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.INFLOW,0],face_center_pos[face_type[:,0]==NodeType.INFLOW,1],c='blue',linewidths=1,s=1)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.OUTFLOW,0],face_center_pos[face_type[:,0]==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=1)
    # plt.show()
    # plt.savefig(f"{path['base_dir']}/{path['case_number']}_face distribution.png")
    # plt.close()
    """ <<<         plot boundary face center pos           <<<"""

    """ >>>         plot boundary cell center pos           >>>"""
    # centroid = mesh["centroid"][0]

    # if (len(cells_type.shape) > 1) and (len(cells_type.shape) < 3):
    #     cells_type = cells_type.reshape(-1)
    # else:
    #     raise ValueError("chk cells_type dim")

    # fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    # ax.cla()
    # ax.set_aspect('equal')
    # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node)
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    # plt.scatter(centroid[cells_type==NodeType.NORMAL,0],centroid[cells_type==NodeType.NORMAL,1],c='red',linewidths=1,s=1)
    # plt.scatter(centroid[cells_type==NodeType.WALL_BOUNDARY,0],centroid[cells_type==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=1)
    # plt.scatter(centroid[cells_type==NodeType.OUTFLOW,0],centroid[cells_type==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=1)
    # plt.scatter(centroid[cells_type==NodeType.INFLOW,0],centroid[cells_type==NodeType.INFLOW,1],c='blue',linewidths=1,s=1)
    # plt.savefig(f"{path['base_dir']}/{path['case_number']}_cell center distribution.png")
    # plt.close()
    """ <<<         plot boundary cell center pos           <<<"""

    if path["saving_h5"]:
        current_traj = h5_writer.create_group(str(index))
        for key, value in mesh.items():
            current_traj.create_dataset(key, data=value)
    print(f'{index}th mesh has been extracted, path is {path["mesh_file_path"]}')

    return True


def analyze_value(value):
    if isinstance(value, np.ndarray):
        value_shape = value.shape
        adptive_value_shape = []
        for shape_idx, shape_num in enumerate(value_shape):
            if shape_idx == 0:
                # first dim should be time steps, skip it.
                adptive_value_shape.append(shape_num)
                continue
            if shape_num > 50:
                # other mesh-wise dim should be adptive shape
                adptive_value_shape.append(-1)
            else:
                adptive_value_shape.append(shape_num)

        return {
            "type": "dynamic" if value.shape[0] > 1 else "static",
            "shape": adptive_value_shape,
            "dtype": str(value.dtype),
        }

    elif isinstance(value, list):
        return value  # For this example, we assume lists are kept as-is.
    else:
        return value


def write_dict_info_to_json(input_dict, output_file):
    info_dict = {}
    trajectory_length = 1

    for key, value in input_dict.items():
        if isinstance(value, dict):
            info_dict[key] = {k: analyze_value(v) for k, v in value.items()}
            for subkey, subvalue in info_dict[key].items():
                if isinstance(subvalue, dict) and subvalue.get("type") == "dynamic":
                    trajectory_length = subvalue["shape"][0]
        else:
            info_dict[key] = analyze_value(value)

    # Add additional fields
    if "features" in info_dict:
        info_dict["field_names"] = list(info_dict["features"].keys())

    # Add the trajectory_length
    info_dict["trajectory_length"] = trajectory_length

    with open(output_file, "w") as file:
        json.dump(info_dict, file, indent=2)


def find_max_distance(points):
    # 获取点的数量
    n_points = points.size(0)

    # 初始化最大距离为0
    max_distance = 0

    # 遍历每一对点
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # 计算两点之间的欧几里得距离
            distance = torch.norm(points[i] - points[j])

            # 更新最大距离
            max_distance = max(max_distance, distance)

    # 返回最大距离
    return max_distance


def cal_mean_u_and_cd(trajectory,path):
    target_on_node = torch.cat((torch.from_numpy(trajectory['target|velocity_on_node'][0]),torch.from_numpy(trajectory['target|pressure_on_node'][0])),dim=1)
    edge_index = torch.from_numpy(trajectory['face'][0])
    target_on_edge = (torch.index_select(target_on_node,0,edge_index[0])+torch.index_select(target_on_node,0,edge_index[1]))/2.
    face_type = torch.from_numpy(trajectory['face_type'][0]).view(-1)
    node_type = torch.from_numpy(trajectory['node_type'][0]).view(-1)
    Inlet = target_on_edge[face_type==NodeType.INFLOW][:,0]
    face_length = torch.from_numpy(trajectory['face_length'])[0][:,0][face_type==NodeType.INFLOW]
    total_u = torch.sum(Inlet*face_length)
    ghosted_mesh_pos = torch.from_numpy(trajectory['mesh_pos'][0])
    mesh_pos = torch.from_numpy(trajectory['mesh_pos'][0])[(node_type!=NodeType.GHOST_INFLOW)&(node_type!=NodeType.GHOST_OUTFLOW)&(node_type!=NodeType.GHOST_WALL)]
    top = torch.max(mesh_pos[:,1]).numpy()
    bottom = torch.min(mesh_pos[:,1]).numpy()
    mean_u = (total_u/(top-bottom)).to(torch.float32).numpy()
    
    # compute aoa
    angles_rad = torch.atan2(target_on_edge[face_type==NodeType.INFLOW][:, 1], target_on_edge[face_type==NodeType.INFLOW][:, 0])

    # 将弧度转换为度
    angles_deg = torch.rad2deg(angles_rad)

    # 计算平均攻角
    average_angle = torch.mean(angles_deg)
    trajectory['aoa'] = average_angle.numpy()
  
    boundary_pos = ghosted_mesh_pos[node_type==NodeType.WALL_BOUNDARY].numpy()
    cylinder_mask = torch.full((boundary_pos.shape[0],1),True).view(-1).numpy()
    cylinder_not_mask = np.logical_not(cylinder_mask)
    cylinder_mask = np.where(((boundary_pos[:,1]>bottom)&(boundary_pos[:,1]<top)),cylinder_mask,cylinder_not_mask)
    cylinder_pos = torch.from_numpy(boundary_pos[cylinder_mask])
    
    # 计算所有点对之间的距离
    distances = torch.cdist(cylinder_pos, cylinder_pos, p=2)

    # 将自身比较的距离设置为0，以便忽略
    distances.fill_diagonal_(0)

    # 找到最长距离
    max_distance = distances.max()
    L0 = max_distance.numpy()
    
    rho = np.array(path['rho'])
    mu = np.array(path['mu'])
    
    trajectory['mean_u'] = mean_u
    trajectory['charac_scale'] = L0
    trajectory['rho'] = rho
    trajectory['rho'] = mu
    trajectory['reynolds_num'] = (mean_u*L0)/mu
    
    return trajectory


def make_dim_less(trajectory, params=None):
    target_on_node = torch.cat(
        (
            torch.from_numpy(trajectory["target|velocity_on_node"][0]),
            torch.from_numpy(trajectory["target|pressure_on_node"][0]),
        ),
        dim=1,
    )
    edge_index = torch.from_numpy(trajectory["face"][0])
    target_on_edge = (
        torch.index_select(target_on_node, 0, edge_index[0])
        + torch.index_select(target_on_node, 0, edge_index[1])
    ) / 2.0
    face_type = torch.from_numpy(trajectory["face_type"][0]).view(-1)
    node_type = torch.from_numpy(trajectory["node_type"][0]).view(-1)
    Inlet = target_on_edge[face_type == NodeType.INFLOW][:, 0]
    face_length = torch.from_numpy(trajectory["face_length"])[0][:, 0][
        face_type == NodeType.INFLOW
    ]
    total_u = torch.sum(Inlet * face_length)
    mesh_pos = torch.from_numpy(trajectory["mesh_pos"][0])
    top = torch.max(mesh_pos[:, 1]).numpy()
    bottom = torch.min(mesh_pos[:, 1]).numpy()
    mean_u = total_u / (top - bottom)

    boundary_pos = mesh_pos[node_type == NodeType.WALL_BOUNDARY].numpy()
    cylinder_mask = torch.full((boundary_pos.shape[0], 1), True).view(-1).numpy()
    cylinder_not_mask = np.logical_not(cylinder_mask)
    cylinder_mask = np.where(
        ((boundary_pos[:, 1] > bottom) & (boundary_pos[:, 1] < top)),
        cylinder_mask,
        cylinder_not_mask,
    )

    cylinder_pos = torch.from_numpy(boundary_pos[cylinder_mask])

    xc, yc, R, _ = hyper_fit(np.asarray(cylinder_pos))

    # R = torch.norm(cylinder_pos[0]-torch.tensor([xc,yc]))
    L0 = R * 2.0

    rho = 1

    trajectory["target|velocity_on_node"] = (
        (trajectory["target|velocity_on_node"]) / mean_u
    ).numpy()

    trajectory["target|pressure_on_node"] = (
        trajectory["target|pressure_on_node"] / ((mean_u**2) * (L0**2) * rho)
    ).numpy()

    trajectory["mean_u"] = mean_u.view(1, 1, 1).numpy()
    trajectory["cylinder_diameter"] = L0.view(1, 1, 1).numpy()
    return trajectory


def seprate_cells(mesh_pos, cells, node_type, density, pressure, velocity, index):
    global marker_cell_sp1
    global marker_cell_sp2
    global mask_of_mesh_pos_sp1
    global mask_of_mesh_pos_sp2
    global inverse_of_marker_cell_sp1
    global inverse_of_marker_cell_sp2
    global pos_dict_1
    global pos_dict_2
    global switch_of_redress
    global new_node_type_iframe_sp1
    global new_node_type_iframe_sp2
    global new_mesh_pos_iframe_sp1
    global new_mesh_pos_iframe_sp2

    marker_cell_sp1 = []
    marker_cell_sp2 = []

    # separate cells into two species and saved as marker cells
    for i in range(cells.shape[1]):
        cell = cells[0][i]
        cell = cell.tolist()
        member = 0
        for j in cell:
            x_cord = mesh_pos[0][j][0]
            y_cord = mesh_pos[0][j][1]
            if dividing_line(index, x_cord) >= 0:
                marker_cell_sp1.append(cell)
                member += 1
                break
        if member == 0:
            marker_cell_sp2.append(cell)
    marker_cell_sp1 = np.asarray(marker_cell_sp1, dtype=np.int32)
    marker_cell_sp2 = np.asarray(marker_cell_sp2, dtype=np.int32)

    # use mask to filter the mesh_pos of sp1 and sp2
    marker_cell_sp1_flat = marker_cell_sp1.reshape(
        (marker_cell_sp1.shape[0]) * (marker_cell_sp1.shape[1])
    )
    marker_cell_sp1_flat_uq = np.unique(marker_cell_sp1_flat)

    marker_cell_sp2_flat = marker_cell_sp2.reshape(
        (marker_cell_sp2.shape[0]) * (marker_cell_sp2.shape[1])
    )
    marker_cell_sp2_flat_uq = np.unique(marker_cell_sp2_flat)

    # mask filter of mesh_pos tensor
    inverse_of_marker_cell_sp1 = np.delete(
        mask_of_mesh_pos_sp1, marker_cell_sp1_flat_uq
    )
    inverse_of_marker_cell_sp2 = np.delete(
        mask_of_mesh_pos_sp2, marker_cell_sp2_flat_uq
    )

    # apply mask for mesh_pos first
    new_mesh_pos_iframe_sp1 = np.delete(mesh_pos[0], inverse_of_marker_cell_sp1, axis=0)
    new_mesh_pos_iframe_sp2 = np.delete(mesh_pos[0], inverse_of_marker_cell_sp2, axis=0)

    # redress the mesh_pos`s indexs in the marker_cells,because,np.delete would only delete the element charged by the index and reduce the length of splited mesh_pos_frame tensor,but the original mark_cells stores the original index of the mesh_pos tensor,so we need to redress the indexs
    count = 0
    for i in range(new_mesh_pos_iframe_sp1.shape[0]):
        pos_dict_1[str(new_mesh_pos_iframe_sp1[i])] = i

    for index in range(marker_cell_sp1.shape[0]):
        cell = marker_cell_sp1[index]
        for j in range(3):
            mesh_point = str(mesh_pos[0][cell[j]])
            if pos_dict_1.get(mesh_point, 10000) < 6000:
                marker_cell_sp1[index][j] = pos_dict_1[mesh_point]
            if marker_cell_sp1[index][j] > new_mesh_pos_iframe_sp1.shape[0]:
                count += 1
                print(
                    "有{0}个cell 里的 meshPoint的索引值超过了meshnodelist的长度".format(
                        count
                    )
                )

    for r in range(new_mesh_pos_iframe_sp2.shape[0]):
        pos_dict_2[str(new_mesh_pos_iframe_sp2[r])] = r

    count_1 = 0
    for index in range(marker_cell_sp2.shape[0]):
        cell = marker_cell_sp2[index]
        for j in range(3):
            mesh_point = str(mesh_pos[0][cell[j]])
            if pos_dict_2.get(mesh_point, 10000) < 6000:
                marker_cell_sp2[index][j] = pos_dict_2[mesh_point]
            if marker_cell_sp2[index][j] > new_mesh_pos_iframe_sp2.shape[0]:
                count_1 += 1
                print(
                    "有{0}个cell 里的 meshPoint的索引值超过了meshnodelist的长度".format(
                        count
                    )
                )

    new_node_type_iframe_sp1 = np.delete(
        node_type[0], inverse_of_marker_cell_sp1, axis=0
    )
    new_node_type_iframe_sp2 = np.delete(
        node_type[0], inverse_of_marker_cell_sp2, axis=0
    )

    new_velocity_sp1 = np.delete(velocity, inverse_of_marker_cell_sp1, axis=1)
    new_density_sp1 = np.delete(density, inverse_of_marker_cell_sp1, axis=1)
    new_pressure_sp1 = np.delete(pressure, inverse_of_marker_cell_sp1, axis=1)

    new_velocity_sp2 = np.delete(velocity, inverse_of_marker_cell_sp2, axis=1)
    new_density_sp2 = np.delete(density, inverse_of_marker_cell_sp2, axis=1)
    new_pressure_sp2 = np.delete(pressure, inverse_of_marker_cell_sp2, axis=1)

    # rearrange_frame with v_sp1 and reshape to 1 dim
    start_time = time.time()
    re_mesh_pos = np.asarray(
        [new_mesh_pos_iframe_sp1]
    )  # mesh is static, so store only once will be enough not 601
    re_cells = np.tile(marker_cell_sp1, (1, 1, 1))  # same as above
    re_node_type = np.asarray([new_node_type_iframe_sp1])  # same as above
    # rearrange_frame_1['node_type'] = re_node_type
    # rearrange_frame_1['cells'] = re_cells
    # rearrange_frame_1['mesh_pos'] = re_mesh_pos
    # rearrange_frame_1['density'] = new_density_sp1
    # rearrange_frame_1['pressure'] = new_pressure_sp1
    # rearrange_frame_1['velocity'] = new_velocity_sp1

    # re_mesh_pos = np.asarray([new_mesh_pos_iframe_sp2])  #mesh is static, so store only once will be enough not 601
    # re_cells = np.tile(marker_cell_sp2,(1,1,1))  #same as above
    # re_node_type = np.asarray([new_node_type_iframe_sp2])  #same as above
    # rearrange_frame_2['node_type'] = re_node_type
    # rearrange_frame_2['cells'] = re_cells
    # rearrange_frame_2['mesh_pos'] = re_mesh_pos
    # rearrange_frame_2['density'] = new_density_sp2
    # rearrange_frame_2['pressure'] = new_pressure_sp2
    # rearrange_frame_2['velocity'] = new_velocity_sp2

    # end_time = time.time()

    # return [rearrange_frame_1,rearrange_frame_2]


def transform_2(mesh_pos, cells, node_type, density, pressure, velocity, index):
    seprate_cells_result = seprate_cells(cells, mesh_pos)
    new_mesh_pos_sp1 = []
    new_mesh_pos_sp2 = []
    marker_cell_sp1 = np.asarray(seprate_cells_result[0], dtype=np.int32)
    marker_cell_sp2 = np.asarray(seprate_cells_result[0], dtype=np.int32)

    for i in range(marker_cell_sp1.shape[0]):
        cell = marker_cell_sp1[0][i]
        mesh_pos[0]


def parse_reshape(ds):
    rearrange_frame = {}
    re_mesh_pos = np.arange(1)
    re_node_type = np.arange(1)
    re_velocity = np.arange(1)
    re_cells = np.arange(1)
    re_density = np.arange(1)
    re_pressure = np.arange(1)
    count = 0
    for index, d in enumerate(ds):
        if count == 0:
            re_mesh_pos = np.expand_dims(d["mesh_pos"].numpy(), axis=0)
            re_node_type = np.expand_dims(d["node_type"].numpy(), axis=0)
            re_velocity = np.expand_dims(d["velocity"].numpy(), axis=0)
            re_cells = np.expand_dims(d["cells"].numpy(), axis=0)
            re_density = np.expand_dims(d["density"].numpy(), axis=0)
            re_pressure = np.expand_dims(d["pressure"].numpy(), axis=0)
            count += 1
            print("No.{0} has been added to the dict".format(index))
        else:
            re_mesh_pos = np.insert(re_mesh_pos, index, d["mesh_pos"].numpy(), axis=0)
            re_node_type = np.insert(
                re_node_type, index, d["node_type"].numpy(), axis=0
            )
            re_velocity = np.insert(re_velocity, index, d["velocity"].numpy(), axis=0)
            re_cells = np.insert(re_cells, index, d["cells"].numpy(), axis=0)
            re_density = np.insert(re_density, index, d["density"].numpy(), axis=0)
            re_pressure = np.insert(re_pressure, index, d["pressure"].numpy(), axis=0)
            print("No.{0} has been added to the dict".format(index))
    rearrange_frame["node_type"] = re_node_type
    rearrange_frame["cells"] = re_cells
    rearrange_frame["mesh_pos"] = re_mesh_pos
    rearrange_frame["density"] = re_density
    rearrange_frame["pressure"] = re_pressure
    rearrange_frame["velocity"] = re_velocity
    print("done")
    return rearrange_frame


def reorder_face(mesh_pos, edges, plot=False):

    senders = edges[:, 0]
    receivers = edges[:, 1]

    edge_vec = torch.index_select(mesh_pos, 0, senders) - torch.index_select(
        mesh_pos, 0, receivers
    )
    e_x = torch.cat(
        (torch.ones(edge_vec.shape[0], 1), (torch.zeros(edge_vec.shape[0], 1))), dim=1
    )

    edge_vec_dot_ex = edge_vec[:, 0] * e_x[:, 0] + edge_vec[:, 1] * e_x[:, 1]

    edge_op = torch.logical_or(
        edge_vec_dot_ex > 0, torch.full(edge_vec_dot_ex.shape, False)
    )
    edge_op = torch.stack((edge_op, edge_op), dim=-1)

    edge_op_1 = torch.logical_and(edge_vec[:, 0] == 0, edge_vec[:, 1] > 0)
    edge_op_1 = torch.stack((edge_op_1, edge_op_1), dim=-1)

    unique_edges = torch.stack((senders, receivers), dim=1)
    inverse_unique_edges = torch.stack((receivers, senders), dim=1)

    edge_with_bias = torch.where(
        ((edge_op) | (edge_op_1)), unique_edges, inverse_unique_edges
    )

    if plot:
        plot_edge_direction(mesh_pos, edge_with_bias)

    return edge_with_bias


def plot_edge_direction(mesh_pos, edges):

    senders = edges[:, 0]
    receivers = edges[:, 1]

    edge_vec = torch.index_select(mesh_pos, 0, senders) - torch.index_select(
        mesh_pos, 0, receivers
    )
    e_x = torch.cat(
        (torch.ones(edge_vec.shape[0], 1), (torch.zeros(edge_vec.shape[0], 1))), dim=1
    )
    e_y = torch.cat(
        (torch.zeros(edge_vec.shape[0], 1), (torch.ones(edge_vec.shape[0], 1))), dim=1
    )

    edge_vec_dot_ex = edge_vec[:, 0] * e_x[:, 0] + edge_vec[:, 1] * e_x[:, 1]
    edge_vec_dot_ey = edge_vec[:, 0] * e_y[:, 0] + edge_vec[:, 1] * e_y[:, 1]

    cosu = edge_vec_dot_ex / ((torch.norm(edge_vec, dim=1) * torch.norm(e_x, dim=1)))
    cosv = edge_vec_dot_ey / ((torch.norm(edge_vec, dim=1) * torch.norm(e_y, dim=1)))

    plt.quiver(
        torch.index_select(mesh_pos[:, 0:1], 0, senders),
        torch.index_select(mesh_pos[:, 1:2], 0, senders),
        edge_vec[:, 0],
        edge_vec[:, 1],
        units="height",
        scale=1.2,
        width=0.0025,
    )

    plt.show()


def triangles_to_faces(faces, mesh_pos, deform=False):
    """Computes mesh edges from triangles."""
    mesh_pos = torch.from_numpy(mesh_pos)
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
        unique_edges = torch.stack((senders, receivers), dim=1)

        # plot_edge_direction(mesh_pos,unique_edges)

        # face_with_bias = reorder_face(mesh_pos, unique_edges, plot=False)
        # edge_with_bias = reorder_face(mesh_pos, packed_edges, plot=False)
        
        return {
            "two_way_connectivity": two_way_connectivity,
            "senders": senders,
            "receivers": receivers,
            "unique_edges": unique_edges,
            "face_with_bias": unique_edges,
            "edge_with_bias": packed_edges,
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


# This function is compromised to Tobias Paffs`s datasets
def mask_face_bonudary(
    face_types, faces, velocity_on_node, pressure_on_node, is_train=False
):

    if is_train:

        velocity_on_face = (
            (
                torch.index_select(velocity_on_node, 0, faces[0])
                + torch.index_select(velocity_on_node, 0, faces[1])
            )
            / 2.0
        ).numpy()
        pressure_on_face = (
            (
                torch.index_select(pressure_on_node, 0, faces[0])
                + torch.index_select(pressure_on_node, 0, faces[1])
            )
            / 2.0
        ).numpy()

    else:

        velocity_on_face = (
            torch.index_select(
                torch.from_numpy(velocity_on_node), 1, torch.from_numpy(faces[0])
            )
            + torch.index_select(
                torch.from_numpy(velocity_on_node), 1, torch.from_numpy(faces[1])
            )
        ) / 2.0
        pressure_on_face = (
            torch.index_select(
                torch.from_numpy(pressure_on_node), 1, torch.from_numpy(faces[0])
            )
            + torch.index_select(
                torch.from_numpy(pressure_on_node), 1, torch.from_numpy(faces[1])
            )
        ) / 2.0
        """
    face_types = torch.from_numpy(face_types)
    mask_of_p = torch.zeros_like(pressure_on_face)
    mask_of_v = torch.zeros_like(velocity_on_face)
    pressure_on_face_t = torch.where((face_types==NodeType.OUTFLOW)|(face_types==NodeType.INFLOW),pressure_on_face,mask_of_p)
    face_types = face_types.repeat(1,1)
    velocity_on_face_t = torch.where((face_types==NodeType.OUTFLOW)|(face_types==NodeType.INFLOW),velocity_on_face,mask_of_v).repeat(1,3)
    """
    return torch.cat((velocity_on_face, pressure_on_face), dim=2).numpy()


def direction_bias(dataset):
    mesh_pos = dataset["mesh_pos"][0]
    edge_vec = dataset["face"]


def renum_data(dataset, unorder=True, index=0, plot=None):
    fig = plot[1]
    ax = plot[2]
    plot = plot[0]
    re_index = np.linspace(
        0, int(dataset["mesh_pos"].shape[1]) - 1, int(dataset["mesh_pos"].shape[1])
    ).astype(np.int64)
    re_cell_index = np.linspace(
        0, int(dataset["cells"].shape[1]) - 1, int(dataset["cells"].shape[1])
    ).astype(np.int64)
    key_list = []
    new_dataset = {}
    for key, value in dataset.items():
        dataset[key] = torch.from_numpy(value)
        key_list.append(key)

    new_dataset = {}
    cells_node = dataset["cells"][0]
    dataset["centroid"] = np.zeros((cells_node.shape[0], 2), dtype=np.float32)
    for index_c in range(cells_node.shape[0]):
        cell = cells_node[index_c]
        centroid_x = 0.0
        centroid_y = 0.0
        for j in range(3):
            centroid_x += dataset["mesh_pos"].numpy()[0][cell[j]][0]
            centroid_y += dataset["mesh_pos"].numpy()[0][cell[j]][1]
        dataset["centroid"][index_c] = np.array(
            [centroid_x / 3, centroid_y / 3], dtype=np.float32
        )
    dataset["centroid"] = torch.from_numpy(np.expand_dims(dataset["centroid"], axis=0))

    for key, value in dataset.items():
        dataset[key] = value.numpy()

    # if unorder:
    #   np.random.shuffle(re_index)
    #   np.random.shuffle(re_cell_index)
    #   for key in key_list:
    #     value = dataset[key]
    #     if key=='cells':
    #       # TODO: cells_node is not correct, need implementation
    #       new_dataset[key]=torch.index_select(value,1,torch.from_numpy(re_cell_index).to(torch.long))
    #     elif  key=='boundary':
    #       new_dataset[key]=value
    #     else:
    #       new_dataset[key] = torch.index_select(value,1,torch.from_numpy(re_index).to(torch.long))
    #   cell_renum_dict = {}
    #   new_cells = np.empty_like(dataset['cells'][0])
    #   for i in range(new_dataset['mesh_pos'].shape[1]):
    #     cell_renum_dict[str(new_dataset['mesh_pos'][0][i].numpy())] = i

    #   for j in range(dataset['cells'].shape[1]):
    #     cell = new_dataset['cells'][0][j]
    #     for node_index in range(cell.shape[0]):
    #       new_cells[j][node_index] = cell_renum_dict[str(dataset['mesh_pos'][0].numpy()[cell[node_index]])]
    #   new_cells = np.repeat(np.expand_dims(new_cells,axis=0),dataset['cells'].shape[0],axis=0 )
    #   new_dataset['cells'] = torch.from_numpy(new_cells)

    #   cells_node = new_dataset['cells'][0]
    #   mesh_pos = new_dataset['mesh_pos']
    #   new_dataset['centroid'] = ((torch.index_select(mesh_pos,1,cells_node[:,0])+torch.index_select(mesh_pos,1,cells_node[:,1])+torch.index_select(mesh_pos,1,cells_node[:,2]))/3.).view(1,-1,2)
    #   for key,value in new_dataset.items():
    #     dataset[key] = value.numpy()
    #     new_dataset[key] = value.numpy()

    # else:

    #   data_cell_centroid = dataset['centroid'].to(torch.float64)[0]
    #   data_cell_Z = -4*data_cell_centroid[:,0]**(2)+data_cell_centroid[:,0]+data_cell_centroid[:,1]+3*data_cell_centroid[:,0]*data_cell_centroid[:,1]-2*data_cell_centroid[:,1]**(2)+20000.
    #   data_node_pos = dataset['mesh_pos'].to(torch.float64)[0]
    #   data_Z = -4*data_node_pos[:,0]**(2)+data_node_pos[:,0]+data_node_pos[:,1]+3*data_node_pos[:,0]*data_node_pos[:,1]-2*data_node_pos[:,1]**(2)+20000.
    #   a = np.unique(data_Z.cpu().numpy(), return_counts=True)
    #   b = np.unique(data_cell_Z.cpu().numpy(), return_counts=True)
    #   if a[0].shape[0] !=data_Z.shape[0] or b[0].shape[0] !=data_cell_Z.shape[0]:
    #     data_cell_Z = data_cell_centroid
    #     data_Z = data_node_pos
    #     print("data{0} renum faild, please consider change the projection function".format(index))

    #   sorted_data_Z,new_data_index = torch.sort(data_Z,descending=False)
    #   sorted_data_cell_Z,new_data_cell_index = torch.sort(data_cell_Z,descending=False)
    #   for key in key_list:
    #     value = dataset[key]
    #     if key=='cells':
    #       new_dataset[key]=torch.index_select(value,1,new_data_cell_index)
    #     elif key=='boundary':
    #       new_dataset[key]=value
    #     else:
    #       new_dataset[key] = torch.index_select(value,1,new_data_index)
    #   cell_renum_dict = {}
    #   new_cells = np.empty_like(dataset['cells'][0])
    #   for i in range(new_dataset['mesh_pos'].shape[1]):
    #     cell_renum_dict[str(new_dataset['mesh_pos'][0][i].numpy())] = i
    #   for j in range(dataset['cells'].shape[1]):
    #     cell = dataset['cells'][0][j]
    #     for node_index in range(cell.shape[0]):
    #       new_cells[j][node_index] = cell_renum_dict[str(dataset['mesh_pos'][0].numpy()[cell[node_index]])]
    #   new_cells = np.repeat(np.expand_dims(new_cells,axis=0),dataset['cells'].shape[0],axis=0 )
    #   new_dataset['cells'] = torch.index_select(torch.from_numpy(new_cells),1,new_data_cell_index)

    #   cells_node = new_dataset['cells'][0]
    #   mesh_pos = new_dataset['mesh_pos'][0]
    #   new_dataset['centroid'] = ((torch.index_select(mesh_pos,0,cells_node[:,0])+torch.index_select(mesh_pos,0,cells_node[:,1])+torch.index_select(mesh_pos,0,cells_node[:,2]))/3.).view(1,-1,2)
    #   for key,value in new_dataset.items():
    #     dataset[key] = value.numpy()
    #     new_dataset[key] = value.numpy()
    #   #new_dataset = reorder_boundaryu_to_front(dataset)
    # if plot is not None and plot=='cell':
    #   # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #   ax.cla()
    #   ax.set_aspect('equal')
    #   #bb_min = mesh['velocity'].min(axis=(0, 1))
    #   #bb_max = mesh['velocity'].max(axis=(0, 1))
    #   mesh_pos = new_dataset['mesh_pos'][0]
    #   faces = new_dataset['cells'][0]
    #   triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #   #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    #   ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    #   #plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
    #   plt.show()
    # elif plot is not None and plot=='node':
    #   # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #   ax.cla()
    #   ax.set_aspect('equal')
    #   #bb_min = mesh['velocity'].min(axis=(0, 1))
    #   #bb_max = mesh['velocity'].max(axis=(0, 1))
    #   mesh_pos = new_dataset['mesh_pos'][0]
    #   #faces = dataset['cells'][0]
    #   #triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #   #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    #   #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    #   plt.scatter(mesh_pos[:,0],mesh_pos[:,1],c='red',linewidths=1)
    #   plt.show()

    # elif plot is not None and plot=='centroid':
    #   # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #   ax.cla()
    #   ax.set_aspect('equal')
    #   #bb_min = mesh['velocity'].min(axis=(0, 1))
    #   #bb_max = mesh['velocity'].max(axis=(0, 1))
    #   mesh_pos = new_dataset['centroid'][0]
    #   #faces = dataset['cells'][0]
    #   #triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
    #   #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    #   #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    #   plt.scatter(mesh_pos[:,0],mesh_pos[:,1],c='red',linewidths=1)
    #   plt.show()

    # elif plot is not None and plot=='plot_order':
    #   fig = plt.figure()  # 创建画布
    #   mesh_pos = new_dataset['mesh_pos'][0]
    #   centroid = new_dataset['centroid'][0]
    #   display_centroid_list=[centroid[0],centroid[1],centroid[2]]
    #   display_pos_list=[mesh_pos[0],mesh_pos[1],mesh_pos[2]]
    #   ax1 = fig.add_subplot(211)
    #   ax2 = fig.add_subplot(212)

    #   def animate(num):

    #     if num < mesh_pos.shape[0]:
    #       display_pos_list.append(mesh_pos[num])
    #     display_centroid_list.append(centroid[num])
    #     if num%3 ==0 and num >0:
    #         display_pos = np.array(display_pos_list)
    #         display_centroid = np.array(display_centroid_list)
    #         p1 = ax1.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
    #         ax1.legend(['node_pos'], loc=2, fontsize=10)
    #         p2 = ax2.scatter(display_centroid[:,0],display_centroid[:,1],c='green',linewidths=1)
    #         ax2.legend(['centriod'], loc=2, fontsize=10)
    #   ani = animation.FuncAnimation(fig, animate, frames=new_dataset['centroid'][0].shape[0], interval=100)
    #   if unorder:
    #     ani.save("unorder"+"test.gif", writer='pillow')
    #   else:
    #     ani.save("order"+"test.gif", writer='pillow')

    return dataset, True


def reorder_boundaryu_to_front(dataset, plot=None):

    boundary_attributes = {}

    node_type = torch.from_numpy(dataset["node_type"][0])[:, 0]
    face_type = torch.from_numpy(dataset["face_type"][0])[:, 0]
    cells_type = torch.from_numpy(dataset["cells_type"][0])[:, 0]

    node_mask_t = torch.full(node_type.shape, True)
    node_mask_i = torch.logical_not(node_mask_t)
    face_mask_t = torch.full(face_type.shape, True)
    face_mask_i = torch.logical_not(face_mask_t)
    cells_mask_t = torch.full(cells_type.shape, True)
    cells_mask_i = torch.logical_not(cells_mask_t)

    node_mask = torch.where(node_type == NodeType.NORMAL, node_mask_t, node_mask_i)
    face_mask = torch.where(face_type == NodeType.NORMAL, face_mask_t, face_mask_i)
    cells_mask = torch.where(cells_type == NodeType.NORMAL, cells_mask_t, cells_mask_i)

    boundary_node_mask = torch.logical_not(node_mask)
    boundary_face_mask = torch.logical_not(face_mask)
    boundary_cells_mask = torch.logical_not(cells_mask)

    """boundary attributes"""
    for key, value in dataset.items():
        if key == "mesh_pos":
            boundary_attributes = value[:, boundary_node_mask, :]
            Interior_attributes = value[:, node_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "target|velocity_on_node":
            boundary_attributes = value[:, boundary_node_mask, :]
            Interior_attributes = value[:, node_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "target|pressure_on_node":
            boundary_attributes = value[:, boundary_node_mask, :]
            Interior_attributes = value[:, node_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "node_type":
            boundary_attributes = value[:, boundary_node_mask, :]
            Interior_attributes = value[:, node_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "cells_node":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "centroid":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "face":
            boundary_attributes = value[:, :, boundary_face_mask]
            Interior_attributes = value[:, :, face_mask]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=2
            )
        elif key == "face_length":
            boundary_attributes = value[:, boundary_face_mask, :]
            Interior_attributes = value[:, face_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "face_type":
            boundary_attributes = value[:, boundary_face_mask, :]
            Interior_attributes = value[:, face_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "cells_face":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "cells_type":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "unit_norm_v":
            boundary_attributes = value[:, boundary_cells_mask, :, :]
            Interior_attributes = value[:, cells_mask, :, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "neighbour_cell":
            boundary_attributes = value[:, :, boundary_face_mask]
            Interior_attributes = value[:, :, face_mask]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=2
            )
        elif key == "cell_factor":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )
        elif key == "cells_area":
            boundary_attributes = value[:, boundary_cells_mask, :]
            Interior_attributes = value[:, cells_mask, :]
            dataset[key] = np.concatenate(
                (boundary_attributes, Interior_attributes), axis=1
            )

    if plot is not None and plot == "cell":
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.cla()
        ax.set_aspect("equal")
        # bb_min = mesh['velocity'].min(axis=(0, 1))
        # bb_max = mesh['velocity'].max(axis=(0, 1))
        mesh_pos = dataset["mesh_pos"][0]
        faces = dataset["cells"][0]
        triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1], faces)
        # ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
        ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
        # plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
        plt.show()
    elif plot is not None and plot == "node":
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.cla()
        ax.set_aspect("equal")
        # bb_min = mesh['velocity'].min(axis=(0, 1))
        # bb_max = mesh['velocity'].max(axis=(0, 1))
        mesh_pos = dataset["mesh_pos"][0]
        # faces = dataset['cells'][0]
        # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
        # ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
        # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        plt.scatter(mesh_pos[:, 0], mesh_pos[:, 1], c="red", linewidths=1)
        plt.show()

    elif plot is not None and plot == "centroid":
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.cla()
        ax.set_aspect("equal")
        # bb_min = mesh['velocity'].min(axis=(0, 1))
        # bb_max = mesh['velocity'].max(axis=(0, 1))
        mesh_pos = dataset["centroid"][0]
        # faces = dataset['cells'][0]
        # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],faces)
        # ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
        # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        plt.scatter(mesh_pos[:, 0], mesh_pos[:, 1], c="red", linewidths=1)
        plt.show()

    elif plot is not None and plot == "plot_order":
        fig = plt.figure()  # 创建画布
        mesh_pos = dataset["mesh_pos"][0]
        centroid = dataset["centroid"][0]
        display_centroid_list = [centroid[0], centroid[1], centroid[2]]
        display_pos_list = [mesh_pos[0], mesh_pos[1], mesh_pos[2]]
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        def animate(num):

            if num < mesh_pos.shape[0]:
                display_pos_list.append(mesh_pos[num])
            display_centroid_list.append(centroid[num])
            if num % 3 == 0 and num > 0:
                display_pos = np.array(display_pos_list)
                display_centroid = np.array(display_centroid_list)
                p1 = ax1.scatter(
                    display_pos[:, 0], display_pos[:, 1], c="red", linewidths=1
                )
                ax1.legend(["node_pos"], loc=2, fontsize=10)
                p2 = ax2.scatter(
                    display_centroid[:, 0],
                    display_centroid[:, 1],
                    c="green",
                    linewidths=1,
                )
                ax2.legend(["centriod"], loc=2, fontsize=10)

        ani = animation.FuncAnimation(
            fig, animate, frames=dataset["centroid"][0].shape[0], interval=100
        )
        plt.show(block=True)
        ani.save("order" + "test.gif", writer="pillow")
    return dataset


def calc_symmetry_pos(two_pos, vertex):
    """(x1,y1),(x2,y2)是两点式确定的直线,(x3,y3)是需要被计算的顶点"""
    x1 = two_pos[0][0]
    y1 = two_pos[0][1]
    x2 = two_pos[1][0]
    y2 = two_pos[1][1]
    x3 = vertex[0]
    y3 = vertex[1]
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - y1 * x2
    x4 = x3 - 2 * A * ((A * x3 + B * y3 + C) / (A * A + B * B))
    y4 = y3 - 2 * B * ((A * x3 + B * y3 + C) / (A * A + B * B))

    return (x4, y4)


def is_between(span: tuple, x):
    """
    Determine if a value x is between two other values a and b.

    Parameters:
    - a (float or int): The lower bound.
    - b (float or int): The upper bound.
    - x (float or int): The value to check.

    Returns:
    - (bool): True if x is between a and b (inclusive), False otherwise.
    """
    a, b = span
    # Check if x is between a and b, inclusive
    if a <= x <= b or b <= x <= a:
        return True
    else:
        return False


def whether_corner(node, pos_to_edge_index):
    """which can determine whether a vertex is a corner vertex in rectangular domain"""

    neighbour_edge_type = torch.cat(pos_to_edge_index[str(node.numpy())])
    stastic_face_type = stastic_nodeface_type(neighbour_edge_type)
    if stastic_face_type[NodeType.WALL_BOUNDARY] > 0 and (
        stastic_face_type[NodeType.INFLOW] > 0
        or stastic_face_type[NodeType.OUTFLOW] > 0
    ):
        return True
    else:
        return False

    # if rtval['inlet'] == 1 and rtval['outlet'] == 1 and rtval['topwall'] == 1 and rtval['bottomwall'] == 1:

    # if (is_between(inlet,x) and is_between(topwall,y)) or (is_between(outlet,x) and is_between(topwall,y)):
    #   return True
    # elif (is_between(inlet,x) and is_between(bottomwall,y)) or (is_between(outlet,x) and is_between(bottomwall,y)):
    #   return True
    # else:
    #   return False


def make_ghost_cell(
    fore_dataset,
    domain: list,
    mesh_pos: torch.Tensor,
    cells,
    node_type,
    mode,
    recover_unorder=False,
    limit=None,
    pos_to_edge_index=None,
    index=None,
    fig=None,
    ax=None,
):
    """compose ghost cell, but TODO: corner cell can not be duplicated twice with current cell type"""

    ghost_start_node_index = torch.max(cells)
    new_mesh_pos = mesh_pos.clone()
    new_cells_node = cells.clone()
    new_node_type = node_type.clone()
    for thread in domain:
        cell_thread = thread[0]
        boundary_cells_node = thread[0][0]
        boundary_cell_three_vertex_type = thread[0][1]
        cell_three_vertex_0 = [
            torch.index_select(mesh_pos, 0, boundary_cells_node[:, 0]),
            boundary_cell_three_vertex_type[:, 0],
        ]
        cell_three_vertex_1 = [
            torch.index_select(mesh_pos, 0, boundary_cells_node[:, 1]),
            boundary_cell_three_vertex_type[:, 1],
        ]
        cell_three_vertex_2 = [
            torch.index_select(mesh_pos, 0, boundary_cells_node[:, 2]),
            boundary_cell_three_vertex_type[:, 2],
        ]

        # for every boundary cell thread
        ghost_pos = []
        ghost_node_type = []
        ghost_cells_node = []
        for i in range(boundary_cells_node.shape[0]):
            if cell_thread[2] == "wall":
                if cell_three_vertex_0[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_1[0][i],
                                        cell_three_vertex_2[0][i],
                                    ],
                                    cell_three_vertex_0[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][1],
                                boundary_cells_node[i][2],
                            ]
                        )
                    )
                    ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))

                elif cell_three_vertex_1[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_0[0][i],
                                        cell_three_vertex_2[0][i],
                                    ],
                                    cell_three_vertex_1[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][0],
                                boundary_cells_node[i][2],
                            ]
                        )
                    )
                    ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))

                elif cell_three_vertex_2[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_1[0][i],
                                        cell_three_vertex_0[0][i],
                                    ],
                                    cell_three_vertex_2[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][1],
                                boundary_cells_node[i][0],
                            ]
                        )
                    )
                    ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))

                else:
                    if (
                        cell_three_vertex_0[1][i] == NodeType.INFLOW
                        or cell_three_vertex_0[1][i] == NodeType.OUTFLOW
                    ) and (
                        not (
                            whether_corner(cell_three_vertex_0[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_1[0][i],
                                            cell_three_vertex_2[0][i],
                                        ],
                                        cell_three_vertex_0[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][1],
                                    boundary_cells_node[i][2],
                                ]
                            )
                        )
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))
                    elif (
                        cell_three_vertex_1[1][i] == NodeType.INFLOW
                        or cell_three_vertex_1[1][i] == NodeType.OUTFLOW
                    ) and (
                        not (
                            whether_corner(cell_three_vertex_1[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_0[0][i],
                                            cell_three_vertex_2[0][i],
                                        ],
                                        cell_three_vertex_1[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][0],
                                    boundary_cells_node[i][2],
                                ]
                            )
                        )
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))
                    elif (
                        cell_three_vertex_2[1][i] == NodeType.INFLOW
                        or cell_three_vertex_2[1][i] == NodeType.OUTFLOW
                    ) and (
                        not (
                            whether_corner(cell_three_vertex_2[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_1[0][i],
                                            cell_three_vertex_0[0][i],
                                        ],
                                        cell_three_vertex_2[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][1],
                                    boundary_cells_node[i][0],
                                ]
                            )
                        )
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_WALL))

            else:
                """inlet thread and outlet thread"""
                if cell_three_vertex_0[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_1[0][i],
                                        cell_three_vertex_2[0][i],
                                    ],
                                    cell_three_vertex_0[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][1],
                                boundary_cells_node[i][2],
                            ]
                        )
                    )
                    if cell_thread[2] == "inlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                    elif cell_thread[2] == "outlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
                elif cell_three_vertex_1[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_0[0][i],
                                        cell_three_vertex_2[0][i],
                                    ],
                                    cell_three_vertex_1[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][0],
                                boundary_cells_node[i][2],
                            ]
                        )
                    )
                    if cell_thread[2] == "inlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                    elif cell_thread[2] == "outlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
                elif cell_three_vertex_2[1][i] == NodeType.NORMAL:
                    ghost_pos.append(
                        torch.from_numpy(
                            np.array(
                                calc_symmetry_pos(
                                    [
                                        cell_three_vertex_1[0][i],
                                        cell_three_vertex_0[0][i],
                                    ],
                                    cell_three_vertex_2[0][i],
                                )
                            )
                        )
                    )
                    ghost_start_node_index += 1
                    ghost_cells_node.append(
                        torch.stack(
                            [
                                ghost_start_node_index,
                                boundary_cells_node[i][1],
                                boundary_cells_node[i][0],
                            ]
                        )
                    )
                    if cell_thread[2] == "inlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                    elif cell_thread[2] == "outlet":
                        ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
                else:
                    if cell_three_vertex_0[1][i] == NodeType.WALL_BOUNDARY and (
                        not (
                            whether_corner(cell_three_vertex_0[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_1[0][i],
                                            cell_three_vertex_2[0][i],
                                        ],
                                        cell_three_vertex_0[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][1],
                                    boundary_cells_node[i][2],
                                ]
                            )
                        )
                        if cell_thread[2] == "inlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                        elif cell_thread[2] == "outlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
                    elif cell_three_vertex_1[1][i] == NodeType.WALL_BOUNDARY and (
                        not (
                            whether_corner(cell_three_vertex_1[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_0[0][i],
                                            cell_three_vertex_2[0][i],
                                        ],
                                        cell_three_vertex_1[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][0],
                                    boundary_cells_node[i][2],
                                ]
                            )
                        )
                        if cell_thread[2] == "inlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                        elif cell_thread[2] == "outlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))
                    elif cell_three_vertex_2[1][i] == NodeType.WALL_BOUNDARY and (
                        not (
                            whether_corner(cell_three_vertex_2[0][i], pos_to_edge_index)
                        )
                    ):
                        ghost_pos.append(
                            torch.from_numpy(
                                np.array(
                                    calc_symmetry_pos(
                                        [
                                            cell_three_vertex_1[0][i],
                                            cell_three_vertex_0[0][i],
                                        ],
                                        cell_three_vertex_2[0][i],
                                    )
                                )
                            )
                        )
                        ghost_start_node_index += 1
                        ghost_cells_node.append(
                            torch.stack(
                                [
                                    ghost_start_node_index,
                                    boundary_cells_node[i][1],
                                    boundary_cells_node[i][0],
                                ]
                            )
                        )
                        if cell_thread[2] == "inlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_INFLOW))
                        elif cell_thread[2] == "outlet":
                            ghost_node_type.append(torch.tensor(NodeType.GHOST_OUTFLOW))

        new_mesh_pos = torch.cat((new_mesh_pos, torch.stack(ghost_pos)), dim=0)
        new_cells_node = torch.cat(
            (new_cells_node, torch.stack(ghost_cells_node)), dim=0
        )
        new_node_type = torch.cat((new_node_type, torch.stack(ghost_node_type)), dim=0)

        new_velocity = torch.cat(
            (
                torch.from_numpy(fore_dataset["velocity"]),
                torch.zeros(
                    (new_mesh_pos.shape[0] - mesh_pos.shape[0], 2), dtype=torch.float32
                )
                .view(1, -1, 2)
                .repeat(fore_dataset["velocity"].shape[0], 1, 1),
            ),
            dim=1,
        )
        new_pressure = torch.cat(
            (
                torch.from_numpy(fore_dataset["pressure"]),
                torch.zeros(
                    (new_mesh_pos.shape[0] - mesh_pos.shape[0], 1), dtype=torch.float32
                )
                .view(1, -1, 1)
                .repeat(fore_dataset["pressure"].shape[0], 1, 1),
            ),
            dim=1,
        )

    """ >>>         plot ghosted boundary node pos           >>>"""
    # fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    # ax.cla()
    # ax.set_aspect('equal')
    # #bb_min = mesh['velocity'].min(axis=(0, 1))
    # #bb_max = mesh['velocity'].max(axis=(0, 1))
    # plt.scatter(new_mesh_pos[new_node_type==NodeType.NORMAL,0],new_mesh_pos[new_node_type==NodeType.NORMAL,1],c='red',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(new_mesh_pos[new_node_type==NodeType.WALL_BOUNDARY,0],new_mesh_pos[new_node_type==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(new_mesh_pos[new_node_type==NodeType.INFLOW,0],new_mesh_pos[new_node_type==NodeType.INFLOW,1],c='blue',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(new_mesh_pos[new_node_type==NodeType.OUTFLOW,0],new_mesh_pos[new_node_type==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=1.5,zorder=5)

    # plt.scatter(new_mesh_pos[new_node_type==NodeType.GHOST_WALL,0],new_mesh_pos[new_node_type==NodeType.GHOST_WALL,1],c='cyan',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(new_mesh_pos[new_node_type==NodeType.GHOST_INFLOW,0],new_mesh_pos[new_node_type==NodeType.GHOST_INFLOW,1],c='yellow',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(new_mesh_pos[new_node_type==NodeType.GHOST_OUTFLOW,0],new_mesh_pos[new_node_type==NodeType.GHOST_OUTFLOW,1],c='magenta',linewidths=1,s=1.5,zorder=5)

    # triang = mtri.Triangulation(new_mesh_pos[:, 0], new_mesh_pos[:, 1],new_cells_node)
    # #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3,zorder=1)
    # #plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
    # plt.savefig("ghosted boundary node pos.png")
    # plt.close()
    """ <<<         plot ghosted boundary node pos           <<<"""

    new_domain = {
        "mesh_pos": new_mesh_pos.view(1, -1, 2)
        .repeat(fore_dataset["mesh_pos"].shape[0], 1, 1)
        .numpy(),
        "cells": new_cells_node.view(1, -1, 3)
        .repeat(fore_dataset["cells"].shape[0], 1, 1)
        .numpy(),
        "node_type": new_node_type.view(1, -1, 1)
        .repeat(fore_dataset["node_type"].shape[0], 1, 1)
        .numpy(),
        "velocity": new_velocity.numpy(),
        "pressure": new_pressure.numpy(),
    }
    new_mesh, nodes_of_cell = recover_ghosted_2_fore_mesh(
        new_domain, mode, recover_unorder, limit, index, fig, ax
    )
    return new_mesh, new_domain, nodes_of_cell


def recover_ghosted_2_fore_mesh(
    ghosted_domain,
    mode="cylinder_mesh",
    unorder=False,
    limit=None,
    index=None,
    fig=None,
    ax=None,
):
    dataset = ghosted_domain
    """cell,node,centroid,plot_order"""
    dataset, rtvalue_renum = renum_data(
        dataset=dataset, unorder=unorder, index=0, plot=[None, fig, ax]
    )
    # import plot_tfrecord as pltf
    # pltf.plot_tfrecord_tmp(dataset)
    if not rtvalue_renum:
        return False
    mesh = {}
    mesh["mesh_pos"] = dataset["mesh_pos"][0]
    mesh["cells_node"] = np.sort(dataset["cells"][0], axis=1)
    cells_node = torch.from_numpy(mesh["cells_node"]).to(torch.int32)
    mesh["cells_node"] = np.expand_dims(cells_node, axis=0)

    """>>>computer centriod crds>>>"""
    # mesh['centroid'] = np.zeros((cells_node.shape[0],2),dtype = np.float32)
    # for index_c in range(cells_node.shape[0]):
    #         cell = cells_node[index_c]
    #         centroid_x = 0.0
    #         centroid_y = 0.0
    #         for j in range(3):
    #             centroid_x += mesh['mesh_pos'][cell[j]][0]
    #             centroid_y += mesh['mesh_pos'][cell[j]][1]
    #         mesh['centroid'][index_c] = np.array([centroid_x/3,centroid_y/3],dtype=np.float32)
    # mesh['centroid'] = np.expand_dims(mesh['centroid'],axis = 0)
    """<<<computer centriod crds<<<"""
    mesh["centroid"] = dataset["centroid"]

    # compose face
    decomposed_cells = triangles_to_faces(cells_node, mesh["mesh_pos"])
    face = decomposed_cells["face_with_bias"]
    senders = face[:, 0]
    receivers = face[:, 1]
    edge_with_bias = decomposed_cells["edge_with_bias"]
    mesh["face"] = face.T.numpy().astype(np.int32)

    # compute face length
    mesh["face_length"] = (
        torch.norm(
            torch.from_numpy(mesh_pos)[senders] - torch.from_numpy(mesh_pos)[receivers],
            dim=-1,
            keepdim=True,
        )
        .to(torch.float32)
        .numpy()
    )

    # check-out face_type
    face_type = np.zeros((mesh["face"].shape[1], 1), dtype=np.int32)
    a = torch.index_select(
        torch.from_numpy(dataset["node_type"][0]), 0, torch.from_numpy(mesh["face"][0])
    ).numpy()
    b = torch.index_select(
        torch.from_numpy(dataset["node_type"][0]), 0, torch.from_numpy(mesh["face"][1])
    ).numpy()
    face_center_pos = (
        torch.index_select(
            torch.from_numpy(mesh["mesh_pos"]), 0, torch.from_numpy(mesh["face"][0])
        ).numpy()
        + torch.index_select(
            torch.from_numpy(mesh["mesh_pos"]), 0, torch.from_numpy(mesh["face"][1])
        ).numpy()
    ) / 2.0

    mesh_pos = dataset["mesh_pos"][0]
    node_type = dataset["node_type"][0].reshape(-1)

    """ >>>         stastic_ghosted_nodeface_type           >>>"""
    # print("After recoverd ghosted data has node type:")
    # stastic_nodeface_type(node_type)
    """ <<<         stastic_ghosted_nodeface_type           <<<"""

    """ >>>         plot boundary node pos           >>>"""
    # fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    # ax.cla()
    # ax.set_aspect('equal')
    # #bb_min = mesh['velocity'].min(axis=(0, 1))
    # #bb_max = mesh['velocity'].max(axis=(0, 1))
    # plt.scatter(mesh_pos[node_type==NodeType.NORMAL,0],mesh_pos[node_type==NodeType.NORMAL,1],c='red',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.WALL_BOUNDARY,0],mesh_pos[node_type==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.INFLOW,0],mesh_pos[node_type==NodeType.INFLOW,1],c='blue',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.OUTFLOW,0],mesh_pos[node_type==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=1.5,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.GHOST_WALL,0],mesh_pos[node_type==NodeType.GHOST_WALL,1],c='cyan',linewidths=1,s=1,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.GHOST_OUTFLOW,0],mesh_pos[node_type==NodeType.GHOST_OUTFLOW,1],c='magenta',linewidths=1,s=1,zorder=5)
    # plt.scatter(mesh_pos[node_type==NodeType.GHOST_INFLOW,0],mesh_pos[node_type==NodeType.GHOST_INFLOW,1],c='teal',linewidths=1,s=1,zorder=5)
    # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node)
    # #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3,zorder=1)
    # #plt.scatter(display_pos[:,0],display_pos[:,1],c='red',linewidths=1)
    # plt.savefig("ghosted node distribution.png")
    # plt.close()
    """ <<<         plot boundary node pos           <<<"""

    if mode.find("airfoil") != -1:
        face_type = torch.from_numpy(face_type)
        Airfoil = torch.full(face_type.shape, NodeType.AIRFOIL).to(torch.int32)
        Interior = torch.full(face_type.shape, NodeType.NORMAL).to(torch.int32)
        Inlet = torch.full(face_type.shape, NodeType.INFLOW).to(torch.int32)
        ghost_Airfoil = torch.full(face_type.shape, NodeType.GHOST_AIRFOIL).to(
            torch.int32
        )
        ghost_Inlet = torch.full(face_type.shape, NodeType.GHOST_INFLOW).to(torch.int32)
        a = torch.from_numpy(a).view(-1)
        b = torch.from_numpy(b).view(-1)
        face_type[(a == b) & (a == NodeType.AIRFOIL) & (b == NodeType.AIRFOIL), :] = (
            Airfoil[(a == b) & (a == NodeType.AIRFOIL) & (b == NodeType.AIRFOIL), :]
        )
        face_type[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :] = (
            Interior[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :]
        )
        face_type[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :] = (
            Inlet[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :]
        )
        face_type[
            (a == b) & (a == NodeType.GHOST_INFLOW) & (b == NodeType.GHOST_INFLOW), :
        ] = ghost_Inlet[
            (a == b) & (a == NodeType.GHOST_INFLOW) & (b == NodeType.GHOST_INFLOW), :
        ]
        face_type[
            (a == b) & (a == NodeType.GHOST_AIRFOIL) & (b == NodeType.GHOST_AIRFOIL), :
        ] = ghost_Airfoil[
            (a == b) & (a == NodeType.GHOST_AIRFOIL) & (b == NodeType.GHOST_AIRFOIL), :
        ]

    else:

        # topwall = np.max(face_center_pos[:,1])
        # bottomwall = np.min(face_center_pos[:,1])
        # outlet = np.max(face_center_pos[:,0])
        # inlet = np.min(face_center_pos[:,0])

        """for more robustness"""
        topwall_Lower_limit, topwall_Upper_limit = limit[0]

        bottomwall_Lower_limit, bottomwall_Upper_limit = limit[1]

        outlet_Lower_limit, outlet_Upper_limit = limit[2]

        inlet_Lower_limit, inlet_Upper_limit = limit[3]

        face_type = torch.from_numpy(face_type)
        WALL_BOUNDARY_t = torch.full(face_type.shape, NodeType.WALL_BOUNDARY).to(
            torch.int32
        )
        Interior = torch.full(face_type.shape, NodeType.NORMAL).to(torch.int32)
        Inlet = torch.full(face_type.shape, NodeType.INFLOW).to(torch.int32)
        Outlet = torch.full(face_type.shape, NodeType.OUTFLOW).to(torch.int32)
        ghost_WALL_BOUNDARY_t = torch.full(face_type.shape, NodeType.GHOST_WALL).to(
            torch.int32
        )
        ghost_Inlet = torch.full(face_type.shape, NodeType.GHOST_INFLOW).to(torch.int32)
        ghost_Outlet = torch.full(face_type.shape, NodeType.GHOST_OUTFLOW).to(
            torch.int32
        )
        a = torch.from_numpy(a).view(-1)
        b = torch.from_numpy(b).view(-1)

        """ Without considering the corner points """
        face_type[
            (a == b) & (a == NodeType.WALL_BOUNDARY) & (b == NodeType.WALL_BOUNDARY), :
        ] = WALL_BOUNDARY_t[
            (a == b) & (a == NodeType.WALL_BOUNDARY) & (b == NodeType.WALL_BOUNDARY), :
        ]
        face_type[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :] = (
            Inlet[(a == b) & (a == NodeType.INFLOW) & (b == NodeType.INFLOW), :]
        )
        face_type[(a == b) & (a == NodeType.OUTFLOW) & (b == NodeType.OUTFLOW), :] = (
            Outlet[(a == b) & (a == NodeType.OUTFLOW) & (b == NodeType.OUTFLOW), :]
        )
        face_type[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :] = (
            Interior[(a == b) & (a == NodeType.NORMAL) & (b == NodeType.NORMAL), :]
        )

        face_type[(a == NodeType.GHOST_WALL) | (b == NodeType.GHOST_WALL), :] = (
            ghost_WALL_BOUNDARY_t[
                (a == NodeType.GHOST_WALL) | (b == NodeType.GHOST_WALL), :
            ]
        )
        face_type[(a == NodeType.GHOST_INFLOW) | (b == NodeType.GHOST_INFLOW), :] = (
            ghost_Inlet[(a == NodeType.GHOST_INFLOW) | (b == NodeType.GHOST_INFLOW), :]
        )
        face_type[(a == NodeType.GHOST_OUTFLOW) | (b == NodeType.GHOST_OUTFLOW), :] = (
            ghost_Outlet[
                (a == NodeType.GHOST_OUTFLOW) | (b == NodeType.GHOST_OUTFLOW), :
            ]
        )

        """ Use position relationship to regulate the corner points """
        face_type[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.INFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.INFLOW))
            )
            & (
                torch.from_numpy(
                    (
                        (face_center_pos[:, 0] < inlet_Upper_limit)
                        & (face_center_pos[:, 0] > inlet_Lower_limit)
                    )
                ).to(torch.bool)
            ),
            :,
        ] = Inlet[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.INFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.INFLOW))
            )
            & (
                torch.from_numpy(
                    (
                        (face_center_pos[:, 0] < inlet_Upper_limit)
                        & (face_center_pos[:, 0] > inlet_Lower_limit)
                    )
                ).to(torch.bool)
            ),
            :,
        ]

        face_type[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.OUTFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.OUTFLOW))
            )
            & (
                (
                    torch.from_numpy(
                        (face_center_pos[:, 0] < outlet_Upper_limit)
                        & (face_center_pos[:, 0] > outlet_Lower_limit)
                    )
                ).to(torch.bool)
            ),
            :,
        ] = Outlet[
            (
                ((a == NodeType.WALL_BOUNDARY) & (b == NodeType.OUTFLOW))
                | ((b == NodeType.WALL_BOUNDARY) & (a == NodeType.OUTFLOW))
            )
            & (
                (
                    torch.from_numpy(
                        (face_center_pos[:, 0] < outlet_Upper_limit)
                        & (face_center_pos[:, 0] > outlet_Lower_limit)
                    )
                ).to(torch.bool)
            ),
            :,
        ]

    mesh["face_type"] = face_type
    """ >>>         stastic_ghosted_nodeface_type           >>>"""
    # print("After recoverd ghosted data has face type:")
    # stastic_nodeface_type(face_type)
    """ <<<         stastic_ghosted_nodeface_type           <<<"""

    """ >>>         plot boundary face center pos           >>>"""
    # fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    # ax.cla()
    # ax.set_aspect('equal')
    # #triang = mtri.Triangulation(display_pos[:, 0], display_pos[:, 1])
    # #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    # #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.NORMAL,0],face_center_pos[face_type[:,0]==NodeType.NORMAL,1],c='red',linewidths=1,s=1,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.WALL_BOUNDARY,0],face_center_pos[face_type[:,0]==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=1,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.INFLOW,0],face_center_pos[face_type[:,0]==NodeType.INFLOW,1],c='blue',linewidths=1,s=1,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.OUTFLOW,0],face_center_pos[face_type[:,0]==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=1,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.GHOST_WALL,0],face_center_pos[face_type[:,0]==NodeType.GHOST_WALL,1],c='cyan',linewidths=1,s=1,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.GHOST_OUTFLOW,0],face_center_pos[face_type[:,0]==NodeType.GHOST_OUTFLOW,1],c='magenta',linewidths=1,s=1,zorder=5)
    # plt.scatter(face_center_pos[face_type[:,0]==NodeType.GHOST_INFLOW,0],face_center_pos[face_type[:,0]==NodeType.GHOST_INFLOW,1],c='teal',linewidths=1,s=1,zorder=5)
    # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node)
    # #ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3,zorder=1)
    # plt.savefig("ghosted face distribution.png")
    # plt.close()
    """ <<<         plot boundary face center pos           <<<"""

    # compute cell_face index and cells_type
    face_list = torch.from_numpy(mesh["face"]).transpose(0, 1).numpy()
    face_index = {}
    for i in range(face_list.shape[0]):
        face_index[str(face_list[i])] = i
    nodes_of_cell = torch.stack(torch.chunk(edge_with_bias, 3, 0), dim=1)

    nodes_of_cell = nodes_of_cell.numpy()
    edges_of_cell = np.ones(
        (nodes_of_cell.shape[0], nodes_of_cell.shape[1]), dtype=np.int32
    )
    cells_type = np.zeros((nodes_of_cell.shape[0], 1), dtype=np.int32)

    for i in range(nodes_of_cell.shape[0]):
        three_face_index = [
            face_index[str(nodes_of_cell[i][0])],
            face_index[str(nodes_of_cell[i][1])],
            face_index[str(nodes_of_cell[i][2])],
        ]
        three_face_type = [
            face_type[three_face_index[0]],
            face_type[three_face_index[1]],
            face_type[three_face_index[2]],
        ]
        INFLOW_t = 0
        WALL_BOUNDARY_t = 0
        OUTFLOW_t = 0
        AIRFOIL_t = 0
        NORMAL_t = 0
        ghost_INFLOW_t = 0
        ghost_WALL_BOUNDARY_t = 0
        ghost_OUTFLOW_t = 0
        ghost_AIRFOIL_t = 0
        for type in three_face_type:
            if type == NodeType.INFLOW:
                INFLOW_t += 1
            elif type == NodeType.WALL_BOUNDARY:
                WALL_BOUNDARY_t += 1
            elif type == NodeType.OUTFLOW:
                OUTFLOW_t += 1
            elif type == NodeType.AIRFOIL:
                AIRFOIL_t += 1
            elif type == NodeType.GHOST_INFLOW:
                ghost_INFLOW_t += 1
            elif type == NodeType.GHOST_WALL:
                ghost_WALL_BOUNDARY_t += 1
            elif type == NodeType.GHOST_OUTFLOW:
                ghost_OUTFLOW_t += 1
            elif type == NodeType.GHOST_AIRFOIL:
                ghost_AIRFOIL_t += 1
            else:
                NORMAL_t += 1
        if ghost_INFLOW_t > 0:
            cells_type[i] = NodeType.GHOST_INFLOW
        elif ghost_WALL_BOUNDARY_t > 0:
            cells_type[i] = NodeType.GHOST_WALL
        elif ghost_OUTFLOW_t > 0:
            cells_type[i] = NodeType.GHOST_OUTFLOW
        elif ghost_AIRFOIL_t > 0:
            cells_type[i] = NodeType.GHOST_AIRFOIL

        # elif INFLOW_t>0 and WALL_BOUNDARY_t>0 and NORMAL_t>0: # left top vertx corner boundary(both wall and inflow)
        #   cells_type[i] = NodeType.IN_WALL

        # elif OUTFLOW_t>0 and WALL_BOUNDARY_t>0 and NORMAL_t>0: # right bottom vertx corner boundary(both wall and outflow)
        #   cells_type[i] = NodeType.OUT_WALL

        elif WALL_BOUNDARY_t > 0 and NORMAL_t > 0 and INFLOW_t == 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t == 0 and INFLOW_t > 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t == 0 and INFLOW_t == 0 and OUTFLOW_t > 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t > 0 and INFLOW_t > 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif WALL_BOUNDARY_t > 0 and NORMAL_t > 0 and INFLOW_t == 0 and OUTFLOW_t > 0:
            cells_type[i] = NodeType.WALL_BOUNDARY

        elif AIRFOIL_t > 0 and NORMAL_t > 0 and INFLOW_t == 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.AIRFOIL

        elif INFLOW_t > 0 and NORMAL_t > 0 and WALL_BOUNDARY_t == 0 and OUTFLOW_t == 0:
            cells_type[i] = NodeType.INFLOW

        elif OUTFLOW_t > 0 and NORMAL_t > 0 and WALL_BOUNDARY_t == 0 and INFLOW_t == 0:
            cells_type[i] = NodeType.OUTFLOW
        else:
            cells_type[i] = NodeType.NORMAL
        for j in range(3):
            single_face_index = face_index[str(nodes_of_cell[i][j])]
            edges_of_cell[i][j] = single_face_index
    mesh["cells_face"] = edges_of_cell
    mesh["cells_type"] = cells_type

    """ >>>         stastic_ghosted_nodeface_type           >>>"""
    print("After recoverd ghosted data has cell type:")
    stastic_type = stastic_nodeface_type(cells_type)
    # if stastic_type[NodeType.INFLOW]+1!=stastic_type[NodeType.GHOST_INFLOW] or stastic_type[NodeType.OUTFLOW]!=stastic_type[NodeType.GHOST_OUTFLOW] or stastic_type[NodeType.WALL_BOUNDARY]!=stastic_type[NodeType.GHOST_WALL]:
    #   raise ValueError("check ghosted result, try to plot it")
    """ <<<         stastic_ghosted_nodeface_type           <<<"""

    """ >>>         plot ghosted boundary cell center pos           >>>"""
    # centroid = mesh['centroid'][0]
    # if (len(cells_type.shape)>1)and (len(cells_type.shape)<3):
    #   cells_type = cells_type.reshape(-1)
    # else:
    #   raise ValueError("chk cells_type dim")
    # fig, ax = plt.subplots(1, 1, figsize=(32, 18))
    # ax.cla()
    # ax.set_aspect('equal')
    # plt.scatter(centroid[cells_type==NodeType.NORMAL,0],centroid[cells_type==NodeType.NORMAL,1],c='red',linewidths=1,s=1,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.WALL_BOUNDARY,0],centroid[cells_type==NodeType.WALL_BOUNDARY,1],c='green',linewidths=1,s=1,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.OUTFLOW,0],centroid[cells_type==NodeType.OUTFLOW,1],c='orange',linewidths=1,s=1,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.INFLOW,0],centroid[cells_type==NodeType.INFLOW,1],c='blue',linewidths=1,s=1,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.GHOST_WALL,0],centroid[cells_type==NodeType.GHOST_WALL,1],c='cyan',linewidths=1,s=1,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.GHOST_OUTFLOW,0],centroid[cells_type==NodeType.GHOST_OUTFLOW,1],c='magenta',linewidths=1,s=1,zorder=5)
    # plt.scatter(centroid[cells_type==NodeType.GHOST_INFLOW,0],centroid[cells_type==NodeType.GHOST_INFLOW,1],c='teal',linewidths=1,s=1,zorder=5)
    # triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1],cells_node)
    # bb_min = dataset['velocity'].min(axis=(0, 1))
    # bb_max = dataset['velocity'].max(axis=(0, 1))
    # ax.tripcolor(triang, dataset['velocity'][500][:, 0], vmin=bb_min[0], vmax=bb_max[0])
    # ax.triplot(triang, 'ko-', ms=0.5, lw=0.3,zorder=1)
    # plt.savefig("ghosted cell center distribution"+str(index)+".png")
    # plt.close()
    """ <<<         plot ghosted boundary cell center pos           <<<"""

    return mesh, nodes_of_cell


def parse_origin_dataset(dataset, unorder=False, index_num=0, plot=None, writer=None):
    re_index = np.linspace(
        0, int(dataset["mesh_pos"].shape[1]) - 1, int(dataset["mesh_pos"].shape[1])
    ).astype(np.int64)
    re_cell_index = np.linspace(
        0, int(dataset["cells"].shape[1]) - 1, int(dataset["cells"].shape[1])
    ).astype(np.int64)
    key_list = []
    new_dataset = {}
    for key, value in dataset.items():
        dataset[key] = torch.from_numpy(value)
        key_list.append(key)

    new_dataset = {}
    cells_node = dataset["cells"][0]
    dataset["centroid"] = np.zeros((cells_node.shape[0], 2), dtype=np.float32)
    for index_c in range(cells_node.shape[0]):
        cell = cells_node[index_c]
        centroid_x = 0.0
        centroid_y = 0.0
        for j in range(3):
            centroid_x += dataset["mesh_pos"].numpy()[0][cell[j]][0]
            centroid_y += dataset["mesh_pos"].numpy()[0][cell[j]][1]
        dataset["centroid"][index_c] = np.array(
            [centroid_x / 3, centroid_y / 3], dtype=np.float32
        )
    dataset["centroid"] = torch.from_numpy(np.expand_dims(dataset["centroid"], axis=0))

    if unorder:
        np.random.shuffle(re_index)
        np.random.shuffle(re_cell_index)
        for key in key_list:
            value = dataset[key]
            if key == "cells":
                # TODO: cells_node is not correct, need implementation
                new_dataset[key] = torch.index_select(
                    value, 1, torch.from_numpy(re_cell_index).to(torch.long)
                )
            elif key == "boundary":
                new_dataset[key] = value
            else:
                new_dataset[key] = torch.index_select(
                    value, 1, torch.from_numpy(re_index).to(torch.long)
                )
        cell_renum_dict = {}
        new_cells = np.empty_like(dataset["cells"][0])
        for i in range(new_dataset["mesh_pos"].shape[1]):
            cell_renum_dict[str(new_dataset["mesh_pos"][0][i].numpy())] = i

        for j in range(dataset["cells"].shape[1]):
            cell = new_dataset["cells"][0][j]
            for node_index in range(cell.shape[0]):
                new_cells[j][node_index] = cell_renum_dict[
                    str(dataset["mesh_pos"][0].numpy()[cell[node_index]])
                ]
        new_cells = np.repeat(
            np.expand_dims(new_cells, axis=0), dataset["cells"].shape[0], axis=0
        )
        new_dataset["cells"] = torch.from_numpy(new_cells)

        cells_node = new_dataset["cells"][0]
        mesh_pos = new_dataset["mesh_pos"]
        new_dataset["centroid"] = (
            (
                torch.index_select(mesh_pos, 1, cells_node[:, 0])
                + torch.index_select(mesh_pos, 1, cells_node[:, 1])
                + torch.index_select(mesh_pos, 1, cells_node[:, 2])
            )
            / 3.0
        ).view(1, -1, 2)
        for key, value in new_dataset.items():
            dataset[key] = value.numpy()
            new_dataset[key] = value.numpy()

    else:

        data_cell_centroid = dataset["centroid"].to(torch.float64)[0]
        data_cell_Z = (
            -8 * data_cell_centroid[:, 0] ** (2)
            + 3 * data_cell_centroid[:, 0] * data_cell_centroid[:, 1]
            - 2 * data_cell_centroid[:, 1] ** (2)
            + 20.0
        )
        data_node_pos = dataset["mesh_pos"].to(torch.float64)[0]
        data_Z = (
            -8 * data_node_pos[:, 0] ** (2)
            + 3 * data_node_pos[:, 0] * data_node_pos[:, 1]
            - 2 * data_node_pos[:, 1] ** (2)
            + 20.0
        )
        a = np.unique(data_Z.cpu().numpy(), return_counts=True)
        b = np.unique(data_cell_Z.cpu().numpy(), return_counts=True)
        if a[0].shape[0] != data_Z.shape[0] or b[0].shape[0] != data_cell_Z.shape[0]:
            print("data renum faild{0}".format(index))
            return False
        else:
            sorted_data_Z, new_data_index = torch.sort(data_Z, descending=False)
            sorted_data_cell_Z, new_data_cell_index = torch.sort(
                data_cell_Z, descending=False
            )
            for key in key_list:
                value = dataset[key]
                if key == "cells":
                    new_dataset[key] = torch.index_select(value, 1, new_data_cell_index)
                elif key == "boundary":
                    new_dataset[key] = value
                else:
                    new_dataset[key] = torch.index_select(value, 1, new_data_index)
            cell_renum_dict = {}
            new_cells = np.empty_like(dataset["cells"][0])
            for i in range(new_dataset["mesh_pos"].shape[1]):
                cell_renum_dict[str(new_dataset["mesh_pos"][0][i].numpy())] = i
            for j in range(dataset["cells"].shape[1]):
                cell = dataset["cells"][0][j]
                for node_index in range(cell.shape[0]):
                    new_cells[j][node_index] = cell_renum_dict[
                        str(dataset["mesh_pos"][0].numpy()[cell[node_index]])
                    ]
            new_cells = np.repeat(
                np.expand_dims(new_cells, axis=0), dataset["cells"].shape[0], axis=0
            )
            new_dataset["cells"] = torch.index_select(
                torch.from_numpy(new_cells), 1, new_data_cell_index
            )

            cells_node = new_dataset["cells"][0]
            mesh_pos = new_dataset["mesh_pos"][0]
            new_dataset["centroid"] = (
                (
                    torch.index_select(mesh_pos, 0, cells_node[:, 0])
                    + torch.index_select(mesh_pos, 0, cells_node[:, 1])
                    + torch.index_select(mesh_pos, 0, cells_node[:, 2])
                )
                / 3.0
            ).view(1, -1, 2)
            for key, value in new_dataset.items():
                dataset[key] = value.numpy()
                new_dataset[key] = value.numpy()
            # new_dataset = reorder_boundaryu_to_front(dataset)
            new_dataset["cells"] = new_dataset["cells"][0:1, :, :]
            new_dataset["mesh_pos"] = new_dataset["mesh_pos"][0:1, :, :]
            new_dataset["node_type"] = new_dataset["node_type"][0:1, :, :]
            write_tfrecord_one_with_writer(writer, new_dataset, mode="cylinder_flow")
            print("origin datasets No.{} has been parsed mesh\n".format(index_num))


if __name__ == "__main__":
    tf.enable_resource_variables()
    tf.enable_eager_execution()

    # choose wether to transform whole datasets into h5 file
    path = {
        "tf_datasetPath": "/data/litianyu/dataset/MeshGN/cylinder_flow/origin_dataset",
        "h5_save_path": "H/data/litianyu/dataset/MeshGN/cylinder_flow/origin_dataset/h5/airfoil",
        "tec_save_path": "/data/litianyu/dataset/MeshGN/cylinder_flow/origin_dataset/cylinder_flow/meshs/",
        "saving_tec": False,
        "saving_h5": True,
    }
    
    solving_params = {"name": "incompressible",
                    "rho":1.,
                    "mu":0.001,
                    "dt":0.01}
    
    pickl_path = path["pickl_save_path"]
    tf_datasetPath = path["tf_datasetPath"]
    numofsd = 2
    os.makedirs(path["tf_datasetPath"], exist_ok=True)

    """set current work directory"""
    imgoutputdir = os.path.split(__file__)[0] + "/imgoutput"
    os.makedirs(imgoutputdir, exist_ok=True)
    current_file_dir = os.chdir(imgoutputdir)

    for split in ["valid", "train", "test"]:
        ds = load_dataset(tf_datasetPath, split)
        rearrange_frame_sp_1 = []
        rearrange_frame_sp_2 = []
        raw_data = {}
        tf_saving_mesh_path = (
            path["mesh_save_path"] + "_" + solving_params["name"] + "_" + split + ".tfrecord"
        )
        save_path = (
            path["h5_save_path"] + "_" + solving_params["name"] + "_" + split + "_" + ".h5"
        )
        with h5py.File(save_path, "w") as h5_writer:
            with tf.io.TFRecordWriter(tf_saving_mesh_path) as tf_writer:
                for index, data_dict in enumerate(ds):
                    for key, value in data_dict.items():
                        raw_data[key] = value.numpy()

                    if path["saving_tec"]:
                        tec_saving_path = (
                            path["tec_save_path"]
                            + solving_params["name"]
                            + "_"
                            + split
                            + "_"
                            + str(index)
                            + ".dat"
                        )
                        write_tecplot_ascii_nodal(
                            raw_data,
                            False,
                            "/home/litianyu/mycode/repos-py/FVM/my_FVNN/rollouts/0.pkl",
                            tec_saving_path,
                        )
                    # precomputing FVM related parameters
                    if path["saving_h5"]:
                        rtval = extract_mesh_state(
                            raw_data,
                            h5_writer,
                            index,
                            params=solving_params,
                            h5_writer=h5_writer,
                            path=path,
                        )
                        if not rtval:
                            print("parse error")
                            exit()

        print("datasets {} has been extracted mesh\n".format(split))
