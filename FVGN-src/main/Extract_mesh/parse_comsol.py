import os
import sys
current_file_path = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(current_file_path)
from utils.utilities import NodeType

import numpy as np
import torch
import re
from parse_tfrecord_refactor import extract_mesh_state
import tensorflow as tf
import os
import matplotlib

matplotlib.use("agg")
import h5py
import random
from contextlib import ExitStack


class Cosmol_manager:
    def __init__(
        self,
        mesh_file,
        data_file,
        tf_writer=None,
        h5_writer=None,
        origin_writer=None,
        path=None,
    ):
        self.node_type = []
        self.tf_writer = tf_writer
        self.h5_writer = h5_writer
        self.origin_writer = origin_writer
        self.path = path

        subdir, mesh_name = os.path.split(mesh_file)
        self.path["base_dir"] = subdir
        self.path["case_number"] = "".join([s for s in mesh_name if s.isdigit()])
        self.path["mesh_file_path"] = mesh_file
        self.path["data_file_path"] = data_file
        self.case_info = {
            "U_mean": 0.0,
            "R": (
                0.3 if "rect" in mesh_file else 0.5 if "NACA0012" in mesh_file else 0.0
            ),  # 修正了条件表达式
            "aoa": 0.0,
        }
        self.mesh_file_handle = self.openreadtxt(mesh_file, mesh_reading=True)
        self.data_file_handle = self.openreadtxt(data_file, data_reading=True)

        if not self.mesh_file_handle or not self.data_file_handle:
            raise ValueError("")

    def openreadtxt(self, file_name, mesh_reading=False, data_reading=False):
        self.raw_txt_data = []
        file = open(file_name, "r")  # 打开文件
        file_data = file.readlines()  # 读取所有行
        count = 0
        index = 0
        self.X = []
        self.Y = []
        self.U = []
        self.V = []
        self.P = []

        while index < len(file_data):

            count += 1
            row = file_data[index]
            if row.find("# number of mesh vertices\n") > 0:
                tmp_list = row.split(" ")  # 按‘ '切分每行的数据
                self.num_of_mesh_pos = int(tmp_list[0])
            elif row == "# Mesh vertex coordinates\n":
                index = self.read_mesh_pos(file_data, index + 1, self.num_of_mesh_pos)
            elif row == "3 edg # type name\n":
                index = self.read_index(file_data, index)
            elif row == "3 tri # type name\n":
                index = self.read_index(file_data, index)
            elif row.find("x (m) @ t") > 0 or row.find("% x") == 0:
                tmp_list = file_data[index + 1]
                tmp_list = re.sub(r"[ ]+", " ", tmp_list).split(" ")
                x_tmp = [np.array(x, dtype=np.float64) for x in tmp_list]
                self.X.append(np.array(x_tmp))
            elif row.find("y (m) @ t") > 0 or row.find("% y") == 0:
                tmp_list = file_data[index + 1]
                tmp_list = re.sub(r"[ ]+", " ", tmp_list).split(" ")
                y_tmp = [np.array(y, dtype=np.float64) for y in tmp_list]
                self.Y.append(np.array(y_tmp))
            elif row.find("u (m/s) @") > 0:
                case_para_info = row.split(",")
                for para in case_para_info:
                    if para.find("U_mean") >= 0:
                        self.case_info["U_mean"] = float(para.split("=")[1])
                    elif para.find("R") >= 0:
                        self.case_info["R"] = float(para.split("=")[1])
                    elif para.find("aoa") >= 0:
                        self.case_info["aoa"] = float(para.split("=")[1])

                tmp_list = file_data[index + 1]
                tmp_list = re.sub(r"[ ]+", " ", tmp_list).split(" ")
                u_tmp = [np.array(u, dtype=np.float64) for u in tmp_list]
                self.U.append(np.array(u_tmp))
            elif row.find("v (m/s) @") > 0:
                tmp_list = file_data[index + 1]
                tmp_list = re.sub(r"[ ]+", " ", tmp_list).split(" ")
                v_tmp = [np.array(v, dtype=np.float64) for v in tmp_list]
                self.V.append(np.array(v_tmp))
            elif row.find("p (Pa) @") > 0:
                tmp_list = file_data[index + 1]
                tmp_list = re.sub(r"[ ]+", " ", tmp_list).split(" ")
                p_tmp = [np.array(p, dtype=np.float64) for p in tmp_list]
                self.P.append(np.array(p_tmp))
            """
            tmp_list = row.split(' ') #按‘，'切分每行的数据
            tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
            self.raw_txt_data.append(tmp_list) #将每行数据插入data中
            """
            index += 1
        try:
            if mesh_reading:
                print("mesh read done")
            elif data_reading and len(self.X) == 0:
                raise ValueError(f"data{file_name} read faild")
            elif data_reading and len(self.X) > 0:
                print("data read done")
            return True
        except:
            return False

    def read_mesh_pos(self, input, start, end):
        self.mesh_pos = []
        for index in range(start, end + start):
            raw_data = input[index].split(" ")
            raw_data[-1] = raw_data[-1].replace("\n", ",")
            raw_x = np.array(raw_data[0], dtype=np.float64)
            raw_y = np.array(raw_data[1], dtype=np.float64)
            raw_pos = np.array([raw_x, raw_y])
            self.mesh_pos.append(raw_pos)
        self.mesh_pos = np.array(self.mesh_pos)
        return index

    def read_header(self, input, start, end):
        self.mesh_pos.append(input, start, end)

    def read_index(self, input, start):
        self.mesh_boundary_index = []
        self.mesh_index = []
        i = start
        while start < len(input):
            if input[start].find("# number of elements\n") > 0:
                raw_data = input[start].split(" ")
                raw_data[-1] = raw_data[-1].replace("\n", ",")
                num_of_elements = int(raw_data[0])
            elif input[start] == "# Elements\n":
                for start in range(start + 1, start + num_of_elements + 1):
                    raw_data = input[start].split(" ")
                    raw_data[-1] = raw_data[-1].replace("\n", ",")
                    raw_x = np.array(int(raw_data[0]))
                    raw_y = np.array(int(raw_data[1]))
                    raw_pos = np.array([raw_x, raw_y])
                    self.mesh_boundary_index.append(raw_pos)
                    start += 1
                self.mesh_boundary_index = np.array(self.mesh_boundary_index)
                break
            elif input[start] == "3 vtx # type name\n":
                break
            start += 1

        while start < len(input):
            if input[start].find("# number of elements\n") > 0:
                raw_data = input[start].split(" ")
                raw_data[-1] = raw_data[-1].replace("\n", ",")
                num_of_elements = int(raw_data[0])
            elif input[start] == "# Elements\n":
                for start in range(start + 1, start + num_of_elements + 1):
                    raw_data = input[start].split(" ")
                    raw_data[-1] = raw_data[-1].replace("\n", ",")
                    raw_x = np.array(int(raw_data[0]))
                    raw_y = np.array(int(raw_data[1]))
                    raw_z = np.array(int(raw_data[2]))
                    raw_pos = np.array([raw_x, raw_y, raw_z])
                    self.mesh_index.append(raw_pos)
                    start += 1
                self.mesh_index = np.array(self.mesh_index)
                break
            elif input[start] == "3 vtx # type name\n":
                break
            start += 1
        return start

    def extract_mesh(self, plot, data_index=None, path=None):
        INFLOW = 0
        WALL_BOUNDARY = 0
        OUTFLOW = 0
        OBSTACLE = 0
        NORMAL = 0
        mesh_pos = torch.from_numpy(self.mesh_pos)
        mesh_boundary_index = torch.from_numpy(self.mesh_boundary_index).to(torch.long)
        boundary_pos_0 = torch.index_select(
            mesh_pos, 0, mesh_boundary_index[:, 0]
        ).numpy()
        boundary_pos_1 = torch.index_select(
            mesh_pos, 0, mesh_boundary_index[:, 1]
        ).numpy()
        self.node_type = np.empty((self.mesh_pos.shape[0], 1))

        # prepare for renumber data
        rearrange_pos_dict = {}
        data_pos = np.stack((np.array(self.X), np.array(self.Y)), axis=-1)[0]

        for index in range(self.mesh_pos.shape[0]):
            rearrange_pos_dict[str(data_pos[index].astype(np.float32))] = index
        data_velocity = np.stack((np.array(self.U), np.array(self.V)), axis=-1)
        data_pressure = np.expand_dims(np.array(self.P), axis=-1)
        rearrange_index = np.zeros(data_velocity.shape[1])

        velocity = np.zeros_like(data_velocity)
        pressure = np.zeros_like(data_pressure)
        self.mesh_pos = self.mesh_pos.astype(np.float32)
        topwall = np.max(self.mesh_pos[:, 1])
        bottomwall = np.min(self.mesh_pos[:, 1])
        outlet = np.max(self.mesh_pos[:, 0])
        inlet = np.min(self.mesh_pos[:, 0])

        WALL_BOUNDARY_list = []
        OBSTACLE_list = []
        OUTFLOW_list = []
        for i in range(self.mesh_pos.shape[0]):

            current_coord = self.mesh_pos[i]
            if (
                (current_coord[0] == inlet)
                and (current_coord[1] > bottomwall)
                and (current_coord[1] < topwall)
            ):
                self.node_type[i] = NodeType.INFLOW
                INFLOW += 1
            elif (current_coord[1] >= topwall) or (current_coord[1] <= bottomwall):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                WALL_BOUNDARY_list.append(current_coord)
                WALL_BOUNDARY += 1
            elif (
                (current_coord[0] == outlet)
                and (current_coord[1] > bottomwall)
                and (current_coord[1] < topwall)
            ):
                self.node_type[i] = NodeType.OUTFLOW
                OUTFLOW += 1
                OUTFLOW_list.append(current_coord)
            elif (
                (i in mesh_boundary_index)
                and (current_coord[0] > 0)
                and (current_coord[0] < outlet)
                and (current_coord[1] > 0)
                and (current_coord[1] < topwall)
            ):
                self.node_type[i] = NodeType.WALL_BOUNDARY
                WALL_BOUNDARY += 1
                OBSTACLE += 1
                OBSTACLE_list.append(current_coord)
            else:
                self.node_type[i] = NodeType.NORMAL
                NORMAL += 1
            rearrange_index[i] = rearrange_pos_dict[str(current_coord)]
        print(
            "After readed data in file has NODE TYPE: NORMAL: {0} OBSTACLE: {1} AIRFOIL: {2} HANDLE: {3} INFLOW: {4} OUTFLOW: {5} WALL_BOUNDARY: {6} SIZE: {7}".format(
                NORMAL, OBSTACLE, 0, 0, INFLOW, OUTFLOW, WALL_BOUNDARY, 0
            )
        )

        velocity = (
            torch.index_select(
                torch.from_numpy(data_velocity).cuda(),
                1,
                torch.from_numpy(rearrange_index).to(torch.long).cuda(),
            )
            .cpu()
            .numpy()
        )
        pressure = (
            torch.index_select(
                torch.from_numpy(data_pressure).cuda(),
                1,
                torch.from_numpy(rearrange_index).to(torch.long).cuda(),
            )
            .cpu()
            .numpy()
        )

        mesh = {
            "mesh_pos": np.repeat(
                np.expand_dims(self.mesh_pos, axis=0), 600, axis=0
            ).astype(np.float32),
            "boundary": np.repeat(
                np.expand_dims(self.mesh_boundary_index, axis=0), 600, axis=0
            ).astype(np.int32),
            "cells": np.repeat(
                np.expand_dims(self.mesh_index, axis=0), 600, axis=0
            ).astype(np.int32),
            "node_type": np.repeat(
                np.expand_dims(self.node_type, axis=0), 600, axis=0
            ).astype(np.int32),
            "velocity": velocity[0:600].astype(np.float32),
            "pressure": pressure[0:600].astype(np.float32),
            "U_mean": self.case_info["U_mean"],
            "R": self.case_info["R"],
            "aoa": self.case_info["aoa"],
        }

        # if plot:
        #     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        #     ax.cla()
        #     ax.set_aspect('equal')
        #     bb_min = mesh['velocity'].min(axis=(0, 1))
        #     bb_max = mesh['velocity'].max(axis=(0, 1))
        #     for i in range(600):
        #         triang = mtri.Triangulation(mesh["mesh_pos"][i,:, 0], mesh["mesh_pos"][i,:, 1])
        #         ax.tripcolor(triang, mesh['velocity'][i][:, 0], vmin=bb_min[0], vmax=bb_max[0])
        #         #ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
        #         plt.show()

        extract_mesh_state(
            dataset=mesh,
            tf_writer=self.tf_writer,
            index=data_index,
            origin_writer=self.origin_writer,
            solving_params=self.path,
            h5_writer=self.h5_writer,
            path=self.path,
        )
        return mesh


def random_samples_no_replacement(arr, num_samples, num_iterations):
    if num_samples * num_iterations > len(arr):
        raise ValueError(
            "Number of samples multiplied by iterations cannot be greater than the length of the array."
        )

    samples = []
    arr_copy = arr.copy()

    for _ in range(num_iterations):
        sample_indices = np.random.choice(len(arr_copy), num_samples, replace=False)
        sample = arr_copy[sample_indices]
        samples.append(sample)

        # 从 arr_copy 中移除已选样本
        arr_copy = np.delete(arr_copy, sample_indices)

    return samples, arr_copy


if __name__ == "__main__":
    dataset_path = "/mnt/f/lty-dataset/Hybrid-dataset-Re200-1500/train_dataset"
    tf_saving_path = "/mnt/f/lty-dataset/Hybrid-dataset-Re200-1500/converted_dataset/tf"
    h5_saving_path = "/mnt/f/lty-dataset/Hybrid-dataset-Re200-1500/converted_dataset/h5"
    origin_saving_path = (
        "/mnt/f/lty-dataset/Hybrid-dataset-Re200-1500/converted_dataset/origin"
    )
    no_split_dataset = True

    case = 0  # 0 stands for 980/PM9A1
    if case == 0:
        path = {
            "simulator": "COMSOL",
            "name":"incompressible",
            "dt": 0.01,
            "rho": 1,
            "mu": 0.001,
            "features": None,
            "comsol_dataset_path": "/mnt/f/lty-dataset/Hybrid-dataset-Re200-1500/train_dataset",
            "h5_save_path": "/mnt/f/lty-dataset/Hybrid-dataset-Re200-1500/converted_dataset/h5",
            "tf_saving_path": "/mnt/f/lty-dataset/Hybrid-dataset-Re200-1500/converted_dataset/tf",
            "origin_saving_path": "/mnt/f/lty-dataset/Hybrid-dataset-Re200-1500/converted_dataset/origin",
            "mode": "cylinder_mesh",
            "saving_tf": True,
            "stastic": False,
            "saving_origin": True,
            "saving_h5": True,
            "print_tf": False,
            "plot_order": False,
        }

    path["comsol_dataset_path"] = dataset_path
    path["h5_save_path"] = h5_saving_path
    path["tf_saving_path"] = tf_saving_path
    path["origin_saving_path"] = origin_saving_path

    # stastic total number of data samples
    total_samples = 0
    for subdir, dirs, files in os.walk(dataset_path):
        for data_name in files:
            if data_name.find("data") >= 0:
                total_samples += 1
    print("total samples: ", total_samples)
    sample_index = np.arange(total_samples)

    def split_dataset(dataset_path, no_split_dataset=True):
        if no_split_dataset:
            # 不划分数据集，收集所有 mphtxt 文件
            all_files = []
            for subdir, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith("mphtxt"):
                        all_files.append(os.path.join(subdir, file))
            return all_files, all_files, all_files
        else:
            # 定义每个类别在不同数据集中的样本数
            train_counts = {"cylinder": 800, "rect": 100, "NACA0012": 100}
            test_val_counts = {"cylinder": 50, "rect": 25, "NACA0012": 25}

            # 初始化数据集列表
            train_set, test_set, val_set = [], [], []

            # 遍历每个类别文件夹
            for category in ["cylinder", "rect", "NACA0012"]:
                category_path = os.path.join(dataset_path, category)
                all_files = []

                # 遍历类别下的所有子文件夹
                for subdir in os.listdir(category_path):
                    subdir_path = os.path.join(category_path, subdir)
                    if os.path.isdir(subdir_path):
                        # 找到所有以 "mphtxt" 结尾的文件
                        all_files.extend(
                            [
                                os.path.join(subdir_path, f)
                                for f in os.listdir(subdir_path)
                                if f.endswith("mphtxt")
                            ]
                        )

                # 确保文件是随机排序的
                np.random.shuffle(all_files)

                # 分配文件到训练集
                train_files = all_files[: train_counts[category]]
                train_set.extend(train_files)

                # 剩余文件用于测试集和验证集
                remaining_files = all_files[train_counts[category] :]

                # 确保测试集和验证集有足够的样本
                while len(remaining_files) < 2 * test_val_counts[category]:
                    remaining_files.append(
                        random.choice(train_files)
                    )  # 从训练集中复制样本

                # 分配文件到测试集和验证集
                test_set.extend(remaining_files[: test_val_counts[category]])
                val_set.extend(
                    remaining_files[
                        test_val_counts[category] : 2 * test_val_counts[category]
                    ]
                )

            return train_set, test_set, val_set

    train_set, test_set, val_set = split_dataset(dataset_path, no_split_dataset)
    print(
        f"train samples:{len(train_set)}, test samples:{len(test_set)}, valid samples:{len(val_set)}"
    )

    # 初始化失败案例列表
    faild_cases = []

    with ExitStack() as stack:
        all_writers = {}
        for writers in ["train", "valid", "test"]:
            tf_writer = stack.enter_context(
                tf.io.TFRecordWriter(f"{tf_saving_path}/{writers}.tfrecord")
            )
            origin_writer = stack.enter_context(
                tf.io.TFRecordWriter(f"{origin_saving_path}/{writers}.tfrecord")
            )
            h5_writer = stack.enter_context(
                h5py.File(f"{h5_saving_path}/{writers}.h5", "w")
            )
            writer_list = [tf_writer, origin_writer, h5_writer]
            all_writers[writers] = writer_list

        # 创建一个函数来处理每个数据集
        def process_dataset(dataset, writers, data_index_start):
            data_index = data_index_start
            for file_path in dataset:
                try:
                    # 提取路径和文件名
                    subdir, mesh_name = os.path.split(file_path)
                    data_name = (
                        "data" + "".join([s for s in mesh_name if s.isdigit()]) + ".txt"
                    )

                    # 获取对应的写入器
                    tf_writer, origin_writer, h5_writer = writers

                    # 使用 Cosmol_manager 进行处理
                    data = Cosmol_manager(
                        mesh_file=file_path,
                        data_file=os.path.join(subdir, data_name),
                        tf_writer=tf_writer,
                        h5_writer=h5_writer,
                        origin_writer=origin_writer,
                        path=path,
                    )  # 确保 path 已正确定义

                    data.extract_mesh(plot=False, data_index=data_index)
                    data_index += 1

                except Exception as e:
                    faild_cases.append(f"{subdir}/{data_name}")
                    print(f"parsing {subdir}/{data_name} failed with error: {e}\n")

            return data_index

        # 处理每个数据集
        train_data_index = process_dataset(train_set, all_writers["train"], 0)
        valid_data_index = process_dataset(val_set, all_writers["valid"], 0)
        test_data_index = process_dataset(test_set, all_writers["test"], 0)

    with open(f"{dataset_path}/faild_case.txt", "w") as f:
        for faild_case in faild_cases:
            f.write(f"parsing {faild_case} faild\n")
            print(f"parsing {faild_case} faild\n")
            # data  = Cosmol_manager("/data/litianyu/dataset/MeshGN/cylinder_flow/test_dataset/tri_wing_0_mesh_flow1.txt","/data/litianyu/dataset/MeshGN/cylinder_flow/test_dataset/tri_wing_fluid_field_0_15_614.txt")
            # mesh_raw = data.extract_mesh(False)
            # print("done")
