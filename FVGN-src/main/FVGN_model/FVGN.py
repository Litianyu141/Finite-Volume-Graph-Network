import os
import sys

current_file_path = os.path.split(os.path.split(__file__)[0])[0]
sys.path.append(current_file_path)
from utils.utilities import NodeType

from .EncoderProcesserDecoder import EncoderProcesserDecoder, Intergrator
import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import enum
import numpy as np


class FVGN(nn.Module):

    def __init__(self, device=None, model_dir=None, params=None) -> None:
        super(FVGN, self).__init__()
        self._device = device
        self.edge_input_size = params.edge_input_size + params.edge_one_hot
        self.cell_input_size = params.cell_input_size + params.cell_one_hot
        self.model_dir = model_dir
        self.dual_edge = params.dual_edge

        self.model = EncoderProcesserDecoder(
            message_passing_num=params.message_passing_num,
            cell_input_size=self.cell_input_size,
            edge_input_size=self.edge_input_size,
            cell_output_size=params.cell_output_size,
            edge_output_size=params.edge_output_size,
            drop_out=params.drop_out,
            mp_times=params.mp_times,
            MultiHead=params.multihead,
            dual_edge=params.dual_edge,
            nn_act=params.nn_act,
        ).to(device)

        self.intergator = Intergrator(
            self.edge_input_size,
            self.cell_input_size,
            params.cell_output_size,
            params.loss_cont,
        )

        self._acc_output_normalizer = normalization.Normalizer(
            size=params.cell_target_input_size,
            max_accumulations=(int(params.dataset_size / params.batch_size) + 1)
            * params.cumulative_length,
            device=device,
        )

        self._face_uvp_flux_output_normalizer = normalization.Normalizer(
            size=params.face_flux_target_input_size,
            max_accumulations=(int(1000 / params.batch_size) + 1)
            * params.cumulative_length,
            device=device,
        )

        """ Fowarding two times has different data distribution,at first it is t0`s ground truth,but t1 has noise produced by the first Fowarding """
        self._edge_normalizer = normalization.Normalizer(
            size=params.edge_input_size + params.edge_one_hot,
            max_accumulations=(int(params.dataset_size / params.batch_size) + 1)
            * params.cumulative_length,
            device=device,
        )

        self._cell_normalizer = normalization.Normalizer(
            size=params.cell_input_size + params.cell_one_hot,
            max_accumulations=(int(params.dataset_size / params.batch_size) + 1)
            * params.cumulative_length,
            device=device,
        )

        """ we have 1000 cases, All of them will be traversed in one epoch """
        self._grid_feature_normalizer = nn.BatchNorm1d(
            num_features=params.grid_feature_size
        )

        print("FVGN model initialized")

    def update_cell_attr(
        self,
        frames,
        one_hot: int,
        types: torch.Tensor,
        accumulation=True,
        normalizer=None,
    ):

        if one_hot > 0:
            cell_feature = []
            cell_feature.append(frames)  # velocity
            cell_type = torch.squeeze(types.long())
            one_hot = torch.nn.functional.one_hot(cell_type, one_hot)
            cell_feature.append(one_hot)
            cell_feats = torch.cat(cell_feature, dim=1)
        else:
            cell_feats = frames

        if normalizer == "1":
            attr = self._cell_normalizer(cell_feats, accumulation)
        else:
            attr = cell_feats
        return attr

    """we do normlize EU and relative_mesh_pos,also normlize cell_differce_on_edge"""

    def update_edge_attr(
        self,
        edge_attr,
        one_hot: int,
        types: torch.Tensor,
        accumulation=True,
        normalizer=None,
        dual_edge=False,
    ):
        edge_feature = []
        edge_feature.append(edge_attr)

        if dual_edge:
            faces_type = types.long().view(-1).repeat(2)
        else:
            faces_type = types.long().view(-1)

        if one_hot > 0:
            one_hot = torch.nn.functional.one_hot(faces_type, one_hot)
            edge_feature.append(one_hot)
            edge_feats = torch.cat(edge_feature, dim=1)

        else:
            edge_feats = edge_attr
        if normalizer == "1":
            attr = self._edge_normalizer(edge_feats, accumulation)
        else:
            attr = edge_feats
        return attr

    """Use first order difference on transient U as the target for NN"""

    def velocity_to_accelation(self, noised_frames, next_cell_attr):

        acc_next = next_cell_attr - noised_frames
        return acc_next

    def forward(
        self,
        graph: Data = None,
        graph_edge: Data = None,
        graph_node: Data = None,
        rho=None,
        mu=None,
        dt=None,
        edge_one_hot=9,
        cell_one_hot=9,
        device=None,
    ):

        if self.training:
            cells_type = graph.x[:, 0:1]
            faces_type = graph_edge.x[:, 0:1]
            # reynolds_num = graph.x[:,1:2]
            current_cell_frames = graph.x[:, 2:4]

            """perform *************************NOISE INJECTION*********************** at cell attr and edge attributes"""

            noised_cell_frames = current_cell_frames
            target_on_cell = graph.y[:, 0:2]
            target_on_edge = graph_edge.y[:, 0:3]

            # calculate target
            target_acceration_on_cell = self.velocity_to_accelation(
                noised_cell_frames[:, 0:2], target_on_cell[:, 0:2]
            )
            target_delta_u = self._acc_output_normalizer(
                target_acceration_on_cell, self.training
            )
            target_face_uvp_normalized = self._face_uvp_flux_output_normalizer(
                target_on_edge, self.training
            )

            # update cell and edge attributes by normalization
            graph.x = self.update_cell_attr(
                noised_cell_frames,
                cell_one_hot,
                cells_type,
                self.training,
                normalizer="1",
            )
            graph.edge_attr = self.update_edge_attr(
                graph.edge_attr,
                edge_one_hot,
                faces_type,
                self.training,
                normalizer="1",
                dual_edge=self.dual_edge,
            )

            # forward model
            _, predicted_edge_attr = self.model(graph, graph_node)

            # Original way
            # rhs_coef = (dt/graph.cell_area)
            # face_area = graph_edge.x[:,1:2]

            # In a normalize way
            rhs_coef = 1.0
            face_area = self._grid_feature_normalizer(
                (
                    graph_edge.x[:, 1:2]
                    * (
                        dt
                        / (
                            (
                                torch.index_select(
                                    graph.cell_area, 0, graph.edge_index[0]
                                )
                                + torch.index_select(
                                    graph.cell_area, 0, graph.edge_index[1]
                                )
                            )
                            / 2
                        )
                    )
                ).view(-1, 1)
            )

            loss_continuity, predicted_delta_u = self.intergator(
                uv_face=predicted_edge_attr[:, 0:2],
                p_face=predicted_edge_attr[:, 2:3].clone().detach(),
                flux_D=predicted_edge_attr[:, 3:5],
                unv=graph.unv,
                rho=rho,
                rhs_coef=rhs_coef,
                face_area=face_area,
                cell_face=graph_edge.face,
                device=self._device,
            )

            return (
                loss_continuity,
                predicted_delta_u,
                target_delta_u,
                predicted_edge_attr,
                target_face_uvp_normalized,
            )

        else:
            """Rolling out results"""
            cells_type = graph.x[:, 0:1]
            faces_type = graph_edge.x[:, 0:1]
            # reynolds_num = graph.x[:,1:2]
            current_cell_frames = graph.x[:, 2:4]
            noised_cell_frames = current_cell_frames
            # target_face_uvp_normalized = self._face_uvp_flux_output_normalizer(graph_edge.y)

            # update normalized value
            graph.x = self.update_cell_attr(
                noised_cell_frames,
                cell_one_hot,
                cells_type,
                self.training,
                normalizer="1",
            )
            graph.edge_attr = self.update_edge_attr(
                graph.edge_attr,
                edge_one_hot,
                faces_type,
                self.training,
                normalizer="1",
                dual_edge=self.dual_edge,
            )

            _, predicted_edge_attr = self.model(graph, graph_node)

            # do not update boundary values
            # predicted_edge_attr[mask,0:2] = target_face_uvp_normalized[mask,0:2]

            # Original way
            # rhs_coef = (dt/graph.cell_area)
            # face_area = graph_edge.x[:,1:2]

            # In a normalize way
            rhs_coef = 1.0
            face_area = self._grid_feature_normalizer(
                (
                    graph_edge.x[:, 1:2]
                    * (
                        dt
                        / (
                            (
                                torch.index_select(
                                    graph.cell_area, 0, graph.edge_index[0]
                                )
                                + torch.index_select(
                                    graph.cell_area, 0, graph.edge_index[1]
                                )
                            )
                            / 2
                        )
                    )
                ).view(-1, 1)
            )

            loss_continuity, predicted_delta_u = self.intergator(
                uv_face=predicted_edge_attr[:, 0:2],
                p_face=predicted_edge_attr[:, 2:3].clone().detach(),
                flux_D=predicted_edge_attr[:, 3:5],
                unv=graph.unv,
                rho=rho,
                rhs_coef=rhs_coef,
                face_area=face_area,
                cell_face=graph_edge.face,
                device=self._device,
            )

            next_UV_on_cell = (
                self._acc_output_normalizer.inverse(predicted_delta_u)
                + current_cell_frames
            )
            predicted_uvp_on_edge = self._face_uvp_flux_output_normalizer.inverse(
                predicted_edge_attr[:, 0:3]
            )

            return predicted_uvp_on_edge, next_UV_on_cell

    def load_checkpoint(
        self,
        optimizer=None,
        scheduler=None,
        ckpdir=None,
        device=None,
        is_traning=True,
        trian_time_steps=None,
    ):

        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir, map_location=device)
        self.load_state_dict(dicts["model"])
        train_time_steps = dicts["trian_time_steps"]
        keys = list(dicts.keys())
        keys.remove("model")
        keys.remove("trian_time_steps")
        if optimizer is not None:
            if type(optimizer) is not list:
                optimizer_t = [optimizer]
            for i, o in enumerate(optimizer_t):
                o.load_state_dict(dicts["optimizer{}".format(i)])
                keys.remove("optimizer{}".format(i))

        if scheduler is not None:
            if type(scheduler) is not list:
                scheduler_t = [scheduler]
            for i, s in enumerate(scheduler_t):
                s.load_state_dict(dicts["scheduler{}".format(i)])
                scheduler_dicts = dicts["scheduler{}".format(i)]
                for key, value in scheduler_dicts.items():
                    object = eval("s." + key)
                    if type(value) == torch.Tensor:
                        value = value.cpu().cuda(device)
                    setattr(s, key, value)
                keys.remove("scheduler{}".format(i))

        for key in keys.copy():
            if key.find("optimizer") >= 0:
                keys.remove(key)
            elif key.find("scheduler") >= 0:
                keys.remove(key)

        print("FVGN model and optimizer/scheduler loaded checkpoint %s" % ckpdir)

        return train_time_steps

    def save_checkpoint(
        self,
        path=None,
        optimizer=None,
        scheduler=None,
        trian_time_steps: np.ndarray = None,
        index="final",
    ):
        if path is None:
            path = self.model_dir

        model = self.state_dict()

        to_save = {"model": model, "trian_time_steps": trian_time_steps}

        if type(optimizer) is not list:
            optimizer = [optimizer]
        for i, o in enumerate(optimizer):
            to_save.update({"optimizer{}".format(i): o.state_dict()})

        if type(scheduler) is not list:
            scheduler = [scheduler]
        for i, s in enumerate(scheduler):
            to_save.update({"scheduler{}".format(i): s.get_variable()})

        torch.save(to_save, path)
        print("FVGN model saved at %s" % path)
