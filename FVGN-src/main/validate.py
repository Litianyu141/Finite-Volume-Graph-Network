import argparse

parser = argparse.ArgumentParser(
    description="train / test a pytorch model to predict frames"
)
# valid parameters
parser.add_argument(
    "--model_dir",
    default="/lvm_data/litianyu/mycode-new/GEP-FVGN/Logger/net GN-Cell; hs 128; mu 0.001; rho 1; dt 0.01;/2023-11-16-03:27:44/states/40.state",
    type=str,
    help="network to valid (default: None)",
)
parser.add_argument(
    "--valid_dataset_dir",
    default="/lvm_data/litianyu/dataset/MeshGN/new_hybrid_dataset/h5",
    type=str,
    help="dataset to valid (default: None)",
)
parser.add_argument(
    "--training_round", default="0", type=str, help="no.training_round (default: 0)"
)
parser.add_argument(
    "--valid_dataset_type",
    default="h5",
    type=str,
    help="valid_dataset_type (default: 0)",
)
parser.add_argument(
    "--host_device",
    default="cuda:1",
    type=str,
    help="where is host process (default: 0)",
)
args = parser.parse_args()
import os
import torch
from FVGN_model.FVGN import FVGN
from dataset import Load_mesh
from utils import get_param
import torch_geometric.transforms as T
from utils.Logger import Logger
from utils.get_param import get_hyperparam


def validate_loop(
    fluid_model=None,
    validation_datasets=None,
    params=None,
    training_round=None,
    logger=None,
):
    fluid_model.eval()
    with torch.no_grad():
        # initialize rollout parameters
        rollout_length = 599
        rollstep = rollout_length - 1
        rollout_batch_index = [0, 30, 60, 95]

        next_UV_predicteds_list = []
        predicted_edge_p_list = []

        # fetching data once
        (
            mbatch_graph_node_v,
            mbatch_graph_edge_v,
            graph_old_v,
            mask_face_v_in,
            mask_face_v_b,
        ) = validation_datasets.require_minibatch_mesh(
            start_epoch=0,
            batch_index=rollout_batch_index,
            is_training=False,
            dual_edge=params.dual_edge,
        )

        Re_v = graph_old_v.x[:, 1:2]
        cells_type_v = graph_old_v.x[:, 0:1]
        edge_RMP_EU_v = graph_old_v.edge_attr[:, 2:5]

        for v_epoch in range(599):
            predicted_edge_uvp, next_UV_on_cell_v = fluid_model(
                graph=graph_old_v,
                graph_edge=mbatch_graph_edge_v,
                graph_node=mbatch_graph_node_v,
                rho=params.rho,
                mu=params.mu,
                dt=params.dt,
                edge_one_hot=params.edge_one_hot,
                cell_one_hot=params.cell_one_hot,
                mask=mask_face_v_b,
            )

            next_UV_predicteds_list.append((next_UV_on_cell_v[:, 0:2]).unsqueeze(0))
            predicted_edge_p_list.append(predicted_edge_uvp[:, 2:3].unsqueeze(0))

            graph_old_v = validation_datasets.create_next_graph(
                graph_old_v,
                mbatch_graph_edge_v,
                mask_face_v_b,
                next_UV_on_cell_v,
                cells_type_v,
                Re_v,
                edge_RMP_EU_v,
                dual_edge=params.dual_edge,
            )

        integrated_flux_UV = torch.cat(next_UV_predicteds_list, dim=0)
        predicted_edge_p = torch.cat(predicted_edge_p_list, dim=0)

        print(f"{training_round}: max_U: {torch.max(integrated_flux_UV)};")

        validation_loss = torch.mean(
            (
                graph_old_v.y.transpose(0, 1)[0:rollstep, :, 0:2]
                - integrated_flux_UV[0:rollstep, :, 0:2]
            )
            ** 2
        ).cpu()

        error_cell = (
            graph_old_v.y.transpose(0, 1)[0:rollstep, :, 0:2]
            - integrated_flux_UV[0:rollstep, :, 0:2]
        ) ** 2
        error_edge_p = (
            mbatch_graph_edge_v.y.transpose(0, 1)[0:rollstep, :, 2:3]
            - predicted_edge_p[0:rollstep, :, :]
        ) ** 2

        for i in range(len(rollout_batch_index)):
            graph_index = rollout_batch_index[i]
            current_graph_error = torch.mean(
                error_cell[:, graph_old_v.batch.cpu() == i, :]
            ).cpu()
            current_edge_p_error = torch.mean(
                error_edge_p[:, mbatch_graph_edge_v.batch.cpu() == i, :]
            ).cpu()
            Re = Re_v[graph_old_v.batch.cpu() == i][0].cpu()

            logger.log(
                "graph_index{0}_Re{1}_loss_validation".format(graph_index, Re),
                current_graph_error.item(),
                training_round,
            )
            logger.log(
                "graph_index{0}_Re{1}_loss_P_validation".format(graph_index, Re),
                current_edge_p_error.item(),
                training_round,
            )
            print(
                f"{training_round}: graph_index_{graph_index}_loss_validation: {current_graph_error.item()};"
            )

        logger.log(f"loss_validation", validation_loss, training_round)
        print(f"{training_round}: validation_loss: {validation_loss};")


def main():

    # configurate parameters
    params = get_param.params(os.path.split(args.model_dir)[0], fore_args_parser=parser)
    tb_path = os.path.split(os.path.split(args.model_dir)[0])[0] + "/validation"

    # for saving model
    torch.manual_seed(0)

    # check cuda
    device = params.host_device

    # initialize fluid model
    model = FVGN(device=device, params=params)

    fluid_model = model.to(device)
    fluid_model.load_checkpoint(
        ckpdir=params.model_dir, device=device, is_traning=False
    )
    fluid_model.train()

    # initialize Logger and load model / optimizer if according parameters were given
    logger = Logger(
        name=get_hyperparam(params),
        use_csv=False,
        use_tensorboard=params.log,
        params=params,
        git_info=None,
        saving_path=tb_path,
        copy_code=False,
    )

    # initialize Validation Dataset
    validation_datasets = Load_mesh.Data_Pool(
        params=params, is_traning=False, device=device
    )
    validation_datasets._set_status(is_training=False)
    validation_datasets.load_mesh_to_cpu(
        mode=args.valid_dataset_type, split="test", dataset_dir=args.valid_dataset_dir
    )
    validation_datasets._set_dataset("full")
    training_round = int(float(args.training_round))
    validate_loop(
        fluid_model=fluid_model,
        validation_datasets=validation_datasets,
        params=params,
        training_round=training_round,
        logger=logger,
    )


if __name__ == "__main__":
    print(args.model_dir)
    main()
    exit(0)
