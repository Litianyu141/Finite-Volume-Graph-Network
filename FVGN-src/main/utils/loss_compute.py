import torch
from torch_geometric.nn import global_mean_pool


def direct_decode_loss(
    params,
    graph_cell,
    graph_edge,
    mask_face_interior,
    predicted_edge_attr,
    target_face_uvp_normalized,
    predicted_delta_u,
    target_delta_u,
    loss_function,
):

    error = (target_delta_u[:, 0:2] - predicted_delta_u[:, 0:2]) ** 2

    loss_cell_uv = torch.mean(torch.sum(error[:, 0:2], dim=1))

    loss_cell = loss_function(predicted_delta_u[:, 0:2], target_delta_u[:, 0:2])


def compute_FVM_loss(
    params=None,
    graph_cell=None,
    graph_edge=None,
    mask_face_interior=None,
    predicted_edge_attr=None,
    target_face_uvp_normalized=None,
    predicted_delta_u=None,
    target_delta_u=None,
    loss_function=None,
):

    if "global_mean" in params.loss:
        # face uv contiunity
        try:
            loss_continuity = global_mean_pool(
                (loss_continuity) ** 2, graph_cell.batch.cuda()
            )
        except:
            loss_continuity = 0.0

        # face uv flux loss
        loss_face_uv = global_mean_pool(
            (
                target_face_uvp_normalized[mask_face_interior, 0:2]
                - predicted_edge_attr[mask_face_interior, 0:2]
            )
            ** 2,
            graph_edge.batch[mask_face_interior].cuda(),
        ).sum(dim=-1, keepdim=True)

        loss_face_uv = (
            loss_face_uv.sum(dim=-1, keepdim=True)
            if params.loss == "global_mean_sum"
            else loss_face_uv.mean(dim=-1, keepdim=True)
        )

        # introudce the face pressure loss function into the optimization process, which means we assum the sum of flux p close to target at time t+dt
        loss_face_p = global_mean_pool(
            (target_face_uvp_normalized[:, 2:3] - predicted_edge_attr[:, 2:3]) ** 2,
            graph_edge.batch.cuda(),
        )

        loss_face_p = (
            loss_face_p.sum(dim=-1, keepdim=True)
            if params.loss == "global_mean_sum"
            else loss_face_p.mean(dim=-1, keepdim=True)
        )

        # cell center momentum loss
        loss_mom = global_mean_pool(
            (target_delta_u - predicted_delta_u) ** 2, graph_cell.batch.cuda()
        )

        loss_mom = (
            loss_mom.sum(dim=-1, keepdim=True)
            if params.loss == "global_mean_sum"
            else loss_mom.mean(dim=-1, keepdim=True)
        )

    else:
        # face uv contiunity
        try:
            loss_continuity = loss_function(loss_continuity)
        except:
            loss_continuity = 0.0

        # face uv flux loss
        loss_face_uv = loss_function(
            predicted_edge_attr[mask_face_interior, 0:2],
            target_face_uvp_normalized[mask_face_interior, 0:2],
        )

        # introudce the grad_p loss function into the optimization process, which means we assum the sum of flux p close to target at time t+dt
        loss_face_p = loss_function(
            predicted_edge_attr[:, 2:3], target_face_uvp_normalized[:, 2:3]
        )

        # cell center loss
        loss_mom = loss_function(predicted_delta_u, target_delta_u)

    # compute total loss with momentum equation and continuity equation
    loss = (
        params.loss_cont * loss_continuity
        + params.loss_mom * loss_mom
        + params.loss_face_uv * loss_face_uv
        + params.loss_face_p * loss_face_p
    )

    loss = torch.mean(torch.log(loss))

    return loss, loss_continuity, loss_mom, loss_face_uv, loss_face_p
