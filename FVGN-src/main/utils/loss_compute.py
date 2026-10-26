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
    loss_continuity=None,
):

    # loss_continuity is the per-cell continuity residual from the Intergrator,
    # it is only computed when params.loss_cont > 0
    enable_continuity = params.loss_cont > 0
    if enable_continuity and not torch.is_tensor(loss_continuity):
        raise ValueError(
            "params.loss_cont > 0 but no continuity residual was passed to compute_FVM_loss"
        )

    if "global_mean" in params.loss:
        # face uv contiunity
        if enable_continuity:
            loss_continuity = global_mean_pool(
                (loss_continuity) ** 2, graph_cell.batch.cuda()
            )
            loss_continuity = (
                loss_continuity.sum(dim=-1, keepdim=True)
                if params.loss == "global_mean_sum"
                else loss_continuity.mean(dim=-1, keepdim=True)
            )
        else:
            loss_continuity = torch.zeros(
                (graph_cell.num_graphs, 1), device=predicted_delta_u.device
            )

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
        # face uv contiunity, the residual should be driven to zero
        if enable_continuity:
            loss_continuity = loss_function(
                loss_continuity, torch.zeros_like(loss_continuity)
            )
        else:
            loss_continuity = torch.zeros((), device=predicted_delta_u.device)

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
