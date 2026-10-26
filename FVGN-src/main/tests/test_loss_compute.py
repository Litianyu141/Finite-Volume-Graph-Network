"""Regression tests for compute_FVM_loss (issue #3: continuity loss was silently zero).

Run from FVGN-src/main:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loss_compute import compute_FVM_loss

CUDA = torch.cuda.is_available()


def make_inputs(device, loss, loss_cont, n_cells=(6, 5), n_faces=(9, 8)):
    torch.manual_seed(0)
    graph_cell = Batch.from_data_list([Data(num_nodes=n) for n in n_cells]).to(device)
    graph_edge = Batch.from_data_list([Data(num_nodes=n) for n in n_faces]).to(device)
    total_cells, total_faces = sum(n_cells), sum(n_faces)

    # a tiny "network" so that gradients w.r.t. parameters can be checked
    net = torch.nn.Linear(1, 1).to(device)
    residual = net(torch.randn(total_cells, 1, device=device))

    params = SimpleNamespace(
        loss=loss, loss_cont=loss_cont, loss_mom=10.0, loss_face_uv=1.0, loss_face_p=1.0
    )
    kwargs = dict(
        params=params,
        graph_cell=graph_cell,
        graph_edge=graph_edge,
        mask_face_interior=torch.rand(total_faces, device=device) > 0.3,
        predicted_edge_attr=torch.randn(total_faces, 3, device=device),
        target_face_uvp_normalized=torch.randn(total_faces, 3, device=device),
        predicted_delta_u=torch.randn(total_cells, 2, device=device),
        target_delta_u=torch.randn(total_cells, 2, device=device),
        loss_function=torch.nn.MSELoss(reduction="mean"),
    )
    return net, residual, kwargs


class ComputeFVMLossTest(unittest.TestCase):
    def modes(self):
        yield "direct_mean_loss", torch.device("cuda" if CUDA else "cpu")
        if CUDA:  # global_mean_* modes call .cuda() internally
            yield "global_mean_sum", torch.device("cuda")
            yield "global_mean_mean", torch.device("cuda")

    def test_nonzero_residual_contributes_loss_and_gradient(self):
        for loss, device in self.modes():
            with self.subTest(loss=loss):
                net, residual, kw = make_inputs(device, loss, loss_cont=1.0)
                total, loss_cont, *_ = compute_FVM_loss(**kw, loss_continuity=residual)
                self.assertTrue(torch.all(loss_cont > 0))

                kw["params"].loss_cont = 0.0
                total_off, *_ = compute_FVM_loss(**kw, loss_continuity=0)
                self.assertFalse(torch.allclose(total, total_off))

                net.zero_grad()
                loss_cont.sum().backward()
                self.assertGreater(net.weight.grad.abs().sum().item(), 0.0)

    def test_disabled_continuity_is_zero(self):
        for loss, device in self.modes():
            with self.subTest(loss=loss):
                _, _, kw = make_inputs(device, loss, loss_cont=0.0)
                _, loss_cont, *_ = compute_FVM_loss(**kw, loss_continuity=0)
                self.assertTrue(torch.all(loss_cont == 0))

    def test_missing_residual_raises(self):
        for loss, device in self.modes():
            with self.subTest(loss=loss):
                _, _, kw = make_inputs(device, loss, loss_cont=1.0)
                with self.assertRaises(ValueError):
                    compute_FVM_loss(**kw)


if __name__ == "__main__":
    unittest.main()
