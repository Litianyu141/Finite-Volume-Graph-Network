"""Regression test for face labelling of incompressible meshes (issue #3).

Needs tensorflow and the DeepMind cylinder_flow dataset (meta.json + valid.tfrecord).
Set FVGN_CYLINDER_DIR to that directory, otherwise the test is skipped.

Run from FVGN-src/main:  FVGN_CYLINDER_DIR=/path/cylinder_flow python -m unittest discover -s tests -v
"""
import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utilities import NodeType

DATASET_DIR = os.environ.get("FVGN_CYLINDER_DIR")


@unittest.skipUnless(
    DATASET_DIR and os.path.exists(os.path.join(DATASET_DIR, "valid.tfrecord")),
    "set FVGN_CYLINDER_DIR to the cylinder_flow dataset to run this test",
)
class IncompressibleFaceTypeTest(unittest.TestCase):
    def test_cylinder_boundary_faces_are_labelled(self):
        import h5py
        from Extract_mesh import parse_tfrecord_refactor as parser

        solving_params = {"name": "incompressible", "rho": 1.0, "mu": 0.001, "dt": 0.01}
        ds = parser.load_dataset(DATASET_DIR, "valid")
        raw = {k: v.numpy() for k, v in next(itertools.islice(ds, 1)).items()}

        with h5py.File("in_memory.h5", "w", driver="core", backing_store=False) as w:
            self.assertTrue(
                parser.extract_mesh_state(
                    raw, w, 0, solving_params=solving_params, h5_writer=w,
                    path={"saving_h5": True},
                )
            )
            face_type = w["0"]["face_type"][0].reshape(-1)
            cells_face = w["0"]["cells_face"][0].reshape(-1)

        # boundary faces belong to exactly one cell
        boundary = np.bincount(cells_face, minlength=face_type.shape[0]) == 1
        self.assertGreater(boundary.sum(), 0)
        self.assertEqual(int((face_type[boundary] == NodeType.NORMAL).sum()), 0)
        self.assertTrue(np.all(face_type[~boundary] == NodeType.NORMAL))
        for t in (NodeType.WALL_BOUNDARY, NodeType.INFLOW, NodeType.OUTFLOW):
            self.assertGreater(int((face_type == t).sum()), 0, t.name)


if __name__ == "__main__":
    unittest.main()
