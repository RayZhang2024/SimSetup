from pathlib import Path
import unittest

import numpy as np
import pyvista as pv

from gui_app import line_mesh_intersection_segments, load_mesh_as_polydata


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"


class MeshLoadingTests(unittest.TestCase):
    def test_load_polyhedral_sat_matches_stl_example(self) -> None:
        sat_mesh = load_mesh_as_polydata(MODEL / "T-weld.sat")
        stl_mesh = load_mesh_as_polydata(MODEL / "T_weld.stl")

        self.assertEqual(sat_mesh.n_points, stl_mesh.n_points)
        self.assertEqual(sat_mesh.n_cells, stl_mesh.n_cells)
        self.assertEqual(tuple(sat_mesh.bounds), tuple(stl_mesh.bounds))

    def test_load_existing_stl_mesh(self) -> None:
        mesh = load_mesh_as_polydata(MODEL / "2_thin_pipe.stl")

        self.assertGreater(mesh.n_points, 0)
        self.assertGreater(mesh.n_cells, 0)

    def test_line_mesh_intersection_segments_clips_to_sample_surface(self) -> None:
        mesh = pv.Cube(center=(0.0, 0.0, 0.0), x_length=10.0, y_length=4.0, z_length=2.0)

        segments = line_mesh_intersection_segments(
            mesh,
            np.array([-20.0, 0.0, 0.0], dtype=float),
            np.array([20.0, 0.0, 0.0], dtype=float),
        )

        self.assertEqual(len(segments), 1)
        np.testing.assert_allclose(segments[0][0], np.array([-5.0, 0.0, 0.0], dtype=float))
        np.testing.assert_allclose(segments[0][1], np.array([5.0, 0.0, 0.0], dtype=float))

    def test_line_mesh_intersection_segments_handles_inside_start(self) -> None:
        mesh = pv.Cube(center=(0.0, 0.0, 0.0), x_length=10.0, y_length=4.0, z_length=2.0)

        segments = line_mesh_intersection_segments(
            mesh,
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([20.0, 0.0, 0.0], dtype=float),
        )

        self.assertEqual(len(segments), 1)
        np.testing.assert_allclose(segments[0][0], np.array([0.0, 0.0, 0.0], dtype=float))
        np.testing.assert_allclose(segments[0][1], np.array([5.0, 0.0, 0.0], dtype=float))


if __name__ == "__main__":
    unittest.main()
