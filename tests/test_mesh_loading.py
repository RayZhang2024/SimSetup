from pathlib import Path
import unittest

from gui_app import load_mesh_as_polydata


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


if __name__ == "__main__":
    unittest.main()
