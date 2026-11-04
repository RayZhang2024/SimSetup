from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import pyvista as pv
    from PyQt5.QtWidgets import QApplication

    from gui_app import PointPickerPanel, load_mesh_as_polydata
except Exception as exc:  # pragma: no cover - depends on optional GUI stack
    GUI_IMPORT_ERROR = exc
else:
    GUI_IMPORT_ERROR = None


@unittest.skipIf(GUI_IMPORT_ERROR is not None, f"GUI stack unavailable: {GUI_IMPORT_ERROR}")
class PointPickerAutoRealignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QApplication.instance() or QApplication([])
        self.panel = PointPickerPanel(enable_3d=False)
        self.panel.model_mesh = pv.Cube(center=(0.0, 0.0, 0.0), x_length=12.0, y_length=6.0, z_length=0.5)
        self.panel.model_mesh.rotate_y(18.0, inplace=True)
        self.panel.model_mesh.rotate_z(-27.0, inplace=True)
        self.panel.normal_axis_combo.setCurrentText("Z")
        self.panel.in_plane_axis_combo.setCurrentText("X")
        self.panel.picked_points = [("G1", np.array([1.2, -0.7, 0.4], dtype=float))]
        self.panel.refresh_points_table()

    def tearDown(self) -> None:
        self.panel.deleteLater()

    def test_preview_and_cancel_leave_active_frame_and_picks_unchanged(self) -> None:
        original_origin = self.panel.active_local_origin.copy()
        original_basis = self.panel.active_local_basis.copy()
        original_pick = self.panel.picked_points[0][1].copy()

        self.panel.suggest_auto_local_coordinate_frame()

        self.assertIsNotNone(self.panel.draft_local_origin)
        np.testing.assert_allclose(self.panel.active_local_origin, original_origin)
        np.testing.assert_allclose(self.panel.active_local_basis, original_basis)
        np.testing.assert_allclose(self.panel.picked_points[0][1], original_pick)

        self.panel.cancel_local_coordinate_frame_draft()

        self.assertIsNone(self.panel.draft_local_origin)
        np.testing.assert_allclose(self.panel.active_local_origin, original_origin)
        np.testing.assert_allclose(self.panel.active_local_basis, original_basis)
        np.testing.assert_allclose(self.panel.picked_points[0][1], original_pick)

    def test_confirm_applies_frame_without_moving_stored_picks(self) -> None:
        original_pick = self.panel.picked_points[0][1].copy()
        self.panel.suggest_auto_local_coordinate_frame()

        self.panel.confirm_local_coordinate_frame()

        self.assertTrue(self.panel.has_custom_local_coordinate_frame())
        np.testing.assert_allclose(
            self.panel.active_local_basis.T @ self.panel.active_local_basis,
            np.eye(3),
            atol=1e-10,
        )
        self.assertAlmostEqual(float(np.linalg.det(self.panel.active_local_basis)), 1.0, places=10)
        np.testing.assert_allclose(self.panel.picked_points[0][1], original_pick)
        self.assertEqual(self.panel.points_table.item(0, 0).text(), "G1")
        self.assertEqual(self.panel.points_table.rowCount(), 1)

    def test_auto_preview_accepts_repository_stl_meshes(self) -> None:
        model_dir = Path(__file__).resolve().parents[1] / "model"
        for filename in ("2_thin_pipe.stl", "big5.stl", "T_weld.stl"):
            mesh_path = model_dir / filename
            self.panel.set_mesh(load_mesh_as_polydata(mesh_path), mesh_path)

            self.panel.suggest_auto_local_coordinate_frame()

            self.assertIsNotNone(self.panel.draft_local_origin, filename)
            self.assertIsNotNone(self.panel.draft_local_axis_point, filename)
            self.assertIsNotNone(self.panel.draft_plane_normal, filename)
            if filename == "T_weld.stl":
                self.assertIn("review this principal-axis suggestion carefully", self.panel.status_label.text())
            self.panel.cancel_local_coordinate_frame_draft()


if __name__ == "__main__":
    unittest.main()
