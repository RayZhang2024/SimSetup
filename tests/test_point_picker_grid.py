from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import pyvista as pv
    from PyQt5.QtWidgets import QApplication

    from gui_app import PointPickerPanel
except Exception as exc:  # pragma: no cover - depends on optional GUI stack
    GUI_IMPORT_ERROR = exc
else:
    GUI_IMPORT_ERROR = None


@unittest.skipIf(GUI_IMPORT_ERROR is not None, f"GUI stack unavailable: {GUI_IMPORT_ERROR}")
class PointPickerGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QApplication.instance() or QApplication([])
        self.panel = PointPickerPanel(enable_3d=False)
        self.panel.model_mesh = pv.Cube(center=(0.0, 0.0, 0.0), x_length=10.0, y_length=10.0, z_length=10.0)
        self.panel.plane_axis_combo.setCurrentText("X")
        self.panel.plane_value_spin.setValue(0.0)
        self.panel.grid_enabled_checkbox.setChecked(True)
        self.panel.grid_snap_checkbox.setChecked(True)
        self.panel.grid_u_size_spin.setValue(10.0)
        self.panel.grid_v_size_spin.setValue(10.0)
        self.panel.grid_u_spacing_spin.setValue(1.0)
        self.panel.grid_v_spacing_spin.setValue(1.0)

    def tearDown(self) -> None:
        self.panel.deleteLater()

    def test_grid_pick_snaps_to_moved_grid_coordinates(self) -> None:
        self.panel.grid_offset_u_spin.setValue(1.0)
        self.panel.grid_offset_v_spin.setValue(-2.0)

        snapped = self.panel.grid_projected_pick_point(np.array([0.0, 3.4, 1.2], dtype=float))

        np.testing.assert_allclose(snapped, np.array([0.0, 3.0, 1.0], dtype=float))

    def test_grid_rotation_changes_pick_axes(self) -> None:
        self.panel.grid_rotation_spin.setValue(90.0)

        snapped = self.panel.grid_projected_pick_point(np.array([0.0, -2.2, 1.6], dtype=float))

        np.testing.assert_allclose(snapped, np.array([0.0, -2.0, 2.0], dtype=float), atol=1e-12)

    def test_plane_pick_adds_point_to_table(self) -> None:
        self.panel.on_plane_picked(np.array([0.0, 1.2, -2.4], dtype=float))

        self.assertEqual(len(self.panel.picked_points), 1)
        self.assertEqual(self.panel.points_table.rowCount(), 1)
        self.assertEqual(self.panel.points_table.item(0, 0).text(), "G1")
        self.assertEqual(self.panel.points_table.item(0, 1).text(), "0.000")
        self.assertEqual(self.panel.points_table.item(0, 2).text(), "1.000")
        self.assertEqual(self.panel.points_table.item(0, 3).text(), "-2.000")


if __name__ == "__main__":
    unittest.main()
