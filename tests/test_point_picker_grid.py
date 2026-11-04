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

    def test_grid_axes_follow_rotated_local_frame(self) -> None:
        angle = np.deg2rad(37.0)
        rotation_z = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        angle_y = np.deg2rad(-21.0)
        rotation_y = np.array(
            [
                [np.cos(angle_y), 0.0, np.sin(angle_y)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle_y), 0.0, np.cos(angle_y)],
            ],
            dtype=float,
        )
        self.panel.active_local_basis = rotation_z @ rotation_y
        self.panel.plane_axis_combo.setCurrentText("X")
        origin, normal, _plane_size = self.panel.current_plane_definition()

        _center, axis_u, axis_v, *_rest = self.panel.current_grid_definition(origin, normal)

        np.testing.assert_allclose(axis_u, self.panel.active_local_basis[:, 1], atol=1e-12)
        np.testing.assert_allclose(axis_v, self.panel.active_local_basis[:, 2], atol=1e-12)
        self.assertAlmostEqual(float(np.dot(axis_u, normal)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(axis_v, normal)), 0.0, places=12)

    def test_grid_uses_local_fallback_axis_when_normal_is_local_z(self) -> None:
        angle = np.deg2rad(29.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.panel.active_local_basis = rotation
        self.panel.plane_axis_combo.setCurrentText("Z")
        origin, normal, _plane_size = self.panel.current_plane_definition()

        _center, axis_u, axis_v, *_rest = self.panel.current_grid_definition(origin, normal)

        np.testing.assert_allclose(axis_u, -self.panel.active_local_basis[:, 1], atol=1e-12)
        np.testing.assert_allclose(axis_v, self.panel.active_local_basis[:, 0], atol=1e-12)

    def test_grid_preview_uses_draft_frame_before_confirmation(self) -> None:
        self.panel.normal_axis_combo.setCurrentText("X")
        self.panel.in_plane_axis_combo.setCurrentText("Y")
        self.panel.plane_axis_combo.setCurrentText("X")
        angle = np.deg2rad(41.0)
        draft_normal = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)
        draft_in_plane = np.array([-np.sin(angle), np.cos(angle), 0.0], dtype=float)
        self.panel.draft_local_origin = np.array([1.0, -2.0, 0.5], dtype=float)
        self.panel.draft_plane_normal = draft_normal
        self.panel.draft_local_axis_point = self.panel.draft_local_origin + draft_in_plane

        origin, normal, _plane_size = self.panel.current_plane_definition()
        _center, axis_u, axis_v, *_rest = self.panel.current_grid_definition(origin, normal)
        _draft_origin, draft_basis = self.panel.displayed_local_coordinate_frame()

        np.testing.assert_allclose(normal, draft_basis[:, 0], atol=1e-12)
        np.testing.assert_allclose(axis_u, draft_basis[:, 1], atol=1e-12)
        np.testing.assert_allclose(axis_v, draft_basis[:, 2], atol=1e-12)
        np.testing.assert_allclose(self.panel.active_local_basis, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.dot(axis_u, normal)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(axis_v, normal)), 0.0, places=12)

    def test_axis_view_buttons_use_draft_frame_before_confirmation(self) -> None:
        angle = np.deg2rad(36.0)
        rotation_z = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        angle_y = np.deg2rad(24.0)
        rotation_y = np.array(
            [
                [np.cos(angle_y), 0.0, np.sin(angle_y)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle_y), 0.0, np.cos(angle_y)],
            ],
            dtype=float,
        )
        draft_basis = rotation_z @ rotation_y
        self.panel.draft_local_origin = np.array([1.0, -2.0, 0.5], dtype=float)
        self.panel.draft_plane_normal = draft_basis[:, 0]
        self.panel.draft_local_axis_point = self.panel.draft_local_origin + draft_basis[:, 1]

        class FakeCamera:
            def SetParallelScale(self, _scale: float) -> None:
                pass

        class FakePlotter:
            def __init__(self) -> None:
                self.camera = FakeCamera()
                self.camera_position = None

            def enable_parallel_projection(self) -> None:
                pass

            def reset_camera_clipping_range(self) -> None:
                pass

            def render(self) -> None:
                pass

        self.panel.plotter = FakePlotter()

        self.panel.view_axis("y")

        camera_position = self.panel.plotter.camera_position
        view_direction = np.asarray(camera_position[0]) - np.asarray(camera_position[1])
        view_direction /= np.linalg.norm(view_direction)
        np.testing.assert_allclose(view_direction, draft_basis[:, 1], atol=1e-12)
        np.testing.assert_allclose(camera_position[2], draft_basis[:, 2], atol=1e-12)
        np.testing.assert_allclose(self.panel.active_local_basis, np.eye(3), atol=1e-12)

    def test_plane_pick_adds_point_to_table(self) -> None:
        self.panel.on_plane_picked(np.array([0.0, 1.2, -2.4], dtype=float))

        self.assertEqual(len(self.panel.picked_points), 1)
        self.assertEqual(self.panel.points_table.rowCount(), 1)
        self.assertEqual(self.panel.points_table.item(0, 0).text(), "G1")
        self.assertEqual(self.panel.points_table.item(0, 1).text(), "0.000")
        self.assertEqual(self.panel.points_table.item(0, 2).text(), "1.000")
        self.assertEqual(self.panel.points_table.item(0, 3).text(), "-2.000")

    def test_show_whole_model_bypasses_slice_clipping(self) -> None:
        origin, normal, _plane_size = self.panel.current_plane_definition()
        clipped = self.panel.mesh_for_slice_display(origin, normal)

        self.assertFalse(self.panel.show_whole_model_checkbox.isChecked())
        self.assertAlmostEqual(clipped.bounds[0], 0.0, places=6)
        self.assertAlmostEqual(clipped.bounds[1], 5.0, places=6)

        self.panel.show_whole_model_checkbox.setChecked(True)
        whole = self.panel.mesh_for_slice_display(origin, normal)

        self.assertIs(whole, self.panel.model_mesh)
        self.assertEqual(whole.bounds, self.panel.model_mesh.bounds)


if __name__ == "__main__":
    unittest.main()
