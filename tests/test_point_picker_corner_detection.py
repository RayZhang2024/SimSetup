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
class PointPickerCornerDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QApplication.instance() or QApplication([])
        self.panel = PointPickerPanel(enable_3d=False)
        self.panel.model_mesh = pv.Cube(center=(0.0, 0.0, 0.0), x_length=10.0, y_length=6.0, z_length=2.0)
        self.panel.picked_points = [("G1", np.array([0.0, 0.0, 0.0], dtype=float))]
        self.panel.refresh_points_table()

    def tearDown(self) -> None:
        self.panel.deleteLater()

    def test_detect_lists_corner_candidates_and_preserves_manual_points(self) -> None:
        self.panel.detect_corner_candidates()

        self.assertEqual(self.panel.points_table.rowCount(), 9)
        labels = [self.panel.picked_points[row][0] for row in range(len(self.panel.picked_points))]
        self.assertEqual(labels, ["G1"] + [f"C{index}" for index in range(1, 9)])
        self.assertIn("Detected 8 corner candidates", self.panel.status_label.text())

    def test_repeated_detection_replaces_candidate_rows_without_duplicates(self) -> None:
        self.panel.detect_corner_candidates()
        first_candidates = [
            point.copy()
            for label, point in self.panel.picked_points
            if self.panel.is_corner_candidate_label(label)
        ]

        self.panel.detect_corner_candidates()

        second_candidates = [
            point
            for label, point in self.panel.picked_points
            if self.panel.is_corner_candidate_label(label)
        ]
        self.assertEqual(len(second_candidates), 8)
        self.assertEqual(self.panel.points_table.rowCount(), 9)
        np.testing.assert_allclose(second_candidates, first_candidates)

    def test_manual_points_keep_g_labels_after_corner_candidates_are_added(self) -> None:
        self.panel.detect_corner_candidates()
        self.panel.require_inside_checkbox.setChecked(False)

        self.panel.add_point(np.array([1.234, 1.234, 0.0], dtype=float))

        self.assertEqual(self.panel.picked_points[0][0], "G1")
        self.assertEqual(self.panel.picked_points[-1][0], "G2")
        self.assertEqual(sum(self.panel.is_corner_candidate_label(label) for label, _point in self.panel.picked_points), 8)


if __name__ == "__main__":
    unittest.main()
