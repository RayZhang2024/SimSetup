from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem

    from gui_app import MainWindow
except Exception as exc:  # pragma: no cover - depends on optional GUI stack
    GUI_IMPORT_ERROR = exc
else:
    GUI_IMPORT_ERROR = None


@unittest.skipIf(GUI_IMPORT_ERROR is not None, f"GUI stack unavailable: {GUI_IMPORT_ERROR}")
class PredictionInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QApplication.instance() or QApplication([])

    def _window_with_prediction_row(self, values: list[str]) -> MainWindow:
        window = MainWindow.__new__(MainWindow)
        table = QTableWidget(1, 4)
        for column, value in enumerate(values):
            table.setItem(0, column, QTableWidgetItem(value))
        table.setCurrentCell(0, 1)
        window.prediction_table = table
        return window

    def test_selected_prediction_row_ignores_transient_invalid_coordinate(self) -> None:
        window = self._window_with_prediction_row(["G1", "e90", "2", "3"])

        self.assertIsNone(window.selected_prediction_row())

        with self.assertRaisesRegex(ValueError, "invalid model coordinate"):
            window.selected_prediction_row(strict=True)

    def test_selected_prediction_row_returns_valid_coordinate(self) -> None:
        window = self._window_with_prediction_row(["G1", "1", "2", "3"])

        label, point = window.selected_prediction_row(strict=True)

        self.assertEqual(label, "G1")
        self.assertEqual(point.tolist(), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
