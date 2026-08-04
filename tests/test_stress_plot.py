from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QTableWidgetSelectionRange

    from gui_app import MainWindow, STRESS_TABLE_HEADERS, SpreadsheetTableWidget, StressTableHeaderView
except Exception as exc:  # pragma: no cover - depends on optional GUI stack
    GUI_IMPORT_ERROR = exc
else:
    GUI_IMPORT_ERROR = None


@unittest.skipIf(GUI_IMPORT_ERROR is not None, f"GUI stack unavailable: {GUI_IMPORT_ERROR}")
class StressPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QApplication.instance() or QApplication([])
        self.window = MainWindow.__new__(MainWindow)
        table = SpreadsheetTableWidget(3, len(STRESS_TABLE_HEADERS))
        table.setHorizontalHeader(StressTableHeaderView(Qt.Horizontal, table))
        table.setHorizontalHeaderLabels(STRESS_TABLE_HEADERS)
        self.window.stress_table = table

    def _set_cell(self, row: int, column: int, value: str) -> None:
        self.window.stress_table.setItem(row, column, QTableWidgetItem(value))

    def test_stress_plot_data_for_single_column_uses_row_numbers(self) -> None:
        self._set_cell(0, 20, "84")
        self._set_cell(1, 20, "")
        self._set_cell(2, 20, "not-number")

        x_label, y_label, x_values, y_values, skipped = self.window.stress_plot_data_for_columns([20])

        self.assertEqual(x_label, "Row")
        self.assertEqual(y_label, "Stress 1 (MPA) σ")
        self.assertEqual(x_values, [1.0])
        self.assertEqual(y_values, [84.0])
        self.assertEqual(skipped, 2)

    def test_stress_plot_data_for_two_columns_pairs_numeric_rows(self) -> None:
        self._set_cell(0, 14, "420")
        self._set_cell(0, 20, "84")
        self._set_cell(1, 14, "-192")
        self._set_cell(1, 20, "")
        self._set_cell(2, 14, "528")
        self._set_cell(2, 20, "109")

        x_label, y_label, x_values, y_values, skipped = self.window.stress_plot_data_for_columns([14, 20])

        self.assertEqual(x_label, "Strain 1 (µε) ε")
        self.assertEqual(y_label, "Stress 1 (MPA) σ")
        self.assertEqual(x_values, [420.0, 528.0])
        self.assertEqual(y_values, [84.0, 109.0])
        self.assertEqual(skipped, 1)

    def test_stress_map_seed_rows_uses_numeric_selected_value_column(self) -> None:
        self._set_cell(0, 20, "84")
        self._set_cell(1, 20, "")
        self._set_cell(2, 20, "not-number")

        values = self.window.stress_map_seed_rows(20)

        self.assertEqual(values, [(0, 84.0)])

    def test_parse_stress_map_table_skips_incomplete_rows(self) -> None:
        table = QTableWidget(3, 3)
        for row, values in enumerate((("0", "0", "84"), ("1", "", "95"), ("2", "3", "109"))):
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))

        x_values, y_values, z_values, skipped = self.window.parse_stress_map_table(table)

        self.assertEqual(x_values, [0.0, 2.0])
        self.assertEqual(y_values, [0.0, 3.0])
        self.assertEqual(z_values, [84.0, 109.0])
        self.assertEqual(skipped, 1)

    def test_aggregate_stress_map_points_averages_duplicate_xy(self) -> None:
        x_values, y_values, z_values = self.window.aggregate_stress_map_points(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [10.0, 14.0, 20.0],
        )

        self.assertEqual(x_values.tolist(), [0.0, 1.0])
        self.assertEqual(y_values.tolist(), [0.0, 1.0])
        self.assertEqual(z_values.tolist(), [12.0, 20.0])

    def test_mapping_table_paste_allows_all_columns(self) -> None:
        table = SpreadsheetTableWidget(2, 3)
        table.setItem(0, 2, QTableWidgetItem("84"))
        table.setCurrentCell(0, 0)

        clipboard = QApplication.clipboard()
        previous_text = clipboard.text()
        try:
            clipboard.setText("1\t2\t999\n3\t4\t888")
            table.paste_from_clipboard()

            self.assertEqual(table.item(0, 0).text(), "1")
            self.assertEqual(table.item(0, 1).text(), "2")
            self.assertEqual(table.item(0, 2).text(), "999")
            self.assertEqual(table.item(1, 0).text(), "3")
            self.assertEqual(table.item(1, 1).text(), "4")
            self.assertEqual(table.item(1, 2).text(), "888")
        finally:
            clipboard.setText(previous_text)

    def test_mapping_table_paste_extends_rows_for_more_data(self) -> None:
        table = SpreadsheetTableWidget(1, 3)
        table.setCurrentCell(0, 0)

        clipboard = QApplication.clipboard()
        previous_text = clipboard.text()
        try:
            clipboard.setText("1\t2\t3\n4\t5\t6")
            table.paste_from_clipboard()

            self.assertEqual(table.rowCount(), 2)
            self.assertEqual(table.item(1, 0).text(), "4")
            self.assertEqual(table.item(1, 1).text(), "5")
            self.assertEqual(table.item(1, 2).text(), "6")
        finally:
            clipboard.setText(previous_text)

    def test_mapping_table_clear_selected_cells(self) -> None:
        table = SpreadsheetTableWidget(2, 3)
        for row in range(2):
            for column in range(3):
                table.setItem(row, column, QTableWidgetItem(f"{row}:{column}"))

        table.setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 2), True)
        table.clear_selected_cells()

        for row in range(2):
            for column in range(3):
                self.assertEqual(table.item(row, column).text(), "")


if __name__ == "__main__":
    unittest.main()
