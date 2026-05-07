#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import trimesh
from PyQt5.QtCore import QEvent, QPointF, QRect, QSize, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QAction,
    QMessageBox,
    QMenuBar,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QStackedWidget,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.vtkCommonTransforms import vtkTransform

try:
    from astropy.io import fits
except ImportError:
    fits = None

try:
    import tifffile
except ImportError:
    tifffile = None

from placement_fit import (
    Measurement,
    Transform,
    fit_from_measurements,
    infer_stage_point_from_readout,
    pretty_vector,
    rotation_matrix_to_euler_zyx_deg,
    rotation_z_deg,
    save_json_report,
)


if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
    pv.OFF_SCREEN = True


TABLE_HEADERS = [
    "Label",
    "Model X",
    "Model Y",
    "Model Z",
    "Stage X",
    "Stage Y",
    "Stage Z",
]

PREDICTION_TABLE_HEADERS = [
    "Label",
    "Model X",
    "Model Y",
    "Model Z",
    "Stage X",
    "Stage Y",
    "Stage Z",
    "Path 1",
    "Path 2",
    "UAmp1",
    "UAmp2",
]

STRESS_TABLE_HEADERS = [
    "Run",
    "Point ID",
    "Lattice parameter 1",
    "uncert 1",
    "D0 1",
    "D0 uncert 1",
    "Lattice parameter 2",
    "uncert 2",
    "D0 2",
    "D0 uncert 2",
    "Lattice parameter 3",
    "uncert 3",
    "D0 3",
    "D0 uncert 3",
    "Strain 1",
    "Strain 1 uncert",
    "Strain 2",
    "Strain 2 uncert",
    "Strain 3",
    "Strain 3 uncert",
    "Stress 1",
    "Stress 1 uncert",
    "Stress 2",
    "Stress 2 uncert",
    "Stress 3",
    "Stress 3 uncert",
]

PROJECT_ARCHIVE_VERSION = 1
PROJECT_MANIFEST_NAME = "project.json"
PROJECT_EMBEDDED_MESH_NAME = "mesh.stl"
PROJECT_FILE_FILTER = "SimSetup project (*.simsetup);;All files (*.*)"
MICROSTRAIN_SCALE = 1_000_000.0
DEFAULT_UI_FONT_POINT_SIZE = 7.0
MIN_UI_FONT_POINT_SIZE = 3.0

RIETVELD_COUNT_TIME_LAWS = {
    "Al": np.array([0.8, 0.9, 1.1, 1.2], dtype=float),
    "Fe, FCC": np.array([0.1, 0.8, 7.4, 22.5], dtype=float),
    "Fe, BCC": np.array([0.1, 0.8, 6.8, 20.4], dtype=float),
    "Ni": np.array([0.2, 3.9, 216.2, 1602.9], dtype=float),
}
COUNT_TIME_REFERENCE_PATHS_MM = np.array([4.0, 20.0, 40.0, 50.0], dtype=float)

DIRECT_DETECTOR_CENTER_WORLD = np.array([700.0, 0.0, 0.0], dtype=float)
DIFFRACTION_BANK_1_CENTER_WORLD = np.array([0.0, 1000.0, 0.0], dtype=float)
DIFFRACTION_BANK_2_CENTER_WORLD = np.array([0.0, -1000.0, 0.0], dtype=float)
ANGLED_DETECTOR_HORIZONTAL_HALF_ANGLE_DEG = 14.0
ANGLED_DETECTOR_VERTICAL_HALF_ANGLE_DEG = 21.0
DIFFRACTION_ANGLE_INTERVAL_DEG = 1.0
DETECTOR_THICKNESS = 10.0


def format_decimal(value: float) -> str:
    return f"{value:.3f}"


def format_trimmed_decimal(value: float, decimals: int = 6) -> str:
    if decimals <= 0:
        return str(int(round(float(value))))
    text = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def format_fixed_decimal(value: float, decimals: int = 6) -> str:
    if decimals <= 0:
        return str(int(round(float(value))))
    return f"{float(value):.{decimals}f}"


def current_application_ui_font_size(app: QApplication) -> float:
    font = QFont(app.font())
    if font.pointSizeF() > 0:
        return float(font.pointSizeF())
    if font.pointSize() > 0:
        return float(font.pointSize())
    return DEFAULT_UI_FONT_POINT_SIZE


def apply_application_ui_font(app: QApplication, point_size: float = DEFAULT_UI_FONT_POINT_SIZE) -> None:
    target_size = max(MIN_UI_FONT_POINT_SIZE, min(18.0, float(point_size)))
    font = QFont(app.font())
    if font.pointSizeF() > 0:
        font.setPointSizeF(target_size)
    else:
        font.setPointSize(int(round(target_size)))
    app.setFont(font)


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelComboBox(QComboBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMaximumWidth(150)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def wheelEvent(self, event) -> None:
        event.ignore()


def make_spin_box(value: float, minimum: float, maximum: float, step: float = 1.0) -> QDoubleSpinBox:
    box = NoWheelDoubleSpinBox()
    box.setDecimals(3)
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setSingleStep(step)
    box.setKeyboardTracking(False)
    box.setMaximumWidth(150)
    box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return box


def make_form_layout(parent: Optional[QWidget] = None, *, compact_fields: bool = False) -> QFormLayout:
    layout = QFormLayout(parent)
    if compact_fields:
        layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    return layout


class FlowLayout(QLayout):
    def __init__(self, parent: Optional[QWidget] = None, margin: int = 0, spacing: int = 0) -> None:
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        size += QSize(left + right, top + bottom)
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        left, top, right, bottom = self.getContentsMargins()
        effective_rect = rect.adjusted(left, top, -right, -bottom)
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._items:
            item_size = item.sizeHint()
            next_x = x + item_size.width() + spacing
            if x > effective_rect.x() and next_x - spacing > effective_rect.right() + 1:
                x = effective_rect.x()
                y += line_height + spacing
                next_x = x + item_size.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(x, y, item_size.width(), item_size.height()))
            x = next_x
            line_height = max(line_height, item_size.height())

        return y + line_height - rect.y() + bottom


def render_toolbar_toggle_icon(kind: str, color: QColor, size: int = 24) -> QPixmap:
    pixel_ratio = 3.0
    pixmap = QPixmap(int(round(size * pixel_ratio)), int(round(size * pixel_ratio)))
    pixmap.setDevicePixelRatio(pixel_ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    pen = QPen(color, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    c = size / 2.0

    def draw_arrow(start: QPointF, end: QPointF, head_a: QPointF, head_b: QPointF) -> None:
        painter.drawLine(start, end)
        painter.drawLine(end, head_a)
        painter.drawLine(end, head_b)

    def draw_corner_label(text: str, align: int = Qt.AlignTop | Qt.AlignLeft) -> None:
        font = QFont()
        font.setBold(True)
        font.setPixelSize(8)
        painter.save()
        painter.setFont(font)
        painter.drawText(QRect(2, 2, size - 4, size - 4), align, text)
        painter.restore()

    def draw_center_label(text: str) -> None:
        font = QFont()
        font.setBold(True)
        font.setPixelSize(10)
        painter.save()
        painter.setFont(font)
        painter.drawText(QRect(2, 2, size - 4, size - 4), Qt.AlignCenter, text)
        painter.restore()

    if kind == "parallel":
        painter.drawLine(int(c - 4), 6, int(c - 4), size - 6)
        painter.drawLine(int(c + 4), 6, int(c + 4), size - 6)
    elif kind == "stage":
        top = QPolygonF(
            [
                QPointF(5, c - 2),
                QPointF(c, 5),
                QPointF(size - 5, c - 2),
                QPointF(c, c + 3),
            ]
        )
        front = QPolygonF(
            [
                QPointF(5, c - 2),
                QPointF(5, size - 8),
                QPointF(c, size - 5),
                QPointF(c, c + 3),
            ]
        )
        side = QPolygonF(
            [
                QPointF(c, c + 3),
                QPointF(c, size - 5),
                QPointF(size - 5, size - 8),
                QPointF(size - 5, c - 2),
            ]
        )
        painter.drawPolygon(top)
        painter.drawPolygon(front)
        painter.drawPolygon(side)
    elif kind == "beam":
        font = QFont()
        font.setBold(True)
        font.setPixelSize(14)
        painter.save()
        painter.setFont(font)
        painter.drawText(QRect(0, int(c - 9), 11, 16), Qt.AlignCenter, "n")
        painter.restore()
        painter.drawLine(QPointF(9, c - 2), QPointF(size - 2, c - 2))
        painter.drawLine(QPointF(9, c + 2), QPointF(size - 2, c + 2))
    elif kind == "cube":
        front = QRect(5, 8, 9, 9)
        back = QRect(8, 5, 9, 9)
        painter.drawRect(front)
        painter.drawRect(back)
        painter.drawLine(front.topLeft(), back.topLeft())
        painter.drawLine(front.topRight(), back.topRight())
        painter.drawLine(front.bottomLeft(), back.bottomLeft())
        painter.drawLine(front.bottomRight(), back.bottomRight())
    elif kind == "imaging":
        front = QRect(4, 7, 11, 11)
        back_top_left = QPointF(8, 3)
        back_top_right = QPointF(size - 4, 3)
        back_bottom_right = QPointF(size - 4, size - 8)
        painter.drawRect(front)
        painter.drawRect(QRect(7, 10, 5, 5))
        painter.drawLine(front.topLeft(), back_top_left)
        painter.drawLine(front.topRight(), back_top_right)
        painter.drawLine(front.bottomRight(), back_bottom_right)
        painter.drawLine(back_top_left, back_top_right)
        painter.drawLine(back_top_right, back_bottom_right)
    elif kind == "features":
        painter.setBrush(color)
        for point in ((6, 7), (15, 6), (10, 12), (16, 16), (6, 16)):
            painter.drawEllipse(QPointF(point[0], point[1]), 2.2, 2.2)
    elif kind == "predicted":
        painter.drawEllipse(QPointF(c, c), 7.0, 7.0)
        painter.drawEllipse(QPointF(c, c), 3.0, 3.0)
        painter.drawLine(int(c), 3, int(c), 7)
        painter.drawLine(int(c), size - 7, int(c), size - 3)
        painter.drawLine(3, int(c), 7, int(c))
        painter.drawLine(size - 7, int(c), size - 3, int(c))
    elif kind == "triad":
        origin = QPointF(6, size - 6)
        painter.drawLine(origin, QPointF(size - 5, size - 6))
        painter.drawLine(origin, QPointF(6, 5))
        painter.drawLine(origin, QPointF(size - 6, 6))
        painter.drawLine(size - 8, size - 8, size - 5, size - 6)
        painter.drawLine(size - 8, size - 4, size - 5, size - 6)
        painter.drawLine(4, 8, 6, 5)
        painter.drawLine(8, 8, 6, 5)
        painter.drawLine(size - 8, 9, size - 6, 6)
        painter.drawLine(size - 10, 6, size - 6, 6)
    elif kind == "sight":
        painter.drawEllipse(QPointF(6, c), 2.5, 2.5)
        painter.drawLine(8, int(c), size - 8, int(c))
        painter.drawEllipse(QPointF(size - 6, c), 4.0, 4.0)
        painter.drawLine(size - 10, int(c), size - 2, int(c))
        painter.drawLine(size - 6, int(c - 4), size - 6, int(c + 4))
    elif kind == "diffraction":
        front = QRect(4, 8, 9, 9)
        back_top_left = QPointF(1, 4)
        back_top_right = QPointF(size - 5, 4)
        back_bottom_left = QPointF(1, size - 4)
        back_bottom_right = QPointF(size - 5, size - 4)
        painter.drawRect(front)
        painter.drawLine(front.topLeft(), back_top_left)
        painter.drawLine(front.topRight(), back_top_right)
        painter.drawLine(front.bottomLeft(), back_bottom_left)
        painter.drawLine(front.bottomRight(), back_bottom_right)
        painter.drawLine(back_top_left, back_top_right)
        painter.drawLine(back_top_right, back_bottom_right)
        painter.drawLine(back_bottom_left, back_bottom_right)
    elif kind == "cam_iso":
        top = QPolygonF(
            [
                QPointF(7, c - 4),
                QPointF(c, 5),
                QPointF(size - 7, c - 4),
                QPointF(c, c + 1),
            ]
        )
        front = QPolygonF(
            [
                QPointF(7, c - 4),
                QPointF(7, size - 8),
                QPointF(c, size - 5),
                QPointF(c, c + 1),
            ]
        )
        side = QPolygonF(
            [
                QPointF(c, c + 1),
                QPointF(c, size - 5),
                QPointF(size - 7, size - 8),
                QPointF(size - 7, c - 4),
            ]
        )
        painter.drawPolygon(top)
        painter.drawPolygon(front)
        painter.drawPolygon(side)
        painter.drawLine(QPointF(c, c + 1), QPointF(c, 3))
        painter.drawLine(QPointF(c, c + 1), QPointF(size - 3, c + 1))
        painter.drawLine(QPointF(c, c + 1), QPointF(4, size - 4))
    elif kind == "cam_px":
        draw_center_label("+X")
    elif kind == "cam_nx":
        draw_center_label("-X")
    elif kind == "cam_py":
        draw_center_label("+Y")
    elif kind == "cam_ny":
        draw_center_label("-Y")
    elif kind == "cam_pz":
        draw_center_label("+Z")
    elif kind == "cam_nz":
        draw_center_label("-Z")
    elif kind == "cam_theodolite":
        painter.drawEllipse(QPointF(c, 8), 3.5, 3.5)
        painter.drawLine(QPointF(c, 11.5), QPointF(c, size - 4))
    elif kind == "act_add":
        painter.drawLine(QPointF(c, 5), QPointF(c, size - 5))
        painter.drawLine(QPointF(5, c), QPointF(size - 5, c))
    elif kind == "act_remove":
        painter.drawLine(QPointF(5, c), QPointF(size - 5, c))
    elif kind == "act_clear_placement":
        painter.drawEllipse(QPointF(c, c), 6.5, 6.5)
        painter.drawLine(QPointF(c, 3), QPointF(c, 7))
        painter.drawLine(QPointF(c, size - 7), QPointF(c, size - 3))
        painter.drawLine(QPointF(3, c), QPointF(7, c))
        painter.drawLine(QPointF(size - 7, c), QPointF(size - 3, c))
        painter.drawLine(QPointF(c - 4, c - 4), QPointF(c + 4, c + 4))
        painter.drawLine(QPointF(c + 4, c - 4), QPointF(c - 4, c + 4))
    elif kind == "act_load":
        painter.drawRect(QRect(5, 8, size - 10, size - 11))
        painter.drawLine(QPointF(c, 4), QPointF(c, 13))
        painter.drawLine(QPointF(c, 13), QPointF(c - 4, 9))
        painter.drawLine(QPointF(c, 13), QPointF(c + 4, 9))
    elif kind == "act_project_load":
        painter.drawRect(QRect(5, 7, size - 10, size - 10))
        painter.drawLine(QPointF(7, 11), QPointF(size - 7, 11))
        painter.drawLine(QPointF(9, 15), QPointF(size - 9, 15))
        draw_arrow(QPointF(c, 3), QPointF(c, size - 5), QPointF(c - 4, size - 9), QPointF(c + 4, size - 9))
    elif kind == "act_project_save":
        painter.drawRect(QRect(5, 5, size - 10, size - 10))
        painter.drawLine(QPointF(8, 8), QPointF(size - 8, 8))
        painter.drawRect(QRect(8, size - 12, size - 16, 5))
        painter.drawLine(QPointF(size - 9, 5), QPointF(size - 5, 9))
    elif kind == "act_mesh_load":
        top = QPolygonF(
            [
                QPointF(6, c - 3),
                QPointF(c, 6),
                QPointF(size - 6, c - 3),
                QPointF(c, c + 2),
            ]
        )
        front = QPolygonF(
            [
                QPointF(6, c - 3),
                QPointF(6, size - 8),
                QPointF(c, size - 5),
                QPointF(c, c + 2),
            ]
        )
        side = QPolygonF(
            [
                QPointF(c, c + 2),
                QPointF(c, size - 5),
                QPointF(size - 6, size - 8),
                QPointF(size - 6, c - 3),
            ]
        )
        painter.drawPolygon(top)
        painter.drawPolygon(front)
        painter.drawPolygon(side)
        draw_arrow(QPointF(c, 3), QPointF(c, c - 2), QPointF(c - 3, c - 5), QPointF(c + 3, c - 5))
    elif kind == "act_table_load":
        painter.drawRect(QRect(5, 7, size - 10, size - 8))
        painter.drawLine(QPointF(5, 12), QPointF(size - 5, 12))
        painter.drawLine(QPointF(5, 17), QPointF(size - 5, 17))
        painter.drawLine(QPointF(11, 7), QPointF(11, size - 1))
        draw_arrow(QPointF(c, 3), QPointF(c, 11), QPointF(c - 4, 7), QPointF(c + 4, 7))
    elif kind == "act_table_save":
        painter.drawRect(QRect(5, 5, size - 10, size - 10))
        painter.drawLine(QPointF(5, 11), QPointF(size - 5, 11))
        painter.drawLine(QPointF(11, 5), QPointF(11, size - 5))
        painter.drawRect(QRect(9, size - 10, size - 18, 4))
    elif kind == "act_save":
        painter.drawRect(QRect(5, 5, size - 10, size - 10))
        painter.drawLine(QPointF(8, 8), QPointF(size - 8, 8))
        painter.drawRect(QRect(8, size - 12, size - 16, 5))
    elif kind == "act_move_pivot":
        painter.drawEllipse(QPointF(c, c), 6.5, 6.5)
        painter.drawLine(QPointF(c, 3), QPointF(c, 7))
        painter.drawLine(QPointF(c, size - 7), QPointF(c, size - 3))
        painter.drawLine(QPointF(3, c), QPointF(7, c))
        painter.drawLine(QPointF(size - 7, c), QPointF(size - 3, c))
        painter.drawLine(QPointF(5, size - 5), QPointF(c - 2, c + 2))
        painter.drawLine(QPointF(c - 2, c + 2), QPointF(c - 4, c + 5))
        painter.drawLine(QPointF(c - 2, c + 2), QPointF(c + 2, c + 3))
    elif kind == "act_readouts":
        painter.drawRect(QRect(4, 5, 9, size - 10))
        painter.drawLine(QPointF(4, 11), QPointF(13, 11))
        painter.drawLine(QPointF(4, 17), QPointF(13, 17))
        painter.drawLine(QPointF(8.5, 5), QPointF(8.5, size - 5))
        draw_arrow(QPointF(15, c), QPointF(size - 4, c), QPointF(size - 8, c - 4), QPointF(size - 8, c + 4))
    elif kind == "act_paths":
        painter.drawLine(QPointF(5, c), QPointF(10, c))
        painter.drawLine(QPointF(10, c), QPointF(size - 6, 6))
        painter.drawLine(QPointF(10, c), QPointF(size - 6, size - 6))
        painter.drawLine(QPointF(size - 10, 7), QPointF(size - 6, 6))
        painter.drawLine(QPointF(size - 9, 10), QPointF(size - 6, 6))
        painter.drawLine(QPointF(size - 10, size - 7), QPointF(size - 6, size - 6))
        painter.drawLine(QPointF(size - 9, size - 10), QPointF(size - 6, size - 6))
    elif kind == "act_detector_map":
        painter.drawRect(QRect(4, 6, size - 8, size - 12))
        painter.drawLine(QPointF(8, 10), QPointF(size - 8, 10))
        painter.drawLine(QPointF(8, 14), QPointF(size - 8, 14))
        painter.drawLine(QPointF(c, 6), QPointF(c, size - 6))
        painter.setBrush(color)
        painter.drawEllipse(QPointF(size - 8, size - 8), 2.3, 2.3)
    elif kind == "act_diffraction_map":
        painter.drawLine(QPointF(4, c), QPointF(c - 1, c))
        painter.drawLine(QPointF(c - 1, c), QPointF(size - 5, 6))
        painter.drawLine(QPointF(c - 1, c), QPointF(size - 5, size - 6))
        painter.drawEllipse(QPointF(c - 1, c), 2.2, 2.2)
        painter.drawLine(QPointF(size - 9, 7), QPointF(size - 5, 6))
        painter.drawLine(QPointF(size - 8, 10), QPointF(size - 5, 6))
        painter.drawLine(QPointF(size - 9, size - 7), QPointF(size - 5, size - 6))
        painter.drawLine(QPointF(size - 8, size - 10), QPointF(size - 5, size - 6))
    elif kind == "diffraction_vectors":
        painter.drawEllipse(QPointF(c, c), 3.0, 3.0)
        draw_arrow(QPointF(c, c), QPointF(size - 5, 6), QPointF(size - 9, 6), QPointF(size - 6, 10))
        draw_arrow(QPointF(c, c), QPointF(size - 5, size - 6), QPointF(size - 9, size - 6), QPointF(size - 6, size - 10))
        draw_arrow(QPointF(c, c), QPointF(5, c), QPointF(9, c - 4), QPointF(9, c + 4))
    elif kind == "act_time":
        painter.drawEllipse(QPointF(c, c), 7.0, 7.0)
        painter.drawLine(QPointF(c, c), QPointF(c, 8))
        painter.drawLine(QPointF(c, c), QPointF(size - 8, c))
    elif kind == "act_scan":
        painter.drawRect(QRect(6, 4, size - 11, size - 8))
        painter.drawLine(QPointF(size - 9, 4), QPointF(size - 5, 8))
        painter.drawLine(QPointF(9, 10), QPointF(size - 9, 10))
        painter.drawLine(QPointF(9, 14), QPointF(size - 9, 14))
        painter.drawLine(QPointF(9, 18), QPointF(size - 12, 18))
    elif kind == "act_mesh_clear":
        top = QPolygonF(
            [
                QPointF(7, c - 3),
                QPointF(c, 6),
                QPointF(size - 7, c - 3),
                QPointF(c, c + 1),
            ]
        )
        painter.drawPolygon(top)
        painter.drawLine(QPointF(7, c - 3), QPointF(7, size - 8))
        painter.drawLine(QPointF(size - 7, c - 3), QPointF(size - 7, size - 8))
        painter.drawLine(QPointF(7, size - 8), QPointF(c, size - 5))
        painter.drawLine(QPointF(size - 7, size - 8), QPointF(c, size - 5))
        painter.drawLine(QPointF(c - 3, c - 2), QPointF(c + 3, c + 4))
        painter.drawLine(QPointF(c + 3, c - 2), QPointF(c - 3, c + 4))
    elif kind == "act_fit":
        painter.drawRect(QRect(6, 6, size - 12, size - 12))
        painter.drawLine(QPointF(c, 4), QPointF(c, 8))
        painter.drawLine(QPointF(c, size - 8), QPointF(c, size - 4))
        painter.drawLine(QPointF(4, c), QPointF(8, c))
        painter.drawLine(QPointF(size - 8, c), QPointF(size - 4, c))
        painter.drawEllipse(QPointF(c, c), 2.5, 2.5)
    elif kind == "act_export":
        painter.drawRect(QRect(6, 4, size - 11, size - 8))
        painter.drawLine(QPointF(size - 9, 4), QPointF(size - 5, 8))
        painter.drawLine(QPointF(9, 10), QPointF(size - 12, 10))
        painter.drawLine(QPointF(9, 14), QPointF(size - 14, 14))
        draw_arrow(QPointF(9, size - 8), QPointF(size - 5, size - 8), QPointF(size - 9, size - 12), QPointF(size - 9, size - 4))
    elif kind == "act_map_export":
        painter.drawRect(QRect(4, 5, size - 9, size - 9))
        painter.drawLine(QPointF(4, 11), QPointF(size - 5, 11))
        painter.drawLine(QPointF(10, 5), QPointF(10, size - 4))
        painter.setBrush(color)
        painter.drawEllipse(QPointF(7, 8), 1.6, 1.6)
        painter.drawEllipse(QPointF(16, 16), 1.6, 1.6)
        painter.setBrush(Qt.NoBrush)
        draw_arrow(QPointF(10, size - 5), QPointF(size - 4, size - 5), QPointF(size - 8, size - 9), QPointF(size - 8, size - 1))
    elif kind == "act_pick":
        painter.drawEllipse(QPointF(c, c), 6.5, 6.5)
        painter.drawLine(QPointF(c, 3), QPointF(c, 7))
        painter.drawLine(QPointF(c, size - 7), QPointF(c, size - 3))
        painter.drawLine(QPointF(3, c), QPointF(7, c))
        painter.drawLine(QPointF(size - 7, c), QPointF(size - 3, c))
        painter.setBrush(color)
        painter.drawEllipse(QPointF(c + 2, c - 2), 1.8, 1.8)
    painter.end()
    return pixmap


def make_toolbar_toggle_icon(kind: str) -> QIcon:
    icon = QIcon()
    off_color = QColor("#5f6773")
    on_color = QColor("#ffffff")
    for mode in (QIcon.Normal, QIcon.Active, QIcon.Selected):
        icon.addPixmap(render_toolbar_toggle_icon(kind, off_color), mode, QIcon.Off)
        icon.addPixmap(render_toolbar_toggle_icon(kind, on_color), mode, QIcon.On)
    return icon


def make_toolbar_toggle_button(
    icon_kind: str,
    tooltip: str,
    checked: bool,
    slot: Callable[[bool], None],
) -> QToolButton:
    button = QToolButton()
    button.setCheckable(True)
    button.setChecked(checked)
    button.setIcon(make_toolbar_toggle_icon(icon_kind))
    button.setIconSize(QSize(18, 18))
    button.setToolTip(tooltip)
    button.setToolTipDuration(4000)
    button.setStatusTip(tooltip)
    button.setWhatsThis(tooltip)
    button.setAccessibleName(tooltip)
    button.setToolButtonStyle(Qt.ToolButtonIconOnly)
    button.setAutoRaise(False)
    button.setFixedSize(24, 24)
    button.toggled.connect(slot)
    return button


def make_toolbar_preset_button(icon_kind: str, tooltip: str, slot: Callable[[], None]) -> QToolButton:
    button = QToolButton()
    button.setCheckable(True)
    button.setIcon(make_toolbar_toggle_icon(icon_kind))
    button.setIconSize(QSize(18, 18))
    button.setToolTip(tooltip)
    button.setToolTipDuration(4000)
    button.setStatusTip(tooltip)
    button.setWhatsThis(tooltip)
    button.setAccessibleName(tooltip)
    button.setToolButtonStyle(Qt.ToolButtonIconOnly)
    button.setAutoRaise(False)
    button.setFixedSize(24, 24)
    button.clicked.connect(slot)
    return button


def make_toolbar_action_button(
    icon_kind: str,
    tooltip: str,
    slot: Callable[[], None],
    *,
    button_size: int = 24,
    icon_size: int = 16,
) -> QToolButton:
    button = QToolButton()
    button.setIcon(make_toolbar_toggle_icon(icon_kind))
    button.setIconSize(QSize(icon_size, icon_size))
    button.setToolTip(tooltip)
    button.setToolTipDuration(4000)
    button.setStatusTip(tooltip)
    button.setWhatsThis(tooltip)
    button.setAccessibleName(tooltip)
    button.setToolButtonStyle(Qt.ToolButtonIconOnly)
    button.setAutoRaise(False)
    button.setFixedSize(button_size, button_size)
    button.clicked.connect(slot)
    return button


def make_toolbar_icon_button(
    icon: QIcon,
    tooltip: str,
    slot: Callable[[], None],
    *,
    button_size: int = 24,
    icon_size: int = 16,
) -> QToolButton:
    button = QToolButton()
    button.setIcon(icon)
    button.setIconSize(QSize(icon_size, icon_size))
    button.setToolTip(tooltip)
    button.setToolTipDuration(4000)
    button.setStatusTip(tooltip)
    button.setWhatsThis(tooltip)
    button.setAccessibleName(tooltip)
    button.setToolButtonStyle(Qt.ToolButtonIconOnly)
    button.setAutoRaise(False)
    button.setFixedSize(button_size, button_size)
    button.clicked.connect(slot)
    return button


def gui_like_label_font_size(widget: QWidget) -> int:
    font = widget.font()
    if font.pixelSize() > 0:
        return max(font.pixelSize(), 16)
    if font.pointSizeF() > 0:
        return max(int(round(font.pointSizeF() * 1.6)), 16)
    return 16


def stage_local_to_world(
    points: np.ndarray,
    stage_readout_local: np.ndarray,
    omega_deg: float,
    pivot_world: np.ndarray,
) -> np.ndarray:
    rotation = np.array(rotation_z_deg(omega_deg), dtype=float)
    return pivot_world + (rotation @ (points + stage_readout_local).T).T


def rotation_matrix_from_euler_xyz_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_matrix = np.array(
        ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)),
        dtype=float,
    )
    ry_matrix = np.array(
        ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)),
        dtype=float,
    )
    rz_matrix = np.array(
        ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )
    return rz_matrix @ ry_matrix @ rx_matrix


def orthonormalize_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    u, _singular_values, vh = np.linalg.svd(np.asarray(rotation, dtype=float))
    orthonormal = u @ vh
    if np.linalg.det(orthonormal) < 0.0:
        u[:, -1] *= -1.0
        orthonormal = u @ vh
    return orthonormal


def rotation_matrix_to_euler_xyz_deg(rotation: np.ndarray) -> Tuple[float, float, float]:
    r = np.asarray(rotation, dtype=float)
    if abs(r[2, 0]) < 1.0 - 1e-9:
        ry = np.arcsin(-r[2, 0])
        cy = np.cos(ry)
        rx = np.arctan2(r[2, 1] / cy, r[2, 2] / cy)
        rz = np.arctan2(r[1, 0] / cy, r[0, 0] / cy)
    else:
        rz = 0.0
        if r[2, 0] <= -1.0:
            ry = np.pi / 2.0
            rx = np.arctan2(r[0, 1], r[0, 2])
        else:
            ry = -np.pi / 2.0
            rx = np.arctan2(-r[0, 1], -r[0, 2])
    return (float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz)))


def normalized(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return vector.copy()
    return vector / length


def orthogonal_plane_basis(normal: np.ndarray, up_hint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal_unit = normalized(np.asarray(normal, dtype=float))
    up_candidate = np.asarray(up_hint, dtype=float) - normal_unit * float(np.dot(up_hint, normal_unit))
    if np.linalg.norm(up_candidate) < 1e-9:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        up_candidate = fallback - normal_unit * float(np.dot(fallback, normal_unit))
    up_unit = normalized(up_candidate)
    right_unit = normalized(np.cross(up_unit, normal_unit))
    up_unit = normalized(np.cross(normal_unit, right_unit))
    return right_unit, up_unit


def capture_camera_state(plotter) -> dict:
    camera = plotter.camera
    return {
        "position": tuple(camera.GetPosition()),
        "focal_point": tuple(camera.GetFocalPoint()),
        "view_up": tuple(camera.GetViewUp()),
        "parallel_scale": float(camera.GetParallelScale()),
        "view_angle": float(camera.GetViewAngle()),
        "clipping_range": tuple(camera.GetClippingRange()),
        "parallel_projection": bool(camera.GetParallelProjection()),
    }


def restore_camera_state(plotter, state: dict) -> None:
    camera = plotter.camera
    camera.SetPosition(*state["position"])
    camera.SetFocalPoint(*state["focal_point"])
    camera.SetViewUp(*state["view_up"])
    camera.SetParallelScale(state["parallel_scale"])
    camera.SetViewAngle(state["view_angle"])
    camera.SetClippingRange(*state["clipping_range"])
    if state["parallel_projection"]:
        camera.ParallelProjectionOn()
    else:
        camera.ParallelProjectionOff()


def make_line_polydata(start: np.ndarray, end: np.ndarray) -> pv.PolyData:
    points = np.array([start, end], dtype=float)
    lines = np.array([2, 0, 1], dtype=np.int64)
    return pv.PolyData(points, lines=lines)


def make_multi_line_polydata(segments: Sequence[Tuple[np.ndarray, np.ndarray]]) -> pv.PolyData:
    if not segments:
        return pv.PolyData(np.empty((0, 3), dtype=float))
    points = []
    lines = []
    for start, end in segments:
        start_idx = len(points)
        points.append(np.asarray(start, dtype=float))
        points.append(np.asarray(end, dtype=float))
        lines.extend([2, start_idx, start_idx + 1])
    return pv.PolyData(np.array(points, dtype=float), lines=np.array(lines, dtype=np.int64))


def make_oriented_box(
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    axis_w: np.ndarray,
    size_u: float,
    size_v: float,
    size_w: float,
) -> pv.PolyData:
    box = pv.Box(
        bounds=(
            -size_u / 2.0,
            size_u / 2.0,
            -size_v / 2.0,
            size_v / 2.0,
            -size_w / 2.0,
            size_w / 2.0,
        )
    )
    local_points = np.asarray(box.points, dtype=float)
    transform = np.column_stack(
        (
            normalized(np.asarray(axis_u, dtype=float)),
            normalized(np.asarray(axis_v, dtype=float)),
            normalized(np.asarray(axis_w, dtype=float)),
        )
    )
    box.points = np.asarray(center, dtype=float) + local_points @ transform.T
    return box


def compute_line_beam_path(
    mesh: pv.PolyData,
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
    tolerance: float = 1e-6,
) -> Tuple[float, List[Tuple[np.ndarray, np.ndarray]]]:
    if mesh is None or mesh.n_points == 0:
        return 0.0, []

    unit_direction = normalized(np.asarray(direction, dtype=float))
    if np.linalg.norm(unit_direction) < 1e-12:
        return 0.0, []

    end = np.asarray(origin, dtype=float) + unit_direction * float(max_distance)
    intersection_points, _ = mesh.ray_trace(origin, end, first_point=False)
    if len(intersection_points) == 0:
        return 0.0, []

    distances = sorted(
        float(np.dot(np.asarray(point, dtype=float) - origin, unit_direction))
        for point in intersection_points
    )
    merged_distances: List[float] = []
    for distance in distances:
        if distance < 0.0 or distance > max_distance:
            continue
        if not merged_distances or abs(distance - merged_distances[-1]) > tolerance:
            merged_distances.append(distance)
    if len(merged_distances) < 2:
        return 0.0, []

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    total_length = 0.0
    for idx in range(0, len(merged_distances) - 1, 2):
        entry_distance = merged_distances[idx]
        exit_distance = merged_distances[idx + 1]
        if exit_distance <= entry_distance + tolerance:
            continue
        entry_point = origin + unit_direction * entry_distance
        exit_point = origin + unit_direction * exit_distance
        segments.append((entry_point, exit_point))
        total_length += exit_distance - entry_distance
    return total_length, segments


def build_line_path_length_evaluator(
    mesh: pv.PolyData,
    direction: np.ndarray,
    max_distance: float,
    tolerance: float = 1e-6,
) -> Callable[[np.ndarray], float]:
    unit_direction = normalized(np.asarray(direction, dtype=float))
    if np.linalg.norm(unit_direction) < 1e-12:
        return lambda origin: 0.0

    obb_tree = mesh.obbTree
    points = vtkPoints()
    cell_ids = vtkIdList()
    trace_distance = float(max_distance)

    def evaluator(origin: np.ndarray) -> float:
        origin = np.asarray(origin, dtype=float)
        end = origin + unit_direction * trace_distance
        points.Reset()
        cell_ids.Reset()
        obb_tree.IntersectWithLine(origin.tolist(), end.tolist(), points, cell_ids)
        point_count = points.GetNumberOfPoints()
        if point_count < 2:
            return 0.0

        distances: List[float] = []
        for point_index in range(point_count):
            point = np.asarray(points.GetPoint(point_index), dtype=float)
            distance = float(np.dot(point - origin, unit_direction))
            if 0.0 <= distance <= trace_distance:
                distances.append(distance)
        if len(distances) < 2:
            return 0.0

        distances.sort()
        merged_distances: List[float] = []
        for distance in distances:
            if not merged_distances or abs(distance - merged_distances[-1]) > tolerance:
                merged_distances.append(distance)
        if len(merged_distances) < 2:
            return 0.0

        total_length = 0.0
        for idx in range(0, len(merged_distances) - 1, 2):
            entry_distance = merged_distances[idx]
            exit_distance = merged_distances[idx + 1]
            if exit_distance > entry_distance + tolerance:
                total_length += exit_distance - entry_distance
        return total_length

    return evaluator


def compute_collimated_beam_map(
    mesh: pv.PolyData,
    slit_center: np.ndarray,
    slit_width: float,
    slit_height: float,
    pixel_size_y: float,
    pixel_size_z: float,
    max_distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution_y = max(2, int(np.floor(float(slit_width) / float(pixel_size_y))) + 1)
    resolution_z = max(2, int(np.floor(float(slit_height) / float(pixel_size_z))) + 1)
    y_coords = np.linspace(slit_center[1] - slit_width / 2.0, slit_center[1] + slit_width / 2.0, resolution_y)
    z_coords = np.linspace(slit_center[2] - slit_height / 2.0, slit_center[2] + slit_height / 2.0, resolution_z)
    path_map = np.zeros((resolution_y, resolution_z), dtype=float)
    beam_direction = np.array([1.0, 0.0, 0.0], dtype=float)
    line_path_length = build_line_path_length_evaluator(mesh, beam_direction, max_distance)
    for y_index, y_value in enumerate(y_coords):
        for z_index, z_value in enumerate(z_coords):
            origin = np.array([slit_center[0], y_value, z_value], dtype=float)
            path_map[y_index, z_index] = line_path_length(origin)
    return y_coords, z_coords, path_map


def compute_incoming_beam_average_map(
    mesh: pv.PolyData,
    slit_center: np.ndarray,
    slit_width: float,
    slit_height: float,
    resolution_y: int,
    resolution_z: int,
    pivot_x: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_coords = np.linspace(slit_center[1] - slit_width / 2.0, slit_center[1] + slit_width / 2.0, resolution_y)
    z_coords = np.linspace(slit_center[2] - slit_height / 2.0, slit_center[2] + slit_height / 2.0, resolution_z)
    path_map = np.zeros((resolution_y, resolution_z), dtype=float)
    for y_index, y_value in enumerate(y_coords):
        for z_index, z_value in enumerate(z_coords):
            origin = np.array([slit_center[0], y_value, z_value], dtype=float)
            end = np.array([pivot_x, y_value, z_value], dtype=float)
            path_length, _segments = compute_segment_path_length(mesh, origin, end)
            path_map[y_index, z_index] = path_length
    return y_coords, z_coords, path_map


def compute_incoming_beam_path_to_point(
    mesh: pv.PolyData,
    slit_center: np.ndarray,
    point: np.ndarray,
) -> float:
    origin = np.array([slit_center[0], point[1], point[2]], dtype=float)
    path_length, _segments = compute_segment_path_length(mesh, origin, np.asarray(point, dtype=float))
    return path_length


def enginx_rietveld_count_time_minutes(path_length_mm: float, gauge_volume_mm3: float, material: str) -> float:
    if material not in RIETVELD_COUNT_TIME_LAWS:
        raise KeyError(f"Unsupported ENGIN-X count-time material: {material}")
    if gauge_volume_mm3 <= 0.0:
        raise ValueError("Gauge volume must be positive.")

    count_times = RIETVELD_COUNT_TIME_LAWS[material]
    reference_paths = COUNT_TIME_REFERENCE_PATHS_MM
    mu = float(np.log(count_times[3] / count_times[2]) / (reference_paths[3] - reference_paths[2]))
    reference_time = float(count_times[2])
    reference_path = float(reference_paths[2])
    volume_factor = 64.0 / float(gauge_volume_mm3)
    return reference_time * float(np.exp(mu * (float(path_length_mm) - reference_path))) * volume_factor


def format_scan_numeric(value: float, decimals: int = 3) -> str:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-9:
        return str(int(rounded))
    return f"{float(value):.{decimals}f}"


def compute_lattice_strain(
    lattice_parameter: float,
    d0: float,
) -> float:
    if abs(float(d0)) < 1e-12:
        raise ValueError("D0 must be non-zero.")
    return (float(lattice_parameter) - float(d0)) / float(d0)


def compute_lattice_strain_uncertainty(
    lattice_parameter: float,
    lattice_uncertainty: Optional[float],
    d0: float,
    d0_uncertainty: Optional[float],
) -> Optional[float]:
    if lattice_uncertainty is None or d0_uncertainty is None:
        return None
    if abs(float(d0)) < 1e-12:
        raise ValueError("D0 must be non-zero.")
    lattice_term = float(lattice_uncertainty) / float(d0)
    d0_term = float(lattice_parameter) * float(d0_uncertainty) / (float(d0) * float(d0))
    return float(np.sqrt(lattice_term * lattice_term + d0_term * d0_term))


def compute_three_dimensional_stress_mpa(
    strains: Sequence[float],
    youngs_modulus_mpa: float,
    poissons_ratio: float,
) -> np.ndarray:
    if len(strains) != 3:
        raise ValueError("Three principal strains are required for 3D stress.")
    denominator = (1.0 + float(poissons_ratio)) * (1.0 - 2.0 * float(poissons_ratio))
    if abs(denominator) < 1e-12:
        raise ValueError("Poisson's ratio is too close to 0.5 for the 3D isotropic stress model.")
    scale = float(youngs_modulus_mpa) / denominator
    strain_array = np.asarray(strains, dtype=float)
    trace = float(np.sum(strain_array))
    return scale * ((1.0 - float(poissons_ratio)) * strain_array + float(poissons_ratio) * (trace - strain_array))


def compute_three_dimensional_stress_uncertainty_mpa(
    strain_uncertainties: Sequence[Optional[float]],
    youngs_modulus_mpa: float,
    poissons_ratio: float,
) -> Optional[np.ndarray]:
    if any(value is None for value in strain_uncertainties):
        return None
    denominator = (1.0 + float(poissons_ratio)) * (1.0 - 2.0 * float(poissons_ratio))
    if abs(denominator) < 1e-12:
        raise ValueError("Poisson's ratio is too close to 0.5 for the 3D isotropic stress model.")
    scale = float(youngs_modulus_mpa) / denominator
    strain_uncertainty_array = np.asarray(strain_uncertainties, dtype=float)
    output = np.zeros(3, dtype=float)
    for stress_index in range(3):
        coefficients = np.full(3, float(poissons_ratio), dtype=float)
        coefficients[stress_index] = 1.0 - float(poissons_ratio)
        output[stress_index] = abs(scale) * float(
            np.sqrt(np.sum((coefficients * strain_uncertainty_array) ** 2))
        )
    return output


def build_mesh_signed_distance_evaluator(mesh: pv.PolyData) -> Callable[[np.ndarray], float]:
    implicit_distance = vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(mesh)

    def evaluator(point: np.ndarray) -> float:
        return float(implicit_distance.EvaluateFunction(np.asarray(point, dtype=float)))

    return evaluator


def point_inside_closed_mesh(
    mesh: pv.PolyData,
    point: np.ndarray,
    tolerance: float = 1e-6,
    signed_distance_evaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> bool:
    if mesh is None or mesh.n_points == 0:
        return False
    if signed_distance_evaluator is not None:
        signed_distance = signed_distance_evaluator(point)
        if signed_distance < -tolerance:
            return True
        if signed_distance > tolerance:
            return False
    probe = pv.PolyData(np.asarray([point], dtype=float))
    selected = probe.select_interior_points(
        mesh,
        method="signed_distance",
        check_surface=False,
    )
    if "selected_points" in selected.array_names:
        mask = np.asarray(selected["selected_points"]).astype(bool)
    elif "SelectedPoints" in selected.array_names:
        mask = np.asarray(selected["SelectedPoints"]).astype(bool)
    else:
        raise KeyError("No interior-point selection array was returned by PyVista.")
    return bool(mask[0]) if len(mask) else False


def point_on_or_inside_mesh(
    mesh: pv.PolyData,
    point: np.ndarray,
    probe_direction: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
    signed_distance_evaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> bool:
    if mesh is None or mesh.n_points == 0:
        return False
    point = np.asarray(point, dtype=float)
    if point_inside_closed_mesh(mesh, point, tolerance=tolerance, signed_distance_evaluator=signed_distance_evaluator):
        return True

    bounds = np.array(mesh.bounds, dtype=float)
    mesh_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=float)
    mesh_max = np.array([bounds[1], bounds[3], bounds[5]], dtype=float)
    scale = max(float(np.linalg.norm(mesh_max - mesh_min)), 1.0)
    epsilon = max(scale * 1e-4, 1e-3)
    candidate_directions: List[np.ndarray] = []
    if probe_direction is not None and np.linalg.norm(probe_direction) > 1e-9:
        candidate_directions.append(np.asarray(probe_direction, dtype=float))
    candidate_directions.extend(
        [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
            normalized(np.array([1.0, 1.0, 1.0], dtype=float)),
        ]
    )
    for direction in candidate_directions:
        unit_direction = normalized(direction)
        if np.linalg.norm(unit_direction) < 1e-9:
            continue
        path_length, _segments = compute_segment_path_length(
            mesh,
            point - unit_direction * epsilon,
            point + unit_direction * epsilon,
            tolerance=tolerance,
            signed_distance_evaluator=signed_distance_evaluator,
            collect_segments=False,
        )
        if path_length > epsilon * 0.15:
            return True
    return False


def interval_midpoint_inside_mesh(
    mesh: pv.PolyData,
    origin: np.ndarray,
    unit_direction: np.ndarray,
    start_distance: float,
    end_distance: float,
    tolerance: float = 1e-6,
    signed_distance_evaluator: Optional[Callable[[np.ndarray], float]] = None,
) -> bool:
    midpoint_distance = (float(start_distance) + float(end_distance)) * 0.5
    midpoint = np.asarray(origin, dtype=float) + np.asarray(unit_direction, dtype=float) * midpoint_distance
    if signed_distance_evaluator is not None:
        signed_distance = signed_distance_evaluator(midpoint)
        if signed_distance < -tolerance:
            return True
        if signed_distance > tolerance:
            return False
        return False
    return point_inside_closed_mesh(mesh, midpoint, tolerance=tolerance)


def compute_segment_path_length(
    mesh: pv.PolyData,
    origin: np.ndarray,
    end: np.ndarray,
    tolerance: float = 1e-6,
    signed_distance_evaluator: Optional[Callable[[np.ndarray], float]] = None,
    collect_segments: bool = True,
) -> Tuple[float, List[Tuple[np.ndarray, np.ndarray]]]:
    if mesh is None or mesh.n_points == 0:
        return 0.0, []

    origin = np.asarray(origin, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - origin
    max_distance = float(np.linalg.norm(segment))
    if max_distance < 1e-12:
        return 0.0, []

    unit_direction = segment / max_distance
    intersection_points, _ = mesh.ray_trace(origin, end, first_point=False)
    distances = sorted(
        float(np.dot(np.asarray(point, dtype=float) - origin, unit_direction))
        for point in intersection_points
    )
    merged_distances: List[float] = []
    for distance in distances:
        if distance < 0.0 or distance > max_distance:
            continue
        if not merged_distances or abs(distance - merged_distances[-1]) > tolerance:
            merged_distances.append(distance)
    merged_distances = [
        distance for distance in merged_distances if tolerance < distance < max_distance - tolerance
    ]

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    total_length = 0.0
    interval_boundaries = [0.0, *merged_distances, max_distance]
    for start_distance, end_distance in zip(interval_boundaries[:-1], interval_boundaries[1:]):
        if end_distance <= start_distance + tolerance:
            continue
        if not interval_midpoint_inside_mesh(
            mesh,
            origin,
            unit_direction,
            start_distance,
            end_distance,
            tolerance=tolerance,
            signed_distance_evaluator=signed_distance_evaluator,
        ):
            continue
        if collect_segments:
            entry_point = origin + unit_direction * start_distance
            exit_point = origin + unit_direction * end_distance
            segments.append((entry_point, exit_point))
        total_length += end_distance - start_distance

    return total_length, segments


def remove_scalar_bar_if_present(plotter: pv.Plotter, title: str) -> None:
    if title in plotter.scalar_bars:
        plotter.remove_scalar_bar(title=title, render=False)


class SpreadsheetTableWidget(QTableWidget):
    def __init__(self, rows: int, columns: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(rows, columns, parent)
        self.read_only_columns: set[int] = set()
        self.after_paste: Optional[Callable[[], None]] = None
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def minimumSizeHint(self) -> QSize:
        return QSize(180, 120)

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Copy):
            self.copy_selection_to_clipboard()
            event.accept()
            return
        if event.matches(QKeySequence.Paste):
            self.paste_from_clipboard()
            event.accept()
            return
        super().keyPressEvent(event)

    def copy_selection_to_clipboard(self) -> None:
        indexes = self.selectedIndexes()
        if not indexes:
            return
        rows = sorted({index.row() for index in indexes})
        columns = sorted({index.column() for index in indexes})
        copied_lines = []
        for row in range(rows[0], rows[-1] + 1):
            values = []
            for column in range(columns[0], columns[-1] + 1):
                item = self.item(row, column)
                values.append("" if item is None else item.text())
            copied_lines.append("\t".join(values))
        QApplication.clipboard().setText("\n".join(copied_lines))

    def paste_from_clipboard(self) -> None:
        clipboard_text = QApplication.clipboard().text()
        if clipboard_text == "":
            return

        lines = clipboard_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        grid = [line.split("\t") for line in lines]
        if not grid:
            return

        was_blocked = self.blockSignals(True)
        try:
            selected_indexes = self.selectedIndexes()
            if len(grid) == 1 and len(grid[0]) == 1 and len(selected_indexes) > 1:
                self._paste_single_value_to_selection(grid[0][0], selected_indexes)
            else:
                self._paste_grid(grid)
        finally:
            self.blockSignals(was_blocked)

        if self.after_paste is not None:
            self.after_paste()

    def _paste_single_value_to_selection(self, value: str, indexes) -> None:
        for index in indexes:
            self._set_cell_text(index.row(), index.column(), value)

    def _paste_grid(self, grid: List[List[str]]) -> None:
        current_row = self.currentRow()
        current_column = self.currentColumn()
        if current_row < 0:
            current_row = 0
        if current_column < 0:
            current_column = 0

        while self.rowCount() < current_row + len(grid):
            self.insertRow(self.rowCount())

        for row_offset, row_values in enumerate(grid):
            for column_offset, value in enumerate(row_values):
                target_column = current_column + column_offset
                if target_column >= self.columnCount():
                    continue
                self._set_cell_text(current_row + row_offset, target_column, value)

    def _set_cell_text(self, row: int, column: int, value: str) -> None:
        if column in self.read_only_columns:
            return
        item = self.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.setItem(row, column, item)
        if not (item.flags() & Qt.ItemIsEditable):
            return
        item.setText(value)


class StressTableHeaderView(QHeaderView):
    def __init__(self, orientation: Qt.Orientation, parent: Optional[QWidget] = None) -> None:
        super().__init__(orientation, parent)
        self.single_headers = {
            0: "Run",
            1: "Point ID",
        }
        self.grouped_headers = [
            (2, "lattice parameter 1", ["a", "\u0394a", "a0", "\u0394a0"]),
            (6, "lattice parameter 2", ["a", "\u0394a", "a0", "\u0394a0"]),
            (10, "lattice parameter 3", ["a", "\u0394a", "a0", "\u0394a0"]),
            (14, "Strain 1 (\u00b5\u03b5)", ["\u03b5", "\u0394\u03b5"]),
            (16, "Strain 2 (\u00b5\u03b5)", ["\u03b5", "\u0394\u03b5"]),
            (18, "Strain 3 (\u00b5\u03b5)", ["\u03b5", "\u0394\u03b5"]),
            (20, "Stress 1 (MPA)", ["\u03c3", "\u0394\u03c3"]),
            (22, "Stress 2 (MPA)", ["\u03c3", "\u0394\u03c3"]),
            (24, "Stress 3 (MPA)", ["\u03c3", "\u0394\u03c3"]),
        ]
        self.setDefaultAlignment(Qt.AlignCenter)
        self.setMinimumHeight(self._target_header_height())

    def _target_header_height(self) -> int:
        metrics = self.fontMetrics()
        return max(metrics.height() * 2 + 8, 32)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(hint.width(), self._target_header_height())

    def sectionSizeFromContents(self, logical_index: int) -> QSize:
        size = super().sectionSizeFromContents(logical_index)
        return QSize(size.width(), self._target_header_height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        background_color = self.palette().color(self.backgroundRole())
        border_color = self.palette().color(self.foregroundRole()).darker(130)
        text_color = self.palette().color(self.foregroundRole())
        painter.fillRect(event.rect(), background_color)

        total_height = self.height()
        top_height = total_height // 2
        bottom_height = total_height - top_height

        for logical_index, text in self.single_headers.items():
            if self.isSectionHidden(logical_index):
                continue
            rect = QRect(
                self.sectionViewportPosition(logical_index),
                0,
                self.sectionSize(logical_index),
                total_height,
            )
            if rect.right() < event.rect().left() or rect.left() > event.rect().right():
                continue
            self._draw_header_cell(painter, rect, text, border_color, text_color)

        for start_column, title, sublabels in self.grouped_headers:
            end_column = start_column + len(sublabels) - 1
            if any(self.isSectionHidden(column) for column in range(start_column, end_column + 1)):
                continue
            left = self.sectionViewportPosition(start_column)
            right = self.sectionViewportPosition(end_column) + self.sectionSize(end_column)
            title_rect = QRect(left, 0, right - left, top_height)
            if not (title_rect.right() < event.rect().left() or title_rect.left() > event.rect().right()):
                self._draw_header_cell(painter, title_rect, title, border_color, text_color)
            for offset, label in enumerate(sublabels):
                logical_index = start_column + offset
                rect = QRect(
                    self.sectionViewportPosition(logical_index),
                    top_height,
                    self.sectionSize(logical_index),
                    bottom_height,
                )
                if rect.right() < event.rect().left() or rect.left() > event.rect().right():
                    continue
                self._draw_header_cell(painter, rect, label, border_color, text_color)

    def _draw_header_cell(self, painter: QPainter, rect: QRect, text: str, border_color, text_color) -> None:
        painter.save()
        painter.setPen(border_color)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.setPen(text_color)
        painter.drawText(rect.adjusted(4, 2, -4, -2), Qt.AlignCenter, text)
        painter.restore()


class CursorZoomQtInteractor(QtInteractor):
    def _event_xy(self, ev) -> Tuple[float, float]:
        try:
            pos = ev.position()
            return pos.x(), pos.y()
        except AttributeError:
            return ev.x(), ev.y()

    def _world_at_cursor_focal_depth(self, ev) -> Optional[np.ndarray]:
        renderer = self.renderer
        if renderer is None:
            return None
        pos_x, pos_y = self._event_xy(ev)

        focal_point = np.array(self.camera.GetFocalPoint(), dtype=float)
        renderer.SetWorldPoint(focal_point[0], focal_point[1], focal_point[2], 1.0)
        renderer.WorldToDisplay()
        _, _, focal_depth = renderer.GetDisplayPoint()

        renderer.SetDisplayPoint(pos_x, pos_y, focal_depth)
        renderer.DisplayToWorld()
        world_point = np.array(renderer.GetWorldPoint(), dtype=float)
        if abs(world_point[3]) < 1e-12:
            return None
        point = world_point[:3] / world_point[3]
        if not np.all(np.isfinite(point)):
            return None
        return point

    def wheelEvent(self, ev) -> None:
        before_point = self._world_at_cursor_focal_depth(ev)
        super().wheelEvent(ev)
        if before_point is None:
            return

        after_point = self._world_at_cursor_focal_depth(ev)
        if after_point is None:
            return

        delta = before_point - after_point
        if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) < 1e-12:
            return

        camera = self.camera
        position = np.array(camera.GetPosition(), dtype=float) + delta
        focal_point = np.array(camera.GetFocalPoint(), dtype=float) + delta
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point)
        self.renderer.ResetCameraClippingRange()
        self.render()


def combine_scene_meshes(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    non_empty = [mesh for mesh in meshes if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0]
    if not non_empty:
        raise ValueError("The selected model did not contain any mesh geometry.")
    if len(non_empty) == 1:
        return non_empty[0]
    return trimesh.util.concatenate(non_empty)


def trimesh_loaded_to_polydata(loaded, source_name: str) -> pv.PolyData:
    if isinstance(loaded, trimesh.Scene):
        mesh = combine_scene_meshes(tuple(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported model type returned for {source_name}.")

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"{source_name} did not contain triangle faces.")
    faces_with_size = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(vertices, faces_with_size).clean()


def load_mesh_as_polydata(path: Path) -> pv.PolyData:
    loaded = trimesh.load(path, force="scene")
    return trimesh_loaded_to_polydata(loaded, path.name)


def load_mesh_bytes_as_polydata(data: bytes, source_name: str, file_type: str = "stl") -> pv.PolyData:
    loaded = trimesh.load(io.BytesIO(data), file_type=file_type, force="scene")
    return trimesh_loaded_to_polydata(loaded, source_name)


def polydata_to_stl_bytes(mesh: pv.PolyData) -> bytes:
    if mesh is None or mesh.n_points == 0:
        raise ValueError("No mesh is available to save.")
    triangle_mesh = mesh.triangulate().clean()
    faces = np.asarray(triangle_mesh.faces, dtype=np.int64)
    if faces.size == 0:
        raise ValueError("The mesh did not contain any triangle faces to save.")
    triangle_faces = faces.reshape((-1, 4))[:, 1:4]
    export_mesh = trimesh.Trimesh(
        vertices=np.asarray(triangle_mesh.points, dtype=float),
        faces=triangle_faces,
        process=False,
    )
    exported = export_mesh.export(file_type="stl")
    if isinstance(exported, str):
        return exported.encode("utf-8")
    return bytes(exported)


def table_to_serializable_rows(table: QTableWidget, headers: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for row_index in range(table.rowCount()):
        row_data = {}
        has_value = False
        for column_index, header in enumerate(headers):
            item = table.item(row_index, column_index)
            text = "" if item is None else item.text().strip()
            row_data[header] = text
            if text != "":
                has_value = True
        if has_value:
            rows.append(row_data)
    return rows


def populate_table_from_serialized_rows(
    table: QTableWidget,
    headers: Sequence[str],
    rows: Sequence[dict],
    read_only_columns: Optional[Sequence[int]] = None,
) -> None:
    read_only = set(read_only_columns or [])
    previous_state = table.blockSignals(True)
    table.setRowCount(0)
    try:
        for row_data in rows:
            row_index = table.rowCount()
            table.insertRow(row_index)
            for column_index, header in enumerate(headers):
                value = row_data.get(header, "")
                item = QTableWidgetItem("" if value is None else str(value))
                if column_index in read_only:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row_index, column_index, item)
    finally:
        table.blockSignals(previous_state)


class PointPickerPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, *, enable_3d: bool = True) -> None:
        super().__init__(parent)

        self.enable_3d = enable_3d
        self.model_mesh: Optional[pv.PolyData] = None
        self.mesh_path: Optional[Path] = None
        self.plane_mesh = None
        self.slice_mesh = None
        self.clipped_mesh = None
        self.picked_points_mesh = None
        self.picked_points: List[Tuple[str, np.ndarray]] = []
        self._surface_picking_initialized = False

        self._build_ui()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(8)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        outer_layout.addWidget(main_splitter, stretch=1)

        controls_panel = QWidget()
        controls_panel.setMinimumWidth(420)
        controls_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.instructions_label = QLabel(
            "Pick in model coordinates. Use direct XYZ entry or define a slice plane and click on the plane within the model."
        )
        self.instructions_label.setWordWrap(True)
        controls_layout.addWidget(self.instructions_label)

        slice_group = QGroupBox("Slice Plane Picking")
        slice_layout = QVBoxLayout(slice_group)
        slice_form = QFormLayout()
        self.plane_mode_combo = NoWheelComboBox()
        self.plane_mode_combo.addItems(["Axis aligned", "Point + normal", "Three points"])
        self.plane_mode_combo.currentTextChanged.connect(self.on_plane_mode_changed)
        slice_form.addRow("Plane mode", self.plane_mode_combo)

        self.plane_mode_stack = QStackedWidget()
        axis_mode_widget = QWidget()
        axis_mode_layout = QFormLayout(axis_mode_widget)
        self.plane_axis_combo = NoWheelComboBox()
        self.plane_axis_combo.addItems(["X", "Y", "Z"])
        self.plane_axis_combo.currentTextChanged.connect(self.on_plane_definition_changed)
        self.plane_value_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_value_spin.valueChanged.connect(self.on_plane_definition_changed)
        axis_mode_layout.addRow("Plane axis", self.plane_axis_combo)
        axis_mode_layout.addRow("Plane value", self.plane_value_spin)
        self.plane_mode_stack.addWidget(axis_mode_widget)

        point_normal_widget = QWidget()
        point_normal_layout = QFormLayout(point_normal_widget)
        self.plane_origin_x_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_origin_y_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_origin_z_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_normal_x_spin = make_spin_box(1.0, -100000.0, 100000.0)
        self.plane_normal_y_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_normal_z_spin = make_spin_box(0.0, -100000.0, 100000.0)
        for spin_box in (
            self.plane_origin_x_spin,
            self.plane_origin_y_spin,
            self.plane_origin_z_spin,
            self.plane_normal_x_spin,
            self.plane_normal_y_spin,
            self.plane_normal_z_spin,
        ):
            spin_box.valueChanged.connect(self.on_plane_definition_changed)
        point_normal_layout.addRow("Origin X", self.plane_origin_x_spin)
        point_normal_layout.addRow("Origin Y", self.plane_origin_y_spin)
        point_normal_layout.addRow("Origin Z", self.plane_origin_z_spin)
        point_normal_layout.addRow("Normal X", self.plane_normal_x_spin)
        point_normal_layout.addRow("Normal Y", self.plane_normal_y_spin)
        point_normal_layout.addRow("Normal Z", self.plane_normal_z_spin)
        self.plane_mode_stack.addWidget(point_normal_widget)

        three_points_widget = QWidget()
        three_points_layout = QFormLayout(three_points_widget)
        self.plane_p1_x_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_p1_y_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_p1_z_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_p2_x_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_p2_y_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_p2_z_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_p3_x_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_p3_y_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.plane_p3_z_spin = make_spin_box(0.0, -100000.0, 100000.0)
        for spin_box in (
            self.plane_p1_x_spin,
            self.plane_p1_y_spin,
            self.plane_p1_z_spin,
            self.plane_p2_x_spin,
            self.plane_p2_y_spin,
            self.plane_p2_z_spin,
            self.plane_p3_x_spin,
            self.plane_p3_y_spin,
            self.plane_p3_z_spin,
        ):
            spin_box.valueChanged.connect(self.on_plane_definition_changed)
        three_points_layout.addRow("P1 X", self.plane_p1_x_spin)
        three_points_layout.addRow("P1 Y", self.plane_p1_y_spin)
        three_points_layout.addRow("P1 Z", self.plane_p1_z_spin)
        three_points_layout.addRow("P2 X", self.plane_p2_x_spin)
        three_points_layout.addRow("P2 Y", self.plane_p2_y_spin)
        three_points_layout.addRow("P2 Z", self.plane_p2_z_spin)
        three_points_layout.addRow("P3 X", self.plane_p3_x_spin)
        three_points_layout.addRow("P3 Y", self.plane_p3_y_spin)
        three_points_layout.addRow("P3 Z", self.plane_p3_z_spin)
        self.plane_mode_stack.addWidget(three_points_widget)

        self.clip_invert_checkbox = QCheckBox("Flip clip side")
        self.clip_invert_checkbox.toggled.connect(self.on_plane_definition_changed)
        self.require_inside_checkbox = QCheckBox("Require point inside model")
        self.require_inside_checkbox.setChecked(True)
        normal_view_button = QPushButton("Normal view")
        normal_view_button.clicked.connect(self.view_plane_normal)
        slice_layout.addLayout(slice_form)
        slice_layout.addWidget(self.plane_mode_stack)
        shared_controls = QFormLayout()
        shared_controls.addRow(self.clip_invert_checkbox)
        shared_controls.addRow(self.require_inside_checkbox)
        shared_controls.addRow(normal_view_button)
        slice_layout.addLayout(shared_controls)
        controls_layout.addWidget(slice_group)

        manual_group = QGroupBox("Direct Coordinates")
        manual_layout = QFormLayout(manual_group)
        self.manual_x_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.manual_y_spin = make_spin_box(0.0, -100000.0, 100000.0)
        self.manual_z_spin = make_spin_box(0.0, -100000.0, 100000.0)
        add_manual_button = QPushButton("Add manual point")
        add_manual_button.clicked.connect(self.add_manual_point)
        manual_layout.addRow("Model X", self.manual_x_spin)
        manual_layout.addRow("Model Y", self.manual_y_spin)
        manual_layout.addRow("Model Z", self.manual_z_spin)
        manual_layout.addRow(add_manual_button)
        controls_layout.addWidget(manual_group)

        points_group = QGroupBox("Picked Points")
        points_layout = QVBoxLayout(points_group)
        points_button_row = QHBoxLayout()
        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self.remove_selected_points)
        undo_button = QPushButton("Undo last")
        undo_button.clicked.connect(self.undo_last_point)
        clear_button = QPushButton("Clear all")
        clear_button.clicked.connect(self.clear_points)
        save_button = QPushButton("Save CSV")
        save_button.clicked.connect(self.save_csv_dialog)
        points_button_row.addWidget(remove_button)
        points_button_row.addWidget(undo_button)
        points_button_row.addWidget(clear_button)
        points_button_row.addWidget(save_button)
        points_button_row.addStretch(1)
        points_layout.addLayout(points_button_row)

        self.points_table = QTableWidget(0, 4)
        self.points_table.setHorizontalHeaderLabels(["Label", "Model X", "Model Y", "Model Z"])
        self.points_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.points_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.points_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        points_layout.addWidget(self.points_table)
        controls_layout.addWidget(points_group, stretch=1)

        self.status_label = QLabel("Load a mesh from this tab or the Placement tab, then use the plane or direct coordinates to add model points.")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        self.live_coordinate_label = QLabel("Cursor: -")
        self.live_coordinate_label.setWordWrap(True)
        controls_layout.addWidget(self.live_coordinate_label)
        main_splitter.addWidget(controls_panel)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)
        if self.enable_3d:
            self.plotter = CursorZoomQtInteractor(self)
            self.plotter.set_background("white")
            self.plotter.add_axes()
            self.plotter.enable_parallel_projection()
            self.plotter.interactor.installEventFilter(self)
            viewer_layout.addWidget(self.plotter.interactor)
        else:
            self.plotter = None
            placeholder = QLabel("3D point-picking viewport disabled for smoke test mode.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet(
                "background-color: #0d1117; color: #d7d7d7; border: 1px solid #2f3a45; padding: 24px;"
            )
            viewer_layout.addWidget(placeholder)
        main_splitter.addWidget(viewer_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([560, 720])
        if self.enable_3d:
            QTimer.singleShot(0, self.initialize_surface_point_picking)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.enable_3d:
            self.initialize_surface_point_picking()

    def closeEvent(self, event) -> None:
        self.cleanup_plotter()
        super().closeEvent(event)

    def cleanup_plotter(self) -> None:
        plotter = getattr(self, "plotter", None)
        if plotter is None:
            return
        try:
            plotter.close()
        except Exception:
            try:
                plotter.Finalize()
            except Exception:
                pass
        self.plotter = None

    def initialize_surface_point_picking(self) -> None:
        if self._surface_picking_initialized:
            return
        if self.plotter is None:
            return
        if getattr(self.plotter, "iren", None) is None:
            return
        self.plotter.enable_surface_point_picking(
            callback=self.on_plane_picked,
            show_message=False,
            show_point=False,
            left_clicking=True,
            clear_on_no_selection=False,
        )
        self._surface_picking_initialized = True

    def set_mesh(self, mesh: Optional[pv.PolyData], mesh_path: Optional[Path] = None) -> None:
        previous_path = None if self.mesh_path is None else str(self.mesh_path)
        next_path = None if mesh_path is None else str(mesh_path)
        if previous_path != next_path:
            self.picked_points = []
            self.refresh_points_table()
        self.mesh_path = mesh_path
        self.model_mesh = mesh.copy(deep=True) if mesh is not None else None
        if self.model_mesh is None or self.model_mesh.n_points == 0:
            self.clear_scene()
            self.status_label.setText("Load a mesh from this tab or the Placement tab, then use the plane or direct coordinates to add model points.")
            self.live_coordinate_label.setText("Cursor: -")
            return
        self.configure_coordinate_ranges()
        self.refresh_scene(reset_camera=True)
        self.view_plane_normal()

    def clear_scene(self) -> None:
        if self.plotter is None:
            return
        for actor_name in ("picker_clipped_mesh", "picker_slice_contour", "picker_plane", "picker_points", "picker_point_labels"):
            self.plotter.remove_actor(actor_name, render=False)
        self.plane_mesh = None
        self.slice_mesh = None
        self.clipped_mesh = None
        self.picked_points_mesh = None
        self.plotter.render()

    def configure_coordinate_ranges(self) -> None:
        if self.model_mesh is None or self.model_mesh.n_points == 0:
            return
        bounds = np.array(self.model_mesh.bounds, dtype=float)
        mesh_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=float)
        mesh_max = np.array([bounds[1], bounds[3], bounds[5]], dtype=float)
        center = 0.5 * (mesh_min + mesh_max)
        diag = max(float(np.linalg.norm(mesh_max - mesh_min)), 1.0)
        padding = max(diag * 0.25, 5.0)
        ranges = [
            (mesh_min[0] - padding, mesh_max[0] + padding, center[0]),
            (mesh_min[1] - padding, mesh_max[1] + padding, center[1]),
            (mesh_min[2] - padding, mesh_max[2] + padding, center[2]),
        ]
        coordinate_spin_boxes = (
            self.manual_x_spin,
            self.manual_y_spin,
            self.manual_z_spin,
            self.plane_origin_x_spin,
            self.plane_origin_y_spin,
            self.plane_origin_z_spin,
            self.plane_p1_x_spin,
            self.plane_p1_y_spin,
            self.plane_p1_z_spin,
            self.plane_p2_x_spin,
            self.plane_p2_y_spin,
            self.plane_p2_z_spin,
            self.plane_p3_x_spin,
            self.plane_p3_y_spin,
            self.plane_p3_z_spin,
        )
        for index, spin_box in enumerate(coordinate_spin_boxes):
            minimum, maximum, center_value = ranges[index % 3]
            was_blocked = spin_box.blockSignals(True)
            spin_box.setRange(minimum, maximum)
            if abs(spin_box.value()) < 1e-12:
                spin_box.setValue(center_value)
            spin_box.blockSignals(was_blocked)
        self.initialize_three_point_plane_defaults(mesh_min, mesh_max, center)
        self.update_plane_value_range()

    def update_plane_value_range(self) -> None:
        if self.model_mesh is None or self.model_mesh.n_points == 0:
            return
        bounds = np.array(self.model_mesh.bounds, dtype=float)
        axis_index = {"X": 0, "Y": 1, "Z": 2}[self.plane_axis_combo.currentText()]
        axis_bounds = [
            (bounds[0], bounds[1]),
            (bounds[2], bounds[3]),
            (bounds[4], bounds[5]),
        ][axis_index]
        center_value = 0.5 * (axis_bounds[0] + axis_bounds[1])
        was_blocked = self.plane_value_spin.blockSignals(True)
        self.plane_value_spin.setRange(float(axis_bounds[0]), float(axis_bounds[1]))
        if self.plane_value_spin.value() < axis_bounds[0] or self.plane_value_spin.value() > axis_bounds[1]:
            self.plane_value_spin.setValue(center_value)
        elif abs(self.plane_value_spin.value()) < 1e-12:
            self.plane_value_spin.setValue(center_value)
        self.plane_value_spin.blockSignals(was_blocked)

    def initialize_three_point_plane_defaults(
        self,
        mesh_min: np.ndarray,
        mesh_max: np.ndarray,
        center: np.ndarray,
    ) -> None:
        span_x = max(float(mesh_max[0] - mesh_min[0]), 1.0)
        span_y = max(float(mesh_max[1] - mesh_min[1]), 1.0)
        default_points = (
            (self.plane_p1_x_spin, self.plane_p1_y_spin, self.plane_p1_z_spin, center),
            (
                self.plane_p2_x_spin,
                self.plane_p2_y_spin,
                self.plane_p2_z_spin,
                center + np.array([span_x * 0.25, 0.0, 0.0], dtype=float),
            ),
            (
                self.plane_p3_x_spin,
                self.plane_p3_y_spin,
                self.plane_p3_z_spin,
                center + np.array([0.0, span_y * 0.25, 0.0], dtype=float),
            ),
        )
        for spin_x, spin_y, spin_z, point in default_points:
            if abs(spin_x.value()) < 1e-12 and abs(spin_y.value()) < 1e-12 and abs(spin_z.value()) < 1e-12:
                blocked_x = spin_x.blockSignals(True)
                blocked_y = spin_y.blockSignals(True)
                blocked_z = spin_z.blockSignals(True)
                spin_x.setValue(float(point[0]))
                spin_y.setValue(float(point[1]))
                spin_z.setValue(float(point[2]))
                spin_x.blockSignals(blocked_x)
                spin_y.blockSignals(blocked_y)
                spin_z.blockSignals(blocked_z)

    def on_plane_mode_changed(self, mode_text: str) -> None:
        mode_index = {"Axis aligned": 0, "Point + normal": 1, "Three points": 2}[mode_text]
        self.plane_mode_stack.setCurrentIndex(mode_index)
        self.on_plane_definition_changed()

    def on_plane_definition_changed(self, *_args) -> None:
        self.update_plane_value_range()
        try:
            self.refresh_scene(reset_camera=False)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            self.live_coordinate_label.setText("Cursor: -")

    def current_plane_definition(self) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.model_mesh is None or self.model_mesh.n_points == 0:
            raise ValueError("No mesh is loaded.")
        bounds = np.array(self.model_mesh.bounds, dtype=float)
        mesh_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=float)
        mesh_max = np.array([bounds[1], bounds[3], bounds[5]], dtype=float)
        center = 0.5 * (mesh_min + mesh_max)
        spans = mesh_max - mesh_min
        mode = self.plane_mode_combo.currentText()
        if mode == "Axis aligned":
            axis_index = {"X": 0, "Y": 1, "Z": 2}[self.plane_axis_combo.currentText()]
            origin = center.copy()
            origin[axis_index] = self.plane_value_spin.value()
            normal = np.zeros(3, dtype=float)
            normal[axis_index] = 1.0
        elif mode == "Point + normal":
            origin = np.array(
                [
                    self.plane_origin_x_spin.value(),
                    self.plane_origin_y_spin.value(),
                    self.plane_origin_z_spin.value(),
                ],
                dtype=float,
            )
            normal = np.array(
                [
                    self.plane_normal_x_spin.value(),
                    self.plane_normal_y_spin.value(),
                    self.plane_normal_z_spin.value(),
                ],
                dtype=float,
            )
            if np.linalg.norm(normal) < 1e-9:
                raise ValueError("Plane normal must be non-zero.")
            normal = normalized(normal)
        elif mode == "Three points":
            point_1 = np.array(
                [self.plane_p1_x_spin.value(), self.plane_p1_y_spin.value(), self.plane_p1_z_spin.value()],
                dtype=float,
            )
            point_2 = np.array(
                [self.plane_p2_x_spin.value(), self.plane_p2_y_spin.value(), self.plane_p2_z_spin.value()],
                dtype=float,
            )
            point_3 = np.array(
                [self.plane_p3_x_spin.value(), self.plane_p3_y_spin.value(), self.plane_p3_z_spin.value()],
                dtype=float,
            )
            normal = np.cross(point_2 - point_1, point_3 - point_1)
            if np.linalg.norm(normal) < 1e-9:
                raise ValueError("Three-point plane requires non-collinear points.")
            origin = point_1
            normal = normalized(normal)
        else:
            raise ValueError(f"Unsupported plane mode: {mode}")
        plane_size = max(float(np.linalg.norm(spans)), 1.0) * 1.35
        return origin, normal, plane_size

    def view_plane_normal(self) -> None:
        if self.model_mesh is None or self.model_mesh.n_points == 0:
            return
        try:
            origin, normal, plane_size = self.current_plane_definition()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        normal = normalized(normal)
        if abs(float(np.dot(normal, np.array([0.0, 0.0, 1.0], dtype=float)))) > 0.97:
            view_up = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            view_up = np.array([0.0, 0.0, 1.0], dtype=float)
        distance = max(plane_size * 1.25, 10.0)
        position = origin + normal * distance
        self.plotter.camera_position = [tuple(position), tuple(origin), tuple(view_up)]
        self.plotter.enable_parallel_projection()
        self.plotter.camera.SetParallelScale(max(plane_size * 0.45, 1.0))
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def eventFilter(self, watched, event) -> bool:
        if self.plotter is not None and watched is self.plotter.interactor:
            if event.type() == QEvent.MouseMove:
                self.update_live_coordinate_label(event)
            elif event.type() == QEvent.Leave:
                self.live_coordinate_label.setText("Cursor: -")
        return super().eventFilter(watched, event)

    def _cursor_plane_point(self, event) -> Optional[np.ndarray]:
        if self.model_mesh is None or self.model_mesh.n_points == 0 or self.plotter.renderer is None:
            return None
        try:
            origin, normal, _plane_size = self.current_plane_definition()
        except ValueError:
            return None
        try:
            pos = event.position()
            pos_x, pos_y = pos.x(), pos.y()
        except AttributeError:
            pos_x, pos_y = event.x(), event.y()

        renderer = self.plotter.renderer
        render_width, render_height = self.plotter.render_window.GetSize()
        if render_width <= 0 or render_height <= 0:
            return None
        interactor_width = max(float(self.plotter.interactor.width()), 1.0)
        interactor_height = max(float(self.plotter.interactor.height()), 1.0)
        display_x = float(pos_x) * (float(render_width) / interactor_width)
        display_y = float(render_height) - float(pos_y) * (float(render_height) / interactor_height) - 1.0

        renderer.SetDisplayPoint(display_x, display_y, 0.0)
        renderer.DisplayToWorld()
        near_world = np.array(renderer.GetWorldPoint(), dtype=float)
        renderer.SetDisplayPoint(display_x, display_y, 1.0)
        renderer.DisplayToWorld()
        far_world = np.array(renderer.GetWorldPoint(), dtype=float)
        if abs(near_world[3]) < 1e-12 or abs(far_world[3]) < 1e-12:
            return None
        near_point = near_world[:3] / near_world[3]
        far_point = far_world[:3] / far_world[3]
        direction = far_point - near_point
        denominator = float(np.dot(direction, normal))
        if abs(denominator) < 1e-12:
            return None
        distance = float(np.dot(origin - near_point, normal) / denominator)
        point = near_point + direction * distance
        if not np.all(np.isfinite(point)):
            return None
        return point

    def update_live_coordinate_label(self, event) -> None:
        point = self._cursor_plane_point(event)
        if point is None:
            self.live_coordinate_label.setText("Cursor: -")
            return
        self.live_coordinate_label.setText(
            f"Cursor: X={format_decimal(point[0])}, Y={format_decimal(point[1])}, Z={format_decimal(point[2])}"
        )

    def refresh_scene(self, reset_camera: bool = False) -> None:
        if self.plotter is None or self.model_mesh is None or self.model_mesh.n_points == 0:
            return
        camera_state = None
        if not reset_camera and self.plotter.renderer is not None:
            camera_state = capture_camera_state(self.plotter)

        origin, normal, plane_size = self.current_plane_definition()
        plane = pv.Plane(center=origin, direction=normal, i_size=plane_size, j_size=plane_size)
        slice_mesh = self.model_mesh.slice(normal=normal, origin=origin)
        clipped_mesh = self.model_mesh.clip(normal=normal, origin=origin, invert=self.clip_invert_checkbox.isChecked())
        self.set_or_add_mesh(
            "clipped_mesh",
            clipped_mesh,
            "picker_clipped_mesh",
            color="#8c9aad",
            opacity=0.35,
            smooth_shading=True,
            show_edges=False,
            pickable=False,
        )
        self.set_or_add_mesh(
            "slice_mesh",
            slice_mesh,
            "picker_slice_contour",
            color="#d62828",
            line_width=4,
            pickable=False,
        )
        self.set_or_add_mesh(
            "plane_mesh",
            plane,
            "picker_plane",
            color="#93c5fd",
            opacity=0.28,
            smooth_shading=False,
            show_edges=False,
            pickable=True,
        )

        if self.picked_points:
            picked_coordinates = np.array([point for _label, point in self.picked_points], dtype=float)
            picked_cloud = pv.PolyData(picked_coordinates)
            picked_cloud["labels"] = np.array([label for label, _point in self.picked_points], dtype=object)
            self.set_or_add_mesh(
                "picked_points_mesh",
                picked_cloud,
                "picker_points",
                render_points_as_spheres=True,
                point_size=12,
                color="#14b8a6",
            )
            self.plotter.remove_actor("picker_point_labels", render=False)
            self.plotter.add_point_labels(
                self.picked_points_mesh,
                "labels",
                font_size=max(gui_like_label_font_size(self) - 1, 14),
                shape=None,
                point_color="#14b8a6",
                text_color="black",
                always_visible=True,
                name="picker_point_labels",
                render=False,
            )
        else:
            self.set_or_add_mesh(
                "picked_points_mesh",
                pv.PolyData(np.empty((0, 3), dtype=float)),
                "picker_points",
                render_points_as_spheres=True,
                point_size=12,
                color="#14b8a6",
            )
            self.plotter.remove_actor("picker_point_labels", render=False)

        if camera_state is not None:
            restore_camera_state(self.plotter, camera_state)
        else:
            self.plotter.reset_camera()
            self.plotter.enable_parallel_projection()
        self.plotter.render()

        mode = self.plane_mode_combo.currentText()
        if mode == "Axis aligned":
            plane_description = f"{self.plane_axis_combo.currentText()} = {format_decimal(self.plane_value_spin.value())}"
        elif mode == "Point + normal":
            plane_description = (
                f"origin=({format_decimal(self.plane_origin_x_spin.value())}, "
                f"{format_decimal(self.plane_origin_y_spin.value())}, "
                f"{format_decimal(self.plane_origin_z_spin.value())}), "
                f"normal=({format_decimal(normal[0])}, {format_decimal(normal[1])}, {format_decimal(normal[2])})"
            )
        else:
            plane_description = "defined by three points"
        source_name = "mesh" if self.mesh_path is None else self.mesh_path.name
        self.status_label.setText(
            f"{source_name}: active slice plane {plane_description}. "
            "Click on the blue plane within the sliced model to add a point."
        )

    def set_or_add_mesh(self, attr_name: str, mesh: pv.PolyData, actor_name: str, **kwargs) -> None:
        existing = getattr(self, attr_name)
        if mesh is None or mesh.n_points == 0:
            if existing is not None:
                self.plotter.remove_actor(actor_name, render=False)
                setattr(self, attr_name, None)
            return
        if existing is None:
            existing = mesh.copy(deep=True)
            setattr(self, attr_name, existing)
            self.plotter.add_mesh(existing, name=actor_name, render=False, reset_camera=False, **kwargs)
        else:
            existing.deep_copy(mesh)

    def add_point(self, point: np.ndarray, label: Optional[str] = None) -> None:
        if self.model_mesh is None or self.model_mesh.n_points == 0:
            return
        point = np.asarray(point, dtype=float)
        try:
            _origin, normal, _plane_size = self.current_plane_definition()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid plane", str(exc))
            return
        if self.require_inside_checkbox.isChecked() and not point_on_or_inside_mesh(
            self.model_mesh,
            point,
            probe_direction=normal,
        ):
            QMessageBox.warning(
                self,
                "Point outside model",
                "The selected point is outside the model volume. Move the slice plane or enter coordinates inside the mesh.",
            )
            return
        next_label = label or f"G{len(self.picked_points) + 1}"
        self.picked_points.append((next_label, point.copy()))
        self.refresh_points_table()
        self.refresh_scene(reset_camera=False)

    def on_plane_picked(self, point: np.ndarray) -> None:
        if self.model_mesh is None or self.model_mesh.n_points == 0:
            return
        try:
            origin, normal, _plane_size = self.current_plane_definition()
        except ValueError:
            return
        point = np.asarray(point, dtype=float)
        projected_point = point - normal * float(np.dot(point - origin, normal))
        self.add_point(projected_point)

    def add_manual_point(self) -> None:
        self.add_point(
            np.array(
                [self.manual_x_spin.value(), self.manual_y_spin.value(), self.manual_z_spin.value()],
                dtype=float,
            )
        )

    def refresh_points_table(self) -> None:
        self.points_table.setRowCount(len(self.picked_points))
        for row_index, (label, point) in enumerate(self.picked_points):
            values = [label, format_decimal(point[0]), format_decimal(point[1]), format_decimal(point[2])]
            for column_index, value in enumerate(values):
                self.points_table.setItem(row_index, column_index, QTableWidgetItem(value))

    def renumber_point_labels(self) -> None:
        self.picked_points = [(f"G{index + 1}", point) for index, (_label, point) in enumerate(self.picked_points)]

    def remove_selected_points(self) -> None:
        rows = sorted({index.row() for index in self.points_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            del self.picked_points[row]
        self.renumber_point_labels()
        self.refresh_points_table()
        self.refresh_scene(reset_camera=False)

    def undo_last_point(self) -> None:
        if not self.picked_points:
            return
        self.picked_points.pop()
        self.renumber_point_labels()
        self.refresh_points_table()
        self.refresh_scene(reset_camera=False)

    def clear_points(self) -> None:
        self.picked_points = []
        self.refresh_points_table()
        self.refresh_scene(reset_camera=False)

    def save_csv_dialog(self) -> None:
        if not self.picked_points:
            QMessageBox.warning(self, "No points", "Pick or enter one or more points before saving.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save picked points CSV",
            str(Path.cwd() / "picked_prediction_points.csv"),
            "CSV files (*.csv)",
        )
        if not path_str:
            return
        path = Path(path_str)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["label", "model_x", "model_y", "model_z", "stage_x", "stage_y", "stage_z", "path_1", "path_2"]
            )
            for label, point in self.picked_points:
                writer.writerow([label, point[0], point[1], point[2], "", "", "", "", ""])
        self.status_label.setText(f"Saved picked model points to {path}.")


class MainWindow(QMainWindow):
    def __init__(self, enable_3d: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("Sample Setup Fitting")
        self.resize(1600, 960)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.enable_3d = enable_3d
        self.project_path: Optional[Path] = None
        self.mesh_path: Optional[Path] = None
        self.measurement_source_text = "Measurements can be loaded from CSV or edited directly."
        self.report_body_text = (
            "Fit report will appear here.\n\n"
            "Workflow:\n"
            "1. Load an STL/mesh.\n"
            "2. Either run Fit placement or enable Manual Sample Placement.\n"
            "3. Adjust the live stage pose to inspect the setup.\n"
            "4. Compute the imaging map if needed."
        )
        self.model_mesh: Optional[pv.PolyData] = None
        self.fit_transform = None
        self.residual_rows = []
        self.manual_rotation_matrix = np.eye(3, dtype=float)
        self.manual_rotation_display_values = np.zeros(3, dtype=float)
        self.top_splitter = None
        self.bottom_splitter = None
        self.left_splitter = None
        self.tables_splitter = None
        self.controls_scroll = None
        self.main_tabs = None
        self.instrument_setup_dialog = None
        self.settings_dialog = None
        self.setup_group = None
        self.point_picker_panel = None
        self._initial_sizes_applied = False
        self.camera_preset = "iso"
        self.parallel_projection_enabled = True
        self.viewer_font_size_offset = 0
        self.ui_font_size_spin = None
        self.view_buttons = {}
        self.view_button_group = None
        self.scene_initialized = False
        self.stage_world_mesh = None
        self.beam_mesh = None
        self.beam_centerline_mesh = None
        self.slit_mesh = None
        self.gauge_volume_mesh = None
        self.detector_mesh = None
        self.diffraction_bank_1_detector_mesh = None
        self.diffraction_bank_2_detector_mesh = None
        self.detector_map_mesh = None
        self.diffraction_bank_1_map_mesh = None
        self.diffraction_bank_2_map_mesh = None
        self.sight_line_mesh = None
        self.crosshair_h_mesh = None
        self.crosshair_v_mesh = None
        self.stage_axis_labels_mesh = None
        self.stage_x_arrow_neg_mesh = None
        self.stage_x_arrow_pos_mesh = None
        self.stage_x_arrow_neg_secondary_mesh = None
        self.stage_x_arrow_pos_secondary_mesh = None
        self.stage_y_arrow_neg_mesh = None
        self.stage_y_arrow_pos_mesh = None
        self.stage_y_arrow_neg_secondary_mesh = None
        self.stage_y_arrow_pos_secondary_mesh = None
        self.diffraction_vector_bank_1_mesh = None
        self.diffraction_vector_bank_2_mesh = None
        self.parallel_projection_checkbox = None
        self.show_stage_checkbox = None
        self.show_beam_checkbox = None
        self.show_gauge_volume_checkbox = None
        self.show_imaging_detector_checkbox = None
        self.show_diffraction_detectors_checkbox = None
        self.show_feature_points_checkbox = None
        self.show_prediction_points_checkbox = None
        self.show_sample_triad_checkbox = None
        self.show_theodolite_sight_line_checkbox = None
        self.show_diffraction_vectors_checkbox = None
        self.model_axes_actor = None
        self.beam_inside_mesh = None
        self.measurement_points_mesh = None
        self.prediction_points_mesh = None
        self.selected_line_mesh = None
        self.model_world_mesh = None
        self.detector_map_state = None
        self.diffraction_bank_1_map_state = None
        self.diffraction_bank_2_map_state = None

        self._build_ui()
        self.update_placement_status()
        self.update_placement_summary_fields()
        self.update_scene(reset_camera=True)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        self.main_tabs = QTabWidget()
        main_layout.addWidget(self.main_tabs)

        placement_tab = QWidget()
        placement_layout = QVBoxLayout(placement_tab)
        placement_layout.setContentsMargins(0, 0, 0, 0)
        placement_layout.setSpacing(0)
        placement_layout.addWidget(self._build_tab_menu_bar("Placement"))

        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setChildrenCollapsible(False)
        placement_layout.addWidget(self.top_splitter)

        self.viewer_font_increase_shortcut = QShortcut(QKeySequence("Shift+>"), self)
        self.viewer_font_increase_shortcut.setContext(Qt.WindowShortcut)
        self.viewer_font_increase_shortcut.activated.connect(lambda: self.adjust_viewer_font_size(1))
        self.viewer_font_decrease_shortcut = QShortcut(QKeySequence("Shift+<"), self)
        self.viewer_font_decrease_shortcut.setContext(Qt.WindowShortcut)
        self.viewer_font_decrease_shortcut.activated.connect(lambda: self.adjust_viewer_font_size(-1))

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.setChildrenCollapsible(False)
        left_layout.addWidget(self.left_splitter)

        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        self.setup_group = self._build_setup_group()
        controls_layout.addWidget(self._build_setup_quick_controls_group())
        controls_layout.addWidget(self._build_manual_placement_group())
        controls_layout.addWidget(self._build_pose_group())
        controls_layout.addStretch(1)
        controls_container.setMinimumWidth(300)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setWidget(controls_container)
        self.left_splitter.addWidget(self.controls_scroll)
        self.report_box = self._build_report_box()
        self.left_splitter.addWidget(self.report_box)
        self.left_splitter.setStretchFactor(0, 1)
        self.left_splitter.setStretchFactor(1, 0)
        self.refresh_report_box()
        self.top_splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self.bottom_splitter = QSplitter(Qt.Vertical)
        self.bottom_splitter.setChildrenCollapsible(False)
        right_layout.addWidget(self.bottom_splitter)

        view_panel = QWidget()
        view_layout = QVBoxLayout(view_panel)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(6)
        view_layout.addWidget(self._build_view_toolbar())

        if self.enable_3d:
            self.plotter = CursorZoomQtInteractor(self)
            self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.plotter.interactor.installEventFilter(self)
            view_layout.addWidget(self.plotter.interactor)
        else:
            self.plotter = None
            placeholder = QLabel("3D viewport disabled for smoke test mode.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet(
                "background-color: #0d1117; color: #d7d7d7; border: 1px solid #2f3a45; padding: 24px;"
            )
            view_layout.addWidget(placeholder)
        self.bottom_splitter.addWidget(view_panel)

        self.tables_splitter = QSplitter(Qt.Horizontal)
        self.tables_splitter.setChildrenCollapsible(False)
        self.tables_splitter.addWidget(self._build_measurement_section())
        self.tables_splitter.addWidget(self._build_prediction_section())

        self.bottom_splitter.addWidget(self.tables_splitter)
        self.bottom_splitter.setStretchFactor(0, 3)
        self.bottom_splitter.setStretchFactor(1, 1)
        self.top_splitter.addWidget(right_panel)
        self.top_splitter.setStretchFactor(0, 0)
        self.top_splitter.setStretchFactor(1, 1)

        self.main_tabs.addTab(placement_tab, "Placement")
        self.main_tabs.addTab(self._build_point_picker_tab(), "Pick Point")
        self.main_tabs.addTab(self._build_stress_tab(), "Residual Stress")
        self.settings_dialog = self._build_settings_dialog()

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Load a mesh, then use Fit placement or Manual Sample Placement.")

    def _apply_initial_splitter_sizes(self) -> None:
        if self.top_splitter is not None:
            self.top_splitter.setChildrenCollapsible(False)
            self.top_splitter.setSizes([340, 1220])
        if self.bottom_splitter is not None:
            self.bottom_splitter.setChildrenCollapsible(False)
            self.bottom_splitter.setSizes([650, 260])
        if self.left_splitter is not None:
            self.left_splitter.setChildrenCollapsible(False)
            self.left_splitter.setSizes([760, 80])
        if self.tables_splitter is not None:
            self.tables_splitter.setSizes([565, 565])

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._initial_sizes_applied:
            self._apply_initial_splitter_sizes()
            self.apply_initial_table_column_widths()
            self._initial_sizes_applied = True
        if self.controls_scroll is not None:
            self.controls_scroll.verticalScrollBar().setValue(0)

    def closeEvent(self, event) -> None:
        point_picker_panel = getattr(self, "point_picker_panel", None)
        if point_picker_panel is not None:
            point_picker_panel.cleanup_plotter()
        plotter = getattr(self, "plotter", None)
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                try:
                    plotter.Finalize()
                except Exception:
                    pass
            self.plotter = None
        super().closeEvent(event)

    def apply_initial_table_column_widths(self) -> None:
        for table in (getattr(self, "measurement_table", None), getattr(self, "prediction_table", None)):
            if table is None:
                continue
            self._set_equal_table_column_widths(table)

    def _set_equal_table_column_widths(self, table: QTableWidget) -> None:
        column_count = table.columnCount()
        if column_count <= 0:
            return
        digit_width = table.fontMetrics().horizontalAdvance("000000")
        column_width = max(56, digit_width + 18)
        for column in range(column_count):
            table.setColumnWidth(column, column_width)

    def _build_tab_menu_bar(self, tab_name: str) -> QMenuBar:
        menu_bar = QMenuBar()
        menu_bar.setNativeMenuBar(False)
        menu_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        menu_bar.setStyleSheet(
            """
            QMenuBar {
                background-color: #f7f9fc;
                border: 1px solid #d7dee8;
                padding: 1px 4px;
            }
            QMenuBar::item {
                padding: 4px 10px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background-color: #e8f0fb;
            }
            """
        )

        file_menu = menu_bar.addMenu("File")
        setting_menu = menu_bar.addMenu("Setting")
        view_menu = menu_bar.addMenu("View")
        about_menu = menu_bar.addMenu("About")

        if tab_name == "Placement":
            self._populate_placement_file_menu(file_menu)
            self._populate_placement_setting_menu(setting_menu)
            self._populate_placement_view_menu(view_menu)
            self._add_tab_menu_action(
                about_menu,
                "About Placement",
                lambda: self.show_tab_about(
                    "Placement",
                    "Use Placement to load a sample mesh, fit or manually set the model-to-stage transform, "
                    "inspect the instrument view, and export placement outputs.",
                ),
            )
        elif tab_name == "Pick Point":
            self._populate_pick_point_file_menu(file_menu)
            self._populate_pick_point_setting_menu(setting_menu)
            self._populate_pick_point_view_menu(view_menu)
            self._add_tab_menu_action(
                about_menu,
                "About Pick Point",
                lambda: self.show_tab_about(
                    "Pick Point",
                    "Use Pick Point to load a mesh, define a slice plane, pick model coordinates, and save the "
                    "picked points as CSV.",
                ),
            )
        elif tab_name == "Residual Stress":
            self._populate_residual_stress_file_menu(file_menu)
            self._populate_residual_stress_setting_menu(setting_menu)
            self._populate_residual_stress_view_menu(view_menu)
            self._add_tab_menu_action(about_menu, "Equations", self.show_stress_equations_dialog)
            about_menu.addSeparator()
            self._add_tab_menu_action(
                about_menu,
                "About Residual Stress",
                lambda: self.show_tab_about(
                    "Residual Stress",
                    "Use Residual Stress to enter lattice parameters and D0 values, calculate microstrain and "
                    "3D stress, and save the table as CSV.",
                ),
            )
        return menu_bar

    def _add_tab_menu_action(
        self,
        menu,
        label: str,
        slot: Callable[[], None],
        *,
        checkable: bool = False,
        checked: bool = False,
        status_tip: Optional[str] = None,
    ) -> QAction:
        action = QAction(label, self)
        action.setCheckable(checkable)
        if checkable:
            action.setChecked(checked)
        action.setStatusTip(status_tip or label)
        action.triggered.connect(lambda _checked=False, target=slot: target())
        menu.addAction(action)
        return action

    def _populate_placement_file_menu(self, menu) -> None:
        self._add_tab_menu_action(menu, "Load Project", self.load_project_dialog)
        self._add_tab_menu_action(menu, "Save Project", self.save_project_dialog)
        menu.addSeparator()
        self._add_tab_menu_action(menu, "Load STL/mesh", self.load_mesh_dialog)
        self._add_tab_menu_action(menu, "Clear Mesh", self.clear_mesh)
        menu.addSeparator()
        self._add_tab_menu_action(menu, "Export Fit JSON", self.export_json_dialog)
        menu.addSeparator()
        self._add_tab_menu_action(menu, "Exit", self.close)

    def _populate_placement_setting_menu(self, menu) -> None:
        self._add_tab_menu_action(menu, "Open Settings", self.open_settings_tab)
        self._add_tab_menu_action(menu, "Instrument Setup", self.open_instrument_setup_dialog)

    def _populate_placement_view_menu(self, menu) -> None:
        preset_actions = []
        for label, preset in (
            ("Isometric View", "iso"),
            ("View +X", "+x"),
            ("View -X", "-x"),
            ("View +Y", "+y"),
            ("View -Y", "-y"),
            ("View +Z", "+z"),
            ("View -Z", "-z"),
            ("Theodolite View", "theodolite"),
        ):
            action = self._add_tab_menu_action(
                menu,
                label,
                lambda target_preset=preset: self.set_camera_preset(target_preset),
                checkable=True,
                checked=self.camera_preset == preset,
            )
            preset_actions.append((action, preset))
        menu.addSeparator()

        toggle_specs = [
            ("Parallel Projection", "parallel_projection_checkbox"),
            ("Show Stage", "show_stage_checkbox"),
            ("Show Beam", "show_beam_checkbox"),
            ("Show Gauge Volume", "show_gauge_volume_checkbox"),
            ("Show Imaging Detector", "show_imaging_detector_checkbox"),
            ("Show Diffraction Detectors", "show_diffraction_detectors_checkbox"),
            ("Show Feature Points", "show_feature_points_checkbox"),
            ("Show Prediction Points", "show_prediction_points_checkbox"),
            ("Show Sample Triad", "show_sample_triad_checkbox"),
            ("Show Theodolite Sight Line", "show_theodolite_sight_line_checkbox"),
            ("Show Diffraction Vectors", "show_diffraction_vectors_checkbox"),
        ]
        toggle_actions = []
        for label, button_attr in toggle_specs:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setStatusTip(label)
            action.triggered.connect(
                lambda checked=False, attr=button_attr: self.set_placement_view_toggle(attr, checked)
            )
            menu.addAction(action)
            toggle_actions.append((action, button_attr))

        menu.aboutToShow.connect(lambda: self.sync_placement_view_menu_actions(preset_actions, toggle_actions))

    def _populate_pick_point_file_menu(self, menu) -> None:
        self._add_tab_menu_action(menu, "Load STL/mesh", self.load_mesh_dialog)
        self._add_tab_menu_action(menu, "Clear Mesh", self.clear_mesh)
        menu.addSeparator()
        self._add_tab_menu_action(menu, "Save Picked Points", self.save_picked_points_dialog)
        menu.addSeparator()
        self._add_tab_menu_action(menu, "Exit", self.close)

    def _populate_pick_point_setting_menu(self, menu) -> None:
        self._add_tab_menu_action(menu, "Open Settings", self.open_settings_tab)

    def _populate_pick_point_view_menu(self, menu) -> None:
        self._add_tab_menu_action(menu, "Normal View", self.point_picker_normal_view)
        self._add_tab_menu_action(menu, "Reset View", self.reset_point_picker_view)

    def _populate_residual_stress_file_menu(self, menu) -> None:
        self._add_tab_menu_action(menu, "Save CSV", self.save_stress_csv_dialog)
        menu.addSeparator()
        self._add_tab_menu_action(menu, "Exit", self.close)

    def _populate_residual_stress_setting_menu(self, menu) -> None:
        self._add_tab_menu_action(menu, "Open Settings", self.open_settings_tab)

    def _populate_residual_stress_view_menu(self, menu) -> None:
        self._add_tab_menu_action(menu, "Calculate", self.calculate_residual_stress)
        self._add_tab_menu_action(menu, "Clear Outputs", self.clear_stress_outputs)
        menu.addSeparator()
        self._add_tab_menu_action(menu, "Add Row", self.add_stress_row)
        self._add_tab_menu_action(menu, "Remove Row", self.remove_selected_stress_rows)

    def sync_placement_view_menu_actions(self, preset_actions, toggle_actions) -> None:
        for action, preset in preset_actions:
            action.setChecked(preset == self.camera_preset)
        for action, button_attr in toggle_actions:
            button = getattr(self, button_attr, None)
            if button is None:
                action.setChecked(False)
                action.setEnabled(False)
                continue
            action.setEnabled(True)
            action.setChecked(button.isChecked())

    def set_placement_view_toggle(self, button_attr: str, checked: bool) -> None:
        button = getattr(self, button_attr, None)
        if button is None:
            return
        button.setChecked(checked)

    def open_settings_tab(self) -> None:
        if self.settings_dialog is None:
            self.settings_dialog = self._build_settings_dialog()
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def show_tab_about(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def save_picked_points_dialog(self) -> None:
        if self.point_picker_panel is None:
            return
        self.point_picker_panel.save_csv_dialog()

    def point_picker_normal_view(self) -> None:
        if self.point_picker_panel is None:
            return
        self.point_picker_panel.view_plane_normal()

    def reset_point_picker_view(self) -> None:
        if self.point_picker_panel is None:
            return
        self.point_picker_panel.refresh_scene(reset_camera=True)

    def _build_files_group(self) -> QGroupBox:
        group = QGroupBox("Files")
        layout = QVBoxLayout(group)
        icons_dir = Path(__file__).resolve().parent / "icons"
        group.setStyleSheet(
            """
            QToolButton {
                padding: 0px;
                min-width: 30px;
                min-height: 30px;
                border: 1px solid #b9c2ce;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QToolButton:hover {
                background-color: #f3f7fb;
                border-color: #8da2bd;
            }
            QToolButton:pressed {
                background-color: #e6eef8;
            }
            """
        )

        button_flow = FlowLayout(spacing=10)

        save_project_button = make_toolbar_icon_button(
            QIcon(str(icons_dir / "save-project.svg")),
            "Save project",
            self.save_project_dialog,
            button_size=30,
            icon_size=18,
        )
        load_project_button = make_toolbar_icon_button(
            QIcon(str(icons_dir / "load-project.svg")),
            "Load project",
            self.load_project_dialog,
            button_size=30,
            icon_size=18,
        )

        load_mesh_button = make_toolbar_icon_button(
            QIcon(str(icons_dir / "import-stl-mesh.svg")),
            "Load STL/mesh",
            self.load_mesh_dialog,
            button_size=30,
            icon_size=18,
        )
        clear_mesh_button = make_toolbar_icon_button(
            QIcon(str(icons_dir / "clear-stl-mesh.svg")),
            "Clear mesh",
            self.clear_mesh,
            button_size=30,
            icon_size=18,
        )

        export_button = make_toolbar_icon_button(
            QIcon(str(icons_dir / "export-fit-json.svg")),
            "Export fit JSON",
            self.export_json_dialog,
            button_size=30,
            icon_size=18,
        )

        for button in (
            load_mesh_button,
            clear_mesh_button,
            load_project_button,
            save_project_button,
            export_button,
        ):
            button_flow.addWidget(button)

        layout.addLayout(button_flow)
        return group

    def _build_setup_quick_controls_group(self) -> QGroupBox:
        group = QGroupBox("Setup")
        layout = make_form_layout(group, compact_fields=True)

        slit_width_proxy = make_spin_box(self.slit_width.value(), self.slit_width.minimum(), self.slit_width.maximum())
        slit_width_proxy.setDecimals(self.slit_width.decimals())
        slit_width_proxy.setSingleStep(self.slit_width.singleStep())

        slit_height_proxy = make_spin_box(
            self.slit_height.value(),
            self.slit_height.minimum(),
            self.slit_height.maximum(),
        )
        slit_height_proxy.setDecimals(self.slit_height.decimals())
        slit_height_proxy.setSingleStep(self.slit_height.singleStep())

        collimator_proxy = NoWheelComboBox()
        for index in range(self.collimator.count()):
            collimator_proxy.addItem(self.collimator.itemText(index))
        collimator_proxy.setCurrentText(self.collimator.currentText())

        material_proxy = NoWheelComboBox()
        for index in range(self.count_time_material.count()):
            material_proxy.addItem(self.count_time_material.itemText(index))
        material_proxy.setCurrentText(self.count_time_material.currentText())

        self._bind_spin_box_proxy(slit_width_proxy, self.slit_width)
        self._bind_spin_box_proxy(slit_height_proxy, self.slit_height)
        self._bind_combo_box_proxy(collimator_proxy, self.collimator)
        self._bind_combo_box_proxy(material_proxy, self.count_time_material)

        layout.addRow("Slit width", slit_width_proxy)
        layout.addRow("Slit height", slit_height_proxy)
        layout.addRow("Collimator", collimator_proxy)
        layout.addRow("Material", material_proxy)
        return group

    def _bind_spin_box_proxy(self, proxy, target) -> None:
        def sync_proxy(value: float) -> None:
            previous_state = proxy.blockSignals(True)
            proxy.setValue(float(value))
            proxy.blockSignals(previous_state)

        def sync_target(value: float) -> None:
            previous_state = target.blockSignals(True)
            target.setValue(float(value))
            target.blockSignals(previous_state)
            self.on_view_parameter_changed()

        proxy.valueChanged.connect(sync_target)
        target.valueChanged.connect(sync_proxy)

    def _bind_combo_box_proxy(self, proxy: QComboBox, target: QComboBox) -> None:
        def sync_proxy(text: str) -> None:
            previous_state = proxy.blockSignals(True)
            proxy.setCurrentText(str(text))
            proxy.blockSignals(previous_state)

        def sync_target(text: str) -> None:
            previous_state = target.blockSignals(True)
            target.setCurrentText(str(text))
            target.blockSignals(previous_state)
            self.on_view_parameter_changed()

        proxy.currentTextChanged.connect(sync_target)
        target.currentTextChanged.connect(sync_proxy)

    def _build_point_picker_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._build_tab_menu_bar("Pick Point"))

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(6)
        toolbar.setStyleSheet(
            """
            QToolButton {
                padding: 0px;
                min-width: 30px;
                min-height: 30px;
                border: 1px solid #b9c2ce;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QToolButton:hover {
                background-color: #f3f7fb;
                border-color: #8da2bd;
            }
            QToolButton:pressed {
                background-color: #e6eef8;
            }
            """
        )
        load_mesh_button = make_toolbar_action_button(
            "act_mesh_load",
            "Load STL/mesh",
            self.load_mesh_dialog,
            button_size=30,
            icon_size=18,
        )
        clear_mesh_button = make_toolbar_action_button(
            "act_mesh_clear",
            "Clear mesh",
            self.clear_mesh,
            button_size=30,
            icon_size=18,
        )
        toolbar_layout.addWidget(load_mesh_button)
        toolbar_layout.addWidget(clear_mesh_button)
        toolbar_layout.addStretch(1)
        layout.addWidget(toolbar)

        self.point_picker_panel = PointPickerPanel(self, enable_3d=self.enable_3d)
        self.point_picker_panel.set_mesh(self.model_mesh, self.mesh_path)
        layout.addWidget(self.point_picker_panel)
        return tab

    def _build_stress_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._build_tab_menu_bar("Residual Stress"))

        metadata_group = QGroupBox("Stress Metadata")
        metadata_layout = QVBoxLayout(metadata_group)
        self.stress_title_edit = QLineEdit()
        self.stress_sample_name_edit = QLineEdit()
        self.stress_sample_material_edit = QLineEdit()
        self.stress_youngs_modulus_mpa = make_spin_box(210000.0, 0.001, 1_000_000_000.0, step=1000.0)
        self.stress_youngs_modulus_mpa.setDecimals(0)
        self.stress_poissons_ratio = make_spin_box(0.300, -0.999, 0.499, step=0.01)
        self.stress_poissons_ratio.setDecimals(3)

        title_layout = QFormLayout()
        title_layout.addRow("Title", self.stress_title_edit)
        metadata_layout.addLayout(title_layout)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Sample name"))
        sample_row.addWidget(self.stress_sample_name_edit, stretch=1)
        sample_row.addWidget(QLabel("Sample material"))
        sample_row.addWidget(self.stress_sample_material_edit, stretch=1)
        metadata_layout.addLayout(sample_row)

        elastic_row = QHBoxLayout()
        elastic_row.addWidget(QLabel("Young's modulus (MPa)"))
        elastic_row.addWidget(self.stress_youngs_modulus_mpa)
        elastic_row.addWidget(QLabel("Poisson's ratio"))
        elastic_row.addWidget(self.stress_poissons_ratio)
        elastic_row.addStretch(1)
        metadata_layout.addLayout(elastic_row)

        layout.addWidget(metadata_group)

        layout.addWidget(self._build_stress_toolbar())
        self.stress_table = self._build_stress_table()
        layout.addWidget(self.stress_table, stretch=1)

        self.stress_status_label = QLabel(
            "Paste lattice parameters and D0 values into the table, then calculate 3D microstrain and stress in MPa."
        )
        self.stress_status_label.setWordWrap(True)
        layout.addWidget(self.stress_status_label)
        return tab

    def _build_settings_content(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout(appearance_group)
        self.ui_font_size_spin = NoWheelDoubleSpinBox()
        self.ui_font_size_spin.setDecimals(1)
        self.ui_font_size_spin.setRange(MIN_UI_FONT_POINT_SIZE, 18.0)
        self.ui_font_size_spin.setSingleStep(0.5)
        app = QApplication.instance()
        initial_font_size = DEFAULT_UI_FONT_POINT_SIZE
        if app is not None:
            initial_font_size = current_application_ui_font_size(app)
        self.ui_font_size_spin.setValue(initial_font_size)
        self.ui_font_size_spin.valueChanged.connect(self.on_ui_font_size_changed)
        appearance_layout.addRow("UI font size (pt)", self.ui_font_size_spin)

        help_label = QLabel(
            "Changes apply to the Qt interface immediately. The 3D viewer label size remains controlled separately."
        )
        help_label.setWordWrap(True)
        appearance_layout.addRow(help_label)

        layout.addWidget(appearance_group)
        layout.addStretch(1)
        return content

    def _build_settings_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.setModal(False)
        dialog.resize(420, 220)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_settings_content())

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(12, 0, 12, 12)
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)
        return dialog

    def _build_stress_toolbar(self) -> QWidget:
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)

        add_row_button = QPushButton("Add row")
        add_row_button.clicked.connect(lambda: self.add_stress_row())
        remove_row_button = QPushButton("Remove row")
        remove_row_button.clicked.connect(self.remove_selected_stress_rows)
        clear_outputs_button = QPushButton("Clear outputs")
        clear_outputs_button.clicked.connect(self.clear_stress_outputs)
        calculate_button = QPushButton("Calculate")
        calculate_button.clicked.connect(self.calculate_residual_stress)
        equations_button = QPushButton("Equations")
        equations_button.clicked.connect(self.show_stress_equations_dialog)
        save_csv_button = QPushButton("Save CSV")
        save_csv_button.clicked.connect(self.save_stress_csv_dialog)

        layout.addWidget(add_row_button)
        layout.addWidget(remove_row_button)
        layout.addWidget(clear_outputs_button)
        layout.addWidget(calculate_button)
        layout.addWidget(equations_button)
        layout.addWidget(save_csv_button)
        layout.addStretch(1)
        return toolbar

    def _build_stress_table(self) -> QTableWidget:
        table = SpreadsheetTableWidget(20, len(STRESS_TABLE_HEADERS))
        table.setHorizontalHeader(StressTableHeaderView(Qt.Horizontal, table))
        table.setHorizontalHeaderLabels(STRESS_TABLE_HEADERS)
        table.setSelectionBehavior(QAbstractItemView.SelectItems)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setMinimumHeight(320)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        table.verticalHeader().setDefaultSectionSize(28)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.setStretchLastSection(False)
        table.read_only_columns = set(range(14, len(STRESS_TABLE_HEADERS)))
        table.after_paste = self.normalize_stress_input_cells
        table.itemChanged.connect(self.on_stress_item_changed)
        self.update_stress_table_metrics(table)
        return table

    def update_stress_table_metrics(self, table: Optional[QTableWidget] = None) -> None:
        target_table = self.stress_table if table is None else table
        if target_table is None:
            return
        column_width = max(target_table.fontMetrics().horizontalAdvance("0" * 8) + 18, 60)
        header = target_table.horizontalHeader()
        header.setDefaultSectionSize(column_width)
        for column in range(target_table.columnCount()):
            target_table.setColumnWidth(column, column_width)
        if isinstance(header, StressTableHeaderView):
            header.setMinimumHeight(header._target_header_height())
        header.updateGeometry()
        header.viewport().update()

    def _build_setup_group(self) -> QGroupBox:
        group = QGroupBox("Setup Geometry")
        layout = make_form_layout(group, compact_fields=True)

        self.pivot_x = make_spin_box(0.0, -100000.0, 100000.0)
        self.pivot_y = make_spin_box(0.0, -100000.0, 100000.0)
        self.pivot_z = make_spin_box(0.0, -100000.0, 100000.0)
        self.theodolite_x = make_spin_box(-250.0, -100000.0, 100000.0)
        self.theodolite_y = make_spin_box(-250.0, -100000.0, 100000.0)
        self.theodolite_z = make_spin_box(0.0, -100000.0, 100000.0)
        self.slit_x = make_spin_box(-300.0, -100000.0, 100000.0)
        self.slit_y = make_spin_box(0.0, -100000.0, 100000.0)
        self.slit_z = make_spin_box(0.0, -100000.0, 100000.0)
        self.slit_width = make_spin_box(4.0, 0.001, 100000.0)
        self.slit_height = make_spin_box(4.0, 0.001, 100000.0)
        self.collimator = NoWheelComboBox()
        self.collimator.addItems(["0.5", "1", "2", "3", "4"])
        self.collimator.setCurrentText("4")
        self.count_time_material = NoWheelComboBox()
        self.count_time_material.addItems(list(RIETVELD_COUNT_TIME_LAWS.keys()))
        self.beam_length = make_spin_box(700.0, 1.0, 100000.0)
        self.detector_width_y = make_spin_box(100.0, 10.0, 200.0)
        self.detector_height_z = make_spin_box(100.0, 10.0, 200.0)
        self.detector_map_pixel_size_y = make_spin_box(0.1, 0.01, 10.0, step=0.01)
        self.detector_map_pixel_size_z = make_spin_box(0.1, 0.01, 10.0, step=0.01)
        self.stage_size_x = make_spin_box(500.0, 1.0, 100000.0)
        self.stage_size_y = make_spin_box(500.0, 1.0, 100000.0)
        self.stage_size_z = make_spin_box(50.0, 1.0, 100000.0)
        self.stage_offset_x = make_spin_box(0.0, -100000.0, 100000.0)
        self.stage_offset_y = make_spin_box(0.0, -100000.0, 100000.0)
        self.stage_offset_z = make_spin_box(-100.0, -100000.0, 100000.0)

        for widget in (
            self.pivot_x,
            self.pivot_y,
            self.pivot_z,
            self.theodolite_x,
            self.theodolite_y,
            self.theodolite_z,
            self.slit_x,
            self.slit_y,
            self.slit_z,
            self.slit_width,
            self.slit_height,
            self.beam_length,
            self.detector_width_y,
            self.detector_height_z,
            self.stage_size_x,
            self.stage_size_y,
            self.stage_size_z,
            self.stage_offset_x,
            self.stage_offset_y,
            self.stage_offset_z,
        ):
            widget.valueChanged.connect(self.on_view_parameter_changed)
        self.collimator.currentTextChanged.connect(lambda _text: self.on_view_parameter_changed())
        for widget in (
            self.detector_map_pixel_size_y,
            self.detector_map_pixel_size_z,
        ):
            widget.valueChanged.connect(self.on_detector_map_parameter_changed)

        instrument_setup_button = QPushButton("Instrument setup...")
        instrument_setup_button.clicked.connect(self.open_instrument_setup_dialog)
        instrument_setup_button.setMaximumWidth(180)
        instrument_setup_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        instrument_setup_row = QHBoxLayout()
        instrument_setup_row.setContentsMargins(0, 0, 0, 0)
        instrument_setup_row.addWidget(instrument_setup_button)
        instrument_setup_row.addStretch(1)
        layout.addRow(instrument_setup_row)
        layout.addRow(self._separator())
        layout.addRow("Slit width", self.slit_width)
        layout.addRow("Slit height", self.slit_height)
        layout.addRow("Collimator", self.collimator)
        layout.addRow("Material", self.count_time_material)
        self.instrument_setup_dialog = self._build_instrument_setup_dialog()
        return group

    def _build_instrument_setup_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("Instrument Setup")
        dialog.setModal(False)
        dialog.resize(360, 560)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        form.addRow("Pivot X", self.pivot_x)
        form.addRow("Pivot Y", self.pivot_y)
        form.addRow("Pivot Z", self.pivot_z)
        form.addRow(self._separator())
        form.addRow("Theodolite X", self.theodolite_x)
        form.addRow("Theodolite Y", self.theodolite_y)
        form.addRow("Theodolite Z", self.theodolite_z)
        form.addRow(self._separator())
        form.addRow("Slit X", self.slit_x)
        form.addRow("Slit Y", self.slit_y)
        form.addRow("Slit Z", self.slit_z)
        form.addRow("Beam length", self.beam_length)
        form.addRow(self._separator())
        form.addRow("Detector width Y", self.detector_width_y)
        form.addRow("Detector height Z", self.detector_height_z)
        form.addRow("Map pixel Y", self.detector_map_pixel_size_y)
        form.addRow("Map pixel Z", self.detector_map_pixel_size_z)
        form.addRow(self._separator())
        form.addRow("Stage size X", self.stage_size_x)
        form.addRow("Stage size Y", self.stage_size_y)
        form.addRow("Stage thickness", self.stage_size_z)
        form.addRow("Stage offset X", self.stage_offset_x)
        form.addRow("Stage offset Y", self.stage_offset_y)
        form.addRow("Stage offset Z", self.stage_offset_z)
        layout.addLayout(form)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button, alignment=Qt.AlignRight)
        return dialog

    def open_instrument_setup_dialog(self) -> None:
        if self.instrument_setup_dialog is None:
            self.instrument_setup_dialog = self._build_instrument_setup_dialog()
        self.instrument_setup_dialog.show()
        self.instrument_setup_dialog.raise_()
        self.instrument_setup_dialog.activateWindow()

    def _build_pose_group(self) -> QGroupBox:
        group = QGroupBox("Live Stage Pose")
        layout = make_form_layout(group, compact_fields=True)

        self.pose_x = make_spin_box(0.0, -100000.0, 100000.0)
        self.pose_y = make_spin_box(0.0, -100000.0, 100000.0)
        self.pose_z = make_spin_box(0.0, -100000.0, 100000.0)
        self.pose_omega = make_spin_box(0.0, -3600.0, 3600.0, step=1.0)

        for widget in (self.pose_x, self.pose_y, self.pose_z, self.pose_omega):
            widget.valueChanged.connect(self.on_view_parameter_changed)

        layout.addRow("Stage X", self.pose_x)
        layout.addRow("Stage Y", self.pose_y)
        layout.addRow("Stage Z", self.pose_z)
        layout.addRow("Omega", self.pose_omega)

        button_row = QHBoxLayout()
        reset_pose_button = QPushButton("Reset pose")
        reset_pose_button.clicked.connect(self.reset_pose)
        reset_pose_button.setMaximumWidth(110)
        reset_pose_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        button_row.addWidget(reset_pose_button)
        button_row.addStretch(1)
        layout.addRow(button_row)
        return group

    def _build_manual_placement_group(self) -> QGroupBox:
        group = QGroupBox("Manual Sample Placement")
        layout = make_form_layout(group, compact_fields=True)
        manual_placement_tooltip = (
            "Manual placement overrides the model-to-stage transform used by the viewer and point-to-pivot tools.\n\n"
            "Model->Stage X/Y/Z translates the sample in stage coordinates. Changing these values makes the model "
            "shift in the 3D view relative to the stage, beam, and detectors.\n\n"
            "Local Rot X/Y/Z shows the current accumulated orientation in degrees. Editing one value applies the "
            "change as an incremental rotation about the sample's current local triad axis, then the boxes update to "
            "the new accumulated orientation.\n\n"
            "When manual placement is enabled, these values are used instead of the fitted transform. "
            "The Live Stage Pose controls still apply separately on top as the current stage readout."
        )

        self.manual_placement_enabled_checkbox = QCheckBox("Use manual placement")
        self.manual_placement_enabled_checkbox.toggled.connect(self.on_manual_placement_mode_toggled)
        manual_help_button = QToolButton()
        manual_help_button.setText("?")
        manual_help_button.setToolTip(manual_placement_tooltip)
        manual_help_button.setToolTipDuration(10000)
        manual_help_button.setStatusTip("Explain how manual sample placement works.")
        manual_help_button.setAutoRaise(True)
        manual_help_button.setCursor(Qt.WhatsThisCursor)
        manual_help_button.setFixedSize(18, 18)
        manual_help_button.setStyleSheet(
            "QToolButton { font-weight: 600; border: 1px solid #b9c2ce; border-radius: 9px; padding: 0px; }"
            "QToolButton:hover { background-color: #f3f7fb; border-color: #8da2bd; }"
        )
        checkbox_row = QHBoxLayout()
        checkbox_row.setContentsMargins(0, 0, 0, 0)
        checkbox_row.setSpacing(6)
        checkbox_row.addWidget(self.manual_placement_enabled_checkbox)
        checkbox_row.addWidget(manual_help_button)
        checkbox_row.addStretch(1)

        self.manual_tx = make_spin_box(0.0, -100000.0, 100000.0)
        self.manual_ty = make_spin_box(0.0, -100000.0, 100000.0)
        self.manual_tz = make_spin_box(0.0, -100000.0, 100000.0)
        self.manual_rx = make_spin_box(0.0, -3600.0, 3600.0, step=1.0)
        self.manual_ry = make_spin_box(0.0, -3600.0, 3600.0, step=1.0)
        self.manual_rz = make_spin_box(0.0, -3600.0, 3600.0, step=1.0)

        for widget in (
            self.manual_tx,
            self.manual_ty,
            self.manual_tz,
        ):
            widget.valueChanged.connect(self.on_manual_placement_changed)
        for widget in (
            self.manual_rx,
            self.manual_ry,
            self.manual_rz,
        ):
            widget.valueChanged.connect(self.on_manual_rotation_increment_changed)

        reset_button = QPushButton("Reset")
        reset_button.setToolTip("Reset the manual model-to-stage translation and rotation to zero.")
        reset_button.clicked.connect(self.reset_manual_placement)
        reset_button.setMaximumWidth(90)
        reset_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        load_fit_button = QPushButton("Load transform")
        load_fit_button.setToolTip("Copy the current fitted transform into the manual placement fields and enable manual placement.")
        load_fit_button.clicked.connect(self.load_fit_into_manual)
        load_fit_button.setMaximumWidth(130)
        load_fit_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        button_row = QHBoxLayout()
        button_row.addWidget(reset_button)
        button_row.addWidget(load_fit_button)
        button_row.addStretch(1)

        layout.addRow(checkbox_row)
        layout.addRow("Model->Stage X", self.manual_tx)
        layout.addRow("Model->Stage Y", self.manual_ty)
        layout.addRow("Model->Stage Z", self.manual_tz)
        layout.addRow("Local Rot X", self.manual_rx)
        layout.addRow("Local Rot Y", self.manual_ry)
        layout.addRow("Local Rot Z", self.manual_rz)
        layout.addRow(button_row)
        self.sync_manual_rotation_spin_boxes_from_matrix()
        return group

    def _build_view_toolbar(self) -> QWidget:
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        toolbar.setStyleSheet(
            """
            QPushButton {
                padding: 6px 12px;
            }
            QPushButton:checked {
                background-color: #cfe8ff;
                border: 2px solid #3a7bd5;
                font-weight: 600;
            }
            QToolButton {
                padding: 0px;
                min-width: 24px;
                min-height: 24px;
                border: 1px solid #b9c2ce;
                border-radius: 6px;
                background-color: #ffffff;
            }
            QToolButton:hover {
                background-color: #f3f7fb;
                border-color: #8da2bd;
            }
            QToolButton:checked {
                background-color: #1f6fca;
                border: 1px solid #1f6fca;
            }
            """
        )
        self.view_button_group = QButtonGroup(toolbar)
        self.view_button_group.setExclusive(True)

        buttons = [
            ("cam_iso", "Isometric view", "iso"),
            ("cam_px", "View +X", "+x"),
            ("cam_nx", "View -X", "-x"),
            ("cam_py", "View +Y", "+y"),
            ("cam_ny", "View -Y", "-y"),
            ("cam_pz", "View +Z", "+z"),
            ("cam_nz", "View -Z", "-z"),
            ("cam_theodolite", "Theodolite view", "theodolite"),
        ]
        for icon_kind, tooltip, preset in buttons:
            button = make_toolbar_preset_button(
                icon_kind,
                tooltip,
                lambda _checked=False, view=preset: self.set_camera_preset(view),
            )
            self.view_button_group.addButton(button)
            self.view_buttons[preset] = button
            layout.addWidget(button)

        toggle_specs = [
            ("parallel_projection_checkbox", "parallel", "Parallel projection", True, self.on_projection_toggled),
            ("show_stage_checkbox", "stage", "Show stage", True, self.on_overlay_visibility_changed),
            ("show_beam_checkbox", "beam", "Show beam", True, self.on_overlay_visibility_changed),
            ("show_gauge_volume_checkbox", "cube", "Show gauge volume", True, self.on_overlay_visibility_changed),
            ("show_imaging_detector_checkbox", "imaging", "Show imaging detector", False, self.on_overlay_visibility_changed),
            ("show_diffraction_detectors_checkbox", "diffraction", "Show diffraction detectors", True, self.on_overlay_visibility_changed),
            ("show_feature_points_checkbox", "features", "Show feature points", True, self.on_overlay_visibility_changed),
            ("show_prediction_points_checkbox", "predicted", "Show predicted points", True, self.on_overlay_visibility_changed),
            ("show_sample_triad_checkbox", "triad", "Show sample triad", True, self.on_overlay_visibility_changed),
            ("show_theodolite_sight_line_checkbox", "sight", "Show sight line", True, self.on_overlay_visibility_changed),
            ("show_diffraction_vectors_checkbox", "diffraction_vectors", "Show diffraction vectors", True, self.on_overlay_visibility_changed),
        ]
        for attribute_name, icon_kind, tooltip, checked, slot in toggle_specs:
            button = make_toolbar_toggle_button(icon_kind, tooltip, checked, slot)
            setattr(self, attribute_name, button)
            layout.addWidget(button)

        compute_detector_map_button = make_toolbar_action_button(
            "act_detector_map",
            "Compute imaging map",
            self.compute_detector_map,
        )
        compute_diffraction_map_button = make_toolbar_action_button(
            "act_diffraction_map",
            "Compute diffraction path",
            self.compute_diffraction_map,
        )
        export_detector_map_button = make_toolbar_action_button(
            "act_map_export",
            "Export detector map",
            self.export_detector_map_dialog,
        )
        layout.addWidget(compute_detector_map_button)
        layout.addWidget(compute_diffraction_map_button)
        layout.addWidget(export_detector_map_button)
        layout.addStretch(1)
        self.sync_view_button_states()
        return toolbar

    def _build_measurement_toolbar(self) -> QWidget:
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        toolbar.setStyleSheet(
            """
            QToolButton {
                padding: 0px;
                min-width: 24px;
                min-height: 24px;
                border: 1px solid #b9c2ce;
                border-radius: 6px;
                background-color: #ffffff;
            }
            QToolButton:hover {
                background-color: #f3f7fb;
                border-color: #8da2bd;
            }
            QToolButton:pressed {
                background-color: #e6eef8;
            }
            """
        )

        add_row_button = make_toolbar_action_button("act_add", "Add row", self.add_measurement_row)
        load_csv_button = make_toolbar_action_button("act_table_load", "Load CSV", self.load_csv_dialog)
        save_csv_button = make_toolbar_action_button("act_table_save", "Save CSV", self.save_csv_dialog)
        remove_row_button = make_toolbar_action_button("act_remove", "Remove row", self.remove_selected_rows)
        move_to_pivot_button = make_toolbar_action_button(
            "act_move_pivot",
            "Move point to pivot",
            self.move_selected_point_to_pivot,
        )
        fit_button = make_toolbar_action_button("act_fit", "Fit placement", self.fit_placement)
        clear_placement_button = make_toolbar_action_button(
            "act_clear_placement",
            "Clear placement",
            self.clear_placement,
        )
        self.auto_move_to_pivot_checkbox = QCheckBox("Auto move on select")
        self.auto_move_to_pivot_checkbox.setChecked(True)

        layout.addWidget(add_row_button)
        layout.addWidget(remove_row_button)
        layout.addWidget(load_csv_button)
        layout.addWidget(save_csv_button)
        layout.addWidget(move_to_pivot_button)
        layout.addWidget(fit_button)
        layout.addWidget(clear_placement_button)
        layout.addWidget(self.auto_move_to_pivot_checkbox)
        layout.addStretch(1)
        return toolbar

    def _build_section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(False)
        label.setMinimumHeight(24)
        label.setStyleSheet("font-weight: 600;")
        return label

    def _build_measurement_section(self) -> QWidget:
        section = QWidget()
        section.setMinimumHeight(0)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._build_section_title("Feature points for transform fit"))
        layout.addWidget(self._build_measurement_toolbar())
        self.measurement_table = self._build_measurement_table()
        layout.addWidget(self.measurement_table)
        return section

    def _build_prediction_toolbar(self) -> QWidget:
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        toolbar.setStyleSheet(
            """
            QToolButton {
                padding: 0px;
                min-width: 24px;
                min-height: 24px;
                border: 1px solid #b9c2ce;
                border-radius: 6px;
                background-color: #ffffff;
            }
            QToolButton:hover {
                background-color: #f3f7fb;
                border-color: #8da2bd;
            }
            QToolButton:pressed {
                background-color: #e6eef8;
            }
            """
        )

        add_row_button = make_toolbar_action_button("act_add", "Add prediction", self.add_prediction_row)
        remove_row_button = make_toolbar_action_button(
            "act_remove",
            "Remove prediction",
            self.remove_selected_prediction_rows,
        )
        load_csv_button = make_toolbar_action_button("act_table_load", "Load CSV", self.load_prediction_csv_dialog)
        save_csv_button = make_toolbar_action_button("act_table_save", "Save CSV", self.save_prediction_csv_dialog)
        generate_button = make_toolbar_action_button(
            "act_readouts",
            "Generate stage readouts",
            self.generate_prediction_stage_readouts,
        )
        generate_paths_button = make_toolbar_action_button(
            "act_paths",
            "Generate paths",
            self.generate_prediction_diffraction_paths,
        )
        estimate_time_button = make_toolbar_action_button(
            "act_time",
            "Estimate time",
            self.generate_prediction_estimated_times,
        )
        create_scan_file_button = make_toolbar_action_button(
            "act_scan",
            "Create scan file",
            self.create_prediction_scan_file,
        )

        layout.addWidget(add_row_button)
        layout.addWidget(remove_row_button)
        layout.addWidget(load_csv_button)
        layout.addWidget(save_csv_button)
        layout.addWidget(generate_button)
        layout.addWidget(generate_paths_button)
        layout.addWidget(estimate_time_button)
        layout.addWidget(create_scan_file_button)
        layout.addStretch(1)
        return toolbar

    def _build_prediction_section(self) -> QWidget:
        section = QWidget()
        section.setMinimumHeight(0)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._build_section_title("Predicted stage readouts from model coordinates"))
        layout.addWidget(self._build_prediction_toolbar())
        self.prediction_table = self._build_prediction_table()
        layout.addWidget(self.prediction_table)
        return section

    def _build_measurement_table(self) -> QTableWidget:
        table = SpreadsheetTableWidget(0, len(TABLE_HEADERS))
        table.setHorizontalHeaderLabels(TABLE_HEADERS)
        table.setSelectionBehavior(QAbstractItemView.SelectItems)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setMinimumHeight(80)
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        table.verticalHeader().setDefaultSectionSize(30)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(False)
        table.itemSelectionChanged.connect(self.on_measurement_selection_changed)
        table.itemChanged.connect(self.on_measurement_item_changed)
        return table

    def _build_prediction_table(self) -> QTableWidget:
        table = SpreadsheetTableWidget(0, len(PREDICTION_TABLE_HEADERS))
        table.setHorizontalHeaderLabels(PREDICTION_TABLE_HEADERS)
        table.setSelectionBehavior(QAbstractItemView.SelectItems)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setMinimumHeight(80)
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        table.verticalHeader().setDefaultSectionSize(30)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(False)
        table.read_only_columns = {7, 8, 9, 10}
        table.after_paste = lambda: self.update_scene(reset_camera=False)
        table.itemSelectionChanged.connect(self.on_prediction_selection_changed)
        table.itemChanged.connect(self.on_prediction_item_changed)
        return table

    def _build_report_box(self) -> QPlainTextEdit:
        report = QPlainTextEdit()
        report.setReadOnly(True)
        report.setMinimumHeight(36)
        return report

    def _separator(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.HLine)
        frame.setFrameShadow(QFrame.Sunken)
        return frame

    def _load_example_measurements_if_present(self) -> None:
        example_path = Path("example_measurements.csv")
        if example_path.exists():
            self.load_csv(example_path)

    def on_view_parameter_changed(self) -> None:
        self.invalidate_detector_map(update_scene=False)
        self.update_scene(reset_camera=False)

    def on_detector_map_parameter_changed(self) -> None:
        self.invalidate_detector_map(update_scene=True)

    def on_overlay_visibility_changed(self, _checked: bool) -> None:
        self.update_scene(reset_camera=False)

    def on_ui_font_size_changed(self, value: float) -> None:
        app = QApplication.instance()
        if app is None:
            return
        apply_application_ui_font(app, float(value))
        self.update_stress_table_metrics()
        if hasattr(self, "instrument_setup_dialog") and self.instrument_setup_dialog is not None:
            self.instrument_setup_dialog.adjustSize()
        self.updateGeometry()
        self.update_scene(reset_camera=False)

    def on_measurement_selection_changed(self) -> None:
        if hasattr(self, "prediction_table"):
            was_blocked = self.prediction_table.blockSignals(True)
            self.prediction_table.clearSelection()
            self.prediction_table.blockSignals(was_blocked)
        if (
            hasattr(self, "auto_move_to_pivot_checkbox")
            and self.auto_move_to_pivot_checkbox.isChecked()
            and (self.fit_transform is not None or self.manual_placement_enabled_checkbox.isChecked())
        ):
            self.move_selected_point_to_pivot(show_error=False)
        self.update_scene(reset_camera=False)

    def on_prediction_selection_changed(self) -> None:
        if hasattr(self, "measurement_table"):
            was_blocked = self.measurement_table.blockSignals(True)
            self.measurement_table.clearSelection()
            self.measurement_table.blockSignals(was_blocked)
        if hasattr(self, "auto_move_to_pivot_checkbox") and self.auto_move_to_pivot_checkbox.isChecked():
            self.move_selected_prediction_point_to_pivot(show_error=False)
        self.update_scene(reset_camera=False)

    def current_pivot_world(self) -> Tuple[float, float, float]:
        return (self.pivot_x.value(), self.pivot_y.value(), self.pivot_z.value())

    def current_stage_pose(self) -> Tuple[np.ndarray, float]:
        stage_readout_local = np.array([self.pose_x.value(), self.pose_y.value(), self.pose_z.value()], dtype=float)
        return stage_readout_local, self.pose_omega.value()

    def current_diffraction_detector_geometry(self, detector_center_world: np.ndarray) -> dict:
        detector_center = np.array(detector_center_world, dtype=float)
        pivot = np.array(self.current_pivot_world(), dtype=float)
        center_to_pivot = pivot - detector_center
        distance = float(np.linalg.norm(center_to_pivot))
        if distance < 1e-9:
            normal = np.array([0.0, -1.0, 0.0], dtype=float)
            distance = 1.0
        else:
            normal = center_to_pivot / distance
        right, up = orthogonal_plane_basis(normal, np.array([0.0, 0.0, 1.0], dtype=float))
        half_width = distance * np.tan(np.radians(ANGLED_DETECTOR_HORIZONTAL_HALF_ANGLE_DEG))
        half_height = distance * np.tan(np.radians(ANGLED_DETECTOR_VERTICAL_HALF_ANGLE_DEG))
        width = 2.0 * half_width
        height = 2.0 * half_height
        corners = np.array(
            [
                detector_center - right * half_width - up * half_height,
                detector_center + right * half_width - up * half_height,
                detector_center + right * half_width + up * half_height,
                detector_center - right * half_width + up * half_height,
            ],
            dtype=float,
        )
        return {
            "center": detector_center,
            "normal": normal,
            "right": right,
            "up": up,
            "distance": distance,
            "width": width,
            "height": height,
            "half_width": half_width,
            "half_height": half_height,
            "corners": corners,
        }

    def current_diffraction_bank_1_geometry(self) -> dict:
        return self.current_diffraction_detector_geometry(DIFFRACTION_BANK_1_CENTER_WORLD)

    def current_diffraction_bank_2_geometry(self) -> dict:
        return self.current_diffraction_detector_geometry(DIFFRACTION_BANK_2_CENTER_WORLD)

    def manual_model_to_stage_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        rotation = np.array(self.manual_rotation_matrix, dtype=float)
        translation = np.array([self.manual_tx.value(), self.manual_ty.value(), self.manual_tz.value()], dtype=float)
        return rotation, translation

    def sync_manual_rotation_spin_boxes_from_matrix(self) -> None:
        if not all(hasattr(self, attr) for attr in ("manual_rx", "manual_ry", "manual_rz")):
            return
        rotation_boxes = (
            (self.manual_rx, self.manual_rotation_display_values[0]),
            (self.manual_ry, self.manual_rotation_display_values[1]),
            (self.manual_rz, self.manual_rotation_display_values[2]),
        )
        previous_states = [(box, box.blockSignals(True)) for box, _value in rotation_boxes]
        try:
            for box, value in rotation_boxes:
                box.setValue(float(value))
        finally:
            for box, previous_state in previous_states:
                box.blockSignals(previous_state)

    def current_model_to_stage_transform(self) -> Tuple[np.ndarray, np.ndarray, str]:
        if self.manual_placement_enabled_checkbox.isChecked():
            rotation, translation = self.manual_model_to_stage_transform()
            return rotation, translation, "manual"
        if self.fit_transform is not None:
            return np.array(self.fit_transform.rotation, dtype=float), np.array(self.fit_transform.translation, dtype=float), "fit"
        return np.eye(3, dtype=float), np.zeros(3, dtype=float), "as-is"

    def stage_readout_for_model_point(self, model_point: np.ndarray) -> np.ndarray:
        rotation, translation, _source = self.current_model_to_stage_transform()
        point_stage = (rotation @ np.asarray(model_point, dtype=float)) + translation
        return -point_stage

    def invalidate_detector_map(self, update_scene: bool = False) -> None:
        self.detector_map_state = None
        self.diffraction_bank_1_map_state = None
        self.diffraction_bank_2_map_state = None
        self.refresh_report_box()
        if update_scene:
            self.update_scene(reset_camera=False)

    def update_placement_status(self) -> None:
        self.refresh_report_box()

    def update_placement_summary_fields(self) -> None:
        self.refresh_report_box()

    def placement_summary_lines(self) -> List[str]:
        lines: List[str] = []
        if self.manual_placement_enabled_checkbox.isChecked():
            rotation, translation, _source = self.current_model_to_stage_transform()
            lines.append("Placement: Manual placement active")
            lines.append(f"Translation: {pretty_vector(translation)}")
            lines.append(f"Euler ZYX: {pretty_vector(rotation_matrix_to_euler_zyx_deg(rotation))}")
        elif self.fit_transform is not None:
            lines.append(f"Placement: Fit placement active with {len(self.residual_rows)} point pairs")
            lines.append(f"Translation: {pretty_vector(self.fit_transform.translation)}")
            lines.append(f"Euler ZYX: {pretty_vector(rotation_matrix_to_euler_zyx_deg(self.fit_transform.rotation))}")
            lines.append(f"RMS error: {format_decimal(self.fit_transform.rms_error)}")
            lines.append(f"Max error: {format_decimal(self.fit_transform.max_error)}")
        else:
            lines.append("Placement: No placement transform; model shown as imported")

        if self.detector_map_state is not None:
            lines.append(f"Imaging path: {format_decimal(self.detector_map_state['average_length'])}")
            lines.append(
                "Imaging map: "
                f"{self.detector_map_state['resolution_y']}x{self.detector_map_state['resolution_z']} "
                f"@ {format_decimal(self.detector_map_state['pixel_size_y'])}/{format_decimal(self.detector_map_state['pixel_size_z'])}, "
                f"max {format_decimal(self.detector_map_state['max_length'])}"
            )

        diffraction_path_parts: List[str] = []
        diffraction_map_parts: List[str] = []
        if self.diffraction_bank_1_map_state is not None:
            diffraction_path_parts.append(f"Bank 1 {format_decimal(self.diffraction_bank_1_map_state['average_length'])}")
            diffraction_map_parts.append(
                f"Bank 1 {len(self.diffraction_bank_1_map_state['horizontal_angles_deg'])}x"
                f"{len(self.diffraction_bank_1_map_state['vertical_angles_deg'])}, "
                f"max {format_decimal(self.diffraction_bank_1_map_state['max_length'])}"
            )
        if self.diffraction_bank_2_map_state is not None:
            diffraction_path_parts.append(f"Bank 2 {format_decimal(self.diffraction_bank_2_map_state['average_length'])}")
            diffraction_map_parts.append(
                f"Bank 2 {len(self.diffraction_bank_2_map_state['horizontal_angles_deg'])}x"
                f"{len(self.diffraction_bank_2_map_state['vertical_angles_deg'])}, "
                f"max {format_decimal(self.diffraction_bank_2_map_state['max_length'])}"
            )
        if diffraction_path_parts:
            lines.append(f"Diffraction path: {'; '.join(diffraction_path_parts)}")
        if diffraction_map_parts:
            lines.append(f"Diffraction map: {'; '.join(diffraction_map_parts)}")
        return lines

    def build_current_model_world_mesh(self) -> Optional[pv.PolyData]:
        stage_readout_local, omega_deg = self.current_stage_pose()
        return self.build_model_world_mesh_for_pose(stage_readout_local, omega_deg)

    def build_model_world_mesh_for_pose(
        self,
        stage_readout_local: np.ndarray,
        omega_deg: float,
    ) -> Optional[pv.PolyData]:
        if self.model_mesh is None:
            return None
        model_world = self.model_mesh.copy(deep=True)
        pivot = np.array(self.current_pivot_world(), dtype=float)
        rotation, translation, _source = self.current_model_to_stage_transform()
        model_stage = (rotation @ self.model_mesh.points.T).T + translation
        model_world.points = stage_local_to_world(model_stage, stage_readout_local, omega_deg, pivot)
        return model_world

    def stage_box_local(self) -> pv.PolyData:
        center = np.array(
            [self.stage_offset_x.value(), self.stage_offset_y.value(), self.stage_offset_z.value()],
            dtype=float,
        )
        size = np.array([self.stage_size_x.value(), self.stage_size_y.value(), self.stage_size_z.value()], dtype=float)
        bounds = (
            center[0] - size[0] / 2.0,
            center[0] + size[0] / 2.0,
            center[1] - size[1] / 2.0,
            center[1] + size[1] / 2.0,
            center[2] - size[2] / 2.0,
            center[2] + size[2] / 2.0,
        )
        return pv.Box(bounds=bounds)

    def measurements_from_table(self) -> List[Measurement]:
        measurements: List[Measurement] = []
        for row in range(self.measurement_table.rowCount()):
            values = []
            for column in range(self.measurement_table.columnCount()):
                item = self.measurement_table.item(row, column)
                values.append("" if item is None else item.text().strip())

            if not any(values):
                continue
            if any(value == "" for value in values):
                raise ValueError(f"Row {row + 1} is incomplete.")

            try:
                measurements.append(
                    Measurement(
                        label=values[0],
                        model_point=(float(values[1]), float(values[2]), float(values[3])),
                        stage_readout=(float(values[4]), float(values[5]), float(values[6])),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Row {row + 1} contains an invalid number.") from exc

        if len(measurements) < 3:
            raise ValueError("At least three measurement rows are required.")
        return measurements

    def prediction_rows_from_table(self) -> List[Tuple[str, np.ndarray]]:
        prediction_rows: List[Tuple[str, np.ndarray]] = []
        for row in range(self.prediction_table.rowCount()):
            label_item = self.prediction_table.item(row, 0)
            label = "" if label_item is None else label_item.text().strip()
            label = label or f"G{row + 1}"
            model_values = []
            incomplete = False
            for column in range(1, 4):
                item = self.prediction_table.item(row, column)
                text = "" if item is None else item.text().strip()
                if text == "":
                    incomplete = True
                    break
                try:
                    model_values.append(float(text))
                except ValueError:
                    incomplete = True
                    break
            if incomplete:
                continue
            prediction_rows.append((label, np.array(model_values, dtype=float)))
        return prediction_rows

    def add_measurement_row(self, measurement: Optional[Measurement] = None) -> None:
        row = self.measurement_table.rowCount()
        self.measurement_table.insertRow(row)
        values = (
            measurement.label if measurement else "",
            measurement.model_point[0] if measurement else "",
            measurement.model_point[1] if measurement else "",
            measurement.model_point[2] if measurement else "",
            measurement.stage_readout[0] if measurement else "",
            measurement.stage_readout[1] if measurement else "",
            measurement.stage_readout[2] if measurement else "",
        )
        for column, value in enumerate(values):
            if column == 0 or value == "":
                display_value = value
            else:
                display_value = format_decimal(float(value))
            item = QTableWidgetItem(str(display_value))
            self.measurement_table.setItem(row, column, item)

    def add_prediction_row(
        self,
        label: Optional[str] = None,
        model_point: Optional[Sequence[float]] = None,
        stage_readout: Optional[Sequence[float]] = None,
        path_values: Optional[Sequence[float]] = None,
        uamp_values: Optional[Sequence[float]] = None,
    ) -> None:
        row = self.prediction_table.rowCount()
        self.prediction_table.insertRow(row)
        default_label = label if label is not None else f"G{row + 1}"
        model_point = model_point if model_point is not None else (0.0, 0.0, 0.0)
        stage_readout = stage_readout if stage_readout is not None else ("", "", "")
        path_values = path_values if path_values is not None else ("", "")
        uamp_values = uamp_values if uamp_values is not None else ("", "")
        values = (
            default_label,
            model_point[0],
            model_point[1],
            model_point[2],
            stage_readout[0],
            stage_readout[1],
            stage_readout[2],
            path_values[0],
            path_values[1],
            uamp_values[0],
            uamp_values[1],
        )
        for column, value in enumerate(values):
            if column == 0:
                display_value = value
            elif value == "":
                display_value = ""
            else:
                display_value = format_decimal(float(value))
            item = QTableWidgetItem(str(display_value))
            if column >= 7:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.prediction_table.setItem(row, column, item)

    def on_measurement_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            return
        text = item.text().strip()
        if text == "":
            return
        try:
            normalized = format_decimal(float(text))
        except ValueError:
            return
        if text == normalized:
            return
        was_blocked = self.measurement_table.blockSignals(True)
        item.setText(normalized)
        self.measurement_table.blockSignals(was_blocked)

    def on_prediction_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self.update_scene(reset_camera=False)
            return
        text = item.text().strip()
        if text == "":
            self.update_scene(reset_camera=False)
            return
        try:
            normalized = format_decimal(float(text))
        except ValueError:
            self.update_scene(reset_camera=False)
            return
        if text == normalized:
            self.update_scene(reset_camera=False)
            return
        was_blocked = self.prediction_table.blockSignals(True)
        item.setText(normalized)
        self.prediction_table.blockSignals(was_blocked)
        self.update_scene(reset_camera=False)

    def add_stress_row(self) -> None:
        row = self.stress_table.rowCount()
        self.stress_table.insertRow(row)
        for column in range(self.stress_table.columnCount()):
            item = QTableWidgetItem("")
            if column in self.stress_table.read_only_columns:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.stress_table.setItem(row, column, item)

    def remove_selected_stress_rows(self) -> None:
        rows = sorted({index.row() for index in self.stress_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            self.stress_table.removeRow(row)
        self.stress_status_label.setText(f"Removed {len(rows)} residual-stress row(s).")

    def clear_stress_outputs(self) -> None:
        was_blocked = self.stress_table.blockSignals(True)
        try:
            for row in range(self.stress_table.rowCount()):
                self.clear_stress_outputs_for_row(row)
        finally:
            self.stress_table.blockSignals(was_blocked)
        self.stress_status_label.setText("Cleared calculated strain/stress outputs.")

    def clear_stress_outputs_for_row(self, row: int) -> None:
        for column in range(14, len(STRESS_TABLE_HEADERS)):
            item = self.stress_table.item(row, column)
            if item is None:
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.stress_table.setItem(row, column, item)
            else:
                item.setText("")

    def on_stress_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() in {0, 1} or item.column() in self.stress_table.read_only_columns:
            return
        text = item.text().strip()
        if text == "":
            return
        try:
            normalized = format_fixed_decimal(float(text), decimals=6)
        except ValueError:
            return
        if text == normalized:
            return
        was_blocked = self.stress_table.blockSignals(True)
        try:
            item.setText(normalized)
        finally:
            self.stress_table.blockSignals(was_blocked)

    def normalize_stress_input_cells(self) -> None:
        was_blocked = self.stress_table.blockSignals(True)
        try:
            for row in range(self.stress_table.rowCount()):
                for column in range(self.stress_table.columnCount()):
                    if column in {0, 1} or column in self.stress_table.read_only_columns:
                        continue
                    item = self.stress_table.item(row, column)
                    if item is None:
                        continue
                    text = item.text().strip()
                    if text == "":
                        continue
                    try:
                        item.setText(format_fixed_decimal(float(text), decimals=6))
                    except ValueError:
                        continue
        finally:
            self.stress_table.blockSignals(was_blocked)

    def _stress_cell_text(self, row: int, column: int) -> str:
        item = self.stress_table.item(row, column)
        return "" if item is None else item.text().strip()

    def _stress_optional_float(self, row: int, column: int) -> Optional[float]:
        text = self._stress_cell_text(row, column)
        if text == "":
            return None
        return float(text)

    def _set_stress_output_value(self, row: int, column: int, value: Optional[float], decimals: int) -> None:
        item = self.stress_table.item(row, column)
        if item is None:
            item = QTableWidgetItem("")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.stress_table.setItem(row, column, item)
        item.setText("" if value is None else format_trimmed_decimal(value, decimals=decimals))

    def calculate_residual_stress(self) -> None:
        youngs_modulus_mpa = float(self.stress_youngs_modulus_mpa.value())
        poissons_ratio = float(self.stress_poissons_ratio.value())
        calculated_strain_rows = 0
        calculated_stress_rows = 0
        incomplete_rows = 0

        was_blocked = self.stress_table.blockSignals(True)
        try:
            for row in range(self.stress_table.rowCount()):
                self.clear_stress_outputs_for_row(row)
                input_texts = [self._stress_cell_text(row, column) for column in range(14)]
                if not any(input_texts):
                    continue

                strain_values: List[Optional[float]] = []
                strain_uncertainties: List[Optional[float]] = []
                row_has_strain = False

                for direction in range(3):
                    lattice_column = 2 + direction * 4
                    lattice_uncert_column = lattice_column + 1
                    d0_column = lattice_column + 2
                    d0_uncert_column = lattice_column + 3

                    lattice_value = self._stress_optional_float(row, lattice_column)
                    lattice_uncertainty = self._stress_optional_float(row, lattice_uncert_column)
                    d0_value = self._stress_optional_float(row, d0_column)
                    d0_uncertainty = self._stress_optional_float(row, d0_uncert_column)

                    direction_texts = [self._stress_cell_text(row, lattice_column + offset) for offset in range(4)]
                    if not any(direction_texts):
                        strain_values.append(None)
                        strain_uncertainties.append(None)
                        continue
                    if lattice_value is None or d0_value is None:
                        strain_values.append(None)
                        strain_uncertainties.append(None)
                        continue

                    strain_value = compute_lattice_strain(lattice_value, d0_value)
                    strain_uncertainty = compute_lattice_strain_uncertainty(
                        lattice_value,
                        lattice_uncertainty,
                        d0_value,
                        d0_uncertainty,
                    )
                    displayed_strain_value = strain_value * MICROSTRAIN_SCALE
                    displayed_strain_uncertainty = None
                    if strain_uncertainty is not None:
                        displayed_strain_uncertainty = strain_uncertainty * MICROSTRAIN_SCALE
                    self._set_stress_output_value(row, 14 + direction * 2, displayed_strain_value, decimals=0)
                    self._set_stress_output_value(row, 15 + direction * 2, displayed_strain_uncertainty, decimals=0)
                    strain_values.append(strain_value)
                    strain_uncertainties.append(strain_uncertainty)
                    row_has_strain = True

                if row_has_strain:
                    calculated_strain_rows += 1

                if any(value is None for value in strain_values):
                    incomplete_rows += 1
                    continue

                stress_values = compute_three_dimensional_stress_mpa(
                    [float(value) for value in strain_values],
                    youngs_modulus_mpa,
                    poissons_ratio,
                )
                stress_uncertainties = compute_three_dimensional_stress_uncertainty_mpa(
                    strain_uncertainties,
                    youngs_modulus_mpa,
                    poissons_ratio,
                )
                for direction in range(3):
                    self._set_stress_output_value(row, 20 + direction * 2, float(stress_values[direction]), decimals=0)
                    stress_uncertainty = None
                    if stress_uncertainties is not None:
                        stress_uncertainty = float(stress_uncertainties[direction])
                    self._set_stress_output_value(row, 21 + direction * 2, stress_uncertainty, decimals=0)
                calculated_stress_rows += 1
        except Exception as exc:
            self.show_error("Residual stress calculation failed", str(exc))
            return
        finally:
            self.stress_table.blockSignals(was_blocked)

        self.stress_status_label.setText(
            f"Calculated strain for {calculated_strain_rows} row(s), 3D stress for {calculated_stress_rows} row(s)."
            + (f" {incomplete_rows} row(s) were left without full stress because one or more directions were missing." if incomplete_rows else "")
        )

    def save_stress_csv_dialog(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save residual stress CSV",
            str(Path.cwd() / "residual_stress.csv"),
            "CSV files (*.csv)",
        )
        if not path_str:
            return
        try:
            with Path(path_str).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(STRESS_TABLE_HEADERS)
                for row in range(self.stress_table.rowCount()):
                    values = []
                    has_value = False
                    for column in range(self.stress_table.columnCount()):
                        item = self.stress_table.item(row, column)
                        text = "" if item is None else item.text().strip()
                        values.append(text)
                        if text != "":
                            has_value = True
                    if not has_value:
                        continue
                    writer.writerow(values)
            self.statusBar().showMessage(f"Saved residual stress table to {path_str}", 5000)
            self.stress_status_label.setText(f"Saved residual stress table to {path_str}")
        except Exception as exc:
            self.show_error("Failed to save residual stress CSV", str(exc))

    def show_stress_equations_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Residual Stress Equations")
        dialog.setModal(False)
        dialog.resize(980, 760)

        layout = QVBoxLayout(dialog)
        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        layout.addWidget(scroll)

        content = QWidget()
        content.setStyleSheet("background-color: white;")
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(18, 18, 18, 18)
        content_layout.setSpacing(14)

        intro_label = QLabel(
            "The residual-stress tab uses the following equations.",
            content,
        )
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet(
            "color: black; font-size: 16px; font-family: 'Times New Roman', Georgia, serif;"
        )
        content_layout.addWidget(intro_label)

        equation_cards = [
            (
                "Lattice strain",
                "<span style='font-style: italic;'>"
                "&epsilon;<sub>i</sub> = "
                "<span style='font-size: 120%;'>(</span>a<sub>i</sub> - a0<sub>i</sub><span style='font-size: 120%;'>)</span>"
                " / a0<sub>i</sub>"
                "</span>",
            ),
            (
                "Strain uncertainty",
                "<span style='font-style: italic;'>"
                "u(&epsilon;<sub>i</sub>) = "
                "&radic;<span style='font-size: 115%;'>(</span>"
                "<span>(u(a<sub>i</sub>) / a0<sub>i</sub>)<sup>2</sup></span>"
                " + "
                "<span>(a<sub>i</sub>u(a0<sub>i</sub>) / a0<sub>i</sub><sup>2</sup>)<sup>2</sup></span>"
                "<span style='font-size: 115%;'>)</span>"
                "</span>",
            ),
            (
                "Microstrain display",
                "<span style='font-style: italic;'>"
                "&epsilon;<sub>&micro;&epsilon;,i</sub> = &epsilon;<sub>i</sub> &times; 10<sup>6</sup>"
                "<br/>"
                "u(&epsilon;<sub>&micro;&epsilon;,i</sub>) = u(&epsilon;<sub>i</sub>) &times; 10<sup>6</sup>"
                "</span>",
            ),
            (
                "3D isotropic stress scale",
                "<span style='font-style: italic;'>"
                "C = E / <span style='font-size: 120%;'>(</span>(1 + &nu;)(1 - 2&nu;)<span style='font-size: 120%;'>)</span>"
                "</span>",
            ),
            (
                "Stress 1 (MPa)",
                "<span style='font-style: italic;'>"
                "&sigma;<sub>1</sub> = C"
                "<span style='font-size: 120%;'>(</span>"
                "(1 - &nu;)&epsilon;<sub>1</sub> + &nu;&epsilon;<sub>2</sub> + &nu;&epsilon;<sub>3</sub>"
                "<span style='font-size: 120%;'>)</span>"
                "</span>",
            ),
            (
                "Stress 2 (MPa)",
                "<span style='font-style: italic;'>"
                "&sigma;<sub>2</sub> = C"
                "<span style='font-size: 120%;'>(</span>"
                "&nu;&epsilon;<sub>1</sub> + (1 - &nu;)&epsilon;<sub>2</sub> + &nu;&epsilon;<sub>3</sub>"
                "<span style='font-size: 120%;'>)</span>"
                "</span>",
            ),
            (
                "Stress 3 (MPa)",
                "<span style='font-style: italic;'>"
                "&sigma;<sub>3</sub> = C"
                "<span style='font-size: 120%;'>(</span>"
                "&nu;&epsilon;<sub>1</sub> + &nu;&epsilon;<sub>2</sub> + (1 - &nu;)&epsilon;<sub>3</sub>"
                "<span style='font-size: 120%;'>)</span>"
                "</span>",
            ),
            (
                "Stress uncertainty",
                "<span style='font-style: italic;'>"
                "u(&sigma;<sub>1</sub>) = |C| &radic;<span style='font-size: 115%;'>(</span>"
                "((1 - &nu;)u(&epsilon;<sub>1</sub>))<sup>2</sup> + "
                "(&nu;u(&epsilon;<sub>2</sub>))<sup>2</sup> + "
                "(&nu;u(&epsilon;<sub>3</sub>))<sup>2</sup>"
                "<span style='font-size: 115%;'>)</span>"
                "<br/>"
                "u(&sigma;<sub>2</sub>) = |C| &radic;<span style='font-size: 115%;'>(</span>"
                "(&nu;u(&epsilon;<sub>1</sub>))<sup>2</sup> + "
                "((1 - &nu;)u(&epsilon;<sub>2</sub>))<sup>2</sup> + "
                "(&nu;u(&epsilon;<sub>3</sub>))<sup>2</sup>"
                "<span style='font-size: 115%;'>)</span>"
                "<br/>"
                "u(&sigma;<sub>3</sub>) = |C| &radic;<span style='font-size: 115%;'>(</span>"
                "(&nu;u(&epsilon;<sub>1</sub>))<sup>2</sup> + "
                "(&nu;u(&epsilon;<sub>2</sub>))<sup>2</sup> + "
                "((1 - &nu;)u(&epsilon;<sub>3</sub>))<sup>2</sup>"
                "<span style='font-size: 115%;'>)</span>"
                "</span>",
            ),
        ]

        for title, html in equation_cards:
            card = QFrame(content)
            card.setStyleSheet(
                "QFrame { background-color: white; border: 1px solid #d9d9d9; border-radius: 6px; }"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 14, 16, 14)
            card_layout.setSpacing(8)

            title_label = QLabel(title, card)
            title_label.setStyleSheet(
                "color: #444444; font-size: 15px; font-weight: 600; font-family: 'Segoe UI', sans-serif;"
            )
            card_layout.addWidget(title_label)

            equation_label = QLabel(card)
            equation_label.setTextFormat(Qt.RichText)
            equation_label.setWordWrap(True)
            equation_label.setText(html)
            equation_label.setStyleSheet(
                "color: black; font-size: 42px; font-family: 'Times New Roman', Georgia, serif;"
            )
            card_layout.addWidget(equation_label)
            content_layout.addWidget(card)

        notes_label = QLabel(
            "E is Young's modulus in MPa. ν is Poisson's ratio. Stress uncertainties are only computed when all three strain uncertainties are available.",
            content,
        )
        notes_label.setWordWrap(True)
        notes_label.setStyleSheet(
            "color: #444444; font-size: 14px; font-family: 'Segoe UI', sans-serif;"
        )
        content_layout.addWidget(notes_label)
        content_layout.addStretch(1)

        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button, alignment=Qt.AlignRight)

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def set_project_path_label(self, path: Optional[Path]) -> None:
        self.project_path = path
        self.refresh_report_box()

    def file_status_header_text(self) -> str:
        project_text = "No project saved or loaded" if self.project_path is None else f"Project: {self.project_path}"
        mesh_text = "No mesh loaded" if self.mesh_path is None else str(self.mesh_path)
        return "\n".join((project_text, mesh_text, self.measurement_source_text))

    def refresh_report_box(self) -> None:
        if not hasattr(self, "report_box") or self.report_box is None:
            return
        summary = "\n".join(self.placement_summary_lines()).strip()
        body = self.report_body_text.strip()
        header = self.file_status_header_text().strip()
        sections = [section for section in (header, summary, body) if section]
        text = "\n\n".join(sections)
        self.report_box.setPlainText(text)

    def set_report_body(self, text: str) -> None:
        self.report_body_text = text
        self.refresh_report_box()

    def sync_point_picker_panel_mesh(self) -> None:
        if self.point_picker_panel is not None:
            self.point_picker_panel.set_mesh(self.model_mesh, self.mesh_path)

    def default_project_path(self) -> Path:
        if self.project_path is not None:
            return self.project_path
        if self.mesh_path is not None:
            return Path.cwd() / f"{self.mesh_path.stem}.simsetup"
        return Path.cwd() / "sample_setup.simsetup"

    def build_fit_state_payload(self) -> Optional[dict]:
        if self.fit_transform is None:
            return None
        residual_rows = []
        for measurement, stage_point, fit_error in self.residual_rows:
            residual_rows.append(
                {
                    "measurement": {
                        "label": measurement.label,
                        "model_point": list(measurement.model_point),
                        "stage_readout": list(measurement.stage_readout),
                    },
                    "stage_point": list(stage_point),
                    "fit_error": float(fit_error),
                }
            )
        return {
            "rotation": [list(row) for row in self.fit_transform.rotation],
            "translation": list(self.fit_transform.translation),
            "quaternion_wxyz": list(self.fit_transform.quaternion_wxyz),
            "rms_error": float(self.fit_transform.rms_error),
            "max_error": float(self.fit_transform.max_error),
            "residual_rows": residual_rows,
            "report_text": self.report_body_text,
        }

    def build_project_payload(self) -> dict:
        return {
            "version": PROJECT_ARCHIVE_VERSION,
            "mesh": (
                {
                    "embedded_name": PROJECT_EMBEDDED_MESH_NAME,
                    "original_name": self.mesh_path.name if self.mesh_path is not None else PROJECT_EMBEDDED_MESH_NAME,
                    "format": "stl",
                }
                if self.model_mesh is not None and self.model_mesh.n_points > 0
                else None
            ),
            "setup": {
                "pivot_x": self.pivot_x.value(),
                "pivot_y": self.pivot_y.value(),
                "pivot_z": self.pivot_z.value(),
                "theodolite_x": self.theodolite_x.value(),
                "theodolite_y": self.theodolite_y.value(),
                "theodolite_z": self.theodolite_z.value(),
                "slit_x": self.slit_x.value(),
                "slit_y": self.slit_y.value(),
                "slit_z": self.slit_z.value(),
                "slit_width": self.slit_width.value(),
                "slit_height": self.slit_height.value(),
                "beam_length": self.beam_length.value(),
                "detector_width_y": self.detector_width_y.value(),
                "detector_height_z": self.detector_height_z.value(),
                "detector_map_pixel_size_y": self.detector_map_pixel_size_y.value(),
                "detector_map_pixel_size_z": self.detector_map_pixel_size_z.value(),
                "stage_size_x": self.stage_size_x.value(),
                "stage_size_y": self.stage_size_y.value(),
                "stage_size_z": self.stage_size_z.value(),
                "stage_offset_x": self.stage_offset_x.value(),
                "stage_offset_y": self.stage_offset_y.value(),
                "stage_offset_z": self.stage_offset_z.value(),
                "collimator": self.collimator.currentText(),
                "count_time_material": self.count_time_material.currentText(),
            },
            "stage_pose": {
                "x": self.pose_x.value(),
                "y": self.pose_y.value(),
                "z": self.pose_z.value(),
                "omega": self.pose_omega.value(),
            },
            "manual_placement": {
                "enabled": self.manual_placement_enabled_checkbox.isChecked(),
                "tx": self.manual_tx.value(),
                "ty": self.manual_ty.value(),
                "tz": self.manual_tz.value(),
                "rx": float(self.manual_rotation_display_values[0]),
                "ry": float(self.manual_rotation_display_values[1]),
                "rz": float(self.manual_rotation_display_values[2]),
                "rotation_matrix": [list(row) for row in np.asarray(self.manual_rotation_matrix, dtype=float)],
            },
            "fit_transform": self.build_fit_state_payload(),
            "measurements": table_to_serializable_rows(self.measurement_table, TABLE_HEADERS),
            "predictions": table_to_serializable_rows(self.prediction_table, PREDICTION_TABLE_HEADERS),
            "view": {
                "camera_preset": self.camera_preset,
                "parallel_projection_enabled": bool(self.parallel_projection_enabled),
                "ui_font_size_pt": (
                    float(self.ui_font_size_spin.value()) if self.ui_font_size_spin is not None else DEFAULT_UI_FONT_POINT_SIZE
                ),
                "viewer_font_size_offset": int(self.viewer_font_size_offset),
                "active_tab": self.main_tabs.currentIndex() if self.main_tabs is not None else 0,
                "auto_move_to_pivot": self.auto_move_to_pivot_checkbox.isChecked(),
                "show_stage": self.show_stage_checkbox.isChecked(),
                "show_beam": self.show_beam_checkbox.isChecked(),
                "show_gauge_volume": self.show_gauge_volume_checkbox.isChecked(),
                "show_imaging_detector": self.show_imaging_detector_checkbox.isChecked(),
                "show_diffraction_detectors": self.show_diffraction_detectors_checkbox.isChecked(),
                "show_feature_points": self.show_feature_points_checkbox.isChecked(),
                "show_prediction_points": self.show_prediction_points_checkbox.isChecked(),
                "show_sample_triad": self.show_sample_triad_checkbox.isChecked(),
                "show_theodolite_sight_line": self.show_theodolite_sight_line_checkbox.isChecked(),
                "show_diffraction_vectors": self.show_diffraction_vectors_checkbox.isChecked(),
            },
        }

    def save_project_dialog(self) -> None:
        default_path = self.default_project_path()
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            str(default_path),
            PROJECT_FILE_FILTER,
        )
        if not path_str:
            return
        output_path = Path(path_str)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".simsetup")
        try:
            self.save_project(output_path)
        except Exception as exc:
            self.show_error("Failed to save project", str(exc))

    def save_project(self, path: Path) -> None:
        payload = self.build_project_payload()
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(PROJECT_MANIFEST_NAME, json.dumps(payload, indent=2))
            if payload["mesh"] is not None:
                archive.writestr(PROJECT_EMBEDDED_MESH_NAME, polydata_to_stl_bytes(self.model_mesh))
        self.set_project_path_label(path)
        self.statusBar().showMessage(f"Saved project to {path}", 5000)

    def load_project_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load project",
            str(self.default_project_path().parent),
            PROJECT_FILE_FILTER,
        )
        if not path_str:
            return
        try:
            self.load_project(Path(path_str))
        except Exception as exc:
            self.show_error("Failed to load project", str(exc))

    def load_project(self, path: Path) -> None:
        with zipfile.ZipFile(path, "r") as archive:
            try:
                payload = json.loads(archive.read(PROJECT_MANIFEST_NAME).decode("utf-8"))
            except KeyError as exc:
                raise ValueError("The selected file does not contain a SimSetup project manifest.") from exc

            version = int(payload.get("version", 0))
            if version != PROJECT_ARCHIVE_VERSION:
                raise ValueError(
                    f"Unsupported project version {version}. Expected version {PROJECT_ARCHIVE_VERSION}."
                )

            mesh_payload = payload.get("mesh")
            loaded_mesh = None
            loaded_mesh_path = None
            if mesh_payload is not None:
                embedded_name = mesh_payload.get("embedded_name", PROJECT_EMBEDDED_MESH_NAME)
                try:
                    mesh_bytes = archive.read(embedded_name)
                except KeyError as exc:
                    raise ValueError("The project archive is missing its embedded mesh file.") from exc
                mesh_name = mesh_payload.get("original_name", PROJECT_EMBEDDED_MESH_NAME)
                mesh_format = str(mesh_payload.get("format", "stl"))
                loaded_mesh = load_mesh_bytes_as_polydata(mesh_bytes, mesh_name, file_type=mesh_format)
                loaded_mesh_path = Path(mesh_name)

        setup = payload.get("setup", {})
        stage_pose = payload.get("stage_pose", {})
        manual_placement = payload.get("manual_placement", {})
        fit_payload = payload.get("fit_transform")
        view = payload.get("view", {})

        spin_box_values = (
            (self.pivot_x, setup.get("pivot_x")),
            (self.pivot_y, setup.get("pivot_y")),
            (self.pivot_z, setup.get("pivot_z")),
            (self.theodolite_x, setup.get("theodolite_x")),
            (self.theodolite_y, setup.get("theodolite_y")),
            (self.theodolite_z, setup.get("theodolite_z")),
            (self.slit_x, setup.get("slit_x")),
            (self.slit_y, setup.get("slit_y")),
            (self.slit_z, setup.get("slit_z")),
            (self.slit_width, setup.get("slit_width")),
            (self.slit_height, setup.get("slit_height")),
            (self.beam_length, setup.get("beam_length")),
            (self.detector_width_y, setup.get("detector_width_y")),
            (self.detector_height_z, setup.get("detector_height_z")),
            (self.detector_map_pixel_size_y, setup.get("detector_map_pixel_size_y")),
            (self.detector_map_pixel_size_z, setup.get("detector_map_pixel_size_z")),
            (self.stage_size_x, setup.get("stage_size_x")),
            (self.stage_size_y, setup.get("stage_size_y")),
            (self.stage_size_z, setup.get("stage_size_z")),
            (self.stage_offset_x, setup.get("stage_offset_x")),
            (self.stage_offset_y, setup.get("stage_offset_y")),
            (self.stage_offset_z, setup.get("stage_offset_z")),
            (self.pose_x, stage_pose.get("x")),
            (self.pose_y, stage_pose.get("y")),
            (self.pose_z, stage_pose.get("z")),
            (self.pose_omega, stage_pose.get("omega")),
            (self.manual_tx, manual_placement.get("tx")),
            (self.manual_ty, manual_placement.get("ty")),
            (self.manual_tz, manual_placement.get("tz")),
        )
        spin_signal_states = [(box, box.blockSignals(True)) for box, _value in spin_box_values]
        try:
            for box, value in spin_box_values:
                if value is not None:
                    box.setValue(float(value))
        finally:
            for box, previous_state in spin_signal_states:
                box.blockSignals(previous_state)

        manual_rotation_payload = manual_placement.get("rotation_matrix")
        if manual_rotation_payload is not None:
            self.manual_rotation_matrix = orthonormalize_rotation_matrix(
                np.array(manual_rotation_payload, dtype=float)
            )
        else:
            self.manual_rotation_matrix = orthonormalize_rotation_matrix(
                rotation_matrix_from_euler_xyz_deg(
                    float(manual_placement.get("rx", 0.0)),
                    float(manual_placement.get("ry", 0.0)),
                    float(manual_placement.get("rz", 0.0)),
                )
            )
        if any(key in manual_placement for key in ("rx", "ry", "rz")):
            self.manual_rotation_display_values = np.array(
                [
                    float(manual_placement.get("rx", 0.0)),
                    float(manual_placement.get("ry", 0.0)),
                    float(manual_placement.get("rz", 0.0)),
                ],
                dtype=float,
            )
        else:
            self.manual_rotation_display_values = np.array(
                rotation_matrix_to_euler_xyz_deg(np.array(self.manual_rotation_matrix, dtype=float)),
                dtype=float,
            )
        self.sync_manual_rotation_spin_boxes_from_matrix()

        combo_states = [
            (self.collimator, self.collimator.blockSignals(True)),
            (self.count_time_material, self.count_time_material.blockSignals(True)),
        ]
        try:
            collimator_text = setup.get("collimator")
            if collimator_text is not None and self.collimator.findText(str(collimator_text)) >= 0:
                self.collimator.setCurrentText(str(collimator_text))
            material_text = setup.get("count_time_material")
            if material_text is not None and self.count_time_material.findText(str(material_text)) >= 0:
                self.count_time_material.setCurrentText(str(material_text))
        finally:
            for combo, previous_state in combo_states:
                combo.blockSignals(previous_state)

        checkbox_values = (
            (self.manual_placement_enabled_checkbox, manual_placement.get("enabled")),
            (self.auto_move_to_pivot_checkbox, view.get("auto_move_to_pivot")),
            (self.parallel_projection_checkbox, view.get("parallel_projection_enabled")),
            (self.show_stage_checkbox, view.get("show_stage")),
            (self.show_beam_checkbox, view.get("show_beam")),
            (self.show_gauge_volume_checkbox, view.get("show_gauge_volume")),
            (self.show_imaging_detector_checkbox, view.get("show_imaging_detector")),
            (self.show_diffraction_detectors_checkbox, view.get("show_diffraction_detectors")),
            (self.show_feature_points_checkbox, view.get("show_feature_points")),
            (self.show_prediction_points_checkbox, view.get("show_prediction_points")),
            (self.show_sample_triad_checkbox, view.get("show_sample_triad")),
            (self.show_theodolite_sight_line_checkbox, view.get("show_theodolite_sight_line")),
            (self.show_diffraction_vectors_checkbox, view.get("show_diffraction_vectors")),
        )
        checkbox_signal_states = [(checkbox, checkbox.blockSignals(True)) for checkbox, _value in checkbox_values]
        try:
            for checkbox, value in checkbox_values:
                if value is not None:
                    checkbox.setChecked(bool(value))
        finally:
            for checkbox, previous_state in checkbox_signal_states:
                checkbox.blockSignals(previous_state)

        populate_table_from_serialized_rows(
            self.measurement_table,
            TABLE_HEADERS,
            payload.get("measurements", []),
        )
        populate_table_from_serialized_rows(
            self.prediction_table,
            PREDICTION_TABLE_HEADERS,
            payload.get("predictions", []),
            read_only_columns=(7, 8, 9, 10),
        )

        if loaded_mesh is None:
            self.model_mesh = None
            self.mesh_path = None
        else:
            self.model_mesh = loaded_mesh
            self.mesh_path = loaded_mesh_path

        self.measurement_source_text = f"Measurements restored from project {path.name}"
        self.set_project_path_label(path)
        ui_font_size_pt = float(view.get("ui_font_size_pt", DEFAULT_UI_FONT_POINT_SIZE))
        app = QApplication.instance()
        if app is not None:
            apply_application_ui_font(app, ui_font_size_pt)
        self.update_stress_table_metrics()
        if self.ui_font_size_spin is not None:
            previous_state = self.ui_font_size_spin.blockSignals(True)
            self.ui_font_size_spin.setValue(ui_font_size_pt)
            self.ui_font_size_spin.blockSignals(previous_state)
        self.viewer_font_size_offset = int(view.get("viewer_font_size_offset", self.viewer_font_size_offset))
        self.parallel_projection_enabled = bool(
            view.get("parallel_projection_enabled", self.parallel_projection_enabled)
        )
        self.camera_preset = str(view.get("camera_preset", self.camera_preset))
        self.sync_view_button_states()

        self.fit_transform = None
        self.residual_rows = []
        if fit_payload is not None:
            self.fit_transform = Transform(
                rotation=tuple(tuple(float(value) for value in row) for row in fit_payload["rotation"]),
                translation=tuple(float(value) for value in fit_payload["translation"]),
                quaternion_wxyz=tuple(float(value) for value in fit_payload["quaternion_wxyz"]),
                rms_error=float(fit_payload["rms_error"]),
                max_error=float(fit_payload["max_error"]),
            )
            residual_rows = []
            for residual_row in fit_payload.get("residual_rows", []):
                measurement_payload = residual_row["measurement"]
                measurement = Measurement(
                    label=str(measurement_payload["label"]),
                    model_point=tuple(float(value) for value in measurement_payload["model_point"]),
                    stage_readout=tuple(float(value) for value in measurement_payload["stage_readout"]),
                )
                residual_rows.append(
                    (
                        measurement,
                        np.array(residual_row["stage_point"], dtype=float),
                        float(residual_row["fit_error"]),
                    )
                )
            self.residual_rows = residual_rows
            report_text = fit_payload.get("report_text")
            if isinstance(report_text, str) and report_text.strip():
                self.set_report_body(report_text)
            else:
                self.set_report_body(self.build_fit_report(self.fit_transform, self.residual_rows))
        else:
            self.set_report_body(
                "Project loaded.\n\nNo saved fit transform was present.\n"
                "Run Fit placement or enable Manual Sample Placement to establish a transform."
            )

        self.update_placement_status()
        self.update_placement_summary_fields()
        self.invalidate_detector_map(update_scene=False)
        self.sync_point_picker_panel_mesh()
        self.update_scene(reset_camera=True)
        if self.main_tabs is not None:
            active_tab = int(view.get("active_tab", 0))
            self.main_tabs.setCurrentIndex(max(0, min(active_tab, self.main_tabs.count() - 1)))
        self.statusBar().showMessage(f"Loaded project {path.name}", 5000)

    def load_csv(self, path: Path) -> None:
        self.measurement_source_text = str(path)
        self.measurement_table.setRowCount(0)
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path.name} does not contain a CSV header.")
            for row in reader:
                measurement = Measurement(
                    label=row["label"].strip(),
                    model_point=(float(row["model_x"]), float(row["model_y"]), float(row["model_z"])),
                    stage_readout=(float(row["stage_x"]), float(row["stage_y"]), float(row["stage_z"])),
                )
                self.add_measurement_row(measurement)
        self.fit_transform = None
        self.residual_rows = []
        self.update_placement_status()
        self.update_placement_summary_fields()
        self.invalidate_detector_map(update_scene=False)
        self.set_report_body(f"Loaded measurements from {path}")
        self.update_scene(reset_camera=False)

    def load_csv_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load measurement CSV",
            str(Path.cwd()),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path_str:
            return
        try:
            self.load_csv(Path(path_str))
        except Exception as exc:
            self.show_error("Failed to load CSV", str(exc))

    def save_csv_dialog(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save measurement CSV",
            str(Path.cwd() / "measurements.csv"),
            "CSV files (*.csv)",
        )
        if not path_str:
            return
        try:
            measurements = self.measurements_from_table()
            with Path(path_str).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["label", "model_x", "model_y", "model_z", "stage_x", "stage_y", "stage_z", "path_1", "path_2"]
                )
                for measurement in measurements:
                    writer.writerow(
                        [
                            measurement.label,
                            *measurement.model_point,
                            *measurement.stage_readout,
                        ]
                    )
            self.statusBar().showMessage(f"Saved measurements to {path_str}", 5000)
        except Exception as exc:
            self.show_error("Failed to save CSV", str(exc))

    def load_prediction_csv(self, path: Path) -> None:
        self.prediction_table.setRowCount(0)
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path.name} does not contain a CSV header.")
            normalized_fieldnames = {name.strip().lower(): name for name in reader.fieldnames}
            required = ("model_x", "model_y", "model_z")
            missing = [name for name in required if name not in normalized_fieldnames]
            if missing:
                raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
            label_key = normalized_fieldnames.get("label")
            model_keys = [normalized_fieldnames[name] for name in required]
            for row_index, row in enumerate(reader):
                label = row[label_key].strip() if label_key and row.get(label_key) is not None else f"G{row_index + 1}"
                model_values = []
                for key in model_keys:
                    value = row.get(key, "")
                    if value is None or value.strip() == "":
                        raise ValueError(f"Row {row_index + 2} is missing a model coordinate.")
                    model_values.append(float(value))
                self.add_prediction_row(
                    label=label,
                    model_point=model_values,
                    stage_readout=("", "", ""),
                    path_values=("", ""),
                    uamp_values=("", ""),
                )
        self.statusBar().showMessage(f"Loaded prediction points from {path.name}", 5000)
        self.update_scene(reset_camera=False)

    def load_prediction_csv_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load prediction CSV",
            str(Path.cwd()),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path_str:
            return
        try:
            self.load_prediction_csv(Path(path_str))
        except Exception as exc:
            self.show_error("Failed to load prediction CSV", str(exc))

    def save_prediction_csv_dialog(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save prediction CSV",
            str(Path.cwd() / "prediction_points.csv"),
            "CSV files (*.csv)",
        )
        if not path_str:
            return
        try:
            with Path(path_str).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "label",
                        "model_x",
                        "model_y",
                        "model_z",
                        "stage_x",
                        "stage_y",
                        "stage_z",
                        "path_1",
                        "path_2",
                        "uamp1",
                        "uamp2",
                    ]
                )
                for row in range(self.prediction_table.rowCount()):
                    values = []
                    for column in range(self.prediction_table.columnCount()):
                        item = self.prediction_table.item(row, column)
                        values.append("" if item is None else item.text().strip())
                    if not any(values):
                        continue
                    writer.writerow(values)
            self.statusBar().showMessage(f"Saved prediction points to {path_str}", 5000)
        except Exception as exc:
            self.show_error("Failed to save prediction CSV", str(exc))

    def load_mesh_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load sample mesh",
            str(Path.cwd()),
            "Mesh files (*.stl *.ply *.obj *.off *.glb *.gltf);;All files (*.*)",
        )
        if not path_str:
            return
        self.load_mesh(Path(path_str))

    def open_point_picker_dialog(self) -> None:
        if self.main_tabs is None:
            return
        if self.point_picker_panel is not None:
            self.point_picker_panel.set_mesh(self.model_mesh, self.mesh_path)
        for index in range(self.main_tabs.count()):
            if self.main_tabs.tabText(index) == "Pick Point":
                self.main_tabs.setCurrentIndex(index)
                break

    def load_mesh(self, path: Path) -> None:
        try:
            self.model_mesh = load_mesh_as_polydata(path)
            self.mesh_path = path
            self.refresh_report_box()
            self.invalidate_detector_map(update_scene=False)
            self.update_placement_status()
            self.sync_point_picker_panel_mesh()
            self.statusBar().showMessage(f"Loaded mesh {path.name}", 5000)
            self.update_scene(reset_camera=True)
        except Exception as exc:
            self.show_error("Failed to load mesh", str(exc))

    def clear_mesh(self) -> None:
        self.mesh_path = None
        self.model_mesh = None
        self.refresh_report_box()
        self.invalidate_detector_map(update_scene=False)
        self.update_placement_status()
        self.sync_point_picker_panel_mesh()
        self.update_scene(reset_camera=False)

    def fit_placement(self) -> None:
        try:
            measurements = self.measurements_from_table()
            transform, residual_rows = fit_from_measurements(measurements)
            self.fit_transform = transform
            self.residual_rows = residual_rows
            self.update_placement_status()
            self.update_placement_summary_fields()
            self.invalidate_detector_map(update_scene=False)
            self.set_report_body(self.build_fit_report(transform, residual_rows))
            self.statusBar().showMessage("Placement fit completed", 5000)
            self.update_scene(reset_camera=False)
        except Exception as exc:
            self.show_error("Fit failed", str(exc))

    def build_fit_report(self, transform, residual_rows) -> str:
        lines = [
            "Fit completed",
            "",
            "Model -> stage transform",
            f"Translation: {pretty_vector(transform.translation)}",
            f"Quaternion WXYZ: {pretty_vector(transform.quaternion_wxyz)}",
            f"Euler ZYX deg: {pretty_vector(rotation_matrix_to_euler_zyx_deg(transform.rotation))}",
            f"RMS error: {format_decimal(transform.rms_error)}",
            f"Max error: {format_decimal(transform.max_error)}",
            "",
            "Per-point residuals:",
        ]
        for measurement, stage_point, fit_error in residual_rows:
            lines.append(
                f"  {measurement.label}: stage-point={pretty_vector(stage_point)} error={fit_error:.6f}"
            )
        return "\n".join(lines)

    def export_json_dialog(self) -> None:
        if self.fit_transform is None or not self.residual_rows:
            self.show_error("No fit available", "Run Fit placement before exporting JSON.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export fit JSON",
            str(Path.cwd() / "fit_report.json"),
            "JSON files (*.json)",
        )
        if not path_str:
            return
        try:
            save_json_report(Path(path_str), self.fit_transform, self.residual_rows, self.current_pivot_world())
            self.statusBar().showMessage(f"Saved fit JSON to {path_str}", 5000)
        except Exception as exc:
            self.show_error("Failed to export JSON", str(exc))

    def remove_selected_rows(self) -> None:
        rows = sorted({index.row() for index in self.measurement_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            self.measurement_table.removeRow(row)
        self.update_scene(reset_camera=False)

    def remove_selected_prediction_rows(self) -> None:
        rows = sorted({index.row() for index in self.prediction_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            self.prediction_table.removeRow(row)

    def generate_prediction_stage_readouts(self) -> None:
        if self.prediction_table.rowCount() == 0:
            self.show_error("No prediction rows", "Add one or more prediction rows first.")
            return
        rotation, translation, placement_source = self.current_model_to_stage_transform()
        if placement_source == "as-is":
            self.statusBar().showMessage(
                "Generated stage readouts using the imported model orientation as the active transform.",
                5000,
            )
        else:
            self.statusBar().showMessage(
                f"Generated stage readouts using the active {placement_source} placement transform.",
                5000,
            )

        was_blocked = self.prediction_table.blockSignals(True)
        try:
            for row in range(self.prediction_table.rowCount()):
                label_item = self.prediction_table.item(row, 0)
                if label_item is None:
                    self.prediction_table.setItem(row, 0, QTableWidgetItem(f"G{row + 1}"))
                model_values = []
                for column in range(1, 4):
                    item = self.prediction_table.item(row, column)
                    text = "" if item is None else item.text().strip()
                    if text == "":
                        raise ValueError(f"Prediction row {row + 1} is missing model coordinates.")
                    model_values.append(float(text))
                model_point = np.array(model_values, dtype=float)
                stage_point = (rotation @ model_point) + translation
                stage_readout = -stage_point
                for index, value in enumerate(stage_readout, start=4):
                    item = self.prediction_table.item(row, index)
                    if item is None:
                        item = QTableWidgetItem()
                        self.prediction_table.setItem(row, index, item)
                    item.setText(format_decimal(float(value)))
        except Exception as exc:
            self.show_error("Generate stage readouts failed", str(exc))
        finally:
            self.prediction_table.blockSignals(was_blocked)

    def generate_prediction_diffraction_paths(self) -> None:
        if self.prediction_table.rowCount() == 0:
            self.show_error("No prediction rows", "Add one or more prediction rows first.")
            return
        if self.model_mesh is None:
            self.show_error("No mesh loaded", "Load a sample mesh before generating diffraction paths.")
            return

        omega_deg = self.pose_omega.value()
        was_blocked = self.prediction_table.blockSignals(True)
        try:
            for row in range(self.prediction_table.rowCount()):
                model_values = []
                for column in range(1, 4):
                    item = self.prediction_table.item(row, column)
                    text = "" if item is None else item.text().strip()
                    if text == "":
                        raise ValueError(f"Prediction row {row + 1} is missing model coordinates.")
                    model_values.append(float(text))
                model_point = np.array(model_values, dtype=float)
                stage_readout_local = self.stage_readout_for_model_point(model_point)
                bank_1_state, bank_2_state = self.compute_diffraction_bank_states_for_pose(stage_readout_local, omega_deg)
                for index, value in enumerate(
                    (float(bank_1_state["average_length"]), float(bank_2_state["average_length"])),
                    start=7,
                ):
                    item = self.prediction_table.item(row, index)
                    if item is None:
                        item = QTableWidgetItem()
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        self.prediction_table.setItem(row, index, item)
                    item.setText(f"{float(value):.1f}")
            self.statusBar().showMessage("Generated diffraction path columns using the current live Omega.", 5000)
        except Exception as exc:
            self.show_error("Generate diffraction paths failed", str(exc))
        finally:
            self.prediction_table.blockSignals(was_blocked)

    def generate_prediction_estimated_times(self) -> None:
        if self.prediction_table.rowCount() == 0:
            self.show_error("No prediction rows", "Add one or more prediction rows first.")
            return

        gauge_volume_mm3 = self.slit_width.value() * self.slit_height.value() * float(self.collimator.currentText())
        count_time_material = self.count_time_material.currentText()
        was_blocked = self.prediction_table.blockSignals(True)
        try:
            for row in range(self.prediction_table.rowCount()):
                path_values = []
                for column in (7, 8):
                    item = self.prediction_table.item(row, column)
                    text = "" if item is None else item.text().strip()
                    if text == "":
                        raise ValueError(
                            f"Prediction row {row + 1} is missing Path {column - 6}. Generate paths first."
                        )
                    path_values.append(float(text))
                for index, value in enumerate(
                    (
                        enginx_rietveld_count_time_minutes(path_values[0], gauge_volume_mm3, count_time_material),
                        enginx_rietveld_count_time_minutes(path_values[1], gauge_volume_mm3, count_time_material),
                    ),
                    start=9,
                ):
                    item = self.prediction_table.item(row, index)
                    if item is None:
                        item = QTableWidgetItem()
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        self.prediction_table.setItem(row, index, item)
                    item.setText(str(int(np.ceil(float(value)))))
            self.statusBar().showMessage(
                f"Estimated Rietveld count times using {count_time_material} and the current gauge volume.",
                5000,
            )
        except Exception as exc:
            self.show_error("Estimate time failed", str(exc))
        finally:
            self.prediction_table.blockSignals(was_blocked)

    def prompt_scan_file_options(self, default_title: str = "scan_file") -> Optional[Tuple[str, str]]:
        dialog = QDialog(self)
        dialog.setWindowTitle("Scan file options")
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        title_edit = QLineEdit(default_title, dialog)
        bank_combo = NoWheelComboBox(dialog)
        bank_combo.addItems(["Both banks", "Bank 1", "Bank 2"])
        form.addRow("Title", title_edit)
        form.addRow("Banks", bank_combo)
        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return None
        title = title_edit.text().strip() or default_title
        return title, bank_combo.currentText()

    def create_prediction_scan_file(self) -> None:
        if self.prediction_table.rowCount() == 0:
            self.show_error("No prediction rows", "Add one or more prediction rows first.")
            return

        scan_rows: List[Tuple[float, float, float, float, int]] = []
        omega_deg = float(self.pose_omega.value())
        scan_options = self.prompt_scan_file_options()
        if scan_options is None:
            return
        title, bank_selection = scan_options
        try:
            for row in range(self.prediction_table.rowCount()):
                stage_values = []
                for column, axis_name in zip((4, 5, 6), ("X", "Y", "Z")):
                    item = self.prediction_table.item(row, column)
                    text = "" if item is None else item.text().strip()
                    if text == "":
                        raise ValueError(
                            f"Prediction row {row + 1} is missing Stage {axis_name}. Generate stage readouts first."
                        )
                    stage_values.append(float(text))

                uamp_values = []
                for column, label in zip((9, 10), ("UAmp1", "UAmp2")):
                    item = self.prediction_table.item(row, column)
                    text = "" if item is None else item.text().strip()
                    if text == "":
                        raise ValueError(
                            f"Prediction row {row + 1} is missing {label}. Estimate time first."
                        )
                    uamp_values.append(float(text))

                if bank_selection == "Bank 1":
                    selected_uamp = int(np.ceil(uamp_values[0]))
                elif bank_selection == "Bank 2":
                    selected_uamp = int(np.ceil(uamp_values[1]))
                else:
                    selected_uamp = int(np.ceil(max(uamp_values)))
                scan_rows.append((stage_values[0], stage_values[1], stage_values[2], omega_deg, selected_uamp))
        except Exception as exc:
            self.show_error("Create scan file failed", str(exc))
            return

        default_path = Path.cwd() / "scan_file.txt"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Create scan file",
            str(default_path),
            "Text files (*.txt);;All files (*.*)",
        )
        if not path_str:
            return

        output_path = Path(path_str)

        try:
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                handle.write(f"{title}\n")
                handle.write("1\n")
                handle.write(f"{len(scan_rows)}\n")
                for stage_x, stage_y, stage_z, omega_value, uamp_value in scan_rows:
                    handle.write(
                        "\t".join(
                            [
                                f"{stage_x:.3f}",
                                f"{stage_y:.3f}",
                                f"{stage_z:.3f}",
                                format_scan_numeric(omega_value, decimals=3),
                                str(int(uamp_value)),
                            ]
                        )
                        + "\n"
                    )
            self.statusBar().showMessage(f"Created scan file {output_path}", 5000)
        except Exception as exc:
            self.show_error("Create scan file failed", str(exc))

    def on_manual_placement_mode_toggled(self, checked: bool) -> None:
        self.update_placement_status()
        self.update_placement_summary_fields()
        self.invalidate_detector_map(update_scene=False)
        mode_text = "enabled" if checked else "disabled"
        self.statusBar().showMessage(f"Manual placement {mode_text}.", 5000)
        self.update_scene(reset_camera=False)

    def on_manual_placement_changed(self) -> None:
        if not self.manual_placement_enabled_checkbox.isChecked():
            return
        self.update_placement_status()
        self.update_placement_summary_fields()
        self.invalidate_detector_map(update_scene=False)
        self.update_scene(reset_camera=False)

    def on_manual_rotation_increment_changed(self) -> None:
        new_values = np.array(
            [self.manual_rx.value(), self.manual_ry.value(), self.manual_rz.value()],
            dtype=float,
        )
        deltas = new_values - self.manual_rotation_display_values
        delta_x, delta_y, delta_z = deltas.tolist()
        if max(abs(delta_x), abs(delta_y), abs(delta_z)) < 1e-12:
            return

        rotation = np.array(self.manual_rotation_matrix, dtype=float)
        if abs(delta_x) >= 1e-12:
            rotation = rotation @ rotation_matrix_from_euler_xyz_deg(delta_x, 0.0, 0.0)
        if abs(delta_y) >= 1e-12:
            rotation = rotation @ rotation_matrix_from_euler_xyz_deg(0.0, delta_y, 0.0)
        if abs(delta_z) >= 1e-12:
            rotation = rotation @ rotation_matrix_from_euler_xyz_deg(0.0, 0.0, delta_z)
        self.manual_rotation_matrix = orthonormalize_rotation_matrix(rotation)
        self.manual_rotation_display_values = new_values
        self.sync_manual_rotation_spin_boxes_from_matrix()

        if not self.manual_placement_enabled_checkbox.isChecked():
            return
        self.update_placement_status()
        self.update_placement_summary_fields()
        self.invalidate_detector_map(update_scene=False)
        self.update_scene(reset_camera=False)

    def reset_manual_placement(self) -> None:
        translation_boxes = (
            self.manual_tx,
            self.manual_ty,
            self.manual_tz,
        )
        previous_states = [box.blockSignals(True) for box in translation_boxes]
        for box in translation_boxes:
            box.setValue(0.0)
        for box, previous in zip(translation_boxes, previous_states):
            box.blockSignals(previous)
        self.manual_rotation_matrix = np.eye(3, dtype=float)
        self.manual_rotation_display_values = np.zeros(3, dtype=float)
        self.sync_manual_rotation_spin_boxes_from_matrix()
        self.update_placement_status()
        self.update_placement_summary_fields()
        self.invalidate_detector_map(update_scene=False)
        self.update_scene(reset_camera=False)

    def load_fit_into_manual(self) -> None:
        if self.fit_transform is None:
            self.show_error("No fit available", "Run Fit placement before loading fit values into manual placement.")
            return
        translation = np.array(self.fit_transform.translation, dtype=float)
        translation_boxes = (
            self.manual_tx,
            self.manual_ty,
            self.manual_tz,
        )
        previous_states = [box.blockSignals(True) for box in translation_boxes]
        self.manual_tx.setValue(float(translation[0]))
        self.manual_ty.setValue(float(translation[1]))
        self.manual_tz.setValue(float(translation[2]))
        for box, previous in zip(translation_boxes, previous_states):
            box.blockSignals(previous)
        self.manual_rotation_matrix = orthonormalize_rotation_matrix(np.array(self.fit_transform.rotation, dtype=float))
        self.manual_rotation_display_values = np.array(
            rotation_matrix_to_euler_xyz_deg(np.array(self.fit_transform.rotation, dtype=float)),
            dtype=float,
        )
        self.sync_manual_rotation_spin_boxes_from_matrix()
        previous_toggle_state = self.manual_placement_enabled_checkbox.blockSignals(True)
        self.manual_placement_enabled_checkbox.setChecked(True)
        self.manual_placement_enabled_checkbox.blockSignals(previous_toggle_state)
        self.update_placement_status()
        self.update_placement_summary_fields()
        self.invalidate_detector_map(update_scene=False)
        self.statusBar().showMessage("Loaded fit transform into manual placement.", 5000)
        self.update_scene(reset_camera=False)

    def clear_placement(self) -> None:
        translation_boxes = (
            self.manual_tx,
            self.manual_ty,
            self.manual_tz,
        )
        previous_states = [box.blockSignals(True) for box in translation_boxes]
        for box in translation_boxes:
            box.setValue(0.0)
        for box, previous in zip(translation_boxes, previous_states):
            box.blockSignals(previous)
        self.manual_rotation_matrix = np.eye(3, dtype=float)
        self.manual_rotation_display_values = np.zeros(3, dtype=float)
        self.sync_manual_rotation_spin_boxes_from_matrix()

        previous_toggle_state = self.manual_placement_enabled_checkbox.blockSignals(True)
        self.manual_placement_enabled_checkbox.setChecked(False)
        self.manual_placement_enabled_checkbox.blockSignals(previous_toggle_state)

        self.fit_transform = None
        self.residual_rows = []
        self.set_report_body(
            "Placement cleared.\n\n"
            "The model is now shown as imported.\n"
            "Run Fit placement or enable Manual Sample Placement to establish a new transform."
        )
        self.update_placement_status()
        self.update_placement_summary_fields()
        self.invalidate_detector_map(update_scene=False)
        self.statusBar().showMessage("Placement transform cleared.", 5000)
        self.update_scene(reset_camera=False)

    def use_selected_row_pose(self) -> None:
        row = self.measurement_table.currentRow()
        if row < 0:
            self.show_error("No row selected", "Select a measurement row first.")
            return
        try:
            values = []
            for column in range(self.measurement_table.columnCount()):
                item = self.measurement_table.item(row, column)
                values.append("" if item is None else item.text().strip())
            if any(value == "" for value in values[4:7]):
                raise ValueError("The selected row does not contain a full stage pose.")
            self.pose_x.setValue(float(values[4]))
            self.pose_y.setValue(float(values[5]))
            self.pose_z.setValue(float(values[6]))
        except Exception as exc:
            self.show_error("Invalid selected row", str(exc))

    def reset_pose(self) -> None:
        self.pose_x.setValue(0.0)
        self.pose_y.setValue(0.0)
        self.pose_z.setValue(0.0)
        self.pose_omega.setValue(0.0)

    def selected_measurement_index(self) -> Optional[int]:
        row = self.measurement_table.currentRow()
        if row < 0:
            return None
        return row

    def selected_prediction_index(self) -> Optional[int]:
        row = self.prediction_table.currentRow()
        if row < 0:
            return None
        return row

    def selected_measurement(self) -> Optional[Measurement]:
        row = self.selected_measurement_index()
        if row is None:
            return None
        values = []
        for column in range(self.measurement_table.columnCount()):
            item = self.measurement_table.item(row, column)
            values.append("" if item is None else item.text().strip())
        if any(value == "" for value in values):
            raise ValueError("The selected row is incomplete.")
        return Measurement(
            label=values[0],
            model_point=(float(values[1]), float(values[2]), float(values[3])),
            stage_readout=(float(values[4]), float(values[5]), float(values[6])),
        )

    def selected_prediction_row(self) -> Optional[Tuple[str, np.ndarray]]:
        row = self.selected_prediction_index()
        if row is None:
            return None
        label_item = self.prediction_table.item(row, 0)
        label = "" if label_item is None else label_item.text().strip()
        label = label or f"G{row + 1}"
        model_values = []
        for column in range(1, 4):
            item = self.prediction_table.item(row, column)
            text = "" if item is None else item.text().strip()
            if text == "":
                raise ValueError("The selected prediction row is missing model coordinates.")
            model_values.append(float(text))
        return label, np.array(model_values, dtype=float)

    def move_model_point_to_pivot(self, model_point: np.ndarray, label: str) -> None:
        stage_readout = self.stage_readout_for_model_point(np.asarray(model_point, dtype=float))

        spin_boxes = (self.pose_x, self.pose_y, self.pose_z)
        previous_states = [box.blockSignals(True) for box in spin_boxes]
        self.pose_x.setValue(float(stage_readout[0]))
        self.pose_y.setValue(float(stage_readout[1]))
        self.pose_z.setValue(float(stage_readout[2]))
        for box, previous in zip(spin_boxes, previous_states):
            box.blockSignals(previous)

        self.invalidate_detector_map(update_scene=False)
        self.statusBar().showMessage(
            f"Moved {label} to the pivot using the stage-local readout model.",
            5000,
        )
        self.update_scene(reset_camera=False)

    def move_selected_point_to_pivot(self, show_error: bool = True) -> None:
        try:
            if self.fit_transform is None and not self.manual_placement_enabled_checkbox.isChecked():
                raise ValueError("Run Fit placement or enable manual placement before moving a point to the pivot.")
            measurement = self.selected_measurement()
            if measurement is None:
                raise ValueError("Select a measurement row first.")
            self.move_model_point_to_pivot(np.array(measurement.model_point, dtype=float), measurement.label)
        except Exception as exc:
            if show_error:
                self.show_error("Move to pivot failed", str(exc))

    def move_selected_prediction_point_to_pivot(self, show_error: bool = True) -> None:
        try:
            prediction_row = self.selected_prediction_row()
            if prediction_row is None:
                raise ValueError("Select a prediction row first.")
            label, model_point = prediction_row
            self.move_model_point_to_pivot(model_point, label)
        except Exception as exc:
            if show_error:
                self.show_error("Move to pivot failed", str(exc))

    def infer_measurement_rows_without_fit(self):
        rows = []
        try:
            measurements = self.measurements_from_table()
        except Exception:
            return rows
        for measurement in measurements:
            rows.append(
                (
                    measurement,
                    infer_stage_point_from_readout(measurement.stage_readout),
                    0.0,
                )
            )
        return rows

    def compute_detector_map(self) -> None:
        try:
            model_world = self.build_current_model_world_mesh()
            if model_world is None or model_world.n_points == 0:
                raise ValueError("Load a sample mesh before computing the detector map.")

            slit_center = np.array([self.slit_x.value(), self.slit_y.value(), self.slit_z.value()], dtype=float)
            slit_width = self.slit_width.value()
            slit_height = self.slit_height.value()
            pixel_size_y = self.detector_map_pixel_size_y.value()
            pixel_size_z = self.detector_map_pixel_size_z.value()
            beam_trace_distance = max(
                self.beam_length.value(),
                DIRECT_DETECTOR_CENTER_WORLD[0] - slit_center[0] + 50.0,
                self.scene_scale() * 2.5,
            )
            y_coords, z_coords, path_map = compute_collimated_beam_map(
                model_world,
                slit_center,
                slit_width,
                slit_height,
                pixel_size_y,
                pixel_size_z,
                beam_trace_distance,
            )
            resolution_y = len(y_coords)
            resolution_z = len(z_coords)
            yy, zz = np.meshgrid(y_coords, z_coords, indexing="ij")
            xx = np.full_like(
                yy,
                DIRECT_DETECTOR_CENTER_WORLD[0] - DETECTOR_THICKNESS / 2.0 - 0.2,
                dtype=float,
            )
            detector_map_mesh = pv.StructuredGrid(xx, yy, zz)
            detector_map_mesh["path_length"] = path_map.ravel(order="F")
            max_length = float(np.max(path_map)) if path_map.size else 0.0
            average_length = float(np.mean(path_map)) if path_map.size else 0.0
            self.detector_map_state = {
                "mesh": detector_map_mesh,
                "path_map": path_map,
                "y_coords": y_coords,
                "z_coords": z_coords,
                "slit_center": slit_center,
                "slit_width": slit_width,
                "slit_height": slit_height,
                "max_length": max_length,
                "average_length": average_length,
                "resolution_y": resolution_y,
                "resolution_z": resolution_z,
                "pixel_size_y": float(pixel_size_y),
                "pixel_size_z": float(pixel_size_z),
                "detector_center": np.array(DIRECT_DETECTOR_CENTER_WORLD, dtype=float),
                "detector_width_y": self.detector_width_y.value(),
                "detector_height_z": self.detector_height_z.value(),
                "stage_pose": {
                    "x": self.pose_x.value(),
                    "y": self.pose_y.value(),
                    "z": self.pose_z.value(),
                    "omega": self.pose_omega.value(),
                },
            }
            self.refresh_report_box()
            self.statusBar().showMessage("Computed imaging path-length map.", 5000)
            self.update_scene(reset_camera=False)
        except Exception as exc:
            self.show_error("Detector map failed", str(exc))

    def compute_diffraction_map(self) -> None:
        try:
            stage_readout_local, omega_deg = self.current_stage_pose()
            self.diffraction_bank_1_map_state, self.diffraction_bank_2_map_state = self.compute_diffraction_bank_states_for_pose(
                stage_readout_local,
                omega_deg,
            )
            self.refresh_report_box()
            self.statusBar().showMessage("Computed diffraction path-length maps for banks 1 and 2.", 5000)
            self.update_scene(reset_camera=False)
        except Exception as exc:
            self.show_error("Diffraction map failed", str(exc))

    def diffraction_angle_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        horizontal_angles_deg = np.arange(
            -ANGLED_DETECTOR_HORIZONTAL_HALF_ANGLE_DEG,
            ANGLED_DETECTOR_HORIZONTAL_HALF_ANGLE_DEG + DIFFRACTION_ANGLE_INTERVAL_DEG * 0.5,
            DIFFRACTION_ANGLE_INTERVAL_DEG,
            dtype=float,
        )
        vertical_angles_deg = np.arange(
            -ANGLED_DETECTOR_VERTICAL_HALF_ANGLE_DEG,
            ANGLED_DETECTOR_VERTICAL_HALF_ANGLE_DEG + DIFFRACTION_ANGLE_INTERVAL_DEG * 0.5,
            DIFFRACTION_ANGLE_INTERVAL_DEG,
            dtype=float,
        )
        return horizontal_angles_deg, vertical_angles_deg

    def compute_diffraction_bank_states_for_pose(
        self,
        stage_readout_local: np.ndarray,
        omega_deg: float,
    ) -> Tuple[dict, dict]:
        model_world = self.build_model_world_mesh_for_pose(stage_readout_local, omega_deg)
        if model_world is None or model_world.n_points == 0:
            raise ValueError("Load a sample mesh before computing the diffraction map.")

        pivot = np.array(self.current_pivot_world(), dtype=float)
        signed_distance_evaluator = build_mesh_signed_distance_evaluator(model_world)
        horizontal_angles_deg, vertical_angles_deg = self.diffraction_angle_axes()
        slit_center = np.array([self.slit_x.value(), self.slit_y.value(), self.slit_z.value()], dtype=float)
        incoming_path_length = compute_incoming_beam_path_to_point(model_world, slit_center, pivot)
        stage_pose_info = {
            "x": float(stage_readout_local[0]),
            "y": float(stage_readout_local[1]),
            "z": float(stage_readout_local[2]),
            "omega": float(omega_deg),
        }
        bank_1_state = self._compute_single_diffraction_bank_map(
            model_world,
            pivot,
            self.current_diffraction_bank_1_geometry(),
            horizontal_angles_deg,
            vertical_angles_deg,
            incoming_path_length,
            stage_pose_info,
            signed_distance_evaluator,
        )
        bank_2_state = self._compute_single_diffraction_bank_map(
            model_world,
            pivot,
            self.current_diffraction_bank_2_geometry(),
            horizontal_angles_deg,
            vertical_angles_deg,
            incoming_path_length,
            stage_pose_info,
            signed_distance_evaluator,
        )
        return bank_1_state, bank_2_state

    def _compute_single_diffraction_bank_map(
        self,
        model_world: pv.PolyData,
        pivot: np.ndarray,
        detector_geometry: dict,
        horizontal_angles_deg: np.ndarray,
        vertical_angles_deg: np.ndarray,
        incoming_path_length: float,
        stage_pose_info: dict,
        signed_distance_evaluator: Callable[[np.ndarray], float],
    ) -> dict:
        center_direction = normalized(detector_geometry["center"] - pivot)
        right = detector_geometry["right"]
        up = detector_geometry["up"]
        horizontal_scales = np.tan(np.radians(horizontal_angles_deg))
        vertical_scales = np.tan(np.radians(vertical_angles_deg))
        path_map = np.full(
            (len(horizontal_angles_deg), len(vertical_angles_deg)),
            incoming_path_length,
            dtype=float,
        )
        detector_points = np.zeros((len(horizontal_angles_deg), len(vertical_angles_deg), 3), dtype=float)
        all_segments: List[Tuple[np.ndarray, np.ndarray]] = []

        for h_index, horizontal_scale in enumerate(horizontal_scales):
            for v_index, vertical_scale in enumerate(vertical_scales):
                ray_direction = normalized(center_direction + horizontal_scale * right + vertical_scale * up)
                denominator = float(np.dot(ray_direction, detector_geometry["normal"]))
                if abs(denominator) < 1e-12:
                    continue
                distance_to_plane = float(
                    np.dot(detector_geometry["center"] - pivot, detector_geometry["normal"]) / denominator
                )
                if distance_to_plane <= 0.0:
                    continue
                detector_point = pivot + ray_direction * distance_to_plane
                detector_points[h_index, v_index] = (
                    detector_point + detector_geometry["normal"] * (DETECTOR_THICKNESS / 2.0 + 0.2)
                )
                path_length, segments = compute_segment_path_length(
                    model_world,
                    pivot,
                    detector_point,
                    signed_distance_evaluator=signed_distance_evaluator,
                    collect_segments=False,
                )
                path_map[h_index, v_index] += path_length
                all_segments.extend(segments)

        detector_map_mesh = pv.StructuredGrid(
            detector_points[..., 0],
            detector_points[..., 1],
            detector_points[..., 2],
        )
        detector_map_mesh["path_length"] = path_map.ravel(order="F")
        max_length = float(np.max(path_map)) if path_map.size else 0.0
        average_length = float(np.mean(path_map)) if path_map.size else 0.0
        return {
            "mesh": detector_map_mesh,
            "path_map": path_map,
            "horizontal_angles_deg": horizontal_angles_deg,
            "vertical_angles_deg": vertical_angles_deg,
            "detector_center": np.array(detector_geometry["center"], dtype=float),
            "detector_normal": np.array(detector_geometry["normal"], dtype=float),
            "detector_width": float(detector_geometry["width"]),
            "detector_height": float(detector_geometry["height"]),
            "max_length": max_length,
            "average_length": average_length,
            "incoming_path_length": incoming_path_length,
            "interval_deg": float(DIFFRACTION_ANGLE_INTERVAL_DEG),
            "segments": all_segments,
            "stage_pose": stage_pose_info,
        }

    def export_detector_map_dialog(self) -> None:
        if self.detector_map_state is None:
            self.show_error("No detector map", "Compute the detector map before exporting it.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export detector map",
            str(Path.cwd() / "detector_map.fits"),
            "FITS files (*.fits);;TIFF files (*.tif *.tiff)",
        )
        if not path_str:
            return
        try:
            path = Path(path_str)
            suffix = path.suffix.lower()
            if suffix == ".fits":
                self.export_detector_map_fits(path)
            elif suffix in {".tif", ".tiff"}:
                self.export_detector_map_tiff(path)
            else:
                raise ValueError("Use a .fits, .tif, or .tiff filename.")
            self.statusBar().showMessage(f"Exported detector map to {path}", 5000)
        except Exception as exc:
            self.show_error("Detector map export failed", str(exc))

    def export_detector_map_fits(self, path: Path) -> None:
        if fits is None:
            raise ImportError("FITS export requires astropy. Install it with pip install astropy.")
        state = self.detector_map_state
        if state is None:
            raise ValueError("No detector map is available.")
        path_map = np.asarray(state["path_map"], dtype=np.float32)
        header = fits.Header()
        header["BUNIT"] = "mm"
        header["CTYPE1"] = "Y"
        header["CTYPE2"] = "Z"
        header["CRPIX1"] = 1.0
        header["CRPIX2"] = 1.0
        header["CRVAL1"] = float(state["y_coords"][0])
        header["CRVAL2"] = float(state["z_coords"][0])
        header["CDELT1"] = float(state["y_coords"][1] - state["y_coords"][0]) if len(state["y_coords"]) > 1 else 0.0
        header["CDELT2"] = float(state["z_coords"][1] - state["z_coords"][0]) if len(state["z_coords"]) > 1 else 0.0
        header["SLITX"] = float(state["slit_center"][0])
        header["SLITY"] = float(state["slit_center"][1])
        header["SLITZ"] = float(state["slit_center"][2])
        header["SLITW"] = float(state["slit_width"])
        header["SLITH"] = float(state["slit_height"])
        header["DETX"] = float(state["detector_center"][0])
        header["DETWY"] = float(state["detector_width_y"])
        header["DETHZ"] = float(state["detector_height_z"])
        header["PIXY"] = float(state["pixel_size_y"])
        header["PIXZ"] = float(state["pixel_size_z"])
        header["AVGPATH"] = float(state["average_length"])
        header["MAXPATH"] = float(state["max_length"])
        header["POSEX"] = float(state["stage_pose"]["x"])
        header["POSEY"] = float(state["stage_pose"]["y"])
        header["POSEZ"] = float(state["stage_pose"]["z"])
        header["OMEGA"] = float(state["stage_pose"]["omega"])
        fits.PrimaryHDU(data=path_map, header=header).writeto(path, overwrite=True)

    def export_detector_map_tiff(self, path: Path) -> None:
        if tifffile is None:
            raise ImportError("TIFF export requires tifffile. Install it with pip install tifffile.")
        state = self.detector_map_state
        if state is None:
            raise ValueError("No detector map is available.")
        metadata = {
            "unit": "mm",
            "axis_1": "Y",
            "axis_2": "Z",
            "y_coords": np.asarray(state["y_coords"], dtype=float).tolist(),
            "z_coords": np.asarray(state["z_coords"], dtype=float).tolist(),
            "slit_center": np.asarray(state["slit_center"], dtype=float).tolist(),
            "slit_width": float(state["slit_width"]),
            "slit_height": float(state["slit_height"]),
            "detector_center": np.asarray(state["detector_center"], dtype=float).tolist(),
            "detector_width_y": float(state["detector_width_y"]),
            "detector_height_z": float(state["detector_height_z"]),
            "pixel_size_y": float(state["pixel_size_y"]),
            "pixel_size_z": float(state["pixel_size_z"]),
            "average_path": float(state["average_length"]),
            "max_path": float(state["max_length"]),
            "stage_pose": state["stage_pose"],
        }
        tifffile.imwrite(
            path,
            np.asarray(state["path_map"], dtype=np.float32),
            description=json.dumps(metadata),
        )

    def on_projection_toggled(self, checked: bool) -> None:
        self.parallel_projection_enabled = checked
        self.apply_projection_mode()

    def apply_projection_mode(self) -> None:
        if not self.enable_3d or self.plotter is None:
            return
        if self.parallel_projection_enabled:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.plotter.render()

    def set_or_add_mesh(self, attr_name: str, mesh: pv.PolyData, actor_name: str, **kwargs) -> None:
        existing = getattr(self, attr_name)
        if mesh is None or mesh.n_points == 0:
            if existing is not None:
                self.plotter.remove_actor(actor_name, render=False)
                setattr(self, attr_name, None)
            return
        if existing is None:
            existing = mesh.copy(deep=True)
            setattr(self, attr_name, existing)
            self.plotter.add_mesh(existing, name=actor_name, render=False, reset_camera=False, **kwargs)
        else:
            existing.deep_copy(mesh)

    def ensure_scene_initialized(self) -> None:
        if self.scene_initialized:
            return
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.scene_initialized = True

    def viewer_font_size(self, base_size: int) -> int:
        return max(8, min(48, int(base_size + self.viewer_font_size_offset)))

    def adjust_viewer_font_size(self, delta: int) -> None:
        new_offset = max(-6, min(20, self.viewer_font_size_offset + delta))
        if new_offset == self.viewer_font_size_offset:
            return
        self.viewer_font_size_offset = new_offset
        self.statusBar().showMessage(f"Viewer font size {self.viewer_font_size(16)}", 3000)
        self.update_scene(reset_camera=False)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.KeyPress:
            key_text = event.text()
            if key_text == ">":
                self.adjust_viewer_font_size(1)
                return True
            if key_text == "<":
                self.adjust_viewer_font_size(-1)
                return True
        return super().eventFilter(watched, event)

    def remove_legacy_model_axis_actors(self) -> None:
        self.plotter.remove_actor("model_axis_labels", render=False)
        self.plotter.remove_actor("model_axis_label_points", render=False)
        for mesh_attr, actor_name in (
            ("model_x_axis_mesh", "model_x_axis"),
            ("model_y_axis_mesh", "model_y_axis"),
            ("model_z_axis_mesh", "model_z_axis"),
        ):
            if hasattr(self, mesh_attr):
                mesh_value = getattr(self, mesh_attr)
                if mesh_value is not None:
                    self.plotter.remove_actor(actor_name, render=False)
                    setattr(self, mesh_attr, None)
        if hasattr(self, "model_axis_labels_mesh"):
            self.model_axis_labels_mesh = None

    def remove_stage_visual_actors(self) -> None:
        self.plotter.remove_actor("stage_world", render=False)
        self.plotter.remove_actor("stage_axis_labels", render=False)
        self.plotter.remove_actor("stage_axis_label_points", render=False)
        for mesh_attr, actor_name in (
            ("stage_x_arrow_pos_mesh", "stage_x_arrow_pos"),
            ("stage_x_arrow_pos_secondary_mesh", "stage_x_arrow_pos_secondary"),
            ("stage_x_arrow_neg_mesh", "stage_x_arrow_neg"),
            ("stage_x_arrow_neg_secondary_mesh", "stage_x_arrow_neg_secondary"),
            ("stage_y_arrow_pos_mesh", "stage_y_arrow_pos"),
            ("stage_y_arrow_pos_secondary_mesh", "stage_y_arrow_pos_secondary"),
            ("stage_y_arrow_neg_mesh", "stage_y_arrow_neg"),
            ("stage_y_arrow_neg_secondary_mesh", "stage_y_arrow_neg_secondary"),
        ):
            mesh_value = getattr(self, mesh_attr)
            if mesh_value is not None:
                self.plotter.remove_actor(actor_name, render=False)
                setattr(self, mesh_attr, None)
        self.stage_world_mesh = None
        self.stage_axis_labels_mesh = None

    def scene_focus_point(self) -> np.ndarray:
        pivot = np.array(self.current_pivot_world(), dtype=float)
        stage_readout_local, omega_deg = self.current_stage_pose()
        stage_box = self.stage_box_local()
        stage_points_world = stage_local_to_world(stage_box.points, stage_readout_local, omega_deg, pivot)
        scene_points = [stage_points_world]

        model_world = self.build_current_model_world_mesh()
        if model_world is not None:
            scene_points.append(np.asarray(model_world.points, dtype=float))
        prediction_rows = self.prediction_rows_from_table()
        if prediction_rows:
            rotation, translation, _source = self.current_model_to_stage_transform()
            prediction_model_points = np.array([row[1] for row in prediction_rows], dtype=float)
            prediction_stage_points = (rotation @ prediction_model_points.T).T + translation
            prediction_world_points = stage_local_to_world(prediction_stage_points, stage_readout_local, omega_deg, pivot)
            scene_points.append(prediction_world_points)

        slit_center = np.array([self.slit_x.value(), self.slit_y.value(), self.slit_z.value()], dtype=float)
        slit_width = self.slit_width.value()
        slit_height = self.slit_height.value()
        direct_detector_center = np.array(DIRECT_DETECTOR_CENTER_WORLD, dtype=float)
        direct_detector_width = self.detector_width_y.value()
        direct_detector_height = self.detector_height_z.value()
        diffraction_bank_1_geometry = self.current_diffraction_bank_1_geometry()
        diffraction_bank_2_geometry = self.current_diffraction_bank_2_geometry()
        scene_points.append(
            np.array(
                [
                    slit_center + np.array([0.0, -slit_width / 2.0, -slit_height / 2.0], dtype=float),
                    slit_center + np.array([self.beam_length.value(), slit_width / 2.0, slit_height / 2.0], dtype=float),
                    direct_detector_center + np.array([0.0, -direct_detector_width / 2.0, -direct_detector_height / 2.0], dtype=float),
                    direct_detector_center + np.array([0.0, direct_detector_width / 2.0, direct_detector_height / 2.0], dtype=float),
                    *diffraction_bank_1_geometry["corners"],
                    *diffraction_bank_2_geometry["corners"],
                ],
                dtype=float,
            )
        )

        focus = np.vstack(scene_points).mean(axis=0)

        if np.any(~np.isfinite(focus)):
            return pivot
        return focus

    def scene_scale(self) -> float:
        dims = np.array([self.stage_size_x.value(), self.stage_size_y.value(), self.stage_size_z.value()], dtype=float)
        scale = max(float(np.linalg.norm(dims)), 200.0)
        if self.model_mesh is not None:
            bounds = np.array(self.model_mesh.bounds, dtype=float)
            model_scale = float(np.linalg.norm(bounds[1] - bounds[0]))
            scale = max(scale, model_scale * 1.5)
        direct_detector_extent = np.linalg.norm(
            np.array([DIRECT_DETECTOR_CENTER_WORLD[0], self.detector_width_y.value(), self.detector_height_z.value()], dtype=float)
        )
        diffraction_bank_1_geometry = self.current_diffraction_bank_1_geometry()
        diffraction_bank_2_geometry = self.current_diffraction_bank_2_geometry()
        diffraction_bank_1_extent = max(
            float(np.linalg.norm(diffraction_bank_1_geometry["center"])),
            float(np.max(np.linalg.norm(diffraction_bank_1_geometry["corners"], axis=1))),
        )
        diffraction_bank_2_extent = max(
            float(np.linalg.norm(diffraction_bank_2_geometry["center"])),
            float(np.max(np.linalg.norm(diffraction_bank_2_geometry["corners"], axis=1))),
        )
        beam_extent = np.linalg.norm(
            np.array([self.beam_length.value(), self.slit_width.value(), self.slit_height.value()], dtype=float)
        )
        scale = max(
            scale,
            float(direct_detector_extent),
            float(diffraction_bank_1_extent),
            float(diffraction_bank_2_extent),
            float(beam_extent),
        )
        return scale

    def apply_camera_preset(self, reset_camera: bool = True) -> None:
        if not self.enable_3d or self.plotter is None:
            return

        focus = self.scene_focus_point()
        pivot = np.array(self.current_pivot_world(), dtype=float)
        distance = self.scene_scale() * 0.8
        preset = self.camera_preset

        def offset(direction: np.ndarray) -> np.ndarray:
            return normalized(np.asarray(direction, dtype=float)) * distance

        if preset == "iso":
            position = focus + offset(np.array([1.0, -1.0, 1.0], dtype=float))
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus
        elif preset == "+x":
            position = focus + offset(np.array([1.0, 0.0, 0.0], dtype=float))
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus
        elif preset == "-x":
            position = focus + offset(np.array([-1.0, 0.0, 0.0], dtype=float))
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus
        elif preset == "+y":
            position = focus + offset(np.array([0.0, 1.0, 0.0], dtype=float))
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus
        elif preset == "-y":
            position = focus + offset(np.array([0.0, -1.0, 0.0], dtype=float))
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus
        elif preset == "+z":
            position = focus + offset(np.array([0.0, 0.0, 1.0], dtype=float))
            view_up = (0.0, 1.0, 0.0)
            focal_point = focus
        elif preset == "-z":
            position = focus + offset(np.array([0.0, 0.0, -1.0], dtype=float))
            view_up = (0.0, 1.0, 0.0)
            focal_point = focus
        elif preset == "theodolite":
            position = np.array(
                [self.theodolite_x.value(), self.theodolite_y.value(), self.theodolite_z.value()],
                dtype=float,
            )
            focal_point = pivot
            view_up = (0.0, 0.0, 1.0)
        else:
            position = focus + offset(np.array([1.0, -1.0, 1.0], dtype=float))
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus

        self.plotter.camera_position = [tuple(position), tuple(focal_point), view_up]
        self.apply_projection_mode()
        self.plotter.reset_camera_clipping_range()
        self.apply_projection_mode()

    def set_camera_preset(self, preset: str) -> None:
        self.camera_preset = preset
        self.sync_view_button_states()
        self.apply_camera_preset(reset_camera=True)

    def sync_view_button_states(self) -> None:
        for preset, button in self.view_buttons.items():
            was_blocked = button.blockSignals(True)
            button.setChecked(preset == self.camera_preset)
            button.blockSignals(was_blocked)

    def update_scene(self, reset_camera: bool = False) -> None:
        if not self.enable_3d or self.plotter is None:
            return

        self.ensure_scene_initialized()

        camera_state = None
        if not reset_camera and self.plotter.renderer is not None:
            camera_state = capture_camera_state(self.plotter)

        pivot = np.array(self.current_pivot_world(), dtype=float)
        stage_readout_local, omega_deg = self.current_stage_pose()
        show_stage = self.show_stage_checkbox is None or self.show_stage_checkbox.isChecked()
        if show_stage:
            stage_box = self.stage_box_local()
            stage_points = stage_local_to_world(stage_box.points, stage_readout_local, omega_deg, pivot)
            stage_world = pv.PolyData(stage_points, stage_box.faces)
            self.set_or_add_mesh(
                "stage_world_mesh",
                stage_world,
                "stage_world",
                color="#d7d7d7",
                opacity=1.0,
                smooth_shading=True,
                show_edges=True,
                edge_color="#10425a",
                line_width=1,
            )
            stage_label_font_size = self.viewer_font_size(gui_like_label_font_size(self))
            stage_size = np.array(
                [self.stage_size_x.value(), self.stage_size_y.value(), self.stage_size_z.value()],
                dtype=float,
            )
            stage_center_local = np.array(
                [self.stage_offset_x.value(), self.stage_offset_y.value(), self.stage_offset_z.value()],
                dtype=float,
            )
            label_clearance = max(min(self.scene_scale() * 0.012, 8.0), 2.5)
            annotation_z = stage_center_local[2]
            side_offsets_y = [
                stage_center_local[1] + stage_size[1] / 2.0 + label_clearance,
                stage_center_local[1] - stage_size[1] / 2.0 - label_clearance,
            ]
            side_offsets_x = [
                stage_center_local[0] + stage_size[0] / 2.0 + label_clearance,
                stage_center_local[0] - stage_size[0] / 2.0 - label_clearance,
            ]
            x_half = stage_size[0] / 2.0
            x_margin = max(stage_size[0] * 0.10, 12.0)
            arrow_start_gap = max(stage_size[0] * 0.06, 8.0)
            arrow_span = max(x_half - x_margin - arrow_start_gap, 5.0)
            y_half = stage_size[1] / 2.0
            y_margin = max(stage_size[1] * 0.10, 12.0)
            arrow_start_gap_y = max(stage_size[1] * 0.06, 8.0)
            arrow_span_y = max(y_half - y_margin - arrow_start_gap_y, 5.0)
            stage_x_label_points_local = []
            for side_y in side_offsets_y:
                stage_x_label_points_local.extend(
                    [
                        np.array([stage_center_local[0] - x_half + x_margin, side_y, annotation_z], dtype=float),
                        np.array([stage_center_local[0], side_y, annotation_z], dtype=float),
                        np.array([stage_center_local[0] + x_half - x_margin, side_y, annotation_z], dtype=float),
                    ]
                )
            stage_y_label_points_local = []
            for side_x in side_offsets_x:
                stage_y_label_points_local.extend(
                    [
                        np.array([side_x, stage_center_local[1] + y_half - y_margin, annotation_z], dtype=float),
                        np.array([side_x, stage_center_local[1], annotation_z], dtype=float),
                        np.array([side_x, stage_center_local[1] - y_half + y_margin, annotation_z], dtype=float),
                    ]
                )
            stage_axis_label_points_local = np.array(stage_x_label_points_local + stage_y_label_points_local, dtype=float)
            stage_axis_label_points_world = stage_local_to_world(
                stage_axis_label_points_local,
                stage_readout_local,
                omega_deg,
                pivot,
            )
            stage_axis_label_cloud = pv.PolyData(stage_axis_label_points_world)
            stage_axis_label_cloud["labels"] = np.array(
                ["-", "X", "+", "-", "X", "+", "+", "Y", "-", "+", "Y", "-"],
                dtype=object,
            )
            self.set_or_add_mesh(
                "stage_axis_labels_mesh",
                stage_axis_label_cloud,
                "stage_axis_label_points",
                render_points_as_spheres=False,
                point_size=1,
                opacity=0.0,
            )
            self.plotter.remove_actor("stage_axis_labels", render=False)
            self.plotter.add_point_labels(
                self.stage_axis_labels_mesh,
                "labels",
                font_size=stage_label_font_size,
                shape=None,
                text_color="#ffea00",
                always_visible=True,
                name="stage_axis_labels",
                render=False,
            )
            arrow_shaft_radius = 0.02
            arrow_tip_radius = 0.06
            arrow_tip_length = 0.18
            stage_x_arrow_pos = pv.Arrow(
                start=np.array([stage_center_local[0] + arrow_start_gap, side_offsets_y[0], annotation_z], dtype=float),
                direction=np.array([arrow_span, 0.0, 0.0], dtype=float),
                tip_length=arrow_tip_length,
                tip_radius=arrow_tip_radius,
                shaft_radius=arrow_shaft_radius,
            )
            stage_x_arrow_pos.points = stage_local_to_world(
                stage_x_arrow_pos.points,
                stage_readout_local,
                omega_deg,
                pivot,
            )
            self.set_or_add_mesh(
                "stage_x_arrow_pos_mesh",
                stage_x_arrow_pos,
                "stage_x_arrow_pos",
                color="#ffea00",
                smooth_shading=True,
            )
            stage_x_arrow_pos_secondary = pv.Arrow(
                start=np.array([stage_center_local[0] + arrow_start_gap, side_offsets_y[1], annotation_z], dtype=float),
                direction=np.array([arrow_span, 0.0, 0.0], dtype=float),
                tip_length=arrow_tip_length,
                tip_radius=arrow_tip_radius,
                shaft_radius=arrow_shaft_radius,
            )
            stage_x_arrow_pos_secondary.points = stage_local_to_world(
                stage_x_arrow_pos_secondary.points,
                stage_readout_local,
                omega_deg,
                pivot,
            )
            self.set_or_add_mesh(
                "stage_x_arrow_pos_secondary_mesh",
                stage_x_arrow_pos_secondary,
                "stage_x_arrow_pos_secondary",
                color="#ffea00",
                smooth_shading=True,
            )
            stage_x_arrow_neg = pv.Arrow(
                start=np.array([stage_center_local[0] - arrow_start_gap, side_offsets_y[0], annotation_z], dtype=float),
                direction=np.array([-arrow_span, 0.0, 0.0], dtype=float),
                tip_length=arrow_tip_length,
                tip_radius=arrow_tip_radius,
                shaft_radius=arrow_shaft_radius,
            )
            stage_x_arrow_neg.points = stage_local_to_world(
                stage_x_arrow_neg.points,
                stage_readout_local,
                omega_deg,
                pivot,
            )
            self.set_or_add_mesh(
                "stage_x_arrow_neg_mesh",
                stage_x_arrow_neg,
                "stage_x_arrow_neg",
                color="#ffea00",
                smooth_shading=True,
            )
            stage_x_arrow_neg_secondary = pv.Arrow(
                start=np.array([stage_center_local[0] - arrow_start_gap, side_offsets_y[1], annotation_z], dtype=float),
                direction=np.array([-arrow_span, 0.0, 0.0], dtype=float),
                tip_length=arrow_tip_length,
                tip_radius=arrow_tip_radius,
                shaft_radius=arrow_shaft_radius,
            )
            stage_x_arrow_neg_secondary.points = stage_local_to_world(
                stage_x_arrow_neg_secondary.points,
                stage_readout_local,
                omega_deg,
                pivot,
            )
            self.set_or_add_mesh(
                "stage_x_arrow_neg_secondary_mesh",
                stage_x_arrow_neg_secondary,
                "stage_x_arrow_neg_secondary",
                color="#ffea00",
                smooth_shading=True,
            )
            stage_y_arrow_pos = pv.Arrow(
                start=np.array([side_offsets_x[0], stage_center_local[1] + arrow_start_gap_y, annotation_z], dtype=float),
                direction=np.array([0.0, arrow_span_y, 0.0], dtype=float),
                tip_length=arrow_tip_length,
                tip_radius=arrow_tip_radius,
                shaft_radius=arrow_shaft_radius,
            )
            stage_y_arrow_pos.points = stage_local_to_world(
                stage_y_arrow_pos.points,
                stage_readout_local,
                omega_deg,
                pivot,
            )
            self.set_or_add_mesh(
                "stage_y_arrow_pos_mesh",
                stage_y_arrow_pos,
                "stage_y_arrow_pos",
                color="#ffea00",
                smooth_shading=True,
            )
            stage_y_arrow_pos_secondary = pv.Arrow(
                start=np.array([side_offsets_x[1], stage_center_local[1] + arrow_start_gap_y, annotation_z], dtype=float),
                direction=np.array([0.0, arrow_span_y, 0.0], dtype=float),
                tip_length=arrow_tip_length,
                tip_radius=arrow_tip_radius,
                shaft_radius=arrow_shaft_radius,
            )
            stage_y_arrow_pos_secondary.points = stage_local_to_world(
                stage_y_arrow_pos_secondary.points,
                stage_readout_local,
                omega_deg,
                pivot,
            )
            self.set_or_add_mesh(
                "stage_y_arrow_pos_secondary_mesh",
                stage_y_arrow_pos_secondary,
                "stage_y_arrow_pos_secondary",
                color="#ffea00",
                smooth_shading=True,
            )
            stage_y_arrow_neg = pv.Arrow(
                start=np.array([side_offsets_x[0], stage_center_local[1] - arrow_start_gap_y, annotation_z], dtype=float),
                direction=np.array([0.0, -arrow_span_y, 0.0], dtype=float),
                tip_length=arrow_tip_length,
                tip_radius=arrow_tip_radius,
                shaft_radius=arrow_shaft_radius,
            )
            stage_y_arrow_neg.points = stage_local_to_world(
                stage_y_arrow_neg.points,
                stage_readout_local,
                omega_deg,
                pivot,
            )
            self.set_or_add_mesh(
                "stage_y_arrow_neg_mesh",
                stage_y_arrow_neg,
                "stage_y_arrow_neg",
                color="#ffea00",
                smooth_shading=True,
            )
            stage_y_arrow_neg_secondary = pv.Arrow(
                start=np.array([side_offsets_x[1], stage_center_local[1] - arrow_start_gap_y, annotation_z], dtype=float),
                direction=np.array([0.0, -arrow_span_y, 0.0], dtype=float),
                tip_length=arrow_tip_length,
                tip_radius=arrow_tip_radius,
                shaft_radius=arrow_shaft_radius,
            )
            stage_y_arrow_neg_secondary.points = stage_local_to_world(
                stage_y_arrow_neg_secondary.points,
                stage_readout_local,
                omega_deg,
                pivot,
            )
            self.set_or_add_mesh(
                "stage_y_arrow_neg_secondary_mesh",
                stage_y_arrow_neg_secondary,
                "stage_y_arrow_neg_secondary",
                color="#ffea00",
                smooth_shading=True,
            )
        else:
            self.remove_stage_visual_actors()

        slit_center = np.array([self.slit_x.value(), self.slit_y.value(), self.slit_z.value()], dtype=float)
        slit_width = self.slit_width.value()
        slit_height = self.slit_height.value()
        beam_length = self.beam_length.value()
        show_beam = self.show_beam_checkbox is None or self.show_beam_checkbox.isChecked()
        show_gauge_volume = self.show_gauge_volume_checkbox is None or self.show_gauge_volume_checkbox.isChecked()
        slit_thickness = max(min(self.scene_scale() * 0.01, 8.0), 2.0)
        slit_plate = pv.Box(
            bounds=(
                slit_center[0] - slit_thickness / 2.0,
                slit_center[0] + slit_thickness / 2.0,
                slit_center[1] - slit_width / 2.0,
                slit_center[1] + slit_width / 2.0,
                slit_center[2] - slit_height / 2.0,
                slit_center[2] + slit_height / 2.0,
            )
        )
        if show_beam:
            self.set_or_add_mesh(
                "slit_mesh",
                slit_plate,
                "incident_slit",
                color="#68778d",
                opacity=0.85,
                smooth_shading=True,
                show_edges=True,
                edge_color="#384658",
                line_width=1,
            )
        elif self.slit_mesh is not None:
            self.plotter.remove_actor("incident_slit", render=False)
            self.slit_mesh = None
        beam_box = pv.Box(
            bounds=(
                slit_center[0],
                slit_center[0] + beam_length,
                slit_center[1] - slit_width / 2.0,
                slit_center[1] + slit_width / 2.0,
                slit_center[2] - slit_height / 2.0,
                slit_center[2] + slit_height / 2.0,
            )
        )
        gauge_volume_depth = float(self.collimator.currentText())
        gauge_volume_box = pv.Box(
            bounds=(
                pivot[0] - gauge_volume_depth / 2.0,
                pivot[0] + gauge_volume_depth / 2.0,
                pivot[1] - slit_width / 2.0,
                pivot[1] + slit_width / 2.0,
                pivot[2] - slit_height / 2.0,
                pivot[2] + slit_height / 2.0,
            )
        )
        if show_beam:
            self.set_or_add_mesh(
                "beam_mesh",
                beam_box,
                "beam_volume",
                color="#8ec5ff",
                opacity=0.20,
                smooth_shading=True,
                show_edges=True,
                edge_color="#1f77b4",
                line_width=1,
            )
            self.set_or_add_mesh(
                "beam_centerline_mesh",
                make_line_polydata(slit_center, slit_center + np.array([beam_length, 0.0, 0.0], dtype=float)),
                "beam_centerline",
                color="#1f77b4",
                line_width=3,
            )
        else:
            if self.beam_mesh is not None:
                self.plotter.remove_actor("beam_volume", render=False)
                self.beam_mesh = None
            if self.beam_centerline_mesh is not None:
                self.plotter.remove_actor("beam_centerline", render=False)
                self.beam_centerline_mesh = None
        if show_gauge_volume:
            self.set_or_add_mesh(
                "gauge_volume_mesh",
                gauge_volume_box,
                "gauge_volume",
                color="#ef4444",
                opacity=0.28,
                smooth_shading=True,
                show_edges=True,
                edge_color="#b91c1c",
                line_width=2,
            )
        else:
            if self.gauge_volume_mesh is not None:
                self.plotter.remove_actor("gauge_volume", render=False)
                self.gauge_volume_mesh = None
        direct_detector_center = np.array(DIRECT_DETECTOR_CENTER_WORLD, dtype=float)
        direct_detector_box = make_oriented_box(
            center=direct_detector_center,
            axis_u=np.array([0.0, 1.0, 0.0], dtype=float),
            axis_v=np.array([0.0, 0.0, 1.0], dtype=float),
            axis_w=np.array([1.0, 0.0, 0.0], dtype=float),
            size_u=self.detector_width_y.value(),
            size_v=self.detector_height_z.value(),
            size_w=DETECTOR_THICKNESS,
        )
        show_imaging_detector = (
            self.show_imaging_detector_checkbox is None or self.show_imaging_detector_checkbox.isChecked()
        )
        if show_imaging_detector:
            self.set_or_add_mesh(
                "detector_mesh",
                direct_detector_box,
                "imaging_detector_plane",
                color="#f4f4f5",
                opacity=0.92,
                smooth_shading=False,
                show_edges=True,
                edge_color="#d11f1f",
                line_width=2,
            )
            self.plotter.remove_actor("imaging_detector_label", render=False)
            self.plotter.add_point_labels(
                pv.PolyData(np.array([direct_detector_center], dtype=float)),
                ["Imaging detector"],
                font_size=self.viewer_font_size(max(gui_like_label_font_size(self) - 1, 14)),
                shape=None,
                text_color="#9a0000",
                always_visible=True,
                name="imaging_detector_label",
                render=False,
            )
        else:
            if self.detector_mesh is not None:
                self.plotter.remove_actor("imaging_detector_plane", render=False)
                self.detector_mesh = None
            self.plotter.remove_actor("imaging_detector_label", render=False)

        diffraction_bank_1_geometry = self.current_diffraction_bank_1_geometry()
        diffraction_bank_1_box = make_oriented_box(
            center=np.asarray(diffraction_bank_1_geometry["center"], dtype=float),
            axis_u=np.asarray(diffraction_bank_1_geometry["right"], dtype=float),
            axis_v=np.asarray(diffraction_bank_1_geometry["up"], dtype=float),
            axis_w=np.asarray(diffraction_bank_1_geometry["normal"], dtype=float),
            size_u=float(diffraction_bank_1_geometry["width"]),
            size_v=float(diffraction_bank_1_geometry["height"]),
            size_w=DETECTOR_THICKNESS,
        )
        diffraction_bank_2_geometry = self.current_diffraction_bank_2_geometry()
        diffraction_bank_2_box = make_oriented_box(
            center=np.asarray(diffraction_bank_2_geometry["center"], dtype=float),
            axis_u=np.asarray(diffraction_bank_2_geometry["right"], dtype=float),
            axis_v=np.asarray(diffraction_bank_2_geometry["up"], dtype=float),
            axis_w=np.asarray(diffraction_bank_2_geometry["normal"], dtype=float),
            size_u=float(diffraction_bank_2_geometry["width"]),
            size_v=float(diffraction_bank_2_geometry["height"]),
            size_w=DETECTOR_THICKNESS,
        )
        show_diffraction_detectors = (
            self.show_diffraction_detectors_checkbox is None or self.show_diffraction_detectors_checkbox.isChecked()
        )
        if show_diffraction_detectors:
            self.set_or_add_mesh(
                "diffraction_bank_1_detector_mesh",
                diffraction_bank_1_box,
                "diffraction_detector_bank_1_plane",
                color="#fff5e6",
                opacity=0.65,
                smooth_shading=False,
                show_edges=True,
                edge_color="#d17a00",
                line_width=2,
            )
            self.plotter.remove_actor("diffraction_detector_bank_1_label", render=False)
            self.plotter.add_point_labels(
                pv.PolyData(np.array([diffraction_bank_1_geometry["center"]], dtype=float)),
                ["Diffraction detector - bank 1"],
                font_size=self.viewer_font_size(max(gui_like_label_font_size(self) - 1, 14)),
                shape=None,
                text_color="#9a5400",
                always_visible=True,
                name="diffraction_detector_bank_1_label",
                render=False,
            )
            self.set_or_add_mesh(
                "diffraction_bank_2_detector_mesh",
                diffraction_bank_2_box,
                "diffraction_detector_bank_2_plane",
                color="#e8f3ff",
                opacity=0.65,
                smooth_shading=False,
                show_edges=True,
                edge_color="#225ea8",
                line_width=2,
            )
            self.plotter.remove_actor("diffraction_detector_bank_2_label", render=False)
            self.plotter.add_point_labels(
                pv.PolyData(np.array([diffraction_bank_2_geometry["center"]], dtype=float)),
                ["Diffraction detector - bank 2"],
                font_size=self.viewer_font_size(max(gui_like_label_font_size(self) - 1, 14)),
                shape=None,
                text_color="#12407a",
                always_visible=True,
                name="diffraction_detector_bank_2_label",
                render=False,
            )
        else:
            if self.diffraction_bank_1_detector_mesh is not None:
                self.plotter.remove_actor("diffraction_detector_bank_1_plane", render=False)
                self.diffraction_bank_1_detector_mesh = None
            if self.diffraction_bank_2_detector_mesh is not None:
                self.plotter.remove_actor("diffraction_detector_bank_2_plane", render=False)
                self.diffraction_bank_2_detector_mesh = None
            self.plotter.remove_actor("diffraction_detector_bank_1_label", render=False)
            self.plotter.remove_actor("diffraction_detector_bank_2_label", render=False)
        if self.detector_map_state is not None:
            remove_scalar_bar_if_present(self.plotter, "Imaging path")
            detector_map_mesh = self.detector_map_state["mesh"]
            detector_map_max = max(float(self.detector_map_state["max_length"]), 1e-9)
            self.set_or_add_mesh(
                "detector_map_mesh",
                detector_map_mesh,
                "detector_map",
                scalars="path_length",
                cmap="viridis",
                clim=[0.0, detector_map_max],
                opacity=0.95,
                smooth_shading=False,
                show_edges=False,
                scalar_bar_args={
                    "title": "Imaging path",
                    "color": "black",
                    "vertical": True,
                    "position_x": 0.90,
                    "position_y": 0.62,
                    "width": 0.06,
                    "height": 0.28,
                },
            )
        elif self.detector_map_mesh is not None:
            remove_scalar_bar_if_present(self.plotter, "Imaging path")
            self.plotter.remove_actor("detector_map", render=False)
            self.detector_map_mesh = None

        diffraction_states = [
            state
            for state in (self.diffraction_bank_1_map_state, self.diffraction_bank_2_map_state)
            if state is not None
        ]
        if diffraction_states:
            remove_scalar_bar_if_present(self.plotter, "Diffraction path")
            shared_diffraction_map_max = max(max(float(state["max_length"]), 1e-9) for state in diffraction_states)
            shared_diffraction_map_kwargs = {
                "scalars": "path_length",
                "cmap": "plasma",
                "clim": [0.0, shared_diffraction_map_max],
                "opacity": 0.92,
                "smooth_shading": False,
                "show_edges": False,
            }
            shared_diffraction_scalar_bar_args = {
                "title": "Diffraction path",
                "color": "black",
                "vertical": True,
                "position_x": 0.90,
                "position_y": 0.04,
                "width": 0.06,
                "height": 0.57,
            }
            for attr_name, actor_name in (
                ("diffraction_bank_1_map_mesh", "diffraction_detector_bank_1_map"),
                ("diffraction_bank_2_map_mesh", "diffraction_detector_bank_2_map"),
            ):
                if getattr(self, attr_name) is not None:
                    self.plotter.remove_actor(actor_name, render=False)
                    setattr(self, attr_name, None)
            if self.diffraction_bank_1_map_state is not None:
                self.set_or_add_mesh(
                    "diffraction_bank_1_map_mesh",
                    self.diffraction_bank_1_map_state["mesh"],
                    "diffraction_detector_bank_1_map",
                    scalar_bar_args=shared_diffraction_scalar_bar_args,
                    **shared_diffraction_map_kwargs,
                )
            if self.diffraction_bank_2_map_state is not None:
                bank_2_kwargs = dict(shared_diffraction_map_kwargs)
                if self.diffraction_bank_1_map_state is not None:
                    bank_2_kwargs["show_scalar_bar"] = False
                else:
                    bank_2_kwargs["scalar_bar_args"] = shared_diffraction_scalar_bar_args
                self.set_or_add_mesh(
                    "diffraction_bank_2_map_mesh",
                    self.diffraction_bank_2_map_state["mesh"],
                    "diffraction_detector_bank_2_map",
                    **bank_2_kwargs,
                )
        else:
            remove_scalar_bar_if_present(self.plotter, "Diffraction path")
            if self.diffraction_bank_1_map_mesh is not None:
                self.plotter.remove_actor("diffraction_detector_bank_1_map", render=False)
                self.diffraction_bank_1_map_mesh = None
            if self.diffraction_bank_2_map_mesh is not None:
                self.plotter.remove_actor("diffraction_detector_bank_2_map", render=False)
                self.diffraction_bank_2_map_mesh = None

        show_diffraction_vectors = (
            self.show_diffraction_vectors_checkbox is None or self.show_diffraction_vectors_checkbox.isChecked()
        )
        if show_diffraction_vectors:
            diffraction_vector_bank_1 = pv.Arrow(
                start=pivot,
                direction=np.array([-100.0, 100.0, 0.0], dtype=float),
                tip_length=0.0225,
                tip_radius=0.011,
                shaft_radius=0.003,
                scale="auto",
            )
            self.set_or_add_mesh(
                "diffraction_vector_bank_1_mesh",
                diffraction_vector_bank_1,
                "diffraction_vector_bank_1",
                color="#d17a00",
                smooth_shading=True,
            )
            diffraction_vector_bank_2 = pv.Arrow(
                start=pivot,
                direction=np.array([-100.0, -100.0, 0.0], dtype=float),
                tip_length=0.0225,
                tip_radius=0.011,
                shaft_radius=0.003,
                scale="auto",
            )
            self.set_or_add_mesh(
                "diffraction_vector_bank_2_mesh",
                diffraction_vector_bank_2,
                "diffraction_vector_bank_2",
                color="#225ea8",
                smooth_shading=True,
            )
        else:
            if self.diffraction_vector_bank_1_mesh is not None:
                self.plotter.remove_actor("diffraction_vector_bank_1", render=False)
                self.diffraction_vector_bank_1_mesh = None
            if self.diffraction_vector_bank_2_mesh is not None:
                self.plotter.remove_actor("diffraction_vector_bank_2", render=False)
                self.diffraction_vector_bank_2_mesh = None

        theodolite = np.array([self.theodolite_x.value(), self.theodolite_y.value(), self.theodolite_z.value()])
        show_theodolite_sight_line = (
            self.show_theodolite_sight_line_checkbox is None
            or self.show_theodolite_sight_line_checkbox.isChecked()
        )
        if show_theodolite_sight_line:
            self.set_or_add_mesh(
                "sight_line_mesh",
                make_line_polydata(theodolite, pivot),
                "sight_line",
                color="#1f77b4",
                line_width=4,
            )
        elif self.sight_line_mesh is not None:
            self.plotter.remove_actor("sight_line", render=False)
            self.sight_line_mesh = None
        if self.camera_preset == "theodolite":
            sight_direction = normalized(pivot - theodolite)
            up_reference = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(sight_direction, up_reference))) > 0.97:
                up_reference = np.array([0.0, 1.0, 0.0], dtype=float)
            right = normalized(np.cross(sight_direction, up_reference))
            up = normalized(np.cross(right, sight_direction))
            crosshair_size = max(min(self.scene_scale() * 0.035, 16.0), 6.0)
            self.set_or_add_mesh(
                "crosshair_h_mesh",
                make_line_polydata(pivot - right * crosshair_size, pivot + right * crosshair_size),
                "crosshair_h",
                color="#111111",
                line_width=3,
            )
            self.set_or_add_mesh(
                "crosshair_v_mesh",
                make_line_polydata(pivot - up * crosshair_size, pivot + up * crosshair_size),
                "crosshair_v",
                color="#111111",
                line_width=3,
            )
        else:
            self.set_or_add_mesh(
                "crosshair_h_mesh",
                make_line_polydata(pivot, pivot),
                "crosshair_h",
                color="#111111",
                line_width=3,
            )
            self.set_or_add_mesh(
                "crosshair_v_mesh",
                make_line_polydata(pivot, pivot),
                "crosshair_v",
                color="#111111",
                line_width=3,
            )

        measurement_rows = self.residual_rows if self.residual_rows else self.infer_measurement_rows_without_fit()
        if measurement_rows and self.show_feature_points_checkbox.isChecked():
            label_font_size = self.viewer_font_size(gui_like_label_font_size(self))
            stage_points_local = np.array([row[1] for row in measurement_rows], dtype=float)
            world_points = stage_local_to_world(stage_points_local, stage_readout_local, omega_deg, pivot)
            point_cloud = pv.PolyData(world_points)
            point_cloud["labels"] = np.array([row[0].label for row in measurement_rows], dtype=object)
            self.set_or_add_mesh(
                "measurement_points_mesh",
                point_cloud,
                "measurement_points",
                render_points_as_spheres=True,
                point_size=12,
                color="#f59e0b",
            )
            self.plotter.remove_actor("measurement_labels", render=False)
            self.plotter.add_point_labels(
                self.measurement_points_mesh,
                "labels",
                font_size=label_font_size,
                shape=None,
                point_color="#f59e0b",
                text_color="black",
                always_visible=True,
                name="measurement_labels",
                render=False,
            )

        else:
            empty_points = pv.PolyData(np.empty((0, 3), dtype=float))
            empty_points["labels"] = np.array([], dtype=object)
            self.set_or_add_mesh(
                "measurement_points_mesh",
                empty_points,
                "measurement_points",
                render_points_as_spheres=True,
                point_size=12,
                color="#f59e0b",
            )
            self.plotter.remove_actor("measurement_labels", render=False)

        prediction_rows = self.prediction_rows_from_table()
        if prediction_rows and self.show_prediction_points_checkbox.isChecked():
            label_font_size = self.viewer_font_size(gui_like_label_font_size(self))
            rotation, translation, _source = self.current_model_to_stage_transform()
            prediction_model_points = np.array([row[1] for row in prediction_rows], dtype=float)
            prediction_stage_points = (rotation @ prediction_model_points.T).T + translation
            prediction_world_points = stage_local_to_world(
                prediction_stage_points,
                stage_readout_local,
                omega_deg,
                pivot,
            )
            prediction_cloud = pv.PolyData(prediction_world_points)
            prediction_cloud["labels"] = np.array([row[0] for row in prediction_rows], dtype=object)
            self.set_or_add_mesh(
                "prediction_points_mesh",
                prediction_cloud,
                "prediction_points",
                render_points_as_spheres=True,
                point_size=11,
                color="#14b8a6",
            )
            self.plotter.remove_actor("prediction_labels", render=False)
            self.plotter.add_point_labels(
                self.prediction_points_mesh,
                "labels",
                font_size=label_font_size,
                shape=None,
                point_color="#14b8a6",
                text_color="black",
                always_visible=True,
                name="prediction_labels",
                render=False,
            )
        else:
            self.set_or_add_mesh(
                "prediction_points_mesh",
                pv.PolyData(np.empty((0, 3), dtype=float)),
                "prediction_points",
                render_points_as_spheres=True,
                point_size=11,
                color="#14b8a6",
            )
            self.plotter.remove_actor("prediction_labels", render=False)

        selected_point_for_line = None
        if self.show_feature_points_checkbox.isChecked():
            selected_index = self.selected_measurement_index()
            if measurement_rows and selected_index is not None and selected_index < len(measurement_rows):
                selected_stage_point = np.array(measurement_rows[selected_index][1], dtype=float)
                selected_point_for_line = stage_local_to_world(
                    np.array([selected_stage_point], dtype=float),
                    stage_readout_local,
                    omega_deg,
                    pivot,
                )[0]
        if selected_point_for_line is None and self.show_prediction_points_checkbox.isChecked():
            selected_prediction = self.selected_prediction_row()
            if selected_prediction is not None:
                rotation, translation, _source = self.current_model_to_stage_transform()
                _label, prediction_model_point = selected_prediction
                prediction_stage_point = (rotation @ prediction_model_point) + translation
                selected_point_for_line = stage_local_to_world(
                    np.array([prediction_stage_point], dtype=float),
                    stage_readout_local,
                    omega_deg,
                    pivot,
                )[0]
        if selected_point_for_line is not None:
            self.set_or_add_mesh(
                "selected_line_mesh",
                make_line_polydata(selected_point_for_line, pivot),
                "selected_line",
                color="#ef4444",
                line_width=2,
            )
        else:
            self.set_or_add_mesh(
                "selected_line_mesh",
                make_line_polydata(pivot, pivot),
                "selected_line",
                color="#ef4444",
                line_width=2,
            )

        model_world = self.build_current_model_world_mesh()
        if model_world is not None:
            rotation_model_to_stage, translation_model_to_stage, placement_source = self.current_model_to_stage_transform()
            if placement_source == "as-is":
                color = "#89b4fa"
                opacity = 0.22
            elif placement_source == "manual":
                color = "#f2b36f"
                opacity = 0.82
            else:
                color = "#8fd14f"
                opacity = 0.78
            self.set_or_add_mesh(
                "model_world_mesh",
                model_world,
                "sample_model",
                color=color,
                opacity=opacity,
                smooth_shading=True,
                show_edges=False,
            )
            model_origin_world = stage_local_to_world(
                np.array([translation_model_to_stage], dtype=float),
                stage_readout_local,
                omega_deg,
                pivot,
            )[0]
            model_rotation_world = np.array(rotation_z_deg(omega_deg), dtype=float) @ np.asarray(
                rotation_model_to_stage,
                dtype=float,
            )
            bounds = np.array(model_world.bounds, dtype=float)
            model_min = np.array([bounds[0], bounds[2], bounds[4]], dtype=float)
            model_max = np.array([bounds[1], bounds[3], bounds[5]], dtype=float)
            model_center = 0.5 * (model_min + model_max)
            model_diag = float(np.linalg.norm(model_max - model_min))
            triad_length = max(model_diag * 0.07, self.scene_scale() * 0.018, 4.5)
            anchor_offset = model_origin_world - model_center
            if np.linalg.norm(anchor_offset) < 1e-9:
                anchor_offset = np.array([-1.0, 1.0, 1.0], dtype=float)
            anchor_world = model_origin_world + normalized(anchor_offset) * max(triad_length * 0.8, model_diag * 0.05, 5.0)
            show_sample_triad = (
                self.show_sample_triad_checkbox is None or self.show_sample_triad_checkbox.isChecked()
            )
            self.remove_legacy_model_axis_actors()
            if show_sample_triad:
                if self.model_axes_actor is None:
                    self.model_axes_actor = self.plotter.add_axes_at_origin(
                        xlabel="X",
                        ylabel="Y",
                        zlabel="Z",
                        line_width=2,
                    )
                    self.model_axes_actor.SetPickable(False)
                    for caption_actor in (
                        self.model_axes_actor.GetXAxisCaptionActor2D(),
                        self.model_axes_actor.GetYAxisCaptionActor2D(),
                        self.model_axes_actor.GetZAxisCaptionActor2D(),
                    ):
                        caption_actor.SetWidth(0.08)
                        caption_actor.SetHeight(0.03)
                        text_property = caption_actor.GetCaptionTextProperty()
                        text_property.BoldOff()
                        text_property.ItalicOff()
                        text_property.ShadowOff()
                for caption_actor in (
                    self.model_axes_actor.GetXAxisCaptionActor2D(),
                    self.model_axes_actor.GetYAxisCaptionActor2D(),
                    self.model_axes_actor.GetZAxisCaptionActor2D(),
                ):
                    text_property = caption_actor.GetCaptionTextProperty()
                    text_property.SetFontSize(self.viewer_font_size(12))
                self.model_axes_actor.SetTotalLength(triad_length, triad_length, triad_length)
                transform = vtkTransform()
                transform.PostMultiply()
                transform.Translate(*anchor_world.tolist())
                rotation_4x4 = np.eye(4, dtype=float)
                rotation_4x4[:3, :3] = model_rotation_world
                transform.Concatenate(rotation_4x4.ravel())
                self.model_axes_actor.SetUserTransform(transform)
            elif self.model_axes_actor is not None:
                self.plotter.remove_actor(self.model_axes_actor, render=False)
                self.model_axes_actor = None
        elif self.model_world_mesh is not None:
            self.plotter.remove_actor("sample_model", render=False)
            self.model_world_mesh = None
            self.remove_legacy_model_axis_actors()
            if self.model_axes_actor is not None:
                self.plotter.remove_actor(self.model_axes_actor, render=False)
                self.model_axes_actor = None

        slit_origin = np.array([self.slit_x.value(), self.slit_y.value(), self.slit_z.value()], dtype=float)
        beam_direction = np.array([1.0, 0.0, 0.0], dtype=float)
        beam_trace_distance = max(
            self.beam_length.value(),
            700.0 - slit_origin[0] + 50.0,
            self.scene_scale() * 2.5,
        )
        beam_inside_segments: List[Tuple[np.ndarray, np.ndarray]] = []
        if self.model_world_mesh is not None and self.model_world_mesh.n_points > 0:
            _beam_path_length, beam_inside_segments = compute_line_beam_path(
                self.model_world_mesh,
                slit_origin,
                beam_direction,
                beam_trace_distance,
            )
        if show_beam:
            self.set_or_add_mesh(
                "beam_inside_mesh",
                make_multi_line_polydata(beam_inside_segments),
                "beam_inside",
                color="#dc2626",
                line_width=6,
            )
        elif self.beam_inside_mesh is not None:
            self.plotter.remove_actor("beam_inside", render=False)
            self.beam_inside_mesh = None

        if camera_state is not None:
            restore_camera_state(self.plotter, camera_state)
        else:
            self.apply_camera_preset(reset_camera=True)
        self.plotter.render()

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 7000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GUI for fitting and visualizing sample placement on the stage.")
    parser.add_argument("--csv", type=Path, help="Optional measurement CSV to load on startup.")
    parser.add_argument("--mesh", type=Path, help="Optional sample mesh to load on startup.")
    parser.add_argument("--smoke-test", action="store_true", help="Initialize the GUI without entering the event loop.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    apply_application_ui_font(app, DEFAULT_UI_FONT_POINT_SIZE)
    window = MainWindow(enable_3d=not args.smoke_test)

    if args.csv:
        window.load_csv(args.csv)
    if args.mesh:
        window.load_mesh(args.mesh)

    if args.smoke_test:
        window.update_scene(reset_camera=True)
        print("GUI smoke test completed")
        window.close()
        app.quit()
        return 0

    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
