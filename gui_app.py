#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import trimesh
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QStackedWidget,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
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
]

DIRECT_DETECTOR_CENTER_WORLD = np.array([700.0, 0.0, 0.0], dtype=float)
DIFFRACTION_BANK_1_CENTER_WORLD = np.array([0.0, 1000.0, 0.0], dtype=float)
DIFFRACTION_BANK_2_CENTER_WORLD = np.array([0.0, -1000.0, 0.0], dtype=float)
ANGLED_DETECTOR_HORIZONTAL_HALF_ANGLE_DEG = 14.0
ANGLED_DETECTOR_VERTICAL_HALF_ANGLE_DEG = 21.0
DIFFRACTION_ANGLE_INTERVAL_DEG = 1.0
DETECTOR_THICKNESS = 10.0


def format_decimal(value: float) -> str:
    return f"{value:.3f}"


def make_spin_box(value: float, minimum: float, maximum: float, step: float = 1.0) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(3)
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setSingleStep(step)
    box.setKeyboardTracking(False)
    return box


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
    for y_index, y_value in enumerate(y_coords):
        for z_index, z_value in enumerate(z_coords):
            origin = np.array([slit_center[0], y_value, z_value], dtype=float)
            path_length, _segments = compute_line_beam_path(mesh, origin, beam_direction, max_distance)
            path_map[y_index, z_index] = path_length
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


def point_inside_closed_mesh(mesh: pv.PolyData, point: np.ndarray, tolerance: float = 1e-6) -> bool:
    if mesh is None or mesh.n_points == 0:
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
) -> bool:
    if mesh is None or mesh.n_points == 0:
        return False
    point = np.asarray(point, dtype=float)
    if point_inside_closed_mesh(mesh, point, tolerance=tolerance):
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
        )
        if path_length > epsilon * 0.15:
            return True
    return False


def compute_segment_path_length(
    mesh: pv.PolyData,
    origin: np.ndarray,
    end: np.ndarray,
    tolerance: float = 1e-6,
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

    epsilon = min(max(max_distance * 1e-6, 1e-4), max_distance * 0.25)
    start_inside = point_inside_closed_mesh(mesh, origin + unit_direction * epsilon, tolerance=tolerance)
    if not start_inside:
        start_inside = point_inside_closed_mesh(mesh, origin, tolerance=tolerance)

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    total_length = 0.0
    inside = start_inside
    current_distance = 0.0 if start_inside else None
    for distance in merged_distances:
        if inside:
            start_distance = 0.0 if current_distance is None else current_distance
            if distance > start_distance + tolerance:
                entry_point = origin + unit_direction * start_distance
                exit_point = origin + unit_direction * distance
                segments.append((entry_point, exit_point))
                total_length += distance - start_distance
            inside = False
            current_distance = None
        else:
            current_distance = distance
            inside = True

    if inside and current_distance is not None and max_distance > current_distance + tolerance:
        entry_point = origin + unit_direction * current_distance
        exit_point = end
        segments.append((entry_point, exit_point))
        total_length += max_distance - current_distance

    return total_length, segments


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


def load_mesh_as_polydata(path: Path) -> pv.PolyData:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = combine_scene_meshes(tuple(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported model type returned for {path.name}.")

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"{path.name} did not contain triangle faces.")
    faces_with_size = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(vertices, faces_with_size).clean()


class PointPickerDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick Model Points")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.resize(1280, 920)

        self.model_mesh: Optional[pv.PolyData] = None
        self.mesh_path: Optional[Path] = None
        self.plane_mesh = None
        self.slice_mesh = None
        self.clipped_mesh = None
        self.picked_points_mesh = None
        self.picked_points: List[Tuple[str, np.ndarray]] = []

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
        self.plane_mode_combo = QComboBox()
        self.plane_mode_combo.addItems(["Axis aligned", "Point + normal", "Three points"])
        self.plane_mode_combo.currentTextChanged.connect(self.on_plane_mode_changed)
        slice_form.addRow("Plane mode", self.plane_mode_combo)

        self.plane_mode_stack = QStackedWidget()
        axis_mode_widget = QWidget()
        axis_mode_layout = QFormLayout(axis_mode_widget)
        self.plane_axis_combo = QComboBox()
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
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.hide)
        points_button_row.addWidget(remove_button)
        points_button_row.addWidget(undo_button)
        points_button_row.addWidget(clear_button)
        points_button_row.addWidget(save_button)
        points_button_row.addStretch(1)
        points_button_row.addWidget(close_button)
        points_layout.addLayout(points_button_row)

        self.points_table = QTableWidget(0, 4)
        self.points_table.setHorizontalHeaderLabels(["Label", "Model X", "Model Y", "Model Z"])
        self.points_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.points_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.points_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        points_layout.addWidget(self.points_table)
        controls_layout.addWidget(points_group, stretch=1)

        self.status_label = QLabel("Load a mesh from the main window, then use the plane or direct coordinates to add model points.")
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
        self.plotter = CursorZoomQtInteractor(self)
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.enable_parallel_projection()
        self.plotter.interactor.installEventFilter(self)
        self.plotter.enable_surface_point_picking(
            callback=self.on_plane_picked,
            show_message=False,
            show_point=False,
            left_clicking=True,
            clear_on_no_selection=False,
        )
        viewer_layout.addWidget(self.plotter.interactor)
        main_splitter.addWidget(viewer_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([560, 720])

    def closeEvent(self, event) -> None:
        self.hide()
        event.ignore()

    def set_mesh(self, mesh: pv.PolyData, mesh_path: Optional[Path] = None) -> None:
        previous_path = None if self.mesh_path is None else str(self.mesh_path)
        next_path = None if mesh_path is None else str(mesh_path)
        if previous_path != next_path:
            self.picked_points = []
            self.refresh_points_table()
        self.mesh_path = mesh_path
        self.model_mesh = mesh.copy(deep=True) if mesh is not None else None
        self.configure_coordinate_ranges()
        self.refresh_scene(reset_camera=True)
        self.view_plane_normal()

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
        if watched is self.plotter.interactor:
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
        if self.model_mesh is None or self.model_mesh.n_points == 0:
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

        self.enable_3d = enable_3d
        self.mesh_path: Optional[Path] = None
        self.model_mesh: Optional[pv.PolyData] = None
        self.fit_transform = None
        self.residual_rows = []
        self.top_splitter = None
        self.bottom_splitter = None
        self.lower_splitter = None
        self.tables_splitter = None
        self.controls_scroll = None
        self.instrument_setup_dialog = None
        self.point_picker_dialog = None
        self._initial_sizes_applied = False
        self.camera_preset = "iso"
        self.parallel_projection_enabled = False
        self.viewer_font_size_offset = 8
        self.view_buttons = {}
        self.view_button_group = None
        self.scene_initialized = False
        self.stage_world_mesh = None
        self.beam_mesh = None
        self.beam_centerline_mesh = None
        self.slit_mesh = None
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
        self.show_stage_checkbox = None
        self.show_beam_checkbox = None
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

        self.top_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.bottom_splitter)
        self.bottom_splitter.addWidget(self.top_splitter)

        self.viewer_font_increase_shortcut = QShortcut(QKeySequence("Shift+>"), self)
        self.viewer_font_increase_shortcut.setContext(Qt.WindowShortcut)
        self.viewer_font_increase_shortcut.activated.connect(lambda: self.adjust_viewer_font_size(1))
        self.viewer_font_decrease_shortcut = QShortcut(QKeySequence("Shift+<"), self)
        self.viewer_font_decrease_shortcut.setContext(Qt.WindowShortcut)
        self.viewer_font_decrease_shortcut.activated.connect(lambda: self.adjust_viewer_font_size(-1))

        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(self._build_files_group())
        controls_layout.addWidget(self._build_setup_group())
        controls_layout.addWidget(self._build_manual_placement_group())
        controls_layout.addWidget(self._build_pose_group())
        controls_layout.addWidget(self._build_results_group())
        controls_layout.addStretch(1)
        controls_container.setMinimumWidth(360)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setWidget(controls_container)
        self.top_splitter.addWidget(self.controls_scroll)

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
            self.top_splitter.setStretchFactor(1, 1)
        else:
            self.plotter = None
            placeholder = QLabel("3D viewport disabled for smoke test mode.")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet(
                "background-color: #0d1117; color: #d7d7d7; border: 1px solid #2f3a45; padding: 24px;"
            )
            view_layout.addWidget(placeholder)
            self.top_splitter.setStretchFactor(1, 1)
        self.top_splitter.addWidget(view_panel)

        lower_panel = QWidget()
        lower_layout = QVBoxLayout(lower_panel)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(8)

        self.tables_splitter = QSplitter(Qt.Horizontal)
        self.tables_splitter.setChildrenCollapsible(False)
        self.tables_splitter.addWidget(self._build_measurement_section())
        self.tables_splitter.addWidget(self._build_prediction_section())

        self.lower_splitter = QSplitter(Qt.Vertical)
        self.lower_splitter.setChildrenCollapsible(False)
        self.lower_splitter.addWidget(self.tables_splitter)
        self.report_box = self._build_report_box()
        self.lower_splitter.addWidget(self.report_box)
        self.lower_splitter.setStretchFactor(0, 4)
        self.lower_splitter.setStretchFactor(1, 1)
        lower_layout.addWidget(self.lower_splitter)
        self.bottom_splitter.addWidget(lower_panel)
        self.bottom_splitter.setStretchFactor(0, 3)
        self.bottom_splitter.setStretchFactor(1, 2)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Load a mesh, then use Fit placement or Manual Sample Placement.")

    def _apply_initial_splitter_sizes(self) -> None:
        if self.top_splitter is not None:
            self.top_splitter.setChildrenCollapsible(False)
            self.top_splitter.setSizes([430, 1130])
        if self.bottom_splitter is not None:
            self.bottom_splitter.setChildrenCollapsible(False)
            self.bottom_splitter.setSizes([640, 260])
        if self.tables_splitter is not None:
            self.tables_splitter.setSizes([820, 740])
        if self.lower_splitter is not None:
            self.lower_splitter.setSizes([320, 110])

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._initial_sizes_applied:
            self._apply_initial_splitter_sizes()
            self._initial_sizes_applied = True
        if self.controls_scroll is not None:
            self.controls_scroll.verticalScrollBar().setValue(0)

    def _build_files_group(self) -> QGroupBox:
        group = QGroupBox("Files")
        layout = QVBoxLayout(group)

        self.mesh_path_label = QLabel("No mesh loaded")
        self.mesh_path_label.setWordWrap(True)
        self.csv_path_label = QLabel("Measurements can be loaded from CSV or edited directly.")
        self.csv_path_label.setWordWrap(True)

        row1 = QHBoxLayout()
        load_mesh_button = QPushButton("Load STL/mesh")
        load_mesh_button.clicked.connect(self.load_mesh_dialog)
        clear_mesh_button = QPushButton("Clear mesh")
        clear_mesh_button.clicked.connect(self.clear_mesh)
        row1.addWidget(load_mesh_button)
        row1.addWidget(clear_mesh_button)

        row2 = QHBoxLayout()
        fit_button = QPushButton("Fit placement")
        fit_button.clicked.connect(self.fit_placement)
        export_button = QPushButton("Export fit JSON")
        export_button.clicked.connect(self.export_json_dialog)
        row2.addWidget(fit_button)
        row2.addWidget(export_button)

        row3 = QHBoxLayout()
        pick_point_button = QPushButton("Pick point")
        pick_point_button.clicked.connect(self.open_point_picker_dialog)
        row3.addWidget(pick_point_button)
        row3.addStretch(1)

        layout.addWidget(self.mesh_path_label)
        layout.addWidget(self.csv_path_label)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        return group

    def _build_setup_group(self) -> QGroupBox:
        group = QGroupBox("Setup Geometry")
        layout = QFormLayout(group)

        self.pivot_x = make_spin_box(0.0, -100000.0, 100000.0)
        self.pivot_y = make_spin_box(0.0, -100000.0, 100000.0)
        self.pivot_z = make_spin_box(0.0, -100000.0, 100000.0)
        self.theodolite_x = make_spin_box(-250.0, -100000.0, 100000.0)
        self.theodolite_y = make_spin_box(-250.0, -100000.0, 100000.0)
        self.theodolite_z = make_spin_box(0.0, -100000.0, 100000.0)
        self.slit_x = make_spin_box(-300.0, -100000.0, 100000.0)
        self.slit_y = make_spin_box(0.0, -100000.0, 100000.0)
        self.slit_z = make_spin_box(0.0, -100000.0, 100000.0)
        self.slit_width = make_spin_box(20.0, 0.001, 100000.0)
        self.slit_height = make_spin_box(20.0, 0.001, 100000.0)
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
        for widget in (
            self.detector_map_pixel_size_y,
            self.detector_map_pixel_size_z,
        ):
            widget.valueChanged.connect(self.on_detector_map_parameter_changed)

        instrument_setup_button = QPushButton("Instrument setup...")
        instrument_setup_button.clicked.connect(self.open_instrument_setup_dialog)
        layout.addRow(instrument_setup_button)
        layout.addRow(self._separator())
        layout.addRow("Slit width", self.slit_width)
        layout.addRow("Slit height", self.slit_height)
        layout.addRow("Beam length", self.beam_length)
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
        layout = QFormLayout(group)

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
        selected_pose_button = QPushButton("Use selected row pose")
        selected_pose_button.clicked.connect(self.use_selected_row_pose)
        reset_pose_button = QPushButton("Reset pose")
        reset_pose_button.clicked.connect(self.reset_pose)
        button_row.addWidget(selected_pose_button)
        button_row.addWidget(reset_pose_button)
        layout.addRow(button_row)
        return group

    def _build_manual_placement_group(self) -> QGroupBox:
        group = QGroupBox("Manual Sample Placement")
        layout = QFormLayout(group)

        self.manual_placement_enabled_checkbox = QCheckBox("Use manual placement")
        self.manual_placement_enabled_checkbox.toggled.connect(self.on_manual_placement_mode_toggled)

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
            self.manual_rx,
            self.manual_ry,
            self.manual_rz,
        ):
            widget.valueChanged.connect(self.on_manual_placement_changed)

        reset_button = QPushButton("Reset manual")
        reset_button.clicked.connect(self.reset_manual_placement)
        load_fit_button = QPushButton("Load fit into manual")
        load_fit_button.clicked.connect(self.load_fit_into_manual)
        button_row = QHBoxLayout()
        button_row.addWidget(reset_button)
        button_row.addWidget(load_fit_button)

        layout.addRow(self.manual_placement_enabled_checkbox)
        layout.addRow("Model->Stage X", self.manual_tx)
        layout.addRow("Model->Stage Y", self.manual_ty)
        layout.addRow("Model->Stage Z", self.manual_tz)
        layout.addRow("Rot X", self.manual_rx)
        layout.addRow("Rot Y", self.manual_ry)
        layout.addRow("Rot Z", self.manual_rz)
        layout.addRow(button_row)
        return group

    def _build_results_group(self) -> QGroupBox:
        group = QGroupBox("Fit Summary")
        layout = QFormLayout(group)

        self.summary_status = QLabel("No fit computed")
        self.summary_translation = QLabel("-")
        self.summary_euler = QLabel("-")
        self.summary_rms = QLabel("-")
        self.summary_max = QLabel("-")
        self.summary_beam_path = QLabel("-")
        self.summary_detector_map = QLabel("-")
        self.summary_diffraction_path = QLabel("-")
        self.summary_diffraction_map = QLabel("-")

        for label in (
            self.summary_status,
            self.summary_translation,
            self.summary_euler,
            self.summary_rms,
            self.summary_max,
            self.summary_beam_path,
            self.summary_detector_map,
            self.summary_diffraction_path,
            self.summary_diffraction_map,
        ):
            label.setWordWrap(True)

        clear_placement_button = QPushButton("Clear placement")
        clear_placement_button.clicked.connect(self.clear_placement)
        compute_detector_map_button = QPushButton("Compute imaging map")
        compute_detector_map_button.clicked.connect(self.compute_detector_map)
        compute_diffraction_map_button = QPushButton("Compute diffraction path")
        compute_diffraction_map_button.clicked.connect(self.compute_diffraction_map)
        export_detector_map_button = QPushButton("Export detector map")
        export_detector_map_button.clicked.connect(self.export_detector_map_dialog)

        layout.addRow("Status", self.summary_status)
        layout.addRow("Translation", self.summary_translation)
        layout.addRow("Euler ZYX", self.summary_euler)
        layout.addRow("RMS error", self.summary_rms)
        layout.addRow("Max error", self.summary_max)
        layout.addRow("Imaging path", self.summary_beam_path)
        layout.addRow("Imaging map", self.summary_detector_map)
        layout.addRow("Diffraction path", self.summary_diffraction_path)
        layout.addRow("Diffraction map", self.summary_diffraction_map)
        layout.addRow(clear_placement_button)
        layout.addRow(compute_detector_map_button)
        layout.addRow(compute_diffraction_map_button)
        layout.addRow(export_detector_map_button)
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
            """
        )
        self.view_button_group = QButtonGroup(toolbar)
        self.view_button_group.setExclusive(True)

        buttons = [
            ("Iso", "iso"),
            ("+X", "+x"),
            ("-X", "-x"),
            ("+Y", "+y"),
            ("-Y", "-y"),
            ("+Z", "+z"),
            ("-Z", "-z"),
            ("Theodolite", "theodolite"),
        ]
        for label, preset in buttons:
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, view=preset: self.set_camera_preset(view))
            self.view_button_group.addButton(button)
            self.view_buttons[preset] = button
            layout.addWidget(button)

        self.parallel_projection_checkbox = QCheckBox("Parallel")
        self.parallel_projection_checkbox.toggled.connect(self.on_projection_toggled)
        layout.addWidget(self.parallel_projection_checkbox)
        self.show_stage_checkbox = QCheckBox("Stage")
        self.show_stage_checkbox.setChecked(True)
        self.show_stage_checkbox.toggled.connect(self.on_overlay_visibility_changed)
        layout.addWidget(self.show_stage_checkbox)
        self.show_beam_checkbox = QCheckBox("Beam")
        self.show_beam_checkbox.setChecked(True)
        self.show_beam_checkbox.toggled.connect(self.on_overlay_visibility_changed)
        layout.addWidget(self.show_beam_checkbox)
        self.show_feature_points_checkbox = QCheckBox("Features")
        self.show_feature_points_checkbox.setChecked(True)
        self.show_feature_points_checkbox.toggled.connect(self.on_overlay_visibility_changed)
        layout.addWidget(self.show_feature_points_checkbox)
        self.show_prediction_points_checkbox = QCheckBox("Predicted")
        self.show_prediction_points_checkbox.setChecked(True)
        self.show_prediction_points_checkbox.toggled.connect(self.on_overlay_visibility_changed)
        layout.addWidget(self.show_prediction_points_checkbox)
        self.show_sample_triad_checkbox = QCheckBox("Sample triad")
        self.show_sample_triad_checkbox.setChecked(True)
        self.show_sample_triad_checkbox.toggled.connect(self.on_overlay_visibility_changed)
        layout.addWidget(self.show_sample_triad_checkbox)
        self.show_theodolite_sight_line_checkbox = QCheckBox("Sight line")
        self.show_theodolite_sight_line_checkbox.setChecked(True)
        self.show_theodolite_sight_line_checkbox.toggled.connect(self.on_overlay_visibility_changed)
        layout.addWidget(self.show_theodolite_sight_line_checkbox)
        self.show_diffraction_vectors_checkbox = QCheckBox("Diffraction vectors")
        self.show_diffraction_vectors_checkbox.setChecked(True)
        self.show_diffraction_vectors_checkbox.toggled.connect(self.on_overlay_visibility_changed)
        layout.addWidget(self.show_diffraction_vectors_checkbox)
        layout.addStretch(1)
        self.sync_view_button_states()
        return toolbar

    def _build_measurement_toolbar(self) -> QWidget:
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)

        add_row_button = QPushButton("Add row")
        add_row_button.clicked.connect(lambda: self.add_measurement_row())
        load_csv_button = QPushButton("Load CSV")
        load_csv_button.clicked.connect(self.load_csv_dialog)
        save_csv_button = QPushButton("Save CSV")
        save_csv_button.clicked.connect(self.save_csv_dialog)
        remove_row_button = QPushButton("Remove row")
        remove_row_button.clicked.connect(self.remove_selected_rows)
        move_to_pivot_button = QPushButton("Move point to pivot")
        move_to_pivot_button.clicked.connect(self.move_selected_point_to_pivot)
        self.auto_move_to_pivot_checkbox = QCheckBox("Auto move on select")
        self.auto_move_to_pivot_checkbox.setChecked(True)

        layout.addWidget(add_row_button)
        layout.addWidget(remove_row_button)
        layout.addWidget(load_csv_button)
        layout.addWidget(save_csv_button)
        layout.addWidget(move_to_pivot_button)
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

        add_row_button = QPushButton("Add prediction")
        add_row_button.clicked.connect(lambda: self.add_prediction_row())
        remove_row_button = QPushButton("Remove prediction")
        remove_row_button.clicked.connect(self.remove_selected_prediction_rows)
        load_csv_button = QPushButton("Load CSV")
        load_csv_button.clicked.connect(self.load_prediction_csv_dialog)
        save_csv_button = QPushButton("Save CSV")
        save_csv_button.clicked.connect(self.save_prediction_csv_dialog)
        generate_button = QPushButton("Generate stage readouts")
        generate_button.clicked.connect(self.generate_prediction_stage_readouts)
        generate_paths_button = QPushButton("Generate paths")
        generate_paths_button.clicked.connect(self.generate_prediction_diffraction_paths)

        layout.addWidget(add_row_button)
        layout.addWidget(remove_row_button)
        layout.addWidget(load_csv_button)
        layout.addWidget(save_csv_button)
        layout.addWidget(generate_button)
        layout.addWidget(generate_paths_button)
        layout.addStretch(1)
        return toolbar

    def _build_prediction_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._build_section_title("Predicted stage readouts from model coordinates"))
        layout.addWidget(self._build_prediction_toolbar())
        self.prediction_table = self._build_prediction_table()
        layout.addWidget(self.prediction_table)
        return section

    def _build_measurement_table(self) -> QTableWidget:
        table = QTableWidget(0, len(TABLE_HEADERS))
        table.setHorizontalHeaderLabels(TABLE_HEADERS)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setMinimumHeight(220)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        table.verticalHeader().setDefaultSectionSize(30)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(True)
        table.itemSelectionChanged.connect(self.on_measurement_selection_changed)
        table.itemChanged.connect(self.on_measurement_item_changed)
        return table

    def _build_prediction_table(self) -> QTableWidget:
        table = QTableWidget(0, len(PREDICTION_TABLE_HEADERS))
        table.setHorizontalHeaderLabels(PREDICTION_TABLE_HEADERS)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setMinimumHeight(180)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        table.verticalHeader().setDefaultSectionSize(30)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(True)
        table.itemSelectionChanged.connect(self.on_prediction_selection_changed)
        table.itemChanged.connect(self.on_prediction_item_changed)
        return table

    def _build_report_box(self) -> QPlainTextEdit:
        report = QPlainTextEdit()
        report.setReadOnly(True)
        report.setMinimumHeight(96)
        report.setMaximumHeight(180)
        report.setPlainText(
            "Fit report will appear here.\n\n"
            "Workflow:\n"
            "1. Load an STL/mesh.\n"
            "2. Either run Fit placement or enable Manual Sample Placement.\n"
            "3. Adjust the live stage pose to inspect the setup.\n"
            "4. Compute the imaging map if needed."
        )
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
        if (
            hasattr(self, "auto_move_to_pivot_checkbox")
            and self.auto_move_to_pivot_checkbox.isChecked()
            and (self.fit_transform is not None or self.manual_placement_enabled_checkbox.isChecked())
        ):
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
        rotation = rotation_matrix_from_euler_xyz_deg(
            self.manual_rx.value(),
            self.manual_ry.value(),
            self.manual_rz.value(),
        )
        translation = np.array([self.manual_tx.value(), self.manual_ty.value(), self.manual_tz.value()], dtype=float)
        return rotation, translation

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
        if hasattr(self, "summary_beam_path"):
            self.summary_beam_path.setText("-")
        if hasattr(self, "summary_detector_map"):
            self.summary_detector_map.setText("-")
        if hasattr(self, "summary_diffraction_path"):
            self.summary_diffraction_path.setText("-")
        if hasattr(self, "summary_diffraction_map"):
            self.summary_diffraction_map.setText("-")
        if update_scene:
            self.update_scene(reset_camera=False)

    def update_placement_status(self) -> None:
        if self.manual_placement_enabled_checkbox.isChecked():
            self.summary_status.setText("Manual placement active")
        elif self.fit_transform is not None:
            self.summary_status.setText(f"Fit placement active with {len(self.residual_rows)} point pairs")
        else:
            self.summary_status.setText("No placement transform; model shown as imported")

    def update_placement_summary_fields(self) -> None:
        if self.manual_placement_enabled_checkbox.isChecked():
            rotation, translation, _source = self.current_model_to_stage_transform()
            self.summary_translation.setText(pretty_vector(translation))
            self.summary_euler.setText(pretty_vector(rotation_matrix_to_euler_zyx_deg(rotation)))
            self.summary_rms.setText("-")
            self.summary_max.setText("-")
        elif self.fit_transform is not None:
            self.summary_translation.setText(pretty_vector(self.fit_transform.translation))
            self.summary_euler.setText(pretty_vector(rotation_matrix_to_euler_zyx_deg(self.fit_transform.rotation)))
            self.summary_rms.setText(format_decimal(self.fit_transform.rms_error))
            self.summary_max.setText(format_decimal(self.fit_transform.max_error))
        else:
            self.summary_translation.setText("-")
            self.summary_euler.setText("-")
            self.summary_rms.setText("-")
            self.summary_max.setText("-")

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
    ) -> None:
        row = self.prediction_table.rowCount()
        self.prediction_table.insertRow(row)
        default_label = label if label is not None else f"G{row + 1}"
        model_point = model_point if model_point is not None else (0.0, 0.0, 0.0)
        stage_readout = stage_readout if stage_readout is not None else ("", "", "")
        path_values = path_values if path_values is not None else ("", "")
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

    def load_csv(self, path: Path) -> None:
        self.csv_path_label.setText(str(path))
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
        self.report_box.setPlainText(f"Loaded measurements from {path}")
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
                self.add_prediction_row(label=label, model_point=model_values, stage_readout=("", "", ""))
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
                    ["label", "model_x", "model_y", "model_z", "stage_x", "stage_y", "stage_z", "path_1", "path_2"]
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
        if self.model_mesh is None or self.model_mesh.n_points == 0:
            self.show_error("No mesh loaded", "Load a sample mesh before opening the point picker.")
            return
        if self.point_picker_dialog is None:
            self.point_picker_dialog = PointPickerDialog(self)
        self.point_picker_dialog.set_mesh(self.model_mesh, self.mesh_path)
        self.point_picker_dialog.show()
        self.point_picker_dialog.raise_()
        self.point_picker_dialog.activateWindow()

    def load_mesh(self, path: Path) -> None:
        try:
            self.model_mesh = load_mesh_as_polydata(path)
            self.mesh_path = path
            self.mesh_path_label.setText(str(path))
            self.invalidate_detector_map(update_scene=False)
            self.update_placement_status()
            self.statusBar().showMessage(f"Loaded mesh {path.name}", 5000)
            self.update_scene(reset_camera=True)
        except Exception as exc:
            self.show_error("Failed to load mesh", str(exc))

    def clear_mesh(self) -> None:
        self.mesh_path = None
        self.model_mesh = None
        self.mesh_path_label.setText("No mesh loaded")
        self.invalidate_detector_map(update_scene=False)
        self.update_placement_status()
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
            self.report_box.setPlainText(self.build_fit_report(transform, residual_rows))
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
                    (bank_1_state["average_length"], bank_2_state["average_length"]),
                    start=7,
                ):
                    item = self.prediction_table.item(row, index)
                    if item is None:
                        item = QTableWidgetItem()
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        self.prediction_table.setItem(row, index, item)
                    item.setText(format_decimal(float(value)))
            self.statusBar().showMessage(
                "Generated diffraction path columns for prediction points using the current live Omega.",
                5000,
            )
        except Exception as exc:
            self.show_error("Generate diffraction paths failed", str(exc))
        finally:
            self.prediction_table.blockSignals(was_blocked)

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

    def reset_manual_placement(self) -> None:
        spin_boxes = (
            self.manual_tx,
            self.manual_ty,
            self.manual_tz,
            self.manual_rx,
            self.manual_ry,
            self.manual_rz,
        )
        previous_states = [box.blockSignals(True) for box in spin_boxes]
        for box in spin_boxes:
            box.setValue(0.0)
        for box, previous in zip(spin_boxes, previous_states):
            box.blockSignals(previous)
        self.update_placement_status()
        self.update_placement_summary_fields()
        self.invalidate_detector_map(update_scene=False)
        self.update_scene(reset_camera=False)

    def load_fit_into_manual(self) -> None:
        if self.fit_transform is None:
            self.show_error("No fit available", "Run Fit placement before loading fit values into manual placement.")
            return
        translation = np.array(self.fit_transform.translation, dtype=float)
        euler_xyz = rotation_matrix_to_euler_xyz_deg(np.array(self.fit_transform.rotation, dtype=float))
        spin_boxes = (
            self.manual_tx,
            self.manual_ty,
            self.manual_tz,
            self.manual_rx,
            self.manual_ry,
            self.manual_rz,
        )
        previous_states = [box.blockSignals(True) for box in spin_boxes]
        self.manual_tx.setValue(float(translation[0]))
        self.manual_ty.setValue(float(translation[1]))
        self.manual_tz.setValue(float(translation[2]))
        self.manual_rx.setValue(float(euler_xyz[0]))
        self.manual_ry.setValue(float(euler_xyz[1]))
        self.manual_rz.setValue(float(euler_xyz[2]))
        for box, previous in zip(spin_boxes, previous_states):
            box.blockSignals(previous)
        previous_toggle_state = self.manual_placement_enabled_checkbox.blockSignals(True)
        self.manual_placement_enabled_checkbox.setChecked(True)
        self.manual_placement_enabled_checkbox.blockSignals(previous_toggle_state)
        self.update_placement_status()
        self.update_placement_summary_fields()
        self.invalidate_detector_map(update_scene=False)
        self.statusBar().showMessage("Loaded fit transform into manual placement.", 5000)
        self.update_scene(reset_camera=False)

    def clear_placement(self) -> None:
        spin_boxes = (
            self.manual_tx,
            self.manual_ty,
            self.manual_tz,
            self.manual_rx,
            self.manual_ry,
            self.manual_rz,
        )
        previous_states = [box.blockSignals(True) for box in spin_boxes]
        for box in spin_boxes:
            box.setValue(0.0)
        for box, previous in zip(spin_boxes, previous_states):
            box.blockSignals(previous)

        previous_toggle_state = self.manual_placement_enabled_checkbox.blockSignals(True)
        self.manual_placement_enabled_checkbox.setChecked(False)
        self.manual_placement_enabled_checkbox.blockSignals(previous_toggle_state)

        self.fit_transform = None
        self.residual_rows = []
        self.report_box.setPlainText(
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
            if self.fit_transform is None and not self.manual_placement_enabled_checkbox.isChecked():
                raise ValueError("Run Fit placement or enable manual placement before moving a point to the pivot.")
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
            self.summary_beam_path.setText(format_decimal(average_length))
            self.summary_detector_map.setText(
                f"{resolution_y}x{resolution_z} @ {format_decimal(pixel_size_y)}/{format_decimal(pixel_size_z)}, max {format_decimal(max_length)}"
            )
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
            self.summary_diffraction_path.setText(
                f"Bank 1: {format_decimal(self.diffraction_bank_1_map_state['average_length'])}, "
                f"Bank 2: {format_decimal(self.diffraction_bank_2_map_state['average_length'])}"
            )
            self.summary_diffraction_map.setText(
                f"Bank 1: {len(self.diffraction_bank_1_map_state['horizontal_angles_deg'])}x{len(self.diffraction_bank_1_map_state['vertical_angles_deg'])}, max {format_decimal(self.diffraction_bank_1_map_state['max_length'])}; "
                f"Bank 2: {len(self.diffraction_bank_2_map_state['horizontal_angles_deg'])}x{len(self.diffraction_bank_2_map_state['vertical_angles_deg'])}, max {format_decimal(self.diffraction_bank_2_map_state['max_length'])}"
            )
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
        horizontal_angles_deg, vertical_angles_deg = self.diffraction_angle_axes()
        slit_center = np.array([self.slit_x.value(), self.slit_y.value(), self.slit_z.value()], dtype=float)
        slit_width = self.slit_width.value()
        slit_height = self.slit_height.value()
        _incoming_y, _incoming_z, incoming_path_map = compute_incoming_beam_average_map(
            model_world,
            slit_center,
            slit_width,
            slit_height,
            len(horizontal_angles_deg),
            len(vertical_angles_deg),
            pivot[0],
        )
        incoming_average_path = float(np.mean(incoming_path_map)) if incoming_path_map.size else 0.0
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
            incoming_average_path,
            stage_pose_info,
        )
        bank_2_state = self._compute_single_diffraction_bank_map(
            model_world,
            pivot,
            self.current_diffraction_bank_2_geometry(),
            horizontal_angles_deg,
            vertical_angles_deg,
            incoming_average_path,
            stage_pose_info,
        )
        return bank_1_state, bank_2_state

    def _compute_single_diffraction_bank_map(
        self,
        model_world: pv.PolyData,
        pivot: np.ndarray,
        detector_geometry: dict,
        horizontal_angles_deg: np.ndarray,
        vertical_angles_deg: np.ndarray,
        incoming_average_path: float,
        stage_pose_info: dict,
    ) -> dict:
        center_direction = normalized(detector_geometry["center"] - pivot)
        right = detector_geometry["right"]
        up = detector_geometry["up"]
        path_map = np.full(
            (len(horizontal_angles_deg), len(vertical_angles_deg)),
            incoming_average_path,
            dtype=float,
        )
        detector_points = np.zeros((len(horizontal_angles_deg), len(vertical_angles_deg), 3), dtype=float)
        all_segments: List[Tuple[np.ndarray, np.ndarray]] = []

        for h_index, horizontal_deg in enumerate(horizontal_angles_deg):
            horizontal_scale = np.tan(np.radians(horizontal_deg))
            for v_index, vertical_deg in enumerate(vertical_angles_deg):
                vertical_scale = np.tan(np.radians(vertical_deg))
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
                path_length, segments = compute_segment_path_length(model_world, pivot, detector_point)
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
            "incoming_average_path": incoming_average_path,
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
        distance = self.scene_scale() * 1.8
        preset = self.camera_preset

        if preset == "iso":
            position = focus + np.array([distance, -distance, distance], dtype=float)
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus
        elif preset == "+x":
            position = focus + np.array([distance, 0.0, 0.0], dtype=float)
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus
        elif preset == "-x":
            position = focus + np.array([-distance, 0.0, 0.0], dtype=float)
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus
        elif preset == "+y":
            position = focus + np.array([0.0, distance, 0.0], dtype=float)
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus
        elif preset == "-y":
            position = focus + np.array([0.0, -distance, 0.0], dtype=float)
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus
        elif preset == "+z":
            position = focus + np.array([0.0, 0.0, distance], dtype=float)
            view_up = (0.0, 1.0, 0.0)
            focal_point = focus
        elif preset == "-z":
            position = focus + np.array([0.0, 0.0, -distance], dtype=float)
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
            position = focus + np.array([distance, -distance, distance], dtype=float)
            view_up = (0.0, 0.0, 1.0)
            focal_point = focus

        self.plotter.camera_position = [tuple(position), tuple(focal_point), view_up]
        self.apply_projection_mode()
        if reset_camera and preset != "theodolite":
            self.plotter.reset_camera()
            self.apply_projection_mode()
        else:
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
        if self.detector_map_state is not None:
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
            self.plotter.remove_actor("detector_map", render=False)
            self.detector_map_mesh = None

        diffraction_states = [
            state
            for state in (self.diffraction_bank_1_map_state, self.diffraction_bank_2_map_state)
            if state is not None
        ]
        if diffraction_states:
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
                direction=np.array([-200.0, 200.0, 0.0], dtype=float),
                tip_length=0.09,
                tip_radius=0.022,
                shaft_radius=0.006,
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
                direction=np.array([-200.0, -200.0, 0.0], dtype=float),
                tip_length=0.09,
                tip_radius=0.022,
                shaft_radius=0.006,
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

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
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

    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
