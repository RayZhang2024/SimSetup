#!/usr/bin/env python3
"""
Fit the placement of a sample model on a stage from point correspondences.

Measurement model:
    point_in_stage_frame = -stage_readout

The stage readout is interpreted in the stage's own x/y/z frame. When a feature
point is brought to the fixed pivot, the required stage-local translation is the
negative of that feature point's coordinate in the stage frame. This makes the
placement fit independent of omega for the measurement workflow described by the
user.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]
Matrix3 = Tuple[Vector3, Vector3, Vector3]
Matrix4 = Tuple[Tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class Measurement:
    label: str
    model_point: Vector3
    stage_readout: Vector3


@dataclass(frozen=True)
class Transform:
    rotation: Matrix3
    translation: Vector3
    quaternion_wxyz: Tuple[float, float, float, float]
    rms_error: float
    max_error: float

    def as_matrix4(self) -> Matrix4:
        r = self.rotation
        t = self.translation
        return (
            (r[0][0], r[0][1], r[0][2], t[0]),
            (r[1][0], r[1][1], r[1][2], t[1]),
            (r[2][0], r[2][1], r[2][2], t[2]),
            (0.0, 0.0, 0.0, 1.0),
        )


def vec_add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(a: Vector3, s: float) -> Vector3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vec_dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_norm(a: Vector3) -> float:
    return math.sqrt(vec_dot(a, a))


def mat_vec_mul(m: Matrix3, v: Vector3) -> Vector3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def centroid(points: Sequence[Vector3]) -> Vector3:
    count = float(len(points))
    sums = (0.0, 0.0, 0.0)
    for point in points:
        sums = vec_add(sums, point)
    return vec_scale(sums, 1.0 / count)


def rotation_z_deg(angle_deg: float) -> Matrix3:
    angle_rad = math.radians(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (
        (c, -s, 0.0),
        (s, c, 0.0),
        (0.0, 0.0, 1.0),
    )


def quaternion_to_matrix(quaternion: Tuple[float, float, float, float]) -> Matrix3:
    w, x, y, z = quaternion
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ),
        (
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ),
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def normalize_quaternion(quaternion: Sequence[float]) -> Tuple[float, float, float, float]:
    length = math.sqrt(sum(component * component for component in quaternion))
    if length == 0.0:
        raise ValueError("Quaternion normalization failed because the vector length is zero.")
    q = tuple(component / length for component in quaternion)
    return (q[0], q[1], q[2], q[3])


def dominant_eigenvector_symmetric_4x4(matrix: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    a = [[float(matrix[row][col]) for col in range(4)] for row in range(4)]
    v = [[1.0 if row == col else 0.0 for col in range(4)] for row in range(4)]

    for _ in range(64):
        p, q = 0, 1
        max_off_diagonal = abs(a[p][q])
        for row in range(4):
            for col in range(row + 1, 4):
                candidate = abs(a[row][col])
                if candidate > max_off_diagonal:
                    p, q = row, col
                    max_off_diagonal = candidate

        if max_off_diagonal < 1e-14:
            break

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]
        tau = (aqq - app) / (2.0 * apq)
        t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau)
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        for k in range(4):
            if k != p and k != q:
                akp = a[k][p]
                akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[p][k] = a[k][p]
                a[k][q] = s * akp + c * akq
                a[q][k] = a[k][q]

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for k in range(4):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq

    eigenvalues = [a[i][i] for i in range(4)]
    dominant_index = max(range(4), key=eigenvalues.__getitem__)
    eigenvector = tuple(v[row][dominant_index] for row in range(4))
    return normalize_quaternion(eigenvector)


def rigid_transform_horn(source_points: Sequence[Vector3], target_points: Sequence[Vector3]) -> Transform:
    if len(source_points) != len(target_points):
        raise ValueError("Source and target point lists must have the same length.")
    if len(source_points) < 3:
        raise ValueError("At least three point pairs are required.")

    src_centroid = centroid(source_points)
    dst_centroid = centroid(target_points)
    src_centered = [vec_sub(point, src_centroid) for point in source_points]
    dst_centered = [vec_sub(point, dst_centroid) for point in target_points]

    s_xx = s_xy = s_xz = 0.0
    s_yx = s_yy = s_yz = 0.0
    s_zx = s_zy = s_zz = 0.0

    for src, dst in zip(src_centered, dst_centered):
        s_xx += src[0] * dst[0]
        s_xy += src[0] * dst[1]
        s_xz += src[0] * dst[2]
        s_yx += src[1] * dst[0]
        s_yy += src[1] * dst[1]
        s_yz += src[1] * dst[2]
        s_zx += src[2] * dst[0]
        s_zy += src[2] * dst[1]
        s_zz += src[2] * dst[2]

    trace = s_xx + s_yy + s_zz
    horn_matrix = (
        (trace, s_yz - s_zy, s_zx - s_xz, s_xy - s_yx),
        (s_yz - s_zy, s_xx - s_yy - s_zz, s_xy + s_yx, s_zx + s_xz),
        (s_zx - s_xz, s_xy + s_yx, -s_xx + s_yy - s_zz, s_yz + s_zy),
        (s_xy - s_yx, s_zx + s_xz, s_yz + s_zy, -s_xx - s_yy + s_zz),
    )

    quaternion = dominant_eigenvector_symmetric_4x4(horn_matrix)
    rotation = quaternion_to_matrix(quaternion)
    translation = vec_sub(dst_centroid, mat_vec_mul(rotation, src_centroid))

    residuals = []
    for src, dst in zip(source_points, target_points):
        predicted = vec_add(mat_vec_mul(rotation, src), translation)
        residuals.append(vec_norm(vec_sub(predicted, dst)))

    rms_error = math.sqrt(sum(error * error for error in residuals) / len(residuals))
    max_error = max(residuals)
    return Transform(rotation, translation, quaternion, rms_error, max_error)


def infer_stage_point_from_readout(readout: Vector3) -> Vector3:
    return vec_scale(readout, -1.0)


def load_measurements(csv_path: Path) -> List[Measurement]:
    required_columns = {
        "label",
        "model_x",
        "model_y",
        "model_z",
        "stage_x",
        "stage_y",
        "stage_z",
    }

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} does not contain a CSV header row.")
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}."
            )

        measurements: List[Measurement] = []
        for row_index, row in enumerate(reader, start=2):
            try:
                measurements.append(
                    Measurement(
                        label=row["label"].strip(),
                        model_point=(
                            float(row["model_x"]),
                            float(row["model_y"]),
                            float(row["model_z"]),
                        ),
                        stage_readout=(
                            float(row["stage_x"]),
                            float(row["stage_y"]),
                            float(row["stage_z"]),
                        ),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {row_index}.") from exc

    if len(measurements) < 3:
        raise ValueError("At least three measurements are required.")
    return measurements


def fit_from_measurements(
    measurements: Sequence[Measurement],
) -> Tuple[Transform, List[Tuple[Measurement, Vector3, float]]]:
    source_points = [measurement.model_point for measurement in measurements]
    target_points = [infer_stage_point_from_readout(measurement.stage_readout) for measurement in measurements]

    transform = rigid_transform_horn(source_points, target_points)
    residual_rows: List[Tuple[Measurement, Vector3, float]] = []
    for measurement, target_point in zip(measurements, target_points):
        fitted_point = vec_add(mat_vec_mul(transform.rotation, measurement.model_point), transform.translation)
        error = vec_norm(vec_sub(fitted_point, target_point))
        residual_rows.append((measurement, target_point, error))
    return transform, residual_rows


def pretty_vector(vector: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value: .3f}" for value in vector) + "]"


def rotation_matrix_to_euler_zyx_deg(rotation: Matrix3) -> Vector3:
    sy = math.sqrt(rotation[0][0] * rotation[0][0] + rotation[1][0] * rotation[1][0])
    singular = sy < 1e-9
    if not singular:
        x = math.atan2(rotation[2][1], rotation[2][2])
        y = math.atan2(-rotation[2][0], sy)
        z = math.atan2(rotation[1][0], rotation[0][0])
    else:
        x = math.atan2(-rotation[1][2], rotation[1][1])
        y = math.atan2(-rotation[2][0], sy)
        z = 0.0
    return tuple(math.degrees(angle) for angle in (z, y, x))


def save_json_report(
    output_path: Path,
    transform: Transform,
    residual_rows: Sequence[Tuple[Measurement, Vector3, float]],
    pivot_world: Optional[Vector3] = None,
) -> None:
    payload = {
        "rotation_matrix": transform.rotation,
        "translation": transform.translation,
        "matrix4x4": transform.as_matrix4(),
        "quaternion_wxyz": transform.quaternion_wxyz,
        "euler_zyx_deg": rotation_matrix_to_euler_zyx_deg(transform.rotation),
        "rms_error": transform.rms_error,
        "max_error": transform.max_error,
        "measurements": [
            {
                **asdict(measurement),
                "stage_point": stage_point,
                "fit_error": fit_error,
            }
            for measurement, stage_point, fit_error in residual_rows
        ],
    }
    if pivot_world is not None:
        payload["pivot_world"] = pivot_world
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_demo() -> None:
    model_points = [
        ("P1", (15.0, 5.0, 0.0)),
        ("P2", (25.0, -8.0, 4.0)),
        ("P3", (-12.0, 18.0, 6.0)),
        ("P4", (2.0, -16.0, 12.0)),
        ("P5", (-20.0, -10.0, 3.0)),
        ("P6", (8.0, 22.0, -5.0)),
    ]
    model_to_stage_rotation = rotation_z_deg(18.0)
    model_to_stage_translation = (120.0, -35.0, 42.0)

    measurements = []
    for label, model_point in model_points:
        stage_point = vec_add(mat_vec_mul(model_to_stage_rotation, model_point), model_to_stage_translation)
        stage_readout = infer_stage_point_from_readout(stage_point)
        measurements.append(Measurement(label, model_point, stage_readout))

    transform, _ = fit_from_measurements(measurements)
    expected_translation_error = vec_norm(vec_sub(transform.translation, model_to_stage_translation))
    print("Demo fit using six synthetic measurements")
    print(f"Recovered translation: {pretty_vector(transform.translation)}")
    print(f"Expected translation:  {pretty_vector(model_to_stage_translation)}")
    print(f"Translation error:     {expected_translation_error:.6e}")
    print(f"Recovered quaternion:  {pretty_vector(transform.quaternion_wxyz)}")
    print(f"RMS point error:       {transform.rms_error:.6e}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a model-to-stage transform from measured feature points and stage-local readouts."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="CSV file with columns: label, model_x, model_y, model_z, stage_x, stage_y, stage_z",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON report path for the fitted transform and per-point residuals.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a built-in synthetic example instead of loading a CSV file.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.csv_path:
        parser.error("Either provide a CSV file or use --demo.")

    csv_path = Path(args.csv_path)
    measurements = load_measurements(csv_path)
    transform, residual_rows = fit_from_measurements(measurements)

    print(f"Loaded {len(measurements)} measurements from {csv_path}")
    print("")
    print("Model -> stage transform")
    print(f"Translation:  {pretty_vector(transform.translation)}")
    print(f"Quaternion:   {pretty_vector(transform.quaternion_wxyz)}")
    print(f"Euler ZYX deg:{pretty_vector(rotation_matrix_to_euler_zyx_deg(transform.rotation))}")
    print("Matrix 4x4:")
    for row in transform.as_matrix4():
        print(f"  {pretty_vector(row)}")
    print("")
    print(f"RMS fit error: {transform.rms_error:.6f}")
    print(f"Max fit error: {transform.max_error:.6f}")
    print("")
    print("Per-point residuals")
    for measurement, stage_point, fit_error in residual_rows:
        print(
            f"  {measurement.label}: "
            f"stage-point={pretty_vector(stage_point)} "
            f"error={fit_error:.6f}"
        )

    if args.output_json:
        save_json_report(args.output_json, transform, residual_rows)
        print("")
        print(f"Wrote JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
