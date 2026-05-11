from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .models import AXIS_D_SPACING, AXIS_TOF, FocusedSpectrum
from .open_genie_his import read_his_spectrum


class UnsupportedDiffractionFormatError(ValueError):
    pass


def _numeric_rows_from_lines(lines: Iterable[str]) -> np.ndarray:
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", ";")):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append([float(part) for part in parts[:3]])
        except ValueError:
            continue
    if not rows:
        raise ValueError("No numeric diffraction data rows were found.")
    width = max(len(row) for row in rows)
    output = np.full((len(rows), width), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        output[row_index, : len(row)] = row
    return output


def _load_delimited_text(path: Path) -> FocusedSpectrum:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = _numeric_rows_from_lines(text.splitlines())
    e = rows[:, 2] if rows.shape[1] >= 3 else None
    return FocusedSpectrum(
        source_path=path,
        x=rows[:, 0],
        y=rows[:, 1],
        e=e,
        native_axis=AXIS_TOF,
        x_is_edges=False,
        x_label="Time-of-Flight",
        y_label="Intensity",
    )


def _column_by_names(fieldnames: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    normalized = {field.strip().lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _load_csv(path: Path) -> FocusedSpectrum:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path.name} does not contain a CSV header.")
            x_key = _column_by_names(reader.fieldnames, ("tof", "x", "time-of-flight", "d", "d_spacing", "d-spacing"))
            y_key = _column_by_names(reader.fieldnames, ("y", "counts", "intensity", "neutron counts"))
            e_key = _column_by_names(reader.fieldnames, ("e", "error", "sigma", "uncertainty"))
            if x_key is None or y_key is None:
                raise ValueError(f"{path.name} is missing x/tof and y/counts columns.")
            x_values = []
            y_values = []
            e_values = []
            for row in reader:
                x_values.append(float(row[x_key]))
                y_values.append(float(row[y_key]))
                if e_key is not None and row.get(e_key, "").strip():
                    e_values.append(float(row[e_key]))
            native_axis = AXIS_D_SPACING if x_key.strip().lower() in {"d", "d_spacing", "d-spacing"} else AXIS_TOF
            e = np.asarray(e_values, dtype=float) if len(e_values) == len(y_values) else None
            return FocusedSpectrum(
                source_path=path,
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(y_values, dtype=float),
                e=e,
                native_axis=native_axis,
                x_is_edges=False,
                x_label="d-spacing" if native_axis == AXIS_D_SPACING else "Time-of-Flight",
                y_label="Intensity",
            )
    return _load_delimited_text(path)


def load_focused_spectrum(path: str | Path) -> FocusedSpectrum:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".his":
        return read_his_spectrum(source_path)
    if suffix == ".csv":
        return _load_csv(source_path)
    if suffix in {".xye", ".dat", ".txt"}:
        return _load_delimited_text(source_path)
    if suffix in {".gss", ".gsa"}:
        raise UnsupportedDiffractionFormatError(
            "GSAS histogram import is reserved for a follow-up once a representative .gss/.gsa file is available."
        )
    raise UnsupportedDiffractionFormatError(f"Unsupported diffraction file format: {suffix or source_path.name}")
