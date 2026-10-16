from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class EnginXInstrumentProfile:
    name: str = "ENGIN-X histogram NXS (DAE3 transmission-detector layout)"
    primary_flight_path_m: float = 50.0
    primary_flight_path_source: str = "ISIS ENGIN-X nominal instrument specification"
    bank_spectrum_ranges: tuple[tuple[int, int, int, str], ...] = (
        (1, 1, 1200, "North/+90 degree"),
        (2, 1201, 2400, "South/-90 degree"),
    )
    transmission_spectrum_range: tuple[int, int] = (2405, 2513)
    bank_initial_calibration: tuple[tuple[int, float, float], ...] = (
        (1, 18356.808, -35.147494),
        (2, 18439.181, -20.947738),
    )

    def seed_for_bank(self, bank_number: int) -> tuple[float, float]:
        for number, difc, tzero in self.bank_initial_calibration:
            if number == bank_number:
                return float(difc), float(tzero)
        raise ValueError(f"No initial calibration is defined for ENGIN-X bank {bank_number}.")


DEFAULT_ENGINX_PROFILE = EnginXInstrumentProfile()


@dataclass(frozen=True)
class RawNexusBank:
    bank_number: int
    bank_name: str
    spectrum_ids: np.ndarray
    histogram_rows: np.ndarray
    counts: np.ndarray
    detector_distance_m: np.ndarray
    polar_angle_deg: np.ndarray
    azimuthal_angle_deg: np.ndarray


@dataclass(frozen=True)
class RawEnginXNexusRun:
    source_path: Path
    run_number: str
    title: str
    period_index: int
    proton_charge_uah: float
    tof_edges_us: np.ndarray
    banks: Mapping[int, RawNexusBank]
    transmission_spectrum_ids: np.ndarray
    metadata: Mapping[str, object]


def _decoded_scalar(dataset, default: str = "") -> str:
    value = np.asarray(dataset).reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else default


def _units(dataset) -> str:
    value = dataset.attrs.get("units", "")
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("ascii", errors="replace")
    return str(value)


def _required(group, path: str):
    if path not in group:
        raise ValueError(f"Unsupported ENGIN-X NXS layout: missing /raw_data_1/{path}.")
    return group[path]


def _validate_edges(edges: np.ndarray) -> None:
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("NXS time_of_flight must be a one-dimensional histogram-edge array.")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("NXS time_of_flight edges must be finite and strictly increasing.")


def read_enginx_histogram_nexus(
    path: str | Path,
    *,
    period_index: int = 0,
    profile: EnginXInstrumentProfile = DEFAULT_ENGINX_PROFILE,
) -> RawEnginXNexusRun:
    """Read the supported ENGIN-X period/spectrum/TOF histogram NeXus layout.

    Geometry and spectrum IDs are kept separate from row indices so shuffled rows
    remain valid. Unlabelled user_table datasets are intentionally ignored.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Reading ENGIN-X NXS files requires h5py.") from exc

    source_path = Path(path)
    with h5py.File(source_path, "r") as handle:
        if "raw_data_1" not in handle:
            raise ValueError("Unsupported NXS layout: missing /raw_data_1.")
        root = handle["raw_data_1"]
        detector = _required(root, "detector_1")
        counts_ds = _required(detector, "counts")
        if counts_ds.ndim != 3:
            raise ValueError(
                "Unsupported ENGIN-X NXS layout: detector counts must be period x spectrum x TOF-bin histograms; event data are not supported."
            )
        period_count, row_count, bin_count = counts_ds.shape
        if period_index < 0 or period_index >= period_count:
            raise ValueError(f"Period {period_index + 1} is unavailable; file contains {period_count} period(s).")

        tof_ds = _required(detector, "time_of_flight")
        if _units(tof_ds).lower() not in {"microsecond", "microseconds", "us", "µs"}:
            raise ValueError(f"Unsupported time_of_flight units: {_units(tof_ds) or '<missing>'}.")
        tof_edges = np.asarray(tof_ds, dtype=float)
        _validate_edges(tof_edges)
        if tof_edges.size != bin_count + 1:
            raise ValueError("NXS time_of_flight must contain histogram edges (counts bins + 1).")

        spectrum_ids = np.asarray(_required(detector, "spectrum_index"), dtype=np.int64)
        if spectrum_ids.shape != (row_count,) or np.unique(spectrum_ids).size != row_count:
            raise ValueError("NXS spectrum_index must contain one unique ID per histogram row.")

        geometry = _required(root, "instrument/detector_1")
        distance_ds = _required(geometry, "distance")
        angle_ds = _required(geometry, "polar_angle")
        azimuth_ds = _required(geometry, "azimuthal_angle")
        if _units(distance_ds).lower() not in {"metre", "meter", "m"}:
            raise ValueError(f"Unsupported detector distance units: {_units(distance_ds) or '<missing>'}.")
        if _units(angle_ds).lower() not in {"degree", "degrees", "deg"}:
            raise ValueError(f"Unsupported detector angle units: {_units(angle_ds) or '<missing>'}.")
        distance = np.asarray(distance_ds, dtype=float)
        angle = np.asarray(angle_ds, dtype=float)
        azimuth = np.asarray(azimuth_ds, dtype=float)
        for name, values in (("distance", distance), ("polar_angle", angle), ("azimuthal_angle", azimuth)):
            if values.shape != (row_count,):
                raise ValueError(f"NXS detector {name} must map one value to every spectrum row.")

        charge_ds = _required(root, "periods/proton_charge")
        if _units(charge_ds).lower() not in {"uah", "µah"}:
            raise ValueError(f"Unsupported proton-charge units: {_units(charge_ds) or '<missing>'}.")
        charges = np.asarray(charge_ds, dtype=float).reshape(-1)
        if charges.size != period_count:
            raise ValueError("NXS periods/proton_charge does not match the number of periods.")
        charge = float(charges[period_index])
        if not np.isfinite(charge) or charge <= 0.0:
            raise ValueError("Recorded proton charge must be finite and greater than zero.")

        banks: dict[int, RawNexusBank] = {}
        diffraction_mask = np.zeros(row_count, dtype=bool)
        for bank_number, first_id, last_id, bank_name in profile.bank_spectrum_ranges:
            id_mask = (spectrum_ids >= first_id) & (spectrum_ids <= last_id)
            expected = last_id - first_id + 1
            if int(np.count_nonzero(id_mask)) != expected:
                raise ValueError(
                    f"ENGIN-X bank {bank_number} requires spectrum IDs {first_id}-{last_id} ({expected} spectra)."
                )
            rows = np.flatnonzero(id_mask)
            bank_angles = angle[rows]
            expected_sign = 1.0 if bank_number == 1 else -1.0
            if np.any(np.sign(bank_angles) != expected_sign) or np.any(np.abs(bank_angles) < 75.0) or np.any(np.abs(bank_angles) > 105.0):
                raise ValueError(f"Spectrum IDs for bank {bank_number} do not have the expected {'positive' if expected_sign > 0 else 'negative'} diffraction angles.")
            if np.any(~np.isfinite(distance[rows])) or np.any(distance[rows] <= 0.0):
                raise ValueError(f"Bank {bank_number} contains invalid sample-to-detector distances.")
            diffraction_mask[rows] = True
            banks[bank_number] = RawNexusBank(
                bank_number=bank_number,
                bank_name=bank_name,
                spectrum_ids=np.array(spectrum_ids[rows], copy=True),
                histogram_rows=np.array(rows, copy=True),
                counts=np.asarray(counts_ds[period_index, rows, :], dtype=float),
                detector_distance_m=np.array(distance[rows], copy=True),
                polar_angle_deg=np.array(bank_angles, copy=True),
                azimuthal_angle_deg=np.array(azimuth[rows], copy=True),
            )

        transmission_first, transmission_last = profile.transmission_spectrum_range
        transmission_mask = (spectrum_ids >= transmission_first) & (spectrum_ids <= transmission_last)
        transmission_ids = spectrum_ids[transmission_mask]
        if transmission_ids.size and not np.all(np.isclose(np.abs(angle[transmission_mask]), 180.0)):
            raise ValueError("Configured transmission spectrum IDs are not at 180 degrees.")
        if np.any(diffraction_mask & transmission_mask):
            raise ValueError("Diffraction and transmission spectrum selections overlap.")

        run_number = _decoded_scalar(_required(root, "run_number"))
        title = _decoded_scalar(root["title"]) if "title" in root else ""
        source_detector_distance = None
        if "source_detector_distance" in geometry:
            source_detector_distance = float(np.asarray(geometry["source_detector_distance"]).reshape(-1)[0])
        metadata = {
            "layout": "raw_data_1/detector_1 period-spectrum-TOF histogram",
            "tof_units": "microsecond",
            "proton_charge_units": "uAh",
            "profile_name": profile.name,
            "primary_flight_path_m": profile.primary_flight_path_m,
            "primary_flight_path_source": profile.primary_flight_path_source,
            "nxs_source_detector_distance_m": source_detector_distance,
            "ignored_unlabelled_user_tables": True,
            "transmission_spectra_excluded": int(transmission_ids.size),
        }
        return RawEnginXNexusRun(
            source_path=source_path,
            run_number=run_number,
            title=title,
            period_index=period_index,
            proton_charge_uah=charge,
            tof_edges_us=np.array(tof_edges, copy=True),
            banks=banks,
            transmission_spectrum_ids=np.array(transmission_ids, copy=True),
            metadata=metadata,
        )
