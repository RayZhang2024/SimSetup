from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .models import AXIS_TOF, FocusedSpectrum, InstrumentCalibration
from .nexus import DEFAULT_ENGINX_PROFILE, EnginXInstrumentProfile, RawEnginXNexusRun, RawNexusBank


NEUTRON_DIFC_US_PER_M_ANGSTROM = 505.5568
FOCUSING_CONVENTION_VERSION = "enginx-geometry-relative-v1"


@dataclass(frozen=True)
class BankFocusingDiagnostics:
    bank_number: int
    input_spectra: int
    masked_spectrum_ids: tuple[int, ...]
    transmission_spectra_used: int
    geometric_difc_min: float
    geometric_difc_median: float
    geometric_difc_max: float
    target_geometric_difc: float
    input_counts: float
    focused_counts: float
    supported_input_counts: float
    covariance_note: str


def geometric_difc(
    detector_distance_m: np.ndarray,
    polar_angle_deg: np.ndarray,
    primary_flight_path_m: float,
) -> np.ndarray:
    distances = np.asarray(detector_distance_m, dtype=float)
    angles = np.asarray(polar_angle_deg, dtype=float)
    if distances.shape != angles.shape or np.any(distances <= 0.0):
        raise ValueError("Detector distance and polar angle arrays must be matching and physically valid.")
    return (
        NEUTRON_DIFC_US_PER_M_ANGSTROM
        * (float(primary_flight_path_m) + distances)
        * np.sin(np.deg2rad(np.abs(angles)) / 2.0)
    )


def rebin_histogram_overlap(
    source_edges: np.ndarray,
    source_counts: np.ndarray,
    target_edges: np.ndarray,
    source_variance: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Count-conserving overlap rebinning using cumulative linear interpolation.

    A split source bin contributes in proportion to overlap. Variance uses Poisson
    thinning (fraction * variance); covariance between split pieces is ignored.
    """
    source_edges = np.asarray(source_edges, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)
    counts = np.asarray(source_counts, dtype=float)
    if source_edges.ndim != 1 or target_edges.ndim != 1:
        raise ValueError("Histogram edges must be one-dimensional.")
    if source_edges.size != counts.size + 1:
        raise ValueError("Source edges must have one more value than source counts.")
    if np.any(np.diff(source_edges) <= 0.0) or np.any(np.diff(target_edges) <= 0.0):
        raise ValueError("Histogram edges must be strictly increasing.")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("Histogram counts must be finite and non-negative.")
    variance = counts if source_variance is None else np.asarray(source_variance, dtype=float)
    if variance.shape != counts.shape or np.any(~np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("Source variance must match counts and be finite and non-negative.")
    cumulative_counts = np.concatenate(([0.0], np.cumsum(counts, dtype=float)))
    cumulative_variance = np.concatenate(([0.0], np.cumsum(variance, dtype=float)))
    target_cumulative_counts = np.interp(
        target_edges, source_edges, cumulative_counts, left=0.0, right=float(cumulative_counts[-1])
    )
    target_cumulative_variance = np.interp(
        target_edges, source_edges, cumulative_variance, left=0.0, right=float(cumulative_variance[-1])
    )
    return np.diff(target_cumulative_counts), np.maximum(np.diff(target_cumulative_variance), 0.0)


def focus_enginx_bank(
    run: RawEnginXNexusRun,
    bank_number: int,
    *,
    masked_spectrum_ids: Iterable[int] = (),
    target_edges_us: Optional[np.ndarray] = None,
    profile: EnginXInstrumentProfile = DEFAULT_ENGINX_PROFILE,
) -> tuple[FocusedSpectrum, BankFocusingDiagnostics]:
    if bank_number not in run.banks:
        raise ValueError(f"Run does not contain ENGIN-X bank {bank_number}.")
    bank: RawNexusBank = run.banks[bank_number]
    masked = {int(value) for value in masked_spectrum_ids}
    keep = np.array([int(value) not in masked for value in bank.spectrum_ids], dtype=bool)
    if not np.any(keep):
        raise ValueError(f"All spectra in bank {bank_number} are masked.")
    difc_values = geometric_difc(
        bank.detector_distance_m,
        bank.polar_angle_deg,
        profile.primary_flight_path_m,
    )
    target_difc = float(np.median(difc_values[keep]))
    target_edges = np.asarray(run.tof_edges_us if target_edges_us is None else target_edges_us, dtype=float)
    if (
        target_edges.ndim != 1
        or target_edges.size < 2
        or np.any(~np.isfinite(target_edges))
        or np.any(np.diff(target_edges) <= 0.0)
    ):
        raise ValueError("Target focusing edges must be finite and strictly increasing.")
    focused_counts = np.zeros(target_edges.size - 1, dtype=float)
    focused_variance = np.zeros_like(focused_counts)
    supported_input_counts = 0.0
    for index in np.flatnonzero(keep):
        scale = target_difc / float(difc_values[index])
        mapped_edges = run.tof_edges_us * scale
        counts, variance = rebin_histogram_overlap(mapped_edges, bank.counts[index], target_edges)
        focused_counts += counts
        focused_variance += variance
        overlap_low = max(float(mapped_edges[0]), float(target_edges[0]))
        overlap_high = min(float(mapped_edges[-1]), float(target_edges[-1]))
        if overlap_high > overlap_low:
            supported, _ = rebin_histogram_overlap(
                mapped_edges,
                bank.counts[index],
                np.array([overlap_low, overlap_high], dtype=float),
            )
            supported_input_counts += float(supported[0])
    seed_difc, seed_tzero = profile.seed_for_bank(bank_number)
    metadata = dict(run.metadata)
    metadata.update(
        {
            "processing_version": FOCUSING_CONVENTION_VERSION,
            "source_run": run.run_number,
            "source_title": run.title,
            "period_index": run.period_index,
            "proton_charge": run.proton_charge_uah,
            "proton_charge_units": "uAh",
            "bank_spectrum_ids": [int(value) for value in bank.spectrum_ids],
            "masked_spectrum_ids": sorted(masked),
            "pixel_grouping": f"spectrum IDs {int(bank.spectrum_ids.min())}-{int(bank.spectrum_ids.max())}",
            "focusing_map": "TOF_bank = TOF_pixel * median(geometry_DIFC) / geometry_DIFC_pixel",
            "geometry_difc_source": "nominal 50 m primary path plus NXS sample-detector distance and polar angle",
            "target_geometric_difc": target_difc,
            "initial_difc": seed_difc,
            "initial_tzero": seed_tzero,
            "calibration_application_count": 0,
            "uncertainty_model": "Poisson counting variance; overlap-split covariance ignored",
        }
    )
    spectrum = FocusedSpectrum(
        source_path=run.source_path,
        x=np.array(target_edges, copy=True),
        y=focused_counts,
        e=np.sqrt(focused_variance),
        native_axis=AXIS_TOF,
        x_is_edges=True,
        calibration=InstrumentCalibration(difc=seed_difc, tzero=seed_tzero, difa=0.0),
        metadata=metadata,
        run_number=run.run_number,
        bank_number=bank_number,
        bank_name=bank.bank_name,
        x_label="Time-of-Flight (microsecond)",
        y_label="Focused counts",
    )
    diagnostics = BankFocusingDiagnostics(
        bank_number=bank_number,
        input_spectra=int(np.count_nonzero(keep)),
        masked_spectrum_ids=tuple(sorted(masked)),
        transmission_spectra_used=0,
        geometric_difc_min=float(np.min(difc_values[keep])),
        geometric_difc_median=float(np.median(difc_values[keep])),
        geometric_difc_max=float(np.max(difc_values[keep])),
        target_geometric_difc=target_difc,
        input_counts=float(np.sum(bank.counts[keep])),
        focused_counts=float(np.sum(focused_counts)),
        supported_input_counts=float(supported_input_counts),
        covariance_note="Poisson thinning variance is propagated; correlations from splitting a source bin are not represented.",
    )
    return spectrum, diagnostics
