from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


AXIS_TOF = "tof"
AXIS_D_SPACING = "d_spacing"


@dataclass(frozen=True)
class InstrumentCalibration:
    difc: float
    tzero: float = 0.0
    difa: float = 0.0

    def has_difa(self) -> bool:
        return abs(float(self.difa)) > 1e-12


def d_to_tof(d_spacing: np.ndarray | Sequence[float] | float, calibration: InstrumentCalibration) -> np.ndarray:
    d_values = np.asarray(d_spacing, dtype=float)
    return (
        float(calibration.tzero)
        + float(calibration.difc) * d_values
        + float(calibration.difa) * d_values * d_values
    )


def tof_to_d(tof: np.ndarray | Sequence[float] | float, calibration: InstrumentCalibration) -> np.ndarray:
    tof_values = np.asarray(tof, dtype=float)
    difc = float(calibration.difc)
    difa = float(calibration.difa)
    tzero = float(calibration.tzero)
    if abs(difc) < 1e-12:
        raise ValueError("DIFC must be non-zero for TOF/d-spacing conversion.")
    if abs(difa) < 1e-12:
        return (tof_values - tzero) / difc
    discriminant = difc * difc + 4.0 * difa * (tof_values - tzero)
    if np.any(discriminant < 0.0):
        raise ValueError("TOF values produce a negative d-spacing conversion discriminant.")
    root = np.sqrt(discriminant)
    candidates = np.stack(
        [
            (-difc + root) / (2.0 * difa),
            (-difc - root) / (2.0 * difa),
        ],
        axis=0,
    )
    positive = np.where(candidates[0] > 0.0, candidates[0], candidates[1])
    return positive


@dataclass
class FocusedSpectrum:
    source_path: Path
    x: np.ndarray
    y: np.ndarray
    e: Optional[np.ndarray] = None
    native_axis: str = AXIS_TOF
    x_is_edges: bool = False
    calibration: Optional[InstrumentCalibration] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    run_number: Optional[str] = None
    bank_number: Optional[int] = None
    bank_name: Optional[str] = None
    x_label: str = "Time-of-Flight"
    y_label: str = "Intensity"

    def __post_init__(self) -> None:
        self.source_path = Path(self.source_path)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.e is not None:
            self.e = np.asarray(self.e, dtype=float)
        if self.x_is_edges:
            expected = self.y.size + 1
            if self.x.size != expected:
                raise ValueError(f"Expected {expected} x bin edges for {self.y.size} y values.")
        elif self.x.size != self.y.size:
            raise ValueError("x and y arrays must have the same length unless x contains bin edges.")
        if self.e is not None and self.e.size != self.y.size:
            raise ValueError("Uncertainty array length must match y values.")
        if self.native_axis not in {AXIS_TOF, AXIS_D_SPACING}:
            raise ValueError(f"Unsupported native axis: {self.native_axis}")

    @property
    def x_centres(self) -> np.ndarray:
        if self.x_is_edges:
            return 0.5 * (self.x[:-1] + self.x[1:])
        return self.x

    def axis_values(self, axis: str) -> np.ndarray:
        if axis == self.native_axis:
            return self.x_centres
        if self.calibration is None:
            raise ValueError("Axis conversion requires embedded calibration metadata.")
        if axis == AXIS_D_SPACING and self.native_axis == AXIS_TOF:
            return tof_to_d(self.x_centres, self.calibration)
        if axis == AXIS_TOF and self.native_axis == AXIS_D_SPACING:
            return d_to_tof(self.x_centres, self.calibration)
        raise ValueError(f"Unsupported axis: {axis}")

    def axis_label(self, axis: str) -> str:
        if axis == AXIS_TOF:
            return "Time-of-Flight"
        if axis == AXIS_D_SPACING:
            return "d-spacing"
        return axis


def spectrum_with_calibration(
    spectrum: FocusedSpectrum,
    calibration: InstrumentCalibration,
    metadata: Optional[Dict[str, object]] = None,
) -> FocusedSpectrum:
    combined_metadata = dict(spectrum.metadata)
    if metadata:
        combined_metadata.update(metadata)
    return FocusedSpectrum(
        source_path=spectrum.source_path,
        x=np.array(spectrum.x, dtype=float, copy=True),
        y=np.array(spectrum.y, dtype=float, copy=True),
        e=None if spectrum.e is None else np.array(spectrum.e, dtype=float, copy=True),
        native_axis=spectrum.native_axis,
        x_is_edges=spectrum.x_is_edges,
        calibration=calibration,
        metadata=combined_metadata,
        run_number=spectrum.run_number,
        bank_number=spectrum.bank_number,
        bank_name=spectrum.bank_name,
        x_label=spectrum.x_label,
        y_label=spectrum.y_label,
    )


@dataclass(frozen=True)
class AtomRecord:
    label: str
    element: str
    x: float
    y: float
    z: float
    occupancy: float


@dataclass(frozen=True)
class PhaseModel:
    index: int
    name: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group: str
    atoms: Tuple[AtomRecord, ...] = ()


@dataclass(frozen=True)
class Reflection:
    h: int
    k: int
    l: int
    d_spacing: float
    multiplicity: int = 1

    @property
    def label(self) -> str:
        return f"{self.h}{self.k}{self.l}"


@dataclass(frozen=True)
class FitQuality:
    chi_square: float
    reduced_chi_square: float
    rwp_percent: float
    points: int


@dataclass(frozen=True)
class PeakFitResult:
    model_name: str
    centre: float
    centre_uncertainty: Optional[float]
    fwhm: float
    fwhm_uncertainty: Optional[float]
    height: float
    area: float
    eta: float
    background_coefficients: Tuple[float, ...]
    quality: FitQuality
    fit_x: Optional[np.ndarray] = None
    fit_y: Optional[np.ndarray] = None
    observed_y: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PawleyReflectionResult:
    reflection: Reflection
    position: float
    intensity: float


@dataclass(frozen=True)
class PawleyFitResult:
    lattice_a: float
    fwhm: float
    eta: float
    background_coefficients: Tuple[float, ...]
    reflections: Tuple[PawleyReflectionResult, ...]
    quality: FitQuality
    fit_x: Optional[np.ndarray] = None
    fit_y: Optional[np.ndarray] = None
    observed_y: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CalibrationPeakResult:
    reflection: Reflection
    expected_tof: float
    observed_tof: Optional[float]
    fitted_tof: Optional[float]
    residual_tof: Optional[float]
    prominence: Optional[float]
    accepted: bool
    rejection_reason: str = ""


@dataclass(frozen=True)
class CalibrationResult:
    calibration: InstrumentCalibration
    initial_calibration: InstrumentCalibration
    phase_name: str
    source_path: Path
    run_number: Optional[str]
    bank_number: Optional[int]
    peak_results: Tuple[CalibrationPeakResult, ...]
    rms_residual_tof: float

    @property
    def accepted_peaks(self) -> Tuple[CalibrationPeakResult, ...]:
        return tuple(peak for peak in self.peak_results if peak.accepted)

    @property
    def rejected_peaks(self) -> Tuple[CalibrationPeakResult, ...]:
        return tuple(peak for peak in self.peak_results if not peak.accepted)


@dataclass(frozen=True)
class NormalizationResult:
    corrected_spectrum: FocusedSpectrum
    sample_source_path: Path
    vanadium_source_path: Path
    sample_run_number: Optional[str]
    vanadium_run_number: Optional[str]
    bank_number: Optional[int]
    scale: float
    valid_bins: int
    invalid_bins: int
    smoothed: bool
