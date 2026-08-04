from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from .convergence import fitting_settings_min_dm_over_m
from .fitting import fit_peak_profile
from .gsas_exp import generate_phase_reflections
from .models import (
    AXIS_D_SPACING,
    AXIS_TOF,
    FIT_SCOPE_PATTERN,
    FittingSettings,
    FocusedSpectrum,
    PawleyFitResult,
    PeakFitResult,
    PhaseModel,
    d_to_tof,
    tof_to_d,
)
from .pawley import fit_pawley
from .profile_metadata import initial_profile_parameters_from_metadata
from .profiles import is_gsas_tof_profile


FitResult = Union[PeakFitResult, PawleyFitResult]


@dataclass(frozen=True)
class SpectrumFitRequest:
    spectrum_index: int
    spectrum: FocusedSpectrum
    settings: FittingSettings
    phase: Optional[PhaseModel]
    display_axis: str
    data_mode: str
    selected_range: tuple[float, float]


@dataclass(frozen=True)
class SpectrumFitRecord:
    spectrum_index: int
    spectrum: FocusedSpectrum
    settings: FittingSettings
    display_axis: str
    fit_axis: str
    data_mode: str
    selected_range: tuple[float, float]
    result: FitResult
    curve_x: Optional[np.ndarray]
    fit_y: Optional[np.ndarray]
    observed_y: Optional[np.ndarray]

    @property
    def residual(self) -> Optional[np.ndarray]:
        if self.observed_y is None or self.fit_y is None:
            return None
        return np.asarray(self.observed_y, dtype=float) - np.asarray(self.fit_y, dtype=float)

    @property
    def weighted_residual(self) -> Optional[np.ndarray]:
        residual = self.residual
        uncertainty = self.result.fit_uncertainty
        if residual is None or uncertainty is None:
            return None
        return residual / np.asarray(uncertainty, dtype=float)


def convert_axis_values(
    values: np.ndarray,
    from_axis: str,
    to_axis: str,
    spectrum: FocusedSpectrum,
) -> np.ndarray:
    if from_axis == to_axis:
        return np.array(values, dtype=float, copy=True)
    if spectrum.calibration is None:
        raise ValueError("Calibration metadata is required for TOF/d-spacing conversion.")
    if from_axis == AXIS_TOF and to_axis == AXIS_D_SPACING:
        return np.asarray(tof_to_d(values, spectrum.calibration), dtype=float)
    if from_axis == AXIS_D_SPACING and to_axis == AXIS_TOF:
        return np.asarray(d_to_tof(values, spectrum.calibration), dtype=float)
    raise ValueError(f"Unsupported axis conversion: {from_axis} to {to_axis}")


def position_to_d_spacing(value: float, axis: str, spectrum: FocusedSpectrum) -> float:
    if axis == AXIS_D_SPACING:
        return float(value)
    if spectrum.calibration is None:
        raise ValueError("Calibration metadata is required to report fitting results in d-spacing.")
    return float(tof_to_d(float(value), spectrum.calibration))


def width_to_d_spacing(width: float, centre: float, axis: str, spectrum: FocusedSpectrum) -> float:
    if axis == AXIS_D_SPACING:
        return abs(float(width))
    if spectrum.calibration is None:
        raise ValueError("Calibration metadata is required to report fitting results in d-spacing.")
    centre_d = float(tof_to_d(float(centre), spectrum.calibration))
    local_slope = float(spectrum.calibration.difc) + 2.0 * float(spectrum.calibration.difa) * centre_d
    if abs(local_slope) < 1e-12:
        raise ValueError("Calibration slope is too small to convert TOF width to d-spacing.")
    return abs(float(width) / local_slope)


def expected_phase_reflection_positions(
    phase: Optional[PhaseModel],
    axis: str,
    calibration,
    axis_range: tuple[float, float],
    reflection_margin_percent: float = 0.0,
) -> Optional[list[float]]:
    reflections = expected_phase_reflections(
        phase,
        axis,
        calibration,
        axis_range,
        reflection_margin_percent=reflection_margin_percent,
    )
    if reflections is None:
        return None
    return [position for position, _reflection in reflections]


def expected_phase_reflections(
    phase: Optional[PhaseModel],
    axis: str,
    calibration,
    axis_range: tuple[float, float],
    reflection_margin_percent: float = 0.0,
):
    if phase is None:
        return None
    axis_low, axis_high = sorted((float(axis_range[0]), float(axis_range[1])))
    if axis == AXIS_D_SPACING:
        d_low, d_high = axis_low, axis_high
    elif axis == AXIS_TOF:
        if calibration is None:
            return None
        d_values = tof_to_d(np.array([axis_low, axis_high], dtype=float), calibration)
        d_low, d_high = sorted((float(d_values[0]), float(d_values[1])))
    else:
        return None
    margin = max(float(reflection_margin_percent), 0.0) / 100.0 * max(abs(d_high - d_low), 1e-12)
    reflections = generate_phase_reflections(phase, d_low - margin, d_high + margin)
    positions = []
    for reflection in reflections:
        if axis == AXIS_D_SPACING:
            position = float(reflection.d_spacing)
        else:
            position = float(d_to_tof(reflection.d_spacing, calibration))
        if axis_low <= position <= axis_high:
            positions.append((position, reflection))
    return sorted(positions, key=lambda item: item[0])


def fit_spectrum(
    request: SpectrumFitRequest,
    abort_check: Optional[Callable[[], Optional[str]]] = None,
    enable_stagnation_detection: Optional[bool] = None,
    use_gsas_convergence: Optional[bool] = None,
    checkpoint_callback: Optional[Callable[[dict[str, object]], None]] = None,
) -> SpectrumFitRecord:
    spectrum = request.spectrum
    settings = request.settings
    x_values = np.asarray(spectrum.axis_values(request.display_axis), dtype=float)
    low, high = sorted((float(request.selected_range[0]), float(request.selected_range[1])))
    mask = np.isfinite(x_values) & (x_values >= low) & (x_values <= high)
    if int(np.count_nonzero(mask)) < 8:
        raise ValueError("Selected range contains fewer than 8 data points.")

    profile_key = settings.pattern_profile_key if settings.fit_scope == FIT_SCOPE_PATTERN else settings.peak_profile_key
    fit_axis = AXIS_TOF if is_gsas_tof_profile(profile_key) else request.display_axis
    fit_x_values = convert_axis_values(x_values, request.display_axis, fit_axis, spectrum)
    e_values = spectrum.e[mask] if settings.use_uncertainties and spectrum.e is not None else None
    initial_profile_parameters = initial_profile_parameters_from_metadata(spectrum.metadata, profile_key)
    if use_gsas_convergence is None:
        use_gsas_convergence = bool(enable_stagnation_detection)
    convergence_relative_improvement = fitting_settings_min_dm_over_m(settings) if use_gsas_convergence else 0.0
    fit_axis_values_in_range = fit_x_values[mask]
    expected_peak_reflections = expected_phase_reflections(
        request.phase,
        fit_axis,
        spectrum.calibration,
        (float(np.min(fit_axis_values_in_range)), float(np.max(fit_axis_values_in_range))),
        reflection_margin_percent=settings.pawley_reflection_margin_percent,
    )
    expected_peak_positions = None if expected_peak_reflections is None else [
        position for position, _reflection in expected_peak_reflections
    ]

    if settings.fit_scope == FIT_SCOPE_PATTERN:
        if request.phase is None:
            raise ValueError("Load a GSAS EXP phase before whole-pattern fitting.")
        def pawley_checkpoint(fit_x, fit_y, observed_y, stage: str) -> None:
            if checkpoint_callback is None:
                return
            curve_x = fit_x
            if fit_axis != request.display_axis:
                curve_x = convert_axis_values(fit_x, fit_axis, request.display_axis, spectrum)
            checkpoint_callback(
                {
                    "spectrum_index": request.spectrum_index,
                    "axis": request.display_axis,
                    "data_mode": request.data_mode,
                    "x": np.array(curve_x, dtype=float, copy=True),
                    "fit_y": np.array(fit_y, dtype=float, copy=True),
                    "observed_y": np.array(observed_y, dtype=float, copy=True),
                    "weighted_residual": None,
                    "label": f"Partial Pawley fit ({stage})",
                }
            )

        result: FitResult = fit_pawley(
            fit_x_values[mask],
            spectrum.y[mask],
            e_values,
            request.phase,
            fit_axis,
            spectrum.calibration,
            polynomial_order=settings.polynomial_order,
            lattice_tolerance_percent=settings.pawley_lattice_tolerance_percent,
            reflection_margin_percent=settings.pawley_reflection_margin_percent,
            eta_initial=settings.pawley_eta_initial,
            eta_bounds=(settings.pawley_eta_min, settings.pawley_eta_max),
            fwhm_min_fraction=settings.pawley_fwhm_min_fraction,
            fwhm_max_fraction=settings.pawley_fwhm_max_fraction,
            max_nfev=settings.max_evaluations,
            profile_key=profile_key,
            width_model=settings.pawley_width_model,
            abort_check=abort_check,
            convergence_relative_improvement=convergence_relative_improvement,
            checkpoint_callback=pawley_checkpoint if checkpoint_callback is not None else None,
            initial_profile_parameters=initial_profile_parameters,
        )
    else:
        result = fit_peak_profile(
            profile_key,
            fit_x_values[mask],
            spectrum.y[mask],
            e_values,
            polynomial_order=settings.polynomial_order,
            eta_initial=settings.pseudo_voigt_eta_initial,
            eta_bounds=(settings.pseudo_voigt_eta_min, settings.pseudo_voigt_eta_max),
            fwhm_min_fraction=settings.pseudo_voigt_fwhm_min_fraction,
            fwhm_max_multiplier=settings.pseudo_voigt_fwhm_max_multiplier,
            maxfev=settings.max_evaluations,
            abort_check=abort_check,
            initial_profile_parameters=initial_profile_parameters,
            convergence_relative_improvement=convergence_relative_improvement,
            expected_peak_positions=expected_peak_positions,
            expected_peak_reflections=None
            if expected_peak_reflections is None
            else [reflection for _position, reflection in expected_peak_reflections],
        )

    curve_x = result.fit_x
    if curve_x is not None and fit_axis != request.display_axis:
        curve_x = convert_axis_values(curve_x, fit_axis, request.display_axis, spectrum)

    return SpectrumFitRecord(
        spectrum_index=request.spectrum_index,
        spectrum=spectrum,
        settings=settings,
        display_axis=request.display_axis,
        fit_axis=fit_axis,
        data_mode=request.data_mode,
        selected_range=(low, high),
        result=result,
        curve_x=None if curve_x is None else np.array(curve_x, dtype=float, copy=True),
        fit_y=None if result.fit_y is None else np.array(result.fit_y, dtype=float, copy=True),
        observed_y=None if result.observed_y is None else np.array(result.observed_y, dtype=float, copy=True),
    )
