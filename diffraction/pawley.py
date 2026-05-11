from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .fitting import _clean_fit_inputs, _fit_quality, _normalised_axis, _polyval, pseudo_voigt_profile
from .gsas_exp import generate_fcc_reflections, validate_ni_cubic_phase
from .models import AXIS_D_SPACING, AXIS_TOF, InstrumentCalibration, PawleyFitResult, PawleyReflectionResult, PhaseModel, d_to_tof, tof_to_d


def _reflection_axis_position(
    h: int,
    k: int,
    l: int,
    lattice_a: float,
    axis: str,
    calibration: Optional[InstrumentCalibration],
) -> float:
    d_spacing = float(lattice_a) / float(np.sqrt(h * h + k * k + l * l))
    if axis == AXIS_D_SPACING:
        return d_spacing
    if calibration is None:
        raise ValueError("Pawley fitting on a TOF axis requires calibration metadata.")
    return float(d_to_tof(d_spacing, calibration))


def fit_pawley(
    x: Sequence[float],
    y: Sequence[float],
    e: Optional[Sequence[float]],
    phase: PhaseModel,
    axis: str,
    calibration: Optional[InstrumentCalibration],
    polynomial_order: int = 2,
) -> PawleyFitResult:
    from scipy.optimize import least_squares

    validate_ni_cubic_phase(phase)
    x_values, y_values, e_values = _clean_fit_inputs(x, y, e)
    axis_min = float(np.min(x_values))
    axis_max = float(np.max(x_values))
    if axis == AXIS_D_SPACING:
        d_bounds = (axis_min, axis_max)
    elif axis == AXIS_TOF:
        if calibration is None:
            raise ValueError("Pawley fitting on a TOF axis requires calibration metadata.")
        d_axis = tof_to_d(np.array([axis_min, axis_max], dtype=float), calibration)
        d_bounds = (float(np.min(d_axis)), float(np.max(d_axis)))
    else:
        raise ValueError(f"Unsupported Pawley axis: {axis}")

    margin = 0.04 * max(abs(d_bounds[1] - d_bounds[0]), 1e-6)
    reflections = generate_fcc_reflections(phase, d_bounds[0] - margin, d_bounds[1] + margin)
    if not reflections:
        raise ValueError("No Ni FCC reflections are expected in the selected range.")

    x_norm, _axis_centre, _axis_half_span = _normalised_axis(x_values)
    background_guess = float(np.percentile(y_values, 20.0))
    span = max(axis_max - axis_min, 1e-9)
    fwhm_guess = max(span / max(20.0, 8.0 * len(reflections)), span / 1000.0)
    peak_signal = np.maximum(y_values - background_guess, 0.0)
    intensity_guesses = []
    for reflection in reflections:
        position = _reflection_axis_position(reflection.h, reflection.k, reflection.l, phase.a, axis, calibration)
        if axis_min <= position <= axis_max:
            intensity_guesses.append(float(np.interp(position, x_values, peak_signal)))
        else:
            intensity_guesses.append(0.0)
    if max(intensity_guesses, default=0.0) <= 0.0:
        intensity_guesses = [max(float(np.max(peak_signal)), 1.0) for _ in reflections]

    order = max(0, min(int(polynomial_order), 5))
    p0 = [phase.a, fwhm_guess, 0.5] + intensity_guesses + [background_guess] + [0.0] * order
    lower = [phase.a * 0.98, span / 10000.0, 0.0] + [0.0] * len(reflections) + [-np.inf] * (order + 1)
    upper = [phase.a * 1.02, span / 2.0, 1.0] + [np.inf] * len(reflections) + [np.inf] * (order + 1)

    def model(params: np.ndarray) -> np.ndarray:
        lattice_a = float(params[0])
        fwhm = float(params[1])
        eta = float(params[2])
        intensities = params[3 : 3 + len(reflections)]
        bg = params[3 + len(reflections) :]
        output = _polyval(bg, x_norm)
        for reflection, intensity in zip(reflections, intensities):
            position = _reflection_axis_position(reflection.h, reflection.k, reflection.l, lattice_a, axis, calibration)
            output = output + float(intensity) * pseudo_voigt_profile(x_values, position, fwhm, eta)
        return output

    def residual(params: np.ndarray) -> np.ndarray:
        values = y_values - model(params)
        if e_values is not None:
            values = values / e_values
        return values

    result = least_squares(
        residual,
        np.asarray(p0, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        max_nfev=50000,
    )
    if not result.success:
        raise ValueError(f"Pawley fit failed: {result.message}")

    params = result.x
    y_fit = model(params)
    lattice_a = float(params[0])
    fwhm = float(params[1])
    eta = float(params[2])
    intensities = params[3 : 3 + len(reflections)]
    bg = params[3 + len(reflections) :]
    reflection_results = []
    for reflection, intensity in zip(reflections, intensities):
        position = _reflection_axis_position(reflection.h, reflection.k, reflection.l, lattice_a, axis, calibration)
        reflection_results.append(
            PawleyReflectionResult(
                reflection=reflection,
                position=float(position),
                intensity=float(intensity),
            )
        )
    quality = _fit_quality(y_values, y_fit, e_values, len(params))
    return PawleyFitResult(
        lattice_a=lattice_a,
        fwhm=fwhm,
        eta=eta,
        background_coefficients=tuple(float(value) for value in bg),
        reflections=tuple(reflection_results),
        quality=quality,
        fit_x=np.array(x_values, dtype=float, copy=True),
        fit_y=np.array(y_fit, dtype=float, copy=True),
        observed_y=np.array(y_values, dtype=float, copy=True),
    )
