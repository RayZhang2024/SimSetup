from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .fitting import _clean_fit_inputs, _fit_quality, _normalised_axis, _polyval
from .gsas_exp import generate_fcc_reflections, validate_ni_cubic_phase
from .models import (
    AXIS_D_SPACING,
    AXIS_TOF,
    PAWLEY_WIDTH_CONSTANT,
    PAWLEY_WIDTH_D_RESOLUTION,
    InstrumentCalibration,
    PawleyFitResult,
    PawleyReflectionResult,
    PhaseModel,
    d_to_tof,
    tof_to_d,
)
from .profiles import (
    PROFILE_GAUSSIAN,
    PROFILE_LORENTZIAN,
    PROFILE_PSEUDO_VOIGT,
    evaluate_peak_profile,
    is_gsas_tof_profile,
    peak_profile_spec,
)


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


def _reflection_d_spacing(h: int, k: int, l: int, lattice_a: float) -> float:
    return float(lattice_a) / float(np.sqrt(h * h + k * k + l * l))


def _d_width_to_axis_width(
    d_spacing: float,
    width_d: float,
    axis: str,
    calibration: Optional[InstrumentCalibration],
) -> float:
    if axis == AXIS_D_SPACING:
        return abs(float(width_d))
    if calibration is None:
        raise ValueError("Pawley fitting on a TOF axis requires calibration metadata.")
    half_width = 0.5 * abs(float(width_d))
    edge_tof = d_to_tof(np.array([float(d_spacing) - half_width, float(d_spacing) + half_width]), calibration)
    return abs(float(edge_tof[1] - edge_tof[0]))


def _d_shift_to_axis_shift(
    d_spacing: float,
    shift_d: float,
    axis: str,
    calibration: Optional[InstrumentCalibration],
) -> float:
    if axis == AXIS_D_SPACING:
        return float(shift_d)
    if calibration is None:
        raise ValueError("Pawley fitting on a TOF axis requires calibration metadata.")
    shifted_d = max(float(d_spacing) + float(shift_d), 1e-12)
    return float(d_to_tof(shifted_d, calibration) - d_to_tof(float(d_spacing), calibration))


def _cubic_hkl_broadening_factor(h: int, k: int, l: int) -> float:
    squared_sum = float(h * h + k * k + l * l)
    if squared_sum <= 0.0:
        return 1.0
    quartic_sum = float(h**4 + k**4 + l**4)
    return float(np.sqrt(max(quartic_sum / (squared_sum * squared_sum), 0.0)))


def _least_squares_parameter_uncertainties(result: object, parameter_count: int) -> Optional[np.ndarray]:
    residual = np.asarray(getattr(result, "fun", []), dtype=float)
    jacobian = np.asarray(getattr(result, "jac", []), dtype=float)
    if residual.size <= int(parameter_count) or jacobian.shape != (residual.size, int(parameter_count)):
        return None
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
        return None
    dof = residual.size - int(parameter_count)
    if dof <= 0:
        return None
    try:
        covariance = np.linalg.pinv(jacobian.T @ jacobian) * (float(np.sum(residual * residual)) / float(dof))
    except np.linalg.LinAlgError:
        return None
    diagonal = np.diag(covariance)
    if diagonal.size != int(parameter_count) or not np.all(np.isfinite(diagonal)):
        return None
    return np.sqrt(np.maximum(diagonal, 0.0))


def fit_pawley(
    x: Sequence[float],
    y: Sequence[float],
    e: Optional[Sequence[float]],
    phase: PhaseModel,
    axis: str,
    calibration: Optional[InstrumentCalibration],
    polynomial_order: int = 2,
    lattice_tolerance_percent: float = 2.0,
    reflection_margin_percent: float = 4.0,
    eta_initial: float = 0.5,
    eta_bounds: tuple[float, float] = (0.0, 1.0),
    fwhm_min_fraction: float = 0.0001,
    fwhm_max_fraction: float = 0.5,
    max_nfev: int = 50000,
    profile_key: str = PROFILE_PSEUDO_VOIGT,
    width_model: str = PAWLEY_WIDTH_CONSTANT,
) -> PawleyFitResult:
    from scipy.optimize import least_squares

    profile = peak_profile_spec(profile_key)
    width_model = str(width_model or PAWLEY_WIDTH_CONSTANT)
    if width_model not in {PAWLEY_WIDTH_CONSTANT, PAWLEY_WIDTH_D_RESOLUTION}:
        raise ValueError(f"Unsupported Pawley width model: {width_model}")
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

    margin = max(float(reflection_margin_percent), 0.0) / 100.0 * max(abs(d_bounds[1] - d_bounds[0]), 1e-6)
    reflections = generate_fcc_reflections(phase, d_bounds[0] - margin, d_bounds[1] + margin)
    if not reflections:
        raise ValueError("No Ni FCC reflections are expected in the selected range.")
    if is_gsas_tof_profile(profile.key) and axis != AXIS_TOF:
        raise ValueError("GSAS TOF profile functions require fitting on the TOF axis.")

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
    eta_min, eta_max = sorted((float(eta_bounds[0]), float(eta_bounds[1])))
    eta_initial = min(max(float(eta_initial), eta_min), eta_max)
    lattice_fraction = max(float(lattice_tolerance_percent), 1e-9) / 100.0
    fwhm_min = max(span * max(float(fwhm_min_fraction), 1e-9), 1e-12)
    fwhm_max = max(span * max(float(fwhm_max_fraction), 1e-6), fwhm_min * 1.001)
    fwhm_guess = min(max(float(fwhm_guess), fwhm_min), fwhm_max)
    tail_guess = max(fwhm_guess, span * 0.01, fwhm_min)
    tail_min = max(fwhm_min, span * 1e-5)
    tail_max = max(span * 2.0, tail_min * 1.001)

    d_centre = 0.5 * (float(d_bounds[0]) + float(d_bounds[1]))
    d_half_span = max(0.5 * abs(float(d_bounds[1]) - float(d_bounds[0])), 1e-9)
    macro_terms = []
    macro_term_by_hkl: dict[tuple[int, int, int], float] = {}
    for reflection in reflections:
        h2_sum = float(reflection.h * reflection.h + reflection.k * reflection.k + reflection.l * reflection.l)
        macro_term = h2_sum * float(reflection.d_spacing) ** 3
        macro_terms.append(macro_term)
        macro_term_by_hkl[(reflection.h, reflection.k, reflection.l)] = macro_term
    macro_term_centre = float(np.mean(macro_terms)) if macro_terms else 0.0
    macro_term_half_span = max(0.5 * (max(macro_terms) - min(macro_terms)) if macro_terms else 0.0, 1e-12)
    shape_names = ["fwhm"] if width_model == PAWLEY_WIDTH_CONSTANT else ["fwhm0_d", "fwhm_d_slope"]
    if width_model == PAWLEY_WIDTH_CONSTANT:
        shape_p0 = [fwhm_guess]
        shape_lower = [fwhm_min]
        shape_upper = [fwhm_max]
    else:
        if axis == AXIS_D_SPACING:
            fwhm0_guess_d = fwhm_guess
            fwhm_min_d = fwhm_min
            fwhm_max_d = fwhm_max
        else:
            if calibration is None:
                raise ValueError("Pawley fitting on a TOF axis requires calibration metadata.")
            difc_scale = max(abs(float(calibration.difc) + 2.0 * float(calibration.difa) * d_centre), 1e-9)
            fwhm0_guess_d = fwhm_guess / difc_scale
            fwhm_min_d = fwhm_min / difc_scale
            fwhm_max_d = fwhm_max / difc_scale
        fwhm0_guess_d = min(max(float(fwhm0_guess_d), fwhm_min_d), fwhm_max_d)
        shape_p0 = [fwhm0_guess_d, fwhm0_guess_d * 0.25]
        shape_lower = [fwhm_min_d, 0.0]
        shape_upper = [fwhm_max_d, max(fwhm_max_d * 4.0, fwhm_min_d * 1.001)]
    if profile.uses_eta:
        shape_names.append("eta")
        shape_p0.append(eta_initial)
        shape_lower.append(eta_min)
        shape_upper.append(eta_max)
    if profile.uses_tail:
        shape_names.append("tail")
        shape_p0.append(tail_guess)
        shape_lower.append(tail_min)
        shape_upper.append(tail_max)
    if profile.uses_back_to_back:
        shape_names.extend(["alpha", "beta"])
        shape_p0.extend([tail_guess, tail_guess])
        shape_lower.extend([tail_min, tail_min])
        shape_upper.extend([tail_max, tail_max])
    if profile.uses_hkl_broadening:
        shape_names.append("hkl_broadening")
        shape_p0.append(0.001)
        shape_lower.append(0.0)
        shape_upper.append(0.05)
    refine_position_shift = profile.uses_position_shift and len(reflections) > 1
    if refine_position_shift:
        shape_names.append("macrostrain")
        shape_p0.append(0.0)
        shape_lower.append(-0.005)
        shape_upper.append(0.005)

    p0 = [phase.a] + shape_p0 + intensity_guesses + [background_guess] + [0.0] * order
    lower = [phase.a * (1.0 - lattice_fraction)] + shape_lower + [0.0] * len(reflections) + [-np.inf] * (order + 1)
    upper = [phase.a * (1.0 + lattice_fraction)] + shape_upper + [np.inf] * len(reflections) + [np.inf] * (order + 1)

    def reflection_fwhm(shape_values: dict[str, float], lattice_a: float, reflection_hkl: tuple[int, int, int]) -> float:
        d_spacing = _reflection_d_spacing(*reflection_hkl, lattice_a)
        if width_model == PAWLEY_WIDTH_CONSTANT:
            base_width = float(shape_values["fwhm"])
        else:
            d_scaled = (d_spacing - d_centre) / d_half_span
            fwhm_d = float(np.sqrt(shape_values["fwhm0_d"] * shape_values["fwhm0_d"] + (shape_values["fwhm_d_slope"] * d_scaled) ** 2))
            base_width = _d_width_to_axis_width(d_spacing, fwhm_d, axis, calibration)
        if profile.uses_hkl_broadening:
            broadening = max(float(shape_values.get("hkl_broadening", 0.0)), 0.0)
            hkl_factor = _cubic_hkl_broadening_factor(*reflection_hkl)
            extra_width_d = broadening * d_spacing * hkl_factor
            extra_width = _d_width_to_axis_width(d_spacing, extra_width_d, axis, calibration)
            return float(np.sqrt(base_width * base_width + extra_width * extra_width))
        return base_width

    def reflection_position(shape_values: dict[str, float], lattice_a: float, reflection_hkl: tuple[int, int, int]) -> float:
        h, k, l = reflection_hkl
        position = _reflection_axis_position(h, k, l, lattice_a, axis, calibration)
        if refine_position_shift:
            d_spacing = _reflection_d_spacing(h, k, l, lattice_a)
            macro_term = macro_term_by_hkl.get((h, k, l), macro_term_centre)
            macro_normalised = (macro_term - macro_term_centre) / macro_term_half_span
            shift_d = float(shape_values.get("macrostrain", 0.0)) * d_spacing * macro_normalised
            position += _d_shift_to_axis_shift(d_spacing, shift_d, axis, calibration)
        return float(position)

    def model(params: np.ndarray) -> np.ndarray:
        lattice_a = float(params[0])
        shape_values = {name: float(value) for name, value in zip(shape_names, params[1 : 1 + len(shape_names)])}
        intensities_start = 1 + len(shape_names)
        bg_start = intensities_start + len(reflections)
        intensities = params[intensities_start:bg_start]
        bg = params[bg_start:]
        output = _polyval(bg, x_norm)
        for reflection, intensity in zip(reflections, intensities):
            hkl = (reflection.h, reflection.k, reflection.l)
            position = reflection_position(shape_values, lattice_a, hkl)
            fwhm_value = reflection_fwhm(shape_values, lattice_a, hkl)
            profile_y = evaluate_peak_profile(
                profile.key,
                x_values,
                position,
                fwhm_value,
                eta=shape_values.get(
                    "eta",
                    0.0 if profile.key == PROFILE_GAUSSIAN else 1.0 if profile.key == PROFILE_LORENTZIAN else eta_initial,
                ),
                tail=shape_values.get("tail", tail_guess),
                alpha=shape_values.get("alpha", tail_guess),
                beta=shape_values.get("beta", tail_guess),
            )
            output = output + float(intensity) * profile_y
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
        max_nfev=max(100, int(max_nfev)),
    )
    if not result.success:
        raise ValueError(f"Pawley fit failed: {result.message}")

    params = result.x
    y_fit = model(params)
    lattice_a = float(params[0])
    uncertainties = _least_squares_parameter_uncertainties(result, len(params))
    lattice_a_uncertainty = float(uncertainties[0]) if uncertainties is not None else None
    shape_values = {name: float(value) for name, value in zip(shape_names, params[1 : 1 + len(shape_names)])}
    if width_model == PAWLEY_WIDTH_CONSTANT:
        fwhm = shape_values["fwhm"]
    else:
        fwhm = _d_width_to_axis_width(d_centre, shape_values["fwhm0_d"], axis, calibration)
    eta = shape_values.get("eta", 0.0 if profile.key == PROFILE_GAUSSIAN else 1.0 if profile.key == PROFILE_LORENTZIAN else eta_initial)
    intensities_start = 1 + len(shape_names)
    bg_start = intensities_start + len(reflections)
    intensities = params[intensities_start:bg_start]
    bg = params[bg_start:]
    reflection_results = []
    for reflection, intensity in zip(reflections, intensities):
        position = reflection_position(shape_values, lattice_a, (reflection.h, reflection.k, reflection.l))
        reflection_results.append(
            PawleyReflectionResult(
                reflection=reflection,
                position=float(position),
                intensity=float(intensity),
            )
        )
    quality = _fit_quality(y_values, y_fit, e_values, len(params))
    profile_parameters = {key: float(value) for key, value in shape_values.items() if key != "fwhm"}
    profile_parameters["reflection_count"] = float(len(reflections))
    profile_parameters["width_model"] = 0.0 if width_model == PAWLEY_WIDTH_CONSTANT else 1.0
    if width_model == PAWLEY_WIDTH_D_RESOLUTION:
        profile_parameters["width_d_centre"] = float(d_centre)
        profile_parameters["width_d_half_span"] = float(d_half_span)
    if profile.gsas_tof_function is not None:
        profile_parameters["gsas_tof_function"] = float(profile.gsas_tof_function)
    return PawleyFitResult(
        lattice_a=lattice_a,
        lattice_a_uncertainty=lattice_a_uncertainty,
        fwhm=fwhm,
        eta=eta,
        background_coefficients=tuple(float(value) for value in bg),
        reflections=tuple(reflection_results),
        quality=quality,
        profile_key=profile.key,
        profile_name=profile.label,
        profile_parameters=profile_parameters,
        fit_x=np.array(x_values, dtype=float, copy=True),
        fit_y=np.array(y_fit, dtype=float, copy=True),
        observed_y=np.array(y_values, dtype=float, copy=True),
    )
