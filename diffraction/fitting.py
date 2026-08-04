from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .convergence import gsas_relative_chi_square_ftol
from .models import FitQuality, PeakComponentResult, PeakFitResult, Reflection
from .profiles import (
    PROFILE_GAUSSIAN,
    PROFILE_LORENTZIAN,
    PROFILE_PSEUDO_VOIGT,
    back_to_back_tail_bounds,
    evaluate_peak_profile,
    gsas2_tof_effective_fwhm,
    gsas2_tof_profile_initial_bounds,
    peak_profile_spec,
    pseudo_voigt_profile,
)


class FitAbortedError(RuntimeError):
    pass


def _raise_if_aborted(abort_check: Optional[Callable[[], Optional[str]]]) -> None:
    if abort_check is None:
        return
    reason = abort_check()
    if reason:
        raise FitAbortedError(reason)


def _normalised_axis(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    centre = 0.5 * (float(np.min(x)) + float(np.max(x)))
    half_span = max(0.5 * (float(np.max(x)) - float(np.min(x))), 1e-12)
    return (x - centre) / half_span, centre, half_span


def _polyval(coefficients: Sequence[float], x_normalised: np.ndarray) -> np.ndarray:
    output = np.zeros_like(x_normalised, dtype=float)
    for power, coefficient in enumerate(coefficients):
        output += float(coefficient) * np.power(x_normalised, power)
    return output


def _fit_quality(y: np.ndarray, y_fit: np.ndarray, weights: Optional[np.ndarray], parameter_count: int) -> FitQuality:
    residual = y - y_fit
    if weights is None:
        chi_square = float(np.sum(residual * residual))
        denominator = max(float(np.sum(y * y)), 1e-12)
        rwp = 100.0 * math.sqrt(float(np.sum(residual * residual)) / denominator)
    else:
        weighted = residual / weights
        chi_square = float(np.sum(weighted * weighted))
        denominator = max(float(np.sum((y / weights) * (y / weights))), 1e-12)
        rwp = 100.0 * math.sqrt(float(np.sum(weighted * weighted)) / denominator)
    dof = max(int(y.size) - int(parameter_count), 1)
    return FitQuality(
        chi_square=chi_square,
        reduced_chi_square=chi_square / float(dof),
        rwp_percent=float(rwp),
        points=int(y.size),
    )


def _model_based_uncertainty(
    observed: np.ndarray,
    calculated: np.ndarray,
    supplied: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Rescale observed-count errors toward the model and apply a robust floor."""
    if supplied is None:
        return None
    observed_abs = np.abs(np.asarray(observed, dtype=float))
    calculated_abs = np.abs(np.asarray(calculated, dtype=float))
    supplied_values = np.asarray(supplied, dtype=float)
    positive_signal = observed_abs[np.isfinite(observed_abs) & (observed_abs > 0.0)]
    signal_scale = float(np.median(positive_signal)) if positive_signal.size else 1.0
    signal_floor = max(
        float(np.percentile(positive_signal, 10.0)) if positive_signal.size else 0.0,
        signal_scale * 0.05,
        1e-12,
    )
    positive_sigma = supplied_values[np.isfinite(supplied_values) & (supplied_values > 0.0)]
    sigma_floor = max(
        float(np.percentile(positive_sigma, 10.0)) if positive_sigma.size else 0.0,
        1e-12,
    )
    ratio = np.sqrt(np.maximum(calculated_abs, signal_floor) / np.maximum(observed_abs, signal_floor))
    effective = supplied_values * np.clip(ratio, 0.25, 4.0)
    return np.maximum(effective, sigma_floor)


def _clean_fit_inputs(
    x: Sequence[float],
    y: Sequence[float],
    e: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    e_values = None
    if e is not None:
        e_candidate = np.asarray(e, dtype=float)
        mask &= np.isfinite(e_candidate) & (e_candidate > 0.0)
        e_values = e_candidate
    x_values = x_values[mask]
    y_values = y_values[mask]
    if e_values is not None:
        e_values = e_values[mask]
    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    if e_values is not None:
        e_values = e_values[order]
    if x_values.size < 8:
        raise ValueError("At least 8 finite data points are required for fitting.")
    return x_values, y_values, e_values


def _initial_fwhm(x: np.ndarray, y: np.ndarray, background: float, peak_index: Optional[int] = None) -> float:
    if peak_index is None:
        peak_index = int(np.argmax(y))
    peak_index = max(0, min(int(peak_index), int(y.size) - 1))
    span = max(float(np.max(x)) - float(np.min(x)), 1e-9)
    peak_height = max(float(y[peak_index] - background), 1e-12)
    half_height = background + 0.5 * peak_height
    left = peak_index
    while left > 0 and y[left] >= half_height:
        left -= 1
    right = peak_index
    while right < y.size - 1 and y[right] >= half_height:
        right += 1
    if right > left:
        return max(abs(float(x[right] - x[left])), span / 10000.0)
    return max(span / 100.0, 1e-9)


def _detect_peak_indices(x: np.ndarray, y: np.ndarray, background: float, max_peaks: int = 12) -> list[int]:
    from scipy.signal import find_peaks

    if int(max_peaks) <= 1:
        return [int(np.argmax(y))]
    signal = np.asarray(y, dtype=float) - float(background)
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return [int(np.argmax(y))]
    distance = max(3, int(x.size / max(max_peaks * 4, 1)))
    peak_indices, properties = find_peaks(signal, distance=distance)
    if peak_indices.size == 0:
        return [int(np.argmax(y))]
    prominences = properties.get("prominences")
    if prominences is None:
        peak_indices, properties = find_peaks(signal, distance=distance, prominence=0.0)
        prominences = properties.get("prominences", np.zeros(peak_indices.size, dtype=float))
    if peak_indices.size == 0:
        return [int(np.argmax(y))]
    max_prominence = float(np.max(prominences)) if prominences.size else 0.0
    if max_prominence > 0.0:
        keep = prominences >= max_prominence * 0.35
        peak_indices = peak_indices[keep]
        prominences = prominences[keep]
    if peak_indices.size == 0:
        return [int(np.argmax(y))]
    order = np.argsort(prominences)[::-1][:max(1, int(max_peaks))]
    selected = sorted(int(index) for index in peak_indices[order])
    if int(np.argmax(y)) not in selected:
        selected.append(int(np.argmax(y)))
        selected = sorted(selected)
    return selected[:max(1, int(max_peaks))]


def _expected_peak_centres(
    x: np.ndarray,
    y: np.ndarray,
    background: float,
    expected_peak_positions: Optional[Sequence[float]],
    max_peaks: int,
) -> list[float]:
    return [
        centre
        for centre, _reflection, _expected_position, _lower_bound, _upper_bound in _expected_peak_candidates(
            x,
            y,
            background,
            expected_peak_positions,
            None,
            max_peaks,
        )
    ]


def _expected_peak_candidates(
    x: np.ndarray,
    y: np.ndarray,
    background: float,
    expected_peak_positions: Optional[Sequence[float]],
    expected_peak_reflections: Optional[Sequence[Reflection]],
    max_peaks: int,
) -> list[tuple[float, Optional[Reflection], float, float, float]]:
    if expected_peak_positions is None:
        return []
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x.size > 1:
        axis_step = float(np.median(np.abs(np.diff(np.sort(x)))))
    else:
        axis_step = 0.0
    merge_tolerance = max(axis_step * 2.0, abs(x_max - x_min) * 1e-9, 1e-12)
    expected_items: list[tuple[float, Optional[Reflection]]] = []
    for index, item in enumerate(expected_peak_positions):
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        if value < x_min or value > x_max:
            continue
        reflection = (
            None
            if expected_peak_reflections is None or index >= len(expected_peak_reflections)
            else expected_peak_reflections[index]
        )
        if expected_items and abs(value - expected_items[-1][0]) <= merge_tolerance:
            continue
        expected_items.append((value, reflection))
    expected_items = sorted(expected_items, key=lambda item: item[0])
    expected_centres = [position for position, _reflection in expected_items]
    limit = max(1, int(max_peaks))
    if expected_peak_reflections is not None:
        limit = min(len(expected_items), max(limit, 48))
    if not expected_centres:
        return []
    scored = []
    signal = np.asarray(y, dtype=float) - float(background)
    step = float(np.median(np.abs(np.diff(np.sort(x))))) if x.size > 1 else 0.0
    for index, (centre, reflection) in enumerate(expected_items):
        left_boundary = x_min if index == 0 else 0.5 * (expected_items[index - 1][0] + centre)
        right_boundary = x_max if index == len(expected_items) - 1 else 0.5 * (centre + expected_items[index + 1][0])
        min_half_width = max(10.0 * step, merge_tolerance)
        left_boundary = min(left_boundary, centre - min_half_width)
        right_boundary = max(right_boundary, centre + min_half_width)
        mask = (x >= left_boundary) & (x <= right_boundary)
        if int(np.count_nonzero(mask)) >= 3:
            local_x = x[mask]
            local_signal = signal[mask]
            local_index = int(np.argmax(local_signal))
            candidate_centre = float(local_x[local_index])
            score = float(local_signal[local_index])
        else:
            nearest = int(np.argmin(np.abs(x - centre)))
            score = float(signal[nearest])
        initial_centre = min(max(float(centre), left_boundary), right_boundary)
        scored.append((score, initial_centre, reflection, centre, left_boundary, right_boundary))
    if len(scored) <= limit:
        return [
            (centre, reflection, expected_position, lower_bound, upper_bound)
            for _score, centre, reflection, expected_position, lower_bound, upper_bound
            in sorted(scored, key=lambda item: item[3])
        ]
    strongest = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    return [
        (centre, reflection, expected_position, lower_bound, upper_bound)
        for _score, centre, reflection, expected_position, lower_bound, upper_bound
        in sorted(strongest, key=lambda item: item[3])
    ]


def _background_coefficients(x: np.ndarray, y: np.ndarray, order: int) -> tuple[float, ...]:
    x_norm, _centre, _half_span = _normalised_axis(x)
    if int(order) <= 0:
        return (float(np.percentile(y, 20.0)),)
    cutoff = float(np.percentile(y, 65.0))
    mask = np.isfinite(x_norm) & np.isfinite(y) & (y <= cutoff)
    if int(np.count_nonzero(mask)) < int(order) + 1:
        mask = np.isfinite(x_norm) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < int(order) + 1:
        return (float(np.percentile(y, 20.0)),) + (0.0,) * int(order)
    coefficients = np.polynomial.polynomial.polyfit(x_norm[mask], y[mask], int(order))
    return tuple(float(value) for value in coefficients)


def _fit_detected_peak_components(
    profile_key: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    e_values: Optional[np.ndarray],
    peak_indices: list[int],
    polynomial_order: int,
    eta_initial: float,
    eta_bounds: tuple[float, float],
    fwhm_min_fraction: float,
    fwhm_max_multiplier: float,
    maxfev: int,
    abort_check: Optional[Callable[[], Optional[str]]],
    model_based_weighting: bool,
    initial_profile_parameters: Optional[Mapping[str, float]],
    convergence_relative_improvement: float,
) -> PeakFitResult:
    profile = peak_profile_spec(profile_key)
    background_coefficients = _background_coefficients(x_values, y_values, polynomial_order)
    x_norm, _centre, _half_span = _normalised_axis(x_values)
    y_fit = _polyval(background_coefficients, x_norm)
    component_results = []
    component_profile_parameters: list[Mapping[str, float]] = []
    peak_indices = sorted(peak_indices)
    for index, peak_index in enumerate(peak_indices):
        _raise_if_aborted(abort_check)
        centre_guess = float(x_values[peak_index])
        low = float(np.min(x_values)) if index == 0 else 0.5 * (float(x_values[peak_indices[index - 1]]) + centre_guess)
        high = float(np.max(x_values)) if index == len(peak_indices) - 1 else 0.5 * (centre_guess + float(x_values[peak_indices[index + 1]]))
        local_mask = (x_values >= low) & (x_values <= high)
        if int(np.count_nonzero(local_mask)) < 8:
            continue
        try:
            local_result = fit_peak_profile(
                profile_key,
                x_values[local_mask],
                y_values[local_mask],
                None if e_values is None else e_values[local_mask],
                polynomial_order=polynomial_order,
                eta_initial=eta_initial,
                eta_bounds=eta_bounds,
                fwhm_min_fraction=fwhm_min_fraction,
                fwhm_max_multiplier=fwhm_max_multiplier,
                maxfev=min(max(100, int(maxfev)), 5000),
                max_peaks=1,
                abort_check=abort_check,
                model_based_weighting=model_based_weighting,
                initial_profile_parameters=initial_profile_parameters,
                convergence_relative_improvement=convergence_relative_improvement,
            )
        except FitAbortedError:
            raise
        except Exception:
            continue
        component = local_result.components[0] if local_result.components else PeakComponentResult(
            centre=local_result.centre,
            fwhm=local_result.fwhm,
            height=local_result.height,
            area=local_result.area,
            eta=local_result.eta,
            centre_uncertainty=local_result.centre_uncertainty,
            fwhm_uncertainty=local_result.fwhm_uncertainty,
        )
        component_results.append(component)
        component_profile_parameters.append(local_result.profile_parameters)
        component_profile = evaluate_peak_profile(
            profile.key,
            x_values,
            component.centre,
            component.fwhm,
            eta=component.eta,
            tail=float(local_result.profile_parameters.get("tail", component.fwhm)),
            alpha=float(local_result.profile_parameters.get("alpha", component.fwhm)),
            beta=float(local_result.profile_parameters.get("beta", component.fwhm)),
            sig=local_result.profile_parameters.get("sig"),
            gam=local_result.profile_parameters.get("gam"),
            alp=local_result.profile_parameters.get("alp"),
            bet=local_result.profile_parameters.get("bet"),
        )
        y_fit = y_fit + component.height * component_profile
    if not component_results:
        return fit_peak_profile(
            profile_key,
            x_values,
            y_values,
            e_values,
            polynomial_order=polynomial_order,
            eta_initial=eta_initial,
            eta_bounds=eta_bounds,
            fwhm_min_fraction=fwhm_min_fraction,
            fwhm_max_multiplier=fwhm_max_multiplier,
            maxfev=maxfev,
            max_peaks=1,
            abort_check=abort_check,
            model_based_weighting=model_based_weighting,
            initial_profile_parameters=initial_profile_parameters,
            convergence_relative_improvement=convergence_relative_improvement,
        )
    component_results = sorted(component_results, key=lambda component: component.centre)
    primary_component = max(component_results, key=lambda component: component.height)
    area = float(sum(component.area for component in component_results))
    parameter_count = len(component_results) * 4 + len(background_coefficients)
    fit_uncertainty = _model_based_uncertainty(y_values, y_fit, e_values) if model_based_weighting else e_values
    quality = _fit_quality(y_values, y_fit, fit_uncertainty, parameter_count)
    profile_parameters = {"peak_count": float(len(component_results))}
    averaged_keys = sorted(
        {
            key
            for parameters in component_profile_parameters
            for key, value in parameters.items()
            if key != "peak_count" and isinstance(value, (int, float, np.floating))
        }
    )
    for key in averaged_keys:
        values = [
            float(parameters[key])
            for parameters in component_profile_parameters
            if key in parameters and np.isfinite(float(parameters[key]))
        ]
        if values:
            profile_parameters[key] = float(np.mean(values))
    return PeakFitResult(
        model_name=profile.label,
        centre=primary_component.centre,
        centre_uncertainty=None,
        fwhm=primary_component.fwhm,
        fwhm_uncertainty=None,
        height=primary_component.height,
        area=area,
        eta=primary_component.eta,
        background_coefficients=background_coefficients,
        quality=quality,
        profile_parameters=profile_parameters,
        components=tuple(component_results),
        fit_x=np.array(x_values, dtype=float, copy=True),
        fit_y=np.array(y_fit, dtype=float, copy=True),
        observed_y=np.array(y_values, dtype=float, copy=True),
        fit_uncertainty=None if fit_uncertainty is None else np.array(fit_uncertainty, dtype=float, copy=True),
    )


def fit_pseudo_voigt(
    x: Sequence[float],
    y: Sequence[float],
    e: Optional[Sequence[float]] = None,
    polynomial_order: int = 2,
    eta_initial: float = 0.5,
    eta_bounds: tuple[float, float] = (0.0, 1.0),
    fwhm_min_fraction: float = 0.0001,
    fwhm_max_multiplier: float = 2.0,
    maxfev: int = 50000,
    abort_check: Optional[Callable[[], Optional[str]]] = None,
) -> PeakFitResult:
    return fit_peak_profile(
        PROFILE_PSEUDO_VOIGT,
        x,
        y,
        e,
        polynomial_order=polynomial_order,
        eta_initial=eta_initial,
        eta_bounds=eta_bounds,
        fwhm_min_fraction=fwhm_min_fraction,
        fwhm_max_multiplier=fwhm_max_multiplier,
        maxfev=maxfev,
        max_peaks=1,
        abort_check=abort_check,
    )


def fit_peak_profile(
    profile_key: str,
    x: Sequence[float],
    y: Sequence[float],
    e: Optional[Sequence[float]] = None,
    polynomial_order: int = 2,
    eta_initial: float = 0.5,
    eta_bounds: tuple[float, float] = (0.0, 1.0),
    fwhm_min_fraction: float = 0.0001,
    fwhm_max_multiplier: float = 2.0,
    maxfev: int = 50000,
    max_peaks: int = 12,
    abort_check: Optional[Callable[[], Optional[str]]] = None,
    model_based_weighting: bool = True,
    initial_profile_parameters: Optional[Mapping[str, float]] = None,
    convergence_relative_improvement: float = 0.0,
    expected_peak_positions: Optional[Sequence[float]] = None,
    expected_peak_reflections: Optional[Sequence[Reflection]] = None,
) -> PeakFitResult:
    from scipy.optimize import curve_fit

    profile = peak_profile_spec(profile_key)
    _raise_if_aborted(abort_check)
    x_values, y_values, e_values = _clean_fit_inputs(x, y, e)
    order = max(0, min(int(polynomial_order), 5))
    _, _axis_centre, _axis_half_span = _normalised_axis(x_values)
    background_guess = float(np.percentile(y_values, 20.0))
    expected_candidates = _expected_peak_candidates(
        x_values,
        y_values,
        background_guess,
        expected_peak_positions,
        expected_peak_reflections,
        max_peaks=max_peaks,
    )
    expected_centres = [centre for centre, _reflection, _expected_position, _lower_bound, _upper_bound in expected_candidates]
    component_reflections: list[Optional[Reflection]] = [
        reflection for _centre, reflection, _expected_position, _lower_bound, _upper_bound in expected_candidates
    ]
    component_centre_bounds: list[tuple[float, float]] = [
        (lower_bound, upper_bound)
        for _centre, _reflection, _expected_position, lower_bound, upper_bound in expected_candidates
    ]
    if expected_centres:
        peak_indices = [int(np.argmin(np.abs(x_values - centre))) for centre in expected_centres]
    else:
        peak_indices = _detect_peak_indices(x_values, y_values, background_guess, max_peaks=max_peaks)
    if len(peak_indices) > 1 and not expected_centres:
        return _fit_detected_peak_components(
            profile.key,
            x_values,
            y_values,
            e_values,
            peak_indices,
            order,
            eta_initial,
            eta_bounds,
            fwhm_min_fraction,
            fwhm_max_multiplier,
            maxfev,
            abort_check,
            model_based_weighting,
            initial_profile_parameters,
            convergence_relative_improvement,
        )
    centre_guesses = expected_centres if expected_centres else [float(x_values[index]) for index in peak_indices]
    height_guesses = [max(float(y_values[index] - background_guess), 1e-9) for index in peak_indices]
    fwhm_guesses = [_initial_fwhm(x_values, y_values, background_guess, index) for index in peak_indices]

    span = max(float(np.max(x_values) - np.min(x_values)), 1e-9)
    fwhm_min = max(span * max(float(fwhm_min_fraction), 1e-9), 1e-12)
    fwhm_max = max(span * max(float(fwhm_max_multiplier), 1e-6), fwhm_min * 1.001)
    eta_min, eta_max = sorted((float(eta_bounds[0]), float(eta_bounds[1])))
    eta_initial = min(max(float(eta_initial), eta_min), eta_max)
    fwhm_guess = float(np.median(fwhm_guesses)) if fwhm_guesses else max(span / 100.0, fwhm_min)
    fwhm_guess = min(max(fwhm_guess, fwhm_min), fwhm_max)
    tail_guess, tail_min, tail_max = back_to_back_tail_bounds(span, fwhm_guess, fwhm_min, fwhm_max)
    component_count = len(centre_guesses)
    use_independent_component_fwhm = bool(expected_centres) and component_count > 1 and not profile.uses_gsas2_tof_widths

    if profile.uses_gsas2_tof_widths:
        shape_initial, shape_lower_values, shape_upper_values = gsas2_tof_profile_initial_bounds(
            span, fwhm_guess, fwhm_min, fwhm_max
        )
        shape_names = ["sig", "gam", "alp", "bet"]
        p0 = [shape_initial[name] for name in shape_names]
        lower = [shape_lower_values[name] for name in shape_names]
        upper = [shape_upper_values[name] for name in shape_names]
    else:
        shape_names = []
        p0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        if not use_independent_component_fwhm:
            shape_names.append("fwhm")
            p0.append(fwhm_guess)
            lower.append(fwhm_min)
            upper.append(fwhm_max)
        if profile.uses_eta:
            shape_names.append("eta")
            p0.append(eta_initial)
            lower.append(eta_min)
            upper.append(eta_max)
        if profile.uses_tail:
            shape_names.append("tail")
            p0.append(tail_guess)
            lower.append(tail_min)
            upper.append(tail_max)
        if profile.uses_back_to_back:
            shape_names.extend(["alpha", "beta"])
            p0.extend([tail_guess, tail_guess])
            lower.extend([tail_min, tail_min])
            upper.extend([tail_max, tail_max])

    if initial_profile_parameters:
        for index, name in enumerate(shape_names):
            if name not in initial_profile_parameters:
                continue
            try:
                seeded_value = float(initial_profile_parameters[name])
            except (TypeError, ValueError):
                continue
            p0[index] = min(max(seeded_value, lower[index]), upper[index])
        if use_independent_component_fwhm and "fwhm" in initial_profile_parameters:
            try:
                seeded_width = float(initial_profile_parameters["fwhm"])
            except (TypeError, ValueError):
                seeded_width = fwhm_guess
            fwhm_guess = min(max(seeded_width, fwhm_min), fwhm_max)

    for index, centre_guess in enumerate(centre_guesses):
        if expected_centres and index < len(component_centre_bounds):
            centre_min, centre_max = component_centre_bounds[index]
            if centre_max <= centre_min:
                centre_min = max(float(np.min(x_values)), float(centre_guess) - 1e-12)
                centre_max = min(float(np.max(x_values)), float(centre_guess) + 1e-12)
            centre_guess = min(max(float(centre_guess), centre_min), centre_max)
        elif expected_centres:
            if len(centre_guesses) == 1:
                neighbour_half_distance = span
            elif index == 0:
                neighbour_half_distance = 0.5 * abs(centre_guesses[index + 1] - centre_guess)
            elif index == len(centre_guesses) - 1:
                neighbour_half_distance = 0.5 * abs(centre_guess - centre_guesses[index - 1])
            else:
                neighbour_half_distance = 0.5 * min(
                    abs(centre_guess - centre_guesses[index - 1]),
                    abs(centre_guesses[index + 1] - centre_guess),
                )
            centre_window = max(
                5.0 * (float(np.median(np.abs(np.diff(np.sort(x_values))))) if x_values.size > 1 else 0.0),
                min(neighbour_half_distance * 0.9, span * 0.05, max(abs(centre_guess) * 0.05, span * 0.01)),
                1e-12,
            )
            centre_min = max(float(np.min(x_values)), float(centre_guess) - centre_window)
            centre_max = min(float(np.max(x_values)), float(centre_guess) + centre_window)
        elif len(centre_guesses) == 1:
            centre_min = float(np.min(x_values))
            centre_max = float(np.max(x_values))
        else:
            centre_min = float(np.min(x_values)) if index == 0 else 0.5 * (centre_guesses[index - 1] + centre_guess)
            centre_max = float(np.max(x_values)) if index == len(centre_guesses) - 1 else 0.5 * (centre_guess + centre_guesses[index + 1])
        p0.append(centre_guess)
        lower.append(centre_min)
        upper.append(centre_max)

    if use_independent_component_fwhm:
        for width_guess in fwhm_guesses:
            p0.append(min(max(float(width_guess), fwhm_min), fwhm_max))
            lower.append(fwhm_min)
            upper.append(fwhm_max)

    max_height = max(height_guesses + [float(np.max(y_values) - background_guess), 1.0])
    for height_guess in height_guesses:
        p0.append(height_guess)
        lower.append(0.0)
        upper.append(max(max_height * 100.0, 1.0))

    p0.extend([background_guess] + [0.0] * order)
    lower.extend([-np.inf] * (order + 1))
    upper.extend([np.inf] * (order + 1))

    def model(x_input: np.ndarray, *params: float) -> np.ndarray:
        _raise_if_aborted(abort_check)
        shape = {name: float(value) for name, value in zip(shape_names, params[: len(shape_names)])}
        centre_start = len(shape_names)
        width_start = centre_start + component_count
        height_start = width_start + component_count if use_independent_component_fwhm else width_start
        bg_start = height_start + component_count
        centres = params[centre_start:height_start]
        if use_independent_component_fwhm:
            centres = params[centre_start:width_start]
            fwhms = params[width_start:height_start]
        else:
            centres = params[centre_start:height_start]
            fwhms = [shape.get("fwhm", fwhm_guess)] * component_count
        heights = params[height_start:bg_start]
        x_input_norm = (x_input - _axis_centre) / _axis_half_span
        bg = params[bg_start:]
        output = _polyval(bg, x_input_norm)
        for centre, fwhm_value, height in zip(centres, fwhms, heights):
            profile_y = evaluate_peak_profile(
                profile.key,
                x_input,
                float(centre),
                float(fwhm_value),
                eta=shape.get("eta", 0.0 if profile.key == PROFILE_GAUSSIAN else 1.0 if profile.key == PROFILE_LORENTZIAN else eta_initial),
                tail=shape.get("tail", tail_guess),
                alpha=shape.get("alpha", tail_guess),
                beta=shape.get("beta", tail_guess),
                sig=shape.get("sig"),
                gam=shape.get("gam"),
                alp=shape.get("alp"),
                bet=shape.get("bet"),
            )
            output = output + float(height) * profile_y
        return output

    sigma = None if model_based_weighting else e_values
    curve_fit_kwargs = {
        "p0": p0,
        "bounds": (lower, upper),
        "sigma": sigma,
        "absolute_sigma": sigma is not None,
        "maxfev": max(100, int(maxfev)),
    }
    ftol = gsas_relative_chi_square_ftol(convergence_relative_improvement)
    if ftol is not None:
        curve_fit_kwargs["ftol"] = ftol
    params, covariance = curve_fit(
        model,
        x_values,
        y_values,
        **curve_fit_kwargs,
    )
    if e_values is not None and model_based_weighting:
        preliminary_fit = model(x_values, *params)
        sigma = _model_based_uncertainty(y_values, preliminary_fit, e_values)
        curve_fit_kwargs = {
            "p0": params,
            "bounds": (lower, upper),
            "sigma": sigma,
            "absolute_sigma": True,
            "maxfev": min(max(100, int(maxfev)), 2000),
        }
        if ftol is not None:
            curve_fit_kwargs["ftol"] = ftol
        params, covariance = curve_fit(
            model,
            x_values,
            y_values,
            **curve_fit_kwargs,
        )
    y_fit = model(x_values, *params)
    fit_uncertainty = _model_based_uncertainty(y_values, y_fit, e_values) if model_based_weighting else e_values
    uncertainties = None
    if covariance is not None and covariance.shape[0] == len(params):
        diagonal = np.diag(covariance)
        if np.all(np.isfinite(diagonal)):
            uncertainties = np.sqrt(np.maximum(diagonal, 0.0))

    fitted_shape = {name: float(value) for name, value in zip(shape_names, params[: len(shape_names)])}
    centre_start = len(shape_names)
    width_start = centre_start + component_count
    height_start = width_start + component_count if use_independent_component_fwhm else width_start
    bg_start = height_start + component_count
    centres = [float(value) for value in params[centre_start:width_start]]
    if use_independent_component_fwhm:
        fwhms = [float(value) for value in params[width_start:height_start]]
    else:
        fwhms = []
    heights = [float(value) for value in params[height_start:bg_start]]
    if profile.uses_gsas2_tof_widths:
        fwhm = gsas2_tof_effective_fwhm(
            fitted_shape["sig"], fitted_shape["gam"], fitted_shape["alp"], fitted_shape["bet"]
        )
        eta = eta_initial
        fwhms = [fwhm] * component_count
    elif use_independent_component_fwhm:
        eta = fitted_shape.get("eta", 0.0 if profile.key == PROFILE_GAUSSIAN else 1.0 if profile.key == PROFILE_LORENTZIAN else eta_initial)
        fwhm = fwhms[int(np.argmax(heights))] if heights else fwhm_guess
    else:
        fwhm = fitted_shape["fwhm"]
        eta = fitted_shape.get("eta", 0.0 if profile.key == PROFILE_GAUSSIAN else 1.0 if profile.key == PROFILE_LORENTZIAN else eta_initial)
        fwhms = [fwhm] * component_count
    components = []
    for component_index, (centre_value, fwhm_value, height_value) in enumerate(zip(centres, fwhms, heights)):
        component_profile = evaluate_peak_profile(
            profile.key,
            x_values,
            centre_value,
            fwhm_value,
            eta=eta,
            tail=fitted_shape.get("tail", tail_guess),
            alpha=fitted_shape.get("alpha", tail_guess),
            beta=fitted_shape.get("beta", tail_guess),
            sig=fitted_shape.get("sig"),
            gam=fitted_shape.get("gam"),
            alp=fitted_shape.get("alp"),
            bet=fitted_shape.get("bet"),
        )
        component_area = float(abs(height_value * np.trapezoid(component_profile, x_values)))
        components.append(
            PeakComponentResult(
                centre=centre_value,
                fwhm=fwhm_value,
                height=height_value,
                area=component_area,
                eta=eta,
                centre_uncertainty=float(uncertainties[centre_start + component_index]) if uncertainties is not None else None,
                fwhm_uncertainty=(
                    None
                    if profile.uses_gsas2_tof_widths or uncertainties is None
                    else float(uncertainties[width_start + component_index])
                    if use_independent_component_fwhm
                    else float(uncertainties[0])
                ),
                reflection=component_reflections[component_index]
                if component_index < len(component_reflections)
                else None,
            )
        )
    primary_centre_index = int(np.argmax(heights))
    components = sorted(components, key=lambda component: component.centre)
    primary_component = max(components, key=lambda component: component.height)
    area = float(sum(component.area for component in components))
    quality = _fit_quality(y_values, y_fit, fit_uncertainty, len(params))
    profile_parameters = {
        key: float(value)
        for key, value in fitted_shape.items()
        if key != "fwhm"
    }
    profile_parameters["peak_count"] = float(len(components))
    if use_independent_component_fwhm:
        profile_parameters["independent_peak_fwhm"] = 1.0
    return PeakFitResult(
        model_name=profile.label,
        centre=primary_component.centre,
        centre_uncertainty=float(uncertainties[centre_start + primary_centre_index])
        if uncertainties is not None
        else None,
        fwhm=primary_component.fwhm,
        fwhm_uncertainty=primary_component.fwhm_uncertainty,
        height=primary_component.height,
        area=float(area),
        eta=eta,
        background_coefficients=tuple(float(value) for value in params[bg_start:]),
        quality=quality,
        profile_parameters=profile_parameters,
        components=tuple(components),
        fit_x=np.array(x_values, dtype=float, copy=True),
        fit_y=np.array(y_fit, dtype=float, copy=True),
        observed_y=np.array(y_values, dtype=float, copy=True),
        fit_uncertainty=None if fit_uncertainty is None else np.array(fit_uncertainty, dtype=float, copy=True),
    )
