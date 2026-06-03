from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .models import FitQuality, PeakComponentResult, PeakFitResult
from .profiles import (
    PROFILE_GAUSSIAN,
    PROFILE_LORENTZIAN,
    PROFILE_PSEUDO_VOIGT,
    evaluate_peak_profile,
    peak_profile_spec,
    pseudo_voigt_profile,
)


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
) -> PeakFitResult:
    profile = peak_profile_spec(profile_key)
    background_coefficients = _background_coefficients(x_values, y_values, polynomial_order)
    x_norm, _centre, _half_span = _normalised_axis(x_values)
    y_fit = _polyval(background_coefficients, x_norm)
    component_results = []
    peak_indices = sorted(peak_indices)
    for index, peak_index in enumerate(peak_indices):
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
            )
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
        component_profile = evaluate_peak_profile(
            profile.key,
            x_values,
            component.centre,
            component.fwhm,
            eta=component.eta,
            tail=float(local_result.profile_parameters.get("tail", component.fwhm)),
            alpha=float(local_result.profile_parameters.get("alpha", component.fwhm)),
            beta=float(local_result.profile_parameters.get("beta", component.fwhm)),
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
        )
    component_results = sorted(component_results, key=lambda component: component.centre)
    primary_component = max(component_results, key=lambda component: component.height)
    area = float(sum(component.area for component in component_results))
    parameter_count = len(component_results) * 4 + len(background_coefficients)
    quality = _fit_quality(y_values, y_fit, e_values, parameter_count)
    profile_parameters = {"peak_count": float(len(component_results))}
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
) -> PeakFitResult:
    from scipy.optimize import curve_fit

    profile = peak_profile_spec(profile_key)
    x_values, y_values, e_values = _clean_fit_inputs(x, y, e)
    order = max(0, min(int(polynomial_order), 5))
    _, _axis_centre, _axis_half_span = _normalised_axis(x_values)
    background_guess = float(np.percentile(y_values, 20.0))
    peak_indices = _detect_peak_indices(x_values, y_values, background_guess, max_peaks=max_peaks)
    if len(peak_indices) > 1:
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
        )
    centre_guesses = [float(x_values[index]) for index in peak_indices]
    height_guesses = [max(float(y_values[index] - background_guess), 1e-9) for index in peak_indices]
    fwhm_guesses = [_initial_fwhm(x_values, y_values, background_guess, index) for index in peak_indices]

    shape_names = ["fwhm"]
    if profile.uses_eta:
        shape_names.append("eta")
    if profile.uses_tail:
        shape_names.append("tail")
    if profile.uses_back_to_back:
        shape_names.extend(["alpha", "beta"])

    span = max(float(np.max(x_values) - np.min(x_values)), 1e-9)
    fwhm_min = max(span * max(float(fwhm_min_fraction), 1e-9), 1e-12)
    fwhm_max = max(span * max(float(fwhm_max_multiplier), 1e-6), fwhm_min * 1.001)
    eta_min, eta_max = sorted((float(eta_bounds[0]), float(eta_bounds[1])))
    eta_initial = min(max(float(eta_initial), eta_min), eta_max)
    fwhm_guess = float(np.median(fwhm_guesses)) if fwhm_guesses else max(span / 100.0, fwhm_min)
    fwhm_guess = min(max(fwhm_guess, fwhm_min), fwhm_max)
    tail_guess = max(fwhm_guess, span * 0.01, fwhm_min)
    tail_min = max(fwhm_min, span * 1e-5)
    tail_max = max(span * 2.0, tail_min * 1.001)

    p0: list[float] = [fwhm_guess]
    lower: list[float] = [fwhm_min]
    upper: list[float] = [fwhm_max]
    if profile.uses_eta:
        p0.append(eta_initial)
        lower.append(eta_min)
        upper.append(eta_max)
    if profile.uses_tail:
        p0.append(tail_guess)
        lower.append(tail_min)
        upper.append(tail_max)
    if profile.uses_back_to_back:
        p0.extend([tail_guess, tail_guess])
        lower.extend([tail_min, tail_min])
        upper.extend([tail_max, tail_max])

    for index, centre_guess in enumerate(centre_guesses):
        if len(centre_guesses) == 1:
            centre_min = float(np.min(x_values))
            centre_max = float(np.max(x_values))
        else:
            centre_min = float(np.min(x_values)) if index == 0 else 0.5 * (centre_guesses[index - 1] + centre_guess)
            centre_max = float(np.max(x_values)) if index == len(centre_guesses) - 1 else 0.5 * (centre_guess + centre_guesses[index + 1])
        p0.append(centre_guess)
        lower.append(centre_min)
        upper.append(centre_max)

    max_height = max(height_guesses + [float(np.max(y_values) - background_guess), 1.0])
    for height_guess in height_guesses:
        p0.append(height_guess)
        lower.append(0.0)
        upper.append(max(max_height * 100.0, 1.0))

    p0.extend([background_guess] + [0.0] * order)
    lower.extend([-np.inf] * (order + 1))
    upper.extend([np.inf] * (order + 1))

    def model(x_input: np.ndarray, *params: float) -> np.ndarray:
        shape = {name: float(value) for name, value in zip(shape_names, params[: len(shape_names)])}
        component_count = len(centre_guesses)
        centre_start = len(shape_names)
        height_start = centre_start + component_count
        bg_start = height_start + component_count
        centres = params[centre_start:height_start]
        heights = params[height_start:bg_start]
        x_input_norm = (x_input - _axis_centre) / _axis_half_span
        bg = params[bg_start:]
        output = _polyval(bg, x_input_norm)
        for centre, height in zip(centres, heights):
            profile_y = evaluate_peak_profile(
                profile.key,
                x_input,
                float(centre),
                shape["fwhm"],
                eta=shape.get("eta", 0.0 if profile.key == PROFILE_GAUSSIAN else 1.0 if profile.key == PROFILE_LORENTZIAN else eta_initial),
                tail=shape.get("tail", tail_guess),
                alpha=shape.get("alpha", tail_guess),
                beta=shape.get("beta", tail_guess),
            )
            output = output + float(height) * profile_y
        return output

    sigma = e_values if e_values is not None else None
    params, covariance = curve_fit(
        model,
        x_values,
        y_values,
        p0=p0,
        bounds=(lower, upper),
        sigma=sigma,
        absolute_sigma=sigma is not None,
        maxfev=max(100, int(maxfev)),
    )
    y_fit = model(x_values, *params)
    uncertainties = None
    if covariance is not None and covariance.shape[0] == len(params):
        diagonal = np.diag(covariance)
        if np.all(np.isfinite(diagonal)):
            uncertainties = np.sqrt(np.maximum(diagonal, 0.0))

    fitted_shape = {name: float(value) for name, value in zip(shape_names, params[: len(shape_names)])}
    component_count = len(centre_guesses)
    centre_start = len(shape_names)
    height_start = centre_start + component_count
    bg_start = height_start + component_count
    centres = [float(value) for value in params[centre_start:height_start]]
    heights = [float(value) for value in params[height_start:bg_start]]
    fwhm = fitted_shape["fwhm"]
    eta = fitted_shape.get("eta", 0.0 if profile.key == PROFILE_GAUSSIAN else 1.0 if profile.key == PROFILE_LORENTZIAN else eta_initial)
    components = []
    for component_index, (centre_value, height_value) in enumerate(zip(centres, heights)):
        component_profile = evaluate_peak_profile(
            profile.key,
            x_values,
            centre_value,
            fwhm,
            eta=eta,
            tail=fitted_shape.get("tail", tail_guess),
            alpha=fitted_shape.get("alpha", tail_guess),
            beta=fitted_shape.get("beta", tail_guess),
        )
        component_area = float(abs(height_value * np.trapezoid(component_profile, x_values)))
        components.append(
            PeakComponentResult(
                centre=centre_value,
                fwhm=fwhm,
                height=height_value,
                area=component_area,
                eta=eta,
                centre_uncertainty=float(uncertainties[centre_start + component_index]) if uncertainties is not None else None,
                fwhm_uncertainty=float(uncertainties[0]) if uncertainties is not None else None,
            )
        )
    primary_centre_index = int(np.argmax(heights))
    components = sorted(components, key=lambda component: component.centre)
    primary_component = max(components, key=lambda component: component.height)
    area = float(sum(component.area for component in components))
    quality = _fit_quality(y_values, y_fit, e_values, len(params))
    profile_parameters = {
        key: float(value)
        for key, value in fitted_shape.items()
        if key != "fwhm"
    }
    profile_parameters["peak_count"] = float(len(components))
    return PeakFitResult(
        model_name=profile.label,
        centre=primary_component.centre,
        centre_uncertainty=float(uncertainties[centre_start + primary_centre_index])
        if uncertainties is not None
        else None,
        fwhm=fwhm,
        fwhm_uncertainty=float(uncertainties[0]) if uncertainties is not None else None,
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
    )
