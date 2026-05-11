from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .models import FitQuality, PeakFitResult


def pseudo_voigt_profile(x: np.ndarray, centre: float, fwhm: float, eta: float) -> np.ndarray:
    width = max(abs(float(fwhm)), 1e-12)
    scaled = (np.asarray(x, dtype=float) - float(centre)) / width
    gaussian = np.exp(-4.0 * math.log(2.0) * scaled * scaled)
    lorentzian = 1.0 / (1.0 + 4.0 * scaled * scaled)
    return float(eta) * lorentzian + (1.0 - float(eta)) * gaussian


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


def _initial_fwhm(x: np.ndarray, y: np.ndarray, background: float) -> float:
    peak_index = int(np.argmax(y))
    peak_height = max(float(y[peak_index] - background), 1e-12)
    half_height = background + 0.5 * peak_height
    above = np.flatnonzero(y >= half_height)
    if above.size >= 2:
        return max(float(x[above[-1]] - x[above[0]]), (float(np.max(x)) - float(np.min(x))) / 1000.0)
    return max((float(np.max(x)) - float(np.min(x))) / 20.0, 1e-9)


def fit_pseudo_voigt(
    x: Sequence[float],
    y: Sequence[float],
    e: Optional[Sequence[float]] = None,
    polynomial_order: int = 2,
) -> PeakFitResult:
    from scipy.optimize import curve_fit

    x_values, y_values, e_values = _clean_fit_inputs(x, y, e)
    order = max(0, min(int(polynomial_order), 5))
    x_norm, _axis_centre, _axis_half_span = _normalised_axis(x_values)
    background_guess = float(np.percentile(y_values, 20.0))
    peak_index = int(np.argmax(y_values))
    centre_guess = float(x_values[peak_index])
    height_guess = max(float(y_values[peak_index] - background_guess), 1e-9)
    fwhm_guess = _initial_fwhm(x_values, y_values, background_guess)

    def model(x_input: np.ndarray, centre: float, fwhm: float, eta: float, height: float, *bg: float) -> np.ndarray:
        x_input_norm = (x_input - _axis_centre) / _axis_half_span
        return height * pseudo_voigt_profile(x_input, centre, fwhm, eta) + _polyval(bg, x_input_norm)

    p0 = [centre_guess, fwhm_guess, 0.5, height_guess] + [background_guess] + [0.0] * order
    span = max(float(np.max(x_values) - np.min(x_values)), 1e-9)
    lower = [float(np.min(x_values)), span / 10000.0, 0.0, 0.0] + [-np.inf] * (order + 1)
    upper = [float(np.max(x_values)), span * 2.0, 1.0, max(height_guess * 100.0, float(np.max(y_values)) * 100.0, 1.0)] + [
        np.inf
    ] * (order + 1)
    sigma = e_values if e_values is not None else None
    params, covariance = curve_fit(
        model,
        x_values,
        y_values,
        p0=p0,
        bounds=(lower, upper),
        sigma=sigma,
        absolute_sigma=sigma is not None,
        maxfev=50000,
    )
    y_fit = model(x_values, *params)
    uncertainties = None
    if covariance is not None and covariance.shape[0] == len(params):
        diagonal = np.diag(covariance)
        if np.all(np.isfinite(diagonal)):
            uncertainties = np.sqrt(np.maximum(diagonal, 0.0))

    centre, fwhm, eta, height = (float(params[0]), float(params[1]), float(params[2]), float(params[3]))
    gaussian_area_factor = math.sqrt(math.pi / (4.0 * math.log(2.0)))
    lorentzian_area_factor = math.pi / 2.0
    area = height * fwhm * (eta * lorentzian_area_factor + (1.0 - eta) * gaussian_area_factor)
    quality = _fit_quality(y_values, y_fit, e_values, len(params))
    return PeakFitResult(
        model_name="pseudo-Voigt",
        centre=centre,
        centre_uncertainty=float(uncertainties[0]) if uncertainties is not None else None,
        fwhm=fwhm,
        fwhm_uncertainty=float(uncertainties[1]) if uncertainties is not None else None,
        height=height,
        area=float(area),
        eta=eta,
        background_coefficients=tuple(float(value) for value in params[4:]),
        quality=quality,
        fit_x=np.array(x_values, dtype=float, copy=True),
        fit_y=np.array(y_fit, dtype=float, copy=True),
        observed_y=np.array(y_values, dtype=float, copy=True),
    )
