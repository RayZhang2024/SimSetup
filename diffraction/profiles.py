from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


PROFILE_PSEUDO_VOIGT = "pseudo_voigt"
PROFILE_GAUSSIAN = "gaussian"
PROFILE_LORENTZIAN = "lorentzian"
PROFILE_VOIGT = "voigt"
PROFILE_EXP_GAUSSIAN = "exp_gaussian"
PROFILE_EXP_LORENTZIAN = "exp_lorentzian"
PROFILE_EXP_VOIGT = "exp_voigt"
PROFILE_GSAS_TOF_1 = "gsas_tof_1"
PROFILE_GSAS_TOF_2 = "gsas_tof_2"
PROFILE_GSAS_TOF = "gsas_tof"
PROFILE_GSAS_TOF_3 = PROFILE_GSAS_TOF
PROFILE_GSAS_TOF_4 = "gsas_tof_4"
PROFILE_GSAS_TOF_5 = "gsas_tof_5"
PROFILE_GSAS2_TOF_PV = "gsas2_tof_pv"
GSAS_TOF_PROFILE_KEYS = (
    PROFILE_GSAS_TOF_1,
    PROFILE_GSAS_TOF_2,
    PROFILE_GSAS_TOF_3,
    PROFILE_GSAS_TOF_4,
    PROFILE_GSAS_TOF_5,
    PROFILE_GSAS2_TOF_PV,
)


@dataclass(frozen=True)
class PeakProfileSpec:
    key: str
    label: str
    uses_eta: bool = False
    uses_tail: bool = False
    uses_back_to_back: bool = False
    gsas_tof_function: Optional[int] = None
    uses_hkl_broadening: bool = False
    uses_position_shift: bool = False
    uses_gsas2_tof_widths: bool = False


PEAK_PROFILE_SPECS: tuple[PeakProfileSpec, ...] = (
    PeakProfileSpec(PROFILE_PSEUDO_VOIGT, "Pseudo-Voigt", uses_eta=True),
    PeakProfileSpec(PROFILE_GAUSSIAN, "Open GENIE Gaussian"),
    PeakProfileSpec(PROFILE_LORENTZIAN, "Open GENIE Lorentzian"),
    PeakProfileSpec(PROFILE_VOIGT, "Open GENIE Voigt", uses_eta=True),
    PeakProfileSpec(PROFILE_EXP_GAUSSIAN, "Open GENIE Gaussian + exponential", uses_tail=True),
    PeakProfileSpec(PROFILE_EXP_LORENTZIAN, "Open GENIE Lorentzian + exponential", uses_tail=True),
    PeakProfileSpec(PROFILE_EXP_VOIGT, "Open GENIE Voigt + exponential", uses_eta=True, uses_tail=True),
    PeakProfileSpec(
        PROFILE_GSAS_TOF_1,
        "GSAS TOF profile 1: back-to-back Gaussian",
        uses_back_to_back=True,
        gsas_tof_function=1,
    ),
    PeakProfileSpec(
        PROFILE_GSAS_TOF_2,
        "GSAS TOF profile 2: Ikeda-Carpenter PV",
        uses_eta=True,
        uses_tail=True,
        gsas_tof_function=2,
    ),
    PeakProfileSpec(
        PROFILE_GSAS_TOF_3,
        "GSAS TOF profile 3: back-to-back PV",
        uses_eta=True,
        uses_back_to_back=True,
        gsas_tof_function=3,
    ),
    PeakProfileSpec(
        PROFILE_GSAS_TOF_4,
        "GSAS TOF profile 4: anisotropic broadening PV",
        uses_eta=True,
        uses_back_to_back=True,
        gsas_tof_function=4,
        uses_hkl_broadening=True,
    ),
    PeakProfileSpec(
        PROFILE_GSAS_TOF_5,
        "GSAS TOF profile 5: macrostrain shift PV",
        uses_eta=True,
        uses_back_to_back=True,
        gsas_tof_function=5,
        uses_position_shift=True,
    ),
    PeakProfileSpec(
        PROFILE_GSAS2_TOF_PV,
        "GSAS-II TOF double-exponential PV",
        gsas_tof_function=6,
        uses_gsas2_tof_widths=True,
    ),
)

PEAK_PROFILE_BY_KEY = {spec.key: spec for spec in PEAK_PROFILE_SPECS}


def peak_profile_spec(profile_key: str) -> PeakProfileSpec:
    try:
        return PEAK_PROFILE_BY_KEY[str(profile_key)]
    except KeyError as exc:
        raise ValueError(f"Unsupported peak profile: {profile_key}") from exc


def is_gsas_tof_profile(profile_key: str) -> bool:
    return peak_profile_spec(profile_key).gsas_tof_function is not None


def gaussian_profile(x: Sequence[float] | np.ndarray, centre: float, fwhm: float) -> np.ndarray:
    width = max(abs(float(fwhm)), 1e-12)
    scaled = (np.asarray(x, dtype=float) - float(centre)) / width
    scaled = np.clip(scaled, -40.0, 40.0)
    return np.exp(-4.0 * math.log(2.0) * scaled * scaled)


def lorentzian_profile(x: Sequence[float] | np.ndarray, centre: float, fwhm: float) -> np.ndarray:
    width = max(abs(float(fwhm)), 1e-12)
    scaled = (np.asarray(x, dtype=float) - float(centre)) / width
    scaled = np.clip(scaled, -1.0e6, 1.0e6)
    return 1.0 / (1.0 + 4.0 * scaled * scaled)


def pseudo_voigt_profile(x: Sequence[float] | np.ndarray, centre: float, fwhm: float, eta: float) -> np.ndarray:
    eta_value = float(np.clip(float(eta), 0.0, 1.0))
    return eta_value * lorentzian_profile(x, centre, fwhm) + (1.0 - eta_value) * gaussian_profile(x, centre, fwhm)


def voigt_profile(x: Sequence[float] | np.ndarray, centre: float, fwhm: float, eta: float) -> np.ndarray:
    from scipy.special import voigt_profile as scipy_voigt_profile

    width = max(abs(float(fwhm)), 1e-12)
    eta_value = float(np.clip(float(eta), 0.0, 1.0))
    sigma = max((1.0 - eta_value) * width / (2.0 * math.sqrt(2.0 * math.log(2.0))), width * 1e-6)
    gamma = max(eta_value * width / 2.0, width * 1e-6)
    values = scipy_voigt_profile(np.asarray(x, dtype=float) - float(centre), sigma, gamma)
    peak = float(scipy_voigt_profile(np.array([0.0], dtype=float), sigma, gamma)[0])
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(np.asarray(x, dtype=float))
    return values / peak


def _axis_step(x: np.ndarray) -> float:
    if x.size < 2:
        return 1.0
    diffs = np.diff(np.sort(np.asarray(x, dtype=float)))
    finite = diffs[np.isfinite(diffs) & (np.abs(diffs) > 1e-12)]
    if finite.size == 0:
        return 1.0
    return float(np.median(np.abs(finite)))


def back_to_back_tail_bounds(
    axis_span: float,
    fwhm_guess: float,
    fwhm_min: float,
    fwhm_max: float,
) -> tuple[float, float, float]:
    """Return bounded back-to-back exponential tail parameters in axis units.

    The back-to-back Gaussian/PV alpha and beta values are decay lengths, so
    allowing them to grow to multiples of the entire selected range makes them
    absorb background and peak-position errors. Keep them broad enough for TOF
    asymmetry but tied to the current peak-width scale.
    """
    span = max(abs(float(axis_span)), 1e-12)
    lower = max(abs(float(fwhm_min)), span * 1e-5, 1e-12)
    width_scale = max(abs(float(fwhm_guess)), lower)
    width_ceiling = max(abs(float(fwhm_max)), lower)
    upper = min(span * 0.25, max(width_scale * 20.0, width_ceiling * 2.0, lower * 1.001))
    upper = max(upper, lower * 1.001)
    guess = min(max(max(width_scale, span * 0.01), lower), upper)
    return guess, lower, upper


def gsas2_tof_profile_initial_bounds(
    axis_span: float,
    fwhm_guess: float,
    fwhm_min: float,
    fwhm_max: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Initial values and bounds for a native GSAS-II-style TOF eps-Voigt profile.

    ``sig`` is treated as a Gaussian variance in axis units squared, ``gam`` as
    a Lorentzian FWHM-like width in axis units, and ``alp``/``bet`` as positive
    exponential rate coefficients in inverse axis units.
    """
    span = max(abs(float(axis_span)), 1e-12)
    fwhm_min_value = max(abs(float(fwhm_min)), span * 1e-5, 1e-12)
    fwhm_max_value = max(abs(float(fwhm_max)), fwhm_min_value * 1.001)
    fwhm_guess_value = min(max(abs(float(fwhm_guess)), fwhm_min_value), fwhm_max_value)
    sigma_to_fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0))
    sig_guess = (0.6 * fwhm_guess_value / sigma_to_fwhm) ** 2
    gam_guess = max(0.4 * fwhm_guess_value, fwhm_min_value)
    sig_lower = (fwhm_min_value / sigma_to_fwhm) ** 2
    sig_upper = (fwhm_max_value / sigma_to_fwhm) ** 2
    gam_lower = fwhm_min_value
    gam_upper = fwhm_max_value
    tail_length_guess, tail_length_lower, tail_length_upper = back_to_back_tail_bounds(
        span, fwhm_guess_value, fwhm_min_value, fwhm_max_value
    )
    # Convert length-like existing bounds to GSAS-II-style positive rates.
    rate_guess = 1.0 / max(tail_length_guess, 1e-12)
    rate_lower = 1.0 / max(tail_length_upper, 1e-12)
    rate_upper = 1.0 / max(tail_length_lower, 1e-12)
    return (
        {"sig": sig_guess, "gam": gam_guess, "alp": rate_guess, "bet": rate_guess},
        {"sig": sig_lower, "gam": gam_lower, "alp": rate_lower, "bet": rate_lower},
        {"sig": sig_upper, "gam": gam_upper, "alp": rate_upper, "bet": rate_upper},
    )


def gsas2_tof_effective_fwhm(sig: float, gam: float, alp: float, bet: float) -> float:
    sigma_fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * math.sqrt(max(float(sig), 0.0))
    gamma_fwhm = max(abs(float(gam)), 0.0)
    voigt_fwhm = 0.5346 * gamma_fwhm + math.sqrt(
        max(0.2166 * gamma_fwhm * gamma_fwhm + sigma_fwhm * sigma_fwhm, 0.0)
    )
    asymmetry_fwhm = math.log(2.0) * (
        1.0 / max(abs(float(alp)), 1e-12) + 1.0 / max(abs(float(bet)), 1e-12)
    )
    return float(max(voigt_fwhm + asymmetry_fwhm, 1e-12))


def _normalise_peak(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    peak = float(np.max(finite))
    if peak <= 0.0:
        return np.zeros_like(values, dtype=float)
    return values / peak


def _convolve_profile(x: np.ndarray, base: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    dx = _axis_step(x)
    area = float(np.sum(kernel) * dx)
    if not np.isfinite(area) or abs(area) < 1e-20:
        return base
    convolved = fftconvolve(base, kernel / area, mode="same") * dx
    return _normalise_peak(np.asarray(convolved, dtype=float))


def _one_sided_exponential_profile(x: np.ndarray, base: np.ndarray, tail: float) -> np.ndarray:
    dx = _axis_step(x)
    n = int(x.size)
    offsets = (np.arange(n, dtype=float) - n // 2) * dx
    tau = max(abs(float(tail)), dx)
    decay = np.exp(np.clip(-offsets / tau, -80.0, 0.0)) / tau
    kernel = np.where(offsets >= 0.0, decay, 0.0)
    return _convolve_profile(x, base, kernel)


def _ikeda_carpenter_like_profile(x: np.ndarray, base: np.ndarray, tail: float) -> np.ndarray:
    dx = _axis_step(x)
    n = int(x.size)
    offsets = (np.arange(n, dtype=float) - n // 2) * dx
    slow_tau = max(abs(float(tail)), dx)
    fast_tau = max(0.25 * slow_tau, dx)
    delta = np.zeros_like(offsets)
    delta[n // 2] = 1.0 / dx
    fast_decay = np.exp(np.clip(-offsets / fast_tau, -80.0, 0.0)) / fast_tau
    slow_decay = np.exp(np.clip(-offsets / slow_tau, -80.0, 0.0)) / slow_tau
    fast = np.where(offsets >= 0.0, fast_decay, 0.0)
    slow = np.where(offsets >= 0.0, slow_decay, 0.0)
    kernel = 0.35 * delta + 0.45 * fast + 0.20 * slow
    return _convolve_profile(x, base, kernel)


def _back_to_back_exponential_profile(x: np.ndarray, base: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    dx = _axis_step(x)
    n = int(x.size)
    offsets = (np.arange(n, dtype=float) - n // 2) * dx
    alpha_value = max(abs(float(alpha)), dx)
    beta_value = max(abs(float(beta)), dx)
    left = np.exp(np.clip(offsets / alpha_value, -80.0, 0.0)) / alpha_value
    right = np.exp(np.clip(-offsets / beta_value, -80.0, 0.0)) / beta_value
    kernel = np.where(offsets < 0.0, left, right)
    return _convolve_profile(x, base, kernel)


def _gsas2_double_exponential_pseudo_voigt_profile(
    x: np.ndarray,
    centre: float,
    sig: float,
    gam: float,
    alp: float,
    bet: float,
) -> np.ndarray:
    from scipy.special import voigt_profile as scipy_voigt_profile

    x_values = np.asarray(x, dtype=float)
    sigma = max(math.sqrt(max(float(sig), 0.0)), _axis_step(x_values) * 1e-3, 1e-12)
    gamma = max(abs(float(gam)) / 2.0, _axis_step(x_values) * 1e-3, 1e-12)
    base_values = scipy_voigt_profile(x_values - float(centre), sigma, gamma)
    base_peak = float(scipy_voigt_profile(np.array([0.0], dtype=float), sigma, gamma)[0])
    if not np.isfinite(base_peak) or base_peak <= 0.0:
        return np.zeros_like(x_values)
    base_values = base_values / base_peak
    dx = _axis_step(x_values)
    n = int(x_values.size)
    offsets = (np.arange(n, dtype=float) - n // 2) * dx
    alpha_rate = max(abs(float(alp)), 1e-12)
    beta_rate = max(abs(float(bet)), 1e-12)
    left = alpha_rate * np.exp(np.clip(alpha_rate * offsets, -80.0, 0.0))
    right = beta_rate * np.exp(np.clip(-beta_rate * offsets, -80.0, 0.0))
    kernel = np.where(offsets < 0.0, left, right)
    return _convolve_profile(x_values, base_values, kernel)


def evaluate_peak_profile(
    profile_key: str,
    x: Sequence[float] | np.ndarray,
    centre: float,
    fwhm: float,
    *,
    eta: float = 0.5,
    tail: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    sig: Optional[float] = None,
    gam: Optional[float] = None,
    alp: Optional[float] = None,
    bet: Optional[float] = None,
) -> np.ndarray:
    x_values = np.asarray(x, dtype=float)
    key = peak_profile_spec(profile_key).key
    if key == PROFILE_PSEUDO_VOIGT:
        return pseudo_voigt_profile(x_values, centre, fwhm, eta)
    if key == PROFILE_GAUSSIAN:
        return gaussian_profile(x_values, centre, fwhm)
    if key == PROFILE_LORENTZIAN:
        return lorentzian_profile(x_values, centre, fwhm)
    if key == PROFILE_VOIGT:
        return voigt_profile(x_values, centre, fwhm, eta)
    if key == PROFILE_EXP_GAUSSIAN:
        return _one_sided_exponential_profile(x_values, gaussian_profile(x_values, centre, fwhm), tail)
    if key == PROFILE_EXP_LORENTZIAN:
        return _one_sided_exponential_profile(x_values, lorentzian_profile(x_values, centre, fwhm), tail)
    if key == PROFILE_EXP_VOIGT:
        return _one_sided_exponential_profile(x_values, voigt_profile(x_values, centre, fwhm, eta), tail)
    if key == PROFILE_GSAS_TOF_1:
        return _back_to_back_exponential_profile(
            x_values,
            gaussian_profile(x_values, centre, fwhm),
            alpha,
            beta,
        )
    if key == PROFILE_GSAS_TOF_2:
        return _ikeda_carpenter_like_profile(
            x_values,
            pseudo_voigt_profile(x_values, centre, fwhm, eta),
            tail,
        )
    if key in {PROFILE_GSAS_TOF_3, PROFILE_GSAS_TOF_4, PROFILE_GSAS_TOF_5}:
        return _back_to_back_exponential_profile(
            x_values,
            pseudo_voigt_profile(x_values, centre, fwhm, eta),
            alpha,
            beta,
        )
    if key == PROFILE_GSAS2_TOF_PV:
        return _gsas2_double_exponential_pseudo_voigt_profile(
            x_values,
            centre,
            fwhm if sig is None else sig,
            fwhm if gam is None else gam,
            alpha if alp is None else alp,
            beta if bet is None else bet,
        )
    raise ValueError(f"Unsupported peak profile: {profile_key}")
