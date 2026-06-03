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
GSAS_TOF_PROFILE_KEYS = (
    PROFILE_GSAS_TOF_1,
    PROFILE_GSAS_TOF_2,
    PROFILE_GSAS_TOF_3,
    PROFILE_GSAS_TOF_4,
    PROFILE_GSAS_TOF_5,
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
    raise ValueError(f"Unsupported peak profile: {profile_key}")
