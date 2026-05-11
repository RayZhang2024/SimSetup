from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from .fitting import fit_pseudo_voigt
from .gsas_exp import generate_fcc_reflections, validate_cubic_fm3m_phase
from .models import (
    CalibrationPeakResult,
    CalibrationResult,
    FocusedSpectrum,
    InstrumentCalibration,
    PhaseModel,
    d_to_tof,
    tof_to_d,
)


def _prominence_threshold(y: np.ndarray) -> float:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return 1.0
    spread = float(np.nanpercentile(finite, 99.5) - np.nanpercentile(finite, 25.0))
    return max(spread * 0.18, float(np.nanmax(finite)) * 0.04, 1.0)


def _match_window(expected_tof: float, x_span: float) -> float:
    return max(90.0, min(450.0, 0.012 * abs(float(expected_tof)), 0.04 * float(x_span)))


def refine_ceo2_calibration(
    ceo2_spectrum: FocusedSpectrum,
    ceo2_phase: PhaseModel,
    *,
    polynomial_order: int = 1,
    minimum_peaks: int = 4,
) -> CalibrationResult:
    from scipy.signal import find_peaks

    validate_cubic_fm3m_phase(ceo2_phase)
    if not any(atom.element.upper().startswith("CE") for atom in ceo2_phase.atoms):
        raise ValueError("CeO2 calibration requires a phase with a Ce atom site.")
    if not any(atom.element.upper().startswith("O") for atom in ceo2_phase.atoms):
        raise ValueError("CeO2 calibration requires a phase with an O atom site.")
    if ceo2_spectrum.calibration is None:
        raise ValueError("CeO2 calibration requires embedded DIFC/TZERO metadata as an initial seed.")

    x = ceo2_spectrum.x_centres
    y = ceo2_spectrum.y
    if x.size != y.size:
        raise ValueError("CeO2 x/y arrays are inconsistent.")
    initial = InstrumentCalibration(
        difc=float(ceo2_spectrum.calibration.difc),
        tzero=float(ceo2_spectrum.calibration.tzero),
        difa=0.0,
    )
    d_axis = tof_to_d(np.array([float(np.min(x)), float(np.max(x))]), initial)
    d_min = max(float(np.min(d_axis)), 0.5)
    d_max = float(np.max(d_axis))
    reflections = generate_fcc_reflections(ceo2_phase, d_min, d_max)
    if not reflections:
        raise ValueError("No CeO2 reflections are expected in the histogram range.")

    prominence_threshold = _prominence_threshold(y)
    peak_indices, properties = find_peaks(y, prominence=prominence_threshold, distance=12)
    prominences = properties.get("prominences", np.zeros(peak_indices.size, dtype=float))
    used_peak_indices: set[int] = set()
    peak_results: list[CalibrationPeakResult] = []
    x_span = float(np.max(x) - np.min(x))

    for reflection in reflections:
        expected_tof = float(d_to_tof(reflection.d_spacing, initial))
        if expected_tof < float(np.min(x)) or expected_tof > float(np.max(x)):
            continue
        window = _match_window(expected_tof, x_span)
        nearby = np.flatnonzero(np.abs(x[peak_indices] - expected_tof) <= window)
        if nearby.size == 0:
            peak_results.append(
                CalibrationPeakResult(
                    reflection=reflection,
                    expected_tof=expected_tof,
                    observed_tof=None,
                    fitted_tof=None,
                    residual_tof=None,
                    prominence=None,
                    accepted=False,
                    rejection_reason="no detected peak nearby",
                )
            )
            continue
        best_local = max(nearby, key=lambda index: float(prominences[index]))
        peak_index = int(peak_indices[int(best_local)])
        if peak_index in used_peak_indices:
            peak_results.append(
                CalibrationPeakResult(
                    reflection=reflection,
                    expected_tof=expected_tof,
                    observed_tof=float(x[peak_index]),
                    fitted_tof=None,
                    residual_tof=None,
                    prominence=float(prominences[int(best_local)]),
                    accepted=False,
                    rejection_reason="peak already matched",
                )
            )
            continue
        fit_half_width = max(window, 8.0 * float(np.median(np.diff(x))))
        fit_mask = (x >= float(x[peak_index]) - fit_half_width) & (x <= float(x[peak_index]) + fit_half_width)
        if int(np.count_nonzero(fit_mask)) < 8:
            peak_results.append(
                CalibrationPeakResult(
                    reflection=reflection,
                    expected_tof=expected_tof,
                    observed_tof=float(x[peak_index]),
                    fitted_tof=None,
                    residual_tof=None,
                    prominence=float(prominences[int(best_local)]),
                    accepted=False,
                    rejection_reason="insufficient local points",
                )
            )
            continue
        try:
            fit_result = fit_pseudo_voigt(
                x[fit_mask],
                y[fit_mask],
                ceo2_spectrum.e[fit_mask] if ceo2_spectrum.e is not None else None,
                polynomial_order=polynomial_order,
            )
        except Exception as exc:
            peak_results.append(
                CalibrationPeakResult(
                    reflection=reflection,
                    expected_tof=expected_tof,
                    observed_tof=float(x[peak_index]),
                    fitted_tof=None,
                    residual_tof=None,
                    prominence=float(prominences[int(best_local)]),
                    accepted=False,
                    rejection_reason=f"fit failed: {exc}",
                )
            )
            continue

        residual = float(fit_result.centre - expected_tof)
        if not math.isfinite(fit_result.centre) or abs(residual) > window:
            accepted = False
            reason = "fitted centre outside match window"
        elif fit_result.height <= 0.0 or fit_result.fwhm <= 0.0:
            accepted = False
            reason = "non-positive fitted peak"
        else:
            accepted = True
            reason = ""
            used_peak_indices.add(peak_index)
        peak_results.append(
            CalibrationPeakResult(
                reflection=reflection,
                expected_tof=expected_tof,
                observed_tof=float(x[peak_index]),
                fitted_tof=float(fit_result.centre),
                residual_tof=residual,
                prominence=float(prominences[int(best_local)]),
                accepted=accepted,
                rejection_reason=reason,
            )
        )

    accepted_candidates = [peak for peak in peak_results if peak.accepted and peak.fitted_tof is not None]
    if len(accepted_candidates) < int(minimum_peaks):
        raise ValueError(f"Only {len(accepted_candidates)} CeO2 peaks were accepted; at least {minimum_peaks} are required.")

    candidate_d = np.array([peak.reflection.d_spacing for peak in accepted_candidates], dtype=float)
    candidate_tof = np.array([float(peak.fitted_tof) for peak in accepted_candidates], dtype=float)
    active_mask = np.ones(candidate_tof.size, dtype=bool)
    difc = initial.difc
    tzero = initial.tzero
    for _iteration in range(4):
        design = np.column_stack([candidate_d[active_mask], np.ones(int(np.count_nonzero(active_mask)), dtype=float)])
        difc, tzero = np.linalg.lstsq(design, candidate_tof[active_mask], rcond=None)[0]
        residuals = candidate_tof - (float(difc) * candidate_d + float(tzero))
        active_residuals = residuals[active_mask]
        median_residual = float(np.median(active_residuals))
        mad = float(np.median(np.abs(active_residuals - median_residual)))
        cutoff = max(60.0, 4.0 * 1.4826 * mad)
        next_mask = np.abs(residuals - median_residual) <= cutoff
        if int(np.count_nonzero(next_mask)) < int(minimum_peaks):
            break
        if np.array_equal(next_mask, active_mask):
            break
        active_mask = next_mask
    accepted_peaks = [peak for peak, keep in zip(accepted_candidates, active_mask) if keep]
    if len(accepted_peaks) < int(minimum_peaks):
        raise ValueError(f"Only {len(accepted_peaks)} CeO2 peaks remained after outlier rejection.")

    refined = InstrumentCalibration(difc=float(difc), tzero=float(tzero), difa=0.0)
    accepted_observed = np.array([float(peak.fitted_tof) for peak in accepted_peaks], dtype=float)
    residuals = accepted_observed - d_to_tof(
        np.array([peak.reflection.d_spacing for peak in accepted_peaks], dtype=float),
        refined,
    )
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    updated_results: list[CalibrationPeakResult] = []
    accepted_keys = {(peak.reflection.h, peak.reflection.k, peak.reflection.l) for peak in accepted_peaks}
    outlier_keys = {
        (peak.reflection.h, peak.reflection.k, peak.reflection.l)
        for peak, keep in zip(accepted_candidates, active_mask)
        if not keep
    }
    residual_by_label = {
        (peak.reflection.h, peak.reflection.k, peak.reflection.l): float(residual)
        for peak, residual in zip(accepted_peaks, residuals)
    }
    for peak in peak_results:
        key = (peak.reflection.h, peak.reflection.k, peak.reflection.l)
        accepted = peak.accepted
        rejection_reason = peak.rejection_reason
        if key in outlier_keys:
            accepted = False
            rejection_reason = "calibration residual outlier"
        elif key in accepted_keys:
            accepted = True
            rejection_reason = ""
        updated_results.append(
            CalibrationPeakResult(
                reflection=peak.reflection,
                expected_tof=peak.expected_tof,
                observed_tof=peak.observed_tof,
                fitted_tof=peak.fitted_tof,
                residual_tof=residual_by_label.get(key, peak.residual_tof),
                prominence=peak.prominence,
                accepted=accepted,
                rejection_reason=rejection_reason,
            )
        )
    return CalibrationResult(
        calibration=refined,
        initial_calibration=initial,
        phase_name=ceo2_phase.name,
        source_path=Path(ceo2_spectrum.source_path),
        run_number=ceo2_spectrum.run_number,
        bank_number=ceo2_spectrum.bank_number,
        peak_results=tuple(updated_results),
        rms_residual_tof=rms,
    )
