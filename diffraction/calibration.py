from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

from .fitting import _clean_fit_inputs, _fit_quality, _normalised_axis, _polyval, fit_peak_profile
from .gsas_exp import generate_fcc_reflections, validate_cubic_fm3m_phase
from .models import (
    CalibrationPeakResult,
    CalibrationResult,
    FocusedSpectrum,
    InstrumentCalibration,
    PawleyFitResult,
    PawleyReflectionResult,
    PhaseModel,
    d_to_tof,
    tof_to_d,
)
from .profiles import PROFILE_EXP_VOIGT, PROFILE_GSAS_TOF, evaluate_peak_profile, peak_profile_spec

CalibrationProgressCallback = Callable[[str, Mapping[str, object]], None]
OPENGENIE_CALIBRATION_TOF_MIN = 15978.6
OPENGENIE_CALIBRATION_TOF_MAX = 40744.4


def _emit_calibration_progress(
    callback: Optional[CalibrationProgressCallback],
    stage: str,
    payload: Mapping[str, object],
) -> None:
    if callback is None:
        return
    callback(stage, payload)


def _prominence_threshold(y: np.ndarray) -> float:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return 1.0
    spread = float(np.nanpercentile(finite, 99.5) - np.nanpercentile(finite, 25.0))
    return max(spread * 0.18, float(np.nanmax(finite)) * 0.04, 1.0)


def _match_window(expected_tof: float, x_span: float) -> float:
    return max(90.0, min(450.0, 0.012 * abs(float(expected_tof)), 0.04 * float(x_span)))


def _parameter_uncertainties(result: object, parameter_count: int) -> Optional[np.ndarray]:
    residual = np.asarray(getattr(result, "fun", []), dtype=float)
    jacobian = np.asarray(getattr(result, "jac", []), dtype=float)
    if residual.size <= int(parameter_count) or jacobian.shape != (residual.size, int(parameter_count)):
        return None
    dof = residual.size - int(parameter_count)
    if dof <= 0 or not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
        return None
    try:
        covariance = np.linalg.pinv(jacobian.T @ jacobian) * (float(np.sum(residual * residual)) / float(dof))
    except np.linalg.LinAlgError:
        return None
    diagonal = np.diag(covariance)
    if diagonal.size != int(parameter_count) or not np.all(np.isfinite(diagonal)):
        return None
    return np.sqrt(np.maximum(diagonal, 0.0))


def _linear_calibration_from_peaks(
    peaks: list[CalibrationPeakResult],
    minimum_peaks: int,
    progress_callback: Optional[CalibrationProgressCallback] = None,
) -> tuple[InstrumentCalibration, float, list[CalibrationPeakResult]]:
    accepted_candidates = [peak for peak in peaks if peak.accepted and peak.fitted_tof is not None]
    if len(accepted_candidates) < int(minimum_peaks):
        raise ValueError(f"Only {len(accepted_candidates)} CeO2 peaks were accepted; at least {minimum_peaks} are required.")

    candidate_d = np.array([peak.reflection.d_spacing for peak in accepted_candidates], dtype=float)
    candidate_tof = np.array([float(peak.fitted_tof) for peak in accepted_candidates], dtype=float)
    candidate_sigma = np.array(
        [
            max(float(peak.tof_uncertainty), 1e-6) if peak.tof_uncertainty is not None else 1.0
            for peak in accepted_candidates
        ],
        dtype=float,
    )
    active_mask = np.ones(candidate_tof.size, dtype=bool)
    difc = 1.0
    tzero = 0.0
    for _iteration in range(4):
        design = np.column_stack([candidate_d[active_mask], np.ones(int(np.count_nonzero(active_mask)), dtype=float)])
        weights = 1.0 / candidate_sigma[active_mask]
        weighted_design = design * weights[:, None]
        weighted_tof = candidate_tof[active_mask] * weights
        difc, tzero = np.linalg.lstsq(weighted_design, weighted_tof, rcond=None)[0]
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

    calibration = InstrumentCalibration(difc=float(difc), tzero=float(tzero), difa=0.0)
    accepted_observed = np.array([float(peak.fitted_tof) for peak in accepted_peaks], dtype=float)
    residuals = accepted_observed - d_to_tof(
        np.array([peak.reflection.d_spacing for peak in accepted_peaks], dtype=float),
        calibration,
    )
    rms = float(np.sqrt(np.mean(residuals * residuals)))
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
    updated_results: list[CalibrationPeakResult] = []
    for peak in peaks:
        key = (peak.reflection.h, peak.reflection.k, peak.reflection.l)
        accepted = peak.accepted
        rejection_reason = peak.rejection_reason
        if key in outlier_keys:
            accepted = False
            rejection_reason = "calibration residual outlier"
        elif key in accepted_keys:
            accepted = True
            rejection_reason = ""
        expected = float(d_to_tof(peak.reflection.d_spacing, calibration))
        updated_results.append(
            CalibrationPeakResult(
                reflection=peak.reflection,
                expected_tof=expected,
                observed_tof=peak.observed_tof,
                fitted_tof=peak.fitted_tof,
                residual_tof=residual_by_label.get(key, None if peak.fitted_tof is None else float(peak.fitted_tof - expected)),
                prominence=peak.prominence,
                accepted=accepted,
                rejection_reason=rejection_reason,
                tof_uncertainty=peak.tof_uncertainty,
                fwhm=peak.fwhm,
                fwhm_uncertainty=peak.fwhm_uncertainty,
                height=peak.height,
                height_uncertainty=peak.height_uncertainty,
                exponential_decay=peak.exponential_decay,
                lorentz_fraction=peak.lorentz_fraction,
            )
        )
    _emit_calibration_progress(
        progress_callback,
        "linear_fit",
        {
            "peaks": tuple(updated_results),
            "calibration": calibration,
            "rms_residual_tof": rms,
        },
    )
    return calibration, rms, updated_results


def _fit_ceo2_pattern_calibration(
    ceo2_spectrum: FocusedSpectrum,
    ceo2_phase: PhaseModel,
    seed: InstrumentCalibration,
    single_peak_results: list[CalibrationPeakResult],
    *,
    polynomial_order: int,
    max_nfev: int = 50000,
    progress_callback: Optional[CalibrationProgressCallback] = None,
    tof_min: float = OPENGENIE_CALIBRATION_TOF_MIN,
    tof_max: float = OPENGENIE_CALIBRATION_TOF_MAX,
) -> tuple[InstrumentCalibration, PawleyFitResult]:
    from scipy.optimize import least_squares

    x_all, y_all, e_all = _clean_fit_inputs(ceo2_spectrum.x_centres, ceo2_spectrum.y, ceo2_spectrum.e)
    fit_tof_min = max(float(np.min(x_all)), min(float(tof_min), float(tof_max)))
    fit_tof_max = min(float(np.max(x_all)), max(float(tof_min), float(tof_max)))
    if fit_tof_max <= fit_tof_min:
        raise ValueError("The OpenGENIE calibration TOF range does not overlap the calibration histogram.")
    d_axis = tof_to_d(np.array([fit_tof_min, fit_tof_max], dtype=float), seed)
    reflections = generate_fcc_reflections(ceo2_phase, max(float(np.min(d_axis)), 0.5), float(np.max(d_axis)))
    accepted_reflections = [
        peak.reflection
        for peak in single_peak_results
        if peak.accepted and peak.fitted_tof is not None
    ]
    if len(accepted_reflections) >= 4:
        reflection_keys = {(ref.h, ref.k, ref.l) for ref in accepted_reflections}
        reflections = [ref for ref in reflections if (ref.h, ref.k, ref.l) in reflection_keys]
    if len(reflections) < 4:
        raise ValueError("At least four CeO2 reflections are required for GSAS-style pattern calibration.")

    seed_positions = np.array([float(d_to_tof(reflection.d_spacing, seed)) for reflection in reflections], dtype=float)
    roi_margin = max(600.0, 0.02 * float(np.max(seed_positions) - np.min(seed_positions)))
    roi_low = max(fit_tof_min, float(np.min(seed_positions)) - roi_margin)
    roi_high = min(fit_tof_max, float(np.max(seed_positions)) + roi_margin)
    roi = (x_all >= roi_low) & (x_all <= roi_high)
    if int(np.count_nonzero(roi)) < 50:
        raise ValueError("Insufficient points in the CeO2 pattern calibration range.")
    x_values = x_all[roi]
    y_values = y_all[roi]
    e_values = None if e_all is None else e_all[roi]
    max_pattern_points = 1500
    if x_values.size > max_pattern_points:
        indices = np.unique(np.linspace(0, x_values.size - 1, max_pattern_points).astype(int))
        x_values = x_values[indices]
        y_values = y_values[indices]
        if e_values is not None:
            e_values = e_values[indices]
    x_norm, _axis_centre, _axis_half_span = _normalised_axis(x_values)

    profile = peak_profile_spec(PROFILE_GSAS_TOF)
    centre_constraint_peaks = [
        peak
        for peak in single_peak_results
        if peak.accepted and peak.fitted_tof is not None
    ]
    centre_d = np.array([peak.reflection.d_spacing for peak in centre_constraint_peaks], dtype=float)
    centre_tof = np.array([float(peak.fitted_tof) for peak in centre_constraint_peaks], dtype=float)
    centre_sigma = np.array(
        [
            max(float(peak.tof_uncertainty), 1.0) if peak.tof_uncertainty is not None else 5.0
            for peak in centre_constraint_peaks
        ],
        dtype=float,
    )
    order = max(0, min(int(polynomial_order), 5))
    background_guess = float(np.percentile(y_values, 20.0))
    peak_signal = np.maximum(y_values - background_guess, 0.0)
    intensity_guesses = [
        max(float(np.interp(float(d_to_tof(reflection.d_spacing, seed)), x_values, peak_signal)), 1.0)
        for reflection in reflections
    ]
    fitted_widths = [float(peak.fwhm) for peak in single_peak_results if peak.accepted and peak.fwhm is not None and peak.fwhm > 0.0]
    fwhm_guess = float(np.median(fitted_widths)) if fitted_widths else max(float(np.max(x_values) - np.min(x_values)) / 400.0, 10.0)
    fwhm_guess = min(max(fwhm_guess, 5.0), 1000.0)
    alpha_guess = max(fwhm_guess * 0.35, 1.0)
    beta_guess = max(fwhm_guess * 0.7, 1.0)

    shape_names = ["fwhm", "eta", "alpha", "beta"]
    p0 = [
        float(seed.difc),
        float(seed.difa),
        float(seed.tzero),
        fwhm_guess,
        0.5,
        alpha_guess,
        beta_guess,
        *intensity_guesses,
        background_guess,
        *([0.0] * order),
    ]
    difc_span = max(abs(float(seed.difc)) * 0.05, 500.0)
    tzero_span = 1000.0
    lower = [
        max(float(seed.difc) - difc_span, 1.0),
        -2000.0,
        float(seed.tzero) - tzero_span,
        1.0,
        0.0,
        0.5,
        0.5,
        *([0.0] * len(reflections)),
        *([-np.inf] * (order + 1)),
    ]
    upper = [
        float(seed.difc) + difc_span,
        2000.0,
        float(seed.tzero) + tzero_span,
        2000.0,
        1.0,
        2000.0,
        2000.0,
        *([max(float(np.max(peak_signal)) * 200.0, 1.0)] * len(reflections)),
        *([np.inf] * (order + 1)),
    ]

    def model(params: np.ndarray) -> np.ndarray:
        calibration = InstrumentCalibration(difc=float(params[0]), difa=float(params[1]), tzero=float(params[2]))
        shape_values = {name: float(value) for name, value in zip(shape_names, params[3 : 3 + len(shape_names)])}
        intensity_start = 3 + len(shape_names)
        bg_start = intensity_start + len(reflections)
        intensities = params[intensity_start:bg_start]
        background = params[bg_start:]
        output = _polyval(background, x_norm)
        for reflection, intensity in zip(reflections, intensities):
            position = float(d_to_tof(reflection.d_spacing, calibration))
            profile_y = evaluate_peak_profile(
                profile.key,
                x_values,
                position,
                shape_values["fwhm"],
                eta=shape_values["eta"],
                alpha=shape_values["alpha"],
                beta=shape_values["beta"],
            )
            output = output + float(intensity) * profile_y
        return output

    def residual(params: np.ndarray) -> np.ndarray:
        values = y_values - model(params)
        if e_values is not None:
            values = values / e_values
        if centre_d.size:
            calibration = InstrumentCalibration(difc=float(params[0]), difa=float(params[1]), tzero=float(params[2]))
            centre_values = (centre_tof - d_to_tof(centre_d, calibration)) / centre_sigma
            values = np.concatenate([values, centre_values])
        return values

    result = least_squares(
        residual,
        np.asarray(p0, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        max_nfev=max(100, int(max_nfev)),
    )
    if not result.success:
        raise ValueError(f"GSAS-style pattern calibration failed: {result.message}")

    params = result.x
    y_fit = model(params)
    calibration = InstrumentCalibration(difc=float(params[0]), difa=float(params[1]), tzero=float(params[2]))
    shape_values = {name: float(value) for name, value in zip(shape_names, params[3 : 3 + len(shape_names)])}
    intensity_start = 3 + len(shape_names)
    bg_start = intensity_start + len(reflections)
    intensities = params[intensity_start:bg_start]
    background = params[bg_start:]
    uncertainties = _parameter_uncertainties(result, len(params))
    calibration_uncertainties = {}
    if uncertainties is not None:
        calibration_uncertainties = {
            "difc_uncertainty": float(uncertainties[0]),
            "difa_uncertainty": float(uncertainties[1]),
            "tzero_uncertainty": float(uncertainties[2]),
        }
    reflection_results = tuple(
        PawleyReflectionResult(
            reflection=reflection,
            position=float(d_to_tof(reflection.d_spacing, calibration)),
            intensity=float(intensity),
        )
        for reflection, intensity in zip(reflections, intensities)
    )
    profile_parameters = {
        key: float(value)
        for key, value in shape_values.items()
        if key != "fwhm"
    }
    profile_parameters.update(calibration_uncertainties)
    profile_parameters["reflection_count"] = float(len(reflections))
    profile_parameters["gsas_tof_function"] = 3.0
    fit_result = PawleyFitResult(
        lattice_a=float(ceo2_phase.a),
        lattice_a_uncertainty=None,
        fwhm=float(shape_values["fwhm"]),
        eta=float(shape_values["eta"]),
        background_coefficients=tuple(float(value) for value in background),
        reflections=reflection_results,
        quality=_fit_quality(y_values, y_fit, e_values, len(params)),
        profile_key=profile.key,
        profile_name=profile.label,
        profile_parameters=profile_parameters,
        fit_x=np.array(x_values, dtype=float, copy=True),
        fit_y=np.array(y_fit, dtype=float, copy=True),
        observed_y=np.array(y_values, dtype=float, copy=True),
    )
    _emit_calibration_progress(
        progress_callback,
        "pattern_fit",
        {
            "fit": fit_result,
            "calibration": calibration,
            "tof_min": fit_tof_min,
            "tof_max": fit_tof_max,
        },
    )
    return calibration, fit_result


def refine_ceo2_calibration(
    ceo2_spectrum: FocusedSpectrum,
    ceo2_phase: PhaseModel,
    *,
    polynomial_order: int = 1,
    minimum_peaks: int = 4,
    progress_callback: Optional[CalibrationProgressCallback] = None,
    tof_min: float = OPENGENIE_CALIBRATION_TOF_MIN,
    tof_max: float = OPENGENIE_CALIBRATION_TOF_MAX,
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
    fit_tof_min = max(float(np.min(x)), min(float(tof_min), float(tof_max)))
    fit_tof_max = min(float(np.max(x)), max(float(tof_min), float(tof_max)))
    if fit_tof_max <= fit_tof_min:
        raise ValueError("The OpenGENIE calibration TOF range does not overlap the calibration histogram.")
    d_axis = tof_to_d(np.array([fit_tof_min, fit_tof_max]), initial)
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
    single_peak_stage_shown = False

    for reflection in reflections:
        expected_tof = float(d_to_tof(reflection.d_spacing, initial))
        if expected_tof < fit_tof_min or expected_tof > fit_tof_max:
            continue
        window = _match_window(expected_tof, x_span)
        nearby = np.flatnonzero(np.abs(x[peak_indices] - expected_tof) <= window)
        if nearby.size == 0:
            local_candidates = np.flatnonzero(np.abs(x - expected_tof) <= window)
            if local_candidates.size == 0:
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
            peak_index = int(local_candidates[int(np.argmax(y[local_candidates]))])
            peak_prominence = None
        else:
            best_local = max(nearby, key=lambda index: float(prominences[index]))
            peak_index = int(peak_indices[int(best_local)])
            peak_prominence = float(prominences[int(best_local)])
        if peak_index in used_peak_indices:
            peak_results.append(
                CalibrationPeakResult(
                    reflection=reflection,
                    expected_tof=expected_tof,
                    observed_tof=float(x[peak_index]),
                    fitted_tof=None,
                    residual_tof=None,
                    prominence=peak_prominence,
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
                    prominence=peak_prominence,
                    accepted=False,
                    rejection_reason="insufficient local points",
                )
            )
            continue
        try:
            fit_result = fit_peak_profile(
                PROFILE_EXP_VOIGT,
                x[fit_mask],
                y[fit_mask],
                ceo2_spectrum.e[fit_mask] if ceo2_spectrum.e is not None else None,
                polynomial_order=polynomial_order,
                max_peaks=1,
            )
        except Exception as exc:
            peak_results.append(
                CalibrationPeakResult(
                    reflection=reflection,
                    expected_tof=expected_tof,
                    observed_tof=float(x[peak_index]),
                    fitted_tof=None,
                    residual_tof=None,
                    prominence=peak_prominence,
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
        if accepted and not single_peak_stage_shown:
            _emit_calibration_progress(
                progress_callback,
                "single_peak_fit",
                {
                    "reflection": reflection,
                    "fit": fit_result,
                    "x": np.array(x[fit_mask], dtype=float, copy=True),
                    "y": np.array(y[fit_mask], dtype=float, copy=True),
                    "expected_tof": expected_tof,
                    "tof_min": fit_tof_min,
                    "tof_max": fit_tof_max,
                },
            )
            single_peak_stage_shown = True
        peak_results.append(
            CalibrationPeakResult(
                reflection=reflection,
                expected_tof=expected_tof,
                observed_tof=float(x[peak_index]),
                fitted_tof=float(fit_result.centre),
                residual_tof=residual,
                prominence=peak_prominence,
                accepted=accepted,
                rejection_reason=reason,
                tof_uncertainty=fit_result.centre_uncertainty,
                fwhm=fit_result.fwhm,
                fwhm_uncertainty=fit_result.fwhm_uncertainty,
                height=fit_result.height,
                exponential_decay=fit_result.profile_parameters.get("tail"),
                lorentz_fraction=fit_result.eta,
            )
        )

    single_peak_calibration, single_peak_rms, updated_results = _linear_calibration_from_peaks(
        peak_results,
        minimum_peaks,
        progress_callback,
    )
    refined, pattern_fit = _fit_ceo2_pattern_calibration(
        ceo2_spectrum,
        ceo2_phase,
        single_peak_calibration,
        updated_results,
        polynomial_order=polynomial_order,
        progress_callback=progress_callback,
        tof_min=fit_tof_min,
        tof_max=fit_tof_max,
    )
    pattern_residuals = [
        float(peak.fitted_tof - d_to_tof(peak.reflection.d_spacing, refined))
        for peak in updated_results
        if peak.accepted and peak.fitted_tof is not None
    ]
    rms = float(np.sqrt(np.mean(np.asarray(pattern_residuals, dtype=float) ** 2))) if pattern_residuals else single_peak_rms
    return CalibrationResult(
        calibration=refined,
        initial_calibration=initial,
        single_peak_calibration=single_peak_calibration,
        phase_name=ceo2_phase.name,
        source_path=Path(ceo2_spectrum.source_path),
        run_number=ceo2_spectrum.run_number,
        bank_number=ceo2_spectrum.bank_number,
        peak_results=tuple(updated_results),
        rms_residual_tof=rms,
        single_peak_rms_residual_tof=single_peak_rms,
        pattern_fit=pattern_fit,
        pattern_profile_parameters=pattern_fit.profile_parameters,
        pattern_lattice_a=pattern_fit.lattice_a,
        pattern_lattice_a_uncertainty=pattern_fit.lattice_a_uncertainty,
    )
