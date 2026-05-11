from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .models import (
    AXIS_TOF,
    CalibrationResult,
    FocusedSpectrum,
    InstrumentCalibration,
    NormalizationResult,
)


def _validate_bank_match(sample: FocusedSpectrum, vanadium: FocusedSpectrum) -> None:
    if sample.bank_number is not None and vanadium.bank_number is not None and sample.bank_number != vanadium.bank_number:
        raise ValueError(
            f"Bank mismatch: sample bank {sample.bank_number} cannot use vanadium bank {vanadium.bank_number}."
        )


def _smooth_response(values: np.ndarray) -> tuple[np.ndarray, bool]:
    finite = np.isfinite(values)
    if int(np.count_nonzero(finite)) < 21:
        return values, False
    from scipy.signal import savgol_filter

    window = min(401, max(21, (values.size // 25) | 1))
    if window >= values.size:
        window = values.size - 1 if values.size % 2 == 0 else values.size
    if window < 21:
        return values, False
    if window % 2 == 0:
        window -= 1
    filled = np.array(values, dtype=float, copy=True)
    if not np.all(finite):
        indices = np.arange(values.size, dtype=float)
        filled[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    smoothed = savgol_filter(filled, window_length=window, polyorder=3, mode="interp")
    return smoothed.astype(float, copy=False), True


def apply_vanadium_normalization(
    sample: FocusedSpectrum,
    vanadium: FocusedSpectrum,
    calibration_result: Optional[CalibrationResult] = None,
) -> NormalizationResult:
    _validate_bank_match(sample, vanadium)
    sample_x = sample.axis_values(AXIS_TOF)
    vanadium_x = vanadium.axis_values(AXIS_TOF)
    order = np.argsort(vanadium_x)
    vanadium_x = vanadium_x[order]
    vanadium_y = vanadium.y[order]
    vanadium_e = vanadium.e[order] if vanadium.e is not None else None

    response = np.interp(sample_x, vanadium_x, vanadium_y, left=np.nan, right=np.nan)
    response_e = None
    if vanadium_e is not None:
        response_e = np.interp(sample_x, vanadium_x, vanadium_e, left=np.nan, right=np.nan)
    smoothed_response, smoothed = _smooth_response(response)
    valid = (
        np.isfinite(smoothed_response)
        & (smoothed_response > 0.0)
        & np.isfinite(sample.y)
        & (sample_x >= float(np.min(vanadium_x)))
        & (sample_x <= float(np.max(vanadium_x)))
    )
    if int(np.count_nonzero(valid)) == 0:
        raise ValueError("Vanadium normalisation produced no valid bins.")
    scale = float(np.nanmedian(smoothed_response[valid]))
    factor = np.full(sample.y.shape, np.nan, dtype=float)
    factor[valid] = scale / smoothed_response[valid]
    corrected_y = sample.y * factor

    corrected_e = None
    if sample.e is not None:
        sample_component = sample.e * factor
        if response_e is not None:
            vanadium_component = np.full(sample.y.shape, np.nan, dtype=float)
            response_error = np.asarray(response_e, dtype=float)
            vanadium_component[valid] = np.abs(sample.y[valid] * scale * response_error[valid] / (smoothed_response[valid] ** 2))
            corrected_e = np.sqrt(sample_component * sample_component + vanadium_component * vanadium_component)
        else:
            corrected_e = sample_component

    calibration: Optional[InstrumentCalibration] = sample.calibration
    calibration_metadata: dict[str, object] = {}
    if calibration_result is not None:
        if (
            sample.bank_number is not None
            and calibration_result.bank_number is not None
            and sample.bank_number != calibration_result.bank_number
        ):
            raise ValueError(
                f"Bank mismatch: sample bank {sample.bank_number} cannot use CeO2 bank {calibration_result.bank_number}."
            )
        calibration = calibration_result.calibration
        calibration_metadata = {
            "calibration_run": calibration_result.run_number,
            "calibration_source": str(calibration_result.source_path),
            "calibration_phase": calibration_result.phase_name,
            "calibration_rms_tof": calibration_result.rms_residual_tof,
        }

    metadata = dict(sample.metadata)
    metadata.update(
        {
            "normalization_source": str(vanadium.source_path),
            "normalization_run": vanadium.run_number,
            "normalization_scale": scale,
            "normalization_valid_bins": int(np.count_nonzero(valid)),
            "normalization_invalid_bins": int(sample.y.size - np.count_nonzero(valid)),
            "normalization_smoothed": smoothed,
        }
    )
    metadata.update(calibration_metadata)
    corrected = FocusedSpectrum(
        source_path=sample.source_path,
        x=np.array(sample.x, dtype=float, copy=True),
        y=corrected_y,
        e=corrected_e,
        native_axis=sample.native_axis,
        x_is_edges=sample.x_is_edges,
        calibration=calibration,
        metadata=metadata,
        run_number=sample.run_number,
        bank_number=sample.bank_number,
        bank_name=sample.bank_name,
        x_label=sample.x_label,
        y_label=f"{sample.y_label} / vanadium",
    )
    return NormalizationResult(
        corrected_spectrum=corrected,
        sample_source_path=Path(sample.source_path),
        vanadium_source_path=Path(vanadium.source_path),
        sample_run_number=sample.run_number,
        vanadium_run_number=vanadium.run_number,
        bank_number=sample.bank_number,
        scale=scale,
        valid_bins=int(np.count_nonzero(valid)),
        invalid_bins=int(sample.y.size - np.count_nonzero(valid)),
        smoothed=smoothed,
    )
