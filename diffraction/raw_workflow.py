from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .models import CalibrationResult, FocusedSpectrum, InstrumentCalibration, spectrum_with_calibration


RAW_WORKFLOW_FORMAT = "SimSetup ENGIN-X raw calibration"
RAW_WORKFLOW_VERSION = 1


def file_identity(path: str | Path) -> dict[str, object]:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(source), "size": source.stat().st_size, "sha256": digest.hexdigest()}


@dataclass(frozen=True)
class RawBankCalibrationRecord:
    bank_number: int
    bank_name: str
    difc: float
    tzero: float
    difa: float
    difc_uncertainty: Optional[float]
    tzero_uncertainty: Optional[float]
    difa_uncertainty: Optional[float]
    rms_residual_tof: float
    usable_tof_range_us: tuple[float, float]
    peak_diagnostics: tuple[Mapping[str, object], ...]
    focusing: Mapping[str, object]
    reference_response_settings: Mapping[str, object]


@dataclass(frozen=True)
class RawCalibrationBundle:
    processing_version: str
    source_runs: Mapping[str, Mapping[str, object]]
    ceo2_standard: Mapping[str, object]
    scientific_status: str
    remaining_validation_gaps: tuple[str, ...]
    banks: tuple[RawBankCalibrationRecord, ...]

    def bank(self, bank_number: int) -> RawBankCalibrationRecord:
        for record in self.banks:
            if record.bank_number == bank_number:
                return record
        raise ValueError(f"Calibration bundle does not contain bank {bank_number}.")


def bank_record_from_result(
    focused_ceo2: FocusedSpectrum,
    calibration_result: CalibrationResult,
    *,
    reference_response_settings: Mapping[str, object],
) -> RawBankCalibrationRecord:
    if focused_ceo2.bank_number is None:
        raise ValueError("Focused CeO2 spectrum must identify its bank.")
    fit_parameters = calibration_result.pattern_profile_parameters
    peak_diagnostics = tuple(
        {
            "hkl": peak.reflection.label,
            "d_spacing_angstrom": peak.reflection.d_spacing,
            "expected_tof_us": peak.expected_tof,
            "fitted_tof_us": peak.fitted_tof,
            "tof_uncertainty_us": peak.tof_uncertainty,
            "residual_tof_us": peak.residual_tof,
            "accepted": peak.accepted,
            "rejection_reason": peak.rejection_reason,
        }
        for peak in calibration_result.peak_results
    )
    focusing_keys = (
        "processing_version",
        "profile_name",
        "primary_flight_path_m",
        "primary_flight_path_source",
        "nxs_source_detector_distance_m",
        "bank_spectrum_ids",
        "masked_spectrum_ids",
        "pixel_grouping",
        "focusing_map",
        "geometry_difc_source",
        "target_geometric_difc",
        "uncertainty_model",
        "transmission_spectra_excluded",
    )
    return RawBankCalibrationRecord(
        bank_number=focused_ceo2.bank_number,
        bank_name=focused_ceo2.bank_name or f"Bank {focused_ceo2.bank_number}",
        difc=calibration_result.calibration.difc,
        tzero=calibration_result.calibration.tzero,
        difa=calibration_result.calibration.difa,
        difc_uncertainty=_optional_float(fit_parameters.get("difc_uncertainty")),
        tzero_uncertainty=_optional_float(fit_parameters.get("tzero_uncertainty")),
        difa_uncertainty=_optional_float(fit_parameters.get("difa_uncertainty")),
        rms_residual_tof=calibration_result.rms_residual_tof,
        usable_tof_range_us=(float(focused_ceo2.x_centres.min()), float(focused_ceo2.x_centres.max())),
        peak_diagnostics=peak_diagnostics,
        focusing={key: focused_ceo2.metadata.get(key) for key in focusing_keys},
        reference_response_settings=dict(reference_response_settings),
    )


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def save_raw_calibration_bundle(bundle: RawCalibrationBundle, path: str | Path) -> None:
    payload = {"format": RAW_WORKFLOW_FORMAT, "version": RAW_WORKFLOW_VERSION, **asdict(bundle)}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_raw_calibration_bundle(path: str | Path) -> RawCalibrationBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != RAW_WORKFLOW_FORMAT or payload.get("version") != RAW_WORKFLOW_VERSION:
        raise ValueError("Unsupported SimSetup raw calibration file format or version.")
    banks = tuple(
        RawBankCalibrationRecord(
            **{
                **record,
                "usable_tof_range_us": tuple(record["usable_tof_range_us"]),
                "peak_diagnostics": tuple(record["peak_diagnostics"]),
            }
        )
        for record in payload["banks"]
    )
    return RawCalibrationBundle(
        processing_version=str(payload["processing_version"]),
        source_runs=dict(payload["source_runs"]),
        ceo2_standard=dict(payload["ceo2_standard"]),
        scientific_status=str(payload["scientific_status"]),
        remaining_validation_gaps=tuple(payload["remaining_validation_gaps"]),
        banks=banks,
    )


def apply_bundle_calibration(spectrum: FocusedSpectrum, bundle: RawCalibrationBundle) -> FocusedSpectrum:
    if spectrum.bank_number is None:
        raise ValueError("A bank number is required to apply raw ENGIN-X calibration.")
    count = int(spectrum.metadata.get("calibration_application_count", 0))
    if count != 0:
        raise ValueError("Calibration has already been applied to this focused spectrum.")
    record = bundle.bank(spectrum.bank_number)
    low, high = record.usable_tof_range_us
    centres = spectrum.x_centres
    if float(np.max(centres)) < low or float(np.min(centres)) > high:
        raise ValueError("Spectrum TOF range does not overlap the saved calibration's usable range.")
    return spectrum_with_calibration(
        spectrum,
        InstrumentCalibration(difc=record.difc, tzero=record.tzero, difa=record.difa),
        {
            "calibration_application_count": 1,
            "raw_calibration_processing_version": bundle.processing_version,
            "raw_calibration_scientific_status": bundle.scientific_status,
            "calibration_rms_tof": record.rms_residual_tof,
            "calibration_usable_tof_range_us": list(record.usable_tof_range_us),
        },
    )
