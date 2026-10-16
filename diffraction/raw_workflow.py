from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .calibration import refine_raw_ceo2_bank_calibration
from .focusing import FOCUSING_CONVENTION_VERSION, focus_enginx_bank, rebin_histogram_overlap
from .models import CalibrationResult, FocusedSpectrum, InstrumentCalibration, PhaseModel, spectrum_with_calibration
from .nexus import read_enginx_histogram_nexus
from .normalization import _smooth_response


RAW_WORKFLOW_FORMAT = "SimSetup ENGIN-X raw calibration"
RAW_WORKFLOW_VERSION = 2
RESPONSE_VALUE_DEFINITION = "charge-normalized histogram bin integral"
RESPONSE_UNITS = "counts/uAh"


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
    calibration_model: str
    difc: float
    tzero: float
    difa: float
    coefficient_units: Mapping[str, str]
    difc_uncertainty: Optional[float]
    tzero_uncertainty: Optional[float]
    difa_uncertainty: Optional[float]
    rms_residual_tof: float
    usable_tof_range_us: tuple[float, float]
    peak_diagnostics: tuple[Mapping[str, object], ...]
    focusing: Mapping[str, object]
    reference_response_settings: Mapping[str, object]
    response_bin_edges_us: tuple[float, ...]
    response_values_per_uah: tuple[float, ...]
    response_uncertainties_per_uah: tuple[float, ...]
    response_valid_mask: tuple[bool, ...]
    response_coverage_fraction: tuple[float, ...]
    response_value_definition: str = RESPONSE_VALUE_DEFINITION
    response_units: str = RESPONSE_UNITS


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


def bundle_fingerprint(bundle: RawCalibrationBundle) -> str:
    encoded = json.dumps(asdict(bundle), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _reference_response(
    focused_reference: FocusedSpectrum,
    *,
    smooth: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    charge = float(focused_reference.metadata.get("proton_charge", np.nan))
    if not np.isfinite(charge) or charge <= 0.0:
        raise ValueError("Focused reference has no valid proton charge.")
    edges = np.asarray(focused_reference.x, dtype=float)
    widths = np.diff(edges)
    values = np.asarray(focused_reference.y, dtype=float) / charge
    uncertainty = (
        np.sqrt(np.maximum(focused_reference.y, 0.0)) / charge
        if focused_reference.e is None
        else np.asarray(focused_reference.e, dtype=float) / charge
    )
    coverage = np.asarray(focused_reference.metadata.get("detector_coverage_fraction", []), dtype=float)
    if coverage.shape != values.shape:
        raise ValueError("Focused reference is missing detector-coverage metadata.")
    if smooth:
        density = values / widths
        smoothed_density, applied = _smooth_response(density)
        if not applied:
            raise ValueError("Reference response is too short for the requested smoothing model.")
        values = smoothed_density * widths
    valid = (
        np.isfinite(values)
        & (values > 0.0)
        & np.isfinite(uncertainty)
        & (uncertainty >= 0.0)
        & (coverage >= 1.0 - 1e-10)
    )
    return values, uncertainty, valid, coverage


def _rebin_valid_coverage(source_edges: np.ndarray, valid: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    source_widths = np.diff(source_edges)
    valid_width, _ = rebin_histogram_overlap(
        source_edges,
        source_widths * np.asarray(valid, dtype=float),
        target_edges,
        np.zeros(source_widths.size, dtype=float),
    )
    return valid_width / np.diff(target_edges)


def apply_saved_reference_response(sample: FocusedSpectrum, record: RawBankCalibrationRecord) -> FocusedSpectrum:
    if sample.bank_number != record.bank_number:
        raise ValueError(f"Sample bank {sample.bank_number} cannot use response bank {record.bank_number}.")
    if not sample.x_is_edges:
        raise ValueError("Raw NXS reference correction requires histogram bin edges.")
    charge = float(sample.metadata.get("proton_charge", np.nan))
    if not np.isfinite(charge) or charge <= 0.0:
        raise ValueError("Sample proton charge must be finite and greater than zero.")
    source_edges = np.asarray(record.response_bin_edges_us, dtype=float)
    source_values = np.asarray(record.response_values_per_uah, dtype=float)
    source_uncertainty = np.asarray(record.response_uncertainties_per_uah, dtype=float)
    source_valid = np.asarray(record.response_valid_mask, dtype=bool)
    target_edges = np.asarray(sample.x, dtype=float)
    response, response_variance = rebin_histogram_overlap(
        source_edges, source_values, target_edges, source_uncertainty * source_uncertainty
    )
    response_valid_coverage = _rebin_valid_coverage(source_edges, source_valid, target_edges)
    sample_coverage = np.asarray(sample.metadata.get("detector_coverage_fraction", []), dtype=float)
    if sample_coverage.shape != sample.y.shape:
        raise ValueError("Focused sample is missing detector-coverage metadata.")
    sample_rate = np.asarray(sample.y, dtype=float) / charge
    sample_rate_e = (
        np.sqrt(np.maximum(sample.y, 0.0)) / charge
        if sample.e is None
        else np.asarray(sample.e, dtype=float) / charge
    )
    valid = (
        np.isfinite(sample_rate)
        & np.isfinite(response)
        & (response > 0.0)
        & (response_valid_coverage >= 1.0 - 1e-10)
        & (sample_coverage >= 1.0 - 1e-10)
    )
    if not np.any(valid):
        raise ValueError("Sample and saved reference have no bins with full shared detector coverage.")
    corrected_y = np.full(sample.y.shape, np.nan, dtype=float)
    corrected_e = np.full(sample.y.shape, np.nan, dtype=float)
    corrected_y[valid] = sample_rate[valid] / response[valid]
    sample_term = sample_rate_e[valid] / response[valid]
    reference_term = sample_rate[valid] * np.sqrt(response_variance[valid]) / (response[valid] ** 2)
    corrected_e[valid] = np.sqrt(sample_term * sample_term + reference_term * reference_term)
    metadata = dict(sample.metadata)
    metadata.update(
        {
            "normalization_formula": "(C_sample/Q_sample)/(saved C_reference/Q_reference)",
            "normalization_response_definition": record.response_value_definition,
            "normalization_response_units": record.response_units,
            "normalization_valid_bins": int(np.count_nonzero(valid)),
            "normalization_invalid_bins": int(valid.size - np.count_nonzero(valid)),
            "shared_detector_coverage_mask": valid.tolist(),
            "reference_response_rebinned": not np.array_equal(source_edges, target_edges),
        }
    )
    return FocusedSpectrum(
        source_path=sample.source_path,
        x=np.array(target_edges, copy=True),
        y=corrected_y,
        e=corrected_e,
        native_axis=sample.native_axis,
        x_is_edges=True,
        calibration=sample.calibration,
        metadata=metadata,
        run_number=sample.run_number,
        bank_number=sample.bank_number,
        bank_name=sample.bank_name,
        x_label=sample.x_label,
        y_label="Charge-normalized sample / saved reference",
    )


def _bank_record_from_result(
    focused_reference: FocusedSpectrum,
    focused_ceo2: FocusedSpectrum,
    calibration_result: CalibrationResult,
    *,
    smooth_reference: bool,
) -> RawBankCalibrationRecord:
    if focused_ceo2.bank_number is None or focused_reference.bank_number != focused_ceo2.bank_number:
        raise ValueError("Focused reference and CeO2 spectra must identify the same bank.")
    response, response_uncertainty, response_valid, response_coverage = _reference_response(
        focused_reference, smooth=smooth_reference
    )
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
        "processing_version", "profile_name", "primary_flight_path_m", "primary_flight_path_source",
        "nxs_source_detector_distance_m", "bank_spectrum_ids", "masked_spectrum_ids", "pixel_grouping",
        "focusing_map", "geometry_difc_source", "target_geometric_difc",
        "pixel_geometric_difc_by_spectrum", "uncertainty_model", "transmission_spectra_excluded",
    )
    valid_centres = focused_ceo2.x_centres[np.isfinite(focused_ceo2.y)]
    return RawBankCalibrationRecord(
        bank_number=focused_ceo2.bank_number,
        bank_name=focused_ceo2.bank_name or f"Bank {focused_ceo2.bank_number}",
        calibration_model="quadratic DIFC/DIFA/TZERO from individual CeO2 peak fits",
        difc=calibration_result.calibration.difc,
        tzero=calibration_result.calibration.tzero,
        difa=calibration_result.calibration.difa,
        coefficient_units={"difc": "microsecond/angstrom", "tzero": "microsecond", "difa": "microsecond/angstrom^2"},
        difc_uncertainty=_optional_float(fit_parameters.get("difc_uncertainty")),
        tzero_uncertainty=_optional_float(fit_parameters.get("tzero_uncertainty")),
        difa_uncertainty=_optional_float(fit_parameters.get("difa_uncertainty")),
        rms_residual_tof=calibration_result.rms_residual_tof,
        usable_tof_range_us=(float(valid_centres.min()), float(valid_centres.max())),
        peak_diagnostics=peak_diagnostics,
        focusing={key: focused_reference.metadata.get(key) for key in focusing_keys},
        reference_response_settings={
            "smoothed": smooth_reference,
            "model": "Savitzky-Golay density smoothing" if smooth_reference else "unsmoothed histogram",
            "uncertainty_note": "Poisson counting uncertainty; smoothing-model uncertainty is not estimated.",
        },
        response_bin_edges_us=tuple(float(value) for value in focused_reference.x),
        response_values_per_uah=tuple(float(value) for value in response),
        response_uncertainties_per_uah=tuple(float(value) for value in response_uncertainty),
        response_valid_mask=tuple(bool(value) for value in response_valid),
        response_coverage_fraction=tuple(float(value) for value in response_coverage),
    )


def create_raw_calibration_bundle(
    reference_path: str | Path,
    ceo2_path: str | Path,
    ceo2_phase: PhaseModel,
    *,
    masked_spectrum_ids: Mapping[int, set[int]] | None = None,
    smooth_reference: bool = False,
) -> tuple[RawCalibrationBundle, Mapping[int, Mapping[str, object]]]:
    reference_run = read_enginx_histogram_nexus(reference_path)
    ceo2_run = read_enginx_histogram_nexus(ceo2_path)
    masks = masked_spectrum_ids or {}
    records = []
    diagnostics: dict[int, Mapping[str, object]] = {}
    for bank_number in (1, 2):
        mask = masks.get(bank_number, set())
        focused_reference, reference_diagnostics = focus_enginx_bank(
            reference_run, bank_number, masked_spectrum_ids=mask, target_edges_us=ceo2_run.tof_edges_us
        )
        saved_pixel_map = {
            int(key): float(value)
            for key, value in focused_reference.metadata["pixel_geometric_difc_by_spectrum"].items()
        }
        focused_ceo2, ceo2_diagnostics = focus_enginx_bank(
            ceo2_run,
            bank_number,
            masked_spectrum_ids=mask,
            target_edges_us=np.asarray(focused_reference.x, dtype=float),
            target_geometric_difc=float(focused_reference.metadata["target_geometric_difc"]),
            pixel_geometric_difc_by_spectrum=saved_pixel_map,
        )
        response, response_uncertainty, response_valid, response_coverage = _reference_response(
            focused_reference, smooth=smooth_reference
        )
        provisional_record = RawBankCalibrationRecord(
            bank_number=bank_number, bank_name=focused_reference.bank_name or f"Bank {bank_number}",
            calibration_model="pending", difc=1.0, tzero=0.0, difa=0.0, coefficient_units={},
            difc_uncertainty=None, tzero_uncertainty=None, difa_uncertainty=None, rms_residual_tof=np.nan,
            usable_tof_range_us=(float(focused_reference.x_centres.min()), float(focused_reference.x_centres.max())),
            peak_diagnostics=(), focusing={}, reference_response_settings={},
            response_bin_edges_us=tuple(float(value) for value in focused_reference.x),
            response_values_per_uah=tuple(float(value) for value in response),
            response_uncertainties_per_uah=tuple(float(value) for value in response_uncertainty),
            response_valid_mask=tuple(bool(value) for value in response_valid),
            response_coverage_fraction=tuple(float(value) for value in response_coverage),
        )
        corrected_ceo2 = apply_saved_reference_response(focused_ceo2, provisional_record)
        local_peak_fits: dict[str, object] = {}

        def retain_local_fit(stage: str, payload: Mapping[str, object]) -> None:
            if stage == "local_peak_fit":
                reflection = payload["reflection"]
                local_peak_fits[reflection.label] = payload["fit"]

        result = refine_raw_ceo2_bank_calibration(
            corrected_ceo2, ceo2_phase, progress_callback=retain_local_fit
        )
        records.append(_bank_record_from_result(focused_reference, corrected_ceo2, result, smooth_reference=smooth_reference))
        diagnostics[bank_number] = {
            "reference": reference_diagnostics,
            "ceo2": ceo2_diagnostics,
            "calibration": result,
            "focused_reference": focused_reference,
            "corrected_ceo2": corrected_ceo2,
            "local_peak_fits": local_peak_fits,
        }
    bundle = RawCalibrationBundle(
        processing_version=FOCUSING_CONVENTION_VERSION,
        source_runs={
            "reference": {"run": reference_run.run_number, "title": reference_run.title, "charge_uah": reference_run.proton_charge_uah, "file_identity": file_identity(reference_path)},
            "ceo2": {"run": ceo2_run.run_number, "title": ceo2_run.title, "charge_uah": ceo2_run.proton_charge_uah, "file_identity": file_identity(ceo2_path)},
        },
        ceo2_standard={"name": ceo2_phase.name, "lattice_parameter_angstrom": ceo2_phase.a, "source": "Hist/CEO2.EXP"},
        scientific_status="provisional: internally fitted, not independently absolute-validated",
        remaining_validation_gaps=(
            "No independently trusted focused ENGIN-X reference was available for absolute validation.",
            "The 50 m primary flight path is nominal; NXS source_detector_distance is zero.",
            "V-Nb reference structure and detector-efficiency separation remain unvalidated.",
            "The empirical fitted DIFA has not been independently separated from focusing or geometry error.",
        ),
        banks=tuple(records),
    )
    validate_raw_calibration_bundle(bundle)
    return bundle, diagnostics


def validate_raw_calibration_bundle(bundle: RawCalibrationBundle) -> None:
    if bundle.processing_version != FOCUSING_CONVENTION_VERSION:
        raise ValueError(f"Unsupported raw calibration processing version: {bundle.processing_version!r}.")
    if {record.bank_number for record in bundle.banks} != {1, 2}:
        raise ValueError("Raw calibration must contain complete records for banks 1 and 2.")
    for record in bundle.banks:
        legacy = record.calibration_model == "linear DIFC/TZERO from individual CeO2 peak fits" and record.difa == 0.0
        quadratic = record.calibration_model == "quadratic DIFC/DIFA/TZERO from individual CeO2 peak fits"
        if not (legacy or quadratic) or not np.all(np.isfinite([record.difc, record.difa, record.tzero])):
            raise ValueError(f"Bank {record.bank_number} has invalid calibration coefficients or model.")
        if quadratic and any(
            value is None or not np.isfinite(value) or value < 0.0
            for value in (record.difc_uncertainty, record.difa_uncertainty, record.tzero_uncertainty)
        ):
            raise ValueError(f"Bank {record.bank_number} has invalid quadratic coefficient uncertainties.")
        edges = np.asarray(record.response_bin_edges_us, dtype=float)
        values = np.asarray(record.response_values_per_uah, dtype=float)
        uncertainty = np.asarray(record.response_uncertainties_per_uah, dtype=float)
        valid = np.asarray(record.response_valid_mask, dtype=bool)
        coverage = np.asarray(record.response_coverage_fraction, dtype=float)
        if edges.size != values.size + 1 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise ValueError(f"Bank {record.bank_number} has invalid reference-response bin edges.")
        if uncertainty.shape != values.shape or valid.shape != values.shape or coverage.shape != values.shape:
            raise ValueError(f"Bank {record.bank_number} has inconsistent reference-response arrays.")
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(uncertainty)) or np.any(uncertainty < 0.0):
            raise ValueError(f"Bank {record.bank_number} has invalid reference-response values.")
        if not np.any(valid) or np.any(coverage < 0.0) or np.any(coverage > 1.0 + 1e-10):
            raise ValueError(f"Bank {record.bank_number} has invalid reference-response coverage.")
        if not record.focusing.get("pixel_geometric_difc_by_spectrum") or not record.focusing.get("target_geometric_difc"):
            raise ValueError(f"Bank {record.bank_number} is missing its saved focusing basis.")


def analyze_sample_nexus(path: str | Path, bundle: RawCalibrationBundle) -> tuple[FocusedSpectrum, ...]:
    validate_raw_calibration_bundle(bundle)
    run = read_enginx_histogram_nexus(path)
    fingerprint = bundle_fingerprint(bundle)
    output = []
    for record in bundle.banks:
        focusing = record.focusing
        pixel_map = {int(key): float(value) for key, value in focusing["pixel_geometric_difc_by_spectrum"].items()}
        focused, _diagnostics = focus_enginx_bank(
            run,
            record.bank_number,
            masked_spectrum_ids={int(value) for value in focusing.get("masked_spectrum_ids", [])},
            target_edges_us=np.asarray(record.response_bin_edges_us, dtype=float),
            target_geometric_difc=float(focusing["target_geometric_difc"]),
            pixel_geometric_difc_by_spectrum=pixel_map,
        )
        corrected = apply_saved_reference_response(focused, record)
        calibrated = apply_bundle_calibration(corrected, bundle)
        calibrated.metadata.update({
            "raw_nxs_processed": True,
            "calibration_bundle_fingerprint": fingerprint,
            "calibration_source_kind": "raw_nxs_json",
        })
        output.append(calibrated)
    return tuple(output)


def save_raw_calibration_bundle(bundle: RawCalibrationBundle, path: str | Path) -> None:
    validate_raw_calibration_bundle(bundle)
    payload = {"format": RAW_WORKFLOW_FORMAT, "version": RAW_WORKFLOW_VERSION, **asdict(bundle)}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_raw_calibration_bundle(path: str | Path) -> RawCalibrationBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != RAW_WORKFLOW_FORMAT or payload.get("version") != RAW_WORKFLOW_VERSION:
        raise ValueError("Unsupported SimSetup raw calibration file format or version.")
    banks = tuple(
        RawBankCalibrationRecord(**{
            **record,
            "usable_tof_range_us": tuple(record["usable_tof_range_us"]),
            "peak_diagnostics": tuple(record["peak_diagnostics"]),
            "response_bin_edges_us": tuple(record["response_bin_edges_us"]),
            "response_values_per_uah": tuple(record["response_values_per_uah"]),
            "response_uncertainties_per_uah": tuple(record["response_uncertainties_per_uah"]),
            "response_valid_mask": tuple(record["response_valid_mask"]),
            "response_coverage_fraction": tuple(record["response_coverage_fraction"]),
        }) for record in payload["banks"]
    )
    bundle = RawCalibrationBundle(
        processing_version=str(payload["processing_version"]), source_runs=dict(payload["source_runs"]),
        ceo2_standard=dict(payload["ceo2_standard"]), scientific_status=str(payload["scientific_status"]),
        remaining_validation_gaps=tuple(payload["remaining_validation_gaps"]), banks=banks,
    )
    validate_raw_calibration_bundle(bundle)
    return bundle


def apply_bundle_calibration(spectrum: FocusedSpectrum, bundle: RawCalibrationBundle) -> FocusedSpectrum:
    validate_raw_calibration_bundle(bundle)
    if spectrum.bank_number is None:
        raise ValueError("A bank number is required to apply raw ENGIN-X calibration.")
    if int(spectrum.metadata.get("calibration_application_count", 0)) != 0:
        raise ValueError("Calibration has already been applied to this focused spectrum.")
    record = bundle.bank(spectrum.bank_number)
    if spectrum.metadata.get("processing_version") != bundle.processing_version:
        raise ValueError("Focused spectrum processing version does not match the calibration bundle.")
    observed_target = float(spectrum.metadata.get("target_geometric_difc", np.nan))
    saved_target = float(record.focusing["target_geometric_difc"])
    if not np.isclose(observed_target, saved_target, rtol=1e-12, atol=1e-9):
        raise ValueError("Focused spectrum basis does not match the saved calibration basis.")
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
            "accepted_calibration_model": record.calibration_model,
        },
    )
