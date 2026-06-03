from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class AcquisitionObservation:
    label: str
    path_length_mm: float
    gauge_volume_mm3: float
    exposure_uamp: float
    uncertainty: float

    def __post_init__(self) -> None:
        for name, value in (
            ("path length", self.path_length_mm),
            ("gauge volume", self.gauge_volume_mm3),
            ("exposure", self.exposure_uamp),
            ("uncertainty", self.uncertainty),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"Calibration {name} must be finite.")
        if self.path_length_mm < 0.0:
            raise ValueError("Calibration path length cannot be negative.")
        if self.gauge_volume_mm3 <= 0.0:
            raise ValueError("Calibration gauge volume must be positive.")
        if self.exposure_uamp <= 0.0:
            raise ValueError("Calibration exposure must be positive.")
        if self.uncertainty <= 0.0:
            raise ValueError("Calibration uncertainty must be positive.")

    def normalized_log_exposure(self) -> float:
        return float(
            math.log(
                float(self.exposure_uamp)
                * float(self.gauge_volume_mm3)
                * float(self.uncertainty) ** 2
            )
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "path_length_mm": float(self.path_length_mm),
            "gauge_volume_mm3": float(self.gauge_volume_mm3),
            "exposure_uamp": float(self.exposure_uamp),
            "uncertainty": float(self.uncertainty),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AcquisitionObservation":
        return cls(
            label=str(payload.get("label", "")),
            path_length_mm=float(payload["path_length_mm"]),
            gauge_volume_mm3=float(payload["gauge_volume_mm3"]),
            exposure_uamp=float(payload["exposure_uamp"]),
            uncertainty=float(payload["uncertainty"]),
        )


@dataclass(frozen=True)
class MaterialAcquisitionProfile:
    name: str
    uncertainty_metric: str
    bank_scope: str
    exposure_unit: str
    observations: tuple[AcquisitionObservation, ...]
    mode: str
    mu_per_mm: Optional[float]
    log_scale: Optional[float]
    rms_log_residual: Optional[float]

    @property
    def path_range_mm(self) -> tuple[float, float]:
        paths = [float(observation.path_length_mm) for observation in self.observations]
        return min(paths), max(paths)

    def required_uamp(
        self,
        path_length_mm: float,
        gauge_volume_mm3: float,
        target_uncertainty: float,
    ) -> float:
        path_length_mm = float(path_length_mm)
        gauge_volume_mm3 = float(gauge_volume_mm3)
        target_uncertainty = float(target_uncertainty)
        if not math.isfinite(path_length_mm) or path_length_mm < 0.0:
            raise ValueError("Prediction path length must be finite and non-negative.")
        if not math.isfinite(gauge_volume_mm3) or gauge_volume_mm3 <= 0.0:
            raise ValueError("Prediction gauge volume must be positive.")
        if not math.isfinite(target_uncertainty) or target_uncertainty <= 0.0:
            raise ValueError("Target uncertainty must be positive.")

        if self.mu_per_mm is None or self.log_scale is None:
            reference_path = float(self.observations[0].path_length_mm)
            if not math.isclose(path_length_mm, reference_path, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"{self.name} has a single-path calibration at {reference_path:g} mm. "
                    "Add a distinct path length or supply effective mu before predicting other paths."
                )
            normalized_logs = [observation.normalized_log_exposure() for observation in self.observations]
            log_value = float(np.mean(normalized_logs))
        else:
            log_value = float(self.log_scale) + float(self.mu_per_mm) * path_length_mm
        return float(math.exp(log_value) / (gauge_volume_mm3 * target_uncertainty**2))

    def normalized_log_exposure_for_path(self, path_length_mm: float) -> float:
        path_length_mm = float(path_length_mm)
        if not math.isfinite(path_length_mm) or path_length_mm < 0.0:
            raise ValueError("Path length must be finite and non-negative.")
        if self.mu_per_mm is None or self.log_scale is None:
            reference_path = float(self.observations[0].path_length_mm)
            if not math.isclose(path_length_mm, reference_path, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"{self.name} has a single-path calibration at {reference_path:g} mm. "
                    "Add a distinct path length or supply effective mu before estimating other paths."
                )
            return float(np.mean([observation.normalized_log_exposure() for observation in self.observations]))
        return float(self.log_scale) + float(self.mu_per_mm) * path_length_mm

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "uncertainty_metric": self.uncertainty_metric,
            "bank_scope": self.bank_scope,
            "exposure_unit": self.exposure_unit,
            "observations": [observation.to_dict() for observation in self.observations],
            "mode": self.mode,
            "mu_per_mm": self.mu_per_mm,
            "log_scale": self.log_scale,
            "rms_log_residual": self.rms_log_residual,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MaterialAcquisitionProfile":
        observations = tuple(
            AcquisitionObservation.from_dict(item) for item in payload.get("observations", [])
        )
        if not observations:
            raise ValueError("A material acquisition profile must contain observations.")
        return cls(
            name=str(payload["name"]).strip(),
            uncertainty_metric=str(payload.get("uncertainty_metric", "Microstrain uncertainty")).strip(),
            bank_scope=str(payload.get("bank_scope", "Both banks independently")).strip(),
            exposure_unit=str(payload.get("exposure_unit", "uAmp")).strip(),
            observations=observations,
            mode=str(payload.get("mode", "single_path")),
            mu_per_mm=(
                None if payload.get("mu_per_mm") is None else float(payload["mu_per_mm"])
            ),
            log_scale=(
                None if payload.get("log_scale") is None else float(payload["log_scale"])
            ),
            rms_log_residual=(
                None
                if payload.get("rms_log_residual") is None
                else float(payload["rms_log_residual"])
            ),
        )


def fit_material_acquisition_profile(
    name: str,
    observations: Sequence[AcquisitionObservation],
    *,
    uncertainty_metric: str = "Microstrain uncertainty",
    bank_scope: str = "Both banks independently",
    exposure_unit: str = "uAmp",
    supplied_mu_per_mm: Optional[float] = None,
) -> MaterialAcquisitionProfile:
    material_name = str(name).strip()
    if not material_name:
        raise ValueError("Material name is required.")
    values = tuple(observations)
    if not values:
        raise ValueError("Add at least one calibration measurement.")
    if supplied_mu_per_mm is not None:
        supplied_mu_per_mm = float(supplied_mu_per_mm)
        if not math.isfinite(supplied_mu_per_mm) or supplied_mu_per_mm < 0.0:
            raise ValueError("Supplied effective mu must be finite and non-negative.")

    x_values = np.asarray([item.path_length_mm for item in values], dtype=float)
    y_values = np.asarray([item.normalized_log_exposure() for item in values], dtype=float)
    distinct_paths = np.unique(np.round(x_values, decimals=9))

    if distinct_paths.size >= 2:
        design = np.column_stack((np.ones(x_values.size, dtype=float), x_values))
        coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(design, y_values, rcond=None)
        log_scale = float(coefficients[0])
        mu_per_mm = float(coefficients[1])
        fitted_values = log_scale + mu_per_mm * x_values
        rms_log_residual = float(np.sqrt(np.mean((y_values - fitted_values) ** 2)))
        mode = "two_point_fit" if len(values) == 2 else "multi_point_fit"
    elif supplied_mu_per_mm is not None:
        mu_per_mm = supplied_mu_per_mm
        log_scale = float(np.mean(y_values - mu_per_mm * x_values))
        fitted_values = log_scale + mu_per_mm * x_values
        rms_log_residual = float(np.sqrt(np.mean((y_values - fitted_values) ** 2)))
        mode = "supplied_mu"
    else:
        mu_per_mm = None
        log_scale = None
        rms_log_residual = None
        mode = "single_path"

    return MaterialAcquisitionProfile(
        name=material_name,
        uncertainty_metric=str(uncertainty_metric).strip() or "Microstrain uncertainty",
        bank_scope=str(bank_scope).strip() or "Both banks independently",
        exposure_unit=str(exposure_unit).strip() or "uAmp",
        observations=values,
        mode=mode,
        mu_per_mm=mu_per_mm,
        log_scale=log_scale,
        rms_log_residual=rms_log_residual,
    )


def estimate_missing_acquisition_value(
    profile: MaterialAcquisitionProfile,
    *,
    path_length_mm: Optional[float],
    gauge_volume_mm3: Optional[float],
    exposure_uamp: Optional[float],
    uncertainty: Optional[float],
) -> tuple[str, float]:
    values = {
        "path_length_mm": path_length_mm,
        "gauge_volume_mm3": gauge_volume_mm3,
        "exposure_uamp": exposure_uamp,
        "uncertainty": uncertainty,
    }
    missing = [name for name, value in values.items() if value is None]
    if len(missing) != 1:
        raise ValueError("Exactly one of path length, gauge volume, exposure, or uncertainty must be blank.")

    def positive(name: str, value: Optional[float]) -> float:
        if value is None:
            raise ValueError(f"{name} is required.")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be positive.")
        return number

    missing_name = missing[0]
    if missing_name == "exposure_uamp":
        return (
            missing_name,
            profile.required_uamp(
                positive("Path length", path_length_mm),
                positive("Gauge volume", gauge_volume_mm3),
                positive("Uncertainty", uncertainty),
            ),
        )

    if missing_name == "path_length_mm":
        if profile.mu_per_mm is None or profile.log_scale is None or abs(float(profile.mu_per_mm)) < 1e-12:
            raise ValueError("Estimating path length requires a fitted or supplied non-zero effective mu.")
        target_log = math.log(
            positive("Exposure", exposure_uamp)
            * positive("Gauge volume", gauge_volume_mm3)
            * positive("Uncertainty", uncertainty) ** 2
        )
        path_length = (target_log - float(profile.log_scale)) / float(profile.mu_per_mm)
        if path_length < 0.0:
            raise ValueError("Estimated path length is negative; check the input values.")
        return missing_name, float(path_length)

    log_value = profile.normalized_log_exposure_for_path(positive("Path length", path_length_mm))
    normalized_value = math.exp(log_value)
    if missing_name == "gauge_volume_mm3":
        return (
            missing_name,
            normalized_value
            / (positive("Exposure", exposure_uamp) * positive("Uncertainty", uncertainty) ** 2),
        )
    if missing_name == "uncertainty":
        return (
            missing_name,
            math.sqrt(
                normalized_value
                / (positive("Exposure", exposure_uamp) * positive("Gauge volume", gauge_volume_mm3))
            ),
        )
    raise ValueError(f"Unsupported acquisition estimate field: {missing_name}")


def load_material_acquisition_profiles(path: Path) -> dict[str, MaterialAcquisitionProfile]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile_payloads = payload.get("profiles", []) if isinstance(payload, dict) else payload
    profiles = [MaterialAcquisitionProfile.from_dict(item) for item in profile_payloads]
    return {profile.name: profile for profile in profiles}


def save_material_acquisition_profiles(
    path: Path,
    profiles: Sequence[MaterialAcquisitionProfile],
) -> None:
    payload = {
        "version": 1,
        "profiles": [profile.to_dict() for profile in sorted(profiles, key=lambda item: item.name.casefold())],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
