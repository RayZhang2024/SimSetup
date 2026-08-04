from __future__ import annotations

from typing import Mapping, Optional

from .models import CalibrationResult


PROFILE_METADATA_KEY = "pattern_calibration_profile_parameters"
PROFILE_KEY_METADATA_KEY = "pattern_calibration_profile_key"


def calibration_profile_metadata(calibration_result: CalibrationResult) -> dict[str, object]:
    fit = calibration_result.pattern_fit
    if fit is None:
        return {
            "pattern_calibration_profile": "",
            PROFILE_KEY_METADATA_KEY: "",
            PROFILE_METADATA_KEY: {},
        }
    parameters = dict(fit.profile_parameters)
    parameters["fwhm"] = float(fit.fwhm)
    parameters["eta"] = float(fit.eta)
    return {
        "pattern_calibration_profile": fit.profile_name,
        PROFILE_KEY_METADATA_KEY: fit.profile_key,
        PROFILE_METADATA_KEY: parameters,
    }


def initial_profile_parameters_from_metadata(
    metadata: Mapping[str, object],
    profile_key: str,
) -> Optional[dict[str, float]]:
    if str(metadata.get(PROFILE_KEY_METADATA_KEY, "")) != str(profile_key):
        return None
    raw_parameters = metadata.get(PROFILE_METADATA_KEY)
    if not isinstance(raw_parameters, Mapping):
        return None
    parameters: dict[str, float] = {}
    for key, value in raw_parameters.items():
        try:
            parameters[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parameters or None
