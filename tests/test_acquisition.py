from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from diffraction.acquisition import (
    AcquisitionObservation,
    MaterialAcquisitionProfile,
    estimate_missing_acquisition_value,
    fit_material_acquisition_profile,
    load_material_acquisition_profiles,
    save_material_acquisition_profiles,
)


class MaterialAcquisitionCalibrationTests(unittest.TestCase):
    def test_single_path_scales_uncertainty_and_volume(self) -> None:
        profile = fit_material_acquisition_profile(
            "Alloy A",
            [AcquisitionObservation("run 1", 40.0, 64.0, 10.0, 200.0)],
        )
        self.assertEqual(profile.mode, "single_path")
        self.assertAlmostEqual(profile.required_uamp(40.0, 32.0, 100.0), 80.0)
        with self.assertRaisesRegex(ValueError, "single-path calibration"):
            profile.required_uamp(50.0, 32.0, 100.0)

    def test_two_path_calibration_recovers_mu_and_predicts(self) -> None:
        mu = 0.2
        observations = [
            AcquisitionObservation("run 1", 40.0, 64.0, 10.0, 200.0),
            AcquisitionObservation("run 2", 50.0, 64.0, 10.0 * math.exp(10.0 * mu), 200.0),
        ]
        profile = fit_material_acquisition_profile("Alloy B", observations)
        self.assertEqual(profile.mode, "two_point_fit")
        self.assertAlmostEqual(float(profile.mu_per_mm), mu, places=12)
        expected = 80.0 * math.exp(10.0 * mu)
        self.assertAlmostEqual(profile.required_uamp(50.0, 32.0, 100.0), expected)

    def test_three_path_profile_reports_residual_and_round_trips(self) -> None:
        observations = [
            AcquisitionObservation("a", 20.0, 64.0, 4.0, 150.0),
            AcquisitionObservation("b", 40.0, 64.0, 20.0, 150.0),
            AcquisitionObservation("c", 50.0, 32.0, 70.0, 150.0),
        ]
        profile = fit_material_acquisition_profile("Alloy C", observations)
        self.assertEqual(profile.mode, "multi_point_fit")
        self.assertIsNotNone(profile.rms_log_residual)
        restored = MaterialAcquisitionProfile.from_dict(profile.to_dict())
        self.assertAlmostEqual(
            restored.required_uamp(35.0, 48.0, 100.0),
            profile.required_uamp(35.0, 48.0, 100.0),
        )

    def test_single_path_with_supplied_mu_allows_path_prediction(self) -> None:
        profile = fit_material_acquisition_profile(
            "Alloy D",
            [AcquisitionObservation("run 1", 40.0, 64.0, 10.0, 200.0)],
            supplied_mu_per_mm=0.1,
        )
        self.assertEqual(profile.mode, "supplied_mu")
        self.assertAlmostEqual(
            profile.required_uamp(50.0, 64.0, 200.0),
            10.0 * math.exp(1.0),
        )

    def test_profile_library_round_trip(self) -> None:
        profile = fit_material_acquisition_profile(
            "Stored alloy",
            [AcquisitionObservation("run 1", 40.0, 64.0, 10.0, 200.0)],
            supplied_mu_per_mm=0.08,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            save_material_acquisition_profiles(path, [profile])
            loaded = load_material_acquisition_profiles(path)
        self.assertIn(profile.name, loaded)
        self.assertAlmostEqual(float(loaded[profile.name].mu_per_mm), 0.08)

    def test_estimates_each_missing_acquisition_value(self) -> None:
        profile = fit_material_acquisition_profile(
            "Estimator",
            [
                AcquisitionObservation("a", 40.0, 64.0, 10.0, 200.0),
                AcquisitionObservation("b", 50.0, 64.0, 10.0 * math.exp(1.0), 200.0),
            ],
        )
        self.assertAlmostEqual(
            estimate_missing_acquisition_value(
                profile,
                path_length_mm=50.0,
                gauge_volume_mm3=32.0,
                exposure_uamp=None,
                uncertainty=100.0,
            )[1],
            80.0 * math.exp(1.0),
        )
        self.assertAlmostEqual(
            estimate_missing_acquisition_value(
                profile,
                path_length_mm=50.0,
                gauge_volume_mm3=None,
                exposure_uamp=80.0 * math.exp(1.0),
                uncertainty=100.0,
            )[1],
            32.0,
        )
        self.assertAlmostEqual(
            estimate_missing_acquisition_value(
                profile,
                path_length_mm=50.0,
                gauge_volume_mm3=32.0,
                exposure_uamp=80.0 * math.exp(1.0),
                uncertainty=None,
            )[1],
            100.0,
        )
        self.assertAlmostEqual(
            estimate_missing_acquisition_value(
                profile,
                path_length_mm=None,
                gauge_volume_mm3=32.0,
                exposure_uamp=80.0 * math.exp(1.0),
                uncertainty=100.0,
            )[1],
            50.0,
        )

    def test_estimation_requires_exactly_one_missing_value(self) -> None:
        profile = fit_material_acquisition_profile(
            "Estimator validation",
            [AcquisitionObservation("a", 40.0, 64.0, 10.0, 200.0)],
        )
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            estimate_missing_acquisition_value(
                profile,
                path_length_mm=40.0,
                gauge_volume_mm3=64.0,
                exposure_uamp=10.0,
                uncertainty=200.0,
            )


if __name__ == "__main__":
    unittest.main()
