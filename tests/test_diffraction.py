from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from diffraction.calibration import OPENGENIE_CALIBRATION_TOF_MAX, OPENGENIE_CALIBRATION_TOF_MIN, refine_ceo2_calibration
from diffraction.batch_fitting import SpectrumFitRequest, fit_spectrum, position_to_d_spacing
from diffraction.convergence import gsas_relative_chi_square_ftol
from diffraction.fitting import FitAbortedError, fit_peak_profile, fit_pseudo_voigt, pseudo_voigt_profile
from diffraction.gsas_exp import generate_fcc_reflections, generate_phase_reflections, parse_gsas_exp
from diffraction.importers import load_focused_spectrum
from diffraction.models import (
    AXIS_D_SPACING,
    AXIS_TOF,
    FIT_SCOPE_PATTERN,
    FIT_SCOPE_PEAK,
    InstrumentCalibration,
    PAWLEY_WIDTH_D_RESOLUTION,
    FittingSettings,
    FocusedSpectrum,
    PhaseModel,
    d_to_tof,
    tof_to_d,
)
from diffraction.normalization import apply_vanadium_normalization
from diffraction.pawley import fit_pawley
from diffraction.profile_metadata import (
    calibration_profile_metadata,
    initial_profile_parameters_from_metadata,
)
from diffraction.profiles import (
    GSAS_TOF_PROFILE_KEYS,
    PROFILE_EXP_VOIGT,
    PROFILE_GSAS2_TOF_PV,
    PROFILE_GSAS_TOF,
    PROFILE_PSEUDO_VOIGT,
    PROFILE_VOIGT,
    back_to_back_tail_bounds,
    evaluate_peak_profile,
)


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "Hist"


def scipy_available() -> bool:
    try:
        import scipy  # noqa: F401
    except Exception:
        return False
    return True


def corrected_ni_bank_1():
    ceo2 = load_focused_spectrum(HIST / "ENGINX373845_1.his")
    ceo2_phase = parse_gsas_exp(HIST / "CEO2.EXP")[0]
    calibration = refine_ceo2_calibration(ceo2, ceo2_phase)
    sample = load_focused_spectrum(HIST / "ENGINX373922_1.his")
    vanadium = load_focused_spectrum(HIST / "ENGINX371347_1.his")
    return apply_vanadium_normalization(sample, vanadium, calibration)


def first_peak_header(headers: list[str], field: str) -> str:
    suffix = f" {field}"
    for header in headers:
        if header.startswith("(") and header.endswith(suffix):
            return header
    return f"Peak 1 {field}"


class DiffractionImportTests(unittest.TestCase):
    def test_his_metadata_import(self) -> None:
        spectrum = load_focused_spectrum(HIST / "ENGINX371347_1.his")
        self.assertEqual(spectrum.run_number, "371347")
        self.assertEqual(spectrum.bank_number, 1)
        self.assertEqual(spectrum.bank_name, "North_bank")
        self.assertEqual(spectrum.x.size, spectrum.y.size + 1)
        self.assertIsNotNone(spectrum.e)
        self.assertIsNotNone(spectrum.calibration)
        self.assertGreater(float(spectrum.calibration.difc), 1000.0)

    def test_tof_d_spacing_round_trip(self) -> None:
        spectrum = load_focused_spectrum(HIST / "ENGINX371347_1.his")
        calibration = spectrum.calibration
        self.assertIsNotNone(calibration)
        tof = spectrum.axis_values(AXIS_TOF)[100:110]
        d_spacing = tof_to_d(tof, calibration)
        round_trip = d_to_tof(d_spacing, calibration)
        np.testing.assert_allclose(round_trip, tof, rtol=1e-10, atol=1e-8)


class GsasExpTests(unittest.TestCase):
    def test_ni_exp_phase(self) -> None:
        phase = parse_gsas_exp(HIST / "NI.EXP")[0]
        self.assertEqual(phase.name, "Nickel")
        self.assertAlmostEqual(phase.a, 3.5226, places=6)
        self.assertAlmostEqual(phase.alpha, 90.0, places=6)
        self.assertEqual(" ".join(phase.space_group.split()), "F m 3 m")
        self.assertTrue(any(atom.element.upper().startswith("NI") for atom in phase.atoms))

    def test_ceo2_exp_phase(self) -> None:
        phase = parse_gsas_exp(HIST / "CEO2.EXP")[0]
        self.assertEqual(phase.name, "Cerium Oxide")
        self.assertAlmostEqual(phase.a, 5.411406, places=6)
        self.assertAlmostEqual(phase.alpha, 90.0, places=6)
        self.assertEqual(" ".join(phase.space_group.split()), "F m 3 m")
        elements = {atom.element.upper() for atom in phase.atoms}
        self.assertIn("CE", elements)
        self.assertIn("O", elements)

    def test_fcc_reflections(self) -> None:
        phase = parse_gsas_exp(HIST / "NI.EXP")[0]
        reflections = generate_fcc_reflections(phase, 1.0, 2.1)
        labels = {reflection.label for reflection in reflections}
        self.assertIn("111", labels)
        self.assertIn("200", labels)
        self.assertIn("220", labels)
        self.assertNotIn("100", labels)
        self.assertNotIn("110", labels)


@unittest.skipUnless(scipy_available(), "scipy is required for fitting tests")
class DiffractionFittingTests(unittest.TestCase):
    def test_fitting_controls_labels_and_bank_specific_range_conversion(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication
        from diffraction.tab import DiffractionTab

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        try:
            for bank in (1, 2):
                tab.load_spectrum_paths([HIST / f"ENGINX373922_{bank}.his"])
                index = bank - 1
                self.assertEqual(tab.spectrum_combo.itemText(index), f"373922_{bank}")
                self.assertIn(f"ENGINX373922_{bank}.his", tab.spectrum_combo.itemData(index, Qt.ToolTipRole))
            self.assertEqual(tab.fit_button.text(), "Fit single")
            self.assertFalse(hasattr(tab, "batch_scope_combo"))
            self.assertFalse(hasattr(tab, "data_combo"))
            self.assertFalse(hasattr(tab, "calibration_button"))
            self.assertFalse(hasattr(tab, "calibration_status_edit"))
            view_menu = next(action.menu() for action in tab.menu_bar.actions() if action.text() == "View")
            self.assertNotIn("Show Raw Data", [action.text() for action in view_menu.actions()])
            for index in (0, 1):
                tab.spectrum_combo.setCurrentIndex(index)
                tab.range_min_edit.setText("20000")
                tab.range_max_edit.setText("30000")
                spectrum = tab.current_spectrum()
                expected = tof_to_d(np.array([20000.0, 30000.0]), spectrum.calibration)
                tab.axis_combo.setCurrentIndex(1)
                self.assertAlmostEqual(float(tab.range_min_edit.text()), float(min(expected)), places=8)
                self.assertAlmostEqual(float(tab.range_max_edit.text()), float(max(expected)), places=8)
                tab.axis_combo.setCurrentIndex(0)
                self.assertAlmostEqual(float(tab.range_min_edit.text()), 20000.0, places=4)
                self.assertAlmostEqual(float(tab.range_max_edit.text()), 30000.0, places=4)
            tab.range_min_edit.setText("30000")
            tab.range_max_edit.setText("20000")
            tab.axis_combo.setCurrentIndex(1)
            self.assertLess(float(tab.range_min_edit.text()), float(tab.range_max_edit.text()))
            tab.axis_combo.setCurrentIndex(0)
            uncalibrated = FocusedSpectrum(
                source_path=Path("uncalibrated.xye"), x=np.arange(20.0), y=np.ones(20),
                run_number="9", bank_number=1,
            )
            tab.spectra.append(uncalibrated)
            tab.corrected_spectra.append(None)
            tab._add_spectrum_item(uncalibrated)
            tab.spectrum_combo.setCurrentIndex(2)
            tab.range_min_edit.setText("2")
            tab.range_max_edit.setText("5")
            tab.axis_combo.setCurrentIndex(1)
            self.assertEqual(tab.range_min_edit.text(), "")
            self.assertEqual(tab.range_max_edit.text(), "")
            self.assertIn("calibration", tab.status_label.text().lower())
        finally:
            tab.close()
            app.processEvents()

    def test_current_fit_request_uses_displayed_correction_and_rejects_raw_nxs_recorrection(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        from diffraction.tab import DiffractionTab

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        try:
            raw = load_focused_spectrum(HIST / "ENGINX373922_1.his")
            corrected = corrected_ni_bank_1().corrected_spectrum
            tab.spectra.append(raw)
            tab.corrected_spectra.append(corrected)
            tab._add_spectrum_item(raw)
            self.assertIs(tab.current_spectrum(), corrected)
            request = tab._fit_request(0, (1.0, 2.0))
            self.assertIs(request.spectrum, corrected)
            self.assertEqual(request.data_mode, "corrected")
            raw_nxs = FocusedSpectrum(
                source_path=Path("sample.nxs"), x=np.arange(20.0), y=np.ones(20),
                metadata={"raw_nxs_processed": True}, run_number="377898", bank_number=2,
            )
            tab.add_processed_spectrum(raw_nxs, update=False)
            self.assertEqual(tab.spectrum_combo.currentText(), "377898_2")
            self.assertIs(tab.current_spectrum(), raw_nxs)
            self.assertEqual(tab._fit_request(1, (1.0, 2.0)).data_mode, "raw")
            with self.assertRaisesRegex(ValueError, "already open-beam normalized"):
                tab._apply_normalisation_to_index(1)
        finally:
            tab.close()
            app.processEvents()

    def test_synthetic_pseudo_voigt(self) -> None:
        x = np.linspace(10.0, 20.0, 300)
        y = 4.0 + 120.0 * pseudo_voigt_profile(x, 14.25, 0.7, 0.35)
        result = fit_pseudo_voigt(x, y, polynomial_order=1)
        self.assertAlmostEqual(result.centre, 14.25, delta=0.03)
        self.assertAlmostEqual(result.fwhm, 0.7, delta=0.05)

    def test_synthetic_open_genie_voigt(self) -> None:
        x = np.linspace(10.0, 20.0, 300)
        y = 2.5 + 90.0 * evaluate_peak_profile(PROFILE_VOIGT, x, 15.2, 0.85, eta=0.4)
        result = fit_peak_profile(PROFILE_VOIGT, x, y, polynomial_order=1)
        self.assertEqual(result.model_name, "Open GENIE Voigt")
        self.assertAlmostEqual(result.centre, 15.2, delta=0.03)
        self.assertGreater(result.fwhm, 0.0)
        self.assertIn("eta", result.profile_parameters)

    def test_exponential_and_gsas_tof_profiles_are_finite(self) -> None:
        x = np.linspace(10000.0, 18000.0, 450)
        exp_values = evaluate_peak_profile(PROFILE_EXP_VOIGT, x, 13500.0, 180.0, eta=0.35, tail=260.0)
        gsas_values = evaluate_peak_profile(PROFILE_GSAS_TOF, x, 13500.0, 180.0, eta=0.35, alpha=120.0, beta=360.0)
        self.assertTrue(np.all(np.isfinite(exp_values)))
        self.assertTrue(np.all(np.isfinite(gsas_values)))
        self.assertGreater(float(np.max(exp_values)), 0.9)
        self.assertGreater(float(np.max(gsas_values)), 0.9)
        low_side = gsas_values[x < 13500.0]
        high_side = gsas_values[x > 13500.0]
        self.assertGreater(abs(float(np.sum(high_side)) - float(np.sum(low_side))), 0.1)
        for profile_key in GSAS_TOF_PROFILE_KEYS:
            values = evaluate_peak_profile(profile_key, x, 13500.0, 180.0, eta=0.35, tail=260.0, alpha=120.0, beta=360.0)
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertGreater(float(np.max(values)), 0.9)

    def test_back_to_back_tail_bounds_are_width_scaled(self) -> None:
        guess, lower, upper = back_to_back_tail_bounds(
            axis_span=20000.0,
            fwhm_guess=250.0,
            fwhm_min=2.0,
            fwhm_max=1000.0,
        )
        self.assertGreaterEqual(guess, lower)
        self.assertLessEqual(guess, upper)
        self.assertLess(upper, 20000.0)
        self.assertLessEqual(upper, 5000.0)

    def test_back_to_back_pv_fit_reports_bounded_shape_parameters(self) -> None:
        x = np.linspace(10000.0, 18000.0, 450)
        y = 2.0 + 130.0 * evaluate_peak_profile(
            PROFILE_GSAS_TOF, x, 13500.0, 180.0, eta=0.35, alpha=120.0, beta=360.0
        )
        result = fit_peak_profile(
            PROFILE_GSAS_TOF,
            x,
            y,
            None,
            polynomial_order=0,
            max_peaks=1,
            maxfev=2000,
        )
        span = float(np.max(x) - np.min(x))
        _guess, lower, upper = back_to_back_tail_bounds(
            span,
            result.fwhm,
            span * 0.0001,
            span * 2.0,
        )
        self.assertGreaterEqual(result.profile_parameters["alpha"], lower)
        self.assertLessEqual(result.profile_parameters["alpha"], upper)
        self.assertGreaterEqual(result.profile_parameters["beta"], lower)
        self.assertLessEqual(result.profile_parameters["beta"], upper)

    def test_gsas2_tof_double_exponential_pv_fit_uses_sig_gam_alp_bet(self) -> None:
        x = np.linspace(10000.0, 18000.0, 500)
        y = 2.0 + 140.0 * evaluate_peak_profile(
            PROFILE_GSAS2_TOF_PV,
            x,
            13500.0,
            120.0,
            sig=1600.0,
            gam=90.0,
            alp=1.0 / 180.0,
            bet=1.0 / 420.0,
        )
        result = fit_peak_profile(
            PROFILE_GSAS2_TOF_PV,
            x,
            y,
            None,
            polynomial_order=0,
            max_peaks=1,
            maxfev=3000,
        )
        self.assertEqual(result.model_name, "GSAS-II TOF double-exponential PV")
        self.assertAlmostEqual(result.centre, 13500.0, delta=80.0)
        self.assertGreater(result.fwhm, 0.0)
        for name in ("sig", "gam", "alp", "bet"):
            self.assertIn(name, result.profile_parameters)
            self.assertGreater(result.profile_parameters[name], 0.0)

    def test_pseudo_voigt_handles_extreme_scaled_values(self) -> None:
        x = np.array([-1.0e12, 0.0, 1.0e12])
        with np.errstate(over="raise", invalid="raise"):
            values = pseudo_voigt_profile(x, 0.0, 1.0e-12, 0.5)
        self.assertTrue(np.all(np.isfinite(values)))

    def test_ceo2_calibration(self) -> None:
        ceo2 = load_focused_spectrum(HIST / "ENGINX373845_1.his")
        phase = parse_gsas_exp(HIST / "CEO2.EXP")[0]
        result = refine_ceo2_calibration(ceo2, phase)
        self.assertEqual(result.run_number, "373845")
        self.assertEqual(result.bank_number, 1)
        self.assertGreaterEqual(len(result.accepted_peaks), 8)
        self.assertTrue(18000.0 < result.calibration.difc < 18700.0)
        self.assertTrue(-200.0 < result.calibration.tzero < 200.0)
        self.assertNotEqual(result.calibration.difa, 0.0)
        self.assertTrue(18000.0 < result.single_peak_calibration.difc < 18700.0)
        self.assertEqual(result.single_peak_calibration.difa, 0.0)
        self.assertIsNotNone(result.pattern_fit)
        self.assertEqual(result.pattern_profile_parameters.get("gsas_tof_function"), 3.0)
        for peak in result.accepted_peaks:
            self.assertGreaterEqual(peak.expected_tof, OPENGENIE_CALIBRATION_TOF_MIN)
            self.assertLessEqual(peak.expected_tof, OPENGENIE_CALIBRATION_TOF_MAX)
        self.assertLess(result.rms_residual_tof, 25.0)
        self.assertLess(result.single_peak_rms_residual_tof, 25.0)

    def test_vanadium_normalisation(self) -> None:
        result = corrected_ni_bank_1()
        corrected = result.corrected_spectrum
        self.assertEqual(result.sample_run_number, "373922")
        self.assertEqual(result.vanadium_run_number, "371347")
        self.assertEqual(result.bank_number, 1)
        self.assertEqual(result.invalid_bins, 0)
        self.assertEqual(int(np.count_nonzero(np.isfinite(corrected.y))), corrected.y.size)
        self.assertIsNotNone(corrected.calibration)
        self.assertEqual(corrected.metadata.get("calibration_run"), "373845")
        self.assertEqual(corrected.metadata.get("normalization_run"), "371347")

    def test_ni_measurement_pseudo_voigt(self) -> None:
        spectrum = corrected_ni_bank_1().corrected_spectrum
        calibration = spectrum.calibration
        self.assertIsNotNone(calibration)
        centre = float(d_to_tof(3.5226 / math.sqrt(3.0), calibration))
        x = spectrum.axis_values(AXIS_TOF)
        mask = (x > centre - 900.0) & (x < centre + 900.0)
        result = fit_pseudo_voigt(x[mask], spectrum.y[mask], spectrum.e[mask] if spectrum.e is not None else None)
        self.assertGreater(result.height, 0.0)
        self.assertGreater(result.fwhm, 0.0)
        self.assertTrue(centre - 900.0 < result.centre < centre + 900.0)

    def test_wide_range_profile_fit_tracks_multiple_peaks(self) -> None:
        spectrum = corrected_ni_bank_1().corrected_spectrum
        x = spectrum.axis_values(AXIS_D_SPACING)
        mask = (x >= 1.0) & (x <= 2.2)
        result = fit_peak_profile(
            PROFILE_VOIGT,
            x[mask],
            spectrum.y[mask],
            spectrum.e[mask] if spectrum.e is not None else None,
            polynomial_order=2,
        )
        self.assertGreaterEqual(len(result.components), 4)
        centres = [component.centre for component in result.components]
        for expected in (1.08, 1.27, 1.80, 2.07):
            self.assertTrue(any(abs(centre - expected) < 0.03 for centre in centres))

    def test_peak_fit_honours_abort_check(self) -> None:
        x = np.linspace(1.0, 2.0, 100)
        y = 2.0 + 100.0 * pseudo_voigt_profile(x, 1.5, 0.05, 0.4)
        with self.assertRaisesRegex(FitAbortedError, "timed out"):
            fit_peak_profile(PROFILE_VOIGT, x, y, abort_check=lambda: "Fit timed out.")

    def test_gsas_min_dm_over_m_maps_to_scipy_ftol(self) -> None:
        self.assertAlmostEqual(gsas_relative_chi_square_ftol(0.0001), 0.0001)
        self.assertIsNone(gsas_relative_chi_square_ftol(0.0))
        self.assertIsNone(gsas_relative_chi_square_ftol(-1.0))

    def test_ni_measurement_pawley(self) -> None:
        spectrum = corrected_ni_bank_1().corrected_spectrum
        phase = parse_gsas_exp(HIST / "NI.EXP")[0]
        x = spectrum.axis_values(AXIS_D_SPACING)
        mask = (x > 1.65) & (x < 2.12)
        result = fit_pawley(
            x[mask],
            spectrum.y[mask],
            spectrum.e[mask] if spectrum.e is not None else None,
            phase,
            AXIS_D_SPACING,
            spectrum.calibration,
        )
        self.assertGreater(result.fwhm, 0.0)
        self.assertIsNotNone(result.lattice_a_uncertainty)
        self.assertGreater(float(result.lattice_a_uncertainty), 0.0)
        self.assertGreaterEqual(len(result.reflections), 2)
        self.assertTrue(3.45 < result.lattice_a < 3.60)
        self.assertIsNotNone(result.fit_uncertainty)
        mean_residual = float(np.mean(result.observed_y - result.fit_y))
        self.assertLess(abs(mean_residual), 0.03 * float(np.median(result.observed_y)))

    def test_non_ni_exp_phase_can_be_pawley_fitted(self) -> None:
        phase = parse_gsas_exp(HIST / "CEO2.EXP")[0]
        reflections = generate_phase_reflections(phase, 1.5, 3.2)
        self.assertGreaterEqual(len(reflections), 3)
        x = np.linspace(1.5, 3.2, 1000)
        y = np.full_like(x, 4.0)
        for index, reflection in enumerate(reflections):
            y += (120.0 + 20.0 * index) * evaluate_peak_profile(
                PROFILE_VOIGT, x, reflection.d_spacing, 0.012, eta=0.4
            )
        result = fit_pawley(
            x,
            y,
            None,
            phase,
            AXIS_D_SPACING,
            None,
            polynomial_order=0,
            lattice_tolerance_percent=1.0,
            reflection_margin_percent=0.0,
            profile_key=PROFILE_VOIGT,
            max_nfev=5000,
        )
        self.assertAlmostEqual(result.lattice_a, phase.a, delta=0.005)
        self.assertAlmostEqual(result.lattice_parameters["b"], phase.b, delta=0.005)
        self.assertNotIn("NI", phase.name.upper())

    def test_calibrated_back_to_back_tail_parameters_are_locally_refined(self) -> None:
        phase = parse_gsas_exp(HIST / "NI.EXP")[0]
        calibration = InstrumentCalibration(difc=18358.9, tzero=-31.35, difa=-16.88)
        x = np.linspace(18000.0, 40000.0, 1800)
        y = np.full_like(x, 0.25)
        for reflection in generate_phase_reflections(phase, 0.9, 2.2):
            position = float(d_to_tof(reflection.d_spacing, calibration))
            if 18000.0 <= position <= 40000.0:
                y += 1.8 * evaluate_peak_profile(
                    PROFILE_GSAS_TOF,
                    x,
                    position,
                    80.0,
                    eta=0.2,
                    alpha=20.0,
                    beta=30.0,
                )

        result = fit_pawley(
            x,
            y,
            None,
            phase,
            AXIS_TOF,
            calibration,
            polynomial_order=1,
            lattice_tolerance_percent=2.0,
            reflection_margin_percent=4.0,
            profile_key=PROFILE_GSAS_TOF,
            width_model=PAWLEY_WIDTH_D_RESOLUTION,
            max_nfev=1000,
            initial_profile_parameters={"alpha": 20.0, "beta": 30.0, "eta": 0.2},
        )

        self.assertGreaterEqual(result.profile_parameters["alpha"], 5.0)
        self.assertLessEqual(result.profile_parameters["alpha"], 80.0)
        self.assertGreaterEqual(result.profile_parameters["beta"], 7.5)
        self.assertLessEqual(result.profile_parameters["beta"], 120.0)

    def test_orthorhombic_phase_refines_independent_cell_lengths(self) -> None:
        phase = PhaseModel(1, "Orthorhombic test", 3.0, 4.0, 5.0, 90.0, 90.0, 90.0, "P m m m")
        reflections = generate_phase_reflections(phase, 1.8, 2.6)
        x = np.linspace(1.8, 2.6, 900)
        y = np.full_like(x, 3.0)
        for index, reflection in enumerate(reflections):
            y += (80.0 + 10.0 * index) * evaluate_peak_profile(
                PROFILE_VOIGT, x, reflection.d_spacing, 0.008, eta=0.35
            )
        result = fit_pawley(
            x, y, None, phase, AXIS_D_SPACING, None,
            polynomial_order=0, lattice_tolerance_percent=1.0,
            reflection_margin_percent=0.0, profile_key=PROFILE_VOIGT,
            max_nfev=5000,
        )
        self.assertAlmostEqual(result.lattice_parameters["a"], 3.0, delta=0.005)
        self.assertAlmostEqual(result.lattice_parameters["b"], 4.0, delta=0.005)
        self.assertAlmostEqual(result.lattice_parameters["c"], 5.0, delta=0.005)

    def test_batch_pawley_uses_per_spectrum_lattice_seed(self) -> None:
        phase = PhaseModel(1, "Cubic seed test", 3.593052, 3.593052, 3.593052, 90.0, 90.0, 90.0, "F m -3 m")
        settings = FittingSettings(
            fit_scope=FIT_SCOPE_PATTERN,
            pattern_profile_key=PROFILE_PSEUDO_VOIGT,
            polynomial_order=0,
            use_uncertainties=False,
            max_evaluations=1000,
            pawley_lattice_tolerance_percent=3.0,
            pawley_reflection_margin_percent=0.0,
        )
        fitted_values = []
        for index, true_a in enumerate((3.58, 3.62)):
            true_phase = PhaseModel(1, "Synthetic cubic", true_a, true_a, true_a, 90.0, 90.0, 90.0, "F m -3 m")
            reflections = generate_phase_reflections(true_phase, 1.98, 2.12)
            x = np.linspace(1.98, 2.12, 500)
            y = np.full_like(x, 2.0)
            for reflection in reflections:
                y += 100.0 * evaluate_peak_profile(
                    PROFILE_PSEUDO_VOIGT, x, reflection.d_spacing, 0.006, eta=0.4
                )
            spectrum = FocusedSpectrum(
                source_path=Path(f"synthetic_{index}.xye"),
                x=x,
                y=y,
                e=None,
                native_axis=AXIS_D_SPACING,
                calibration=None,
                run_number=str(index),
                bank_number=1,
            )
            record = fit_spectrum(
                SpectrumFitRequest(
                    spectrum_index=index,
                    spectrum=spectrum,
                    settings=settings,
                    phase=phase,
                    display_axis=AXIS_D_SPACING,
                    data_mode="raw",
                    selected_range=(1.98, 2.12),
                )
            )
            fitted_values.append(record.result.lattice_parameters["a"])

        self.assertAlmostEqual(fitted_values[0], 3.58, delta=0.002)
        self.assertAlmostEqual(fitted_values[1], 3.62, delta=0.002)
        self.assertGreater(fitted_values[1] - fitted_values[0], 0.03)

    def test_individual_peak_fit_uses_expected_phase_reflection_positions(self) -> None:
        phase = PhaseModel(1, "Synthetic cubic", 4.0, 4.0, 4.0, 90.0, 90.0, 90.0, "P m -3 m")
        reflections = generate_phase_reflections(phase, 1.75, 2.9)
        expected_positions = [reflection.d_spacing for reflection in reflections]
        x = np.linspace(1.75, 2.9, 1200)
        y = np.full_like(x, 2.0)
        amplitudes = {}
        for reflection in reflections:
            amplitude = 400.0 if reflection.label == "200" else 60.0
            amplitudes[reflection.label] = amplitude
            y += amplitude * evaluate_peak_profile(
                PROFILE_VOIGT,
                x,
                reflection.d_spacing,
                0.006,
                eta=0.35,
            )
        settings = FittingSettings(
            fit_scope=FIT_SCOPE_PEAK,
            peak_profile_key=PROFILE_VOIGT,
            polynomial_order=0,
            use_uncertainties=False,
            max_evaluations=3000,
            pawley_reflection_margin_percent=0.0,
        )
        spectrum = FocusedSpectrum(
            source_path=Path("phase_guided_peaks.xye"),
            x=x,
            y=y,
            native_axis=AXIS_D_SPACING,
            calibration=None,
            run_number="1",
            bank_number=1,
        )
        record = fit_spectrum(
            SpectrumFitRequest(
                spectrum_index=0,
                spectrum=spectrum,
                settings=settings,
                phase=phase,
                display_axis=AXIS_D_SPACING,
                data_mode="raw",
                selected_range=(1.75, 2.9),
            ),
            use_gsas_convergence=True,
        )
        fitted_centres = [component.centre for component in record.result.components]
        self.assertGreaterEqual(len(fitted_centres), min(len(expected_positions), 4))
        for expected in expected_positions[:4]:
            self.assertTrue(any(abs(centre - expected) < 0.01 for centre in fitted_centres), expected)
        fitted_reflections = [component.reflection for component in record.result.components]
        self.assertTrue(all(reflection is not None for reflection in fitted_reflections))
        self.assertGreaterEqual(len({reflection.label for reflection in fitted_reflections if reflection is not None}), 2)

    def test_individual_peak_hkl_assignment_follows_expected_reflection_not_nearest_fitted_centre(self) -> None:
        from diffraction.models import Reflection
        from diffraction.fitting import _expected_peak_candidates

        ref_222 = Reflection(2, 2, 2, 1.06)
        ref_311 = Reflection(3, 1, 1, 1.12)
        x = np.linspace(1.0, 1.18, 900)
        y = 1.0 + 120.0 * evaluate_peak_profile(PROFILE_VOIGT, x, 1.06, 0.004, eta=0.3)
        y += 18.0 * evaluate_peak_profile(PROFILE_VOIGT, x, 1.12, 0.004, eta=0.3)
        shifted = y + 300.0 * evaluate_peak_profile(PROFILE_VOIGT, x, 1.058, 0.003, eta=0.3)
        candidates = _expected_peak_candidates(
            x,
            shifted,
            1.0,
            [ref_222.d_spacing, ref_311.d_spacing],
            [ref_222, ref_311],
            max_peaks=2,
        )
        self.assertEqual(candidates[0][0], ref_222.d_spacing)
        self.assertEqual(candidates[1][0], ref_311.d_spacing)

        result = fit_peak_profile(
            PROFILE_VOIGT,
            x,
            y,
            None,
            polynomial_order=0,
            max_peaks=2,
            expected_peak_positions=[ref_222.d_spacing, ref_311.d_spacing],
            expected_peak_reflections=[ref_222, ref_311],
        )
        by_label = {
            component.reflection.label: component.centre
            for component in result.components
            if component.reflection is not None
        }

        self.assertAlmostEqual(by_label["222"], 1.06, delta=0.01)
        self.assertAlmostEqual(by_label["311"], 1.12, delta=0.01)
        self.assertGreater(by_label["311"], 0.5 * (ref_222.d_spacing + ref_311.d_spacing))

    def test_phase_guided_peak_fit_keeps_expected_reflections_beyond_default_peak_limit(self) -> None:
        from diffraction.models import Reflection

        reflections = [Reflection(index + 1, 0, 0, 1.0 + 0.04 * index) for index in range(13)]
        x = np.linspace(0.95, 1.53, 1800)
        y = np.full_like(x, 1.0)
        for index, reflection in enumerate(reflections):
            height = 8.0 + index
            if index == 12:
                height = 14.0
            y += height * evaluate_peak_profile(PROFILE_VOIGT, x, reflection.d_spacing, 0.003, eta=0.3)

        result = fit_peak_profile(
            PROFILE_VOIGT,
            x,
            y,
            None,
            polynomial_order=0,
            max_peaks=12,
            expected_peak_positions=[reflection.d_spacing for reflection in reflections],
            expected_peak_reflections=reflections,
        )
        fitted_labels = {
            component.reflection.label
            for component in result.components
            if component.reflection is not None
        }

        self.assertIn(reflections[-1].label, fitted_labels)

    def test_phase_guided_individual_peak_centres_refine_between_tensile_runs(self) -> None:
        phase = parse_gsas_exp(HIST / "FE_GAMMA.EXP")[0]
        settings = FittingSettings(
            fit_scope=FIT_SCOPE_PEAK,
            peak_profile_key=PROFILE_EXP_VOIGT,
            use_uncertainties=True,
            max_evaluations=1000,
        )
        fitted_d_by_run = []
        for run in (375058, 375061):
            spectrum = load_focused_spectrum(HIST / "tensile" / f"ENGINX{run}_1.his")
            record = fit_spectrum(
                SpectrumFitRequest(
                    spectrum_index=0,
                    spectrum=spectrum,
                    settings=settings,
                    phase=phase,
                    display_axis=AXIS_TOF,
                    data_mode="raw",
                    selected_range=(18000.0, 32500.0),
                ),
                use_gsas_convergence=True,
            )
            self.assertGreaterEqual(len(record.result.components), 2)
            fitted_d_by_run.append({
                component.reflection.label: position_to_d_spacing(component.centre, record.fit_axis, spectrum)
                for component in record.result.components
                if component.reflection is not None
            })
        common_labels = set(fitted_d_by_run[0]).intersection(fitted_d_by_run[1])
        self.assertTrue(common_labels)
        self.assertGreater(
            max(abs(fitted_d_by_run[1][label] - fitted_d_by_run[0][label]) for label in common_labels),
            0.0005,
        )

    def test_phase_guided_individual_peak_fit_refines_independent_fwhm_values(self) -> None:
        x = np.linspace(1.0, 4.0, 1200)
        centres = (1.45, 2.35, 3.25)
        widths = (0.025, 0.07, 0.14)
        y = np.full_like(x, 2.0)
        for centre, width, height in zip(centres, widths, (80.0, 110.0, 95.0)):
            y += height * evaluate_peak_profile(PROFILE_VOIGT, x, centre, width, eta=0.35)

        result = fit_peak_profile(
            PROFILE_VOIGT,
            x,
            y,
            None,
            polynomial_order=0,
            max_peaks=3,
            maxfev=5000,
            expected_peak_positions=centres,
        )

        fitted_widths = [component.fwhm for component in result.components]
        self.assertEqual(len(fitted_widths), 3)
        self.assertIn("independent_peak_fwhm", result.profile_parameters)
        self.assertGreater(max(fitted_widths) - min(fitted_widths), 0.08)
        for fitted, expected in zip(fitted_widths, widths):
            self.assertAlmostEqual(fitted, expected, delta=0.03)

    def test_calibration_profile_metadata_seeds_matching_profile(self) -> None:
        fit = SimpleNamespace(
            profile_name="GSAS TOF profile 3: back-to-back PV",
            profile_key=PROFILE_GSAS_TOF,
            fwhm=123.0,
            eta=0.42,
            profile_parameters={"alpha": 210.0, "beta": 320.0, "gsas_tof_function": 3.0},
        )
        calibration_result = SimpleNamespace(pattern_fit=fit)
        metadata = calibration_profile_metadata(calibration_result)

        parameters = initial_profile_parameters_from_metadata(metadata, PROFILE_GSAS_TOF)
        self.assertIsNotNone(parameters)
        self.assertEqual(parameters["fwhm"], 123.0)
        self.assertEqual(parameters["eta"], 0.42)
        self.assertEqual(parameters["alpha"], 210.0)
        self.assertEqual(parameters["beta"], 320.0)
        self.assertIsNone(initial_profile_parameters_from_metadata(metadata, PROFILE_VOIGT))

    def test_fit_spectrum_uses_calibration_profile_seed_metadata(self) -> None:
        settings = FittingSettings(
            fit_scope=FIT_SCOPE_PATTERN,
            pattern_profile_key=PROFILE_GSAS_TOF,
            use_uncertainties=False,
        )
        spectrum = FocusedSpectrum(
            source_path=Path("seeded_tof.xye"),
            x=np.linspace(10000.0, 11000.0, 100),
            y=np.linspace(1.0, 2.0, 100),
            native_axis=AXIS_TOF,
            metadata={
                "pattern_calibration_profile_key": PROFILE_GSAS_TOF,
                "pattern_calibration_profile_parameters": {
                    "fwhm": 120.0,
                    "eta": 0.35,
                    "alpha": 90.0,
                    "beta": 230.0,
                },
            },
        )
        request = SpectrumFitRequest(
            spectrum_index=0,
            spectrum=spectrum,
            settings=settings,
            phase=PhaseModel(1, "Ni", 3.52, 3.52, 3.52, 90.0, 90.0, 90.0, "F m -3 m"),
            display_axis=AXIS_TOF,
            data_mode="raw",
            selected_range=(10000.0, 11000.0),
        )
        captured = {}

        def fake_fit_pawley(*args, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("captured")

        with patch("diffraction.batch_fitting.fit_pawley", side_effect=fake_fit_pawley):
            with self.assertRaisesRegex(RuntimeError, "captured"):
                fit_spectrum(request, use_gsas_convergence=True)

        self.assertEqual(
            captured["initial_profile_parameters"],
            {"fwhm": 120.0, "eta": 0.35, "alpha": 90.0, "beta": 230.0},
        )
        self.assertEqual(captured["convergence_relative_improvement"], settings.gsas_min_delta_chi_square)

    def test_ni_measurement_pawley_d_dependent_resolution(self) -> None:
        spectrum = corrected_ni_bank_1().corrected_spectrum
        phase = parse_gsas_exp(HIST / "NI.EXP")[0]
        x = spectrum.axis_values(AXIS_D_SPACING)
        mask = (x > 1.0) & (x < 2.12)
        result = fit_pawley(
            x[mask],
            spectrum.y[mask],
            spectrum.e[mask] if spectrum.e is not None else None,
            phase,
            AXIS_D_SPACING,
            spectrum.calibration,
            width_model=PAWLEY_WIDTH_D_RESOLUTION,
        )
        self.assertGreater(result.fwhm, 0.0)
        self.assertIsNotNone(result.lattice_a_uncertainty)
        self.assertGreater(float(result.lattice_a_uncertainty), 0.0)
        self.assertGreaterEqual(len(result.reflections), 4)
        self.assertTrue(3.45 < result.lattice_a < 3.65)
        self.assertIn("fwhm0_d", result.profile_parameters)
        self.assertIn("fwhm_d_slope", result.profile_parameters)
        self.assertIn("width_d_centre", result.profile_parameters)

    def test_pawley_accepts_all_gsas_tof_profiles(self) -> None:
        spectrum = load_focused_spectrum(HIST / "ENGINX371347_1.his")
        phase = parse_gsas_exp(HIST / "NI.EXP")[0]
        calibration = spectrum.calibration
        self.assertIsNotNone(calibration)
        centre = float(d_to_tof(phase.a / math.sqrt(3.0), calibration))
        x = np.linspace(centre - 500.0, centre + 700.0, 500)

        for profile_key in GSAS_TOF_PROFILE_KEYS:
            with self.subTest(profile_key=profile_key):
                y = 3.0 + 150.0 * evaluate_peak_profile(
                    profile_key,
                    x,
                    centre,
                    120.0,
                    eta=0.35,
                    tail=220.0,
                    alpha=90.0,
                    beta=230.0,
                    sig=1600.0,
                    gam=90.0,
                    alp=1.0 / 180.0,
                    bet=1.0 / 420.0,
                )
                result = fit_pawley(
                    x,
                    y,
                    None,
                    phase,
                    AXIS_TOF,
                    calibration,
                    polynomial_order=0,
                    reflection_margin_percent=0.5,
                    max_nfev=4000,
                    profile_key=profile_key,
                )
                self.assertAlmostEqual(result.lattice_a, phase.a, delta=0.003)
                self.assertIn("gsas_tof_function", result.profile_parameters)

    def test_diffraction_tab_correction_smoke(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import CalibrationDialog, DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        self.assertEqual(tab.main_splitter.orientation(), Qt.Horizontal)
        self.assertEqual(tab.results_splitter.orientation(), Qt.Vertical)
        self.assertIs(tab.result_details_edit.parentWidget(), tab.controls_panel)
        self.assertIs(tab.results_table.parentWidget(), tab.results_splitter)
        menu_titles = [action.text() for action in tab.menu_bar.actions()]
        self.assertEqual(menu_titles, ["File", "Calibration", "View", "Fit", "About"])
        sample = load_focused_spectrum(HIST / "ENGINX373922_1.his")
        tab.spectra.append(sample)
        tab.corrected_spectra.append(None)
        tab.normalization_results.append(None)
        tab.spectrum_combo.addItem("ENGINX373922_1.his")
        tab.spectrum_combo.setCurrentIndex(0)
        tab.calibration_spectrum = load_focused_spectrum(HIST / "ENGINX373845_1.his")
        tab.calibration_phase = parse_gsas_exp(HIST / "CEO2.EXP")[0]
        tab.vanadium_spectrum = load_focused_spectrum(HIST / "ENGINX371347_1.his")
        tab.refine_current_calibration()
        self.assertIsNotNone(tab.calibration_result)
        self.assertIsNotNone(tab.calibration_fit_spectrum)
        self.assertEqual(tab.calibration_fit_spectrum.metadata.get("normalization_run"), "371347")
        tab.apply_current_normalisation()
        self.assertIsNotNone(tab.corrected_spectra[0])
        calibration_dialog = CalibrationDialog(tab)
        self.assertEqual(calibration_dialog.ceo2_run_status.text(), "373845")
        self.assertEqual(calibration_dialog.vanadium_run_status.text(), "371347")
        self.assertIn("normalised", calibration_dialog.correction_status.text())
        self.assertNotEqual(calibration_dialog.difc_status.text(), "")
        calibration_dialog.close()
        tab.view_min_edit.setText("1.6")
        tab.view_max_edit.setText("2.2")
        tab.axis_combo.setCurrentIndex(1)
        tab.update_plot()
        self.assertIsNotNone(tab.toolbar)
        x_limits = tab.axes.get_xlim()
        self.assertAlmostEqual(x_limits[0], 1.6, places=3)
        self.assertAlmostEqual(x_limits[1], 2.2, places=3)
        spectrum = tab.current_spectrum()
        self.assertIsNotNone(spectrum)
        x_values = spectrum.axis_values(AXIS_D_SPACING)
        auto_y = tab._auto_y_range(x_values, spectrum.y, (1.6, 2.2))
        y_limits = tab.axes.get_ylim()
        self.assertAlmostEqual(y_limits[0], auto_y[0], places=3)
        self.assertAlmostEqual(y_limits[1], auto_y[1], places=3)
        tab.y_view_min_edit.setText("0.5")
        tab.y_view_max_edit.setText("2.5")
        tab.update_plot()
        y_limits = tab.axes.get_ylim()
        self.assertAlmostEqual(y_limits[0], 0.5, places=3)
        self.assertAlmostEqual(y_limits[1], 2.5, places=3)
        dot_lines = [line for line in tab.axes.lines if line.get_marker() == "." and line.get_linestyle() == "None"]
        self.assertTrue(dot_lines)
        tab.range_min_edit.setText("1.95")
        tab.range_max_edit.setText("2.1")
        tab.fit_selected_range()
        self.assertIsNotNone(tab.latest_fit_curve)
        self.assertTrue(tab.residual_axes.has_data())
        self.assertTrue(any(line.get_label() == "Open GENIE Voigt + exponential fit" for line in tab.axes.lines))
        first_fit_row = tab.results_table.rowCount() - 1
        headers = [
            tab.results_table.horizontalHeaderItem(column).text()
            for column in range(tab.results_table.columnCount())
        ]
        self.assertNotIn("Peak function", headers)
        self.assertNotIn("Scope / method", headers)
        self.assertNotIn("Run", headers)
        self.assertNotIn("Bank", headers)
        self.assertNotIn("Axis", headers)
        self.assertNotIn("Range min", headers)
        self.assertNotIn("Range max", headers)
        source_item = tab.results_table.item(first_fit_row, headers.index("Source"))
        self.assertEqual(source_item.text(), "373922_1")
        self.assertIn("ENGINX373922_1.his", source_item.toolTip())
        self.assertEqual(tab.results_table.item(first_fit_row, headers.index("Lattice a")).text(), "")
        self.assertTrue(tab.results_table.isColumnHidden(headers.index("Lattice a")))
        self.assertTrue(tab.results_table.isColumnHidden(headers.index("Lattice a unc")))
        self.assertTrue(tab.results_table.isColumnHidden(headers.index("Lattice b")))
        self.assertTrue(tab.results_table.isColumnHidden(headers.index("Lattice c")))
        self.assertTrue(1.95 <= float(tab.results_table.item(first_fit_row, headers.index(first_peak_header(headers, "d"))).text()) <= 2.1)
        details_text = tab.result_details_edit.toPlainText()
        self.assertIn("reported value: peak_d", details_text)
        self.assertIn("peak function: Open GENIE Voigt + exponential", details_text)
        self.assertIn("scope method: Individual peak fitting", details_text)
        self.assertIn("axis: d_spacing", details_text)
        self.assertIn("range min: 1.95", details_text)
        self.assertIn("range max: 2.1", details_text)
        calibration = spectrum.calibration
        self.assertIsNotNone(calibration)
        tof_centre = float(d_to_tof(3.5226 / math.sqrt(3.0), calibration))
        tab.axis_combo.setCurrentIndex(0)
        tab.range_min_edit.setText(f"{tof_centre - 900.0:.6g}")
        tab.range_max_edit.setText(f"{tof_centre + 900.0:.6g}")
        tab.fit_selected_range()
        tof_fit_row = tab.results_table.rowCount() - 1
        headers = [
            tab.results_table.horizontalHeaderItem(column).text()
            for column in range(tab.results_table.columnCount())
        ]
        self.assertNotIn("Range min", headers)
        self.assertNotIn("Range max", headers)
        tof_details_text = tab.result_details_edit.toPlainText()
        self.assertIn("axis: tof", tof_details_text)
        self.assertIn("range min:", tof_details_text)
        self.assertIn("range max:", tof_details_text)
        self.assertTrue(1.8 <= float(tab.results_table.item(tof_fit_row, headers.index(first_peak_header(headers, "d"))).text()) <= 2.2)
        self.assertLess(float(tab.results_table.item(tof_fit_row, headers.index(first_peak_header(headers, "FWHM d"))).text()), 1.0)
        tab.axis_combo.setCurrentIndex(1)
        tab.reset_plot_view()
        self.assertEqual(tab.view_min_edit.text(), "")
        self.assertEqual(tab.view_max_edit.text(), "")
        self.assertEqual(tab.y_view_min_edit.text(), "")
        self.assertEqual(tab.y_view_max_edit.text(), "")
        spectrum = tab.current_spectrum()
        self.assertIsNotNone(spectrum)
        self.assertIsNotNone(spectrum.calibration)
        default_d_min = float(tof_to_d(10000.0, spectrum.calibration))
        reset_x_limits = tab.axes.get_xlim()
        self.assertAlmostEqual(reset_x_limits[0], default_d_min, places=3)
        tab.range_min_edit.clear()
        tab.range_max_edit.clear()
        mask, selected_range = tab._fit_range_mask(x_values, spectrum)
        self.assertGreater(int(np.count_nonzero(mask)), 8)
        self.assertAlmostEqual(selected_range[0], default_d_min, places=3)
        tab.close()
        app.processEvents()

    def test_gsas_tof_profile_can_fit_from_d_spacing_axis(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        result = corrected_ni_bank_1()
        tab.spectra.append(result.corrected_spectrum)
        tab.corrected_spectra.append(None)
        tab.normalization_results.append(result)
        tab.spectrum_combo.addItem("ENGINX373922_1.his")
        tab.spectrum_combo.setCurrentIndex(0)
        tab.axis_combo.setCurrentIndex(1)
        tab.range_min_edit.setText("1.95")
        tab.range_max_edit.setText("2.1")
        tab.fitting_settings.peak_profile_key = PROFILE_GSAS_TOF
        tab.fit_selected_range()

        self.assertGreater(tab.results_table.rowCount(), 0)
        row = tab.results_table.rowCount() - 1
        headers = [
            tab.results_table.horizontalHeaderItem(column).text()
            for column in range(tab.results_table.columnCount())
        ]
        self.assertEqual(tab.results_table.item(row, headers.index("Source")).text(), "373922_1")
        self.assertTrue(1.95 <= float(tab.results_table.item(row, headers.index(first_peak_header(headers, "d"))).text()) <= 2.1)
        details_text = tab.result_details_edit.toPlainText()
        self.assertIn("peak function: GSAS TOF profile 3: back-to-back PV", details_text)
        self.assertIn("scope method: Individual peak fitting", details_text)
        self.assertIn("fit axis: tof", details_text)
        tab.close()
        app.processEvents()

    def test_wide_gui_peak_fit_uses_one_row_with_peak_columns(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        result = corrected_ni_bank_1()
        tab.spectra.append(result.corrected_spectrum)
        tab.corrected_spectra.append(None)
        tab.normalization_results.append(result)
        tab.spectrum_combo.addItem("ENGINX373922_1.his")
        tab.spectrum_combo.setCurrentIndex(0)
        tab.axis_combo.setCurrentIndex(1)
        tab.range_min_edit.setText("1.0")
        tab.range_max_edit.setText("2.2")
        tab.fitting_settings.peak_profile_key = PROFILE_VOIGT
        tab.fit_selected_range()

        self.assertEqual(tab.results_table.rowCount(), 1)
        headers = [
            tab.results_table.horizontalHeaderItem(column).text()
            for column in range(tab.results_table.columnCount())
        ]
        hkl_headers = [header for header in headers if header.startswith("(") and header.endswith(" d")]
        self.assertGreaterEqual(len(hkl_headers), 4)
        self.assertNotIn("Peak 1 d", headers)
        self.assertNotIn("Peak function", headers)
        self.assertNotIn("Scope / method", headers)
        row = tab.results_table.rowCount() - 1
        self.assertEqual(tab.results_table.item(row, headers.index("Source")).text(), "373922_1")
        peak_d_values = [
            float(tab.results_table.item(row, column).text())
            for column, header in enumerate(headers)
            if header.startswith("(") and header.endswith(" d")
        ]
        for expected in (1.25, 1.76, 2.07):
            self.assertTrue(any(abs(value - expected) < 0.03 for value in peak_d_values))
        self.assertGreaterEqual(len(peak_d_values), 4)
        self.assertIn("peak count:", tab.result_details_edit.toPlainText())
        tab.close()
        app.processEvents()

    def test_hkl_peak_columns_do_not_shift_when_a_peak_is_missing(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        tab._append_structured_result_row(
            ["run_1", "", "", "1", "10", "Success", "", "", "", "", ""],
            [
                ["1.05", "", "", "", "", "", ""],
                ["1.17", "", "", "", "", "", ""],
                ["1.31", "", "", "", "", "", ""],
            ],
            "",
            peak_labels=["(1 1 1)", "(2 0 0)", "(3 1 1)"],
        )
        tab._append_structured_result_row(
            ["run_2", "", "", "1", "10", "Success", "", "", "", "", ""],
            [
                ["1.06", "", "", "", "", "", ""],
                ["1.32", "", "", "", "", "", ""],
            ],
            "",
            peak_labels=["(1 1 1)", "(3 1 1)"],
        )

        headers = [
            tab.results_table.horizontalHeaderItem(column).text()
            for column in range(tab.results_table.columnCount())
        ]
        row = 1
        self.assertEqual(tab.results_table.item(row, headers.index("(1 1 1) d")).text(), "1.06")
        self.assertEqual(tab.results_table.item(row, headers.index("(2 0 0) d")).text(), "")
        self.assertEqual(tab.results_table.item(row, headers.index("(3 1 1) d")).text(), "1.32")
        tab.close()
        app.processEvents()

    def test_pawley_gui_reports_whole_pattern_lattice_parameter(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        result = corrected_ni_bank_1()
        tab.phase = parse_gsas_exp(HIST / "NI.EXP")[0]
        tab.spectra.append(result.corrected_spectrum)
        tab.corrected_spectra.append(None)
        tab.normalization_results.append(result)
        tab.spectrum_combo.addItem("ENGINX373922_1.his")
        tab.spectrum_combo.setCurrentIndex(0)
        tab.axis_combo.setCurrentIndex(1)
        tab.range_min_edit.setText("1.65")
        tab.range_max_edit.setText("2.12")
        tab.fit_scope_combo.setCurrentIndex(tab.fit_scope_combo.findData(FIT_SCOPE_PATTERN))
        tab.fitting_settings.pattern_profile_key = PROFILE_VOIGT
        tab.fitting_settings.pawley_width_model = PAWLEY_WIDTH_D_RESOLUTION
        tab.fit_selected_range()

        row = tab.results_table.rowCount() - 1
        headers = [
            tab.results_table.horizontalHeaderItem(column).text()
            for column in range(tab.results_table.columnCount())
        ]
        self.assertEqual(tab.results_table.item(row, headers.index("Source")).text(), "373922_1")
        self.assertNotIn("Pattern FWHM d", headers)
        self.assertNotIn("Total intensity", headers)
        self.assertNotIn("Peak function", headers)
        self.assertNotIn("Scope / method", headers)
        self.assertFalse(tab.results_table.isColumnHidden(headers.index("Lattice a")))
        self.assertFalse(tab.results_table.isColumnHidden(headers.index("Lattice a unc")))
        self.assertTrue(tab.results_table.isColumnHidden(headers.index("Lattice b")))
        self.assertTrue(tab.results_table.isColumnHidden(headers.index("Lattice b unc")))
        self.assertTrue(tab.results_table.isColumnHidden(headers.index("Lattice c")))
        self.assertTrue(tab.results_table.isColumnHidden(headers.index("Lattice c unc")))
        self.assertTrue(3.45 <= float(tab.results_table.item(row, headers.index("Lattice a")).text()) <= 3.60)
        self.assertGreater(float(tab.results_table.item(row, headers.index("Lattice a unc")).text()), 0.0)
        details_text = tab.result_details_edit.toPlainText()
        self.assertIn("reported value: lattice_a", details_text)
        self.assertIn("peak function: Open GENIE Voigt", details_text)
        self.assertIn("scope method: Whole pattern: Pawley", details_text)
        self.assertIn("width model: d_resolution", details_text)
        tab.close()
        app.processEvents()

    def test_diffraction_tab_auto_corrects_import_after_calibration(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        tab.calibration_spectrum = load_focused_spectrum(HIST / "ENGINX373845_1.his")
        tab.calibration_phase = parse_gsas_exp(HIST / "CEO2.EXP")[0]
        tab.vanadium_spectrum = load_focused_spectrum(HIST / "ENGINX371347_1.his")

        self.assertTrue(tab.run_calibration_sequence())
        self.assertIsNotNone(tab.calibration_result)
        self.assertEqual(tab.spectrum_combo.count(), 0)

        tab.load_spectrum_paths([HIST / "ENGINX373922_1.his"])

        self.assertEqual(tab.spectrum_combo.count(), 1)
        self.assertIsNotNone(tab.corrected_spectra[0])
        self.assertIsNotNone(tab.normalization_results[0])
        self.assertIs(tab.current_spectrum(), tab.corrected_spectra[0])
        self.assertIs(tab.current_spectrum(), tab.corrected_spectra[0])
        tab.close()
        app.processEvents()

    def test_fitting_settings_dialog_updates_tab_settings(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication, QScrollArea
            from diffraction.tab import DiffractionTab, FittingSettingsDialog
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        dialog = FittingSettingsDialog(tab)
        self.assertIsNotNone(dialog.findChild(QScrollArea))
        self.assertIs(dialog.button_box.parentWidget(), dialog)
        screen = QApplication.primaryScreen()
        if screen is not None:
            self.assertLessEqual(dialog.height(), screen.availableGeometry().height())
        dialog.fit_scope_combo.setCurrentIndex(dialog.fit_scope_combo.findData(FIT_SCOPE_PATTERN))
        dialog.profile_combo.setCurrentIndex(dialog.profile_combo.findData(PROFILE_VOIGT))
        dialog.pawley_width_model_combo.setCurrentIndex(dialog.pawley_width_model_combo.findData(PAWLEY_WIDTH_D_RESOLUTION))
        dialog.background_order_spin.setValue(3)
        dialog.max_evaluations_spin.setValue(12345)
        dialog.pawley_lattice_tolerance_spin.setValue(1.25)
        dialog.pawley_reflection_margin_spin.setValue(6.5)
        dialog.range_min_edit.setText("1.7")
        dialog.range_max_edit.setText("2.1")
        dialog.apply_settings()

        self.assertEqual(tab.fitting_settings.fit_scope, FIT_SCOPE_PATTERN)
        self.assertEqual(tab.fitting_settings.peak_profile_key, PROFILE_VOIGT)
        self.assertEqual(tab.fitting_settings.pattern_profile_key, PROFILE_VOIGT)
        self.assertEqual(tab.fitting_settings.pawley_width_model, PAWLEY_WIDTH_D_RESOLUTION)
        self.assertEqual(tab.fit_scope_combo.currentData(), FIT_SCOPE_PATTERN)
        self.assertEqual(tab.poly_order.value(), 3)
        self.assertEqual(tab.fitting_settings.max_evaluations, 12345)
        self.assertAlmostEqual(tab.fitting_settings.pawley_lattice_tolerance_percent, 1.25)
        self.assertAlmostEqual(tab.fitting_settings.pawley_reflection_margin_percent, 6.5)
        self.assertEqual(tab.range_min_edit.text(), "1.7")
        self.assertEqual(tab.range_max_edit.text(), "2.1")
        dialog.close()
        tab.close()
        app.processEvents()

    def test_fitting_settings_dialog_selects_open_genie_profile(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import DiffractionTab, FittingSettingsDialog
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        dialog = FittingSettingsDialog(tab)
        self.assertEqual(dialog.fit_scope_combo.count(), 2)
        self.assertEqual(tab.fitting_settings.peak_profile_key, PROFILE_EXP_VOIGT)
        self.assertEqual(tab.fitting_settings.pattern_profile_key, PROFILE_GSAS_TOF)
        self.assertEqual(tab.fitting_settings.pawley_width_model, PAWLEY_WIDTH_D_RESOLUTION)
        self.assertEqual(dialog.pawley_width_model_combo.currentData(), PAWLEY_WIDTH_D_RESOLUTION)
        self.assertFalse(hasattr(dialog, "peak_profile_combo"))
        self.assertFalse(hasattr(dialog, "pattern_profile_combo"))
        self.assertGreaterEqual(dialog.profile_combo.count(), 12)
        for profile_key in GSAS_TOF_PROFILE_KEYS:
            self.assertGreaterEqual(dialog.profile_combo.findData(profile_key), 0)
        dialog.fit_scope_combo.setCurrentIndex(dialog.fit_scope_combo.findData(FIT_SCOPE_PEAK))
        dialog.profile_combo.setCurrentIndex(dialog.profile_combo.findData(PROFILE_VOIGT))
        dialog.apply_settings()

        self.assertEqual(tab.fitting_settings.fit_scope, FIT_SCOPE_PEAK)
        self.assertEqual(tab.fitting_settings.peak_profile_key, PROFILE_VOIGT)
        self.assertEqual(tab.fitting_settings.pattern_profile_key, PROFILE_VOIGT)
        dialog.close()
        tab.close()
        app.processEvents()

    def test_batch_fit_all_loaded_spectra_retains_curves_and_json(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        normalized = corrected_ni_bank_1()
        sample = load_focused_spectrum(HIST / "ENGINX373922_1.his")
        for label in ("valid", "missing correction"):
            tab.spectra.append(sample)
            tab.spectrum_combo.addItem(label)
        tab.corrected_spectra.extend([normalized.corrected_spectrum, None])
        tab.normalization_results.extend([normalized, None])
        tab.spectrum_combo.setCurrentIndex(0)
        tab.axis_combo.setCurrentIndex(tab.axis_combo.findData(AXIS_D_SPACING))
        tab.range_min_edit.setText("1.95")
        tab.range_max_edit.setText("2.1")
        tab._batch_process_isolation = False  # Child-process path is covered by the explicit integration run.
        tab.fit_batch()

        deadline = time.monotonic() + 30.0
        while tab._batch_thread is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        self.assertIsNone(tab._batch_thread)
        self.assertEqual(tab.results_table.rowCount(), 2)
        headers = [
            tab.results_table.horizontalHeaderItem(column).text()
            for column in range(tab.results_table.columnCount())
        ]
        status_column = headers.index("Status")
        error_column = headers.index("Error")
        statuses = [tab.results_table.item(row, status_column).text() for row in range(2)]
        errors = [tab.results_table.item(row, error_column).text() for row in range(2)]
        self.assertCountEqual(statuses, ["Success", "Success"], msg=str(errors))
        self.assertEqual(len(tab.fit_records_by_row), 2)

        success_row = next(iter(tab.fit_records_by_row))
        tab.results_table.selectRow(success_row)
        app.processEvents()
        self.assertIsNotNone(tab.latest_fit_curve)
        self.assertTrue(tab.residual_axes.has_data())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch.json"
            tab._export_results_json(output)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["records"]), 2)
        success = next(item for item in payload["records"] if item["status"] == "success")
        self.assertEqual(success["run"], "373922")
        self.assertIsNotNone(success["curve"]["residual"])
        self.assertIsNotNone(success["curve"]["weighted_residual"])
        tab.close()
        app.processEvents()

    def test_batch_worker_continues_after_fit_failure(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import BatchFitWorker
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        QApplication.instance() or QApplication([])
        spectrum = load_focused_spectrum(HIST / "ENGINX373922_1.his")
        settings = SimpleNamespace(batch_timeout_seconds=120)
        requests = [
            SimpleNamespace(spectrum_index=index, spectrum=spectrum, settings=settings)
            for index in range(2)
        ]
        worker = BatchFitWorker(requests, threading.Event(), use_processes=False)
        failures = []
        results = []
        progress = []
        worker.failed.connect(lambda index, message: failures.append((index, message)))
        worker.result_ready.connect(results.append)
        worker.progress.connect(lambda completed, total: progress.append((completed, total)))
        with patch("diffraction.tab.fit_spectrum", side_effect=[ValueError("bad fit"), "second result"]):
            worker.run()
        self.assertEqual(failures, [(0, "bad fit")])
        self.assertEqual(results, ["second result"])
        self.assertEqual(progress, [(1, 2), (2, 2)])

    def test_spectrum_selection_restores_each_stored_fit_curve(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.batch_fitting import SpectrumFitRecord
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        calibration = load_focused_spectrum(HIST / "ENGINX373922_1.his").calibration
        x = np.linspace(19500.0, 20500.0, 300)
        y = 2.0 + 100.0 * pseudo_voigt_profile(x, 20000.0, 80.0, 0.4)
        fit_result = fit_peak_profile(PROFILE_VOIGT, x, y, polynomial_order=0)
        tab = DiffractionTab()
        settings = FittingSettings(fit_scope=FIT_SCOPE_PEAK, use_uncertainties=False)
        for index in range(2):
            spectrum = FocusedSpectrum(
                source_path=Path(f"spectrum_{index}.xye"), x=x, y=y,
                calibration=calibration, run_number=str(index), bank_number=1,
            )
            tab.spectra.append(spectrum)
            tab.corrected_spectra.append(None)
            tab.normalization_results.append(None)
            tab.spectrum_combo.addItem(spectrum.source_path.name)
            record = SpectrumFitRecord(
                spectrum_index=index, spectrum=spectrum, settings=settings,
                display_axis=AXIS_TOF, fit_axis=AXIS_TOF, data_mode="raw",
                selected_range=(19500.0, 20500.0), result=fit_result,
                curve_x=fit_result.fit_x, fit_y=fit_result.fit_y,
                observed_y=fit_result.observed_y,
            )
            tab._append_fit_record(record)

        self.assertEqual(tab._current_fit_curve()["spectrum_index"], 1)
        tab.spectrum_combo.setCurrentIndex(0)
        app.processEvents()
        self.assertEqual(tab._current_fit_curve()["spectrum_index"], 0)
        self.assertTrue(any(line.get_label().endswith("fit") for line in tab.axes.lines))
        tab.spectrum_combo.setCurrentIndex(1)
        app.processEvents()
        self.assertEqual(tab._current_fit_curve()["spectrum_index"], 1)
        partial_curve = {
            "spectrum_index": 0,
            "axis": AXIS_TOF,
            "data_mode": "raw",
            "x": fit_result.fit_x,
            "fit_y": fit_result.fit_y * 0.9,
            "observed_y": fit_result.observed_y,
            "weighted_residual": None,
            "label": "Partial Pawley fit (initial refinement)",
        }
        tab._append_failed_fit(0, "Fit timed out.", (19500.0, 20500.0), curve_payload=partial_curve)
        app.processEvents()
        self.assertEqual(tab.spectrum_combo.currentIndex(), 0)
        self.assertTrue(any(line.get_label().startswith("Partial Pawley fit") for line in tab.axes.lines))
        tab.clear_fitting_results()
        app.processEvents()
        self.assertEqual(tab.results_table.rowCount(), 0)
        self.assertEqual(tab._peak_result_column_count, 0)
        self.assertFalse(tab.fit_records_by_row)
        self.assertFalse(tab.fit_records_by_spectrum)
        self.assertFalse(tab.fit_curve_payloads_by_row)
        self.assertFalse(tab.fit_curve_payloads_by_spectrum)
        self.assertIsNone(tab._current_fit_curve())
        self.assertFalse(any(line.get_label().endswith("fit") for line in tab.axes.lines))
        self.assertEqual(tab.result_details_edit.toPlainText(), "")
        tab.close()
        app.processEvents()

    def test_repeated_plot_range_selections_replace_previous_highlight(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from matplotlib.backend_bases import MouseEvent
        from PyQt5.QtWidgets import QApplication
        from diffraction.tab import DiffractionTab

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        try:
            tab.load_spectrum_paths([HIST / "ENGINX373922_1.his"])
            for low, high in ((20000.0, 21000.0), (23000.0, 22000.0), (25000.0, 26000.0)):
                tab._on_span_selected(low, high)
                tab.update_plot()
                self.assertEqual(float(tab.range_min_edit.text()), min(low, high))
                self.assertEqual(float(tab.range_max_edit.text()), max(low, high))
                visible = [patch for patch in tab.axes.patches if patch.get_visible()]
                self.assertEqual(len(visible), 1)
                self.assertAlmostEqual(visible[0].get_x(), min(low, high))
                self.assertAlmostEqual(visible[0].get_x() + visible[0].get_width(), max(low, high))
            tab.canvas.draw()
            y_mid = sum(tab.axes.get_ylim()) / 2.0
            for name, x_value in (("button_press_event", 27000.0), ("motion_notify_event", 28000.0), ("button_release_event", 28000.0)):
                pixel_x, pixel_y = tab.axes.transData.transform((x_value, y_mid))
                event = MouseEvent(name, tab.canvas, pixel_x, pixel_y, button=1)
                tab.canvas.callbacks.process(name, event)
            self.assertAlmostEqual(float(tab.range_min_edit.text()), 27000.0, places=2)
            self.assertAlmostEqual(float(tab.range_max_edit.text()), 28000.0, places=2)
            self.assertEqual(sum(patch.get_visible() for patch in tab.axes.patches), 1)
        finally:
            tab.close()
            app.processEvents()

    def test_delete_selected_results_reindexes_records_curves_and_details(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtCore import QItemSelectionModel
        from PyQt5.QtWidgets import QApplication
        from diffraction.batch_fitting import SpectrumFitRecord
        from diffraction.tab import DiffractionTab

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        try:
            calibration = load_focused_spectrum(HIST / "ENGINX373922_1.his").calibration
            x = np.linspace(19500.0, 20500.0, 120)
            y = 2.0 + 100.0 * pseudo_voigt_profile(x, 20000.0, 80.0, 0.4)
            fit = fit_peak_profile(PROFILE_VOIGT, x, y, polynomial_order=0)
            settings = FittingSettings(fit_scope=FIT_SCOPE_PEAK, use_uncertainties=False)
            for index in range(2):
                spectrum = FocusedSpectrum(
                    source_path=Path(f"spectrum_{index}.xye"), x=x, y=y,
                    calibration=calibration, run_number=str(index), bank_number=1,
                )
                tab.spectra.append(spectrum)
                tab.corrected_spectra.append(None)
                tab.normalization_results.append(None)
                tab._add_spectrum_item(spectrum)
            records = []
            for index, scale in ((0, 1.0), (1, 1.1), (0, 1.2)):
                record = SpectrumFitRecord(
                    spectrum_index=index, spectrum=tab.spectra[index], settings=settings,
                    display_axis=AXIS_TOF, fit_axis=AXIS_TOF, data_mode="raw",
                    selected_range=(19500.0, 20500.0), result=fit,
                    curve_x=fit.fit_x, fit_y=fit.fit_y * scale, observed_y=fit.observed_y,
                )
                records.append(record)
                tab._append_fit_record(record)
            self.assertEqual(tab.results_table.rowCount(), 3)
            self.assertTrue(tab.delete_selected_button.isEnabled())
            tab._batch_thread = object()
            tab._update_delete_selected_button()
            self.assertFalse(tab.delete_selected_button.isEnabled())
            tab.delete_selected_results()
            self.assertEqual(tab.results_table.rowCount(), 3)
            tab._batch_thread = None
            tab.results_table.selectRow(1)
            tab.delete_selected_results()
            self.assertEqual(tab.results_table.rowCount(), 2)
            self.assertEqual(tab.result_details_edit.toPlainText(), "")
            self.assertEqual(len(tab.spectra), 2)
            self.assertIs(tab.fit_records_by_row[1], records[2])
            self.assertIs(tab.fit_records_by_spectrum[(0, AXIS_TOF, "raw")], records[2])
            self.assertNotIn((1, AXIS_TOF, "raw"), tab.fit_records_by_spectrum)
            self.assertEqual(tab.results_table.verticalHeaderItem(1).text(), "2")
            tab.results_table.selectRow(1)
            tab.delete_selected_results()
            self.assertEqual(tab.results_table.rowCount(), 1)
            self.assertIs(tab.fit_records_by_spectrum[(0, AXIS_TOF, "raw")], records[0])
            self.assertIs(tab.fit_curve_payloads_by_spectrum[(0, AXIS_TOF, "raw")], tab.fit_curve_payloads_by_row[0])
            tab.spectrum_combo.setCurrentIndex(0)
            np.testing.assert_allclose(tab._current_fit_curve()["fit_y"], fit.fit_y)
            tab._append_fit_record(records[1])
            tab._append_fit_record(records[2])
            tab.results_table.selectRow(1)
            self.assertIn("run: 1", tab.result_details_edit.toPlainText())
            model = tab.results_table.selectionModel()
            model.select(tab.results_table.model().index(2, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
            self.assertEqual(len(model.selectedRows()), 2)
            tab.delete_selected_results()
            self.assertEqual(tab.results_table.rowCount(), 1)
            self.assertIs(tab.fit_records_by_row[0], records[0])
            tab.results_table.selectRow(0)
            tab.delete_selected_results()
            self.assertEqual(tab.results_table.rowCount(), 0)
            partial_curve = {
                "spectrum_index": 0, "axis": AXIS_TOF, "data_mode": "raw",
                "x": fit.fit_x, "fit_y": fit.fit_y, "observed_y": fit.observed_y,
                "label": "Partial fit",
            }
            tab._append_failed_fit(0, "Timed out", (19500.0, 20500.0), curve_payload=partial_curve)
            self.assertIs(tab._current_fit_curve(), tab.fit_curve_payloads_by_row[0])
            tab.delete_selected_results()
            self.assertEqual(tab.results_table.rowCount(), 0)
            self.assertFalse(tab.fit_records_by_row)
            self.assertFalse(tab.fit_records_by_spectrum)
            self.assertFalse(tab.fit_curve_payloads_by_row)
            self.assertFalse(tab.fit_curve_payloads_by_spectrum)
            self.assertIsNone(tab._current_fit_curve())
            self.assertEqual(tab.result_details_edit.toPlainText(), "")
            self.assertFalse(tab.delete_selected_button.isEnabled())
            self.assertEqual(len(tab.spectra), 2)
        finally:
            tab.close()
            app.processEvents()

    def test_remove_multiple_loaded_spectra_clears_stale_fit_state(self) -> None:
        try:
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Diffraction tab unavailable: {exc}")

        class FakeCombo:
            def __init__(self) -> None:
                self.items: list[str] = []
                self.current = -1

            def currentIndex(self) -> int:
                return self.current

            def setCurrentIndex(self, index: int) -> None:
                self.current = int(index)

            def itemText(self, index: int) -> str:
                return self.items[index]

            def blockSignals(self, _blocked: bool) -> bool:
                return False

            def clear(self) -> None:
                self.items.clear()
                self.current = -1

            def addItem(self, label: str) -> None:
                self.items.append(label)
                if self.current < 0:
                    self.current = 0

            def count(self) -> int:
                return len(self.items)

        class FakeTable:
            def __init__(self) -> None:
                self.rows = 1

            def rowCount(self) -> int:
                return self.rows

        class FakeTab:
            _batch_thread = None

            def __init__(self) -> None:
                self.spectrum_combo = FakeCombo()
                self.results_table = FakeTable()
                self.status_text = ""
                self.clear_called = False

            def clear_fitting_results(self) -> None:
                self.clear_called = True
                self.latest_fit_curve = None
                self.fit_records_by_row.clear()
                self.fit_records_by_spectrum.clear()
                self.fit_curve_payloads_by_row.clear()
                self.fit_curve_payloads_by_spectrum.clear()
                self.results_table.rows = 0

            def update_plot(self) -> None:
                pass

            def _set_status(self, text: str) -> None:
                self.status_text = text

        tab = FakeTab()
        x = np.linspace(1.0, 2.0, 20)
        y = np.linspace(10.0, 20.0, 20)
        tab.spectra = []
        tab.corrected_spectra = []
        tab.normalization_results = []
        for index in range(3):
            spectrum = FocusedSpectrum(
                source_path=Path(f"spectrum_{index}.xye"),
                x=x,
                y=y + index,
                native_axis=AXIS_D_SPACING,
                run_number=str(index),
                bank_number=1,
            )
            tab.spectra.append(spectrum)
            tab.corrected_spectra.append(spectrum if index == 1 else None)
            tab.normalization_results.append(None)
            tab.spectrum_combo.addItem(spectrum.source_path.name)
        tab.spectrum_combo.setCurrentIndex(1)
        tab.fit_records_by_row = {}
        tab.fit_records_by_spectrum = {}
        tab.fit_records_by_spectrum[(2, AXIS_D_SPACING, "raw")] = object()
        tab.fit_curve_payloads_by_row = {}
        tab.fit_curve_payloads_by_spectrum = {}
        tab.fit_curve_payloads_by_spectrum[(2, AXIS_D_SPACING, "raw")] = {"spectrum_index": 2}
        tab.latest_fit_curve = {"spectrum_index": 2}

        DiffractionTab.remove_spectrum_indices(tab, [0, 2])

        self.assertEqual(len(tab.spectra), 1)
        self.assertEqual(tab.spectra[0].source_path.name, "spectrum_1.xye")
        self.assertEqual(tab.spectrum_combo.count(), 1)
        self.assertEqual(tab.spectrum_combo.currentIndex(), 0)
        self.assertIs(tab.corrected_spectra[0], tab.spectra[0])
        self.assertEqual(tab.results_table.rowCount(), 0)
        self.assertFalse(tab.fit_records_by_spectrum)
        self.assertFalse(tab.fit_curve_payloads_by_spectrum)
        self.assertIsNone(tab.latest_fit_curve)
        self.assertTrue(tab.clear_called)
        self.assertIn("Removed 2 spectra", tab.status_text)

    def test_isolated_batch_cancel_terminates_active_process(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import BatchFitWorker
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        QApplication.instance() or QApplication([])

        class FakeProcess:
            def __init__(self) -> None:
                self.returncode = None
                self.terminated = False

            def poll(self):
                return -15 if self.terminated else None

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout=None):
                self.returncode = -15
                return self.returncode

            def kill(self) -> None:
                self.terminated = True

        cancel_event = threading.Event()
        cancel_event.set()
        worker = BatchFitWorker([], cancel_event)
        fake_process = FakeProcess()
        request = SimpleNamespace(settings=SimpleNamespace(batch_timeout_seconds=120))
        with patch("diffraction.tab.subprocess.Popen", return_value=fake_process):
            status, message = worker._run_isolated_fit(request, time.monotonic() + 30.0)
        self.assertEqual(status, "cancelled")
        self.assertIn("cancelled", str(message).lower())
        self.assertTrue(fake_process.terminated)

    def test_cancel_batch_resets_progress_bar(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        tab._batch_cancel_event = threading.Event()
        tab._batch_completed_count = 4
        tab.batch_progress.setRange(0, 10)
        tab.batch_progress.setValue(4)
        tab.cancel_batch_fit()
        self.assertTrue(tab._batch_cancel_event.is_set())
        self.assertEqual(tab.batch_progress.minimum(), 0)
        self.assertEqual(tab.batch_progress.maximum(), 1)
        self.assertEqual(tab.batch_progress.value(), 0)
        tab.close()
        app.processEvents()

    def test_single_fit_uses_timeout_and_gsas_convergence_guard(self) -> None:
        try:
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Diffraction tab unavailable: {exc}")

        class FakeCombo:
            def currentIndex(self) -> int:
                return 7

        class FakeTab:
            _batch_thread = None

            def __init__(self) -> None:
                self.fitting_settings = FittingSettings(batch_timeout_seconds=1)
                self.spectrum_combo = FakeCombo()
                self.status_text = ""

            def _set_status(self, text: str) -> None:
                self.status_text = text

            def _sync_fitting_settings_from_controls(self) -> None:
                pass

            def _selected_fit_range(self) -> tuple[float, float]:
                return (1.0, 2.0)

            def _fit_request(self, index: int, selected_range: tuple[float, float]):
                self.request_index = index
                self.request_range = selected_range
                return "request"

        captured = {}

        def fake_fit_spectrum(request, **kwargs):
            captured["request"] = request
            captured.update(kwargs)
            raise RuntimeError("stop after capture")

        tab = FakeTab()
        with patch("diffraction.tab.fit_spectrum", side_effect=fake_fit_spectrum):
            DiffractionTab.fit_selected_range(tab)

        self.assertEqual(captured["request"], "request")
        self.assertEqual(tab.request_index, 7)
        self.assertEqual(tab.request_range, (1.0, 2.0))
        self.assertTrue(captured["use_gsas_convergence"])
        self.assertTrue(callable(captured["abort_check"]))
        self.assertIsNone(captured["abort_check"]())
        self.assertIn("Fit failed: stop after capture", tab.status_text)


if __name__ == "__main__":
    unittest.main()
