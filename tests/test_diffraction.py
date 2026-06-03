from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

import numpy as np

from diffraction.calibration import OPENGENIE_CALIBRATION_TOF_MAX, OPENGENIE_CALIBRATION_TOF_MIN, refine_ceo2_calibration
from diffraction.fitting import fit_peak_profile, fit_pseudo_voigt, pseudo_voigt_profile
from diffraction.gsas_exp import generate_fcc_reflections, parse_gsas_exp
from diffraction.importers import load_focused_spectrum
from diffraction.models import (
    AXIS_D_SPACING,
    AXIS_TOF,
    FIT_SCOPE_PATTERN,
    FIT_SCOPE_PEAK,
    PAWLEY_WIDTH_D_RESOLUTION,
    d_to_tof,
    tof_to_d,
)
from diffraction.normalization import apply_vanadium_normalization
from diffraction.pawley import fit_pawley
from diffraction.profiles import (
    GSAS_TOF_PROFILE_KEYS,
    PROFILE_EXP_VOIGT,
    PROFILE_GSAS_TOF,
    PROFILE_VOIGT,
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
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import CalibrationDialog, DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
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
        self.assertEqual(tab.results_table.item(first_fit_row, 1).text(), "Individual peak fitting")
        self.assertEqual(tab.results_table.item(first_fit_row, 5).text(), AXIS_D_SPACING)
        self.assertEqual(tab.results_table.item(first_fit_row, 8).text(), "")
        self.assertTrue(1.95 <= float(tab.results_table.item(first_fit_row, headers.index("Peak 1 d")).text()) <= 2.1)
        self.assertIn("reported value: peak_d", tab.result_details_edit.toPlainText())
        calibration = spectrum.calibration
        self.assertIsNotNone(calibration)
        tof_centre = float(d_to_tof(3.5226 / math.sqrt(3.0), calibration))
        tab.axis_combo.setCurrentIndex(0)
        tab.range_min_edit.setText(f"{tof_centre - 900.0:.6g}")
        tab.range_max_edit.setText(f"{tof_centre + 900.0:.6g}")
        tab.fit_selected_range()
        tof_fit_row = tab.results_table.rowCount() - 1
        self.assertEqual(tab.results_table.item(tof_fit_row, 1).text(), "Individual peak fitting")
        self.assertEqual(tab.results_table.item(tof_fit_row, 5).text(), AXIS_TOF)
        self.assertGreater(float(tab.results_table.item(tof_fit_row, 6).text()), 10000.0)
        self.assertTrue(1.8 <= float(tab.results_table.item(tof_fit_row, headers.index("Peak 1 d")).text()) <= 2.2)
        self.assertLess(float(tab.results_table.item(tof_fit_row, headers.index("Peak 1 FWHM d")).text()), 1.0)
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
        self.assertEqual(tab.results_table.item(row, 0).text(), "GSAS TOF profile 3: back-to-back PV")
        self.assertEqual(tab.results_table.item(row, 1).text(), "Individual peak fitting")
        self.assertEqual(tab.results_table.item(row, 5).text(), AXIS_D_SPACING)
        headers = [
            tab.results_table.horizontalHeaderItem(column).text()
            for column in range(tab.results_table.columnCount())
        ]
        self.assertTrue(1.95 <= float(tab.results_table.item(row, headers.index("Peak 1 d")).text()) <= 2.1)
        self.assertIn("fit axis: tof", tab.result_details_edit.toPlainText())
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
        self.assertIn("Peak 1 d", headers)
        self.assertIn("Peak 4 d", headers)
        row = tab.results_table.rowCount() - 1
        self.assertEqual(tab.results_table.item(row, 0).text(), "Open GENIE Voigt")
        self.assertEqual(tab.results_table.item(row, 1).text(), "Individual peak fitting")
        peak_d_values = [
            float(tab.results_table.item(row, headers.index(f"Peak {index} d")).text())
            for index in range(1, 5)
        ]
        for expected in (1.08, 1.27, 1.80, 2.07):
            self.assertTrue(any(abs(value - expected) < 0.03 for value in peak_d_values))
        self.assertIn("peak count: 4", tab.result_details_edit.toPlainText())
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
        self.assertEqual(tab.results_table.item(row, 0).text(), "Open GENIE Voigt")
        self.assertEqual(tab.results_table.item(row, 1).text(), "Whole pattern: Pawley")
        headers = [
            tab.results_table.horizontalHeaderItem(column).text()
            for column in range(tab.results_table.columnCount())
        ]
        self.assertNotIn("Pattern FWHM d", headers)
        self.assertNotIn("Total intensity", headers)
        self.assertTrue(3.45 <= float(tab.results_table.item(row, headers.index("Lattice a")).text()) <= 3.60)
        self.assertGreater(float(tab.results_table.item(row, headers.index("Lattice a unc")).text()), 0.0)
        details_text = tab.result_details_edit.toPlainText()
        self.assertIn("reported value: lattice_a", details_text)
        self.assertIn("peak function: Open GENIE Voigt", details_text)
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
        self.assertEqual(tab.data_combo.currentData(), "corrected")
        self.assertIs(tab.current_spectrum(), tab.corrected_spectra[0])
        tab.close()
        app.processEvents()

    def test_fitting_settings_dialog_updates_tab_settings(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import DiffractionTab, FittingSettingsDialog
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        dialog = FittingSettingsDialog(tab)
        dialog.fit_scope_combo.setCurrentIndex(dialog.fit_scope_combo.findData(FIT_SCOPE_PATTERN))
        dialog.pattern_profile_combo.setCurrentIndex(dialog.pattern_profile_combo.findData(PROFILE_VOIGT))
        dialog.pawley_width_model_combo.setCurrentIndex(dialog.pawley_width_model_combo.findData(PAWLEY_WIDTH_D_RESOLUTION))
        dialog.background_order_spin.setValue(3)
        dialog.max_evaluations_spin.setValue(12345)
        dialog.pawley_lattice_tolerance_spin.setValue(1.25)
        dialog.pawley_reflection_margin_spin.setValue(6.5)
        dialog.range_min_edit.setText("1.7")
        dialog.range_max_edit.setText("2.1")
        dialog.apply_settings()

        self.assertEqual(tab.fitting_settings.fit_scope, FIT_SCOPE_PATTERN)
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
        self.assertGreaterEqual(dialog.peak_profile_combo.count(), 12)
        self.assertGreaterEqual(dialog.pattern_profile_combo.count(), 12)
        for profile_key in GSAS_TOF_PROFILE_KEYS:
            self.assertGreaterEqual(dialog.pattern_profile_combo.findData(profile_key), 0)
        dialog.fit_scope_combo.setCurrentIndex(dialog.fit_scope_combo.findData(FIT_SCOPE_PEAK))
        dialog.peak_profile_combo.setCurrentIndex(dialog.peak_profile_combo.findData(PROFILE_VOIGT))
        dialog.apply_settings()

        self.assertEqual(tab.fitting_settings.fit_scope, FIT_SCOPE_PEAK)
        self.assertEqual(tab.fitting_settings.peak_profile_key, PROFILE_VOIGT)
        dialog.close()
        tab.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
