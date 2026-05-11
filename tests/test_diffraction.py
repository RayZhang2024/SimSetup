from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

import numpy as np

from diffraction.calibration import refine_ceo2_calibration
from diffraction.fitting import fit_pseudo_voigt, pseudo_voigt_profile
from diffraction.gsas_exp import generate_fcc_reflections, parse_gsas_exp
from diffraction.importers import load_focused_spectrum
from diffraction.models import AXIS_D_SPACING, AXIS_TOF, d_to_tof, tof_to_d
from diffraction.normalization import apply_vanadium_normalization
from diffraction.pawley import fit_pawley


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

    def test_ceo2_calibration(self) -> None:
        ceo2 = load_focused_spectrum(HIST / "ENGINX373845_1.his")
        phase = parse_gsas_exp(HIST / "CEO2.EXP")[0]
        result = refine_ceo2_calibration(ceo2, phase)
        self.assertEqual(result.run_number, "373845")
        self.assertEqual(result.bank_number, 1)
        self.assertGreaterEqual(len(result.accepted_peaks), 8)
        self.assertTrue(18000.0 < result.calibration.difc < 18700.0)
        self.assertTrue(-200.0 < result.calibration.tzero < 200.0)
        self.assertLess(result.rms_residual_tof, 25.0)

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
        self.assertGreaterEqual(len(result.reflections), 2)
        self.assertTrue(3.45 < result.lattice_a < 3.60)

    def test_diffraction_tab_correction_smoke(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
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
        tab.apply_current_normalisation()
        self.assertIsNotNone(tab.corrected_spectra[0])
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
        self.assertTrue(any(line.get_label() == "Pseudo-Voigt fit" for line in tab.axes.lines))
        tab.reset_plot_view()
        self.assertEqual(tab.view_min_edit.text(), "")
        self.assertEqual(tab.view_max_edit.text(), "")
        self.assertEqual(tab.y_view_min_edit.text(), "")
        self.assertEqual(tab.y_view_max_edit.text(), "")
        tab.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
