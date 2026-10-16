from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from diffraction.calibration import _quadratic_calibration_from_peaks
from diffraction.focusing import FOCUSING_CONVENTION_VERSION, focus_enginx_bank, geometric_difc, rebin_histogram_overlap
from diffraction.models import CalibrationPeakResult, FocusedSpectrum, InstrumentCalibration, Reflection, d_to_tof, tof_to_d
from diffraction.nexus import read_enginx_histogram_nexus
from diffraction.normalization import apply_charge_reference_normalization
from diffraction.raw_workflow import (
    RawBankCalibrationRecord,
    RawCalibrationBundle,
    analyze_sample_nexus,
    apply_saved_reference_response,
    apply_bundle_calibration,
    bundle_fingerprint,
    load_raw_calibration_bundle,
    save_raw_calibration_bundle,
    validate_raw_calibration_bundle,
)


WORKSPACE_TEMP = Path(__file__).resolve().parents[1] / ".codex-temp"


@contextmanager
def _temporary_path(suffix: str):
    handle = tempfile.NamedTemporaryFile(dir=WORKSPACE_TEMP, suffix=suffix, delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _write_synthetic_nexus(path: Path, *, reordered: bool = True) -> None:
    import h5py

    spectrum_ids = np.concatenate((np.arange(1, 2401), np.arange(2405, 2514))).astype(np.int32)
    rng = np.random.default_rng(42)
    order = rng.permutation(spectrum_ids.size) if reordered else np.arange(spectrum_ids.size)
    spectrum_ids = spectrum_ids[order]
    tof_edges = np.linspace(5000.0, 45000.0, 81)
    counts = np.zeros((1, spectrum_ids.size, tof_edges.size - 1), dtype=np.int32)
    distance = np.empty(spectrum_ids.size, dtype=np.float32)
    angle = np.empty(spectrum_ids.size, dtype=np.float32)
    for row, spectrum_id in enumerate(spectrum_ids):
        if spectrum_id <= 1200:
            fraction = (spectrum_id - 1) / 1199.0
            angle[row] = 75.0 + 30.0 * fraction
            distance[row] = 1.2 + 0.7 * fraction
            counts[0, row] = spectrum_id % 7 + 1
        elif spectrum_id <= 2400:
            fraction = (spectrum_id - 1201) / 1199.0
            angle[row] = -105.0 + 30.0 * fraction
            distance[row] = 1.1 + 0.8 * fraction
            counts[0, row] = spectrum_id % 5 + 1
        else:
            angle[row] = 180.0
            distance[row] = 0.03
            counts[0, row] = 1000000
    with h5py.File(path, "w") as handle:
        root = handle.create_group("raw_data_1")
        detector = root.create_group("detector_1")
        counts_ds = detector.create_dataset("counts", data=counts)
        counts_ds.attrs["units"] = "counts"
        tof_ds = detector.create_dataset("time_of_flight", data=tof_edges)
        tof_ds.attrs["units"] = "microsecond"
        detector.create_dataset("spectrum_index", data=spectrum_ids)
        instrument = root.create_group("instrument")
        geometry = instrument.create_group("detector_1")
        distance_ds = geometry.create_dataset("distance", data=distance)
        distance_ds.attrs["units"] = "metre"
        angle_ds = geometry.create_dataset("polar_angle", data=angle)
        angle_ds.attrs["units"] = "degree"
        azimuth_ds = geometry.create_dataset("azimuthal_angle", data=np.full_like(angle, 2.0))
        azimuth_ds.attrs["units"] = "degree"
        geometry.create_dataset("source_detector_distance", data=[0.0])
        periods = root.create_group("periods")
        charge = periods.create_dataset("proton_charge", data=[12.5])
        charge.attrs["units"] = "uAh"
        root.create_dataset("proton_charge", data=[12.5]).attrs["units"] = "uAh"
        root.create_dataset("run_number", data=[123456])
        root.create_dataset("title", data=[b"synthetic"])


def _record(bank: int) -> RawBankCalibrationRecord:
    first_id = 1 if bank == 1 else 1201
    return RawBankCalibrationRecord(
        bank_number=bank,
        bank_name=f"Bank {bank}",
        calibration_model="linear DIFC/TZERO from individual CeO2 peak fits",
        difc=18350.0 + bank,
        tzero=-30.0,
        difa=0.0,
        coefficient_units={"difc": "microsecond/angstrom", "tzero": "microsecond", "difa": "microsecond/angstrom^2"},
        difc_uncertainty=1.0,
        tzero_uncertainty=2.0,
        difa_uncertainty=None,
        rms_residual_tof=3.0,
        usable_tof_range_us=(0.0, 3.0),
        peak_diagnostics=({"hkl": "111", "accepted": True},),
        focusing={
            "masked_spectrum_ids": [],
            "target_geometric_difc": 18000.0,
            "pixel_geometric_difc_by_spectrum": {str(first_id): 18000.0},
        },
        reference_response_settings={"smoothed": False},
        response_bin_edges_us=(0.0, 1.0, 2.0, 3.0),
        response_values_per_uah=(1.0, 1.0, 1.0),
        response_uncertainties_per_uah=(0.1, 0.1, 0.1),
        response_valid_mask=(True, True, True),
        response_coverage_fraction=(1.0, 1.0, 1.0),
    )


def _bundle(offset: float = 0.0) -> RawCalibrationBundle:
    records = []
    for bank in (1, 2):
        record = _record(bank)
        if offset:
            record = RawBankCalibrationRecord(**{**record.__dict__, "difc": record.difc + offset})
        records.append(record)
    return RawCalibrationBundle(
        processing_version=FOCUSING_CONVENTION_VERSION,
        source_runs={"reference": {"run": "1"}, "ceo2": {"run": "2"}},
        ceo2_standard={"name": "CeO2", "lattice_parameter_angstrom": 5.41165},
        scientific_status="provisional",
        remaining_validation_gaps=("independent focused reference unavailable",),
        banks=tuple(records),
    )


class QuadraticCalibrationTests(unittest.TestCase):
    @staticmethod
    def _peaks(d_values: np.ndarray, *, outlier_index: int | None = None) -> list[CalibrationPeakResult]:
        calibration = InstrumentCalibration(difc=18350.0, difa=-72.0, tzero=-28.0)
        peaks = []
        for index, d in enumerate(d_values):
            expected = float(d_to_tof(d, calibration))
            observed = expected + (500.0 if index == outlier_index else 0.05 * (-1)**index)
            peaks.append(CalibrationPeakResult(
                reflection=Reflection(index + 1, 0, 0, float(d)),
                expected_tof=expected, observed_tof=observed, fitted_tof=observed,
                residual_tof=observed - expected, prominence=100.0, accepted=True,
                tof_uncertainty=2.0 + index * 0.1,
            ))
        return peaks

    def test_weighted_quadratic_recovers_coefficients_and_residuals(self) -> None:
        peaks = self._peaks(np.linspace(1.0, 2.4, 8))
        calibration, rms, updated, uncertainties = _quadratic_calibration_from_peaks(peaks, 4)
        self.assertAlmostEqual(calibration.difc, 18350.0, delta=1.0)
        self.assertAlmostEqual(calibration.difa, -72.0, delta=1.0)
        self.assertAlmostEqual(calibration.tzero, -28.0, delta=1.0)
        self.assertLess(rms, 0.1)
        self.assertTrue(all(peak.accepted for peak in updated))
        self.assertTrue(all(np.isfinite(value) and value > 0.0 for value in uncertainties.values()))
        for peak in updated:
            self.assertAlmostEqual(peak.residual_tof, peak.fitted_tof - d_to_tof(peak.reflection.d_spacing, calibration))

    def test_quadratic_outlier_and_insufficient_span_are_reported(self) -> None:
        peaks = self._peaks(np.linspace(1.0, 2.4, 8), outlier_index=3)
        calibration, _rms, updated, _uncertainties = _quadratic_calibration_from_peaks(peaks, 4)
        self.assertFalse(updated[3].accepted)
        self.assertIn("outlier", updated[3].rejection_reason)
        self.assertAlmostEqual(calibration.difa, -72.0, delta=1.0)
        with self.assertRaisesRegex(ValueError, "span is too small"):
            _quadratic_calibration_from_peaks(self._peaks(np.linspace(1.0, 1.000001, 5)), 4)
        incomplete = self._peaks(np.linspace(1.0, 2.0, 4))
        incomplete[0] = replace(incomplete[0], tof_uncertainty=None)
        with self.assertRaisesRegex(ValueError, "finite, positive TOF uncertainties"):
            _quadratic_calibration_from_peaks(incomplete, 4)


class RawNexusReaderTests(unittest.TestCase):
    def test_calibration_save_dialog_suggests_source_run_numbers(self) -> None:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        from diffraction.raw_dialog import CreateCalibrationDialog
        from diffraction.tab import DiffractionTab

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        dialog = CreateCalibrationDialog(tab)
        try:
            dialog.bundle = replace(
                _bundle(),
                source_runs={"reference": {"run": "377438"}, "ceo2": {"run": "377898"}},
            )
            with patch("diffraction.raw_dialog.QFileDialog.getSaveFileName", return_value=("", "")) as choose_path:
                dialog.save_bundle()
            self.assertEqual(choose_path.call_args.args[2], "ENGINX_377438_377898_calibration.json")
            dialog.bundle = _bundle()
            with patch("diffraction.raw_dialog.QFileDialog.getSaveFileName", return_value=("", "")) as choose_path:
                dialog.save_bundle()
            self.assertEqual(choose_path.call_args.args[2], "ENGINX_000001_000002_calibration.json")
        finally:
            dialog.close()
            tab.close()
            app.processEvents()

    def test_reader_maps_reordered_rows_by_spectrum_id_and_excludes_transmission(self) -> None:
        with _temporary_path(".nxs") as path:
            _write_synthetic_nexus(path)
            run = read_enginx_histogram_nexus(path)
        self.assertEqual(run.run_number, "123456")
        self.assertEqual(run.proton_charge_uah, 12.5)
        self.assertEqual(run.banks[1].spectrum_ids.size, 1200)
        self.assertEqual(run.banks[2].spectrum_ids.size, 1200)
        self.assertEqual(set(run.banks[1].spectrum_ids), set(range(1, 1201)))
        self.assertEqual(set(run.banks[2].spectrum_ids), set(range(1201, 2401)))
        self.assertEqual(run.transmission_spectrum_ids.size, 109)
        spectrum, diagnostics = focus_enginx_bank(run, 1)
        self.assertEqual(diagnostics.transmission_spectra_used, 0)
        self.assertLess(float(np.max(spectrum.y)), 1200.0 * 8.0)

    def test_invalid_charge_is_actionable(self) -> None:
        with _temporary_path(".nxs") as path:
            _write_synthetic_nexus(path)
            import h5py

            with h5py.File(path, "r+") as handle:
                handle["raw_data_1/periods/proton_charge"][0] = 0.0
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                read_enginx_histogram_nexus(path)


class FocusingTests(unittest.TestCase):
    def test_overlap_rebin_conserves_full_support_and_handles_partial_overlap(self) -> None:
        counts, variance = rebin_histogram_overlap(
            np.array([0.0, 1.0, 2.0]),
            np.array([10.0, 20.0]),
            np.array([0.0, 0.5, 1.5, 2.0]),
        )
        np.testing.assert_allclose(counts, [5.0, 15.0, 10.0])
        np.testing.assert_allclose(variance, counts)
        partial, _ = rebin_histogram_overlap(
            np.array([0.0, 1.0, 2.0]),
            np.array([10.0, 20.0]),
            np.array([0.5, 1.5]),
        )
        np.testing.assert_allclose(partial, [15.0])

    def test_geometry_focusing_aligns_known_pixel_peaks(self) -> None:
        from diffraction.nexus import RawEnginXNexusRun, RawNexusBank

        edges = np.linspace(10000.0, 40000.0, 3001)
        centres = 0.5 * (edges[:-1] + edges[1:])
        distances = np.array([1.1, 1.5, 1.9])
        angles = np.array([75.0, 90.0, 105.0])
        pixel_difc = geometric_difc(distances, angles, 50.0)
        d_spacing = 1.5
        counts = np.stack(
            [1000.0 * np.exp(-0.5 * ((centres - difc * d_spacing) / 25.0) ** 2) for difc in pixel_difc]
        )
        bank = RawNexusBank(1, "North/+90 degree", np.array([1, 2, 3]), np.arange(3), counts, distances, angles, np.zeros(3))
        run = RawEnginXNexusRun(
            Path("synthetic.nxs"), "1", "synthetic", 0, 1.0, edges, {1: bank}, np.array([], dtype=int), {}
        )
        focused, diagnostics = focus_enginx_bank(run, 1)
        observed = focused.x_centres[int(np.argmax(focused.y))]
        self.assertAlmostEqual(observed, diagnostics.target_geometric_difc * d_spacing, delta=15.0)

    def test_saved_pixel_basis_rejects_changed_geometry(self) -> None:
        with _temporary_path(".nxs") as path:
            _write_synthetic_nexus(path, reordered=False)
            run = read_enginx_histogram_nexus(path)
        focused, _ = focus_enginx_bank(run, 1)
        saved = {int(key): float(value) for key, value in focused.metadata["pixel_geometric_difc_by_spectrum"].items()}
        changed = dict(saved)
        changed[1] *= 1.01
        with self.assertRaisesRegex(ValueError, "geometry is incompatible"):
            focus_enginx_bank(run, 1, pixel_geometric_difc_by_spectrum=changed)

    def test_saved_basis_does_not_require_masked_pixel_entries(self) -> None:
        with _temporary_path(".nxs") as path:
            _write_synthetic_nexus(path, reordered=False)
            run = read_enginx_histogram_nexus(path)
        focused, _ = focus_enginx_bank(run, 1, masked_spectrum_ids={1})
        saved = {int(key): float(value) for key, value in focused.metadata["pixel_geometric_difc_by_spectrum"].items()}
        reapplied, _ = focus_enginx_bank(
            run,
            1,
            masked_spectrum_ids={1},
            target_geometric_difc=float(focused.metadata["target_geometric_difc"]),
            pixel_geometric_difc_by_spectrum=saved,
        )
        np.testing.assert_allclose(reapplied.y, focused.y)


class ChargeNormalizationTests(unittest.TestCase):
    @staticmethod
    def _spectrum(counts: np.ndarray, charge: float, bank: int = 1) -> FocusedSpectrum:
        return FocusedSpectrum(
            source_path=Path("synthetic.nxs"),
            x=np.arange(counts.size + 1, dtype=float),
            y=np.asarray(counts, dtype=float),
            e=np.sqrt(np.asarray(counts, dtype=float)),
            x_is_edges=True,
            metadata={"proton_charge": charge, "proton_charge_units": "uAh"},
            bank_number=bank,
        )

    def test_exposure_change_leaves_ratio_unchanged(self) -> None:
        reference = self._spectrum(np.array([100.0, 200.0, 400.0]), 10.0)
        first = apply_charge_reference_normalization(self._spectrum(np.array([50.0, 100.0, 200.0]), 5.0), reference)
        second = apply_charge_reference_normalization(self._spectrum(np.array([150.0, 300.0, 600.0]), 15.0), reference)
        np.testing.assert_allclose(first.corrected_spectrum.y, second.corrected_spectrum.y)

    def test_zero_reference_is_masked(self) -> None:
        sample = self._spectrum(np.array([10.0, 10.0, 10.0]), 1.0)
        reference = self._spectrum(np.array([10.0, 0.0, 5.0]), 1.0)
        result = apply_charge_reference_normalization(sample, reference)
        self.assertTrue(np.isnan(result.corrected_spectrum.y[1]))
        self.assertEqual(result.invalid_bins, 1)


class PersistenceTests(unittest.TestCase):
    def test_quadratic_bundle_round_trip_and_sample_application(self) -> None:
        records = tuple(replace(
            record,
            calibration_model="quadratic DIFC/DIFA/TZERO from individual CeO2 peak fits",
            difa=-42.0,
            difa_uncertainty=3.0,
        ) for record in _bundle().banks)
        bundle = replace(_bundle(), banks=records)
        with _temporary_path(".json") as path:
            save_raw_calibration_bundle(bundle, path)
            loaded = load_raw_calibration_bundle(path)
        self.assertEqual(loaded.bank(1).difa, -42.0)
        self.assertEqual(loaded.bank(1).difa_uncertainty, 3.0)
        spectrum = FocusedSpectrum(
            source_path=Path("sample.nxs"), x=np.array([0.0, 1.0, 2.0, 3.0]),
            y=np.ones(3), x_is_edges=True,
            metadata={"processing_version": FOCUSING_CONVENTION_VERSION, "target_geometric_difc": 18000.0},
            bank_number=1,
        )
        calibrated = apply_bundle_calibration(spectrum, loaded)
        self.assertEqual(calibrated.calibration.difa, -42.0)
        d = np.array([1.2, 1.8])
        np.testing.assert_allclose(tof_to_d(d_to_tof(d, calibrated.calibration), calibrated.calibration), d)
        invalid = replace(bundle, banks=(replace(records[0], difa=float("nan")), records[1]))
        with self.assertRaisesRegex(ValueError, "invalid calibration coefficients"):
            validate_raw_calibration_bundle(invalid)

    def test_round_trip_and_exactly_once_application(self) -> None:
        bundle = _bundle()
        with _temporary_path(".json") as path:
            save_raw_calibration_bundle(bundle, path)
            loaded = load_raw_calibration_bundle(path)
        spectrum = FocusedSpectrum(
            source_path=Path("sample.nxs"),
            x=np.array([0.0, 1.0, 2.0, 3.0]),
            y=np.array([1.0, 2.0, 3.0]),
            x_is_edges=True,
            metadata={
                "calibration_application_count": 0,
                "processing_version": FOCUSING_CONVENTION_VERSION,
                "target_geometric_difc": 18000.0,
            },
            bank_number=1,
        )
        calibrated = apply_bundle_calibration(spectrum, loaded)
        self.assertEqual(calibrated.calibration.difc, 18351.0)
        self.assertEqual(calibrated.metadata["calibration_application_count"], 1)
        with self.assertRaisesRegex(ValueError, "already been applied"):
            apply_bundle_calibration(calibrated, loaded)

    def test_response_rebin_is_unbiased_for_different_bin_widths(self) -> None:
        sample = FocusedSpectrum(
            source_path=Path("sample.nxs"),
            x=np.array([0.0, 0.5, 1.5, 3.0]),
            y=np.array([5.0, 10.0, 15.0]),
            e=np.sqrt(np.array([5.0, 10.0, 15.0])),
            x_is_edges=True,
            metadata={"proton_charge": 10.0, "detector_coverage_fraction": [1.0, 1.0, 1.0]},
            bank_number=1,
        )
        corrected = apply_saved_reference_response(sample, _record(1))
        np.testing.assert_allclose(corrected.y, [1.0, 1.0, 1.0])
        self.assertTrue(corrected.metadata["reference_response_rebinned"])

    def test_partial_sample_detector_coverage_is_masked(self) -> None:
        sample = FocusedSpectrum(
            source_path=Path("sample.nxs"), x=np.arange(4.0), y=np.ones(3), x_is_edges=True,
            metadata={"proton_charge": 1.0, "detector_coverage_fraction": [1.0, 0.8, 1.0]}, bank_number=1,
        )
        corrected = apply_saved_reference_response(sample, _record(1))
        self.assertTrue(np.isnan(corrected.y[1]))


class RawNexusUiTests(unittest.TestCase):
    def test_calibration_menu_and_creation_dialog_have_separate_workflows(self) -> None:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt5.QtWidgets import QApplication
            from diffraction.raw_dialog import RawNexusCalibrationDialog
            from diffraction.tab import DiffractionTab
        except Exception as exc:
            self.skipTest(f"Qt smoke test unavailable: {exc}")
        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        dialog = RawNexusCalibrationDialog(tab)
        self.assertFalse(hasattr(dialog, "sample_path"))
        self.assertTrue(hasattr(dialog, "reference_path"))
        self.assertTrue(hasattr(dialog, "ceo2_path"))
        self.assertFalse(dialog.smooth_reference.isChecked())
        calibration_menu = next(action.menu() for action in tab.menu_bar.actions() if action.text() == "Calibration")
        self.assertEqual(
            [action.text() for action in calibration_menu.actions()],
            ["Create Calibration…", "Load Calibration JSON…", "Calibration Details…"],
        )
        dialog.close()
        tab.close()
        app.processEvents()

    def test_raw_json_calibration_is_not_overwritten_and_stale_samples_are_removed(self) -> None:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        from diffraction.tab import DiffractionTab

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        spectrum = FocusedSpectrum(
            source_path=Path("sample.nxs"), x=np.arange(4.0), y=np.ones(3), x_is_edges=True,
            calibration=InstrumentCalibration(18351.0, -30.0, 0.0), bank_number=1,
            metadata={
                "raw_nxs_processed": True,
                "calibration_source_kind": "raw_nxs_json",
                "calibration_application_count": 1,
                "calibration_bundle_fingerprint": bundle_fingerprint(_bundle()),
            },
        )
        tab.calibration_result = object()
        self.assertIs(tab._with_refined_calibration_if_matching(spectrum), spectrum)
        tab.add_processed_spectrum(spectrum, update=False)
        tab.activate_raw_calibration(_bundle(10.0), Path("replacement.json"))
        self.assertEqual(tab.spectra, [])
        tab.close()
        app.processEvents()

    def test_real_run_previews_use_measured_intermediates_without_activation(self) -> None:
        import os

        reference = Path.home() / "Downloads" / "ENGINX00377438.nxs"
        ceo2 = Path.home() / "Downloads" / "ENGINX00377898.nxs"
        if not reference.is_file() or not ceo2.is_file():
            self.skipTest("Local ENGIN-X measurements are unavailable.")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        from diffraction.raw_dialog import CreateCalibrationDialog
        from diffraction.raw_previews import OpenBeamPreview, Ceo2FitPreview, QuadraticCalibrationPreview
        from diffraction.tab import DiffractionTab

        app = QApplication.instance() or QApplication([])
        tab = DiffractionTab()
        dialog = CreateCalibrationDialog(tab)
        self.assertFalse(dialog.open_beam_preview_button.isEnabled())
        self.assertFalse(dialog.ceo2_review_button.isEnabled())
        self.assertFalse(dialog.quadratic_review_button.isEnabled())
        dialog.reference_path.setText(str(reference))
        dialog.ceo2_path.setText(str(ceo2))
        dialog.process_banks()
        self.assertIsNotNone(dialog.bundle, dialog.status.text())
        fingerprint = bundle_fingerprint(dialog.bundle)
        previews = [
            OpenBeamPreview(dialog, dialog.bundle, dialog._run_diagnostics),
            Ceo2FitPreview(dialog, dialog.bundle, dialog._run_diagnostics),
            QuadraticCalibrationPreview(dialog, dialog.bundle, dialog._run_diagnostics),
        ]
        for button, preview in zip(
            (dialog.open_beam_preview_button, dialog.ceo2_review_button, dialog.quadratic_review_button), previews
        ):
            self.assertTrue(button.isEnabled())
            self.assertEqual(preview.tabs.count(), 2)
        for bank in (1, 2):
            page = previews[0].tabs.widget(bank - 1)
            plotted = page.layout().itemAt(0).widget().figure.axes[0].lines
            self.assertEqual(len(plotted), 2)
            reference_spectrum = dialog._run_diagnostics[bank]["focused_reference"]
            np.testing.assert_allclose(
                plotted[0].get_ydata(),
                reference_spectrum.y / reference_spectrum.metadata["proton_charge"],
            )
            local_fits = dialog._run_diagnostics[bank]["local_peak_fits"]
            self.assertGreater(len(local_fits), 0)
            selected = previews[1]._selectors[bank].currentData()
            peak = dialog._run_diagnostics[bank]["calibration"].peak_results[selected]
            fit = local_fits[peak.reflection.label]
            local_axes = previews[1]._local_axes[bank]
            np.testing.assert_allclose(local_axes[1].lines[1].get_ydata(), fit.fit_y)
            np.testing.assert_allclose(local_axes[2].lines[0].get_ydata(), fit.observed_y - fit.fit_y)
            linear_page = previews[2].tabs.widget(bank - 1)
            linear_axis = linear_page.layout().itemAt(0).widget().figure.axes[0]
            record = dialog.bundle.bank(bank)
            self.assertTrue(np.isfinite(record.difa))
            self.assertIsNotNone(record.difa_uncertainty)
            self.assertGreater(record.difa_uncertainty, 0.0)
            for peak in record.peak_diagnostics:
                if peak["fitted_tof_us"] is not None:
                    predicted = (record.difc * peak["d_spacing_angstrom"]
                                 + record.difa * peak["d_spacing_angstrom"]**2 + record.tzero)
                    self.assertAlmostEqual(peak["residual_tof_us"], peak["fitted_tof_us"] - predicted, places=5)
            fitted_line = next(line for line in linear_axis.lines if line.get_label() == "Weighted quadratic fit")
            np.testing.assert_allclose(
                fitted_line.get_ydata(), record.difc * fitted_line.get_xdata()
                + record.difa * fitted_line.get_xdata()**2 + record.tzero
            )
        sample_path = Path.home() / "Downloads" / "ENGINX00378089.nxs"
        if sample_path.is_file():
            with _temporary_path(".json") as calibration_path:
                save_raw_calibration_bundle(dialog.bundle, calibration_path)
                reloaded = load_raw_calibration_bundle(calibration_path)
            processed = analyze_sample_nexus(sample_path, reloaded)
            self.assertEqual(len(processed), 2)
            for spectrum in processed:
                self.assertEqual(spectrum.calibration.difa, reloaded.bank(spectrum.bank_number).difa)
        self.assertEqual(bundle_fingerprint(dialog.bundle), fingerprint)
        self.assertIsNone(tab.active_raw_calibration)
        for preview in previews:
            preview.close()
        dialog.close()
        tab.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
