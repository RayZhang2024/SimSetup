from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from .calibration import refine_raw_ceo2_bank_calibration
from .focusing import BankFocusingDiagnostics, FOCUSING_CONVENTION_VERSION, focus_enginx_bank
from .models import CalibrationResult, FocusedSpectrum
from .nexus import RawEnginXNexusRun, read_enginx_histogram_nexus
from .normalization import apply_charge_reference_normalization
from .raw_workflow import (
    RawCalibrationBundle,
    apply_bundle_calibration,
    bank_record_from_result,
    file_identity,
    load_raw_calibration_bundle,
    save_raw_calibration_bundle,
)


class RawNexusCalibrationDialog(QDialog):
    def __init__(self, diffraction_tab) -> None:
        super().__init__(diffraction_tab)
        self.diffraction_tab = diffraction_tab
        self.setWindowTitle("Raw ENGIN-X NXS calibration")
        self.resize(820, 650)
        self._focused: dict[str, FocusedSpectrum] = {}
        self._diagnostics: dict[str, BankFocusingDiagnostics] = {}
        self._calibration_result: Optional[CalibrationResult] = None
        self._bundle: Optional[RawCalibrationBundle] = None
        self._bundle_loaded_from_disk = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.reference_path = self._path_row(form, "V-Nb reference")
        self.ceo2_path = self._path_row(form, "CeO2 calibrant")
        self.sample_path = self._path_row(form, "Sample")
        self.bank_combo = QComboBox()
        self.bank_combo.addItem("Bank 1 - North/+90 degree", 1)
        self.bank_combo.addItem("Bank 2 - South/-90 degree", 2)
        form.addRow("Diffraction bank", self.bank_combo)
        self.smooth_reference = QCheckBox("Smooth V-Nb response (explicit; may suppress real structure)")
        self.smooth_reference.setChecked(False)
        form.addRow("Reference response", self.smooth_reference)
        self.mask_edit = QLineEdit()
        self.mask_edit.setPlaceholderText("Optional shared spectrum IDs, e.g. 15,18,21-24")
        form.addRow("Pixel mask", self.mask_edit)
        layout.addLayout(form)

        actions = QHBoxLayout()
        process = QPushButton("Process and fit selected bank")
        process.clicked.connect(self.process_selected_bank)
        process_loaded = QPushButton("Process with loaded calibration")
        process_loaded.clicked.connect(self.process_with_loaded_calibration)
        apply_button = QPushButton("Apply accepted calibration")
        apply_button.clicked.connect(self.apply_selected_bank)
        save_button = QPushButton("Save calibration")
        save_button.clicked.connect(self.save_bundle)
        load_button = QPushButton("Load calibration")
        load_button.clicked.connect(self.load_bundle)
        for button in (process, process_loaded, apply_button, save_button, load_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.status = QLabel("Select the three NXS roles. Processing is bank-specific and excludes transmission spectra.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.diagnostics = QTextEdit()
        self.diagnostics.setReadOnly(True)
        layout.addWidget(self.diagnostics, 1)

    def _path_row(self, form: QFormLayout, label: str) -> QLineEdit:
        edit = QLineEdit()
        button = QPushButton("Browse")
        button.clicked.connect(lambda _checked=False, target=edit: self._browse_nxs(target))
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(button)
        form.addRow(label, row)
        return edit

    def _browse_nxs(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select ENGIN-X NXS run", str(Path.home()), "NeXus (*.nxs);;All files (*.*)")
        if path:
            target.setText(path)

    def _paths(self) -> dict[str, Path]:
        values = {
            "reference": Path(self.reference_path.text().strip()),
            "ceo2": Path(self.ceo2_path.text().strip()),
            "sample": Path(self.sample_path.text().strip()),
        }
        missing = [role for role, path in values.items() if not path.is_file()]
        if missing:
            raise ValueError(f"Select existing NXS files for: {', '.join(missing)}.")
        return values

    def _masked_spectrum_ids(self) -> set[int]:
        values: set[int] = set()
        text = self.mask_edit.text().strip()
        if not text:
            return values
        for item in text.split(","):
            token = item.strip()
            if not token:
                continue
            if "-" in token:
                first_text, last_text = token.split("-", 1)
                first, last = int(first_text), int(last_text)
                if last < first:
                    raise ValueError(f"Invalid descending mask range: {token}.")
                values.update(range(first, last + 1))
            else:
                values.add(int(token))
        return values

    def process_selected_bank(self) -> None:
        try:
            paths = self._paths()
            bank = int(self.bank_combo.currentData())
            masked_ids = self._masked_spectrum_ids()
            runs: dict[str, RawEnginXNexusRun] = {role: read_enginx_histogram_nexus(path) for role, path in paths.items()}
            common_edges = runs["ceo2"].tof_edges_us
            self._focused = {}
            self._diagnostics = {}
            for role, run in runs.items():
                self._focused[role], self._diagnostics[role] = focus_enginx_bank(
                    run, bank, masked_spectrum_ids=masked_ids, target_edges_us=common_edges
                )
            smooth = self.smooth_reference.isChecked()
            ceo2 = apply_charge_reference_normalization(
                self._focused["ceo2"], self._focused["reference"], smooth_response=smooth
            ).corrected_spectrum
            self._focused["sample_corrected"] = apply_charge_reference_normalization(
                self._focused["sample"], self._focused["reference"], smooth_response=smooth
            ).corrected_spectrum
            if self.diffraction_tab.calibration_phase is None:
                raise ValueError("The CeO2 phase definition is unavailable.")
            self._calibration_result = refine_raw_ceo2_bank_calibration(ceo2, self.diffraction_tab.calibration_phase)
            record = bank_record_from_result(
                ceo2,
                self._calibration_result,
                reference_response_settings={"smooth": smooth, "source_run": runs["reference"].run_number},
            )
            retained_records = ()
            if self._bundle is not None and not self._bundle_loaded_from_disk:
                retained_records = tuple(item for item in self._bundle.banks if item.bank_number != bank)
            self._bundle = RawCalibrationBundle(
                processing_version=FOCUSING_CONVENTION_VERSION,
                source_runs={
                    role: {
                        "run": run.run_number,
                        "title": run.title,
                        "charge_uah": run.proton_charge_uah,
                        "file_identity": file_identity(paths[role]),
                    }
                    for role, run in runs.items()
                },
                ceo2_standard={
                    "name": self.diffraction_tab.calibration_phase.name,
                    "lattice_parameter_angstrom": self.diffraction_tab.calibration_phase.a,
                    "source": "Hist/CEO2.EXP",
                },
                scientific_status="provisional: internally fitted, not independently absolute-validated",
                remaining_validation_gaps=(
                    "No independently trusted focused ENGIN-X reference was available for absolute validation.",
                    "The 50 m primary flight path is a nominal instrument-profile value; NXS source_detector_distance is zero.",
                    "V-Nb reference structure and detector-efficiency separation remain unvalidated.",
                ),
                banks=tuple(sorted((*retained_records, record), key=lambda item: item.bank_number)),
            )
            self._bundle_loaded_from_disk = False
            self._show_diagnostics()
            self.status.setText("Bank processing and provisional linear CeO2 calibration complete. Review diagnostics, then apply explicitly.")
        except Exception as exc:
            self.status.setText(f"Raw NXS processing failed: {exc}")

    def process_with_loaded_calibration(self) -> None:
        try:
            if self._bundle is None or not self._bundle_loaded_from_disk:
                raise ValueError("Load a saved raw calibration first.")
            if self._bundle.processing_version != FOCUSING_CONVENTION_VERSION:
                raise ValueError(
                    f"Saved processing version {self._bundle.processing_version!r} is not supported by this build."
                )
            paths = {
                "reference": Path(self.reference_path.text().strip()),
                "sample": Path(self.sample_path.text().strip()),
            }
            missing = [role for role, path in paths.items() if not path.is_file()]
            if missing:
                raise ValueError(f"Select existing NXS files for: {', '.join(missing)}.")
            bank = int(self.bank_combo.currentData())
            record = self._bundle.bank(bank)
            masked_ids = {int(value) for value in (record.focusing.get("masked_spectrum_ids") or [])}
            self.mask_edit.setText(",".join(str(value) for value in sorted(masked_ids)))
            smooth = bool(record.reference_response_settings.get("smooth", False))
            self.smooth_reference.setChecked(smooth)
            expected_reference = self._bundle.source_runs.get("reference", {}).get("file_identity", {})
            actual_reference = file_identity(paths["reference"])
            if expected_reference.get("sha256") and actual_reference["sha256"] != expected_reference["sha256"]:
                raise ValueError("Selected reference file does not match the saved calibration reference identity.")
            reference_run = read_enginx_histogram_nexus(paths["reference"])
            sample_run = read_enginx_histogram_nexus(paths["sample"])
            target_edges = sample_run.tof_edges_us
            reference, reference_diag = focus_enginx_bank(
                reference_run, bank, masked_spectrum_ids=masked_ids, target_edges_us=target_edges
            )
            sample, sample_diag = focus_enginx_bank(
                sample_run, bank, masked_spectrum_ids=masked_ids, target_edges_us=target_edges
            )
            self._focused = {"reference": reference, "sample": sample}
            self._diagnostics = {"reference": reference_diag, "sample": sample_diag}
            self._focused["sample_corrected"] = apply_charge_reference_normalization(
                sample,
                reference,
                smooth_response=smooth,
            ).corrected_spectrum
            self.diagnostics.setPlainText(
                f"Loaded calibration bank {bank}.\n"
                f"Reference run {reference.run_number}: Q={reference.metadata['proton_charge']} uAh.\n"
                f"Sample run {sample.run_number}: Q={sample.metadata['proton_charge']} uAh.\n"
                f"Focused {sample_diag.input_spectra} sample spectra; transmission used={sample_diag.transmission_spectra_used}.\n"
                "Calibration has not yet been applied; use Apply accepted calibration."
            )
            self.status.setText("Focused and reference-corrected the sample with the loaded processing convention; apply explicitly when ready.")
        except Exception as exc:
            self.status.setText(f"Could not process with loaded calibration: {exc}")

    def _show_diagnostics(self) -> None:
        if self._calibration_result is None:
            return
        result = self._calibration_result
        lines = [
            "Convention: TOF (microsecond) = DIFC (microsecond/angstrom) * d (angstrom) + TZERO (microsecond)",
            "Geometry is used for relative pixel alignment only; CeO2 determines the accepted bank calibration.",
            f"Bank {result.bank_number}: DIFC={result.calibration.difc:.8g}, TZERO={result.calibration.tzero:.8g}, DIFA=0",
            f"Accepted linear-fit RMS={result.rms_residual_tof:.5g} microsecond",
        ]
        for role, diag in self._diagnostics.items():
            lines.append(
                f"{role}: {diag.input_spectra} diffraction spectra, transmission used={diag.transmission_spectra_used}, "
                f"geometry DIFC={diag.geometric_difc_min:.3f}..{diag.geometric_difc_max:.3f}, "
                f"supported counts={diag.supported_input_counts:.6g}"
            )
        corrected_sample = self._focused.get("sample_corrected")
        if corrected_sample is not None:
            response = np.asarray(corrected_sample.metadata.get("normalization_response", []), dtype=float)
            finite_response = response[np.isfinite(response) & (response > 0.0)]
            if finite_response.size:
                lines.append(
                    f"Applied V-Nb response ({'smoothed' if corrected_sample.metadata.get('normalization_smoothed') else 'unsmoothed'}): "
                    f"min={finite_response.min():.6g}, median={np.median(finite_response):.6g}, max={finite_response.max():.6g}; "
                    f"valid bins={corrected_sample.metadata.get('normalization_valid_bins')}"
                )
        lines.append("\nCeO2 peaks:")
        for peak in result.peak_results:
            state = "accepted" if peak.accepted else f"rejected: {peak.rejection_reason}"
            lines.append(
                f"{peak.reflection.label:>4} d={peak.reflection.d_spacing:.7g} fitted={peak.fitted_tof} "
                f"unc={peak.tof_uncertainty} residual={peak.residual_tof} {state}"
            )
        lines.append("\nScientific status: provisional; fit residuals do not establish independent absolute accuracy.")
        self.diagnostics.setPlainText("\n".join(lines))

    def apply_selected_bank(self) -> None:
        try:
            if self._bundle is None or "sample_corrected" not in self._focused:
                raise ValueError("Process a bank and review diagnostics, or load a calibration and process a sample first.")
            calibrated = apply_bundle_calibration(self._focused["sample_corrected"], self._bundle)
            self.diffraction_tab.add_processed_spectrum(calibrated)
            self.status.setText(f"Applied accepted calibration once and added run {calibrated.run_number} bank {calibrated.bank_number} to Diffraction.")
        except Exception as exc:
            self.status.setText(f"Could not apply calibration: {exc}")

    def save_bundle(self) -> None:
        if self._bundle is None:
            self.status.setText("Process and fit a bank before saving calibration.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save raw ENGIN-X calibration", "enginx_raw_calibration.json", "JSON (*.json)")
        if path:
            save_raw_calibration_bundle(self._bundle, path)
            self.status.setText(f"Saved calibration to {path}.")

    def load_bundle(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load raw ENGIN-X calibration", str(Path.cwd()), "JSON (*.json)")
        if path:
            try:
                self._bundle = load_raw_calibration_bundle(path)
                self._bundle_loaded_from_disk = True
                banks = ", ".join(str(record.bank_number) for record in self._bundle.banks)
                self.status.setText(f"Loaded provisional raw calibration for bank(s) {banks}. Process matching raw runs before applying.")
            except Exception as exc:
                self.status.setText(f"Could not load calibration: {exc}")
