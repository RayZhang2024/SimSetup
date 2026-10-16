from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QVBoxLayout,
)

from .raw_previews import Ceo2FitPreview, OpenBeamPreview, QuadraticCalibrationPreview
from .raw_workflow import RawCalibrationBundle, create_raw_calibration_bundle, save_raw_calibration_bundle


def _parse_mask(text: str) -> set[int]:
    values: set[int] = set()
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


def bundle_details(bundle: RawCalibrationBundle, filename: Optional[str] = None) -> str:
    legacy = all(record.calibration_model.startswith("linear") for record in bundle.banks)
    lines = [
        f"Active calibration: {filename or '(not yet saved)'}",
        f"Status: {bundle.scientific_status}",
        f"Processing convention: {bundle.processing_version}",
        "Accepted model: TOF = DIFC * d + TZERO (DIFA = 0)" if legacy
        else "Accepted model: TOF = DIFC * d + DIFA * d^2 + TZERO",
        "Reference response: charge-normalized histogram bin integrals (counts/uAh)",
    ]
    for record in bundle.banks:
        valid = np.asarray(record.response_valid_mask, dtype=bool)
        coverage = np.asarray(record.response_coverage_fraction, dtype=float)
        lines.extend([
            "",
            f"Bank {record.bank_number}",
            f"  Calibration ({record.calibration_model}): DIFC={record.difc:.8g} {record.coefficient_units.get('difc')}, "
            f"DIFA={record.difa:.8g} {record.coefficient_units.get('difa')}, "
            f"TZERO={record.tzero:.8g} {record.coefficient_units.get('tzero')}",
            f"  Uncertainties: DIFC={record.difc_uncertainty}, DIFA={record.difa_uncertainty}, "
            f"TZERO={record.tzero_uncertainty}",
            f"  RMS residual={record.rms_residual_tof:.6g} us",
            f"  Reference bins={len(record.response_values_per_uah)}, valid={int(valid.sum())}, "
            f"full coverage={int(np.count_nonzero(coverage >= 1.0 - 1e-9))}",
            f"  Fixed target geometric DIFC={record.focusing.get('target_geometric_difc')}",
            f"  Masked spectra={record.focusing.get('masked_spectrum_ids', [])}",
            f"  Individual-peak {'linear' if record.calibration_model.startswith('linear') else 'quadratic'} calibration diagnostics:",
        ])
        for peak in record.peak_diagnostics:
            state = "accepted" if peak.get("accepted") else f"rejected: {peak.get('rejection_reason')}"
            lines.append(
                f"    {peak.get('hkl')} d={peak.get('d_spacing_angstrom')} A, "
                f"TOF={peak.get('fitted_tof_us')} us, uncertainty={peak.get('tof_uncertainty_us')} us, "
                f"residual={peak.get('residual_tof_us')} us, {state}"
            )
    lines.extend(["", "Provenance:"])
    for role, provenance in bundle.source_runs.items():
        lines.append(f"  {role}: run {provenance.get('run')}, {provenance.get('file_identity', {})}")
    lines.extend(["", "Remaining scientific validation gaps:"])
    lines.extend(f"  - {gap}" for gap in bundle.remaining_validation_gaps)
    return "\n".join(lines)


class CreateCalibrationDialog(QDialog):
    def __init__(self, diffraction_tab) -> None:
        super().__init__(diffraction_tab)
        self.diffraction_tab = diffraction_tab
        self.setWindowTitle("Create raw ENGIN-X calibration")
        self.resize(860, 680)
        self.bundle: Optional[RawCalibrationBundle] = None
        self._run_diagnostics = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.reference_path = self._path_row(form, "Open-beam / V-Nb reference")
        self.ceo2_path = self._path_row(form, "CeO2 calibrant")
        self.bank1_mask = QLineEdit()
        self.bank1_mask.setPlaceholderText("Optional spectrum IDs, e.g. 15,18,21-24")
        form.addRow("Bank 1 pixel mask", self.bank1_mask)
        self.bank2_mask = QLineEdit()
        self.bank2_mask.setPlaceholderText("Optional spectrum IDs, e.g. 1215,1218,1221-1224")
        form.addRow("Bank 2 pixel mask", self.bank2_mask)
        self.smooth_reference = QCheckBox("Smooth reference response")
        form.addRow("Reference processing", self.smooth_reference)
        layout.addLayout(form)

        actions = QHBoxLayout()
        process = QPushButton("Process both banks")
        process.clicked.connect(self.process_banks)
        self.save_button = QPushButton("Save calibration JSON")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_bundle)
        actions.addWidget(process)
        actions.addWidget(self.save_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        previews = QHBoxLayout()
        self.open_beam_preview_button = QPushButton("Open-beam preview")
        self.ceo2_review_button = QPushButton("CeO2 fit review")
        self.quadratic_review_button = QPushButton("Quadratic calibration fit review")
        for button, preview_class in (
            (self.open_beam_preview_button, OpenBeamPreview),
            (self.ceo2_review_button, Ceo2FitPreview),
            (self.quadratic_review_button, QuadraticCalibrationPreview),
        ):
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, cls=preview_class: self._open_preview(cls))
            previews.addWidget(button)
        layout.addLayout(previews)

        self.status = QLabel(
            "Select only the reference and CeO2 NXS files. Both 1,200-spectrum diffraction banks are processed; "
            "transmission spectra are excluded."
        )
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ENGIN-X NXS run", str(Path.home()), "NeXus (*.nxs);;All files (*.*)"
        )
        if path:
            target.setText(path)

    def process_banks(self) -> None:
        self.bundle = None
        self._run_diagnostics = None
        self.save_button.setEnabled(False)
        for button in (self.open_beam_preview_button, self.ceo2_review_button, self.quadratic_review_button):
            button.setEnabled(False)
        try:
            reference = Path(self.reference_path.text().strip())
            ceo2 = Path(self.ceo2_path.text().strip())
            missing = [name for name, path in (("reference", reference), ("CeO2", ceo2)) if not path.is_file()]
            if missing:
                raise ValueError(f"Select existing NXS files for: {', '.join(missing)}.")
            if self.diffraction_tab.calibration_phase is None:
                raise ValueError("The CeO2 phase definition is unavailable.")
            masks = {1: _parse_mask(self.bank1_mask.text()), 2: _parse_mask(self.bank2_mask.text())}
            self.status.setText("Processing both diffraction banks...")
            self.bundle, self._run_diagnostics = create_raw_calibration_bundle(
                reference, ceo2, self.diffraction_tab.calibration_phase,
                masked_spectrum_ids=masks,
                smooth_reference=self.smooth_reference.isChecked(),
            )
            self.diagnostics.setPlainText(bundle_details(self.bundle))
            self.save_button.setEnabled(True)
            for button in (self.open_beam_preview_button, self.ceo2_review_button, self.quadratic_review_button):
                button.setEnabled(True)
            self.status.setText("Both banks processed. Review the diagnostics, then save the self-contained JSON.")
        except Exception as exc:
            self.diagnostics.clear()
            self.status.setText(f"Calibration creation failed: {exc}")

    def _open_preview(self, preview_class) -> None:
        if self.bundle is None or self._run_diagnostics is None:
            return
        preview = preview_class(self, self.bundle, self._run_diagnostics)
        preview.exec_()

    def save_bundle(self) -> None:
        if self.bundle is None:
            self.status.setText("Process both banks before saving.")
            return
        reference_run = str(self.bundle.source_runs["reference"]["run"]).zfill(6)
        ceo2_run = str(self.bundle.source_runs["ceo2"]["run"]).zfill(6)
        default_name = f"ENGINX_{reference_run}_{ceo2_run}_calibration.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save raw ENGIN-X calibration", default_name, "JSON (*.json)"
        )
        if path:
            try:
                save_raw_calibration_bundle(self.bundle, path)
                self.diffraction_tab.activate_raw_calibration(self.bundle, Path(path))
                self.status.setText(f"Saved and activated calibration: {path}")
                self.diagnostics.setPlainText(bundle_details(self.bundle, Path(path).name))
            except Exception as exc:
                self.status.setText(f"Could not save calibration: {exc}")


class CalibrationDetailsDialog(QDialog):
    def __init__(self, parent, bundle: RawCalibrationBundle, path: Optional[Path]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration details")
        self.resize(820, 650)
        layout = QVBoxLayout(self)
        details = QTextEdit()
        details.setReadOnly(True)
        details.setPlainText(bundle_details(bundle, path.name if path else None))
        layout.addWidget(details)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


RawNexusCalibrationDialog = CreateCalibrationDialog
