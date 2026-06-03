from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidgetAction,
    QWidget,
)

from .calibration import OPENGENIE_CALIBRATION_TOF_MAX, OPENGENIE_CALIBRATION_TOF_MIN, refine_ceo2_calibration
from .fitting import fit_peak_profile
from .gsas_exp import parse_gsas_exp
from .importers import load_focused_spectrum
from .models import (
    AXIS_D_SPACING,
    AXIS_TOF,
    FIT_SCOPE_PATTERN,
    FIT_SCOPE_PEAK,
    PAWLEY_WIDTH_CONSTANT,
    PAWLEY_WIDTH_D_RESOLUTION,
    CalibrationResult,
    FittingSettings,
    FocusedSpectrum,
    NormalizationResult,
    PhaseModel,
    d_to_tof,
    spectrum_with_calibration,
    tof_to_d,
)
from .normalization import apply_vanadium_normalization
from .pawley import fit_pawley
from .profiles import PEAK_PROFILE_SPECS, PROFILE_EXP_VOIGT, PROFILE_GSAS_TOF, is_gsas_tof_profile

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib.widgets import SpanSelector
except Exception:  # pragma: no cover - exercised only when optional dependency is absent
    FigureCanvas = None
    NavigationToolbar = None
    Figure = None
    SpanSelector = None


RESULT_HEADERS = [
    "Peak function",
    "Scope / method",
    "Source",
    "Run",
    "Bank",
    "Axis",
    "Range min",
    "Range max",
    "Lattice a",
    "Lattice a unc",
    "Chi2",
    "Rwp %",
]
PEAK_RESULT_FIELD_HEADERS = ("d", "d unc", "FWHM d", "FWHM d unc", "Height", "Area", "Eta")


def _populate_fit_scope_combo(combo: QComboBox) -> None:
    combo.addItem("Individual peak fitting", FIT_SCOPE_PEAK)
    combo.addItem("Whole pattern fitting", FIT_SCOPE_PATTERN)


def _populate_profile_combo(combo: QComboBox) -> None:
    for profile in PEAK_PROFILE_SPECS:
        combo.addItem(profile.label, profile.key)


def _populate_pawley_width_model_combo(combo: QComboBox) -> None:
    combo.addItem("Constant FWHM", PAWLEY_WIDTH_CONSTANT)
    combo.addItem("d-dependent resolution", PAWLEY_WIDTH_D_RESOLUTION)


class CalibrationDialog(QDialog):
    def __init__(self, diffraction_tab: "DiffractionTab") -> None:
        super().__init__(diffraction_tab)
        self.diffraction_tab = diffraction_tab
        self.setWindowTitle("Calibration")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        form = QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

        self.open_beam_button = QPushButton("Load open beam")
        self.open_beam_button.clicked.connect(self.load_open_beam)
        self.open_beam_status = QLineEdit()
        self.open_beam_status.setReadOnly(True)

        self.ceo2_button = QPushButton("Load CeO2")
        self.ceo2_button.clicked.connect(self.load_ceo2)
        self.ceo2_status = QLineEdit()
        self.ceo2_status.setReadOnly(True)

        self.ceo2_phase_status = QLineEdit()
        self.ceo2_phase_status.setReadOnly(True)
        self.calibration_status = QLineEdit()
        self.calibration_status.setReadOnly(True)
        self.correction_status = QLineEdit()
        self.correction_status.setReadOnly(True)
        self.ceo2_run_status = QLineEdit()
        self.ceo2_run_status.setReadOnly(True)
        self.vanadium_run_status = QLineEdit()
        self.vanadium_run_status.setReadOnly(True)
        self.difc_status = QLineEdit()
        self.difc_status.setReadOnly(True)
        self.tzero_status = QLineEdit()
        self.tzero_status.setReadOnly(True)
        self.difa_status = QLineEdit()
        self.difa_status.setReadOnly(True)
        self.single_difc_status = QLineEdit()
        self.single_difc_status.setReadOnly(True)
        self.single_tzero_status = QLineEdit()
        self.single_tzero_status.setReadOnly(True)
        self.pattern_status = QLineEdit()
        self.pattern_status.setReadOnly(True)

        form.addWidget(self.open_beam_button, 0, 0)
        form.addWidget(self.open_beam_status, 0, 1)
        form.addWidget(self.ceo2_button, 1, 0)
        form.addWidget(self.ceo2_status, 1, 1)
        form.addWidget(QLabel("CeO2 phase"), 2, 0)
        form.addWidget(self.ceo2_phase_status, 2, 1)
        form.addWidget(QLabel("Status"), 3, 0)
        form.addWidget(self.calibration_status, 3, 1)
        form.addWidget(QLabel("Current correction"), 4, 0)
        form.addWidget(self.correction_status, 4, 1)
        form.addWidget(QLabel("CeO2 run"), 5, 0)
        form.addWidget(self.ceo2_run_status, 5, 1)
        form.addWidget(QLabel("Vanadium run"), 6, 0)
        form.addWidget(self.vanadium_run_status, 6, 1)
        form.addWidget(QLabel("DIFC"), 7, 0)
        form.addWidget(self.difc_status, 7, 1)
        form.addWidget(QLabel("TZERO"), 8, 0)
        form.addWidget(self.tzero_status, 8, 1)
        form.addWidget(QLabel("DIFA"), 9, 0)
        form.addWidget(self.difa_status, 9, 1)
        form.addWidget(QLabel("Single peak DIFC1"), 10, 0)
        form.addWidget(self.single_difc_status, 10, 1)
        form.addWidget(QLabel("Single peak ZERO1"), 11, 0)
        form.addWidget(self.single_tzero_status, 11, 1)
        form.addWidget(QLabel("Pattern fit"), 12, 0)
        form.addWidget(self.pattern_status, 12, 1)
        form.setColumnStretch(1, 1)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run calibration + normalization")
        self.run_button.clicked.connect(self.run_calibration)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        buttons.addWidget(self.run_button)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.refresh()

    def load_open_beam(self) -> None:
        self.diffraction_tab.load_vanadium_spectrum_dialog()
        self.refresh()

    def load_ceo2(self) -> None:
        self.diffraction_tab.load_calibration_spectrum_dialog()
        self.refresh()

    def run_calibration(self) -> None:
        self.diffraction_tab.run_calibration_sequence()
        self.refresh()

    def refresh(self) -> None:
        tab = self.diffraction_tab
        self.open_beam_status.setText(tab._spectrum_summary(tab.vanadium_spectrum))
        self.ceo2_status.setText(tab._spectrum_summary(tab.calibration_spectrum))
        self.ceo2_phase_status.setText(tab._phase_summary(tab.calibration_phase))
        self.calibration_status.setText(tab._calibration_status_text())
        spectrum = tab.current_spectrum()
        self.correction_status.setText(tab._correction_summary(spectrum))
        self.ceo2_run_status.setText(tab._calibration_run_text(spectrum))
        self.vanadium_run_status.setText(tab._vanadium_run_text(spectrum))
        result = tab.calibration_result
        calibration = result.calibration if result is not None else None if spectrum is None else spectrum.calibration
        self.difc_status.setText("" if calibration is None else f"{calibration.difc:.8g}")
        self.tzero_status.setText("" if calibration is None else f"{calibration.tzero:.8g}")
        self.difa_status.setText("" if calibration is None else f"{calibration.difa:.8g}")
        single = None if result is None else result.single_peak_calibration
        self.single_difc_status.setText("" if single is None else f"{single.difc:.8g}")
        self.single_tzero_status.setText("" if single is None else f"{single.tzero:.8g}")
        if result is not None and result.pattern_fit is not None:
            self.pattern_status.setText(
                f"{result.pattern_fit.profile_name}, Rwp={result.pattern_fit.quality.rwp_percent:.3g}%, "
                f"RMS={result.rms_residual_tof:.4g} us"
            )
        else:
            self.pattern_status.setText("")


class FittingSettingsDialog(QDialog):
    def __init__(self, diffraction_tab: "DiffractionTab") -> None:
        super().__init__(diffraction_tab)
        self.diffraction_tab = diffraction_tab
        self.setWindowTitle("Fitting Settings")
        self.setModal(True)

        settings = diffraction_tab.fitting_settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        model_form = QFormLayout()
        self.fit_scope_combo = QComboBox()
        _populate_fit_scope_combo(self.fit_scope_combo)
        self._set_combo_data(self.fit_scope_combo, settings.fit_scope)
        self.peak_profile_combo = QComboBox()
        _populate_profile_combo(self.peak_profile_combo)
        self._set_combo_data(self.peak_profile_combo, settings.peak_profile_key)
        self.pattern_profile_combo = QComboBox()
        _populate_profile_combo(self.pattern_profile_combo)
        self._set_combo_data(self.pattern_profile_combo, settings.pattern_profile_key)
        self.background_order_spin = QSpinBox()
        self.background_order_spin.setRange(0, 5)
        self.background_order_spin.setValue(settings.polynomial_order)
        self.use_uncertainties_check = QCheckBox()
        self.use_uncertainties_check.setChecked(settings.use_uncertainties)
        self.max_evaluations_spin = QSpinBox()
        self.max_evaluations_spin.setRange(100, 1_000_000)
        self.max_evaluations_spin.setSingleStep(1000)
        self.max_evaluations_spin.setValue(settings.max_evaluations)
        model_form.addRow("Fit scope", self.fit_scope_combo)
        model_form.addRow("Single peak function", self.peak_profile_combo)
        model_form.addRow("Whole pattern function", self.pattern_profile_combo)
        model_form.addRow("Background order", self.background_order_spin)
        model_form.addRow("Use uncertainties", self.use_uncertainties_check)
        model_form.addRow("Max evaluations", self.max_evaluations_spin)
        layout.addLayout(model_form)

        range_form = QGridLayout()
        self.range_min_edit = QLineEdit(diffraction_tab.range_min_edit.text())
        self.range_max_edit = QLineEdit(diffraction_tab.range_max_edit.text())
        range_form.addWidget(QLabel("Range min"), 0, 0)
        range_form.addWidget(self.range_min_edit, 0, 1)
        range_form.addWidget(QLabel("Range max"), 1, 0)
        range_form.addWidget(self.range_max_edit, 1, 1)
        self.use_view_button = QPushButton("Use current view")
        self.use_view_button.clicked.connect(self.use_current_view_range)
        self.clear_range_button = QPushButton("Clear range")
        self.clear_range_button.clicked.connect(self.clear_range)
        range_buttons = QHBoxLayout()
        range_buttons.addWidget(self.use_view_button)
        range_buttons.addWidget(self.clear_range_button)
        range_buttons.addStretch(1)
        range_form.addLayout(range_buttons, 2, 0, 1, 2)
        layout.addLayout(range_form)

        pseudo_form = QFormLayout()
        self.pv_eta_initial_spin = self._make_double_spin(0.0, 1.0, 0.01, settings.pseudo_voigt_eta_initial, 3)
        self.pv_eta_min_spin = self._make_double_spin(0.0, 1.0, 0.01, settings.pseudo_voigt_eta_min, 3)
        self.pv_eta_max_spin = self._make_double_spin(0.0, 1.0, 0.01, settings.pseudo_voigt_eta_max, 3)
        self.pv_fwhm_min_fraction_spin = self._make_double_spin(0.000001, 0.1, 0.0001, settings.pseudo_voigt_fwhm_min_fraction, 6)
        self.pv_fwhm_max_multiplier_spin = self._make_double_spin(0.01, 20.0, 0.1, settings.pseudo_voigt_fwhm_max_multiplier, 3)
        pseudo_form.addRow("Peak eta initial", self.pv_eta_initial_spin)
        pseudo_form.addRow("Peak eta min", self.pv_eta_min_spin)
        pseudo_form.addRow("Peak eta max", self.pv_eta_max_spin)
        pseudo_form.addRow("Peak FWHM min fraction", self.pv_fwhm_min_fraction_spin)
        pseudo_form.addRow("Peak FWHM max multiplier", self.pv_fwhm_max_multiplier_spin)
        layout.addLayout(pseudo_form)

        pawley_form = QFormLayout()
        self.pawley_width_model_combo = QComboBox()
        _populate_pawley_width_model_combo(self.pawley_width_model_combo)
        self._set_combo_data(self.pawley_width_model_combo, settings.pawley_width_model)
        self.pawley_lattice_tolerance_spin = self._make_double_spin(0.0001, 20.0, 0.1, settings.pawley_lattice_tolerance_percent, 4)
        self.pawley_reflection_margin_spin = self._make_double_spin(0.0, 50.0, 0.5, settings.pawley_reflection_margin_percent, 3)
        self.pawley_eta_initial_spin = self._make_double_spin(0.0, 1.0, 0.01, settings.pawley_eta_initial, 3)
        self.pawley_eta_min_spin = self._make_double_spin(0.0, 1.0, 0.01, settings.pawley_eta_min, 3)
        self.pawley_eta_max_spin = self._make_double_spin(0.0, 1.0, 0.01, settings.pawley_eta_max, 3)
        self.pawley_fwhm_min_fraction_spin = self._make_double_spin(0.000001, 0.1, 0.0001, settings.pawley_fwhm_min_fraction, 6)
        self.pawley_fwhm_max_fraction_spin = self._make_double_spin(0.001, 2.0, 0.01, settings.pawley_fwhm_max_fraction, 4)
        self.pawley_intensity_model = QLineEdit("Independent per HKL")
        self.pawley_intensity_model.setReadOnly(True)
        pawley_form.addRow("Pawley width model", self.pawley_width_model_combo)
        pawley_form.addRow("Pawley lattice tolerance %", self.pawley_lattice_tolerance_spin)
        pawley_form.addRow("Pawley reflection margin %", self.pawley_reflection_margin_spin)
        pawley_form.addRow("Pawley eta initial", self.pawley_eta_initial_spin)
        pawley_form.addRow("Pawley eta min", self.pawley_eta_min_spin)
        pawley_form.addRow("Pawley eta max", self.pawley_eta_max_spin)
        pawley_form.addRow("Pawley FWHM min fraction", self.pawley_fwhm_min_fraction_spin)
        pawley_form.addRow("Pawley FWHM max fraction", self.pawley_fwhm_max_fraction_spin)
        pawley_form.addRow("Pawley intensity model", self.pawley_intensity_model)
        layout.addLayout(pawley_form)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        self.button_box.accepted.connect(self.accept_settings)
        self.button_box.rejected.connect(self.reject)
        apply_button = self.button_box.button(QDialogButtonBox.Apply)
        if apply_button is not None:
            apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.button_box)

    def _make_double_spin(self, minimum: float, maximum: float, step: float, value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _set_combo_data(self, combo: QComboBox, data: str) -> None:
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)

    def use_current_view_range(self) -> None:
        spectrum = self.diffraction_tab.current_spectrum()
        if spectrum is None:
            return
        try:
            x_values, _label = self.diffraction_tab._axis_data(spectrum)
            view_range = self.diffraction_tab._current_view_range_or_none(x_values, spectrum)
        except Exception:
            return
        if view_range is None:
            return
        self.range_min_edit.setText(f"{view_range[0]:.6g}")
        self.range_max_edit.setText(f"{view_range[1]:.6g}")

    def clear_range(self) -> None:
        self.range_min_edit.clear()
        self.range_max_edit.clear()

    def _settings_from_controls(self) -> FittingSettings:
        return FittingSettings(
            fit_scope=str(self.fit_scope_combo.currentData() or FIT_SCOPE_PEAK),
            peak_profile_key=str(self.peak_profile_combo.currentData() or PROFILE_EXP_VOIGT),
            pattern_profile_key=str(self.pattern_profile_combo.currentData() or PROFILE_GSAS_TOF),
            polynomial_order=int(self.background_order_spin.value()),
            use_uncertainties=self.use_uncertainties_check.isChecked(),
            max_evaluations=int(self.max_evaluations_spin.value()),
            pseudo_voigt_eta_initial=float(self.pv_eta_initial_spin.value()),
            pseudo_voigt_eta_min=float(self.pv_eta_min_spin.value()),
            pseudo_voigt_eta_max=float(self.pv_eta_max_spin.value()),
            pseudo_voigt_fwhm_min_fraction=float(self.pv_fwhm_min_fraction_spin.value()),
            pseudo_voigt_fwhm_max_multiplier=float(self.pv_fwhm_max_multiplier_spin.value()),
            pawley_lattice_tolerance_percent=float(self.pawley_lattice_tolerance_spin.value()),
            pawley_reflection_margin_percent=float(self.pawley_reflection_margin_spin.value()),
            pawley_eta_initial=float(self.pawley_eta_initial_spin.value()),
            pawley_eta_min=float(self.pawley_eta_min_spin.value()),
            pawley_eta_max=float(self.pawley_eta_max_spin.value()),
            pawley_fwhm_min_fraction=float(self.pawley_fwhm_min_fraction_spin.value()),
            pawley_fwhm_max_fraction=float(self.pawley_fwhm_max_fraction_spin.value()),
            pawley_width_model=str(self.pawley_width_model_combo.currentData() or PAWLEY_WIDTH_CONSTANT),
        )

    def apply_settings(self) -> None:
        tab = self.diffraction_tab
        tab.fitting_settings = self._settings_from_controls()
        tab._apply_fitting_settings_to_controls()
        tab.range_min_edit.setText(self.range_min_edit.text().strip())
        tab.range_max_edit.setText(self.range_max_edit.text().strip())
        tab._set_status("Updated fitting settings.")
        tab.update_plot()

    def accept_settings(self) -> None:
        self.apply_settings()
        self.accept()


class DiffractionTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.spectra: list[FocusedSpectrum] = []
        self.corrected_spectra: list[Optional[FocusedSpectrum]] = []
        self.normalization_results: list[Optional[NormalizationResult]] = []
        self.phase: Optional[PhaseModel] = None
        self.calibration_phase: Optional[PhaseModel] = None
        self.calibration_spectrum: Optional[FocusedSpectrum] = None
        self.calibration_fit_spectrum: Optional[FocusedSpectrum] = None
        self.vanadium_spectrum: Optional[FocusedSpectrum] = None
        self.calibration_result: Optional[CalibrationResult] = None
        self.fitting_settings = FittingSettings()
        self.latest_fit_curve: Optional[dict[str, object]] = None
        self._span_selector = None
        self._build_ui()
        self._load_default_phase()
        self._load_default_calibration_phase()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        if FigureCanvas is None or Figure is None:
            label = QLabel("matplotlib is required for the Diffraction tab. Install requirements.txt and restart.")
            label.setWordWrap(True)
            layout.addWidget(label)
            return

        controls = QFrame(self)
        controls.setFrameShape(QFrame.StyledPanel)
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(380)
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)

        self.load_button = QPushButton("Load spectrum")
        self.load_button.clicked.connect(self.load_spectrum_dialog)
        self.calibration_button = QPushButton("Calibration")
        self.calibration_button.clicked.connect(self.open_calibration_dialog)
        self.calibration_status_edit = QLineEdit()
        self.calibration_status_edit.setReadOnly(True)
        self.phase_button = QPushButton("Load phase")
        self.phase_button.clicked.connect(self.load_phase_dialog)
        self.phase_status_edit = QLineEdit()
        self.phase_status_edit.setReadOnly(True)
        self.spectrum_combo = QComboBox()
        self.spectrum_combo.currentIndexChanged.connect(self.update_plot)
        self.axis_combo = QComboBox()
        self.axis_combo.addItem("TOF", AXIS_TOF)
        self.axis_combo.addItem("d-spacing", AXIS_D_SPACING)
        self.axis_combo.currentIndexChanged.connect(self.update_plot)
        self.data_combo = QComboBox()
        self.data_combo.addItem("Raw", "raw")
        self.data_combo.addItem("Corrected", "corrected")
        self.data_combo.currentIndexChanged.connect(self.update_plot)
        self.fit_scope_combo = QComboBox()
        _populate_fit_scope_combo(self.fit_scope_combo)
        self.fit_scope_combo.currentIndexChanged.connect(self._sync_fitting_settings_from_controls)
        self.poly_order = QSpinBox()
        self.poly_order.setRange(0, 5)
        self.poly_order.setValue(2)
        self.poly_order.valueChanged.connect(self._sync_fitting_settings_from_controls)
        self.range_min_edit = QLineEdit()
        self.range_max_edit = QLineEdit()
        self.view_min_edit = QLineEdit()
        self.view_max_edit = QLineEdit()
        self.view_min_edit.returnPressed.connect(self.update_plot)
        self.view_max_edit.returnPressed.connect(self.update_plot)
        self.y_view_min_edit = QLineEdit()
        self.y_view_max_edit = QLineEdit()
        self.y_view_min_edit.returnPressed.connect(self.update_plot)
        self.y_view_max_edit.returnPressed.connect(self.update_plot)
        self.fit_button = QPushButton("Fit selected range")
        self.fit_button.clicked.connect(self.fit_selected_range)
        self.export_button = QPushButton("Export results")
        self.export_button.clicked.connect(self.export_results_dialog)
        self._configure_view_limit_edits()
        self.menu_bar = self._build_menu_bar()

        controls_layout.addWidget(self.calibration_button, 0, 0)
        controls_layout.addWidget(self.calibration_status_edit, 0, 1)
        controls_layout.addWidget(self.phase_button, 1, 0)
        controls_layout.addWidget(self.phase_status_edit, 1, 1)
        controls_layout.addWidget(self.load_button, 2, 0, 1, 2)
        controls_layout.addWidget(QLabel("Spectrum"), 3, 0)
        controls_layout.addWidget(self.spectrum_combo, 3, 1)
        controls_layout.addWidget(QLabel("Axis"), 4, 0)
        controls_layout.addWidget(self.axis_combo, 4, 1)
        controls_layout.addWidget(QLabel("Data"), 5, 0)
        controls_layout.addWidget(self.data_combo, 5, 1)
        controls_layout.addWidget(QLabel("Fit scope"), 6, 0)
        controls_layout.addWidget(self.fit_scope_combo, 6, 1)
        controls_layout.addWidget(QLabel("Background order"), 7, 0)
        controls_layout.addWidget(self.poly_order, 7, 1)
        controls_layout.addWidget(QLabel("Range min"), 8, 0)
        controls_layout.addWidget(self.range_min_edit, 8, 1)
        controls_layout.addWidget(QLabel("Range max"), 9, 0)
        controls_layout.addWidget(self.range_max_edit, 9, 1)
        controls_layout.addWidget(self.fit_button, 10, 0, 1, 2)
        controls_layout.addWidget(self.export_button, 11, 0, 1, 2)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setRowStretch(12, 1)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.addWidget(controls)

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        grid_spec = self.figure.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.0)
        self.axes = self.figure.add_subplot(grid_spec[0])
        self.residual_axes = self.figure.add_subplot(grid_spec[1], sharex=self.axes)
        self.figure.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.11, hspace=0.0)
        plot_panel = QWidget(self)
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        self.toolbar = NavigationToolbar(self.canvas, plot_panel) if NavigationToolbar is not None else None
        if self.toolbar is not None:
            plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        top_splitter.addWidget(plot_panel)
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)
        splitter.addWidget(top_splitter)

        self._peak_result_column_count = 0
        self.results_table = QTableWidget(0, len(RESULT_HEADERS))
        self.results_table.setHorizontalHeaderLabels(RESULT_HEADERS)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setWordWrap(False)
        self.results_table.itemSelectionChanged.connect(self._update_selected_result_details)
        splitter.addWidget(self.results_table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        details_frame = QFrame(self)
        details_layout = QVBoxLayout(details_frame)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(3)
        details_label = QLabel("Result details")
        self.result_details_edit = QTextEdit()
        self.result_details_edit.setReadOnly(True)
        self.result_details_edit.setMinimumHeight(72)
        self.result_details_edit.setMaximumHeight(140)
        self.result_details_edit.setPlaceholderText("Select a result row to view fit details.")
        details_layout.addWidget(details_label)
        details_layout.addWidget(self.result_details_edit)
        layout.addWidget(details_frame)

        status_frame = QFrame(self)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Load a reduced focused histogram to begin.")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_frame)

    def _build_menu_bar(self) -> QMenuBar:
        menu_bar = QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        menu_bar.setStyleSheet(
            """
            QMenuBar {
                background-color: #f7f9fc;
                border: 1px solid #d7dee8;
                padding: 1px 4px;
            }
            QMenuBar::item {
                padding: 4px 10px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background-color: #e8f0fb;
            }
            """
        )

        file_menu = menu_bar.addMenu("File")
        self._add_menu_action(file_menu, "Load Spectrum", self.load_spectrum_dialog)
        self._add_menu_action(file_menu, "Load Phase", self.load_phase_dialog)
        file_menu.addSeparator()
        self._add_menu_action(file_menu, "Export Results", self.export_results_dialog)

        calibration_menu = menu_bar.addMenu("Calibration")
        self._add_menu_action(calibration_menu, "Open Calibration Dialog", self.open_calibration_dialog)
        self._add_menu_action(calibration_menu, "Run Calibration + Normalization", self.run_calibration_sequence)

        view_menu = menu_bar.addMenu("View")
        self._add_menu_action(view_menu, "Show Raw Data", lambda: self._set_combo_data(self.data_combo, "raw"))
        self._add_menu_action(view_menu, "Show Corrected Data", lambda: self._set_combo_data(self.data_combo, "corrected"))
        view_menu.addSeparator()
        self._add_menu_action(view_menu, "TOF Axis", lambda: self._set_combo_data(self.axis_combo, AXIS_TOF))
        self._add_menu_action(view_menu, "d-spacing Axis", lambda: self._set_combo_data(self.axis_combo, AXIS_D_SPACING))
        view_menu.addSeparator()
        self._add_view_limits_menu_widget(view_menu)
        view_menu.addSeparator()
        self._add_menu_action(view_menu, "Apply View", self.update_plot)
        self._add_menu_action(view_menu, "Reset View", self.reset_plot_view)

        fit_menu = menu_bar.addMenu("Fit")
        self._add_menu_action(fit_menu, "Fitting Settings...", self.open_fitting_settings_dialog)
        self._add_menu_action(fit_menu, "Fit Selected Range", self.fit_selected_range)

        about_menu = menu_bar.addMenu("About")
        self._add_menu_action(about_menu, "About Diffraction", self.show_about_dialog)
        return menu_bar

    def _add_menu_action(self, menu, label: str, slot) -> QAction:
        action = QAction(label, self)
        action.setStatusTip(label)
        action.triggered.connect(lambda _checked=False, target=slot: target())
        menu.addAction(action)
        return action

    def _configure_view_limit_edits(self) -> None:
        self.view_min_edit.setPlaceholderText("default 10000 TOF")
        self.view_max_edit.setPlaceholderText("auto")
        self.y_view_min_edit.setPlaceholderText("auto")
        self.y_view_max_edit.setPlaceholderText("auto")
        for edit in (self.view_min_edit, self.view_max_edit, self.y_view_min_edit, self.y_view_max_edit):
            edit.setMinimumWidth(120)

    def _add_view_limits_menu_widget(self, menu) -> None:
        panel = QWidget(menu)
        panel_layout = QGridLayout(panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setHorizontalSpacing(8)
        panel_layout.setVerticalSpacing(6)
        panel_layout.addWidget(QLabel("View x min"), 0, 0)
        panel_layout.addWidget(self.view_min_edit, 0, 1)
        panel_layout.addWidget(QLabel("View x max"), 1, 0)
        panel_layout.addWidget(self.view_max_edit, 1, 1)
        panel_layout.addWidget(QLabel("View y min"), 2, 0)
        panel_layout.addWidget(self.y_view_min_edit, 2, 1)
        panel_layout.addWidget(QLabel("View y max"), 3, 0)
        panel_layout.addWidget(self.y_view_max_edit, 3, 1)
        action = QWidgetAction(menu)
        action.setDefaultWidget(panel)
        menu.addAction(action)

    def _set_combo_data(self, combo: QComboBox, data: str) -> None:
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)

    def show_about_dialog(self) -> None:
        QMessageBox.information(
            self,
            "About Diffraction",
            "Use Diffraction to load focused histograms, calibrate with CeO2, apply open beam normalization, "
            "fit selected ranges, and export fit results.",
        )

    def _load_default_phase(self) -> None:
        default_path = Path.cwd() / "Hist" / "NI.EXP"
        if default_path.exists():
            try:
                self.phase = parse_gsas_exp(default_path)[0]
                self._set_status(f"Loaded default phase {self.phase.name} from {default_path.name}.")
            except Exception as exc:
                self._set_status(f"Default phase load failed: {exc}")
        self._refresh_status_fields()

    def _load_default_calibration_phase(self) -> None:
        for filename in ("CeO2.exp", "CEO2.EXP", "CeO2.EXP"):
            default_path = Path.cwd() / "Hist" / filename
            if default_path.exists():
                try:
                    self.calibration_phase = parse_gsas_exp(default_path)[0]
                    self._set_status(f"Loaded default calibration phase {self.calibration_phase.name} from {default_path.name}.")
                except Exception as exc:
                    self._set_status(f"Default CeO2 phase load failed: {exc}")
                return

    def _set_status(self, text: str) -> None:
        if hasattr(self, "status_label"):
            self.status_label.setText(text)

    def _phase_summary(self, phase: Optional[PhaseModel]) -> str:
        return phase.name if phase is not None else "No phase"

    def _spectrum_summary(self, spectrum: Optional[FocusedSpectrum]) -> str:
        if spectrum is None:
            return "Not loaded"
        run = f" run {spectrum.run_number}" if spectrum.run_number else ""
        bank = f" bank {spectrum.bank_number}" if spectrum.bank_number is not None else ""
        return f"{spectrum.source_path.name}{run}{bank}"

    def _calibration_status_text(self) -> str:
        return "Calibrated" if self.calibration_result is not None else "Not calibrated"

    def _correction_summary(self, spectrum: Optional[FocusedSpectrum]) -> str:
        if spectrum is None:
            return "No spectrum selected"
        metadata = spectrum.metadata
        if "normalization_run" in metadata:
            return "calibrated+normalised" if "calibration_run" in metadata else "normalised"
        if "calibration_run" in metadata:
            return "calibrated"
        return "raw"

    def _calibration_run_text(self, spectrum: Optional[FocusedSpectrum]) -> str:
        if spectrum is not None and spectrum.metadata.get("calibration_run") is not None:
            return str(spectrum.metadata.get("calibration_run"))
        if self.calibration_result is not None and self.calibration_result.run_number is not None:
            return str(self.calibration_result.run_number)
        if self.calibration_spectrum is not None and self.calibration_spectrum.run_number is not None:
            return str(self.calibration_spectrum.run_number)
        return ""

    def _vanadium_run_text(self, spectrum: Optional[FocusedSpectrum]) -> str:
        if spectrum is not None and spectrum.metadata.get("normalization_run") is not None:
            return str(spectrum.metadata.get("normalization_run"))
        if self.vanadium_spectrum is not None and self.vanadium_spectrum.run_number is not None:
            return str(self.vanadium_spectrum.run_number)
        return ""

    def _refresh_status_fields(self) -> None:
        if hasattr(self, "calibration_status_edit"):
            self.calibration_status_edit.setText(self._calibration_status_text())
        if hasattr(self, "phase_status_edit"):
            self.phase_status_edit.setText(self._phase_summary(self.phase))

    def open_calibration_dialog(self) -> None:
        dialog = CalibrationDialog(self)
        dialog.exec_()

    def open_fitting_settings_dialog(self) -> None:
        self._sync_fitting_settings_from_controls()
        dialog = FittingSettingsDialog(self)
        dialog.exec_()

    def _apply_fitting_settings_to_controls(self) -> None:
        scope_state = self.fit_scope_combo.blockSignals(True)
        order_state = self.poly_order.blockSignals(True)
        self._set_combo_data(self.fit_scope_combo, self.fitting_settings.fit_scope)
        self.poly_order.setValue(self.fitting_settings.polynomial_order)
        self.fit_scope_combo.blockSignals(scope_state)
        self.poly_order.blockSignals(order_state)

    def _sync_fitting_settings_from_controls(self) -> None:
        if not hasattr(self, "fit_scope_combo") or not hasattr(self, "poly_order"):
            return
        self.fitting_settings.fit_scope = str(self.fit_scope_combo.currentData() or FIT_SCOPE_PEAK)
        self.fitting_settings.polynomial_order = int(self.poly_order.value())

    def current_raw_spectrum(self) -> Optional[FocusedSpectrum]:
        index = self.spectrum_combo.currentIndex()
        if index < 0 or index >= len(self.spectra):
            return None
        return self.spectra[index]

    def current_spectrum(self) -> Optional[FocusedSpectrum]:
        index = self.spectrum_combo.currentIndex()
        if index < 0 or index >= len(self.spectra):
            return None
        if (
            hasattr(self, "data_combo")
            and self.data_combo.currentData() == "corrected"
            and index < len(self.corrected_spectra)
            and self.corrected_spectra[index] is not None
        ):
            return self.corrected_spectra[index]
        raw = self.spectra[index]
        return self._with_refined_calibration_if_matching(raw)

    def _with_refined_calibration_if_matching(self, spectrum: FocusedSpectrum) -> FocusedSpectrum:
        result = self.calibration_result
        if result is None:
            return spectrum
        if spectrum.bank_number is not None and result.bank_number is not None and spectrum.bank_number != result.bank_number:
            return spectrum
        return spectrum_with_calibration(
            spectrum,
            result.calibration,
            {
                "calibration_run": result.run_number,
                "calibration_source": str(result.source_path),
                "calibration_phase": result.phase_name,
                "calibration_rms_tof": result.rms_residual_tof,
                "single_peak_calibration_difc": result.single_peak_calibration.difc,
                "single_peak_calibration_tzero": result.single_peak_calibration.tzero,
                "single_peak_calibration_rms_tof": result.single_peak_rms_residual_tof,
                "pattern_calibration_profile": result.pattern_fit.profile_name if result.pattern_fit is not None else "",
            },
        )

    def load_spectrum_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load focused diffraction data",
            str(Path.cwd() / "Hist"),
            "Diffraction data (*.his *.xye *.dat *.csv);;All files (*.*)",
        )
        self.load_spectrum_paths([Path(path_str) for path_str in paths])

    def load_spectrum_paths(self, paths: list[Path]) -> None:
        corrected_count = 0
        loaded_count = 0
        last_corrected_index: Optional[int] = None
        last_loaded_name = ""
        for path in paths:
            try:
                spectrum = load_focused_spectrum(path)
            except Exception as exc:
                self._set_status(f"Failed to load {path.name}: {exc}")
                continue
            index = len(self.spectra)
            self.spectra.append(spectrum)
            self.corrected_spectra.append(None)
            self.normalization_results.append(None)
            bank = f" bank {spectrum.bank_number}" if spectrum.bank_number is not None else ""
            run = f" run {spectrum.run_number}" if spectrum.run_number else ""
            self.spectrum_combo.addItem(f"{spectrum.source_path.name}{run}{bank}")
            last_loaded_name = spectrum.source_path.name
            loaded_count += 1
            if self._can_auto_normalise_imports():
                try:
                    self._apply_normalisation_to_index(index)
                    corrected_count += 1
                    last_corrected_index = index
                except Exception as exc:
                    self._set_status(f"Auto-correction failed for {spectrum.source_path.name}: {exc}")
        if loaded_count:
            self.spectrum_combo.setCurrentIndex(last_corrected_index if last_corrected_index is not None else len(self.spectra) - 1)
            if corrected_count:
                self.data_combo.setCurrentIndex(1)
                noun = "spectrum" if corrected_count == 1 else "spectra"
                self._set_status(f"Loaded and auto-corrected {corrected_count} {noun}.")
            elif last_loaded_name:
                self._set_status(f"Loaded {last_loaded_name}.")
            self.update_plot()

    def load_phase_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load GSAS phase EXP",
            str(Path.cwd() / "Hist"),
            "GSAS EXP files (*.exp *.EXP);;All files (*.*)",
        )
        if not path_str:
            return
        try:
            self.phase = parse_gsas_exp(Path(path_str))[0]
            self._set_status(f"Loaded phase {self.phase.name} from {Path(path_str).name}.")
        except Exception as exc:
            self._set_status(f"Failed to load phase: {exc}")
        self._refresh_status_fields()

    def load_calibration_spectrum_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load CeO2 calibration histogram",
            str(Path.cwd() / "Hist"),
            "Open GENIE histogram (*.his);;All files (*.*)",
        )
        if not path_str:
            return
        try:
            self.calibration_spectrum = load_focused_spectrum(Path(path_str))
            self.calibration_result = None
            self.calibration_fit_spectrum = None
            self._set_status(
                f"Loaded CeO2 calibration run {self.calibration_spectrum.run_number or ''} "
                f"bank {self.calibration_spectrum.bank_number or ''}."
            )
        except Exception as exc:
            self._set_status(f"Failed to load CeO2 calibration histogram: {exc}")
        self._refresh_status_fields()

    def load_calibration_phase_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load CeO2 phase EXP",
            str(Path.cwd() / "Hist"),
            "GSAS EXP files (*.exp *.EXP);;All files (*.*)",
        )
        if not path_str:
            return
        try:
            self.calibration_phase = parse_gsas_exp(Path(path_str))[0]
            self.calibration_result = None
            self.calibration_fit_spectrum = None
            self._set_status(f"Loaded calibration phase {self.calibration_phase.name} from {Path(path_str).name}.")
        except Exception as exc:
            self._set_status(f"Failed to load CeO2 phase: {exc}")
        self._refresh_status_fields()

    def load_vanadium_spectrum_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load open beam normalization histogram",
            str(Path.cwd() / "Hist"),
            "Open GENIE histogram (*.his);;All files (*.*)",
        )
        if not path_str:
            return
        try:
            self.vanadium_spectrum = load_focused_spectrum(Path(path_str))
            self.calibration_result = None
            self.calibration_fit_spectrum = None
            self._set_status(
                f"Loaded open beam run {self.vanadium_spectrum.run_number or ''} "
                f"bank {self.vanadium_spectrum.bank_number or ''}."
            )
        except Exception as exc:
            self._set_status(f"Failed to load open beam histogram: {exc}")
        self._refresh_status_fields()

    def _draw_calibration_stage(self, status: str) -> None:
        if not hasattr(self, "canvas"):
            return
        self.axes.legend(loc="best")
        self.canvas.draw_idle()
        self.canvas.flush_events()
        QApplication.processEvents()
        self._set_status(status)

    def _calibration_plot_y_label(self) -> str:
        if self.calibration_fit_spectrum is not None:
            return self.calibration_fit_spectrum.y_label
        return "Counts"

    def _show_calibration_histogram(self) -> None:
        if not hasattr(self, "axes") or self.calibration_fit_spectrum is None:
            return
        spectrum = self.calibration_fit_spectrum
        x_values = spectrum.axis_values(AXIS_TOF)
        self.axes.clear()
        self.residual_axes.clear()
        corrected = "normalization_run" in spectrum.metadata
        self.axes.plot(
            x_values,
            spectrum.y,
            color="#2563a9",
            linestyle="None",
            marker=".",
            markersize=3,
            alpha=0.9,
            label="Open-beam corrected CeO2" if corrected else "CeO2 calibration histogram",
        )
        range_low = max(float(np.nanmin(x_values)), OPENGENIE_CALIBRATION_TOF_MIN)
        range_high = min(float(np.nanmax(x_values)), OPENGENIE_CALIBRATION_TOF_MAX)
        if range_high > range_low:
            self.axes.axvspan(range_low, range_high, color="#f59e0b", alpha=0.16, label="Calibration TOF range")
        self.axes.set_ylabel(spectrum.y_label)
        self.axes.tick_params(axis="x", labelbottom=False)
        self.residual_axes.set_xlabel("Time-of-Flight")
        self.residual_axes.set_ylabel("Residual")
        self.residual_axes.axhline(0.0, color="#94a3b8", linewidth=0.8)
        title_parts = ["Open-beam corrected calibration histogram" if corrected else "Calibration histogram", spectrum.source_path.name]
        if spectrum.run_number:
            title_parts.append(f"run {spectrum.run_number}")
        if spectrum.bank_number is not None:
            title_parts.append(f"bank {spectrum.bank_number}")
        title_parts.append(f"TOF {OPENGENIE_CALIBRATION_TOF_MIN:.1f}-{OPENGENIE_CALIBRATION_TOF_MAX:.1f} us")
        self.axes.set_title(" - ".join(title_parts))
        self.axes.set_ylim(*self._auto_y_range(x_values, spectrum.y, (float(np.nanmin(x_values)), float(np.nanmax(x_values)))))
        self._draw_calibration_stage(
            "Calibration stage 1/4: displaying open-beam corrected CeO2 calibration histogram."
            if corrected
            else "Calibration stage 1/4: displaying CeO2 calibration histogram."
        )

    def _show_calibration_progress(self, stage: str, payload: dict[str, object]) -> None:
        try:
            if stage == "single_peak_fit":
                self._show_calibration_single_peak_fit(payload)
            elif stage == "linear_fit":
                self._show_calibration_linear_fit(payload)
            elif stage == "pattern_fit":
                self._show_calibration_pattern_fit(payload)
        except Exception as exc:
            self._set_status(f"Calibration plot update failed: {exc}")

    def _show_calibration_single_peak_fit(self, payload: dict[str, object]) -> None:
        fit = payload.get("fit")
        x_values = np.asarray(payload.get("x"), dtype=float)
        y_values = np.asarray(payload.get("y"), dtype=float)
        if fit is None or x_values.size == 0 or y_values.size == 0:
            return
        fit_x = getattr(fit, "fit_x", None)
        fit_y = getattr(fit, "fit_y", None)
        observed_y = getattr(fit, "observed_y", None)
        if fit_x is None or fit_y is None:
            fit_x = x_values
            fit_y = np.zeros_like(x_values)
        if observed_y is None:
            observed_y = y_values
        reflection = payload.get("reflection")
        reflection_label = getattr(reflection, "label", "")
        self.axes.clear()
        self.residual_axes.clear()
        self.axes.plot(x_values, y_values, color="#2563a9", linestyle="None", marker=".", markersize=4, label="Peak data")
        self.axes.plot(fit_x, fit_y, color="#dc2626", linewidth=1.4, label="VEXP single-peak fit")
        expected_tof = payload.get("expected_tof")
        if expected_tof is not None:
            self.axes.axvline(float(expected_tof), color="#16a34a", linewidth=1.0, linestyle="--", label="Expected CeO2 TOF")
        residual = np.asarray(observed_y, dtype=float) - np.asarray(fit_y, dtype=float)
        self.residual_axes.plot(fit_x, residual, color="#475569", linewidth=0.9)
        self.residual_axes.axhline(0.0, color="#94a3b8", linewidth=0.8)
        self.axes.set_ylabel(self._calibration_plot_y_label())
        self.axes.tick_params(axis="x", labelbottom=False)
        self.residual_axes.set_xlabel("Time-of-Flight")
        self.residual_axes.set_ylabel("Residual")
        self.axes.set_title(f"Calibration single-peak fit {reflection_label}".strip())
        self.residual_axes.set_ylim(*self._auto_y_range(np.asarray(fit_x, dtype=float), residual, (float(np.nanmin(fit_x)), float(np.nanmax(fit_x)))))
        self._draw_calibration_stage("Calibration stage 2/4: displaying Open GENIE VEXP-style single-peak fit.")

    def _show_calibration_linear_fit(self, payload: dict[str, object]) -> None:
        peaks = [peak for peak in payload.get("peaks", ()) if getattr(peak, "accepted", False) and getattr(peak, "fitted_tof", None) is not None]
        calibration = payload.get("calibration")
        if not peaks or calibration is None:
            return
        d_values = np.array([peak.reflection.d_spacing for peak in peaks], dtype=float)
        tof_values = np.array([float(peak.fitted_tof) for peak in peaks], dtype=float)
        order = np.argsort(d_values)
        d_values = d_values[order]
        tof_values = tof_values[order]
        line_d = np.linspace(float(np.min(d_values)), float(np.max(d_values)), 300)
        line_tof = d_to_tof(line_d, calibration)
        residual = tof_values - d_to_tof(d_values, calibration)
        self.axes.clear()
        self.residual_axes.clear()
        self.axes.plot(d_values, tof_values, color="#2563a9", linestyle="None", marker="D", markersize=5, label="Fitted peak centres")
        self.axes.plot(line_d, line_tof, color="#dc2626", linewidth=1.4, label="TOF = DIFC1*d + ZERO1")
        self.residual_axes.plot(d_values, residual, color="#475569", linestyle="None", marker=".", markersize=5)
        self.residual_axes.axhline(0.0, color="#94a3b8", linewidth=0.8)
        self.axes.set_ylabel("Time-of-Flight")
        self.axes.tick_params(axis="x", labelbottom=False)
        self.residual_axes.set_xlabel("d-spacing")
        self.residual_axes.set_ylabel("Residual")
        self.axes.set_title(f"Calibration linear fit: DIFC1={calibration.difc:.6g}, ZERO1={calibration.tzero:.6g}")
        self.residual_axes.set_ylim(*self._auto_y_range(d_values, residual, (float(np.min(d_values)), float(np.max(d_values)))))
        self._draw_calibration_stage("Calibration stage 3/4: displaying linear DIFC1/ZERO1 fit.")

    def _show_calibration_pattern_fit(self, payload: dict[str, object]) -> None:
        fit = payload.get("fit")
        if fit is None or getattr(fit, "fit_x", None) is None or getattr(fit, "fit_y", None) is None or getattr(fit, "observed_y", None) is None:
            return
        fit_x = np.asarray(fit.fit_x, dtype=float)
        fit_y = np.asarray(fit.fit_y, dtype=float)
        observed_y = np.asarray(fit.observed_y, dtype=float)
        residual = observed_y - fit_y
        self.axes.clear()
        self.residual_axes.clear()
        self.axes.plot(fit_x, observed_y, color="#2563a9", linestyle="None", marker=".", markersize=3, label="CeO2 pattern")
        self.axes.plot(fit_x, fit_y, color="#dc2626", linewidth=1.4, label=str(fit.profile_name))
        self.residual_axes.plot(fit_x, residual, color="#475569", linewidth=0.9)
        self.residual_axes.axhline(0.0, color="#94a3b8", linewidth=0.8)
        self.axes.set_ylabel(self._calibration_plot_y_label())
        self.axes.tick_params(axis="x", labelbottom=False)
        self.residual_axes.set_xlabel("Time-of-Flight")
        self.residual_axes.set_ylabel("Residual")
        self.axes.set_title(f"Calibration whole-pattern fit: {fit.profile_name}, Rwp={fit.quality.rwp_percent:.4g}%")
        self.residual_axes.set_ylim(*self._auto_y_range(fit_x, residual, (float(np.nanmin(fit_x)), float(np.nanmax(fit_x)))))
        self._draw_calibration_stage("Calibration stage 4/4: displaying GSAS profile-3 whole-pattern fit.")

    def refine_current_calibration(self) -> bool:
        try:
            self.calibration_result = None
            self._refresh_status_fields()
            if self.calibration_spectrum is None:
                raise ValueError("Load a CeO2 calibration histogram first.")
            if self.calibration_phase is None:
                raise ValueError("Load a CeO2 phase EXP first.")
            if self.vanadium_spectrum is None:
                raise ValueError("Load an open beam histogram before calibration.")
            raw = self.current_raw_spectrum()
            if (
                raw is not None
                and raw.bank_number is not None
                and self.calibration_spectrum.bank_number is not None
                and raw.bank_number != self.calibration_spectrum.bank_number
            ):
                raise ValueError(
                    f"Bank mismatch: sample bank {raw.bank_number} cannot use CeO2 bank {self.calibration_spectrum.bank_number}."
                )
            if (
                self.calibration_spectrum.bank_number is not None
                and self.vanadium_spectrum.bank_number is not None
                and self.calibration_spectrum.bank_number != self.vanadium_spectrum.bank_number
            ):
                raise ValueError(
                    f"Bank mismatch: CeO2 bank {self.calibration_spectrum.bank_number} cannot use open beam bank {self.vanadium_spectrum.bank_number}."
                )
            calibration_normalization = apply_vanadium_normalization(self.calibration_spectrum, self.vanadium_spectrum, None)
            self.calibration_fit_spectrum = calibration_normalization.corrected_spectrum
            self.latest_fit_curve = None
            self._show_calibration_histogram()
            self.calibration_result = refine_ceo2_calibration(
                self.calibration_fit_spectrum,
                self.calibration_phase,
                progress_callback=self._show_calibration_progress,
            )
            self.latest_fit_curve = None
            accepted = len(self.calibration_result.accepted_peaks)
            self._set_status(
                f"Refined calibration from CeO2 run {self.calibration_result.run_number}: "
                f"DIFC={self.calibration_result.calibration.difc:.6g}, "
                f"DIFA={self.calibration_result.calibration.difa:.6g}, "
                f"TZERO={self.calibration_result.calibration.tzero:.6g}, "
                f"{accepted} VEXP peak(s), GSAS profile 3 RMS={self.calibration_result.rms_residual_tof:.4g} TOF "
                f"after open beam run {self.vanadium_spectrum.run_number or ''}."
            )
            self._refresh_status_fields()
            return True
        except Exception as exc:
            self.calibration_fit_spectrum = None
            self._set_status(f"Calibration failed: {exc}")
            self._refresh_status_fields()
            return False

    def run_calibration_sequence(self) -> bool:
        if not self.refine_current_calibration():
            return False
        if self.current_raw_spectrum() is None:
            self._set_status("Calibration complete from open-beam corrected CeO2. Imported spectra will be corrected automatically.")
            return True
        normalised = self.apply_current_normalisation()
        if normalised and self.calibration_result is not None and self.calibration_result.pattern_fit is not None:
            self._show_calibration_pattern_fit({"fit": self.calibration_result.pattern_fit})
            self._set_status("Calibration and open beam normalization complete. Final calibration pattern fit is displayed.")
        return normalised

    def apply_current_normalisation(self) -> bool:
        try:
            index = self.spectrum_combo.currentIndex()
            if self.current_raw_spectrum() is None or index < 0:
                raise ValueError("Load a sample spectrum before applying open beam normalization.")
            if self.vanadium_spectrum is None:
                raise ValueError("Load an open beam normalization histogram first.")
            result = self._apply_normalisation_to_index(index)
            self.latest_fit_curve = None
            self.data_combo.setCurrentIndex(1)
            self._set_status(
                f"Applied open beam run {result.vanadium_run_number} to sample run {result.sample_run_number}: "
                f"{result.valid_bins} valid bin(s), {result.invalid_bins} invalid."
            )
            self._refresh_status_fields()
            self.update_plot()
            return True
        except Exception as exc:
            self._set_status(f"Open beam normalization failed: {exc}")
            self._refresh_status_fields()
            return False

    def _can_auto_normalise_imports(self) -> bool:
        return self.calibration_result is not None and self.vanadium_spectrum is not None

    def _apply_normalisation_to_index(self, index: int) -> NormalizationResult:
        if index < 0 or index >= len(self.spectra):
            raise ValueError("Spectrum index is out of range.")
        if self.vanadium_spectrum is None:
            raise ValueError("Load an open beam normalization histogram first.")
        result = apply_vanadium_normalization(self.spectra[index], self.vanadium_spectrum, self.calibration_result)
        self.corrected_spectra[index] = result.corrected_spectrum
        self.normalization_results[index] = result
        self.latest_fit_curve = None
        return result

    def _selected_axis(self) -> str:
        return str(self.axis_combo.currentData() or AXIS_TOF)

    def _axis_data(self, spectrum: FocusedSpectrum) -> tuple[np.ndarray, str]:
        axis = self._selected_axis()
        x_values = spectrum.axis_values(axis)
        return x_values, spectrum.axis_label(axis)

    def update_plot(self) -> None:
        if not hasattr(self, "axes"):
            return
        self.axes.clear()
        self.residual_axes.clear()
        spectrum = self.current_spectrum()
        if spectrum is None:
            self.axes.set_title("No spectrum loaded")
            self.residual_axes.set_ylabel("Residual")
            self.canvas.draw_idle()
            return
        try:
            x_values, x_label = self._axis_data(spectrum)
        except Exception as exc:
            self._set_status(str(exc))
            self.axis_combo.setCurrentIndex(0)
            x_values, x_label = spectrum.axis_values(AXIS_TOF), spectrum.axis_label(AXIS_TOF)
        self.axes.plot(
            x_values,
            spectrum.y,
            color="#2563a9",
            linestyle="None",
            marker=".",
            markersize=3,
            alpha=0.9,
            label="Spectrum",
        )
        self.axes.set_ylabel(spectrum.y_label)
        self.axes.tick_params(axis="x", labelbottom=False)
        self.residual_axes.set_xlabel(x_label)
        self.residual_axes.set_ylabel("Residual")
        title_parts = [spectrum.source_path.name]
        if spectrum.run_number:
            title_parts.append(f"run {spectrum.run_number}")
        if spectrum.bank_number is not None:
            title_parts.append(f"bank {spectrum.bank_number}")
        if "normalization_run" in spectrum.metadata:
            title_parts.append("corrected")
        elif "calibration_run" in spectrum.metadata:
            title_parts.append("calibrated")
        self.axes.set_title(" - ".join(title_parts))
        view_range = self._current_view_range_or_none(x_values, spectrum)
        if view_range is not None:
            self.axes.set_xlim(view_range[0], view_range[1])
        else:
            view_range = (float(np.nanmin(x_values)), float(np.nanmax(x_values)))
        auto_y_range = self._auto_y_range(x_values, spectrum.y, view_range)
        y_range = self._current_y_view_range_or_none(auto_y_range)
        if y_range is not None:
            self.axes.set_ylim(y_range[0], y_range[1])
        fit_curve = self._current_fit_curve()
        if fit_curve is not None:
            fit_x = fit_curve["x"]
            fit_y = fit_curve["fit_y"]
            observed_y = fit_curve["observed_y"]
            if isinstance(fit_x, np.ndarray) and isinstance(fit_y, np.ndarray) and isinstance(observed_y, np.ndarray):
                self.axes.plot(fit_x, fit_y, color="#dc2626", linewidth=1.4, label=str(fit_curve["label"]))
                residual = observed_y - fit_y
                self.residual_axes.plot(
                    fit_x,
                    residual,
                    color="#475569",
                    linestyle="-",
                    linewidth=0.9,
                    alpha=0.9,
                )
                self.residual_axes.axhline(0.0, color="#94a3b8", linewidth=0.8)
                self.residual_axes.set_ylim(*self._auto_y_range(fit_x, residual, view_range))
                self.axes.legend(loc="best")
        else:
            self.residual_axes.axhline(0.0, color="#94a3b8", linewidth=0.8)
        range_values = self._current_range_or_none()
        if range_values is not None:
            self.axes.axvspan(range_values[0], range_values[1], color="#f59e0b", alpha=0.18)
        if SpanSelector is not None:
            self._span_selector = SpanSelector(
                self.axes,
                self._on_span_selected,
                "horizontal",
                useblit=True,
                props={"alpha": 0.18, "facecolor": "#f59e0b"},
                interactive=True,
            )
        self.canvas.draw_idle()

    def _current_fit_curve(self) -> Optional[dict[str, object]]:
        curve = self.latest_fit_curve
        if curve is None:
            return None
        if curve.get("spectrum_index") != self.spectrum_combo.currentIndex():
            return None
        if curve.get("axis") != self._selected_axis():
            return None
        if curve.get("data_mode") != self.data_combo.currentData():
            return None
        return curve

    def _store_latest_fit_curve(
        self,
        fit_x: Optional[np.ndarray],
        fit_y: Optional[np.ndarray],
        observed_y: Optional[np.ndarray],
        axis: str,
        label: str,
    ) -> None:
        if fit_x is None or fit_y is None or observed_y is None:
            self.latest_fit_curve = None
            return
        self.latest_fit_curve = {
            "spectrum_index": self.spectrum_combo.currentIndex(),
            "axis": axis,
            "data_mode": self.data_combo.currentData(),
            "x": np.array(fit_x, dtype=float, copy=True),
            "fit_y": np.array(fit_y, dtype=float, copy=True),
            "observed_y": np.array(observed_y, dtype=float, copy=True),
            "label": label,
        }

    def _on_span_selected(self, xmin: float, xmax: float) -> None:
        low, high = sorted((float(xmin), float(xmax)))
        self.range_min_edit.setText(f"{low:.6g}")
        self.range_max_edit.setText(f"{high:.6g}")

    def _current_range_or_none(self) -> Optional[tuple[float, float]]:
        try:
            low = float(self.range_min_edit.text())
            high = float(self.range_max_edit.text())
        except ValueError:
            return None
        return tuple(sorted((low, high)))

    def _current_view_range_or_none(self, x_values: np.ndarray, spectrum: FocusedSpectrum) -> Optional[tuple[float, float]]:
        min_text = self.view_min_edit.text().strip()
        max_text = self.view_max_edit.text().strip()
        x_min = float(np.nanmin(x_values))
        x_max = float(np.nanmax(x_values))
        try:
            low = float(min_text) if min_text else self._default_view_min(x_values, spectrum)
            high = float(max_text) if max_text else x_max
        except ValueError:
            self._set_status("View x bounds must be numeric.")
            return None
        if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
            self._set_status("View x bounds must be finite and different.")
            return None
        return (low, high)

    def _default_view_min(self, x_values: np.ndarray, spectrum: FocusedSpectrum) -> float:
        finite_x = x_values[np.isfinite(x_values)]
        if finite_x.size == 0:
            return 0.0
        x_min = float(np.nanmin(finite_x))
        x_max = float(np.nanmax(finite_x))
        default_min = 10000.0
        if self._selected_axis() == AXIS_D_SPACING and spectrum.calibration is not None:
            default_min = float(tof_to_d(10000.0, spectrum.calibration))
        if not np.isfinite(default_min):
            return x_min
        if default_min <= x_min:
            return x_min
        if default_min >= x_max:
            return x_min
        return default_min

    def _auto_y_range(self, x_values: np.ndarray, y_values: np.ndarray, x_range: tuple[float, float]) -> tuple[float, float]:
        low, high = sorted((float(x_range[0]), float(x_range[1])))
        mask = (x_values >= low) & (x_values <= high) & np.isfinite(y_values)
        visible_y = y_values[mask]
        if visible_y.size == 0:
            visible_y = y_values[np.isfinite(y_values)]
        if visible_y.size == 0:
            return (0.0, 1.0)
        y_min = float(np.nanmin(visible_y))
        y_max = float(np.nanmax(visible_y))
        if abs(y_max - y_min) < 1e-12:
            padding = max(abs(y_min) * 0.05, 1.0)
        else:
            padding = 0.06 * (y_max - y_min)
        return (y_min - padding, y_max + padding)

    def _current_y_view_range_or_none(self, auto_range: tuple[float, float]) -> Optional[tuple[float, float]]:
        min_text = self.y_view_min_edit.text().strip()
        max_text = self.y_view_max_edit.text().strip()
        if not min_text and not max_text:
            return auto_range
        try:
            low = float(min_text) if min_text else float(auto_range[0])
            high = float(max_text) if max_text else float(auto_range[1])
        except ValueError:
            self._set_status("View y bounds must be numeric.")
            return auto_range
        if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
            self._set_status("View y bounds must be finite and different.")
            return auto_range
        return tuple(sorted((low, high)))

    def reset_plot_view(self) -> None:
        self.view_min_edit.clear()
        self.view_max_edit.clear()
        self.y_view_min_edit.clear()
        self.y_view_max_edit.clear()
        self.update_plot()

    def _fit_range_mask(self, x_values: np.ndarray, spectrum: FocusedSpectrum) -> tuple[np.ndarray, tuple[float, float]]:
        selected_range = self._current_range_or_none()
        if selected_range is None:
            selected_range = self._current_view_range_or_none(x_values, spectrum)
        if selected_range is None:
            selected_range = (float(np.nanmin(x_values)), float(np.nanmax(x_values)))
        if not self.range_min_edit.text().strip() or not self.range_max_edit.text().strip():
            self.range_min_edit.setText(f"{selected_range[0]:.6g}")
            self.range_max_edit.setText(f"{selected_range[1]:.6g}")
        mask = (x_values >= selected_range[0]) & (x_values <= selected_range[1])
        if int(np.count_nonzero(mask)) < 8:
            raise ValueError("Selected range contains fewer than 8 data points.")
        return mask, selected_range

    def _convert_axis_values(
        self,
        values: np.ndarray,
        from_axis: str,
        to_axis: str,
        spectrum: FocusedSpectrum,
    ) -> np.ndarray:
        if from_axis == to_axis:
            return np.array(values, dtype=float, copy=True)
        if spectrum.calibration is None:
            raise ValueError("Calibration metadata is required for TOF/d-spacing conversion.")
        if from_axis == AXIS_TOF and to_axis == AXIS_D_SPACING:
            return tof_to_d(values, spectrum.calibration)
        if from_axis == AXIS_D_SPACING and to_axis == AXIS_TOF:
            return d_to_tof(values, spectrum.calibration)
        raise ValueError(f"Unsupported axis conversion: {from_axis} to {to_axis}")

    def _fit_axis_for_mode(self, selected_axis: str, fit_mode: str) -> str:
        if is_gsas_tof_profile(fit_mode):
            return AXIS_TOF
        return selected_axis

    def _profile_key_for_scope(self, settings: FittingSettings) -> str:
        if settings.fit_scope == FIT_SCOPE_PATTERN:
            return settings.pattern_profile_key
        return settings.peak_profile_key

    def _d_spacing_fit_centre_fwhm(
        self,
        centre: float,
        fwhm: float,
        fit_axis: str,
        spectrum: FocusedSpectrum,
    ) -> tuple[float, float]:
        return float(self._axis_position_to_d_spacing(centre, fit_axis, spectrum)), self._axis_width_to_d_spacing(fwhm, centre, fit_axis, spectrum)

    def _axis_position_to_d_spacing(self, value: float, axis: str, spectrum: FocusedSpectrum) -> float:
        if axis == AXIS_D_SPACING:
            return float(value)
        if spectrum.calibration is None:
            raise ValueError("Calibration metadata is required to report fitting results in d-spacing.")
        return float(tof_to_d(float(value), spectrum.calibration))

    def _axis_width_to_d_spacing(self, width: float, centre: float, axis: str, spectrum: FocusedSpectrum) -> float:
        if axis == AXIS_D_SPACING:
            return abs(float(width))
        if spectrum.calibration is None:
            raise ValueError("Calibration metadata is required to report fitting results in d-spacing.")
        centre_d = float(tof_to_d(float(centre), spectrum.calibration))
        local_slope = float(spectrum.calibration.difc) + 2.0 * float(spectrum.calibration.difa) * centre_d
        if abs(local_slope) < 1e-12:
            raise ValueError("Calibration slope is too small to convert TOF width to d-spacing.")
        return abs(float(width) / local_slope)

    def _optional_axis_width_to_d_spacing(
        self,
        width: Optional[float],
        centre: float,
        axis: str,
        spectrum: FocusedSpectrum,
    ) -> Optional[float]:
        if width is None:
            return None
        return self._axis_width_to_d_spacing(float(width), centre, axis, spectrum)

    def _format_optional_value(self, value: Optional[float], precision: int = 6) -> str:
        if value is None:
            return ""
        return f"{float(value):.{precision}g}"

    def fit_selected_range(self) -> None:
        spectrum = self.current_spectrum()
        if spectrum is None:
            self._set_status("Load a spectrum before fitting.")
            return
        try:
            self._sync_fitting_settings_from_controls()
            settings = self.fitting_settings
            axis = self._selected_axis()
            x_values, _label = self._axis_data(spectrum)
            mask, selected_range = self._fit_range_mask(x_values, spectrum)
            e_values = spectrum.e[mask] if settings.use_uncertainties and spectrum.e is not None else None
            fit_scope = settings.fit_scope
            profile_key = self._profile_key_for_scope(settings)
            fit_axis = self._fit_axis_for_mode(axis, profile_key)
            fit_x_values = x_values if fit_axis == axis else self._convert_axis_values(x_values, axis, fit_axis, spectrum)
            if fit_scope == FIT_SCOPE_PATTERN:
                if self.phase is None:
                    raise ValueError("Load a GSAS EXP phase before whole-pattern fitting.")
                result = fit_pawley(
                    fit_x_values[mask],
                    spectrum.y[mask],
                    e_values,
                    self.phase,
                    fit_axis,
                    spectrum.calibration,
                    polynomial_order=settings.polynomial_order,
                    lattice_tolerance_percent=settings.pawley_lattice_tolerance_percent,
                    reflection_margin_percent=settings.pawley_reflection_margin_percent,
                    eta_initial=settings.pawley_eta_initial,
                    eta_bounds=(settings.pawley_eta_min, settings.pawley_eta_max),
                    fwhm_min_fraction=settings.pawley_fwhm_min_fraction,
                    fwhm_max_fraction=settings.pawley_fwhm_max_fraction,
                    max_nfev=settings.max_evaluations,
                    profile_key=profile_key,
                    width_model=settings.pawley_width_model,
                )
                reflection_details = "; ".join(
                    f"{item.reflection.label}@{self._axis_position_to_d_spacing(item.position, fit_axis, spectrum):.6g}:I={item.intensity:.6g}"
                    for item in result.reflections
                )
                profile_details = ",".join(f"{key}={value:.6g}" for key, value in result.profile_parameters.items())
                if profile_details:
                    profile_details = ";profile_parameters=" + profile_details
                self._append_structured_result_row(
                    [
                        result.profile_name,
                        "Whole pattern: Pawley",
                        spectrum.source_path.name,
                        spectrum.run_number or "",
                        str(spectrum.bank_number or ""),
                        axis,
                        f"{selected_range[0]:.6g}",
                        f"{selected_range[1]:.6g}",
                        f"{result.lattice_a:.8g}",
                        self._format_optional_value(result.lattice_a_uncertainty),
                        f"{result.quality.reduced_chi_square:.6g}",
                        f"{result.quality.rwp_percent:.4g}",
                    ],
                    [],
                    "reported_value=lattice_a;reported_axis=d-spacing;fit_axis="
                        + fit_axis
                        + ";peak_function="
                        + result.profile_name
                        + ";width_model="
                        + settings.pawley_width_model
                        + profile_details
                        + ";reflections="
                        + reflection_details,
                )
                curve_x = result.fit_x
                if curve_x is not None and fit_axis != axis:
                    curve_x = self._convert_axis_values(curve_x, fit_axis, axis, spectrum)
                self._store_latest_fit_curve(curve_x, result.fit_y, result.observed_y, axis, f"Pawley {result.profile_name} fit")
                self._set_status(f"Whole-pattern Pawley fit complete with {result.profile_name}: a={result.lattice_a:.6g}, Rwp={result.quality.rwp_percent:.4g}%.")
            else:
                result = fit_peak_profile(
                    profile_key,
                    fit_x_values[mask],
                    spectrum.y[mask],
                    e_values,
                    polynomial_order=settings.polynomial_order,
                    eta_initial=settings.pseudo_voigt_eta_initial,
                    eta_bounds=(settings.pseudo_voigt_eta_min, settings.pseudo_voigt_eta_max),
                    fwhm_min_fraction=settings.pseudo_voigt_fwhm_min_fraction,
                    fwhm_max_multiplier=settings.pseudo_voigt_fwhm_max_multiplier,
                    maxfev=settings.max_evaluations,
                )
                extra_details = ",".join(f"{key}={value:.6g}" for key, value in result.profile_parameters.items())
                if extra_details:
                    extra_details = ";" + extra_details
                components = result.components or ()
                curve_x = result.fit_x
                if curve_x is not None and fit_axis != axis:
                    curve_x = self._convert_axis_values(curve_x, fit_axis, axis, spectrum)
                peak_groups = []
                for component in components:
                    centre_d = self._axis_position_to_d_spacing(component.centre, fit_axis, spectrum)
                    fwhm_d = self._axis_width_to_d_spacing(component.fwhm, component.centre, fit_axis, spectrum)
                    centre_uncertainty_d = self._optional_axis_width_to_d_spacing(
                        component.centre_uncertainty,
                        component.centre,
                        fit_axis,
                        spectrum,
                    )
                    fwhm_uncertainty_d = self._optional_axis_width_to_d_spacing(
                        component.fwhm_uncertainty,
                        component.centre,
                        fit_axis,
                        spectrum,
                    )
                    peak_groups.append(
                        [
                            f"{centre_d:.8g}",
                            self._format_optional_value(centre_uncertainty_d),
                            f"{fwhm_d:.6g}",
                            self._format_optional_value(fwhm_uncertainty_d),
                            f"{component.height:.6g}",
                            f"{component.area:.6g}",
                            f"{component.eta:.4g}",
                        ]
                    )
                self._append_structured_result_row(
                    [
                        result.model_name,
                        "Individual peak fitting",
                        spectrum.source_path.name,
                        spectrum.run_number or "",
                        str(spectrum.bank_number or ""),
                        axis,
                        f"{selected_range[0]:.6g}",
                        f"{selected_range[1]:.6g}",
                        "",
                        "",
                        f"{result.quality.reduced_chi_square:.6g}",
                        f"{result.quality.rwp_percent:.4g}",
                    ],
                    peak_groups,
                    "reported_value=peak_d;reported_axis=d-spacing;fit_axis="
                    + fit_axis
                    + f";peak_count={len(components)}"
                    + ";background="
                    + ",".join(f"{value:.6g}" for value in result.background_coefficients)
                    + extra_details,
                )
                self._store_latest_fit_curve(curve_x, result.fit_y, result.observed_y, axis, f"{result.model_name} fit")
                self._set_status(
                    f"Individual peak fit complete: {len(components)} {result.model_name} peak(s) in one row, Rwp={result.quality.rwp_percent:.4g}%."
                )
            self.update_plot()
        except Exception as exc:
            self._set_status(f"Fit failed: {exc}")

    def _append_result_row(self, values: list[str]) -> int:
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        for column, value in enumerate(values):
            self.results_table.setItem(row, column, QTableWidgetItem(value))
        self.results_table.resizeColumnsToContents()
        return row

    def _ensure_peak_result_columns(self, peak_count: int) -> None:
        peak_count = max(0, int(peak_count))
        if not hasattr(self, "results_table"):
            return
        while self._peak_result_column_count < peak_count:
            peak_number = self._peak_result_column_count + 1
            insert_at = self.results_table.columnCount()
            for offset, field in enumerate(PEAK_RESULT_FIELD_HEADERS):
                column = insert_at + offset
                self.results_table.insertColumn(column)
                self.results_table.setHorizontalHeaderItem(column, QTableWidgetItem(f"Peak {peak_number} {field}"))
            self._peak_result_column_count += 1

    def _append_structured_result_row(
        self,
        base_values: list[str],
        peak_groups: list[list[str]],
        details: str,
    ) -> None:
        expected_base_values = len(RESULT_HEADERS)
        values = list(base_values[:expected_base_values])
        values.extend([""] * max(0, expected_base_values - len(values)))
        self._ensure_peak_result_columns(len(peak_groups))
        for peak_values in peak_groups:
            group = list(peak_values[: len(PEAK_RESULT_FIELD_HEADERS)])
            values.extend(group + [""] * (len(PEAK_RESULT_FIELD_HEADERS) - len(group)))
        missing_peak_groups = max(0, self._peak_result_column_count - len(peak_groups))
        values.extend([""] * missing_peak_groups * len(PEAK_RESULT_FIELD_HEADERS))
        row = self._append_result_row(values)
        self.results_table.setVerticalHeaderItem(row, QTableWidgetItem(str(row + 1)))
        header_item = self.results_table.verticalHeaderItem(row)
        if header_item is not None:
            header_item.setData(Qt.UserRole, details)
        self.results_table.selectRow(row)

    def _update_selected_result_details(self) -> None:
        if not hasattr(self, "result_details_edit"):
            return
        selected_rows = sorted({index.row() for index in self.results_table.selectedIndexes()})
        if not selected_rows:
            self.result_details_edit.clear()
            return
        header_item = self.results_table.verticalHeaderItem(selected_rows[-1])
        details = "" if header_item is None else str(header_item.data(Qt.UserRole) or "")
        self.result_details_edit.setPlainText(self._format_result_details(details))

    def _format_result_details(self, details: str) -> str:
        if not details:
            return ""
        parts = [part for part in str(details).split(";") if part]
        lines = []
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                lines.append(f"{key.replace('_', ' ')}: {value}")
            else:
                lines.append(part)
        return "\n".join(lines)

    def export_results_dialog(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export diffraction results",
            str(Path.cwd() / "diffraction_results.csv"),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path_str:
            return
        try:
            with Path(path_str).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                headers = [
                        self.results_table.horizontalHeaderItem(column).text()
                        if self.results_table.horizontalHeaderItem(column) is not None
                        else ""
                        for column in range(self.results_table.columnCount())
                    ]
                headers.append("Details")
                writer.writerow(headers)
                for row in range(self.results_table.rowCount()):
                    values = [
                            self.results_table.item(row, column).text()
                            if self.results_table.item(row, column) is not None
                            else ""
                            for column in range(self.results_table.columnCount())
                        ]
                    header_item = self.results_table.verticalHeaderItem(row)
                    values.append("" if header_item is None else str(header_item.data(Qt.UserRole) or ""))
                    writer.writerow(values)
            self._set_status(f"Exported diffraction results to {path_str}.")
        except Exception as exc:
            self._set_status(f"Export failed: {exc}")
