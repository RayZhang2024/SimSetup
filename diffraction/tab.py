from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .calibration import refine_ceo2_calibration
from .fitting import fit_pseudo_voigt
from .gsas_exp import parse_gsas_exp
from .importers import load_focused_spectrum
from .models import AXIS_D_SPACING, AXIS_TOF, CalibrationResult, FocusedSpectrum, NormalizationResult, PhaseModel, spectrum_with_calibration
from .normalization import apply_vanadium_normalization
from .pawley import fit_pawley

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
    "Type",
    "Source",
    "Run",
    "Bank",
    "Axis",
    "Range min",
    "Range max",
    "Centre / a",
    "FWHM",
    "Intensity",
    "Eta",
    "Chi2",
    "Rwp %",
    "Correction",
    "CeO2 run",
    "Vanadium run",
    "DIFC",
    "TZERO",
    "DIFA",
    "Details",
]


class DiffractionTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.spectra: list[FocusedSpectrum] = []
        self.corrected_spectra: list[Optional[FocusedSpectrum]] = []
        self.normalization_results: list[Optional[NormalizationResult]] = []
        self.phase: Optional[PhaseModel] = None
        self.calibration_phase: Optional[PhaseModel] = None
        self.calibration_spectrum: Optional[FocusedSpectrum] = None
        self.vanadium_spectrum: Optional[FocusedSpectrum] = None
        self.calibration_result: Optional[CalibrationResult] = None
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
        self.phase_button = QPushButton("Load phase")
        self.phase_button.clicked.connect(self.load_phase_dialog)
        self.calibration_spectrum_button = QPushButton("Load CeO2 histogram")
        self.calibration_spectrum_button.clicked.connect(self.load_calibration_spectrum_dialog)
        self.calibration_phase_button = QPushButton("Load CeO2 phase")
        self.calibration_phase_button.clicked.connect(self.load_calibration_phase_dialog)
        self.vanadium_button = QPushButton("Load vanadium")
        self.vanadium_button.clicked.connect(self.load_vanadium_spectrum_dialog)
        self.refine_calibration_button = QPushButton("Refine calibration")
        self.refine_calibration_button.clicked.connect(self.refine_current_calibration)
        self.normalise_button = QPushButton("Apply normalisation")
        self.normalise_button.clicked.connect(self.apply_current_normalisation)
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
        self.fit_combo = QComboBox()
        self.fit_combo.addItem("Pseudo-Voigt peak", "pseudo_voigt")
        self.fit_combo.addItem("Pawley", "pawley")
        self.poly_order = QSpinBox()
        self.poly_order.setRange(0, 5)
        self.poly_order.setValue(2)
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
        self.apply_view_button = QPushButton("Apply view")
        self.apply_view_button.clicked.connect(self.update_plot)
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.clicked.connect(self.reset_plot_view)
        self.fit_button = QPushButton("Fit selected range")
        self.fit_button.clicked.connect(self.fit_selected_range)
        self.export_button = QPushButton("Export results")
        self.export_button.clicked.connect(self.export_results_dialog)

        controls_layout.addWidget(self.load_button, 0, 0, 1, 2)
        controls_layout.addWidget(self.phase_button, 1, 0, 1, 2)
        controls_layout.addWidget(QLabel("Spectrum"), 2, 0)
        controls_layout.addWidget(self.spectrum_combo, 2, 1)
        controls_layout.addWidget(QLabel("Axis"), 3, 0)
        controls_layout.addWidget(self.axis_combo, 3, 1)
        controls_layout.addWidget(QLabel("Data"), 4, 0)
        controls_layout.addWidget(self.data_combo, 4, 1)
        controls_layout.addWidget(QLabel("Fit"), 5, 0)
        controls_layout.addWidget(self.fit_combo, 5, 1)
        controls_layout.addWidget(QLabel("Background order"), 6, 0)
        controls_layout.addWidget(self.poly_order, 6, 1)
        controls_layout.addWidget(QLabel("Range min"), 7, 0)
        controls_layout.addWidget(self.range_min_edit, 7, 1)
        controls_layout.addWidget(QLabel("Range max"), 8, 0)
        controls_layout.addWidget(self.range_max_edit, 8, 1)
        controls_layout.addWidget(self.fit_button, 9, 0, 1, 2)
        controls_layout.addWidget(self.calibration_spectrum_button, 10, 0, 1, 2)
        controls_layout.addWidget(self.calibration_phase_button, 11, 0, 1, 2)
        controls_layout.addWidget(self.refine_calibration_button, 12, 0, 1, 2)
        controls_layout.addWidget(self.vanadium_button, 13, 0, 1, 2)
        controls_layout.addWidget(self.normalise_button, 14, 0, 1, 2)
        controls_layout.addWidget(QLabel("View x min"), 15, 0)
        controls_layout.addWidget(self.view_min_edit, 15, 1)
        controls_layout.addWidget(QLabel("View x max"), 16, 0)
        controls_layout.addWidget(self.view_max_edit, 16, 1)
        controls_layout.addWidget(QLabel("View y min"), 17, 0)
        controls_layout.addWidget(self.y_view_min_edit, 17, 1)
        controls_layout.addWidget(QLabel("View y max"), 18, 0)
        controls_layout.addWidget(self.y_view_max_edit, 18, 1)
        controls_layout.addWidget(self.apply_view_button, 19, 0, 1, 2)
        controls_layout.addWidget(self.reset_view_button, 20, 0, 1, 2)
        controls_layout.addWidget(self.export_button, 21, 0, 1, 2)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setRowStretch(22, 1)

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

        self.results_table = QTableWidget(0, len(RESULT_HEADERS))
        self.results_table.setHorizontalHeaderLabels(RESULT_HEADERS)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setWordWrap(False)
        splitter.addWidget(self.results_table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        status_frame = QFrame(self)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Load a reduced focused histogram to begin.")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_frame)

    def _load_default_phase(self) -> None:
        default_path = Path.cwd() / "Hist" / "NI.EXP"
        if default_path.exists():
            try:
                self.phase = parse_gsas_exp(default_path)[0]
                self._set_status(f"Loaded default phase {self.phase.name} from {default_path.name}.")
            except Exception as exc:
                self._set_status(f"Default phase load failed: {exc}")

    def _load_default_calibration_phase(self) -> None:
        for filename in ("CEO2.EXP", "CeO2.exp", "CeO2.EXP"):
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
            },
        )

    def load_spectrum_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load focused diffraction data",
            str(Path.cwd() / "Hist"),
            "Diffraction data (*.his *.xye *.dat *.csv);;All files (*.*)",
        )
        for path_str in paths:
            try:
                spectrum = load_focused_spectrum(Path(path_str))
            except Exception as exc:
                self._set_status(f"Failed to load {Path(path_str).name}: {exc}")
                continue
            self.spectra.append(spectrum)
            self.corrected_spectra.append(None)
            self.normalization_results.append(None)
            bank = f" bank {spectrum.bank_number}" if spectrum.bank_number is not None else ""
            run = f" run {spectrum.run_number}" if spectrum.run_number else ""
            self.spectrum_combo.addItem(f"{spectrum.source_path.name}{run}{bank}")
        if paths:
            self.spectrum_combo.setCurrentIndex(len(self.spectra) - 1)
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
            self._set_status(
                f"Loaded CeO2 calibration run {self.calibration_spectrum.run_number or ''} "
                f"bank {self.calibration_spectrum.bank_number or ''}."
            )
        except Exception as exc:
            self._set_status(f"Failed to load CeO2 calibration histogram: {exc}")

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
            self._set_status(f"Loaded calibration phase {self.calibration_phase.name} from {Path(path_str).name}.")
        except Exception as exc:
            self._set_status(f"Failed to load CeO2 phase: {exc}")

    def load_vanadium_spectrum_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load vanadium normalisation histogram",
            str(Path.cwd() / "Hist"),
            "Open GENIE histogram (*.his);;All files (*.*)",
        )
        if not path_str:
            return
        try:
            self.vanadium_spectrum = load_focused_spectrum(Path(path_str))
            self._set_status(
                f"Loaded vanadium run {self.vanadium_spectrum.run_number or ''} "
                f"bank {self.vanadium_spectrum.bank_number or ''}."
            )
        except Exception as exc:
            self._set_status(f"Failed to load vanadium histogram: {exc}")

    def refine_current_calibration(self) -> None:
        try:
            if self.calibration_spectrum is None:
                raise ValueError("Load a CeO2 calibration histogram first.")
            if self.calibration_phase is None:
                raise ValueError("Load a CeO2 phase EXP first.")
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
            self.calibration_result = refine_ceo2_calibration(self.calibration_spectrum, self.calibration_phase)
            self.latest_fit_curve = None
            accepted = len(self.calibration_result.accepted_peaks)
            self._set_status(
                f"Refined calibration from CeO2 run {self.calibration_result.run_number}: "
                f"DIFC={self.calibration_result.calibration.difc:.6g}, "
                f"TZERO={self.calibration_result.calibration.tzero:.6g}, "
                f"{accepted} peak(s), RMS={self.calibration_result.rms_residual_tof:.4g} TOF."
            )
            self.update_plot()
        except Exception as exc:
            self._set_status(f"Calibration failed: {exc}")

    def apply_current_normalisation(self) -> None:
        try:
            index = self.spectrum_combo.currentIndex()
            raw = self.current_raw_spectrum()
            if raw is None or index < 0:
                raise ValueError("Load a sample spectrum before normalising.")
            if self.vanadium_spectrum is None:
                raise ValueError("Load a vanadium normalisation histogram first.")
            result = apply_vanadium_normalization(raw, self.vanadium_spectrum, self.calibration_result)
            self.corrected_spectra[index] = result.corrected_spectrum
            self.normalization_results[index] = result
            self.latest_fit_curve = None
            self.data_combo.setCurrentIndex(1)
            self._set_status(
                f"Applied vanadium run {result.vanadium_run_number} to sample run {result.sample_run_number}: "
                f"{result.valid_bins} valid bin(s), {result.invalid_bins} invalid."
            )
            self.update_plot()
        except Exception as exc:
            self._set_status(f"Normalisation failed: {exc}")

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
        view_range = self._current_view_range_or_none(x_values)
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

    def _current_view_range_or_none(self, x_values: np.ndarray) -> Optional[tuple[float, float]]:
        min_text = self.view_min_edit.text().strip()
        max_text = self.view_max_edit.text().strip()
        if not min_text and not max_text:
            return None
        x_min = float(np.nanmin(x_values))
        x_max = float(np.nanmax(x_values))
        try:
            low = float(min_text) if min_text else x_min
            high = float(max_text) if max_text else x_max
        except ValueError:
            self._set_status("View x bounds must be numeric.")
            return None
        if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
            self._set_status("View x bounds must be finite and different.")
            return None
        return (low, high)

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

    def _fit_range_mask(self, x_values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        selected_range = self._current_range_or_none()
        if selected_range is None:
            selected_range = (float(np.min(x_values)), float(np.max(x_values)))
            self.range_min_edit.setText(f"{selected_range[0]:.6g}")
            self.range_max_edit.setText(f"{selected_range[1]:.6g}")
        mask = (x_values >= selected_range[0]) & (x_values <= selected_range[1])
        if int(np.count_nonzero(mask)) < 8:
            raise ValueError("Selected range contains fewer than 8 data points.")
        return mask, selected_range

    def fit_selected_range(self) -> None:
        spectrum = self.current_spectrum()
        if spectrum is None:
            self._set_status("Load a spectrum before fitting.")
            return
        try:
            axis = self._selected_axis()
            x_values, _label = self._axis_data(spectrum)
            mask, selected_range = self._fit_range_mask(x_values)
            e_values = spectrum.e[mask] if spectrum.e is not None else None
            fit_mode = str(self.fit_combo.currentData())
            correction_cells = self._correction_cells(spectrum)
            if fit_mode == "pawley":
                if self.phase is None:
                    raise ValueError("Load a GSAS EXP phase before Pawley fitting.")
                result = fit_pawley(
                    x_values[mask],
                    spectrum.y[mask],
                    e_values,
                    self.phase,
                    axis,
                    spectrum.calibration,
                    polynomial_order=int(self.poly_order.value()),
                )
                reflection_details = "; ".join(
                    f"{item.reflection.label}@{item.position:.6g}:I={item.intensity:.6g}"
                    for item in result.reflections
                )
                self._append_result_row(
                    [
                        "Pawley",
                        spectrum.source_path.name,
                        spectrum.run_number or "",
                        str(spectrum.bank_number or ""),
                        axis,
                        f"{selected_range[0]:.6g}",
                        f"{selected_range[1]:.6g}",
                        f"{result.lattice_a:.8g}",
                        f"{result.fwhm:.6g}",
                        f"{sum(item.intensity for item in result.reflections):.6g}",
                        f"{result.eta:.4g}",
                        f"{result.quality.reduced_chi_square:.6g}",
                        f"{result.quality.rwp_percent:.4g}",
                        *correction_cells,
                        reflection_details,
                    ]
                )
                self._store_latest_fit_curve(result.fit_x, result.fit_y, result.observed_y, axis, "Pawley fit")
                self._set_status(f"Pawley fit complete: a={result.lattice_a:.6g}, Rwp={result.quality.rwp_percent:.4g}%.")
            else:
                result = fit_pseudo_voigt(
                    x_values[mask],
                    spectrum.y[mask],
                    e_values,
                    polynomial_order=int(self.poly_order.value()),
                )
                self._append_result_row(
                    [
                        "Pseudo-Voigt",
                        spectrum.source_path.name,
                        spectrum.run_number or "",
                        str(spectrum.bank_number or ""),
                        axis,
                        f"{selected_range[0]:.6g}",
                        f"{selected_range[1]:.6g}",
                        f"{result.centre:.8g}",
                        f"{result.fwhm:.6g}",
                        f"{result.area:.6g}",
                        f"{result.eta:.4g}",
                        f"{result.quality.reduced_chi_square:.6g}",
                        f"{result.quality.rwp_percent:.4g}",
                        *correction_cells,
                        "background=" + ",".join(f"{value:.6g}" for value in result.background_coefficients),
                    ]
                )
                self._store_latest_fit_curve(result.fit_x, result.fit_y, result.observed_y, axis, "Pseudo-Voigt fit")
                self._set_status(
                    f"Pseudo-Voigt fit complete: centre={result.centre:.6g}, Rwp={result.quality.rwp_percent:.4g}%."
                )
            self.update_plot()
        except Exception as exc:
            self._set_status(f"Fit failed: {exc}")

    def _append_result_row(self, values: list[str]) -> None:
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        for column, value in enumerate(values):
            self.results_table.setItem(row, column, QTableWidgetItem(value))
        self.results_table.resizeColumnsToContents()

    def _correction_cells(self, spectrum: FocusedSpectrum) -> list[str]:
        metadata = spectrum.metadata
        if "normalization_run" in metadata:
            correction = "calibrated+normalised" if "calibration_run" in metadata else "normalised"
        elif "calibration_run" in metadata:
            correction = "calibrated"
        else:
            correction = "raw"
        calibration = spectrum.calibration
        return [
            correction,
            "" if metadata.get("calibration_run") is None else str(metadata.get("calibration_run")),
            "" if metadata.get("normalization_run") is None else str(metadata.get("normalization_run")),
            "" if calibration is None else f"{calibration.difc:.8g}",
            "" if calibration is None else f"{calibration.tzero:.8g}",
            "" if calibration is None else f"{calibration.difa:.8g}",
        ]

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
                writer.writerow(RESULT_HEADERS)
                for row in range(self.results_table.rowCount()):
                    writer.writerow(
                        [
                            self.results_table.item(row, column).text()
                            if self.results_table.item(row, column) is not None
                            else ""
                            for column in range(self.results_table.columnCount())
                        ]
                    )
            self._set_status(f"Exported diffraction results to {path_str}.")
        except Exception as exc:
            self._set_status(f"Export failed: {exc}")
