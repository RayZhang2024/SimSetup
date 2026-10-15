from __future__ import annotations

from typing import Mapping

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QTabWidget, QTextEdit, QVBoxLayout, QWidget

from .normalization import _smooth_response
from .raw_workflow import RawCalibrationBundle


def _plot(parent: QWidget, *, residuals: bool = False):
    figure = Figure(figsize=(9, 5), tight_layout=True)
    if residuals:
        axes = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        main, lower = axes
    else:
        main = figure.subplots()
        lower = None
    canvas = FigureCanvas(figure)
    parent.layout().addWidget(canvas, 1)
    return canvas, main, lower


class _BankPreview(QDialog):
    def __init__(self, parent, title: str, bundle: RawCalibrationBundle, diagnostics: Mapping[int, Mapping[str, object]]):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1080, 720)
        self.setStyleSheet(
            "QDialog, QTabWidget::pane { background: #f7f8fa; color: #202a32; }"
            "QLabel { color: #202a32; }"
            "QTabBar::tab { background: #e5e9ec; color: #202a32; padding: 7px 14px; }"
            "QTabBar::tab:selected { background: #ffffff; }"
            "QComboBox, QTextEdit { background: #ffffff; color: #202a32; border: 1px solid #bfc8cf; }"
            "QComboBox QAbstractItemView { background: #ffffff; color: #202a32; }"
        )
        self.bundle = bundle
        self.diagnostics = diagnostics
        self.tabs = QTabWidget(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

    def _page(self, bank: int) -> QWidget:
        page = QWidget(self)
        QVBoxLayout(page)
        self.tabs.addTab(page, f"Bank {bank}")
        return page


class OpenBeamPreview(_BankPreview):
    def __init__(self, parent, bundle: RawCalibrationBundle, diagnostics: Mapping[int, Mapping[str, object]]):
        super().__init__(parent, "Open-beam response preview", bundle, diagnostics)
        for record in bundle.banks:
            bank = record.bank_number
            page = self._page(bank)
            reference = diagnostics[bank]["focused_reference"]
            edges = np.asarray(reference.x, dtype=float)
            centres = reference.x_centres
            widths = np.diff(edges)
            charge = float(reference.metadata["proton_charge"])
            unsmoothed = np.asarray(reference.y, dtype=float) / charge
            smoothed_density, available = _smooth_response(unsmoothed / widths)
            canvas, axes, _ = _plot(page)
            axes.plot(centres, unsmoothed, color="#2877ae", lw=0.8, label="Focused open beam (unsmoothed)")
            if available:
                axes.plot(centres, smoothed_density * widths, color="#cf6d28", lw=1.2, label="Smoothed preview")
            axes.set_xlabel("Time of flight (us)")
            axes.set_ylabel("Reference response (counts/uAh per bin)")
            axes.set_title(f"Bank {bank}: {record.bank_name}")
            axes.legend(loc="best")
            canvas.draw()
            applied = "smoothed" if record.reference_response_settings.get("smoothed") else "unsmoothed"
            note = QLabel(
                f"Applied response: {applied}. Smoothed curve is comparison only when smoothing is off. "
                f"Charge: {charge:.8g} uAh; valid bins: {sum(record.response_valid_mask)}."
            )
            note.setWordWrap(True)
            page.layout().addWidget(note)


class Ceo2FitPreview(_BankPreview):
    def __init__(self, parent, bundle: RawCalibrationBundle, diagnostics: Mapping[int, Mapping[str, object]]):
        super().__init__(parent, "CeO2 fit review", bundle, diagnostics)
        self._selectors: dict[int, QComboBox] = {}
        self._local_axes = {}
        for record in bundle.banks:
            bank = record.bank_number
            page = self._page(bank)
            result = diagnostics[bank]["calibration"]
            spectrum = diagnostics[bank]["corrected_ceo2"]
            pattern_canvas, pattern_axes, _ = _plot(page)
            pattern_axes.plot(spectrum.x_centres, spectrum.y, color="#2877ae", lw=0.8, label="Open-beam-normalized CeO2")
            for peak in result.peak_results:
                if peak.accepted and peak.fitted_tof is not None:
                    pattern_axes.axvline(peak.fitted_tof, color="#22805c", alpha=0.55, lw=0.9)
                    pattern_axes.text(peak.fitted_tof, 0.98, peak.reflection.label, rotation=90,
                                      transform=pattern_axes.get_xaxis_transform(), ha="right", va="top", fontsize=7)
            pattern_axes.set_xlabel("Time of flight (us)")
            pattern_axes.set_ylabel("CeO2 / open beam (dimensionless)")
            pattern_axes.set_title(f"Bank {bank}: matched CeO2 reflections")
            pattern_axes.legend(loc="best")
            pattern_canvas.draw()

            row = QHBoxLayout()
            row.addWidget(QLabel("Reflection"))
            selector = QComboBox(page)
            for index, peak in enumerate(result.peak_results):
                state = "accepted" if peak.accepted else f"rejected: {peak.rejection_reason}"
                selector.addItem(f"{peak.reflection.label}  d={peak.reflection.d_spacing:.6g} A  ({state})", index)
            row.addWidget(selector, 1)
            page.layout().addLayout(row)
            local_canvas, local_axes, residual_axes = _plot(page, residuals=True)
            self._selectors[bank] = selector
            self._local_axes[bank] = (local_canvas, local_axes, residual_axes)
            selector.currentIndexChanged.connect(lambda _index, b=bank: self._draw_local(b))
            self._draw_local(bank)

    def _draw_local(self, bank: int) -> None:
        result = self.diagnostics[bank]["calibration"]
        peak = result.peak_results[int(self._selectors[bank].currentData())]
        fit = self.diagnostics[bank]["local_peak_fits"].get(peak.reflection.label)
        canvas, axes, residual_axes = self._local_axes[bank]
        axes.clear()
        residual_axes.clear()
        if fit is not None and fit.fit_x is not None and fit.fit_y is not None:
            x = np.asarray(fit.fit_x, dtype=float)
            y = np.asarray(fit.observed_y, dtype=float)
            model = np.asarray(fit.fit_y, dtype=float)
            x_normalized = (x - 0.5 * (x.min() + x.max())) / max(0.5 * (x.max() - x.min()), 1e-12)
            background = np.polynomial.polynomial.polyval(x_normalized, fit.background_coefficients)
            axes.plot(x, y, ".", color="#2877ae", markersize=3, label="Measured pattern")
            axes.plot(x, model, color="#bf513c", lw=1.4, label=f"{fit.model_name} fit")
            axes.plot(x, background, "--", color="#5e676e", lw=1, label="Linear background")
            axes.axvline(peak.fitted_tof, color="#22805c", lw=0.9, label="Fitted centre")
            residual_axes.plot(x, y - model, color="#394b59", lw=0.8)
            axes.legend(loc="best")
        else:
            axes.text(0.5, 0.5, f"No local fit: {peak.rejection_reason}", ha="center", va="center", transform=axes.transAxes)
        residual_axes.axhline(0, color="#888888", lw=0.8)
        axes.set_ylabel("CeO2 / open beam")
        residual_axes.set_ylabel("Residual")
        residual_axes.set_xlabel("Time of flight (us)")
        centre = f"{peak.fitted_tof:.4f} us" if peak.fitted_tof is not None else "unavailable"
        uncertainty = f"{peak.tof_uncertainty:.4f} us" if peak.tof_uncertainty is not None else "unavailable"
        axes.set_title(f"{peak.reflection.label}: centre {centre}; uncertainty {uncertainty}")
        canvas.draw()


class LinearCalibrationPreview(_BankPreview):
    def __init__(self, parent, bundle: RawCalibrationBundle, diagnostics: Mapping[int, Mapping[str, object]]):
        super().__init__(parent, "Linear calibration fit review", bundle, diagnostics)
        for record in bundle.banks:
            bank = record.bank_number
            page = self._page(bank)
            result = diagnostics[bank]["calibration"]
            accepted = [peak for peak in result.peak_results if peak.accepted and peak.fitted_tof is not None]
            rejected = [peak for peak in result.peak_results if not peak.accepted]
            d = np.asarray([peak.reflection.d_spacing for peak in accepted], dtype=float)
            centres = np.asarray([peak.fitted_tof for peak in accepted], dtype=float)
            sigma = np.asarray([
                max(float(peak.tof_uncertainty), 1e-6) if peak.tof_uncertainty is not None else 1.0
                for peak in accepted
            ])
            line_d = np.linspace(float(d.min()), float(d.max()), 300)
            fitted_line = record.difc * line_d + record.tzero
            residual = centres - (record.difc * d + record.tzero)
            canvas, axes, residual_axes = _plot(page, residuals=True)
            axes.errorbar(d, centres, yerr=sigma, fmt="o", ms=4, color="#2877ae", capsize=2,
                          label="Accepted fitted peak centres (TOF uncertainty)")
            axes.plot(line_d, fitted_line, color="#bf513c", lw=1.4, label="Weighted linear fit")
            for peak in rejected:
                if peak.fitted_tof is not None:
                    axes.plot(peak.reflection.d_spacing, peak.fitted_tof, "x", color="#777777")
            residual_axes.errorbar(d, residual, yerr=sigma, fmt="o", ms=4, color="#2877ae", capsize=2)
            residual_axes.axhline(0, color="#888888", lw=0.8)
            axes.set_ylabel("Fitted centre (us)")
            residual_axes.set_ylabel("Residual (us)")
            residual_axes.set_xlabel("Known CeO2 d-spacing (A)")
            axes.set_title(
                f"Bank {bank}: DIFC={record.difc:.8g} us/A, TZERO={record.tzero:.8g} us, "
                f"RMS={record.rms_residual_tof:.5g} us, DIFA=0"
            )
            axes.legend(loc="best")
            canvas.draw()
            status = QTextEdit(page)
            status.setReadOnly(True)
            status.setMaximumHeight(110)
            status.setPlainText("\n".join(
                f"{peak.reflection.label}: {'accepted' if peak.accepted else 'rejected (' + peak.rejection_reason + ')'}; "
                f"d={peak.reflection.d_spacing:.7g} A; centre={peak.fitted_tof} us; "
                f"uncertainty={peak.tof_uncertainty} us; residual={peak.residual_tof} us"
                for peak in result.peak_results
            ))
            page.layout().addWidget(status)
