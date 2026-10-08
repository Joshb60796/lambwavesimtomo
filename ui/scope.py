"""Phosphor A-scan and the per-receiver sparkline strip for one shot."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from pulse_echo.features import TraceFeatures, envelope

from .theme import AMBER, MUTED, PHOSPHOR, TEAL, mono_font, ui_font


def _decimate(values: np.ndarray, limit: int = 1200) -> np.ndarray:
    step = max(1, values.size // limit)
    return values[::step]


class ScopeView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(250)
        self.time_s = np.zeros(2)
        self.wave = np.zeros(2)
        self.env = np.zeros(2)
        self.feature = TraceFeatures.invalid(0.0)
        self.baseline = TraceFeatures.invalid(0.0)
        self.t_max = 80e-6
        self.title = "A-SCAN"
        self.drive = False
        self.f0 = 550e3

    def set_trace(
        self,
        time_s: np.ndarray,
        wave: np.ndarray,
        feature: TraceFeatures,
        baseline: TraceFeatures,
        t_max: float,
        title: str,
        drive: bool,
        frequency_hz: float,
    ) -> None:
        self.time_s = time_s
        self.wave = wave
        self.env = envelope(wave) if wave.size > 8 and not drive else np.zeros_like(wave)
        self.feature = feature
        self.baseline = baseline
        self.t_max = t_max
        self.title = title
        self.drive = drive
        self.f0 = frequency_hz
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#10120c"))
        plot = QRectF(54, 32, max(10.0, self.width() * 0.64 - 54), max(10.0, self.height() - 56))
        self._grid(painter, plot)
        self._trace(painter, plot)
        self._hud(painter, QRectF(plot.right() + 16, 18, self.width() - plot.right() - 28, self.height() - 30))
        painter.setFont(ui_font(12, weight=600))
        painter.setPen(QColor(AMBER))
        painter.drawText(12, 16, "A-SCAN")
        painter.setPen(QColor(MUTED))
        painter.setFont(mono_font(10))
        painter.drawText(72, 16, self.title)
        painter.end()

    def _grid(self, painter: QPainter, plot: QRectF) -> None:
        painter.fillRect(plot, QColor("#0c0e0a"))
        painter.setPen(QPen(QColor(48, 56, 36, 180), 1))
        for i in range(1, 4):
            y = plot.top() + plot.height() * i / 4.0
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        for i in range(1, 6):
            x = plot.left() + plot.width() * i / 6.0
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        painter.setPen(QPen(QColor("#243024"), 1))
        painter.drawRect(plot)
        painter.setFont(mono_font(9))
        painter.setPen(QColor(MUTED))
        for i in range(7):
            t_us = self.t_max * 1e6 * i / 6.0
            x = plot.left() + plot.width() * i / 6.0
            painter.drawText(QRectF(x - 24, plot.bottom() + 2, 48, 14), Qt.AlignmentFlag.AlignCenter, f"{t_us:.0f}")
        painter.drawText(QRectF(plot.right() - 36, plot.bottom() + 14, 40, 14), Qt.AlignmentFlag.AlignRight, "µs")

    def _window(self) -> np.ndarray:
        return (self.time_s >= 0.0) & (self.time_s <= self.t_max)

    def _trace(self, painter: QPainter, plot: QRectF) -> None:
        mask = self._window()
        if not np.any(mask):
            return
        samples = self.wave[mask]
        times = self.time_s[mask]
        peak = float(np.max(np.abs(samples))) if samples.size else 1.0
        if self.drive:
            peak = max(peak, 15.0)
        span = max(peak * 1.25, 0.05)
        times = _decimate(times)
        samples = _decimate(samples)

        def point(t: float, volts: float) -> QPointF:
            x = plot.left() + plot.width() * (t / self.t_max)
            y = plot.center().y() - plot.height() * 0.5 * (volts / span)
            return QPointF(x, y)

        zero_y = plot.center().y()
        painter.setPen(QPen(QColor(198, 214, 140, 50), 1))
        painter.drawLine(QPointF(plot.left(), zero_y), QPointF(plot.right(), zero_y))
        glow = QPen(QColor(198, 220, 120, 55), 3.2)
        glow.setCosmetic(True)
        painter.setPen(glow)
        for i in range(1, len(samples)):
            painter.drawLine(point(times[i - 1], samples[i - 1]), point(times[i], samples[i]))
        painter.setPen(QPen(QColor(PHOSPHOR), 1.25))
        for i in range(1, len(samples)):
            painter.drawLine(point(times[i - 1], samples[i - 1]), point(times[i], samples[i]))
        if self.drive or self.env.size != self.wave.size:
            painter.setFont(mono_font(9))
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(2, plot.top() - 2, 48, 14), Qt.AlignmentFlag.AlignRight, f"{span:.2f} V")
            return
        env = _decimate(self.env[mask])
        painter.setPen(QPen(QColor(226, 161, 34, 160), 1.0, Qt.PenStyle.DashLine))
        for i in range(1, len(env)):
            painter.drawLine(point(times[i - 1], env[i - 1]), point(times[i], env[i]))
            painter.drawLine(point(times[i - 1], -env[i - 1]), point(times[i], -env[i]))
        self._marker(painter, plot, point, self.feature.s0_peak_time_s, self.feature.s0_peak_v, QColor(TEAL), "S0")
        self._marker(painter, plot, point, self.feature.a0_peak_time_s, self.feature.a0_peak_v, QColor(AMBER), "A0")
        self._fwhm(painter, plot, point)
        painter.setFont(mono_font(9))
        painter.setPen(QColor(MUTED))
        painter.drawText(QRectF(2, plot.top() - 2, 48, 14), Qt.AlignmentFlag.AlignRight, f"{span:.2f} V")
        painter.drawText(QRectF(2, plot.bottom() - 12, 48, 14), Qt.AlignmentFlag.AlignRight, f"-{span:.2f}")

    def _marker(self, painter, plot, point, time_s, volts, color, name) -> None:
        if not np.isfinite(time_s) or time_s < 0 or time_s > self.t_max:
            return
        top = point(time_s, 0.0)
        painter.setPen(QPen(color, 1))
        painter.drawLine(QPointF(top.x(), plot.top()), QPointF(top.x(), plot.bottom()))
        painter.setFont(mono_font(9))
        painter.drawText(QRectF(top.x() + 3, plot.top() + 2, 28, 12), Qt.AlignmentFlag.AlignLeft, name)

    def _fwhm(self, painter, plot, point) -> None:
        feature = self.feature
        if not np.isfinite(feature.a0_fwhm_s) or not np.isfinite(feature.a0_peak_time_s) or not np.isfinite(feature.a0_peak_v):
            return
        half = 0.5 * feature.a0_peak_v
        t0 = feature.a0_peak_time_s - 0.5 * feature.a0_fwhm_s
        t1 = feature.a0_peak_time_s + 0.5 * feature.a0_fwhm_s
        painter.setPen(QPen(QColor(226, 161, 34, 200), 1.4))
        painter.drawLine(point(t0, half), point(t1, half))

    def _hud(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor("#34382c"), 1))
        painter.setBrush(QColor("#181b14"))
        painter.drawRect(rect)
        painter.setFont(ui_font(12, weight=600))
        painter.setPen(QColor(AMBER))
        painter.drawText(QRectF(rect.left() + 10, rect.top() + 8, rect.width() - 16, 16), "EXTRACTED")
        lines = self._lines()
        painter.setFont(mono_font(11))
        y = rect.top() + 32
        for label, value, color in lines:
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(rect.left() + 10, y, rect.width() * 0.46, 16), label)
            painter.setPen(QColor(color))
            painter.drawText(QRectF(rect.left() + rect.width() * 0.46, y, rect.width() * 0.5, 16), value)
            y += 18

    def _lines(self) -> list[tuple[str, str, str]]:
        if self.drive:
            return [
                ("CHANNEL", "PULSER", PHOSPHOR),
                ("DRIVE", "±15 V", AMBER),
                ("WINDOW", "HANNING", "#e7e1d4"),
                ("CARRIER", "550 kHz", "#e7e1d4"),
            ]
        feature = self.feature
        base = self.baseline

        def us(value: float) -> str:
            return f"{value * 1e6:6.2f} µs" if np.isfinite(value) else "      —"

        def delta_us(value: float, reference: float) -> str:
            if not np.isfinite(value) or not np.isfinite(reference):
                return "      —"
            return f"{(value - reference) * 1e6:+6.2f} µs"

        phase = feature.a0_phase_rad
        loss = "      —"
        if np.isfinite(feature.a0_peak_v) and np.isfinite(base.a0_peak_v) and base.a0_peak_v > 0:
            loss = f"{-20.0 * np.log10(max(feature.a0_peak_v, 1e-8) / base.a0_peak_v):+6.1f} dB"
        dphi = "      —"
        if np.isfinite(phase) and np.isfinite(base.a0_phase_rad):
            dphi = f"{np.angle(np.exp(1j * (phase - base.a0_phase_rad))):+6.2f} rad"
        return [
            ("A0 TOF", us(feature.a0_tof_s), PHOSPHOR),
            ("ΔTOF", delta_us(feature.a0_tof_s, base.a0_tof_s), AMBER),
            ("A0 FWHM", us(feature.a0_fwhm_s), "#e7e1d4"),
            ("ΔFWHM", delta_us(feature.a0_fwhm_s, base.a0_fwhm_s), "#e7e1d4"),
            ("PHASE", f"{phase:+6.2f} rad" if np.isfinite(phase) else "      —", "#e7e1d4"),
            ("ΔPHASE", dphi, TEAL),
            ("LOSS", loss, "#d4653c" if loss != "      —" else MUTED),
            ("S0 TOF", us(feature.s0_tof_s), TEAL),
            ("A0 PEAK", f"{feature.a0_peak_v:6.3f} V" if np.isfinite(feature.a0_peak_v) else "      —", PHOSPHOR),
        ]


class SparkStrip(QWidget):
    picked = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(78)
        self.setMaximumHeight(96)
        self.time_s = np.zeros(2)
        self.traces: list[np.ndarray] = []
        self.labels: list[str] = []
        self.tx = -1
        self.monitor = -1
        self.t_max = 80e-6

    def set_shot(
        self,
        time_s: np.ndarray,
        traces: list[np.ndarray],
        labels: list[str],
        tx: int,
        monitor: int,
        t_max: float,
    ) -> None:
        self.time_s = time_s
        self.traces = traces
        self.labels = labels
        self.tx = tx
        self.monitor = monitor
        self.t_max = t_max
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self.traces:
            return
        width = self.width() / len(self.traces)
        index = int(event.position().x() / max(width, 1))
        index = min(max(index, 0), len(self.traces) - 1)
        self.picked.emit(index)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#161910"))
        if not self.traces:
            painter.setPen(QColor(MUTED))
            painter.setFont(ui_font(12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Receivers appear when a shot is armed")
            painter.end()
            return
        count = len(self.traces)
        gap = 4
        width = (self.width() - gap * (count + 1)) / count
        mask = (self.time_s >= 0.0) & (self.time_s <= self.t_max)
        for i, trace in enumerate(self.traces):
            rect = QRectF(gap + i * (width + gap), 6, width, self.height() - 12)
            active = i == self.monitor
            transmitter = i == self.tx
            painter.fillRect(rect, QColor("#2a2614" if active else "#0e100c"))
            painter.setPen(QPen(QColor(AMBER if active else "#2c3126"), 1))
            painter.drawRect(rect)
            samples = _decimate(trace[mask] if np.any(mask) else trace, 200)
            if samples.size < 2:
                continue
            peak = max(float(np.max(np.abs(samples))), 1e-4)
            if transmitter:
                peak = max(peak, 15.0)

            def point(k: int, rect: QRectF = rect, samples: np.ndarray = samples, peak: float = peak) -> QPointF:
                x = rect.left() + rect.width() * k / (len(samples) - 1)
                y = rect.center().y() - rect.height() * 0.42 * (samples[k] / peak)
                return QPointF(x, y)

            color = QColor(AMBER) if transmitter else QColor(PHOSPHOR)
            painter.setPen(QPen(color, 1))
            for k in range(1, len(samples)):
                painter.drawLine(point(k - 1), point(k))
            painter.setFont(mono_font(8))
            painter.setPen(QColor("#f3ecdf" if active else MUTED))
            painter.drawText(QRectF(rect.left() + 3, rect.top() + 1, rect.width() - 4, 12), self.labels[i])
        painter.end()
