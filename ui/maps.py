"""Fused damage index and the four feature images that feed it."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget
from scipy.ndimage import zoom

from pulse_echo.tomography import Reconstruction

from .theme import AMBER, MUTED, ui_font

_STOPS = np.array([0.0, 0.34, 0.67, 1.0])
_COLORS = np.array(
    [
        [20, 22, 16],
        [62, 72, 46],
        [226, 161, 34],
        [212, 96, 56],
    ],
    dtype=float,
)


def colorize(values: np.ndarray) -> QImage:
    flat = np.clip(values, 0.0, 1.0)
    rgb = np.empty(flat.shape + (3,), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = np.interp(flat, _STOPS, _COLORS[:, channel]).astype(np.uint8)
    alpha = np.full(flat.shape + (1,), 255, dtype=np.uint8)
    packed = np.concatenate([rgb, alpha], axis=-1)
    image = QImage(packed.data, packed.shape[1], packed.shape[0], packed.strides[0], QImage.Format.Format_RGBA8888)
    return image.copy()


def _upsample(values: np.ndarray, factor: int) -> np.ndarray:
    return zoom(values, factor, order=1)


class MapCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(360, 420)
        self.reconstruction: Reconstruction | None = None
        self.acquired = False
        self._fused = QImage()
        self._small: list[QImage] = []

    def set_reconstruction(self, reconstruction: Reconstruction | None, acquired: bool) -> None:
        self.reconstruction = reconstruction
        self.acquired = acquired and reconstruction is not None
        self._fused = QImage()
        self._small = []
        if self.acquired and reconstruction is not None:
            self._fused = colorize(_upsample(reconstruction.fused, 48))
            channels = (
                reconstruction.amplitude_db / 8.0,
                reconstruction.tof_us / 1.0,
                reconstruction.phase_us / 1.0,
                reconstruction.fwhm_us / 0.8,
            )
            self._small = [colorize(_upsample(np.clip(channel, 0.0, 1.0), 28)) for channel in channels]
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1c1f17"))
        painter.setFont(ui_font(12, weight=600))
        painter.setPen(QColor(AMBER))
        painter.drawText(12, 18, "RECONSTRUCTION")
        if not self.acquired or self.reconstruction is None:
            painter.setFont(ui_font(16, weight=600))
            painter.setPen(QColor("#e7e1d4"))
            painter.drawText(self.rect().adjusted(20, 0, -20, -40), Qt.AlignmentFlag.AlignCenter, "WAITING FOR THE SEQUENCE")
            painter.setFont(ui_font(13))
            painter.setPen(QColor(MUTED))
            painter.drawText(
                self.rect().adjusted(28, 36, -28, -28),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                "Time of flight, width, phase, and amplitude\nare inverted after the last transducer fires.",
            )
            painter.end()
            return
        side = min(self.width() - 28, self.height() - 196)
        side = max(120.0, side)
        left = (self.width() - side) / 2.0
        fused = QRectF(left, 32, side, side)
        self._draw_fused(painter, fused)
        names = ("LOSS dB", "TOF us", "PHASE us", "FWHM us")
        gap = 8.0
        strip_w = self.width() - 24
        small = min(72.0, (strip_w - 3 * gap) / 4.0)
        strip_left = (self.width() - (4 * small + 3 * gap)) / 2.0
        strip_top = fused.bottom() + 36
        for i, image in enumerate(self._small):
            rect = QRectF(strip_left + i * (small + gap), strip_top, small, small)
            painter.setPen(QColor(MUTED))
            painter.setFont(ui_font(11, weight=600))
            painter.drawText(QRectF(rect.left(), rect.top() - 14, rect.width(), 13), Qt.AlignmentFlag.AlignCenter, names[i])
            if not image.isNull():
                painter.drawImage(rect, image)
            painter.setPen(QPen(QColor("#d9d3c6"), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
        bar = QRectF(fused.left() + 52, strip_top + small + 12, max(40.0, fused.width() - 104), 8)
        self._colorbar(painter, bar)
        painter.setFont(ui_font(11, weight=600))
        painter.setPen(QColor(MUTED))
        painter.drawText(QRectF(bar.left() - 50, bar.top() - 3, 46, 14), Qt.AlignmentFlag.AlignRight, "CLEAR")
        painter.drawText(QRectF(bar.right() + 6, bar.top() - 3, 46, 14), "HOLE")
        calls = int(np.sum(self.reconstruction.hole_calls))
        painter.setFont(ui_font(12))
        painter.drawText(
            QRectF(12, bar.bottom() + 10, self.width() - 24, 18),
            Qt.AlignmentFlag.AlignCenter,
            f"{calls} cell call{'s' if calls != 1 else ''}    {self.reconstruction.n_rays} rays    threshold 0.42",
        )
        painter.end()

    def _draw_fused(self, painter: QPainter, rect: QRectF) -> None:
        assert self.reconstruction is not None
        if not self._fused.isNull():
            painter.drawImage(rect, self._fused)
        painter.setPen(QPen(QColor("#d9d3c6"), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        cell_w = rect.width() / 3.0
        cell_h = rect.height() / 3.0
        painter.setPen(QPen(QColor(244, 239, 228, 80), 1))
        for i in range(1, 3):
            painter.drawLine(rect.left() + i * cell_w, rect.top(), rect.left() + i * cell_w, rect.bottom())
            painter.drawLine(rect.left(), rect.top() + i * cell_h, rect.right(), rect.top() + i * cell_h)
        values = self.reconstruction.fused
        for row in range(3):
            for col in range(3):
                value = float(values[row, col])
                called = bool(self.reconstruction.hole_calls[row, col])
                box = QRectF(rect.left() + col * cell_w, rect.top() + row * cell_h, cell_w, cell_h)
                painter.setPen(QColor("#1a140c") if value > 0.55 else QColor("#f4efe4"))
                painter.setFont(ui_font(13 if called else 12, weight=600))
                label = f"{value:.2f}"
                painter.drawText(box.adjusted(0, -8, 0, 0), Qt.AlignmentFlag.AlignCenter, label)
                if called:
                    painter.setFont(ui_font(11, weight=600))
                    painter.drawText(box.adjusted(0, 14, 0, 0), Qt.AlignmentFlag.AlignCenter, "HOLE")
                    painter.setPen(QPen(QColor("#f4efe4"), 1.3))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(box.adjusted(7, 7, -7, -7))
        painter.setPen(QColor("#f4efe4"))
        painter.setFont(ui_font(12, weight=600))
        painter.drawText(QRectF(rect.left() + 6, rect.top() + 3, rect.width() - 12, 16), "FUSED INDEX")

    def _colorbar(self, painter: QPainter, rect: QRectF) -> None:
        ramp = colorize(np.linspace(0.0, 1.0, 256)[None, :])
        painter.drawImage(rect, ramp)


class MapBoard(QWidget):
    weightsChanged = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.canvas = MapCanvas()
        self._sliders: dict[str, QSlider] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, stretch=1)
        row = QHBoxLayout()
        row.setContentsMargins(10, 0, 10, 8)
        defaults = (("amplitude", "AMP", 100), ("tof", "TOF", 35), ("phase", "PHASE", 20), ("fwhm", "FWHM", 10))
        for key, label, value in defaults:
            caption = QLabel(label)
            caption.setObjectName("muted")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(value)
            slider.valueChanged.connect(self._emit)
            self._sliders[key] = slider
            row.addWidget(caption)
            row.addWidget(slider, stretch=1)
        layout.addLayout(row)
        note = QLabel("Fusion weights. The inverse itself is straight-ray SIRT on the four A0 residuals.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        wrap = QVBoxLayout()
        wrap.setContentsMargins(12, 0, 12, 8)
        wrap.addWidget(note)
        layout.addLayout(wrap)

    def set_reconstruction(self, reconstruction: Reconstruction | None, acquired: bool) -> None:
        self.canvas.set_reconstruction(reconstruction, acquired)

    def weights(self) -> dict[str, float]:
        return {key: slider.value() / 100.0 for key, slider in self._sliders.items()}

    def _emit(self) -> None:
        self.weightsChanged.emit(self.weights())
