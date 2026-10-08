"""Top view of the coupon, the two PWAS columns, and the live wavefront."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from pulse_echo.geometry import PWAS_DIAMETER_M, Coupon

from .theme import AMBER, MUTED, OXIDE, PHOSPHOR, mono_font, ui_font


class PlateView(QWidget):
    holeToggled = Signal(int, int)
    receiverPicked = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(440, 480)
        self.coupon: Coupon | None = None
        self.holes: set[tuple[int, int]] = set()
        self.diameter_m = 0.012
        self.tx: int | None = None
        self.monitor: int | None = None
        self.rays: list[tuple[float, float, float, float, float | None]] = []
        self.ring = 0.0
        self.playing = False
        self.cg_s0 = 5307.0
        self.cg_a0 = 2818.0
        self.hover: tuple[int, int] | None = None
        self._ox = 0.0
        self._oy = 0.0
        self._scale = 1.0

    def set_scene(
        self,
        coupon: Coupon,
        holes: set[tuple[int, int]],
        diameter_m: float,
        tx: int | None,
        monitor: int | None,
        rays: list[tuple[float, float, float, float, float | None]],
        ring: float,
        playing: bool,
        cg_s0: float,
        cg_a0: float,
    ) -> None:
        self.coupon = coupon
        self.holes = set(holes)
        self.diameter_m = diameter_m
        self.tx = tx
        self.monitor = monitor
        self.rays = rays
        self.ring = ring
        self.playing = playing
        self.cg_s0 = cg_s0
        self.cg_a0 = cg_a0
        self.update()

    def set_ring(self, ring: float) -> None:
        self.ring = ring
        self.update()

    def _layout(self) -> None:
        coupon = self.coupon
        assert coupon is not None
        span = coupon.plate_width_m
        avail_w = max(1.0, self.width() - 16.0)
        avail_h = max(1.0, self.height() - 46.0)
        self._scale = min(avail_w, avail_h) / span
        drawn = span * self._scale
        # Map plate coordinates so x = -margin is the left edge and y = width + margin is the top.
        inset_x = 8.0 + (avail_w - drawn) / 2.0
        inset_y = 26.0 + (avail_h - drawn) / 2.0
        self._ox = inset_x + coupon.margin_m * self._scale
        self._oy = inset_y + coupon.margin_m * self._scale

    def _to_px(self, x: float, y: float) -> QPointF:
        coupon = self.coupon
        assert coupon is not None
        px = self._ox + x * self._scale
        py = self._oy + (coupon.width_m - y) * self._scale
        return QPointF(px, py)

    def _from_px(self, px: float, py: float) -> tuple[float, float]:
        coupon = self.coupon
        assert coupon is not None
        x = (px - self._ox) / self._scale
        y = coupon.width_m - (py - self._oy) / self._scale
        return x, y

    def _cell_at(self, x: float, y: float) -> tuple[int, int] | None:
        coupon = self.coupon
        if coupon is None:
            return None
        if x < 0.0 or y < 0.0 or x >= coupon.width_m or y >= coupon.width_m:
            return None
        cell = coupon.cell_m
        col = min(2, max(0, int(x / cell)))
        row = min(2, max(0, int((coupon.width_m - y) / cell)))
        return row, col

    def _transducer_at(self, px: float, py: float) -> int | None:
        if self.coupon is None:
            return None
        radius = max(8.0, 0.5 * PWAS_DIAMETER_M * self._scale + 4.0)
        best = None
        best_d = radius
        for transducer in self.coupon.transducers:
            point = self._to_px(transducer.x, transducer.y)
            dist = math.hypot(point.x() - px, point.y() - py)
            if dist <= best_d:
                best = transducer.index
                best_d = dist
        return best

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.coupon is None:
            return
        self._layout()
        x, y = self._from_px(event.position().x(), event.position().y())
        hover = self._cell_at(x, y)
        if hover != self.hover:
            self.hover = hover
            self.update()
        on_target = hover is not None or self._transducer_at(event.position().x(), event.position().y()) is not None
        self.setCursor(Qt.CursorShape.PointingHandCursor if on_target else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.coupon is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self._layout()
        px = event.position().x()
        py = event.position().y()
        transducer = self._transducer_at(px, py)
        if transducer is not None:
            self.receiverPicked.emit(transducer)
            return
        x, y = self._from_px(px, py)
        cell = self._cell_at(x, y)
        if cell is not None:
            self.holeToggled.emit(*cell)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1c1f17"))
        painter.setFont(ui_font(12, weight=600))
        painter.setPen(QColor(AMBER))
        painter.drawText(12, 18, "COUPON")
        painter.setFont(mono_font(10))
        painter.setPen(QColor(MUTED))
        plate_mm = self.coupon.plate_width_m * 1e3 if self.coupon is not None else 0.0
        painter.drawText(78, 18, f"6061-T6   1/32 IN   PLATE {plate_mm:.0f} mm")
        if self.coupon is None:
            painter.end()
            return
        self._layout()
        self._paint_sheet(painter)
        self._paint_roi(painter)
        self._paint_rays(painter)
        self._paint_rings(painter)
        self._paint_holes(painter)
        self._paint_transducers(painter)
        self._paint_caption(painter)
        painter.end()

    def _paint_sheet(self, painter: QPainter) -> None:
        coupon = self.coupon
        assert coupon is not None
        xmin, ymin, xmax, ymax = coupon.plate_bounds()
        top_left = self._to_px(xmin, ymax)
        bottom_right = self._to_px(xmax, ymin)
        rect = QRectF(top_left, bottom_right).normalized()
        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0.0, QColor("#b7b2a6"))
        gradient.setColorAt(0.45, QColor("#8e8a80"))
        gradient.setColorAt(1.0, QColor("#6d6a62"))
        painter.setPen(QPen(QColor("#4a483f"), 1.2))
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QPen(QColor(40, 38, 32, 28), 1))
        y = rect.top() + 3
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left() + 2, y), QPointF(rect.right() - 2, y))
            y += 3

    def _paint_roi(self, painter: QPainter) -> None:
        coupon = self.coupon
        assert coupon is not None
        origin = self._to_px(0.0, coupon.width_m)
        side = coupon.width_m * self._scale
        rect = QRectF(origin.x(), origin.y(), side, side)
        painter.fillRect(rect, QColor(28, 32, 24, 210))
        cell = side / 3.0
        painter.setPen(QPen(QColor(186, 180, 166, 150), 1))
        for i in range(1, 3):
            painter.drawLine(QPointF(rect.left() + i * cell, rect.top()), QPointF(rect.left() + i * cell, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), rect.top() + i * cell), QPointF(rect.right(), rect.top() + i * cell))
        if self.hover is not None:
            row, col = self.hover
            hover = QRectF(rect.left() + col * cell, rect.top() + row * cell, cell, cell)
            painter.fillRect(hover, QColor(226, 161, 34, 36))
        painter.setPen(QPen(QColor("#d9d3c6"), 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

    def _paint_rays(self, painter: QPainter) -> None:
        for x0, y0, x1, y1, ratio in self.rays:
            if ratio is None:
                color = QColor(200, 196, 184, 50)
            elif ratio < 0.45:
                color = QColor(212, 101, 60, 170)
            elif ratio < 0.8:
                color = QColor(226, 161, 34, 140)
            else:
                color = QColor(198, 214, 140, 60)
            painter.setPen(QPen(color, 1.15))
            painter.drawLine(self._to_px(x0, y0), self._to_px(x1, y1))

    def _paint_rings(self, painter: QPainter) -> None:
        if self.tx is None or self.ring <= 0.01 or self.coupon is None:
            return
        transducer = self.coupon.transducers[self.tx]
        center = self._to_px(transducer.x, transducer.y)
        xmin, ymin, xmax, ymax = self.coupon.plate_bounds()
        far = max(
            math.hypot(x - transducer.x, y - transducer.y)
            for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
        )
        far_px = far * self._scale
        reach = self.ring * far_px
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(127, 175, 163, 150), 1.4))
        s0 = reach * (self.cg_s0 / max(self.cg_a0, 1.0))
        if s0 < far_px * 1.05:
            painter.drawEllipse(center, s0, s0)
        painter.setPen(QPen(QColor(226, 161, 34, 190), 1.6))
        painter.drawEllipse(center, reach, reach)

    def _paint_holes(self, painter: QPainter) -> None:
        if self.coupon is None:
            return
        radius = max(2.0, 0.5 * self.diameter_m * self._scale)
        for row, col in self.holes:
            center = self._to_px(*self.coupon.cell_center(row, col))
            gradient = QRadialGradient(center, radius)
            gradient.setColorAt(0.0, QColor("#070605"))
            gradient.setColorAt(0.72, QColor("#1a120e"))
            gradient.setColorAt(1.0, QColor("#3a2418"))
            painter.setPen(QPen(QColor(OXIDE), 1.3))
            painter.setBrush(gradient)
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QPen(QColor(255, 236, 210, 90), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(
                QRectF(center.x() - radius * 0.62, center.y() - radius * 0.62, radius * 1.24, radius * 1.24),
                40 * 16,
                70 * 16,
            )

    def _paint_transducers(self, painter: QPainter) -> None:
        if self.coupon is None:
            return
        radius = max(6.0, 0.5 * PWAS_DIAMETER_M * self._scale)
        painter.setFont(ui_font(11, weight=600))
        for transducer in self.coupon.transducers:
            center = self._to_px(transducer.x, transducer.y)
            active = transducer.index == self.tx
            listening = transducer.index == self.monitor
            if active:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(226, 161, 34, 50))
                painter.drawEllipse(center, radius + 7, radius + 7)
            disc = QRadialGradient(center, radius)
            disc.setColorAt(0.0, QColor("#f4f1e8"))
            disc.setColorAt(0.55, QColor("#d9c7a4"))
            disc.setColorAt(1.0, QColor("#8d7048"))
            painter.setBrush(disc)
            if active:
                painter.setPen(QPen(QColor(AMBER), 1.6))
            elif listening:
                painter.setPen(QPen(QColor(PHOSPHOR), 1.5))
            else:
                painter.setPen(QPen(QColor("#3a3124"), 1))
            painter.drawEllipse(center, radius, radius)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2a241c"))
            painter.drawEllipse(center, radius * 0.28, radius * 0.28)
            if transducer.ordinal == 1:
                box = QRectF(center.x() - 18, center.y() - radius - 18, 36, 16)
                align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
            elif transducer.ordinal == self.coupon.n_per_side:
                box = QRectF(center.x() - 18, center.y() + radius + 1, 36, 16)
                align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
            elif transducer.side == "L":
                box = QRectF(center.x() - radius - 40, center.y() - 9, 34, 18)
                align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            else:
                box = QRectF(center.x() + radius + 6, center.y() - 9, 34, 18)
                align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            painter.setPen(QColor("#f7f1e6") if active or listening else QColor("#2a261f"))
            painter.drawText(box, align, transducer.label)

    def _paint_caption(self, painter: QPainter) -> None:
        painter.setFont(ui_font(12))
        painter.setPen(QColor(MUTED))
        text = "Click a cell to drill or fill. Click a sensor to pin the A-scan."
        if self.playing:
            text = "Wavefront live.  Teal ring is S0, amber ring is A0."
        if self.hover is not None and self.coupon is not None:
            row, col = self.hover
            state = "drilled" if (row, col) in self.holes else "clear"
            text = f"Cell {row + 1},{col + 1}  {state}. Click to toggle."
        painter.drawText(QRectF(12, self.height() - 22, self.width() - 24, 16), text)
