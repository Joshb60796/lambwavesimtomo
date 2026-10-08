"""Bench window: sequence the pulsers, watch the A-scans, read the nine-cell image."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pulse_echo.dispersion import lamb_curve
from pulse_echo.features import TraceFeatures
from pulse_echo.geometry import Coupon, make_coupon
from pulse_echo.material import Aluminum6061, Drive
from pulse_echo.simulate import SimConfig, run_campaign
from pulse_echo.tomography import reconstruct

from .maps import MapBoard
from .plate import PlateView
from .scope import ScopeView, SparkStrip
from .theme import AMBER, MUTED, load_fonts, mono_font, stylesheet, ui_font

COLUMNS = ["SHOT", "TX", "RX", "TOF us", "dTOF", "FWHM", "dFWHM", "dPHASE", "LOSS dB"]


class LabeledSlider(QWidget):
    def __init__(self, name: str, low: int, high: int, value: int, suffix: str = "") -> None:
        super().__init__()
        self.suffix = suffix
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(low, high)
        self.slider.setValue(value)
        self.name = QLabel(name)
        self.name.setObjectName("muted")
        self.readout = QLabel()
        self.readout.setObjectName("readout")
        self.readout.setMinimumWidth(52)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.name)
        row.addWidget(self.slider, stretch=1)
        row.addWidget(self.readout)
        self.slider.valueChanged.connect(lambda _v: self._sync())
        self._sync()

    def _sync(self) -> None:
        self.readout.setText(f"{self.slider.value()}{self.suffix}")

    def value(self) -> int:
        return self.slider.value()


class ShotPips(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.captured = 0
        self.active = -1
        self.labels: list[str] = []
        self.setFixedHeight(28)

    def set_state(self, labels: list[str], captured: int, active: int) -> None:
        self.labels = labels
        self.n = len(labels)
        self.captured = captured
        self.active = active
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1c1f17"))
        if self.n == 0:
            painter.end()
            return
        gap = 4
        width = min(36, (self.width() - gap * (self.n + 1)) / self.n)
        total = self.n * width + (self.n - 1) * gap
        x = max(8, (self.width() - total) / 2)
        painter.setFont(mono_font(8))
        for i, label in enumerate(self.labels):
            rect_x = x + i * (width + gap)
            if i < self.captured:
                color = QColor(AMBER)
            elif i == self.active:
                color = QColor("#f3ddb0")
            else:
                color = QColor("#3a3f32")
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(rect_x, 4, width, 8, 1, 1)
            painter.setPen(QColor("#f4efe4") if i == self.active else QColor(MUTED))
            painter.drawText(QRect(rect_x - 2, 13, width + 4, 12), Qt.AlignmentFlag.AlignCenter, label)
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lamb Bench — Pulse-Echo Tomography")
        self.resize(1480, 940)
        self.setMinimumSize(1180, 760)
        self.curve = lamb_curve(Aluminum6061(), drive_hz=550e3)
        self.drive = Drive()
        self.holes: set[tuple[int, int]] = {(1, 1)}
        self.campaign = None
        self.dirty = True
        self.shot = 0
        self.captured = 0
        self.revealed = False
        self.playing = False
        self.monitor: int | None = None
        self.pinned = False
        self.seed = 7
        self._busy = False
        self._rows: list[tuple[int, int]] = []
        self._dwell_into = 0
        self._ring = 0.0
        self._build()
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)
        self._refresh_plate()
        self._status("Center cell is drilled. Run the sequence to capture the pitch-catch set.")

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 6)
        root.setSpacing(8)
        root.addLayout(self._header())

        self.plate = PlateView()
        self.plate.holeToggled.connect(self._toggle_hole)
        self.plate.receiverPicked.connect(self._pick_receiver)
        self.scope = ScopeView()
        self.sparks = SparkStrip()
        self.sparks.picked.connect(self._pick_receiver)
        self.maps = MapBoard()
        self.maps.weightsChanged.connect(self._refit)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellClicked.connect(self._on_table)
        self.pips = ShotPips()

        left = QWidget()
        left.setObjectName("panel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)
        left_layout.addWidget(self.plate, stretch=1)
        left_layout.addWidget(self.pips)
        left_layout.addLayout(self._presets())
        left_layout.addWidget(self._sliders())
        left_layout.addLayout(self._buttons())

        center = QWidget()
        center.setObjectName("panel")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(6)
        center_layout.addWidget(self.scope, stretch=3)
        center_layout.addWidget(self.sparks)
        center_layout.addWidget(self.table, stretch=2)

        right = QWidget()
        right.setObjectName("panel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.maps)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        splitter.setStretchFactor(2, 4)
        splitter.setSizes([500, 580, 400])
        root.addWidget(splitter, stretch=1)
        self.setStatusBar(QStatusBar())

    def _header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        kicker = QLabel("PITCH-CATCH  ·  GUIDED WAVE")
        kicker.setObjectName("kicker")
        title = QLabel("LAMB BENCH")
        title.setObjectName("title")
        subtitle = QLabel("Fire down the left column, then up the right. Every other channel listens.")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(kicker)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        row.addLayout(title_box)
        row.addStretch(1)
        chips = QHBoxLayout()
        chips.setSpacing(6)
        a0 = f"A0  {self.curve.a0_phase_at_drive:.0f} / {self.curve.a0_group_at_drive:.0f} m/s"
        s0 = f"S0  {self.curve.s0_phase_at_drive:.0f} / {self.curve.s0_group_at_drive:.0f} m/s"
        fd = 550e3 * Aluminum6061().thickness_m * 1e-3
        for text in ("6061-T6", '1/32"', "550 kHz", "±15 V", f"fd {fd:.3f} MHz·mm", a0, s0):
            chip = QLabel(text)
            chip.setObjectName("chip")
            chips.addWidget(chip)
        row.addLayout(chips)
        return row

    def _presets(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        for name, holes in (
            ("CLEAR", set()),
            ("CENTER", {(1, 1)}),
            ("DIAGONAL", {(0, 0), (1, 1), (2, 2)}),
            ("CORNERS", {(0, 0), (0, 2), (2, 0), (2, 2)}),
            ("EDGE", {(0, 1), (1, 0), (1, 2), (2, 1)}),
        ):
            button = QPushButton(name)
            button.setObjectName("preset")
            button.clicked.connect(lambda _checked=False, chosen=holes: self._apply_preset(chosen))
            row.addWidget(button)
        return row

    def _sliders(self) -> QWidget:
        box = QWidget()
        grid = QVBoxLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        self.diameter = LabeledSlider("HOLE DIA", 4, 18, 12, " mm")
        self.snr = LabeledSlider("SNR", 12, 60, 42, " dB")
        self.cycles = LabeledSlider("CYCLES", 3, 8, 5)
        self.width_mm = LabeledSlider("SPAN", 70, 160, 90, " mm")
        self.n_side = LabeledSlider("ELEMENTS", 3, 8, 6)
        self.speed = LabeledSlider("PACE", 1, 10, 4)
        for slider in (self.diameter, self.snr, self.cycles, self.width_mm, self.n_side):
            slider.slider.valueChanged.connect(self._mark_dirty)
        cycle_row = QWidget()
        cycle_layout = QHBoxLayout(cycle_row)
        cycle_layout.setContentsMargins(0, 0, 0, 0)
        cycle_layout.setSpacing(4)
        cycle_layout.addWidget(self.cycles, stretch=1)
        for count in (3, 5):
            button = QPushButton(f"{count}-COUNT")
            button.setObjectName("preset")
            button.setToolTip(f"{count}-cycle Hanning tone burst at 550 kHz")
            button.clicked.connect(lambda _checked=False, n=count: self.cycles.slider.setValue(n))
            cycle_layout.addWidget(button)
        grid.addWidget(self.diameter)
        grid.addWidget(self.snr)
        grid.addWidget(cycle_row)
        grid.addWidget(self.width_mm)
        grid.addWidget(self.n_side)
        grid.addWidget(self.speed)
        return box

    def _buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.run_button = QPushButton("RUN SEQUENCE")
        self.run_button.setObjectName("primary")
        self.step_button = QPushButton("STEP")
        self.instant_button = QPushButton("INSTANT")
        self.export_button = QPushButton("EXPORT")
        self.reseed_button = QPushButton("NEW NOISE")
        self.run_button.clicked.connect(self._on_run)
        self.step_button.clicked.connect(self._on_step)
        self.instant_button.clicked.connect(self._on_instant)
        self.export_button.clicked.connect(self._export)
        self.reseed_button.clicked.connect(self._reseed)
        for button in (self.run_button, self.step_button, self.instant_button, self.export_button, self.reseed_button):
            row.addWidget(button)
        return row

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not self._busy:
            self._on_run()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right and not self._busy:
            self._on_step()
            event.accept()
            return
        super().keyPressEvent(event)

    def _coupon(self) -> Coupon:
        return make_coupon(self.width_mm.value() / 1000.0, self.n_side.value())

    def _config(self) -> SimConfig:
        return SimConfig(
            width_m=self.width_mm.value() / 1000.0,
            n_per_side=self.n_side.value(),
            n_cycles=self.cycles.value(),
            hole_diameter_m=self.diameter.value() / 1000.0,
            snr_db=float(self.snr.value()),
            seed=self.seed,
            fusion_weights=self.maps.weights(),
        )

    def _t_max(self) -> float:
        width = self.width_mm.value() / 1000.0
        return self.drive.envelope_center_s + width * 1.55 / self.curve.a0_group_at_drive + 30e-6

    def _dwell_ms(self) -> int:
        return int(1600 / max(1, self.speed.value()))

    def _tx_index(self) -> int | None:
        coupon = self._coupon() if self.campaign is None or self.dirty else self.campaign.coupon
        if self.captured <= 0 or not coupon.fire_order:
            return None
        shot = min(self.shot, len(coupon.fire_order) - 1)
        return coupon.fire_order[shot]

    def _status(self, text: str) -> None:
        speeds = (
            f"A0 cp {self.curve.a0_phase_at_drive:.0f}  cg {self.curve.a0_group_at_drive:.0f}    "
            f"S0 cp {self.curve.s0_phase_at_drive:.0f}  cg {self.curve.s0_group_at_drive:.0f}"
        )
        self.statusBar().showMessage(f"{text}     {speeds}     seed {self.seed}")

    def _mark_dirty(self) -> None:
        self.timer.stop()
        self.playing = False
        self.run_button.setText("RUN SEQUENCE")
        self.dirty = True
        self.captured = 0
        self.shot = 0
        self.pinned = False
        self.revealed = False
        self._ring = 0.0
        self._dwell_into = 0
        self.table.setRowCount(0)
        self._rows = []
        self.maps.set_reconstruction(None, False)
        self.sparks.set_shot(np.zeros(2), [], [], -1, -1, self._t_max())
        self._refresh_plate()
        self._status("Specimen changed. The previous capture is cleared.")

    def _apply_preset(self, holes: set[tuple[int, int]]) -> None:
        self.holes = set(holes)
        self._mark_dirty()

    def _toggle_hole(self, row: int, col: int) -> None:
        if (row, col) in self.holes:
            self.holes.discard((row, col))
        else:
            self.holes.add((row, col))
        self._mark_dirty()

    def _reseed(self) -> None:
        self.seed += 1
        self._mark_dirty()
        self._status(f"Noise draw {self.seed}. Run to recapture.")

    def _ensure(self) -> bool:
        if self.campaign is not None and not self.dirty:
            return True
        if self._busy:
            return False
        self._busy = True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status("Synthesizing the two-mode pitch-catch set…")
        QApplication.processEvents()
        try:
            self.campaign = run_campaign(self._config(), set(self.holes))
        except Exception as exc:  # surface solver or geometry failures in the status bar
            self._status(f"Synthesis failed: {exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
            self._busy = False
        self.dirty = False
        self.captured = 0
        self.shot = 0
        self.monitor = None
        self.revealed = False
        self._rebuild_table()
        self._status(
            f"Synthesized {self.campaign.n_shots} shots × {self.campaign.coupon.n_transducers - 1} listeners."
        )
        return True

    def _on_run(self) -> None:
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.run_button.setText("RUN SEQUENCE")
            self._status("Sequence paused.")
            return
        if not self._ensure() or self.campaign is None:
            return
        n = self.campaign.n_shots
        if self.captured >= n:
            self.shot = 0
        elif self.captured == 0:
            self.shot = 0
            self.captured = 1
        else:
            self.shot = self.captured
            self.captured += 1
        self.playing = True
        self._dwell_into = 0
        self._ring = 0.0
        self.run_button.setText("PAUSE")
        self._show_current()
        self.timer.start()

    def _on_step(self) -> None:
        self.playing = False
        self.timer.stop()
        self.run_button.setText("RUN SEQUENCE")
        self._ring = 0.0
        if not self._ensure() or self.campaign is None:
            return
        n = self.campaign.n_shots
        if self.captured < n:
            self.shot = self.captured
            self.captured += 1
        else:
            self.shot = (self.shot + 1) % n
        self._show_current()
        if self.captured >= n:
            self._reveal()

    def _on_instant(self) -> None:
        self.playing = False
        self.timer.stop()
        self.run_button.setText("RUN SEQUENCE")
        self._ring = 0.0
        if not self._ensure() or self.campaign is None:
            return
        self.captured = self.campaign.n_shots
        self.shot = self.captured - 1
        self._fill_all_rows()
        self._show_current()
        self._reveal()

    def _tick(self) -> None:
        if not self.playing or self.campaign is None:
            return
        self._dwell_into += self.timer.interval()
        self._ring = min(1.0, self._dwell_into / self._dwell_ms())
        self.plate.set_ring(self._ring)
        if self._dwell_into < self._dwell_ms():
            return
        self._dwell_into = 0
        self._ring = 0.0
        if self.shot + 1 >= self.campaign.n_shots:
            self.playing = False
            self.timer.stop()
            self.run_button.setText("RUN SEQUENCE")
            self._reveal()
            return
        self.shot += 1
        self.captured = max(self.captured, self.shot + 1)
        self._show_current()

    def _reveal(self) -> None:
        self.revealed = True
        if self.campaign is not None:
            self.maps.set_reconstruction(self.campaign.reconstruction, True)
        self._refresh_plate()
        calls = 0 if self.campaign is None else int(np.sum(self.campaign.reconstruction.hole_calls))
        self._status(f"Sequence complete. {calls} cell call{'s' if calls != 1 else ''} on the fused map.")

    def _refit(self) -> None:
        if self.campaign is None or self.dirty or not self.revealed:
            return
        self.campaign.reconstruction = reconstruct(
            self.campaign.coupon,
            self.campaign.rays,
            self.campaign.drive.frequency_hz,
            self.maps.weights(),
        )
        self.maps.set_reconstruction(self.campaign.reconstruction, True)

    def _pick_receiver(self, index: int) -> None:
        self.monitor = index
        self.pinned = True
        if self.captured > 0:
            self._show_current()
        else:
            self._refresh_plate()

    def _show_current(self) -> None:
        if self.campaign is None:
            return
        coupon = self.campaign.coupon
        tx = coupon.fire_order[self.shot]
        if not self.pinned or self.monitor is None or self.monitor == tx:
            self.monitor = _loudest_loss(self.campaign, self.shot, tx)
        self._fill_rows_until(self.captured)
        self._show_scope(tx)
        self._show_sparks(tx)
        self._select_table_row()
        self._refresh_plate()
        label = coupon.transducers[tx].label
        self._status(f"Shot {self.shot + 1}/{self.campaign.n_shots}  transmitter {label}.")

    def _show_scope(self, tx: int) -> None:
        assert self.campaign is not None and self.monitor is not None
        coupon = self.campaign.coupon
        monitor = self.monitor
        title = f"{coupon.transducers[tx].label}  to  {coupon.transducers[monitor].label}"
        if monitor == tx:
            self.scope.set_trace(
                self.campaign.time_s,
                self.campaign.excitation,
                TraceFeatures.invalid(0.0),
                TraceFeatures.invalid(0.0),
                self._t_max(),
                f"{coupon.transducers[tx].label}  PULSER MONITOR",
                True,
                self.drive.frequency_hz,
            )
            return
        self.scope.set_trace(
            self.campaign.time_s,
            self.campaign.waveforms[self.shot, monitor],
            self.campaign.features[self.shot][monitor],
            self.campaign.baseline_features[self.shot][monitor],
            self._t_max(),
            title,
            False,
            self.drive.frequency_hz,
        )

    def _show_sparks(self, tx: int) -> None:
        assert self.campaign is not None and self.monitor is not None
        coupon = self.campaign.coupon
        traces = [self.campaign.waveforms[self.shot, rx] for rx in range(coupon.n_transducers)]
        labels = [t.label for t in coupon.transducers]
        self.sparks.set_shot(self.campaign.time_s, traces, labels, tx, self.monitor, self._t_max())

    def _refresh_plate(self) -> None:
        coupon = self.campaign.coupon if self.campaign is not None and not self.dirty else self._coupon()
        tx = None
        if self.captured > 0 and self.campaign is not None and not self.dirty:
            tx = self.campaign.coupon.fire_order[min(self.shot, self.campaign.n_shots - 1)]
        rays = []
        if tx is not None and self.campaign is not None:
            origin = coupon.transducers[tx]
            for rx, other in enumerate(coupon.transducers):
                if rx == tx:
                    continue
                base = self.campaign.baseline_features[self.shot][rx].a0_peak_v
                meas = self.campaign.features[self.shot][rx].a0_peak_v
                ratio = float(meas / base) if np.isfinite(base) and np.isfinite(meas) and base > 0 else None
                rays.append((origin.x, origin.y, other.x, other.y, ratio))
        labels = [coupon.transducers[i].label for i in coupon.fire_order]
        self.pips.set_state(labels, self.captured if not self.dirty else 0, self.shot if self.playing else -1)
        self.plate.set_scene(
            coupon,
            self.holes,
            self.diameter.value() / 1000.0,
            tx,
            self.monitor,
            rays,
            self._ring if self.playing else 0.0,
            self.playing,
            self.curve.s0_group_at_drive,
            self.curve.a0_group_at_drive,
        )

    def _rebuild_table(self) -> None:
        assert self.campaign is not None
        coupon = self.campaign.coupon
        self._rows = []
        for shot, tx in enumerate(coupon.fire_order):
            for rx in range(coupon.n_transducers):
                if rx != tx:
                    self._rows.append((shot, rx))
        self.table.setRowCount(len(self._rows))
        for row, (shot, rx) in enumerate(self._rows):
            tx = coupon.fire_order[shot]
            values = [f"{shot + 1:02d}", coupon.transducers[tx].label, coupon.transducers[rx].label] + ["."] * (len(COLUMNS) - 3)
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setForeground(QColor("#6d6a60"))
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def _fill_rows_until(self, captured: int) -> None:
        if self.campaign is None:
            return
        for row, (shot, _rx) in enumerate(self._rows):
            if shot < captured:
                self._write_row(row)

    def _fill_all_rows(self) -> None:
        self._fill_rows_until(self.captured)

    def _write_row(self, row: int) -> None:
        assert self.campaign is not None
        shot, rx = self._rows[row]
        tx = self.campaign.coupon.fire_order[shot]
        measured = self.campaign.features[shot][rx]
        base = self.campaign.baseline_features[shot][rx]
        loss = _loss_db(measured.a0_peak_v, base.a0_peak_v)
        phase_delta = _phase_delta(measured.a0_phase_rad, base.a0_phase_rad)
        values = [
            f"{shot + 1:02d}",
            self.campaign.coupon.transducers[tx].label,
            self.campaign.coupon.transducers[rx].label,
            _us(measured.a0_tof_s),
            _dus(measured.a0_tof_s, base.a0_tof_s),
            _us(measured.a0_fwhm_s),
            _dus(measured.a0_fwhm_s, base.a0_fwhm_s),
            phase_delta,
            loss,
        ]
        tone = QColor("#e7e1d4")
        if loss not in {"·", ""}:
            try:
                if float(loss) > 6.0:
                    tone = QColor("#e2a122")
            except ValueError:
                tone = QColor("#e7e1d4")
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setForeground(tone)
            self.table.setItem(row, col, item)

    def _select_table_row(self) -> None:
        if self.monitor is None:
            return
        self.table.blockSignals(True)
        for row, (shot, rx) in enumerate(self._rows):
            if shot == self.shot and rx == self.monitor:
                self.table.selectRow(row)
                item = self.table.item(row, 0)
                if item is not None:
                    self.table.scrollToItem(item)
                break
        self.table.blockSignals(False)

    def _on_table(self, row: int, _col: int) -> None:
        if row < 0 or row >= len(self._rows) or self.campaign is None or self.dirty:
            return
        shot, rx = self._rows[row]
        if shot >= self.captured:
            return
        self.playing = False
        self.timer.stop()
        self.run_button.setText("RUN SEQUENCE")
        self.shot = shot
        self.monitor = rx
        self.pinned = True
        self._show_current()

    def _export(self) -> None:
        if self.campaign is None or self.dirty or not self.revealed:
            self._status("Capture the full sequence before exporting.")
            return
        default = str(Path.cwd() / "exports" / "lamb_campaign.npz")
        path, _selected = QFileDialog.getSaveFileName(self, "Export synthetic campaign", default, "NumPy archive (*.npz)")
        if not path:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        campaign = self.campaign
        meta = {
            "material": "6061-T6",
            "thickness_m": campaign.plate.thickness_m,
            "frequency_hz": campaign.drive.frequency_hz,
            "voltage": campaign.drive.voltage,
            "cycles": campaign.drive.n_cycles,
            "width_m": campaign.coupon.width_m,
            "plate_width_m": campaign.coupon.plate_width_m,
            "n_per_side": campaign.coupon.n_per_side,
            "hole_diameter_m": campaign.config.hole_diameter_m,
            "holes_row_col": campaign.holes,
            "snr_db": campaign.config.snr_db,
            "seed": campaign.config.seed,
            "a0_phase_m_s": campaign.curve.a0_phase_at_drive,
            "a0_group_m_s": campaign.curve.a0_group_at_drive,
            "s0_phase_m_s": campaign.curve.s0_phase_at_drive,
            "s0_group_m_s": campaign.curve.s0_group_at_drive,
            "fire_order": [campaign.coupon.transducers[i].label for i in campaign.coupon.fire_order],
        }
        positions = np.array([[t.x, t.y] for t in campaign.coupon.transducers])
        np.savez_compressed(
            target,
            time_s=campaign.time_s,
            excitation=campaign.excitation,
            waveforms=campaign.waveforms,
            baseline=campaign.baseline,
            transducer_xy=positions,
            fused=campaign.reconstruction.fused,
            amplitude_db=campaign.reconstruction.amplitude_db,
            tof_us=campaign.reconstruction.tof_us,
            phase_us=campaign.reconstruction.phase_us,
            fwhm_us=campaign.reconstruction.fwhm_us,
            metadata=np.array(json.dumps(meta)),
        )
        csv_path = target.with_suffix(".csv")
        export_columns = [
            "shot", "tx", "rx", "path_mm", "a0_tof_us", "dtof_us", "a0_fwhm_us", "dfwhm_us",
            "a0_phase_rad", "dphase_rad", "loss_db", "s0_tof_us", "a0_peak_v",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(export_columns)
            for shot, tx in enumerate(campaign.coupon.fire_order):
                for rx, other in enumerate(campaign.coupon.transducers):
                    if rx == tx:
                        continue
                    measured = campaign.features[shot][rx]
                    base = campaign.baseline_features[shot][rx]
                    writer.writerow(
                        [
                            shot + 1,
                            campaign.coupon.transducers[tx].label,
                            other.label,
                            f"{measured.distance_m * 1e3:.2f}",
                            _us(measured.a0_tof_s),
                            _dus(measured.a0_tof_s, base.a0_tof_s),
                            _us(measured.a0_fwhm_s),
                            _dus(measured.a0_fwhm_s, base.a0_fwhm_s),
                            _rad(measured.a0_phase_rad),
                            _phase_delta(measured.a0_phase_rad, base.a0_phase_rad),
                            _loss_db(measured.a0_peak_v, base.a0_peak_v),
                            _us(measured.s0_tof_s),
                            f"{measured.a0_peak_v:.4f}" if np.isfinite(measured.a0_peak_v) else "",
                        ]
                    )
        self._status(f"Wrote {target.name} and {csv_path.name}.")


def _loudest_loss(campaign, shot: int, transmitter: int) -> int:
    """Opposite-side listener with the greatest A0 amplitude loss on this shot."""
    coupon = campaign.coupon
    side = coupon.transducers[transmitter].side
    best = None
    best_ratio = 1e9
    for rx, transducer in enumerate(coupon.transducers):
        if rx == transmitter or transducer.side == side:
            continue
        base = campaign.baseline_features[shot][rx].a0_peak_v
        measured = campaign.features[shot][rx].a0_peak_v
        if not np.isfinite(base) or not np.isfinite(measured) or base <= 0:
            continue
        ratio = measured / base
        if ratio < best_ratio:
            best_ratio = ratio
            best = rx
    if best is not None:
        return best
    return (transmitter + coupon.n_per_side) % coupon.n_transducers


def _us(value: float) -> str:
    return f"{value * 1e6:.2f}" if np.isfinite(value) else "·"


def _dus(value: float, reference: float) -> str:
    if not np.isfinite(value) or not np.isfinite(reference):
        return "·"
    return f"{(value - reference) * 1e6:+.2f}"


def _rad(value: float) -> str:
    return f"{value:+.2f}" if np.isfinite(value) else "·"


def _phase_delta(value: float, reference: float) -> str:
    if not np.isfinite(value) or not np.isfinite(reference):
        return "·"
    return f"{np.angle(np.exp(1j * (value - reference))):+.2f}"


def _loss_db(measured: float, baseline: float) -> str:
    if not np.isfinite(measured) or not np.isfinite(baseline) or baseline <= 0:
        return "·"
    return f"{-20.0 * np.log10(max(measured, 1e-8) / baseline):+.1f}"


def main() -> None:
    app = QApplication.instance() or QApplication([])
    load_fonts()
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheet())
    app.setFont(ui_font(14))
    window = MainWindow()
    window.show()
    app.exec()
