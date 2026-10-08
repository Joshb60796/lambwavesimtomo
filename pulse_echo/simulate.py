"""Run one full firing sequence and invert the nine-cell damage index."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dispersion import LambCurve, lamb_curve
from .features import TraceFeatures, extract_trace
from .geometry import Coupon, make_coupon
from .material import Aluminum6061, Drive
from .signals import A0_REFERENCE_VOLT, CROSSTALK_FRACTION, Timebase, propagate_pair, synthesize_baseline_scales
from .tomography import RayRecord, Reconstruction, path_matrix_for_pair, reconstruct


@dataclass
class SimConfig:
    width_m: float = 0.090
    n_per_side: int = 6
    frequency_hz: float = 550e3
    voltage: float = 15.0
    n_cycles: int = 5
    hole_diameter_m: float = 0.012
    snr_db: float = 32.0
    shadow: float = 2.3
    scatter: float = 0.08
    coda_fraction: float = 0.6
    fs_hz: float = 10e6
    n_samples: int = 4096
    seed: int = 7
    fusion_weights: dict[str, float] = field(
        default_factory=lambda: {"amplitude": 1.0, "tof": 0.35, "phase": 0.20, "fwhm": 0.10}
    )


@dataclass
class Campaign:
    config: SimConfig
    plate: Aluminum6061
    drive: Drive
    curve: LambCurve
    coupon: Coupon
    time_s: np.ndarray
    excitation: np.ndarray
    holes: tuple[tuple[int, int], ...]
    # waveforms[shot, receiver, sample]. The transmitter channel holds the drive.
    waveforms: np.ndarray
    baseline: np.ndarray
    features: list[list[TraceFeatures]]
    baseline_features: list[list[TraceFeatures]]
    rays: list[RayRecord]
    reconstruction: Reconstruction
    noise_sigma: float

    @property
    def n_shots(self) -> int:
        return len(self.coupon.fire_order)


def _hole_list(coupon: Coupon, holes: set[tuple[int, int]], diameter_m: float) -> list[tuple[tuple[float, float], float]]:
    radius = max(0.0, diameter_m) / 2.0
    return [(center, radius) for center, _cell in coupon.hole_centers(holes)]


def _pair_waveform(
    coupon: Coupon,
    tx: int,
    rx: int,
    holes: list[tuple[tuple[float, float], float]],
    spectrum: np.ndarray,
    curve: LambCurve,
    timebase: Timebase,
    a0_scale: float,
    s0_scale: float,
    shadow: float,
    scatter: float,
    coda_delay_s: float = 0.0,
    coda_fraction: float = 0.0,
) -> np.ndarray:
    start = np.array([coupon.transducers[tx].x, coupon.transducers[tx].y])
    end = np.array([coupon.transducers[rx].x, coupon.transducers[rx].y])
    return propagate_pair(
        start, end, holes, spectrum, curve, timebase, a0_scale, s0_scale,
        shadow, scatter, coda_delay_s, coda_fraction,
    )


def run_campaign(config: SimConfig | None = None, holes: set[tuple[int, int]] | None = None) -> Campaign:
    config = config or SimConfig()
    holes = set(holes or [])
    plate = Aluminum6061()
    drive = Drive(frequency_hz=config.frequency_hz, voltage=config.voltage, n_cycles=config.n_cycles)
    curve = lamb_curve(plate, drive_hz=config.frequency_hz)
    coupon = make_coupon(config.width_m, config.n_per_side)
    timebase = Timebase(config.fs_hz, config.n_samples)
    excitation, spectrum, a0_scale, s0_scale = synthesize_baseline_scales(curve, timebase, drive, coupon.width_m)
    hole_models = _hole_list(coupon, holes, config.hole_diameter_m)

    n_tx = coupon.n_transducers
    n_shots = len(coupon.fire_order)
    pristine = np.zeros((n_shots, n_tx, timebase.n_samples))
    damaged = np.zeros_like(pristine)
    for shot, tx in enumerate(coupon.fire_order):
        pristine[shot, tx] = excitation
        damaged[shot, tx] = excitation
        for rx in range(n_tx):
            if rx == tx:
                continue
            coda_delay = 0.34 * drive.duration_s
            pristine[shot, rx] = _pair_waveform(
                coupon, tx, rx, [], spectrum, curve, timebase, a0_scale, s0_scale,
                config.shadow, config.scatter, coda_delay, config.coda_fraction,
            )
            damaged[shot, rx] = _pair_waveform(
                coupon, tx, rx, hole_models, spectrum, curve, timebase, a0_scale, s0_scale,
                config.shadow, config.scatter, coda_delay, config.coda_fraction,
            )
            crosstalk = CROSSTALK_FRACTION * excitation
            pristine[shot, rx] += crosstalk
            damaged[shot, rx] += crosstalk

    # SNR is referred to the calibrated pristine A0 peak on a path of length `width`.
    noise_sigma = A0_REFERENCE_VOLT / (10.0 ** (config.snr_db / 20.0))
    rng = np.random.default_rng(config.seed)
    noise = rng.normal(0.0, noise_sigma, size=damaged.shape)
    for shot, tx in enumerate(coupon.fire_order):
        noise[shot, tx] = 0.0
    measured = damaged + noise

    def featurize(cube: np.ndarray) -> list[list[TraceFeatures]]:
        table: list[list[TraceFeatures]] = []
        for shot, tx in enumerate(coupon.fire_order):
            row: list[TraceFeatures] = []
            origin = coupon.transducers[tx]
            for rx in range(n_tx):
                if rx == tx:
                    row.append(TraceFeatures.invalid(0.0))
                    continue
                other = coupon.transducers[rx]
                distance = float(np.hypot(other.x - origin.x, other.y - origin.y))
                sigma = 0.0 if cube is pristine else noise_sigma
                row.append(
                    extract_trace(
                        cube[shot, rx],
                        timebase.time_s,
                        timebase.fs_hz,
                        distance,
                        drive,
                        curve.a0_group_at_drive,
                        curve.s0_group_at_drive,
                        sigma,
                    )
                )
            table.append(row)
        return table

    baseline_features = featurize(pristine)
    features = featurize(measured)
    rays: list[RayRecord] = []
    for shot, tx in enumerate(coupon.fire_order):
        for rx in range(n_tx):
            if rx == tx:
                continue
            origin = coupon.transducers[tx]
            other = coupon.transducers[rx]
            distance = float(np.hypot(other.x - origin.x, other.y - origin.y))
            rays.append(
                RayRecord(
                    shot=shot,
                    transmitter=tx,
                    receiver=rx,
                    distance_m=distance,
                    path_m=path_matrix_for_pair(coupon, tx, rx),
                    measured=features[shot][rx],
                    baseline=baseline_features[shot][rx],
                )
            )
    image = reconstruct(coupon, rays, drive.frequency_hz, config.fusion_weights)
    return Campaign(
        config=config,
        plate=plate,
        drive=drive,
        curve=curve,
        coupon=coupon,
        time_s=timebase.time_s,
        excitation=excitation,
        holes=tuple(sorted(holes)),
        waveforms=measured,
        baseline=pristine,
        features=features,
        baseline_features=baseline_features,
        rays=rays,
        reconstruction=image,
        noise_sigma=noise_sigma,
    )
