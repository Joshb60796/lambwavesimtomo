"""Straight-ray SIRT images of the nine cells from differential A0 features.

Each crossing ray supplies path length through the cells. Amplitude loss,
group-delay, phase delay, and envelope broadening are inverted separately and
then fused into a damage index on a 0–1 scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import TraceFeatures, phase_delay_s
from .geometry import N_CELLS, Coupon, aperture_offsets, grid_path_lengths


@dataclass(frozen=True)
class RayRecord:
    shot: int
    transmitter: int
    receiver: int
    distance_m: float
    path_m: np.ndarray  # shape (3, 3)
    measured: TraceFeatures
    baseline: TraceFeatures


@dataclass(frozen=True)
class Reconstruction:
    amplitude_db: np.ndarray
    tof_us: np.ndarray
    phase_us: np.ndarray
    fwhm_us: np.ndarray
    fused: np.ndarray
    hole_calls: np.ndarray
    n_rays: int
    weights: dict[str, float]


def sirt(kernel: np.ndarray, data: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """Simultaneous iterative reconstruction. `kernel` is (rays, cells), flattened row-major."""
    x = np.zeros(kernel.shape[1], dtype=float)
    ray_norm = np.sum(kernel, axis=1)
    cell_norm = np.sum(kernel, axis=0)
    usable = ray_norm > 1e-9
    cell_norm = np.maximum(cell_norm, 1e-12)
    for _ in range(n_iter):
        residual = np.zeros_like(data)
        predicted = kernel @ x
        residual[usable] = (data[usable] - predicted[usable]) / ray_norm[usable]
        x += (kernel.T @ residual) / cell_norm
        np.maximum(x, 0.0, out=x)
    return x


def path_matrix_for_pair(coupon: Coupon, transmitter: int, receiver: int) -> np.ndarray:
    """Aperture-averaged path length (m) through each cell."""
    tx = coupon.transducers[transmitter]
    rx = coupon.transducers[receiver]
    start = np.array([tx.x, tx.y])
    end = np.array([rx.x, rx.y])
    delta = end - start
    distance = float(np.hypot(delta[0], delta[1]))
    if distance < 1e-9:
        return np.zeros((N_CELLS, N_CELLS))
    normal = np.array([-delta[1], delta[0]]) / distance
    offsets, weights = aperture_offsets()
    acc = np.zeros((N_CELLS, N_CELLS))
    for offset, weight in zip(offsets, weights):
        acc += weight * grid_path_lengths(tuple(start + normal * offset), tuple(end + normal * offset), coupon.width_m)
    return acc


def _saturate(value: np.ndarray, scale: float) -> np.ndarray:
    return 1.0 - np.exp(-np.maximum(value, 0.0) / scale)


def reconstruct(
    coupon: Coupon,
    rays: list[RayRecord],
    frequency_hz: float,
    weights: dict[str, float] | None = None,
    call_threshold: float = 0.42,
) -> Reconstruction:
    weights = weights or {"amplitude": 1.0, "tof": 0.35, "phase": 0.20, "fwhm": 0.10}
    usable = [ray for ray in rays if float(np.sum(ray.path_m)) > 0.15 * coupon.cell_m and ray.measured.valid and ray.baseline.valid]
    n_cells = N_CELLS * N_CELLS
    if not usable:
        zeros = np.zeros((N_CELLS, N_CELLS))
        return Reconstruction(zeros, zeros, zeros, zeros, zeros, np.zeros_like(zeros, dtype=bool), 0, weights)

    kernel_rows = []
    amp_rows = []
    tof_rows = []
    phase_rows = []
    fwhm_rows = []
    row_weights = []
    for ray in usable:
        base_peak = ray.baseline.a0_peak_v
        meas_peak = ray.measured.a0_peak_v
        if not np.isfinite(base_peak) or base_peak <= 1e-8 or not np.isfinite(meas_peak):
            continue
        loss_np = -np.log(np.clip(meas_peak / base_peak, 1e-4, None))
        dtof = ray.measured.a0_tof_s - ray.baseline.a0_tof_s
        if not np.isfinite(dtof):
            continue
        dphase = phase_delay_s(ray.measured.a0_phase_rad, ray.baseline.a0_phase_rad, frequency_hz, dtof)
        dfwhm = ray.measured.a0_fwhm_s - ray.baseline.a0_fwhm_s
        kernel_rows.append(ray.path_m.reshape(-1))
        amp_rows.append(max(0.0, float(loss_np)))
        tof_rows.append(max(0.0, float(dtof)))
        phase_rows.append(max(0.0, float(dphase)) if np.isfinite(dphase) else 0.0)
        fwhm_rows.append(max(0.0, float(dfwhm)) if np.isfinite(dfwhm) else 0.0)
        row_weights.append(base_peak)

    if not kernel_rows:
        zeros = np.zeros((N_CELLS, N_CELLS))
        return Reconstruction(zeros, zeros, zeros, zeros, zeros, np.zeros_like(zeros, dtype=bool), 0, weights)

    kernel = np.vstack(kernel_rows)
    w = np.asarray(row_weights, dtype=float)
    w *= len(w) / np.sum(w)
    kernel_w = kernel * w[:, None]

    def invert(data: list[float]) -> np.ndarray:
        vector = np.asarray(data, dtype=float) * w
        image = sirt(kernel_w, vector).reshape(N_CELLS, N_CELLS)
        return image

    # Attenuation image is Np/m. Convert a one-cell transit to dB for display.
    alpha = invert(amp_rows)
    slowness = invert(tof_rows)
    phase_slowness = invert(phase_rows)
    breadth = invert(fwhm_rows)
    cell = coupon.cell_m
    amplitude_db = alpha * cell * (20.0 / np.log(10.0))
    tof_us = slowness * cell * 1e6
    phase_us = phase_slowness * cell * 1e6
    fwhm_us = breadth * cell * 1e6

    # A 12 mm through-hole casts about 6 dB of straight-ray amplitude shadow
    # in one cell. Its Keller creeping delay is a fraction of a microsecond,
    # so time and phase must not be asked to carry the call by themselves.
    i_amp = _saturate(amplitude_db, 3.0)
    i_tof = _saturate(tof_us, 0.40)
    i_phase = _saturate(phase_us, 0.40)
    i_fwhm = _saturate(fwhm_us, 0.70)
    wsum = sum(max(0.0, weights[name]) for name in ("amplitude", "tof", "phase", "fwhm"))
    if wsum <= 0.0:
        wsum = 1.0
    fused = (
        weights["amplitude"] * i_amp
        + weights["tof"] * i_tof
        + weights["phase"] * i_phase
        + weights["fwhm"] * i_fwhm
    ) / wsum
    return Reconstruction(
        amplitude_db=amplitude_db,
        tof_us=tof_us,
        phase_us=phase_us,
        fwhm_us=fwhm_us,
        fused=np.clip(fused, 0.0, 1.0),
        hole_calls=fused >= call_threshold,
        n_rays=len(kernel_rows),
        weights=weights,
    )
