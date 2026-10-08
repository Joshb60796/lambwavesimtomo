"""Time of flight, envelope FWHM, and carrier phase for S0 and A0 arrivals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.signal import hilbert
from scipy.signal.windows import tukey

from .material import Drive


@dataclass(frozen=True)
class TraceFeatures:
    valid: bool
    distance_m: float
    a0_peak_v: float
    a0_peak_time_s: float
    a0_tof_s: float
    a0_fwhm_s: float
    a0_phase_rad: float
    s0_peak_v: float
    s0_peak_time_s: float
    s0_tof_s: float
    s0_fwhm_s: float
    s0_phase_rad: float

    @staticmethod
    def invalid(distance_m: float) -> "TraceFeatures":
        nan = float("nan")
        return TraceFeatures(False, distance_m, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)


def envelope(waveform: np.ndarray) -> np.ndarray:
    if np.allclose(waveform, 0.0):
        return np.zeros_like(waveform)
    return np.abs(hilbert(waveform))


def _interp_crossing(time_s: np.ndarray, env: np.ndarray, i0: int, i1: int, level: float) -> float:
    v0 = env[i0] - level
    v1 = env[i1] - level
    if v1 == v0:
        return float(time_s[i1])
    frac = np.clip(v0 / (v0 - v1), 0.0, 1.0)
    return float(time_s[i0] + frac * (time_s[i1] - time_s[i0]))


def fwhm(time_s: np.ndarray, env: np.ndarray, peak_index: int, max_half_width_s: float) -> float:
    """Full width at half the peak, walking outward and interpolating the crossings."""
    level = 0.5 * env[peak_index]
    if level <= 0.0:
        return float("nan")
    left = peak_index
    while left > 0 and env[left] > level and (time_s[peak_index] - time_s[left]) < max_half_width_s:
        left -= 1
    right = peak_index
    last = len(env) - 1
    while right < last and env[right] > level and (time_s[right] - time_s[peak_index]) < max_half_width_s:
        right += 1
    if env[left] > level:
        t_left = float(time_s[peak_index] - max_half_width_s)
    else:
        t_left = _interp_crossing(time_s, env, left, min(left + 1, peak_index), level)
    if env[right] > level:
        t_right = float(time_s[peak_index] + max_half_width_s)
    else:
        t_right = _interp_crossing(time_s, env, max(right - 1, peak_index), right, level)
    return max(0.0, t_right - t_left)


def carrier_phase(waveform: np.ndarray, fs_hz: float, center_index: int, drive: Drive) -> float:
    """Phase at the drive frequency, referenced to t = 0 of the record.

    The arrival is Tukey-gated. A gate that starts at t0 measures the phase of
    the segment as if t0 were the origin, so the record phase is
    angle(Y_gate) - ω t0. With that correction a pure delay τ changes the phase
    by -ωτ (Fourier shift theorem) even when the gate follows the envelope peak.
    """
    half = max(4, int(round(0.65 * drive.duration_s * fs_hz)))
    i0 = max(0, center_index - half)
    i1 = min(len(waveform), center_index + half)
    segment = waveform[i0:i1].copy()
    if segment.size < 8:
        return float("nan")
    taper = tukey(segment.size, alpha=0.35)
    # Inner product against exp(-j ω t) on the record clock. This is the
    # Fourier component at the drive frequency, so a pure delay of τ contributes
    # exactly -ωτ when the same gate shape follows the packet.
    times = np.arange(i0, i1) / fs_hz
    kernel = np.exp(-1j * 2.0 * np.pi * drive.frequency_hz * times)
    value = np.dot(segment * taper, kernel)
    if abs(value) == 0.0:
        return float("nan")
    return float(np.angle(value))


def _peak_in(time_s: np.ndarray, env: np.ndarray, t0: float, t1: float) -> int | None:
    mask = (time_s >= t0) & (time_s <= t1)
    if not np.any(mask):
        return None
    indices = np.flatnonzero(mask)
    local = int(indices[np.argmax(env[indices])])
    return local


def extract_trace(
    waveform: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    distance_m: float,
    drive: Drive,
    cg_a0: float,
    cg_s0: float,
    noise_sigma: float,
) -> TraceFeatures:
    if distance_m < 1e-9 or not np.any(np.isfinite(waveform)):
        return TraceFeatures.invalid(distance_m)
    env = envelope(waveform)
    center = drive.envelope_center_s
    t_s0 = center + distance_m / cg_s0
    t_a0 = center + distance_m / cg_a0
    split = 0.5 * (t_s0 + t_a0)
    margin = max(12e-6, 0.8 * drive.duration_s)
    s0_index = _peak_in(time_s, env, max(0.0, t_s0 - margin), split)
    a0_index = _peak_in(time_s, env, split, t_a0 + max(30e-6, 0.45 * distance_m / cg_a0))
    floor = max(4.0 * noise_sigma, 1e-5)
    max_half = 2.6 * drive.duration_s

    def pack(index: int | None) -> tuple[float, float, float, float, float]:
        if index is None or env[index] < floor:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        peak_t = float(time_s[index])
        return (
            float(env[index]),
            peak_t,
            peak_t - center,
            fwhm(time_s, env, index, max_half),
            carrier_phase(waveform, fs_hz, index, drive),
        )

    s0 = pack(s0_index)
    a0 = pack(a0_index)
    valid = bool(np.isfinite(a0[0]) or np.isfinite(s0[0]))
    return TraceFeatures(
        valid=valid,
        distance_m=distance_m,
        a0_peak_v=a0[0],
        a0_peak_time_s=a0[1],
        a0_tof_s=a0[2],
        a0_fwhm_s=a0[3],
        a0_phase_rad=a0[4],
        s0_peak_v=s0[0],
        s0_peak_time_s=s0[1],
        s0_tof_s=s0[2],
        s0_fwhm_s=s0[3],
        s0_phase_rad=s0[4],
    )


def wrapped_delta(phase: float, reference: float) -> float:
    if not np.isfinite(phase) or not np.isfinite(reference):
        return float("nan")
    return float(np.angle(np.exp(1j * (phase - reference))))


def phase_delay_s(phase: float, reference: float, frequency_hz: float, tof_delta_s: float) -> float:
    """Extra delay from the carrier phase, unwrapped with the measured TOF as a guide.

    A pure delay multiplies the spectrum by exp(-j ω τ), so τ = -Δφ / ω.
    """
    delta = wrapped_delta(phase, reference)
    if not np.isfinite(delta) or frequency_hz <= 0.0:
        return float("nan")
    if not np.isfinite(tof_delta_s):
        return float(-delta / (2.0 * np.pi * frequency_hz))
    expected = -2.0 * np.pi * frequency_hz * tof_delta_s
    unwrapped = expected + float(np.angle(np.exp(1j * (delta - expected))))
    return float(-unwrapped / (2.0 * np.pi * frequency_hz))
