"""Tone burst and two-mode pitch-catch synthesis.

The excitation is the n-count Hanning tone burst used throughout the PWAS
literature (Giurgiutiu, J. Intell. Mater. Syst. Struct. 16, 2005; Shen and
Giurgiutiu, J. Intell. Mater. Syst. Struct. 25, 2014, who drive a 7 mm wafer
with a 5-count burst). Three counts is the same waveform with a shorter gate.

Propagation of each mode is the far-field circular-crested plate wave. With the
FFT sign convention e^{+jωt}, the outgoing wave is the Hankel function of the
second kind. Its far field (Viktorov; Rose) is

    (k r)^{-1/2} exp(-j (k r - π/4)).

The π/4 is kept. The wavenumber tilt is normalized to 1 at the drive frequency
so the reference volts stay a scope calibration. Geometric spreading still
falls as r^{-1/2}.

Attenuation uses a frequency-independent quality factor,
α(f) = α(f0) k(f) / k(f0), so Q = π f / (α c) does not depend on frequency.
α proportional to f alone is that law only when the phase speed is constant.
The Np/m values are effective coefficients inside the range reported for
aluminum sheet near 0.5 MHz. They are not a fit to one coupon.

A0 and S0 voltage scales are one instrument gain per mode. Ideal-bond tuning
also multiplies by a trigonometric factor (a strip) or by J1(xi a) (a circular
wafer; Raghavan and Cesnik) and by the residue of the Rayleigh-Lamb
determinant. Those factors are not separable from bond and charge-amplifier
gain in this bench, so they stay inside the two reference volts.

A through-hole replaces the straight length by Keller's shortest creeping path
(tangent, boundary arc, tangent). The shadow exponent, the copy delayed by one
chord, and the fixed coda fraction are phenomenological stand-ins for
unresolved rim scattering. They are not diffraction coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq

from .dispersion import LambCurve
from .geometry import HoleHit, aperture_offsets, hole_hit
from .material import Drive

# Volts, after spreading, on a pristine cross-plate path of length `reference_range`.
A0_REFERENCE_VOLT = 2.40
S0_REFERENCE_VOLT = 1.05
CROSSTALK_FRACTION = 0.012
# Np/m at the drive frequency. Aluminum sheet is a low-loss guide at 550 kHz.
A0_ATTEN_NP_PER_M = 1.4
S0_ATTEN_NP_PER_M = 0.35
SPREAD_FLOOR_M = 0.008


@dataclass(frozen=True)
class Timebase:
    fs_hz: float
    n_samples: int

    @property
    def dt(self) -> float:
        return 1.0 / self.fs_hz

    @property
    def time_s(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.fs_hz

    @property
    def freqs(self) -> np.ndarray:
        return rfftfreq(self.n_samples, d=self.dt)


def toneburst(time_s: np.ndarray, drive: Drive) -> np.ndarray:
    """±V Hanning window times a sine. The window is zero at both ends."""
    duration = drive.duration_s
    signal = np.zeros_like(time_s, dtype=float)
    live = (time_s >= 0.0) & (time_s <= duration)
    if not np.any(live):
        return signal
    t = time_s[live]
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / duration))
    signal[live] = drive.voltage * window * np.sin(2.0 * np.pi * drive.frequency_hz * t)
    return signal


def _quality_alpha(wavenumber: np.ndarray, wavenumber_ref: float, alpha_ref: float) -> np.ndarray:
    """Spatial attenuation for a constant Q: α / k is independent of frequency."""
    alpha = np.zeros_like(wavenumber, dtype=float)
    if wavenumber_ref <= 0.0:
        return alpha
    np.divide(wavenumber, wavenumber_ref, out=alpha)
    alpha *= alpha_ref
    return alpha


def _outgoing(wavenumber: np.ndarray, wavenumber_ref: float, length_m: float) -> np.ndarray:
    """Far-field H0^(2)(k r), normalized to unit wavenumber tilt at k_ref."""
    geometric = np.sqrt(SPREAD_FLOOR_M / max(length_m, SPREAD_FLOOR_M))
    tilt = np.zeros_like(wavenumber, dtype=float)
    positive = wavenumber > 0.0
    tilt[positive] = np.sqrt(wavenumber_ref / wavenumber[positive])
    phase = np.exp(-1j * wavenumber * length_m) * np.exp(1j * 0.25 * np.pi)
    return geometric * tilt * phase


def _mode_scales(
    drive_spectrum: np.ndarray,
    curve: LambCurve,
    timebase: Timebase,
    reference_range_m: float,
) -> tuple[float, float]:
    """Calibrate mode amplitudes so a pristine path of `reference_range_m` hits the target volts."""
    scales = []
    for mode, target, atten0 in (
        ("A0", A0_REFERENCE_VOLT, A0_ATTEN_NP_PER_M),
        ("S0", S0_REFERENCE_VOLT, S0_ATTEN_NP_PER_M),
    ):
        k = curve.wavenumber(timebase.freqs, mode)
        k_ref = float(curve.wavenumber(np.array([curve.drive_hz]), mode)[0])
        alpha = _quality_alpha(k, k_ref, atten0)
        transfer = _outgoing(k, k_ref, reference_range_m) * np.exp(-alpha * reference_range_m)
        waveform = irfft(drive_spectrum * transfer, n=timebase.n_samples)
        peak = float(np.max(np.abs(waveform)))
        scales.append(target / peak if peak > 0 else 0.0)
    return scales[0], scales[1]


def propagate_pair(
    start: np.ndarray,
    end: np.ndarray,
    holes: list[tuple[tuple[float, float], float]],
    drive_spectrum: np.ndarray,
    curve: LambCurve,
    timebase: Timebase,
    a0_scale: float,
    s0_scale: float,
    shadow: float,
    scatter: float,
    coda_delay_s: float = 0.0,
    coda_fraction: float = 0.0,
) -> np.ndarray:
    """Coherent aperture sum of S0 + A0 for one transmitter-receiver pair."""
    offsets, weights = aperture_offsets()
    delta = end - start
    distance = float(np.hypot(delta[0], delta[1]))
    if distance < 1e-9:
        return np.zeros(timebase.n_samples)
    tangent = delta / distance
    normal = np.array([-tangent[1], tangent[0]])
    k_a0 = curve.wavenumber(timebase.freqs, "A0")
    k_s0 = curve.wavenumber(timebase.freqs, "S0")
    k_a0_ref = float(curve.wavenumber(np.array([curve.drive_hz]), "A0")[0])
    k_s0_ref = float(curve.wavenumber(np.array([curve.drive_hz]), "S0")[0])
    alpha_a0 = _quality_alpha(k_a0, k_a0_ref, A0_ATTEN_NP_PER_M)
    alpha_s0 = _quality_alpha(k_s0, k_s0_ref, S0_ATTEN_NP_PER_M)
    spectrum = np.zeros_like(drive_spectrum)
    for offset, weight in zip(offsets, weights):
        p0 = start + normal * offset
        p1 = end + normal * offset
        ray = p1 - p0
        length = float(np.hypot(ray[0], ray[1]))
        if length < 1e-9:
            continue
        hits: list[HoleHit] = []
        for center, radius in holes:
            hit = hole_hit(center, radius, p0, p1)
            if hit.depth > 0.0:
                hits.append(hit)
        extra = float(sum(hit.extra_m for hit in hits))
        shadow_gain = float(np.exp(-shadow * sum(hit.depth for hit in hits)))
        primary_path = length + extra
        for k, k_ref, alpha, scale in (
            (k_a0, k_a0_ref, alpha_a0, a0_scale),
            (k_s0, k_s0_ref, alpha_s0, s0_scale),
        ):
            primary = scale * shadow_gain * _outgoing(k, k_ref, primary_path) * np.exp(-alpha * primary_path)
            scattered = 0.0
            for hit in hits:
                # One chord later than the creeping arrival. The fraction is a stand-in
                # for a later rim arrival, not a second Keller path.
                scatter_path = primary_path + hit.chord_m
                scattered = scattered + (
                    scale * scatter * hit.depth * _outgoing(k, k_ref, scatter_path) * np.exp(-alpha * scatter_path)
                )
            coda = 0.0
            if hits and coda_fraction > 0.0 and coda_delay_s > 0.0:
                # A diffracted coda trails the creeping arrival by a fraction of the
                # burst length, which is what stretches the half-maximum width.
                coda = primary * coda_fraction * np.exp(-1j * 2.0 * np.pi * timebase.freqs * coda_delay_s)
            spectrum += weight * drive_spectrum * (primary + scattered + coda)
    return irfft(spectrum, n=timebase.n_samples).real


def synthesize_baseline_scales(
    curve: LambCurve,
    timebase: Timebase,
    drive: Drive,
    reference_range_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    excitation = toneburst(timebase.time_s, drive)
    spectrum = rfft(excitation)
    a0_scale, s0_scale = _mode_scales(spectrum, curve, timebase, reference_range_m)
    return excitation, spectrum, a0_scale, s0_scale
