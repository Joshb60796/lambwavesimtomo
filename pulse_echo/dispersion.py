"""Fundamental S0 and A0 roots of the Rayleigh-Lamb equation.

The residual is the stress-free plate characteristic function in the form
used by Rose, Ultrasonic Waves in Solid Media, and by Viktorov. The plate
runs from -h to +h, so h is the half-thickness and the full sheet thickness
is d = 2h:

    symmetric:
        (q^2 - k^2)^2 sin(qh) cos(ph) + 4 k^2 p q cos(qh) sin(ph) = 0
    antisymmetric:
        (q^2 - k^2)^2 cos(qh) sin(ph) + 4 k^2 p q sin(qh) cos(ph) = 0

with p^2 = (ω/c_L)^2 - k^2 and q^2 = (ω/c_T)^2 - k^2. Below the shear speed
both p and q are imaginary; the branch with a nonnegative real part keeps the
antisymmetric residual on its imaginary part and the symmetric residual on
its real part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .material import Aluminum6061

_CURVE_CACHE: dict[tuple[float, float, float, int], "LambCurve"] = {}


def _branch(z: complex) -> complex:
    if z.real < 0.0:
        return -z
    return z


def characteristic(phase_velocity: float, frequency_hz: float, plate: Aluminum6061, antisym: bool) -> float:
    """Signed residual. A0 is tracked on the imaginary part below c_T; S0 on the real part above c_T."""
    omega = 2.0 * np.pi * frequency_hz
    k = omega / phase_velocity
    p = _branch(np.sqrt((omega / plate.c_longitudinal) ** 2 - k**2 + 0j))
    q = _branch(np.sqrt((omega / plate.c_shear) ** 2 - k**2 + 0j))
    # Rose's h is the half-thickness. plate.thickness_m is the full sheet.
    half_m = 0.5 * plate.thickness_m
    ph = p * half_m
    qh = q * half_m
    membrane = (q**2 - k**2) ** 2
    if antisym:
        value = membrane * np.cos(qh) * np.sin(ph) + 4.0 * k**2 * p * q * np.sin(qh) * np.cos(ph)
        return float(value.imag)
    value = membrane * np.sin(qh) * np.cos(ph) + 4.0 * k**2 * p * q * np.cos(qh) * np.sin(ph)
    return float(value.real)


def _find_root(
    frequency_hz: float,
    plate: Aluminum6061,
    antisym: bool,
    lo: float,
    hi: float,
    n_scan: int = 240,
) -> float | None:
    if hi <= lo:
        return None
    speeds = np.linspace(lo, hi, n_scan)
    values = np.array([characteristic(float(c), frequency_hz, plate, antisym) for c in speeds])
    for i in range(len(speeds) - 1):
        left, right = values[i], values[i + 1]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if abs(left) > 1e15 or abs(right) > 1e15:
            continue
        if left == 0.0 or left * right < 0.0:
            try:
                return float(
                    brentq(
                        lambda c: characteristic(c, frequency_hz, plate, antisym),
                        float(speeds[i]),
                        float(speeds[i + 1]),
                        xtol=1e-4,
                    )
                )
            except ValueError:
                continue
    return None


def phase_velocity(frequency_hz: float, plate: Aluminum6061, antisym: bool, guess: float | None = None) -> float:
    if antisym:
        lo, hi = 40.0, plate.c_shear - 2.0
        if guess is not None:
            lo = max(lo, guess * 0.75)
            hi = min(hi, guess * 1.25)
        root = _find_root(frequency_hz, plate, True, lo, hi)
        if root is None and guess is not None:
            root = _find_root(frequency_hz, plate, True, 40.0, plate.c_shear - 2.0, n_scan=480)
    else:
        lo, hi = plate.c_shear + 30.0, plate.c_longitudinal * 0.998
        if guess is not None:
            lo = max(lo, guess * 0.9)
            hi = min(hi, guess * 1.1)
        root = _find_root(frequency_hz, plate, False, lo, hi)
        if root is None and guess is not None:
            root = _find_root(frequency_hz, plate, False, plate.c_shear + 30.0, plate.c_longitudinal * 0.998, n_scan=480)
    if root is None:
        raise RuntimeError(f"No {'A0' if antisym else 'S0'} root at {frequency_hz:.0f} Hz")
    return root


@dataclass(frozen=True)
class LambCurve:
    """Phase velocity sampled across the tone-burst band, plus center-frequency speeds."""

    frequency_hz: np.ndarray
    a0_phase: np.ndarray
    s0_phase: np.ndarray
    a0_phase_at_drive: float
    s0_phase_at_drive: float
    a0_group_at_drive: float
    s0_group_at_drive: float
    drive_hz: float

    def wavenumber(self, freqs: np.ndarray, mode: str) -> np.ndarray:
        """Interpolate k(f) = 2πf/c_ph. Bins outside the table hold the end speed."""
        table = self.a0_phase if mode == "A0" else self.s0_phase
        speed = np.interp(np.abs(freqs), self.frequency_hz, table)
        nonzero = np.abs(freqs) > 0.0
        k = np.zeros_like(freqs, dtype=float)
        k[nonzero] = 2.0 * np.pi * np.abs(freqs[nonzero]) / speed[nonzero]
        return k


def _group_velocity(frequency_hz: float, phase: float, plate: Aluminum6061, antisym: bool) -> float:
    df = 2_000.0
    above = phase_velocity(frequency_hz + df, plate, antisym, guess=phase)
    below = phase_velocity(frequency_hz - df, plate, antisym, guess=phase)
    dcdf = (above - below) / (2.0 * df)
    denom = phase - frequency_hz * dcdf
    if abs(denom) < 1e-6:
        return phase
    return float(phase**2 / denom)


def lamb_curve(
    plate: Aluminum6061 | None = None,
    drive_hz: float = 550e3,
    f_min: float = 180e3,
    f_max: float = 1.15e6,
    n_points: int = 64,
) -> LambCurve:
    plate = plate or Aluminum6061()
    key = (round(plate.thickness_m, 9), round(drive_hz, 3), round(f_min, 3), n_points)
    cached = _CURVE_CACHE.get(key)
    if cached is not None:
        return cached

    freqs = np.linspace(f_min, f_max, n_points)
    a0 = np.empty(n_points)
    s0 = np.empty(n_points)
    a0_guess = phase_velocity(drive_hz, plate, True)
    s0_guess = phase_velocity(drive_hz, plate, False)
    # Walk outward from the drive frequency so each bracket stays on the same branch.
    drive_index = int(np.argmin(np.abs(freqs - drive_hz)))

    def fill(antisym: bool, dest: np.ndarray, seed: float) -> None:
        dest[drive_index] = phase_velocity(float(freqs[drive_index]), plate, antisym, guess=seed)
        guess = dest[drive_index]
        for i in range(drive_index + 1, n_points):
            dest[i] = phase_velocity(float(freqs[i]), plate, antisym, guess=guess)
            guess = dest[i]
        guess = dest[drive_index]
        for i in range(drive_index - 1, -1, -1):
            dest[i] = phase_velocity(float(freqs[i]), plate, antisym, guess=guess)
            guess = dest[i]

    fill(True, a0, a0_guess)
    fill(False, s0, s0_guess)
    a0_drive = phase_velocity(drive_hz, plate, True, guess=a0_guess)
    s0_drive = phase_velocity(drive_hz, plate, False, guess=s0_guess)
    curve = LambCurve(
        frequency_hz=freqs,
        a0_phase=a0,
        s0_phase=s0,
        a0_phase_at_drive=a0_drive,
        s0_phase_at_drive=s0_drive,
        a0_group_at_drive=_group_velocity(drive_hz, a0_drive, plate, True),
        s0_group_at_drive=_group_velocity(drive_hz, s0_drive, plate, False),
        drive_hz=drive_hz,
    )
    _CURVE_CACHE[key] = curve
    return curve
