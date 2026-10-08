"""Identities the simulator must satisfy, and the scattering terms it must not pretend to derive.

Closed forms below are the isotropic Hooke law, the Rayleigh-Lamb plate equation
(Rose; Viktorov), the Kirchhoff flexural limit, the Fourier shift theorem, and
the far-field cylindrical plate wave. The Hanning n-cycle tone burst is the
excitation in Giurgiutiu, J. Intell. Mater. Syst. Struct. 16 (2005) 291-305,
and in Shen and Giurgiutiu, J. Intell. Mater. Syst. Struct. 25 (2014) 506-520
(5-count, 7 mm PWAS). Three counts is the same waveform with a shorter gate.

The shadow exponent, the scatter fraction, and the coda fraction are not
scattering coefficients. Tests that touch them only check the geometric path
and that the surrogate does what the code claims.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
from scipy.special import j1
from scipy.signal import hilbert

from pulse_echo.dispersion import characteristic, lamb_curve, phase_velocity
from pulse_echo.features import TraceFeatures, carrier_phase, extract_trace, fwhm, phase_delay_s, wrapped_delta
from pulse_echo.geometry import aperture_offsets, grid_path_lengths, hole_hit, make_coupon
from pulse_echo.material import INCH, Aluminum6061, Drive
from pulse_echo.signals import (
    A0_ATTEN_NP_PER_M,
    A0_REFERENCE_VOLT,
    S0_ATTEN_NP_PER_M,
    S0_REFERENCE_VOLT,
    SPREAD_FLOOR_M,
    Timebase,
    propagate_pair,
    synthesize_baseline_scales,
    toneburst,
)
from pulse_echo.tomography import RayRecord, reconstruct, sirt

DRIVE_HZ = 550e3


def _kirchhoff_flexure(plate: Aluminum6061, frequency_hz: float) -> tuple[float, float]:
    """Low-frequency A0 limit: cp = ω^{1/2} (D / ρh)^{1/4}, cg = 2 cp.

    D = E h^3 / (12 (1-ν^2)) and h is the full thickness. This is the
    Kirchhoff plate, which Rayleigh-Lamb approaches from below as f -> 0.
    """
    h = plate.thickness_m
    flexural = plate.young_pa * h**3 / (12.0 * (1.0 - plate.poisson**2))
    omega = 2.0 * np.pi * frequency_hz
    phase = float(np.sqrt(omega) * (flexural / (plate.density * h)) ** 0.25)
    return phase, 2.0 * phase


def _single_ray():
    return patch("pulse_echo.signals.aperture_offsets", return_value=(np.array([0.0]), np.array([1.0])))


class HandbookTests(unittest.TestCase):
    def test_6061_t6_constants_and_the_speeds_they_imply(self) -> None:
        plate = Aluminum6061()
        young = 68.9e9
        poisson = 0.33
        density = 2700.0
        self.assertEqual(plate.young_pa, young)
        self.assertEqual(plate.poisson, poisson)
        self.assertEqual(plate.density, density)
        self.assertAlmostEqual(plate.thickness_m, INCH / 32.0, places=12)
        self.assertAlmostEqual(plate.thickness_in, 1.0 / 32.0, places=12)
        # ASM lists G ≈ 26.0 GPa as an estimate. Isotropic Hooke gives E/2(1+ν).
        shear = young / (2.0 * (1.0 + poisson))
        self.assertAlmostEqual(plate.shear_modulus, shear, places=6)
        self.assertLess(abs(shear - 26.0e9) / 26.0e9, 0.005)
        lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.assertAlmostEqual(plate.lame_lambda, lame, places=4)
        self.assertAlmostEqual(plate.c_shear, np.sqrt(shear / density), places=6)
        self.assertAlmostEqual(plate.c_longitudinal, np.sqrt((lame + 2.0 * shear) / density), places=6)
        plate_speed = np.sqrt(young / (density * (1.0 - poisson**2)))
        self.assertAlmostEqual(plate.c_plate, plate_speed, places=6)
        self.assertGreater(plate.c_longitudinal, plate.c_plate)
        self.assertGreater(plate.c_plate, plate.c_shear)

    def test_only_the_fundamental_modes_propagate_at_the_drive(self) -> None:
        plate = Aluminum6061()
        # Lowest higher-order cutoff is the A1 thickness shear, f = c_T / (2d).
        a1_cutoff = plate.c_shear / (2.0 * plate.thickness_m)
        self.assertGreater(a1_cutoff, 1.5e6)
        self.assertLess(DRIVE_HZ, a1_cutoff)
        fd_mhz_mm = DRIVE_HZ * plate.thickness_m * 1e-3
        self.assertAlmostEqual(fd_mhz_mm, 0.4365625, places=6)
        # A1 cutoff in the same units is c_T / 2.
        self.assertLess(fd_mhz_mm, (plate.c_shear / 2.0) * 1e-3)


class RayleighLambTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plate = Aluminum6061()
        cls.curve = lamb_curve(cls.plate, drive_hz=DRIVE_HZ)

    def test_roots_are_zeros_of_the_characteristic_function(self) -> None:
        for antisym, speed in (
            (True, self.curve.a0_phase_at_drive),
            (False, self.curve.s0_phase_at_drive),
        ):
            on_root = abs(characteristic(speed, DRIVE_HZ, self.plate, antisym))
            off_root = abs(characteristic(speed * 1.01, DRIVE_HZ, self.plate, antisym))
            self.assertGreater(off_root, 0.0)
            self.assertLess(on_root / off_root, 1e-3)

    def test_one_fundamental_root_in_each_open_interval(self) -> None:
        def n_crossings(antisym: bool, lo: float, hi: float) -> int:
            speeds = np.linspace(lo, hi, 500)
            values = np.array([characteristic(float(c), DRIVE_HZ, self.plate, antisym) for c in speeds])
            flips = 0
            for i in range(len(values) - 1):
                if np.isfinite(values[i]) and np.isfinite(values[i + 1]) and values[i] * values[i + 1] < 0.0:
                    flips += 1
            return flips

        shear = self.plate.c_shear
        self.assertEqual(n_crossings(True, 40.0, shear - 2.0), 1)
        self.assertEqual(n_crossings(False, shear + 30.0, self.plate.c_longitudinal * 0.998), 1)

    def test_drive_frequency_ordering(self) -> None:
        plate = self.plate
        a0 = self.curve.a0_phase_at_drive
        s0 = self.curve.s0_phase_at_drive
        self.assertGreater(a0, 1600.0)
        self.assertLess(a0, 2000.0)
        self.assertLess(a0, plate.c_shear)
        self.assertGreater(self.curve.a0_group_at_drive, a0)
        self.assertLess(self.curve.a0_group_at_drive / a0, 1.8)
        self.assertGreater(s0, plate.c_shear)
        self.assertLess(s0, plate.c_plate)
        self.assertGreater(s0, plate.c_plate - 80.0)
        self.assertLess(self.curve.s0_group_at_drive, s0)
        self.assertGreater(self.curve.s0_group_at_drive, 5100.0)
        self.assertTrue(np.all(np.diff(self.curve.a0_phase) > 0.0))
        self.assertTrue(np.all(np.diff(self.curve.s0_phase) < 0.0))

    def test_group_velocity_is_domega_dk(self) -> None:
        df = 2_000.0
        for antisym, phase, reported in (
            (True, self.curve.a0_phase_at_drive, self.curve.a0_group_at_drive),
            (False, self.curve.s0_phase_at_drive, self.curve.s0_group_at_drive),
        ):
            above = phase_velocity(DRIVE_HZ + df, self.plate, antisym, guess=phase)
            below = phase_velocity(DRIVE_HZ - df, self.plate, antisym, guess=phase)
            k_above = 2.0 * np.pi * (DRIVE_HZ + df) / above
            k_below = 2.0 * np.pi * (DRIVE_HZ - df) / below
            group = 2.0 * np.pi / ((k_above - k_below) / (2.0 * df))
            self.assertAlmostEqual(group, reported, delta=0.5)

    def test_low_frequency_a0_approaches_kirchhoff_and_s0_approaches_the_plate_speed(self) -> None:
        frequency = 10e3
        a0 = phase_velocity(frequency, self.plate, True)
        kirchhoff, twice = _kirchhoff_flexure(self.plate, frequency)
        # Rayleigh-Lamb is slower than Kirchhoff once rotary inertia and shear count,
        # and the two agree as the thickness-frequency product vanishes.
        self.assertLess(a0, kirchhoff)
        self.assertAlmostEqual(a0 / kirchhoff, 1.0, delta=0.015)
        df = 500.0
        above = phase_velocity(frequency + df, self.plate, True, guess=a0)
        below = phase_velocity(frequency - df, self.plate, True, guess=a0)
        slope = (above - below) / (2.0 * df)
        group = a0**2 / (a0 - frequency * slope)
        self.assertAlmostEqual(group / a0, 2.0, delta=0.03)
        self.assertAlmostEqual(group / twice, 1.0, delta=0.03)
        s0 = phase_velocity(frequency, self.plate, False)
        self.assertAlmostEqual(s0, self.plate.c_plate, delta=1.0)
        self.assertLess(s0, self.plate.c_plate)
        # Kirchhoff phase speed scales as f^{1/2} while the sheet is thin.
        low = phase_velocity(12e3, self.plate, True)
        high = phase_velocity(48e3, self.plate, True)
        self.assertAlmostEqual(high / low, 2.0, delta=0.04)

    def test_real_rayleigh_lamb_residual_on_two_different_plates(self) -> None:
        # Real hyperbolic form, written out so it is not the complex residual
        # the solver itself drives to zero. Rose's h is the half-thickness.
        plates = (
            self.plate,
            Aluminum6061(young_pa=110.0e9, poisson=0.25, density=4500.0, thickness_m=1.5e-3),
        )
        for plate in plates:
            for frequency in (80e3, 250e3, 700e3):
                a0 = phase_velocity(frequency, plate, True)
                s0 = phase_velocity(frequency, plate, False)
                self.assertLess(_a0_residual(a0, frequency, plate) / _a0_residual(a0 * 1.02, frequency, plate), 1e-3)
                self.assertLess(abs(_s0_residual(s0, frequency, plate) / _s0_residual(s0 * 0.99, frequency, plate)), 1e-3)
                self.assertLess(a0, plate.c_shear)
                self.assertGreater(s0, plate.c_shear)
                self.assertLess(s0, plate.c_plate)


class ToneburstTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timebase = Timebase(10e6, 4096)

    def _burst(self, n_cycles: int) -> tuple[Drive, np.ndarray]:
        drive = Drive(frequency_hz=DRIVE_HZ, voltage=15.0, n_cycles=n_cycles)
        return drive, toneburst(self.timebase.time_s, drive)

    def test_three_and_five_count_hanning_bursts_match_the_definition(self) -> None:
        time_s = self.timebase.time_s
        for n_cycles in (3, 5):
            drive, signal = self._burst(n_cycles)
            duration = n_cycles / DRIVE_HZ
            self.assertAlmostEqual(drive.duration_s, duration, places=12)
            self.assertAlmostEqual(drive.envelope_center_s, 0.5 * duration, places=12)
            live = (time_s >= 0.0) & (time_s <= duration)
            window = 0.5 * (1.0 - np.cos(2.0 * np.pi * time_s[live] / duration))
            carrier = np.sin(2.0 * np.pi * DRIVE_HZ * time_s[live])
            self.assertTrue(np.allclose(signal[live], 15.0 * window * carrier))
            self.assertTrue(np.all(signal[~live] == 0.0))
            self.assertAlmostEqual(float(signal[0]), 0.0, places=12)
            at_end = float(np.interp(duration, time_s, signal))
            self.assertAlmostEqual(at_end, 0.0, delta=0.05)
            self.assertLessEqual(float(np.max(signal)), 15.0)
            self.assertGreaterEqual(float(np.min(signal)), -15.0)
            self.assertGreater(float(np.max(signal)), 10.0)
            self.assertLess(float(np.min(signal)), -10.0)
            # The Hann window itself is at half-maximum at T/4 and 3T/4.
            half = 0.5
            live_time = time_s[live]
            self.assertAlmostEqual(float(np.interp(0.25 * duration, live_time, window)), half, delta=1e-4)
            self.assertAlmostEqual(float(np.interp(0.75 * duration, live_time, window)), half, delta=1e-4)

    def test_hann_window_fwhm_is_half_the_burst_and_the_envelope_tracks_it(self) -> None:
        time_s = self.timebase.time_s
        for n_cycles in (3, 5):
            drive, signal = self._burst(n_cycles)
            duration = drive.duration_s
            window = np.zeros_like(time_s)
            live = (time_s >= 0.0) & (time_s <= duration)
            window[live] = 0.5 * (1.0 - np.cos(2.0 * np.pi * time_s[live] / duration))
            peak = int(np.argmax(window))
            self.assertAlmostEqual(fwhm(time_s, window, peak, duration), 0.5 * duration, delta=1.0 / self.timebase.fs_hz)
            env = np.abs(hilbert(signal))
            env_peak = int(np.argmax(env))
            width = fwhm(time_s, env, env_peak, duration)
            self.assertAlmostEqual(width, 0.5 * duration, delta=0.12 * duration)

    def test_spectrum_is_centered_on_the_drive_and_three_counts_are_broader(self) -> None:
        bands = []
        df = self.timebase.fs_hz / self.timebase.n_samples
        for n_cycles in (3, 5):
            drive, signal = self._burst(n_cycles)
            spectrum = np.abs(rfft(signal))
            freqs = rfftfreq(self.timebase.n_samples, d=self.timebase.dt)
            peak = int(np.argmax(spectrum[1:])) + 1
            self.assertLess(abs(freqs[peak] - DRIVE_HZ), 2.0 * df)
            half = spectrum[peak] / np.sqrt(2.0)
            above = np.flatnonzero(spectrum >= half)
            bands.append(float(freqs[above[-1]] - freqs[above[0]]))
            # Hann main-lobe scale is set by 1/T. 1.44/T is the half-power width of this gate.
            self.assertAlmostEqual(bands[-1], 1.44 / drive.duration_s, delta=0.15 / drive.duration_s)
        self.assertGreater(bands[0], bands[1])

    def test_default_excitation_stays_five_counts(self) -> None:
        self.assertEqual(Drive().n_cycles, 5)
        self.assertEqual(Drive().voltage, 15.0)
        self.assertEqual(Drive().frequency_hz, DRIVE_HZ)


class PropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plate = Aluminum6061()
        cls.curve = lamb_curve(cls.plate, drive_hz=DRIVE_HZ)
        cls.timebase = Timebase(10e6, 4000)

    def _mode_wave(self, distance_m: float, n_cycles: int, mode: str, holes=None, shadow=0.0, scatter=0.0, coda=0.0):
        drive = Drive(frequency_hz=DRIVE_HZ, voltage=15.0, n_cycles=n_cycles)
        _excitation, spectrum, a0_scale, s0_scale = synthesize_baseline_scales(
            self.curve, self.timebase, drive, 0.09
        )
        if mode == "A0":
            s0_scale = 0.0
        else:
            a0_scale = 0.0
        start = np.array([0.0, 0.0])
        end = np.array([distance_m, 0.0])
        with _single_ray():
            wave = propagate_pair(
                start, end, holes or [], spectrum, self.curve, self.timebase,
                a0_scale, s0_scale, shadow, scatter, 0.34 * drive.duration_s, coda,
            )
        return drive, wave

    def _spectral_magnitude(self, wave: np.ndarray, frequency_hz: float) -> float:
        spectrum = rfft(wave)
        freqs = self.timebase.freqs
        index = int(np.argmin(np.abs(freqs - frequency_hz)))
        self.assertAlmostEqual(freqs[index], frequency_hz, delta=1.0)
        return float(abs(spectrum[index]))

    def test_cylindrical_spreading_and_constant_q_attenuation(self) -> None:
        # |U| ~ r^{-1/2} exp(-α(f) r). Constant Q means α(f) / k(f) is constant,
        # so α(f) = α(f0) k(f) / k(f0). The Hankel (k)^{-1/2} tilt cancels in a
        # same-frequency ratio of two ranges.
        r1, r2 = 0.09, 0.18
        for mode, alpha0 in (("A0", A0_ATTEN_NP_PER_M), ("S0", S0_ATTEN_NP_PER_M)):
            _drive, near = self._mode_wave(r1, 5, mode)
            _drive, far = self._mode_wave(r2, 5, mode)
            k_ref = float(self.curve.wavenumber(np.array([DRIVE_HZ]), mode)[0])
            measured = []
            for fraction in (0.7, 1.0):
                frequency = fraction * DRIVE_HZ
                k = float(self.curve.wavenumber(np.array([frequency]), mode)[0])
                alpha = alpha0 * k / k_ref
                ratio = self._spectral_magnitude(far, frequency) / self._spectral_magnitude(near, frequency)
                expected = np.sqrt(r1 / r2) * np.exp(-alpha * (r2 - r1))
                self.assertAlmostEqual(ratio, expected, delta=0.01 * expected)
                measured.append(alpha / k)
            self.assertAlmostEqual(measured[0], measured[1], places=6)
        self.assertGreater(A0_ATTEN_NP_PER_M, S0_ATTEN_NP_PER_M)
        self.assertGreater(S0_ATTEN_NP_PER_M, 0.05)
        self.assertLess(A0_ATTEN_NP_PER_M, 8.0)

    def test_group_arrival_tracks_distance_over_group_velocity(self) -> None:
        distance = 0.20
        errors = {}
        for n_cycles in (3, 5):
            for mode, group in (
                ("S0", self.curve.s0_group_at_drive),
                ("A0", self.curve.a0_group_at_drive),
            ):
                drive, wave = self._mode_wave(distance, n_cycles, mode)
                features = extract_trace(
                    wave, self.timebase.time_s, self.timebase.fs_hz, distance, drive,
                    self.curve.a0_group_at_drive, self.curve.s0_group_at_drive, 0.0,
                )
                measured = features.s0_tof_s if mode == "S0" else features.a0_tof_s
                errors[(n_cycles, mode)] = measured - distance / group
        self.assertAlmostEqual(errors[(5, "S0")], 0.0, delta=0.3e-6)
        self.assertAlmostEqual(errors[(3, "S0")], 0.0, delta=0.3e-6)
        self.assertAlmostEqual(errors[(5, "A0")], 0.0, delta=0.6e-6)
        # A0 is dispersive. A 3-count burst is wider in frequency, so its
        # envelope peak walks farther off distance/cg than a 5-count burst.
        self.assertGreater(abs(errors[(3, "A0")]), abs(errors[(5, "A0")]) + 0.2e-6)

    def test_carrier_phase_obeys_the_shift_theorem(self) -> None:
        drive = Drive(n_cycles=5)
        signal = toneburst(self.timebase.time_s, drive)
        tau = 3.2e-6
        delayed = irfft(rfft(signal) * np.exp(-1j * 2.0 * np.pi * self.timebase.freqs * tau), n=signal.size)
        peak = int(np.argmax(np.abs(hilbert(signal))))
        delayed_peak = int(np.argmax(np.abs(hilbert(delayed))))
        phase = carrier_phase(signal, self.timebase.fs_hz, peak, drive)
        phase_delayed = carrier_phase(delayed, self.timebase.fs_hz, delayed_peak, drive)
        recovered = phase_delay_s(phase_delayed, phase, DRIVE_HZ, tau)
        self.assertAlmostEqual(recovered, tau, delta=0.05e-6)
        self.assertAlmostEqual(wrapped_delta(phase_delayed, phase), np.angle(np.exp(-1j * 2.0 * np.pi * DRIVE_HZ * tau)), delta=0.05)

    def test_s0_phase_tracks_wavenumber_times_range(self) -> None:
        r1, r2 = 0.08, 0.16
        phases = []
        for distance in (r1, r2):
            drive, wave = self._mode_wave(distance, 8, "S0")
            features = extract_trace(
                wave, self.timebase.time_s, self.timebase.fs_hz, distance, drive,
                self.curve.a0_group_at_drive, self.curve.s0_group_at_drive, 0.0,
            )
            phases.append(features.s0_phase_rad)
        delta = wrapped_delta(phases[1], phases[0])
        expected = -2.0 * np.pi * DRIVE_HZ * (r2 - r1) / self.curve.s0_phase_at_drive
        self.assertAlmostEqual(delta, np.angle(np.exp(1j * expected)), delta=0.15)

    def test_diametral_creeping_path_delays_s0_and_does_not_invent_width(self) -> None:
        radius = 0.006
        distance = 0.09
        center = (0.5 * distance, 0.0)
        hit = hole_hit(center, radius, np.array([0.0, 0.0]), np.array([distance, 0.0]))
        reach = 0.5 * distance
        glide = np.arccos(radius / reach)
        creep = 2.0 * np.sqrt(reach**2 - radius**2) + radius * (np.pi - 2.0 * glide)
        self.assertAlmostEqual(hit.extra_m, creep - distance, places=6)
        # The semicircle between the pierce points is the wrong first arrival.
        self.assertLess(hit.extra_m, 0.25 * radius * (np.pi - 2.0))
        self.assertAlmostEqual(hit.chord_m, 2.0 * radius, places=6)
        drive, pristine = self._mode_wave(distance, 5, "S0")
        _drive, blocked = self._mode_wave(distance, 5, "S0", holes=[(center, radius)])
        base = extract_trace(
            pristine, self.timebase.time_s, self.timebase.fs_hz, distance, drive,
            self.curve.a0_group_at_drive, self.curve.s0_group_at_drive, 0.0,
        )
        damaged = extract_trace(
            blocked, self.timebase.time_s, self.timebase.fs_hz, distance, drive,
            self.curve.a0_group_at_drive, self.curve.s0_group_at_drive, 0.0,
        )
        self.assertAlmostEqual(
            damaged.s0_tof_s - base.s0_tof_s,
            hit.extra_m / self.curve.s0_group_at_drive,
            delta=0.15e-6,
        )
        self.assertAlmostEqual(damaged.s0_fwhm_s, base.s0_fwhm_s, delta=0.05e-6)
        _drive, shadowed = self._mode_wave(distance, 5, "S0", holes=[(center, radius)], shadow=2.3)
        shadowed_features = extract_trace(
            shadowed, self.timebase.time_s, self.timebase.fs_hz, distance, drive,
            self.curve.a0_group_at_drive, self.curve.s0_group_at_drive, 0.0,
        )
        # The exponent is a stand-in. The waveform must still scale by exp(-shadow * depth).
        self.assertAlmostEqual(shadowed_features.s0_peak_v / damaged.s0_peak_v, np.exp(-2.3), places=4)

    def test_coda_fraction_is_what_broadens_the_envelope(self) -> None:
        # The fixed coda fraction is a surrogate for unresolved rim reverberation.
        # On a diametral ray it stretches the A0 envelope. Turning it off leaves
        # the width at the tone-burst width. This is not a scattering coefficient.
        distance = 0.09
        drive = Drive(frequency_hz=DRIVE_HZ, voltage=15.0, n_cycles=5)
        _excitation, spectrum, a0_scale, s0_scale = synthesize_baseline_scales(
            self.curve, self.timebase, drive, distance
        )
        start = np.array([0.0, 0.0])
        end = np.array([distance, 0.0])
        hole = [((0.5 * distance, 0.0), 0.006)]

        def waveform(holes, coda: float) -> np.ndarray:
            with _single_ray():
                return propagate_pair(
                    start, end, holes, spectrum, self.curve, self.timebase,
                    a0_scale, s0_scale, 2.3, 0.08, 0.34 * drive.duration_s, coda,
                )

        def a0_width(wave: np.ndarray) -> float:
            features = extract_trace(
                wave, self.timebase.time_s, self.timebase.fs_hz, distance, drive,
                self.curve.a0_group_at_drive, self.curve.s0_group_at_drive, 0.0,
            )
            return features.a0_fwhm_s

        pristine = a0_width(waveform([], 0.0))
        blocked = a0_width(waveform(hole, 0.0))
        with_coda = a0_width(waveform(hole, 0.6))
        self.assertAlmostEqual(blocked, pristine, delta=0.4e-6)
        self.assertGreater(with_coda, pristine + 1.0e-6)

    def test_reference_volts_are_a_calibration_not_the_bessel_tuning_ratio(self) -> None:
        # J1(x)/x -> 1/2 is the long-wavelength limit of the circular-wafer factor.
        tiny = 1e-4
        self.assertAlmostEqual(j1(tiny) / tiny, 0.5, places=6)
        radius = 3.5e-3
        factors = {}
        for mode, speed in (("A0", self.curve.a0_phase_at_drive), ("S0", self.curve.s0_phase_at_drive)):
            xi = 2.0 * np.pi * DRIVE_HZ / speed
            argument = xi * radius
            factors[mode] = abs(j1(argument) / argument)
        self.assertGreater(factors["S0"], factors["A0"])
        calibration = A0_REFERENCE_VOLT / S0_REFERENCE_VOLT
        bessel = factors["A0"] / factors["S0"]
        self.assertGreater(abs(calibration - bessel), 0.5)
        self.assertEqual(A0_REFERENCE_VOLT, 2.40)
        self.assertEqual(S0_REFERENCE_VOLT, 1.05)

    def test_hankel_amplitude_falls_as_wavenumber_grows(self) -> None:
        distance = 0.12
        drive, wave = self._mode_wave(distance, 5, "S0")
        source = toneburst(self.timebase.time_s, drive)
        f1, f2 = 440e3, DRIVE_HZ
        observed = self._spectral_magnitude(wave, f1) / self._spectral_magnitude(source, f1)
        observed /= self._spectral_magnitude(wave, f2) / self._spectral_magnitude(source, f2)
        k1 = float(self.curve.wavenumber(np.array([f1]), "S0")[0])
        k2 = float(self.curve.wavenumber(np.array([f2]), "S0")[0])
        k_ref = k2
        alpha1 = S0_ATTEN_NP_PER_M * k1 / k_ref
        alpha2 = S0_ATTEN_NP_PER_M
        expected = np.sqrt(k2 / k1) * np.exp(-(alpha1 - alpha2) * distance)
        self.assertAlmostEqual(observed, expected, delta=0.02 * expected)

    def test_propagation_is_reciprocal(self) -> None:
        drive = Drive(n_cycles=5)
        _excitation, spectrum, a0_scale, s0_scale = synthesize_baseline_scales(
            self.curve, self.timebase, drive, 0.09
        )
        start = np.array([0.0, 0.02])
        end = np.array([0.09, 0.07])
        holes = [((0.04, 0.045), 0.007)]
        forward = propagate_pair(
            start, end, holes, spectrum, self.curve, self.timebase,
            a0_scale, s0_scale, 2.3, 0.08, 1e-6, 0.2,
        )
        backward = propagate_pair(
            end, start, holes, spectrum, self.curve, self.timebase,
            a0_scale, s0_scale, 2.3, 0.08, 1e-6, 0.2,
        )
        self.assertTrue(np.allclose(forward, backward, rtol=1e-6, atol=1e-8))


class GeometryAndSolverTests(unittest.TestCase):
    def test_columns_are_equidistant_and_span_the_grid(self) -> None:
        coupon = make_coupon(0.09, 6)
        left = [t for t in coupon.transducers if t.side == "L"]
        right = [t for t in coupon.transducers if t.side == "R"]
        spacing = 0.09 / 5.0
        for column, x in ((left, 0.0), (right, 0.09)):
            ys = [t.y for t in column]
            self.assertAlmostEqual(ys[0], 0.09, places=9)
            self.assertAlmostEqual(ys[-1], 0.0, places=9)
            self.assertTrue(np.allclose(np.diff(ys), -spacing))
            self.assertTrue(all(t.x == x for t in column))
        labels = [coupon.transducers[i].label for i in coupon.fire_order]
        self.assertEqual(labels, ["L1", "L2", "L3", "L4", "L5", "L6", "R6", "R5", "R4", "R3", "R2", "R1"])
        self.assertEqual(coupon.cell_center(1, 1), (0.045, 0.045))
        # The sheet is the array span plus one span of free plate on every side.
        self.assertAlmostEqual(coupon.margin_m, 0.09)
        self.assertAlmostEqual(coupon.plate_width_m, 0.27)
        xmin, ymin, xmax, ymax = coupon.plate_bounds()
        self.assertAlmostEqual(xmin, -0.09)
        self.assertAlmostEqual(ymin, -0.09)
        self.assertAlmostEqual(xmax, 0.18)
        self.assertAlmostEqual(ymax, 0.18)

    def test_path_lengths_add_up_to_the_clipped_segment(self) -> None:
        width = 0.09
        segments = (
            ((0.0, 0.02), (0.09, 0.07)),
            ((0.0, 0.09), (0.09, 0.0)),
            ((0.01, 0.0), (0.08, 0.09)),
            ((-0.02, 0.04), (0.11, 0.05)),
        )
        for p0, p1 in segments:
            lengths = grid_path_lengths(p0, p1, width)
            clipped_span = float(np.hypot(min(max(p1[0], 0.0), width) - min(max(p0[0], 0.0), width), 0.0))
            # Use the same clip the grid walker uses, via the sum of a monotone interior ray.
            self.assertAlmostEqual(float(np.sum(lengths)), self._interior_length(p0, p1, width), places=5)
            self.assertGreaterEqual(float(np.min(lengths)), -1e-12)
            self.assertGreaterEqual(clipped_span, 0.0)
        edge = grid_path_lengths((0.0, 0.0), (0.0, width), width)
        self.assertEqual(float(np.sum(edge)), 0.0)
        same_side = grid_path_lengths((0.09, 0.09), (0.09, 0.0), width)
        self.assertEqual(float(np.sum(same_side)), 0.0)

    @staticmethod
    def _interior_length(p0, p1, width: float) -> float:
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for origin, delta in ((x0, dx), (y0, dy)):
            if abs(delta) < 1e-15:
                if origin < -1e-12 or origin > width + 1e-12:
                    return 0.0
                continue
            ta, tb = (0.0 - origin) / delta, (width - origin) / delta
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return 0.0
        return float(np.hypot(dx, dy) * (t1 - t0))

    def test_aperture_weights_are_a_partition_of_unity(self) -> None:
        offsets, weights = aperture_offsets()
        self.assertEqual(len(offsets), 5)
        self.assertAlmostEqual(float(offsets[0]), -3.5e-3, places=9)
        self.assertAlmostEqual(float(offsets[-1]), 3.5e-3, places=9)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
        self.assertTrue(np.all(weights > 0.0))

    def test_sirt_recovers_a_nonnegative_slowness_and_clips_the_rest(self) -> None:
        kernel = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.4, 0.4, 1.0],
                [0.0, 0.2, 0.8],
            ]
        )
        truth = np.array([0.2, 0.0, 1.5])
        recovered = sirt(kernel, kernel @ truth, n_iter=80)
        self.assertTrue(np.allclose(recovered, truth, atol=1e-6))
        clipped = sirt(np.eye(2), np.array([-1.0, 0.3]), n_iter=20)
        self.assertEqual(float(clipped[0]), 0.0)
        self.assertAlmostEqual(float(clipped[1]), 0.3, places=6)

    def test_phase_unwrap_follows_the_shift_theorem(self) -> None:
        tau = 4.7 / DRIVE_HZ
        phase = float(np.angle(np.exp(-1j * 2.0 * np.pi * DRIVE_HZ * tau)))
        self.assertAlmostEqual(phase_delay_s(phase, 0.0, DRIVE_HZ, tau), tau, places=9)
        self.assertTrue(np.isnan(phase_delay_s(np.nan, 0.0, DRIVE_HZ, 0.0)))
        self.assertAlmostEqual(wrapped_delta(0.2, -0.2), 0.4, places=9)
        advance = -2.3 / DRIVE_HZ
        phase = float(np.angle(np.exp(-1j * 2.0 * np.pi * DRIVE_HZ * advance)))
        self.assertAlmostEqual(phase_delay_s(phase, 0.0, DRIVE_HZ, advance), advance, places=9)
        self.assertTrue(np.isnan(phase_delay_s(0.2, 0.0, 0.0, 0.0)))

    def test_one_neper_of_amplitude_is_about_8_686_db_and_tof_is_linear(self) -> None:
        # A ratio e^{-1} in amplitude is 20 log10(e) dB, not the power factor 10 log10(e).
        coupon = make_coupon(0.09, 6)
        cell = coupon.cell_m
        path = np.zeros((3, 3))
        path[1, 1] = cell

        def feat(peak: float, tof: float) -> TraceFeatures:
            return TraceFeatures(True, cell, peak, 0.0, tof, 1e-6, 0.0, 0.1, 0.0, tof, 1e-6, 0.0)

        ray = RayRecord(
            shot=0, transmitter=0, receiver=1, distance_m=cell, path_m=path,
            measured=feat(np.exp(-1.0), 2e-6), baseline=feat(1.0, 1e-6),
        )
        image = reconstruct(coupon, [ray], DRIVE_HZ)
        self.assertAlmostEqual(float(image.amplitude_db[1, 1]), 20.0 / np.log(10.0), places=4)
        self.assertAlmostEqual(float(image.tof_us[1, 1]), 1.0, places=4)
        self.assertAlmostEqual(float(np.sum(image.amplitude_db) - image.amplitude_db[1, 1]), 0.0, places=6)

    def test_keller_path_off_the_diameter_and_at_the_edges(self) -> None:
        radius = 0.006
        start = np.array([0.0, 0.0])
        end = np.array([0.09, 0.0])
        center = np.array([0.045, 0.003])
        hit = hole_hit(tuple(center), radius, start, end)
        self.assertAlmostEqual(hit.extra_m, _keller_extra(start, end, center, radius), places=6)
        self.assertGreater(hit.extra_m, 0.0)
        self.assertAlmostEqual(hit.depth, np.sqrt(1.0 - (0.003 / radius) ** 2), places=6)
        center_hit = hole_hit((0.045, 0.0), radius, start, end)
        grazing = hole_hit((0.045, radius * 0.98), radius, start, end)
        self.assertLess(grazing.extra_m, 0.05 * center_hit.extra_m)
        self.assertEqual(hole_hit((0.045, 0.0), 0.0, start, end).extra_m, 0.0)
        self.assertEqual(hole_hit((0.045, 0.0), -0.01, start, end).extra_m, 0.0)
        buried = hole_hit((0.0, 0.0), 0.01, np.array([0.0, 0.0]), end)
        self.assertEqual(buried.extra_m, 0.0)
        self.assertGreater(buried.depth, 0.0)
        with self.assertRaises(ValueError):
            make_coupon(0.09, 1)
        with self.assertRaises(ValueError):
            make_coupon(0.0, 4)

    def test_grid_edges_corners_and_an_exact_envelope_width(self) -> None:
        width = 0.09
        diagonal = grid_path_lengths((0.0, 0.0), (width, width), width)
        self.assertAlmostEqual(float(np.sum(diagonal)), width * np.sqrt(2.0), places=4)
        outside = grid_path_lengths((-0.2, 0.4), (-0.05, 0.1), width)
        self.assertEqual(float(np.sum(outside)), 0.0)
        boundary = grid_path_lengths((width / 3.0, 0.0), (width / 3.0, width), width)
        self.assertAlmostEqual(float(np.sum(boundary)), width, places=4)
        self.assertEqual(int(np.count_nonzero(np.sum(boundary, axis=0) > 1e-6)), 1)
        time_s = np.linspace(0.0, 1.0, 2001)
        env = np.maximum(0.0, 1.0 - np.abs(time_s - 0.4) / 0.2)
        width_s = fwhm(time_s, env, int(np.argmax(env)), 1.0)
        self.assertAlmostEqual(width_s, 0.2, delta=2.0 / 2000.0)

    def test_fundamental_packets_are_separated_on_the_coupon(self) -> None:
        # The extractor splits S0 and A0 at the midpoint of the two group arrivals.
        # That gate is valid on this sheet only while the packets do not overlap.
        curve = lamb_curve(Aluminum6061(), drive_hz=DRIVE_HZ)
        gap = 0.09 * (1.0 / curve.a0_group_at_drive - 1.0 / curve.s0_group_at_drive)
        self.assertGreater(gap, Drive(n_cycles=5).duration_s)


def _a0_residual(speed: float, frequency: float, plate: Aluminum6061) -> float:
    """Antisymmetric Rayleigh-Lamb residual for c < c_T, in real hyperbolic functions."""
    omega = 2.0 * np.pi * frequency
    k = omega / speed
    p_mag = np.sqrt(k**2 - (omega / plate.c_longitudinal) ** 2)
    q_mag = np.sqrt(k**2 - (omega / plate.c_shear) ** 2)
    half = 0.5 * plate.thickness_m
    membrane = (q_mag**2 + k**2) ** 2
    return float(
        membrane * np.cosh(q_mag * half) * np.sinh(p_mag * half)
        - 4.0 * k**2 * p_mag * q_mag * np.sinh(q_mag * half) * np.cosh(p_mag * half)
    )


def _s0_residual(speed: float, frequency: float, plate: Aluminum6061) -> float:
    """Symmetric Rayleigh-Lamb residual for c_T < c < c_L."""
    omega = 2.0 * np.pi * frequency
    k = omega / speed
    q = np.sqrt((omega / plate.c_shear) ** 2 - k**2)
    p_mag = np.sqrt(k**2 - (omega / plate.c_longitudinal) ** 2)
    half = 0.5 * plate.thickness_m
    membrane = (q**2 - k**2) ** 2
    return float(
        membrane * np.sin(q * half) * np.cosh(p_mag * half)
        - 4.0 * k**2 * p_mag * q * np.cos(q * half) * np.sinh(p_mag * half)
    )


def _keller_extra(start: np.ndarray, end: np.ndarray, center: np.ndarray, radius: float) -> float:
    source = np.asarray(start, float) - np.asarray(center, float)
    receiver = np.asarray(end, float) - np.asarray(center, float)
    d1 = float(np.linalg.norm(source))
    d2 = float(np.linalg.norm(receiver))
    alpha1 = float(np.arccos(radius / d1))
    alpha2 = float(np.arccos(radius / d2))
    psi = float(np.arccos(np.clip(np.dot(source, receiver) / (d1 * d2), -1.0, 1.0)))
    theta = psi - alpha1 - alpha2
    straight = float(np.linalg.norm(np.asarray(end, float) - np.asarray(start, float)))
    creep = np.sqrt(d1**2 - radius**2) + np.sqrt(d2**2 - radius**2) + radius * theta
    return float(creep - straight)


if __name__ == "__main__":
    unittest.main()
