"""Handbook properties for 6061-T6 sheet and the bench excitation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INCH = 0.0254


@dataclass(frozen=True)
class Aluminum6061:
    """Isotropic constants used for the Rayleigh-Lamb solver.

    Values are typical 6061-T6 handbook figures, not a heat-lot certificate.
    """

    young_pa: float = 68.9e9
    poisson: float = 0.33
    density: float = 2700.0
    thickness_m: float = (1.0 / 32.0) * INCH

    @property
    def shear_modulus(self) -> float:
        return self.young_pa / (2.0 * (1.0 + self.poisson))

    @property
    def lame_lambda(self) -> float:
        nu = self.poisson
        return self.young_pa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def c_shear(self) -> float:
        return float(np.sqrt(self.shear_modulus / self.density))

    @property
    def c_longitudinal(self) -> float:
        return float(np.sqrt((self.lame_lambda + 2.0 * self.shear_modulus) / self.density))

    @property
    def c_plate(self) -> float:
        return float(np.sqrt(self.young_pa / (self.density * (1.0 - self.poisson**2))))

    @property
    def thickness_mm(self) -> float:
        return self.thickness_m * 1e3

    @property
    def thickness_in(self) -> float:
        return self.thickness_m / INCH


@dataclass(frozen=True)
class Drive:
    """Hanning-windowed tone burst from a bipolar pulser."""

    frequency_hz: float = 550e3
    voltage: float = 15.0
    n_cycles: int = 5

    @property
    def duration_s(self) -> float:
        return self.n_cycles / self.frequency_hz

    @property
    def envelope_center_s(self) -> float:
        return 0.5 * self.duration_s
