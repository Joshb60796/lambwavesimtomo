"""Localization checks for the synthetic Lamb-wave campaign."""

from __future__ import annotations

import unittest

import numpy as np

from pulse_echo.dispersion import lamb_curve
from pulse_echo.geometry import hole_hit, make_coupon, grid_path_lengths
from pulse_echo.material import Aluminum6061
from pulse_echo.simulate import SimConfig, run_campaign


class DispersionTests(unittest.TestCase):
    def test_fundamental_modes_sit_where_a_thin_sheet_should(self) -> None:
        curve = lamb_curve(Aluminum6061(), drive_hz=550e3)
        # Half-thickness Rayleigh-Lamb root on 1/32 in 6061-T6 at 550 kHz.
        # A plate solved with the full thickness in Rose's h is about sqrt(2) fast.
        self.assertGreater(curve.a0_phase_at_drive, 1650.0)
        self.assertLess(curve.a0_phase_at_drive, 1900.0)
        self.assertGreater(curve.a0_group_at_drive, curve.a0_phase_at_drive)
        self.assertLess(curve.a0_group_at_drive, 3000.0)
        self.assertGreater(curve.s0_phase_at_drive, 5300.0)
        self.assertLess(curve.s0_phase_at_drive, 5360.0)
        self.assertGreater(curve.s0_group_at_drive, 5200.0)
        self.assertLess(curve.s0_group_at_drive, curve.s0_phase_at_drive)
        self.assertTrue(np.all(np.diff(curve.a0_phase) > 0.0))


class GeometryTests(unittest.TestCase):
    def test_fire_order_walks_down_the_left_then_up_the_right(self) -> None:
        coupon = make_coupon(0.09, 4)
        labels = [coupon.transducers[i].label for i in coupon.fire_order]
        self.assertEqual(labels, ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"])
        left = [t for t in coupon.transducers if t.side == "L"]
        self.assertAlmostEqual(left[0].y, 0.09)
        self.assertAlmostEqual(left[-1].y, 0.0)
        self.assertAlmostEqual(coupon.transducers[0].x, 0.0)
        self.assertAlmostEqual(coupon.transducers[-1].x, 0.09)

    def test_horizontal_midline_crosses_one_row(self) -> None:
        lengths = grid_path_lengths((0.0, 0.045), (0.09, 0.045), 0.09)
        self.assertAlmostEqual(float(np.sum(lengths[1])), 0.09, places=4)
        self.assertAlmostEqual(float(np.sum(lengths[[0, 2]])), 0.0, places=6)

    def test_center_hole_adds_the_creeping_detour(self) -> None:
        radius = 0.004
        start = np.array([0.0, 0.045])
        end = np.array([0.09, 0.045])
        hit = hole_hit((0.045, 0.045), radius, start, end)
        # Keller tangent-arc-tangent, not the semicircle between the pierce points.
        distance_to_center = 0.045
        glide = np.arccos(radius / distance_to_center)
        creep = 2.0 * np.sqrt(distance_to_center**2 - radius**2) + radius * (np.pi - 2.0 * glide)
        self.assertAlmostEqual(hit.extra_m, creep - 0.09, places=6)
        self.assertLess(hit.extra_m, radius * (np.pi - 2.0))
        self.assertAlmostEqual(hit.chord_m, 2.0 * radius, places=6)
        self.assertAlmostEqual(hit.depth, 1.0, places=5)
        miss = hole_hit((0.045, 0.075), radius, np.array([0.0, 0.015]), np.array([0.09, 0.015]))
        self.assertEqual(miss.extra_m, 0.0)


class ReconstructionTests(unittest.TestCase):
    def _peak(self, image: np.ndarray) -> tuple[int, int]:
        return tuple(int(i) for i in np.unravel_index(int(np.argmax(image)), image.shape))

    def _campaign(self, holes: set[tuple[int, int]], seed: int = 4):
        # Six elements per side and a 12 mm hole put several independent
        # rays through every cell center on a 90 mm coupon.
        return run_campaign(SimConfig(snr_db=32, seed=seed), holes=holes)

    def test_empty_plate_stays_quiet(self) -> None:
        campaign = self._campaign(set())
        self.assertLess(float(np.max(campaign.reconstruction.fused)), 0.28)
        self.assertFalse(bool(np.any(campaign.reconstruction.hole_calls)))

    def test_every_single_hole_is_the_bright_cell(self) -> None:
        # Straight-ray SIRT with transducers only on the left and right edges
        # smears some energy into the neighboring cell (McKeon and Hinders,
        # J. Sound Vib. 224, 1999). With the Keller delay, the call is carried
        # by the amplitude shadow, and the brightest cell is the drilled one.
        for row in range(3):
            for col in range(3):
                campaign = self._campaign({(row, col)}, seed=4 + row * 3 + col)
                image = campaign.reconstruction.fused
                self.assertEqual(self._peak(image), (row, col))
                self.assertTrue(bool(campaign.reconstruction.hole_calls[row, col]))
                ordered = np.sort(image.ravel())
                self.assertGreater(float(ordered[-1]), float(ordered[-2]))
        center = self._campaign({(1, 1)}, seed=7)
        self.assertGreater(float(center.reconstruction.fused[1, 1]), 0.5)
        self.assertEqual(int(np.sum(center.reconstruction.hole_calls)), 1)

    def test_diagonal_holes_outrank_the_clear_cells(self) -> None:
        holes = {(0, 0), (1, 1), (2, 2)}
        campaign = self._campaign(holes, seed=11)
        image = campaign.reconstruction.fused
        damaged = [image[r, c] for r, c in holes]
        clear = [image[r, c] for r in range(3) for c in range(3) if (r, c) not in holes]
        self.assertGreater(min(damaged), max(clear))

    def test_three_count_burst_localizes_the_center(self) -> None:
        # A 3-count burst still resolves the center cell. A corner hole does not:
        # the Keller detour is too short to break the straight-ray amplitude tie
        # with the neighboring cell, which is the limited-angle artifact.
        for hole, seed in (({(1, 1)}, 4), (set(), 4)):
            campaign = run_campaign(SimConfig(snr_db=42, seed=seed, n_cycles=3), holes=hole)
            image = campaign.reconstruction.fused
            self.assertEqual(campaign.drive.n_cycles, 3)
            if not hole:
                self.assertLess(float(np.max(image)), 0.28)
                self.assertFalse(bool(np.any(campaign.reconstruction.hole_calls)))
                continue
            self.assertEqual(self._peak(image), (1, 1))
            self.assertGreater(float(image[1, 1]), 0.5)
            self.assertEqual(int(np.sum(campaign.reconstruction.hole_calls)), 1)
        corner = run_campaign(SimConfig(snr_db=42, seed=9, n_cycles=3), holes={(0, 0)})
        self.assertGreater(
            float(corner.reconstruction.fused[0, 1]),
            float(corner.reconstruction.fused[0, 0]),
        )


if __name__ == "__main__":
    unittest.main()
