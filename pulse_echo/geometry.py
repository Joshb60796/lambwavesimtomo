"""Coupon geometry: a square 3x3 region grid between two vertical PWAS columns.

The columns sit on the left and right edges of the grid. Their center-to-center
separation equals the grid width, and the top-to-bottom center span of each
column equals that same length. Transducers are equally spaced, including the
corner elements.

The aluminum sheet is larger than that grid. Each free edge is set outward by
one grid width, so the plate is three times the array span on a side and the
boundary is not part of the pitch-catch paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CELLS = 3
PWAS_DIAMETER_M = 7.0e-3


@dataclass(frozen=True)
class Transducer:
    index: int
    side: str
    ordinal: int
    x: float
    y: float

    @property
    def label(self) -> str:
        return f"{self.side}{self.ordinal}"


@dataclass(frozen=True)
class Coupon:
    width_m: float
    n_per_side: int
    transducers: tuple[Transducer, ...]
    fire_order: tuple[int, ...]

    @property
    def cell_m(self) -> float:
        return self.width_m / N_CELLS

    @property
    def margin_m(self) -> float:
        """Free-plate border beyond the array, on every side."""
        return self.width_m

    @property
    def plate_width_m(self) -> float:
        return self.width_m + 2.0 * self.margin_m

    def plate_bounds(self) -> tuple[float, float, float, float]:
        """xmin, ymin, xmax, ymax of the sheet. y increases upward."""
        margin = self.margin_m
        return (-margin, -margin, self.width_m + margin, self.width_m + margin)

    @property
    def n_transducers(self) -> int:
        return len(self.transducers)

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        """Row 0 is the top row. y increases upward."""
        cell = self.cell_m
        x = (col + 0.5) * cell
        y = self.width_m - (row + 0.5) * cell
        return (x, y)

    def hole_centers(self, holes: set[tuple[int, int]]) -> list[tuple[tuple[float, float], tuple[int, int]]]:
        return [(self.cell_center(r, c), (r, c)) for r, c in sorted(holes)]


def make_coupon(width_m: float = 0.090, n_per_side: int = 4) -> Coupon:
    if n_per_side < 2:
        raise ValueError("Need at least two transducers on a side")
    if width_m <= 0.0:
        raise ValueError("Plate width must be positive")
    elements: list[Transducer] = []
    for side, x in (("L", 0.0), ("R", float(width_m))):
        for ordinal in range(1, n_per_side + 1):
            # Ordinal 1 is the top element. Endpoints sit on the grid corners.
            y = width_m * (n_per_side - ordinal) / (n_per_side - 1)
            elements.append(Transducer(index=len(elements), side=side, ordinal=ordinal, x=float(x), y=float(y)))
    left = [t.index for t in elements if t.side == "L"]
    right = [t.index for t in elements if t.side == "R"]
    # Top-left down the left column, then bottom-right up the right column.
    fire = tuple(left + list(reversed(right)))
    return Coupon(width_m=float(width_m), n_per_side=n_per_side, transducers=tuple(elements), fire_order=fire)


def clip_segment(
    x0: float, y0: float, x1: float, y1: float, length: float
) -> tuple[float, float, float, float] | None:
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0
    for origin, delta, lower, upper in ((x0, dx, 0.0, length), (y0, dy, 0.0, length)):
        if abs(delta) < 1e-15:
            if origin < lower - 1e-12 or origin > upper + 1e-12:
                return None
            continue
        ta = (lower - origin) / delta
        tb = (upper - origin) / delta
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
        if t0 > t1:
            return None
    return (x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy)


def grid_path_lengths(p0: tuple[float, float], p1: tuple[float, float], width: float, n: int = N_CELLS) -> np.ndarray:
    """Path length inside each cell. Row 0 is the top row.

    A ray that rides the left or right transducer column, outside the open
    square, contributes nothing. Rays that merely graze a grid line are
    handled by the caller via a finite PWAS aperture.
    """
    out = np.zeros((n, n), dtype=float)
    x0, y0 = p0
    x1, y1 = p1
    if (x0 <= 1e-9 and x1 <= 1e-9) or (x0 >= width - 1e-9 and x1 >= width - 1e-9):
        return out
    if (y0 <= 1e-9 and y1 <= 1e-9) or (y0 >= width - 1e-9 and y1 >= width - 1e-9):
        return out
    clipped = clip_segment(x0, y0, x1, y1, width)
    if clipped is None:
        return out
    x0, y0, x1, y1 = clipped
    dx = x1 - x0
    dy = y1 - y0
    span = float(np.hypot(dx, dy))
    if span < 1e-12:
        return out
    cell = width / n
    eps = 1e-10

    def cell_index(value: float, toward: float) -> int:
        nudged = value + eps * np.sign(toward if abs(toward) > 0 else 1.0)
        index = int(np.floor(np.clip(nudged, 0.0, width - eps) / cell))
        return min(max(index, 0), n - 1)

    ix = cell_index(x0, dx)
    iy = cell_index(y0, dy)
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    t_delta_x = (cell / abs(dx)) if step_x else np.inf
    t_delta_y = (cell / abs(dy)) if step_y else np.inf

    def boundary_t(index: int, step: int, origin: float, delta: float) -> float:
        if step == 0:
            return np.inf
        boundary = (index + (1 if step > 0 else 0)) * cell
        return (boundary - origin) / delta

    t_max_x = boundary_t(ix, step_x, x0, dx)
    t_max_y = boundary_t(iy, step_y, y0, dy)
    t = 0.0
    for _ in range(n * n + 3):
        t_next = min(t_max_x, t_max_y, 1.0)
        segment = max(0.0, t_next - t) * span
        row = n - 1 - iy
        col = ix
        if segment > 0.0 and 0 <= row < n and 0 <= col < n:
            out[row, col] += segment
        if t_next >= 1.0 - 1e-12:
            break
        if t_max_x < t_max_y - 1e-15:
            ix += step_x
            t = t_max_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_x - 1e-15:
            iy += step_y
            t = t_max_y
            t_max_y += t_delta_y
        else:
            ix += step_x
            iy += step_y
            t = t_next
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        if not (0 <= ix < n and 0 <= iy < n):
            break
    return out


def aperture_offsets(radius: float = PWAS_DIAMETER_M / 2.0, n_rays: int = 5) -> tuple[np.ndarray, np.ndarray]:
    if n_rays < 1:
        raise ValueError("n_rays must be positive")
    if n_rays == 1:
        return np.array([0.0]), np.array([1.0])
    offsets = np.linspace(-radius, radius, n_rays)
    weights = np.cos(0.5 * np.pi * offsets / radius)
    weights = np.clip(weights, 0.0, None)
    weights /= np.sum(weights)
    return offsets, weights


@dataclass(frozen=True)
class HoleHit:
    depth: float
    chord_m: float
    extra_m: float


def _keller_extra(start: np.ndarray, end: np.ndarray, center: np.ndarray, radius: float, straight: float) -> float:
    """Extra length of the shortest Keller creeping path around a circular void.

    Keller, J. Opt. Soc. Am. 52 (1962) 116-130: the diffracted ray in the shadow
    is the tangent from the source, an arc of the boundary, and the tangent to
    the receiver. The arc between those tangent points is
    ψ − arccos(R/d1) − arccos(R/d2), with ψ the angle at the center between the
    endpoints. Replacing the straight chord by the semicircle between the pierce
    points is a longer path and is not this arrival.
    """
    source = start - center
    receiver = end - center
    d1 = float(np.hypot(source[0], source[1]))
    d2 = float(np.hypot(receiver[0], receiver[1]))
    if d1 <= radius or d2 <= radius or straight <= 0.0:
        return 0.0
    alpha1 = float(np.arccos(np.clip(radius / d1, -1.0, 1.0)))
    alpha2 = float(np.arccos(np.clip(radius / d2, -1.0, 1.0)))
    psi = float(np.arccos(np.clip(float(np.dot(source, receiver)) / (d1 * d2), -1.0, 1.0)))
    theta = psi - alpha1 - alpha2
    if theta <= 0.0:
        theta = (2.0 * np.pi - psi) - alpha1 - alpha2
    if theta <= 0.0:
        return 0.0
    creep = (
        float(np.sqrt(d1 * d1 - radius * radius))
        + float(np.sqrt(d2 * d2 - radius * radius))
        + radius * theta
    )
    return float(max(0.0, creep - straight))


def hole_hit(center: tuple[float, float], radius: float, start: np.ndarray, end: np.ndarray) -> HoleHit:
    """Shadow and Keller detour where a straight segment cuts a circular through-hole.

    `chord_m` is the straight intercept. `extra_m` is the Keller creeping path
    minus the straight length. `depth` is the normalized impact parameter,
    a geometric weight for the phenomenological shadow, not a diffraction coefficient.
    """
    if radius <= 0.0:
        return HoleHit(0.0, 0.0, 0.0)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    c = np.asarray(center, dtype=float)
    d = b - a
    length = float(np.hypot(d[0], d[1]))
    if length < 1e-12:
        return HoleHit(0.0, 0.0, 0.0)
    t_line = float(np.dot(c - a, d) / (length * length))
    foot = a + t_line * d
    t_seg = float(np.clip(t_line, 0.0, 1.0))
    closest = a + t_seg * d
    dist = float(np.hypot(*(closest - c)))
    if dist >= radius:
        return HoleHit(0.0, 0.0, 0.0)
    line_dist = float(np.hypot(*(foot - c)))
    if t_line < 0.0 or t_line > 1.0:
        depth = float(np.sqrt(max(0.0, 1.0 - (dist / radius) ** 2)))
        return HoleHit(depth, 0.0, 0.0)
    half = float(np.sqrt(max(radius**2 - line_dist**2, 0.0)))
    dt = half / length
    t_enter = max(0.0, t_line - dt)
    t_exit = min(1.0, t_line + dt)
    if t_exit <= t_enter:
        return HoleHit(0.0, 0.0, 0.0)
    chord = (t_exit - t_enter) * length
    depth = float(np.sqrt(max(0.0, 1.0 - (line_dist / radius) ** 2)))
    extra = _keller_extra(a, b, c, radius, length)
    return HoleHit(depth=depth, chord_m=chord, extra_m=extra)
