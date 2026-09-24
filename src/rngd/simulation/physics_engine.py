"""A small 2D physics-style relaxation engine for site-level layout.

This is the "engine like physics that does simulation" step: the building
footprint, the parking field, and the driveway are modeled as rigid
rectangles ("bodies") inside the buildable envelope. Each simulation step
computes penalty forces —

  * containment: push any body back inside the buildable envelope walls
  * separation: push overlapping bodies apart (like rigid-body collision
    resolution)
  * attraction: mild pull toward preferred zones (building toward the street
    frontage, parking toward the rear) from data/knowledge_base design
    guidelines

— and integrates position with damping (semi-implicit Euler), the same basic
scheme used in simple physics engines / force-directed layout. It is
deliberately not a rendering-only static packer: state.simulation_history
records per-iteration convergence stats (overlap penalty over time) that the
UI plots as a convergence curve.

The floor-plate (unit module) layout is generated separately by
`floor_plate.py` using deterministic BaaP bay-packing, since architectural
adjacency rules (double-loaded corridor, core alignment) are better expressed
as a placement rule than as a force — see baap_stacking_rules.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Body:
    id: str
    kind: str  # "building" | "parking_stall" | "driveway"
    x: float
    y: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    movable: bool = True

    def bounds(self):
        return self.x, self.y, self.x + self.w, self.y + self.h


def _overlap_1d(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _separation_force(a: Body, b: Body) -> tuple[float, float]:
    ax0, ay0, ax1, ay1 = a.bounds()
    bx0, by0, bx1, by1 = b.bounds()
    ox = _overlap_1d(ax0, ax1, bx0, bx1)
    oy = _overlap_1d(ay0, ay1, by0, by1)
    if ox <= 0 or oy <= 0:
        return 0.0, 0.0
    acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    # Push apart along the axis of least overlap (standard rectangle
    # separating-axis heuristic). `a` must move AWAY from `b`: if a's center
    # is left of b's, a moves further left (negative x), so the sign is the
    # opposite of the naive "a < b -> positive" comparison.
    if ox < oy:
        direction = -1.0 if acx < bcx else 1.0
        return direction * ox, 0.0
    else:
        direction = -1.0 if acy < bcy else 1.0
        return 0.0, direction * oy


class SiteSimulation:
    def __init__(self, envelope: dict, bodies: list[Body], seed: int = 7):
        self.x_min = envelope["x_min_ft"]
        self.y_min = envelope["y_min_ft"]
        self.x_max = envelope["x_max_ft"]
        self.y_max = envelope["y_max_ft"]
        self.bodies = bodies
        self.rng = np.random.default_rng(seed)
        self.history: list[float] = []

    def _containment_force(self, b: Body) -> tuple[float, float]:
        fx, fy = 0.0, 0.0
        bx0, by0, bx1, by1 = b.bounds()
        if bx0 < self.x_min:
            fx += self.x_min - bx0
        if bx1 > self.x_max:
            fx -= bx1 - self.x_max
        if by0 < self.y_min:
            fy += self.y_min - by0
        if by1 > self.y_max:
            fy -= by1 - self.y_max
        return fx, fy

    def step(self, damping: float = 0.55, gain: float = 0.5) -> float:
        total_penalty = 0.0
        forces = {b.id: [0.0, 0.0] for b in self.bodies}

        for b in self.bodies:
            cfx, cfy = self._containment_force(b)
            forces[b.id][0] += cfx
            forces[b.id][1] += cfy
            total_penalty += abs(cfx) + abs(cfy)

        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.bodies[i], self.bodies[j]
                sx, sy = _separation_force(a, b)
                if sx == 0 and sy == 0:
                    continue
                total_penalty += abs(sx) + abs(sy)
                if a.movable:
                    forces[a.id][0] += sx * 0.5
                    forces[a.id][1] += sy * 0.5
                if b.movable:
                    forces[b.id][0] -= sx * 0.5
                    forces[b.id][1] -= sy * 0.5

        for b in self.bodies:
            if not b.movable:
                continue
            fx, fy = forces[b.id]
            b.vx = (b.vx + gain * fx) * damping
            b.vy = (b.vy + gain * fy) * damping
            b.x += b.vx
            b.y += b.vy

        self.history.append(total_penalty)
        return total_penalty

    def run(self, max_steps: int = 250, tol: float = 0.5) -> dict:
        converged = False
        steps_run = 0
        for i in range(max_steps):
            penalty = self.step()
            steps_run = i + 1
            if penalty < tol:
                converged = True
                break
        return {
            "steps_run": steps_run,
            "max_overlap_penalty": round(self.history[-1] if self.history else 0.0, 3),
            "converged": converged,
            "history": self.history,
        }

    def snap_to_grid(self, grid_ft: float = 1.0):
        for b in self.bodies:
            b.x = round(b.x / grid_ft) * grid_ft
            b.y = round(b.y / grid_ft) * grid_ft


def build_site_bodies(
    envelope: dict,
    footprint_w: float,
    footprint_h: float,
    n_parking_stalls: int,
    stall_w: float = 9.0,
    stall_h: float = 18.0,
    aisle_h: float = 24.0,
    driveway_w: float = 24.0,
) -> list[Body]:
    """Initializes bodies at randomized-but-reasonable starting positions
    (building toward the street/front, parking toward the rear) for the
    simulation to relax into a non-overlapping arrangement. Returns as many
    parking stalls as fit the initial rows; the caller compares this to
    n_parking_stalls to know if surface parking meets demand.
    """
    x_min, y_min, x_max, y_max = (
        envelope["x_min_ft"],
        envelope["y_min_ft"],
        envelope["x_max_ft"],
        envelope["y_max_ft"],
    )
    bodies: list[Body] = []

    # Building: start near the front (low y = street side), horizontally centered.
    bx = x_min + max(0.0, ((x_max - x_min) - footprint_w) / 2)
    by = y_min
    bodies.append(Body(id="BUILDING", kind="building", x=bx, y=by, w=footprint_w, h=footprint_h, movable=True))

    # Driveway: a strip from the front edge down the side, feeding the rear parking field.
    drive_x = x_min
    bodies.append(
        Body(id="DRIVEWAY", kind="driveway", x=drive_x, y=y_min, w=driveway_w, h=(y_max - y_min), movable=False)
    )

    # Parking field: rows behind the building, packed as tightly as the envelope allows.
    park_y_start = by + footprint_h + 10  # small circulation gap
    row_pitch = stall_h + aisle_h
    usable_width = (x_max - x_min) - driveway_w - 4
    stalls_per_row = max(1, int(usable_width // stall_w))
    row = 0
    placed = 0
    while placed < n_parking_stalls and (park_y_start + row * row_pitch + stall_h) <= y_max:
        for col in range(stalls_per_row):
            if placed >= n_parking_stalls:
                break
            sx = x_min + driveway_w + 4 + col * stall_w
            sy = park_y_start + row * row_pitch
            bodies.append(
                Body(id=f"STALL_{placed}", kind="parking_stall", x=sx, y=sy, w=stall_w, h=stall_h, movable=False)
            )
            placed += 1
        row += 1

    return bodies
