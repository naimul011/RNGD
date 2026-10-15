"""Generative hotel floor plates for the Hotel Schematic optimizer.

Built from the 10/08/2026 team meeting (M. Edmonds' review of the GA result):
  * do not fix the shape: generate Bar, L, U, T, H and Courtyard plates inside the buildable
    area and let the user dictate one shape, ask for all, or a blend ("at least one rectangle
    and one L, then everything else");
  * rooms run perpendicular to a central corridor (double-loaded, one room depth from the
    facade); wide sites get a second wing (H / U / Courtyard) instead of windowless rooms;
  * the core is generative too: elevator core + stairs are placed by scheme (ends, centre,
    knuckle) and the search picks the best; the elevator count follows the key count;
  * if the keys do not fit, add a floor (the floor count is part of the search);
  * report an upper bound on keys per floor (frontage / module) so "can 72 fit?" is answered.

The Hotel Schematic program (room type, count, total SF, category) becomes an optimizer Program
(BaaP room sizes: width along the corridor = area / 26 ft depth). Each plate becomes a Site of
1 ft cells, so planner.py (decoder) and metrics.py (evaluator) run on it unchanged. All of this
is deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..hotel.program import Program as HotelProgram
from .program import Adjacency, Program, RoomType
from .site import Site, site_from_grids

ROOM_DEPTH_FT = 26.0
CORRIDOR_FT = 6.0
STAIR_CORE = (14, 26)   # one exit stair (ft along the wing, ft deep)
MARGIN = 2              # OUTSIDE cells around the plate
SHAPES = ("Bar", "L", "U", "T", "H", "Courtyard")
SHAPE_TEXT = {"Bar": "rectangle: one double-loaded bar", "L": "two wings at a corner", "U": "three wings around a forecourt",
              "T": "bar with a central stem", "H": "two parallel bars and a link", "Courtyard": "four wings around a court"}
CORE_SCHEMES = {"ends": "elevator core near one end, stair near the other",
                "center": "elevator core mid-plate, a stair near each end",
                "knuckle": "elevator core at the wing junction, stairs at the wing ends"}

SKIP = ("corridor", "stair")   # generated (corridor) or inside the cores (stairs)
L1_PREFS = {  # name word -> (category, exterior, near_core, priority)
    "lobby": ("public", True, True, 10), "retail": ("public", True, False, 6), "gym": ("public", True, False, 5),
    "kitchen": ("service", False, True, 6), "pantry": ("service", False, True, 6), "mens": ("public", False, True, 6),
    "womens": ("public", False, True, 6), "luggage": ("public", False, True, 5), "loading": ("service", True, False, 5),
    "mep": ("service", False, False, 5), "bus": ("public", False, False, 6),
}


def elevators_for(keys: int) -> int:
    """Rule of thumb for select-service hotels: about one elevator per 75 keys, at least two."""
    return max(2, math.ceil(keys / 75))


def main_core_size(keys: int) -> tuple[int, int]:
    """Stair + elevators on both sides of an elevator lobby + housekeeping (ft along wing, ft deep)."""
    e = elevators_for(keys)
    return (14 + 10 + 8 * math.ceil(e / 2) + (8 if e > 2 else 0), 26)


# ------------------------------------------------------------------ program
def _rt_for_row(name, count, area, cat, upper, ground) -> RoomType | None:
    low = name.lower()
    if count <= 0 or area <= 0 or any(low.startswith(s) for s in SKIP):
        return None
    each = area / count
    depth = ROOM_DEPTH_FT
    if cat == "guest":
        kind = "ada" if "ada" in low else ("suite" if "suite" in low else "guest")
        return RoomType(name, max(9.0, round(each / depth, 1)), depth, count,
                        priority=9 if kind == "ada" else (6 if kind == "suite" else 5), floors=list(upper),
                        category=kind, exterior=True, near_core=kind == "ada", corner="corner" in low,
                        min_per_floor=0)
    if count == len(upper) and each < 200:   # Ice, IT, Jan, Supplies: one per guest floor
        d = depth if each > 120 else max(6.0, round(math.sqrt(each), 1))
        return RoomType(name, max(5.0, round(each / d, 1)), d, 1, priority=7, per_floor=True, floors=list(upper),
                        category="service", exterior=False, near_core=True)
    pref = next((v for k, v in L1_PREFS.items() if k in low), None)
    rcat, ext, near, pri = pref or (("public", True, False, 5) if cat in ("shared", "tenant", "business")
                                    else ("service", False, True, 6))
    d = depth if each > 150 else max(6.0, round(math.sqrt(each), 1))
    return RoomType(name, max(5.0, round(each / d, 1)), d, count, priority=pri, floors=[ground], category=rcat,
                    exterior=ext, near_core=near)


def level_names(guest_floors: int) -> list[str]:
    return [f"Level {i + 1}" for i in range(guest_floors + 1)]


def program_from_hotel(hp: HotelProgram, names: list[str]) -> Program:
    """Hotel Schematic rows -> optimizer Program on the given levels (first = ground)."""
    ground, upper = names[0], names[1:]
    rooms = [rt for rt in (_rt_for_row(n, c, a, cat, upper, ground) for n, c, a, cat in hp.rows) if rt]
    for rt in rooms:   # ADA 224.5: spread ADA rooms over the floors
        if rt.category == "ada":
            rt.min_per_floor = 0
    lower = {rt.name.lower(): rt.name for rt in rooms}

    def find(word):
        return next((v for k, v in lower.items() if word in k), None)

    adj = []
    for a, b, w, d in (("lobby", "luggage", 1.0, 30), ("lobby", "mens", 0.8, 40), ("lobby", "womens", 0.8, 40),
                       ("lobby", "kitchen", 0.6, 50), ("loading", "mep", 0.5, 50), ("loading", "housekeeping", 0.6, 50),
                       ("lobby", "gym", 0.4, 60), ("bus", "lobby", 0.5, 50)):
        x, y = find(a), find(b)
        if x and y and x != y:
            adj.append(Adjacency(x, y, w, d))
    return Program(rooms=rooms, adjacency=adj, door_width_ft=3.0, corridor_width_ft=CORRIDOR_FT,
                   exterior_slack_ft=6.0, source=f"Hotel Schematic program ({hp.source})",
                   width_along_corridor=True)


def need_by_level(prog: Program, names: list[str]) -> list[float]:
    per = [0.0] * len(names)
    for inst in prog.instances(names):
        for l in inst.levels:
            per[l] += inst.type.area_sf / len(inst.levels)
    return per


# ------------------------------------------------------------------ plates
@dataclass
class Plate:
    name: str
    shape: str
    core_scheme: str
    rects: list[tuple[int, int, int, int]]                 # wings (x0, y0, w, h), ft
    cores: list[tuple[int, int, int, int, str]] = field(default_factory=list)   # (x0, y0, w, h, kind)
    clipped: bool = False                                  # shrunk to fit the buildable area

    @property
    def size(self) -> tuple[int, int]:
        return (max(x + w for x, y, w, h in self.rects), max(y + h for x, y, w, h in self.rects))

    def footprint_sf(self) -> int:
        m = np.zeros(self.size[::-1], bool)
        for x, y, w, h in self.rects:
            m[y:y + h, x:x + w] = True
        return int(m.sum())

    def facade_ft(self) -> int:
        m = np.zeros((self.size[1] + 2, self.size[0] + 2), bool)
        for x, y, w, h in self.rects:
            m[y + 1:y + h + 1, x + 1:x + w + 1] = True
        return int((m[1:, :] != m[:-1, :]).sum() + (m[:, 1:] != m[:, :-1]).sum())


def wing_depth() -> int:
    return int(2 * ROOM_DEPTH_FT + CORRIDOR_FT)


def _wings(shape: str, a: int, b: int, wd: int) -> list[tuple[int, int, int, int]]:
    if shape == "Bar":
        return [(0, 0, a, wd)]
    if shape == "L":
        return [(0, 0, a, wd), (0, 0, wd, b)]
    if shape == "U":
        return [(0, 0, a, wd), (0, 0, wd, b), (a - wd, 0, wd, b)]
    if shape == "T":
        return [(0, b - wd, a, wd), ((a - wd) // 2, 0, wd, b)]
    if shape == "H":
        return [(0, 0, wd, b), (a - wd, 0, wd, b), (wd, (b - wd) // 2, a - 2 * wd, wd)]
    if shape == "Courtyard":
        return [(0, 0, a, wd), (0, b - wd, a, wd), (0, 0, wd, b), (a - wd, 0, wd, b)]
    raise ValueError(shape)


def _run_ft(shape: str, a: int, b: int, wd: int) -> float:
    """Length of double-loaded corridor the shape offers (corners count once)."""
    return {"Bar": a, "L": a + b - wd, "U": a + 2 * b - 2 * wd, "T": a + b - wd, "H": 2 * b + a - 2 * wd,
            "Courtyard": 2 * a + 2 * b - 4 * wd}[shape]


def _min_dims(shape: str, wd: int) -> tuple[int, int]:
    court = 40   # light court / forecourt between facing wings
    return {"Bar": (3 * wd, wd), "L": (2 * wd, 2 * wd), "U": (2 * wd + court, wd + court), "T": (2 * wd + 20, 2 * wd),
            "H": (2 * wd + court, 2 * wd + 20), "Courtyard": (2 * wd + court, 2 * wd + court)}[shape]


def _dims_for(shape: str, run: float, ratio: float, wd: int, env: tuple[int, int]) -> tuple[int, int, bool]:
    """Bounding box (a, b) giving about `run` ft of corridor at aspect b/a = ratio, clipped to env."""
    amin, bmin = _min_dims(shape, wd)
    if shape == "Bar":
        a, b = int(math.ceil(run)), wd
    else:
        lo, hi = 1, 4000
        while lo < hi:   # smallest a with enough corridor
            mid = (lo + hi) // 2
            if _run_ft(shape, max(mid, amin), max(int(mid * ratio), bmin), wd) >= run:
                hi = mid
            else:
                lo = mid + 1
        a, b = max(lo, amin), max(int(lo * ratio), bmin)
    clipped = a > env[0] or b > env[1]
    a, b = min(a, env[0]), min(b, env[1])
    if a < amin or b < bmin:
        return 0, 0, True
    return a, b, clipped


def _place_core(wing, t: str, size, side: str):
    """Core on one facade side of a wing at its start / centre / end (one bay in from the end)."""
    x0, y0, w, h = wing
    cw, cd = size
    off = 26
    if w >= h:   # horizontal wing: core spans x, sits on the south (lo) or north (hi) facade
        x = {"start": x0 + off, "end": x0 + w - off - cw, "center": x0 + (w - cw) // 2}[t]
        y = y0 if side == "lo" else y0 + h - cd
        return (max(x0, x), y, cw, cd)
    y = {"start": y0 + off, "end": y0 + h - off - cw, "center": y0 + (h - cw) // 2}[t]
    x = x0 if side == "lo" else x0 + w - cd
    return (x, max(y0, y), cd, cw)


def _overlap(a, b) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _far_stairs(wings, st, main) -> list[tuple]:
    """Two stairs at the plate's extremities: wing-end positions, not in a wing junction or on the
    main core, as far apart as possible (IBC exit separation; short dead ends at both ends)."""
    cands = []
    for i, w in enumerate(wings):
        for t in ("start", "end"):
            for side in ("lo", "hi"):
                c = _place_core(w, t, st, side)
                others = [o for j, o in enumerate(wings) if j != i]
                if _overlap(c, main) or any(_overlap(c, o) for o in others):
                    continue
                cands.append(c)
    if len(cands) < 2:
        return [_place_core(wings[0], "start", st, "hi"), _place_core(wings[-1], "end", st, "hi")]
    ctr = lambda c: (c[0] + c[2] / 2, c[1] + c[3] / 2)  # noqa: E731
    best = max(((a, b) for i, a in enumerate(cands) for b in cands[i + 1:] if not _overlap(a, b)),
               key=lambda ab: (math.dist(ctr(ab[0]), ctr(ab[1])), -ab[0][1], -ab[0][0]))
    return list(best)


def _cores(shape: str, scheme: str, wings, main, wd) -> list[tuple] | None:
    st = STAIR_CORE
    W0 = wings[0]
    last = wings[-1] if shape not in ("Courtyard",) else wings[1]
    if scheme == "ends":
        if shape == "Bar":
            return [(*_place_core(W0, "start", main, "lo"), "main"), (*_place_core(W0, "end", st, "lo"), "stair")]
        far = wings[1] if shape in ("L", "T") else wings[2] if shape in ("U", "H") else wings[1]
        return [(*_place_core(W0, "end", main, "lo"), "main"), (*_place_core(far, "end", st, "lo"), "stair")]
    if scheme == "center":
        long = max(wings, key=lambda r: max(r[2], r[3]))
        m = _place_core(long, "center", main, "lo")
        return [(*m, "main")] + [(*c, "stair") for c in _far_stairs(wings, st, m)]
    if scheme == "knuckle":
        if shape == "Bar":
            return None
        # main core where the first two wings meet (inside corner), stairs at the far ends
        x0, y0, w, h = wings[0]
        x1, y1, w1, h1 = wings[1]
        if shape == "T":
            m = (x1 + wd, y0, main[0], main[1])      # bar's inner facade, beside the stem
            return [(*m, "main"), (*_place_core(wings[0], "start", st, "hi"), "stair"),
                    (*_place_core(wings[1], "start", st, "lo"), "stair")]
        if shape == "H":
            link = wings[2]
            m = (link[0] + (link[2] - main[0]) // 2, link[1], main[0], main[1])
            return [(*m, "main"), (*_place_core(wings[0], "start", st, "lo"), "stair"),
                    (*_place_core(wings[1], "end", st, "hi"), "stair")]
        m = (wd, wd - main[1], main[0], main[1])      # inside corner of the L / U / Courtyard
        far = [_place_core(wings[0], "end", st, "lo"), _place_core(wings[1], "end", st, "lo")]
        return [(*m, "main"), (*far[0], "stair"), (*far[1], "stair")]
    return None


def generate_plates(prog: Program, names: list[str], keys: int, envelope: tuple[int, int],
                    shapes=SHAPES, schemes=tuple(CORE_SCHEMES), ratios=(0.6, 1.0), slack: float = 1.04) -> list[Plate]:
    """All plates for this floor count: shape x proportion x core scheme, inside the envelope."""
    wd = wing_depth()
    main = main_core_size(keys)
    need = max(need_by_level(prog, names)) * slack + main[0] * main[1] + 2 * STAIR_CORE[0] * STAIR_CORE[1]
    run = need / (2 * ROOM_DEPTH_FT * 0.93)
    out, seen = [], set()
    for shape in shapes:
        for ratio in ((1.0,) if shape == "Bar" else ratios):
            for env in (envelope, envelope[::-1]):   # both orientations on the site
                a, b, clipped = _dims_for(shape, run, ratio, wd, env)
                if not a:
                    continue
                wings = _wings(shape, a, b, wd)
                for scheme in schemes:
                    cores = _cores(shape, scheme, wings, main, wd)
                    if not cores:
                        continue
                    key = (shape, a, b, scheme)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(Plate(f"{shape} {a}x{b} · {scheme} core", shape, scheme, wings, cores, clipped))
    return out


def keys_upper_bound(plate: Plate, module_ft: float) -> int:
    """Upper bound on keys per typical floor: both sides of every corridor ft / module, less the cores."""
    wd = wing_depth()
    run = _run_ft(plate.shape, *plate.size, wd)
    core_front = sum(max(c[2], c[3]) for c in plate.cores)
    return max(0, int((2 * run - core_front) // module_ft))


def plate_site(plate: Plate, names: list[str], elevations_ft: list[float]) -> Site:
    W, H = plate.size
    g = np.full((H + 2 * MARGIN, W + 2 * MARGIN), "OUTSIDE", dtype="<U9")
    for x, y, w, h in plate.rects:
        g[y + MARGIN:y + h + MARGIN, x + MARGIN:x + w + MARGIN] = "PLACEABLE"
    for x, y, w, h, _ in plate.cores:
        g[y + MARGIN:y + h + MARGIN, x + MARGIN:x + w + MARGIN] = "CORE"
    key = f"hotel_{plate.shape}_{W}x{H}_{plate.core_scheme}_{len(names)}lv"
    site = site_from_grids(key, [g.copy() for _ in names], names, elevations_ft)
    site.core_kinds = [_kind_at(plate, c.centroid) for c in site.cores]
    return site


def _kind_at(plate: Plate, centroid) -> str:
    r, c = centroid
    for x, y, w, h, k in plate.cores:
        if y + MARGIN <= r < y + h + MARGIN and x + MARGIN <= c < x + w + MARGIN:
            return k
    return "stair"
