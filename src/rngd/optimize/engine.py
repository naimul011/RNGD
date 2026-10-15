"""Genome -> corridor network -> room placements.

Genome (all genes in [0, 1]):
  * corridor: N_SEG segments x (u, v, orientation, length). A segment starts at the placeable
    cell nearest (v*rows, u*cols), runs horizontally (orientation < 0.5) or vertically, and
    extends both ways up to length*span/2 while the full corridor width is placeable on every
    level. Segments with length < 0.12 are switched off, so the number of corridors evolves.
  * order: one random key per room instance. Rooms are placed in order of
    priority/10 * PRIORITY_PULL + key, so priorities dominate but the search can reorder.
Repair: every core and every corridor piece is joined into one network with least-cost
paths (skimage MCP; reusing existing corridor is cheap), carved at the corridor width.
Packing: each room goes to the best valid anchor over all allowed levels, orientations and
facing directions; validity and scores are computed for every anchor at once with integral
images (rooms must open onto the corridor and fit entirely in free placeable cells).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric

from .program import Instance, Program
from .site import Site

N_SEG = 10
PRIORITY_PULL = 0.6
FAR = 9999.0
DIRS = ("N", "S", "E", "W")
# Slot-scoring weights used by the greedy packer (tunable from the app).
PACK = {"exterior": 1.0, "no_window": 1.2, "corner": 0.6, "core": 1.0, "core_pull": 0.15, "tight": 2.0, "back": 1.0}


@dataclass
class Placement:
    name: str
    type_name: str
    category: str
    level: int
    r0: int
    c0: int
    h: int
    w: int
    facing: str           # direction the room extends from the corridor
    frontage_ft: float
    depth_ft: float
    exterior: bool
    corner: bool
    door_dist_ft: float   # corridor distance from the door to the nearest core
    priority: float

    @property
    def center(self) -> tuple[float, float]:
        return self.r0 + self.h / 2, self.c0 + self.w / 2

    @property
    def area_sf(self) -> float:
        return self.frontage_ft * self.depth_ft


@dataclass
class Design:
    site: Site
    program: Program
    corridor: np.ndarray                 # bool R x C (same on every level)
    placements: list[Placement]
    unplaced: list[str]
    core_dist: np.ndarray                # corridor geodesic distance to nearest reached core (ft)
    cores_reached: int
    dead_end_ft: float
    genome: np.ndarray | None = None
    source: str = "optimizer"
    extra: dict = field(default_factory=dict)


def genome_length(n_instances: int) -> int:
    return N_SEG * 4 + n_instances


# --------------------------------------------------------------------- corridor
def _placeable_all(site: Site) -> np.ndarray:
    m = site.placeable[0].copy()
    for p in site.placeable[1:]:
        m &= p
    return m


def _carve_segments(site: Site, seg: np.ndarray, width: int, pall: np.ndarray, coords: np.ndarray) -> np.ndarray:
    R, C = site.shape
    mask = np.zeros((R, C), bool)
    half_w = width // 2
    for u, v, o, ln in seg:
        if ln < 0.12:
            continue
        target = np.array([v * (R - 1), u * (C - 1)])
        r, c = coords[np.argmin(((coords - target) ** 2).sum(1))]
        horizontal = o < 0.5
        reach = int(ln * (C if horizontal else R) / 2)
        if horizontal:
            r0 = int(np.clip(r - half_w, 0, R - width))
            band = pall[r0:r0 + width, :].all(0)
            if not band[c]:
                continue
            lo = c
            while lo - 1 >= 0 and band[lo - 1] and c - lo < reach:
                lo -= 1
            hi = c
            while hi + 1 < C and band[hi + 1] and hi - c < reach:
                hi += 1
            mask[r0:r0 + width, lo:hi + 1] = True
        else:
            c0 = int(np.clip(c - half_w, 0, C - width))
            band = pall[:, c0:c0 + width].all(1)
            if not band[r]:
                continue
            lo = r
            while lo - 1 >= 0 and band[lo - 1] and r - lo < reach:
                lo -= 1
            hi = r
            while hi + 1 < R and band[hi + 1] and hi - r < reach:
                hi += 1
            mask[lo:hi + 1, c0:c0 + width] = True
    return mask


def _connect(site: Site, corridor: np.ndarray, width: int, pall: np.ndarray) -> np.ndarray:
    """Join all cores and corridor pieces into one network with least-cost paths."""
    cores = [c.access for c in site.cores]
    if not cores:
        return corridor
    lab, n = ndimage.label(corridor)
    pieces = [lab == i for i in range(1, n + 1)]
    targets = cores[1:] + pieces
    network = cores[0].copy()
    near = ndimage.binary_dilation(network)
    for k in range(len(pieces) - 1, -1, -1):          # pieces already touching core 0 join for free
        if (pieces[k] & near).any():
            network |= pieces[k]
    remaining = [t for t in targets if not (t & network).any()]
    kernel = np.ones((width, width), bool)
    guard = 0
    while remaining and guard < 40:
        guard += 1
        cost = np.where(pall, 1.0, np.inf)
        cost[corridor] = 0.25
        for cm in cores:
            cost[cm] = 1.0
        starts = list(zip(*np.where(network)))
        ends_mask = np.zeros_like(network)
        for t in remaining:
            ends_mask |= t
        ends = list(zip(*np.where(ends_mask)))
        mcp = MCP_Geometric(cost)
        costs, _ = mcp.find_costs(starts, ends, find_all_ends=False)
        reached = [(costs[e], e) for e in ends if np.isfinite(costs[e])]
        if not reached:
            break
        _, end = min(reached)
        path = np.zeros_like(network)
        for rr, cc in mcp.traceback(end):
            path[rr, cc] = True
        carved = ndimage.binary_dilation(path, structure=kernel) & pall
        corridor = corridor | carved
        network |= carved | path
        hit = [t for t in remaining if (t & ndimage.binary_dilation(network)).any()]
        for t in hit:
            network |= t
        remaining = [t for t in remaining if not any(t is h for h in hit)]
    # keep only corridor connected to the network
    lab, n = ndimage.label(corridor)
    keep = np.zeros_like(corridor)
    grown = ndimage.binary_dilation(network & ~corridor) | (network & corridor)
    for i in range(1, n + 1):
        piece = lab == i
        if (piece & grown).any():
            keep |= piece
    return keep


def _distances(site: Site, corridor: np.ndarray, cell_ft: float, width: int):
    """Corridor geodesic distance from each core; reached cores; dead-end length (ft)."""
    walk = corridor.copy()
    for c in site.cores:
        walk |= c.access
    cost = np.where(walk, 1.0, np.inf)
    per_core, reached = [], []
    for c in site.cores:
        if not (ndimage.binary_dilation(c.access) & corridor).any():
            continue
        d, _ = MCP_Geometric(cost).find_costs(list(zip(*np.where(c.access))))
        per_core.append(d)
        reached.append(c.index)
    if not per_core:
        return np.full(corridor.shape, FAR), 0, FAR
    dmin = np.minimum.reduce(per_core) if len(per_core) > 1 else per_core[0]
    dmin = np.where(corridor, dmin, FAR) * cell_ft
    if len(per_core) == 1:
        dead = float(np.max(np.where(corridor, per_core[0], 0))) * cell_ft
        return dmin, 1, dead
    route = np.zeros(corridor.shape, bool)
    for i in range(len(per_core)):
        for j in range(i + 1, len(per_core)):
            s = per_core[i] + per_core[j]
            best = np.min(np.where(corridor, s, np.inf))
            if np.isfinite(best):
                route |= corridor & (s <= best + 2.0 * width)
    if not route.any():
        return dmin, len(per_core), float(np.max(np.where(corridor, dmin, 0)))
    # a corridor cell at a reached core is at an exit door: a pocket is measured back to the nearest
    # exit door or through-route, whichever comes first (IBC dead end = one direction of travel only)
    for c in site.cores:
        if c.index in reached:
            route |= corridor & ndimage.binary_dilation(c.access)
    d_route, _ = MCP_Geometric(np.where(corridor, 1.0, np.inf)).find_costs(list(zip(*np.where(route))))
    dead = float(np.max(np.where(corridor & np.isfinite(d_route), d_route, 0))) * cell_ft
    return dmin, len(per_core), dead


def build_corridor(site: Site, seg_genes: np.ndarray, width: int):
    pall = _placeable_all(site)
    coords = np.argwhere(pall)
    raw = _carve_segments(site, seg_genes.reshape(N_SEG, 4), width, pall, coords)
    if not raw.any():  # always have something to grow from: a spine through the middle
        raw = _carve_segments(site, np.array([[0.5, 0.5, 0.0, 1.0]] + [[0, 0, 0, 0]] * (N_SEG - 1)), width, pall, coords)
    return _connect(site, raw, width, pall)


# --------------------------------------------------------------------- packing
def _integral(a: np.ndarray, m: int) -> np.ndarray:
    p = np.pad(a.astype(np.float64), m)
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    return ii


def _blocks(ii: np.ndarray, r_off: int, c_off: int, nr: int, nc: int, h: int, w: int, m: int) -> np.ndarray:
    r, c = m + r_off, m + c_off
    return (ii[r + h:r + h + nr, c + w:c + w + nc] - ii[r:r + nr, c + w:c + w + nc]
            - ii[r + h:r + h + nr, c:c + nc] + ii[r:r + nr, c:c + nc])


def _geometry(dirn: str, f: int, d: int):
    """(H, W, contact, back, side1, side2, door) with strips as (r_off, c_off, h, w)."""
    if dirn in ("N", "S"):
        H, W = d, f
        contact = (-1, 0, 1, f) if dirn == "N" else (d, 0, 1, f)
        back = (d - 1, 0, 1, f) if dirn == "N" else (0, 0, 1, f)
        sides = ((0, -1, d, 1), (0, f, d, 1))
        door = (-1, f // 2) if dirn == "N" else (d, f // 2)
    else:
        H, W = f, d
        contact = (0, -1, f, 1) if dirn == "E" else (0, d, f, 1)
        back = (0, d - 1, f, 1) if dirn == "E" else (0, 0, f, 1)
        sides = ((-1, 0, 1, d), (f, 0, 1, d))
        door = (f // 2, -1) if dirn == "E" else (f // 2, d)
    return H, W, contact, back, sides, door


def _backout(dirn: str, f: int, d: int):
    """Strip just behind the room's back wall (outside the room)."""
    return {"N": (d, 0, 1, f), "S": (-1, 0, 1, f), "E": (0, d, f, 1), "W": (0, -1, f, 1)}[dirn]


class Packer:
    def __init__(self, site: Site, program: Program, corridor: np.ndarray, core_dist: np.ndarray):
        self.site, self.program, self.corridor = site, program, corridor
        self.R, self.C = site.shape
        self.m = 64
        cf = site.cell_ft
        self.slack = int(math.ceil(program.exterior_slack_ft / cf))
        self.door = int(math.ceil(program.door_width_ft / cf))
        self.ii_corr = _integral(corridor, self.m)
        self.dpad = np.pad(core_dist, self.m, constant_values=FAR)
        self.ii_ext = [_integral((site.ext_dist[l] <= self.slack + 1) & ~site.outside[l], self.m)
                       for l in range(site.n_levels)]
        self.free = [site.placeable[l] & ~corridor for l in range(site.n_levels)]
        self._refresh_all()
        self.static: dict = {}
        self.pack = dict(PACK)

    def _refresh(self, l: int):
        self.ii_free[l] = _integral(self.free[l], self.m)
        self.ii_solid[l] = _integral(~self.free[l] & ~self.corridor, self.m)

    def _refresh_all(self):
        self.ii_free = [None] * self.site.n_levels
        self.ii_solid = [None] * self.site.n_levels
        for l in range(self.site.n_levels):
            self._refresh(l)

    def _static(self, l: int, dirn: str, f: int, d: int):
        key = (l, dirn, f, d)
        if key in self.static:
            return self.static[key]
        H, W, contact, back, sides, door = _geometry(dirn, f, d)
        nr, nc = self.R - H + 1, self.C - W + 1
        if nr <= 0 or nc <= 0:
            self.static[key] = None
            return None
        m = self.m
        need = max(self.door, int(math.ceil(0.5 * f)))
        con = _blocks(self.ii_corr, *contact[:2], nr, nc, *contact[2:], m) >= need
        ext = _blocks(self.ii_ext[l], *back[:2], nr, nc, *back[2:], m) >= 0.5 * f
        side_ext = [_blocks(self.ii_ext[l], *s[:2], nr, nc, *s[2:], m) for s in sides]
        corner = ext & ((side_ext[0] >= 0.5 * d) | (side_ext[1] >= 0.5 * d))
        dr, dc = door
        dd = self.dpad[m + dr:m + dr + nr, m + dc:m + dc + nc]
        val = (H, W, nr, nc, con, ext, corner, np.minimum(dd, FAR) * 1.0, sides)
        self.static[key] = val
        return val

    def best_slot(self, inst: Instance, level_counts: dict):
        rt = inst.type
        cf = self.site.cell_ft
        a, b = int(round(rt.width_ft / cf)), int(round(rt.depth_ft / cf))
        if rt.category in ("guest", "ada", "suite"):
            shapes = [(min(a, b), max(a, b))] if a != b else [(a, b)]
        else:
            shapes = [(a, b), (b, a)] if a != b else [(a, b)]
        best = None
        for l in inst.levels:
            counts = level_counts.get(rt.name, {})
            mean = (sum(counts.values()) / max(1, len(inst.levels))) if counts else 0.0
            lv_term = 0.0
            if rt.min_per_floor and counts.get(l, 0) < rt.min_per_floor:
                lv_term += 1.5
            if counts.get(l, 0) < mean:
                lv_term += 0.25
            for f, d in shapes:
                for dirn in DIRS:
                    st = self._static(l, dirn, f, d)
                    if st is None:
                        continue
                    H, W, nr, nc, con, ext, corner, dd, sides = st
                    backout = _backout(dirn, f, d)
                    ok = con & (_blocks(self.ii_free[l], 0, 0, nr, nc, H, W, self.m) >= H * W - 0.5)
                    if not ok.any():
                        continue
                    P = self.pack
                    score = -P["core_pull"] * np.minimum(dd, 300) / 300 + lv_term
                    if rt.exterior:
                        score = score + np.where(ext, P["exterior"], -P["no_window"])
                    if rt.corner:
                        score = score + P["corner"] * corner
                    if rt.near_core:
                        score = score + P["core"] * (1.0 - np.minimum(dd, 150) / 150)
                    s1 = _blocks(self.ii_solid[l], *sides[0][:2], nr, nc, *sides[0][2:], self.m)
                    s2 = _blocks(self.ii_solid[l], *sides[1][:2], nr, nc, *sides[1][2:], self.m)
                    side_len = sides[0][2] * sides[0][3]
                    score = score + P["tight"] * ((s1 >= 0.8 * side_len).astype(float) + (s2 >= 0.8 * side_len))
                    if P["back"]:
                        bk = _blocks(self.ii_solid[l], *backout[:2], nr, nc, *backout[2:], self.m)
                        score = score + P["back"] * (bk >= 0.8 * f)
                    score = np.where(ok, score, -np.inf)
                    idx = int(np.argmax(score))
                    sc = score.flat[idx]
                    if best is None or sc > best[0]:
                        r, c = divmod(idx, nc)
                        best = (sc, l, r, c, H, W, dirn, f, d, bool(ext.flat[idx]), bool(corner.flat[idx]),
                                float(dd.flat[idx]))
        return best

    def place(self, l: int, r: int, c: int, H: int, W: int):
        self.free[l][r:r + H, c:c + W] = False
        self._refresh(l)


def decode(site: Site, program: Program, genome: np.ndarray, instances: list[Instance] | None = None) -> Design:
    instances = instances if instances is not None else program.instances(site.level_names)
    width = max(1, int(round(program.corridor_width_ft / site.cell_ft)))
    seg = genome[:N_SEG * 4]
    keys = genome[N_SEG * 4:N_SEG * 4 + len(instances)]
    corridor = build_corridor(site, seg, width)
    core_dist, reached, dead = _distances(site, corridor, site.cell_ft, width)
    packer = Packer(site, program, corridor, core_dist)
    order = np.argsort(-(np.array([i.type.priority for i in instances]) / 10.0 * PRIORITY_PULL + keys))
    placements, unplaced, counts = [], [], {}
    cf = site.cell_ft
    for k in order:
        inst = instances[k]
        slot = packer.best_slot(inst, counts)
        if slot is None:
            unplaced.append(inst.name)
            continue
        _, l, r, c, H, W, dirn, f, d, ext, corner, dd = slot
        packer.place(l, r, c, H, W)
        counts.setdefault(inst.type.name, {}).setdefault(l, 0)
        counts[inst.type.name][l] += 1
        placements.append(Placement(inst.name, inst.type.name, inst.type.category, l, r, c, H, W, dirn, f * cf, d * cf,
                                    ext, corner, dd, inst.type.priority))
    return Design(site, program, corridor, placements, unplaced, core_dist, reached, dead, genome=genome)


def random_genome(rng: np.random.Generator, n_instances: int) -> np.ndarray:
    g = rng.random(genome_length(n_instances))
    seg = g[:N_SEG * 4].reshape(N_SEG, 4)
    seg[:, 3] = np.where(rng.random(N_SEG) < 0.45, rng.random(N_SEG) * 0.1, 0.35 + 0.65 * rng.random(N_SEG))
    return g


def spine_genomes(n_instances: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Seeds: single and double-loaded spines in both directions (classic hotel corridors)."""
    seeds = []
    for spec in ([[0.5, 0.5, 0.0, 1.0]], [[0.5, 0.5, 1.0, 1.0]], [[0.5, 0.33, 0.0, 1.0], [0.5, 0.66, 0.0, 1.0]],
                 [[0.33, 0.5, 1.0, 1.0], [0.66, 0.5, 1.0, 1.0]], [[0.5, 0.5, 0.0, 1.0], [0.5, 0.5, 1.0, 1.0]]):
        g = random_genome(rng, n_instances)
        seg = np.zeros((N_SEG, 4))
        seg[:len(spec)] = spec
        g[:N_SEG * 4] = seg.ravel()
        seeds.append(g)
    return seeds


def facade_seeds(site: Site, program: Program, n_instances: int, rng: np.random.Generator, n: int = 6) -> list[np.ndarray]:
    """Seeds from domain knowledge: corridor centre-lines one room depth in from the facade
    (ext_dist ~ D + w/2), where D is the typical guest-room depth, so window rooms on the outer
    side exactly fill to the exterior wall. Picks the best-supported rows and columns."""
    keys = [rt for rt in program.rooms if rt.is_key] or program.rooms
    D = float(np.median([max(rt.width_ft, rt.depth_ft) for rt in keys])) / site.cell_ft
    w = program.corridor_width_ft / site.cell_ft
    # centre column/row c of a corridor of width w occupies c - w//2 .. c + w//2, so a room of depth D fits between
    # the facade and the corridor when ext_dist(c) = D + w//2 + 1 (ext_dist counts the edge cell as 1)
    target = D + int(w) // 2 + 1
    pall = _placeable_all(site)
    R, C = site.shape
    lines = []
    for tol in (0.6, 1.6, 3.0):           # exact offset first; widen only if the site has no exact lines
        near = pall & (np.abs(site.ext_dist[0] - target) <= tol)
        lines = []
        for axis, counts in ((0, near.sum(1)), (1, near.sum(0))):      # axis 0: rows -> horizontal corridors
            order = np.argsort(-counts)
            taken = []
            for idx in order[:40]:
                if counts[idx] < 15 or any(abs(idx - t) < 2 * D + w for t in taken):
                    continue
                taken.append(int(idx))
                cells = np.where(near[idx, :] if axis == 0 else near[:, idx])[0]
                lines.append((int(counts[idx]), axis, int(idx), float(np.median(cells))))
                if len(taken) >= 3:
                    break
        if lines:
            break
    lines.sort(reverse=True)
    seeds = []
    combos = [lines[:1], lines[:2], lines[:3], [l for l in lines if l[1] == 0][:2], [l for l in lines if l[1] == 1][:2],
              lines[1:3]]
    for combo in combos[:n]:
        if not combo:
            continue
        seg = np.zeros((N_SEG, 4))
        for k, (_, axis, idx, mid) in enumerate(combo):
            if axis == 0:
                seg[k] = [mid / max(C - 1, 1), idx / max(R - 1, 1), 0.0, 1.0]
            else:
                seg[k] = [idx / max(C - 1, 1), mid / max(R - 1, 1), 1.0, 1.0]
        g = random_genome(rng, n_instances)
        g[:N_SEG * 4] = seg.ravel()
        seeds.append(g)
    return seeds
