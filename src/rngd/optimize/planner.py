"""Architectural decoder: genome -> corridor spines -> modular bays -> rooms.

How hotel floor plates are actually laid out, and what this decoder enforces:
  1. Corridor SPINES are straight and sit one room-depth in from the facade, so the rooms on
     their sides reach a window (double-loaded where the wing is ~2 rooms + corridor wide).
     Candidate spines come from the footprint: every row/column is scored by how many window
     rooms it could serve on each side; the best non-overlapping lines become candidates.
  2. Spines are trimmed to their last room (short dead ends) and joined to every core with
     ORTHOGONAL L-shaped connectors (never diagonal).
  3. Each corridor side is cut into continuous BAYS of one module width (the main guest-room
     frontage), so rooms form unbroken rows; the last bay of a run absorbs the remainder, which
     gives the wider end/corner bays suites want. This is the BaaP module grid.
  4. Rooms take whole bays (King 1, suite 2, lobby 3 ...); small service rooms share half a bay.
     Every floor has the same bays, and a stacking bonus keeps the same room type above/below,
     so walls and plumbing stack (typical floors).
  5. Room-to-bay assignment is greedy in genome order and rewards windows, corners for suites,
     the core for ADA/service, required adjacencies, floor balance and stacking.

Genome: N_SEG blocks of 4 genes -- one block per candidate spine (trim_lo, trim_hi, shift,
active), the last block holds connector-elbow genes -- then one order key per room instance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.graph import MCP_Geometric

from .engine import FAR, Design, Placement, _distances, _placeable_all
from .program import Instance, Program
from .site import Site

N_SP = 10
N_SEG = N_SP + 1
PRIORITY_PULL = 0.6
# assignment weights (tunable from the app)
PACK = {"exterior": 1.0, "no_window": 1.5, "corner": 0.8, "core": 1.0, "adjacency": 0.6, "stack": 0.4,
        "group": 0.15, "core_pull": 0.1}
# hard design rules (tunable from the app)
RULES = {"guest_window_required": True}   # guest rooms (King, QQ, suites, ADA) only in bays with a window


@dataclass
class Spine:
    axis: str      # "H" (corridor along a row) or "V" (along a column)
    coord: int     # centre row (H) or column (V)
    lo: int
    hi: int        # inclusive extent along the corridor
    score: float


@dataclass
class Bay:
    level: int
    idx: int          # same idx on every floor (stacking)
    interval: int
    pos: int          # order within its interval
    r0: int
    c0: int
    h: int
    w: int
    facing: str
    window: bool
    corner: bool
    door: tuple[int, int]
    frontage: int
    depth: int
    used: float = 0.0  # 0 free, 0.5 half used, 1 used
    type_name: str = ""


# ------------------------------------------------------------------ geometry helpers
def _runs(free: np.ndarray, outside: np.ndarray):
    """Run lengths of free cells starting at each cell, in 4 directions, and whether the run ends
    at the facade (outside cell or grid edge)."""
    R, C = free.shape
    U = np.zeros((R, C), np.int32); SU = np.zeros((R, C), bool)
    D = np.zeros((R, C), np.int32); SD = np.zeros((R, C), bool)
    for r in range(R - 1, -1, -1):
        nxt_u, nxt_s = (U[r + 1], SU[r + 1]) if r + 1 < R else (np.zeros(C, np.int32), np.ones(C, bool))
        U[r] = np.where(free[r], nxt_u + 1, 0)
        SU[r] = np.where(free[r], nxt_s, outside[r])
    for r in range(R):
        prv, ps = (D[r - 1], SD[r - 1]) if r > 0 else (np.zeros(C, np.int32), np.ones(C, bool))
        D[r] = np.where(free[r], prv + 1, 0)
        SD[r] = np.where(free[r], ps, outside[r])
    Rt = np.zeros((R, C), np.int32); SR = np.zeros((R, C), bool)
    Lf = np.zeros((R, C), np.int32); SL = np.zeros((R, C), bool)
    for c in range(C - 1, -1, -1):
        n, s = (Rt[:, c + 1], SR[:, c + 1]) if c + 1 < C else (np.zeros(R, np.int32), np.ones(R, bool))
        Rt[:, c] = np.where(free[:, c], n + 1, 0)
        SR[:, c] = np.where(free[:, c], s, outside[:, c])
    for c in range(C):
        p, s = (Lf[:, c - 1], SL[:, c - 1]) if c > 0 else (np.zeros(R, np.int32), np.ones(R, bool))
        Lf[:, c] = np.where(free[:, c], p + 1, 0)
        SL[:, c] = np.where(free[:, c], s, outside[:, c])
    # stop flags describe the cell AFTER the run: shift so SU[r] tells what follows the run from r
    return {"U": (U, SU), "D": (D, SD), "R": (Rt, SR), "L": (Lf, SL)}


def _true_runs(mask: np.ndarray):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


class Params:
    def __init__(self, site: Site, program: Program):
        cf = site.cell_ft
        self.site, self.program = site, program
        self.w = max(1, int(round(program.corridor_width_ft / cf)))
        keys = [rt for rt in program.rooms if rt.is_key] or program.rooms
        fronts, depths, weights = [], [], []
        for rt in keys:
            fronts.append(min(rt.width_ft, rt.depth_ft) if rt.category != "suite" else min(rt.width_ft, rt.depth_ft) / 2)
            depths.append(max(rt.width_ft, rt.depth_ft))
            weights.append(rt.count)
        order = np.argsort(fronts)
        cum = np.cumsum(np.array(weights)[order])
        self.M = max(6, int(round(np.array(fronts)[order][np.searchsorted(cum, cum[-1] / 2)] / cf)))   # weighted median
        self.Dnom = int(round(np.median(depths) / cf))
        self.Dmin = max(8, int(math.floor(min(depths) * 0.92 / cf)))
        self.Dmax = int(round(max(depths) / cf)) + 2   # rooms may run a little deeper to reach the facade
        self.slack = int(math.ceil(program.exterior_slack_ft / cf))
        self.pall = _placeable_all(site)
        self.outside = site.outside[0]
        self.candidates = self._candidates()

    def _candidates(self) -> list[Spine]:
        free, w = self.pall, self.w
        runs = _runs(free, self.outside)
        R, C = free.shape
        cands: list[Spine] = []
        self.line_scores: dict = {}   # score of every row ("H") / column ("V"), kept for reports
        for axis in ("H", "V"):
            n_lines = R if axis == "H" else C
            scores, cols_score = np.zeros(n_lines), {}
            for k in range(w // 2, n_lines - (w - w // 2)):
                b0, b1 = k - w // 2, k - w // 2 + w
                if axis == "H":
                    band = free[b0:b1, :].all(0)
                    a, sa = (runs["U"][0][b1], runs["U"][1][b1]) if b1 < R else (np.zeros(C), np.ones(C, bool))
                    b, sb = (runs["D"][0][b0 - 1], runs["D"][1][b0 - 1]) if b0 > 0 else (np.zeros(C), np.ones(C, bool))
                else:
                    band = free[:, b0:b1].all(1)
                    a, sa = (runs["R"][0][:, b1], runs["R"][1][:, b1]) if b1 < C else (np.zeros(R), np.ones(R, bool))
                    b, sb = (runs["L"][0][:, b0 - 1], runs["L"][1][:, b0 - 1]) if b0 > 0 else (np.zeros(R), np.ones(R, bool))
                ga, gb = a >= self.Dmin, b >= self.Dmin
                # a window room on that side; strongly preferred when the facade is ~one nominal room deep
                near_a = ga & sa & (a <= self.Dnom + 2 + self.slack)
                near_b = gb & sb & (b <= self.Dnom + 2 + self.slack)
                far_a = ga & sa & (a <= self.Dmax + self.slack)
                far_b = gb & sb & (b <= self.Dmax + self.slack)
                per = band * (ga.astype(float) + gb + 1.5 * near_a + 1.5 * near_b + 0.5 * far_a + 0.5 * far_b)
                scores[k] = per.sum()
                cols_score[k] = (band & (ga | gb), per)
            self.line_scores[axis] = scores
            chosen = []
            for k in np.argsort(-scores):
                if scores[k] <= 0 or len(chosen) >= 5:
                    break
                if any(abs(k - j) < self.Dmin + w for j in chosen):
                    continue
                chosen.append(int(k))
                ok, per = cols_score[k]
                for lo, hi in _true_runs(ok):
                    if hi - lo + 1 >= 2 * self.M:
                        cands.append(Spine(axis, int(k), lo, hi, float(per[lo:hi + 1].sum())))
        cands.sort(key=lambda s: -s.score)
        return cands[:N_SP]


_PARAMS: dict = {}


def params_for(site: Site, program: Program) -> Params:
    key = (str(site.path), program.corridor_width_ft, program.exterior_slack_ft,
           tuple((r.name, r.width_ft, r.depth_ft, r.count, r.category) for r in program.rooms))
    if key not in _PARAMS:
        _PARAMS[key] = Params(site, program)
    return _PARAMS[key]


# ------------------------------------------------------------------ corridor
def _band(mask_shape, axis: str, coord: int, lo: int, hi: int, w: int) -> np.ndarray:
    m = np.zeros(mask_shape, bool)
    b0 = coord - w // 2
    if axis == "H":
        m[max(b0, 0):b0 + w, max(lo, 0):hi + 1] = True
    else:
        m[max(lo, 0):hi + 1, max(b0, 0):b0 + w] = True
    return m


def _active_spines(P: Params, genome: np.ndarray) -> list[Spine]:
    blocks = genome[:N_SEG * 4].reshape(N_SEG, 4)
    out = []
    for i, sp in enumerate(P.candidates):
        t_lo, t_hi, shift, active = blocks[i]
        if active < 0.12:
            continue
        coord = sp.coord + int(round((shift - 0.5) * 4))
        n = sp.hi - sp.lo + 1
        lo = sp.lo + int(t_lo * 0.35 * n)
        hi = sp.hi - int(t_hi * 0.35 * n)
        band = _band(P.pall.shape, sp.axis, coord, sp.lo, sp.hi, P.w) & ~P.pall
        if band.any():  # shifted band hits an obstacle -> keep the scored position
            coord = sp.coord
        if hi - lo + 1 >= 2 * P.M:
            out.append(Spine(sp.axis, coord, lo, hi, sp.score))
    if not out and P.candidates:
        out.append(P.candidates[0])
    return out


def _boundary(mask: np.ndarray, step: int = 2) -> np.ndarray:
    edge = mask & ~ndimage.binary_erosion(mask)
    pts = np.argwhere(edge)
    return pts[::step] if len(pts) > 400 else pts


def _l_path(P: Params, allowed: np.ndarray, p, q, elbow_first_h: bool):
    """Orthogonal path p -> q (row/col), carved at corridor width; None if blocked."""
    (r1, c1), (r2, c2) = p, q
    w = P.w
    legs = [("H", r1, min(c1, c2), max(c1, c2)), ("V", c2, min(r1, r2), max(r1, r2))] if elbow_first_h else \
        [("V", c1, min(r1, r2), max(r1, r2)), ("H", r2, min(c1, c2), max(c1, c2))]
    carved = np.zeros(P.pall.shape, bool)
    for axis, coord, lo, hi in legs:
        band = _band(P.pall.shape, axis, coord, lo - w // 2, hi + w // 2, w)
        if not allowed[band].all():
            return None
        carved |= band
    return carved & P.pall


def _connect(P: Params, corridor: np.ndarray, elbow_genes: np.ndarray) -> np.ndarray:
    site = P.site
    cores = [c.access for c in site.cores]
    if not cores:
        return corridor
    allowed = P.pall.copy()
    for cm in cores:
        allowed |= cm
    lab, n = ndimage.label(corridor)
    targets = [lab == i for i in range(1, n + 1)] + cores
    network = targets.pop(0) if n else cores[0]
    if not n:
        targets = cores[1:]
    k = 0
    for _ in range(len(targets) + 2):
        grown = ndimage.binary_dilation(network)
        targets = [t for t in targets if not (t & grown).any() or not _merge(network, t)]
        if not targets:
            break
        net_pts = _boundary(network)
        best = None
        for t in targets:
            tp = _boundary(t)
            d = cdist(net_pts, tp, "cityblock")
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if best is None or d[i, j] < best[0]:
                best = (d[i, j], tuple(net_pts[i]), tuple(tp[j]), t)
        _, p, q, t = best
        first_h = elbow_genes[k % len(elbow_genes)] < 0.5
        k += 1
        path = _l_path(P, allowed, p, q, first_h)
        if path is None:
            path = _l_path(P, allowed, p, q, not first_h)
        if path is None:   # fallback: 4-connected least-cost path (rectilinear)
            cost = np.where(allowed, 1.0, np.inf)
            cost[corridor] = 0.3
            mcp = MCP_Geometric(cost, fully_connected=False)
            costs, _ = mcp.find_costs([p], [q])
            if not np.isfinite(costs[q]):
                targets.remove(t)
                continue
            line = np.zeros(P.pall.shape, bool)
            for rr, cc in mcp.traceback(q):
                line[rr, cc] = True
            path = ndimage.binary_dilation(line, structure=np.ones((P.w, P.w), bool)) & P.pall
        corridor = corridor | path
        network = network | path | t
    # drop pieces that ended up unconnected
    lab, n = ndimage.label(corridor)
    keep = np.zeros_like(corridor)
    reach = ndimage.binary_dilation(network)
    for i in range(1, n + 1):
        piece = lab == i
        if (piece & reach).any():
            keep |= piece
    return keep


def _merge(network: np.ndarray, t: np.ndarray) -> bool:
    network |= t
    return True


# ------------------------------------------------------------------ bays
def _segments(spines: list[Spine], corridor: np.ndarray, w: int) -> list[tuple]:
    """Straight corridor segments to line with bays: the spines plus straight runs of connectors."""
    segs = [(s.axis, s.coord, s.lo, s.hi) for s in spines]
    covered = np.zeros_like(corridor)
    for s in spines:
        covered |= _band(corridor.shape, s.axis, s.coord, s.lo, s.hi, w)
    rest = corridor & ~covered
    lab, n = ndimage.label(rest)
    for i in range(1, n + 1):
        rr, cc = np.where(lab == i)
        hgt, wid = rr.max() - rr.min() + 1, cc.max() - cc.min() + 1
        if wid >= hgt and wid >= 2 * w:
            segs.append(("H", int(round(rr.mean())), int(cc.min()), int(cc.max())))
        elif hgt >= 2 * w:
            segs.append(("V", int(round(cc.mean())), int(rr.min()), int(rr.max())))
    return segs


def _bays(P: Params, corridor: np.ndarray, segs: list[tuple]) -> list[Bay]:
    free = P.pall & ~corridor
    runs = _runs(free, P.outside)
    R, C = free.shape
    occ = ~free
    bays: list[Bay] = []
    interval = 0
    M, w = P.M, P.w
    # two passes over all corridor sides: bays with a window first, then the windowless ones in what
    # is left, so an interior bay of one spine never takes the facade strip of another
    for want_window in (True, False):
        for axis, coord, lo, hi in segs:
            b0, b1 = coord - w // 2, coord - w // 2 + w
            sides = (("N", "U", b1), ("S", "D", b0 - 1)) if axis == "H" else (("E", "R", b1), ("W", "L", b0 - 1))
            for facing, key, edge in sides:
                if not (0 <= edge < (R if axis == "H" else C)):
                    continue
                dep, stop = runs[key]
                if axis == "H":
                    d_line, s_line = dep[edge, :], stop[edge, :]
                else:
                    d_line, s_line = dep[:, edge], stop[:, edge]
                lo_c, hi_c = max(lo, 0), min(hi, len(d_line) - 1)
                valid = np.zeros(len(d_line), bool)
                valid[lo_c:hi_c + 1] = d_line[lo_c:hi_c + 1] >= P.Dmin
                for a, b in _true_runs(valid):
                    interval += 1
                    x, pos, last_end = a, 0, a
                    while x + 4 - 1 <= b:
                        rest = b - x + 1
                        if rest >= M:
                            # the last full bay absorbs a small remainder (<40% of a module); a larger
                            # remainder becomes its own narrow bay (service / flex) instead of an oversized room
                            tail = rest - M
                            width = M + tail if 0 < tail < 0.4 * M and tail < M else M
                        else:
                            width = rest           # narrow end bay
                            if width < 4:
                                break
                        seg_d = d_line[x:x + width]
                        d = int(min(seg_d.min(), P.Dmax))
                        if d < P.Dmin:
                            x += 1
                            continue
                        if axis == "H":
                            r0 = edge if facing == "N" else edge - d + 1
                            c0, h, wd = x, d, width
                        else:
                            c0 = edge if facing == "E" else edge - d + 1
                            r0, h, wd = x, width, d
                        if occ[r0:r0 + h, c0:c0 + wd].any():
                            x += 1
                            continue
                        window = bool(s_line[x:x + width].all() and (seg_d <= d + P.slack).all())
                        if window != want_window:      # the other pass handles this bay
                            x += width
                            continue
                        if x != last_end:              # gap: rooms may not span it
                            interval += 1
                            pos = 0
                        mid = x + width // 2
                        if axis == "H":
                            door = (edge - 1 if facing == "N" else edge + 1, mid)
                            left = (r0 + h // 2, c0 - 1)
                            right = (r0 + h // 2, c0 + wd)
                        else:
                            door = (mid, edge - 1 if facing == "E" else edge + 1)
                            left = (r0 - 1, c0 + wd // 2)
                            right = (r0 + h, c0 + wd // 2)
                        side_out = any(not (0 <= rr < R and 0 <= cc < C) or P.outside[rr, cc] for rr, cc in (left, right))
                        occ[r0:r0 + h, c0:c0 + wd] = True
                        bays.append(Bay(0, len(bays), interval, pos, r0, c0, h, wd, facing, window,
                                        bool(window and side_out), door, width, d))
                        pos += 1
                        x += width
                        last_end = x
    return bays


# ------------------------------------------------------------------ assignment
def _assign(P: Params, bays_proto: list[Bay], instances: list[Instance], keys: np.ndarray, core_dist: np.ndarray):
    site = P.site
    cf = site.cell_ft
    nl = site.n_levels
    floors = [[Bay(**{**b.__dict__, "level": l}) for b in bays_proto] for l in range(nl)]
    by_int: dict = {}
    for b in bays_proto:
        by_int.setdefault(b.interval, []).append(b.idx)
    bay_area = P.M * P.Dnom
    dd = np.array([core_dist[min(max(b.door[0], 0), core_dist.shape[0] - 1),
                             min(max(b.door[1], 0), core_dist.shape[1] - 1)] for b in bays_proto]) if bays_proto else np.zeros(0)
    order = np.argsort(-(np.array([i.type.priority for i in instances]) / 10.0 * PRIORITY_PULL + keys))
    placements, unplaced, counts = [], [], {}
    adj = {}
    for a in P.program.adjacency:
        adj.setdefault(a.a, []).append((a.b, a.weight, a.max_ft))
        adj.setdefault(a.b, []).append((a.a, a.weight, a.max_ft))
    placed_by_type: dict = {}
    W = PACK
    static: dict = {}
    for oi in order:
        inst = instances[oi]
        rt = inst.type
        area = rt.area_sf / (cf * cf)
        small = area <= 0.45 * bay_area
        if P.program.width_along_corridor:   # hotel programs: width = frontage (wide suites)
            fn, dn = rt.width_ft / cf, min(rt.depth_ft / cf, float(P.Dnom))
        else:                                # QA programs: the shorter side faces the corridor
            fn, dn = min(rt.width_ft, rt.depth_ft) / cf, max(rt.width_ft, rt.depth_ft) / cf
        k = 1 if small else max(1, int(area / bay_area + 0.5))
        if rt.is_key and not small:
            k = max(k, math.ceil(0.85 * fn / P.M - 1e-9))   # enough bays for its frontage
        best = None
        tcounts = counts.get(rt.name, {})
        mean = sum(tcounts.values()) / max(1, len(inst.levels))
        for l in inst.levels:
            fl = floors[l]
            lv = 0.0
            if rt.min_per_floor and tcounts.get(l, 0) < rt.min_per_floor:
                lv += 1.5
            if tcounts.get(l, 0) < mean:
                lv += 0.25
            cands = static.get(rt.name)
            if cands is None:
                cands = static[rt.name] = _static_candidates(P, bays_proto, by_int, rt, area, small, k, fn, dn, dd, W)
            for iid, s, idx_run, sc0, win, d0 in cands:
                run = [fl[i] for i in idx_run]
                if small:
                    if run[0].used >= 1.0 or (run[0].used == 0.5 and run[0].type_name and
                                             run[0].type_name not in _SMALL_OK):
                        continue
                elif any(b.used > 0 for b in run):
                    continue
                idxs = by_int[iid]
                sc = lv + sc0
                for other, wgt, mx in adj.get(rt.name, []):
                    for (ol, orr, occ_) in placed_by_type.get(other, []):
                        if ol == l:
                            dist = (abs(orr - run[0].r0) + abs(occ_ - run[0].c0)) * cf
                            sc += W["adjacency"] * wgt * max(0.0, 1.0 - max(0.0, dist - mx) / mx)
                stack = sum(1 for o in range(nl) if o != l and floors[o][run[0].idx].type_name == rt.name)
                sc += W["stack"] * min(stack, 2) / 2
                nb = [fl[i] for i in idxs[max(0, s - 1):s]] + [fl[i] for i in idxs[s + k:s + k + 1]]
                if any(b.type_name == rt.name for b in nb):
                    sc += W["group"]
                if best is None or sc > best[0]:
                    best = (sc, l, run, win, d0)
        if best is None:
            unplaced.append(inst.name)
            continue
        _, l, run, win, d0 = best
        if small:
            b = run[0]
            half = b.used == 0.5
            b.used += 0.5
            b.type_name = b.type_name or rt.name
            fr = b.frontage // 2
            if b.facing in ("N", "S"):
                c0 = b.c0 + (fr if half else 0)
                rect = (b.r0, c0, b.h, (b.w - fr) if half else fr)
            else:
                r0 = b.r0 + (fr if half else 0)
                rect = (r0, b.c0, (b.h - fr) if half else fr, b.w)
            front = (b.frontage - fr) if half else fr
        else:
            for b in run:
                b.used, b.type_name = 1.0, rt.name
            r0 = min(b.r0 for b in run); c0 = min(b.c0 for b in run)
            r1 = max(b.r0 + b.h for b in run); c1 = max(b.c0 + b.w for b in run)
            rect = (r0, c0, r1 - r0, c1 - c0)
            front = sum(b.frontage for b in run)
        depth = min(b.depth for b in run)
        counts.setdefault(rt.name, {}).setdefault(l, 0)
        counts[rt.name][l] += 1
        placed_by_type.setdefault(rt.name, []).append((l, rect[0], rect[1]))
        placements.append(Placement(inst.name, rt.name, rt.category, l, rect[0], rect[1], rect[2], rect[3], run[0].facing,
                                    front * cf, depth * cf, bool(win), any(b.corner for b in run), d0, rt.priority))
    flex = [(b.level, b.r0, b.c0, b.h, b.w, b.window) for fl in floors for b in fl if b.used == 0]
    return placements, unplaced, flex, floors


def _static_candidates(P: Params, bays_proto, by_int, rt, area, small, k, fn, dn, dd, W) -> list[tuple]:
    """Bay runs a room type may take and the part of their score that does not change while
    rooms are assigned (size, window, corner, core distance). Same on every floor."""
    out = []
    for iid, idxs in by_int.items():
        for s in range(len(idxs) - k + 1):
            run = [bays_proto[i] for i in idxs[s:s + k]]
            if not small and any(b.frontage < 0.75 * P.M for b in run):
                continue
            front = sum(b.frontage for b in run)
            depth = min(b.depth for b in run)
            if rt.is_key and (front < 0.85 * fn or depth < 0.85 * dn):
                continue                      # a guest room must keep ~its nominal size
            win = all(b.window for b in run)
            if rt.is_key and rt.exterior and RULES["guest_window_required"] and not win:
                continue                      # no windowless guest rooms
            sc = 0.0
            if not small:
                over = front * depth / max(area, 1.0)
                sc -= 1.2 * max(0.0, over - 1.2)  # oversize rooms waste area
            if rt.exterior:
                sc += W["exterior"] if win else -W["no_window"]
            elif win and not small:
                sc -= 0.4          # do not spend window bays on rooms that do not need them
            if rt.corner and any(b.corner for b in run):
                sc += W["corner"]
            d0 = float(min(dd[b.idx] for b in run))
            if rt.near_core:
                sc += W["core"] * (1.0 - min(d0, 150) / 150)
            sc -= W["core_pull"] * min(d0, 300) / 300
            out.append((iid, s, tuple(b.idx for b in run), sc, win, d0))
    return out


_SMALL_OK = set()  # any small room may share a bay with another small room


def _prune_stubs(P: Params, corridor: np.ndarray, spines: list[Spine], floors) -> np.ndarray:
    """Cut each spine back to what it serves: the doors of placed rooms (any level) and the points
    where it meets another corridor piece or a core. Only the free ends are removed, so the network
    stays connected."""
    doors = np.zeros(corridor.shape, bool)
    R, C = corridor.shape
    for fl in floors:
        for b in fl:
            if b.used > 0:
                r, c = min(max(b.door[0], 0), R - 1), min(max(b.door[1], 0), C - 1)
                doors[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
    core = np.zeros(corridor.shape, bool)
    for cm in P.site.cores:
        core |= cm.access
    out = corridor.copy()
    for s in spines:
        band = _band(corridor.shape, s.axis, s.coord, s.lo, s.hi, P.w) & corridor
        if not band.any():
            continue
        others = (out & ~band) | core
        anchor = band & (ndimage.binary_dilation(others) | doors)
        if not anchor.any():
            continue
        rr, cc = np.where(anchor)
        cut = band.copy()
        if s.axis == "H":
            cut[:, max(cc.min() - 1, 0):cc.max() + 2] = False
        else:
            cut[max(rr.min() - 1, 0):rr.max() + 2, :] = False
        # keep cells that another spine also covers
        for t in spines:
            if t is not s:
                cut &= ~_band(corridor.shape, t.axis, t.coord, t.lo, t.hi, P.w)
        out &= ~cut
    return out


def _door_dist(p: Placement, core_dist: np.ndarray) -> float:
    R, C = core_dist.shape
    if p.facing == "N":
        strip = core_dist[max(p.r0 - 1, 0):p.r0, p.c0:p.c0 + p.w]
    elif p.facing == "S":
        strip = core_dist[p.r0 + p.h:min(p.r0 + p.h + 1, R), p.c0:p.c0 + p.w]
    elif p.facing == "E":
        strip = core_dist[p.r0:p.r0 + p.h, max(p.c0 - 1, 0):p.c0]
    else:
        strip = core_dist[p.r0:p.r0 + p.h, p.c0 + p.w:min(p.c0 + p.w + 1, C)]
    return float(strip.min()) if strip.size else p.door_dist_ft


# ------------------------------------------------------------------ public API
def genome_length(n_instances: int) -> int:
    return N_SEG * 4 + n_instances


def random_genome(rng: np.random.Generator, n_instances: int) -> np.ndarray:
    g = rng.random(genome_length(n_instances))
    blocks = g[:N_SEG * 4].reshape(N_SEG, 4)
    blocks[:N_SP, 3] = np.where(rng.random(N_SP) < 0.4, rng.random(N_SP) * 0.1, 0.3 + 0.7 * rng.random(N_SP))
    blocks[:N_SP, :2] *= 0.5   # mostly mild trims
    return g


def _seed(rng, n_instances, active: list[int]) -> np.ndarray:
    g = random_genome(rng, n_instances)
    blocks = g[:N_SEG * 4].reshape(N_SEG, 4)
    blocks[:N_SP, 0:2] = 0.0
    blocks[:N_SP, 2] = 0.5
    blocks[:N_SP, 3] = 0.0
    for i in active:
        if i < N_SP:
            blocks[i, 3] = 1.0
    return g


def facade_seeds(site: Site, program: Program, n_instances: int, rng: np.random.Generator, n: int = 6) -> list[np.ndarray]:
    P = params_for(site, program)
    m = len(P.candidates)
    combos = [list(range(m)), [0], [0, 1], [0, 1, 2], list(range(min(m, 4))), [i for i in range(m) if i % 2 == 0]]
    return [_seed(rng, n_instances, c) for c in combos[:n] if c]


def spine_genomes(n_instances: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [random_genome(rng, n_instances) for _ in range(3)]


def decode(site: Site, program: Program, genome: np.ndarray, instances: list[Instance] | None = None) -> Design:
    instances = instances if instances is not None else program.instances(site.level_names)
    P = params_for(site, program)
    blocks = genome[:N_SEG * 4].reshape(N_SEG, 4)
    keys = genome[N_SEG * 4:N_SEG * 4 + len(instances)]
    spines = _active_spines(P, genome)
    corridor = np.zeros(P.pall.shape, bool)
    for s in spines:
        corridor |= _band(corridor.shape, s.axis, s.coord, s.lo, s.hi, P.w) & P.pall
    # pass 1: trim each spine to the extent of its bays (no corridor beyond the last room)
    first = _bays(P, corridor, [(s.axis, s.coord, s.lo, s.hi) for s in spines])
    trimmed = []
    for s in spines:
        mine = [b for b in first if (s.axis == "H" and b.facing in ("N", "S") and abs((b.door[0]) - s.coord) <= P.w)
                or (s.axis == "V" and b.facing in ("E", "W") and abs(b.door[1] - s.coord) <= P.w)]
        if mine:
            lo = min((b.c0 if s.axis == "H" else b.r0) for b in mine)
            hi = max((b.c0 + b.w - 1 if s.axis == "H" else b.r0 + b.h - 1) for b in mine)
            trimmed.append(Spine(s.axis, s.coord, max(s.lo, lo), min(s.hi, hi), s.score))
    spines = trimmed or spines
    corridor = np.zeros(P.pall.shape, bool)
    for s in spines:
        corridor |= _band(corridor.shape, s.axis, s.coord, s.lo, s.hi, P.w) & P.pall
    corridor = _connect(P, corridor, blocks[N_SP])
    core_dist, reached, dead = _distances(site, corridor, site.cell_ft, P.w)
    bays = _bays(P, corridor, _segments(spines, corridor, P.w))
    placements, unplaced, flex, floors = _assign(P, bays, instances, keys, core_dist)
    pruned = _prune_stubs(P, corridor, spines, floors)
    if pruned.sum() < corridor.sum():   # corridor stubs beyond the last room are removed (shorter dead ends)
        corridor = pruned
        core_dist, reached, dead = _distances(site, corridor, site.cell_ft, P.w)
        for p in placements:
            p.door_dist_ft = _door_dist(p, core_dist)
        near = ndimage.binary_dilation(corridor, iterations=2)
        flex = [f for f in flex if near[f[1]:f[1] + f[3], f[2]:f[2] + f[4]].any()]
    d = Design(site, program, corridor, placements, unplaced, core_dist, reached, dead, genome=genome,
               source="optimizer (architectural decoder)")
    d.extra = {"flex_bays": flex, "bays_per_floor": len(bays), "module_ft": P.M * site.cell_ft,
               "spines": [s.__dict__ for s in spines]}
    return d
