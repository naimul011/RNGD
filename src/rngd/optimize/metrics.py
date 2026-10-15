"""Objectives, code checks and fitness for a Design.

Every objective is normalised to [0, 1] (1 = best). Fitness is a weighted sum minus
penalties; the weights can be tuned by the LLM review agent (see review.py). Code limits
come from RNGD's IBC rule CSV (IBC-R1-023 travel, -025 corridor width, -027 dead end,
-018 exit separation, -015 exits) and the 2010 ADA guest-room table 224.2.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..hotel.details import ada_required
from .engine import Design

DEFAULT_WEIGHTS = {
    "placement": 0.30,   # priority-weighted share of the program placed
    "daylight": 0.14,    # rooms that need a window have one
    "efficiency": 0.10,  # little corridor per placeable area
    "egress": 0.12,      # travel distance and dead ends within IBC limits
    "ada": 0.08,         # ADA Table 224.2 count and dispersion over floors
    "core_access": 0.07, # service / ADA rooms open close to a core
    "adjacency": 0.07,   # required adjacencies (e.g. lobby-luggage)
    "corner": 0.03,      # suites on corners
    "balance": 0.05,     # keys spread evenly over guest floors
    "compactness": 0.04, # shared walls between rooms (prefab-friendly)
}
OBJECTIVE_TEXT = {
    "placement": "share of the program placed (priority weighted)",
    "daylight": "rooms that need a window have an exterior wall",
    "efficiency": "corridor area kept low",
    "egress": "exit travel and dead-end corridor within IBC limits",
    "ada": "ADA mobility rooms: count (Table 224.2) and spread over floors",
    "core_access": "service and ADA rooms open close to a core/elevator",
    "adjacency": "required room adjacencies met",
    "corner": "suites on corners",
    "balance": "keys spread evenly over floors",
    "compactness": "rooms share walls (compact, prefab-friendly)",
}


def code_limits() -> dict:
    """Numeric limits from RNGD's IBC CSV, with safe defaults if the file is missing."""
    lim = {"travel_ft": 250.0, "dead_end_ft": 50.0, "corridor_in": 44.0, "separation_ratio": 1 / 3, "exits": 2}
    try:
        from .. import real_docs
        df = real_docs.load_rules_df()
        if df is not None:
            by = {r.rule_id: r for r in df.itertuples()}
            num = lambda rid: float(re.findall(r"[\d.]+", str(by[rid].value))[0])  # noqa: E731
            lim.update(travel_ft=num("IBC-R1-023"), dead_end_ft=num("IBC-R1-027"), corridor_in=num("IBC-R1-025"),
                       exits=int(num("IBC-R1-015")))
    except Exception:
        pass
    return lim


LIMITS = code_limits()


def _shared_wall_ft(design: Design) -> float:
    total = 0.0
    R, C = design.site.shape
    for l in range(design.site.n_levels):
        lab = np.zeros((R, C), np.int32)
        for i, p in enumerate(design.placements, 1):
            if p.level == l:
                lab[p.r0:p.r0 + p.h, p.c0:p.c0 + p.w] = i
        a, b = lab[:, :-1], lab[:, 1:]
        total += np.count_nonzero((a > 0) & (b > 0) & (a != b))
        a, b = lab[:-1, :], lab[1:, :]
        total += np.count_nonzero((a > 0) & (b > 0) & (a != b))
    return total * design.site.cell_ft


def evaluate(design: Design, weights: dict | None = None) -> dict:
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    site, prog = design.site, design.program
    cf = site.cell_ft
    insts = prog.instances(site.level_names)
    pl = design.placements
    by_type = {rt.name: rt for rt in prog.rooms}

    total_pw = sum(i.type.priority * i.type.area_sf for i in insts) or 1.0
    # credit at most the nominal area: an oversize room is not worth more than the room it stands for
    placed_pw = sum(p.priority * min(p.area_sf, by_type[p.type_name].area_sf) for p in pl)
    f_place = placed_pw / total_pw

    need_ext = [p for p in pl if by_type[p.type_name].exterior]
    f_day = sum(p.exterior for p in need_ext) / len(need_ext) if need_ext else 1.0

    placeable_all = float(site.placeable[0].sum())
    corridor_ratio = float(design.corridor.sum()) / placeable_all if placeable_all else 1.0
    f_eff = float(np.clip(1.0 - corridor_ratio / 0.30, 0.0, 1.0))

    travel = max([p.door_dist_ft + p.depth_ft for p in pl], default=0.0)
    dead = design.dead_end_ft
    f_egress = 1.0 - 0.5 * min(1.0, max(0.0, travel - LIMITS["travel_ft"]) / LIMITS["travel_ft"]) \
        - 0.5 * min(1.0, max(0.0, dead - LIMITS["dead_end_ft"]) / LIMITS["dead_end_ft"])

    keys = [p for p in pl if p.category in ("guest", "ada", "suite")]
    n_keys = len(keys)
    ada_req = ada_required(n_keys)[2] if n_keys else 0
    ada_rooms = [p for p in pl if p.category == "ada"]
    ada_levels = len({p.level for p in ada_rooms})
    guest_levels = sorted({i for inst in insts if inst.type.is_key for i in inst.levels})
    disp_target = min(len(guest_levels), max(ada_req, 1)) if guest_levels else 1
    f_ada = (min(1.0, len(ada_rooms) / ada_req) if ada_req else 1.0) * min(1.0, ada_levels / disp_target if disp_target else 1)

    near = [p for p in pl if by_type[p.type_name].near_core]
    f_core = sum(1.0 - min(p.door_dist_ft, 120.0) / 120.0 for p in near) / len(near) if near else 1.0

    adj_scores, adj_w = [], 0.0
    for a in prog.adjacency:
        pa = [p for p in pl if p.type_name == a.a or p.name == a.a]
        pb = [p for p in pl if p.type_name == a.b or p.name == a.b]
        best = 0.0
        for x in pa:
            for y in pb:
                if x.level != y.level:
                    continue
                dist = (abs(x.center[0] - y.center[0]) + abs(x.center[1] - y.center[1])) * cf
                best = max(best, max(0.0, 1.0 - max(0.0, dist - a.max_ft) / a.max_ft))
        adj_scores.append(a.weight * best)
        adj_w += a.weight
    f_adj = sum(adj_scores) / adj_w if adj_w else 1.0

    corner_rooms = [p for p in pl if by_type[p.type_name].corner]
    f_corner = sum(p.corner for p in corner_rooms) / len(corner_rooms) if corner_rooms else 1.0

    per_level = [sum(1 for p in keys if p.level == l) for l in guest_levels] if guest_levels else [0]
    mean = float(np.mean(per_level)) if per_level else 0.0
    f_bal = float(np.clip(1.0 - (np.std(per_level) / mean if mean else 1.0), 0.0, 1.0))

    shared = _shared_wall_ft(design)
    perim = sum(2 * (p.frontage_ft + p.depth_ft) for p in pl) or 1.0
    f_comp = float(np.clip(2 * shared / perim / 0.6, 0.0, 1.0))

    terms = {"placement": f_place, "daylight": f_day, "efficiency": f_eff, "egress": f_egress, "ada": f_ada,
             "core_access": f_core, "adjacency": f_adj, "corner": f_corner, "balance": f_bal, "compactness": f_comp}
    total_w = sum(w[k] for k in terms) or 1.0
    fitness = sum(w[k] * terms[k] for k in terms) / total_w
    n_cores = len(site.cores)
    penalty = 0.5 * max(0, min(n_cores, LIMITS["exits"]) - design.cores_reached)
    fitness -= penalty

    # code checks for display / report
    sep = None
    if len(site.cores) >= 2:   # the two exits furthest apart
        sep = max(math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1])
                  for i, a in enumerate(site.cores) for b in site.cores[i + 1:]) * cf
    diag = site.diagonal_ft()
    width_in = prog.corridor_width_ft * 12
    checks = [
        ("IBC-R1-015", "Exits per story (cores reached)", f">= {LIMITS['exits']}", str(design.cores_reached),
         design.cores_reached >= LIMITS["exits"]),
        ("IBC-R1-018", "Exit separation (core to core)", f">= 1/3 diagonal = {diag / 3:.0f} ft",
         f"{sep:.0f} ft" if sep is not None else "-", sep is not None and sep >= diag / 3),
        ("IBC-R1-023", "Exit access travel", f"<= {LIMITS['travel_ft']:.0f} ft", f"{travel:.0f} ft",
         travel <= LIMITS["travel_ft"]),
        ("IBC-R1-025", "Corridor width", f">= {LIMITS['corridor_in']:.0f} in", f"{width_in:.0f} in",
         width_in >= LIMITS["corridor_in"]),
        ("IBC-R1-027", "Dead-end corridor", f"<= {LIMITS['dead_end_ft']:.0f} ft", f"{dead:.0f} ft",
         dead <= LIMITS["dead_end_ft"]),
        ("ADA-224.2", "ADA mobility rooms", f">= {ada_req} for {n_keys} keys", str(len(ada_rooms)),
         len(ada_rooms) >= ada_req),
    ]
    by_level = []
    for l, name in enumerate(site.level_names):
        lv = [p for p in pl if p.level == l]
        by_level.append({"level": name, "rooms": len(lv), "keys": sum(1 for p in lv if p.category in ("guest", "ada", "suite")),
                         "area_sf": round(sum(p.area_sf for p in lv)),
                         "by_type": {t: sum(1 for p in lv if p.type_name == t) for t in sorted({p.type_name for p in lv})}})
    unplaced_by_type: dict = {}
    for name in design.unplaced:
        t = next((i.type.name for i in insts if i.name == name), name)
        unplaced_by_type[t] = unplaced_by_type.get(t, 0) + 1
    return {
        "fitness": round(float(fitness), 4), "terms": {k: round(float(v), 4) for k, v in terms.items()},
        "weights": {k: round(w[k], 4) for k in terms},
        "rooms_placed": len(pl), "rooms_total": len(insts), "keys_placed": n_keys,
        "keys_requested": prog.keys_requested(site.level_names),
        "placed_area_sf": round(sum(p.area_sf for p in pl)), "program_area_sf": round(prog.total_area_sf(site.level_names)),
        "corridor_sf": round(float(design.corridor.sum()) * cf * cf), "corridor_ratio": round(corridor_ratio, 4),
        "placeable_sf_per_level": round(placeable_all * cf * cf),
        "max_travel_ft": round(travel, 1), "dead_end_ft": round(dead, 1), "cores_reached": design.cores_reached,
        "cores_total": n_cores, "exterior_fraction": round(f_day, 4), "ada_required": ada_req,
        "ada_placed": len(ada_rooms), "shared_wall_ft": round(shared),
        "unplaced_by_type": unplaced_by_type, "levels": by_level,
        "checks": [{"rule_id": a, "check": b, "required": c, "provided": d, "status": "PASS" if e else "FAIL"}
                   for a, b, c, d, e in checks],
    }


def fitness_from_terms(terms: dict, weights: dict, cores_reached: int, n_cores: int) -> float:
    """Re-score cached objective terms under new weights (no re-decoding needed)."""
    total_w = sum(weights[k] for k in terms) or 1.0
    f = sum(weights[k] * terms[k] for k in terms) / total_w
    return f - 0.5 * max(0, min(n_cores, LIMITS["exits"]) - cores_reached)
