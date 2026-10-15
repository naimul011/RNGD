"""Worked examples for the optimization-algorithms report (src/build_optimization_report.py).

Everything here is computed by the real optimizer code on site fp1 with QA/rooms_hotel.json
(and the default hotel program for the plate examples); nothing is typed in by hand.
Output: reports/optimization/examples.json + figures (PNG).
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rngd.hotel.program import default_program  # noqa: E402
from rngd.optimize import baseline, decide, engine as freeform, metrics, planner, render  # noqa: E402
from rngd.optimize import search as S  # noqa: E402
from rngd.optimize import typology as T  # noqa: E402
from rngd.optimize.explore import ExploreConfig  # noqa: E402
from rngd.optimize.program import load_program  # noqa: E402
from rngd.optimize.site import load_site  # noqa: E402

QA = ROOT / "QA"
OUT = ROOT / "reports" / "optimization"
STUDY = ROOT / "reports" / "walkthrough" / "optimizer" / "study.json"
POP, GENS, SEED = 24, 30, 7
NSEG4 = planner.N_SEG * 4


def _r(x, n=4):
    return round(float(x), n)


M_CAND = [planner.N_SP]   # number of candidate spines on the example site (set in main)


def _active(g) -> list[int]:
    b = np.asarray(g)[:NSEG4].reshape(planner.N_SEG, 4)
    return [i for i in range(min(planner.N_SP, M_CAND[0])) if b[i, 3] >= 0.12]


def describe_move(parent: np.ndarray, child: np.ndarray) -> str:
    """Words for the genes that differ between two genomes (the moves of explore._neighbours)."""
    diff = np.where(np.abs(np.asarray(parent) - np.asarray(child)) > 1e-9)[0]
    parts, keys = [], []
    for i in diff:
        if i < planner.N_SP * 4:
            s, j = divmod(int(i), 4)
            d = child[i] - parent[i]
            if s >= M_CAND[0]:
                continue                      # gene of a spine this site does not have: no effect
            if j == 3:
                parts.append(f"spine {s} switched {'on' if child[i] >= 0.12 else 'off'}")
            elif j == 2:
                parts.append(f"spine {s} shifted {'+' if d > 0 else '-'}1 ft")
            else:
                parts.append(f"spine {s} trimmed at its {'low' if j == 0 else 'high'} end")
        elif i < NSEG4:
            if "connector elbows redrawn" not in parts:
                parts.append("connector elbows redrawn")
        else:
            keys.append(child[i] - parent[i])
    if keys:
        parts.append(f"{len(keys)} rooms moved {'earlier' if np.mean(keys) > 0 else 'later'} in the order")
    return "; ".join(parts) or "only genes without effect on this site"


# ---------------------------------------------------------------------------------- decoder walk-through
def decoder_example(site, prog, genome) -> dict:
    P = planner.params_for(site, prog)
    cf = site.cell_ft
    inst = prog.instances(site.level_names)
    ex: dict = {}
    keys_rt = [rt for rt in prog.rooms if rt.is_key]
    ex["program"] = [{"name": rt.name, "width_ft": rt.width_ft, "depth_ft": rt.depth_ft, "count": rt.count,
                      "instances": sum(1 for i in inst if i.type is rt), "priority": rt.priority, "category": rt.category,
                      "window": rt.exterior, "near_core": rt.near_core, "per_floor": rt.per_floor, "area_sf": rt.area_sf}
                     for rt in prog.rooms]
    ex["module"] = {"fronts": [[rt.name, min(rt.width_ft, rt.depth_ft) if rt.category != "suite" else
                                min(rt.width_ft, rt.depth_ft) / 2, rt.count] for rt in keys_rt],
                    "depths": [[rt.name, max(rt.width_ft, rt.depth_ft)] for rt in keys_rt],
                    "M": P.M * cf, "Dnom": P.Dnom * cf, "Dmin": P.Dmin * cf, "Dmax": P.Dmax * cf, "w": P.w * cf,
                    "slack": P.slack * cf}
    ex["site"] = {"name": site.name, "rows": site.shape[0], "cols": site.shape[1], "levels": site.n_levels,
                  "level_names": site.level_names, "placeable_sf": round(site.placeable_sf(0)), "cores": len(site.cores),
                  "rooms": len(inst), "keys": sum(1 for i in inst if i.type.is_key), "genome_len": planner.genome_length(len(inst))}
    ex["candidates"] = [{"i": i, "axis": s.axis, "coord": s.coord, "lo": s.lo, "hi": s.hi, "length": s.hi - s.lo + 1,
                         "score": _r(s.score, 1), "per_ft": _r(s.score / (s.hi - s.lo + 1), 2)}
                        for i, s in enumerate(P.candidates)]
    # genome -> active spines
    blocks = np.asarray(genome)[:NSEG4].reshape(planner.N_SEG, 4)
    ex["genes"] = [{"i": i, "trim_lo": _r(blocks[i, 0], 2), "trim_hi": _r(blocks[i, 1], 2), "shift": _r(blocks[i, 2], 2),
                    "active": _r(blocks[i, 3], 2), "on": bool(blocks[i, 3] >= 0.12)} for i in range(len(P.candidates))]
    ex["elbows"] = [_r(x, 2) for x in blocks[planner.N_SP]]
    # the decode sequence of planner.decode, kept step by step
    spines = planner._active_spines(P, genome)
    corridor = np.zeros(P.pall.shape, bool)
    for s in spines:
        corridor |= planner._band(corridor.shape, s.axis, s.coord, s.lo, s.hi, P.w) & P.pall
    first = planner._bays(P, corridor, [(s.axis, s.coord, s.lo, s.hi) for s in spines])
    trimmed = []
    for s in spines:
        mine = [b for b in first if (s.axis == "H" and b.facing in ("N", "S") and abs(b.door[0] - s.coord) <= P.w)
                or (s.axis == "V" and b.facing in ("E", "W") and abs(b.door[1] - s.coord) <= P.w)]
        if mine:
            lo = min((b.c0 if s.axis == "H" else b.r0) for b in mine)
            hi = max((b.c0 + b.w - 1 if s.axis == "H" else b.r0 + b.h - 1) for b in mine)
            trimmed.append(planner.Spine(s.axis, s.coord, max(s.lo, lo), min(s.hi, hi), s.score))
    spines = trimmed or spines
    spine_only = np.zeros(P.pall.shape, bool)
    for s in spines:
        spine_only |= planner._band(spine_only.shape, s.axis, s.coord, s.lo, s.hi, P.w) & P.pall
    corridor = planner._connect(P, spine_only.copy(), blocks[planner.N_SP])
    core_dist, reached, dead0 = planner._distances(site, corridor, cf, P.w)
    bays = planner._bays(P, corridor, planner._segments(spines, corridor, P.w))
    ex["spines"] = [{"axis": s.axis, "coord": s.coord, "lo": s.lo, "hi": s.hi, "length": s.hi - s.lo + 1} for s in spines]
    ex["corridor"] = {"spine_cells": int(spine_only.sum()), "connector_cells": int((corridor & ~spine_only).sum()),
                      "cores_reached": int(reached), "dead_end_before_prune_ft": _r(dead0, 1)}
    by_int: dict = {}
    for b in bays:
        by_int.setdefault(b.interval, []).append(b)
    ints = sorted(by_int.values(), key=lambda v: -len(v))
    ex["bays"] = {"per_floor": len(bays), "window": sum(b.window for b in bays), "windowless": sum(not b.window for b in bays),
                  "corner": sum(b.corner for b in bays), "intervals": len(by_int),
                  "examples": [{"facing": v[0].facing, "window": bool(v[0].window), "length_ft": sum(b.frontage for b in v) * cf,
                                "n": len(v), "widths": [b.frontage * cf for b in v], "depth_ft": min(b.depth for b in v) * cf}
                               for v in ints[:4]]}
    # assignment: order and bays per room
    bay_area = P.M * P.Dnom
    rows = []
    for rt in prog.rooms:
        area = rt.area_sf / (cf * cf)
        small = area <= 0.45 * bay_area
        if prog.width_along_corridor:
            fn, dn = rt.width_ft / cf, min(rt.depth_ft / cf, float(P.Dnom))
        else:
            fn, dn = min(rt.width_ft, rt.depth_ft) / cf, max(rt.width_ft, rt.depth_ft) / cf
        k_area = 1 if small else max(1, int(area / bay_area + 0.5))
        k_front = math.ceil(0.85 * fn / P.M - 1e-9) if (rt.is_key and not small) else 0
        k = max(k_area, k_front)
        rows.append({"name": rt.name, "area": rt.area_sf, "ratio": _r(area / bay_area, 2), "k_area": k_area, "fn": fn,
                     "k_front": k_front, "k": k, "small": small, "order_value": _r(rt.priority / 10 * planner.PRIORITY_PULL + 0.5, 2)})
    ex["assign_rows"] = rows
    ex["bay_area"] = bay_area * cf * cf
    # static candidate scores (the part of sigma that does not depend on the rooms already placed), with the breakdown
    by_idx: dict = {}
    for b in bays:
        by_idx.setdefault(b.interval, []).append(b.idx)
    dd = np.array([core_dist[min(max(b.door[0], 0), core_dist.shape[0] - 1), min(max(b.door[1], 0), core_dist.shape[1] - 1)]
                   for b in bays])
    W = planner.PACK
    ex["sigma"] = []
    for rt in prog.rooms:
        row = next(r for r in rows if r["name"] == rt.name)
        area = rt.area_sf / (cf * cf)
        fn, dn = min(rt.width_ft, rt.depth_ft) / cf, max(rt.width_ft, rt.depth_ft) / cf
        cands = planner._static_candidates(P, bays, by_idx, rt, area, row["small"], row["k"], fn, dn, dd, W)
        out = []
        for iid, s_, idx_run, sc, win, d0 in sorted(cands, key=lambda c: -c[3]):
            run = [bays[i] for i in idx_run]
            front, depth = sum(b.frontage for b in run), min(b.depth for b in run)
            over = front * depth / max(area, 1.0)
            t = {"front_ft": front * cf, "depth_ft": depth * cf, "over": _r(over, 2),
                 "oversize": _r(-1.2 * max(0.0, over - 1.2), 3),
                 "window": _r((W["exterior"] if win else -W["no_window"]) if rt.exterior else (-0.4 if win else 0.0), 3),
                 "corner": _r(W["corner"], 3) if (rt.corner and any(b.corner for b in run)) else 0.0,
                 "d0_ft": _r(d0, 1), "core": _r(W["core"] * (1.0 - min(d0, 150) / 150), 3) if rt.near_core else 0.0,
                 "pull": _r(-W["core_pull"] * min(d0, 300) / 300, 3), "score": _r(sc, 3), "facing": run[0].facing}
            assert abs(t["oversize"] + t["window"] + t["corner"] + t["core"] + t["pull"] - sc) < 3e-3, (t, sc)
            out.append(t)
        ex["sigma"].append({"room": rt.name, "k": row["k"], "runs_total": sum(max(0, len(v) - row["k"] + 1) for v in by_idx.values()),
                            "runs_allowed": len(out), "best": out[:2], "worst": out[-1:], "near_core": rt.near_core,
                            "needs_window": rt.exterior, "is_key": rt.is_key, "need_front": _r(0.85 * fn * cf, 1),
                            "need_depth": _r(0.85 * dn * cf, 2)})
    ex["pack_weights"] = dict(W)
    design = planner.decode(site, prog, genome, inst)
    ex["after"] = {"corridor_cells_before": int(corridor.sum()), "corridor_cells_after": int(design.corridor.sum()),
                   "dead_end_ft": _r(design.dead_end_ft, 1), "rooms": len(design.placements), "unplaced": len(design.unplaced),
                   "flex_bays": len(design.extra.get("flex_bays", []))}

    # ---- figure: the four steps
    R, C = site.shape
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.6 / 2 * R / C * 2 + 0.8))
    axes = axes.ravel()
    base = np.ones((R, C, 3))
    for lab, col in render.BG.items():
        base[site.grids[1] == lab] = matplotlib.colors.to_rgb(col)

    def show(ax, title):
        ax.imshow(base, origin="lower", extent=(-0.5, C - 0.5, -0.5, R - 0.5), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_title(title, fontsize=10)

    ax = axes[0]
    show(ax, "1  Candidate spines (scored lines)")
    on = {c["i"] for c in ex["genes"] if c["on"]}
    for i, s in enumerate(P.candidates):
        xy = ([s.lo, s.hi], [s.coord, s.coord]) if s.axis == "H" else ([s.coord, s.coord], [s.lo, s.hi])
        ax.plot(*xy, color="#1E5AA8" if i in on else "#B0590B", lw=2.4, ls="-" if i in on else "--")
        ax.text(np.mean(xy[0]) + (2 if s.axis == "V" else 0), np.mean(xy[1]) + (2 if s.axis == "H" else 0), str(i),
                fontsize=9, color="#111", weight="bold", bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.8))
    ax = axes[1]
    show(ax, "2  Active spines trimmed, connectors added")
    img = np.zeros((R, C, 4))
    img[spine_only] = matplotlib.colors.to_rgba("#4D7798")
    img[corridor & ~spine_only] = matplotlib.colors.to_rgba("#E08A1E")
    ax.imshow(img, origin="lower", extent=(-0.5, C - 0.5, -0.5, R - 0.5), interpolation="nearest")
    ax = axes[2]
    show(ax, f"3  Bays: {ex['bays']['window']} with a window, {ex['bays']['windowless']} without")
    img[corridor] = matplotlib.colors.to_rgba("#4D7798")
    ax.imshow(img, origin="lower", extent=(-0.5, C - 0.5, -0.5, R - 0.5), interpolation="nearest")
    for b in bays:
        ax.add_patch(Rectangle((b.c0 - 0.5, b.r0 - 0.5), b.w, b.h, fc="#CFE6F7" if b.window else "#E4E4E4",
                               ec="#333", lw=0.5, hatch="xx" if b.corner else None))
    render.draw_level(axes[3], design, 1, title="4  Rooms assigned, corridor stubs cut (Level 2)")
    axes[3].title.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(OUT / "decoder_steps.png", dpi=170)
    plt.close(fig)

    # ---- figure: line scores
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0))
    for ax, axis, name in ((axes[0], "H", "row (horizontal corridor)"), (axes[1], "V", "column (vertical corridor)")):
        sc = P.line_scores[axis]
        ax.fill_between(np.arange(len(sc)), sc, color="#B9CDE6")
        ax.plot(sc, color="#123A6E", lw=1)
        for i, s in enumerate(P.candidates):
            if s.axis == axis:
                ax.axvline(s.coord, color="#B0590B", lw=1, ls="--")
                ax.text(s.coord, sc.max() * 1.02, str(i), ha="center", fontsize=8, color="#B0590B", weight="bold")
        ax.set_xlabel(f"{name}, ft"); ax.set_ylabel("line score s(k)"); ax.set_ylim(0, sc.max() * 1.15)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "line_scores.png", dpi=170)
    plt.close(fig)
    ex["line_score_max"] = {a: _r(P.line_scores[a].max(), 1) for a in ("H", "V")}
    return ex, design


# ---------------------------------------------------------------------------------- fitness
def fitness_example(design) -> dict:
    m = metrics.evaluate(design)
    w = metrics.DEFAULT_WEIGHTS
    rows = [{"name": k, "text": metrics.OBJECTIVE_TEXT[k], "value": m["terms"][k], "weight": w[k],
             "product": _r(w[k] * m["terms"][k], 4)} for k in m["terms"]]
    pl = design.placements
    keys = [p for p in pl if p.category in ("guest", "ada", "suite")]
    guest_levels = sorted({i for inst in design.program.instances(design.site.level_names) if inst.type.is_key for i in inst.levels})
    per_level = [sum(1 for p in keys if p.level == l) for l in guest_levels]
    near = [p for p in pl if next(rt for rt in design.program.rooms if rt.name == p.type_name).near_core]
    perim = sum(2 * (p.frontage_ft + p.depth_ft) for p in pl)
    return {"rows": rows, "sum_w": _r(sum(w.values()), 2), "sum_products": _r(sum(r["product"] for r in rows), 4),
            "fitness": m["fitness"], "corridor_ratio": m["corridor_ratio"], "corridor_sf": m["corridor_sf"],
            "placeable_sf": m["placeable_sf_per_level"], "travel_ft": m["max_travel_ft"], "dead_end_ft": m["dead_end_ft"],
            "limits": {k: metrics.LIMITS[k] for k in ("travel_ft", "dead_end_ft", "exits")},
            "ada_required": m["ada_required"], "ada_placed": m["ada_placed"],
            "ada_levels": len({p.level for p in pl if p.category == "ada"}), "keys": m["keys_placed"],
            "per_level": per_level, "sd": _r(np.std(per_level), 3), "mean": _r(np.mean(per_level), 2),
            "near_n": len(near), "near_mean_ft": _r(np.mean([p.door_dist_ft for p in near]), 1) if near else 0,
            "shared_ft": m["shared_wall_ft"], "perimeter_ft": round(perim), "rooms_placed": m["rooms_placed"],
            "rooms_total": m["rooms_total"], "cores_reached": m["cores_reached"], "cores_total": m["cores_total"],
            "checks": m["checks"]}


# ---------------------------------------------------------------------------------- GA with a crossover log
class LoggedGA(S.Optimizer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.xlog = []

    def _crossover(self, a, b):
        child = super()._crossover(a, b)
        if not np.array_equal(child, a):
            self.xlog.append((a.copy(), b.copy(), child.copy()))
        return child


def ga_example(site, prog) -> dict:
    cfg = S.SearchConfig(method="ga", population=POP, generations=GENS, seed=SEED)
    opt = LoggedGA(site, prog, cfg)
    t0 = time.time()
    res = opt.run()
    secs = time.time() - t0
    best = res["best"]
    hist = res["history"]
    sig = lambda g: cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * (g - 1) / max(1, cfg.generations - 1)  # noqa: E731
    pick = [1, 2, 3, 5, 10, 15, 20, 25, 30]
    ex = {"config": {k: getattr(cfg, k) for k in ("population", "generations", "elite", "tournament", "crossover_rate",
                                                  "sigma_start", "sigma_end", "segment_toggle_rate", "key_swap_rate",
                                                  "local_search_steps", "immigrants", "seed")},
          "history": [{"gen": h["gen"], "sigma": _r(sig(h["gen"]), 3), "best": h["best_fitness"], "mean": h["mean"],
                       "rooms": h["best_placed"], "evals": h["evaluations"]} for h in hist if h["gen"] in pick],
          "curve": [[h["evaluations"], h["best_fitness"], h["best_placed"], h["mean"]] for h in hist],
          "evaluations": res["evaluations"], "seconds": _r(secs, 1), "fitness": best.metrics["fitness"],
          "rooms": best.metrics["rooms_placed"], "rooms_total": best.metrics["rooms_total"],
          "active_spines": _active(best.genome)}
    # what crossover did: child (before mutation) against its two parents
    n0 = opt.evaluations
    recs = []
    for a, b, c in opt.xlog:
        fa, fb, fc = opt._eval(a).fitness, opt._eval(b).fitness, opt._eval(c).fitness
        recs.append((fa, fb, fc, a, b, c))
    hi = [max(r[0], r[1]) for r in recs]
    lo = [min(r[0], r[1]) for r in recs]
    better = sum(r[2] > h + 1e-6 for r, h in zip(recs, hi))
    equal = sum(abs(r[2] - h) <= 1e-6 for r, h in zip(recs, hi))
    between = sum(l - 1e-6 <= r[2] < h - 1e-6 for r, h, l in zip(recs, hi, lo))
    worse = sum(r[2] < l - 1e-6 for r, l in zip(recs, lo))
    diff = [r for r in recs if _active(r[3]) != _active(r[4])]   # parents with different corridor sets
    ex["crossover"] = {"n": len(recs), "better": better, "equal": equal, "between": between, "worse": worse,
                       "extra_evals": opt.evaluations - n0, "different_spine_sets": len(diff),
                       "diff_better": sum(r[2] > max(r[0], r[1]) + 1e-6 for r in diff),
                       "diff_worse_both": sum(r[2] < min(r[0], r[1]) - 1e-6 for r in diff),
                       "mean_parent_best": _r(np.mean(hi), 4), "mean_child": _r(np.mean([r[2] for r in recs]), 4)}

    def one(r):
        fa, fb, fc, a, b, c = r
        ka, kb, kc = a[NSEG4:], b[NSEG4:], c[NSEG4:]
        return {"fa": _r(fa), "fb": _r(fb), "fc": _r(fc), "spines_a": _active(a), "spines_b": _active(b), "spines_c": _active(c),
                "blocks_from_b": [i for i in range(planner.N_SEG) if not np.allclose(c[i * 4:i * 4 + 4], a[i * 4:i * 4 + 4])],
                "keys_from_b": int(np.sum(~np.isclose(kc, ka))), "keys_total": int(len(kc)),
                "rooms_a": opt._eval(a).metrics["rooms_placed"], "rooms_b": opt._eval(b).metrics["rooms_placed"],
                "rooms_c": opt._eval(c).metrics["rooms_placed"]}

    if diff:
        ex["crossover"]["example_worse"] = one(min(diff, key=lambda r: r[2] - min(r[0], r[1])))
        ex["crossover"]["example_better"] = one(max(diff, key=lambda r: r[2] - max(r[0], r[1])))
    # one mutation, gene by gene
    rng_state = opt.rng.bit_generator.state
    g0 = best.genome.copy()
    g1 = opt._mutate(g0, 0.15)
    opt.rng.bit_generator.state = rng_state
    d = np.abs(g1 - g0) > 1e-9
    ex["mutation"] = {"sigma": 0.15, "corridor_changed": int(d[:NSEG4].sum()), "corridor_total": NSEG4,
                      "keys_changed": int(d[NSEG4:].sum()), "keys_total": int(len(g0) - NSEG4),
                      "spines_before": _active(g0), "spines_after": _active(g1),
                      "fitness_before": _r(opt._eval(g0).fitness), "fitness_after": _r(opt._eval(g1).fitness),
                      "sample": [[int(i), _r(g0[i], 3), _r(g1[i], 3)] for i in np.where(d[:NSEG4])[0][:4]]}
    return ex, best


# ---------------------------------------------------------------------------------- systematic search with an evaluation log
class LoggedSys(S.Optimizer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.elog = []

    def _eval(self, g):
        ind = super()._eval(g)
        self.elog.append((ind.genome.copy(), ind.fitness, ind.metrics["rooms_placed"], self.evaluations))
        return ind


def systematic_example(site, prog) -> dict:
    cfg = S.SearchConfig(method="systematic", population=POP, generations=GENS, seed=SEED)
    opt = LoggedSys(site, prog, cfg)
    t0 = time.time()
    res = opt.run()
    secs = time.time() - t0
    log = opt.elog
    P = planner.params_for(site, prog)
    m = len(P.candidates)
    top = list(range(min(m, 7)))
    combos = [list(range(m))] + [list(c) for k in range(1, min(4, len(top)) + 1) for c in itertools.combinations(top, k)]
    n1 = len(combos)
    stage1 = [{"spines": c, "fitness": _r(log[i][1]), "rooms": log[i][2]} for i, c in enumerate(combos)]
    order = sorted(stage1, key=lambda r: -r["fitness"])
    ex = {"m": m, "n_sets": n1, "by_size": {k: math.comb(len(top), k) for k in range(1, min(4, len(top)) + 1)},
          "stage1_top": order[:6], "stage1_bottom": order[-3:], "stage1_best": order[0]["fitness"],
          "stage1_spread": [order[-1]["fitness"], order[0]["fitness"]], "budget": POP * GENS,
          "evaluations": res["evaluations"], "calls": len(log), "seconds": _r(secs, 1),
          "fitness": res["best"].metrics["fitness"], "rooms": res["best"].metrics["rooms_placed"],
          "active_spines": _active(res["best"].genome),
          "curve": [[h["evaluations"], h["best_fitness"], h["best_placed"]] for h in res["history"]]}
    # improvements after stage 1: find the parent (closest earlier genome) and name the move
    best, steps = max(l[1] for l in log[:n1]), []
    for i in range(n1, len(log)):
        g, f, rooms, ev = log[i]
        if f > best + 1e-6:
            j = min(range(i), key=lambda q: int(np.sum(np.abs(log[q][0] - g) > 1e-9)))
            steps.append({"call": i + 1, "from": _r(log[j][1]), "to": _r(f), "rooms_from": log[j][2], "rooms": rooms,
                          "move": describe_move(log[j][0], g), "genes": int(np.sum(np.abs(log[j][0] - g) > 1e-9))})
            best = f
    ex["improvements"] = steps
    run_best, exact = -1.0, []
    for g, f, rooms, ev in log:   # best fitness after every decoder call (the history is only recorded per batch)
        run_best = max(run_best, f)
        exact.append([ev, _r(run_best)])
    ex["exact_curve"] = exact
    ex["moves_per_neighbourhood"] = planner.N_SP + 4 * len(_active(res["best"].genome)) + 1 + 4
    # a full pattern-search descent from the best enumerated layout, every accepted move named (same rule and move set)
    from rngd.optimize.explore import _neighbours
    inst = prog.instances(site.level_names)
    rng = np.random.default_rng(SEED)
    cur_g = log[max(range(n1), key=lambda q: log[q][1])][0]
    cur = opt._eval(cur_g)
    trace, tried_total = [], 0
    for step in range(12):
        improved = False
        for t, h in enumerate(_neighbours(cur.genome, len(inst), rng, step), 1):
            cand = opt._eval(h)
            tried_total += 1
            if cand.fitness > cur.fitness + 1e-6:
                trace.append({"step": step + 1, "tried": t, "move": describe_move(cur.genome, cand.genome),
                              "from": _r(cur.fitness), "to": _r(cand.fitness), "rooms": cand.metrics["rooms_placed"]})
                cur, improved = cand, True
                break
        if not improved:
            trace.append({"step": step + 1, "tried": t, "move": "no neighbour is better: local optimum", "from": _r(cur.fitness),
                          "to": _r(cur.fitness), "rooms": cur.metrics["rooms_placed"]})
            break
    ex["descent"] = {"start_spines": _active(cur_g), "trace": trace, "tried_total": tried_total}
    # same search again: identical?
    r2 = S.Optimizer(site, prog, cfg).run()
    ex["repeat_identical"] = bool(r2["best"].metrics["fitness"] == res["best"].metrics["fitness"]
                                  and np.array_equal(r2["best"].genome, res["best"].genome))
    return ex, res["best"]


# ---------------------------------------------------------------------------------- free-form decoder (first attempt)
def freeform_example(site, prog) -> dict:
    inst = prog.instances(site.level_names)
    rng = np.random.default_rng(SEED)
    seeds = freeform.facade_seeds(site, prog, len(inst), rng)
    g = seeds[0]
    d = freeform.decode(site, prog, g, inst)
    m = metrics.evaluate(d)
    seg = g[:freeform.N_SEG * 4].reshape(freeform.N_SEG, 4)
    keys = [p for p in d.placements if p.category in ("guest", "suite", "ada")]
    fig, ax = plt.subplots(figsize=(4.6, 4.6 * site.shape[0] / site.shape[1] + 0.5))
    render.draw_level(ax, d, 1, title=f"Free-form decoder, one seed genome: {m['rooms_placed']}/{m['rooms_total']} rooms")
    fig.tight_layout()
    fig.savefig(OUT / "freeform_seed.png", dpi=170)
    plt.close(fig)
    return {"segments": [[_r(x, 2) for x in s] for s in seg if s[3] >= 0.12], "n_seg": freeform.N_SEG,
            "rooms": m["rooms_placed"], "rooms_total": m["rooms_total"], "fitness": m["fitness"],
            "windowless": sum(1 for p in keys if not p.exterior), "corridor_ratio": m["corridor_ratio"]}


# ---------------------------------------------------------------------------------- reference GA (from its result files)
def reference_example() -> dict:
    out = {"sites": []}
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, col in zip((1, 2, 3), ("#1E5AA8", "#B0590B", "#1E7B3A")):
        d = json.loads((QA / f"floorplan{i}.json").read_text(encoding="utf-8"))
        h = d["history"]
        un = [x[0] for x in h]
        last = max(k for k in range(len(un)) if k == 0 or un[k] != un[k - 1])
        best_i = int(np.argmin(un))
        out["sites"].append({"site": f"fp{i}", "generations": len(h), "start_unplaced": un[0], "end_unplaced": un[-1],
                             "start_score": h[0][1], "end_score": h[-1][1], "first_best_gen": best_i + 1,
                             "last_change_gen": last + 1, "rooms_placed": d["rooms_placed"], "rooms_total": d["rooms_total"],
                             "score": d["score"], "seed": d["seed"], "genes": len(d["corridor_genome"]),
                             "genome_head": d["corridor_genome"][:3], "corridor_width_ft": d["corridor_width_ft"],
                             "reported_exterior_fraction": d["exterior_fraction"], "adjacency_weight": d["adjacency_weight"],
                             "balance_weight": d["balance_weight"]})
        ax.plot(np.arange(1, len(un) + 1), un, color=col, lw=1.3, label=f"fp{i}")
    ax.set_xlabel("generation"); ax.set_ylabel("rooms not placed"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "reference_history.png", dpi=170)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------------- hotel plates
def plate_example() -> dict:
    hp = default_program()
    cfg = ExploreConfig()
    F = 4
    names = T.level_names(F)
    prog = T.program_from_hotel(hp, names)
    need = T.need_by_level(prog, names)
    main = T.main_core_size(hp.keys)
    wd = T.wing_depth()
    core_area = main[0] * main[1] + 2 * T.STAIR_CORE[0] * T.STAIR_CORE[1]
    run = (max(need) * 1.04 + core_area) / (2 * T.ROOM_DEPTH_FT * 0.93)
    plates = T.generate_plates(prog, names, hp.keys, tuple(cfg.envelope), cfg.shapes, cfg.schemes, cfg.ratios)
    key_fronts = sorted(((rt.width_ft, rt.count) for rt in prog.rooms if rt.is_key))
    cum, half, M = 0, sum(c for _, c in key_fronts) / 2, key_fronts[0][0]
    for f, c in key_fronts:
        cum += c
        if cum >= half:
            M = f
            break
    M = max(6, int(round(M)))
    rows, seen = [], set()
    for p in plates:
        a, b = p.size
        rows.append({"name": p.name, "shape": p.shape, "scheme": p.core_scheme, "a": a, "b": b,
                     "run": T._run_ft(p.shape, a, b, wd), "footprint": p.footprint_sf(), "facade": p.facade_ft(),
                     "clipped": p.clipped, "cores": len(p.cores), "ub": T.keys_upper_bound(p, M)})
    dims = []
    for r in rows:
        k = (r["shape"], r["a"], r["b"])
        if k not in seen:
            seen.add(k)
            dims.append({k2: r[k2] for k2 in ("shape", "a", "b", "run", "footprint", "facade", "clipped")}
                        | {"schemes": sorted({x["scheme"] for x in rows if (x["shape"], x["a"], x["b"]) == k})})
    ex = {"keys": hp.keys, "floors": F, "envelope": list(cfg.envelope), "need_by_level": [round(x) for x in need],
          "need_max": round(max(need)), "busiest": names[int(np.argmax(need))], "main_core": list(main),
          "stair": list(T.STAIR_CORE), "core_area": core_area, "run": _r(run, 1), "wing_depth": wd, "room_depth": T.ROOM_DEPTH_FT,
          "corridor_ft": T.CORRIDOR_FT, "elevators": T.elevators_for(hp.keys), "plates": len(plates), "dims": dims,
          "module_ft": M, "floor_options": [x for x in (F, F + 1, F + 2, F + 3)], "seeds_per_plate": cfg.seeds_per_plate,
          "refine_top": cfg.refine_top, "refine_steps": cfg.refine_steps,
          "ub_example": next(r for r in rows if r["shape"] == "Bar" and r["scheme"] == "center")}
    # figure: one plate per shape, with its cores
    pick = []
    for sh in T.SHAPES:
        c = [p for p in plates if p.shape == sh]
        if c:
            pick.append(next((p for p in c if p.core_scheme == "center"), c[0]))
    fig, axes = plt.subplots(1, len(pick), figsize=(2.6 * len(pick), 3.4))
    for ax, p in zip(np.atleast_1d(axes), pick):
        for x, y, w, h in p.rects:
            ax.add_patch(Rectangle((x, y), w, h, fc="#F4F1E8", ec="#444", lw=0.8))
        for x, y, w, h in p.rects:   # corridor centre-line of each wing
            if w >= h:
                ax.plot([x, x + w], [y + h / 2] * 2, color="#4D7798", lw=2)
            else:
                ax.plot([x + w / 2] * 2, [y, y + h], color="#4D7798", lw=2)
        for x, y, w, h, kind in p.cores:
            ax.add_patch(Rectangle((x, y), w, h, fc="#7A2E4D" if kind == "main" else "#C0607F", ec="#222", lw=0.6))
        a, b = p.size
        ax.set_xlim(-8, max(a, b) + 8); ax.set_ylim(-8, max(a, b) + 8); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{p.shape}\n{a} x {b} ft, run {T._run_ft(p.shape, a, b, wd):.0f} ft", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plates.png", dpi=170)
    plt.close(fig)
    return ex


# ---------------------------------------------------------------------------------- decision
def decision_example(study: dict) -> dict:
    top = study["hotel"]["top"]
    by_shape = {r["shape"]: r for r in study["hotel"]["best_by_shape"]}
    rows = [top[0], by_shape.get("L", top[1]), by_shape.get("Courtyard", top[6])]
    rows.append(next((r for r in top if r["floors"] != rows[0]["floors"]), top[-1]))
    keys = ["fitness", "gross_sf_per_key", "dead_end_ft"]
    weights = {k: decide.PRESETS["Balanced"][k] for k in keys}
    X = np.array([[float(r[k]) for k in keys] for r in rows])
    sign = np.array([decide.CRITERIA[k][0] for k in keys])
    norm = np.sqrt((X ** 2).sum(0))
    w = np.array([weights[k] for k in keys], float)
    V = X / norm * (w / w.sum())
    ideal = np.where(sign > 0, V.max(0), V.min(0))
    worst = np.where(sign > 0, V.min(0), V.max(0))
    dp = np.sqrt(((V - ideal) ** 2).sum(1))
    dm = np.sqrt(((V - worst) ** 2).sum(1))
    C = dm / (dp + dm)
    alts = [SimpleNamespace(criteria={k: float(r[k]) for k in decide.CRITERIA}) for r in rows]
    assert np.allclose(decide.topsis(alts, weights), C)
    front = decide.pareto_mask(alts, keys)
    dom = []
    S_ = X * sign
    for i in range(len(rows)):
        by = [j for j in range(len(rows)) if j != i and np.all(S_[j] >= S_[i]) and np.any(S_[j] > S_[i])]
        dom.append(by)
    return {"keys": keys, "labels": [decide.CRITERIA[k][1] for k in keys], "sign": sign.tolist(), "weights": weights,
            "rows": [{"layout": r["layout"], "floors": r["floors"], "x": [_r(v, 3) for v in X[i]],
                      "r": [_r(v, 4) for v in (X / norm)[i]], "v": [_r(v, 4) for v in V[i]], "dp": _r(dp[i], 4),
                      "dm": _r(dm[i], 4), "C": _r(C[i], 3), "pareto": bool(front[i]), "dominated_by": dom[i],
                      "full_rank": r["rank"]} for i, r in enumerate(rows)],
            "norm": [_r(v, 3) for v in norm], "wn": [_r(v, 3) for v in w / w.sum()], "ideal": [_r(v, 4) for v in ideal],
            "worst": [_r(v, 4) for v in worst], "order": [int(i) for i in np.argsort(-C)],
            "presets": {k: v for k, v in decide.PRESETS.items()}, "n_criteria": len(decide.CRITERIA)}


def curves(ga: dict, sy: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
    ax = axes[0]
    ax.step([c[0] for c in ga["curve"]], [c[1] for c in ga["curve"]], where="post", color="#4D7798", lw=1.8,
            label="C  genetic algorithm")
    ax.step([c[0] for c in sy["exact_curve"]], [c[1] for c in sy["exact_curve"]], where="post", color="#1E7B3A", lw=1.8,
            label="D  systematic search")
    ax.axvline(sy["n_sets"], color="#1E7B3A", lw=0.9, ls=":")
    ax.set_xlabel("layouts evaluated"); ax.set_ylabel("best fitness F so far"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    ax.plot([c[0] for c in ga["curve"]], [c[1] for c in ga["curve"]], color="#4D7798", lw=1.8, label="best individual")
    ax.plot([c[0] for c in ga["curve"]], [c[3] for c in ga["curve"]], color="#B0590B", lw=1.2, ls="--",
            label="mean of the population")
    ax.set_xlabel("layouts evaluated (GA)"); ax.set_ylabel("fitness F"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=170)
    plt.close(fig)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    site, prog = load_site(QA / "fp1_multi.json"), load_program(QA / "rooms_hotel.json")
    M_CAND[0] = len(planner.params_for(site, prog).candidates)
    t0 = time.time()
    sy, best_d = systematic_example(site, prog)
    print(f"systematic: {sy['rooms']} rooms, F {sy['fitness']}, {time.time() - t0:.0f}s", flush=True)
    ga, best_c = ga_example(site, prog)
    print(f"ga: {ga['rooms']} rooms, F {ga['fitness']}, {time.time() - t0:.0f}s", flush=True)
    dec, design = decoder_example(site, prog, best_d.genome)
    ex = {"built": time.strftime("%Y-%m-%d %H:%M"), "budget": [POP, GENS, SEED], "decoder": dec,
          "fitness": fitness_example(design), "ga": ga, "systematic": sy, "freeform": freeform_example(site, prog),
          "reference": reference_example(), "plates": plate_example(), "decision": decision_example(study),
          "genomes": {"C": [float(x) for x in best_c.genome], "D": [float(x) for x in best_d.genome]}}
    curves(ga, sy)
    # reference design of fp1 next to the final layout (same evaluator)
    ref, _ = baseline.load_reference(QA / "floorplan1.json", site, prog)
    mr = metrics.evaluate(ref)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8 * site.shape[0] / site.shape[1] + 0.9))
    render.draw_level(axes[0], ref, 1, title=f"Reference GA, Level 2: {mr['rooms_placed']}/{mr['rooms_total']} rooms, F = {mr['fitness']:.3f}")
    render.draw_level(axes[1], design, 1, title=f"Systematic search, Level 2: {len(design.placements)}/{mr['rooms_total']} rooms, "
                                                f"F = {ex['fitness']['fitness']:.3f}")
    fig.legend(handles=render.legend_handles(), loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "before_after.png", dpi=170)
    plt.close(fig)
    ex["reference"]["fp1_eval"] = {"fitness": mr["fitness"], "terms": mr["terms"], "rooms": mr["rooms_placed"]}
    (OUT / "examples.json").write_text(json.dumps(ex, indent=1), encoding="utf-8")
    print(f"examples written in {time.time() - t0:.0f}s")
    return ex


if __name__ == "__main__":
    main()
