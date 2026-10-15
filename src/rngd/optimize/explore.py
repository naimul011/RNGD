"""Deterministic layout exploration for the Hotel Schematic (no LLM in the loop).

Why not the genetic algorithm: crossover splices half of one plan onto half of another, which
rarely makes architectural sense (raised in the 10/08/2026 meeting), and the GA needed ~10
minutes. The fitness here is pure geometry + code arithmetic (metrics.evaluate), so the same
inputs always give the same ranking, in seconds. The search runs in three stages:

  1. ENUMERATE  every plate (shape x proportion x core scheme, typology.py) for each candidate
                floor count, and on each plate a fixed set of corridor-spine combinations
                (all spines, the best one, best two, ...). Plates run in parallel processes.
  2. REFINE     the best plates of each shape with a deterministic pattern search: switch
                each spine on/off, trim/shift it, and reorder room groups; a move is kept
                only if it improves fitness (first-improvement hill climbing, seeded).
  3. ARCHIVE    every distinct layout (plate + floors + corridor set) is kept with its
                objectives, so the decision step (decide.py) can rank them and the user can
                browse all of them.
"""
from __future__ import annotations

import os
import sys
import time
import types
import zlib
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import metrics, planner, typology as T
from .program import Program


@dataclass
class ExploreConfig:
    guest_floors: int = 5
    floors_mode: str = "auto"          # "auto" (try F-1..F+2) or "fixed"
    max_guest_floors: int = 8          # code / zoning height limit on guest floors
    envelope: tuple[int, int] = (170, 265)
    shapes: tuple[str, ...] = T.SHAPES
    schemes: tuple[str, ...] = tuple(T.CORE_SCHEMES)
    ratios: tuple[float, ...] = (0.6, 1.0)
    seeds_per_plate: int = 6
    refine_top: int = 8                # plates refined after screening (best of each shape first)
    refine_steps: int = 24
    floor_height_ft: float = 10.75
    ground_height_ft: float = 16.0
    workers: int = 0                   # 0 = auto, 1 = serial
    seed: int = 7
    weights: dict = field(default_factory=lambda: dict(metrics.DEFAULT_WEIGHTS))


@dataclass
class Alternative:
    id: str
    plate: T.Plate
    floors: int                        # guest floors (levels = floors + 1)
    genome: np.ndarray
    metrics: dict
    stage: str                         # "screen" | "refined"
    upper_bound_keys: int = 0          # per typical floor
    criteria: dict = field(default_factory=dict)

    @property
    def fitness(self) -> float:
        return self.metrics["fitness"]


def elevations(cfg: ExploreConfig, floors: int) -> list[float]:
    return [0.0] + [cfg.ground_height_ft + i * cfg.floor_height_ft for i in range(floors)]


def _floor_options(cfg: ExploreConfig) -> list[int]:
    f = max(1, int(cfg.guest_floors))
    opts = [f] if cfg.floors_mode == "fixed" else [x for x in (f - 1, f, f + 1, f + 2) if x >= 2]
    return [x for x in opts if x <= cfg.max_guest_floors] or [min(f, cfg.max_guest_floors)]


def _rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def _setup(rows, source, plate: T.Plate, floors: int, cfg: ExploreConfig):
    from ..hotel.program import Program as HP
    names = T.level_names(floors)
    prog = T.program_from_hotel(HP(rows=list(rows), source=source), names)
    site = T.plate_site(plate, names, elevations(cfg, floors))
    return prog, site, prog.instances(names)


MAX_STORIES = 5   # IBC-R1-007 in RNGD's rule set: Type IIA, sprinklered R-1


def _evaluate(design, cfg: ExploreConfig) -> dict:
    """metrics.evaluate plus the story-count check (the plate search changes the floor count)."""
    m = metrics.evaluate(design, cfg.weights)
    stories = design.site.n_levels
    m["checks"].append({"rule_id": "IBC-R1-007", "check": "Stories above grade",
                        "required": f"<= {MAX_STORIES} (Type IIA, sprinklered R-1)", "provided": str(stories),
                        "status": "PASS" if stories <= MAX_STORIES else "FAIL"})
    return m


def _det_keys(g: np.ndarray, n_inst: int) -> np.ndarray:
    g = g.copy()
    g[planner.N_SEG * 4:] = 0.5   # priority order only: deterministic, the decoder ranks by priority
    return g


def _screen_plate(args):
    rows, source, plate, floors, cfg = args
    t0 = time.process_time()
    prog, site, inst = _setup(rows, source, plate, floors, cfg)
    rng = _rng_for(cfg.seed, plate.name, floors)
    out = []
    for g in planner.facade_seeds(site, prog, len(inst), rng, n=cfg.seeds_per_plate):
        g = _det_keys(g, len(inst))
        m = _evaluate(planner.decode(site, prog, g, inst), cfg)
        out.append((m["fitness"], g, m))
    P = planner.params_for(site, prog)
    return plate, floors, out, T.keys_upper_bound(plate, P.M * site.cell_ft), time.process_time() - t0


def _neighbours(g: np.ndarray, n_inst: int, rng: np.random.Generator, step: int):
    """Deterministic move set: spine toggles first, then trims/shifts, then room-group swaps."""
    blocks = g[:planner.N_SEG * 4].reshape(planner.N_SEG, 4)
    moves = []
    for i in range(planner.N_SP):
        h = g.copy()
        hb = h[:planner.N_SEG * 4].reshape(planner.N_SEG, 4)
        hb[i, 3] = 0.0 if blocks[i, 3] >= 0.12 else 1.0
        moves.append(h)
    for i in range(planner.N_SP):
        if blocks[i, 3] < 0.12:
            continue
        for j, d in ((0, 0.15), (1, 0.15), (2, 0.25), (2, -0.25)):
            h = g.copy()
            hb = h[:planner.N_SEG * 4].reshape(planner.N_SEG, 4)
            hb[i, j] = np.clip(hb[i, j] + d, 0, 1)
            moves.append(h)
    h = g.copy()
    h[planner.N_SEG * 4:planner.N_SEG * 4 + 4] = rng.random(4)   # connector elbows
    moves.append(h)
    for _ in range(4):   # nudge a few rooms ahead in the assignment order
        h = g.copy()
        idx = rng.choice(n_inst, size=max(1, n_inst // 12), replace=False)
        h[planner.N_SEG * 4 + idx] = np.clip(h[planner.N_SEG * 4 + idx] + rng.choice([-0.3, 0.3]), 0, 1)
        moves.append(h)
    order = rng.permutation(len(moves))
    return [moves[i] for i in order]


def _refine(args):
    rows, source, plate, floors, cfg, g0, f0 = args
    prog, site, inst = _setup(rows, source, plate, floors, cfg)
    rng = _rng_for(cfg.seed, "refine", plate.name, floors)
    best_g, best_m = g0, _evaluate(planner.decode(site, prog, g0, inst), cfg)
    trace = [best_m["fitness"]]
    evals = 0
    step = 0
    while evals < cfg.refine_steps:
        improved = False
        for h in _neighbours(best_g, len(inst), rng, step):
            if evals >= cfg.refine_steps:
                break
            m = _evaluate(planner.decode(site, prog, h, inst), cfg)
            evals += 1
            if m["fitness"] > best_m["fitness"] + 1e-6:
                best_g, best_m, improved = h, m, True
                trace.append(m["fitness"])
                break
        step += 1
        if not improved:
            break
    return plate, floors, best_g, best_m, trace


@contextmanager
def _plain_main():
    """Spawned workers re-import the parent's __main__. Under Streamlit that is the app script
    (app.py), so every worker would re-run the whole app (models, data). Hide it while workers start."""
    main = sys.modules.get("__main__")
    if main is not None and "streamlit" in sys.modules and getattr(main, "__file__", None):
        sys.modules["__main__"] = types.ModuleType("__main__")
        try:
            yield
        finally:
            sys.modules["__main__"] = main
    else:
        yield


_POOL: dict = {}
LAST_MODE = {"mode": "serial", "reason": ""}


def _pool(cfg: ExploreConfig, n_tasks: int):
    """A process pool kept between runs (spawning workers on Windows costs seconds)."""
    n = cfg.workers or min(8, max(1, (os.cpu_count() or 2) - 2))
    if n <= 1 or n_tasks <= 1:
        return None
    if n not in _POOL:
        _POOL.clear()
        _POOL[n] = ProcessPoolExecutor(max_workers=n)
    return _POOL[n]


def _map(cfg, fn, tasks, on_done=None):
    out = []
    try:
        pool = _pool(cfg, len(tasks))
    except Exception as e:
        pool, LAST_MODE["reason"] = None, f"pool: {e!r}"
    if pool is not None:
        try:
            with _plain_main():
                results = pool.map(fn, tasks, timeout=900)   # submits (and spawns) eagerly
            for r in results:
                out.append(r)
                if on_done:
                    on_done(len(out), len(tasks))
            LAST_MODE.update(mode=f"{pool._max_workers} processes", reason="")
            return out
        except Exception as e:   # e.g. no process spawning in this host: fall back to serial
            _POOL.clear()
            out, LAST_MODE["reason"] = [], f"parallel failed: {e!r}"[:300]
    LAST_MODE["mode"] = "serial"
    for t in tasks:
        out.append(fn(t))
        if on_done:
            on_done(len(out), len(tasks))
    return out


def explore(hotel_program, cfg: ExploreConfig, on_progress=None) -> dict:
    """Run the three stages; returns alternatives (all distinct layouts) and a log."""
    t0 = time.time()
    rows, source = tuple(hotel_program.rows), hotel_program.source
    keys = hotel_program.keys
    tasks = []
    for F in _floor_options(cfg):
        names = T.level_names(F)
        prog = T.program_from_hotel(hotel_program, names)
        for plate in T.generate_plates(prog, names, keys, tuple(cfg.envelope), cfg.shapes, cfg.schemes, cfg.ratios):
            tasks.append((rows, source, plate, F, cfg))
    log = [f"{len(tasks)} plate x floor-count variants (shapes: {', '.join(cfg.shapes)}; "
           f"floors: {', '.join(map(str, _floor_options(cfg)))})"]

    def prog_cb(stage):
        return (lambda i, n: on_progress(stage, i, n)) if on_progress else None

    screened = _map(cfg, _screen_plate, tasks, prog_cb("screen"))
    alts: dict[tuple, Alternative] = {}
    n_eval = 0

    def add(plate, F, g, m, stage, ub):
        # distinct = different built result (several genomes can decode to the same plan)
        sig = (plate.name, F, m["rooms_placed"], m["corridor_sf"], m["shared_wall_ft"], round(m["fitness"], 3))
        cur = alts.get(sig)
        if cur is None or m["fitness"] > cur.fitness:
            alts[sig] = Alternative("", plate, F, g, m, stage, ub)

    ub_by = {}
    cpu = sum(x[4] for x in screened)
    screened = [x[:4] for x in screened]
    for plate, F, res, ub in screened:
        ub_by[(plate.name, F)] = ub
        for f, g, m in res:
            n_eval += 1
            add(plate, F, g, m, "screen", ub)
    log.append(f"screened {n_eval} layouts in {time.time() - t0:.1f}s ({cpu / max(n_eval, 1):.2f} CPU-s per layout)")

    # refine: best plate of every shape first (diversity), then the next best overall
    best_per_plate = {}
    for a in alts.values():
        k = (a.plate.name, a.floors)
        if k not in best_per_plate or a.fitness > best_per_plate[k].fitness:
            best_per_plate[k] = a
    ranked = sorted(best_per_plate.values(), key=lambda a: -a.fitness)
    picks, shapes_done = [], set()
    for a in ranked:
        if a.plate.shape not in shapes_done:
            picks.append(a)
            shapes_done.add(a.plate.shape)
    for a in ranked:
        if len(picks) >= max(cfg.refine_top, len(shapes_done)):
            break
        if a not in picks:
            picks.append(a)
    t1 = time.time()
    refined = _map(cfg, _refine, [(rows, source, a.plate, a.floors, cfg, a.genome, a.fitness) for a in picks],
                   prog_cb("refine"))
    traces = []
    for (plate, F, g, m, trace), a in zip(refined, picks):
        n_eval += cfg.refine_steps
        add(plate, F, g, m, "refined", ub_by[(plate.name, F)])
        traces.append({"plate": plate.name, "floors": F, "start": trace[0], "end": trace[-1], "trace": trace})
    log.append(f"refined {len(picks)} plates by pattern search in {time.time() - t1:.1f}s")
    out = sorted(alts.values(), key=lambda a: -a.fitness)
    for i, a in enumerate(out, 1):
        a.id = f"A{i:02d}"
        a.criteria = criteria(a, cfg)
    log.append(f"{len(out)} distinct layouts in {time.time() - t0:.1f}s, no LLM calls; ran on {LAST_MODE['mode']}"
               + (f" ({LAST_MODE['reason']})" if LAST_MODE["reason"] else ""))
    return {"alternatives": out, "log": log, "traces": traces, "evaluations": n_eval,
            "seconds": round(time.time() - t0, 1), "config": cfg}


def criteria(a: Alternative, cfg: ExploreConfig) -> dict:
    """Decision criteria for one layout (direction in decide.CRITERIA)."""
    m = a.metrics
    levels = a.floors + 1
    gross = a.plate.footprint_sf() * levels
    keys = max(1, m["keys_placed"])
    return {
        "fitness": m["fitness"],
        "keys": m["keys_placed"],
        "program_placed": m["rooms_placed"] / max(1, m["rooms_total"]),
        "daylight": m["terms"]["daylight"],
        "corridor_ratio": m["corridor_ratio"],
        "gross_sf_per_key": gross / keys,
        "facade_ft_per_key": a.plate.facade_ft() * levels / keys,
        "height_ft": elevations(cfg, a.floors)[-1] + cfg.floor_height_ft,
        "dead_end_ft": m["dead_end_ft"],
        "max_travel_ft": m["max_travel_ft"],
        "code_pass": sum(c["status"] == "PASS" for c in m["checks"]) / max(1, len(m["checks"])),
        "compactness": m["terms"]["compactness"],
    }


def design_for(hotel_program, a: Alternative, cfg: ExploreConfig):
    """Rebuild the full Design (geometry) of one alternative for drawing / export."""
    prog, site, inst = _setup(tuple(hotel_program.rows), hotel_program.source, a.plate, a.floors, cfg)
    return planner.decode(site, prog, a.genome, inst)
