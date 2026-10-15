"""Memetic genetic algorithm with an LLM/RAG reviewer in the loop.

Why this design (and not a plain GA or pure simulated annealing):
  * The genome mixes continuous corridor geometry and a room-order permutation, and the
    decoder (engine.decode) is a strong greedy constructor. A GA explores corridor
    topologies broadly; a short local search (hill climbing on the elite) then polishes the
    best layout cheaply. This memetic combination is the standard choice for
    decoder-based layout problems.
  * The LLM reviewer is expensive, so it runs only at a few checkpoints on a handful of
    distinct elite designs. Its score is blended into those designs' fitness and its weight
    multipliers re-weight the objectives for everyone (cached terms are re-scored without
    re-decoding).

Operators: tournament selection (k=3); segment-wise uniform crossover for corridor genes and
uniform crossover for order keys; Gaussian mutation with a decaying sigma, segment
switch-on/off, and key swaps; elitism; immigrants when diversity collapses.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import metrics, review
from . import planner as engine  # architectural decoder (spines, modular bays, stacking)
from .program import Program
from .site import Site


@dataclass
class SearchConfig:
    population: int = 24
    generations: int = 30
    elite: int = 2
    tournament: int = 3
    crossover_rate: float = 0.9
    sigma_start: float = 0.15
    sigma_end: float = 0.03
    segment_toggle_rate: float = 0.08
    key_swap_rate: float = 0.15
    local_search_steps: int = 4
    immigrants: int = 2
    seed: int = 7
    method: str = "systematic"  # "systematic" (enumeration + pattern search) or "ga" (memetic GA)
    use_llm: bool = False       # deterministic by default (no LLM in the fitness)
    llm_checkpoints: int = 0
    llm_candidates: int = 4
    llm_lambda: float = 0.25
    weights: dict = field(default_factory=lambda: dict(metrics.DEFAULT_WEIGHTS))


@dataclass
class Individual:
    genome: np.ndarray
    metrics: dict
    fitness: float
    llm_score: float | None = None
    design: engine.Design | None = None

    @property
    def total(self) -> float:
        return self.fitness


def _blend(ind: Individual, lam: float) -> float:
    return ind.fitness if ind.llm_score is None else (1 - lam) * ind.fitness + lam * ind.llm_score / 10.0


class Optimizer:
    def __init__(self, site: Site, program: Program, config: SearchConfig, vs=None):
        self.site, self.program, self.cfg = site, program, config
        self.instances = program.instances(site.level_names)
        self.n = engine.genome_length(len(self.instances))
        self.rng = np.random.default_rng(config.seed)
        self.weights = dict(config.weights)
        self.history: list[dict] = []
        self.reviews: list[dict] = []
        self.evaluations = 0
        self.vs = vs
        self.cache: dict = {}

    # ---------------------------------------------------------------- evaluation
    def _eval(self, g: np.ndarray) -> Individual:
        g = np.clip(g, 0.0, 1.0)
        key = np.round(g, 4).tobytes()
        if key in self.cache:
            m = self.cache[key]
        else:
            d = engine.decode(self.site, self.program, g, self.instances)
            m = metrics.evaluate(d, self.weights)
            self.cache[key] = m
            self.evaluations += 1
        f = metrics.fitness_from_terms(m["terms"], self.weights, m["cores_reached"], m["cores_total"])
        return Individual(g, m, f)

    def _rescore(self, pop: list[Individual]):
        for ind in pop:
            ind.fitness = metrics.fitness_from_terms(ind.metrics["terms"], self.weights, ind.metrics["cores_reached"],
                                                     ind.metrics["cores_total"])

    # ---------------------------------------------------------------- operators
    def _tournament(self, pop):
        idx = self.rng.integers(0, len(pop), self.cfg.tournament)
        return max((pop[i] for i in idx), key=lambda x: _blend(x, self.cfg.llm_lambda))

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.cfg.crossover_rate:
            return a.copy()
        child = a.copy()
        nseg = engine.N_SEG
        take = self.rng.random(nseg) < 0.5
        seg_a, seg_b = a[:nseg * 4].reshape(nseg, 4), b[:nseg * 4].reshape(nseg, 4)
        child[:nseg * 4] = np.where(take[:, None], seg_b, seg_a).ravel()
        mask = self.rng.random(self.n - nseg * 4) < 0.5
        child[nseg * 4:] = np.where(mask, b[nseg * 4:], a[nseg * 4:])
        return child

    def _mutate(self, g: np.ndarray, sigma: float) -> np.ndarray:
        g = g.copy()
        nseg = engine.N_SEG
        seg = g[:nseg * 4].reshape(nseg, 4)
        seg += self.rng.normal(0, sigma, seg.shape) * (self.rng.random(seg.shape) < 0.3)
        flip = self.rng.random(nseg) < self.cfg.segment_toggle_rate
        seg[flip, 3] = np.where(seg[flip, 3] < 0.12, 0.4 + 0.6 * self.rng.random(flip.sum()), 0.05)
        g[:nseg * 4] = seg.ravel()
        keys = g[nseg * 4:]
        if len(keys) > 1 and self.rng.random() < self.cfg.key_swap_rate * 3:
            for _ in range(max(1, int(len(keys) * self.cfg.key_swap_rate / 3))):
                i, j = self.rng.integers(0, len(keys), 2)
                keys[i], keys[j] = keys[j], keys[i]
        keys += self.rng.normal(0, sigma, keys.shape) * (self.rng.random(keys.shape) < 0.1)
        return np.clip(g, 0.0, 1.0)

    def _local_search(self, ind: Individual, sigma: float) -> Individual:
        best = ind
        for _ in range(self.cfg.local_search_steps):
            cand = self._eval(self._mutate(best.genome, sigma * 0.5))
            if cand.fitness > best.fitness:
                cand.llm_score = None
                best = cand
        return best

    # ---------------------------------------------------------------- review
    def _review(self, pop: list[Individual], gen: int, on_review=None):
        uniq, seen = [], set()
        for ind in sorted(pop, key=lambda x: -x.fitness):
            sig = (ind.metrics["rooms_placed"], round(ind.fitness, 3))
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append(ind)
            if len(uniq) >= self.cfg.llm_candidates:
                break
        cands = [(f"G{gen}-{i + 1}", ind.metrics) for i, ind in enumerate(uniq)]
        res = review.review(cands, use_llm=self.cfg.use_llm, vs=self.vs)
        scores = {r["id"]: r["score"] for r in res["reviews"]}
        for (cid, _), ind in zip(cands, uniq):
            if cid in scores:
                ind.llm_score = scores[cid]
        before = dict(self.weights)
        for k, mult in res.get("weight_multipliers", {}).items():
            self.weights[k] = self.weights[k] * mult
        tot = sum(self.weights.values())
        self.weights = {k: v / tot for k, v in self.weights.items()}
        self._rescore(pop)
        entry = {"gen": gen, "candidates": [{"id": cid, "fitness": ind.fitness, "rooms_placed": ind.metrics["rooms_placed"],
                                             "llm_score": ind.llm_score} for (cid, _), ind in zip(cands, uniq)],
                 "weights_before": before, "weights_after": dict(self.weights), **res}
        self.reviews.append(entry)
        if on_review:
            on_review(entry)

    # ---------------------------------------------------------------- run
    def run(self, on_generation=None, on_review=None) -> dict:
        if self.cfg.method == "systematic":
            return self._run_systematic(on_generation)
        cfg = self.cfg
        t0 = time.time()
        seeds = (engine.facade_seeds(self.site, self.program, len(self.instances), self.rng)
                 + engine.spine_genomes(len(self.instances), self.rng))
        pop = [self._eval(g) for g in seeds[:cfg.population]]
        while len(pop) < cfg.population:
            pop.append(self._eval(engine.random_genome(self.rng, len(self.instances))))
        checkpoints = set()
        if cfg.llm_checkpoints > 0 and cfg.generations > 0:
            checkpoints = {max(1, round(cfg.generations * (i + 1) / cfg.llm_checkpoints)) for i in range(cfg.llm_checkpoints)}
        for gen in range(1, cfg.generations + 1):
            sigma = cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * (gen - 1) / max(1, cfg.generations - 1)
            pop.sort(key=lambda x: -_blend(x, cfg.llm_lambda))
            nxt = pop[:cfg.elite]
            nxt[0] = self._local_search(nxt[0], sigma)
            fits = np.array([p.fitness for p in pop])
            immigrants = cfg.immigrants if fits.std() < 0.01 else 0
            for _ in range(immigrants):
                nxt.append(self._eval(engine.random_genome(self.rng, len(self.instances))))
            while len(nxt) < cfg.population:
                child = self._crossover(self._tournament(pop).genome, self._tournament(pop).genome)
                nxt.append(self._eval(self._mutate(child, sigma)))
            pop = nxt
            if gen in checkpoints:
                self._review(pop, gen, on_review)
            pop.sort(key=lambda x: -_blend(x, cfg.llm_lambda))
            best = pop[0]
            rec = {"gen": gen, "best": round(_blend(best, cfg.llm_lambda), 4), "best_fitness": round(best.fitness, 4),
                   "mean": round(float(np.mean([p.fitness for p in pop])), 4), "best_placed": best.metrics["rooms_placed"],
                   "best_keys": best.metrics["keys_placed"], "evaluations": self.evaluations,
                   "seconds": round(time.time() - t0, 1)}
            self.history.append(rec)
            if on_generation:
                on_generation(rec, best)
        pop.sort(key=lambda x: -_blend(x, cfg.llm_lambda))
        best = pop[0]
        best.design = engine.decode(self.site, self.program, best.genome, self.instances)
        best.metrics = metrics.evaluate(best.design, self.weights)
        return {"best": best, "history": self.history, "reviews": self.reviews, "weights": self.weights,
                "evaluations": self.evaluations, "seconds": round(time.time() - t0, 1), "config": asdict(cfg)}

    # ---------------------------------------------------------------- systematic (default)
    def _run_systematic(self, on_generation=None) -> dict:
        """Deterministic alternative to the GA (10/08 meeting: crossover of half-plans rarely makes
        sense). Stage 1 enumerates corridor-spine combinations (every subset of up to 4 of the 7
        best facade spines, plus all spines) with rooms in priority order; stage 2 refines the 3
        best distinct layouts by first-improvement pattern search (spine toggles, trims, shifts,
        room-order nudges); stage 3 spends the rest of the budget on iterated local search (a
        two-move kick of the best layout, a new descent, kept only if better). Same inputs -> same
        result. `generations` sets the number of progress steps and population x generations the
        total evaluation budget (the same as the GA's)."""
        from .explore import _neighbours
        cfg = self.cfg
        t0 = time.time()
        n_inst = len(self.instances)
        P = engine.params_for(self.site, self.program)
        m = len(P.candidates)
        top = list(range(min(m, 7)))
        combos = [list(range(m))] + [list(c) for k in range(1, min(4, len(top)) + 1)
                                     for c in itertools.combinations(top, k)]
        genomes = []
        for c in combos:
            g = engine._seed(self.rng, n_inst, c)
            g[engine.N_SEG * 4:] = 0.5      # rooms in priority order
            genomes.append(g)
        G = max(2, cfg.generations)
        stage1 = max(1, G // 2)
        pop: list[Individual] = []
        gen = 0

        def record(batch):
            nonlocal gen
            gen += 1
            best = max(pop, key=lambda x: x.fitness)
            rec = {"gen": gen, "best": round(best.fitness, 4), "best_fitness": round(best.fitness, 4),
                   "mean": round(float(np.mean([b.fitness for b in batch])), 4) if batch else round(best.fitness, 4),
                   "best_placed": best.metrics["rooms_placed"], "best_keys": best.metrics["keys_placed"],
                   "evaluations": self.evaluations, "seconds": round(time.time() - t0, 1)}
            self.history.append(rec)
            if on_generation:
                on_generation(rec, best)

        for idx in np.array_split(np.arange(len(genomes)), stage1):
            batch = [self._eval(genomes[i]) for i in idx]
            pop += batch
            record(batch)
        # stage 2: pattern search from the best distinct layouts
        pop.sort(key=lambda x: -x.fitness)
        starts, seen = [], set()
        for ind in pop:
            sig = (ind.metrics["rooms_placed"], round(ind.fitness, 3))
            if sig not in seen:
                seen.add(sig)
                starts.append(ind)
            if len(starts) >= 3:
                break
        budget = max(20, cfg.population * cfg.generations - len(genomes))   # same total as the GA
        per_step = max(1, budget // max(1, G - stage1))
        used, step, cur = 0, 0, 0
        stuck = [False] * len(starts)
        while used < budget and gen < G and not all(stuck):
            batch = []
            for _ in range(per_step):
                if all(stuck) or used >= budget:
                    break
                i = cur % len(starts)
                cur += 1
                if stuck[i]:
                    continue
                improved = False
                for h in _neighbours(starts[i].genome, n_inst, self.rng, step):
                    cand = self._eval(h)
                    used += 1
                    batch.append(cand)
                    if cand.fitness > starts[i].fitness + 1e-6:
                        starts[i] = cand
                        improved = True
                        break
                    if used >= budget:
                        break
                stuck[i] = not improved
            step += 1
            pop += batch
            record(batch)
        # stage 3 (iterated local search): when every start is a local optimum and budget is left,
        # kick the best layout with two moves at once, descend again by pattern search, and keep the
        # result only if it beats the best
        cur_best = max(pop, key=lambda x: x.fitness)
        batch = []

        def flush(force=False):
            nonlocal batch
            if batch and (force or len(batch) >= per_step):
                pop.extend(batch)
                record(batch)
                batch = []

        while used < budget:
            h = _neighbours(cur_best.genome, n_inst, self.rng, step)[0]
            cand = self._eval(_neighbours(h, n_inst, self.rng, step)[0])
            used += 1
            batch.append(cand)
            improved = True
            while improved and used < budget:
                improved = False
                for nb in _neighbours(cand.genome, n_inst, self.rng, step):
                    c2 = self._eval(nb)
                    used += 1
                    batch.append(c2)
                    if c2.fitness > cand.fitness + 1e-6:
                        cand, improved = c2, True
                        break
                    if used >= budget:
                        break
                flush()
            if cand.fitness > cur_best.fitness + 1e-6:
                cur_best = cand
            step += 1
        flush(force=True)
        best = max(pop, key=lambda x: x.fitness)
        best.design = engine.decode(self.site, self.program, best.genome, self.instances)
        best.metrics = metrics.evaluate(best.design, self.weights)
        return {"best": best, "history": self.history, "reviews": self.reviews, "weights": self.weights,
                "evaluations": self.evaluations, "seconds": round(time.time() - t0, 1), "config": asdict(cfg)}
