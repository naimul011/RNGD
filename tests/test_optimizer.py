from pathlib import Path

import numpy as np
import pytest

from rngd.optimize import baseline, export, metrics, planner, review
from rngd.optimize.program import load_program
from rngd.optimize.report import fact_check
from rngd.optimize.search import Optimizer, SearchConfig
from rngd.optimize.site import load_site

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "QA"
EXT = ROOT / "data" / "optimizer" / "hotel_program_extended.json"
pytestmark = pytest.mark.skipif(not (QA / "fp1_multi.json").exists(), reason="QA inputs not present")


@pytest.fixture(scope="module")
def fp1():
    return load_site(QA / "fp1_multi.json")


def test_site_and_programs_load(fp1):
    assert fp1.shape == (151, 100) and fp1.n_levels == 3 and len(fp1.cores) == 2
    assert len(load_program(QA / "rooms_hotel.json").instances(fp1.level_names)) == 72
    ext = load_program(EXT)
    assert ext.keys_requested(fp1.level_names) == 48
    # sized to fit: program area below the placeable area of the three levels
    assert ext.total_area_sf(fp1.level_names) < 3 * fp1.placeable_sf()


@pytest.mark.parametrize("i,shared", [(1, 982), (2, 737), (3, 913)])
def test_reference_designs_reconstruct_exactly(i, shared):
    site = load_site(QA / f"fp{i}_multi.json")
    d, info = baseline.load_reference(QA / f"floorplan{i}.json", site, load_program(QA / "rooms_hotel.json"))
    m = metrics.evaluate(d)
    assert m["rooms_placed"] == info["reported_rooms_placed"]
    assert m["shared_wall_ft"] == shared  # matches the value the reference optimizer reported


def _valid(design):
    site = design.site
    for l in range(site.n_levels):
        occ = np.zeros(site.shape, int)
        for p in design.placements:
            if p.level != l:
                continue
            blk = (slice(p.r0, p.r0 + p.h), slice(p.c0, p.c0 + p.w))
            assert site.placeable[l][blk].all(), p.name
            assert not design.corridor[blk].any(), p.name
            occ[blk] += 1
            ring = (slice(max(p.r0 - 2, 0), p.r0 + p.h + 2), slice(max(p.c0 - 2, 0), p.c0 + p.w + 2))
            assert design.corridor[ring].any(), f"{p.name} does not open onto the corridor"
        assert occ.max() <= 1


def test_decoder_is_valid_and_deterministic(fp1):
    prog = load_program(EXT)
    inst = prog.instances(fp1.level_names)
    g = planner.facade_seeds(fp1, prog, len(inst), np.random.default_rng(1))[0]
    d1, d2 = planner.decode(fp1, prog, g, inst), planner.decode(fp1, prog, g, inst)
    _valid(d1)
    assert d1.cores_reached == 2
    assert [(p.name, p.r0, p.c0, p.level) for p in d1.placements] == [(p.name, p.r0, p.c0, p.level) for p in d2.placements]


def test_search_improves_and_exports(fp1):
    prog = load_program(QA / "rooms_hotel.json")
    # fixed weights (no reviews): elitism + accept-if-better local search never lose the best design
    res = Optimizer(fp1, prog, SearchConfig(method="ga", population=6, generations=3, use_llm=False,
                                            llm_checkpoints=0, local_search_steps=1)).run()
    best = [h["best_fitness"] for h in res["history"]]
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(best, best[1:]))
    _valid(res["best"].design)
    # with a review checkpoint the reviewer runs and the weights stay normalised
    res2 = Optimizer(fp1, prog, SearchConfig(method="ga", population=6, generations=2, use_llm=False,
                                             llm_checkpoints=1, local_search_steps=0)).run()
    assert res2["reviews"] and res2["reviews"][0]["engine"] == "heuristic"
    assert abs(sum(res2["weights"].values()) - 1.0) < 1e-6
    out = export.design_to_dict(res["best"].design, res["best"].metrics)
    for key in ("corridor_cells", "placements", "rooms_placed", "unplaced_rooms", "code_checks"):
        assert key in out


def test_heuristic_review_and_weight_tuning(fp1):
    prog = load_program(EXT)
    inst = prog.instances(fp1.level_names)
    d = planner.decode(fp1, prog, planner.facade_seeds(fp1, prog, len(inst), np.random.default_rng(0))[0], inst)
    m = metrics.evaluate(d)
    r = review.heuristic_review([("A", m)])
    assert 0 <= r["reviews"][0]["score"] <= 10
    assert all(0.7 <= v <= 1.3 for v in r["weight_multipliers"].values())


def test_fact_check_flags_invented_numbers():
    data = {"optimized": {"rooms_placed": 50, "rooms_total": 72, "keys_placed": 48, "keys_requested": 69,
                          "dead_end_ft": 31.4, "max_travel_ft": 120.0, "shared_wall_ft": 800, "fitness": 0.87}}
    res = {c["number"]: c["traced"] for c in fact_check({"s": "We placed 50 of 72 rooms; dead end 31.4 ft; cost $4,250,000."}, data)}
    assert res["50"] and res["72"] and res["31.4"]
    assert not res["4,250,000"]


def test_layout_quality_on_reference_program(fp1):
    """Architectural decoder: straight spines, guest rooms keep their size and get windows."""
    prog = load_program(QA / "rooms_hotel.json")
    inst = prog.instances(fp1.level_names)
    best = max((planner.decode(fp1, prog, g, inst) for g in planner.facade_seeds(fp1, prog, len(inst), np.random.default_rng(0))),
               key=lambda d: metrics.evaluate(d)["fitness"])
    m = metrics.evaluate(best)
    assert m["rooms_placed"] >= 61                      # at least the reference GA's 61 on this site
    assert m["exterior_fraction"] >= 0.9
    assert all(p.frontage_ft >= 0.85 * 12 and p.depth_ft >= 0.85 * 25 for p in best.placements if p.type_name == "King")
    assert m["dead_end_ft"] <= 50


def test_systematic_search_is_deterministic_and_monotonic(fp1):
    prog = load_program(QA / "rooms_hotel.json")
    cfg = dict(population=6, generations=4)
    r1 = Optimizer(fp1, prog, SearchConfig(**cfg)).run()
    r2 = Optimizer(fp1, prog, SearchConfig(**cfg)).run()
    assert r1["best"].metrics["fitness"] == r2["best"].metrics["fitness"]
    best = [h["best"] for h in r1["history"]]
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(best, best[1:]))
    assert not r1["reviews"]                      # no LLM in the loop
    _valid(r1["best"].design)
