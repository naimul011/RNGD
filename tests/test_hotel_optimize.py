"""Hotel Schematic optimizer: generative plates, deterministic exploration, decision ranking."""
import numpy as np

from rngd.hotel.program import default_program
from rngd.optimize import decide, explore, planner, typology as T


def _small_cfg(**kw):
    base = dict(shapes=("Bar", "L"), schemes=("ends", "knuckle"), floors_mode="fixed", guest_floors=4,
                seeds_per_plate=3, refine_top=2, refine_steps=4, workers=1)
    base.update(kw)
    return explore.ExploreConfig(**base)


def test_plates_fit_envelope_and_have_cores():
    hp = default_program()
    names = T.level_names(4)
    prog = T.program_from_hotel(hp, names)
    assert prog.keys_requested(names) == hp.keys
    assert not any(rt.name.lower().startswith(("corridor", "stair")) for rt in prog.rooms)
    plates = T.generate_plates(prog, names, hp.keys, (170, 265))
    assert {p.shape for p in plates} == set(T.SHAPES)
    for p in plates:
        W, H = p.size
        assert (W <= 170 and H <= 265) or (W <= 265 and H <= 170), p.name
        assert len(p.cores) >= 2 and sum(k == "main" for *_, k in p.cores) == 1
        site = T.plate_site(p, names, explore.elevations(_small_cfg(), 4))
        assert len(site.cores) >= 2, p.name


def test_elevators_follow_keys():
    assert T.elevators_for(60) == 2 and T.elevators_for(200) == 3
    assert T.main_core_size(200)[0] > T.main_core_size(100)[0]


def test_explore_is_deterministic_and_places_the_keys():
    hp = default_program()
    r1 = explore.explore(hp, _small_cfg())
    r2 = explore.explore(hp, _small_cfg())
    a1, a2 = r1["alternatives"], r2["alternatives"]
    assert [(a.plate.name, a.fitness) for a in a1] == [(a.plate.name, a.fitness) for a in a2]
    best = a1[0]
    assert best.metrics["keys_placed"] >= 0.95 * hp.keys
    assert best.metrics["exterior_fraction"] == 1.0          # every guest room has a window
    assert any(c["rule_id"] == "IBC-R1-007" for c in best.metrics["checks"])
    d = explore.design_for(hp, best, r1["config"])
    assert len(d.placements) == best.metrics["rooms_placed"]


def test_decision_ranking_pareto_and_topsis():
    hp = default_program()
    alts = explore.explore(hp, _small_cfg())["alternatives"]
    rows = decide.rank(alts, decide.PRESETS["Balanced"])
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert all(0 <= r["score"] <= 1 for r in rows)
    assert any(r["pareto"] for r in rows)
    # a Pareto layout is never dominated by another one
    front = [r for r in rows if r["pareto"]]
    for f in front:
        for o in rows:
            better = all((o[k] - f[k]) * s >= 0 for k, (s, _, _) in decide.CRITERIA.items())
            strictly = any((o[k] - f[k]) * s > 0 for k, (s, _, _) in decide.CRITERIA.items())
            assert not (better and strictly)
    most_keys = decide.rank(alts, decide.PRESETS["Most keys"])[0]
    assert most_keys["keys"] == max(r["keys"] for r in rows)
    ok, msg = decide.jev_status()
    assert isinstance(msg, str)
