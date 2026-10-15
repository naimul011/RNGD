import pytest

from rngd.hotel import layout
from rngd.hotel.program import Program, default_program, load_program_file


def test_default_program_totals():
    p = default_program()
    assert p.keys == 117
    assert p.net_sf == pytest.approx(59350, abs=5)
    # Sum of the per-level gross figures; the source's printed total is 74,197 (2 SF rounding).
    assert p.gross_sf == 74195


def test_net_to_gross_rounds_up_to_hundred():
    n = default_program().net_to_gross(33, 8)
    assert n["required_gross_sf"] % 100 == 0
    assert n["required_gross_sf"] > default_program().net_sf


def test_schematic_places_all_keys():
    s = layout.build(default_program(), {"envelope_w": 170, "envelope_d": 265})
    assert s.unplaced_keys == 0
    assert sum(lv.keys_placed for lv in s.levels[1:]) == 117


def test_six_levels_flagged_against_five_story_rule():
    s = layout.build(default_program(), {"envelope_w": 170, "envelope_d": 265})
    stories = next(c for c in s.checks if c["check"] == "Stories above grade")
    assert stories["status"] == "FAIL"


def test_upload_csv(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("room_type,count,area_sf\nKing,10,350\nLobby,1,2000\n", encoding="utf-8")
    p = load_program_file(f)
    # area_sf is the row total (same convention as the source table)
    assert p.keys == 10 and p.net_sf == 2350
    assert p.rows[1][3] == "shared"  # inferred from name when category column is absent


SITE = {"envelope_w": 170, "envelope_d": 265}


def test_detailed_program_matches_source_schedule():
    p = default_program()
    assert len(p.rows) == 28
    assert p.net_sf == pytest.approx(59350, abs=2)
    assert sum(c for n, c, a, cat in p.rows if n == "Stair") == 12


def test_stairs_and_core_stack_on_every_level():
    s = layout.build(default_program(), SITE)
    stairs = {lv.name: sorted((r.x, r.y) for r in lv.rooms if r.kind == "stair") for lv in s.levels}
    assert len(set(map(tuple, stairs.values()))) == 1  # same positions on all levels
    assert len(next(iter(stairs.values()))) == 2


def test_per_floor_support_rooms_on_each_guest_floor():
    s = layout.build(default_program(), SITE)
    for lv in s.levels[1:]:
        assert {"Ice", "IT", "Jan", "Supplies"} <= {r.label for r in lv.rooms}


def test_two_end_stairs_clear_dead_end_and_fit_envelope():
    s = layout.build(default_program(), SITE)
    status = {c["check"]: c["status"] for c in s.checks}
    assert status["Dead-end corridor (ft)"] == "PASS"
    assert status["Exit separation (ft)"] == "PASS"
    assert status["Fits buildable envelope"] == "PASS"


def test_ground_floor_places_every_non_guest_row():
    p = default_program()
    s = layout.build(p, SITE)
    labels = " ".join(r.label for lv in s.levels for r in lv.rooms)
    for name, count, area, cat in p.rows:
        if cat != "guest" and name not in ("Corridor", "Stair"):
            assert name in labels, name


def test_uploaded_program_draws_a_plan_sheet(tmp_path):
    from rngd.hotel import drawings

    f = tmp_path / "p.csv"
    f.write_text("room_type,count,area_sf\nKing,40,13600\nQQ,20,6900\nLobby,1,2000\nGym,1,800\n", encoding="utf-8")
    s = layout.build(load_program_file(f), SITE)
    assert s.unplaced_keys == 0
    pdf = drawings.plan_sheet(s, s.levels[:2], "A101", "Plans", tmp_path)
    assert pdf.exists() and (tmp_path / "A101.png").exists()


def test_ada_tables_match_2010_standards():
    from rngd.hotel.details import ada_comm_required, ada_required

    assert ada_required(117) == (5, 2, 7)  # Table 224.2, 101-150 rooms
    assert ada_required(60) == (3, 1, 4)
    assert ada_comm_required(117) == 12  # Table 224.4
    s = layout.build(default_program(), SITE)
    ada = next(c for c in s.checks if c["rule_id"] == "ADA-224.2")
    assert ada["status"] == "PASS" and ada["provided"] == "7"


def test_program_detail_and_mix_by_floor():
    from rngd.hotel.details import mix_by_floor, program_detail

    s = layout.build(default_program(), SITE)
    det = program_detail(s)
    assert len(det) == 28 and det["Total area (SF)"].sum() == pytest.approx(59350, abs=2)
    king = det[det["Room type"] == "King"].iloc[0]
    assert king["Beds"] == "1 king" and king["Placed on"] == "every guest floor"
    mix = mix_by_floor(s)
    assert mix.loc["Keys per floor", "Total"] == 117


def test_per_floor_rooms_follow_floor_count():
    s = layout.build(default_program(), SITE, upper_levels=4)
    assert s.unplaced_keys == 0 and len(s.levels) == 5
    ice = [r for lv in s.levels[1:] for r in lv.rooms if r.label == "Ice"]
    assert len(ice) == 4 and sum(r.sf for r in ice) == pytest.approx(297)


def test_interactive_figures_build():
    from rngd.hotel import interactive

    s = layout.build(default_program(), SITE)
    assert len(interactive.plan_figure(s.levels[1]).data) == len(s.levels[1].rooms)
    assert len(interactive.massing_figure(s, explode=10).data) >= 5
