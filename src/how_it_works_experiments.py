"""Experiments behind reports/how_it_works/RNGD_how_it_works.md: does the system respond to
its inputs (dynamic) or make things up (hallucinate)?

A. Hotel engine (deterministic): four different programs / settings, each built twice and
   hashed; a sweep of key counts; figures of the resulting plans.
B. LLM agents (grounded or not):
   - intake agent on the real request and on an edited request (different numbers),
   - zoning agent twice, compared with the values printed in the knowledge base,
   - every number in the saved arbiter narrative traced back to computed data.
C. Physics engine: penalty-vs-step history of one relaxation.

Usage: python src/how_it_works_experiments.py [--no-llm]
Output: reports/how_it_works/experiments.json and reports/how_it_works/fig_*.png
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rngd.hotel import details, drawings, layout  # noqa: E402
from rngd.hotel.program import Program, default_program  # noqa: E402

OUT = ROOT / "reports" / "how_it_works"
OUT.mkdir(parents=True, exist_ok=True)
SITE = {"envelope_w": 170, "envelope_d": 265}
results: dict = {}


# ------------------------------------------------------------ A. hotel ---
def scaled(p: Program, factor: float, name: str) -> Program:
    rows = [(n, max(1, round(c * factor)) if cat == "guest" else c,
             (a / c) * max(1, round(c * factor)) if cat == "guest" and c else a, cat) for n, c, a, cat in p.rows]
    return Program(rows=rows, source=name)


def select_service() -> Program:
    """A different, smaller hotel program (made up for the test, not RNGD data)."""
    return Program(rows=[
        ("King", 44, 44 * 330, "guest"), ("QQ", 28, 28 * 350, "guest"), ("ADA King", 3, 3 * 420, "guest"),
        ("ADA QQ", 1, 440, "guest"), ("Lobby", 1, 1800, "shared"), ("Breakfast", 1, 900, "shared"),
        ("Fitness", 1, 500, "shared"), ("BOH / Laundry", 1, 1200, "business"), ("Ice", 4, 200, "boh"),
        ("Stair", 10, 1500, "boh"), ("Corridor", 1, 1100, "shared")],
        source="Test program: 76-key select-service hotel (synthetic)")


def fingerprint(sch) -> str:
    rooms = [(lv.name, r.label, round(r.x, 3), round(r.y, 3), round(r.w, 3), round(r.d, 3))
             for lv in sch.levels for r in lv.rooms]
    return hashlib.sha256(json.dumps([rooms, [c["status"] for c in sch.checks]]).encode()).hexdigest()[:16]


def summarize(sch) -> dict:
    p = sch.program
    fails = [c["check"] for c in sch.checks if c["status"] == "FAIL"]
    ada = next(c for c in sch.checks if c["rule_id"] == "ADA-224.2")
    return {"keys": p.keys, "net_sf": round(p.net_sf), "levels": len(sch.levels),
            "keys_per_floor": [lv.keys_placed for lv in sch.levels[1:]],
            "tower_len_ft": round(sch.levels[1].length_ft, 1), "podium_len_ft": round(sch.levels[0].length_ft, 1),
            "gross_drawn_sf": round(sum(lv.gross_sf for lv in sch.levels)), "height_ft": sch.height_ft,
            "dead_end_ft": round(sch.dead_end_ft, 1), "travel_ft": round(sch.max_travel_ft, 1),
            "ada_required": ada["required"], "ada_provided": ada["provided"],
            "pass": sum(c["status"] == "PASS" for c in sch.checks), "fail": fails}


def hotel_experiments():
    base = default_program()
    cases = [
        ("A1 default L+L program, 5 guest floors", base, {}),
        ("A2 same program, 4 guest floors", base, {"upper_levels": 4}),
        ("A3 different program: 76-key select-service", select_service(), {"upper_levels": 4}),
        ("A4 L+L program scaled to ~160 keys", scaled(base, 1.37, "L+L scaled x1.37"), {}),
    ]
    rows, schs = [], []
    for name, prog, lk in cases:
        s1 = layout.build(prog, SITE, **lk)
        s2 = layout.build(prog, SITE, **lk)
        row = {"case": name, **summarize(s1), "hash_run1": fingerprint(s1), "hash_run2": fingerprint(s2)}
        row["identical_reruns"] = row["hash_run1"] == row["hash_run2"]
        rows.append(row)
        schs.append((name, s1))
    results["hotel_cases"] = rows

    # Figure: Level 2 of each case at the same scale
    fig, axes = plt.subplots(len(schs), 1, figsize=(15, 3.3 * len(schs)))
    gx_max = max(max(lv.length_ft for lv in s.levels) for _, s in schs)
    for ax, (name, s) in zip(axes, schs):
        drawings.draw_level(ax, s, s.levels[1], drawings.grid_lines(s), show_grid=False, show_dims=False, fs=3.5)
        ax.set_xlim(-5, gx_max + 5)
        ax.set_ylim(-5, layout.WING_W_FT + 5)
        ax.set_aspect("equal")
        ax.axis("off")
        sm = summarize(s)
        ax.set_title(f"{name}: Level 2 has {s.levels[1].keys_placed} keys, wing {sm['tower_len_ft']:.0f} ft; "
                     f"{sm['keys']} keys total, {sm['pass']}/{len(s.checks)} checks pass", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hotel_inputs_level2.png", dpi=130)
    plt.close(fig)

    # Sweep: keys -> geometry and checks
    sweep = []
    for f in [0.35, 0.5, 0.65, 0.8, 1.0, 1.2, 1.4, 1.6, 1.75]:
        for floors in (4, 5, 6):
            prog = scaled(base, f, f"x{f}")
            s = layout.build(prog, SITE, upper_levels=floors)
            sm = summarize(s)
            sweep.append({"factor": f, "floors": floors, "keys": sm["keys"], "tower_len_ft": sm["tower_len_ft"],
                          "gross_drawn_sf": sm["gross_drawn_sf"], "fits": "Fits buildable envelope" not in sm["fail"],
                          "stories_ok": "Stories above grade" not in sm["fail"], "travel_ft": sm["travel_ft"]})
    results["hotel_sweep"] = sweep
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for floors, color in ((4, "#4D7798"), (5, "#9B7FB0"), (6, "#E8B85C")):
        pts = [r for r in sweep if r["floors"] == floors]
        a1.plot([r["keys"] for r in pts], [r["tower_len_ft"] for r in pts], "o-", color=color, label=f"{floors} guest floors")
        a2.plot([r["keys"] for r in pts], [r["travel_ft"] for r in pts], "o-", color=color, label=f"{floors} guest floors")
        bad = [r for r in pts if not r["fits"]]
        a1.plot([r["keys"] for r in bad], [r["tower_len_ft"] for r in bad], "x", color="#B22222", ms=10)
    a1.axhline(SITE["envelope_d"], color="#B22222", ls="--", lw=1)
    a1.text(45, SITE["envelope_d"] + 5, "envelope depth 265 ft", color="#B22222", fontsize=8)
    a1.set_xlabel("keys in the program")
    a1.set_ylabel("guest-floor wing length (ft)")
    a1.set_title("Wing length grows with keys, shrinks with floors\n(x = Level 1 does not fit the envelope)", fontsize=10)
    a2.axhline(250, color="#B22222", ls="--", lw=1)
    a2.text(45, 243, "IBC-R1-023 limit 250 ft", color="#B22222", fontsize=8, va="top")
    a2.set_xlabel("keys in the program")
    a2.set_ylabel("exit access travel (ft)")
    a2.set_title("Exit travel = half the stair-to-stair run + room depth", fontsize=10)
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hotel_sweep.png", dpi=130)
    plt.close(fig)

    # Figure: one guest room's local geometry (u, v) and its transform into a plan
    fig, ax = plt.subplots(figsize=(4.2, 6))
    w = 343 / layout.ROOM_DEPTH_FT
    g = drawings.unit_geometry("King", w)
    drawings.draw_unit(ax, g, lambda u, v: (u, v), lw_scale=1.5)
    ax.annotate("", xy=(w + 2, 0), xytext=(-1, 0), arrowprops=dict(arrowstyle="->", color="#B22222"))
    ax.annotate("", xy=(-1, layout.ROOM_DEPTH_FT + 2), xytext=(-1, 0), arrowprops=dict(arrowstyle="->", color="#B22222"))
    ax.text(w + 2.2, 0, "u (along corridor)", color="#B22222", fontsize=8, va="center")
    ax.text(-1, layout.ROOM_DEPTH_FT + 2.5, "v (to window)", color="#B22222", fontsize=8, ha="center")
    ax.text(w / 2, -2.2, "corridor side, v = 0", ha="center", fontsize=8)
    ax.set_xlim(-4, w + 9)
    ax.set_ylim(-4, layout.ROOM_DEPTH_FT + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"King room in local coordinates\n{w:.1f} ft x 26 ft = 343 SF", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_unit_local_coords.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------- C. physics ---
def physics_experiment():
    from rngd.simulation.physics_engine import SiteSimulation, build_site_bodies
    from rngd.simulation.renderer import render_site_plan

    env = {"x_min_ft": 10.0, "x_max_ft": 180.0, "y_min_ft": 15.0, "y_max_ft": 280.0}
    runs = []
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, fw, fh, stalls in [("P1 120x120 ft building, 60 stalls asked", 120, 120, 60),
                                  ("P2 128x220 ft building (Elm St.), 150 stalls asked", 127.5, 220, 150),
                                  ("P3 190 ft wide building in a 170 ft envelope", 190, 200, 40)]:
        bodies = build_site_bodies(envelope=env, footprint_w=fw, footprint_h=fh, n_parking_stalls=stalls)
        sim = SiteSimulation(env, bodies)
        r = sim.run()
        placed = sum(b.kind == "parking_stall" for b in bodies)
        runs.append({"case": label, "footprint": [fw, fh], "stalls_requested": stalls, "stalls_placed": placed,
                     "steps": r["steps_run"], "final_penalty": r["max_overlap_penalty"], "converged": r["converged"]})
        ax.semilogy(range(1, len(r["history"]) + 1), [max(h, 1e-3) for h in r["history"]], "o-", ms=3, label=f"{label}: {r['steps_run']} steps")
    ax.axhline(0.5, color="#B22222", ls="--", lw=1)
    ax.text(2, 0.55, "convergence tolerance 0.5", color="#B22222", fontsize=8)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("total penalty (ft, log scale)")
    ax.set_title("Physics relaxation: penalty per step", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_physics_convergence.png", dpi=130)
    plt.close(fig)
    results["physics"] = runs


# -------------------------------------------------------------- B. LLM ---
ZONING_TRUTH = {"max_height_ft": 65, "max_stories": 5, "max_far": 2.5, "max_lot_coverage_pct": 60,
                "front_setback_ft": 15, "side_setback_ft": 10, "rear_setback_ft": 20, "parking_ratio_per_unit": 1.25,
                "fire_lane_clear_width_ft": 20}


def llm_experiments():
    from rngd import config
    from rngd.agents import intake_agent, zoning_agent
    from rngd.vectorstore import get_vectorstore

    rfp = (config.SAMPLE_PROJECTS_DIR / "project_001" / "customer_rfp.txt").read_text(encoding="utf-8")
    edited = (rfp.replace("4-story, 120-unit", "6-story, 84-unit").replace("120 total\nunits", "84 total\nunits")
              .replace("roughly 20% studios, 45% one-bedrooms, and 35% two-bedrooms",
                       "roughly 10% studios, 50% one-bedrooms, and 40% two-bedrooms")
              .replace("approximately $28,000,000", "approximately $19,500,000"))
    intake = []
    for label, text in [("original request, run 1", rfp), ("original request, run 2", rfp), ("edited request", edited)]:
        prog, ms = intake_agent.run(text)
        intake.append({"input": label, "units": prog.unit_count_total, "floors": prog.floor_count,
                       "mix": prog.unit_mix_pct, "budget": prog.budget_usd, "parking": prog.parking_preference,
                       "hard": prog.hard_requirements, "latency_s": round(ms / 1000, 1)})
    results["intake"] = intake

    vs = get_vectorstore()
    site_conditions = json.loads((config.SAMPLE_PROJECTS_DIR / "project_001" / "site_conditions.json").read_text())
    zoning = []
    for run in (1, 2):
        site, ms = zoning_agent.run(site_conditions, vs)
        z = site.zoning
        got = {"max_height_ft": z.max_height_ft, "max_stories": z.max_stories, "max_far": z.max_far,
               "max_lot_coverage_pct": z.max_lot_coverage_pct, "front_setback_ft": site.setbacks_ft.get("front"),
               "side_setback_ft": site.setbacks_ft.get("side_east"), "rear_setback_ft": site.setbacks_ft.get("rear"),
               "parking_ratio_per_unit": z.parking_ratio_per_unit, "fire_lane_clear_width_ft": z.fire_lane_clear_width_ft}
        match = {k: (got[k] is not None and abs(float(got[k]) - v) < 1e-6) for k, v in ZONING_TRUTH.items()}
        zoning.append({"run": run, "extracted": got, "matches_kb": match, "all_match": all(match.values()),
                       "envelope": [site.buildable_envelope.width_ft, site.buildable_envelope.depth_ft],
                       "citations": site.citations[:4], "latency_s": round(ms / 1000, 1)})
    results["zoning"] = zoning
    results["retrieval_backend"] = vs.backend_name
    results["kb_chunks"] = {"total": len(vs.chunks), "real": vs.real_chunk_count}
    hits = vs.retrieve("dead-end corridor maximum length sprinklered", 3)
    results["retrieval_example"] = [{"source": h["source"], "heading": h["heading"], "score": round(h["score"], 3)}
                                    for h in hits]


def narrative_check():
    """Trace every number in the saved arbiter narrative back to the computed pipeline data."""
    f = ROOT / "reports" / "walkthrough" / "screens" / "pipeline_state.json"
    if not f.exists():
        f = ROOT / "outputs" / "project_001" / "project_state.json"
    if not f.exists():
        return
    st = json.loads(f.read_text(encoding="utf-8"))
    narrative = st["final_decision"]["narrative"]
    data = json.dumps({k: st[k] for k in ("program", "site", "baap", "conflicts", "cv_metrics_history",
                                           "simulation_history", "critic_history")})
    data_nums = {float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", data)}
    found = []
    for tok in sorted(set(re.findall(r"\d[\d,]*\.?\d*", narrative)), key=lambda t: float(t.replace(",", "").rstrip("."))):
        v = float(tok.replace(",", "").rstrip("."))
        derived = any(abs(v - (a - b)) < 0.51 for a in data_nums for b in data_nums if a > b and a - b < 10 * max(v, 1)) \
            if v not in data_nums else False
        found.append({"number": tok.rstrip("."), "in_computed_data": v in data_nums or any(abs(v - d) < 0.051 for d in data_nums),
                      "is_difference_of_two_computed_values": derived})
    results["arbiter_numbers"] = {"source_file": f.name, "decision": st["final_decision"]["decision"],
                                  "numbers": found}


if __name__ == "__main__":
    hotel_experiments()
    physics_experiment()
    narrative_check()
    if "--no-llm" not in sys.argv:
        llm_experiments()
    prev = OUT / "experiments.json"
    if "--no-llm" in sys.argv and prev.exists():  # keep the LLM results of the last full run
        old = json.loads(prev.read_text(encoding="utf-8"))
        results = {**{k: v for k, v in old.items() if k in ("intake", "zoning", "retrieval_backend", "kb_chunks",
                                                             "retrieval_example")}, **results}
    prev.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: (v if k != "hotel_sweep" else f"{len(v)} points") for k, v in results.items()}, indent=1,
                     default=str)[:6000])
