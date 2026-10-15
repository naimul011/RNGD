"""Design Optimizer mode for the Streamlit app.

Inputs in the QA format (site grids fp*_multi.json, room program JSON), every optimizer
setting exposed as a field, a live run (memetic GA + LLM/RAG review), the optimized design,
a comparison with the QA reference design, and an AI-written PDF report.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rngd.optimize import baseline, engine, export, metrics, planner, render, report
from rngd.optimize.program import Program, load_program, program_from_dict, program_to_dict
from rngd.optimize.search import Optimizer, SearchConfig
from rngd.optimize.site import available_sites, load_site

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "QA"
DEFAULT_PROGRAM = ROOT / "data" / "optimizer" / "hotel_program_extended.json"
OUT = ROOT / "reports" / "optimizer" / "app"
CATS = ["guest", "suite", "ada", "service", "public"]
DEF_CFG = SearchConfig()


# ------------------------------------------------------------------ sidebar
def sidebar(ss) -> dict:
    st.subheader("1 · Site (grid)")
    sites = available_sites(QA)
    names = [p.name for p in sites]
    pick = st.selectbox("Site grid", names + ["Upload..."], key="opt_site_pick",
                        help="Multi-level 1 ft occupancy grid (OUTSIDE / STRUCTURE / CORE / PLACEABLE).")
    site_path = sites[names.index(pick)] if pick in names else None
    if pick == "Upload...":
        up = st.file_uploader("Site grid (*_multi.json)", type=["json"], key="opt_site_up")
        if up is not None:
            site_path = Path(tempfile.gettempdir()) / up.name
            site_path.write_bytes(up.getvalue())

    st.subheader("2 · Program")
    src = st.radio("Program", ["RNGD hotel program (L+L based)", "QA program (rooms_hotel.json)", "Upload JSON"],
                   key="opt_prog_src")
    prog_path = DEFAULT_PROGRAM if src.startswith("RNGD") else QA / "rooms_hotel.json"
    if src == "Upload JSON":
        up = st.file_uploader("Program JSON", type=["json"], key="opt_prog_up")
        prog_path = None
        if up is not None:
            prog_path = Path(tempfile.gettempdir()) / up.name
            prog_path.write_bytes(up.getvalue())
    if prog_path and ss.get("opt_prog_loaded") != str(prog_path):
        ss.opt_program = program_to_dict(load_program(prog_path))
        ss.opt_program["_source"] = prog_path.name
        ss.opt_prog_loaded = str(prog_path)
        ss.opt_prog_ver = ss.get("opt_prog_ver", 0) + 1

    st.subheader("3 · Search")
    st.selectbox("Search method", ["systematic", "ga"], key="opt_method",
                 format_func=lambda m: {"systematic": "Systematic + pattern search (deterministic, fast)",
                                        "ga": "Memetic genetic algorithm (seeded)"}[m],
                 help="Systematic: every combination of the best corridor spines, then first-improvement "
                      "refinement of the 3 best. GA settings below apply to the GA only; Generations also sets "
                      "the progress steps and Population x Generations / 2 the refinement budget.")
    c1, c2 = st.columns(2)
    c1.number_input("Population", 6, 200, DEF_CFG.population, 2, key="opt_pop")
    c2.number_input("Generations", 1, 500, DEF_CFG.generations, 1, key="opt_gens")
    c1.number_input("Elite kept", 1, 10, DEF_CFG.elite, key="opt_elite")
    c2.number_input("Tournament size", 2, 8, DEF_CFG.tournament, key="opt_tour")
    c1.number_input("Crossover rate", 0.0, 1.0, DEF_CFG.crossover_rate, 0.05, key="opt_cx")
    c2.number_input("Local-search steps", 0, 30, DEF_CFG.local_search_steps, key="opt_ls")
    c1.number_input("Mutation sigma start", 0.0, 0.5, DEF_CFG.sigma_start, 0.01, key="opt_s0")
    c2.number_input("Mutation sigma end", 0.0, 0.5, DEF_CFG.sigma_end, 0.01, key="opt_s1")
    c1.number_input("Corridor on/off rate", 0.0, 0.5, DEF_CFG.segment_toggle_rate, 0.01, key="opt_tog")
    c2.number_input("Order swap rate", 0.0, 0.5, DEF_CFG.key_swap_rate, 0.01, key="opt_swap")
    c1.number_input("Immigrants", 0, 20, DEF_CFG.immigrants, key="opt_imm")
    c2.number_input("Random seed", 0, 10_000, DEF_CFG.seed, key="opt_seed")

    st.subheader("4 · Design review (optional)")
    st.checkbox("Use the LLM reviewer (RAG) at checkpoints", value=False, key="opt_llm",
                help="Off by default: the fitness is deterministic geometry + code arithmetic. On = slower and "
                     "not repeatable (cloud LLM).")
    c1, c2 = st.columns(2)
    c1.number_input("Review checkpoints", 0, 10, DEF_CFG.llm_checkpoints, key="opt_ckpt")
    c2.number_input("Candidates per review", 1, 8, DEF_CFG.llm_candidates, key="opt_cands")
    st.slider("Weight of the LLM score (lambda)", 0.0, 1.0, DEF_CFG.llm_lambda, 0.05, key="opt_lam",
              help="F_total = (1 - lambda) F + lambda x score/10 for reviewed designs.")

    st.subheader("5 · Geometry & code limits")
    prog = ss.get("opt_program", {})
    c1, c2 = st.columns(2)
    c1.number_input("Corridor width (ft)", 3.67, 12.0, float(prog.get("corridor_width_ft", 5.0)), 0.5, key="opt_corr")
    c2.number_input("Door width (ft)", 2.67, 6.0, float(prog.get("door_width_ft", 3.0)), 0.25, key="opt_door")
    c1.number_input("Window slack (ft)", 0.0, 20.0, float(prog.get("exterior_slack_ft", 6.0)), 1.0, key="opt_slack",
                    help="How far a room's back wall may sit from the exterior and still count as having a window.")
    st.checkbox("Guest rooms must have a window", value=True, key="opt_win_rule",
                help="Hard rule: King/QQ/suite/ADA rooms only in bays on the facade. Off = allowed with a penalty.")
    c2.number_input("Max exit travel (ft)", 50.0, 400.0, float(metrics.code_limits()["travel_ft"]), 10.0, key="opt_travel",
                    help="IBC-R1-023 (RNGD CSV): 250 ft sprinklered.")
    c1.number_input("Max dead end (ft)", 10.0, 150.0, float(metrics.code_limits()["dead_end_ft"]), 5.0, key="opt_dead",
                    help="IBC-R1-027 (RNGD CSV): 50 ft sprinklered R-1.")
    return {"site_path": site_path, "prog_path": prog_path}


# ------------------------------------------------------------------ helpers
def _program(ss) -> Program:
    d = dict(ss.opt_program)
    d.update(corridor_width_ft=ss.opt_corr, door_width_ft=ss.opt_door, exterior_slack_ft=ss.opt_slack)
    return program_from_dict(d, source=d.get("_source", "edited program"))


def _config(ss) -> SearchConfig:
    w = {k: ss.get(f"opt_w_{k}", v) for k, v in metrics.DEFAULT_WEIGHTS.items()}
    return SearchConfig(method=ss.get("opt_method", "systematic"), population=int(ss.opt_pop), generations=int(ss.opt_gens), elite=int(ss.opt_elite),
                        tournament=int(ss.opt_tour), crossover_rate=float(ss.opt_cx), sigma_start=float(ss.opt_s0),
                        sigma_end=float(ss.opt_s1), segment_toggle_rate=float(ss.opt_tog), key_swap_rate=float(ss.opt_swap),
                        local_search_steps=int(ss.opt_ls), immigrants=int(ss.opt_imm), seed=int(ss.opt_seed),
                        use_llm=bool(ss.opt_llm), llm_checkpoints=int(ss.opt_ckpt), llm_candidates=int(ss.opt_cands),
                        llm_lambda=float(ss.opt_lam), weights=w)


def _apply_limits(ss):
    metrics.LIMITS["travel_ft"] = float(ss.opt_travel)
    metrics.LIMITS["dead_end_ft"] = float(ss.opt_dead)
    planner.RULES["guest_window_required"] = bool(ss.get("opt_win_rule", True))
    for k in planner.PACK:
        planner.PACK[k] = float(ss.get(f"opt_p_{k}", planner.PACK[k]))


@st.cache_data(show_spinner=False)
def _site_preview(path: str) -> bytes:
    import io

    import matplotlib.pyplot as plt
    site = load_site(path)
    dummy = engine.Design(site, Program(rooms=[]), site.placeable[0] & False, [], [], site.placeable[0] * 0.0, 0, 0.0)
    fig, axes = plt.subplots(1, site.n_levels, figsize=(4.2 * site.n_levels, 4.2))
    for l, ax in enumerate(axes if site.n_levels > 1 else [axes]):
        render.draw_level(ax, dummy, l, title=f"{site.level_names[l]} (elev. {site.elevations_ft[l]:.1f} ft)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return buf.getvalue()


def _reference_for(site_path: Path) -> Path | None:
    stem = site_path.stem.replace("_multi", "")
    if stem.startswith("fp"):
        p = QA / f"floorplan{stem[2:]}.json"
        return p if p.exists() else None
    return None


# ------------------------------------------------------------------ tabs
def tab_inputs(ss, paths):
    site_path = paths["site_path"]
    if not site_path:
        st.info("Choose or upload a site grid in the sidebar.")
        return
    site = load_site(site_path)
    prog = _program(ss)
    lv = site.level_names
    usable = site.placeable_sf() * site.n_levels
    st.html(f"<div style='font-size:14px'><b>{site.name}</b>: {site.shape[0]} x {site.shape[1]} ft grid, "
            f"{site.n_levels} levels ({', '.join(lv)}), {site.placeable_sf():,.0f} SF placeable per level, "
            f"{len(site.cores)} cores. <b>Program</b>: {len(prog.instances(lv))} rooms, "
            f"{prog.keys_requested(lv)} keys, {prog.total_area_sf(lv):,.0f} SF "
            f"({100 * prog.total_area_sf(lv) / usable:.0f}% of placeable area before corridors).</div>")
    st.image(_site_preview(str(site_path)), use_container_width=True)
    st.markdown("**Room program** — edit any field; add or delete rows. `floors` is a comma list (e.g. `L1` or "
                "`L2, L3`); empty = all floors. Press **Apply program**.")
    rows = []
    for r in ss.opt_program["rooms"]:
        rows.append({"name": r["name"], "width_ft": r["width_ft"], "depth_ft": r["depth_ft"], "count": r.get("count", 1),
                     "priority": r.get("priority", 3), "category": r.get("category") or "guest",
                     "exterior": bool(r.get("exterior", True)), "near_core": bool(r.get("near_core", False)),
                     "corner": bool(r.get("corner", False)), "per_floor": bool(r.get("per_floor", False)),
                     "floors": ", ".join(r.get("floors") or []), "min_per_floor": int(r.get("min_per_floor", 0))})
    df = pd.DataFrame(rows)
    ed = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True,
                        key=f"opt_prog_editor_{ss.get('opt_prog_ver', 0)}", column_config={
                            "category": st.column_config.SelectboxColumn(options=CATS),
                            "priority": st.column_config.NumberColumn(min_value=0, max_value=10, step=0.5),
                            "count": st.column_config.NumberColumn(min_value=0, step=1)})
    adj = pd.DataFrame(ss.opt_program.get("adjacency") or [], columns=["a", "b", "weight", "max_ft"])
    st.markdown("**Adjacencies** (room type or instance names; weight and maximum distance in ft)")
    ed_adj = st.data_editor(adj, num_rows="dynamic", use_container_width=True, hide_index=True,
                            key=f"opt_adj_editor_{ss.get('opt_prog_ver', 0)}")
    c1, c2 = st.columns([1, 3])
    if c1.button("Apply program", type="primary"):
        rooms = []
        for _, r in ed.dropna(subset=["name"]).iterrows():
            floors = [f.strip() for f in str(r["floors"] or "").split(",") if f.strip()]
            rooms.append({"name": str(r["name"]), "width_ft": float(r["width_ft"]), "depth_ft": float(r["depth_ft"]),
                          "count": int(r["count"]), "priority": float(r["priority"]), "category": r["category"],
                          "exterior": bool(r["exterior"]), "near_core": bool(r["near_core"]), "corner": bool(r["corner"]),
                          "per_floor": bool(r["per_floor"]), "floors": floors or None,
                          "min_per_floor": int(r["min_per_floor"] or 0)})
        adjl = [{"a": str(a["a"]), "b": str(a["b"]), "weight": float(a["weight"] or 1), "max_ft": float(a["max_ft"] or 40)}
                for _, a in ed_adj.dropna(subset=["a", "b"]).iterrows()]
        ss.opt_program = {**ss.opt_program, "rooms": rooms, "adjacency": adjl, "_source": "edited in app"}
        ss.opt_prog_ver = ss.get("opt_prog_ver", 0) + 1
        st.rerun()
    c2.download_button("Download program JSON", json.dumps({k: v for k, v in ss.opt_program.items() if k != "_source"},
                                                           indent=2), "hotel_program.json", "application/json")


def tab_optimize(ss, paths):
    if not paths["site_path"]:
        st.info("Choose a site first.")
        return
    with st.expander("Objective weights (fitness)", expanded=False):
        cols = st.columns(2)
        for i, (k, v) in enumerate(metrics.DEFAULT_WEIGHTS.items()):
            cols[i % 2].slider(f"{k} — {metrics.OBJECTIVE_TEXT[k]}", 0.0, 0.6, v, 0.01, key=f"opt_w_{k}")
    with st.expander("Room-to-bay assignment weights (how rooms choose their bays)", expanded=False):
        cols = st.columns(4)
        help_ = {"exterior": "reward a window bay for rooms that need daylight",
                 "no_window": "penalty for a windowless bay (rooms that need daylight)",
                 "corner": "reward a corner bay (suites)", "core": "reward bays near a core (ADA, service)",
                 "adjacency": "reward meeting the program adjacencies", "stack": "reward the same room type above/below (typical floors)",
                 "group": "reward the same type next door (repetition)", "core_pull": "pull every room toward the cores"}
        for i, k in enumerate(planner.PACK):
            cols[i % 4].number_input(k, 0.0, 5.0, float(planner.PACK[k]), 0.05, key=f"opt_p_{k}", help=help_[k])
    cfg = _config(ss)
    est = cfg.population * cfg.generations * 0.08
    st.caption(f"About {cfg.population * (cfg.generations + 1):,} designs will be evaluated (~{est / 60:.1f} min, "
               f"plus {cfg.llm_checkpoints if cfg.use_llm else 0} LLM reviews).")
    if st.button("Run optimization", type="primary"):
        _apply_limits(ss)
        site, prog = load_site(paths["site_path"]), _program(ss)
        bar = st.progress(0.0, text="Starting...")
        chart = st.empty()
        log = st.container()
        hist = []

        def on_gen(rec, best):
            hist.append(rec)
            bar.progress(rec["gen"] / cfg.generations, text=f"Generation {rec['gen']}/{cfg.generations}: best "
                         f"{rec['best']:.4f}, {rec['best_placed']} rooms placed, {rec['evaluations']} designs evaluated")
            fig = go.Figure()
            fig.add_scatter(x=[h["gen"] for h in hist], y=[h["best"] for h in hist], name="best", line=dict(width=3))
            fig.add_scatter(x=[h["gen"] for h in hist], y=[h["mean"] for h in hist], name="mean")
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="generation", yaxis_title="fitness")
            chart.plotly_chart(fig, use_container_width=True)

        def on_review(e):
            log.html(f"<div class='rngd-item medium'><div class='item-top'><span>Review at generation {e['gen']}</span>"
                     f"<span class='rngd-tag'>{e['engine']}</span></div><div class='item-detail'>Scores: "
                     f"{', '.join(str(c['llm_score']) for c in e['candidates'])} · weights "
                     f"{', '.join(f'{k} x{v:.2f}' for k, v in e.get('weight_multipliers', {}).items()) or 'unchanged'}"
                     f"<br>{e.get('rationale', '')}</div></div>")

        res = Optimizer(site, prog, cfg).run(on_generation=on_gen, on_review=on_review)
        best = res["best"]
        OUT.mkdir(parents=True, exist_ok=True)
        plans = render.plan_figure(best.design, OUT / f"{site.name}_plans.png",
                                   f"{site.name}: {best.metrics['rooms_placed']}/{best.metrics['rooms_total']} rooms")
        conv = render.convergence_figure(res["history"], OUT / f"{site.name}_convergence.png", res["reviews"])
        ref = None
        rp = _reference_for(Path(paths["site_path"]))
        if rp:
            rprog = load_program(QA / "rooms_hotel.json")
            rd, info = baseline.load_reference(rp, site, rprog)
            rm = metrics.evaluate(rd)
            rimg = render.plan_figure(rd, OUT / f"{site.name}_reference_plans.png",
                                      f"Reference {info['file']}: {rm['rooms_placed']}/{rm['rooms_total']} rooms")
            ref = {"file": info["file"], "generations": info["generations"], "metrics": rm, "image": str(rimg)}
        ss.opt_result = {"site": site.name, "program": prog.source, "metrics": best.metrics, "history": res["history"],
                         "reviews": res["reviews"], "weights": res["weights"], "config": res["config"],
                         "seconds": res["seconds"], "evaluations": res["evaluations"], "reference": ref,
                         "figures": {"plans": str(plans), "convergence": str(conv),
                                     "reference": ref["image"] if ref else None},
                         "json": export.design_to_dict(best.design, best.metrics, history=res["history"],
                                                       reviews=res["reviews"], config=res["config"])}
        bar.progress(1.0, text=f"Done in {res['seconds']} s: {best.metrics['rooms_placed']} rooms placed, "
                               f"fitness {best.metrics['fitness']}. See the Design tab.")


def _kpis(m):
    from hotel_ui import kpis
    fails = [c for c in m["checks"] if c["status"] == "FAIL"]
    kpis([("Rooms placed", f"{m['rooms_placed']}/{m['rooms_total']}", "", ""),
          ("Keys", f"{m['keys_placed']}/{m['keys_requested']}", "", ""),
          ("Fitness", f"{m['fitness']:.3f}", "weighted objectives", ""),
          ("Corridor", f"{100 * m['corridor_ratio']:.1f}%", "of placeable", ""),
          ("Windows", f"{100 * m['exterior_fraction']:.0f}%", "rooms needing one", ""),
          ("Travel / dead end", f"{m['max_travel_ft']:.0f} / {m['dead_end_ft']:.0f}", "ft", ""),
          ("Code checks", f"{len(m['checks']) - len(fails)}/{len(m['checks'])}", f"{len(fails)} fail" if fails else "all pass",
           "bad" if fails else "good")])


def tab_design(ss):
    r = ss.get("opt_result")
    if not r:
        st.info("Run the optimization first.")
        return
    m = r["metrics"]
    _kpis(m)
    st.image(r["figures"]["plans"], use_container_width=True)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("**Per level**")
        st.dataframe(pd.DataFrame([{"level": lv["level"], "rooms": lv["rooms"], "keys": lv["keys"], "area SF": lv["area_sf"],
                                    "mix": ", ".join(f"{t} {n}" for t, n in lv["by_type"].items())} for lv in m["levels"]]),
                     hide_index=True, use_container_width=True)
        st.markdown("**Code checks** (limits from RNGD's IBC CSV and ADA Table 224.2)")
        st.dataframe(pd.DataFrame(m["checks"]), hide_index=True, use_container_width=True)
        if m["unplaced_by_type"]:
            st.warning("Not placed: " + ", ".join(f"{k} x{v}" for k, v in m["unplaced_by_type"].items()))
    with c2:
        fig = go.Figure(go.Bar(x=list(m["terms"].values()), y=list(m["terms"].keys()), orientation="h",
                               text=[f"{v:.2f} (w {m['weights'][k]:.2f})" for k, v in m["terms"].items()]))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), title="Objectives (score, final weight)",
                          xaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)
        st.image(r["figures"]["convergence"], use_container_width=True)
    if r["reviews"]:
        st.markdown("**LLM design reviews inside the fitness function**")
        for e in r["reviews"]:
            with st.expander(f"Generation {e['gen']} — {e['engine']} — scores "
                             f"{', '.join(str(c['llm_score']) for c in e['candidates'])}"):
                st.write(e.get("rationale", ""))
                for rv in e.get("reviews", []):
                    st.markdown(f"- **{rv['id']}** {rv['score']}/10 — strengths: {', '.join(rv['strengths']) or '-'}; "
                                f"issues: {', '.join(rv['issues']) or '-'}")
                st.caption("Weight changes: " + (", ".join(f"{k} x{v:.2f}" for k, v in e.get("weight_multipliers", {}).items())
                                                 or "none") + (" · sources: " + "; ".join(e.get("citations", [])) if e.get("citations") else ""))
    st.download_button("Download optimized design (JSON, QA schema)", json.dumps(r["json"], indent=1, default=str),
                       f"{r['site']}_optimized.json", "application/json")


def tab_compare(ss):
    r = ss.get("opt_result")
    if not r or not r.get("reference"):
        st.info("Run the optimization on a QA site (fp1/fp2/fp3) to compare with the reference design.")
        return
    ref, m = r["reference"], r["metrics"]
    rm = ref["metrics"]
    st.caption(f"Both designs are scored by the same evaluator. The reference ({ref['file']}, {ref['generations']} "
               f"generations) was made for the QA program; the optimized design used: {r['program']}.")
    rows = []
    for lab, k in (("Rooms placed", "rooms_placed"), ("Keys", "keys_placed"), ("Fitness", "fitness"),
                   ("Corridor ratio", "corridor_ratio"), ("Window fraction", "exterior_fraction"),
                   ("Max travel ft", "max_travel_ft"), ("Dead end ft", "dead_end_ft"), ("Cores reached", "cores_reached"),
                   ("Shared walls ft", "shared_wall_ft")):
        rows.append({"measure": lab, "reference": rm[k], "optimized": m[k]})
    rows.append({"measure": "Failed checks", "reference": ", ".join(c["rule_id"] for c in rm["checks"] if c["status"] == "FAIL") or "none",
                 "optimized": ", ".join(c["rule_id"] for c in m["checks"] if c["status"] == "FAIL") or "none"})
    st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.image(ref["image"], caption="Reference", use_container_width=True)
    c2.image(r["figures"]["plans"], caption="Optimized", use_container_width=True)


def tab_report(ss):
    r = ss.get("opt_result")
    if not r:
        st.info("Run the optimization first.")
        return
    st.markdown("The **AI report writer** (LLM + retrieved code passages) writes the narrative; all tables and figures come "
                "from the data, and every number in the AI text is fact-checked against the data.")
    c1, c2 = st.columns([1, 2])
    use_llm = c1.checkbox("Use the LLM to write the report", value=True, key="opt_rep_llm")
    from rngd import config as _cfg
    models = list(dict.fromkeys([_cfg.REPORT_MODEL, "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5",
                                 "openai/gpt-5.2", "google/gemini-3.1-pro-preview", _cfg.LLM_REASONING]))
    model = c2.selectbox("Writer model (OpenRouter)", models, key="opt_rep_model",
                         help="Used once per report for the prose only. Default: OPENROUTER_REPORT_MODEL or "
                              "anthropic/claude-sonnet-5.5.")
    if st.button("Write report with AI and export PDF", type="primary"):
        with st.spinner("Writing and compiling the report..."):
            try:
                pdf = report.build_report(r, OUT / "report", r["figures"], use_llm=use_llm, model=model)
                ss.opt_report = str(pdf)
            except Exception as e:
                st.error(f"Report failed: {e}")
    p = ss.get("opt_report")
    if p and Path(p).exists():
        st.download_button("Download the PDF report", Path(p).read_bytes(), Path(p).name, "application/pdf", type="primary")
        sec = Path(p).with_name(Path(p).stem + "_sections.json")
        if sec.exists():
            d = json.loads(sec.read_text(encoding="utf-8"))
            fc = d["fact_check"]
            bad = [c["number"] for c in fc if not c["traced"]]
            st.caption(f"Written by {d['engine']}. Fact check: {len(fc) - len(bad)}/{len(fc)} numbers traced to the data"
                       + (f"; not traced: {', '.join(bad)}" if bad else "."))
            st.markdown(f"### {d['sections'].get('title', '')}")
            st.write(d["sections"].get("executive_summary", ""))


def render_mode(ss, paths):
    tabs = st.tabs(["Site & program", "Optimize", "Design", "Compare with reference", "AI report (PDF)"])
    with tabs[0]:
        tab_inputs(ss, paths)
    with tabs[1]:
        tab_optimize(ss, paths)
    with tabs[2]:
        tab_design(ss)
    with tabs[3]:
        tab_compare(ss)
    with tabs[4]:
        tab_report(ss)
