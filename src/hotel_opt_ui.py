"""Hotel Schematic -> "Optimize layouts" tab.

Generates every floor plate the site allows (Bar, L, U, T, H, Courtyard x proportions x core
placement x floor count), lays each one out with the architectural decoder, scores it with the
deterministic fitness (no LLM), refines the best by pattern search, then ranks all distinct
layouts with a multi-criteria decision model (Pareto + TOPSIS; Jev optional as an advisor).
The user browses the top N or all layouts from a dropdown and inspects any of them.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rngd.optimize import decide, explore, export, metrics, plan_view, typology

SHOW = ["Top 5", "Top 10", "Top 25", "Pareto front", "All layouts"]
SHAPE_MODES = ["Generative: all shapes", "Blend: choose shapes", "Dictate one shape"]


@st.cache_resource(show_spinner=False, max_entries=8)
def _store() -> dict:
    return {}


def _cfg(ss) -> explore.ExploreConfig:
    mode = ss.get("hopt_shape_mode", SHAPE_MODES[0])
    if mode == SHAPE_MODES[2]:
        shapes = (ss.get("hopt_one_shape", "Bar"),)
    elif mode == SHAPE_MODES[1]:
        shapes = tuple(ss.get("hopt_shapes") or typology.SHAPES)
    else:
        shapes = typology.SHAPES
    weights = {k: float(ss.get(f"hopt_w_{k}", v)) for k, v in metrics.DEFAULT_WEIGHTS.items()}
    return explore.ExploreConfig(
        guest_floors=int(ss.get("hot_floors", 5)), floors_mode="auto" if ss.get("hopt_floors_auto", True) else "fixed",
        max_guest_floors=int(ss.get("hopt_max_floors", 8)),
        envelope=(int(ss.get("hopt_env_w", ss.get("hot_w", 170))), int(ss.get("hopt_env_d", ss.get("hot_d", 265)))),
        shapes=shapes, schemes=tuple(ss.get("hopt_schemes") or typology.CORE_SCHEMES),
        ratios=(0.6, 1.0), seeds_per_plate=int(ss.get("hopt_seeds", 6)), refine_top=int(ss.get("hopt_refine", 8)),
        refine_steps=int(ss.get("hopt_steps", 24)), floor_height_ft=float(ss.get("hot_fl", 10.75)),
        ground_height_ft=float(ss.get("hot_l1", 16.0)), workers=int(ss.get("hopt_workers", 0)), weights=weights)


def _sig(prog, cfg) -> str:
    c = dict(cfg.__dict__)
    return json.dumps({"rows": prog.rows, "cfg": c}, default=str, sort_keys=True)


def _settings(ss):
    st.markdown("Generates the floor plates the site allows, lays out every one of them, and ranks the results. "
                "The fitness is geometry and code arithmetic only (**no LLM**), so the same inputs give the same "
                "answer every time. Built on the 10/08 meeting: the shape is not fixed, the core and stairs are placed "
                "by the search, a floor is added when the keys do not fit, and rooms run perpendicular to a central "
                "corridor.")
    with st.form("hopt_form", border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("Buildable width (ft)", 60, 800, int(ss.get("hot_w", 170)), key="hopt_env_w",
                        help="The area the building may occupy (after setbacks, parking, egress).")
        c2.number_input("Buildable depth (ft)", 60, 800, int(ss.get("hot_d", 265)), key="hopt_env_d")
        c3.checkbox("Let it add / remove floors", value=True, key="hopt_floors_auto",
                    help=f"Tries {int(ss.get('hot_floors', 5)) - 1} to {int(ss.get('hot_floors', 5)) + 2} guest floors "
                         "(sidebar value +/-); off = exactly the sidebar value.")
        c4.number_input("Max guest floors (height limit)", 2, 12, 8, key="hopt_max_floors")
        c1, c2, c3 = st.columns([1.2, 1.6, 1.6])
        mode = c1.radio("Building shape", SHAPE_MODES, key="hopt_shape_mode",
                        help="Generative: try all shapes. Blend: the shapes you pick. Dictate: one shape only.")
        c2.multiselect("Shapes (blend)", list(typology.SHAPES), default=list(typology.SHAPES), key="hopt_shapes",
                       format_func=lambda s: f"{s} - {typology.SHAPE_TEXT[s]}", disabled=False)
        c2.selectbox("Shape (dictate)", list(typology.SHAPES), key="hopt_one_shape")
        c3.multiselect("Core / stair placement", list(typology.CORE_SCHEMES), default=list(typology.CORE_SCHEMES),
                       key="hopt_schemes", format_func=lambda s: f"{s}: {typology.CORE_SCHEMES[s]}",
                       help="All selected = generative core placement (the search picks).")
        with st.expander("Search effort and objective weights", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("Corridor variants per plate", 2, 6, 6, key="hopt_seeds",
                            help="Spine combinations tried on every plate (all, best one, best two, ...).")
            c2.number_input("Plates refined", 1, 30, 8, key="hopt_refine",
                            help="Best plate of each shape first, then the next best.")
            c3.number_input("Refinement steps per plate", 0, 200, 24, key="hopt_steps")
            c4.number_input("Parallel workers (0 = auto, 1 = off)", 0, 32, 0, key="hopt_workers")
            st.caption("Objective weights (fitness):")
            cols = st.columns(5)
            for i, (k, v) in enumerate(metrics.DEFAULT_WEIGHTS.items()):
                cols[i % 5].slider(k.replace("_", " "), 0.0, 0.5, float(v), 0.01, key=f"hopt_w_{k}",
                                   help=metrics.OBJECTIVE_TEXT[k])
        run = st.form_submit_button("Optimize layouts", type="primary", use_container_width=True)
    return run


def _run(ss, prog, cfg):
    sig = _sig(prog, cfg)
    store = _store()
    if sig in store:
        ss.hopt_res = store[sig]
        return
    bar = st.progress(0.0, text="Screening plates...")

    def cb(stage, i, n):
        frac = (0.8 * i / n) if stage == "screen" else (0.8 + 0.2 * i / n)
        bar.progress(min(frac, 1.0), text=f"{'Screening' if stage == 'screen' else 'Refining'} {i}/{n}")

    res = explore.explore(prog, cfg, on_progress=cb)
    bar.empty()
    store[sig] = res
    ss.hopt_res = res


def _design(ss, prog, res, alt):
    cache = ss.setdefault("hopt_designs", {})
    key = (id(res), alt.id)
    if key not in cache:
        cache[key] = explore.design_for(prog, alt, res["config"])
    return cache[key]


@st.cache_data(show_spinner=False, max_entries=256)
def _thumb(_design, _plate, key: str, level: int, title: str) -> bytes:
    return plan_view.thumbnail(_design, _plate, level, title)


def _fmt_row(r: dict) -> str:
    return (f"#{r['rank']} · {r['id']} · {r['layout']} · {r['floors']} guest floors · {r['keys']} keys · "
            f"score {r['score']:.3f}{' · Pareto' if r['pareto'] else ''}")


def render_tab(ss, kpis):
    prog = ss.hotel_program
    run = _settings(ss)
    cfg = _cfg(ss)
    if run:
        if not cfg.shapes:
            st.warning("Pick at least one shape.")
            return
        _run(ss, prog, cfg)
    res = ss.get("hopt_res")
    if not res:
        st.info("Set the site and options above and press **Optimize layouts** (about 30-60 s for all shapes; "
                "repeat runs with the same inputs are instant).")
        return
    alts = res["alternatives"]
    if not alts:
        st.error("No plate fits the buildable area; enlarge it or allow more floors.")
        return
    keys_req = prog.keys

    # ---------------------------------------------------------------- decision
    st.markdown("#### Decide")
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    preset = c1.selectbox("Decision priority", list(decide.PRESETS), key="hopt_preset",
                          help="Weights of the multi-criteria ranking (TOPSIS). Adjust them below.")
    show = c2.selectbox("Show", SHOW[:-1] + [f"All layouts ({len(alts)})"], index=1, key="hopt_show")
    shapes_present = sorted({a.plate.shape for a in alts})
    flt = c3.multiselect("Filter shapes", shapes_present, default=shapes_present, key="hopt_filter")
    with st.expander("Decision weights (criteria)", expanded=False):
        cols = st.columns(4)
        dw = {}
        for i, (k, (sgn, label, unit)) in enumerate(decide.CRITERIA.items()):
            dw[k] = cols[i % 4].slider(f"{label} ({'higher' if sgn > 0 else 'lower'} is better)", 0.0, 5.0,
                                       float(decide.PRESETS[preset].get(k, 0.0)), 0.5, key=f"hopt_dw_{preset}_{k}")
    pool = [a for a in alts if a.plate.shape in flt] or alts
    rows = decide.rank(pool, dw)
    if show.startswith("Top"):
        rows_shown = rows[:int(show.split()[1])]
    elif show == "Pareto front":
        rows_shown = [r for r in rows if r["pareto"]]
    else:
        rows_shown = rows
    best = rows[0]
    ub = next(a for a in alts if a.id == best["id"]).upper_bound_keys
    kpis([("Layouts explored", f"{len(alts)}", f"{res['evaluations']} evaluations", ""),
          ("Time", f"{res['seconds']:.0f} s", "no LLM calls", "good"),
          ("Shapes", f"{len(shapes_present)}", ", ".join(shapes_present), ""),
          ("Pareto layouts", f"{sum(r['pareto'] for r in rows)}", "not beaten on every criterion", ""),
          ("Best keys", f"{max(a.metrics['keys_placed'] for a in alts)}", f"of {keys_req} requested",
           "good" if max(a.metrics['keys_placed'] for a in alts) >= keys_req else "bad"),
          ("Top choice", best["id"], f"{best['shape']} · {best['floors']} fl", "")])

    df = pd.DataFrame(rows_shown)
    view = df[["rank", "id", "layout", "floors", "score", "pareto", "keys", "fitness", "program_placed", "daylight",
               "corridor_ratio", "gross_sf_per_key", "facade_ft_per_key", "height_ft", "dead_end_ft", "code_pass"]]
    st.dataframe(view, hide_index=True, use_container_width=True, height=min(420, 38 + 35 * len(view)),
                 column_config={"score": st.column_config.ProgressColumn("TOPSIS score", min_value=0, max_value=1,
                                                                         format="%.3f"),
                                "fitness": st.column_config.NumberColumn(format="%.3f"),
                                "program_placed": st.column_config.NumberColumn("program", format="%.2f"),
                                "daylight": st.column_config.NumberColumn(format="%.2f"),
                                "corridor_ratio": st.column_config.NumberColumn("corridor", format="%.3f"),
                                "gross_sf_per_key": st.column_config.NumberColumn("gross SF/key", format="%.0f"),
                                "facade_ft_per_key": st.column_config.NumberColumn("facade ft/key", format="%.1f"),
                                "height_ft": st.column_config.NumberColumn("height ft", format="%.0f"),
                                "dead_end_ft": st.column_config.NumberColumn("dead end ft", format="%.0f"),
                                "code_pass": st.column_config.NumberColumn("code", format="%.2f")})
    c1, c2 = st.columns([1, 3])
    c1.download_button("Download ranking (CSV)", df.to_csv(index=False), "hotel_layout_ranking.csv", "text/csv")
    with c2.expander("Search log", expanded=False):
        st.code("\n".join(res["log"]), language=None)

    # ---------------------------------------------------------------- gallery
    st.markdown("#### Compare")
    lvl_thumb = 1 if any(a.floors for a in alts) else 0
    top = rows_shown[:8]
    cols = st.columns(4)
    for i, r in enumerate(top):
        a = next(x for x in alts if x.id == r["id"])
        d = _design(ss, prog, res, a)
        with cols[i % 4]:
            st.image(_thumb(d, a.plate, f"{id(res)}-{a.id}", lvl_thumb, f"#{r['rank']} {a.id} · {a.plate.shape}"),
                     use_container_width=True)
            st.caption(f"{r['keys']} keys · {r['floors']} fl · {r['gross_sf_per_key']:.0f} SF/key · "
                       f"{'Pareto' if r['pareto'] else 'dominated'}")
    c1, c2 = st.columns(2)
    sc = pd.DataFrame(rows)
    fig = px.scatter(sc, x="gross_sf_per_key", y="keys", color="shape", symbol="pareto", size="score",
                     hover_data=["id", "layout", "floors", "fitness", "height_ft"], height=360,
                     labels={"gross_sf_per_key": "Gross SF per key (lower = cheaper)", "keys": "Keys placed"},
                     title="Every layout: keys vs. area per key (Pareto = filled symbols)")
    fig.add_hline(y=keys_req, line_dash="dot", annotation_text=f"{keys_req} requested")
    c1.plotly_chart(fig, use_container_width=True)
    if res.get("traces"):
        f2 = go.Figure()
        for t in res["traces"]:
            f2.add_trace(go.Scatter(y=t["trace"], mode="lines+markers", name=f"{t['plate']} ({t['floors']} fl)"))
        f2.update_layout(height=360, title="Refinement (pattern search): fitness of each improving move",
                         xaxis_title="improvement #", yaxis_title="fitness", legend=dict(font=dict(size=9)))
        c2.plotly_chart(f2, use_container_width=True)

    # ---------------------------------------------------------------- inspect one
    st.markdown("#### Inspect a layout")
    labels = [_fmt_row(r) for r in rows_shown]
    pick = st.selectbox("Layout", labels, key="hopt_pick", help="Choose any of the shown layouts (see 'Show' above).")
    r = rows_shown[labels.index(pick)] if pick in labels else rows_shown[0]
    a = next(x for x in alts if x.id == r["id"])
    d = _design(ss, prog, res, a)
    m = a.metrics
    fails = [c for c in m["checks"] if c["status"] == "FAIL"]
    st.markdown(f"**{a.plate.name}**, {a.floors} guest floors over Level 1. {typology.SHAPE_TEXT[a.plate.shape]}; "
                f"{typology.CORE_SCHEMES[a.plate.core_scheme]}. {decide.why(r, rows)}"
                + (" Plate clipped to the buildable area." if a.plate.clipped else "")
                + (f" Runs along the {res['config'].envelope[1]} ft site depth (drawn rotated 90 degrees)."
                   if a.plate.size[0] > res["config"].envelope[0] else ""))
    kpis([("Keys", f"{m['keys_placed']}", f"of {m['keys_requested']}", "good" if m['keys_placed'] >= m['keys_requested'] else "bad"),
          ("Upper bound", f"{a.upper_bound_keys}", "keys / typical floor (frontage / module)", ""),
          ("Rooms", f"{m['rooms_placed']}/{m['rooms_total']}", "program rooms", ""),
          ("Footprint", f"{a.plate.size[0]}x{a.plate.size[1]}", f"{a.plate.footprint_sf():,} SF", ""),
          ("Height", f"{r['height_ft']:.0f} ft", f"{a.floors + 1} stories", ""),
          ("Corridor", f"{m['corridor_ratio']:.1%}", "of plate (circulation)", ""),
          ("Dead end", f"{m['dead_end_ft']:.0f} ft", "limit 50", "bad" if m['dead_end_ft'] > 50 else "good"),
          ("Code checks", f"{len(m['checks']) - len(fails)}/{len(m['checks'])}", "IBC + ADA", "bad" if fails else "good"),
          ("Fitness", f"{m['fitness']:.3f}", "deterministic", "")])
    t1, t2, t3, t4 = st.tabs(["Floor plan", "3D massing", "Code, program & objectives", "Jev / export"])
    with t1:
        c1, c2, c3 = st.columns([2.4, 1.4, 1])
        lv = c1.radio("Level", d.site.level_names, index=min(1, d.site.n_levels - 1), horizontal=True, key="hopt_level")
        types = sorted({p.type_name for p in d.placements})
        hl = c2.selectbox("Highlight", ["(none)"] + types, key="hopt_hl")
        furn = c3.toggle("Furniture", value=True, key="hopt_furn")
        li = d.site.level_names.index(lv)
        st.plotly_chart(plan_view.plan_figure(d, a.plate, li, None if hl == "(none)" else hl, furn),
                        use_container_width=True)
        lvm = next((x for x in m["levels"] if x["level"] == lv), None)
        if lvm:
            st.caption(f"{lv}: {lvm['rooms']} rooms, {lvm['keys']} keys, {lvm['area_sf']:,} SF of rooms. "
                       "Blue facade lines = windows; dotted arcs = entry doors; grey dashed bays = unassigned (flex).")
    with t2:
        ex = st.slider("Explode floors (ft)", 0.0, 30.0, 0.0, 1.0, key="hopt_explode")
        st.plotly_chart(plan_view.massing_figure(d, a.plate, ex), use_container_width=True)
    with t3:
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame(m["checks"]), hide_index=True, use_container_width=True)
        c2.dataframe(pd.DataFrame([{"level": x["level"], "rooms": x["rooms"], "keys": x["keys"], "room SF": x["area_sf"]}
                                   for x in m["levels"]]), hide_index=True, use_container_width=True)
        if m["unplaced_by_type"]:
            st.warning("Not placed: " + ", ".join(f"{k} x{v}" for k, v in m["unplaced_by_type"].items()))
        terms = pd.DataFrame({"objective": list(m["terms"]), "value": list(m["terms"].values()),
                              "weight": [m["weights"][k] for k in m["terms"]]})
        st.plotly_chart(px.bar(terms, x="objective", y="value", color="weight", range_y=[0, 1], height=300,
                               title="Objective values (0-1, 1 = best)"), use_container_width=True)
    with t4:
        ok, msg = decide.jev_status()
        (st.success if ok else st.info)(msg)
        st.caption("Sources: https://docs.openclaw.ai/plugins/typesafe · "
                   "https://shop.zimaspace.com/blogs/tech-ai-hub/jev-use-cases-decision-model-examples")
        if ok and st.button("Ask Jev to choose among the shown layouts"):
            with st.spinner("Asking Jev..."):
                ans = decide.jev_choice(rows_shown, "Which hotel floor-plate layout should the developer build?")
            st.json(ans)
        out = export.design_to_dict(d, m)
        out["plate"] = {"name": a.plate.name, "shape": a.plate.shape, "core_scheme": a.plate.core_scheme,
                        "rects": a.plate.rects, "cores": a.plate.cores}
        out["decision"] = {k: r[k] for k in ("rank", "score", "pareto")}
        st.download_button("Download this layout (JSON)", json.dumps(out, default=str, indent=1),
                           f"hotel_layout_{a.id}.json", "application/json")
