"""RNGD AI Studio — unified app.

Three modes sharing one window: the concept-to-schematic multi-agent
pipeline, the parametric IFC Design Studio, and an IFC file explorer for
opening a REAL, arbitrary .ifc file — all with a persistent chat assistant
that knows about every component in the project.

Run: streamlit run src/app.py
"""
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

from rngd import chat, config, real_docs, summary_card
from rngd.ifc import code_check, explorer
from rngd.ifc.design import SCENARIOS, ZONING, Params, generate, module_counts
from rngd.ifc.ifc_writer import read_meshes, write_ifc
from rngd.ifc.viewer3d import build_figure, build_figure_generic
from rngd.orchestrator import ask_agent, run_pipeline
from rngd.vectorstore import get_vectorstore

st.set_page_config(page_title="RNGD AI Studio", layout="wide", page_icon=":material/architecture:")

# ============================================================== theme ====
st.markdown(
    """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --ink:#182228; --muted:#5B6670; --paper:#F5F1E7;
  --blueprint:#1E5AA8; --blueprint-deep:#123A6E; --blueprint-wash:rgba(30,90,168,.08);
  --amber:#9C6B0B; --amber-wash:rgba(196,140,10,.14);
  --critical:#A32A21; --high:#B0590B; --medium:#8A7411; --ok:#2E6B47; --ok-wash:rgba(46,107,71,.10);
}
html, body, [class*="css"] { font-family: "IBM Plex Sans", sans-serif; }
h1,h2,h3,h4 { font-family: "IBM Plex Sans", sans-serif !important; font-weight:700 !important; }
code, .stCode, [data-testid="stMetricValue"] { font-family: "IBM Plex Mono", monospace !important; }

.rngd-header{
  background: linear-gradient(135deg, var(--blueprint-deep), var(--blueprint));
  color:#F5F1E7; border-radius:10px; padding:20px 26px; margin-bottom:6px;
  border-bottom:3px solid var(--amber);
}
.rngd-header .eyebrow{ font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.09em;
  text-transform:uppercase; opacity:.85; margin-bottom:4px; }
.rngd-header h1{ color:#fff !important; margin:0 !important; font-size:26px !important; }
.rngd-header .sub{ opacity:.9; font-size:14px; margin-top:6px; max-width:70ch; }

.rngd-pill{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:100px;
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; font-weight:500; margin-right:6px; }
.rngd-pill.ok{ background:var(--ok-wash); color:#DCEEE2; border:1px solid rgba(255,255,255,.25); }
.rngd-pill::before{ content:""; width:6px; height:6px; border-radius:50%; background:currentColor; }

[data-testid="stMetric"]{ background:var(--blueprint-wash); border:1px solid rgba(30,90,168,.18);
  border-radius:8px; padding:10px 14px 6px; }
[data-testid="stSidebar"]{ border-right:1px solid rgba(24,34,40,.10); }
.stTabs [data-baseweb="tab-list"]{ gap:2px; }
.stTabs [data-baseweb="tab"]{ font-family:"IBM Plex Mono",monospace; font-size:13px; }

[data-testid="stChatMessage"]{ border-radius:10px; padding:4px 2px; }
.rngd-chat-title{ font-family:"IBM Plex Mono",monospace; text-transform:uppercase; letter-spacing:.08em;
  font-size:11px; color:var(--muted); margin:2px 0 10px; display:flex; align-items:center; gap:6px; }
.rngd-chat-title::before{ content:""; width:7px; height:7px; border-radius:50%; background:var(--ok); box-shadow:0 0 0 3px var(--ok-wash); }

.status-chip{ display:inline-block; padding:1px 8px; border-radius:100px; font-family:"IBM Plex Mono",monospace;
  font-size:11px; font-weight:500; }
.status-chip.PASS{ background:var(--ok-wash); color:var(--ok); }
.status-chip.FAIL{ background:rgba(163,42,33,.10); color:var(--critical); }
.status-chip.WARN{ background:rgba(176,89,11,.10); color:var(--high); }

/* ---- chat panel: summary card ---- */
.rngd-card{ background:var(--blueprint-wash); border:1px solid rgba(30,90,168,.18); border-radius:10px;
  padding:14px 16px; margin-bottom:12px; }
.rngd-card .card-top{ display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
.rngd-card .card-title{ font-weight:700; font-size:14px; }
.rngd-card .card-badge{ font-family:"IBM Plex Mono",monospace; font-size:10.5px; color:var(--muted); }
.rngd-card .card-headline{ font-size:13px; margin:6px 0 8px; color:var(--ink); }
.rngd-match{ margin:8px 0; }
.rngd-match .bar{ height:8px; border-radius:100px; background:rgba(24,34,40,.10); overflow:hidden; }
.rngd-match .fill{ height:100%; border-radius:100px; }
.rngd-match .label{ font-family:"IBM Plex Mono",monospace; font-size:10.5px; color:var(--muted); margin-top:3px; display:flex; justify-content:space-between; }
.rngd-kv{ display:grid; grid-template-columns:1fr 1fr; gap:4px 10px; font-size:12px; margin-top:6px; }
.rngd-kv div{ display:flex; justify-content:space-between; gap:6px; border-bottom:1px dashed rgba(24,34,40,.08); padding:2px 0; }
.rngd-kv .k{ color:var(--muted); }
.rngd-kv .v{ font-family:"IBM Plex Mono",monospace; }
.rngd-sec-label{ font-family:"IBM Plex Mono",monospace; font-size:10.5px; text-transform:uppercase; letter-spacing:.07em;
  color:var(--muted); margin:10px 0 4px; }
.rngd-item{ border-radius:8px; padding:7px 10px; margin-bottom:5px; font-size:12.5px; border-left:3px solid; }
.rngd-item .item-top{ display:flex; justify-content:space-between; gap:8px; font-weight:600; }
.rngd-item .item-detail{ color:var(--ink); opacity:.85; margin-top:2px; }
.rngd-item.critical{ background:rgba(163,42,33,.08); border-color:var(--critical); }
.rngd-item.high{ background:rgba(176,89,11,.08); border-color:var(--high); }
.rngd-item.medium{ background:rgba(138,116,17,.08); border-color:var(--medium); }
.rngd-item.low, .rngd-item.info{ background:rgba(91,102,112,.08); border-color:var(--muted); }
.rngd-item.ok{ background:var(--ok-wash); border-color:var(--ok); }
.rngd-tag{ font-family:"IBM Plex Mono",monospace; font-size:9.5px; text-transform:uppercase; letter-spacing:.05em;
  padding:1px 6px; border-radius:100px; background:rgba(24,34,40,.08); color:var(--muted); white-space:nowrap; }

div[data-testid="stVerticalBlockBorderWrapper"] .stButton button{ font-size:12px; padding:4px 10px; white-space:normal; height:auto; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================ session ====
ss = st.session_state
ss.setdefault("pipeline_state", None)
ss.setdefault("chat_history", [])
ss.setdefault("ifc_layout", None)
ss.setdefault("ifc_params", None)
ss.setdefault("pipeline_suggested_for", None)
ss.setdefault("explorer_summary", None)
ss.setdefault("explorer_meshes", None)
ss.setdefault("explorer_file", None)
ss.setdefault("explorer_suggested_for", None)
ss.setdefault("wide_view", False)
for k, v in dict(units=120, floors=4, zoning=list(ZONING)[0], parking="surface", wings=0, studio=20, onebr=45).items():
    ss.setdefault(k, v)

vs = get_vectorstore()

# ============================================================= header ====
n_scenarios = len(SCENARIOS)
decision = ss.pipeline_state.final_decision.decision if ss.pipeline_state else None
st.markdown(
    f"""<div class="rngd-header">
  <div class="eyebrow">RNGD &middot; Concept-to-Schematic AI &middot; Semester Proof of Concept</div>
  <h1>RNGD AI Studio</h1>
  <div class="sub">Multi-agent schematic pipeline (Groq + Claude, physics simulation, classical CV), a parametric
  IFC Design Studio, and a real-IFC file explorer — side by side with a chat assistant that knows every component.
  Pipeline/IFC-studio zoning and BaaP data is synthetic; the file explorer opens real models.</div>
  <div style="margin-top:12px">
    <span class="rngd-pill ok">10 pipeline agents</span>
    <span class="rngd-pill ok">{n_scenarios} IFC scenarios</span>
    <span class="rngd-pill ok">{len(explorer.list_available_files())} openable IFC files</span>
    <span class="rngd-pill ok">{len(vs.chunks)} KB chunks ({vs.backend_name})</span>
    <span class="rngd-pill ok">{vs.real_chunk_count} from RNGD's real IBC/ADA docs</span>
    {f'<span class="rngd-pill ok">last decision: {decision}</span>' if decision else ""}
  </div>
</div>""",
    unsafe_allow_html=True,
)

# ============================================================= sidebar ===
with st.sidebar:
    mode = st.radio("Mode", ["Multi-Agent Pipeline", "IFC Design Studio", "Open IFC File"], key="mode")
    st.checkbox("Wide 3D view (collapses the chat panel)", key="wide_view",
                help="Gives the model viewer most of the window. Chat stays usable, just narrower.")
    st.divider()

    if mode == "Multi-Agent Pipeline":
        st.subheader("Run")
        project_id = st.selectbox("Sample project", options=["project_001"])
        max_iter = st.slider("Max simulation iterations", 1, 5, config.MAX_SIMULATION_ITERATIONS)
        run_clicked = st.button("Run full pipeline", type="primary", use_container_width=True)

    elif mode == "Open IFC File":
        st.subheader("Open a real IFC file")
        available = explorer.list_available_files()
        if not available:
            st.warning("No .ifc files found. Drop one into `data/ifc_samples/` and reload the page.")
            explorer_path = None
        else:
            explorer_path = st.selectbox("File", available, format_func=lambda p: f"{p.name}  ({p.stat().st_size / 1e6:.0f} MB)")
        skip_framing = st.checkbox("Skip fine curtain-wall framing (faster)", value=True,
                                    help="Hides individual mullions/panels (IfcMember/IfcPlate) — much faster to triangulate, walls/doors/windows/slabs still show.")
        explorer_load_clicked = st.button("Load / open file", type="primary", use_container_width=True, disabled=available == [])
        st.caption("First open of a large file can take 1-3 minutes (one-time — cached to `outputs/ifc_cache/` after that).")
        if ss.explorer_summary is not None:
            st.divider()
            classes = list(ss.explorer_summary["categories"])
            explorer_class_filter = st.multiselect("Show categories", classes, default=[c for c in classes if c not in ("IfcFurnishingElement", "IfcFlowTerminal", "IfcCovering", "IfcMember", "IfcPlate")], key="explorer_classes")

    else:
        st.subheader("Scenarios")

        def _load(p):
            ss.update(units=p.units, floors=p.floors, zoning=p.zoning, parking=p.parking, wings=p.wings,
                      studio=round(p.mix[0] * 100), onebr=round(p.mix[1] * 100))

        for name, p in SCENARIOS.items():
            st.button(name, on_click=_load, args=(p,), use_container_width=True, key=f"scn_{name}")
        st.divider()
        st.subheader("Requirements")
        with st.expander("Paste a customer request (AI parse)"):
            text = st.text_area("Request text", (config.SAMPLE_PROJECTS_DIR / "project_001/customer_rfp.txt").read_text(encoding="utf-8"), height=120)
            if st.button("Extract requirements"):
                from rngd.agents import intake_agent
                try:
                    prog, _ = intake_agent.run(text)
                    mix = prog.unit_mix_pct
                    ss.update(units=prog.unit_count_total or 120, floors=prog.floor_count or 4,
                              studio=round(mix.get("studio", .2) * 100), onebr=round(mix.get("1br", .45) * 100))
                    st.success("Loaded into the controls below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not parse: {e}")
        st.number_input("Units", 10, 400, key="units", step=6)
        st.slider("Residential floors", 1, 9, key="floors")
        st.slider("Studio %", 0, 100, key="studio")
        st.slider("1-bedroom %", 0, 100 - ss.studio, key="onebr")
        st.caption(f"2-bedroom %: {100 - ss.studio - ss.onebr}")
        st.divider()
        st.subheader("Site, zoning, options")
        st.selectbox("Zoning preset", list(ZONING), key="zoning")
        st.radio("Parking", ["surface", "podium"], key="parking", horizontal=True)
        st.select_slider("Wings (0 = auto)", [0, 1, 2, 3], key="wings")
        level_view = st.select_slider("View floors up to", ["all"] + list(range(1, 10)), value="all", key="level_view")
        st.checkbox("Show site context (ground/easement/driveway/parking) in 3D view", value=False, key="show_site",
                    help="Off by default: the full lot is usually much bigger than the building, which made the building look tiny.")

    st.divider()
    with st.expander(f":material/library_books: Documents ({len(vs.chunks)} chunks, {vs.backend_name})", expanded=False):
        st.caption("Synthetic sample (data/knowledge_base/*.md): zoning preset, BaaP catalog, stacking rules, design guidelines.")
        st.caption(f"**Real, RNGD-supplied (docs/), {vs.real_chunk_count} chunks:**")
        avail = real_docs.available()
        st.markdown(
            f"- {'✅' if avail['ibc_csv'] else '❌'} 2021 IBC rules — 42 machine-testable rules, 5-story R-1 hotel prototype (CSV)\n"
            f"- {'✅' if avail['ibc_xlsx'] else '❌'} Companion workbook — project config, reference projects, occupant-load factors, hotel space mapping\n"
            f"- {'✅' if avail['ada_pdf'] else '❌'} ADA 2010 Standards for Accessible Design — official DOJ PDF, 279 pages"
        )
        doc_q = st.text_input("Search these documents", value="dead-end corridor", key="doc_search")
        if doc_q:
            for h in vs.retrieve(doc_q, k=4):
                badge = "real" if h["source_kind"] == "real" else "sample"
                with st.expander(f"[{badge}] {h['source']} | {h['heading']} (score {h['score']:.2f})"):
                    st.text(h["text"])

# ======================================================= status hook =====
STATUS_ICON = {"ok": ":material/check_circle:", "degraded": ":material/warning:", "error": ":material/error:"}


def _render_step(container, entry):
    icon = STATUS_ICON.get(entry.status, ":material/circle:")
    container.write(f"{icon} **{entry.agent}** — {entry.action}  ·  `{entry.latency_ms:.0f} ms`  \n{entry.summary}")


# =========================================================== columns =====
main_col, chat_col = st.columns([5.5, 1] if ss.wide_view else [1.8, 1.3], gap="large")

# ------------------------------------------------------- pipeline mode ---
with main_col:
    if mode == "Multi-Agent Pipeline":
        if run_clicked:
            status = st.status("Running multi-agent pipeline...", expanded=True)
            status.write(":material/visibility: **vision_input** — reading site sketch with classical CV...")
            state = run_pipeline(project_id, max_iterations=max_iter, on_step=lambda e: _render_step(status, e))
            status.update(label=f"Pipeline complete — {state.final_decision.decision}", state="complete", expanded=False)
            ss.pipeline_state = state

        state = ss.pipeline_state
        if state is None:
            st.info("Click **Run full pipeline** in the sidebar. Meanwhile, here are the raw inputs:")
            st.subheader("Sample customer request")
            st.text((config.SAMPLE_PROJECTS_DIR / "project_001/customer_rfp.txt").read_text(encoding="utf-8"))
            st.image(str(config.SAMPLE_PROJECTS_DIR / "project_001/site_sketch.png"), width=380)
        else:
            fingerprint = f"{state.final_decision.decision}-{len(state.trace)}"
            if ss.pipeline_suggested_for != fingerprint:
                with st.spinner("Assistant reviewing the result..."):
                    note, _ = chat.suggest_after_pipeline(state, vs)
                ss.chat_history.append({"role": "assistant", "content": f"**Pipeline finished for `{state.project_id}`.**\n\n{note}"})
                ss.pipeline_suggested_for = fingerprint

            tabs = st.tabs(["Overview", "Agent trace", "Program & site", "BaaP & conflicts", "Site simulation", "Floor plates", "Final decision", "Ask an agent"])

            with tabs[0]:
                c1, c2, c3 = st.columns(3)
                c1.metric("Decision", state.final_decision.decision)
                c2.metric("Confidence", f"{state.final_decision.confidence:.0%}")
                c3.metric("Conflicts found", len(state.conflicts.conflicts) if state.conflicts else 0)
                st.write(state.final_decision.narrative)
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Generated site plan")
                    if state.site_plan_image_path:
                        st.image(state.site_plan_image_path)
                with c2:
                    st.subheader("Typical upper floor")
                    if state.floor_plate_image_path:
                        st.image(state.floor_plate_image_path)

            with tabs[1]:
                st.subheader("Agent trace (harness telemetry)")
                rows = [{"step": e.step, "agent": e.agent, "action": e.action, "model": e.model,
                         "status": e.status, "latency_ms": e.latency_ms, "summary": e.summary} for e in state.trace]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("Model routing: light Groq model for extraction, larger Groq reasoning model for multi-constraint "
                           "judgment, classical CV for anything image-based, and exactly one Anthropic call — the final arbiter.")

            with tabs[2]:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Parsed customer program")
                    st.json(state.program.model_dump())
                    st.subheader("Raw customer request")
                    st.text(state.raw_rfp_text)
                with c2:
                    st.subheader("Site & zoning constraints")
                    st.json(state.site.model_dump())
                    st.subheader("Input site sketch (parsed by classical CV)")
                    st.image(str(config.SAMPLE_PROJECTS_DIR / "project_001/site_sketch.png"), width=320)

            with tabs[3]:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("BaaP module matching plan")
                    st.json(state.baap.model_dump())
                with c2:
                    st.subheader("Conflict report")
                    st.json(state.conflicts.model_dump())

            with tabs[4]:
                st.subheader("Physics-style site simulation + CV QA loop")
                for i, (sim_stats, cv) in enumerate(zip(state.simulation_history, state.cv_metrics_history), start=1):
                    st.markdown(f"**Iteration {i}**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Steps run", sim_stats.steps_run)
                    c2.metric("Converged", "yes" if sim_stats.converged else "no")
                    c3.metric("Geometry passed", "yes" if cv.geometry_passed else "no")
                    c4.metric("Parking shortfall", cv.parking_shortfall)
                    img = config.OUTPUTS_DIR / project_id / f"site_plan_iter{i}.png"
                    if img.exists():
                        st.image(str(img))
                if state.critic_history:
                    st.subheader("Design critic suggestions")
                    for i, c in enumerate(state.critic_history, start=1):
                        st.write(f"Iteration {i}: {c.rationale}")
                        st.write(c.adjustments)
                if state.convergence_chart_path:
                    st.image(state.convergence_chart_path)

            with tabs[5]:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Ground floor")
                    gp = config.OUTPUTS_DIR / project_id / "floor_plate_ground.png"
                    if gp.exists():
                        st.image(str(gp))
                with c2:
                    st.subheader("Typical upper floor")
                    if state.floor_plate_image_path:
                        st.image(state.floor_plate_image_path)

            with tabs[6]:
                fd = state.final_decision
                st.subheader(f"Decision: {fd.decision}  (confidence {fd.confidence:.0%})")
                st.write(fd.narrative)
                st.markdown("**Unresolved issues**")
                for u in fd.unresolved_issues:
                    st.markdown(f"- {u}")
                st.markdown("**Recommended next steps**")
                for s in fd.recommended_next_steps:
                    st.markdown(f"- {s}")
                st.caption(f"This is the single Anthropic ({config.ANTHROPIC_MODEL}) call in the whole pipeline.")

            with tabs[7]:
                st.subheader("Ask any agent, about any part of the project, at any time")
                agent_choice = st.selectbox("Agent", ["zoning", "baap_matching", "conflict", "intake", "design_critic"])
                question = st.text_input("Question", value="What is driving the parking shortfall and what are our options?")
                if st.button("Ask"):
                    with st.spinner("Thinking..."):
                        st.write(ask_agent(state, agent_choice, question))

    # ---------------------------------------------------------- IFC mode -
    elif mode == "IFC Design Studio":
        studio, onebr = ss.studio / 100, ss.onebr / 100
        params = Params(units=ss.units, floors=ss.floors, zoning=ss.zoning, parking=ss.parking, wings=ss.wings,
                        mix=(studio, onebr, max(0.0, 1 - studio - onebr)))

        @st.cache_data(show_spinner="Generating IFC design...")
        def _build(p: Params):
            lay = generate(p)
            ifc = write_ifc(lay, Path(tempfile.gettempdir()) / "rngd_design.ifc")
            return lay, ifc.read_bytes(), read_meshes(ifc)

        with st.status("Generating design from requirements + zoning + BaaP catalog...", expanded=False) as gstatus:
            gstatus.write(":material/straighten: computing buildable envelope from setbacks/easement")
            gstatus.write(":material/apartment: interleaving unit mix and packing double-loaded corridors")
            gstatus.write(":material/local_parking: placing driveway and surface/podium parking")
            lay, ifc_bytes, meshes = _build(params)
            gstatus.write(":material/description: writing IFC4 file and re-reading it to build the 3D view")
            gstatus.update(label="Design ready", state="complete")

        ss.ifc_layout, ss.ifc_params = lay, params
        m = lay.metrics
        fails = [c for c in lay.checks if c["status"] != "PASS"]

        c = st.columns(6)
        c[0].metric("Units", m["units_placed"], f"of {params.units}")
        c[1].metric("Stories", m["stories"])
        c[2].metric("FAR", m["far"])
        c[3].metric("Coverage", f"{m['coverage_pct']}%")
        c[4].metric("Parking", f"{m['parking_provided']}/{m['parking_required']}")
        c[5].metric("Checks failed", len(fails))

        tab3d, tabplan, tabchk, tabibc, tabexplain, tabdl = st.tabs(
            ["3D model", "Site plan", "Compliance (sample zoning)", "Real IBC check", "How this was built", "Download"])

        with tab3d:
            lv = None if ss.level_view == "all" else int(ss.level_view)
            st.plotly_chart(build_figure(meshes, lv, show_site=ss.show_site), use_container_width=True)
            st.info(":material/3d_rotation: **Drag** to rotate · **scroll / pinch** to zoom · **right-click drag** to pan · "
                    "**double-click** to reset the view · click a legend entry to hide/show that unit type. "
                    "Turn on \"Wide 3D view\" or \"Show site context\" in the sidebar if you need more room or the full lot.", icon=":material/info:")

        with tabplan:
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.add_patch(Rectangle((0, 0), params.lot_w, params.lot_d, fill=False, lw=2))
            if params.easement_east:
                ax.add_patch(Rectangle((params.lot_w - params.easement_east, 0), params.easement_east, params.lot_d, color="#FFBF00", alpha=.8))
            e = lay.env
            ax.add_patch(Rectangle((e["x0"], e["y0"]), e["w"], e["d"], fill=False, ls="--", ec="#CC0000"))
            if lay.drive:
                d = lay.drive
                ax.add_patch(Rectangle((d["x"], d["y"]), d["w"], d["d"], color="#555555"))
            for stl in lay.stalls:
                ax.add_patch(Rectangle((stl["x"], stl["y"]), stl["w"], stl["d"], fc="#8C8C8C", ec="white", lw=.4))
            for w in lay.wings:
                ax.add_patch(Rectangle((w["x"], w["y"]), w["w"], w["h"], fc="#3E7CB1", alpha=.85, ec="black"))
            ax.set_xlim(-10, params.lot_w + 10)
            ax.set_ylim(params.lot_d + 10, -10)
            ax.set_aspect("equal")
            ax.set_title("Top view: street at top")
            st.pyplot(fig)

        with tabchk:
            df = pd.DataFrame(lay.checks).astype(str)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption(f"Module counts requested: {module_counts(params)}. Zoning: {ZONING[params.zoning]}")
            if st.button(":material/forum: Ask the assistant to review this design", use_container_width=True):
                with st.spinner("Assistant reviewing..."):
                    note, _ = chat.suggest_after_ifc(params, lay, vs)
                ss.chat_history.append({"role": "assistant", "content": f"**IFC design review ({params.units} units, {params.zoning}).**\n\n{note}"})
                st.rerun()

        with tabibc:
            ibc_rows = code_check.evaluate(params, lay)
            if not ibc_rows:
                st.warning("docs/ not found — add RNGD's IBC rule files to enable this check.")
            else:
                st.caption(f"Cross-checked against {len(ibc_rows)} of RNGD's real, machine-testable 2021 IBC rules "
                           f"(docs/RNGD_IBC_2021_R1_Five_Story_AI_Rules_v0_2.csv) for their five-story R-1 hotel "
                           f"prototype — not the synthetic zoning preset used elsewhere. "
                           f"{code_check.not_yet_checked_count()} further rules in that set need space-level occupancy "
                           f"modeling (assembly/mercantile/kitchen areas, egress capacity) this prototype doesn't build yet.")
                ibc_df = pd.DataFrame(ibc_rows)[["rule_id", "rule_name", "section", "required", "measured", "status", "note"]].astype(str)
                st.dataframe(ibc_df, hide_index=True, use_container_width=True)
                n_fail = sum(1 for r in ibc_rows if r["status"] == "FAIL")
                if n_fail:
                    st.error(f"{n_fail} real-code check(s) failed — see the chat panel's summary card for a colored breakdown.")
                else:
                    st.success("All cross-checked real IBC rules pass for this layout.")

        with tabexplain:
            st.markdown("Step-by-step account of this specific run (see `reports/ifc_design/ifc_design_report.pdf` for the full method):")
            envd = lay.env
            steps = [
                f"**Envelope.** Lot {params.lot_w:.0f}x{params.lot_d:.0f} ft minus setbacks/easement -> "
                f"buildable {envd['w']:.0f}x{envd['d']:.0f} ft ({envd['w'] * envd['d']:,.0f} sf).",
                f"**Unit mix.** {params.units} units at {ss.studio}/{ss.onebr}/{100 - ss.studio - ss.onebr}% -> "
                f"{module_counts(params)}.",
                f"**Wings.** Tried 1-3 wings, kept **{m['wings']}** (best score: units placed, inside envelope, "
                f"has fire lane, fewest wings).",
                f"**Parking.** Zoning ratio {ZONING[params.zoning]['parking']}/unit -> {m['parking_required']} stalls required; "
                f"{m['parking_provided']} achieved ({'podium + surface' if params.parking == 'podium' else 'surface scan of free space'}).",
                f"**Result.** {m['units_placed']}/{params.units} units placed, FAR {m['far']} "
                f"(limit {ZONING[params.zoning]['far']}), coverage {m['coverage_pct']}% (limit {ZONING[params.zoning]['cov']}%), "
                f"{len(fails)} check(s) failed." if fails else f"**Result.** All ten compliance checks pass.",
            ]
            for i, s in enumerate(steps, 1):
                st.markdown(f"{i}. {s}")

        with tabdl:
            st.download_button("Download .ifc", ifc_bytes, "rngd_design.ifc", "application/x-step")
            html = build_figure(meshes).to_html(include_plotlyjs="cdn")
            st.download_button("Download 3D viewer (.html, opens in Chrome)", html, "rngd_design_3d.html", "text/html")
            st.caption("The .ifc opens in any BIM tool (BIMvision, Blender-BIM, FreeCAD, Autodesk Viewer).")

    # ------------------------------------------------------ explorer mode -
    else:
        if explorer_load_clicked and explorer_path is not None:
            with st.spinner(f"Opening {explorer_path.name} — triangulating geometry (first time only, can take a while)..."):
                summary, meshes = explorer.load(explorer_path, skip_framing=skip_framing)
            ss.explorer_summary, ss.explorer_meshes, ss.explorer_file = summary, meshes, explorer_path.name

        if ss.explorer_summary is None:
            st.info("Pick a file in the sidebar and click **Load / open file**.")
            if available:
                st.caption(f"Available: {', '.join(p.name for p in available)}")
        else:
            summary = ss.explorer_summary
            fingerprint = ss.explorer_file
            if ss.explorer_suggested_for != fingerprint:
                with st.spinner("Assistant reviewing the model..."):
                    note, _ = chat.suggest_after_explorer(summary, ss.explorer_file, vs)
                ss.chat_history.append({"role": "assistant", "content": f"**Opened `{ss.explorer_file}`.**\n\n{note}"})
                ss.explorer_suggested_for = fingerprint

            c = st.columns(5)
            c[0].metric("Schema", summary["schema"])
            c[1].metric("Storeys", len(summary["storeys"]))
            c[2].metric("Elements", summary["total_elements"])
            c[3].metric("Rendered", summary.get("meshes_triangulated", len(ss.explorer_meshes)))
            c[4].metric("Project", summary.get("project_name") or "—")

            etab3d, etabsum, etabbrowse = st.tabs(["3D model", "Summary", "Element browser"])

            with etab3d:
                fig = build_figure_generic(ss.explorer_meshes, classes=ss.get("explorer_classes") or None)
                st.plotly_chart(fig, use_container_width=True)
                st.info(":material/3d_rotation: **Drag** to rotate · **scroll / pinch** to zoom · **right-click drag** to pan · "
                        "**double-click** to reset the view · click a legend entry to hide/show a category. This is the REAL "
                        "model's actual geometry, re-read from the .ifc file. Turn on \"Wide 3D view\" in the sidebar for more room, "
                        "or narrow \"Show categories\" in the sidebar if it looks cluttered.", icon=":material/info:")

            with etabsum:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Storeys")
                    st.dataframe(pd.DataFrame(summary["storeys"]), hide_index=True, use_container_width=True)
                    st.subheader("Element categories")
                    st.dataframe(pd.DataFrame(sorted(summary["categories"].items(), key=lambda x: -x[1]), columns=["IFC class", "count"]),
                                 hide_index=True, use_container_width=True)
                with col2:
                    st.subheader("Notable named items")
                    st.dataframe(pd.DataFrame(summary["notable_named_elements"], columns=["name", "count"]),
                                 hide_index=True, use_container_width=True)
                    st.caption("Grouped by base name (before any ':' suffix) across furnishing, proxy and flow-terminal elements.")
                if summary.get("excluded_outliers"):
                    st.subheader("Excluded far-off outliers")
                    st.dataframe(pd.DataFrame(summary["excluded_outliers"]), hide_index=True, use_container_width=True)
                    st.caption(f"The model was recentered on its structural elements (walls/slabs/roof/doors/windows/columns) so the "
                               f"building fills the 3D view; the {len(summary['excluded_outliers'])} item(s) above sit implausibly far from "
                               f"everything else (e.g. a misplaced asset in the source file) and were left out of the 3D view. Nothing was "
                               f"changed in the original .ifc file on disk.")

            with etabbrowse:
                q = st.text_input("Search element names", "")
                rows = [{"name": m["name"], "class": m["cls"]} for m in ss.explorer_meshes if not q or q.lower() in m["name"].lower()]
                st.dataframe(pd.DataFrame(rows[:500]), hide_index=True, use_container_width=True)
                st.caption(f"{len(rows)} matching elements (showing up to 500).")

# ================================================================ chat ===
MATCH_COLOR = lambda p: "var(--ok)" if p >= 80 else ("var(--high)" if p >= 50 else "var(--critical)")


def _render_card(card: summary_card.SummaryCard) -> None:
    html = [f'<div class="rngd-card"><div class="card-top"><span class="card-title">{card.title}</span>'
            f'<span class="card-badge">{card.badge}</span></div><div class="card-headline">{card.headline}</div>']
    if card.match_pct is not None:
        color = MATCH_COLOR(card.match_pct)
        html.append(
            f'<div class="rngd-match"><div class="bar"><div class="fill" style="width:{card.match_pct:.0f}%;background:{color}"></div></div>'
            f'<div class="label"><span>Match to requirements</span><span style="color:{color}">{card.match_pct:.0f}%</span></div></div>'
        )
    if card.metrics:
        html.append('<div class="rngd-kv">' + "".join(f'<div><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in card.metrics) + "</div>")
    if card.setbacks:
        html.append('<div class="rngd-sec-label">Setbacks / envelope</div><div class="rngd-kv">' +
                    "".join(f'<div><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in card.setbacks) + "</div>")
    html.append("</div>")
    if card.items:
        html.append(f'<div class="rngd-sec-label">Conflicts &amp; failed checks ({len(card.items)})</div>')
        for it in card.items[:12]:
            tag = f'<span class="rngd-tag">{it.tag}</span>' if it.tag else ""
            html.append(f'<div class="rngd-item {it.severity}"><div class="item-top"><span>{it.label}</span>{tag}</div>'
                        f'<div class="item-detail">{it.detail}</div></div>')
    st.markdown("".join(html), unsafe_allow_html=True)


QUICK_QUESTIONS = {
    "Multi-Agent Pipeline": {
        None: ["What does this pipeline do?", "Explain each agent", "What is FAR / a buildable envelope?"],
        "ready": ["Summarize this design", "What conflicts were found, and how severe?",
                 "How close is this to the client's requirements?", "What would fix the parking shortfall?"],
    },
    "IFC Design Studio": {
        "ready": ["Summarize this design", "Why did any checks fail?", "How does this compare to the real IBC rules?",
                 "What's the smallest change to make this compliant?"],
    },
    "Open IFC File": {
        "ready": ["What is in this building?", "List notable items in this model", "What got excluded from the 3D view, and why?"],
    },
}


def _quick_questions() -> list[str]:
    bucket = QUICK_QUESTIONS.get(mode, {})
    if mode == "Multi-Agent Pipeline":
        return bucket["ready"] if ss.pipeline_state else bucket[None]
    if mode == "IFC Design Studio":
        return bucket.get("ready", [])
    if mode == "Open IFC File":
        return bucket.get("ready", []) if ss.explorer_summary else ["What can this mode do?"]
    return []


def _ask(question: str) -> None:
    ss.chat_history.append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        reply_text, _ = chat.reply(ss.chat_history, vs, ss.pipeline_state, ss.ifc_params, ss.ifc_layout,
                                   ss.explorer_summary, ss.explorer_file)
    ss.chat_history.append({"role": "assistant", "content": reply_text})


with chat_col:
    st.markdown('<div class="rngd-chat-title">Assistant — online</div>', unsafe_allow_html=True)

    card = {"Multi-Agent Pipeline": summary_card.pipeline_card(ss.pipeline_state),
            "IFC Design Studio": summary_card.ifc_card(ss.ifc_params, ss.ifc_layout) if ss.ifc_params else None,
            "Open IFC File": summary_card.explorer_card(ss.explorer_summary, ss.explorer_file)}.get(mode)
    if card:
        _render_card(card)

    qs = _quick_questions()
    if qs:
        st.markdown('<div class="rngd-sec-label">Ask about this design</div>', unsafe_allow_html=True)
        qcols = st.columns(2)
        for i, q in enumerate(qs):
            if qcols[i % 2].button(q, key=f"qq_{mode}_{i}", use_container_width=True):
                _ask(q)
                st.rerun()

    chat_box = st.container(height=420)
    with chat_box:
        if not ss.chat_history:
            st.chat_message("assistant").write(
                "Hi! I can explain any part of this project — the pipeline agents, the physics "
                "simulation, the IFC Design Studio, the real IBC rules RNGD supplied, or your latest "
                "results. I'll also chime in with suggestions after a run finishes. Try one of the "
                "buttons above, or ask your own question below."
            )
        for msg in ss.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input("Ask about any component or result...")
    if prompt:
        _ask(prompt)
        st.rerun()
