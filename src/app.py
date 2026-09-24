"""RNGD Concept-to-Schematic Multi-Agent Harness — Streamlit demo.

Run: streamlit run src/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from rngd import config
from rngd.orchestrator import ask_agent, run_pipeline
from rngd.vectorstore import get_vectorstore

st.set_page_config(page_title="RNGD Concept-to-Schematic Harness", layout="wide")

if "state" not in st.session_state:
    st.session_state.state = None

st.title("RNGD Concept-to-Schematic Multi-Agent Harness")
st.caption(
    "Proof-of-concept: unstructured customer program + site sketch -> multi-agent BaaP "
    "schematic design, with a physics-style site simulation, classical computer-vision QA, "
    "and one final Anthropic Claude arbiter decision. All sample data is synthetic."
)

with st.sidebar:
    st.header("Run")
    project_id = st.selectbox("Sample project", options=["project_001"])
    max_iter = st.slider("Max simulation iterations", 1, 5, config.MAX_SIMULATION_ITERATIONS)
    run_clicked = st.button("Run full pipeline", type="primary")

    st.divider()
    st.header("Knowledge base")
    vs = get_vectorstore()
    st.caption(f"Backend: {vs.backend_name}")
    st.caption(f"{len(vs.chunks)} chunks indexed from data/knowledge_base/*.md")
    kb_query = st.text_input("Try a retrieval query", value="parking ratio per unit")
    if kb_query:
        for h in vs.retrieve(kb_query, k=3):
            with st.expander(f"{h['source']} | {h['heading']} (score {h['score']:.2f})"):
                st.text(h["text"])

if run_clicked:
    with st.spinner("Running multi-agent pipeline (intake -> zoning -> BaaP -> conflicts -> simulation/CV loop -> arbiter)..."):
        st.session_state.state = run_pipeline(project_id, max_iterations=max_iter)

state = st.session_state.state

if state is None:
    st.info("Click **Run full pipeline** in the sidebar to generate a schematic for the sample project.")
    st.subheader("Sample customer request (raw input)")
    rfp_path = config.SAMPLE_PROJECTS_DIR / project_id / "customer_rfp.txt"
    st.text(rfp_path.read_text(encoding="utf-8"))
    st.subheader("Sample site sketch (raw input)")
    st.image(str(config.SAMPLE_PROJECTS_DIR / project_id / "site_sketch.png"), width=400)
    st.stop()

tabs = st.tabs(
    [
        "Overview",
        "Agent trace",
        "Program & site",
        "BaaP plan & conflicts",
        "Site simulation",
        "Floor plates",
        "Final decision",
        "Ask an agent",
    ]
)

with tabs[0]:
    col1, col2, col3 = st.columns(3)
    col1.metric("Decision", state.final_decision.decision)
    col2.metric("Confidence", f"{state.final_decision.confidence:.0%}")
    col3.metric("Conflicts found", len(state.conflicts.conflicts) if state.conflicts else 0)
    st.write(state.final_decision.narrative)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Generated site plan")
        if state.site_plan_image_path:
            st.image(state.site_plan_image_path)
    with col2:
        st.subheader("Typical upper floor")
        if state.floor_plate_image_path:
            st.image(state.floor_plate_image_path)

with tabs[1]:
    st.subheader("Agent trace (harness telemetry)")
    rows = [
        {
            "step": e.step,
            "agent": e.agent,
            "action": e.action,
            "model": e.model,
            "status": e.status,
            "latency_ms": e.latency_ms,
            "summary": e.summary,
        }
        for e in state.trace
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Model routing: light/fast Groq model for structured extraction, larger Groq reasoning "
        "model for multi-constraint judgment, classical CV for anything image-based, and exactly "
        "one Anthropic Claude call — the final arbiter — for the highest-stakes synthesis step."
    )

with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Parsed customer program (Intake Agent)")
        st.json(state.program.model_dump())
        st.subheader("Raw customer request")
        st.text(state.raw_rfp_text)
    with col2:
        st.subheader("Site & zoning constraints (Zoning Agent)")
        st.json(state.site.model_dump())
        st.subheader("Input site sketch (parsed by classical CV)")
        sketch_path = config.SAMPLE_PROJECTS_DIR / project_id / "site_sketch.png"
        st.image(str(sketch_path), width=350)

with tabs[3]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("BaaP module matching plan")
        st.json(state.baap.model_dump())
    with col2:
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
        img_path = Path(config.OUTPUTS_DIR / project_id / f"site_plan_iter{i}.png")
        if img_path.exists():
            st.image(str(img_path))
    if state.critic_history:
        st.subheader("Design critic suggestions")
        for i, c in enumerate(state.critic_history, start=1):
            st.write(f"Iteration {i}: {c.rationale}")
            st.write(c.adjustments)
    if state.convergence_chart_path:
        st.image(state.convergence_chart_path)

with tabs[5]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ground floor")
        gp = Path(config.OUTPUTS_DIR / project_id / "floor_plate_ground.png")
        if gp.exists():
            st.image(str(gp))
    with col2:
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
    st.caption(
        "Not locked to the fixed pipeline order — this queries the knowledge base and the current "
        "project state directly, so you can interrogate the zoning rules, the BaaP plan, or the "
        "conflict list without re-running the whole pipeline."
    )
    agent_choice = st.selectbox(
        "Agent", ["zoning", "baap_matching", "conflict", "intake", "design_critic"]
    )
    question = st.text_input("Question", value="What is driving the parking shortfall and what are our options?")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            answer = ask_agent(state, agent_choice, question)
        st.write(answer)
