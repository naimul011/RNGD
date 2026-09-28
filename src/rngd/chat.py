"""The persistent chat assistant shown in the app's right-hand panel.

Unlike `orchestrator.ask_agent` (one question -> one agent's domain), this is
a general-purpose, multi-turn assistant that knows about *every* component in
the project — both the concept-to-schematic pipeline and the IFC design
studio — so it can explain what something is, why a result came out the way
it did, or suggest what to try next, whether or not a run has happened yet.
"""
from __future__ import annotations

from . import config, llm_groq
from .ifc.design import ZONING
from .schemas import ProjectState
from .vectorstore import VectorStore

SYSTEM_PROMPT = """You are the built-in assistant for RNGD's concept-to-schematic AI harness, \
shown as a chat panel next to the app. You help a teammate understand the system, interpret \
results, and decide what to try next. Be concise (short paragraphs or bullet points), concrete, \
and honest about what is synthetic/sample data versus a real result of this run.

## The system has two parts

**1. Multi-agent concept-to-schematic pipeline** (`src/rngd/`) — turns an unstructured customer \
request + site conditions into a BaaP schematic:
- Vision Input Agent (classical computer vision): reads the site sketch PNG, cross-checks it \
against the stated site_conditions.json.
- Intake Agent (Groq, fast model): parses the free-text customer request into a structured \
program (unit count, mix, floors, amenities, hard vs. soft requirements).
- Zoning Agent (Groq + RAG over the knowledge base): retrieves the applicable sample zoning \
rules and computes the buildable envelope (setbacks, easements) in plain Python arithmetic.
- BaaP Matching Agent (Groq + RAG): converts the unit mix into exact module counts from the \
BaaP catalog and proposes an adjacency plan.
- Conflict Agent (Groq): computes feasibility numbers (FAR, lot coverage, parking demand vs. \
available land) and flags conflicts with severities.
- Physics Simulation Engine: a 2D rigid-body relaxation (containment + separation forces) that \
places the building footprint, driveway and parking within the buildable envelope.
- Vision QA Agent (classical computer vision): re-measures the *rendered* site plan image \
independently, checking setback/easement/overlap violations and counting parking stalls.
- Design Critic Agent (Groq): decides whether another simulation pass can fix a violation or \
whether it's a program/site conflict no geometry change will resolve.
- Floor Plate Packer (rule-based, not AI): lays out unit modules along a double-loaded corridor \
for a ground floor and a typical upper floor.
- Arbiter Agent (Anthropic Claude — the ONE expensive call per run): synthesizes everything into \
a final decision (APPROVE / APPROVE_WITH_CONDITIONS / REVISE / INSUFFICIENT_INFORMATION).

**2. IFC Design Studio** (`src/rngd/ifc/`) — a separate, faster, parametric generator: \
requirements + a zoning preset + the BaaP catalog -> a rule-based building layout -> a real IFC4 \
file (via IfcOpenShell) -> an interactive 3D view in the browser. It scores every design against \
ten pass/fail checks (units placed, stories, height, FAR, lot coverage, parking stalls, inside \
setbacks/easement, fire lane, accessible units, exit travel distance). It does not call any LLM \
by default (it's deterministic), except the optional "extract requirements from text" button \
which reuses the Intake Agent.

**3. Open IFC File** (`src/rngd/ifc/explorer.py`) — opens a REAL, arbitrary `.ifc` file (not one \
this app generated), triangulates its geometry for the 3D viewer, and summarizes what's in it \
(schema, storeys, element counts by IFC class, notable named items) by reading the IFC file's own \
data — this is a real building's actual model, not synthetic sample data. It does not run the \
BaaP/zoning pipeline (a real file has whatever building type it has); it's for viewing and asking \
questions about a specific real model.

## Ground rules
- All zoning rules and the BaaP catalog are SYNTHETIC SAMPLE DATA (a fictional "Rivermont" \
jurisdiction etc.), standing in for RNGD's real data. Say so if it's relevant to the question.
- Every output here is planning-level, not code-approved; a licensed professional must review \
any real design.
- If asked about a specific run's results, use the CURRENT STATE context given below. If no run \
has happened yet, say so and suggest running one rather than inventing numbers.
- If a question is about a term (FAR, easement, double-loaded corridor, BaaP, IFC...), just \
explain it plainly.
"""


def zoning_reference() -> str:
    lines = ["Available zoning presets (synthetic samples):"]
    for name, z in ZONING.items():
        lines.append(
            f"- {name}: setbacks front/side/rear {z['front']}/{z['side']}/{z['rear']} ft, "
            f"FAR {z['far']}, lot coverage {z['cov']}%, height {z['height']} ft, "
            f"{z['stories']} stories max, parking {z['parking']}/unit"
        )
    return "\n".join(lines)


def summarize_pipeline_state(state: ProjectState | None) -> str:
    if state is None:
        return "No pipeline run yet in this session."
    parts = [f"Pipeline run for project '{state.project_id}':"]
    if state.program:
        parts.append(f"- Program: {state.program.unit_count_total} units, {state.program.floor_count} floors, "
                      f"mix {state.program.unit_mix_pct}, hard requirements: {state.program.hard_requirements}")
    if state.site and state.site.buildable_envelope:
        e = state.site.buildable_envelope
        parts.append(f"- Buildable envelope: {e.width_ft:.0f} x {e.depth_ft:.0f} ft ({e.area_sf:.0f} sf)")
    if state.baap:
        parts.append(f"- BaaP plan: {state.baap.module_counts}, {state.baap.estimated_building_gsf:.0f} GSF")
    if state.conflicts:
        parts.append(f"- Conflicts ({len(state.conflicts.conflicts)}): " +
                      "; ".join(f"[{c.severity}] {c.description}" for c in state.conflicts.conflicts))
    if state.cv_metrics_history:
        cv = state.cv_metrics_history[-1]
        parts.append(f"- Latest CV QA: geometry_passed={cv.geometry_passed}, lot coverage {cv.lot_coverage_pct}%, "
                      f"parking {cv.parking_stalls_detected}/{cv.parking_stalls_required} (shortfall {cv.parking_shortfall})")
    if state.final_decision:
        fd = state.final_decision
        parts.append(f"- FINAL DECISION: {fd.decision} (confidence {fd.confidence:.0%}). {fd.narrative}")
        parts.append(f"- Unresolved issues: {fd.unresolved_issues}")
    return "\n".join(parts)


def summarize_explorer_state(summary: dict | None, file_label: str | None) -> str:
    if summary is None:
        return "No real IFC file opened in the Explorer yet."
    parts = [f"Open IFC File: '{file_label}' — REAL model data (not synthetic), schema {summary['schema']}."]
    if summary.get("project_name"):
        parts.append(f"- Project: {summary['project_name']}")
    parts.append(f"- Storeys: {[s['name'] for s in summary['storeys']]}")
    parts.append(f"- Element categories: {summary['categories']}")
    parts.append(f"- Total elements: {summary['total_elements']} ({summary.get('meshes_triangulated', '?')} rendered"
                 f"{'; curtain-wall framing (Members/Plates) hidden for speed' if summary.get('framing_skipped') else ''})")
    if summary.get("notable_named_elements"):
        top = ", ".join(f"{name} x{n}" for name, n in summary["notable_named_elements"][:10])
        parts.append(f"- Notable named items: {top}")
    return "\n".join(parts)


def summarize_ifc_state(params, layout) -> str:
    if layout is None:
        return "No IFC design generated yet in this session."
    m = layout.metrics
    fails = [c["check"] for c in layout.checks if c["status"] != "PASS"]
    return (
        f"IFC Design Studio, current design: {params.units} units requested, {params.floors} floors, "
        f"zoning preset '{params.zoning}', parking={params.parking}, wings={m['wings']}.\n"
        f"Metrics: units placed {m['units_placed']}, stories {m['stories']}, FAR {m['far']}, "
        f"lot coverage {m['coverage_pct']}%, parking {m['parking_provided']}/{m['parking_required']}.\n"
        f"Failed checks: {fails or 'none — this design passes all checks'}."
    )


def reply(
    history: list[dict[str, str]],
    vs: VectorStore,
    pipeline_state: ProjectState | None,
    ifc_params=None,
    ifc_layout=None,
    explorer_summary: dict | None = None,
    explorer_file_label: str | None = None,
) -> tuple[str, float]:
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    kb_context = vs.retrieve_as_context(last_user, k=3) if last_user else ""
    context = (
        f"{zoning_reference()}\n\n"
        f"{summarize_pipeline_state(pipeline_state)}\n\n"
        f"{summarize_ifc_state(ifc_params, ifc_layout) if ifc_params else 'IFC Design Studio not opened yet.'}\n\n"
        f"{summarize_explorer_state(explorer_summary, explorer_file_label)}\n\n"
        f"Retrieved knowledge-base context for the latest question:\n{kb_context}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n## Current state\n" + context},
        *history,
    ]
    return llm_groq.call_chat(config.GROQ_MODEL_REASONING, messages)


def suggest_after_pipeline(state: ProjectState, vs: VectorStore) -> tuple[str, float]:
    prompt = (
        "The pipeline run you were watching just finished. Write a short (3-5 sentence) proactive "
        "note to the teammate: what the headline result means, and one concrete, specific thing "
        "they could try next in either the pipeline or the IFC Design Studio to move it forward. "
        "Do not repeat the full narrative verbatim, add insight."
    )
    return reply([{"role": "user", "content": prompt}], vs, state)


def suggest_after_explorer(summary: dict, file_label: str, vs: VectorStore) -> tuple[str, float]:
    prompt = (
        "You just finished loading a REAL IFC file (not synthetic). Write a short (3-5 sentence) "
        "orientation note: what kind of building this looks like, what's notable about it, and one "
        "specific question the user might want to ask next about this model."
    )
    return reply([{"role": "user", "content": prompt}], vs, None, explorer_summary=summary, explorer_file_label=file_label)


def suggest_after_ifc(params, layout, vs: VectorStore) -> tuple[str, float]:
    prompt = (
        "The design you were watching in the IFC Design Studio just finished generating. Write a "
        "short (3-5 sentence) proactive note: what the compliance results mean, and one concrete "
        "parameter change worth trying next (which control to move and why)."
    )
    return reply([{"role": "user", "content": prompt}], vs, None, ifc_params=params, ifc_layout=layout)
