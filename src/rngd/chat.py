"""The persistent chat assistant shown in the app's right-hand panel.

Unlike `orchestrator.ask_agent` (one question -> one agent's domain), this is
a general-purpose, multi-turn assistant that knows about *every* component in
the project — both the concept-to-schematic pipeline and the IFC design
studio — so it can explain what something is, why a result came out the way
it did, or suggest what to try next, whether or not a run has happened yet.
"""
from __future__ import annotations

from . import config, llm
from .ifc.design import ZONING
from .schemas import ProjectState
from .vectorstore import VectorStore

SYSTEM_PROMPT = """You are the built-in assistant for RNGD's concept-to-schematic AI harness, \
shown as a chat panel next to the app. You help a teammate understand the system, interpret \
results, and decide what to try next. Be concise (short paragraphs or bullet points), concrete, \
and honest about what is synthetic/sample data versus a real result of this run.

## The system has four parts

**1. Multi-agent concept-to-schematic pipeline** (`src/rngd/`) — turns an unstructured customer \
request + site conditions into a BaaP schematic:
- Vision Input Agent (classical computer vision): reads the site sketch PNG, cross-checks it \
against the stated site_conditions.json.
- Intake Agent (light model): parses the free-text customer request into a structured \
program (unit count, mix, floors, amenities, hard vs. soft requirements).
- Zoning Agent (LLM + RAG over the knowledge base): retrieves the applicable sample zoning \
rules and computes the buildable envelope (setbacks, easements) in plain Python arithmetic.
- BaaP Matching Agent (LLM + RAG): converts the unit mix into exact module counts from the \
BaaP catalog and proposes an adjacency plan.
- Conflict Agent (LLM): computes feasibility numbers (FAR, lot coverage, parking demand vs. \
available land) and flags conflicts with severities.
- Physics Simulation Engine: a 2D rigid-body relaxation (containment + separation forces) that \
places the building footprint, driveway and parking within the buildable envelope.
- Vision QA Agent (classical computer vision): re-measures the *rendered* site plan image \
independently, checking setback/easement/overlap violations and counting parking stalls.
- Design Critic Agent (LLM): decides whether another simulation pass can fix a violation or \
whether it's a program/site conflict no geometry change will resolve.
- Floor Plate Packer (rule-based, not AI): lays out unit modules along a double-loaded corridor \
for a ground floor and a typical upper floor.
- Arbiter Agent (a Claude model via OpenRouter — the ONE expensive call per run): synthesizes everything into \
a final decision (APPROVE / APPROVE_WITH_CONDITIONS / REVISE / INSUFFICIENT_INFORMATION).

**2. IFC Design Studio** (`src/rngd/ifc/`) — a separate, faster, parametric generator: \
requirements + a zoning preset + the BaaP catalog -> a rule-based building layout -> a real IFC4 \
file (via IfcOpenShell) -> an interactive 3D view in the browser. It scores every design against \
ten pass/fail checks (units placed, stories, height, FAR, lot coverage, parking stalls, inside \
setbacks/easement, fire lane, accessible units, exit travel distance). It does not call any LLM \
by default (it's deterministic), except the optional "extract requirements from text" button \
which reuses the Intake Agent.

**4. Hotel Schematic** (`src/rngd/hotel/`) - RNGD's first use case (hotels). A detailed program (default: the Project Renegade Living + Lodging cover-sheet space allocation, 28 rows, 117 keys; or an uploaded / edited CSV/XLSX) -> a deterministic double-loaded-corridor layout (Level 1 podium with lobby, BOH and retail; guest floors above; core and Stair 1 west, Stair 2 east) -> a code screen against RNGD's real IBC rule CSV and the 2010 ADA guest-room tables -> an 18x12 PDF drawing set in RNGD's L+L sheet format (overview, plans, elevations, enlarged unit plans, unit 3D views) plus appendices. When the user asks about the hotel, use the Hotel Schematic state below.

**3. Open IFC File** (`src/rngd/ifc/explorer.py`) — opens a REAL, arbitrary `.ifc` file (not one \
this app generated), triangulates its geometry for the 3D viewer, and summarizes what's in it \
(schema, storeys, element counts by IFC class, notable named items) by reading the IFC file's own \
data — this is a real building's actual model, not synthetic sample data. It does not run the \
BaaP/zoning pipeline (a real file has whatever building type it has); it's for viewing and asking \
questions about a specific real model.

**4. Real RNGD documents** (`docs/`, indexed into the same knowledge base, tagged `source_kind: "real"`) \
— RNGD's own structured 2021 IBC rule database for their five-story R-1 hotel prototype (42 \
machine-testable rules with operator/value/units, occupant-load factors, hotel space mapping, \
reference projects) plus the official DOJ ADA 2010 Standards PDF. `src/rngd/ifc/code_check.py` \
cross-checks a generated IFC Design Studio layout against ~8 of those real rules it can actually \
compute from the layout (height, stories, exit travel distance, dead-end corridor length, exits \
per story via an occupant-load estimate) — shown in the Design Studio's "Real IBC check" tab and \
in the chat panel's summary card, clearly separate from the synthetic zoning preset's own ten \
checks. The other ~34 real rules need space-level occupancy modeling not yet built.

## Ground rules
- The zoning PRESETS (Rivermont etc.) and the BaaP catalog are SYNTHETIC SAMPLE DATA. The IBC \
rules and ADA standards from `docs/` are REAL — RNGD's own data. Always be clear which is which.
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
    if summary.get("excluded_outliers"):
        parts.append(f"- Note: {len(summary['excluded_outliers'])} element(s) were excluded from the 3D view as far-off "
                     f"outliers (e.g. a misplaced asset in the source file), so the view frames the actual building tightly.")
    return "\n".join(parts)


def summarize_ifc_state(params, layout) -> str:
    if layout is None:
        return "No IFC design generated yet in this session."
    m = layout.metrics
    fails = [c["check"] for c in layout.checks if c["status"] != "PASS"]
    parts = [
        f"IFC Design Studio, current design: {params.units} units requested, {params.floors} floors, "
        f"zoning preset '{params.zoning}' (SYNTHETIC), parking={params.parking}, wings={m['wings']}.",
        f"Metrics: units placed {m['units_placed']}, stories {m['stories']}, FAR {m['far']}, "
        f"lot coverage {m['coverage_pct']}%, parking {m['parking_provided']}/{m['parking_required']}.",
        f"Sample-zoning failed checks: {fails or 'none — this design passes all ten sample checks'}.",
    ]
    try:
        from .ifc import code_check
        ibc = code_check.evaluate(params, layout)
        ibc_fails = [f"{r['rule_id']} ({r['rule_name']}): required {r['required']}, measured {r['measured']}"
                     for r in ibc if r["status"] == "FAIL"]
        parts.append(f"Real IBC cross-check (RNGD's own rule data, {len(ibc)} rules checked, "
                     f"{code_check.not_yet_checked_count()} not yet checked): " +
                     (f"FAILED: {ibc_fails}" if ibc_fails else "all checked rules pass."))
    except Exception:
        pass
    return "\n".join(parts)


def summarize_hotel_state(sch) -> str:
    if sch is None:
        return "Hotel Schematic mode not opened yet."
    from .hotel.details import mix_by_floor

    p = sch.program
    rows = "; ".join(f"{n} x{c} ({a:,.0f} SF, {cat})" for n, c, a, cat in p.rows)
    checks = "; ".join(f"{c['rule_id']} {c['check']}: required {c['required']}, provided {c['provided']} -> {c['status']}"
                       + (f" ({c['note']})" if c.get("note") else "") for c in sch.checks)
    mix = mix_by_floor(sch)
    floors = "; ".join(f"{col}: " + ", ".join(f"{t} {int(v)}" for t, v in mix[col].items() if v and t != "Keys per floor")
                       for col in mix.columns if col != "Total")
    return (f"Hotel Schematic (RNGD Living + Lodging style), program source: {p.source}. {p.keys} keys, net {p.net_sf:,.0f} SF, "
            f"source gross {p.gross_sf:,.0f} SF. Massing: Level 1 {sch.levels[0].length_ft:.0f} x {sch.levels[0].width_ft:.0f} ft, "
            f"{len(sch.levels) - 1} guest floors {sch.levels[1].length_ft:.0f} x {sch.levels[1].width_ft:.0f} ft, double-loaded "
            f"6 ft corridor, Stair 1 + 2 elevators at the west core, Stair 2 at the east end, height {sch.height_ft:.1f} ft. "
            f"Envelope (synthetic) {sch.site['envelope_d']} x {sch.site['envelope_w']} ft.\nProgram rows: {rows}.\n"
            f"Room mix by floor: {floors}.\nCode screen (RNGD IBC CSV + 2010 ADA tables): {checks}.")


def summarize_optimizer_state(result: dict | None) -> str:
    if not result:
        return "Design Optimizer not run yet."
    m = result["metrics"]
    revs = "; ".join(f"gen {r['gen']} by {r.get('engine')}: scores {[c['llm_score'] for c in r['candidates']]}, "
                     f"weights {r.get('weight_multipliers')}, rationale: {str(r.get('rationale'))[:200]}"
                     for r in result.get("reviews", []))
    ref = result.get("reference")
    refs = (f" Reference design {ref['file']} (same evaluator): {ref['metrics']['rooms_placed']}/{ref['metrics']['rooms_total']} "
            f"rooms, fitness {ref['metrics']['fitness']}, dead end {ref['metrics']['dead_end_ft']} ft, failed checks "
            f"{[c['rule_id'] for c in ref['metrics']['checks'] if c['status'] == 'FAIL']}." if ref else "")
    return (f"Design Optimizer result for site {result['site']} with program {result['program']}: memetic genetic algorithm "
            f"(corridor genome + room-order keys, greedy packing decoder, local search) with an LLM/RAG design reviewer in "
            f"the fitness. Best design: {m['rooms_placed']}/{m['rooms_total']} rooms, {m['keys_placed']}/{m['keys_requested']} keys, "
            f"fitness {m['fitness']}, objectives {m['terms']}, final weights {m['weights']}, corridor ratio {m['corridor_ratio']}, "
            f"windows {m['exterior_fraction']}, max travel {m['max_travel_ft']} ft, dead end {m['dead_end_ft']} ft, "
            f"unplaced {m['unplaced_by_type']}, per level {[(lv['level'], lv['by_type']) for lv in m['levels']]}, "
            f"checks {[(c['rule_id'], c['status']) for c in m['checks']]}. Reviews: {revs}.{refs}")


def reply(
    history: list[dict[str, str]],
    vs: VectorStore,
    pipeline_state: ProjectState | None,
    ifc_params=None,
    ifc_layout=None,
    explorer_summary: dict | None = None,
    explorer_file_label: str | None = None,
    hotel_schematic=None,
    optimizer_result: dict | None = None,
) -> tuple[str, float]:
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    kb_context = vs.retrieve_as_context(last_user, k=3) if last_user else ""
    context = (
        f"{zoning_reference()}\n\n"
        f"{summarize_pipeline_state(pipeline_state)}\n\n"
        f"{summarize_ifc_state(ifc_params, ifc_layout) if ifc_params else 'IFC Design Studio not opened yet.'}\n\n"
        f"{summarize_explorer_state(explorer_summary, explorer_file_label)}\n\n"
        f"{summarize_hotel_state(hotel_schematic)}\n\n"
        f"{summarize_optimizer_state(optimizer_result)}\n\n"
        f"Retrieved knowledge-base context for the latest question:\n{kb_context}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n## Current state\n" + context},
        *history,
    ]
    return llm.call_chat(config.LLM_REASONING, messages)


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
