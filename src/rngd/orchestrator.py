"""The harness: a shared-state (blackboard) pipeline over all agents, plus an
`ask_agent` entry point so any agent can be queried on demand against the
current project state and knowledge base, not only as part of the fixed
pipeline order.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config, llm_groq
from .agents import arbiter_agent, baap_matching_agent, conflict_agent, design_critic_agent, intake_agent, zoning_agent
from .schemas import CVMetrics, ProjectState, SimulationStepStats
from .simulation import floor_plate, renderer
from .simulation.physics_engine import Body, SiteSimulation, build_site_bodies
from .trace import reset_step_counter, traced_call
from .vectorstore import VectorStore, get_vectorstore
from .vision import layout_qa, site_sketch_parser

FT_PER_STALL_AREA_ALLOWANCE = 330


def _project_dir(project_id: str) -> Path:
    return config.SAMPLE_PROJECTS_DIR / project_id


def _outputs_dir(project_id: str) -> Path:
    d = config.OUTPUTS_DIR / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_raw_inputs(project_id: str) -> tuple[str, dict]:
    pdir = _project_dir(project_id)
    rfp_text = (pdir / "customer_rfp.txt").read_text(encoding="utf-8")
    site_conditions = json.loads((pdir / "site_conditions.json").read_text())
    return rfp_text, site_conditions


def run_vision_input_check(project_id: str, state: ProjectState) -> dict:
    pdir = _project_dir(project_id)
    png, georef = pdir / "site_sketch.png", pdir / "site_sketch_georef.json"
    with traced_call(state, "vision_input", "parse_site_sketch", "classical-cv") as t:
        measured = site_sketch_parser.parse_site_sketch(png, georef)
        cross_check = site_sketch_parser.cross_check_against_spec(measured, state.raw_site_conditions)
        t.summary = "consistent" if cross_check["consistent"] else f"{len(cross_check['discrepancies'])} discrepancy(ies)"
        t.details = {"measured": measured["measurements"], "cross_check": cross_check}
    return {"measured": measured, "cross_check": cross_check}


def run_pipeline(
    project_id: str = "project_001",
    max_iterations: int | None = None,
    on_step=None,
) -> ProjectState:
    """`on_step`, if given, is called with the latest AgentTraceEntry right
    after each stage completes — the hook the UI uses to show live progress
    (st.status updates) during a run instead of only a trace table at the end.
    """
    reset_step_counter()
    max_iterations = max_iterations or config.MAX_SIMULATION_ITERATIONS
    vs = get_vectorstore()
    rfp_text, site_conditions = load_raw_inputs(project_id)
    state = ProjectState(project_id=project_id, raw_rfp_text=rfp_text, raw_site_conditions=site_conditions)

    def _fire():
        if on_step:
            on_step(state.trace[-1])

    vision_input = run_vision_input_check(project_id, state)
    _fire()

    with traced_call(state, "intake", "parse_customer_program", config.MODEL_ROUTE["intake"]) as t:
        state.program, latency = intake_agent.run(rfp_text)
        t.summary = f"{state.program.unit_count_total} units, {state.program.floor_count} floors"
        t.details = {"latency_ms": latency}
    _fire()

    with traced_call(state, "zoning", "analyze_site_and_code", config.MODEL_ROUTE["zoning"]) as t:
        state.site, latency = zoning_agent.run(site_conditions, vs, cv_cross_check=vision_input["cross_check"])
        env = state.site.buildable_envelope
        t.summary = f"buildable envelope {env.width_ft:.0f} x {env.depth_ft:.0f} ft ({env.area_sf:.0f} sf)"
        t.details = {"latency_ms": latency}
    _fire()

    with traced_call(state, "baap_matching", "match_program_to_baap", config.MODEL_ROUTE["baap_matching"]) as t:
        state.baap, latency = baap_matching_agent.run(state.program, vs)
        t.summary = f"{sum(state.baap.module_counts.values())} unit modules, {state.baap.estimated_building_gsf:.0f} GSF"
        t.details = {"latency_ms": latency}
    _fire()

    with traced_call(state, "conflict", "check_program_site_baap_conflicts", config.MODEL_ROUTE["conflict"]) as t:
        state.conflicts, numbers, latency = conflict_agent.run(state.program, state.site, state.baap, vs)
        t.summary = f"{len(state.conflicts.conflicts)} conflict(s), feasible_as_requested={state.conflicts.feasible_as_requested}"
        t.details = {"latency_ms": latency, "numbers": numbers}
    _fire()

    _run_simulation_loop(state, numbers, max_iterations, on_step=on_step)
    _generate_floor_plates(state, on_step=on_step)

    with traced_call(state, "arbiter", "final_decision", config.ANTHROPIC_MODEL) as t:
        state.final_decision, latency = arbiter_agent.run(state)
        t.summary = f"{state.final_decision.decision} (confidence {state.final_decision.confidence:.2f})"
        t.details = {"latency_ms": latency}
    _fire()

    out_dir = _outputs_dir(project_id)
    (out_dir / "project_state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return state


def _run_simulation_loop(state: ProjectState, numbers: dict, max_iterations: int, on_step=None) -> None:
    def _fire():
        if on_step:
            on_step(state.trace[-1])

    site = state.site
    baap = state.baap
    env = site.buildable_envelope
    out_dir = _outputs_dir(state.project_id)

    footprint_w = min(env.width_ft * 0.75, 150.0)
    footprint_h = baap.estimated_footprint_sf_per_floor / footprint_w if footprint_w else 0.0
    driveway_w = 24.0
    required_stalls = numbers.get("required_parking_stalls") or 0

    for iteration in range(1, max_iterations + 1):
        with traced_call(state, "simulation", f"physics_relax_iteration_{iteration}", "physics-engine") as t:
            bodies = build_site_bodies(
                envelope=env.model_dump(),
                footprint_w=footprint_w,
                footprint_h=footprint_h,
                n_parking_stalls=required_stalls,
                driveway_w=driveway_w,
            )
            sim = SiteSimulation(env.model_dump(), bodies)
            run_result = sim.run()
            sim.snap_to_grid(1.0)
            stats = SimulationStepStats(
                iteration=iteration,
                steps_run=run_result["steps_run"],
                max_overlap_penalty=run_result["max_overlap_penalty"],
                converged=run_result["converged"],
            )
            state.simulation_history.append(stats)
            t.summary = f"{run_result['steps_run']} steps, penalty={run_result['max_overlap_penalty']}, converged={run_result['converged']}"
        _fire()

        site_plan_path = out_dir / f"site_plan_iter{iteration}.png"
        renderer.render_site_plan(
            site_plan_path,
            lot={"frontage_ft": site.lot_frontage_ft, "depth_ft": site.lot_depth_ft},
            easements=[e.model_dump() for e in site.easements],
            setbacks=site.setbacks_ft,
            buildable_envelope=env.model_dump(),
            bodies=bodies,
        )
        state.site_plan_image_path = str(site_plan_path)

        with traced_call(state, "vision_qa", f"cv_measure_iteration_{iteration}", "classical-cv") as t:
            cv_dict = layout_qa.qa_site_plan(
                site_plan_path,
                buildable_envelope=env.model_dump(),
                lot={"area_sf": site.lot_area_sf},
                required_parking_stalls=required_stalls,
            )
            cv = CVMetrics.model_validate(cv_dict)
            state.cv_metrics_history.append(cv)
            t.summary = (
                f"geometry_passed={cv.geometry_passed}, stalls {cv.parking_stalls_detected}/"
                f"{cv.parking_stalls_required} (shortfall {cv.parking_shortfall}), coverage {cv.lot_coverage_pct}%"
            )
            t.details = cv_dict
        _fire()

        # A clean geometry pass with zero parking shortfall needs no critic
        # opinion — the design is complete. Everything else (a geometry
        # violation, or any parking shortfall) goes to the critic, which
        # decides whether another simulation pass can plausibly help or
        # whether this is a capacity/program conflict no geometry tweak
        # will fix (see design_critic_agent.py).
        if cv.geometry_passed and cv.parking_shortfall == 0:
            break

        with traced_call(state, "design_critic", f"propose_adjustment_iteration_{iteration}", config.MODEL_ROUTE["design_critic"]) as t:
            critique, latency = design_critic_agent.run(cv, stats, iteration, max_iterations)
            state.critic_history.append(critique)
            t.summary = f"continue={critique.continue_iterating}: {critique.rationale[:120]}"
            t.details = {"latency_ms": latency, "adjustments": critique.adjustments}
        _fire()

        if not critique.continue_iterating:
            break
        footprint_w = max(20.0, footprint_w + critique.param_deltas.get("footprint_width_delta_ft", 0.0))
        footprint_h = max(20.0, footprint_h + critique.param_deltas.get("footprint_depth_delta_ft", 0.0))
        driveway_w = max(20.0, driveway_w + critique.param_deltas.get("driveway_width_delta_ft", 0.0))

    if len(state.simulation_history) > 1:
        chart_path = out_dir / "convergence_chart.png"
        # history stored on sim objects isn't kept; approximate with penalty trend across iterations
        renderer.render_convergence_chart(chart_path, [[s.max_overlap_penalty] for s in state.simulation_history])
        state.convergence_chart_path = str(chart_path)


def _generate_floor_plates(state: ProjectState, on_step=None) -> None:
    def _fire():
        if on_step:
            on_step(state.trace[-1])

    catalog = json.loads(config.BAAP_CATALOG_JSON.read_text())
    unit_specs = catalog["unit_modules"]
    core_spec = catalog["building_modules"]["CORE_STD"]
    corridor_w = catalog["building_modules"]["CORRIDOR_STD"]["clear_width_ft"]
    out_dir = _outputs_dir(state.project_id)

    floor_count = state.program.floor_count or 1
    module_counts = state.baap.module_counts
    per_floor = {m: n // floor_count for m, n in module_counts.items()}
    remainder = {m: n - per_floor[m] * floor_count for m, n in module_counts.items()}

    env = state.site.buildable_envelope
    # Run the double-loaded corridor along whichever buildable-envelope axis
    # is longer (here: depth, 265 ft, not width, 170 ft) — a corridor forced
    # into the shorter axis leaves no room for most units and the packer
    # silently defers them, which is a layout mistake, not a real capacity
    # limit. A real multi-wing footprint could do better still; that's
    # Phase 2 (see REPORT.md limitations).
    max_width = max(env.width_ft, env.depth_ft)

    with traced_call(state, "floor_plate", "pack_typical_upper_floor", "rule-based-packer") as t:
        typical_counts = {m: per_floor[m] + remainder[m] for m in module_counts}  # remainder to a typical (non-ground) floor
        typical = floor_plate.pack_typical_floor(typical_counts, unit_specs, core_spec, corridor_w, max_width)
        path = out_dir / "floor_plate_typical.png"
        renderer.render_floor_plate(path, typical, "Typical Upper Floor (generated)")
        state.floor_plate_image_path = str(path)
        t.summary = f"{typical['units_placed']} units placed, {typical['units_unplaced']} deferred"
        t.details = typical
    _fire()

    amenity_depth = core_spec["depth_ft"]
    building_modules = catalog["building_modules"]
    extra_top = [
        {"module_id": "AMENITY_LOBBY", "kind": "amenity", "width_ft": round(building_modules["AMENITY_LOBBY"]["gsf"] / amenity_depth, 1), "depth_ft": amenity_depth},
        {"module_id": "AMENITY_FIT", "kind": "amenity", "width_ft": round(building_modules["AMENITY_FIT"]["gsf"] / amenity_depth, 1), "depth_ft": amenity_depth},
    ]
    extra_bottom = [
        {"module_id": "BOH_LOADING", "kind": "boh", "width_ft": round(building_modules["BOH_LOADING"]["gsf"] / amenity_depth, 1), "depth_ft": amenity_depth},
    ]
    with traced_call(state, "floor_plate", "pack_ground_floor", "rule-based-packer") as t:
        ground = floor_plate.pack_typical_floor(
            per_floor, unit_specs, core_spec, corridor_w, max_width, extra_top=extra_top, extra_bottom=extra_bottom
        )
        path = out_dir / "floor_plate_ground.png"
        renderer.render_floor_plate(path, ground, "Ground Floor (generated)")
        t.summary = f"{ground['units_placed']} units placed, {ground['units_unplaced']} deferred (amenities + BOH reserved)"
        t.details = ground
    _fire()


def ask_agent(state: ProjectState, agent_name: str, question: str) -> str:
    """Free-form, on-demand query against any agent's domain: routes to the
    right knowledge-base slice plus whatever of the current ProjectState is
    already populated, so a user can interrogate any part of the pipeline at
    any time rather than only seeing the fixed run order.
    """
    vs = get_vectorstore()
    domain_queries = {
        "intake": "customer program hard soft requirements",
        "zoning": "zoning setbacks height FAR parking fire lane ADA",
        "baap_matching": "BaaP module catalog stacking adjacency",
        "conflict": "insufficient information program does not fit",
        "design_critic": "site planning parking structured podium guidance",
    }
    context = vs.retrieve_as_context(domain_queries.get(agent_name, question), k=4)
    state_context = state.model_dump_json(indent=2, exclude={"trace", "raw_rfp_text"})
    system = (
        f"You are RNGD's {agent_name} agent, answering an ad-hoc question from a teammate "
        "about the current project. Use the retrieved knowledge-base context and the current "
        "project state. If the answer isn't grounded in either, say what's missing rather than "
        "guessing."
    )
    user = f"Question: {question}\n\nRetrieved knowledge base context:\n{context}\n\nCurrent project state:\n{state_context}"
    text, _latency = llm_groq.call_text(config.MODEL_ROUTE.get("ask_agent", config.GROQ_MODEL_LIGHT), system, user)
    return text
