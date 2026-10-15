# src/rngd/orchestrator.py
# A clean geometry pass with zero parking shortfall needs no critic
# opinion -- the design is complete. Everything else (a geometry
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
