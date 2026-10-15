# src/rngd/agents/design_critic_agent.py
def run(cv: CVMetrics, sim_stats: SimulationStepStats, iteration: int, max_iterations: int) -> tuple[CriticSuggestion, float]:
    model = config.MODEL_ROUTE["design_critic"]
    parsed, latency_ms = llm.call_json(
        model=model,
        system=SYSTEM,
        user=(
            f"Iteration {iteration} of max {max_iterations}.\n"
            f"CV QA measurements:\n{json.dumps(cv.model_dump(), indent=2)}\n\n"
            f"Simulation convergence stats:\n{json.dumps(sim_stats.model_dump(), indent=2)}"
        ),
    )
    if iteration >= max_iterations:
        parsed["continue_iterating"] = False
    return CriticSuggestion.model_validate(parsed), latency_ms
