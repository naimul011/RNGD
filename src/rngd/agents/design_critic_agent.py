"""Design Critic Agent: closes the optimize loop between the physics
simulation and the classical CV QA measurements. Reads what CV actually
measured on the rendered plan (not what the simulation intended) and proposes
concrete geometry deltas for the next simulation iteration, or says to stop.
"""
from __future__ import annotations

import json

from .. import config, llm_groq
from ..schemas import CVMetrics, CriticSuggestion, SimulationStepStats

SYSTEM = """You are RNGD's Design Critic Agent, closing the loop between a physics-style \
site layout simulation and a computer-vision QA pass on the rendered result. You get the \
CV measurements (ground truth: what the drawing actually shows) and the simulation's own \
convergence stats. Propose the smallest concrete geometry adjustment that would fix a real \
violation, or say to stop iterating if the design is acceptable or if no further geometry \
change can fix a fundamental capacity shortfall (e.g. required parking simply does not fit \
the site — that should be reported as a program/site conflict, not chased with more \
iterations).

Return ONLY a JSON object with exactly these keys:
- adjustments (array of short human-readable strings describing what to change)
- param_deltas (object; allowed keys only: "footprint_width_delta_ft", "footprint_depth_delta_ft", "driveway_width_delta_ft" — numbers, positive or negative, 0 if unused)
- continue_iterating (boolean — true only if a further simulation pass with these deltas is likely to help)
- rationale (short string)"""


def run(cv: CVMetrics, sim_stats: SimulationStepStats, iteration: int, max_iterations: int) -> tuple[CriticSuggestion, float]:
    model = config.MODEL_ROUTE["design_critic"]
    parsed, latency_ms = llm_groq.call_json(
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
