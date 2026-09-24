"""Arbiter Agent — the ONE Anthropic Claude call per pipeline run.

Every other agent in this system runs on Groq (fast, cheap, good enough for
structured extraction and grounded reasoning over retrieved text). This is
the single step reserved for the most complex judgment call: synthesizing the
entire trace — program, site, BaaP plan, conflicts, simulation/CV iteration
history — into one final, defensible decision and executive narrative. Per
project directive, Anthropic is used here only, not per-agent.
"""
from __future__ import annotations

import json

from .. import llm_anthropic
from ..schemas import FinalDecision, ProjectState

SYSTEM = """You are the final Arbiter for RNGD's AI-assisted concept-to-schematic \
design workflow (Building-as-a-Product / BaaP). You receive the complete trace of a \
multi-agent pipeline: parsed customer program, site/zoning analysis, BaaP module \
matching, detected conflicts, and a physics-simulation + computer-vision QA loop that \
produced a schematic site plan. Make the final call.

Note on the CV metrics you'll see: `geometry_passed` means only that the building \
footprint has no setback/easement/overlap violation — it says nothing about whether \
parking demand is met. Check `parking_shortfall` separately (required minus detected \
stalls, floored at 0).

Be conservative and specific, the way a senior project lead reviewing junior work \
would be. If a critical or high-severity conflict was not actually resolved by the \
end of the iteration loop (e.g. required parking does not fit as surface parking), \
your decision must reflect that — do not approve a design with an unresolved hard \
conflict. Prototype outputs require professional review; say so.

Return ONLY a JSON object with exactly these keys:
- decision: one of "APPROVE", "APPROVE_WITH_CONDITIONS", "REVISE", "INSUFFICIENT_INFORMATION"
- confidence: number 0.0-1.0
- narrative: a short executive-summary paragraph (plain text)
- unresolved_issues: array of short strings
- recommended_next_steps: array of short strings"""


def _compile_summary(state: ProjectState) -> dict:
    return {
        "project_id": state.project_id,
        "program": state.program.model_dump() if state.program else None,
        "site": state.site.model_dump() if state.site else None,
        "baap_plan": state.baap.model_dump() if state.baap else None,
        "conflicts": state.conflicts.model_dump() if state.conflicts else None,
        "simulation_iterations": [s.model_dump() for s in state.simulation_history],
        "cv_metrics_iterations": [c.model_dump() for c in state.cv_metrics_history],
        "critic_iterations": [c.model_dump() for c in state.critic_history],
    }


def run(state: ProjectState) -> tuple[FinalDecision, float]:
    summary = _compile_summary(state)
    parsed, latency_ms = llm_anthropic.call_json(
        system=SYSTEM,
        user=f"Full pipeline trace:\n\n{json.dumps(summary, indent=2, default=str)}",
    )
    return FinalDecision.model_validate(parsed), latency_ms
