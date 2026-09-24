"""Program Intake Agent: unstructured customer RFP text -> CustomerProgram."""
from __future__ import annotations

from .. import config, llm_groq
from ..schemas import CustomerProgram

SYSTEM = """You are RNGD's Program Intake Agent for a Building-as-a-Product (BaaP) \
concept-to-schematic design workflow. Extract a structured customer program from \
free-text customer requests.

Return ONLY a JSON object with exactly these keys:
- unit_count_total (integer or null)
- unit_mix_pct (object mapping "studio"|"1br"|"2br"|"3br" -> fraction 0-1, should sum to ~1.0)
- floor_count (integer or null)
- amenities (array of short strings)
- parking_preference (short string or null)
- budget_usd (number or null)
- schedule_note (short string or null)
- hard_requirements (array of short strings — things the customer said must not change)
- soft_requirements (array of short strings — things flagged as flexible/preference)
- assumptions (array of short strings — anything you had to infer because it was not stated)

Be conservative: only mark something "hard" if the text says so explicitly (e.g. \
"firm", "must", "cannot go below", "hard requirement"). Everything else defaults to soft \
or goes in assumptions if you had to infer it."""


def run(rfp_text: str) -> tuple[CustomerProgram, float]:
    model = config.MODEL_ROUTE["intake"]
    parsed, latency_ms = llm_groq.call_json(
        model=model,
        system=SYSTEM,
        user=f"Customer request:\n\n{rfp_text}",
    )
    return CustomerProgram.model_validate(parsed), latency_ms
