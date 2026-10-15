"""LLM design-review agent (with retrieval) used inside the fitness function.

At a few checkpoints the search sends the current best distinct candidates to a reviewer.
The reviewer gets:
  * each candidate's computed metrics, per-floor mix and failed code checks (numbers only
    come from metrics.py, never from the model),
  * passages retrieved from the knowledge base (RNGD's IBC rule CSV, the 2010 ADA Standards,
    the sample design guidelines) for the issues present,
and returns (JSON, schema-checked):
  * a 0-10 score per candidate with reasons (qualitative design quality: guest experience,
    wayfinding, ADA dispersion, service logic, code risk),
  * multipliers (0.7-1.3) for the objective weights, i.e. which objectives deserve more
    emphasis in the next generations.
The score enters the fitness of the reviewed candidates as
    F_total = (1 - lam) * F + lam * score / 10,
and the multipliers re-weight the deterministic objectives for the whole population.
Without an API key (or with use_llm=False) a transparent heuristic reviewer is used instead.
"""
from __future__ import annotations

import json

from .metrics import DEFAULT_WEIGHTS, OBJECTIVE_TEXT

SYSTEM = """You are a senior hotel architect reviewing candidate schematic floor plans produced by an \
optimizer for RNGD (Building-as-a-Product). You receive computed metrics for each candidate (trust them; \
do not recompute) and retrieved excerpts from building codes and design guidance.

Judge qualitative design quality the metrics cannot fully capture: guest experience (rooms with windows, \
suites on corners), wayfinding (simple corridors, short dead ends), ADA dispersion and access to the \
elevator/core, service logic (housekeeping, ice, IT near the core; lobby with luggage and restrooms on \
Level 1), code risk (IBC egress limits), and buildability (compact, repetitive rooms suit prefab).

Return ONLY JSON with exactly these keys:
- "reviews": array of {"id": str, "score": number 0-10, "strengths": [short strings], "issues": [short strings]}
- "weight_multipliers": object mapping some of these objective names to a number between 0.7 and 1.3:
  OBJECTIVES
- "rationale": short string explaining the weight changes
- "citations": array of short strings naming the retrieved sources you relied on
Use only numbers that appear in the candidate data."""

QUERIES = {
    "egress": "exit access travel distance dead-end corridor maximum sprinklered R-1",
    "ada": "ADA guest rooms mobility features dispersion transient lodging Table 224.2",
    "daylight": "hotel guest room exterior window natural light design guideline",
    "core_access": "service rooms near elevator core housekeeping linen ice",
    "efficiency": "double-loaded corridor efficiency net to gross module stacking",
}


def _candidate_brief(cid: str, m: dict) -> dict:
    return {
        "id": cid, "fitness": m["fitness"], "objectives": m["terms"],
        "rooms_placed": f'{m["rooms_placed"]}/{m["rooms_total"]}', "keys": f'{m["keys_placed"]}/{m["keys_requested"]}',
        "corridor_ratio": m["corridor_ratio"], "max_travel_ft": m["max_travel_ft"], "dead_end_ft": m["dead_end_ft"],
        "exterior_fraction": m["exterior_fraction"], "ada": f'{m["ada_placed"]} placed, {m["ada_required"]} required',
        "unplaced_by_type": m["unplaced_by_type"],
        "levels": [{"level": lv["level"], "keys": lv["keys"], "by_type": lv["by_type"]} for lv in m["levels"]],
        "failed_checks": [f'{c["rule_id"]} {c["check"]}: required {c["required"]}, provided {c["provided"]}'
                          for c in m["checks"] if c["status"] == "FAIL"],
    }


def retrieve_context(metrics_list: list[dict], vs=None, k_each: int = 2) -> tuple[str, list[str]]:
    """Retrieve passages for the objectives that are weakest across the candidates."""
    weakest = sorted(QUERIES, key=lambda t: sum(m["terms"].get(t, 1.0) for m in metrics_list))[:3]
    queries = [QUERIES[t] for t in weakest] + ["hotel corridor guest room layout guidelines"]
    if vs is None:
        try:
            from ..vectorstore import get_vectorstore
            vs = get_vectorstore()
        except Exception:
            return "", queries
    return vs.retrieve_multi_as_context(queries, k_each=k_each), queries


def heuristic_review(candidates: list[tuple[str, dict]]) -> dict:
    """Offline stand-in: score = 10 x fitness; emphasise the weakest objective a little."""
    reviews = []
    mean_terms: dict = {}
    for cid, m in candidates:
        issues = [c["check"] for c in m["checks"] if c["status"] == "FAIL"]
        issues += [f"{k} low ({v:.2f})" for k, v in m["terms"].items() if v < 0.6]
        reviews.append({"id": cid, "score": round(10 * max(0.0, min(1.0, m["fitness"])), 2),
                        "strengths": [k for k, v in m["terms"].items() if v >= 0.9][:3], "issues": issues[:4]})
        for k, v in m["terms"].items():
            mean_terms[k] = mean_terms.get(k, 0.0) + v / len(candidates)
    weakest = min(mean_terms, key=mean_terms.get) if mean_terms else None
    return {"reviews": reviews, "weight_multipliers": {weakest: 1.15} if weakest else {},
            "rationale": f"heuristic reviewer: emphasise the weakest objective ({weakest})", "citations": [],
            "engine": "heuristic"}


def llm_review(candidates: list[tuple[str, dict]], vs=None, model: str | None = None) -> dict:
    from .. import config, llm

    context, queries = retrieve_context([m for _, m in candidates], vs)
    objectives = "\n".join(f"  - {k}: {v}" for k, v in OBJECTIVE_TEXT.items())
    user = ("Candidates (computed by the optimizer):\n"
            + json.dumps([_candidate_brief(cid, m) for cid, m in candidates], indent=1)
            + "\n\nRetrieved excerpts (code and design guidance):\n" + context)
    parsed, latency = llm.call_json(model or config.LLM_REASONING, SYSTEM.replace("OBJECTIVES", objectives), user)
    ids = {cid for cid, _ in candidates}
    reviews = []
    for r in parsed.get("reviews", []):
        if str(r.get("id")) in ids:
            try:
                score = float(r.get("score", 0))
            except (TypeError, ValueError):
                continue
            reviews.append({"id": str(r["id"]), "score": max(0.0, min(10.0, score)),
                            "strengths": [str(s) for s in r.get("strengths", [])][:5],
                            "issues": [str(s) for s in r.get("issues", [])][:5]})
    mult = {}
    for k, v in (parsed.get("weight_multipliers") or {}).items():
        if k in DEFAULT_WEIGHTS:
            try:
                mult[k] = max(0.7, min(1.3, float(v)))
            except (TypeError, ValueError):
                pass
    return {"reviews": reviews, "weight_multipliers": mult, "rationale": str(parsed.get("rationale", ""))[:600],
            "citations": [str(c) for c in parsed.get("citations", [])][:8], "queries": queries,
            "engine": model or config.LLM_REASONING, "latency_s": round(latency / 1000, 1)}


def review(candidates: list[tuple[str, dict]], use_llm: bool = True, vs=None) -> dict:
    if use_llm:
        try:
            out = llm_review(candidates, vs)
            if out["reviews"]:
                return out
        except Exception as e:  # network / key / schema problems fall back, but say so
            fb = heuristic_review(candidates)
            fb["rationale"] += f" (LLM review unavailable: {type(e).__name__})"
            return fb
    return heuristic_review(candidates)
