"""BaaP Matching Agent: customer program -> module palette from RNGD's BaaP
catalog. Quantities and GSF are computed deterministically in Python from
data/baap_catalog.json (never guessed by the LLM); the LLM's job is the
qualitative adjacency plan and utilization narrative, grounded in the
retrieved stacking-rules text.
"""
from __future__ import annotations

import json
import math

from .. import config, llm_groq
from ..schemas import BaaPPlan
from ..vectorstore import VectorStore

UNIT_TYPE_TO_MODULE = {
    "studio": "STUDIO_A",
    "1br": "ONEBR_A",
    "2br": "TWOBR_A",
    "3br": "TWOBR_A",  # no 3BR module in the sample catalog; closest substitute, flagged in notes
}

SYSTEM = """You are RNGD's BaaP Matching Agent. You are given: the customer's requested \
unit module counts (already computed), retrieved BaaP stacking/adjacency rules, and \
computed building-level metrics. Write the qualitative adjacency plan and a short \
utilization narrative — do not recompute or restate numbers, just reason about them.

Return ONLY a JSON object with exactly these keys:
- adjacency_plan (array of short strings, e.g. "AMENITY_LOBBY placed at main entrance, adjacent to CORE_STD")
- utilization_notes (a short paragraph: how well the requested program maps to standard BaaP modules, any substitutions made, any accessible-unit distribution notes)
- citations (array of short strings naming which retrieved rule each planning decision follows)"""


def _module_counts(unit_count_total: int, unit_mix_pct: dict[str, float]) -> dict[str, int]:
    counts: dict[str, int] = {}
    assigned = 0
    items = list(unit_mix_pct.items())
    for i, (utype, pct) in enumerate(items):
        module_id = UNIT_TYPE_TO_MODULE.get(utype.lower(), "ONEBR_A")
        if i == len(items) - 1:
            n = unit_count_total - assigned  # last bucket absorbs rounding remainder
        else:
            n = round(unit_count_total * pct)
        assigned += n
        counts[module_id] = counts.get(module_id, 0) + max(0, n)
    return counts


def run(program, vs: VectorStore) -> tuple[BaaPPlan, float]:
    catalog = json.loads(config.BAAP_CATALOG_JSON.read_text())
    unit_specs = catalog["unit_modules"]
    grossing_factor = catalog["grossing_factor"]

    unit_count_total = program.unit_count_total or 0
    unit_mix_pct = program.unit_mix_pct or {"1br": 1.0}
    module_counts = _module_counts(unit_count_total, unit_mix_pct)

    total_unit_gsf = sum(unit_specs[m]["gsf"] * n for m, n in module_counts.items())
    estimated_building_gsf = total_unit_gsf / grossing_factor if grossing_factor else total_unit_gsf
    floor_count = program.floor_count or 1
    estimated_footprint_sf_per_floor = estimated_building_gsf / floor_count if floor_count else estimated_building_gsf

    units_per_floor = unit_count_total / floor_count if floor_count else unit_count_total
    core_count = 1 if units_per_floor <= 45 else math.ceil(units_per_floor / 45)

    accessible_min_pct = catalog["stacking_rules"]["accessible_unit_min_pct_of_total"]
    accessible_units_required = max(1, math.ceil(unit_count_total * accessible_min_pct)) if unit_count_total else 0

    query = "double-loaded corridor adjacency lobby amenity accessible unit distribution core alignment"
    context = vs.retrieve_as_context(query, k=4)

    model = config.MODEL_ROUTE["baap_matching"]
    parsed, latency_ms = llm_groq.call_json(
        model=model,
        system=SYSTEM,
        user=(
            f"Module counts: {json.dumps(module_counts)}\n"
            f"Core count: {core_count}\n"
            f"Accessible units required (min): {accessible_units_required}\n"
            f"Total unit GSF: {round(total_unit_gsf)}\n"
            f"Estimated building GSF (incl. grossing factor {grossing_factor}): {round(estimated_building_gsf)}\n"
            f"Estimated footprint per floor: {round(estimated_footprint_sf_per_floor)} sf over {floor_count} floors\n"
            f"Amenities requested: {program.amenities}\n\n"
            f"Retrieved rules:\n{context}"
        ),
    )

    plan = BaaPPlan(
        module_counts=module_counts,
        total_unit_gsf=round(total_unit_gsf, 1),
        grossing_factor=grossing_factor,
        estimated_building_gsf=round(estimated_building_gsf, 1),
        estimated_footprint_sf_per_floor=round(estimated_footprint_sf_per_floor, 1),
        core_count=core_count,
        utilization_notes=parsed.get("utilization_notes", ""),
        adjacency_plan=parsed.get("adjacency_plan", []),
        citations=parsed.get("citations", []),
    )
    return plan, latency_ms
