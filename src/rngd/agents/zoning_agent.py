"""Site & Zoning Agent: site_conditions.json + retrieved zoning code text ->
SiteConstraints, with the buildable envelope computed deterministically in
Python (arithmetic should never be left to an LLM) and the qualitative rule
values (heights, ratios, distances) extracted from retrieved code text by the
LLM, grounded with citations.
"""
from __future__ import annotations

from .. import config, llm_groq
from ..schemas import BuildableEnvelope, Easement, SiteConstraints, ZoningRules
from ..vectorstore import VectorStore

SYSTEM = """You are RNGD's Site & Zoning Agent. You will be given retrieved excerpts \
from the applicable zoning code. Extract the specific numeric standards that apply.

Return ONLY a JSON object with exactly these keys (use null if genuinely not found \
in the excerpts — never invent a number):
- max_height_ft (number or null)
- max_stories (integer or null)
- max_far (number or null)
- max_lot_coverage_pct (number or null, as a percent e.g. 60 not 0.6)
- front_setback_ft, side_setback_ft, rear_setback_ft (numbers or null)
- parking_ratio_per_unit (number or null)
- fire_lane_clear_width_ft (number or null)
- fire_lane_coverage_ft (number or null, the "within N feet of every point" distance)
- ada_route_min_width_ft (number or null)
- max_exit_travel_ft (number or null)
- citations (array of short strings identifying which section each number came from, \
e.g. "4.2 Bulk and Dimensional Standards: max height")"""


def _compute_buildable_envelope(lot: dict, setbacks: dict, easements: list[Easement]) -> BuildableEnvelope:
    frontage, depth = lot["frontage_ft"], lot["depth_ft"]
    x_min = setbacks.get("side_west", 0.0)
    x_max = frontage - setbacks.get("side_east", 0.0)
    y_min = setbacks.get("front", 0.0)
    y_max = depth - setbacks.get("rear", 0.0)

    for e in easements:
        if not e.buildable:
            if e.side == "east":
                x_max = min(x_max, frontage - e.width_ft)
            elif e.side == "west":
                x_min = max(x_min, e.width_ft)
            elif e.side == "north":
                y_min = max(y_min, e.width_ft)
            elif e.side == "south":
                y_max = min(y_max, depth - e.width_ft)

    width = max(0.0, x_max - x_min)
    depth_env = max(0.0, y_max - y_min)
    return BuildableEnvelope(
        x_min_ft=x_min,
        x_max_ft=x_max,
        y_min_ft=y_min,
        y_max_ft=y_max,
        width_ft=width,
        depth_ft=depth_env,
        area_sf=round(width * depth_env, 1),
    )


def run(site_conditions: dict, vs: VectorStore, cv_cross_check: dict | None = None) -> tuple[SiteConstraints, float]:
    district = site_conditions.get("zoning_district", "")
    queries = [
        f"{district} multifamily front side rear setback feet",
        f"{district} multifamily maximum height stories floor area ratio lot coverage",
        f"{district} multifamily residential parking ratio stalls per unit accessible stall",
        "fire apparatus access lane clear width travel distance",
        "accessible route minimum width running slope ADA",
        "maximum exit travel distance dead-end corridor",
    ]
    context = vs.retrieve_multi_as_context(queries, k_each=2)

    model = config.MODEL_ROUTE["zoning"]
    parsed, latency_ms = llm_groq.call_json(
        model=model,
        system=SYSTEM,
        user=f"Zoning district: {district}\n\nRetrieved code excerpts:\n\n{context}",
    )

    setbacks = {
        "front": parsed.get("front_setback_ft") or 0.0,
        "rear": parsed.get("rear_setback_ft") or 0.0,
        "side_east": parsed.get("side_setback_ft") or 0.0,
        "side_west": parsed.get("side_setback_ft") or 0.0,
    }
    easements = [Easement(**e) for e in site_conditions.get("easements", [])]
    lot = site_conditions.get("lot", {})
    envelope = _compute_buildable_envelope(lot, setbacks, easements)

    zoning = ZoningRules(
        max_height_ft=parsed.get("max_height_ft"),
        max_stories=parsed.get("max_stories"),
        max_far=parsed.get("max_far"),
        max_lot_coverage_pct=parsed.get("max_lot_coverage_pct"),
        parking_ratio_per_unit=parsed.get("parking_ratio_per_unit"),
        fire_lane_clear_width_ft=parsed.get("fire_lane_clear_width_ft"),
        fire_lane_coverage_ft=parsed.get("fire_lane_coverage_ft"),
        ada_route_min_width_ft=parsed.get("ada_route_min_width_ft"),
        max_exit_travel_ft=parsed.get("max_exit_travel_ft"),
    )

    site = SiteConstraints(
        lot_frontage_ft=lot.get("frontage_ft", 0.0),
        lot_depth_ft=lot.get("depth_ft", 0.0),
        lot_area_sf=lot.get("area_sf", 0.0),
        setbacks_ft=setbacks,
        easements=easements,
        buildable_envelope=envelope,
        zoning=zoning,
        citations=parsed.get("citations", []),
        cv_cross_check=cv_cross_check or {},
    )
    return site, latency_ms
