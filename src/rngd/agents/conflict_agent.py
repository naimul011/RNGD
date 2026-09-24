"""Conflict Detection Agent: cross-checks program vs. site vs. BaaP plan.

All quantitative feasibility checks (FAR, lot coverage, parking area) are
computed in Python first; the LLM reasons about severity, phrasing, and
recommendations grounded in those numbers plus retrieved design guidance —
it is not asked to do the arithmetic itself.
"""
from __future__ import annotations

import json

from .. import config, llm_groq
from ..schemas import BaaPPlan, ConflictReport, CustomerProgram, SiteConstraints
from ..vectorstore import VectorStore

SF_PER_STALL_WITH_AISLE_ALLOWANCE = 330  # rule-of-thumb surface-lot planning factor

SYSTEM = """You are RNGD's Conflict Detection Agent. You are given precomputed \
quantitative feasibility numbers (do not recompute them) plus retrieved design \
guidance. Identify conflicts between the customer program, the site/zoning \
constraints, and the BaaP plan, and judge severity and recommendations.

Return ONLY a JSON object with exactly these keys:
- conflicts (array of objects: {"id": str, "category": str, "severity": "low"|"medium"|"high"|"critical", "description": str, "recommendation": str})
- feasible_as_requested (boolean)
- program_adjustment_suggestions (array of short strings; empty if none needed)

Guidance: a genuine geometric or code shortfall that cannot be resolved without \
changing the program or site is "critical" or "high". A close-but-workable number is \
"medium" or "low". If information needed to judge a standard is missing, say so \
explicitly rather than guessing."""


def _compute_feasibility(program: CustomerProgram, site: SiteConstraints, baap: BaaPPlan) -> dict:
    lot_area = site.lot_area_sf
    zr = site.zoning
    max_gsf_by_far = lot_area * zr.max_far if zr.max_far else None
    max_footprint_by_coverage = lot_area * (zr.max_lot_coverage_pct / 100) if zr.max_lot_coverage_pct else None

    required_stalls = None
    if zr.parking_ratio_per_unit and program.unit_count_total:
        required_stalls = round(zr.parking_ratio_per_unit * program.unit_count_total)

    surface_parking_area_needed = required_stalls * SF_PER_STALL_WITH_AISLE_ALLOWANCE if required_stalls else None

    envelope_area = site.buildable_envelope.area_sf if site.buildable_envelope else None
    footprint = baap.estimated_footprint_sf_per_floor
    land_left_for_parking = (envelope_area - footprint) if envelope_area is not None else None

    return {
        "lot_area_sf": lot_area,
        "max_buildable_gsf_by_far": max_gsf_by_far,
        "requested_building_gsf": baap.estimated_building_gsf,
        "max_footprint_by_lot_coverage_sf": max_footprint_by_coverage,
        "estimated_footprint_per_floor_sf": footprint,
        "buildable_envelope_area_sf": envelope_area,
        "required_parking_stalls": required_stalls,
        "surface_parking_area_needed_sf": surface_parking_area_needed,
        "land_left_for_parking_after_footprint_sf": land_left_for_parking,
        "surface_parking_likely_fits": (
            (surface_parking_area_needed <= land_left_for_parking)
            if surface_parking_area_needed is not None and land_left_for_parking is not None
            else None
        ),
    }


def run(
    program: CustomerProgram, site: SiteConstraints, baap: BaaPPlan, vs: VectorStore
) -> tuple[ConflictReport, dict, float]:
    numbers = _compute_feasibility(program, site, baap)
    context = vs.retrieve_as_context("insufficient information program does not fit site structured parking", k=3)

    model = config.MODEL_ROUTE["conflict"]
    parsed, latency_ms = llm_groq.call_json(
        model=model,
        system=SYSTEM,
        user=(
            f"Precomputed feasibility numbers:\n{json.dumps(numbers, indent=2)}\n\n"
            f"Customer hard requirements: {program.hard_requirements}\n"
            f"Customer soft requirements: {program.soft_requirements}\n"
            f"Parking preference stated by customer: {program.parking_preference}\n\n"
            f"Retrieved design guidance:\n{context}"
        ),
    )
    report = ConflictReport.model_validate(parsed)
    return report, numbers, latency_ms
