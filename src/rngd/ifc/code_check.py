"""Cross-checks a generated IFC Design Studio layout against a curated subset
of the REAL 2021 IBC rules RNGD supplied (docs/*) — limited to rules this
codebase can actually compute from the generated layout (height, stories,
exit travel distance, dead-end corridor, exits per story via an occupant-load
estimate). The remaining ~34 rules in RNGD's 42-rule set need space-level
occupancy modeling (assembly/mercantile/kitchen areas, egress capacity in
inches, etc) this prototype doesn't build — see `not_yet_checked()`.

This is deliberately separate from design.py's own ten zoning/BaaP checks,
which are against the SYNTHETIC sample zoning preset — this module checks
against RNGD's REAL code data instead, and the two are shown side by side in
the UI so it's clear which is which.
"""
from __future__ import annotations

import math

from .. import real_docs
from . import design

CHECKED_RULE_IDS = [
    "IBC-R1-005", "IBC-R1-006", "IBC-R1-007", "IBC-R1-008",
    "IBC-R1-023", "IBC-R1-027", "IBC-R1-015", "IBC-R1-016",
]

_OPS = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "eq": lambda a, b: a == b,
        "<": lambda a, b: a < b, ">": lambda a, b: a > b}


def _get(lay, check_name):
    return next((c["provided"] for c in lay.checks if c["check"] == check_name), None)


def _add(out, by_id, rule_id, measured, note=""):
    r = by_id.get(rule_id)
    if r is None or measured is None:
        return
    op, val = r["operator"], r["value"]
    try:
        ok = _OPS[op](measured, float(val))
    except (KeyError, TypeError, ValueError):
        ok = None
    out.append({
        "rule_id": rule_id,
        "rule_name": r["rule_name"],
        "section": f"{r['base_code']} {r['base_section']}",
        "required": f"{op} {val} {r['units']}",
        "measured": round(measured, 1) if isinstance(measured, float) else measured,
        "status": "PASS" if ok else ("FAIL" if ok is False else "INFO"),
        "note": note or str(r["rule_description"])[:160],
    })


def evaluate(params, lay) -> list[dict]:
    """Returns a list of {rule_id, rule_name, section, required, measured, status, note}."""
    df = real_docs.load_rules_df()
    if df is None or df.empty:
        return []
    by_id = {row.rule_id: row for _, row in df.iterrows()}
    m = lay.metrics
    out: list[dict] = []

    _add(out, by_id, "IBC-R1-005", m["height_ft"], "Type IIA sprinklered height limit")
    _add(out, by_id, "IBC-R1-006", m["height_ft"], "Type IIB sprinklered height limit")
    _add(out, by_id, "IBC-R1-007", m["stories"], "Type IIA sprinklered story limit")
    _add(out, by_id, "IBC-R1-008", m["stories"], "Type IIB sprinklered story limit")

    travel_ft = _get(lay, "Exit travel to core (ft)")
    if travel_ft is not None:
        _add(out, by_id, "IBC-R1-023", travel_ft, "Exit access travel distance screen, sprinklered Group R")
        _add(out, by_id, "IBC-R1-027", travel_ft, "Dead-end corridor screen (using the same measured corridor-to-core distance)")

    # Occupant load / exits per floor: BaaP unit GSF through IBC Table 1004.5's
    # "Residential" factor (200 gross sf/occupant) — a real rule applied to a
    # real number, not a guess.
    counts = design.module_counts(params)
    total_unit_gsf = sum(design.CATALOG["unit_modules"][mod]["gsf"] * n for mod, n in counts.items())
    building_gsf = total_unit_gsf / design.CATALOG["grossing_factor"]
    per_floor_gsf = building_gsf / params.floors if params.floors else 0
    occupant_load_per_floor = math.ceil(per_floor_gsf / 200) if per_floor_gsf else 0
    exits_provided = m["wings"] * 2  # each core = 2 stairs per the BaaP catalog's CORE_STD
    exit_rule_id = "IBC-R1-016" if occupant_load_per_floor > 500 else "IBC-R1-015"
    r = by_id.get(exit_rule_id)
    if r is not None:
        required_exits = int(float(r["value"]))
        out.append({
            "rule_id": exit_rule_id,
            "rule_name": r["rule_name"],
            "section": f"{r['base_code']} {r['base_section']}",
            "required": f"{required_exits} exits (occupant load {'501-1000' if exit_rule_id == 'IBC-R1-016' else '1-500'})",
            "measured": f"{exits_provided} exits ({m['wings']} core(s) x 2 stairs), ~{occupant_load_per_floor} occupants/floor",
            "status": "PASS" if exits_provided >= required_exits else "FAIL",
            "note": "Occupant load estimated from BaaP unit GSF via IBC Table 1004.5 'Residential' factor (200 gross sf/occupant).",
        })
    return out


def not_yet_checked_count() -> int:
    df = real_docs.load_rules_df()
    return max(0, len(df) - len(CHECKED_RULE_IDS)) if df is not None else 0
