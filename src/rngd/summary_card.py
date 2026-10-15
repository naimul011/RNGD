"""Pure-Python (no Streamlit) builders for the chat panel's persistent
summary card: headline, requirement-match %, setbacks, and a colored list of
conflicts/checks. app.py renders whatever this returns; keeping the
computation here keeps app.py's layout code from drowning in it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ifc.design import ZONING


@dataclass
class CardItem:
    label: str
    detail: str
    severity: str  # "ok" | "low" | "medium" | "high" | "critical" | "info"
    tag: str = ""  # e.g. "sample zoning" vs "real IBC" — shown as a small badge


@dataclass
class SummaryCard:
    title: str
    badge: str
    headline: str
    match_pct: float | None = None
    metrics: list[tuple[str, str]] = field(default_factory=list)
    setbacks: list[tuple[str, str]] = field(default_factory=list)
    items: list[CardItem] = field(default_factory=list)


_CONFLICT_PENALTY = {"critical": 30, "high": 15, "medium": 8, "low": 3}


def pipeline_card(state) -> SummaryCard | None:
    if state is None:
        return None
    conflicts = state.conflicts.conflicts if state.conflicts else []
    penalty = sum(_CONFLICT_PENALTY.get(c.severity, 5) for c in conflicts)
    match_pct = max(0, 100 - penalty)

    metrics = []
    if state.program:
        metrics.append(("Units requested", str(state.program.unit_count_total)))
        metrics.append(("Floors", str(state.program.floor_count)))
    if state.baap:
        metrics.append(("Building GSF", f"{state.baap.estimated_building_gsf:,.0f}"))
    if state.cv_metrics_history:
        cv = state.cv_metrics_history[-1]
        metrics.append(("Lot coverage", f"{cv.lot_coverage_pct}%"))
        metrics.append(("Parking", f"{cv.parking_stalls_detected}/{cv.parking_stalls_required}"))

    setbacks = []
    if state.site:
        setbacks = [(k.replace("_", " "), f"{v:.0f} ft") for k, v in state.site.setbacks_ft.items()]

    items = [
        CardItem(label=c.id, detail=c.description, severity=c.severity, tag="pipeline conflict")
        for c in conflicts
    ]
    if state.cv_metrics_history:
        cv = state.cv_metrics_history[-1]
        if not cv.geometry_passed:
            for v in cv.setback_violations + cv.easement_violations + cv.overlap_violations:
                items.append(CardItem(label="Geometry violation", detail=v, severity="critical", tag="CV QA"))

    decision = state.final_decision.decision if state.final_decision else "in progress"
    return SummaryCard(
        title="Multi-Agent Pipeline",
        badge=state.project_id,
        headline=f"Decision: {decision}" + (f" ({state.final_decision.confidence:.0%} confidence)" if state.final_decision else ""),
        match_pct=match_pct,
        metrics=metrics,
        setbacks=setbacks,
        items=items,
    )


def ifc_card(params, lay) -> SummaryCard:
    m = lay.metrics
    checks_passed = sum(1 for c in lay.checks if c["status"] == "PASS")
    match_pct = 100 * checks_passed / len(lay.checks) if lay.checks else None

    metrics = [
        ("Units", f"{m['units_placed']}/{params.units}"),
        ("Stories", str(m["stories"])),
        ("FAR", str(m["far"])),
        ("Coverage", f"{m['coverage_pct']}%"),
        ("Parking", f"{m['parking_provided']}/{m['parking_required']}"),
    ]
    z = ZONING[params.zoning]
    setbacks = [("front", f"{z['front']} ft"), ("side", f"{z['side']} ft"), ("rear", f"{z['rear']} ft"),
                ("easement (east)", f"{params.easement_east:.0f} ft")]

    items = [
        CardItem(label=c["check"], detail=f"required {c['required']}, got {c['provided']}",
                 severity={"FAIL": "high", "WARN": "medium"}.get(c["status"], "ok"), tag="sample zoning")
        for c in lay.checks if c["status"] != "PASS"
    ]

    try:
        from .ifc import code_check
        for r in code_check.evaluate(params, lay):
            if r["status"] == "FAIL":
                items.append(CardItem(label=r["rule_id"], detail=f"{r['rule_name']} — required {r['required']}, got {r['measured']}",
                                      severity="critical", tag="real IBC"))
    except Exception:
        pass

    return SummaryCard(
        title="IFC Design Studio",
        badge=params.zoning,
        headline=f"{checks_passed}/{len(lay.checks)} sample-zoning checks pass" if lay.checks else "generating...",
        match_pct=match_pct,
        metrics=metrics,
        setbacks=setbacks,
        items=items,
    )


def explorer_card(summary: dict | None, file_label: str | None) -> SummaryCard | None:
    if summary is None:
        return None
    metrics = [
        ("Schema", summary["schema"]),
        ("Storeys", str(len(summary["storeys"]))),
        ("Elements", str(summary["total_elements"])),
        ("Rendered", str(summary.get("meshes_triangulated", "—"))),
    ]
    items = []
    for o in summary.get("excluded_outliers", []):
        items.append(CardItem(label=o["name"], detail=f"{o['distance_m']} m from the building — excluded from 3D view",
                              severity="medium", tag="outlier"))
    return SummaryCard(
        title="Open IFC File",
        badge=file_label or "",
        headline=summary.get("project_name") or "Real model opened",
        match_pct=None,
        metrics=metrics,
        setbacks=[],
        items=items,
    )


def hotel_card(sch, circulation_pct: float = 33, wall_pct: float = 8) -> SummaryCard:
    """Hotel Schematic mode: program, massing and the IBC / ADA screen."""
    from .hotel.details import failing_advice

    p = sch.program
    scored = [c for c in sch.checks if c["status"] in ("PASS", "FAIL")]
    passed = sum(c["status"] == "PASS" for c in scored)
    n = p.net_to_gross(circulation_pct, wall_pct)
    items = []
    for c in sch.checks:
        if c["status"] == "PASS":
            continue
        sev = {"FAIL": "critical", "REVIEW": "medium", "ASSUMED": "info"}.get(c["status"], "low")
        tip = failing_advice(c)
        items.append(CardItem(label=f"{c['check']}", detail=f"required {c['required']}, provided {c['provided']}"
                              + (f". Next: {tip}" if tip else ""), severity=sev,
                              tag=c["rule_id"] if c["rule_id"] != "-" else c.get("source", "")))
    return SummaryCard(
        title="Hotel Schematic", badge=p.source[:38] + ("..." if len(p.source) > 38 else ""),
        headline=f"{p.keys} keys on {len(sch.levels) - 1} guest floors; {passed}/{len(scored)} code checks pass",
        match_pct=100 * passed / len(scored) if scored else None,
        metrics=[("Keys", str(p.keys)), ("Net SF", f"{p.net_sf:,.0f}"),
                 ("Gross drawn", f"{sum(lv.gross_sf for lv in sch.levels):,.0f}"),
                 (f"Gross {circulation_pct:.0f}%+{wall_pct:.0f}%", f"{n['required_gross_sf']:,.0f}"), ("Levels", str(len(sch.levels))),
                 ("Height", f"{sch.height_ft:.1f} ft")],
        setbacks=[("Envelope", f"{sch.site['envelope_d']:.0f} x {sch.site['envelope_w']:.0f} ft"),
                  ("Podium", f"{sch.levels[0].length_ft:.0f} x {sch.levels[0].width_ft:.0f} ft"),
                  ("Tower", f"{sch.levels[1].length_ft:.0f} x {sch.levels[1].width_ft:.0f} ft")],
        items=items,
    )


def optimizer_card(result: dict) -> SummaryCard:
    """Design Optimizer mode: the optimized layout and its code screen."""
    m = result["metrics"]
    scored = m["checks"]
    passed = sum(c["status"] == "PASS" for c in scored)
    items = [CardItem(label=c["check"], detail=f"required {c['required']}, provided {c['provided']}", severity="critical",
                      tag=c["rule_id"]) for c in scored if c["status"] == "FAIL"]
    if m["unplaced_by_type"]:
        items.append(CardItem(label="Rooms not placed", detail=", ".join(f"{k} x{v}" for k, v in m["unplaced_by_type"].items()),
                              severity="medium", tag="program"))
    ref = result.get("reference")
    setbacks = []
    if ref:
        rm = ref["metrics"]
        setbacks = [("Reference rooms", f"{rm['rooms_placed']}/{rm['rooms_total']}"), ("Reference fitness", f"{rm['fitness']}"),
                    ("Reference dead end", f"{rm['dead_end_ft']} ft")]
    return SummaryCard(
        title="Design Optimizer", badge=f"{result['site']} · {result['program']}"[:40],
        headline=f"{m['rooms_placed']}/{m['rooms_total']} rooms placed, fitness {m['fitness']}; {passed}/{len(scored)} checks pass",
        match_pct=100 * m["rooms_placed"] / max(1, m["rooms_total"]),
        metrics=[("Keys", f"{m['keys_placed']}/{m['keys_requested']}"), ("Corridor", f"{100 * m['corridor_ratio']:.1f}%"),
                 ("Windows", f"{100 * m['exterior_fraction']:.0f}%"), ("Travel", f"{m['max_travel_ft']} ft"),
                 ("Dead end", f"{m['dead_end_ft']} ft"), ("LLM reviews", str(len(result.get("reviews", []))))],
        setbacks=setbacks, items=items)
