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
