"""Write an optimized design in the same JSON schema as the QA reference outputs
(floorplan*.json), plus the extra fields this optimizer adds (objectives, code checks,
LLM reviews), so tools built for the earlier optimizer can read it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .engine import Design


def _fmt_ft(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    ft = int(x)
    inch = (x - ft) * 12
    return f"{sign}{ft}'-{inch:.0f}\""


def design_to_dict(design: Design, metrics: dict, history: list[dict] | None = None,
                   reviews: list[dict] | None = None, config: dict | None = None) -> dict:
    site = design.site
    cf = site.cell_ft
    placements = []
    for p in design.placements:
        horizontal = p.facing in ("N", "S")
        # anchor = corner on the corridor side (same convention as the reference files)
        if p.facing == "N":
            ar, ac = p.r0, p.c0
        elif p.facing == "S":
            ar, ac = p.r0 + p.h - 1, p.c0
        elif p.facing == "E":
            ar, ac = p.r0, p.c0
        else:
            ar, ac = p.r0, p.c0 + p.w - 1
        x = site.origin_x_ft + p.c0 * cf
        y = site.origin_y_ft + p.r0 * cf
        placements.append({
            "room": p.name, "type": p.type_name, "category": p.category, "level": site.level_names[p.level],
            "level_index": p.level, "facing": p.facing, "anchor_row": int(ar), "anchor_col": int(ac),
            "frontage_cells": int(round(p.frontage_ft / cf)), "depth_cells": int(round(p.depth_ft / cf)),
            "frontage_ft": p.frontage_ft, "depth_ft": p.depth_ft, "horizontal_run": horizontal,
            "exterior": p.exterior, "corner": p.corner, "door_to_core_ft": round(p.door_dist_ft, 1),
            "x_ft": round(x, 4), "y_ft": round(y, 4), "x_display": _fmt_ft(x), "y_display": _fmt_ft(y)})
    rr, cc = np.where(design.corridor)
    return {
        "units": "feet", "generator": "RNGD memetic GA + LLM/RAG review (src/rngd/optimize)",
        "site_file": site.path.name, "program": design.program.source,
        "generated_corridor": True, "corridor_width_ft": design.program.corridor_width_ft,
        "corridor_cells": [[int(r), int(c)] for r, c in zip(rr, cc)], "corridor_applies_to_all_levels": True,
        "cores_total": metrics["cores_total"], "cores_reached": metrics["cores_reached"],
        "exterior_slack_ft": design.program.exterior_slack_ft, "exterior_fraction": metrics["exterior_fraction"],
        "shared_wall_ft": metrics["shared_wall_ft"], "rooms_placed": metrics["rooms_placed"],
        "rooms_total": metrics["rooms_total"], "score": metrics["fitness"], "cell_size_ft": cf,
        "origin_x_ft": site.origin_x_ft, "origin_y_ft": site.origin_y_ft,
        "levels": [{"level": lv["level"], "elevation_ft": site.elevations_ft[i], "rooms": lv["rooms"],
                    "by_type": lv["by_type"]} for i, lv in enumerate(metrics["levels"])],
        "unplaced_rooms": design.unplaced, "unplaced_by_type": metrics["unplaced_by_type"],
        "room_types": [rt.name for rt in design.program.rooms], "placements": placements,
        "history": [[h["gen"], h["best"], h["best_placed"]] for h in (history or [])],
        "objectives": metrics["terms"], "weights": metrics["weights"], "code_checks": metrics["checks"],
        "max_travel_ft": metrics["max_travel_ft"], "dead_end_ft": metrics["dead_end_ft"],
        "corridor_ratio": metrics["corridor_ratio"], "llm_reviews": reviews or [], "search_config": config or {},
    }


def save_design(path: str | Path, design: Design, metrics: dict, **kw) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(design_to_dict(design, metrics, **kw), indent=1, default=str), encoding="utf-8")
    return p
