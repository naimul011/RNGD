"""Load the reference designs in QA/ (floorplan*.json, output of the earlier genetic
algorithm) as Design objects, so they are scored by exactly the same metrics.

Placement geometry (verified: 0 invalid cells, 0 overlaps, every room touches the corridor):
the anchor is the room's corner on the corridor side, and the room extends in its
`facing` direction; frontage runs along the corridor, depth away from it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .engine import FAR, Design, Placement, _distances
from .program import Program
from .site import Site


def _rect(p: dict) -> tuple[int, int, int, int]:
    a, b = p["frontage_cells"], p["depth_cells"]
    if p["horizontal_run"]:
        h, w = b, a
        c0 = p["anchor_col"]
        r0 = p["anchor_row"] if p["facing"] == "N" else p["anchor_row"] - h + 1
    else:
        h, w = a, b
        r0 = p["anchor_row"]
        c0 = p["anchor_col"] if p["facing"] == "E" else p["anchor_col"] - w + 1
    return r0, c0, h, w


def load_reference(path: str | Path, site: Site, program: Program) -> tuple[Design, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    R, C = site.shape
    corridor = np.zeros((R, C), bool)
    for r, c in data["corridor_cells"]:
        corridor[r, c] = True
    width = max(1, int(round(float(data.get("corridor_width_ft", 5.0)) / site.cell_ft)))
    core_dist, reached, dead = _distances(site, corridor, site.cell_ft, width)
    by_type = {rt.name: rt for rt in program.rooms}
    slack = int(math.ceil(program.exterior_slack_ft / site.cell_ft))
    level_index = {n: i for i, n in enumerate(site.level_names)}
    placements = []
    for p in data["placements"]:
        r0, c0, h, w = _rect(p)
        l = level_index.get(p["level"], p.get("level_index", 0))
        f, d = p["frontage_cells"], p["depth_cells"]
        ext_ok = (site.ext_dist[l] <= slack + 1) & ~site.outside[l]
        fac = p["facing"]
        if fac == "N":
            back, door, sides = ext_ok[r0 + h - 1, c0:c0 + w], (r0 - 1, c0 + w // 2), (ext_ok[r0:r0 + h, max(c0 - 1, 0)], ext_ok[r0:r0 + h, min(c0 + w, C - 1)])
        elif fac == "S":
            back, door, sides = ext_ok[r0, c0:c0 + w], (r0 + h, c0 + w // 2), (ext_ok[r0:r0 + h, max(c0 - 1, 0)], ext_ok[r0:r0 + h, min(c0 + w, C - 1)])
        elif fac == "E":
            back, door, sides = ext_ok[r0:r0 + h, c0 + w - 1], (r0 + h // 2, c0 - 1), (ext_ok[max(r0 - 1, 0), c0:c0 + w], ext_ok[min(r0 + h, R - 1), c0:c0 + w])
        else:
            back, door, sides = ext_ok[r0:r0 + h, c0], (r0 + h // 2, c0 + w), (ext_ok[max(r0 - 1, 0), c0:c0 + w], ext_ok[min(r0 + h, R - 1), c0:c0 + w])
        ext = back.sum() >= 0.5 * f
        corner = bool(ext and (sides[0].sum() >= 0.5 * d or sides[1].sum() >= 0.5 * d))
        dr, dc = int(np.clip(door[0], 0, R - 1)), int(np.clip(door[1], 0, C - 1))
        win = core_dist[max(dr - 2, 0):dr + 3, max(dc - 2, 0):dc + 3]
        dd = float(win.min()) if win.size else FAR
        rt = by_type.get(p["type"])
        placements.append(Placement(p["room"], p["type"], rt.category if rt else "guest", l, r0, c0, h, w, fac,
                                    p["frontage_ft"], p["depth_ft"], bool(ext), corner, dd,
                                    rt.priority if rt else 3.0))
    design = Design(site, program, corridor, placements, list(data.get("unplaced_rooms", [])), core_dist, reached,
                    dead, source=f"reference: {Path(path).name}")
    info = {"file": Path(path).name, "reported_score": data.get("score"), "reported_rooms_placed": data.get("rooms_placed"),
            "reported_rooms_total": data.get("rooms_total"), "seed": data.get("seed"),
            "generations": len(data.get("history", [])), "history": data.get("history", []),
            "reported_shared_wall_ft": data.get("shared_wall_ft")}
    return design, info
