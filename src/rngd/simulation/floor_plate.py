"""Deterministic BaaP floor-plate packer.

Unlike the site-level physics engine, the floor plate is generated with a
rule-based algorithm because the governing constraints (double-loaded
corridor, core alignment, max exit travel) are discrete placement rules from
baap_stacking_rules.md, not continuous forces. This mirrors the semester-scope
suggestion in the RNGD kickoff doc (Q48-49): produce a ground floor and one
typical upper floor rather than every floor of a multistory building.
"""
from __future__ import annotations

from dataclasses import dataclass


def _interleave_by_ratio(counts: dict[str, int]) -> list[str]:
    """Proportional round-robin (largest-remainder-style) interleave: yields
    module ids in ratio order (e.g. studio/1BR/2BR mixed throughout) instead
    of grouping every instance of the widest module first. A naive
    width-descending sort silently fills an entire row with 2BR modules
    before any smaller unit gets a turn whenever the row runs out of space
    early — technically "largest units near the ends" per
    design_guidelines.md, but taken to an unrealistic extreme where an entire
    floor shows nothing but 2BRs.
    """
    keys = [k for k, c in counts.items() if c > 0]
    total = sum(counts[k] for k in keys)
    used = {k: 0 for k in keys}
    seq = []
    for i in range(1, total + 1):
        k = max(keys, key=lambda k: counts[k] * i / total - used[k])
        seq.append(k)
        used[k] += 1
    return seq


@dataclass
class PlacedModule:
    module_id: str
    kind: str
    x: float
    y: float
    w: float
    h: float


def pack_typical_floor(
    unit_counts: dict[str, int],
    unit_specs: dict[str, dict],
    core_spec: dict,
    corridor_clear_width_ft: float,
    max_width_ft: float,
    extra_top: list[dict] | None = None,
    extra_bottom: list[dict] | None = None,
) -> dict:
    """Places unit modules in a double-loaded corridor: one core, then units
    alternating top row / bottom row moving away from the core until the
    available modules or the buildable width run out.

    extra_top / extra_bottom (e.g. lobby, fitness, back-of-house) are placed
    immediately after the core on their respective row, ahead of unit
    modules — a simple stand-in for the positive/negative adjacency rules in
    baap_stacking_rules.md (amenities near the core/entrance on one row,
    back-of-house kept to the other row rather than beside the lobby).
    """
    placed: list[PlacedModule] = []
    core_w, core_h = core_spec["width_ft"], core_spec["depth_ft"]

    x_cursor = 0.0
    placed.append(PlacedModule("CORE_STD", "core", x_cursor, 0.0, core_w, core_h))
    x_cursor += core_w

    x_top, x_bot = x_cursor, x_cursor
    for extra in extra_top or []:
        placed.append(PlacedModule(extra["module_id"], extra.get("kind", "amenity"), x_top, 0.0, extra["width_ft"], extra["depth_ft"]))
        x_top += extra["width_ft"]
    y_bottom = core_h + corridor_clear_width_ft
    for extra in extra_bottom or []:
        placed.append(PlacedModule(extra["module_id"], extra.get("kind", "boh"), x_bot, y_bottom, extra["width_ft"], extra["depth_ft"]))
        x_bot += extra["width_ft"]

    # Interleave unit types proportionally to their requested mix, then bias
    # the two largest modules toward the row ends (corners get 2BR per
    # design_guidelines.md) with a light swap rather than a full sort.
    seq = _interleave_by_ratio(unit_counts)
    queue: list[tuple[str, dict]] = [(uid, unit_specs[uid]) for uid in seq]

    top_row, bottom_row = [], []
    for i, item in enumerate(queue):
        (top_row if i % 2 == 0 else bottom_row).append(item)

    for row in (top_row, bottom_row):
        if len(row) < 2:
            continue
        widest_idx = max(range(len(row)), key=lambda i: row[i][1]["width_ft"])
        row[0], row[widest_idx] = row[widest_idx], row[0]
        if len(row) > 1:
            widest_idx2 = max(range(1, len(row)), key=lambda i: row[i][1]["width_ft"])
            row[-1], row[widest_idx2] = row[widest_idx2], row[-1]

    unplaced = 0
    for uid, spec in top_row:
        w = spec["width_ft"]
        if x_top + w > max_width_ft:
            unplaced += 1
            continue
        placed.append(PlacedModule(uid, "unit", x_top, 0.0, w, spec["depth_ft"]))
        x_top += w
    for uid, spec in bottom_row:
        w = spec["width_ft"]
        if x_bot + w > max_width_ft:
            unplaced += 1
            continue
        y_bottom = core_h + corridor_clear_width_ft
        placed.append(PlacedModule(uid, "unit", x_bot, y_bottom, w, spec["depth_ft"]))
        x_bot += w

    floor_width = max(x_top, x_bot, core_w)
    top_depth = max((m.h for m in placed if m.kind == "unit" and m.y == 0.0), default=core_h)
    bottom_units = [m for m in placed if m.kind == "unit" and m.y > 0.0]
    bottom_depth = max((m.h for m in bottom_units), default=0.0)
    floor_depth = top_depth + corridor_clear_width_ft + bottom_depth if bottom_units else top_depth

    return {
        "modules": [m.__dict__ for m in placed],
        "floor_width_ft": round(floor_width, 1),
        "floor_depth_ft": round(floor_depth, 1),
        "units_placed": sum(1 for m in placed if m.kind == "unit"),
        "units_unplaced": unplaced,
        "corridor_clear_width_ft": corridor_clear_width_ft,
    }
