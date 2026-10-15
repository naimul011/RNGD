"""Detailed program views and ADA scoping for a hotel Schematic.

- unit_spec(): fit-out, beds and sleeping capacity per guest-room type (from the type name).
- program_detail(): one row per program line with per-unit area, size, placement,
  occupant load and fit-out.
- mix_by_floor(): keys of each type on each guest floor.
- ada_checks(): 2010 ADA Standards Tables 224.2 / 224.4 / 224.5
  (docs/ada_2010_standards_for_accessible_design.pdf, pp. 83-84).
"""
from __future__ import annotations

import math

import pandas as pd

from . import layout as L
from .program import GROUP_TITLES

# Table 224.2: (max rooms, without roll-in, with roll-in)
ADA_MOBILITY = [(25, 1, 0), (50, 2, 0), (75, 3, 1), (100, 4, 1), (150, 5, 2), (200, 6, 2), (300, 7, 3),
                (400, 8, 4), (500, 9, 4)]
# Table 224.4: (max rooms, rooms with communication features)
ADA_COMM = [(1, 0), (25, 2), (50, 4), (75, 7), (100, 9), (150, 12), (200, 14), (300, 17), (400, 20), (500, 22)]

# IBC Table 1004.5 factors as carried in RNGD's CSV (IBC-OL-*), SF per occupant.
OL_FACTORS = {"guest": (200, "IBC-OL-012 residential"), "business": (150, "IBC-OL-005 business"),
              "service": (300, "IBC-OL-008 accessory storage / mechanical"), "tenant": (60, "IBC-OL-009 mercantile"),
              "shared": (15, "IBC-OL-004 assembly unconcentrated (net)")}


def ada_required(rooms: int) -> tuple[int, int, int]:
    """(without roll-in, with roll-in, total) mobility rooms for `rooms` guest rooms."""
    for cap, a, b in ADA_MOBILITY:
        if rooms <= cap:
            return a, b, a + b
    if rooms <= 1000:
        return math.ceil(rooms * 0.02), math.ceil(rooms * 0.01), math.ceil(rooms * 0.03)
    extra = math.ceil((rooms - 1000) / 100)
    return 20 + extra, 10 + extra, 30 + 2 * extra


def ada_comm_required(rooms: int) -> int:
    for cap, n in ADA_COMM:
        if rooms <= cap:
            return n
    if rooms <= 1000:
        return math.ceil(rooms * 0.05)
    return 50 + 3 * math.ceil((rooms - 1000) / 100)


def unit_spec(name: str, area_each: float) -> dict:
    n = name.lower()
    ada, spa = "ada" in n, "spa" in n
    w = area_each / L.ROOM_DEPTH_FT
    suite = "suite" in n or w >= 20
    if suite:
        beds, sleeps, layout_txt = "1 king + sofa", 4, "living room + separate bedroom"
    elif "qq" in n or "queen" in n or "double" in n:
        beds, sleeps, layout_txt = "2 queens", 4, "open room"
    else:
        beds, sleeps, layout_txt = "1 king", 2, "open room"
    fit = ["shower" if not ada else "roll-in / transfer shower", "vanity", "WC", "closet", "dresser / TV"]
    if spa:
        fit.append("freestanding tub")
    if suite:
        fit.append("sofa, coffee table, lounge chairs")
    if ada:
        fit.append("60 in turning circles, 32 in clear doors (ADA 806.2)")
    return {"beds": beds, "sleeps": sleeps, "layout": layout_txt, "fit_out": ", ".join(fit), "ada": ada,
            "width_ft": w, "depth_ft": L.ROOM_DEPTH_FT}


def _placement(sch, name: str) -> str:
    found = [lv.name.replace("Level ", "L") for lv in sch.levels if any(r.label == name for r in lv.rooms)]
    if not found:
        n = name.lower()
        if "stair" in n:
            return "2 per level (stacked)"
        if "corridor" in n:
            return "every guest floor"
        return "not placed"
    if len(found) == len(sch.levels) - 1 and "L1" not in found:
        return "every guest floor"
    return ", ".join(found)


def program_detail(sch) -> pd.DataFrame:
    p = sch.program
    rows = []
    for name, count, area, cat in p.rows:
        each = area / count if count else 0.0
        kind = L._kind_for(name, cat)
        if cat == "guest":
            s = unit_spec(name, each)
            size = f"{s['width_ft']:.1f}' x {s['depth_ft']:.0f}'"
            fit, beds, sleeps, ada = s["fit_out"], s["beds"], s["sleeps"] * count, "yes" if s["ada"] else ""
            factor, basis = OL_FACTORS["guest"]
        else:
            size, fit, beds, sleeps, ada = "", "", "", "", ""
            factor, basis = OL_FACTORS.get(kind, (300, "IBC-OL-008"))
            nl = name.lower()
            if "gym" in nl:
                factor, basis = 50, "IBC-OL-006 exercise"
            elif "kitchen" in nl:
                factor, basis = 200, "IBC-OL-007 commercial kitchen"
            elif any(k in nl for k in ("mens", "womens", "restroom", "corridor", "stair")):
                factor, basis = None, "not separately counted"
        rows.append({
            "Group": GROUP_TITLES.get(cat, cat.title()), "Room type": name, "Count": count,
            "Area each (SF)": round(each), "Total area (SF)": round(area),
            "% of net": round(100 * area / p.net_sf, 1) if p.net_sf else 0.0,
            "Size (W x D)": size, "Beds": beds, "Sleeps": sleeps, "ADA": ada,
            "Placed on": _placement(sch, name),
            "Occupant load": math.ceil(area / factor) if factor else "", "Load basis": basis,
            "Fit-out": fit,
        })
    return pd.DataFrame(rows)


def mix_by_floor(sch) -> pd.DataFrame:
    data = {}
    for lv in sch.levels[1:]:
        for r in lv.rooms:
            if r.kind == "guest":
                data.setdefault(r.label, {}).setdefault(lv.name, 0)
                data[r.label][lv.name] += 1
    df = pd.DataFrame(data).T.fillna(0).astype(int)
    df = df[[lv.name for lv in sch.levels[1:] if lv.name in df.columns]]
    df["Total"] = df.sum(axis=1)
    df.loc["Keys per floor"] = df.sum(axis=0)
    return df


def level_rooms(lvl) -> pd.DataFrame:
    return pd.DataFrame([{
        "Room": r.label or "(continuation)", "Kind": r.kind, "Program SF": round(r.sf) if r.sf else None,
        "Drawn SF": round(r.w * r.d), "Width (ft)": round(r.w, 1), "Depth (ft)": round(r.d, 1),
        "x (ft)": round(r.x, 1), "y (ft)": round(r.y, 1), "Side": {"S": "south", "N": "north"}.get(r.side, ""),
    } for r in lvl.rooms])


def ada_checks(program) -> list[dict]:
    rooms = program.keys
    ada_rows = [(n, c) for n, c, a, cat in program.rows if cat == "guest" and "ada" in n.lower()]
    provided = sum(c for _, c in ada_rows)
    no_roll, roll, total = ada_required(rooms)
    comm = ada_comm_required(rooms)
    types_all = {_base_type(n) for n, c, a, cat in program.rows if cat == "guest" and c}
    types_ada = {_base_type(n) for n, _ in ada_rows}
    src = "2010 ADA Standards"
    return [
        {"rule_id": "ADA-224.2", "check": "Accessible guest rooms (mobility)",
         "required": f">= {total} for {rooms} rooms ({no_roll} + {roll} with roll-in)", "provided": str(provided),
         "status": "PASS" if provided >= total else "FAIL", "source": src,
         "note": f"Confirm which {roll} of the ADA units get roll-in showers (Table 224.2)."},
        {"rule_id": "ADA-224.4", "check": "Rooms with communication features",
         "required": f">= {comm}", "provided": "not designated", "status": "REVIEW", "source": src,
         "note": f"Designate {comm} rooms; at most 10% of the mobility rooms may count (224.5)."},
        {"rule_id": "ADA-224.5", "check": "Dispersion across room types",
         "required": f"spread over {len(types_all)} types", "provided": f"{len(types_ada)} types: {', '.join(sorted(types_ada)) or '-'}",
         "status": "PASS" if len(types_ada) >= min(3, len(types_all)) else "REVIEW", "source": src, "note": ""},
    ]


def _base_type(name: str) -> str:
    n = name.lower().replace("ada", "").strip()
    if "suite" in n:
        return "suite"
    if "qq" in n or "queen" in n:
        return "two-bed"
    return "king"


def failing_advice(check: dict) -> str:
    """Short, specific next step for a failing / review check."""
    rid = check.get("rule_id", "")
    if rid == "IBC-R1-007":
        return ("Use a Type IA podium for Level 1 (IBC 510.2) so the five guest floors count as a separate building, "
                "or drop to four guest floors and lengthen the wing.")
    if rid == "IBC-R1-005":
        return "Lower floor-to-floor heights or remove a floor."
    if rid == "IBC-R1-027":
        return "Add a stair near the corridor end, or shorten the dead-end run."
    if rid == "IBC-R1-023":
        return "Add a stair or relocate the core toward the middle of the wing."
    if rid == "ADA-224.2":
        return "Convert standard rooms to ADA units (keep them spread across room types)."
    if rid == "ADA-224.4":
        return "Mark the communication-feature rooms in the program (visual alarms, notification devices)."
    if check.get("check") == "Fits buildable envelope":
        return "Add a guest floor, move retail into the front band, or confirm a longer lot."
    if check.get("check") == "Keys placed":
        return "Some keys did not fit; add a floor or allow a longer wing."
    return ""
