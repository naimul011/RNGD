"""Rule-driven parametric layout: requirements + zoning preset + BaaP catalog -> geometry + compliance.
All dimensions in feet here; the IFC writer converts to metres. Planning-level, not code-approved."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace

from .. import config

CATALOG = json.loads(config.BAAP_CATALOG_JSON.read_text())

# Sample zoning presets (synthetic; swap with real jurisdiction data).
ZONING = {
    "Rivermont R-4 (sample)": dict(front=15, side=10, rear=20, far=2.5, cov=60, height=65, stories=5, parking=1.25),
    "Dense urban R-6 (sample)": dict(front=10, side=5, rear=15, far=3.5, cov=75, height=85, stories=7, parking=0.75),
    "Suburban R-2 (sample)": dict(front=25, side=10, rear=25, far=1.0, cov=40, height=35, stories=3, parking=2.0),
}


@dataclass(frozen=True)
class Params:
    lot_w: float = 200
    lot_d: float = 300
    easement_east: float = 20
    zoning: str = "Rivermont R-4 (sample)"
    units: int = 120
    mix: tuple = (0.20, 0.45, 0.35)  # studio, 1br, 2br
    floors: int = 4  # residential floors
    wings: int = 0  # 0 = auto
    parking: str = "surface"  # "surface" | "podium"
    corridor: float = 6
    floor_h: float = 10.5
    ground_h: float = 14.0


@dataclass
class Layout:
    p: Params
    env: dict
    wings: list = field(default_factory=list)  # rects x,y,w,h
    units: list = field(default_factory=list)  # level,module,x,y,w,d
    cores: list = field(default_factory=list)  # level,x,y,w,d
    amenities: list = field(default_factory=list)
    stalls: list = field(default_factory=list)  # x,y,w,d (surface)
    drive: dict | None = None
    storeys: list = field(default_factory=list)  # name, elevation_ft, height_ft
    podium_capacity: int = 0
    metrics: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)


def envelope(p: Params) -> dict:
    z = ZONING[p.zoning]
    x0, x1 = z["side"], p.lot_w - max(z["side"], p.easement_east)
    y0, y1 = z["front"], p.lot_d - z["rear"]
    return dict(x0=x0, x1=x1, y0=y0, y1=y1, w=x1 - x0, d=y1 - y0)


def _interleave(counts: dict) -> list:
    keys = [k for k, c in counts.items() if c > 0]
    total = sum(counts[k] for k in keys)
    used = {k: 0 for k in keys}
    out = []
    for i in range(1, total + 1):
        k = max(keys, key=lambda k: counts[k] * i / total - used[k])
        out.append(k)
        used[k] += 1
    return out


def module_counts(p: Params) -> dict:
    total = p.units
    s, o = round(total * p.mix[0]), round(total * p.mix[1])
    return {"STUDIO_A": s, "ONEBR_A": o, "TWOBR_A": max(0, total - s - o)}


def _to_xy(env, horizontal_long, u, v, w_u, w_v):
    """local (u along long axis, v across) -> x,y,w,h rect."""
    if horizontal_long:
        return env["x0"] + u, env["y0"] + v, w_u, w_v
    return env["x0"] + v, env["y0"] + u, w_v, w_u


def _build(p: Params, wings: int) -> Layout:
    env = envelope(p)
    cat = CATALOG
    U = cat["unit_modules"]
    core = cat["building_modules"]["CORE_STD"]
    lay = Layout(p, env)
    podium = p.parking == "podium"

    seq = _interleave(module_counts(p))
    dmax = max(U[m]["depth_ft"] for m in set(seq)) if seq else 36
    bar_w = 2 * dmax + p.corridor
    gap = 10
    drive_w = 24
    horizontal = env["w"] > env["d"]
    L_avail = env["w"] if horizontal else env["d"]
    across_avail = env["d"] if horizontal else env["w"]
    need_across = wings * bar_w + (wings - 1) * gap
    has_drive = need_across + drive_w <= across_avail
    v_off = drive_w if has_drive else 0

    levels = list(range(1, p.floors + 1))
    lay.storeys = ([("Podium parking", 0.0, p.ground_h)] if podium else [])
    z = p.ground_h if podium else 0.0
    for n in levels:
        h = p.ground_h if (n == 1 and not podium) else p.floor_h
        lay.storeys.append((f"Level {n}", z, h))
        z += h
    lay.storeys.append(("Roof", z, 0.0))

    wing_len = [core_len_ for core_len_ in [core["depth_ft"]] * wings]
    core_len = core["depth_ft"]
    amen = [("AMENITY_LOBBY", CATALOG["building_modules"]["AMENITY_LOBBY"]["gsf"]),
            ("AMENITY_FIT", CATALOG["building_modules"]["AMENITY_FIT"]["gsf"]),
            ("BOH_LOADING", CATALOG["building_modules"]["BOH_LOADING"]["gsf"])]

    idx = 0
    max_u = 0.0
    for n in levels:
        rows = []  # each: cursor, v0(row top), dir
        for k in range(wings):
            v0 = v_off + k * (bar_w + gap)
            rows.append(dict(cur=core_len, vA=v0, vB=v0 + dmax + p.corridor, wing=k))
            rows.append(dict(cur=core_len, vA=v0, vB=v0 + dmax + p.corridor, wing=k, second=True))
        if n == 1:  # amenities replace slots on wing 0 first row
            r = rows[0]
            for name, gsf in amen:
                w = round(gsf / core["depth_ft"], 1)
                if r["cur"] + w <= L_avail:
                    lay.amenities.append(dict(level=n, module=name, **dict(zip("xywd", _to_xy(env, horizontal, r["cur"], r["vA"] + dmax - core["depth_ft"], w, core["depth_ft"])))))
                    r["cur"] += w
        for k in range(wings):
            uc = _to_xy(env, horizontal, 0, v_off + k * (bar_w + gap) + bar_w / 2 - core["width_ft"] / 2, core_len, core["width_ft"])
            lay.cores.append(dict(level=n, x=uc[0], y=uc[1], w=uc[2], d=uc[3]))
        target = math.ceil((len(seq) - idx) / (len(levels) - n + 1))
        placed_here = 0
        while idx < len(seq) and placed_here < target:
            r = min(rows, key=lambda r: r["cur"])
            m = seq[idx]
            w, d = U[m]["width_ft"], U[m]["depth_ft"]
            if r["cur"] + w > L_avail:
                # try another row that still fits
                fit = [x for x in rows if x["cur"] + w <= L_avail]
                if not fit:
                    break
                r = min(fit, key=lambda x: x["cur"])
            v = (r["vB"] if r.get("second") else r["vA"] + dmax - d)
            x, y, ww, dd = _to_xy(env, horizontal, r["cur"], v, w, d)
            lay.units.append(dict(level=n, module=m, x=x, y=y, w=ww, d=dd))
            r["cur"] += w
            max_u = max(max_u, r["cur"])
            wing_len[r["wing"]] = max(wing_len[r["wing"]], r["cur"])
            idx += 1
            placed_here += 1
        for r in rows:
            wing_len[r["wing"]] = max(wing_len[r["wing"]], r["cur"])

    for k in range(wings):
        v0 = v_off + k * (bar_w + gap)
        lay.wings.append(dict(zip("xywh", _to_xy(env, horizontal, 0, v0, wing_len[k], bar_w))))

    # accessible units: convert first N one-bedrooms at lowest level
    need_acc = max(1, math.ceil(0.02 * p.units))
    acc = 0
    for u in lay.units:
        if acc < need_acc and u["module"] == "ONEBR_A":
            u["module"] = "ACCESSIBLE_1BR"
            acc += 1

    # driveway
    if has_drive:
        lay.drive = dict(x=env["x0"], y=env["y0"], w=drive_w, d=env["d"]) if not horizontal else dict(x=env["x0"], y=env["y0"], w=env["w"], d=drive_w)
    elif L_avail - max(wing_len) >= drive_w:  # cross lane at the far end of the wings
        lay.drive = dict(x=env["x0"], y=env["y1"] - drive_w, w=env["w"], d=drive_w) if not horizontal else dict(x=env["x1"] - drive_w, y=env["y0"], w=drive_w, d=env["d"])

    # surface parking: scan free space
    obst = [(w["x"], w["y"], w["w"], w["h"]) for w in lay.wings]
    if lay.drive:
        obst.append((lay.drive["x"], lay.drive["y"], lay.drive["w"], lay.drive["d"]))
    z_ = ZONING[p.zoning]
    required = math.ceil(z_["parking"] * p.units)
    if podium:
        footprint = sum(w["w"] * w["h"] for w in lay.wings)
        lay.podium_capacity = int(footprint * 0.85 / 330)
    def free(r):
        x, y, w, h = r
        if x < env["x0"] or y < env["y0"] or x + w > env["x1"] or y + h > env["y1"]:
            return False
        return all(x + w <= a or a + c <= x or y + h <= b or b + e <= y for a, b, c, e in obst)
    stalls = []
    rows_y = sorted({env["y0"] + i * 42 for i in range(int(env["d"] // 42) + 1)} | {env["y1"] - 18 - i * 42 for i in range(int(env["d"] // 42) + 1)})
    need_surface = max(0, required - lay.podium_capacity)
    for y in rows_y:
        x = env["x0"]
        while x + 9 <= env["x1"] and len(stalls) < need_surface:
            r = (x, y, 9, 18)
            if free(r) and all(r[0] + 9 <= s[0] or s[0] + 9 <= r[0] or r[1] + 18 <= s[1] or s[1] + 18 <= r[1] for s in stalls):
                stalls.append(r)
                x += 9
            else:
                x += 1
    lay.stalls = [dict(x=s[0], y=s[1], w=9, d=18) for s in stalls]

    _checks(lay, seq, required, horizontal, max_u, core_len, need_acc, acc)
    return lay


def _checks(lay, seq, required, horizontal, max_u, core_len, need_acc, acc):
    p, env, z = lay.p, lay.env, ZONING[lay.p.zoning]
    lot_area = p.lot_w * p.lot_d
    footprint = sum(w["w"] * w["h"] for w in lay.wings)
    n_levels = p.floors + (1 if p.parking == "podium" else 0)
    gross = footprint * n_levels
    height = sum(s[2] for s in lay.storeys)
    placed = len(lay.units)
    in_env = all(w["x"] >= env["x0"] - .01 and w["y"] >= env["y0"] - .01 and w["x"] + w["w"] <= env["x1"] + .01 and w["y"] + w["h"] <= env["y1"] + .01 for w in lay.wings)
    provided = len(lay.stalls) + lay.podium_capacity
    travel = max_u - core_len
    def add(name, req, got, ok, warn=False):
        lay.checks.append(dict(check=name, required=req, provided=got, status="PASS" if ok else ("WARN" if warn else "FAIL")))
    add("Units placed", p.units, placed, placed >= p.units)
    add("Stories", f"<= {z['stories']}", n_levels, n_levels <= z["stories"])
    add("Height (ft)", f"<= {z['height']}", round(height, 1), height <= z["height"])
    add("FAR", f"<= {z['far']}", round(gross / lot_area, 2), gross / lot_area <= z["far"])
    add("Lot coverage %", f"<= {z['cov']}", round(100 * footprint / lot_area, 1), 100 * footprint / lot_area <= z["cov"])
    add("Parking stalls", required, provided, provided >= required)
    add("Building inside setbacks/easement", "yes", "yes" if in_env else "no", in_env)
    add("Fire lane / driveway (24 ft)", "yes", "yes" if lay.drive else "no", bool(lay.drive))
    add("Accessible units (2%)", need_acc, acc, acc >= need_acc)
    add("Exit travel to core (ft)", "<= 200", round(travel, 1), travel <= 200)
    lay.metrics = dict(units_placed=placed, footprint_sf=round(footprint), gross_sf=round(gross), far=round(gross / lot_area, 2),
                       coverage_pct=round(100 * footprint / lot_area, 1), height_ft=round(height, 1), parking_required=required,
                       parking_provided=provided, wings=len(lay.wings), stories=n_levels)


def _score(lay):
    ok = {c["check"]: c["status"] == "PASS" for c in lay.checks}
    return (ok["Units placed"], ok["Building inside setbacks/easement"], ok["Fire lane / driveway (24 ft)"], -lay.metrics["wings"])


def generate(p: Params) -> Layout:
    if p.wings:
        return _build(p, p.wings)
    return max((_build(p, k) for k in (1, 2, 3)), key=_score)


SCENARIOS = {
    "Baseline: 4 floors, surface parking": Params(),
    "Right-sized: 60 units, podium (compliant)": Params(units=60, floors=4, parking="podium"),
    "Podium parking, 5 floors": Params(floors=5, parking="podium"),
    "Dense R-6, 7 floors + podium": Params(zoning="Dense urban R-6 (sample)", floors=7, parking="podium"),
    "Suburban R-2, 60 units, 3 floors": Params(zoning="Suburban R-2 (sample)", units=60, floors=3),
}
