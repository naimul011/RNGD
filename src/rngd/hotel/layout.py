"""Schematic generator for a hotel program: double-loaded corridor per level.

Deterministic and rule based, modelled on the Project Renegade L+L plans:
- Upper levels: guest rooms on both sides of a 6 ft corridor. Room depth is fixed
  (26 ft), room width = area / depth. A core at the west end holds Stair 1, two
  elevators and shafts; the per-floor support rooms (Ice, IT, Jan, Supplies) sit
  opposite it. Stair 2 is at the east end of the north side, so the corridor has
  an exit at each end.
- Level 1: the same core stacked below, the remaining shared / BOH / business
  rooms packed along the corridor, and the tenant space (retail) as a full-depth
  block at the east end.
Rooms carry their program area (`sf`) when it differs from the drawn area, so
the drawings can print what the program asked for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .program import Program

ROOM_DEPTH_FT = 26.0
CORRIDOR_FT = 6.0
CORE_W_FT = 34.0
STAIR_W_FT = 10.0
ELEV_FT = 8.0
SUPPORT_DEPTH_FT = 12.0
DOOR_FT = 3.0
WING_W_FT = 2 * ROOM_DEPTH_FT + CORRIDOR_FT
GROUND_LEVEL = "Level 1"

# Program category -> drawing kind (colour). Names refine it below.
_KIND = {"shared": "shared", "boh": "service", "business": "business", "tenant": "tenant", "guest": "guest"}


@dataclass
class Room:
    label: str
    x: float
    y: float
    w: float
    d: float
    kind: str  # guest | corridor | shared | business | service | tenant | stair | elevator | shaft | flex
    sf: float | None = None  # program area to print; None = drawn area
    side: str = ""  # guest rooms: "S" (corridor above, y = d) or "N" (corridor below, y = 0)

    @property
    def area(self) -> float:
        return self.sf if self.sf is not None else self.w * self.d


@dataclass
class Level:
    name: str
    rooms: list[Room] = field(default_factory=list)
    keys_placed: int = 0
    length_ft: float = 0.0
    width_ft: float = 0.0
    elevation_ft: float = 0.0
    height_ft: float = 0.0

    @property
    def gross_sf(self) -> float:
        return self.length_ft * self.width_ft


@dataclass
class Schematic:
    program: Program
    levels: list[Level]
    site: dict
    unplaced_keys: int
    max_travel_ft: float
    stories: int
    height_ft: float
    footprint_sf: float
    dead_end_ft: float = 0.0
    stair_doors_x: tuple[float, float] = (0.0, 0.0)
    checks: list[dict] = field(default_factory=list)


def _kind_for(name: str, cat: str) -> str:
    n = name.lower()
    if "stair" in n:
        return "stair"
    if "corridor" in n:
        return "corridor"
    return _KIND.get(cat, "shared")


def split_rows(program: Program, upper_levels: int):
    """Sort non-guest rows into: per-floor support (small rooms with count >= 2), core
    support on Level 1 (small BOH rooms), Level 1 band rooms, tenant blocks.
    Stairs and corridors are generated, not packed."""
    per_floor, core_l1, band, tenant = [], [], [], []
    for name, count, area, cat in program.rows:
        if cat == "guest" or count == 0:
            continue
        kind = _kind_for(name, cat)
        if kind in ("stair", "corridor"):
            continue
        each = area / count
        if count >= 2 and each < 150:
            # Small repeated rooms (Ice, IT, Jan, Supplies): one per guest floor, total area kept.
            per_floor.append((name, area / upper_levels, kind))
        elif cat == "boh" and area < 150:
            core_l1.append((name, area, kind))
        elif kind == "tenant":
            tenant.append((name, area, kind))
        else:
            band += [(name, each, kind)] * count
    return per_floor, core_l1, band, tenant


def _core(x_len: float) -> list[Room]:
    """West-end core, identical on every level so stairs and shafts stack."""
    S, E = STAIR_W_FT, ELEV_FT
    return [
        Room("Stair 1", 0, 0, S, ROOM_DEPTH_FT, "stair"),
        Room("Elev", S, ROOM_DEPTH_FT - E, E, E, "elevator"),
        Room("Elev", S + E, ROOM_DEPTH_FT - E, E, E, "elevator"),
        Room("Elec / Shaft", S, 0, 2 * E, ROOM_DEPTH_FT - E, "shaft"),
        Room("Mech", S + 2 * E, 0, CORE_W_FT - S - 2 * E, ROOM_DEPTH_FT, "shaft"),
        Room("Corridor", 0, ROOM_DEPTH_FT, x_len, CORRIDOR_FT, "corridor"),
    ]


def _support_strip(items, x0: float, x1: float, y0: float) -> list[Room]:
    """Small support rooms in one strip of SUPPORT_DEPTH_FT, widths scaled to fill x0..x1,
    plus the leftover depth as unprogrammed storage."""
    rooms = []
    total = sum(a for _, a, _ in items)
    if total > 0:
        x = x0
        for name, a, kind in items:
            w = (x1 - x0) * a / total
            rooms.append(Room(name, x, y0, w, SUPPORT_DEPTH_FT, kind, sf=a))
            x += w
        rooms.append(Room("Storage (unprogrammed)", x0, y0 + SUPPORT_DEPTH_FT, x1 - x0,
                          ROOM_DEPTH_FT - SUPPORT_DEPTH_FT, "flex"))
    else:
        rooms.append(Room("Storage (unprogrammed)", x0, y0, x1 - x0, ROOM_DEPTH_FT, "flex"))
    return rooms


def _pack_band(items, x0: float, x1: float, y: float, side: str) -> tuple[list[Room], float]:
    rooms, x = [], x0
    for name, a, kind in items:
        w = a / ROOM_DEPTH_FT
        rooms.append(Room(name, x, y, w, ROOM_DEPTH_FT, kind, sf=a, side=side))
        x += w
    if x1 - x > 1.0:
        rooms.append(Room("Linen / flex", x, y, x1 - x, ROOM_DEPTH_FT, "flex", side=side))
    return rooms, x


def build(program: Program, site: dict, upper_levels: int = 5, floor_height_ft: float = 10.75,
          ground_height_ft: float = 16.0) -> Schematic:
    """site: dict(envelope_w, envelope_d) in feet (buildable envelope)."""
    per_floor, core_l1, band_items, tenant = split_rows(program, upper_levels)
    keys = program.guest_rooms()
    keys.sort(key=lambda r: -r["area_sf"])
    per = [[] for _ in range(upper_levels)]
    for i, r in enumerate(keys):
        per[i % upper_levels].append(r)

    ys, yn = 0.0, ROOM_DEPTH_FT + CORRIDOR_FT
    # First pass: guest wings, balanced so both sides end near the same x.
    wings = []
    for group in per:
        sides = {"S": [], "N": []}
        run = {"S": 0.0, "N": STAIR_W_FT}  # north side also has to fit Stair 2
        for r in group:
            s = "S" if run["S"] <= run["N"] else "N"
            sides[s].append(r)
            run[s] += r["area_sf"] / ROOM_DEPTH_FT
        wings.append((sides, run))
    length = CORE_W_FT + max(max(run.values()) for _, run in wings)

    levels: list[Level] = []
    for idx, (sides, run) in enumerate(wings):
        lvl = Level(name=f"Level {idx + 2}", length_ft=length, width_ft=WING_W_FT,
                    elevation_ft=ground_height_ft + idx * floor_height_ft, height_ft=floor_height_ft)
        lvl.rooms += _core(length)
        lvl.rooms += _support_strip(per_floor, STAIR_W_FT, CORE_W_FT, yn)
        lvl.rooms.append(Room("Stair 2", length - STAIR_W_FT, yn, STAIR_W_FT, ROOM_DEPTH_FT, "stair"))
        lvl.rooms.append(Room("Linen / flex", 0, yn, STAIR_W_FT, ROOM_DEPTH_FT, "flex"))
        for s, y, end in (("S", ys, length), ("N", yn, length - STAIR_W_FT)):
            x = CORE_W_FT
            for r in sides[s]:
                w = r["area_sf"] / ROOM_DEPTH_FT
                lvl.rooms.append(Room(r["type"], x, y, w, ROOM_DEPTH_FT, "guest", sf=r["area_sf"], side=s))
                x += w
                lvl.keys_placed += 1
            if end - x > 1.0:
                lvl.rooms.append(Room("Linen / flex", x, y, end - x, ROOM_DEPTH_FT, "flex", side=s))
        levels.append(lvl)

    # Level 1: core stacked, Stair 2 stacked, band rooms between, tenant block at the east.
    ground = Level(name=GROUND_LEVEL, width_ft=WING_W_FT, elevation_ft=0.0, height_ft=ground_height_ft)
    lobby = [b for b in band_items if "lobby" in b[0].lower()]
    rest = sorted([b for b in band_items if "lobby" not in b[0].lower()], key=lambda b: -b[1])
    south, north = list(lobby), []
    run = {"S": sum(a for _, a, _ in south) / ROOM_DEPTH_FT, "N": 0.0}
    for b in rest:
        s = "S" if run["S"] <= run["N"] else "N"
        (south if s == "S" else north).append(b)
        run[s] += b[1] / ROOM_DEPTH_FT
    ground.rooms += _support_strip(core_l1, STAIR_W_FT, CORE_W_FT, yn)
    ground.rooms.append(Room("Linen / flex", 0, yn, STAIR_W_FT, ROOM_DEPTH_FT, "flex"))
    stair2_x0 = length - STAIR_W_FT
    band_end = CORE_W_FT + max(run["S"], run["N"])
    tenant_sf = sum(a for _, a, _ in tenant)
    stair_sf = STAIR_W_FT * ROOM_DEPTH_FT
    if stair2_x0 >= band_end:
        # Stair 2 falls east of the band rooms: the east block (tenant, or flex if none)
        # wraps around it, full depth, and must reach at least the tower length.
        ground.length_ft = max(length, band_end + (tenant_sf + stair_sf) / WING_W_FT)
        ground.rooms += _core(band_end)
        ground.rooms += _pack_band(south, CORE_W_FT, band_end, ys, "S")[0]
        ground.rooms += _pack_band(north, CORE_W_FT, band_end, yn, "N")[0]
        name = " + ".join(t[0] for t in tenant) or "Flex (unprogrammed)"
        kind = "tenant" if tenant else "flex"
        G = ground.length_ft
        ground.rooms.append(Room(name, band_end, 0, G - band_end, yn, kind, sf=tenant_sf or None))
        if stair2_x0 - band_end > 0.5:
            ground.rooms.append(Room("", band_end, yn, stair2_x0 - band_end, ROOM_DEPTH_FT, kind))
        if G - length > 0.5:
            ground.rooms.append(Room("", length, yn, G - length, ROOM_DEPTH_FT, kind))
    else:
        # Band rooms run past Stair 2: the north band skips the stair slot.
        band_end = max(length, CORE_W_FT + max(run["S"], run["N"] + STAIR_W_FT))
        ground.length_ft = band_end + tenant_sf / WING_W_FT
        ground.rooms += _core(band_end)
        ground.rooms += _pack_band(south, CORE_W_FT, band_end, ys, "S")[0]
        x = CORE_W_FT
        for name, a, kind in north:
            w = a / ROOM_DEPTH_FT
            if x < stair2_x0 + STAIR_W_FT and x + w > stair2_x0:
                if stair2_x0 - x > 1.0:
                    ground.rooms.append(Room("Linen / flex", x, yn, stair2_x0 - x, ROOM_DEPTH_FT, "flex", side="N"))
                x = length
            ground.rooms.append(Room(name, x, yn, w, ROOM_DEPTH_FT, kind, sf=a, side="N"))
            x += w
        if band_end - x > 1.0:
            ground.rooms.append(Room("Linen / flex", x, yn, band_end - x, ROOM_DEPTH_FT, "flex", side="N"))
        tx = band_end
        for name, a, kind in tenant:
            w = a / WING_W_FT
            ground.rooms.append(Room(name, tx, 0, w, WING_W_FT, kind, sf=a))
            tx += w
    ground.rooms.append(Room("Stair 2", stair2_x0, yn, STAIR_W_FT, ROOM_DEPTH_FT, "stair"))
    levels.insert(0, ground)

    # Egress geometry: stair doors open off the corridor at mid-stair-width.
    s1_door = STAIR_W_FT / 2
    s2_door = length - STAIR_W_FT / 2
    dead_end = max(s1_door, length - s2_door)
    travel = (s2_door - s1_door) / 2 + ROOM_DEPTH_FT  # corridor run to nearest stair + walk inside the room
    stories = upper_levels + 1
    height = ground_height_ft + floor_height_ft * upper_levels
    unplaced = len(keys) - sum(lv.keys_placed for lv in levels[1:])
    footprint = max(lv.gross_sf for lv in levels)

    sch = Schematic(program=program, levels=levels, site=site, unplaced_keys=unplaced, max_travel_ft=travel,
                    stories=stories, height_ft=height, footprint_sf=footprint, dead_end_ft=dead_end,
                    stair_doors_x=(s1_door, s2_door))
    from .code_check import evaluate  # local import: code_check reads Schematic
    sch.checks = evaluate(sch)
    return sch


def max_length(sch: Schematic) -> float:
    return max(lv.length_ft for lv in sch.levels)


def diagonal(sch: Schematic) -> float:
    return math.hypot(sch.levels[1].length_ft, sch.levels[1].width_ft)
