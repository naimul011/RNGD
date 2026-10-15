"""Room program for the optimizer.

Reads the QA format (`rooms_hotel.json`: name, width_ft, depth_ft, count, priority, optional
per_floor + floors, plus an `adjacency` list) and accepts optional extra fields per room:

  category   guest | ada | suite | service | public    (inferred from the name if absent)
  exterior   needs a window wall (default: true for guest rooms and public rooms)
  near_core  should open close to a core / elevator (default: service and ADA rooms)
  corner     prefers a corner (two exterior walls)
  min_per_floor  minimum on each allowed floor (e.g. ADA dispersion)

Adjacency entries are {"a": type or instance, "b": ..., "weight": w, "max_ft": d}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

GUEST_WORDS = ("king", "queen", "qq", "suite", "double", "guest", "room")
SERVICE_WORDS = ("janitor", "jan", "ice", "vending", "linen", "housekeeping", "it ", "it/", "elec", "idf", "storage",
                 "supplies", "mech")
PUBLIC_WORDS = ("lobby", "breakfast", "lounge", "fitness", "gym", "meeting", "office", "boh", "luggage", "restroom",
                "front desk", "market", "bar", "pool")


@dataclass
class RoomType:
    name: str
    width_ft: float
    depth_ft: float
    count: int
    priority: float = 3.0
    per_floor: bool = False
    floors: list[str] | None = None
    category: str = "guest"
    exterior: bool = True
    near_core: bool = False
    corner: bool = False
    min_per_floor: int = 0

    @property
    def area_sf(self) -> float:
        return self.width_ft * self.depth_ft

    @property
    def is_key(self) -> bool:
        return self.category in ("guest", "ada", "suite")


@dataclass
class Adjacency:
    a: str
    b: str
    weight: float = 1.0
    max_ft: float = 40.0


@dataclass
class Instance:
    name: str
    type: RoomType
    levels: list[int]   # allowed level indexes


@dataclass
class Program:
    rooms: list[RoomType]
    adjacency: list[Adjacency] = field(default_factory=list)
    door_width_ft: float = 3.0
    corridor_width_ft: float = 5.0
    exterior_slack_ft: float = 6.0
    source: str = ""
    width_along_corridor: bool = False   # True: width_ft is always the frontage (hotel schematic programs)

    def instances(self, level_names: list[str]) -> list[Instance]:
        out: list[Instance] = []
        for rt in self.rooms:
            allowed = [i for i, n in enumerate(level_names) if not rt.floors or n in rt.floors]
            if not allowed:
                continue
            if rt.per_floor:
                for i in allowed:
                    for k in range(rt.count):
                        suffix = f" {k + 1}" if rt.count > 1 else ""
                        out.append(Instance(f"{rt.name}{suffix} ({level_names[i]})", rt, [i]))
            else:
                for k in range(rt.count):
                    out.append(Instance(f"{rt.name} {k + 1}" if rt.count > 1 else rt.name, rt, allowed))
        return out

    def keys_requested(self, level_names: list[str]) -> int:
        return sum(1 for inst in self.instances(level_names) if inst.type.is_key)

    def total_area_sf(self, level_names: list[str]) -> float:
        return sum(inst.type.area_sf for inst in self.instances(level_names))


def _infer_category(name: str) -> str:
    n = f" {name.lower()} "
    if "ada" in n:
        return "ada"
    if any(w in n for w in SERVICE_WORDS):
        return "service"
    if any(w in n for w in PUBLIC_WORDS):
        return "public"
    if "suite" in n:
        return "suite"
    return "guest"


def program_from_dict(data: dict, source: str = "") -> Program:
    rooms = []
    for r in data["rooms"]:
        cat = r.get("category") or _infer_category(r["name"])
        rooms.append(RoomType(
            name=r["name"], width_ft=float(r["width_ft"]), depth_ft=float(r["depth_ft"]), count=int(r.get("count", 1)),
            priority=float(r.get("priority", 3)), per_floor=bool(r.get("per_floor", False)), floors=r.get("floors"),
            category=cat,
            exterior=bool(r.get("exterior", cat in ("guest", "ada", "suite", "public"))),
            near_core=bool(r.get("near_core", cat in ("service", "ada"))),
            corner=bool(r.get("corner", False)), min_per_floor=int(r.get("min_per_floor", 0))))
    adj = []
    for a in data.get("adjacency", []):
        if isinstance(a, dict):
            adj.append(Adjacency(a["a"], a["b"], float(a.get("weight", 1.0)), float(a.get("max_ft", 40.0))))
        elif isinstance(a, (list, tuple)) and len(a) >= 2:
            adj.append(Adjacency(a[0], a[1], float(a[2]) if len(a) > 2 else 1.0))
    return Program(rooms=rooms, adjacency=adj, door_width_ft=float(data.get("door_width_ft", 3.0)),
                   corridor_width_ft=float(data.get("corridor_width_ft", 5.0)),
                   exterior_slack_ft=float(data.get("exterior_slack_ft", 6.0)), source=source)


def load_program(path: str | Path) -> Program:
    p = Path(path)
    return program_from_dict(json.loads(p.read_text(encoding="utf-8")), source=p.name)


def program_to_dict(prog: Program) -> dict:
    return {
        "door_width_ft": prog.door_width_ft, "corridor_width_ft": prog.corridor_width_ft,
        "exterior_slack_ft": prog.exterior_slack_ft,
        "rooms": [{k: v for k, v in rt.__dict__.items() if v not in (None, False) or k in ("exterior",)}
                  for rt in prog.rooms],
        "adjacency": [a.__dict__ for a in prog.adjacency],
    }
