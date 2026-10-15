"""Hotel program: room types, counts, areas, and gross-up to building area.

Default program = "Project Renegade Living + Lodging Prototype", cover sheet
(docs/General/.../Scheme 2 - 223 Hotel, Redesigned/Project Renegade
Living+Lodging_Drawings.pdf, page 1, dated 06/20/25). Uploaded programs
(CSV or XLSX with columns room_type, count, area_sf, category) replace it.
area_sf is the row total for that room type (same convention as the source table).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SOURCE = "Project Renegade Living + Lodging Prototype, cover sheet (06/20/25)"

# Gross area per level from the same cover sheet.
GROSS_BY_LEVEL = {"Level 1": 12485, "Level 2": 12342, "Level 3": 12342,
                  "Level 4": 12342, "Level 5": 12342, "Level 6": 12342}

# (room_type, count, total_area_sf, category), row for row from the cover
# sheet's "Space Allocation" schedule. Areas are the row total for that type.
# Categories follow the schedule's groups: shared, boh, tenant, guest
# ("Hotel Units") and business.
DEFAULT_ROWS = [
    ("Corridor", 1, 1335, "shared"),
    ("Gym", 1, 968, "shared"),
    ("Lobby", 1, 2865, "shared"),
    ("Mens", 1, 205, "shared"),
    ("Womens", 1, 200, "shared"),
    ("BOH", 1, 69, "boh"),
    ("Hotel Kitchen / Pantry", 1, 331, "boh"),
    ("Housekeeping", 1, 103, "boh"),
    ("Ice", 5, 297, "boh"),
    ("IT", 5, 372, "boh"),
    ("Jan", 5, 269, "boh"),
    ("Janitor", 1, 107, "boh"),
    ("Loading", 1, 282, "boh"),
    ("MEP Service", 1, 309, "boh"),
    ("Stair", 12, 1868, "boh"),
    ("Supplies", 5, 394, "boh"),
    ("White Box Retail", 1, 3533, "tenant"),
    ("ADA Corner Suite", 1, 852, "guest"),
    ("ADA King Spa", 2, 920, "guest"),
    ("ADA QQ", 2, 892, "guest"),
    ("ADA Suite", 2, 1278, "guest"),
    ("Corner Suite", 2, 1728, "guest"),
    ("King", 69, 23692, "guest"),
    ("King Spa", 3, 1379, "guest"),
    ("QQ", 33, 11342, "guest"),
    ("Suite", 3, 1920, "guest"),
    ("BOH / Bus", 1, 1718, "business"),
    ("Luggage", 1, 122, "business"),
]
GROUP_TITLES = {"shared": "Shared", "boh": "BOH", "tenant": "Tenant", "guest": "Hotel Units",
                "business": "Business"}

# Sizing rules from the 06/2026 kickoff meeting (Matthew Edmonds). These are
# NOT client-confirmed; the report flags them as open questions.
MEETING_RULES = {"circulation_pct": 33.0, "wall_burden_pct": 8.0, "round_to_sf": 100}
SOURCE_RATIO_NOTE = "Source sheets: 74,197 SF gross vs 59,350 SF net (ratio 1.25)."


@dataclass
class Program:
    rows: list[tuple[str, int, float, str]]
    source: str = SOURCE
    gross_by_level: dict[str, float] = field(default_factory=lambda: dict(GROSS_BY_LEVEL))

    @property
    def keys(self) -> int:
        return sum(c for _, c, _, cat in self.rows if cat == "guest")

    @property
    def net_sf(self) -> float:
        return float(sum(a for _, _, a, _ in self.rows))

    @property
    def gross_sf(self) -> float:
        return float(sum(self.gross_by_level.values()))

    def guest_rooms(self) -> list[dict]:
        """One entry per key, with its own footprint area (type average)."""
        out = []
        for name, count, area, cat in self.rows:
            if cat != "guest" or count == 0:
                continue
            per = area / count
            out += [{"type": name, "area_sf": per}] * count
        return out

    def net_to_gross(self, circulation_pct: float, wall_burden_pct: float, round_to_sf: int = 100) -> dict:
        raw = self.net_sf * (1 + circulation_pct / 100) * (1 + wall_burden_pct / 100)
        required = math.ceil(raw / round_to_sf) * round_to_sf
        return {"net_sf": self.net_sf, "circulation_pct": circulation_pct,
                "wall_burden_pct": wall_burden_pct, "required_gross_sf": required,
                "source_gross_sf": self.gross_sf}


def default_program() -> Program:
    return Program(rows=list(DEFAULT_ROWS))


GUEST_WORDS = ("king", "queen", "qq", "suite", "spa", "guest")


def _infer_category(name: str) -> str:
    return "guest" if any(w in name.lower() for w in GUEST_WORDS) else "shared"


def load_program_file(path: str | Path) -> Program:
    """CSV or XLSX with columns room_type, count, area_sf, category (category optional)."""
    p = Path(path)
    df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"room_type", "count", "area_sf"} - set(df.columns)
    if missing:
        raise ValueError(f"Program file is missing columns: {sorted(missing)}")
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["room_type"]):
            continue
        name = str(r["room_type"]).strip()
        cat = str(r["category"]).strip().lower() if "category" in df.columns and not pd.isna(r.get("category")) else _infer_category(name)
        rows.append((name, int(r["count"]), float(r["area_sf"]), cat))
    if not rows:
        raise ValueError("Program file has no rows.")
    return Program(rows=rows, source=f"Uploaded file: {p.name}", gross_by_level=dict(GROSS_BY_LEVEL))
