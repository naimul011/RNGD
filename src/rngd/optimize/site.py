"""Site grids in the QA format (`fp*_multi.json`).

Each level is an R x C grid of 1 ft cells labelled OUTSIDE, STRUCTURE, CORRIDOR, CORE or
PLACEABLE. Row 0 is the south edge (rows increase northward); column 0 is the west edge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage


@dataclass
class Core:
    index: int
    mask: np.ndarray          # bool R x C
    area_sf: float
    centroid: tuple[float, float]  # (row, col)
    access: np.ndarray = None  # core + its wall ring (structure within 2 cells): a door can pass the wall


@dataclass
class Site:
    name: str
    path: Path
    cell_ft: float
    origin_x_ft: float
    origin_y_ft: float
    level_names: list[str]
    elevations_ft: list[float]
    grids: list[np.ndarray]           # str arrays, one per level
    placeable: list[np.ndarray] = field(default_factory=list)   # bool per level
    outside: list[np.ndarray] = field(default_factory=list)
    blocked: list[np.ndarray] = field(default_factory=list)     # core | structure | outside
    ext_dist: list[np.ndarray] = field(default_factory=list)    # cells to nearest outside / grid edge
    cores: list[Core] = field(default_factory=list)              # from level 0 (cores stack)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grids[0].shape

    @property
    def n_levels(self) -> int:
        return len(self.grids)

    def placeable_sf(self, level: int = 0) -> float:
        return float(self.placeable[level].sum()) * self.cell_ft ** 2

    def diagonal_ft(self) -> float:
        rows, cols = np.where(~self.outside[0])
        return float(np.hypot(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)) * self.cell_ft


def load_site(path: str | Path) -> Site:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    meta = data["meta"]
    levels = data.get("levels") or [{"level": meta.get("level", "L1"), "grid": data["grid"], "elevation_ft": 0.0}]
    site = Site(name=p.stem, path=p, cell_ft=float(meta.get("cell_size_ft", 1.0)),
                origin_x_ft=float(meta.get("origin_x_ft", 0.0)), origin_y_ft=float(meta.get("origin_y_ft", 0.0)),
                level_names=[lv["level"] for lv in levels],
                elevations_ft=[float(lv.get("elevation_ft", 0.0)) for lv in levels],
                grids=[np.array(lv["grid"]) for lv in levels])
    return finalize(site)


def finalize(site: Site) -> Site:
    """Derive masks, facade distances and cores from site.grids."""
    for g in site.grids:
        outside = g == "OUTSIDE"
        site.placeable.append(g == "PLACEABLE")
        site.outside.append(outside)
        site.blocked.append((g == "CORE") | (g == "STRUCTURE") | outside)
        padded = np.pad(~outside, 1, constant_values=False)  # grid edge counts as outside
        site.ext_dist.append(ndimage.distance_transform_edt(padded)[1:-1, 1:-1])
    core = site.grids[0] == "CORE"
    lab, n = ndimage.label(core)
    for i in range(1, n + 1):
        m = lab == i
        r, c = np.where(m)
        wall = (site.grids[0] == "STRUCTURE") & ndimage.binary_dilation(m, iterations=2)
        site.cores.append(Core(index=i - 1, mask=m, area_sf=float(m.sum()) * site.cell_ft ** 2,
                               centroid=(float(r.mean()), float(c.mean())), access=m | wall))
    return site


def available_sites(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).glob("*_multi.json"))


def site_from_grids(name: str, grids: list[np.ndarray], level_names: list[str], elevations_ft: list[float],
                    cell_ft: float = 1.0) -> Site:
    """Build a Site from generated label grids (used by the hotel typology generator)."""
    site = Site(name=name, path=Path(f"{name}.generated"), cell_ft=cell_ft, origin_x_ft=0.0, origin_y_ft=0.0,
                level_names=list(level_names), elevations_ft=list(elevations_ft), grids=[np.asarray(g) for g in grids])
    return finalize(site)
