"""Open an arbitrary real-world .ifc file (not one this app generated) for
viewing and Q&A: triangulate its geometry, summarize what's in it (schema,
storeys, element categories, notable named items), and cache both to disk so
a large file only pays the triangulation cost once.

This is deliberately separate from ifc/design.py (our own parametric
generator, which writes files in a controlled naming scheme) and
ifc/ifc_writer.py's read_meshes (which assumes that scheme) — a real client
file has arbitrary IFC classes and names, so summarization here works off the
IFC schema itself (element classes, storeys, property sets) instead.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import numpy as np

from .. import config

CACHE_DIR = config.OUTPUTS_DIR / "ifc_cache"

# Curtain-wall framing (mullions/panels) can be hundreds of tiny elements;
# skipping them gives a much faster first pass at a small visual cost — the
# wall/roof/slab/door/window massing is what matters most for orientation.
FRAMING_CLASSES = {"IfcMember", "IfcPlate"}

CATEGORY_CLASSES = [
    "IfcWall", "IfcCurtainWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcRoof",
    "IfcColumn", "IfcBeam", "IfcStair", "IfcStairFlight", "IfcRailing",
    "IfcCovering", "IfcFurnishingElement", "IfcFlowTerminal", "IfcMember",
    "IfcPlate", "IfcBuildingElementProxy", "IfcSpace",
]


def list_available_files() -> list[Path]:
    files = []
    sample_dir = config.DATA_DIR / "ifc_samples"
    if sample_dir.exists():
        files += sorted(sample_dir.glob("*.ifc"))
    gen_dir = config.OUTPUTS_DIR / "ifc_design"
    if gen_dir.exists():
        files += sorted(gen_dir.glob("*.ifc"))
    return files


def _cache_path(path: Path, skip_framing: bool) -> Path:
    st = path.stat()
    key = f"{path.stem}_{st.st_size}_{int(st.st_mtime)}_{'nf' if skip_framing else 'full'}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.pkl"


def summarize(f: ifcopenshell.file) -> dict:
    def names(cls):
        try:
            return f.by_type(cls)
        except RuntimeError:
            return []

    project = f.by_type("IfcProject")
    site = f.by_type("IfcSite")
    building = f.by_type("IfcBuilding")
    storeys = sorted(
        [{"name": s.Name, "elevation_m": round(s.Elevation or 0.0, 2)} for s in names("IfcBuildingStorey")],
        key=lambda s: s["elevation_m"],
    )
    categories = {cls: len(names(cls)) for cls in CATEGORY_CLASSES if len(names(cls))}

    # A curated peek at named furnishing/proxy elements — useful "what's
    # actually in here" context for the chat assistant (e.g. "ATM MACHINE").
    notable = {}
    for cls in ("IfcFurnishingElement", "IfcBuildingElementProxy", "IfcFlowTerminal"):
        for el in names(cls):
            base = (el.Name or "Unnamed").split(":")[0]
            notable[base] = notable.get(base, 0) + 1
    notable_top = sorted(notable.items(), key=lambda x: -x[1])[:20]

    return {
        "schema": f.schema,
        "project_name": project[0].Name if project else None,
        "site_name": site[0].Name if site else None,
        "building_name": (building[0].Name if building and building[0].Name else None),
        "storeys": storeys,
        "categories": categories,
        "total_elements": len(f.by_type("IfcElement")),
        "notable_named_elements": notable_top,
    }


def _triangulate(f: ifcopenshell.file, skip_framing: bool, workers: int = 8) -> list[dict]:
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)

    elements = {e.id(): e for e in f.by_type("IfcElement") if not (skip_framing and e.is_a() in FRAMING_CLASSES)}
    if not elements:
        return []

    out = []
    iterator = ifcopenshell.geom.iterator(settings, f, workers, include=list(elements.values()))
    if iterator.initialize():
        while True:
            shape = iterator.get()
            el = elements.get(shape.id)
            if el is not None:
                v = np.array(shape.geometry.verts, dtype=float).reshape(-1, 3)
                t = np.array(shape.geometry.faces, dtype=np.int32).reshape(-1, 3)
                out.append({"name": el.Name or el.is_a(), "cls": el.is_a(), "verts": v, "tris": t})
            if not iterator.next():
                break
    return out


def load(path: str | Path, skip_framing: bool = True, workers: int = 8) -> tuple[dict, list[dict]]:
    """Returns (summary, meshes). Cached to disk keyed by file identity +
    skip_framing, so a repeat open of the same unchanged file is instant."""
    path = Path(path)
    cache = _cache_path(path, skip_framing)
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    f = ifcopenshell.open(str(path))
    summary = summarize(f)
    meshes = _triangulate(f, skip_framing=skip_framing, workers=workers)
    summary["meshes_triangulated"] = len(meshes)
    summary["framing_skipped"] = skip_framing

    with open(cache, "wb") as fh:
        pickle.dump((summary, meshes), fh, protocol=4)
    return summary, meshes
