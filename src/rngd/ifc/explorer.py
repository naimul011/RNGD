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
    "IfcWall", "IfcWallStandardCase", "IfcCurtainWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcRoof",
    "IfcColumn", "IfcBeam", "IfcStair", "IfcStairFlight", "IfcRailing",
    "IfcCovering", "IfcFurnishingElement", "IfcFlowTerminal", "IfcMember",
    "IfcPlate", "IfcBuildingElementProxy", "IfcSpace",
]

# Classes trusted to define "where the building actually is" — used to pick a
# recentering origin and to detect outliers. Furnishing/proxy/flow-terminal
# elements are excluded from this set on purpose: a misplaced furniture or
# MEP-fixture instance (e.g. an Enscape people asset dropped at a stray
# coordinate by mistake in the source model) is a common real-world IFC
# authoring error, and one such point hundreds of meters from the building
# is enough to make an unguarded auto-fit view zoom out until the actual
# building looks like a speck.
STRUCTURAL_CLASSES = {
    "IfcWall", "IfcWallStandardCase", "IfcCurtainWall", "IfcSlab", "IfcRoof",
    "IfcDoor", "IfcWindow", "IfcColumn", "IfcBeam", "IfcStair", "IfcStairFlight",
}


def list_available_files() -> list[Path]:
    files = []
    sample_dir = config.DATA_DIR / "ifc_samples"
    if sample_dir.exists():
        files += sorted(sample_dir.glob("*.ifc"))
    gen_dir = config.OUTPUTS_DIR / "ifc_design"
    if gen_dir.exists():
        files += sorted(gen_dir.glob("*.ifc"))
    return files


def _cache_path(path: Path, skip_framing: bool, recenter: bool) -> Path:
    st = path.stat()
    key = f"{path.stem}_{st.st_size}_{int(st.st_mtime)}_{'nf' if skip_framing else 'full'}_{'rc' if recenter else 'raw'}"
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


def _recenter(meshes: list[dict], outlier_factor: float = 4.0, min_outlier_dist_m: float = 50.0) -> tuple[list[dict], dict]:
    """Translates every mesh so the building sits near the origin, and drops
    elements sitting implausibly far from the rest of the model (see
    STRUCTURAL_CLASSES above). This is a display-space transform applied only
    to the in-memory/cached mesh data — the source .ifc file on disk is never
    modified. Returns (kept_meshes, report)."""
    structural = [m for m in meshes if m["cls"] in STRUCTURAL_CLASSES]
    reference = structural or meshes
    ref_verts = np.concatenate([m["verts"] for m in reference], axis=0)
    origin = np.median(ref_verts, axis=0)
    diag = float(np.linalg.norm(ref_verts.max(axis=0) - ref_verts.min(axis=0)))
    threshold = max(diag * outlier_factor, min_outlier_dist_m)

    kept, dropped = [], []
    for m in meshes:
        centroid = m["verts"].mean(axis=0)
        dist = float(np.linalg.norm(centroid - origin))
        if dist > threshold:
            dropped.append({"name": m["name"], "cls": m["cls"], "distance_m": round(dist, 1)})
            continue
        m = dict(m, verts=m["verts"] - origin)
        kept.append(m)

    report = {
        "origin_offset_m": [round(x, 2) for x in origin.tolist()],
        "building_diagonal_m": round(diag, 1),
        "excluded_outliers": dropped,
    }
    return kept, report


def load(path: str | Path, skip_framing: bool = True, workers: int = 8, recenter: bool = True) -> tuple[dict, list[dict]]:
    """Returns (summary, meshes). Cached to disk keyed by file identity +
    skip_framing, so a repeat open of the same unchanged file is instant."""
    path = Path(path)
    cache = _cache_path(path, skip_framing, recenter)
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    f = ifcopenshell.open(str(path))
    summary = summarize(f)
    meshes = _triangulate(f, skip_framing=skip_framing, workers=workers)
    summary["meshes_triangulated"] = len(meshes)
    summary["framing_skipped"] = skip_framing

    if recenter:
        meshes, recenter_report = _recenter(meshes)
        summary.update(recenter_report)
        summary["meshes_triangulated"] = len(meshes)

    with open(cache, "wb") as fh:
        pickle.dump((summary, meshes), fh, protocol=4)
    return summary, meshes
