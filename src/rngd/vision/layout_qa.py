"""Classical CV QA agent for the *output* side: re-measures the rendered
site_plan.png independently of the Python objects that drew it, so a
rendering bug (or a future swap to a different renderer) can't silently slip
a bad drawing past the pipeline. Uses color segmentation + connected-component
counting (skimage/scipy) plus the pixel<->feet calibration written by
renderer.py, then checks the measurements against the hard site-level rules
in baap_stacking_rules.md (setback/easement encroachment, parking count).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ..simulation.renderer import COLOR_BUILDING, COLOR_EASEMENT, COLOR_PARKING


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _color_mask(arr: np.ndarray, rgb: tuple[int, int, int], tolerance: int = 30) -> np.ndarray:
    diff = np.abs(arr[:, :, :3].astype(int) - np.array(rgb)).sum(axis=2)
    return diff <= tolerance


class PixelToFeet:
    def __init__(self, geom_sidecar: dict):
        (x0_ft, y0_ft), (x1_ft, y1_ft) = geom_sidecar["ref_ft"]
        (x0_px, y0_px), (x1_px, y1_px) = geom_sidecar["ref_px"]
        self.sx = (x1_ft - x0_ft) / (x1_px - x0_px)
        self.sy = (y1_ft - y0_ft) / (y1_px - y0_px)
        self.x0_ft, self.x0_px = x0_ft, x0_px
        self.y0_ft, self.y0_px = y0_ft, y0_px

    def to_ft(self, x_px: float, y_px: float) -> tuple[float, float]:
        x_ft = self.x0_ft + (x_px - self.x0_px) * self.sx
        y_ft = self.y0_ft + (y_px - self.y0_px) * self.sy
        return x_ft, y_ft


def _bbox_ft(mask: np.ndarray, xform: PixelToFeet) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0f, y0f = xform.to_ft(xs.min(), ys.min())
    x1f, y1f = xform.to_ft(xs.max(), ys.max())
    return min(x0f, x1f), min(y0f, y1f), max(x0f, x1f), max(y0f, y1f)


def qa_site_plan(
    png_path: Path,
    buildable_envelope: dict,
    lot: dict,
    required_parking_stalls: int,
) -> dict:
    geom = json.loads(Path(str(png_path) + ".geom.json").read_text())
    xform = PixelToFeet(geom)
    img = np.array(Image.open(png_path).convert("RGB"))

    result: dict = {
        "backend": "classical-cv:color-segmentation+connected-components",
        "footprint_area_sf": 0.0,
        "lot_coverage_pct": 0.0,
        "setback_violations": [],
        "easement_violations": [],
        "overlap_violations": [],
        "parking_stalls_detected": 0,
        "parking_stalls_required": required_parking_stalls,
        "parking_shortfall": required_parking_stalls,
        "geometry_passed": True,
    }

    building_mask = _color_mask(img, _hex_to_rgb(COLOR_BUILDING))
    bbox = _bbox_ft(building_mask, xform)
    if bbox is None:
        result["setback_violations"].append("Could not detect a building footprint in the rendered plan.")
        result["geometry_passed"] = False
        return result

    x0, y0, x1, y1 = bbox
    footprint_w, footprint_h = (x1 - x0), (y1 - y0)
    result["footprint_area_sf"] = round(footprint_w * footprint_h, 1)
    result["lot_coverage_pct"] = round(100 * result["footprint_area_sf"] / lot["area_sf"], 1)

    env = buildable_envelope
    tol = 1.0  # feet, absorbs anti-aliasing / line-width pixel noise
    if x0 < env["x_min_ft"] - tol or x1 > env["x_max_ft"] + tol:
        result["setback_violations"].append(
            f"Building x-extent [{x0:.1f}, {x1:.1f}] exceeds buildable envelope "
            f"[{env['x_min_ft']:.1f}, {env['x_max_ft']:.1f}]"
        )
    if y0 < env["y_min_ft"] - tol or y1 > env["y_max_ft"] + tol:
        result["setback_violations"].append(
            f"Building y-extent [{y0:.1f}, {y1:.1f}] exceeds buildable envelope "
            f"[{env['y_min_ft']:.1f}, {env['y_max_ft']:.1f}]"
        )

    easement_mask = _color_mask(img, _hex_to_rgb(COLOR_EASEMENT))
    overlap = building_mask & easement_mask
    if overlap.sum() > 5:  # a handful of anti-aliased boundary pixels is not a real encroachment
        result["easement_violations"].append(
            f"Building footprint overlaps a mapped easement ({int(overlap.sum())} overlapping px)"
        )

    parking_mask = _color_mask(img, _hex_to_rgb(COLOR_PARKING))
    labeled, n_stalls = ndimage.label(parking_mask)
    # Filter tiny specks (anti-aliasing noise) below a plausible stall's pixel area.
    sizes = ndimage.sum(parking_mask, labeled, range(1, n_stalls + 1))
    real_stalls = int((sizes > (sizes.max() * 0.15) if len(sizes) else []).sum()) if n_stalls else 0
    result["parking_stalls_detected"] = real_stalls
    result["parking_shortfall"] = max(0, required_parking_stalls - real_stalls)

    if result["setback_violations"] or result["easement_violations"] or result["overlap_violations"]:
        result["geometry_passed"] = False
    return result
