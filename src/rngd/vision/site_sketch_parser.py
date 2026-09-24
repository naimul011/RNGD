"""Classical computer-vision agent: reads a raster site sketch (the kind of
scanned survey/CAD export referenced in RNGD kickoff questions 17-18) and
extracts lot geometry by color-segmentation, independent of whatever numbers
are in site_conditions.json. The orchestrator then cross-checks the two so a
disagreement between "the drawing" and "the spec sheet" — common in real AEC
handoffs — gets caught automatically instead of silently trusting one source.

This is genuine pixel-level image processing (color thresholding + bounding
box extraction + a georeferenced pixel->feet transform), not an LLM
describing an image — see llm_groq.py's module docstring for why.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def _color_mask(arr: np.ndarray, rgb: tuple[int, int, int], tolerance: int) -> np.ndarray:
    diff = np.abs(arr[:, :, :3].astype(int) - np.array(rgb)).sum(axis=2)
    return diff <= tolerance


def _bbox_px(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _largest_component_bbox(mask: np.ndarray, dilate: int = 2) -> tuple[int, int, int, int] | None:
    """Isolates the largest connected shape in a binary mask by bounding-box
    area, so a thin scale bar or small text label (also matching the target
    color) doesn't get lumped into the lot-boundary measurement. A small
    dilation first bridges the boundary rectangle's 4 separate line segments
    (corners) into one connected component.
    """
    dilated = ndimage.binary_dilation(mask, iterations=dilate)
    labeled, n = ndimage.label(dilated)
    if n == 0:
        return None
    best_area, best_bbox = 0, None
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        area = (x1 - x0) * (y1 - y0)
        if area > best_area:
            best_area, best_bbox = area, (int(x0), int(y0), int(x1), int(y1))
    return best_bbox


def parse_site_sketch(png_path: Path, georef_path: Path) -> dict:
    georef = json.loads(Path(georef_path).read_text())
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img)

    px_per_ft = georef["px_per_ft"]
    ox, oy = georef["origin_px"]

    def px_to_ft(x_px: float, y_px: float) -> tuple[float, float]:
        return (x_px - ox) / px_per_ft, (y_px - oy) / px_per_ft

    # Lot boundary: near-black outline pixels.
    lot_mask = _color_mask(arr, tuple(georef["lot_boundary_color_rgb"]), georef["lot_boundary_color_tolerance"])
    lot_bbox_px = _largest_component_bbox(lot_mask)
    result: dict = {"source_image": str(png_path), "measurements": {}, "warnings": []}
    if lot_bbox_px is None:
        result["warnings"].append("Could not detect a lot boundary in the sketch.")
        return result

    x0f, y0f = px_to_ft(lot_bbox_px[0], lot_bbox_px[1])
    x1f, y1f = px_to_ft(lot_bbox_px[2], lot_bbox_px[3])
    frontage_ft = round(x1f - x0f, 1)
    depth_ft = round(y1f - y0f, 1)
    result["measurements"]["lot_frontage_ft"] = frontage_ft
    result["measurements"]["lot_depth_ft"] = depth_ft
    result["measurements"]["lot_area_sf"] = round(frontage_ft * depth_ft, 1)

    # Easement: distinct fill color.
    ease_mask = _color_mask(arr, tuple(georef["easement_color_rgb"]), georef["easement_color_tolerance"])
    ease_bbox_px = _largest_component_bbox(ease_mask)
    if ease_bbox_px is not None:
        ex0f, ey0f = px_to_ft(ease_bbox_px[0], ease_bbox_px[1])
        ex1f, ey1f = px_to_ft(ease_bbox_px[2], ease_bbox_px[3])
        ease_width_ft = round(ex1f - ex0f, 1)
        # Determine which side of the lot the easement hugs.
        lot_width_px = lot_bbox_px[2] - lot_bbox_px[0]
        dist_to_east_edge = abs(ease_bbox_px[2] - lot_bbox_px[2])
        dist_to_west_edge = abs(ease_bbox_px[0] - lot_bbox_px[0])
        side = "east" if dist_to_east_edge < dist_to_west_edge else "west"
        runs_full_depth = (ey1f - ey0f) >= 0.9 * depth_ft
        result["measurements"]["easement"] = {
            "side": side,
            "width_ft": ease_width_ft,
            "runs_full_depth": bool(runs_full_depth),
        }
    else:
        result["measurements"]["easement"] = None

    result["backend"] = "classical-cv:color-segmentation+georef-transform"
    return result


def cross_check_against_spec(cv_measurements: dict, site_conditions: dict, tolerance_ft: float = 2.0) -> dict:
    """tolerance_ft defaults to 2 ft to absorb line-stroke-width measurement
    noise inherent to reading a rasterized sketch (the boundary line itself is
    a few pixels wide), not because 2 ft of real discrepancy is acceptable."""
    """Flags disagreements between the CV-extracted geometry and the stated
    site_conditions.json, rather than assuming either is automatically right.
    """
    m = cv_measurements.get("measurements", {})
    lot = site_conditions.get("lot", {})
    discrepancies = []

    for field, spec_key in [("lot_frontage_ft", "frontage_ft"), ("lot_depth_ft", "depth_ft")]:
        cv_val = m.get(field)
        spec_val = lot.get(spec_key)
        if cv_val is not None and spec_val is not None and abs(cv_val - spec_val) > tolerance_ft:
            discrepancies.append(
                f"{field}: sketch measured {cv_val} ft vs. site_conditions.json states {spec_val} ft"
            )

    cv_ease = m.get("easement")
    spec_eases = site_conditions.get("easements", [])
    if cv_ease and spec_eases:
        spec_ease = spec_eases[0]
        if cv_ease["side"] != spec_ease.get("side"):
            discrepancies.append(
                f"easement side: sketch shows '{cv_ease['side']}' vs. spec states '{spec_ease.get('side')}'"
            )
        if abs(cv_ease["width_ft"] - spec_ease.get("width_ft", 0)) > tolerance_ft:
            discrepancies.append(
                f"easement width: sketch measured {cv_ease['width_ft']} ft vs. spec states {spec_ease.get('width_ft')} ft"
            )

    return {
        "consistent": len(discrepancies) == 0,
        "discrepancies": discrepancies,
    }
