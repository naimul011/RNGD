"""
Generates the synthetic site-sketch raster (+ georeference sidecar) that the
Vision/CV input agent parses. Standing in for a scanned survey/CAD export until
RNGD provides a real one (kickoff questions 17-18).

Run once: python src/make_sample_data.py
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT / "data" / "sample_projects" / "project_001"

PX_PER_FT = 3.0
ORIGIN_PX = (100, 100)  # pixel location of ft-coordinate (0, 0)
LOT_FRONTAGE_FT = 200
LOT_DEPTH_FT = 300
EASEMENT_SIDE = "east"
EASEMENT_WIDTH_FT = 20


def ft_to_px(x_ft, y_ft):
    return (ORIGIN_PX[0] + x_ft * PX_PER_FT, ORIGIN_PX[1] + y_ft * PX_PER_FT)


def main():
    width_px = int(ORIGIN_PX[0] * 2 + LOT_FRONTAGE_FT * PX_PER_FT)
    height_px = int(ORIGIN_PX[1] * 2 + LOT_DEPTH_FT * PX_PER_FT + 60)

    img = Image.new("RGB", (width_px, height_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    lot_tl = ft_to_px(0, 0)
    lot_br = ft_to_px(LOT_FRONTAGE_FT, LOT_DEPTH_FT)

    # Easement strip (east side, full depth) — distinct solid color for CV segmentation
    ease_x0_ft = LOT_FRONTAGE_FT - EASEMENT_WIDTH_FT
    ease_tl = ft_to_px(ease_x0_ft, 0)
    ease_br = ft_to_px(LOT_FRONTAGE_FT, LOT_DEPTH_FT)
    EASEMENT_COLOR = (255, 191, 0)
    draw.rectangle([ease_tl, ease_br], fill=EASEMENT_COLOR)
    # hatch texture on top (cosmetic only; segmentation uses base color, not hatch)
    for i in range(0, int(LOT_DEPTH_FT * PX_PER_FT), 14):
        y = ease_tl[1] + i
        draw.line([(ease_tl[0], y), (ease_br[0], y - 20)], fill=(200, 140, 0), width=1)

    # Lot boundary
    draw.rectangle([lot_tl, lot_br], outline=(0, 0, 0), width=4)

    # Street frontage marker (north edge) — thick blue line
    draw.line([lot_tl, (lot_br[0], lot_tl[1])], fill=(20, 60, 200), width=6)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((lot_tl[0], lot_tl[1] - 30), "ELM STREET", fill=(20, 60, 200), font=font)
    draw.text((ease_tl[0] - 10, (ease_tl[1] + ease_br[1]) // 2), "UTIL\nESMT", fill=(120, 80, 0), font=font)

    # Scale bar: 50 ft
    bar_ft = 50
    bar_p0 = (ORIGIN_PX[0], height_px - 40)
    bar_p1 = (ORIGIN_PX[0] + bar_ft * PX_PER_FT, height_px - 40)
    draw.line([bar_p0, bar_p1], fill=(0, 0, 0), width=3)
    draw.text((bar_p0[0], height_px - 30), f"{bar_ft} FT", fill=(0, 0, 0), font=font)

    out_png = PROJECT_DIR / "site_sketch.png"
    img.save(out_png)

    georef = {
        "_note": "Sidecar georeference for site_sketch.png, analogous to a real-world .pgw/.wld file shipped with a scanned survey.",
        "px_per_ft": PX_PER_FT,
        "origin_px": list(ORIGIN_PX),
        "x_axis_ft_direction": "+x_ft = east = +x_px",
        "y_axis_ft_direction": "+y_ft = south/depth = +y_px",
        "easement_color_rgb": list(EASEMENT_COLOR),
        "easement_color_tolerance": 25,
        "lot_boundary_color_rgb": [0, 0, 0],
        "lot_boundary_color_tolerance": 60,
    }
    (PROJECT_DIR / "site_sketch_georef.json").write_text(json.dumps(georef, indent=2))
    print(f"Wrote {out_png} and site_sketch_georef.json")


if __name__ == "__main__":
    main()
