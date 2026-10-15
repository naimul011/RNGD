"""Screenshots of Hotel Schematic -> "Optimize layouts" for the walkthrough report.

Usage: python src/capture_optimizer.py [port]     # with the app running (default port 8501)
Output: reports/walkthrough/screens/32..36_*.png + optimizer_shots.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

import capture_app as C

OUT = Path(__file__).resolve().parents[1] / "reports" / "walkthrough" / "screens"


def main(port: str = "8501") -> None:
    shots = []

    def shot(pg, name, caption, anchor=None, wheel=0):
        if anchor is not None:
            anchor.evaluate("e => e.scrollIntoView({block: 'start'})")
            pg.mouse.wheel(0, wheel - 70)
        pg.wait_for_timeout(1800)
        pg.screenshot(path=str(OUT / f"{name}.png"))
        shots.append({"file": f"{name}.png", "caption": caption, "mode": "Hotel Schematic"})
        print("  shot", name)

    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1050})
        pg.goto(f"http://localhost:{port}")
        C.idle(pg, 300)
        C.sidebar_click(pg, "Hotel Schematic")
        C.idle(pg, 300)
        C.tab(pg, "Optimize layouts")
        shot(pg, "32_hotel_optimize_settings",
             "Optimize layouts: buildable area, shape mode (generative, blend or one shape), core placement and floors.",
             pg.get_by_role("tab", name="Optimize layouts", exact=True).first)
        t0 = time.time()
        pg.get_by_role("button", name="Optimize layouts").first.click()
        C.idle(pg, 1200, settle=3)
        secs = round(time.time() - t0, 1)
        shot(pg, "33_hotel_optimize_decide",
             "Decision step: every distinct layout ranked by TOPSIS under the chosen priority, with the Pareto flag; the "
             "Show dropdown switches between the top 5, 10 or 25, the Pareto front and all layouts.",
             pg.get_by_text("Decide", exact=True).first)
        shot(pg, "34_hotel_optimize_compare",
             "Comparison: typical-floor thumbnails of the first-ranked layouts, keys against gross area per key for every "
             "layout, and the fitness gained by pattern search.", pg.get_by_text("Compare", exact=True).first)
        insp = pg.get_by_text("Inspect a layout", exact=True).first
        shot(pg, "35_hotel_optimize_inspect", "One layout inspected: keys, upper bound, footprint, corridor share, dead end, "
             "code checks and fitness.", insp)
        shot(pg, "36_hotel_optimize_plan", "Interactive furnished plan of the selected layout (hover gives room data).",
             insp, wheel=470)
        C.tab(pg, "3D massing")
        pg.wait_for_timeout(2500)
        shot(pg, "37_hotel_optimize_3d", "Stacked 3D massing of the selected layout, coloured by room type.",
             pg.get_by_role("tab", name="3D massing", exact=True).first)
        errs = pg.locator('[data-testid="stException"]').all_inner_texts()
        b.close()
    (OUT / "optimizer_shots.json").write_text(json.dumps(shots, indent=1), encoding="utf-8")
    print(f"optimization run in the app: {secs} s; on-page errors: {len(errs)}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
