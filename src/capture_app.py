"""Drive the running Streamlit app in Chrome and screenshot every mode and tab, the way a
user would: run the pipeline, generate IFC scenarios, open the real IFC file, use the hotel
workspace, generate the hotel PDF and ask the assistant.

Usage: streamlit run src/app.py   (in another terminal), then
       python src/capture_app.py [url]
Output: reports/walkthrough/screens/*.png and screens/capture_log.json (captions, timings,
        on-page errors, assistant replies).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "walkthrough" / "screens"
URL = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8501")
VIEW = {"width": 1680, "height": 1050}

log: dict = {"url": URL, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "shots": [], "timings": {}, "errors": [],
             "assistant": {}}


def idle(pg, timeout: float = 900, settle: float = 2.0) -> float:
    """Wait until Streamlit's 'Running...' widget has been gone for `settle` seconds."""
    t0, quiet = time.time(), None
    while time.time() - t0 < timeout:
        if pg.locator('[data-testid="stStatusWidget"]').count() > 0:
            quiet = None
        else:
            quiet = quiet or time.time()
            if time.time() - quiet >= settle:
                return round(time.time() - t0, 1)
        pg.wait_for_timeout(300)
    raise TimeoutError(f"app still running after {timeout}s")


def errors_on_page(pg, where: str) -> None:
    for e in pg.locator('[data-testid="stException"]').all_inner_texts():
        log["errors"].append({"where": where, "text": e[:800]})


def shot(pg, name: str, caption: str, mode: str, top=None, wait_ms: int = 800) -> None:
    if top is not None:
        top.evaluate("e => e.scrollIntoView({block: 'start'})")
        pg.mouse.wheel(0, -70)
    pg.wait_for_timeout(wait_ms)
    path = OUT / f"{name}.png"
    pg.screenshot(path=str(path))
    errors_on_page(pg, name)
    log["shots"].append({"file": path.name, "caption": caption, "mode": mode})
    print(f"  shot {name}")


def shot3d(pg, name: str, caption: str, mode: str) -> None:
    """3D charts: widen the main column, reset the camera (double-click), then capture."""
    wide = pg.locator('[data-testid="stSidebar"]').get_by_text("Wide 3D view", exact=False).first
    wide.click()
    idle(pg, settle=2)
    plot = pg.locator(".js-plotly-plot").first
    plot.evaluate("e => e.scrollIntoView({block: 'start'})")
    pg.mouse.wheel(0, -40)
    pg.wait_for_timeout(2500)
    box = plot.bounding_box()
    if box:
        pg.mouse.dblclick(box["x"] + box["width"] * 0.45, box["y"] + box["height"] * 0.6)
    pg.wait_for_timeout(2000)
    pg.mouse.move(VIEW["width"] - 5, VIEW["height"] - 5)  # no hover tooltip in the shot
    pg.wait_for_timeout(1200)
    shot(pg, name, caption, mode, wait_ms=500)
    wide.click()
    idle(pg, settle=2)


def tab(pg, name: str):
    t = pg.get_by_role("tab", name=name, exact=True).first
    t.click()
    idle(pg, settle=1.5)
    return t


def sidebar_click(pg, text: str) -> None:
    pg.locator('[data-testid="stSidebar"]').get_by_text(text, exact=True).first.click()


def chat_texts(pg) -> list[str]:
    return pg.locator('[data-testid="stChatMessage"]').all_inner_texts()


PIPELINE_STATE = ROOT / "outputs" / "project_001" / "project_state.json"


def capture_pipeline(pg) -> None:
    """Run the pipeline in the UI, screenshot it, and keep a copy of exactly this run's result
    (the test suite also runs the pipeline and would overwrite outputs/project_001/)."""
    m = "Multi-Agent Pipeline"
    print(m)
    shot(pg, "01_pipeline_inputs", "Multi-Agent Pipeline before a run: the sample customer request and site sketch "
         "that feed the agents; the assistant panel on the right.", m)
    n0 = len(chat_texts(pg))
    sidebar_click(pg, "Run full pipeline")
    pg.wait_for_timeout(25000)
    shot(pg, "02_pipeline_running", "The pipeline running live: each agent reports its step as it finishes.", m)
    log["timings"]["pipeline_run_s"] = round(25 + idle(pg, timeout=1500, settle=3), 1)
    shutil.copy(PIPELINE_STATE, OUT / "pipeline_state.json")
    texts = chat_texts(pg)
    log["assistant"]["after_pipeline"] = texts[-1] if len(texts) > n0 else ""
    for i, (t, cap) in enumerate([
        ("Overview", "Pipeline result overview: the arbiter's decision, confidence and conflict count."),
        ("Agent trace", "Agent trace: every agent step with model, status and latency."),
        ("Program & site", "Program parsed from the request and the computed site constraints."),
        ("BaaP & conflicts", "BaaP module counts and the conflicts the Conflict Agent found."),
        ("Site simulation", "Physics-style site simulation and the computer-vision QA of the rendered plan."),
        ("Floor plates", "Rule-based floor plates (ground and typical floor)."),
        ("Final decision", "The arbiter's final decision, unresolved issues and next steps."),
    ], start=3):
        top = tab(pg, t)
        shot(pg, f"{i:02d}_pipeline_{t.lower().replace(' & ', '_').replace(' ', '_')}", cap, m, top, wait_ms=1500)
    for name in ("site_plan_iter1.png", "floor_plate_ground.png", "floor_plate_typical.png"):
        src = PIPELINE_STATE.parent / name
        if src.exists():
            shutil.copy(src, OUT / f"pipeline_{name}")


def retake_pipeline() -> None:
    """Redo only the pipeline part and merge it into the existing log."""
    old = json.loads((OUT / "capture_log.json").read_text(encoding="utf-8"))
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport=VIEW)
        pg.goto(URL, wait_until="networkidle")
        idle(pg, timeout=300)
        capture_pipeline(pg)
        b.close()
    keep = [s for s in old["shots"] if not any(s["file"].startswith(f"{i:02d}_") for i in range(1, 10))]
    old["shots"] = sorted(keep + log["shots"], key=lambda s: s["file"])
    old["timings"]["pipeline_run_s"] = log["timings"]["pipeline_run_s"]
    old["assistant"]["after_pipeline"] = log["assistant"]["after_pipeline"]
    old["errors"] = [e for e in old.get("errors", []) if not e["where"][:2].isdigit() or int(e["where"][:2]) > 9] + log["errors"]
    old["pipeline_rerun"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "capture_log.json").write_text(json.dumps(old, indent=2), encoding="utf-8")
    print(f"retook {len(log['shots'])} pipeline screenshots")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport=VIEW)
        pg.goto(URL, wait_until="networkidle")
        log["timings"]["startup_s"] = idle(pg, timeout=300)

        capture_pipeline(pg)

        # -------------------------------------------------- 2 IFC studio -
        m = "IFC Design Studio"
        print(m)
        sidebar_click(pg, m)
        idle(pg, settle=2.5)
        sidebar_click(pg, "Baseline: 4 floors, surface parking")
        log["timings"]["ifc_baseline_s"] = idle(pg, timeout=600, settle=2.5)
        for i, (t, cap) in enumerate([
            ("3D model", "IFC Design Studio, Baseline scenario (120 units, 4 floors, surface parking): the generated "
                         "IFC4 model in the 3D viewer."),
            ("Site plan", "Baseline site plan: building, parking, easement and fire lane on the synthetic lot."),
            ("Compliance (sample zoning)", "Baseline compliance against the ten sample-zoning checks."),
            ("Real IBC check", "Baseline cross-checked against RNGD's real 2021 IBC rules."),
            ("How this was built", "Step-by-step explanation of how the layout was generated."),
        ], start=10):
            top = tab(pg, t)
            name = f"{i:02d}_ifc_baseline_{t.split(' (')[0].lower().replace(' ', '_')}"
            shot3d(pg, name, cap, m) if t == "3D model" else shot(pg, name, cap, m, top, wait_ms=1200)
        sidebar_click(pg, "Right-sized: 60 units, podium (compliant)")
        log["timings"]["ifc_rightsized_s"] = idle(pg, timeout=600, settle=2.5)
        top = tab(pg, "Compliance (sample zoning)")
        shot(pg, "15_ifc_rightsized_compliance", "Right-sized scenario (60 units, podium parking): the compliance "
             "checks after the fix.", m, top, wait_ms=1200)
        tab(pg, "3D model")
        shot3d(pg, "16_ifc_rightsized_3d", "Right-sized scenario in 3D.", m)

        # ------------------------------------------------- 3 open IFC file -
        m = "Open IFC File"
        print(m)
        sidebar_click(pg, m)
        idle(pg, settle=2)
        sidebar_click(pg, "Load / open file")
        log["timings"]["ifc_open_s"] = idle(pg, timeout=1200, settle=3)
        for i, (t, cap) in enumerate([
            ("3D model", "A real IFC file (CDI Solutions, Metairie Bank, IFC2X3, 89 MB) recentred and shown in 3D."),
            ("Summary", "What is in the real model: schema, storeys and element counts by IFC class."),
            ("Element browser", "Searchable list of the model's elements."),
        ], start=17):
            top = tab(pg, t)
            name = f"{i:02d}_explorer_{t.lower().replace(' ', '_')}"
            shot3d(pg, name, cap, m) if t == "3D model" else shot(pg, name, cap, m, top, wait_ms=1200)

        # ---------------------------------------------------- 4 hotel ----
        m = "Hotel Schematic"
        print(m)
        sidebar_click(pg, m)
        idle(pg, settle=2.5)
        for i, (t, cap) in enumerate([
            ("Overview & 3D", "Hotel Schematic workspace: KPI tiles, the room-by-room 3D model and the key results "
                              "with suggested fixes."),
            ("Program", "The editable program (Project Renegade L+L space allocation, 28 rows) with area and "
                        "room-mix charts."),
            ("Floor plans", "Interactive floor plan of Level 1 (hover shows each room's area and size)."),
        ], start=20):
            top = tab(pg, t)
            shot(pg, f"{i:02d}_hotel_{t.split(' ')[0].lower()}", cap, m, top, wait_ms=2500)
        # scroll to the detailed program table
        tab(pg, "Program")
        pg.get_by_text("Detailed program", exact=False).first.evaluate("e => e.scrollIntoView({block: 'start'})")
        shot(pg, "23_hotel_program_detail", "Detailed program: per-room size, beds, sleeping capacity, placement, IBC "
             "occupant load and fit-out.", m, wait_ms=1500)
        top = tab(pg, "Floor plans")
        pg.get_by_text("Level 2", exact=True).first.click()
        idle(pg, settle=1.5)
        shot(pg, "24_hotel_plan_level2", "Interactive plan of Level 2, a typical guest floor (24 keys).", m, top, wait_ms=2500)
        pg.get_by_text("Drawing", exact=True).first.click()
        idle(pg, settle=2)
        shot(pg, "25_hotel_plan_drawing", "The same level in drawing style (as used on the PDF sheets).", m, top, wait_ms=2000)
        for i, (t, cap) in enumerate([
            ("Room mix", "Keys of each room type on each guest floor."),
            ("Unit types", "Unit type detail: enlarged plan and two 3D interior views."),
            ("Code & ADA", "Code screen against RNGD's IBC rule set and the 2010 ADA guest-room tables, with rule text."),
        ], start=26):
            top = tab(pg, t)
            shot(pg, f"{i:02d}_hotel_{t.split(' ')[0].lower()}", cap, m, top, wait_ms=2000)
        top = tab(pg, "PDF report")
        pg.get_by_role("button", name="Generate PDF report").first.click()
        log["timings"]["hotel_pdf_s"] = idle(pg, timeout=900, settle=3)
        shot(pg, "29_hotel_report_results", "PDF report tab after generating: summary numbers and the results that "
             "went into the report.", m, top, wait_ms=2500)
        pg.get_by_text("All sheets", exact=True).first.evaluate("e => e.scrollIntoView({block: 'center'})")
        shot(pg, "30_hotel_report_sheets", "Sheet gallery of the generated drawing set.", m, wait_ms=2500)

        # ------------------------------------------------ 5 assistant ----
        n0 = len(chat_texts(pg))
        q = "Why does the story check fail, and how do we fix it?"
        pg.get_by_role("button", name=q).first.click()
        log["timings"]["assistant_reply_s"] = idle(pg, timeout=300, settle=2)
        texts = chat_texts(pg)
        log["assistant"]["hotel_question"] = q
        log["assistant"]["hotel_answer"] = texts[-1] if len(texts) > n0 else ""
        box = pg.locator('[data-testid="stChatMessage"]').last
        box.evaluate("e => e.scrollIntoView({block: 'end'})")
        shot(pg, "31_assistant_answer", "The assistant answering a one-click question about the hotel design, "
             "using the live design state.", "Assistant", wait_ms=1500)
        b.close()
    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "capture_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"{len(log['shots'])} screenshots, {len(log['errors'])} on-page errors")


def retake_3d() -> None:
    """Retake only the 3D screenshots (no pipeline run) and merge them into the existing log."""
    global log
    old = json.loads((OUT / "capture_log.json").read_text(encoding="utf-8"))
    keep = [s for s in old["shots"] if s["file"] not in ("10_ifc_baseline_3d_model.png", "16_ifc_rightsized_3d.png",
                                                         "17_explorer_3d_model.png")]
    caps = {s["file"]: s["caption"] for s in old["shots"]}
    log["shots"] = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport=VIEW)
        pg.goto(URL, wait_until="networkidle")
        idle(pg, timeout=300)
        sidebar_click(pg, "IFC Design Studio")
        idle(pg, settle=2.5)
        sidebar_click(pg, "Baseline: 4 floors, surface parking")
        idle(pg, timeout=600, settle=2.5)
        tab(pg, "3D model")
        shot3d(pg, "10_ifc_baseline_3d_model", caps.get("10_ifc_baseline_3d_model.png", ""), "IFC Design Studio")
        sidebar_click(pg, "Right-sized: 60 units, podium (compliant)")
        idle(pg, timeout=600, settle=2.5)
        tab(pg, "3D model")
        shot3d(pg, "16_ifc_rightsized_3d", caps.get("16_ifc_rightsized_3d.png", ""), "IFC Design Studio")
        sidebar_click(pg, "Open IFC File")
        idle(pg, settle=2)
        sidebar_click(pg, "Load / open file")
        idle(pg, timeout=1200, settle=3)
        tab(pg, "3D model")
        shot3d(pg, "17_explorer_3d_model", caps.get("17_explorer_3d_model.png", ""), "Open IFC File")
        b.close()
    old["shots"] = sorted(keep + log["shots"], key=lambda s: s["file"])
    old["errors"] = old.get("errors", []) + log["errors"]
    (OUT / "capture_log.json").write_text(json.dumps(old, indent=2), encoding="utf-8")
    print(f"retook {len(log['shots'])} 3D screenshots")


if __name__ == "__main__":
    if "--only-3d" in sys.argv:
        retake_3d()
    elif "--only-pipeline" in sys.argv:
        retake_pipeline()
    else:
        run()
