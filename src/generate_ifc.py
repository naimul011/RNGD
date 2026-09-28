"""CLI: python generate_ifc.py [scenario index] -> outputs/ifc_design/<scenario>.ifc + .html
(double-click the .html in Chrome)."""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rngd import config
from rngd.ifc.design import SCENARIOS, generate
from rngd.ifc.ifc_writer import read_meshes, write_ifc
from rngd.ifc.viewer3d import build_figure

OUT = config.OUTPUTS_DIR / "ifc_design"


def run(name, params):
    lay = generate(params)
    slug = re.sub(r"\W+", "_", name).strip("_").lower()
    ifc = write_ifc(lay, OUT / f"{slug}.ifc")
    meshes = read_meshes(ifc)
    html = OUT / f"{slug}.html"
    build_figure(meshes).write_html(str(html), include_plotlyjs="cdn")
    return lay, ifc, html


if __name__ == "__main__":
    items = list(SCENARIOS.items())
    sel = [items[int(sys.argv[1])]] if len(sys.argv) > 1 else items
    for name, p in sel:
        lay, ifc, html = run(name, p)
        print(f"\n== {name}\n   {ifc.name} ({ifc.stat().st_size // 1024} KB)  {html.name}")
        print("  ", lay.metrics)
        for c in lay.checks:
            print(f"   [{c['status']}] {c['check']}: required {c['required']}, provided {c['provided']}")
