"""CLI: python generate.py [scenario index]  -> out/<scenario>.ifc + .html (open the HTML in Chrome)."""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from design import SCENARIOS, generate
from ifc_writer import read_meshes, write_ifc
from viewer3d import build_figure

OUT = Path(__file__).parent / "out"


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
