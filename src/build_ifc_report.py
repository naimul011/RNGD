"""Builds report/ifc_design_report.tex + .pdf from real generated designs.  python build_report.py"""
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rngd import config
from rngd.ifc.design import SCENARIOS, ZONING, generate, module_counts
from rngd.ifc.ifc_writer import read_meshes, write_ifc
from rngd.ifc.viewer3d import COLORS, _key

ROOT = HERE.parent
OUT = ROOT / "reports" / "ifc_design"
OUT.mkdir(parents=True, exist_ok=True)
TECTONIC = ROOT / "Testt" / "tools" / "tectonic.exe"


def esc(s):
    s = str(s)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("<=", r"$\le$"), (">=", r"$\ge$")]:
        s = s.replace(a, b)
    return s


def plan(lay, path):
    p = lay.p
    fig, ax = plt.subplots(figsize=(3.2, 4.2))
    ax.add_patch(Rectangle((0, 0), p.lot_w, p.lot_d, fill=False, lw=1.5))
    ax.add_patch(Rectangle((p.lot_w - p.easement_east, 0), p.easement_east, p.lot_d, color="#FFBF00", alpha=.8))
    e = lay.env
    ax.add_patch(Rectangle((e["x0"], e["y0"]), e["w"], e["d"], fill=False, ls="--", ec="#CC0000", lw=1))
    if lay.drive:
        d = lay.drive
        ax.add_patch(Rectangle((d["x"], d["y"]), d["w"], d["d"], color="#555555"))
    for s in lay.stalls:
        ax.add_patch(Rectangle((s["x"], s["y"]), s["w"], s["d"], fc="#8C8C8C", ec="white", lw=.3))
    for w in lay.wings:
        ax.add_patch(Rectangle((w["x"], w["y"]), w["w"], w["h"], fc="#3E7CB1", alpha=.85, ec="black", lw=.6))
    ax.set_xlim(-5, p.lot_w + 5)
    ax.set_ylim(p.lot_d + 5, -5)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render3d(meshes, path):
    """Walls only, coloured by unit type (slabs/roof would hide the interior layout)."""
    import numpy as np
    fig = plt.figure(figsize=(5.4, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    tris, cols = [], []
    for m in meshes:
        key = _key(m["name"])
        if m["cls"] != "IfcWall":
            continue
        for t in m["verts"][m["tris"]]:
            tris.append(t)
            cols.append(COLORS.get(key, "#AAAAAA"))
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors="#00000033", linewidths=0.15))
    v = np.vstack([m["verts"] for m in meshes if m["cls"] == "IfcWall"])
    lo, hi = v.min(0), v.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
    ax.view_init(32, -58)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


rows, blocks = [], []
for i, (name, p) in enumerate(SCENARIOS.items()):
    lay = generate(p)
    ifc = write_ifc(lay, Path(tempfile.gettempdir()) / f"rep_{i}.ifc")
    plan(lay, OUT / f"plan_{i}.png")
    if i in (0, 1):
        render3d(read_meshes(ifc), OUT / f"view_{i}.png")
    m = lay.metrics
    fails = [c["check"] for c in lay.checks if c["status"] != "PASS"]
    rows.append(rf"{esc(name)} & {m['units_placed']}/{p.units} & {m['stories']} & {m['far']} & {m['coverage_pct']} & {m['parking_provided']}/{m['parking_required']} & {len(fails)} \\")
    blocks.append((i, name, lay))

zrows = "".join(rf"{esc(k)} & {v['front']}/{v['side']}/{v['rear']} & {v['far']} & {v['cov']} & {v['height']} & {v['stories']} & {v['parking']} \\" + "\n" for k, v in ZONING.items())
chk_rows = "".join(rf"{esc(c['check'])} & {esc(c['required'])} & {esc(c['provided'])} & {c['status']} \\" + "\n" for c in blocks[1][2].checks)
chk_base = "".join(rf"{esc(c['check'])} & {esc(c['required'])} & {esc(c['provided'])} & {c['status']} \\" + "\n" for c in blocks[0][2].checks)
plans = "\n".join(rf"\begin{{subfigure}}{{0.32\linewidth}}\centering\includegraphics[width=\linewidth]{{plan_{i}.png}}\caption{{\scriptsize {esc(n)}}}\end{{subfigure}}" for i, n, _ in blocks[:3])

base = blocks[0][2]
env = base.env
counts = module_counts(base.p)
subs = {
    "@@ZROWS@@": zrows,
    "@@SCENROWS@@": chr(10).join(rows),
    "@@PLANS@@": plans,
    "@@CHK_GOOD@@": chk_rows,
    "@@CHK_BASE@@": chk_base,
    "@@ENV_W@@": f"{env['w']:.0f}", "@@ENV_D@@": f"{env['d']:.0f}", "@@ENV_AREA@@": f"{env['w'] * env['d']:,.0f}",
    "@@LOT_AREA@@": f"{base.p.lot_w * base.p.lot_d:,.0f}",
    "@@COUNTS@@": f"{counts['STUDIO_A']} studios, {counts['ONEBR_A']} one-bedrooms and {counts['TWOBR_A']} two-bedrooms",
    "@@WINGS@@": str(base.metrics["wings"]), "@@FOOT@@": f"{base.metrics['footprint_sf']:,}",
    "@@COV@@": str(base.metrics["coverage_pct"]), "@@FAR@@": str(base.metrics["far"]),
    "@@STALLS@@": str(base.metrics["parking_provided"]),
}
tex = (ROOT / "reports" / "ifc_design" / "report_template.tex").read_text(encoding="utf-8")
for k, v in subs.items():
    tex = tex.replace(k, v)
(OUT / "ifc_design_report.tex").write_text(tex, encoding="utf-8")
r = subprocess.run([str(TECTONIC), "ifc_design_report.tex"], cwd=OUT, capture_output=True, text=True)
print(r.stderr[-1200:])
sys.exit(r.returncode)
