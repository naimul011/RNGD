"""Build the final paper (Overleaf project + PDF).

reports/paper/ holds an Overleaf-ready project:
  main.tex           the paper (hand-written; edit this)
  references.bib     bibliography
  figures/*.jpg      screenshots, drawing sheets, experiment figures (copied + compressed here)
  gallery.tex        appendix of all screenshots (generated from the capture log)
  optimizer.tex      layout-optimization section (generated from reports/walkthrough/optimizer/study.json,
                     i.e. from src/build_optimizer_study.py; shared with the walkthrough report)
  optimizer_numbers.tex  the headline numbers of that study as LaTeX macros (used in the abstract and text)
  code/*.py          code listings, extracted VERBATIM from src/ by this script (ASCII only, so
                     Overleaf's default pdfLaTeX compiles them)
This script refreshes figures/ and code/, compiles main.pdf with Tectonic and writes
RNGD_paper_overleaf.zip (upload it to Overleaf: New Project > Upload Project).

Usage: python src/build_paper.py [--no-figures]
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports" / "paper"
FIG = PAPER / "figures"
CODE = PAPER / "code"
TECTONIC = ROOT / "Testt" / "tools" / "tectonic.exe"
S = ROOT / "src" / "rngd"

# (output file, source file, [qualified function / class / assignment names] or ("lines", start marker, end marker))
SNIPPETS = [
    ("llm_guard.py", S / "llm.py", ["assert_allowed", "call_json"]),
    ("intake_prompt.py", S / "agents" / "intake_agent.py", ["SYSTEM", "run"]),
    ("envelope.py", S / "agents" / "zoning_agent.py", ["_compute_buildable_envelope"]),
    ("modules.py", S / "agents" / "baap_matching_agent.py", ["_module_counts"]),
    ("feasibility.py", S / "agents" / "conflict_agent.py", ["_compute_feasibility"]),
    ("retrieval.py", S / "vectorstore.py", ["_EmbeddingBackend.retrieve", "VectorStore.retrieve_multi_as_context"]),
    ("physics.py", S / "simulation" / "physics_engine.py", ["_separation_force", "SiteSimulation.step", "SiteSimulation.run"]),
    ("calibration.py", S / "vision" / "layout_qa.py", ["PixelToFeet"]),
    ("loop.py", S / "orchestrator.py", ("lines", "        # A clean geometry pass", "driveway_w = max(")),
    ("critic.py", S / "agents" / "design_critic_agent.py", ["run"]),
    ("recenter.py", S / "ifc" / "explorer.py", ["_recenter"]),
    ("split_rows.py", S / "hotel" / "layout.py", ["split_rows"]),
    ("hotel_wing.py", S / "hotel" / "layout.py", ("lines", "    # First pass: guest wings", "    length = CORE_W_FT")),
    ("hotel_egress.py", S / "hotel" / "layout.py", ("lines", "    # Egress geometry", "    travel = ")),
    ("code_check.py", S / "hotel" / "code_check.py", ["evaluate"]),
    ("ada.py", S / "hotel" / "details.py", ["ada_required"]),
    ("transform.py", S / "hotel" / "drawings.py", ["room_xf", "Iso.__init__", "Iso.p"]),
    ("opt_fitness.py", S / "optimize" / "metrics.py", ["fitness_from_terms"]),
    ("opt_enumerate.py", S / "optimize" / "search.py", ("lines", "        top = list(range(min(m, 7)))", "            genomes.append(g)")),
    ("opt_plate.py", S / "optimize" / "typology.py", ["elevators_for", "_run_ft"]),
    ("opt_decide.py", S / "optimize" / "decide.py", ["pareto_mask", "topsis"]),
]

ASCII = {"—": "--", "–": "-", "≤": "<=", "≥": ">=", "→": "->", "×": "x", "‘": "'",
         "’": "'", "“": '"', "”": '"', "…": "...", "·": "*", "≈": "~", "°": " deg"}


def to_ascii(text: str) -> str:
    for a, b in ASCII.items():
        text = text.replace(a, b)
    return text.encode("ascii", "replace").decode("ascii")


def _find(tree: ast.Module, qualname: str):
    parts = qualname.split(".")
    nodes = tree.body
    found = None
    for part in parts:
        found = None
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part:
                found = n
            elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == part for t in n.targets):
                found = n
            if found:
                break
        if found is None:
            raise KeyError(f"{qualname} not found")
        nodes = getattr(found, "body", [])
    return found


def extract(src: Path, spec) -> str:
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines()
    if isinstance(spec, tuple):
        _, start, end = spec
        i = next(k for k, ln in enumerate(lines) if start in ln)
        j = next(k for k in range(i + 1, len(lines)) if end in lines[k])
        return textwrap.dedent("\n".join(lines[i:j + 1]))
    tree = ast.parse(text)
    parts = []
    for name in spec:
        node = _find(tree, name)
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        parts.append(textwrap.dedent("\n".join(lines[first - 1:node.end_lineno])))
    return "\n\n".join(parts)


def refresh_code() -> None:
    CODE.mkdir(parents=True, exist_ok=True)
    for out, src, spec in SNIPPETS:
        body = to_ascii(extract(src, spec))
        header = f"# {src.relative_to(ROOT).as_posix()}\n"
        (CODE / out).write_text(header + body.rstrip() + "\n", encoding="ascii")
    print(f"{len(SNIPPETS)} code listings extracted")


def refresh_figures() -> None:
    from PIL import Image

    FIG.mkdir(parents=True, exist_ok=True)
    src = {}
    for f in sorted((ROOT / "reports" / "walkthrough" / "screens").glob("*.png")):
        if f.name[:2].isdigit():
            src[f"snap_{f.stem}"] = f
    for n in ("G001", "G002", "A001", "A101", "A102", "A201", "A401", "A402", "A501"):
        src[f"sheet_{n}"] = ROOT / "reports" / "final" / "sheets" / f"{n}.png"
    for n in ("fig_hotel_inputs_level2", "fig_hotel_sweep", "fig_physics_convergence", "fig_unit_local_coords"):
        src[n] = ROOT / "reports" / "how_it_works" / f"{n}.png"
    for n in ("site_plan_iter1", "floor_plate_typical", "floor_plate_ground"):
        src[f"pipe_{n}"] = ROOT / "outputs" / "project_001" / f"{n}.png"
    src["input_site_sketch"] = ROOT / "data" / "sample_projects" / "project_001" / "site_sketch.png"
    for name, p in src.items():
        if not p.exists():
            print(f"  missing source figure: {p}")
            continue
        im = Image.open(p).convert("RGB")
        if im.width > 2400:
            im = im.resize((2400, round(im.height * 2400 / im.width)), Image.LANCZOS)
        im.save(FIG / f"{name}.jpg", "JPEG", quality=86, optimize=True)
    print(f"{len(src)} figures refreshed")


def write_optimizer() -> None:
    """Layout-optimization section and its figures, from the optimizer study (same text as the walkthrough report)."""
    import json
    import re

    from PIL import Image

    import walkthrough_optimizer as wo

    study = wo.load_study()
    if not study:
        raise SystemExit("run src/build_optimizer_study.py first (reports/walkthrough/optimizer/study.json is missing)")
    FIG.mkdir(parents=True, exist_ok=True)
    src = {f"opt_{f.stem}": f for f in sorted((wo.OUT / "optimizer").glob("*.png"))}
    if wo.SHOTS.exists():
        for sh in json.loads(wo.SHOTS.read_text(encoding="utf-8")):
            f = wo.OUT / "screens" / sh["file"]
            if f.exists():
                src[f"snap_{f.stem}"] = f
    for name, f in src.items():
        im = Image.open(f).convert("RGB")
        if im.width > 2400:
            im = im.resize((2400, round(im.height * 2400 / im.width)), Image.LANCZOS)
        im.save(FIG / f"{name}.jpg", "JPEG", quality=86, optimize=True)
    tex = wo.section(study)
    tex = re.sub(r"\{optimizer/([^{}]+)\.png\}", r"{figures/opt_\1.jpg}", tex)
    tex = re.sub(r"\{screens/([^{}]+)\.png\}", r"{figures/snap_\1.jpg}", tex)
    # the paper's text block is narrower than the walkthrough report's: set the two-column fitness display smaller
    a, b = "\\begin{align*}\nf_{\\text{place}}", "P_{\\text{rooms}}}\\Big).\n\\end{align*}"
    assert a in tex and b in tex
    tex = tex.replace(a, "\\begingroup\\small" + a).replace(b, b + "\\endgroup")
    # ... and its tables get a smaller font and 5% narrower columns
    tex = tex.replace("{\\small\\begin{longtable}", "{\\footnotesize\\begin{longtable}")
    tex = re.sub(r"p\{(\d+\.\d+)cm\}", lambda m: "p{%.2fcm}" % (float(m.group(1)) * 0.95), tex)
    # a longtable steps the table counter even without a caption (captions come from \\captionof): undo it
    tex = tex.replace("\\end{longtable}}", "\\end{longtable}}\\addtocounter{table}{-1}")
    (PAPER / "optimizer.tex").write_text(
        "% generated by src/build_paper.py from reports/walkthrough/optimizer/study.json\n" + to_ascii(tex),
        encoding="ascii")

    sites, hotel = study["sites"], study["hotel"]
    tot = lambda t, k: sum(x["variants"][t][k] for x in sites)
    mean = lambda t: sum(x["variants"][t]["fitness"] for x in sites) / len(sites)
    b, bm = hotel["best"], hotel["best"]["metrics"]
    num = {"RoomsTotal": tot("A", "rooms_total"), "Sites": len(sites),
           "HotelLayouts": hotel["layouts"], "HotelVariants": hotel["variants"], "HotelSecs": f"{hotel['seconds']:.0f}",
           "HotelPareto": hotel["pareto"], "HotelBest": wo._esc(b["plate"]), "HotelFloors": b["floors"],
           "HotelKeys": bm["keys_placed"], "HotelKeysAsked": bm["keys_requested"], "HotelRooms": bm["rooms_placed"],
           "HotelRoomsTotal": bm["rooms_total"],
           "HotelChecks": f"{sum(c['status'] == 'PASS' for c in bm['checks'])}/{len(bm['checks'])}",
           "Budget": f"{study['population']} $\\times$ {study['generations']}",
           "GensA": sites[0]["variants"]["A"].get("generations", 500)}
    for t in "ABCD":
        num[f"Rooms{t}"] = tot(t, "rooms_placed")
        num[f"NoWin{t}"] = tot(t, "windowless_guest_rooms")
        num[f"Fit{t}"] = f"{mean(t):.3f}"
        if t != "A":
            num[f"Secs{t}"] = f"{sum(x['variants'][t]['seconds'] for x in sites):.0f}"
    lines = ["% generated by src/build_paper.py from reports/walkthrough/optimizer/study.json"]
    lines += [f"\\newcommand{{\\opt{k}}}{{{v}}}" for k, v in num.items()]
    (PAPER / "optimizer_numbers.tex").write_text(to_ascii("\n".join(lines)) + "\n", encoding="ascii")
    print(f"optimizer section: {len(src)} figures, {len(num)} numbers")


def _tex_escape(t: str) -> str:
    t = to_ascii(t)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        t = t.replace(a, b)
    return t


def write_gallery() -> None:
    """Appendix with every screenshot of the browser run and the caption recorded during capture."""
    import json

    log_f = ROOT / "reports" / "walkthrough" / "screens" / "capture_log.json"
    shots = json.loads(log_f.read_text(encoding="utf-8"))["shots"] if log_f.exists() else []
    out = ["% generated by src/build_paper.py from reports/walkthrough/screens/capture_log.json"]
    mode = None
    for sh in sorted(shots, key=lambda x: x["file"]):
        stem = Path(sh["file"]).stem
        if not (FIG / f"snap_{stem}.jpg").exists():
            continue
        if sh["mode"] != mode:
            mode = sh["mode"]
            out.append(r"\subsection*{" + _tex_escape(mode) + "}")
        out.append(r"\begin{figure}[H]\centering\includegraphics[width=0.92\linewidth,height=0.4\textheight,"
                   r"keepaspectratio]{figures/snap_" + stem + "}"
                   + r"\caption*{\textbf{Screen " + stem[:2] + ".} " + _tex_escape(sh["caption"]) + r"}\end{figure}")
    (PAPER / "gallery.tex").write_text("\n".join(out) + "\n", encoding="ascii")
    print(f"gallery: {len(out) - 1} entries")


def compile_pdf() -> Path:
    r = subprocess.run([str(TECTONIC), "main.tex"], cwd=PAPER, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise SystemExit("LaTeX failed")
    warn = [ln for ln in r.stderr.splitlines() if "undefined" in ln.lower() or "multiply defined" in ln.lower()]
    for w in warn[:10]:
        print("  warning:", w.strip())
    return PAPER / "main.pdf"


def make_zip() -> Path:
    z = PAPER / "RNGD_paper_overleaf.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in [PAPER / "main.tex", PAPER / "references.bib", PAPER / "gallery.tex", PAPER / "optimizer.tex",
                  PAPER / "optimizer_numbers.tex"] + sorted(FIG.glob("*.jpg")) + sorted(CODE.glob("*.py")):
            zf.write(f, f.relative_to(PAPER).as_posix())
    return z


if __name__ == "__main__":
    if "--no-figures" not in sys.argv:
        refresh_figures()
    refresh_code()
    write_gallery()
    write_optimizer()
    pdf = compile_pdf()
    final = PAPER / "RNGD_final_paper.pdf"
    final.write_bytes(pdf.read_bytes())
    print(final)
    print(make_zip())
