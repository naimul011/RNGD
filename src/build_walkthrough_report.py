"""Detailed walkthrough report (PDF) of the whole RNGD AI Studio, built from a real run:
screenshots captured by src/capture_app.py plus the actual results of that run
(pipeline state, IFC studio scenarios, the real IFC file, the hotel schematic and its
drawing set, assistant replies) and the test-suite results.

Usage: python src/capture_app.py           # with the app running on :8501
       python src/build_walkthrough_report.py [--no-tests | --reuse-tests]
Output: reports/walkthrough/RNGD_walkthrough_report.pdf
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rngd import config  # noqa: E402
from rngd.schemas import ProjectState  # noqa: E402
import walkthrough_optimizer  # noqa: E402

OUT = ROOT / "reports" / "walkthrough"
SCREENS = OUT / "screens"
TECTONIC = ROOT / "Testt" / "tools" / "tectonic.exe"
FINAL = ROOT / "reports" / "final"


# ------------------------------------------------------------- latex utils -
def esc(s) -> str:
    s = "" if s is None else str(s)
    s = "".join(ch for ch in s if ord(ch) < 0x10000)  # drop emoji / astral chars the font lacks
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return (s.replace("<=", r"$\le$").replace(">=", r"$\ge$").replace("\u2264", r"$\le$").replace("\u2265", r"$\ge$")
            .replace("\u202f", " ").replace("\u2011", "-"))


def _tab_table(rows: list[list[str]]) -> str:
    n = max(len(r) for r in rows)
    spec = f"p{{{15.0 / n:.2f}cm}}" * n

    def fmt(c: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", esc(c))

    head, body = rows[0], rows[1:]
    lines = [" & ".join(r"\textbf{" + fmt(c) + "}" for c in head + [""] * (n - len(head))) + r" \\ \midrule"]
    lines += [" & ".join(fmt(c) for c in r + [""] * (n - len(r))) + r" \\" for r in body]
    return ("{\\footnotesize\\begin{longtable}{" + spec + "}\\toprule\n" + "\n".join(lines)
            + "\n\\bottomrule\\end{longtable}}")


def clean_chat(text: str) -> str:
    """Chat text as scraped from the page: drop the avatar icon name on the first line."""
    return re.sub(r"^\s*(smart_toy|face|person|assistant|user)\s*\n", "", text or "")


def md(text: str) -> str:
    """Small markdown -> LaTeX for LLM replies: **bold**, `code`, bullet and numbered lists, headings,
    and tab-separated rows (how Streamlit tables come out of the page) as a table."""
    out, in_list, tab_rows = [], None, []
    for raw in (text or "").splitlines() + [""]:
        line = raw.rstrip()
        if "\t" in line:
            tab_rows.append([c.strip() for c in line.split("\t")])
            continue
        if tab_rows:
            out.append(_tab_table(tab_rows))
            tab_rows = []
        bullet = re.match(r"^\s*[-*]\s+(.*)", line)
        num = re.match(r"^\s*\d+[.)]\s+(.*)", line)
        kind = "itemize" if bullet else "enumerate" if num else None
        if in_list and kind != in_list:
            out.append(rf"\end{{{in_list}}}")
            in_list = None
        if kind and not in_list:
            out.append(rf"\begin{{{kind}}}\setlength\itemsep{{1pt}}")
            in_list = kind
        body = (bullet or num).group(1) if kind else line
        body = esc(body)
        body = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", body)
        body = re.sub(r"`(.+?)`", r"\\texttt{\1}", body)
        body = re.sub(r"^#+\s*(.*)", r"\\textbf{\1}", body)
        if kind:
            out.append(r"\item " + body)
        elif body.strip():
            out.append(body + r"\par")
    if in_list:
        out.append(rf"\end{{{in_list}}}")
    return "\n".join(out)


def table(headers, rows, spec, small=True) -> str:
    head = " & ".join(r"\textbf{" + esc(h) + "}" for h in headers)
    body = "\n".join(" & ".join(c if isinstance(c, Raw) else esc(c) for c in r) + r" \\" for r in rows)
    t = (rf"\begin{{longtable}}{{{spec}}}" + "\n\\toprule\n" + head + r" \\ \midrule \endhead" + "\n" + body +
         "\n\\bottomrule\n\\end{longtable}")
    return r"{\small " + t + "}" if small else t


class Raw(str):
    """Cell content that is already LaTeX."""


def status_cell(s: str) -> Raw:
    color = {"PASS": "pass", "FAIL": "fail", "WARN": "warn", "REVIEW": "warn", "ASSUMED": "muted",
             "ok": "pass", "degraded": "warn", "error": "fail"}.get(str(s), "muted")
    return Raw(rf"\textcolor{{{color}}}{{\textbf{{{esc(s)}}}}}")


def fig(path: Path, caption: str, width: str = r"\linewidth") -> str:
    if not path.exists():
        return rf"\textit{{(missing figure: {esc(path.name)})}}\par"
    rel = path.relative_to(OUT).as_posix() if OUT in path.parents else path.as_posix()
    return (rf"\begin{{figure}}[H]\centering\includegraphics[width={width},keepaspectratio,height=0.42\textheight]"
            rf"{{{rel}}}\caption{{{esc(caption)}}}\end{{figure}}")


# ------------------------------------------------------------- results ----
def pipeline_results() -> tuple[ProjectState | None, str]:
    """(state, note). Prefer the copy the capture kept of the browser run; otherwise fall back to the
    latest run in outputs/ and say so, because LLM results differ between runs."""
    snap = SCREENS / "pipeline_state.json"
    if snap.exists():
        return ProjectState.model_validate_json(snap.read_text(encoding="utf-8")), ""
    f = ROOT / "outputs" / "project_001" / "project_state.json"
    if not f.exists():
        return None, ""
    when = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    note = (f"Data source: the tables in this section come from the most recent completed pipeline run on this "
            f"machine ({when}, the automated test suite's run through the app). The screenshots come from the "
            "browser run that started {started}. Both runs used the same inputs and code, but LLM outputs differ "
            "between runs, so the wording, conflict details and some figures in the screenshots differ from the tables.")
    return ProjectState.model_validate_json(f.read_text(encoding="utf-8")), note


def ifc_results() -> list[dict]:
    from rngd.ifc import code_check
    from rngd.ifc.design import SCENARIOS, generate

    out = []
    for name, p in SCENARIOS.items():
        lay = generate(p)
        ibc = code_check.evaluate(p, lay)
        out.append({"name": name, "params": p, "metrics": lay.metrics, "checks": lay.checks, "ibc": ibc})
    return out


def explorer_results() -> dict | None:
    from rngd.ifc import explorer

    files = [f for f in explorer.list_available_files() if "ifc_samples" in str(f)]
    if not files:
        return None
    summary, meshes = explorer.load(files[0])
    summary["file"] = files[0].name
    summary["size_mb"] = round(files[0].stat().st_size / 1e6, 1)
    return summary


def hotel_results():
    import build_final_report
    from rngd.hotel import details, layout
    from rngd.hotel.program import default_program

    man = build_final_report.load_manifest()
    lk = (man or {}).get("layout") or {}
    site = (man or {}).get("site") or build_final_report.DEFAULT_SITE
    sch = layout.build(default_program(), site, **lk)
    return sch, details, man


def run_tests(reuse: bool = False) -> dict:
    junit = OUT / "pytest_results.xml"

    class _R:  # stand-in for a CompletedProcess when reusing the last run
        stdout, returncode = "", 0

    r = _R() if reuse and junit.exists() else subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", f"--junitxml={junit}"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    cases = []
    if junit.exists():
        for tc in ET.parse(junit).getroot().iter("testcase"):
            outcome = "FAIL" if tc.find("failure") is not None or tc.find("error") is not None else \
                "SKIP" if tc.find("skipped") is not None else "PASS"
            cases.append({"file": tc.get("classname", "").split(".")[-1], "name": tc.get("name"),
                          "time": float(tc.get("time", 0)), "outcome": outcome})
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-1:] or [
        f"{sum(c['outcome'] == 'PASS' for c in cases)} passed (results of the last full run, {junit.name})"]
    return {"cases": cases, "summary": tail[0], "returncode": r.returncode}


# ------------------------------------------------------------- sections ---
PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[margin=0.8in,headheight=14pt]{geometry}
\usepackage{fontspec}\setmainfont{Arial}\setmonofont{Consolas}
\usepackage{amsmath,amssymb}
\usepackage{graphicx,booktabs,longtable,array,float,xcolor,caption,fancyhdr,tikz,enumitem}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}
\usepackage[hidelinks]{hyperref}
\definecolor{pass}{HTML}{1E7B3A}\definecolor{fail}{HTML}{B22222}\definecolor{warn}{HTML}{B0590B}
\definecolor{muted}{HTML}{5B6670}\definecolor{bp}{HTML}{1E5AA8}\definecolor{bpdeep}{HTML}{123A6E}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.5em}\setlength{\parindent}{0pt}
\setlist{nosep,leftmargin=1.4em}
\pagestyle{fancy}\fancyhf{}\lhead{\small RNGD AI Studio --- walkthrough and results}\rhead{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\arraystretch}{1.12}
"""

ARCH = r"""\begin{figure}[H]\centering
\begin{tikzpicture}[node distance=7mm and 5mm, every node/.style={font=\footnotesize},
 box/.style={draw=bpdeep, fill=bp!8, rounded corners=2pt, align=center, minimum height=9mm, text width=24mm},
 data/.style={draw=muted, fill=white, align=center, minimum height=8mm, text width=24mm},
 ai/.style={box, fill=bp!20}, arr/.style={-{Stealth[length=2mm]}, thick, bpdeep}]
\node[data] (rfp) {Customer request (RFP text)};
\node[data, below=of rfp] (site) {Site conditions + site sketch PNG};
\node[data, below=of site] (docs) {Knowledge base + RNGD IBC / ADA docs};
\node[box, right=of site] (vec) {Vector store (MiniLM embeddings)};
\node[ai, right=of rfp, xshift=29mm] (intake) {Intake + Zoning agents};
\node[ai, right=of vec] (baap) {BaaP matching + Conflict agents};
\node[box, below=of baap] (sim) {Physics site simulation};
\node[box, right=of sim] (cv) {Computer-vision QA};
\node[ai, above=of cv] (crit) {Design Critic (loop)};
\node[ai, above=of crit] (arb) {Arbiter: final decision};
\draw[arr] (rfp) -- (intake); \draw[arr] (site) -- (vec); \draw[arr] (docs) -- (vec);
\draw[arr] (vec) -- (intake); \draw[arr] (vec) -- (baap); \draw[arr] (intake) -- (baap);
\draw[arr] (baap) -- (sim); \draw[arr] (sim) -- (cv); \draw[arr] (cv) -- (crit);
\draw[arr] (crit.west) -| node[above, pos=0.3]{\scriptsize iterate} (sim.north east); \draw[arr] (crit) -- (arb);
\end{tikzpicture}
\caption{Multi-agent pipeline. Shaded boxes call an LLM through OpenRouter; the others are deterministic Python.}
\end{figure}
\begin{figure}[H]\centering
\begin{tikzpicture}[node distance=5mm and 6mm, every node/.style={font=\footnotesize},
 box/.style={draw=bpdeep, fill=bp!8, rounded corners=2pt, align=center, minimum height=9mm, text width=27mm},
 arr/.style={-{Stealth[length=2mm]}, thick, bpdeep}]
\node[box] (p) {Program (L+L default, CSV/XLSX upload or in-app edit)};
\node[box, right=of p] (l) {Layout engine: podium + double-loaded guest floors};
\node[box, right=of l] (c) {Code screen: RNGD IBC CSV + 2010 ADA tables};
\node[box, right=of c] (d) {Drawing set (18x12 sheets) + PDF report};
\node[box, below=of l, text width=55mm, xshift=17mm] (ui) {App: KPIs, 3D, interactive plans, room mix, unit types, report results, assistant};
\draw[arr] (p) -- (l); \draw[arr] (l) -- (c); \draw[arr] (c) -- (d); \draw[arr] (l) -- (ui); \draw[arr] (c) -- (ui);
\end{tikzpicture}
\caption{Hotel Schematic mode (RNGD's first use case): deterministic, no LLM needed except the assistant.}
\end{figure}
"""


METHODS = r"""
\section{How the designs were created: methods and technology}
\label{sec:methods}
Every design in this report is produced by code in this repository. Only the agent steps of the pipeline and the chat
assistant use a language model; all geometry (site plans, floor plates, IFC models, hotel plans, drawing sheets and
3D views) is computed deterministically, so the same inputs always give the same design.

\subsection{Technology stack}
{\small
\begin{longtable}{p{3.6cm}p{4.4cm}p{8cm}}\toprule
\textbf{Layer} & \textbf{Technology} & \textbf{Used for} \\ \midrule \endhead
Language & Python 3.11 & All engines, agents, drawings and report builders. \\
User interface & Streamlit 1.52 & The five-mode web app, chat panel, program editor and report downloads. \\
Language models & OpenRouter (OpenAI-compatible SDK) & Agent steps and the assistant. A light model for parsing, a reasoning model for analysis, a Claude model for the single arbiter call. Chinese-developed models are blocked in code; requests time out after 90 s. \\
Retrieval (RAG) & sentence-transformers all-MiniLM-L6-v2 (TF-IDF fallback) & Vector store over the sample knowledge base plus RNGD's real IBC rule CSV / workbook and the ADA 2010 PDF, tagged synthetic vs real. \\
Shared state & pydantic models (\texttt{ProjectState}) & A blackboard every pipeline agent reads and writes, with a full trace of each step. \\
Layout optimization & NumPy, SciPy ndimage, scikit-image (least-cost paths), process pool & Architectural decoder, ten-objective fitness, systematic search and memetic GA, plate generator, Pareto + TOPSIS decision (Section~\ref{sec:opt}). No language model in the loop. \\
Physics simulation & NumPy / SciPy & 2D rigid-body relaxation that places the building, driveway and parking inside the buildable envelope. \\
Computer vision & SciPy ndimage, scikit-image & Reads the site sketch and re-measures the rendered site plan (setbacks, easement, overlaps, parking stalls). \\
BIM / IFC & IfcOpenShell & Writes IFC4 models (IFC Design Studio) and reads / triangulates real IFC2X3 files (Open IFC File). \\
Data inputs & pandas, openpyxl, PyMuPDF & Program CSV/XLSX upload, RNGD's IBC rule CSV and workbook, text of the ADA PDF. \\
Interactive graphics & Plotly & 3D models (Mesh3d) and hoverable floor plans in the app. \\
Drawings & Matplotlib (vector PDF + PNG) & Site plans, floor plates, the 18 x 12 in hotel drawing sheets and the isometric unit / building views. \\
Documents & LaTeX via Tectonic (XeTeX), PyMuPDF & PDF reports, merged with the drawing sheets. \\
Verification & pytest, Streamlit AppTest, Playwright + Chrome & Automated tests and the browser run that produced the screenshots in this report. \\
\bottomrule\end{longtable}}

\subsection{Multi-agent pipeline (concept to schematic)}
\begin{enumerate}
\item \textbf{Inputs.} A free-text customer request, \texttt{site\_conditions.json} and a site sketch PNG with a georeference file.
\item \textbf{Vision input (classical CV).} The sketch is thresholded and its lot outline and easement measured in feet using the georeference, then cross-checked against the stated site conditions.
\item \textbf{Intake agent (LLM).} Turns the request into a structured program: unit count and mix, floors, amenities, parking preference, budget, hard vs soft requirements. The model must return JSON; malformed JSON is sent back for correction.
\item \textbf{Zoning agent (LLM + RAG).} Retrieves the applicable zoning passages and extracts limits (height, stories, FAR, coverage, parking ratio, setbacks). The buildable envelope is then computed in plain Python, not by the model.
\item \textbf{BaaP matching agent (LLM + RAG).} Converts the unit mix into exact module counts from the BaaP catalog and estimates gross area and footprint per floor.
\item \textbf{Conflict agent (LLM).} Computes feasibility numbers (FAR, coverage, parking demand vs land) and lists conflicts with severities and recommendations.
\item \textbf{Physics simulation.} Building, driveway and parking blocks are rigid bodies with containment forces (stay inside the envelope) and separation forces (no overlaps); the system is relaxed until the overlap penalty converges, then rendered as a site plan.
\item \textbf{Vision QA.} The rendered plan image is re-measured independently at a known DPI to count stalls and detect setback, easement and overlap violations.
\item \textbf{Design critic (LLM) and loop.} Decides whether another simulation pass with adjusted parameters can fix a violation, or whether it is a program / site conflict no geometry change can solve (up to the configured number of iterations).
\item \textbf{Floor plate packer (rules).} Lays unit modules along a double-loaded corridor for a ground and a typical floor.
\item \textbf{Arbiter (one Claude call).} Reads the whole blackboard and returns APPROVE, APPROVE WITH CONDITIONS, REVISE or INSUFFICIENT INFORMATION with a narrative, unresolved issues and next steps.
\end{enumerate}

\subsection{IFC Design Studio (parametric IFC4 model)}
\begin{enumerate}
\item \textbf{Parameters.} Lot, east easement, zoning preset, unit count and mix, floors, wings, parking type (surface or podium), corridor width and floor heights. Five preset scenarios fill these in.
\item \textbf{Modules.} The unit mix is turned into module counts from the shared BaaP catalog (\texttt{data/baap\_catalog.json}), including accessible units, a core, lobby, fitness and BOH/loading.
\item \textbf{Layout.} Units are placed on double-loaded corridor wings inside the setbacks and easement. The generator tries one to three wings and keeps the best by score: all units placed, building inside the envelope, a fire lane present, fewest wings.
\item \textbf{Parking.} Surface parking is found by scanning the free part of the lot for stall rows; podium parking places stalls under the building as well.
\item \textbf{Checks.} Ten sample-zoning checks (units, stories, height, FAR, coverage, parking, setbacks/easement, fire lane, accessible units, exit travel), plus a cross-check of measurable rules from RNGD's real IBC CSV (height, stories, exit travel, dead-end corridor, exits per story).
\item \textbf{IFC output.} IfcOpenShell writes a standard IFC4 file with project, site, building and one storey per level, and the unit, core and amenity volumes and slabs on each storey. The file opens in any BIM tool; the app re-reads it for the 3D viewer.
\end{enumerate}

\subsection{Open IFC File (real model viewer)}
\begin{enumerate}
\item IfcOpenShell reads the file's header, project, site, building, storeys and element counts by class.
\item Its geometry iterator triangulates each element (fine curtain-wall framing can be skipped for speed); the meshes are merged per class with NumPy and cached so a second open is fast.
\item The model is recentred on the median of its structural elements; elements more than four building-diagonals (at least 50 m) away are dropped from the view and listed, which is why the real model fills the screen. The original file is never modified.
\end{enumerate}

\subsection{Hotel Schematic (RNGD's first use case)}
\begin{enumerate}
\item \textbf{Program.} Default: the Project Renegade Living + Lodging cover-sheet space allocation, row for row (28 rows, 117 keys, 59,350 SF). A CSV/XLSX upload or the in-app editor can replace it; area is the total for each row, and the category (guest, shared, BOH, business, tenant) is read or inferred from the name.
\item \textbf{Sorting the program.} Guest rows become keys. Small repeated rooms (Ice, IT, Jan, Supplies) are placed once on every guest floor; small single BOH rooms go into the Level 1 core zone; tenant space becomes a retail block; stairs and corridors are generated, not packed.
\item \textbf{Guest floors.} Double-loaded corridor, 6 ft clear; room depth 26 ft and room width = program area / 26 ft. Keys are spread evenly over the floors, largest units first, and alternately to the south and north sides so both ends line up. A 34 ft core at the west end holds Stair 1, two elevators and shafts; Stair 2 is at the east end, so the corridor has an exit at each end. Alternate rooms are mirrored so bathrooms pair up.
\item \textbf{Level 1.} The same core stacked below; the lobby first on the street side, other rooms balanced between the two sides; the retail block full depth at the east end, wrapping Stair 2 so stairs stack on every level.
\item \textbf{Massing and areas.} Floor-to-floor heights come from the L+L elevations (Level 2 at 16 ft, then 10.75 ft); gross area is drawn length times width per level and is compared with the source sheets and with the meeting rule (33\% circulation, 8\% wall burden, rounded up to 100 SF).
\item \textbf{Code screen.} Rules from RNGD's IBC CSV are evaluated from the geometry: stories, height, area per story, high-rise screen, exits, exit separation (1/3 diagonal), travel distance, corridor width, dead-end corridor and door width, with occupant loads from the CSV's factors. ADA 2010 Tables 224.2, 224.4 and 224.5 set the accessible-room counts. Each failing item gets a suggested fix.
\item \textbf{Room detailing.} Each room type is described once in local coordinates (walls with door openings, door swings, bathroom fixtures, beds sized by type, closet, desk, suite partition, spa tub, ADA turning circles) and then transformed into every room on every plan, so plans, enlarged plans and 3D views always agree.
\item \textbf{Drawing sheets.} Matplotlib draws vector sheets at true scale in the RNGD L+L format: title block, colour-coded zones, column grid at 30--34 ft bays with bubbles, dimension strings, elevations with an indicative prefab panel breakdown, schedules (gross area, room count, space allocation), enlarged plans and unit 3D views.
\item \textbf{3D views.} In the app, Plotly Mesh3d draws every room as a coloured block (hover for details). On the sheets, a small orthographic projector draws boxes with back-face culling and depth sorting (a painter's algorithm), shading faces by orientation; OpenGL was not used.
\item \textbf{Report.} LaTeX (Tectonic) builds the front matter and appendices; PyMuPDF merges them with the sheets; a JSON manifest records the results so the app's report tab shows them later.
\end{enumerate}

\subsection{Layout optimization (Hotel Schematic ``Optimize layouts'' and Design Optimizer)}
Section~\ref{sec:opt} gives the architecture, the mathematics and the measured comparison with the reference genetic
algorithm. In short: a decoder turns a short genome into a valid plan (straight corridor spines one room depth from the
facade, modular bays, rooms assigned to bays), a deterministic fitness scores it, and a systematic search with pattern-search
refinement (or a GA) drives the genome. In Hotel Schematic the building shape, the core and stair positions and the floor
count are generated as well, and a multi-criteria decision step ranks every distinct layout.

\subsection{How this report was made}
\texttt{src/capture\_app.py} drives the running app in headless Chrome with Playwright, waits for each step to finish, takes the
screenshots and records timings, on-page errors and the assistant's replies. \texttt{src/build\_walkthrough\_report.py}
reads those results, recomputes the deterministic ones, runs or reuses the pytest results and compiles this PDF.
\texttt{src/build\_optimizer\_study.py} runs the optimization comparison (reference GA against the updated optimizer, and the
hotel layout exploration) and saves its numbers and figures for Section~\ref{sec:opt}.
"""


def shots_by_prefix(log, prefix):
    return [s for s in log.get("shots", []) if s["file"].startswith(prefix)]


def shot_figs(log, prefixes) -> str:
    out = []
    for s in log.get("shots", []):
        if any(s["file"].startswith(p) for p in prefixes):
            out.append(fig(SCREENS / s["file"], s["caption"]))
    return "\n".join(out)


def sec_pipeline(st: ProjectState | None, log, note: str = "") -> str:
    if st is None:
        return r"\section{Multi-agent pipeline}No pipeline result was found.\par"
    fd = st.final_decision
    pr, si, bp, cf = st.program, st.site, st.baap, st.conflicts
    trace = [[str(t.step), t.agent, t.action, t.model, status_cell(t.status), f"{t.latency_ms / 1000:.1f}",
              t.summary[:150]] for t in st.trace]
    total_s = sum(t.latency_ms for t in st.trace) / 1000
    conflicts = [[c.id, status_cell(c.severity.upper()) if c.severity in ("critical", "high") else c.severity,
                  c.category, c.description, c.recommendation] for c in (cf.conflicts if cf else [])]
    sims = [[str(s.iteration), str(s.steps_run), f"{s.max_overlap_penalty:.3f}", "yes" if s.converged else "no"]
            for s in st.simulation_history]
    cvs = [[str(i + 1), f"{m.footprint_area_sf:,.0f}", f"{m.lot_coverage_pct:.1f}", str(len(m.setback_violations)),
            str(len(m.easement_violations)), str(len(m.overlap_violations)),
            f"{m.parking_stalls_detected}/{m.parking_stalls_required}", status_cell("PASS" if m.geometry_passed else "FAIL")]
           for i, m in enumerate(st.cv_metrics_history)]
    crit = "\n".join(rf"\item Iteration {i + 1}: {esc(c.rationale)} (continue: {'yes' if c.continue_iterating else 'no'})"
                     for i, c in enumerate(st.critic_history)) or r"\item No critic iterations were needed."
    z = si.zoning if si else None
    env = si.buildable_envelope if si else None
    images = "\n".join(fig(SCREENS / p, cap, r"0.8\linewidth") for p, cap in [
        ("pipeline_site_plan_iter1.png", "Site plan rendered by the physics simulation (iteration 1)."),
        ("pipeline_floor_plate_ground.png", "Ground floor plate from the rule-based packer."),
        ("pipeline_floor_plate_typical.png", "Typical floor plate."),
    ]) if (SCREENS / "pipeline_state.json").exists() else ""  # outputs/ images may be from another run
    note_box = (r"\fbox{\parbox{0.96\linewidth}{\small\textbf{Note.} " + esc(note.format(started=log.get("started", "?")))
                + "}}\\par") if note else ""
    assistant = clean_chat(log.get("assistant", {}).get("after_pipeline", ""))
    unresolved = "".join(r"\item " + esc(u) for u in (fd.unresolved_issues if fd else [])) or r"\item none"
    next_steps = "".join(r"\item " + esc(u) for u in (fd.recommended_next_steps if fd else [])) or r"\item none"
    rfp = esc(st.raw_rfp_text[:900]) + ("..." if len(st.raw_rfp_text) > 900 else "")
    return rf"""
\section{{Multi-agent pipeline (project\_001, Elm Street Multifamily --- sample data)}}
\subsection{{What it does}}
The pipeline turns an unstructured customer request and the site conditions into a Building-as-a-Product (BaaP)
schematic: agents parse the program and zoning, match BaaP modules, find conflicts, a physics-style solver places the
building and parking, computer vision re-measures the rendered plan, a critic decides whether to iterate, and one
arbiter call makes the final decision. The inputs are synthetic sample data.

\textbf{{Customer request (excerpt):}}\par
\begin{{quote}}\small {rfp}\end{{quote}}

\subsection{{Result}}
{note_box}
\begin{{tabular}}{{p{{4.2cm}}p{{11.8cm}}}}\toprule
Decision & {status_cell('PASS' if fd and fd.decision == 'APPROVE' else 'WARN')}\ \textbf{{{esc(fd.decision if fd else '-')}}} (confidence {esc(f"{fd.confidence:.0%}")}) \\
Conflicts found & {len(cf.conflicts) if cf else 0} ({', '.join(f"{sum(c.severity == s for c in cf.conflicts)} {s}" for s in ['critical', 'high', 'medium', 'low']) if cf else ''}) \\
Feasible as requested & {'yes' if cf and cf.feasible_as_requested else 'no'} \\
Simulation iterations & {len(st.simulation_history)} \\
Agent steps / total time & {len(st.trace)} steps, {total_s:.1f} s of agent time \\
\bottomrule\end{{tabular}}

\textbf{{Arbiter narrative}} (LLM output, verbatim). {esc(fd.narrative if fd else '')}

\textbf{{Unresolved issues}}
\begin{{itemize}}{unresolved}\end{{itemize}}
\textbf{{Recommended next steps}}
\begin{{itemize}}{next_steps}\end{{itemize}}

\subsection{{Program and site}}
{table(['Item', 'Value'], [
        ['Units', str(pr.unit_count_total) if pr else '-'],
        ['Unit mix', ', '.join(f'{k} {v:.0%}' for k, v in (pr.unit_mix_pct.items() if pr else []))],
        ['Floors', str(pr.floor_count) if pr else '-'],
        ['Parking preference', pr.parking_preference if pr else '-'],
        ['Budget', f'${pr.budget_usd:,.0f}' if pr and pr.budget_usd else '-'],
        ['Hard requirements', '; '.join(pr.hard_requirements) if pr else '-'],
        ['Lot', f'{si.lot_frontage_ft:.0f} x {si.lot_depth_ft:.0f} ft ({si.lot_area_sf:,.0f} SF)' if si else '-'],
        ['Setbacks', ', '.join(f'{k} {v:.0f} ft' for k, v in si.setbacks_ft.items()) if si else '-'],
        ['Buildable envelope', f'{env.width_ft:.0f} x {env.depth_ft:.0f} ft ({env.area_sf:,.0f} SF)' if env else '-'],
        ['Zoning limits', f'height {z.max_height_ft} ft, {z.max_stories} stories, FAR {z.max_far}, coverage {z.max_lot_coverage_pct}%, parking {z.parking_ratio_per_unit}/unit' if z else '-'],
        ['Site sketch cross-check (CV)', 'consistent' if si and si.cv_cross_check.get('consistent') else 'discrepancies found'],
    ], 'p{4.2cm}p{11.8cm}')}

\subsection{{BaaP plan and conflicts}}
Modules: {esc(', '.join(f'{k} x{v}' for k, v in (bp.module_counts.items() if bp else [])))}; unit area {bp.total_unit_gsf if bp else 0:,.0f} SF;
estimated building {bp.estimated_building_gsf if bp else 0:,.0f} GSF; footprint {bp.estimated_footprint_sf_per_floor if bp else 0:,.0f} SF per floor.

{table(['ID', 'Severity', 'Category', 'Description', 'Recommendation'], conflicts or [['-', '-', '-', 'none', '-']], 'p{1cm}p{1.4cm}p{1.8cm}p{5.6cm}p{5.6cm}')}

\subsection{{Simulation, vision QA and critic}}
{table(['Iteration', 'Steps', 'Max overlap penalty', 'Converged'], sims or [['-', '-', '-', '-']], 'p{2cm}p{2cm}p{3.5cm}p{2cm}')}
{table(['Pass', 'Footprint SF', 'Coverage %', 'Setback viol.', 'Easement viol.', 'Overlaps', 'Parking', 'Geometry'], cvs or [['-'] * 8], 'p{0.9cm}p{1.8cm}p{1.6cm}p{1.6cm}p{1.8cm}p{1.4cm}p{1.6cm}p{1.5cm}')}
\begin{{itemize}}{crit}\end{{itemize}}

{images}

\subsection{{Agent trace}}
{table(['#', 'Agent', 'Action', 'Model', 'Status', 's', 'Summary'], trace, 'p{0.5cm}p{2.0cm}p{2.6cm}p{2.6cm}p{1.2cm}p{0.7cm}p{6.0cm}')}

\subsection{{Assistant note after the run}}
{md(assistant) if assistant else 'Not captured.'}

\subsection{{Screenshots}}
{shot_figs(log, ['01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_'])}
"""


def sec_ifc(res, log) -> str:
    rows = []
    for r in res:
        m = r["metrics"]
        fails = [c["check"] for c in r["checks"] if c["status"] == "FAIL"]
        ibc_f = [c["rule_id"] for c in r["ibc"] if c["status"] == "FAIL"]
        rows.append([r["name"], f"{m['units_placed']}/{r['params'].units}", str(m["stories"]), str(m["far"]),
                     f"{m['coverage_pct']}%", f"{m['parking_provided']}/{m['parking_required']}",
                     status_cell("PASS") if not fails else Raw(rf"\textcolor{{fail}}{{{len(fails)} fail}}"),
                     status_cell("PASS") if not ibc_f else Raw(rf"\textcolor{{fail}}{{{esc(', '.join(ibc_f))}}}")])
    base = res[0]
    right = next((r for r in res if r["name"].startswith("Right-sized")), res[1])
    cmp_rows = []
    for a, b in zip(base["checks"], right["checks"]):
        cmp_rows.append([a["check"], str(a["required"]), str(a["provided"]), status_cell(a["status"]),
                         str(b["provided"]), status_cell(b["status"])])
    ibc_rows = [[c["rule_id"], c["rule_name"], str(c["required"]), str(c["measured"]), status_cell(c["status"])]
                for c in base["ibc"]]
    return rf"""
\section{{IFC Design Studio (parametric IFC4 generator)}}
\subsection{{What it does}}
A deterministic generator: requirements + a zoning preset + the BaaP catalog produce a building layout, written as a
real IFC4 file with IfcOpenShell and shown in an interactive 3D viewer. Every design is scored against ten
sample-zoning checks and cross-checked against a subset of RNGD's real 2021 IBC rules. Zoning presets are synthetic.

\subsection{{All five scenarios}}
{table(['Scenario', 'Units', 'Stories', 'FAR', 'Coverage', 'Parking', 'Sample zoning', 'Real IBC'], rows,
       'p{4.4cm}p{1.2cm}p{1.0cm}p{0.8cm}p{1.3cm}p{1.4cm}p{1.8cm}p{2.6cm}')}

\subsection{{Baseline vs right-sized}}
Side by side: the Baseline (120 units, surface parking) and the Right-sized scenario (60 units, podium parking)
that the app offers as the fix.
{table(['Check', 'Required', 'Baseline', '', 'Right-sized', ''], cmp_rows, 'p{4.4cm}p{2.4cm}p{2.2cm}p{1.2cm}p{2.2cm}p{1.2cm}')}

\subsection{{Baseline against RNGD's real IBC rules}}
{table(['Rule', 'Rule name', 'Required', 'Measured', 'Status'], ibc_rows, 'p{1.8cm}p{6.2cm}p{2.4cm}p{2.4cm}p{1.4cm}')}

\subsection{{Screenshots}}
{shot_figs(log, ['10_', '11_', '12_', '13_', '14_', '15_', '16_'])}
"""


def sec_explorer(s, log) -> str:
    if not s:
        return r"\section{Open IFC File}No real IFC file was available.\par"
    cats = s.get("categories") or {}
    if isinstance(cats, dict):
        cat_rows = [[k, f"{v:,}"] for k, v in sorted(cats.items(), key=lambda x: -x[1])[:18]]
    else:
        cat_rows = [[str(c[0]), str(c[1])] for c in list(cats)[:18]]
    storeys = s.get("storeys") or []
    notable = s.get("notable_named_elements") or []
    out = s.get("excluded_outliers") or []
    return rf"""
\section{{Open IFC File (a real model)}}
\subsection{{What it does}}
Opens an arbitrary real \texttt{{.ifc}} file, triangulates its geometry, recentres it on the structure (dropping far-away
outliers so the building fills the view) and summarises what is in it. The file used is real, supplied for testing.

\subsection{{Result}}
{table(['Item', 'Value'], [
        ['File', f"{s['file']} ({s['size_mb']} MB)"], ['Schema', s.get('schema')], ['Project', s.get('project_name')],
        ['Site / building', f"{s.get('site_name')} / {s.get('building_name')}"],
        ['Storeys', ', '.join(f"{x['name']} ({x['elevation_m']} m)" for x in storeys) if storeys and isinstance(storeys[0], dict) else str(len(storeys))],
        ['IFC elements', f"{s.get('total_elements', 0):,}"], ['Meshes triangulated', f"{s.get('meshes_triangulated', 0):,}"],
        ['Building diagonal', f"{s.get('building_diagonal_m', '-')} m"],
        ['Outliers excluded from the view', f"{len(out)}" + (f" (e.g. {out[0]['name']} at {out[0]['distance_m']} m)" if out else '')],
    ], 'p{4.2cm}p{11.8cm}')}
{table(['IFC class', 'Count'], cat_rows, 'p{7cm}p{3cm}')}
Notable named items: {esc(', '.join(f'{n} ({c})' for n, c in notable[:12]))}.

\subsection{{Screenshots}}
{shot_figs(log, ['17_', '18_', '19_'])}
"""


def sec_hotel(sch, details, man, log) -> str:
    p = sch.program
    n = p.net_to_gross(33, 8)
    det = details.program_detail(sch)
    prog_rows = [[r["Group"], r["Room type"], str(r["Count"]), f"{r['Area each (SF)']:,}", f"{r['Total area (SF)']:,}",
                  r["Size (W x D)"], r["Beds"], str(r["Sleeps"]), r["Placed on"], str(r["Occupant load"])]
                 for _, r in det.iterrows()]
    mix = details.mix_by_floor(sch)
    mix_rows = [[idx] + [str(int(v)) for v in row.values] for idx, row in mix.iterrows()]
    checks = [[c["rule_id"], c["check"], c["required"], c["provided"], status_cell(c["status"])] for c in sch.checks]
    notes = "".join(rf"\item \textbf{{{esc(c['rule_id'])}}}: {esc(c['note'])}" for c in sch.checks if c.get("note"))
    adv = "".join(rf"\item \textbf{{{esc(c['check'])}}}: {esc(details.failing_advice(c))}"
                  for c in sch.checks if c["status"] != "PASS" and details.failing_advice(c))
    levels = [[lv.name, f"{lv.elevation_ft:.2f}", f"{lv.length_ft:.0f} x {lv.width_ft:.0f}", f"{lv.gross_sf:,.0f}",
               f"{p.gross_by_level.get(lv.name, 0):,.0f}", str(lv.keys_placed)] for lv in sch.levels]
    sheets = (man or {}).get("sheets", [])
    sheet_rows = [[x["number"], x["title"]] for x in sheets]
    sheet_figs = "\n".join(fig(Path(x["png"]), f"{x['number']} - {x['title']} (from the generated PDF drawing set)")
                           for x in sheets if x["number"] in ("G001", "A101", "A102", "A201", "A401", "A501"))
    return rf"""
\section{{Hotel Schematic (RNGD's first use case)}}\label{{sec:hotel}}
\subsection{{What it does}}
Takes a detailed hotel program --- by default the Project Renegade Living + Lodging cover-sheet space allocation
(28 rows, 117 keys), or an uploaded / in-app edited CSV or XLSX --- and produces a schematic: a Level 1 podium with
lobby, BOH and retail, five double-loaded guest floors with a core and a stair at each end, a code screen against
RNGD's real IBC rule CSV and the 2010 ADA guest-room tables, and an 18 x 12 in drawing set in RNGD's L+L sheet format.

\subsection{{Headline results}}
\begin{{tabular}}{{p{{5cm}}p{{11cm}}}}\toprule
Keys & {p.keys} on {len(sch.levels) - 1} guest floors \\
Net program area & {p.net_sf:,.0f} SF ({len(p.rows)} rows) \\
Gross area: source sheets / meeting rule / drawn & {p.gross_sf:,.0f} / {n['required_gross_sf']:,.0f} / {sum(lv.gross_sf for lv in sch.levels):,.0f} SF \\
Massing & podium {sch.levels[0].length_ft:.0f} x {sch.levels[0].width_ft:.0f} ft, tower {sch.levels[1].length_ft:.0f} x {sch.levels[1].width_ft:.0f} ft, {sch.height_ft:.2f} ft to roof \\
Code screen & {sum(c['status'] == 'PASS' for c in sch.checks)} pass, {sum(c['status'] == 'FAIL' for c in sch.checks)} fail, {sum(c['status'] == 'REVIEW' for c in sch.checks)} review, {sum(c['status'] == 'ASSUMED' for c in sch.checks)} assumed \\
Drawing set & {len(sheets)} sheets, {(man or {}).get('pages', '-')} pages, generated {esc((man or {}).get('generated', '-').replace('T', ' '))} \\
\bottomrule\end{{tabular}}

\subsection{{Code and ADA screen}}
{table(['Rule', 'Check', 'Required', 'Provided', 'Status'], checks, 'p{1.8cm}p{4.4cm}p{4.4cm}p{3.4cm}p{1.4cm}')}
\textbf{{Notes}}\begin{{itemize}}{notes}\end{{itemize}}
\textbf{{Suggested next steps for the open items}}\begin{{itemize}}{adv}\end{{itemize}}

\subsection{{Levels}}
{table(['Level', 'Floor at (ft)', 'Footprint (ft)', 'Gross drawn (SF)', 'Gross source (SF)', 'Keys'], levels, 'p{2cm}p{2cm}p{2.6cm}p{2.8cm}p{2.8cm}p{1.2cm}')}

\subsection{{Detailed program}}
{table(['Group', 'Type', 'Count', 'Each SF', 'Total SF', 'Size', 'Beds', 'Sleeps', 'Placed on', 'Occ.'], prog_rows,
       'p{1.4cm}p{3.0cm}p{0.8cm}p{1.0cm}p{1.1cm}p{1.6cm}p{1.6cm}p{0.9cm}p{2.2cm}p{0.7cm}')}

\subsection{{Room mix by floor}}
{table(['Type'] + list(mix.columns), mix_rows, 'p{3.4cm}' + 'p{1.5cm}' * len(mix.columns))}

\subsection{{App screenshots}}
{shot_figs(log, ['20_', '21_', '22_', '23_', '24_', '25_', '26_', '27_', '28_', '29_', '30_'])}

\subsection{{Generated drawing set}}
The PDF report (\texttt{{reports/final/RNGD\_final\_schematic\_report.pdf}}) contains these sheets:
{table(['Sheet', 'Content'], sheet_rows, 'p{1.6cm}p{13cm}')}
{sheet_figs}
"""


def sec_assistant(log) -> str:
    a = log.get("assistant", {})
    return rf"""
\section{{Assistant}}
The chat panel answers with the live state of whichever mode is open (pipeline result, IFC design, real IFC summary,
hotel schematic) plus retrieved passages from the knowledge base and RNGD's documents.

\textbf{{Question (one-click button):}} {esc(a.get('hotel_question', '-'))}

\textbf{{Answer, as given in the run}} (reproduced verbatim; LLM answers are not fact-checked and can misstate code details --- the code screen in Section~\ref{{sec:hotel}} is the authoritative result):\par
\begin{{quote}}{md(clean_chat(a.get('hotel_answer', ''))) or 'Not captured.'}\end{{quote}}
{shot_figs(log, ['31_'])}
"""


def sec_tests(t) -> str:
    if not t:
        return r"\section{Verification}Tests were not run for this build (--no-tests).\par"
    rows = [[c["file"], c["name"], f"{c['time']:.1f}", status_cell(c["outcome"])] for c in t["cases"]]
    passed = sum(c["outcome"] == "PASS" for c in t["cases"])
    return rf"""
\section{{Verification: automated tests}}
The full pytest suite was run as part of building this report: {passed} of {len(t['cases'])} tests passed
({esc(t['summary'])}). App tests drive the real Streamlit app through Streamlit's AppTest, including a full pipeline
run with real LLM calls.
{table(['File', 'Test', 's', 'Result'], rows, 'p{2.6cm}p{9.6cm}p{1.0cm}p{1.4cm}')}
"""


def build(tests: bool = True) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    log_f = SCREENS / "capture_log.json"
    log = json.loads(log_f.read_text(encoding="utf-8")) if log_f.exists() else {}
    print("collecting results...")
    st, pipe_note = pipeline_results()
    ifc = ifc_results()
    exp = explorer_results()
    sch, details, man = hotel_results()
    t = run_tests(reuse="--reuse-tests" in sys.argv) if tests else None
    study = walkthrough_optimizer.load_study()
    if study:
        study["report_model"] = config.REPORT_MODEL
    print("writing LaTeX...")
    fd = st.final_decision if st else None
    ifc_ok = [r["name"] for r in ifc if not any(c["status"] == "FAIL" for c in r["checks"])]
    hotel_fail = [c["check"] for c in sch.checks if c["status"] == "FAIL"]
    timings = log.get("timings", {})
    base = ifc[0]
    base_fails = [c["check"] for c in base["checks"] if c["status"] == "FAIL"]
    rs = next((r for r in ifc if r["name"].startswith("Right-sized")), None)
    rs_ibc = [f"{c['rule_id']} ({c['rule_name']})" for c in (rs["ibc"] if rs else []) if c["status"] == "FAIL"]
    ifc_finding = (r"\item \textbf{IFC Design Studio:} " + esc(
        f"the Baseline ({base['params'].units} units) fails {len(base_fails)} sample-zoning check(s): {', '.join(base_fails) or 'none'}. "
        f"Scenarios passing all sample checks: {', '.join(ifc_ok) or 'none'}. "
        + (f"Against RNGD's real IBC rules the Right-sized scenario still fails: {'; '.join(rs_ibc)}." if rs_ibc
           else "The Right-sized scenario also passes the real IBC subset.")))
    ada = next((c for c in sch.checks if c["rule_id"] == "ADA-224.2"), None)
    hotel_finding = (r"\item \textbf{Hotel:} " + esc(
        f"{sch.program.keys - sch.unplaced_keys} of {sch.program.keys} keys placed on {len(sch.levels) - 1} guest floors; "
        f"failing checks: {', '.join(hotel_fail) or 'none'}"
        + (" (a Type IA podium or one fewer guest floor resolves the story limit)" if any('Stories' in f for f in hotel_fail) else "")
        + (f". ADA mobility rooms: {ada['provided']} provided vs {ada['required']} ({ada['status']})." if ada else ".")))
    errs = log.get("errors", [])
    summary_rows = [
        ["Multi-Agent Pipeline", f"{fd.decision} ({fd.confidence:.0%})" if fd else "-",
         f"{len(st.conflicts.conflicts) if st and st.conflicts else 0} conflicts; {len(st.trace) if st else 0} agent steps"],
        ["IFC Design Studio", f"{len(ifc_ok)}/{len(ifc)} scenarios pass sample zoning",
         f"passing: {', '.join(ifc_ok)}" if ifc_ok else "all scenarios fail at least one check"],
        ["Open IFC File", f"{exp['schema']}, {exp['total_elements']:,} elements" if exp else "-",
         f"{len(exp.get('storeys', []))} storeys; {len(exp.get('excluded_outliers', []))} outliers excluded" if exp else "-"],
        ["Hotel Schematic", f"{sch.program.keys} keys, {len(sch.levels)} levels",
         f"{sum(c['status'] == 'PASS' for c in sch.checks)}/{len(sch.checks)} checks pass; fails: {', '.join(hotel_fail) or 'none'}"],
        ["Automated tests", f"{sum(c['outcome'] == 'PASS' for c in t['cases'])}/{len(t['cases'])} pass" if t else "not run", ""],
    ]
    opt_finding = ""
    if study:
        ss_, hh = study["sites"], study["hotel"]
        ra, rd = (sum(x["variants"][k]["rooms_placed"] for x in ss_) for k in ("A", "D"))
        wa, wd = (sum(x["variants"][k]["windowless_guest_rooms"] for x in ss_) for k in ("A", "D"))
        sd = sum(x["variants"]["D"]["seconds"] for x in ss_)
        hb = hh["best"]
        summary_rows.insert(4, ["Layout optimization", f"{rd} rooms placed on the 3 QA sites, {wd} without a window",
                                f"reference GA: {ra} rooms, {wa} without a window; hotel: {hh['layouts']} layouts in "
                                f"{hh['seconds']:.0f} s, first choice {hb['plate']}"])
        opt_finding = (r"\item \textbf{Layout optimization (Section~\ref{sec:opt}):} " + esc(
            f"on the three QA sites the updated optimizer places {rd} rooms with {wd} windowless guest rooms in {sd:.0f} s, "
            f"against {ra} rooms with {wa} windowless for the reference GA (same program, same evaluator). In Hotel "
            f"Schematic it generated and ranked {hh['layouts']} layouts in {hh['seconds']:.0f} s; the first-ranked one "
            f"({hb['plate']}, {hb['floors']} guest floors) places {hb['metrics']['keys_placed']} of "
            f"{hb['metrics']['keys_requested']} keys. No language model is used in the optimization."))
    timing_rows = [[k.replace("_s", "").replace("_", " "), f"{v:.1f} s"] for k, v in timings.items()]
    shot_rows = [[s["file"], s["mode"], s["caption"]] for s in log.get("shots", [])]
    err_sec = (r"\section{On-page errors}" + table(["Where", "Error"], [[e["where"], e["text"][:300]] for e in errs],
                                                 "p{4cm}p{12cm}")) if errs else ""
    import streamlit
    env_rows = [["Run date", log.get("started", datetime.now().strftime("%Y-%m-%d %H:%M"))],
                ["Python / Streamlit", f"{platform.python_version()} / {streamlit.__version__}"],
                ["OS", f"{platform.system()} {platform.release()}"],
                ["LLM provider", "OpenRouter (Chinese-developed models are blocked in code)"],
                ["Light model", config.LLM_LIGHT], ["Reasoning model", config.LLM_REASONING],
                ["Arbiter model", config.ARBITER_MODEL], ["Report-writer model", config.REPORT_MODEL],
                ["Embeddings", config.EMBEDDING_MODEL_NAME],
                ["Screenshots", f"{len(log.get('shots', []))} at {1680} x {1050} px, headless Chrome"],
                ["On-page errors during the run", str(len(errs))]]
    tex = PREAMBLE + rf"""\begin{{document}}
\begin{{titlepage}}\centering
\vspace*{{2cm}}{{\fontsize{{40}}{{44}}\selectfont\bfseries RNGD}}\par\vspace{{0.6cm}}
{{\LARGE Concept-to-Schematic AI Studio}}\par\vspace{{0.3cm}}
{{\Large System walkthrough and results}}\par\vspace{{1cm}}
{{\large A full run of every mode, with screenshots and the actual results of that run}}\par\vspace{{2cm}}
\begin{{tabular}}{{ll}}Run & {esc(log.get('started', '-'))} to {esc(log.get('finished', '-'))} \\
Report built & {datetime.now().strftime('%Y-%m-%d %H:%M')} \\ App & \texttt{{src/app.py}} (Streamlit) \\\end{{tabular}}
\vfill{{\small\color{{fail}} Semester proof of concept. Pipeline, IFC-studio zoning and BaaP data are synthetic; the
IFC file, IBC rule set, ADA standards and hotel program are RNGD-supplied. Not for construction.}}
\end{{titlepage}}
\tableofcontents\newpage

\section{{Executive summary}}
This report documents a complete run of the RNGD AI Studio: every mode was driven through the real app in a
browser, the way a user would use it, and each screen was captured. The numbers below come from that run.
{table(['Mode', 'Headline result', 'Detail'], summary_rows, 'p{3.4cm}p{5.2cm}p{7.6cm}')}
\textbf{{Key findings}}
\begin{{itemize}}
\item \textbf{{Pipeline:}}{' (see the data-source note in the pipeline section)' if pipe_note else ''} the arbiter returned \textbf{{{esc(fd.decision if fd else '-')}}}. {esc((fd.narrative if fd else '')[:420])}{'...' if fd and len(fd.narrative) > 420 else ''}
{ifc_finding}
\item \textbf{{Real IFC file:}} opened and summarised without errors; far-away outliers are excluded so the building fills the view.
{hotel_finding}
{opt_finding}
\item \textbf{{Quality:}} {len(errs)} on-page errors during the browser run; {(str(sum(c['outcome'] == 'PASS' for c in t['cases'])) + ' of ' + str(len(t['cases'])) + ' automated tests passed') if t else 'tests not run'}.
\end{{itemize}}

\section{{How the run was done}}
\begin{{enumerate}}
\item The app was started with \texttt{{streamlit run src/app.py}} on \texttt{{localhost:8501}}.
\item \texttt{{src/capture\_app.py}} drove headless Chrome through every mode: ran the full pipeline, generated the
Baseline and Right-sized IFC scenarios, opened the real IFC file, used every tab of the hotel workspace, generated the
hotel PDF, and asked the assistant a question. It waited for each step to finish before taking a screenshot.
\item This report reads the results those steps saved (\texttt{{outputs/project\_001/project\_state.json}}, the hotel
report manifest), recomputes the deterministic results (IFC scenarios, hotel schematic) and runs the test suite.
\end{{enumerate}}
{table(['Setting', 'Value'], env_rows, 'p{5cm}p{11cm}')}
{table(['Step', 'Time'], timing_rows, 'p{6cm}p{3cm}') if timing_rows else ''}

\section{{System architecture}}
Section~\ref{{sec:methods}} explains how each design is generated and which technology is used.
{ARCH}
{METHODS}
{sec_pipeline(st, log, pipe_note)}
{sec_ifc(ifc, log)}
{sec_explorer(exp, log)}
{sec_hotel(sch, details, man, log)}
{walkthrough_optimizer.section(study)}
{sec_assistant(log)}
{sec_tests(t)}

\section{{Limitations}}
\begin{{itemize}}
\item Pipeline and IFC-studio inputs (customer request, site, zoning presets, BaaP catalog) are synthetic sample data.
\item LLM outputs vary between runs; the pipeline numbers above are from this run only.
\item The hotel layout is a rule-based schematic: parking, structure and MEP are not modelled; fixtures are typical.
\item The optimized layouts are rectangles on a 1 ft grid with one plate for all levels; open items are listed at the end of Section~\ref{{sec:opt}}.
\item Code screens are planning-level, not code determinations. 3D views are WebGL (app) and painter's-algorithm axonometrics (PDF).
\end{{itemize}}

\appendix
\section{{Screenshot index}}
{table(['File', 'Mode', 'Caption'], shot_rows, 'p{4.4cm}p{2.8cm}p{8.8cm}')}
{err_sec}
\end{{document}}
"""
    texf = OUT / "RNGD_walkthrough_report.tex"
    texf.write_text(tex, encoding="utf-8")
    print("compiling...")
    r = subprocess.run([str(TECTONIC), texf.name], cwd=OUT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("LaTeX failed")
    return OUT / "RNGD_walkthrough_report.pdf"


if __name__ == "__main__":
    print(build(tests="--no-tests" not in sys.argv))
