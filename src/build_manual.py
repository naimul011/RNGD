"""Builds reports/RNGD_manual.tex + .pdf — a user manual for the unified app
(both modes + chat assistant), using real numbers from the latest pipeline
run and real generated images. Run: python src/build_manual.py"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from rngd import config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
TECTONIC = ROOT / "Testt" / "tools" / "tectonic.exe"

state_path = config.OUTPUTS_DIR / "project_001" / "project_state.json"
if not state_path.exists():
    sys.exit("No pipeline run found. Run `python src/run_pipeline.py` at least once first.")
state = json.loads(state_path.read_text(encoding="utf-8"))
fd = state["final_decision"]
cv = state["cv_metrics_history"][-1]
conflicts = state["conflicts"]["conflicts"]
n_trace = len(state["trace"])

for name in ("site_plan_iter1.png", "floor_plate_typical.png", "floor_plate_ground.png"):
    shutil.copy(config.OUTPUTS_DIR / "project_001" / name, OUT / name)

tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx,booktabs,hyperref,float,enumitem,xcolor,array}}
\setlength{{\parskip}}{{0.5em}}\setlength{{\parindent}}{{0pt}}
\title{{RNGD AI Studio\\\large User Manual}}
\author{{RNGD semester proof-of-concept}}
\date{{\today}}
\begin{{document}}
\maketitle
\tableofcontents

\section{{What this is}}
RNGD AI Studio is one Streamlit app, \texttt{{src/app.py}}, with two modes and a chat assistant that is always
visible regardless of which mode you're in.

\begin{{itemize}}
\item \textbf{{Multi-Agent Pipeline}} — ten agents turn an unstructured customer request and raw site conditions
into a schematic: a site plan, a floor plate, a conflict report, and one final AI decision.
\item \textbf{{IFC Design Studio}} — a faster, parametric generator: move a slider or click a scenario and get an
instant 3D building written as a real, standard IFC4 file, scored against ten compliance checks.
\item \textbf{{Assistant}} — a chat panel on the right that knows about every component of both modes, answers
questions in plain language, and posts its own observations after a run finishes.
\end{{itemize}}
All zoning rules and the BaaP module catalog are synthetic sample data. Outputs are planning-level, not
code-approved; a licensed professional must review any real design.

\section{{Getting started}}
\begin{{enumerate}}[nosep]
\item \texttt{{pip install -r requirements.txt}}
\item Create \texttt{{RNGD/.env}} with \texttt{{GROQ\_API\_KEY}} and \texttt{{ANTHROPIC\_API\_KEY}} (not committed to git).
\item \texttt{{python src/make\_sample\_data.py}} (once, generates the sample site sketch).
\item \texttt{{streamlit run src/app.py}} — opens in your browser at \texttt{{http://localhost:8501}}.
\end{{enumerate}}

\section{{The interface}}
The page has three regions, all visible at once on a normal-width window:
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{@{{}}lp{{10.5cm}}@{{}}}}\toprule
Region & Contents \\ \midrule
Header & Title banner with live status pills: how many pipeline agents, how many IFC scenarios, the knowledge-base
backend and chunk count, and the last decision once a pipeline run has completed. \\
Sidebar (left) & A Mode switch (Multi-Agent Pipeline / IFC Design Studio) and that mode's controls — everything
below the switch changes depending on which mode is selected. \\
Main column (center) & Whichever mode's tabs and content are showing. \\
Chat panel (right) & Always the assistant, in every mode, with its own scrollable message history and an input box
pinned to the bottom of the page. \\ \bottomrule
\end{{tabular}}\end{{table}}

\section{{Mode 1: Multi-Agent Pipeline}}
\subsection{{Running it}}
In the sidebar: pick the sample project (currently one, \texttt{{project\_001}}), set the maximum number of
simulation iterations (default {config.MAX_SIMULATION_ITERATIONS}), and click \textbf{{Run full pipeline}}.

\subsection{{Watching it run}}
A live status panel opens in the main column and updates in place as each agent finishes — you see the agent
name, what it did, how long it took, and a one-line summary, in order, as it happens, not just a table at the
end. When the run finishes the panel collapses to a single line showing the final decision.

\subsection{{Reading the results}}
Eight tabs appear once a run has completed:
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{@{{}}lp{{10cm}}@{{}}}}\toprule
Tab & What's there \\ \midrule
Overview & Decision, confidence, conflict count, the narrative, and the two headline images (site plan, typical floor). \\
Agent trace & Every step as a table: model used, status, latency, summary. \\
Program \& site & The parsed customer program and the computed zoning/site constraints, plus the input sketch. \\
BaaP \& conflicts & The module-matching plan and the full conflict report with severities. \\
Site simulation & Each physics-simulation iteration's stats and rendered site plan, the critic's suggestions, and a convergence chart if more than one iteration ran. \\
Floor plates & Ground floor and typical upper floor, side by side. \\
Final decision & The Arbiter's full narrative, unresolved issues and recommended next steps. \\
Ask an agent & Pick one agent and ask it a one-off question grounded in its own domain (separate from the chat panel — this talks to a single named agent). \\ \bottomrule
\end{{tabular}}\end{{table}}

\subsection{{Sample run (most recent, this app)}}
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{ll}}\toprule
Decision & \textbf{{{fd['decision']}}} (confidence {fd['confidence'] * 100:.0f}\%) \\
Trace steps & {n_trace} \\
Conflicts found & {len(conflicts)} \\
Geometry passed (CV QA) & {cv['geometry_passed']} \\
Parking & {cv['parking_stalls_detected']} / {cv['parking_stalls_required']} stalls \\
Lot coverage & {cv['lot_coverage_pct']}\% \\ \bottomrule
\end{{tabular}}\end{{table}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.46\linewidth]{{site_plan_iter1.png}}\hfill\includegraphics[width=0.46\linewidth]{{floor_plate_typical.png}}
\caption{{Generated site plan (left) and typical upper floor (right) from the run above.}}
\end{{figure}}

\section{{Mode 2: IFC Design Studio}}
\subsection{{Using it}}
Switch the sidebar Mode to \textbf{{IFC Design Studio}}. Either click a scenario button to load a preset in one
click, or move the controls yourself: units, residential floors, studio/1BR/2BR mix, zoning preset, parking type
(surface or podium), and number of wings (0 = let the program choose). Every change regenerates the design —
there is no separate "run" button, results update as you move a slider.

You can also paste a free-text customer request into the ``Paste a customer request'' expander and click
\textbf{{Extract requirements}} to have the same AI intake agent used by the pipeline fill in units/floors/mix.

\subsection{{Reading the results}}
The top strip always shows: units placed, stories, FAR, lot coverage, parking provided/required, and how many of
the ten checks failed. Five tabs follow:
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{@{{}}lp{{10cm}}@{{}}}}\toprule
Tab & What's there \\ \midrule
3D model & An interactive Plotly view of the actual IFC file (drag to rotate, scroll to zoom, click a legend entry to hide a unit type, limit visible floors). \\
Site plan & A top-down 2D view: lot, easement, buildable envelope, driveway, parking, wings. \\
Compliance & The ten PASS/FAIL checks as a table, plus a button to have the assistant write a short review of this specific design. \\
How this was built & A plain-language, step-by-step account of \emph{{this}} design: the envelope math, the unit mix, why this many wings were chosen, the parking calculation, and the result. \\
Download & The \texttt{{.ifc}} file and a standalone \texttt{{.html}} 3D viewer you can open in Chrome without the app running. \\ \bottomrule
\end{{tabular}}\end{{table}}

\subsection{{Two contrasting designs}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.46\linewidth]{{../reports/ifc_design/plan_0.png}}\hfill\includegraphics[width=0.46\linewidth]{{../reports/ifc_design/plan_1.png}}
\caption{{Baseline 120 units, surface parking (left, fails coverage/FAR/parking) vs. the right-sized 60-unit
podium design (right, passes all ten checks).}}
\end{{figure}}
For the full method behind these numbers (formulas, packing algorithm, every scenario), see
\texttt{{reports/ifc\_design/ifc\_design\_report.pdf}}.

\section{{The chat assistant}}
The panel on the right is available in both modes and keeps its conversation across mode switches. It is told,
in its own system prompt, about every agent in the pipeline and every part of the IFC studio, so it can answer
general questions (``what is FAR'', ``what does the Design Critic Agent do'') even before you run anything.

Once you do run something, the assistant is given a compact summary of the current state — the parsed program,
the buildable envelope, the conflict list, the latest CV metrics and the final decision for the pipeline; the
requested parameters, metrics and failed checks for the IFC studio — so its answers are grounded in \emph{{your}}
actual numbers, not generic ones. It also volunteers a short note on its own:
\begin{{itemize}}[nosep]
\item automatically, right after a pipeline run finishes;
\item on demand, from the ``Ask the assistant to review this design'' button in the IFC studio's Compliance tab
(this is on-demand rather than automatic there, since every slider move regenerates a design and an automatic
note on every tiny change would be noisy).
\end{{itemize}}
Example questions: \emph{{``Why did parking fail?''}}, \emph{{``What's the difference between the two modes?''}},
\emph{{``If I switch to podium parking, what should I expect?''}}, \emph{{``What is a double-loaded corridor?''}}

\section{{Testing}}
\texttt{{python -m pytest tests/}} runs the full suite:
\begin{{itemize}}[nosep]
\item pipeline agent and IFC-engine logic tests (module counts, compliance scoring, IFC round-trip through IfcOpenShell);
\item app-level tests using Streamlit's \texttt{{AppTest}}, which drive the real app: load without error, switch
mode and click a scenario button, submit a chat message, and — the slowest one — click \emph{{Run full pipeline}}
in the actual sidebar and check the live-status wiring, the final decision and the proactive chat note all worked.
\end{{itemize}}
This last test is what would have caught an earlier bug where the app crashed on a relative launch path; it stays
in the suite for exactly that reason.

\section{{Troubleshooting}}
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{@{{}}p{{5.2cm}}p{{7.5cm}}@{{}}}}\toprule
Symptom & Fix \\ \midrule
\texttt{{IndexError}} on startup mentioning \texttt{{parents[...]}} & Launch from the repo root as \texttt{{streamlit run src/app.py}}, or make sure any \texttt{{Path(\_\_file\_\_)}} use is \texttt{{.resolve()}}d first — already fixed in this app, kept here as a reminder if you copy the pattern elsewhere. \\
"Port already in use" & Add \texttt{{--server.port 8502}} (or any free port) to the \texttt{{streamlit run}} command. \\
Chat button greyed out / no reply & Check \texttt{{GROQ\_API\_KEY}} is set in \texttt{{.env}} and the process was restarted after adding it. \\
"Extract requirements" fails & The Groq intake agent needs \texttt{{GROQ\_API\_KEY}}; everything else in the IFC studio works without any key. \\
First load feels slow & The knowledge-base embedding model downloads/loads once per process start (a few seconds); subsequent reruns reuse it. \\ \bottomrule
\end{{tabular}}\end{{table}}

\section{{Where things live}}
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{@{{}}lp{{9cm}}@{{}}}}\toprule
Path & Contents \\ \midrule
\texttt{{src/app.py}} & The unified app (this manual). \\
\texttt{{src/rngd/agents/}}, \texttt{{orchestrator.py}} & The nine pipeline agents + the harness that runs them. \\
\texttt{{src/rngd/ifc/}} & The IFC studio engine: \texttt{{design.py}} (layout + rules), \texttt{{ifc\_writer.py}} (IFC I/O), \texttt{{viewer3d.py}} (3D figure). \\
\texttt{{src/rngd/chat.py}} & The chat assistant. \\
\texttt{{data/}} & Sample knowledge base, BaaP catalog, and the one sample project. \\
\texttt{{outputs/}} & Generated images/files from the last run of each mode. \\
\texttt{{reports/}} & This manual, the IFC studio's own deep-dive report, and the RVT-reading report. \\
\texttt{{tests/}} & The full pytest suite described above. \\
\texttt{{Testt/}} & Separate side experiments that read raw \texttt{{.rvt}} files and a pyRevit wrapper; see \texttt{{Testt/README.md}}. \\ \bottomrule
\end{{tabular}}\end{{table}}

\section{{Limitations}}
\begin{{itemize}}[nosep]
\item All zoning rules and the BaaP catalog are synthetic; swapping in RNGD's real data is a data change, not a code change.
\item The pipeline's simulation/CV loop and the IFC studio's layout algorithm are both planning-level heuristics, not licensed engineering.
\item The chat assistant answers from the current state and the knowledge base; it says so when it doesn't have enough grounding rather than guessing.
\end{{itemize}}
\end{{document}}
"""
(OUT / "RNGD_manual.tex").write_text(tex, encoding="utf-8")
r = subprocess.run([str(TECTONIC), "RNGD_manual.tex"], cwd=OUT, capture_output=True, text=True)
print(r.stderr[-1500:])
sys.exit(r.returncode)
