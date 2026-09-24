"""Builds report/rvt_report.tex from the real RVT data and compiles it to PDF
with the bundled tectonic binary.  Run: python build_report.py"""
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import rvt_tools as rt

HERE = Path(__file__).parent
OUT = HERE / "report"
OUT.mkdir(exist_ok=True)
TECTONIC = HERE / "tools" / "tectonic.exe"


def esc(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s


files = rt.find_rvts(HERE / "extracted")
sections = []
for f in files:
    tag = f.parent.name.replace("rvtqsg-Sample_", "")
    info = rt.basic_info(f)
    cats = rt.category_counts(f)
    top = list(cats.items())[:10]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#1E5AA8")
    ax.set_xlabel("keyword hits in element database")
    fig.tight_layout()
    chart = OUT / f"cats_{tag}.png"
    fig.savefig(chart, dpi=160)
    plt.close(fig)

    prev = rt.extract_preview(f, OUT / f"preview_{tag}.png")
    im = Image.open(prev)
    im.resize((im.width * 3, im.height * 3), Image.LANCZOS).save(prev)

    names = {t: [s for s, _ in rt.search_strings(f, t, 6)] for t in ("Level ", "Bedroom", "Kitchen")}
    sections.append((tag, f, info, top, chart.name, prev.name, names))

tex = [r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx,booktabs,hyperref,float}
\title{Reading Revit (\texttt{.rvt}) Files Without Revit\\\large Test Report}
\author{RNGD side-testing project}
\date{\today}
\begin{document}
\maketitle
\section{Summary}
Three archives were extracted from \texttt{Testt/}, yielding three Revit~2022 sample
models. \textbf{The architecture and structural archives contain the byte-identical file}
(\texttt{racbasicsampleproject.rvt}, same MD5), so only two distinct models exist.
Using only \texttt{olefile} and \texttt{zlib} we can read: file metadata, the embedded
thumbnail, and the element database's text (level, room, family and view names).
We \emph{cannot} recover 3D geometry; that needs Revit, Autodesk Platform Services
or Dynamo. The included tools are a Python module (\texttt{rvt\_tools.py}), a CLI, a
Streamlit viewer, and a 17-test pytest suite (all passing).

\section{Method}
An \texttt{.rvt} is an OLE2 compound file. \texttt{BasicFileInfo} gives the Revit version
and build; \texttt{RevitPreview4.0} holds a PNG thumbnail; \texttt{Partitions/12} is the
element database, stored as many raw-deflate chunks behind gzip headers. Decompressing
every valid chunk and scanning for UTF-16 strings yields names of levels, rooms and
families. Category counts are keyword hit counts, a heuristic and not exact element counts.
"""]
for tag, f, info, top, chart, prev, names in sections:
    tex.append(rf"""
\section{{{esc(tag.capitalize())} model}}
\begin{{tabular}}{{ll}}\toprule
File & \texttt{{{esc(f.name)}}} \\
Size & {info['file_size_mb']} MB \\
Revit version / build & {info['revit_version']} / {esc(info['build'] or '')} \\
\bottomrule\end{{tabular}}

\begin{{figure}}[H]\centering
\includegraphics[width=0.45\linewidth]{{{prev}}}
\caption{{Embedded preview thumbnail (128\,px original, upscaled 3$\times$).}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.8\linewidth]{{{chart}}}
\caption{{Category keyword signal (heuristic).}}
\end{{figure}}
\noindent Sample recovered names: levels: {esc(', '.join(names['Level '][:4]) or 'none')};
bedrooms: {esc(', '.join(names['Bedroom'][:3]) or 'none')};
kitchen items: {esc(', '.join(names['Kitchen'][:3]) or 'none')}.
""")
import pyrevit_tools as pt

pr_rows = []
seen = set()
for f in files:
    info = pt.file_info(f)
    key = info.get("document_id")
    if key in seen:
        continue
    seen.add(key)
    pi = info["project_information"]
    pr_rows.append((esc(f.name), esc(pi.get("Project Name", "")), esc(pi.get("Client Name", "")), esc(info.get("document_id", "")[:8]), esc(info.get("workshared", ""))))
tbl = "".join(rf"{a} & {b} & {c} & {d} & {e} \\" + "\n" for a, b, c, d, e in pr_rows)
tex.append(rf"""
\section{{pyRevit integration}}
pyRevit {esc(pt.cli_version().split('+')[0].replace('pyrevit ', ''))} was installed per-user
(\texttt{{\%APPDATA\%/pyRevit-Master}}). \textbf{{Revit itself is not installed on this machine}}
(\texttt{{pyrevit env}} lists no Revit installs), so pyRevit cannot attach and scripts that use the
Revit API cannot run here. Two layers were built:
\begin{{enumerate}}
\item \textbf{{Works now, no Revit:}} \texttt{{pyrevit\_tools.py}} wraps
\texttt{{pyrevit revits fileinfo}}, which reads project name, client, build, workshared flag and
document GUID directly from the file. Results (identical files de-duplicated):
\end{{enumerate}}
\begin{{center}}\small\begin{{tabular}}{{lllll}}\toprule
File & Project name & Client & Doc ID & Workshared \\ \midrule
{tbl}\bottomrule\end{{tabular}}\end{{center}}
\begin{{enumerate}}\setcounter{{enumi}}{{1}}
\item \textbf{{Needs Revit:}} the extension \texttt{{pyrevit\_extension/RNGDTest.extension}}
(registered in pyRevit's search path) adds an \emph{{RNGD}} ribbon tab with two buttons:
\emph{{Summarize model}} (element counts per category, levels, rooms with areas, wall types $\rightarrow$ JSON)
and \emph{{Export walls}} (centerlines, heights, types $\rightarrow$ JSON). These are \textbf{{untested}}
until Revit is available; \texttt{{pyrevit\_tools.load\_revit\_summary}} reads their output.
\end{{enumerate}}
The RVT sample files were saved by Revit 2022 (build 20210129); opening them needs Revit 2022 or newer.
""")
tex.append(r"""
\section{Limitations and next steps}
\begin{itemize}
\item Only a 128\,px thumbnail is embedded; there is no geometry to render here.
\item For real geometry/quantities: Autodesk Platform Services Model Derivative API
(cloud, needs an APS key), or Revit/Dynamo/pyRevit locally.
\item The string harvest is heuristic; treat category counts as relative signal only.
\item Relevance to RNGD: these sample models could seed the BaaP catalog work once
extraction via APS or Revit is available.
\end{itemize}
\end{document}
""")
(OUT / "rvt_report.tex").write_text("".join(tex), encoding="utf-8")
r = subprocess.run([str(TECTONIC), "rvt_report.tex"], cwd=OUT, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-1500:])
sys.exit(r.returncode)
