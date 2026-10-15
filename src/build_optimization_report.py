"""Report: how the layout-optimization algorithms work, with worked examples.

Every number and figure in the examples comes from src/optimization_examples.py (the real
optimizer run on site fp1) and from the comparison study (src/build_optimizer_study.py).

Usage: python src/build_optimization_report.py [--recompute]
Output: reports/optimization/RNGD_optimization_algorithms.pdf (+ .tex, figures, examples.json)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "optimization"
STUDY = ROOT / "reports" / "walkthrough" / "optimizer" / "study.json"
TECTONIC = ROOT / "Testt" / "tools" / "tectonic.exe"
NAME = "RNGD_optimization_algorithms"


# ------------------------------------------------------------------ small LaTeX helpers
def esc(s) -> str:
    s = "" if s is None else str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("·", r"$\cdot$"), ("×", r"$\times$")]:
        s = s.replace(a, b)
    return s


def T(text: str, **kw) -> str:
    """Fill <<name>> placeholders."""
    for k, v in kw.items():
        text = text.replace(f"<<{k}>>", str(v))
    assert "<<" not in text, text[text.index("<<"):text.index("<<") + 60]
    return text


def table(headers, rows, spec, caption="", label="", size=r"\small") -> str:
    head = " & ".join(r"\textbf{" + h + "}" for h in headers)
    body = "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in rows)
    cap = (r"\caption{" + caption + "}" + (r"\label{" + label + "}" if label else "")) if caption else ""
    return (r"\begin{table}[H]\centering" + size + cap + "\n" + r"\begin{tabular}{" + spec + "}\n\\toprule\n" + head
            + r" \\ \midrule" + "\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


def fig(name, caption, width=r"\linewidth", height="0.5") -> str:
    assert (OUT / name).exists(), name
    return (r"\begin{figure}[H]\centering\includegraphics[width=" + width + ",height=" + height
            + r"\textheight,keepaspectratio]{" + name + r"}\caption{" + caption + r"}\end{figure}")


def f3(x) -> str:
    return f"{float(x):.3f}"


def f4(x) -> str:
    return f"{float(x):.4f}"


def num(x) -> str:
    x = float(x)
    return f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:,.1f}"


def sset(lst) -> str:
    return r"\{" + ", ".join(str(i) for i in lst) + r"\}" if lst else r"\{\}"


PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[margin=2.2cm,headheight=14pt]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{booktabs,tabularx,array}
\usepackage[table]{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage{float}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[most]{tcolorbox}
\usepackage[colorlinks=true,linkcolor=bpdeep,urlcolor=bp]{hyperref}
\definecolor{bp}{HTML}{1E5AA8}
\definecolor{bpdeep}{HTML}{123A6E}
\definecolor{bpwash}{HTML}{EAF1FA}
\definecolor{amber}{HTML}{9C6B0B}
\definecolor{okc}{HTML}{1E7B3A}
\definecolor{failc}{HTML}{B22222}
\definecolor{muted}{HTML}{5B6670}
\titleformat{\section}{\Large\bfseries\color{bpdeep}}{\thesection}{0.8em}{}
\titleformat{\subsection}{\large\bfseries\color{bpdeep}}{\thesubsection}{0.7em}{}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}
\setlist{nosep,leftmargin=1.5em}
\renewcommand{\arraystretch}{1.15}
\pagestyle{fancy}
\fancyhf{}
\lhead{\small\color{muted}Layout optimization algorithms}
\rhead{\small\color{muted}\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\newcommand{\code}[1]{\texttt{\small #1}}
\newtcolorbox{example}[1]{enhanced,breakable,colback=bpwash,colframe=bp,boxrule=0.5pt,arc=2pt,
  left=6pt,right=6pt,top=4pt,bottom=4pt,fonttitle=\bfseries\small,title={Worked example: #1}}
\newtcolorbox{note}{enhanced,breakable,colback=amber!8,colframe=amber,boxrule=0.5pt,arc=2pt,left=6pt,right=6pt,top=4pt,bottom=4pt}
\tikzset{box/.style={draw=bpdeep, fill=bpwash, rounded corners=2pt, align=center, font=\footnotesize, minimum height=9mm,
  inner sep=3pt}, arr/.style={-{Stealth[length=2mm]}, thick, bpdeep}}
\begin{document}
"""


def build(ex: dict, study: dict) -> str:
    d, fit, ga, sy = ex["decoder"], ex["fitness"], ex["ga"], ex["systematic"]
    ref, ff, pl, dc = ex["reference"], ex["freeform"], ex["plates"], ex["decision"]
    site, mod, cands = d["site"], d["module"], d["candidates"]
    P_, G_, seed = ex["budget"]
    sites = study["sites"]
    tot = lambda t, k: sum(s["variants"][t][k] for s in sites)  # noqa: E731
    A1 = sites[0]["variants"]["A"]
    hotel = study["hotel"]
    out = [PREAMBLE]

    # ------------------------------------------------------------------ title and summary
    out.append(T(r"""
\begin{titlepage}\centering
\vspace*{1.5cm}
{\color{bpdeep}\rule{\linewidth}{1.2pt}}\par\vspace{0.6cm}
{\huge\bfseries\color{bpdeep} How the Layout Optimization\\[0.3em] Algorithms Work}\par\vspace{0.5cm}
{\large The reference genetic algorithm, the updated genetic algorithm, the systematic search\\
and the hotel layout exploration, each with worked examples}\par\vspace{0.4cm}
{\color{bpdeep}\rule{\linewidth}{1.2pt}}\par\vspace{1.0cm}
{\large Naimul Haque}\par{\normalsize Semester project with RNGD}\par\vspace{0.3cm}{\normalsize October 2026}\par
\vspace{1.0cm}
\begin{minipage}{0.88\linewidth}\small
\textbf{What this report is.} A step-by-step explanation of the optimization algorithms in the RNGD AI Studio. It
covers only the algorithms: what they search over, how one candidate layout is built and scored, and how each search
moves from one candidate to the next. Every step is followed by an example with real numbers.

\textbf{Where the examples come from.} All examples were computed by the code itself on one case and written into this
report by a script, so they can be reproduced:
\begin{itemize}
\item site \code{fp1} of RNGD's design-optimization prototype (<<levels>> levels, <<placeable>> SF placeable per level,
<<cores>> cores) with the prototype's hotel program (<<rooms>> rooms, <<keys>> of them guest rooms);
\item search budget <<P>> $\times$ <<G>> $=$ <<PG>> layouts, random seed <<seed>>, no language model anywhere;
\item for the hotel sections, the Project Renegade Living\,+\,Lodging program (<<hkeys>> keys).
\end{itemize}

\textbf{The short version.}
\begin{enumerate}
\item No algorithm here moves rooms around directly. Each one searches over a short list of numbers (a \emph{genome}),
and a fixed procedure (the \emph{decoder}) turns that list into a complete floor plan, which is then scored (the
\emph{fitness}).
\item The reference genetic algorithm (GA) and my first attempt let corridor segments sit anywhere. Their plans left
guest rooms without windows.
\item The updated decoder only builds plans an architect would accept: straight corridors one room depth from the
facade, rooms in modular bays, a window for every guest room. On the example site it places <<rooms>> of <<rooms>>
rooms where the reference placed <<refrooms>>.
\item Two searches drive that decoder: a genetic algorithm with local search, and a deterministic search that
enumerates corridor combinations and then improves the best ones move by move. The second is the default.
\item In the hotel mode an outer loop also generates the building shape, the core position and the floor count, and a
separate ranking step (Pareto front and TOPSIS) orders the layouts found.
\end{enumerate}
\end{minipage}
\vfill
{\small\color{muted} Built <<built>> from \code{reports/optimization/examples.json}.}
\end{titlepage}
\tableofcontents
\clearpage
""", levels=site["levels"], placeable=num(site["placeable_sf"]), cores=site["cores"], rooms=site["rooms"],
                 keys=site["keys"], P=P_, G=G_, PG=P_ * G_, seed=seed, hkeys=pl["keys"], refrooms=A1["rooms_placed"],
                 built=ex["built"]))

    # ------------------------------------------------------------------ 1 problem
    prog_rows = [[esc(r["name"]), f"{num(r['width_ft'])} $\\times$ {num(r['depth_ft'])}", num(r["area_sf"]), r["instances"],
                  num(r["priority"]), r["category"], "yes" if r["window"] else "no", "yes" if r["near_core"] else "no"]
                 for r in d["program"]]
    out.append(T(r"""
\section{The problem and the running example}
\subsection{What has to be decided}
A site is a stack of levels. Each level is a grid of 1\,ft cells, and every cell is one of: \emph{placeable},
\emph{core} (stairs and elevators), \emph{structure} or \emph{outside}. A program lists the rooms to place: a size,
a count, a priority and a few flags (needs a window, should be near a core, one per floor).

A layout answers two questions:
\begin{itemize}
\item \textbf{Where is the corridor?} A set of placeable cells, the same on every level so that the floors stack.
\item \textbf{Where is each room?} A rectangle on one level, opening onto the corridor, not overlapping anything.
\end{itemize}
The layout should place as many rooms as possible, give guest rooms a window, keep the corridor short, satisfy the
egress rules, and so on. Section~\ref{sec:fitness} turns that list into one number, the fitness $F$.

\subsection{The running example}
The example used throughout is site \code{fp1}: an L-shaped plate of <<rows>> $\times$ <<cols>> cells, <<levels>>
levels (<<lnames>>), <<placeable>> SF placeable per level and <<cores>> cores. The program is
\code{QA/rooms\_hotel.json}:
<<progtable>>
The janitor's closet is one per floor, so the program has <<rooms>> room instances in total, <<keys>> of them keys.

<<beforeafter>>

\subsection{Why the search does not move rooms directly}
The number of ways to put <<rooms>> rectangles on a <<rows>> $\times$ <<cols>> grid is astronomically large, and almost
all of them are useless: rooms overlap, have no door on a corridor, or sit in the middle of the plate. All the
algorithms in this report therefore share one structure (Figure~\ref{fig:loop}):
\begin{enumerate}
\item the search proposes a \textbf{genome}, a short vector of numbers between 0 and 1;
\item a \textbf{decoder} turns the genome into a complete layout by fixed rules;
\item the \textbf{fitness} scores that layout;
\item the search uses the score to propose the next genome.
\end{enumerate}
The algorithms differ in two places only: what the decoder does (Sections~\ref{sec:ref}, \ref{sec:free} and
\ref{sec:decoder}), and how the search proposes the next genome (Sections~\ref{sec:ga} and \ref{sec:sys}).

\begin{figure}[H]\centering
\begin{tikzpicture}[node distance=9mm]
\node[box, text width=28mm] (s) {\textbf{Search}\\GA or systematic};
\node[box, text width=28mm, right=of s] (g) {\textbf{Genome}\\<<glen>> numbers in $[0,1]$};
\node[box, text width=28mm, right=of g] (dcd) {\textbf{Decoder}\\genome $\to$ layout};
\node[box, text width=28mm, right=of dcd] (f) {\textbf{Fitness}\\layout $\to$ $F$};
\draw[arr] (s) -- (g); \draw[arr] (g) -- (dcd); \draw[arr] (dcd) -- (f);
\draw[arr] (f.south) -- ++(0,-6mm) -| node[pos=0.25, below, font=\scriptsize]{score of this genome} (s.south);
\end{tikzpicture}
\caption{The loop shared by all the algorithms. For the running example the genome has <<glen>> numbers.}\label{fig:loop}
\end{figure}

<<algtable>>
""", rows=site["rows"], cols=site["cols"], levels=site["levels"], lnames=", ".join(site["level_names"]),
                 placeable=num(site["placeable_sf"]), cores=site["cores"], rooms=site["rooms"], keys=site["keys"],
                 glen=site["genome_len"],
                 progtable=table(["Room type", "Size, ft", "Area, SF", "Instances", "Priority", "Category", "Window",
                                  "Near core"], prog_rows, "@{}l c r r r l c c@{}",
                                 "The example program. Category, window and near-core flags are inferred from the room name.",
                                 "tab:program"),
                 beforeafter=fig("before_after.png",
                                 "The running example, Level 2. Left: the layout delivered by the reference GA. Right: the layout "
                                 "found by the systematic search with the updated decoder. A red dashed outline marks a guest room "
                                 "without a window; blue lines are window walls.", height="0.42"),
                 algtable=table(["", "Algorithm", "Decoder", "Search", "Same result every run"],
                                [["A", "Reference GA (prototype)", "free corridor genes", "genetic algorithm, 500 generations",
                                  "not recorded"],
                                 ["B", "First attempt", "free corridor segments, greedy packer", "memetic GA", "yes (seeded)"],
                                 ["C", "Updated GA", "architectural decoder", "memetic GA", "yes (seeded)"],
                                 ["D", "Systematic search (default)", "architectural decoder",
                                  "enumeration, pattern search, iterated local search", "yes"],
                                 ["E", "Hotel exploration", "architectural decoder on generated plates",
                                  "enumeration of plates, pattern search", "yes"]],
                                "@{}l p{3.3cm} p{4.3cm} p{4.6cm} p{2.3cm}@{}", "The algorithms covered in this report.", "tab:algs")))

    # ------------------------------------------------------------------ 2 reference GA
    r1 = ref["sites"][0]
    ref_rows = [[s["site"], s["genes"], s["generations"], num(s["start_unplaced"]), num(s["end_unplaced"]),
                 s["last_change_gen"], f"{s['rooms_placed']}/{s['rooms_total']}", s["seed"]] for s in ref["sites"]]
    out.append(T(r"""
\section{Algorithm A: the reference genetic algorithm}\label{sec:ref}
\begin{note}
I have the reference GA's \emph{result files} (\code{QA/floorplan1--3.json}), not its source code. This section
describes what those files record and what can be measured from them. It does not guess at internals.
\end{note}

\subsection{What a genetic algorithm does}
A genetic algorithm keeps a \emph{population} of candidate genomes. In every \emph{generation} it
\begin{enumerate}
\item scores every candidate,
\item \emph{selects} better candidates as parents,
\item makes children by \emph{crossover} (taking some genes from one parent and some from the other),
\item \emph{mutates} the children (small random changes),
\item and replaces the population with the children, usually keeping the best old candidates (\emph{elitism}).
\end{enumerate}
Good gene combinations survive and spread; mutation keeps introducing new ones.

\subsection{What the result files record}
\begin{itemize}
\item \textbf{A corridor genome.} A list of corridor genes, each three integers. For \code{fp1} there are
<<genes>> genes; the first three are <<head>>. The corridor they produce is stored as a list of cells
(corridor width <<cw>>\,ft, the same on all levels).
\item \textbf{Room placements.} For each placed room: level, anchor cell, the side it faces, frontage and depth.
\item \textbf{A history.} One entry per generation with two numbers: the rooms not yet placed and a score. All three
runs have 500 entries.
\item \textbf{Settings.} A seed, an adjacency weight (<<adjw>>) and a balance weight (<<balw>>).
\end{itemize}
<<reftable>>
<<refhist>>

\begin{example}{reading the history of \code{fp1}}
The run starts with <<u0>> of <<rt>> rooms unplaced and ends with <<u1>> unplaced, so <<placed>> rooms are placed. The
last improvement happens in generation <<last>> of 500: for the remaining <<rest>> generations the best plan does not
change.
\end{example}

\subsection{What goes wrong, measured}
I rebuilt each delivered layout cell by cell and scored it with the fitness of Section~\ref{sec:fitness}, the same one
used for every other algorithm. For \code{fp1}:
\begin{itemize}
\item <<placed>> of <<rt>> rooms are placed;
\item <<nowin>> guest rooms have no window, although the file reports an exterior fraction of <<extf>>
(the two measures differ: mine counts a window only if at least half of the room's back wall lies within 6\,ft of
the facade);
\item <<corr>>\% of the placeable area is corridor and <<unused>>\% is neither room nor corridor;
\item its fitness is $F = <<fa>>$.
\end{itemize}
The cause is the representation. A corridor gene can put a corridor segment anywhere on the plate, and rooms are then
attached wherever they touch a corridor. Nothing in the genome knows where the facade is, so whether a room gets a
window is left to chance, and the gaps between rooms are too irregular to use. Longer runs do not fix this:
the search is looking in a space where almost every candidate is a poor plan.
""", genes=r1["genes"], head=", ".join("(" + ", ".join(str(x) for x in g) + ")" for g in r1["genome_head"]),
                 cw=num(r1["corridor_width_ft"]), adjw=r1["adjacency_weight"], balw=r1["balance_weight"],
                 reftable=table(["Site", "Genes", "Generations", "Unplaced, start", "Unplaced, end",
                                 "Last improvement", "Placed", "Seed"], ref_rows, "@{}l r r r r r r r@{}",
                                "The three reference runs, as recorded in their result files.", "tab:ref"),
                 refhist=fig("reference_history.png", "Reference GA: rooms not placed in each of the 500 generations "
                                                      "(from the history stored in the result files).", width=r"0.7\linewidth",
                             height="0.28"),
                 u0=num(r1["start_unplaced"]), u1=num(r1["end_unplaced"]), rt=r1["rooms_total"], placed=r1["rooms_placed"],
                 last=r1["last_change_gen"], rest=500 - r1["last_change_gen"], nowin=A1["windowless_guest_rooms"],
                 extf=r1["reported_exterior_fraction"], corr=f"{100 * A1['corridor_ratio']:.0f}",
                 unused=f"{100 * A1['unused_share']:.0f}", fa=f3(A1["fitness"])))

    # ------------------------------------------------------------------ 3 free-form
    seg = ff["segments"][0]
    B1 = sites[0]["variants"].get("B", {})
    out.append(T(r"""
\section{Algorithm B: my first attempt (free-form corridors)}\label{sec:free}
My first optimizer kept the reference idea and improved the parts around it. It is described here because it shows
that better scoring and a better search were not enough.

\subsection{Genome and decoder}
The genome has <<nseg>> corridor segments of four genes each, then one \emph{order key} per room:
\begin{itemize}
\item \textbf{Segment genes} $(u, v, o, \ell)$. The segment starts at the placeable cell nearest to column $u \cdot
(\text{cols}-1)$ and row $v \cdot (\text{rows}-1)$. It is horizontal if $o < 0.5$, otherwise vertical. It grows both
ways by up to $\ell \cdot \text{span}/2$ cells while the corridor width stays placeable. A segment with $\ell < 0.12$
is switched off.
\item \textbf{Repair.} All corridor pieces and all cores are joined into one network by least-cost paths.
\item \textbf{Order keys.} Rooms are placed in decreasing order of $0.06 \cdot \text{priority} + \text{key}$.
\item \textbf{Greedy packing.} Each room in turn is put at the best free position that touches the corridor. Positions
are scored for a window, a corner, distance to a core and how tightly the room sits against its neighbours.
\end{itemize}

\begin{example}{decoding one segment}
One of the seed genomes has a single active segment $(u, v, o, \ell) = (<<u>>, <<v>>, <<o>>, <<l>>)$ on the
<<rows>> $\times$ <<cols>> grid:
\begin{itemize}
\item start near column $<<u>> \times <<c1>> \approx <<uc>>$ and row $<<v>> \times <<r1>> \approx <<vr>>$;
\item $o = <<o>> < 0.5$, so it runs horizontally;
\item reach $= <<l>> \times <<cols>> / 2 = <<reach>>$ cells each way, cut short where the plate ends.
\end{itemize}
Packing the <<rt>> rooms around this corridor places <<rooms>> of them, <<nowin>> guest rooms without a window, for a
fitness of <<f>> (Figure~\ref{fig:free}).
\end{example}
<<freefig>>

\subsection{Result and lesson}
With a memetic GA around this decoder (the same search as Section~\ref{sec:ga}) and the full budget, the example site
reached <<brooms>> of <<rt>> rooms with <<bnowin>> windowless guest rooms and $F = <<bf>>$, in <<bsecs>>\,s. Over the three
sites it placed <<btot>> rooms against the reference's <<atot>>, with <<bnw>> windowless guest rooms against
<<anw>>. The scoring was better and the search was seeded and repeatable, but the plans still had the same faults.
The representation had to change.
""", nseg=ff["n_seg"], u=seg[0], v=seg[1], o=seg[2], l=seg[3], rows=site["rows"], cols=site["cols"], c1=site["cols"] - 1,
                 r1=site["rows"] - 1, uc=f"{seg[0] * (site['cols'] - 1):.0f}", vr=f"{seg[1] * (site['rows'] - 1):.0f}",
                 reach=f"{seg[3] * site['cols'] / 2:.0f}", rt=ff["rooms_total"], rooms=ff["rooms"], nowin=ff["windowless"],
                 f=f3(ff["fitness"]),
                 freefig=fig("freeform_seed.png", r"The free-form decoder on one seed genome (Level 2).\label{fig:free}",
                             width=r"0.42\linewidth", height="0.36"),
                 brooms=B1.get("rooms_placed", "-"), bnowin=B1.get("windowless_guest_rooms", "-"),
                 bf=f3(B1.get("fitness", 0)), bsecs=f"{B1.get('seconds', 0):.0f}", btot=tot("B", "rooms_placed"),
                 atot=tot("A", "rooms_placed"), bnw=tot("B", "windowless_guest_rooms"), anw=tot("A", "windowless_guest_rooms")))

    # ------------------------------------------------------------------ 4 architectural decoder
    fronts = "; ".join(f"{esc(n)} {num(f)}\\,ft ($\\times${c})" for n, f, c in mod["fronts"])
    depths = ", ".join(f"{num(x)}" for _, x in mod["depths"])
    cand_rows = [[c["i"], "horizontal (row)" if c["axis"] == "H" else "vertical (column)", c["coord"],
                  f"{c['lo']}--{c['hi']}", c["length"], num(c["score"]), f"{c['per_ft']:.2f}"] for c in cands]
    gene_rows = [[g["i"], f"{g['trim_lo']:.2f}", f"{g['trim_hi']:.2f}", f"{g['shift']:.2f}", f"{g['active']:.2f}",
                  "on" if g["on"] else "off"] for g in d["genes"]]
    m = len(cands)
    trimmed = next((g for g in d["genes"] if g["on"] and (g["trim_lo"] > 0 or g["trim_hi"] > 0)), None)
    if trimmed:
        c = cands[trimmed["i"]]
        n = c["length"]
        cut_lo, cut_hi = int(trimmed["trim_lo"] * 0.35 * n), int(trimmed["trim_hi"] * 0.35 * n)
        trim_txt = T(r"""Spine <<i>> runs from <<lo>> to <<hi>> ($n = <<n>>$\,ft) and has trim genes
$(<<tl>>, <<th>>)$. The low end moves in by $\lfloor <<tl>> \times 0.35 \times <<n>> \rfloor = <<cl>>$\,ft and the
high end by $\lfloor <<th>> \times 0.35 \times <<n>> \rfloor = <<ch>>$\,ft, so it becomes <<nlo>>--<<nhi>>. A trim
gene can remove at most 35\% of a spine from each end.""", i=c["i"], lo=c["lo"], hi=c["hi"], n=n,
                     tl=f"{trimmed['trim_lo']:.2f}", th=f"{trimmed['trim_hi']:.2f}", cl=cut_lo, ch=cut_hi,
                     nlo=c["lo"] + cut_lo, nhi=c["hi"] - cut_hi)
    else:
        trim_txt = (r"No active spine is trimmed in this genome. A trim gene $t$ moves an end in by "
                    r"$\lfloor 0.35\,t\,n \rfloor$\,ft for a spine of length $n$.")
    kept = "; ".join(f"{'row' if s['axis'] == 'H' else 'column'} {s['coord']}, {s['lo']}--{s['hi']} ({s['length']}\\,ft)"
                     for s in d["spines"])
    bay_items = []
    M = mod["M"]
    for b in d["bays"]["examples"]:
        L_ = b["length_ft"]
        n_full = int(L_ // M)
        rho = L_ - n_full * M
        if rho == 0:
            rule = "no remainder"
        elif rho < 0.4 * M:
            rule = (f"remainder ${num(rho)} < 0.4 \\times {num(M)} = {num(0.4 * M)}$\\,ft, so the last bay absorbs it and "
                    f"is {num(M + rho)}\\,ft wide")
        else:
            rule = (f"remainder ${num(rho)} \\ge {num(0.4 * M)}$\\,ft, so it becomes a narrow bay of its own")
        bay_items.append(f"\\item A run of {num(L_)}\\,ft facing {b['facing']}: $\\lfloor {num(L_)}/{num(M)} \\rfloor = "
                         f"{n_full}$ full bays, {rule}. Bay widths: {', '.join(num(x) for x in b['widths'])}\\,ft; "
                         f"depth {num(b['depth_ft'])}\\,ft.")
    k_rows = [[esc(r["name"]), num(r["area"]), f"{r['ratio']:.2f}", r["k_area"], num(r["fn"]),
               r["k_front"] if r["k_front"] else "--", r"\textbf{" + str(r["k"]) + "}", f"{r['order_value']:.2f}"]
              for r in d["assign_rows"]]
    sig_rows = [[esc(s["room"]), s["k"], s["runs_total"], s["runs_allowed"],
                 f3(s["best"][0]["score"]) if s["best"] else "--", f3(s["worst"][0]["score"]) if s["worst"] else "--"]
                for s in d["sigma"]]
    sg = next((s for s in d["sigma"] if s["room"] == "ADA suite" and s["best"]), d["sigma"][0])
    strict = min((s for s in d["sigma"] if s["is_key"]), key=lambda s: s["runs_allowed"] / max(1, s["runs_total"]))
    bk_rows = []
    for lab, t in (("best", sg["best"][0]), ("worst allowed", sg["worst"][0])):
        bk_rows.append([lab, f"{num(t['front_ft'])} $\\times$ {num(t['depth_ft'])}", f"{t['over']:.2f}", f3(t["oversize"] + 0.0),
                        f"{t['window']:+.3f}", f"{num(t['d0_ft'])}", f"{t['core']:+.3f}", f"{t['pull']:+.3f}",
                        r"\textbf{" + f3(t["score"]) + "}"])
    W = d["pack_weights"]
    aft = d["after"]
    out.append(T(r"""
\section{The architectural decoder (used by C, D and E)}\label{sec:decoder}
The updated decoder builds a floor plan the way a hotel floor is normally laid out. The search no longer decides
\emph{where corridor cells are}. It decides \emph{which of a few sensible corridors exist} and \emph{in which order
rooms claim their places}. This section follows one genome through all seven steps: the genome of the best layout the
systematic search found for the running example.

\subsection{Step 1: the bay module}
All rooms are placed on a grid of \emph{bays} along the corridor. One bay is one standard guest-room frontage wide.
\begin{itemize}
\item The module $M$ is the \textbf{weighted median} of the guest-room frontages, weighted by room count (a suite
counts as half its frontage, since it takes two bays).
\item $D$ is the median guest-room depth; $D_{\min} = \lfloor 0.92\, d_{\min} \rfloor$ is the shallowest strip a room may
use; $D_{\max} = d_{\max} + 2$ the deepest.
\end{itemize}
\begin{example}{module of the example program}
Guest-room frontages (the shorter side of each room): <<fronts>>. Sorted by frontage, the middle one of the <<keys1>> guest
rooms is a King, so $M = <<M>>$\,ft. Depths are <<depths>>\,ft, so $D = <<D>>$\,ft, $D_{\min} = \lfloor 0.92 \times <<dminraw>> \rfloor =
<<Dmin>>$\,ft and $D_{\max} = <<Dmax>>$\,ft. The corridor is $w = <<w>>$\,ft wide. One bay is therefore
$<<M>> \times <<D>> = <<bayarea>>$\,SF.
\end{example}

\subsection{Step 2: candidate spines}
A \emph{spine} is a straight corridor line: a row (horizontal corridor) or a column (vertical corridor). Every row and
column $k$ of the plate that can hold a corridor band gets a score
\[ s(k) = \sum_{x \in \text{band}(k)} \big[\, g_a(x) + g_b(x) + 1.5\,n_a(x) + 1.5\,n_b(x) + 0.5\,r_a(x) + 0.5\,r_b(x) \,\big]. \]
At each foot $x$ along the line, and for each of its two sides $a$ and $b$:
\begin{itemize}
\item $g = 1$ if a room at least $D_{\min}$ deep fits on that side;
\item $n = 1$ if, in addition, the facade is about one room depth away (at most $D + 2 +$ slack), so a room there has
a window;
\item $r = 1$ if the facade is within the deepest room's depth.
\end{itemize}
So one foot of corridor with window rooms on both sides scores $2 \times (1 + 1.5 + 0.5) = 6$, with window rooms on one
side only $3$, and with deep windowless space on both sides $2$.

In each direction the best lines are taken in order of score, skipping any line closer than $D_{\min} + w$ to one
already chosen (two corridors that close would leave no room between them). The ten best pieces overall become the
candidates. A plate may have fewer than ten.

\begin{example}{candidate spines of \code{fp1}}
The plate has $m = <<m>>$ candidates (Figure~\ref{fig:scores}, Table~\ref{tab:cands}). Spine 0 is the column one room
depth in from the long west facade: it scores <<s0>> over <<l0>>\,ft, or <<p0>> per foot, close to the maximum of 6.
The lowest-ranked spine scores <<plast>> per foot.
\end{example}
<<candtable>>
<<scorefig>>

\subsection{Step 3: from genome to active spines}
For <<rooms>> rooms the genome has $10 \times 4 + 4 + <<rooms>> = <<glen>>$ numbers:
\begin{itemize}
\item \textbf{ten spine blocks} of four genes: trim at the low end, trim at the high end, shift, and an on/off gene
(the spine is on if that gene is at least 0.12);
\item \textbf{four connector genes}, which choose the elbow direction of L-shaped connectors;
\item \textbf{one order key per room}.
\end{itemize}
Blocks for candidates the plate does not have are simply ignored (here blocks <<m>> to 9).

A shift gene $h$ moves the line sideways by $\operatorname{round}(4(h - 0.5))$\,ft, so by at most 2\,ft. If the shifted
line would hit an obstacle, the scored position is kept. A trimmed spine shorter than $2M$ is dropped.

\begin{example}{the spine genes of the example genome}
<<genetable>>
<<trimtxt>> All shift genes are 0.5, so no line moves.
\end{example}

\subsection{Step 4: trimming to the rooms and connecting}
\begin{enumerate}
\item A first pass builds bays along every active spine (Step 5) and \textbf{cuts each spine back to its own bays}, so
no corridor runs past the last possible room. A spine that ends up with no bays of its own is dropped.
\item The remaining spines and all cores are joined into one network. The nearest unconnected piece is linked by an
\textbf{L-shaped path}: one horizontal and one vertical leg, never a diagonal. A connector gene picks which leg comes
first. If both L-shapes are blocked, a rectilinear least-cost path is used.
\item Pieces that still cannot be reached are removed.
\end{enumerate}
\begin{example}{corridor of the example genome}
All <<m>> spines are switched on, but after the first pass the corridor consists of: <<kept>>. The other spines
received no bays of their own and were dropped. These spines occupy <<spcells>> corridor cells and the connectors add
<<concells>>: here the two spines cross and both cores lie on them, so no connector is needed. The corridor reaches
<<reached>> of <<ncores>> cores.
\end{example}

\subsection{Step 5: bays}
Each side of each straight corridor run is cut into bays of width $M$:
\begin{itemize}
\item a run of length $\ell$ gives $n = \lfloor \ell / M \rfloor$ full bays;
\item a remainder $\rho = \ell - nM$ smaller than $0.4M$ is absorbed by the last bay (a wider end bay, which suits a
suite); a larger remainder becomes a narrow bay of its own (for a service room);
\item a bay's depth is the free depth behind it, capped at $D_{\max}$;
\item a bay \textbf{has a window} if its back wall is on the facade, and is a \textbf{corner} bay if a side wall is too.
\end{itemize}
Bays are built in \textbf{two passes over all corridors}: first every bay that has a window, then windowless bays in
whatever space is left. Without this, the windowless inner side of one corridor could take the strip of floor that
another corridor needs for its window rooms.

Every level gets the same bays, which is what makes the floors stack.

\begin{example}{bays of the example layout}
There are <<bays>> bays per floor: <<bwin>> with a window, <<bnowin>> without, <<bcorner>> on a corner. Some of the runs:
\begin{itemize}
<<bayitems>>
\end{itemize}
\end{example}

\subsection{Step 6: assigning rooms to bays}
\textbf{Order.} Rooms are taken in decreasing order of
\[ 0.06 \times \text{priority} + \text{order key}. \]
With all keys equal (0.5), this is pure priority order. The search changes the order by changing keys.

\textbf{Bays per room.} A room of nominal area $\bar a$ and frontage $f$ takes
\[ k = \max\!\Big(\operatorname{round}\frac{\bar a}{M D},\ \Big\lceil \frac{0.85 f}{M} \Big\rceil\Big) \]
consecutive bays (the second term applies to guest rooms and makes sure a wide room gets its width).
<<ktable>>

\textbf{Hard rules.} A run of bays is not allowed for a guest room unless
\begin{itemize}
\item every bay in the run has a window, and
\item the run gives at least 85\% of the room's nominal frontage and 85\% of its nominal depth.
\end{itemize}

\textbf{Score.} Among the allowed runs on the allowed levels the room takes the one with the highest score
\begin{align*} \sigma ={}& W_{\text{ext}}\,[\text{window}] - W_{\text{nw}}\,[\text{no window}] + W_{\text{corner}}\,[\text{corner}]
+ W_{\text{core}}\Big(1 - \tfrac{\min(\delta,150)}{150}\Big) - W_{\text{pull}}\,\tfrac{\min(\delta,300)}{300} \\
&- 1.2\max(0, o - 1.2) + W_{\text{adj}}\,A + W_{\text{stack}}\,\tfrac{\min(S,2)}{2} + W_{\text{grp}}\,[\text{same type next door}] + b_\ell ,
\end{align*}
where $\delta$ is the corridor distance from the door to the nearest core, $o$ the ratio of the bays' area to the
room's nominal area (oversized rooms waste floor), $A$ the adjacency credit, $S$ the number of other floors that have
the same room type in the same bay (stacking), and $b_\ell$ a small bonus for a level that has fewer of this room type
than average (it spreads rooms over the floors). The window and core terms apply only to rooms that need a window or
want to be near a core. A room that does not need a window loses 0.4 for taking a window bay. The weights are
$W_{\text{ext}} = <<Wext>>$, $W_{\text{nw}} = <<Wnw>>$, $W_{\text{corner}} = <<Wcorner>>$, $W_{\text{core}} = <<Wcore>>$,
$W_{\text{pull}} = <<Wpull>>$, $W_{\text{adj}} = <<Wadj>>$, $W_{\text{stack}} = <<Wstack>>$, $W_{\text{grp}} = <<Wgrp>>$.

\begin{example}{how many runs are allowed, and how they score}
Table~\ref{tab:runs} counts, for each room type, the runs of $k$ consecutive bays on one floor and how many pass the
hard rules. The <<strictname>> is the strictest: it needs <<strictfront>>\,ft of frontage and
$0.85 \times$ its depth $= <<strictdepth>>$\,ft, and only <<strictallowed>> of <<stricttotal>> runs are deep enough.
<<sigtable>>
The terms of the score for the <<sgname>>, for its best and its worst allowed run (on an empty floor, before any
room is placed):
<<bktable>>
The worst run is wider than the room needs ($o = <<oworst>>$, so it loses $1.2 \times (<<oworst>> - 1.2) = <<oloss>>$)
and its door is farther from the core.
\end{example}

\subsection{Step 7: cutting stubs and measuring distances}
After the rooms are assigned, each spine is cut back to what it actually serves: the doors of the placed rooms and its
junctions with other corridors and cores. This removes corridor stubs that lead nowhere.

Then two distances are measured along the corridor:
\begin{itemize}
\item \textbf{Exit travel} $T$: for each room, the distance from its door to the nearest core plus the room depth; the
largest value counts.
\item \textbf{Dead end} $E$: the longest distance from any corridor cell to the nearest exit door or to the
through-route between two exits. It measures how far someone can walk with only one way out.
\end{itemize}
\begin{example}{pruning the example corridor}
The corridor shrinks from <<cbefore>> to <<cafter>> cells, and the dead end falls from <<debefore>>\,ft to
<<deafter>>\,ft (the limit is 50\,ft). The final layout places <<prooms>> rooms and leaves <<punplaced>> unplaced.
\end{example}
<<stepsfig>>

\subsection{The decoder as pseudo-code}
\begin{algorithm}[H]
\caption{Architectural decoder}
\begin{algorithmic}[1]
\Require site, program, genome $g$
\State compute $M$, $D$, $D_{\min}$, $D_{\max}$ and the candidate spines \Comment{once per site and program}
\State $S \gets$ candidates whose on/off gene $\ge 0.12$, trimmed and shifted by their genes
\State build bays along $S$; cut each spine to its own bays; drop spines with none
\State corridor $\gets S$ joined to each other and to every core by L-shaped connectors
\State bays $\gets$ window bays of all corridor sides, then windowless bays in the space left
\For{each room, in decreasing order of $0.06 \cdot \text{priority} + \text{key}$}
  \State $k \gets$ bays this room needs
  \State take the allowed run of $k$ free bays with the highest score $\sigma$; if there is none, the room is unplaced
\EndFor
\State cut each spine back to the doors and junctions it serves
\State measure exit travel and the dead end along the corridor
\State \Return layout
\end{algorithmic}
\end{algorithm}
Every genome decodes to a valid plan: corridors are straight, rooms are perpendicular to them, nothing overlaps and
every room has a door on the corridor. The search can no longer produce nonsense; it can only choose between plans.
""", fronts=fronts, keys1=site["keys"], M=num(mod["M"]), depths=depths, D=num(mod["Dnom"]),
                 dminraw=num(min(x for _, x in mod["depths"])), Dmin=num(mod["Dmin"]), Dmax=num(mod["Dmax"]), w=num(mod["w"]),
                 bayarea=num(d["bay_area"]), m=m, s0=num(cands[0]["score"]), l0=cands[0]["length"],
                 p0=f"{cands[0]['per_ft']:.2f}", plast=f"{cands[-1]['per_ft']:.2f}",
                 candtable=table(["Spine", "Direction", "Row / column", "Extent", "Length, ft", "Score $s$", "Score per ft"],
                                 cand_rows, "@{}c l r c r r r@{}", r"Candidate spines of \code{fp1}, best first.", "tab:cands"),
                 scorefig=fig("line_scores.png", r"The line score $s(k)$ of every row and every column of \code{fp1}. Dashed "
                                                 r"lines mark the candidates chosen (numbered as in Table~\ref{tab:cands})."
                                                 r"\label{fig:scores}", height="0.24"),
                 rooms=site["rooms"], glen=site["genome_len"],
                 genetable=table(["Spine", "Trim low", "Trim high", "Shift", "On/off gene", "State"], gene_rows,
                                 "@{}c r r r r l@{}", "", "", size=r"\small").replace(r"\begin{table}[H]", r"\begin{center}")
                 .replace(r"\end{table}", r"\end{center}"),
                 trimtxt=trim_txt, kept=kept, spcells=num(d["corridor"]["spine_cells"]),
                 concells=num(d["corridor"]["connector_cells"]), reached=d["corridor"]["cores_reached"], ncores=site["cores"],
                 bays=d["bays"]["per_floor"], bwin=d["bays"]["window"], bnowin=d["bays"]["windowless"],
                 bcorner=d["bays"]["corner"], bayitems="\n".join(bay_items),
                 ktable=table(["Room type", "Area $\\bar a$", "$\\bar a / (MD)$", "by area", "Frontage $f$", "by frontage",
                               "$k$", "Order value"], k_rows, "@{}l r r r r r r r@{}",
                              "Bays per room type, and the order value with all keys at 0.5 (the janitor's closets go first, "
                              "the Kings last).", "tab:k"),
                 Wext=W["exterior"], Wnw=W["no_window"], Wcorner=W["corner"], Wcore=W["core"], Wpull=W["core_pull"],
                 Wadj=W["adjacency"], Wstack=W["stack"], Wgrp=W["group"],
                 strictname=esc(strict["room"]), strictfront=num(strict["need_front"]), strictdepth=num(strict["need_depth"]),
                 strictallowed=strict["runs_allowed"], stricttotal=strict["runs_total"],
                 sigtable=table(["Room type", "$k$", "Runs on a floor", "Allowed", "Best $\\sigma$", "Worst $\\sigma$"],
                                sig_rows, "@{}l r r r r r@{}", "Runs of bays available to each room type on an empty floor.",
                                "tab:runs"),
                 sgname=esc(sg["room"]),
                 bktable=table(["Run", "Front $\\times$ depth, ft", "$o$", "Oversize", "Window", "$\\delta$, ft", "Core",
                                "Pull", "$\\sigma$"], bk_rows, "@{}l c r r r r r r r@{}", "", "").replace(
                     r"\begin{table}[H]", r"\begin{center}").replace(r"\end{table}", r"\end{center}"),
                 oworst=f"{sg['worst'][0]['over']:.2f}", oloss=f"{1.2 * max(0.0, sg['worst'][0]['over'] - 1.2):.2f}",
                 cbefore=num(aft["corridor_cells_before"]), cafter=num(aft["corridor_cells_after"]),
                 debefore=num(d["corridor"]["dead_end_before_prune_ft"]), deafter=num(aft["dead_end_ft"]),
                 prooms=aft["rooms"], punplaced=aft["unplaced"],
                 stepsfig=fig("decoder_steps.png",
                              "The decoder on the example genome. 1: candidate spines (solid: switched on in this genome). "
                              "2: the spines that keep bays of their own, trimmed (blue), with connectors (orange, none needed "
                              "here). 3: bays (light blue: with a window; hatched: corner). 4: rooms assigned.", height="0.86")))

    # ------------------------------------------------------------------ 5 fitness
    terms_a = ref["fp1_eval"]["terms"]
    sym = {"placement": "place", "daylight": "day", "efficiency": "eff", "egress": "egress", "ada": "ada",
           "core_access": "core", "adjacency": "adj", "corner": "corner", "balance": "bal", "compactness": "comp"}
    fit_rows = [[f"$f_{{\\text{{{sym[r['name']]}}}}}$", esc(r["text"]), f"{r['weight']:.2f}", f4(r["value"]), f4(r["product"]),
                 f4(terms_a[r["name"]])] for r in fit["rows"]]
    fit_rows.append([r"\midrule \textbf{Sum}", "", f"{fit['sum_w']:.2f}", "", r"\textbf{" + f4(fit["sum_products"]) + "}",
                     r"\textbf{" + f4(ref["fp1_eval"]["fitness"]) + "}"])
    lim = fit["limits"]
    E_bad = 74.0
    eg_bad = 1 - 0.5 * min(1, max(0, E_bad - lim["dead_end_ft"]) / lim["dead_end_ft"])
    out.append(T(r"""
\section{The fitness: one number for a layout}\label{sec:fitness}
The fitness contains no language model. It is arithmetic on the layout's geometry, so the same layout always gets the
same score.

\subsection{Ten objectives}
Each objective $f_k$ is scaled to lie between 0 (worst) and 1 (best):
\begingroup\footnotesize
\begin{align*}
f_{\text{place}} &= \frac{\sum_{i\ \text{placed}} p_i \min(a_i, \bar a_i)}{\sum_{i} p_i \bar a_i}, &
f_{\text{day}} &= \frac{\#\{\text{rooms needing a window that have one}\}}{\#\{\text{rooms needing a window}\}}, \\
f_{\text{eff}} &= \operatorname{clip}\!\Big(1 - \frac{\rho}{0.30}\Big), &
f_{\text{egress}} &= 1 - \tfrac12 \min\!\Big(1, \frac{(T - T_{\max})^+}{T_{\max}}\Big) - \tfrac12 \min\!\Big(1, \frac{(E - E_{\max})^+}{E_{\max}}\Big), \\
f_{\text{ada}} &= \min\!\Big(1, \frac{n_{\text{ADA}}}{n_{224.2}}\Big)\cdot\min\!\Big(1, \frac{\text{floors with ADA}}{\text{target}}\Big), &
f_{\text{core}} &= \operatorname{mean}_i \Big(1 - \frac{\min(\delta_i, 120)}{120}\Big), \\
f_{\text{adj}} &= \frac{\sum_j \omega_j \max\big(0,\ 1 - (\Delta_j - m_j)^+/m_j\big)}{\sum_j \omega_j}, &
f_{\text{corner}} &= \frac{\#\{\text{corner rooms on a corner}\}}{\#\{\text{corner rooms}\}}, \\
f_{\text{bal}} &= \operatorname{clip}\!\Big(1 - \frac{\operatorname{sd}(K_\ell)}{\operatorname{mean}(K_\ell)}\Big), &
f_{\text{comp}} &= \operatorname{clip}\!\Big(\frac{2 S_{\text{shared}}}{0.6\, P_{\text{rooms}}}\Big).
\end{align*}
\endgroup
Here $p_i$ is a room's priority, $a_i$ its drawn area and $\bar a_i$ its nominal area; $\rho$ is the corridor's share of
the placeable area; $T$ and $E$ are the exit travel and dead end of Section~\ref{sec:decoder}, with limits
$T_{\max} = <<Tmax>>$\,ft and $E_{\max} = <<Emax>>$\,ft read from RNGD's IBC rule file; $n_{224.2}$ is the number of
accessible rooms required by ADA Table 224.2; $\delta_i$ is the door-to-core distance of a room that should be near a
core; $\Delta_j$ is the distance between the two rooms of a required adjacency and $m_j$ the distance allowed;
$K_\ell$ is the number of keys on guest floor $\ell$; $S_{\text{shared}}$ is the total length of walls shared by two
rooms and $P_{\text{rooms}}$ the total room perimeter. $(x)^+$ means $\max(0, x)$, and clip limits a value to $[0, 1]$.
An objective with nothing to measure (no required adjacency, no corner room) is 1.

\subsection{The weighted sum and the exit penalty}
\[ F = \frac{\sum_k w_k f_k}{\sum_k w_k} \; - \; \tfrac12 \max\!\big(0,\ \min(n_c, 2) - c\big), \]
where $n_c$ is the number of cores on the plate and $c$ the number the corridor reaches. A layout whose corridor
misses a required exit loses half the scale, so it can never beat one that reaches both.

\begin{example}{fitness of the example layout, term by term}
<<fittable>>
\begin{itemize}
\item \textbf{Placement.} All <<rp>> of <<rt>> rooms are placed, yet $f_{\text{place}} = <<fplace>>$, not 1. Credit is
capped at the nominal area, and a few rooms are drawn slightly smaller than nominal (the hard rule allows down to
85\%).
\item \textbf{Efficiency.} The corridor is <<csf>>\,SF of <<psf>>\,SF placeable, $\rho = <<rho>>$, so
$f_{\text{eff}} = 1 - <<rho>>/0.30 = <<feff>>$.
\item \textbf{Egress.} $T = <<T>>$\,ft $\le <<Tmax>>$ and $E = <<E>>$\,ft $\le <<Emax>>$, so nothing is subtracted:
$f_{\text{egress}} = 1$. If the dead end were <<Ebad>>\,ft instead, the term would be
$1 - \tfrac12 \times (<<Ebad>> - <<Emax>>)/<<Emax>> = <<egbad>>$.
\item \textbf{ADA.} <<fkeys>> keys require <<adareq>> accessible rooms (Table 224.2). The layout has <<adaplaced>>,
on <<adalevels>> floors, so $f_{\text{ada}} = 1$.
\item \textbf{Core access.} The <<nearn>> rooms that should be near a core have their doors <<nearmean>>\,ft from one
on average.
\item \textbf{Balance.} Keys per floor are <<perlevel>>: standard deviation <<sd>>, so $f_{\text{bal}} = <<fbal>>$.
\item \textbf{Compactness.} Shared walls total <<shared>>\,ft and room perimeters <<perim>>\,ft:
$2 \times <<shared>> / <<perim>> / 0.6 = <<fcomp>>$.
\item \textbf{Exit penalty.} The corridor reaches <<cr>> of <<ct>> cores, so the penalty is 0.
\end{itemize}
The weights sum to <<sumw>>, so $F = <<F>>$. The last column scores the reference GA's layout for the same site with
the same formulas: it loses most on placement, daylight and corridor efficiency.
\end{example}

\begin{example}{what the exit penalty does}
Take a layout with $F = 0.90$ before the penalty on a plate with two cores. If its corridor reaches only one of them,
$F = 0.90 - \tfrac12 (2 - 1) = 0.40$. Any layout that connects both exits scores higher, however poor it is otherwise.
\end{example}
""", Tmax=num(lim["travel_ft"]), Emax=num(lim["dead_end_ft"]),
                 fittable=table(["Term", "Meaning", "$w_k$", "$f_k$", "$w_k f_k$", "Reference $f_k$"], fit_rows,
                                "@{}l p{6.2cm} r r r r@{}", "", "").replace(r"\begin{table}[H]", r"\begin{center}").replace(
                     r"\end{table}", r"\end{center}"),
                 rp=fit["rooms_placed"], rt=fit["rooms_total"], fplace=f3(fit["rows"][0]["value"]),
                 csf=num(fit["corridor_sf"]), psf=num(fit["placeable_sf"]), rho=f"{fit['corridor_ratio']:.4f}",
                 feff=f3(fit["rows"][2]["value"]), T=num(fit["travel_ft"]), E=num(fit["dead_end_ft"]), Ebad=num(E_bad),
                 egbad=f"{eg_bad:.2f}", fkeys=fit["keys"], adareq=fit["ada_required"], adaplaced=fit["ada_placed"],
                 adalevels=fit["ada_levels"], nearn=fit["near_n"], nearmean=num(fit["near_mean_ft"]),
                 perlevel=", ".join(str(x) for x in fit["per_level"]), sd=f"{fit['sd']:.2f}", fbal=f3(fit["rows"][8]["value"]),
                 shared=num(fit["shared_ft"]), perim=num(fit["perimeter_ft"]), fcomp=f3(fit["rows"][9]["value"]),
                 cr=fit["cores_reached"], ct=fit["cores_total"], sumw=f"{fit['sum_w']:.2f}", F=f4(fit["fitness"])))

    # ------------------------------------------------------------------ 6 GA
    cfg = ga["config"]
    xo = ga["crossover"]
    hist_rows = [[h["gen"], f3(h["sigma"]), f4(h["best"]), f4(h["mean"]), h["rooms"], h["evals"]] for h in ga["history"]]
    mu = ga["mutation"]

    def xo_example(e, title):
        return T(r"""\begin{example}{<<title>>}
\begin{itemize}
\item Parent 1: spines <<sa>>, <<ra>> rooms, $F = <<fa>>$.
\item Parent 2: spines <<sb>>, <<rb>> rooms, $F = <<fb>>$.
\item The child takes spine blocks <<blocks>> and <<kb>> of its <<kt>> order keys from parent 2, the rest from parent 1.
\item Child (before mutation): spines <<sc>>, <<rc>> rooms, $F = <<fc>>$.
\end{itemize}
<<verdict>>
\end{example}""", title=title, sa=sset(e["spines_a"]), sb=sset(e["spines_b"]), sc=sset(e["spines_c"]), ra=e["rooms_a"],
                 rb=e["rooms_b"], rc=e["rooms_c"], fa=f3(e["fa"]), fb=f3(e["fb"]), fc=f3(e["fc"]),
                 blocks=sset([b for b in e["blocks_from_b"] if b < m]) + (" and the connector genes" if 10 in e["blocks_from_b"] else ""),
                 kb=e["keys_from_b"], kt=e["keys_total"],
                 verdict=("The child is better than both parents: the mix happened to combine a good corridor set with a "
                          "workable room order." if e["fc"] > max(e["fa"], e["fb"]) else
                          "The child is worse than both parents. Each parent's room order was tuned to its own corridors; "
                          "put on a different corridor set, the order no longer fits."))

    pct = lambda x: f"{100 * x / max(1, xo['n']):.0f}"  # noqa: E731
    out.append(T(r"""
\section{Algorithm C: the updated genetic algorithm}\label{sec:ga}
The updated GA uses the architectural decoder and adds a short local search to every generation, which makes it a
\emph{memetic} algorithm. It is seeded, so a run can be repeated exactly.

\subsection{One generation}
\begin{algorithm}[H]
\caption{Updated (memetic) GA}
\begin{algorithmic}[1]
\Require population size $P$, generations $G$, seed
\State population $\gets$ six spine-set seeds $+$ three random genomes $+$ random genomes up to $P$
\For{$t = 1$ \textbf{to} $G$}
  \State $\sigma_t \gets \sigma_{\text{start}} + (\sigma_{\text{end}} - \sigma_{\text{start}})\,\frac{t-1}{G-1}$ \Comment{mutation strength falls}
  \State next $\gets$ the <<elite>> best individuals \Comment{elitism}
  \State improve the best one by <<ls>> trial mutations of strength $\sigma_t/2$, keeping each one only if it is better \Comment{local search}
  \If{the population's fitness values are nearly equal (standard deviation $< 0.01$)}
    \State add <<imm>> random genomes \Comment{immigrants}
  \EndIf
  \While{next has fewer than $P$ individuals}
    \State pick two parents, each the best of <<tour>> individuals drawn at random \Comment{tournament selection}
    \State child $\gets$ crossover(parents) with probability <<xr>>, otherwise a copy of the first parent
    \State child $\gets$ mutate(child, $\sigma_t$); decode and score it; add it to next
  \EndWhile
  \State population $\gets$ next
\EndFor
\State \Return the best individual
\end{algorithmic}
\end{algorithm}

\subsection{The starting population}
A purely random start wastes generations rediscovering the obvious. Six seeds therefore switch on fixed sets of
spines: all of them, the best one, the best two, the best three, the best four, and every second one. Three more
genomes are random, and the rest of the population is filled with random genomes in which each spine is off with
probability 0.4.

\subsection{Selection}
To pick a parent, <<tour>> individuals are drawn at random and the fittest wins. Better individuals are picked more
often, but a weaker one still wins when it happens to be drawn with even weaker ones, which keeps variety in the
population.

\subsection{Crossover}
\begin{itemize}
\item For each of the eleven gene blocks (ten spines and the connector block) a coin decides which parent it comes
from. A spine is inherited whole: its trims, its shift and its on/off gene stay together.
\item For each room's order key another coin decides.
\end{itemize}
<<xbetter>>
<<xworse>>

\begin{example}{what crossover achieved over a whole run}
I logged every crossover in the example run and scored each child \emph{before} mutation, next to its two parents
(<<xn>> crossovers):
\begin{itemize}
\item <<xb>> children (<<pb>>\%) were better than their better parent;
\item <<xe>> (<<pe>>\%) scored the same as their better parent;
\item <<xm>> (<<pm>>\%) fell between the two parents;
\item <<xw>> (<<pw>>\%) were worse than both.
\end{itemize}
On average the better parent scored <<mpb>> and the child <<mc>>. Crossover sometimes produces the jump that matters
(the first example above), but most of the time it does not improve on what the parents already had. This matches
the review comment that splicing two plans seldom makes architectural sense, and it is why the default search
(Section~\ref{sec:sys}) does without it.
\end{example}

\subsection{Mutation}
A child is changed in four ways:
\begin{itemize}
\item \textbf{Gaussian noise} of standard deviation $\sigma_t$ on about 30\% of the 44 corridor genes;
\item \textbf{spine toggles}: each of the eleven blocks flips its on/off gene with probability <<tog>>;
\item \textbf{key swaps}: with probability <<swp>>, <<nswap>> pairs of rooms exchange their order keys;
\item \textbf{Gaussian noise} on about 10\% of the order keys.
\end{itemize}
All genes are then clipped to $[0, 1]$. The strength $\sigma_t$ falls linearly from <<s0>> in the first generation to
<<s1>> in the last: wide exploration early, fine adjustment late.

\begin{example}{one mutation at full strength}
Mutating the final best genome once with $\sigma = <<msig>>$ changed <<mc1>> of <<mc2>> corridor genes and
<<mk1>> of <<mk2>> order keys. The active spines changed from <<mb>> to <<ma>>, and the fitness went from <<mfb>> to
<<mfa>>. One toggled spine is enough to change the plan completely, which is why the best individual is always carried
over unchanged (elitism) and why mutation gets weaker over the run.
\end{example}

\subsection{The example run}
<<histtable>>
The run evaluated <<evals>> layouts: $P = <<P>>$ to start, then in each of <<G>> generations
$P - <<elite>> = <<children>>$ children and <<ls>> local-search trials, $<<P>> + <<G>> \times (<<children>> + <<ls>>) = <<evals>>$.
It ended with <<rooms>> of <<rt>> rooms and $F = <<F>>$. The best layout was already close after the first
generation (the seeds do most of the work); the rest of the run adds small gains.
""", elite=cfg["elite"], ls=cfg["local_search_steps"], imm=cfg["immigrants"], tour=cfg["tournament"],
                 xr=cfg["crossover_rate"], xbetter=xo_example(xo["example_better"], "a crossover that helps"),
                 xworse=xo_example(xo["example_worse"], "a crossover that hurts"), xn=xo["n"], xb=xo["better"],
                 pb=pct(xo["better"]), xe=xo["equal"], pe=pct(xo["equal"]), xm=xo["between"], pm=pct(xo["between"]),
                 xw=xo["worse"], pw=pct(xo["worse"]), mpb=f4(xo["mean_parent_best"]), mc=f4(xo["mean_child"]),
                 tog=cfg["segment_toggle_rate"], swp=f"{min(1.0, cfg['key_swap_rate'] * 3):.2f}",
                 nswap=max(1, int(site["rooms"] * cfg["key_swap_rate"] / 3)), s0=cfg["sigma_start"], s1=cfg["sigma_end"],
                 msig=mu["sigma"], mc1=mu["corridor_changed"], mc2=mu["corridor_total"], mk1=mu["keys_changed"],
                 mk2=mu["keys_total"], mb=sset(mu["spines_before"]), ma=sset(mu["spines_after"]),
                 mfb=f3(mu["fitness_before"]), mfa=f3(mu["fitness_after"]),
                 histtable=table(["Generation", "$\\sigma_t$", "Best $F$", "Mean $F$ of population", "Rooms (best)",
                                  "Layouts evaluated"], hist_rows, "@{}r r r r r r@{}",
                                 f"The GA on the running example (population {P_}, {G_} generations, seed {seed}).", "tab:ga"),
                 evals=ga["evaluations"], P=P_, G=G_, children=P_ - cfg["elite"], rooms=ga["rooms"], rt=ga["rooms_total"],
                 F=f4(ga["fitness"])))

    # ------------------------------------------------------------------ 7 systematic
    s1_rows = [[sset(r["spines"]), r["rooms"], f4(r["fitness"])] for r in sy["stage1_top"]]
    s1_rows.append([r"\ldots", "", ""])
    s1_rows += [[sset(r["spines"]), r["rooms"], f4(r["fitness"])] for r in sy["stage1_bottom"]]
    sizes = sy["by_size"]
    tr_rows = [[t["step"], t["tried"], esc(t["move"]), f4(t["from"]), f4(t["to"]), t["rooms"]] for t in sy["descent"]["trace"]]
    imp_rows = [[t["call"], esc(t["move"]), f4(t["from"]), f4(t["to"])] for t in sy["improvements"]]
    a_act = len(sy["active_spines"])
    out.append(T(r"""
\section{Algorithm D: the systematic search (default)}\label{sec:sys}
With the architectural decoder the important decision is small: which of at most ten spines are on. A space that
small can be searched in an orderly way, with no population and no crossover. The search has three stages and a fixed
budget of $P \times G$ layouts, the same as the GA's.

\subsection{Stage 1: enumerate the spine sets}
With $m$ candidate spines the first stage tries
\begin{itemize}
\item all spines switched on, and
\item every set of one, two, three or four of the seven best spines,
\end{itemize}
each with no trim, no shift and all order keys at 0.5 (rooms in priority order). That is
\[ 1 + \sum_{k=1}^{4} \binom{\min(m,7)}{k} \;\le\; 1 + 7 + 21 + 35 + 35 = 99 \text{ layouts.} \]

\begin{example}{stage 1 on \code{fp1}}
The plate has $m = <<m>>$ candidates, so stage 1 evaluates $1 + <<c1>> + <<c2>> + <<c3>> + <<c4>> = <<nsets>>$ spine sets.
<<s1table>>
Fitness ranges from <<lo>> to <<hi>>. Two things stand out:
\begin{itemize}
\item \textbf{The corridor set decides almost everything.} Spine 4 alone places <<worstrooms>> rooms; any set containing
spines 0 and 1 places all <<rt>>.
\item \textbf{Different sets can give the same plan.} The best sets all contain spines 0 and 1 and have the same
fitness, because the decoder drops the extra spines when they get no bays of their own (Step 4 of
Section~\ref{sec:decoder}).
\end{itemize}
\end{example}

\subsection{Stage 2: pattern search from the three best}
The three best \emph{distinct} layouts of stage 1 become starting points. For each one the search generates its
\emph{neighbours}, the layouts one small move away:
\begin{itemize}
\item switch one spine on or off (10 moves);
\item for each active spine: trim the low end, trim the high end, shift it one way, shift it the other (4 moves each);
\item redraw the connector elbows (1 move);
\item move a few rooms earlier or later in the assignment order (4 moves, about one room in twelve each).
\end{itemize}
The neighbours are evaluated in a seeded random order, and the \textbf{first one that is better is accepted}
($F' > F + 10^{-6}$). The search then continues from the new layout. When no neighbour is better, that start is a
\emph{local optimum} and is left alone.

\begin{example}{one descent, move by move}
Starting from the best stage-1 layout (spines <<startsp>>) and applying the rule above:
<<trtable>>
The descent tried <<tried>> neighbours in total. With <<aact>> active spines a neighbourhood has
$10 + 4 \times <<aact>> + 1 + 4 = <<nbh>>$ moves; the last row shows all of them tried without success.
\end{example}

\subsection{Stage 3: iterated local search}
If all three starts are local optima and budget remains, the search \emph{kicks} the best layout: it applies two
random moves at once, which is enough to leave the local optimum. It then descends again with the same pattern search
and keeps the result only if it beats the best layout so far. This repeats until the budget is used.

\subsection{The whole run on the example}
\begin{example}{every improvement after stage 1}
The real run made <<calls>> decoder calls (<<evals>> distinct layouts; a genome seen before is not decoded again).
After the <<nsets>> layouts of stage 1, the best fitness rose <<nimp>> times:
<<imptable>>
The run ended with <<rooms>> of <<rt>> rooms and $F = <<F>>$. A second run with the same inputs returned exactly the
same genome and fitness: <<same>>.
\end{example}
<<convfig>>

\begin{algorithm}[H]
\caption{Systematic search}
\begin{algorithmic}[1]
\Require budget $B = P \times G$ layouts, seed
\State evaluate every spine set of stage 1 \Comment{at most 99 layouts}
\State starts $\gets$ the three best distinct layouts
\While{budget remains and some start is not a local optimum}
  \For{each start $x$ that is not a local optimum}
    \State evaluate the neighbours of $x$ in seeded random order; move $x$ to the first better one
    \State if none is better, mark $x$ as a local optimum
  \EndFor
\EndWhile
\While{budget remains} \Comment{iterated local search}
  \State $y \gets$ best layout so far, changed by two random moves
  \State descend from $y$ by the same pattern search
  \State if $y$ is now better than the best layout, it becomes the best
\EndWhile
\State \Return the best layout
\end{algorithmic}
\end{algorithm}
""", m=sy["m"], c1=sizes.get("1", 0), c2=sizes.get("2", 0), c3=sizes.get("3", 0), c4=sizes.get("4", 0), nsets=sy["n_sets"],
                 s1table=table(["Spines switched on", "Rooms placed", "Fitness $F$"], s1_rows, "@{}l r r@{}", "", "").replace(
                     r"\begin{table}[H]", r"\begin{center}").replace(r"\end{table}", r"\end{center}"),
                 lo=f3(sy["stage1_spread"][0]), hi=f3(sy["stage1_spread"][1]), worstrooms=sy["stage1_bottom"][-1]["rooms"],
                 rt=site["rooms"], startsp=sset(sy["descent"]["start_spines"]),
                 trtable=table(["Step", "Neighbours tried", "Accepted move", "$F$ before", "$F$ after", "Rooms"], tr_rows,
                               "@{}r r p{6.2cm} r r r@{}", "", "").replace(r"\begin{table}[H]", r"\begin{center}").replace(
                     r"\end{table}", r"\end{center}"),
                 tried=sy["descent"]["tried_total"], aact=a_act, nbh=10 + 4 * a_act + 5, calls=sy["calls"],
                 evals=sy["evaluations"], nimp=len(sy["improvements"]),
                 imptable=table(["Decoder call", "Move that produced it", "$F$ of its parent", "$F$ after"], imp_rows,
                                "@{}r p{7.5cm} r r@{}", "", "").replace(r"\begin{table}[H]", r"\begin{center}").replace(
                     r"\end{table}", r"\end{center}"),
                 rooms=sy["rooms"], F=f4(sy["fitness"]), same="yes" if sy["repeat_identical"] else "no",
                 convfig=fig("convergence.png",
                             "The two searches on the running example, same decoder and same budget. Left: best fitness so far "
                             "against layouts evaluated; the dotted line marks the end of stage 1 of the systematic search. "
                             "Right: the GA's best individual and the mean fitness of its population, which stays well below "
                             "the best because mutation keeps breaking good plans.", height="0.28")))

    # ------------------------------------------------------------------ 8 comparison
    lab = {"A": "A reference GA", "B": "B first attempt", "C": "C updated GA", "D": "D systematic"}
    comp_rows = []
    for s in sites:
        for t in "ABCD":
            if t not in s["variants"]:
                continue
            v = s["variants"][t]
            comp_rows.append([s["site"].replace("_multi", "") if t == "A" else "", lab[t],
                              f"{v['rooms_placed']}/{v['rooms_total']}", v["windowless_guest_rooms"], f3(v["fitness"]),
                              f"{v['evaluations']:,}" if v.get("evaluations") else "500 gen.",
                              f"{v['seconds']:.0f}\\,s" if v.get("seconds") else "n/a",
                              f"{v['checks_pass']}/{v['checks_total']}"])
        if s is not sites[-1]:
            comp_rows[-1][-1] += r" \\ \midrule"
    comp = table(["Site", "Algorithm", "Rooms", "No window", "$F$", "Evaluated", "Time", "Checks"], comp_rows,
                 "@{}l l r r r r r r@{}", "All four algorithms on the three sites, same program, same fitness "
                                          f"(budget {P_} $\\times$ {G_} for B, C and D, seed {seed}). ``No window'' counts "
                                          "guest rooms without a facade wall.", "tab:comp").replace(r" \\ \midrule \\", r" \\ \midrule")
    out.append(T(r"""
\section{The algorithms side by side}
<<comp>>
\begin{itemize}
\item \textbf{The decoder matters more than the search.} B has a better search and scoring than A but the same kind
of decoder, and its plans keep the windowless rooms. C and D share the new decoder and neither has any.
\item \textbf{GA or systematic?} At the same budget the GA (C) finishes slightly ahead on fitness: <<fc>> against
<<fd>> on average, <<rc>> rooms against <<rd>>. Its random order keys and crossover occasionally find a room order
that the move-by-move search does not reach.
\item \textbf{Why the systematic search is still the default.} It has no population size, crossover rate or mutation
schedule to tune, it explains itself (every improvement is one named move), and it gives the same answer every time.
The GA is one setting away in the Design Optimizer when the last few thousandths matter.
\end{itemize}
""", comp=comp, fc=f3(tot("C", "fitness") / len(sites)), fd=f3(tot("D", "fitness") / len(sites)),
                 rc=tot("C", "rooms_placed"), rd=tot("D", "rooms_placed")))

    # ------------------------------------------------------------------ 9 hotel exploration
    dim_rows = [[x["shape"], f"{x['a']} $\\times$ {x['b']}", num(x["run"]), num(x["footprint"]), num(x["facade"]),
                 "yes" if x["clipped"] else "no", ", ".join(x["schemes"])] for x in pl["dims"]]
    ub = pl["ub_example"]
    hb = hotel["best"]
    refine = "; ".join(f"{esc(t['plate'])}, {t['floors']} floors: {t['start']:.3f} $\\to$ {t['end']:.3f}"
                       for t in hotel["refine"][:3])
    out.append(T(r"""
\section{Algorithm E: exploring hotel layouts (shape, core and floors)}\label{sec:hotel}
On the prototype's sites the plate and the cores are given. In the Hotel Schematic mode they are not: only the program
and the buildable area are. An outer loop therefore \emph{generates} plates and runs the decoder on each.

\subsection{Sizing a plate}
For $K$ keys on $F$ guest floors, let $A$ be the net room area of the busiest level. With a slack factor
$\varsigma = 1.04$, core area $A_c$ and a packing efficiency $\eta = 0.93$, the plate must offer a corridor of length
\[ R = \frac{\varsigma A + A_c}{2 D \eta}, \]
because every foot of double-loaded corridor serves $2D$ square feet of rooms. Each shape with bounding box
$a \times b$ and wing depth $W_d = 2D + w$ offers
\[ R_{\text{Bar}} = a,\quad R_{\text{L}} = R_{\text{T}} = a + b - W_d,\quad R_{\text{U}} = R_{\text{H}} = a + 2b - 2W_d,\quad
R_{\text{Courtyard}} = 2a + 2b - 4W_d . \]
For each shape the generator finds the smallest $a$ with $R_{\text{shape}}(a, \beta a) \ge R$ at aspect ratios
$\beta \in \{0.6, 1\}$, in both orientations, and clips the result to the buildable area.

\begin{example}{plates for <<K>> keys on <<F>> guest floors}
\begin{itemize}
\item Net area per level: <<needs>>\,SF. The busiest is <<busiest>> with $A = <<A>>$\,SF.
\item Elevators: $\max(2, \lceil <<K>>/75 \rceil) = <<elev>>$. The main core is <<mc0>> $\times$ <<mc1>>\,ft and each
stair <<st0>> $\times$ <<st1>>\,ft, so $A_c = <<Ac>>$\,SF.
\item Room depth $D = <<D>>$\,ft and corridor $w = <<w>>$\,ft give a wing depth $W_d = <<Wd>>$\,ft.
\item $R = (1.04 \times <<A>> + <<Ac>>) / (2 \times <<D>> \times 0.93) = <<R>>$\,ft.
\end{itemize}
A Bar needs $a = <<Rceil>>$\,ft, more than the <<env1>>\,ft the buildable area allows, so it is clipped to <<env1>>\,ft
(the layout then has to do with less corridor than the estimate asks for). The generator produces <<nplates>> plates for this floor count:
<<dimtable>>
\end{example}
<<platefig>>

\subsection{Cores and stairs}
Each plate is tried with up to three core schemes:
\begin{itemize}
\item \textbf{ends}: the elevator core near one end, a stair near the other;
\item \textbf{center}: the elevator core at the middle of the longest wing, and two stairs at the pair of wing ends
that are farthest apart (which helps exit separation and keeps dead ends short);
\item \textbf{knuckle}: the elevator core where two wings meet, with stairs at the far ends.
\end{itemize}

\subsection{The three stages}
\begin{enumerate}
\item \textbf{Screen.} Every plate, for every floor count tried (the requested count, one fewer, and up to two more,
within the height limit), is decoded with <<seeds>> fixed spine sets (all spines, the best one, two, three and four,
and every second one), rooms in priority order. Plates are independent, so they run in parallel processes.
\item \textbf{Refine.} The best plate of each shape, then the next best overall up to <<rtop>> plates, are improved by
the pattern search of Section~\ref{sec:sys} for <<rsteps>> evaluations each.
\item \textbf{Archive.} Every distinct layout is kept. Two layouts count as the same if they have the same plate,
floor count, rooms placed, corridor area, shared-wall length and fitness (to three decimals).
\end{enumerate}
One extra code check is added here because the floor count now varies: the number of stories (guest floors plus the
ground level) must not exceed 5.

\begin{example}{the run on the Living\,+\,Lodging program}
Buildable area <<e0>> $\times$ <<e1>>\,ft. The exploration screened <<variants>> plate and floor-count variants,
evaluated <<hevals>> layouts ($<<variants>> \times <<seeds>>$ in screening plus $<<rtop>> \times <<rsteps>>$ in
refinement) and kept <<layouts>> distinct ones in <<hsecs>>\,s on eight processes. Refinement gains were small:
<<refine>>. The layout ranked first (next section) is <<best>> with <<bf>> guest floors.
\end{example}

\subsection{An upper bound per floor}
Along a corridor of length $R$, both sides together offer $2R$ feet of frontage, less the frontage $c_f$ taken by the
cores. No layout can have more than
\[ K_{\text{floor}} = \Big\lfloor \frac{2R - c_f}{M} \Big\rfloor \]
keys on a typical floor.
\begin{example}{bound for the <<ubname>> plate}
$R = <<ubrun>>$\,ft, the cores take $c_f = <<cf>>$\,ft (the longer side of each of the <<ubcores>> cores) and the module
is $M = <<ubM>>$\,ft: $\lfloor (2 \times <<ubrun>> - <<cf>>)/<<ubM>> \rfloor = <<ub>>$ keys per floor, so at most
$<<ub>> \times <<F>> = <<ubtot>>$ on <<F>> floors, against <<K>> requested. The bound is loose (it ignores suites, which
take two bays, and the per-floor service rooms), but it shows at once whether a request can fit.
\end{example}
""", K=pl["keys"], F=pl["floors"], needs=", ".join(num(x) for x in pl["need_by_level"]), busiest=pl["busiest"],
                 A=num(pl["need_max"]), elev=pl["elevators"], mc0=pl["main_core"][0], mc1=pl["main_core"][1],
                 st0=pl["stair"][0], st1=pl["stair"][1], Ac=num(pl["core_area"]), D=num(pl["room_depth"]),
                 w=num(pl["corridor_ft"]), Wd=pl["wing_depth"], R=num(pl["run"]), Rceil=int(-(-pl["run"] // 1)),
                 env1=max(pl["envelope"]), nplates=pl["plates"],
                 dimtable=table(["Shape", "$a \\times b$, ft", "Corridor run, ft", "Footprint, SF", "Facade, ft", "Clipped",
                                 "Core schemes"], dim_rows, "@{}l c r r r c l@{}", "", "").replace(
                     r"\begin{table}[H]", r"\begin{center}").replace(r"\end{table}", r"\end{center}"),
                 platefig=fig("plates.png", "One generated plate per shape with the centre-core scheme. Dark: elevator core; "
                                            "lighter: stairs; blue: corridor centre-lines.", height="0.26"),
                 seeds=pl["seeds_per_plate"], rtop=pl["refine_top"], rsteps=pl["refine_steps"], e0=hotel["envelope"][0],
                 e1=hotel["envelope"][1], variants=hotel["variants"], hevals=f"{hotel['evaluations']:,}",
                 layouts=hotel["layouts"], hsecs=f"{hotel['seconds']:.0f}", refine=refine, best=esc(hb["plate"]),
                 bf=hb["floors"], ubname=esc(ub["name"]), ubrun=ub["run"], cf=pl["main_core"][0] + 2 * max(pl["stair"]),
                 ubcores=ub["cores"], ubM=pl["module_ft"], ub=ub["ub"], ubtot=ub["ub"] * pl["floors"]))

    # ------------------------------------------------------------------ 10 decision
    rows = dc["rows"]
    names = [f"{'ABCD'[i]}" for i in range(len(rows))]
    lay_rows = [[names[i], esc(r["layout"]), r["floors"], f3(r["x"][0]), num(round(r["x"][1])), num(r["x"][2]),
                 "on the front" if r["pareto"] else "dominated by " + names[r["dominated_by"][0]]]
                for i, r in enumerate(rows)]
    top_rows = [[names[i], *[f4(x) for x in r["r"]], *[f4(x) for x in r["v"]], f4(r["dp"]), f4(r["dm"]),
                 r"\textbf{" + f3(r["C"]) + "}"] for i, r in enumerate(rows)]
    order = " $>$ ".join(names[i] for i in dc["order"])
    dom = next((i for i, r in enumerate(rows) if r["dominated_by"]), None)
    dom_txt = ""
    if dom is not None:
        j = rows[dom]["dominated_by"][0]
        dom_txt = (f"Layout {names[dom]} is dominated by {names[j]}: {names[j]} has higher fitness "
                   f"({f3(rows[j]['x'][0])} against {f3(rows[dom]['x'][0])}), less area per key "
                   f"({num(round(rows[j]['x'][1]))} against {num(round(rows[dom]['x'][1]))}\\,SF) and a dead end that is no "
                   f"longer ({num(rows[j]['x'][2])} against {num(rows[dom]['x'][2])}\\,ft).")
    front = [names[i] for i, r in enumerate(rows) if r["pareto"]]
    front_txt = ""
    if len(front) >= 2:
        i, j = [k for k, r in enumerate(rows) if r["pareto"]][:2]
        front_txt = (f"Layouts {front[0]} and {front[1]} are both on the front: {names[j]} has the higher fitness "
                     f"({f3(rows[j]['x'][0])} against {f3(rows[i]['x'][0])}) but needs more area per key "
                     f"({num(round(rows[j]['x'][1]))} against {num(round(rows[i]['x'][1]))}\\,SF), so neither beats the other "
                     "on everything." if rows[j]["x"][0] > rows[i]["x"][0] else
                     f"Layouts {front[0]} and {front[1]} are both on the front: each is better than the other on at least one criterion.")
    wsum = sum(dc["weights"].values())
    out.append(T(r"""
\section{Choosing among the layouts: Pareto front and TOPSIS}\label{sec:decide}
The exploration returns hundreds of layouts. Which is best depends on what the owner values: the most keys, the lowest
cost, the lowest building. That choice is kept separate from the fitness and is made in two deterministic steps over
<<ncrit>> criteria (fitness, keys, share of the program placed, daylight, corridor share, gross area per key, facade
per key, height, dead end, exit travel, code checks passed, compactness).

\subsection{Pareto front}
Layout $x$ \emph{dominates} layout $y$ if $x$ is at least as good as $y$ on every criterion and strictly better on at
least one. A layout that nothing dominates is on the \emph{Pareto front}. A dominated layout is never the right
choice, whatever the priorities.

\subsection{TOPSIS}
TOPSIS ranks all layouts for a given set of weights $\lambda_q$:
\begin{enumerate}
\item \textbf{Normalise} each criterion $q$ over the layouts $j$: $r_{jq} = x_{jq} / \sqrt{\sum_j x_{jq}^2}$.
\item \textbf{Weight}: $v_{jq} = \dfrac{\lambda_q}{\sum_q \lambda_q}\, r_{jq}$.
\item Find the \textbf{ideal} $v_q^+$ (the best value of each criterion over all layouts) and the \textbf{worst}
$v_q^-$. ``Best'' is the largest value for a criterion to maximise and the smallest for one to minimise.
\item Measure each layout's \textbf{distance} to both: $d_j^{\pm} = \sqrt{\sum_q (v_{jq} - v_q^{\pm})^2}$.
\item The \textbf{closeness} $C_j = \dfrac{d_j^-}{d_j^+ + d_j^-}$ lies between 0 and 1. The layout with the largest
$C_j$ is ranked first.
\end{enumerate}

\begin{example}{four real layouts, three criteria}
To keep the arithmetic visible, take four layouts from the hotel run and only three of the criteria, with the weights
of the Balanced preset: <<l0>> (maximise, $\lambda = <<w0>>$), <<l1>> (minimise, $\lambda = <<w1>>$) and <<l2>>
(minimise, $\lambda = <<w2>>$).
<<laytable>>
<<domtxt>> <<fronttxt>>

Normalisers $\sqrt{\sum_j x_{jq}^2}$ are <<n0>>, <<n1>> and <<n2>>; the weights become <<wn0>>, <<wn1>>, <<wn2>>
(each $\lambda_q$ divided by <<wsum>>).
<<toptable>>
The ideal point is $(<<i0>>, <<i1>>, <<i2>>)$ and the worst $(<<o0>>, <<o1>>, <<o2>>)$. For layout A,
$C = <<dmA>> / (<<dpA>> + <<dmA>>) = <<CA>>$. The ranking is <<order>>.

With all <<ncrit>> criteria and all <<nlay>> layouts the same procedure produces the ranking shown in the app. The
Pareto flags can differ from this small example, because a layout dominated on three criteria may be the best on a
fourth.
\end{example}

\subsection{Presets}
The app offers five weight presets (Balanced, Most keys, Lowest cost, Guest experience, Low-rise) and sliders. Changing
the weights re-ranks the stored layouts instantly; nothing is searched again. In the hotel run the Balanced preset
ranks <<bal>> first, while Most keys ranks <<mk>> first.
""", ncrit=dc["n_criteria"], l0=dc["labels"][0], l1=dc["labels"][1], l2=dc["labels"][2], w0=dc["weights"][dc["keys"][0]],
                 w1=dc["weights"][dc["keys"][1]], w2=dc["weights"][dc["keys"][2]],
                 laytable=table(["", "Layout", "Floors", "Fitness", "SF per key", "Dead end, ft", "On these three criteria"],
                                lay_rows, "@{}l l r r r r l@{}", "", "", size=r"\footnotesize").replace(r"\begin{table}[H]", r"\begin{center}").replace(
                     r"\end{table}", r"\end{center}"),
                 domtxt=dom_txt, fronttxt=front_txt, n0=num(dc["norm"][0]) if dc["norm"][0] > 10 else f3(dc["norm"][0]),
                 n1=num(dc["norm"][1]), n2=num(dc["norm"][2]), wn0=dc["wn"][0], wn1=dc["wn"][1], wn2=dc["wn"][2], wsum=num(wsum),
                 toptable=table(["", "$r_1$", "$r_2$", "$r_3$", "$v_1$", "$v_2$", "$v_3$", "$d^+$", "$d^-$", "$C$"], top_rows,
                                "@{}l r r r r r r r r r@{}", "", "").replace(r"\begin{table}[H]", r"\begin{center}").replace(
                     r"\end{table}", r"\end{center}"),
                 i0=f4(dc["ideal"][0]), i1=f4(dc["ideal"][1]), i2=f4(dc["ideal"][2]), o0=f4(dc["worst"][0]),
                 o1=f4(dc["worst"][1]), o2=f4(dc["worst"][2]), dmA=f4(rows[0]["dm"]), dpA=f4(rows[0]["dp"]),
                 CA=f3(rows[0]["C"]), order=order, nlay=hotel["layouts"],
                 bal=esc(hotel["presets"]["Balanced"]["layout"]), mk=esc(hotel["presets"]["Most keys"]["layout"])))

    # ------------------------------------------------------------------ 11 bounds, parameters, reproduce
    b = sites[0]["bounds"]
    D1 = sites[0]["variants"]["D"]
    par_rows = [["Population $P$", cfg["population"], "C; with $G$ it also sets the budget of D"],
                ["Generations $G$", cfg["generations"], "C; budget $P \\times G$ for D"],
                ["Elite", cfg["elite"], "C: best individuals copied unchanged"],
                ["Tournament size", cfg["tournament"], "C: selection"],
                ["Crossover rate", cfg["crossover_rate"], "C"],
                ["Mutation $\\sigma$", f"{cfg['sigma_start']} $\\to$ {cfg['sigma_end']}", "C: linear schedule"],
                ["Spine toggle rate", cfg["segment_toggle_rate"], "C: per block, per mutation"],
                ["Local-search trials", cfg["local_search_steps"], "C: per generation, on the best individual"],
                ["Immigrants", cfg["immigrants"], "C: only when the population has converged"],
                ["Seed", cfg["seed"], "all: fixes every random choice"],
                ["On/off threshold", "0.12", "decoder: a spine is on at or above it"],
                ["Maximum trim", "35\\% per end", "decoder"],
                ["Maximum shift", "$\\pm 2$\\,ft", "decoder"],
                ["Size tolerance", "85\\%", "decoder: minimum frontage and depth of a guest room"],
                ["Remainder rule", "$0.4M$", "decoder: smaller remainders join the last bay"],
                ["Starts in stage 2", "3", "D"],
                ["Spine sets in stage 1", "up to 99", "D: sets of up to 4 of the best 7, plus all"],
                ["Seeds per plate", pl["seeds_per_plate"], "E: screening"],
                ["Plates refined", pl["refine_top"], f"E: {pl['refine_steps']} evaluations each"]]
    out.append(T(r"""
\section{How good can any algorithm be? Two upper bounds}
Before judging a search it helps to know what is possible at all. Two bounds hold for every layout.

\textbf{Area.} A corridor of width $w$ serves at most two room depths, so at least $w/(2D + w)$ of the used floor is
corridor. With $L$ levels of placeable area $|\Omega|$ and mean key area $\bar a$:
\[ K \le K_{\text{area}} = \Big\lfloor \frac{L\,|\Omega|\,\big(1 - \tfrac{w}{2D+w}\big)}{\bar a} \Big\rfloor . \]
\textbf{Windows.} If every guest room needs a window, the rooms of one floor share the facade length $P_f$. With mean
key frontage $\bar f$:
\[ K \le K_{\text{window}} = L \Big\lfloor \frac{P_f}{\bar f} \Big\rfloor . \]
\begin{example}{bounds for \code{fp1}}
$L = <<L>>$, $|\Omega| = <<omega>>$\,SF, minimum corridor share <<cmin>>, mean key area <<abar>>\,SF:
$K_{\text{area}} = \lfloor <<L>> \times <<omega>> \times (1 - <<cmin>>) / <<abar>> \rfloor = <<Karea>>$.
The facade is <<facade>>\,ft per level and the mean key frontage <<fbar>>\,ft:
$K_{\text{window}} = <<L>> \times \lfloor <<facade>> / <<fbar>> \rfloor = <<Kwin>>$.
The program asks for <<kreq>> keys, below both bounds, so the request is feasible in principle. The systematic search
places <<kd>> of them and the reference GA <<ka>>.
\end{example}

\section{Summary}
\begin{itemize}
\item \textbf{Representation first.} The reference GA and my first attempt searched over free corridor segments, a
space in which most candidates are poor plans. The updated decoder restricts the search to plans with straight,
facade-aligned corridors and modular bays. That single change removed every windowless guest room on all three sites.
\item \textbf{Then a simple search is enough.} With at most ten spines to switch on or off, enumerating the
combinations and improving the best ones move by move finds a layout within about 0.01 of the GA's fitness on every
site, with the same answer on every run.
\item \textbf{The GA is still useful.} It finishes slightly ahead at equal budget, and the logged run shows why: most
crossovers do not help, but the few that do combine a corridor set and a room order that single moves reach slowly.
\item \textbf{Score and choice are separate.} The fitness is fixed arithmetic. Which good layout to build is decided
afterwards, by Pareto dominance and TOPSIS under weights the user controls.
\end{itemize}

\appendix
\section{Parameters}
<<partable>>

\section{Reproducing this report}
\begin{verbatim}
python src/build_optimizer_study.py                  # three-site comparison
python src/build_optimization_report.py --recompute  # worked examples + this PDF
python src/run_optimizer.py --method systematic      # one run of algorithm D
python src/run_optimizer.py --method ga              # one run of algorithm C
\end{verbatim}
The code is in \code{src/rngd/optimize/}: \code{planner.py} (decoder), \code{metrics.py} (fitness), \code{search.py}
(algorithms C and D), \code{engine.py} (the free-form decoder of B), \code{baseline.py} (loader for the reference
results), \code{typology.py} and \code{explore.py} (algorithm E) and \code{decide.py} (Pareto and TOPSIS). The worked
examples are produced by \code{src/optimization\_examples.py}.
\end{document}
""", L=b["levels"], omega=num(b["placeable_sf_per_level"]), cmin=b["min_corridor_share"], abar=num(b["mean_key_area_sf"]),
                 Karea=b["area_bound_keys"], facade=num(b["facade_ft_per_level"]), fbar=b["mean_key_frontage_ft"],
                 Kwin=b["window_bound_keys"], kreq=b["keys_requested"], kd=D1["keys_placed"], ka=A1["keys_placed"],
                 partable=table(["Parameter", "Value", "Used by"], par_rows, "@{}l l l@{}", "Main parameters and their default "
                                                                                           "values.", "tab:par")))
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--recompute" in sys.argv or not (OUT / "examples.json").exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import optimization_examples
        optimization_examples.main()
    ex = json.loads((OUT / "examples.json").read_text(encoding="utf-8"))
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    tex = build(ex, study)
    (OUT / f"{NAME}.tex").write_text(tex, encoding="utf-8")
    r = subprocess.run([str(TECTONIC), f"{NAME}.tex"], cwd=OUT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise SystemExit("LaTeX failed")
    for ln in sorted({ln for ln in r.stderr.splitlines() if "Overfull" in ln or "undefined" in ln.lower()}):
        print("  warning:", ln.strip()[:160])
    print(OUT / f"{NAME}.pdf")


if __name__ == "__main__":
    main()
