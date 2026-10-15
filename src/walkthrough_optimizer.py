"""Walkthrough-report section on layout optimization: architecture, the problems observed in
the reference GA, how each was solved, the mathematics, and the measured comparison.

All numbers come from reports/walkthrough/optimizer/study.json (src/build_optimizer_study.py).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "walkthrough"
STUDY = OUT / "optimizer" / "study.json"
SHOTS = OUT / "screens" / "optimizer_shots.json"


def load_study() -> dict | None:
    return json.loads(STUDY.read_text(encoding="utf-8")) if STUDY.exists() else None


def _esc(s) -> str:
    s = "" if s is None else str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("·", r"$\cdot$"), ("×", r"$\times$")]:
        s = s.replace(a, b)
    return s


def _table(headers, rows, spec) -> str:
    head = " & ".join(r"\textbf{" + h + "}" for h in headers)
    body = "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in rows)
    return (r"{\small\begin{longtable}{" + spec + "}\n\\toprule\n" + head + r" \\ \midrule \endhead" + "\n" + body
            + "\n\\bottomrule\n\\end{longtable}}")


def _fig(name: str, caption: str, height: str = "0.42") -> str:
    p = OUT / "optimizer" / name
    if not p.exists():
        return ""
    return (r"\begin{figure}[H]\centering\includegraphics[width=\linewidth,keepaspectratio,height=" + height
            + r"\textheight]{optimizer/" + name + r"}\caption{" + caption + r"}\end{figure}")


def _shot_figs() -> str:
    if not SHOTS.exists():
        return ""
    out = []
    for s in json.loads(SHOTS.read_text(encoding="utf-8")):
        if (OUT / "screens" / s["file"]).exists():
            out.append(r"\begin{figure}[H]\centering\includegraphics[width=\linewidth,keepaspectratio,height=0.42\textheight]"
                       r"{screens/" + s["file"] + r"}\caption{" + _esc(s["caption"]) + r"}\end{figure}")
    return "\n".join(out)


ARCH = r"""\begin{figure}[H]\centering
\begin{tikzpicture}[node distance=6mm and 5mm, every node/.style={font=\footnotesize},
 box/.style={draw=bpdeep, fill=bp!8, rounded corners=2pt, align=center, minimum height=10mm, text width=27mm},
 data/.style={draw=muted, fill=white, align=center, minimum height=9mm, text width=27mm},
 ai/.style={box, fill=warn!18, draw=warn}, loop/.style={box, fill=bp!22},
 arr/.style={-{Stealth[length=2mm]}, thick, bpdeep}]
\node[data] (site) {Site: 1 ft grid (QA) or buildable area W x D (hotel)};
\node[data, below=of site] (prog) {Room program: sizes, counts, priority, adjacency};
\node[data, below=of prog] (code) {Code limits: RNGD IBC CSV, ADA 224.2};
\node[box, right=of site] (plate) {1 Plate generator: shape, proportion, cores, floors (hotel only)};
\node[loop, right=of plate] (dec) {2 Decoder: spines, connectors, bays, room assignment};
\node[loop, below=of dec] (eval) {3 Evaluator: ten objectives, six or seven code checks, fitness $F$};
\node[loop, left=of eval] (search) {4 Search: enumerate spine sets, pattern search (or GA)};
\node[box, right=of dec] (arch) {5 Archive of distinct layouts};
\node[box, below=of arch] (dm) {6 Decision: Pareto front, TOPSIS ranking};
\node[box, below=of dm] (ui) {7 App: dropdown of layouts, furnished plans, 3D, JSON};
\node[ai, below=of eval] (llm) {Optional LLM: report prose, chat, Jev advisor};
\draw[arr] (site) -- (plate); \draw[arr] (plate) -- (dec); \draw[arr] (dec) -- (eval);
\draw[arr] (eval) -- (search); \draw[arr] (search) -- node[left, pos=0.4]{\scriptsize genome} (dec.south west);
\draw[arr] (prog.east) -- ++(2mm,0) |- (search.west); \draw[arr] (code.east) -- ++(4mm,0) |- (eval.south west);
\draw[arr] (dec) -- (arch); \draw[arr] (arch) -- (dm); \draw[arr] (dm) -- (ui);
\draw[arr, dashed, warn] (ui.west) -- ++(-3mm,0) |- (llm.east);
\end{tikzpicture}
\caption{Architecture of the layout optimizer. The darker boxes form the optimization loop and are deterministic.
The language model (orange) sits outside the loop: it only writes report prose, answers chat questions, or, if a
TypeSafe key is configured, advises on the final shortlist.}
\end{figure}"""


MATH = r"""
\subsection{The mathematics}
\label{sec:optmath}
\paragraph{Problem.} A site is a stack of $L$ levels of 1 ft cells. On each level a set $\Omega$ of cells is
\emph{placeable}; cores (stairs, elevators) and structure are fixed in the QA sites and generated for the hotel. The program
is a list of room instances $i = 1..N$, each with a type $t(i)$, nominal frontage $f_i$ and depth $d_i$ (area
$\bar a_i = f_i d_i$), a priority $p_i$, a set of allowed levels and flags (needs a window, wants a corner, wants to be near a
core). A layout chooses a corridor $C \subset \Omega$ (the same on every level, so floors stack) and an axis-aligned rectangle
for some of the rooms, such that rectangles do not overlap each other or $C$, lie inside $\Omega$, and each room has a door on
$C$. The goal is to maximise the fitness $F$ below.

\paragraph{Decoder: from a short genome to a full layout.} The search never moves rooms directly. It works on a genome
$g \in [0,1]^{44+N}$ (ten candidate spines with four genes each: two trims, a shift, an on/off gene; four connector genes;
one order key per room) and the decoder builds the layout:
\begin{enumerate}
\item \textbf{Module.} The bay width is the weighted median guest-room frontage,
$M = \operatorname{median}_w\{f_t\}$ with weights equal to the room counts; $D$ is the median guest-room depth and $w$ the
corridor width.
\item \textbf{Candidate spines.} Every row and column $k$ that can hold a straight corridor band gets the score
\[ s(k) = \sum_{x \in \text{band}(k)} \big[\, g_a(x) + g_b(x) + 1.5\,n_a(x) + 1.5\,n_b(x) + 0.5\,r_a(x) + 0.5\,r_b(x) \,\big], \]
where, for the two sides $a$ and $b$ of the band at position $x$, $g=1$ if a room at least $0.92\,d_{\min}$ deep fits,
$n=1$ if in addition the facade is about one room depth away (a window room), and $r=1$ if it is within the deepest room's
depth. So a line that is double-loaded with window rooms on both sides scores $2 \times (1 + 1.5 + 0.5) = 6$ per foot, a
single-loaded line 3, and a line through deep space with no facade 2. The best non-overlapping lines (at least $d_{\min}+w$
apart) become the ten candidates.
\item \textbf{Connectors.} Active spines are trimmed to their last room and joined to each other and to every core with
orthogonal L-shaped paths (never diagonal); pieces that stay unconnected are dropped.
\item \textbf{Bays.} Each side of each straight corridor run of length $\ell$ is cut into
$n = \lfloor \ell / M \rfloor$ bays. The remainder $\rho = \ell - nM$ is absorbed by the last bay if $\rho < 0.4M$ and becomes
a narrow bay otherwise. A bay has a window if its back wall is on the facade. Bays are built in two passes over all
corridors, window bays first and windowless bays in what is left, so the interior side of one corridor never takes
the facade strip that another corridor's window rooms need.
\item \textbf{Assignment.} Rooms are taken in the order of $0.06\,p_i + g_i$ (priority, then the order key) and each takes
\[ k_i = \max\!\Big(\operatorname{round}\frac{\bar a_i}{M D},\ \Big\lceil \frac{0.85 f_i}{M} \Big\rceil\Big) \]
consecutive free bays on one allowed level, the run with the highest score
\begin{align*} \sigma ={}& W_{\text{ext}}\,[\text{window}] - W_{\text{nw}}\,[\text{no window}] + W_{\text{corner}}\,[\text{corner}]
+ W_{\text{core}}\Big(1 - \tfrac{\min(\delta,150)}{150}\Big) + W_{\text{adj}}\,A \\
&+ W_{\text{stack}}\,\tfrac{\min(S,2)}{2} + W_{\text{grp}}\,[\text{same type next door}] - 1.2\max(0, o - 1.2),
\end{align*}
with $\delta$ the corridor distance from the door to the nearest core, $A$ the adjacency credit, $S$ the number of floors
with the same room type in the same bay, and $o$ the ratio of the bay area to the room's nominal area. Two hard rules apply:
a guest room is only placed where every bay has a window, and only if the run gives at least 85\% of its nominal frontage
and depth.
\end{enumerate}
After the assignment, each spine is cut back to what it serves (the doors of placed rooms and its junctions with
other corridors and cores), which removes corridor stubs and shortens dead ends. The dead end $E$ is the longest
corridor distance from any corridor cell to the nearest exit door or through-route between two exits.

Every decoded layout is therefore valid by construction (straight corridors, rooms perpendicular to them, no overlaps, all
rooms on the corridor), and the search only has to decide \emph{which} corridors exist and \emph{in what order} rooms claim
bays.

\paragraph{Fitness.} Ten objectives $f_k \in [0,1]$ (1 is best) are combined with weights $w_k$:
\[ F = \frac{\sum_k w_k f_k}{\sum_k w_k} \; - \; \tfrac12 \max\!\big(0,\ \min(n_c, 2) - c\big), \]
where $n_c$ is the number of cores and $c$ the number the corridor reaches (an exit that is not connected costs half the
scale). With $\rho$ the corridor share of the plate, $T$ the longest exit travel and $E$ the longest dead end:
\begin{align*}
f_{\text{place}} &= \frac{\sum_{i\ \text{placed}} p_i \min(a_i, \bar a_i)}{\sum_{i} p_i \bar a_i}, &
f_{\text{day}} &= \frac{\#\{\text{rooms needing a window that have one}\}}{\#\{\text{rooms needing a window}\}}, \\
f_{\text{eff}} &= \operatorname{clip}\!\Big(1 - \frac{\rho}{0.30}\Big), &
f_{\text{egress}} &= 1 - \tfrac12 \min\!\Big(1, \frac{(T - @@TRAVEL@@)^+}{@@TRAVEL@@}\Big) - \tfrac12 \min\!\Big(1, \frac{(E - @@DEAD@@)^+}{@@DEAD@@}\Big), \\
f_{\text{ada}} &= \min\!\Big(1, \frac{n_{\text{ADA}}}{n_{224.2}}\Big)\cdot\min\!\Big(1, \frac{\text{floors with ADA}}{\text{target}}\Big), &
f_{\text{core}} &= \operatorname{mean}_i \Big(1 - \frac{\min(\delta_i, 120)}{120}\Big), \\
f_{\text{adj}} &= \frac{\sum_j \omega_j \max\big(0,\ 1 - (\Delta_j - m_j)^+/m_j\big)}{\sum_j \omega_j}, &
f_{\text{corner}} &= \frac{\#\{\text{corner rooms on a corner}\}}{\#\{\text{corner rooms}\}}, \\
f_{\text{bal}} &= \operatorname{clip}\!\Big(1 - \frac{\operatorname{sd}(K_\ell)}{\operatorname{mean}(K_\ell)}\Big), &
f_{\text{comp}} &= \operatorname{clip}\!\Big(\frac{2 S_{\text{shared}}}{0.6\, P_{\text{rooms}}}\Big).
\end{align*}
Here $a_i$ is the drawn area (credit is capped at the nominal area, so an oversized room is not worth more),
$\Delta_j$ the distance between the two rooms of adjacency $j$ with allowed distance $m_j$, $K_\ell$ the keys on guest floor
$\ell$, $S_{\text{shared}}$ the total length of walls shared by two rooms and $P_{\text{rooms}}$ the total room perimeter.
The limits @@TRAVEL@@ ft (exit travel) and @@DEAD@@ ft (dead end) are read from RNGD's IBC rule CSV. The weights are
@@WEIGHTS@@. There is no language model in $F$: it is arithmetic on the geometry.

\paragraph{Upper bounds (can the keys fit at all?).} Two bounds hold for any layout, whatever the algorithm.
A corridor of width $w$ serves at most two room depths, so at least $w/(2D+w)$ of the used floor is corridor:
\[ K \le K_{\text{area}} = \Big\lfloor \frac{L\,|\Omega|\,\big(1 - \tfrac{w}{2D+w}\big)}{\bar a} \Big\rfloor . \]
If every guest room needs a window, the rooms of one floor share the facade length $P_f$:
\[ K \le K_{\text{window}} = L \Big\lfloor \frac{P_f}{\bar f} \Big\rfloor . \]
For a generated hotel plate with corridor length $R$ and cores taking a frontage $c_f$, the bound per floor is
$\lfloor (2R - c_f)/M \rfloor$. The bounds are loose (they ignore corners and the public rooms), but a request above them
is impossible, and a result close to them is close to the best any search can do.

\paragraph{Search.} With $m$ candidate spines the decoder is driven by a subset of them, so the first stage enumerates
\[ 1 + \sum_{k=1}^{4} \binom{\min(m,7)}{k} \le 1 + 7 + 21 + 35 + 35 = 99 \]
spine sets (all spines, and every set of up to four of the seven best), with rooms in priority order. The second stage
starts from the three best distinct layouts and repeats: generate the neighbours (switch one spine on or off: 10 moves; trim
either end or shift an active spine: 4 moves each; new connector elbows; four room-order nudges), evaluate them in a seeded
random order, and accept the first one with $F' > F + 10^{-6}$. When all three starts are local optima, the rest of the budget is spent on iterated local search: the best
layout is ``kicked'' with two moves at once, improved again by the same pattern search, and kept only if it beats
the best so far. The total budget is $PG$ evaluations ($P$ = population, $G$ = generations, kept as the budget
controls), the same as the GA's. The genetic algorithm (variant C)
evaluates $P(G+1)$ layouts plus local-search steps, here about @@GAEVALS@@. Crossover is not used in the systematic search:
splicing the corridor genes of one plan with the room order of another seldom gives a better plan, because the value of a
corridor depends on which other corridors exist.

\paragraph{Plate generation (hotel).} For $K$ keys on $F$ guest floors the busiest level needs net area $A$; with slack
$\varsigma = 1.04$, core area $A_c$ and packing efficiency $\eta = 0.93$ the corridor length to provide is
\[ R = \frac{\varsigma A + A_c}{2 D \eta}. \]
Each shape with bounding box $a \times b$ and wing depth $W_d = 2D + w$ (58 ft) offers
$R_{\text{Bar}} = a$, $R_{\text{L}} = R_{\text{T}} = a + b - W_d$, $R_{\text{U}} = a + 2b - 2W_d$,
$R_{\text{H}} = a + 2b - 2W_d$ and $R_{\text{Courtyard}} = 2a + 2b - 4W_d$; the generator solves $R_{\text{shape}}(a, \beta a) \ge R$
for the smallest $a$ at aspect ratios $\beta \in \{0.6, 1\}$ and clips the result to the buildable area in both
orientations. The elevator count is $e = \max(2, \lceil K/75 \rceil)$, and the two stairs of the ``centre core'' scheme are
the pair of wing-end positions with the largest distance between them (exit separation, short dead ends).

\paragraph{Decision.} Each layout $j$ has a vector of twelve criteria $x_{j1..12}$ (keys, gross SF per key, facade per key,
height, corridor share, dead end, code checks passed and others). Layout $j$ is \emph{dominated} if another layout is at
least as good on every criterion and strictly better on one; the rest form the Pareto front. TOPSIS then ranks all layouts:
\[ r_{jq} = \frac{x_{jq}}{\sqrt{\sum_j x_{jq}^2}}, \quad v_{jq} = \frac{\lambda_q}{\sum_q \lambda_q}\, r_{jq}, \quad
d_j^{\pm} = \sqrt{\sum_q (v_{jq} - v_q^{\pm})^2}, \quad C_j = \frac{d_j^-}{d_j^+ + d_j^-} \in [0,1], \]
where $v_q^+$ is the best and $v_q^-$ the worst value of criterion $q$ over all layouts and $\lambda_q$ the weight from the
chosen preset (Balanced, Most keys, Lowest cost, Guest experience, Low-rise) or the sliders. The layout with the largest
$C_j$ is ranked first.
"""


def section(study: dict | None) -> str:
    if not study:
        return (r"\section{Layout optimization}The optimization study has not been run "
                r"(\texttt{python src/build\_optimizer\_study.py}).\par")
    sites, hotel = study["sites"], study["hotel"]
    tags = [t for t in "ABCD" if t in sites[0]["variants"]]
    label = {"A": "A Reference GA", "B": "B GA, free-form", "C": "C Updated GA", "D": "D Systematic"}

    def v(s, t, k):
        return s["variants"][t][k]

    # ------------------------------------------------------------ comparison table
    rows = []
    for s in sites:
        for t in tags:
            x = s["variants"][t]
            rows.append([_esc(s["site"].replace("_multi", "")) if t == tags[0] else "", label[t],
                         f"{x['rooms_placed']}/{x['rooms_total']}", str(x["windowless_guest_rooms"]),
                         f"{x['fitness']:.3f}", f"{x['corridor_ratio']:.3f}", f"{x['unused_share']:.2f}",
                         f"{x['dead_end_ft']:.0f}", f"{x['checks_pass']}/{x['checks_total']}",
                         f"{x['evaluations']:,}" if x.get("evaluations") else f"{x.get('generations', '-')} gen.",
                         f"{x['seconds']:.0f} s" if x.get("seconds") else "n/a"])
        rows[-1][-1] += r" \\ \midrule" if s is not sites[-1] else ""
    comp = _table(["Site", "Variant", "Rooms", "No window", "Fitness", "Corridor", "Unused", "Dead end ft", "Checks",
                   "Evaluated", "Time"], [[c for c in r] for r in rows],
                  "p{0.8cm}p{2.9cm}p{1.1cm}p{1.2cm}p{1.1cm}p{1.2cm}p{1.1cm}p{1.3cm}p{1.0cm}p{1.5cm}p{1.0cm}").replace(
        r" \\ \midrule \\", r" \\ \midrule")
    brows = []
    for s in sites:
        b = s["bounds"]
        brows.append([_esc(s["site"].replace("_multi", "")), f"{b['placeable_sf_per_level']:,}", f"{b['program_area_sf']:,}",
                      f"{b['program_over_placeable']:.2f}", f"{b['facade_ft_per_level']:,}", f"{b['mean_key_frontage_ft']}",
                      str(b["area_bound_keys"]), str(b["window_bound_keys"]), str(b["keys_requested"]),
                      str(v(s, "A", "keys_placed")), str(v(s, "D", "keys_placed"))])
    bounds = _table(["Site", "Placeable SF / level", "Program SF", "Program / placeable", "Facade ft / level",
                     "Mean key frontage ft", "$K_{\\text{area}}$", "$K_{\\text{window}}$", "Keys asked", "Keys A", "Keys D"],
                    brows, "p{0.8cm}p{1.6cm}p{1.3cm}p{1.5cm}p{1.4cm}p{1.6cm}p{1.0cm}p{1.2cm}p{1.0cm}p{0.9cm}p{0.9cm}")
    tot = {t: sum(v(s, t, "rooms_placed") for s in sites) for t in tags}
    nowin = {t: sum(v(s, t, "windowless_guest_rooms") for s in sites) for t in tags}
    secs = {t: sum(v(s, t, "seconds") or 0 for s in sites) for t in tags if t != "A"}
    evals = {t: sum(v(s, t, "evaluations") or 0 for s in sites) for t in tags if t != "A"}
    fit = {t: sum(v(s, t, "fitness") for s in sites) / len(sites) for t in tags}
    fails_a = sorted({f for s in sites for f in v(s, "A", "failed_checks")})
    gens_a = sites[0]["variants"]["A"].get("generations", 500)
    det = all(s.get("deterministic_repeat") for s in sites)
    per_site = "; ".join(f"{_esc(s['site'].replace('_multi', ''))}: {v(s, 'A', 'rooms_placed')} $\\to$ "
                         f"{v(s, 'D', 'rooms_placed')} rooms, {v(s, 'A', 'windowless_guest_rooms')} $\\to$ "
                         f"{v(s, 'D', 'windowless_guest_rooms')} windowless, fitness {v(s, 'A', 'fitness'):.3f} $\\to$ "
                         f"{v(s, 'D', 'fitness'):.3f}" for s in sites)
    b_line = ""
    if "B" in tags:
        b_line = (f" Variant B, the first attempt, kept the reference idea (a GA that moves free corridor segments and packs "
                  f"rooms) and placed {tot['B']} rooms in total with {nowin['B']} windowless guest rooms in "
                  f"{secs['B']:.0f} s: better scoring alone did not fix the plans, which is why the representation was changed.")
    fewer = [s for s in sites if v(s, "D", "rooms_placed") < v(s, "A", "rooms_placed")]
    fewer_txt = ""
    if fewer:
        fewer_txt = (" On " + ", ".join(_esc(s["site"].replace("_multi", "")) for s in fewer)
                     + " the updated optimizer places \\emph{fewer} rooms than the reference. That is the window rule at work: "
                       "the reference count includes guest rooms with no facade, which the updated optimizer refuses to place "
                       "(the rule can be switched off in the app, and the window bound $K_{\\text{window}}$ in "
                       "Table~\\ref{tab:bounds} shows how many window rooms the facade allows).")

    problems = _table(
        ["Observed problem", "Cause", "What was changed", "Evidence in this run"],
        [["Guest rooms with no window; corridors loaded on one side only (review in the 10/08 meeting).",
          "The corridor genome places segments anywhere; rooms are packed wherever they touch a corridor, so the distance "
          "to the facade is left to chance.",
          "Decoder: corridor spines are chosen from lines scored for double-loaded window rooms (score $s(k)$); guest "
          "rooms may only take bays with a window (hard rule).",
          f"Windowless guest rooms over the three sites: {nowin['A']} (A) $\\to$ {nowin['D']} (D)."],
         ["Floor area left unused; rooms rotated or isolated.",
          "Rooms are placed one by one as free rectangles, so leftovers between rooms cannot be used.",
          "Modular bays of one room frontage $M$ along the whole corridor; rooms take whole bays, the last bay absorbs the "
          "remainder; unused bays are reported as flex space.",
          "Unused share and corridor share per variant in Table~\\ref{tab:optcompare}."],
         ["Not clear whether the requested keys can fit at all.",
          "No bound was computed; the GA simply reports what it placed.",
          "Area and window upper bounds, shown next to every result.",
          "Table~\\ref{tab:bounds}."],
         ["Crossover produces plans that do not make sense; slow convergence.",
          "Half of one plan spliced onto half of another: the value of a corridor depends on the other corridors.",
          "Search over a small structured space: enumerate spine sets, then first-improvement pattern search. The GA is "
          "kept as an option and now runs on the same decoder.",
          f"At the same budget D is within {max(abs(v(s, 'C', 'fitness') - v(s, 'D', 'fitness')) for s in sites):.3f} "
          f"fitness of the GA (C) with no crossover and no population: {tot['D']} against {tot['C']} rooms. Most of the "
          "gain over A comes from the decoder, not from the search."],
         ["With crossing corridors, facade strips stayed empty (seen on the square site fp2 in the first version of the "
          "new decoder: 39 rooms).",
          "Bays were built corridor by corridor, so the windowless interior side of the first corridor took the strip "
          "that the next corridor's window rooms needed.",
          "Two passes over all corridors: window bays first, windowless bays in what is left.",
          f"fp2: {v(sites[1], 'D', 'rooms_placed')} rooms (D) and {v(sites[1], 'C', 'rooms_placed')} (C), all guest rooms "
          "with a window."],
         ["Corridor stubs beyond the last room; dead ends over the 50 ft limit.",
          "Spines were cut to the bays they could serve, not to the rooms actually placed.",
          "After the assignment each spine is cut back to the doors and junctions it serves; the dead end is measured to "
          "the nearest exit door or through-route.",
          f"Longest dead end, D: {max(v(s, 'D', 'dead_end_ft') for s in sites):.0f} ft; A: "
          f"{max(v(s, 'A', 'dead_end_ft') for s in sites):.0f} ft (same measure)."],
         ["Runs take minutes; results change between runs.",
          f"{gens_a} generations in the reference; an LLM reviewer inside the fitness in the earlier version of this app "
          "(about 10 minutes, cloud calls).",
          "No LLM in the fitness; deterministic seeds; cached static scores in the assignment; plates evaluated in parallel "
          "processes.",
          f"Variant D: {secs['D']:.0f} s for the three sites; identical result on a repeat run: {'yes' if det else 'no'}."],
         ["Egress and accessibility not part of the score.",
          "The reference score rewards placed rooms and contiguity.",
          "Ten objectives including exit travel, dead ends, ADA Table 224.2 count and dispersion, core access; limits read "
          "from RNGD's IBC CSV.",
          ("Reference designs fail: " + ", ".join(_esc(f) for f in fails_a) + " (same evaluator)." if fails_a
           else "All variants pass the checks on these sites; the checks are now reported for every layout.")],
         ["The shape, the core and the floor count are inputs.",
          "The grid fixes the L-shaped plate and both cores.",
          "Hotel Schematic: plates (six shapes), core schemes (three) and the floor count are generated and searched; a "
          "dropdown lists the top or all layouts.",
          f"{hotel['layouts']} distinct layouts in {hotel['seconds']:.0f} s (Section~\\ref{{sec:opthotel}})."]],
        "p{3.3cm}p{3.7cm}p{4.9cm}p{3.6cm}")

    # ------------------------------------------------------------ hotel tables
    top_rows = [[str(r["rank"]), _esc(r["layout"]), str(r["floors"]), f"{r['score']:.3f}", "yes" if r["pareto"] else "no",
                 str(r["keys"]), f"{r['fitness']:.3f}", f"{r['corridor_ratio']:.3f}", f"{r['gross_sf_per_key']:.0f}",
                 f"{r['height_ft']:.0f}", f"{r['dead_end_ft']:.0f}", f"{r['code_pass']:.2f}"] for r in hotel["top"]]
    top = _table(["\\#", "Layout", "Guest floors", "TOPSIS $C_j$", "Pareto", "Keys", "Fitness", "Corridor", "SF / key",
                  "Height ft", "Dead end ft", "Code"], top_rows,
                 "p{0.4cm}p{4.2cm}p{1.0cm}p{1.2cm}p{0.9cm}p{0.8cm}p{1.0cm}p{1.1cm}p{0.9cm}p{0.9cm}p{1.0cm}p{0.8cm}")
    shape_rows = [[_esc(r["shape"]), _esc(r["layout"]), str(r["floors"]), str(r["rank"]), str(r["keys"]),
                   f"{r['gross_sf_per_key']:.0f}", f"{r['facade_ft_per_key']:.1f}", f"{r['dead_end_ft']:.0f}",
                   f"{r['code_pass']:.2f}"] for r in hotel["best_by_shape"]]
    shapes = _table(["Shape", "Best layout of that shape", "Guest floors", "Overall rank", "Keys", "SF / key",
                     "Facade ft / key", "Dead end ft", "Code"], shape_rows,
                    "p{1.7cm}p{4.6cm}p{1.1cm}p{1.2cm}p{0.9cm}p{1.1cm}p{1.4cm}p{1.2cm}p{0.9cm}")
    preset_rows = [[_esc(k), _esc(p["layout"]), str(p["floors"]), str(p["keys"]), f"{p['gross_sf_per_key']:.0f}",
                    f"{p['height_ft']:.0f}", f"{p['code_pass']:.2f}"] for k, p in hotel["presets"].items()]
    presets = _table(["Decision priority", "Layout ranked first", "Guest floors", "Keys", "SF / key", "Height ft", "Code"],
                     preset_rows, "p{2.8cm}p{5.2cm}p{1.3cm}p{1.0cm}p{1.2cm}p{1.3cm}p{1.0cm}")
    b = hotel["best"]
    bm = b["metrics"]
    check_rows = [[_esc(c["rule_id"]), _esc(c["check"]), _esc(c["required"]).replace("<=", r"$\le$").replace(">=", r"$\ge$"),
                   _esc(c["provided"]), r"\textcolor{" + ("pass" if c["status"] == "PASS" else "fail") + r"}{\textbf{"
                   + c["status"] + "}}"] for c in bm["checks"]]
    checks = _table(["Rule", "Check", "Required", "Provided", "Status"], check_rows,
                    "p{1.9cm}p{4.6cm}p{4.6cm}p{2.4cm}p{1.3cm}")
    unplaced = ", ".join(f"{_esc(k)} x{n}" for k, n in bm["unplaced_by_type"].items()) or "none"
    refine = "; ".join(f"{_esc(t['plate'])} ({t['floors']} fl.) {t['start']:.3f} $\\to$ {t['end']:.3f}"
                       for t in hotel["refine"][:4])
    weights = ", ".join(f"{k.replace('_', ' ')} {w:g}" for k, w in hotel["weights"].items())
    lim = study["limits"]
    math = (MATH.replace("@@TRAVEL@@", f"{lim['travel_ft']:.0f}").replace("@@DEAD@@", f"{lim['dead_end_ft']:.0f}")
            .replace("@@WEIGHTS@@", weights).replace("@@GAEVALS@@", f"{evals['C'] // len(sites):,}"))
    site_figs = "\n".join(_fig(f"compare_fp{i}.png",
                               f"Site fp{i}, Level 2, same program and same evaluator. Blue lines are window walls; a red "
                               "dashed outline marks a guest room with no window; hatched bays are unused.", "0.80")
                          for i in (1, 2, 3))

    return rf"""
\section{{Layout optimization: from the reference GA to the updated optimizer}}\label{{sec:opt}}
\subsection{{Summary}}
RNGD's design-optimization prototype (B\"u\c{{s}}ra's genetic algorithm, files in \texttt{{QA/}}) takes a multi-level 1 ft
site grid and a room program and evolves a corridor and a room placement for {gens_a} generations. This project rebuilt
that optimizer in four steps and measured every step on the same three sites with the same program and the same evaluator
(Table~\ref{{tab:optcompare}}):
\begin{{itemize}}
\item \textbf{{A, the reference GA}} as delivered: {tot['A']} rooms placed over the three sites, {nowin['A']} of the guest rooms without a window.
\item \textbf{{C, the updated GA}}: a memetic GA on a new architectural decoder, no LLM in the fitness: {tot['C']} rooms, {nowin['C']} without a window, mean fitness {fit['C']:.3f} against {fit['A']:.3f}.
\item \textbf{{D, systematic search}} (the default now): the same decoder, driven by enumeration and pattern search instead of crossover: {tot['D']} rooms, {nowin['D']} without a window, mean fitness {fit['D']:.3f}, in {secs['D']:.0f} s for all three sites and with the same answer on every run.
\end{{itemize}}
Per site (A $\to$ D): {per_site}.{b_line}{fewer_txt}

The same engine now also runs inside \textbf{{Hotel Schematic}}, where the shape of the building, the position of the
core and stairs and the number of floors are generated instead of given (Section~\ref{{sec:opthotel}}).

\subsection{{Architecture}}
{ARCH}
{{\small\begin{{longtable}}{{p{{0.5cm}}p{{3.6cm}}p{{3.8cm}}p{{7.6cm}}}}\toprule
\textbf{{\#}} & \textbf{{Component}} & \textbf{{File}} & \textbf{{What it does}} \\ \midrule \endhead
1 & Plate generator & \texttt{{optimize/typology.py}} & Turns the hotel program into rooms with BaaP sizes; generates Bar, L, U, T, H and Courtyard plates inside the buildable area, with the elevator core and stairs placed by scheme (ends, centre, knuckle); one site grid per plate. \\
2 & Decoder & \texttt{{optimize/planner.py}} & Genome $\to$ corridor spines $\to$ orthogonal connectors $\to$ modular bays $\to$ rooms assigned to bays. Every output is a valid plan. \\
3 & Evaluator & \texttt{{optimize/metrics.py}} & Ten objectives in $[0,1]$, the code checks (exits, exit separation, travel, corridor width, dead end, ADA; stories for the hotel) and the fitness $F$. Limits come from RNGD's IBC CSV. \\
4 & Search & \texttt{{optimize/search.py}}, \texttt{{explore.py}} & Systematic enumeration of spine sets plus pattern search (default) or the memetic GA; for the hotel, all plates are screened in parallel processes and the best of each shape refined. \\
5 & Archive & \texttt{{optimize/explore.py}} & Keeps every distinct layout with its objectives and criteria. \\
6 & Decision & \texttt{{optimize/decide.py}} & Pareto front and TOPSIS ranking over twelve criteria; optional Jev advisor (TypeSafe) on the shortlist only. \\
7 & Views & \texttt{{optimize/plan\_view.py}}, \texttt{{hotel\_opt\_ui.py}}, \texttt{{optimizer\_ui.py}} & Dropdown of the top or all layouts, furnished interactive plans, 3D massing, comparison charts, JSON export. \\
-- & Reference loader & \texttt{{optimize/baseline.py}} & Rebuilds the reference designs from \texttt{{QA/floorplan*.json}} so they are scored by the same evaluator (their shared-wall lengths reproduce the values stored in the files). \\
-- & Report writer & \texttt{{optimize/report.py}} & One LLM call (\texttt{{{_esc(study.get('report_model', 'anthropic/claude-sonnet-5.5'))}}} through OpenRouter) writes the prose of the per-run PDF from the computed data and retrieved code passages; every number in its text is traced back to the data. \\
\bottomrule\end{{longtable}}}}
Only the last row uses a language model. Steps 1 to 6 are plain Python on NumPy, SciPy and scikit-image arrays, so the
same inputs always give the same layouts and the same ranking.

\subsection{{What was wrong, and how it was solved}}
The problems below were seen in the reference results, in the first version of this app's optimizer (which followed the
same idea and added an LLM reviewer to the fitness) and in the team review of 10/08/2026.
{problems}

{math}

\subsection{{Results on the three QA sites}}\label{{sec:optqa}}
All variants use \texttt{{QA/rooms\_hotel.json}} (72 rooms on three levels) and are scored by the same evaluator. Variants
B, C and D were run with population {study['population']} and {study['generations']} generations as the budget, seed 7, no LLM.
For A the search time is not stored in the files; its history holds {gens_a} generations.
\begingroup\captionof{{table}}{{Reference GA and the three later variants. ``No window'' counts guest rooms without a facade wall; ``Unused'' is the share of placeable area that is neither room nor corridor; ``Checks'' are the code checks passed.}}\label{{tab:optcompare}}\endgroup
{comp}
\begingroup\captionof{{table}}{{Upper bounds on the number of keys (Section~\ref{{sec:optmath}}) against what was asked and placed.}}\label{{tab:bounds}}\endgroup
{bounds}
{_fig('compare_bars.png', 'The four variants on the three sites (same program, same evaluator).', '0.30')}
{_fig('convergence.png', 'Left: rooms placed by the reference GA over its generations (from the history in the delivered files). Right: best rooms placed against layouts evaluated for the later variants.', '0.30')}
{site_figs}

\subsection{{Results in Hotel Schematic (generative shape, core and floors)}}\label{{sec:opthotel}}
Input: the Project Renegade Living + Lodging program ({hotel['keys_requested']} keys) and a buildable area of
{hotel['envelope'][0]} x {hotel['envelope'][1]} ft. The search screened {hotel['variants']} plate and floor-count variants,
evaluated {hotel['evaluations']:,} layouts and kept {hotel['layouts']} distinct ones in {hotel['seconds']:.0f} s on eight
processes, with no LLM calls; a second run returned the same list: {'yes' if hotel['deterministic_repeat'] else 'no'}.
{hotel['pareto']} layouts are on the Pareto front. Pattern search improved the screened plates as follows:
{refine}.
\begingroup\captionof{{table}}{{Top ten layouts under the Balanced decision weights.}}\endgroup
{top}
\begingroup\captionof{{table}}{{Best layout of each shape (Balanced weights).}}\endgroup
{shapes}
\begingroup\captionof{{table}}{{The first-ranked layout changes with the decision priority: the trade-off is the user's, the search only supplies the options.}}\endgroup
{presets}
\textbf{{First-ranked layout: {_esc(b['plate'])}, {b['floors']} guest floors.}} Footprint {b['size'][0]} x {b['size'][1]} ft
({b['footprint_sf']:,} SF), {b['elevators']} elevators, bay module {b['module_ft']:.0f} ft. It places
{bm['keys_placed']} of {bm['keys_requested']} keys and {bm['rooms_placed']} of {bm['rooms_total']} program rooms; the
frontage bound is {b['upper_bound_keys']} keys per typical floor. Corridor share {bm['corridor_ratio']:.3f}, longest exit
travel {bm['max_travel_ft']:.0f} ft, dead end {bm['dead_end_ft']:.0f} ft, fitness {bm['fitness']:.3f}. Not placed: {unplaced}.
{checks}
{_fig('hotel_gallery.png', 'The eight first-ranked layouts (typical guest floor). Dark red: core and stairs; blue: corridor.', '0.45')}
{_fig('hotel_pareto.png', 'Every layout found: keys placed against gross area per key. Outlined points are on the Pareto front.', '0.34')}
{_fig('hotel_best_L2.png', 'First-ranked layout, Level 2, as drawn in the app: furnished rooms, door swings, windows (blue), stairs and elevators.', '0.40')}
{_fig('hotel_best_L1.png', 'First-ranked layout, Level 1 (public and back-of-house rooms).', '0.40')}
{_fig('hotel_best_3d.png', 'First-ranked layout, stacked massing with the floors pulled apart.', '0.38')}

\subsection{{App screenshots}}
{_shot_figs()}

\subsection{{What is still open}}
\begin{{itemize}}
\item Level 1 uses the same plate as the guest floors (no wider podium), so large ground-floor rooms can be left out; they are listed for every layout.
\item The decoder draws rectangles on a 1 ft grid: angled wings, curved facades and structural grids are not modelled.
\item The upper bounds are loose; a tight bound would need an exact packing model.
\item The centre-core scheme uses two stairs; on T, U and H plates the third wing can exceed the dead-end limit. The check flags it and the ranking moves such layouts down.
\item Parking, cost and energy are not in the fitness. Facade length per key and gross area per key stand in for cost in the decision step.
\item Jev (TypeSafe's hosted decision model) is wired only as an optional advisor; no key was configured for this run.
\end{{itemize}}
"""
