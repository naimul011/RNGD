"""Builds reports/hotel/hotel_schematic_report.pdf: program, schematic plans, code checks,
assumptions and open questions. Uses the default program unless a CSV/XLSX path is given.

Usage: python src/build_hotel_report.py [program.csv|program.xlsx]
"""
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rngd.hotel import layout  # noqa: E402
from rngd.hotel.program import default_program, load_program_file  # noqa: E402

OUT = ROOT / "reports" / "hotel"
OUT.mkdir(parents=True, exist_ok=True)
TECTONIC = ROOT / "Testt" / "tools" / "tectonic.exe"
COLORS = {"guest": "#8FB8DE", "core": "#7A2E4D", "corridor": "#DDDDDD"}


def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("<=", r"$\le$"), (">=", r"$\ge$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}")]:
        s = s.replace(a, b)
    return s


def plan_png(lvl, path):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for r in lvl.rooms:
        ax.add_patch(Rectangle((r.x, r.y), r.w, r.d, fc=COLORS.get(r.kind, "#FFFFFF"), ec="black", lw=0.6))
        if r.kind == "guest" and r.w > 12:
            ax.text(r.x + r.w / 2, r.y + r.d / 2, r.label.replace(" Suite", "\nSuite").replace(" Spa", "\nSpa")
                    .replace(" Corner", "\nCorner"), ha="center", va="center", fontsize=4.2)
        if r.kind == "core":
            ax.text(r.x + r.w / 2, r.y + r.d / 2, "CORE\n2 stairs\n+ elev.", ha="center", va="center", fontsize=5, color="white")
        if r.kind == "corridor":
            ax.text(r.x + r.w / 2, r.y + r.d / 2, "CORRIDOR", ha="center", va="center", fontsize=5)
    ax.set_xlim(-4, lvl.length_ft + 4)
    ax.set_ylim(lvl.width_ft + 4, -4)
    ax.set_aspect("equal")
    ax.set_title(f"{lvl.name} schematic: {lvl.keys_placed} keys", fontsize=9)
    ax.set_xlabel("ft (core at left)", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def site_png(sch, path):
    fig, ax = plt.subplots(figsize=(4.2, 5.2))
    env = sch.site
    ax.add_patch(Rectangle((0, 0), env["envelope_w"], env["envelope_d"], fill=False, ls="--", ec="#CC0000"))
    ax.add_patch(Rectangle((10, 15), sch.levels[1].length_ft, sch.levels[1].width_ft, fc="#8FB8DE", ec="black"))
    ax.set_xlim(-10, env["envelope_w"] + 10)
    ax.set_ylim(env["envelope_d"] + 10, -10)
    ax.set_aspect("equal")
    ax.set_title("Typical wing footprint in buildable envelope", fontsize=8)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build(program):
    site = {"envelope_w": 170, "envelope_d": 265}
    sch = layout.build(program, site)
    site_png(sch, OUT / "site.png")
    for lv in sch.levels[1:]:
        plan_png(lv, OUT / f"{lv.name.lower().replace(' ', '_')}.png")
    n = program.net_to_gross(33, 8)
    s = program.net_to_gross(0, 0)

    prog_rows = "\n".join(f"{esc(t)} & {c} & {a:,.0f} & {esc(cat)} \\\\" for t, c, a, cat in program.rows)
    gross_rows = "\n".join(f"{esc(k)} & {v:,.0f} \\\\" for k, v in program.gross_by_level.items())
    check_rows = "\n".join(f"{esc(c['check'])} & {esc(c['required'])} & {esc(c['provided'])} & {c['status']} \\\\" for c in sch.checks)
    plan_figs = "\n".join(
        rf"\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{{lv.name.lower().replace(' ', '_')}.png}}"
        rf"\caption{{{esc(lv.name)}: {lv.keys_placed} guest keys, wing {lv.length_ft:.0f} x {lv.width_ft:.0f} ft.}}\end{{figure}}"
        for lv in sch.levels[1:]
    )

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.9in]{{geometry}}
\usepackage{{graphicx,booktabs,float,longtable,xcolor}}
\setlength{{\parskip}}{{0.4em}}\setlength{{\parindent}}{{0pt}}
\title{{Hotel Schematic Proposal\\\large Program, planning-level layout, code screen and open questions}}
\author{{RNGD side project --- prototype output, not for construction}}
\date{{\today}}
\begin{{document}}
\maketitle

\section{{Summary}}
Program: {program.keys} guest keys in {len(program.rows)} rows, {program.net_sf:,.0f} SF net shared/BOH/guest area,
across six levels. Source gross area: {program.gross_sf:,.0f} SF. Planning-level schematic: five upper levels of
double-loaded corridor, each with a core at the west end. The schematic fails two screens from RNGD's
own IBC rule set: six stories above grade (rule limit five) and a dead-end corridor of about
{sch.max_travel_ft:.0f} ft (limit 50 ft). These are design findings, not code determinations.

\section{{Program}}
\begin{{longtable}}{{p{{7.2cm}}rrl}}\toprule
Room type & Count & Area (SF) & Category \\ \midrule
\endhead
{prog_rows}
\bottomrule
\end{{longtable}}
Source: {esc(program.source)}. Guest-room areas are type averages (type total divided by count).
Source subtotals reconcile to 59,350 SF; the table sums to {program.net_sf:,.0f} SF because of one-square-foot rounding in the source.

\section{{Gross area and net-to-gross}}
\begin{{tabular}}{{lr}}\toprule
Level & Gross SF (source) \\ \midrule
{gross_rows}
\midrule Total & {program.gross_sf:,.0f} \\ \bottomrule
\end{{tabular}}

Source ratio gross/net: {program.gross_sf / program.net_sf:.2f}. Meeting-derived rule (33\% circulation, 8\% wall burden, rounded up to 100 SF):
{n['required_gross_sf']:,.0f} SF. These rules were stated in the kickoff meeting, not confirmed by RNGD, and are shown as a comparison only.

\section{{Schematic}}
Envelope: {site['envelope_w']} x {site['envelope_d']} ft (synthetic, pending the real lot). Each upper level is one double-loaded wing:
rooms on both sides of a 6-ft corridor, room depth 26 ft, width from the type area, core (two stairs and elevators) at the west end.
Guest-room types are placed in the order of area, so suites are spread across levels.

\begin{{figure}}[H]\centering\includegraphics[width=0.45\linewidth]{{site.png}}\caption{{Wing footprint in the buildable envelope.}}\end{{figure}}
{plan_figs}

\section{{Code and program screen}}
Checks use the IBC rule subset in RNGD's rule database (2021 IBC, Type IIA sprinklered R-1 hotel). The overall planning screen is not a code determination.
\begin{{longtable}}{{p{{4.2cm}}p{{4.6cm}}p{{3cm}}l}}\toprule
Check & Required & Provided & Status \\ \midrule
\endhead
{check_rows}
\bottomrule
\end{{longtable}}

\section{{Assumptions}}
\begin{{itemize}}
\item Synthetic lot and envelope (170 x 265 ft). The real site and setbacks are not yet supplied.
\item Corridor double-loaded, 6 ft clear; room depth 26 ft; core 34 x 24 ft; two stairs per core.
\item Ground level shows program area only (no room layout). Parking is not modeled.
\item Accessible rooms, fire lane and fire access are not modeled. Code parameters from the R-1 rule set only.
\item Keys are split evenly across five upper levels in order of area.
\end{{itemize}}

\section{{Open questions}}
\begin{{enumerate}}
\item \textbf{{Which area of the Level 1 and Level 2 drawing was meant?}} The reference coordinate note
(scale 1.08 to original image) did not identify a room, stair or gridline. Please mark the spot.
\item Confirm circulation (33\%) and wall burden (8\%) percentages, or give RNGD's own factors.
\item Confirm the real lot geometry, setbacks and fire-lane requirements.
\item Confirm story limit: the prototype shows six levels above grade, and the R-1 rule set allows five for Type IIA.
\item Confirm dead-end corridor treatment: a second core or a shorter corridor.
\item Confirm whether the program should include parking and which ADA room counts apply.
\end{{enumerate}}

\section{{Sources}}
\begin{{itemize}}
\item Project Renegade Living + Lodging Prototype, cover sheet and plans, 06/20/25 (Scheme 2 folder).
\item RNGD IBC 2021 R1 five-story rule database (docs/RNGD\_IBC\_2021\_R1\_Five\_Story\_AI\_Rules\_v0\_2.csv).
\item Kickoff meeting transcript (circulation and wall-burden rules; egress and column-grid discussion).
\item RNGD Product Requirements Document (hotels as first use case).
\end{{itemize}}
\end{{document}}
"""
    (OUT / "hotel_schematic_report.tex").write_text(tex, encoding="utf-8")
    r = subprocess.run([str(TECTONIC), "hotel_schematic_report.tex"], cwd=OUT, capture_output=True, text=True)
    print(r.stderr[-1500:])
    return r.returncode


if __name__ == "__main__":
    prog = load_program_file(sys.argv[1]) if len(sys.argv) > 1 else default_program()
    sys.exit(build(prog))
