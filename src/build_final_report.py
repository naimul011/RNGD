"""Final schematic proposal (PDF) in the style of RNGD's Project Renegade L+L drawing set.

Order: front matter (summary, sheet index) -> drawing set (18 x 12 sheets: overview,
code summary, site, plans, elevations, enlarged unit plans, unit 3D views) ->
appendices (program, net-to-gross, code checks, RNGD IBC rule CSV, RNGD workbook
sheets, assumptions, open questions, sources).

Usage: python src/build_final_report.py [program.csv|program.xlsx]
Output: reports/final/RNGD_final_schematic_report.pdf, sheets as PDF + PNG in reports/final/sheets/
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rngd.hotel import drawings, layout  # noqa: E402
from rngd.hotel.code_check import checked_rule_ids  # noqa: E402
from rngd.hotel.program import GROUP_TITLES, MEETING_RULES, default_program, load_program_file  # noqa: E402

OUT = ROOT / "reports" / "final"
TECTONIC = ROOT / "Testt" / "tools" / "tectonic.exe"
RULES_CSV = ROOT / "docs" / "RNGD_IBC_2021_R1_Five_Story_AI_Rules_v0_2.csv"
WORKBOOK = ROOT / "docs" / "rngd_ibc_2021_r1_five_story_ai_ready_v0_2.xlsx"
DEFAULT_SITE = {"envelope_w": 170, "envelope_d": 265}

OPEN_QUESTIONS = [
    "Which area of the Level 1 / Level 2 drawing was meant? The reference coordinate note (scale 1.08 to the "
    "original image) did not name a room, stair or gridline. Please mark the spot on the L+L plans sheet.",
    "Story limit: the program needs Level 1 plus five guest floors (six stories), but RNGD rule IBC-R1-007 allows "
    "five stories for sprinklered Type IIA R-1. Is a Type IA podium (IBC 510.2) intended, as in the L+L prototype?",
    "Confirm the gross-up rule: kickoff meeting said 33% circulation + 8% wall burden; the L+L source sheets imply 1.25.",
    "Real lot, setbacks, street frontage and fire-lane requirements (the envelope used here is synthetic).",
    "Is parking part of this phase? It is not modelled.",
    "Confirm the per-floor support rooms (Ice, IT, Jan, Supplies) and the Level 1 BOH arrangement.",
    "Column grid: 30-34 ft bays assumed from the kickoff meeting. Is there a preferred structural module?",
    "Prefab panel / module catalog to replace the indicative panel breakdown on the elevations.",
]
ASSUMPTIONS = [
    "Envelope from the app sidebar (default synthetic lot 170 x 265 ft). Street assumed on the south side.",
    "Double-loaded corridor, 6 ft clear; room depth 26 ft; room width = program area / 26 ft.",
    "Floor-to-floor: Level 1 16 ft, guest floors 10.75 ft (from the L+L elevations).",
    "Core at the west end (Stair 1, two elevators, shafts); Stair 2 at the east end of the north side.",
    "Rows whose count equals the number of guest floors (Ice, IT, Jan, Supplies) are placed once per floor.",
    "Room mix spread evenly across floors, largest units first. Fixtures and furniture are typical, not from RNGD unit drawings.",
    "Occupant loads use the RNGD CSV factors (IBC Table 1004.5); lobby as assembly unconcentrated (15 SF net).",
]
SOURCES = [
    "Project Renegade Living + Lodging Prototype drawings, cover sheet space allocation and plans (06/20/25).",
    "RNGD Model Training data: ADA Suite, Corner Suite and King groups (floor plans and 3D views); Schemes 1-3.",
    "RNGD IBC 2021 R1 rule set: RNGD_IBC_2021_R1_Five_Story_AI_Rules_v0_2.csv and the companion xlsx workbook.",
    "Kickoff meeting transcript (circulation and wall-burden rule, 30-34 ft bays, 2D-first output, upload of programs).",
    "RNGD Product Requirements Document (hotels as the first use case; proposal contents).",
]


def esc(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s.replace("<=", r"$\le$").replace(">=", r"$\ge$").replace("≤", r"$\le$").replace("≥", r"$\ge$")


def longtable(df: pd.DataFrame, widths: list[str], header: list[str] | None = None) -> str:
    spec = "".join(f"p{{{w}}}" for w in widths)
    head = " & ".join(r"\textbf{" + esc(h) + "}" for h in (header or list(df.columns)))
    rows = "\n".join(" & ".join(esc(v) for v in r) + r" \\" for r in df.astype(str).values.tolist())
    return (rf"\begin{{longtable}}{{{spec}}}" + "\n" + r"\toprule" + "\n" + head + r" \\ \midrule \endhead" + "\n"
            + rows + "\n" + r"\bottomrule\end{longtable}")


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{booktabs,longtable,xcolor,array}
\usepackage{fontspec}\setmainfont{Arial}
\setlength{\parskip}{0.45em}\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.12}
\definecolor{pass}{HTML}{1E7B3A}\definecolor{fail}{HTML}{B22222}
"""


def front_tex(sch, sheets) -> str:
    p = sch.program
    n = p.net_to_gross(MEETING_RULES["circulation_pct"], MEETING_RULES["wall_burden_pct"])
    fails = [c for c in sch.checks if c["status"] == "FAIL"]
    drawn = sum(lv.gross_sf for lv in sch.levels)
    index = "\n".join(rf"{esc(path.stem)} & {esc(title)} \\" for path, title in sheets)
    fail_txt = ("; ".join(f"{esc(c['check'])} ({esc(c['rule_id'])}: required {esc(c['required'])}, provided "
                          f"{esc(c['provided'])})" for c in fails) or "none")
    return PREAMBLE + rf"""\begin{{document}}
\begin{{center}}
{{\fontsize{{44}}{{48}}\selectfont\bfseries RNGD}}\\[0.6em]
{{\Large PROJECT RENEGADE --- LIVING + LODGING PROTOTYPE}}\\[0.3em]
{{\large Schematic proposal generated from the program}}\\[0.3em]
{{\small\color{{fail}} AI-generated concept schematic --- not for construction}}
\end{{center}}
\vspace{{1em}}
\section*{{Summary}}
\begin{{tabular}}{{p{{5.2cm}}p{{10.5cm}}}}
Program & {esc(p.source)} \\
Keys & {p.keys} in {sum(1 for r in p.rows if r[3] == 'guest')} unit types \\
Net program area & {p.net_sf:,.0f} SF in {len(p.rows)} program rows \\
Gross area, source sheets & {p.gross_sf:,.0f} SF (ratio {p.gross_sf / p.net_sf:.2f}) \\
Gross area, meeting rule & {n['required_gross_sf']:,.0f} SF (33\% circulation + 8\% walls, rounded up to 100 SF) \\
Gross area, this schematic & {drawn:,.0f} SF over {len(sch.levels)} levels \\
Massing & Level 1 {sch.levels[0].length_ft:.0f} x {sch.levels[0].width_ft:.0f} ft; guest floors {sch.levels[1].length_ft:.0f} x {sch.levels[1].width_ft:.0f} ft; {sch.height_ft:.1f} ft to roof \\
Keys per floor & {", ".join(f"{esc(lv.name)}: {lv.keys_placed}" for lv in sch.levels[1:])} \\
Code screen & {sum(c['status'] == 'PASS' for c in sch.checks)} pass, {len(fails)} fail, {sum(c['status'] == 'ASSUMED' for c in sch.checks)} assumed \\
\end{{tabular}}

\textbf{{Failing checks:}} {fail_txt}.

The drawing set follows the RNGD Living + Lodging sheet format: an overview sheet with the axonometric and
the gross-area, room-count and space-allocation schedules; colour-coded plans with column grid, dimension
strings, doors and furniture; elevations with an indicative prefab panel breakdown; enlarged plans and 3D
views for every unit type. Each sheet is also saved as a PNG in \texttt{{reports/final/sheets/}}.

\section*{{Drawing set}}
\begin{{tabular}}{{lp{{12.5cm}}}}\toprule \textbf{{Sheet}} & \textbf{{Content}} \\ \midrule
{index}
\bottomrule\end{{tabular}}

Appendices after the drawings: program, gross-area reconciliation, code checks, the full RNGD IBC rule set,
RNGD workbook tables, assumptions, open questions and sources.
\end{{document}}
"""


def back_tex(sch) -> str:
    p = sch.program
    prog_rows = []
    order = ["shared", "boh", "tenant", "guest", "business"]
    cats = order + sorted({r[3] for r in p.rows} - set(order))
    for cat in cats:
        rows = [r for r in p.rows if r[3] == cat]
        if not rows:
            continue
        prog_rows.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{esc(GROUP_TITLES.get(cat, cat.title()))}}}}} \\")
        prog_rows += [rf"{c} & {esc(n)} & {a:,.0f} \\" for n, c, a, _ in rows]
        prog_rows.append(rf"\midrule {sum(r[1] for r in rows)} & & {sum(r[2] for r in rows):,.0f} \\")
    prog_rows.append(rf"\midrule \textbf{{{sum(r[1] for r in p.rows)}}} & \textbf{{Total}} & \textbf{{{p.net_sf:,.0f}}} \\")
    prog = "\n".join(prog_rows)

    n = p.net_to_gross(MEETING_RULES["circulation_pct"], MEETING_RULES["wall_burden_pct"])
    by_level = "\n".join(
        rf"{esc(lv.name)} & {p.gross_by_level.get(lv.name, 0):,.0f} & {lv.gross_sf:,.0f} & {lv.keys_placed} \\"
        for lv in sch.levels)

    def status(s):
        return {"PASS": r"\textcolor{pass}{\textbf{PASS}}", "FAIL": r"\textcolor{fail}{\textbf{FAIL}}"}.get(s, esc(s))

    checks = "\n".join(rf"{esc(c['rule_id'])} & {esc(c['check'])} & {esc(c['required'])} & {esc(c['provided'])} & "
                       rf"{status(c['status'])} \\" for c in sch.checks)
    notes = "\n".join(rf"\item {esc(c['rule_id'])}: {esc(c['note'])}" for c in sch.checks if c.get("note"))

    rules = pd.read_csv(RULES_CSV, encoding="utf-8", encoding_errors="replace")
    ids = checked_rule_ids(sch.checks)
    rules["here"] = rules["rule_id"].isin(ids).map({True: "yes", False: "no"})
    rules["value"] = rules["value"].astype(str).str.replace("�", "x", regex=False)
    rule_tab = longtable(rules[["rule_id", "category", "rule_name", "operator", "value", "units", "hard_soft", "here"]],
                         ["1.75cm", "2.0cm", "4.7cm", "1.2cm", "1.5cm", "1.7cm", "0.9cm", "1.1cm"],
                         ["Rule", "Category", "Rule", "Op", "Value", "Units", "Hard", "Checked"])

    wb_parts = []
    if WORKBOOK.exists():
        xl = pd.ExcelFile(WORKBOOK)
        for name in ["Project_Config", "Occupant_Load_Factors", "Hotel_Space_Mapping", "Rule_Test_Cases"]:
            if name not in xl.sheet_names:
                continue
            df = xl.parse(name).fillna("").astype(str)
            cols = list(df.columns)[:5]
            df = df[cols].map(lambda v: v if len(v) < 70 else v[:67] + "...")
            wb_parts.append(rf"\subsection*{{Workbook sheet: {esc(name)}}}" + "\n" +
                            longtable(df.head(25), [f"{15.6 / len(cols):.2f}cm"] * len(cols)) +
                            (rf"{{\small Showing 25 of {len(xl.parse(name))} rows.}}" if len(xl.parse(name)) > 25 else ""))
    wb = "\n".join(wb_parts)
    items = lambda xs: "\n".join(rf"\item {esc(x)}" for x in xs)  # noqa: E731

    return PREAMBLE + rf"""\begin{{document}}
\section*{{Appendix A --- Program}}
Source: {esc(p.source)}. Areas are the total for each row, as in the source schedule.
\begin{{longtable}}{{p{{1.4cm}}p{{9.5cm}}p{{3cm}}}}\toprule \textbf{{Count}} & \textbf{{Type}} & \textbf{{Total area (SF)}} \\ \midrule \endhead
{prog}
\bottomrule\end{{longtable}}

\section*{{Appendix B --- Gross area}}
\begin{{tabular}}{{p{{3cm}}p{{3.6cm}}p{{3.6cm}}p{{2cm}}}}\toprule \textbf{{Level}} & \textbf{{Source sheets (SF)}} & \textbf{{This schematic (SF)}} & \textbf{{Keys}} \\ \midrule
{by_level}
\midrule Total & {p.gross_sf:,.0f} & {sum(lv.gross_sf for lv in sch.levels):,.0f} & {p.keys} \\ \bottomrule
\end{{tabular}}

Meeting rule: {p.net_sf:,.0f} SF net $\times$ 1.33 $\times$ 1.08 = {n['required_gross_sf']:,.0f} SF (rounded up to the
nearest 100 SF). The source sheets carry {p.gross_sf:,.0f} SF, a ratio of {p.gross_sf / p.net_sf:.2f}. The two rules
disagree by {n['required_gross_sf'] - p.gross_sf:,.0f} SF; see open question 3.

\section*{{Appendix C --- Code checks}}
Screened against RNGD's 2021 IBC rule set. Planning level only, not a code determination.
\begin{{longtable}}{{p{{1.8cm}}p{{3.8cm}}p{{4.1cm}}p{{3.2cm}}p{{1.4cm}}}}\toprule
\textbf{{Rule}} & \textbf{{Check}} & \textbf{{Required}} & \textbf{{Provided}} & \textbf{{Status}} \\ \midrule \endhead
{checks}
\bottomrule\end{{longtable}}
\begin{{itemize}}
{notes}
\end{{itemize}}

\section*{{Appendix D --- RNGD IBC rule set}}
{len(rules)} rules from the CSV. {int((rules.here == 'yes').sum())} are evaluated against this schematic (column
``Checked''). Most of the rest need space-level occupancy or detailed egress modelling.
{{\small\setlength{{\tabcolsep}}{{3pt}} {rule_tab}}}

\section*{{Appendix E --- RNGD workbook}}
{{\small {wb}}}

\section*{{Appendix F --- Assumptions}}
\begin{{itemize}}
{items(ASSUMPTIONS)}
\item 3D views are drawn with a painter's-algorithm axonometric in matplotlib. OpenGL renderers (moderngl, pyvista) are not installed in this environment.
\end{{itemize}}

\section*{{Appendix G --- Open questions for RNGD}}
\begin{{enumerate}}
{items(OPEN_QUESTIONS)}
\end{{enumerate}}

\section*{{Appendix H --- Sources}}
\begin{{itemize}}
{items(SOURCES)}
\end{{itemize}}
\end{{document}}
"""


def _tex(name: str, body: str, out_dir: Path) -> Path | None:
    (out_dir / f"{name}.tex").write_text(body, encoding="utf-8")
    if not TECTONIC.exists():
        return None
    r = subprocess.run([str(TECTONIC), f"{name}.tex"], cwd=out_dir, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return None
    return out_dir / f"{name}.pdf"


def build(program=None, site: dict | None = None, out_dir: Path = OUT, layout_kwargs: dict | None = None) -> Path:
    """Build the full report; returns the merged PDF path. Also writes report_manifest.json
    (the results the app shows: summary numbers, checks, sheets, open questions)."""
    program = program or default_program()
    site = site or DEFAULT_SITE
    out_dir.mkdir(parents=True, exist_ok=True)
    sch = layout.build(program, site, **(layout_kwargs or {}))
    sheets = drawings.build_sheets(sch, out_dir / "sheets", OPEN_QUESTIONS, ASSUMPTIONS)
    current = {p.stem for p, _ in sheets}
    for f in (out_dir / "sheets").glob("*"):  # drop sheets from an earlier, longer set
        if f.suffix in (".pdf", ".png") and f.stem not in current:
            f.unlink()
    front = _tex("front", front_tex(sch, sheets), out_dir)
    back = _tex("appendices", back_tex(sch), out_dir)
    merged = pymupdf.open()
    for part in [front] + [p for p, _ in sheets] + [back]:
        if part is not None and part.exists():
            with pymupdf.open(part) as doc:
                merged.insert_pdf(doc)
    final = out_dir / "RNGD_final_schematic_report.pdf"
    pages = merged.page_count
    merged.save(final, garbage=3, deflate=True)
    merged.close()
    if front is None or back is None:
        print("LaTeX front/appendices were not built (tectonic missing or failed); the PDF has the drawing set only.")
    _manifest(sch, sheets, final, pages, out_dir, site, layout_kwargs or {})
    return final


def _manifest(sch, sheets, final: Path, pages: int, out_dir: Path, site: dict, layout_kwargs: dict) -> None:
    from datetime import datetime

    from rngd.hotel.details import mix_by_floor

    p = sch.program
    n = p.net_to_gross(MEETING_RULES["circulation_pct"], MEETING_RULES["wall_burden_pct"])
    mix = mix_by_floor(sch)
    data = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "pdf": str(final), "pages": pages, "program_source": p.source, "site": site,
        # effective values (not just what was passed) so the app can tell whether the report is current
        "layout": {"upper_levels": len(sch.levels) - 1, "ground_height_ft": sch.levels[0].height_ft,
                   "floor_height_ft": sch.levels[1].height_ft},
        "summary": {"keys": p.keys, "net_sf": p.net_sf, "gross_source_sf": p.gross_sf,
                    "gross_meeting_sf": n["required_gross_sf"], "gross_drawn_sf": sum(lv.gross_sf for lv in sch.levels),
                    "levels": len(sch.levels), "height_ft": sch.height_ft,
                    "tower": [sch.levels[1].length_ft, sch.levels[1].width_ft],
                    "podium": [sch.levels[0].length_ft, sch.levels[0].width_ft]},
        "checks": sch.checks,
        "mix_by_floor": {k: {c: int(v) for c, v in row.items()} for k, row in mix.iterrows()},
        "sheets": [{"number": path.stem, "title": title, "pdf": str(path), "png": str(path.with_suffix(".png"))}
                   for path, title in sheets],
        "open_questions": OPEN_QUESTIONS, "assumptions": ASSUMPTIONS, "sources": SOURCES,
    }
    (out_dir / "report_manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_manifest(out_dir: Path = OUT) -> dict | None:
    f = out_dir / "report_manifest.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


if __name__ == "__main__":
    prog = load_program_file(sys.argv[1]) if len(sys.argv) > 1 else default_program()
    print(build(prog))
