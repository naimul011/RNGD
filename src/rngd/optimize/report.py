"""AI-written optimization report (PDF).

The LLM report-writer agent writes the prose sections from the computed results plus
retrieved code passages. Every table and figure is generated from the data itself, and every
number the AI writes is fact-checked against the data; numbers that cannot be traced are
listed in the PDF. Without an LLM, a deterministic template writes the sections instead.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from .metrics import OBJECTIVE_TEXT

ROOT = Path(__file__).resolve().parents[3]
TECTONIC = ROOT / "Testt" / "tools" / "tectonic.exe"

WRITER_SYSTEM = """You are writing the narrative sections of a design-optimization report for RNGD, a \
Building-as-a-Product company. The data you receive was computed by an optimizer (a memetic genetic \
algorithm with an LLM design reviewer) for a hotel floor plan on a fixed site grid. Write for an \
architect and a developer: clear, specific, no hype.

Rules: use ONLY numbers that appear in the data; do not invent percentages, costs or dates, and do not convert or re-round them (quote a ratio exactly as it appears in the data instead of turning it into a percentage). Objective values are 0-1 scores, not distances or areas. When you \
discuss a code topic, rely on the retrieved excerpts and name the rule (e.g. IBC-R1-027). Say plainly \
what failed and what is uncertain.

Return ONLY JSON with exactly these keys (plain text, no markdown):
- "title": short report title
- "executive_summary": one paragraph (4-6 sentences)
- "approach": one paragraph on how the design was optimized (decoder, search method, fitness; mention an LLM review only if it was used)
- "results": one paragraph on the optimized design (rooms, keys, efficiency, daylight, per-floor mix)
- "code_and_accessibility": one paragraph on IBC egress and ADA results
- "comparison": one paragraph comparing with the reference design, or "" if no reference is given
- "recommendations": array of 3-6 short, concrete next steps
- "limitations": array of 3-5 short items"""


def _esc(s) -> str:
    s = "" if s is None else str(s)
    s = s.encode("ascii", "replace").decode("ascii") if False else s
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    for a, b in [("‑", "-"), ("‐", "-"), ("‒", "-"), (" ", " "), (" ", " "), ("≈", "~"),
                 ("—", "---"), ("–", "--"), ("≤", r"$\le$"), ("≥", r"$\ge$"), ("→", r"$\to$"),
                 ("×", r"$\times$"), ("‘", "`"), ("’", "'"), ("“", "``"), ("”", "''"),
                 ("…", "..."), ("<=", r"$\le$"), (">=", r"$\ge$")]:
        s = s.replace(a, b)
    return "".join(ch for ch in s if ord(ch) < 0x2500)


def _brief(result: dict) -> dict:
    m = result["metrics"]
    out = {
        "site": result["site"], "program": result["program"],
        "search": {k: result["config"].get(k) for k in ("method", "population", "generations", "use_llm", "llm_checkpoints",
                                                        "llm_lambda", "seed")},
        "evaluations": result.get("evaluations"), "runtime_s": result.get("seconds"),
        "optimized": {k: m[k] for k in ("fitness", "rooms_placed", "rooms_total", "keys_placed", "keys_requested",
                                        "placed_area_sf", "program_area_sf", "corridor_sf", "corridor_ratio",
                                        "max_travel_ft", "dead_end_ft", "cores_reached", "cores_total",
                                        "exterior_fraction", "ada_required", "ada_placed", "shared_wall_ft",
                                        "unplaced_by_type")},
        "objective_scores_0_to_1 (1 = best; these are scores, not distances or areas)": m["terms"],
        "objective_weights": m["weights"],
        "levels": m["levels"], "code_checks": m["checks"],
        "llm_reviews": [{"generation": r["gen"], "engine": r.get("engine"),
                         "scores": {c["id"]: c["llm_score"] for c in r["candidates"]},
                         "issues": [i for rv in r.get("reviews", []) for i in rv.get("issues", [])][:6],
                         "weight_changes": r.get("weight_multipliers"), "rationale": r.get("rationale")}
                        for r in result.get("reviews", [])],
        "first_generation_best": result["history"][0]["best"] if result.get("history") else None,
        "last_generation_best": result["history"][-1]["best"] if result.get("history") else None,
    }
    ref = result.get("reference")
    if ref:
        rm = ref["metrics"]
        out["reference"] = {"file": ref["file"], "generations": ref.get("generations"),
                            **{k: rm[k] for k in ("fitness", "rooms_placed", "rooms_total", "keys_placed", "corridor_ratio",
                                                  "max_travel_ft", "dead_end_ft", "cores_reached", "exterior_fraction",
                                                  "shared_wall_ft")},
                            "failed_checks": [c["rule_id"] for c in rm["checks"] if c["status"] == "FAIL"]}
    return out


def _numbers(obj) -> set[float]:
    found: set[float] = set()

    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            found.add(round(float(o), 4))
        elif isinstance(o, str):
            for t in re.findall(r"-?\d[\d,]*\.?\d*", o):
                try:
                    found.add(round(float(t.replace(",", "").rstrip(".")), 4))
                except ValueError:
                    pass
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(k)
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(obj)
    return found


def fact_check(sections: dict, data: dict) -> list[dict]:
    """Every number in the AI text, and whether it can be traced to the data (value, x100 for
    ratios, rounding, or a difference / ratio of two headline values)."""
    nums = _numbers(data)
    derived = set()
    for v in list(nums):
        derived |= {round(v * 100, 1), round(v * 100), round(v, 1), round(v, 2), round(v), round(v / 100, 4)}
    head = [data["optimized"][k] for k in ("rooms_placed", "rooms_total", "keys_placed", "keys_requested",
                                           "dead_end_ft", "max_travel_ft", "shared_wall_ft") if isinstance(data["optimized"].get(k), (int, float))]
    if "reference" in data:
        head += [data["reference"][k] for k in ("rooms_placed", "rooms_total", "keys_placed", "dead_end_ft",
                                                "max_travel_ft", "shared_wall_ft", "fitness") if isinstance(data["reference"].get(k), (int, float))]
        head.append(data["optimized"]["fitness"])
    for a in head:
        for b in head:
            derived.add(round(abs(a - b), 1))
            derived.add(round(abs(a - b)))
            derived.add(round(abs(a - b), 4))
            if b:
                derived.add(round(100 * (a - b) / b))
                derived.add(round(100 * a / b))
    ok_set = nums | derived
    text = " ".join(str(v) if not isinstance(v, list) else " ".join(map(str, v)) for v in sections.values())
    out = []
    for tok in sorted(set(re.findall(r"(?<![A-Za-z-])\d[\d,]*\.?\d*", text))):
        try:
            v = float(tok.replace(",", "").rstrip("."))
        except ValueError:
            continue
        if v in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) and "." not in tok:
            continue  # small counts / list numbering are not claims worth flagging
        hit = any(abs(v - abs(x)) <= max(0.051, abs(x) * 0.005) for x in ok_set)  # rule ids parse as negatives
        out.append({"number": tok.rstrip("."), "traced": hit})
    return out


def template_sections(data: dict) -> dict:
    o = data["optimized"]
    fails = [c["rule_id"] for c in data["code_checks"] if c["status"] == "FAIL"]
    ref = data.get("reference")
    comp = ""
    if ref:
        comp = (f"Scored with the same evaluator, the reference design {ref['file']} places {ref['rooms_placed']} of "
                f"{ref['rooms_total']} rooms with fitness {ref['fitness']}, dead end {ref['dead_end_ft']} ft and "
                f"corridor ratio {ref['corridor_ratio']}; the optimized design places {o['rooms_placed']} with fitness "
                f"{o['fitness']}, dead end {o['dead_end_ft']} ft and corridor ratio {o['corridor_ratio']}.")
    return {
        "title": f"Optimized hotel floor plan: {data['site']}",
        "executive_summary": (f"The optimizer placed {o['rooms_placed']} of {o['rooms_total']} program rooms "
                              f"({o['keys_placed']} of {o['keys_requested']} keys) on {data['site']}. Code checks failing: "
                              f"{', '.join(fails) or 'none'}. Fitness {o['fitness']}."),
        "approach": ("An architectural decoder laid straight corridor spines one room-depth from the facade, cut them "
                     "into modular bays and assigned rooms to bays; a deterministic search over spine combinations with "
                     "pattern-search refinement chose the layout; the fitness combined ten objectives."),
        "results": (f"Corridor ratio {o['corridor_ratio']}, exterior fraction {o['exterior_fraction']}, shared walls "
                    f"{o['shared_wall_ft']} ft."),
        "code_and_accessibility": (f"Maximum exit travel {o['max_travel_ft']} ft and dead end {o['dead_end_ft']} ft; "
                                   f"ADA mobility rooms {o['ada_placed']} placed, {o['ada_required']} required."),
        "comparison": comp,
        "recommendations": ["Review unplaced rooms with the client", "Confirm which CORE regions are exit stairs",
                            "Run more generations for a final design"],
        "limitations": ["Rule-based packer", "Planning-level code screen", "Synthetic adjacency weights"],
    }


def write_sections(data: dict, use_llm: bool = True, model: str | None = None) -> tuple[dict, str, list[str]]:
    if use_llm:
        try:
            from .. import config, llm
            model = model or config.REPORT_MODEL
            from ..vectorstore import get_vectorstore
            fails = [c["check"] for c in data["code_checks"] if c["status"] == "FAIL"]
            queries = ["dead-end corridor maximum sprinklered R-1", "exit access travel distance",
                       "ADA guest rooms mobility features dispersion"] + fails
            context = get_vectorstore().retrieve_multi_as_context(queries, k_each=2)
            parsed, _ = llm.call_json(model, WRITER_SYSTEM,
                                      "Data:\n" + json.dumps(data, indent=1, default=str)
                                      + "\n\nRetrieved excerpts:\n" + context)
            keys = ("title", "executive_summary", "approach", "results", "code_and_accessibility", "comparison",
                    "recommendations", "limitations")
            sec = {k: parsed.get(k, "" if k not in ("recommendations", "limitations") else []) for k in keys}
            if sec["executive_summary"]:
                return sec, model, queries
        except Exception as e:
            sec = template_sections(data)
            return sec, f"template (LLM unavailable: {type(e).__name__})", []
    return template_sections(data), "template", []


def _table(headers, rows, spec):
    head = " & ".join(r"\textbf{" + _esc(h) + "}" for h in headers)
    body = "\n".join(" & ".join(_esc(c) for c in r) + r" \\" for r in rows)
    return (r"{\small\begin{longtable}{" + spec + "}\\toprule\n" + head + r" \\ \midrule \endhead" + "\n" + body
            + "\n\\bottomrule\\end{longtable}}")


def build_report(result: dict, out_dir: Path, figures: dict, use_llm: bool = True, model: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _brief(result)
    sections, engine_name, queries = write_sections(data, use_llm, model)
    checks = fact_check(sections, data)
    untraced = [c["number"] for c in checks if not c["traced"]]
    m = result["metrics"]
    para = lambda k: _esc(sections.get(k) or "")  # noqa: E731
    items = lambda k: "\n".join(r"\item " + _esc(x) for x in (sections.get(k) or [])) or r"\item ---"  # noqa: E731
    fig = lambda key, cap, wd="\\linewidth": (rf"\begin{{figure}}[H]\centering\includegraphics[width={wd}]{{{Path(figures[key]).name}}}"  # noqa: E731
                                              rf"\caption{{{_esc(cap)}}}\end{{figure}}") if figures.get(key) else ""
    for k, p in figures.items():
        if p and Path(p).exists():
            (out_dir / Path(p).name).write_bytes(Path(p).read_bytes())

    summary_rows = [["Rooms placed", f"{m['rooms_placed']} / {m['rooms_total']}"],
                    ["Keys placed", f"{m['keys_placed']} / {m['keys_requested']}"],
                    ["Placed area", f"{m['placed_area_sf']:,} SF of {m['program_area_sf']:,} SF"],
                    ["Corridor", f"{m['corridor_sf']:,} SF per level (ratio {m['corridor_ratio']})"],
                    ["Rooms with a window (where needed)", f"{m['exterior_fraction']}"],
                    ["Max exit travel / dead end", f"{m['max_travel_ft']} ft / {m['dead_end_ft']} ft"],
                    ["Cores reached", f"{m['cores_reached']} / {m['cores_total']}"],
                    ["ADA mobility rooms", f"{m['ada_placed']} placed, {m['ada_required']} required"],
                    ["Shared walls", f"{m['shared_wall_ft']:,} ft"], ["Fitness", f"{m['fitness']}"]]
    obj_rows = [[k, OBJECTIVE_TEXT[k], f"{m['terms'][k]:.3f}", f"{m['weights'][k]:.3f}"] for k in m["terms"]]
    chk_rows = [[c["rule_id"], c["check"], c["required"], c["provided"], c["status"]] for c in m["checks"]]
    lvl_rows = [[lv["level"], str(lv["rooms"]), str(lv["keys"]), f"{lv['area_sf']:,}",
                 ", ".join(f"{t} {n}" for t, n in lv["by_type"].items())] for lv in m["levels"]]
    rev_rows = []
    for r in result.get("reviews", []):
        scores = ", ".join(f"{c['id']}: {c['llm_score']}" for c in r["candidates"] if c["llm_score"] is not None)
        changes = ", ".join(f"{k} x{v:.2f}" for k, v in (r.get("weight_multipliers") or {}).items()) or "none"
        rev_rows.append([str(r["gen"]), str(r.get("engine")), scores, changes, (r.get("rationale") or "")[:200]])
    comp = ""
    ref = result.get("reference")
    if ref:
        rm = ref["metrics"]
        rows = [[lab, str(rm[k]), str(m[k])] for lab, k in
                (("Rooms placed", "rooms_placed"), ("Keys placed", "keys_placed"), ("Fitness (same evaluator)", "fitness"),
                 ("Corridor ratio", "corridor_ratio"), ("Window fraction", "exterior_fraction"),
                 ("Max travel (ft)", "max_travel_ft"), ("Dead end (ft)", "dead_end_ft"), ("Cores reached", "cores_reached"),
                 ("Shared walls (ft)", "shared_wall_ft"))]
        rows.append(["Failed code checks", ", ".join(c["rule_id"] for c in rm["checks"] if c["status"] == "FAIL") or "none",
                     ", ".join(c["rule_id"] for c in m["checks"] if c["status"] == "FAIL") or "none"])
        caveat = ""
        if result.get("program") != "rooms_hotel.json":
            caveat = (r"\textbf{Note:} the reference was made for the QA program (rooms\_hotel.json) and this design for "
                      + _esc(result.get("program")) + "; room counts are therefore not directly comparable, while the "
                      "code checks and efficiency measures are.\n\n")
        comp = (r"\section{Comparison with the reference design}" + caveat + para("comparison") + "\n\n"
                + _table(["Measure", f"Reference ({ref['file']}, {ref.get('generations')} gens)", "Optimized"], rows,
                         "p{5cm}p{5.2cm}p{5.2cm}") + fig("reference", f"Reference design {ref['file']}"))
    fc_rows = [[c["number"], "traced to data" if c["traced"] else "NOT traced"] for c in checks]

    fig_plans = fig("plans", "Optimized plans, all levels (corridor blue, cores maroon).")
    fig_conv = fig("convergence", "Convergence: best and mean fitness per generation; dotted lines mark LLM reviews.",
                   "0.85" + chr(92) + "linewidth")
    tbl_summary = _table(["Measure", "Optimized design"], summary_rows, "p{6cm}p{9.5cm}")
    tbl_levels = _table(["Level", "Rooms", "Keys", "Area SF", "Mix"], lvl_rows, "p{1.4cm}p{1.3cm}p{1.2cm}p{1.6cm}p{9.5cm}")
    tbl_checks = _table(["Rule", "Check", "Required", "Provided", "Status"], chk_rows, "p{1.9cm}p{4.3cm}p{3.6cm}p{3cm}p{1.4cm}")
    tbl_obj = _table(["Objective", "Meaning", "Score", "Final weight"], obj_rows, "p{2.4cm}p{8.2cm}p{1.6cm}p{2cm}")
    tbl_rev = (_table(["Gen", "Reviewer", "Scores (0-10)", "Weight changes", "Rationale"], rev_rows,
                      "p{0.8cm}p{2.2cm}p{3.2cm}p{3cm}p{6cm}") if rev_rows else "No reviews in this run.")
    tbl_fc = _table(["Number in text", "Result"], fc_rows, "p{4cm}p{5cm}") if fc_rows else "No numbers in the AI text."
    rec_items, lim_items = items("recommendations"), items("limitations")
    title = _esc(sections.get("title") or "Optimized hotel floor plan")
    disclosure = (f"The prose sections were written by {_esc(engine_name)} from the computed results"
                  + (", grounded with retrieved code passages" if queries else "") + ". All tables and figures come "
                  f"directly from the data. {len(checks)} numbers in the AI text were checked against the data: "
                  f"{len(checks) - len(untraced)} traced" + (", NOT traced: " + _esc(", ".join(untraced)) if untraced else "") + ".")
    stamp = f"RNGD design optimizer --- {datetime.now().strftime('%Y-%m-%d %H:%M')} --- site {_esc(result['site'])}, program {_esc(result['program'])}"
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[T1]{{fontenc}}\usepackage{{lmodern}}\usepackage[margin=2cm]{{geometry}}
\usepackage{{graphicx,booktabs,longtable,float,xcolor}}\usepackage[hidelinks]{{hyperref}}
\setlength{{\parskip}}{{0.5em}}\setlength{{\parindent}}{{0pt}}
\definecolor{{bp}}{{HTML}}{{123A6E}}
\begin{{document}}
{{\color{{bp}}\Large\bfseries {title}}}\par
{{\small {stamp}}}\par
\fcolorbox{{bp}}{{white}}{{\parbox{{0.96\linewidth}}{{\small\textbf{{AI disclosure.}} {disclosure}}}}}

\section{{Executive summary}}
{para('executive_summary')}

{tbl_summary}

\section{{Approach}}
{para('approach')}

\section{{Results}}
{para('results')}
{fig_plans}
{tbl_levels}
{fig_conv}

\section{{Code and accessibility}}
{para('code_and_accessibility')}
{tbl_checks}

{comp}

\section{{Objectives and weights}}
{tbl_obj}

\section{{LLM design reviews (inside the fitness function)}}
{tbl_rev}

\section{{Recommendations}}
\begin{{itemize}}
{rec_items}
\end{{itemize}}

\section{{Limitations}}
\begin{{itemize}}
{lim_items}
\end{{itemize}}

\section*{{Appendix: fact check of the AI text}}
{tbl_fc}
\end{{document}}
"""
    name = f"{result['site']}_optimization_report"
    (out_dir / f"{name}.tex").write_text(tex, encoding="utf-8")
    r = subprocess.run([str(TECTONIC), f"{name}.tex"], cwd=out_dir, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("LaTeX failed: " + r.stderr[-1500:])
    (out_dir / f"{name}_sections.json").write_text(json.dumps({"engine": engine_name, "sections": sections,
                                                               "fact_check": checks}, indent=1), encoding="utf-8")
    return out_dir / f"{name}.pdf"
