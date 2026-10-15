"""Command-line optimizer run, and a benchmark against the QA reference designs.

  python src/run_optimizer.py QA/fp1_multi.json [program.json] [--gens 30] [--pop 24] [--no-llm] [--seed 7]
  python src/run_optimizer.py --benchmark [--gens 30] [--pop 24] [--no-llm]

Outputs go to reports/optimizer/: <site>_optimized.json (QA schema + extras),
<site>_plans.png, <site>_convergence.png and, for --benchmark, benchmark.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from rngd.optimize import baseline, export, metrics, render, report  # noqa: E402
from rngd.optimize.program import load_program  # noqa: E402
from rngd.optimize.search import Optimizer, SearchConfig  # noqa: E402
from rngd.optimize.site import load_site  # noqa: E402

OUT = ROOT / "reports" / "optimizer"
QA = ROOT / "QA"


def optimize(site_path: Path, program_path: Path, cfg: SearchConfig, tag: str = "") -> dict:
    site = load_site(site_path)
    prog = load_program(program_path)
    print(f"== {site.name} with {prog.source}: {len(prog.instances(site.level_names))} rooms, "
          f"pop {cfg.population} x {cfg.generations} gens, LLM {'on' if cfg.use_llm else 'off'}", flush=True)
    opt = Optimizer(site, prog, cfg)
    res = opt.run(on_generation=lambda r, b: print(f"  gen {r['gen']:3d} best {r['best']:.4f} placed {r['best_placed']} "
                                                   f"({r['seconds']}s)", flush=True) if r["gen"] % 5 == 0 else None,
                  on_review=lambda e: print(f"  review @gen {e['gen']} by {e['engine']}: "
                                            f"{[c['llm_score'] for c in e['candidates']]} | {e['rationale'][:120]}", flush=True))
    best = res["best"]
    stem = f"{site.name}{tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    export.save_design(OUT / f"{stem}_optimized.json", best.design, best.metrics, history=res["history"],
                       reviews=res["reviews"], config=res["config"])
    render.plan_figure(best.design, OUT / f"{stem}_plans.png",
                       f"{site.name}: {best.metrics['rooms_placed']}/{best.metrics['rooms_total']} rooms, "
                       f"fitness {best.metrics['fitness']}")
    render.convergence_figure(res["history"], OUT / f"{stem}_convergence.png", res["reviews"])
    return {"site": site.name, "program": prog.source, "metrics": best.metrics, "history": res["history"],
            "reviews": res["reviews"], "seconds": res["seconds"], "evaluations": res["evaluations"],
            "weights": res["weights"]}


def benchmark(cfg: SearchConfig) -> Path:
    """Same program and grids as the reference runs; both scored by metrics.evaluate."""
    prog_path = QA / "rooms_hotel.json"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in (1, 2, 3):
        site_path = QA / f"fp{i}_multi.json"
        site, prog = load_site(site_path), load_program(prog_path)
        ref, info = baseline.load_reference(QA / f"floorplan{i}.json", site, prog)
        ref_m = metrics.evaluate(ref)
        render.plan_figure(ref, OUT / f"{site.name}_reference_plans.png",
                           f"Reference {info['file']}: {ref_m['rooms_placed']}/{ref_m['rooms_total']} rooms")
        ours = optimize(site_path, prog_path, cfg, tag="_busra_program")
        rows.append({"site": site.name, "reference": {"file": info["file"], "generations": info["generations"],
                                                       "reported_score": info["reported_score"], "metrics": ref_m},
                     "optimized": ours})
    p = OUT / "benchmark.json"
    p.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("site", nargs="?")
    ap.add_argument("program", nargs="?", default=str(ROOT / "data" / "optimizer" / "hotel_program_extended.json"))
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-llm", action="store_true", help="write the report from the template (no LLM)")
    ap.add_argument("--method", default="systematic", choices=["systematic", "ga"])
    ap.add_argument("--llm-review", action="store_true", help="GA only: LLM design review at checkpoints")
    ap.add_argument("--model", default=None, help="OpenRouter model for the report prose")
    ap.add_argument("--report", action="store_true", help="also write the AI report PDF")
    a = ap.parse_args()
    cfg = SearchConfig(method=a.method, population=a.pop, generations=a.gens, seed=a.seed, use_llm=a.llm_review,
                       llm_checkpoints=3 if a.llm_review else 0)
    if a.benchmark:
        print(benchmark(cfg))
    else:
        r = optimize(Path(a.site), Path(a.program), cfg)
        if a.report:
            site = load_site(a.site)
            stem = site.name
            figs = {"plans": str(OUT / f"{stem}_plans.png"), "convergence": str(OUT / f"{stem}_convergence.png"), "reference": None}
            ref_file = QA / f"floorplan{stem.replace('_multi', '')[2:]}.json" if stem.startswith("fp") else None
            if ref_file and ref_file.exists():
                rd, info = baseline.load_reference(ref_file, site, load_program(QA / "rooms_hotel.json"))
                rm = metrics.evaluate(rd)
                figs["reference"] = str(render.plan_figure(rd, OUT / f"{stem}_reference_plans.png",
                                                           f"Reference {info['file']}: {rm['rooms_placed']}/{rm['rooms_total']} rooms"))
                r["reference"] = {"file": info["file"], "generations": info["generations"], "metrics": rm}
            r["config"] = {"population": a.pop, "generations": a.gens, "method": cfg.method, "use_llm": cfg.use_llm, "seed": a.seed,
                           "llm_checkpoints": cfg.llm_checkpoints, "llm_lambda": cfg.llm_lambda}
            r["program"] = Path(a.program).name
            print(report.build_report(r, OUT / "report", figs, use_llm=not a.no_llm, model=a.model))
        print(json.dumps({k: r["metrics"][k] for k in ("fitness", "rooms_placed", "rooms_total", "keys_placed",
                                                         "corridor_ratio", "max_travel_ft", "dead_end_ft")}, indent=1))
