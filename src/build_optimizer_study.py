"""Optimization study for the walkthrough report: the reference GA (QA/floorplan*.json) against
the updated optimizer, on the same three sites and the same room program, scored by the same
evaluator (rngd.optimize.metrics). Also runs the Hotel Schematic layout exploration.

Four variants per site:
  A  reference GA                    QA/floorplan{i}.json, 500 generations (as delivered)
  B  GA + free-form decoder          first attempt: same kind of search, corridor segments anywhere
  C  GA + architectural decoder      the updated GA (memetic, seeded, no LLM)
  D  systematic + pattern search     the default now (deterministic, no crossover)

Usage: python src/build_optimizer_study.py [--skip-b] [--no-shots]
Output: reports/walkthrough/optimizer/study.json + figures (PNG)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rngd.hotel.program import default_program  # noqa: E402
from rngd.optimize import baseline, decide, engine as freeform, explore, metrics, plan_view, planner, render  # noqa: E402
from rngd.optimize import search as S  # noqa: E402
from rngd.optimize import typology as T  # noqa: E402
from rngd.optimize.program import load_program  # noqa: E402
from rngd.optimize.site import load_site  # noqa: E402

QA = ROOT / "QA"
OUT = ROOT / "reports" / "walkthrough" / "optimizer"
POP, GENS = 24, 30
COL = {"A": "#9A9A9A", "B": "#E8B85C", "C": "#4D7798", "D": "#1E7B3A"}
NAME = {"A": "A · Reference GA (500 gen.)", "B": "B · GA + free-form decoder", "C": "C · GA + architectural decoder",
        "D": "D · Systematic + pattern search"}


def summary(design, m: dict, evaluations=None, seconds=None) -> dict:
    keys = [p for p in design.placements if p.category in ("guest", "suite", "ada")]
    levels = design.site.n_levels
    placeable = m["placeable_sf_per_level"] * levels
    corridor = m["corridor_sf"] * levels
    return {"rooms_placed": m["rooms_placed"], "rooms_total": m["rooms_total"], "keys_placed": m["keys_placed"],
            "keys_requested": m["keys_requested"], "windowless_guest_rooms": sum(1 for p in keys if not p.exterior),
            "exterior_fraction": m["exterior_fraction"], "fitness": m["fitness"], "terms": m["terms"],
            "corridor_ratio": m["corridor_ratio"], "dead_end_ft": m["dead_end_ft"], "max_travel_ft": m["max_travel_ft"],
            "cores_reached": m["cores_reached"], "cores_total": m["cores_total"],
            "checks_pass": sum(c["status"] == "PASS" for c in m["checks"]), "checks_total": len(m["checks"]),
            "failed_checks": [c["rule_id"] for c in m["checks"] if c["status"] == "FAIL"],
            "placed_area_sf": m["placed_area_sf"], "program_area_sf": m["program_area_sf"],
            "unused_share": round(max(0.0, placeable - corridor - m["placed_area_sf"]) / placeable, 4),
            "shared_wall_ft": m["shared_wall_ft"], "evaluations": evaluations, "seconds": seconds}


def bounds(site, prog) -> dict:
    """Upper bounds on what can fit (answers 'can 72 rooms fit at all?')."""
    names = site.level_names
    inst = prog.instances(names)
    L = site.n_levels
    a_place = site.placeable_sf(0)
    a_prog = sum(i.type.area_sf for i in inst)
    P = planner.params_for(site, prog)
    w_ft = P.w * site.cell_ft
    # facade: placeable cells with an OUTSIDE neighbour (4-connected), counted once per exposed side
    pl, out = site.placeable[0], np.pad(site.outside[0], 1, constant_values=True)
    facade = int((pl & out[:-2, 1:-1]).sum() + (pl & out[2:, 1:-1]).sum() + (pl & out[1:-1, :-2]).sum()
                 + (pl & out[1:-1, 2:]).sum()) * site.cell_ft
    keys = [i for i in inst if i.type.is_key]
    f_mean = float(np.mean([min(i.type.width_ft, i.type.depth_ft) for i in keys])) if keys else 0.0
    a_mean = float(np.mean([i.type.area_sf for i in keys])) if keys else 0.0
    d_mean = float(np.mean([max(i.type.width_ft, i.type.depth_ft) for i in keys])) if keys else 0.0
    # area bound: a double-loaded corridor serves 2 room depths, so corridor share >= w / (2 d + w)
    c_min = w_ft / (2 * d_mean + w_ft) if d_mean else 0.0
    area_bound = int(L * a_place * (1 - c_min) // a_mean) if a_mean else 0
    window_bound = int(L * (facade // f_mean)) if f_mean else 0
    return {"levels": L, "placeable_sf_per_level": round(a_place), "program_area_sf": round(a_prog),
            "program_over_placeable": round(a_prog / (L * a_place), 3), "facade_ft_per_level": round(facade),
            "mean_key_frontage_ft": round(f_mean, 1), "mean_key_area_sf": round(a_mean), "mean_key_depth_ft": round(d_mean, 1),
            "min_corridor_share": round(c_min, 3), "area_bound_keys": area_bound, "window_bound_keys": window_bound,
            "keys_requested": len(keys), "rooms_requested": len(inst), "module_ft": P.M * site.cell_ft,
            "spine_candidates": len(P.candidates)}


def run_site(i: int, skip_b: bool) -> dict:
    site, prog = load_site(QA / f"fp{i}_multi.json"), load_program(QA / "rooms_hotel.json")
    ref_raw = json.loads((QA / f"floorplan{i}.json").read_text(encoding="utf-8"))
    ref, info = baseline.load_reference(QA / f"floorplan{i}.json", site, prog)
    out = {"site": site.name, "bounds": bounds(site, prog), "variants": {}, "history": {}}
    designs = {"A": ref}
    out["variants"]["A"] = summary(ref, metrics.evaluate(ref), None, None)
    out["variants"]["A"]["generations"] = len(ref_raw.get("history", []))
    out["history"]["A"] = [[int(u), float(s)] for u, s in ref_raw.get("history", [])]   # (unplaced rooms, own score)
    runs = [("C", planner, "ga"), ("D", planner, "systematic")]
    if not skip_b:
        runs.insert(0, ("B", freeform, "ga"))
    for tag, dec, method in runs:
        S.engine = dec
        t0 = time.time()
        res = S.Optimizer(site, prog, S.SearchConfig(method=method, population=POP, generations=GENS)).run()
        S.engine = planner
        best = res["best"]
        designs[tag] = best.design
        out["variants"][tag] = summary(best.design, best.metrics, res["evaluations"], round(time.time() - t0, 1))
        out["history"][tag] = [[h["evaluations"], h["best_fitness"], h["best_placed"]] for h in res["history"]]
        print(f"  fp{i} {tag}: {best.metrics['rooms_placed']} rooms, fitness {best.metrics['fitness']}, "
              f"{res['evaluations']} evals, {time.time() - t0:.0f}s", flush=True)
    # repeat D to show determinism
    r2 = S.Optimizer(site, prog, S.SearchConfig(method="systematic", population=POP, generations=GENS)).run()
    out["deterministic_repeat"] = r2["best"].metrics["fitness"] == out["variants"]["D"]["fitness"]
    # figure: typical floor (level index 1) of every variant
    tags = [t for t in "ABCD" if t in designs]
    R_, C_ = site.shape
    ncol = 2 if len(tags) > 2 else len(tags)
    nrow = int(np.ceil(len(tags) / ncol))
    pw = 5.0
    fig, axes = plt.subplots(nrow, ncol, figsize=(pw * ncol, (pw * R_ / C_ + 0.6) * nrow + 0.6))
    for ax in np.atleast_1d(axes).ravel()[len(tags):]:
        ax.axis("off")
    for ax, t in zip(np.atleast_1d(axes).ravel(), tags):
        v = out["variants"][t]
        render.draw_level(ax, designs[t], 1, title=f"{NAME[t]}\n{v['rooms_placed']}/{v['rooms_total']} rooms · "
                                                   f"{v['windowless_guest_rooms']} windowless · fitness {v['fitness']:.3f}")
        ax.title.set_fontsize(8)
    fig.legend(handles=render.legend_handles(), loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / f"compare_fp{i}.png", dpi=170)
    plt.close(fig)
    return out


def charts(sites: list[dict]) -> None:
    tags = [t for t in "ABCD" if t in sites[0]["variants"]]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    specs = [("rooms_placed", "Rooms placed (of 72)"), ("windowless_guest_rooms", "Guest rooms without a window"),
             ("fitness", "Fitness (same evaluator)"), ("unused_share", "Unused floor area (share)")]
    x = np.arange(len(sites))
    for ax, (k, title) in zip(axes, specs):
        for j, t in enumerate(tags):
            vals = [s["variants"][t][k] for s in sites]
            b = ax.bar(x + (j - (len(tags) - 1) / 2) * 0.2, vals, 0.2, color=COL[t], label=NAME[t])
            for r, v in zip(b, vals):
                ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{v:.2f}" if isinstance(v, float) else str(v),
                        ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x, [s["site"].replace("_multi", "") for s in sites])
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "compare_bars.png", dpi=170)
    plt.close(fig)
    # convergence: reference (unplaced rooms by generation) and ours (rooms placed by evaluations)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for s, ls in zip(sites, ("-", "--", ":")):
        h = s["history"]["A"]
        axes[0].plot(range(1, len(h) + 1), [s["variants"]["A"]["rooms_total"] - u for u, _ in h], ls, color=COL["A"],
                     label=f"{s['site'].replace('_multi', '')} reference GA")
        for t in tags[1:]:
            hh = s["history"][t]
            axes[1].plot([e for e, _, _ in hh], [p for _, _, p in hh], ls, color=COL[t],
                         label=f"{s['site'].replace('_multi', '')} {t}")
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("rooms placed (72 - unplaced)")
    axes[0].set_title("Reference GA: 500 generations (history stored in floorplan*.json)", fontsize=9)
    axes[1].set_xlabel("layouts evaluated")
    axes[1].set_ylabel("rooms placed (best so far)")
    axes[1].set_title("Updated optimizer: B, C (GA) and D (systematic)", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=170)
    plt.close(fig)


def plotly_png(figs: dict) -> None:
    """Render Plotly figures with headless Chrome (the same figures the app shows)."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        for name, (fig, w, h, wait) in figs.items():
            pg = b.new_page(viewport={"width": w, "height": h})
            html = OUT / f"_{name}.html"
            fig.update_layout(height=h - 20, width=w - 20)
            fig.write_html(html, include_plotlyjs=True, full_html=True, config={"displayModeBar": False})
            pg.goto(html.as_uri())
            pg.wait_for_timeout(wait)
            pg.screenshot(path=str(OUT / f"{name}.png"))
            pg.close()
            html.unlink()
        b.close()


def run_hotel(shots: bool) -> dict:
    hp = default_program()
    cfg = explore.ExploreConfig()
    res = explore.explore(hp, cfg)
    res2 = explore.explore(hp, cfg)
    alts = res["alternatives"]
    rows = decide.rank(alts, decide.PRESETS["Balanced"])
    by_id = {a.id: a for a in alts}
    presets = {}
    for name, w in decide.PRESETS.items():
        r = decide.rank(alts, w)[0]
        presets[name] = {k: r[k] for k in ("id", "layout", "floors", "keys", "gross_sf_per_key", "height_ft", "code_pass",
                                           "score")}
    best_by_shape = {}
    for r in rows:
        best_by_shape.setdefault(r["shape"], r)
    # gallery of the top 8
    top = rows[:8]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4))
    designs = {}
    for ax, r in zip(axes.ravel(), top):
        a = by_id[r["id"]]
        d = designs[r["id"]] = explore.design_for(hp, a, cfg)
        img = plt.imread(__import__("io").BytesIO(plan_view.thumbnail(d, a.plate, 1)), format="png")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"#{r['rank']} {a.plate.name}\n{r['floors']} guest floors · {r['keys']} keys · "
                     f"{r['gross_sf_per_key']:.0f} SF/key · score {r['score']:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "hotel_gallery.png", dpi=160)
    plt.close(fig)
    # every layout: keys vs gross SF per key
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    shapes = sorted({r["shape"] for r in rows})
    cmap = dict(zip(shapes, ["#1E5AA8", "#B0590B", "#1E7B3A", "#9B7FB0", "#B22222", "#5B6670"]))
    for sh in shapes:
        rr = [r for r in rows if r["shape"] == sh]
        ax.scatter([r["gross_sf_per_key"] for r in rr if not r["pareto"]], [r["keys"] for r in rr if not r["pareto"]],
                   s=10, color=cmap[sh], alpha=0.25)
        ax.scatter([r["gross_sf_per_key"] for r in rr if r["pareto"]], [r["keys"] for r in rr if r["pareto"]],
                   s=26, color=cmap[sh], edgecolor="#111", lw=0.5, label=f"{sh} (outlined = Pareto)")
    ax.axhline(hp.keys, ls=":", color="#111")
    ax.set_xlabel("gross SF per key (lower = cheaper)")
    ax.set_ylabel("keys placed")
    ax.set_xlim(min(r["gross_sf_per_key"] for r in rows) * 0.9, 2500)
    ax.text(2480, hp.keys, f"{hp.keys} keys requested", va="bottom", ha="right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "hotel_pareto.png", dpi=170)
    plt.close(fig)
    b = by_id[rows[0]["id"]]
    d = designs[rows[0]["id"]]
    if shots:
        try:
            W, H = b.plate.size
            hpx = int(min(1100, max(520, 1500 * (H + 30) / (W + 30))))
            plotly_png({"hotel_best_L2": (plan_view.plan_figure(d, b.plate, 1), 1500, hpx, 1800),
                        "hotel_best_L1": (plan_view.plan_figure(d, b.plate, 0), 1500, hpx, 1800),
                        "hotel_best_3d": (plan_view.massing_figure(d, b.plate, 6.0), 1300, 760, 3500)})
        except Exception as e:  # report falls back to the gallery
            print("plotly screenshots failed:", e)
    m = b.metrics
    return {"keys_requested": hp.keys, "envelope": list(cfg.envelope), "variants": int(res["log"][0].split()[0]),
            "layouts": len(alts), "evaluations": res["evaluations"], "seconds": res["seconds"], "log": res["log"],
            "deterministic_repeat": [(a.plate.name, a.floors, a.fitness) for a in alts]
            == [(a.plate.name, a.floors, a.fitness) for a in res2["alternatives"]],
            "pareto": sum(r["pareto"] for r in rows), "top": rows[:10], "presets": presets,
            "best_by_shape": list(best_by_shape.values()),
            "refine": [{k: t[k] for k in ("plate", "floors", "start", "end")} for t in res["traces"]],
            "best": {"id": b.id, "plate": b.plate.name, "shape": b.plate.shape, "core_scheme": b.plate.core_scheme,
                     "floors": b.floors, "size": list(b.plate.size), "footprint_sf": b.plate.footprint_sf(),
                     "upper_bound_keys": b.upper_bound_keys, "metrics": {k: m[k] for k in (
                         "fitness", "terms", "rooms_placed", "rooms_total", "keys_placed", "keys_requested",
                         "corridor_ratio", "dead_end_ft", "max_travel_ft", "unplaced_by_type", "checks", "levels")},
                     "module_ft": d.extra.get("module_ft"), "elevators": T.elevators_for(hp.keys)},
            "weights": metrics.DEFAULT_WEIGHTS, "criteria": {k: list(v) for k, v in decide.CRITERIA.items()},
            "balanced_weights": decide.PRESETS["Balanced"]}


def main() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    skip_b = "--skip-b" in sys.argv
    sites = []
    for i in (1, 2, 3):
        print(f"site fp{i}...", flush=True)
        sites.append(run_site(i, skip_b))
    charts(sites)
    print("hotel exploration...", flush=True)
    hotel = run_hotel("--no-shots" not in sys.argv)
    study = {"built": time.strftime("%Y-%m-%d %H:%M"), "population": POP, "generations": GENS, "sites": sites,
             "hotel": hotel, "limits": metrics.LIMITS}
    p = OUT / "study.json"
    p.write_text(json.dumps(study, indent=1, default=str), encoding="utf-8")
    return p


if __name__ == "__main__":
    print(main())
