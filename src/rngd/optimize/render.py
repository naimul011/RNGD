"""Drawings of optimized (and reference) designs: one plan per level plus a convergence chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from .engine import Design  # noqa: E402

CAT_FILL = {"guest": "#CDBBD6", "suite": "#B49AC6", "ada": "#8FB8DE", "service": "#BDBDBD", "public": "#9FD5CB"}
CAT_NAME = {"guest": "Guest room", "suite": "Suite", "ada": "ADA room", "service": "Service", "public": "Public / BOH"}
BG = {"OUTSIDE": "#FFFFFF", "STRUCTURE": "#222222", "CORE": "#7A2E4D", "PLACEABLE": "#F4F1E8", "CORRIDOR": "#4D7798"}


def _short(name: str) -> str:
    n = name.replace("Housekeeping / Linen", "Linen").replace("Lobby / Front Desk", "Lobby").replace(
        "Breakfast / Lounge", "Breakfast").replace("Public Restrooms", "WC").replace("BOH / Office", "BOH")
    return n.replace("Ice / Vending", "Ice").replace("IT / Elec", "IT").replace("Janitor's closet", "Jan")


def draw_level(ax, design: Design, level: int, labels: bool = True, title: str | None = None):
    site = design.site
    g = site.grids[level]
    R, C = g.shape
    img = np.ones((R, C, 3))
    for lab, col in BG.items():
        img[g == lab] = matplotlib.colors.to_rgb(col)
    img[design.corridor] = matplotlib.colors.to_rgb(BG["CORRIDOR"])
    ax.imshow(img, origin="lower", extent=(-0.5, C - 0.5, -0.5, R - 0.5), interpolation="nearest")
    for (l, r0, c0, h, w, win) in design.extra.get("flex_bays", []):
        if l == level:
            ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), w, h, fc="#FBFAF6", ec="#BBBBBB", lw=0.4, hatch="///"))
    for p in design.placements:
        if p.level != level:
            continue
        x0, y0 = p.c0 - 0.5, p.r0 - 0.5
        ax.add_patch(Rectangle((x0, y0), p.w, p.h, fc=CAT_FILL.get(p.category, "#DDD"), ec="#222", lw=0.7))
        # window on the facade side (back wall), door on the corridor side
        if p.facing == "N":
            back, door = ((x0, y0 + p.h), (x0 + p.w, y0 + p.h)), (x0 + p.w / 2, y0)
        elif p.facing == "S":
            back, door = ((x0, y0), (x0 + p.w, y0)), (x0 + p.w / 2, y0 + p.h)
        elif p.facing == "E":
            back, door = ((x0 + p.w, y0), (x0 + p.w, y0 + p.h)), (x0, y0 + p.h / 2)
        else:
            back, door = ((x0, y0), (x0, y0 + p.h)), (x0 + p.w, y0 + p.h / 2)
        if p.exterior:
            (bx0, by0), (bx1, by1) = back
            inset = 0.2
            if by0 == by1:
                ax.plot([bx0 + 2, bx1 - 2], [by0, by1], color="#2F7FC0", lw=2.2, solid_capstyle="butt")
            else:
                ax.plot([bx0, bx1], [by0 + 2, by1 - 2], color="#2F7FC0", lw=2.2, solid_capstyle="butt")
        elif p.category in ("guest", "suite", "ada"):
            ax.add_patch(Rectangle((x0, y0), p.w, p.h, fill=False, ec="#B22222", lw=1.2, ls="--"))
        ax.plot(*door, marker="o", ms=2.2, color="#222")
        if labels:
            rot = 90 if p.h > p.w * 1.3 else 0
            big = p.w * p.h >= 120
            txt = f"{_short(p.type_name)}\n{p.area_sf:,.0f} SF" if big else _short(p.type_name)
            ax.text(x0 + p.w / 2, y0 + p.h / 2, txt, fontsize=3.9, ha="center", va="center", rotation=rot,
                    linespacing=1.0)
    ax.set_xlim(-1, C)
    ax.set_ylim(-1, R)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    n = sum(1 for p in design.placements if p.level == level)
    ax.set_title(title or f"{site.level_names[level]}: {n} rooms", fontsize=9)


def legend_handles():
    hs = [Patch(fc=c, ec="#222", label=CAT_NAME[k]) for k, c in CAT_FILL.items()]
    hs += [Patch(fc=BG["CORRIDOR"], label="Corridor"), Patch(fc=BG["CORE"], label="Core (stair/elevator)"),
           Patch(fc=BG["STRUCTURE"], label="Structure"),
           Patch(fill=False, ec="#B22222", ls="--", label="Guest room without window"),
           Patch(fc="#FBFAF6", ec="#BBBBBB", hatch="///", label="Unused bay (flex)"),
           matplotlib.lines.Line2D([], [], color="#2F7FC0", lw=2.2, label="Window wall"),
           matplotlib.lines.Line2D([], [], marker="o", ls="", color="#222", ms=3, label="Door")]
    return hs


def plan_figure(design: Design, path: Path, title: str = "", dpi: int = 150) -> Path:
    n = design.site.n_levels
    R, C = design.site.shape
    per_w = 5.2 if C <= R else 6.5 * C / max(R, 1) ** 0.5 / 10
    fig, axes = plt.subplots(1, n, figsize=(max(4.5, min(7.0, 5.0 * C / R)) * n, 6.2))
    axes = np.atleast_1d(axes)
    for l, ax in enumerate(axes):
        draw_level(ax, design, l)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def convergence_figure(history: list[dict], path: Path, reviews: list[dict] | None = None) -> Path:
    gens = [h["gen"] for h in history]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(gens, [h["best"] for h in history], "-", color="#123A6E", lw=2, label="best fitness")
    ax.plot(gens, [h["mean"] for h in history], "-", color="#9B7FB0", lw=1.2, label="population mean")
    ax2 = ax.twinx()
    ax2.plot(gens, [h["best_placed"] for h in history], "--", color="#E8B85C", lw=1.2, label="rooms placed (best)")
    ax2.set_ylabel("rooms placed", color="#9C6B0B")
    for r in reviews or []:
        ax.axvline(r["gen"], color="#B22222", ls=":", lw=1)
        ax.text(r["gen"], ax.get_ylim()[1], " LLM review", color="#B22222", fontsize=7, va="top", rotation=90)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
