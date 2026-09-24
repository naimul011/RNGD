"""Matplotlib rendering of simulation output to PNG. Colors are shared
constants with vision/layout_qa.py, which re-measures these exact PNGs by
color segmentation — the renderer and the CV QA agent must agree on the
palette or the QA step can't verify anything. Each render also writes a small
pixel<->feet calibration sidecar (two known reference points and where they
landed in the final PNG), the rendering equivalent of the georeference file
used for the input site sketch, so the CV QA step can convert its pixel
measurements back to feet exactly rather than guessing a scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLOR_LOT_BOUNDARY = "#000000"
COLOR_EASEMENT = "#FFBF00"
COLOR_SETBACK_LINE = "#CC0000"
COLOR_BUILDING = "#1E5AA8"
COLOR_DRIVEWAY = "#555555"
COLOR_PARKING = "#8C8C8C"
COLOR_CORE = "#8E44AD"
COLOR_CORRIDOR = "#DDDDDD"
COLOR_UNIT = "#2E8B57"

RENDER_DPI = 140  # must match the dpi passed to every savefig() call below —
# _save_geom_sidecar computes pixel coordinates from fig.dpi, and a mismatch
# between that and the actual savefig dpi silently scales every CV
# measurement (this bit us once: fig.dpi defaulted to 100 while savefig used
# 140, scaling every extent by 1.4x). Setting dpi at figure creation time
# keeps fig.dpi and the saved raster in agreement.


def _save_geom_sidecar(fig, ax, out_path: Path, ref_ft: list[tuple[float, float]]) -> None:
    """Records where two known ft-space points land in the saved PNG's pixel
    space, so a downstream CV step can invert the affine transform exactly.
    """
    fig.canvas.draw()
    dpi = fig.dpi
    fig_h_px = fig.get_size_inches()[1] * dpi
    ref_px = []
    for x_ft, y_ft in ref_ft:
        disp = ax.transData.transform((x_ft, y_ft))
        px_x = float(disp[0])
        px_y = float(fig_h_px - disp[1])  # flip: matplotlib origin bottom-left -> image origin top-left
        ref_px.append([px_x, px_y])
    sidecar = {"ref_ft": [list(p) for p in ref_ft], "ref_px": ref_px}
    Path(str(out_path) + ".geom.json").write_text(json.dumps(sidecar, indent=2))


def render_site_plan(
    out_path: Path,
    lot: dict,
    easements: list[dict],
    setbacks: dict,
    buildable_envelope: dict,
    bodies: list,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 9), dpi=RENDER_DPI)
    lot_w, lot_d = lot["frontage_ft"], lot["depth_ft"]

    ax.add_patch(Rectangle((0, 0), lot_w, lot_d, fill=False, edgecolor=COLOR_LOT_BOUNDARY, linewidth=2))

    for e in easements:
        w = e["width_ft"]
        if e["side"] == "east":
            ax.add_patch(Rectangle((lot_w - w, 0), w, lot_d, facecolor=COLOR_EASEMENT, alpha=0.9, edgecolor="none"))
        elif e["side"] == "west":
            ax.add_patch(Rectangle((0, 0), w, lot_d, facecolor=COLOR_EASEMENT, alpha=0.9, edgecolor="none"))

    env = buildable_envelope
    ax.add_patch(
        Rectangle(
            (env["x_min_ft"], env["y_min_ft"]),
            env["width_ft"],
            env["depth_ft"],
            fill=False,
            edgecolor=COLOR_SETBACK_LINE,
            linestyle="--",
            linewidth=1.5,
        )
    )

    for b in bodies:
        if b.kind == "building":
            color, edge, lw = COLOR_BUILDING, "black", 0.6
        elif b.kind == "driveway":
            color, edge, lw = COLOR_DRIVEWAY, "black", 0.6
        else:
            color, edge, lw = COLOR_PARKING, "white", 1.2
        ax.add_patch(Rectangle((b.x, b.y), b.w, b.h, facecolor=color, edgecolor=edge, linewidth=lw))

    ax.set_xlim(-10, lot_w + 10)
    ax.set_ylim(lot_d + 10, -10)  # y grows south/down like the sketch convention
    ax.set_aspect("equal")
    ax.set_title("Schematic Site Plan (generated)")
    ax.set_xlabel("ft (east ->)")
    ax.set_ylabel("ft (south, from street)")
    fig.tight_layout()
    _save_geom_sidecar(fig, ax, out_path, ref_ft=[(0, 0), (lot_w, lot_d)])
    fig.savefig(out_path, dpi=RENDER_DPI)
    plt.close(fig)
    return out_path


def render_floor_plate(out_path: Path, floor_plate: dict, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=RENDER_DPI)
    for m in floor_plate["modules"]:
        if m["kind"] == "core":
            color = COLOR_CORE
        elif m["kind"] == "unit":
            color = COLOR_UNIT
        else:
            color = COLOR_CORRIDOR
        ax.add_patch(Rectangle((m["x"], m["y"]), m["w"], m["h"], facecolor=color, edgecolor="black", linewidth=0.6))
        ax.text(
            m["x"] + m["w"] / 2,
            m["y"] + m["h"] / 2,
            m["module_id"].replace("_A", "").replace("_STD", ""),
            ha="center",
            va="center",
            fontsize=6,
        )

    ax.set_xlim(-5, floor_plate["floor_width_ft"] + 5)
    ax.set_ylim(floor_plate["floor_depth_ft"] + 5, -5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("ft")
    fig.tight_layout()
    fig.savefig(out_path, dpi=RENDER_DPI)
    plt.close(fig)
    return out_path


def render_convergence_chart(out_path: Path, iteration_histories: list[list[float]]) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=RENDER_DPI)
    for idx, hist in enumerate(iteration_histories):
        ax.plot(hist, label=f"simulation iteration {idx + 1}")
    ax.set_xlabel("physics step")
    ax.set_ylabel("total overlap/containment penalty (ft)")
    ax.set_title("Site simulation convergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=RENDER_DPI)
    plt.close(fig)
    return out_path
