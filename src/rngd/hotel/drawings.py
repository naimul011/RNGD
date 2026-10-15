"""Drawing sheets for a hotel Schematic, styled after RNGD's Project Renegade L+L set
(docs/General/.../Scheme 2 - 223 Hotel, Redesigned/Project Renegade Living+Lodging_Drawings.pdf):
18 x 12 in landscape sheets with the RNGD title block, colour-filled plans with
column grid and dimension strings, elevations with an indicative panel breakdown,
enlarged unit plans and isometric unit / building views.

Everything is drawn with matplotlib (vector PDF + PNG). The isometric views use
a small painter's-algorithm projector, not OpenGL.
"""
from __future__ import annotations

import math
import textwrap
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402

from . import layout as L  # noqa: E402
from .program import GROUP_TITLES  # noqa: E402

matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                            "pdf.fonttype": 42, "svg.fonttype": "none"})

SHEET_W, SHEET_H = 18.0, 12.0
INK = "#222222"
GRID = "#9A9A9A"
GLASS = "#9CC9E6"
PANEL = "#1E8A44"
FILL = {"guest": "#CDBBD6", "corridor": "#4D7798", "shared": "#4D7798", "business": "#9FD5CB",
        "service": "#BDBDBD", "tenant": "#F9DDA6", "stair": "#E3E3E3", "elevator": "#FFFFFF",
        "shaft": "#D3D3D3", "flex": "#F3F3F3"}
DARK = {"corridor", "shared"}
LEGEND = [("guest", "Hotel units"), ("shared", "Shared / circulation"), ("business", "Business / BOH"),
          ("service", "Service"), ("tenant", "Tenant"), ("stair", "Stairs"), ("flex", "Unprogrammed")]
PROJECT = ("PROJECT RENEGADE", "LIVING + LODGING PROTOTYPE")
GRID_Y = [0.0, L.ROOM_DEPTH_FT, L.ROOM_DEPTH_FT + L.CORRIDOR_FT, L.WING_W_FT]


# --------------------------------------------------------------- helpers -
def ftin(x: float) -> str:
    ft = int(math.floor(x + 1e-9))
    inch = int(round((x - ft) * 12))
    if inch == 12:
        ft, inch = ft + 1, 0
    return f"{ft}' - {inch}\""


def wrap(text: str, width_ft: float, ft_per_char: float = 0.8) -> str:
    n = max(4, int(width_ft / ft_per_char))
    return "\n".join(textwrap.wrap(text, n)) if len(text) > n else text


def new_sheet(title: str, number: str, project=PROJECT, issued: str | None = None, legend: bool = False):
    """Blank 18x12 sheet with the title block. Returns (fig, sheet_ax) where sheet_ax is in inches."""
    fig = plt.figure(figsize=(SHEET_W, SHEET_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SHEET_W)
    ax.set_ylim(0, SHEET_H)
    ax.axis("off")
    ax.add_patch(Rectangle((0.08, 0.08), SHEET_W - 0.16, SHEET_H - 0.16, fill=False, ec="#BBBBBB", lw=0.6))
    ax.plot([0.6, SHEET_W - 0.3], [1.38, 1.38], color="#888888", lw=0.7)
    ax.plot([0.6, SHEET_W - 0.3], [0.32, 0.32], color="#888888", lw=0.7)
    ax.text(0.85, 0.72, title.upper(), fontsize=11, va="center")
    ax.text(0.85, 0.5, issued or date.today().strftime("%m/%d/%y"), fontsize=6.5, va="center")
    ax.text(SHEET_W - 0.45, 1.0, "RNGD", fontsize=40, fontweight="heavy", ha="right", va="center", color="#111111")
    ax.text(SHEET_W - 0.45, 0.6, project[0], fontsize=10, ha="right", va="center")
    ax.text(SHEET_W - 0.45, 0.43, project[1], fontsize=6.5, ha="right", va="center")
    ax.text(SHEET_W - 3.6, 0.72, number, fontsize=16, ha="right", va="center", fontweight="bold")
    ax.text(SHEET_W - 3.6, 0.47, "SCHEMATIC - NOT FOR CONSTRUCTION", fontsize=5.5, ha="right", va="center",
            color="#AA2222")
    ax.text(SHEET_W - 3.6, 1.12, "AI-generated concept schematic", fontsize=5.5, ha="right", va="center", color="#666666")
    if legend:
        x = 4.2
        for kind, lab in LEGEND:
            ax.add_patch(Rectangle((x, 0.62), 0.28, 0.2, fc=FILL[kind], ec=INK, lw=0.4))
            ax.text(x + 0.36, 0.72, lab, fontsize=6.5, va="center")
            x += 0.5 + 0.075 * len(lab)
    return fig, ax


def view_axes(fig, x_in: float, y_in: float, xr: tuple, yr: tuple, ft_per_in: float):
    """Axes at a true drawing scale: data range xr/yr (ft) drawn at ft_per_in."""
    w = (xr[1] - xr[0]) / ft_per_in
    h = (yr[1] - yr[0]) / ft_per_in
    ax = fig.add_axes([x_in / SHEET_W, y_in / SHEET_H, w / SHEET_W, h / SHEET_H])
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax, w, h


def view_title(ax, x, y, num, name, scale, size=1.0):
    ax.add_patch(Circle((x, y), 1.6 * size, fill=False, ec=INK, lw=0.5))
    ax.text(x, y, str(num), fontsize=5, ha="center", va="center")
    ax.text(x + 3 * size, y + 0.8 * size, name.upper(), fontsize=7, va="center")
    ax.text(x + 3 * size, y - 1.6 * size, scale, fontsize=6, va="center")
    ax.plot([x + 2.5 * size, x + 60 * size], [y - 0.3 * size, y - 0.3 * size], color=INK, lw=0.4)


def dim_h(ax, x0, x1, y, text=None, fs=4.3, tick=1.0, ext_from=None):
    ax.plot([x0 - 1, x1 + 1], [y, y], color=INK, lw=0.35)
    for x in (x0, x1):
        ax.plot([x - tick / 2, x + tick / 2], [y - tick / 2, y + tick / 2], color=INK, lw=0.8)
        if ext_from is not None:
            ax.plot([x, x], [ext_from, y + 0.8], color=INK, lw=0.25)
    ax.text((x0 + x1) / 2, y + 0.5, text or ftin(x1 - x0), fontsize=fs, ha="center", va="bottom")


def dim_v(ax, y0, y1, x, text=None, fs=4.3, tick=1.0, ext_from=None):
    ax.plot([x, x], [y0 - 1, y1 + 1], color=INK, lw=0.35)
    for y in (y0, y1):
        ax.plot([x - tick / 2, x + tick / 2], [y - tick / 2, y + tick / 2], color=INK, lw=0.8)
        if ext_from is not None:
            ax.plot([ext_from, x - 0.8], [y, y], color=INK, lw=0.25)
    ax.text(x - 0.5, (y0 + y1) / 2, text or ftin(y1 - y0), fontsize=fs, ha="right", va="center", rotation=90)


def bubble(ax, x, y, text, r=3.0, fs=6.5):
    ax.add_patch(Circle((x, y), r, fc="white", ec=GRID, lw=0.6, zorder=5))
    ax.text(x, y, text, fontsize=fs, ha="center", va="center", color="#555555", zorder=6)


def grid_lines(sch) -> list[float]:
    """Column grid: core line, then even bays of <= 32 ft (meeting: 30-34 ft bays) to
    the tower end, continued over any Level 1 extension."""
    tower = sch.levels[1].length_ft
    ground = sch.levels[0].length_ft
    xs = [0.0, L.CORE_W_FT]
    n = max(1, math.ceil((tower - L.CORE_W_FT) / 32.0))
    xs += [L.CORE_W_FT + (tower - L.CORE_W_FT) * i / n for i in range(1, n + 1)]
    if ground > tower + 1:
        m = max(1, math.ceil((ground - tower) / 32.0))
        xs += [tower + (ground - tower) * i / m for i in range(1, m + 1)]
    return xs


# ------------------------------------------------------- unit geometry -
def unit_geometry(name: str, w: float, d: float = L.ROOM_DEPTH_FT) -> dict:
    """Guest unit in local coords: u along the corridor (0..w), v from corridor (0) to
    exterior (d). Returns walls, doors, fixtures (2D), furniture boxes (3D), windows."""
    n = name.lower()
    ada, spa = "ada" in n, "spa" in n
    qq = "qq" in n or "queen" in n or "double" in n
    suite = "suite" in n or w >= 20
    bd = 10.0 if ada else 8.5
    bw = min(w - 4.5, 10.5 if ada else 8.5)
    bw = max(bw, 5.0)
    bx = w - bw
    g = {"walls": [], "doors": [], "fix": [], "furn": [], "win": [], "ada": ada}
    W, Dr, F, B = g["walls"], g["doors"], g["fix"], g["furn"]
    W.append(((0, 0), (w, 0), 0.5, [(1.0, 4.0)]))  # corridor wall + entry
    W.append(((0, d), (w, d), 0.8, []))  # exterior
    W.append(((0, 0), (0, d), 0.4, []))
    W.append(((w, 0), (w, d), 0.4, []))
    W.append(((bx, 0), (bx, bd), 0.33, [(bd - 3.6, bd - 0.6)]))
    W.append(((bx, bd), (w, bd), 0.33, []))
    Dr.append(((1.0, 0.0), (1, 0), (0, 1), 3.0))
    Dr.append(((bx, bd - 0.6), (0, -1), (-1, 0) if ada else (1, 0), 3.0))
    # bath
    F.append(("rect", w - 2.0, 0.4, 1.8, 3.0, "solid"))
    F.append(("ellipse", w - 1.0, 1.9, 0.6, 0.8, "solid"))
    F.append(("rect", w - 0.8, 3.6, 0.7, 1.8, "solid"))
    F.append(("ellipse", w - 1.9, 4.5, 1.1, 0.75, "solid"))
    sh = (bx + 0.15, bd - 2.75, bw - 0.3, 2.6)
    F.append(("rect", *sh, "dashed" if ada else "shower"))
    B += [(w - 2.0, 0.4, 1.8, 3.0, 2.9, "#E8E4DC"), (w - 1.6, 3.6, 1.5, 1.8, 1.4, "#FFFFFF")]
    if ada:
        F.append(("circle", bx + bw / 2, bd / 2, 2.5, "dashed"))
        F.append(("line", [(w - 0.2, 3.2), (w - 0.2, 6.2)], "bar"))
    # closet in the entry lane
    if bx >= 3.5:
        F.append(("rect", 0.25, bd - 4.2, 2.0, 4.0, "closet"))
        B.append((0.25, bd - 4.2, 2.0, 4.0, 7.0, "#C9B38F"))
    z0, z1 = bd + 0.5, d - 0.4
    vc = z0 + (z1 - z0) * 0.45

    def bed(u0, v0, du, dv, head_at_u0=True):
        F.append(("rect", u0, v0, du, dv, "bed"))
        hu = u0 if head_at_u0 else u0 + du - 0.5
        F.append(("rect", hu, v0, 0.5, dv, "solid"))
        pu = u0 + 0.7 if head_at_u0 else u0 + du - 2.2
        if dv > 5.5:
            F.append(("rect", pu, v0 + 0.4, 1.5, dv / 2 - 0.6, "solid"))
            F.append(("rect", pu, v0 + dv / 2 + 0.2, 1.5, dv / 2 - 0.6, "solid"))
        else:
            F.append(("rect", pu, v0 + 0.4, 1.5, dv - 0.8, "solid"))
        B.append((u0, v0, du, dv, 2.0, "#FAFAFA"))
        B.append((hu, v0, 0.5, dv, 4.0, "#7B5E45"))

    def stand(u0, v0):
        F.append(("rect", u0, v0, 1.6, 1.6, "solid"))
        B.append((u0, v0, 1.6, 1.6, 2.0, "#8A6A4F"))

    if suite:
        up = w * 0.45
        W.append(((up, bd), (up, d), 0.33, [(1.0, 4.0)]))
        Dr.append(((up, bd + 1.0), (0, 1), (1, 0), 3.0))
        bed(w - 7.25, vc - 3.35, 7.0, 6.7, head_at_u0=False)
        stand(w - 1.85, vc - 3.35 - 1.8)
        stand(w - 1.85, vc + 3.35 + 0.2)
        F.append(("rect", 0.25, vc - 3.5, 3.0, 7.0, "solid"))  # sofa
        F.append(("rect", 4.3, vc - 1.5, 2.2, 3.0, "solid"))  # coffee table
        F.append(("circle", 8.3, vc - 2.4, 1.1, "solid"))
        F.append(("circle", 8.3, vc + 2.4, 1.1, "solid"))
        B += [(0.25, vc - 3.5, 3.0, 7.0, 2.6, "#7F93A6"), (4.3, vc - 1.5, 2.2, 3.0, 1.5, "#8A6A4F"),
              (7.2, vc - 3.5, 2.2, 2.2, 2.6, "#B98C5E"), (7.2, vc + 1.3, 2.2, 2.2, 2.6, "#B98C5E")]
        g["win"] += [(up * 0.2, up * 0.8), (up + (w - up) * 0.2, up + (w - up) * 0.8)]
    elif qq:
        stand(0.25, z0 + 0.2)
        bed(0.25, z0 + 2.0, 6.7, 5.0)
        stand(0.25, z0 + 7.2)
        bed(0.25, z0 + 8.9, 6.7, 5.0)
        F.append(("rect", w - 1.8, z0 + 3.0, 1.8, 6.0, "solid"))
        B.append((w - 1.8, z0 + 3.0, 1.8, 6.0, 3.0, "#8A6A4F"))
        g["win"].append((w * 0.25, w * 0.75))
    else:
        stand(0.25, vc - 3.35 - 1.8)
        bed(0.25, vc - 3.35, 7.0, 6.7)
        stand(0.25, vc + 3.35 + 0.2)
        F.append(("rect", w - 1.8, vc - 3.0, 1.8, 6.0, "solid"))
        B.append((w - 1.8, vc - 3.0, 1.8, 6.0, 3.0, "#8A6A4F"))
        if not spa and not ada:
            F.append(("circle", w - 2.2, z1 - 2.2, 1.2, "solid"))
            B.append((w - 3.3, z1 - 3.3, 2.2, 2.2, 2.6, "#B98C5E"))
        g["win"].append((w * 0.25, w * 0.75))
    if spa:
        F.append(("ellipse", w - 3.2, z1 - 3.2, 1.4, 2.8, "solid"))
        B.append((w - 4.6, z1 - 6.0, 2.8, 5.6, 2.0, "#FFFFFF"))
    if ada:
        F.append(("circle", w - 3.0 if not suite else w * 0.45 - 3.0, z1 - 3.2, 2.5, "dashed"))
    return g


def _split(a, b, holes):
    """Solid pieces of segment a->b with holes given as distances along it."""
    length = math.dist(a, b)
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    cuts, s = [], 0.0
    for h0, h1 in sorted(holes):
        if h0 > s:
            cuts.append((s, h0))
        s = max(s, h1)
    if s < length:
        cuts.append((s, length))
    return [((a[0] + ux * s0, a[1] + uy * s0), (a[0] + ux * s1, a[1] + uy * s1)) for s0, s1 in cuts]


def _wall_poly(p, q, t):
    (x0, y0), (x1, y1) = p, q
    if abs(x0 - x1) < 1e-6:
        return [(x0 - t / 2, y0), (x0 + t / 2, y0), (x1 + t / 2, y1), (x1 - t / 2, y1)]
    return [(x0, y0 - t / 2), (x1, y1 - t / 2), (x1, y1 + t / 2), (x0, y0 + t / 2)]


def draw_unit(ax, g, xf, lw_scale=1.0, wall_color=INK, walls=True):
    """Draw unit geometry `g` through transform xf(u, v) -> (x, y)."""
    T = lambda pts: [xf(u, v) for u, v in pts]  # noqa: E731
    if walls:
        for a, b, t, holes in g["walls"]:
            for p, q in _split(a, b, holes):
                ax.add_patch(Polygon(T(_wall_poly(p, q, t)), closed=True, fc=wall_color, ec="none", zorder=4))
    for (hu, hv), (au, av), (iu, iv), wd in g["doors"]:
        leaf_end = (hu + iu * wd, hv + iv * wd)
        ang0, ang1 = math.atan2(iv, iu), math.atan2(av, au)
        if ang1 - ang0 > math.pi:
            ang1 -= 2 * math.pi
        elif ang0 - ang1 > math.pi:
            ang1 += 2 * math.pi
        arc = [(hu + wd * math.cos(t), hv + wd * math.sin(t)) for t in np.linspace(ang0, ang1, 16)]
        ax.plot(*zip(*T([(hu, hv), leaf_end])), color=INK, lw=0.6 * lw_scale, zorder=5)
        ax.plot(*zip(*T(arc)), color=INK, lw=0.3 * lw_scale, zorder=5)
    for f in g["fix"]:
        kind, style = f[0], f[-1]
        ls = "--" if style == "dashed" else "-"
        lw = 0.35 * lw_scale
        if kind == "rect":
            _, u, v, du, dv, _ = f
            pts = [(u, v), (u + du, v), (u + du, v + dv), (u, v + dv)]
            fc = {"bed": "white", "closet": "#F5F0E6"}.get(style, "none")
            ax.add_patch(Polygon(T(pts), closed=True, fc=fc, ec=INK, lw=lw, ls=ls, zorder=4.5))
            if style == "shower":
                ax.plot(*zip(*T([(u, v), (u + du, v + dv)])), color=INK, lw=0.25 * lw_scale, zorder=4.6)
                ax.plot(*zip(*T([(u, v + dv), (u + du, v)])), color=INK, lw=0.25 * lw_scale, zorder=4.6)
            if style == "closet":
                ax.plot(*zip(*T([(u + du / 2, v), (u + du / 2, v + dv)])), color=INK, lw=0.25 * lw_scale, zorder=4.6)
        elif kind == "ellipse":
            _, cu, cv, ru, rv, _ = f
            pts = [(cu + ru * math.cos(t), cv + rv * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 28)]
            ax.add_patch(Polygon(T(pts), closed=True, fc="white", ec=INK, lw=lw, zorder=4.7))
        elif kind == "circle":
            _, cu, cv, r, _ = f
            pts = [(cu + r * math.cos(t), cv + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 40)]
            ax.add_patch(Polygon(T(pts), closed=True, fc="none", ec=INK if style != "dashed" else "#666666",
                                 lw=lw, ls=ls, zorder=4.7))
        elif kind == "line":
            ax.plot(*zip(*T(f[1])), color=INK, lw=1.0 * lw_scale, zorder=4.7)
    d = max(b[1] for _, b, _, _ in g["walls"])
    for u0, u1 in g["win"]:
        for off in (-0.25, 0.25):
            ax.plot(*zip(*T([(u0, d + off), (u1, d + off)])), color="#3B6FA0", lw=0.6 * lw_scale, zorder=6)
        ax.add_patch(Polygon(T([(u0, d - 0.4), (u1, d - 0.4), (u1, d + 0.4), (u0, d + 0.4)]), closed=True,
                             fc="white", ec="none", zorder=5.5))


def room_xf(r, mirror: bool):
    """Local (u, v) of a guest room -> plan (x, y); v runs away from the corridor."""
    def xf(u, v):
        x = r.x + (r.w - u if mirror else u)
        y = r.y + r.d - v if r.side == "S" else r.y + v
        return x, y
    return xf


# ------------------------------------------------------------- plans -
def draw_stair(ax, r):
    ax.add_patch(Rectangle((r.x, r.y), r.w, r.d, fc=FILL["stair"], ec=INK, lw=0.6, zorder=3))
    land = 4.5
    y0, y1 = r.y + land, r.y + r.d - land
    xm = r.x + r.w / 2
    ax.plot([xm, xm], [y0, y1], color=INK, lw=0.6, zorder=3.5)
    for y in np.arange(y0, y1 + 0.01, 0.92):
        ax.plot([r.x + 0.3, r.x + r.w - 0.3], [y, y], color="#777777", lw=0.25, zorder=3.4)
    ax.annotate("", xy=(r.x + r.w * 0.25, y1 - 1), xytext=(r.x + r.w * 0.25, y0 + 1),
                arrowprops=dict(arrowstyle="->", lw=0.5, color=INK), zorder=3.6)
    ax.text(xm + r.w * 0.25, (y0 + y1) / 2, "STAIR", fontsize=3.8, ha="center", va="center", rotation=90, zorder=7)


def draw_elevator(ax, r):
    ax.add_patch(Rectangle((r.x, r.y), r.w, r.d, fc="white", ec=INK, lw=0.6, zorder=3))
    ax.plot([r.x, r.x + r.w], [r.y, r.y + r.d], color=INK, lw=0.3, zorder=3.5)
    ax.plot([r.x, r.x + r.w], [r.y + r.d, r.y], color=INK, lw=0.3, zorder=3.5)


def _door(ax, hx, hy, along, into, wd=3.0, lw=0.5):
    ang0, ang1 = math.atan2(into[1], into[0]), math.atan2(along[1], along[0])
    if ang1 - ang0 > math.pi:
        ang1 -= 2 * math.pi
    elif ang0 - ang1 > math.pi:
        ang1 += 2 * math.pi
    ax.plot([hx, hx + along[0] * wd], [hy, hy + along[1] * wd], color="white", lw=1.6, zorder=4.8)
    ax.plot([hx, hx + into[0] * wd], [hy, hy + into[1] * wd], color=INK, lw=lw, zorder=5)
    t = np.linspace(ang0, ang1, 16)
    ax.plot(hx + wd * np.cos(t), hy + wd * np.sin(t), color=INK, lw=lw * 0.5, zorder=5)


def draw_level(ax, sch, lvl, gx, show_grid=True, show_dims=True, fs=4.3):
    W = lvl.width_ft
    Lx = lvl.length_ft
    tower = sch.levels[1].length_ft
    yc0, yc1 = L.ROOM_DEPTH_FT, L.ROOM_DEPTH_FT + L.CORRIDOR_FT
    if show_grid:
        for i, x in enumerate(gx):
            if x > Lx + 0.5:
                continue
            ax.plot([x, x], [-3, W + 16], color=GRID, lw=0.4, ls=(0, (8, 3, 2, 3)), zorder=1)
            bubble(ax, x, W + 19, str(i + 1))
        for j, y in enumerate(GRID_Y):
            ax.plot([-15, Lx + 3], [y, y], color=GRID, lw=0.4, ls=(0, (8, 3, 2, 3)), zorder=1)
            bubble(ax, -19, y, "ABCDEFGH"[j])
    # fills
    for r in lvl.rooms:
        ax.add_patch(Rectangle((r.x, r.y), r.w, r.d, fc=FILL.get(r.kind, "white"), ec="none", zorder=2))
    for r in lvl.rooms:
        if r.kind not in ("stair", "elevator"):
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.d, fill=False, ec=INK, lw=0.45, zorder=3))
        if r.kind == "shaft":
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.d, fill=False, ec="#999999", lw=0, hatch="////", zorder=2.5))
    # guest units with fixtures, alternate units mirrored so baths pair up
    count = {"S": 0, "N": 0}
    for r in lvl.rooms:
        if r.kind == "guest":
            g = unit_geometry(r.label, r.w, r.d)
            draw_unit(ax, g, room_xf(r, count[r.side] % 2 == 1), lw_scale=0.8)
            count[r.side] += 1
        elif r.kind == "stair":
            draw_stair(ax, r)
        elif r.kind == "elevator":
            draw_elevator(ax, r)
    # doors off the corridor for non-guest rooms
    for r in lvl.rooms:
        if r.kind in ("guest", "corridor", "elevator") or not r.label:
            continue
        on_s = abs(r.y + r.d - yc0) < 0.1 and r.x < (lvl.rooms and max(rr.x + rr.w for rr in lvl.rooms if rr.kind == "corridor"))
        on_n = abs(r.y - yc1) < 0.1 and r.x < max(rr.x + rr.w for rr in lvl.rooms if rr.kind == "corridor")
        wd = min(3.0, max(1.5, r.w - 1.0))
        if r.kind == "stair" and r.y < yc0 and on_s:
            _door(ax, r.x + 2.0, yc0, (1, 0), (0, -1), wd)
        elif r.kind == "stair" and on_n:
            _door(ax, r.x + 2.0, yc1, (1, 0), (0, 1), wd)
        elif on_s and r.kind != "stair":
            _door(ax, r.x + 1.0, yc0, (1, 0), (0, -1), wd)
        elif on_n and r.kind != "stair":
            _door(ax, r.x + 1.0, yc1, (1, 0), (0, 1), wd)
    # Level 1 specifics: entries, storefront, stair discharge
    if lvl.name == L.GROUND_LEVEL:
        for r in lvl.rooms:
            n = r.label.lower()
            if "lobby" in n:
                cx = r.x + r.w / 2
                _door(ax, cx - 6, 0, (1, 0), (0, 1), 3.0)
                _door(ax, cx + 6, 0, (-1, 0), (0, 1), 3.0)
                for x0 in np.arange(r.x + 4, r.x + r.w - 6, 8):
                    if abs(x0 + 3 - cx) > 8:
                        ax.plot([x0, x0 + 6], [0, 0], color=GLASS, lw=2.4, zorder=8)
            if r.kind == "tenant" and r.label:
                for x0 in np.arange(r.x + 2, r.x + r.w - 6, 8):
                    ax.plot([x0, x0 + 6], [0, 0], color=GLASS, lw=2.4, zorder=8)
                _door(ax, r.x + r.w / 2 - 1.5, 0, (1, 0), (0, 1), 3.0)
            if "loading" in n:
                ax.plot([r.x + 1, r.x + r.w - 1], [W, W], color="white", lw=2.5, zorder=6)
                ax.add_patch(Rectangle((r.x + 1, W - 3), r.w - 2, 3, fill=False, ec=INK, lw=0.35, ls="--", zorder=6))
            if r.kind == "stair" and r.y >= yc1:
                _door(ax, r.x + 3.5, W, (1, 0), (0, 1), 3.0)
            if r.kind == "stair" and r.y < yc0:
                _door(ax, 0, r.y + 3.0, (0, 1), (-1, 0), 3.0)
    # exterior walls: podium and tower outline
    ax.add_patch(Rectangle((0, 0), Lx, W, fill=False, ec=INK, lw=1.8, zorder=7))
    if lvl.name == L.GROUND_LEVEL and Lx > tower + 0.5:
        ax.plot([tower, tower], [0, W], color=INK, lw=0.5, ls="--", zorder=7)
        ax.text(tower + 0.8, W - 1.5, "TOWER ABOVE", fontsize=3.2, va="top", rotation=90, zorder=8)
    # labels
    for r in lvl.rooms:
        if not r.label or r.kind in ("elevator", "stair") or r.w < 3.5:
            continue
        col = "white" if r.kind in DARK else INK
        if r.kind == "guest":
            v = (8.5 if "ada" not in r.label.lower() else 10.0) + 1.6
            cy = r.y + r.d - v if r.side == "S" else r.y + v
            txt = wrap(r.label.upper(), r.w + 3, 0.95)
            ax.text(r.x + r.w / 2, cy, f"{txt}\n{r.area:,.0f} SF", fontsize=fs * 0.82, ha="center",
                    va="center", color=col, zorder=8, fontweight="bold", linespacing=1.1)
        elif r.kind == "corridor":
            if lvl.name == L.GROUND_LEVEL:
                ax.text(r.x + r.w * 0.55, r.y + r.d / 2, "CORRIDOR", fontsize=fs, ha="center", va="center",
                        color=col, zorder=8, fontweight="bold")
            else:
                ax.text(r.x + r.w * 0.55, r.y + r.d / 2, f"CORRIDOR   {r.w * r.d - L.CORE_W_FT * r.d:,.0f} SF",
                        fontsize=fs, ha="center", va="center", color=col, zorder=8, fontweight="bold")
        else:
            txt = wrap(r.label.upper(), r.w, 0.95 if r.w > 8 else 0.7)
            ax.text(r.x + r.w / 2, r.y + r.d / 2, f"{txt}\n{r.area:,.0f} SF", fontsize=fs * (0.9 if r.w > 9 else 0.65),
                    ha="center", va="center", color=col, zorder=8, fontweight="bold", linespacing=1.1,
                    rotation=90 if r.w < 6 and r.d > 10 else 0)
    if show_dims:
        xs = [x for x in gx if x <= Lx + 0.5]
        for a, b in zip(xs, xs[1:]):
            dim_h(ax, a, b, W + 5, fs=fs * 0.85)
        dim_h(ax, 0, Lx, W + 10.5, fs=fs, ext_from=W + 1)
        for a, b in zip(GRID_Y, GRID_Y[1:]):
            dim_v(ax, a, b, -7, fs=fs * 0.85)
        dim_v(ax, 0, W, -12, fs=fs)


def plan_sheet(sch, levels, number, title, out_dir: Path, scale=20.0):
    fig, sax = new_sheet(title, number, legend=True)
    gx = grid_lines(sch)
    Lmax = max(lv.length_ft for lv in sch.levels)
    xr = (-26.0, Lmax + 6.0)
    yr = (-13.0, L.WING_W_FT + 23.0)
    h = (yr[1] - yr[0]) / scale
    w = (xr[1] - xr[0]) / scale
    x0 = (SHEET_W - w) / 2
    tops = [SHEET_H - 0.35 - h, SHEET_H - 0.35 - 2 * h - 0.15]
    for i, lvl in enumerate(levels):
        ax, _, _ = view_axes(fig, x0, tops[i], xr, yr, scale)
        draw_level(ax, sch, lvl, gx)
        typ = "" if lvl.name == L.GROUND_LEVEL else f"  -  {lvl.keys_placed} KEYS"
        view_title(ax, -22, -8.5, i + 1, f"{lvl.name}{typ}", f"1\" = {scale:.0f}'-0\"  (on 18 x 12 sheet)")
    return _save(fig, out_dir, number)


# ------------------------------------------------- enlarged unit plans -
def unit_types(program):
    out = []
    for name, count, area, cat in program.rows:
        if cat == "guest" and count:
            out.append((name, count, area / count))
    return out


def enlarged_sheets(sch, out_dir: Path, scale=8.0):
    types = unit_types(sch.program)
    d = L.ROOM_DEPTH_FT
    cell_h = (d + 13) / scale
    avail_w = SHEET_W - 1.0
    rows, row, used = [], [], 0.0
    for t in types:
        wft = t[2] / d
        cw = (wft + 10.5) / scale + 0.12
        if row and used + cw > avail_w:
            rows.append(row)
            row, used = [], 0.0
        row.append(t)
        used += cw
    if row:
        rows.append(row)
    per_sheet = max(1, int((SHEET_H - 1.9) // cell_h))
    paths, k = [], 0
    for si in range(0, len(rows), per_sheet):
        number = f"A40{si // per_sheet + 1}"
        fig, sax = new_sheet("L+L Enlarged Plans", number)
        for ri, row in enumerate(rows[si:si + per_sheet]):
            x = 0.5
            ytop = SHEET_H - 0.4 - (ri + 1) * cell_h
            for name, count, area in row:
                wft = area / d
                xr = (-4.0, wft + 6.5)
                yr = (-9.0, d + 4.0)
                ax, aw, _ = view_axes(fig, x, ytop, xr, yr, scale)
                ax.add_patch(Rectangle((0, 0), wft, d, fc="#F8F5EF", ec="none", zorder=1))
                ax.add_patch(Rectangle((-2, -3), wft + 4, 3, fc=FILL["corridor"], ec="none", alpha=0.35, zorder=1))
                g = unit_geometry(name, wft, d)
                draw_unit(ax, g, lambda u, v: (u, v), lw_scale=1.4)
                ax.text(wft * 0.5 if wft < 20 else wft * 0.72, g_bd(g) + 2.5, f"{name.upper()}\n{area:,.0f} SF",
                        fontsize=5.5, ha="center", va="center", fontweight="bold", zorder=9)
                dim_h(ax, 0, wft, d + 2.2, fs=4.5, ext_from=d + 0.5)
                dim_v(ax, 0, d, wft + 4.0, fs=4.5, ext_from=wft + 0.5)
                dim_v(ax, 0, g_bd(g), -2.0, fs=3.8)
                k += 1
                view_title(ax, -2.5, -6.6, k, f"Enlarged plan - {name}  (x{count})", "1/8\" = 1'-0\"", size=0.45)
                x += aw + 0.12
        paths.append(_save(fig, out_dir, number))
    return paths


def g_bd(g):
    for a, b, t, holes in g["walls"]:
        if abs(a[1] - b[1]) < 1e-6 and 0 < a[1] < 15 and t < 0.4:
            return a[1]
    return 8.5


# ------------------------------------------------------- isometric -
class Iso:
    """Orthographic axonometric projector with painter's-algorithm drawing."""

    def __init__(self, az_deg=-55.0, el_deg=32.0):
        th, ph = math.radians(az_deg), math.radians(el_deg)
        self.c = np.array([math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th), math.sin(ph)])
        self.r = np.array([-math.sin(th), math.cos(th), 0.0])
        self.u = np.cross(self.c, self.r)
        self.items = []  # (depth, polys[(pts, fc, ec, lw)])

    def p(self, pts):
        a = np.asarray(pts, float)
        return np.c_[a @ self.r, a @ self.u]

    def box(self, x, y, z, dx, dy, dz, color, edge="#555555", lw=0.35, extra=0.0):
        x1, y1, z1 = x + dx, y + dy, z + dz
        faces = [((0, 0, 1), [(x, y, z1), (x1, y, z1), (x1, y1, z1), (x, y1, z1)], 1.0),
                 ((0, 0, -1), [(x, y, z), (x1, y, z), (x1, y1, z), (x, y1, z)], 0.5),
                 ((1, 0, 0), [(x1, y, z), (x1, y1, z), (x1, y1, z1), (x1, y, z1)], 0.80),
                 ((-1, 0, 0), [(x, y, z), (x, y1, z), (x, y1, z1), (x, y, z1)], 0.80),
                 ((0, 1, 0), [(x, y1, z), (x1, y1, z), (x1, y1, z1), (x, y1, z1)], 0.66),
                 ((0, -1, 0), [(x, y, z), (x1, y, z), (x1, y, z1), (x, y, z1)], 0.66)]
        rgb = np.array(matplotlib.colors.to_rgb(color))
        polys = []
        for n, pts, shade in faces:
            if np.dot(n, self.c) > 1e-6:
                polys.append((self.p(pts), tuple(np.clip(rgb * shade, 0, 1)), edge, lw))
        cen = np.array([x + dx / 2, y + dy / 2, z + dz / 2])
        self.items.append((float(cen @ self.c) + extra, polys))

    def quad(self, pts, normal, color, edge="#555555", lw=0.3, lines=(), extra=0.05):
        if np.dot(normal, self.c) <= 1e-6:
            return
        cen = np.mean(np.asarray(pts, float), axis=0)
        polys = [(self.p(pts), color, edge, lw)]  # extra: draw-order bias
        for a, b in lines:
            polys.append((self.p([a, b]), None, edge, lw * 0.8))
        self.items.append((float(cen @ self.c) + extra, polys))

    def render(self, ax, sort=True):
        items = sorted(self.items, key=lambda t: t[0]) if sort else self.items
        for _, polys in items:
            for pts, fc, ec, lw in polys:
                if fc is None:  # a patch, not ax.plot, so lines obey the painter's order
                    ax.add_patch(Polygon(pts, closed=False, fill=False, ec=ec, lw=lw))
                else:
                    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw))
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.axis("off")


def iso_unit(ax, name, w, d=L.ROOM_DEPTH_FT, az=-55.0, H=9.0):
    """Unit model: walls with door headers, closed doors, windows, floor and furniture."""
    g = unit_geometry(name, w, d)
    iso = Iso(az, 34)
    iso.box(-0.4, -0.4, -0.6, w + 0.8, d + 0.8, 0.6, "#D9CDB6", lw=0.3, extra=-1e4)
    def order(a, b):
        # Perimeter walls: camera-side ones go last (they hide the interior), far ones first.
        normal = None
        if a[1] == b[1] == 0:
            normal = (0, -1, 0)
        elif a[1] == b[1] == d:
            normal = (0, 1, 0)
        elif a[0] == b[0] == 0:
            normal = (-1, 0, 0)
        elif a[0] == b[0] == w:
            normal = (1, 0, 0)
        if normal is None:
            return 0.0
        return 1e4 if np.dot(normal, iso.c) > 0 else -1e3

    for a, b, t, holes in g["walls"]:
        ex = order(a, b)
        for p, q in _split(a, b, holes):
            pp = _wall_poly(p, q, t)
            px, py = [c[0] for c in pp], [c[1] for c in pp]
            iso.box(min(px), min(py), 0, max(px) - min(px), max(py) - min(py), H, "#CFCFCF", extra=ex)
        for h0, h1 in holes:
            length = math.dist(a, b)
            ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            p = (a[0] + ux * h0, a[1] + uy * h0)
            q = (a[0] + ux * h1, a[1] + uy * h1)
            pp = _wall_poly(p, q, t)
            px, py = [c[0] for c in pp], [c[1] for c in pp]
            iso.box(min(px), min(py), 7.0, max(px) - min(px), max(py) - min(py), H - 7.0, "#CFCFCF", extra=ex)
            iso.box(min(px) + (0.0 if ux else t * 0.35), min(py) + (t * 0.35 if ux else 0.0), 0,
                    (max(px) - min(px)) if ux else t * 0.3, t * 0.3 if ux else (max(py) - min(py)), 7.0,
                    "#9C6B3F", edge="#5A3B20", extra=ex)
    ext = order((0, d), (w, d))
    for u0, u1 in g["win"]:
        for vv, nrm in ((d - 0.41, (0, -1, 0)), (d + 0.41, (0, 1, 0))):
            pts = [(u0, vv, 2.5), (u1, vv, 2.5), (u1, vv, 8.0), (u0, vv, 8.0)]
            um = (u0 + u1) / 2
            iso.quad(pts, nrm, "#BFE0F2", edge="#3B6FA0", lw=0.5, extra=ext + 50.0,
                     lines=[((um, vv, 2.5), (um, vv, 8.0)), ((u0, vv, 5.6), (u1, vv, 5.6))])
    for u, v, du, dv, h, col in g["furn"]:
        iso.box(u, v, 0, du, dv, h, col, edge="#666666", lw=0.25)
    iso.render(ax)


def iso_building(ax, sch, az=-58.0, el=30.0):
    up = sch.levels[1:]
    g0 = sch.levels[0]
    W = L.WING_W_FT
    iso = Iso(az, el)
    top = sch.height_ft
    # soft ground shadow
    sh = [(0, 0, 0), (g0.length_ft, 0, 0), (g0.length_ft + 30, W + 25, 0), (30, W + 25, 0)]
    ax.add_patch(Polygon(iso.p(sh), closed=True, fc="#EDEDED", ec="none"))
    iso.box(0, 0, 0, g0.length_ft, W, g0.height_ft, "#E2E2E2", edge="#777777", lw=0.5)
    for lv in up:
        iso.box(0, 0, lv.elevation_ft, lv.length_ft, W, lv.height_ft, "#EBEBEB", edge="#777777", lw=0.5)
    Lt = up[0].length_ft
    iso.box(0, 0, top, Lt, W, 3.0, "#F4F4F4", edge="#777777", lw=0.5)
    iso.box(Lt * 0.35, W * 0.35, top + 3.0, 26, 12, 5, "#F7F7F7", edge="#888888", lw=0.4)
    iso.render(ax, sort=False)
    iso.items = []
    # windows on guest rooms (both long faces), storefront at Level 1
    for lv in up:
        z = lv.elevation_ft
        for r in lv.rooms:
            if r.kind != "guest":
                continue
            cx = r.x + r.w / 2
            if r.side == "S":
                iso.quad([(cx - 1.7, -0.05, z + 2.5), (cx + 1.7, -0.05, z + 2.5), (cx + 1.7, -0.05, z + 8.6),
                          (cx - 1.7, -0.05, z + 8.6)], (0, -1, 0), "#B9C9D6", edge="#555555", lw=0.35)
            else:
                iso.quad([(cx - 1.7, W + 0.05, z + 2.5), (cx + 1.7, W + 0.05, z + 2.5), (cx + 1.7, W + 0.05, z + 8.6),
                          (cx - 1.7, W + 0.05, z + 8.6)], (0, 1, 0), "#B9C9D6", edge="#555555", lw=0.35)
        for yy in (8.0, 20.0, 38.0, 50.0):
            iso.quad([(Lt + 0.05, yy - 1.7, z + 2.5), (Lt + 0.05, yy + 1.7, z + 2.5), (Lt + 0.05, yy + 1.7, z + 8.6),
                      (Lt + 0.05, yy - 1.7, z + 8.6)], (1, 0, 0), "#B9C9D6", edge="#555555", lw=0.35)
    for r in g0.rooms:
        n = r.label.lower()
        if r.y > 0.1 or not ("lobby" in n or r.kind == "tenant"):
            continue
        for x0 in np.arange(r.x + 2, r.x + r.w - 6, 8.5):
            iso.quad([(x0, -0.05, 1.0), (x0 + 7, -0.05, 1.0), (x0 + 7, -0.05, 11.0), (x0, -0.05, 11.0)],
                     (0, -1, 0), "#A9C6DA", edge="#555555", lw=0.4)
    for yy in np.arange(4, W - 6, 9):
        iso.quad([(g0.length_ft + 0.05, yy, 1.0), (g0.length_ft + 0.05, yy + 7, 1.0),
                  (g0.length_ft + 0.05, yy + 7, 11.0), (g0.length_ft + 0.05, yy, 11.0)],
                 (1, 0, 0), "#A9C6DA", edge="#555555", lw=0.4)
    iso.render(ax, sort=False)
    # level markers on the west corner
    for lv in sch.levels + [None]:
        z = top if lv is None else lv.elevation_ft
        name = "ROOFTOP" if lv is None else lv.name.upper()
        p0 = iso.p([(0, 0, z)])[0]
        p1 = iso.p([(-40, 0, z)])[0]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#999999", lw=0.4, ls=(0, (6, 3)))
        ax.add_patch(Circle(p1, 1.4, fc="white", ec=INK, lw=0.5))
        ax.text(p1[0] - 2.5, p1[1] + 0.5, f"{name}\n{ftin(z)}", fontsize=5.5, ha="right", va="center")
    ax.autoscale_view()


# ------------------------------------------------------- elevations -
def elevation_south(ax, sch, gx):
    g0, up = sch.levels[0], sch.levels[1:]
    Lt = up[0].length_ft
    top = sch.height_ft
    ax.add_patch(Rectangle((0, 0), g0.length_ft, g0.height_ft, fc="#F4F4F4", ec=INK, lw=0.8))
    ax.add_patch(Rectangle((0, g0.height_ft), Lt, top - g0.height_ft + 3.0, fc="#F7F7F7", ec=INK, lw=0.8))
    ax.plot([-15, g0.length_ft + 15], [0, 0], color=INK, lw=1.4)
    for lv in up:
        z = lv.elevation_ft
        for r in lv.rooms:
            if r.kind == "guest" and r.side == "S":
                cx = r.x + r.w / 2
                ax.add_patch(Rectangle((cx - 1.7, z + 2.5), 3.4, 6.1, fc="white", ec=INK, lw=0.5))
                ax.add_patch(Rectangle((cx - 1.45, z + 2.75), 2.9, 5.6, fill=False, ec=INK, lw=0.25))
                ax.plot([cx - 1.2, cx + 0.4], [z + 3.2, z + 6.0], color="#BBBBBB", lw=0.25)
    for r in g0.rooms:
        n = r.label.lower()
        if r.y > 0.1:
            continue
        if "lobby" in n or r.kind == "tenant":
            for x0 in np.arange(r.x + 2, r.x + r.w - 6, 8.5):
                ax.add_patch(Rectangle((x0, 0.3), 7, 10.5, fc="white", ec=INK, lw=0.5))
                ax.add_patch(Rectangle((x0 + 0.3, 0.6), 6.4, 9.9, fill=False, ec=INK, lw=0.25))
        elif r.kind == "stair":
            ax.add_patch(Rectangle((r.x + 3, 0.3), 3.5, 7.5, fill=False, ec=INK, lw=0.5))
    # panel breakdown: joints at grid lines and floor lines (indicative)
    floors = [lv.elevation_ft for lv in up] + [top]
    for z in floors:
        ax.plot([-2, Lt + 2], [z, z], color=PANEL, lw=1.6, ls=(0, (5, 2.5)))
    for x in gx:
        if x <= Lt + 0.5:
            ax.plot([x, x], [up[0].elevation_ft - 1.5, top + 1.5], color=PANEL, lw=1.6, ls=(0, (5, 2.5)))
    for i, x in enumerate(gx):
        ax.plot([x, x], [top + 4, top + 9], color=GRID, lw=0.4, ls=(0, (6, 3)))
        bubble(ax, x, top + 12, str(i + 1))
    for lv in sch.levels + [None]:
        z = top if lv is None else lv.elevation_ft
        name = "ROOFTOP" if lv is None else lv.name.upper()
        ax.plot([-14, g0.length_ft + 18], [z, z], color="#AAAAAA", lw=0.3, ls=(0, (10, 3, 2, 3)))
        ax.add_patch(Circle((g0.length_ft + 20, z), 1.1, fc="white", ec=INK, lw=0.5))
        ax.add_patch(Polygon([(g0.length_ft + 20, z), (g0.length_ft + 21.1, z), (g0.length_ft + 20, z + 1.1)],
                             closed=True, fc=INK, ec="none"))
        ax.text(g0.length_ft + 22.5, z + 0.6, name, fontsize=5.5, va="bottom")
        ax.text(g0.length_ft + 22.5, z - 0.4, ftin(z), fontsize=5.5, va="top")


def elevation_east(ax, sch):
    g0, up = sch.levels[0], sch.levels[1:]
    W = L.WING_W_FT
    top = sch.height_ft
    ax.add_patch(Rectangle((0, 0), W, g0.height_ft, fc="#F4F4F4", ec=INK, lw=0.8))
    ax.add_patch(Rectangle((0, g0.height_ft), W, top - g0.height_ft + 3.0, fc="#F7F7F7", ec=INK, lw=0.8))
    ax.plot([-8, W + 8], [0, 0], color=INK, lw=1.4)
    for lv in up:
        for cx in (8.0, 20.0, 38.0, 50.0):
            ax.add_patch(Rectangle((cx - 1.7, lv.elevation_ft + 2.5), 3.4, 6.1, fc="white", ec=INK, lw=0.5))
    for x0 in np.arange(4, W - 6, 9):
        ax.add_patch(Rectangle((x0, 0.3), 7, 10.5, fc="white", ec=INK, lw=0.5))
    for z in [lv.elevation_ft for lv in up] + [top]:
        ax.plot([-1, W + 1], [z, z], color=PANEL, lw=1.6, ls=(0, (5, 2.5)))


def elevation_sheet(sch, out_dir: Path, scale=20.0):
    fig, sax = new_sheet("L+L Elevations", "A201")
    gx = grid_lines(sch)
    g0 = sch.levels[0]
    top = sch.height_ft
    xr = (-20.0, g0.length_ft + 45.0)
    yr = (-14.0, top + 17.0)
    ax, w, h = view_axes(fig, 0.8, SHEET_H - 0.5 - (yr[1] - yr[0]) / scale, xr, yr, scale)
    elevation_south(ax, sch, gx)
    view_title(ax, -16, -9, 1, "Building elevation - south (street)", f"1\" = {scale:.0f}'-0\"")
    xr2 = (-12.0, L.WING_W_FT + 12.0)
    yr2 = (-12.0, top + 6.0)
    ax2, w2, h2 = view_axes(fig, 0.8, SHEET_H - 0.8 - h - (yr2[1] - yr2[0]) / scale, xr2, yr2, scale)
    elevation_east(ax2, sch)
    view_title(ax2, -10, -9, 2, "Building elevation - east", f"1\" = {scale:.0f}'-0\"")
    sax.add_patch(Rectangle((6.0, 2.2), 0.8, 0.12, fc=PANEL, ec="none"))
    sax.text(6.95, 2.26, "PANEL BREAKDOWN (indicative: joints at column grid and floor lines, for prefab review)",
             fontsize=8, color=PANEL, va="center")
    return _save(fig, out_dir, "A201")


# ---------------------------------------------------------- overview -
def schedule(sax, x, y, title, header, rows, widths, row_h=0.165, fs=6.0, total=None):
    """Schedule table in sheet inches, styled like the Revit schedules on the L+L cover."""
    tw = sum(widths)
    sax.add_patch(Rectangle((x, y - row_h * 1.2), tw, row_h * 1.2, fill=False, ec="#777777", lw=0.5))
    sax.text(x + tw / 2, y - row_h * 0.6, title.upper(), fontsize=fs, ha="center", va="center")
    y -= row_h * 1.2
    cx = x
    for wd, hd in zip(widths, header):
        sax.add_patch(Rectangle((cx, y - row_h), wd, row_h, fill=False, ec="#777777", lw=0.4))
        sax.text(cx + wd / 2, y - row_h / 2, hd.upper(), fontsize=fs * 0.9, ha="center", va="center")
        cx += wd
    y -= row_h * 1.5
    for row in rows:
        if isinstance(row, str):  # group heading
            sax.text(x + 0.03, y - row_h / 2, row, fontsize=fs, va="center")
            y -= row_h
            continue
        is_total = row and row[0] == "__total__"
        cells = row[1:] if is_total else row
        cx = x
        for wd, val in zip(widths, cells):
            if not is_total:
                sax.add_patch(Rectangle((cx, y - row_h), wd, row_h, fill=False, ec="#777777", lw=0.3))
            sax.text(cx + 0.04, y - row_h / 2, str(val), fontsize=fs * 0.9, va="center")
            cx += wd
        y -= row_h
    return y


def overview_sheet(sch, out_dir: Path):
    p = sch.program
    fig, sax = new_sheet("L+L Overview", "G001")
    ax = fig.add_axes([0.03, 0.17, 0.62, 0.79])
    iso_building(ax, sch)
    ax.set_aspect("equal")
    ax.axis("off")
    sax.text(0.9, 1.6, "1  3D VIEW - AXON (massing, windows and storefront indicative)", fontsize=7)
    x = 12.3
    rows = [(lv.name.upper(), f"{lv.gross_sf:,.0f} SF") for lv in sch.levels]
    rows.append(("__total__", "", f"{sum(lv.gross_sf for lv in sch.levels):,.0f} SF"))
    y = schedule(sax, x, 11.5, "Gross building area (drawn)", ["Level", "Area"], rows, [1.4, 1.4])
    rc = [(c, n.upper()) for n, c, a, cat in p.rows if cat == "guest"]
    rc.append(("__total__", str(p.keys), ""))
    y = schedule(sax, x, y - 0.2, "RNGD_Hotel room count", ["Count", "Type"], rc, [0.8, 2.0])
    sa = []
    for cat in ["shared", "boh", "tenant", "guest", "business"]:
        rr = [(c, n.upper(), f"{a:,.0f} SF") for n, c, a, k in p.rows if k == cat]
        if not rr:
            continue
        sa.append(GROUP_TITLES.get(cat, cat.title()))
        sa += rr
        sa.append(("__total__", str(sum(int(r[0]) for r in rr)), "", f"{sum(float(r[2][:-3].replace(',', '')) for r in rr):,.0f} SF"))
    sa.append(("__total__", str(sum(c for _, c, _, _ in p.rows)), "", f"{p.net_sf:,.0f} SF"))
    others = sorted({k for _, _, _, k in p.rows} - {"shared", "boh", "tenant", "guest", "business"})
    for cat in others:
        sa.insert(-1, f"{cat.title()} ({sum(1 for r in p.rows if r[3] == cat)} rows)")
    schedule(sax, x, y - 0.2, "Space allocation (program)", ["Count", "Type", "Total area of type"], sa,
             [0.55, 2.35, 1.4], row_h=0.142, fs=5.6)
    return _save(fig, out_dir, "G001")


def code_sheet(sch, out_dir: Path, open_questions: list[str], assumptions: list[str]):
    fig, sax = new_sheet("Code & Program Summary", "G002")
    n = sch.program.net_to_gross(33, 8)
    rows = [(c["rule_id"], c["check"], c["required"], c["provided"], c["status"]) for c in sch.checks]
    y = 11.5
    widths = [1.1, 2.6, 2.9, 1.7, 0.8]
    sax.text(0.9, y, "CODE SCREEN - RNGD 2021 IBC RULE SET (planning level, not a code determination)",
             fontsize=9, fontweight="bold")
    y -= 0.3
    cx = 0.9
    for wd, hd in zip(widths, ["Rule", "Check", "Required", "Provided", "Status"]):
        sax.text(cx, y, hd.upper(), fontsize=6.5, fontweight="bold")
        cx += wd
    y -= 0.08
    sax.plot([0.9, 0.9 + sum(widths)], [y, y], color=INK, lw=0.5)
    for row in rows:
        y -= 0.24
        cx = 0.9
        for i, (wd, val) in enumerate(zip(widths, row)):
            col = {"PASS": "#1E7B3A", "FAIL": "#B22222"}.get(val, INK) if i == 4 else INK
            sax.text(cx, y, textwrap.shorten(str(val), int(wd * 17), placeholder="..."), fontsize=6.5, color=col,
                     fontweight="bold" if i == 4 else "normal")
            cx += wd
    notes = [c for c in sch.checks if c.get("note")]
    for c in notes:
        y -= 0.24
        sax.text(0.9, y, f"* {c['rule_id']}: {c['note']}", fontsize=6.2, color="#444444")
    x2 = 10.4
    y2 = 11.5
    sax.text(x2, y2, "PROGRAM SUMMARY", fontsize=9, fontweight="bold")
    items = [("Keys", f"{sch.program.keys}"), ("Net program area", f"{sch.program.net_sf:,.0f} SF"),
             ("Gross, source sheets", f"{sch.program.gross_sf:,.0f} SF (ratio {sch.program.gross_sf / sch.program.net_sf:.2f})"),
             ("Gross, 33% circ. + 8% walls", f"{n['required_gross_sf']:,.0f} SF"),
             ("Gross, drawn schematic", f"{sum(lv.gross_sf for lv in sch.levels):,.0f} SF"),
             ("Levels", f"{len(sch.levels)} (Level 1 + {len(sch.levels) - 1} guest floors)"),
             ("Height to roof", f"{sch.height_ft:.1f} ft"),
             ("Tower / podium length", f"{sch.levels[1].length_ft:.0f} ft / {sch.levels[0].length_ft:.0f} ft x {L.WING_W_FT:.0f} ft"),
             ("Program source", sch.program.source)]
    for k, v in items:
        y2 -= 0.24
        sax.text(x2, y2, k, fontsize=6.8, color="#555555")
        sax.text(x2 + 2.2, y2, textwrap.shorten(v, 70, placeholder="..."), fontsize=6.8)
    y2 -= 0.5
    sax.text(x2, y2, "ASSUMPTIONS", fontsize=9, fontweight="bold")
    for a in assumptions:
        for j, line in enumerate(textwrap.wrap(a, 105)):
            y2 -= 0.2
            sax.text(x2 + (0 if j == 0 else 0.12), y2, ("- " if j == 0 else "") + line, fontsize=6.5)
    y2 -= 0.5
    sax.text(x2, y2, "OPEN QUESTIONS FOR RNGD", fontsize=9, fontweight="bold", color="#B22222")
    for i, q in enumerate(open_questions, 1):
        for j, line in enumerate(textwrap.wrap(q, 105)):
            y2 -= 0.2
            sax.text(x2 + (0 if j == 0 else 0.18), y2, (f"{i}. " if j == 0 else "") + line, fontsize=6.5)
    return _save(fig, out_dir, "G002")


def site_sheet(sch, out_dir: Path, scale=30.0):
    fig, sax = new_sheet("Site Plan", "A001")
    env_w, env_d = sch.site["envelope_w"], sch.site["envelope_d"]
    g0 = sch.levels[0]
    xr = (-40.0, env_d + 60.0)
    yr = (-55.0, env_w + 30.0)
    w = (xr[1] - xr[0]) / scale
    h = (yr[1] - yr[0]) / scale
    ax, _, _ = view_axes(fig, (SHEET_W - w) / 2, 1.6 + (SHEET_H - 2.0 - h) / 2, xr, yr, scale)
    ax.add_patch(Rectangle((-15, -15), env_d + 30, env_w + 30, fc="#F1F5EC", ec=INK, lw=0.8))
    ax.add_patch(Rectangle((0, 0), env_d, env_w, fill=False, ec="#C0392B", lw=1.0, ls="--"))
    ax.text(env_d / 2, env_w + 2, f"BUILDABLE ENVELOPE {env_d:.0f}' x {env_w:.0f}' (synthetic lot - setbacks assumed)",
            fontsize=6.5, ha="center", va="bottom", color="#C0392B")
    ox = max(0.0, (env_d - g0.length_ft) / 2)
    ax.add_patch(Rectangle((ox, 0), g0.length_ft, L.WING_W_FT, fc="#D9D9D9", ec=INK, lw=1.4, hatch="////"))
    ax.add_patch(Rectangle((ox, 0), sch.levels[1].length_ft, L.WING_W_FT, fill=False, ec=INK, lw=0.6, ls="--"))
    ax.text(ox + g0.length_ft / 2, L.WING_W_FT / 2, f"HOTEL  {len(sch.levels)} LEVELS\nFOOTPRINT {g0.gross_sf:,.0f} SF",
            fontsize=7, ha="center", va="center", fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.add_patch(Rectangle((ox, L.WING_W_FT + 12), g0.length_ft, env_w - L.WING_W_FT - 12, fc="#E8E8E8",
                           ec="#888888", lw=0.5, ls=":"))
    ax.text(ox + g0.length_ft / 2, (L.WING_W_FT + env_w) / 2 + 6,
            "PARKING / SERVICE YARD / LANDSCAPE\n(not modelled in this phase)", fontsize=6.5, ha="center",
            va="center", color="#666666")
    ax.add_patch(Rectangle((-40, -55), env_d + 100, 35, fc="#DADADA", ec="none"))
    ax.text(env_d / 2, -37, "STREET (assumed frontage - lobby and retail face this side)", fontsize=7, ha="center",
            va="center")
    dim_h(ax, 0, env_d, -10, fs=5.5)
    dim_v(ax, 0, env_w, -8, fs=5.5)
    nx, ny = env_d + 35, env_w - 10
    ax.add_patch(Polygon([(nx, ny + 10), (nx - 4, ny - 2), (nx, ny + 1), (nx + 4, ny - 2)], closed=True, fc=INK))
    ax.text(nx, ny + 13, "N", fontsize=9, ha="center", fontweight="bold")
    view_title(ax, -35, -48, 1, "Site plan", f"1\" = {scale:.0f}'-0\"", size=1.4)
    return _save(fig, out_dir, "A001")


def unit_3d_sheet(sch, out_dir: Path):
    types = sorted(unit_types(sch.program), key=lambda t: -t[1])
    pick = []
    for want in ("king", "qq", "suite", "ada"):
        for t in types:
            if want in t[0].lower() and t not in pick and ("ada" in t[0].lower()) == (want == "ada"):
                pick.append(t)
                break
    for t in types:
        if len(pick) >= 4:
            break
        if t not in pick:
            pick.append(t)
    fig, sax = new_sheet("Unit 3D Views", "A501")
    cols = len(pick)
    cw = (SHEET_W - 1.0) / max(cols, 1)
    for i, (name, count, area) in enumerate(pick):
        wft = area / L.ROOM_DEPTH_FT
        for j, az in enumerate((-55.0, 125.0)):
            ax = fig.add_axes([(0.5 + i * cw) / SHEET_W, (6.6 - j * 4.75) / SHEET_H, (cw - 0.2) / SHEET_W, 4.4 / SHEET_H])
            iso_unit(ax, name, wft, az=az)
        sax.text(0.6 + i * cw, 11.55, f"{name.upper()} GROUP", fontsize=11)
        sax.text(0.6 + i * cw, 11.3, f"3D views  -  {area:,.0f} SF, {wft:.1f}' x {L.ROOM_DEPTH_FT:.0f}'  -  x{count}",
                 fontsize=7, color="#555555")
    sax.text(0.6, 1.6, "Walls, doors, windows and furniture are schematic blocks (painter's-algorithm axonometric, "
             "not an OpenGL render). Corridor wall at the entry door; exterior wall with the window.", fontsize=6.5,
             color="#555555")
    return _save(fig, out_dir, "A501")


def _save(fig, out_dir: Path, number: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{number}.pdf"
    fig.savefig(pdf)
    fig.savefig(out_dir / f"{number}.png", dpi=150)
    plt.close(fig)
    return pdf


def build_sheets(sch, out_dir: Path, open_questions: list[str], assumptions: list[str]) -> list[tuple[Path, str]]:
    """Render the whole set; returns (sheet PDF, title) in set order. PNGs sit beside the PDFs."""
    lv = sch.levels
    sheets = [(overview_sheet(sch, out_dir), "L+L overview: axonometric, gross area, room count, space allocation"),
              (code_sheet(sch, out_dir, open_questions, assumptions), "Code screen and program summary"),
              (site_sheet(sch, out_dir), "Site plan")]
    pairs = [lv[i:i + 2] for i in range(0, len(lv), 2)]
    for k, pair in enumerate(pairs):
        names = " & ".join(x.name.replace("Level ", "") for x in pair)
        sheets.append((plan_sheet(sch, pair, f"A10{k + 1}", f"L+L Plans - Levels {names}", out_dir),
                       f"Floor plans: Levels {names}"))
    sheets.append((elevation_sheet(sch, out_dir), "Elevations with indicative panel breakdown"))
    sheets += [(p, "Enlarged unit plans") for p in enlarged_sheets(sch, out_dir)]
    sheets.append((unit_3d_sheet(sch, out_dir), "Unit 3D views (interiors)"))
    return sheets
