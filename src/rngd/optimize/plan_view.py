"""Interactive drawings of an optimized hotel layout (Plotly): a furnished floor plan with
doors, windows, stairs and elevators, a 3D stacked massing, and small comparison thumbnails.

Coordinates: 1 grid cell = 1 ft; x = column (west -> east), y = row (south -> north), with the
plate margin removed so the plate starts at (0, 0).
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from scipy import ndimage

from .engine import Design
from .typology import MARGIN, Plate

TYPE_COLORS = ["#C9B6D9", "#A9C6E8", "#B5D9B0", "#F2C9A0", "#E6A5A5", "#D9D49A", "#9FD5CB", "#C7C7E8", "#F0B6D0",
               "#BFD8E0"]
CAT_COLORS = {"service": "#C9C9C9", "public": "#8FCFC3"}
CAT_NAME = {"guest": "Guest room", "suite": "Suite", "ada": "ADA room", "service": "Service / BOH", "public": "Public"}
CORRIDOR = "#5B87AD"
CORE = "#7A2E4D"
PLATE = "#F6F3EC"
WINDOW = "#1F77B4"


def type_colors(design: Design) -> dict:
    keys = sorted({p.type_name for p in design.placements if p.category in ("guest", "suite", "ada")})
    out = {t: TYPE_COLORS[i % len(TYPE_COLORS)] for i, t in enumerate(keys)}
    for p in design.placements:
        if p.type_name not in out:
            out[p.type_name] = CAT_COLORS.get(p.category, "#DDDDDD")
    return out


def _frame(p):
    """Map room-local (u along the corridor, v from the corridor wall to the facade) to plan x, y."""
    x0, y0 = p.c0 - MARGIN, p.r0 - MARGIN
    w, h = p.w, p.h
    if p.facing == "N":
        return (lambda u, v: (x0 + u, y0 + v)), w, h
    if p.facing == "S":
        return (lambda u, v: (x0 + w - u, y0 + h - v)), w, h
    if p.facing == "E":
        return (lambda u, v: (x0 + v, y0 + h - u)), h, w
    return (lambda u, v: (x0 + w - v, y0 + u)), h, w


def _poly(f, pts):
    xy = [f(u, v) for u, v in pts]
    xs, ys = zip(*xy)
    return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in xy) + " Z", xs, ys


def _rect_path(f, u0, v0, u1, v1):
    return _poly(f, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])[0]


def _furniture(p, f, F, D) -> list[dict]:
    """Bathroom, bed(s), desk, door swing and window for one guest room (BaaP-like layout)."""
    line = dict(color="#555", width=0.8)
    shapes = []
    bath_w, bath_d = min(6.0, F * 0.45), min(8.5, D * 0.33)
    shapes.append(dict(type="path", path=_rect_path(f, 0.3, 0.3, bath_w, bath_d), fillcolor="#FFFFFF", line=line))
    # tub/shower and WC inside the bath
    shapes.append(dict(type="path", path=_rect_path(f, 0.6, bath_d - 2.8, bath_w - 0.3, bath_d - 0.3),
                       fillcolor="#E8F1F8", line=dict(color="#7A9CB8", width=0.6)))
    # door swing from the corridor (quarter circle beside the bath)
    du = bath_w + 0.4
    r = min(3.0, F - du - 0.3)
    if r > 1.5:
        pts = [(du, 0)] + [(du + r * math.cos(a), r * math.sin(a)) for a in np.linspace(0, math.pi / 2, 8)] + [(du, 0)]
        shapes.append(dict(type="path", path=_poly(f, pts)[0], fillcolor="rgba(0,0,0,0)",
                           line=dict(color="#333", width=0.7, dash="dot")))
    # beds against the side wall, in the room's back half
    king = "q" not in p.type_name.lower()
    bed_len = 7.0
    v_mid = bath_d + (D - bath_d) * 0.42
    if p.category == "suite" or F >= 22:
        shapes.append(dict(type="path", path=_rect_path(f, F * 0.55, v_mid, F * 0.55 + 6.5, v_mid + bed_len),
                           fillcolor="#FFFFFF", line=line))
        shapes.append(dict(type="path", path=_rect_path(f, F * 0.12, D - 9, F * 0.12 + 7, D - 6),   # sofa
                           fillcolor="#E9E2D5", line=line))
    elif king:
        u0 = max(0.6, (F - 6.5) / 2)
        shapes.append(dict(type="path", path=_rect_path(f, u0, v_mid, u0 + min(6.5, F - 1.2), v_mid + bed_len),
                           fillcolor="#FFFFFF", line=line))
    else:
        bw = min(4.6, (F - 2.0) / 2)
        for u0 in (0.5, F - 0.5 - bw):
            shapes.append(dict(type="path", path=_rect_path(f, u0, v_mid, u0 + bw, v_mid + 6.7),
                               fillcolor="#FFFFFF", line=line))
    # desk on the opposite wall near the window
    shapes.append(dict(type="path", path=_rect_path(f, F - 2.2, D - 7.5, F - 0.3, D - 3.0), fillcolor="#E9E2D5",
                       line=line))
    return shapes


def _stairs(x, y, w, h) -> list[dict]:
    out = [dict(type="rect", x0=x, y0=y, x1=x + w, y1=y + h, fillcolor="#EDE3E8", line=dict(color=CORE, width=1.2))]
    if w >= h:
        for k in np.arange(x + 1, x + w - 0.5, 1.0):
            out.append(dict(type="line", x0=k, x1=k, y0=y + 0.5, y1=y + h - 0.5, line=dict(color=CORE, width=0.5)))
        out.append(dict(type="line", x0=x + 0.5, x1=x + w - 0.5, y0=y + h / 2, y1=y + h / 2, line=dict(color=CORE, width=1)))
    else:
        for k in np.arange(y + 1, y + h - 0.5, 1.0):
            out.append(dict(type="line", y0=k, y1=k, x0=x + 0.5, x1=x + w - 0.5, line=dict(color=CORE, width=0.5)))
        out.append(dict(type="line", y0=y + 0.5, y1=y + h - 0.5, x0=x + w / 2, x1=x + w / 2, line=dict(color=CORE, width=1)))
    return out


def _core_shapes(c) -> list[dict]:
    """Main core: stair + elevators + elevator lobby; second core: a stair."""
    x, y, w, h, kind = c
    out = [dict(type="rect", x0=x, y0=y, x1=x + w, y1=y + h, fillcolor="#F3E9EE", line=dict(color=CORE, width=2))]
    horizontal = w >= h
    L = w if horizontal else h
    if kind != "main":
        return out + _stairs(x + 0.5, y + 0.5, w - 1, h - 1)
    stair_len = 13
    if horizontal:
        out += _stairs(x + 0.5, y + 0.5, stair_len, h - 1)
        n = max(2, int((L - stair_len - 10) // 8) * 2)
        ex = x + stair_len + 1
        for i in range(n // 2):
            for row in (0, 1):
                ey = y + 1 if row == 0 else y + h - 8
                out.append(dict(type="rect", x0=ex + i * 8, y0=ey, x1=ex + i * 8 + 7.5, y1=ey + 7,
                                fillcolor="#FFFFFF", line=dict(color=CORE, width=1)))
                out.append(dict(type="line", x0=ex + i * 8, y0=ey, x1=ex + i * 8 + 7.5, y1=ey + 7,
                                line=dict(color=CORE, width=0.6)))
                out.append(dict(type="line", x0=ex + i * 8, y0=ey + 7, x1=ex + i * 8 + 7.5, y1=ey,
                                line=dict(color=CORE, width=0.6)))
    else:
        out += _stairs(x + 0.5, y + 0.5, w - 1, stair_len)
        n = max(2, int((L - stair_len - 10) // 8) * 2)
        ey = y + stair_len + 1
        for i in range(n // 2):
            for col in (0, 1):
                ex = x + 1 if col == 0 else x + w - 8
                out.append(dict(type="rect", x0=ex, y0=ey + i * 8, x1=ex + 7, y1=ey + i * 8 + 7.5,
                                fillcolor="#FFFFFF", line=dict(color=CORE, width=1)))
                out.append(dict(type="line", x0=ex, y0=ey + i * 8, x1=ex + 7, y1=ey + i * 8 + 7.5,
                                line=dict(color=CORE, width=0.6)))
    return out


def _mask_rects(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover a boolean mask with rectangles (row runs merged vertically): (x, y, w, h)."""
    rects, open_ = [], {}
    R, C = mask.shape
    for r in range(R + 1):
        runs = set()
        if r < R:
            row = mask[r]
            c = 0
            while c < C:
                if row[c]:
                    s = c
                    while c < C and row[c]:
                        c += 1
                    runs.add((s, c))
                else:
                    c += 1
        nxt = {}
        for run, r0 in open_.items():
            if run in runs:
                nxt[run] = r0
            else:
                rects.append((run[0], r0, run[1] - run[0], r - r0))
        for run in runs:
            nxt.setdefault(run, r)
        open_ = nxt
    return rects


def plan_figure(design: Design, plate: Plate, level: int, highlight: str | None = None, furniture: bool = True,
                height: int = 640) -> go.Figure:
    site = design.site
    colors = type_colors(design)
    W, H = plate.size
    fig = go.Figure()
    shapes = []
    for x, y, w, h in plate.rects:   # plate
        shapes.append(dict(type="rect", x0=x, y0=y, x1=x + w, y1=y + h, fillcolor=PLATE, line=dict(width=0), layer="below"))
    corr = design.corridor[MARGIN:-MARGIN, MARGIN:-MARGIN] if MARGIN else design.corridor
    for x, y, w, h in _mask_rects(corr):
        shapes.append(dict(type="rect", x0=x, y0=y, x1=x + w, y1=y + h, fillcolor=CORRIDOR, opacity=0.55,
                           line=dict(width=0), layer="below"))
    for c in plate.cores:
        shapes += _core_shapes(c)
    seen = set()
    for p in design.placements:
        if p.level != level:
            continue
        f, F, D = _frame(p)
        dim = bool(highlight) and highlight not in (p.type_name, p.category)
        col = colors.get(p.type_name, "#DDD")
        _, xs, ys = _poly(f, [(0, 0), (F, 0), (F, D), (0, D)])
        hover = (f"<b>{p.name}</b><br>{p.type_name} · {CAT_NAME.get(p.category, p.category)}<br>"
                 f"{p.frontage_ft:.0f}' x {p.depth_ft:.0f}' = {p.area_sf:,.0f} SF<br>"
                 f"window: {'yes' if p.exterior else 'no'}{' · corner' if p.corner else ''}<br>"
                 f"door to nearest core: {p.door_dist_ft:.0f} ft")
        show = p.type_name not in seen
        seen.add(p.type_name)
        fig.add_trace(go.Scatter(x=list(xs) + [xs[0]], y=list(ys) + [ys[0]], fill="toself", mode="lines",
                                 fillcolor=col, opacity=0.25 if dim else 1.0, line=dict(color="#222", width=1.1),
                                 hoveron="fills", hoverinfo="text", text=hover, name=p.type_name,
                                 legendgroup=p.type_name, showlegend=show))
        if p.exterior:   # window on the facade wall
            a, b = f(F * 0.15, D), f(F * 0.85, D)
            shapes.append(dict(type="line", x0=a[0], y0=a[1], x1=b[0], y1=b[1], line=dict(color=WINDOW, width=4)))
        if furniture and p.category in ("guest", "suite", "ada") and F >= 9 and D >= 18 and not dim:
            shapes += _furniture(p, f, F, D)
        elif not dim:   # door mark for other rooms
            a = f(F / 2, 0)
            shapes.append(dict(type="circle", x0=a[0] - 0.8, y0=a[1] - 0.8, x1=a[0] + 0.8, y1=a[1] + 0.8,
                               fillcolor="#333", line=dict(width=0)))
        if F >= 11 or p.category not in ("guest", "suite", "ada"):
            cx, cy = f(F / 2, D * (0.82 if furniture and p.category in ("guest", "suite", "ada") else 0.5))
            label = p.type_name.replace(" / ", "/")
            if p.category in ("guest", "suite", "ada"):
                label = "".join(w[0] for w in label.split()) if len(label) > 8 and F < 20 else label
            fig.add_annotation(x=cx, y=cy, text=label, showarrow=False, font=dict(size=8, color="#222"),
                               textangle=-90 if (p.facing in ("N", "S") and F < 16 and len(label) > 4) else 0)
    for x, y, w, h, kind in plate.cores:
        fig.add_annotation(x=x + w / 2, y=y + h / 2, text="CORE" if kind == "main" else "STAIR", showarrow=False,
                           font=dict(size=9, color="white"), bgcolor=CORE, opacity=0.9)
    for (l, r0, c0, h, w, win) in design.extra.get("flex_bays", []):
        if l == level:
            shapes.append(dict(type="rect", x0=c0 - MARGIN, y0=r0 - MARGIN, x1=c0 - MARGIN + w, y1=r0 - MARGIN + h,
                               fillcolor="rgba(200,200,200,0.25)", line=dict(color="#AAA", width=0.6, dash="dot")))
    for x, y, w, h in plate.rects:   # exterior walls on top
        shapes.append(dict(type="rect", x0=x, y0=y, x1=x + w, y1=y + h, fillcolor="rgba(0,0,0,0)",
                           line=dict(color="#111", width=1.2)))
    # overall dimensions + north arrow
    fig.add_annotation(x=W / 2, y=-7, text=f"{W} ft", showarrow=False, font=dict(size=11))
    fig.add_annotation(x=-8, y=H / 2, text=f"{H} ft", showarrow=False, textangle=-90, font=dict(size=11))
    fig.add_annotation(x=W + 6, y=H - 4, ax=W + 6, ay=H - 16, xref="x", yref="y", axref="x", ayref="y", text="N",
                       showarrow=True, arrowhead=2, arrowwidth=2, font=dict(size=12))
    fig.update_layout(shapes=shapes, height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)), hoverlabel=dict(bgcolor="white"),
                      dragmode="pan")
    fig.update_xaxes(range=[-14, W + 14], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-12, H + 8], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False,
                     showticklabels=False)
    return fig


def _box(x, y, z, dx, dy, dz, off):
    xs = [x, x + dx, x + dx, x, x, x + dx, x + dx, x]
    ys = [y, y, y + dy, y + dy, y, y, y + dy, y + dy]
    zs = [z, z, z, z, z + dz, z + dz, z + dz, z + dz]
    tri = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
           (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return xs, ys, zs, [(a + off, b + off, c + off) for a, b, c in tri]


def massing_figure(design: Design, plate: Plate, explode: float = 0.0, height: int = 560) -> go.Figure:
    site = design.site
    colors = type_colors(design)
    groups: dict = {}

    def add(key, color, x, y, z, dx, dy, dz, text):
        g = groups.setdefault(key, {"c": color, "x": [], "y": [], "z": [], "i": [], "j": [], "k": [], "t": []})
        xs, ys, zs, tris = _box(x, y, z, dx, dy, dz, len(g["x"]))
        g["x"] += xs; g["y"] += ys; g["z"] += zs
        for a, b, c in tris:
            g["i"].append(a); g["j"].append(b); g["k"].append(c)
        g["t"] += [text] * 8

    elev = site.elevations_ft + [site.elevations_ft[-1] + (site.elevations_ft[-1] - site.elevations_ft[-2]
                                                         if len(site.elevations_ft) > 1 else 10)]
    for l in range(site.n_levels):
        z0 = elev[l] + l * explode
        h = (elev[l + 1] - elev[l]) * 0.9
        for p in design.placements:
            if p.level == l:
                add(p.type_name, colors.get(p.type_name, "#DDD"), p.c0 - MARGIN, p.r0 - MARGIN, z0, p.w, p.h, h,
                    f"{p.name}<br>{site.level_names[l]}<br>{p.area_sf:,.0f} SF")
        for x, y, w, hh, kind in plate.cores:
            add("Core / stairs", CORE, x, y, z0, w, hh, h, f"{'Elevator core' if kind == 'main' else 'Stair'}<br>{site.level_names[l]}")
    fig = go.Figure()
    for key, g in groups.items():
        fig.add_trace(go.Mesh3d(x=g["x"], y=g["y"], z=g["z"], i=g["i"], j=g["j"], k=g["k"], color=g["c"], opacity=1.0,
                                flatshading=True, name=key, showlegend=True, hoverinfo="text", text=g["t"],
                                lighting=dict(ambient=0.55, diffuse=0.7)))
    W, H = plate.size
    top = elev[-1] + site.n_levels * explode
    s = max(W, H)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.02),
                      scene=dict(aspectmode="manual", aspectratio=dict(x=W / s * 2, y=H / s * 2, z=top / s * 2),
                                 xaxis_title="ft", yaxis_title="ft", zaxis_title="ft",
                                 camera=dict(eye=dict(x=-1.4, y=-1.9, z=1.2))),
                      scene_dragmode="turntable")
    return fig


def thumbnail(design: Design, plate: Plate, level: int, title: str = ""):
    """Small matplotlib plan for the comparison gallery (PNG bytes)."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colors = type_colors(design)
    W, H = plate.size
    fig, ax = plt.subplots(figsize=(3.2, 3.2 * max(0.45, min(1.6, H / max(W, 1)))))
    for x, y, w, h in plate.rects:
        ax.add_patch(Rectangle((x, y), w, h, fc=PLATE, ec="none"))
    corr = design.corridor[MARGIN:-MARGIN, MARGIN:-MARGIN]
    for x, y, w, h in _mask_rects(corr):
        ax.add_patch(Rectangle((x, y), w, h, fc=CORRIDOR, ec="none", alpha=0.7))
    for p in design.placements:
        if p.level == level:
            ax.add_patch(Rectangle((p.c0 - MARGIN, p.r0 - MARGIN), p.w, p.h, fc=colors.get(p.type_name, "#DDD"),
                                   ec="#333", lw=0.3))
    for x, y, w, h, kind in plate.cores:
        ax.add_patch(Rectangle((x, y), w, h, fc=CORE, ec="none"))
    ax.set_xlim(-3, W + 3)
    ax.set_ylim(-3, H + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
