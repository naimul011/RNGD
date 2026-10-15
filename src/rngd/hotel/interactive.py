"""Interactive Plotly views of a hotel Schematic for the app: a hoverable floor plan
and a 3D massing model with every room as a coloured block."""
from __future__ import annotations

import plotly.graph_objects as go

from . import layout as L
from .drawings import FILL, LEGEND

KIND_NAME = dict(LEGEND) | {"corridor": "Corridor", "elevator": "Elevator", "shaft": "Shaft / mech"}
KIND_NAME["shared"] = "Shared"


def _hover(r, lvl_name: str) -> str:
    sf = f"{r.sf:,.0f} SF program" if r.sf else f"{r.w * r.d:,.0f} SF drawn"
    side = {"S": " - south side", "N": " - north side"}.get(r.side, "")
    return (f"<b>{r.label or 'continuation'}</b><br>{lvl_name}{side}<br>{sf}<br>"
            f"{r.w:.1f}' x {r.d:.1f}'<br>{KIND_NAME.get(r.kind, r.kind)}")


def plan_figure(lvl, highlight: str | None = None, height: int = 430) -> go.Figure:
    fig = go.Figure()
    seen = set()
    for r in lvl.rooms:
        dim = highlight and r.label != highlight
        fig.add_trace(go.Scatter(
            x=[r.x, r.x + r.w, r.x + r.w, r.x, r.x], y=[r.y, r.y, r.y + r.d, r.y + r.d, r.y],
            fill="toself", mode="lines", fillcolor=FILL.get(r.kind, "#FFFFFF"), opacity=0.25 if dim else 1.0,
            line=dict(color="#222222", width=0.8), hoveron="fills", hoverinfo="text", text=_hover(r, lvl.name),
            name=KIND_NAME.get(r.kind, r.kind), legendgroup=r.kind, showlegend=r.kind not in seen))
        seen.add(r.kind)
        if r.label and r.w >= 10 and r.kind not in ("elevator",) and len(r.label) * 1.6 < max(r.w, r.d):
            short = r.label.replace("(unprogrammed)", "").replace(" / ", "/")
            fig.add_annotation(x=r.x + r.w / 2, y=r.y + r.d / 2, text=short.upper() if r.w > 16 else short,
                               showarrow=False, font=dict(size=9, color="white" if r.kind in ("shared", "corridor") else "#222"),
                               textangle=0 if r.w >= 12 else -90)
    fig.add_shape(type="rect", x0=0, y0=0, x1=lvl.length_ft, y1=lvl.width_ft, line=dict(color="#111", width=3))
    fig.update_xaxes(range=[-5, lvl.length_ft + 5], title="ft (west -> east)", showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-5, lvl.width_ft + 5], scaleanchor="x", scaleratio=1, title="ft", showgrid=False,
                     zeroline=False, constrain="domain")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.08, x=0), hoverlabel=dict(bgcolor="white"))
    return fig


def _box(x, y, z, dx, dy, dz, off):
    xs = [x, x + dx, x + dx, x, x, x + dx, x + dx, x]
    ys = [y, y, y + dy, y + dy, y, y, y + dy, y + dy]
    zs = [z, z, z, z, z + dz, z + dz, z + dz, z + dz]
    tri = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
           (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return xs, ys, zs, [(a + off, b + off, c + off) for a, b, c in tri]


def massing_figure(sch, explode: float = 0.0, levels: list[str] | None = None, height: int = 560) -> go.Figure:
    """Every room as a block, grouped by kind; `explode` adds a gap (ft) between floors."""
    groups: dict[str, dict] = {}
    for i, lv in enumerate(sch.levels):
        if levels and lv.name not in levels:
            continue
        z0 = lv.elevation_ft + i * explode
        h = lv.height_ft * 0.92
        for r in lv.rooms:
            g = groups.setdefault(r.kind, {"x": [], "y": [], "z": [], "i": [], "j": [], "k": [], "t": []})
            xs, ys, zs, tris = _box(r.x, r.y, z0, r.w, r.d, h, len(g["x"]))
            g["x"] += xs
            g["y"] += ys
            g["z"] += zs
            for a, b, c in tris:
                g["i"].append(a)
                g["j"].append(b)
                g["k"].append(c)
            g["t"] += [_hover(r, lv.name)] * 8
    fig = go.Figure()
    for kind, g in groups.items():
        fig.add_trace(go.Mesh3d(x=g["x"], y=g["y"], z=g["z"], i=g["i"], j=g["j"], k=g["k"],
                                color=FILL.get(kind, "#FFFFFF"), opacity=1.0, flatshading=True,
                                name=KIND_NAME.get(kind, kind), showlegend=True, hoverinfo="text", text=g["t"],
                                lighting=dict(ambient=0.55, diffuse=0.7), contour=dict(show=False)))
    top = sch.height_ft + len(sch.levels) * explode
    Lmax = max(lv.length_ft for lv in sch.levels)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.02),
                      scene=dict(aspectmode="manual", aspectratio=dict(x=2.4, y=L.WING_W_FT / Lmax * 2.4, z=top / Lmax * 2.4),
                                 xaxis_title="ft", yaxis_title="ft", zaxis_title="ft",
                                 camera=dict(eye=dict(x=-1.5, y=-2.6, z=1.5))),
                      scene_dragmode="turntable")
    return fig
