"""Meshes -> Plotly 3D figure (also exportable as a standalone HTML file for Chrome)."""
import numpy as np
import plotly.graph_objects as go

COLORS = {
    "STUDIO_A": "#4C9F70", "ONEBR_A": "#3E7CB1", "TWOBR_A": "#E0A030", "ACCESSIBLE_1BR": "#8E5BB5",
    "CORE": "#7A2E4D", "AMENITY_LOBBY": "#D9534F", "AMENITY_FIT": "#E58E5C", "BOH_LOADING": "#8A8A8A",
    "SLAB": "#C9C5BA", "PODIUM_SLAB": "#9A9A9A", "ground": "#DDE8D0", "easement": "#FFBF00",
    "drive": "#555555", "stall": "#8C8C8C",
}


def _key(name):
    parts = name.split("|")
    return parts[1] if len(parts) > 1 else parts[0]


def _level(name):
    t = name.split("|")[0]
    return int(t[1:]) if t.startswith("L") and t[1:].isdigit() else None


def build_figure(meshes, level=None, show_site=True) -> go.Figure:
    groups: dict[str, list] = {}
    for m in meshes:
        key = _key(m["name"])
        lv = _level(m["name"])
        if key in ("ground", "easement", "drive", "stall") and not show_site:
            continue
        if level is not None and lv is not None and lv > level:
            continue
        if level is not None and key == "SLAB" and m["name"].startswith("ROOF"):
            continue
        if m["name"].startswith("ROOF") and level is not None:
            continue
        groups.setdefault(key, []).append(m)
    fig = go.Figure()
    for key, items in groups.items():
        xs, ys, zs, I, J, K = [], [], [], [], [], []
        off = 0
        for m in items:
            v, t = m["verts"], m["tris"]
            xs += list(v[:, 0]); ys += list(v[:, 1]); zs += list(v[:, 2])
            I += list(t[:, 0] + off); J += list(t[:, 1] + off); K += list(t[:, 2] + off)
            off += len(v)
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=I, j=J, k=K, name=key, showlegend=True,
                                color=COLORS.get(key, "#AAAAAA"), flatshading=True, opacity=0.35 if key == "ground" else 1.0,
                                lighting=dict(ambient=0.6, diffuse=0.7)))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                                 camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0))),
                      margin=dict(l=0, r=0, t=0, b=0), height=640, legend=dict(itemsizing="constant"))
    return fig
