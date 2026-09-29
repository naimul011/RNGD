"""Meshes -> Plotly 3D figure (also exportable as a standalone HTML file for Chrome)."""
import numpy as np
import plotly.graph_objects as go

FIG_HEIGHT = 800


def _merge(items: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates a group's per-element vert/triangle arrays with numpy,
    not Python list `+=` — that loop is the bottleneck on real-world files
    with hundreds of elements (a 885-mesh model took ~65s in pure Python
    list concatenation vs. well under a second with numpy)."""
    verts = [m["verts"] for m in items]
    tris = [m["tris"] for m in items]
    offsets = np.cumsum([0, *(len(v) for v in verts[:-1])])
    v = np.concatenate(verts, axis=0)
    t = np.concatenate([tri + off for tri, off in zip(tris, offsets)], axis=0)
    return v[:, 0], v[:, 1], v[:, 2], t[:, 0], t[:, 1], t[:, 2]


def _fit_to_bounds(fig: go.Figure, all_verts: list[np.ndarray]) -> go.Figure:
    """Explicitly ranges the 3D scene to the bounding box of what's actually
    shown, with a small margin. Plotly's own autorange looks at every trace
    added to the figure — a single oversized "ground" plane (the whole site,
    much bigger than the building on it) or one stray far-off vertex is
    enough to zoom the camera out until the actual building looks tiny.
    Setting the range explicitly guarantees a snug fit regardless."""
    if not all_verts:
        return fig
    v = np.concatenate(all_verts, axis=0)
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = span * 0.08
    lo, hi = lo - pad, hi + pad
    # With explicit axis ranges set below, Plotly's camera "eye" is in a
    # normalized coordinate system relative to the scene's own bounding box —
    # a fixed distance here frames consistently regardless of whether the
    # model is 10m or 1000m across, unlike a fixed eye vector with no
    # explicit range (which is what let one oversized trace zoom everything
    # else out to looking tiny).
    d = 1.35
    fig.update_layout(scene=dict(
        xaxis=dict(range=[lo[0], hi[0]], title="x (m)"),
        yaxis=dict(range=[lo[1], hi[1]], title="y (m)"),
        zaxis=dict(range=[lo[2], hi[2]], title="z (m)"),
        aspectmode="data",
        camera=dict(eye=dict(x=d, y=-d * 1.05, z=d * 0.65)),
    ))
    return fig


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


def build_figure(meshes, level=None, show_site=False) -> go.Figure:
    """`show_site` defaults to False: the ground/easement/driveway/parking
    slabs cover the whole lot, which is usually much larger than the
    building itself — including them by default is what made the building
    look tiny. Turn it on to see the full site context."""
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
    all_verts = []
    for key, items in groups.items():
        xs, ys, zs, I, J, K = _merge(items)
        all_verts.append(np.column_stack([xs, ys, zs]))
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=I, j=J, k=K, name=key, showlegend=True,
                                color=COLORS.get(key, "#AAAAAA"), flatshading=True, opacity=0.35 if key == "ground" else 1.0,
                                lighting=dict(ambient=0.6, diffuse=0.7)))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=FIG_HEIGHT, legend=dict(itemsizing="constant"))
    _fit_to_bounds(fig, all_verts)
    return fig


# ---- generic viewer for arbitrary real-world IFC files (ifc/explorer.py) --
# Unlike build_figure() above, these meshes have no "L1|STUDIO_A" naming
# convention to key off — group and color by IFC class instead.
CLASS_COLORS = {
    "IfcWall": "#8C8F92", "IfcCurtainWall": "#8FC1E3", "IfcDoor": "#8B5E3C",
    "IfcWindow": "#79C7C7", "IfcSlab": "#C9C0A8", "IfcRoof": "#7A4A3A",
    "IfcColumn": "#5B6670", "IfcBeam": "#6E7B85", "IfcStair": "#A0785A",
    "IfcStairFlight": "#A0785A", "IfcRailing": "#4C4C4C", "IfcCovering": "#DAD3C2",
    "IfcFurnishingElement": "#4C9F70", "IfcFlowTerminal": "#E0A030",
    "IfcMember": "#B9B2A0", "IfcPlate": "#CFEAF7", "IfcBuildingElementProxy": "#9C7FB5",
    "IfcSpace": "#F5F1E7",
}


def build_figure_generic(meshes: list[dict], classes: list[str] | None = None) -> go.Figure:
    groups: dict[str, list] = {}
    for m in meshes:
        if classes is not None and m["cls"] not in classes:
            continue
        groups.setdefault(m["cls"], []).append(m)

    fig = go.Figure()
    all_verts = []
    for cls, items in groups.items():
        xs, ys, zs, I, J, K = _merge(items)
        all_verts.append(np.column_stack([xs, ys, zs]))
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=I, j=J, k=K, name=f"{cls} ({len(items)})", showlegend=True,
                                color=CLASS_COLORS.get(cls, "#AAAAAA"), flatshading=True, opacity=0.4 if cls == "IfcSpace" else 1.0,
                                lighting=dict(ambient=0.6, diffuse=0.7)))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=FIG_HEIGHT, legend=dict(itemsizing="constant"))
    _fit_to_bounds(fig, all_verts)
    return fig
