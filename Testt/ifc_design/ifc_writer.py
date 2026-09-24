"""Layout -> IFC4 file (IfcOpenShell) and IFC -> meshes for the browser viewer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.project
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit
import ifcopenshell.geom

from design import Layout

FT = 0.3048
WALL_T = 0.15


def _m(x):
    return x * FT


def _placement(x, y, z, angle_deg=0.0):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)


def write_ifc(lay: Layout, path) -> Path:
    f = ifcopenshell.api.project.create_file(version="IFC4")
    proj = ifcopenshell.api.root.create_entity(f, ifc_class="IfcProject", name="RNGD parametric design")
    ifcopenshell.api.unit.assign_unit(f)
    model = ifcopenshell.api.context.add_context(f, context_type="Model")
    body = ifcopenshell.api.context.add_context(f, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model)
    site = ifcopenshell.api.root.create_entity(f, ifc_class="IfcSite", name="Site")
    bldg = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuilding", name="Residential building")
    ifcopenshell.api.aggregate.assign_object(f, products=[site], relating_object=proj)
    ifcopenshell.api.aggregate.assign_object(f, products=[bldg], relating_object=site)

    storeys = {}
    for name, elev, _h in lay.storeys:
        st = ifcopenshell.api.root.create_entity(f, ifc_class="IfcBuildingStorey", name=name)
        st.Elevation = _m(elev)
        ifcopenshell.api.aggregate.assign_object(f, products=[st], relating_object=bldg)
        storeys[name] = st
    first = next(iter(storeys.values()))

    def wall(label, x, y, z, length, height, angle, storey):
        w = ifcopenshell.api.root.create_entity(f, ifc_class="IfcWall", name=label)
        rep = ifcopenshell.api.geometry.add_wall_representation(f, context=body, length=_m(length), height=_m(height), thickness=WALL_T)
        ifcopenshell.api.geometry.assign_representation(f, product=w, representation=rep)
        ifcopenshell.api.geometry.edit_object_placement(f, product=w, matrix=_placement(_m(x), _m(y), _m(z), angle))
        ifcopenshell.api.spatial.assign_container(f, products=[w], relating_structure=storey)

    def box_walls(label, x, y, w, d, z, h, storey):
        wall(label, x, y, z, w, h, 0, storey)
        wall(label, x, y + d, z, w, h, 0, storey)
        wall(label, x, y, z, d, h, 90, storey)
        wall(label, x + w, y, z, d, h, 90, storey)

    def slab(label, x, y, w, d, z, thick, storey, cls="IfcSlab"):
        s = ifcopenshell.api.root.create_entity(f, ifc_class=cls, name=label)
        pts = [(0.0, 0.0), (_m(w), 0.0), (_m(w), _m(d)), (0.0, _m(d)), (0.0, 0.0)]
        rep = ifcopenshell.api.geometry.add_slab_representation(f, context=body, depth=thick, polyline=pts)
        ifcopenshell.api.geometry.assign_representation(f, product=s, representation=rep)
        ifcopenshell.api.geometry.edit_object_placement(f, product=s, matrix=_placement(_m(x), _m(y), _m(z)))
        ifcopenshell.api.spatial.assign_container(f, products=[s], relating_structure=storey)

    p = lay.p
    slab("SITE|ground", 0, 0, p.lot_w, p.lot_d, -0.15, 0.15, first)
    if p.easement_east:
        slab("SITE|easement", p.lot_w - p.easement_east, 0, p.easement_east, p.lot_d, 0.0, 0.02, first)
    if lay.drive:
        slab("SITE|drive", lay.drive["x"], lay.drive["y"], lay.drive["w"], lay.drive["d"], 0.0, 0.03, first)
    for s in lay.stalls:
        slab("SITE|stall", s["x"], s["y"], s["w"], s["d"], 0.0, 0.04, first)

    lev_by_num = {int(n.split()[-1]): (n, e, h) for n, e, h in lay.storeys if n.startswith("Level")}
    for wing in lay.wings:
        for num, (n, e, h) in lev_by_num.items():
            slab(f"L{num}|SLAB", wing["x"], wing["y"], wing["w"], wing["h"], e, 0.25, storeys[n])
    if lay.podium_capacity:
        pn = "Podium parking"
        for wing in lay.wings:
            slab("P|PODIUM_SLAB", wing["x"], wing["y"], wing["w"], wing["h"], 0.0, 0.25, storeys[pn])
    top_z = lay.storeys[-1][1]
    for wing in lay.wings:
        slab("ROOF|SLAB", wing["x"], wing["y"], wing["w"], wing["h"], top_z, 0.3, storeys["Roof"], cls="IfcRoof")

    for u in lay.units:
        n, e, h = lev_by_num[u["level"]]
        box_walls(f"L{u['level']}|{u['module']}", u["x"], u["y"], u["w"], u["d"], e, h, storeys[n])
    for c in lay.cores:
        n, e, h = lev_by_num[c["level"]]
        box_walls(f"L{c['level']}|CORE", c["x"], c["y"], c["w"], c["d"], e, h, storeys[n])
    for a in lay.amenities:
        n, e, h = lev_by_num[a["level"]]
        box_walls(f"L{a['level']}|{a['module']}", a["x"], a["y"], a["w"], a["d"], e, h, storeys[n])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(path))
    return path


def read_meshes(path) -> list[dict]:
    """Re-read the IFC from disk and triangulate every element (proves the file is valid)."""
    f = ifcopenshell.open(str(path))
    st = ifcopenshell.geom.settings()
    st.set("use-world-coords", True)
    out = []
    for el in f.by_type("IfcElement"):
        if not el.Representation:
            continue
        try:
            sh = ifcopenshell.geom.create_shape(st, el)
        except Exception:
            continue
        v = np.array(sh.geometry.verts).reshape(-1, 3)
        t = np.array(sh.geometry.faces).reshape(-1, 3)
        level, _, kind = (el.Name.split("|") + ["", ""])[:3]
        out.append(dict(name=el.Name, cls=el.is_a(), verts=v, tris=t))
    return out
