# -*- coding: utf-8 -*-
"""Export wall footprints (start/end points, height, type) of the active model
as JSON, ready for downstream layout/analysis code. Runs INSIDE Revit."""
__title__ = "Export\nwalls"
__doc__ = "Wall centerlines, heights and types as JSON."

import json
import os

from Autodesk.Revit.DB import BuiltInParameter, FilteredElementCollector, Wall
from pyrevit import revit, script

doc = revit.doc
walls = []
for w in FilteredElementCollector(doc).OfClass(Wall).WhereElementIsNotElementType():
    loc = w.Location
    curve = getattr(loc, "Curve", None)
    if curve is None:
        continue
    a, b = curve.GetEndPoint(0), curve.GetEndPoint(1)
    walls.append({
        "id": w.Id.IntegerValue,
        "type": w.WallType.Name,
        "start_ft": [round(a.X, 3), round(a.Y, 3)],
        "end_ft": [round(b.X, 3), round(b.Y, 3)],
        "length_ft": round(curve.Length, 3),
        "height_ft": round(w.get_Parameter(BuiltInParameter.WALL_USER_HEIGHT_PARAM).AsDouble(), 3),
    })
path = os.path.join(os.environ.get("TEMP", "."), "revit_walls_%s.json" % doc.Title)
with open(path, "w") as f:
    json.dump(walls, f, indent=2)
script.get_output().print_md("Exported **%d** walls to `%s`" % (len(walls), path))
