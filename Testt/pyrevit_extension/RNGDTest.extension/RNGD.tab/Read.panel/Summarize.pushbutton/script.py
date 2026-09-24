# -*- coding: utf-8 -*-
"""Summarize the open Revit model: element counts per category, levels,
rooms with areas, wall types. Writes JSON next to the model and prints it.
Runs INSIDE Revit via pyRevit (IronPython-safe syntax)."""
__title__ = "Summarize\nmodel"
__doc__ = "Category counts, levels, rooms and wall types of the active model."

import json
import os
from collections import Counter

from Autodesk.Revit.DB import (BuiltInCategory, BuiltInParameter,
                               FilteredElementCollector, WallType)
from pyrevit import forms, revit, script

doc = revit.doc
out = script.get_output()


def category_counts():
    c = Counter()
    for el in FilteredElementCollector(doc).WhereElementIsNotElementType():
        cat = el.Category
        if cat is not None:
            c[cat.Name] += 1
    return dict(c.most_common())


def levels():
    from Autodesk.Revit.DB import Level
    return sorted(
        [{"name": l.Name, "elevation_ft": round(l.Elevation, 2)}
         for l in FilteredElementCollector(doc).OfClass(Level)],
        key=lambda d: d["elevation_ft"])


def rooms():
    res = []
    for r in FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_Rooms):
        if r.Area > 0:
            res.append({"name": r.get_Parameter(BuiltInParameter.ROOM_NAME).AsString(),
                        "number": r.Number, "area_sf": round(r.Area, 1),
                        "level": r.Level.Name if r.Level else None})
    return res


def wall_types():
    return sorted(set(t.get_Parameter(BuiltInParameter.SYMBOL_FAMILY_AND_TYPE_NAMES_PARAM).AsString()
                      or t.Name for t in FilteredElementCollector(doc).OfClass(WallType)))


summary = {
    "title": doc.Title,
    "categories": category_counts(),
    "levels": levels(),
    "rooms": rooms(),
    "wall_types": wall_types(),
}
path = os.path.join(os.environ.get("TEMP", "."), "revit_summary_%s.json" % doc.Title)
with open(path, "w") as f:
    json.dump(summary, f, indent=2)
out.print_md("**Model:** %s  \n**Categories:** %d  \n**Rooms:** %d  \nSaved: `%s`" % (
    doc.Title, len(summary["categories"]), len(summary["rooms"]), path))
