import sys
from pathlib import Path

import ifcopenshell
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from design import SCENARIOS, Params, generate, module_counts
from ifc_writer import read_meshes, write_ifc


def test_module_counts_sum_to_units():
    assert sum(module_counts(Params(units=97)).values()) == 97


def test_compliant_scenario_passes_all_checks():
    lay = generate(SCENARIOS["Right-sized: 60 units, podium (compliant)"])
    assert [c for c in lay.checks if c["status"] != "PASS"] == []


def test_baseline_reports_parking_shortfall():
    lay = generate(Params())
    parking = next(c for c in lay.checks if c["check"] == "Parking stalls")
    assert parking["status"] == "FAIL"


def test_higher_floor_count_shrinks_footprint():
    a, b = generate(Params(units=60, floors=2)), generate(Params(units=60, floors=5))
    assert b.metrics["footprint_sf"] < a.metrics["footprint_sf"]


def test_zoning_preset_changes_limits():
    a = generate(Params(zoning="Suburban R-2 (sample)", units=60, floors=3))
    assert next(c for c in a.checks if c["check"] == "FAR")["required"] == "<= 1.0"


def test_accessible_units_present():
    lay = generate(Params(units=60, floors=4))
    assert sum(u["module"] == "ACCESSIBLE_1BR" for u in lay.units) >= 2


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_ifc_roundtrip(name, tmp_path):
    lay = generate(SCENARIOS[name])
    path = write_ifc(lay, tmp_path / "t.ifc")
    f = ifcopenshell.open(str(path))
    assert len(f.by_type("IfcBuildingStorey")) == len(lay.storeys)
    assert len(f.by_type("IfcWall")) >= 4 * len(lay.units)
    meshes = read_meshes(path)
    assert len(meshes) > len(lay.units)


def test_app_buttons_change_design():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=180).run()
    assert not at.exception
    before = at.metric[0].value
    btn = next(b for b in at.sidebar.button if b.label.startswith("Suburban"))
    btn.click().run()
    assert not at.exception
    assert at.metric[0].value != before or at.session_state["units"] == 60
