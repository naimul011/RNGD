"""Run: python -m pytest Testt/test_rvt_tools.py -v   (from RNGD/ or Testt/)"""
import hashlib
from pathlib import Path

import pytest

import rvt_tools as rt

HERE = Path(__file__).parent
FILES = rt.find_rvts(HERE / "extracted")
pytestmark = pytest.mark.skipif(not FILES, reason="run extraction first (see README)")


@pytest.fixture(scope="module", params=FILES, ids=lambda p: p.parent.name)
def rvt(request):
    return request.param


def test_streams_present(rvt):
    names = {s["stream"] for s in rt.list_streams(rvt)}
    assert {"BasicFileInfo", "RevitPreview4.0", "Partitions/12"} <= names


def test_version_is_2022(rvt):
    assert rt.basic_info(rvt)["revit_version"] == "2022"


def test_preview_is_valid_png(rvt, tmp_path):
    out = rt.extract_preview(rvt, tmp_path / "p.png")
    assert out is not None and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_element_database_decompresses(rvt):
    assert len(rt.decompress_stream(rvt, "Partitions/12")) > 10_000_000


def test_walls_and_levels_found(rvt):
    cats = rt.category_counts(rvt)
    assert cats["Wall"] > 0 and cats["Level"] > 0


def test_mep_file_has_ducts_and_pipes():
    mep = next(f for f in FILES if "MEP" in f.parent.name)
    cats = rt.category_counts(mep)
    assert cats["Duct"] > 0 and cats["Pipe"] > 0


def test_architecture_and_structural_are_identical_files():
    """Documents a finding: both zips contain the same racbasicsampleproject.rvt."""
    arch = next(f for f in FILES if "architecture" in f.parent.name)
    struct = next(f for f in FILES if "structural" in f.parent.name)
    h = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
    assert h(arch) == h(struct)
