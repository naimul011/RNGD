from pathlib import Path

import pytest

import pyrevit_tools as pt
import rvt_tools as rt

HERE = Path(__file__).parent
FILES = rt.find_rvts(HERE / "extracted")
needs_cli = pytest.mark.skipif(not pt.CLI.exists(), reason="pyRevit not installed")


@needs_cli
def test_cli_version():
    assert pt.cli_version().startswith("pyrevit v")


@needs_cli
def test_no_revit_installed_is_reported_as_list():
    assert isinstance(pt.installed_revits(), list)


@needs_cli
@pytest.mark.skipif(not FILES, reason="extract first")
def test_file_info_sample_house():
    arch = next(f for f in FILES if "architecture" in f.parent.name)
    info = pt.file_info(arch)
    assert info["project_information"]["Project Name"] == "Sample House"
    assert info["build"].startswith("2021")


@needs_cli
def test_extension_registered():
    assert (pt.extension_path() / "RNGDTest.extension").exists()


def test_extension_scripts_exist():
    scripts = list(pt.extension_path().rglob("script.py"))
    assert len(scripts) >= 2
