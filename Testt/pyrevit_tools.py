"""Wrappers around the pyRevit CLI that work WITHOUT Revit installed.

`pyrevit revits fileinfo` reads model metadata straight from the file. Anything
that needs the Revit API (element collectors, geometry) lives in
pyrevit_extension/ and only runs inside Revit.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

CLI = Path.home() / "AppData" / "Roaming" / "pyRevit-Master" / "bin" / "pyrevit.exe"


def _run(*args: str, timeout: int = 120) -> str:
    if not CLI.exists():
        raise FileNotFoundError(f"pyRevit CLI not found at {CLI}; install pyRevit first")
    r = subprocess.run([str(CLI), *args], capture_output=True, text=True, timeout=timeout)
    return r.stdout


def cli_version() -> str:
    return _run("--version").splitlines()[0]


def installed_revits() -> list[str]:
    """Revit installs pyRevit can see. Empty list = pyRevit scripts cannot run."""
    out = _run("env")
    block = out.split("==> Installed Revits")[1].split("==>")[0]
    return [l.strip() for l in block.splitlines() if l.strip()]


def file_info(path) -> dict:
    """Metadata for a .rvt/.rfa/.rte/.rft via `pyrevit revits fileinfo`."""
    out = _run("revits", "fileinfo", str(path))
    info: dict = {"project_information": {}}
    section = None
    for line in out.splitlines():
        if line.startswith("Project Information"):
            section = "props"
        elif section == "props" and "=" in line:
            k, v = line.strip().split(" = ", 1)
            info["project_information"][k] = v
        elif ":" in line:
            k, v = line.split(":", 1)
            info[re.sub(r"\W+", "_", k.strip()).lower()] = v.strip()
    return info


def batch_file_info(root) -> dict[str, dict]:
    return {str(p): file_info(p) for p in sorted(Path(root).rglob("*.rvt"))}


def extension_path() -> Path:
    return Path(__file__).parent / "pyrevit_extension"


def register_extension() -> str:
    """Add the test extension folder to pyRevit's search path (loads when Revit starts)."""
    return _run("extensions", "paths", "add", str(extension_path()))


def load_revit_summary(json_path) -> dict:
    """Read a summary produced INSIDE Revit by the 'Summarize model' button."""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print(cli_version())
    print("Installed Revits:", installed_revits() or "none")
    root = Path(__file__).parent / "extracted"
    print(json.dumps(batch_file_info(root), indent=2))
