"""Test helpers for reading Revit (.rvt) files WITHOUT Revit installed.

An .rvt is an OLE2 compound file. What we can read without Autodesk software:
  - BasicFileInfo / ProjectInformation : Revit version, build, original path, GUIDs
  - RevitPreview4.0                    : embedded PNG thumbnail of the saved view
  - Partitions/*                       : the element database, stored as many
                                         raw-deflate ("gzip-header") chunks
  - Global/*, Formats/Latest           : gzip streams with schema/history data

Full 3D geometry is NOT recoverable this way (proprietary binary object graph);
for that use Revit itself, Autodesk Platform Services (Model Derivative API),
or pyRevit/Dynamo. Everything here is best-effort and for testing.
"""
from __future__ import annotations

import collections
import re
import zlib
from pathlib import Path

import olefile

ASCII_RE = re.compile(rb"[\x20-\x7e]{6,}")
UTF16_RE = re.compile(rb"(?:[\x20-\x7e]\x00){4,}")
GZIP_MAGIC = b"\x1f\x8b\x08\x00"

CATEGORY_KEYWORDS = [
    "Wall", "Door", "Window", "Floor", "Roof", "Ceiling", "Stair", "Railing",
    "Column", "Beam", "Room", "Level", "Grid", "Duct", "Pipe", "Conduit",
    "Cable Tray", "Lighting", "Furniture", "Plumbing", "Mechanical", "Electrical",
    "Sprinkler", "Casework", "Curtain", "Site", "Topography", "Foundation",
]


def _clean(b: bytes) -> str:
    return re.sub(r"[^\x20-\x7e]+", " ", b.decode("utf-16le", "ignore")).strip()


def open_rvt(path) -> olefile.OleFileIO:
    return olefile.OleFileIO(str(path))


def list_streams(path) -> list[dict]:
    with open_rvt(path) as o:
        return [{"stream": "/".join(e), "bytes": o.get_size("/".join(e))} for e in o.listdir()]


def basic_info(path) -> dict:
    """Revit version/build/original path/GUIDs from BasicFileInfo."""
    with open_rvt(path) as o:
        text = _clean(o.openstream("BasicFileInfo").read())
        proj = _clean(o.openstream("ProjectInformation").read())
    version = re.search(r"\b(20\d\d)\b", text)
    build = re.search(r"\b(\d{8}_\d{4}\(x\d+\))", text)
    orig = re.search(r"([A-Z]:\\[^|]+?\.rvt)", text)
    return {
        "revit_version": version.group(1) if version else None,
        "build": build.group(1) if build else None,
        "original_path": orig.group(1) if orig else None,
        "guids": re.findall(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", text),
        "project_information_raw": proj,
        "file_size_mb": round(Path(path).stat().st_size / 1e6, 2),
    }


def extract_preview(path, out_png=None) -> Path | None:
    """Pull the embedded PNG thumbnail. Returns the output path or None."""
    with open_rvt(path) as o:
        raw = o.openstream("RevitPreview4.0").read()
    start = raw.find(b"\x89PNG\r\n\x1a\n")
    if start < 0:
        return None
    end = raw.rfind(b"IEND")
    png = raw[start : end + 8] if end > 0 else raw[start:]
    out = Path(out_png) if out_png else Path(path).with_suffix(".preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    return out


def dump_streams(path, out_dir) -> list[Path]:
    """Write every raw OLE stream to disk (names have '/' -> '__')."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with open_rvt(path) as o:
        for e in o.listdir():
            name = "/".join(e)
            dest = out_dir / (name.replace("/", "__") + ".bin")
            dest.write_bytes(o.openstream(name).read())
            written.append(dest)
    return written


def _inflate_chunks(blob: bytes) -> bytes:
    """Concatenate every valid raw-deflate chunk that follows a gzip header.
    Some magic-number hits are false positives; those are skipped."""
    parts = []
    for m in re.finditer(re.escape(GZIP_MAGIC), blob):
        try:
            parts.append(zlib.decompressobj(-15).decompress(blob[m.start() + 10 :]))
        except zlib.error:
            continue
    return b"".join(parts)


def decompress_stream(path, stream: str = "Partitions/12") -> bytes:
    with open_rvt(path) as o:
        blob = o.openstream(stream).read()
    if stream.startswith("Partitions"):
        return _inflate_chunks(blob)
    # Global/*, Formats/Latest: 8-byte header, then one gzip member
    try:
        return zlib.decompressobj(-15).decompress(blob[8 + 10 :])
    except zlib.error:
        return b""


def harvest_strings(path, stream: str = "Partitions/12", min_count: int = 1) -> collections.Counter:
    """UTF-16 (and ASCII) strings from the decompressed element database."""
    data = decompress_stream(path, stream)
    c: collections.Counter = collections.Counter()
    for s in UTF16_RE.findall(data):
        c[s.decode("utf-16le", "ignore")] += 1
    for s in ASCII_RE.findall(data):
        c[s.decode("ascii", "ignore")] += 1
    return collections.Counter({k: v for k, v in c.items() if v >= min_count})


def search_strings(path, term: str, limit: int = 40) -> list[tuple[str, int]]:
    hits = [(s, n) for s, n in harvest_strings(path).items()
            if term.lower() in s.lower() and not s.startswith("autodesk.")]
    return sorted(hits, key=lambda x: -x[1])[:limit]


def category_counts(path) -> dict[str, int]:
    """Rough element-category signal: how often each Revit category keyword
    appears in the element database strings. A heuristic, not an exact count."""
    strings = harvest_strings(path)
    out = {}
    for kw in CATEGORY_KEYWORDS:
        out[kw] = sum(n for s, n in strings.items() if kw.lower() in s.lower() and not s.startswith("autodesk."))
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def summarize(path) -> dict:
    info = basic_info(path)
    return {
        "file": Path(path).name,
        **info,
        "streams": list_streams(path),
        "category_signal": category_counts(path),
    }


def find_rvts(root) -> list[Path]:
    return sorted(Path(root).rglob("*.rvt"))
