"""Loads the REAL documents RNGD supplied under docs/ — a structured 2021 IBC
rule set for their five-story R-1 hotel prototype (CSV + companion workbook)
and the official ADA 2010 Standards PDF — and turns them into:

1. Searchable chunks merged into the shared knowledge base (vectorstore.py),
   clearly tagged `source_kind: "real"` so the UI/chat can distinguish them
   from the synthetic Rivermont zoning sample used elsewhere in this project.
2. A small set of machine-testable rules (code_check.py uses these) that can
   actually be evaluated against numbers the IFC Design Studio computes.

Unlike data/knowledge_base/*.md, nothing here is synthetic — it's RNGD's own
starter rule database (see docs/rngd_ibc_2021_r1_five_story_ai_ready_v0_2.xlsx
README sheet) and the DOJ's official accessibility standard.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

DOCS_DIR = config.ROOT / "docs"
IBC_CSV = DOCS_DIR / "RNGD_IBC_2021_R1_Five_Story_AI_Rules_v0_2.csv"
IBC_XLSX = DOCS_DIR / "rngd_ibc_2021_r1_five_story_ai_ready_v0_2.xlsx"
ADA_PDF = DOCS_DIR / "ada_2010_standards_for_accessible_design.pdf"


def available() -> dict[str, bool]:
    return {"ibc_csv": IBC_CSV.exists(), "ibc_xlsx": IBC_XLSX.exists(), "ada_pdf": ADA_PDF.exists()}


@dataclass
class RealChunk:
    text: str
    source: str
    heading: str


def _ibc_rule_chunks() -> list[RealChunk]:
    if not IBC_CSV.exists():
        return []
    df = pd.read_csv(IBC_CSV)
    chunks = []
    for _, r in df.iterrows():
        text = (
            f"IBC rule {r['rule_id']} ({r['category']}): {r['rule_name']}\n"
            f"Source: {r['base_code']} {r['base_section']} | Occupancy: {r['occupancy']} | "
            f"Construction type: {r.get('construction_type', '')} | Sprinkler: {r.get('sprinkler_condition', '')}\n"
            f"Rule: {r['rule_description']}\n"
            f"Applies when: {r['applies_when']}\n"
            f"Test: {r['operator']} {r['value']} {r['units']} ({r['hard_soft']}, severity={r.get('severity', '')})\n"
            f"Verification: {r['verification_status']} | Status: {r.get('rule_status', '')} | "
            f"Professional review required: {r.get('professional_review_required', '')}"
        )
        chunks.append(RealChunk(text=text, source="RNGD_IBC_2021_R1_Five_Story_AI_Rules_v0_2.csv", heading=f"{r['rule_id']} {r['rule_name']}"))
    return chunks


def _ibc_workbook_chunks() -> list[RealChunk]:
    if not IBC_XLSX.exists():
        return []
    xl = pd.ExcelFile(IBC_XLSX)
    chunks = []
    for sheet in ("README", "Project_Config", "Reference_Projects", "Hotel_Space_Mapping", "Occupant_Load_Factors"):
        if sheet not in xl.sheet_names:
            continue
        df = xl.parse(sheet).fillna("")
        text = f"{sheet} (from RNGD's IBC workbook):\n" + df.to_string(index=False)
        chunks.append(RealChunk(text=text[:3000], source="rngd_ibc_2021_r1_five_story_ai_ready_v0_2.xlsx", heading=sheet))
    return chunks


def _ada_pdf_chunks(max_pages: int = 279, chars_per_chunk: int = 1200) -> list[RealChunk]:
    if not ADA_PDF.exists():
        return []
    import pymupdf

    chunks = []
    doc = pymupdf.open(str(ADA_PDF))
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text = page.get_text().strip()
        if len(text) < 80:  # skip near-blank pages (covers, dividers)
            continue
        for j in range(0, len(text), chars_per_chunk):
            chunks.append(RealChunk(text=text[j : j + chars_per_chunk], source="ada_2010_standards_for_accessible_design.pdf", heading=f"page {i + 1}"))
    doc.close()
    return chunks


def load_all_chunks() -> list[RealChunk]:
    return _ibc_rule_chunks() + _ibc_workbook_chunks() + _ada_pdf_chunks()


def load_rules_df() -> pd.DataFrame | None:
    return pd.read_csv(IBC_CSV) if IBC_CSV.exists() else None
