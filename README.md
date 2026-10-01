# RNGD Concept-to-Schematic AI Studio (Semester Proof-of-Concept)

A multi-agent AI prototype for RNGD's "concept-to-schematic" workflow, now unified into one app
with three modes and a wide, persistent chat assistant:

- **Multi-Agent Pipeline** — turns an unstructured customer request + site conditions into a
  Building-as-a-Product (BaaP) schematic site plan and floor plate, with automated conflict
  detection and one final AI decision (Groq agents + a physics-style site simulation + classical
  computer-vision QA + a single Anthropic Claude arbiter call).
- **IFC Design Studio** — a faster, parametric generator: requirements + a zoning preset + the
  BaaP catalog -> a rule-based building layout -> a real, standard **IFC4** file (via
  IfcOpenShell) -> an interactive 3D model in the browser, scored against ten sample-zoning checks
  **and** cross-checked against a curated subset of RNGD's own **real** 2021 IBC rules (see below).
- **Open IFC File** — opens an arbitrary real `.ifc` file (not one this app generated) for 3D
  viewing, browsing and Q&A; auto-recenters and drops implausible far-off outlier elements so the
  actual building fills the view.
- **Chat assistant** — a wide persistent panel with a live, color-coded summary card (requirement
  match %, setbacks, conflicts) for whichever mode is active, one-click "ask about this design"
  question buttons, and proactive suggestions after a run finishes.

## Real documents from RNGD (`docs/`)

Alongside the synthetic sample data, `docs/` holds real material RNGD supplied: a structured 2021
IBC rule database for their five-story R-1 hotel prototype (42 machine-testable rules with
operator/value/units, occupant-load factors, hotel space mapping, reference projects — see
`docs/rngd_ibc_2021_r1_five_story_ai_ready_v0_2.xlsx`'s README sheet) and the official DOJ **ADA
2010 Standards for Accessible Design** PDF. These are vectorized into the same knowledge base as
the synthetic docs, tagged `source_kind: "real"` so the UI and chat are always clear about which
is which. `src/rngd/ifc/code_check.py` cross-checks a generated IFC Design Studio layout against
the ~8 of those real rules this prototype can actually compute from the layout (height, stories,
exit travel distance, dead-end corridor length, exits per story); the rest need space-level
occupancy modeling not yet built. The Design Studio's "Real IBC check" tab shows the result, and
one real finding already surfaced: the single-core double-loaded corridor layout passes its own
sample "exit travel ≤ 200 ft" check but **fails** the real IBC dead-end-corridor limit (≤ 50 ft) —
a genuine design implication worth knowing about.

See `REPORT.md` and `reports/RNGD_manual.pdf` for the full write-ups (architecture, design
rationale, how this maps to RNGD's own kickoff questions, results, and limitations), and
`reports/ifc_design/ifc_design_report.pdf` for a deep dive on the IFC Design Studio's algorithm.

## Quick start

```bash
pip install -r requirements.txt
python src/make_sample_data.py     # only needed once, or if you edit the sample site
streamlit run src/app.py           # the unified app: pipeline + IFC studio + chat
python src/run_pipeline.py         # CLI alternative: runs the pipeline, prints trace + decision
python src/generate_ifc.py         # CLI alternative: writes every IFC scenario to outputs/ifc_design/
python -m pytest tests/            # full test suite (pipeline, IFC engine, app)
```

Requires a `.env` in the project root with `GROQ_API_KEY` and `ANTHROPIC_API_KEY` (already
present in this project; **not** committed to git).

## Layout

- `data/knowledge_base/` — sample zoning code, BaaP catalog, stacking rules, design guidelines
  (markdown, RAG-indexed at startup).
- `data/baap_catalog.json` — machine-readable twin of the BaaP catalog, shared by both modes.
- `data/sample_projects/project_001/` — one synthetic sample project: a raw customer RFP letter,
  `site_conditions.json`, and a synthetic site sketch PNG + georeference sidecar.
- `data/ifc_samples/` — real `.ifc` files to open in "Open IFC File" mode (git-ignored; see its README).
- `docs/` — real RNGD-supplied reference documents (IBC rule database + ADA standards PDF), vectorized
  alongside the synthetic knowledge base.
- `src/rngd/` — the shared package:
  - `agents/`, `orchestrator.py`, `vectorstore.py`, `simulation/`, `vision/` — the multi-agent pipeline.
  - `ifc/` — the parametric IFC Design Studio engine (`design.py`, `ifc_writer.py`, `viewer3d.py`),
    the real-file explorer (`explorer.py`), and the real-IBC cross-check (`code_check.py`).
  - `chat.py` — the chat assistant (multi-turn, grounded in all three modes' state + the knowledge base).
  - `real_docs.py` — loads/chunks the real RNGD documents under `docs/`.
  - `summary_card.py` — builds the chat panel's color-coded summary card (pure Python, no Streamlit).
- `src/app.py` — the unified Streamlit app (both modes + chat).
- `src/generate_ifc.py`, `src/build_ifc_report.py` — IFC studio CLI + its LaTeX/PDF report builder.
- `tests/` — pytest suite: pipeline agents, IFC engine (IFC round-trip), and app-level smoke tests
  (via Streamlit's `AppTest`, including a full pipeline run driven through the actual UI).
- `outputs/<project_id>/` and `outputs/ifc_design/` — generated site plans, floor plates, IFC
  files and 3D viewer HTML from the last run of each mode.
- `reports/` — generated PDF/LaTeX reports.

## Side tests (Revit / pyRevit)

`Testt/` holds two smaller, standalone experiments that read `.rvt` files directly (metadata,
embedded preview, element names) without needing Revit installed, and a pyRevit CLI wrapper +
extension for when Revit is available. See `Testt/README.md`. These are kept separate from the
main app because they depend on Autodesk sample files that aren't in this repo.

## Everything is sample data

RNGD has not yet provided real customer programs, site surveys, zoning sources, or BaaP catalog
data (see the open questions in `RNGDTeamPreparationQuestions.pdf`). Every file under `data/` is
clearly marked synthetic and is a placeholder for the real thing — swapping in real inputs does
not require changing any code, in either mode.
