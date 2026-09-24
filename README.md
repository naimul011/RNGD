# RNGD Concept-to-Schematic Multi-Agent Harness (Semester Proof-of-Concept)

A multi-agent AI prototype for RNGD's "concept-to-schematic" workflow: turn an
unstructured customer request + site conditions into a Building-as-a-Product
(BaaP) schematic site plan and floor plate, with automated conflict detection
and one final, defensible AI decision.

See `REPORT.md` for the full write-up (architecture, design rationale, how
this maps to RNGD's own kickoff questions, results, and limitations).

## Quick start

```bash
pip install -r requirements.txt
python src/make_sample_data.py     # only needed once, or if you edit the sample site
python src/run_pipeline.py         # CLI: runs the pipeline, prints the trace + decision
streamlit run src/app.py           # interactive demo UI
```

Requires a `.env` in the project root with `GROQ_API_KEY` and
`ANTHROPIC_API_KEY` (already present in this project).

## Layout

- `data/knowledge_base/` — sample zoning code, BaaP catalog, stacking rules,
  design guidelines (markdown, RAG-indexed at startup).
- `data/baap_catalog.json` — machine-readable twin of the BaaP catalog.
- `data/sample_projects/project_001/` — one synthetic sample project: a raw
  customer RFP letter, `site_conditions.json`, and a synthetic site sketch
  PNG + georeference sidecar.
- `src/rngd/` — the package: config, schemas, vector store, LLM wrappers,
  agents, classical CV, physics simulation, and the orchestrator (harness).
- `outputs/<project_id>/` — generated site plans, floor plates, convergence
  chart, and the full `project_state.json` trace from the last run.

## Everything is sample data

RNGD has not yet provided real customer programs, site surveys, zoning
sources, or BaaP catalog data (see the open questions in
`RNGDTeamPreparationQuestions.pdf`). Every file under `data/` is clearly
marked synthetic and is a placeholder for the real thing — swapping in real
inputs does not require changing any code.

## Side tests (Revit / IFC)
See `Testt/README.md` for how to run the IFC design studio, the RVT reader and the pyRevit tools.
