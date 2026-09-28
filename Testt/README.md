# Testt: Revit side tests

Two experiments that read `.rvt` files, both runnable without Revit. The IFC parametric design
studio that used to live here has **moved into the main project** (`src/rngd/ifc/`, run via
`streamlit run src/app.py` from the repo root) — see the root `README.md`.

| Folder / file | What it does |
|---|---|
| `rvt_tools.py`, `rvt_cli.py`, `rvt_viewer.py` | Read `.rvt` files without Revit: metadata, embedded 128px preview, element-database names |
| `pyrevit_tools.py`, `pyrevit_extension/` | pyRevit CLI wrapper (works without Revit); in-Revit buttons (need Revit, untested) |
| `build_report.py` | Builds `report/rvt_report.pdf` (LaTeX) for the RVT experiment |

## 0. One-time setup (Windows, Python 3.11)

```bash
git clone https://github.com/naimul011/RNGD.git
cd RNGD
pip install -r requirements.txt
pip install olefile pymupdf
```

The `.env` file is **not** in the repository and is not needed for anything in this folder.

## 1. RVT reading experiment

Needs the three Autodesk sample archives (`rvtqsg-Sample_architecture.zip`, `..._MEP.zip`, `..._structural.zip`), which are **not** in the repo.
Put them in `Testt/` and extract each into `Testt/extracted/<zip name>/`:

```bash
cd Testt
python -c "import zipfile,glob,os; [zipfile.ZipFile(z).extractall('extracted/'+os.path.basename(z)[:-4]) for z in glob.glob('*.zip')]"
python rvt_cli.py info                 # version / build / path
python rvt_cli.py preview              # writes previews/*.png
python rvt_cli.py categories           # rough element-category signal
python rvt_cli.py search Bedroom       # search names in the element database
streamlit run rvt_viewer.py            # visual viewer (first load 10-30 s)
python -m pytest test_rvt_tools.py     # 17 tests (about 100 s)
```

Limit: only metadata, a tiny thumbnail and names are readable. Real 3D geometry needs Revit, Dynamo or the Autodesk cloud API.
That gap is exactly why the IFC Design Studio (now in the main app) exists: it builds a real, readable 3D model from scratch
instead of trying to read one out of a closed `.rvt` file.

## 2. pyRevit

pyRevit only runs inside Revit. Without Revit, only the CLI wrapper works:

```bash
# install: download pyRevit_..._signed.exe from https://github.com/pyrevitlabs/pyRevit/releases and run it (per-user, no admin)
cd Testt
python pyrevit_tools.py                 # file info via `pyrevit revits fileinfo`
python -m pytest test_pyrevit_tools.py
```

`pyrevit_extension/RNGDTest.extension` adds an RNGD tab (Summarize model, Export walls) once Revit 2022+ is installed. It is untested.

## 3. Building the PDF report (LaTeX)

Compiled with [Tectonic](https://github.com/tectonic-typesetting/tectonic/releases), a single-file LaTeX engine. Download the
Windows zip, extract `tectonic.exe` into `Testt/tools/`, then run `python build_report.py` (internet needed on first run to fetch LaTeX packages).
The main app's reports (`reports/RNGD_manual.pdf`, `reports/ifc_design/ifc_design_report.pdf`) use the same `tools/tectonic.exe`.

## Notes

- All zoning and BaaP data are synthetic samples; outputs are planning-level, not code-approved.
- If `streamlit` reports a port in use, add `--server.port 8502`.
