# Testt: Revit and IFC side tests

Two independent experiments, both runnable without Revit.

| Folder / file | What it does |
|---|---|
| `rvt_tools.py`, `rvt_cli.py`, `rvt_viewer.py` | Read `.rvt` files without Revit: metadata, embedded 128px preview, element-database names |
| `pyrevit_tools.py`, `pyrevit_extension/` | pyRevit CLI wrapper (works without Revit); in-Revit buttons (need Revit, untested) |
| `build_report.py` | Builds `report/rvt_report.pdf` (LaTeX) for the RVT experiment |
| `ifc_design/` | **Parametric IFC design studio**: requirements + zoning + BaaP -> `.ifc` + 3D viewer in Chrome |

## 0. One-time setup (Windows, Python 3.11)

```bash
git clone https://github.com/naimul011/RNGD.git
cd RNGD
pip install -r requirements.txt
pip install olefile pymupdf ifcopenshell plotly pytest
```

The `.env` file is **not** in the repository. Only the optional AI request parser in the IFC app needs it. Create `RNGD/.env` with:

```
GROQ_API_KEY=your_key_here
```

Everything else works without any key.

## 1. IFC design studio (main demo)

```bash
cd Testt/ifc_design
streamlit run app.py          # opens http://localhost:8501 in Chrome
```

In the app: click a scenario button on the left, or change units, floors, unit mix, zoning preset, parking type and wings.
Tabs: **3D model** (drag to rotate, click the legend to hide unit types, "view floors up to" slider), **Site plan**, **Compliance**
(ten PASS/FAIL checks), **Download** (`.ifc` and a standalone `.html` viewer).

Other commands (run inside `Testt/ifc_design`):

```bash
python generate.py            # all scenarios -> out/*.ifc and out/*.html (double-click the .html)
python generate.py 1          # one scenario by index
python -m pytest test_design.py -v   # 12 tests (takes about 90 s)
python build_report.py        # rebuild report/ifc_design_report.pdf (needs Tectonic, see section 4)
```

Ready-made outputs are committed in `ifc_design/out/`. Open any `.html` in Chrome, or any `.ifc` in BIMvision, Blender-BIM, FreeCAD or the free Autodesk Viewer.
The full explanation is in `ifc_design/report/ifc_design_report.pdf` and `ifc_design/README.md`.

## 2. RVT reading experiment

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

## 3. pyRevit

pyRevit only runs inside Revit. Without Revit, only the CLI wrapper works:

```bash
# install: download pyRevit_..._signed.exe from https://github.com/pyrevitlabs/pyRevit/releases and run it (per-user, no admin)
cd Testt
python pyrevit_tools.py                 # file info via `pyrevit revits fileinfo`
python -m pytest test_pyrevit_tools.py
```

`pyrevit_extension/RNGDTest.extension` adds an RNGD tab (Summarize model, Export walls) once Revit 2022+ is installed. It is untested.

## 4. Building the PDF reports (LaTeX)

Reports are compiled with [Tectonic](https://github.com/tectonic-typesetting/tectonic/releases), a single-file LaTeX engine. Download the
Windows zip, extract `tectonic.exe` into `Testt/tools/`, then run the `build_report.py` scripts above (internet needed on first run to fetch LaTeX packages).

## Notes

- All zoning and BaaP data are synthetic samples; outputs are planning-level, not code-approved.
- If `streamlit` reports a port in use, add `--server.port 8502`.
