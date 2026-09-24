# IFC design studio (no Revit needed)

    pip install ifcopenshell plotly        # streamlit, matplotlib, pandas already present
    streamlit run app.py                   # Chrome app: http://localhost:8501
    python generate.py [0-4]               # writes out/<scenario>.ifc + .html (double-click the .html in Chrome)
    python -m pytest test_design.py        # 12 tests

Files: `design.py` (requirements + zoning presets + BaaP catalog -> layout + compliance checks),
`ifc_writer.py` (layout -> IFC4, and IFC -> meshes), `viewer3d.py` (Plotly 3D), `app.py` (UI).
Sample zoning/BaaP data is synthetic; planning-level only, not code-approved.
