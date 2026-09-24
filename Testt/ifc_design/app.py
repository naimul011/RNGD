"""Chrome app: requirements + zoning + BaaP -> parametric IFC design you can click through.
Run: streamlit run app.py"""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

from design import SCENARIOS, ZONING, Params, generate, module_counts
from ifc_writer import read_meshes, write_ifc
from viewer3d import build_figure

st.set_page_config(page_title="IFC design studio", layout="wide")
OUT = HERE / "out"

DEFAULTS = dict(units=120, floors=4, zoning="Rivermont R-4 (sample)", parking="surface", wings=0, studio=20, onebr=45)


def load(p: Params):
    st.session_state.update(units=p.units, floors=p.floors, zoning=p.zoning, parking=p.parking, wings=p.wings,
                            studio=round(p.mix[0] * 100), onebr=round(p.mix[1] * 100))


for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.title("IFC design studio")
st.caption("Requirements + sample zoning + BaaP catalog -> parametric building -> standard .ifc file. Sample data, planning-level, not code-approved.")

with st.sidebar:
    st.subheader("Try a scenario")
    for name, p in SCENARIOS.items():
        st.button(name, on_click=load, args=(p,), use_container_width=True)
    st.divider()
    st.subheader("Requirements")
    with st.expander("Paste a customer request (AI parse)"):
        text = st.text_area("Request text", (HERE.parents[1] / "data/sample_projects/project_001/customer_rfp.txt").read_text(encoding="utf-8"), height=140)
        if st.button("Extract requirements"):
            try:
                from rngd.agents import intake_agent
                prog, _ = intake_agent.run(text)
                mix = prog.unit_mix_pct
                st.session_state.update(units=prog.unit_count_total or 120, floors=prog.floor_count or 4,
                                        studio=round(mix.get("studio", .2) * 100), onebr=round(mix.get("1br", .45) * 100))
                st.success("Requirements loaded into the controls below.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not parse: {e}")
    st.number_input("Units", 10, 400, key="units", step=6)
    st.slider("Residential floors", 1, 9, key="floors")
    st.slider("Studio %", 0, 100, key="studio")
    st.slider("1-bedroom %", 0, 100 - st.session_state.studio, key="onebr")
    st.caption(f"2-bedroom %: {100 - st.session_state.studio - st.session_state.onebr}")
    st.divider()
    st.subheader("Site, zoning, options")
    st.selectbox("Zoning preset", list(ZONING), key="zoning")
    st.radio("Parking", ["surface", "podium"], key="parking", horizontal=True)
    st.select_slider("Wings (0 = auto)", [0, 1, 2, 3], key="wings")
    level_view = st.select_slider("View floors up to", ["all"] + list(range(1, 10)), value="all")

s = st.session_state
studio, onebr = s.studio / 100, s.onebr / 100
params = Params(units=s.units, floors=s.floors, zoning=s.zoning, parking=s.parking, wings=s.wings, mix=(studio, onebr, max(0, 1 - studio - onebr)))


@st.cache_data(show_spinner="Generating IFC...")
def build(p: Params):
    lay = generate(p)
    ifc = write_ifc(lay, Path(tempfile.gettempdir()) / "rngd_design.ifc")
    return lay, ifc.read_bytes(), read_meshes(ifc)


lay, ifc_bytes, meshes = build(params)
m = lay.metrics
fails = [c for c in lay.checks if c["status"] == "FAIL"]

c = st.columns(6)
c[0].metric("Units", m["units_placed"], f"of {params.units}")
c[1].metric("Stories", m["stories"])
c[2].metric("FAR", m["far"])
c[3].metric("Coverage", f"{m['coverage_pct']}%")
c[4].metric("Parking", f"{m['parking_provided']}/{m['parking_required']}")
c[5].metric("Checks failed", len(fails))

tab3d, tabplan, tabchk, tabdl = st.tabs(["3D model", "Site plan", "Compliance", "Download"])
with tab3d:
    st.plotly_chart(build_figure(meshes, None if level_view == "all" else int(level_view)), use_container_width=True)
    st.caption("Drag to rotate, scroll to zoom, click legend entries to hide or show unit types. Rebuilt from the saved .ifc file.")

with tabplan:
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.add_patch(Rectangle((0, 0), params.lot_w, params.lot_d, fill=False, lw=2))
    if params.easement_east:
        ax.add_patch(Rectangle((params.lot_w - params.easement_east, 0), params.easement_east, params.lot_d, color="#FFBF00", alpha=.8))
    e = lay.env
    ax.add_patch(Rectangle((e["x0"], e["y0"]), e["w"], e["d"], fill=False, ls="--", ec="#CC0000"))
    if lay.drive:
        d = lay.drive
        ax.add_patch(Rectangle((d["x"], d["y"]), d["w"], d["d"], color="#555555"))
    for st_ in lay.stalls:
        ax.add_patch(Rectangle((st_["x"], st_["y"]), st_["w"], st_["d"], fc="#8C8C8C", ec="white", lw=.4))
    for w in lay.wings:
        ax.add_patch(Rectangle((w["x"], w["y"]), w["w"], w["h"], fc="#3E7CB1", alpha=.85, ec="black"))
    ax.set_xlim(-10, params.lot_w + 10)
    ax.set_ylim(params.lot_d + 10, -10)
    ax.set_aspect("equal")
    ax.set_title("Top view: street at top")
    st.pyplot(fig)

with tabchk:
    df = pd.DataFrame(lay.checks).astype(str)
    st.dataframe(df.style.map(lambda v: {"PASS": "color:#2E7D52", "FAIL": "color:#B3261E", "WARN": "color:#B0590B"}.get(v, ""), subset=["status"]),
                 hide_index=True, use_container_width=True)
    st.caption(f"Module counts requested: {module_counts(params)}. Zoning: {ZONING[params.zoning]}")

with tabdl:
    st.download_button("Download .ifc", ifc_bytes, "rngd_design.ifc", "application/x-step")
    html = build_figure(meshes).to_html(include_plotlyjs="cdn")
    st.download_button("Download 3D viewer (.html, opens in Chrome)", html, "rngd_design_3d.html", "text/html")
    st.caption("The .ifc opens in any BIM tool (BIMvision, Blender-BIM, FreeCAD, Autodesk Viewer).")
