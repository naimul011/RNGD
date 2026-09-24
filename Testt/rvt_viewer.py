"""RVT test viewer.  Run: streamlit run rvt_viewer.py"""
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

import rvt_tools as rt

HERE = Path(__file__).parent
st.set_page_config(page_title="RVT test viewer", layout="wide")
st.title("RVT test viewer")
st.caption("Reads Revit files without Revit: metadata, embedded preview, element-database strings.")

files = rt.find_rvts(HERE / "extracted")
if not files:
    st.error("No .rvt files under Testt/extracted. Unzip the archives first.")
    st.stop()

label = {f: f"{f.parent.name} / {f.name}" for f in files}
choice = st.sidebar.selectbox("File", files, format_func=lambda f: label[f])


@st.cache_data(show_spinner="Decompressing element database (10-30 s first time)...")
def load(path: str):
    p = Path(path)
    return rt.basic_info(p), rt.list_streams(p), rt.category_counts(p), rt.harvest_strings(p)


info, streams, cats, strings = load(str(choice))

tab1, tab2, tab3, tab4 = st.tabs(["Preview & info", "Categories", "Search names", "Streams"])

with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        png = rt.extract_preview(choice, HERE / "previews" / f"{choice.parent.name}.png")
        if png:
            img = Image.open(png)
            st.image(img.resize((img.width * 3, img.height * 3), Image.LANCZOS), caption=f"Embedded preview ({img.width}x{img.height}px, upscaled 3x)")
        else:
            st.warning("No preview image in this file.")
    with c2:
        st.json({k: v for k, v in info.items() if k != "project_information_raw"})
    st.info("Revit only embeds a tiny thumbnail. Full 3D geometry needs Revit, Autodesk Platform Services, or Dynamo.")

with tab2:
    df = pd.DataFrame({"category": list(cats), "string hits": list(cats.values())}).set_index("category")
    st.bar_chart(df)
    st.caption("Heuristic: occurrences of category keywords in the element database, not exact element counts.")

with tab3:
    q = st.text_input("Search names (levels, rooms, families, views...)", "Level")
    if q:
        hits = [(s, n) for s, n in strings.items() if q.lower() in s.lower() and not s.startswith("autodesk.")]
        hits.sort(key=lambda x: -x[1])
        st.dataframe(pd.DataFrame(hits[:200], columns=["name", "count"]), use_container_width=True, hide_index=True)

with tab4:
    st.dataframe(pd.DataFrame(streams), use_container_width=True, hide_index=True)
