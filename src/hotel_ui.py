"""Hotel Schematic mode for the Streamlit app: program editor, interactive plans and
3D, room mix, unit types, code / ADA screen and the PDF drawing set with its results.

app.py calls render(); everything heavy (sheet images) is cached by program + settings.
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

import build_final_report  # noqa: E402
from rngd.hotel import details as hd  # noqa: E402
from rngd.hotel import drawings, layout  # noqa: E402
from rngd.hotel import interactive as hi  # noqa: E402
from rngd.hotel.program import GROUP_TITLES, Program  # noqa: E402

CATS = ["guest", "shared", "boh", "business", "tenant"]
STATUS_SEV = {"PASS": "ok", "FAIL": "critical", "REVIEW": "medium", "ASSUMED": "info"}
GROUP_COLORS = {"Hotel Units": "#9B7FB0", "Shared": "#4D7798", "BOH": "#9A9A9A", "Business": "#5FB3A6",
                "Tenant": "#E8B85C"}


def settings(ss) -> tuple[dict, dict, float, float]:
    site = {"envelope_w": ss.get("hot_w", 170), "envelope_d": ss.get("hot_d", 265)}
    lk = {"upper_levels": int(ss.get("hot_floors", 5)), "ground_height_ft": float(ss.get("hot_l1", 16.0)),
          "floor_height_ft": float(ss.get("hot_fl", 10.75))}
    return site, lk, float(ss.get("hot_circ", 33)), float(ss.get("hot_wall", 8))


def _key(prog: Program, site: dict, lk: dict):
    return tuple(prog.rows), prog.source, tuple(sorted(site.items())), tuple(sorted(lk.items()))


def _rebuild(rows, source, site_t, lk_t):
    return layout.build(Program(rows=list(rows), source=source), dict(site_t), **dict(lk_t))


def _png(fig, dpi=130) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=64)
def level_drawing(rows, source, site_t, lk_t, level_idx: int) -> bytes:
    sch = _rebuild(rows, source, site_t, lk_t)
    lvl = sch.levels[level_idx]
    fig, ax = plt.subplots(figsize=(15, 5.2))
    drawings.draw_level(ax, sch, lvl, drawings.grid_lines(sch))
    ax.set_xlim(-26, max(lv.length_ft for lv in sch.levels) + 6)
    ax.set_ylim(-10, layout.WING_W_FT + 23)
    ax.set_aspect("equal")
    ax.axis("off")
    return _png(fig, 150)


@st.cache_data(show_spinner=False, max_entries=64)
def unit_images(name: str, area_each: float) -> tuple[bytes, bytes, bytes]:
    w = area_each / layout.ROOM_DEPTH_FT
    d = layout.ROOM_DEPTH_FT
    fig, ax = plt.subplots(figsize=(4.6, 5.6))
    ax.add_patch(plt.Rectangle((0, 0), w, d, fc="#F8F5EF", ec="none"))
    ax.add_patch(plt.Rectangle((-2, -3), w + 4, 3, fc=drawings.FILL["corridor"], alpha=0.35, ec="none"))
    g = drawings.unit_geometry(name, w, d)
    drawings.draw_unit(ax, g, lambda u, v: (u, v), lw_scale=1.6)
    drawings.dim_h(ax, 0, w, d + 2.2, fs=8, ext_from=d + 0.5)
    drawings.dim_v(ax, 0, d, w + 4, fs=8, ext_from=w + 0.5)
    ax.text(w / 2, -1.5, "CORRIDOR", ha="center", va="center", fontsize=8, color="#123A6E")
    ax.set_xlim(-4, w + 6)
    ax.set_ylim(-4, d + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    plan = _png(fig)
    out = [plan]
    for az in (-55.0, 125.0):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        drawings.iso_unit(ax, name, w, az=az)
        out.append(_png(fig))
    return tuple(out)


@st.cache_data(show_spinner=False)
def rule_texts() -> dict[str, str]:
    try:
        df = pd.read_csv(build_final_report.RULES_CSV, encoding="utf-8", encoding_errors="replace")
        return {r.rule_id: f"{r.rule_name}. {r.rule_description} ({r.base_code} {r.base_section})"
                for r in df.itertuples()}
    except Exception:
        return {}


ADA_TEXT = {
    "ADA-224.2": "Table 224.2: transient lodging must provide guest rooms with mobility features (806.2); "
                 "for 101-150 rooms: 5 without roll-in showers + 2 with roll-in showers = 7.",
    "ADA-224.4": "Table 224.4: guest rooms with communication features (806.3); 101-150 rooms: 12.",
    "ADA-224.5": "224.5 Dispersion: accessible rooms spread among room types, number of beds and amenities.",
}


def _item(c: dict, advice: str = "") -> str:
    sev = STATUS_SEV.get(c["status"], "info")
    tag = c["rule_id"] if c["rule_id"] != "-" else c.get("source", "")
    extra = f"<div class='item-detail'><b>Note:</b> {c['note']}</div>" if c.get("note") else ""
    nxt = f"<div class='item-detail'><b>Next step:</b> {advice}</div>" if advice else ""
    return (f"<div class='rngd-item {sev}'><div class='item-top'><span>{c['status']} · {c['check']}</span>"
            f"<span class='rngd-tag'>{tag}</span></div><div class='item-detail'>Required {c['required']} · "
            f"provided <b>{c['provided']}</b></div>{extra}{nxt}</div>")


KPI_CSS = """<style>
.hk-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:8px;margin:4px 0 10px}
.hk{background:var(--blueprint-wash);border:1px solid rgba(30,90,168,.18);border-radius:8px;padding:8px 11px}
.hk .l{font-size:11.5px;color:var(--muted)}
.hk .v{font-family:"IBM Plex Mono",monospace;font-size:19px;font-weight:600;color:var(--ink);line-height:1.25;overflow-wrap:anywhere}
.hk .s{font-size:11px;color:var(--muted)}
.hk.bad{background:rgba(163,42,33,.07);border-color:rgba(163,42,33,.3)} .hk.bad .v{color:var(--critical)}
.hk.good{background:var(--ok-wash);border-color:rgba(46,107,71,.3)} .hk.good .v{color:var(--ok)}
.hk-title{font-size:13px;color:var(--muted);margin:2px 0 2px}
.hk-title b{color:var(--ink);font-size:16px}
</style>"""


def kpis(items, title: str = "") -> None:
    """items: (label, value, sub, tone) with tone '' | 'good' | 'bad'. Wraps instead of truncating."""
    tiles = "".join(f"<div class='hk {t}'><div class='l'>{l}</div><div class='v'>{v}</div>"
                    f"<div class='s'>{s}</div></div>" for l, v, s, t in items)
    st.html(KPI_CSS + (f"<div class='hk-title'>{title}</div>" if title else "") + f"<div class='hk-grid'>{tiles}</div>")


# ------------------------------------------------------------------ tabs -
def tab_overview(sch, circ, wall):
    p = sch.program
    n = p.net_to_gross(circ, wall)
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        c1, c2 = st.columns([1, 1])
        explode = c1.slider("Separate floors (ft)", 0, 30, 0, key="hot_explode",
                            help="Pull the floors apart to see every level's rooms.")
        lv_pick = c2.multiselect("Show levels", [lv.name for lv in sch.levels], key="hot_lv_pick",
                                 placeholder="All levels")
        st.plotly_chart(hi.massing_figure(sch, explode=explode, levels=lv_pick or None), use_container_width=True)
        st.caption("Every room is a block coloured like the drawings. Hover for name, area and size; drag to orbit; "
                   "click a legend entry to hide a room kind.")
    with right:
        st.markdown("**Key results**")
        bad = [c for c in sch.checks if c["status"] != "PASS"]
        st.html("".join(_item(c, hd.failing_advice(c)) for c in bad) or
                "<div class='rngd-item ok'>All checks pass.</div>")
        st.markdown("**Gross area: three ways**")
        gdf = pd.DataFrame({
            "Basis": ["Net program", "Source sheets (L+L)", f"Meeting rule ({circ:.0f}% + {wall:.0f}%)", "This schematic (drawn)"],
            "SF": [p.net_sf, p.gross_sf, n["required_gross_sf"], sum(lv.gross_sf for lv in sch.levels)]})
        fig = px.bar(gdf, x="SF", y="Basis", orientation="h", text="SF", color="Basis",
                     color_discrete_sequence=["#9B7FB0", "#9A9A9A", "#E8B85C", "#4D7798"])
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="inside")
        fig.update_layout(height=220, showlegend=False, margin=dict(l=0, r=0, t=0, b=0), yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Source ratio {p.gross_sf / p.net_sf:.2f}; meeting rule ratio "
                   f"{n['required_gross_sf'] / p.net_sf:.2f}; drawn ratio {sum(lv.gross_sf for lv in sch.levels) / p.net_sf:.2f}.")


def tab_program(ss, sch):
    p = sch.program
    st.markdown("**Edit the program.** Change counts or areas, add rows at the bottom, or delete rows "
                "(select the row, then the bin icon). *Area* is the total for the row, as in the L+L schedule. "
                "Press **Apply** to redraw everything.")
    df = pd.DataFrame(p.rows, columns=["Room type", "Count", "Total area (SF)", "Category"])
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, hide_index=True, height=380,
        key=f"hot_editor_{ss.get('hot_prog_ver', 0)}",
        column_config={
            "Room type": st.column_config.TextColumn(required=True),
            "Count": st.column_config.NumberColumn(min_value=0, step=1, required=True),
            "Total area (SF)": st.column_config.NumberColumn(min_value=0, step=1, format="%d", required=True),
            "Category": st.column_config.SelectboxColumn(options=CATS, required=True,
                                                         help="guest = hotel unit (a key); the rest follow the L+L groups"),
        })
    c1, c2, c3 = st.columns([1, 1, 2])
    changed = not edited.reset_index(drop=True).equals(df)
    if c1.button("Apply changes", type="primary", disabled=not changed, use_container_width=True):
        rows = [(str(r["Room type"]).strip(), int(r["Count"]), float(r["Total area (SF)"]), str(r["Category"]))
                for _, r in edited.dropna(subset=["Room type", "Count", "Total area (SF)", "Category"]).iterrows()
                if str(r["Room type"]).strip()]
        if not any(cat == "guest" and c > 0 for _, c, _, cat in rows):
            st.error("The program needs at least one guest-room row.")
        else:
            src = p.source if p.source.startswith("Edited") else f"Edited in app (from: {p.source})"
            ss.hotel_program = Program(rows=rows, source=src, gross_by_level=dict(p.gross_by_level))
            ss.hot_prog_ver = ss.get("hot_prog_ver", 0) + 1
            st.rerun()
    if c2.button("Discard edits", disabled=not changed, use_container_width=True):
        ss.hot_prog_ver = ss.get("hot_prog_ver", 0) + 1
        st.rerun()
    c3.caption("Unsaved edits" if changed else f"{len(p.rows)} rows · {p.keys} keys · {p.net_sf:,.0f} SF net")

    st.divider()
    det = hd.program_detail(sch)
    g1, g2 = st.columns(2)
    by_group = det.groupby("Group", as_index=False)["Total area (SF)"].sum()
    fig = px.bar(by_group.sort_values("Total area (SF)"), x="Total area (SF)", y="Group", orientation="h",
                 color="Group", color_discrete_map=GROUP_COLORS, text="Total area (SF)", title="Net area by group")
    fig.update_traces(texttemplate="%{text:,.0f} SF", textposition="outside", textfont_size=12, cliponaxis=False)
    fig.update_layout(height=280, showlegend=False, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="")
    g1.plotly_chart(fig, use_container_width=True)
    keys = det[det["Group"] == GROUP_TITLES["guest"]]
    fig = px.pie(keys, names="Room type", values="Count", hole=0.5, title=f"Room mix ({p.keys} keys)")
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
    g2.plotly_chart(fig, use_container_width=True)

    st.markdown("**Detailed program** — sizes, beds, sleeping capacity, where each line is placed, IBC occupant "
                "load and fit-out.")
    f1, f2 = st.columns([1, 2])
    groups = f1.multiselect("Groups", list(det["Group"].unique()), key="hot_det_groups", placeholder="All groups")
    q = f2.text_input("Search", key="hot_det_q", placeholder="e.g. suite, ADA, stair")
    view = det
    if groups:
        view = view[view["Group"].isin(groups)]
    if q:
        view = view[view.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    view = view.astype({"Sleeps": str, "Occupant load": str})  # blanks for non-guest rows
    st.dataframe(view, hide_index=True, use_container_width=True, height=420, column_config={
        "% of net": st.column_config.ProgressColumn(min_value=0, max_value=float(det["% of net"].max()), format="%.1f%%"),
        "Area each (SF)": st.column_config.NumberColumn(format="%d"),
        "Total area (SF)": st.column_config.NumberColumn(format="%d"),
        "Fit-out": st.column_config.TextColumn(width="large")})
    t = det[["Count", "Total area (SF)", "Sleeps", "Occupant load"]].apply(pd.to_numeric, errors="coerce").sum()
    st.caption(f"Totals: {int(t['Count'])} items · {t['Total area (SF)']:,.0f} SF · sleeps {int(t['Sleeps'])} · "
               f"IBC occupant load {int(t['Occupant load'])} (rooms marked 'not separately counted' excluded).")


def tab_plans(sch, key_t):
    names = [lv.name for lv in sch.levels]
    c1, c2, c3 = st.columns([2.4, 1.2, 1.4])
    lvl_name = c1.radio("Level", names, horizontal=True, key="hot_level")
    style = c2.radio("View", ["Interactive", "Drawing"], horizontal=True, key="hot_plan_style",
                     help="Interactive: hover any room. Drawing: the sheet-style plan used in the PDF.")
    idx = names.index(lvl_name)
    lvl = sch.levels[idx]
    types = sorted({r.label for r in lvl.rooms if r.label})
    hl = c3.selectbox("Highlight", ["(none)"] + types, key=f"hot_hl_{idx}")
    kpis([("Keys", f"{lvl.keys_placed}", lvl.name, ""), ("Gross", f"{lvl.gross_sf:,.0f}", "SF", ""),
          ("Footprint", f"{lvl.length_ft:.0f} x {lvl.width_ft:.0f}", "ft", ""),
          ("Floor at", f"{lvl.elevation_ft:.2f}", "ft above grade", ""),
          ("Rooms", f"{sum(1 for r in lvl.rooms if r.label and r.kind != 'corridor')}", "incl. core", "")])
    if style == "Interactive":
        st.plotly_chart(hi.plan_figure(lvl, highlight=None if hl == "(none)" else hl), use_container_width=True)
    else:
        with st.spinner("Drawing..."):
            st.image(level_drawing(*key_t, idx), use_container_width=True)
    with st.expander(f"Room schedule — {lvl.name}", expanded=False):
        st.dataframe(hd.level_rooms(lvl), hide_index=True, use_container_width=True)


def tab_mix(sch):
    mix = hd.mix_by_floor(sch)
    st.markdown("**Keys of each type on each guest floor.** Larger units are placed first and spread evenly; "
                "ADA units are dispersed across floors and types.")
    st.dataframe(mix.style.background_gradient(cmap="Purples", subset=pd.IndexSlice[mix.index[:-1], mix.columns[:-1]]),
                 use_container_width=True)
    long = mix.drop(index="Keys per floor").drop(columns="Total").reset_index(names="Type").melt(
        id_vars="Type", var_name="Level", value_name="Keys")
    fig = px.bar(long, x="Level", y="Keys", color="Type", title="Room mix by floor")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


def tab_units(sch):
    types = [(n, c, a) for n, c, a, cat in sch.program.rows if cat == "guest" and c]
    names = [t[0] for t in types]
    pick = st.selectbox("Unit type", names, key="hot_unit")
    name, count, area = types[names.index(pick)]
    each = area / count
    s = hd.unit_spec(name, each)
    floors = hd.mix_by_floor(sch)
    on = ", ".join(f"{c.replace('Level ', 'L')}: {int(v)}" for c, v in floors.loc[name].items() if c != "Total" and v) \
        if name in floors.index else "-"
    kpis([("Count", f"{count}", "keys", ""), ("Area each", f"{each:,.0f}", "SF", ""),
          ("Size", f"{s['width_ft']:.1f}' x {s['depth_ft']:.0f}'", "W x D", ""), ("Beds", s["beds"], "", ""),
          ("Sleeps", f"{s['sleeps']}", "per room", ""), ("ADA", "yes" if s["ada"] else "no", "mobility features", "")])
    st.caption(f"**Layout:** {s['layout']} · **Fit-out:** {s['fit_out']} · **Floors:** {on}")
    plan, iso_a, iso_b = unit_images(name, each)
    c1, c2, c3 = st.columns([1, 1.15, 1.15])
    c1.image(plan, caption="Enlarged plan (corridor at bottom, window at top)", use_container_width=True)
    c2.image(iso_a, caption="3D view from the corridor side", use_container_width=True)
    c3.image(iso_b, caption="3D view from the window side", use_container_width=True)


def tab_code(sch):
    counts = {s: sum(c["status"] == s for c in sch.checks) for s in ["PASS", "FAIL", "REVIEW", "ASSUMED"]}
    st.html("".join(f"<span class='status-chip {'PASS' if s == 'PASS' else 'FAIL' if s == 'FAIL' else 'WARN'}' "
                    f"style='margin-right:6px'>{s} {n}</span>" for s, n in counts.items()))
    show = st.multiselect("Show", list(counts), default=list(counts), key="hot_code_show")
    texts = rule_texts() | ADA_TEXT
    for c in sch.checks:
        if c["status"] not in show:
            continue
        st.html(_item(c, hd.failing_advice(c) if c["status"] != "PASS" else ""))
        if c["rule_id"] in texts:
            with st.expander(f"Rule text · {c['rule_id']}"):
                st.write(texts[c["rule_id"]])
    st.download_button("Download checks (CSV)", pd.DataFrame(sch.checks).to_csv(index=False),
                       file_name="hotel_code_checks.csv", mime="text/csv")
    st.caption("Sources: RNGD IBC 2021 R1 rule CSV (docs/) and the 2010 ADA Standards PDF (Tables 224.2, 224.4, 224.5). "
               "Planning-level screen, not a code determination.")


def tab_report(ss, sch, site, lk):
    man = build_final_report.load_manifest()
    c1, c2 = st.columns([1, 2])
    if c1.button("Generate PDF report", type="primary", use_container_width=True):
        with st.spinner("Drawing 10 sheets and compiling the report (about a minute)..."):
            try:
                build_final_report.build(sch.program, site, layout_kwargs=lk)
                man = build_final_report.load_manifest()
                ss.hotel_report = man["pdf"]
            except Exception as e:
                st.error(f"Report failed: {e}")
    c2.caption("Builds the RNGD L+L-style drawing set from the program and settings on screen: overview + "
               "schedules, code summary, site plan, floor plans, elevations, enlarged unit plans and unit 3D views, "
               "plus appendices (program, gross area, code checks, IBC rule set, workbook, open questions).")
    if not man:
        st.info("No report yet. Press **Generate PDF report**.")
        return
    stale = (man.get("program_source") != sch.program.source or man["summary"]["keys"] != sch.program.keys
             or man.get("site") != site or man.get("layout") != lk
             or abs(man["summary"]["net_sf"] - sch.program.net_sf) > 0.5)
    if stale:
        st.warning("This report was built from different program or settings than what is on screen. "
                   "Regenerate to update it.")
    pdf = Path(man["pdf"])
    s = man["summary"]
    st.markdown(f"**Latest report** · generated {man['generated'].replace('T', ' ')} · {man['pages']} pages · "
                f"program: {man['program_source']}")
    kpis([("Keys", f"{s['keys']}", "", ""), ("Net SF", f"{s['net_sf']:,.0f}", "", ""),
          ("Gross drawn", f"{s['gross_drawn_sf']:,.0f}", "SF", ""), ("Gross rule", f"{s['gross_meeting_sf']:,.0f}", "SF", ""),
          ("Levels", f"{s['levels']}", "", ""), ("Height", f"{s['height_ft']:.1f} ft", "", ""),
          ("Sheets", f"{len(man['sheets'])}", f"{man['pages']} pages total", "")])
    if pdf.exists():
        st.download_button("Download the PDF report", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf",
                           type="primary")
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        sheets = [sh for sh in man["sheets"] if Path(sh["png"]).exists()]
        if sheets:
            labels = [f"{sh['number']} — {sh['title']}" for sh in sheets]
            pick = st.radio("Sheet", labels, horizontal=False, key="hot_sheet", label_visibility="collapsed") \
                if len(labels) <= 4 else st.selectbox("Sheet", labels, key="hot_sheet")
            sh = sheets[labels.index(pick)]
            st.image(sh["png"], use_container_width=True)
            st.download_button(f"Download {sh['number']}.png", Path(sh["png"]).read_bytes(),
                               file_name=f"{sh['number']}.png", mime="image/png")
            st.markdown("**All sheets**")
            cols = st.columns(5)
            for i, x in enumerate(sheets):
                cols[i % 5].image(x["png"], caption=x["number"], use_container_width=True)
    with right:
        st.markdown("**Results in the report**")
        st.html("".join(_item(c) for c in man["checks"]))
        st.markdown("**Open questions for RNGD**")
        for i, q in enumerate(man["open_questions"], 1):
            st.markdown(f"{i}. {q}")
        with st.expander("Assumptions"):
            for a in man["assumptions"]:
                st.markdown(f"- {a}")
        with st.expander("Keys by floor"):
            st.dataframe(pd.DataFrame(man["mix_by_floor"]).T, use_container_width=True)


def render(ss):
    prog: Program = ss.hotel_program
    site, lk, circ, wall = settings(ss)
    sch = layout.build(prog, site, **lk)
    ss.hotel_sch = sch
    key_t = _key(prog, site, lk)
    scored = [c for c in sch.checks if c["status"] in ("PASS", "FAIL")]
    fails = [c for c in sch.checks if c["status"] == "FAIL"]
    n = prog.net_to_gross(circ, wall)

    drawn = sum(lv.gross_sf for lv in sch.levels)
    kpis([
        ("Keys", f"{prog.keys}", f"{len(sch.levels) - 1} guest floors", ""),
        ("Net program", f"{prog.net_sf:,.0f}", "SF", ""),
        ("Gross drawn", f"{drawn:,.0f}", f"{drawn - n['required_gross_sf']:+,.0f} vs rule", ""),
        (f"Gross rule {circ:.0f}%+{wall:.0f}%", f"{n['required_gross_sf']:,.0f}", "SF", ""),
        ("Massing", f"{sch.levels[0].length_ft:.0f}x{sch.levels[0].width_ft:.0f}", f"podium · tower {sch.levels[1].length_ft:.0f} ft", ""),
        ("Height", f"{sch.height_ft:.1f} ft", f"{len(sch.levels)} levels", ""),
        ("Code checks", f"{len(scored) - len(fails)}/{len(scored)}", f"{len(fails)} fail" if fails else "all pass",
         "bad" if fails else "good"),
    ], title=f"<b>Hotel schematic</b> · RNGD Living + Lodging format · program: {prog.source}")
    if sch.unplaced_keys:
        st.error(f"{sch.unplaced_keys} keys could not be placed.")

    tabs = st.tabs(["Overview & 3D", "Optimize layouts", "Program", "Floor plans", "Room mix", "Unit types",
                    "Code & ADA", "PDF report"])
    with tabs[0]:
        tab_overview(sch, circ, wall)
    with tabs[1]:
        import hotel_opt_ui
        hotel_opt_ui.render_tab(ss, kpis)
    with tabs[2]:
        tab_program(ss, sch)
    with tabs[3]:
        tab_plans(sch, key_t)
    with tabs[4]:
        tab_mix(sch)
    with tabs[5]:
        tab_units(sch)
    with tabs[6]:
        tab_code(sch)
    with tabs[7]:
        tab_report(ss, sch, site, lk)
