# app/pages/02_Gaps.py

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.components.charts import (  # noqa: E402
    gap_type_treemap,
    gaps_by_paper_bar,
    papers_gap_heatmap,
    severity_donut,
)
from app.components.db_reader import get_gaps_df, get_papers_df  # noqa: E402
from app.components.pipeline_state import is_running  # noqa: E402

st.set_page_config(page_title="Gaps — ClaimCheck", layout="wide")
st.header("🔬 Reproducibility Gaps")

# ── Pipeline-in-progress banner ───────────────────────────────────────────────
_sid = st.session_state.get("audit_sid", "")
if _sid and is_running(_sid):
    st.info(
        "⏳ **Audit is running** — gap data will appear here once Gap Analysis "
        "(step 5) completes.",
        icon="⏳",
    )

gaps_df = get_gaps_df()
paper_df = get_papers_df()

if gaps_df.empty:
    st.info(
        "No reproducibility gaps yet.  \n"
        "Run an audit from the **Home** page to populate this view."
    )
    if st.button("Go to Home to start an audit", type="primary"):
        st.switch_page("streamlit_app.py")
    st.stop()

# ── Severity metrics ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Gaps", len(gaps_df))
c2.metric("🔴 Critical", int((gaps_df.severity == "critical").sum()), delta_color="inverse")
c3.metric("🟠 Major", int((gaps_df.severity == "major").sum()), delta_color="inverse")
c4.metric("🟡 Minor", int((gaps_df.severity == "minor").sum()))

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filters")
    sev_filter = st.multiselect(
        "Severity", ["critical", "major", "minor"],
        default=["critical", "major", "minor"],
    )
    gap_types = sorted(gaps_df["gap_type"].dropna().unique())
    type_filter = st.multiselect("Gap Type", gap_types)

df = gaps_df.copy()
if sev_filter:
    df = df[df["severity"].isin(sev_filter)]
if type_filter:
    df = df[df["gap_type"].isin(type_filter)]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_paper, tab_heatmap = st.tabs([
    "📊 Overview Charts", "📄 Per Paper", "🗺️ Heatmap",
])

with tab_overview:
    if df.empty:
        st.info("No gaps match the current filters.")
    else:
        ch1, ch2 = st.columns(2)
        ch1.plotly_chart(severity_donut(df), use_container_width=True)
        ch2.plotly_chart(gaps_by_paper_bar(df, paper_df), use_container_width=True)
        st.plotly_chart(gap_type_treemap(df), use_container_width=True)

with tab_paper:
    if df.empty:
        st.info("No gaps match the current filters.")
    else:
        _sev_order = {"critical": 0, "major": 1, "minor": 2}
        for pid in sorted(df["paper_id"].dropna().unique()):
            pgaps = df[df["paper_id"] == pid].copy()
            pgaps["_ord"] = pgaps["severity"].map(_sev_order)
            pgaps = pgaps.sort_values("_ord").drop(columns=["_ord"])

            crit = int((pgaps.severity == "critical").sum())
            maj = int((pgaps.severity == "major").sum())
            minor_cnt = int((pgaps.severity == "minor").sum())

            paper_row = paper_df[paper_df["arxiv_id"] == pid] if not paper_df.empty else None
            title = (
                paper_row["title"].values[0][:60]
                if (paper_row is not None and not paper_row.empty)
                else pid
            )
            badges = []
            if crit:
                badges.append(f"🔴 {crit} Critical")
            if maj:
                badges.append(f"🟠 {maj} Major")
            if minor_cnt:
                badges.append(f"🟡 {minor_cnt} Minor")

            with st.expander(
                f"**{pid}** — {title}  |  {'  ·  '.join(badges)}",
                expanded=(crit > 0),
            ):
                def _sev_style(val: str) -> str:
                    return {
                        "critical": "background-color:#fdecea;color:#c62828;font-weight:700",
                        "major":    "background-color:#fff3e0;color:#e65100;font-weight:700",
                        "minor":    "background-color:#e8f5e9;color:#2e7d32;font-weight:600",
                    }.get(val, "")

                styled = pgaps[
                    ["gap_type", "severity", "description", "paper_value", "code_value"]
                ].style.map(_sev_style, subset=["severity"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

with tab_heatmap:
    st.plotly_chart(papers_gap_heatmap(paper_df, df), use_container_width=True)
