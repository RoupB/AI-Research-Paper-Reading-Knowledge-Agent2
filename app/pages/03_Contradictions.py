# app/pages/03_Contradictions.py

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.components.charts import (  # noqa: E402
    contradiction_network_chart,
    contradictions_by_metric_chart,
)
from app.components.db_reader import get_contradictions_df, get_papers_df  # noqa: E402
from app.components.pipeline_state import is_running  # noqa: E402

st.set_page_config(page_title="Contradictions — ClaimCheck", layout="wide")
st.header("⚡ Cross-Paper Contradictions")

# ── Pipeline-in-progress banner ───────────────────────────────────────────────
_sid = st.session_state.get("audit_sid", "")
if _sid and is_running(_sid):
    st.info(
        "⏳ **Audit is running** — contradiction data will appear here once "
        "Contradiction Mapping (step 6) completes.",
        icon="⏳",
    )

contra_df = get_contradictions_df()
paper_df = get_papers_df()

if contra_df.empty:
    st.info(
        "No contradictions detected yet.  \n"
        "Run an audit from the **Home** page to populate this view."
    )
    if st.button("Go to Home to start an audit", type="primary"):
        st.switch_page("streamlit_app.py")
    st.stop()

# ── Severity metrics ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", len(contra_df))
c2.metric("🔴 High", int((contra_df.severity == "high").sum()), delta_color="inverse")
c3.metric("🟠 Medium", int((contra_df.severity == "medium").sum()), delta_color="inverse")
c4.metric("🟡 Low", int((contra_df.severity == "low").sum()))

# ── Sidebar filter ────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filters")
    sev_filter = st.multiselect(
        "Severity", ["high", "medium", "low"],
        default=["high", "medium", "low"],
    )
    types = sorted(contra_df["contradiction_type"].dropna().unique())
    type_filter = st.multiselect("Contradiction Type", types)

df = contra_df.copy()
if sev_filter:
    df = df[df["severity"].isin(sev_filter)]
if type_filter:
    df = df[df["contradiction_type"].isin(type_filter)]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_network, tab_detail, tab_table = st.tabs([
    "🕸️ Network View", "🔍 Detail View", "📄 Table",
])

with tab_network:
    if df.empty:
        st.info("No contradictions match the current filters.")
    else:
        st.plotly_chart(contradiction_network_chart(df, paper_df), use_container_width=True)
        st.plotly_chart(contradictions_by_metric_chart(df), use_container_width=True)

with tab_detail:
    if df.empty:
        st.info("No contradictions match the current filters.")
    else:
        _sev_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}
        _sev_style = {
            "high":   "background:#fdecea;color:#c62828;padding:2px 8px;border-radius:4px;font-weight:700",
            "medium": "background:#fff3e0;color:#e65100;padding:2px 8px;border-radius:4px;font-weight:700",
            "low":    "background:#e8f5e9;color:#2e7d32;padding:2px 8px;border-radius:4px;font-weight:600",
        }
        sorted_df = df.sort_values(
            "severity", key=lambda s: s.map({"high": 0, "medium": 1, "low": 2})
        )
        for _, row in sorted_df.iterrows():
            icon = _sev_icon.get(row["severity"], "⚪")
            with st.expander(
                f"{icon} {row['paper_a_id']} ↔ {row['paper_b_id']} — `{row['contradiction_type']}`",
                expanded=(row["severity"] == "high"),
            ):
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown(f"**Paper A:** `{row['paper_a_id']}`")
                    if not paper_df.empty:
                        pa = paper_df[paper_df["arxiv_id"] == row["paper_a_id"]]
                        if not pa.empty:
                            st.caption(pa["title"].values[0][:80])
                with d2:
                    st.markdown(f"**Paper B:** `{row['paper_b_id']}`")
                    if not paper_df.empty:
                        pb = paper_df[paper_df["arxiv_id"] == row["paper_b_id"]]
                        if not pb.empty:
                            st.caption(pb["title"].values[0][:80])
                st.markdown(f"**Type:** `{row['contradiction_type']}`")
                st.markdown(
                    f"**Severity:** <span style='{_sev_style.get(row['severity'], '')}'>"
                    f"{row['severity'].upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Description:** {row['description']}")

with tab_table:
    if df.empty:
        st.info("No contradictions match the current filters.")
    else:
        st.dataframe(
            df[["paper_a_id", "paper_b_id", "contradiction_type", "severity", "description"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "paper_a_id": st.column_config.TextColumn("Paper A", width="small"),
                "paper_b_id": st.column_config.TextColumn("Paper B", width="small"),
                "contradiction_type": st.column_config.TextColumn("Type"),
                "severity": st.column_config.TextColumn("Severity"),
                "description": st.column_config.TextColumn("Description", width="large"),
            },
        )
