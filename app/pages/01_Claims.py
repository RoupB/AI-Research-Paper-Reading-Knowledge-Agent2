# app/pages/01_Claims.py

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.components.charts import (  # noqa: E402
    claims_by_metric_chart,
    claims_heatmap,
    claims_violin_chart,
    reported_values_box,
)
from app.components.db_reader import get_all_claims_df  # noqa: E402
from app.components.pipeline_state import is_running  # noqa: E402

st.set_page_config(page_title="Claims — ClaimCheck", layout="wide")
st.header("📋 Benchmark Claims")

# ── Pipeline-in-progress banner ───────────────────────────────────────────────
_sid = st.session_state.get("audit_sid", "")
if _sid and is_running(_sid):
    st.info(
        "⏳ **Audit is running** — claims will appear here once Claim Extraction "
        "(step 3) completes. This page will reflect data already written to the DB.",
        icon="⏳",
    )

df_all = get_all_claims_df()

if df_all.empty:
    st.info(
        "No claims in the database yet.  \n"
        "Run an audit from the **Home** page to populate this view."
    )
    if st.button("Go to Home to start an audit", type="primary"):
        st.switch_page("streamlit_app.py")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filters")
    search_text = st.text_input("Search description / section", placeholder="e.g. accuracy GLUE")
    metrics = st.multiselect("Metric", sorted(df_all["metric"].dropna().unique()))
    datasets = st.multiselect("Dataset", sorted(df_all["dataset"].dropna().unique()))
    models = st.multiselect("Base model", sorted(df_all["model_base"].dropna().unique()))
    papers = st.multiselect("Paper (arXiv ID)", sorted(df_all["paper_id"].dropna().unique()))
    cond_only = st.checkbox("Conditional claims only")

df = df_all.copy()
if search_text:
    mask = (
        df["source_section"].str.contains(search_text, case=False, na=False)
        | df["raw_text"].str.contains(search_text, case=False, na=False)
    )
    df = df[mask]
if metrics:
    df = df[df["metric"].isin(metrics)]
if datasets:
    df = df[df["dataset"].isin(datasets)]
if models:
    df = df[df["model_base"].isin(models)]
if papers:
    df = df[df["paper_id"].isin(papers)]
if cond_only:
    df = df[df["is_conditional"]]

# ── Summary strip ─────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Showing", len(df))
m2.metric("Total", len(df_all))
m3.metric("Conditional", int(df["is_conditional"].sum()))
m4.metric("Papers", df["paper_id"].nunique())

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_charts, tab_cond, tab_table, tab_add = st.tabs([
    "📊 Charts", "⚠️ Conditional Claims", "📄 All Claims", "➕ Add Manual Claim",
])

with tab_charts:
    if df.empty:
        st.info("No claims match the current filters.")
    else:
        row1_c1, row1_c2 = st.columns(2)
        row1_c1.plotly_chart(claims_by_metric_chart(df), use_container_width=True)
        row1_c2.plotly_chart(claims_heatmap(df), use_container_width=True)
        row2_c1, row2_c2 = st.columns(2)
        row2_c1.plotly_chart(claims_violin_chart(df), use_container_width=True)
        row2_c2.plotly_chart(reported_values_box(df), use_container_width=True)

with tab_cond:
    cond_df = df[df["is_conditional"]]
    if cond_df.empty:
        st.info("No conditional claims match the current filters.")
    else:
        st.caption(
            f"{len(cond_df)} claims are conditional — results only valid under "
            "specific hyperparameter/dataset conditions."
        )
        for paper_id, group in cond_df.groupby("paper_id"):
            with st.expander(f"**{paper_id}** — {len(group)} conditional claims"):
                for _, row in group.iterrows():
                    cond_str = (
                        ", ".join(f"`{k}={v}`" for k, v in row["conditions"].items())
                        if isinstance(row["conditions"], dict)
                        else str(row["conditions"])
                    )
                    st.markdown(
                        f"- **{row['metric']}** on *{row['dataset']}* "
                        f"({row['model_base']}): **{row['reported_value']}** {row.get('unit') or ''}"
                        f"  \n  Conditions: {cond_str or '—'}"
                    )

with tab_table:
    if df.empty:
        st.info("No claims match the current filters.")
    else:
        display_cols = [
            "paper_id", "metric", "dataset", "model_base",
            "reported_value", "is_conditional", "source_section",
        ]
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            height=450,
            hide_index=True,
            column_config={
                "paper_id": st.column_config.TextColumn("ArXiv ID", width="small"),
                "metric": st.column_config.TextColumn("Metric"),
                "dataset": st.column_config.TextColumn("Dataset"),
                "model_base": st.column_config.TextColumn("Base Model"),
                "reported_value": st.column_config.NumberColumn("Value", format="%.4f"),
                "is_conditional": st.column_config.CheckboxColumn("Conditional"),
                "source_section": st.column_config.TextColumn("Section"),
            },
        )

with tab_add:
    with st.form("add_claim"):
        c1, c2, c3 = st.columns(3)
        paper_id = c1.text_input("ArXiv ID")
        metric = c2.text_input("Metric (e.g. accuracy)")
        dataset = c3.text_input("Dataset (e.g. GLUE/MNLI)")
        c4, c5 = st.columns(2)
        model = c4.text_input("Base Model")
        value = c5.number_input("Reported Value", format="%.4f")
        conditions_raw = st.text_input(
            'Conditions (JSON, e.g. {"rank": "8"})', value="{}"
        )
        is_conditional = st.checkbox("Is conditional")
        source_section = st.text_input("Source section", value="manual")
        raw_text = st.text_area("Raw text from paper (verbatim)")
        save = st.form_submit_button("Save Claim", type="primary")

        if save and paper_id and metric:
            from db import queries
            from models import BenchmarkClaim

            try:
                conditions = json.loads(conditions_raw or "{}")
            except json.JSONDecodeError:
                conditions = {}
            claim = BenchmarkClaim(
                paper_id=paper_id,
                claim_id=str(uuid.uuid4()),
                metric=metric,
                dataset=dataset or "unknown",
                model_base=model or "unknown",
                reported_value=float(value),
                conditions={str(k): str(v) for k, v in conditions.items()},
                is_conditional=is_conditional,
                source_section=source_section,
                raw_text=raw_text,
            )
            asyncio.run(queries.insert_claim(claim))
            st.cache_data.clear()
            st.success(f"Claim saved for {paper_id}")
