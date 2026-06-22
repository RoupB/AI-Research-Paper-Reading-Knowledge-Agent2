# app/pages/04_Report.py

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from app.components.db_reader import (  # noqa: E402
    get_audit_stats,
    get_contradictions_df,
    get_gaps_df,
    get_papers_df,
    get_papers_with_gap_counts,
)
from app.components.pipeline_state import get_progress, is_running  # noqa: E402
from config import settings  # noqa: E402

st.set_page_config(page_title="Report — ClaimCheck", layout="wide")
st.header("📊 Audit Report")

# ── Pipeline-in-progress banner ───────────────────────────────────────────────
_sid = st.session_state.get("audit_sid", "")
_running = _sid and is_running(_sid)
if _running:
    st.info(
        "⏳ **Audit is running** — the report will be available here once "
        "Report Generation (step 7) completes.",
        icon="⏳",
    )

# ── Determine report location ─────────────────────────────────────────────────
report_dir = Path(settings.report_output_dir)

# Use report_path from pipeline result if available and more specific
_progress = get_progress(_sid) if _sid else {}
_result_path = _progress.get("result")
if _result_path:
    report_dir = Path(_result_path).parent

md_path = report_dir / "audit_report.md"
html_path = report_dir / "audit_report.html"
report_exists = md_path.exists()

# ── Quick download bar ────────────────────────────────────────────────────────
if report_exists:
    dl_c1, dl_c2, dl_c3 = st.columns([1, 1, 3])
    dl_c1.download_button(
        label="⬇️ Download Markdown",
        data=md_path.read_text(encoding="utf-8"),
        file_name="audit_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if html_path.exists():
        dl_c2.download_button(
            label="⬇️ Download HTML",
            data=html_path.read_text(encoding="utf-8"),
            file_name="audit_report.html",
            mime="text/html",
            use_container_width=True,
        )
    dl_c3.caption(f"Report location: `{report_dir}`")
    st.markdown("---")
elif not _running:
    st.info(
        "No report generated yet.  \n"
        "Run an audit from the **Home** page or via `python main.py run`."
    )
    if st.button("Go to Home to start an audit", type="primary"):
        st.switch_page("streamlit_app.py")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_preview, tab_gaps, tab_contra, tab_stats, tab_dl = st.tabs([
    "🌐 HTML Preview", "🔬 Gap Analysis", "⚡ Contradictions", "📈 Statistics", "⬇️ Download",
])

with tab_preview:
    if html_path.exists():
        st.caption("Styled HTML report rendered inline.")
        components.html(html_path.read_text(encoding="utf-8"), height=820, scrolling=True)
    elif report_exists:
        st.markdown(md_path.read_text(encoding="utf-8"))
    else:
        st.info("No report yet. Run an audit from the **Home** page.")

with tab_gaps:
    gaps_df = get_gaps_df()
    papers_df = get_papers_df()

    if gaps_df.empty:
        st.info("No gap data in database yet.")
    else:
        _sev_order = {"critical": 0, "major": 1, "minor": 2}
        show_only = st.radio("Show", ["All", "Critical only", "Major+"], horizontal=True)
        df = gaps_df.copy()
        if show_only == "Critical only":
            df = df[df["severity"] == "critical"]
        elif show_only == "Major+":
            df = df[df["severity"].isin(["critical", "major"])]

        g1, g2, g3 = st.columns(3)
        g1.metric("🔴 Critical", int((gaps_df.severity == "critical").sum()))
        g2.metric("🟠 Major", int((gaps_df.severity == "major").sum()))
        g3.metric("🟡 Minor", int((gaps_df.severity == "minor").sum()))

        for pid in sorted(df["paper_id"].dropna().unique()):
            pgaps = df[df["paper_id"] == pid].copy()
            pgaps["_ord"] = pgaps["severity"].map(_sev_order)
            pgaps = pgaps.sort_values("_ord").drop(columns=["_ord"])

            crit = int((pgaps.severity == "critical").sum())
            maj = int((pgaps.severity == "major").sum())
            minor_cnt = int((pgaps.severity == "minor").sum())

            paper_row = papers_df[papers_df["arxiv_id"] == pid] if not papers_df.empty else None
            title = (
                paper_row["title"].values[0][:65]
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

                st.dataframe(
                    pgaps[["gap_type", "severity", "description", "paper_value", "code_value"]]
                    .style.map(_sev_style, subset=["severity"]),
                    use_container_width=True,
                    hide_index=True,
                )

with tab_contra:
    contra_df = get_contradictions_df()
    if contra_df.empty:
        st.info("No contradictions detected.")
    else:
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("🔴 High", int((contra_df.severity == "high").sum()))
        cc2.metric("🟠 Medium", int((contra_df.severity == "medium").sum()))
        cc3.metric("🟡 Low", int((contra_df.severity == "low").sum()))

        papers_df2 = get_papers_df()
        _sev_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}
        for _, row in contra_df.sort_values(
            "severity", key=lambda s: s.map({"high": 0, "medium": 1, "low": 2})
        ).iterrows():
            icon = _sev_icon.get(row["severity"], "⚪")
            with st.expander(
                f"{icon} {row['paper_a_id']} ↔ {row['paper_b_id']} — `{row['contradiction_type']}`",
                expanded=(row["severity"] == "high"),
            ):
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown(f"**Paper A:** `{row['paper_a_id']}`")
                    if not papers_df2.empty:
                        pa = papers_df2[papers_df2["arxiv_id"] == row["paper_a_id"]]
                        if not pa.empty:
                            st.caption(pa["title"].values[0][:80])
                with d2:
                    st.markdown(f"**Paper B:** `{row['paper_b_id']}`")
                    if not papers_df2.empty:
                        pb = papers_df2[papers_df2["arxiv_id"] == row["paper_b_id"]]
                        if not pb.empty:
                            st.caption(pb["title"].values[0][:80])
                st.markdown(f"**Description:** {row['description']}")

with tab_stats:
    try:
        stats = get_audit_stats()
        pwg = get_papers_with_gap_counts()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load stats: {exc}")
        st.stop()

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Papers Audited", stats["papers_total"])
    r1c2.metric("Claims Extracted", stats["claims_total"])
    r1c3.metric("Total Gaps", stats["gaps_total"])
    r1c4.metric("Contradictions", stats["contradictions_total"])

    if not pwg.empty:
        st.subheader("Per-Paper Breakdown")
        display = pwg[[
            "arxiv_id", "title", "lora_variant_tag", "status",
            "gap_count", "critical_count", "major_count", "minor_count",
        ]].copy()
        display.columns = [
            "ArXiv ID", "Title", "Variant", "Status",
            "Total Gaps", "Critical", "Major", "Minor",
        ]
        display["Title"] = display["Title"].str[:50]
        st.dataframe(display, use_container_width=True, hide_index=True)

with tab_dl:
    if not report_exists:
        st.info("No report generated yet. Run an audit from the **Home** page.")
    else:
        st.subheader("Download Report Files")
        col1, col2 = st.columns(2)
        col1.download_button(
            label="⬇️ Download Markdown (.md)",
            data=md_path.read_text(encoding="utf-8"),
            file_name="audit_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if html_path.exists():
            col2.download_button(
                label="⬇️ Download HTML (.html)",
                data=html_path.read_text(encoding="utf-8"),
                file_name="audit_report.html",
                mime="text/html",
                use_container_width=True,
            )
        st.markdown("---")
        st.subheader("Report Location")
        st.code(str(report_dir), language=None)
