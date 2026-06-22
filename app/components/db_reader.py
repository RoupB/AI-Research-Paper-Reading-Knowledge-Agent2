# app/components/db_reader.py
#
# Read-only sync wrappers around the DB for Streamlit (Streamlit runs synchronously).
# All functions are cached with a 60-second TTL; call st.cache_data.clear() after
# a pipeline run to surface fresh data immediately.

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import streamlit as st

from config import settings


@st.cache_resource
def _get_connection_factory():
    """Returns a factory fn; cached so the import is only done once."""
    return sqlite3.connect


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=30)
def get_audit_stats() -> dict:
    with _conn() as conn:
        papers_total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        claims_total = conn.execute("SELECT COUNT(*) FROM benchmark_claims").fetchone()[0]
        gaps_total = conn.execute("SELECT COUNT(*) FROM reproducibility_gaps").fetchone()[0]
        contras = conn.execute("SELECT COUNT(*) FROM contradictions").fetchone()[0]
        gaps_sev = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT severity, COUNT(*) FROM reproducibility_gaps GROUP BY severity"
            )
        }
    return {
        "papers_total": papers_total,
        "claims_total": claims_total,
        "gaps_total": gaps_total,
        "contradictions_total": contras,
        "gaps_by_severity": gaps_sev,
    }


@st.cache_data(ttl=30)
def get_all_claims_df() -> pd.DataFrame:
    with _conn() as conn:
        df = pd.read_sql("SELECT * FROM benchmark_claims", conn)
    if df.empty:
        return df
    df["conditions"] = df["conditions"].apply(lambda x: json.loads(x) if x else {})
    df["is_conditional"] = df["is_conditional"].astype(bool)
    df["arxiv_id"] = df["paper_id"]
    return df


@st.cache_data(ttl=30)
def get_gaps_df() -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql("SELECT * FROM reproducibility_gaps", conn)


@st.cache_data(ttl=30)
def get_papers_df() -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql("SELECT * FROM papers", conn)


@st.cache_data(ttl=30)
def get_contradictions_df() -> pd.DataFrame:
    with _conn() as conn:
        return pd.read_sql("SELECT * FROM contradictions", conn)


@st.cache_data(ttl=30)
def get_papers_with_gap_counts() -> pd.DataFrame:
    with _conn() as conn:
        df = pd.read_sql(
            """
            SELECT p.arxiv_id, p.title, p.lora_variant_tag, p.status, p.repo_url,
                   COUNT(g.gap_id) AS gap_count,
                   SUM(CASE WHEN g.severity='critical' THEN 1 ELSE 0 END) AS critical_count,
                   SUM(CASE WHEN g.severity='major'    THEN 1 ELSE 0 END) AS major_count,
                   SUM(CASE WHEN g.severity='minor'    THEN 1 ELSE 0 END) AS minor_count
            FROM papers p
            LEFT JOIN reproducibility_gaps g ON p.arxiv_id = g.paper_id
            GROUP BY p.arxiv_id
            ORDER BY critical_count DESC, major_count DESC
            """,
            conn,
        )
    return df


@st.cache_data(ttl=30)
def get_gaps_with_paper_titles() -> pd.DataFrame:
    with _conn() as conn:
        df = pd.read_sql(
            """
            SELECT g.*, p.title AS paper_title, p.lora_variant_tag
            FROM reproducibility_gaps g
            LEFT JOIN papers p ON g.paper_id = p.arxiv_id
            """,
            conn,
        )
    return df
