# app/components/charts.py
#
# All Plotly chart helpers. Return go.Figure objects — never render here.

from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, annotations=[
        dict(text="No data", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    ])
    return fig


def claims_by_metric_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart: number of claims per metric."""
    if df.empty:
        return _empty_fig("Claims by Metric")
    counts = (
        df.groupby("metric").size().reset_index(name="count").sort_values("count", ascending=False)
    )
    return px.bar(
        counts, x="metric", y="count", title="Claims by Metric",
        color="count", color_continuous_scale="Blues",
    )


def claims_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: metric × dataset claim count."""
    if df.empty:
        return _empty_fig("Claims Density (Metric × Dataset)")
    pivot = df.groupby(["metric", "dataset"]).size().reset_index(name="count")
    return px.density_heatmap(
        pivot, x="dataset", y="metric", z="count",
        title="Claims Density (Metric × Dataset)", color_continuous_scale="Viridis",
    )


def reported_values_box(df: pd.DataFrame) -> go.Figure:
    """Box plot: reported value distribution per metric."""
    if df.empty:
        return _empty_fig("Reported Values by Metric")
    return px.box(df, x="metric", y="reported_value", title="Reported Values by Metric")


def severity_donut(gaps_df: pd.DataFrame) -> go.Figure:
    """Donut chart: gap severity distribution."""
    if gaps_df.empty:
        return _empty_fig("Gap Severity Distribution")
    counts = gaps_df["severity"].value_counts().reset_index()
    counts.columns = ["severity", "count"]
    colors = {"critical": "#e53935", "major": "#fb8c00", "minor": "#43a047"}
    return px.pie(
        counts, names="severity", values="count", hole=0.5,
        title="Gap Severity Distribution", color="severity", color_discrete_map=colors,
    )


def gaps_by_paper_bar(gaps_df: pd.DataFrame, papers_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: total gaps per paper coloured by severity."""
    if gaps_df.empty or papers_df.empty:
        return _empty_fig("Gaps per Paper")
    merged = gaps_df.merge(
        papers_df[["arxiv_id", "title"]], left_on="paper_id", right_on="arxiv_id"
    )
    counts = merged.groupby(["title", "severity"]).size().reset_index(name="count")
    return px.bar(
        counts, x="count", y="title", color="severity", orientation="h",
        title="Gaps per Paper",
        color_discrete_map={"critical": "#e53935", "major": "#fb8c00", "minor": "#43a047"},
    )


def gap_type_treemap(gaps_df: pd.DataFrame) -> go.Figure:
    """Treemap: gap type breakdown by severity."""
    if gaps_df.empty:
        return _empty_fig("Gap Type Breakdown")
    counts = gaps_df.groupby(["gap_type", "severity"]).size().reset_index(name="count")
    return px.treemap(
        counts, path=["gap_type", "severity"], values="count",
        title="Gap Type Breakdown",
    )


def contradiction_network_chart(contra_df: pd.DataFrame, papers_df: pd.DataFrame) -> go.Figure:
    """Force-directed network: nodes=papers, edges=contradictions coloured by severity."""
    if contra_df.empty:
        return _empty_fig("Contradiction Network")
    G = nx.Graph()
    title_map = (
        dict(zip(papers_df["arxiv_id"], papers_df["title"].str[:40]))
        if not papers_df.empty
        else {}
    )
    for _, row in contra_df.iterrows():
        G.add_node(row["paper_a_id"], label=title_map.get(row["paper_a_id"], row["paper_a_id"]))
        G.add_node(row["paper_b_id"], label=title_map.get(row["paper_b_id"], row["paper_b_id"]))
        G.add_edge(
            row["paper_a_id"], row["paper_b_id"],
            severity=row["severity"], ctype=row["contradiction_type"],
        )

    pos = nx.spring_layout(G, seed=42)
    edge_traces, node_trace = _build_network_traces(G, pos, contra_df)
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title="Contradiction Network", showlegend=True, hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


def contradictions_by_metric_chart(contra_df: pd.DataFrame) -> go.Figure:
    """Bar chart: contradictions by type."""
    if contra_df.empty:
        return _empty_fig("Contradictions by Type")
    counts = (
        contra_df.groupby("contradiction_type").size().reset_index(name="count")
    )
    return px.bar(
        counts, x="contradiction_type", y="count", title="Contradictions by Type",
        color="count", color_continuous_scale="Reds",
    )


def pipeline_status_chart(papers_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: paper count by pipeline status."""
    if papers_df.empty:
        return _empty_fig("Pipeline Status")
    counts = papers_df.groupby("status").size().reset_index(name="count")
    return px.bar(
        counts, x="count", y="status", orientation="h", title="Pipeline Status",
        color="count", color_continuous_scale="Teal",
    )


def claims_violin_chart(df: pd.DataFrame) -> go.Figure:
    """Violin plot: reported value distribution per metric (numeric claims only)."""
    if df.empty:
        return _empty_fig("Value Distribution by Metric")
    numeric = df.copy()
    numeric["value_num"] = pd.to_numeric(numeric["reported_value"], errors="coerce")
    numeric = numeric.dropna(subset=["value_num"])
    if numeric.empty:
        return _empty_fig("Value Distribution by Metric")
    return px.violin(
        numeric, x="metric", y="value_num", box=True, points="outliers",
        title="Value Distribution by Metric",
        labels={"value_num": "Reported Value", "metric": "Metric"},
    )


def papers_gap_heatmap(papers_df: pd.DataFrame, gaps_df: pd.DataFrame) -> go.Figure:
    """Heatmap: paper × gap_type occurrence count."""
    if papers_df.empty or gaps_df.empty:
        return _empty_fig("Gap Type × Paper Heatmap")
    merged = gaps_df.merge(
        papers_df[["arxiv_id", "title"]], left_on="paper_id", right_on="arxiv_id"
    )
    merged["short_title"] = merged["title"].str[:35]
    pivot = merged.groupby(["short_title", "gap_type"]).size().reset_index(name="count")
    return px.density_heatmap(
        pivot, x="gap_type", y="short_title", z="count",
        title="Gap Type × Paper Heatmap", color_continuous_scale="OrRd",
        labels={"gap_type": "Gap Type", "short_title": "Paper", "count": "Count"},
    )


def _build_network_traces(G, pos, contra_df):
    color_map = {"high": "#e53935", "medium": "#fb8c00", "low": "#43a047"}
    sev_edge: dict[str, tuple] = {"high": ([], []), "medium": ([], []), "low": ([], [])}
    for u, v, data in G.edges(data=True):
        s = data.get("severity", "low")
        if s not in sev_edge:
            s = "low"
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        sev_edge[s][0].extend([x0, x1, None])
        sev_edge[s][1].extend([y0, y1, None])

    edge_traces = [
        go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color_map[s], width=2),
            hoverinfo="none", name=f"{s.capitalize()} contradiction",
        )
        for s, (xs, ys) in sev_edge.items() if xs
    ]
    nx_vals = [pos[n] for n in G.nodes()]
    node_trace = go.Scatter(
        x=[p[0] for p in nx_vals], y=[p[1] for p in nx_vals],
        mode="markers+text",
        text=[G.nodes[n].get("label", n) for n in G.nodes()],
        textposition="top center",
        marker=dict(size=14, color="#1565c0", line=dict(width=2, color="white")),
        hoverinfo="text", name="Papers",
    )
    return edge_traces, node_trace
