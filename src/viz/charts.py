"""Plotly figures for the arXiv trends Streamlit dashboard.

These functions do no database access, making them easy to test and reuse.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PRIMARY = "#2563eb"
COLORS = ["#2563eb", "#0d9488", "#ea580c", "#8b5cf6", "#d946ef", "#475569"]


def _finish(figure: go.Figure, height: int = 380) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=36, b=18),
        font=dict(family="Arial, sans-serif", color="#334155"),
        hovermode="closest",
        legend_title_text="",
    )
    return figure


def publications_by_year(frame: pd.DataFrame) -> go.Figure:
    """Year-by-year publication counts, broken down by arXiv category."""
    figure = px.line(
        frame.sort_values("year"),
        x="year",
        y="papers",
        color="category",
        markers=True,
        color_discrete_sequence=COLORS,
        labels={"year": "Publication year", "papers": "Papers", "category": "Category"},
    )
    figure.update_traces(hovertemplate="Year %{x}<br>Papers %{y:,}<extra>%{fullData.name}</extra>")
    figure.update_xaxes(dtick=1 if len(frame["year"].unique()) <= 12 else None)
    return _finish(figure, height=400)


def top_keywords(frame: pd.DataFrame) -> go.Figure:
    """Rank up to 20 concepts by the number of distinct papers mentioning them."""
    frame = frame.sort_values(["paper_count", "keyword"], ascending=[True, False])
    figure = px.bar(
        frame,
        x="paper_count",
        y="keyword",
        orientation="h",
        labels={"paper_count": "Papers with keyword", "keyword": "Keyword"},
        color_discrete_sequence=[PRIMARY],
        text="paper_count",
    )
    figure.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
    return _finish(figure, height=max(350, 32 * len(frame) + 90))


def keyword_cooccurrence(frame: pd.DataFrame, keywords: list[str]) -> go.Figure:
    """Symmetric keyword pair counts (diagonal = single-keyword paper count)."""
    ordered = list(dict.fromkeys(keywords))[:12]
    index = {word: i for i, word in enumerate(ordered)}
    matrix = [[0 for _ in ordered] for _ in ordered]
    for row in frame.itertuples(index=False):
        if row.keyword_a in index and row.keyword_b in index:
            i, j = index[row.keyword_a], index[row.keyword_b]
            matrix[i][j] = int(row.papers)
            matrix[j][i] = int(row.papers)

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=ordered,
            y=ordered,
            colorscale=[[0, "#eff6ff"], [0.5, "#60a5fa"], [1, "#1e3a8a"]],
            colorbar=dict(title="Papers"),
            hovertemplate="%{y} + %{x}<br>%{z:,} papers<extra></extra>",
        )
    )
    figure.update_xaxes(side="bottom", tickangle=-35)
    return _finish(figure, height=max(360, 36 * len(ordered) + 170))


def abstract_length_distribution(frame: pd.DataFrame) -> go.Figure:
    """Distribution of abstract lengths in words, with a marginal boxplot."""
    figure = px.histogram(
        frame,
        x="word_count",
        nbins=40,
        marginal="box",
        color_discrete_sequence=["#0d9488"],
        labels={"word_count": "Words per abstract", "count": "Papers"},
    )
    figure.update_yaxes(title_text="Papers", row=2, col=1)
    return _finish(figure, height=400)


def keyword_trends(frame: pd.DataFrame) -> go.Figure:
    """Track up to five frequent extracted keywords over publication years."""
    figure = px.line(
        frame.sort_values("year"),
        x="year",
        y="papers",
        color="keyword",
        markers=True,
        color_discrete_sequence=COLORS,
        labels={"year": "Publication year", "papers": "Papers", "keyword": "Keyword"},
    )
    return _finish(figure, height=400)
