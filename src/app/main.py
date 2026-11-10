"""Frontend contribution: Streamlit dashboard for arXiv keyword analytics.

Launch from the repository root:
    streamlit run src/app/main.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.data_access import (  # noqa: E402
    DashboardDataError,
    DashboardFilters,
    connect_read_only,
    load_abstract_lengths,
    load_filter_options,
    load_keyword_cooccurrence,
    load_keyword_trends,
    load_publication_trends,
    load_top_keywords,
)
from src.viz.charts import (  # noqa: E402
    abstract_length_distribution,
    keyword_cooccurrence,
    keyword_trends,
    publications_by_year,
    top_keywords,
)


DEFAULT_DB_PATHS = (
    "data/arxiv.db",
    "data/arxiv.sqlite",
    "data/arxiv.sqlite3",
    "data/processed/arxiv.db",
    "data/database.db",
)


def default_database_path() -> str:
    """Find the team's SQLite database; allow an environment override."""
    configured = os.environ.get("ARXIV_DB_PATH")
    if configured:
        return configured
    for candidate in DEFAULT_DB_PATHS:
        if (ROOT / candidate).exists():
            return str(ROOT / candidate)
    return str(ROOT / DEFAULT_DB_PATHS[0])


def database_fingerprint(path: str) -> int:
    """Invalidate cached SQL aggregates after an ETL run updates the database."""
    try:
        return Path(path).expanduser().stat().st_mtime_ns
    except OSError:
        return 0


@st.cache_data(ttl=300, show_spinner=False)
def get_options(path: str, fingerprint: int) -> tuple[tuple[int, int], list[str], list[str]]:
    with closing(connect_read_only(path)) as connection:
        return load_filter_options(connection)


@st.cache_data(ttl=300, show_spinner=False)
def get_chart_data(path: str, fingerprint: int, kind: str, filters: DashboardFilters):
    """Cache small aggregated queries, never the complete 2M+ paper corpus."""
    with closing(connect_read_only(path)) as connection:
        if kind == "publications":
            return load_publication_trends(connection, filters)
        if kind == "keywords":
            return load_top_keywords(connection, filters)
        if kind == "lengths":
            return load_abstract_lengths(connection, filters)
        raise ValueError(f"Unknown dataset kind: {kind}")


@st.cache_data(ttl=300, show_spinner=False)
def get_keyword_detail_data(
    path: str, fingerprint: int, kind: str, filters: DashboardFilters, terms: tuple[str, ...]
) -> pd.DataFrame:
    with closing(connect_read_only(path)) as connection:
        if kind == "cooccurrence":
            return load_keyword_cooccurrence(connection, filters, list(terms))
        if kind == "trends":
            return load_keyword_trends(connection, filters, list(terms))
        raise ValueError(f"Unknown keyword dataset kind: {kind}")


def draw_chart(label: str, data: pd.DataFrame, chart_builder, *, help_text: str = "") -> None:
    st.subheader(label)
    if help_text:
        st.caption(help_text)
    if data.empty:
        st.info("No records found for the selected filters.")
        return
    st.plotly_chart(chart_builder(data), use_container_width=True)


def load_or_warn(load_function, *args):
    """Let one unfinished DB component fail without hiding other charts."""
    try:
        return load_function(*args)
    except (DashboardDataError, sqlite3.Error, ValueError) as error:
        st.info(str(error))
        return pd.DataFrame()


def render_analytics(path: str) -> None:
    st.markdown("#### Explore the arXiv corpus")
    st.caption("Interactive SQL aggregates. Filters affect every chart below.")
    fingerprint = database_fingerprint(path)

    try:
        (min_year, max_year), categories, methods = get_options(path, fingerprint)
    except (DashboardDataError, sqlite3.Error) as error:
        st.warning(str(error))
        st.info(
            "Once your teammate loads the database, set its path in the sidebar. "
            "Expected tables: `papers`, `categories`, `paper_categories`, `paper_keywords`. "
            "The NLP tab can be tested independently."
        )
        return

    with st.sidebar:
        st.header("Analytics filters")
        if min_year == max_year:
            st.caption(f"Publication year: {min_year} (only year in the database)")
            years = (min_year, max_year)
        else:
            years = st.slider(
                "Publication years",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
            )
        selected_categories = st.multiselect(
            "arXiv categories", options=categories, default=[], placeholder="All categories"
        )
        selected_method = st.selectbox(
            "Keyword extraction method",
            options=["All methods", *methods],
            help="Restrict keyword charts to one extraction algorithm when method data exists.",
        )
        if st.button("Refresh database", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    filters = DashboardFilters(
        start_year=years[0],
        end_year=years[1],
        categories=tuple(selected_categories),
        keyword_method=None if selected_method == "All methods" else selected_method,
    )

    publications = load_or_warn(get_chart_data, path, fingerprint, "publications", filters)
    if not publications.empty:
        # Category memberships may overlap; don't misrepresent their sum as unique papers.
        st.metric("Publication/category records", f"{publications['papers'].sum():,}")
    draw_chart("1. Publications by year and category", publications, publications_by_year)

    st.divider()
    top_terms = load_or_warn(get_chart_data, path, fingerprint, "keywords", filters)
    left, right = st.columns(2, gap="large")
    with left:
        draw_chart("2. Top 20 extracted terms", top_terms, top_keywords)
    with right:
        terms = tuple(top_terms["keyword"].head(12).tolist()) if not top_terms.empty else ()
        pairs = (
            load_or_warn(get_keyword_detail_data, path, fingerprint, "cooccurrence", filters, terms)
            if terms
            else pd.DataFrame()
        )
        st.subheader("3. Term co-occurrence")
        st.caption("Number of papers containing both terms; first 4,000 matching paper IDs.")
        if terms:
            st.plotly_chart(keyword_cooccurrence(pairs, list(terms)), use_container_width=True)
        else:
            st.info("Keyword data is needed to draw this heatmap.")

    st.divider()
    lengths = load_or_warn(get_chart_data, path, fingerprint, "lengths", filters)
    top_five = tuple(top_terms["keyword"].head(5).tolist()) if not top_terms.empty else ()
    trends = (
        load_or_warn(get_keyword_detail_data, path, fingerprint, "trends", filters, top_five)
        if top_five
        else pd.DataFrame()
    )
    lower_left, lower_right = st.columns(2, gap="large")
    with lower_left:
        draw_chart(
            "4. Abstract lengths",
            lengths,
            abstract_length_distribution,
            help_text="Words per abstract; first 5,000 matching paper IDs (bounded sample).",
        )
    with lower_right:
        draw_chart(
            "5. Keyword evolution",
            trends,
            keyword_trends,
            help_text="Annual paper counts for the five most frequent extracted terms.",
        )

    if not top_terms.empty:
        st.download_button(
            "Download top terms as CSV",
            data=top_terms.to_csv(index=False).encode("utf-8"),
            file_name="arxiv_top_keywords.csv",
            mime="text/csv",
        )


def render_inference() -> None:
    st.markdown("#### Extract keywords from your abstract")
    st.caption("This calls the team's shared `src.nlp.interface.extract_keywords()` function.")

    with st.form("keyword_inference"):
        abstract = st.text_area(
            "Abstract text",
            placeholder="Paste a research abstract in English…",
            height=180,
            max_chars=20000,
        )
        method_col, preprocess_col, count_col = st.columns([1, 1, 1])
        with method_col:
            method = st.selectbox(
                "Algorithm", ["tfidf", "yake", "frequency_baseline"],
                format_func=lambda value: {
                    "tfidf": "TF-IDF", "yake": "YAKE", "frequency_baseline": "Frequency baseline"
                }[value],
            )
        with preprocess_col:
            preprocess = st.selectbox("Preprocessing", ["lemmatization", "stemming", "none"])
        with count_col:
            top_k = st.slider("Number of keywords", 3, 20, 10)
        submitted = st.form_submit_button("Extract keywords", type="primary", use_container_width=True)

    if submitted:
        if not abstract.strip():
            st.warning("Please enter an abstract before running the extractor.")
        else:
            try:
                from src.nlp.interface import extract_keywords

                with st.spinner("Extracting keywords…"):
                    result = extract_keywords(
                        text=abstract.strip(), method=method, top_k=top_k,
                        preprocess_mode=preprocess,
                    )
                st.session_state["last_extraction"] = (
                    result.to_dict() if hasattr(result, "to_dict") else result
                )
            except NotImplementedError:
                st.warning(
                    "The NLP extraction function has not been implemented on this branch yet. "
                    "The inference UI is ready to use when the NLP branch is merged."
                )
                st.session_state.pop("last_extraction", None)
            except Exception as error:
                st.error(f"Keyword extraction failed: {error}")
                st.session_state.pop("last_extraction", None)

    result = st.session_state.get("last_extraction")
    if not result:
        return
    st.success("Extraction completed")
    metrics = result.get("metrics") or {}
    m1, m2, m3 = st.columns(3)
    m1.metric("Execution time", f"{float(metrics.get('execution_time_ms', 0)):.1f} ms")
    m2.metric("Peak memory", f"{float(metrics.get('peak_memory_mb', 0)):.2f} MB")
    m3.metric("Tokens processed", f"{int(metrics.get('total_tokens_processed', 0)):,}")

    frame = pd.DataFrame(result.get("keywords") or [])
    if not frame.empty and {"keyword", "score"}.issubset(frame.columns):
        frame = frame[["keyword", "score"]].sort_values("score", ascending=False)
        st.subheader("Extracted keywords")
        st.dataframe(frame, hide_index=True, use_container_width=True)
    else:
        st.info("The extractor returned no keywords.")


def main() -> None:
    st.set_page_config(page_title="arXiv Topic Explorer", page_icon="🔎", layout="wide")
    st.markdown(
        """
        <style>
          .block-container {max-width: 1500px; padding-top: 1.7rem;}
          h1, h2, h3 {letter-spacing: -.025em;}
          [data-testid="stMetric"] {background: #f1f5f9; border-radius: 12px; padding: 12px 16px;}
          [data-testid="stMetricLabel"] {color: #475569;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("arXiv Topic Explorer")
    st.caption("Keyword extraction • Topic evolution • SQL analytics")
    with st.sidebar:
        st.title("Research dashboard")
        st.caption("AAI-2605M · Programming for Data Analysis and Database")
        database_path = st.text_input(
            "SQLite database path", value=default_database_path(),
            help="Created by the data-engineering pipeline; the dashboard opens it read-only.",
        )
        st.divider()
    analytics, inference, about = st.tabs(["📊 Analytics", "🧠 Inference", "ℹ️ About"])
    with analytics:
        render_analytics(database_path)
    with inference:
        render_inference()
    with about:
        st.markdown(
            """
            **Data source:** [Cornell arXiv metadata on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)

            **Team integration**
            - ETL/SQLite: `src/etl/` and `src/db/`
            - NLP extraction: `src/nlp/interface.py`
            - Frontend and figures: `src/app/` and `src/viz/`

            Data figures use SQLite results. The inference page calls the shared Python
            interface. No sample results are presented as real research findings.
            """
        )


if __name__ == "__main__":
    main()
