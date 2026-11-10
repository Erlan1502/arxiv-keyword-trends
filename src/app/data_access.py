"""Read-only, bounded SQLite queries used by the Streamlit dashboard.

The data-engineering branch owns the database. This adapter supports its planned
normalized schema and a few common naming variations without changing it.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class DashboardDataError(Exception):
    """An expected database/schema/integration problem for the UI to display."""


@dataclass(frozen=True)
class DashboardFilters:
    start_year: int
    end_year: int
    categories: tuple[str, ...] = ()
    keyword_method: str | None = None


@dataclass(frozen=True)
class DashboardSchema:
    paper_id: str
    year_expression: str
    abstract_column: str | None
    category_join: str
    category_expression: str
    keyword_join: str | None
    keyword_expression: str | None
    keyword_method_column: str | None


def connect_read_only(path: str | Path) -> sqlite3.Connection:
    """Open an existing database without accidentally creating an empty file."""
    database = Path(path).expanduser().resolve()
    if not database.is_file():
        raise DashboardDataError(f"SQLite database not found: {database}")
    try:
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=10)
        connection.execute("PRAGMA query_only = ON")
        return connection
    except sqlite3.Error as error:
        raise DashboardDataError(f"Cannot open SQLite database: {error}") from error


def _identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise DashboardDataError(f"Unsafe SQL identifier in schema: {name!r}")
    return f'"{name}"'


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({_identifier(table)})")}


def _first(columns: set[str], *candidates: str) -> str | None:
    return next((name for name in candidates if name in columns), None)


def inspect_schema(connection: sqlite3.Connection) -> DashboardSchema:
    """Discover the shared schema and report missing parts explicitly."""
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "papers" not in tables:
        raise DashboardDataError("The 'papers' table is missing. Run the ETL/database pipeline first.")

    paper_columns = _columns(connection, "papers")
    paper_id = _first(paper_columns, "id", "paper_id", "arxiv_id")
    year_column = _first(paper_columns, "year", "published_year")
    date_column = _first(paper_columns, "published", "published_date", "update_date")
    abstract_column = _first(paper_columns, "abstract", "summary")
    if not paper_id or not (year_column or date_column):
        raise DashboardDataError(
            "The 'papers' table needs id/paper_id and year/published/published_date columns."
        )
    year_expression = (
        f'CAST(p.{_identifier(year_column)} AS INTEGER)'
        if year_column
        else f'CAST(SUBSTR(p.{_identifier(date_column)}, 1, 4) AS INTEGER)'
    )

    category_join = ""
    category_expression = "'All papers'"
    if {"paper_categories", "categories"}.issubset(tables):
        pc_columns = _columns(connection, "paper_categories")
        category_columns = _columns(connection, "categories")
        pc_paper = _first(pc_columns, "paper_id", "arxiv_id")
        pc_category = _first(pc_columns, "category_id")
        category_id = _first(category_columns, "id", "category_id")
        category_label = _first(category_columns, "name", "code", "category", "slug")
        if all((pc_paper, pc_category, category_id, category_label)):
            category_join = (
                f' LEFT JOIN paper_categories pc ON pc.{_identifier(pc_paper)} = p.{_identifier(paper_id)}'
                f' LEFT JOIN categories c ON c.{_identifier(category_id)} = pc.{_identifier(pc_category)}'
            )
            category_expression = f"COALESCE(NULLIF(c.{_identifier(category_label)}, ''), 'Uncategorized')"

    if not category_join:
        paper_category = _first(paper_columns, "primary_category", "category", "categories")
        if paper_category:
            category_expression = (
                f"COALESCE(NULLIF(p.{_identifier(paper_category)}, ''), 'Uncategorized')"
            )

    keyword_join = None
    keyword_expression = None
    keyword_method_column = None
    if "paper_keywords" in tables:
        pk_columns = _columns(connection, "paper_keywords")
        pk_paper = _first(pk_columns, "paper_id", "arxiv_id")
        keyword_text = _first(pk_columns, "keyword", "term", "phrase", "keyword_text")
        keyword_method_column = _first(pk_columns, "method", "extractor", "algorithm")
        if pk_paper:
            keyword_join = (
                f' JOIN paper_keywords pk ON pk.{_identifier(pk_paper)} = p.{_identifier(paper_id)}'
            )
            if keyword_text:
                keyword_expression = f"LOWER(TRIM(pk.{_identifier(keyword_text)}))"
            elif "keywords" in tables and "keyword_id" in pk_columns:
                key_columns = _columns(connection, "keywords")
                key_id = _first(key_columns, "id", "keyword_id")
                key_text = _first(key_columns, "keyword", "term", "text", "phrase")
                if key_id and key_text:
                    keyword_join += f' JOIN keywords k ON k.{_identifier(key_id)} = pk."keyword_id"'
                    keyword_expression = f"LOWER(TRIM(k.{_identifier(key_text)}))"

    return DashboardSchema(
        paper_id=paper_id,
        year_expression=year_expression,
        abstract_column=abstract_column,
        category_join=category_join,
        category_expression=category_expression,
        keyword_join=keyword_join,
        keyword_expression=keyword_expression,
        keyword_method_column=keyword_method_column,
    )


def _filtered_where(schema: DashboardSchema, filters: DashboardFilters) -> tuple[str, list]:
    if filters.start_year > filters.end_year:
        raise DashboardDataError("Start year must not exceed end year.")
    clauses = [f"{schema.year_expression} BETWEEN ? AND ?"]
    params: list = [filters.start_year, filters.end_year]
    if filters.categories:
        clauses.append(
            f"{schema.category_expression} IN ({','.join('?' for _ in filters.categories)})"
        )
        params.extend(filters.categories)
    return " AND ".join(clauses), params


def _keyword_predicate(schema: DashboardSchema, filters: DashboardFilters) -> tuple[str, list]:
    if schema.keyword_join is None or schema.keyword_expression is None:
        raise DashboardDataError(
            "The 'paper_keywords' data is not ready. Run keyword extraction and DB import first."
        )
    if filters.keyword_method and schema.keyword_method_column:
        return f' AND pk.{_identifier(schema.keyword_method_column)} = ?', [filters.keyword_method]
    return "", []


def load_filter_options(connection: sqlite3.Connection) -> tuple[tuple[int, int], list[str], list[str]]:
    """Read date/category/method options. No corpus text leaves the DB."""
    schema = inspect_schema(connection)
    try:
        years = connection.execute(
            f"SELECT MIN({schema.year_expression}), MAX({schema.year_expression}) "
            f"FROM papers p WHERE {schema.year_expression} BETWEEN 1900 AND 2100"
        ).fetchone()
        if not years or years[0] is None:
            raise DashboardDataError("There are no papers with valid publication years in the database.")
        categories = [
            row[0]
            for row in connection.execute(
                f"SELECT DISTINCT {schema.category_expression} AS category "
                f"FROM papers p {schema.category_join} ORDER BY category"
            )
        ]
        methods: list[str] = []
        if schema.keyword_method_column:
            methods = [
                str(row[0])
                for row in connection.execute(
                    f'SELECT DISTINCT {_identifier(schema.keyword_method_column)} '
                    f'FROM paper_keywords WHERE {_identifier(schema.keyword_method_column)} IS NOT NULL '
                    f'ORDER BY {_identifier(schema.keyword_method_column)}'
                )
            ]
        return (int(years[0]), int(years[1])), categories, methods
    except sqlite3.Error as error:
        raise DashboardDataError(f"Cannot load filter options: {error}") from error


def load_publication_trends(connection: sqlite3.Connection, filters: DashboardFilters) -> pd.DataFrame:
    schema = inspect_schema(connection)
    where, params = _filtered_where(schema, filters)
    query = (
        f"SELECT {schema.year_expression} AS year, {schema.category_expression} AS category, "
        f'COUNT(DISTINCT p.{_identifier(schema.paper_id)}) AS papers '
        f"FROM papers p {schema.category_join} WHERE {where} "
        "GROUP BY year, category ORDER BY year, category"
    )
    return pd.read_sql_query(query, connection, params=params)


def load_top_keywords(
    connection: sqlite3.Connection, filters: DashboardFilters, limit: int = 20
) -> pd.DataFrame:
    schema = inspect_schema(connection)
    condition, method_params = _keyword_predicate(schema, filters)
    where, params = _filtered_where(schema, filters)
    limit = max(1, min(limit, 100))
    # DISTINCT papers first prevents category joins multiplying keyword counts.
    query = (
        "WITH selected AS ("
        f'SELECT DISTINCT p.{_identifier(schema.paper_id)} AS paper_id FROM papers p '
        f"{schema.category_join} WHERE {where}"
        ") "
        f"SELECT {schema.keyword_expression} AS keyword, "
        "COUNT(DISTINCT s.paper_id) AS paper_count "
        "FROM selected s JOIN papers p ON p."
        f'{_identifier(schema.paper_id)} = s.paper_id {schema.keyword_join} '
        f"WHERE TRIM({schema.keyword_expression}) <> '' {condition} "
        "GROUP BY keyword ORDER BY paper_count DESC, keyword ASC LIMIT ?"
    )
    return pd.read_sql_query(query, connection, params=params + method_params + [limit])


def load_keyword_cooccurrence(
    connection: sqlite3.Connection,
    filters: DashboardFilters,
    keywords: list[str],
    sample_size: int = 4000,
) -> pd.DataFrame:
    schema = inspect_schema(connection)
    condition, method_params = _keyword_predicate(schema, filters)
    chosen = list(dict.fromkeys(keywords))[:12]
    if not chosen:
        return pd.DataFrame(columns=["keyword_a", "keyword_b", "papers"])
    where, params = _filtered_where(schema, filters)
    sample_size = max(1, min(sample_size, 20000))
    query = (
        "WITH selected AS ("
        f'SELECT DISTINCT p.{_identifier(schema.paper_id)} AS paper_id FROM papers p '
        f'{schema.category_join} WHERE {where} ORDER BY paper_id LIMIT ?), '
        "terms AS ("
        "SELECT DISTINCT s.paper_id, "
        f"{schema.keyword_expression} AS keyword "
        "FROM selected s JOIN papers p ON p."
        f'{_identifier(schema.paper_id)} = s.paper_id {schema.keyword_join} '
        f"WHERE {schema.keyword_expression} IN ({','.join('?' for _ in chosen)}) {condition}"
        ") SELECT a.keyword AS keyword_a, b.keyword AS keyword_b, "
        "COUNT(*) AS papers FROM terms a JOIN terms b "
        "ON a.paper_id = b.paper_id AND a.keyword <= b.keyword "
        "GROUP BY a.keyword, b.keyword"
    )
    return pd.read_sql_query(
        query, connection, params=params + [sample_size] + chosen + method_params
    )


def load_abstract_lengths(
    connection: sqlite3.Connection, filters: DashboardFilters, sample_size: int = 5000
) -> pd.DataFrame:
    schema = inspect_schema(connection)
    if not schema.abstract_column:
        raise DashboardDataError("The 'papers' table has no abstract/summary column.")
    where, params = _filtered_where(schema, filters)
    sample_size = max(1, min(sample_size, 20000))
    query = (
        f'SELECT DISTINCT p.{_identifier(schema.paper_id)} AS paper_id, '
        f'p.{_identifier(schema.abstract_column)} AS abstract '
        f"FROM papers p {schema.category_join} WHERE {where} "
        f'AND p.{_identifier(schema.abstract_column)} IS NOT NULL '
        "ORDER BY paper_id LIMIT ?"
    )
    frame = pd.read_sql_query(query, connection, params=params + [sample_size])
    if frame.empty:
        return pd.DataFrame(columns=["word_count"])
    return pd.DataFrame({"word_count": frame["abstract"].str.split().str.len().astype(int)})


def load_keyword_trends(
    connection: sqlite3.Connection, filters: DashboardFilters, keywords: list[str]
) -> pd.DataFrame:
    schema = inspect_schema(connection)
    condition, method_params = _keyword_predicate(schema, filters)
    chosen = list(dict.fromkeys(keywords))[:5]
    if not chosen:
        return pd.DataFrame(columns=["year", "keyword", "papers"])
    where, params = _filtered_where(schema, filters)
    query = (
        "WITH selected AS ("
        f'SELECT DISTINCT p.{_identifier(schema.paper_id)} AS paper_id, '
        f"{schema.year_expression} AS year FROM papers p {schema.category_join} WHERE {where}"
        ") SELECT s.year AS year, "
        f"{schema.keyword_expression} AS keyword, "
        "COUNT(DISTINCT s.paper_id) AS papers "
        "FROM selected s JOIN papers p ON p."
        f'{_identifier(schema.paper_id)} = s.paper_id {schema.keyword_join} '
        f"WHERE {schema.keyword_expression} IN ({','.join('?' for _ in chosen)}) {condition} "
        "GROUP BY s.year, keyword ORDER BY s.year, keyword"
    )
    return pd.read_sql_query(query, connection, params=params + chosen + method_params)
