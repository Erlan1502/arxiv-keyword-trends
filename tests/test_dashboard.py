"""Frontend contract tests on a tiny temporary SQLite fixture (no Kaggle download)."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src.app.data_access import (
    DashboardDataError,
    DashboardFilters,
    connect_read_only,
    inspect_schema,
    load_abstract_lengths,
    load_filter_options,
    load_keyword_cooccurrence,
    load_keyword_trends,
    load_publication_trends,
    load_top_keywords,
)


class DashboardDataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "sample.sqlite"
        # sqlite3.Connection.__exit__ commits but does not close the handle.
        # Windows refuses to delete a temporary .sqlite file while it is open.
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript(
                """
                CREATE TABLE papers (
                    id TEXT PRIMARY KEY, published TEXT NOT NULL, abstract TEXT NOT NULL
                );
                CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT UNIQUE);
                CREATE TABLE paper_categories (paper_id TEXT, category_id INTEGER);
                CREATE TABLE paper_keywords (paper_id TEXT, keyword TEXT, method TEXT, score REAL);
                INSERT INTO papers VALUES
                  ('a', '2022-01-15', 'graph learning improves AI'),
                  ('b', '2022-02-21', 'graph model studies graphs'),
                  ('c', '2023-05-01', 'neural network for vision'),
                  ('d', '2023-07-01', 'a transformer model');
                INSERT INTO categories VALUES (1, 'cs.AI'), (2, 'cs.LG');
                INSERT INTO paper_categories VALUES
                  ('a',1), ('a',2), ('b',1), ('c',2), ('d',1);
                INSERT INTO paper_keywords VALUES
                  ('a','graph','tfidf',0.8), ('a','learning','tfidf',0.7),
                  ('b','graph','tfidf',0.9), ('b','model','tfidf',0.6),
                  ('c','neural','tfidf',0.8), ('d','model','tfidf',0.8),
                  ('a','graph','yake',0.5);
                """
            )
            connection.commit()
        self.connection = connect_read_only(self.path)
        self.all = DashboardFilters(2022, 2023, keyword_method="tfidf")

    def tearDown(self):
        self.connection.close()
        self.folder.cleanup()

    def test_filter_options(self):
        years, categories, methods = load_filter_options(self.connection)
        self.assertEqual(years, (2022, 2023))
        self.assertEqual(categories, ["cs.AI", "cs.LG"])
        self.assertEqual(methods, ["tfidf", "yake"])

    def test_publications_category_join(self):
        frame = load_publication_trends(
            self.connection, DashboardFilters(2022, 2022, categories=("cs.AI",))
        )
        self.assertEqual(frame.to_dict("records"), [{"year": 2022, "category": "cs.AI", "papers": 2}])

    def test_top_terms_do_not_double_count_multi_category_paper(self):
        frame = load_top_keywords(self.connection, self.all)
        graph = frame.set_index("keyword").loc["graph"]
        self.assertEqual(int(graph["paper_count"]), 2)

    def test_category_filter_prevents_unrelated_results(self):
        filters = DashboardFilters(2023, 2023, categories=("cs.LG",), keyword_method="tfidf")
        frame = load_top_keywords(self.connection, filters)
        self.assertEqual(frame["keyword"].tolist(), ["neural"])

    def test_topic_trends_and_cooccurrence(self):
        trends = load_keyword_trends(self.connection, self.all, ["graph", "model"])
        graph = trends[trends["keyword"] == "graph"]
        self.assertEqual(graph["papers"].tolist(), [2])
        pairs = load_keyword_cooccurrence(self.connection, self.all, ["graph", "model"])
        self.assertEqual(int(pairs[pairs["keyword_a"] == "graph"].iloc[0]["papers"]), 2)

    def test_abstract_word_counts(self):
        frame = load_abstract_lengths(self.connection, self.all)
        self.assertEqual(sorted(frame["word_count"].tolist()), [3, 4, 4, 4])

    def test_read_only_and_missing_database(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.connection.execute("INSERT INTO categories VALUES (4, 'cs.CV')")
        with self.assertRaises(DashboardDataError):
            connect_read_only(self.path.parent / "missing.sqlite")

    def test_rejects_unknown_schema(self):
        with sqlite3.connect(":memory:") as connection:
            with self.assertRaises(DashboardDataError):
                inspect_schema(connection)


class DashboardChartTests(unittest.TestCase):
    def test_all_five_charts_return_plotly_figures(self):
        try:
            import pandas as pd
            import plotly.graph_objects as go
            from src.viz import charts
        except ImportError:
            self.skipTest("plotly/pandas not installed")
        cases = [
            (charts.publications_by_year, pd.DataFrame([{"year": 2022, "category": "cs.AI", "papers": 2}])),
            (charts.top_keywords, pd.DataFrame([{"keyword": "graph", "paper_count": 2}])),
            (charts.abstract_length_distribution, pd.DataFrame([{"word_count": 23}, {"word_count": 42}])),
            (charts.keyword_trends, pd.DataFrame([{"year": 2022, "keyword": "graph", "papers": 2}])),
        ]
        for chart, data in cases:
            with self.subTest(chart=chart.__name__):
                self.assertIsInstance(chart(data), go.Figure)
        pairs = pd.DataFrame([{"keyword_a": "graph", "keyword_b": "graph", "papers": 2}])
        self.assertIsInstance(charts.keyword_cooccurrence(pairs, ["graph"]), go.Figure)


if __name__ == "__main__":
    unittest.main()
