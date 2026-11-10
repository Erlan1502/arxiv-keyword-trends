# arxiv-keyword-trends

Final project (AAI-2605M): **Keyword & Topic Trends in arXiv Abstracts**.

The team analyzes [Cornell's arXiv metadata dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
with Python NLP methods and a normalized SQLite database.

## Frontend / visualization (feature/dashboard-ui)

This branch adds a Streamlit interface and five interactive Plotly charts:

1. Publication counts by year and category.
2. Top 20 keywords by distinct papers.
3. Keyword co-occurrence heatmap.
4. Abstract word-count histogram with boxplot (up to 5,000 records).
5. Keyword trends by year for the top five terms.

The **Inference** tab accepts an abstract, method (TF-IDF, YAKE, frequency
baseline), preprocessing option and top-k. It calls the shared
`src.nlp.interface.extract_keywords()` function and displays its real metrics.
Until the NLP branch implements that function, the app shows an explicit
"not yet implemented" notice rather than making up results.

### Run

From the repository root, with Python 3.10+:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app/main.py
```

Prepare the database with the team's ETL scripts. The default location is
`data/arxiv.db`; the sidebar accepts another existing `.db` file, or set:

```bash
# Linux / macOS
export ARXIV_DB_PATH=/path/to/arxiv.db
# Windows PowerShell:
# $env:ARXIV_DB_PATH="C:\path\to\arxiv.db"
```

The dashboard opens SQLite **read-only** and does not download the Kaggle
dataset or load 2M+ rows into memory. It queries SQL aggregates and bounded
samples. Re-running the ETL changes the database modification time and clears
cached results automatically (cache TTL: five minutes).

### Integration contract

Database schema comes from `src/db/` (owned by the Data Engineering teammate).
The frontend adapter supports these planned tables:

| Table | Needed columns | Purpose |
| --- | --- | --- |
| `papers` | `id` (or `paper_id`), `year` (or `published`/`published_date`), `abstract` | Years and abstract distribution |
| `categories` | `id`, `name` (or `code`) | Category filter labels |
| `paper_categories` | `paper_id`, `category_id` | Category memberships |
| `paper_keywords` | `paper_id`, `keyword` (or `term`), optional `method` | Top terms and trends |

The frontend also accepts a single category column on `papers` when normalized
category tables have not been loaded. A `keywords(id, keyword)` lookup with
`paper_keywords.keyword_id` is supported. An empty/missing keywords table makes
the related visualizations display an honest integration notice.

**NLP function contract:** `extract_keywords(text, method, top_k,
preprocess_mode)` from `src/nlp/interface.py` returns `ExtractionResult`, with
`keywords=[ScoredKeyword(keyword, score)]` and `metrics` containing
`execution_time_ms`, `peak_memory_mb`, `total_tokens_processed`.

### Test the frontend

```bash
python -m unittest discover -s tests -v
python -m compileall -q src/app src/viz
```

The tests create a temporary sample SQLite database, verify the aggregate
logic, read-only connection, and all five Plotly chart functions. This is a
test fixture only, **not the scientific dataset**. Do not report the fixture's
numbers as project findings.

### Frontend files

| Path | Responsibility |
| --- | --- |
| `src/app/main.py` | Streamlit tabs, filters, input and inference integration |
| `src/app/data_access.py` | Read-only SQLite queries and schema adapter |
| `src/viz/charts.py` | Five Plotly figures, separate from SQL |
| `tests/test_dashboard.py` | Automatic frontend/SQL contract tests |
| `docs/report_draft.md` | Team report and presentation outline |

The other branches remain responsible for ingestion, schema, SQL analytical
queries, NLP benchmarking and actual dataset/results. See `docs/report_draft.md`.
