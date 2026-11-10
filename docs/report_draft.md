# Final Project — report and presentation draft

**Topic:** Keyword & Topic Trends in arXiv Abstracts

**Program:** Applied Artificial Intelligence, AAI-2605M

**Team:** Erlan Kubesov, Nurali Oryngaliev, Hayilati Shamuhaer

> This is a structure for the team's final report. Add actual dataset counts,
> benchmark measurements, charts and interpretations after the ETL and NLP
> branches are integrated. Do not present placeholders as experimental results.

## 1. Introduction and research questions

- Which research topics occur most often in arXiv abstracts?
- How does the relative frequency of important keywords change over time?
- How do frequency-based extraction, TF-IDF and YAKE compare in speed,
  resource use and keyword usefulness?

## 2. Dataset and data engineering (Data Engineering contributor)

- Source: [Cornell arXiv metadata](https://www.kaggle.com/datasets/Cornell-University/arxiv).
- Record and explain the real downloaded row count, date range, and categories.
- Describe chunked JSON parsing, missing-value handling, type optimization,
  Parquet partitions, and memory comparison before/after.
- Include ER diagram: `papers` → `paper_categories` → `categories`, plus
  `paper_keywords` and the `authors` table.
- Describe at least eight analytical SQL queries including JOIN, GROUP BY,
  HAVING, CTE/subqueries, and window functions; show Pandas readback.

## 3. NLP extraction and benchmarking (Erlan Kubesov)

- **Text preprocessing (`src/nlp/preprocess.py`):**
  - Robust regex stripping of LaTeX equations (`$O(N^2)$`, `\begin{...}`), citations, and academic artifacts common in arXiv abstracts.
  - Hybrid stop-words dictionary: standard English stop-words combined with domain-specific academic filler words (`paper`, `propose`, `results`, `demonstrate`).
  - Normalization modes: WordNet lemmatization vs. Porter stemming vs. raw tokens.

- **Extraction algorithms (`src/nlp/extractors.py`):**
  - *Frequency Baseline:* counts unigrams and high-frequency bigrams.
  - *TF-IDF:* term frequency - inverse document frequency with bi-gram support, usable with corpus fitting or sentence-level single-document decomposition.
  - *YAKE:* unsupervised multi-feature scoring (casing, word position, context frequency, spread), normalized to $[0.0, 1.0]$.
  - *RAKE:* rapid automatic keyword phrase extraction with custom lightweight sentence tokenization.

- **Benchmark and experimental results (`src/nlp/benchmark.py`):**

| Method | Preprocessing | Avg Time (ms) | Peak Memory (MB) | Processed Tokens / doc | Sample Top Extracted Terms |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Frequency Baseline** | Lemmatization | **1.13 ms** | 0.0096 MB | 32 | `language`, `transformer`, `achieved`, `performance` |
| **Frequency Baseline** | Stemming | 4.76 ms | 0.0090 MB | 32 | `languag`, `transform`, `achiev`, `perform` |
| **TF-IDF** | Lemmatization | 6.90 ms | 0.0185 MB | 45 | `language`, `accuracy`, `achieved state-of-the-art` |
| **TF-IDF** | Stemming | 8.48 ms | 0.0197 MB | 45 | `languag`, `accuraci`, `achiev state-of-the-art` |
| **TF-IDF** | None | **4.35 ms** | 0.0185 MB | 45 | `language`, `accuracy`, `achieved state-of-the-art` |
| **YAKE** | None | 25.33 ms | 0.1714 MB | 45 | `gpu memory`, `performance across`, `natural processing` |
| **YAKE** | Lemmatization | 26.13 ms | 0.1481 MB | 45 | `performance across`, `perplexity accuracy`, `processing task` |
| **RAKE** | None | **1.33 ms** | 0.0218 MB | 45 | `language modeling benchmarks`, `efficient attention mechanism` |
| **RAKE** | Lemmatization | 2.24 ms | 0.0180 MB | 45 | `sequence length evaluate`, `language modeling benchmark` |

- **Method Overlap Matrix (Jaccard Similarity on Top-10 Keywords):**
  - Frequency vs. TF-IDF: **0.176** (moderate overlap on high-frequency topical unigrams).
  - YAKE vs. RAKE: **0.125** (both identify multi-word phrases, though with different boundary heuristics).
  - TF-IDF vs. YAKE: **0.111**.
  - Frequency vs. YAKE: **0.000** (frequency favors frequent single tokens, while YAKE prioritizes discriminative multi-token phrases).

- **Qualitative Comparison & Error Analysis:**
  - *Lemmatization vs. Stemming:* Stemming produces truncated roots (`languag`, `accuraci`) that look unprofessional in dashboard presentation, whereas lemmatization outputs canonical English words (`language`, `accuracy`) at a negligible speed difference.
  - *Unigrams vs. Phrases:* TF-IDF and Frequency are prone to picking generic scientific nouns unless filtered by custom academic stop-words. YAKE and RAKE successfully isolate informative multi-word phrases (e.g., `efficient attention mechanism`, `language modeling benchmarks`).


## 4. Frontend, SQL analytics and visualizations (Nurali Oryngaliev)

**Implementation:** `src/app/main.py`, `src/app/data_access.py`,
`src/viz/charts.py`. Streamlit provides a Python-only dashboard with two
functional tabs and an About tab. The Analytics tab filters the SQLite corpus
by publication years, arXiv categories and extraction method. The Inference
tab accepts a new abstract, chooses the extraction algorithm and preprocessing
mode, and displays keyword scores, elapsed time, memory and token count.

**Figures to insert after integration:**

1. Line chart: year/category publication trends. Interpret differences
   carefully: one paper may belong to multiple arXiv categories.
2. Horizontal bar chart: top 20 extracted terms, ranked by *distinct papers*.
3. Heatmap: pairwise co-occurrence of top terms across up to 4,000 matched
   paper IDs. State that this is a bounded sample, not necessarily the full corpus.
4. Histogram with marginal boxplot: abstract word lengths for up to 5,000
   papers in the selected filters. Identify long-tail observations.
5. Multi-line plot: annual counts for five frequent keywords. Discuss trends
   **only after verifying** date coverage and the chosen extraction method.

**Architecture:** The UI delegates SQL aggregation to a read-only adapter,
and Plotly receives only compact Pandas DataFrames. The NLP inference tab uses
the team's shared `extract_keywords` interface. Missing integrations produce
clear warnings rather than invented analytical results. The database is opened
in read-only mode, cache expires after five minutes, and expensive sample views
have explicit row limits.

**Validation:** Run `python -m unittest discover -s tests -v`; these unit
tests cover SQL correctness on a small fixture, multi-category deduplication,
date/category filters, no-write permissions, and Plotly figure creation.

## 5. Discussion and limitations (all contributors)

- Bias from arXiv coverage, publication dates and category assignments.
- Keyword ambiguity, spelling variants, discipline-specific phrases.
- Difference between keyword occurrence, extraction score and actual topic.
- Sample-size limits on heatmap and length distribution.
- Future work: embeddings, semantic clusters, normalized counts per year.

## 6. Conclusion (all contributors)

Summarize observed patterns only once the data and benchmarks are available.
Ensure each team member's contribution is stated in the report and slides.

## Suggested presentation (8–10 slides)

1. Title, team and responsibilities.
2. Problem, research questions and dataset.
3. ETL pipeline, memory optimization and database ER diagram.
4. SQL queries and aggregated results.
5. NLP methods and preprocessing comparison.
6. Benchmark table and error analysis.
7. Dashboard demonstration: analytics filters and charts.
8. Dashboard demonstration: abstract inference and runtime metrics.
9. Main results, limitations and conclusion.

**Before submission checklist:** Python source in GitHub, real Kaggle source
link or data, README launch instructions, screenshots of actual dashboard,
filled report with experiment values, PDF/PPTX slides, contribution statements.
