-- Схема БД проекта "Keyword & Topic Trends in arXiv Abstracts".
-- ВНИМАНИЕ: индексы здесь НЕ создаются. Они строятся после массовой
-- загрузки данных (см. src/db/database.py -> INDEXES). Причина: поддержка
-- шести индексов во время вставки ~19 млн строк замедляет загрузку в 3-5 раз.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers (
    id               INTEGER PRIMARY KEY,
    arxiv_id         TEXT    NOT NULL UNIQUE,
    title            TEXT    NOT NULL,
    abstract         TEXT    NOT NULL,
    year             INTEGER NOT NULL,          -- год ПОДАЧИ (versions[0].created)
    month            INTEGER,                   -- 1..12
    primary_category TEXT    NOT NULL,          -- первая категория из строки
    n_authors        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS categories (
    code TEXT PRIMARY KEY,                      -- напр. cs.LG
    name TEXT                                   -- человекочитаемое имя (опционально)
);

CREATE TABLE IF NOT EXISTS paper_categories (
    paper_id      INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    category_code TEXT    NOT NULL REFERENCES categories(code),
    PRIMARY KEY (paper_id, category_code)
);

CREATE TABLE IF NOT EXISTS authors (
    id       INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    name     TEXT    NOT NULL,
    position INTEGER NOT NULL                   -- порядок автора в статье, с 0
);

CREATE TABLE IF NOT EXISTS paper_keywords (
    id       INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    keyword  TEXT    NOT NULL,
    score    REAL    NOT NULL,
    method   TEXT    NOT NULL                   -- tfidf | rake | yake | baseline
);
