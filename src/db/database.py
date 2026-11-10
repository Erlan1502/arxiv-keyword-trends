"""Загрузка партиционированного Parquet в SQLite.

Запуск:
    python -m src.db.database
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/arxiv.db")
SCHEMA = Path("src/db/schema.sql")
PARQUET = Path("data/processed")

# Индексы строятся ПОСЛЕ загрузки: поддержка их во время вставки
# ~19 млн строк замедляет загрузку в разы.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_papers_year      ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_cat       ON papers(primary_category);
CREATE INDEX IF NOT EXISTS idx_pc_category      ON paper_categories(category_code);
CREATE INDEX IF NOT EXISTS idx_authors_paper    ON authors(paper_id);
CREATE INDEX IF NOT EXISTS idx_keywords_paper   ON paper_keywords(paper_id);
CREATE INDEX IF NOT EXISTS idx_keywords_word    ON paper_keywords(keyword);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """WAL + отключённый fsync: ускоряет массовую вставку на порядок."""
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = OFF")
    con.execute("PRAGMA temp_store = MEMORY")
    con.execute("PRAGMA cache_size = -200000")   # ~200 МБ кэша страниц
    return con


def iter_year_frames():
    """Год берём из имени каталога (Hive-партиция), а не из файла."""
    for year_dir in sorted(PARQUET.glob("year=*")):
        year = int(year_dir.name.split("=")[1])
        for fp in sorted(year_dir.glob("*.parquet")):
            df = pd.read_parquet(fp)
            df["year"] = year
            yield year, df


def to_rows(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    """numpy-типы -> питоновские: sqlite3 не умеет биндить np.int16."""
    sub = df[cols].astype(object)
    sub = sub.where(sub.notna(), None)
    return list(sub.itertuples(index=False, name=None))


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA.read_text(encoding="utf-8"))


def load(con: sqlite3.Connection) -> dict[str, int]:
    next_pid = 1
    stats = {"papers": 0, "authors": 0, "links": 0}
    all_categories: set[str] = set()
    t0 = time.time()

    # FK проверяем ПОСЛЕ загрузки: иначе каждая вставка ищет родителя.
    con.execute("PRAGMA foreign_keys = OFF")

    for year, df in iter_year_frames():
        n = len(df)
        df = df.assign(id=range(next_pid, next_pid + n))
        next_pid += n

        con.executemany(
            "INSERT INTO papers (id, arxiv_id, title, abstract, year, month,"
            " primary_category, n_authors) VALUES (?,?,?,?,?,?,?,?)",
            to_rows(df, ["id", "arxiv_id", "title", "abstract",
                         "year", "month", "primary_category", "n_authors"]),
        )
        stats["papers"] += n

        # авторы: authors_parsed = [[фамилия, инициалы, суффикс], ...]
        # ВАЖНО: после parquet ячейка приходит как numpy.ndarray, а не list,
        # поэтому проверяем наличие __len__ (у NaN его нет).
        author_rows = []
        for pid, parsed in zip(df["id"].tolist(), df["authors_parsed"].tolist()):
            if parsed is None or not hasattr(parsed, "__len__"):
                continue
            for pos, a in enumerate(parsed):
                name = " ".join(str(p).strip() for p in a if str(p).strip())
                if name:
                    author_rows.append((int(pid), name, pos))
        con.executemany(
            "INSERT INTO authors (paper_id, name, position) VALUES (?,?,?)",
            author_rows,
        )
        stats["authors"] += len(author_rows)

        # категории: одна статья может быть в нескольких
        link_rows = []
        for pid, cats in zip(df["id"].tolist(), df["all_categories"].tolist()):
            for code in str(cats).split():
                link_rows.append((int(pid), code))
                all_categories.add(code)
        con.executemany(
            "INSERT INTO paper_categories (paper_id, category_code) VALUES (?,?)",
            link_rows,
        )
        stats["links"] += len(link_rows)

        con.commit()
        print(f"year={year} papers={n} authors={len(author_rows)} "
              f"links={len(link_rows)}  [{time.time()-t0:.0f}s]", flush=True)

    con.executemany(
        "INSERT OR IGNORE INTO categories (code) VALUES (?)",
        [(c,) for c in sorted(all_categories)],
    )
    con.commit()
    return stats


def build_indexes(con: sqlite3.Connection) -> None:
    t0 = time.time()
    con.executescript(INDEXES)
    con.commit()
    print(f"indexes built in {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = connect()
    create_schema(con)
    stats = load(con)
    build_indexes(con)
    con.execute("PRAGMA foreign_keys = ON")
    bad = con.execute("PRAGMA foreign_key_check").fetchall()
    print("\n=== SUMMARY ===")
    print(f"papers={stats['papers']} authors={stats['authors']} "
          f"links={stats['links']}")
    print(f"fk violations: {len(bad)}")
    print(f"db size: {DB_PATH.stat().st_size / 2**30:.2f} GB")
    con.close()


if __name__ == "__main__":
    main()
