"""Потоковый ETL: arxiv-metadata-oai-snapshot.json -> партиционированный Parquet.

Дамп весит ~5.3 ГБ и содержит ~2.9 млн строк (JSON Lines). Целиком в память
его загрузить нельзя, поэтому читаем чанками и каждый чанк сразу пишем на диск.

Запуск:
    python -m src.etl.parser                  # весь файл (~7 минут)
    python -m src.etl.parser --max-chunks 1   # только первый чанк (проверка)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/arxiv-metadata-oai-snapshot.json")
OUT = Path("data/processed")
CHUNK = 100_000

# arXiv стартовал в 1991; верхнюю границу берём динамически, иначе при
# жёстком 2025 теряются статьи, поданные в текущем году (в дампе от
# 2026-11 их ~430 тыс. по update_date).
MIN_YEAR = 1991
MAX_YEAR = pd.Timestamp.now(tz="UTC").year

KEEP = ["id", "title", "abstract", "categories",
        "authors_parsed", "versions", "update_date"]


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка и нормализация одного чанка."""
    df = df[KEEP].copy()

    # 1. обязательные поля + нормализация пробелов (в исходнике есть \n)
    df = df.dropna(subset=["id", "title", "abstract", "categories", "update_date"])
    df["title"] = df["title"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["abstract"] = df["abstract"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["title"].str.len() > 0]
    df = df[df["abstract"].str.len() >= 50]      # отсекаем пустые аннотации

    # 2. дата подачи: versions[0].created, иначе update_date.
    #    update_date = дата ПОСЛЕДНЕГО изменения, она сдвигает старые статьи
    #    в свежие годы и портит динамику.
    def _submitted(v):
        if isinstance(v, list) and v:
            return v[0].get("created")
        return None

    df = df.assign(_submitted=df["versions"].apply(_submitted))
    dt = pd.to_datetime(df["_submitted"], errors="coerce", utc=True,
                        format="%a, %d %b %Y %H:%M:%S %Z")
    fallback = pd.to_datetime(df["update_date"], errors="coerce",
                              format="%Y-%m-%d")
    dt = dt.fillna(fallback)
    df = df.assign(year=dt.dt.year, month=dt.dt.month)
    df = df.dropna(subset=["year"])
    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)]

    # 3. категории: строка вида "cs.LG math.OC stat.ML"
    cats = df["categories"].str.split()
    df = df.assign(primary_category=cats.str[0], all_categories=df["categories"])
    df = df.dropna(subset=["primary_category"])

    # 4. авторы
    df = df.assign(n_authors=df["authors_parsed"].apply(
        lambda x: len(x) if isinstance(x, list) else 0))

    df = df.drop(columns=["categories", "update_date", "versions", "_submitted"])
    df = df.rename(columns={"id": "arxiv_id"})
    df = df.drop_duplicates(subset=["arxiv_id"])
    return df.reset_index(drop=True)


def optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Приведение типов: экономия RAM (замеряется в run())."""
    return df.astype({
        "arxiv_id": "string",
        "title": "string",
        "abstract": "string",
        "primary_category": "category",
        "all_categories": "string",
        "year": "int16",
        "month": "int8",
        "n_authors": "int16",
    })


def run(max_chunks: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reader = pd.read_json(RAW, lines=True, chunksize=CHUNK)

    total_raw = total_clean = 0
    mem_raw = mem_opt = 0
    t0 = time.time()

    for i, chunk in enumerate(reader):
        if max_chunks is not None and i >= max_chunks:
            break

        raw_rows = len(chunk)
        raw_bytes = chunk.memory_usage(deep=True).sum()

        clean = clean_chunk(chunk)
        opt = optimize_types(clean)

        total_raw += raw_rows
        total_clean += len(opt)
        mem_raw += raw_bytes
        mem_opt += opt.memory_usage(deep=True).sum()

        # Партиционирование по годам. Колонку year В ФАЙЛ НЕ ПИШЕМ:
        # каталог year=YYYY -- это Hive-партиция, pyarrow восстановит
        # колонку из пути. Если записать её ещё и внутрь файла, при чтении
        # будет конфликт типов (int16 vs dictionary).
        for year, part in opt.groupby("year", observed=True):
            d = OUT / f"year={int(year)}"
            d.mkdir(parents=True, exist_ok=True)
            part.drop(columns=["year"]).to_parquet(
                d / f"part_{i:05d}.parquet", index=False)

        print(f"chunk {i:03d}: raw={raw_rows:>6} kept={len(opt):>6} "
              f"mem {raw_bytes/2**20:.0f}->{opt.memory_usage(deep=True).sum()/2**20:.0f} MB",
              flush=True)

    dt = time.time() - t0
    print("\n=== SUMMARY ===")
    print(f"rows: {total_raw} -> {total_clean}")
    print(f"memory raw: {mem_raw/2**30:.2f} GB")
    print(f"memory optimized: {mem_opt/2**30:.2f} GB")
    print(f"saving: {100*(1-mem_opt/mem_raw):.1f}%")
    print(f"time: {dt:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="ограничить число чанков (для быстрой проверки)")
    run(ap.parse_args().max_chunks)
