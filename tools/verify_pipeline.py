#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端验收：数据库里被 8 条查询用到的表，一张都不能是空的。

用法：
    python verify_pipeline.py --db data/arxiv.db
任何一张必需的表为空，或 Q4/Q7 返回 0 行，脚本以退出码 1 结束。
把这一步放进合并前的检查里，就不会再出现"查询能跑、结果是空"的情况。
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

# 8 条查询实际用到的表；少一张都说明流水线没接通
REQUIRED_TABLES = ["papers", "categories", "paper_categories", "authors", "paper_keywords"]

Q4 = """
WITH yearly AS (
    SELECT p.year AS year, k.keyword AS keyword, COUNT(DISTINCT p.id) AS papers
    FROM papers p
    JOIN paper_keywords k ON k.paper_id = p.id
    WHERE p.year BETWEEN 2010 AND 2025
    GROUP BY p.year, k.keyword
),
top_terms AS (
    SELECT keyword FROM yearly GROUP BY keyword ORDER BY SUM(papers) DESC LIMIT 10
)
SELECT y.keyword, y.year, y.papers
FROM yearly y JOIN top_terms t ON t.keyword = y.keyword
ORDER BY y.keyword, y.year;
"""

Q7 = """
WITH kw AS (
    SELECT p.year AS year, k.keyword AS keyword, COUNT(DISTINCT p.id) AS papers
    FROM papers p JOIN paper_keywords k ON k.paper_id = p.id
    GROUP BY p.year, k.keyword
),
ranked AS (
    SELECT year, keyword, papers,
           RANK() OVER (PARTITION BY year ORDER BY papers DESC) AS rnk
    FROM kw
)
SELECT year, rnk, keyword, papers
FROM ranked WHERE rnk <= 5 ORDER BY year DESC, rnk;
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"[FAIL] 数据库不存在: {args.db}")
        return 1

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30)
    problems: list[str] = []

    print(f"数据库: {args.db}\n")
    print(f"{'表':<22}{'行数':>14}")
    print("-" * 38)
    for t in REQUIRED_TABLES:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()
        if not exists:
            print(f"{t:<22}{'不存在':>14}")
            problems.append(f"表 {t} 不存在")
            continue
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"{t:<22}{n:>14,}")
        if n == 0:
            problems.append(f"表 {t} 是空的")

    # 关键词覆盖：有多少篇论文真的拿到了关键词
    if "paper_keywords" in [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]:
        total = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0] or 1
        covered = con.execute("SELECT COUNT(DISTINCT paper_id) FROM paper_keywords").fetchone()[0]
        print(f"\n关键词覆盖: {covered:,} / {total:,} 篇 ({covered/total*100:.1f}%)")
        if covered < total * 0.5:
            problems.append(f"关键词只覆盖 {covered/total*100:.1f}% 的论文")

        # 按十年看覆盖，防止"只填了某几年"
        print("\n按年代的关键词覆盖:")
        rows = con.execute("""
            SELECT (p.year/10)*10 AS decade,
                   COUNT(DISTINCT p.id) AS papers,
                   COUNT(DISTINCT k.paper_id) AS with_kw
            FROM papers p LEFT JOIN paper_keywords k ON k.paper_id = p.id
            GROUP BY decade ORDER BY decade
        """).fetchall()
        for dec, papers, with_kw in rows:
            pct = with_kw / papers * 100 if papers else 0
            print(f"  {dec}s: {with_kw:>9,} / {papers:>9,}  ({pct:5.1f}%)")

    # 实跑 Q4 / Q7
    print()
    for label, sql in (("Q4", Q4), ("Q7", Q7)):
        t0 = time.time()
        try:
            rows = con.execute(sql).fetchall()
        except sqlite3.Error as e:
            print(f"{label}: 执行失败 — {e}")
            problems.append(f"{label} 执行失败")
            continue
        dt = time.time() - t0
        print(f"{label}: {len(rows):,} 行, {dt:.2f}s")
        if rows:
            print("   前 3 行:", rows[:3])
        else:
            problems.append(f"{label} 返回 0 行")

    con.close()

    print("\n" + "=" * 38)
    if problems:
        print("结果: FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("结果: PASS — 8 条查询用到的表全部有数据")
    return 0


if __name__ == "__main__":
    sys.exit(main())
