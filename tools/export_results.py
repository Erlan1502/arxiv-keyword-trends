#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑 8 条查询，导出 CSV + Markdown；可选生成 Q4 / Q7 的图。

    python export_results.py --db data/arxiv.db --out results --figures

任何数据库都能用：只要表结构一致（papers / categories / paper_categories /
authors / paper_keywords）。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import time

MARKER = re.compile(r"^-- (Q\d+) \[([^\]]+)\]")


def load_queries(sql_path: str):
    """把 queries.sql 切成 (id, kind, sql) 列表。"""
    text = open(sql_path, encoding="utf-8").read()
    blocks = re.split(r"\n(?=-- Q\d+ \[)", text)
    out = []
    for b in blocks:
        m = MARKER.match(b.strip().splitlines()[0]) if b.strip() else None
        if not m:
            continue
        body = "\n".join(l for l in b.splitlines() if not l.strip().startswith("--"))
        if body.strip():
            out.append((m.group(1), m.group(2), body))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--sql", default=None, help="queries.sql 路径")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", action="store_true", help="生成 Q4/Q7 图片")
    args = ap.parse_args()

    sql_path = args.sql or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "src", "db", "queries.sql")
    queries = load_queries(sql_path)
    os.makedirs(args.out, exist_ok=True)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=60)
    summary = []
    data = {}

    for qid, kind, sql in queries:
        t0 = time.time()
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        dt = time.time() - t0
        data[qid] = (cols, rows)
        path = os.path.join(args.out, f"{qid}.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as f:   # BOM：Excel 打开不乱码
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        print(f"{qid}: {len(rows):,} rows, {dt:.2f}s -> {path}")
        summary.append((qid, kind, len(rows), dt, cols, rows))

    # Markdown 汇总
    md = ["# Результаты 8 аналитических запросов\n"]
    for qid, kind, n, dt, cols, rows in summary:
        md.append(f"## {qid} [{kind}] — {n} строк, {dt:.2f} с\n")
        if rows:
            md.append("| " + " | ".join(cols) + " |")
            md.append("|" + "---|" * len(cols))
            for r in rows[:30]:
                md.append("| " + " | ".join(str(x) for x in r) + " |")
            if len(rows) > 30:
                md.append(f"\n*… всего {n} строк, полный файл `{qid}.csv`*\n")
        md.append("")
    open(os.path.join(args.out, "all_results.md"), "w", encoding="utf-8").write("\n".join(md))

    con.close()

    if args.figures:
        make_figures(data, os.path.dirname(args.out) or ".")


def make_figures(data, outdir: str) -> None:
    """Q4：10 个关键词 × 16 年的小多图；Q7：几个年份的 Top-5 榜单。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#c9d3dc",
        "axes.linewidth": 0.8,
        "text.color": "#14181f",
        "axes.labelcolor": "#475564",
        "xtick.color": "#788695",
        "ytick.color": "#788695",
    })
    ink, accent, grid = "#14181f", "#16606b", "#e3e8ee"

    # ---------- Q4 ----------
    if data.get("Q4") and data["Q4"][1]:
        _, rows = data["Q4"]
        years = sorted({r[1] for r in rows})
        kws = []
        for r in rows:
            if r[0] not in kws:
                kws.append(r[0])
        series = {k: {y: 0 for y in years} for k in kws}
        for k, y, n in rows:
            series[k][y] = n

        fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True)
        fig.patch.set_facecolor("white")
        for ax, k in zip(axes.ravel(), kws):
            ys = [series[k][y] for y in years]
            ax.plot(years, ys, color=accent, linewidth=2.2, solid_capstyle="round")
            ax.fill_between(years, ys, color=accent, alpha=0.10, linewidth=0)
            ax.scatter([years[-1]], [ys[-1]], s=18, color=accent, zorder=3)
            ax.annotate(str(ys[-1]), (years[-1], ys[-1]), textcoords="offset points",
                        xytext=(4, 4), fontsize=10, color=ink, fontweight="bold")
            ax.set_title(k, fontsize=14, color=ink, fontweight="bold", pad=8)
            ax.set_ylim(0, max(ys) * 1.18 + 1)
            ax.grid(axis="y", color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=9)
            ax.set_xticks([years[0], years[len(years)//2], years[-1]])
        fig.suptitle("Q4 — papers mentioning each keyword per year (TF-IDF, sample)",
                     fontsize=16, color=ink, x=0.01, ha="left", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = os.path.join(outdir, "figures", "Q4_keyword_trends.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("figure:", p)

    # ---------- Q1：类别 Top-20 ----------
    if data.get("Q1") and data["Q1"][1]:
        _, rows = data["Q1"]
        rows = rows[:20][::-1]                    # 反向，最大值画在最上面
        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("white")
        ax.barh(labels, vals, color=accent, height=0.68)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:,}", va="center", fontsize=9.5, color=ink)
        ax.set_xlabel("papers")
        ax.set_title("Q1 — top-20 arXiv categories by number of papers",
                     fontsize=15, color=ink, loc="left", pad=12)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9.5)
        ax.set_xlim(0, max(vals) * 1.12)
        fig.tight_layout()
        p = os.path.join(outdir, "figures", "Q1_top20_categories.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("figure:", p)

    # ---------- Q5：每个年代的 Top-5 类别（小多图，避免 5 条线抢颜色） ----------
    if data.get("Q5") and data["Q5"][1]:
        _, rows = data["Q5"]
        years = sorted({r[0] for r in rows})
        total = {}
        for y, rnk, cat, n in rows:
            total[cat] = total.get(cat, 0) + n
        top5 = [c for c, _ in sorted(total.items(), key=lambda x: -x[1])[:5]]
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharex=True)
        fig.patch.set_facecolor("white")
        for ax, cat in zip(axes, top5):
            pts = sorted((r[0], r[3]) for r in rows if r[2] == cat)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=accent, linewidth=2.0)
            ax.set_title(cat, fontsize=12.5, color=ink, fontweight="bold")
            ax.grid(axis="y", color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=8.5)
        fig.suptitle("Q5 — papers per year for the five most frequent top-5 categories",
                     fontsize=14, color=ink, x=0.01, ha="left")
        fig.text(0.01, 0.90, "Lines start or end where a category entered or left the yearly top-5 — a gap is not zero papers.",
                 fontsize=9.5, color="#788695", ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        p = os.path.join(outdir, "figures", "Q5_top_categories_over_time.png")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("figure:", p)

    # ---------- Q6：每年论文数 + 增长率 ----------
    if data.get("Q6") and data["Q6"][1]:
        _, rows = data["Q6"]
        rows = sorted(rows, key=lambda r: r[0])
        ys = [r[0] for r in rows]
        papers = [r[1] for r in rows]
        growth = [r[3] for r in rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6))
        fig.patch.set_facecolor("white")
        a1.bar(ys, papers, color=accent, width=0.75)
        a1.set_title("Papers per year", fontsize=13, color=ink, loc="left")
        a1.grid(axis="y", color=grid, linewidth=0.8); a1.set_axisbelow(True)
        a1.tick_params(labelsize=8.5)
        # 1992 的 +800% 是“1991 只有半年数据”的伪影，画进去会把整条线压平
        gx = [y for y, g in zip(ys, growth) if y >= 1993]
        gy = [g for y, g in zip(ys, growth) if y >= 1993]
        a2.plot(gx, gy, color=accent, linewidth=2.0)
        a2.axhline(0, color="#c9d3dc", linewidth=0.8)
        a2.text(0.02, 0.92, "1992: +800% — partial 1991 (omitted)", transform=a2.transAxes,
                fontsize=9, color="#788695")
        a2.set_title("Year-over-year growth, %", fontsize=13, color=ink, loc="left")
        a2.grid(axis="y", color=grid, linewidth=0.8); a2.set_axisbelow(True)
        a2.tick_params(labelsize=8.5)
        for ax in (a1, a2):
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.suptitle("Q6 — corpus growth (1991 and 1992 are partial years)",
                     fontsize=14, color=ink, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = os.path.join(outdir, "figures", "Q6_corpus_growth.png")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("figure:", p)

    # ---------- Q8：高产作者（注意：是名字，不是人） ----------
    if data.get("Q8") and data["Q8"][1]:
        _, rows = data["Q8"]
        rows = rows[:20][::-1]
        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 6.4))
        fig.patch.set_facecolor("white")
        ax.barh(labels, vals, color=accent, height=0.68)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:,}", va="center", fontsize=9.5, color=ink)
        ax.set_xlabel("papers")
        ax.set_title("Q8 — most frequent author names (not deduplicated people)",
                     fontsize=14, color=ink, loc="left", pad=12)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.set_xlim(0, max(vals) * 1.12)
        fig.tight_layout()
        p = os.path.join(outdir, "figures", "Q8_top_author_names.png")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("figure:", p)

    # ---------- Q7 ----------
    if data.get("Q7") and data["Q7"][1]:
        _, rows = data["Q7"]
        want = [2005, 2010, 2015, 2020, 2024, 2026]
        table = {y: {} for y in want}
        for y, rnk, kw, n in rows:
            if y in table:
                table[y][rnk] = (kw, n)

        fig, ax = plt.subplots(figsize=(13, 4.6))
        fig.patch.set_facecolor("white")
        ax.axis("off")
        ncol = len(want)
        pitch = 1.45                          # 每列宽度：1.0 会把数字挤到下一列上
        ax.set_xlim(0, ncol * pitch + 0.4)
        ax.set_ylim(0, 7)
        ax.text(0.02, 6.6, "Q7 — top-5 keywords each year", fontsize=16,
                color=ink, fontweight="bold")
        for j, y in enumerate(want):
            x = 0.9 + j * pitch
            ax.text(x, 5.9, str(y), fontsize=14, color=ink, fontweight="bold", ha="center")
            for rnk in range(1, 6):
                if rnk not in table[y]:
                    continue
                kw, n = table[y][rnk]
                yy = 5.35 - (rnk - 0.5) * 1.0
                ax.text(x - 0.62, yy + 0.08, str(rnk), fontsize=10, color="#788695", ha="center")
                ax.text(x - 0.50, yy + 0.08, kw, fontsize=12.5, color=ink, ha="left", va="center")
                ax.text(x - 0.50, yy - 0.22, str(n), fontsize=9, color="#9aa7b4", ha="left", va="center")
        p = os.path.join(outdir, "figures", "Q7_top5_by_year.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("figure:", p)


if __name__ == "__main__":
    main()
