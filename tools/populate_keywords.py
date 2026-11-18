#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Заполнение paper_keywords. 独立脚本，不依赖项目里的任何模块。

用法示例（先跑样本库）：
    python populate_keywords.py --db sample.db --method tfidf --top-k 10
用法示例（用样本库估计 idf，再给主库打分）：
    python populate_keywords.py --db arxiv.db --method tfidf --df-from sample.db --top-k 10

只写 paper_keywords 一张表，不修改任何其它表。
"""
from __future__ import annotations

import argparse        # 命令行参数
import math            # 计算 idf 要用 log
import os              # 文件存在性检查
import re              # 分词正则
import sqlite3         # 数据库
import sys             # 退出码与输出
import time            # 计时
from collections import Counter  # 计数

# 词形还原（可选）：models -> model，studies -> study。
# 项目 requirements 里本来就有 nltk，缺了也不影响脚本运行。
try:
    from nltk.stem import WordNetLemmatizer  # noqa: N812
    _LEMMATIZER = WordNetLemmatizer()
except Exception:                              # nltk 没装或没下 wordnet 数据
    _LEMMATIZER = None
_LEMMA_CACHE: dict[str, str] = {}              # 缓存，避免同一单词反复还原

# ---------------------------------------------------------------------------
# 分词与清洗
# ---------------------------------------------------------------------------

# 论文摘要里常出现 LaTeX 公式，$...$ 里的内容不是自然语言，先整段去掉
MATH_RE = re.compile(r"\$[^$]{0,400}\$")
# 破折号：arXiv 摘要把 em dash 写成 "--"，不处理会切出 "star--triangle" 这种伪词
DASH_RE = re.compile(r"-{2,}|[‒-―]")
# 单词：以小写字母开头，允许用单个连字符连接（state-of-the-art 算一个词）
WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")

# 停用词：常见虚词 + 论文八股（这些词在几乎每篇摘要里都有，做成关键词没有信息量）
STOP = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers herself him himself his how however i if in into is it its itself just me more most my
myself no nor not now of off on once only or other our ours ourselves out over own same she should so some
such than that the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your yours yourself
paper papers propose proposed proposes we our us study studies result results method methods approach approaches
present presents presented show shows shown demonstrate demonstrates described describe use used uses using
based new novel different well many much first second third two three one also additionally furthermore moreover
thus therefore can may might must should could would within without across among between per via etc al
""".split())

def normalize(text: str) -> str:
    """转小写 + 去掉 LaTeX 公式段 + 把多连字符破折号换成空格。"""
    if not text:
        return ""
    return DASH_RE.sub(" ", MATH_RE.sub(" ", text.lower()))

def lemma(word: str) -> str:
    """取名词词元；带缓存。没有 nltk 时原样返回。"""
    if _LEMMATIZER is None:
        return word
    hit = _LEMMA_CACHE.get(word)
    if hit is not None:
        return hit
    out = _LEMMATIZER.lemmatize(word, pos="n")   # 按名词还原：models -> model
    _LEMMA_CACHE[word] = out
    return out


def tokenize(text: str, lemmatize: bool = True) -> list[str]:
    """把一段文本切成词；停用词在这里就过滤掉，后面不用再判。

    lemmatize=True 时先做词形还原，再把还原后落进停用词的（papers -> paper）丢掉；
    长度 < 3 的碎片（a, of, xy）一并丢掉。
    """
    out: list[str] = []
    for w in WORD_RE.findall(normalize(text)):
        if lemmatize:
            w = lemma(w)
        if len(w) >= 3 and w not in STOP:
            out.append(w)
    return out

# ---------------------------------------------------------------------------
# 数据库连接
# ---------------------------------------------------------------------------

def connect_ro(path: str) -> sqlite3.Connection:
    """只读连接：读 papers 时用，保证不会误写。"""
    if not os.path.exists(path):
        sys.exit(f"[error] 找不到数据库: {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    con.execute("PRAGMA query_only = ON")
    return con

def connect_rw(path: str) -> sqlite3.Connection:
    """读写连接：只用来往 paper_keywords 里插数据。"""
    con = sqlite3.connect(path, timeout=60)
    # WAL 让写入更快，也允许读连接并发
    con.execute("PRAGMA journal_mode = WAL")
    # 批量导入时关掉同步落盘，速度提升明显；导入结束会再同步
    con.execute("PRAGMA synchronous = OFF")
    con.execute("PRAGMA temp_store = MEMORY")
    return con

# ---------------------------------------------------------------------------
# 读论文
# ---------------------------------------------------------------------------

def iter_papers(con: sqlite3.Connection, years: tuple[int, int] | None,
                limit: int | None, start_id: int = 0, batch: int = 5000):
    """按 id 分页读取论文。

    ⚠️ 关键：每一批用独立的 fetchall 并立刻结束读事务。
    如果用一个长期开着的游标（fetchmany 循环），SQLite 在 WAL 模式下
    无法推进 checkpoint，-wal 文件会一路涨到把磁盘写满。
    """
    last = start_id
    remaining = limit
    while True:
        this = batch if remaining is None else min(batch, remaining)
        if this <= 0:
            break
        sql = "SELECT id, title, abstract, year FROM papers WHERE id > ?"
        params: list = [last]
        if years:
            sql += " AND year BETWEEN ? AND ?"
            params += [years[0], years[1]]
        sql += " ORDER BY id LIMIT ?"
        params.append(this)
        rows = con.execute(sql, params).fetchall()   # 取完即提交读事务
        if not rows:
            break
        for row in rows:
            yield row
        last = rows[-1][0]
        if remaining is not None:
            remaining -= len(rows)

# ---------------------------------------------------------------------------
# 两种提取方法
# ---------------------------------------------------------------------------

def compute_df(db: str, years, limit, lemmatize: bool) -> tuple[Counter, int]:
    """统计文档频率 df：每个词出现在多少篇摘要里（每篇只算一次）。

    这是 TF-IDF 的 idf 部分。只扫一遍，内存占用与词表大小成正比。
    """
    con = connect_ro(db)
    df: Counter = Counter()
    n_docs = 0
    for _, title, abstract, _ in iter_papers(con, years, limit):
        words = set(tokenize(f"{title} {abstract}", lemmatize))  # 同一篇里重复只算 1 次
        df.update(words)
        n_docs += 1
        if n_docs % 100000 == 0:                      # 进度提示
            print(f"  [df] {n_docs:,} papers, {len(df):,} types", flush=True)
    con.close()
    return df, n_docs

def keywords_baseline(words: list[str], k: int) -> list[tuple[str, float]]:
    """词频基线：直接数次数。作为对照组，简单但会出泛词。"""
    if not words:
        return []
    counts = Counter(words)
    total = sum(counts.values())
    return [(w, c / total) for w, c in counts.most_common(k)]

def keywords_tfidf(words: list[str], df: Counter, n_docs: int,
                   max_df: float, k: int) -> list[tuple[str, float]]:
    """TF-IDF：词频 × 逆文档频率，再过滤掉过常见的词。"""
    if not words:
        return []
    counts = Counter(words)
    total = sum(counts.values())
    scored: list[tuple[str, float]] = []
    for w, c in counts.items():
        d = df.get(w, 0)
        if d == 0:                                   # 训练集里没见过的词，跳过
            continue
        if d / n_docs > max_df:                      # 出现在超过 max_df 比例文档里 -> 太泛
            continue
        tf = c / total
        idf = math.log((n_docs + 1) / (d + 1)) + 1.0
        scored.append((w, tf * idf))
    scored.sort(key=lambda x: x[1], reverse=True)    # 按分数从高到低
    return scored[:k]

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Fill paper_keywords for Q4/Q7")
    ap.add_argument("--db", required=True, help="要填充的数据库（会被写入）")
    ap.add_argument("--method", choices=["baseline", "tfidf"], default="tfidf")
    ap.add_argument("--df-from", help="tfidf 的 df 来源库；省略则扫描 --db 本身（两遍）")
    ap.add_argument("--top-k", type=int, default=10, help="每篇保留几个关键词")
    ap.add_argument("--max-df", type=float, default=0.5, help="df/N 超过该值的词丢弃")
    ap.add_argument("--years", help="只处理某段年份，例如 2010-2025")
    ap.add_argument("--limit", type=int, help="只处理前 N 篇（调试用）")
    ap.add_argument("--reset", action="store_true", help="先删掉该 method 的旧数据")
    ap.add_argument("--dry-run", action="store_true", help="只看前 3 篇的结果，不写库")
    ap.add_argument("--resume", action="store_true",
                    help="从中断处续跑：跳过已有的最大 paper_id")
    ap.add_argument("--lemmatize", choices=["auto", "on", "off"], default="auto",
                    help="词形还原：auto=有 nltk 就开（默认）")
    args = ap.parse_args()

    # auto：装得上 nltk + wordnet 就还原，否则原样
    use_lemma = args.lemmatize == "on" or (args.lemmatize == "auto" and _LEMMATIZER is not None)
    print(f"词形还原: {'开' if use_lemma else '关'}", flush=True)

    years = None
    if args.years:
        lo, hi = args.years.split("-")
        years = (int(lo), int(hi))

    t0 = time.time()

    # 1) 准备 idf
    df: Counter | None = None
    n_docs = 0
    if args.method == "tfidf":
        src = args.df_from or args.db
        print(f"[1/2] 统计 df：{src}", flush=True)
        df, n_docs = compute_df(src, years, args.limit, use_lemma)
        print(f"      完成：{n_docs:,} 篇，{len(df):,} 个词，{time.time()-t0:.1f}s", flush=True)

    # 2) 写入
    print(f"[2/2] 提取并写入：{args.db}", flush=True)
    reader = connect_ro(args.db)
    writer = None if args.dry_run else connect_rw(args.db)

    if writer and args.reset:
        cur = writer.execute("DELETE FROM paper_keywords WHERE method = ?", (args.method,))
        writer.commit()
        print(f"      已清除旧数据：{cur.rowcount:,} 行", flush=True)

    # 续跑：已经写入过关键词的最大 paper_id，从它后面接着做
    start_id = 0
    if args.resume:
        tmp = sqlite3.connect(args.db, timeout=60)
        start_id = tmp.execute(
            "SELECT COALESCE(MAX(paper_id), 0) FROM paper_keywords WHERE method = ?",
            (args.method,)).fetchone()[0]
        tmp.close()
        print(f"      续跑：从 paper_id > {start_id:,} 开始", flush=True)

    buf: list[tuple] = []          # 待插入的缓冲
    done = 0
    written = 0
    for pid, title, abstract, year in iter_papers(reader, years, args.limit, start_id):
        words = tokenize(f"{title} {abstract}", use_lemma)
        if args.method == "baseline":
            pairs = keywords_baseline(words, args.top_k)
        else:
            pairs = keywords_tfidf(words, df, n_docs, args.max_df, args.top_k)

        if args.dry_run:                                   # 只看不写，看完 3 篇就退出
            print(f"  #{pid} ({year}): {[w for w, _ in pairs]}")
            done += 1
            if done >= 3:
                break
            continue
        buf.extend((pid, w, round(s, 6), args.method) for w, s in pairs)
        done += 1

        if buf and len(buf) >= 20000:          # 攒够 2 万行就提交一次
            writer.executemany(
                "INSERT INTO paper_keywords (paper_id, keyword, score, method) VALUES (?, ?, ?, ?)", buf)
            writer.commit()
            written += len(buf)
            buf.clear()

        if writer and done % 200000 == 0:      # 定期把 WAL 合并回去，防止它无限增长
            writer.execute("PRAGMA wal_checkpoint(PASSIVE)")

        if done % 50000 == 0:                  # 每 5 万篇报一次进度
            el = time.time() - t0
            rate = done / el if el else 0
            print(f"  {done:,} papers | {written+len(buf):,} rows | {rate:,.0f} p/s | "
                  f"剩约 {(3232618-done)/rate/60:.0f} min" if rate else f"  {done:,} papers", flush=True)

    if writer and buf:                          # 最后一批
        writer.executemany(
            "INSERT INTO paper_keywords (paper_id, keyword, score, method) VALUES (?, ?, ?, ?)", buf)
        writer.commit()
        written += len(buf)

    if writer:
        writer.execute("ANALYZE")               # 更新统计信息，帮助查询优化器
        writer.commit()
        total = writer.execute("SELECT COUNT(*) FROM paper_keywords").fetchone()[0]
        writer.close()
        print(f"\n完成：{done:,} 篇 -> 写入 {written:,} 行；表内共 {total:,} 行；"
              f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    reader.close()

if __name__ == "__main__":
    main()
