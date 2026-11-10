"""Реализация алгоритмов извлечения ключевых слов (NLP/ML).

Модуль содержит:
1. FrequencyBaselineExtractor - базовый частотный метод (Count / Frequency).
2. TfidfKeywordExtractor - статистический метод на основе TF-IDF.
3. YakeKeywordExtractor - метод YAKE (Yet Another Keyword Extractor).
4. RakeKeywordExtractor - метод RAKE (Rapid Automatic Keyword Extraction).
5. Фасадные функции extract_keywords и process_paper_chunk для UI и ETL.
"""

from __future__ import annotations

import collections
import re
import time
import tracemalloc
from typing import Literal

from sklearn.feature_extraction.text import TfidfVectorizer
import yake
from rake_nltk import Rake

from src.nlp.interface import (
    BaseKeywordExtractor,
    BenchmarkMetrics,
    ExtractionResult,
    ScoredKeyword,
)
from src.nlp.preprocess import (
    DEFAULT_STOPWORDS,
    clean_text,
    preprocess_text,
    tokenize,
)


def _measure_execution(func, *args, **kwargs):
    """Вспомогательная функция для замера времени выполнения и пиковой памяти."""
    tracemalloc.start()
    start_time = time.perf_counter()

    result = func(*args, **kwargs)

    end_time = time.perf_counter()
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    execution_time_ms = (end_time - start_time) * 1000.0
    peak_memory_mb = peak_memory_bytes / (1024.0 * 1024.0)

    return result, execution_time_ms, peak_memory_mb


# =====================================================================
# 1. Frequency Baseline Extractor (Частотный базовый метод)
# =====================================================================

class FrequencyBaselineExtractor:
    """Базовый частотный экстрактор.

    Считает частоту отдельных слов и биграмм (2-словных сочетаний)
    после удаления стоп-слов.
    """

    def __init__(self, preprocess_mode: Literal["lemmatization", "stemming", "none"] = "lemmatization"):
        self.preprocess_mode = preprocess_mode

    def fit(self, corpus: list[str]) -> None:
        """Для чисто частотного метода обучение по корпусу не требуется."""
        pass

    def _extract_core(self, text: str, top_k: int = 10) -> tuple[list[ScoredKeyword], int]:
        processed = preprocess_text(text, mode=self.preprocess_mode, remove_stopwords=True)
        tokens = processed.split()
        total_tokens = len(tokens)

        if not tokens:
            return [], 0

        # Подсчёт частот отдельных слов (униграмм)
        counts = collections.Counter(tokens)

        # Подсчёт биграмм для извлечения устойчивых словосочетаний
        bigrams = []
        for i in range(len(tokens) - 1):
            pair = f"{tokens[i]} {tokens[i + 1]}"
            bigrams.append(pair)

        bigram_counts = collections.Counter(bigrams)

        # Объединяем униграммы и встречающиеся чаще 1 раза биграммы
        combined_scores: dict[str, float] = {}
        for word, count in counts.items():
            if len(word) > 2:
                combined_scores[word] = float(count)

        for bigram, count in bigram_counts.items():
            if count > 1:
                # Даем небольшой бонус устойчивым фразам
                combined_scores[bigram] = float(count * 1.5)

        # Сортируем по убыванию частоты
        sorted_pairs = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

        # Нормализуем веса к диапазону [0.0, 1.0]
        max_score = sorted_pairs[0][1] if sorted_pairs else 1.0
        keywords: list[ScoredKeyword] = []
        for term, score in sorted_pairs[:top_k]:
            normalized_score = round(score / max_score, 4)
            keywords.append(ScoredKeyword(keyword=term, score=normalized_score))

        return keywords, total_tokens

    def extract(self, text: str, top_k: int = 10, paper_id: str | None = None) -> ExtractionResult:
        (keywords, total_tokens), exec_ms, mem_mb = _measure_execution(
            self._extract_core, text, top_k=top_k
        )
        metrics: BenchmarkMetrics = {
            "execution_time_ms": round(exec_ms, 2),
            "peak_memory_mb": round(mem_mb, 4),
            "total_tokens_processed": total_tokens,
        }
        return ExtractionResult(
            paper_id=paper_id,
            method="frequency_baseline",
            keywords=keywords,
            metrics=metrics,
        )

    def extract_batch(self, texts: list[str], top_k: int = 10) -> list[ExtractionResult]:
        results = []
        for text in texts:
            results.append(self.extract(text, top_k=top_k))
        return results


# =====================================================================
# 2. TF-IDF Extractor
# =====================================================================

class TfidfKeywordExtractor:
    """Извлечение ключевых слов с помощью TF-IDF (Term Frequency - Inverse Document Frequency).

    Может работать как с предварительно обученным корпусом, так и
    с отдельным текстом (разбивая его на предложения как мини-документы).
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 50000,
        preprocess_mode: Literal["lemmatization", "stemming", "none"] = "lemmatization",
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.preprocess_mode = preprocess_mode
        self.is_fitted = False
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words=list(DEFAULT_STOPWORDS),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{1,}\b",
        )

    def fit(self, corpus: list[str]) -> None:
        """Обучение словаря TF-IDF на коллекции аннотаций."""
        prepared_corpus = []
        for doc in corpus:
            prepared_doc = preprocess_text(doc, mode=self.preprocess_mode, remove_stopwords=False)
            prepared_corpus.append(prepared_doc)

        if prepared_corpus:
            self.vectorizer.fit(prepared_corpus)
            self.is_fitted = True

    def _extract_core(self, text: str, top_k: int = 10) -> tuple[list[ScoredKeyword], int]:
        tokens = tokenize(text)
        total_tokens = len(tokens)
        if total_tokens == 0:
            return [], 0

        prepared_text = preprocess_text(text, mode=self.preprocess_mode, remove_stopwords=False)

        if self.is_fitted:
            # Используем уже обученный глобальный TF-IDF
            matrix = self.vectorizer.transform([prepared_text])
            feature_names = self.vectorizer.get_feature_names_out()
            row = matrix.tocoo()

            word_scores: list[tuple[str, float]] = []
            for col_idx, score in zip(row.col, row.data):
                word_scores.append((feature_names[col_idx], float(score)))

        else:
            # Если глобальный корпус не был передан, обучаем локально
            # по предложениям текущего абстракта
            sentences = [s.strip() for s in prepared_text.split(".") if len(s.strip()) > 5]
            if len(sentences) < 2:
                sentences = [prepared_text]

            local_vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                stop_words=list(DEFAULT_STOPWORDS),
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{1,}\b",
            )
            try:
                matrix = local_vectorizer.fit_transform(sentences)
                feature_names = local_vectorizer.get_feature_names_out()
                # Берём средний вес каждого термина по предложениям
                mean_scores = matrix.mean(axis=0).A1
                word_scores = []
                for idx, score in enumerate(mean_scores):
                    if score > 0.0:
                        word_scores.append((feature_names[idx], float(score)))
            except ValueError:
                # В случае слишком короткого текста или отсутствия значимых токенов
                return [], total_tokens

        # Сортируем по убыванию веса
        word_scores.sort(key=lambda item: item[1], reverse=True)

        # Нормализуем веса к [0.0, 1.0]
        max_score = word_scores[0][1] if word_scores else 1.0
        keywords: list[ScoredKeyword] = []
        for word, score in word_scores[:top_k]:
            normalized_score = round(score / max_score, 4) if max_score > 0 else 0.0
            keywords.append(ScoredKeyword(keyword=word, score=normalized_score))

        return keywords, total_tokens

    def extract(self, text: str, top_k: int = 10, paper_id: str | None = None) -> ExtractionResult:
        (keywords, total_tokens), exec_ms, mem_mb = _measure_execution(
            self._extract_core, text, top_k=top_k
        )
        metrics: BenchmarkMetrics = {
            "execution_time_ms": round(exec_ms, 2),
            "peak_memory_mb": round(mem_mb, 4),
            "total_tokens_processed": total_tokens,
        }
        return ExtractionResult(
            paper_id=paper_id,
            method="tfidf",
            keywords=keywords,
            metrics=metrics,
        )

    def extract_batch(self, texts: list[str], top_k: int = 10) -> list[ExtractionResult]:
        results = []
        for text in texts:
            results.append(self.extract(text, top_k=top_k))
        return results


# =====================================================================
# 3. YAKE Extractor (Yet Another Keyword Extractor)
# =====================================================================

class YakeKeywordExtractor:
    """Извлечение ключевых слов методом YAKE.

    YAKE не требует внешнего обучающего корпуса и опирается на 5 признаков:
    регистр, позицию слова, контекстную частоту, дисперсию и связность.
    """

    def __init__(
        self,
        max_ngram_size: int = 2,
        deduplication_threshold: float = 0.9,
        preprocess_mode: Literal["lemmatization", "stemming", "none"] = "none",
    ):
        self.max_ngram_size = max_ngram_size
        self.deduplication_threshold = deduplication_threshold
        self.preprocess_mode = preprocess_mode

        # Инициализируем YAKE экстрактор
        self.extractor = yake.KeywordExtractor(
            lan="en",
            n=self.max_ngram_size,
            dedupLim=self.deduplication_threshold,
            stopwords=DEFAULT_STOPWORDS,
        )

    def fit(self, corpus: list[str]) -> None:
        """YAKE - unsupervised алгоритм без необходимости предварительного обучения."""
        pass

    def _extract_core(self, text: str, top_k: int = 10) -> tuple[list[ScoredKeyword], int]:
        tokens = tokenize(text)
        total_tokens = len(tokens)
        if total_tokens == 0:
            return [], 0

        # YAKE лучше всего работает на очищенном тексте с сохранением пунктуации
        cleaned = clean_text(text)
        if self.preprocess_mode != "none":
            cleaned = preprocess_text(cleaned, mode=self.preprocess_mode, remove_stopwords=False)

        raw_keywords = self.extractor.extract_keywords(cleaned)

        # В YAKE меньший score означает большую значимость (0 - лучший).
        # Чтобы шкала была единообразной для пользователя (чем выше, тем лучше),
        # преобразуем: relevance = 1.0 / (1.0 + yake_score)
        relevance_list: list[tuple[str, float]] = []
        for word, y_score in raw_keywords:
            relevance = 1.0 / (1.0 + float(y_score))
            relevance_list.append((word.lower(), relevance))

        # Сортируем по убыванию нашей нормализованной важности
        relevance_list.sort(key=lambda item: item[1], reverse=True)

        keywords: list[ScoredKeyword] = []
        for word, score in relevance_list[:top_k]:
            keywords.append(ScoredKeyword(keyword=word, score=round(score, 4)))

        return keywords, total_tokens

    def extract(self, text: str, top_k: int = 10, paper_id: str | None = None) -> ExtractionResult:
        (keywords, total_tokens), exec_ms, mem_mb = _measure_execution(
            self._extract_core, text, top_k=top_k
        )
        metrics: BenchmarkMetrics = {
            "execution_time_ms": round(exec_ms, 2),
            "peak_memory_mb": round(mem_mb, 4),
            "total_tokens_processed": total_tokens,
        }
        return ExtractionResult(
            paper_id=paper_id,
            method="yake",
            keywords=keywords,
            metrics=metrics,
        )

    def extract_batch(self, texts: list[str], top_k: int = 10) -> list[ExtractionResult]:
        results = []
        for text in texts:
            results.append(self.extract(text, top_k=top_k))
        return results


# =====================================================================
# 4. RAKE Extractor (Rapid Automatic Keyword Extraction)
# =====================================================================

def _simple_sent_tokenize(text: str) -> list[str]:
    parts = re.split(r"[.\n;!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _simple_word_tokenize(sentence: str) -> list[str]:
    return [w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", sentence) if w]


class RakeKeywordExtractor:
    """Извлечение ключевых фраз алгоритмом RAKE."""

    def __init__(self, preprocess_mode: Literal["lemmatization", "stemming", "none"] = "none"):
        self.preprocess_mode = preprocess_mode

    def fit(self, corpus: list[str]) -> None:
        pass

    def _extract_core(self, text: str, top_k: int = 10) -> tuple[list[ScoredKeyword], int]:
        tokens = tokenize(text)
        total_tokens = len(tokens)
        if total_tokens == 0:
            return [], 0

        cleaned = clean_text(text)
        if self.preprocess_mode != "none":
            cleaned = preprocess_text(cleaned, mode=self.preprocess_mode, remove_stopwords=False)

        r = Rake(
            stopwords=DEFAULT_STOPWORDS,
            min_length=1,
            max_length=3,
            sentence_tokenizer=_simple_sent_tokenize,
            word_tokenizer=_simple_word_tokenize,
        )
        r.extract_keywords_from_text(cleaned)

        ranked_phrases = r.get_ranked_phrases_with_scores()
        if not ranked_phrases:
            return [], total_tokens

        max_score = ranked_phrases[0][0] if ranked_phrases else 1.0
        keywords: list[ScoredKeyword] = []
        for score, phrase in ranked_phrases[:top_k]:
            normalized_score = round(score / max_score, 4) if max_score > 0 else 0.0
            keywords.append(ScoredKeyword(keyword=phrase.lower(), score=normalized_score))

        return keywords, total_tokens

    def extract(self, text: str, top_k: int = 10, paper_id: str | None = None) -> ExtractionResult:
        (keywords, total_tokens), exec_ms, mem_mb = _measure_execution(
            self._extract_core, text, top_k=top_k
        )
        metrics: BenchmarkMetrics = {
            "execution_time_ms": round(exec_ms, 2),
            "peak_memory_mb": round(mem_mb, 4),
            "total_tokens_processed": total_tokens,
        }
        return ExtractionResult(
            paper_id=paper_id,
            method="rake",
            keywords=keywords,
            metrics=metrics,
        )

    def extract_batch(self, texts: list[str], top_k: int = 10) -> list[ExtractionResult]:
        results = []
        for text in texts:
            results.append(self.extract(text, top_k=top_k))
        return results


# =====================================================================
# 5. Фасадные функции для UI и ETL
# =====================================================================

def extract_keywords(
    text: str,
    method: Literal["frequency_baseline", "tfidf", "yake", "rake", "keybert"] = "tfidf",
    top_k: int = 10,
    preprocess_mode: Literal["stemming", "lemmatization", "none"] = "lemmatization",
) -> ExtractionResult:
    """Основная точка входа для извлечения ключевых слов (вызывается из Streamlit)."""
    if method == "frequency_baseline":
        extractor = FrequencyBaselineExtractor(preprocess_mode=preprocess_mode)
    elif method == "tfidf":
        extractor = TfidfKeywordExtractor(preprocess_mode=preprocess_mode)
    elif method == "yake":
        extractor = YakeKeywordExtractor(preprocess_mode=preprocess_mode)
    elif method == "rake":
        extractor = RakeKeywordExtractor(preprocess_mode=preprocess_mode)
    else:
        # Fallback к TF-IDF
        extractor = TfidfKeywordExtractor(preprocess_mode=preprocess_mode)

    return extractor.extract(text, top_k=top_k)


def process_paper_chunk(
    papers: list[dict[str, str]],
    method: Literal["tfidf", "yake"] = "tfidf",
    top_k: int = 10,
) -> list[ExtractionResult]:
    """Пакетная обработка списка статей для ETL-пайплайна.

    Каждый элемент словаря papers содержит как минимум:
    - 'id' (или 'paper_id'): идентификатор статьи
    - 'abstract': текст аннотации
    """
    if not papers:
        return []

    if method == "tfidf":
        extractor = TfidfKeywordExtractor()
        # Обучаем TF-IDF на всём поступившем пакете аннотаций
        corpus = [p.get("abstract", "") for p in papers]
        extractor.fit(corpus)
    else:
        extractor = YakeKeywordExtractor()

    results: list[ExtractionResult] = []
    for paper in papers:
        paper_id = paper.get("id") or paper.get("paper_id")
        abstract = paper.get("abstract", "")
        res = extractor.extract(abstract, top_k=top_k, paper_id=str(paper_id) if paper_id else None)
        results.append(res)

    return results
