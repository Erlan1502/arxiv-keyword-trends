"""Unit tests for the NLP extraction and preprocessing modules."""

from __future__ import annotations

import unittest

from src.nlp.extractors import (
    FrequencyBaselineExtractor,
    RakeKeywordExtractor,
    TfidfKeywordExtractor,
    YakeKeywordExtractor,
    extract_keywords,
    process_paper_chunk,
)
from src.nlp.interface import ExtractionResult, ScoredKeyword
from src.nlp.preprocess import clean_text, preprocess_text, tokenize

SAMPLE_TEXT = (
    "Transformers have achieved state-of-the-art performance across diverse natural language "
    "processing tasks. In this paper, we propose an efficient attention mechanism that scales "
    "linearly with sequence length $O(N)$. We evaluate our method on language modeling benchmarks."
)


class TestNLPPreprocessing(unittest.TestCase):
    def test_clean_text_removes_latex_and_urls(self):
        dirty = r"Attention is all you need $O(N^2)$ \cite{vaswani} https://arxiv.org/abs/1706.03762"
        cleaned = clean_text(dirty)
        self.assertNotIn("$O(N^2)$", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("Attention is all you need", cleaned)

    def test_tokenize_returns_words(self):
        tokens = tokenize("Deep learning and neural networks!")
        self.assertIn("deep", tokens)
        self.assertIn("learning", tokens)
        self.assertIn("neural", tokens)
        self.assertIn("networks", tokens)

    def test_preprocess_text_lemmatization_and_stemming(self):
        text = "Deep neural networks are training efficiently"
        lemmatized = preprocess_text(text, mode="lemmatization")
        stemmed = preprocess_text(text, mode="stemming")

        # Проверяем, что результат - непустая строка
        self.assertTrue(len(lemmatized) > 0)
        self.assertTrue(len(stemmed) > 0)
        # Стоп-слово 'are' должно быть удалено
        self.assertNotIn(" are ", f" {lemmatized} ")
        self.assertNotIn(" are ", f" {stemmed} ")


class TestNLPExtractors(unittest.TestCase):
    def test_frequency_baseline_extractor(self):
        extractor = FrequencyBaselineExtractor(preprocess_mode="lemmatization")
        res = extractor.extract(SAMPLE_TEXT, top_k=5)

        self.assertIsInstance(res, ExtractionResult)
        self.assertEqual(res.method, "frequency_baseline")
        self.assertTrue(len(res.keywords) > 0)
        self.assertLessEqual(len(res.keywords), 5)
        # Проверяем нормализацию скоров
        for kw in res.keywords:
            self.assertIsInstance(kw, ScoredKeyword)
            self.assertGreaterEqual(kw.score, 0.0)
            self.assertLessEqual(kw.score, 1.0)
        # Проверяем метрики
        self.assertIn("execution_time_ms", res.metrics)
        self.assertIn("peak_memory_mb", res.metrics)

    def test_tfidf_extractor_single_doc_and_corpus(self):
        extractor = TfidfKeywordExtractor(preprocess_mode="lemmatization")

        # Без fit (локальный TF-IDF)
        res_single = extractor.extract(SAMPLE_TEXT, top_k=5)
        self.assertIsInstance(res_single, ExtractionResult)
        self.assertEqual(res_single.method, "tfidf")
        self.assertTrue(len(res_single.keywords) > 0)

        # С обучением на мини-корпусе
        corpus = [SAMPLE_TEXT, "Reinforcement learning in robotics using deep Q-networks."]
        extractor.fit(corpus)
        res_corpus = extractor.extract(SAMPLE_TEXT, top_k=5)
        self.assertTrue(len(res_corpus.keywords) > 0)

    def test_yake_extractor(self):
        extractor = YakeKeywordExtractor()
        res = extractor.extract(SAMPLE_TEXT, top_k=5)

        self.assertIsInstance(res, ExtractionResult)
        self.assertEqual(res.method, "yake")
        self.assertTrue(len(res.keywords) > 0)
        for kw in res.keywords:
            self.assertGreater(kw.score, 0.0)

    def test_rake_extractor(self):
        extractor = RakeKeywordExtractor()
        res = extractor.extract(SAMPLE_TEXT, top_k=5)

        self.assertIsInstance(res, ExtractionResult)
        self.assertEqual(res.method, "rake")
        self.assertTrue(len(res.keywords) > 0)

    def test_facade_extract_keywords(self):
        for method in ["frequency_baseline", "tfidf", "yake", "rake"]:
            res = extract_keywords(SAMPLE_TEXT, method=method, top_k=5)
            self.assertIsInstance(res, ExtractionResult)
            self.assertEqual(res.method, method)
            self.assertTrue(len(res.keywords) > 0)

    def test_process_paper_chunk(self):
        chunk = [
            {"id": "2101.00001", "abstract": SAMPLE_TEXT},
            {"id": "2101.00002", "abstract": "Convolutional networks for real-time object tracking."},
        ]
        results = process_paper_chunk(chunk, method="tfidf", top_k=5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].paper_id, "2101.00001")
        self.assertEqual(results[1].paper_id, "2101.00002")


if __name__ == "__main__":
    unittest.main()
