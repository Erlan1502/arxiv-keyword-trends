
from typing import Protocol, Literal, TypedDict
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ScoredKeyword:
    """Ключевое слово или фраза с нормализованным весом значимости."""
    keyword: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkMetrics(TypedDict):
    """Метрики ресурсоемкости для обязательного сравнения методов."""
    execution_time_ms: float
    peak_memory_mb: float
    total_tokens_processed: int


@dataclass
class ExtractionResult:
    """Итоговый контейнер, возвращаемый Streamlit и БД."""
    paper_id: str | None
    method: Literal["frequency_baseline", "tfidf", "yake", "rake", "keybert"]
    keywords: list[ScoredKeyword]
    metrics: BenchmarkMetrics

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "method": self.method,
            "keywords": [kw.to_dict() for kw in self.keywords],
            "metrics": dict(self.metrics),
        }


class BaseKeywordExtractor(Protocol):
    """Интерфейс для алгоритмов извлечения ключевых слов в src/nlp/."""

    def fit(self, corpus: list[str]) -> None:
        """Обучение словаря/статистик (TF-IDF; no-op для RAKE/YAKE)."""
        ...

    def extract(self, text: str, top_k: int = 10) -> ExtractionResult:
        """Извлечение top_k ключевых слов из текста с замером времени и памяти."""
        ...

    def extract_batch(self, texts: list[str], top_k: int = 10) -> list[ExtractionResult]:
        """Пакетное извлечение для обработки чанков."""
        ...


def extract_keywords(
    text: str,
    method: Literal["frequency_baseline", "tfidf", "yake", "rake", "keybert"] = "tfidf",
    top_k: int = 10,
    preprocess_mode: Literal["stemming", "lemmatization", "none"] = "lemmatization",
) -> ExtractionResult:
    """Фасадная функция для вызова из Streamlit (src/app/main.py)."""
    from src.nlp.extractors import extract_keywords as _impl
    return _impl(text=text, method=method, top_k=top_k, preprocess_mode=preprocess_mode)


def process_paper_chunk(
    papers: list[dict[str, str]],
    method: Literal["tfidf", "yake"] = "tfidf",
    top_k: int = 10,
) -> list[ExtractionResult]:
    """Фасадная функция для вызова из ETL-пайплайна (src/etl/parser.py)."""
    from src.nlp.extractors import process_paper_chunk as _impl
    return _impl(papers=papers, method=method, top_k=top_k)

    