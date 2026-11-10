"""NLP preprocessing pipeline for arXiv scientific abstracts.

Handles LaTeX formulas, special characters, tokenization, stop-words filtering,
and stemming/lemmatization.
"""

from __future__ import annotations

import re
from typing import Literal

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Scientific domain-specific stopwords common in academic abstracts
SCIENTIFIC_STOPWORDS = {
    "paper",
    "propose",
    "proposed",
    "presents",
    "presented",
    "present",
    "show",
    "shows",
    "shown",
    "demonstrate",
    "demonstrates",
    "study",
    "results",
    "result",
    "method",
    "methods",
    "approach",
    "approaches",
    "based",
    "using",
    "use",
    "used",
    "also",
    "well",
    "however",
    "therefore",
    "thus",
    "et",
    "al",
    "author",
    "authors",
    "via",
    "within",
    "new",
    "different",
    "several",
}

# Standard English stopwords (fallback if NLTK stopwords corpus is missing)
FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}

try:
    DEFAULT_STOPWORDS = set(stopwords.words("english")) | SCIENTIFIC_STOPWORDS
except Exception:
    DEFAULT_STOPWORDS = FALLBACK_STOPWORDS | SCIENTIFIC_STOPWORDS

_STEMMER = PorterStemmer()
try:
    _LEMMATIZER = WordNetLemmatizer()
    # Test if wordnet corpus is loaded
    _LEMMATIZER.lemmatize("testing")
except Exception:
    _LEMMATIZER = None

# Regex patterns for cleaning arXiv abstracts
RE_LATEX_INLINE = re.compile(r"\$[^\$]+\$")
RE_LATEX_COMMANDS = re.compile(r"\\[a-zA-Z]+(\{[^}]*\})?")
RE_URLS = re.compile(r"https?://\S+|www\.\S+")
RE_NON_ALPHA = re.compile(r"[^a-zA-Z0-9\s-]")
RE_SPACES = re.compile(r"\s+")
RE_WORD = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{1,}\b")


def clean_text(text: str) -> str:
    """Очистка сырого текста аннотации от LaTeX-разметки, ссылок и спецсимволов."""
    if not text:
        return ""
    # Удаляем LaTeX math выражения и формулы
    cleaned = RE_LATEX_INLINE.sub(" ", text)
    cleaned = RE_LATEX_COMMANDS.sub(" ", cleaned)
    cleaned = RE_URLS.sub(" ", cleaned)
    # Заменяем дефисы между пробелами, сохраняя составные слова (e.g. self-supervised)
    cleaned = re.sub(r"(?<=\s)-(?=\s)", " ", cleaned)
    cleaned = RE_NON_ALPHA.sub(" ", cleaned)
    cleaned = RE_SPACES.sub(" ", cleaned).strip()
    return cleaned


def tokenize(text: str) -> list[str]:
    """Токенизация текста с выделением валидных слов."""
    cleaned = clean_text(text).lower()
    return RE_WORD.findall(cleaned)


def normalize_token(
    token: str,
    mode: Literal["lemmatization", "stemming", "none"] = "lemmatization",
) -> str:
    """Нормализация отдельного токена в зависимости от выбранного режима."""
    token = token.lower().strip()
    if mode == "stemming":
        return _STEMMER.stem(token)
    elif mode == "lemmatization":
        if _LEMMATIZER is not None:
            try:
                return _LEMMATIZER.lemmatize(token)
            except Exception:
                pass
        # Fallback при отсутствии WordNet
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("es") and len(token) > 3:
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss") and len(token) > 2:
            return token[:-1]
        return token
    return token


def preprocess_text(
    text: str,
    mode: Literal["lemmatization", "stemming", "none"] = "lemmatization",
    remove_stopwords: bool = True,
    custom_stopwords: set[str] | None = None,
) -> str:
    """Полная предобработка текста аннотации в нормализованную строку.

    Args:
        text: Исходный текст аннотации.
        mode: Режим нормализации ('lemmatization', 'stemming', 'none').
        remove_stopwords: Флаг удаления стоп-слов.
        custom_stopwords: Опциональный набор дополнительных стоп-слов.

    Returns:
        Предобработанная строка токенов, разделенных пробелом.
    """
    tokens = tokenize(text)
    active_stopwords = (DEFAULT_STOPWORDS | custom_stopwords) if custom_stopwords else DEFAULT_STOPWORDS

    processed_tokens: list[str] = []
    for token in tokens:
        if remove_stopwords and token in active_stopwords:
            continue
        if len(token) <= 2 and not token.isupper():
            continue
        norm = normalize_token(token, mode=mode)
        if remove_stopwords and norm in active_stopwords:
            continue
        if len(norm) > 1:
            processed_tokens.append(norm)

    return " ".join(processed_tokens)
