# arXiv Keyword & Topic Trends

Командный финальный проект (AAI-2605M) по анализу метаданных научных статей [Cornell arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) с использованием методов Data Engineering, NLP и интерактивной визуализации.

---

## Архитектура проекта

```
arxiv-keyword-trends/
├── data/              # SQLite база данных (arxiv.db) и Parquet-партиции
├── docs/              # Отчёт и черновик презентации (report_draft.md)
├── src/
│   ├── etl/           # Потоковый парсер дампа JSON -> Parquet (parser.py)
│   ├── db/            # Схема SQLite (schema.sql), bulk-load (database.py), 8 SQL-запросов (queries.sql)
│   ├── nlp/           # Предобработка (preprocess.py), экстракторы (extractors.py), бенчмарк (benchmark.py)
│   ├── app/           # Streamlit дашборд (main.py) и read-only доступ к БД (data_access.py)
│   └── viz/           # Интерактивные графики Plotly (charts.py)
└── tests/             # Юнит-тесты дашборда и NLP-пайплайна (18 тестов)
```

---

## Быстрый старт

### 1. Настройка окружения
Требуется Python 3.10+:

```bash
# Клонирование и переход в репозиторий
cd arxiv-keyword-trends

# Создание и активация виртуального окружения
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Запуск компонентов

- **Веб-дашборд (Streamlit):**
  ```bash
  streamlit run src/app/main.py
  ```
  *Включает вкладку аналитики корпуса (5 графиков Plotly) и вкладку ML-инференса для извлечения ключевых слов на лету.*

- **NLP Бенчмарк:**
  ```bash
  python -m src.nlp.benchmark
  ```
  *Сравнивает скорость (мс), потребление памяти (МБ) и overlap (Жаккар) для Frequency, TF-IDF, YAKE и RAKE.*

- **ETL-пайплайн (при наличии raw-дампа):**
  ```bash
  python -m src.etl.parser --max-chunks 1   # Тестовый прогон 1 чанка
  python -m src.db.database                # Загрузка в SQLite и построение индексов
  ```

---

## Тестирование

Запуск полного набора юнит-тестов (18 тестов):

```bash
python -m unittest discover -s tests -v
```

Тесты проверяют корректность SQL-агрегатов, генерацию фигур Plotly, предобработку текстов, работу всех экстракторов и интерфейсный контракт.

---

## Вклад участников

| Участник | Роль | Основной вклад |
| :--- | :--- | :--- |
| **Hayilati Shamuhaer** | Data Engineering | Потоковый парсинг 5.3 ГБ дампа чанками по 100k, Parquet-партиции, 3NF схема SQLite, WAL-режим и 8 аналитических SQL-запросов. |
| **Erlan Kubesov** | Machine Learning & NLP | Очистка LaTeX и стоп-слов, реализация 4 экстракторов (TF-IDF, YAKE, RAKE, Baseline), замер метрик (мс, МБ), бенчмарк и юнит-тесты. |
| **Nurali Oryngaliev** | Frontend & Visualization | Streamlit-дашборд, 5 графиков Plotly (тренды, топ-20, heatmap, распределение длин), read-only адаптер данных и автотесты UI. |
