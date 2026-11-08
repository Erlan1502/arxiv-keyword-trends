# ER-диаграмма БД `arxiv.db`

Схема: 5 таблиц, одна связь «многие ко многим» развязана промежуточной таблицей.

```mermaid
erDiagram
    papers ||--o{ authors : "1 : N"
    papers ||--o{ paper_categories : "1 : N"
    papers ||--o{ paper_keywords : "1 : N"
    categories ||--o{ paper_categories : "1 : N"

    papers {
        INTEGER id PK
        TEXT    arxiv_id UK "напр. 0704.0001"
        TEXT    title
        TEXT    abstract
        INTEGER year "год ПОДАЧИ (versions[0].created)"
        INTEGER month
        TEXT    primary_category FK "-> categories.code"
        INTEGER n_authors
    }

    categories {
        TEXT code PK "напр. cs.LG"
        TEXT name
    }

    paper_categories {
        INTEGER paper_id FK "-> papers.id"
        TEXT    category_code FK "-> categories.code"
    }

    authors {
        INTEGER id PK
        INTEGER paper_id FK "-> papers.id"
        TEXT    name
        INTEGER position "порядок автора, с 0"
    }

    paper_keywords {
        INTEGER id PK
        INTEGER paper_id FK "-> papers.id"
        TEXT    keyword
        REAL    score
        TEXT    method "tfidf | rake | yake | baseline"
    }
```

## Почему так

| Решение | Причина |
|---|---|
| `paper_categories` — отдельная таблица | поле `categories` в исходнике содержит несколько кодов через пробел; при хранении строкой приходится искать `LIKE '%cs.LG%'`, что ломает индекс и даёт ложные совпадения (`cs.LG` ⊂ `cs.LGx`) |
| `year` / `month` — целые числа | хранение `update_date` текстом заставляет писать `substr(...)` в `GROUP BY`, из-за чего индекс не используется |
| `paper_keywords.method` | позволяет хранить результаты нескольких экстракторов (Erlan) в одной таблице и сравнивать их между собой |
| Индексы создаются после загрузки | поддержка 6 индексов при вставке ~19 млн строк замедляет загрузку в 3–5 раз |

## Объёмы после загрузки

| Таблица | Строк |
|---|---|
| papers | см. вывод `python -m src.db.database` |
| authors | |
| paper_categories | |
| categories | |
| paper_keywords | заполняется модулем NLP (`src/nlp/`) |
