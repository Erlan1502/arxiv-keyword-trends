-- 8 аналитических запросов к БД arXiv.
-- Каждый запрос помечен типом конструкции, требуемой силлабусом.
-- Запуск: sqlite3 data/arxiv.db < src/db/queries.sql


-- ---------------------------------------------------------------------------
-- Q1 [JOIN] Топ-20 категорий по числу статей.
-- ---------------------------------------------------------------------------
SELECT c.code            AS category,
       COUNT(*)          AS papers
FROM categories c
JOIN paper_categories pc ON pc.category_code = c.code
GROUP BY c.code
ORDER BY papers DESC
LIMIT 20;


-- ---------------------------------------------------------------------------
-- Q2 [GROUP BY + HAVING] Крупные категории среди свежих статей (с 2015).
-- ---------------------------------------------------------------------------
SELECT c.code      AS category,
       COUNT(*)    AS papers
FROM categories c
JOIN paper_categories pc ON pc.category_code = c.code
JOIN papers p            ON p.id = pc.paper_id
WHERE p.year >= 2015
GROUP BY c.code
HAVING COUNT(*) > 10000
ORDER BY papers DESC;


-- ---------------------------------------------------------------------------
-- Q3 [ПОДЗАПРОС] Статьи с аннотацией длиннее средней по всей базе.
-- ---------------------------------------------------------------------------
SELECT p.arxiv_id,
       p.primary_category,
       LENGTH(p.abstract) AS abstract_len
FROM papers p
WHERE LENGTH(p.abstract) > (SELECT AVG(LENGTH(abstract)) FROM papers)
ORDER BY abstract_len DESC
LIMIT 20;


-- ---------------------------------------------------------------------------
-- Q4 [CTE] Динамика по годам для 10 самых частых ключевых слов.
-- ---------------------------------------------------------------------------
WITH yearly AS (
    SELECT p.year            AS year,
           k.keyword         AS keyword,
           COUNT(DISTINCT p.id) AS papers
    FROM papers p
    JOIN paper_keywords k ON k.paper_id = p.id
    WHERE p.year BETWEEN 2010 AND 2025
    GROUP BY p.year, k.keyword
),
top_terms AS (
    SELECT keyword
    FROM yearly
    GROUP BY keyword
    ORDER BY SUM(papers) DESC
    LIMIT 10
)
SELECT y.keyword, y.year, y.papers
FROM yearly y
JOIN top_terms t ON t.keyword = y.keyword
ORDER BY y.keyword, y.year;


-- ---------------------------------------------------------------------------
-- Q5 [ОКОННАЯ ФУНКЦИЯ: RANK] Топ-5 категорий в каждом году.
-- ---------------------------------------------------------------------------
WITH per_year AS (
    SELECT p.year             AS year,
           p.primary_category AS category,
           COUNT(*)           AS papers
    FROM papers p
    GROUP BY p.year, p.primary_category
),
ranked AS (
    SELECT year, category, papers,
           RANK() OVER (PARTITION BY year ORDER BY papers DESC) AS rnk
    FROM per_year
)
SELECT year, rnk, category, papers
FROM ranked
WHERE rnk <= 5
ORDER BY year, rnk;


-- ---------------------------------------------------------------------------
-- Q6 [ОКОННАЯ ФУНКЦИЯ: LAG] Годовой прирост числа статей, %.
-- ---------------------------------------------------------------------------
WITH per_year AS (
    SELECT year, COUNT(*) AS papers
    FROM papers
    GROUP BY year
),
growth AS (
    SELECT year,
           papers,
           LAG(papers) OVER (ORDER BY year) AS prev_year,
           ROUND(100.0 * (papers - LAG(papers) OVER (ORDER BY year))
                 / LAG(papers) OVER (ORDER BY year), 1) AS growth_pct
    FROM per_year
)
SELECT year, papers, prev_year, growth_pct
FROM growth
WHERE prev_year IS NOT NULL
ORDER BY year;


-- ---------------------------------------------------------------------------
-- Q7 [CTE + ОКОННАЯ ФУНКЦИЯ] Топ-5 ключевых слов в каждом году.
-- ---------------------------------------------------------------------------
WITH kw AS (
    SELECT p.year            AS year,
           k.keyword         AS keyword,
           COUNT(DISTINCT p.id) AS papers
    FROM papers p
    JOIN paper_keywords k ON k.paper_id = p.id
    GROUP BY p.year, k.keyword
),
ranked AS (
    SELECT year, keyword, papers,
           RANK() OVER (PARTITION BY year ORDER BY papers DESC) AS rnk
    FROM kw
)
SELECT year, rnk, keyword, papers
FROM ranked
WHERE rnk <= 5
ORDER BY year DESC, rnk;


-- ---------------------------------------------------------------------------
-- Q8 [JOIN + агрегация] Самые продуктивные авторы (>=100 статей).
-- ---------------------------------------------------------------------------
SELECT a.name                       AS author,
       COUNT(DISTINCT a.paper_id)   AS papers,
       MIN(p.year)                  AS first_year,
       MAX(p.year)                  AS last_year
FROM authors a
JOIN papers p ON p.id = a.paper_id
GROUP BY a.name
HAVING COUNT(DISTINCT a.paper_id) >= 100
ORDER BY papers DESC
LIMIT 20;
