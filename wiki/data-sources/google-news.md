---
title: Google News
aliases: [News Data, News Articles, Google News API]
tags: [data-source, signal, news, nlp]
created: 2026-04-12
---

# Google News

Large-scale news article corpus collected via Google News RSS, providing **geopolitical and social signals** through sentiment analysis and event detection.

## Source

- **Method**: Google News RSS feed scraping
- **Coverage**: 15 countries × 8 migration-related topics = 120 country-topic combinations
- **Volume**: 170,754 articles collected → 104,333 valid (61.1% pipeline success rate)
- **Query definitions**: `src/collection/news_queries.json`

## Collection

The [[news-scraper]] (`src/collection/news.py`) is an optimized async pipeline:

1. **Query dispatch**: For each country-topic pair, fetch Google News RSS results
2. **Batch URL decoding**: Decode Google News redirect URLs in batches of 20 (40–50% faster)
3. **Parallel extraction**: Download HTML with `httpx` async + extract content via `trafilatura`
4. **Fault tolerance**: Checkpoint every 25 articles; atomic JSON writes prevent corruption
5. **Adaptive throttling**: Detect 429 responses and back off dynamically

**Concurrency model**: Bounded with `Semaphore(12)` to avoid rate-limiting. HTML extraction uses `ProcessPoolExecutor(4)` to bypass the GIL.

Output: JSON files per country/year in `data/raw/news/` and `data/raw/news_json/`.

## Processing

Two-stage NLP pipeline via [[nlp-enrichment]]:

| Stage | Tool | Output |
|-------|------|--------|
| JSON → Parquet | [[processing-module]] (`src/processing/news.py`) | `data/processed/news/news_*.parquet` |
| Embedding | [[jina-v5-embeddings]] (TensorRT INT8, 768-dim) | `data/processed/news_embeddings/` |
| Clustering | [[event-clustering]] (HDBSCAN via cuML GPU) | Cluster assignments |
| Labeling | [[flan-t5-summarization]] (TensorRT INT8) | 2–3 word cluster labels |
| Sentiment | [[sentiment-analysis]] (rule-based lexicon) | Score ∈ [−1, +1] per article |

## Key Findings

News sentiment is the **strongest individual predictor** of migration surges:
- Dominican Republic "first presidency" event: r = 0.617 at 3-month lag
- Cuba Trump policy coverage: r = −0.595
- 58 statistically significant event-visa signals found after [[multiple-comparison-correction]]

See [[event-visa-findings]] for full results.

## See Also

- [[news-scraper]] — Technical deep dive on the collection pipeline
- [[nlp-enrichment]] — The full NLP processing chain
- [[event-clustering]] — How articles become labeled events
- [[sentiment-analysis]] — Scoring methodology
- [[event-visa-findings]] — What the news data revealed
