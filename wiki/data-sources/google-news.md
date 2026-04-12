# Google News

Large-scale news article corpus collected via Google News RSS, providing **geopolitical and social signals** through sentiment analysis and event detection.

## Source

- **Method**: Google News RSS feed scraping
- **Coverage**: 15 countries × 8 migration-related topics = 120 country-topic combinations
- **Volume**: 170,754 articles collected → 104,333 valid (61.1% pipeline success rate)
- **Query definitions**: `src/collection/news_queries.json`

## Collection

The [News Scraper](modules/news-scraper) (`src/collection/news.py`) is an optimized async pipeline:

1. **Query dispatch**: For each country-topic pair, fetch Google News RSS results
2. **Batch URL decoding**: Decode Google News redirect URLs in batches of 20 (40–50% faster)
3. **Parallel extraction**: Download HTML with `httpx` async + extract content via `trafilatura`
4. **Fault tolerance**: Checkpoint every 25 articles; atomic JSON writes prevent corruption
5. **Adaptive throttling**: Detect 429 responses and back off dynamically

**Concurrency model**: Bounded with `Semaphore(12)` to avoid rate-limiting. HTML extraction uses `ProcessPoolExecutor(4)` to bypass the GIL.

Output: JSON files per country/year in `data/raw/news/` and `data/raw/news_json/`.

## Processing

Two-stage NLP pipeline via [NLP Enrichment](pipeline/nlp-enrichment):

| Stage | Tool | Output |
|-------|------|--------|
| JSON → Parquet | [Processing Module](modules/processing-module) (`src/processing/news.py`) | `data/processed/news/news_*.parquet` |
| Embedding | [Jina v5 Embeddings](models/jina-v5-embeddings) (TensorRT INT8, 768-dim) | `data/processed/news_embeddings/` |
| Clustering | [Event Clustering](analysis/event-clustering) (HDBSCAN via cuML GPU) | Cluster assignments |
| Labeling | [Flan-T5 Summarization](models/flan-t5-summarization) (TensorRT INT8) | 2–3 word cluster labels |
| Sentiment | [Sentiment Analysis](analysis/sentiment-analysis) (rule-based lexicon) | Score ∈ [−1, +1] per article |

## Key Findings

News sentiment is the **strongest individual predictor** of migration surges:
- Dominican Republic "first presidency" event: r = 0.617 at 3-month lag
- Cuba Trump policy coverage: r = −0.595
- 58 statistically significant event-visa signals found after [Multiple Comparison Correction](analysis/multiple-comparison-correction)

See [Event-Visa Findings](findings/event-visa-findings) for full results.

## See Also

- [News Scraper](modules/news-scraper) — Technical deep dive on the collection pipeline
- [NLP Enrichment](pipeline/nlp-enrichment) — The full NLP processing chain
- [Event Clustering](analysis/event-clustering) — How articles become labeled events
- [Sentiment Analysis](analysis/sentiment-analysis) — Scoring methodology
- [Event-Visa Findings](findings/event-visa-findings) — What the news data revealed
