# News Scraper

Deep dive into `src/collection/news.py` — the high-throughput async news article collection pipeline that fetches 170K+ articles from Google News.

## Architecture

```
Google News RSS → Batch URL Decode → Async Download → HTML Extraction → JSON Output
```

## Key Components

### 1. Google News Decoding

Google News URLs are redirect links that must be decoded to get the real article URL.

- `get_decoding_params()` — Fetches session signature/timestamp for decoding
- `decode_urls_batch()` — Decodes 20 URLs per request (40–50% fewer HTTP calls than individual decoding)

### 2. Parallel Download

- `download_with_retries()` — Downloads with exponential backoff
- Bounded concurrency: `asyncio.Semaphore(12)`
- Connection pooling: max 16 concurrent via `httpx.AsyncClient`
- Adaptive throttling: Detects 429 responses and increases delay

### 3. Content Extraction

- **Library**: `trafilatura` for HTML → Markdown conversion
- **Parallelism**: `ProcessPoolExecutor(4)` bypasses GIL for CPU-bound HTML parsing
- Fallback chain for article date extraction:
  1. YAML response field
  2. RFC 2822 header
  3. `published_parsed` attribute

### 4. Checkpoint & Recovery

- Progress saved every 25 articles
- **Atomic JSON writes**: Write to temp file → rename (prevents corruption on crash)
- Resumption: Skips already-downloaded articles on restart

## Configuration (`config.py`)

| Setting | Value | Purpose |
|---------|-------|---------|
| `MAX_CONCURRENT` | 16 | HTTP connection pool size |
| `SEMAPHORE_LIMIT` | 12 | Async concurrency cap |
| `BATCH_SIZE` | 20 | URLs per decode request |
| `CHECKPOINT_INTERVAL` | 25 | Articles between saves |
| Retryable codes | 429, 503, 504 | Trigger exponential backoff |
| Non-retryable codes | 400, 403, 404 | Fail immediately |

## Output

JSON files per country/year in `data/raw/news/` with fields:
- `title`, `headline`, `status_code`, `response` (HTML/Markdown), `date`

## Performance

- **10–15× faster** than sequential pipeline (batch decode + async + multiprocess extraction)
- 170,754 articles collected, 104,333 valid (61.1% success rate)
- Failures due to: paywalls, dead links, anti-bot measures, non-English content

## See Also

- [Google News](data-sources/google-news) — Data source context
- [Data Collection](pipeline/data-collection) — Broader ingestion architecture
- [Collection Module](modules/collection-module) — Full package reference
- [NLP Enrichment](pipeline/nlp-enrichment) — Downstream NLP processing
