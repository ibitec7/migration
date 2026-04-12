# Data Collection

The ingestion layer that fetches raw data from four external sources into `data/raw/`. Designed for high-throughput, fault-tolerant, and reproducible operation.

## Architecture

```
src/collection/
├── config.py      → Concurrency settings, timeouts, batch sizes
├── news.py        → Async Google News scraper (170K+ articles)
├── visa.py        → Travel.State.Gov PDF/Excel downloader
├── encounter.py   → CBP CSV downloader
├── trends.py      → HF Hub snapshot sync
├── hf_sync.py     → Bidirectional Hugging Face sync
└── utils.py       → Retry logic, logging, throttling
```

## Data Streams

| Stream | Module | Method | Output |
|--------|--------|--------|--------|
| [Visa Data](data-sources/visa-data) | `visa.py` | Scrape + parallel download PDFs | `data/raw/visa/` |
| [Encounter Data](data-sources/encounter-data) | `encounter.py` | Async semaphore-controlled CSVs | `data/raw/encounter/` |
| [Google News](data-sources/google-news) | `news.py` | Async RSS + batch URL decode | `data/raw/news/` |
| [Google Trends](data-sources/google-trends) | `trends.py` | HF Hub `snapshot_download()` | `data/raw/trends/` |

## Design Principles

### Bounded Concurrency
All network operations use semaphore-limited concurrency to avoid rate-limiting:
- News scraping: `Semaphore(12)` with adaptive throttling on 429 responses
- HTTP connection pooling (max 16 concurrent)

### Retry Logic
The `retry_errors()` decorator in `utils.py` implements:
- Exponential backoff with jitter (prevents thundering herd)
- Distinguishes retryable errors (429, 503, 504) from non-retryable (400, 403, 404)
- Parses `Retry-After` headers when present

### Checkpoint Recovery
News collection checkpoints progress every 25 articles with atomic JSON writes, enabling resumption after interruption.

### Batch Processing
Google News URL decoding is batched (20 URLs per request), achieving 40–50% fewer HTTP calls. HTML extraction uses `ProcessPoolExecutor(4)` to bypass the GIL.

## Running

```bash
# Full pipeline via run.sh
bash run.sh

# Individual collectors
python -m src.collection.visa
python -m src.collection.encounter
python -m src.collection.trends
```

Or bootstrap everything from Hugging Face:
```bash
python -m src.main bootstrap --org sdsc2005-migration
```

## See Also

- [Collection Module](modules/collection-module) — Full source code documentation
- [News Scraper](modules/news-scraper) — Deep dive on news collection
- [HF Sync](modules/hf-sync) — Hugging Face sync mechanism
- [Data Processing](pipeline/data-processing) — Next stage in the pipeline
- [Reproducibility](infrastructure/reproducibility) — End-to-end setup instructions
