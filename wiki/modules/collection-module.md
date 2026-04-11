---
title: Collection Module
aliases: [src/collection, Collection Package]
tags: [module, collection, source-code]
created: 2026-04-12
---

# Collection Module

`src/collection/` — the data ingestion layer that fetches raw data from external sources.

## Files

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `config.py` | Centralized configuration | HTTP concurrency (max 16), timeouts, batch sizes, TRT settings |
| `news.py` | Google News RSS scraper | `get_decoding_params()`, `decode_urls_batch()`, `download_with_retries()` |
| `visa.py` | Travel.State.Gov downloader | Scrapes PDF/Excel links, parallel download |
| `encounter.py` | CBP CSV downloader | Async semaphore-controlled downloads |
| `trends.py` | HF Hub trends sync | `snapshot_download()` from `sdsc2005-migration/trends` |
| `hf_sync.py` | Bidirectional HF Hub sync | `download_defaults()`, `download_models()`, `upload_missing()` |
| `utils.py` | Shared utilities | `setup_logger()`, `retry_errors()` decorator |
| `news_queries.json` | Query configuration | 15 countries list for news search |

## Dependencies Between Files

```
config.py ← news.py (reads concurrency settings)
utils.py  ← news.py, visa.py, encounter.py (retry logic, logging)
news_queries.json ← news.py (country list)
hf_sync.py ← main.py (bootstrap command)
```

## Key Design Patterns

- **Async I/O**: `httpx` + `asyncio` with bounded semaphores
- **Exponential backoff**: Jitter-based retry (429/503/504 vs. 400/403/404 distinction)
- **Atomic writes**: JSON checkpoint files written atomically to prevent corruption
- **Connection pooling**: Max 16 concurrent HTTP connections

## See Also

- [[data-collection]] — Pipeline context
- [[news-scraper]] — Deep dive on news.py
- [[hf-sync]] — Deep dive on hf_sync.py
- [[main-entrypoint]] — CLI that invokes collection
- [[processing-module]] — Downstream consumer
