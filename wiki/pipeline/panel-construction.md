---
title: Panel Construction
aliases: [Feature Engineering, Train Panel, Panel Data]
tags: [pipeline, feature-engineering, panel-data]
created: 2026-04-12
---

# Panel Construction

The feature engineering stage that merges all data sources into a single monthly × country panel dataset ready for model training.

## Implementation

`src/processing/build_panel.py` — see [[build-panel-detail]] for technical deep dive.

## Input Sources

| Source | Granularity | Module |
|--------|-------------|--------|
| [[visa-data]] | Monthly issuances by country | `visa_master.parquet` |
| [[exchange-rates]] | Monthly average REER | OECD data |
| [[google-news]] | Event counts per month by country | `news_embeddings_labeled/` |
| [[google-trends]] | Monthly keyword interest | Trend Parquets |

## Feature Schema

### Lag Features (18 total)
For each of 3 core signals, 6 months of history:

| Signal | Features |
|--------|----------|
| Visa volume | `visa_lag_1` through `visa_lag_6` |
| Exchange rate | `exchange_rate_lag_1` through `exchange_rate_lag_6` |
| News events | `news_events_lag_1` through `news_events_lag_6` |

### Lead Targets (6 total)
Forward-looking prediction targets:
- `target_visa_lead_1` through `target_visa_lead_6`

### Dimensions

| Metric | Value |
|--------|-------|
| Countries | 15 |
| Months | ~120 (2017–2025) |
| Total observations | ~1,800 |
| Features per observation | 18 lag + metadata |
| Targets per observation | 6 leads |

## Missing Data Strategy

- **Exchange rates**: Forward/backward fill for gaps; zero-fill for countries without REER data (9 of 15)
- **News events**: Zero-fill for months with no articles
- **Visa lags**: Drop observations with insufficient history (first 6 months per country)

## Output

`data/processed/train_panel.parquet` — consumed by [[training-pipeline]] and [[inference-pipeline]].

## Sequential Input Shape

The [[models-module|model layer]] reshapes lag features into sequences:

```
(batch_size, 6, 3) → 6 time steps × 3 signals [visa, exchange_rate, news_events]
```

This is handled by `build_sequential_tensors()` in `src/models/surge_model.py` with `StandardScaler` normalization.

## See Also

- [[build-panel-detail]] — Technical deep dive (forward-fill, edge cases)
- [[data-processing]] — Previous stage
- [[training-pipeline]] — Next stage (model training)
- [[lead-lag-analysis]] — Statistical basis for the lag window choice
