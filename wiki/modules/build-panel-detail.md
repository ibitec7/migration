---
title: Build Panel Detail
aliases: [Panel Builder, build_panel.py, Feature Construction]
tags: [module, deep-dive, panel, features, engineering]
created: 2026-04-12
---

# Build Panel Detail

Deep dive into `src/processing/build_panel.py` — the feature engineering module that merges all data sources into a single training panel.

## Input Sources

| Source | Aggregation | Module |
|--------|-------------|--------|
| Visa master | Monthly sum of issuances by country | `visa_master.parquet` |
| Exchange rates | Monthly average REER from OECD | External dataset |
| News events | Event count per country-month | `news_embeddings_labeled/` |

## Feature Construction

### Lag Features (Lookback)

For each signal, create 6 months of history:

```python
for signal in ['visa_volume', 'exchange_rate', 'news_events']:
    for lag in range(1, 7):
        panel[f'{signal}_lag_{lag}'] = panel.groupby('country')[signal].shift(lag)
```

This produces **18 lag features** (3 signals × 6 lags).

### Lead Targets (Forecast)

For each target horizon:

```python
for lead in range(1, 7):
    panel[f'target_visa_lead_{lead}'] = panel.groupby('country')['visa_volume'].shift(-lead)
```

This produces **6 lead targets**.

## Missing Data Handling

| Scenario | Strategy |
|----------|----------|
| Exchange rate gaps (within series) | Forward-fill, then backward-fill |
| Countries without REER data (9/15) | Zero-fill (no signal, no harm) |
| News events (months with no articles) | Zero-fill |
| Insufficient lag history (first 6 months) | Drop observation |
| Insufficient lead targets (last 6 months) | Drop observation |

## Output Schema

```
train_panel.parquet
├── country (str)
├── date (datetime)
├── visa_volume (float)
├── exchange_rate (float)
├── news_events (float)
├── visa_lag_1 ... visa_lag_6
├── exchange_rate_lag_1 ... exchange_rate_lag_6
├── news_events_lag_1 ... news_events_lag_6
├── target_visa_lead_1 ... target_visa_lead_6
└── ~1,800 observations (15 countries × ~120 months, minus edge effects)
```

## Sequential Tensor Reshaping

The [[models-module]] (`surge_model.py`) reshapes the flat lag vector into sequences for LSTM/Transformer:

```python
def build_sequential_tensors(panel):
    # Reshape 18 features into (batch, 6, 3):
    # 6 time steps × [visa, exchange_rate, news_events]
    # Also fits StandardScaler and returns scalers for inverse transform
```

## See Also

- [[panel-construction]] — Pipeline context
- [[data-processing]] — Previous stage
- [[training-pipeline]] — Next stage
- [[processing-module]] — Full package reference
