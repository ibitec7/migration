---
title: Inference Pipeline
aliases: [Production Inference, Prediction Pipeline, MigrationSurgeEnsemble]
tags: [pipeline, inference, production, ensemble]
created: 2026-04-12
---

# Inference Pipeline

The production prediction flow that loads trained models, runs ensemble inference, and produces 6-month surge forecasts.

## Implementation

`src/models/inference.py` — the `MigrationSurgeEnsemble` class.

## Input

For each country, a 6-month history window:

```
Shape: (6, 3) → 6 time steps × [visa_volume, exchange_rate, news_sentiment]
```

Data comes from the live [[panel-construction]] pipeline or precomputed panels.

## Ensemble Architecture

The `MigrationSurgeEnsemble` loads and combines three model families:

| Component | Files Loaded | Role |
|-----------|-------------|------|
| [[random-forest]] | `rf_lead_1.joblib` … `rf_lead_6.joblib` | Short-horizon precision |
| [[lstm]] | `lstm.pth` + `country_map.json` | Mid-horizon with country context |
| [[transformer]] | `transformer.pth` | Long-horizon pattern capture |
| Scalers | `scaler_x.joblib`, `scaler_y.joblib` | Feature/target normalization |

## Horizon-Aware Weighting

The ensemble dynamically adjusts model contributions based on the forecast horizon. See [[horizon-aware-ensemble]] for the full weight table.

```
Prediction(lead_h) = w_rf(h) × RF(h) + w_lstm(h) × LSTM(h) + w_tf(h) × TF(h)
```

## Output

For each country:
- **6 volume predictions**: Forecasted visa issuances at Lead 1 through Lead 6
- **6 surge flags**: Binary indicators (1 = predicted surge, 0 = normal)
- **Surge threshold**: > 1.5σ above rolling mean

## Production Flow

```
1. Load train_panel.parquet (or live data)
2. Extract most recent 6-month window per country
3. Scale features with scaler_x
4. Run RF, LSTM, Transformer inference
5. Blend predictions via horizon-aware weights
6. Inverse-scale with scaler_y
7. Apply surge threshold → binary flags
8. Output: country × horizon predictions
```

## See Also

- [[training-pipeline]] — How models are trained
- [[horizon-aware-ensemble]] — Weight table and rationale
- [[panel-construction]] — Input data structure
- [[model-performance]] — Production evaluation results
- [[models-module]] — Source code reference
