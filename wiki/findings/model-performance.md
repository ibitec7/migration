# Model Performance

Comprehensive evaluation of all four architectures on the out-of-time (OOT) test set (train ≤ 2022, test 2023–2025).

## Ensemble Results (Production Model)

| Metric | Lead 1 | Lead 2 | Lead 3 | Lead 4 | Lead 5 | Lead 6 |
|--------|--------|--------|--------|--------|--------|--------|
| **F1 Score** | 0.96 | 0.95 | 0.93 | 0.91 | 0.88 | 0.86 |
| **Precision** | 0.96 | 0.94 | 0.92 | 0.89 | 0.84 | 0.80 |
| **Recall** | 0.96 | 0.95 | 0.94 | 0.93 | 0.92 | 0.92 |
| **RMSE** | 277–298 | ~310 | 346–351 | ~370 | ~390 | 401–424 |

**Critical achievement**: Recall > 92% across all horizons — the system catches 9+ of every 10 genuine crises while maintaining ≤ 20% false alarm rate.

## Individual Model Comparison

### F1 Score by Horizon

| Model | Lead 1 | Lead 3 | Lead 6 | Best At |
|-------|--------|--------|--------|---------|
| [RF](models/random-forest) | **0.97** | 0.91 | 0.82 | Short horizons |
| [LSTM](models/lstm) | 0.95 | 0.92 | 0.84 | Mid horizons |
| [Transformer](models/transformer) | 0.93 | 0.92 | **0.87** | Long horizons |
| **Ensemble** | 0.96 | **0.93** | 0.86 | All horizons |

### Precision vs. Recall Tradeoff

| Model | Lead 1 Precision | Lead 1 Recall | Lead 6 Precision | Lead 6 Recall |
|-------|-----------------|---------------|-----------------|---------------|
| RF | 0.96 | **0.97** | 0.78 | 0.87 |
| LSTM | **0.97** | 0.85 | 0.82 | 0.79 |
| Transformer | 0.91 | 0.94 | 0.82 | **0.90** |
| Ensemble | 0.96 | 0.96 | 0.80 | **0.92** |

### Interpretation

- **RF** excels at not missing surges (highest recall at Lead 1) but precision drops at long horizons
- **LSTM** has the fewest false alarms (highest precision at Lead 1) thanks to [SurgeJointLoss](models/surge-joint-loss)
- **Transformer** maintains recall at long horizons where other models degrade
- **Ensemble** blends strengths: borrows RF's short-range recall and Transformer's long-range stability

## Volume RMSE

| Model | Lead 1 | Lead 3 | Lead 6 |
|-------|--------|--------|--------|
| RF | 277 | 346 | 424 |
| LSTM | 280 | 348 | 410 |
| Transformer | 298 | 351 | 401 |
| Ensemble | **277** | **346** | **401** |

RMSE naturally increases with horizon distance (predicting farther ahead is harder).

## Why Ensemble Wins

The [Horizon-Aware Ensemble](models/horizon-aware-ensemble) exploits a key insight: **no single model dominates across all horizons**.

```
Lead 1-2: RF dominance  → Short-term local patterns
Lead 3-4: Balanced      → Transition zone
Lead 5-6: TF dominance  → Long-range geopolitical patterns
```

## Visualizations

2 plots in `data/plots/model_performance/`:
- F1 score by horizon (all models)
- Precision/recall comparison chart

## Operational Interpretation

| Scenario | Lead 1 | Lead 6 |
|----------|--------|--------|
| If the model predicts "surge" | 96% chance it's genuine | 80% chance it's genuine |
| If a real surge happens | 96% chance model catches it | 92% chance model catches it |
| Misses per 10 genuine surges | < 1 | < 1 |
| False alarms per 10 alerts | < 1 | 2 |

The system is **operationally useful at all horizons**, with a false alarm rate that stays manageable even at 6-month forecasts.

## See Also

- [Horizon-Aware Ensemble](models/horizon-aware-ensemble) — Ensemble architecture and weights
- [Random Forest](models/random-forest), [LSTM](models/lstm), [Transformer](models/transformer) — Individual model details
- [Surge Detection](analysis/surge-detection) — How surges are defined
- [Training Pipeline](pipeline/training-pipeline) — Training methodology
- [Project Overview](project-overview) — Research questions answered
