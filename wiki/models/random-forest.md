# Random Forest

GPU-accelerated Random Forest ensemble trained via cuML, producing **one model per forecast horizon** (6 total). Excels at short-term predictions (Lead 1–2).

## Architecture

| Parameter | Value |
|-----------|-------|
| Library | cuML (NVIDIA RAPIDS) |
| Trees per model | 50 |
| Models | 6 (one per Lead 1–6) |
| Input | Flat feature vector from [Panel Construction](pipeline/panel-construction) |
| Training | GPU-accelerated |

Unlike the sequence models ([LSTM](models/lstm), [Transformer](models/transformer)), each RF model is trained independently on a specific lead target, receiving the full 18-feature lag vector as a flat input.

## Strengths

- **Best short-term F1**: 0.97 at Lead 1
- **Highest recall** across short horizons — catches nearly all genuine surges
- **Robust**: No hyperparameter sensitivity, handles missing features gracefully
- **Fast training**: GPU acceleration via cuML

## Weaknesses

- **RMSE degrades** at Lead 4–6 (long-horizon patterns are harder for tree ensembles)
- **No sequence modeling**: Treats lag features independently rather than as a temporal sequence
- **6 separate models**: No parameter sharing across horizons

## Role in Ensemble

The [Horizon-Aware Ensemble](models/horizon-aware-ensemble) gives RF the highest weight at short horizons:

| Lead | RF Weight |
|------|-----------|
| 1 | 60% |
| 2 | 50% |
| 3 | 30% |
| 4 | 20% |
| 5 | 10% |
| 6 | 5% |

## Performance

| Lead | F1 | Precision | Recall | RMSE |
|------|-----|-----------|--------|------|
| 1 | 0.97 | 0.96 | 0.97 | Best at lead 1 |
| 6 | ~0.82 | ~0.78 | ~0.87 | Degrades |

See [Model Performance](findings/model-performance) for full comparison.

## Artifacts

```
src/models/trained_models/
├── rf_lead_1.joblib
├── rf_lead_2.joblib
├── rf_lead_3.joblib
├── rf_lead_4.joblib
├── rf_lead_5.joblib
└── rf_lead_6.joblib
```

## See Also

- [Horizon-Aware Ensemble](models/horizon-aware-ensemble) — How RF is weighted in production
- [LSTM](models/lstm) — Mid-horizon alternative
- [Transformer](models/transformer) — Long-horizon alternative
- [Training Pipeline](pipeline/training-pipeline) — Training process
- [GPU Acceleration](infrastructure/gpu-acceleration) — cuML infrastructure
