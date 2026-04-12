# Horizon-Aware Ensemble

The production model that **dynamically re-weights** three architectures based on forecast horizon, exploiting each model's strengths at different time scales.

## Rationale

No single model dominates across all horizons:
- [Random Forest](models/random-forest) excels at Lead 1–2 (tree-based local patterns)
- [LSTM](models/lstm) provides consistent mid-range performance with country awareness
- [Transformer](models/transformer) captures slow geopolitical shifts visible at Lead 5–6

## Weight Table

| Horizon | RF | Transformer | LSTM | Rationale |
|---------|----|-------------|------|-----------|
| Lead 1 | 60% | 10% | 30% | Tree + LSTM maximize short-term precision |
| Lead 2 | 50% | 20% | 30% | RF still dominant; Transformer rising |
| Lead 3 | 30% | 40% | 30% | Balanced transition |
| Lead 4 | 20% | 60% | 20% | Transformer increasingly critical |
| Lead 5 | 10% | 80% | 10% | Transformer captures slow geopolitical shifts |
| Lead 6 | 5% | 85% | 10% | Maximum Transformer dominance for recall |

## Prediction Formula

$$
\hat{y}_h = w_{RF}(h) \cdot \hat{y}_{RF}(h) + w_{LSTM}(h) \cdot \hat{y}_{LSTM}(h) + w_{TF}(h) \cdot \hat{y}_{TF}(h)
$$

Where $h$ is the horizon (1–6) and weights sum to 1.0 at each horizon.

## Ensemble Performance (OOT Test Set)

| Metric | Lead 1 | Lead 2 | Lead 3 | Lead 4 | Lead 5 | Lead 6 |
|--------|--------|--------|--------|--------|--------|--------|
| F1 | 0.96 | 0.95 | 0.93 | 0.91 | 0.88 | 0.86 |
| Precision | 0.96 | 0.94 | 0.92 | 0.89 | 0.84 | 0.80 |
| Recall | 0.96 | 0.95 | 0.94 | 0.93 | 0.92 | 0.92 |

**Critical achievement**: Recall stays **> 92% across all horizons** — the system catches 9+ of every 10 genuine crises.

## Surge Detection

Surge flag is set when predicted volume exceeds:

$$
\hat{y}_h > \mu_{rolling} + 1.5\sigma_{rolling}
$$

See [Surge Detection](analysis/surge-detection) for the full threshold methodology.

## Implementation

`MigrationSurgeEnsemble` in `src/models/inference.py` — see [Inference Pipeline](pipeline/inference-pipeline) for the production flow.

## See Also

- [Random Forest](models/random-forest), [LSTM](models/lstm), [Transformer](models/transformer) — Individual architectures
- [SurgeJointLoss](models/surge-joint-loss) — LSTM's custom loss function
- [Training Pipeline](pipeline/training-pipeline) — How all models are trained
- [Inference Pipeline](pipeline/inference-pipeline) — Production prediction flow
- [Model Performance](findings/model-performance) — Detailed evaluation
