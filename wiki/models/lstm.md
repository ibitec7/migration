# LSTM

A 2-layer LSTM with learned country embeddings and a custom [SurgeJointLoss](models/surge-joint-loss), providing **country-aware sequence modeling** for migration volume forecasting.

## Architecture

```
Input: (batch, 6, 3)  →  LSTM(2 layers, hidden=64)  →  FC  →  6 lead predictions
                  ↑
        Country ID → Embedding(8-dim) → Concat per timestep
```

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Layers | 2 LSTM layers |
| Hidden size | 64 |
| Country embedding | 8-dimensional learned |
| Input features | 3 per timestep (visa, exchange_rate, news_events) |
| Sequence length | 6 (months of history) |
| Output | 6 simultaneous lead predictions |
| Loss | [SurgeJointLoss](models/surge-joint-loss) (0.6 × Huber + 0.4 × BCE) |
| Epochs | 25 |

## Country Embeddings

Each of the 15 countries gets a learned 8-dimensional embedding vector. This is concatenated with the 3 input features at each timestep, allowing the model to learn country-specific dynamics (e.g., Cuba's political isolation vs. Mexico's border proximity).

The country → index mapping is stored in `country_map.json`.

## Strengths

- **Highest precision**: 0.97 at Lead 1 — fewest false alarms
- **Country awareness**: Embeddings capture cross-country behavioral differences
- **Dual objective**: [SurgeJointLoss](models/surge-joint-loss) explicitly optimizes for both volume accuracy AND crisis detection

## Weaknesses

- **Lower recall**: 85% at Lead 1 → 79% at Lead 6 (misses some surges)
- **Training complexity**: SurgeJointLoss requires surge threshold tuning (1.5σ)
- **Sequential inference**: Slower than RF for single predictions

## Role in Ensemble

The [Horizon-Aware Ensemble](models/horizon-aware-ensemble) gives LSTM consistent but moderate weight:

| Lead | LSTM Weight |
|------|-------------|
| 1 | 30% |
| 2 | 30% |
| 3 | 30% |
| 4 | 20% |
| 5 | 10% |
| 6 | 10% |

## Artifacts

```
src/models/trained_models/
├── lstm.pth           → Model state dict
└── country_map.json   → Country name → embedding index
```

## See Also

- [SurgeJointLoss](models/surge-joint-loss) — Custom loss function
- [Transformer](models/transformer) — Alternative sequence model
- [Random Forest](models/random-forest) — Short-horizon alternative
- [Horizon-Aware Ensemble](models/horizon-aware-ensemble) — Ensemble integration
- [Training Pipeline](pipeline/training-pipeline) — Training process
- [Models Module](modules/models-module) — Source code: `surge_model.py`
