# Training Pipeline

The model training and evaluation stage that produces four forecasting architectures and evaluates them with out-of-time (OOT) validation.

## Implementation

`src/models/train_and_evaluate.py` — see [Models Module](modules/models-module) for code reference.

## Train/Test Split

**Out-of-time strategy** prevents data leakage:

| Set | Period | Purpose |
|-----|--------|---------|
| Train | Data through December 2022 | Fit model parameters |
| Test | Rolling predictions on 2023–2025 | Evaluate on unseen future data |

## Training Sequence

### 1. Random Forest (6 models)

One model per lead month (Lead 1 through Lead 6):
- **Library**: cuML (GPU-accelerated)
- **Trees**: 50 per model
- **Input**: Flat feature vector from [Panel Construction](pipeline/panel-construction)
- **See**: [Random Forest](models/random-forest)

### 2. Transformer (1 model)

- **Architecture**: [MigrationTransformer](models/transformer) — 4-head, 2-layer encoder
- **Loss**: MSELoss
- **Epochs**: 20
- **Input**: (batch, 6, 3) sequential tensor
- **Output**: 6-month predictions simultaneously

### 3. LSTM (1 model)

- **Architecture**: [MigrationLSTM](models/lstm) — 2-layer, 64 hidden, country embeddings
- **Loss**: [SurgeJointLoss](models/surge-joint-loss) (0.6 × Huber + 0.4 × BCE)
- **Epochs**: 25
- **Input**: (batch, 6, 3) sequential tensor + country ID
- **Output**: 6-month predictions simultaneously

### 4. Ensemble Aggregation

The [Horizon-Aware Ensemble](models/horizon-aware-ensemble) combines all three architectures with dynamic weights that shift based on forecast horizon.

## Evaluation

All models evaluated on the OOT test set using:
- **Volume accuracy**: RMSE at each lead horizon
- **Surge detection**: Precision, Recall, F1 (surge = > 1.5σ above rolling mean)

See [Model Performance](findings/model-performance) for full results.

## Artifacts

```
src/models/trained_models/
├── lstm.pth                → LSTM state dict
├── transformer.pth         → Transformer state dict
├── rf_lead_1.joblib ...    → Random Forest models (one per lead)
├── scaler_x.joblib         → Feature StandardScaler
├── scaler_y.joblib         → Target StandardScaler
└── country_map.json        → Country name → embedding index
```

## See Also

- [Panel Construction](pipeline/panel-construction) — Input data
- [Random Forest](models/random-forest), [LSTM](models/lstm), [Transformer](models/transformer) — Individual architectures
- [Horizon-Aware Ensemble](models/horizon-aware-ensemble) — How models are combined
- [SurgeJointLoss](models/surge-joint-loss) — Custom LSTM loss function
- [Model Performance](findings/model-performance) — Evaluation results
- [Inference Pipeline](pipeline/inference-pipeline) — Production deployment
