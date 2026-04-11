---
title: Models Module
aliases: [src/models, Models Package]
tags: [module, models, source-code, training, inference]
created: 2026-04-12
---

# Models Module

`src/models/` — ML architectures, training, inference, and TensorRT engine wrappers.

## Files

| File | Purpose | Key Classes/Functions |
|------|---------|---------------------|
| `surge_model.py` | Core ML architectures | `MigrationLSTM`, `MigrationTransformer`, `SurgeJointLoss`, `build_sequential_tensors()` |
| `train_and_evaluate.py` | Training pipeline | Full train loop: RF → Transformer → LSTM → Ensemble → Evaluate |
| `inference.py` | Production prediction | `MigrationSurgeEnsemble` — loads all models, horizon-aware weighting |
| `surge_metrics.py` | Surge evaluation | `detect_surge()`, `evaluate_surge_performance()` |
| `embedding.py` | Jina v5 embedding runner | Batch embed articles, append to Parquets, LZ4 compression |
| `jinav5_engine.py` | Jina TensorRT wrapper | `JinaV5EmbeddingTrtModel` — CUDA memory, dynamic shapes |
| `flant5_engine.py` | Flan-T5 TensorRT wrapper | `TensorRTFlanT5Engine` — tokenize, beam search, generate |
| `led_engine.py` | LED TensorRT wrapper | `allenai/led-base-16384` — long-context labeling |
| `utils.py` | Jina ONNX export | Export Jina-v5-nano to ONNX with mean pooling |

## Model Architectures (surge_model.py)

### MigrationLSTM
```
Input: (B, 6, 3) + country_id → LSTM(2 layers, H=64) + Embedding(8) → FC → (B, 6)
```

### MigrationTransformer
```
Input: (B, 6, 3) → Linear(64) + PosEnc → TransformerEncoder(2L, 4H) → FC → (B, 6)
```

### SurgeJointLoss
```
L = 0.6 × Huber(ŷ, y) + 0.4 × BCE(ŷ - 1.5σ, surge_flag)
```

## Training Flow (train_and_evaluate.py)

```
1. Load train_panel.parquet → temporal split (≤2022 / 2023+)
2. Train 6 × cuML Random Forest (one per lead month)
3. Train MigrationTransformer (MSE, 20 epochs)
4. Train MigrationLSTM (SurgeJointLoss, 25 epochs)
5. Ensemble with horizon-aware weights
6. Evaluate all 4 on OOT test set
7. Save artifacts to src/models/trained_models/
```

## Trained Model Artifacts

```
src/models/trained_models/
├── lstm.pth, transformer.pth
├── rf_lead_{1..6}.joblib
├── scaler_x.joblib, scaler_y.joblib
└── country_map.json
```

## TensorRT Engines

```
src/models/tensor-rt/
├── flan-t5-large/int8_wo_cpu/1-gpu/
└── (jina, led engine paths configured externally)
```

## See Also

- [[training-pipeline]] — Pipeline context
- [[inference-pipeline]] — Production flow
- [[random-forest]], [[lstm]], [[transformer]] — Architecture details
- [[horizon-aware-ensemble]] — Ensemble weights
- [[surge-joint-loss]] — Custom loss function
- [[tensorrt-engines]] — Engine deep dive
