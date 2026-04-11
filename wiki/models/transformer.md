---
title: Transformer
aliases: [MigrationTransformer, Transformer Model, Attention Model]
tags: [model, transformer, pytorch, attention, long-horizon]
created: 2026-04-12
---

# Transformer

A 2-layer Transformer encoder with multi-head attention, excelling at **long-horizon forecasting** (Lead 5–6) where sequential patterns span many months.

## Architecture

```
Input: (batch, 6, 3)  →  Linear Embedding(64)  →  Positional Encoding
                                    ↓
                    TransformerEncoder(2 layers, 4 heads)
                                    ↓
                            FC Head  →  6 lead predictions
```

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Encoder layers | 2 |
| Attention heads | 4 |
| Model dimension (d_model) | 64 |
| Positional encoding | Learnable |
| Input features | 3 per timestep |
| Sequence length | 6 |
| Output | 6 simultaneous lead predictions |
| Loss | MSELoss (pure regression) |
| Epochs | 20 |

## Strengths

- **Best long-horizon recall**: Maintains 90% recall at Lead 6
- **Best long-horizon F1**: 0.87 at Lead 6
- **Attention mechanism** captures slow-moving geopolitical patterns that unfold over months
- **Simultaneous multi-horizon** output from a single forward pass

## Weaknesses

- **Lower short-horizon precision**: Attention can overweight distant patterns at nearby horizons
- **Pure regression loss**: No explicit surge objective (unlike [[lstm|LSTM]]'s [[surge-joint-loss]])
- **Requires more data**: Attention benefits from larger datasets

## Role in Ensemble

The [[horizon-aware-ensemble]] gives the Transformer increasing weight at longer horizons:

| Lead | Transformer Weight |
|------|--------------------|
| 1 | 10% |
| 2 | 20% |
| 3 | 40% |
| 4 | 60% |
| 5 | 80% |
| 6 | 85% |

At Lead 6, the Transformer dominates the ensemble (85% weight).

## Artifacts

```
src/models/trained_models/
└── transformer.pth  → Model state dict
```

## See Also

- [[lstm]] — Alternative sequence model with country embeddings
- [[random-forest]] — Short-horizon alternative
- [[horizon-aware-ensemble]] — Ensemble integration
- [[training-pipeline]] — Training process
- [[model-performance]] — Comparative results
