---
title: SurgeJointLoss
aliases: [Surge Joint Loss, Dual-Objective Loss, Custom Loss]
tags: [model, loss-function, surge, training]
created: 2026-04-12
---

# SurgeJointLoss

A custom dual-objective loss function that trains the [[lstm|LSTM]] to simultaneously optimize for **volume accuracy** and **crisis detection**.

## Motivation

Standard regression losses (MSE, Huber) optimize volume accuracy but don't explicitly penalize missed surges. The system needs high recall for genuine crises — a missed surge is far more costly than a false alarm.

SurgeJointLoss addresses this by combining:
1. A **regression term** for accurate volume prediction
2. A **classification term** for surge detection

## Formula

$$
\mathcal{L} = \alpha \cdot \text{Huber}(\hat{y}, y) + (1 - \alpha) \cdot \text{BCE}(\hat{y} - 1.5\sigma, \mathbb{1}[\text{surge}])
$$

| Component | Weight | Purpose |
|-----------|--------|---------|
| **Huber Loss** | α = 0.6 | Accurate volume regression, robust to outliers |
| **BCE Loss** | 1 − α = 0.4 | Explicit penalty for missed/false-alarm surges |

## Surge Threshold

A data point is labeled as a surge when:

$$
y > \mu_{rolling} + 1.5\sigma_{rolling}
$$

This threshold is computed from the training data's rolling statistics and produces binary labels for the BCE term.

## Effect on Training

- The model learns to optimize for both volume RMSE **and** surge F1 simultaneously
- BCE term acts as a regularizer that pulls predictions toward the correct side of the surge boundary
- Result: [[lstm|LSTM]] achieves highest precision (0.97 at Lead 1) among individual models

## Comparison with Other Losses

| Loss | Used By | Volume Accuracy | Surge Detection |
|------|---------|----------------|-----------------|
| **SurgeJointLoss** | [[lstm]] | Good (Huber) | Explicit (BCE) |
| MSELoss | [[transformer]] | Best (MSE) | Implicit only |
| N/A (sklearn) | [[random-forest]] | Good | Implicit only |

## Implementation

`SurgeJointLoss` class in `src/models/surge_model.py`:

```python
class SurgeJointLoss(nn.Module):
    # α * HuberLoss + (1-α) * BCEWithLogitsLoss
    # Threshold: 1.5 std above rolling mean
```

## See Also

- [[lstm]] — The model that uses this loss
- [[surge-detection]] — How surges are defined
- [[training-pipeline]] — Training context
- [[model-performance]] — Impact on results
