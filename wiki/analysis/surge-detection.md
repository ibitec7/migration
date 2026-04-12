# Surge Detection

Methods for identifying **significant spikes** in migration volume, used both in statistical analysis and as model training labels.

## Two Definitions

The system uses different surge definitions for different purposes:

### 1. Analysis Surge (Lead-Lag Studies)

Used in [Lead-Lag Analysis](analysis/lead-lag-analysis) and [Event-Visa Findings](findings/event-visa-findings):

A month is flagged as a surge if **either**:
- Volume > **75th percentile** of the country's historical distribution, **OR**
- Month-over-month growth > **30%**

Implementation: `detect_surges_list()` in `src/analysis/events.py`

### 2. Model Surge (Training Labels)

Used in [SurgeJointLoss](models/surge-joint-loss) and [Model Performance](findings/model-performance) evaluation:

A month is flagged as a surge if:

$$
y > \mu_{rolling} + 1.5\sigma_{rolling}
$$

Where $\mu$ and $\sigma$ are computed over a rolling window of the training data.

Implementation: `detect_surge()` in `src/models/surge_metrics.py`

## Why Two Definitions?

| Context | Definition | Rationale |
|---------|-----------|-----------|
| Lead-lag analysis | 75th pctile or 30% MoM | Captures diverse surge patterns, more inclusive |
| Model training | 1.5σ above rolling mean | Statistical rigor, adapts to trend, fewer false positives |

## Evaluation Metrics

Surge predictions are evaluated with:
- **Precision**: % of predicted surges that are genuine
- **Recall**: % of genuine surges that are caught
- **F1**: Harmonic mean of precision and recall

Implementation: `evaluate_surge_performance()` in `src/models/surge_metrics.py`

## Exchange Rate Shocks

A separate definition for [exchange rate](data-sources/exchange-rates) volatility:
- Flagged when month-over-month REER change > **80th percentile**
- Implementation: `detect_exchange_shocks()` in `src/analysis/exchange_rate.py`

## See Also

- [SurgeJointLoss](models/surge-joint-loss) — How surge labels train the LSTM
- [Lead-Lag Analysis](analysis/lead-lag-analysis) — Where analysis surges are used
- [Model Performance](findings/model-performance) — Surge detection evaluation results
- [Horizon-Aware Ensemble](models/horizon-aware-ensemble) — Production surge flagging
