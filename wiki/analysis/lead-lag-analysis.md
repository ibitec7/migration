---
title: Lead-Lag Analysis
aliases: [Lead Lag, Lagged Correlation, Temporal Offset Analysis]
tags: [analysis, lead-lag, correlation, pearson]
created: 2026-04-12
---

# Lead-Lag Analysis

The core statistical methodology that identifies **temporal offsets** between predictive signals and migration outcomes, answering [[project-overview|RQ3]]: which data modality is the strongest early indicator?

## Concept

A "lead-lag" relationship exists when changes in signal X systematically precede changes in target Y by a fixed number of months. If news sentiment drops 3 months before visa applications rise, that's a **lag of 3**.

## Methodology

For each (country, signal, lag) combination:

1. **Shift** the signal by 0–6 months relative to the target
2. Compute **Pearson correlation** ($r$) between shifted signal and target
3. Record the correlation strength and p-value
4. Apply [[multiple-comparison-correction|Benjamini-Hochberg FDR]] to control false discoveries

## Signal Types Analyzed

| Signal | Source | Analysis Module |
|--------|--------|----------------|
| News event counts | [[google-news]] | `event_visa_analysis.py` |
| News sentiment | [[sentiment-analysis]] | `event_visa_analysis.py` |
| Exchange rate (REER) | [[exchange-rates]] | `exchange_rate.py` |
| Google Trends keywords | [[google-trends]] | `trends_analysis.py` |

## Key Results

### News Events → Visas
- **58 significant signals** after FDR correction
- Strongest: Dominican Republic "first presidency" cluster at lag 3 (r = 0.617)
- See [[event-visa-findings]]

### Exchange Rates → Visas
- Dominican Republic: r = 0.498 at lag 2 (p = 2.6e-06)
- See [[exchange-rate-findings]]

### Google Trends → Visas/Encounters
- 55% of keywords significant, but weak out-of-sample translation
- See [[cross-correlation]]

## Output Files

```
data/processed/production_outputs/
├── event_visa_best_lags_review.csv
├── event_visa_lead_lag_results_review.csv
├── event_sentiment_best_lags_review.csv
├── event_sentiment_lead_lag_results_review.csv
├── exchange_visa_best_lags_review.csv
└── exchange_visa_correlation_report.md
```

## See Also

- [[multiple-comparison-correction]] — Statistical correction for many tests
- [[surge-detection]] — How surges are defined
- [[cross-correlation]] — CCF-based alternative analysis
- [[event-visa-findings]] — News results
- [[exchange-rate-findings]] — Exchange rate results
- [[analysis-module]] — Source code reference
