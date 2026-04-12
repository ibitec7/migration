# Lead-Lag Analysis

The core statistical methodology that identifies **temporal offsets** between predictive signals and migration outcomes, answering [RQ3](project-overview): which data modality is the strongest early indicator?

## Concept

A "lead-lag" relationship exists when changes in signal X systematically precede changes in target Y by a fixed number of months. If news sentiment drops 3 months before visa applications rise, that's a **lag of 3**.

## Methodology

For each (country, signal, lag) combination:

1. **Shift** the signal by 0–6 months relative to the target
2. Compute **Pearson correlation** ($r$) between shifted signal and target
3. Record the correlation strength and p-value
4. Apply [Benjamini-Hochberg FDR](analysis/multiple-comparison-correction) to control false discoveries

## Signal Types Analyzed

| Signal | Source | Analysis Module |
|--------|--------|----------------|
| News event counts | [Google News](data-sources/google-news) | `event_visa_analysis.py` |
| News sentiment | [Sentiment Analysis](analysis/sentiment-analysis) | `event_visa_analysis.py` |
| Exchange rate (REER) | [Exchange Rates](data-sources/exchange-rates) | `exchange_rate.py` |
| Google Trends keywords | [Google Trends](data-sources/google-trends) | `trends_analysis.py` |

## Key Results

### News Events → Visas
- **58 significant signals** after FDR correction
- Strongest: Dominican Republic "first presidency" cluster at lag 3 (r = 0.617)
- See [Event-Visa Findings](findings/event-visa-findings)

### Exchange Rates → Visas
- Dominican Republic: r = 0.498 at lag 2 (p = 2.6e-06)
- See [Exchange Rate Findings](findings/exchange-rate-findings)

### Google Trends → Visas/Encounters
- 55% of keywords significant, but weak out-of-sample translation
- See [Cross-Correlation Analysis](analysis/cross-correlation)

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

- [Multiple Comparison Correction](analysis/multiple-comparison-correction) — Statistical correction for many tests
- [Surge Detection](analysis/surge-detection) — How surges are defined
- [Cross-Correlation Analysis](analysis/cross-correlation) — CCF-based alternative analysis
- [Event-Visa Findings](findings/event-visa-findings) — News results
- [Exchange Rate Findings](findings/exchange-rate-findings) — Exchange rate results
- [Analysis Module](modules/analysis-module) — Source code reference
