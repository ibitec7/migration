---
title: Exchange Rate Findings
aliases: [REER Findings, Exchange Rate Results, Economic Signal Findings]
tags: [findings, exchange-rate, correlation, economic, results]
created: 2026-04-12
---

# Exchange Rate Findings

Results from the [[lead-lag-analysis]] of IMF Real Effective Exchange Rate (REER) against visa issuance data. Exchange rates provide a **meaningful but geographically limited** signal.

## Summary

- **Only 6 of 15 target countries** have exchange rate data (major coverage gap)
- Dominican Republic shows the **strongest economic signal** (r = 0.498)
- Exchange rate movements precede visa surges by **2–6 months**

## Country Results

| Rank | Country | Pearson r | Best Lag | p-value | Significant? |
|------|---------|-----------|----------|---------|-------------|
| 1 | Dominican Republic | **0.498** | 2 months | **2.6e-06** | Yes |
| 2 | Mexico | 0.289 | 6 months | 0.0175 | Yes |
| 3 | Colombia | −0.224 | 4 months | 0.0573 | Marginal |
| 4 | Pakistan | 0.208 | 0 months | 0.0604 | Marginal |
| 5 | Philippines | 0.140 | 3 months | 0.2341 | No |
| 6 | China | 0.049 | 5 months | 0.9087 | No |

## Interpretation

### Dominican Republic (Strongest)
- REER movements lead visa surges by 2 months
- Currency weakening → increased migration pressure
- Strongest economic signal across all countries

### Mexico
- 6-month lag suggests slow economic transmission
- r = 0.289 is moderate but statistically significant

### Missing Countries
The 9 countries without REER data include some of the most volatile migration origins:
- **Afghanistan, Cuba, Haiti, Venezuela** — countries where economic collapse is a primary push factor
- This is a major data gap that limits the economic signal's contribution

## Exchange Rate Shocks

Months where month-over-month REER change exceeds the 80th percentile are flagged as **exchange rate shocks** (see [[surge-detection]]). These shocks show elevated correlation with subsequent visa spikes in Dominican Republic and Mexico.

## Output Files

```
data/processed/production_outputs/
├── exchange_visa_best_lags_review.csv
├── exchange_visa_correlation_report.md
├── exchange_visa_overlay_index.csv
└── exchange_visa_lead_lag_results_review.csv
```

## Visualizations

6 country-level overlay plots in `data/plots/exchange_vs_visas/`.

## Implications for Modeling

Despite limited coverage, exchange rates contribute to the [[horizon-aware-ensemble|ensemble]] through:
- Lag features in [[panel-construction]] (`exchange_rate_lag_1` through `exchange_rate_lag_6`)
- Missing countries receive zero-filled values (no signal, no harm)
- Future work should add more economic indicators: **inflation, unemployment, remittances, commodity prices**

## See Also

- [[exchange-rates]] — Source data details
- [[lead-lag-analysis]] — Statistical methodology
- [[event-visa-findings]] — Complementary news-based findings
- [[model-performance]] — Combined model results
- [[project-overview]] — Limitations discussion
