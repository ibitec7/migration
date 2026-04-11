---
title: Event-Visa Findings
aliases: [News Findings, Event Findings, News-Visa Correlation]
tags: [findings, events, news, correlation, results]
created: 2026-04-12
---

# Event-Visa Findings

Key results from the [[lead-lag-analysis]] of news event clusters and sentiment against visa issuance data. News sentiment emerged as the **strongest individual predictor** of migration surges.

## Summary

- **58 statistically significant signals** after [[multiple-comparison-correction|Benjamini-Hochberg FDR correction]]
- News events can lead visa changes by **1–6 months**
- Both event counts and sentiment scores provide predictive value

## Top Correlations

### By Event Cluster

| Country | Event Cluster | Pearson r | Best Lag | Significant? |
|---------|---------------|-----------|----------|-------------|
| Dominican Republic | "First Presidency" | **0.617** | 3 months | Yes |
| Cuba | Trump Policy Coverage | **−0.595** | 2 months | Yes |
| Honduras | Evangelical Pastors | **0.581** | 4 months | Yes |
| Guatemala | Border Policy | 0.523 | 3 months | Yes |
| Venezuela | Economic Crisis | 0.509 | 5 months | Yes |

### Interpretation

- **Dominican Republic**: Political events ("first presidency") preceded visa application surges by 3 months — political instability drives migration planning
- **Cuba**: Negative Trump policy coverage (r = −0.595) saw *increased* migration — restrictive rhetoric paradoxically accelerates departure timelines
- **Honduras**: Religious community events (Evangelical Pastors) correlated with migration activity, likely reflecting community network mobilization

## Sentiment Signals

The [[sentiment-analysis|sentiment scoring]] revealed that negative news sentiment consistently leads visa increases:
- Negative sentiment (crisis, violence, conflict) → higher visa numbers 2–4 months later
- This aligns with the "push factor" hypothesis — deteriorating conditions in origin countries drive emigration

## Data Coverage

| Metric | Value |
|--------|-------|
| Total articles analyzed | 104,333 |
| Countries covered | 15 |
| Event clusters detected | Varies by country (8–30+) |
| Lag range tested | 0–6 months |
| Minimum overlap threshold | 12 months |

## Output Files

```
data/processed/production_outputs/
├── event_visa_best_lags_review.csv
├── event_visa_lead_lag_results_review.csv
├── event_sentiment_best_lags_review.csv
├── event_sentiment_lead_lag_results_review.csv
└── event_visa_overlay_index.csv
```

## Visualizations

15 country-level overlay plots in `data/plots/events_vs_visas/`.

## Limitations

- **English-language bias**: News corpus skews toward English-language sources
- **Cluster quality**: HDBSCAN noise points are discarded, potentially losing signal
- **Correlation ≠ causation**: Events may be symptoms rather than causes
- **15-country scope**: Results may not generalize to other origin countries

## See Also

- [[google-news]] — Source data
- [[event-clustering]] — How events are detected
- [[sentiment-analysis]] — Scoring methodology
- [[lead-lag-analysis]] — Statistical methodology
- [[multiple-comparison-correction]] — How significance is determined
- [[exchange-rate-findings]] — Complementary economic signal
- [[model-performance]] — How findings translate to model accuracy
