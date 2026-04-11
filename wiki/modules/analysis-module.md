---
title: Analysis Module
aliases: [src/analysis, Analysis Package]
tags: [module, analysis, source-code]
created: 2026-04-12
---

# Analysis Module

`src/analysis/` — statistical analysis, lead-lag correlation, and visualization.

## Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `events.py` | Event detection & sentiment | `sentiment_score()`, `detect_surges_list()`, `safe_corr_list()`, `benjamini_hochberg()`, `shift_list()` |
| `event_visa_analysis.py` | Event → visa correlation | Loads clustered/labeled news, computes lagged correlations, surge pattern detection |
| `exchange_rate.py` | REER → visa analysis | `detect_exchange_shocks()`, lagged correlation with visa issuances |
| `trends_analysis.py` | Trends → migration analysis | `load_focus_countries()`, `parse_trend_file()`, `load_trends_long()`, `load_visa_monthly()` |
| `label_events_with_led.py` | LED cluster labeling | Samples articles, builds prompts, generates 2-3 word labels, normalizes output |
| `plots.py` | Visualization suite | `setup_styling()`, `load_data()`, `create_dual_axis_plot()`, publication-quality 300 DPI PNGs |
| `utils.py` | Analysis utilities | `sentiment_score()`, `detect_surges_list()`, `detect_exchange_shocks()`, `benjamini_hochberg()`, `load_exchange_monthly_lazy()`, `load_visa_monthly_lazy()` |

## Word Lists (events.py / utils.py)

```
POSITIVE_WORDS: improve, growth, peace, jobs, stability
NEGATIVE_WORDS: crisis, violence, conflict, inflation, poverty, trafficking
```

## Running Analyses

```bash
# Event-visa lead-lag analysis
python -m src.analysis.event_visa_analysis --max-lag 6 --top-labels 8 --min-overlap 12 --min-event-months 12

# Exchange rate analysis
python -m src.analysis.exchange_rate

# Trends analysis
python -m src.analysis.trends_analysis
```

## Output Destinations

| Analysis | Output Path |
|----------|------------|
| Event-visa | `data/processed/production_outputs/event_visa_*` |
| Sentiment | `data/processed/production_outputs/event_sentiment_*` |
| Exchange | `data/processed/production_outputs/exchange_visa_*` |
| Trends | `data/processed/production_outputs/trends_*` |
| Plots | `data/plots/{events_vs_visas,exchange_vs_visas,trends_vs_migration}/` |

## See Also

- [[lead-lag-analysis]] — Statistical methodology
- [[sentiment-analysis]] — Scoring approach
- [[event-clustering]] — HDBSCAN + labeling
- [[cross-correlation]] — CCF/VAR analysis
- [[processing-module]] — Upstream data provider
- [[models-module]] — Downstream model training
