# Cross-Correlation Analysis

CCF (Cross-Correlation Function) analysis and VAR (Vector Autoregression) benchmarking for evaluating [Google Trends](data-sources/google-trends) as a migration predictor.

## Methodology

### 1. Stationarity Testing (ADF)

Before correlation analysis, all time series are tested for stationarity using the **Augmented Dickey-Fuller (ADF) test**. Non-stationary series are differenced until stationary.

### 2. Cross-Correlation Function (CCF)

For each (country, keyword, target) combination:
- Compute the correlation between the trends keyword and migration target at multiple lag offsets
- Identify the lag with maximum absolute correlation
- Test significance against a white-noise null hypothesis

### 3. VAR Benchmarking

**Vector Autoregression** models are fitted to evaluate whether Google Trends adds predictive power beyond autoregressive baselines:
- Baseline: AR model using only lagged visa/encounter values
- Test: VAR model adding Google Trends keywords
- Metric: RMSE improvement over baseline

## Key Results

| Finding | Value |
|---------|-------|
| Keywords showing significance | 55% |
| Mean RMSE improvement (out-of-sample) | −5.97% (slightly worse) |
| Best keywords | `us_asylum`, `cbp_one` (for encounter prediction) |
| Worst keywords | Generic terms like `us_immigration` |

**Conclusion**: Google Trends has limited standalone predictive power but contributes as a complementary signal in the [multi-modal ensemble](models/horizon-aware-ensemble).

## Focus Countries

The analysis focuses on high-volume countries:
- Mexico, Guatemala, Honduras, El Salvador (Northern Triangle)
- Venezuela, Cuba, Colombia
- Auto-detected from news data via `load_focus_countries()`

## Output Files

```
data/processed/production_outputs/
├── trends_corr_summary.parquet
├── trends_country_best_keywords.parquet
├── trends_panel_diagnostics.parquet
├── trends_var_benchmark.parquet
├── trends_analysis_report.md
└── trends_findings_report.md
```

## Visualizations

30 plots in `data/plots/trends_vs_migration/`:
- Country-specific CCF plots
- Composite tracking plots (trends overlaid with encounters/visas)

## Implementation

`src/analysis/trends_analysis.py` — key functions:
- `load_focus_countries()` — Auto-detect analysis countries
- `parse_trend_file()` — Handle `"<1"` encoding
- `load_trends_long()` — Wide-to-long unpivot
- `load_visa_monthly()` — Aggregate visa by country/month

## See Also

- [Google Trends](data-sources/google-trends) — Source data
- [Lead-Lag Analysis](analysis/lead-lag-analysis) — Complementary Pearson correlation approach
- [Multiple Comparison Correction](analysis/multiple-comparison-correction) — Statistical correction
- [Analysis Module](modules/analysis-module) — Full source code reference
- [Project Overview](project-overview) — RQ3 context
