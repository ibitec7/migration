# Trends Analysis Findings (Production Run)

## Scope
- Analysis script: `src/analysis/trends_analysis.py`
- Data sources:
  - Google Trends: `data/raw/trends`
  - Visa outcomes: `data/processed/visa_master.parquet`
  - Border encounters: `data/raw/encounter/*.csv`
- Country scope: 15 non-US countries of interest
- Production settings: `max_lag=6`, `min_overlap=12`, `test_periods=6`

## Coverage
- Countries in production panel: **15**
- Correlation summary rows: **120**
- Country-best keyword rows: **15**
- VAR benchmark rows: **19**
  - `issuances`: 15 countries
  - `encounter_count`: 4 countries (limited encounter overlap coverage)

## Statistical Signal
- Visa lead/correlation rows significant at `q <= 0.10`: **66**
- Encounter lead/correlation rows significant at `q <= 0.10`: **21**

Interpretation:
- There are statistically significant lag relationships between some trend keywords and migration outcomes.
- Significant lag correlation does **not** automatically imply better out-of-sample forecasting.

## Predictive Benchmark Results (Out-of-time RMSE)
Compared against a baseline autoregressive model:

### Visa (`issuances`)
- Trends-enhanced VAR improved RMSE in **5 / 15** countries
- Mean improvement: **-5.97%**

### Encounters (`encounter_count`)
- Trends-enhanced VAR improved RMSE in **2 / 4** countries
- Mean improvement: **-6.02%**

Interpretation:
- Negative mean improvement indicates that, on average, trends-enhanced VAR underperformed baseline in this run.

## Overall Conclusion
Google Trends currently shows **mixed and weak predictive power** for migration forecasting in this setup:
- Useful country/keyword signals exist,
- But model performance gains are not robust across countries,
- And aggregate out-of-sample performance is worse than baseline on average.

## Practical Takeaways
1. Keep trends as an auxiliary signal, not a standalone predictor.
2. Use country-specific feature selection rather than a uniform keyword set.
3. Consider richer models (regularized or nonlinear) and stricter validation windows.
4. Improve encounter coverage consistency to strengthen cross-country conclusions.

## Related Output Artifacts
- Main findings report: `data/processed/production_outputs/trends_findings_report.md`
- Diagnostics: `data/processed/production_outputs/trends_panel_diagnostics.parquet`
- Correlation summary: `data/processed/production_outputs/trends_corr_summary.parquet`
- Best keywords by country: `data/processed/production_outputs/trends_country_best_keywords.parquet`
- Benchmark metrics: `data/processed/production_outputs/trends_var_benchmark.parquet`
- Plot index: `data/processed/production_outputs/trends_plot_index.csv`
