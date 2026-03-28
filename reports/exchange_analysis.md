# Exchange Rate vs Visa Analysis Findings (Production Run)

## Scope
- Analysis script: `src/analysis/exchange_rate.py`
- Inputs:
  - Exchange rates: `data/processed/exchange_rate.parquet`
  - Visas: `data/processed/visa_master.parquet`
- Production settings:
  - `max_lag=6`
  - `min_overlap=12`
  - output dir: `data/processed/production_outputs`

## Coverage
- Requested scope was 15 non-US countries of interest, but exchange data availability limited analysis coverage.
- Countries analyzed (overlap available): **6**
  - China, Colombia, Dominican Republic, Mexico, Pakistan, Philippines
- Total lag rows evaluated: **42**
- Best-lag rows (one per analyzed country): **6**

## Strength of Relationship
- Mean Pearson correlation across all country-lag rows: **0.108**
- Mean absolute correlation across all rows: **0.183**
- Significant rows (`q <= 0.10`): **23 / 42 (54.8%)**

### Strongest Correlation Found
- Country: **Dominican Republic**
- Lag: **2 months**
- Pearson correlation: **0.4980**
- q-value: **0.000003**

### Top Countries by Best-Lag |Correlation|
1. Dominican Republic — corr=0.4980, lag=2, q=0.000003
2. Mexico — corr=0.2894, lag=6, q=0.0175
3. Colombia — corr=-0.2241, lag=4, q=0.0573
4. Pakistan — corr=0.2084, lag=0, q=0.0604
5. China — corr=0.0486, lag=5, q=0.9087

## Predictive Signal for Visa Surges
The script evaluates whether exchange-rate **shock months** align with visa **surge months** using precision/lift metrics.

- Mean surge lift across all rows: **1.009**
- Median surge lift: **1.071**
- Rows with lift > 1: **23 / 42 (54.8%)**

Interpretation:
- On average, exchange shocks provide only a **slight** uplift over baseline surge rate.
- Some country-lag combinations are useful, but effect size is generally modest.

## Conclusion: Is There Predictive Power?
**Yes, but limited and uneven.**

- There is statistically meaningful lead-lag association in a subset of country-lag combinations.
- The strongest practical signal appears in **Dominican Republic** (and to a lesser extent **Mexico**).
- Overall, exchange rates alone are **not a consistently strong predictor** across all analyzed countries.

## Practical Recommendation
- Use exchange rate features as **one component** in a multi-signal model (with events, sentiment, and trends), not as a standalone predictor.
- Prioritize country-specific lag configurations rather than global one-size-fits-all settings.

## Related Output Artifacts
- `data/processed/production_outputs/exchange_visa_monthly_merged.parquet`
- `data/processed/production_outputs/exchange_visa_lead_lag_results.parquet`
- `data/processed/production_outputs/exchange_visa_best_lags.parquet`
- `data/processed/production_outputs/exchange_visa_overlay_index.csv`
- `data/processed/production_outputs/exchange_visa_correlation_report.md`
