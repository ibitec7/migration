# Events vs Visa Analysis Findings (Production Run)

## Scope
- Analysis script: `src/analysis/event_visa_analysis.py`
- Inputs:
  - Labeled event/news embeddings: `data/processed/news_embeddings_labeled`
  - Visa outcomes: `data/processed/visa_master.parquet`
- Production settings:
  - `max_lag=6`
  - `top_labels=8`
  - `min_overlap=12`
  - `min_event_months=12`
- Output dir: `data/processed/production_outputs`

## Coverage
- Countries analyzed: **15**
  - Afghanistan, China, Colombia, Cuba, Dominican Republic, El Salvador, Guatemala, Haiti, Honduras, India, Mexico, Pakistan, Philippines, Venezuela, Vietnam
- Total country-label-lag rows evaluated: **392**
- Best-lag rows (country-label winners): **56**

## Statistical Signal
- Significant rows (`q <= 0.10`): **58 / 392 (14.8%)**
- Mean correlation (signed): **-0.0486**
- Mean absolute correlation: **0.1261**

Interpretation:
- There is measurable lead-lag association for a subset of event labels,
- but average effect size across all tested combinations is modest.

## Strongest Relationships Found
### Highest positive correlation
- Country: **Honduras**
- Label: **Honduras Evangelical Pastors**
- Lag: **2 months**
- Pearson correlation: **0.4250**
- q-value: **0.0114**
- Surge lift: **1.2291**

### Strongest absolute correlation
- Country: **Mexico**
- Label: **Mexicos World Watch The ...**
- Lag: **1 month**
- Pearson correlation: **-0.4360**
- q-value: **0.000943**

## Predictive Signal for Visa Surges
The script also evaluates whether event surge months align with visa surge months.

- Rows with `surge_lift > 1` (better-than-baseline targeting of surges): **162 / 392 (41.3%)**
- Median `surge_lift` across all rows: **0.937**
- Country-best median `surge_lift`: **0.842**

Interpretation:
- Some event-label/lags provide useful surge targeting,
- but uplift is not consistently strong across all countries/labels.

## Country-Level Summary
- Countries with at least one strong best-lag signal (`q <= 0.10` in country-best rows): **11 / 15**
- Mean country-best absolute correlation: **0.2931**

Interpretation:
- At country level, most countries show at least one useful signal,
- which supports event features as a practical early-warning component.

## Conclusion: Is There Predictive Power?
**Yes — moderate, country/label-specific predictive power exists.**

- Events have stronger practical signal than a purely weak/noisy predictor would,
- but predictive value is concentrated in selected labels and lags,
- and should be used as part of a multi-signal model (events + sentiment + exchange/trends), not as a standalone universal predictor.

## Practical Recommendations
1. Prioritize country-specific top labels and lags from the best-lag outputs.
2. Use `q-value` and `surge_lift` jointly for feature selection.
3. Exclude labels with high correlation but poor surge-lift for operational forecasting.
4. Combine with sentiment and macro signals to improve robustness.

## Related Output Artifacts
- `data/processed/production_outputs/event_visa_lead_lag_results.parquet`
- `data/processed/production_outputs/event_visa_best_lags.parquet`
- `data/processed/production_outputs/event_monthly_sentiment.parquet`
- `data/processed/production_outputs/event_sentiment_lead_lag_results.parquet`
- `data/processed/production_outputs/event_sentiment_best_lags.parquet`
- `data/processed/production_outputs/event_visa_overlay_index.csv`
