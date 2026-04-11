---
title: Exchange Rates
aliases: [REER, IMF Exchange Rates, Exchange Rate Data]
tags: [data-source, signal, exchange-rate, economic]
created: 2026-04-12
---

# Exchange Rates

IMF Real Effective Exchange Rate (REER) data serving as an **economic volatility proxy** for migration pressure.

## Source

- **Publisher**: International Monetary Fund (IMF)
- **Indicator**: Real Effective Exchange Rate (REER) — a trade-weighted currency index adjusted for inflation
- **Coverage**: Limited to **6 of 15 target countries** (data availability gap)
- **Cadence**: Monthly

### Country Coverage

| Available (6) | Missing (9) |
|----------------|-------------|
| China | Afghanistan |
| Colombia | Cuba |
| Dominican Republic | El Salvador |
| Mexico | Guatemala |
| Pakistan | Haiti |
| Philippines | Honduras |
| | India |
| | Venezuela |
| | Vietnam |

## Processing

The [[analysis-module]] (`src/analysis/exchange_rate.py`) handles exchange rate analysis:

1. Load monthly REER from OECD data
2. Forward/backward fill missing values in [[build-panel-detail]]
3. Detect exchange shocks: flag months with > 80th percentile month-over-month change
4. Compute lagged correlations with [[visa-data|visa issuances]] at 0–6 month offsets

## Key Findings

Meaningful but geographically limited signal:

| Country | Pearson r | Best Lag | p-value | Significant? |
|---------|-----------|----------|---------|-------------|
| Dominican Republic | 0.498 | 2 months | 2.6e-06 | Yes |
| Mexico | 0.289 | 6 months | 0.0175 | Yes |
| Colombia | −0.224 | 4 months | 0.0573 | Marginal |
| Pakistan | 0.208 | 0 months | 0.0604 | Marginal |
| China | 0.049 | 5 months | 0.9087 | No |

See [[exchange-rate-findings]] for full analysis.

## Role in the Pipeline

Exchange rate enters [[panel-construction]] as a lag feature (`exchange_rate_lag_1` through `exchange_rate_lag_6`). Missing countries receive forward-filled or zero-filled values. The limited coverage (6/15) is a **known limitation** of the system.

## See Also

- [[exchange-rate-findings]] — Full analysis results
- [[panel-construction]] — Feature integration
- [[lead-lag-analysis]] — Correlation methodology
- [[project-overview]] — Data gap limitations
