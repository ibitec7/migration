# Project Overview

The **Migration Surge Prediction Early Warning System** is a multi-modal machine learning system that forecasts migration surges 1–6 months in advance. It transforms migration management from reactive crisis response to proactive preparation by detecting early signals in digital data streams.

## Motivation

Governments and humanitarian organizations are regularly caught off-guard by sudden migration spikes. Observable digital signals — news sentiment, search behavior, economic indicators — appear months *before* surges materialize, enabling advance preparation for:

- Policy makers adjusting immigration processing capacity
- Humanitarian organizations (UNHCR, IOM, NGOs) pre-positioning supplies
- Diplomats engaging proactively with origin countries

## Research Questions

| ID | Question |
|----|----------|
| **RQ1** | How can multi-modal digital signals (news, search intent, exchange rates) be synthesized to predict migration surges? |
| **RQ2** | How does predictive accuracy vary across 1–6 month forecasting horizons using different ML architectures? |
| **RQ3** | Which data modality (news / trends / exchange rates) is the strongest early indicator? |

## Target Countries

The 15 highest-volume US migration origin countries:

Afghanistan, China, Colombia, Cuba, Dominican Republic, El Salvador, Guatemala, Haiti, Honduras, India, Mexico, Pakistan, Philippines, Venezuela, Vietnam

## Methodology Summary

1. **Collect** four data streams at monthly cadence → [Data Collection](pipeline/data-collection)
   - [Visa Data](data-sources/visa-data) and [Encounter Data](data-sources/encounter-data) as ground truth
   - [Google News](data-sources/google-news), [Google Trends](data-sources/google-trends), [Exchange Rates](data-sources/exchange-rates) as predictive signals
2. **Process** raw data into model-ready panels → [Data Processing](pipeline/data-processing), [Panel Construction](pipeline/panel-construction)
3. **Enrich** news articles with NLP → [NLP Enrichment](pipeline/nlp-enrichment)
4. **Analyze** lead-lag relationships → [Lead-Lag Analysis](analysis/lead-lag-analysis), [Cross-Correlation Analysis](analysis/cross-correlation)
5. **Train** four ML architectures → [Training Pipeline](pipeline/training-pipeline)
6. **Ensemble** predictions with horizon-aware weighting → [Horizon-Aware Ensemble](models/horizon-aware-ensemble)
7. **Evaluate** with out-of-time validation → [Model Performance](findings/model-performance)

## Key Results

- **Ensemble F1**: 0.96 at 1-month horizon → 0.86 at 6-month horizon
- **Recall**: >92% across all horizons (catches 9+ of every 10 genuine crises)
- **Strongest single predictor**: News sentiment (r = 0.617 for Dominican Republic at 3-month lag)
- **All three modalities contribute**, but no single one is sufficient → [Event-Visa Findings](findings/event-visa-findings)

## Team

SDSC 2005 — Group 10 (6 UC San Diego students)

## See Also

- [Glossary](glossary) — Key terms and definitions
- [Reproducibility](infrastructure/reproducibility) — How to set up and run the project
- [Model Performance](findings/model-performance) — Detailed performance tables
