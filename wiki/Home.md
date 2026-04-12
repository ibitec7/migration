# Migration Surge Prediction Wiki

Welcome to the knowledge graph for the **Migration Surge Prediction Early Warning System**. This wiki documents the full project — data sources, pipeline stages, ML models, analysis methods, source modules, key findings, and infrastructure.


---

## Quick Start

- [Project Overview](project-overview) — Goals, research questions, methodology, and team
- [Glossary](glossary) — Key terms used throughout this wiki

---

## Data Sources

Raw inputs that feed the prediction system.

| Page | Description |
|------|-------------|
| [Visa Data](data-sources/visa-data) | US Department of State visa issuance statistics (108 monthly PDFs) |
| [Encounter Data](data-sources/encounter-data) | CBP Southwest border encounter statistics (FY2019–2026) |
| [Google News](data-sources/google-news) | 170K+ news articles across 15 countries × 8 topics |
| [Google Trends](data-sources/google-trends) | Monthly search-interest time series (15 countries × 8 keywords) |
| [Exchange Rates](data-sources/exchange-rates) | IMF Real Effective Exchange Rate for 6 countries |

---

## Pipeline

The end-to-end flow from raw data to production forecasts.

```
Collection → Processing → NLP Enrichment → Panel Construction → Training → Inference
```

| Page | Description |
|------|-------------|
| [Data Collection](pipeline/data-collection) | Ingestion layer: async scraping, bounded concurrency, retry logic |
| [Data Processing](pipeline/data-processing) | PDF parsing, JSON→Parquet, encounter merging |
| [NLP Enrichment](pipeline/nlp-enrichment) | Embedding → Clustering → Labeling → Sentiment |
| [Panel Construction](pipeline/panel-construction) | Feature engineering: 18 lag features, 6 lead targets |
| [Training Pipeline](pipeline/training-pipeline) | Out-of-time train/test split, 4 architectures |
| [Inference Pipeline](pipeline/inference-pipeline) | Horizon-aware ensemble, production prediction flow |

---

## Models

Machine learning architectures and their roles in the ensemble.

| Page | Description |
|------|-------------|
| [Random Forest](models/random-forest) | cuML GPU Random Forest — best at short horizons (Lead 1–2) |
| [LSTM](models/lstm) | MigrationLSTM — country-aware with SurgeJointLoss |
| [Transformer](models/transformer) | MigrationTransformer — best at long horizons (Lead 5–6) |
| [Horizon-Aware Ensemble](models/horizon-aware-ensemble) | Dynamic weighting: RF→short, Transformer→long |
| [SurgeJointLoss](models/surge-joint-loss) | Dual-objective loss: Huber + BCE for crisis detection |
| [Jina v5 Embeddings](models/jina-v5-embeddings) | TensorRT INT8 news article embeddings (768-dim) |
| [Flan-T5 Summarization](models/flan-t5-summarization) | TensorRT INT8 cluster labeling engine |

---

## Analysis Methods

Statistical techniques driving the lead-lag and surge analysis.

| Page | Description |
|------|-------------|
| [Lead-Lag Analysis](analysis/lead-lag-analysis) | Pearson correlation at 0–6 month offsets |
| [Surge Detection](analysis/surge-detection) | Quantile-based and σ-threshold spike identification |
| [Sentiment Analysis](analysis/sentiment-analysis) | Rule-based lexicon scoring for migration-relevant news |
| [Event Clustering](analysis/event-clustering) | HDBSCAN GPU clustering + LED label generation |
| [Cross-Correlation Analysis](analysis/cross-correlation) | CCF analysis, VAR benchmarking, ADF stationarity tests |
| [Multiple Comparison Correction](analysis/multiple-comparison-correction) | Benjamini-Hochberg FDR for 58 significant signals |

---

## Key Findings

What the system discovered about migration predictability.

| Page | Description |
|------|-------------|
| [Event-Visa Findings](findings/event-visa-findings) | News events as leading indicators (r=0.617 at 3-month lag) |
| [Exchange Rate Findings](findings/exchange-rate-findings) | Exchange rate signals (DR r=0.498 at 2-month lag) |
| [Model Performance](findings/model-performance) | Ensemble results: F1=0.96 at Lead 1, F1=0.86 at Lead 6 |

---

## Source Modules

Reference documentation for every `src/` subpackage and key files.

| Page | Description |
|------|-------------|
| [Main Entry Point](modules/main-entrypoint) | `src/main.py` CLI: bootstrap, collect-live, sync-data |
| [Collection Module](modules/collection-module) | `src/collection/*` — visa, encounter, news, trends, HF sync |
| [Processing Module](modules/processing-module) | `src/processing/*` — parse, merge, build_panel, summarize |
| [Analysis Module](modules/analysis-module) | `src/analysis/*` — events, exchange_rate, trends_analysis, plots |
| [Models Module](modules/models-module) | `src/models/*` — surge_model, train_and_evaluate, inference |
| [News Scraper](modules/news-scraper) | Deep dive: batch decoding, checkpoint recovery, throttling |
| [PDF Parser](modules/pdf-parser) | Deep dive: PyMuPDF table extraction, VISA_MAP normalization |
| [TensorRT Engines](modules/tensorrt-engines) | Deep dive: Jina-v5, Flan-T5, LED TensorRT engines |
| [Build Panel Detail](modules/build-panel-detail) | Deep dive: lag/lead construction, forward-fill strategies |
| [HF Sync](modules/hf-sync) | Deep dive: bidirectional Hugging Face Hub sync |

---

## Infrastructure

Compute, reproducibility, and operational details.

| Page | Description |
|------|-------------|
| [GPU Acceleration](infrastructure/gpu-acceleration) | TensorRT INT8, cuML, CUDA streams, NVML profiling |
| [Reproducibility](infrastructure/reproducibility) | HF bootstrap, run.sh pipeline, dependency checking |

---

