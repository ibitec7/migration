---
title: Migration Surge Prediction Wiki
aliases: [Home, MOC, Map of Content]
tags: [index, navigation]
created: 2026-04-12
---

# Migration Surge Prediction Wiki

Welcome to the knowledge graph for the **Migration Surge Prediction Early Warning System**. This wiki documents the full project — data sources, pipeline stages, ML models, analysis methods, source modules, key findings, and infrastructure.

> **Tip**: Open the Obsidian **Graph View** (`Ctrl+G`) to explore how all concepts interconnect.

---

## Quick Start

- [[project-overview]] — Goals, research questions, methodology, and team
- [[glossary]] — Key terms used throughout this wiki

---

## Data Sources

Raw inputs that feed the prediction system.

| Page | Description |
|------|-------------|
| [[visa-data]] | US Department of State visa issuance statistics (108 monthly PDFs) |
| [[encounter-data]] | CBP Southwest border encounter statistics (FY2019–2026) |
| [[google-news]] | 170K+ news articles across 15 countries × 8 topics |
| [[google-trends]] | Monthly search-interest time series (15 countries × 8 keywords) |
| [[exchange-rates]] | IMF Real Effective Exchange Rate for 6 countries |

---

## Pipeline

The end-to-end flow from raw data to production forecasts.

```
Collection → Processing → NLP Enrichment → Panel Construction → Training → Inference
```

| Page | Description |
|------|-------------|
| [[data-collection]] | Ingestion layer: async scraping, bounded concurrency, retry logic |
| [[data-processing]] | PDF parsing, JSON→Parquet, encounter merging |
| [[nlp-enrichment]] | Embedding → Clustering → Labeling → Sentiment |
| [[panel-construction]] | Feature engineering: 18 lag features, 6 lead targets |
| [[training-pipeline]] | Out-of-time train/test split, 4 architectures |
| [[inference-pipeline]] | Horizon-aware ensemble, production prediction flow |

---

## Models

Machine learning architectures and their roles in the ensemble.

| Page | Description |
|------|-------------|
| [[random-forest]] | cuML GPU Random Forest — best at short horizons (Lead 1–2) |
| [[lstm]] | MigrationLSTM — country-aware with SurgeJointLoss |
| [[transformer]] | MigrationTransformer — best at long horizons (Lead 5–6) |
| [[horizon-aware-ensemble]] | Dynamic weighting: RF→short, Transformer→long |
| [[surge-joint-loss]] | Dual-objective loss: Huber + BCE for crisis detection |
| [[jina-v5-embeddings]] | TensorRT INT8 news article embeddings (768-dim) |
| [[flan-t5-summarization]] | TensorRT INT8 cluster labeling engine |

---

## Analysis Methods

Statistical techniques driving the lead-lag and surge analysis.

| Page | Description |
|------|-------------|
| [[lead-lag-analysis]] | Pearson correlation at 0–6 month offsets |
| [[surge-detection]] | Quantile-based and σ-threshold spike identification |
| [[sentiment-analysis]] | Rule-based lexicon scoring for migration-relevant news |
| [[event-clustering]] | HDBSCAN GPU clustering + LED label generation |
| [[cross-correlation]] | CCF analysis, VAR benchmarking, ADF stationarity tests |
| [[multiple-comparison-correction]] | Benjamini-Hochberg FDR for 58 significant signals |

---

## Key Findings

What the system discovered about migration predictability.

| Page | Description |
|------|-------------|
| [[event-visa-findings]] | News events as leading indicators (r=0.617 at 3-month lag) |
| [[exchange-rate-findings]] | Exchange rate signals (DR r=0.498 at 2-month lag) |
| [[model-performance]] | Ensemble results: F1=0.96 at Lead 1, F1=0.86 at Lead 6 |

---

## Source Modules

Reference documentation for every `src/` subpackage and key files.

| Page | Description |
|------|-------------|
| [[main-entrypoint]] | `src/main.py` CLI: bootstrap, collect-live, sync-data |
| [[collection-module]] | `src/collection/*` — visa, encounter, news, trends, HF sync |
| [[processing-module]] | `src/processing/*` — parse, merge, build_panel, summarize |
| [[analysis-module]] | `src/analysis/*` — events, exchange_rate, trends_analysis, plots |
| [[models-module]] | `src/models/*` — surge_model, train_and_evaluate, inference |
| [[news-scraper]] | Deep dive: batch decoding, checkpoint recovery, throttling |
| [[pdf-parser]] | Deep dive: PyMuPDF table extraction, VISA_MAP normalization |
| [[tensorrt-engines]] | Deep dive: Jina-v5, Flan-T5, LED TensorRT engines |
| [[build-panel-detail]] | Deep dive: lag/lead construction, forward-fill strategies |
| [[hf-sync]] | Deep dive: bidirectional Hugging Face Hub sync |

---

## Infrastructure

Compute, reproducibility, and operational details.

| Page | Description |
|------|-------------|
| [[gpu-acceleration]] | TensorRT INT8, cuML, CUDA streams, NVML profiling |
| [[reproducibility]] | HF bootstrap, run.sh pipeline, dependency checking |

---

## Optional Enhancements

- Install the **Dataview** plugin to generate dynamic indexes from frontmatter tags
- Use **Graph View** color groups (already configured) to distinguish categories
- Future: add 15 country-specific pages for per-country findings
