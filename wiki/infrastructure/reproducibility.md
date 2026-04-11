---
title: Reproducibility
aliases: [Setup, Environment, Getting Started, Installation]
tags: [infrastructure, reproducibility, setup, environment]
created: 2026-04-12
---

# Reproducibility

How to reproduce the full project from scratch.

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd migration

# 2. Create environment and install dependencies
pip install -e ".[dev]"

# 3. Bootstrap data + models from HF Hub
python -m src.main bootstrap --org sdsc2005-migration
```

Step 3 downloads **all datasets and TRT engine files** (~5 GB), making the full pipeline runnable without manual data collection.

## Dependencies

Managed via `pyproject.toml` with key packages:

| Category | Packages |
|----------|----------|
| Data | polars, pandas, pyarrow, openpyxl |
| ML | torch, cuml, scikit-learn, joblib |
| NLP | transformers, tokenizers, trafilatura, hdbscan |
| TensorRT | tensorrt, cuda-python, pycuda |
| Collection | httpx, aiohttp, feedparser, fitz (PyMuPDF) |
| Visualization | matplotlib, seaborn |
| Hub | huggingface_hub |
| CLI | typer |

## Data Pipeline Reproducibility

### Option A: Full Re-Collection (Slow)

```bash
# Collect visa PDFs
python -m src.collection.visa

# Collect encounter CSVs
python -m src.collection.encounter

# Collect news articles (~24-48 hours)
python -m src.collection.news

# Process all collected data
python -m src.processing.parse
python -m src.processing.news
python -m src.processing.merge
```

### Option B: Bootstrap from HF Hub (Fast)

```bash
python -m src.main bootstrap --org sdsc2005-migration
```

This gives you all processed data and model artifacts, ready for analysis or re-training.

## Model Training Reproducibility

```bash
# Build panel from processed data
python -m src.processing.build_panel

# Train all models + evaluate
python -m src.models.train_and_evaluate
```

Outputs saved to `src/models/trained_models/`.

## Analysis Reproducibility

```bash
# Event-visa analysis
python -m src.analysis.event_visa_analysis

# Exchange rate analysis
python -m src.analysis.exchange_rate

# Trends analysis
python -m src.analysis.trends_analysis
```

## Shell Scripts

| Script | Purpose |
|--------|---------|
| `run.sh` | Full pipeline: collect → process → train → analyze |
| `download_trends.sh` | Download Google Trends data separately |

## Directory Structure

After bootstrap, the full project structure is:

```
data/
├── raw/           ← Source data (PDFs, CSVs, JSON articles)
├── processed/     ← Parquet files, embeddings, outputs
└── plots/         ← Generated visualizations
src/
├── collection/    ← Data ingestion
├── processing/    ← Data transformation
├── analysis/      ← Statistical analysis
└── models/        ← ML training & inference
    ├── trained_models/  ← Saved model artifacts
    └── tensor-rt/       ← TRT engine files
```

## See Also

- [[main-entrypoint]] — CLI commands
- [[hf-sync]] — HF Hub sync details
- [[gpu-acceleration]] — Hardware requirements
- [[data-collection]] — Collection pipeline
- [[training-pipeline]] — Training pipeline
- [[project-overview]] — What this project does
