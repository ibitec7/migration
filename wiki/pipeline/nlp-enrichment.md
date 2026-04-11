---
title: NLP Enrichment
aliases: [NLP Pipeline, Text Processing, News Enrichment]
tags: [pipeline, nlp, embeddings, clustering, sentiment]
created: 2026-04-12
---

# NLP Enrichment

The NLP pipeline that transforms 104,333 valid news articles into structured event-level features: embeddings, cluster labels, and sentiment scores.

## Pipeline Stages

```
Articles (Parquet) → Embedding → Clustering → Labeling → Sentiment → Monthly Aggregation
```

### 1. Embedding — [[jina-v5-embeddings]]

- **Model**: Jina v5 via TensorRT INT8
- **Output**: 768-dimensional dense vectors per article
- **Batch size**: 1–4 (dynamic shapes)
- **Input**: `data/processed/news/news_*.parquet` (token_ids + attention_mask columns)
- **Output**: `data/processed/news_embeddings/news_*.parquet`

### 2. Clustering — [[event-clustering]]

- **Algorithm**: HDBSCAN via cuML (GPU-accelerated)
- **Input**: 768-dim embeddings
- **Output**: Cluster assignments per article
- **Properties**: Variable-density clusters, noise detection, no pre-set k

### 3. Labeling — [[flan-t5-summarization]]

- **Model**: Flan-T5-Large via TensorRT INT8 (or LED for longer inputs)
- **Process**: Sample articles from each cluster → generate 2–3 word labels
- **Examples**: "Border Closure", "Violence Escalation", "Policy Reform"
- **Output**: `data/processed/news_embeddings_labeled/`

### 4. Sentiment — [[sentiment-analysis]]

- **Method**: Rule-based lexicon scoring
- **Positive words**: improve, growth, peace, jobs, stability
- **Negative words**: crisis, violence, conflict, inflation, poverty, trafficking
- **Score range**: [−1, +1]

### 5. Monthly Aggregation

Country × cluster sentiment and event counts are aggregated to monthly cadence, aligning with [[visa-data]] and [[encounter-data]] for [[panel-construction]].

## Performance

- **Total articles processed**: 170,754 collected → 104,333 valid (61.1% success rate)
- **TensorRT acceleration**: INT8 quantization for both embedding and labeling models
- **GPU memory**: Managed via CUDA streams in [[tensorrt-engines]]

## See Also

- [[google-news]] — Source data
- [[jina-v5-embeddings]] — Embedding model details
- [[event-clustering]] — Clustering methodology
- [[flan-t5-summarization]] — Label generation engine
- [[sentiment-analysis]] — Scoring methodology
- [[panel-construction]] — Where enriched features go next
