# NLP Enrichment

The NLP pipeline that transforms 104,333 valid news articles into structured event-level features: embeddings, cluster labels, and sentiment scores.

## Pipeline Stages

```
Articles (Parquet) → Embedding → Clustering → Labeling → Sentiment → Monthly Aggregation
```

### 1. Embedding — [Jina v5 Embeddings](models/jina-v5-embeddings)

- **Model**: Jina v5 via TensorRT INT8
- **Output**: 768-dimensional dense vectors per article
- **Batch size**: 1–4 (dynamic shapes)
- **Input**: `data/processed/news/news_*.parquet` (token_ids + attention_mask columns)
- **Output**: `data/processed/news_embeddings/news_*.parquet`

### 2. Clustering — [Event Clustering](analysis/event-clustering)

- **Algorithm**: HDBSCAN via cuML (GPU-accelerated)
- **Input**: 768-dim embeddings
- **Output**: Cluster assignments per article
- **Properties**: Variable-density clusters, noise detection, no pre-set k

### 3. Labeling — [Flan-T5 Summarization](models/flan-t5-summarization)

- **Model**: Flan-T5-Large via TensorRT INT8 (or LED for longer inputs)
- **Process**: Sample articles from each cluster → generate 2–3 word labels
- **Examples**: "Border Closure", "Violence Escalation", "Policy Reform"
- **Output**: `data/processed/news_embeddings_labeled/`

### 4. Sentiment — [Sentiment Analysis](analysis/sentiment-analysis)

- **Method**: Rule-based lexicon scoring
- **Positive words**: improve, growth, peace, jobs, stability
- **Negative words**: crisis, violence, conflict, inflation, poverty, trafficking
- **Score range**: [−1, +1]

### 5. Monthly Aggregation

Country × cluster sentiment and event counts are aggregated to monthly cadence, aligning with [Visa Data](data-sources/visa-data) and [Encounter Data](data-sources/encounter-data) for [Panel Construction](pipeline/panel-construction).

## Performance

- **Total articles processed**: 170,754 collected → 104,333 valid (61.1% success rate)
- **TensorRT acceleration**: INT8 quantization for both embedding and labeling models
- **GPU memory**: Managed via CUDA streams in [TensorRT Engines](modules/tensorrt-engines)

## See Also

- [Google News](data-sources/google-news) — Source data
- [Jina v5 Embeddings](models/jina-v5-embeddings) — Embedding model details
- [Event Clustering](analysis/event-clustering) — Clustering methodology
- [Flan-T5 Summarization](models/flan-t5-summarization) — Label generation engine
- [Sentiment Analysis](analysis/sentiment-analysis) — Scoring methodology
- [Panel Construction](pipeline/panel-construction) — Where enriched features go next
