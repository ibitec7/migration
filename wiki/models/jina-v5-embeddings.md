---
title: Jina v5 Embeddings
aliases: [Jina, Jina v5, News Embeddings, Article Embeddings]
tags: [model, embeddings, jina, tensorrt, nlp]
created: 2026-04-12
---

# Jina v5 Embeddings

TensorRT-accelerated text embeddings that convert news articles into 768-dimensional dense vectors for [[event-clustering|clustering]] and similarity analysis.

## Model

| Property | Value |
|----------|-------|
| Base model | Jina Embeddings v5 Nano |
| Quantization | INT8 (TensorRT) |
| Output dimensions | 768 |
| Pooling | Mean pooling |
| Max sequence length | 8,192 tokens |
| Batch size range | 1–4 (dynamic shapes) |

## TensorRT Engine

Implemented in `src/models/jinav5_engine.py` as `JinaV5EmbeddingTrtModel`:

- **CUDA memory management**: Explicit allocation/deallocation
- **Dynamic shapes**: Supports batch 1–4, sequence 512–8,192
- **Padding**: Short sequences padded to profile minimum
- **Async execution**: Uses CUDA streams for overlapped compute/transfer

See [[tensorrt-engines]] for cross-engine details.

## Pipeline Role

Part of the [[nlp-enrichment]] pipeline:

```
Articles (Parquet) → Tokenize → Jina v5 Embed → HDBSCAN Cluster → Label → Sentiment
```

### Input
- `data/processed/news/news_*.parquet` — contains `token_ids` and `attention_mask` columns

### Output
- `data/processed/news_embeddings/news_*.parquet` — original data + `embedding` column (768-dim vector)
- LZ4 compression for storage efficiency

## Processing (`src/models/embedding.py`)

1. Discover Parquet files in `data/processed/news/`
2. Load `token_ids` and `attention_mask` columns
3. Batch inference (max batch size 4)
4. Validate input shapes and sequence lengths
5. Append `embedding` column
6. Save updated Parquets

Supports `--max-files` cap for testing.

## See Also

- [[nlp-enrichment]] — Full NLP pipeline context
- [[event-clustering]] — Downstream clustering of embeddings
- [[tensorrt-engines]] — TensorRT infrastructure deep dive
- [[gpu-acceleration]] — GPU compute details
- [[models-module]] — Source code reference
