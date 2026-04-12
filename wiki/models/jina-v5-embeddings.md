# Jina v5 Embeddings

TensorRT-accelerated text embeddings that convert news articles into 768-dimensional dense vectors for [clustering](analysis/event-clustering) and similarity analysis.

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

See [TensorRT Engines](modules/tensorrt-engines) for cross-engine details.

## Pipeline Role

Part of the [NLP Enrichment](pipeline/nlp-enrichment) pipeline:

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

- [NLP Enrichment](pipeline/nlp-enrichment) — Full NLP pipeline context
- [Event Clustering](analysis/event-clustering) — Downstream clustering of embeddings
- [TensorRT Engines](modules/tensorrt-engines) — TensorRT infrastructure deep dive
- [GPU Acceleration](infrastructure/gpu-acceleration) — GPU compute details
- [Models Module](modules/models-module) — Source code reference
