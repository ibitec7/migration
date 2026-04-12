# Event Clustering

GPU-accelerated HDBSCAN clustering of news article embeddings, followed by LED/Flan-T5 label generation, to identify **discrete migration-relevant events** from unstructured news.

## Pipeline

```
768-dim Embeddings → HDBSCAN (cuML GPU) → Cluster Assignments → LED/Flan-T5 → Event Labels
```

## Clustering: HDBSCAN

**Algorithm**: Hierarchical Density-Based Spatial Clustering of Applications with Noise

| Property | Value |
|----------|-------|
| Library | cuML (GPU-accelerated) |
| Input | 768-dimensional [Jina v5 Embeddings](models/jina-v5-embeddings) vectors |
| Key feature | Variable-density clusters, automatic noise detection |
| No pre-set k | Number of clusters determined by data density |

HDBSCAN groups semantically similar articles into clusters representing distinct events (e.g., "Cuba Policy Change", "Honduras Violence Surge", "Afghan Refugee Crisis").

## Labeling: LED + Flan-T5

Two engines available for generating 2–3 word cluster labels:

### LED (Longformer Encoder-Decoder)
- **Model**: `allenai/led-base-16384` via TensorRT
- **Max input**: 16,384 tokens (handles many article headlines)
- **Implementation**: `src/analysis/label_events_with_led.py`
- **Process**:
  1. Sample articles from each cluster
  2. Build prompt: `"Headlines from [country]: ...\nQuestion: What is the main action?\nAnswer: The main action is"`
  3. Generate 2–3 word labels

### Flan-T5
- **Model**: [Flan-T5 Summarization](models/flan-t5-summarization) via TensorRT INT8
- **Used for**: Shorter inputs, batch processing
- **Implementation**: `src/processing/summarize.py`

## Output

| Path | Content |
|------|---------|
| `data/processed/news_embeddings_labeled/` | Parquets with cluster assignments + labels |
| `data/processed/cluster_event_labels.csv` | Summary of all cluster labels |
| `data/processed/cluster_labels_test/` | Per-country JSON label files |
| `data/plots/cluster_plots/` | 19 visualization files (silhouette, noise metrics) |

### Example Labels
- "Border Closure"
- "Violence Escalation"
- "Policy Reform"
- "Evangelical Pastors" (Honduras — strongest predictor)

## Post-Clustering Analysis

Cluster labels feed into:
1. **Event counting**: Monthly event counts per country-cluster
2. **[Sentiment Analysis](analysis/sentiment-analysis)**: Average sentiment per cluster
3. **[Lead-Lag Analysis](analysis/lead-lag-analysis)**: Correlation of event-cluster time series with [Visa Data](data-sources/visa-data)

## See Also

- [Jina v5 Embeddings](models/jina-v5-embeddings) — Upstream embedding generation
- [Flan-T5 Summarization](models/flan-t5-summarization) — One of the labeling engines
- [NLP Enrichment](pipeline/nlp-enrichment) — Full NLP pipeline context
- [Sentiment Analysis](analysis/sentiment-analysis) — Downstream sentiment scoring
- [Event-Visa Findings](findings/event-visa-findings) — What clustering revealed
- [GPU Acceleration](infrastructure/gpu-acceleration) — cuML HDBSCAN on GPU
