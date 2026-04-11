---
title: GPU Acceleration
aliases: [CUDA, cuML, GPU Infrastructure]
tags: [infrastructure, gpu, cuda, tensorrt, performance]
created: 2026-04-12
---

# GPU Acceleration

Overview of GPU-accelerated components throughout the pipeline.

## GPU-Accelerated Components

| Component | Library | GPU Benefit |
|-----------|---------|-------------|
| Random Forest training | **cuML** | 10–100× vs sklearn on large datasets |
| LSTM / Transformer training | **PyTorch CUDA** | Standard deep learning acceleration |
| Article embeddings | **TensorRT INT8** (Jina v5) | ~4× vs FP32 inference |
| Cluster labeling | **TensorRT INT8** (Flan-T5) | ~4× vs FP32 inference |
| Event labeling | **TensorRT INT8** (LED) | ~4× vs FP32 inference |
| HDBSCAN clustering | **cuML** | GPU-accelerated density clustering |
| Ensemble inference | **PyTorch CUDA** | Parallel 3-model forward pass |

## TensorRT INT8 Quantization

All three NLP engines use INT8 quantization:
- **4× memory reduction** vs FP32 (32 bits → 8 bits per weight)
- **~4× throughput increase** via reduced memory bandwidth
- Calibrated using representative datasets (not post-training quantization)
- See [[tensorrt-engines]] for engine-specific details

## cuML GPU Libraries

- `cuml.ensemble.RandomForestClassifier` for random forest (replaces sklearn)
- `cuml.cluster.HDBSCAN` for news article clustering (replaces hdbscan)
- Both require NVIDIA GPU with CUDA toolkit

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 8 GB | 16+ GB |
| CUDA version | 11.8 | 12.x |
| TensorRT | 8.x | Latest |
| System RAM | 16 GB | 32+ GB |

## Profiling

`scripts/profile_trt_engines.py` provides benchmarking across batch sizes, measuring:
- Throughput (tokens/sec, embeddings/sec)
- Latency (ms per call)
- Peak GPU memory (MB)
- GPU utilization (%)

## See Also

- [[tensorrt-engines]] — Engine implementation details
- [[random-forest]] — cuML random forest
- [[jina-v5-embeddings]] — TRT embedding engine
- [[flan-t5-summarization]] — TRT labeling engine
- [[event-clustering]] — GPU HDBSCAN
- [[reproducibility]] — Environment setup
