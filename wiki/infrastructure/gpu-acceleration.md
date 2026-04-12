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
- See [TensorRT Engines](modules/tensorrt-engines) for engine-specific details

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

- [TensorRT Engines](modules/tensorrt-engines) — Engine implementation details
- [Random Forest](models/random-forest) — cuML random forest
- [Jina v5 Embeddings](models/jina-v5-embeddings) — TRT embedding engine
- [Flan-T5 Summarization](models/flan-t5-summarization) — TRT labeling engine
- [Event Clustering](analysis/event-clustering) — GPU HDBSCAN
- [Reproducibility](infrastructure/reproducibility) — Environment setup
