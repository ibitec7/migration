# TensorRT Engines

Deep dive into the three TensorRT inference engine wrappers used for GPU-accelerated NLP throughout the pipeline.

## Engines Overview

| Engine | File | Model | Purpose | Quantization |
|--------|------|-------|---------|-------------|
| Jina v5 | `jinav5_engine.py` | Jina Embeddings v5 Nano | Article embeddings (768-dim) | INT8 |
| Flan-T5 | `flant5_engine.py` | google/flan-t5-large | Cluster labeling / summarization | INT8 |
| LED | `led_engine.py` | allenai/led-base-16384 | Long-context event labeling | INT8 |

## Common Patterns

All three engines share these design patterns:

### CUDA Memory Management
- Explicit allocation/deallocation of GPU memory
- CUDA streams for async execution
- Memory cleanup on engine destruction

### Dynamic Shapes
- Support variable batch sizes and sequence lengths
- Padding short sequences to profile minimums
- Profile-based optimization (min/opt/max shapes)

### Lazy Loading
- Engines loaded on first call, not at import
- Reduces startup time and memory usage when engines aren't needed

## Jina v5 Engine (`JinaV5EmbeddingTrtModel`)

| Property | Value |
|----------|-------|
| Batch range | 1–4 |
| Sequence range | 512–8,192 tokens |
| Output | 768-dimensional mean-pooled embedding |
| Execution | Async CUDA streams |

Used by `src/models/embedding.py` → see [Jina v5 Embeddings](models/jina-v5-embeddings).

## Flan-T5 Engine (`TensorRTFlanT5Engine`)

| Property | Value |
|----------|-------|
| Architecture | Encoder-decoder |
| Tokenizer | HuggingFace AutoTokenizer |
| Max input | 1,024 tokens (configurable) |
| Max output | 256 tokens (configurable) |
| Generation | Beam search |
| Engine path | `src/models/tensor-rt/flan-t5-large/int8_wo_cpu/1-gpu/` |

Used by `src/processing/summarize.py` → see [Flan-T5 Summarization](models/flan-t5-summarization).

## LED Engine

| Property | Value |
|----------|-------|
| Architecture | Longformer encoder-decoder |
| Max input | 16,384 tokens |
| Generation | Beam search with decoder start token |
| Special tokens | Filtered from output |

Used by `src/analysis/label_events_with_led.py` → see [Event Clustering](analysis/event-clustering).

## Profiling

`scripts/profile_trt_engines.py` benchmarks all engines:
- Measures throughput (tokens/sec), latency (ms), peak GPU memory (MB), GPU utilization (%)
- Tests across batch sizes [1, 4, 8]
- Uses NVIDIA NVML for real-time GPU monitoring
- Warmup + 3 averaging runs per configuration

## See Also

- [Jina v5 Embeddings](models/jina-v5-embeddings) — Embedding model context
- [Flan-T5 Summarization](models/flan-t5-summarization) — Labeling model context
- [GPU Acceleration](infrastructure/gpu-acceleration) — Broader GPU infrastructure
- [Models Module](modules/models-module) — Full source code reference
