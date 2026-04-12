# Flan-T5 Summarization

TensorRT-accelerated Flan-T5-Large used for **cluster labeling** — generating concise 2–3 word event labels from news article clusters.

## Model

| Property | Value |
|----------|-------|
| Base model | google/flan-t5-large |
| Quantization | INT8 (TensorRT) |
| Architecture | Encoder-Decoder |
| Tokenizer | HuggingFace AutoTokenizer |
| Default max input | 1,024 tokens |
| Default max output | 256 tokens |
| Engine path | `src/models/tensor-rt/flan-t5-large/int8_wo_cpu/1-gpu/` |

## TensorRT Engine

Implemented in `src/models/flant5_engine.py` as `TensorRTFlanT5Engine`:

- **Lazy loading**: Encoder-decoder runner loaded on first call
- **Beam search**: Configurable beam width for generation quality
- **Output cleaning**: Special tokens automatically stripped

See [TensorRT Engines](modules/tensorrt-engines) for cross-engine details.

## Prompt Templates

Defined in `src/processing/prompts.py` via the `PromptTemplate` class:

| Template | Purpose |
|----------|---------|
| `SUMMARIZATION_PROMPT` | Extract key events and facts |
| `EXTRACTION_PROMPT` | Focus on policies and statistics |
| `EVENTS_FOCUSED_PROMPT` | Structured event summary |

## Pipeline Role

Part of the [NLP Enrichment](pipeline/nlp-enrichment) pipeline, specifically the labeling stage:

```
HDBSCAN Clusters → Sample Articles → Flan-T5 Prompt → "Border Closure" (label)
```

### Process
1. Sample representative articles from each cluster
2. Build prompt with headlines and context
3. Run Flan-T5 inference to generate label
4. Normalize labels, filter duplicates/noise
5. Add `summary_t5` field to article records

### Orchestration
- `src/processing/summarize.py` — `NewsArticleSummarizer` class (batch processing, stats tracking)
- `src/processing/run_summarization.py` — CLI with argument validation, dry-run, stats-only modes

## Alternative: LED Engine

For longer input contexts (up to 16,384 tokens), the system can use [LED](analysis/event-clustering) (`src/models/led_engine.py`) based on `allenai/led-base-16384`. Used specifically in `src/analysis/label_events_with_led.py`.

## See Also

- [NLP Enrichment](pipeline/nlp-enrichment) — Full NLP pipeline context
- [Event Clustering](analysis/event-clustering) — How clusters are formed before labeling
- [TensorRT Engines](modules/tensorrt-engines) — TensorRT infrastructure
- [Processing Module](modules/processing-module) — Summarization orchestration code
- [GPU Acceleration](infrastructure/gpu-acceleration) — GPU compute details
