# Processing Module

`src/processing/` — transforms raw data into model-ready formats.

## Files

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `parse.py` | PDF visa table extraction | PyMuPDF `fitz`, `ProcessPoolExecutor`, `VISA_MAP` normalization |
| `news.py` | JSON articles → Parquet | `discover_country_directories()`, `load_and_flatten_country_articles()`, `tokenize_batch()` |
| `merge.py` | Consolidate encounter CSVs | Column mapping, Polars lazy frame concatenation |
| `build_panel.py` | Multi-source panel construction | Lag/lead features, forward-fill, monthly × country merge |
| `summarize.py` | FLAN-T5 cluster labeling | `NewsArticleSummarizer` class, batch processing, stats tracking |
| `run_summarization.py` | CLI orchestrator | Argument validation, dry-run, stats-only modes |
| `prompts.py` | Prompt templates | `PromptTemplate` class, `SUMMARIZATION_PROMPT`, `EXTRACTION_PROMPT`, `EVENTS_FOCUSED_PROMPT` |
| `utils.py` | Shared utilities | `setup_logger()`, `get_optimal_process_count()`, `MONTHS_MAP`, `VISA_MAP` |
| `maps.json` | Lookup tables | Country name normalization, visa class mapping, month abbreviations |

## Data Flow

```
parse.py:       data/raw/visa/pdf/       → data/processed/visa_master.parquet
news.py:        data/raw/news/           → data/processed/news/news_*.parquet
merge.py:       data/raw/encounter/*.csv → merged encounter DataFrame
build_panel.py: All processed sources    → data/processed/train_panel.parquet
summarize.py:   news articles + TRT engine → articles with summary_t5 field
```

## Key Design Patterns

- **Polars lazy frames**: Memory-efficient operations on large datasets
- **ProcessPoolExecutor**: Parallel PDF parsing bypassing GIL
- **Configurable prompts**: `PromptTemplate` with max_input/max_output tokens
- **Checkpoint-friendly**: Summarization tracks processed/skipped/error counts

## See Also

- [Data Processing](pipeline/data-processing) — Pipeline context
- [PDF Parser](modules/pdf-parser) — Deep dive on parse.py
- [Build Panel Detail](modules/build-panel-detail) — Deep dive on build_panel.py
- [Panel Construction](pipeline/panel-construction) — Feature engineering context
- [Collection Module](modules/collection-module) — Upstream data provider
- [Models Module](modules/models-module) — Downstream consumer
