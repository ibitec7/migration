# Data Processing

Transforms raw data from `data/raw/` into model-ready formats in `data/processed/`. Handles PDF parsing, JSON normalization, CSV consolidation, and schema standardization.

## Architecture

```
src/processing/
├── parse.py              → PDF visa table extraction (PyMuPDF)
├── news.py               → JSON articles → Parquet + tokenization
├── merge.py              → Consolidate encounter CSVs
├── build_panel.py        → Multi-source panel construction
├── summarize.py          → FLAN-T5 cluster labeling
├── run_summarization.py  → CLI orchestrator for NLP pipeline
├── prompts.py            → Configurable prompt templates
└── utils.py              → Shared helpers, month/visa maps
```

## Processing Steps

### 1. Visa PDF Parsing → [PDF Parser](modules/pdf-parser)
- Input: 108 monthly PDFs from `data/raw/visa/pdf/`
- Tool: PyMuPDF (`fitz`) table extraction
- Output: `data/processed/visa_master.parquet`
- Parallelized with `ProcessPoolExecutor`

### 2. News JSON → Parquet
- Input: JSON files per country/year from `data/raw/news/`
- Process: Flatten articles, parse dates (multi-source extraction), tokenize with Jina tokenizer
- Output: `data/processed/news/news_*.parquet`

### 3. Encounter CSV Merge
- Input: 8 CSVs from `data/raw/encounter/`
- Process: Column mapping (Fiscal Year → year, month abbreviation → month_num), lazy frame concatenation (Polars)
- Output: Unified encounter DataFrame

### 4. Panel Construction → [Panel Construction](pipeline/panel-construction)
- Merges visa, exchange rate, and news event data into a monthly × country panel

## Data Flow

```
data/raw/visa/pdf/       → parse.py      → data/processed/visa_master.parquet
data/raw/news/           → news.py       → data/processed/news/news_*.parquet
data/raw/encounter/*.csv → merge.py      → merged encounter DataFrame
data/raw/trends/         → (passed through to analysis)
All processed data       → build_panel.py → data/processed/train_panel.parquet
```

## Key Design Choices

- **Polars lazy frames** for memory-efficient large dataset operations
- **Country name canonicalization** via `maps.json` to handle variants (Dominican_Republic → Dominican Republic)
- **Parallel PDF parsing** via `ProcessPoolExecutor` with optimal CPU count detection
- **Parquet output** with LZ4 compression for efficient storage and fast reads

## See Also

- [Data Collection](pipeline/data-collection) — Previous stage (ingestion)
- [PDF Parser](modules/pdf-parser) — Deep dive on PDF extraction
- [Panel Construction](pipeline/panel-construction) — Next stage (feature engineering)
- [Processing Module](modules/processing-module) — Full source code documentation
- [NLP Enrichment](pipeline/nlp-enrichment) — Parallel NLP processing track
