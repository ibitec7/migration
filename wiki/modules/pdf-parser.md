# PDF Parser

Deep dive into `src/processing/parse.py` — the PyMuPDF-based table extraction system that converts 108 monthly visa PDFs into structured data.

## Architecture

```
PDF Files → PyMuPDF Table Detection → Header Cleaning → Country Normalization → Date Conversion → Parquet
```

## Process

### 1. PDF Reading
- Library: **PyMuPDF (`fitz`)**
- Reads each page, detects table regions
- Handles variable table layouts across different months/years

### 2. Table Extraction
- Extracts cell values from detected tables
- Cleans header rows (removes artifacts, normalizes column names)
- Handles merged cells and multi-line entries

### 3. Country Normalization
- `VISA_MAP` from `maps.json` maps visa class codes to standardized names
- Country name variants normalized: `Dominican_Republic` → `Dominican Republic`, etc.
- Invalid/unknown entries logged and skipped

### 4. Date Standardization
- Month abbreviations mapped: JAN → 1, FEB → 2, ..., DEC → 12
- Converted to ISO date format (YYYY-MM-DD)
- Source: `MONTHS_MAP` from `maps.json`

### 5. Output
- **File**: `data/processed/visa_master.parquet`
- **Columns**: `date`, `issuances`, `visa_type`, `country`
- **Volume**: ~100,000 records

## Parallelization

```python
ProcessPoolExecutor(get_optimal_process_count())
```

- PDF parsing is CPU-bound → multi-process parallelism
- Each PDF processed independently
- `get_optimal_process_count()` returns `os.cpu_count()` for optimal utilization

## Edge Cases Handled

- Variable table formats across months/years
- Missing or corrupted PDFs (logged, skipped)
- Non-standard country name spellings
- Empty table cells and padding artifacts

## See Also

- [Visa Data](data-sources/visa-data) — Source data context
- [Data Processing](pipeline/data-processing) — Pipeline context
- [Processing Module](modules/processing-module) — Full package reference
- [Panel Construction](pipeline/panel-construction) — Where parsed visa data goes next
