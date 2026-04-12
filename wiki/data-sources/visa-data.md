# Visa Data

Monthly US visa issuance statistics from the Department of State, serving as the **primary ground truth** for legal immigration volume.

## Source

- **Publisher**: Bureau of Consular Affairs, US Department of State
- **URL**: [travel.state.gov](https://travel.state.gov)
- **Format**: 108 monthly PDF reports + supplementary Excel files
- **Coverage**: 15 target countries, January 2017 – September 2025
- **Volume**: ~100,000 records after parsing

## Collection

The [Collection Module](modules/collection-module) (`src/collection/visa.py`) scrapes download links from travel.state.gov and fetches PDFs/Excel files in parallel with retry logic. Files land in `data/raw/visa/pdf/` and `data/raw/visa/excel/`.

See [Data Collection](pipeline/data-collection) for the broader ingestion architecture.

## Processing

The [PDF Parser](modules/pdf-parser) (`src/processing/parse.py`) extracts tables from PDFs using **PyMuPDF (fitz)**:

1. Detect and extract tables from each PDF page
2. Clean headers and normalize country names
3. Standardize visa classes via `VISA_MAP` (from `maps.json`)
4. Map month abbreviations (JAN → 1) and convert to ISO dates
5. Output: `data/processed/visa_master.parquet` with columns: `date`, `issuances`, `visa_type`, `country`

Parsing is parallelized across CPUs with `ProcessPoolExecutor`.

## Key Characteristics

- **Zipfian distribution**: Top 20% of countries account for 88.3% of all visas
- **Highest outliers**: Cuba, Mexico, Afghanistan
- **Seasonal patterns**: Peak April–September (70–100% above baseline); trough January–March
- **Monthly cadence** aligns with all other data sources in [Panel Construction](pipeline/panel-construction)

## Role in the Pipeline

Visa issuances are the primary **target variable** for the forecasting models. The [Build Panel Detail](modules/build-panel-detail) module creates:
- 6 lag features (`visa_lag_1` through `visa_lag_6`)
- 6 lead targets (`target_visa_lead_1` through `target_visa_lead_6`)

These feed directly into [Training Pipeline](pipeline/training-pipeline) and [Inference Pipeline](pipeline/inference-pipeline).

## See Also

- [Encounter Data](data-sources/encounter-data) — The complementary ground truth for illegal border crossings
- [Panel Construction](pipeline/panel-construction) — How visa data becomes model features/targets
- [PDF Parser](modules/pdf-parser) — Technical deep dive on PDF extraction
- [Collection Module](modules/collection-module) — Source code reference
