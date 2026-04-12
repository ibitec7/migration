# Google Trends

Monthly Google search interest time series capturing **real-time migration intent** through keyword popularity.

## Source

- **Method**: Google Trends API via `pytrends` library, synced from Hugging Face Hub
- **Coverage**: 15 countries × 8 migration-related keywords
- **Keywords**: `us_asylum`, `cbp_one`, `us_visa`, `us_immigration`, `us_green_card`, `us_work_permit`, `how_to_immigrate_us`, `us_embassy`
- **Format**: Monthly aggregates as Parquet files (16 files: 15 countries + world)
- **Storage**: `data/raw/trends/`

## Collection

The [Collection Module](modules/collection-module) (`src/collection/trends.py`) syncs the full trends dataset from Hugging Face Hub:

```
Dataset: sdsc2005-migration/trends
Method: snapshot_download() for efficient bulk sync
```

The `download_trends.sh` shell script wraps this in a virtual-env-aware launcher.

## Processing

Trends data is integrated in [Panel Construction](pipeline/panel-construction) via the [Analysis Module](modules/analysis-module) (`src/analysis/trends_analysis.py`):

1. Parse trend Parquets, handling `"<1"` encoding (below-threshold values)
2. Unpivot keywords from wide to long format
3. Align monthly cadence with [Visa Data](data-sources/visa-data) and [Encounter Data](data-sources/encounter-data)
4. Compute lagged correlations at 0–6 month offsets

## Key Findings

Mixed results as a standalone predictor:
- **55% of keywords** showed statistical significance in correlation tests
- **Weak out-of-sample translation**: −5.97% mean RMSE improvement over baseline
- Google Trends works best as a **complementary signal** alongside [news sentiment](data-sources/google-news) and [Exchange Rates](data-sources/exchange-rates)

VAR benchmarking showed limited standalone predictive power. See [Cross-Correlation Analysis](analysis/cross-correlation) for the full CCF and VAR analysis.

## Role in the Pipeline

Trends features enter [Panel Construction](pipeline/panel-construction) as lag variables and are consumed by all four model architectures in [Training Pipeline](pipeline/training-pipeline). Their value is primarily in the multi-modal fusion — no single modality is sufficient ([RQ3](project-overview)).

## See Also

- [Cross-Correlation Analysis](analysis/cross-correlation) — CCF and VAR analysis of trends vs. migration
- [Panel Construction](pipeline/panel-construction) — How trends become model features
- [Google News](data-sources/google-news) — The stronger complementary signal
- [Collection Module](modules/collection-module) — Source code reference
