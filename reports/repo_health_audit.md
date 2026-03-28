# Repository Health Audit & Maintainer Checklist

Date: 2026-03-28

## Current Health Snapshot

- Reproducible bootstrap path exists via `python -m src.main bootstrap`.
- Safe end-to-end validation exists via dry-run mode:
  - `python -m src.main bootstrap --org sdsc2005-migration --dry-run`
  - `python -m src.main sync-data --org sdsc2005-migration --dry-run`
- Critical quality gates are active and passing:
  - `uvx ruff check src scripts --select E9,F`
  - `python -m compileall -q src`
  - `python scripts/check_dependency_consistency.py`
- Hugging Face organization is configured and populated for reproducibility:
  - Datasets: trends, news, news_json, news_embedding, news_cluster_labeled, encounter, visa, production_outputs
  - Models: flan-t5-tensorrt-int8_wo-engine, jina-v5-tensorrt-int8_wo-engine

## Maintainer Runbook (New Machine)

1. **Clone and install**
   - `uv sync --locked`

2. **Bootstrap reproducible assets from Hugging Face**
   - `python -m src.main bootstrap --org sdsc2005-migration`

3. **Validate code health gates**
   - `uvx ruff check src scripts --select E9,F`
   - `python -m compileall -q src`
   - `python scripts/check_dependency_consistency.py`

4. **Optional: run a production analysis smoke**
   - `python src/analysis/event_visa_analysis.py --max-lag 6 --top-labels 8 --min-overlap 12 --min-event-months 12 --output-dir data/processed/production_outputs --plots-dir data/plots/events_vs_visas`

## Maintainer Dry-Run Checks (No Large Downloads)

Use these before CI changes or onboarding tests:

- `python -m src.main bootstrap --org sdsc2005-migration --dry-run`
- `python -m src.main sync-data --org sdsc2005-migration --dry-run`
- `python -m src.collection.hf_sync list --org sdsc2005-migration`

Expected behavior:
- Commands complete successfully.
- Dry-run output prints all expected dataset/model download targets.

## HF Data Sync Operations

### Upload missing local datasets

- `python -m src.collection.hf_sync upload-missing --org sdsc2005-migration`

This syncs:
- `data/raw/encounter` -> `sdsc2005-migration/encounter`
- `data/raw/visa` -> `sdsc2005-migration/visa`
- `data/processed/production_outputs` -> `sdsc2005-migration/production_outputs`

### Upload a custom dataset folder

- `python -m src.collection.hf_sync upload <dataset_name> <local_path> --org sdsc2005-migration`

## CI Expectations

Workflow: `.github/workflows/ci.yml`

Current checks:
- `uv sync --locked`
- dependency consistency script
- source compile check
- strict critical lint for `src` and `scripts`
- CLI smoke checks for core commands

## Open Follow-Ups (Recommended)

- Add non-critical style linting (`ruff` full rule set) behind a separate CI job.
- Add unit tests for:
  - `src.collection.hf_sync` command parsing + dry-run behavior
  - `scripts/check_dependency_consistency.py`
  - `src.main` command routing
- Add one integration test that runs `bootstrap --dry-run` in CI and validates required targets.
