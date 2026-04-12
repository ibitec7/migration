# Main Entry Point

`src/main.py` — the CLI entry point that orchestrates top-level project commands.

## Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `bootstrap` | Initialize project from Hugging Face | `python -m src.main bootstrap --org sdsc2005-migration` |
| `collect-live` | Run live data collection | `python -m src.main collect-live` |
| `sync-data` | Sync processed data to/from HF Hub | `python -m src.main sync-data` |

## Bootstrap Flow

The `bootstrap` command sets up a fully reproducible environment:

1. Download default datasets from HF Hub (trends, news, embeddings, production outputs)
2. Download model artifacts (TensorRT engines)
3. Validate that all expected files exist

```bash
python -m src.main bootstrap --org sdsc2005-migration
```

This is the **recommended first step** when setting up the project. See [Reproducibility](infrastructure/reproducibility).

## Dependencies

```
main.py
├── src.collection.hf_sync   → download_defaults(), download_models()
├── src.collection.visa       → visa collection
├── src.collection.encounter  → encounter collection
├── src.collection.trends     → trends sync
└── src.collection.news       → news article scraping
```

## CLI Framework

Built with **Typer** for type-safe argument parsing and auto-generated help.

## See Also

- [Reproducibility](infrastructure/reproducibility) — Full setup instructions
- [Data Collection](pipeline/data-collection) — What bootstrap triggers
- [HF Sync](modules/hf-sync) — HF Hub sync mechanism
- [Collection Module](modules/collection-module) — Source code of collectors
