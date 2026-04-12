# HF Sync

Deep dive into `src/collection/hf_sync.py` — the bidirectional synchronization utility between local data and the Hugging Face Hub.

## Organization

All datasets and models are stored under the **`sdsc2005-migration`** Hugging Face organization, enabling:
- **Full reproducibility**: Any team member can bootstrap from scratch
- **Version control**: Datasets and models are versioned via HF Hub
- **Collaboration**: Centralized data sharing

## Functions

### `download_defaults()`

Syncs 6 default datasets from HF Hub to local paths:

| HF Dataset | Local Path |
|-----------|-----------|
| `sdsc2005-migration/trends` | `data/raw/trends/` |
| `sdsc2005-migration/news` | `data/raw/news/` |
| `sdsc2005-migration/news_embeddings` | `data/processed/news_embeddings/` |
| `sdsc2005-migration/news_embeddings_labeled` | `data/processed/news_embeddings_labeled/` |
| `sdsc2005-migration/production_outputs` | `data/processed/production_outputs/` |
| `sdsc2005-migration/cluster_labels` | `data/processed/cluster_labels_test/` |

### `download_models()`

Syncs 3 TensorRT model artifacts:

| HF Model | Local Path |
|----------|-----------|
| Flan-T5 engine | `src/models/tensor-rt/flan-t5-large/` |
| Jina v5 engine | (configured externally) |
| LED engine | (configured externally) |

### `upload_missing()`

Uploads local processed data back to HF Hub for team sharing:
- Scans local directories for files not yet in the remote repo
- Supports **dry-run mode** for validation without actual upload

## Bootstrap Flow

```bash
python -m src.main bootstrap --org sdsc2005-migration
```

This triggers:
1. `download_defaults()` — all datasets
2. `download_models()` — TensorRT engines
3. Validation — checks all expected files exist

## Dry-Run Support

All sync operations support `--dry-run` to preview what would be downloaded/uploaded without actually performing transfers. Essential for CI/CD validation.

## See Also

- [Main Entry Point](modules/main-entrypoint) — CLI that invokes sync
- [Reproducibility](infrastructure/reproducibility) — Full setup guide
- [Data Collection](pipeline/data-collection) — Where synced data is used
- [Collection Module](modules/collection-module) — Full package reference
