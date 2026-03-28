from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_ORG = "sdsc2005-migration"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_TARGETS = {
    "trends": PROJECT_ROOT / "data" / "raw" / "trends",
    "news": PROJECT_ROOT / "data" / "raw" / "news",
    "news_json": PROJECT_ROOT / "data" / "raw" / "news_json",
    "news_embedding": PROJECT_ROOT / "data" / "processed" / "news_embeddings",
    "news_cluster_labeled": PROJECT_ROOT / "data" / "processed" / "news_embeddings_labeled",
    "encounter": PROJECT_ROOT / "data" / "raw" / "encounter",
    "visa": PROJECT_ROOT / "data" / "raw" / "visa",
    "production_outputs": PROJECT_ROOT / "data" / "processed" / "production_outputs",
}

DEFAULT_UPLOAD_TARGETS = {
    "encounter": PROJECT_ROOT / "data" / "raw" / "encounter",
    "visa": PROJECT_ROOT / "data" / "raw" / "visa",
    "production_outputs": PROJECT_ROOT / "data" / "processed" / "production_outputs",
}

DEFAULT_MODELS = {
    "flan-t5-tensorrt-int8_wo-engine": PROJECT_ROOT / "src" / "models" / "tensor-rt" / "flan-t5-tensorrt-int8_wo-engine",
    "jina-v5-tensorrt-int8_wo-engine": PROJECT_ROOT / "src" / "models" / "tensor-rt" / "jina-v5-tensorrt-int8_wo-engine",
}


def list_org(org: str) -> None:
    api = HfApi()
    datasets = sorted(repo.id for repo in api.list_datasets(author=org, full=False))
    models = sorted(repo.id for repo in api.list_models(author=org, full=False))

    print(f"Organization: {org}")
    print(f"Datasets ({len(datasets)}):")
    for repo_id in datasets:
        print(f"  - {repo_id}")

    print(f"Models ({len(models)}):")
    for repo_id in models:
        print(f"  - {repo_id}")


def _download_dataset(org: str, dataset_name: str, local_dir: Path, dry_run: bool = False) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    repo_id = f"{org}/{dataset_name}"
    if dry_run:
        print(f"[DRY-RUN] Would download {repo_id} -> {local_dir}")
        return

    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )


def download_defaults(org: str, dry_run: bool = False) -> None:
    for dataset_name, local_dir in DEFAULT_DATASET_TARGETS.items():
        _download_dataset(org, dataset_name, local_dir, dry_run=dry_run)


def download_models(org: str, dry_run: bool = False) -> None:
    for model_name, local_dir in DEFAULT_MODELS.items():
        local_dir.mkdir(parents=True, exist_ok=True)
        repo_id = f"{org}/{model_name}"
        if dry_run:
            print(f"[DRY-RUN] Would download {repo_id} -> {local_dir}")
            continue

        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )


def upload_missing(org: str, private: bool, dry_run: bool = False) -> None:
    api = HfApi()
    for dataset_name, local_dir in DEFAULT_UPLOAD_TARGETS.items():
        if not local_dir.exists():
            print(f"Skipping {dataset_name}: local path missing ({local_dir})")
            continue

        repo_id = f"{org}/{dataset_name}"
        if dry_run:
            print(f"[DRY-RUN] Would ensure repo exists: {repo_id}")
            print(f"[DRY-RUN] Would upload {local_dir} -> {repo_id}")
            continue

        print(f"Ensuring repo exists: {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

        print(f"Uploading {local_dir} -> {repo_id}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local_dir),
            commit_message=f"Sync {dataset_name} from migration workspace",
            ignore_patterns=["**/.ipynb_checkpoints/**", "**/__pycache__/**"],
        )


def upload_single(org: str, dataset_name: str, local_path: Path, private: bool, dry_run: bool = False) -> None:
    api = HfApi()
    if not local_path.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    repo_id = f"{org}/{dataset_name}"
    if dry_run:
        print(f"[DRY-RUN] Would ensure repo exists: {repo_id}")
        print(f"[DRY-RUN] Would upload {local_path} -> {repo_id}")
        return

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_path),
        commit_message=f"Upload {dataset_name} from migration workspace",
        ignore_patterns=["**/.ipynb_checkpoints/**", "**/__pycache__/**"],
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync migration data/models with Hugging Face")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_cmd = subparsers.add_parser("list", help="List datasets and models in an organization")
    list_cmd.add_argument("--org", default=DEFAULT_ORG)

    download_cmd = subparsers.add_parser("download-defaults", help="Download default dataset repos")
    download_cmd.add_argument("--org", default=DEFAULT_ORG)
    download_cmd.add_argument("--dry-run", action="store_true")

    model_cmd = subparsers.add_parser("download-models", help="Download default model repos")
    model_cmd.add_argument("--org", default=DEFAULT_ORG)
    model_cmd.add_argument("--dry-run", action="store_true")

    upload_missing_cmd = subparsers.add_parser("upload-missing", help="Upload missing local datasets")
    upload_missing_cmd.add_argument("--org", default=DEFAULT_ORG)
    upload_missing_cmd.add_argument("--private", action="store_true")
    upload_missing_cmd.add_argument("--dry-run", action="store_true")

    upload_cmd = subparsers.add_parser("upload", help="Upload a specific local directory as dataset")
    upload_cmd.add_argument("dataset_name")
    upload_cmd.add_argument("local_path")
    upload_cmd.add_argument("--org", default=DEFAULT_ORG)
    upload_cmd.add_argument("--private", action="store_true")
    upload_cmd.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "list":
        list_org(args.org)
    elif args.command == "download-defaults":
        download_defaults(args.org, dry_run=args.dry_run)
    elif args.command == "download-models":
        download_models(args.org, dry_run=args.dry_run)
    elif args.command == "upload-missing":
        upload_missing(args.org, args.private, dry_run=args.dry_run)
    elif args.command == "upload":
        upload_single(args.org, args.dataset_name, Path(args.local_path).resolve(), args.private, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
