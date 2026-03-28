from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _python_module(module_name: str, *args: str) -> None:
    _run([sys.executable, "-m", module_name, *args])


def cmd_bootstrap(args: argparse.Namespace) -> None:
    if not args.skip_data:
        sync_args = [
            "download-defaults",
            "--org",
            args.org,
        ]
        if args.dry_run:
            sync_args.append("--dry-run")

        _python_module(
            "src.collection.hf_sync",
            *sync_args,
        )

    if not args.skip_models:
        model_args = [
            "download-models",
            "--org",
            args.org,
        ]
        if args.dry_run:
            model_args.append("--dry-run")

        _python_module(
            "src.collection.hf_sync",
            *model_args,
        )


def cmd_collect_live(_args: argparse.Namespace) -> None:
    _python_module("src.collection.visa")
    _python_module("src.collection.encounter")
    _python_module("src.collection.trends")
    _python_module("src.processing.parse")


def cmd_sync_data(args: argparse.Namespace) -> None:
    sync_args = [
        "download-defaults",
        "--org",
        args.org,
    ]
    if args.dry_run:
        sync_args.append("--dry-run")

    _python_module(
        "src.collection.hf_sync",
        *sync_args,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migration project CLI for reproducible setup and data operations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap",
        help="Download default datasets and model artifacts from Hugging Face",
    )
    bootstrap.add_argument("--org", default="sdsc2005-migration")
    bootstrap.add_argument("--skip-data", action="store_true")
    bootstrap.add_argument("--skip-models", action="store_true")
    bootstrap.add_argument("--dry-run", action="store_true")
    bootstrap.set_defaults(func=cmd_bootstrap)

    collect_live = subparsers.add_parser(
        "collect-live",
        help="Collect live source data and regenerate processed visa parquet",
    )
    collect_live.set_defaults(func=cmd_collect_live)

    sync_data = subparsers.add_parser(
        "sync-data",
        help="Download default datasets from Hugging Face",
    )
    sync_data.add_argument("--org", default="sdsc2005-migration")
    sync_data.add_argument("--dry-run", action="store_true")
    sync_data.set_defaults(func=cmd_sync_data)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
