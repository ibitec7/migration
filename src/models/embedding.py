from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

try:
	from tqdm.auto import tqdm
except ModuleNotFoundError:
	tqdm = None

try:
	from src.models.jinav5_engine import JinaV5EmbeddingTrtModel
except ModuleNotFoundError:
	from jinav5_engine import JinaV5EmbeddingTrtModel


def setup_logger(log_level: str) -> logging.Logger:
	logger = logging.getLogger("embed_news")
	logger.setLevel(getattr(logging, log_level.upper()))
	if not logger.handlers:
		formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
		handler = logging.StreamHandler()
		handler.setFormatter(formatter)
		logger.addHandler(handler)
	return logger


@dataclass
class FileStats:
	file_path: Path
	total_rows: int
	valid_rows: int
	dropped_rows: int
	embedding_dim: int
	duration_s: float


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate Jina v5 TensorRT embeddings for all parquet files in data/processed/news"
	)
	parser.add_argument(
		"--engine-path",
		type=Path,
		required=True,
		help="Path to TensorRT engine file",
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path("data/processed/news"),
		help="Directory containing input parquet files",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data/processed/news_embeddings"),
		help="Directory where updated parquet files are written",
	)
	parser.add_argument(
		"--glob",
		type=str,
		default="news_*.parquet",
		help="Glob pattern for parquet files",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=4,
		help="Inference batch size (engine max is 4)",
	)
	parser.add_argument(
		"--pad-token-id",
		type=int,
		default=0,
		help="Pad token id",
	)
	parser.add_argument(
		"--input-ids-column",
		type=str,
		default="token_ids",
		help="Column containing token id lists",
	)
	parser.add_argument(
		"--attention-mask-column",
		type=str,
		default="attention_mask",
		help="Column containing attention mask lists",
	)
	parser.add_argument(
		"--embeddings-column",
		type=str,
		default="embeddings",
		help="Output column for embeddings",
	)
	parser.add_argument(
		"--compression",
		type=str,
		default="lz4",
		help="Parquet compression codec",
	)
	parser.add_argument(
		"--max-files",
		type=int,
		default=None,
		help="Optional cap for number of input files",
	)
	parser.add_argument(
		"--max-rows-per-file",
		type=int,
		default=None,
		help="Optional cap for rows processed per file (for smoke tests)",
	)
	parser.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		help="Logger verbosity",
	)
	parser.add_argument(
		"--fail-fast",
		action="store_true",
		help="Stop on first file-level failure",
	)
	parser.add_argument(
		"--no-progress",
		action="store_true",
		help="Disable progress bars",
	)
	parser.add_argument(
		"--overwrite-existing",
		action="store_true",
		help="Reprocess files even if output parquet already exists",
	)
	return parser.parse_args()


def discover_parquet_files(input_dir: Path, pattern: str, max_files: int | None) -> list[Path]:
	files = sorted(input_dir.glob(pattern))
	if max_files is not None:
		files = files[:max_files]
	return files


def validate_args(args: argparse.Namespace) -> None:
	if not args.engine_path.exists():
		raise FileNotFoundError(f"Engine file not found: {args.engine_path}")
	if not args.input_dir.exists() or not args.input_dir.is_dir():
		raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
	if args.batch_size <= 0:
		raise ValueError("--batch-size must be > 0")
	if args.max_files is not None and args.max_files <= 0:
		raise ValueError("--max-files must be > 0")
	if args.max_rows_per_file is not None and args.max_rows_per_file <= 0:
		raise ValueError("--max-rows-per-file must be > 0")


def _build_valid_rows(
	df: pl.DataFrame,
	input_ids_column: str,
	attention_mask_column: str,
	max_seq_len: int,
) -> pl.DataFrame:
	df_idx = df.with_row_index("__row_idx")
	valid_expr = (
		pl.col(input_ids_column).is_not_null()
		& pl.col(attention_mask_column).is_not_null()
		& (pl.col(input_ids_column).list.len() > 0)
		& (pl.col(input_ids_column).list.len() == pl.col(attention_mask_column).list.len())
		& (pl.col(input_ids_column).list.len() <= max_seq_len)
	)
	return df_idx.filter(valid_expr)


def _batched_indices_by_len(lengths: list[int], batch_size: int) -> list[list[int]]:
	order = sorted(range(len(lengths)), key=lengths.__getitem__)
	return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]


def infer_embeddings_for_rows(
	model: JinaV5EmbeddingTrtModel,
	token_rows: list[list[int]],
	attention_rows: list[list[int]],
	pad_token_id: int,
	batch_size: int,
	show_progress: bool = False,
	progress_desc: str = "Batches",
) -> np.ndarray:
	if not token_rows:
		return np.empty((0, 0), dtype=np.float32)

	lengths = [len(tokens) for tokens in token_rows]
	batches = _batched_indices_by_len(lengths, batch_size=batch_size)
	all_embeddings: np.ndarray | None = None
	batch_iterator = batches
	if show_progress and tqdm is not None:
		batch_iterator = tqdm(
			batches,
			total=len(batches),
			desc=progress_desc,
			unit="batch",
			leave=False,
		)

	for batch_indices in batch_iterator:
		actual_size = len(batch_indices)
		target_batch_size = max(model.min_batch_size, actual_size)
		if target_batch_size > model.max_batch_size:
			raise ValueError(
				f"Batch size {target_batch_size} exceeds engine profile max {model.max_batch_size}"
			)

		seq_len = max(lengths[idx] for idx in batch_indices)
		if seq_len > model.max_seq_len:
			raise ValueError(f"Found seq_len={seq_len} above engine max_seq_len={model.max_seq_len}")

		input_ids = torch.full(
			(target_batch_size, seq_len),
			fill_value=pad_token_id,
			dtype=torch.int32,
			device="cpu",
		)
		attention_mask = torch.zeros(
			(target_batch_size, seq_len),
			dtype=torch.int32,
			device="cpu",
		)

		for row_pos, src_idx in enumerate(batch_indices):
			ids = token_rows[src_idx]
			mask = attention_rows[src_idx]
			row_len = len(ids)
			input_ids[row_pos, :row_len] = torch.tensor(ids, dtype=torch.int32)
			attention_mask[row_pos, :row_len] = torch.tensor(mask, dtype=torch.int32)

		embeddings = model.infer(input_ids=input_ids, attention_mask=attention_mask)
		embeddings_cpu = embeddings[:actual_size].detach().float().cpu().numpy()

		if all_embeddings is None:
			all_embeddings = np.empty((len(token_rows), embeddings_cpu.shape[1]), dtype=np.float32)

		for out_pos, src_idx in enumerate(batch_indices):
			all_embeddings[src_idx] = embeddings_cpu[out_pos]

	if all_embeddings is None:
		return np.empty((0, 0), dtype=np.float32)
	return all_embeddings


def process_one_file(
	file_path: Path,
	output_path: Path,
	model: JinaV5EmbeddingTrtModel,
	args: argparse.Namespace,
	show_progress: bool,
) -> FileStats:
	start = time.perf_counter()
	df = pl.read_parquet(file_path)

	if args.max_rows_per_file is not None:
		df = df.head(args.max_rows_per_file)

	required_columns = {args.input_ids_column, args.attention_mask_column}
	missing = required_columns.difference(df.columns)
	if missing:
		raise ValueError(f"Missing required columns in {file_path.name}: {sorted(missing)}")

	valid_df = _build_valid_rows(
		df=df,
		input_ids_column=args.input_ids_column,
		attention_mask_column=args.attention_mask_column,
		max_seq_len=model.max_seq_len,
	)
	dropped_rows = df.height - valid_df.height

	token_rows = valid_df.get_column(args.input_ids_column).to_list()
	attention_rows = valid_df.get_column(args.attention_mask_column).to_list()

	embeddings_array = infer_embeddings_for_rows(
		model=model,
		token_rows=token_rows,
		attention_rows=attention_rows,
		pad_token_id=args.pad_token_id,
		batch_size=args.batch_size,
		show_progress=show_progress,
		progress_desc=f"{file_path.name} batches",
	)
	embedding_dim = 0 if embeddings_array.size == 0 else int(embeddings_array.shape[1])

	output_df = valid_df.drop("__row_idx")
	embeddings_series = pl.Series(
		args.embeddings_column,
		embeddings_array.tolist(),
		dtype=pl.List(pl.Float32),
	)
	output_df = output_df.with_columns(embeddings_series)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_df.lazy().sink_parquet(
		str(output_path),
		compression=args.compression,
	)

	return FileStats(
		file_path=file_path,
		total_rows=df.height,
		valid_rows=output_df.height,
		dropped_rows=dropped_rows,
		embedding_dim=embedding_dim,
		duration_s=time.perf_counter() - start,
	)


def main() -> int:
	args = parse_args()
	logger = setup_logger(args.log_level)
	validate_args(args)

	parquet_files = discover_parquet_files(args.input_dir, args.glob, args.max_files)
	if not parquet_files:
		logger.warning(f"No parquet files found under {args.input_dir} with pattern {args.glob}")
		return 0

	if args.overwrite_existing:
		files_to_process = parquet_files
		skipped_existing = 0
	else:
		files_to_process = [
			file_path
			for file_path in parquet_files
			if not (args.output_dir / file_path.name).exists()
		]
		skipped_existing = len(parquet_files) - len(files_to_process)

	logger.info(f"Discovered {len(parquet_files)} parquet files")
	if skipped_existing:
		logger.info(f"Skipping {skipped_existing} files with existing outputs")
	if not files_to_process:
		logger.info("No files left to process")
		return 0
	show_progress = not args.no_progress
	if show_progress and tqdm is None:
		logger.warning("tqdm is not installed; progress bars are disabled")
		show_progress = False

	model = JinaV5EmbeddingTrtModel(args.engine_path, pad_token_id=args.pad_token_id)
	if args.batch_size > model.max_batch_size:
		raise ValueError(
			f"--batch-size ({args.batch_size}) exceeds engine max batch size ({model.max_batch_size})"
		)

	stats: list[FileStats] = []
	failures = 0

	pipeline_start = time.perf_counter()

	file_iterator = files_to_process
	if show_progress:
		file_iterator = tqdm(files_to_process, total=len(files_to_process), desc="Files", unit="file")

	for index, file_path in enumerate(file_iterator, start=1):
		output_path = args.output_dir / file_path.name
		logger.info(f"[{index}/{len(files_to_process)}] Processing {file_path.name}")
		try:
			file_stats = process_one_file(
				file_path=file_path,
				output_path=output_path,
				model=model,
				args=args,
				show_progress=show_progress,
			)
			stats.append(file_stats)
			throughput = 0.0
			if file_stats.duration_s > 0:
				throughput = file_stats.valid_rows / file_stats.duration_s
			logger.info(
				f"Completed {file_path.name}: rows={file_stats.total_rows}, "
				f"kept={file_stats.valid_rows}, dropped={file_stats.dropped_rows}, "
				f"dim={file_stats.embedding_dim}, throughput={throughput:.2f} rows/s"
			)
		except Exception as exc:  # noqa: BLE001
			failures += 1
			logger.exception(f"Failed on {file_path.name}: {exc}")
			if args.fail_fast:
				raise

	total_time = time.perf_counter() - pipeline_start
	total_rows = sum(item.total_rows for item in stats)
	total_valid_rows = sum(item.valid_rows for item in stats)
	total_dropped_rows = sum(item.dropped_rows for item in stats)
	total_throughput = (total_valid_rows / total_time) if total_time > 0 else 0.0

	logger.info("=" * 80)
	logger.info(
		f"Finished. files_ok={len(stats)}, files_failed={failures}, files_skipped_existing={skipped_existing}, "
		f"rows_total={total_rows}, rows_kept={total_valid_rows}, "
		f"rows_dropped={total_dropped_rows}, elapsed={total_time:.2f}s, "
		f"overall_throughput={total_throughput:.2f} rows/s"
	)
	logger.info("=" * 80)

	return 1 if failures else 0


if __name__ == "__main__":
	raise SystemExit(main())