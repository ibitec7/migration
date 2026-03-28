#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


class _SimpleLogger:
    @staticmethod
    def set_level(_: str) -> None:
        return None

    @staticmethod
    def info(message: str) -> None:
        print(f"[INFO] {message}")

    @staticmethod
    def warning(message: str) -> None:
        print(f"[WARN] {message}")


logger = _SimpleLogger()


DEFAULT_INPUT_DIR = Path("data/processed/news_embeddings")
DEFAULT_OUTPUT_DIR = Path("data/processed/news_embeddings_labeled")
DEFAULT_SUMMARY_PATH = Path("data/processed/cluster_event_labels.csv")


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    default_engine_dir = (
        root_dir / "src/models/tensor-rt/engines/flan-t5-large/int8_wo_cpu/1-gpu"
    )

    parser = argparse.ArgumentParser(
        description="Generate event labels per HDBSCAN cluster using LED TensorRT-LLM"
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary_path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--engine_dir", type=Path, default=default_engine_dir)
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="google/flan-t5-large",
        help="HuggingFace model/tokenizer id or local tokenizer path.",
    )
    parser.add_argument("--cluster_col", type=str, default="cluster")
    parser.add_argument("--label_col", type=str, default="cluster_label")
    parser.add_argument("--sample_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument("--max_output_len", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_noise", action="store_true")
    parser.add_argument(
        "--headlines_only",
        action="store_true",
        help="Use headline-like text only. If text comes from response YAML, extract the title line.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Optional text column override. If omitted, script auto-detects.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Logging level (debug, info, warning, error).",
    )
    return parser.parse_args()


def choose_text_column(df: pl.DataFrame, preferred: str | None = None) -> str:
    if preferred is not None:
        if preferred not in df.columns:
            raise ValueError(f"Requested text column '{preferred}' not found in dataframe")
        return preferred

    candidates = [
        "summary_t5",
        "headline",
        "response",
        "summary",
        "title",
        "content",
        "article",
        "text",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not auto-detect text column. Pass --text_column explicitly. "
        f"Available columns: {df.columns}"
    )


def build_prompt(country: str, cluster_id: int, samples: list[str]) -> str:
    sample_block = "\n\n".join(
        f"Example {idx + 1}:\n{sample.strip()}" for idx, sample in enumerate(samples)
    )
    return (
        f"Headlines from {country}:\n"
        f"{sample_block}\n\n"
        "Question: What is the main action occurring in these titles?\n"
        "Constraint: Answer in exactly 2 or 3 words. Do not use the word 'Migration' or source names.\n"
        "Answer: The main action is "
    )


def normalize_sample_text(text: str, max_chars: int = 900) -> str:
    cleaned = str(text)
    lines = cleaned.splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[A-Za-z_ ]{2,40}:\s", stripped):
            continue
        filtered.append(stripped)
    cleaned = " ".join([line for line in filtered if line])
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def extract_headline_from_response(text: str, max_chars: int = 220) -> str:
    raw = str(text or "")

    title_match = re.search(r"^title:\s*(.+)$", raw, flags=re.MULTILINE | re.IGNORECASE)
    if title_match:
        headline = title_match.group(1).strip()
        return re.sub(r"\s+", " ", headline)[:max_chars]

    first_line = raw.splitlines()[0].strip() if raw.splitlines() else ""
    first_line = re.sub(r"\s+", " ", first_line)
    return first_line[:max_chars]


def normalize_event_label(raw_label: str, country: str) -> str:
    label = str(raw_label or "").strip()
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"[^A-Za-z0-9\-\s]", "", label).strip()

    parts = label.split()
    if len(parts) > 6:
        label = " ".join(parts[:6])
        parts = label.split()

    country_token = country.replace("_", " ").strip().lower()
    if not label or len(parts) < 2 or label.lower() == country_token:
        return "Migration event update"

    if re.search(r"(.)\1\1\1", label):
        return "Migration event update"

    return label


def decode_output_ids(
    tokenizer: AutoTokenizer,
    output_ids: torch.Tensor,
    decoder_start_token_id: int | None,
) -> list[str]:
    texts: list[str] = []
    cpu_ids = output_ids.detach().cpu()

    if cpu_ids.dim() == 2:
        cpu_ids = cpu_ids.unsqueeze(1)

    for row in cpu_ids:
        token_ids = row[0].tolist()
        filtered = []
        for tok in token_ids:
            if tok in {
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
                tokenizer.bos_token_id,
                decoder_start_token_id,
            }:
                continue
            filtered.append(tok)

        text = tokenizer.decode(filtered, skip_special_tokens=True).strip()
        texts.append(text)

    return texts


class LedClusterLabeler:
    def __init__(
        self,
        engine_dir: Path,
        tokenizer_dir: str,
        max_input_tokens: int,
        max_output_len: int,
        num_beams: int,
        batch_size: int,
    ) -> None:
        try:
            from tensorrt_llm.runtime import EncDecModelRunner
        except Exception as exc:
            raise RuntimeError(
                "Failed to import TensorRT-LLM runtime. Ensure TensorRT-LLM and MPI dependencies are installed."
            ) from exc

        encoder_engine = engine_dir / "encoder/rank0.engine"
        decoder_engine = engine_dir / "decoder/rank0.engine"
        if not encoder_engine.exists() or not decoder_engine.exists():
            raise FileNotFoundError(
                f"Missing LED engines under {engine_dir}. "
                f"Expected {encoder_engine} and {decoder_engine}."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.model_config = AutoConfig.from_pretrained(tokenizer_dir)
        self.runner = EncDecModelRunner.from_engine(
            engine_name="enc_dec",
            engine_dir=str(engine_dir),
            debug_mode=False,
        )
        self.max_input_tokens = max_input_tokens
        self.max_output_len = max_output_len
        self.num_beams = num_beams
        self.batch_size = batch_size

    @staticmethod
    def _extract_engine_max_input_tokens(error_text: str) -> int | None:
        match = re.search(r"engine supports \[min, opt, max\]\s*=\s*\[\(\d+,\),\s*\(\d+,\),\s*\((\d+),\)\]", error_text)
        if match:
            return int(match.group(1))
        return None

    def generate_labels(self, prompts: list[str], progress_desc: str | None = None) -> list[str]:
        if not prompts:
            return []

        all_outputs: list[str] = []
        total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        batch_iterator = tqdm(
            range(0, len(prompts), self.batch_size),
            total=total_batches,
            desc=progress_desc or "Generating labels",
            unit="batch",
            leave=False,
        )

        for start in batch_iterator:
            batch_prompts = prompts[start : start + self.batch_size]
            current_max_input_tokens = self.max_input_tokens
            max_input_retry_done = False

            while True:
                tokenized = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=current_max_input_tokens,
                )

                encoder_input_ids = tokenized.input_ids.type(torch.IntTensor).cuda()
                attention_mask = tokenized.attention_mask.cuda()

                decoder_start_token_id = self.model_config.decoder_start_token_id
                decoder_input_ids = torch.full(
                    (encoder_input_ids.shape[0], 1),
                    fill_value=decoder_start_token_id,
                    dtype=torch.int32,
                    device="cuda",
                )

                try:
                    outputs = self.runner.generate(
                        encoder_input_ids=encoder_input_ids,
                        decoder_input_ids=decoder_input_ids,
                        max_new_tokens=self.max_output_len,
                        num_beams=self.num_beams,
                        bos_token_id=self.tokenizer.bos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        attention_mask=attention_mask,
                        debug_mode=False,
                        return_dict=True,
                    )
                    torch.cuda.synchronize()
                    break
                except ValueError as exc:
                    supported_max = self._extract_engine_max_input_tokens(str(exc))
                    if max_input_retry_done or supported_max is None:
                        raise
                    logger.warning(
                        f"Engine max input length is {supported_max}; retrying batch with truncated inputs."
                    )
                    current_max_input_tokens = min(current_max_input_tokens, supported_max)
                    max_input_retry_done = True

            output_ids = outputs["output_ids"] if isinstance(outputs, dict) else outputs
            decoded = decode_output_ids(
                tokenizer=self.tokenizer,
                output_ids=output_ids,
                decoder_start_token_id=decoder_start_token_id,
            )

            for text in decoded:
                all_outputs.append(text if text else "Unlabeled Event")

        return all_outputs


def collect_cluster_prompts(
    df: pl.DataFrame,
    country: str,
    cluster_col: str,
    text_col: str,
    sample_size: int,
    include_noise: bool,
    seed: int,
    headlines_only: bool,
) -> tuple[list[dict], list[str]]:
    cluster_values = sorted(
        int(v)
        for v in df.get_column(cluster_col).drop_nulls().unique().to_list()
        if include_noise or int(v) != -1
    )

    records: list[dict] = []
    prompts: list[str] = []

    for cluster_id in cluster_values:
        subset = df.filter(pl.col(cluster_col) == cluster_id)
        subset = subset.with_columns(pl.col(text_col).cast(pl.Utf8).alias(text_col))
        subset = subset.filter(
            pl.col(text_col).is_not_null() & (pl.col(text_col).str.strip_chars() != "")
        )

        if subset.height == 0:
            continue

        sampled = (
            subset.sample(n=sample_size, seed=seed)
            if subset.height > sample_size
            else subset
        )
        raw_samples = sampled.get_column(text_col).to_list()
        if headlines_only:
            samples = [extract_headline_from_response(s) for s in raw_samples]
        else:
            samples = [normalize_sample_text(s) for s in raw_samples]
        samples = [s for s in samples if s]

        if not samples:
            continue

        prompt = build_prompt(country=country, cluster_id=cluster_id, samples=samples)
        records.append(
            {
                "country": country,
                "cluster": cluster_id,
                "sample_count": len(samples),
                "prompt": prompt,
            }
        )
        prompts.append(prompt)

    return records, prompts


def file_country_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("news_"):
        return stem.replace("news_", "", 1)
    return stem


def run(args: argparse.Namespace) -> None:
    logger.set_level(args.log_level)

    input_dir = args.input_dir
    output_dir = args.output_dir
    summary_path = args.summary_path

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found under: {input_dir}")
        return

    labeler = LedClusterLabeler(
        engine_dir=args.engine_dir,
        tokenizer_dir=args.tokenizer_dir,
        max_input_tokens=args.max_input_tokens,
        max_output_len=args.max_output_len,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
    )

    cluster_summary_records: list[dict] = []

    for parquet_path in tqdm(parquet_files, desc="Labeling clusters", unit="file"):
        df = pl.read_parquet(parquet_path)

        if args.cluster_col not in df.columns:
            logger.warning(f"Skipping {parquet_path.name}: missing '{args.cluster_col}' column")
            continue

        text_col = choose_text_column(df, args.text_column)
        country = file_country_name(parquet_path)

        records, prompts = collect_cluster_prompts(
            df=df,
            country=country,
            cluster_col=args.cluster_col,
            text_col=text_col,
            sample_size=args.sample_size,
            include_noise=args.include_noise,
            seed=args.seed,
            headlines_only=args.headlines_only,
        )

        if not records:
            logger.warning(f"Skipping {parquet_path.name}: no clusters with usable text")
            continue

        labels = labeler.generate_labels(prompts, progress_desc=f"Label batches ({country})")
        for idx, label in enumerate(labels):
            records[idx][args.label_col] = normalize_event_label(
                raw_label=label,
                country=records[idx]["country"],
            )

        mapping_df = pl.DataFrame(
            {
                args.cluster_col: [r["cluster"] for r in records],
                args.label_col: [r[args.label_col] for r in records],
            }
        )

        labeled_df = df.join(mapping_df, on=args.cluster_col, how="left")
        out_path = output_dir / parquet_path.name
        labeled_df.write_parquet(out_path)

        for row in records:
            cluster_summary_records.append(
                {
                    "source_file": parquet_path.name,
                    "country": row["country"],
                    "cluster": row["cluster"],
                    "sample_count": row["sample_count"],
                    args.label_col: row[args.label_col],
                }
            )

    if cluster_summary_records:
        summary_df = pl.DataFrame(cluster_summary_records).sort(["country", "cluster"])
        summary_df.write_csv(summary_path)
        logger.info(f"Saved cluster label summary to: {summary_path}")
    else:
        logger.warning("No cluster labels were generated.")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run(parse_args())
