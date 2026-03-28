#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import EncDecModelRunner


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_engine_dir = root_dir / "artifacts/led-base-16384/trt_engines/int4_wo/1-gpu/int4_wo_b4_s16384"

    parser = argparse.ArgumentParser(
        description="Run inference with the built LED TensorRT-LLM encoder-decoder engines."
    )
    parser.add_argument(
        "--engine_dir",
        type=Path,
        default=default_engine_dir,
        help="Directory containing encoder/decoder engine subdirectories.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="allenai/led-base-16384",
        help="HuggingFace model/tokenizer id or local tokenizer path.",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        nargs="+",
        default=["summarize: NVIDIA announced new GPU software updates for long-context inference."],
        help="Input prompt text (one or more strings).",
    )
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=128,
        help="Maximum generated output tokens.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam search width (1 = greedy decoding).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="error",
        help="Logging level (debug, info, warning, error).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.set_level(args.log_level)

    # Verify engine files exist
    engine_dir = args.engine_dir
    encoder_engine = engine_dir / "encoder/rank0.engine"
    decoder_engine = engine_dir / "decoder/rank0.engine"
    if not encoder_engine.exists() or not decoder_engine.exists():
        raise FileNotFoundError(
            f"Could not find encoder/decoder engines under: {engine_dir}\n"
            f"Expected: {encoder_engine} and {decoder_engine}"
        )

    # Handle input_text (could be list or single string)
    if isinstance(args.input_text, list):
        input_text = args.input_text
    else:
        input_text = [args.input_text]

    logger.info(f"Loading tokenizer from: {args.tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    logger.info(f"Loading LED model from engines: {engine_dir}")
    model_config = AutoConfig.from_pretrained(args.tokenizer_dir)
    
    # Load the encoder-decoder model runtime
    tllm_model = EncDecModelRunner.from_engine(
        engine_name="enc_dec",
        engine_dir=str(engine_dir),
        debug_mode=False,
    )

    logger.info(f"Model loaded. Inference dtype: {tllm_model.encoder_model_config.dtype}")

    # Tokenize input text
    logger.info(f"Tokenizing {len(input_text)} input(s)...")
    tokenized_inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoder_input_ids = tokenized_inputs.input_ids.type(torch.IntTensor).cuda()
    attention_mask = tokenized_inputs.attention_mask.cuda()

    # Prepare decoder start token
    decoder_input_ids = torch.IntTensor(
        [[model_config.decoder_start_token_id] * encoder_input_ids.shape[0]]
    ).t().cuda()

    logger.info(f"Input shape: {encoder_input_ids.shape}")
    logger.info(f"Input text: {input_text}")
    logger.info(
        f"Running inference with max_output_len={args.max_output_len}, num_beams={args.num_beams}"
    )

    # Run inference
    tllm_output = tllm_model.generate(
        encoder_input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=args.max_output_len,
        num_beams=args.num_beams,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        debug_mode=False,
        return_dict=True,
    )
    torch.cuda.synchronize()

    # Extract and decode output
    if isinstance(tllm_output, dict):
        output_ids = tllm_output["output_ids"]
    else:
        output_ids = tllm_output

    # Decode outputs (remove input tokens and decode)
    batch_size = encoder_input_ids.shape[0]
    results = []
    for i in range(batch_size):
        # Get generated tokens (after input length)
        input_len = (attention_mask[i] != 0).sum().item()
        generated_ids = output_ids[i, 0, input_len:].tolist()

        # Remove padding and eos tokens
        generated_ids = [
            tok for tok in generated_ids
            if tok not in [tokenizer.pad_token_id, tokenizer.eos_token_id]
        ]

        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(output_text)

    # Print results
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)
    for i, (inp, out) in enumerate(zip(input_text, results)):
        print(f"\n[Request {i}]")
        print(f"Input:  {inp}")
        print(f"Output: {out}")
    print("\n" + "=" * 80)

    if tensorrt_llm.mpi_rank() == 0:
        logger.info("Inference completed successfully")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
