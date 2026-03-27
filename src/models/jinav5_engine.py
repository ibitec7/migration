#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)


def _trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


class JinaV5EmbeddingTrtModel:
    """Inference wrapper for a Jina v5 TensorRT embedding engine.

    Expected engine profile for the requested build:
    - max batch size: 4
    - max sequence length: 8192

    Dynamic shapes are supported. Shorter sequences are accepted directly, and
    sequences shorter than the profile minimum are padded to the profile minimum.
    """

    def __init__(self, engine_path: str | Path, device: str = "cuda", pad_token_id: int = 0):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine not found at {self.engine_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TensorRT inference")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"Only CUDA devices are supported, got {self.device}")
        self.pad_token_id = int(pad_token_id)
        self.stream = torch.cuda.Stream(device=self.device)

        self.runtime = trt.Runtime(LOGGER)
        with open(self.engine_path, "rb") as engine_file:
            self.engine = self.runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize TensorRT engine. Rebuild with a compatible TensorRT version.")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.input_names = []
        self.output_names = []
        for index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(index)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            if tensor_mode == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor_name)
            else:
                self.output_names.append(tensor_name)

        if "input_ids" not in self.input_names or "attention_mask" not in self.input_names:
            raise RuntimeError(f"Engine inputs must include input_ids and attention_mask. Got: {self.input_names}")
        if "embeddings" not in self.output_names:
            raise RuntimeError(f"Engine outputs must include embeddings. Got: {self.output_names}")

        min_shape, _, max_shape = self.engine.get_tensor_profile_shape("input_ids", 0)
        self.min_batch_size = int(min_shape[0])
        self.min_seq_len = int(min_shape[1])
        self.max_batch_size = int(max_shape[0])
        self.max_seq_len = int(max_shape[1])

    def _prepare_inputs(self,
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 tensor [batch, seq]")

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).to(dtype=torch.int32)
        elif attention_mask.ndim != 2:
            raise ValueError("attention_mask must be a rank-2 tensor [batch, seq]")

        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids and attention_mask must have the same shape. Got {input_ids.shape} vs {attention_mask.shape}")

        batch_size = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[1])

        if batch_size < self.min_batch_size or batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} is outside profile [{self.min_batch_size}, {self.max_batch_size}]")
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds profile max {self.max_seq_len}")

        input_ids_cuda = input_ids.contiguous().to(device=self.device, dtype=torch.int32)
        attention_mask_cuda = attention_mask.contiguous().to(device=self.device, dtype=torch.int32)

        if seq_len < self.min_seq_len:
            pad_tokens = self.min_seq_len - seq_len
            input_ids_cuda = torch.nn.functional.pad(input_ids_cuda, (0, pad_tokens), mode="constant", value=0)
            attention_mask_cuda = torch.nn.functional.pad(attention_mask_cuda,
                                                          (0, pad_tokens),
                                                          mode="constant",
                                                          value=0)

        return input_ids_cuda, attention_mask_cuda

    @torch.inference_mode()
    def infer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        input_ids_cuda, attention_mask_cuda = self._prepare_inputs(input_ids, attention_mask)

        self.context.set_input_shape("input_ids", tuple(input_ids_cuda.shape))
        self.context.set_input_shape("attention_mask", tuple(attention_mask_cuda.shape))

        self.context.set_tensor_address("input_ids", int(input_ids_cuda.data_ptr()))
        self.context.set_tensor_address("attention_mask", int(attention_mask_cuda.data_ptr()))

        output_shape = tuple(self.context.get_tensor_shape("embeddings"))
        output_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype("embeddings"))
        embeddings = torch.empty(output_shape, device=self.device, dtype=output_dtype)
        self.context.set_tensor_address("embeddings", int(embeddings.data_ptr()))

        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT inference failed")
        self.stream.synchronize()
        return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for Jina v5 TensorRT embedding engine")
    parser.add_argument("--engine_path",
                        type=str,
                        default="user_artifacts/jina_embed_trt_8192_bs32/model_int8.engine",
                        help="Path to TensorRT engine file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for dummy input")
    parser.add_argument("--seq_len", type=int, default=8192, help="Sequence length for dummy input")
    parser.add_argument("--vocab_size", type=int, default=32768, help="Vocab size for dummy token generation")
    parser.add_argument("--pad_token_id",
                        type=int,
                        default=0,
                        help="Pad token id used when inferring attention_mask")
    parser.add_argument("--infer_attention_mask",
                        action="store_true",
                        help="Do not provide attention_mask; infer it from input_ids != pad_token_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = JinaV5EmbeddingTrtModel(args.engine_path, pad_token_id=args.pad_token_id)

    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), dtype=torch.int32)
    if args.seq_len > 16:
        input_ids[:, -16:] = args.pad_token_id

    attention_mask = None
    if not args.infer_attention_mask:
        attention_mask = (input_ids != args.pad_token_id).to(dtype=torch.int32)

    embeddings = model.infer(input_ids=input_ids, attention_mask=attention_mask)
    print(f"Engine profile: batch [{model.min_batch_size}, {model.max_batch_size}], "
          f"seq [{model.min_seq_len}, {model.max_seq_len}]")
    print(f"Output shape: {tuple(embeddings.shape)}")
    print(f"Output dtype: {embeddings.dtype}")


if __name__ == "__main__":
    main()