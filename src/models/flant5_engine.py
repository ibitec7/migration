"""
TensorRT inference engine for FLAN-T5-Large (int8 quantized).

This module provides a wrapper around the FLAN-T5-Large encoder-decoder
TensorRT engines built with TensorRT-LLM.
"""

import json
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig

class TensorRTFlanT5Engine:
    """
    TensorRT-LLM inference engine for FLAN-T5-Large (int8 quantized).
    """
    
    DEFAULT_ENGINE_DIR = "tensor-rt/engines/flan-t5-large/int8_wo_cpu/1-gpu"
    
    def __init__(
        self,
        engine_dir: Optional[str] = None,
        device: int = 0,
        lazy_load: bool = False
    ):
        self.device = device
        self._engine_initialized = False
        self.runner = None
        self.tokenizer = None
        self.model_config = None
        
        # Resolve engine directory path
        if engine_dir is None:
            migration_root = Path(__file__).parent.parent.parent
            engine_dir = migration_root / "src" / "models" / self.DEFAULT_ENGINE_DIR
        else:
            engine_dir = Path(engine_dir)
            
        self.engine_dir = engine_dir.resolve()
        
        if not lazy_load:
            self._load_engines()

    def _load_engines(self):
        if self._engine_initialized:
            return
            
        try:
            from tensorrt_llm.runtime import EncDecModelRunner
        except ImportError as e:
            raise ImportError("tensorrt_llm required. Install with NVIDIA pip feed.") from e
            
        model_id = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model_config = AutoConfig.from_pretrained(model_id)
        
        print(f"Loading EncDecModelRunner from {self.engine_dir}...")
        self.runner = EncDecModelRunner.from_engine(
            engine_name="enc_dec",
            engine_dir=str(self.engine_dir),
            debug_mode=False
        )
        self._engine_initialized = True

    def tokenize(self, texts: Union[str, List[str]], max_length: int = 512) -> dict:
        if not self._engine_initialized:
            self._load_engines()
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        return {
            "input_ids": encoded.input_ids.cuda(),
            "attention_mask": encoded.attention_mask.cuda()
        }

    def generate(self, texts: Union[str, List[str]], max_new_tokens: int = 128, max_input_length: int = 512, num_beams: int = 1) -> Union[str, List[str]]:
        if not self._engine_initialized:
            self._load_engines()
            
        encoded = self.tokenize(texts, max_length=max_input_length)
        encoder_input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        batch_size = encoder_input_ids.shape[0]
        decoder_input_ids = torch.IntTensor(
            [[self.model_config.decoder_start_token_id] * batch_size]
        ).t().cuda()

        out = self.runner.generate(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        output_ids = out["output_ids"] if isinstance(out, dict) else out
        
        results = []
        for i in range(batch_size):
            input_len = (attention_mask[i] != 0).sum().item()
            generated_ids = output_ids[i, 0, input_len:].tolist()
            # Clean special tokens
            generated_ids = [tok for tok in generated_ids if tok not in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]]
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(decoded)
            
        return results[0] if isinstance(texts, str) and len(results) == 1 else results

    def forward(self, texts: Union[str, List[str]], max_summary_length: int = 128) -> Union[str, List[str]]:
        return self.generate(texts, max_new_tokens=max_summary_length)
        
    def __call__(self, texts: Union[str, List[str]], max_summary_length: int = 128) -> Union[str, List[str]]:
        return self.generate(texts, max_new_tokens=max_summary_length)
