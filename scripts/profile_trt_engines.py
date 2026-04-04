import time
import threading
import numpy as np
import torch
import pynvml
import pandas as pd
from tabulate import tabulate
import sys
import os

# Add src to path to allow importing models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.flant5_engine import TensorRTFlanT5Engine
from src.models.jinav5_engine import JinaV5EmbeddingTrtModel

class GPUProfiler:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.running = False
        self.peak_mem = 0
        self.utilizations = []
        
    def _monitor(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        while self.running:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.peak_mem = max(self.peak_mem, mem_info.used / (1024 * 1024))
            self.utilizations.append(util_info.gpu)
            time.sleep(0.01)  # poll every 10ms
            
    def start(self):
        self.peak_mem = 0
        self.utilizations = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        self.running = False
        self.thread.join()
        pynvml.nvmlShutdown()
        avg_util = sum(self.utilizations) / max(len(self.utilizations), 1)
        peak_util = max(self.utilizations) if self.utilizations else 0
        return self.peak_mem, peak_util, avg_util

def profile_flan_t5(batch_sizes=[1, 4, 8], seq_len=512):
    print("\nLoading Flan-T5 Engine...")
    try:
        engine = TensorRTFlanT5Engine()
        engine._load_engines()
    except Exception as e:
        print(f"Failed to load Flan-T5: {e}")
        return []
        
    results = []
    profiler = GPUProfiler()
    
    for bs in batch_sizes:
        print(f"Profiling Flan-T5 (End-to-End Generate) [bs={bs}, input_len={seq_len}]...")
        # Dummy inputs for generate
        dummy_texts = ["test text " * (seq_len // 2)] * bs
        
        # Warmup
        try:
            _ = engine.generate(dummy_texts, max_new_tokens=32, max_input_length=seq_len)
        except Exception as e:
            print(f"Flan-T5 generation failed for bs={bs}: {e}")
            continue
            
        profiler.start()
        start_time = time.time()
        for _ in range(3):  # average over 3 runs
            _ = engine.generate(dummy_texts, max_new_tokens=32, max_input_length=seq_len)
            torch.cuda.synchronize()
        end_time = time.time()
        
        peak_mem, peak_util, avg_util = profiler.stop()
        latency_ms = (end_time - start_time) * 1000 / 3
        
        # approximate tokens per sec
        # encoder reads bs * seq_len
        # decoder generates bs * 32
        # We can just say output tokens per sec
        tokens_per_sec = (bs * 32 * 3) / (end_time - start_time)
        
        results.append({
            "Model Phase": "Flan-T5 E2E (ins=512, outs=32)",
            "Batch Size": bs,
            "Seq Len": seq_len,
            "Tokens/sec (out)": round(tokens_per_sec, 2),
            "Latency (ms)": round(latency_ms, 2),
            "Peak Mem (MB)": round(peak_mem, 2),
            "Peak GPU Util (%)": round(peak_util, 2)
        })
        
    return results

def profile_jina_v5(batch_sizes=[1, 2, 4], seq_len=8192):
    print("\nLoading Jina-v5 Engine...")
    base_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'tensor-rt', 'engines')
    engine_path = os.path.join(base_path, 'jina_embed_trt_8192_bs32', 'model_int8.engine')
    if not os.path.exists(engine_path):
        print(f"Warning: Jina engine not found at {engine_path}. Assuming default or skipping...")
        
    try:
        engine = JinaV5EmbeddingTrtModel(engine_path=engine_path)
    except Exception as e:
        print(f"Failed to load Jina-v5: {e}")
        return []
        
    results = []
    profiler = GPUProfiler()
    
    for bs in batch_sizes:
        print(f"Profiling Jina-v5 [bs={bs}, seq_len={seq_len}]...")
        input_ids = torch.ones((bs, seq_len), dtype=torch.int32, device="cuda")
        attention_mask = torch.ones((bs, seq_len), dtype=torch.int32, device="cuda")
        
        # Warmup
        try:
            _ = engine.infer(input_ids, attention_mask)
        except Exception as e:
            print(f"Jina-v5 infer failed for bs={bs}: {e}")
            continue
            
        profiler.start()
        start_time = time.time()
        for _ in range(5):  # average over 5 runs
            _ = engine.infer(input_ids, attention_mask)
            torch.cuda.synchronize()
        end_time = time.time()
        
        peak_mem, peak_util, avg_util = profiler.stop()
        latency_ms = (end_time - start_time) * 1000 / 5
        tokens_per_sec = (bs * seq_len * 5) / (end_time - start_time)
        
        results.append({
            "Model Phase": "Jina-v5 Embed",
            "Batch Size": bs,
            "Seq Len": seq_len,
            "Tokens/sec": round(tokens_per_sec, 2),
            "Latency (ms)": round(latency_ms, 2),
            "Peak Mem (MB)": round(peak_mem, 2),
            "Peak GPU Util (%)": round(peak_util, 2)
        })
        
    return results

def main():
    print("Starting TensorRT Profile...")
    all_results = []
    
    results = profile_flan_t5(batch_sizes=[1, 4, 8], seq_len=512)
    all_results.extend(results)
    
    results = profile_jina_v5(batch_sizes=[1, 2, 4], seq_len=8192)
    all_results.extend(results)
    
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        print("\n" + "="*80)
        print("PROFILING RESULTS")
        print("="*80)
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        print("="*80)
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
