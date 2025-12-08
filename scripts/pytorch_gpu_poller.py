#!/usr/bin/env python3
"""
PyTorch GPU allocator poller: write JSONL of allocator stats every N seconds.
Usage: python scripts/pytorch_gpu_poller.py --interval 1 --out pytorch_gpu_poll.jsonl
"""
import time
import json
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--interval", type=float, default=1.0)
parser.add_argument("--out", type=str, default="pytorch_gpu_poll.jsonl")
args = parser.parse_args()

with open(args.out, "a") as f:
    try:
        while True:
            ts = time.time()
            allocated = float(torch.cuda.memory_allocated()) / 1e6 if torch.cuda.is_available() else 0.0
            reserved = float(torch.cuda.memory_reserved()) / 1e6 if torch.cuda.is_available() else 0.0
            peak_alloc = float(torch.cuda.max_memory_allocated()) / 1e6 if torch.cuda.is_available() else 0.0
            peak_reserved = float(torch.cuda.max_memory_reserved()) / 1e6 if torch.cuda.is_available() else 0.0
            f.write(json.dumps({
                "ts": ts,
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "peak_alloc_mb": peak_alloc,
                "peak_reserved_mb": peak_reserved
            }) + "\n")
            f.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
