"""
Per-process VRAM report (debug): what the torch allocator of this process holds on each device it uses.
Called on every tensor-parallel rank through abtest.call_all. Raw cudaMalloc allocations of the
extension and the CUDA context itself are not in these numbers (compare with nvidia-smi).
"""
import os
import torch


def print_mem(tag: str = ""):
    for d in range(torch.cuda.device_count()):
        a, r = torch.cuda.memory_allocated(d), torch.cuda.memory_reserved(d)
        if a == 0 and r == 0:
            continue
        print(f" -- mem {tag} pid {os.getpid()} cuda:{d}: allocated {a / 2**20:.0f} MiB (peak {torch.cuda.max_memory_allocated(d) / 2**20:.0f}), "
              f"reserved {r / 2**20:.0f} (peak {torch.cuda.max_memory_reserved(d) / 2**20:.0f})", flush = True)


def reset_peak():
    for d in range(torch.cuda.device_count()):
        if torch.cuda.memory_reserved(d):
            torch.cuda.reset_peak_memory_stats(d)
