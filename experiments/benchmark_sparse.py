import torch
import time
import numpy as np
import os
import sys

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

from digital_brain.modules.cortex import CorticalMicrocircuit
from digital_brain.modules.sparse_cortex import SparseCorticalMicrocircuit

def benchmark_forward(module, d_in, n_iters=100, batch_size=1, device='cuda'):
    module.to(device)
    x = torch.randn(batch_size, d_in, device=device)
    state = None
    
    # Warmup
    for _ in range(10):
        _, state = module(x, state)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _, state = module(x, state)
    torch.cuda.synchronize()
    
    return (time.time() - start) * 1000 / n_iters # ms per step

def compare_sparse_dense():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}")
    
    neuron_counts = [128, 256, 512, 1024, 2048, 4096]
    sparsity = 0.05
    d_in = 9
    
    results = {
        'neurons': [],
        'dense_memory_mb': [],
        'sparse_memory_mb': [],
        'dense_time_ms': [],
        'sparse_time_ms': [],
    }
    
    for n in neuron_counts:
        print(f"Testing n={n}...")
        # Dense version
        dense_cortex = CorticalMicrocircuit(d_in=d_in, d_z=n)
        dense_mem = sum(p.numel() * 4 for p in dense_cortex.parameters()) / 1e6
        dense_time = benchmark_forward(dense_cortex, d_in)
        
        # Sparse version
        sparse_cortex = SparseCorticalMicrocircuit(d_in=d_in, d_z=n, sparsity=sparsity)
        sparse_mem = sum(p.numel() * 4 for p in sparse_cortex.parameters()) / 1e6
        sparse_time = benchmark_forward(sparse_cortex, d_in)
        
        results['neurons'].append(n)
        results['dense_memory_mb'].append(dense_mem)
        results['sparse_memory_mb'].append(sparse_mem)
        results['dense_time_ms'].append(dense_time)
        results['sparse_time_ms'].append(sparse_time)
        
        # Free memory
        del dense_cortex, sparse_cortex
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'Neurons':<10} | {'Dense ms':<10} | {'Sparse ms':<10} | {'Mem D (MB)':<10} | {'Mem S (MB)':<10}")
    print("-" * 65)
    for i in range(len(neuron_counts)):
        print(f"{results['neurons'][i]:<10} | {results['dense_time_ms'][i]:<10.2f} | {results['sparse_time_ms'][i]:<10.2f} | {results['dense_memory_mb'][i]:<10.1f} | {results['sparse_memory_mb'][i]:<10.1f}")

    print("\nBenchmark complete. (Plotting skipped due to missing matplotlib)")

if __name__ == "__main__":
    compare_sparse_dense()
