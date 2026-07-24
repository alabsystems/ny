#!/usr/bin/env python3
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Benchmark comparison: ny (Rust) vs Auto-LiRPA (Python)

Tests IBP bound propagation on equivalent workloads.
"""

import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# MLP benchmark dimensions matching ny synthetic bench
BATCH = 1
SEQ = 16
HIDDEN = 384
INTERMEDIATE = 1536
HEADS = 6
HEAD_DIM = 64

class SimpleMLP(nn.Module):
    """MLP matching ny benchmark dimensions"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()  # Use ReLU since Auto-LiRPA has best support for it
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Linear(nn.Module):
    """Single linear layer"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def benchmark_autolirpa_ibp(model, input_shape, epsilon, warmup=5, iterations=100):
    """Benchmark Auto-LiRPA IBP"""
    model.eval()
    x = torch.randn(input_shape)
    ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    bounded_x = BoundedTensor(x, ptb)

    # Wrap model
    bounded_model = BoundedModule(model, x)

    # Warmup
    for _ in range(warmup):
        lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method='IBP')

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method='IBP')
        times.append(time.perf_counter() - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'iterations': iterations,
        'bound_width': (ub - lb).mean().item()
    }

def parse_time_to_ms(time_str):
    """Parse time string to milliseconds (handles ms, µs, s)"""
    time_str = time_str.strip()
    if 'µs' in time_str:
        return float(time_str.replace('µs', '').strip()) / 1000.0
    if 'ms' in time_str:
        return float(time_str.replace('ms', '').strip())
    if 's' in time_str:
        return float(time_str.replace('s', '').strip()) * 1000.0
    return float(time_str)

def run_ny_bench():
    """Run ny synthetic benchmark and parse output"""
    cmd = ['./target/release/ny', 'bench', '--benchmark', 'layer']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    # Parse results from output
    results = {}
    for line in result.stdout.split('\n'):
        if 'Linear IBP [384->1536]:' in line:
            time_part = line.split(':')[1].split('per')[0]
            results['linear_up'] = parse_time_to_ms(time_part)
        elif 'Linear IBP [1536->384]:' in line:
            time_part = line.split(':')[1].split('per')[0]
            results['linear_down'] = parse_time_to_ms(time_part)
        elif 'GELU IBP' in line:
            time_part = line.split(':')[1].split('per')[0]
            results['gelu'] = parse_time_to_ms(time_part)
        elif 'Full MLP IBP' in line:
            time_part = line.split(':')[1].split('per')[0]
            results['mlp'] = parse_time_to_ms(time_part)

    return results

def main():
    print("="*70)
    print("ny vs Auto-LiRPA IBP Benchmark Comparison")
    print("="*70)
    print(f"\nDimensions: batch={BATCH}, seq={SEQ}, hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    print(f"Input shape: ({BATCH}, {SEQ}, {HIDDEN})")
    print()

    epsilon = 0.01

    # Benchmark ny (Rust)
    print("Running ny (Rust) benchmark...")
    ny_results = run_ny_bench()
    print(f"  Linear [384->1536]: {ny_results.get('linear_up', 'N/A'):.3f} ms")
    print(f"  GELU [1536]:        {ny_results.get('gelu', 'N/A'):.3f} ms")
    print(f"  Linear [1536->384]: {ny_results.get('linear_down', 'N/A'):.3f} ms")
    print(f"  Full MLP:           {ny_results.get('mlp', 'N/A'):.3f} ms")

    # Benchmark Auto-LiRPA (Python)
    print("\nRunning Auto-LiRPA (Python) benchmark...")

    # Linear up
    model = Linear(HIDDEN, INTERMEDIATE)
    result = benchmark_autolirpa_ibp(model, (BATCH * SEQ, HIDDEN), epsilon)
    autolirpa_linear_up = result['mean_ms']
    print(f"  Linear [384->1536]: {result['mean_ms']:.3f} ms (±{result['std_ms']:.3f})")

    # ReLU (GELU not supported by Auto-LiRPA ONNX export)
    model = nn.Sequential(nn.ReLU())
    result = benchmark_autolirpa_ibp(model, (BATCH * SEQ, INTERMEDIATE), epsilon)
    autolirpa_relu = result['mean_ms']
    print(f"  ReLU [1536]:        {result['mean_ms']:.3f} ms (±{result['std_ms']:.3f})")

    # Linear down
    model = Linear(INTERMEDIATE, HIDDEN)
    result = benchmark_autolirpa_ibp(model, (BATCH * SEQ, INTERMEDIATE), epsilon)
    autolirpa_linear_down = result['mean_ms']
    print(f"  Linear [1536->384]: {result['mean_ms']:.3f} ms (±{result['std_ms']:.3f})")

    # Full MLP
    model = SimpleMLP(HIDDEN, INTERMEDIATE, HIDDEN)
    result = benchmark_autolirpa_ibp(model, (BATCH * SEQ, HIDDEN), epsilon)
    autolirpa_mlp = result['mean_ms']
    print(f"  Full MLP:           {result['mean_ms']:.3f} ms (±{result['std_ms']:.3f})")

    # Comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Layer':<25} {'Auto-LiRPA (ms)':<18} {'ny (ms)':<18} {'Speedup':<10}")
    print("-"*70)

    comparisons = [
        ('Linear [384->1536]', autolirpa_linear_up, ny_results.get('linear_up', 0)),
        ('ReLU [1536]', autolirpa_relu, ny_results.get('gelu', 0)),  # ny has GELU, compare to ReLU
        ('Linear [1536->384]', autolirpa_linear_down, ny_results.get('linear_down', 0)),
        ('Full MLP (ReLU)', autolirpa_mlp, ny_results.get('mlp', 0)),  # Auto-LiRPA uses ReLU MLP
    ]

    total_autolirpa = 0
    total_ny = 0

    for name, al_time, ny_time in comparisons:
        if ny_time > 0:
            speedup = al_time / ny_time
            print(f"{name:<25} {al_time:<18.3f} {ny_time:<18.3f} {speedup:<10.1f}x")
            total_autolirpa += al_time
            total_ny += ny_time
        else:
            print(f"{name:<25} {al_time:<18.3f} {'N/A':<18} {'N/A':<10}")

    print("-"*70)
    if total_ny > 0:
        print(f"{'Total':<25} {total_autolirpa:<18.3f} {total_ny:<18.3f} {total_autolirpa/total_ny:<10.1f}x")

    print("\nNOTE: ny times include 100 iterations averaged. Auto-LiRPA uses same iteration count.")

if __name__ == '__main__':
    main()
