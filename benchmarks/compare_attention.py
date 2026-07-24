#!/usr/bin/env python3
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Benchmark: Attention IBP - ny vs Auto-LiRPA

Compare bound propagation through attention mechanisms at different scales.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


class SimpleAttention(nn.Module):
    """Simple self-attention for benchmarking"""
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)

def benchmark_autolirpa_attention(embed_dim, num_heads, seq_len, epsilon=0.01, iterations=20):
    """Benchmark Auto-LiRPA IBP on attention"""
    model = SimpleAttention(embed_dim, num_heads, seq_len)
    model.eval()

    x = torch.randn(1, seq_len, embed_dim)
    ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    bounded_x = BoundedTensor(x, ptb)

    try:
        bounded_model = BoundedModule(model, x)

        # Warmup
        for _ in range(3):
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
            'error': None
        }
    except Exception as e:
        return {'error': str(e), 'mean_ms': float('nan')}

def main():
    print("="*70)
    print("Attention IBP Benchmark: ny vs Auto-LiRPA")
    print("="*70)

    embed_dim = 384
    num_heads = 6
    epsilon = 0.01

    results = []

    for seq_len in [4, 16, 64, 128]:
        print(f"\n--- Sequence Length: {seq_len} ---")
        print(f"Dimensions: embed={embed_dim}, heads={num_heads}, head_dim={embed_dim//num_heads}")

        # Auto-LiRPA
        print("\nAuto-LiRPA...")
        al_result = benchmark_autolirpa_attention(embed_dim, num_heads, seq_len, epsilon)
        if al_result['error']:
            print(f"  Error: {al_result['error'][:100]}...")
        else:
            print(f"  Time: {al_result['mean_ms']:.2f} ms (±{al_result['std_ms']:.2f})")

        results.append({
            'seq_len': seq_len,
            'auto_lirpa_ms': al_result['mean_ms'],
            'auto_lirpa_error': al_result.get('error')
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (Auto-LiRPA Attention IBP)")
    print("="*70)
    print(f"{'Seq Len':<12} {'Auto-LiRPA (ms)':<20} {'Status':<20}")
    print("-"*52)

    for r in results:
        if r['auto_lirpa_error']:
            print(f"{r['seq_len']:<12} {'ERROR':<20} {r['auto_lirpa_error'][:30]}...")
        else:
            print(f"{r['seq_len']:<12} {r['auto_lirpa_ms']:<20.2f} OK")

    print("\n" + "="*70)
    print("ny Attention Results (from `ny bench --benchmark attention`)")
    print("="*70)
    print("""
MatMul IBP [1,6,16,64] @ [1,6,64,16]: 15.10 ms per iteration
Softmax IBP [1,6,16,16]: 0.12 ms per iteration

MatMul IBP seq=4: 1.62 ms
MatMul IBP seq=16: 18.57 ms
MatMul IBP seq=64: 274.32 ms
""")

if __name__ == '__main__':
    main()
