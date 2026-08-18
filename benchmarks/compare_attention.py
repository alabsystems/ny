#!/usr/bin/env python3
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Benchmark Auto-LiRPA IBP through attention mechanisms at different scales.

This script does not invoke ny and therefore does not report a cross-tool comparison.
"""

import time

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
except (ImportError, OSError) as exc:
    np = None
    torch = None
    nn = None
    BoundedModule = None
    BoundedTensor = None
    PerturbationLpNorm = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_dependencies():
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Attention comparison prerequisites are missing: numpy, torch, and "
            "auto_LiRPA are required. Install them before selecting this explicit "
            "benchmark tool (for example: pip install torch auto-LiRPA)."
        ) from _IMPORT_ERROR


class SimpleAttention(nn.Module if nn is not None else object):
    """Simple self-attention for benchmarking"""
    def __init__(self, embed_dim, num_heads, seq_len):
        _require_dependencies()
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
    _require_dependencies()
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be positive and divisible by num_heads")
    if seq_len <= 0 or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("seq_len must be positive and epsilon finite/non-negative")
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
            if tuple(lb.shape) != (1, seq_len, embed_dim):
                raise RuntimeError(
                    f"Auto-LiRPA warm-up lower-bound shape mismatch: {tuple(lb.shape)}"
                )
            if tuple(ub.shape) != tuple(lb.shape):
                raise RuntimeError("Auto-LiRPA warm-up upper-bound shape mismatch")
            if not torch.isfinite(lb).all() or not torch.isfinite(ub).all():
                raise RuntimeError("Auto-LiRPA warm-up returned non-finite bounds")
            if not torch.all(lb <= ub):
                raise RuntimeError("Auto-LiRPA warm-up returned inverted bounds")

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method='IBP')
            times.append(time.perf_counter() - start)

        if tuple(lb.shape) != (1, seq_len, embed_dim):
            raise RuntimeError(
                f"Auto-LiRPA lower-bound shape mismatch: {tuple(lb.shape)}"
            )
        if tuple(ub.shape) != tuple(lb.shape):
            raise RuntimeError("Auto-LiRPA upper-bound shape mismatch")
        if not torch.isfinite(lb).all() or not torch.isfinite(ub).all():
            raise RuntimeError("Auto-LiRPA returned non-finite attention bounds")
        if not torch.all(lb <= ub):
            raise RuntimeError("Auto-LiRPA returned inverted attention bounds")
    except Exception as e:
        raise RuntimeError(
            f"Auto-LiRPA attention IBP failed for sequence length {seq_len}"
        ) from e
    mean_ms = float(np.mean(times) * 1000)
    std_ms = float(np.std(times) * 1000)
    if not np.isfinite(mean_ms) or mean_ms <= 0:
        raise RuntimeError(f"Auto-LiRPA returned invalid mean timing: {mean_ms}")
    if not np.isfinite(std_ms) or std_ms < 0:
        raise RuntimeError(f"Auto-LiRPA returned invalid timing deviation: {std_ms}")
    return {'mean_ms': mean_ms, 'std_ms': std_ms}

def main():
    _require_dependencies()
    print("="*70)
    print("Auto-LiRPA Attention IBP Diagnostic")
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
        print(f"  Time: {al_result['mean_ms']:.2f} ms (±{al_result['std_ms']:.2f})")

        results.append({
            'seq_len': seq_len,
            'auto_lirpa_ms': al_result['mean_ms'],
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (Auto-LiRPA Attention IBP)")
    print("="*70)
    print(f"{'Seq Len':<12} {'Auto-LiRPA (ms)':<20} {'Status':<20}")
    print("-"*52)

    for r in results:
        print(f"{r['seq_len']:<12} {r['auto_lirpa_ms']:<20.2f} OK")
    if len(results) != 4:
        raise RuntimeError(f"incomplete attention benchmark: {len(results)}/4")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
