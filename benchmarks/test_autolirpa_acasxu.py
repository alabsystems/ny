#!/usr/bin/env python3
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Compare ny vs Auto-LiRPA bounds on ACAS-Xu 1_1 model.

Run: source .venv/bin/activate && python benchmarks/test_autolirpa_acasxu.py
"""

import numpy as np


def _require_torch():
    """Lazy import torch - allows pytest collection without torch installed."""
    try:
        import torch
        import torch.nn as nn
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    except ModuleNotFoundError as e:
        raise AssertionError(
            "Missing dependencies: torch and auto_LiRPA required.\n"
            "Install with: pip install torch auto-LiRPA"
        ) from e
    return torch, nn, BoundedModule, BoundedTensor, PerturbationLpNorm


def load_nnet(path):
    """Load NNet format model."""
    with open(path) as f:
        lines = [line for line in f if not line.startswith('//')]

    header = lines[0].strip().strip(',').split(',')
    num_layers = int(header[0])
    layer_sizes = [int(x.strip()) for x in lines[1].strip().strip(',').split(',') if x.strip()]

    line_idx = 7  # Weight data starts at line 7
    weights = []
    biases = []
    for i in range(num_layers):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]

        wmat = np.zeros((out_dim, in_dim))
        for j in range(out_dim):
            row = [float(x) for x in lines[line_idx].strip().strip(',').split(',') if x.strip()]
            wmat[j] = row
            line_idx += 1

        bvec = np.zeros(out_dim)
        for j in range(out_dim):
            bvec[j] = float(lines[line_idx].strip().strip(',').split(',')[0])
            line_idx += 1

        weights.append(wmat)
        biases.append(bvec)

    return weights, biases, layer_sizes


def main():
    torch, nn, bounded_module_cls, bounded_tensor_cls, perturbation_cls = _require_torch()

    # Load model
    weights, biases, layer_sizes = load_nnet('tests/models/acasxu_1_1.nnet')
    print(f'Loaded ACAS-Xu 1_1: {len(weights)} layers, sizes: {layer_sizes}')

    # Create PyTorch model
    layers = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        linear = nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data = torch.tensor(w, dtype=torch.float32)
        linear.bias.data = torch.tensor(b, dtype=torch.float32)
        layers.append(linear)
        if i < len(weights) - 1:
            layers.append(nn.ReLU())

    model = nn.Sequential(*layers)
    model.eval()

    # VNNLIB Property 1 bounds (normalized space)
    lower = torch.tensor([[0.6, -0.5, -0.5, 0.45, -0.5]])
    upper = torch.tensor([[0.679857769, 0.5, 0.5, 0.5, -0.45]])
    center = (lower + upper) / 2

    print('\nInput bounds:')
    for i in range(5):
        print(f'  X_{i}: [{lower[0, i].item():.6f}, {upper[0, i].item():.6f}]')

    # Create bounded model
    bounded_model = bounded_module_cls(model, center)
    ptb = perturbation_cls(x_L=lower, x_U=upper)
    bounded_input = bounded_tensor_cls(center, ptb)

    # Compute bounds
    print('\n=== Auto-LiRPA Bounds ===')

    methods = [
        ('IBP', 'IBP'),
        ('CROWN (backward)', 'backward'),
        ('CROWN-Optimized (alpha)', 'CROWN-Optimized'),
    ]

    for name, method in methods:
        try:
            lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method=method)
            print(f'\n{name}:')
            for i in range(5):
                print(f'  Y_{i}: [{lb[0, i].item():.4f}, {ub[0, i].item():.4f}]')
            print(f'  Total width: {(ub - lb).sum().item():.4f}')
        except Exception as e:
            print(f'\n{name}: Error - {e}')

    # Property threshold
    threshold = 3.991125645861615
    print('\n=== Property Status ===')
    print(f'Threshold: {threshold:.6f}')
    print(f'Property: Y_0 >= {threshold} is UNSAFE')
    print(f'To verify safety: need upper_bound[0] < {threshold}')


if __name__ == '__main__':
    main()
