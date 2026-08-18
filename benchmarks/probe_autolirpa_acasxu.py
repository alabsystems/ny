#!/usr/bin/env python3
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Inspect Auto-LiRPA bounds on the ACAS-Xu 1_1 model.

This is an explicit external-tool probe, not a pytest test. It fails nonzero
when its comparator dependencies or committed ACAS-Xu fixture are unavailable.

Run: source .venv/bin/activate && python benchmarks/probe_autolirpa_acasxu.py
"""

from pathlib import Path


def _require_dependencies():
    """Load every dependency required by the explicitly selected probe."""
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    except (ImportError, OSError) as e:
        raise RuntimeError(
            "ACAS-Xu Auto-LiRPA probe prerequisites are missing: numpy, torch, "
            "and auto_LiRPA are required. Install them before selecting this "
            "tool (for example: pip install torch auto-LiRPA)."
        ) from e
    return np, torch, nn, BoundedModule, BoundedTensor, PerturbationLpNorm


def load_nnet(path, np):
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
    (
        np,
        torch,
        nn,
        bounded_module_cls,
        bounded_tensor_cls,
        perturbation_cls,
    ) = _require_dependencies()

    # Load model
    model_path = Path(__file__).resolve().parents[1] / "tests/models/acasxu_1_1.nnet"
    weights, biases, layer_sizes = load_nnet(model_path, np)
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
        except Exception as e:
            raise RuntimeError(f"Auto-LiRPA {name} bound computation failed") from e
        if tuple(lb.shape) != (1, 5) or tuple(ub.shape) != (1, 5):
            raise RuntimeError(
                f"Auto-LiRPA {name} returned wrong bound shapes: "
                f"{tuple(lb.shape)}, {tuple(ub.shape)}"
            )
        if not torch.isfinite(lb).all() or not torch.isfinite(ub).all():
            raise RuntimeError(f"Auto-LiRPA {name} returned non-finite bounds")
        if not torch.all(lb <= ub):
            raise RuntimeError(f"Auto-LiRPA {name} returned inverted bounds")
        print(f'\n{name}:')
        for i in range(5):
            print(f'  Y_{i}: [{lb[0, i].item():.4f}, {ub[0, i].item():.4f}]')
        total_width = (ub - lb).sum().item()
        if not np.isfinite(total_width) or total_width < 0:
            raise RuntimeError(f"Auto-LiRPA {name} returned invalid total width")
        print(f'  Total width: {total_width:.4f}')

    # Property threshold
    threshold = 3.991125645861615
    print('\n=== Property Status ===')
    print(f'Threshold: {threshold:.6f}')
    print(f'Property: Y_0 >= {threshold} is UNSAFE')
    print(f'To verify safety: need upper_bound[0] < {threshold}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
