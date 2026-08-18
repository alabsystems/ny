#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Validate analytic activation interval formulas against sampled PyTorch outputs.

This is a reference-oracle self-check; it does not invoke ny or establish
coverage of ny's activation implementation. For each activation:

1. Create input bounds covering various ranges
2. Sample many points uniformly within bounds
3. Compute PyTorch outputs for each sample
4. Verify the analytic interval contains all sampled PyTorch outputs

Usage:
    python scripts/validate_activations_vs_pytorch.py
    python scripts/validate_activations_vs_pytorch.py --verbose
    python scripts/validate_activations_vs_pytorch.py --samples 10000
"""

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

@dataclass
class ValidationResult:
    """Result of validating an activation against PyTorch."""
    activation: str
    interval: Tuple[float, float]
    reference_lb: float
    reference_ub: float
    pytorch_min: float
    pytorch_max: float
    soundness_ok: bool
    margin_lower: float  # How much reference_lb is below pytorch_min
    margin_upper: float  # How much reference_ub is above pytorch_max
    num_samples: int


def create_activation_onnx(activation: str, output_path: str):
    """Create a simple ONNX model with a single activation."""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])

    op_type_map = {
        'relu': 'Relu',
        'gelu': 'Gelu',  # ONNX has Gelu in opset 20+
        'tanh': 'Tanh',
        'sigmoid': 'Sigmoid',
        'softplus': 'Softplus',
        'sin': 'Sin',
        'cos': 'Cos',
    }

    op_type = op_type_map.get(activation)
    if op_type is None:
        raise ValueError(f"Unknown activation: {activation}")

    # GELU needs special handling - use Erf-based approximation
    if activation == 'gelu':
        # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        # Create as: x * 0.5 * (1 + Erf(x / sqrt(2)))

        # Constants
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        half = 0.5
        one = 1.0

        sqrt2_inv_init = helper.make_tensor('sqrt2_inv', TensorProto.FLOAT, [], [sqrt2_inv])
        half_init = helper.make_tensor('half', TensorProto.FLOAT, [], [half])
        one_init = helper.make_tensor('one', TensorProto.FLOAT, [], [one])

        nodes = [
            # t1 = x / sqrt(2)
            helper.make_node('Mul', ['input', 'sqrt2_inv'], ['t1'], name='mul_sqrt2'),
            # t2 = erf(t1)
            helper.make_node('Erf', ['t1'], ['t2'], name='erf'),
            # t3 = 1 + t2
            helper.make_node('Add', ['one', 't2'], ['t3'], name='add_one'),
            # t4 = 0.5 * t3
            helper.make_node('Mul', ['half', 't3'], ['t4'], name='mul_half'),
            # output = x * t4
            helper.make_node('Mul', ['input', 't4'], ['output'], name='mul_x'),
        ]

        graph = helper.make_graph(
            nodes,
            'gelu_model',
            [input_tensor],
            [output_tensor],
            [sqrt2_inv_init, half_init, one_init]
        )
    else:
        node = helper.make_node(op_type, ['input'], ['output'], name='activation')
        graph = helper.make_graph([node], f'{activation}_model', [input_tensor], [output_tensor])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 20)])
    onnx.save(model, output_path)


def get_pytorch_activation(activation: str) -> Callable:
    """Get the PyTorch activation function."""
    activations = {
        'relu': torch.relu,
        'gelu': lambda x: torch.nn.functional.gelu(x, approximate='none'),  # Erf-based
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'softplus': torch.nn.functional.softplus,
        'sin': torch.sin,
        'cos': torch.cos,
    }
    return activations.get(activation)


def validate_activation(
    activation: str,
    lower: float,
    upper: float,
    num_samples: int = 1000,
    verbose: bool = False
) -> ValidationResult:
    """Validate an analytic interval against sampled PyTorch outputs."""

    pytorch_fn = get_pytorch_activation(activation)
    if pytorch_fn is None:
        raise ValueError(f"Unknown activation: {activation}")

    # Sample uniformly in [lower, upper]
    samples = torch.linspace(lower, upper, num_samples)
    pytorch_outputs = pytorch_fn(samples).numpy()

    pytorch_min = float(pytorch_outputs.min())
    pytorch_max = float(pytorch_outputs.max())

    # Sanity-check that the reference activation can be serialized to ONNX.
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        model_path = f.name

    try:
        create_activation_onnx(activation, model_path)
        # Compute the analytic interval based on function monotonicity/extrema.
        if activation in ['relu', 'sigmoid', 'tanh', 'softplus']:
            # Monotonically increasing activations: bounds are f(lower), f(upper)
            t_lower = torch.tensor([lower])
            t_upper = torch.tensor([upper])
            reference_lb = float(pytorch_fn(t_lower).item())
            reference_ub = float(pytorch_fn(t_upper).item())

            # For ReLU, need to clamp
            if activation == 'relu':
                reference_lb = max(0.0, lower)
                reference_ub = max(0.0, upper)
        elif activation == 'gelu':
            # GELU is NOT monotonic! It has a local minimum around x ≈ -0.752 where GELU(x) ≈ -0.1699
            # GELU'(x) = Φ(x) + x * φ(x) where Φ is CDF and φ is PDF of standard normal
            # The derivative is 0 at approximately x ≈ -0.752
            # For x < -0.752: GELU is decreasing
            # For x > -0.752: GELU is increasing
            GELU_MIN_X = -0.7522526  # Approximate location of GELU minimum
            GELU_MIN_VAL = -0.16996664  # GELU(GELU_MIN_X)

            t_lower = torch.tensor([lower])
            t_upper = torch.tensor([upper])
            val_lower = float(pytorch_fn(t_lower).item())
            val_upper = float(pytorch_fn(t_upper).item())

            reference_lb = min(val_lower, val_upper)
            reference_ub = max(val_lower, val_upper)

            # If interval contains the minimum point, lower bound is the minimum value
            if lower <= GELU_MIN_X <= upper:
                reference_lb = min(reference_lb, GELU_MIN_VAL)
        elif activation == 'sin':
            # Sin is not monotonic - need to check for extrema in interval
            # sin'(x) = cos(x) = 0 at x = π/2 + kπ
            # sin is max (1) at π/2 + 2kπ, min (-1) at -π/2 + 2kπ
            reference_lb = min(np.sin(lower), np.sin(upper))
            reference_ub = max(np.sin(lower), np.sin(upper))

            # Check if interval contains any extrema
            k_start = int(np.floor((lower - np.pi/2) / np.pi))
            k_end = int(np.ceil((upper - np.pi/2) / np.pi))
            for k in range(k_start, k_end + 1):
                extremum = np.pi/2 + k * np.pi
                if lower <= extremum <= upper:
                    val = np.sin(extremum)
                    reference_lb = min(reference_lb, val)
                    reference_ub = max(reference_ub, val)
        elif activation == 'cos':
            # cos'(x) = -sin(x) = 0 at x = kπ
            # cos is max (1) at 2kπ, min (-1) at π + 2kπ
            reference_lb = min(np.cos(lower), np.cos(upper))
            reference_ub = max(np.cos(lower), np.cos(upper))

            # Check if interval contains any extrema
            k_start = int(np.floor(lower / np.pi))
            k_end = int(np.ceil(upper / np.pi))
            for k in range(k_start, k_end + 1):
                extremum = k * np.pi
                if lower <= extremum <= upper:
                    val = np.cos(extremum)
                    reference_lb = min(reference_lb, val)
                    reference_ub = max(reference_ub, val)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    finally:
        Path(model_path).unlink(missing_ok=True)

    # Check the analytic reference interval contains all sampled outputs.
    # Allow small tolerance for floating point
    tolerance = 1e-5
    soundness_ok = (
        reference_lb <= pytorch_min + tolerance
        and reference_ub >= pytorch_max - tolerance
    )

    margin_lower = pytorch_min - reference_lb
    margin_upper = reference_ub - pytorch_max

    if verbose:
        status = "PASS" if soundness_ok else "FAIL"
        print(f"  [{lower:.2f}, {upper:.2f}]: "
              f"analytic=[{reference_lb:.6f}, {reference_ub:.6f}], "
              f"pytorch=[{pytorch_min:.6f}, {pytorch_max:.6f}], "
              f"margins=[{margin_lower:.6f}, {margin_upper:.6f}] "
              f"{status}")

    return ValidationResult(
        activation=activation,
        interval=(lower, upper),
        reference_lb=reference_lb,
        reference_ub=reference_ub,
        pytorch_min=pytorch_min,
        pytorch_max=pytorch_max,
        soundness_ok=soundness_ok,
        margin_lower=margin_lower,
        margin_upper=margin_upper,
        num_samples=num_samples,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate analytic activation intervals against PyTorch samples"
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Number of samples per interval (default: 1000)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Analytic Activation Interval Validation vs PyTorch")
    print("=" * 70)
    print(f"Samples per interval: {args.samples}")

    # Activations to test
    activations = ['relu', 'gelu', 'tanh', 'sigmoid', 'softplus', 'sin', 'cos']

    # Test intervals covering various ranges
    test_intervals = [
        (-5.0, 5.0),      # Wide symmetric
        (-1.0, 1.0),      # Narrow symmetric
        (0.0, 2.0),       # Positive only
        (-2.0, 0.0),      # Negative only
        (-0.1, 0.1),      # Near zero
        (-10.0, 10.0),    # Very wide
        (1.0, 3.0),       # Positive offset
        (-3.0, -1.0),     # Negative offset
    ]

    # Additional intervals for sin/cos to test extrema handling
    trig_intervals = [
        (0.0, np.pi),           # Contains max at π/2
        (np.pi/2, 3*np.pi/2),   # Contains min at 3π/2
        (0.0, 2*np.pi),         # Full period
        (-np.pi, np.pi),        # Symmetric period
    ]

    results: List[ValidationResult] = []

    for activation in activations:
        print(f"\n{activation.upper()}")
        print("-" * 50)

        intervals = test_intervals.copy()
        if activation in ['sin', 'cos']:
            intervals.extend(trig_intervals)

        for lower, upper in intervals:
            result = validate_activation(
                activation, lower, upper,
                num_samples=args.samples,
                verbose=args.verbose
            )
            results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r.soundness_ok)
    failed = total - passed

    if failed > 0:
        print(f"\n{failed} FAILURES:")
        for r in results:
            if not r.soundness_ok:
                print(f"  {r.activation} [{r.interval[0]:.2f}, {r.interval[1]:.2f}]: "
                      f"analytic=[{r.reference_lb:.6f}, {r.reference_ub:.6f}], "
                      f"pytorch=[{r.pytorch_min:.6f}, {r.pytorch_max:.6f}]")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\nAll analytic reference intervals contain the sampled outputs.")
        print("This diagnostic does not exercise or validate ny.")
        return 0
    print(f"\n{failed} soundness violations detected!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
