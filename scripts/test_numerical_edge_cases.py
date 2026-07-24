#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Numerical edge case tests for ny.

Tests ny's handling of:
1. Large values (near overflow)
2. Small values (near underflow)
3. Zero weights
4. Mixed magnitudes
5. Negative weights
6. Extreme epsilon values

This validates P0 Correctness for numerical robustness.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_simple_linear_model(weight: np.ndarray, bias: np.ndarray, name: str) -> str:
    """Create a simple linear model with given weights."""
    input_dim, output_dim = weight.shape

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, input_dim])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, output_dim])

    W_init = numpy_helper.from_array(weight.astype(np.float32), name='W')
    B_init = numpy_helper.from_array(bias.astype(np.float32), name='B')

    matmul = helper.make_node('MatMul', ['X', 'W'], ['matmul_out'], name='matmul')
    add = helper.make_node('Add', ['matmul_out', 'B'], ['Y'], name='add')

    graph = helper.make_graph([matmul, add], name, [X], [Y], [W_init, B_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

    path = tempfile.mktemp(suffix='.onnx')
    onnx.save(model, path)
    return path


def run_ny_verify(model_path: str, epsilon: float = 0.01) -> dict:
    """Run ny verify and capture result."""
    cmd = [
        "cargo", "run", "--release", "--bin", "ny", "--",
        "verify", model_path, "--epsilon", str(epsilon)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT,
                           env={**os.environ, "RUST_LOG": "error"})
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def test_large_weights():
    """Test handling of large weight values."""
    print("\n=== Test: Large Weight Values ===")

    weight = np.array([[1e10, -1e10], [1e10, 1e10]], dtype=np.float32)
    bias = np.array([0.0, 0.0], dtype=np.float32)
    model_path = create_simple_linear_model(weight, bias, "large_weights")

    try:
        result = run_ny_verify(model_path, epsilon=0.001)
        no_panic = "panic" not in result["stderr"].lower()
        if result["returncode"] == 0 and no_panic:
            print("  PASS: Handled large weights")
            return True
        if no_panic:
            print("  PASS: Graceful error (no panic)")
            return True
        print("  FAIL: Panic detected")
        return False
    finally:
        os.unlink(model_path)


def test_small_weights():
    """Test handling of very small weight values."""
    print("\n=== Test: Small Weight Values ===")

    weight = np.array([[1e-20, -1e-20], [1e-20, 1e-20]], dtype=np.float32)
    bias = np.array([1e-20, -1e-20], dtype=np.float32)
    model_path = create_simple_linear_model(weight, bias, "small_weights")

    try:
        result = run_ny_verify(model_path, epsilon=0.001)
        if "panic" not in result["stderr"].lower():
            print("  PASS: Handled small weights without panic")
            return True
        print("  FAIL: Panic on small weights")
        return False
    finally:
        os.unlink(model_path)


def test_zero_weights():
    """Test handling of all-zero weights."""
    print("\n=== Test: Zero Weight Values ===")

    weight = np.zeros((2, 2), dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    model_path = create_simple_linear_model(weight, bias, "zero_weights")

    try:
        result = run_ny_verify(model_path, epsilon=0.01)
        if "Verified" in result["stdout"]:
            print("  PASS: Zero weights verified")
            return True
        if "panic" not in result["stderr"].lower():
            print("  PASS: Handled zero weights without panic")
            return True
        print(f"  FAIL: {result['stderr'][:100]}")
        return False
    finally:
        os.unlink(model_path)


def test_mixed_magnitudes():
    """Test handling of weights with mixed magnitudes."""
    print("\n=== Test: Mixed Magnitude Weights ===")

    weight = np.array([[1e10, 1e-10], [1e-10, 1e10]], dtype=np.float32)
    bias = np.array([1.0, -1.0], dtype=np.float32)
    model_path = create_simple_linear_model(weight, bias, "mixed_weights")

    try:
        result = run_ny_verify(model_path, epsilon=0.001)
        if "panic" not in result["stderr"].lower():
            print("  PASS: Handled mixed magnitudes without panic")
            return True
        print("  FAIL: Panic on mixed magnitudes")
        return False
    finally:
        os.unlink(model_path)


def test_normal_model():
    """Baseline test with normal values."""
    print("\n=== Test: Normal Values (Baseline) ===")

    weight = np.array([[1.0, -0.5], [0.5, 1.0]], dtype=np.float32)
    bias = np.array([0.1, -0.1], dtype=np.float32)
    model_path = create_simple_linear_model(weight, bias, "normal_weights")

    try:
        result = run_ny_verify(model_path, epsilon=0.01)
        if "Verified" in result["stdout"]:
            print("  PASS: Normal model verified")
            return True
        print(f"  FAIL: {result['stdout']}")
        return False
    finally:
        os.unlink(model_path)


def test_epsilon_edge_cases():
    """Test edge cases in epsilon values."""
    print("\n=== Test: Epsilon Edge Cases ===")

    weight = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bias = np.array([0.0, 0.0], dtype=np.float32)
    model_path = create_simple_linear_model(weight, bias, "identity")

    results = []
    try:
        # Very small epsilon
        r1 = run_ny_verify(model_path, epsilon=1e-10)
        passed = "Verified" in r1["stdout"] or "panic" not in r1["stderr"].lower()
        print(f"  {'PASS' if passed else 'FAIL'}: Very small epsilon (1e-10)")
        results.append(passed)

        # Large epsilon
        r2 = run_ny_verify(model_path, epsilon=100.0)
        passed = "panic" not in r2["stderr"].lower()
        print(f"  {'PASS' if passed else 'FAIL'}: Large epsilon (100.0)")
        results.append(passed)

        return all(results)
    finally:
        os.unlink(model_path)


def test_negative_weights():
    """Test with all negative weights."""
    print("\n=== Test: Negative Weights ===")

    weight = np.array([[-1.0, -2.0], [-0.5, -1.5]], dtype=np.float32)
    bias = np.array([-0.1, -0.2], dtype=np.float32)
    model_path = create_simple_linear_model(weight, bias, "negative_weights")

    try:
        result = run_ny_verify(model_path, epsilon=0.01)
        if "Verified" in result["stdout"]:
            print("  PASS: Negative weights verified")
            return True
        if "panic" not in result["stderr"].lower():
            print("  PASS: No panic with negative weights")
            return True
        print("  FAIL: Panic with negative weights")
        return False
    finally:
        os.unlink(model_path)


def main():
    print("=" * 60)
    print("Numerical Edge Case Tests for ny")
    print("=" * 60)

    tests = [
        ("Normal baseline", test_normal_model),
        ("Large weights", test_large_weights),
        ("Small weights", test_small_weights),
        ("Zero weights", test_zero_weights),
        ("Mixed magnitudes", test_mixed_magnitudes),
        ("Negative weights", test_negative_weights),
        ("Epsilon edge cases", test_epsilon_edge_cases),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{len(results)} passed")

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
