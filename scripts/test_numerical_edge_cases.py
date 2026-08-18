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

import json
import math
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
        "verify", model_path, "--epsilon", str(epsilon), "--json",
        "--strict", "--require-sound",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT,
                           env={**os.environ, "RUST_LOG": "error"})
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def verification_completed(result: dict) -> bool:
    """Require a valid status/exit contract and two finite output bounds."""
    stdout = result["stdout"].strip()
    json_start = stdout.find("{")
    if json_start < 0 or "panic" in result["stderr"].lower():
        return False
    try:
        data = json.loads(stdout[json_start:])
    except json.JSONDecodeError:
        return False

    aliases = {"safe": "verified", "violated": "falsified"}
    raw_status = str(data.get("property_status", data.get("status", ""))).lower()
    status = aliases.get(raw_status, raw_status)
    expected_codes = {
        "verified": 0,
        "falsified": 1,
        "unknown": 2,
        "timeout": 3,
    }
    bounds = data.get("output_bounds")
    soundness = data.get("soundness")
    try:
        bounds_valid = (
            isinstance(bounds, list)
            and len(bounds) == 2
            and all(
                math.isfinite(float(bound["lower"]))
                and math.isfinite(float(bound["upper"]))
                and float(bound["lower"]) <= float(bound["upper"])
                for bound in bounds
            )
        )
    except (KeyError, TypeError, ValueError):
        bounds_valid = False
    return (
        status == "verified"
        and result["returncode"] == expected_codes["verified"]
        and isinstance(soundness, dict)
        and soundness.get("mode") == "sound"
        and bounds_valid
    )


def test_large_weights():
    """Test handling of large weight values."""
    print("\n=== Test: Large Weight Values ===")

    weight = np.array([[1e10, -1e10], [1e10, 1e10]], dtype=np.float32)
    bias = np.array([0.0, 0.0], dtype=np.float32)
    model_path = create_simple_linear_model(weight, bias, "large_weights")

    try:
        result = run_ny_verify(model_path, epsilon=0.001)
        if verification_completed(result):
            print("  PASS: Handled large weights")
            return True
        print(
            f"  FAIL: ny did not complete verification "
            f"(code={result['returncode']}, stderr={result['stderr'][:100]!r})"
        )
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
        if verification_completed(result):
            print("  PASS: Handled small weights")
            return True
        print(f"  FAIL: ny did not complete verification (code={result['returncode']})")
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
        if verification_completed(result):
            print("  PASS: Zero weights verified")
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
        if verification_completed(result):
            print("  PASS: Handled mixed magnitudes")
            return True
        print(f"  FAIL: ny did not complete verification (code={result['returncode']})")
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
        if verification_completed(result):
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
        passed = verification_completed(r1)
        print(f"  {'PASS' if passed else 'FAIL'}: Very small epsilon (1e-10)")
        results.append(passed)

        # Large epsilon
        r2 = run_ny_verify(model_path, epsilon=100.0)
        passed = verification_completed(r2)
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
        if verification_completed(result):
            print("  PASS: Negative weights verified")
            return True
        print(f"  FAIL: ny did not complete verification (code={result['returncode']})")
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
