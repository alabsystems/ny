#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Bug injection test for ny diff.

This script tests that ny diff correctly identifies differences between models:
1. Creates a "correct" and "broken" version of a model with different weights
2. Runs ny diff to compare them
3. Verifies that ny correctly detects the divergence

This validates P0 Correctness by proving ny diff finds real bugs.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

REPO_ROOT = Path(__file__).parent.parent
TEST_MODELS_DIR = REPO_ROOT / "tests" / "models"


def create_broken_model(original_path: str, output_path: str, weight_delta: float = 0.1) -> str:
    """Create a broken version of an ONNX model by perturbing weights.

    Args:
        original_path: Path to original ONNX model
        output_path: Path for the broken model
        weight_delta: Amount to add to first weight tensor

    Returns:
        Name of the modified initializer
    """
    model = onnx.load(original_path)
    modified_name = None

    # Find first weight tensor and modify it
    for initializer in model.graph.initializer:
        # Skip small tensors (likely bias) and non-float tensors
        if initializer.data_type == onnx.TensorProto.FLOAT:
            weight = numpy_helper.to_array(initializer).copy()  # Make writable copy
            if weight.size > 1:  # Skip scalar constants
                # Add delta to first element
                weight.flat[0] += weight_delta

                # Update initializer
                new_init = numpy_helper.from_array(weight, name=initializer.name)
                initializer.CopyFrom(new_init)
                modified_name = initializer.name
                break

    if modified_name is None:
        raise ValueError(f"No modifiable weight found in {original_path}")

    onnx.save(model, output_path)
    return modified_name


def create_test_input(shape: tuple, seed: int = 42) -> str:
    """Create a test input file with non-zero values.

    Returns path to temporary .npy file.
    """
    np.random.seed(seed)
    # Use small non-zero values to exercise weights
    data = np.random.randn(*shape).astype(np.float32)

    # Save to temp file
    fd, path = tempfile.mkstemp(suffix='.npy')
    os.close(fd)
    np.save(path, data)
    return path


def run_ny_diff(model_a: str, model_b: str, input_path: str = None, tolerance: float = 1e-5) -> dict:
    """Run ny diff and parse output.

    Args:
        model_a: Path to first model
        model_b: Path to second model
        input_path: Path to test input (.npy). Required for meaningful comparison.
        tolerance: Maximum allowed difference
    """
    cmd = [
        "cargo", "run", "--release", "--bin", "ny", "--",
        "diff",
        model_a,
        model_b,
        "--tolerance", str(tolerance),
    ]

    if input_path:
        cmd.extend(["--input", input_path])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "RUST_LOG": "error"}
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def test_identical_models(model_path: str, input_shape: tuple, verbose: bool = False) -> bool:
    """Test that ny diff reports identical models as equivalent."""
    if verbose:
        print(f"\nTesting identical models: {Path(model_path).name}")

    # Create test input with non-zero values
    input_path = create_test_input(input_shape)

    try:
        result = run_ny_diff(model_path, model_path, input_path=input_path)

        # Should pass (identical models)
        passed = "EQUIVALENT" in result["stdout"] or result["returncode"] == 0

        if verbose:
            if passed:
                print("  PASS: Identical models reported as equivalent")
            else:
                print("  FAIL: Identical models not reported as equivalent")
                print(f"  stdout: {result['stdout'][:200]}")

        return passed
    finally:
        os.unlink(input_path)


def test_broken_model_detection(model_path: str, input_shape: tuple, verbose: bool = False) -> bool:
    """Test that ny diff detects a broken model."""
    if verbose:
        print(f"\nTesting broken model detection: {Path(model_path).name}")

    # Create temporary broken model
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        broken_path = f.name

    # Create test input with non-zero values
    input_path = create_test_input(input_shape)

    try:
        modified_weight = create_broken_model(model_path, broken_path, weight_delta=0.5)

        if verbose:
            print(f"  Modified weight: {modified_weight}")

        result = run_ny_diff(model_path, broken_path, input_path=input_path, tolerance=1e-5)

        # Should fail (divergent models)
        found_divergence = (
            "DIVERGENT" in result["stdout"] or
            result["returncode"] != 0
        )

        if verbose:
            if found_divergence:
                print("  PASS: Divergence detected in broken model")
                # Extract first line mentioning divergence
                for line in result["stdout"].split('\n'):
                    if "DIVERGENT" in line or "Root Cause" in line:
                        print(f"  {line.strip()}")
            else:
                print("  FAIL: Divergence NOT detected")
                print(f"  stdout: {result['stdout'][:300]}")

        return found_divergence

    finally:
        os.unlink(broken_path)
        os.unlink(input_path)


def test_small_perturbation_detection(model_path: str, input_shape: tuple, verbose: bool = False) -> bool:
    """Test that ny diff detects small perturbations (sensitivity test)."""
    if verbose:
        print(f"\nTesting small perturbation detection: {Path(model_path).name}")

    # Create temporary broken model with small perturbation
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        broken_path = f.name

    # Create test input with non-zero values
    input_path = create_test_input(input_shape)

    try:
        # Very small perturbation
        modified_weight = create_broken_model(model_path, broken_path, weight_delta=0.001)

        if verbose:
            print(f"  Modified weight: {modified_weight} (delta=0.001)")

        # Use tight tolerance to catch small differences
        result = run_ny_diff(model_path, broken_path, input_path=input_path, tolerance=1e-6)

        found_divergence = (
            "DIVERGENT" in result["stdout"] or
            result["returncode"] != 0
        )

        if verbose:
            if found_divergence:
                print("  PASS: Small perturbation detected")
            else:
                print("  FAIL: Small perturbation NOT detected")

        return found_divergence

    finally:
        os.unlink(broken_path)
        os.unlink(input_path)


def main():
    parser = argparse.ArgumentParser(description="Bug injection test for ny diff")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("Bug Injection Test for ny diff")
    print("=" * 60)

    # Test models with weights: (path, input_shape)
    test_models = [
        (str(TEST_MODELS_DIR / "single_linear.onnx"), (1, 2)),
        (str(TEST_MODELS_DIR / "simple_mlp.onnx"), (1, 2)),
    ]

    results = []

    for model, input_shape in test_models:
        model_name = Path(model).name

        # Test 1: Identical models
        passed_identical = test_identical_models(model, input_shape, args.verbose)
        results.append(("identical", model_name, passed_identical))

        # Test 2: Broken model detection
        passed_broken = test_broken_model_detection(model, input_shape, args.verbose)
        results.append(("broken_detection", model_name, passed_broken))

        # Test 3: Small perturbation detection
        passed_small = test_small_perturbation_detection(model, input_shape, args.verbose)
        results.append(("small_perturbation", model_name, passed_small))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Test':<25} {'Model':<25} {'Status':<10}")
    print("-" * 60)

    for test_name, model_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:<25} {model_name:<25} {status:<10}")

    passed_count = sum(1 for _, _, p in results if p)
    total_count = len(results)

    print("-" * 60)
    print(f"\nTotal: {passed_count}/{total_count} passed")

    if passed_count == total_count:
        print("\nAll bug injection tests PASSED!")
        print("  - ny diff correctly identifies identical models")
        print("  - ny diff correctly detects divergent models")
        return 0
    print(f"\n{total_count - passed_count} tests FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
