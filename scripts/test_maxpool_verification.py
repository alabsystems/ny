#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Test MaxPool2d verification using ny-propagate.

This script tests the MaxPool2d implementation by verifying a simple
Conv2d + ReLU + MaxPool2d network.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Generate a simple CNN with MaxPool for testing
def create_test_cnn_onnx(output_dir: Path):
    """Create a test CNN ONNX model: Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear."""
    try:
        import onnx
        import torch
        import torch.nn as nn

        class TestCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                # After 8x8 input + conv(pad=1) -> 8x8, maxpool(2,2) -> 4x4
                # 4 channels * 4 * 4 = 64
                self.fc = nn.Linear(64, 2)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = TestCNN()

        # Use torch.jit for export with explicit settings
        dummy_input = torch.randn(1, 1, 8, 8)

        # Save as TorchScript first
        ts_path = output_dir / "test_cnn_maxpool.pt"
        output_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy_input)
            torch.jit.save(traced, ts_path)
        print(f"Saved TorchScript model to {ts_path}")

        # Manual ONNX creation using onnx helper
        from onnx import TensorProto, helper, numpy_helper

        # Get weights
        with torch.no_grad():
            conv_weight = model.conv.weight.numpy()
            conv_bias = model.conv.bias.numpy()
            fc_weight = model.fc.weight.numpy()
            fc_bias = model.fc.bias.numpy()

        # Create initializers
        conv_w_init = numpy_helper.from_array(conv_weight, "conv_weight")
        conv_b_init = numpy_helper.from_array(conv_bias, "conv_bias")
        fc_w_init = numpy_helper.from_array(fc_weight, "fc_weight")
        fc_b_init = numpy_helper.from_array(fc_bias, "fc_bias")

        # Create nodes
        conv_node = helper.make_node(
            "Conv",
            ["input", "conv_weight", "conv_bias"],
            ["conv_out"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1]
        )

        relu_node = helper.make_node(
            "Relu",
            ["conv_out"],
            ["relu_out"]
        )

        maxpool_node = helper.make_node(
            "MaxPool",
            ["relu_out"],
            ["pool_out"],
            kernel_shape=[2, 2],
            strides=[2, 2]
        )

        flatten_node = helper.make_node(
            "Flatten",
            ["pool_out"],
            ["flat_out"],
            axis=1
        )

        gemm_node = helper.make_node(
            "Gemm",
            ["flat_out", "fc_weight", "fc_bias"],
            ["output"],
            transB=1
        )

        # Create graph
        graph = helper.make_graph(
            [conv_node, relu_node, maxpool_node, flatten_node, gemm_node],
            "test_cnn_maxpool",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
            [conv_w_init, conv_b_init, fc_w_init, fc_b_init]
        )

        # Create model
        model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

        onnx_path = output_dir / "test_cnn_maxpool.onnx"
        onnx.save(model_proto, onnx_path)
        onnx.checker.check_model(model_proto)
        print(f"Saved ONNX model to {onnx_path}")

        return onnx_path, ts_path

    except ImportError as e:
        print(f"Warning: Could not create model: {e}")
        return None, None


def test_maxpool_verification(onnx_path: Path, output_dir: Path) -> bool:
    """Test verification of CNN with MaxPool."""

    # Create a simple VNN-LIB property
    vnnlib_path = output_dir / "test_cnn_maxpool.vnnlib"
    with open(vnnlib_path, 'w') as f:
        # Input variables for 1x8x8 = 64 inputs
        for i in range(64):
            f.write(f"(declare-const X_{i} Real)\n")

        # Output variables
        f.write("(declare-const Y_0 Real)\n")
        f.write("(declare-const Y_1 Real)\n")

        # Input constraints: [0.4, 0.6] for all inputs
        for i in range(64):
            f.write(f"(assert (>= X_{i} 0.4))\n")
            f.write(f"(assert (<= X_{i} 0.6))\n")

        # Output: class 0 should be less than class 1 (adversarial property)
        f.write("(assert (< Y_0 Y_1))\n")

    print(f"Created VNN-LIB property: {vnnlib_path}")

    # Run ny verify
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "ny-cli", "--",
         "verify", onnx_path,
         "--property", vnnlib_path,
         "--method", "ibp",
         "--json",
         "--allow-unknown"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    print("\n=== ny verify output ===")
    print(result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)

    if result.returncode != 0:
        print(f"FAIL: ny verify exited with code {result.returncode}")
        return False

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"FAIL: Could not parse JSON output: {exc}")
        return False

    status = output.get("status")
    bounds = output.get("output_bounds")
    print(f"\nVerification status: {status or 'missing'}")
    print(f"Method: {output.get('method', 'missing')}")
    if status not in {"verified", "violated", "unknown"}:
        print("FAIL: JSON output did not contain a recognized status")
        return False
    if output.get("method") != "ibp":
        print("FAIL: JSON output did not report the requested IBP method")
        return False
    if not isinstance(bounds, list) or not bounds:
        print("FAIL: JSON output did not contain output bounds")
        return False

    print(f"Output bounds: {bounds}")
    return True


def test_maxpool_ibp():
    """Direct test of MaxPool2d IBP propagation using Rust CLI."""

    # Build and run a simple test through the CLI
    result = subprocess.run(
        ["cargo", "test", "-p", "ny-propagate", "--",
         "test_maxpool2d", "--nocapture"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    print("=== MaxPool2d Unit Tests ===")
    print(result.stdout)
    if result.returncode != 0:
        print("stderr:", result.stderr)
        return False

    passed_counts = [
        int(count)
        for count in re.findall(r"test result: ok\. (\d+) passed", result.stdout)
    ]
    if not passed_counts or max(passed_counts) == 0:
        print("FAIL: cargo succeeded without running a matching MaxPool2d test")
        return False
    return True


def main() -> int:
    print("Testing MaxPool2d verification support\n")
    results: list[tuple[str, str]] = []

    # Test 1: Run unit tests
    print("1. Running MaxPool2d unit tests...")
    if test_maxpool_ibp():
        print("   PASS: All MaxPool2d unit tests passed\n")
        results.append(("MaxPool2d Rust unit tests", "PASS"))
    else:
        print("   FAIL: Unit tests failed\n")
        results.append(("MaxPool2d Rust unit tests", "FAIL"))

    # Test 2: Create and verify CNN with MaxPool
    print("2. Creating test CNN with MaxPool2d...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            onnx_path, _ = create_test_cnn_onnx(output_dir)

            if onnx_path:
                print("\n3. Testing verification with MaxPool2d...")
                passed = test_maxpool_verification(onnx_path, output_dir)
                results.append(("MaxPool2d ONNX verification", "PASS" if passed else "FAIL"))
            else:
                print("   SKIP: PyTorch/ONNX not available")
                results.append(("MaxPool2d ONNX verification", "SKIP"))
    except Exception as exc:
        print(f"   FAIL: Model writer or verification raised an exception: {exc}")
        results.append(("MaxPool2d ONNX verification", "FAIL"))

    passed = sum(status == "PASS" for _, status in results)
    failed = sum(status == "FAIL" for _, status in results)
    skipped = sum(status == "SKIP" for _, status in results)
    print("\nSummary:")
    for name, status in results:
        print(f"  {name}: {status}")
    print(f"  Total: {passed} passed, {failed} failed, {skipped} skipped")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
