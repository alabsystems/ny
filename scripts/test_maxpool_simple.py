#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Simple test of MaxPool2d verification without Flatten complications."""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def create_simple_maxpool_onnx(output_dir: Path):
    """Create Conv -> ReLU -> MaxPool model (no flatten/linear)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Simple weights
    rng = np.random.default_rng(0)
    conv_weight = rng.standard_normal((2, 1, 3, 3)).astype(np.float32) * 0.1
    conv_bias = np.zeros(2).astype(np.float32)

    conv_w_init = numpy_helper.from_array(conv_weight, "conv_weight")
    conv_b_init = numpy_helper.from_array(conv_bias, "conv_bias")

    # Nodes
    conv_node = helper.make_node(
        "Conv", ["input", "conv_weight", "conv_bias"], ["conv_out"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    relu_node = helper.make_node("Relu", ["conv_out"], ["relu_out"])
    maxpool_node = helper.make_node(
        "MaxPool", ["relu_out"], ["output"],
        kernel_shape=[2, 2], strides=[2, 2]
    )

    # Graph: input [1,1,8,8] -> conv [1,2,8,8] -> relu -> maxpool [1,2,4,4]
    graph = helper.make_graph(
        [conv_node, relu_node, maxpool_node],
        "conv_relu_maxpool",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])],
        [conv_w_init, conv_b_init]
    )

    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

    onnx_path = output_dir / "conv_relu_maxpool.onnx"
    onnx.save(model_proto, onnx_path)
    onnx.checker.check_model(model_proto)
    print(f"Created {onnx_path}")

    return onnx_path, conv_weight


def create_vnnlib(output_dim, output_dir: Path):
    """Create simple VNN-LIB property."""
    # 64 inputs (1*8*8), output_dim outputs
    vnnlib_path = output_dir / "conv_relu_maxpool.vnnlib"
    with open(vnnlib_path, 'w') as f:
        for i in range(64):
            f.write(f"(declare-const X_{i} Real)\n")
        for i in range(output_dim):
            f.write(f"(declare-const Y_{i} Real)\n")

        # Input bounds: small perturbation around 0.5
        for i in range(64):
            f.write(f"(assert (>= X_{i} 0.45))\n")
            f.write(f"(assert (<= X_{i} 0.55))\n")

        # Output: any element > 0 is unsafe (just for testing)
        f.write("(assert (> Y_0 0))\n")

    print(f"Created {vnnlib_path}")
    return vnnlib_path


def run_ny_verify(onnx_path, vnnlib_path, method="ibp", expected_outputs=32):
    """Run ny verify and parse output."""
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "ny-cli", "--",
         "verify", onnx_path, "--property", vnnlib_path, "--method", method,
         "--json", "--require-sound"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )

    print(f"\n=== ny verify ({method}) ===")
    print(result.stdout)
    if result.stderr and "Compiling" not in result.stderr:
        print("stderr:", result.stderr[:500])

    json_start = result.stdout.find("{")
    try:
        if json_start < 0:
            raise json.JSONDecodeError("no JSON object", result.stdout, 0)
        output = json.loads(result.stdout[json_start:])
    except json.JSONDecodeError:
        print("FAIL: ny did not return JSON")
        return False

    status = output.get("property_status", output.get("status"))
    aliases = {"safe": "verified", "violated": "falsified"}
    status = str(status).lower()
    status = aliases.get(status, status)
    expected_codes = {
        "verified": 0,
        "falsified": 1,
        "unknown": 2,
        "timeout": 3,
    }
    bounds = output.get("output_bounds")
    soundness = output.get("soundness")
    actual_method = (
        str(output.get("actual_method", ""))
        .lower()
        .replace("-", "")
        .replace("_", "")
    )
    allowed_actual_methods = {"ibp": {"ibp"}, "crown": {"crown", "ibp"}}
    try:
        bounds_valid = (
            isinstance(bounds, list)
            and len(bounds) == expected_outputs
            and all(
                math.isfinite(float(bound["lower"]))
                and math.isfinite(float(bound["upper"]))
                and float(bound["lower"]) <= float(bound["upper"])
                for bound in bounds
            )
        )
    except (KeyError, TypeError, ValueError):
        bounds_valid = False
    passed = (
        status in expected_codes
        and result.returncode == expected_codes[status]
        and isinstance(soundness, dict)
        and soundness.get("mode") == "sound"
        and output.get("method") == method
        and actual_method in allowed_actual_methods.get(method, set())
        and bounds_valid
    )
    if not passed:
        print(
            f"FAIL: invalid ny result contract "
            f"(code={result.returncode}, status={status!r}, "
            f"bounds={len(bounds) if isinstance(bounds, list) else 'invalid'}/"
            f"{expected_outputs})"
        )
    return passed


def main() -> int:
    print("Testing MaxPool2d verification (simple model)\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create model
        onnx_path, _ = create_simple_maxpool_onnx(output_dir)

        # Create property (output is 2*4*4 = 32 elements)
        vnnlib_path = create_vnnlib(32, output_dir)

        # Test IBP
        print("\nRunning IBP verification...")
        ibp_passed = run_ny_verify(onnx_path, vnnlib_path, "ibp")

        # Test CROWN (falls back to IBP for MaxPool)
        print("\nRunning CROWN verification...")
        crown_passed = run_ny_verify(onnx_path, vnnlib_path, "crown")

    failed = int(not ibp_passed) + int(not crown_passed)
    print(f"\nSummary: {2 - failed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
