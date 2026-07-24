#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Simple test of MaxPool2d verification without Flatten complications."""

import subprocess
from pathlib import Path

import numpy as np


def create_simple_maxpool_onnx():
    """Create Conv -> ReLU -> MaxPool model (no flatten/linear)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Simple weights
    conv_weight = np.random.randn(2, 1, 3, 3).astype(np.float32) * 0.1
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

    onnx_path = "tests/models/conv_relu_maxpool.onnx"
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_proto, onnx_path)
    onnx.checker.check_model(model_proto)
    print(f"Created {onnx_path}")

    return onnx_path, conv_weight


def create_vnnlib(output_dim):
    """Create simple VNN-LIB property."""
    # 64 inputs (1*8*8), output_dim outputs
    vnnlib_path = "tests/models/conv_relu_maxpool.vnnlib"
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


def run_ny_verify(onnx_path, vnnlib_path, method="ibp"):
    """Run ny verify and parse output."""
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "ny-cli", "--",
         "verify", onnx_path, "--property", vnnlib_path, "--method", method],
        capture_output=True, text=True
    )

    print(f"\n=== ny verify ({method}) ===")
    print(result.stdout)
    if result.stderr and "Compiling" not in result.stderr:
        print("stderr:", result.stderr[:500])

    return result.returncode == 0


def main():
    print("Testing MaxPool2d verification (simple model)\n")

    # Create model
    onnx_path, conv_weight = create_simple_maxpool_onnx()

    # Create property (output is 2*4*4 = 32 elements)
    vnnlib_path = create_vnnlib(32)

    # Test IBP
    print("\nRunning IBP verification...")
    run_ny_verify(onnx_path, vnnlib_path, "ibp")

    # Test CROWN (falls back to IBP for MaxPool)
    print("\nRunning CROWN verification...")
    run_ny_verify(onnx_path, vnnlib_path, "crown")

    print("\nDone!")


if __name__ == "__main__":
    main()
