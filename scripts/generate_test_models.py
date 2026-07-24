#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Generate simple ONNX models for testing ny ONNX loading.

Uses pure ONNX (no PyTorch) to create minimal test models.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_single_linear():
    """Create a single Linear layer: y = Wx + b (2 -> 3)"""
    # Weight: 3x2 matrix
    weight_data = np.array([
        [1.0, 2.0],
        [3.0, -1.0],
        [-2.0, 1.0]
    ], dtype=np.float32)

    # Bias: 3-element vector
    bias_data = np.array([0.5, -0.5, 1.0], dtype=np.float32)

    # Create initializers (constants)
    weight_init = numpy_helper.from_array(weight_data, name="weight")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    # Create the MatMul node (input @ weight.T for standard linear layer convention)
    # But ONNX Gemm is: Y = alpha * A' * B' + beta * C
    # With transB=1: Y = A @ B.T + C
    # We want: y = x @ W.T + b where W is 3x2, so x(1,2) @ W.T(2,3) = y(1,3)
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["input", "weight", "bias"],
        outputs=["output"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1  # Transpose weight
    )

    # Create graph
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    graph = helper.make_graph(
        [gemm_node],
        "single_linear",
        [input_tensor],
        [output_tensor],
        [weight_init, bias_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    output_path = os.path.join(OUTPUT_DIR, "single_linear.onnx")
    onnx.save(model, output_path)
    print(f"Created: {output_path}")

    # Verification
    print("  weight (3x2):")
    print("    [1.0, 2.0]")
    print("    [3.0, -1.0]")
    print("    [-2.0, 1.0]")
    print("  bias (3): [0.5, -0.5, 1.0]")
    print("  Test: input=[1,1] -> output=[3.5, 1.5, 0.0]")

    return model


def create_linear_relu():
    """Create Linear + ReLU: y = ReLU(Wx + b)"""
    weight_data = np.array([
        [1.0, 2.0],
        [3.0, -1.0],
        [-2.0, 1.0]
    ], dtype=np.float32)
    bias_data = np.array([0.5, -0.5, 1.0], dtype=np.float32)

    weight_init = numpy_helper.from_array(weight_data, name="weight")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    gemm_node = helper.make_node(
        "Gemm",
        inputs=["input", "weight", "bias"],
        outputs=["linear_out"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1
    )

    relu_node = helper.make_node(
        "Relu",
        inputs=["linear_out"],
        outputs=["output"]
    )

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    graph = helper.make_graph(
        [gemm_node, relu_node],
        "linear_relu",
        [input_tensor],
        [output_tensor],
        [weight_init, bias_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    output_path = os.path.join(OUTPUT_DIR, "linear_relu.onnx")
    onnx.save(model, output_path)
    print(f"Created: {output_path}")

    return model


def create_simple_mlp():
    """Create MLP: Linear(2->4) -> ReLU -> Linear(4->2)"""
    # First layer: 4x2
    w1_data = np.array([
        [1.0, 0.5],
        [-1.0, 0.5],
        [0.5, 1.0],
        [0.5, -1.0]
    ], dtype=np.float32)
    b1_data = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)

    # Second layer: 2x4
    w2_data = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0]
    ], dtype=np.float32)
    b2_data = np.array([0.0, 0.0], dtype=np.float32)

    w1_init = numpy_helper.from_array(w1_data, name="w1")
    b1_init = numpy_helper.from_array(b1_data, name="b1")
    w2_init = numpy_helper.from_array(w2_data, name="w2")
    b2_init = numpy_helper.from_array(b2_data, name="b2")

    gemm1 = helper.make_node("Gemm", ["input", "w1", "b1"], ["fc1_out"],
                             alpha=1.0, beta=1.0, transA=0, transB=1)
    relu = helper.make_node("Relu", ["fc1_out"], ["relu_out"])
    gemm2 = helper.make_node("Gemm", ["relu_out", "w2", "b2"], ["output"],
                             alpha=1.0, beta=1.0, transA=0, transB=1)

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

    graph = helper.make_graph(
        [gemm1, relu, gemm2],
        "simple_mlp",
        [input_tensor],
        [output_tensor],
        [w1_init, b1_init, w2_init, b2_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    output_path = os.path.join(OUTPUT_DIR, "simple_mlp.onnx")
    onnx.save(model, output_path)
    print(f"Created: {output_path}")

    return model


def create_single_conv2d():
    """Create single Conv2d: (1, 1, 5, 5) -> (1, 1, 3, 3) with 3x3 kernel"""
    # Edge detection kernel
    kernel_data = np.array([[[
        [-1.0, -1.0, -1.0],
        [-1.0, 8.0, -1.0],
        [-1.0, -1.0, -1.0]
    ]]], dtype=np.float32)  # Shape: (1, 1, 3, 3)

    bias_data = np.array([0.0], dtype=np.float32)

    kernel_init = numpy_helper.from_array(kernel_data, name="kernel")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "kernel", "bias"],
        outputs=["output"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0]  # No padding
    )

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 5, 5])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3, 3])

    graph = helper.make_graph(
        [conv_node],
        "single_conv2d",
        [input_tensor],
        [output_tensor],
        [kernel_init, bias_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    output_path = os.path.join(OUTPUT_DIR, "single_conv2d.onnx")
    onnx.save(model, output_path)
    print(f"Created: {output_path}")

    return model


def create_conv_relu():
    """Create Conv2d + ReLU: (1, 1, 4, 4) -> Conv(2ch) -> ReLU -> (1, 2, 3, 3)"""
    # Two kernels
    kernel_data = np.array([
        [[[1.0, 1.0], [1.0, 1.0]]],   # Summing kernel
        [[[1.0, -1.0], [-1.0, 1.0]]]  # Difference kernel
    ], dtype=np.float32)  # Shape: (2, 1, 2, 2)

    bias_data = np.array([0.0, 0.0], dtype=np.float32)

    kernel_init = numpy_helper.from_array(kernel_data, name="kernel")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "kernel", "bias"],
        outputs=["conv_out"],
        kernel_shape=[2, 2],
        strides=[1, 1],
        pads=[0, 0, 0, 0]
    )

    relu_node = helper.make_node("Relu", ["conv_out"], ["output"])

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 3, 3])

    graph = helper.make_graph(
        [conv_node, relu_node],
        "conv_relu",
        [input_tensor],
        [output_tensor],
        [kernel_init, bias_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    output_path = os.path.join(OUTPUT_DIR, "conv_relu.onnx")
    onnx.save(model, output_path)
    print(f"Created: {output_path}")

    return model


def create_matmul_transpose_b_const():
    """Create MatMul with constant weight and transpose_b attribute."""
    # Input: (1, 3), Weight: (2, 3), transpose_b=1 => output: (1, 2)
    weight_data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)

    weight_init = numpy_helper.from_array(weight_data, name="weight")

    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "weight"],
        outputs=["output"],
        transpose_b=1,
    )

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

    graph = helper.make_graph(
        [matmul_node],
        "matmul_transpose_b_const",
        [input_tensor],
        [output_tensor],
        [weight_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    output_path = os.path.join(OUTPUT_DIR, "matmul_transpose_b_const.onnx")
    onnx.save(model, output_path)
    print(f"Created: {output_path}")

    return model


def create_mnist_conv():
    """Create small CNN classifier: Conv(1->2,3x3) -> ReLU -> Flatten -> Linear(72->5).

    Mimics MNIST-CONV architecture at reduced scale (1ch 8x8 input instead of 28x28).
    Used for end-to-end Patches mode verification gate (#2613).
    """
    # Conv kernel: (out_c=2, in_c=1, kH=3, kW=3)
    # Use small known weights for reproducibility
    np.random.seed(42)
    kernel_data = np.random.randn(2, 1, 3, 3).astype(np.float32) * 0.5
    conv_bias_data = np.zeros(2, dtype=np.float32)

    # After Conv(3x3, no pad, stride 1) on 8x8: output is 2x6x6 = 72
    # Gemm: 72 -> 5 (5-class classifier)
    fc_weight_data = np.random.randn(5, 72).astype(np.float32) * 0.1
    fc_bias_data = np.zeros(5, dtype=np.float32)

    kernel_init = numpy_helper.from_array(kernel_data, name="conv_kernel")
    conv_bias_init = numpy_helper.from_array(conv_bias_data, name="conv_bias")
    fc_weight_init = numpy_helper.from_array(fc_weight_data, name="fc_weight")
    fc_bias_init = numpy_helper.from_array(fc_bias_data, name="fc_bias")

    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_kernel", "conv_bias"],
        outputs=["conv_out"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    relu_node = helper.make_node("Relu", ["conv_out"], ["relu_out"])
    flatten_node = helper.make_node(
        "Flatten", ["relu_out"], ["flat_out"], axis=1
    )
    gemm_node = helper.make_node(
        "Gemm",
        ["flat_out", "fc_weight", "fc_bias"],
        ["output"],
        alpha=1.0, beta=1.0, transA=0, transB=1,
    )

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])

    graph = helper.make_graph(
        [conv_node, relu_node, flatten_node, gemm_node],
        "mnist_conv",
        [input_tensor],
        [output_tensor],
        [kernel_init, conv_bias_init, fc_weight_init, fc_bias_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    output_path = os.path.join(OUTPUT_DIR, "mnist_conv.onnx")
    onnx.save(model, output_path)
    print(f"Created: {output_path}")
    print("  Architecture: Conv(1->2, 3x3) -> ReLU -> Flatten -> Linear(72->5)")
    print("  Input: (1, 1, 8, 8), Output: (1, 5)")

    return model


def create_minimal_attention_core():
    """Create minimal attention core graph: QK^T -> Softmax -> V."""
    wq_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    wk_data = np.array([[1.0, 0.5], [-0.5, 1.0]], dtype=np.float32)
    wv_data = np.array([[0.2, 0.1], [0.3, -0.4]], dtype=np.float32)

    wq_init = numpy_helper.from_array(wq_data, name="wq")
    wk_init = numpy_helper.from_array(wk_data, name="wk")
    wv_init = numpy_helper.from_array(wv_data, name="wv")

    q_matmul = helper.make_node("MatMul", ["input", "wq"], ["q"])
    k_matmul = helper.make_node("MatMul", ["input", "wk"], ["k"])
    v_matmul = helper.make_node("MatMul", ["input", "wv"], ["v"])
    scores = helper.make_node("MatMul", ["q", "k"], ["scores"], transpose_b=1)
    softmax = helper.make_node("Softmax", ["scores"], ["probs"])
    context = helper.make_node("MatMul", ["probs", "v"], ["output"])

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

    graph = helper.make_graph(
        [q_matmul, k_matmul, v_matmul, scores, softmax, context],
        "minimal_attention_core",
        [input_tensor],
        [output_tensor],
        [wq_init, wk_init, wv_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    output_path = os.path.join(OUTPUT_DIR, "minimal_attention_core.onnx")
    onnx.save(model, output_path)
    print(f"Created: {output_path}")

    return model


def create_duration_predictor_surrogate():
    """Feed-forward surrogate for Kokoro's duration predictor (LSTM unsupported).

    Same contract: encoded_features [1, T, 8] -> duration_logits [1, T, 50].
    Source: designs/2026-03-11-avoice-phase1-onnx-execution.md (section 4)
    """
    hidden_dim, duration_buckets, seq_len = 8, 50, 4
    rng = np.random.RandomState(42)
    weight_data = (rng.randn(hidden_dim, duration_buckets) * 0.1).astype(np.float32)

    weight_init = numpy_helper.from_array(weight_data, name="weight")
    bias_init = numpy_helper.from_array(np.zeros(duration_buckets, dtype=np.float32), name="bias")

    matmul_node = helper.make_node("MatMul", ["encoded_features", "weight"], ["matmul_out"])
    add_node = helper.make_node("Add", ["matmul_out", "bias"], ["duration_logits"])

    input_tensor = helper.make_tensor_value_info(
        "encoded_features", TensorProto.FLOAT, [1, seq_len, hidden_dim])
    output_tensor = helper.make_tensor_value_info(
        "duration_logits", TensorProto.FLOAT, [1, seq_len, duration_buckets])

    graph = helper.make_graph(
        [matmul_node, add_node], "kokoro_duration_predictor_surrogate",
        [input_tensor], [output_tensor], [weight_init, bias_init])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    output_path = os.path.join(OUTPUT_DIR, "kokoro_duration_predictor_surrogate.onnx")
    onnx.save(model, output_path)
    print(f"Created: {output_path} (surrogate duration predictor)")
    return model


def main():
    print("Generating test ONNX models...\n")

    create_single_linear()
    create_linear_relu()
    create_simple_mlp()
    create_single_conv2d()
    create_conv_relu()
    create_matmul_transpose_b_const()
    create_mnist_conv()
    create_minimal_attention_core()
    create_duration_predictor_surrogate()

    print(f"\nAll models created in {OUTPUT_DIR}")

    # List created files
    print("\nGenerated files:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".onnx"):
            path = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(path)
            print(f"  {f}: {size} bytes")


if __name__ == "__main__":
    main()
