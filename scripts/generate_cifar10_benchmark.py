#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Generate CIFAR-10 benchmark models and properties for VNN-COMP style verification.

CIFAR-10 has 32×32×3 = 3072 input dimensions, which stress-tests verification
on larger input spaces compared to MNIST (784 inputs).

Creates:
- Small MLP model (3072 -> 100 -> 100 -> 10) in ONNX format
- VNN-LIB property files for adversarial robustness verification

Usage:
    python scripts/generate_cifar10_benchmark.py
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Output directories
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def create_cifar10_mlp():
    """Create a small CIFAR-10 MLP: 3072 -> 100 -> 100 -> 10 with ReLU.

    This is proportionally larger than MNIST to handle the 4x larger input space.

    Network architecture:
    - Input: 3072 (32x32x3 flattened CIFAR-10 image)
    - Hidden 1: 100 neurons + ReLU (scaled from MNIST's 50)
    - Hidden 2: 100 neurons + ReLU
    - Output: 10 (CIFAR-10 classes)

    Total ReLUs: 200
    Total parameters: 3072*100 + 100 + 100*100 + 100 + 100*10 + 10 = 318,410
    """
    # Use fixed seed for reproducibility
    np.random.seed(43)  # Different seed from MNIST

    # Initialize weights with Xavier-like scaling
    w1 = np.random.randn(100, 3072).astype(np.float32) * np.sqrt(2.0 / 3072)
    b1 = np.zeros(100, dtype=np.float32)

    w2 = np.random.randn(100, 100).astype(np.float32) * np.sqrt(2.0 / 100)
    b2 = np.zeros(100, dtype=np.float32)

    w3 = np.random.randn(10, 100).astype(np.float32) * np.sqrt(2.0 / 100)
    b3 = np.zeros(10, dtype=np.float32)

    # Create initializers
    w1_init = numpy_helper.from_array(w1, name="w1")
    b1_init = numpy_helper.from_array(b1, name="b1")
    w2_init = numpy_helper.from_array(w2, name="w2")
    b2_init = numpy_helper.from_array(b2, name="b2")
    w3_init = numpy_helper.from_array(w3, name="w3")
    b3_init = numpy_helper.from_array(b3, name="b3")

    # Layer 1: Linear + ReLU
    gemm1 = helper.make_node(
        "Gemm", inputs=["input", "w1", "b1"], outputs=["fc1"],
        alpha=1.0, beta=1.0, transA=0, transB=1
    )
    relu1 = helper.make_node("Relu", inputs=["fc1"], outputs=["relu1"])

    # Layer 2: Linear + ReLU
    gemm2 = helper.make_node(
        "Gemm", inputs=["relu1", "w2", "b2"], outputs=["fc2"],
        alpha=1.0, beta=1.0, transA=0, transB=1
    )
    relu2 = helper.make_node("Relu", inputs=["fc2"], outputs=["relu2"])

    # Layer 3: Linear (no activation - logits)
    gemm3 = helper.make_node(
        "Gemm", inputs=["relu2", "w3", "b3"], outputs=["output"],
        alpha=1.0, beta=1.0, transA=0, transB=1
    )

    # Create graph
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3072])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    graph = helper.make_graph(
        [gemm1, relu1, gemm2, relu2, gemm3],
        "cifar10_mlp_2x100",
        [input_tensor],
        [output_tensor],
        [w1_init, b1_init, w2_init, b2_init, w3_init, b3_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    output_path = os.path.join(MODELS_DIR, "cifar10_mlp_2x100.onnx")
    onnx.save(model, output_path)
    print(f"Created CIFAR-10 MLP: {output_path}")
    print("  Architecture: 3072 -> 100 -> 100 -> 10")
    print("  ReLU neurons: 200")
    print("  Parameters: ~318k")

    return model


def create_cifar10_vnnlib_robustness(epsilon=0.02, true_label=0):
    """Create VNN-LIB property for L-infinity adversarial robustness.

    Property: Given input x with true label, verify that no perturbation
    within epsilon ball changes the classification.

    For VNN-LIB robustness properties:
    - Input bounds: [pixel - eps, pixel + eps] for each input
    - Output constraint: y_true > y_other for all other classes
      (Unsafe region: any y_other >= y_true)

    Args:
        epsilon: L-infinity perturbation bound (default 0.02 = 2% of [0,1] range)
        true_label: The true class label (0-9)
    """
    # Create a sample "image" - center of input space (0.5 for normalized [0,1])
    center_value = 0.5

    lines = []
    lines.append("; VNN-LIB robustness property for CIFAR-10")
    lines.append("; Input dimension: 3072 (32x32x3)")
    lines.append(f"; True label: {true_label}")
    lines.append(f"; Epsilon: {epsilon}")
    lines.append("")

    # Declare input variables X_0 to X_3071
    for i in range(3072):
        lines.append(f"(declare-const X_{i} Real)")
    lines.append("")

    # Declare output variables Y_0 to Y_9
    for i in range(10):
        lines.append(f"(declare-const Y_{i} Real)")
    lines.append("")

    # Input constraints: each pixel in [center - eps, center + eps], clipped to [0, 1]
    lines.append("; Input bounds (L-infinity ball around center)")
    for i in range(3072):
        lower = max(0.0, center_value - epsilon)
        upper = min(1.0, center_value + epsilon)
        lines.append(f"(assert (>= X_{i} {lower:.6f}))")
        lines.append(f"(assert (<= X_{i} {upper:.6f}))")
    lines.append("")

    # Output constraint: unsafe if any other class score >= true class score
    lines.append("; Unsafe region: some other class has score >= true class")
    lines.append("(assert (or")
    for i in range(10):
        if i != true_label:
            lines.append(f"  (>= Y_{i} Y_{true_label})")
    lines.append("))")
    lines.append("")

    output_path = os.path.join(MODELS_DIR, f"cifar10_robustness_eps{epsilon:.3f}_label{true_label}.vnnlib")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Created VNN-LIB property: {output_path}")
    print(f"  True label: {true_label}")
    print(f"  Epsilon: {epsilon}")
    print("  Property: verify classification is robust to L-inf perturbations")

    return output_path


def main():
    print("=" * 60)
    print("Generating CIFAR-10 VNN-COMP benchmark models and properties")
    print("=" * 60)
    print()

    # Create CIFAR-10 MLP model
    create_cifar10_mlp()
    print()

    # Create robustness properties with different epsilons
    for eps in [0.01, 0.02, 0.05]:
        create_cifar10_vnnlib_robustness(epsilon=eps, true_label=0)
    print()

    print("=" * 60)
    print("CIFAR-10 benchmarks created successfully!")
    print()
    print("Usage examples:")
    print("  # IBP verification")
    print("  ny verify tests/models/cifar10_mlp_2x100.onnx \\")
    print("    --property tests/models/cifar10_robustness_eps0.020_label0.vnnlib --method ibp")
    print()
    print("  # CROWN verification")
    print("  ny verify tests/models/cifar10_mlp_2x100.onnx \\")
    print("    --property tests/models/cifar10_robustness_eps0.020_label0.vnnlib --method crown")
    print()
    print("  # alpha-CROWN verification")
    print("  ny verify tests/models/cifar10_mlp_2x100.onnx \\")
    print("    --property tests/models/cifar10_robustness_eps0.020_label0.vnnlib --method alpha")
    print("=" * 60)


if __name__ == "__main__":
    main()
