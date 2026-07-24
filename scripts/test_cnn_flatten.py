#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Test full CNN pipeline with Flatten layer.

This script tests the complete CNN verification pipeline:
Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear

This validates that ny can now verify complete CNN architectures.
"""

import json
import subprocess
from pathlib import Path


def create_cnn_with_flatten():
    """Create a CNN model with Flatten layer."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed. Cannot create test model.")
        return None

    class SimpleCNN(nn.Module):
        """Simple CNN: Conv -> ReLU -> MaxPool -> Flatten -> Linear."""

        def __init__(self):
            super().__init__()
            # Input: (1, 1, 8, 8) - batch=1, channels=1, 8x8 image
            # Conv: 1 -> 4 channels, kernel=3, pad=1 -> (1, 4, 8, 8)
            self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
            # MaxPool: kernel=2, stride=2 -> (1, 4, 4, 4)
            self.pool = nn.MaxPool2d(2, 2)
            # Flatten: (1, 4, 4, 4) -> (1, 64)
            self.flatten = nn.Flatten()
            # Linear: 64 -> 2
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

    return SimpleCNN()


def export_to_onnx(model, output_path: str):
    """Export model to ONNX format."""
    import warnings

    import torch

    dummy_input = torch.randn(1, 1, 8, 8)

    # Try legacy exporter first
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Attempt to use dynamo=False for legacy export
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=13,
                do_constant_folding=True,
                export_params=True,
                dynamo=False,  # Force legacy exporter
            )
        except TypeError:
            # Fallback if dynamo parameter not supported
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=13,
                do_constant_folding=True,
                export_params=True,
            )
    print(f"Exported ONNX model to {output_path}")


def create_vnnlib_property(model_path: str, output_path: str, epsilon: float = 0.01):
    """Create a VNN-LIB property file for robustness verification."""
    import torch

    # Load model to get reference output
    model = torch.jit.load(model_path.replace(".onnx", ".pt"))

    # Create sample input (centered at 0.5)
    sample = torch.ones(1, 1, 8, 8) * 0.5
    with torch.no_grad():
        output = model(sample)
        true_class = output.argmax().item()

    # Write VNN-LIB property file
    with open(output_path, "w") as f:
        # Input bounds: pixel values in [0.5 - epsilon, 0.5 + epsilon]
        for i in range(64):  # 8x8 = 64 pixels
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Output variables (2 classes)
        for i in range(2):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Input constraints
        f.write("; Input constraints: pixel perturbation\n")
        for i in range(64):
            f.write(f"(assert (>= X_{i} {0.5 - epsilon:.6f}))\n")
            f.write(f"(assert (<= X_{i} {0.5 + epsilon:.6f}))\n")
        f.write("\n")

        # Output constraint: adversarial example (other class > true class)
        f.write("; Output constraint: adversarial (wrong class wins)\n")
        other_class = 1 - true_class
        f.write(f"(assert (>= Y_{other_class} Y_{true_class}))\n")

    print(f"Created VNN-LIB property at {output_path}")
    print(f"  - True class: {true_class}")
    print(f"  - Epsilon: {epsilon}")


def run_ny_verify(model_path: str, property_path: str, method: str = "ibp"):
    """Run ny verify on the model."""
    cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "ny-cli",
        "--",
        "verify",
        model_path,
        "--property",
        property_path,
        "--method",
        method,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"stdout: {result.stdout}")
    if result.returncode != 0:
        print(f"stderr: {result.stderr}")
        return {"status": "error", "message": result.stderr}

    # Parse the output - look for status in the output
    output_lines = result.stdout.strip().split("\n")
    status = "unknown"
    for line in output_lines:
        if "property_status" in line.lower() or "status" in line.lower():
            status = line
            break

    return {"status": status, "raw_output": result.stdout}


def main():
    print("=" * 60)
    print("Testing Full CNN Pipeline with Flatten Layer")
    print("=" * 60)

    # Paths
    models_dir = Path("tests/models")
    models_dir.mkdir(exist_ok=True)

    onnx_path = str(models_dir / "cnn_with_flatten.onnx")
    pt_path = str(models_dir / "cnn_with_flatten.pt")
    vnnlib_path = str(models_dir / "cnn_with_flatten.vnnlib")

    # Step 1: Create and export model
    print("\n1. Creating CNN model with Flatten layer...")
    model = create_cnn_with_flatten()
    if model is None:
        print("Skipping test (PyTorch not available)")
        return

    # Save TorchScript for reference
    import torch

    dummy_input = torch.randn(1, 1, 8, 8)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced, pt_path)
    print(f"Saved TorchScript to {pt_path}")

    # Export to ONNX
    export_to_onnx(model, onnx_path)

    # Step 2: Create property file
    print("\n2. Creating VNN-LIB property file...")
    create_vnnlib_property(onnx_path, vnnlib_path, epsilon=0.01)

    # Step 3: Test with ny verify
    print("\n3. Running ny verify...")

    # Test with IBP first
    print("\n--- IBP Method ---")
    ibp_result = run_ny_verify(onnx_path, vnnlib_path, "ibp")
    if ibp_result:
        print(f"Result: {json.dumps(ibp_result, indent=2)}")

    # Test with CROWN
    print("\n--- CROWN Method ---")
    crown_result = run_ny_verify(onnx_path, vnnlib_path, "crown")
    if crown_result:
        print(f"Result: {json.dumps(crown_result, indent=2)}")

    print("\n" + "=" * 60)
    print("Full CNN Pipeline Test Complete!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print("  - Model architecture: Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear")
    print("  - Input shape: (1, 1, 8, 8)")
    print("  - Output shape: (1, 2)")
    if ibp_result:
        print(f"  - IBP result: {ibp_result.get('status', 'N/A')}")
    if crown_result:
        print(f"  - CROWN result: {crown_result.get('status', 'N/A')}")


if __name__ == "__main__":
    main()
