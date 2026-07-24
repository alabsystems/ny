#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Analyze CIFAR-10 verification results comparing trained vs untrained models."""

import json
import subprocess


def run_verify(model_path, property_path, method="crown"):
    """Run ny verify and return parsed JSON output."""
    cmd = [
        "cargo", "run", "--release", "--",
        "verify", model_path,
        "--property", property_path,
        "--method", method,
        "--json"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return json.loads(result.stdout)


def analyze_bounds(result, true_label=0):
    """Analyze verification bounds and margin violations."""
    bounds = result.get("output_bounds", [])
    if not bounds:
        return None

    total_width = sum(b["upper"] - b["lower"] for b in bounds)
    true_lower = bounds[true_label]["lower"]

    violations = 0
    for i, b in enumerate(bounds):
        if i != true_label and b["upper"] > true_lower:
            violations += 1

    return {
        "status": result.get("property_status", "N/A"),
        "width": total_width,
        "violations": violations,
        "true_class_bounds": bounds[true_label]
    }


def main():
    print("=" * 60)
    print("CIFAR-10 Verification Analysis: Trained vs Untrained")
    print("=" * 60)

    # Test configurations
    configs = [
        ("Untrained", "tests/models/cifar10_mlp_2x100.onnx",
         "tests/models/cifar10_robustness_eps0.010_label0.vnnlib", 0),
        ("Trained", "tests/models/cifar10_mlp_2x100_trained.onnx",
         "tests/models/cifar10_trained_robustness_eps0.010_sample0.vnnlib", 3),
    ]

    print("\n--- eps=0.01 Comparison ---")
    for name, model, prop, label in configs:
        for method in ["ibp", "crown", "alpha"]:
            result = run_verify(model, prop, method)
            if result:
                analysis = analyze_bounds(result, label)
                if analysis:
                    print(f"{name} {method:6s}: status={analysis['status']:7s}, "
                          f"width={analysis['width']:7.2f}, violations={analysis['violations']}/9")

    # Try smaller epsilon for trained model
    print("\n--- Trained model with different epsilons ---")
    for eps in ["0.010", "0.005", "0.002", "0.001"]:
        # Create a quick property file with smaller epsilon
        prop_path = f"tests/models/cifar10_trained_robustness_eps{eps}_sample0.vnnlib"
        result = run_verify(
            "tests/models/cifar10_mlp_2x100_trained.onnx",
            prop_path,
            "crown"
        )
        if result:
            analysis = analyze_bounds(result, 3)
            if analysis:
                print(f"eps={eps}: status={analysis['status']:7s}, "
                      f"width={analysis['width']:7.2f}, violations={analysis['violations']}/9")


if __name__ == "__main__":
    main()
