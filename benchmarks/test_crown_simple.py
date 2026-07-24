#!/usr/bin/env python3
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Compare ny vs Auto-LiRPA on simple 2-3 layer networks.

Goal: Isolate where CROWN bound discrepancy starts.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
NY_BIN = REPO_ROOT / "target" / "release" / "ny"
TEST_MODELS_DIR = REPO_ROOT / "tests" / "models"

# Lazy torch import - allows pytest collection without torch installed
_torch = None
_nn = None
_HAS_AUTOLIRPA = None
_BoundedModule = None
_BoundedTensor = None
_PerturbationLpNorm = None


def _require_torch():
    """Lazy import torch and auto_LiRPA."""
    global \
        _torch, \
        _nn, \
        _HAS_AUTOLIRPA, \
        _BoundedModule, \
        _BoundedTensor, \
        _PerturbationLpNorm
    if _torch is not None:
        return _torch, _nn
    try:
        import torch
        import torch.nn as nn

        _torch = torch
        _nn = nn
    except ModuleNotFoundError as e:
        raise AssertionError(
            "Missing dependency: torch required.\nInstall with: pip install torch"
        ) from e

    try:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

        _HAS_AUTOLIRPA = True
        _BoundedModule = BoundedModule
        _BoundedTensor = BoundedTensor
        _PerturbationLpNorm = PerturbationLpNorm
    except ImportError:
        _HAS_AUTOLIRPA = False
        print("Warning: Auto-LiRPA not available. Install with: pip install auto-LiRPA")

    return _torch, _nn


def create_simple_model(layers_config, seed=42):
    """Create a simple feedforward network.

    layers_config: list of (in_dim, out_dim) tuples
    Returns: nn.Sequential, weights, biases
    """
    torch, nn = _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    layers = []
    weights = []
    biases = []

    for i, (in_dim, out_dim) in enumerate(layers_config):
        linear = nn.Linear(in_dim, out_dim)
        # Use small random weights for better numerical behavior
        nn.init.uniform_(linear.weight, -0.5, 0.5)
        nn.init.uniform_(linear.bias, -0.1, 0.1)

        weights.append(linear.weight.data.numpy().copy())
        biases.append(linear.bias.data.numpy().copy())

        layers.append(linear)
        if i < len(layers_config) - 1:  # ReLU after all but last layer
            layers.append(nn.ReLU())

    model = nn.Sequential(*layers)
    model.eval()
    return model, weights, biases


def save_as_nnet(weights, biases, filepath):
    """Save weights as NNet format for ny."""
    num_layers = len(weights)
    input_size = weights[0].shape[1]
    output_size = weights[-1].shape[0]
    layer_sizes = [input_size] + [w.shape[0] for w in weights]
    max_layer = max(layer_sizes)

    with open(filepath, "w") as f:
        f.write("// NNet format - simple test network\n")
        f.write(f"{num_layers},{input_size},{output_size},{max_layer},\n")
        f.write(",".join(map(str, layer_sizes)) + ",\n")
        f.write("0,\n")  # Unused
        f.write(",".join(["0.0"] * input_size) + ",\n")  # Input means
        f.write(",".join(["1.0"] * input_size) + ",\n")  # Input ranges
        f.write("0.0,\n")  # Output mean
        f.write("1.0,\n")  # Output range

        for wmat, bvec in zip(weights, biases):
            f.writelines(",".join(map(str, row)) + ",\n" for row in wmat)
            f.writelines(f"{val},\n" for val in bvec)


def run_autolirpa(model, lower, upper):
    """Run Auto-LiRPA and get bounds."""
    _require_torch()  # Ensure imports are loaded
    if not _HAS_AUTOLIRPA:
        return None

    center = (lower + upper) / 2
    results = {}

    try:
        # IBP
        bounded_model = _BoundedModule(model, center)
        ptb = _PerturbationLpNorm(x_L=lower, x_U=upper)
        bounded_input = _BoundedTensor(center, ptb)
        lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method="IBP")
        results["IBP"] = {
            "lower": lb.detach().numpy()[0],
            "upper": ub.detach().numpy()[0],
        }
    except Exception as e:
        print(f"Auto-LiRPA IBP error: {e}")

    try:
        # CROWN (backward) - need fresh model
        bounded_model = _BoundedModule(model, center)
        ptb = _PerturbationLpNorm(x_L=lower, x_U=upper)
        bounded_input = _BoundedTensor(center, ptb)
        lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method="backward")
        results["CROWN"] = {
            "lower": lb.detach().numpy()[0],
            "upper": ub.detach().numpy()[0],
        }
    except Exception as e:
        print(f"Auto-LiRPA CROWN error: {e}")

    return results


def run_ny(nnet_path, lower, upper):
    """Run ny and get bounds."""
    # Create VNNLIB file
    input_dim = len(lower)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
        # Declare input variables
        for i in range(input_dim):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variable - just Y_0 for simple networks
        f.write("(declare-const Y_0 Real)\n")

        # Input constraints
        for i in range(input_dim):
            f.write(f"(assert (>= X_{i} {lower[i]}))\n")
            f.write(f"(assert (<= X_{i} {upper[i]}))\n")

        # Dummy property (we just want bounds, not verification)
        f.write("(assert (<= Y_0 1000000.0))\n")

        vnnlib_path = f.name

    try:
        # Run ny with IBP
        result_ibp = subprocess.run(
            [
                str(NY_BIN),
                "verify",
                nnet_path,
                "-p",
                vnnlib_path,
                "--method",
                "ibp",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=REPO_ROOT,
        )

        # Run ny with CROWN
        result_crown = subprocess.run(
            [
                str(NY_BIN),
                "verify",
                nnet_path,
                "-p",
                vnnlib_path,
                "--method",
                "crown",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=REPO_ROOT,
        )

        # Run ny with CROWN-IBP (should match Auto-LiRPA backward)
        result_crown_ibp = subprocess.run(
            [
                str(NY_BIN),
                "verify",
                nnet_path,
                "-p",
                vnnlib_path,
                "--method",
                "crown-ibp",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=REPO_ROOT,
        )

        results = {}

        # Parse IBP output - full JSON blob
        try:
            data = json.loads(result_ibp.stdout)
            if "output_bounds" in data:
                results["IBP"] = {
                    "lower": np.array([b["lower"] for b in data["output_bounds"]]),
                    "upper": np.array([b["upper"] for b in data["output_bounds"]]),
                }
        except json.JSONDecodeError as e:
            print(f"IBP JSON parse error: {e}")
            print(f"IBP stdout: {result_ibp.stdout[:500]}")

        # Parse CROWN output - full JSON blob
        try:
            data = json.loads(result_crown.stdout)
            if "output_bounds" in data:
                results["CROWN"] = {
                    "lower": np.array([b["lower"] for b in data["output_bounds"]]),
                    "upper": np.array([b["upper"] for b in data["output_bounds"]]),
                }
        except json.JSONDecodeError as e:
            print(f"CROWN JSON parse error: {e}")
            print(f"CROWN stdout: {result_crown.stdout[:500]}")

        # Parse CROWN-IBP output - full JSON blob
        try:
            data = json.loads(result_crown_ibp.stdout)
            if "output_bounds" in data:
                results["CROWN-IBP"] = {
                    "lower": np.array([b["lower"] for b in data["output_bounds"]]),
                    "upper": np.array([b["upper"] for b in data["output_bounds"]]),
                }
        except json.JSONDecodeError as e:
            print(f"CROWN-IBP JSON parse error: {e}")
            print(f"CROWN-IBP stdout: {result_crown_ibp.stdout[:500]}")

        return results, result_ibp.stdout, result_crown.stdout

    finally:
        Path(vnnlib_path).unlink()


def compare_bounds(auto_results, ny_results, method):
    """Compare bounds between Auto-LiRPA and ny."""
    if auto_results is None or method not in auto_results:
        print(f"  {method}: Auto-LiRPA data missing")
        if ny_results and method in ny_results:
            ny = ny_results[method]
            ny_width = np.sum(ny["upper"] - ny["lower"])
            print(
                f"    ny only: lower={ny['lower']}, upper={ny['upper']}, width={ny_width:.6f}"
            )
        return None

    if method not in ny_results:
        print(f"  {method}: ny data missing")
        return None

    auto = auto_results[method]
    ny = ny_results[method]

    auto_width = np.sum(auto["upper"] - auto["lower"])
    ny_width = np.sum(ny["upper"] - ny["lower"])

    print(f"\n  {method}:")
    print(f"    Auto-LiRPA: lower={np.array2string(auto['lower'], precision=4)}")
    print(f"                upper={np.array2string(auto['upper'], precision=4)}")
    print(f"                width={auto_width:.6f}")
    print(f"    ny:    lower={np.array2string(ny['lower'], precision=4)}")
    print(f"                upper={np.array2string(ny['upper'], precision=4)}")
    print(f"                width={ny_width:.6f}")

    if auto_width > 0:
        ratio = ny_width / auto_width
        if ratio > 1.01:
            verdict = "ny LOOSER"
        elif ratio < 0.99:
            verdict = "ny TIGHTER"
        else:
            verdict = "MATCH"
        print(f"    Ratio (γ/auto): {ratio:.4f}x  [{verdict}]")
        return ratio
    return None


def test_2layer_network():
    """Test simple 2-layer network: input -> Linear -> ReLU -> Linear -> output."""
    torch, nn = _require_torch()
    print("\n" + "=" * 60)
    print("Test 1: 2-Layer Network (5 -> 10 -> 5)")
    print("=" * 60)

    # Create model
    model, weights, biases = create_simple_model([(5, 10), (10, 5)])

    # Save for ny
    nnet_path = str(TEST_MODELS_DIR / "simple_2layer.nnet")
    save_as_nnet(weights, biases, nnet_path)

    # Define input bounds
    lower = torch.tensor([[-0.5, -0.5, -0.5, -0.5, -0.5]], dtype=torch.float32)
    upper = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

    print(f"Input bounds: [{lower[0].tolist()}, {upper[0].tolist()}]")

    # Run both
    auto_results = run_autolirpa(model, lower, upper)
    ny_results, ibp_out, crown_out = run_ny(
        nnet_path, lower[0].numpy(), upper[0].numpy()
    )

    if auto_results:
        compare_bounds(auto_results, ny_results, "IBP")
        compare_bounds(auto_results, ny_results, "CROWN")
        # CROWN-IBP should match Auto-LiRPA's CROWN (backward) on deeper networks
        compare_bounds(auto_results, ny_results, "CROWN-IBP")
    else:
        print("Auto-LiRPA not available, showing ny only:")
        for method, data in ny_results.items():
            print(f"  {method}: {data}")


def test_3layer_network():
    """Test 3-layer network similar to ACAS-Xu structure."""
    torch, nn = _require_torch()
    print("\n" + "=" * 60)
    print("Test 2: 3-Layer Network (5 -> 50 -> 50 -> 5)")
    print("=" * 60)

    # Create model
    model, weights, biases = create_simple_model([(5, 50), (50, 50), (50, 5)])

    # Save for ny
    nnet_path = str(TEST_MODELS_DIR / "simple_3layer.nnet")
    save_as_nnet(weights, biases, nnet_path)

    # Define input bounds
    lower = torch.tensor([[-0.5, -0.5, -0.5, -0.5, -0.5]], dtype=torch.float32)
    upper = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

    print(f"Input bounds: [{lower[0].tolist()}, {upper[0].tolist()}]")

    # Run both
    auto_results = run_autolirpa(model, lower, upper)
    ny_results, ibp_out, crown_out = run_ny(
        nnet_path, lower[0].numpy(), upper[0].numpy()
    )

    if auto_results:
        compare_bounds(auto_results, ny_results, "IBP")
        compare_bounds(auto_results, ny_results, "CROWN")
        # CROWN-IBP should match Auto-LiRPA's CROWN (backward)
        compare_bounds(auto_results, ny_results, "CROWN-IBP")


def test_single_relu():
    """Test single ReLU: just Linear -> ReLU -> Linear."""
    torch, nn = _require_torch()
    print("\n" + "=" * 60)
    print("Test 3: Minimal ReLU Network (2 -> 3 -> 1)")
    print("=" * 60)

    # Very simple network
    model, weights, biases = create_simple_model([(2, 3), (3, 1)])

    # Print weights for debugging
    print(f"Layer 0 weights:\n{weights[0]}")
    print(f"Layer 0 bias: {biases[0]}")
    print(f"Layer 1 weights:\n{weights[1]}")
    print(f"Layer 1 bias: {biases[1]}")

    # Save for ny
    nnet_path = str(TEST_MODELS_DIR / "minimal_relu.nnet")
    save_as_nnet(weights, biases, nnet_path)

    # Define input bounds
    lower = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
    upper = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    print(f"\nInput bounds: [{lower[0].tolist()}, {upper[0].tolist()}]")

    # Run both
    auto_results = run_autolirpa(model, lower, upper)
    ny_results, ibp_out, crown_out = run_ny(
        nnet_path, lower[0].numpy(), upper[0].numpy()
    )

    compare_bounds(auto_results, ny_results, "IBP")
    compare_bounds(auto_results, ny_results, "CROWN")


def test_crossing_relu():
    """Test with a ReLU that definitely crosses zero.

    This is the critical case for CROWN relaxation.
    """
    torch, nn = _require_torch()
    print("\n" + "=" * 60)
    print("Test 4: Crossing ReLU Network (1 -> 2 -> 1)")
    print("=" * 60)

    # Manually construct weights so we know exactly what's happening
    torch.manual_seed(0)

    layers = []

    # Layer 1: input -> 2 neurons
    l1 = nn.Linear(1, 2)
    l1.weight.data = torch.tensor([[1.0], [-1.0]])  # One positive, one negative
    l1.bias.data = torch.tensor([0.0, 0.0])
    layers.append(l1)
    layers.append(nn.ReLU())

    # Layer 2: 2 -> 1
    l2 = nn.Linear(2, 1)
    l2.weight.data = torch.tensor([[1.0, 1.0]])
    l2.bias.data = torch.tensor([0.0])
    layers.append(l2)

    model = nn.Sequential(*layers)
    model.eval()

    weights = [l1.weight.data.numpy(), l2.weight.data.numpy()]
    biases = [l1.bias.data.numpy(), l2.bias.data.numpy()]

    print(f"Layer 0 weights: {weights[0]}, bias: {biases[0]}")
    print(f"Layer 1 weights: {weights[1]}, bias: {biases[1]}")

    # Save for ny
    nnet_path = str(TEST_MODELS_DIR / "crossing_relu.nnet")
    save_as_nnet(weights, biases, nnet_path)

    # Define input bounds: x in [-1, 1]
    # After layer 0: neuron 0 in [-1, 1], neuron 1 in [-1, 1]
    # After ReLU: neuron 0 in [0, 1], neuron 1 in [0, 1]
    # After layer 1: output in [0, 2]
    #
    # But with CROWN relaxation, should be tighter for specific cases
    lower = torch.tensor([[-1.0]], dtype=torch.float32)
    upper = torch.tensor([[1.0]], dtype=torch.float32)

    print(f"\nInput bounds: [{lower[0].tolist()}, {upper[0].tolist()}]")
    print("\nExpected (exact):")
    print("  Pre-ReLU 0: [-1, 1] (crossing)")
    print("  Pre-ReLU 1: [-1, 1] (crossing)")
    print("  Post-ReLU: [0, 1] each")
    print("  Output: [0, 2]")

    # Run both
    auto_results = run_autolirpa(model, lower, upper)
    ny_results, ibp_out, crown_out = run_ny(
        nnet_path, lower[0].numpy(), upper[0].numpy()
    )

    compare_bounds(auto_results, ny_results, "IBP")
    compare_bounds(auto_results, ny_results, "CROWN")


if __name__ == "__main__":
    if not NY_BIN.exists():
        raise SystemExit("Build ny first: cargo build --release -p ny-cli")

    test_single_relu()
    test_crossing_relu()
    test_2layer_network()
    test_3layer_network()
