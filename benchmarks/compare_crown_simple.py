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

This is an explicitly selected cross-tool diagnostic, not a pytest test. It
fails nonzero if the release ny binary or any comparator dependency is absent.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

REPO_ROOT = Path(__file__).resolve().parents[1]
NY_BIN = REPO_ROOT / "target" / "release" / "ny"

# Lazy imports keep startup cheap while preserving fail-hard admission once the
# explicit comparison tool is selected.
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
        raise RuntimeError(
            "CROWN comparison prerequisite torch is missing. "
            "Install it before selecting this tool: pip install torch"
        ) from e

    try:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

        _HAS_AUTOLIRPA = True
        _BoundedModule = BoundedModule
        _BoundedTensor = BoundedTensor
        _PerturbationLpNorm = PerturbationLpNorm
    except ImportError:
        _HAS_AUTOLIRPA = False

    return _torch, _nn


def _require_comparator():
    """Require both optional packages needed for a real comparison."""
    if np is None:
        raise RuntimeError(
            "CROWN comparison prerequisite NumPy is missing. "
            "Install it before selecting this tool: pip install numpy"
        )
    torch, nn = _require_torch()
    if not _HAS_AUTOLIRPA:
        raise RuntimeError(
            "CROWN comparison prerequisite Auto-LiRPA is missing. "
            "Install it before selecting this tool: pip install auto-LiRPA"
        )
    return torch, nn


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
        raise RuntimeError(
            "Auto-LiRPA comparator is unavailable; install it before selecting "
            "this tool: pip install auto-LiRPA"
        )

    center = (lower + upper) / 2
    results = {}

    # IBP
    bounded_model = _BoundedModule(model, center)
    ptb = _PerturbationLpNorm(x_L=lower, x_U=upper)
    bounded_input = _BoundedTensor(center, ptb)
    lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method="IBP")
    results["IBP"] = {
        "lower": lb.detach().numpy()[0],
        "upper": ub.detach().numpy()[0],
    }

    # CROWN (backward) - use a fresh bounded model.
    bounded_model = _BoundedModule(model, center)
    ptb = _PerturbationLpNorm(x_L=lower, x_U=upper)
    bounded_input = _BoundedTensor(center, ptb)
    lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method="backward")
    results["CROWN"] = {
        "lower": lb.detach().numpy()[0],
        "upper": ub.detach().numpy()[0],
    }

    return results


def run_ny(weights, biases, lower, upper):
    """Run ny on a generated NNet without modifying committed fixtures."""
    with tempfile.TemporaryDirectory(prefix="ny-crown-comparison-") as temp_dir:
        nnet_path = Path(temp_dir) / "model.nnet"
        save_as_nnet(weights, biases, nnet_path)
        return _run_ny_file(nnet_path, lower, upper)


def _run_ny_file(nnet_path, lower, upper):
    """Run ny against one temporary NNet and return its reported bounds."""
    if not NY_BIN.is_file():
        raise RuntimeError(f"ny binary is missing: {NY_BIN}")

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

        def parse_result(label, result):
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"ny {label} invocation failed with code {result.returncode}: "
                    f"{detail[:500]}"
                )
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"ny {label} returned invalid JSON: {result.stdout[:500]}"
                ) from exc

            output_bounds = data.get("output_bounds")
            if not isinstance(output_bounds, list) or not output_bounds:
                raise AssertionError(f"ny {label} JSON omitted output_bounds")
            try:
                lower_bounds = np.array(
                    [bound["lower"] for bound in output_bounds], dtype=np.float64
                )
                upper_bounds = np.array(
                    [bound["upper"] for bound in output_bounds], dtype=np.float64
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise AssertionError(f"ny {label} output_bounds are malformed") from exc
            return {"lower": lower_bounds, "upper": upper_bounds}

        results = {
            "IBP": parse_result("IBP", result_ibp),
            "CROWN": parse_result("CROWN", result_crown),
            "CROWN-IBP": parse_result("CROWN-IBP", result_crown_ibp),
        }

        return results, result_ibp.stdout, result_crown.stdout

    finally:
        Path(vnnlib_path).unlink()


def compare_bounds(auto_results, ny_results, method):
    """Validate and print a diagnostic bound comparison.

    Tightness differences are intentionally non-gating, but missing, malformed,
    or non-finite comparator data is a failure rather than a false pass.
    """
    auto_method = "CROWN" if method == "CROWN-IBP" else method
    if auto_results is None or auto_method not in auto_results:
        raise AssertionError(f"Auto-LiRPA {auto_method} data is missing")

    if method not in ny_results:
        raise AssertionError(f"ny {method} data is missing")

    auto = auto_results[auto_method]
    ny = ny_results[method]
    auto_lower = np.asarray(auto["lower"], dtype=np.float64)
    auto_upper = np.asarray(auto["upper"], dtype=np.float64)
    ny_lower = np.asarray(ny["lower"], dtype=np.float64)
    ny_upper = np.asarray(ny["upper"], dtype=np.float64)

    if auto_lower.shape != auto_upper.shape:
        raise AssertionError(f"Auto-LiRPA {auto_method} lower/upper shapes differ")
    if ny_lower.shape != ny_upper.shape:
        raise AssertionError(f"ny {method} lower/upper shapes differ")
    if auto_lower.shape != ny_lower.shape:
        raise AssertionError(
            f"{method} output shape mismatch: Auto-LiRPA {auto_lower.shape}, "
            f"ny {ny_lower.shape}"
        )
    if not all(
        np.all(np.isfinite(values))
        for values in (auto_lower, auto_upper, ny_lower, ny_upper)
    ):
        raise AssertionError(f"{method} comparison contains non-finite bounds")
    if np.any(auto_lower > auto_upper) or np.any(ny_lower > ny_upper):
        raise AssertionError(f"{method} comparison contains inverted bounds")

    auto_width = float(np.sum(auto_upper - auto_lower))
    ny_width = float(np.sum(ny_upper - ny_lower))

    print(f"\n  {method}:")
    print(f"    Auto-LiRPA: lower={np.array2string(auto_lower, precision=4)}")
    print(f"                upper={np.array2string(auto_upper, precision=4)}")
    print(f"                width={auto_width:.6f}")
    print(f"    ny:    lower={np.array2string(ny_lower, precision=4)}")
    print(f"                upper={np.array2string(ny_upper, precision=4)}")
    print(f"                width={ny_width:.6f}")
    print(f"    Max lower delta: {np.max(np.abs(ny_lower - auto_lower)):.6e}")
    print(f"    Max upper delta: {np.max(np.abs(ny_upper - auto_upper)):.6e}")

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
    if ny_width == 0:
        print("    Ratio (γ/auto): exact zero-width match")
        return 1.0
    print("    Ratio (γ/auto): infinite (Auto-LiRPA width is zero)")
    return float("inf")


def compare_2layer_network():
    """Compare a 2-layer network: input -> Linear -> ReLU -> Linear -> output."""
    torch, nn = _require_comparator()
    print("\n" + "=" * 60)
    print("Test 1: 2-Layer Network (5 -> 10 -> 5)")
    print("=" * 60)

    # Create model
    model, weights, biases = create_simple_model([(5, 10), (10, 5)])

    # Define input bounds
    lower = torch.tensor([[-0.5, -0.5, -0.5, -0.5, -0.5]], dtype=torch.float32)
    upper = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

    print(f"Input bounds: [{lower[0].tolist()}, {upper[0].tolist()}]")

    # Run both
    auto_results = run_autolirpa(model, lower, upper)
    ny_results, ibp_out, crown_out = run_ny(
        weights, biases, lower[0].numpy(), upper[0].numpy()
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


def compare_3layer_network():
    """Compare a 3-layer network similar to ACAS-Xu structure."""
    torch, nn = _require_comparator()
    print("\n" + "=" * 60)
    print("Test 2: 3-Layer Network (5 -> 50 -> 50 -> 5)")
    print("=" * 60)

    # Create model
    model, weights, biases = create_simple_model([(5, 50), (50, 50), (50, 5)])

    # Define input bounds
    lower = torch.tensor([[-0.5, -0.5, -0.5, -0.5, -0.5]], dtype=torch.float32)
    upper = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

    print(f"Input bounds: [{lower[0].tolist()}, {upper[0].tolist()}]")

    # Run both
    auto_results = run_autolirpa(model, lower, upper)
    ny_results, ibp_out, crown_out = run_ny(
        weights, biases, lower[0].numpy(), upper[0].numpy()
    )

    if auto_results:
        compare_bounds(auto_results, ny_results, "IBP")
        compare_bounds(auto_results, ny_results, "CROWN")
        # CROWN-IBP should match Auto-LiRPA's CROWN (backward)
        compare_bounds(auto_results, ny_results, "CROWN-IBP")


def compare_single_relu():
    """Compare a single ReLU: just Linear -> ReLU -> Linear."""
    torch, nn = _require_comparator()
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

    # Define input bounds
    lower = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
    upper = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    print(f"\nInput bounds: [{lower[0].tolist()}, {upper[0].tolist()}]")

    # Run both
    auto_results = run_autolirpa(model, lower, upper)
    ny_results, ibp_out, crown_out = run_ny(
        weights, biases, lower[0].numpy(), upper[0].numpy()
    )

    compare_bounds(auto_results, ny_results, "IBP")
    compare_bounds(auto_results, ny_results, "CROWN")


def compare_crossing_relu():
    """Compare a ReLU that definitely crosses zero.

    This is the critical case for CROWN relaxation.
    """
    torch, nn = _require_comparator()
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
        weights, biases, lower[0].numpy(), upper[0].numpy()
    )

    compare_bounds(auto_results, ny_results, "IBP")
    compare_bounds(auto_results, ny_results, "CROWN")


def main() -> int:
    """Run every comparison and report explicit success or failure."""
    comparisons = [
        ("minimal ReLU", compare_single_relu),
        ("crossing ReLU", compare_crossing_relu),
        ("2-layer network", compare_2layer_network),
        ("3-layer network", compare_3layer_network),
    ]
    results = []

    if not NY_BIN.is_file():
        print(f"FAIL: ny binary is missing: {NY_BIN}")
        print("Build with: cargo build --release -p ny-cli")
        return 1

    for name, comparison in comparisons:
        try:
            comparison()
        except Exception as exc:
            print(f"\nFAIL: {name}: {exc}")
            results.append((name, "FAIL", str(exc)))
        else:
            results.append((name, "PASS", "comparison completed"))

    passed = sum(status == "PASS" for _, status, _ in results)
    failed = sum(status == "FAIL" for _, status, _ in results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status, detail in results:
        print(f"  {name}: {status} ({detail})")
    print(f"\nTotal: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
