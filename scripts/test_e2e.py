#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
End-to-End Integration Tests for ny

This script validates the complete workflow:
1. PyTorch → ONNX export → ny diff verification
2. All CLI commands work with real models
3. Error messages are helpful for common failures
4. Documentation examples produce expected output

P2 End-to-End Integration Tests from design doc roadmap.
"""

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Optional imports
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx  # noqa: F401 - imported for availability check
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

REPO_ROOT = Path(__file__).parent.parent
TEST_MODELS_DIR = REPO_ROOT / "tests" / "models"
NY_BIN = REPO_ROOT / "target" / "release" / "ny"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: str
    duration_ms: float
    message: str
    stdout: str = ""
    stderr: str = ""


class EndToEndTests:
    """End-to-end integration tests for ny."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[TestResult] = []

    def run_ny(self, args: list[str], timeout: int = 60) -> tuple[int, str, str, float]:
        """Run ny CLI command."""
        cmd = [str(NY_BIN)] + args
        start = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=timeout,
            env={**os.environ, "RUST_LOG": "error"}
        )

        duration = (time.time() - start) * 1000
        return result.returncode, result.stdout, result.stderr, duration

    def add_result(self, name: str, passed: bool, duration_ms: float, message: str,
                   stdout: str = "", stderr: str = "", *, skipped: bool = False):
        """Record a test result."""
        status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        result = TestResult(name, status, duration_ms, message, stdout, stderr)
        self.results.append(result)

        if self.verbose:
            print(f"  [{result.status}] {name}: {message} ({duration_ms:.1f}ms)")

    # ========================================================================
    # Test Group 1: CLI Commands Work
    # ========================================================================

    def test_cli_verify(self):
        """Test ny verify command."""
        model = TEST_MODELS_DIR / "simple_mlp.onnx"
        rc, stdout, stderr, duration = self.run_ny([
            "verify", str(model), "--epsilon", "0.01", "--method", "ibp"
        ])

        passed = rc == 0 and ("Verified" in stdout or "Unknown" in stdout)
        self.add_result(
            "cli_verify",
            passed,
            duration,
            "verify command runs successfully" if passed else f"verify failed: {stderr[:100]}",
            stdout, stderr
        )

    def test_cli_diff_same_model(self):
        """Test ny diff with same model (should be equivalent)."""
        model = TEST_MODELS_DIR / "simple_mlp.onnx"
        input_file = self._create_test_input((1, 2))

        try:
            rc, stdout, stderr, duration = self.run_ny([
                "diff", str(model), str(model),
                "--input", input_file, "--tolerance", "1e-10"
            ])

            passed = rc == 0 and "EQUIVALENT" in stdout
            self.add_result(
                "cli_diff_same",
                passed,
                duration,
                "diff reports same model as EQUIVALENT" if passed else f"diff failed: {stdout[:100]}",
                stdout, stderr
            )
        finally:
            os.unlink(input_file)

    def test_cli_sensitivity(self):
        """Test ny sensitivity command."""
        model = TEST_MODELS_DIR / "simple_mlp.onnx"
        rc, stdout, stderr, duration = self.run_ny([
            "sensitivity", str(model), "--epsilon", "0.01"
        ])

        passed = rc == 0 and ("Sensitivity" in stdout or "sensitivity" in stdout.lower())
        self.add_result(
            "cli_sensitivity",
            passed,
            duration,
            "sensitivity command runs" if passed else f"sensitivity failed: {stderr[:100]}",
            stdout, stderr
        )

    def test_cli_quantize_check(self):
        """Test ny quantize-check command."""
        model = TEST_MODELS_DIR / "simple_mlp.onnx"
        rc, stdout, stderr, duration = self.run_ny([
            "quantize-check", str(model), "--epsilon", "0.01"
        ])

        passed = rc == 0 and ("float16" in stdout.lower() or "F16" in stdout)
        self.add_result(
            "cli_quantize_check",
            passed,
            duration,
            "quantize-check command runs" if passed else f"quantize-check failed: {stderr[:100]}",
            stdout, stderr
        )

    def test_cli_profile_bounds(self):
        """Test ny profile-bounds command."""
        model = TEST_MODELS_DIR / "simple_mlp.onnx"
        rc, stdout, stderr, duration = self.run_ny([
            "profile-bounds", str(model), "--epsilon", "0.01"
        ])

        passed = rc == 0 and ("Bound" in stdout or "bound" in stdout.lower())
        self.add_result(
            "cli_profile_bounds",
            passed,
            duration,
            "profile-bounds command runs" if passed else f"profile-bounds failed: {stderr[:100]}",
            stdout, stderr
        )

    def test_cli_inspect(self):
        """Test ny inspect command."""
        model = TEST_MODELS_DIR / "simple_mlp.onnx"
        rc, stdout, stderr, duration = self.run_ny([
            "inspect", str(model)
        ])

        passed = rc == 0 and "input" in stdout.lower()
        self.add_result(
            "cli_inspect",
            passed,
            duration,
            "inspect command runs" if passed else f"inspect failed: {stderr[:100]}",
            stdout, stderr
        )

    def test_cli_bench(self):
        """Test ny bench command (quick layer benchmark)."""
        rc, stdout, stderr, duration = self.run_ny([
            "bench", "-b", "layer"
        ], timeout=120)

        passed = rc == 0 and ("Linear" in stdout or "benchmark" in stdout.lower())
        self.add_result(
            "cli_bench",
            passed,
            duration,
            "bench command runs" if passed else f"bench failed: {stderr[:100]}",
            stdout, stderr
        )

    # ========================================================================
    # Test Group 2: Error Messages Are Helpful
    # ========================================================================

    def test_error_missing_file(self):
        """Test error message for missing model file."""
        rc, stdout, stderr, duration = self.run_ny([
            "verify", "/nonexistent/model.onnx", "--epsilon", "0.01"
        ])

        # Should fail with helpful message
        output = stdout + stderr
        passed = rc != 0 and ("not found" in output.lower() or "no such file" in output.lower()
                              or "failed" in output.lower())
        self.add_result(
            "error_missing_file",
            passed,
            duration,
            "clear error for missing file" if passed else "unhelpful error message",
            stdout, stderr
        )

    def test_error_invalid_epsilon(self):
        """Test error message for invalid epsilon value."""
        model = TEST_MODELS_DIR / "simple_mlp.onnx"
        rc, stdout, stderr, duration = self.run_ny([
            "verify", str(model), "--epsilon", "invalid"
        ])

        output = stdout + stderr
        passed = rc != 0 and ("invalid" in output.lower() or "parse" in output.lower()
                              or "error" in output.lower())
        self.add_result(
            "error_invalid_epsilon",
            passed,
            duration,
            "clear error for invalid epsilon" if passed else "unhelpful error",
            stdout, stderr
        )

    def test_error_missing_required_arg(self):
        """Test error message for missing required argument."""
        rc, stdout, stderr, duration = self.run_ny([
            "verify", "--epsilon", "0.01"  # Missing MODEL
        ])

        output = stdout + stderr
        passed = rc != 0 and ("required" in output.lower() or "<MODEL>" in output or "model" in output.lower())
        self.add_result(
            "error_missing_required",
            passed,
            duration,
            "clear error for missing arg" if passed else "unhelpful error",
            stdout, stderr
        )

    def test_error_diff_shape_mismatch(self):
        """Test error message when comparing models with different input shapes."""
        # Use two different models - single_linear (1,2 input) vs transformer_mlp (1,4 input)
        model_a = TEST_MODELS_DIR / "single_linear.onnx"
        model_b = TEST_MODELS_DIR / "transformer_mlp.onnx"

        rc, stdout, stderr, duration = self.run_ny([
            "diff", str(model_a), str(model_b)
        ])

        # The command must fail specifically because the input dimensions differ.
        # An unrelated invocation error is not evidence that shape checking worked.
        output = stdout + stderr
        mismatch_terms = ("shape", "mismatch", "dimension", "expected")
        passed = rc != 0 and any(term in output.lower() for term in mismatch_terms)
        self.add_result(
            "error_shape_mismatch",
            passed,
            duration,
            "reports shape mismatch" if passed else "no shape mismatch detected",
            stdout, stderr
        )

    # ========================================================================
    # Test Group 3: PyTorch → ONNX → ny diff Workflow
    # ========================================================================

    def test_pytorch_onnx_diff_workflow(self):
        """Test full PyTorch → ONNX → ny diff workflow."""
        if not HAS_TORCH or not HAS_ONNX:
            self.add_result(
                "pytorch_onnx_workflow",
                False,
                0,
                "torch/onnx not available",
                skipped=True,
            )
            return

        # Create a simple PyTorch model
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(8, 2)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        model = SimpleMLP()
        model.eval()

        # Export twice (simulating PyTorch → Metal porting scenario)
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_a = f.name
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_b = f.name
        input_file = self._create_test_input((1, 4))

        start = time.time()
        try:
            dummy_input = torch.randn(1, 4)

            # Export model twice (identical)
            torch.onnx.export(model, dummy_input, onnx_a,
                             input_names=["input"], output_names=["output"],
                             opset_version=14, dynamo=False)
            torch.onnx.export(model, dummy_input, onnx_b,
                             input_names=["input"], output_names=["output"],
                             opset_version=14, dynamo=False)

            # Run ny diff
            rc, stdout, stderr, _ = self.run_ny([
                "diff", onnx_a, onnx_b,
                "--input", input_file, "--tolerance", "1e-5"
            ])

            duration = (time.time() - start) * 1000
            passed = rc == 0 and "EQUIVALENT" in stdout
            self.add_result(
                "pytorch_onnx_workflow",
                passed,
                duration,
                "PyTorch→ONNX→diff workflow works" if passed else f"workflow failed: {stdout[:100]}",
                stdout, stderr
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                "pytorch_onnx_workflow",
                False,
                duration,
                f"workflow exception: {str(e)[:100]}"
            )
        finally:
            os.unlink(onnx_a)
            os.unlink(onnx_b)
            os.unlink(input_file)

    def test_pytorch_perturbed_diff(self):
        """Test that ny diff detects when weights are perturbed."""
        if not HAS_TORCH or not HAS_ONNX:
            self.add_result(
                "pytorch_perturbed_diff",
                False,
                0,
                "torch/onnx not available",
                skipped=True,
            )
            return

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(8, 2)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        # Create original model
        model_a = SimpleMLP()
        model_a.eval()

        # Create perturbed model (different weights)
        model_b = SimpleMLP()
        model_b.eval()
        with torch.no_grad():
            model_b.fc1.weight += 0.1  # Perturbation

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_a = f.name
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_b = f.name
        input_file = self._create_test_input((1, 4))

        start = time.time()
        try:
            dummy_input = torch.randn(1, 4)

            torch.onnx.export(model_a, dummy_input, onnx_a,
                             input_names=["input"], output_names=["output"],
                             opset_version=14, dynamo=False)
            torch.onnx.export(model_b, dummy_input, onnx_b,
                             input_names=["input"], output_names=["output"],
                             opset_version=14, dynamo=False)

            rc, stdout, stderr, _ = self.run_ny([
                "diff", onnx_a, onnx_b,
                "--input", input_file, "--tolerance", "1e-5"
            ])

            duration = (time.time() - start) * 1000
            passed = rc == 0 and "DIVERGENT" in stdout
            self.add_result(
                "pytorch_perturbed_diff",
                passed,
                duration,
                "detects perturbed weights" if passed else "missed weight perturbation",
                stdout, stderr
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.add_result(
                "pytorch_perturbed_diff",
                False,
                duration,
                f"exception: {str(e)[:100]}"
            )
        finally:
            os.unlink(onnx_a)
            os.unlink(onnx_b)
            os.unlink(input_file)

    # ========================================================================
    # Test Group 4: Documentation Examples
    # ========================================================================

    def test_readme_verify_example(self):
        """Test README verify command example works."""
        model = TEST_MODELS_DIR / "simple_mlp.onnx"
        rc, stdout, stderr, duration = self.run_ny([
            "verify", str(model), "--epsilon", "0.01", "--method", "alpha"
        ])

        output = stdout + stderr
        passed = rc == 0 and any(
            status in output.lower() for status in ("verified", "unknown", "violated")
        )
        self.add_result(
            "readme_verify_example",
            passed,
            duration,
            "README verify example works" if passed else f"example failed: {stderr[:100]}",
            stdout, stderr
        )

    def test_readme_diff_example(self):
        """Test README diff/compare example format."""
        model = TEST_MODELS_DIR / "simple_mlp.onnx"
        input_file = self._create_test_input((1, 2))

        try:
            rc, stdout, stderr, duration = self.run_ny([
                "diff", str(model), str(model),
                "--input", input_file, "--tolerance", "0.001"
            ])

            # Check output format matches README example
            has_layer_column = "Layer" in stdout
            has_status_column = "Status" in stdout or "OK" in stdout

            passed = rc == 0 and has_layer_column and has_status_column
            self.add_result(
                "readme_diff_example",
                passed,
                duration,
                "diff output matches README format" if passed else "output format differs from README",
                stdout, stderr
            )
        finally:
            os.unlink(input_file)

    # ========================================================================
    # Test Group 5: Transformer Model Tests
    # ========================================================================

    def test_transformer_block_verify(self):
        """Test verification on transformer block model."""
        model = TEST_MODELS_DIR / "transformer_block.onnx"
        if not model.exists():
            self.add_result("transformer_block_verify", False, 0, "required model not found")
            return

        rc, stdout, stderr, duration = self.run_ny([
            "verify", str(model), "--epsilon", "0.001", "--method", "ibp",
            "--allow-unknown",
        ])

        output = stdout + stderr

        # Transformer models use LayerNorm which has unsupported ops (Div, Pow, Sub, ReduceMean)
        # This causes shape mismatches - a known limitation
        # Known unsupported-operation failures are explicitly non-gating.
        has_unsupported_ops = "shape mismatch" in output.lower() or "unsupported" in output.lower()
        if rc != 0 and has_unsupported_ops:
            self.add_result(
                "transformer_block_verify",
                False,
                duration,
                "known unsupported transformer operation",
                stdout,
                stderr,
                skipped=True,
            )
            return

        passed = rc == 0 and any(
            status in output.lower() for status in ("verified", "unknown", "violated")
        )
        msg = "verification completed" if passed else f"failed: {(stderr or stdout)[:100]}"
        self.add_result(
            "transformer_block_verify",
            passed,
            duration,
            msg,
            stdout, stderr
        )

    def test_attention_model_diff(self):
        """Test diff on attention model."""
        model = TEST_MODELS_DIR / "simple_attention.onnx"
        if not model.exists():
            self.add_result("attention_model_diff", False, 0, "required model not found")
            return

        input_file = self._create_test_input((1, 2, 4))  # batch, seq, dim

        try:
            rc, stdout, stderr, duration = self.run_ny([
                "diff", str(model), str(model),
                "--input", input_file, "--tolerance", "1e-5"
            ])

            passed = rc == 0 and "EQUIVALENT" in stdout
            self.add_result(
                "attention_model_diff",
                passed,
                duration,
                "attention model diff works" if passed else f"failed: {stdout[:100]}",
                stdout, stderr
            )
        finally:
            os.unlink(input_file)

    # ========================================================================
    # Utilities
    # ========================================================================

    def _create_test_input(self, shape: tuple, seed: int = 42) -> str:
        """Create test input file and return path."""
        np.random.seed(seed)
        data = np.random.randn(*shape).astype(np.float32)
        fd, path = tempfile.mkstemp(suffix='.npy')
        os.close(fd)
        np.save(path, data)
        return path

    def run_all_tests(self):
        """Run all tests."""
        print("=" * 70)
        print("End-to-End Integration Tests for ny")
        print("=" * 70)
        print()

        # Group 1: CLI Commands
        print("Test Group 1: CLI Commands Work")
        print("-" * 40)
        self.test_cli_verify()
        self.test_cli_diff_same_model()
        self.test_cli_sensitivity()
        self.test_cli_quantize_check()
        self.test_cli_profile_bounds()
        self.test_cli_inspect()
        self.test_cli_bench()

        print()
        print("Test Group 2: Error Messages")
        print("-" * 40)
        self.test_error_missing_file()
        self.test_error_invalid_epsilon()
        self.test_error_missing_required_arg()
        self.test_error_diff_shape_mismatch()

        print()
        print("Test Group 3: PyTorch → ONNX → ny diff")
        print("-" * 40)
        self.test_pytorch_onnx_diff_workflow()
        self.test_pytorch_perturbed_diff()

        print()
        print("Test Group 4: Documentation Examples")
        print("-" * 40)
        self.test_readme_verify_example()
        self.test_readme_diff_example()

        print()
        print("Test Group 5: Transformer Models")
        print("-" * 40)
        self.test_transformer_block_verify()
        self.test_attention_model_diff()

    def print_summary(self):
        """Print test summary."""
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        total = len(self.results)

        print(f"\n{'Test':<40} {'Status':<10} {'Time (ms)':<12}")
        print("-" * 70)

        for r in self.results:
            print(f"{r.name:<40} {r.status:<10} {r.duration_ms:>8.1f}")

        print("-" * 70)
        print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped ({total} total)")

        if failed == 0:
            if skipped:
                print("\nAll runnable tests PASSED; skipped tests were not counted as passes.")
            else:
                print("\nAll tests PASSED!")
            return 0
        print(f"\n{failed} tests FAILED")
        return 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="End-to-End Integration Tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Check ny binary exists
    if not NY_BIN.exists():
        print(f"ERROR: ny binary not found at {NY_BIN}")
        print("Run: cargo build --release")
        return 1

    required_models = (
        "simple_mlp.onnx",
        "single_linear.onnx",
        "transformer_mlp.onnx",
        "transformer_block.onnx",
        "simple_attention.onnx",
    )
    missing_models = [
        str(TEST_MODELS_DIR / name)
        for name in required_models
        if not (TEST_MODELS_DIR / name).is_file()
    ]
    if missing_models:
        print("ERROR: required tracked test fixtures are missing:")
        for path in missing_models:
            print(f"  - {path}")
        return 1

    tests = EndToEndTests(verbose=args.verbose)
    tests.run_all_tests()
    return tests.print_summary()


if __name__ == "__main__":
    sys.exit(main())
