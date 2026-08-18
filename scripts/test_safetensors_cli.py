#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Test SafeTensors CLI integration.

This script tests the `ny weights` CLI commands with SafeTensors files.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure safetensors is available
try:
    import safetensors.torch as st
    import torch
except ImportError:
    print("SKIP: safetensors and torch required")
    sys.exit(0)

def run_ny(args: list[str]) -> tuple[int, str, str]:
    """Run ny CLI command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["cargo", "run", "-p", "ny-cli", "--"] + args,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.returncode, result.stdout, result.stderr


def test_weights_info():
    """Test ny weights info command with SafeTensors file."""
    print("\n=== Test: ny weights info ===")

    # Create test safetensors file
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            "layer1.weight": torch.randn(128, 64),
            "layer1.bias": torch.randn(128),
            "layer2.weight": torch.randn(32, 128),
            "layer2.bias": torch.randn(32),
        }
        st.save_file(tensors, f.name)

        # Test basic info
        code, stdout, stderr = run_ny(["weights", "info", "-f", f.name])
        if code != 0:
            print(f"FAIL: Exit code {code}")
            print(stderr)
            return False

        print(stdout)

        if "SafeTensors" not in stdout:
            print("FAIL: Format not shown")
            return False
        if "Tensors: 4" not in stdout:
            print("FAIL: Tensor count not shown")
            return False

        # Test detailed output
        code, stdout, stderr = run_ny(["weights", "info", "-f", f.name, "--detailed"])
        if code != 0:
            print(f"FAIL: Exit code {code}")
            return False

        if "layer1.weight" not in stdout:
            print("FAIL: Detailed tensor not shown")
            return False

        # Test JSON output
        code, stdout, stderr = run_ny(["weights", "info", "-f", f.name, "--json"])
        if code != 0:
            print(f"FAIL: Exit code {code}")
            return False

        data = json.loads(stdout)
        if data["tensor_count"] != 4:
            print("FAIL: JSON tensor count incorrect")
            return False

        os.unlink(f.name)

    print("PASS")
    return True


def test_weights_diff():
    """Test ny weights diff command."""
    print("\n=== Test: ny weights diff ===")

    # Create two test safetensors files
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f1:
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f2:
            # Same tensors
            base_tensors = {
                "layer.weight": torch.randn(64, 32),
                "layer.bias": torch.randn(64),
            }
            st.save_file(base_tensors, f1.name)
            st.save_file(base_tensors, f2.name)

            # Test same files
            code, stdout, stderr = run_ny([
                "weights", "diff",
                "--file-a", f1.name,
                "--file-b", f2.name
            ])
            if code != 0:
                print(f"FAIL: Exit code {code}")
                print(stderr)
                return False

            print(stdout)

            if "MATCH" not in stdout:
                print("FAIL: Same files should match")
                return False

            # Create file with different values
            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f3:
                different_tensors = {
                    "layer.weight": torch.randn(64, 32),  # Different random values
                    "layer.bias": torch.randn(64),
                }
                st.save_file(different_tensors, f3.name)

                # Test different files
                code, stdout, stderr = run_ny([
                    "weights", "diff",
                    "--file-a", f1.name,
                    "--file-b", f3.name
                ])
                if code != 0:
                    print(f"FAIL: Exit code {code}")
                    return False

                print(stdout)

                if "DIFFERS" not in stdout:
                    print("FAIL: Different files should differ")
                    return False

                os.unlink(f3.name)

            os.unlink(f1.name)
            os.unlink(f2.name)

    print("PASS")
    return True


def test_onnx_to_safetensors_diff() -> bool | None:
    """Document the currently unsupported cross-format fixture comparison."""
    print("\n=== Diagnostic: ONNX to SafeTensors diff ===")

    # Keep this diagnostic tied to the real tracked fixture.
    test_model = REPO_ROOT / "tests" / "models" / "single_linear.onnx"
    if not test_model.is_file():
        print(f"FAIL: Required tracked ONNX fixture is missing: {test_model}")
        return False

    # This is intentionally non-gating until a SafeTensors fixture with matching
    # tensor names is checked in. Do not claim comparator coverage.
    print("SKIP: No matching-name SafeTensors comparator fixture is available")
    return None


def main():
    print("SafeTensors CLI Integration Tests")
    print("=" * 50)

    results = []
    results.append(("weights info", test_weights_info()))
    results.append(("weights diff", test_weights_diff()))
    results.append(("cross-format diff", test_onnx_to_safetensors_diff()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    total = len(results)

    for name, result in results:
        status = "SKIP" if result is None else ("PASS" if result else "FAIL")
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped ({total} total)")

    if failed == 0:
        if skipped:
            print("\nAll runnable tests PASSED; skipped diagnostics were not counted as passes.")
        else:
            print("\nAll tests PASSED!")
        return 0
    print("\nSome tests FAILED!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
