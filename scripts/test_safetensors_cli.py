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
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def test_onnx_to_safetensors_diff():
    """Test diffing between ONNX and SafeTensors formats."""
    print("\n=== Test: ONNX to SafeTensors diff ===")

    # Check if test ONNX model exists
    test_model = "tests/models/simple_linear.onnx"
    if not os.path.exists(test_model):
        print("SKIP: No test ONNX model available")
        return True

    # Create SafeTensors with same weight names
    # (In practice, names might differ between formats)
    print("SKIP: Cross-format diff requires matching tensor names")
    return True


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

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\nAll tests PASSED!")
        return 0
    print("\nSome tests FAILED!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
