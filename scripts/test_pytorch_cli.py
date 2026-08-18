#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Test PyTorch format support in ny CLI."""

import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


def create_test_pytorch_file(path: Path):
    """Create a test .pt file with some tensors."""
    state_dict = {
        "layer1.weight": torch.randn(128, 64),
        "layer1.bias": torch.randn(128),
        "layer2.weight": torch.randn(32, 128),
        "layer2.bias": torch.randn(32),
    }
    torch.save(state_dict, path)
    return state_dict

def run_ny(*args) -> subprocess.CompletedProcess:
    """Run ny CLI command."""
    cmd = ["cargo", "run", "-p", "ny-cli", "--"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    return result

def test_pytorch_weights_info():
    """Test ny weights info on PyTorch file."""
    print("Testing: ny weights info on PyTorch file...")

    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = Path(tmpdir) / "test_model.pt"
        create_test_pytorch_file(pt_path)

        # Test basic info
        result = run_ny("weights", "info", "-f", str(pt_path))
        if result.returncode != 0:
            print(f"FAIL: weights info failed: {result.stderr}")
            return False

        output = result.stdout
        if "Format: PyTorch" not in output:
            print("FAIL: Expected 'Format: PyTorch' in output")
            return False

        if "Tensors: 4" not in output:
            print("FAIL: Expected 4 tensors")
            return False

        print("  - Basic info: PASS")

        # Test detailed info
        result = run_ny("weights", "info", "-f", str(pt_path), "--detailed")
        if result.returncode != 0:
            print(f"FAIL: weights info --detailed failed: {result.stderr}")
            return False

        output = result.stdout
        if "layer1.weight" not in output:
            print("FAIL: Expected tensor details in output")
            return False

        print("  - Detailed info: PASS")

        # Test JSON output
        result = run_ny("weights", "info", "-f", str(pt_path), "--json")
        if result.returncode != 0:
            print(f"FAIL: weights info --json failed: {result.stderr}")
            return False

        import json
        try:
            data = json.loads(result.stdout)
            assert data["format"] == "pytorch"
            assert data["tensor_count"] == 4
        except (json.JSONDecodeError, AssertionError) as e:
            print(f"FAIL: Invalid JSON output: {e}")
            return False

        print("  - JSON output: PASS")

    return True

def test_pytorch_weights_diff():
    """Test ny weights diff with PyTorch files."""
    print("Testing: ny weights diff on PyTorch files...")

    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path_a = Path(tmpdir) / "model_a.pt"
        pt_path_b = Path(tmpdir) / "model_b.pt"

        # Create two identical models
        state_dict_a = {
            "layer1.weight": torch.ones(10, 10),
            "layer1.bias": torch.zeros(10),
        }
        torch.save(state_dict_a, pt_path_a)

        # Create slightly different model
        state_dict_b = {
            "layer1.weight": torch.ones(10, 10) + 0.001,  # Small diff
            "layer1.bias": torch.zeros(10),
        }
        torch.save(state_dict_b, pt_path_b)

        # Compare
        result = run_ny("weights", "diff", "--file-a", str(pt_path_a), "--file-b", str(pt_path_b))
        if result.returncode != 0:
            print(f"FAIL: weights diff failed: {result.stderr}")
            return False

        output = result.stdout
        # Should detect the difference
        if "DIFFERS" not in output or "layer1.weight" not in output:
            print("FAIL: Expected a DIFFERS result naming layer1.weight")
            return False

        print("  - PyTorch to PyTorch diff: PASS")

    return True

def test_pytorch_safetensors_diff() -> bool | None:
    """Test cross-format comparison: PyTorch vs SafeTensors."""
    print("Testing: ny weights diff PyTorch vs SafeTensors...")

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("  - SKIP: safetensors not installed")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = Path(tmpdir) / "model.pt"
        st_path = Path(tmpdir) / "model.safetensors"

        # Create identical models in both formats
        state_dict = {
            "layer1.weight": torch.ones(10, 10),
            "layer1.bias": torch.zeros(10),
        }
        torch.save(state_dict, pt_path)
        save_file(state_dict, st_path)

        # Compare across formats
        result = run_ny("weights", "diff", "--file-a", str(pt_path), "--file-b", str(st_path))
        if result.returncode != 0:
            print(f"FAIL: cross-format diff failed: {result.stderr}")
            return False

        output = result.stdout
        if "MATCH" not in output:
            print("FAIL: Identical cross-format tensors were not reported as MATCH")
            return False

        print("  - PyTorch to SafeTensors diff: PASS")

    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("PyTorch Format Support Tests")
    print("=" * 60)
    print()

    if torch is None:
        print("SKIP: torch is required for PyTorch format integration tests")
        return 0

    tests = [
        test_pytorch_weights_info,
        test_pytorch_weights_diff,
        test_pytorch_safetensors_diff,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            result = test()
            if result is True:
                passed += 1
            elif result is False:
                failed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} raised exception: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
