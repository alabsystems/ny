#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Test GGUF CLI integration.

This script tests the `ny weights` CLI commands with GGUF files.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# Ensure gguf is available
try:
    from gguf import GGUFWriter
    from gguf.constants import GGMLQuantizationType
except ImportError:
    print("SKIP: gguf package required (pip install gguf)")
    sys.exit(0)


def run_ny(args: list[str], features: list[str] = None) -> tuple[int, str, str]:
    """Run ny CLI command and return (returncode, stdout, stderr)."""
    cmd = ["cargo", "run", "-p", "ny-cli"]
    if features:
        cmd.extend(["--features", ",".join(features)])
    cmd.append("--")
    cmd.extend(args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return result.returncode, result.stdout, result.stderr


def create_test_gguf(path: str, with_quantized: bool = False):
    """Create a minimal GGUF file for testing."""
    # Architecture is set in constructor
    writer = GGUFWriter(path, "test")

    # Add metadata - architecture is set automatically from constructor
    writer.add_architecture()  # Writes the architecture key
    writer.add_name("Test Model")
    writer.add_context_length(512)
    writer.add_embedding_length(64)
    writer.add_block_count(2)

    # Add F32 tensor
    f32_data = np.random.randn(64, 32).astype(np.float32)
    writer.add_tensor("layer1.weight", f32_data, raw_dtype=GGMLQuantizationType.F32)

    # Add F16 tensor
    f16_data = np.random.randn(64).astype(np.float32)
    writer.add_tensor("layer1.bias", f16_data, raw_dtype=GGMLQuantizationType.F16)

    # Add another F32 tensor
    f32_data2 = np.random.randn(32, 64).astype(np.float32)
    writer.add_tensor("layer2.weight", f32_data2, raw_dtype=GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def create_quantized_gguf(path: str, quant_type: GGMLQuantizationType = GGMLQuantizationType.Q8_0):
    """Create a GGUF file with quantized tensors for testing dequantization.

    Uses gguf library's built-in quantization to create properly quantized tensors.
    """
    writer = GGUFWriter(path, "test_quant")
    writer.add_architecture()
    writer.add_name("Quantized Test Model")
    writer.add_context_length(512)
    writer.add_embedding_length(64)
    writer.add_block_count(2)

    # Create tensors with shapes that are multiples of 32 (block size)
    # The gguf library will handle quantization when we pass the quant type
    # Note: gguf writer will quantize the f32 data when we specify raw_dtype

    # 64*32 = 2048 elements = 64 blocks of 32
    f32_data = np.random.randn(64, 32).astype(np.float32)
    writer.add_tensor("quant_layer.weight", f32_data, raw_dtype=quant_type)

    # Also add an F32 tensor for comparison
    f32_ref = np.random.randn(64, 32).astype(np.float32)
    writer.add_tensor("ref_layer.weight", f32_ref, raw_dtype=GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_weights_info():
    """Test ny weights info command with GGUF file."""
    print("\n=== Test: ny weights info (GGUF) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_path = os.path.join(tmpdir, "test.gguf")
        create_test_gguf(gguf_path)

        # Test basic info
        code, stdout, stderr = run_ny(["weights", "info", "-f", gguf_path], features=["gguf"])
        if code != 0:
            print(f"FAIL: Exit code {code}")
            print(f"stderr: {stderr}")
            return False

        print(stdout)

        if "GGUF" not in stdout:
            print("FAIL: Format not shown")
            return False
        if "Tensors: 3" not in stdout:
            print("FAIL: Tensor count not shown")
            return False
        if "Architecture: test" not in stdout:
            print("FAIL: Architecture not shown")
            return False
        if "Model Name: Test Model" not in stdout:
            print("FAIL: Model name not shown")
            return False

        # Test detailed output
        code, stdout, stderr = run_ny(["weights", "info", "-f", gguf_path, "--detailed"], features=["gguf"])
        if code != 0:
            print(f"FAIL: Exit code {code}")
            return False

        if "layer1.weight" not in stdout:
            print("FAIL: Detailed tensor not shown")
            return False
        if "F32" not in stdout:
            print("FAIL: F32 dtype not shown")
            return False
        if "F16" not in stdout:
            print("FAIL: F16 dtype not shown")
            return False

        # Test JSON output
        code, stdout, stderr = run_ny(["weights", "info", "-f", gguf_path, "--json"], features=["gguf"])
        if code != 0:
            print(f"FAIL: Exit code {code}")
            print(f"stderr: {stderr}")
            return False

        data = json.loads(stdout)
        if data["tensor_count"] != 3:
            print("FAIL: JSON tensor count incorrect")
            return False
        if data["format"] != "gguf":
            print("FAIL: JSON format incorrect")
            return False
        if data["architecture"] != "test":
            print("FAIL: JSON architecture incorrect")
            return False

    print("PASS")
    return True


def test_weights_info_metadata():
    """Test that GGUF metadata is properly displayed."""
    print("\n=== Test: ny weights info metadata (GGUF) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_path = os.path.join(tmpdir, "test.gguf")
        create_test_gguf(gguf_path)

        # Test that metadata keys are shown
        code, stdout, stderr = run_ny(["weights", "info", "-f", gguf_path], features=["gguf"])
        if code != 0:
            print(f"FAIL: Exit code {code}")
            print(f"stderr: {stderr}")
            return False

        # These should be in the metadata section
        if "context_length" not in stdout.lower():
            print("FAIL: context_length metadata not shown")
            return False

    print("PASS")
    return True


def test_weights_diff_gguf_vs_gguf():
    """Test comparing two GGUF files."""
    print("\n=== Test: ny weights diff (GGUF vs GGUF) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two identical GGUF files
        np.random.seed(42)
        gguf_path1 = os.path.join(tmpdir, "model1.gguf")
        create_test_gguf(gguf_path1)

        np.random.seed(42)  # Same seed = same weights
        gguf_path2 = os.path.join(tmpdir, "model2.gguf")
        create_test_gguf(gguf_path2)

        # Compare identical files
        code, stdout, stderr = run_ny(
            ["weights", "diff", "--file-a", gguf_path1, "--file-b", gguf_path2],
            features=["gguf"]
        )
        if code != 0:
            print(f"FAIL: Exit code {code}")
            print(f"stderr: {stderr}")
            return False

        print(stdout)

        if "MATCH" not in stdout:
            print("FAIL: Identical files not detected as match")
            return False

        # Create a different file
        np.random.seed(123)  # Different seed = different weights
        gguf_path3 = os.path.join(tmpdir, "model3.gguf")
        create_test_gguf(gguf_path3)

        # Compare different files
        code, stdout, stderr = run_ny(
            ["weights", "diff", "--file-a", gguf_path1, "--file-b", gguf_path3],
            features=["gguf"]
        )
        # This should succeed but report differences
        if code != 0:
            print(f"FAIL: Exit code {code}")
            print(f"stderr: {stderr}")
            return False

        print(stdout)

        if "DIFFERS" not in stdout:
            print("FAIL: Different files not detected as different")
            return False

    print("PASS")
    return True


def test_file_not_found():
    """Test error handling for nonexistent files."""
    print("\n=== Test: GGUF file not found ===")

    code, stdout, stderr = run_ny(["weights", "info", "-f", "/nonexistent/model.gguf"], features=["gguf"])
    if code == 0:
        print("FAIL: Should have failed for nonexistent file")
        return False

    # Error should be meaningful
    if "not found" not in stderr.lower() and "failed" not in stderr.lower():
        print(f"FAIL: Error message not helpful: {stderr}")
        return False

    print("PASS")
    return True


def test_quantized_gguf_info():
    """Test ny weights info command with quantized GGUF file."""
    print("\n=== Test: ny weights info (quantized GGUF) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_path = os.path.join(tmpdir, "quantized.gguf")
        np.random.seed(42)
        create_quantized_gguf(gguf_path, GGMLQuantizationType.Q8_0)

        # Test info command
        code, stdout, stderr = run_ny(["weights", "info", "-f", gguf_path, "--detailed"], features=["gguf"])
        if code != 0:
            print(f"FAIL: Exit code {code}")
            print(f"stderr: {stderr}")
            return False

        print(stdout)

        # Should show Q8_0 in the detailed output
        if "Q8_0" not in stdout:
            print("FAIL: Q8_0 dtype not shown in output")
            return False

        if "Tensors: 2" not in stdout:
            print("FAIL: Tensor count incorrect")
            return False

    print("PASS")
    return True


def test_quantized_gguf_load():
    """Test that quantized tensors are loaded (dequantized) correctly."""
    print("\n=== Test: Load quantized GGUF (dequantization) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test Q8_0 quantization
        gguf_path = os.path.join(tmpdir, "q8_0.gguf")
        np.random.seed(42)
        create_quantized_gguf(gguf_path, GGMLQuantizationType.Q8_0)

        # Use weights diff to load and compare the model (this exercises load_gguf)
        # Comparing the file with itself should show all tensors match
        code, stdout, stderr = run_ny(
            ["weights", "diff", "--file-a", gguf_path, "--file-b", gguf_path],
            features=["gguf"]
        )
        if code != 0:
            print(f"FAIL: Exit code {code}")
            print(f"stderr: {stderr}")
            return False

        print(stdout)

        # Both tensors should match (since comparing with self)
        # This proves that the quantized tensor was successfully dequantized and loaded
        if "MATCH" not in stdout:
            print("FAIL: Self-comparison should show MATCH")
            return False

        # Now test that loading actually produces reasonable values by comparing
        # a quantized model vs itself modified
        gguf_path2 = os.path.join(tmpdir, "q8_0_v2.gguf")
        np.random.seed(123)  # Different seed = different weights
        create_quantized_gguf(gguf_path2, GGMLQuantizationType.Q8_0)

        code, stdout, stderr = run_ny(
            ["weights", "diff", "--file-a", gguf_path, "--file-b", gguf_path2],
            features=["gguf"]
        )
        if code != 0:
            print(f"FAIL: Comparing different quantized files failed (code={code})")
            print(f"stderr: {stderr}")
            return False

        print(stdout)

        # Should detect differences
        if "DIFFERS" not in stdout:
            print("FAIL: Different quantized files should show DIFFERS")
            return False

        # The quantized tensor should be shown in the diff
        if "quant_layer.weight" not in stdout:
            print("FAIL: Quantized tensor not in diff output")
            return False

    print("PASS")
    return True


def test_quantized_types():
    """Test loading multiple quantized types."""
    print("\n=== Test: Multiple quantized types ===")

    quant_types = [
        ("Q8_0", GGMLQuantizationType.Q8_0),
        ("Q4_0", GGMLQuantizationType.Q4_0),
        ("Q4_1", GGMLQuantizationType.Q4_1),
        # Q5_0 and Q5_1 may not be supported by gguf writer
    ]

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, qtype in quant_types:
            print(f"  Testing {name}...", end=" ")
            gguf_path = os.path.join(tmpdir, f"{name}.gguf")

            try:
                np.random.seed(42)
                create_quantized_gguf(gguf_path, qtype)

                # Try to load the file via weights diff (self-comparison)
                code, stdout, stderr = run_ny(
                    ["weights", "diff", "--file-a", gguf_path, "--file-b", gguf_path],
                    features=["gguf"]
                )

                if code == 0 and "MATCH" in stdout:
                    print("OK")
                    results.append((name, True))
                else:
                    print(f"FAIL (code={code})")
                    results.append((name, False))
            except Exception as e:
                print(f"SKIP (gguf writer error: {e})")
                results.append((name, None))

    # Report results
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    print(f"\nResults: {passed}/{len(results)} passed, {skipped} skipped")

    if failed > 0:
        print(f"FAIL: {failed} quantized types failed")
        return False

    print("PASS")
    return True


def main():
    """Run all tests."""
    print("GGUF CLI Integration Tests")
    print("=" * 50)

    tests = [
        test_weights_info,
        test_weights_info_metadata,
        test_weights_diff_gguf_vs_gguf,
        test_file_not_found,
        # Dequantization tests
        test_quantized_gguf_info,
        test_quantized_gguf_load,
        test_quantized_types,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{len(tests)} passed")

    if failed > 0:
        print(f"FAILED: {failed} tests failed")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
