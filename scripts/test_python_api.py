#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""
Python API Unit Tests for ny

This script tests the Python bindings directly (not via CLI).
Validates that all exported Python functions work correctly.

Run: python scripts/test_python_api.py
"""

import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path for ny import
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import ny
    HAS_NY = True
except ImportError as e:
    HAS_NY = False
    NY_IMPORT_ERROR = str(e)

TEST_MODELS_DIR = REPO_ROOT / "tests" / "models"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: str
    duration_ms: float
    message: str

    @property
    def passed(self) -> bool:
        """Backward-compatible boolean view; skipped results are not passes."""
        return self.status == "PASS"


class PythonAPITests:
    """Unit tests for the ny Python API."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[TestResult] = []

    def add_result(
        self,
        name: str,
        passed: bool,
        duration_ms: float,
        message: str,
        *,
        skipped: bool = False,
    ):
        """Record a test result."""
        status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        result = TestResult(name, status, duration_ms, message)
        self.results.append(result)

        if self.verbose:
            print(f"  [{result.status}] {name}: {message} ({duration_ms:.1f}ms)")

    # ========================================================================
    # Test Group 1: ny.diff()
    # ========================================================================

    def test_diff_same_model(self):
        """Test diff of a model against itself (should be equivalent)."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.diff(model, model)
            passed = (
                result.is_equivalent and
                result.max_divergence == 0.0 and
                result.first_bad_layer is None and
                len(result.layers) > 0
            )
            message = f"is_equivalent={result.is_equivalent}, max_div={result.max_divergence:.2e}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("diff_same_model", passed, duration, message)

    def test_diff_with_tolerance(self):
        """Test diff with custom tolerance."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.diff(model, model, tolerance=1e-3)
            passed = result.is_equivalent and result.tolerance > 1e-4
            message = f"tolerance={result.tolerance:.2e}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("diff_with_tolerance", passed, duration, message)

    def test_diff_layer_count(self):
        """Test that diff returns correct number of layers."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.diff(model, model)
            # simple_mlp has 3 layers (fc1, relu, fc2)
            passed = len(result.layers) == 3
            message = f"layers={len(result.layers)}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("diff_layer_count", passed, duration, message)

    def test_diff_layer_comparison_attrs(self):
        """Test LayerComparison attributes."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.diff(model, model)
            layer = result.layers[0]
            # Check all expected attributes exist
            passed = (
                hasattr(layer, 'name') and
                hasattr(layer, 'max_diff') and
                hasattr(layer, 'mean_diff') and
                hasattr(layer, 'exceeds_tolerance') and
                hasattr(layer, 'shape_a') and
                hasattr(layer, 'shape_b') and
                isinstance(layer.name, str) and
                isinstance(layer.max_diff, float) and
                isinstance(layer.shape_a, list)
            )
            message = f"layer.name={layer.name}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("diff_layer_attrs", passed, duration, message)

    def test_diff_with_input(self):
        """Test diff with custom input."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            # simple_mlp expects input shape [1, 2]
            input_data = np.array([[0.5, -0.5]], dtype=np.float32)
            result = ny.diff(model, model, input=input_data)
            passed = result.is_equivalent and result.max_divergence == 0.0
            message = f"is_equivalent={result.is_equivalent} with custom input"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("diff_with_input", passed, duration, message)

    # ========================================================================
    # Test Group 2: ny.sensitivity_analysis()
    # ========================================================================

    def test_sensitivity_basic(self):
        """Test basic sensitivity analysis."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.sensitivity_analysis(model)
            passed = (
                len(result.layers) > 0 and
                hasattr(result, 'max_sensitivity_layer_name') and
                hasattr(result, 'max_sensitivity')
            )
            message = f"layers={len(result.layers)}, max_sens={result.max_sensitivity:.2f}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("sensitivity_basic", passed, duration, message)

    def test_sensitivity_with_epsilon(self):
        """Test sensitivity analysis with custom epsilon."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.sensitivity_analysis(model, epsilon=0.001)
            passed = len(result.layers) > 0
            message = f"layers={len(result.layers)} with epsilon=0.001"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("sensitivity_with_epsilon", passed, duration, message)

    def test_sensitivity_layer_attrs(self):
        """Test LayerSensitivity attributes."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.sensitivity_analysis(model)
            layer = result.layers[0]
            passed = (
                hasattr(layer, 'name') and
                hasattr(layer, 'sensitivity') and
                hasattr(layer, 'input_width') and
                hasattr(layer, 'output_width') and
                isinstance(layer.sensitivity, float)
            )
            message = f"layer={layer.name}, sens={layer.sensitivity:.2f}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("sensitivity_layer_attrs", passed, duration, message)

    # ========================================================================
    # Test Group 3: ny.quantize_check()
    # ========================================================================

    def test_quantize_check_basic(self):
        """Test basic quantization checking."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.quantize_check(model)
            passed = (
                len(result.layers) > 0 and
                hasattr(result, 'float16_safe') and
                hasattr(result, 'int8_safe')
            )
            message = f"f16_safe={result.float16_safe}, i8_safe={result.int8_safe}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("quantize_check_basic", passed, duration, message)

    def test_quantize_check_with_epsilon(self):
        """Test quantize_check with custom epsilon."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.quantize_check(model, epsilon=0.1)
            passed = len(result.layers) > 0
            message = f"layers={len(result.layers)} with epsilon=0.1"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("quantize_check_with_epsilon", passed, duration, message)

    def test_quantize_layer_attrs(self):
        """Test LayerQuantization attributes."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.quantize_check(model)
            layer = result.layers[0]
            passed = (
                hasattr(layer, 'name') and
                hasattr(layer, 'float16_safety') and
                hasattr(layer, 'int8_safety') and
                hasattr(layer, 'max_abs')  # Not max_abs_value
            )
            message = f"layer={layer.name}, f16={layer.float16_safety}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("quantize_layer_attrs", passed, duration, message)

    # ========================================================================
    # Test Group 4: ny.profile_bounds()
    # ========================================================================

    def test_profile_bounds_basic(self):
        """Test basic bound profiling."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.profile_bounds(model, epsilon=0.01)
            passed = (
                len(result.layers) > 0 and
                hasattr(result, 'difficulty_score') and
                hasattr(result, 'max_growth_ratio')
            )
            message = f"difficulty={result.difficulty_score:.1f}, growth={result.max_growth_ratio:.2f}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("profile_bounds_basic", passed, duration, message)

    def test_profile_layer_attrs(self):
        """Test LayerProfile attributes."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.profile_bounds(model, epsilon=0.01)
            layer = result.layers[0]
            passed = (
                hasattr(layer, 'name') and
                hasattr(layer, 'output_width') and
                hasattr(layer, 'growth_ratio') and
                hasattr(layer, 'status')
            )
            message = f"layer={layer.name}, width={layer.output_width:.2e}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("profile_layer_attrs", passed, duration, message)

    # ========================================================================
    # Test Group 5: ny.load_model_info()
    # ========================================================================

    def test_load_model_info_basic(self):
        """Test loading model info."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            info = ny.load_model_info(model)
            passed = isinstance(info, ny.ModelInfo) and isinstance(info.inputs, list) and isinstance(info.outputs, list) and isinstance(info.layer_count, int)
            message = f"layer_count={info.layer_count}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("load_model_info_basic", passed, duration, message)

    def test_load_model_info_shapes(self):
        """Test model info contains shape information."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            info = ny.load_model_info(model)
            inputs = info.inputs
            passed = len(inputs) > 0 and isinstance(inputs[0], ny.TensorSpec) and isinstance(inputs[0].shape, list) and isinstance(inputs[0].dtype, str)
            message = f"inputs={len(inputs)}, dtype={inputs[0].dtype if inputs else 'n/a'}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("load_model_info_shapes", passed, duration, message)

    # ========================================================================
    # Test Group 6: ny.load_npy()
    # ========================================================================

    def test_load_npy(self):
        """Test loading numpy arrays."""
        start = time.time()

        try:
            # Create a temp npy file
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                temp_path = f.name
                test_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
                np.save(f, test_data)

            loaded = ny.load_npy(temp_path)
            passed = (
                isinstance(loaded, np.ndarray) and
                loaded.shape == (1, 3) and
                np.allclose(loaded, test_data)
            )
            message = f"shape={loaded.shape}"

            os.unlink(temp_path)
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("load_npy", passed, duration, message)

    # ========================================================================
    # Test Group 7: Error Handling
    # ========================================================================

    def test_diff_missing_file(self):
        """Test diff with non-existent file raises error."""
        start = time.time()

        try:
            ny.diff("nonexistent.onnx", "nonexistent.onnx")
            passed = False  # Should have raised
            message = "No exception raised"
        except Exception as e:
            # Should raise ValueError or similar
            passed = True
            message = f"Correctly raised: {type(e).__name__}"

        duration = (time.time() - start) * 1000
        self.add_result("diff_missing_file", passed, duration, message)

    def test_sensitivity_missing_file(self):
        """Test sensitivity with non-existent file raises error."""
        start = time.time()

        try:
            ny.sensitivity_analysis("nonexistent.onnx")
            passed = False
            message = "No exception raised"
        except Exception as e:
            passed = True
            message = f"Correctly raised: {type(e).__name__}"

        duration = (time.time() - start) * 1000
        self.add_result("sensitivity_missing_file", passed, duration, message)

    # ========================================================================
    # Test Group 8: ny.verify()
    # ========================================================================

    def test_verify_basic(self):
        """Test basic verification: without output_bounds there is no property,
        so the status must be Unknown (bounds computed, nothing verified)."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.verify(model)
            passed = (
                hasattr(result, 'status') and
                hasattr(result, 'is_verified') and
                hasattr(result, 'output_bounds') and
                hasattr(result, 'method') and
                hasattr(result, 'epsilon') and
                str(result.status) == "UNKNOWN" and
                not result.is_verified and
                result.reason is not None and
                "no output specification" in result.reason
            )
            message = f"status={result.status}, method={result.method}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("verify_basic", passed, duration, message)

    def test_verify_with_method(self):
        """Test verification with different methods."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            # Test IBP (fastest method)
            result = ny.verify(model, method="ibp")
            passed = result.method == "ibp"
            message = f"method={result.method}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("verify_with_method", passed, duration, message)

    def test_verify_with_epsilon(self):
        """Test verification with custom epsilon."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.verify(model, epsilon=0.001)
            passed = abs(result.epsilon - 0.001) < 1e-6
            message = f"epsilon={result.epsilon:.4f}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("verify_with_epsilon", passed, duration, message)

    def test_verify_output_bounds(self):
        """Test that verify returns output bounds."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.verify(model)
            # Check output bounds exist and have expected attributes
            passed = (
                result.output_bounds is not None and
                len(result.output_bounds) > 0 and
                hasattr(result.output_bounds[0], 'lower') and
                hasattr(result.output_bounds[0], 'upper') and
                hasattr(result.output_bounds[0], 'width')
            )
            if passed:
                message = f"bounds={len(result.output_bounds)}, width={result.output_bounds[0].width:.2e}"
            else:
                message = "Missing output_bounds or attributes"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("verify_output_bounds", passed, duration, message)

    def test_verify_result_attrs(self):
        """Test VerifyResult attributes."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.verify(model)
            # Test max_output_width method
            max_width = result.max_output_width()
            # Test summary method
            summary = result.summary()
            passed = (
                max_width is not None and
                isinstance(max_width, float) and
                isinstance(summary, str) and
                "Verification Result" in summary
            )
            message = f"max_width={max_width:.2e}" if max_width else "No max_width"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("verify_result_attrs", passed, duration, message)

    def test_verify_output_bounds_verified(self):
        """Test that a satisfied output_bounds property is Verified."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            bounds_only = ny.verify(model, method="ibp")
            n = len(bounds_only.output_bounds)
            # Generous finite requirement: the certified bounds land inside it.
            result = ny.verify(model, method="ibp", output_bounds=[(-1e6, 1e6)] * n)
            passed = result.is_verified and str(result.status) == "VERIFIED"
            message = f"status={result.status} with {n} finite requirements"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("verify_output_bounds_verified", passed, duration, message)

    def test_verify_output_bounds_unsatisfied(self):
        """Test that an unsatisfiable output_bounds property is never Verified."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            bounds_only = ny.verify(model, method="ibp")
            n = len(bounds_only.output_bounds)
            # Requirement box far outside any certified bound for this model.
            result = ny.verify(model, method="ibp", output_bounds=[(1e9, 2e9)] * n)
            passed = (not result.is_verified) and str(result.status) == "UNKNOWN"
            message = f"status={result.status} for impossible requirement"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("verify_output_bounds_unsatisfied", passed, duration, message)

    def test_verify_all_infinite_output_bounds(self):
        """Test that all-(-inf, +inf) output_bounds raises ValueError.

        Such a requirement is satisfied by every network, so accepting it
        would report a vacuous Verified; the API rejects it instead.
        """
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            bounds_only = ny.verify(model, method="ibp")
            n = len(bounds_only.output_bounds)
            inf = float("inf")
            try:
                ny.verify(model, method="ibp", output_bounds=[(-inf, inf)] * n)
                passed = False
                message = "No exception raised"
            except ValueError as e:
                passed = "unconstrained" in str(e)
                message = (
                    "Correctly raised ValueError"
                    if passed else f"Wrong message: {e}"
                )
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("verify_all_infinite_output_bounds", passed, duration, message)

    def test_verify_invalid_method(self):
        """Test verify with invalid method raises error."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            ny.verify(model, method="invalid_method")
            passed = False
            message = "No exception raised"
        except Exception as e:
            passed = "Unknown method" in str(e)
            message = f"Correctly raised: {type(e).__name__}"

        duration = (time.time() - start) * 1000
        self.add_result("verify_invalid_method", passed, duration, message)

    def test_verify_missing_file(self):
        """Test verify with non-existent file raises error."""
        start = time.time()

        try:
            ny.verify("nonexistent.onnx")
            passed = False
            message = "No exception raised"
        except Exception as e:
            passed = True
            message = f"Correctly raised: {type(e).__name__}"

        duration = (time.time() - start) * 1000
        self.add_result("verify_missing_file", passed, duration, message)

    # ========================================================================
    # Test Group 9: ny.compare()
    # ========================================================================

    def test_compare_same_model(self):
        """Test comparing a model against itself (should be equivalent)."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.compare(model, model)
            passed = (
                result.is_equivalent and
                result.max_lower_diff == 0.0 and
                result.max_upper_diff == 0.0 and
                len(result.violations) == 0
            )
            message = f"is_equivalent={result.is_equivalent}, max_diff={result.max_lower_diff:.2e}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("compare_same_model", passed, duration, message)

    def test_compare_with_tolerance(self):
        """Test compare with custom tolerance."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.compare(model, model, tolerance=0.01)
            passed = result.is_equivalent and abs(result.tolerance - 0.01) < 1e-6
            message = f"tolerance={result.tolerance:.4f}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("compare_with_tolerance", passed, duration, message)

    def test_compare_with_method(self):
        """Test compare with different methods."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.compare(model, model, method="ibp")
            passed = result.method == "ibp" and result.is_equivalent
            message = f"method={result.method}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("compare_with_method", passed, duration, message)

    def test_compare_result_attrs(self):
        """Test CompareResult attributes."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.compare(model, model)
            # Check all expected attributes
            passed = (
                hasattr(result, 'is_equivalent') and
                hasattr(result, 'max_lower_diff') and
                hasattr(result, 'max_upper_diff') and
                hasattr(result, 'tolerance') and
                hasattr(result, 'overlap_pct') and
                hasattr(result, 'ref_max_width') and
                hasattr(result, 'target_max_width') and
                hasattr(result, 'method') and
                hasattr(result, 'epsilon') and
                hasattr(result, 'violations')
            )
            # Test summary method
            if passed:
                summary = result.summary()
                passed = isinstance(summary, str) and "Model Comparison Result" in summary
            message = f"overlap={result.overlap_pct:.1f}%" if passed else "Missing attributes"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("compare_result_attrs", passed, duration, message)

    def test_compare_invalid_method(self):
        """Test compare with invalid method raises error."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            ny.compare(model, model, method="invalid")
            passed = False
            message = "No exception raised"
        except Exception as e:
            passed = "Unknown method" in str(e)
            message = f"Correctly raised: {type(e).__name__}"

        duration = (time.time() - start) * 1000
        self.add_result("compare_invalid_method", passed, duration, message)

    # ========================================================================
    # Test Group 10: ny.weights_info()
    # ========================================================================

    def test_weights_info_onnx(self):
        """Test weights_info on ONNX model."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            info = ny.weights_info(model)
            passed = (
                hasattr(info, 'format') and
                hasattr(info, 'tensor_count') and
                hasattr(info, 'total_params') and
                hasattr(info, 'tensors') and
                info.format == "ONNX" and
                info.tensor_count > 0
            )
            message = f"format={info.format}, tensors={info.tensor_count}, params={info.total_params}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_info_onnx", passed, duration, message)

    def test_weights_info_tensor_attrs(self):
        """Test TensorInfo attributes from weights_info."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            info = ny.weights_info(model)
            tensor = info.tensors[0]
            passed = (
                hasattr(tensor, 'name') and
                hasattr(tensor, 'shape') and
                hasattr(tensor, 'elements') and
                isinstance(tensor.name, str) and
                isinstance(tensor.shape, list) and
                isinstance(tensor.elements, int)
            )
            message = f"tensor={tensor.name}, shape={tensor.shape}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_info_tensor_attrs", passed, duration, message)

    def test_weights_info_summary(self):
        """Test WeightsInfo summary method."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            info = ny.weights_info(model)
            summary = info.summary()
            passed = (
                isinstance(summary, str) and
                "Weights Info" in summary and
                "Parameters" in summary
            )
            message = "summary generated"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_info_summary", passed, duration, message)

    def test_weights_info_unsupported(self):
        """Test weights_info with unsupported format raises error."""
        start = time.time()

        try:
            ny.weights_info("model.unsupported")
            passed = False
            message = "No exception raised"
        except Exception as e:
            passed = "Unsupported format" in str(e)
            message = f"Correctly raised: {type(e).__name__}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_info_unsupported", passed, duration, message)

    def test_weights_info_pytorch(self):
        """Test weights_info on PyTorch model file."""
        start = time.time()
        # Use a known PyTorch model from research repos
        pytorch_model = REPO_ROOT / "research" / "repos" / "auto_LiRPA" / "examples" / "vision" / "pretrained" / "mnist_fc_3layer.pth"

        if not pytorch_model.exists():
            # Skip test if model not available
            duration = (time.time() - start) * 1000
            self.add_result(
                "weights_info_pytorch",
                False,
                duration,
                "optional PyTorch model not found",
                skipped=True,
            )
            return

        try:
            result = ny.weights_info(str(pytorch_model))
            passed = (
                result.format == "PyTorch" and
                result.tensor_count > 0 and
                result.total_params > 0
            )
            message = f"format={result.format}, tensors={result.tensor_count}, params={result.total_params:,}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_info_pytorch", passed, duration, message)

    def test_weights_diff_pytorch(self):
        """Test weights_diff on PyTorch model file."""
        start = time.time()
        # Use a known PyTorch model from research repos
        pytorch_model = REPO_ROOT / "research" / "repos" / "auto_LiRPA" / "examples" / "vision" / "pretrained" / "mnist_fc_3layer.pth"

        if not pytorch_model.exists():
            # Skip test if model not available
            duration = (time.time() - start) * 1000
            self.add_result(
                "weights_diff_pytorch",
                False,
                duration,
                "optional PyTorch model not found",
                skipped=True,
            )
            return

        try:
            result = ny.weights_diff(str(pytorch_model), str(pytorch_model))
            passed = result.is_match and result.max_diff == 0.0
            message = f"is_match={result.is_match}, max_diff={result.max_diff:.2e}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_diff_pytorch", passed, duration, message)

    # ========================================================================
    # Test Group 11: ny.weights_diff()
    # ========================================================================

    def test_weights_diff_same_model(self):
        """Test weights_diff on same model (should match)."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.weights_diff(model, model)
            passed = (
                result.is_match and
                result.max_diff == 0.0 and
                result.differing_count == 0
            )
            message = f"is_match={result.is_match}, max_diff={result.max_diff:.2e}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_diff_same_model", passed, duration, message)

    def test_weights_diff_with_tolerance(self):
        """Test weights_diff with custom tolerance."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.weights_diff(model, model, tolerance=1e-3)
            passed = result.is_match and abs(result.tolerance - 1e-3) < 1e-9
            message = f"tolerance={result.tolerance:.2e}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_diff_tolerance", passed, duration, message)

    def test_weights_diff_result_attrs(self):
        """Test WeightsDiffResult attributes."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.weights_diff(model, model)
            passed = hasattr(result, 'is_match') and hasattr(result, 'max_diff') and hasattr(result, 'tolerance') and hasattr(result, 'differing_count') and hasattr(result, 'total_tensors_a') and hasattr(result, 'total_tensors_b') and hasattr(result, 'comparisons') and (len(result.comparisons) == 0 or isinstance(result.comparisons[0].status, ny.TensorComparisonStatus))
            message = f"tensors_a={result.total_tensors_a}, tensors_b={result.total_tensors_b}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_diff_result_attrs", passed, duration, message)

    def test_weights_diff_summary(self):
        """Test WeightsDiffResult summary method."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.weights_diff(model, model)
            summary = result.summary()
            passed = (
                isinstance(summary, str) and
                "Weights Diff Result" in summary and
                "MATCH" in summary
            )
            message = "summary generated"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_diff_summary", passed, duration, message)

    def test_weights_diff_matching_tensors(self):
        """Test matching_tensors and differing_tensors methods."""
        start = time.time()
        model = str(TEST_MODELS_DIR / "simple_mlp.onnx")

        try:
            result = ny.weights_diff(model, model)
            matching = result.matching_tensors()
            differing = result.differing_tensors()
            # All should match, none should differ
            passed = (
                len(matching) == len(result.comparisons) and
                len(differing) == 0
            )
            message = f"matching={len(matching)}, differing={len(differing)}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("weights_diff_methods", passed, duration, message)

    # ========================================================================
    # Test Group 12: ny.bench()
    # ========================================================================

    def test_bench_default(self):
        """Test bench with default arguments (layer benchmark)."""
        start = time.time()

        try:
            result = ny.bench()
            passed = (
                hasattr(result, 'benchmark_type') and
                hasattr(result, 'valid_type') and
                hasattr(result, 'dimensions') and
                hasattr(result, 'results') and
                result.benchmark_type == "layer" and
                result.valid_type is True and
                len(result.results) > 0
            )
            message = f"type={result.benchmark_type}, results={len(result.results)}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("bench_default", passed, duration, message)

    def test_bench_attention(self):
        """Test bench with attention benchmark type."""
        start = time.time()

        try:
            result = ny.bench("attention")
            passed = (
                result.benchmark_type == "attention" and
                result.valid_type is True and
                len(result.results) > 0
            )
            message = f"type={result.benchmark_type}, results={len(result.results)}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("bench_attention", passed, duration, message)

    def test_bench_full(self):
        """Test bench with full benchmark type."""
        start = time.time()

        try:
            result = ny.bench("full")
            passed = (
                result.benchmark_type == "full" and
                result.valid_type is True and
                len(result.results) > 0
            )
            message = f"type={result.benchmark_type}, results={len(result.results)}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("bench_full", passed, duration, message)

    def test_bench_invalid_type(self):
        """Test bench with invalid benchmark type returns valid_type=False."""
        start = time.time()

        try:
            result = ny.bench("invalid")
            passed = (
                result.benchmark_type == "invalid" and
                result.valid_type is False and
                len(result.results) == 0
            )
            message = f"valid_type={result.valid_type}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("bench_invalid_type", passed, duration, message)

    def test_bench_dimensions(self):
        """Test BenchDimensions attributes."""
        start = time.time()

        try:
            result = ny.bench()
            dims = result.dimensions
            passed = (
                hasattr(dims, 'batch') and
                hasattr(dims, 'seq_len') and
                hasattr(dims, 'hidden_dim') and
                hasattr(dims, 'intermediate_dim') and
                hasattr(dims, 'num_heads') and
                hasattr(dims, 'head_dim') and
                hasattr(dims, 'epsilon') and
                dims.batch > 0 and
                dims.hidden_dim == 384  # Whisper-tiny dimensions
            )
            message = f"batch={dims.batch}, hidden={dims.hidden_dim}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("bench_dimensions", passed, duration, message)

    def test_bench_result_item(self):
        """Test BenchResultItem attributes."""
        start = time.time()

        try:
            result = ny.bench()
            item = result.results[0]
            passed = (
                hasattr(item, 'name') and
                hasattr(item, 'iterations') and
                hasattr(item, 'per_iter_ns') and
                hasattr(item, 'per_iter_us') and
                hasattr(item, 'per_iter_ms') and
                hasattr(item, 'total_ns') and
                hasattr(item, 'total_ms') and
                item.iterations > 0 and
                item.per_iter_ms > 0
            )
            message = f"name={item.name}, per_iter_ms={item.per_iter_ms:.3f}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("bench_result_item", passed, duration, message)

    def test_bench_summary(self):
        """Test BenchResult summary method."""
        start = time.time()

        try:
            result = ny.bench()
            summary = result.summary()
            passed = (
                isinstance(summary, str) and
                "Benchmark: layer" in summary and
                "Dimensions:" in summary and
                "Results:" in summary
            )
            message = f"summary_len={len(summary)}"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("bench_summary", passed, duration, message)

    # ========================================================================
    # Test Group 13: Module Attributes
    # ========================================================================

    def test_module_version(self):
        """Test module has version attribute."""
        start = time.time()

        try:
            passed = hasattr(ny, '__version__') and isinstance(ny.__version__, str)
            message = f"version={ny.__version__}" if passed else "No __version__"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("module_version", passed, duration, message)

    def test_module_exports(self):
        """Test module exports expected classes and functions."""
        start = time.time()

        expected = [
            'diff', 'sensitivity_analysis', 'quantize_check',
            'profile_bounds', 'load_model_info', 'load_npy', 'verify',
            'compare', 'weights_info', 'weights_diff', 'bench',
            'DiffResult', 'DiffStatus', 'LayerComparison', 'TensorSpec', 'ModelInfo', 'SensitivityResult', 'QuantizationResult', 'ProfileResult',
            'VerifyResult', 'VerifyStatus', 'OutputBound',
            'CompareResult', 'BoundViolation',
            'WeightsInfo', 'TensorInfo', 'WeightsDiffResult', 'TensorComparison', 'TensorComparisonStatus',
            'BenchResult', 'BenchResultItem', 'BenchDimensions'
        ]

        try:
            missing = [name for name in expected if not hasattr(ny, name)]
            passed = len(missing) == 0
            message = f"missing: {missing}" if missing else "all exports present"
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        duration = (time.time() - start) * 1000
        self.add_result("module_exports", passed, duration, message)

    # ========================================================================
    # Run all tests
    # ========================================================================

    def run_all(self) -> bool:
        """Run all tests and return True if all pass."""
        if not HAS_NY:
            print(f"FATAL: Cannot import ny module: {NY_IMPORT_ERROR}")
            print("Build with: maturin develop --release")
            return False
        required_model = TEST_MODELS_DIR / "simple_mlp.onnx"
        if not required_model.is_file():
            print(f"FATAL: Required tracked test fixture not found: {required_model}")
            return False

        print("=" * 70)
        print("ny Python API Unit Tests")
        print("=" * 70)

        # Group 1: diff
        print("\nTest Group 1: ny.diff()")
        print("-" * 40)
        self.test_diff_same_model()
        self.test_diff_with_tolerance()
        self.test_diff_layer_count()
        self.test_diff_layer_comparison_attrs()
        self.test_diff_with_input()

        # Group 2: sensitivity
        print("\nTest Group 2: ny.sensitivity_analysis()")
        print("-" * 40)
        self.test_sensitivity_basic()
        self.test_sensitivity_with_epsilon()
        self.test_sensitivity_layer_attrs()

        # Group 3: quantize_check
        print("\nTest Group 3: ny.quantize_check()")
        print("-" * 40)
        self.test_quantize_check_basic()
        self.test_quantize_check_with_epsilon()
        self.test_quantize_layer_attrs()

        # Group 4: profile_bounds
        print("\nTest Group 4: ny.profile_bounds()")
        print("-" * 40)
        self.test_profile_bounds_basic()
        self.test_profile_layer_attrs()

        # Group 5: load_model_info
        print("\nTest Group 5: ny.load_model_info()")
        print("-" * 40)
        self.test_load_model_info_basic()
        self.test_load_model_info_shapes()

        # Group 6: load_npy
        print("\nTest Group 6: ny.load_npy()")
        print("-" * 40)
        self.test_load_npy()

        # Group 7: Error handling
        print("\nTest Group 7: Error Handling")
        print("-" * 40)
        self.test_diff_missing_file()
        self.test_sensitivity_missing_file()

        # Group 8: verify
        print("\nTest Group 8: ny.verify()")
        print("-" * 40)
        self.test_verify_basic()
        self.test_verify_with_method()
        self.test_verify_with_epsilon()
        self.test_verify_output_bounds()
        self.test_verify_result_attrs()
        self.test_verify_output_bounds_verified()
        self.test_verify_output_bounds_unsatisfied()
        self.test_verify_all_infinite_output_bounds()
        self.test_verify_invalid_method()
        self.test_verify_missing_file()

        # Group 9: compare
        print("\nTest Group 9: ny.compare()")
        print("-" * 40)
        self.test_compare_same_model()
        self.test_compare_with_tolerance()
        self.test_compare_with_method()
        self.test_compare_result_attrs()
        self.test_compare_invalid_method()

        # Group 10: weights_info
        print("\nTest Group 10: ny.weights_info()")
        print("-" * 40)
        self.test_weights_info_onnx()
        self.test_weights_info_tensor_attrs()
        self.test_weights_info_summary()
        self.test_weights_info_unsupported()
        self.test_weights_info_pytorch()

        # Group 11: weights_diff
        print("\nTest Group 11: ny.weights_diff()")
        print("-" * 40)
        self.test_weights_diff_same_model()
        self.test_weights_diff_with_tolerance()
        self.test_weights_diff_result_attrs()
        self.test_weights_diff_summary()
        self.test_weights_diff_matching_tensors()
        self.test_weights_diff_pytorch()

        # Group 12: bench
        print("\nTest Group 12: ny.bench()")
        print("-" * 40)
        self.test_bench_default()
        self.test_bench_attention()
        self.test_bench_full()
        self.test_bench_invalid_type()
        self.test_bench_dimensions()
        self.test_bench_result_item()
        self.test_bench_summary()

        # Group 13: Module attributes
        print("\nTest Group 13: Module Attributes")
        print("-" * 40)
        self.test_module_version()
        self.test_module_exports()

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Test':<40} {'Status':<10} {'Time (ms)':<12}")
        print("-" * 70)

        passed_count = 0
        failed_count = 0
        skipped_count = 0
        for result in self.results:
            print(f"{result.name:<40} {result.status:<10} {result.duration_ms:>8.1f}")
            if result.status == "PASS":
                passed_count += 1
            elif result.status == "FAIL":
                failed_count += 1
            else:
                skipped_count += 1

        print("-" * 70)
        print(
            f"\nTotal: {passed_count} passed, {failed_count} failed, "
            f"{skipped_count} skipped ({len(self.results)} total)"
        )

        all_passed = failed_count == 0
        if all_passed:
            if skipped_count:
                print("\nAll runnable tests PASSED; skipped tests were not counted as passes.")
            else:
                print("\nAll tests PASSED!")
        else:
            print("\nSome tests FAILED!")
            for result in self.results:
                if result.status == "FAIL":
                    print(f"  - {result.name}: {result.message}")

        return all_passed


def main():
    """Main entry point."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    tests = PythonAPITests(verbose=verbose)
    success = tests.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
