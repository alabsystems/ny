# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Tests for ny Python bindings.

Run with: pytest tests/test_ny.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_NY_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import ny as _ny_mod
except ModuleNotFoundError as e:
    _ny_mod = None
    _NY_IMPORT_ERROR = e


class _MissingNy(ModuleType):
    def __getattr__(self, name: str):  # pragma: no cover
        raise AssertionError(
            "Python bindings not installed: cannot import `ny`.\n"
            "Build/install with: `python -m pip install -e crates/ny-python` "
            "(requires `maturin`)."
        ) from _NY_IMPORT_ERROR


ny = _ny_mod if _ny_mod is not None else _MissingNy("ny")

# Test models directory
MODELS_DIR = Path(__file__).resolve().parents[3] / "tests" / "models"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _torch_and_onnx_available():
    try:
        import torch
    except ModuleNotFoundError:
        return None, False
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError:
        return torch, False
    return torch, True


def _build_torch_models(torch):
    torch.manual_seed(0)
    model_a = torch.nn.Linear(2, 2)
    model_b = torch.nn.Linear(2, 2)
    model_b.load_state_dict(model_a.state_dict())
    example_input = torch.randn(1, 2)
    return model_a, model_b, example_input


class TestDiff:
    """Tests for ny.diff()."""

    def test_same_model_is_equivalent(self):
        """Same model compared to itself should be equivalent."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.diff(model_path, model_path)

        assert result.is_equivalent
        assert result.max_divergence == 0.0
        assert result.first_bad_layer is None
        assert result.first_bad_layer_name is None

    def test_multiple_layers_compared(self):
        """Diff should compare multiple intermediate layers."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.diff(model_path, model_path)

        # simple_mlp has 3 layers: fc1_out, relu_out, output
        assert len(result.layers) >= 2

    def test_custom_tolerance(self):
        """Custom tolerance should be respected."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")

        # With any tolerance, same model should pass
        result = ny.diff(model_path, model_path, tolerance=1e-10)
        assert result.is_equivalent
        assert result.tolerance == pytest.approx(1e-10)

    def test_custom_input(self):
        """Custom input array should be used."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")

        # Create custom input
        input_data = np.array([[1.0, 2.0]], dtype=np.float32)

        result = ny.diff(model_path, model_path, input=input_data)
        assert result.is_equivalent

    def test_layer_comparison_attributes(self):
        """LayerComparison should have correct attributes."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.diff(model_path, model_path)

        for layer in result.layers:
            assert isinstance(layer.name, str)
            assert isinstance(layer.max_diff, float)
            assert isinstance(layer.mean_diff, float)
            assert isinstance(layer.exceeds_tolerance, bool)
            assert isinstance(layer.shape_a, list)
            assert isinstance(layer.shape_b, list)

    def test_summary_output(self):
        """Summary should produce formatted table."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.diff(model_path, model_path)

        summary = result.summary()
        assert "Layer-by-Layer Comparison" in summary
        assert "Layer" in summary
        assert "Max Diff" in summary
        assert "Status" in summary


class TestTorchDiffAndCompare:
    """Tests for torch model diff/compare helpers."""

    def test_diff_torch_smoke(self):
        torch, onnx_available = _torch_and_onnx_available()
        if torch is None:
            with pytest.raises(ValueError, match="PyTorch is required for diff_torch"):
                ny.diff_torch(object(), object(), object())
            return

        model_a, model_b, example_input = _build_torch_models(torch)

        if not onnx_available:
            with pytest.raises(Exception) as excinfo:
                ny.diff_torch(model_a, model_b, example_input)
            assert "onnx" in str(excinfo.value).lower()
            return

        result = ny.diff_torch(
            model_a,
            model_b,
            example_input,
            tolerance=1e-5,
        )
        assert result.is_equivalent
        assert result.max_divergence <= 1e-5

    def test_compare_torch_smoke(self):
        torch, onnx_available = _torch_and_onnx_available()
        if torch is None:
            with pytest.raises(ValueError, match="PyTorch is required for compare_torch"):
                ny.compare_torch(object(), object(), object())
            return

        model_a, model_b, example_input = _build_torch_models(torch)

        if not onnx_available:
            with pytest.raises(Exception) as excinfo:
                ny.compare_torch(model_a, model_b, example_input)
            assert "onnx" in str(excinfo.value).lower()
            return

        result = ny.compare_torch(
            model_a,
            model_b,
            example_input,
            tolerance=1e-5,
            epsilon=0.0,
            method="crown",
        )
        assert result.is_equivalent
        assert result.max_lower_diff <= 1e-5
        assert result.max_upper_diff <= 1e-5


class TestRunWithIntermediates:
    """Tests for ny.run_with_intermediates()."""

    def test_returns_dict(self):
        """Should return dict of numpy arrays."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        input_data = np.zeros((1, 2), dtype=np.float32)

        outputs = ny.run_with_intermediates(model_path, input_data)

        assert isinstance(outputs, dict)
        assert len(outputs) > 0

        for name, arr in outputs.items():
            assert isinstance(name, str)
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.float32

    def test_includes_intermediate_layers(self):
        """Should include intermediate layer outputs."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        input_data = np.zeros((1, 2), dtype=np.float32)

        outputs = ny.run_with_intermediates(model_path, input_data)

        # simple_mlp has fc1_out, relu_out, output
        assert "output" in outputs
        # At least one intermediate
        assert len(outputs) >= 2


class TestLoadModelInfo:
    """Tests for ny.load_model_info()."""

    def test_returns_typed_model_info(self):
        """Should return typed model info."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")

        info = ny.load_model_info(model_path)

        assert isinstance(
            info, ny.ModelInfo
        ), f"Expected ny.ModelInfo, got {type(info)!r}"
        assert isinstance(info.inputs, list), f"Expected list inputs, got {type(info.inputs)!r}"
        assert isinstance(
            info.outputs, list
        ), f"Expected list outputs, got {type(info.outputs)!r}"
        assert isinstance(
            info.layer_count, int
        ), f"Expected int layer_count, got {type(info.layer_count)!r}"
        assert isinstance(
            info.layer_names, list
        ), f"Expected list layer_names, got {type(info.layer_names)!r}"

    def test_input_info(self):
        """Should have input name and shape."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")

        info = ny.load_model_info(model_path)

        assert len(info.inputs) >= 1, "Expected at least one typed input spec"
        input_info = info.inputs[0]
        assert isinstance(
            input_info, ny.TensorSpec
        ), f"Expected ny.TensorSpec, got {type(input_info)!r}"
        assert isinstance(
            input_info.name, str
        ), f"Expected TensorSpec.name to be str, got {type(input_info.name)!r}"
        assert isinstance(
            input_info.shape, list
        ), f"Expected TensorSpec.shape to be list, got {type(input_info.shape)!r}"
        assert isinstance(
            input_info.dtype, str
        ), f"Expected TensorSpec.dtype to be str, got {type(input_info.dtype)!r}"

    def test_diff_status_str(self):
        """DiffStatus should provide a clean __str__."""
        assert (
            str(ny.DiffStatus.Ok) == "ok"
        ), f"Expected str(DiffStatus.Ok) == 'ok', got {str(ny.DiffStatus.Ok)!r}"
        assert (
            str(ny.DiffStatus.ShapeMismatch) == "shape_mismatch"
        ), "Expected snake-case DiffStatus string form"


class TestDiffResult:
    """Tests for DiffResult class."""

    def test_repr(self):
        """DiffResult should have readable repr."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.diff(model_path, model_path)

        repr_str = repr(result)
        assert "DiffResult" in repr_str
        assert "layers=" in repr_str
        assert "max_divergence=" in repr_str


class TestVersion:
    """Test module version."""

    def test_version_exists(self):
        """Module should have version."""
        assert hasattr(ny, "__version__")
        assert isinstance(ny.__version__, str)
        assert ny.__version__ == "0.1.0"


# ==============================================================================
# Custom Ops API Tests
# ==============================================================================


class TestCustomOps:
    """Tests for ny custom-op registration behavior."""

    def test_register_custom_op_rejected_until_dispatch_is_wired(self):
        def bound_impl(*args, **kwargs):
            return args, kwargs

        with pytest.raises(
            NotImplementedError,
            match=r"register_custom_op is not implemented yet",
        ):
            ny.register_custom_op("com.example", "MyCustomOp", None, bound_impl)


# ==============================================================================
# P2: Sensitivity Analysis Tests
# ==============================================================================


class TestSensitivityAnalysis:
    """Tests for ny.sensitivity_analysis()."""

    def test_basic_sensitivity(self):
        """Should analyze sensitivity of simple model."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.sensitivity_analysis(model_path)

        assert len(result.layers) > 0
        assert result.max_sensitivity > 0
        assert result.total_sensitivity > 0
        assert result.input_epsilon == pytest.approx(0.01)

    def test_custom_epsilon(self):
        """Custom epsilon should be respected."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.sensitivity_analysis(model_path, epsilon=0.1)

        assert result.input_epsilon == pytest.approx(0.1)

    def test_layer_sensitivity_attributes(self):
        """LayerSensitivity should have correct attributes."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.sensitivity_analysis(model_path)

        for layer in result.layers:
            assert isinstance(layer.name, str)
            assert isinstance(layer.layer_type, str)
            assert isinstance(layer.sensitivity, float)
            assert isinstance(layer.input_width, float)
            assert isinstance(layer.output_width, float)
            assert isinstance(layer.has_overflow, bool)
            assert isinstance(layer.output_shape, list)

    def test_max_sensitivity_layer_name(self):
        """Should identify layer with max sensitivity."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.sensitivity_analysis(model_path)

        if result.max_sensitivity_layer is not None:
            assert result.max_sensitivity_layer_name is not None
            assert isinstance(result.max_sensitivity_layer_name, str)

    def test_hot_spots(self):
        """hot_spots should filter by threshold."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.sensitivity_analysis(model_path)

        # Very high threshold should return no layers
        hot = result.hot_spots(1e10)
        assert len(hot) == 0

        # Very low threshold should return all layers with sensitivity > threshold
        hot = result.hot_spots(0.0)
        assert len(hot) == len([layer for layer in result.layers if layer.sensitivity > 0.0])

    def test_summary(self):
        """Summary should produce formatted table."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.sensitivity_analysis(model_path)

        summary = result.summary()
        assert "Sensitivity Analysis" in summary
        assert "Layer" in summary
        assert "Sens." in summary

    def test_layer_is_contractive(self):
        """is_contractive should work correctly."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.sensitivity_analysis(model_path)

        for layer in result.layers:
            assert layer.is_contractive() == (layer.sensitivity < 1.0)


class TestSensitivityResult:
    """Tests for SensitivityResult class."""

    def test_repr(self):
        """SensitivityResult should have readable repr."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.sensitivity_analysis(model_path)

        repr_str = repr(result)
        assert "SensitivityResult" in repr_str
        assert "layers=" in repr_str
        assert "max_sensitivity=" in repr_str

    def test_has_overflow_property(self):
        """has_overflow property should work."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.sensitivity_analysis(model_path)

        # Simple model shouldn't overflow
        assert isinstance(result.has_overflow, bool)


# ==============================================================================
# P2: Quantization Safety Tests
# ==============================================================================


class TestQuantizeCheck:
    """Tests for ny.quantize_check()."""

    def test_basic_quantization_check(self):
        """Should analyze quantization safety of simple model."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.quantize_check(model_path)

        assert len(result.layers) > 0
        assert isinstance(result.float16_safe, bool)
        assert isinstance(result.int8_safe, bool)
        assert result.input_epsilon == pytest.approx(0.01)

    def test_custom_epsilon(self):
        """Custom epsilon should be respected."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.quantize_check(model_path, epsilon=0.1)

        assert result.input_epsilon == pytest.approx(0.1)

    def test_legacy_format_flags_accepted(self):
        """Legacy check_float16/check_int8 kwargs should remain callable."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.quantize_check(
            model_path, epsilon=0.01, check_float16=False, check_int8=True
        )

        assert len(result.layers) > 0
        assert isinstance(result.float16_safe, bool)

    def test_layer_quantization_attributes(self):
        """LayerQuantization should have correct attributes."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.quantize_check(model_path)

        for layer in result.layers:
            assert isinstance(layer.name, str)
            assert isinstance(layer.layer_type, str)
            assert isinstance(layer.min_bound, float)
            assert isinstance(layer.max_bound, float)
            assert isinstance(layer.max_abs, float)
            assert isinstance(layer.output_shape, list)
            assert isinstance(layer.has_overflow, bool)
            # QuantSafety enum values
            assert layer.float16_safety is not None
            assert layer.int8_safety is not None

    def test_float16_unsafe_layers(self):
        """float16_unsafe_layers should filter correctly."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.quantize_check(model_path)

        unsafe = result.float16_unsafe_layers()
        # Simple model should be safe for float16
        assert isinstance(unsafe, list)

    def test_int8_unsafe_layers(self):
        """int8_unsafe_layers should filter correctly."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.quantize_check(model_path)

        unsafe = result.int8_unsafe_layers()
        assert isinstance(unsafe, list)

    def test_summary(self):
        """Summary should produce formatted table."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.quantize_check(model_path)

        summary = result.summary()
        assert "Quantization Safety Analysis" in summary
        assert "Layer" in summary
        assert "F16" in summary
        assert "I8" in summary

    def test_layer_is_safe_methods(self):
        """is_float16_safe/is_int8_safe should work."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.quantize_check(model_path)

        for layer in result.layers:
            assert isinstance(layer.is_float16_safe(), bool)
            assert isinstance(layer.is_int8_safe(), bool)


class TestQuantSafety:
    """Tests for QuantSafety enum."""

    def test_enum_variants_exist(self):
        """QuantSafety should have expected variants."""
        assert hasattr(ny.QuantSafety, "Safe")
        assert hasattr(ny.QuantSafety, "Denormal")
        assert hasattr(ny.QuantSafety, "ScalingRequired")
        assert hasattr(ny.QuantSafety, "Overflow")
        assert hasattr(ny.QuantSafety, "Unknown")

    def test_repr(self):
        """QuantSafety should have readable repr."""
        assert "QuantSafety.Safe" in repr(ny.QuantSafety.Safe)
        assert "QuantSafety.Overflow" in repr(ny.QuantSafety.Overflow)


class TestQuantizationResult:
    """Tests for QuantizationResult class."""

    def test_repr(self):
        """QuantizationResult should have readable repr."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.quantize_check(model_path)

        repr_str = repr(result)
        assert "QuantizationResult" in repr_str
        assert "layers=" in repr_str
        assert "float16_safe=" in repr_str


# ==============================================================================
# P2: Bound Width Profiling Tests
# ==============================================================================


class TestProfileBounds:
    """Tests for ny.profile_bounds()."""

    def test_basic_profile(self):
        """Should profile bounds of simple model."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.profile_bounds(model_path)

        assert len(result.layers) > 0
        assert result.total_expansion > 0
        assert result.difficulty_score >= 0
        assert result.difficulty_score <= 100
        assert result.input_epsilon == pytest.approx(0.01)

    def test_custom_epsilon(self):
        """Custom epsilon should be respected."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.profile_bounds(model_path, epsilon=0.1)

        assert result.input_epsilon == pytest.approx(0.1)

    def test_layer_profile_attributes(self):
        """LayerProfile should have correct attributes."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.profile_bounds(model_path)

        for layer in result.layers:
            assert isinstance(layer.name, str)
            assert isinstance(layer.layer_type, str)
            assert isinstance(layer.input_width, float)
            assert isinstance(layer.output_width, float)
            assert isinstance(layer.growth_ratio, float)
            assert isinstance(layer.cumulative_expansion, float)
            assert isinstance(layer.output_shape, list)
            assert isinstance(layer.num_elements, int)
            # BoundStatus enum
            assert layer.status is not None

    def test_max_growth_layer_name(self):
        """Should identify layer with max growth."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.profile_bounds(model_path)

        if result.max_growth_layer is not None:
            assert result.max_growth_layer_name is not None
            assert isinstance(result.max_growth_layer_name, str)

    def test_choke_points(self):
        """choke_points should filter by threshold."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.profile_bounds(model_path)

        # Very high threshold should return no layers
        chokes = result.choke_points(1e10)
        assert len(chokes) == 0

        # Very low threshold should return layers with growth > threshold
        chokes = result.choke_points(0.0)
        assert len(chokes) == len([layer for layer in result.layers if layer.growth_ratio > 0.0])

    def test_problematic_layers(self):
        """problematic_layers should filter correctly."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.profile_bounds(model_path)

        problematic = result.problematic_layers()
        assert isinstance(problematic, list)

    def test_summary(self):
        """Summary should produce formatted table."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.profile_bounds(model_path)

        summary = result.summary()
        assert "Bound Width Profile" in summary
        assert "Layer" in summary
        assert "Growth" in summary
        assert "difficulty" in summary.lower()

    def test_layer_is_choke_point(self):
        """is_choke_point should work correctly."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.profile_bounds(model_path)

        for layer in result.layers:
            # Check with layer's own growth ratio
            threshold = layer.growth_ratio
            assert layer.is_choke_point(threshold - 0.001)
            assert not layer.is_choke_point(threshold + 0.001)


class TestBoundStatus:
    """Tests for BoundStatus enum."""

    def test_enum_variants_exist(self):
        """BoundStatus should have expected variants."""
        assert hasattr(ny.BoundStatus, "Tight")
        assert hasattr(ny.BoundStatus, "Moderate")
        assert hasattr(ny.BoundStatus, "Wide")
        assert hasattr(ny.BoundStatus, "VeryWide")
        assert hasattr(ny.BoundStatus, "Overflow")

    def test_repr(self):
        """BoundStatus should have readable repr."""
        assert "BoundStatus.Tight" in repr(ny.BoundStatus.Tight)
        assert "BoundStatus.Wide" in repr(ny.BoundStatus.Wide)


class TestProfileResult:
    """Tests for ProfileResult class."""

    def test_repr(self):
        """ProfileResult should have readable repr."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.profile_bounds(model_path)

        repr_str = repr(result)
        assert "ProfileResult" in repr_str
        assert "layers=" in repr_str
        assert "expansion=" in repr_str
        assert "difficulty=" in repr_str

    def test_has_overflow_property(self):
        """has_overflow property should work."""
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"
        result = ny.profile_bounds(model_path)

        # Simple model shouldn't overflow
        assert isinstance(result.has_overflow, bool)


# ==============================================================================
# Example Pytest Integration Tests
# ==============================================================================


# Example of real-world usage test
class TestPytestIntegration:
    """Example tests showing pytest integration for NN testing."""

    def test_model_port_equivalent(self):
        """
        Example: Verify model port produces identical outputs.

        In real usage, you'd compare PyTorch export vs CoreML export.
        """
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"

        diff = ny.diff(model_path, model_path, tolerance=1e-5)
        assert diff.is_equivalent, f"Diverges at {diff.first_bad_layer_name}"

    def test_model_tolerance_assertion(self):
        """
        Example: Assert max divergence is within acceptable range.
        """
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"

        diff = ny.diff(model_path, model_path)
        assert diff.max_divergence < 1e-5, (
            f"Max divergence {diff.max_divergence} exceeds tolerance\n"
            f"{diff.summary()}"
        )

    def test_quantization_safe(self):
        """
        Example: Verify model is safe for quantization.
        """
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"

        result = ny.quantize_check(model_path)
        assert result.float16_safe, (
            f"Model has float16 overflow risk:\n{result.summary()}"
        )

    def test_verification_difficulty(self):
        """
        Example: Check verification difficulty is reasonable.
        """
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"

        result = ny.profile_bounds(model_path)
        assert result.difficulty_score < 50, (
            f"Model verification difficulty {result.difficulty_score}/100 is high\n"
            f"Problematic layers: {[lyr.name for lyr in result.problematic_layers()]}"
        )

    def test_no_high_sensitivity_layers(self):
        """
        Example: Check no layers have extreme sensitivity.
        """
        model_path = f"{MODELS_DIR}/simple_mlp.onnx"

        result = ny.sensitivity_analysis(model_path)
        hot_spots = result.hot_spots(100.0)  # >100x amplification
        assert len(hot_spots) == 0, (
            f"Found {len(hot_spots)} high-sensitivity layers:\n"
            + "\n".join(f"  {lyr.name}: {lyr.sensitivity:.2f}x" for lyr in hot_spots)
        )


# ==============================================================================
# Weights API Tests
# ==============================================================================


class TestWeightsInfo:
    """Tests for ny.weights_info()."""

    def test_weights_info_onnx(self):
        """Should load ONNX weights info."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        info = ny.weights_info(model_path)

        assert isinstance(info.format, str)
        assert info.format == "ONNX"
        assert info.tensor_count > 0
        assert info.total_params > 0
        assert len(info.tensors) > 0

    def test_weights_info_gguf(self, gguf_model):
        """Should load GGUF weights info."""
        info = ny.weights_info(str(gguf_model))

        assert isinstance(info.format, str)
        assert info.format == "GGUF"
        assert info.tensor_count > 0
        assert info.total_params > 0
        assert len(info.tensors) > 0

    def test_weights_info_tensor_attributes(self):
        """TensorInfo should have correct attributes."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        info = ny.weights_info(model_path)

        for tensor in info.tensors:
            assert isinstance(tensor.name, str)
            assert isinstance(tensor.shape, list)
            assert isinstance(tensor.elements, int)
            assert all(isinstance(dim, int) for dim in tensor.shape)

    def test_weights_info_summary(self):
        """Summary should produce formatted output."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        info = ny.weights_info(model_path)

        summary = info.summary()
        assert "Weight" in summary or "Tensor" in summary
        assert isinstance(summary, str)

    def test_weights_info_unsupported_format_error(self, tmp_path):
        """Should raise error for unsupported format."""
        bad_file = tmp_path / "model.xyz"
        bad_file.write_text("dummy")

        with pytest.raises(ValueError, match=r"Unsupported format"):
            ny.weights_info(str(bad_file))

    def test_weights_info_file_not_found(self):
        """Should raise error for missing file."""
        with pytest.raises(ValueError, match=r"(?i)(not found|no such file|missing)"):
            ny.weights_info("/nonexistent/path/model.onnx")


class TestWeightsDiff:
    """Tests for ny.weights_diff()."""

    def test_weights_diff_same_file(self):
        """Same file compared to itself should match."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.weights_diff(model_path, model_path)

        assert result.is_match
        assert result.max_diff == 0.0
        assert result.differing_count == 0

    def test_weights_diff_attributes(self):
        """WeightsDiffResult should have correct attributes."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.weights_diff(model_path, model_path)

        assert isinstance(result.is_match, bool)
        assert isinstance(result.max_diff, float)
        assert isinstance(result.tolerance, float)
        assert isinstance(result.differing_count, int)
        assert isinstance(result.total_tensors_a, int)
        assert isinstance(result.total_tensors_b, int)
        assert isinstance(result.comparisons, list)

    def test_weights_diff_tolerance(self):
        """Custom tolerance should be respected."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.weights_diff(model_path, model_path, tolerance=1e-8)

        assert result.tolerance == pytest.approx(1e-8)
        assert result.is_match

    def test_weights_diff_summary(self):
        """Summary should produce formatted output."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.weights_diff(model_path, model_path)

        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_weights_diff_matching_tensors(self):
        """matching_tensors should return matching ones."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.weights_diff(model_path, model_path)

        matching = result.matching_tensors()
        assert isinstance(matching, list)
        # Same file should have all matching
        assert len(matching) == len(result.comparisons)

    def test_weights_diff_differing_tensors(self):
        """differing_tensors should return differing ones."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.weights_diff(model_path, model_path)

        differing = result.differing_tensors()
        assert isinstance(differing, list)
        # Same file should have no differing
        assert len(differing) == 0

    def test_tensor_comparison_attributes(self):
        """TensorComparison should have correct attributes."""
        model_path = str(MODELS_DIR / "simple_mlp.onnx")
        result = ny.weights_diff(model_path, model_path)

        for comp in result.comparisons:
            assert isinstance(
                comp.name, str
            ), f"Expected TensorComparison.name to be str, got {type(comp.name)!r}"
            assert isinstance(
                comp.status, ny.TensorComparisonStatus
            ), f"Expected TensorComparison.status to be TensorComparisonStatus, got {type(comp.status)!r}"
            # max_diff and shapes are optional
            if comp.max_diff is not None:
                assert isinstance(
                    comp.max_diff, float
                ), f"Expected max_diff float when present, got {type(comp.max_diff)!r}"

    def test_tensor_comparison_status_str(self):
        """TensorComparisonStatus should provide a stable string form."""
        assert (
            str(ny.TensorComparisonStatus.Match) == "match"
        ), "Expected TensorComparisonStatus.Match to stringify as 'match'"
        assert (
            str(ny.TensorComparisonStatus.MissingInB) == "missing_in_b"
        ), "Expected TensorComparisonStatus.MissingInB to stringify as 'missing_in_b'"


class TestWeightsInfoGGUF:
    """Tests specifically for GGUF format support."""

    def test_gguf_tensor_info(self, gguf_model):
        """GGUF tensors should have proper info."""
        info = ny.weights_info(str(gguf_model))

        # Check that we got meaningful data
        assert info.tensor_count > 0, "Expected at least some tensors"
        assert info.total_params > 0, "Expected non-zero params"

        # Verify tensor info looks reasonable
        for tensor in info.tensors[:5]:
            assert len(tensor.name) > 0
            assert len(tensor.shape) > 0
            assert tensor.elements > 0

    def test_gguf_summary_content(self, gguf_model):
        """GGUF summary should show meaningful content."""
        info = ny.weights_info(str(gguf_model))
        summary = info.summary()

        # Summary should mention the format and contain useful info
        assert "GGUF" in summary or "tensor" in summary.lower()


class TestShardedSafeTensors:
    """Tests for sharded SafeTensors directory support."""

    @pytest.fixture
    def sharded_dir(self, tmp_path):
        """Create a temporary directory with sharded SafeTensors files."""
        from safetensors.numpy import save_file  # Will fail if not installed
        rng = np.random.default_rng(0)

        # Create a config.json to make it look like a HuggingFace model directory
        config = {"model_type": "test", "hidden_size": 8}
        (tmp_path / "config.json").write_text(str(config).replace("'", '"'))

        # Create first shard with some weights
        shard1_tensors = {
            "model.layer1.weight": rng.standard_normal((8, 4), dtype=np.float32),
            "model.layer1.bias": rng.standard_normal(8, dtype=np.float32),
        }
        save_file(shard1_tensors, str(tmp_path / "model-00001-of-00002.safetensors"))

        # Create second shard with more weights
        shard2_tensors = {
            "model.layer2.weight": rng.standard_normal((4, 8), dtype=np.float32),
            "model.layer2.bias": rng.standard_normal(4, dtype=np.float32),
        }
        save_file(shard2_tensors, str(tmp_path / "model-00002-of-00002.safetensors"))

        return tmp_path

    def test_weights_info_sharded_directory(self, sharded_dir):
        """Should load weights from sharded SafeTensors directory."""
        info = ny.weights_info(str(sharded_dir))

        assert info.format == "SafeTensors (sharded)"
        assert info.tensor_count == 4  # 2 weights + 2 biases
        assert info.total_params > 0

        # Check that all expected tensors are present
        tensor_names = {t.name for t in info.tensors}
        assert "model.layer1.weight" in tensor_names
        assert "model.layer1.bias" in tensor_names
        assert "model.layer2.weight" in tensor_names
        assert "model.layer2.bias" in tensor_names

    def test_weights_diff_sharded_same_dir(self, sharded_dir):
        """Same sharded directory compared to itself should match."""
        result = ny.weights_diff(str(sharded_dir), str(sharded_dir))

        assert result.is_match
        assert result.max_diff == 0.0
        assert result.differing_count == 0
        assert result.total_tensors_a == 4
        assert result.total_tensors_b == 4

    def test_weights_diff_sharded_vs_single(self, sharded_dir, tmp_path):
        """Should compare sharded directory to single SafeTensors file."""
        from safetensors.numpy import load_file, save_file  # Will fail if not installed

        # Load all tensors from sharded directory and save to single file
        all_tensors = {}
        for sf in sharded_dir.glob("*.safetensors"):
            all_tensors.update(load_file(str(sf)))

        single_file = tmp_path / "combined.safetensors"
        save_file(all_tensors, str(single_file))

        # Compare sharded to single
        result = ny.weights_diff(str(sharded_dir), str(single_file))

        assert result.is_match
        assert result.max_diff == 0.0
        assert result.total_tensors_a == 4
        assert result.total_tensors_b == 4

    def test_weights_info_empty_directory_error(self, tmp_path):
        """Should raise error for empty directory."""
        # Create empty directory with config.json only
        (tmp_path / "config.json").write_text('{"model_type": "test"}')

        with pytest.raises(ValueError, match=r"(?i)(safetensors|directory)") as exc_info:
            ny.weights_info(str(tmp_path))
        # Should indicate no safetensors files found or similar
        assert "safetensors" in str(exc_info.value).lower() or "directory" in str(exc_info.value).lower()

    def test_weights_info_summary_sharded(self, sharded_dir):
        """Summary should show sharded format info."""
        info = ny.weights_info(str(sharded_dir))
        summary = info.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should show tensor information
        assert "layer" in summary.lower() or "weight" in summary.lower()


class TestPyTorchCheckpointDirectories:
    """Tests for HuggingFace-style PyTorch checkpoint directories."""

    @pytest.fixture
    def pytorch_checkpoint_dir(self, tmp_path):
        src = FIXTURES_DIR / "pytorch_model.bin"
        assert src.exists(), f"Missing test fixture: {src}"
        shutil.copyfile(src, tmp_path / "pytorch_model.bin")
        return tmp_path

    @pytest.fixture
    def pytorch_sharded_dir(self, tmp_path):
        shard1 = FIXTURES_DIR / "pytorch_model-00001-of-00002.bin"
        shard2 = FIXTURES_DIR / "pytorch_model-00002-of-00002.bin"
        index = FIXTURES_DIR / "pytorch_model.bin.index.json"
        assert shard1.exists(), f"Missing test fixture: {shard1}"
        assert shard2.exists(), f"Missing test fixture: {shard2}"
        assert index.exists(), f"Missing test fixture: {index}"

        shutil.copyfile(shard1, tmp_path / shard1.name)
        shutil.copyfile(shard2, tmp_path / shard2.name)
        shutil.copyfile(index, tmp_path / index.name)
        return tmp_path

    def test_weights_info_pytorch_checkpoint_directory(self, pytorch_checkpoint_dir):
        """Should load weights from directory containing pytorch_model.bin."""
        info = ny.weights_info(str(pytorch_checkpoint_dir))

        assert info.format == "PyTorch (checkpoint)"
        assert info.tensor_count == 2
        assert info.total_params > 0

        tensor_names = {t.name for t in info.tensors}
        assert "model.layer1.weight" in tensor_names
        assert "model.layer1.bias" in tensor_names

    def test_weights_info_pytorch_sharded_directory(self, pytorch_sharded_dir):
        """Should load weights from directory containing pytorch_model-*.bin + index."""
        info = ny.weights_info(str(pytorch_sharded_dir))

        assert info.format == "PyTorch (sharded)"
        assert info.tensor_count == 4
        assert info.total_params > 0

        tensor_names = {t.name for t in info.tensors}
        assert "model.layer1.weight" in tensor_names
        assert "model.layer1.bias" in tensor_names
        assert "model.layer2.weight" in tensor_names
        assert "model.layer2.bias" in tensor_names

    def test_weights_diff_pytorch_sharded_vs_single(self, pytorch_sharded_dir, tmp_path):
        """Should compare a sharded PyTorch directory to a single PyTorch checkpoint."""
        combined_src = FIXTURES_DIR / "pytorch_combined.bin"
        assert combined_src.exists(), f"Missing test fixture: {combined_src}"
        combined_path = tmp_path / "combined.bin"
        shutil.copyfile(combined_src, combined_path)

        result = ny.weights_diff(str(pytorch_sharded_dir), str(combined_path))
        assert result.is_match
        assert result.max_diff == 0.0
        assert result.total_tensors_a == 4
        assert result.total_tensors_b == 4
