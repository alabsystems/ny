# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""
Tests for ny pytest assertions.

These tests verify that assertion functions work correctly and
produce meaningful error messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ny_pytest.assertions import (
    assert_bounds,
    assert_equivalent,
    assert_quantization_safe,
    assert_verified,
)
from ny_pytest.errors import (
    BoundsError,
    EquivalenceError,
    QuantizationError,
    VerificationError,
)


# Mock classes for testing without the actual ny module
@dataclass
class MockBound:
    lower: float
    upper: float


@dataclass
class MockVerifyResult:
    status: str
    method: str = "crown"
    reason: str | None = None
    output_bounds: list[MockBound] | None = None
    counterexample: list[float] | None = None
    counterexample_output: float | None = None


@dataclass
class MockLayerComparison:
    name: str
    max_diff: float
    exceeds_tolerance: bool
    shape_a: list[int]
    shape_b: list[int]


@dataclass
class MockDiffResult:
    layers: list[MockLayerComparison]
    is_equivalent: bool
    max_divergence: float
    tolerance: float
    first_bad_layer_name: str | None = None
    drift_start_layer: int | None = None
    suggestion: str | None = None


@dataclass
class MockProfileLayer:
    name: str
    output_lower: float
    output_upper: float


@dataclass
class MockProfileResult:
    layers: list[MockProfileLayer]


@dataclass
class MockQuantLayer:
    name: str
    has_overflow: bool
    max_value: float = 0.0


@dataclass
class MockQuantResult:
    layers: list[MockQuantLayer]
    safe_for_float16: bool
    safe_for_int8: bool
    summary: str = ""


class TestAssertVerified:
    """Tests for assert_verified()."""

    def test_verified_passes(self) -> None:
        """Verified result should not raise."""
        result = MockVerifyResult(status="VerifyStatus.Verified")
        assert_verified(result)  # Should not raise

    def test_violated_raises(self) -> None:
        """Violated result should raise VerificationError."""
        result = MockVerifyResult(
            status="VerifyStatus.Violated",
            reason="Counterexample found",
            counterexample=[0.1, -0.1, 0.2],
            counterexample_output=-0.5,
        )
        with pytest.raises(VerificationError) as exc_info:
            assert_verified(result)

        error = exc_info.value
        assert error.result is result
        assert "Verification FAILED" in str(error)
        assert "Counterexample" in str(error)

    def test_unknown_raises(self) -> None:
        """Unknown result should raise VerificationError."""
        result = MockVerifyResult(
            status="VerifyStatus.Unknown",
            reason="Timeout",
            output_bounds=[MockBound(-1.0, 2.0), MockBound(-0.5, 1.5)],
        )
        with pytest.raises(VerificationError) as exc_info:
            assert_verified(result)

        error_str = str(exc_info.value)
        assert "Unknown" in error_str
        assert "Timeout" in error_str

    def test_invalid_type_raises(self) -> None:
        """Invalid input type should raise TypeError."""
        with pytest.raises(TypeError, match="Expected VerifyResult"):
            assert_verified({"status": "Verified"})


class TestAssertEquivalent:
    """Tests for assert_equivalent()."""

    def test_equivalent_passes(self) -> None:
        """Equivalent models should not raise."""
        result = MockDiffResult(
            layers=[
                MockLayerComparison("layer1", 1e-6, False, [10], [10]),
                MockLayerComparison("layer2", 1e-7, False, [10], [10]),
            ],
            is_equivalent=True,
            max_divergence=1e-6,
            tolerance=1e-5,
        )
        assert_equivalent(result)  # Should not raise

    def test_not_equivalent_raises(self) -> None:
        """Non-equivalent models should raise EquivalenceError."""
        result = MockDiffResult(
            layers=[
                MockLayerComparison("layer1", 1e-6, False, [10], [10]),
                MockLayerComparison("bad_layer", 0.1, True, [10], [10]),
            ],
            is_equivalent=False,
            max_divergence=0.1,
            tolerance=1e-5,
            first_bad_layer_name="bad_layer",
            suggestion="Check activation functions",
        )
        with pytest.raises(EquivalenceError) as exc_info:
            assert_equivalent(result)

        error_str = str(exc_info.value)
        assert "equivalence FAILED" in error_str
        assert "bad_layer" in error_str
        assert "Suggestion" in error_str

    def test_custom_tolerance(self) -> None:
        """Custom tolerance should override diff_result.tolerance."""
        result = MockDiffResult(
            layers=[MockLayerComparison("layer1", 1e-4, False, [10], [10])],
            is_equivalent=True,
            max_divergence=1e-4,
            tolerance=1e-3,  # Original tolerance is OK
        )
        # Should pass with original tolerance
        assert_equivalent(result)

        # Should fail with stricter tolerance
        with pytest.raises(EquivalenceError):
            assert_equivalent(result, tolerance=1e-5)

    def test_shape_mismatch_in_error(self) -> None:
        """Shape mismatch should be included in error message."""
        result = MockDiffResult(
            layers=[
                MockLayerComparison("mismatched", 0.0, False, [10, 20], [10, 30]),
            ],
            is_equivalent=False,
            max_divergence=0.0,
            tolerance=1e-5,
        )
        with pytest.raises(EquivalenceError) as exc_info:
            assert_equivalent(result)

        assert "SHAPE MISMATCH" in str(exc_info.value)

    def test_invalid_type_raises(self) -> None:
        """Invalid input type should raise TypeError."""
        with pytest.raises(TypeError, match="Expected DiffResult"):
            assert_equivalent({"is_equivalent": True})


class TestAssertBounds:
    """Tests for assert_bounds()."""

    def test_reasonable_bounds_pass(self) -> None:
        """Reasonable bounds should not raise."""
        result = MockProfileResult(
            layers=[
                MockProfileLayer("layer1", -1.0, 1.0),
                MockProfileLayer("layer2", -10.0, 10.0),
            ]
        )
        assert_bounds(result, max_width=100)  # Should not raise

    def test_exploded_bounds_raise(self) -> None:
        """Exploded bounds should raise BoundsError."""
        result = MockProfileResult(
            layers=[
                MockProfileLayer("layer1", -1.0, 1.0),
                MockProfileLayer("exploded", -1e10, 1e10),
                MockProfileLayer("also_bad", -1e8, 1e8),
            ]
        )
        with pytest.raises(BoundsError) as exc_info:
            assert_bounds(result, max_width=1e6)

        error = exc_info.value
        assert len(error.violations) == 2
        assert error.max_width == 1e6
        assert "exploded" in str(error)

    def test_default_max_width(self) -> None:
        """Default max_width should be 1e6."""
        result = MockProfileResult(
            layers=[MockProfileLayer("huge", -1e7, 1e7)]
        )
        with pytest.raises(BoundsError) as exc_info:
            assert_bounds(result)  # Uses default max_width=1e6

        assert exc_info.value.max_width == 1e6

    def test_empty_layers_pass(self) -> None:
        """Empty layers list should pass (no violations)."""
        result = MockProfileResult(layers=[])
        assert_bounds(result, max_width=1.0)  # Should not raise

    def test_no_matching_attributes_pass(self) -> None:
        """Layers without bound attributes should be skipped."""
        @dataclass
        class LayerWithoutBounds:
            name: str

        @dataclass
        class ResultWithMixedLayers:
            layers: list[Any]

        result = ResultWithMixedLayers(
            layers=[LayerWithoutBounds("no_bounds")]
        )
        assert_bounds(result, max_width=1.0)  # Should not raise


class TestAssertQuantizationSafe:
    """Tests for assert_quantization_safe()."""

    def test_safe_model_passes(self) -> None:
        """Safe model should not raise."""
        result = MockQuantResult(
            layers=[
                MockQuantLayer("layer1", has_overflow=False),
                MockQuantLayer("layer2", has_overflow=False),
            ],
            safe_for_float16=True,
            safe_for_int8=True,
        )
        assert_quantization_safe(result, dtype="float16")
        assert_quantization_safe(result, dtype="int8")

    def test_unsafe_float16_raises(self) -> None:
        """Unsafe float16 model should raise QuantizationError."""
        result = MockQuantResult(
            layers=[
                MockQuantLayer("overflow_layer", has_overflow=True, max_value=1e38),
            ],
            safe_for_float16=False,
            safe_for_int8=False,
            summary="Max value exceeds float16 range",
        )
        with pytest.raises(QuantizationError) as exc_info:
            assert_quantization_safe(result, dtype="float16")

        error_str = str(exc_info.value)
        assert "float16" in error_str
        assert "overflow_layer" in error_str

    def test_unsafe_int8_raises(self) -> None:
        """Unsafe int8 model should raise QuantizationError."""
        # Value 200 is safe for float16 but exceeds int8 range [-128, 127]
        result = MockQuantResult(
            layers=[MockQuantLayer("int8_overflow", has_overflow=False, max_value=200)],
            safe_for_float16=True,
            safe_for_int8=False,
        )
        # Should pass for float16 (no overflow, safe_for_float16=True)
        assert_quantization_safe(result, dtype="float16")

        # Should fail for int8 (safe_for_int8=False)
        with pytest.raises(QuantizationError):
            assert_quantization_safe(result, dtype="int8")

    def test_invalid_dtype_raises(self) -> None:
        """Invalid dtype should raise ValueError."""
        result = MockQuantResult(
            layers=[],
            safe_for_float16=True,
            safe_for_int8=True,
        )
        with pytest.raises(ValueError, match="Unknown dtype"):
            assert_quantization_safe(result, dtype="bfloat16")

    def test_invalid_type_raises(self) -> None:
        """Invalid input type should raise TypeError."""
        with pytest.raises(TypeError, match="Expected QuantizeResult"):
            assert_quantization_safe({"safe": True})
