# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""
pytest-style assertions for neural network verification.

These assertions provide clear, actionable error messages when
verification checks fail.
"""

from __future__ import annotations

from typing import Any

from ny_pytest.errors import (
    BoundsError,
    EquivalenceError,
    QuantizationError,
    VerificationError,
)


def assert_verified(result: Any, threshold: float = 0.0) -> None:
    """Assert that verification succeeded.

    Verifies that the model satisfies the specified property within the
    given threshold. Raises VerificationError with detailed diagnostics
    if verification fails.

    Args:
        result: VerifyResult from ny.verify()
        threshold: Output threshold for verified property (default: 0.0)

    Raises:
        VerificationError: If verification failed or was unknown

    Example:
        >>> result = ny.verify("model.onnx", epsilon=0.01, output_bounds=[(-1.0, 1.0)] * 10)
        >>> assert_verified(result)  # Raises if not verified

        >>> # With custom threshold
        >>> assert_verified(result, threshold=0.5)
    """
    # Check if result has the expected attributes
    if not hasattr(result, "status"):
        raise TypeError(
            f"Expected VerifyResult from ny.verify(), got {type(result).__name__}"
        )

    # Convert status to string for comparison
    status_str = str(result.status)
    if "Verified" not in status_str:
        raise VerificationError(result, threshold)


def assert_equivalent(
    diff_result: Any,
    tolerance: float | None = None,
) -> None:
    """Assert that two models are equivalent within tolerance.

    Checks that the DiffResult indicates model equivalence. The tolerance
    can be specified here to override the tolerance used during diffing.

    Args:
        diff_result: DiffResult from ny.diff()
        tolerance: Maximum allowed divergence (uses diff_result.tolerance if None)

    Raises:
        EquivalenceError: If models are not equivalent

    Example:
        >>> diff = ny.diff("torch.onnx", "metal.onnx")
        >>> assert_equivalent(diff)

        >>> # With custom tolerance
        >>> assert_equivalent(diff, tolerance=1e-3)
    """
    if not hasattr(diff_result, "is_equivalent"):
        raise TypeError(
            f"Expected DiffResult from ny.diff(), got {type(diff_result).__name__}"
        )

    # Use provided tolerance or fall back to diff_result's tolerance
    effective_tolerance = tolerance if tolerance is not None else diff_result.tolerance

    # Check equivalence
    if not diff_result.is_equivalent:
        raise EquivalenceError(diff_result, effective_tolerance)

    # Also check if max_divergence exceeds custom tolerance
    if tolerance is not None and diff_result.max_divergence > tolerance:
        raise EquivalenceError(diff_result, tolerance)


def assert_bounds(
    result: Any,
    max_width: float = 1e6,
) -> None:
    """Assert that bounds are within reasonable limits.

    Checks that no layer has bound width exceeding max_width. This is
    useful for detecting bound explosion in verification.

    Args:
        result: ProfileResult from ny.profile_bounds() or sensitivity result
        max_width: Maximum allowed bound width (default: 1e6)

    Raises:
        BoundsError: If any layer exceeds max_width

    Example:
        >>> result = ny.profile_bounds("model.onnx", epsilon=0.001)
        >>> assert_bounds(result, max_width=1e4)
    """
    violations: list[tuple[str, float]] = []

    # Handle ProfileResult
    if hasattr(result, "layers"):
        for layer in result.layers:
            # Calculate width from bounds if available
            if hasattr(layer, "output_lower") and hasattr(layer, "output_upper"):
                width = abs(layer.output_upper - layer.output_lower)
            elif hasattr(layer, "bound_width"):
                width = layer.bound_width
            elif hasattr(layer, "max_bound_width"):
                width = layer.max_bound_width
            else:
                continue

            if width > max_width:
                name = getattr(layer, "name", str(layer))
                violations.append((name, width))

    # Handle SensitivityResult
    elif hasattr(result, "sensitivities"):
        for layer in result.sensitivities:
            if hasattr(layer, "sensitivity") and layer.sensitivity > max_width:
                violations.append((layer.name, layer.sensitivity))

    # Handle VerifyResult with output_bounds
    elif hasattr(result, "output_bounds") and result.output_bounds:
        for i, bound in enumerate(result.output_bounds):
            width = bound.upper - bound.lower
            if width > max_width:
                violations.append((f"output_{i}", width))

    if violations:
        raise BoundsError(result, max_width, violations)


def assert_quantization_safe(
    result: Any,
    dtype: str = "float16",
) -> None:
    """Assert that model is safe for quantization to dtype.

    Checks that no activations would overflow when quantized to the
    specified dtype.

    Args:
        result: QuantizeResult from ny.quantize_check()
        dtype: Target quantization dtype ("float16" or "int8")

    Raises:
        QuantizationError: If model is not safe for quantization

    Example:
        >>> result = ny.quantize_check("model.onnx")
        >>> assert_quantization_safe(result, dtype="float16")
    """
    if not hasattr(result, "layers"):
        raise TypeError(
            f"Expected QuantizeResult from ny.quantize_check(), "
            f"got {type(result).__name__}"
        )

    # Check if any layer has overflow
    has_overflow = any(
        getattr(layer, "has_overflow", False)
        for layer in result.layers
    )

    # For float16, check overall safety
    if dtype == "float16":
        if hasattr(result, "safe_for_float16") and not result.safe_for_float16:
            raise QuantizationError(result, dtype)
        if has_overflow:
            raise QuantizationError(result, dtype)

    # For int8, check int8-specific safety
    elif dtype == "int8":
        if hasattr(result, "safe_for_int8") and not result.safe_for_int8:
            raise QuantizationError(result, dtype)
        if has_overflow:
            raise QuantizationError(result, dtype)

    else:
        raise ValueError(f"Unknown dtype: {dtype}. Use 'float16' or 'int8'.")
