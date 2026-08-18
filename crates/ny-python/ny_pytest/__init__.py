# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""
pytest plugin for neural network verification.

This module provides pytest-native testing infrastructure for neural networks,
enabling assertions about model robustness, equivalence, and bounds.

Example usage:
    >>> from ny_pytest import assert_verified, assert_equivalent, assert_bounds
    >>> import ny
    >>>
    >>> def test_classifier_robust():
    ...     result = ny.verify("model.onnx", epsilon=0.01, output_bounds=[(-1.0, 1.0)] * 10)
    ...     assert_verified(result)
    ...
    >>> def test_port_equivalent():
    ...     diff = ny.diff("torch.onnx", "metal.onnx")
    ...     assert_equivalent(diff, tolerance=1e-4)
"""

from ny_pytest.assertions import (
    assert_bounds,
    assert_equivalent,
    assert_quantization_safe,
    assert_verified,
)
from ny_pytest.config import NyConfig
from ny_pytest.errors import (
    BoundsError,
    EquivalenceError,
    QuantizationError,
    VerificationError,
)

__all__ = [
    # Assertions
    "assert_verified",
    "assert_equivalent",
    "assert_bounds",
    "assert_quantization_safe",
    # Configuration
    "NyConfig",
    # Errors
    "VerificationError",
    "EquivalenceError",
    "BoundsError",
    "QuantizationError",
]

__version__ = "0.2.0"
