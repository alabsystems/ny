# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""
Custom error types for ny pytest assertions.

These errors provide rich diagnostic information for verification failures.
"""

from __future__ import annotations

from typing import Any


class VerificationError(AssertionError):
    """Verification failed with diagnostic info.

    Raised when `assert_verified()` fails. Contains detailed information
    about why verification failed, including the method used, bounds
    computed, and any counterexamples found.

    Attributes:
        result: The VerifyResult from ny.verify()
        threshold: The threshold that was exceeded
    """

    def __init__(self, result: Any, threshold: float = 0.0) -> None:
        self.result = result
        self.threshold = threshold
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = ["Verification FAILED:"]
        lines.append(f"  Method: {self.result.method}")
        lines.append(f"  Status: {self.result.status}")

        if self.result.reason:
            lines.append(f"  Reason: {self.result.reason}")

        if self.result.output_bounds:
            lines.append("  Output bounds:")
            for i, bound in enumerate(self.result.output_bounds[:5]):
                lines.append(f"    [{i}]: [{bound.lower:.4g}, {bound.upper:.4g}]")
            if len(self.result.output_bounds) > 5:
                lines.append(f"    ... ({len(self.result.output_bounds) - 5} more)")

        if self.result.counterexample:
            ce = self.result.counterexample
            preview = ce[:5] if len(ce) > 5 else ce
            lines.append(f"  Counterexample: {preview}{'...' if len(ce) > 5 else ''}")

        if self.result.counterexample_output:
            lines.append(f"  Counterexample output: {self.result.counterexample_output}")

        return "\n".join(lines)


class EquivalenceError(AssertionError):
    """Model equivalence check failed.

    Raised when `assert_equivalent()` fails. Contains information about
    where the models diverge and by how much.

    Attributes:
        diff_result: The DiffResult from ny.diff()
        tolerance: The tolerance that was exceeded
    """

    def __init__(self, diff_result: Any, tolerance: float) -> None:
        self.diff_result = diff_result
        self.tolerance = tolerance
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = ["Model equivalence FAILED:"]
        lines.append(f"  Tolerance: {self.tolerance:.2e}")
        lines.append(f"  Max divergence: {self.diff_result.max_divergence:.2e}")

        if self.diff_result.first_bad_layer_name:
            lines.append(f"  First bad layer: {self.diff_result.first_bad_layer_name}")

        if self.diff_result.drift_start_layer is not None:
            idx = self.diff_result.drift_start_layer
            if idx < len(self.diff_result.layers):
                layer = self.diff_result.layers[idx]
                lines.append(f"  Drift starts at: {layer.name} (index {idx})")

        if self.diff_result.suggestion:
            lines.append(f"  Suggestion: {self.diff_result.suggestion}")

        # Show first few problematic layers
        bad_layers = [
            l for l in self.diff_result.layers
            if l.exceeds_tolerance or l.shape_a != l.shape_b
        ]
        if bad_layers:
            lines.append("  Problematic layers:")
            for layer in bad_layers[:3]:
                status = "SHAPE MISMATCH" if layer.shape_a != layer.shape_b else "EXCEEDS"
                lines.append(f"    {layer.name}: {layer.max_diff:.2e} [{status}]")
            if len(bad_layers) > 3:
                lines.append(f"    ... ({len(bad_layers) - 3} more)")

        return "\n".join(lines)


class BoundsError(AssertionError):
    """Bounds check failed.

    Raised when `assert_bounds()` fails. Contains information about
    which layers have bounds that are too wide.

    Attributes:
        result: The ProfileResult or sensitivity result
        max_width: The maximum allowed bound width
        violations: List of (layer_name, bound_width) tuples that exceeded max_width
    """

    def __init__(
        self,
        result: Any,
        max_width: float,
        violations: list[tuple[str, float]],
    ) -> None:
        self.result = result
        self.max_width = max_width
        self.violations = violations
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = ["Bounds check FAILED:"]
        lines.append(f"  Max allowed width: {self.max_width:.2e}")
        lines.append(f"  Violations found: {len(self.violations)}")

        lines.append("  Offending layers:")
        for name, width in self.violations[:5]:
            lines.append(f"    {name}: width = {width:.2e}")
        if len(self.violations) > 5:
            lines.append(f"    ... ({len(self.violations) - 5} more)")

        return "\n".join(lines)


class QuantizationError(AssertionError):
    """Quantization safety check failed.

    Raised when `assert_quantization_safe()` fails. Contains information
    about which layers have potential overflow issues.

    Attributes:
        result: The QuantizeResult from ny.quantize_check()
        dtype: The target quantization dtype
    """

    def __init__(self, result: Any, dtype: str) -> None:
        self.result = result
        self.dtype = dtype
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = [f"Quantization safety FAILED for {self.dtype}:"]

        overflow_layers = [
            layer for layer in self.result.layers
            if getattr(layer, "has_overflow", False)
        ]

        if overflow_layers:
            lines.append(f"  Layers with overflow risk: {len(overflow_layers)}")
            for layer in overflow_layers[:5]:
                max_val = getattr(layer, "max_value", float("nan"))
                lines.append(f"    {layer.name}: max={max_val:.2e}")
            if len(overflow_layers) > 5:
                lines.append(f"    ... ({len(overflow_layers) - 5} more)")

        if hasattr(self.result, "summary") and self.result.summary:
            lines.append(f"  Summary: {self.result.summary}")

        return "\n".join(lines)
