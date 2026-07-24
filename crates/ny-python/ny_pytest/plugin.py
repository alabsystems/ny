# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""
pytest plugin hooks for ny neural network verification.

This module registers ny-specific command-line options, markers,
and fixtures with pytest.

To use, either:
1. Add 'ny_pytest.plugin' to pytest_plugins in conftest.py
2. Install ny[pytest] which auto-registers via entry points
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ny_pytest.config import NyConfig

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


def pytest_addoption(parser: Parser) -> None:
    """Add ny-specific command line and ini options."""
    group = parser.getgroup("ny", "ny neural network verification")

    group.addoption(
        "--ny-timeout",
        action="store",
        default=60,
        type=int,
        help="Timeout for verification in seconds (default: 60)",
    )

    group.addoption(
        "--ny-epsilon",
        action="store",
        default=0.01,
        type=float,
        help="Default perturbation epsilon (default: 0.01)",
    )

    group.addoption(
        "--ny-method",
        action="store",
        default="crown",
        choices=["ibp", "crown", "alpha", "beta"],
        help="Verification method (default: crown)",
    )

    # Register ini options for pyproject.toml / pytest.ini
    parser.addini("ny_epsilon", "Default perturbation epsilon", default="0.01")
    parser.addini("ny_method", "Verification method (ibp, crown, alpha, beta)", default="crown")
    parser.addini("ny_timeout", "Verification timeout in seconds", default="60")
    parser.addini("ny_tolerance", "Tolerance for equivalence checks", default="1e-5")


def pytest_configure(config: Config) -> None:
    """Register ny markers."""
    config.addinivalue_line(
        "markers",
        "ny_verify(epsilon, method, timeout): mark test as neural network verification",
    )
    config.addinivalue_line(
        "markers",
        "ny_diff(tolerance): mark test as model comparison",
    )
    config.addinivalue_line(
        "markers",
        "ny_slow: mark test as slow verification (skip with -m 'not ny_slow')",
    )


@pytest.fixture
def ny_config(request: pytest.FixtureRequest) -> NyConfig:
    """Fixture providing ny configuration.

    Merges configuration from:
    1. pytest.ini defaults
    2. Command-line options (--ny-*)
    3. Test markers (@pytest.mark.ny_verify)

    Example:
        >>> def test_verification(ny_config):
        ...     print(f"Using method: {ny_config.method}")
        ...     print(f"Epsilon: {ny_config.epsilon}")
    """
    return NyConfig.from_pytest(request)


@pytest.fixture
def ny_model(request: pytest.FixtureRequest):
    """Fixture for loading models with verification config.

    Use with @pytest.mark.parametrize to test multiple models:

    Example:
        >>> @pytest.mark.parametrize("ny_model", [
        ...     "models/whisper-tiny.onnx",
        ...     "models/whisper-small.onnx",
        ... ], indirect=True)
        ... def test_models_robust(ny_model):
        ...     result = ny_model.verify(epsilon=0.001, output_bounds=[(-1.0, 1.0)] * 10)
        ...     assert_verified(result)
    """
    try:
        import ny  # noqa: F401
    except ImportError as e:
        pytest.fail(f"ny module not available: {e}")

    model_path = request.param
    # Return a simple wrapper that holds the path and provides verify/diff
    return _NyModelWrapper(model_path)


class _NyModelWrapper:
    """Wrapper providing convenient verify/diff methods for a model path."""

    def __init__(self, path: str) -> None:
        self.path = path

    def verify(
        self,
        epsilon: float = 0.01,
        method: str = "crown",
        timeout: int = 60,
        output_bounds=None,
    ):
        """Verify the model with given parameters.

        output_bounds is the property to check: one (lower, upper) requirement
        per model output. Without it, no property is checked and the result
        status is Unknown with the certified bounds attached.
        """
        import ny
        return ny.verify(
            self.path,
            epsilon=epsilon,
            method=method,
            timeout=timeout,
            output_bounds=output_bounds,
        )

    def diff(
        self,
        other: str,
        tolerance: float = 1e-5,
    ):
        """Compare this model to another."""
        import ny
        return ny.diff(self.path, other, tolerance=tolerance)

    def profile_bounds(self, epsilon: float = 0.01):
        """Profile bounds through the model."""
        import ny
        return ny.profile_bounds(self.path, epsilon=epsilon)

    def sensitivity_analysis(self, epsilon: float = 0.01):
        """Analyze layer sensitivities."""
        import ny
        return ny.sensitivity_analysis(self.path, epsilon=epsilon)

    def quantize_check(self, epsilon: float = 0.01):
        """Check quantization safety."""
        import ny
        return ny.quantize_check(self.path, epsilon=epsilon)

    def __repr__(self) -> str:
        return f"NyModel({self.path!r})"
