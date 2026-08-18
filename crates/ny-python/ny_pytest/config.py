# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""
Configuration for ny pytest plugin.

Handles loading configuration from pytest.ini, command line options,
and markers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal


def _finite_nonnegative_float(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number, got {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return parsed


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from error
    if parsed <= 0 or str(value).strip() != str(parsed):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


@dataclass
class NyConfig:
    """Configuration for ny verification tests.

    This class holds all configuration options for running verification
    tests. Configuration can come from:
    - pytest.ini file
    - Command-line options (--ny-*)
    - Test markers (@pytest.mark.ny_verify)
    - Direct instantiation

    Attributes:
        epsilon: Perturbation epsilon for verification (default: 0.01)
        method: Verification method: ibp, crown, alpha, beta (default: "crown")
        timeout: Verification timeout in seconds (default: 60)
        tolerance: Tolerance for equivalence checks (default: 1e-5)
        max_width: Maximum allowed bound width (default: 1e6)
        continue_on_failure: Continue verification after first failure (default: True)
    """

    epsilon: float = 0.01
    method: Literal["ibp", "crown", "alpha", "beta"] = "crown"
    timeout: int = 60
    tolerance: float = 1e-5
    max_width: float = 1e6
    continue_on_failure: bool = True

    @classmethod
    def from_pytest(cls, request: Any) -> NyConfig:
        """Create configuration from pytest request object.

        Reads configuration from pytest.ini and command-line options,
        then overrides with any marker-specific settings.

        Args:
            request: pytest request fixture

        Returns:
            NyConfig with merged settings

        Example:
            >>> @pytest.fixture
            ... def ny_config(request):
            ...     return NyConfig.from_pytest(request)
        """
        config = cls()

        # Try to read from pytest config
        if hasattr(request, "config"):
            ini_config = request.config

            # Read from pytest.ini
            if hasattr(ini_config, "getini"):
                epsilon = ini_config.getini("ny_epsilon")
                if epsilon not in (None, ""):
                    config.epsilon = _finite_nonnegative_float(
                        "ny_epsilon", epsilon
                    )

                method = ini_config.getini("ny_method")
                if method not in (None, ""):
                    if method not in ("ibp", "crown", "alpha", "beta"):
                        raise ValueError(
                            "ny_method must be one of ibp, crown, alpha, or beta, "
                            f"got {method!r}"
                        )
                    config.method = method

                timeout = ini_config.getini("ny_timeout")
                if timeout not in (None, ""):
                    config.timeout = _positive_int("ny_timeout", timeout)

                tolerance = ini_config.getini("ny_tolerance")
                if tolerance not in (None, ""):
                    config.tolerance = _finite_nonnegative_float(
                        "ny_tolerance", tolerance
                    )

            # Read from command-line options
            if hasattr(ini_config, "getoption"):
                epsilon = ini_config.getoption("--ny-epsilon", default=None)
                if epsilon is not None:
                    config.epsilon = _finite_nonnegative_float(
                        "--ny-epsilon", epsilon
                    )
                method = ini_config.getoption("--ny-method", default=None)
                if method is not None:
                    if method not in ("ibp", "crown", "alpha", "beta"):
                        raise ValueError(
                            "--ny-method must be one of ibp, crown, alpha, or beta, "
                            f"got {method!r}"
                        )
                    config.method = method
                timeout = ini_config.getoption("--ny-timeout", default=None)
                if timeout is not None:
                    config.timeout = _positive_int("--ny-timeout", timeout)

        # Override with marker settings
        if hasattr(request, "node"):
            marker = request.node.get_closest_marker("ny_verify")
            if marker:
                if "epsilon" in marker.kwargs:
                    config.epsilon = _finite_nonnegative_float(
                        "ny_verify epsilon", marker.kwargs["epsilon"]
                    )
                if "method" in marker.kwargs:
                    method = marker.kwargs["method"]
                    if method not in ("ibp", "crown", "alpha", "beta"):
                        raise ValueError(
                            "ny_verify method must be one of ibp, crown, alpha, "
                            f"or beta, got {method!r}"
                        )
                    config.method = method
                if "timeout" in marker.kwargs:
                    config.timeout = _positive_int(
                        "ny_verify timeout", marker.kwargs["timeout"]
                    )

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for passing to ny functions."""
        return {
            "epsilon": self.epsilon,
            "method": self.method,
            "timeout": self.timeout,
        }
