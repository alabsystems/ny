# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""
Configuration for ny pytest plugin.

Handles loading configuration from pytest.ini, command line options,
and markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


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
                if ini_config.getini("ny_epsilon"):
                    try:
                        config.epsilon = float(ini_config.getini("ny_epsilon"))
                    except (TypeError, ValueError):
                        pass

                if ini_config.getini("ny_method"):
                    method = ini_config.getini("ny_method")
                    if method in ("ibp", "crown", "alpha", "beta"):
                        config.method = method

                if ini_config.getini("ny_timeout"):
                    try:
                        config.timeout = int(ini_config.getini("ny_timeout"))
                    except (TypeError, ValueError):
                        pass

                if ini_config.getini("ny_tolerance"):
                    try:
                        config.tolerance = float(ini_config.getini("ny_tolerance"))
                    except (TypeError, ValueError):
                        pass

            # Read from command-line options
            if hasattr(ini_config, "getoption"):
                if ini_config.getoption("--ny-epsilon", default=None):
                    config.epsilon = ini_config.getoption("--ny-epsilon")
                if ini_config.getoption("--ny-method", default=None):
                    config.method = ini_config.getoption("--ny-method")
                if ini_config.getoption("--ny-timeout", default=None):
                    config.timeout = ini_config.getoption("--ny-timeout")

        # Override with marker settings
        if hasattr(request, "node"):
            marker = request.node.get_closest_marker("ny_verify")
            if marker:
                if "epsilon" in marker.kwargs:
                    config.epsilon = marker.kwargs["epsilon"]
                if "method" in marker.kwargs:
                    config.method = marker.kwargs["method"]
                if "timeout" in marker.kwargs:
                    config.timeout = marker.kwargs["timeout"]

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for passing to ny functions."""
        return {
            "epsilon": self.epsilon,
            "method": self.method,
            "timeout": self.timeout,
        }
