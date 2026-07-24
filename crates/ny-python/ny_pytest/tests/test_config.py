# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""
Tests for ny pytest configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ny_pytest.config import NyConfig


class TestNyConfig:
    """Tests for NyConfig class."""

    def test_default_values(self) -> None:
        """Default values should be set correctly."""
        config = NyConfig()

        assert config.epsilon == 0.01
        assert config.method == "crown"
        assert config.timeout == 60
        assert config.tolerance == 1e-5
        assert config.max_width == 1e6
        assert config.continue_on_failure is True

    def test_custom_values(self) -> None:
        """Custom values should be stored correctly."""
        config = NyConfig(
            epsilon=0.001,
            method="beta",
            timeout=120,
            tolerance=1e-4,
            max_width=1e8,
            continue_on_failure=False,
        )

        assert config.epsilon == 0.001
        assert config.method == "beta"
        assert config.timeout == 120
        assert config.tolerance == 1e-4
        assert config.max_width == 1e8
        assert config.continue_on_failure is False

    def test_to_dict(self) -> None:
        """to_dict should return verification parameters."""
        config = NyConfig(epsilon=0.005, method="alpha", timeout=30)
        d = config.to_dict()

        assert d["epsilon"] == 0.005
        assert d["method"] == "alpha"
        assert d["timeout"] == 30


class TestNyConfigFromPytest:
    """Tests for NyConfig.from_pytest()."""

    def test_no_config_uses_defaults(self) -> None:
        """Missing pytest config should use defaults."""
        # Mock request with no config
        @dataclass
        class MockRequest:
            pass

        request = MockRequest()
        config = NyConfig.from_pytest(request)

        assert config.epsilon == 0.01
        assert config.method == "crown"
        assert config.timeout == 60

    def test_marker_override(self) -> None:
        """Marker settings should override defaults."""
        # Mock marker with kwargs
        @dataclass
        class MockMarker:
            kwargs: dict[str, Any]

        @dataclass
        class MockNode:
            marker: MockMarker | None

            def get_closest_marker(self, name: str) -> MockMarker | None:
                return self.marker if name == "ny_verify" else None

        @dataclass
        class MockRequest:
            node: MockNode

        marker = MockMarker(kwargs={"epsilon": 0.005, "method": "beta", "timeout": 120})
        request = MockRequest(node=MockNode(marker=marker))
        config = NyConfig.from_pytest(request)

        assert config.epsilon == 0.005
        assert config.method == "beta"
        assert config.timeout == 120
