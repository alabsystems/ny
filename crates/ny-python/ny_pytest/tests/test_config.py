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

    @dataclass
    class MockPytestConfig:
        values: dict[str, Any]
        options: dict[str, Any] | None = None

        def getini(self, name: str) -> Any:
            return self.values.get(name)

        def getoption(self, name: str, *, default: Any = None) -> Any:
            return (self.options or {}).get(name, default)

    @dataclass
    class MockRequest:
        config: Any

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

    def test_ini_values_are_used_without_cli_overrides(self) -> None:
        request = self.MockRequest(
            config=self.MockPytestConfig(
                values={
                    "ny_epsilon": "0.125",
                    "ny_method": "alpha",
                    "ny_timeout": "45",
                    "ny_tolerance": "0.0002",
                }
            )
        )

        config = NyConfig.from_pytest(request)

        assert config.epsilon == 0.125
        assert config.method == "alpha"
        assert config.timeout == 45
        assert config.tolerance == 0.0002

    def test_cli_values_override_ini_values(self) -> None:
        request = self.MockRequest(
            config=self.MockPytestConfig(
                values={
                    "ny_epsilon": "0.125",
                    "ny_method": "alpha",
                    "ny_timeout": "45",
                    "ny_tolerance": "0.0002",
                },
                options={
                    "--ny-epsilon": 0.25,
                    "--ny-method": "beta",
                    "--ny-timeout": 90,
                },
            )
        )

        config = NyConfig.from_pytest(request)

        assert config.epsilon == 0.25
        assert config.method == "beta"
        assert config.timeout == 90

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("ny_epsilon", "not-a-number"),
            ("ny_epsilon", "nan"),
            ("ny_method", "heuristic"),
            ("ny_timeout", "1.5"),
            ("ny_timeout", "0"),
            ("ny_tolerance", "-0.1"),
        ],
    )
    def test_invalid_ini_values_fail_closed(self, name: str, value: str) -> None:
        values = {
            "ny_epsilon": "0.01",
            "ny_method": "crown",
            "ny_timeout": "60",
            "ny_tolerance": "1e-5",
        }
        values[name] = value
        request = self.MockRequest(config=self.MockPytestConfig(values=values))

        with pytest.raises(ValueError, match=name):
            NyConfig.from_pytest(request)
