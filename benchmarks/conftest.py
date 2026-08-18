# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

"""Pytest controls shared by the explicit benchmark diagnostics."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ny-benchmark")
    group.addoption(
        "--ny-benchmark-timeout",
        type=int,
        default=10,
        help="per-instance timeout for explicit NY benchmark diagnostics",
    )
    group.addoption(
        "--ny-benchmark-method",
        choices=("ibp", "crown", "alpha", "beta"),
        default="crown",
        help="verification method for explicit NY benchmark diagnostics",
    )
    group.addoption(
        "--ny-benchmark-results",
        help="optional JSON output path for explicit benchmark diagnostics",
    )


@pytest.fixture
def verify_timeout(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--ny-benchmark-timeout")


@pytest.fixture
def method(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--ny-benchmark-method")
