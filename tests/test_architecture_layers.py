# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Pytest wrapper for the architecture layer guard.

Runs scripts/check_architecture_layers.py and fails if any layer
violations are detected. Part of #2126.

Usage:
    pytest tests/test_architecture_layers.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_architecture_layers.py"


def test_architecture_layers_no_violations():
    """Workspace crate dependencies must conform to the layer policy."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    data = json.loads(result.stdout)
    violations = data.get("violations", [])
    assert data["pass"], (
        f"Architecture layer violations detected ({len(violations)}):\n"
        + "\n".join(v.get("reason", str(v)) for v in violations)
    )
