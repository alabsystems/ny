# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Version-coherence tests for the Python distribution and pytest plugin."""

from __future__ import annotations

from pathlib import Path

from ny_pytest import __version__


def _table_version(path: Path, table: str) -> str:
    """Read a simple string version from one named TOML table."""
    active_table = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            active_table = line[1:-1].strip()
            continue
        if active_table == table and line.startswith("version"):
            key, separator, value = line.partition("=")
            if separator and key.strip() == "version":
                return value.strip().strip('"')
    raise AssertionError(f"{path} has no version in [{table}]")


def test_python_versions_match_workspace() -> None:
    """Cargo, wheel metadata, and the bundled plugin must advance together."""
    repo_root = Path(__file__).resolve().parents[4]
    workspace_version = _table_version(repo_root / "Cargo.toml", "workspace.package")
    wheel_version = _table_version(
        repo_root / "crates/ny-python/pyproject.toml",
        "project",
    )

    assert workspace_version == wheel_version == __version__
