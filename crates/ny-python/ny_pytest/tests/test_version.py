# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Version-coherence tests for the Python distribution and pytest plugin.

`[workspace.package] version` in the root Cargo.toml is the single source of
truth, and nothing in the Python package repeats it. The wheel takes it from
Cargo (`dynamic = ["version"]`, stamped by maturin) and the pure-Python plugin
derives it (ny_pytest/_version.py: the Cargo manifests in a checkout, the
distribution metadata in an installed wheel). The structure is also enforced
in the default Cargo lane by
crates/ny-test-utils/tests/python_package_version.rs, so a hand-written copy
cannot come back unnoticed.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import ny_pytest
from ny_pytest import __version__, _version

if TYPE_CHECKING:
    import pytest


def _table_lines(path: Path, table: str) -> list[str]:
    """The comment-stripped body lines of one named TOML table."""
    active_table = ""
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            active_table = line[1:-1].strip()
            continue
        if active_table == table and line:
            lines.append(line)
    return lines


def _table_value(path: Path, table: str, key: str) -> str | None:
    """Read one simple `key = value` entry from a named TOML table."""
    for line in _table_lines(path, table):
        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            return value.strip()
    return None


def _checkout(root: Path, crate_package: str, workspace_version: str = "7.8.0") -> Path:
    """A minimal checkout: a root manifest and `crates/ny-python/Cargo.toml`."""
    crate_dir = root / "crates" / "ny-python"
    crate_dir.mkdir(parents=True)
    (root / "Cargo.toml").write_text(
        "[workspace]\n"
        'members = ["crates/ny-python"]\n'
        "\n"
        "[workspace.package]\n"
        f'version = "{workspace_version}"  # the single source\n'
        'edition = "2021"\n',
        encoding="utf-8",
    )
    (crate_dir / "Cargo.toml").write_text(
        f"[package]\n{crate_package}\n", encoding="utf-8"
    )
    return crate_dir


def test_python_versions_match_workspace() -> None:
    """Cargo is the source; the wheel derives from it and so does the plugin."""
    repo_root = Path(__file__).resolve().parents[4]
    workspace_version = _table_value(
        repo_root / "Cargo.toml", "workspace.package", "version"
    )
    assert workspace_version is not None, (
        "Cargo.toml has no [workspace.package] version"
    )
    workspace_version = workspace_version.strip('"')

    pyproject = repo_root / "crates/ny-python/pyproject.toml"
    assert _table_value(pyproject, "project", "version") is None, (
        "pyproject.toml must not pin a static [project] version: it went stale "
        "at 0.4.0 when the workspace moved to 0.5.0 (ac4cf1483)"
    )
    dynamic = _table_value(pyproject, "project", "dynamic")
    assert '"version"' in (dynamic or ""), (
        "pyproject.toml must list `version` in [project].dynamic so maturin "
        "stamps the wheel from Cargo"
    )

    assert __version__ == workspace_version


def test_plugin_carries_no_version_literal() -> None:
    """`__version__` is computed; a literal here is a copy a bump would miss."""
    source = Path(ny_pytest.__file__).read_text(encoding="utf-8")
    assignments = [
        line for line in source.splitlines() if line.startswith("__version__ =")
    ]
    assert assignments == ["__version__ = resolve_version()"]


def test_checkout_version_follows_the_workspace_manifest(tmp_path: Path) -> None:
    crate_dir = _checkout(tmp_path, 'name = "ny-python"\nversion.workspace = true')

    assert _version.checkout_version(crate_dir) == "7.8.0"
    assert _version.resolve_version(crate_dir) == "7.8.0"


def test_checkout_version_honours_a_crate_that_pins_its_own(tmp_path: Path) -> None:
    crate_dir = _checkout(tmp_path, 'name = "ny-python"\nversion = "1.2.0"')

    assert _version.checkout_version(crate_dir) == "1.2.0"


def test_another_projects_manifest_is_not_a_checkout(tmp_path: Path) -> None:
    # An installed plugin whose `site-packages` happens to sit beside some
    # other project's Cargo.toml must not report that project's version.
    crate_dir = _checkout(tmp_path, 'name = "something-else"\nversion.workspace = true')

    assert _version.checkout_version(crate_dir) is None


def test_installed_wheel_reports_the_distribution_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def stamped(distribution: str) -> str:
        asked.append(distribution)
        return "9.9.0"

    monkeypatch.setattr(metadata, "version", stamped)

    assert _version.resolve_version(tmp_path) == "9.9.0"
    assert asked == ["ny"]


def test_a_version_nothing_can_supply_is_marked_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def absent(distribution: str) -> str:
        raise metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(metadata, "version", absent)

    assert _version.resolve_version(tmp_path) == "0+unknown"
