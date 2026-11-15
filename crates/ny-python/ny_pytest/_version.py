# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""The plugin's version, derived from Cargo and never written in Python.

The `ny` distribution's version has ONE source: `[workspace.package] version`
in the repository's root Cargo.toml. The ny-python crate inherits it, and
maturin stamps the wheel from the crate (`dynamic = ["version"]` in
pyproject.toml). This module reads that value from wherever the package is
running:

* a source checkout (`crates/ny-python/ny_pytest`): the Cargo manifests
  themselves, so the pure-Python plugin still imports without the compiled
  `ny` extension;
* an installed wheel: the distribution metadata maturin stamped.

A literal used to live in `ny_pytest/__init__.py`. A release bump moves Cargo
manifests only, so it stayed at 0.4.0 when the workspace moved to 0.5.0
(ac4cf1483) and lane [g] of scripts/gate.sh was red for a month. With nothing
to edit here, a bump has nothing to leave behind;
crates/ny-test-utils/tests/python_package_version.rs keeps a literal from
coming back.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

DISTRIBUTION = "ny"
CRATE = "ny-python"
# What `__version__` says when neither source exists (a copied package
# directory with no checkout around it and no installed distribution).
UNKNOWN = "0+unknown"


def _table_value(path: Path, table: str, key: str) -> str | None:
    """One simple `key = value` entry of a named TOML table, comments stripped."""
    active_table = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            active_table = line[1:-1].strip()
            continue
        name, separator, value = line.partition("=")
        if active_table == table and separator and name.strip() == key:
            return value.strip()
    return None


def checkout_version(crate_dir: Path) -> str | None:
    """The Cargo version when `crate_dir` is `crates/ny-python` of a checkout.

    `None` when it is not (an installed wheel's `site-packages`, any other
    project's directory) or when the manifests cannot be read.
    """
    crate_manifest = crate_dir / "Cargo.toml"
    workspace_manifest = crate_dir.parent.parent / "Cargo.toml"
    try:
        if not crate_manifest.is_file():
            return None
        if _table_value(crate_manifest, "package", "name") != f'"{CRATE}"':
            return None
        own = _table_value(crate_manifest, "package", "version")
        if own is not None:
            return own.strip('"') or None
        if _table_value(crate_manifest, "package", "version.workspace") != "true":
            return None
        if not workspace_manifest.is_file():
            return None
        inherited = _table_value(workspace_manifest, "workspace.package", "version")
    except (OSError, UnicodeDecodeError):
        return None
    return None if inherited is None else inherited.strip('"') or None


def resolve_version(crate_dir: Path | None = None) -> str:
    """The version of the distribution this package belongs to."""
    if crate_dir is None:
        crate_dir = Path(__file__).resolve().parent.parent
    version = checkout_version(crate_dir)
    if version is not None:
        return version
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return UNKNOWN
