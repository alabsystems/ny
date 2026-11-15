#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Align active AY claims with the exact revision *and version* in Cargo.lock.

The scored-path audit and branch-hint canary are active dependency-identity
surfaces.  A revision-only rewrite is insufficient when AY advances its
workspace version, because Cargo can otherwise resolve a split revision/version
graph while the prose still appears current.

Usage:

    python3 scripts/align_scored_audit_ay_pin.py [--check]

``--check`` is read-only and fails if Cargo.lock itself is split or either
active document does not name the single locked AY identity.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


DEFAULT_ROOT = Path(__file__).resolve().parent.parent
AY_URL = "https://github.com/alabsystems/ay.git"
FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}")
WORKSPACE_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


class AyIdentityError(ValueError):
    """Raised when an AY identity surface is absent, split, or malformed."""


@dataclass(frozen=True)
class AyIdentity:
    revision: str
    version: str


def _is_ay_workspace_package(package: dict[object, object]) -> bool:
    name = str(package.get("name", ""))
    return name == "ay" or name.startswith("ay-")


def _is_ay_package(package: dict[object, object]) -> bool:
    source = str(package.get("source", ""))
    return _is_ay_workspace_package(package) or "alabsystems/ay" in source.lower()


def resolved_ay_identity(root: Path = DEFAULT_ROOT) -> AyIdentity:
    """Return the one AY source revision and AY workspace-package version.

    Independently versioned vendor packages exported by that source still
    participate in revision unification, without defining the workspace version.
    """

    lock_path = root / "Cargo.lock"
    lock_data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = [
        package
        for package in lock_data.get("package", [])
        if isinstance(package, dict) and _is_ay_package(package)
    ]
    if not packages:
        raise AyIdentityError("Cargo.lock contains no AY packages")

    source_pattern = re.compile(
        rf"^git\+{re.escape(AY_URL)}\?rev=([0-9a-f]{{40}})#([0-9a-f]{{40}})$"
    )
    pairs: set[tuple[str, str]] = set()
    versions: set[str] = set()
    for package in packages:
        source = str(package.get("source", ""))
        match = source_pattern.fullmatch(source)
        if match is None:
            raise AyIdentityError(
                "Cargo.lock contains a non-canonical AY source for "
                f"{package.get('name')!r}: {source!r}"
            )
        pairs.add(match.groups())
        # A source export can include independently versioned vendor crates
        # (for example trust-exec-policy). They must share the immutable AY
        # revision above, but their package version is not AY's workspace
        # version. Cargo validates each vendor's SemVer during the transaction.
        # Never let one supply or mask the AY workspace identity.
        if _is_ay_workspace_package(package):
            version = str(package.get("version", ""))
            if WORKSPACE_VERSION.fullmatch(version) is None:
                raise AyIdentityError(
                    f"Cargo.lock contains an invalid AY version for "
                    f"{package.get('name')!r}: {version!r}"
                )
            versions.add(version)

    if not versions:
        raise AyIdentityError("Cargo.lock contains no AY workspace packages")
    if len(pairs) != 1:
        raise AyIdentityError(f"Cargo.lock contains split AY revisions: {sorted(pairs)!r}")
    if len(versions) != 1:
        raise AyIdentityError(f"Cargo.lock contains split AY versions: {sorted(versions)!r}")
    requested, resolved = next(iter(pairs))
    if requested != resolved:
        raise AyIdentityError(
            f"Cargo.lock AY request {requested} resolved to different commit {resolved}"
        )
    return AyIdentity(resolved, next(iter(versions)))


def _replace_required(
    text: str,
    pattern: str,
    replacement,
    *,
    label: str,
) -> str:
    updated, count = re.subn(pattern, replacement, text)
    if count == 0:
        raise AyIdentityError(f"active AY identity pattern is missing: {label}")
    return updated


def aligned_doc_texts(root: Path, identity: AyIdentity) -> dict[Path, str]:
    """Return both active documents rewritten to ``identity`` in memory."""

    scored_audit = root / "docs" / "SCORED_REPRO_AUDIT_2026-07-19.md"
    branch_hint = root / "docs" / "AY_BRANCH_HINT_CANARY.md"
    scored = scored_audit.read_text(encoding="utf-8")
    hint = branch_hint.read_text(encoding="utf-8")

    scored = _replace_required(
        scored,
        r"(revision-pinned\s+to AY at\s+`)[0-9a-f]{40}(`)",
        lambda match: f"{match.group(1)}{identity.revision}{match.group(2)}",
        label="scored audit full revision",
    )
    scored = _replace_required(
        scored,
        r"((?:revision-pinned to|Exact Git-pinned) AY `)[0-9a-f]{8}(`)",
        lambda match: f"{match.group(1)}{identity.revision[:8]}{match.group(2)}",
        label="scored audit abbreviated revision",
    )
    scored = _replace_required(
        scored,
        r"(with every selected AY package\s+at workspace version )[0-9]+\.[0-9]+\.[0-9]+(\.)",
        lambda match: f"{match.group(1)}{identity.version}{match.group(2)}",
        label="scored audit workspace version",
    )

    hint = _replace_required(
        hint,
        r"(`ay-milp` Git dependency at\s+`)[0-9a-f]{40}(`)",
        lambda match: f"{match.group(1)}{identity.revision}{match.group(2)}",
        label="branch-hint full revision",
    )
    hint = _replace_required(
        hint,
        r"(claim for the current AY )[0-9]+\.[0-9]+\.[0-9]+( dependency)",
        lambda match: f"{match.group(1)}{identity.version}{match.group(2)}",
        label="branch-hint workspace version",
    )
    return {scored_audit: scored, branch_hint: hint}


def _atomic_write(path: Path, contents: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def align_docs(root: Path, *, check: bool) -> tuple[AyIdentity, tuple[Path, ...]]:
    identity = resolved_ay_identity(root)
    rendered = aligned_doc_texts(root, identity)
    stale = tuple(
        path
        for path, contents in rendered.items()
        if path.read_text(encoding="utf-8") != contents
    )
    if check or not stale:
        return identity, stale
    for path in stale:
        _atomic_write(path, rendered[path])
    return identity, stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="NY repository root (defaults to the script's repository)",
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if stale; write nothing"
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        identity, stale = align_docs(root, check=args.check)
    except (AyIdentityError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if stale and args.check:
        paths = ", ".join(path.relative_to(root).as_posix() for path in stale)
        print(
            f"STALE: {paths} does not name AY "
            f"{identity.revision[:8]}/{identity.version}"
        )
        return 1
    if stale:
        print(
            f"aligned {len(stale)} docs to AY "
            f"{identity.revision[:8]}/{identity.version}"
        )
    else:
        print(f"already aligned to AY {identity.revision[:8]}/{identity.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
