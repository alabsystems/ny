#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Transactionally bump NY's complete AY dependency identity.

AY changes its workspace package version independently of its Git revision.
Both values are therefore mandatory inputs.  This tool updates the only two
authorized manifest legs, refreshes and validates Cargo.lock, aligns the active
scored/hint claims and advice protocol revision, and rolls every tracked surface
back on any failure.

Usage:

    python3 scripts/bump_ay_pin.py REV VERSION
    python3 scripts/bump_ay_pin.py --dry-run REV VERSION
    python3 scripts/bump_ay_pin.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from align_scored_audit_ay_pin import (
    AY_URL,
    FULL_GIT_SHA,
    WORKSPACE_VERSION,
    AyIdentity,
    AyIdentityError,
    align_docs,
    resolved_ay_identity,
)
from git_dep_pins import GitDependencyPinError, workspace_git_dependency_pins

DEFAULT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AyLeg:
    manifest: Path
    dependency: str


EXPECTED_LEGS = (
    AyLeg(Path("crates/ny-mip/Cargo.toml"), "ay-milp"),
    AyLeg(Path("crates/ny-cli/Cargo.toml"), "ay-sat"),
)
ADVICE_SOURCE = Path("crates/ny-mip/src/head_phase_ay.rs")
TRANSACTION_PATHS = (
    *(leg.manifest for leg in EXPECTED_LEGS),
    ADVICE_SOURCE,
    Path("Cargo.lock"),
    Path("docs/SCORED_REPRO_AUDIT_2026-07-19.md"),
    Path("docs/AY_BRANCH_HINT_CANARY.md"),
)


class AyBumpError(RuntimeError):
    """Raised when the AY bump cannot establish every postcondition."""


def _advice_revision(text: str) -> re.Match[str]:
    declarations = list(
        re.finditer(r"^[ \t]*const[ \t]+AY_REVISION\b[^\n]*$", text, re.MULTILINE)
    )
    if len(declarations) != 1:
        raise AyBumpError("advice source requires exactly one AY_REVISION declaration")
    declaration = declarations[0]
    if re.fullmatch(r'const AY_REVISION: &str = "[0-9a-f]{40}";', declaration[0]) is None:
        raise AyBumpError("advice AY_REVISION must be a full lowercase Git SHA")
    return declaration


def _updated_advice(path: Path, target: AyIdentity) -> str:
    text = path.read_text(encoding="utf-8")
    declaration = _advice_revision(text)
    return (
        text[: declaration.start()]
        + f'const AY_REVISION: &str = "{target.revision}";'
        + text[declaration.end() :]
    )


def _atomic_write(path: Path, contents: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dependency_body(text: str, dependency: str) -> re.Match[str]:
    section_match = re.search(r"(?m)^\[dependencies\]\s*(?:#.*)?$", text)
    if section_match is None:
        raise AyBumpError("manifest has no [dependencies] section")
    next_section = re.search(r"(?m)^\[[^]]+\]\s*(?:#.*)?$", text[section_match.end() :])
    section_end = (
        section_match.end() + next_section.start() if next_section is not None else len(text)
    )
    body = text[section_match.end() : section_end]
    declaration = re.compile(
        rf"(?m)^(?P<prefix>\s*{re.escape(dependency)}\s*=\s*\{{)"
        r"(?P<body>[^}\n]*)(?P<suffix>\}\s*(?:#.*)?$)"
    )
    matches = list(declaration.finditer(body))
    if len(matches) != 1:
        raise AyBumpError(
            f"expected exactly one inline [dependencies].{dependency} declaration"
        )
    match = matches[0]
    # Return a match over the complete manifest so replacement offsets are safe.
    absolute = declaration.search(text, section_match.end(), section_end)
    assert absolute is not None
    return absolute


def _manifest_leg_identity(path: Path, dependency: str) -> AyIdentity:
    text = path.read_text(encoding="utf-8")
    match = _dependency_body(text, dependency)
    body = match.group("body")
    git_values = re.findall(r'\bgit\s*=\s*"([^"]+)"', body)
    rev_values = re.findall(r'\brev\s*=\s*"([^"]+)"', body)
    version_values = re.findall(r'\bversion\s*=\s*"([^"]+)"', body)
    if git_values != [AY_URL]:
        raise AyBumpError(f"{path}: {dependency} must use canonical AY Git URL")
    if len(rev_values) != 1 or FULL_GIT_SHA.fullmatch(rev_values[0]) is None:
        raise AyBumpError(f"{path}: {dependency} must use one full lowercase SHA")
    if (
        len(version_values) != 1
        or WORKSPACE_VERSION.fullmatch(version_values[0]) is None
    ):
        raise AyBumpError(
            f"{path}: {dependency} must use one explicit X.Y.Z workspace version"
        )
    return AyIdentity(rev_values[0], version_values[0])


def _updated_manifest(path: Path, dependency: str, target: AyIdentity) -> str:
    text = path.read_text(encoding="utf-8")
    match = _dependency_body(text, dependency)
    body = match.group("body")
    updated, rev_count = re.subn(
        r'(?P<prefix>\brev\s*=\s*")[^"]+(?P<suffix>")',
        rf"\g<prefix>{target.revision}\g<suffix>",
        body,
    )
    updated, version_count = re.subn(
        r'(?P<prefix>\bversion\s*=\s*")[^"]+(?P<suffix>")',
        rf"\g<prefix>{target.version}\g<suffix>",
        updated,
    )
    if rev_count != 1 or version_count != 1:
        raise AyBumpError(
            f"{path}: could not uniquely rewrite {dependency}'s rev and version"
        )
    return text[: match.start("body")] + updated + text[match.end("body") :]


def _validate_manifest_authority(root: Path, target: AyIdentity | None = None) -> AyIdentity:
    root_manifest = root / "Cargo.toml"
    pins = [
        pin
        for pin in workspace_git_dependency_pins(root_manifest)
        if pin.name == "ay"
        or pin.name.startswith("ay-")
        or "alabsystems/ay" in pin.git_url.lower()
    ]
    noncanonical_urls = sorted({pin.git_url for pin in pins if pin.git_url != AY_URL})
    if noncanonical_urls:
        raise AyBumpError(f"non-canonical AY Git URLs: {noncanonical_urls!r}")
    observed_pin_legs = {
        (pin.manifest.relative_to(root), pin.section, pin.name) for pin in pins
    }
    expected_pin_legs = {
        (leg.manifest, ("dependencies",), leg.dependency) for leg in EXPECTED_LEGS
    }
    if observed_pin_legs != expected_pin_legs:
        raise AyBumpError(
            "AY dependency authority changed; expected only ny-mip/ay-milp and "
            f"ny-cli/ay-sat, got {sorted(observed_pin_legs)!r}"
        )

    identities = {
        _manifest_leg_identity(root / leg.manifest, leg.dependency)
        for leg in EXPECTED_LEGS
    }
    if len(identities) != 1:
        split = sorted((identity.revision, identity.version) for identity in identities)
        raise AyBumpError(f"manifest AY identity is split: {split!r}")
    identity = next(iter(identities))
    if target is not None and identity != target:
        raise AyBumpError(
            f"manifest AY identity {identity!r} does not match target {target!r}"
        )
    return identity


def validate_repository(root: Path, target: AyIdentity | None = None) -> AyIdentity:
    manifest_identity = _validate_manifest_authority(root, target)
    lock_identity = resolved_ay_identity(root)
    if manifest_identity != lock_identity:
        raise AyBumpError(
            f"manifest AY identity {manifest_identity!r} does not match "
            f"Cargo.lock {lock_identity!r}"
        )
    advice = (root / ADVICE_SOURCE).read_text(encoding="utf-8")
    if _advice_revision(advice)[0] != f'const AY_REVISION: &str = "{lock_identity.revision}";':
        raise AyBumpError("advice AY_REVISION does not match Cargo.lock")
    identity, stale_docs = align_docs(root, check=True)
    if identity != lock_identity:
        raise AyBumpError("AY document aligner did not derive the locked identity")
    if stale_docs:
        stale = ", ".join(path.relative_to(root).as_posix() for path in stale_docs)
        raise AyBumpError(f"active AY identity documents are stale: {stale}")
    return lock_identity


def _run_cargo(root: Path, arguments: list[str], *, quiet_stdout: bool = False) -> None:
    # `$CARGO` when run under targo, else targo itself (owner directive
    # 2026-09-28: the Trust toolchain; no stock fallback). Metadata resolution
    # does not compile and needs no lane.
    cargo = os.environ.get("CARGO", "targo")
    try:
        completed = subprocess.run(
            [cargo, *arguments],
            cwd=root,
            check=False,
            stdout=subprocess.DEVNULL if quiet_stdout else None,
        )
    except OSError as error:
        raise AyBumpError(f"could not run Cargo: {error}") from error
    if completed.returncode != 0:
        raise AyBumpError(
            f"Cargo command failed with exit {completed.returncode}: "
            f"{cargo} {' '.join(arguments)}"
        )


def bump_ay(root: Path, target: AyIdentity, *, dry_run: bool) -> tuple[Path, ...]:
    manifest_identity = _validate_manifest_authority(root)
    updates = {
        root / leg.manifest: _updated_manifest(
            root / leg.manifest, leg.dependency, target
        ).encode("utf-8")
        for leg in EXPECTED_LEGS
    }
    changed_manifests = tuple(
        path for path, contents in updates.items() if path.read_bytes() != contents
    )
    advice_path = root / ADVICE_SOURCE
    updates[advice_path] = _updated_advice(advice_path, target).encode("utf-8")
    changed_inputs = tuple(
        path for path, contents in updates.items() if path.read_bytes() != contents
    )
    if dry_run:
        lock_identity = resolved_ay_identity(root)
        _doc_identity, stale_docs = align_docs(root, check=True)
        if target != manifest_identity or target != lock_identity:
            return (
                *changed_inputs,
                root / "Cargo.lock",
                root / "docs/SCORED_REPRO_AUDIT_2026-07-19.md",
                root / "docs/AY_BRANCH_HINT_CANARY.md",
            )
        return (*changed_inputs, *stale_docs)

    transaction_paths = tuple(root / relative for relative in TRANSACTION_PATHS)
    snapshots = {path: path.read_bytes() for path in transaction_paths}
    try:
        for path in changed_inputs:
            _atomic_write(path, updates[path])

        current_lock = resolved_ay_identity(root)
        if current_lock != target or changed_manifests:
            # Every manifest leg already names the immutable target revision.
            # Let Cargo resolve that change conservatively. Explicit update -p
            # can reselect unrelated broad-range registry dependencies (observed
            # Windows edges), while full metadata preserves existing lock choices.
            _run_cargo(
                root,
                ["metadata", "--format-version", "1"],
                quiet_stdout=True,
            )
        _run_cargo(
            root,
            ["metadata", "--locked", "--format-version", "1"],
            quiet_stdout=True,
        )

        lock_identity = resolved_ay_identity(root)
        if lock_identity != target:
            raise AyBumpError(
                f"Cargo.lock resolved AY as {lock_identity!r}, expected {target!r}"
            )
        align_docs(root, check=False)
        validate_repository(root, target)
    except BaseException:
        rollback_failures: list[str] = []
        for path, contents in snapshots.items():
            try:
                if not path.exists() or path.read_bytes() != contents:
                    _atomic_write(path, contents)
            except OSError as error:
                rollback_failures.append(f"{path}: {error}")
        if rollback_failures:
            raise AyBumpError(
                "AY bump failed and rollback was incomplete: "
                + "; ".join(rollback_failures)
            )
        raise

    return tuple(
        path for path in transaction_paths if path.read_bytes() != snapshots[path]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="NY repository root (defaults to the script's repository)",
    )
    parser.add_argument("--dry-run", action="store_true", help="write nothing")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate manifest/lock/docs/advice identity; takes no REV/VERSION",
    )
    parser.add_argument("revision", nargs="?")
    parser.add_argument("version", nargs="?")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.check:
            if args.dry_run or args.revision is not None or args.version is not None:
                raise AyBumpError("--check takes no revision, version, or --dry-run")
            identity = validate_repository(root)
            print(f"AY identity coherent: {identity.revision}/{identity.version}")
            return 0

        if args.revision is None or args.version is None:
            raise AyBumpError("REV and VERSION are both required")
        if FULL_GIT_SHA.fullmatch(args.revision) is None:
            raise AyBumpError(
                "revision must be a full lowercase 40-character Git SHA"
            )
        if WORKSPACE_VERSION.fullmatch(args.version) is None:
            raise AyBumpError("version must have canonical X.Y.Z form")
        target = AyIdentity(args.revision, args.version)
        changed = bump_ay(root, target, dry_run=args.dry_run)
        action = "Would update" if args.dry_run else "Updated"
        for path in changed:
            print(f"{action} {path.relative_to(root).as_posix()}")
        if not changed:
            print(f"Already current: AY {target.revision}/{target.version}")
        return 0
    except (
        AyBumpError,
        AyIdentityError,
        GitDependencyPinError,
        OSError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
