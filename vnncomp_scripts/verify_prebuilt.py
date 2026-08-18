#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed verifier for the packaged VNN-COMP Linux prebuilt.

The Rust packager performs the same checks before creating a submission.  This
small, standard-library-only verifier repeats them on the evaluation machine so
the installer never treats a checksum-only archive as proven release output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import lzma
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

ARCHIVE_RELATIVE = Path("dist/bin/ny-x86_64-linux.xz")
CHECKSUM_RELATIVE = Path("dist/bin/ny-x86_64-linux.xz.sha256")
PROVENANCE_RELATIVE = Path("dist/bin/ny-x86_64-linux.provenance.txt")
BUILDER_RELATIVE = Path("scripts/vnncomp_trust_linux_build.sh")
PREBUILT_FILES = {
    ARCHIVE_RELATIVE.as_posix(),
    CHECKSUM_RELATIVE.as_posix(),
    PROVENANCE_RELATIVE.as_posix(),
}

SCHEMA = "ny-vnncomp-prebuilt-v1"
TARGET = "x86_64-unknown-linux-gnu"
FEATURES = "mip,cuda"
RECEIPT_SCHEMA = "ny-trust-gate-receipt-v1"
BUILD_PROVENANCE_PREFIX = b"ny.vnncomp.build.v1|"
TRUST_GATE_COMMANDS_V1 = (
    "trust-types:json_digest\n"
    "trust-clean:instantiator_ordering\n"
    "trust-router:production_\n"
    "rust-ui:valtree-node-limit-unit-enum-array\n"
    "rust-ui:clean-island-collapsed-order\n"
    "check_all:check\n"
    "e2e_targo_trust_cli\n"
    "trust_falsification_gate\n"
    "targo-trust:version\n"
    "targo-trust:doctor-json\n"
)
MANIFEST_KEYS = {
    "schema",
    "target",
    "features",
    "trust_commit",
    "trust_bootstrap_mode",
    "trust_gate_status",
    "trust_gate_receipt_sha256",
    "trust_gate_commands_sha256",
    "trust_gate_log_sha256",
    "trustc_sha256",
    "trustc_version_sha256",
    "ny_commit",
    "cargo_lock_sha256",
    "ay_lock_commit",
    "builder_script_sha256",
    "onnxruntime_static_sha256",
    "binary_sha256",
    "package_sha256",
}
HASH_KEYS = {
    "trust_gate_receipt_sha256",
    "trust_gate_commands_sha256",
    "trust_gate_log_sha256",
    "trustc_sha256",
    "trustc_version_sha256",
    "cargo_lock_sha256",
    "builder_script_sha256",
    "onnxruntime_static_sha256",
    "binary_sha256",
    "package_sha256",
}
PACKAGE_ROOTS = (
    ".cargo",
    "benchmarks/download_benchmarks.sh",
    "Cargo.lock",
    "Cargo.toml",
    "dist",
    "LICENSE",
    "README.md",
    "_typos.toml",
    "clippy.toml",
    "configs",
    "crates",
    "install_tool.sh",
    "prepare_instance.sh",
    "requirements.txt",
    "rust-toolchain.toml",
    "run_instance.sh",
    "scripts/vnncomp_coverage.py",
    "vnncomp_scripts",
)

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_BINARY_BYTES = 512 * 1024 * 1024
MAX_CHECKSUM_BYTES = 1024
MAX_PROVENANCE_BYTES = 16 * 1024
LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}", re.ASCII)
LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
AY_SOURCE = re.compile(
    r'^source = "git\+https://github\.com/alabsystems/ay\.git'
    r'\?rev=([0-9a-f]{40})#([0-9a-f]{40})"$',
    re.ASCII,
)
CANONICAL_SEAL = re.compile(
    rb"ny\.vnncomp\.build\.v1\|status=sealed\|"
    rb"target=x86_64-unknown-linux-gnu\|features=mip,cuda\|"
    rb"ny_commit=[0-9a-f]{40}\|cargo_lock_sha256=[0-9a-f]{64}\|"
    rb"ay_commit=[0-9a-f]{40}\|builder_script_sha256=[0-9a-f]{64}\|"
    rb"trust_commit=[0-9a-f]{40}\|trustc_sha256=[0-9a-f]{64}\|"
    rb"trustc_version_sha256=[0-9a-f]{64}\|"
    rb"trust_gate_receipt_sha256=[0-9a-f]{64}\|"
    rb"onnxruntime_static_sha256=[0-9a-f]{64}\|",
    re.ASCII,
)
AY_BUILD_COMMIT = re.compile(rb"build\.commit=([0-9a-f]{40})(-dirty)?", re.ASCII)


class VerificationError(RuntimeError):
    """A prebuilt failed one of the release-admission invariants."""


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def read_regular_file(path: Path, limit: int, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise VerificationError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise VerificationError(
            f"{label} must be a regular file, not a symlink: {path}"
        )
    if metadata.st_size > limit:
        raise VerificationError(
            f"{label} is too large ({metadata.st_size} bytes; limit {limit}): {path}"
        )
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise VerificationError(f"cannot read {label} {path}: {error}") from error
    if len(contents) > limit:
        raise VerificationError(f"{label} grew beyond its size limit: {path}")
    return contents


def parse_manifest(contents: bytes) -> dict[str, str]:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError("prebuilt provenance is not UTF-8") from error
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise VerificationError(
                f"empty line in prebuilt provenance at line {line_number}"
            )
        key, separator, value = line.partition("=")
        if not separator:
            raise VerificationError(
                f"malformed prebuilt provenance at line {line_number}"
            )
        if key not in MANIFEST_KEYS:
            raise VerificationError(f"unknown prebuilt provenance key {key!r}")
        if not value or value.strip() != value:
            raise VerificationError(f"invalid value for provenance key {key!r}")
        if key in values:
            raise VerificationError(f"duplicate prebuilt provenance key {key!r}")
        values[key] = value
    if set(values) != MANIFEST_KEYS:
        missing = sorted(MANIFEST_KEYS - set(values))
        extra = sorted(set(values) - MANIFEST_KEYS)
        raise VerificationError(
            f"prebuilt provenance has the wrong key set: missing={missing}, extra={extra}"
        )
    return values


def require_value(values: dict[str, str], key: str, expected: str) -> None:
    actual = values[key]
    if actual != expected:
        raise VerificationError(
            f"prebuilt provenance {key!r} mismatch: expected {expected!r}, got {actual!r}"
        )


def exact_ay_lock_commit(lock_contents: bytes) -> str:
    try:
        lines = lock_contents.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise VerificationError("Cargo.lock is not UTF-8") from error
    ay_sources = [line for line in lines if "alabsystems/ay" in line.lower()]
    if not ay_sources:
        raise VerificationError("Cargo.lock contains no AY source entries")
    commits: set[str] = set()
    for line in ay_sources:
        match = AY_SOURCE.fullmatch(line)
        if match is None:
            raise VerificationError(f"non-canonical AY Cargo.lock source: {line}")
        requested, resolved = match.groups()
        if requested != resolved:
            raise VerificationError(
                f"AY requested revision {requested} resolved to {resolved}"
            )
        commits.add(resolved)
    if len(commits) != 1:
        raise VerificationError(
            f"expected exactly one AY commit across Cargo.lock, found {sorted(commits)}"
        )
    return commits.pop()


def git(
    repo_root: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(repo_root), *arguments],
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise VerificationError(
            f"cannot run git for source binding: {error}"
        ) from error
    if check and result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise VerificationError(
            f"git {' '.join(arguments)} failed with status {result.returncode}: {stderr}"
        )
    return result


def verify_git_source_binding(repo_root: Path, source_commit: str) -> None:
    """Verify the non-circular source/artifact relationship when Git is present."""
    if not (repo_root / ".git").exists():
        return

    top_level = git(repo_root, "rev-parse", "--show-toplevel").stdout
    try:
        git_root = Path(top_level.decode("utf-8").strip()).resolve(strict=True)
    except (OSError, UnicodeDecodeError) as error:
        raise VerificationError("Git returned an invalid worktree root") from error
    if git_root != repo_root:
        raise VerificationError(
            f"installer root {repo_root} is not the Git worktree root {git_root}"
        )

    dirty = git(repo_root, "diff", "--quiet", "HEAD", "--", check=False)
    if dirty.returncode == 1:
        raise VerificationError(
            "compiled/package inputs are dirty relative to release HEAD"
        )
    if dirty.returncode != 0:
        raise VerificationError(
            f"git diff failed while validating source state: {dirty.returncode}"
        )

    untracked = git(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *PACKAGE_ROOTS,
    ).stdout
    unexpected = [
        item.decode("utf-8", errors="replace")
        for item in untracked.split(b"\0")
        if item
    ]
    if unexpected:
        raise VerificationError(
            f"package inputs are untracked and not covered by the source seal: {unexpected}"
        )

    head = (
        git(repo_root, "rev-parse", "--verify", "HEAD^{commit}")
        .stdout.decode("ascii")
        .strip()
    )
    if LOWER_HEX_40.fullmatch(head) is None:
        raise VerificationError(
            f"Git returned a non-canonical release commit: {head!r}"
        )
    if head == source_commit:
        return

    exists = git(
        repo_root, "cat-file", "-e", f"{source_commit}^{{commit}}", check=False
    )
    if exists.returncode != 0:
        raise VerificationError(
            f"prebuilt source commit {source_commit} is unavailable from release HEAD {head}"
        )
    ancestor = git(
        repo_root, "merge-base", "--is-ancestor", source_commit, head, check=False
    )
    if ancestor.returncode != 0:
        raise VerificationError(
            f"prebuilt source commit {source_commit} is not an ancestor of release HEAD {head}"
        )
    changed_bytes = git(
        repo_root, "diff", "--name-only", "-z", source_commit, head, "--"
    ).stdout
    changed = {
        item.decode("utf-8", errors="replace")
        for item in changed_bytes.split(b"\0")
        if item
    }
    if changed != PREBUILT_FILES:
        raise VerificationError(
            "release HEAD differs from sealed source outside the exact prebuilt "
            f"triplet: changed={sorted(changed)}, expected={sorted(PREBUILT_FILES)}"
        )


def expected_seal(values: dict[str, str]) -> bytes:
    return (
        "ny.vnncomp.build.v1|status=sealed|"
        f"target={TARGET}|features={FEATURES}|"
        f"ny_commit={values['ny_commit']}|"
        f"cargo_lock_sha256={values['cargo_lock_sha256']}|"
        f"ay_commit={values['ay_lock_commit']}|"
        f"builder_script_sha256={values['builder_script_sha256']}|"
        f"trust_commit={values['trust_commit']}|"
        f"trustc_sha256={values['trustc_sha256']}|"
        f"trustc_version_sha256={values['trustc_version_sha256']}|"
        f"trust_gate_receipt_sha256={values['trust_gate_receipt_sha256']}|"
        f"onnxruntime_static_sha256={values['onnxruntime_static_sha256']}|"
    ).encode("ascii")


def verify_binary(binary: bytes, values: dict[str, str]) -> None:
    if (
        len(binary) < 20
        or binary[:4] != b"\x7fELF"
        or binary[4] != 2
        or binary[5] != 1
        or int.from_bytes(binary[18:20], "little") != 62
    ):
        raise VerificationError("prebuilt is not an ELF64 little-endian x86_64 binary")
    require_value(values, "binary_sha256", sha256_bytes(binary))

    expected = expected_seal(values)
    canonical_records: list[bytes] = []
    cursor = 0
    while True:
        start = binary.find(BUILD_PROVENANCE_PREFIX, cursor)
        if start < 0:
            break
        candidate = binary[start : start + len(expected)]
        if CANONICAL_SEAL.fullmatch(candidate) is not None:
            canonical_records.append(candidate)
        cursor = start + len(BUILD_PROVENANCE_PREFIX)
    exact_count = sum(record == expected for record in canonical_records)
    if len(canonical_records) != 1 or exact_count != 1:
        raise VerificationError(
            "embedded NY/Trust provenance mismatch: expected one exact sealed "
            f"record, found canonical={len(canonical_records)}, exact={exact_count}"
        )

    commits = {
        match.group(1).decode("ascii") + ("-dirty" if match.group(2) else "")
        for match in AY_BUILD_COMMIT.finditer(binary)
    }
    if commits != {values["ay_lock_commit"]}:
        raise VerificationError(
            "embedded AY build.commit mismatch: expected "
            f"{values['ay_lock_commit']}, found {sorted(commits)}"
        )


def decompress_bounded(archive_contents: bytes, output: Path) -> None:
    total = 0
    descriptor: int | None = None
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(output, flags, 0o600)
        created = True
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = None
            with lzma.LZMAFile(io.BytesIO(archive_contents), "rb") as compressed:
                while True:
                    chunk = compressed.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BINARY_BYTES:
                        raise VerificationError(
                            f"decompressed prebuilt exceeds {MAX_BINARY_BYTES} bytes"
                        )
                    destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
    except VerificationError:
        if created:
            output.unlink(missing_ok=True)
        raise
    except (lzma.LZMAError, EOFError, OSError) as error:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            output.unlink(missing_ok=True)
        raise VerificationError(
            f"cannot exclusively stage and decompress prebuilt at {output}: {error}"
        ) from error


def verify(
    repo_root: Path,
    archive: Path,
    checksum: Path,
    provenance: Path,
    output: Path,
) -> None:
    try:
        root = repo_root.resolve(strict=True)
    except OSError as error:
        raise VerificationError(f"cannot resolve repository root: {error}") from error
    if not root.is_dir():
        raise VerificationError(f"repository root is not a directory: {root}")
    expected_paths = {
        "archive": root / ARCHIVE_RELATIVE,
        "checksum": root / CHECKSUM_RELATIVE,
        "provenance": root / PROVENANCE_RELATIVE,
    }
    supplied = {
        "archive": archive.absolute(),
        "checksum": checksum.absolute(),
        "provenance": provenance.absolute(),
    }
    for label, expected in expected_paths.items():
        if supplied[label] != expected:
            raise VerificationError(
                f"{label} path must be exactly {expected}, got {supplied[label]}"
            )

    for relative_parent in (Path("dist"), Path("dist/bin")):
        parent = root / relative_parent
        try:
            metadata = parent.lstat()
        except OSError as error:
            raise VerificationError(
                f"cannot inspect prebuilt parent {parent}: {error}"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise VerificationError(
                f"prebuilt parent must be a real directory, not a symlink: {parent}"
            )

    archive_contents = read_regular_file(
        expected_paths["archive"], MAX_ARCHIVE_BYTES, "compressed prebuilt"
    )
    checksum_contents = read_regular_file(
        expected_paths["checksum"], MAX_CHECKSUM_BYTES, "checksum sidecar"
    )
    provenance_contents = read_regular_file(
        expected_paths["provenance"], MAX_PROVENANCE_BYTES, "provenance manifest"
    )
    archive_sha256 = sha256_bytes(archive_contents)
    expected_checksum = f"{archive_sha256}  ny-x86_64-linux.xz\n".encode("ascii")
    if checksum_contents != expected_checksum:
        raise VerificationError(
            "prebuilt checksum sidecar is not the exact sha256sum record"
        )

    values = parse_manifest(provenance_contents)
    require_value(values, "schema", SCHEMA)
    require_value(values, "target", TARGET)
    require_value(values, "features", FEATURES)
    require_value(values, "trust_bootstrap_mode", "seed")
    require_value(values, "trust_gate_status", "passed")
    require_value(values, "package_sha256", archive_sha256)
    for key in HASH_KEYS:
        if LOWER_HEX_64.fullmatch(values[key]) is None:
            raise VerificationError(f"provenance {key!r} is not 64 lowercase hex")
    for key in ("trust_commit", "ny_commit", "ay_lock_commit"):
        if LOWER_HEX_40.fullmatch(values[key]) is None:
            raise VerificationError(f"provenance {key!r} is not 40 lowercase hex")

    commands_sha256 = sha256_bytes(TRUST_GATE_COMMANDS_V1.encode("utf-8"))
    require_value(values, "trust_gate_commands_sha256", commands_sha256)
    receipt = (
        f"schema={RECEIPT_SCHEMA}\n"
        f"trust_commit={values['trust_commit']}\n"
        f"trustc_sha256={values['trustc_sha256']}\n"
        f"trustc_version_sha256={values['trustc_version_sha256']}\n"
        f"trust_gate_commands_sha256={values['trust_gate_commands_sha256']}\n"
        f"trust_gate_log_sha256={values['trust_gate_log_sha256']}\n"
        "status=passed\n"
    )
    require_value(
        values, "trust_gate_receipt_sha256", sha256_bytes(receipt.encode("utf-8"))
    )

    lock_contents = read_regular_file(
        root / "Cargo.lock", 64 * 1024 * 1024, "Cargo.lock"
    )
    require_value(values, "cargo_lock_sha256", sha256_bytes(lock_contents))
    require_value(values, "ay_lock_commit", exact_ay_lock_commit(lock_contents))
    builder_contents = read_regular_file(
        root / BUILDER_RELATIVE, 1024 * 1024, "Trust Linux builder"
    )
    require_value(values, "builder_script_sha256", sha256_bytes(builder_contents))
    verify_git_source_binding(root, values["ny_commit"])

    decompress_bounded(archive_contents, output)
    try:
        binary = read_regular_file(output, MAX_BINARY_BYTES, "staged prebuilt")
        verify_binary(binary, values)
    except Exception:
        output.unlink(missing_ok=True)
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="verify and decompress the sealed NY VNN-COMP prebuilt",
        allow_abbrev=False,
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--checksum", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        verify(
            arguments.repo_root,
            arguments.archive,
            arguments.checksum,
            arguments.provenance,
            arguments.output,
        )
    except VerificationError as error:
        print(f"ERROR: prebuilt verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
