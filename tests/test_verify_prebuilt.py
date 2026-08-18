# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Offline end-to-end tests for the sealed VNN-COMP prebuilt verifier."""

from __future__ import annotations

import hashlib
import lzma
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFIER = REPO_ROOT / "vnncomp_scripts" / "verify_prebuilt.py"

ARCHIVE = Path("dist/bin/ny-x86_64-linux.xz")
CHECKSUM = Path("dist/bin/ny-x86_64-linux.xz.sha256")
PROVENANCE = Path("dist/bin/ny-x86_64-linux.provenance.txt")
PREBUILT_TRIPLET = (ARCHIVE, CHECKSUM, PROVENANCE)
BUILDER = Path("scripts/vnncomp_trust_linux_build.sh")

AY_COMMIT = "1" * 40
TRUST_COMMIT = "a" * 40
TRUST_GATE_COMMANDS = (
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
MANIFEST_ORDER = (
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
)


@dataclass
class PrebuiltFixture:
    repo: Path
    source_commit: str
    binary: bytes
    values: dict[str, str]


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"git {' '.join(arguments)} failed with {result.returncode}: {result.stderr}"
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _write_manifest(fixture: PrebuiltFixture) -> None:
    contents = "".join(f"{key}={fixture.values[key]}\n" for key in MANIFEST_ORDER)
    (fixture.repo / PROVENANCE).write_text(contents, encoding="utf-8")


def _expected_seal(values: dict[str, str]) -> bytes:
    return (
        "ny.vnncomp.build.v1|status=sealed|"
        "target=x86_64-unknown-linux-gnu|features=mip,cuda|"
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


def _make_fixture(tmp_path: Path) -> PrebuiltFixture:
    repo = tmp_path / "fixture-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "NY Verifier Test")
    _git(repo, "config", "user.email", "ny-verifier@example.invalid")

    (repo / "scripts").mkdir()
    (repo / BUILDER).write_text("#!/bin/bash\nset -euo pipefail\n", encoding="utf-8")
    (repo / BUILDER).chmod(0o755)
    lock = (
        "version = 4\n\n"
        "[[package]]\n"
        'name = "ay-milp"\n'
        'version = "0.11.0"\n'
        f'source = "git+https://github.com/alabsystems/ay.git?rev={AY_COMMIT}'
        f'#{AY_COMMIT}"\n'
    ).encode()
    (repo / "Cargo.lock").write_bytes(lock)
    (repo / "Cargo.toml").write_text(
        '[workspace]\nresolver = "2"\nmembers = []\n', encoding="utf-8"
    )
    (repo / ".gitignore").write_text("/dist/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "Cargo.lock", "Cargo.toml", BUILDER.as_posix())
    source_commit = _commit(repo, "sealed source")

    command_digest = _sha256(TRUST_GATE_COMMANDS.encode("utf-8"))
    values = {
        "schema": "ny-vnncomp-prebuilt-v1",
        "target": "x86_64-unknown-linux-gnu",
        "features": "mip,cuda",
        "trust_commit": TRUST_COMMIT,
        "trust_bootstrap_mode": "seed",
        "trust_gate_status": "passed",
        "trust_gate_commands_sha256": command_digest,
        "trust_gate_log_sha256": "b" * 64,
        "trustc_sha256": "c" * 64,
        "trustc_version_sha256": "d" * 64,
        "ny_commit": source_commit,
        "cargo_lock_sha256": _sha256(lock),
        "ay_lock_commit": AY_COMMIT,
        "builder_script_sha256": _sha256((repo / BUILDER).read_bytes()),
        "onnxruntime_static_sha256": "e" * 64,
    }
    receipt = (
        "schema=ny-trust-gate-receipt-v1\n"
        f"trust_commit={values['trust_commit']}\n"
        f"trustc_sha256={values['trustc_sha256']}\n"
        f"trustc_version_sha256={values['trustc_version_sha256']}\n"
        f"trust_gate_commands_sha256={values['trust_gate_commands_sha256']}\n"
        f"trust_gate_log_sha256={values['trust_gate_log_sha256']}\n"
        "status=passed\n"
    ).encode()
    values["trust_gate_receipt_sha256"] = _sha256(receipt)

    binary = bytearray(64)
    binary[:4] = b"\x7fELF"
    binary[4] = 2  # ELFCLASS64
    binary[5] = 1  # ELFDATA2LSB
    binary[6] = 1  # Current ELF version.
    binary[18:20] = (62).to_bytes(2, "little")  # EM_X86_64
    binary.extend(b"\0build.commit=" + AY_COMMIT.encode("ascii") + b"\0")
    binary.extend(_expected_seal(values) + b"\0")
    binary_bytes = bytes(binary)
    values["binary_sha256"] = _sha256(binary_bytes)

    archive = lzma.compress(binary_bytes, format=lzma.FORMAT_XZ)
    values["package_sha256"] = _sha256(archive)
    (repo / ARCHIVE).parent.mkdir(parents=True)
    (repo / ARCHIVE).write_bytes(archive)
    (repo / CHECKSUM).write_text(
        f"{values['package_sha256']}  {ARCHIVE.name}\n", encoding="ascii"
    )
    fixture = PrebuiltFixture(repo, source_commit, binary_bytes, values)
    _write_manifest(fixture)
    return fixture


def _commit_artifact_triplet(fixture: PrebuiltFixture) -> str:
    _git(
        fixture.repo,
        "add",
        "-f",
        "--",
        *(path.as_posix() for path in PREBUILT_TRIPLET),
    )
    return _commit(fixture.repo, "publish sealed prebuilt triplet")


def _run_verifier(
    fixture: PrebuiltFixture, output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            os.fspath(VERIFIER),
            "--repo-root",
            os.fspath(fixture.repo),
            "--archive",
            os.fspath(fixture.repo / ARCHIVE),
            "--checksum",
            os.fspath(fixture.repo / CHECKSUM),
            "--provenance",
            os.fspath(fixture.repo / PROVENANCE),
            "--output",
            os.fspath(output),
        ],
        cwd=fixture.repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_accepts_prebuilt_sealed_at_current_source_commit(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "verified-ny"

    result = _run_verifier(fixture, output)

    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == fixture.binary
    assert stat.S_ISREG(output.lstat().st_mode)


def test_accepts_exact_artifact_only_descendant_commit(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    release_commit = _commit_artifact_triplet(fixture)
    output = tmp_path / "verified-from-release-commit"

    result = _run_verifier(fixture, output)

    assert release_commit != fixture.source_commit
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == fixture.binary


@pytest.mark.parametrize("tamper", ["manifest", "binary"])
def test_manifest_and_binary_tampering_fail_closed(tmp_path: Path, tamper: str) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / f"untrusted-{tamper}"

    if tamper == "manifest":
        with (fixture.repo / PROVENANCE).open("a", encoding="utf-8") as manifest:
            manifest.write("unexpected_release_claim=true\n")
    else:
        tampered_binary = fixture.binary + b"post-build-tamper"
        archive = lzma.compress(tampered_binary, format=lzma.FORMAT_XZ)
        (fixture.repo / ARCHIVE).write_bytes(archive)
        fixture.values["package_sha256"] = _sha256(archive)
        (fixture.repo / CHECKSUM).write_text(
            f"{fixture.values['package_sha256']}  {ARCHIVE.name}\n",
            encoding="ascii",
        )
        # Keep the original binary digest while making all archive sidecars
        # self-consistent, so the decompressed-binary check is the backstop.
        _write_manifest(fixture)

    result = _run_verifier(fixture, output)

    assert result.returncode == 1
    assert "prebuilt verification failed" in result.stderr
    assert not output.exists(), "failed verification must remove staged output"
    if tamper == "manifest":
        assert "unknown prebuilt provenance key" in result.stderr
    else:
        assert "binary_sha256" in result.stderr


def test_descendant_with_source_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _commit_artifact_triplet(fixture)
    with (fixture.repo / "Cargo.toml").open("a", encoding="utf-8") as manifest:
        manifest.write("\n# unsealed release drift\n")
    _git(fixture.repo, "add", "Cargo.toml")
    _commit(fixture.repo, "unsealed source drift")
    output = tmp_path / "must-not-exist"

    result = _run_verifier(fixture, output)

    assert result.returncode == 1
    assert "outside the exact prebuilt triplet" in result.stderr
    assert "Cargo.toml" in result.stderr
    assert not output.exists()


def test_output_creation_is_exclusive_and_never_follows_symlinks(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)

    existing = tmp_path / "existing-output"
    existing.write_bytes(b"keep-existing")
    existing_result = _run_verifier(fixture, existing)
    assert existing_result.returncode == 1
    assert existing.read_bytes() == b"keep-existing"

    referent = tmp_path / "symlink-referent"
    referent.write_bytes(b"keep-referent")
    symlink = tmp_path / "symlink-output"
    symlink.symlink_to(referent)
    symlink_result = _run_verifier(fixture, symlink)
    assert symlink_result.returncode == 1
    assert symlink.is_symlink()
    assert symlink.resolve() == referent.resolve()
    assert referent.read_bytes() == b"keep-referent"
