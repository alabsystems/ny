#!/usr/bin/env python3
# ruff: noqa: E402, EXE001
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Run preset-driven VNN-COMP beta-crown benchmarks with an external timeout."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports" / "benchmarks"
NY_PREFLIGHT_TIMEOUT_SECS = 5.0
NY_RECEIPT_TIMEOUT_SECS = 60.0
T = TypeVar("T")
LOG = logging.getLogger(__name__)
MAX_STAGED_VNNLIB_BYTES = 256 * 1024 * 1024
MAX_STAGED_ONNX_BYTES = 8 * 1024 * 1024 * 1024
SUPPORTED_STATUS_EXIT_CODES = {
    "verified": frozenset({0}),
    "falsified": frozenset({1}),
    # Historical captured runners sometimes emitted a non-decisive JSON
    # verdict and returned zero. Keep that compatibility narrowly scoped: a
    # non-decisive status may use either its documented code or legacy zero,
    # never another verdict's code.
    "unknown": frozenset({0, 2}),
    "timeout": frozenset({0, 3}),
}
HARNESS_OWNED_EXTRA_FLAGS = frozenset(
    {
        "--property",
        "-p",
        "--preset",
        "--timeout",
        "--json",
        "--max-domains",
        "--domain-batch-metrics-jsonl",
        "--input-split-metrics-jsonl",
    }
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks._shared import get_benchmark_instances
from scripts.benchmark_vnncomp_preset_bounded_results import (
    BenchmarkResult,
    NyProvenance,
    RunProvenance,
    default_output_path,
    write_results,
)

NY_RECEIPT_FIELDS = (
    "schema",
    "binary_sha256",
    "source_kind",
    "source_commit",
    "source_state_sha256",
    "cargo_lock_sha256",
    "ay_commit",
    "features",
    "toolchain_kind",
    "toolchain_sha256",
    "artifact_provenance_sha256",
)
NY_RECEIPT_SCHEMA = "ny-submission-binary-receipt-v1"
NY_RECEIPT_MAX_BYTES = 8192


def _discover_git_executable() -> Path | None:
    candidates = [Path("/usr/bin/git"), Path("/bin/git")]
    discovered = shutil.which("git")
    if discovered is not None:
        candidates.append(Path(discovered))
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    return None


CONTROL_GIT_EXECUTABLE = _discover_git_executable()


@dataclass(frozen=True)
class NyReceiptEvidence:
    identity_json: str
    file_sha256: str


def _resolve_ny_binary(explicit: str | None) -> tuple[Path, str]:
    """Resolve the ny binary path and return (path, source).

    Source is one of: "explicit", "shared-default".
    Policy (#4346): shared repo binaries are preferred over worker-local.
    Worker-local binaries require explicit --ny-binary to be used.
    """
    if explicit:
        # Resolve once against the caller's current directory.  Solver runs use
        # REPO_ROOT as their cwd, while validation and hashing otherwise use the
        # caller's cwd; retaining a relative path could therefore authenticate
        # one file and execute another file with the same relative spelling.
        return Path(explicit).resolve(), "explicit"

    candidates: list[Path] = [
        REPO_ROOT / "target" / "release" / "ny",
        REPO_ROOT / "target" / "debug" / "ny",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate, "shared-default"

    raise FileNotFoundError(
        "No ny binary found. Build ny-cli or pass --ny-binary explicitly."
    )


def _compute_provenance(
    ny_binary: Path,
    source: str,
    *,
    recorded_binary: Path | None = None,
    process_env: dict[str, str] | None = None,
    receipt_evidence: NyReceiptEvidence | None = None,
) -> NyProvenance:
    """Compute binary provenance metadata for CSV recording (#4346)."""
    try:
        version_result = subprocess.run(
            [str(ny_binary), "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
            cwd=REPO_ROOT,
            env=process_env,
        )
        version = (
            (version_result.stdout or "").strip()
            if version_result.returncode == 0
            else "unknown"
        ) or "unknown"
    except (subprocess.TimeoutExpired, OSError):
        version = "unknown"

    try:
        sha256 = _sha256_file(ny_binary)
    except OSError:
        sha256 = "unknown"

    return NyProvenance(
        source=source,
        binary=str(recorded_binary or ny_binary),
        version=version,
        sha256=sha256,
        receipt_json=(receipt_evidence.identity_json if receipt_evidence else ""),
        receipt_sha256=(receipt_evidence.file_sha256 if receipt_evidence else ""),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _parent_environment_evidence(
    process_env: dict[str, str],
) -> tuple[str, str]:
    """Serialize the same ambient treatment flags as NY's flight recorder."""
    relevant = {
        name: value
        for name, value in process_env.items()
        if name.startswith("NY_") or name == "OMP_NUM_THREADS"
    }
    encoded = _canonical_json(relevant)
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve_ny_receipt(explicit: str | None, ny_binary: Path) -> Path:
    """Resolve an explicit receipt once or use the binary's adjacent sidecar."""
    if explicit:
        # Make caller-relative spelling stable without following a symlink;
        # the receipt authority's contract rejects symlink sidecars.
        return Path(explicit).absolute()
    return Path(f"{ny_binary}.receipt")


def _stage_private_ny_receipt(source: Path, staged_binary: Path) -> Path:
    """Copy a receipt beside the already-staged binary and make it read-only."""
    if source.is_symlink() or not source.is_file():
        raise OSError(f"NY receipt is not a regular non-symlink file: {source}")
    staged = staged_binary.with_name("ny.receipt")
    shutil.copyfile(source, staged)
    staged.chmod(stat.S_IRUSR)
    return staged


def _lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(char in "0123456789abcdef" for char in value)


def _parse_ny_receipt(receipt: Path) -> NyReceiptEvidence:
    """Strictly parse the fixed, helper-validated receipt representation."""
    try:
        raw = receipt.read_bytes()
    except OSError as error:
        raise RuntimeError(f"Cannot read NY receipt {receipt}: {error}") from error
    if len(raw) > NY_RECEIPT_MAX_BYTES:
        raise RuntimeError(f"NY receipt is oversized: {receipt}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"NY receipt is not UTF-8: {receipt}") from error
    lines = text.splitlines()
    if len(lines) != len(NY_RECEIPT_FIELDS):
        raise RuntimeError(
            f"NY receipt has the wrong field count: expected "
            f"{len(NY_RECEIPT_FIELDS)}, observed {len(lines)}"
        )
    identity: dict[str, str] = {}
    for field, line in zip(NY_RECEIPT_FIELDS, lines):
        prefix = f"{field}="
        if not line.startswith(prefix):
            raise RuntimeError(f"NY receipt has malformed field {field}")
        identity[field] = line[len(prefix) :]
    canonical_raw = "".join(
        f"{field}={identity[field]}\n" for field in NY_RECEIPT_FIELDS
    ).encode("utf-8")
    if raw != canonical_raw:
        raise RuntimeError("NY receipt is not in its canonical line representation")

    digest_fields = (
        "binary_sha256",
        "source_state_sha256",
        "toolchain_sha256",
    )
    if identity["schema"] != NY_RECEIPT_SCHEMA:
        raise RuntimeError(f"NY receipt has unknown schema {identity['schema']!r}")
    if any(not _lower_hex(identity[field], 64) for field in digest_fields):
        raise RuntimeError("NY receipt contains an invalid SHA-256 field")
    if identity["source_kind"] not in {"git", "archive", "prebuilt"}:
        raise RuntimeError("NY receipt contains an invalid source_kind")
    if not _lower_hex(identity["source_commit"], 40):
        raise RuntimeError("NY receipt contains an invalid source_commit")
    for field in ("cargo_lock_sha256", "artifact_provenance_sha256"):
        if identity[field] != "none" and not _lower_hex(identity[field], 64):
            raise RuntimeError(f"NY receipt contains an invalid {field}")
    if identity["ay_commit"] != "none" and not _lower_hex(
        identity["ay_commit"], 40
    ):
        raise RuntimeError("NY receipt contains an invalid ay_commit")
    if re.fullmatch(r"[a-z0-9]+(?:,[a-z0-9]+)*", identity["features"]) is None:
        raise RuntimeError("NY receipt contains non-canonical features")
    if identity["toolchain_kind"] not in {"rustc-vv", "trust-sealed"}:
        raise RuntimeError("NY receipt contains an invalid toolchain_kind")
    return NyReceiptEvidence(
        identity_json=_canonical_json(identity),
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _receipt_validation_environment(
    _process_env: dict[str, str],
) -> dict[str, str]:
    """Use a minimal environment for the source-identity authority.

    In particular, do not let PATH/LD_PRELOAD/DYLD_*/TMPDIR, shell startup
    hooks, or Git redirection inherited by the measured solver interpose on
    receipt validation. The helper only needs standard platform tools.
    """
    control_path = os.defpath
    if CONTROL_GIT_EXECUTABLE is not None:
        control_path += os.pathsep + str(CONTROL_GIT_EXECUTABLE.parent)
    return {
        "PATH": control_path,
        "HOME": "/nonexistent/ny-receipt-git-home",
        "XDG_CONFIG_HOME": "/nonexistent/ny-receipt-git-xdg",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _authenticate_ny_receipt(
    ny_binary: Path,
    receipt: Path,
    process_env: dict[str, str],
) -> NyReceiptEvidence:
    """Authenticate staged bytes and receipt against this checkout's source."""
    helper = REPO_ROOT / "vnncomp_scripts" / "submission_binary_receipt.sh"
    if helper.is_symlink() or not helper.is_file():
        raise RuntimeError(
            f"NY receipt validation helper is not a regular file: {helper}"
        )
    try:
        control_git_sha256 = (
            _sha256_file(CONTROL_GIT_EXECUTABLE)
            if CONTROL_GIT_EXECUTABLE is not None
            else None
        )
    except OSError as error:
        raise RuntimeError(f"Cannot authenticate Git executable: {error}") from error
    try:
        process = subprocess.run(
            [
                "bash",
                str(helper),
                "validate",
                str(ny_binary),
                str(REPO_ROOT),
                str(receipt),
            ],
            cwd=REPO_ROOT,
            env=_receipt_validation_environment(process_env),
            capture_output=True,
            text=True,
            timeout=NY_RECEIPT_TIMEOUT_SECS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"NY receipt validation could not complete: {error}") from error
    if CONTROL_GIT_EXECUTABLE is not None:
        try:
            final_control_git_sha256 = _sha256_file(CONTROL_GIT_EXECUTABLE)
        except OSError as error:
            raise RuntimeError(f"Cannot revalidate Git executable: {error}") from error
        if final_control_git_sha256 != control_git_sha256:
            raise RuntimeError("Git executable changed during NY receipt validation")
    if process.returncode != 0:
        detail = (process.stderr or "").strip() or (process.stdout or "").strip()
        raise RuntimeError(f"NY receipt validation failed: {detail or process.returncode}")
    evidence = _parse_ny_receipt(receipt)
    identity = json.loads(evidence.identity_json)
    binary_sha256 = _sha256_file(ny_binary)
    if identity["binary_sha256"] != binary_sha256:
        raise RuntimeError(
            "NY receipt binary identity differs from the staged executable: "
            f"receipt={identity['binary_sha256']}, staged={binary_sha256}"
        )
    return evidence


def _stage_private_ny_binary(
    source: Path,
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Copy NY to a private, read/execute-only path for the complete run.

    Provenance records the selected source path and the staged copy's exact
    bytes. Executing only the private copy prevents replacement of a shared
    target binary after hashing from silently changing later measured rows.
    """
    if not source.is_file():
        raise OSError(f"NY binary is not a regular file: {source}")
    source_mode = stat.S_IMODE(source.stat().st_mode)
    stage = tempfile.TemporaryDirectory(prefix="ny-vnncomp-binary-")
    staged = Path(stage.name) / "ny"
    try:
        shutil.copyfile(source, staged)
        # Strip write permissions while preserving execute permissions. Ensure
        # the current owner can read and execute even for an unusual source
        # mode; preflight remains the authority on actual launchability.
        staged.chmod(
            (source_mode & (stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO))
            & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            | stat.S_IRUSR
            | stat.S_IXUSR
        )
    except OSError:
        stage.cleanup()
        raise
    return staged, stage


def _matches_expected_suffix(path: Path, expected: str) -> bool:
    expected_path = Path(expected)
    if expected_path.is_absolute() or ".." in expected_path.parts:
        return False
    expected_parts = expected_path.parts
    return bool(expected_parts) and path.parts[-len(expected_parts) :] == expected_parts


def _validate_expected_row_bindings(
    selected: list[tuple[int, tuple[Path, Path, int]]],
    encoded_bindings: list[str],
) -> str | None:
    """Validate corpus IDs against selected source rows and immutable inputs."""
    if not encoded_bindings:
        return None
    bindings: dict[int, dict] = {}
    for encoded in encoded_bindings:
        try:
            binding = json.loads(encoded)
        except json.JSONDecodeError as error:
            return f"malformed --expected-row-binding JSON: {error}"
        if not isinstance(binding, dict):
            return "each --expected-row-binding must encode a JSON object"
        index = binding.get("source_index_zero_based")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            return "expected row binding has an invalid source_index_zero_based"
        if index in bindings:
            return f"duplicate expected row binding for source index {index}"
        bindings[index] = binding

    selected_indices = [index for index, _instance in selected]
    if set(bindings) != set(selected_indices) or len(bindings) != len(selected_indices):
        return (
            "expected row bindings do not exactly cover selected source indices: "
            f"expected={sorted(bindings)}, selected={selected_indices}"
        )

    hashes: dict[Path, str] = {}
    for index, (model_path, property_path, timeout) in selected:
        binding = bindings[index]
        corpus_id = binding.get("corpus_id", f"source-index-{index}")
        expected_timeout = binding.get("timeout_seconds")
        if (
            isinstance(expected_timeout, bool)
            or not isinstance(expected_timeout, int)
            or expected_timeout <= 0
        ):
            return f"corpus binding {corpus_id!r} has invalid timeout_seconds"
        if timeout != expected_timeout:
            return (
                f"corpus binding {corpus_id!r} timeout mismatch: "
                f"expected {expected_timeout}, observed {timeout}"
            )
        for label, actual_path in (
            ("model", model_path),
            ("property", property_path),
        ):
            expected_path = binding.get(label)
            if not isinstance(expected_path, str) or not _matches_expected_suffix(
                actual_path, expected_path
            ):
                return (
                    f"corpus binding {corpus_id!r} {label} path mismatch: "
                    f"expected suffix {expected_path!r}, selected {actual_path}"
                )
            expected_hash = binding.get(f"{label}_sha256")
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(char not in "0123456789abcdef" for char in expected_hash)
            ):
                return f"corpus binding {corpus_id!r} has invalid {label}_sha256"
            try:
                if actual_path not in hashes:
                    hashes[actual_path] = _sha256_file(actual_path)
                actual_hash = hashes[actual_path]
            except OSError as error:
                return f"cannot hash selected {label} input {actual_path}: {error}"
            if actual_hash != expected_hash:
                return (
                    f"corpus binding {corpus_id!r} {label} hash mismatch: "
                    f"expected {expected_hash}, observed {actual_hash}"
                )
    return None


def _stage_gzip_only_inputs(
    selected: list[tuple[int, tuple[Path, Path, int]]],
) -> tuple[
    list[tuple[int, tuple[Path, Path, int]]],
    tempfile.TemporaryDirectory[str] | None,
]:
    """Stage missing logical ONNX/VNNLIB files from adjacent gzip archives.

    Staged paths preserve the conventional `onnx/<name>` or `vnnlib/<name>`
    suffix, so row-binding checks still validate the official logical identity.
    Hashing is performed later over decompressed bytes, exactly matching the
    manifest. A per-source directory prevents basename collisions without
    changing that suffix, and repeated logical inputs are decompressed once.
    """
    # Validate the complete selection before decompressing anything. This is
    # both fail-closed and important for incomplete large corpora: a missing
    # late row must not first consume gigabytes staging all earlier rows.
    for source_index, (model_path, property_path, _timeout) in selected:
        for label, logical_path, expected_suffix in (
            ("model", model_path, ".onnx"),
            ("property", property_path, ".vnnlib"),
        ):
            if logical_path.suffix != expected_suffix:
                raise ValueError(
                    f"selected source row {source_index} {label} has unexpected "
                    f"logical suffix: {logical_path}"
                )
            archive = Path(f"{logical_path}.gz")
            if not logical_path.is_file() and not archive.is_file():
                raise ValueError(
                    f"selected source row {source_index} has unavailable {label} "
                    f"input: {logical_path} (adjacent archive {archive} also absent); "
                    "refusing to substitute a different manifest row"
                )

    stage: tempfile.TemporaryDirectory[str] | None = None
    staged_by_logical_path: dict[Path, Path] = {}
    resolved: list[tuple[int, tuple[Path, Path, int]]] = []
    for source_index, (model_path, property_path, timeout) in selected:
        staged_paths: list[Path] = []
        for label, logical_path, expected_suffix, size_cap in (
            ("model", model_path, ".onnx", MAX_STAGED_ONNX_BYTES),
            ("property", property_path, ".vnnlib", MAX_STAGED_VNNLIB_BYTES),
        ):
            staged_path = logical_path
            archive = Path(f"{logical_path}.gz")
            if (
                not logical_path.is_file()
                and logical_path.suffix == expected_suffix
                and archive.is_file()
            ):
                if logical_path in staged_by_logical_path:
                    staged_paths.append(staged_by_logical_path[logical_path])
                    continue
                if stage is None:
                    stage = tempfile.TemporaryDirectory(prefix="ny-vnncomp-inputs-")
                kind_dir = "onnx" if expected_suffix == ".onnx" else "vnnlib"
                # Preserve every safe path component below the conventional
                # ONNX/VNNLIB directory.  Official categories such as safenlp
                # use nested paths (`onnx/medical/...`), and immutable row
                # bindings include that complete logical suffix.
                logical_parts = logical_path.parts
                kind_positions = [
                    position
                    for position, part in enumerate(logical_parts)
                    if part == kind_dir
                ]
                if kind_positions:
                    staged_relative = Path(*logical_parts[kind_positions[-1] :])
                else:
                    staged_relative = Path(kind_dir) / logical_path.name
                if (
                    staged_relative.is_absolute()
                    or ".." in staged_relative.parts
                    or staged_relative.parts[0] != kind_dir
                    or staged_relative.suffix != expected_suffix
                ):
                    raise ValueError(
                        f"selected source row {source_index} {label} has unsafe "
                        f"logical staging path: {logical_path}"
                    )
                staged_path = (
                    Path(stage.name)
                    / f"source-{source_index}"
                    / staged_relative
                )
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                total = 0
                try:
                    with gzip.open(archive, "rb") as source, staged_path.open("xb") as target:
                        while block := source.read(1024 * 1024):
                            total += len(block)
                            if total > size_cap:
                                raise ValueError(
                                    f"decompressed {label} exceeds {size_cap} bytes"
                                )
                            target.write(block)
                except (OSError, EOFError, ValueError) as error:
                    raise ValueError(
                        f"cannot stage gzip-only {label} {archive}: {error}"
                    ) from error
                staged_by_logical_path[logical_path] = staged_path
                LOG.info(
                    "Staged gzip-only source row %d %s %s (%d bytes)",
                    source_index,
                    label,
                    logical_path.name,
                    total,
                )
            staged_paths.append(staged_path)
        resolved.append(
            (source_index, (staged_paths[0], staged_paths[1], timeout))
        )
    return resolved, stage


def _preflight_ny_binary(
    ny_binary: Path,
    timeout_secs: float | None = None,
    process_env: dict[str, str] | None = None,
) -> None:
    """Fail fast when the chosen binary cannot answer `--version`."""
    timeout_secs = NY_PREFLIGHT_TIMEOUT_SECS if timeout_secs is None else timeout_secs
    command = [str(ny_binary), "--version"]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True,
            timeout=timeout_secs,
            cwd=REPO_ROOT,
            env=process_env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Ny binary preflight timed out after {timeout_secs:.1f}s "
            f"while running `--version`: {ny_binary}. "
            "Use a clean rebuilt binary or pass --ny-binary explicitly."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Ny binary preflight could not execute {ny_binary}: {exc}"
        ) from exc
    if process.returncode != 0:
        detail = (process.stderr or "").strip() or (process.stdout or "").strip() or f"exit_code={process.returncode}"
        raise RuntimeError(
            f"Ny binary preflight failed while running `--version`: {ny_binary} ({detail})"
        )


def _parse_json_from_output(text: str) -> dict | None:
    """Return the last complete verdict object embedded in process output.

    JSON-aware decoding is required here: brace counting breaks on braces in a
    quoted reason, and accepting the first arbitrary JSON diagnostic can hide
    the final verdict object.
    """
    decoder = json.JSONDecoder()
    verdict: dict | None = None
    start = text.find("{")
    while start != -1:
        try:
            payload, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(payload, dict) and (
            "status" in payload or "property_status" in payload
        ):
            verdict = payload
        start = text.find("{", start + max(consumed, 1))
    return verdict


def _normalize_status(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    status = raw.lower()
    if status == "safe":
        return "verified"
    if status == "violated":
        return "falsified"
    if status == "potential_violation":
        return "unknown"
    return status if status in SUPPORTED_STATUS_EXIT_CODES else None


def _validated_payload_status(payload: dict) -> tuple[str | None, str | None]:
    """Validate top-level/property verdict fields and return (status, error)."""
    top_present = "status" in payload
    property_present = "property_status" in payload
    top_status = _normalize_status(payload.get("status")) if top_present else None
    property_status = (
        _normalize_status(payload.get("property_status"))
        if property_present
        else None
    )
    if top_present and top_status is None:
        return None, f"unsupported status: {payload.get('status')!r}"
    if property_present and property_status is None:
        return None, (
            f"unsupported property_status: {payload.get('property_status')!r}"
        )
    if not top_present and not property_present:
        return None, "JSON verdict has no status or property_status"
    if (
        top_status is not None
        and property_status is not None
        and top_status != property_status
    ):
        return None, (
            "status/property_status mismatch: "
            f"status={payload.get('status')!r}, "
            f"property_status={payload.get('property_status')!r}"
        )
    return property_status or top_status, None


def _select_unambiguous_verdict_payload(
    stdout: str, stderr: str
) -> tuple[dict | None, str | None]:
    """Select one verdict without allowing either output stream to mask another."""
    stdout_payload = _parse_json_from_output(stdout)
    stderr_payload = _parse_json_from_output(stderr)
    for stream_name, stream_text, payload in (
        ("stdout", stdout, stdout_payload),
        ("stderr", stderr, stderr_payload),
    ):
        if payload is None and (
            '"status"' in stream_text or '"property_status"' in stream_text
        ):
            return None, f"malformed JSON verdict marker in {stream_name}"

    if stdout_payload is None:
        return stderr_payload, None
    if stderr_payload is None:
        return stdout_payload, None

    stdout_status, stdout_error = _validated_payload_status(stdout_payload)
    stderr_status, stderr_error = _validated_payload_status(stderr_payload)
    if stdout_error is not None or stderr_error is not None:
        return None, (
            "ambiguous stdout/stderr verdicts: "
            f"stdout={stdout_error or stdout_status}, "
            f"stderr={stderr_error or stderr_status}"
        )
    if stdout_status != stderr_status:
        return None, (
            "conflicting stdout/stderr verdict statuses: "
            f"stdout={stdout_status}, stderr={stderr_status}"
        )

    # These fields carry the decisive witness/treatment and the search evidence
    # copied into CSV. Missing counters are canonically zero, matching the row
    # parser; all other fields must agree exactly, including presence.
    for field in (
        "effective_config",
        "execution_observations",
        "counterexample",
        "counterexample_vnnlib",
    ):
        if stdout_payload.get(field) != stderr_payload.get(field):
            return None, f"conflicting stdout/stderr verdict field {field!r}"
    for field in (
        "domains_explored",
        "domains_verified",
        "max_depth_reached",
    ):
        if stdout_payload.get(field, 0) != stderr_payload.get(field, 0):
            return None, f"conflicting stdout/stderr verdict field {field!r}"
    return stdout_payload, None


def _definitive_status_matches_exit_code(status: str, returncode: int) -> bool:
    """Require shell/JSON agreement before recording a decisive verdict.

    Ny reserves exit code 0 for verified and 1 for a confirmed violation.
    Unknown/timeout compatibility remains permissive because older captured
    runners returned zero after emitting those non-decisive JSON statuses.
    Operational error codes (>=4) can never authenticate a verdict.
    """
    return returncode in SUPPORTED_STATUS_EXIT_CODES.get(status, frozenset())


def _validate_extra_args(extra_args: list[str]) -> str | None:
    """Reject appended arguments that can rewrite harness-owned authority."""
    for argument in extra_args:
        flag = argument.split("=", 1)[0]
        if flag in HARNESS_OWNED_EXTRA_FLAGS or (
            flag.startswith("-p") and not flag.startswith("--")
        ):
            return (
                f"--extra-arg may not override harness-owned flag {flag!r}: "
                f"{argument!r}"
            )
    return None


def _nonnegative_int_field(payload: dict, field: str) -> int:
    value = payload.get(field, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {field}: {value!r}")
    return value


def _effective_config_evidence(payload: dict) -> tuple[str, str]:
    """Return canonical JSON and its SHA-256 for an observed treatment."""
    effective_config = payload.get("effective_config")
    if not isinstance(effective_config, dict):
        return "", ""
    try:
        canonical = json.dumps(
            effective_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        encoded = canonical.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        return "", ""
    return canonical, hashlib.sha256(encoded).hexdigest()


def _execution_observations_evidence(payload: dict) -> tuple[str, str]:
    """Return canonical JSON and SHA-256 for observed runtime treatment use.

    Execution observations deliberately remain separate from effective config:
    their counters can differ by instance without changing the static treatment
    identity shared by every row in a factorial arm.
    """
    observations = payload.get("execution_observations")
    if not isinstance(observations, dict):
        return "", ""
    try:
        canonical = json.dumps(
            observations,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        encoded = canonical.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        return "", ""
    return canonical, hashlib.sha256(encoded).hexdigest()


def _sample_evenly(items: list[T], sample_size: int) -> list[tuple[int, T]]:
    if sample_size >= len(items):
        return list(enumerate(items))

    step = len(items) / sample_size
    sampled: list[tuple[int, T]] = []
    for sample_index in range(sample_size):
        item_index = int(sample_index * step)
        sampled.append((item_index, items[item_index]))
    return sampled


def _select_instances(
    instances: list[tuple[Path, Path, int]],
    sample: int,
    indices: list[int] | None,
) -> list[tuple[int, tuple[Path, Path, int]]]:
    if indices:
        selected: list[tuple[int, tuple[Path, Path, int]]] = []
        for index in indices:
            if index < 0 or index >= len(instances):
                raise ValueError(f"Index {index} out of range for {len(instances)} instances")
            selected.append((index, instances[index]))
        return selected

    if sample > 0:
        return _sample_evenly(instances, sample)

    return list(enumerate(instances))


def _build_command(
    ny_binary: Path, model_path: Path, property_path: Path,
    preset_path: Path, timeout: int, max_domains: int | None,
    domain_batch_metrics_jsonl: Path | None, extra_args: list[str],
) -> list[str]:
    cmd = [str(ny_binary), "beta-crown", str(model_path),
           "--property", str(property_path), "--preset", str(preset_path),
           "--timeout", str(timeout), "--json"]
    if max_domains is not None:
        cmd.extend(["--max-domains", str(max_domains)])
    if domain_batch_metrics_jsonl is not None:
        cmd.extend(["--domain-batch-metrics-jsonl", str(domain_batch_metrics_jsonl)])
    cmd.extend(extra_args)
    return cmd


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _write_attempt_artifacts(
    artifact_dir: Path | None,
    *,
    command: list[str],
    stdout: str,
    stderr: str,
    result: BenchmarkResult,
    elapsed: float,
    returncode: int | None,
    external_timeout: int | None,
) -> None:
    """Retain complete decoded process output and attempt metadata when requested."""
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(REPO_ROOT),
                "external_timeout_seconds": external_timeout,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (artifact_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    (artifact_dir / "result.txt").write_text(result.result + "\n", encoding="utf-8")
    if result.effective_config_json:
        effective_config = json.loads(result.effective_config_json)
        (artifact_dir / "effective_config.json").write_text(
            json.dumps(effective_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (artifact_dir / "effective_config.sha256").write_text(
            result.effective_config_sha256 + "\n",
            encoding="utf-8",
        )
    if result.execution_observations_json:
        execution_observations = json.loads(result.execution_observations_json)
        (artifact_dir / "execution_observations.json").write_text(
            json.dumps(execution_observations, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (artifact_dir / "execution_observations.sha256").write_text(
            result.execution_observations_sha256 + "\n",
            encoding="utf-8",
        )


def _run_single_attempt(
    ny_binary: Path,
    preset_path: Path,
    model_path: Path,
    property_path: Path,
    timeout: int,
    timeout_slack: int | None,
    max_domains: int | None,
    domain_batch_metrics_jsonl: Path | None,
    extra_args: list[str],
    source_index_zero_based: int = -1,
    artifact_dir: Path | None = None,
    process_env: dict[str, str] | None = None,
) -> BenchmarkResult:
    command = _build_command(
        ny_binary=ny_binary, model_path=model_path,
        property_path=property_path, preset_path=preset_path,
        timeout=timeout, max_domains=max_domains,
        domain_batch_metrics_jsonl=domain_batch_metrics_jsonl,
        extra_args=extra_args,
    )
    sidecar_str = str(domain_batch_metrics_jsonl or "")
    common = {
        "model": model_path.name,
        "property": property_path.name,
        "source_index_zero_based": source_index_zero_based,
        "timeout": timeout,
        "domain_batch_metrics_jsonl": sidecar_str,
    }
    external_timeout = None if timeout_slack is None else timeout + timeout_slack
    start = time.time()
    try:
        process = subprocess.run(
            command, capture_output=True, text=True,
            timeout=external_timeout,
            cwd=REPO_ROOT,
            env=process_env,
        )
        elapsed = time.time() - start
        stdout = process.stdout or ""
        stderr = process.stderr or ""
        returncode: int | None = process.returncode
    except subprocess.TimeoutExpired as error:
        elapsed = time.time() - start
        stdout = _timeout_output(error.stdout)
        stderr = _timeout_output(error.stderr)
        result = BenchmarkResult(
            **common, result="timeout_ext", elapsed=elapsed,
            domains_explored=0, domains_verified=0, max_depth_reached=0,
            reason=f"external_timeout_{external_timeout}s",
        )
        _write_attempt_artifacts(
            artifact_dir,
            command=command,
            stdout=stdout,
            stderr=stderr,
            result=result,
            elapsed=elapsed,
            returncode=None,
            external_timeout=external_timeout,
        )
        return result
    payload, payload_selection_error = _select_unambiguous_verdict_payload(
        stdout, stderr
    )
    if payload_selection_error is not None:
        result = BenchmarkResult(
            **common,
            result="error",
            elapsed=elapsed,
            domains_explored=0,
            domains_verified=0,
            max_depth_reached=0,
            reason=payload_selection_error,
        )
    elif payload:
        status, status_error = _validated_payload_status(payload)
        if status_error is not None or status is None:
            result = BenchmarkResult(
                **common,
                result="error",
                elapsed=elapsed,
                domains_explored=0,
                domains_verified=0,
                max_depth_reached=0,
                reason=status_error or "invalid JSON verdict status",
            )
        elif not _definitive_status_matches_exit_code(status, process.returncode):
            result = BenchmarkResult(
                **common,
                result="error",
                elapsed=elapsed,
                domains_explored=0,
                domains_verified=0,
                max_depth_reached=0,
                reason=(
                    f"status/exit mismatch: status={status}, "
                    f"exit_code={process.returncode}"
                ),
            )
        else:
            try:
                domains_explored = _nonnegative_int_field(
                    payload, "domains_explored"
                )
                domains_verified = _nonnegative_int_field(
                    payload, "domains_verified"
                )
                max_depth_reached = _nonnegative_int_field(
                    payload, "max_depth_reached"
                )
            except ValueError as error:
                result = BenchmarkResult(
                    **common,
                    result="error",
                    elapsed=elapsed,
                    domains_explored=0,
                    domains_verified=0,
                    max_depth_reached=0,
                    reason=str(error),
                )
            else:
                effective_config_json, effective_config_sha256 = (
                    _effective_config_evidence(payload)
                )
                execution_observations_json, execution_observations_sha256 = (
                    _execution_observations_evidence(payload)
                )
                result = BenchmarkResult(
                    **common,
                    result=status,
                    elapsed=elapsed,
                    domains_explored=domains_explored,
                    domains_verified=domains_verified,
                    max_depth_reached=max_depth_reached,
                    reason=str(payload.get("reason", "")),
                    effective_config_json=effective_config_json,
                    effective_config_sha256=effective_config_sha256,
                    execution_observations_json=execution_observations_json,
                    execution_observations_sha256=execution_observations_sha256,
                )
    else:
        output_tail = (stdout + stderr).strip().splitlines()
        reason = output_tail[-1][:200] if output_tail else f"exit_code={process.returncode}"
        result = BenchmarkResult(
            **common, result="error", elapsed=elapsed,
            domains_explored=0, domains_verified=0, max_depth_reached=0,
            reason=reason,
        )
    _write_attempt_artifacts(
        artifact_dir,
        command=command,
        stdout=stdout,
        stderr=stderr,
        result=result,
        elapsed=elapsed,
        returncode=returncode,
        external_timeout=external_timeout,
    )
    return result


def _is_presearch_result(result: BenchmarkResult) -> bool:
    """A result is "pre-search" when it ended before BaB search started.

    Predicate from #4412 design: domains_explored == 0 and status is not
    verified or falsified.
    """
    if result.result in ("verified", "falsified"):
        return False
    return result.domains_explored == 0


def _build_retry_notes(
    warmup_runs: int, retry_count: int, initial: BenchmarkResult,
) -> str:
    """Build provenance notes for a retried pre-search row (#4412 Packet C)."""
    parts: list[str] = []
    if warmup_runs > 0:
        parts.append(f"warmup_runs={warmup_runs}")
    parts.append(f"measured_attempts={1 + retry_count}")
    parts.append(f"presearch_retry={retry_count}")
    parts.append(f"initial_result={initial.result}")
    parts.append(f"initial_domains={initial.domains_explored}")
    if initial.reason:
        parts.append(f"initial_reason={initial.reason[:100]}")
    return "; ".join(parts)


def _run_instance(
    ny_binary: Path,
    preset_path: Path,
    model_path: Path,
    property_path: Path,
    timeout: int,
    timeout_slack: int,
    max_domains: int | None,
    domain_batch_metrics_jsonl: Path | None,
    extra_args: list[str],
    source_index_zero_based: int = -1,
    warmup_runs: int = 0,
    rerun_presearch: int = 0,
    raw_artifact_dir: Path | None = None,
    process_env: dict[str, str] | None = None,
) -> BenchmarkResult:
    """Run one instance with optional warmup and pre-search rerun policy (#4412)."""
    base = {
        "ny_binary": ny_binary,
        "preset_path": preset_path,
        "model_path": model_path,
        "property_path": property_path,
        "timeout": timeout,
        "max_domains": max_domains,
        "extra_args": extra_args,
        "source_index_zero_based": source_index_zero_based,
        "process_env": process_env,
    }
    for warmup_idx in range(warmup_runs):
        LOG.info("    warmup %d/%d", warmup_idx + 1, warmup_runs)
        _run_single_attempt(
            **base,
            # Warmups are excluded from measured timing, but they are still
            # externally bounded: a hung warmup must never hang this bounded
            # harness indefinitely.
            timeout_slack=timeout_slack,
            domain_batch_metrics_jsonl=None,
            artifact_dir=(
                raw_artifact_dir / f"warmup-{warmup_idx + 1:02d}"
                if raw_artifact_dir is not None
                else None
            ),
        )

    result = _run_single_attempt(
        **base,
        timeout_slack=timeout_slack,
        domain_batch_metrics_jsonl=domain_batch_metrics_jsonl,
        artifact_dir=(
            raw_artifact_dir / "measured-01"
            if raw_artifact_dir is not None
            else None
        ),
    )

    if rerun_presearch > 0 and _is_presearch_result(result):
        initial_result = result
        for retry_idx in range(rerun_presearch):
            LOG.info(
                "    presearch retry %d/%d (initial: %s, domains=%d)",
                retry_idx + 1, rerun_presearch,
                initial_result.result, initial_result.domains_explored,
            )
            if domain_batch_metrics_jsonl is not None:
                sidecar = Path(domain_batch_metrics_jsonl)
                if sidecar.exists():
                    sidecar.unlink()
            result = _run_single_attempt(
                **base,
                timeout_slack=timeout_slack,
                domain_batch_metrics_jsonl=domain_batch_metrics_jsonl,
                artifact_dir=(
                    raw_artifact_dir / f"measured-{retry_idx + 2:02d}"
                    if raw_artifact_dir is not None
                    else None
                ),
            )
            if not _is_presearch_result(result):
                break
        result.notes = _build_retry_notes(warmup_runs, retry_idx + 1, initial_result)
    elif warmup_runs > 0:
        result.notes = f"warmup_runs={warmup_runs}"

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preset-driven VNN-COMP beta-crown benchmarks with an external timeout."
    )
    parser.add_argument("--year", type=int, default=2025, help="VNN-COMP year (default: 2025)")
    parser.add_argument("--category", required=True, help="Benchmark category name")
    parser.add_argument(
        "--benchmark-root",
        default="",
        help=(
            "Explicit directory containing category folders. "
            "Default: benchmarks/vnncomp<year>/benchmarks in this checkout"
        ),
    )
    parser.add_argument(
        "--preset",
        default="",
        help="Preset YAML path (default: configs/vnncomp<year>/<category>.yaml)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Evenly sample N instances instead of running the full category",
    )
    parser.add_argument(
        "--indices",
        default="",
        help=(
            "Comma-separated zero-based indices in the unfiltered instances.csv "
            "data rows; selected rows with unavailable inputs fail closed"
        ),
    )
    parser.add_argument(
        "--timeout-slack",
        type=int,
        default=5,
        help="Extra seconds before the external timeout kills ny (default: 5)",
    )
    parser.add_argument(
        "--timeout-cap",
        type=int,
        default=0,
        help=(
            "Cap each official instance timeout for non-promotional pilot runs "
            "(default: 0, use the official timeout)"
        ),
    )
    parser.add_argument(
        "--max-domains",
        type=int,
        default=-1,
        help="Optional max-domains override (default: use preset/CLI defaults)",
    )
    parser.add_argument(
        "--ny-binary",
        default="",
        help="Path to ny binary (default: shared repo release/debug)",
    )
    parser.add_argument(
        "--ny-receipt",
        default="",
        help=(
            "Explicit ny submission receipt path; supplying it enables strict "
            "receipt/source authentication"
        ),
    )
    parser.add_argument(
        "--require-ny-receipt",
        action="store_true",
        help=(
            "Require an authenticated adjacent ny.receipt (or --ny-receipt); "
            "promotion harnesses must use this"
        ),
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix tag for the output CSV name",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Explicit output CSV path",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra beta-crown CLI argument to append (repeatable)",
    )
    parser.add_argument(
        "--expected-row-binding",
        action="append",
        default=[],
        help=(
            "Machine-generated JSON binding a selected source index to expected "
            "model/property paths and SHA-256 hashes (repeatable)"
        ),
    )
    parser.add_argument(
        "--domain-batch-metrics-dir",
        default="",
        help="Directory for per-run graph domain-batch JSONL sidecars",
    )
    parser.add_argument(
        "--raw-artifact-dir",
        default="",
        help=(
            "Directory in which to retain complete stdout, stderr, command, "
            "timing, and result output for every warmup/measured attempt"
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Untimed row-local warmup runs before the first measured attempt (default: 0)",
    )
    parser.add_argument(
        "--rerun-presearch",
        type=int,
        default=0,
        help="Max extra measured attempts when the current attempt ends before search (default: 0)",
    )
    return parser.parse_args()


def _validate_inputs(
    ny_binary: Path,
    preset_path: Path,
    process_env: dict[str, str] | None = None,
) -> NyProvenance | None:
    """Validate binary and preset, return provenance or None on error."""
    if not preset_path.exists():
        LOG.error("Preset not found: %s", preset_path)
        return None
    if not ny_binary.exists():
        LOG.error("Ny binary not found: %s", ny_binary)
        return None
    try:
        _preflight_ny_binary(ny_binary, process_env=process_env)
    except RuntimeError as err:
        LOG.error("%s", err)
        return None
    return NyProvenance(source="", binary="", version="", sha256="")


def main() -> int:
    args = parse_args()
    # Freeze once: every NY subprocess in this run sees exactly this mapping,
    # even if the parent environment changes while rows are being measured.
    process_env = dict(os.environ)
    parent_env_json, parent_env_sha256 = _parent_environment_evidence(process_env)
    if (
        args.sample < 0
        or args.timeout_slack < 0
        or args.timeout_cap < 0
        or args.warmup_runs < 0
        or args.rerun_presearch < 0
    ):
        raise ValueError("sample/timeout/warmup/rerun values must be non-negative")
    extra_arg_error = _validate_extra_args(args.extra_arg)
    if extra_arg_error is not None:
        LOG.error("%s", extra_arg_error)
        return 2
    selected_ny_binary, ny_source = _resolve_ny_binary(args.ny_binary or None)
    authenticate_receipt = args.require_ny_receipt or bool(args.ny_receipt)
    selected_ny_receipt = (
        _resolve_ny_receipt(args.ny_receipt or None, selected_ny_binary)
        if authenticate_receipt
        else None
    )
    preset_path = (
        # Solver subprocesses run with REPO_ROOT as cwd. Resolve an explicit
        # caller-relative preset once so validation and execution cannot refer
        # to different files with the same spelling.
        Path(args.preset).resolve()
        if args.preset
        else REPO_ROOT
        / "configs"
        / f"vnncomp{args.year % 100:02d}"
        / f"{args.category}.yaml"
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else default_output_path(REPORTS_DIR, args.category, args.tag or None)
    )
    if os.path.lexists(output_path):
        LOG.error(
            "Output CSV already exists; refusing to reuse stale evidence: %s",
            output_path,
        )
        return 2

    try:
        ny_binary, staged_ny_binary_dir = _stage_private_ny_binary(
            selected_ny_binary
        )
        staged_ny_receipt = (
            _stage_private_ny_receipt(selected_ny_receipt, ny_binary)
            if selected_ny_receipt is not None
            else None
        )
    except OSError as error:
        LOG.error("Cannot stage private NY binary/receipt: %s", error)
        return 2
    # Keep the private executable alive through every warmup, retry, measured
    # row, and the final integrity check.
    _staged_ny_binary_dir = staged_ny_binary_dir

    receipt_evidence: NyReceiptEvidence | None = None
    if staged_ny_receipt is not None:
        try:
            receipt_evidence = _authenticate_ny_receipt(
                ny_binary, staged_ny_receipt, process_env
            )
        except RuntimeError as error:
            LOG.error("%s", error)
            return 2

    sentinel = _validate_inputs(ny_binary, preset_path, process_env)
    if sentinel is None:
        return 2
    try:
        preset_sha256 = _sha256_file(preset_path)
    except OSError as error:
        LOG.error("Cannot hash preset %s: %s", preset_path, error)
        return 2

    provenance = _compute_provenance(
        ny_binary,
        ny_source,
        recorded_binary=selected_ny_binary,
        process_env=process_env,
        receipt_evidence=receipt_evidence,
    )
    if (
        provenance.version == "unknown"
        or len(provenance.sha256) != 64
        or any(char not in "0123456789abcdef" for char in provenance.sha256)
    ):
        LOG.error(
            "Cannot establish complete NY binary provenance for %s",
            selected_ny_binary,
        )
        return 2
    LOG.info(
        "Binary provenance: source=%s, sha256=%s..., version=%s",
        provenance.source, provenance.sha256[:16], provenance.version,
    )

    indices = [int(part) for part in args.indices.split(",") if part.strip()] or None
    benchmark_root = Path(args.benchmark_root).resolve() if args.benchmark_root else None
    try:
        instances = get_benchmark_instances(
            args.year,
            args.category,
            benchmark_root=benchmark_root,
            preserve_source_rows=True,
        )
    except (OSError, ValueError) as error:
        LOG.error("Cannot read benchmark manifest: %s", error)
        return 1
    if not instances:
        LOG.error("No instances found for %s %s", args.year, args.category)
        return 1

    selected = _select_instances(instances, sample=args.sample, indices=indices)
    try:
        selected, staged_input_dir = _stage_gzip_only_inputs(selected)
    except ValueError as error:
        LOG.error("%s", error)
        return 1
    # Keep the temporary stage alive through hashing and every solver attempt.
    # It is cleaned automatically when `main` returns.
    _staged_input_dir = staged_input_dir
    for index, (model_path, property_path, _timeout) in selected:
        missing = [
            str(path)
            for path in (model_path, property_path)
            if not path.is_file()
        ]
        if missing:
            LOG.error(
                "Selected source row %d (zero-based) has unavailable input(s): %s. "
                "Indices are bound to unfiltered instances.csv data rows; refusing "
                "to substitute a different locally materialized row.",
                index,
                ", ".join(missing),
            )
            return 1
    binding_error = _validate_expected_row_bindings(
        selected, args.expected_row_binding
    )
    if binding_error is not None:
        LOG.error("%s", binding_error)
        return 1
    max_domains = args.max_domains if args.max_domains >= 0 else None
    domain_batch_metrics_dir = (
        Path(args.domain_batch_metrics_dir).resolve()
        if args.domain_batch_metrics_dir
        else None
    )
    if domain_batch_metrics_dir is not None:
        domain_batch_metrics_dir.mkdir(parents=True, exist_ok=True)
    raw_artifact_dir = (
        Path(args.raw_artifact_dir).resolve() if args.raw_artifact_dir else None
    )
    if raw_artifact_dir is not None:
        raw_artifact_dir.mkdir(parents=True, exist_ok=False)

    LOG.info(
        "Running %d/%d %s instances with %s and preset %s",
        len(selected), len(instances), args.category, selected_ny_binary, preset_path,
    )

    results: list[BenchmarkResult] = []
    counts: dict[str, int] = {}
    for position, (index, (model_path, property_path, timeout)) in enumerate(selected, start=1):
        effective_timeout = (
            min(timeout, args.timeout_cap) if args.timeout_cap > 0 else timeout
        )
        LOG.info(
            "[%d/%d] idx=%d %s / %s (timeout=%ss%s)",
            position, len(selected), index,
            model_path.name, property_path.name, timeout,
            (
                f", pilot_cap={effective_timeout}s"
                if effective_timeout != timeout
                else ""
            ),
        )
        domain_batch_metrics_jsonl = None
        if domain_batch_metrics_dir is not None:
            domain_batch_metrics_jsonl = domain_batch_metrics_dir / f"{args.category}_idx{index:04d}.jsonl"
        row_artifact_dir = (
            raw_artifact_dir / f"{args.category}_idx{index:04d}"
            if raw_artifact_dir is not None
            else None
        )
        result = _run_instance(
            ny_binary=ny_binary, preset_path=preset_path,
            model_path=model_path, property_path=property_path,
            timeout=effective_timeout, timeout_slack=args.timeout_slack,
            max_domains=max_domains,
            domain_batch_metrics_jsonl=domain_batch_metrics_jsonl,
            extra_args=args.extra_arg,
            warmup_runs=args.warmup_runs,
            rerun_presearch=args.rerun_presearch,
            raw_artifact_dir=row_artifact_dir,
            process_env=process_env,
            # The bounded runner always preserves the complete manifest row
            # sequence, so this identity is exact for full, sampled, and
            # explicitly indexed runs alike.
            source_index_zero_based=index,
        )
        counts[result.result] = counts.get(result.result, 0) + 1
        LOG.info(
            "  -> %s (%.2fs, domains=%d, verified=%d)",
            result.result, result.elapsed,
            result.domains_explored, result.domains_verified,
        )
        results.append(result)

    binding_error = _validate_expected_row_bindings(
        selected, args.expected_row_binding
    )
    if binding_error is not None:
        LOG.error(
            "Benchmark input changed during execution: %s; refusing to emit evidence",
            binding_error,
        )
        return 1
    try:
        final_preset_sha256 = _sha256_file(preset_path)
    except OSError as error:
        LOG.error("Cannot revalidate preset after execution: %s", error)
        return 1
    if final_preset_sha256 != preset_sha256:
        LOG.error(
            "Preset changed during the benchmark run: expected %s, observed %s; "
            "refusing to emit evidence",
            preset_sha256,
            final_preset_sha256,
        )
        return 1
    try:
        final_ny_sha256 = _sha256_file(ny_binary)
    except OSError as error:
        LOG.error("Cannot revalidate private NY binary after execution: %s", error)
        return 1
    if final_ny_sha256 != provenance.sha256:
        LOG.error(
            "Private NY binary changed during the benchmark run: expected %s, observed %s; "
            "refusing to emit stale-provenance evidence",
            provenance.sha256,
            final_ny_sha256,
        )
        return 1

    if staged_ny_receipt is not None and receipt_evidence is not None:
        try:
            final_receipt_evidence = _authenticate_ny_receipt(
                ny_binary, staged_ny_receipt, process_env
            )
        except RuntimeError as error:
            LOG.error(
                "Cannot revalidate NY receipt/source after execution: %s; "
                "refusing to emit evidence",
                error,
            )
            return 1
        if final_receipt_evidence != receipt_evidence:
            LOG.error(
                "Private NY receipt changed during the benchmark run; "
                "refusing to emit stale-provenance evidence"
            )
            return 1

    if os.path.lexists(output_path):
        LOG.error(
            "Output CSV appeared during execution; refusing to overwrite evidence: %s",
            output_path,
        )
        return 1
    write_results(
        output_path,
        results,
        provenance,
        RunProvenance(
            preset_sha256=preset_sha256,
            parent_env_json=parent_env_json,
            parent_env_sha256=parent_env_sha256,
        ),
    )
    LOG.info("Saved results to %s", output_path)
    LOG.info(json.dumps({"counts": counts, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
