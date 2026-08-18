#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Qualify one attack acceleration axis on scored CIFAR100 rows.

This is deliberately narrower than a whole-category sweep.  It runs two exact,
score-relevant cohorts at the official 100 second budget:

* every official-field SAT for ``CIFAR100_resnet_large`` (15 rows at the pinned
  2025 results revision), which is the opportunity set for the measured +9;
* the 15 official-field UNSAT rows in the first 32 large-model instances.  Those
  are the exact cardinality/order cohort on which the restored CROWN-IBP
  collector previously produced 15 proofs before the BatchNorm-fold regression.

Each row is run in both arms with alternating arm order.  Every child gets its
own cgroup memory ceiling and the launch environment is scrubbed of inherited
NY_/AY_ experiments.  A promotion pass requires all 60 runs, >=9 SAT gains,
zero SAT losses, all 15 proof guards solved in both arms, no field
contradictions, and flight-sidecar authentication of the selected arm.

Usage:
  python3 scripts/cifar100_attack_fast_kernels_qualify.py \
      --official-root /path/to/vnncomp2025_results

The historical/default ``--axis fast-kernels`` campaign toggles
``NY_ATTACK_POINT_FAST_KERNELS`` exactly as before.  The
``--axis wrapper-vjp`` campaign instead holds that legacy gate off, pins the
wrapper width at K=64, and compares the shipped exact-VJP pre-wave against its
exact kill switch (``NY_ORT_REFINE_VJP_BATCH=1`` versus ``0``).  Wrapper-VJP
arms are accepted only when their logs carry the expected engaged/disabled
telemetry disposition in addition to the flight-sidecar environment seal.

Use ``--dry-run`` to inspect the exact cohort and commands without executing.
Partial ``--limit`` runs are useful canaries but can never pass promotion.
After an interruption, use ``--resume --out EXISTING_DIR`` with the exact launch
arguments. Authenticated atomic row fragments are skipped. An arm whose attempt
was presealed but did not commit a row makes that directory permanently
non-promotable: start a new output directory rather than retrying it. A summary
is actionable only when the directory also contains its atomic
``completion.json`` seal. This campaign authorizes at most a CIFAR100
ResNet-large route; it never authorizes a global/default-on gate.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = REPO / "benchmarks/vnncomp2025/benchmarks/cifar100_2024"
CANONICAL_INPUTS_PATH = (
    REPO / "scripts/cifar100_attack_fast_kernels_canonical_inputs.json"
)
TOOLS = (
    "alpha_beta_crown",
    "neuralsat",
    "cora",
    "pyrat",
    "nnv",
    "rover",
)
OFFICIAL_BUDGET_SECS = 100
# Promotion uses the exact official budget. The outer watchdog remains longer
# only so a timed-out child can flush evidence; watchdog grace never counts as
# a scored solve.
OFFICIAL_CLOCK_TOLERANCE_SECS = 0.0
MAX_LOAD_PER_LOGICAL_CORE = 1.0
EXPECTED_GAIN_ROWS = 15
EXPECTED_GUARD_ROWS = 15
MINIMUM_PROMOTION_GAINS = 9
CATEGORY = "cifar100_2024"
FLIGHT_SCHEMA_VERSION = 3
SUPPORTED_FLIGHT_SCHEMA_VERSIONS = frozenset({2, FLIGHT_SCHEMA_VERSION})
LEVER_RECEIPT_SCHEMA = "ny-levers/receipt/v2"
LEVER_SOURCES = frozenset(
    {"default", "config", "legacy_env", "legacy_env_rejected"}
)
LEVER_BUCKETS = frozenset({"default_on", "auto", "cli", "debug"})
LEVER_MOATS = frozenset({"none", "low", "high"})
LEVER_PROVENANCE = frozenset(
    {"value_neutral", "measured", "unmeasured", "guard"}
)
LEVER_NAME = re.compile(r"^NY_[A-Z0-9_]+$")
LAUNCH_SCHEMA = "ny_cifar100_attack_fast_kernels_launch_v2"
INPUTS_SCHEMA = "ny_cifar100_attack_fast_kernels_inputs_v1"
SUMMARY_SCHEMA = "ny_cifar100_attack_fast_kernels_qualification_v2"
COMPLETION_SCHEMA = "ny_cifar100_attack_fast_kernels_completion_v1"
ATTEMPT_SCHEMA = "ny_cifar100_attack_fast_kernels_attempt_v1"
ATTEMPT_SCOPE_IDENTITY_SCHEMA = "ny_cifar100_attack_fast_kernels_scope_identity_v1"
ROW_FRAGMENT_SCHEMA = "ny_cifar100_attack_fast_kernels_row_v1"
CANONICAL_INPUTS_SCHEMA = "ny_cifar100_attack_fast_kernels_canonical_inputs_v1"
PROMOTION_SCOPE = "cifar100_2024:CIFAR100_resnet_large-only"
FAST_KERNELS_AXIS = "fast-kernels"
WRAPPER_VJP_AXIS = "wrapper-vjp"
DEFAULT_AXIS = FAST_KERNELS_AXIS
EXPERIMENT_AXES = (FAST_KERNELS_AXIS, WRAPPER_VJP_AXIS)
WRAPPER_VJP_WIDTH = 64
WRAPPER_VJP_LOG_PREFIX = "ORT-refine grad lane: exact-VJP pre-wave"
WRAPPER_VJP_ARMED_MARKER = f"{WRAPPER_VJP_LOG_PREFIX} armed"
WRAPPER_VJP_SUCCESS_PREFIX = (
    "ORT-refine grad lane: ORT-confirmed violation in exact-VJP pre-wave"
)
WRAPPER_VJP_DECLINED_PREFIX = f"{WRAPPER_VJP_LOG_PREFIX} declined"
WRAPPER_VJP_DECLINE_REASONS = re.compile(
    rf"^{re.escape(WRAPPER_VJP_DECLINED_PREFIX)} "
    rf"\(reason=([a-z][a-z0-9_]*)\)$",
    re.MULTILINE,
)
# Both terminal dispositions include ``(<N> wave steps, ...)``.  Requiring one
# rules out an on-arm that merely reached allocation and then failed before the
# wave produced an auditable disposition.
WRAPPER_VJP_TERMINAL_STEPS = re.compile(
    rf"(?:{re.escape(WRAPPER_VJP_LOG_PREFIX)} found no trusted violation|"
    rf"{re.escape(WRAPPER_VJP_SUCCESS_PREFIX)})[^\n]*\([^\n)]*?"
    rf"\b(\d+) wave steps(?:,|\))"
)
VALID_STATUSES = {"sat", "unsat", "timeout", "unknown", "error"}
VALID_BACKENDS = {"cpu-only", "cuda", "metal", "gpu"}
SYSTEMD_SCOPE_REQUIRED_PROPERTIES = (
    "LoadState",
    "Result",
)
SYSTEMD_SCOPE_OPTIONAL_PROPERTIES = (
    "OOMKills",
    "MemoryPeak",
    "MemoryMax",
    "MemorySwapPeak",
    "MemorySwapMax",
)
SYSTEMD_SCOPE_PROPERTIES = (
    *SYSTEMD_SCOPE_REQUIRED_PROPERTIES,
    *SYSTEMD_SCOPE_OPTIONAL_PROPERTIES,
)
SYSTEMD_SCOPE_QUERY_TIMEOUT_SECS = 5
SYSTEMD_SCOPE_RESET_TIMEOUT_SECS = 5
SYSTEMD_U64_UNSET = (1 << 64) - 1
RESULT_FIELDS = (
    "cohort",
    "cohort_index",
    "order_index",
    "arm",
    "onnx",
    "vnnlib",
    "ground_truth",
    "status",
    "wall_secs",
    "exit_code",
    "attack_steps",
    "arm_authenticated",
    "trusted_upfront_sat",
    "backend_kind",
    "regime_sha256",
    "flight_publish_secs",
    "flight_terminal_secs",
    "within_official_cutoff",
    "peak_load_per_core",
    "load_acceptable",
    "result_sha256",
    "flight_sha256",
    "log_sha256",
)

# These inherited knobs can materially change CPU/GPU selection, thread count,
# allocator behaviour, or dynamic-library resolution.  Thread/allocator tuning
# is scrubbed to a canonical baseline; device/library selectors are preserved
# but sealed in launch.json so a resumed campaign cannot silently change them.
# Shell bookkeeping is also scrubbed: `_` and `SHLVL` vary with the invocation
# path but cannot change verifier semantics, and sealing them would make a
# documented resume spuriously fail after moving through an extra shell.
SCRUBBED_COMPUTE_ENV_KEYS = frozenset(
    {
        "_",
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "SHLVL",
        "VECLIB_MAXIMUM_THREADS",
    }
)
SEALED_COMPUTE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)


class QualificationError(RuntimeError):
    """The qualification inputs or evidence are incomplete."""


@dataclass(frozen=True)
class Target:
    cohort: str
    cohort_index: int
    onnx: str
    vnnlib: str
    ground_truth: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one stable regular file, failing if it changes during the read."""

    try:
        resolved = path.resolve(strict=True)
        descriptor = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise QualificationError(
            f"cannot open immutable input {path}: {error}"
        ) from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QualificationError(f"immutable input is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def fingerprint(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    try:
        path_after = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise QualificationError(
            f"immutable input disappeared while hashing: {path}"
        ) from error
    if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(
        path_after
    ):
        raise QualificationError(f"immutable input changed while hashing: {path}")
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
    except OSError as error:
        raise QualificationError(
            f"cannot identify immutable input {path}: {error}"
        ) from error
    if not stat.S_ISREG(before.st_mode):
        raise QualificationError(f"immutable input is not a regular file: {path}")
    digest = sha256_file(resolved)
    after = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise QualificationError(f"immutable input changed while identifying: {path}")
    return {
        "path": str(resolved),
        "size": after.st_size,
        "sha256": digest,
    }


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def compact_file_identity(identity: dict[str, object]) -> dict[str, object]:
    """Drop host-specific paths from an input identity before canonical comparison."""

    return {"size": identity["size"], "sha256": identity["sha256"]}


def require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise QualificationError(f"{label} is not an exact SHA-256 digest")
    return value


def load_canonical_inputs_manifest() -> dict[str, object]:
    manifest = load_json_object(CANONICAL_INPUTS_PATH, CANONICAL_INPUTS_SCHEMA)
    expected_fields = {
        "schema",
        "benchmark_git_head",
        "benchmark_manifest_sha256",
        "canonical_selected_inputs_sha256",
        "official_git_head",
        "official_results_manifest_sha256",
        "selected_model",
        "selected_vnnlib_count",
        "source",
        "targets_csv_sha256",
    }
    if set(manifest) != expected_fields:
        raise QualificationError("canonical input manifest has unexpected fields")
    for field in ("benchmark_git_head", "official_git_head"):
        value = manifest.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise QualificationError(f"canonical input manifest has invalid {field}")
    for field in (
        "benchmark_manifest_sha256",
        "canonical_selected_inputs_sha256",
        "official_results_manifest_sha256",
        "targets_csv_sha256",
    ):
        require_digest(manifest.get(field), f"canonical manifest {field}")
    if (
        type(manifest.get("selected_vnnlib_count")) is not int
        or manifest["selected_vnnlib_count"] <= 0
    ):
        raise QualificationError("canonical selected VNNLIB count is invalid")
    model = manifest.get("selected_model")
    if not isinstance(model, dict) or set(model) != {
        "asset",
        "logical_name",
        "sha256",
        "size",
    }:
        raise QualificationError("canonical selected-model identity is invalid")
    if (
        not all(
            isinstance(model.get(field), str) and bool(model[field])
            for field in ("asset", "logical_name")
        )
        or model["asset"] != model["logical_name"]
    ):
        raise QualificationError("canonical selected-model names are invalid")
    require_digest(model.get("sha256"), "canonical selected-model digest")
    if type(model.get("size")) is not int or model["size"] <= 0:
        raise QualificationError("canonical selected-model size is invalid")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "benchmark_repository",
            "official_results_repository",
        }
        or not all(isinstance(value, str) and value for value in source.values())
    ):
        raise QualificationError("canonical input source attribution is invalid")
    return manifest


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_regular_file(path: Path) -> None:
    """Make one completed arm artifact durable without following symlinks."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise QualificationError(
            f"cannot open arm artifact for fsync: {path}: {error}"
        ) from error
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise QualificationError(f"arm artifact is not a regular file: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: Path, data: bytes, *, require_absent: bool = False
) -> None:
    """Durably publish bytes by same-directory rename and parent fsync."""

    if path.is_symlink():
        raise QualificationError(f"refusing to replace symlink evidence file: {path}")
    if require_absent and path.exists():
        raise QualificationError(f"evidence file already exists: {path}")
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(data)
            sink.flush()
            os.fsync(sink.fileno())
        if require_absent and path.exists():
            raise QualificationError(f"evidence file appeared concurrently: {path}")
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: Path, payload: object, *, require_absent: bool = False
) -> None:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, body, require_absent=require_absent)


def load_json_object(path: Path, schema: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"missing regular evidence file: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"malformed JSON evidence {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise QualificationError(f"wrong or missing schema in {path}")
    return payload


@contextlib.contextmanager
def campaign_lock(output: Path) -> Iterator[None]:
    lock_path = output / ".qualification.lock"
    if lock_path.is_symlink():
        raise QualificationError(f"campaign lock must not be a symlink: {lock_path}")
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QualificationError(f"campaign is already active: {output}") from error
        yield


def git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_bytes(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise QualificationError(
            f"git {' '.join(args)} failed in {root}: {detail or result.returncode}"
        )
    return result.stdout


def capture_repo_identity(root: Path) -> dict[str, object]:
    head = git_value(root, "rev-parse", "HEAD")
    if head is None or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise QualificationError(f"cannot resolve exact Git HEAD in {root}")
    status = git_bytes(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    ).decode("utf-8", errors="strict")
    tracked_diff = git_bytes(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    untracked_raw = git_bytes(root, "ls-files", "--others", "--exclude-standard", "-z")
    untracked: dict[str, dict[str, object]] = {}
    for encoded in untracked_raw.split(b"\0"):
        if not encoded:
            continue
        try:
            relative = encoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise QualificationError("NY has a non-UTF-8 untracked path") from error
        untracked[relative] = file_identity(root / relative)
    return {
        "root": str(root.resolve(strict=True)),
        "head": head,
        "clean": status == "",
        "status": status,
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "untracked": untracked,
    }


def directory_identity(root: Path) -> dict[str, object]:
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise QualificationError(
            f"cannot resolve input directory {root}: {error}"
        ) from error
    if not resolved.is_dir():
        raise QualificationError(f"input directory is missing: {root}")
    before_paths = sorted(resolved.rglob("*"))
    files: dict[str, dict[str, object]] = {}
    for path in before_paths:
        if path.is_symlink():
            raise QualificationError(f"input directory contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise QualificationError(f"input directory contains a non-file: {path}")
        relative = path.relative_to(resolved).as_posix()
        files[relative] = file_identity(path)
    after_paths = sorted(resolved.rglob("*"))
    if [path.relative_to(resolved) for path in before_paths] != [
        path.relative_to(resolved) for path in after_paths
    ]:
        raise QualificationError(f"input directory changed while identifying: {root}")
    return {"root": str(resolved), "files": files}


def load_results(path: Path) -> dict[tuple[str, str], str]:
    if not path.is_file():
        raise QualificationError(f"missing official results: {path}")
    out: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.reader(source):
            if len(row) < 5:
                continue
            key = (Path(row[1]).name, Path(row[2]).name)
            status = row[4].strip()
            if key in out and out[key] != status:
                raise QualificationError(
                    f"ambiguous official verdict for {key} in {path}"
                )
            out[key] = status
    return out


def load_instances(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise QualificationError(f"missing benchmark manifest: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        return [
            (Path(row[0]).name, Path(row[1]).name)
            for row in csv.reader(source)
            if len(row) >= 2
        ]


def derive_targets(
    official_root: Path,
    benchmark_root: Path,
) -> tuple[list[Target], dict[str, dict[tuple[str, str], str]], tuple[str, ...]]:
    official = {
        tool: load_results(official_root / tool / "2025_cifar100_2024/results.csv")
        for tool in TOOLS
    }
    instances = load_instances(benchmark_root / "instances.csv")
    canonical = official["alpha_beta_crown"]
    missing = [key for key in instances if key not in canonical]
    if missing:
        raise QualificationError(
            f"official results miss {len(missing)} benchmark rows; first is {missing[0]}"
        )

    # An UNSAT-reporting tool that contradicts any field SAT must not define a
    # proof guard.  This reproduces the repository's existing CIFAR target-set
    # rule and excludes NNV's known contradictory UNSAT reports.
    field_sat = {
        key
        for key in instances
        if any(official[tool].get(key) == "sat" for tool in TOOLS)
    }
    sound_unsat_tools = tuple(
        tool
        for tool in TOOLS
        if not any(official[tool].get(key) == "unsat" for key in field_sat)
    )

    large = [key for key in instances if "resnet_large" in key[0]]
    gain_keys = [key for key in large if key in field_sat]
    guard_keys = [
        key
        for key in large[:32]
        if any(official[tool].get(key) == "unsat" for tool in sound_unsat_tools)
    ]
    if len(gain_keys) != EXPECTED_GAIN_ROWS:
        raise QualificationError(
            f"expected {EXPECTED_GAIN_ROWS} large-model field SAT rows, got {len(gain_keys)}"
        )
    if len(guard_keys) != EXPECTED_GUARD_ROWS:
        raise QualificationError(
            "the first-32 restoration guard drifted: expected "
            f"{EXPECTED_GUARD_ROWS} sound-field UNSAT rows, got {len(guard_keys)}"
        )
    if set(gain_keys) & set(guard_keys):
        raise QualificationError("gain and proof-guard cohorts overlap")

    targets = [
        Target("gain", index, onnx, vnnlib, "sat")
        for index, (onnx, vnnlib) in enumerate(gain_keys, 1)
    ]
    targets.extend(
        Target("guard", index, onnx, vnnlib, "unsat")
        for index, (onnx, vnnlib) in enumerate(guard_keys, 1)
    )
    return targets, official, sound_unsat_tools


def target_record(target: Target) -> dict[str, object]:
    return {
        "cohort": target.cohort,
        "cohort_index": target.cohort_index,
        "onnx": target.onnx,
        "vnnlib": target.vnnlib,
        "ground_truth": target.ground_truth,
    }


def render_targets(targets: Iterable[Target]) -> bytes:
    sink = io.StringIO(newline="")
    writer = csv.writer(sink, lineterminator="\n")
    writer.writerow(("cohort", "cohort_index", "onnx", "vnnlib", "ground_truth"))
    for target in targets:
        writer.writerow(
            (
                target.cohort,
                target.cohort_index,
                target.onnx,
                target.vnnlib,
                target.ground_truth,
            )
        )
    return sink.getvalue().encode("utf-8")


def load_targets_file(path: Path) -> tuple[list[Target], bytes]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"missing regular target evidence: {path}")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise QualificationError(
            f"torn target evidence (missing final newline): {path}"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        expected_fields = [
            "cohort",
            "cohort_index",
            "onnx",
            "vnnlib",
            "ground_truth",
        ]
        if reader.fieldnames != expected_fields:
            raise QualificationError(f"target header drift in {path}")
        targets: list[Target] = []
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise QualificationError(f"malformed target row {number} in {path}")
            raw_index = row["cohort_index"]
            try:
                cohort_index = int(raw_index)
            except ValueError as error:
                raise QualificationError(
                    f"non-integer target index at row {number} in {path}"
                ) from error
            if str(cohort_index) != raw_index or cohort_index <= 0:
                raise QualificationError(
                    f"non-canonical target index at row {number} in {path}"
                )
            target = Target(
                cohort=row["cohort"],
                cohort_index=cohort_index,
                onnx=row["onnx"],
                vnnlib=row["vnnlib"],
                ground_truth=row["ground_truth"],
            )
            if target.cohort not in {"gain", "guard"}:
                raise QualificationError(
                    f"invalid target cohort at row {number} in {path}"
                )
            expected_ground_truth = "sat" if target.cohort == "gain" else "unsat"
            if target.ground_truth != expected_ground_truth:
                raise QualificationError(
                    f"invalid target ground truth at row {number} in {path}"
                )
            if not target.onnx or not target.vnnlib:
                raise QualificationError(
                    f"empty target identity at row {number} in {path}"
                )
            targets.append(target)
    except (UnicodeDecodeError, csv.Error) as error:
        raise QualificationError(
            f"malformed target evidence {path}: {error}"
        ) from error
    if len({(target.cohort, target.cohort_index) for target in targets}) != len(
        targets
    ):
        raise QualificationError(f"duplicate target identity in {path}")
    return targets, raw


def arm_order(order_index: int) -> tuple[str, str]:
    return ("off", "on") if order_index % 2 else ("on", "off")


def validate_axis(axis: str) -> str:
    if axis not in EXPERIMENT_AXES:
        raise QualificationError(f"unknown experiment axis: {axis!r}")
    return axis


def axis_from_args(args: argparse.Namespace) -> str:
    """Read an axis while keeping old in-process callers on the legacy arm."""

    return validate_axis(str(getattr(args, "axis", DEFAULT_AXIS)))


def axis_from_launch(launch: dict[str, object]) -> str:
    inputs = launch.get("immutable_inputs")
    if not isinstance(inputs, dict):
        raise QualificationError("launch immutable inputs are invalid")
    campaign = inputs.get("campaign")
    if not isinstance(campaign, dict):
        raise QualificationError("launch campaign is invalid")
    # Launches produced before the named-axis option are, by construction, the
    # historical fast-kernel campaign.
    return validate_axis(str(campaign.get("experiment_axis", DEFAULT_AXIS)))


def forced_arm_environment(axis: str, arm: str) -> dict[str, str]:
    validate_axis(axis)
    if arm not in {"off", "on"}:
        raise QualificationError(f"invalid experiment arm: {arm!r}")
    common = {
        "NY_PHASE_TELEMETRY": "1",
        "OMP_NUM_THREADS": "1",
    }
    if axis == FAST_KERNELS_AXIS:
        return {
            "NY_ATTACK_POINT_FAST_KERNELS": "1" if arm == "on" else "0",
            **common,
        }
    return {
        # Hold the historical sequential-gradient acceleration constant so the
        # sole changed variable is the wrapper's bounded wide pre-wave.
        "NY_ATTACK_POINT_FAST_KERNELS": "0",
        # Admit only the bounded wrapper pre-wave under this harness's isolated
        # MemoryMax/MemorySwapMax scope. The narrow override deliberately does
        # not enable margin-row BaB or post-BaB graph-heavy tails. Pin it in
        # BOTH arms; the exact-VJP kill switch remains the sole A/B variable.
        "NY_ORT_REFINE_VJP_UNDER_MEMORY_LIMIT": "1",
        "NY_ORT_REFINE_VJP_BATCH": "1" if arm == "on" else "0",
        "NY_ORT_REFINE_VJP_K": str(WRAPPER_VJP_WIDTH),
        **common,
    }


def wrapper_vjp_log_authenticated(log_text: str, arm: str) -> bool:
    """Authenticate reachability for the wrapper-VJP axis from exact markers."""

    armed = WRAPPER_VJP_ARMED_MARKER in log_text
    terminal_steps = WRAPPER_VJP_TERMINAL_STEPS.findall(log_text)
    declined = wrapper_vjp_decline_reasons(log_text)
    if arm == "off":
        # The kill switch returns before emitting any wrapper marker.  Reject
        # both a crossed arm and partial/unexpected wrapper execution.
        return (
            not armed
            and not terminal_steps
            and WRAPPER_VJP_LOG_PREFIX not in log_text
            and WRAPPER_VJP_SUCCESS_PREFIX not in log_text
            and not declined
        )
    if arm != "on":
        return False
    return (
        not declined
        and armed
        and any(int(steps) > 0 for steps in terminal_steps)
    )


def wrapper_vjp_decline_reasons(log_text: str) -> tuple[str, ...]:
    """Return exact pre-arm decline reasons without granting authentication."""

    return tuple(WRAPPER_VJP_DECLINE_REASONS.findall(log_text))


def axis_log_authenticated(log_text: str, axis: str, arm: str) -> bool:
    validate_axis(axis)
    if axis == FAST_KERNELS_AXIS:
        # Preserve the historical campaign's authentication contract. Positive
        # fast-kernel steps remain required separately for every claimed gain.
        return True
    return wrapper_vjp_log_authenticated(log_text, arm)


def _valid_lever_value(value: object) -> bool:
    """Whether ``value`` is one of the scalar shapes emitted by ny-levers."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if type(value) is int:
        return 0 <= value <= (1 << 64) - 1
    return type(value) is float and math.isfinite(value)


def _valid_v3_lever_state(
    value: object, *, require_resolved: bool, ambient_env: object
) -> bool:
    """Validate the versioned, count-consistent Phase-0c lever envelope."""
    if not isinstance(value, dict) or not isinstance(value.get("status"), str):
        return False
    if value["status"] == "not_materialized":
        return not require_resolved and set(value) == {"status"}
    if value["status"] == "invalid_config":
        return (
            not require_resolved
            and set(value) == {"status", "reason"}
            and isinstance(value["reason"], str)
            and bool(value["reason"])
        )
    if value["status"] != "resolved" or set(value) != {"status", "receipt"}:
        return False

    receipt = value["receipt"]
    expected_fields = {
        "schema",
        "lever_count",
        "env_present",
        "env_accepted",
        "env_rejected",
        "levers",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_fields:
        return False
    if receipt["schema"] != LEVER_RECEIPT_SCHEMA:
        return False
    count_fields = (
        "lever_count",
        "env_present",
        "env_accepted",
        "env_rejected",
    )
    if any(
        type(receipt[field]) is not int or receipt[field] < 0
        for field in count_fields
    ):
        return False
    levers = receipt["levers"]
    if (
        not isinstance(levers, list)
        or not levers
        or receipt["lever_count"] != len(levers)
        or not isinstance(ambient_env, dict)
    ):
        return False

    names: set[str] = set()
    source_counts = {"legacy_env": 0, "legacy_env_rejected": 0}
    required_entry_fields = {
        "name",
        "value",
        "source",
        "bucket",
        "moat",
        "provenance",
    }
    optional_entry_fields = {"rejected_raw", "env_utf8"}
    for entry in levers:
        if (
            not isinstance(entry, dict)
            or not required_entry_fields <= set(entry)
            or not set(entry) <= required_entry_fields | optional_entry_fields
        ):
            return False
        name = entry["name"]
        source = entry["source"]
        if (
            not isinstance(name, str)
            or LEVER_NAME.fullmatch(name) is None
            or name in names
        ):
            return False
        if not isinstance(source, str) or source not in LEVER_SOURCES:
            return False
        if (
            not isinstance(entry["bucket"], str)
            or entry["bucket"] not in LEVER_BUCKETS
            or not isinstance(entry["moat"], str)
            or entry["moat"] not in LEVER_MOATS
        ):
            return False
        if (
            not isinstance(entry["provenance"], str)
            or entry["provenance"] not in LEVER_PROVENANCE
        ):
            return False
        if (
            entry["provenance"] == "unmeasured"
            and entry["bucket"] == "default_on"
        ) or (
            entry["provenance"] == "guard" and entry["bucket"] == "auto"
        ):
            return False
        if not _valid_lever_value(entry["value"]):
            return False
        env_backed = source in source_counts
        if env_backed != (name in ambient_env):
            return False
        if source == "legacy_env":
            if "rejected_raw" in entry or type(entry.get("env_utf8")) is not bool:
                return False
        elif source == "legacy_env_rejected":
            if (
                not isinstance(entry.get("rejected_raw"), str)
                or type(entry.get("env_utf8")) is not bool
                or entry["rejected_raw"] != ambient_env[name]
            ):
                return False
        elif set(entry) & optional_entry_fields:
            return False
        names.add(name)
        if source in source_counts:
            source_counts[source] += 1

    accepted = source_counts["legacy_env"]
    rejected = source_counts["legacy_env_rejected"]
    return (
        receipt["env_accepted"] == accepted
        and receipt["env_rejected"] == rejected
        and receipt["env_present"] == accepted + rejected
    )


def flight_evidence(
    path: Path,
    arm: str,
    status: str,
    axis: str = DEFAULT_AXIS,
) -> dict[str, object]:
    invalid = {
        "authenticated": False,
        "trusted_sat": False,
        "backend": "missing",
        "regime_sha256": "",
        "publish_secs": None,
        "terminal_secs": None,
        "within_official_cutoff": False,
        "peak_load_per_core": None,
        "load_acceptable": False,
    }
    if path.is_symlink() or not path.is_file():
        return invalid
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {**invalid, "backend": "invalid"}
    if not isinstance(payload, dict):
        return {**invalid, "backend": "invalid"}

    backend = payload.get("backend_kind")
    backend_summary = payload.get("backend_summary")
    if not isinstance(backend, str) or not backend:
        backend = "invalid"
    expected_env = forced_arm_environment(axis, arm)
    host = payload.get("host")
    host_valid = (
        isinstance(host, dict)
        and set(host) == {"hostname", "cpu_model", "logical_cores", "ram_bytes"}
        and isinstance(host.get("hostname"), str)
        and bool(host["hostname"])
        and isinstance(host.get("cpu_model"), str)
        and bool(host["cpu_model"])
        and type(host.get("logical_cores")) is int
        and host["logical_cores"] > 0
        and type(host.get("ram_bytes")) is int
        and host["ram_bytes"] > 0
    )

    def validated_load(field: str) -> list[float] | None:
        value = payload.get(field)
        if not isinstance(value, list) or len(value) != 3:
            return None
        out: list[float] = []
        for sample in value:
            if (
                isinstance(sample, bool)
                or not isinstance(sample, (int, float))
                or not math.isfinite(float(sample))
                or float(sample) < 0.0
            ):
                return None
            out.append(float(sample))
        return out

    load_begin = validated_load("load_avg_at_begin")
    load_end = validated_load("load_avg_at_end")
    regime_sha256 = ""
    peak_load_per_core: float | None = None
    load_acceptable = False
    if host_valid and isinstance(backend_summary, str) and backend_summary:
        regime_sha256 = canonical_json_sha256(
            {
                "backend_kind": backend,
                "backend_summary": backend_summary,
                "host": host,
            }
        )
    if host_valid and load_begin is not None and load_end is not None:
        peak_load_per_core = max(load_begin[0], load_end[0]) / int(
            host["logical_cores"]
        )
        load_acceptable = peak_load_per_core <= MAX_LOAD_PER_LOGICAL_CORE
    schema_version = payload.get("schema_version")
    schema_valid = (
        type(schema_version) is int
        and schema_version in SUPPORTED_FLIGHT_SCHEMA_VERSIONS
        and (
            schema_version != FLIGHT_SCHEMA_VERSION
            or _valid_v3_lever_state(
                payload.get("levers"),
                require_resolved=True,
                ambient_env=payload.get("ambient_env"),
            )
        )
    )
    authenticated = (
        schema_valid
        and payload.get("category") == CATEGORY
        and payload.get("budget_secs") == OFFICIAL_BUDGET_SECS
        and not isinstance(payload.get("budget_secs"), bool)
        and payload.get("ambient_env") == expected_env
        and isinstance(backend, str)
        and backend in VALID_BACKENDS
        and isinstance(backend_summary, str)
        and bool(backend_summary)
        and host_valid
        and load_begin is not None
        and load_end is not None
    )
    events_raw = payload.get("events")
    if not isinstance(events_raw, list) or not events_raw:
        return {
            "authenticated": False,
            "trusted_sat": False,
            "backend": str(backend),
            "regime_sha256": regime_sha256,
            "publish_secs": None,
            "terminal_secs": None,
            "within_official_cutoff": False,
            "peak_load_per_core": peak_load_per_core,
            "load_acceptable": load_acceptable,
        }

    events: list[dict[str, object]] = []
    previous_at = -1.0
    for event in events_raw:
        if not isinstance(event, dict):
            authenticated = False
            # Keep parsing fail-closed without letting malformed JSON values
            # escape as an AttributeError in the lifecycle checks below.
            continue
        method = event.get("method")
        disposition = event.get("status")
        at_secs = event.get("at_secs")
        expected_event_fields = {"method", "status", "at_secs"}
        if "reason" in event:
            expected_event_fields.add("reason")
        if (
            set(event) != expected_event_fields
            or not isinstance(method, str)
            or not method
            or disposition not in {"ran", "skipped", "not_reached", "complete"}
            or isinstance(at_secs, bool)
            or not isinstance(at_secs, (int, float))
            or not math.isfinite(float(at_secs))
            or float(at_secs) < 0.0
            or float(at_secs) < previous_at
            or ("reason" in event and not isinstance(event.get("reason"), str))
            or (method != "run_complete" and disposition == "complete")
        ):
            authenticated = False
        else:
            previous_at = float(at_secs)
        events.append(event)

    if len(events) != len(events_raw):
        authenticated = False

    terminal_indices = [
        index
        for index, event in enumerate(events)
        if event.get("method") == "run_complete"
    ]
    authenticated = (
        authenticated
        and terminal_indices == [len(events) - 1]
        and events[-1].get("status") == "complete"
        and events[-1].get("reason") == status
    )
    all_publish_indices = [
        index
        for index, event in enumerate(events[:-1])
        if event.get("method") == "result_publish"
    ]
    publish_indices = [
        index
        for index in all_publish_indices
        if events[index].get("status") == "ran"
        and events[index].get("reason") == status
    ]
    authenticated = (
        authenticated and len(all_publish_indices) == 1 and len(publish_indices) == 1
    )
    upfront_indices = [
        index
        for index, event in enumerate(events[:-1])
        if event.get("method") == "upfront_attack"
    ]
    authenticated = authenticated and len(upfront_indices) == 1
    trusted_indices = [
        index
        for index, event in enumerate(events[:-1])
        if event.get("method") == "upfront_attack"
        and event.get("status") == "ran"
        and event.get("reason")
        == "sat: trusted-oracle gate confirmed the upfront candidate"
    ]
    trusted_sat = (
        status == "sat"
        and len(trusted_indices) == 1
        and len(publish_indices) == 1
        and trusted_indices[0] < publish_indices[0]
    )
    publish_secs = (
        float(events[publish_indices[0]]["at_secs"])
        if len(publish_indices) == 1
        else None
    )
    terminal_secs = (
        float(events[-1]["at_secs"]) if terminal_indices == [len(events) - 1] else None
    )
    cutoff = OFFICIAL_BUDGET_SECS + OFFICIAL_CLOCK_TOLERANCE_SECS
    within_official_cutoff = (
        publish_secs is not None
        and terminal_secs is not None
        and publish_secs <= cutoff
        and terminal_secs <= cutoff
    )
    return {
        "authenticated": authenticated,
        "trusted_sat": trusted_sat,
        "backend": str(backend),
        "regime_sha256": regime_sha256,
        "publish_secs": publish_secs,
        "terminal_secs": terminal_secs,
        "within_official_cutoff": within_official_cutoff,
        "peak_load_per_core": peak_load_per_core,
        "load_acceptable": load_acceptable,
    }


def parse_steps(log_text: str, axis: str = DEFAULT_AXIS) -> int | None:
    validate_axis(axis)
    if axis == WRAPPER_VJP_AXIS:
        matches = WRAPPER_VJP_TERMINAL_STEPS.findall(log_text)
        # More than one gradient-guided lane can be reached in one process.
        # Promotion needs proof that at least one engaged wave made progress,
        # so retain the maximum audited step count rather than whichever lane
        # happened to log last.
        return max(map(int, matches)) if matches else None
    matches = re.findall(r"(\d+) gradient steps", log_text)
    return int(matches[-1]) if matches else None


def scrubbed_environment(arm: str, axis: str = DEFAULT_AXIS) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("NY_")
        and not key.startswith("AY_")
        and not key.startswith("MIMALLOC_")
        and key not in SCRUBBED_COMPUTE_ENV_KEYS
    }
    environment.update(forced_arm_environment(axis, arm))
    return environment


def compute_environment_contract(axis: str = DEFAULT_AXIS) -> dict[str, object]:
    validate_axis(axis)
    environments = {arm: scrubbed_environment(arm, axis) for arm in ("off", "on")}
    return {
        "experiment_axis": axis,
        # Seal every byte passed to the child without copying credentials or
        # other sensitive inherited values into launch.json. Recomputing the
        # launch inputs before and after every arm detects any passthrough drift.
        "effective_child_environment_sha256": {
            arm: canonical_json_sha256(environment)
            for arm, environment in environments.items()
        },
        "forced_by_arm": {
            arm: forced_arm_environment(axis, arm) for arm in ("off", "on")
        },
        # Retain the historical key for readers that only consume the common
        # process-wide pins. ``forced_by_arm`` is authoritative for the named
        # experiment variable and all axis-specific constants.
        "forced": {
            "NY_PHASE_TELEMETRY": "1",
            "OMP_NUM_THREADS": "1",
        },
        "scrubbed_exact_keys": sorted(SCRUBBED_COMPUTE_ENV_KEYS),
        "scrubbed_prefixes": ["AY_", "MIMALLOC_", "NY_"],
        "sealed_passthrough": {
            key: environments["off"].get(key) for key in SEALED_COMPUTE_ENV_KEYS
        },
    }


def benchmark_asset(directory: Path, basename: str) -> Path:
    plain = directory / basename
    if plain.is_file():
        return plain
    compressed = Path(f"{plain}.gz")
    if compressed.is_file():
        return compressed
    raise QualificationError(f"missing benchmark input: {plain} (or {compressed.name})")


def validate_canonical_observation(
    manifest: dict[str, object], observed: dict[str, object]
) -> None:
    benchmark = observed.get("benchmark")
    official_results = observed.get("official_results")
    if not isinstance(benchmark, dict) or not isinstance(official_results, dict):
        raise QualificationError("canonical input observation is malformed")
    vnnlibs = benchmark.get("vnnlibs")
    models = benchmark.get("models")
    if not isinstance(vnnlibs, dict) or not isinstance(models, dict):
        raise QualificationError("canonical benchmark observation is malformed")
    checks = {
        "official Git revision": observed.get("official_git_head")
        == manifest["official_git_head"],
        "benchmark Git revision": benchmark.get("git_head")
        == manifest["benchmark_git_head"],
        "target CSV digest": observed.get("targets_csv_sha256")
        == manifest["targets_csv_sha256"],
        "official-results manifest": canonical_json_sha256(official_results)
        == manifest["official_results_manifest_sha256"],
        "benchmark manifest": canonical_json_sha256(benchmark)
        == manifest["benchmark_manifest_sha256"],
        "complete selected-input manifest": canonical_json_sha256(observed)
        == manifest["canonical_selected_inputs_sha256"],
        "selected VNNLIB count": len(vnnlibs) == manifest["selected_vnnlib_count"],
    }
    selected_model = manifest["selected_model"]
    assert isinstance(selected_model, dict)
    model_name = str(selected_model["logical_name"])
    checks["selected model identity"] = models.get(model_name) == {
        "asset": selected_model["asset"],
        "size": selected_model["size"],
        "sha256": selected_model["sha256"],
    }
    checks["selected VNNLIB assets"] = all(
        isinstance(name, str)
        and isinstance(identity, dict)
        and identity.get("asset") == name
        for name, identity in vnnlibs.items()
    )
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise QualificationError(
            "qualification inputs are not the checked-in canonical cohort: "
            + ", ".join(failed)
        )


def capture_canonical_input_set(
    args: argparse.Namespace,
) -> tuple[
    list[Target],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    """Hash and enforce the one official cohort this campaign may qualify."""

    canonical_targets, _official, _sound_tools = derive_targets(
        args.official_root, args.benchmark_root
    )
    official_head = git_value(args.official_root, "rev-parse", "HEAD")
    official_status = git_value(args.official_root, "status", "--porcelain")
    benchmark_head = git_value(args.benchmark_root, "rev-parse", "HEAD")
    if official_head is None or re.fullmatch(r"[0-9a-f]{40}", official_head) is None:
        raise QualificationError("official results root lacks an exact Git revision")
    if official_status is None or official_status:
        raise QualificationError("official results root must be a clean Git checkout")
    if benchmark_head is None or re.fullmatch(r"[0-9a-f]{40}", benchmark_head) is None:
        raise QualificationError("benchmark root lacks an exact Git revision")

    official_files = {
        tool: file_identity(args.official_root / tool / f"2025_{CATEGORY}/results.csv")
        for tool in TOOLS
    }
    instances = file_identity(args.benchmark_root / "instances.csv")
    models = {
        onnx: file_identity(benchmark_asset(args.benchmark_root / "onnx", onnx))
        for onnx in sorted({target.onnx for target in canonical_targets})
    }
    vnnlibs = {
        vnnlib: file_identity(benchmark_asset(args.benchmark_root / "vnnlib", vnnlib))
        for vnnlib in sorted({target.vnnlib for target in canonical_targets})
    }
    observed = {
        "official_git_head": official_head,
        "official_results": {
            tool: compact_file_identity(identity)
            for tool, identity in official_files.items()
        },
        "targets_csv_sha256": sha256_bytes(render_targets(canonical_targets)),
        "benchmark": {
            "git_head": benchmark_head,
            "instances": compact_file_identity(instances),
            "models": {
                name: {
                    "asset": Path(str(identity["path"])).name,
                    **compact_file_identity(identity),
                }
                for name, identity in models.items()
            },
            "vnnlibs": {
                name: {
                    "asset": Path(str(identity["path"])).name,
                    **compact_file_identity(identity),
                }
                for name, identity in vnnlibs.items()
            },
        },
    }
    manifest = load_canonical_inputs_manifest()
    validate_canonical_observation(manifest, observed)
    return canonical_targets, official_files, instances, models, vnnlibs, observed


def capture_immutable_inputs(
    args: argparse.Namespace,
    targets: list[Target],
    target_csv_sha256: str,
    receipt_valid: bool,
) -> dict[str, object]:
    axis = axis_from_args(args)
    (
        canonical_targets,
        official_files,
        instances,
        canonical_models,
        canonical_vnnlibs,
        canonical_observed,
    ) = capture_canonical_input_set(args)
    canonical_by_identity = {
        (target.cohort, target.cohort_index): target for target in canonical_targets
    }
    if any(
        canonical_by_identity.get((target.cohort, target.cohort_index)) != target
        for target in targets
    ):
        raise QualificationError(
            "selected targets are not a subset of the canonical cohort"
        )
    models = {
        name: canonical_models[name]
        for name in sorted({target.onnx for target in targets})
    }
    vnnlibs = {
        name: canonical_vnnlibs[name]
        for name in sorted({target.vnnlib for target in targets})
    }
    return {
        "schema": INPUTS_SCHEMA,
        "ny_source": capture_repo_identity(REPO),
        "runner": file_identity(Path(__file__)),
        "binary": file_identity(args.binary),
        "binary_receipt_valid": receipt_valid,
        "configs": directory_identity(REPO / "configs"),
        "canonical_inputs": {
            "manifest": file_identity(CANONICAL_INPUTS_PATH),
            "observed_sha256": canonical_json_sha256(canonical_observed),
        },
        "compute_environment": compute_environment_contract(axis),
        "launch_tools": {
            "systemd_run": file_identity(Path(args.systemd_run)),
            "systemctl": file_identity(Path(args.systemctl)),
            "timeout": file_identity(Path(args.timeout)),
        },
        "official_results": {
            "root": str(args.official_root.resolve(strict=True)),
            "git_head": canonical_observed["official_git_head"],
            "git_clean": True,
            "files": official_files,
        },
        "benchmark": {
            "root": str(args.benchmark_root.resolve(strict=True)),
            "git_head": canonical_observed["benchmark"]["git_head"],
            "instances": instances,
            "models": models,
            "vnnlibs": vnnlibs,
        },
        "campaign": {
            "category": CATEGORY,
            "promotion_scope": PROMOTION_SCOPE,
            "experiment_axis": axis,
            "default_policy": (
                "NY_ATTACK_POINT_FAST_KERNELS remains default-off"
                if axis == FAST_KERNELS_AXIS
                else "NY_ORT_REFINE_VJP_BATCH=1 measures the shipped default; "
                "arm 0 is its exact kill switch"
            ),
            "official_budget_secs": OFFICIAL_BUDGET_SECS,
            "memory_max": args.memory_max,
            "partial_limit": args.limit,
            "allow_dirty": bool(args.allow_dirty),
            "allow_unreceipted": bool(args.allow_unreceipted),
            "output": str(args.out),
            "arm_order": "alternating off-on/on-off by combined cohort order",
            "inherited_ny_ay_environment": "scrubbed",
            "target_csv_sha256": target_csv_sha256,
            "targets": [target_record(target) for target in targets],
        },
    }


def build_launch_manifest(inputs: dict[str, object]) -> dict[str, object]:
    return {
        "schema": LAUNCH_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_inputs_sha256": canonical_json_sha256(inputs),
        "immutable_inputs": inputs,
    }


def validate_stored_launch_manifest(path: Path) -> dict[str, object]:
    launch = load_json_object(path, LAUNCH_SCHEMA)
    if set(launch) != {
        "schema",
        "created_at_utc",
        "immutable_inputs_sha256",
        "immutable_inputs",
    }:
        raise QualificationError(f"launch manifest has unexpected fields: {path}")
    if not isinstance(launch.get("created_at_utc"), str):
        raise QualificationError(f"launch timestamp is invalid: {path}")
    stored_inputs = launch.get("immutable_inputs")
    if (
        not isinstance(stored_inputs, dict)
        or stored_inputs.get("schema") != INPUTS_SCHEMA
    ):
        raise QualificationError(f"launch immutable inputs are invalid: {path}")
    stored_digest = launch.get("immutable_inputs_sha256")
    require_digest(stored_digest, "launch immutable-input digest")
    if stored_digest != canonical_json_sha256(stored_inputs):
        raise QualificationError(f"launch immutable-input digest is invalid: {path}")
    return launch


def validate_launch_manifest(
    path: Path,
    current_inputs: dict[str, object],
) -> dict[str, object]:
    launch = validate_stored_launch_manifest(path)
    stored_inputs = launch["immutable_inputs"]
    if stored_inputs != current_inputs:
        raise QualificationError(
            "immutable qualification inputs drifted since campaign launch"
        )
    return launch


def artifact_paths(output: Path, target: Target, arm: str) -> tuple[Path, Path, Path]:
    stem = f"{target.cohort}-{target.cohort_index:02d}-{arm}"
    result_path = output / f"{stem}.result"
    return result_path, Path(f"{result_path}.flight.json"), output / f"{stem}.log"


def arm_command(
    target: Target,
    order_index: int,
    arm: str,
    args: argparse.Namespace,
    output: Path,
    launch_inputs_sha256: str,
) -> list[str]:
    result_path, _flight_path, _log_path = artifact_paths(output, target, arm)
    onnx_path = benchmark_asset(args.benchmark_root / "onnx", target.onnx)
    vnnlib_path = benchmark_asset(args.benchmark_root / "vnnlib", target.vnnlib)
    watchdog = OFFICIAL_BUDGET_SECS + 15
    scope_unit = attempt_scope_unit_name(target, order_index, arm, launch_inputs_sha256)
    return [
        args.systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "--unit",
        scope_unit,
        "-p",
        "MemoryAccounting=yes",
        "-p",
        f"MemoryMax={args.memory_max}",
        "-p",
        "MemorySwapMax=0",
        args.timeout,
        "--signal=TERM",
        "--kill-after=5s",
        f"{watchdog}s",
        str(args.binary),
        "-v",
        "vnncomp",
        "v1",
        CATEGORY,
        str(onnx_path),
        str(vnnlib_path),
        str(result_path),
        str(OFFICIAL_BUDGET_SECS),
        "--configs-dir",
        str(REPO / "configs"),
    ]


def attempt_scope_unit_name(
    target: Target, order_index: int, arm: str, launch_inputs_sha256: str
) -> str:
    """Return the scope name bound only to the sealed attempt identity."""

    launch_digest = require_digest(
        launch_inputs_sha256, "attempt launch immutable-input digest"
    )
    if order_index <= 0:
        raise QualificationError("attempt order index must be positive")
    if arm not in {"off", "on"}:
        raise QualificationError(f"invalid attempt arm: {arm!r}")
    identity = {
        "schema": ATTEMPT_SCOPE_IDENTITY_SCHEMA,
        "attempt_number": 1,
        "launch_inputs_sha256": launch_digest,
        "target": target_record(target),
        "order_index": order_index,
        "arm": arm,
    }
    return f"ny-cifar-fast-{canonical_json_sha256(identity)}.scope"


def query_systemd_scope(systemctl: str, scope_unit: str) -> dict[str, str] | None:
    """Read bounded post-exit scope diagnostics; never supply verdict evidence."""

    command = [systemctl, "--user", "show", scope_unit, "--no-pager"]
    for field in SYSTEMD_SCOPE_PROPERTIES:
        command.extend(("-p", field))
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=SYSTEMD_SCOPE_QUERY_TIMEOUT_SECS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if (
            separator != "="
            or key not in SYSTEMD_SCOPE_PROPERTIES
            or key in values
            or len(value) > 128
        ):
            return None
        values[key] = value
    if not set(SYSTEMD_SCOPE_REQUIRED_PROPERTIES).issubset(values):
        return None
    # `systemctl show missing.scope` can exit zero and synthesize sentinel
    # defaults (notably OOMKills=UINT64_MAX).  Only a real retained transient
    # unit is admissible diagnostic evidence.
    if values["LoadState"] != "loaded":
        return None
    return values


def reset_failed_systemd_scope(systemctl: str, scope_unit: str) -> None:
    """Best-effort bounded cleanup for a deterministic transient scope name."""

    try:
        subprocess.run(
            [systemctl, "--user", "reset-failed", scope_unit],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=SYSTEMD_SCOPE_RESET_TIMEOUT_SECS,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def scope_diagnostics_prove_cgroup_oom(
    diagnostics: dict[str, str] | None,
) -> bool:
    if diagnostics is None or diagnostics.get("LoadState") != "loaded":
        return False
    if diagnostics.get("Result") == "oom-kill":
        return True
    raw_oom_kills = diagnostics.get("OOMKills")
    if (
        raw_oom_kills is None
        or not raw_oom_kills.isascii()
        or not raw_oom_kills.isdigit()
    ):
        return False
    oom_kills = int(raw_oom_kills)
    if raw_oom_kills != str(oom_kills):
        return False
    # systemd uses UINT64_MAX as an unset/not-found sentinel on some releases.
    return 0 < oom_kills < SYSTEMD_U64_UNSET


def format_scope_failure(diagnostics: dict[str, str] | None) -> str:
    if diagnostics is None:
        return "cause=child-failure scope_diagnostics=unavailable"
    unavailable = "<unavailable>"
    raw_oom_kills = diagnostics.get("OOMKills", unavailable)
    cause = (
        "cgroup-oom"
        if scope_diagnostics_prove_cgroup_oom(diagnostics)
        else "child-failure"
    )
    return (
        f"cause={cause} scope_result={diagnostics.get('Result', unavailable)!r} "
        f"oom_kills={raw_oom_kills!r} "
        f"memory_peak={diagnostics.get('MemoryPeak', unavailable)!r} "
        f"memory_max={diagnostics.get('MemoryMax', unavailable)!r} "
        f"memory_swap_peak={diagnostics.get('MemorySwapPeak', unavailable)!r} "
        f"memory_swap_max={diagnostics.get('MemorySwapMax', unavailable)!r}"
    )


def attempt_paths(output: Path, target: Target, arm: str) -> tuple[Path, Path]:
    stem = f"{target.cohort}-{target.cohort_index:02d}-{arm}"
    return output / f"{stem}.attempt.json", output / f"{stem}.row.json"


def expected_run_sequence(
    targets: list[Target],
) -> list[tuple[Target, int, str]]:
    return [
        (target, order_index, arm)
        for order_index, target in enumerate(targets, 1)
        for arm in arm_order(order_index)
    ]


def args_from_launch(launch: dict[str, object]) -> argparse.Namespace:
    inputs = launch.get("immutable_inputs")
    if not isinstance(inputs, dict):
        raise QualificationError("launch immutable inputs are invalid")
    binary = inputs.get("binary")
    benchmark = inputs.get("benchmark")
    tools = inputs.get("launch_tools")
    campaign = inputs.get("campaign")
    if not all(
        isinstance(value, dict) for value in (binary, benchmark, tools, campaign)
    ):
        raise QualificationError("launch command inputs are invalid")
    assert isinstance(binary, dict)
    assert isinstance(benchmark, dict)
    assert isinstance(tools, dict)
    assert isinstance(campaign, dict)
    systemd_run = tools.get("systemd_run")
    systemctl = tools.get("systemctl")
    timeout = tools.get("timeout")
    if (
        not isinstance(systemd_run, dict)
        or not isinstance(systemctl, dict)
        or not isinstance(timeout, dict)
    ):
        raise QualificationError("launch-tool identities are invalid")
    values = {
        "binary": binary.get("path"),
        "benchmark_root": benchmark.get("root"),
        "out": campaign.get("output"),
        "memory_max": campaign.get("memory_max"),
        "systemd_run": systemd_run.get("path"),
        "systemctl": systemctl.get("path"),
        "timeout": timeout.get("path"),
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        raise QualificationError("launch command paths or memory policy are invalid")
    return argparse.Namespace(
        binary=Path(str(values["binary"])),
        benchmark_root=Path(str(values["benchmark_root"])),
        out=Path(str(values["out"])),
        memory_max=str(values["memory_max"]),
        axis=axis_from_launch(launch),
        systemd_run=str(values["systemd_run"]),
        systemctl=str(values["systemctl"]),
        timeout=str(values["timeout"]),
    )


def preseal_attempt(
    target: Target,
    order_index: int,
    arm: str,
    args: argparse.Namespace,
    output: Path,
    launch: dict[str, object],
) -> None:
    axis = axis_from_args(args)
    attempt_path, row_path = attempt_paths(output, target, arm)
    result_path, flight_path, log_path = artifact_paths(output, target, arm)
    occupied = [
        path
        for path in (attempt_path, row_path, result_path, flight_path, log_path)
        if path.exists() or path.is_symlink()
    ]
    if occupied:
        raise QualificationError(
            "refusing to overwrite prior arm evidence: "
            + ", ".join(str(path) for path in occupied)
        )
    inputs = launch.get("immutable_inputs")
    if not isinstance(inputs, dict) or not isinstance(
        inputs.get("compute_environment"), dict
    ):
        raise QualificationError("launch compute-environment contract is invalid")
    launch_inputs_sha256 = require_digest(
        launch.get("immutable_inputs_sha256"), "launch immutable-input digest"
    )
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "attempt_number": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": target_record(target),
        "order_index": order_index,
        "arm": arm,
        "launch_inputs_sha256": launch_inputs_sha256,
        "compute_environment_sha256": canonical_json_sha256(
            inputs["compute_environment"]
        ),
        "ambient_env": forced_arm_environment(axis, arm),
        "command": arm_command(
            target,
            order_index,
            arm,
            args,
            output,
            launch_inputs_sha256,
        ),
        "artifacts": {
            "result": result_path.name,
            "flight": flight_path.name,
            "log": log_path.name,
            "row": row_path.name,
        },
    }
    atomic_write_json(attempt_path, attempt, require_absent=True)


def validate_attempt(
    path: Path,
    target: Target,
    order_index: int,
    arm: str,
    launch: dict[str, object],
    args: argparse.Namespace,
    output: Path,
) -> dict[str, object]:
    axis = axis_from_args(args)
    attempt = load_json_object(path, ATTEMPT_SCHEMA)
    expected_fields = {
        "schema",
        "attempt_number",
        "started_at_utc",
        "target",
        "order_index",
        "arm",
        "launch_inputs_sha256",
        "compute_environment_sha256",
        "ambient_env",
        "command",
        "artifacts",
    }
    if set(attempt) != expected_fields:
        raise QualificationError(f"attempt record has unexpected fields: {path}")
    inputs = launch.get("immutable_inputs")
    if not isinstance(inputs, dict) or not isinstance(
        inputs.get("compute_environment"), dict
    ):
        raise QualificationError("launch compute-environment contract is invalid")
    launch_inputs_sha256 = require_digest(
        launch.get("immutable_inputs_sha256"), "launch immutable-input digest"
    )
    result_path, flight_path, log_path = artifact_paths(output, target, arm)
    _attempt_path, row_path = attempt_paths(output, target, arm)
    expected = {
        "attempt_number": 1,
        "target": target_record(target),
        "order_index": order_index,
        "arm": arm,
        "launch_inputs_sha256": launch_inputs_sha256,
        "compute_environment_sha256": canonical_json_sha256(
            inputs["compute_environment"]
        ),
        "ambient_env": forced_arm_environment(axis, arm),
        "command": arm_command(
            target,
            order_index,
            arm,
            args,
            output,
            launch_inputs_sha256,
        ),
        "artifacts": {
            "result": result_path.name,
            "flight": flight_path.name,
            "log": log_path.name,
            "row": row_path.name,
        },
    }
    for field, value in expected.items():
        if attempt.get(field) != value:
            raise QualificationError(f"attempt record has wrong {field}: {path}")
    if (
        not isinstance(attempt.get("started_at_utc"), str)
        or not attempt["started_at_utc"]
    ):
        raise QualificationError(f"attempt timestamp is invalid: {path}")
    return attempt


def commit_row_fragment(
    output: Path,
    target: Target,
    order_index: int,
    arm: str,
    row: dict[str, str],
    axis: str = DEFAULT_AXIS,
) -> None:
    attempt_path, row_path = attempt_paths(output, target, arm)
    validate_row_artifacts(row, target, order_index, arm, output, axis)
    atomic_write_json(
        row_path,
        {
            "schema": ROW_FRAGMENT_SCHEMA,
            "attempt_sha256": sha256_file(attempt_path),
            "row": row,
        },
        require_absent=True,
    )


def load_attempt_state(
    output: Path,
    targets: list[Target],
    launch: dict[str, object],
) -> tuple[list[dict[str, str]], list[str]]:
    args = args_from_launch(launch)
    axis = axis_from_args(args)
    sequence = expected_run_sequence(targets)
    expected_attempts = {
        attempt_paths(output, target, arm)[0] for target, _order_index, arm in sequence
    }
    expected_fragments = {
        attempt_paths(output, target, arm)[1] for target, _order_index, arm in sequence
    }
    observed_attempts = set(output.glob("*.attempt.json"))
    observed_fragments = set(output.glob("*.row.json"))
    unexpected = (observed_attempts - expected_attempts) | (
        observed_fragments - expected_fragments
    )
    if unexpected:
        raise QualificationError(
            "unexpected attempt evidence: "
            + ", ".join(str(path) for path in sorted(unexpected))
        )

    rows: list[dict[str, str]] = []
    incomplete: list[str] = []
    for target, order_index, arm in sequence:
        attempt_path, row_path = attempt_paths(output, target, arm)
        has_attempt = attempt_path.exists() or attempt_path.is_symlink()
        has_row = row_path.exists() or row_path.is_symlink()
        label = f"{target.cohort}:{target.cohort_index}:{arm}"
        if has_row and not has_attempt:
            raise QualificationError(f"row fragment has no presealed attempt: {label}")
        if not has_attempt:
            continue
        validate_attempt(attempt_path, target, order_index, arm, launch, args, output)
        if not has_row:
            incomplete.append(label)
            continue
        fragment = load_json_object(row_path, ROW_FRAGMENT_SCHEMA)
        if set(fragment) != {"schema", "attempt_sha256", "row"}:
            raise QualificationError(f"row fragment has unexpected fields: {row_path}")
        if fragment.get("attempt_sha256") != sha256_file(attempt_path):
            raise QualificationError(
                f"row fragment is not bound to its attempt: {row_path}"
            )
        row = fragment.get("row")
        if not isinstance(row, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in row.items()
        ):
            raise QualificationError(f"row fragment payload is invalid: {row_path}")
        typed_row = dict(row)
        validate_row_artifacts(typed_row, target, order_index, arm, output, axis)
        rows.append(typed_row)
    return rows, incomplete


def run_one(
    target: Target,
    order_index: int,
    arm: str,
    args: argparse.Namespace,
    output: Path,
    launch_inputs_sha256: str,
) -> dict[str, str]:
    axis = axis_from_args(args)
    result_path, flight_path, log_path = artifact_paths(output, target, arm)
    watchdog = OFFICIAL_BUDGET_SECS + 15
    command = arm_command(
        target,
        order_index,
        arm,
        args,
        output,
        launch_inputs_sha256,
    )
    scope_unit = attempt_scope_unit_name(target, order_index, arm, launch_inputs_sha256)
    if args.dry_run:
        prefix = f"{target.cohort}:{target.cohort_index:02d} {arm}: "
        if axis == WRAPPER_VJP_AXIS:
            forced = " ".join(
                f"{key}={value}"
                for key, value in sorted(forced_arm_environment(axis, arm).items())
            )
            print(prefix + f"env {forced} " + " ".join(command))
        else:
            print(prefix + " ".join(command))
        return {}

    # Administrative cleanup is outside the measured arm interval. A failed
    # transient unit can otherwise collide with this deterministic identity.
    reset_failed_systemd_scope(args.systemctl, scope_unit)
    started = time.monotonic()
    try:
        log_descriptor = os.open(
            log_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        raise QualificationError(
            f"cannot create arm log {log_path}: {error}"
        ) from error
    with os.fdopen(log_descriptor, "wb") as log:
        try:
            completed = subprocess.run(
                command,
                check=False,
                env=scrubbed_environment(arm, axis),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=watchdog + 20,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
        log.flush()
        os.fsync(log.fileno())
    # Make the child-produced result and flight sidecar durable before they are
    # authenticated and bound into an atomic row fragment. A power loss cannot
    # preserve a committed fragment while losing one of its artifacts.
    for artifact in (result_path, flight_path):
        if not artifact.is_symlink() and artifact.is_file():
            fsync_regular_file(artifact)
    fsync_directory(output)
    wall = time.monotonic() - started
    scope_diagnostics = query_systemd_scope(args.systemctl, scope_unit)
    reset_failed_systemd_scope(args.systemctl, scope_unit)
    result_ready = not result_path.is_symlink() and result_path.is_file()
    flight_ready = not flight_path.is_symlink() and flight_path.is_file()
    cgroup_oom = scope_diagnostics_prove_cgroup_oom(scope_diagnostics)
    if not result_ready or not flight_ready or cgroup_oom:
        missing = [
            label
            for label, ready in (("result", result_ready), ("flight", flight_ready))
            if not ready
        ]
        missing_summary = ",".join(missing) if missing else "none"
        raise QualificationError(
            f"{target.cohort}:{target.cohort_index:02d}:{arm} child failed before "
            "authenticated flight evidence: "
            f"exit_code={exit_code} missing_or_invalid_artifacts={missing_summary} "
            f"scope_unit={scope_unit!r} {format_scope_failure(scope_diagnostics)}; "
            "the presealed attempt is permanently non-promotable and will not be retried"
        )
    if result_path.is_file():
        first_line = result_path.read_text(errors="replace").splitlines()
        status = first_line[0].strip() if first_line else "error"
    else:
        status = "error"
    if status not in VALID_STATUSES:
        status = "error"
    log_text = log_path.read_text(errors="replace")
    decline_reasons = wrapper_vjp_decline_reasons(log_text)
    if axis == WRAPPER_VJP_AXIS and decline_reasons:
        reasons = ",".join(dict.fromkeys(decline_reasons))
        raise QualificationError(
            f"{target.cohort}:{target.cohort_index:02d}:{arm} wrapper-VJP "
            "qualification is capability-inconclusive and permanently "
            "non-promotable: the exact-VJP pre-wave declined before its armed "
            f"marker (reason={reasons}); no on-arm efficacy comparison is "
            "possible on this host"
        )
    flight = flight_evidence(flight_path, arm, status, axis)
    log_authenticated = axis_log_authenticated(log_text, axis, arm)
    steps = parse_steps(log_text, axis)
    cutoff = OFFICIAL_BUDGET_SECS + OFFICIAL_CLOCK_TOLERANCE_SECS
    publish_secs = flight["publish_secs"]
    terminal_secs = flight["terminal_secs"]
    peak_load_per_core = flight["peak_load_per_core"]
    within_official_cutoff = bool(flight["within_official_cutoff"]) and wall <= cutoff
    row = {
        "cohort": target.cohort,
        "cohort_index": str(target.cohort_index),
        "order_index": str(order_index),
        "arm": arm,
        "onnx": target.onnx,
        "vnnlib": target.vnnlib,
        "ground_truth": target.ground_truth,
        "status": status,
        "wall_secs": f"{wall:.6f}",
        "exit_code": str(exit_code),
        "attack_steps": "" if steps is None else str(steps),
        "arm_authenticated": str(
            bool(flight["authenticated"]) and log_authenticated
        ).lower(),
        "trusted_upfront_sat": str(bool(flight["trusted_sat"])).lower(),
        "backend_kind": str(flight["backend"]),
        "regime_sha256": str(flight["regime_sha256"]),
        "flight_publish_secs": (
            "" if publish_secs is None else f"{float(publish_secs):.9f}"
        ),
        "flight_terminal_secs": (
            "" if terminal_secs is None else f"{float(terminal_secs):.9f}"
        ),
        "within_official_cutoff": str(within_official_cutoff).lower(),
        "peak_load_per_core": (
            "" if peak_load_per_core is None else f"{float(peak_load_per_core):.9f}"
        ),
        "load_acceptable": str(bool(flight["load_acceptable"])).lower(),
        "result_sha256": sha256_file(result_path) if result_path.is_file() else "",
        "flight_sha256": sha256_file(flight_path) if flight_path.is_file() else "",
        "log_sha256": sha256_file(log_path),
    }
    print(
        f"{target.cohort}:{target.cohort_index:02d} {arm}: "
        f"{status} in {wall:.1f}s (steps={row['attack_steps'] or '-'})",
        flush=True,
    )
    return row


def write_targets(path: Path, targets: Iterable[Target]) -> None:
    atomic_write_bytes(path, render_targets(targets), require_absent=True)


def render_results_snapshot(rows: Iterable[dict[str, str]]) -> bytes:
    sink = io.StringIO(newline="")
    writer = csv.DictWriter(
        sink, fieldnames=RESULT_FIELDS, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(RESULT_FIELDS):
            raise QualificationError(
                "result row fields do not match the qualification schema"
            )
        writer.writerow(row)
    return sink.getvalue().encode("utf-8")


def initialize_results(path: Path) -> None:
    atomic_write_bytes(path, render_results_snapshot([]), require_absent=True)


def publish_results_snapshot(path: Path, rows: list[dict[str, str]]) -> None:
    """Atomically rebuild the derived TSV from durable per-arm row fragments."""

    if path.is_symlink():
        raise QualificationError(f"results evidence must not be a symlink: {path}")
    atomic_write_bytes(path, render_results_snapshot(rows))


def synchronize_results_snapshot(
    path: Path, rows: list[dict[str, str]], *, allow_repair: bool
) -> None:
    expected = render_results_snapshot(rows)
    try:
        observed = path.read_bytes()
    except OSError as error:
        if not allow_repair:
            raise QualificationError(
                f"cannot read result evidence: {path}: {error}"
            ) from error
        observed = b""
    if observed == expected:
        return
    if not allow_repair:
        raise QualificationError("results snapshot differs from atomic row fragments")
    publish_results_snapshot(path, rows)


def expected_rows(
    targets: list[Target],
) -> dict[tuple[str, int, str], tuple[Target, int, str]]:
    expected: dict[tuple[str, int, str], tuple[Target, int, str]] = {}
    for order_index, target in enumerate(targets, 1):
        for arm in ("off", "on"):
            key = (target.cohort, target.cohort_index, arm)
            if key in expected:
                raise QualificationError(f"duplicate expected campaign row: {key}")
            expected[key] = (target, order_index, arm)
    return expected


def canonical_integer(raw: str, label: str, *, minimum: int | None = None) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise QualificationError(f"{label} is not an integer: {raw!r}") from error
    if str(value) != raw or (minimum is not None and value < minimum):
        raise QualificationError(f"{label} is not canonical: {raw!r}")
    return value


def validate_row_identity(
    row: dict[str, str], target: Target, order_index: int, arm: str
) -> None:
    if set(row) != set(RESULT_FIELDS) or any(
        not isinstance(value, str) for value in row.values()
    ):
        raise QualificationError(
            "result row does not have the exact string field schema"
        )
    exact = {
        "cohort": target.cohort,
        "cohort_index": str(target.cohort_index),
        "order_index": str(order_index),
        "arm": arm,
        "onnx": target.onnx,
        "vnnlib": target.vnnlib,
        "ground_truth": target.ground_truth,
    }
    for field, expected in exact.items():
        if row[field] != expected:
            raise QualificationError(
                f"result row {target.cohort}:{target.cohort_index}:{arm} "
                f"has wrong {field}: {row[field]!r} != {expected!r}"
            )
    if row["status"] not in VALID_STATUSES:
        raise QualificationError(f"invalid result status: {row['status']!r}")
    if re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{6}", row["wall_secs"]) is None:
        raise QualificationError(f"non-canonical wall time: {row['wall_secs']!r}")
    wall = float(row["wall_secs"])
    if not math.isfinite(wall) or wall < 0:
        raise QualificationError(f"invalid wall time: {row['wall_secs']!r}")
    canonical_integer(row["exit_code"], "exit_code")
    if row["attack_steps"]:
        canonical_integer(row["attack_steps"], "attack_steps", minimum=0)
    for field in (
        "arm_authenticated",
        "trusted_upfront_sat",
        "within_official_cutoff",
        "load_acceptable",
    ):
        if row[field] not in {"true", "false"}:
            raise QualificationError(f"{field} is not an exact boolean")
    if row["backend_kind"] not in VALID_BACKENDS:
        raise QualificationError(f"backend_kind is invalid: {row['backend_kind']!r}")
    require_digest(row["regime_sha256"], "regime_sha256")
    timing_values: dict[str, float] = {}
    for field in (
        "flight_publish_secs",
        "flight_terminal_secs",
        "peak_load_per_core",
    ):
        if re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{9}", row[field]) is None:
            raise QualificationError(f"{field} is not canonical: {row[field]!r}")
        value = float(row[field])
        if not math.isfinite(value) or value < 0.0:
            raise QualificationError(f"{field} is invalid: {row[field]!r}")
        timing_values[field] = value
    cutoff = OFFICIAL_BUDGET_SECS + OFFICIAL_CLOCK_TOLERANCE_SECS
    expected_cutoff = (
        wall <= cutoff
        and timing_values["flight_publish_secs"] <= cutoff
        and timing_values["flight_terminal_secs"] <= cutoff
    )
    if row["within_official_cutoff"] != str(expected_cutoff).lower():
        raise QualificationError("official-cutoff flag disagrees with sealed clocks")
    expected_load = timing_values["peak_load_per_core"] <= MAX_LOAD_PER_LOGICAL_CORE
    if row["load_acceptable"] != str(expected_load).lower():
        raise QualificationError("load-acceptance flag disagrees with sealed load")
    for field in ("result_sha256", "flight_sha256", "log_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", row[field]) is None:
            raise QualificationError(f"{field} is not an exact SHA-256 digest")


def validate_row_artifacts(
    row: dict[str, str],
    target: Target,
    order_index: int,
    arm: str,
    output: Path,
    axis: str = DEFAULT_AXIS,
) -> None:
    validate_row_identity(row, target, order_index, arm)
    result_path, flight_path, log_path = artifact_paths(output, target, arm)
    for label, path in (
        ("result", result_path),
        ("flight", flight_path),
        ("log", log_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise QualificationError(f"persisted {label} artifact is missing: {path}")
    observed_hashes = {
        "result_sha256": sha256_file(result_path),
        "flight_sha256": sha256_file(flight_path),
        "log_sha256": sha256_file(log_path),
    }
    for field, observed in observed_hashes.items():
        if row[field] != observed:
            raise QualificationError(
                f"persisted artifact hash mismatch for {target.cohort}:"
                f"{target.cohort_index}:{arm} ({field})"
            )
    lines = result_path.read_text(encoding="utf-8", errors="replace").splitlines()
    result_status = lines[0].strip() if lines else "error"
    if result_status != row["status"]:
        raise QualificationError("persisted result verdict does not match its TSV row")
    if row["status"] == "sat" and not any(line.strip() for line in lines[1:]):
        raise QualificationError("persisted SAT result has no counterexample witness")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    observed_steps = parse_steps(log_text, axis)
    if row["attack_steps"] != ("" if observed_steps is None else str(observed_steps)):
        raise QualificationError("persisted attack-step count does not match its log")
    flight = flight_evidence(flight_path, arm, row["status"], axis)
    expected_authenticated = bool(flight["authenticated"]) and axis_log_authenticated(
        log_text, axis, arm
    )
    if not expected_authenticated or row["arm_authenticated"] != "true":
        raise QualificationError(
            "persisted arm lacks authenticated flight/log evidence"
        )
    if row["trusted_upfront_sat"] != str(bool(flight["trusted_sat"])).lower():
        raise QualificationError(
            "persisted trusted-SAT flag does not match flight evidence"
        )
    if row["backend_kind"] != flight["backend"]:
        raise QualificationError("persisted backend does not match flight evidence")
    if row["regime_sha256"] != flight["regime_sha256"]:
        raise QualificationError("persisted regime does not match flight evidence")
    expected_flight_fields = {
        "flight_publish_secs": f"{float(flight['publish_secs']):.9f}",
        "flight_terminal_secs": f"{float(flight['terminal_secs']):.9f}",
        "within_official_cutoff": str(
            bool(flight["within_official_cutoff"])
            and float(row["wall_secs"])
            <= OFFICIAL_BUDGET_SECS + OFFICIAL_CLOCK_TOLERANCE_SECS
        ).lower(),
        "peak_load_per_core": f"{float(flight['peak_load_per_core']):.9f}",
        "load_acceptable": str(bool(flight["load_acceptable"])).lower(),
    }
    for field, expected in expected_flight_fields.items():
        if row[field] != expected:
            raise QualificationError(
                f"persisted {field} does not match flight evidence"
            )


def load_persisted_rows(
    path: Path,
    targets: list[Target],
    output: Path,
    axis: str = DEFAULT_AXIS,
) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"missing regular result evidence: {path}")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise QualificationError(
            f"torn result evidence (missing final newline): {path}"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
        reader = csv.DictReader(
            io.StringIO(text, newline=""), delimiter="\t", strict=True
        )
        if reader.fieldnames != list(RESULT_FIELDS):
            raise QualificationError(f"result header drift in {path}")
        allowed = expected_rows(targets)
        seen: set[tuple[str, int, str]] = set()
        rows: list[dict[str, str]] = []
        for number, raw_row in enumerate(reader, 2):
            if None in raw_row or any(value is None for value in raw_row.values()):
                raise QualificationError(f"malformed result row {number} in {path}")
            row = dict(raw_row)
            cohort_index = canonical_integer(
                row["cohort_index"], f"cohort_index at row {number}", minimum=1
            )
            key = (row["cohort"], cohort_index, row["arm"])
            if key not in allowed:
                raise QualificationError(f"unexpected campaign row {key} in {path}")
            if key in seen:
                raise QualificationError(f"duplicate campaign row {key} in {path}")
            target, order_index, arm = allowed[key]
            validate_row_artifacts(row, target, order_index, arm, output, axis)
            seen.add(key)
            rows.append(row)
    except (UnicodeDecodeError, csv.Error) as error:
        raise QualificationError(
            f"malformed result evidence {path}: {error}"
        ) from error
    return rows


def summarize(rows: list[dict[str, str]], targets: list[Target]) -> dict[str, object]:
    allowed = expected_rows(targets)
    by_key: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        cohort_index = canonical_integer(row["cohort_index"], "cohort_index", minimum=1)
        key = (row["cohort"], cohort_index, row["arm"])
        if key not in allowed:
            raise QualificationError(f"unexpected campaign row in summary: {key}")
        if key in by_key:
            raise QualificationError(f"duplicate campaign row in summary: {key}")
        target, order_index, arm = allowed[key]
        validate_row_identity(row, target, order_index, arm)
        by_key[key] = row
    gain_pairs = [
        (by_key.get(("gain", index, "off")), by_key.get(("gain", index, "on")))
        for index in range(1, EXPECTED_GAIN_ROWS + 1)
    ]
    guard_pairs = [
        (by_key.get(("guard", index, "off")), by_key.get(("guard", index, "on")))
        for index in range(1, EXPECTED_GUARD_ROWS + 1)
    ]
    differential_gains = sum(
        off is not None
        and on is not None
        and off["status"] != "sat"
        and on["status"] == "sat"
        for off, on in gain_pairs
    )
    evidenced_gains = sum(
        off is not None
        and on is not None
        and off["status"] != "sat"
        and on["status"] == "sat"
        and on["attack_steps"] != ""
        and int(on["attack_steps"]) > 0
        for off, on in gain_pairs
    )
    sat_losses = sum(
        off is not None
        and on is not None
        and off["status"] == "sat"
        and on["status"] != "sat"
        for off, on in gain_pairs
    )
    proof_losses = sum(
        off is not None
        and on is not None
        and off["status"] == "unsat"
        and on["status"] != "unsat"
        for off, on in guard_pairs
    )
    guards_closed = sum(
        off is not None
        and on is not None
        and off["status"] == "unsat"
        and on["status"] == "unsat"
        for off, on in guard_pairs
    )
    contradictions = [
        row
        for row in rows
        if (row["ground_truth"] == "sat" and row["status"] == "unsat")
        or (row["ground_truth"] == "unsat" and row["status"] == "sat")
    ]
    unauthenticated = [row for row in rows if row["arm_authenticated"] != "true"]
    nonzero_exits = [row for row in rows if row["exit_code"] != "0"]
    untrusted_gain_sats = [
        on
        for off, on in gain_pairs
        if off is not None
        and on is not None
        and off["status"] != "sat"
        and on["status"] == "sat"
        and on["trusted_upfront_sat"] != "true"
    ]
    backend_mismatches = sum(
        off is not None and on is not None and off["backend_kind"] != on["backend_kind"]
        for off, on in gain_pairs + guard_pairs
    )
    regime_mismatches = sum(
        off is not None
        and on is not None
        and off["regime_sha256"] != on["regime_sha256"]
        for off, on in gain_pairs + guard_pairs
    )
    campaign_regimes = {row["regime_sha256"] for row in rows}
    late_rows = [row for row in rows if row["within_official_cutoff"] != "true"]
    overloaded_rows = [row for row in rows if row["load_acceptable"] != "true"]
    gains_without_fast_evidence = differential_gains - evidenced_gains
    expected_keys = set(allowed)
    observed_keys = set(by_key)
    complete = len(rows) == len(expected_keys) and observed_keys == expected_keys
    full_narrow_scope = (
        len(targets) == EXPECTED_GAIN_ROWS + EXPECTED_GUARD_ROWS
        and {target.cohort_index for target in targets if target.cohort == "gain"}
        == set(range(1, EXPECTED_GAIN_ROWS + 1))
        and {target.cohort_index for target in targets if target.cohort == "guard"}
        == set(range(1, EXPECTED_GUARD_ROWS + 1))
        and all("resnet_large" in target.onnx for target in targets)
    )
    promotion = (
        complete
        and full_narrow_scope
        and evidenced_gains >= MINIMUM_PROMOTION_GAINS
        and sat_losses == 0
        and proof_losses == 0
        and guards_closed == EXPECTED_GUARD_ROWS
        and not contradictions
        and not unauthenticated
        and not nonzero_exits
        and not untrusted_gain_sats
        and backend_mismatches == 0
        and regime_mismatches == 0
        and len(campaign_regimes) == 1
        and not late_rows
        and not overloaded_rows
        and gains_without_fast_evidence == 0
        and all(row["status"] != "error" for row in rows)
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "promotion_scope": PROMOTION_SCOPE,
        "global_or_default_on_authorized": False,
        "runs_observed": len(rows),
        "runs_expected": len(expected_keys),
        "complete": complete,
        "exact_expected_keys": observed_keys == expected_keys,
        "full_narrow_scope": full_narrow_scope,
        "differential_sat_gains": differential_gains,
        "sat_gains": evidenced_gains,
        "gains_without_positive_fast_kernel_evidence": gains_without_fast_evidence,
        "sat_losses": sat_losses,
        "proof_guards_closed_both_arms": guards_closed,
        "proof_losses": proof_losses,
        "field_contradictions": len(contradictions),
        "unauthenticated_arm_rows": len(unauthenticated),
        "nonzero_exit_rows": len(nonzero_exits),
        "backend_pair_mismatches": backend_mismatches,
        "regime_pair_mismatches": regime_mismatches,
        "campaign_regimes": len(campaign_regimes),
        "late_rows": len(late_rows),
        "overloaded_rows": len(overloaded_rows),
        "gained_sat_rows_without_trusted_upfront_event": len(untrusted_gain_sats),
        "promotion_pass": promotion,
    }


def summarize_with_launch_policy(
    rows: list[dict[str, str]],
    targets: list[Target],
    launch: dict[str, object],
) -> dict[str, object]:
    inputs = launch.get("immutable_inputs")
    if not isinstance(inputs, dict) or inputs.get("schema") != INPUTS_SCHEMA:
        raise QualificationError("launch immutable-input schema is invalid")
    campaign = inputs.get("campaign")
    ny_source = inputs.get("ny_source")
    if not isinstance(campaign, dict) or not isinstance(ny_source, dict):
        raise QualificationError("launch campaign/source policy is invalid")
    allow_dirty = campaign.get("allow_dirty")
    allow_unreceipted = campaign.get("allow_unreceipted")
    partial_limit = campaign.get("partial_limit")
    receipt_valid = inputs.get("binary_receipt_valid")
    source_clean = ny_source.get("clean")
    if type(allow_dirty) is not bool or type(allow_unreceipted) is not bool:
        raise QualificationError("launch diagnostic policy booleans are invalid")
    if partial_limit is not None and (
        type(partial_limit) is not int or partial_limit <= 0
    ):
        raise QualificationError("launch partial-limit policy is invalid")
    if type(receipt_valid) is not bool or type(source_clean) is not bool:
        raise QualificationError("launch receipt/source dispositions are invalid")
    if campaign.get("promotion_scope") != PROMOTION_SCOPE:
        raise QualificationError("launch campaign has the wrong promotion scope")
    if campaign.get("official_budget_secs") != OFFICIAL_BUDGET_SECS or isinstance(
        campaign.get("official_budget_secs"), bool
    ):
        raise QualificationError("launch campaign has the wrong official budget")
    axis = axis_from_launch(launch)

    summary = summarize(rows, targets)
    summary.update(
        {
            "experiment_axis": axis,
            "clean_source_required": not allow_dirty,
            "receipt_required": not allow_unreceipted,
            "source_clean": source_clean,
            "binary_receipt_valid": receipt_valid,
            "partial_limit": partial_limit,
            "one_attempt_per_arm": True,
            "evidence_actionable_only_with_completion_seal": True,
        }
    )
    if (
        allow_dirty
        or allow_unreceipted
        or partial_limit is not None
        or not source_clean
        or not receipt_valid
    ):
        summary["promotion_pass"] = False
    return summary


def find_launch_tool(name: str) -> str | None:
    """Return the canonical executable path sealed into campaign evidence."""

    discovered = shutil.which(name)
    if discovered is None:
        return None
    try:
        return str(Path(discovered).resolve(strict=True))
    except OSError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--binary", type=Path, default=REPO / "target/release/ny")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--memory-max", default="4G")
    parser.add_argument(
        "--axis",
        choices=EXPERIMENT_AXES,
        default=DEFAULT_AXIS,
        help=(
            "A/B variable: historical attack-point fast kernels (default), or "
            "the wrapper exact-VJP K=64 pre-wave and its exact kill switch"
        ),
    )
    parser.add_argument("--limit", type=int, help="run at most N rows per cohort")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue an authenticated --out directory only when it has no "
            "unfinished presealed attempt"
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="diagnostic only; the resulting partial evidence is not promotable",
    )
    parser.add_argument(
        "--allow-unreceipted",
        action="store_true",
        help="diagnostic only; skip submission binary receipt validation",
    )
    args = parser.parse_args()
    args.official_root = args.official_root.resolve()
    args.benchmark_root = args.benchmark_root.resolve()
    args.binary = args.binary.resolve()
    # Use the same canonical paths for execution, the immutable launch
    # manifest, and attempt validation.  `file_identity()` resolves symlinks;
    # retaining a `which` result here would make a symlinked tool (for example
    # coreutils' timeout on some hosts) validate differently after the run.
    args.systemd_run = find_launch_tool("systemd-run")
    args.systemctl = find_launch_tool("systemctl")
    args.timeout = find_launch_tool("timeout")
    if not args.systemd_run or not args.systemctl or not args.timeout:
        parser.error(
            "systemd-run, systemctl, and timeout are required for OOM-safe qualification"
        )
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.resume and args.dry_run:
        parser.error("--resume and --dry-run are mutually exclusive")
    if args.resume and args.out is None:
        parser.error("--resume requires an explicit --out directory")
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        campaign_name = (
            "cifar-fast-kernels"
            if args.axis == FAST_KERNELS_AXIS
            else "cifar-wrapper-vjp"
        )
        args.out = REPO / f"reports/measured-runs/{campaign_name}-{stamp}"
    else:
        args.out = args.out.resolve()
    return args


def repo_is_dirty() -> bool:
    status = git_value(REPO, "status", "--porcelain")
    if status is None:
        raise QualificationError("cannot inspect NY worktree cleanliness")
    return bool(status)


def validate_binary_receipt(args: argparse.Namespace) -> bool:
    if args.allow_unreceipted:
        return False
    receipt = subprocess.run(
        [
            "bash",
            str(REPO / "vnncomp_scripts/submission_binary_receipt.sh"),
            "validate",
            str(args.binary),
            str(REPO),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if receipt.returncode != 0:
        detail = receipt.stderr.strip() or receipt.stdout.strip()
        raise QualificationError(
            "submission binary receipt validation failed"
            + (f": {detail}" if detail else "")
        )
    return True


def validate_target_evidence(path: Path, targets: list[Target]) -> bytes:
    observed, raw = load_targets_file(path)
    expected = render_targets(targets)
    if observed != targets or raw != expected:
        raise QualificationError(
            "persisted target evidence differs from derived targets"
        )
    return raw


def build_completion_manifest(
    output: Path,
    launch: dict[str, object],
    summary: dict[str, object],
    rows: list[dict[str, str]],
    targets: list[Target],
) -> dict[str, object]:
    axis = axis_from_launch(launch)
    if validate_stored_launch_manifest(output / "launch.json") != launch:
        raise QualificationError("launch manifest drifted before completion sealing")
    if load_json_object(output / "summary.json", SUMMARY_SCHEMA) != summary:
        raise QualificationError("summary drifted before completion sealing")
    committed_rows, incomplete = load_attempt_state(output, targets, launch)
    if incomplete:
        raise QualificationError(
            "cannot seal completion with incomplete attempts: " + ", ".join(incomplete)
        )
    if committed_rows != rows:
        raise QualificationError("completion rows differ from atomic row fragments")
    if len(rows) != len(expected_rows(targets)):
        raise QualificationError(
            "cannot seal completion before every selected arm commits"
        )
    if summarize_with_launch_policy(rows, targets, launch) != summary:
        raise QualificationError(
            "summary is not the exact recomputation from durable rows"
        )
    allowed = expected_rows(targets)
    artifacts: dict[str, dict[str, object]] = {}
    for row in rows:
        key = (row["cohort"], int(row["cohort_index"]), row["arm"])
        target, order_index, arm = allowed[key]
        validate_row_artifacts(row, target, order_index, arm, output, axis)
        result_path, flight_path, log_path = artifact_paths(output, target, arm)
        attempt_path, row_path = attempt_paths(output, target, arm)
        label = f"{target.cohort}:{target.cohort_index}:{arm}"
        artifacts[label] = {
            "attempt": file_identity(attempt_path),
            "row": file_identity(row_path),
            "result": file_identity(result_path),
            "flight": file_identity(flight_path),
            "log": file_identity(log_path),
        }
    return {
        "schema": COMPLETION_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_inputs_sha256": launch["immutable_inputs_sha256"],
        "promotion_scope": PROMOTION_SCOPE,
        "promotion_pass": bool(summary["promotion_pass"]),
        "runs_observed": len(rows),
        "files": {
            "launch": file_identity(output / "launch.json"),
            "targets": file_identity(output / "targets.csv"),
            "results": file_identity(output / "results.tsv"),
            "summary": file_identity(output / "summary.json"),
        },
        "artifacts": artifacts,
    }


def validate_completion_manifest(
    output: Path, targets: list[Target]
) -> dict[str, object]:
    completion = load_json_object(output / "completion.json", COMPLETION_SCHEMA)
    expected_fields = {
        "schema",
        "completed_at_utc",
        "immutable_inputs_sha256",
        "promotion_scope",
        "promotion_pass",
        "runs_observed",
        "files",
        "artifacts",
    }
    if set(completion) != expected_fields:
        raise QualificationError("completion manifest has unexpected fields")
    if not isinstance(completion.get("completed_at_utc"), str):
        raise QualificationError("completion timestamp is invalid")
    if (
        not isinstance(completion.get("immutable_inputs_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", completion["immutable_inputs_sha256"]) is None
    ):
        raise QualificationError("completion immutable-input digest is invalid")
    if type(completion.get("promotion_pass")) is not bool:
        raise QualificationError("completion promotion disposition is invalid")
    if (
        type(completion.get("runs_observed")) is not int
        or completion["runs_observed"] < 0
    ):
        raise QualificationError("completion row count is invalid")
    if not isinstance(completion.get("artifacts"), dict):
        raise QualificationError("completion artifact manifest is invalid")
    launch = validate_stored_launch_manifest(output / "launch.json")
    axis = axis_from_launch(launch)
    summary = load_json_object(output / "summary.json", SUMMARY_SCHEMA)
    if completion.get("immutable_inputs_sha256") != launch.get(
        "immutable_inputs_sha256"
    ):
        raise QualificationError("completion seal is not bound to the launch inputs")
    if completion.get("promotion_scope") != PROMOTION_SCOPE:
        raise QualificationError("completion seal has the wrong promotion scope")
    if completion.get("promotion_pass") is not summary.get("promotion_pass"):
        raise QualificationError("completion and summary promotion dispositions differ")
    validate_target_evidence(output / "targets.csv", targets)
    fragment_rows, incomplete = load_attempt_state(output, targets, launch)
    if incomplete:
        raise QualificationError(
            "completion contains incomplete attempts: " + ", ".join(incomplete)
        )
    synchronize_results_snapshot(
        output / "results.tsv", fragment_rows, allow_repair=False
    )
    rows = load_persisted_rows(output / "results.tsv", targets, output, axis)
    if rows != fragment_rows:
        raise QualificationError("completion TSV differs from atomic row fragments")
    expected_summary = summarize_with_launch_policy(rows, targets, launch)
    if summary != expected_summary:
        raise QualificationError("completion summary is not an exact row recomputation")
    if completion.get("runs_observed") != len(rows):
        raise QualificationError("completion row count differs from durable results")
    expected_file_paths = {
        "launch": output / "launch.json",
        "targets": output / "targets.csv",
        "results": output / "results.tsv",
        "summary": output / "summary.json",
    }
    files = completion.get("files")
    if not isinstance(files, dict) or set(files) != set(expected_file_paths):
        raise QualificationError("completion campaign-file manifest is invalid")
    for label, path in expected_file_paths.items():
        if files[label] != file_identity(path):
            raise QualificationError(f"completion campaign file drifted: {label}")
    rebuilt = build_completion_manifest(output, launch, summary, rows, targets)
    if completion.get("artifacts") != rebuilt["artifacts"]:
        raise QualificationError("completion arm artifacts drifted")
    return completion


def derive_selected_targets(
    args: argparse.Namespace,
) -> tuple[list[Target], tuple[str, ...]]:
    targets, _official, sound_unsat_tools = derive_targets(
        args.official_root, args.benchmark_root
    )
    if args.limit is not None:
        targets = [target for target in targets if target.cohort_index <= args.limit]
    return targets, sound_unsat_tools


def validate_current_inputs(
    args: argparse.Namespace,
    targets: list[Target],
    target_bytes: bytes,
    receipt_valid: bool,
) -> dict[str, object]:
    rederived, _sound_unsat_tools = derive_selected_targets(args)
    if rederived != targets:
        raise QualificationError(
            "derived qualification target set drifted during campaign"
        )
    current = capture_immutable_inputs(
        args, targets, sha256_bytes(target_bytes), receipt_valid
    )
    launch = validate_launch_manifest(args.out / "launch.json", current)
    validate_target_evidence(args.out / "targets.csv", targets)
    return launch


def main() -> int:
    args = parse_args()
    axis = axis_from_args(args)
    targets, sound_unsat_tools = derive_selected_targets(args)
    print(
        f"axis={axis}; cohort: {sum(t.cohort == 'gain' for t in targets)} gain + "
        f"{sum(t.cohort == 'guard' for t in targets)} guard rows; "
        f"sound UNSAT tools={','.join(sound_unsat_tools)}"
    )

    dirty = repo_is_dirty()
    if dirty and not (args.allow_dirty or args.dry_run):
        raise QualificationError("NY worktree is dirty; commit before qualification")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise QualificationError(f"binary is missing or not executable: {args.binary}")
    receipt_valid = validate_binary_receipt(args)
    target_bytes = render_targets(targets)
    inputs = capture_immutable_inputs(
        args, targets, sha256_bytes(target_bytes), receipt_valid
    )
    if args.dry_run:
        # Bind displayed commands to the exact immutable inputs that a real
        # campaign launch would seal, without creating campaign evidence.
        launch_inputs_sha256 = canonical_json_sha256(inputs)
        for order_index, target in enumerate(targets, 1):
            for arm in arm_order(order_index):
                run_one(
                    target,
                    order_index,
                    arm,
                    args,
                    args.out,
                    launch_inputs_sha256,
                )
        return 0

    if args.resume:
        if args.out.is_symlink() or not args.out.is_dir():
            raise QualificationError(
                f"resume output is not a regular directory: {args.out}"
            )
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        fsync_directory(args.out.parent)

    with campaign_lock(args.out):
        completion_path = args.out / "completion.json"
        if completion_path.is_symlink():
            raise QualificationError(
                f"campaign completion seal must not be a symlink: {completion_path}"
            )
        if completion_path.exists():
            if not args.resume:
                raise QualificationError(
                    f"campaign already has a completion seal: {completion_path}"
                )
            launch = validate_launch_manifest(args.out / "launch.json", inputs)
            validate_target_evidence(args.out / "targets.csv", targets)
            completion = validate_completion_manifest(args.out, targets)
            summary = load_json_object(args.out / "summary.json", SUMMARY_SCHEMA)
            print(json.dumps(summary, indent=2, sort_keys=True))
            print(f"validated existing completion seal: {completion_path}")
            return 0 if completion["promotion_pass"] else 2
        if args.resume:
            launch = validate_launch_manifest(args.out / "launch.json", inputs)
            persisted_target_bytes = validate_target_evidence(
                args.out / "targets.csv", targets
            )
            if (
                sha256_bytes(persisted_target_bytes)
                != inputs["campaign"]["target_csv_sha256"]
            ):
                raise QualificationError(
                    "target evidence is not bound by launch metadata"
                )
        else:
            write_targets(args.out / "targets.csv", targets)
            launch = build_launch_manifest(inputs)
            atomic_write_json(args.out / "launch.json", launch, require_absent=True)
            initialize_results(args.out / "results.tsv")

        results_path = args.out / "results.tsv"
        rows, incomplete_attempts = load_attempt_state(args.out, targets, launch)
        if incomplete_attempts:
            raise QualificationError(
                "campaign contains a presealed incomplete attempt and is permanently "
                "non-promotable; start a new output directory (no retries): "
                + ", ".join(incomplete_attempts)
            )
        # Row fragments are the atomic source of truth. Repair only this derived
        # snapshot after a crash between fragment publication and TSV rename.
        synchronize_results_snapshot(results_path, rows, allow_repair=True)
        rows = load_persisted_rows(results_path, targets, args.out, axis)
        observed = {
            (row["cohort"], int(row["cohort_index"]), row["arm"]): row for row in rows
        }
        for order_index, target in enumerate(targets, 1):
            for arm in arm_order(order_index):
                key = (target.cohort, target.cohort_index, arm)
                if key in observed:
                    print(
                        f"resume: {target.cohort}:{target.cohort_index:02d} {arm}: "
                        "authenticated persisted arm",
                        flush=True,
                    )
                    continue
                validate_current_inputs(args, targets, target_bytes, receipt_valid)
                preseal_attempt(target, order_index, arm, args, args.out, launch)
                row = run_one(
                    target,
                    order_index,
                    arm,
                    args,
                    args.out,
                    require_digest(
                        launch.get("immutable_inputs_sha256"),
                        "launch immutable-input digest",
                    ),
                )
                validate_current_inputs(args, targets, target_bytes, receipt_valid)
                # A presealed attempt is never retried. Publish its validated row
                # as one atomic fragment, then atomically refresh the derived TSV.
                validate_row_artifacts(row, target, order_index, arm, args.out, axis)
                commit_row_fragment(args.out, target, order_index, arm, row, axis)
                rows, incomplete_attempts = load_attempt_state(
                    args.out, targets, launch
                )
                if incomplete_attempts:
                    raise QualificationError(
                        "completed arm left an incomplete attempt ledger: "
                        + ", ".join(incomplete_attempts)
                    )
                publish_results_snapshot(results_path, rows)
                observed[key] = row

        # Reconstruct from durable evidence; never promote from process memory.
        rows, incomplete_attempts = load_attempt_state(args.out, targets, launch)
        if incomplete_attempts:
            raise QualificationError(
                "campaign has incomplete attempts: " + ", ".join(incomplete_attempts)
            )
        synchronize_results_snapshot(results_path, rows, allow_repair=False)
        rows = load_persisted_rows(results_path, targets, args.out, axis)
        receipt_valid_at_completion = validate_binary_receipt(args)
        if receipt_valid_at_completion != receipt_valid:
            raise QualificationError(
                "binary receipt disposition drifted during campaign"
            )
        launch = validate_current_inputs(
            args, targets, target_bytes, receipt_valid_at_completion
        )

        summary = summarize_with_launch_policy(rows, targets, launch)
        atomic_write_json(args.out / "summary.json", summary)

        # Close the small summary/completion race by rechecking every immutable
        # input and every persisted arm immediately before publishing the seal.
        rows = load_persisted_rows(results_path, targets, args.out, axis)
        final_receipt_valid = validate_binary_receipt(args)
        launch = validate_current_inputs(
            args, targets, target_bytes, final_receipt_valid
        )
        completion = build_completion_manifest(args.out, launch, summary, rows, targets)
        atomic_write_json(completion_path, completion, require_absent=True)
        validate_completion_manifest(args.out, targets)

        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"evidence: {args.out}")
        print(f"completion seal: {completion_path}")
        return 0 if summary["promotion_pass"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationError as error:
        print(f"qualification failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None
