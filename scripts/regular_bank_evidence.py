#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Validate the evidence behind promoted VNN-COMP 2025 regular-bank rows.

The evidence index is a transaction log, not a snapshot hash of the whole
measured directory.  Each current entry binds one canonical official
occurrence, its sealed measurement artifacts, and the exact before/after bank
row.  Historical whole-file transaction hashes are retained for provenance,
but later promotions are allowed to change the surrounding CSV bytes.

This module is intentionally read-only.  ``promote_regular_bank.py`` owns the
small atomic mutation layer and uses the validation objects exposed here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import lzma
import math
import re
import stat
import subprocess
import sys
import tarfile
import zlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import archive_vnncomp_sat_result as archive  # noqa: E402
import main16_gap_audit as gap  # noqa: E402
import ny_measurement_provenance as provenance  # noqa: E402
import ny_retroactive_scorecard as retro  # noqa: E402

INDEX_SCHEMA = "ny_regular_bank_evidence_index_v1"
PREVIOUS_ENTRY_SCHEMA = "ny_regular_bank_evidence_entry_v3"
V2_ENTRY_SCHEMA = "ny_regular_bank_evidence_entry_v2"
PRE_PROFILE_ENTRY_SCHEMA = "ny_regular_bank_evidence_entry_v4"
PRE_PROFILE_DYNAMIC_ENTRY_SCHEMA = "ny_regular_bank_evidence_entry_v5"
ENTRY_SCHEMA = "ny_regular_bank_evidence_entry_v6"
DYNAMIC_ENTRY_SCHEMA = "ny_regular_bank_evidence_entry_v7"
LEGACY_DECIDED_ROW_MIGRATION = "ny_regular_bank_legacy_decided_row_migration_v1"
ORGANIZER_RESCORE_SCHEMA = "ny_vnncomp2025_dynamic_organizer_rescore_v1"
OFFICIAL_RESULTS_COMMIT = "ea89fbc2518b6729f17c96eeec22c56c88e496a9"
OFFICIAL_RESULTS_TREE = "a9a7157d900124da491ea57c5b8066276b7cf864"
OFFICIAL_RESULTS_ORIGIN = "https://github.com/VNN-COMP/vnncomp2025_results"
OFFICIAL_ARTIFACT_SHA256: dict[str, str] = {
    "alpha_beta_crown/results.csv": (
        "ae0d8c11d6012ea3e560af332a0182e1b244dceab63c5f58f6d106223cef9a73"
    ),
    "SCORING-ZERO-TOL/latex/longtable.tex": (
        "c8e58f37d8d88e6bcb3af123a8946a229e4b3649b44b96bddebd7cfdfed7284f"
    ),
    "SCORING-ZERO-TOL/latex/scored.tex": (
        "3af6bc118a944eb9041d51d0b030c848eafc3029b375918859ab865a51aa84b8"
    ),
}
ORGANIZER_RESCORE_ARTIFACT_SHA256: dict[str, str] = {
    "alpha_beta_crown/results.csv": (
        "ae0d8c11d6012ea3e560af332a0182e1b244dceab63c5f58f6d106223cef9a73"
    ),
    "cora/results.csv": (
        "5052cb20458d1b48314c14bf67eee7b21288bf6b22a57f0c356e82ebdddad9f2"
    ),
    "neuralsat/results.csv": (
        "ad6e852758ff7d1495f4337abea10cce493fa18ea4093a2189a0dd54bfe2e4a7"
    ),
    "nnenum/results.csv": (
        "e87503beb63f5f0dcacb4f7c55a2f68ebe6cbefe8c8dcc5fcf77ff0a3c58ebfe"
    ),
    "nnv/results.csv": (
        "bdecf3bbede8f284be5acf41230280170960c493e142b94e8949a174ba1d0ed2"
    ),
    "pyrat/results.csv": (
        "fed23601fda086e6a58b7d35b1c5ddbdf87715dc3f6618832e57d77696cdfc30"
    ),
    "sobolbox/results.csv": (
        "07d5ed1d025bbf31c50bb39513a5c07e6ecf670ebef520ba925c1df9a785e073"
    ),
    "SCORING-ZERO-TOL/process_results.py": (
        "5fac024bd7a4a8e67f0983b400b1747a5462457d32ea640efd7d86aa0c89341c"
    ),
    "SCORING-ZERO-TOL/settings.py": (
        "ceeefbd2498cb0a943ee2950440e40a517697cfa899f15a61092f846936256f1"
    ),
    "SCORING-ZERO-TOL/counterexamples.py": (
        "4df1208bb08c1b589dc3f2ac098add44467cf538f7141a7641e7d13001e94e3b"
    ),
    "SCORING-ZERO-TOL/results.txt": (
        "2686e1365738a92b6e10302bebb38f93a5c05a53a0c9fd0fa92409e0256178a2"
    ),
}
ORGANIZER_PARTICIPANTS = (
    "alpha_beta_crown",
    "cora",
    "neuralsat",
    "nnenum",
    "nnv",
    "pyrat",
    "sobolbox",
)
OFFICIAL_BENCHMARK_ORIGIN = "https://github.com/VNN-COMP/vnncomp2025_benchmarks"
OFFICIAL_BENCHMARK_COMMIT = "8b7b811b78ce6a329dc96f04ae6652da3c245948"
OFFICIAL_BENCHMARK_TREE = "29097e0fe614a7a9290e5f1ed98edb75671c8d21"
OFFICIAL_BENCHMARKS_TREE = "db116a77d44618c0883b623a9d79914783e46d28"
PINNED_GIT_EXECUTABLE = Path(
    "<home>/.local/opt/git-1_2.53.0-1ubuntu1/usr/lib/git-core/git"
)
PINNED_GIT_SHA256 = "5516c9f362c29376ab9a499a33082f9f611941d8c75930c880e30ad109e39c9a"
PINNED_REPLAY_RUNNER_SHA256 = (
    "c8d20b67304d0bc52e74ae0a0d279ed10198107379aad670681891b774d372d2"
)
PINNED_REPLAY_WORKER_SHA256 = (
    "001d1ac6af69e61fa108fe60d4589a54a850ad6bf1b7bd72ef5a428bc1410c63"
)
PINNED_LARGE_MODEL_ROOT = Path(
    "<home>/ny-vnncomp2025-large-models-exact-20260731T083000Z"
)
PINNED_LARGE_MODEL_MANIFEST_SHA256 = (
    "f7243cb9fa4dbacee49d439233563cfa08da194b7775af4dfd6966390d7170aa"
)
PINNED_LARGE_MODEL_MANIFEST_SIZE = 1665
LARGE_MODEL_MANIFEST_SCHEMA = "ny_vnncomp2025_large_model_payloads_v1"
EXPECTED_LARGE_MODEL_MANIFEST: dict[str, Any] = {
    "official_benchmark": {
        "commit": OFFICIAL_BENCHMARK_COMMIT,
        "origin": OFFICIAL_BENCHMARK_ORIGIN,
        "setup": {
            "git_blob": "a18991a929b05dde2ff9d725dfd4188d80b27763",
            "git_path": "setup.sh",
            "sha256": (
                "ae1ff4d4c66cccd98e1f49b6b0c2dd280358a4d8a28ec534ea2051fb13fbc46e"
            ),
        },
    },
    "payloads": {
        ("benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx"): {
            "compressed_sha256": (
                "59bb1a768a4e1c2c99053ea9396bc334368cad39bcd24d481bec3267d5f6093e"
            ),
            "compressed_size_bytes": 252305846,
            "compression": "gzip",
            "payload_sha256": (
                "10be6af09db7f6cd116a8b820bb93121b80a6b845df77c0347db4c443b354e35"
            ),
            "payload_size_bytes": 272784587,
            "retained_artifact": (
                "cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx.gz"
            ),
            "source_relative_path": (
                "cgan_2023/seed_896832480/onnx/"
                "cGAN_imgSz32_nCh_3_small_transformer.onnx.gz"
            ),
        },
        "benchmarks/vggnet16_2022/onnx/vgg16-7.onnx": {
            "compressed_sha256": (
                "a49fa13c4c64f5246e509e02f450eff26adfb58b8b5698c1cf38c4cb402683cc"
            ),
            "compressed_size_bytes": 511378437,
            "compression": "gzip",
            "payload_sha256": (
                "f20805a3ecccaa88647bbab4ad011ff2412a9838485ea844de0fcbce349820b9"
            ),
            "payload_size_bytes": 553437328,
            "retained_artifact": "vggnet16_2023/onnx/vgg16-7.onnx.gz",
            "source_relative_path": (
                "vggnet16_2023/seed_896832480/onnx/vgg16-7.onnx.gz"
            ),
        },
    },
    "schema": LARGE_MODEL_MANIFEST_SCHEMA,
    "source": {
        "base_url": "https://rwth-aachen.sciebo.de/public.php/webdav",
        "selected_seed": "896832480",
        "share_id": "RapAoed1dxG1PMs",
    },
}
UNRESOLVED_LITERAL_VERDICTS = frozenset({"timeout", "unknown", "error"})
DECIDED_VERDICTS = frozenset({"sat", "unsat"})
CLAIM_SCOPE = (
    "local_reproducible_internal_counterfactual_not_official_or_independently_attested"
)
FLIGHT_RECORD_CAPTURE_POLICY = (
    "validated-structured-row-metadata-or-explicit-missing-v1"
)
LEGACY_START_PROFILE = "legacy_without_build_coherence_or_flight_v1"
BUILD_COHERENCE_START_PROFILE = "build_coherence_without_flight_v1"
FLIGHT_START_PROFILE = "build_coherence_and_flight_v1"

START_KEYS = frozenset(
    {
        "benchmark",
        "dependencies",
        "environment",
        "host",
        "host_state",
        "measurement",
        "ny",
        "provenance_tools",
        "run_id",
        "rust_toolchain",
        "schema",
        "solver_binary",
        "started_at_utc",
    }
)
NY_WORKTREE_KEYS = frozenset(
    {
        "branch",
        "clean",
        "commit",
        "repo_root",
        "status_porcelain_v1_z_entries",
        "status_sha256",
        "tracked_diff_bytes",
        "tracked_diff_format",
        "tracked_diff_sha256",
        "tracked_worktree_paths",
        "untracked_files",
        "worktree_evidence_sha256",
    }
)
BENCHMARK_WORKTREE_KEYS = (
    NY_WORKTREE_KEYS | {"benchmark_root", "remotes"} - {"repo_root"} | {"repo_root"}
)
LEGACY_MEASUREMENT_KEYS = frozenset(
    {
        "artifact_root",
        "benchmark_root",
        "categories",
        "categories_raw",
        "config_inputs",
        "csv_columns",
        "instance_index",
        "max_rows_per_category",
        "output_dir",
        "result_file",
        "scratch_dir",
        "sealed_config_inputs",
        "solver_command_template",
        "solver_environment",
        "solver_environment_overrides",
        "solver_environment_unsets",
        "solver_log_file",
        "solver_output_capture",
        "sweep_invocation",
        "timeout_cap_seconds",
        "vnnlib_version_selection",
        "watchdog_grace_seconds",
    }
)
MEASUREMENT_KEYS = LEGACY_MEASUREMENT_KEYS | {
    "flight_record_capture",
    "flight_record_file",
}
LEGACY_SOLVER_BINARY_KEYS = frozenset(
    {
        "declared_build_features",
        "declared_build_features_raw",
        "fingerprint",
        "path",
        "sealed_execution",
        "sha256",
        "size_bytes",
        "version_returncode",
        "version_stderr",
        "version_stdout",
    }
)
SOLVER_BINARY_KEYS = LEGACY_SOLVER_BINARY_KEYS | {"build_coherence"}
COMPLETION_KEYS = frozenset(
    {
        "completed_successfully",
        "ended_at_utc",
        "exit_status",
        "host_state",
        "input_hash_cache",
        "integrity",
        "run_id",
        "schema",
        "start_manifest",
        "start_manifest_sha256",
    }
)
COMPLETION_CHECKS = frozenset(
    {
        "solver_binary",
        "sealed_solver_binary",
        "ay_executable",
        "sealed_ay_executable",
        "config_inputs",
        "sealed_config_inputs",
        "cuda_runtime",
        "git_executable",
        "rust_toolchain",
        "containment",
        "ny_worktree",
        "benchmark",
        "git_executable_post",
        "run_evidence",
        "input_hash_cache",
    }
)
RUN_EVIDENCE_KEYS = frozenset(
    {
        "csv_evidence",
        "csv_evidence_sha256",
        "csv_row_count",
        "input_hash_cache_entry_count",
        "metadata_count",
        "preflight_count",
        "produced_rows",
        "records",
        "records_sha256",
        "referenced_input_hash_cache_entry_count",
        "result_count",
        "schema",
        "solver_log_count",
        "status",
        "validated_record_count",
    }
)
RUN_RECORD_KEYS = frozenset(
    {
        "category",
        "elapsed_seconds",
        "input_hash_cache_keys",
        "instance_index",
        "metadata",
        "onnx",
        "preflight",
        "result",
        "solver_exit_status",
        "solver_log",
        "solver_verdict",
        "timeout_seconds",
        "vnnlib",
    }
)
LEGACY_METADATA_KEYS = frozenset(
    {
        "captured_at_utc",
        "category",
        "config_inputs",
        "counterexample_validation",
        "elapsed_seconds",
        "execution_config_inputs",
        "execution_inputs",
        "input_hash_cache",
        "input_preflight",
        "instance_index",
        "onnx",
        "raw_result_sha256",
        "result_artifact",
        "result_sha256",
        "run_id",
        "schema",
        "schema_version",
        "solver_exit_status",
        "solver_log",
        "solver_verdict",
        "source_csv",
        "start_manifest",
        "start_manifest_sha256",
        "timeout_seconds",
        "vnnlib",
        "witness_present",
    }
)
METADATA_KEYS = LEGACY_METADATA_KEYS | {"flight_record"}
BUILD_COHERENCE_KEYS = frozenset(
    {
        "behaviour_input_paths",
        "behaviour_inputs_last_commit_epoch",
        "binary_mtime_epoch",
        "build_input_paths",
        "build_inputs_last_commit_epoch",
    }
)
FLIGHT_RECORD_CAPTURE_KEYS = frozenset(
    {"record", "size_bytes", "source_sha256", "status"}
)
FLIGHT_RECORD_BASE_KEYS = frozenset(
    {
        "ambient_env",
        "backend_kind",
        "backend_summary",
        "budget_secs",
        "category",
        "events",
        "host",
        "schema_version",
    }
)
FLIGHT_RECORD_OPTIONAL_KEYS = frozenset(
    {"load_avg_at_begin", "load_avg_at_end"}
)
FLIGHT_HOST_KEYS = frozenset(
    {"cpu_model", "hostname", "logical_cores", "ram_bytes"}
)
FLIGHT_EVENT_KEYS = frozenset({"method", "status"})
FLIGHT_EVENT_OPTIONAL_KEYS = frozenset({"at_secs", "reason"})
FLIGHT_V2_LEVER_RECEIPT_SCHEMA = "ny-levers/receipt/v1"
FLIGHT_V2_LEVER_RECEIPT_KEYS = frozenset(
    {"env_overridden", "lever_count", "levers", "schema"}
)
FLIGHT_V2_LEVER_ENTRY_KEYS = frozenset(
    {"bucket", "moat", "name", "provenance", "source", "value"}
)
FLIGHT_V2_LEVER_ENTRY_OPTIONAL_KEYS = frozenset({"rejected_raw"})
FLIGHT_V2_LEVER_SOURCES = frozenset({"default", "env"})
PREFLIGHT_KEYS = frozenset(
    {
        "captured_at_utc",
        "category",
        "inputs",
        "instance_index",
        "run_id",
        "schema",
        "start_manifest",
        "start_manifest_sha256",
    }
)
INPUT_HASH_CACHE_KEYS = frozenset(
    {"entries", "run_id", "schema", "start_manifest_sha256", "updated_at_utc"}
)

_LEGACY_ENTRY_KEYS = frozenset(
    {
        "artifact_root",
        "benchmark",
        "category",
        "completion",
        "exact_commit",
        "measured_csv",
        "policy",
        "published_truth",
        "run_id",
        "runtime_seconds",
        "source_csv",
        "start_manifest",
        "verdict",
    }
)
_CURRENT_ENTRY_KEYS = _LEGACY_ENTRY_KEYS | {
    "entry_schema",
    "official_benchmark",
    "official_results",
    "sat_replay",
    "source_snapshot",
}
_PRE_PROFILE_DYNAMIC_ENTRY_KEYS = _CURRENT_ENTRY_KEYS | {"organizer_rescore"}
_PROFILED_ENTRY_KEYS = _CURRENT_ENTRY_KEYS | {"containment_profile"}
_PROFILED_DYNAMIC_ENTRY_KEYS = _PROFILED_ENTRY_KEYS | {"organizer_rescore"}
_V3_ENTRY_KEYS = _CURRENT_ENTRY_KEYS
_V2_ENTRY_KEYS = _LEGACY_ENTRY_KEYS | {
    "entry_schema",
    "official_results",
    "sat_replay",
}


class EvidenceError(RuntimeError):
    """Indexed evidence cannot be interpreted or reconciled safely."""


_PINNED_GIT_FINGERPRINT: dict[str, int] | None = None


@dataclass(frozen=True)
class PinnedOfficialResults:
    root: Path
    context: gap.OfficialContext
    identity: dict[str, Any]


@dataclass(frozen=True)
class PinnedOfficialBenchmark:
    benchmark_root: Path
    repository_root: Path
    identity: dict[str, Any]


@dataclass(frozen=True)
class AuthoritativeInput:
    declared_name: str
    git_path: str | None
    git_blob: str | None
    compression: str
    compressed_sha256: str
    compressed_size_bytes: int
    sha256: str
    size_bytes: int
    retained_setup_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class ValidatedPromotionEvidence:
    artifact_root: Path
    benchmark_root: Path
    official_benchmark: PinnedOfficialBenchmark
    official: PinnedOfficialResults
    run_id: str
    category: str
    instance_index: int
    exact_commit: str
    occurrence: retro.OfficialInstanceOccurrence
    benchmark_occurrence: dict[str, Any]
    start_path: Path
    start: dict[str, Any]
    start_sha256: str
    start_size_bytes: int
    completion_path: Path
    completion: dict[str, Any]
    completion_sha256: str
    completion_size_bytes: int
    sealed: gap.SealedRecord
    raw_record: dict[str, Any]
    verdict: str
    runtime_seconds: str
    source_csv: Path
    source_row: list[str]
    source_csv_data: bytes
    published_truth: str
    policy: str
    sat_replay: dict[str, Any] | None
    organizer_rescore: dict[str, Any] | None
    containment_profile: str | None
    authoritative_inputs: dict[str, AuthoritativeInput]
    source_snapshot: dict[str, Any]


@dataclass(frozen=True)
class BankRow:
    line_index: int
    fields: list[str]


@dataclass(frozen=True)
class ValidatedIndexEntry:
    row_key: str
    entry: dict[str, Any]
    evidence: ValidatedPromotionEvidence
    measured_path: Path
    bank_row: list[str]
    bank_state: str
    legacy_entry: bool


@dataclass(frozen=True)
class ValidatedEvidenceIndex:
    path: Path
    existed: bool
    data: bytes | None
    value: dict[str, Any]
    official: PinnedOfficialResults
    entries: tuple[ValidatedIndexEntry, ...]

    @property
    def dangling_entries(self) -> tuple[ValidatedIndexEntry, ...]:
        return tuple(entry for entry in self.entries if entry.bank_state == "dangling")

    @property
    def creditable_entries(self) -> tuple[ValidatedIndexEntry, ...]:
        """Only fully applied rows are eligible for scorecard credit."""

        return tuple(entry for entry in self.entries if entry.bank_state == "applied")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _decimal(value: object, *, label: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise EvidenceError(f"{label} is not a numeric scalar")
    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation as error:
        raise EvidenceError(f"{label} is not numeric: {value!r}") from error
    if not parsed.is_finite() or parsed < 0 or (positive and parsed <= 0):
        qualifier = "positive" if positive else "nonnegative"
        raise EvidenceError(f"{label} must be finite and {qualifier}: {value!r}")
    return parsed


def resolved_directory(path: Path, label: str) -> Path:
    try:
        return gap._require_directory(path, label)
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error


def resolved_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise EvidenceError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as error:
        raise EvidenceError(f"{label} is unavailable: {path}") from error
    if not stat.S_ISREG(mode):
        raise EvidenceError(f"{label} is not a regular file: {resolved}")
    return resolved


def stable_bytes(path: Path, label: str) -> bytes:
    resolved = resolved_regular_file(path, label)
    try:
        data, _, _ = provenance._stable_file_bytes(resolved)
    except (OSError, provenance.ProvenanceError) as error:
        raise EvidenceError(
            f"could not read stable {label} {resolved}: {error}"
        ) from error
    return data


def _git_environment() -> dict[str, str]:
    """Return an environment that cannot redirect or replace Git objects."""

    # Git is used only for local object inspection.  Do not inherit loader,
    # shell-startup, locale, credential, user-config, or repository-redirection
    # state from the caller.
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent/ny-evidence-git-home",
        "XDG_CONFIG_HOME": "/nonexistent/ny-evidence-git-xdg",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _trusted_git_executable(*, force_hash: bool = False) -> Path:
    global _PINNED_GIT_FINGERPRINT
    path = resolved_regular_file(PINNED_GIT_EXECUTABLE, "pinned Git executable")
    if path != PINNED_GIT_EXECUTABLE:
        raise EvidenceError("pinned Git executable path is not canonical")
    try:
        fingerprint = provenance._file_fingerprint(path)
    except OSError as error:
        raise EvidenceError("could not stat pinned Git executable") from error
    if (
        force_hash
        or _PINNED_GIT_FINGERPRINT is None
        or fingerprint != _PINNED_GIT_FINGERPRINT
    ):
        data = stable_bytes(path, "pinned Git executable")
        if sha256(data) != PINNED_GIT_SHA256:
            raise EvidenceError("pinned Git executable bytes differ")
        _PINNED_GIT_FINGERPRINT = fingerprint
    return path


def _git(
    repository: Path,
    *arguments: str,
    input_data: bytes | None = None,
    allow_failure: bool = False,
) -> bytes | None:
    command = [
        str(_trusted_git_executable()),
        "--no-replace-objects",
        "-c",
        "core.useReplaceRefs=false",
        "-C",
        str(repository),
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            check=False,
            timeout=60,
            env=_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvidenceError(
            f"could not inspect Git repository {repository}: {error}"
        ) from error
    if result.returncode != 0:
        if allow_failure:
            return None
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise EvidenceError(
            f"Git inspection failed for {repository}: {detail or result.returncode}"
        )
    return result.stdout


def _git_text(repository: Path, *arguments: str) -> str:
    data = _git(repository, *arguments)
    assert data is not None
    try:
        return data.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise EvidenceError(
            f"Git returned non-UTF-8 identity data for {repository}"
        ) from error


def validate_official_benchmark(root: Path) -> PinnedOfficialBenchmark:
    """Bind the benchmark to immutable objects in the pinned official commit."""

    benchmark_root = resolved_directory(root, "benchmark root")
    repository = Path(
        _git_text(benchmark_root, "rev-parse", "--show-toplevel").strip()
    ).resolve(strict=True)
    if benchmark_root != repository / "benchmarks":
        raise EvidenceError(
            "official benchmark root must be the benchmarks/ tree of its repository"
        )
    commit = _git_text(repository, "rev-parse", "HEAD").strip()
    tree = _git_text(repository, "rev-parse", "HEAD^{tree}").strip()
    benchmarks_tree = _git_text(repository, "rev-parse", "HEAD:benchmarks").strip()
    origin = _git_text(repository, "remote", "get-url", "origin").strip()
    if (
        commit != OFFICIAL_BENCHMARK_COMMIT
        or tree != OFFICIAL_BENCHMARK_TREE
        or benchmarks_tree != OFFICIAL_BENCHMARKS_TREE
        or origin != OFFICIAL_BENCHMARK_ORIGIN
    ):
        raise EvidenceError(
            "official 2025 benchmark Git identity differs from the pinned "
            "commit/tree/origin"
        )
    object_type = _git_text(repository, "cat-file", "-t", benchmarks_tree).strip()
    if object_type != "tree":
        raise EvidenceError("pinned official benchmarks object is not a Git tree")
    # Replacement refs and grafts are forbidden even though every plumbing call
    # also disables replacement lookup.
    replace_refs = _git_text(
        repository, "for-each-ref", "--format=%(refname)", "refs/replace/"
    ).splitlines()
    grafts = repository / ".git" / "info" / "grafts"
    if replace_refs or (grafts.exists() and stable_bytes(grafts, "Git grafts").strip()):
        raise EvidenceError(
            "official benchmark repository contains replacement or graft state"
        )
    identity = {
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "benchmarks_tree": benchmarks_tree,
    }
    return PinnedOfficialBenchmark(benchmark_root, repository, identity)


def revalidate_official_benchmark(benchmark: PinnedOfficialBenchmark) -> None:
    _trusted_git_executable(force_hash=True)
    current = validate_official_benchmark(benchmark.benchmark_root)
    if current.identity != benchmark.identity:
        raise EvidenceError("official benchmark identity changed during validation")


def _git_blob(
    benchmark: PinnedOfficialBenchmark, git_path: str
) -> tuple[str, bytes] | None:
    spec = f"{OFFICIAL_BENCHMARK_COMMIT}:{git_path}"
    object_id_data = _git(
        benchmark.repository_root,
        "rev-parse",
        "--verify",
        spec,
        allow_failure=True,
    )
    if object_id_data is None:
        return None
    object_id = object_id_data.decode("ascii", "strict").strip()
    if not re_fullmatch_hex(object_id, 40):
        raise EvidenceError(f"invalid Git object identity for {git_path}")
    object_type = _git_text(
        benchmark.repository_root, "cat-file", "-t", object_id
    ).strip()
    if object_type != "blob":
        raise EvidenceError(f"official benchmark path is not a blob: {git_path}")
    data = _git(benchmark.repository_root, "cat-file", "blob", object_id)
    assert data is not None
    return object_id, data


def re_fullmatch_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_benchmark_name(value: str, *, label: str) -> str:
    if not value or "\\" in value or "\0" in value:
        raise EvidenceError(f"official {label} path is unsafe: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise EvidenceError(f"official {label} path is absolute: {value!r}")
    parts = list(path.parts)
    while parts and parts[0] == ".":
        parts.pop(0)
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise EvidenceError(f"official {label} path is unsafe: {value!r}")
    return PurePosixPath(*parts).as_posix()


def _strict_decompress(data: bytes, *, compression: str, label: str) -> bytes:
    try:
        if compression == "none":
            return data
        if compression == "gzip":
            decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
            result = decompressor.decompress(data)
            result += decompressor.flush()
            if (
                not decompressor.eof
                or decompressor.unused_data
                or decompressor.unconsumed_tail
            ):
                raise EvidenceError(
                    f"official {label} gzip is truncated, multi-member, or trailing"
                )
            return result
        if compression == "xz":
            decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
            result = decompressor.decompress(data)
            if not decompressor.eof or decompressor.unused_data:
                raise EvidenceError(
                    f"official {label} xz is truncated, multi-stream, or trailing"
                )
            return result
    except (lzma.LZMAError, zlib.error) as error:
        raise EvidenceError(f"official {label} compression is invalid") from error
    raise EvidenceError(f"internal: unsupported compression {compression}")


def _validate_large_model_inventory(root: Path) -> None:
    expected = {
        "cgan_2023": ("directory", 0o555),
        "cgan_2023/onnx": ("directory", 0o555),
        ("cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx.gz"): (
            "file",
            0o444,
        ),
        "manifest.json": ("file", 0o444),
        "vggnet16_2023": ("directory", 0o555),
        "vggnet16_2023/onnx": ("directory", 0o555),
        "vggnet16_2023/onnx/vgg16-7.onnx.gz": ("file", 0o444),
    }
    try:
        root_mode = stat.S_IMODE(root.lstat().st_mode)
    except OSError as error:
        raise EvidenceError("retained large-model root is unavailable") from error
    if root.is_symlink() or root_mode != 0o555:
        raise EvidenceError(
            "retained large-model root must be a canonical mode-0555 directory"
        )

    observed: dict[str, tuple[str, int]] = {}
    pending = [root]
    try:
        while pending:
            directory = pending.pop()
            for child in directory.iterdir():
                relative = child.relative_to(root).as_posix()
                child_stat = child.lstat()
                if stat.S_ISLNK(child_stat.st_mode):
                    raise EvidenceError(
                        f"retained large-model inventory contains a symlink: {relative}"
                    )
                if stat.S_ISDIR(child_stat.st_mode):
                    kind = "directory"
                    pending.append(child)
                elif stat.S_ISREG(child_stat.st_mode):
                    kind = "file"
                    if child_stat.st_nlink != 1:
                        raise EvidenceError(
                            "retained large-model inventory contains a hard-linked "
                            f"file: {relative}"
                        )
                else:
                    raise EvidenceError(
                        "retained large-model inventory contains a special file: "
                        f"{relative}"
                    )
                observed[relative] = (
                    kind,
                    stat.S_IMODE(child_stat.st_mode),
                )
    except OSError as error:
        raise EvidenceError(
            "could not inspect retained large-model inventory"
        ) from error
    if observed != expected:
        raise EvidenceError("retained large-model inventory or immutable modes differ")


def _retained_source_binding(
    *,
    root: Path,
    manifest_path: Path,
    logical_path: str,
    setup: dict[str, Any],
    payload_binding: dict[str, Any],
    retained_path: Path,
) -> dict[str, Any]:
    return {
        "kind": "official_setup_retained_payload_v1",
        "logical_path": logical_path,
        "manifest": {
            "path": str(manifest_path),
            "schema": LARGE_MODEL_MANIFEST_SCHEMA,
            "sha256": PINNED_LARGE_MODEL_MANIFEST_SHA256,
            "size_bytes": PINNED_LARGE_MODEL_MANIFEST_SIZE,
        },
        "official_setup": setup,
        "retained_artifact": {
            "path": str(retained_path),
            "relative_path": retained_path.relative_to(root).as_posix(),
        },
        "source": {
            **EXPECTED_LARGE_MODEL_MANIFEST["source"],
            "relative_path": payload_binding["source_relative_path"],
        },
    }


def _retained_large_model_payload(
    *,
    benchmark: PinnedOfficialBenchmark,
    logical_path: str,
    declared_name: str,
    label: str,
) -> tuple[AuthoritativeInput, bytes] | None:
    expected_payloads = EXPECTED_LARGE_MODEL_MANIFEST["payloads"]
    if logical_path not in expected_payloads:
        return None

    root = resolved_directory(PINNED_LARGE_MODEL_ROOT, "retained large-model root")
    if root != PINNED_LARGE_MODEL_ROOT:
        raise EvidenceError("retained large-model root path is not canonical")
    _validate_large_model_inventory(root)

    manifest_path = resolved_regular_file(
        root / "manifest.json", "retained large-model manifest"
    )
    manifest_data = stable_bytes(manifest_path, "retained large-model manifest")
    if (
        len(manifest_data) != PINNED_LARGE_MODEL_MANIFEST_SIZE
        or sha256(manifest_data) != PINNED_LARGE_MODEL_MANIFEST_SHA256
    ):
        raise EvidenceError("retained large-model manifest bytes differ")
    manifest = _json_object(
        manifest_data,
        path=manifest_path,
        label="retained large-model manifest",
    )
    if manifest != EXPECTED_LARGE_MODEL_MANIFEST:
        raise EvidenceError("retained large-model manifest content differs")

    official = manifest["official_benchmark"]
    if official.get("commit") != benchmark.identity.get("commit") or official.get(
        "origin"
    ) != benchmark.identity.get("origin"):
        raise EvidenceError("retained large-model manifest benchmark identity differs")
    setup = official["setup"]
    setup_blob = _git_blob(benchmark, setup["git_path"])
    if (
        setup_blob is None
        or setup_blob[0] != setup["git_blob"]
        or sha256(setup_blob[1]) != setup["sha256"]
    ):
        raise EvidenceError(
            "retained large-model manifest is not bound to pinned setup.sh"
        )

    payload_binding = expected_payloads[logical_path]
    retained_relative = _safe_benchmark_name(
        payload_binding["retained_artifact"],
        label=f"retained {label}",
    )
    retained_path = resolved_regular_file(
        root.joinpath(*PurePosixPath(retained_relative).parts),
        f"retained official {label}",
    )
    try:
        retained_path.relative_to(root)
        compressed, compressed_digest, compressed_fingerprint = (
            provenance._stable_file_bytes(retained_path)
        )
    except (OSError, ValueError, provenance.ProvenanceError) as error:
        raise EvidenceError(
            f"could not read stable retained official {label}"
        ) from error
    if (
        compressed_digest != payload_binding["compressed_sha256"]
        or len(compressed) != payload_binding["compressed_size_bytes"]
    ):
        raise EvidenceError(f"retained official {label} compressed bytes differ")
    payload = _strict_decompress(
        compressed,
        compression=payload_binding["compression"],
        label=f"retained {label}",
    )
    if (
        sha256(payload) != payload_binding["payload_sha256"]
        or len(payload) != payload_binding["payload_size_bytes"]
    ):
        raise EvidenceError(f"retained official {label} payload bytes differ")
    try:
        final_digest, final_fingerprint = provenance._stable_file_hash(retained_path)
    except (OSError, provenance.ProvenanceError) as error:
        raise EvidenceError(f"could not recheck retained official {label}") from error
    if (
        final_digest != compressed_digest
        or final_fingerprint != compressed_fingerprint
        or stable_bytes(manifest_path, "retained large-model manifest") != manifest_data
    ):
        raise EvidenceError(
            f"retained official {label} source changed during validation"
        )
    _validate_large_model_inventory(root)

    retained_source = _retained_source_binding(
        root=root,
        manifest_path=manifest_path,
        logical_path=logical_path,
        setup=setup,
        payload_binding=payload_binding,
        retained_path=retained_path,
    )
    return (
        AuthoritativeInput(
            declared_name=declared_name,
            git_path=None,
            git_blob=None,
            compression=payload_binding["compression"],
            compressed_sha256=payload_binding["compressed_sha256"],
            compressed_size_bytes=payload_binding["compressed_size_bytes"],
            sha256=payload_binding["payload_sha256"],
            size_bytes=payload_binding["payload_size_bytes"],
            retained_setup_payload=retained_source,
        ),
        payload,
    )


def authoritative_benchmark_input(
    *,
    benchmark: PinnedOfficialBenchmark,
    category: str,
    declared_name: str,
    label: str,
    payload_cache: dict[str, tuple[AuthoritativeInput, bytes]] | None = None,
) -> tuple[AuthoritativeInput, bytes]:
    normalized = _safe_benchmark_name(declared_name, label=label)
    base = f"benchmarks/{category}/{normalized}"
    candidates: list[tuple[str, str, str, bytes]] = []
    for suffix, compression in (("", "none"), (".gz", "gzip"), (".xz", "xz")):
        found = _git_blob(benchmark, base + suffix)
        if found is not None:
            object_id, data = found
            candidates.append((base + suffix, object_id, compression, data))
    if not candidates:
        cached = payload_cache.get(base) if payload_cache is not None else None
        if cached is not None:
            authoritative, payload = cached
            return replace(authoritative, declared_name=declared_name), payload
        retained = _retained_large_model_payload(
            benchmark=benchmark,
            logical_path=base,
            declared_name=declared_name,
            label=label,
        )
        if retained is not None:
            if payload_cache is not None:
                payload_cache[base] = retained
            return retained
    if len(candidates) != 1:
        raise EvidenceError(
            f"official {label} payload is missing or ambiguous in the pinned "
            f"benchmark commit (found {len(candidates)})"
        )
    git_path, object_id, compression, compressed = candidates[0]
    payload = _strict_decompress(compressed, compression=compression, label=label)
    return (
        AuthoritativeInput(
            declared_name=declared_name,
            git_path=git_path,
            git_blob=object_id,
            compression=compression,
            compressed_sha256=sha256(compressed),
            compressed_size_bytes=len(compressed),
            sha256=sha256(payload),
            size_bytes=len(payload),
        ),
        payload,
    )


def _json_object(data: bytes, *, path: Path, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise EvidenceError(
                    f"{label} contains duplicate JSON key {key!r}: {path}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except EvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise EvidenceError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object: {path}")
    return value


def _parse_csv(data: bytes, *, path: Path) -> list[list[str]]:
    try:
        return list(
            csv.reader(
                io.StringIO(data.decode("utf-8"), newline=""),
                strict=True,
            )
        )
    except (UnicodeDecodeError, csv.Error) as error:
        raise EvidenceError(f"invalid CSV artifact {path}: {error}") from error


def _safe_index_artifact(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(
        component not in {"", ".", ".."} for component in path.parts
    )


def _safe_source_archive_path(value: object) -> bool:
    """Accept only a canonical relative UTF-8 Git/archive path.

    Archive member names may legitimately exceed the ustar name field and use
    a PAX ``path`` record.  They must not inherit the narrower evidence-index
    representation rule merely because both kinds of value are relative paths.
    """

    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        return False
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and all(component not in {"", ".", ".."} for component in path.parts)
    )


def validate_official_results(
    root: Path,
) -> PinnedOfficialResults:
    """Load official results only after their authoritative bytes match."""

    resolved = resolved_directory(root, "official result root")
    artifacts: dict[str, dict[str, Any]] = {}
    for relative, expected_sha256 in OFFICIAL_ARTIFACT_SHA256.items():
        path = resolved.joinpath(*PurePosixPath(relative).parts)
        data = stable_bytes(path, f"official result artifact {relative}")
        actual_sha256 = sha256(data)
        if actual_sha256 != expected_sha256:
            raise EvidenceError(
                "official 2025 result identity mismatch for "
                f"{relative}: expected {expected_sha256}, found {actual_sha256}"
            )
        artifacts[relative] = {
            "sha256": actual_sha256,
            "size_bytes": len(data),
        }
    try:
        context = gap.load_official_context(resolved)
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error
    identity = {
        "release_commit": OFFICIAL_RESULTS_COMMIT,
        "artifacts": artifacts,
    }
    return PinnedOfficialResults(resolved, context, identity)


def revalidate_official_results(official: PinnedOfficialResults) -> None:
    current = validate_official_results(official.root)
    if current.identity != official.identity:
        raise EvidenceError("official result identity changed during validation")


def _load_organizer_rescore_artifacts(
    official: PinnedOfficialResults,
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    """Open the raw organizer corpus needed for a truth-changing rescore.

    These inputs are deliberately separate from the frozen three-file official
    identity used by legacy/v2-v4 entries.  Consequently, adding this closure
    does not rewrite or reinterpret any already-published evidence entry.
    """

    payloads: dict[str, bytes] = {}
    identity: dict[str, dict[str, Any]] = {}
    for relative, expected_digest in ORGANIZER_RESCORE_ARTIFACT_SHA256.items():
        path = official.root.joinpath(*PurePosixPath(relative).parts)
        data = stable_bytes(path, f"organizer rescore artifact {relative}")
        digest = sha256(data)
        if digest != expected_digest:
            raise EvidenceError(
                "organizer rescore artifact identity mismatch for "
                f"{relative}: expected {expected_digest}, found {digest}"
            )
        payloads[relative] = data
        identity[relative] = {
            "sha256": digest,
            "size_bytes": len(data),
        }
    return payloads, identity


def _organizer_results_repository_identity(
    official: PinnedOfficialResults,
) -> dict[str, str]:
    top = _git(official.root, "rev-parse", "--show-toplevel")
    head = _git(official.root, "rev-parse", "HEAD")
    tree = _git(official.root, "rev-parse", "HEAD^{tree}")
    origin = _git(official.root, "config", "--get", "remote.origin.url")
    status = _git(
        official.root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if any(value is None for value in (top, head, tree, origin, status)):
        raise EvidenceError("organizer result repository identity is incomplete")
    assert top is not None
    assert head is not None
    assert tree is not None
    assert origin is not None
    assert status is not None
    try:
        top_path = Path(top.decode("utf-8", "strict").strip()).resolve(strict=True)
        observed_head = head.decode("ascii", "strict").strip()
        observed_tree = tree.decode("ascii", "strict").strip()
        observed_origin = origin.decode("utf-8", "strict").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(
            "organizer result repository identity is malformed"
        ) from error
    if top_path != official.root:
        raise EvidenceError("organizer result root is not the Git repository root")
    if observed_head != OFFICIAL_RESULTS_COMMIT:
        raise EvidenceError("organizer result repository is not at the pinned commit")
    if observed_tree != OFFICIAL_RESULTS_TREE:
        raise EvidenceError("organizer result repository tree differs from the pin")
    if observed_origin != OFFICIAL_RESULTS_ORIGIN:
        raise EvidenceError("organizer result repository origin differs from the pin")
    if status:
        raise EvidenceError("organizer result repository worktree is not clean")
    return {
        "commit": observed_head,
        "origin": observed_origin,
        "tree": observed_tree,
    }


def _organizer_category_score_id(
    official: PinnedOfficialResults,
    *,
    category: str,
    occurrence: retro.OfficialInstanceOccurrence,
) -> int:
    rows = official.context.reference_order.get(category, [])
    matches = [
        index
        for index, score_key in enumerate(rows)
        if score_key == occurrence.score_key
    ]
    if len(matches) != 1:
        raise EvidenceError(
            "organizer rescore cannot bind the target to one published score ID"
        )
    return matches[0]


def _organizer_raw_category_rows(
    *,
    data: bytes,
    path: Path,
    category: str,
) -> list[tuple[int, list[str], tuple[str, str, int]]]:
    parsed = _parse_csv(data, path=path)
    rows: list[tuple[int, list[str], tuple[str, str, int]]] = []
    occurrences: dict[tuple[str, str], int] = {}
    for physical_index, row in enumerate(parsed, start=1):
        if len(row) < 6 or row[0].strip() != category:
            continue
        onnx, vnnlib = row[1], row[2]
        if retro.is_harness_test_instance(onnx, vnnlib):
            continue
        base = retro.key(onnx, vnnlib)
        pair_occurrence = occurrences.get(base, 0)
        occurrences[base] = pair_occurrence + 1
        rows.append((physical_index, row, (*base, pair_occurrence)))
    return rows


_LATEX_TOOL_TO_RAW = {
    "$\\alpha$-$\\beta$-CROWN": "alpha_beta_crown",
    "CORA": "cora",
    "NeuralSAT": "neuralsat",
    "nnenum": "nnenum",
    "NNV": "nnv",
    "PyRAT": "pyrat",
    "SobolBox": "sobolbox",
}


def _published_category_tool_totals(
    *,
    scored_data: bytes,
    scored_path: Path,
    category: str,
) -> dict[str, dict[str, int]]:
    try:
        text = scored_data.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise EvidenceError("published scored.tex is not UTF-8") from error
    start_marker = f"% Category 2025_{category} "
    starts = [
        match.start() for match in re.finditer(rf"(?m)^{re.escape(start_marker)}", text)
    ]
    if len(starts) != 1:
        raise EvidenceError(
            f"published scored.tex has {len(starts)} tables for {category}"
        )
    end_match = re.search(r"(?m)^% Category 2025_", text[starts[0] + 1 :])
    end = starts[0] + 1 + end_match.start() if end_match is not None else len(text)
    section = text[starts[0] : end]
    totals: dict[str, dict[str, int]] = {}
    for line in section.splitlines():
        if re.match(r"^\d+\s*&", line) is None:
            continue
        fields = [field.strip() for field in line.split("&")]
        if len(fields) < 7:
            raise EvidenceError(
                f"published {category} score row is incomplete in {scored_path}"
            )
        tool = _LATEX_TOOL_TO_RAW.get(fields[1])
        if tool is None or tool in totals:
            raise EvidenceError(
                f"published {category} score row has an unknown/duplicate tool"
            )
        try:
            verified = int(fields[2])
            falsified = int(fields[3])
            fastest = int(fields[4])
            penalties = int(fields[5])
            points = int(fields[6])
        except ValueError as error:
            raise EvidenceError(
                f"published {category} score row has noninteger counts"
            ) from error
        totals[tool] = {
            "verified": verified,
            "falsified": falsified,
            "fastest": fastest,
            "penalties": penalties,
            "points": points,
        }
    if not totals:
        raise EvidenceError(f"published scored.tex has no rows for {category}")
    return totals


def _organizer_detailed_category_block(
    *,
    results_data: bytes,
    category: str,
) -> tuple[str, tuple[str, ...]]:
    try:
        text = results_data.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise EvidenceError("organizer results.txt is not UTF-8") from error
    pattern = re.compile(
        rf"(?m)^Category 2025_{re.escape(category)}:$\n"
        rf"Category 2025_{re.escape(category)} has \d+ \(from [^)]+\)$\n"
        r"(\d+) participating tools: (\[[^\n]+\])$"
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise EvidenceError(
            f"organizer results.txt has {len(matches)} detailed blocks for {category}"
        )
    match = matches[0]
    try:
        parsed_tools = ast.literal_eval(match.group(2))
    except (SyntaxError, ValueError) as error:
        raise EvidenceError("organizer participant list is malformed") from error
    if (
        not isinstance(parsed_tools, list)
        or not all(isinstance(tool, str) for tool in parsed_tools)
        or int(match.group(1)) != len(parsed_tools)
        or len(set(parsed_tools)) != len(parsed_tools)
        or any(tool not in ORGANIZER_PARTICIPANTS for tool in parsed_tools)
    ):
        raise EvidenceError("organizer participant list is invalid")
    next_category = re.search(r"(?m)^Category 2025_", text[match.end() :])
    end = (
        match.end() + next_category.start() if next_category is not None else len(text)
    )
    return text[match.end() : end], tuple(parsed_tools)


_ORGANIZER_SCORE_LINE = re.compile(
    r"(?m)^(\d+): ([a-z0-9_]+) score: (-?\d+), "
    r"is_ver: (True|False), is_fals: (True|False), "
    r"is_fastest: (True|False)$"
)


def _organizer_logged_instance(
    *,
    block: str,
    score_id: int,
    participants: tuple[str, ...],
) -> tuple[str, dict[str, str], dict[str, int]]:
    matches = list(_ORGANIZER_SCORE_LINE.finditer(block))
    selected = [match for match in matches if int(match.group(1)) == score_id]
    if len(selected) != len(participants):
        raise EvidenceError(
            "organizer results.txt does not score every participant on the target"
        )
    if tuple(match.group(2) for match in selected) != participants:
        raise EvidenceError(
            "organizer target score order differs from the participant list"
        )
    previous = [match for match in matches if int(match.group(1)) < score_id]
    segment_start = previous[-1].end() if previous else 0
    segment = block[segment_start : selected[0].start()]
    truths = re.findall(r"(?m)^True Result: (sat|unsat|-)$", segment)
    if len(truths) != 1:
        raise EvidenceError(
            "organizer results.txt has no unique target truth classification"
        )
    if truths[0] == "-":
        raise EvidenceError(
            "organizer results.txt target truth classification is indeterminate"
        )
    ce_lines = re.findall(
        r"(?m)^were violated counterexamples valid\?: (\{[^\n]*\})$",
        segment,
    )
    if len(ce_lines) > 1:
        raise EvidenceError(
            "organizer results.txt has duplicate target CE classifications"
        )
    classifications: dict[str, str] = {}
    if ce_lines:
        try:
            parsed = ast.literal_eval(ce_lines[0])
        except (SyntaxError, ValueError) as error:
            raise EvidenceError(
                "organizer target CE classification is malformed"
            ) from error
        if not isinstance(parsed, dict) or not all(
            isinstance(tool, str) and isinstance(result, str)
            for tool, result in parsed.items()
        ):
            raise EvidenceError("organizer target CE classification is invalid")
        classifications = dict(parsed)
    return (
        "violated" if truths[0] == "sat" else "holds",
        classifications,
        {match.group(2): int(match.group(3)) for match in selected},
    )


def _counterexample_classification(value: str) -> Any:
    mapping = {
        "correct": gap.competitive.CounterexampleResult.CORRECT,
        "correct_up_to_tolerance": (
            gap.competitive.CounterexampleResult.CORRECT_UP_TO_TOLERANCE
        ),
        "exec_doesnt_match": (gap.competitive.CounterexampleResult.EXEC_DOESNT_MATCH),
        "spec_not_violated": (gap.competitive.CounterexampleResult.SPEC_NOT_VIOLATED),
        "wrong_shape": gap.competitive.CounterexampleResult.EXEC_DOESNT_MATCH,
        "no_ce": gap.competitive.CounterexampleResult.NO_COUNTEREXAMPLE,
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise EvidenceError(
            f"unsupported organizer counterexample classification: {value}"
        ) from error


def _score_outcome(points: int) -> str:
    if points == retro.POINTS_CORRECT:
        return "correct"
    if points == retro.PENALTY_INCORRECT:
        return "penalty"
    if points == 0:
        return "no_credit"
    raise EvidenceError(f"unexpected organizer instance score: {points}")


def dynamic_organizer_rescore(
    *,
    official: PinnedOfficialResults,
    category: str,
    occurrence: retro.OfficialInstanceOccurrence,
) -> dict[str, Any]:
    """Recompute one strict truth change from the pinned organizer corpus."""

    repository_identity = _organizer_results_repository_identity(official)
    payloads, artifact_identity = _load_organizer_rescore_artifacts(official)
    score_id = _organizer_category_score_id(
        official,
        category=category,
        occurrence=occurrence,
    )
    results_relative = "SCORING-ZERO-TOL/results.txt"
    block, participants = _organizer_detailed_category_block(
        results_data=payloads[results_relative],
        category=category,
    )
    scored_relative = "SCORING-ZERO-TOL/latex/scored.tex"
    scored_data = stable_bytes(
        official.root.joinpath(*PurePosixPath(scored_relative).parts),
        "published ZERO-TOL scored.tex",
    )
    if sha256(scored_data) != OFFICIAL_ARTIFACT_SHA256[scored_relative]:
        raise EvidenceError("published ZERO-TOL scored.tex changed during rescore")
    baseline = _published_category_tool_totals(
        scored_data=scored_data,
        scored_path=official.root / scored_relative,
        category=category,
    )
    if set(baseline) != set(participants):
        raise EvidenceError(
            "published score table and organizer participant set differ"
        )
    logged_truth, ce_results, logged_scores = _organizer_logged_instance(
        block=block,
        score_id=score_id,
        participants=participants,
    )
    published_truth = official.context.ground_truth.get(category, {}).get(
        occurrence.score_key
    )
    if published_truth != "holds" or logged_truth != published_truth:
        raise EvidenceError(
            "dynamic organizer rescore requires matching published holds truth"
        )

    raw_rows: dict[str, dict[str, Any]] = {}
    official_field: list[Any] = []
    instance_name = canonical_row_key(category, occurrence)
    reference_order = official.context.reference_order.get(category, [])
    for tool in participants:
        relative = f"{tool}/results.csv"
        rows = _organizer_raw_category_rows(
            data=payloads[relative],
            path=official.root / relative,
            category=category,
        )
        observed_order = [score_key for _, _, score_key in rows]
        if observed_order != reference_order:
            raise EvidenceError(
                f"organizer raw result order differs for participant {tool}"
            )
        physical_row, row, score_key = rows[score_id]
        if score_key != occurrence.score_key:
            raise EvidenceError(
                f"organizer raw target binding differs for participant {tool}"
            )
        raw_result = row[4]
        result = gap.competitive.normalize_result(raw_result)
        classification = ce_results.get(tool)
        if (result == "violated") != (classification is not None):
            raise EvidenceError(
                f"organizer CE classification coverage differs for {tool}"
            )
        counterexample = (
            _counterexample_classification(classification)
            if classification is not None
            else None
        )
        official_field.append(
            gap.competitive.InstanceResult(
                tool=tool,
                benchmark=category,
                instance=instance_name,
                result=result,
                counterexample=counterexample,
                ce_required=True,
            )
        )
        raw_rows[tool] = {
            "artifact": relative,
            "canonical_result": result,
            "counterexample_classification": classification,
            "physical_row": physical_row,
            "raw_result": raw_result,
            "row_sha256": provenance._identity_sha256(row),
        }

    candidate = gap.competitive.InstanceResult(
        tool="ny",
        benchmark=category,
        instance=instance_name,
        result="violated",
        counterexample=gap.competitive.CounterexampleResult.CORRECT,
        ce_required=True,
    )
    rescored_field = [*official_field, candidate]
    participants_payload: dict[str, dict[str, Any]] = {}
    rescored_totals: dict[str, int] = {}
    for result in official_field:
        old_points = gap.competitive.score_instance(result, official_field)
        if logged_scores.get(result.tool) != old_points:
            raise EvidenceError(
                f"local scorer does not reproduce organizer points for {result.tool}"
            )
        new_points = gap.competitive.score_instance(result, rescored_field)
        published = baseline[result.tool]
        new_penalties = published["penalties"]
        if old_points == retro.PENALTY_INCORRECT:
            new_penalties -= 1
        if new_points == retro.PENALTY_INCORRECT:
            new_penalties += 1
        if new_penalties < 0:
            raise EvidenceError("rescored organizer penalty count became negative")
        rescored_total = published["points"] - old_points + new_points
        rescored_totals[result.tool] = rescored_total
        participants_payload[result.tool] = {
            **raw_rows[result.tool],
            "published_category": published,
            "published_instance_outcome": _score_outcome(old_points),
            "published_instance_points": old_points,
            "rescored_category_penalties": new_penalties,
            "rescored_category_points": rescored_total,
            "rescored_instance_outcome": _score_outcome(new_points),
            "rescored_instance_points": new_points,
            "score_delta": new_points - old_points,
        }

    candidate_points = gap.competitive.score_instance(candidate, rescored_field)
    if candidate_points != retro.POINTS_CORRECT:
        raise EvidenceError("strict NY witness did not earn organizer credit")
    published_denominator = max(value["points"] for value in baseline.values())
    rescored_official_denominator = max(rescored_totals.values())
    return {
        "schema": ORGANIZER_RESCORE_SCHEMA,
        "official_results_commit": OFFICIAL_RESULTS_COMMIT,
        "official_results_repository": repository_identity,
        "artifacts": artifact_identity,
        "category": category,
        "organizer_score_id": score_id,
        "occurrence": list(occurrence.score_key),
        "participant_order": list(participants),
        "participants": participants_payload,
        "truth": {
            "published": "holds",
            "rescored": "violated",
            "cause": "ny_strictly_correct_exact_2025_counterexample",
        },
        "candidate": {
            "counterexample_classification": "correct",
            "instance_outcome": "correct",
            "instance_points": candidate_points,
        },
        "denominator": {
            "published_official_points": published_denominator,
            "rescored_official_points": rescored_official_denominator,
            "candidate_instance_points": candidate_points,
        },
        "scoring_semantics": {
            "correct_points": retro.POINTS_CORRECT,
            "incorrect_penalty": retro.PENALTY_INCORRECT,
            "normalized_percent_floor": 0,
            "time_bonus_enabled": False,
        },
    }


def revalidate_organizer_rescore(
    official: PinnedOfficialResults,
    rescore: dict[str, Any],
) -> None:
    repository_identity = _organizer_results_repository_identity(official)
    _, identity = _load_organizer_rescore_artifacts(official)
    if (
        rescore.get("official_results_repository") != repository_identity
        or rescore.get("artifacts") != identity
    ):
        raise EvidenceError("organizer rescore artifacts changed during validation")


def canonical_row_key(
    category: str, occurrence: retro.OfficialInstanceOccurrence
) -> str:
    return json.dumps(
        [category, *occurrence.score_key],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def parse_row_key(row_key: object, *, path: Path) -> tuple[str, str, str, int]:
    try:
        identity = json.loads(row_key)
    except (TypeError, json.JSONDecodeError) as error:
        raise EvidenceError(
            f"evidence index contains an invalid row key: {path}"
        ) from error
    if (
        not isinstance(identity, list)
        or len(identity) != 4
        or not all(isinstance(value, str) for value in identity[:3])
        or type(identity[3]) is not int
        or identity[3] < 0
    ):
        raise EvidenceError(f"evidence index contains an invalid row identity: {path}")
    canonical = json.dumps(identity, ensure_ascii=True, separators=(",", ":"))
    if row_key != canonical:
        raise EvidenceError(f"evidence index row key is not canonical JSON: {path}")
    return identity[0], identity[1], identity[2], identity[3]


def _resolve_run_directory(root: Path, run_id: str) -> Path:
    if provenance.SAFE_COMPONENT.fullmatch(run_id) is None:
        raise EvidenceError(f"invalid run ID: {run_id!r}")
    candidate = root / "runs" / run_id
    if candidate.is_symlink():
        raise EvidenceError(f"run directory must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise EvidenceError(
            f"run directory is missing or escapes artifact root: {candidate}"
        ) from error
    if not resolved.is_dir():
        raise EvidenceError(f"run path is not a directory: {resolved}")
    return resolved


def _require_exact_keys(
    value: object,
    expected: frozenset[str] | set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise EvidenceError(
            f"{label} does not have the canonical required fields "
            f"(observed {observed!r})"
        )
    return value


def _validate_build_coherence(solver: dict[str, Any]) -> None:
    coherence = _require_exact_keys(
        solver.get("build_coherence"),
        BUILD_COHERENCE_KEYS,
        label="start solver build coherence",
    )
    fingerprint = solver.get("fingerprint")
    binary_mtime = coherence.get("binary_mtime_epoch")
    if (
        not isinstance(fingerprint, dict)
        or type(fingerprint.get("mtime_ns")) is not int
        or type(binary_mtime) is not int
        or binary_mtime < 0
        or binary_mtime != fingerprint["mtime_ns"] // 1_000_000_000
        or coherence.get("build_input_paths")
        != list(provenance._BUILD_INPUT_PATHS)
        or coherence.get("behaviour_input_paths")
        != list(provenance._BEHAVIOUR_INPUT_PATHS)
    ):
        raise EvidenceError("start solver build-coherence identity is invalid")
    for label in (
        "build_inputs_last_commit_epoch",
        "behaviour_inputs_last_commit_epoch",
    ):
        epoch = coherence.get(label)
        if (
            epoch is not None
            and (type(epoch) is not int or epoch < 0 or binary_mtime < epoch)
        ):
            raise EvidenceError(
                "start solver build-coherence epoch is invalid or stale"
            )


def _validate_flight_capture_fields(measurement: dict[str, Any]) -> None:
    result_file = measurement.get("result_file")
    flight_file = measurement.get("flight_record_file")
    if (
        not isinstance(result_file, str)
        or not isinstance(flight_file, str)
        or not Path(flight_file).is_absolute()
        or flight_file != f"{result_file}.flight.json"
        or measurement.get("flight_record_capture")
        != FLIGHT_RECORD_CAPTURE_POLICY
    ):
        raise EvidenceError(
            "start flight-record capture does not bind the adjacent result sidecar"
        )


def validate_start_schema_profile(start: object) -> str:
    """Accept only the three producer profiles that existed in chronological order."""

    value = _require_exact_keys(start, START_KEYS, label="start manifest")
    measurement = value.get("measurement")
    solver = value.get("solver_binary")
    if not isinstance(measurement, dict) or not isinstance(solver, dict):
        raise EvidenceError("start measurement or solver identity is missing")
    key_pair = (frozenset(measurement), frozenset(solver))
    profiles = {
        (
            LEGACY_MEASUREMENT_KEYS,
            LEGACY_SOLVER_BINARY_KEYS,
        ): LEGACY_START_PROFILE,
        (
            LEGACY_MEASUREMENT_KEYS,
            SOLVER_BINARY_KEYS,
        ): BUILD_COHERENCE_START_PROFILE,
        (
            MEASUREMENT_KEYS,
            SOLVER_BINARY_KEYS,
        ): FLIGHT_START_PROFILE,
    }
    profile = profiles.get(key_pair)
    if profile is None:
        raise EvidenceError(
            "start measurement/solver fields do not match a complete canonical "
            "legacy or current producer profile"
        )
    if profile != LEGACY_START_PROFILE:
        _validate_build_coherence(solver)
    if profile == FLIGHT_START_PROFILE:
        _validate_flight_capture_fields(measurement)
    return profile


def validate_metadata_schema_profile(metadata: object) -> str:
    """Return the closed metadata profile without accepting partial additions."""

    if not isinstance(metadata, dict):
        raise EvidenceError("measurement metadata is not an object")
    keys = frozenset(metadata)
    if keys == LEGACY_METADATA_KEYS:
        return LEGACY_START_PROFILE
    if keys == METADATA_KEYS:
        return FLIGHT_START_PROFILE
    raise EvidenceError(
        "measurement metadata does not match a complete canonical legacy or "
        "current producer profile"
    )


def _sorted_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sorted_json_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [_sorted_json_value(item) for item in value]
    return value


def _valid_v2_lever_receipt(value: object, *, ambient_env: object) -> bool:
    """Validate the direct receipt emitted during flight schema v2's final era."""

    if (
        not isinstance(value, dict)
        or set(value) != FLIGHT_V2_LEVER_RECEIPT_KEYS
        or value.get("schema") != FLIGHT_V2_LEVER_RECEIPT_SCHEMA
        or type(value.get("lever_count")) is not int
        or type(value.get("env_overridden")) is not int
        or value["lever_count"] <= 0
        or value["env_overridden"] < 0
        or not isinstance(value.get("levers"), list)
        or value["lever_count"] != len(value["levers"])
        or not isinstance(ambient_env, dict)
    ):
        return False

    names: set[str] = set()
    env_overridden = 0
    for entry in value["levers"]:
        if not isinstance(entry, dict):
            return False
        expected_keys = set(FLIGHT_V2_LEVER_ENTRY_KEYS)
        expected_keys.update(
            set(entry) & FLIGHT_V2_LEVER_ENTRY_OPTIONAL_KEYS
        )
        if set(entry) != expected_keys:
            return False
        name = entry.get("name")
        source = entry.get("source")
        bucket = entry.get("bucket")
        moat = entry.get("moat")
        lever_provenance = entry.get("provenance")
        if (
            not isinstance(name, str)
            or archive.LEVER_NAME.fullmatch(name) is None
            or name in names
            or not isinstance(source, str)
            or source not in FLIGHT_V2_LEVER_SOURCES
            or not isinstance(bucket, str)
            or bucket not in archive.LEVER_BUCKETS
            or not isinstance(moat, str)
            or moat not in archive.LEVER_MOATS
            or not isinstance(lever_provenance, str)
            or lever_provenance not in archive.LEVER_PROVENANCE
            or not archive._valid_lever_value(entry.get("value"))
            or (
                lever_provenance == "unmeasured"
                and bucket == "default_on"
            )
            or (lever_provenance == "guard" and bucket == "auto")
        ):
            return False
        names.add(name)

        rejected_present = "rejected_raw" in entry
        rejected_raw = entry.get("rejected_raw")
        if source == "env":
            if name not in ambient_env or rejected_present:
                return False
            env_overridden += 1
        elif rejected_present and (
            not isinstance(rejected_raw, str)
            or ambient_env.get(name) != rejected_raw
        ):
            return False
        elif not rejected_present and name in ambient_env:
            return False

    return value["env_overridden"] == env_overridden


def _serde_json_float(value: float) -> str:
    """Match serde_json's finite-f64 spelling, including its exponent cutoff."""

    if not math.isfinite(value):
        raise ValueError("serde_json flight value is non-finite")
    rendered = repr(value).lower()
    if "e" not in rendered:
        return rendered
    mantissa, exponent_text = rendered.split("e", 1)
    exponent = int(exponent_text)
    # Python changes to exponent notation at 1e-5; serde_json keeps fixed
    # notation through that decade and changes at 1e-6.
    if -5 <= exponent < 0:
        sign = ""
        if mantissa.startswith("-"):
            sign = "-"
            mantissa = mantissa[1:]
        integral, separator, fractional = mantissa.partition(".")
        digits = integral + (fractional if separator else "")
        point = len(integral) + exponent
        if point <= 0:
            return f"{sign}0.{('0' * -point)}{digits}"
        if point >= len(digits):
            return f"{sign}{digits}{('0' * (point - len(digits)))}.0"
        return f"{sign}{digits[:point]}.{digits[point:]}"
    exponent_sign = "+" if exponent >= 0 else "-"
    return f"{mantissa}e{exponent_sign}{abs(exponent)}"


def _serde_json_pretty(value: Any, *, indentation: int = 0) -> str:
    """Serialize the flight subset exactly like serde_json::to_string_pretty."""

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if type(value) is int:
        return str(value)
    if type(value) is float:
        return _serde_json_float(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        inner = ",\n".join(
            f"{' ' * (indentation + 2)}"
            f"{_serde_json_pretty(item, indentation=indentation + 2)}"
            for item in value
        )
        return f"[\n{inner}\n{' ' * indentation}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        if not all(isinstance(key, str) for key in value):
            raise TypeError("serde_json flight object key is not a string")
        inner = ",\n".join(
            f"{' ' * (indentation + 2)}{json.dumps(key, ensure_ascii=False)}: "
            f"{_serde_json_pretty(item, indentation=indentation + 2)}"
            for key, item in value.items()
        )
        return f"{{\n{inner}\n{' ' * indentation}}}"
    raise TypeError(f"unsupported serde_json flight value: {type(value).__name__}")


def _flight_record_source_bytes(record: dict[str, Any]) -> bytes:
    """Recreate the exact serde_json pretty bytes embedded by the producer."""

    ordered: dict[str, Any] = {
        "schema_version": record["schema_version"],
        "backend_kind": record["backend_kind"],
        "backend_summary": record["backend_summary"],
    }
    host = record["host"]
    ordered["host"] = {
        key: host[key]
        for key in ("hostname", "cpu_model", "logical_cores", "ram_bytes")
    }
    for key in ("load_avg_at_begin", "load_avg_at_end"):
        if key in record:
            ordered[key] = record[key]
    ordered["category"] = record["category"]
    ordered["budget_secs"] = record["budget_secs"]
    ambient = record["ambient_env"]
    ordered["ambient_env"] = {key: ambient[key] for key in sorted(ambient)}
    if "levers" in record:
        levers = record["levers"]
        if record["schema_version"] == 2:
            ordered["levers"] = _sorted_json_value(levers)
        else:
            ordered_levers = {"status": levers["status"]}
            if "reason" in levers:
                ordered_levers["reason"] = levers["reason"]
            if "receipt" in levers:
                ordered_levers["receipt"] = _sorted_json_value(levers["receipt"])
            ordered["levers"] = ordered_levers
    events: list[dict[str, Any]] = []
    for event in record["events"]:
        ordered_event = {
            "method": event["method"],
            "status": event["status"],
        }
        if "reason" in event:
            ordered_event["reason"] = event["reason"]
        if "at_secs" in event:
            ordered_event["at_secs"] = event["at_secs"]
        events.append(ordered_event)
    ordered["events"] = events
    return _serde_json_pretty(ordered).encode("utf-8")


def _validate_flight_record(
    value: object,
    *,
    measurement: dict[str, Any],
    category: object,
    timeout_seconds: object,
    solver_verdict: object,
) -> None:
    if value == {"status": "missing"}:
        return
    capture = _require_exact_keys(
        value,
        FLIGHT_RECORD_CAPTURE_KEYS,
        label="metadata flight-record capture",
    )
    if capture.get("status") != "captured":
        raise EvidenceError("metadata flight-record capture status is unsupported")
    source_digest = capture.get("source_sha256")
    source_size = capture.get("size_bytes")
    record = capture.get("record")
    if (
        not _is_sha256(source_digest)
        or type(source_size) is not int
        or source_size <= 0
        or source_size > archive.MAX_FLIGHT_RECORD_BYTES
        or not isinstance(record, dict)
    ):
        raise EvidenceError("metadata flight-record capture identity is invalid")

    schema_version = record.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version not in archive.SUPPORTED_FLIGHT_SCHEMA_VERSIONS
    ):
        raise EvidenceError("embedded flight-record schema is unsupported")
    expected_record_keys = set(FLIGHT_RECORD_BASE_KEYS)
    expected_record_keys.update(set(record) & FLIGHT_RECORD_OPTIONAL_KEYS)
    if schema_version == archive.FLIGHT_SCHEMA_VERSION or (
        schema_version == 2 and "levers" in record
    ):
        expected_record_keys.add("levers")
    _require_exact_keys(
        record,
        expected_record_keys,
        label="embedded flight record",
    )

    backend_kind = record.get("backend_kind")
    backend_summary = record.get("backend_summary")
    host = _require_exact_keys(
        record.get("host"), FLIGHT_HOST_KEYS, label="embedded flight host"
    )
    if (
        not isinstance(backend_kind, str)
        or not backend_kind
        or not isinstance(backend_summary, str)
        or not backend_summary
        or not isinstance(host.get("hostname"), str)
        or not isinstance(host.get("cpu_model"), str)
        or type(host.get("logical_cores")) is not int
        or host["logical_cores"] < 0
        or type(host.get("ram_bytes")) is not int
        or host["ram_bytes"] < 0
        or record.get("category") != category
        or type(record.get("budget_secs")) is not int
        or record.get("budget_secs") != timeout_seconds
    ):
        raise EvidenceError("embedded flight-record row/backend identity differs")

    for label in ("load_avg_at_begin", "load_avg_at_end"):
        if label not in record:
            continue
        load = record[label]
        if (
            not isinstance(load, list)
            or len(load) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) < 0
                for item in load
            )
        ):
            raise EvidenceError(f"embedded flight-record {label} is invalid")

    environment = measurement.get("solver_environment")
    values = environment.get("values") if isinstance(environment, dict) else None
    ambient = record.get("ambient_env")
    if not isinstance(values, dict) or not isinstance(ambient, dict):
        raise EvidenceError("embedded flight-record environment is missing")
    expected_ambient = {
        name: setting
        for name, setting in values.items()
        if isinstance(name, str)
        and isinstance(setting, str)
        and (name.startswith("NY_") or name == "OMP_NUM_THREADS")
    }
    if ambient != expected_ambient:
        raise EvidenceError(
            "embedded flight-record environment differs from sealed execution"
        )
    if schema_version == archive.FLIGHT_SCHEMA_VERSION:
        if not archive._valid_v3_lever_state(
            record.get("levers"), ambient_env=ambient
        ):
            raise EvidenceError("embedded flight-record lever receipt is invalid")
    elif "levers" in record and not _valid_v2_lever_receipt(
        record.get("levers"), ambient_env=ambient
    ):
        raise EvidenceError(
            "embedded flight-record v2 lever receipt is invalid"
        )

    raw_events = record.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise EvidenceError("embedded flight record has no events")
    previous_at = -1.0
    terminal_count = 0
    for index, event in enumerate(raw_events):
        if not isinstance(event, dict):
            raise EvidenceError("embedded flight-record event is not an object")
        expected_event_keys = set(FLIGHT_EVENT_KEYS)
        expected_event_keys.update(set(event) & FLIGHT_EVENT_OPTIONAL_KEYS)
        _require_exact_keys(
            event,
            expected_event_keys,
            label="embedded flight-record event",
        )
        method = event.get("method")
        status = event.get("status")
        reason = event.get("reason")
        at_secs = event.get("at_secs")
        if (
            not isinstance(method, str)
            or not method
            or status not in {"ran", "skipped", "not_reached", "complete"}
            or ("reason" in event and not isinstance(reason, str))
            or ("at_secs" in event and (
                isinstance(at_secs, bool)
                or not isinstance(at_secs, (int, float))
                or not math.isfinite(float(at_secs))
                or float(at_secs) < previous_at
            ))
            or ((method == "run_complete") != (status == "complete"))
        ):
            raise EvidenceError("embedded flight-record event is invalid")
        if "at_secs" in event:
            previous_at = float(at_secs)
        if status == "complete":
            terminal_count += 1
            if (
                index != len(raw_events) - 1
                or reason != solver_verdict
                or "at_secs" not in event
            ):
                raise EvidenceError(
                    "embedded flight-record terminal verdict/order differs"
                )
    if terminal_count != 1:
        raise EvidenceError("embedded flight record has no unique terminal event")

    try:
        source = _flight_record_source_bytes(record)
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(
            "embedded flight record cannot be serialized canonically"
        ) from error
    if len(source) != source_size or sha256(source) != source_digest:
        raise EvidenceError(
            "embedded flight record differs from its captured source bytes"
        )


def validate_flight_record_binding(
    *, start: object, metadata: object
) -> str:
    """Bind the current flight envelope to the sealed start and row metadata."""

    start_profile = validate_start_schema_profile(start)
    metadata_profile = validate_metadata_schema_profile(metadata)
    expected_metadata_profile = (
        FLIGHT_START_PROFILE
        if start_profile == FLIGHT_START_PROFILE
        else LEGACY_START_PROFILE
    )
    if metadata_profile != expected_metadata_profile:
        raise EvidenceError(
            "start and metadata producer profiles are inconsistent"
        )
    if start_profile != FLIGHT_START_PROFILE:
        return start_profile
    assert isinstance(start, dict)
    assert isinstance(metadata, dict)
    measurement = start["measurement"]
    assert isinstance(measurement, dict)
    _validate_flight_record(
        metadata.get("flight_record"),
        measurement=measurement,
        category=metadata.get("category"),
        timeout_seconds=metadata.get("timeout_seconds"),
        solver_verdict=metadata.get("solver_verdict"),
    )
    return start_profile


def _rehash_absolute_file(
    identity: object,
    *,
    path_key: str,
    label: str,
    expected_root: Path | None = None,
) -> tuple[Path, bytes]:
    if not isinstance(identity, dict):
        raise EvidenceError(f"{label} identity is missing")
    path_value = identity.get(path_key)
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not _is_sha256(identity.get("sha256"))
        or type(identity.get("size_bytes")) is not int
        or identity["size_bytes"] < 0
    ):
        raise EvidenceError(f"{label} identity is incomplete")
    path = resolved_regular_file(Path(path_value), label)
    if str(path) != path_value:
        raise EvidenceError(f"{label} path is not canonical")
    if expected_root is not None:
        try:
            path.relative_to(expected_root)
        except ValueError as error:
            raise EvidenceError(f"{label} escapes its required root") from error
    data = stable_bytes(path, label)
    if identity.get("sha256") != sha256(data) or identity.get("size_bytes") != len(
        data
    ):
        raise EvidenceError(f"{label} bytes differ from their start identity")
    fingerprint = identity.get("fingerprint")
    if isinstance(fingerprint, dict):
        try:
            observed = provenance._file_fingerprint(path)
        except OSError as error:
            raise EvidenceError(f"could not stat {label}: {path}") from error
        if fingerprint != observed:
            raise EvidenceError(f"{label} fingerprint differs from its start identity")
    return path, data


def _git_tree_entries(repository: Path, commit: str) -> dict[str, tuple[str, str]]:
    data = _git(
        repository,
        "ls-tree",
        "-rz",
        "--full-tree",
        commit,
    )
    assert data is not None
    entries: dict[str, tuple[str, str]] = {}
    for raw in data.split(b"\0"):
        if not raw:
            continue
        try:
            header, path_data = raw.split(b"\t", 1)
            mode, object_type, object_id = header.decode("ascii").split(" ", 2)
            path = path_data.decode("utf-8", "strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise EvidenceError("source Git tree inventory is malformed") from error
        if object_type != "blob" or mode not in {"100644", "100755", "120000"}:
            raise EvidenceError(
                f"source snapshot contains unsupported Git object {object_type} {path}"
            )
        if path in entries or not _safe_index_artifact(path):
            raise EvidenceError(f"source Git tree contains unsafe path: {path!r}")
        entries[path] = (mode, object_id)
    return entries


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - matching Git's pinned SHA-1 object format
        f"blob {len(data)}\0".encode("ascii") + data
    ).hexdigest()


def _validate_tar_end_padding(payload: bytes) -> None:
    """Reject data hidden after the first canonical tar end marker."""

    block_size = 512
    offset = 0
    zero = b"\0" * block_size
    while offset + block_size <= len(payload):
        header = payload[offset : offset + block_size]
        if header == zero:
            if payload[offset + block_size : offset + 2 * block_size] != zero:
                raise EvidenceError("source snapshot tar has no two-block end marker")
            if any(payload[offset + 2 * block_size :]):
                raise EvidenceError(
                    "source snapshot tar contains nonzero data after its end marker"
                )
            return
        raw_size = header[124:136]
        if raw_size[:1] and raw_size[0] & 0x80:
            raise EvidenceError(
                "source snapshot tar uses unsupported base-256 member size"
            )
        try:
            size_text = raw_size.rstrip(b"\0 ").lstrip(b" ")
            size = int(size_text or b"0", 8)
        except ValueError as error:
            raise EvidenceError(
                "source snapshot tar has an invalid member size"
            ) from error
        offset += block_size + ((size + block_size - 1) // block_size) * block_size
        if offset > len(payload):
            raise EvidenceError("source snapshot tar member is truncated")
    raise EvidenceError("source snapshot tar has no canonical end marker")


def _fits_ustar_path_field(value: str) -> bool:
    encoded = value.encode("utf-8", "strict")
    if len(encoded) <= 100:
        return True
    for index, byte in enumerate(encoded):
        if byte != ord("/"):
            continue
        if index <= 155 and len(encoded) - index - 1 <= 100:
            return True
    return False


def _canonical_source_member_pax(
    member: tarfile.TarInfo,
    *,
    commit: str,
) -> bool:
    headers = dict(member.pax_headers)
    if headers.pop("comment", None) != commit:
        return False

    path_override = headers.pop("path", None)
    path_fits = _fits_ustar_path_field(member.name)
    if path_override is None:
        if not path_fits:
            return False
    elif path_override != member.name or path_fits:
        # A PAX path is canonical only when ustar cannot represent the exact
        # UTF-8 member name and the override is byte-for-byte equivalent.
        return False

    link_override = headers.pop("linkpath", None)
    if member.issym():
        link_fits = len(member.linkname.encode("utf-8", "strict")) <= 100
        if link_override is None:
            if not link_fits:
                return False
        elif link_override != member.linkname or link_fits:
            return False
    elif link_override is not None:
        return False
    return not headers


def _validate_source_archive(
    *,
    archive: Path,
    repository: Path,
    commit: str,
) -> tuple[str, int]:
    archive_data = stable_bytes(archive, "source snapshot archive")
    # ``tarfile`` accepts concatenated/trailing gzip data.  That makes an
    # otherwise member-equivalent archive non-canonical, so reject it before
    # parsing the tar stream.
    archive_payload = _strict_decompress(
        archive_data,
        compression="gzip",
        label="source snapshot archive",
    )
    _validate_tar_end_padding(archive_payload)
    expected = _git_tree_entries(repository, commit)
    expected_directories = {
        parent.as_posix()
        for path in expected
        for parent in PurePosixPath(path).parents
        if parent != PurePosixPath(".")
    }
    observed: dict[str, tuple[str, str]] = {}
    observed_directories: set[str] = set()
    observed_names: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as source:
            expected_pax = {"comment": commit}
            if source.pax_headers != expected_pax:
                raise EvidenceError(
                    "source snapshot archive does not embed the exact commit ID"
                )
            for member in source:
                name = member.name
                if (
                    not _safe_source_archive_path(name)
                    or name in observed_names
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                    or not _canonical_source_member_pax(
                        member,
                        commit=commit,
                    )
                ):
                    raise EvidenceError(
                        f"source snapshot archive contains unsafe member: {name!r}"
                    )
                observed_names.add(name)
                if member.isdir():
                    observed_directories.add(name)
                    continue
                if not (member.isfile() or member.issym()):
                    raise EvidenceError(
                        f"source snapshot archive contains unsupported member: {name}"
                    )
                mode = (
                    "120000"
                    if member.issym()
                    else ("100755" if member.mode & 0o111 else "100644")
                )
                if member.issym():
                    payload = member.linkname.encode("utf-8")
                else:
                    stream = source.extractfile(member)
                    if stream is None:
                        raise EvidenceError(
                            f"source snapshot archive member is unreadable: {name}"
                        )
                    payload = stream.read()
                observed[name] = (mode, _git_blob_sha1(payload))
    except (OSError, tarfile.TarError, UnicodeEncodeError) as error:
        raise EvidenceError("source snapshot archive is invalid") from error
    if observed != expected:
        missing = sorted(set(expected) - set(observed))[:3]
        extra = sorted(set(observed) - set(expected))[:3]
        changed = sorted(
            path
            for path in set(expected) & set(observed)
            if expected[path] != observed[path]
        )[:3]
        raise EvidenceError(
            "source snapshot archive differs from the exact Git commit "
            f"(missing={missing}, extra={extra}, changed={changed})"
        )
    if observed_directories != expected_directories:
        missing = sorted(expected_directories - observed_directories)[:3]
        extra = sorted(observed_directories - expected_directories)[:3]
        raise EvidenceError(
            "source snapshot archive directory inventory differs from the "
            f"exact Git tree (missing={missing}, extra={extra})"
        )
    if stable_bytes(archive, "source snapshot archive") != archive_data:
        raise EvidenceError("source snapshot archive changed during validation")
    return sha256(archive_data), len(archive_data)


def _validate_source_snapshot(
    *,
    start: dict[str, Any],
    exact_commit: str,
) -> dict[str, Any]:
    ny = start.get("ny")
    if not isinstance(ny, dict):
        raise EvidenceError("start manifest has no NY source identity")
    repo_value = ny.get("repo_root")
    if not isinstance(repo_value, str) or not Path(repo_value).is_absolute():
        raise EvidenceError("start NY repository root is missing or not absolute")
    repository = resolved_directory(Path(repo_value), "NY source snapshot repository")
    if str(repository) != repo_value:
        raise EvidenceError("start NY repository root is not canonical")
    commit = _git_text(repository, "rev-parse", "HEAD").strip()
    tree = _git_text(repository, "rev-parse", "HEAD^{tree}").strip()
    object_format = _git_text(repository, "rev-parse", "--show-object-format").strip()
    if commit != exact_commit or ny.get("commit") != exact_commit:
        raise EvidenceError("source snapshot repository is not at the exact commit")
    if object_format != "sha1" or not re_fullmatch_hex(tree, 40):
        raise EvidenceError("source snapshot Git object format/tree is unsupported")
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    assert status is not None
    if status:
        raise EvidenceError("source snapshot repository is no longer clean")
    # The historical adjacent ``.tar.gz`` predates the final commit and is not
    # a full Git archive.  Never silently accept it.  The explicitly retained
    # full-source rebinding is verified member-for-member and embeds the commit.
    archive = Path(f"{repository}.full-source.tar.gz")
    archive_path = resolved_regular_file(archive, "source snapshot archive")
    archive_digest, archive_size = _validate_source_archive(
        archive=archive_path,
        repository=repository,
        commit=commit,
    )
    return {
        "repository_root": str(repository),
        "commit": commit,
        "tree": tree,
        "archive": str(archive_path),
        "archive_sha256": archive_digest,
        "archive_size_bytes": archive_size,
    }


def _rehash_start_artifacts(
    *,
    root: Path,
    start: dict[str, Any],
    source_snapshot: dict[str, Any],
) -> None:
    repository = Path(source_snapshot["repository_root"])
    provenance_tools = start.get("provenance_tools")
    git_identity = (
        provenance_tools.get("git") if isinstance(provenance_tools, dict) else None
    )
    git_path, _ = _rehash_absolute_file(
        git_identity,
        path_key="resolved_path",
        label="captured Git executable",
    )
    if (
        git_path != PINNED_GIT_EXECUTABLE
        or not isinstance(git_identity, dict)
        or git_identity.get("sha256") != PINNED_GIT_SHA256
    ):
        raise EvidenceError(
            "start manifest did not bind the validator's pinned Git executable"
        )
    solver = start.get("solver_binary")
    if not isinstance(solver, dict):
        raise EvidenceError("start solver-binary identity is missing")
    solver_path, solver_data = _rehash_absolute_file(
        solver,
        path_key="path",
        label="solver binary",
        expected_root=repository,
    )
    sealed_solver = solver.get("sealed_execution")
    sealed_path, sealed_data = _rehash_absolute_file(
        sealed_solver,
        path_key="path",
        label="sealed solver binary",
        expected_root=root,
    )
    if solver_data != sealed_data:
        raise EvidenceError("sealed solver binary differs from its source binary")
    if not solver_path.is_file() or not sealed_path.is_file():
        raise EvidenceError("solver execution files are unavailable")

    measurement = start.get("measurement")
    if not isinstance(measurement, dict):
        raise EvidenceError("start measurement identity is missing")
    for name in ("config_inputs", "sealed_config_inputs"):
        expected = measurement.get(name)
        if not isinstance(expected, dict):
            raise EvidenceError(f"start {name} identity is missing")
        declared = expected.get("declared_path")
        if not isinstance(declared, str):
            raise EvidenceError(f"start {name} path is missing")
        if name == "sealed_config_inputs":
            try:
                Path(declared).resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as error:
                raise EvidenceError(
                    "sealed config inputs escape artifact root"
                ) from error
        try:
            observed = provenance._capture_config_inputs(Path(declared))
        except (OSError, provenance.ProvenanceError) as error:
            raise EvidenceError(f"could not rehash {name}: {error}") from error
        if observed != expected:
            raise EvidenceError(f"{name} bytes differ from their start identity")

    dependencies = start.get("dependencies")
    ay = dependencies.get("ay") if isinstance(dependencies, dict) else None
    if not isinstance(ay, dict):
        raise EvidenceError("start AY dependency identity is missing")
    executable = ay.get("executable")
    sealed = ay.get("sealed_executable")
    if not isinstance(executable, dict) or not isinstance(sealed, dict):
        raise EvidenceError("start AY executable/seal identity is missing")
    _, original_data = _rehash_absolute_file(
        executable,
        path_key="resolved_path",
        label="AY executable",
    )
    _, sealed_data = _rehash_absolute_file(
        sealed,
        path_key="path",
        label="sealed AY executable",
        expected_root=root,
    )
    if original_data != sealed_data:
        raise EvidenceError("sealed AY executable differs from its source")
    cuda = dependencies.get("cuda_runtime") if isinstance(dependencies, dict) else None
    try:
        cuda_root = provenance._validate_sealed_cuda_runtime(cuda, hash_files=True)
    except (OSError, provenance.ProvenanceError) as error:
        raise EvidenceError(
            f"sealed CUDA runtime validation failed: {error}"
        ) from error
    if cuda_root is not None:
        try:
            cuda_root.relative_to(root)
        except ValueError as error:
            raise EvidenceError("sealed CUDA runtime escapes artifact root") from error


def _validate_execution_binding(
    *,
    root: Path,
    start: dict[str, Any],
) -> None:
    measurement = start.get("measurement")
    solver = start.get("solver_binary")
    dependencies = start.get("dependencies")
    if not all(
        isinstance(value, dict) for value in (measurement, solver, dependencies)
    ):
        raise EvidenceError("start execution identities are missing")
    assert isinstance(measurement, dict)
    assert isinstance(solver, dict)
    assert isinstance(dependencies, dict)
    sealed_solver = solver.get("sealed_execution")
    sealed_configs = measurement.get("sealed_config_inputs")
    ay = dependencies.get("ay")
    if not all(
        isinstance(value, dict) for value in (sealed_solver, sealed_configs, ay)
    ):
        raise EvidenceError("start sealed execution identities are missing")
    assert isinstance(sealed_solver, dict)
    assert isinstance(sealed_configs, dict)
    assert isinstance(ay, dict)
    sealed_ay = ay.get("sealed_executable")
    if not isinstance(sealed_ay, dict):
        raise EvidenceError("start sealed AY execution identity is missing")
    result_file = measurement.get("result_file")
    expected_template = [
        sealed_solver.get("path"),
        "vnncomp",
        "v1",
        "<category>",
        "<onnx>",
        "<vnnlib>",
        result_file,
        "<capped_timeout_seconds>",
        "--configs-dir",
        sealed_configs.get("declared_path"),
    ]
    if (
        not isinstance(result_file, str)
        or measurement.get("solver_command_template") != expected_template
        or measurement.get("solver_output_capture")
        != "combined_stdout_stderr_exact_bytes"
    ):
        raise EvidenceError(
            "start solver command does not exactly bind the sealed solver, "
            "config, result path, and official placeholders"
        )
    for label in ("result_file", "solver_log_file"):
        value = measurement.get(label)
        scratch = measurement.get("scratch_dir")
        if not isinstance(value, str) or not isinstance(scratch, str):
            raise EvidenceError(f"start {label}/scratch identity is missing")
        try:
            Path(value).relative_to(Path(scratch))
        except ValueError as error:
            raise EvidenceError(
                f"start {label} escapes the scratch directory"
            ) from error
    environment = _require_exact_keys(
        measurement.get("solver_environment"),
        {"mode", "values"},
        label="start solver environment",
    )
    values = environment.get("values")
    overrides = measurement.get("solver_environment_overrides")
    unsets = measurement.get("solver_environment_unsets")
    if (
        environment.get("mode") != "env-i-reviewed-record-v1"
        or not isinstance(values, dict)
        or not isinstance(overrides, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in values.items()
        )
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in overrides.items()
        )
        or not isinstance(unsets, list)
        or unsets != sorted(set(unsets))
        or not all(isinstance(value, str) for value in unsets)
        or bool(set(values) & set(unsets))
        or any(values.get(key) != value for key, value in overrides.items())
    ):
        raise EvidenceError("start solver environment is not canonical")
    required_overrides = {
        "PATH": "/usr/bin:/bin",
        "RUST_LOG": "error",
        "NY_AY": str(sealed_ay.get("path")),
    }
    cuda = dependencies.get("cuda_runtime")
    if not isinstance(cuda, dict):
        raise EvidenceError("start CUDA runtime dependency is missing")
    if cuda.get("status") == "not_required":
        required_overrides["NY_NO_CUDA"] = "1"
    elif cuda.get("status") == "captured":
        sealed_cuda = cuda.get("sealed_execution")
        if not isinstance(sealed_cuda, dict) or not isinstance(
            sealed_cuda.get("path"), str
        ):
            raise EvidenceError("captured CUDA runtime seal is missing")
        required_overrides["LD_LIBRARY_PATH"] = str(sealed_cuda["path"])
    else:
        raise EvidenceError("start CUDA runtime status is unsupported")
    if any(overrides.get(key) != value for key, value in required_overrides.items()):
        raise EvidenceError(
            "start solver environment does not bind sealed AY/CUDA execution"
        )
    try:
        Path(str(sealed_solver["path"])).resolve(strict=True).relative_to(root)
        Path(str(sealed_configs["declared_path"])).resolve(strict=True).relative_to(
            root
        )
        Path(str(sealed_ay["path"])).resolve(strict=True).relative_to(root)
        if cuda.get("status") == "captured":
            assert isinstance(sealed_cuda, dict)
            Path(str(sealed_cuda["path"])).resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        raise EvidenceError("sealed execution path escapes artifact root") from error


def _load_occurrence(
    *,
    category: str,
    instance_index: int,
    benchmark: PinnedOfficialBenchmark,
    official: PinnedOfficialResults,
) -> tuple[retro.OfficialInstanceOccurrence, dict[str, Any]]:
    if category not in retro.REGULAR:
        raise EvidenceError(f"category is not in the 2025 regular track: {category}")
    if type(instance_index) is not int or instance_index <= 0:
        raise EvidenceError("instance index must be a positive one-based integer")
    instances_git_path = f"benchmarks/{category}/instances.csv"
    committed = _git_blob(benchmark, instances_git_path)
    if committed is None:
        raise EvidenceError(
            f"official instances.csv is absent from the pinned commit: {category}"
        )
    instances_blob, committed_data = committed
    caller_path = benchmark.benchmark_root / category / "instances.csv"
    caller_data = stable_bytes(caller_path, "official instances.csv")
    if caller_data != committed_data:
        raise EvidenceError(
            f"caller instances.csv bytes differ from the pinned commit: {category}"
        )
    try:
        occurrences = retro.load_official_instance_occurrences(
            benchmark.benchmark_root, category
        )
    except retro.MeasurementBudgetError as error:
        raise EvidenceError(str(error)) from error
    if instance_index > len(occurrences):
        raise EvidenceError(
            f"instance index {instance_index} exceeds the benchmark row count"
        )
    occurrence = occurrences[instance_index - 1]
    reference = official.context.reference_order.get(category, [])
    if instance_index > len(reference):
        raise EvidenceError(
            "benchmark occurrence has no same-position published result identity"
        )
    if reference[instance_index - 1] != occurrence.score_key:
        raise EvidenceError(
            "benchmark instance and published result occurrence identities differ"
        )
    if reference.count(occurrence.score_key) != 1:
        raise EvidenceError("published canonical occurrence is missing or ambiguous")
    if occurrence.score_key not in official.context.ground_truth.get(category, {}):
        raise EvidenceError(
            "regular-bank promotion is supported only when the pinned published "
            "truth is holds or violated; published ZERO-TOL truth is unavailable "
            "for the occurrence"
        )
    row = _parse_csv(committed_data, path=caller_path)[instance_index - 1]
    if (
        len(row) != 3
        or row[0] != occurrence.onnx
        or row[1] != occurrence.vnnlib
        or _decimal(
            row[2],
            label=f"official timeout {category}:{instance_index}",
            positive=True,
        )
        != occurrence.timeout_seconds
    ):
        raise EvidenceError(
            "parsed benchmark occurrence differs from its exact committed CSV row"
        )
    pair_occurrence = sum(
        1
        for previous in _parse_csv(committed_data, path=caller_path)[
            : instance_index - 1
        ]
        if len(previous) >= 2
        and (previous[0], previous[1]) == (occurrence.onnx, occurrence.vnnlib)
    )
    if pair_occurrence != occurrence.pair_occurrence:
        raise EvidenceError("committed raw-pair occurrence identity differs")
    binding = {
        "instance_index": instance_index,
        "instances_csv": str(caller_path.resolve(strict=True)),
        "instances_csv_git_path": instances_git_path,
        "instances_csv_git_blob": instances_blob,
        "instances_csv_sha256": sha256(committed_data),
        "official_timeout_seconds": str(occurrence.timeout_seconds),
        "onnx": occurrence.onnx,
        "pair_occurrence": occurrence.pair_occurrence,
        "row_sha256": provenance._identity_sha256(row),
        "vnnlib": occurrence.vnnlib,
    }
    return occurrence, binding


def _load_start(
    *,
    root: Path,
    run_id: str,
    category: str,
    instance_index: int,
    benchmark: PinnedOfficialBenchmark,
    exact_commit: str,
) -> tuple[Path, dict[str, Any], str, int, dict[str, Any]]:
    if gap.EXACT_COMMIT_RE.fullmatch(exact_commit) is None:
        raise EvidenceError("exact commit must be exactly 40 lowercase hex digits")
    run_dir = _resolve_run_directory(root, run_id)
    start_path = run_dir / "start.json"
    try:
        start, digest, size = gap._validate_start(start_path, root, exact_commit)
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error
    validate_start_schema_profile(start)
    measurement = start.get("measurement")
    benchmark_identity = start.get("benchmark")
    ny = start.get("ny")
    _require_exact_keys(ny, NY_WORKTREE_KEYS, label="start NY worktree")
    _require_exact_keys(
        benchmark_identity,
        BENCHMARK_WORKTREE_KEYS,
        label="start benchmark worktree",
    )
    if not isinstance(measurement, dict) or not isinstance(benchmark_identity, dict):
        raise EvidenceError("start manifest lacks measurement or benchmark identity")
    _validate_execution_binding(root=root, start=start)
    if measurement.get("instance_index") != instance_index:
        raise EvidenceError(
            "requested instance index differs from the start manifest selection"
        )
    categories = measurement.get("categories")
    if not isinstance(categories, list) or categories != [category]:
        raise EvidenceError(
            "start manifest must select exactly the requested regular category"
        )
    for owner, value in (
        ("measurement", measurement.get("benchmark_root")),
        ("benchmark", benchmark_identity.get("benchmark_root")),
    ):
        if not isinstance(value, str):
            raise EvidenceError(f"start manifest {owner} benchmark root is missing")
        try:
            declared = Path(value).resolve(strict=True)
        except OSError as error:
            raise EvidenceError(
                f"start manifest {owner} benchmark root is unavailable: {value}"
            ) from error
        if declared != benchmark.benchmark_root:
            raise EvidenceError(
                f"caller benchmark root differs from start {owner} identity"
            )
    source_snapshot = _validate_source_snapshot(start=start, exact_commit=exact_commit)
    _rehash_start_artifacts(
        root=root,
        start=start,
        source_snapshot=source_snapshot,
    )
    try:
        _, final_digest, final_size = gap._stable_json(start_path, "start manifest")
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error
    if (final_digest, final_size) != (digest, size):
        raise EvidenceError("start manifest changed while artifacts were rehashed")
    return start_path, start, digest, size, source_snapshot


def _start_containment_profile(start: dict[str, Any]) -> str | None:
    """Reopen the optional profile selector from its canonical start field.

    Historical manifests predate named containment profiles.  They remain
    readable, but a present selector is always validated against the same
    closed set accepted by the measurement harness.
    """

    host = start.get("host")
    containment = host.get("containment") if isinstance(host, dict) else None
    if not isinstance(containment, dict):
        raise EvidenceError("start containment identity is missing")
    profile = containment.get("containment_profile")
    if profile is None:
        return None
    if (
        not isinstance(profile, str)
        or profile not in provenance.ALLOWED_CONTAINMENT_PROFILES
    ):
        raise EvidenceError("start containment profile is unsupported")
    return profile


def _validate_completion_check_bindings(
    *,
    start: dict[str, Any],
    checks: dict[str, Any],
) -> None:
    measurement = start.get("measurement")
    dependencies = start.get("dependencies")
    provenance_tools = start.get("provenance_tools")
    host = start.get("host")
    if not all(
        isinstance(value, dict)
        for value in (measurement, dependencies, provenance_tools, host)
    ):
        raise EvidenceError("start identities required by completion are missing")
    assert isinstance(measurement, dict)
    assert isinstance(dependencies, dict)
    assert isinstance(provenance_tools, dict)
    assert isinstance(host, dict)
    ay = dependencies.get("ay")
    if not isinstance(ay, dict):
        raise EvidenceError("start AY dependency identity is missing")

    identity_checks = {
        "benchmark": start.get("benchmark"),
        "config_inputs": measurement.get("config_inputs"),
        "containment": host.get("containment"),
        "cuda_runtime": dependencies.get("cuda_runtime"),
        "git_executable": provenance_tools.get("git"),
        "git_executable_post": provenance_tools.get("git"),
        "ny_worktree": start.get("ny"),
        "rust_toolchain": start.get("rust_toolchain"),
        "sealed_config_inputs": measurement.get("sealed_config_inputs"),
    }
    for name, expected in identity_checks.items():
        if not isinstance(expected, dict) or not expected:
            raise EvidenceError(
                f"start {name} identity required by completion is empty"
            )
        check = _require_exact_keys(
            checks.get(name),
            {"expected_identity_sha256", "observed_identity_sha256", "status"},
            label=f"completion {name} check",
        )
        digest = provenance._identity_sha256(expected)
        if (
            check.get("status") != "valid"
            or check.get("expected_identity_sha256") != digest
            or check.get("observed_identity_sha256") != digest
        ):
            raise EvidenceError(
                f"completion {name} check does not bind its start identity"
            )

    solver = start.get("solver_binary")
    if not isinstance(solver, dict):
        raise EvidenceError("start solver identity is missing")
    solver_check = _require_exact_keys(
        checks.get("solver_binary"),
        {
            "expected_fingerprint",
            "expected_sha256",
            "observed_fingerprint",
            "observed_sha256",
            "path",
            "resolved_path",
            "status",
        },
        label="completion solver-binary check",
    )
    if (
        solver_check.get("status") != "valid"
        or solver_check.get("path") != solver.get("path")
        or solver_check.get("resolved_path") != solver.get("path")
        or solver_check.get("expected_sha256") != solver.get("sha256")
        or solver_check.get("observed_sha256") != solver.get("sha256")
        or solver_check.get("expected_fingerprint") != solver.get("fingerprint")
        or solver_check.get("observed_fingerprint") != solver.get("fingerprint")
    ):
        raise EvidenceError("completion solver-binary check differs from start")

    sealed_checks = {
        "sealed_solver_binary": solver.get("sealed_execution"),
        "sealed_ay_executable": ay.get("sealed_executable"),
    }
    for name, expected in sealed_checks.items():
        if not isinstance(expected, dict):
            raise EvidenceError(f"start {name} identity is missing")
        check = _require_exact_keys(
            checks.get(name),
            {
                "expected_fingerprint",
                "expected_sha256",
                "observed_fingerprint",
                "observed_sha256",
                "path",
                "status",
            },
            label=f"completion {name} check",
        )
        if (
            check.get("status") != "valid"
            or check.get("path") != expected.get("path")
            or check.get("expected_sha256") != expected.get("sha256")
            or check.get("observed_sha256") != expected.get("sha256")
            or check.get("expected_fingerprint") != expected.get("fingerprint")
            or check.get("observed_fingerprint") != expected.get("fingerprint")
        ):
            raise EvidenceError(f"completion {name} check differs from start")

    ay_executable = ay.get("executable")
    if not isinstance(ay_executable, dict):
        raise EvidenceError("start AY executable identity is missing")
    ay_check = _require_exact_keys(
        checks.get("ay_executable"),
        {
            "expected_identity_sha256",
            "observed_identity_sha256",
            "resolved_path",
            "status",
        },
        label="completion AY-executable check",
    )
    ay_digest = provenance._identity_sha256(ay_executable)
    if (
        ay_check.get("status") != "valid"
        or ay_check.get("expected_identity_sha256") != ay_digest
        or ay_check.get("observed_identity_sha256") != ay_digest
        or ay_check.get("resolved_path") != ay_executable.get("resolved_path")
    ):
        raise EvidenceError("completion AY-executable check differs from start")


def _load_completion(
    *,
    root: Path,
    start_path: Path,
    start: dict[str, Any],
    start_digest: str,
    start_size: int,
    official: PinnedOfficialResults,
    benchmark: PinnedOfficialBenchmark,
    replay_session: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], str, int, gap.SealedRecord, dict[str, Any]]:
    completion_path = start_path.with_name("completion.json")
    try:
        completion, completion_digest, completion_size = gap._stable_json(
            completion_path, "completion manifest"
        )
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error
    _require_exact_keys(completion, COMPLETION_KEYS, label="completion manifest")
    integrity = completion.get("integrity")
    _require_exact_keys(
        integrity,
        {"schema", "status", "violations", "checks"},
        label="completion integrity",
    )
    checks = integrity.get("checks") if isinstance(integrity, dict) else None
    if (
        completion.get("schema") != "ny_measurement_completion_v1"
        or completion.get("run_id") != start.get("run_id")
        or completion.get("start_manifest") != "start.json"
        or completion.get("start_manifest_sha256") != start_digest
        or completion.get("exit_status") != 0
        or completion.get("completed_successfully") is not True
        or not isinstance(integrity, dict)
        or integrity.get("schema") != "ny_measurement_completion_integrity_v1"
        or integrity.get("status") != "valid"
        or integrity.get("violations") != []
        or not isinstance(checks, dict)
        or not checks
    ):
        raise EvidenceError(
            "completion is not a successfully completed, violation-free manifest"
        )
    if set(checks) != COMPLETION_CHECKS:
        raise EvidenceError(
            "completion integrity checks are not the exact canonical check set"
        )
    _validate_completion_check_bindings(start=start, checks=checks)
    invalid_checks = sorted(
        str(name)
        for name, check in checks.items()
        if not isinstance(name, str)
        or not isinstance(check, dict)
        or check.get("status") != "valid"
    )
    if invalid_checks:
        raise EvidenceError(
            "completion contains non-valid integrity checks: "
            + ", ".join(invalid_checks)
        )
    try:
        records = gap._validate_completion(
            root=root,
            start_path=start_path,
            start=start,
            start_digest=start_digest,
            start_size=start_size,
            official=official.context,
            official_evidence=official,
            benchmark_evidence=benchmark,
            replay_session=replay_session,
        )
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error
    if len(records) != 1:
        raise EvidenceError(
            "completion must contain exactly one run-evidence record, "
            f"found {len(records)}"
        )
    run_evidence = checks.get("run_evidence")
    assert isinstance(run_evidence, dict)
    _require_exact_keys(
        run_evidence, RUN_EVIDENCE_KEYS, label="completion run evidence"
    )
    raw_records = run_evidence.get("records")
    if (
        not isinstance(raw_records, list)
        or len(raw_records) != 1
        or not isinstance(raw_records[0], dict)
    ):
        raise EvidenceError("completion must retain exactly one raw evidence record")
    if run_evidence.get("produced_rows") is not True:
        raise EvidenceError("completion does not declare a produced result row")
    _require_exact_keys(
        raw_records[0], RUN_RECORD_KEYS, label="completion run-evidence record"
    )
    if run_evidence.get("csv_evidence_sha256") != provenance._identity_sha256(
        run_evidence.get("csv_evidence")
    ):
        raise EvidenceError("completion CSV-evidence digest is invalid")
    cache_binding = _require_exact_keys(
        completion.get("input_hash_cache"),
        {"artifact", "entry_count", "present", "sha256", "size_bytes"},
        label="completion input-hash-cache binding",
    )
    if cache_binding.get("present") is not True:
        raise EvidenceError("completion input-hash cache is not present")
    if cache_binding.get("artifact") != "input_hash_cache.json":
        raise EvidenceError("completion input-hash-cache artifact name is invalid")
    cache_path = resolved_regular_file(
        start_path.with_name("input_hash_cache.json"), "input hash cache"
    )
    if cache_path.parent != start_path.parent:
        raise EvidenceError("input hash cache escapes the run directory")
    cache_data = stable_bytes(cache_path, "input hash cache")
    cache_digest = sha256(cache_data)
    cache_size = len(cache_data)
    cache = _json_object(
        cache_data,
        path=cache_path,
        label="input hash cache",
    )
    _require_exact_keys(cache, INPUT_HASH_CACHE_KEYS, label="input hash cache")
    if (
        cache.get("schema") != "ny_measurement_input_hash_cache_v1"
        or cache.get("run_id") != start.get("run_id")
        or cache.get("start_manifest_sha256") != start_digest
        or cache_binding.get("sha256") != cache_digest
        or cache_binding.get("size_bytes") != cache_size
    ):
        raise EvidenceError("input hash cache identity differs from completion")
    entries = cache.get("entries")
    if not isinstance(entries, dict) or cache_binding.get("entry_count") != len(
        entries
    ):
        raise EvidenceError("input hash cache entries are missing or miscounted")
    rehashed_entries: list[dict[str, Any]] = []
    for cache_key, cache_entry in entries.items():
        if not _is_sha256(cache_key):
            raise EvidenceError("input hash cache key is invalid")
        _require_exact_keys(
            cache_entry,
            {"fingerprint", "path", "sha256"},
            label="input hash cache entry",
        )
        if not isinstance(cache_entry, dict):
            raise EvidenceError("input hash cache entry is invalid")
        path_value = cache_entry.get("path")
        fingerprint = cache_entry.get("fingerprint")
        if (
            not isinstance(path_value, str)
            or not isinstance(fingerprint, dict)
            or set(fingerprint)
            != {"device", "inode", "size_bytes", "mtime_ns", "ctime_ns"}
            or not _is_sha256(cache_entry.get("sha256"))
        ):
            raise EvidenceError("input hash cache entry identity is incomplete")
        if provenance._input_cache_key(path_value, fingerprint) != cache_key:
            raise EvidenceError("input hash cache key is not content-addressed")
        cache_file, cache_bytes = _rehash_absolute_file(
            {
                "path": path_value,
                "sha256": cache_entry["sha256"],
                "size_bytes": fingerprint["size_bytes"],
                "fingerprint": fingerprint,
            },
            path_key="path",
            label="cached benchmark input",
        )
        if sha256(cache_bytes) != cache_entry["sha256"]:
            raise EvidenceError("input hash cache entry digest differs")
        rehashed_entries.append(
            {
                "key": cache_key,
                "path": str(cache_file),
                "sha256": sha256(cache_bytes),
                "size_bytes": len(cache_bytes),
            }
        )
    referenced_keys = raw_records[0].get("input_hash_cache_keys")
    if (
        not isinstance(referenced_keys, list)
        or len(referenced_keys) != len(set(referenced_keys))
        or set(referenced_keys) != set(entries)
        or run_evidence.get("input_hash_cache_entry_count") != len(entries)
        or run_evidence.get("referenced_input_hash_cache_entry_count")
        != len(referenced_keys)
    ):
        raise EvidenceError(
            "run evidence does not bind the exact input-hash-cache entries"
        )
    cache_check = _require_exact_keys(
        checks.get("input_hash_cache"),
        {
            "entries_sha256",
            "entry_count",
            "referenced_entry_count",
            "rehashed_entry_count",
            "sha256",
            "status",
        },
        label="completion input-hash-cache check",
    )
    rehashed_entries.sort(key=lambda value: str(value["key"]))
    if (
        cache_check.get("status") != "valid"
        or cache_check.get("sha256") != cache_digest
        or cache_check.get("entry_count") != len(entries)
        or cache_check.get("referenced_entry_count") != len(referenced_keys)
        or cache_check.get("rehashed_entry_count") != len(entries)
        or cache_check.get("entries_sha256")
        != provenance._identity_sha256(rehashed_entries)
    ):
        raise EvidenceError(
            "completion input-hash-cache check differs from rehashed entries"
        )
    try:
        _, final_digest, final_size = gap._stable_json(
            completion_path, "completion manifest"
        )
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error
    if (final_digest, final_size) != (completion_digest, completion_size):
        raise EvidenceError("completion changed while it was being validated")
    return (
        completion_path,
        completion,
        completion_digest,
        completion_size,
        records[0],
        raw_records[0],
    )


def _metadata_object(
    root: Path, raw_record: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    try:
        metadata_path, metadata_data, _, _ = gap._checked_artifact(
            root, raw_record.get("metadata"), "metadata"
        )
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error
    return metadata_path, _json_object(
        metadata_data, path=metadata_path, label="measurement metadata"
    )


def _validate_row_artifacts(
    *,
    root: Path,
    start_path: Path,
    start_digest: str,
    start: dict[str, Any],
    benchmark: PinnedOfficialBenchmark,
    occurrence: retro.OfficialInstanceOccurrence,
    raw_record: dict[str, Any],
    metadata_path: Path,
    metadata: dict[str, Any],
    authoritative_cache: dict[str, tuple[AuthoritativeInput, bytes]] | None = None,
) -> dict[str, AuthoritativeInput]:
    validate_flight_record_binding(start=start, metadata=metadata)
    if (
        metadata.get("schema") != "ny_measurement_result_v2"
        or metadata.get("schema_version") != 2
    ):
        raise EvidenceError("measurement metadata schema is not canonical v2")
    try:
        preflight_path, preflight_data, preflight_digest, preflight_size = (
            gap._checked_artifact(root, raw_record.get("preflight"), "input preflight")
        )
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error
    preflight = _json_object(
        preflight_data, path=preflight_path, label="input preflight"
    )
    _require_exact_keys(preflight, PREFLIGHT_KEYS, label="input preflight")
    expected_start_artifact = start_path.relative_to(root).as_posix()
    if (
        preflight.get("schema") != "ny_measurement_input_preflight_v1"
        or preflight.get("run_id") != start.get("run_id")
        or preflight.get("category") != occurrence.category
        or preflight.get("instance_index") != occurrence.instance_index
        or preflight.get("start_manifest") != expected_start_artifact
        or preflight.get("start_manifest_sha256") != start_digest
    ):
        raise EvidenceError("input preflight identity differs from the sealed run")
    metadata_preflight = _require_exact_keys(
        metadata.get("input_preflight"),
        {"artifact", "schema", "sha256"},
        label="metadata input-preflight link",
    )
    if (
        metadata_preflight.get("schema") != "ny_measurement_input_preflight_v1"
        or metadata_preflight.get("artifact")
        != preflight_path.relative_to(root).as_posix()
        or metadata_preflight.get("sha256") != preflight_digest
    ):
        raise EvidenceError("metadata input-preflight link differs")
    raw_preflight = raw_record.get("preflight")
    if (
        not isinstance(raw_preflight, dict)
        or raw_preflight.get("sha256") != preflight_digest
        or raw_preflight.get("size_bytes") != preflight_size
    ):
        raise EvidenceError("run-evidence preflight binding differs")

    inputs = _require_exact_keys(
        preflight.get("inputs"), {"onnx", "vnnlib"}, label="preflight inputs"
    )
    execution_inputs = _require_exact_keys(
        metadata.get("execution_inputs"),
        {"onnx", "vnnlib"},
        label="metadata execution inputs",
    )
    raw_summary = raw_preflight.get("inputs")
    _require_exact_keys(
        raw_summary, {"onnx", "vnnlib"}, label="run-evidence preflight inputs"
    )
    authoritative: dict[str, AuthoritativeInput] = {}
    cache_keys: list[str] = []
    for label, declared in (
        ("onnx", occurrence.onnx),
        ("vnnlib", occurrence.vnnlib),
    ):
        authoritative_input, authoritative_data = authoritative_benchmark_input(
            benchmark=benchmark,
            category=occurrence.category,
            declared_name=declared,
            label=label,
            payload_cache=authoritative_cache,
        )
        authoritative[label] = authoritative_input
        metadata_original = _require_exact_keys(
            metadata.get(label),
            {
                "declared_path",
                "hash_cache_hit",
                "hash_cache_key",
                "resolved_path",
                "sha256",
                "size_bytes",
            },
            label=f"metadata {label}",
        )
        preflight_input = _require_exact_keys(
            inputs.get(label),
            {"declared_name", "original", "sealed"},
            label=f"preflight {label}",
        )
        original = _require_exact_keys(
            preflight_input.get("original"),
            {
                "declared_path",
                "fingerprint",
                "resolved_path",
                "sha256",
                "size_bytes",
            },
            label=f"preflight original {label}",
        )
        sealed = _require_exact_keys(
            preflight_input.get("sealed"),
            {
                "artifact",
                "fingerprint",
                "mode",
                "resolved_path",
                "sha256",
                "size_bytes",
            },
            label=f"preflight sealed {label}",
        )
        execution = _require_exact_keys(
            execution_inputs.get(label),
            {
                "artifact",
                "fingerprint",
                "mode",
                "resolved_path",
                "sha256",
                "size_bytes",
            },
            label=f"metadata sealed {label}",
        )
        summary = _require_exact_keys(
            raw_summary.get(label),
            {"original_sha256", "sealed_artifact", "sealed_sha256"},
            label=f"run-evidence preflight {label}",
        )
        normalized = _safe_benchmark_name(declared, label=label)
        expected_original = (
            (benchmark.benchmark_root / occurrence.category)
            .joinpath(*PurePosixPath(normalized).parts)
            .resolve(strict=True)
        )
        if (
            metadata_original.get("declared_path") != declared
            or preflight_input.get("declared_name") != declared
            or original.get("declared_path") != str(expected_original)
            or original.get("resolved_path") != str(expected_original)
            or metadata_original.get("resolved_path") != str(expected_original)
        ):
            raise EvidenceError(
                f"{label} raw declared/resolved identity differs from the "
                "committed benchmark row"
            )
        original_path, original_data = _rehash_absolute_file(
            original,
            path_key="resolved_path",
            label=f"original {label}",
        )
        sealed_path, sealed_data = _rehash_absolute_file(
            sealed,
            path_key="resolved_path",
            label=f"sealed {label}",
            expected_root=root,
        )
        if original_path != expected_original:
            raise EvidenceError(f"original {label} path differs from benchmark")
        if (
            original_data != authoritative_data
            or sealed_data != authoritative_data
            or metadata_original.get("sha256") != authoritative_input.sha256
            or metadata_original.get("size_bytes") != authoritative_input.size_bytes
            or original.get("sha256") != authoritative_input.sha256
            or original.get("size_bytes") != authoritative_input.size_bytes
            or sealed.get("sha256") != authoritative_input.sha256
            or sealed.get("size_bytes") != authoritative_input.size_bytes
            or sealed.get("mode") != "read_only"
            or sealed.get("artifact") != sealed_path.relative_to(root).as_posix()
            or execution != sealed
            or summary
            != {
                "original_sha256": authoritative_input.sha256,
                "sealed_artifact": sealed_path.relative_to(root).as_posix(),
                "sealed_sha256": authoritative_input.sha256,
            }
        ):
            raise EvidenceError(
                f"{label} input bytes do not match the authoritative pinned payload"
            )
        cache_key = metadata_original.get("hash_cache_key")
        original_fingerprint = original.get("fingerprint")
        if (
            not _is_sha256(cache_key)
            or not isinstance(metadata_original.get("hash_cache_hit"), bool)
            or not isinstance(original_fingerprint, dict)
            or provenance._input_cache_key(
                str(expected_original),
                original_fingerprint,
            )
            != cache_key
        ):
            raise EvidenceError(f"metadata {label} input-hash-cache key is invalid")
        cache_keys.append(str(cache_key))

    if raw_record.get("input_hash_cache_keys") != cache_keys:
        raise EvidenceError("run-evidence input-hash-cache keys differ from metadata")
    expected_cache = (
        start_path.with_name("input_hash_cache.json").relative_to(root).as_posix()
    )
    if metadata.get("input_hash_cache") != expected_cache:
        raise EvidenceError("metadata input-hash-cache artifact differs")
    metadata_log = _require_exact_keys(
        metadata.get("solver_log"),
        {"artifact", "sha256", "size_bytes", "stream"},
        label="metadata solver-log binding",
    )
    raw_log = raw_record.get("solver_log")
    if (
        not isinstance(raw_log, dict)
        or metadata_log.get("stream") != "combined_stdout_stderr"
        or {key: metadata_log.get(key) for key in ("artifact", "sha256", "size_bytes")}
        != raw_log
    ):
        raise EvidenceError("metadata solver-log binding differs from completion")
    result_link = raw_record.get("result")
    if (
        not isinstance(result_link, dict)
        or metadata.get("result_artifact") != result_link.get("artifact")
        or metadata.get("result_sha256") != result_link.get("sha256")
        or metadata.get("raw_result_sha256") != result_link.get("sha256")
    ):
        raise EvidenceError("metadata raw-result binding differs from completion")
    counterexample = _require_exact_keys(
        metadata.get("counterexample_validation"),
        {"checker", "status"},
        label="metadata counterexample state",
    )
    verdict = raw_record.get("solver_verdict")
    expected_counterexample_status = (
        "not_checked" if verdict == "sat" else "not_applicable"
    )
    if (
        counterexample.get("checker") is not None
        or counterexample.get("status") != expected_counterexample_status
        or metadata.get("witness_present") is not (verdict == "sat")
    ):
        raise EvidenceError("metadata witness/counterexample state is inconsistent")
    measurement = start.get("measurement")
    assert isinstance(measurement, dict)
    config = measurement.get("config_inputs")
    sealed_config = measurement.get("sealed_config_inputs")
    expected_config = provenance._expected_metadata_config_identity(start)
    if (
        metadata.get("config_inputs") != expected_config
        or metadata.get("execution_config_inputs") != sealed_config
        or not isinstance(config, dict)
    ):
        raise EvidenceError("metadata config-input identities differ from start")
    if stable_bytes(preflight_path, "input preflight") != preflight_data:
        raise EvidenceError("input preflight changed during validation")
    return authoritative


def _validate_record_details(
    *,
    root: Path,
    run_id: str,
    category: str,
    instance_index: int,
    benchmark: PinnedOfficialBenchmark,
    start_path: Path,
    start_digest: str,
    start: dict[str, Any],
    occurrence: retro.OfficialInstanceOccurrence,
    sealed: gap.SealedRecord,
    raw_record: dict[str, Any],
    authoritative_cache: dict[str, tuple[AuthoritativeInput, bytes]] | None = None,
) -> tuple[str, str, Path, Path, dict[str, AuthoritativeInput]]:
    if (
        sealed.run_id != run_id
        or sealed.category != category
        or sealed.instance_index != instance_index
        or sealed.instance != occurrence.score_key
    ):
        raise EvidenceError(
            "sealed record identity differs from the requested occurrence"
        )
    verdict = sealed.verdict
    if verdict not in DECIDED_VERDICTS:
        raise EvidenceError(f"sealed verdict is not promotable: {verdict}")
    if (
        raw_record.get("category") != category
        or raw_record.get("instance_index") != instance_index
        or raw_record.get("onnx") != occurrence.onnx
        or raw_record.get("vnnlib") != occurrence.vnnlib
        or raw_record.get("solver_verdict") != verdict
        or raw_record.get("solver_exit_status") != 0
    ):
        raise EvidenceError(
            "run-evidence record does not match canonical identity/verdict"
        )
    metadata_path, metadata = _metadata_object(root, raw_record)
    authoritative_inputs = _validate_row_artifacts(
        root=root,
        start_path=start_path,
        start_digest=start_digest,
        start=start,
        benchmark=benchmark,
        occurrence=occurrence,
        raw_record=raw_record,
        metadata_path=metadata_path,
        metadata=metadata,
        authoritative_cache=authoritative_cache,
    )
    for label, expected in (("onnx", occurrence.onnx), ("vnnlib", occurrence.vnnlib)):
        value = metadata.get(label)
        if not isinstance(value, dict) or value.get("declared_path") != expected:
            raise EvidenceError(f"metadata {label} identity differs from benchmark")
    if (
        metadata.get("timeout_seconds") != raw_record.get("timeout_seconds")
        or metadata.get("elapsed_seconds") != raw_record.get("elapsed_seconds")
        or metadata.get("solver_exit_status") != 0
    ):
        raise EvidenceError("metadata timing/exit identity differs from run evidence")
    try:
        _, result_data, _, _ = gap._checked_artifact(
            root, raw_record.get("result"), "raw result"
        )
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error
    raw_lines = result_data.splitlines()
    try:
        raw_verdict = raw_lines[0].decode("utf-8").strip().lower()
    except (IndexError, UnicodeDecodeError) as error:
        raise EvidenceError("raw result has no UTF-8 verdict line") from error
    if raw_verdict != verdict:
        raise EvidenceError("raw result verdict differs from the sealed verdict")

    measurement = start.get("measurement")
    assert isinstance(measurement, dict)
    cap = _decimal(
        measurement.get("timeout_cap_seconds"),
        label="start timeout cap",
        positive=True,
    )
    official_timeout = occurrence.timeout_seconds
    effective_timeout = min(cap, official_timeout)
    recorded_timeout = _decimal(
        raw_record.get("timeout_seconds"),
        label="record timeout",
        positive=True,
    )
    elapsed = _decimal(raw_record.get("elapsed_seconds"), label="record elapsed time")
    if recorded_timeout != effective_timeout:
        raise EvidenceError(
            f"record timeout {recorded_timeout}s does not match the effective "
            f"official/capped timeout {effective_timeout}s"
        )
    if elapsed > recorded_timeout or elapsed > official_timeout:
        raise EvidenceError(
            f"refusing over-budget result at {elapsed}s; official budget is "
            f"{official_timeout}s"
        )

    expected_source = measurement.get("output_dir")
    metadata_source_value = metadata.get("source_csv")
    if not isinstance(expected_source, str) or not isinstance(
        metadata_source_value, str
    ):
        raise EvidenceError("measurement source CSV identity is missing")
    source_csv = Path(expected_source) / f"{category}.csv"
    try:
        source_resolved = source_csv.resolve(strict=True)
        metadata_source = Path(metadata_source_value).resolve(strict=True)
    except OSError as error:
        raise EvidenceError("measurement source CSV is unavailable") from error
    if metadata_source != source_resolved:
        raise EvidenceError("metadata source CSV differs from start output identity")
    return (
        verdict,
        str(raw_record["elapsed_seconds"]),
        source_resolved,
        metadata_path,
        authoritative_inputs,
    )


def _validate_source_csv(
    *,
    run_id: str,
    raw_record: dict[str, Any],
    completion: dict[str, Any],
    expected_path: Path,
) -> tuple[list[str], bytes]:
    checks = completion["integrity"]["checks"]
    run_evidence = checks["run_evidence"]
    csv_evidence = run_evidence.get("csv_evidence")
    if not isinstance(csv_evidence, list) or len(csv_evidence) != 1:
        raise EvidenceError("completion must bind exactly one source CSV artifact")
    record = csv_evidence[0]
    if not isinstance(record, dict):
        raise EvidenceError("completion source CSV evidence is invalid")
    declared = record.get("path")
    if not isinstance(declared, str):
        raise EvidenceError("completion source CSV path is missing")
    declared_path = resolved_regular_file(Path(declared), "source CSV")
    if declared_path != expected_path:
        raise EvidenceError(
            "completion source CSV path differs from measurement metadata"
        )
    data = stable_bytes(declared_path, "source CSV")
    if record.get("sha256") != sha256(data) or record.get("size_bytes") != len(data):
        raise EvidenceError("source CSV bytes differ from completion evidence")
    rows = _parse_csv(data, path=declared_path)
    misplaced = [
        row
        for row in rows
        if run_id in row and not (len(row) == 7 and row[6] == run_id)
    ]
    if misplaced:
        raise EvidenceError(
            "source CSV contains the run ID outside its provenance column"
        )
    current = [row for row in rows if len(row) > 6 and row[6] == run_id]
    if (
        record.get("current_run_row_count") != 1
        or run_evidence.get("csv_row_count") != 1
        or len(current) != 1
        or record.get("current_run_rows_sha256") != provenance._identity_sha256(current)
    ):
        raise EvidenceError("source CSV does not retain exactly one bound run row")
    source_row = current[0]
    expected = [
        str(raw_record["category"]),
        str(raw_record["onnx"]),
        str(raw_record["vnnlib"]),
        "0",
        str(raw_record["solver_verdict"]),
        str(raw_record["elapsed_seconds"]),
        run_id,
    ]
    if source_row != expected:
        raise EvidenceError("source CSV row differs from the run-evidence record")
    return source_row, data


def _revalidate_retained_payload_session(session: dict[str, Any]) -> None:
    cache = session.get("authoritative_payload_cache")
    if cache is None:
        return
    benchmark = session.get("authoritative_benchmark")
    if not isinstance(cache, dict) or not isinstance(
        benchmark, PinnedOfficialBenchmark
    ):
        raise EvidenceError("validation session authoritative state is invalid")
    retained = {
        logical_path: value
        for logical_path, value in cache.items()
        if (
            isinstance(logical_path, str)
            and isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], AuthoritativeInput)
            and value[0].retained_setup_payload is not None
        )
    }
    if len(retained) != len(cache):
        raise EvidenceError("validation session authoritative cache is malformed")
    if not retained:
        return

    root = resolved_directory(PINNED_LARGE_MODEL_ROOT, "retained large-model root")
    if root != PINNED_LARGE_MODEL_ROOT:
        raise EvidenceError("retained large-model root path is not canonical")
    _validate_large_model_inventory(root)
    manifest_path = resolved_regular_file(
        root / "manifest.json", "retained large-model manifest"
    )
    manifest_data = stable_bytes(manifest_path, "retained large-model manifest")
    if (
        len(manifest_data) != PINNED_LARGE_MODEL_MANIFEST_SIZE
        or sha256(manifest_data) != PINNED_LARGE_MODEL_MANIFEST_SHA256
        or _json_object(
            manifest_data,
            path=manifest_path,
            label="retained large-model manifest",
        )
        != EXPECTED_LARGE_MODEL_MANIFEST
    ):
        raise EvidenceError("retained large-model manifest changed during validation")
    setup = EXPECTED_LARGE_MODEL_MANIFEST["official_benchmark"]["setup"]
    setup_blob = _git_blob(benchmark, setup["git_path"])
    if (
        setup_blob is None
        or setup_blob[0] != setup["git_blob"]
        or sha256(setup_blob[1]) != setup["sha256"]
    ):
        raise EvidenceError(
            "retained large-model setup source changed during validation"
        )

    expected_payloads = EXPECTED_LARGE_MODEL_MANIFEST["payloads"]
    for logical_path, (authoritative, payload) in retained.items():
        binding = expected_payloads.get(logical_path)
        source = authoritative.retained_setup_payload
        if (
            not isinstance(binding, dict)
            or not isinstance(source, dict)
            or sha256(payload) != authoritative.sha256
            or len(payload) != authoritative.size_bytes
            or authoritative.sha256 != binding["payload_sha256"]
            or authoritative.size_bytes != binding["payload_size_bytes"]
            or authoritative.compressed_sha256 != binding["compressed_sha256"]
            or authoritative.compressed_size_bytes != binding["compressed_size_bytes"]
        ):
            raise EvidenceError("cached retained large-model payload binding differs")
        retained_artifact = binding.get("retained_artifact")
        if not isinstance(retained_artifact, str):
            raise EvidenceError("retained large-model manifest payload path is invalid")
        retained_path = resolved_regular_file(
            root.joinpath(*PurePosixPath(retained_artifact).parts),
            "retained official payload",
        )
        expected_source = _retained_source_binding(
            root=root,
            manifest_path=manifest_path,
            logical_path=logical_path,
            setup=setup,
            payload_binding=binding,
            retained_path=retained_path,
        )
        if source != expected_source:
            raise EvidenceError("retained large-model source path binding differs")
        try:
            observed_digest, _ = provenance._stable_file_hash(retained_path)
            observed_size = retained_path.stat().st_size
        except (OSError, provenance.ProvenanceError) as error:
            raise EvidenceError(
                "could not recheck retained large-model payload"
            ) from error
        if (
            observed_digest != authoritative.compressed_sha256
            or observed_size != authoritative.compressed_size_bytes
        ):
            raise EvidenceError(
                "retained large-model payload changed during validation"
            )
    _validate_large_model_inventory(root)


def revalidate_replay_session(session: dict[str, Any]) -> None:
    _revalidate_retained_payload_session(session)
    snapshot = session.get("snapshot")
    if snapshot is None:
        return
    try:
        import replay_vnncomp2025_counterexample as replay2025  # noqa: PLC0415
    except ImportError as error:
        raise EvidenceError("exact 2025 replay validator is unavailable") from error
    try:
        replay2025.revalidate_replay_snapshot(snapshot)
    except (OSError, replay2025.ReplayError) as error:
        raise EvidenceError(
            f"exact 2025 replay snapshot final recheck failed: {error}"
        ) from error


def _exact_replay_input_binding(
    authoritative: AuthoritativeInput,
    *,
    label: str,
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "sha256": authoritative.sha256,
        "size_bytes": authoritative.size_bytes,
    }
    if authoritative.retained_setup_payload is None:
        if authoritative.git_path is None or authoritative.git_blob is None:
            raise EvidenceError(f"exact 2025 replay {label} Git identity is missing")
        binding.update(
            {
                "official_git_path": authoritative.git_path,
                "official_git_blob": authoritative.git_blob,
            }
        )
    else:
        if authoritative.git_path is not None or authoritative.git_blob is not None:
            raise EvidenceError(
                f"exact 2025 replay {label} source identity is ambiguous"
            )
        binding["official_retained_setup_payload"] = (
            authoritative.retained_setup_payload
        )
    return binding


def validate_exact_2025_sat_replay(
    *,
    root: Path,
    metadata_path: Path,
    metadata_digest: str,
    metadata_size: int,
    result_path: Path,
    result_digest: str,
    result_size: int,
    result_data: bytes,
    start_path: Path,
    start_digest: str,
    start_size: int,
    run_id: str,
    category: str,
    instance_index: int,
    official: PinnedOfficialResults,
    benchmark: PinnedOfficialBenchmark,
    authoritative_inputs: dict[str, AuthoritativeInput],
    replay_session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen one exact-2025 ZERO-TOL SAT replay sidecar fail-closed."""

    # Imported lazily because the replay producer itself imports this module.
    try:
        import replay_vnncomp2025_counterexample as replay2025  # noqa: PLC0415
    except ImportError as error:
        raise EvidenceError("exact 2025 replay validator is unavailable") from error

    sidecar_path = metadata_path.with_name(
        f"{metadata_path.stem}.vnncomp2025-zero-tol-validation.json"
    )
    try:
        resolved_sidecar = resolved_regular_file(
            sidecar_path, "exact 2025 counterexample replay sidecar"
        )
        resolved_sidecar.relative_to(root)
    except ValueError as error:
        raise EvidenceError(
            "exact 2025 counterexample replay sidecar escapes artifact root"
        ) from error
    sidecar_data = stable_bytes(
        resolved_sidecar, "exact 2025 counterexample replay sidecar"
    )
    try:
        sidecar = replay2025._json_loads(
            sidecar_data,
            "exact 2025 counterexample replay sidecar",
        )
        replay2025._validate_sidecar_shape(sidecar)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        replay2025.ReplayError,
    ) as error:
        raise EvidenceError(
            "exact 2025 counterexample replay sidecar is invalid"
        ) from error
    assert isinstance(sidecar, dict)

    result = sidecar.get("official_result")
    credit = result in replay2025.CREDIT_RESULTS
    timestamp = sidecar.get("validated_at_utc")
    try:
        parsed_timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(
            "exact 2025 replay timestamp is not canonical UTC"
        ) from error
    if (
        not isinstance(timestamp, str)
        or not timestamp.endswith("Z")
        or parsed_timestamp.tzinfo != timezone.utc
        or sidecar.get("schema") != replay2025.SCHEMA
        or sidecar.get("schema_version") != replay2025.SCHEMA_VERSION
        or sidecar.get("status") != "validated"
        or sidecar.get("classification") != ("valid" if credit else "invalid")
        or sidecar.get("score_credit") is not credit
        or sidecar.get("scoring_year") != 2025
        or sidecar.get("official_result") != sidecar["response"].get("result")
        or sidecar.get("rationale") != sidecar["response"].get("message")
    ):
        raise EvidenceError(
            "exact 2025 replay status/result/classification is inconsistent"
        )

    expected_settings = {
        "ignore_ce_y": False,
        "counterexample_atol": 1e-4,
        "counterexample_rtol": 1e-3,
        "scoring_zero_tolerance": True,
    }
    if sidecar.get("settings") != expected_settings:
        raise EvidenceError("exact 2025 replay settings differ from ZERO-TOL")
    if sidecar.get("measurement") != {
        "run_id": run_id,
        "category": category,
        "instance_index": instance_index,
    }:
        raise EvidenceError("exact 2025 replay measurement identity differs")

    owns_snapshot = replay_session is None
    session = replay_session if replay_session is not None else {}
    try:
        snapshot = session.get("snapshot")
        if snapshot is None:
            snapshot = replay2025.capture_replay_snapshot()
            session["snapshot"] = snapshot
        if not isinstance(snapshot, dict):
            raise replay2025.ReplayError("replay snapshot is not an object")
        expected_checker = replay2025._checker_identity(
            official.root, replay2025.PINNED_RUNTIME_ROOT
        )
        expected_harness = snapshot.get("harness")
        expected_runtime = snapshot.get("runtime")
    except (OSError, replay2025.ReplayError) as error:
        raise EvidenceError(
            f"exact 2025 replay checker/runtime identity is unavailable: {error}"
        ) from error
    if sidecar.get("checker") != expected_checker:
        raise EvidenceError("exact 2025 replay checker identity differs")
    if (
        not isinstance(expected_harness, dict)
        or expected_harness.get("runner_sha256") != PINNED_REPLAY_RUNNER_SHA256
        or expected_harness.get("worker_sha256") != PINNED_REPLAY_WORKER_SHA256
        or expected_harness.get("protocol") != "ny_vnncomp2025_zero_tol_worker_v1"
    ):
        raise EvidenceError("pinned exact 2025 replay harness hashes differ")
    if sidecar.get("harness") != expected_harness:
        raise EvidenceError("exact 2025 replay harness identity differs")
    if (
        not isinstance(expected_runtime, dict)
        or expected_runtime.get("execution_scope") != "host_bound_local_replay"
    ):
        raise EvidenceError("exact 2025 replay runtime claim scope differs")
    if sidecar.get("runtime") != expected_runtime:
        raise EvidenceError("exact 2025 replay runtime identity differs")

    expected_links = {
        "metadata": {
            "artifact": metadata_path.relative_to(root).as_posix(),
            "sha256": metadata_digest,
            "size_bytes": metadata_size,
        },
        "raw_result": {
            "artifact": result_path.relative_to(root).as_posix(),
            "sha256": result_digest,
            "size_bytes": result_size,
        },
        "start_manifest": {
            "artifact": start_path.relative_to(root).as_posix(),
            "sha256": start_digest,
            "size_bytes": start_size,
        },
    }
    replay_evidence = sidecar.get("evidence")
    assert isinstance(replay_evidence, dict)
    if any(
        replay_evidence.get(label) != expected
        for label, expected in expected_links.items()
    ):
        raise EvidenceError("exact 2025 replay artifact links differ")
    try:
        assignment = replay2025._extract_assignment(result_data)
    except replay2025.ReplayError as error:
        raise EvidenceError(str(error)) from error
    if replay_evidence.get("extracted_assignment") != {
        "sha256": sha256(assignment),
        "size_bytes": len(assignment),
        "transformation": "removed_standalone_sat_verdict_line_only",
    }:
        raise EvidenceError("exact 2025 replay assignment binding differs")
    for label in ("onnx", "vnnlib"):
        authoritative = authoritative_inputs.get(label)
        if authoritative is None:
            raise EvidenceError(f"exact 2025 replay has no authoritative {label} input")
        expected_input_binding = _exact_replay_input_binding(
            authoritative,
            label=label,
        )
        if replay_evidence.get(label) != expected_input_binding:
            raise EvidenceError(
                f"exact 2025 replay {label} authoritative input binding differs"
            )
    try:
        authoritative_payloads = {
            label: authoritative_benchmark_input(
                benchmark=benchmark,
                category=category,
                declared_name=authoritative_inputs[label].declared_name,
                label=label,
                payload_cache=(
                    replay_session.get("authoritative_payload_cache")
                    if replay_session is not None
                    else None
                ),
            )[1]
            for label in ("onnx", "vnnlib")
        }
        independently_observed = replay2025.replay_bound_payloads(
            onnx_payload=authoritative_payloads["onnx"],
            vnnlib_payload=authoritative_payloads["vnnlib"],
            assignment_bytes=assignment,
            timeout_seconds=600,
            snapshot=snapshot,
        )
    except (OSError, replay2025.ReplayError) as error:
        raise EvidenceError(
            f"independent exact 2025 counterexample replay failed: {error}"
        ) from error
    if any(
        independently_observed.get(label) != sidecar.get(label)
        for label in ("harness", "runtime", "response", "worker_receipt")
    ):
        raise EvidenceError("sidecar differs from independent exact 2025 bound replay")

    final_metadata = stable_bytes(metadata_path, "measurement metadata")
    final_result = stable_bytes(result_path, "raw result")
    final_start = stable_bytes(start_path, "start manifest")
    final_sidecar = stable_bytes(
        resolved_sidecar, "exact 2025 counterexample replay sidecar"
    )
    if (
        (sha256(final_metadata), len(final_metadata))
        != (metadata_digest, metadata_size)
        or (sha256(final_result), len(final_result)) != (result_digest, result_size)
        or final_result != result_data
        or (sha256(final_start), len(final_start)) != (start_digest, start_size)
        or final_sidecar != sidecar_data
    ):
        raise EvidenceError("exact 2025 replay evidence changed during validation")
    try:
        final_checker = replay2025._checker_identity(
            official.root, replay2025.PINNED_RUNTIME_ROOT
        )
        if owns_snapshot:
            replay2025.revalidate_replay_snapshot(snapshot)
    except (OSError, replay2025.ReplayError) as error:
        raise EvidenceError(
            f"exact 2025 replay checker/runtime final recheck failed: {error}"
        ) from error
    if final_checker != expected_checker:
        raise EvidenceError(
            "exact 2025 replay checker/runtime changed during validation"
        )
    return {
        "artifact": resolved_sidecar.relative_to(root).as_posix(),
        "sha256": sha256(sidecar_data),
        "size_bytes": len(sidecar_data),
        "schema": replay2025.SCHEMA,
        "classification": sidecar["classification"],
        "official_result": result,
        "score_credit": credit,
        "checker_commit": sidecar["checker"]["commit"],
        "settings": sidecar["settings"],
    }


def published_policy(
    *,
    sealed: gap.SealedRecord,
    truth: str,
    sat_replay: dict[str, Any] | None,
    organizer_rescore: dict[str, Any] | None,
) -> str:
    """Apply the pinned 2025 scoring policy to one sealed decision.

    A positive pinned SAT replay earns credit for its own tool even when the
    ZERO-TOL ``holds`` truth was produced because no witness was *strictly*
    correct.  This is the organizer's tolerance-aware scoring behavior.
    """

    if sealed.verdict == "sat" and (
        sat_replay is None or sat_replay.get("score_credit") is not True
    ):
        raise EvidenceError(
            "SAT promotion requires a creditable exact-2025 ZERO-TOL replay"
        )
    if (
        sealed.verdict == "sat"
        and sat_replay is not None
        and sat_replay.get("official_result") == "correct"
        and truth == "holds"
    ):
        if (
            not isinstance(organizer_rescore, dict)
            or organizer_rescore.get("schema") != ORGANIZER_RESCORE_SCHEMA
            or organizer_rescore.get("truth")
            != {
                "published": "holds",
                "rescored": "violated",
                "cause": "ny_strictly_correct_exact_2025_counterexample",
            }
        ):
            raise EvidenceError(
                "strictly correct SAT against published holds requires a "
                "complete dynamic organizer rescore"
            )
        return "exact_2025_zero_tol_replay_correct_dynamic_rescore_v1"
    if organizer_rescore is not None:
        raise EvidenceError(
            "dynamic organizer rescore is only valid for a strict SAT truth change"
        )
    if sealed.verdict == "unsat" and truth != "holds":
        raise EvidenceError("UNSAT promotion requires published holds truth")
    points = gap._score_record(sealed, truth)
    if points != retro.POINTS_CORRECT:
        raise EvidenceError(
            f"sealed {sealed.verdict} does not earn positive published credit "
            f"(truth={truth}, score={points})"
        )
    if sealed.verdict == "sat":
        assert sat_replay is not None
        return f"exact_2025_zero_tol_replay_{sat_replay['official_result']}_v1"
    return "sealed_unsat_plus_published_holds_v1"


def validate_promotion_evidence(
    *,
    artifact_root: Path,
    run_id: str,
    category: str,
    instance_index: int,
    benchmark_root: Path,
    official_results: Path,
    exact_commit: str,
    pinned_official: PinnedOfficialResults | None = None,
    pinned_benchmark: PinnedOfficialBenchmark | None = None,
    replay_session: dict[str, Any] | None = None,
) -> ValidatedPromotionEvidence:
    """Fully reopen and validate one sealed candidate promotion."""

    owns_validation_session = replay_session is None
    active_session = replay_session if replay_session is not None else {}
    root = resolved_directory(artifact_root, "measurement artifact root")
    benchmark = resolved_directory(benchmark_root, "benchmark root")
    pinned_benchmark = pinned_benchmark or validate_official_benchmark(benchmark)
    if pinned_benchmark.benchmark_root != benchmark:
        raise EvidenceError("caller benchmark root differs from pinned benchmark root")
    official = pinned_official or validate_official_results(official_results)
    requested_official = resolved_directory(official_results, "official result root")
    if requested_official != official.root:
        raise EvidenceError("caller official root differs from pinned official root")
    session_benchmark = active_session.setdefault(
        "authoritative_benchmark", pinned_benchmark
    )
    if session_benchmark != pinned_benchmark:
        raise EvidenceError("validation session was reused across benchmark snapshots")
    authoritative_cache = active_session.setdefault("authoritative_payload_cache", {})
    if not isinstance(authoritative_cache, dict):
        raise EvidenceError("validation session authoritative cache is invalid")
    occurrence, benchmark_occurrence = _load_occurrence(
        category=category,
        instance_index=instance_index,
        benchmark=pinned_benchmark,
        official=official,
    )
    (
        start_path,
        start,
        start_digest,
        start_size,
        source_snapshot,
    ) = _load_start(
        root=root,
        run_id=run_id,
        category=category,
        instance_index=instance_index,
        benchmark=pinned_benchmark,
        exact_commit=exact_commit,
    )
    containment_profile = _start_containment_profile(start)
    (
        completion_path,
        completion,
        completion_digest,
        completion_size,
        sealed,
        raw_record,
    ) = _load_completion(
        root=root,
        start_path=start_path,
        start=start,
        start_digest=start_digest,
        start_size=start_size,
        official=official,
        benchmark=pinned_benchmark,
        replay_session=active_session,
    )
    (
        verdict,
        runtime,
        source_csv,
        metadata_path,
        authoritative_inputs,
    ) = _validate_record_details(
        root=root,
        run_id=run_id,
        category=category,
        instance_index=instance_index,
        benchmark=pinned_benchmark,
        start_path=start_path,
        start_digest=start_digest,
        start=start,
        occurrence=occurrence,
        sealed=sealed,
        raw_record=raw_record,
        authoritative_cache=authoritative_cache,
    )
    source_row, source_csv_data = _validate_source_csv(
        run_id=run_id,
        raw_record=raw_record,
        completion=completion,
        expected_path=source_csv,
    )
    sat_replay = None
    if verdict == "sat":
        metadata_data = stable_bytes(metadata_path, "measurement metadata")
        try:
            result_path, result_data, result_digest, result_size = (
                gap._checked_artifact(root, raw_record.get("result"), "raw result")
            )
        except gap.AuditError as error:
            raise EvidenceError(str(error)) from error
        sat_replay = validate_exact_2025_sat_replay(
            root=root,
            metadata_path=metadata_path,
            metadata_digest=sha256(metadata_data),
            metadata_size=len(metadata_data),
            result_path=result_path,
            result_digest=result_digest,
            result_size=result_size,
            result_data=result_data,
            start_path=start_path,
            start_digest=start_digest,
            start_size=start_size,
            run_id=run_id,
            category=category,
            instance_index=instance_index,
            official=official,
            benchmark=pinned_benchmark,
            authoritative_inputs=authoritative_inputs,
            replay_session=active_session,
        )
        replay_result = str(sat_replay["official_result"])
        replay_mapping = {
            "correct": gap.competitive.CounterexampleResult.CORRECT,
            "correct_up_to_tolerance": (
                gap.competitive.CounterexampleResult.CORRECT_UP_TO_TOLERANCE
            ),
            "no_ce": gap.competitive.CounterexampleResult.NO_COUNTEREXAMPLE,
            "exec_doesnt_match": (
                gap.competitive.CounterexampleResult.EXEC_DOESNT_MATCH
            ),
            "wrong_shape": gap.competitive.CounterexampleResult.EXEC_DOESNT_MATCH,
            "spec_not_violated": (
                gap.competitive.CounterexampleResult.SPEC_NOT_VIOLATED
            ),
        }
        sealed = replace(
            sealed,
            counterexample=replay_mapping[replay_result],
            sat_replay_state=f"exact_2025_zero_tol:{replay_result}",
        )
    truth = official.context.ground_truth.get(category, {}).get(occurrence.score_key)
    if truth not in {"holds", "violated"}:
        raise EvidenceError(
            "regular-bank promotion is supported only when the pinned published "
            f"truth is holds or violated; no decided truth exists for "
            f"{canonical_row_key(category, occurrence)}"
        )
    organizer_rescore = None
    if (
        sealed.verdict == "sat"
        and sat_replay is not None
        and sat_replay.get("official_result") == "correct"
        and truth == "holds"
    ):
        organizer_rescore = dynamic_organizer_rescore(
            official=official,
            category=category,
            occurrence=occurrence,
        )
    policy = published_policy(
        sealed=sealed,
        truth=truth,
        sat_replay=sat_replay,
        organizer_rescore=organizer_rescore,
    )

    if stable_bytes(source_csv, "source CSV") != source_csv_data:
        raise EvidenceError("source CSV changed while evidence was validated")
    try:
        _, final_start_digest, final_start_size = gap._stable_json(
            start_path, "start manifest"
        )
        _, final_completion_digest, final_completion_size = gap._stable_json(
            completion_path, "completion manifest"
        )
    except gap.AuditError as error:
        raise EvidenceError(str(error)) from error
    if (final_start_digest, final_start_size) != (start_digest, start_size):
        raise EvidenceError("start manifest changed while evidence was validated")
    if (final_completion_digest, final_completion_size) != (
        completion_digest,
        completion_size,
    ):
        raise EvidenceError("completion changed while evidence was validated")
    revalidate_official_benchmark(pinned_benchmark)
    revalidate_official_results(official)
    if organizer_rescore is not None:
        revalidate_organizer_rescore(official, organizer_rescore)
    if owns_validation_session:
        revalidate_replay_session(active_session)
    return ValidatedPromotionEvidence(
        artifact_root=root,
        benchmark_root=benchmark,
        official_benchmark=pinned_benchmark,
        official=official,
        run_id=run_id,
        category=category,
        instance_index=instance_index,
        exact_commit=exact_commit,
        occurrence=occurrence,
        benchmark_occurrence=benchmark_occurrence,
        start_path=start_path,
        start=start,
        start_sha256=start_digest,
        start_size_bytes=start_size,
        completion_path=completion_path,
        completion=completion,
        completion_sha256=completion_digest,
        completion_size_bytes=completion_size,
        sealed=sealed,
        raw_record=raw_record,
        verdict=verdict,
        runtime_seconds=runtime,
        source_csv=source_csv,
        source_row=source_row,
        source_csv_data=source_csv_data,
        published_truth=truth,
        policy=policy,
        sat_replay=sat_replay,
        organizer_rescore=organizer_rescore,
        containment_profile=containment_profile,
        authoritative_inputs=authoritative_inputs,
        source_snapshot=source_snapshot,
    )


def _physical_csv_rows(data: bytes, *, path: Path) -> list[BankRow]:
    rows: list[BankRow] = []
    for line_index, line in enumerate(data.splitlines(keepends=True)):
        body = line
        if body.endswith(b"\r\n"):
            body = body[:-2]
        elif body.endswith((b"\n", b"\r")):
            body = body[:-1]
        if not body:
            continue
        parsed = _parse_csv(body, path=path)
        if len(parsed) != 1:
            raise EvidenceError(
                f"measured CSV must use one physical line per row: {path}"
            )
        rows.append(BankRow(line_index, parsed[0]))
    return rows


def locate_bank_row(
    *,
    measured_path: Path,
    data: bytes,
    category: str,
    occurrence: retro.OfficialInstanceOccurrence,
    official_reference: list[tuple],
) -> BankRow:
    target_base = occurrence.score_key[:2]
    expected_multiplicity = sum(
        1 for key in official_reference if key[:2] == target_base
    )
    matching: list[BankRow] = []
    for bank_row in _physical_csv_rows(data, path=measured_path):
        row = bank_row.fields
        if not row or row[0].strip().lower() in {"cat", "category"}:
            continue
        if row[0].strip() != category:
            continue
        if len(row) not in {6, 7}:
            raise EvidenceError(
                "measured row must have six or seven columns: "
                f"{measured_path}:{bank_row.line_index + 1}"
            )
        if retro.is_harness_test_instance(row[1], row[2]):
            continue
        if retro.key(row[1], row[2]) == target_base:
            matching.append(bank_row)
    if expected_multiplicity <= 0 or len(matching) != expected_multiplicity:
        raise EvidenceError(
            "measured bank has a duplicate or missing canonical target row "
            f"(expected {expected_multiplicity}, found {len(matching)})"
        )
    occurrence_number = int(occurrence.score_key[2])
    if occurrence_number >= len(matching):
        raise EvidenceError("measured bank cannot identify the target occurrence")
    return matching[occurrence_number]


def _benchmark_binding(
    evidence: ValidatedPromotionEvidence,
) -> dict[str, Any]:
    def input_binding(value: AuthoritativeInput) -> dict[str, Any]:
        binding: dict[str, Any] = {
            "declared_name": value.declared_name,
            "compression": value.compression,
            "compressed_sha256": value.compressed_sha256,
            "compressed_size_bytes": value.compressed_size_bytes,
            "sha256": value.sha256,
            "size_bytes": value.size_bytes,
        }
        if value.retained_setup_payload is None:
            if value.git_path is None or value.git_blob is None:
                raise EvidenceError(
                    "authoritative Git input has no Git object identity"
                )
            binding.update(
                {
                    "source_kind": "git_blob",
                    "git_path": value.git_path,
                    "git_blob": value.git_blob,
                }
            )
        else:
            if value.git_path is not None or value.git_blob is not None:
                raise EvidenceError(
                    "retained authoritative input also claims a Git payload"
                )
            binding["source_kind"] = "official_setup_retained_payload"
            binding["retained_setup_payload"] = value.retained_setup_payload
        return binding

    return {
        **evidence.benchmark_occurrence,
        "inputs": {
            label: input_binding(value)
            for label, value in sorted(evidence.authoritative_inputs.items())
        },
    }


def _v2_benchmark_binding(
    evidence: ValidatedPromotionEvidence,
) -> dict[str, Any]:
    occurrence = evidence.occurrence
    return {
        "instance_index": evidence.instance_index,
        "instances_csv": str(occurrence.instances_csv),
        "instances_csv_sha256": occurrence.instances_csv_sha256,
        "official_timeout_seconds": str(occurrence.timeout_seconds),
        "onnx": occurrence.onnx,
        "pair_occurrence": occurrence.pair_occurrence,
        "vnnlib": occurrence.vnnlib,
    }


def _v3_benchmark_binding(
    evidence: ValidatedPromotionEvidence,
) -> dict[str, Any]:
    if any(
        value.retained_setup_payload is not None
        for value in evidence.authoritative_inputs.values()
    ):
        raise EvidenceError(
            "v3 evidence cannot represent a retained setup payload source"
        )
    return {
        **evidence.benchmark_occurrence,
        "inputs": {
            label: {
                "declared_name": value.declared_name,
                "git_path": value.git_path,
                "git_blob": value.git_blob,
                "compression": value.compression,
                "compressed_sha256": value.compressed_sha256,
                "compressed_size_bytes": value.compressed_size_bytes,
                "sha256": value.sha256,
                "size_bytes": value.size_bytes,
            }
            for label, value in sorted(evidence.authoritative_inputs.items())
        },
    }


def _official_benchmark_binding(
    evidence: ValidatedPromotionEvidence,
) -> dict[str, Any]:
    return {
        "root": str(evidence.official_benchmark.benchmark_root),
        "repository_root": str(evidence.official_benchmark.repository_root),
        **evidence.official_benchmark.identity,
    }


def _official_binding(evidence: ValidatedPromotionEvidence) -> dict[str, Any]:
    return {
        "root": str(evidence.official.root),
        **evidence.official.identity,
    }


def _static_entry_payload(
    evidence: ValidatedPromotionEvidence,
    *,
    version: int,
) -> dict[str, Any]:
    if version not in {1, 2, 3, 4, 5, 6, 7}:
        raise EvidenceError(f"internal: unsupported evidence entry version {version}")
    if (version in {5, 7}) != (evidence.organizer_rescore is not None):
        raise EvidenceError(
            "dynamic organizer rescore and evidence entry version disagree"
        )
    if version in {6, 7} and evidence.containment_profile is None:
        raise EvidenceError(
            "new regular evidence requires start.host.containment."
            "containment_profile"
        )
    payload: dict[str, Any] = {
        "artifact_root": str(evidence.artifact_root),
        "benchmark": (
            _benchmark_binding(evidence)
            if version >= 4
            else (
                _v3_benchmark_binding(evidence)
                if version == 3
                else _v2_benchmark_binding(evidence)
            )
        ),
        "category": evidence.category,
        "completion": {
            "artifact": evidence.completion_path.relative_to(
                evidence.artifact_root
            ).as_posix(),
            "sha256": evidence.completion_sha256,
            "size_bytes": evidence.completion_size_bytes,
        },
        "exact_commit": evidence.exact_commit,
        "policy": evidence.policy,
        "published_truth": evidence.published_truth,
        "run_id": evidence.run_id,
        "runtime_seconds": evidence.runtime_seconds,
        "source_csv": {
            "path": str(evidence.source_csv),
            "row_sha256": provenance._identity_sha256(evidence.source_row),
            "sha256": sha256(evidence.source_csv_data),
        },
        "start_manifest": {
            "artifact": evidence.start_path.relative_to(
                evidence.artifact_root
            ).as_posix(),
            "sha256": evidence.start_sha256,
            "size_bytes": evidence.start_size_bytes,
        },
        "verdict": evidence.verdict,
    }
    if version >= 2:
        schema_by_version = {
            2: V2_ENTRY_SCHEMA,
            3: PREVIOUS_ENTRY_SCHEMA,
            4: PRE_PROFILE_ENTRY_SCHEMA,
            5: PRE_PROFILE_DYNAMIC_ENTRY_SCHEMA,
            6: ENTRY_SCHEMA,
            7: DYNAMIC_ENTRY_SCHEMA,
        }
        payload.update(
            {
                "entry_schema": schema_by_version[version],
                "official_results": _official_binding(evidence),
                "sat_replay": evidence.sat_replay,
            }
        )
    if version >= 3:
        payload.update(
            {
                "official_benchmark": _official_benchmark_binding(evidence),
                "source_snapshot": evidence.source_snapshot,
            }
        )
    if version in {6, 7}:
        payload["containment_profile"] = evidence.containment_profile
    if version in {5, 7}:
        payload["organizer_rescore"] = evidence.organizer_rescore
    return payload


def promotion_evidence_binding(
    evidence: ValidatedPromotionEvidence,
    *,
    allow_pre_profile_start: bool = False,
) -> dict[str, Any]:
    """Return the immutable, bank-independent binding for a promotion."""

    if evidence.containment_profile is None and allow_pre_profile_start:
        version = 5 if evidence.organizer_rescore is not None else 4
    else:
        version = 7 if evidence.organizer_rescore is not None else 6
    return _static_entry_payload(
        evidence,
        version=version,
    )


def make_index_entry(
    evidence: ValidatedPromotionEvidence,
    *,
    measured_path: Path,
    measured_before: bytes,
    measured_after: bytes,
    row_before: list[str],
    row_after: list[str],
    migrate_legacy_decided_row: bool = False,
) -> dict[str, Any]:
    if len(row_before) not in {6, 7}:
        raise EvidenceError("bank row-before binding has an invalid width")
    if migrate_legacy_decided_row:
        if row_before[4] not in DECIDED_VERDICTS:
            raise EvidenceError(
                "legacy decided-row migration requires a canonical decided verdict"
            )
        if row_before[4] != evidence.verdict:
            raise EvidenceError(
                "legacy decided-row verdict differs from sealed evidence"
            )
    elif row_before[4].strip().lower() not in UNRESOLVED_LITERAL_VERDICTS:
        raise EvidenceError("bank row-before binding is not unresolved")
    if row_before[:3] != [
        evidence.category,
        evidence.occurrence.onnx,
        evidence.occurrence.vnnlib,
    ]:
        raise EvidenceError("bank row-before identity is not the official occurrence")
    expected_after = row_before[:4] + [
        evidence.verdict,
        evidence.runtime_seconds,
        evidence.run_id,
    ]
    if row_after != expected_after:
        raise EvidenceError("bank row-after binding differs from sealed evidence")
    entry = promotion_evidence_binding(evidence)
    measured_binding = {
        "path": str(measured_path),
        "row_after": row_after,
        "row_after_sha256": provenance._identity_sha256(row_after),
        "row_before": row_before,
        "row_before_sha256": provenance._identity_sha256(row_before),
        # These are immutable transaction hashes.  They are deliberately not
        # interpreted as hashes of the bank's future/current whole-file state.
        "sha256_after": sha256(measured_after),
        "sha256_before": sha256(measured_before),
    }
    if migrate_legacy_decided_row:
        measured_binding["migration"] = LEGACY_DECIDED_ROW_MIGRATION
    entry["measured_csv"] = measured_binding
    return entry


def migrate_legacy_index_entry(
    evidence: ValidatedPromotionEvidence,
    *,
    legacy_entry: dict[str, Any],
    measured_path: Path,
    row_after: list[str],
) -> dict[str, Any]:
    """Upgrade an applied legacy entry without inventing its lost before-row."""

    measured = legacy_entry.get("measured_csv")
    allowed = {
        frozenset({"path", "sha256_after", "sha256_before"}),
        frozenset(
            {
                "path",
                "row_after",
                "row_after_sha256",
                "sha256_after",
                "sha256_before",
            }
        ),
        frozenset(
            {
                "path",
                "row_after",
                "row_after_sha256",
                "row_before",
                "row_before_sha256",
                "sha256_after",
                "sha256_before",
            }
        ),
    }
    if (
        not isinstance(measured, dict)
        or frozenset(measured) not in allowed
        or measured.get("path") != str(measured_path)
        or not _is_sha256(measured.get("sha256_before"))
        or not _is_sha256(measured.get("sha256_after"))
    ):
        raise EvidenceError("legacy measured transaction binding is invalid")
    expected_after = [
        evidence.category,
        evidence.occurrence.onnx,
        evidence.occurrence.vnnlib,
        row_after[3],
        evidence.verdict,
        evidence.runtime_seconds,
        evidence.run_id,
    ]
    if row_after != expected_after:
        raise EvidenceError("legacy applied row differs from reopened evidence")
    migrated = promotion_evidence_binding(evidence)
    migrated_measured = dict(measured)
    migrated_measured.update(
        {
            "path": str(measured_path),
            "row_after": row_after,
            "row_after_sha256": provenance._identity_sha256(row_after),
            "sha256_after": measured["sha256_after"],
            "sha256_before": measured["sha256_before"],
        }
    )
    migrated["measured_csv"] = migrated_measured
    return migrated


def _validate_hash_size_binding(value: object, *, label: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"artifact", "sha256", "size_bytes"}
        or not _safe_index_artifact(value.get("artifact"))
        or not _is_sha256(value.get("sha256"))
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] < 0
    ):
        raise EvidenceError(f"evidence index has an invalid {label} binding")


def _validate_entry_shape(
    row_key: str, entry: object, *, path: Path
) -> tuple[int, tuple[str, str, str, int]]:
    identity = parse_row_key(row_key, path=path)
    if not isinstance(entry, dict):
        raise EvidenceError(f"evidence index row {row_key} is not an object")
    schema = entry.get("entry_schema")
    if schema is None:
        version = 1
        expected_keys = _LEGACY_ENTRY_KEYS
    elif schema == V2_ENTRY_SCHEMA:
        version = 2
        expected_keys = _V2_ENTRY_KEYS
    elif schema == PREVIOUS_ENTRY_SCHEMA:
        version = 3
        expected_keys = _V3_ENTRY_KEYS
    elif schema == PRE_PROFILE_ENTRY_SCHEMA:
        version = 4
        expected_keys = _CURRENT_ENTRY_KEYS
    elif schema == PRE_PROFILE_DYNAMIC_ENTRY_SCHEMA:
        version = 5
        expected_keys = _PRE_PROFILE_DYNAMIC_ENTRY_KEYS
    elif schema == ENTRY_SCHEMA:
        version = 6
        expected_keys = _PROFILED_ENTRY_KEYS
    elif schema == DYNAMIC_ENTRY_SCHEMA:
        version = 7
        expected_keys = _PROFILED_DYNAMIC_ENTRY_KEYS
    else:
        raise EvidenceError(
            f"evidence index row {row_key} has unsupported entry schema"
        )
    if set(entry) != expected_keys:
        raise EvidenceError(
            f"evidence index row {row_key} has unsupported entry fields"
        )
    if entry.get("category") != identity[0]:
        raise EvidenceError(f"evidence index row {row_key} category differs")
    artifact_root = entry.get("artifact_root")
    run_id = entry.get("run_id")
    if (
        not isinstance(artifact_root, str)
        or not Path(artifact_root).is_absolute()
        or not isinstance(run_id, str)
        or provenance.SAFE_COMPONENT.fullmatch(run_id) is None
        or gap.EXACT_COMMIT_RE.fullmatch(str(entry.get("exact_commit"))) is None
        or entry.get("verdict") not in DECIDED_VERDICTS
    ):
        raise EvidenceError(f"evidence index row {row_key} has invalid run identity")
    _validate_hash_size_binding(entry.get("completion"), label="completion")
    _validate_hash_size_binding(entry.get("start_manifest"), label="start manifest")
    return version, identity


def _validate_current_measured_binding(
    *,
    row_key: str,
    value: object,
    measured_path: Path,
    current_row: list[str],
    evidence: ValidatedPromotionEvidence,
) -> str:
    resumable = {
        "path",
        "row_after",
        "row_after_sha256",
        "row_before",
        "row_before_sha256",
        "sha256_after",
        "sha256_before",
    }
    legacy_decided_migration = resumable | {"migration"}
    migrated_legacy = {
        "path",
        "row_after",
        "row_after_sha256",
        "sha256_after",
        "sha256_before",
    }
    if not isinstance(value, dict):
        raise EvidenceError(
            f"evidence index row {row_key} has invalid measured-row binding"
        )
    field_names = frozenset(value)
    if field_names not in {
        frozenset(resumable),
        frozenset(legacy_decided_migration),
        frozenset(migrated_legacy),
    }:
        raise EvidenceError(
            f"evidence index row {row_key} has invalid measured-row binding"
        )
    if value.get("path") != str(measured_path):
        raise EvidenceError(
            f"evidence index row {row_key} measured path differs from caller bank"
        )
    after = value.get("row_after")
    if (
        not isinstance(after, list)
        or not all(isinstance(field, str) for field in after)
        or len(after) != 7
        or value.get("row_after_sha256") != provenance._identity_sha256(after)
        or not _is_sha256(value.get("sha256_before"))
        or not _is_sha256(value.get("sha256_after"))
        or after[:3]
        != [
            evidence.category,
            evidence.occurrence.onnx,
            evidence.occurrence.vnnlib,
        ]
        or after[4:] != [evidence.verdict, evidence.runtime_seconds, evidence.run_id]
    ):
        raise EvidenceError(
            f"evidence index row {row_key} has malformed after-row binding"
        )
    if field_names == migrated_legacy:
        if current_row != after:
            raise EvidenceError(
                "measured bank row does not match migrated indexed after-row: "
                f"{row_key}"
            )
        return "applied"

    before = value.get("row_before")
    if (
        not isinstance(before, list)
        or not all(isinstance(field, str) for field in before)
        or len(before) not in {6, 7}
        or value.get("row_before_sha256") != provenance._identity_sha256(before)
    ):
        raise EvidenceError(
            f"evidence index row {row_key} has malformed transaction hashes"
        )
    expected_after = before[:4] + [
        evidence.verdict,
        evidence.runtime_seconds,
        evidence.run_id,
    ]
    is_legacy_decided_migration = field_names == frozenset(legacy_decided_migration)
    if is_legacy_decided_migration:
        if (
            value.get("migration") != LEGACY_DECIDED_ROW_MIGRATION
            or before[4] not in DECIDED_VERDICTS
            or before[4] != evidence.verdict
        ):
            raise EvidenceError(
                f"evidence index row {row_key} has an invalid legacy "
                "decided-row migration"
            )
    elif before[4].strip().lower() not in UNRESOLVED_LITERAL_VERDICTS:
        raise EvidenceError(
            f"evidence index row {row_key} before-row is not unresolved"
        )
    if (
        before[:3]
        != [
            evidence.category,
            evidence.occurrence.onnx,
            evidence.occurrence.vnnlib,
        ]
        or after != expected_after
    ):
        raise EvidenceError(
            f"evidence index row {row_key} before/after rows are inconsistent"
        )
    if current_row == after:
        return "applied"
    if current_row == before:
        return "dangling"
    raise EvidenceError(
        f"measured bank row does not match indexed before or after binding: {row_key}"
    )


def _validate_legacy_measured_binding(
    *,
    row_key: str,
    value: object,
    measured_path: Path,
    current_row: list[str],
    evidence: ValidatedPromotionEvidence,
) -> str:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256_after", "sha256_before"}
        or value.get("path") != str(measured_path)
        or not _is_sha256(value.get("sha256_before"))
        or not _is_sha256(value.get("sha256_after"))
    ):
        raise EvidenceError(
            f"legacy evidence index row {row_key} has invalid transaction binding"
        )
    if (
        len(current_row) != 7
        or current_row[:3]
        != [
            evidence.category,
            evidence.occurrence.onnx,
            evidence.occurrence.vnnlib,
        ]
        or current_row[4:]
        != [evidence.verdict, evidence.runtime_seconds, evidence.run_id]
    ):
        raise EvidenceError(
            f"legacy evidence index row {row_key} is not applied in the measured bank"
        )
    return "applied"


def _validate_index_entry(
    *,
    row_key: str,
    raw_entry: object,
    index_path: Path,
    benchmark_root: Path,
    measured_dir: Path,
    official: PinnedOfficialResults,
    official_benchmark: PinnedOfficialBenchmark,
    measured_cache: dict[Path, bytes],
    replay_session: dict[str, Any],
) -> ValidatedIndexEntry:
    version, identity = _validate_entry_shape(row_key, raw_entry, path=index_path)
    assert isinstance(raw_entry, dict)
    category, _, _, _ = identity
    benchmark_binding = raw_entry.get("benchmark")
    if not isinstance(benchmark_binding, dict):
        raise EvidenceError(f"evidence index row {row_key} has no benchmark binding")
    instance_index = benchmark_binding.get("instance_index")
    if type(instance_index) is not int or instance_index <= 0:
        raise EvidenceError(f"evidence index row {row_key} has invalid instance index")
    evidence = validate_promotion_evidence(
        artifact_root=Path(str(raw_entry["artifact_root"])),
        run_id=str(raw_entry["run_id"]),
        category=category,
        instance_index=instance_index,
        benchmark_root=benchmark_root,
        official_results=official.root,
        exact_commit=str(raw_entry["exact_commit"]),
        pinned_official=official,
        pinned_benchmark=official_benchmark,
        replay_session=replay_session,
    )
    if canonical_row_key(category, evidence.occurrence) != row_key:
        raise EvidenceError(
            f"evidence index row {row_key} differs from official occurrence"
        )
    expected_static = _static_entry_payload(evidence, version=version)
    for name, expected in expected_static.items():
        if raw_entry.get(name) != expected:
            raise EvidenceError(
                f"evidence index row {row_key} {name} binding differs "
                "from reopened evidence"
            )

    measured_path = resolved_regular_file(
        measured_dir / f"{category}.csv", "measured category CSV"
    )
    measured_data = measured_cache.get(measured_path)
    if measured_data is None:
        measured_data = stable_bytes(measured_path, "measured category CSV")
        measured_cache[measured_path] = measured_data
    bank_row = locate_bank_row(
        measured_path=measured_path,
        data=measured_data,
        category=category,
        occurrence=evidence.occurrence,
        official_reference=official.context.reference_order[category],
    ).fields
    if version == 1:
        bank_state = _validate_legacy_measured_binding(
            row_key=row_key,
            value=raw_entry.get("measured_csv"),
            measured_path=measured_path,
            current_row=bank_row,
            evidence=evidence,
        )
    else:
        bank_state = _validate_current_measured_binding(
            row_key=row_key,
            value=raw_entry.get("measured_csv"),
            measured_path=measured_path,
            current_row=bank_row,
            evidence=evidence,
        )
    return ValidatedIndexEntry(
        row_key=row_key,
        entry=raw_entry,
        evidence=evidence,
        measured_path=measured_path,
        bank_row=bank_row,
        bank_state=bank_state,
        legacy_entry=version < 4,
    )


def validate_regular_evidence_index(
    *,
    evidence_index: Path,
    benchmark_root: Path,
    official_results: Path,
    measured_dir: Path,
    allow_missing: bool = False,
    pinned_official: PinnedOfficialResults | None = None,
    pinned_benchmark: PinnedOfficialBenchmark | None = None,
    replay_session: dict[str, Any] | None = None,
) -> ValidatedEvidenceIndex:
    """Read, fully revalidate, and reconcile every indexed promotion."""

    index_path = evidence_index.absolute()
    if index_path.is_symlink():
        raise EvidenceError(f"evidence index must not be a symlink: {index_path}")
    if not index_path.exists() and not allow_missing:
        raise EvidenceError(f"evidence index is unavailable: {index_path}")
    benchmark = resolved_directory(benchmark_root, "benchmark root")
    official_benchmark = pinned_benchmark or validate_official_benchmark(benchmark)
    if official_benchmark.benchmark_root != benchmark:
        raise EvidenceError("caller benchmark root differs from pinned benchmark root")
    measured = resolved_directory(measured_dir, "measured bank directory")
    official = pinned_official or validate_official_results(official_results)
    requested_official = resolved_directory(official_results, "official result root")
    if requested_official != official.root:
        raise EvidenceError("caller official root differs from pinned official root")
    if not index_path.exists():
        value = {"schema": INDEX_SCHEMA, "entries": {}}
        return ValidatedEvidenceIndex(index_path, False, None, value, official, ())
    resolved_index = resolved_regular_file(index_path, "evidence index")
    data = stable_bytes(resolved_index, "evidence index")
    value = _json_object(data, path=resolved_index, label="evidence index")
    if (
        set(value) != {"schema", "entries"}
        or value.get("schema") != INDEX_SCHEMA
        or not isinstance(value.get("entries"), dict)
    ):
        raise EvidenceError(f"unsupported evidence index schema: {resolved_index}")
    raw_entries = value["entries"]
    assert isinstance(raw_entries, dict)
    entries: list[ValidatedIndexEntry] = []
    measured_cache: dict[Path, bytes] = {}
    owns_replay_session = replay_session is None
    active_replay_session = replay_session if replay_session is not None else {}
    seen_runs: set[tuple[str, str]] = set()
    for row_key in sorted(raw_entries):
        entry = _validate_index_entry(
            row_key=row_key,
            raw_entry=raw_entries[row_key],
            index_path=resolved_index,
            benchmark_root=benchmark,
            measured_dir=measured,
            official=official,
            official_benchmark=official_benchmark,
            measured_cache=measured_cache,
            replay_session=active_replay_session,
        )
        run_identity = (
            str(entry.evidence.artifact_root),
            entry.evidence.run_id,
        )
        if run_identity in seen_runs:
            raise EvidenceError(
                "evidence index binds one sealed run to multiple bank rows"
            )
        seen_runs.add(run_identity)
        entries.append(entry)
    if stable_bytes(resolved_index, "evidence index") != data:
        raise EvidenceError("evidence index changed while it was validated")
    for measured_path, measured_data in measured_cache.items():
        if stable_bytes(measured_path, "measured category CSV") != measured_data:
            raise EvidenceError(
                f"measured category CSV changed while validated: {measured_path}"
            )
    revalidate_official_benchmark(official_benchmark)
    revalidate_official_results(official)
    if owns_replay_session:
        revalidate_replay_session(active_replay_session)
    return ValidatedEvidenceIndex(
        resolved_index,
        True,
        data,
        value,
        official,
        tuple(entries),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--official-results", type=Path, required=True)
    parser.add_argument("--measured-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-index",
        type=Path,
        help="defaults to <measured-dir>/regular_evidence_index.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    index = (
        args.evidence_index
        if args.evidence_index is not None
        else args.measured_dir / "regular_evidence_index.json"
    )
    try:
        validated = validate_regular_evidence_index(
            evidence_index=index,
            benchmark_root=args.benchmark_root,
            official_results=args.official_results,
            measured_dir=args.measured_dir,
        )
    except (EvidenceError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    summary = {
        "schema": INDEX_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "evidence_index": str(validated.path),
        "official_results": {
            "root": str(validated.official.root),
            **validated.official.identity,
        },
        "entries": len(validated.entries),
        "applied": len(validated.creditable_entries),
        "creditable_entries": len(validated.creditable_entries),
        "dangling": [
            entry.row_key
            for entry in validated.entries
            if entry.bank_state == "dangling"
        ],
        "status": "valid",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
