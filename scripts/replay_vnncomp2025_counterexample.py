#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Replay one sealed SAT result with the exact VNN-COMP 2025 ZERO-TOL checker.

This is a deliberately narrow evidence tool.  It accepts only canonical
VNN-COMP 2025 regular-track measurement metadata, obtains the model and
property bytes from the pinned official benchmark Git commit, executes the
organizer's exact ZERO-TOL ``get_ce_diff`` implementation in a retained,
exact-dependency runtime, and publishes one immutable validation sidecar.

The replay fails closed on any source, runtime, dependency, setting, provider,
artifact, row-identity, or metric mismatch.  It never edits the measurement
archive or translates the submitted assignment.

The retained environment is deliberately reported as a host-bound local
replay.  Its complete import trees and actually mapped native dependencies are
byte-pinned, but this evidence is not an official organizer attestation and
does not claim relocatability or independent external reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ny_measurement_provenance as provenance  # noqa: E402
import regular_bank_evidence as regular  # noqa: E402

SCHEMA = "ny_vnncomp2025_zero_tol_counterexample_validation_v2"
SCHEMA_VERSION = 2
SCORING_YEAR = 2025
CPU_PROVIDER = "CPUExecutionProvider"

OFFICIAL_RESULTS_REPOSITORY = "https://github.com/VNN-COMP/vnncomp2025_results"
OFFICIAL_RESULTS_COMMIT = "ea89fbc2518b6729f17c96eeec22c56c88e496a9"
OFFICIAL_SOURCE_SHA256 = {
    "SCORING-ZERO-TOL/counterexamples.py": (
        "4df1208bb08c1b589dc3f2ac098add44467cf538f7141a7641e7d13001e94e3b"
    ),
    "SCORING-ZERO-TOL/requirements.txt": (
        "e5892704893ef261302c6a2a46fe89eab90aa7a1afe4b6648ae1e1065f106afc"
    ),
    "SCORING-ZERO-TOL/settings.py": (
        "ceeefbd2498cb0a943ee2950440e40a517697cfa899f15a61092f846936256f1"
    ),
    "SCORING-ZERO-TOL/vnnlib.py": (
        "f1731b74d3c20419cf5a70a7065b772f585a1ba2f82a1bcd0b1fc7b5c8d3c576"
    ),
}

IGNORE_CE_Y = False
COUNTEREXAMPLE_ATOL = 1e-4
COUNTEREXAMPLE_RTOL = 1e-3
SCORING_ZERO_TOLERANCE = True

PINNED_REQUIREMENTS = {
    "cachier": "2.2.2",
    "coloredlogs": "15.0.1",
    "flatbuffers": "23.5.26",
    "humanfriendly": "10.0",
    "mpmath": "1.3.0",
    "numpy": "1.24.4",
    "onnx": "1.15.0",
    "onnxruntime": "1.16.3",
    "packaging": "23.2",
    "portalocker": "2.8.2",
    "protobuf": "4.25.1",
    "sympy": "1.12",
    "watchdog": "3.0.0",
}
PINNED_PYTHON_VERSION = "3.11.15"

PINNED_RUNTIME_ROOT = Path("<home>/ny-vnncomp2025-checker-exact-20260731T074000Z")
PINNED_BENCHMARK_REPOSITORY = Path("<home>/ay/benchmarks/vnncomp/2025/benchmarks")
PINNED_PYTHON_RELATIVE = Path("python-base/bin/python3.11")
PINNED_PYTHON_SHA256 = (
    "8deffe5dd9ebcf98a062917a4e73bb8fbb7d5846f83dec01fb7506fd5d41c54e"
)
PINNED_SITE_PACKAGES_RELATIVE = Path("lib/python3.11/site-packages")
PINNED_STDLIB_RELATIVE = Path("python-base/lib/python3.11")
PINNED_SCORING_RELATIVE = Path("results/SCORING-ZERO-TOL")
PINNED_HARNESS_RELATIVE = Path("harness")
PINNED_RETAINED_RUNNER_RELATIVE = (
    PINNED_HARNESS_RELATIVE / "replay_vnncomp2025_counterexample.py"
)
PINNED_WORKER_RELATIVE = PINNED_HARNESS_RELATIVE / "vnncomp2025_zero_tol_worker.py"
WORKER_SOURCE_PATH = SCRIPT_DIR / "vnncomp2025_zero_tol_worker.py"
WORKER_PROTOCOL = "ny_vnncomp2025_zero_tol_worker_v1"
PINNED_WORKER_SHA256 = (
    "001d1ac6af69e61fa108fe60d4589a54a850ad6bf1b7bd72ef5a428bc1410c63"
)
# Filled from deterministic, read-only retained trees.  These are deliberately
# constants rather than self-attested values from the child process.
PINNED_STDLIB_MANIFEST_SHA256 = (
    "63fc594de67acb2834d34aa8c1c123bcb9d2550341e35a67bcf41b720a6f3d33"
)
PINNED_SITE_PACKAGES_MANIFEST_SHA256 = (
    "99ac4a2ea01890f7fac33e61151a9160d2ca8918c25ce216cac1ceec72d641d3"
)
PINNED_SCORING_MANIFEST_SHA256 = (
    "2a1526c45d1b7abd4016c7a2a9f8ef31e70d6ad6b446751b57d2a4d0d801a050"
)
PINNED_NATIVE_DEPENDENCIES: dict[str, str] = {
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/google/_upb/_message.abi3.so"
    ): "7c036740b38bb69727e9b6e8bce8ccdb53cf7afd25a14eee282249178b7fb3f0",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy.libs/"
        "libgfortran-040039e1.so.5.0.0"
    ): "47ab3b68295b0a3ce8990a448de7fab11abddbc160f8895972ca9aa712cf86d0",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy.libs/"
        "libopenblas64_p-r0-15028c96.3.21.so"
    ): "554bde1d8a0c71d8dc21ae74de05c44da4fff5dbc6791a819f6acf5adfe90bd9",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy.libs/"
        "libquadmath-96973f99.so.0.0.0"
    ): "97cda85ddb5163e2da6e1edb4e1d6b557833a99a40eda079ae37e5039465b65d",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/core/"
        "_multiarray_tests.cpython-311-x86_64-linux-gnu.so"
    ): "e95222a0370c0a6d568370cf333e663bb68f9bf739340b42271ec962aea38490",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/core/"
        "_multiarray_umath.cpython-311-x86_64-linux-gnu.so"
    ): "db283346a443cde0c8cb4dcdde56490714d2c34e3ad16fe2c30ae8c20458fc86",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/fft/"
        "_pocketfft_internal.cpython-311-x86_64-linux-gnu.so"
    ): "4cd2ced815ccc0eeeb34d54b8c339b116923f681ce3ed10f6b8ca4c70a7cd968",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/linalg/"
        "_umath_linalg.cpython-311-x86_64-linux-gnu.so"
    ): "77d989daf30e91b14b657f0a125377d1c5bf49d7d950522baefdce9a7a93f927",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/random/"
        "_bounded_integers.cpython-311-x86_64-linux-gnu.so"
    ): "79ac960abf9a90df7b2abd4aaf940e6d00b9373e7a84c05233ae460a3e2ebd42",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/random/"
        "_common.cpython-311-x86_64-linux-gnu.so"
    ): "3983ebad61e09b23694cedb76338dda942929cca53ef52c6c5aa603ebc68eda9",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/random/"
        "_generator.cpython-311-x86_64-linux-gnu.so"
    ): "c6dd2c1fffaf9472da1a91ba5f75f0e7cf0e02b5f398111fb75573d4a0fec1dc",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/random/"
        "_mt19937.cpython-311-x86_64-linux-gnu.so"
    ): "815add55349e343ccaad24d16ee2d5e194f67289f6fd91076f70e316c6597970",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/random/"
        "_pcg64.cpython-311-x86_64-linux-gnu.so"
    ): "193c2b35096fb4cde565d92fb7419231e4bd67bf3d3cef8e2a852e3b1bac52ca",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/random/"
        "_philox.cpython-311-x86_64-linux-gnu.so"
    ): "e39409351843c69d08af2face384590621449609e037117288cfc01c9af86f5b",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/random/"
        "_sfc64.cpython-311-x86_64-linux-gnu.so"
    ): "bebe3d679912104185f75ef552ab2ee2dbcaf0bd09184c10bfc894b7ddeee6db",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/random/"
        "bit_generator.cpython-311-x86_64-linux-gnu.so"
    ): "6c7aabac046ac4dc000e37a14520d0f67b9c4731ee527624cdd23dc966d43639",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/numpy/random/"
        "mtrand.cpython-311-x86_64-linux-gnu.so"
    ): "25801b4d3c8377e122ed1c78a5626d290480d84f7bee34061a9bf0a52f7d933d",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/onnx/"
        "onnx_cpp2py_export.cpython-311-x86_64-linux-gnu.so"
    ): "e2acb8e1de3f23d3b9cd9a3e4aef096659a8bd0868a4dd18b3fecbc522b89858",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/onnxruntime/capi/"
        "libonnxruntime_providers_shared.so"
    ): "162d10127a6d3d1ee0ceb704da3f4082d86e80ea40a19709f0e5e15a56eae935",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "lib/python3.11/site-packages/onnxruntime/capi/"
        "onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so"
    ): "01de43d014e80f3e53d131eae6b79e862f161fd8ba6dc1e22f39c7467ce7a692",
    (
        "<home>/ny-vnncomp2025-checker-exact-20260731T074000Z/"
        "python-base/bin/python3.11"
    ): "8deffe5dd9ebcf98a062917a4e73bb8fbb7d5846f83dec01fb7506fd5d41c54e",
    "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2": (
        "c5e80a563850d6ab5c2f2482e4202d9c1b71fbf44854b8c399e63527202c64e1"
    ),
    "/usr/lib/x86_64-linux-gnu/libc.so.6": (
        "a3947513a02831ec692ebf13053c07614882ab54a2101fb91a1b15724062ed0c"
    ),
    "/usr/lib/x86_64-linux-gnu/libdl.so.2": (
        "7d293f8361fcead4f9691561adc0413f724f3607b959abe0d4fb243072956079"
    ),
    "/usr/lib/x86_64-linux-gnu/libgcc_s.so.1": (
        "9d339ecb409578d6a5d587e6c537a8f9589b8a13fefba30d167433a4b5758bee"
    ),
    "/usr/lib/x86_64-linux-gnu/libm.so.6": (
        "beea4eeacfcfa2cd96011b959a826c97cf4a774017e214f6a34d7eea3d49cd88"
    ),
    "/usr/lib/x86_64-linux-gnu/libpthread.so.0": (
        "f93acb6e78dcf0213c8a85f922d21916249148e24de426079c40b6304c42085d"
    ),
    "/usr/lib/x86_64-linux-gnu/librt.so.1": (
        "5df2508a1ef33bd8024d77271e2a4a2607cb63e59dd753dd46f8e7a2ed44962d"
    ),
    "/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.35": (
        "5bb0d21308f123b6ad46c6f35b42cedfcb8d6d439a53aa3dae04d880aaffdde3"
    ),
    "/usr/lib/x86_64-linux-gnu/libutil.so.1": (
        "ad58c7fed81a532338c68548b5e4df9762b83bef541390761e1220f17e4bd47a"
    ),
    "/usr/lib/x86_64-linux-gnu/libz.so.1.3.1": (
        "fbf56b0e59287033b6579bbbeae2f9de2fe86ad5bf2bd44d44aad67a15109318"
    ),
}
ORT_PYBIND_RELATIVE = Path(
    "lib/python3.11/site-packages/onnxruntime/capi/"
    "onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so"
)
ORT_UPSTREAM_RELATIVE = Path(
    "provenance/onnxruntime_pybind11_state.cpython-311-x86_64-linux-gnu.so.upstream"
)
ORT_UPSTREAM_SHA256 = "22d666c3c24b9efb4d98c8d4d810960014fab055a6deba77f5dda74e3334845c"
ORT_PATCHED_SHA256 = "01de43d014e80f3e53d131eae6b79e862f161fd8ba6dc1e22f39c7467ce7a692"

PINNED_TOOL_ROOT = Path("<home>/ny-vnncomp2025-checker-tools-20260731T074000Z")
PINNED_PATCHELF_RELATIVE = Path("extracted/usr/bin/patchelf")
PINNED_PATCHELF_SHA256 = (
    "f8a2b35d3c22a19343e04c62a36ba3ceba9e6c91641615cb7248a9e82dc0e081"
)
PINNED_PATCHELF_DEB_RELATIVE = Path("patchelf_0.18.0-1.4build1_amd64.deb")
PINNED_PATCHELF_DEB_SHA256 = (
    "dd6cde91e0a77a73335a93a4ce41801f21dac36d2158539093c241e46e11b9fc"
)

TOP_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "validated_at_utc",
        "status",
        "classification",
        "official_result",
        "rationale",
        "score_credit",
        "scoring_year",
        "settings",
        "checker",
        "harness",
        "worker_receipt",
        "runtime",
        "measurement",
        "evidence",
        "response",
    }
)
SETTINGS_KEYS = frozenset(
    {
        "ignore_ce_y",
        "counterexample_atol",
        "counterexample_rtol",
        "scoring_zero_tolerance",
    }
)
CHECKER_KEYS = frozenset({"repository", "commit", "source_sha256"})
HARNESS_KEYS = frozenset({"runner_sha256", "worker_sha256", "protocol", "import_roots"})
WORKER_RECEIPT_KEYS = frozenset(
    {
        "protocol",
        "request_sha256",
        "onnx",
        "vnnlib",
        "counterexample",
        "response_sha256",
        "native_dependencies_sha256",
    }
)
WORKER_FILE_RECEIPT_KEYS = frozenset({"sha256", "size_bytes"})
RUNTIME_KEYS = frozenset(
    {
        "python_executable",
        "python_sha256",
        "python_version",
        "venv",
        "execution_scope",
        "requirements_sha256",
        "installed_versions",
        "onnxruntime_version",
        "provider",
        "stdlib_manifest_sha256",
        "site_packages_manifest_sha256",
        "scoring_tree_manifest_sha256",
        "native_dependencies",
        "ort_pybind_upstream_sha256",
        "ort_pybind_patched_sha256",
        "execstack_patch",
    }
)
EXECSTACK_PATCH_KEYS = frozenset(
    {
        "tool",
        "tool_version",
        "operation",
        "changed_byte_count",
        "before_gnu_stack",
        "after_gnu_stack",
    }
)
MEASUREMENT_KEYS = frozenset({"run_id", "category", "instance_index"})
EVIDENCE_KEYS = frozenset(
    {
        "metadata",
        "raw_result",
        "extracted_assignment",
        "start_manifest",
        "onnx",
        "vnnlib",
    }
)
FILE_LINK_KEYS = frozenset({"artifact", "sha256", "size_bytes"})
ASSIGNMENT_KEYS = frozenset({"sha256", "size_bytes", "transformation"})
GIT_INPUT_BINDING_KEYS = frozenset(
    {"sha256", "size_bytes", "official_git_path", "official_git_blob"}
)
RETAINED_INPUT_BINDING_KEYS = frozenset(
    {"sha256", "size_bytes", "official_retained_setup_payload"}
)
RESPONSE_KEYS = frozenset({"result", "message", "diff", "rel_error"})
REPLAY_SNAPSHOT_KEYS = frozenset({"harness", "runtime"})

METADATA_INPUT_KEYS = frozenset(
    {
        "declared_path",
        "hash_cache_hit",
        "hash_cache_key",
        "resolved_path",
        "sha256",
        "size_bytes",
    }
)
SEALED_INPUT_KEYS = frozenset(
    {
        "artifact",
        "fingerprint",
        "mode",
        "resolved_path",
        "sha256",
        "size_bytes",
    }
)
COUNTEREXAMPLE_STATE_KEYS = frozenset({"checker", "status"})

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
METRIC_RE = re.compile(
    r"L-inf norm difference between onnx execution and CE file output:"
    r"\s*([^;\n]+?)\s*\(rel error:\s*([^)]+?)\s*\);"
)
KNOWN_RESULTS = frozenset(
    {
        "correct",
        "correct_up_to_tolerance",
        "no_ce",
        "exec_doesnt_match",
        "spec_not_violated",
        "wrong_shape",
    }
)
CREDIT_RESULTS = frozenset({"correct", "correct_up_to_tolerance"})


class ReplayError(RuntimeError):
    """An exact replay precondition or invariant failed."""


@dataclass(frozen=True)
class FileEvidence:
    path: Path
    sha256: str
    size_bytes: int
    fingerprint: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class InputEvidence:
    authoritative: regular.AuthoritativeInput
    payload: bytes
    original: FileEvidence
    sealed: FileEvidence


@dataclass(frozen=True)
class ArchiveEvidence:
    artifact_root: Path
    metadata_path: Path
    metadata: dict[str, Any]
    metadata_file: FileEvidence
    result_file: FileEvidence
    result_bytes: bytes
    assignment_bytes: bytes
    start_file: FileEvidence
    start: dict[str, Any]
    benchmark: regular.PinnedOfficialBenchmark
    official: regular.PinnedOfficialResults
    occurrence: Any
    onnx: InputEvidence
    vnnlib: InputEvidence


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: object) -> str:
    data = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(data)


def _exact_object(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ReplayError(
            f"{label} does not have the exact canonical keys: {observed!r}"
        )
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReplayError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ReplayError(f"evidence path is not a regular file: {path}")
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _stable_read(path: Path, label: str) -> tuple[bytes, FileEvidence]:
    if path.is_symlink():
        raise ReplayError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReplayError(f"{label} is unavailable: {path}") from error
    before = _fingerprint(resolved)
    try:
        data = resolved.read_bytes()
    except OSError as error:
        raise ReplayError(f"could not read {label}: {resolved}") from error
    after = _fingerprint(resolved)
    if before != after:
        raise ReplayError(f"{label} changed while it was read: {resolved}")
    return data, FileEvidence(resolved, _sha256(data), len(data), after)


def _tree_manifest_sha256(root: Path, label: str) -> str:
    """Hash every retained import-root entry, byte identity, mode, and type."""

    if root.is_symlink():
        raise ReplayError(f"{label} root must not be a symlink")
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ReplayError(f"{label} root is unavailable") from error
    if not resolved.is_dir() or resolved.stat().st_mode & 0o222:
        raise ReplayError(f"{label} root is not a read-only directory")
    entries: list[dict[str, Any]] = []
    try:
        paths = sorted(resolved.rglob("*"), key=lambda path: path.as_posix())
    except OSError as error:
        raise ReplayError(f"could not inventory {label}") from error
    for path in paths:
        relative = path.relative_to(resolved).as_posix()
        if (
            "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
            or path.is_symlink()
        ):
            raise ReplayError(f"{label} contains an unsafe cache/symlink: {relative}")
        try:
            info = path.stat()
        except OSError as error:
            raise ReplayError(f"could not stat {label} entry {relative}") from error
        mode = stat.S_IMODE(info.st_mode)
        if mode & 0o222:
            raise ReplayError(f"{label} entry is writable: {relative}")
        if stat.S_ISDIR(info.st_mode):
            entries.append({"kind": "directory", "mode": mode, "path": relative})
        elif stat.S_ISREG(info.st_mode):
            _, evidence = _stable_read(path, f"{label} entry {relative}")
            entries.append(
                {
                    "kind": "file",
                    "mode": mode,
                    "path": relative,
                    "sha256": evidence.sha256,
                    "size_bytes": evidence.size_bytes,
                }
            )
        else:
            raise ReplayError(f"{label} contains unsupported file type: {relative}")
    if not entries:
        raise ReplayError(f"{label} manifest is empty")
    data = json.dumps(
        entries,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(data)


def _same_file(before: FileEvidence, label: str) -> None:
    _, after = _stable_read(before.path, label)
    if before != after:
        raise ReplayError(f"{label} changed during official replay")


def _check_no_symlink_components(path: Path, root: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ReplayError(f"{label} escapes the artifact root") from error
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ReplayError(f"{label} contains an unsafe path component")
    current = root
    for part in relative.parts:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise ReplayError(f"{label} traverses a symlink: {current}")
        except OSError as error:
            raise ReplayError(f"{label} is unavailable: {current}") from error
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        raise ReplayError(f"{label} resolves outside its required root") from error


def _artifact_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ReplayError(f"unsafe {label} artifact path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ReplayError(f"unsafe {label} artifact path: {value!r}")
    candidate = root.joinpath(*relative.parts)
    _check_no_symlink_components(candidate, root, label)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ReplayError(f"{label} artifact escapes its root") from error
    return resolved


def _file_link(path: Path, root: Path, evidence: FileEvidence) -> dict[str, Any]:
    return {
        "artifact": path.relative_to(root).as_posix(),
        "sha256": evidence.sha256,
        "size_bytes": evidence.size_bytes,
    }


def _extract_assignment(result: bytes) -> bytes:
    lines = result.splitlines(keepends=True)
    if not lines or lines[0].strip() != b"sat":
        raise ReplayError("raw result must start with the exact SAT verdict")
    assignment = b"".join(lines[1:])
    if not assignment.strip():
        raise ReplayError("raw SAT result has no submitted assignment")
    try:
        text = assignment.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReplayError("submitted assignment is not UTF-8") from error
    if "\0" in text:
        raise ReplayError("submitted assignment contains a NUL byte")
    return assignment


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ReplayError(f"{label} is not numeric")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as error:
        raise ReplayError(f"{label} is not numeric") from error
    if not parsed.is_finite() or parsed < 0:
        raise ReplayError(f"{label} is not finite and nonnegative")
    return parsed


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReplayError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ReplayError(f"non-finite JSON constant: {value}")


def _json_loads(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except ReplayError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ReplayError(f"{label} is not strict JSON") from error


def _validate_effective_timing(
    *,
    metadata: dict[str, Any],
    measurement: dict[str, Any],
    official_timeout: Decimal,
) -> Decimal:
    timeout_cap = _decimal(measurement.get("timeout_cap_seconds"), "start timeout cap")
    if timeout_cap <= 0:
        raise ReplayError("start timeout cap must be positive")
    effective_timeout = min(timeout_cap, official_timeout)
    recorded_timeout = _decimal(metadata.get("timeout_seconds"), "metadata timeout")
    elapsed = _decimal(metadata.get("elapsed_seconds"), "metadata elapsed time")
    if recorded_timeout != effective_timeout:
        raise ReplayError(
            "metadata timeout differs from the effective official/capped budget"
        )
    if elapsed > recorded_timeout or elapsed > official_timeout:
        raise ReplayError("metadata records an over-budget SAT result")
    return effective_timeout


def _validate_declared_input(
    *,
    root: Path,
    metadata: dict[str, Any],
    occurrence: Any,
    benchmark: regular.PinnedOfficialBenchmark,
    category: str,
    label: str,
) -> InputEvidence:
    declared = occurrence.onnx if label == "onnx" else occurrence.vnnlib
    original_record = _exact_object(
        metadata.get(label), METADATA_INPUT_KEYS, f"metadata {label}"
    )
    execution_inputs = metadata.get("execution_inputs")
    if not isinstance(execution_inputs, dict) or set(execution_inputs) != {
        "onnx",
        "vnnlib",
    }:
        raise ReplayError("metadata execution_inputs is not canonical")
    sealed_record = _exact_object(
        execution_inputs.get(label),
        SEALED_INPUT_KEYS,
        f"sealed {label}",
    )
    if (
        original_record.get("declared_path") != declared
        or not isinstance(original_record.get("hash_cache_hit"), bool)
        or SHA256_RE.fullmatch(str(original_record.get("hash_cache_key"))) is None
    ):
        raise ReplayError(f"metadata {label} declared/cache identity differs")

    authoritative, payload = regular.authoritative_benchmark_input(
        benchmark=benchmark,
        category=category,
        declared_name=declared,
        label=label,
    )
    expected_original = (benchmark.benchmark_root / category).joinpath(
        *PurePosixPath(declared).parts
    )
    if original_record.get("resolved_path") != str(expected_original):
        raise ReplayError(f"metadata {label} path differs from the official row")
    original_data, original = _stable_read(expected_original, f"original {label}")

    sealed_path = _artifact_file(root, sealed_record.get("artifact"), f"sealed {label}")
    if sealed_record.get("resolved_path") != str(sealed_path):
        raise ReplayError(f"sealed {label} resolved/artifact paths differ")
    sealed_data, sealed = _stable_read(sealed_path, f"sealed {label}")
    if sealed_record.get("mode") != "read_only":
        raise ReplayError(f"sealed {label} does not assert read-only mode")
    if sealed_path.stat().st_mode & 0o222:
        raise ReplayError(f"sealed {label} is writable")
    try:
        expected_fingerprint = provenance._file_fingerprint(sealed_path)
    except OSError as error:
        raise ReplayError(f"could not fingerprint sealed {label}") from error
    if sealed_record.get("fingerprint") != expected_fingerprint:
        raise ReplayError(f"sealed {label} fingerprint differs")

    for record_label, record, observed in (
        ("metadata", original_record, original),
        ("sealed", sealed_record, sealed),
    ):
        digest = _require_sha256(
            record.get("sha256"), f"{record_label} {label} SHA-256"
        )
        size = record.get("size_bytes")
        if (
            type(size) is not int
            or size < 0
            or digest != observed.sha256
            or size != observed.size_bytes
        ):
            raise ReplayError(f"{record_label} {label} bytes differ")
    if original_data != payload or sealed_data != payload:
        raise ReplayError(
            f"{label} bytes differ from the authoritative pinned Git payload"
        )
    return InputEvidence(authoritative, payload, original, sealed)


def _load_archive(
    *,
    metadata_path: Path,
    artifact_root: Path,
    benchmark_root: Path,
    official_results: Path,
) -> ArchiveEvidence:
    if artifact_root.is_symlink():
        raise ReplayError(f"artifact root must not be a symlink: {artifact_root}")
    try:
        root = artifact_root.resolve(strict=True)
    except OSError as error:
        raise ReplayError(f"artifact root is unavailable: {artifact_root}") from error
    if not root.is_dir() or root.is_symlink():
        raise ReplayError(f"artifact root is not a canonical directory: {root}")

    if not metadata_path.is_absolute():
        raise ReplayError("metadata path must be absolute")
    _check_no_symlink_components(metadata_path, root, "metadata")
    metadata_data, metadata_file = _stable_read(metadata_path, "metadata")
    metadata = _json_loads(metadata_data, "metadata")
    try:
        regular.validate_metadata_schema_profile(metadata)
    except regular.EvidenceError as error:
        raise ReplayError(str(error)) from error
    assert isinstance(metadata, dict)
    if (
        metadata.get("schema") != "ny_measurement_result_v2"
        or metadata.get("schema_version") != 2
        or metadata.get("solver_verdict") != "sat"
        or metadata.get("solver_exit_status") != 0
        or metadata.get("witness_present") is not True
    ):
        raise ReplayError("metadata is not a canonical successful SAT result")
    counterexample = _exact_object(
        metadata.get("counterexample_validation"),
        COUNTEREXAMPLE_STATE_KEYS,
        "counterexample state",
    )
    if counterexample != {"checker": None, "status": "not_checked"}:
        raise ReplayError("SAT metadata is not in the unvalidated state")

    run_id = metadata.get("run_id")
    category = metadata.get("category")
    instance_index = metadata.get("instance_index")
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise ReplayError("metadata run ID is invalid")
    if not isinstance(category, str) or category not in regular.retro.REGULAR:
        raise ReplayError("metadata category is not a 2025 regular category")
    if type(instance_index) is not int or instance_index <= 0:
        raise ReplayError("metadata instance index is invalid")

    result_path = _artifact_file(root, metadata.get("result_artifact"), "raw result")
    result_data, result_file = _stable_read(result_path, "raw result")
    for key in ("result_sha256", "raw_result_sha256"):
        if _require_sha256(metadata.get(key), key) != result_file.sha256:
            raise ReplayError(f"{key} does not bind the raw result")
    assignment = _extract_assignment(result_data)

    start_path = _artifact_file(root, metadata.get("start_manifest"), "start manifest")
    start_data, start_file = _stable_read(start_path, "start manifest")
    if (
        _require_sha256(metadata.get("start_manifest_sha256"), "start manifest SHA-256")
        != start_file.sha256
    ):
        raise ReplayError("metadata does not bind the start manifest")
    start = _json_loads(start_data, "start manifest")
    try:
        regular.validate_flight_record_binding(start=start, metadata=metadata)
    except regular.EvidenceError as error:
        raise ReplayError(str(error)) from error
    assert isinstance(start, dict)
    if (
        start.get("schema") != "ny_measurement_start_v1"
        or start.get("run_id") != run_id
    ):
        raise ReplayError("start manifest identity differs from metadata")

    try:
        official = regular.validate_official_results(official_results)
        benchmark = regular.validate_official_benchmark(benchmark_root)
        occurrence, _ = regular._load_occurrence(
            category=category,
            instance_index=instance_index,
            benchmark=benchmark,
            official=official,
        )
    except regular.EvidenceError as error:
        raise ReplayError(str(error)) from error

    measurement = start.get("measurement")
    assert isinstance(measurement, dict)
    benchmark_identity = _exact_object(
        start.get("benchmark"),
        regular.BENCHMARK_WORKTREE_KEYS,
        "start benchmark",
    )
    if (
        measurement.get("categories") != [category]
        or measurement.get("instance_index") != instance_index
        or measurement.get("benchmark_root") != str(benchmark.benchmark_root)
        or benchmark_identity.get("benchmark_root") != str(benchmark.benchmark_root)
    ):
        raise ReplayError("start selection differs from the official occurrence")
    _validate_effective_timing(
        metadata=metadata,
        measurement=measurement,
        official_timeout=occurrence.timeout_seconds,
    )

    onnx = _validate_declared_input(
        root=root,
        metadata=metadata,
        occurrence=occurrence,
        benchmark=benchmark,
        category=category,
        label="onnx",
    )
    vnnlib = _validate_declared_input(
        root=root,
        metadata=metadata,
        occurrence=occurrence,
        benchmark=benchmark,
        category=category,
        label="vnnlib",
    )
    return ArchiveEvidence(
        artifact_root=root,
        metadata_path=metadata_file.path,
        metadata=metadata,
        metadata_file=metadata_file,
        result_file=result_file,
        result_bytes=result_data,
        assignment_bytes=assignment,
        start_file=start_file,
        start=start,
        benchmark=benchmark,
        official=official,
        occurrence=occurrence,
        onnx=onnx,
        vnnlib=vnnlib,
    )


def _checker_identity(official_results: Path, runtime_root: Path) -> dict[str, Any]:
    try:
        root = regular.resolved_directory(official_results, "official result root")
        repository = Path(
            regular._git_text(root, "rev-parse", "--show-toplevel").strip()
        ).resolve(strict=True)
        commit = regular._git_text(repository, "rev-parse", "HEAD").strip()
        origin = regular._git_text(repository, "remote", "get-url", "origin").strip()
        status = regular._git_text(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        )
    except (OSError, regular.EvidenceError) as error:
        raise ReplayError(
            f"could not validate official checker Git: {error}"
        ) from error
    if (
        repository != root
        or commit != OFFICIAL_RESULTS_COMMIT
        or origin != OFFICIAL_RESULTS_REPOSITORY
        or status
    ):
        raise ReplayError(
            "official checker repository is not the clean pinned 2025 release"
        )

    runtime_scoring = runtime_root / "results" / "SCORING-ZERO-TOL"
    observed: dict[str, str] = {}
    for relative, expected in OFFICIAL_SOURCE_SHA256.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        worktree_data, worktree = _stable_read(path, f"official {relative}")
        try:
            committed = regular._git(
                repository,
                "show",
                f"{OFFICIAL_RESULTS_COMMIT}:{relative}",
            )
        except regular.EvidenceError as error:
            raise ReplayError(str(error)) from error
        assert committed is not None
        runtime_path = runtime_scoring / PurePosixPath(relative).name
        runtime_data, _ = _stable_read(runtime_path, f"retained checker {relative}")
        if (
            worktree.sha256 != expected
            or _sha256(committed) != expected
            or runtime_data != worktree_data
        ):
            raise ReplayError(f"official checker source mismatch: {relative}")
        observed[relative] = expected
    return {
        "repository": OFFICIAL_RESULTS_REPOSITORY,
        "commit": OFFICIAL_RESULTS_COMMIT,
        "source_sha256": observed,
    }


def _harness_identity(runtime_root: Path) -> dict[str, Any]:
    current_runner = Path(__file__)
    if current_runner.is_symlink():
        raise ReplayError("executing replay runner must not be a symlink")
    current_runner_data, current_runner_file = _stable_read(
        current_runner, "executing replay runner"
    )
    retained_runner = runtime_root / PINNED_RETAINED_RUNNER_RELATIVE
    retained_runner_data, retained_runner_file = _stable_read(
        retained_runner, "retained replay runner"
    )
    source_worker_data, source_worker = _stable_read(
        WORKER_SOURCE_PATH, "workspace worker source"
    )
    retained_worker = runtime_root / PINNED_WORKER_RELATIVE
    retained_worker_data, retained_worker_file = _stable_read(
        retained_worker, "retained worker"
    )
    harness_root = runtime_root / PINNED_HARNESS_RELATIVE
    if (
        harness_root.is_symlink()
        or not harness_root.is_dir()
        or harness_root.stat().st_mode & 0o222
        or retained_runner.stat().st_mode & 0o222
        or retained_worker.stat().st_mode & 0o222
    ):
        raise ReplayError("retained harness paths are writable")
    if (
        current_runner_data != retained_runner_data
        or current_runner_file.sha256 != retained_runner_file.sha256
    ):
        raise ReplayError(
            "executing replay runner differs from the retained pinned producer"
        )
    if (
        source_worker_data != retained_worker_data
        or source_worker.sha256 != PINNED_WORKER_SHA256
        or retained_worker_file.sha256 != PINNED_WORKER_SHA256
    ):
        raise ReplayError("worker source differs from the retained pinned worker")
    return {
        "runner_sha256": current_runner_file.sha256,
        "worker_sha256": PINNED_WORKER_SHA256,
        "protocol": WORKER_PROTOCOL,
        "import_roots": [
            str(runtime_root / PINNED_SCORING_RELATIVE),
            str(runtime_root / PINNED_SITE_PACKAGES_RELATIVE),
            str(runtime_root / PINNED_STDLIB_RELATIVE),
        ],
    }


def _regular_file_with_hash(path: Path, digest: str, label: str) -> FileEvidence:
    data, evidence = _stable_read(path, label)
    if evidence.sha256 != digest:
        raise ReplayError(f"{label} SHA-256 differs from its retained identity")
    if not data:
        raise ReplayError(f"{label} is empty")
    return evidence


def _gnu_stack_flags(data: bytes, label: str) -> str:
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ReplayError(f"{label} is not a little-endian ELF64 file")
    phoff = int.from_bytes(data[32:40], "little")
    phentsize = int.from_bytes(data[54:56], "little")
    phnum = int.from_bytes(data[56:58], "little")
    if phentsize < 8 or phnum <= 0 or phoff + phentsize * phnum > len(data):
        raise ReplayError(f"{label} has an invalid program-header table")
    found: list[int] = []
    for index in range(phnum):
        offset = phoff + index * phentsize
        kind = int.from_bytes(data[offset : offset + 4], "little")
        if kind == 0x6474E551:  # PT_GNU_STACK
            found.append(int.from_bytes(data[offset + 4 : offset + 8], "little"))
    if len(found) != 1 or found[0] & ~0x7:
        raise ReplayError(f"{label} has no unique canonical GNU_STACK header")
    flags = found[0]
    return "".join(name for bit, name in ((4, "R"), (2, "W"), (1, "E")) if flags & bit)


def _runtime_probe(
    *, runtime_root: Path, python: Path, timeout_seconds: int = 60
) -> dict[str, Any]:
    worker = runtime_root / PINNED_WORKER_RELATIVE
    with tempfile.TemporaryDirectory(prefix="ny-vnncomp2025-probe-") as raw:
        try:
            result = subprocess.run(
                [str(python), "-I", "-S", "-B", str(worker), "--probe"],
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                cwd=raw,
                env=_worker_environment(Path(raw)),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ReplayError(f"retained runtime probe failed: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ReplayError(
            f"retained runtime probe exited {result.returncode}: {detail}"
        )
    return _strict_json_response(result.stdout, "retained runtime probe")


def _runtime_identity(runtime_root: Path) -> dict[str, Any]:
    try:
        root = runtime_root.resolve(strict=True)
    except OSError as error:
        raise ReplayError(f"retained runtime is unavailable: {runtime_root}") from error
    if root != PINNED_RUNTIME_ROOT or root.is_symlink() or not root.is_dir():
        raise ReplayError("checker runtime path differs from the pinned retained root")
    python = root / PINNED_PYTHON_RELATIVE
    _regular_file_with_hash(python, PINNED_PYTHON_SHA256, "embedded Python")
    site_packages = root / PINNED_SITE_PACKAGES_RELATIVE
    stdlib = root / PINNED_STDLIB_RELATIVE
    scoring = root / PINNED_SCORING_RELATIVE
    worker = root / PINNED_WORKER_RELATIVE
    python_zip = root / "python-base/lib/python311.zip"
    if site_packages.is_symlink() or not site_packages.is_dir():
        raise ReplayError("retained site-packages path is not canonical")
    if python_zip.exists() or python_zip.is_symlink():
        raise ReplayError("unexpected higher-priority Python stdlib zip exists")
    _regular_file_with_hash(worker, PINNED_WORKER_SHA256, "retained worker")
    stdlib_manifest = _tree_manifest_sha256(stdlib, "embedded stdlib")
    site_packages_manifest = _tree_manifest_sha256(
        site_packages, "retained site-packages"
    )
    scoring_manifest = _tree_manifest_sha256(scoring, "retained scoring source")
    if (
        stdlib_manifest != PINNED_STDLIB_MANIFEST_SHA256
        or site_packages_manifest != PINNED_SITE_PACKAGES_MANIFEST_SHA256
        or scoring_manifest != PINNED_SCORING_MANIFEST_SHA256
    ):
        raise ReplayError("retained import-root tree manifest differs")
    if not PINNED_NATIVE_DEPENDENCIES:
        raise ReplayError("pinned native-dependency closure is empty")
    for path_value, digest in PINNED_NATIVE_DEPENDENCIES.items():
        path = Path(path_value)
        if (
            not path.is_absolute()
            or str(path.resolve(strict=True)) != path_value
            or path.is_symlink()
        ):
            raise ReplayError(f"native dependency path is not canonical: {path}")
        _regular_file_with_hash(path, digest, f"native dependency {path}")

    requirements = root / "results" / "SCORING-ZERO-TOL" / "requirements.txt"
    requirements_data, requirements_file = _stable_read(
        requirements, "official requirements"
    )
    if (
        requirements_file.sha256
        != OFFICIAL_SOURCE_SHA256["SCORING-ZERO-TOL/requirements.txt"]
    ):
        raise ReplayError("retained requirements bytes differ")
    parsed_requirements: dict[str, str] = {}
    for raw in requirements_data.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ReplayError(f"requirement is not exactly pinned: {line!r}")
        name, version = line.split("==", 1)
        if name in parsed_requirements:
            raise ReplayError(f"duplicate checker requirement: {name}")
        parsed_requirements[name] = version
    if parsed_requirements != PINNED_REQUIREMENTS:
        raise ReplayError("retained checker requirements differ from exact pins")

    upstream_path = root / ORT_UPSTREAM_RELATIVE
    patched_path = root / ORT_PYBIND_RELATIVE
    upstream_data, _ = _stable_read(upstream_path, "upstream ORT pybind")
    patched_data, _ = _stable_read(patched_path, "patched ORT pybind")
    if (
        _sha256(upstream_data) != ORT_UPSTREAM_SHA256
        or _sha256(patched_data) != ORT_PATCHED_SHA256
        or len(upstream_data) != len(patched_data)
    ):
        raise ReplayError("retained ORT pybind identities differ")
    changed = sum(before != after for before, after in zip(upstream_data, patched_data))
    before_stack = _gnu_stack_flags(upstream_data, "upstream ORT pybind")
    after_stack = _gnu_stack_flags(patched_data, "patched ORT pybind")
    if (changed, before_stack, after_stack) != (1, "RWE", "RW"):
        raise ReplayError(
            "retained ORT execstack repair is not the exact one-byte patch"
        )

    tool_root = PINNED_TOOL_ROOT.resolve(strict=True)
    patchelf = tool_root / PINNED_PATCHELF_RELATIVE
    deb = tool_root / PINNED_PATCHELF_DEB_RELATIVE
    _regular_file_with_hash(patchelf, PINNED_PATCHELF_SHA256, "retained patchelf")
    _regular_file_with_hash(
        deb, PINNED_PATCHELF_DEB_SHA256, "retained patchelf package"
    )
    try:
        version = subprocess.run(
            [str(patchelf), "--version"],
            capture_output=True,
            check=False,
            timeout=30,
            env={"PATH": "/usr/bin:/bin"},
        )
        before = subprocess.run(
            [str(patchelf), "--print-execstack", str(upstream_path)],
            capture_output=True,
            check=False,
            timeout=30,
            env={"PATH": "/usr/bin:/bin"},
        )
        after = subprocess.run(
            [str(patchelf), "--print-execstack", str(patched_path)],
            capture_output=True,
            check=False,
            timeout=30,
            env={"PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReplayError(f"could not validate retained patchelf: {error}") from error
    if (
        version.returncode != 0
        or version.stdout.strip() != b"patchelf 0.18.0"
        or before.returncode != 0
        or before.stdout.strip() != b"execstack: X"
        or after.returncode != 0
        or after.stdout.strip() != b"execstack: -"
    ):
        raise ReplayError("retained patchelf cannot reproduce execstack identities")

    probe = _runtime_probe(runtime_root=root, python=python)
    expected_probe_keys = frozenset(
        {
            "python_executable",
            "python_version",
            "prefix",
            "base_prefix",
            "installed_versions",
            "onnxruntime_version",
            "available_providers",
            "ort_pybind_path",
            "sys_path",
        }
    )
    _exact_object(probe, expected_probe_keys, "runtime probe")
    expected_prefix = str(root / "python-base")
    if (
        probe.get("python_executable") != str(python)
        or probe.get("python_version") != PINNED_PYTHON_VERSION
        or probe.get("prefix") != expected_prefix
        or probe.get("base_prefix") != expected_prefix
        or probe.get("installed_versions") != PINNED_REQUIREMENTS
        or probe.get("onnxruntime_version") != PINNED_REQUIREMENTS["onnxruntime"]
        or probe.get("available_providers")
        != [
            "AzureExecutionProvider",
            CPU_PROVIDER,
        ]
        or probe.get("ort_pybind_path") != str(patched_path)
        or probe.get("sys_path")
        != [
            str(root / "python-base/lib/python311.zip"),
            str(stdlib),
            str(stdlib / "lib-dynload"),
            str(site_packages),
        ]
    ):
        raise ReplayError("retained Python/dependency/provider identity differs")

    benchmark_link = root / "vnncomp2025_benchmarks"
    if (
        not benchmark_link.is_symlink()
        or benchmark_link.resolve(strict=True) != PINNED_BENCHMARK_REPOSITORY
    ):
        raise ReplayError("retained benchmark path is not the pinned repository")
    return {
        "python_executable": str(python),
        "python_sha256": PINNED_PYTHON_SHA256,
        "python_version": PINNED_PYTHON_VERSION,
        "venv": str(root),
        "execution_scope": "host_bound_local_replay",
        "requirements_sha256": requirements_file.sha256,
        "installed_versions": dict(PINNED_REQUIREMENTS),
        "onnxruntime_version": PINNED_REQUIREMENTS["onnxruntime"],
        "provider": CPU_PROVIDER,
        "stdlib_manifest_sha256": stdlib_manifest,
        "site_packages_manifest_sha256": site_packages_manifest,
        "scoring_tree_manifest_sha256": scoring_manifest,
        "native_dependencies": dict(PINNED_NATIVE_DEPENDENCIES),
        "ort_pybind_upstream_sha256": ORT_UPSTREAM_SHA256,
        "ort_pybind_patched_sha256": ORT_PATCHED_SHA256,
        "execstack_patch": {
            "tool": "patchelf",
            "tool_version": "0.18.0",
            "operation": "--clear-execstack",
            "changed_byte_count": 1,
            "before_gnu_stack": "RWE",
            "after_gnu_stack": "RW",
        },
    }


def _worker_environment(temporary: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(temporary),
        "TMPDIR": str(temporary),
        "XDG_CACHE_HOME": str(temporary / "cache"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def _strict_json_response(data: bytes, label: str) -> dict[str, Any]:
    value = _json_loads(data, label)
    if not isinstance(value, dict):
        raise ReplayError(f"{label} response is not an object")
    return value


def _parse_official_metrics(message: str) -> tuple[float, float]:
    if not isinstance(message, str):
        raise ReplayError("official checker message is not a string")
    matches = METRIC_RE.findall(message)
    if len(matches) != 1:
        raise ReplayError("official checker message has no unique diff/rel_error")
    values: list[float] = []
    for label, raw in zip(("diff", "rel_error"), matches[0]):
        try:
            value = float(raw.strip())
        except ValueError as error:
            raise ReplayError(f"official {label} is not numeric") from error
        if not math.isfinite(value) or value < 0:
            raise ReplayError(f"official {label} is not finite and nonnegative")
        values.append(value)
    return values[0], values[1]


def _validate_machine_response(response: object) -> dict[str, Any]:
    value = _exact_object(response, RESPONSE_KEYS, "official machine response")
    result = value.get("result")
    message = value.get("message")
    if not isinstance(result, str) or result not in KNOWN_RESULTS:
        raise ReplayError(f"unknown official checker result: {result!r}")
    diff, rel_error = _parse_official_metrics(message)
    for label, observed, expected in (
        ("diff", value.get("diff"), diff),
        ("rel_error", value.get("rel_error"), rel_error),
    ):
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) != expected
        ):
            raise ReplayError(f"official {label} response/message mismatch")
    return {
        "result": result,
        "message": message,
        "diff": diff,
        "rel_error": rel_error,
    }


def _payload_receipt(data: bytes) -> dict[str, Any]:
    return {"sha256": _sha256(data), "size_bytes": len(data)}


def _validate_worker_receipt(
    value: object,
    *,
    onnx_payload: bytes,
    vnnlib_payload: bytes,
    assignment_bytes: bytes,
    response: dict[str, Any],
    native_dependencies: dict[str, str],
) -> dict[str, Any]:
    receipt = _exact_object(value, WORKER_RECEIPT_KEYS, "worker receipt")
    expected_inputs = {
        "onnx": _payload_receipt(onnx_payload),
        "vnnlib": _payload_receipt(vnnlib_payload),
        "counterexample": _payload_receipt(assignment_bytes),
    }
    for label, expected in expected_inputs.items():
        observed = _exact_object(
            receipt.get(label),
            WORKER_FILE_RECEIPT_KEYS,
            f"worker receipt {label}",
        )
        if observed != expected:
            raise ReplayError(f"worker receipt {label} differs from supplied bytes")
    request_binding = {
        "protocol": WORKER_PROTOCOL,
        "abs_tolerance": COUNTEREXAMPLE_ATOL,
        "rel_tolerance": COUNTEREXAMPLE_RTOL,
        **expected_inputs,
    }
    expected = {
        "protocol": WORKER_PROTOCOL,
        "request_sha256": _canonical_sha256(request_binding),
        **expected_inputs,
        "response_sha256": _canonical_sha256(response),
        "native_dependencies_sha256": _canonical_sha256(native_dependencies),
    }
    if receipt != expected:
        raise ReplayError("worker receipt hashes differ from the exact replay")
    return receipt


def _invoke_exact_worker(
    *,
    onnx_payload: bytes,
    vnnlib_payload: bytes,
    assignment_bytes: bytes,
    runtime_root: Path,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if timeout_seconds <= 0:
        raise ReplayError("checker timeout must be positive")
    python = runtime_root / PINNED_PYTHON_RELATIVE
    worker = runtime_root / PINNED_WORKER_RELATIVE
    with tempfile.TemporaryDirectory(prefix="ny-vnncomp2025-zero-tol-") as raw:
        temporary = Path(raw)
        model = temporary / "model.onnx"
        prop = temporary / "property.vnnlib"
        witness = temporary / "counterexample.txt"
        native_dependencies = temporary / "native-dependencies.json"
        receipt_path = temporary / "worker-receipt.json"
        model.write_bytes(onnx_payload)
        prop.write_bytes(vnnlib_payload)
        witness.write_bytes(assignment_bytes)
        for path in (model, prop, witness):
            path.chmod(0o444)
        request = {
            "protocol": WORKER_PROTOCOL,
            "onnx_path": str(model),
            "vnnlib_path": str(prop),
            "counterexample_path": str(witness),
            "native_dependencies_path": str(native_dependencies),
            "receipt_path": str(receipt_path),
            "abs_tolerance": COUNTEREXAMPLE_ATOL,
            "rel_tolerance": COUNTEREXAMPLE_RTOL,
        }
        try:
            result = subprocess.run(
                [
                    str(python),
                    "-I",
                    "-S",
                    "-B",
                    str(worker),
                    "--check",
                ],
                input=json.dumps(request, sort_keys=True, allow_nan=False).encode(
                    "utf-8"
                ),
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                cwd=temporary,
                env=_worker_environment(temporary),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ReplayError(f"official checker process failed: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip()
            raise ReplayError(
                f"official checker exited with status {result.returncode}: {detail}"
            )
        native_data, _ = _stable_read(
            native_dependencies, "worker native-dependency evidence"
        )
        native = _json_loads(native_data, "worker native dependencies")
        if native != PINNED_NATIVE_DEPENDENCIES:
            raise ReplayError(
                "actual worker native-dependency closure differs from the pin"
            )
        response = _validate_machine_response(
            _strict_json_response(result.stdout, "official checker")
        )
        receipt_data, _ = _stable_read(receipt_path, "worker receipt")
        receipt = _validate_worker_receipt(
            _json_loads(receipt_data, "worker receipt"),
            onnx_payload=onnx_payload,
            vnnlib_payload=vnnlib_payload,
            assignment_bytes=assignment_bytes,
            response=response,
            native_dependencies=native,
        )
        return response, receipt


def replay_bound_payloads(
    *,
    onnx_payload: bytes,
    vnnlib_payload: bytes,
    assignment_bytes: bytes,
    runtime_root: Path = PINNED_RUNTIME_ROOT,
    timeout_seconds: int = 600,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently replay exact bound bytes for downstream consumers.

    Evidence consumers must call this function with authoritative Git model
    and property payloads plus the exact assignment extracted from the sealed
    raw result.  They must compare all four returned objects with the sidecar;
    trusting a previously written response is intentionally insufficient.
    """

    for label, payload in (
        ("ONNX", onnx_payload),
        ("VNN-LIB", vnnlib_payload),
        ("counterexample", assignment_bytes),
    ):
        if not isinstance(payload, bytes) or not payload:
            raise ReplayError(f"bound {label} payload must be nonempty bytes")
    if timeout_seconds <= 0:
        raise ReplayError("checker timeout must be positive")
    root = _pinned_runtime_root(runtime_root)
    owns_snapshot = snapshot is None
    if snapshot is None:
        snapshot = capture_replay_snapshot(runtime_root=root)
    else:
        _validate_replay_snapshot_shape(snapshot)
    harness = snapshot["harness"]
    runtime = snapshot["runtime"]
    response, worker_receipt = _invoke_exact_worker(
        onnx_payload=onnx_payload,
        vnnlib_payload=vnnlib_payload,
        assignment_bytes=assignment_bytes,
        runtime_root=root,
        timeout_seconds=timeout_seconds,
    )
    if owns_snapshot:
        revalidate_replay_snapshot(snapshot, runtime_root=root)
    return {
        "harness": harness,
        "runtime": runtime,
        "response": response,
        "worker_receipt": worker_receipt,
    }


def _pinned_runtime_root(runtime_root: Path) -> Path:
    if runtime_root != PINNED_RUNTIME_ROOT or runtime_root.is_symlink():
        raise ReplayError(
            "checker runtime argument must be the exact pinned retained path"
        )
    return runtime_root.resolve(strict=True)


def _validate_replay_snapshot_shape(snapshot: object) -> dict[str, Any]:
    value = _exact_object(snapshot, REPLAY_SNAPSHOT_KEYS, "replay snapshot")
    _exact_object(value["harness"], HARNESS_KEYS, "replay snapshot harness")
    runtime = _exact_object(value["runtime"], RUNTIME_KEYS, "replay snapshot runtime")
    _exact_object(
        runtime["execstack_patch"],
        EXECSTACK_PATCH_KEYS,
        "replay snapshot execstack patch",
    )
    return value


def capture_replay_snapshot(
    *, runtime_root: Path = PINNED_RUNTIME_ROOT
) -> dict[str, Any]:
    """Hash/probe the host-bound replay environment once for a batch."""

    root = _pinned_runtime_root(runtime_root)
    snapshot = {
        "harness": _harness_identity(root),
        "runtime": _runtime_identity(root),
    }
    _validate_replay_snapshot_shape(snapshot)
    return snapshot


def revalidate_replay_snapshot(
    snapshot: dict[str, Any],
    *,
    runtime_root: Path = PINNED_RUNTIME_ROOT,
) -> None:
    """Fail if any pinned harness/runtime byte changed since capture."""

    expected = _validate_replay_snapshot_shape(snapshot)
    observed = capture_replay_snapshot(runtime_root=runtime_root)
    if observed != expected:
        raise ReplayError("host-bound replay snapshot changed during batch")


def _build_sidecar(
    *,
    archive: ArchiveEvidence,
    checker: dict[str, Any],
    harness: dict[str, Any],
    worker_receipt: dict[str, Any],
    runtime: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    def input_binding(evidence: InputEvidence) -> dict[str, Any]:
        authoritative = evidence.authoritative
        binding: dict[str, Any] = {
            "sha256": authoritative.sha256,
            "size_bytes": authoritative.size_bytes,
        }
        if authoritative.retained_setup_payload is None:
            if authoritative.git_path is None or authoritative.git_blob is None:
                raise ReplayError(
                    "authoritative Git input has no Git object identity"
                )
            binding.update(
                {
                    "official_git_path": authoritative.git_path,
                    "official_git_blob": authoritative.git_blob,
                }
            )
        else:
            if authoritative.git_path is not None or authoritative.git_blob is not None:
                raise ReplayError(
                    "retained authoritative input also claims a Git payload"
                )
            binding["official_retained_setup_payload"] = (
                authoritative.retained_setup_payload
            )
        return binding

    result = response["result"]
    score_credit = result in CREDIT_RESULTS
    sidecar = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "validated_at_utc": _utc_now(),
        "status": "validated",
        "classification": "valid" if score_credit else "invalid",
        "official_result": result,
        "rationale": response["message"],
        "score_credit": score_credit,
        "scoring_year": SCORING_YEAR,
        "settings": {
            "ignore_ce_y": IGNORE_CE_Y,
            "counterexample_atol": COUNTEREXAMPLE_ATOL,
            "counterexample_rtol": COUNTEREXAMPLE_RTOL,
            "scoring_zero_tolerance": SCORING_ZERO_TOLERANCE,
        },
        "checker": checker,
        "harness": harness,
        "worker_receipt": worker_receipt,
        "runtime": runtime,
        "measurement": {
            "run_id": archive.metadata["run_id"],
            "category": archive.metadata["category"],
            "instance_index": archive.metadata["instance_index"],
        },
        "evidence": {
            "metadata": _file_link(
                archive.metadata_path,
                archive.artifact_root,
                archive.metadata_file,
            ),
            "raw_result": _file_link(
                archive.result_file.path,
                archive.artifact_root,
                archive.result_file,
            ),
            "extracted_assignment": {
                "sha256": _sha256(archive.assignment_bytes),
                "size_bytes": len(archive.assignment_bytes),
                "transformation": "removed_standalone_sat_verdict_line_only",
            },
            "start_manifest": _file_link(
                archive.start_file.path,
                archive.artifact_root,
                archive.start_file,
            ),
            "onnx": input_binding(archive.onnx),
            "vnnlib": input_binding(archive.vnnlib),
        },
        "response": response,
    }
    _validate_sidecar_shape(sidecar)
    return sidecar


def _validate_input_binding(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReplayError(f"validation {label} input binding is not an object")
    keys = frozenset(value)
    if keys == GIT_INPUT_BINDING_KEYS:
        binding = _exact_object(
            value,
            GIT_INPUT_BINDING_KEYS,
            f"validation {label} Git input",
        )
        git_path = binding.get("official_git_path")
        if (
            not isinstance(git_path, str)
            or not git_path
            or PurePosixPath(git_path).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(git_path).parts)
            or GIT_OBJECT_RE.fullmatch(
                str(binding.get("official_git_blob"))
            )
            is None
        ):
            raise ReplayError(
                f"validation {label} Git source identity is invalid"
            )
    elif keys == RETAINED_INPUT_BINDING_KEYS:
        binding = _exact_object(
            value,
            RETAINED_INPUT_BINDING_KEYS,
            f"validation {label} retained input",
        )
        source = binding.get("official_retained_setup_payload")
        if not isinstance(source, dict):
            raise ReplayError(
                f"validation {label} retained source is not an object"
            )
        logical_path = source.get("logical_path")
        payloads = regular.EXPECTED_LARGE_MODEL_MANIFEST["payloads"]
        payload_binding = (
            payloads.get(logical_path) if isinstance(logical_path, str) else None
        )
        if not isinstance(payload_binding, dict):
            raise ReplayError(
                f"validation {label} retained source is not allowlisted"
            )
        retained_path = regular.PINNED_LARGE_MODEL_ROOT.joinpath(
            *PurePosixPath(payload_binding["retained_artifact"]).parts
        )
        expected_source = regular._retained_source_binding(
            root=regular.PINNED_LARGE_MODEL_ROOT,
            manifest_path=regular.PINNED_LARGE_MODEL_ROOT / "manifest.json",
            logical_path=logical_path,
            setup=regular.EXPECTED_LARGE_MODEL_MANIFEST[
                "official_benchmark"
            ]["setup"],
            payload_binding=payload_binding,
            retained_path=retained_path,
        )
        if (
            source != expected_source
            or binding.get("sha256") != payload_binding["payload_sha256"]
            or binding.get("size_bytes") != payload_binding["payload_size_bytes"]
        ):
            raise ReplayError(
                f"validation {label} retained source binding differs"
            )
    else:
        raise ReplayError(
            f"validation {label} input binding has unsupported source fields"
        )
    _require_sha256(binding.get("sha256"), f"validation {label} input SHA-256")
    size = binding.get("size_bytes")
    if type(size) is not int or size < 0:
        raise ReplayError(f"validation {label} input size is invalid")
    return binding


def _validate_sidecar_shape(sidecar: object) -> None:
    value = _exact_object(sidecar, TOP_KEYS, "validation sidecar")
    settings = _exact_object(value["settings"], SETTINGS_KEYS, "validation settings")
    checker = _exact_object(value["checker"], CHECKER_KEYS, "validation checker")
    if (
        value.get("schema") != SCHEMA
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "validated"
        or value.get("scoring_year") != SCORING_YEAR
        or settings
        != {
            "ignore_ce_y": IGNORE_CE_Y,
            "counterexample_atol": COUNTEREXAMPLE_ATOL,
            "counterexample_rtol": COUNTEREXAMPLE_RTOL,
            "scoring_zero_tolerance": SCORING_ZERO_TOLERANCE,
        }
        or checker
        != {
            "repository": OFFICIAL_RESULTS_REPOSITORY,
            "commit": OFFICIAL_RESULTS_COMMIT,
            "source_sha256": OFFICIAL_SOURCE_SHA256,
        }
    ):
        raise ReplayError("validation schema/settings/checker identity differs")
    harness = _exact_object(value["harness"], HARNESS_KEYS, "validation harness")
    _require_sha256(harness.get("runner_sha256"), "validation runner SHA-256")
    if (
        harness.get("worker_sha256") != PINNED_WORKER_SHA256
        or harness.get("protocol") != WORKER_PROTOCOL
        or harness.get("import_roots")
        != [
            str(PINNED_RUNTIME_ROOT / PINNED_SCORING_RELATIVE),
            str(PINNED_RUNTIME_ROOT / PINNED_SITE_PACKAGES_RELATIVE),
            str(PINNED_RUNTIME_ROOT / PINNED_STDLIB_RELATIVE),
        ]
    ):
        raise ReplayError("validation harness identity differs")
    worker_receipt = _exact_object(
        value["worker_receipt"],
        WORKER_RECEIPT_KEYS,
        "validation worker receipt",
    )
    for label in ("onnx", "vnnlib", "counterexample"):
        _exact_object(
            worker_receipt[label],
            WORKER_FILE_RECEIPT_KEYS,
            f"validation worker receipt {label}",
        )
    runtime = _exact_object(value["runtime"], RUNTIME_KEYS, "validation runtime")
    _exact_object(
        runtime["execstack_patch"],
        EXECSTACK_PATCH_KEYS,
        "validation execstack patch",
    )
    if (
        runtime.get("execution_scope") != "host_bound_local_replay"
        or runtime.get("native_dependencies") != PINNED_NATIVE_DEPENDENCIES
    ):
        raise ReplayError("validation runtime is not the exact host-bound closure")
    _exact_object(value["measurement"], MEASUREMENT_KEYS, "validation measurement")
    evidence = _exact_object(value["evidence"], EVIDENCE_KEYS, "validation evidence")
    for label in ("metadata", "raw_result", "start_manifest"):
        _exact_object(evidence[label], FILE_LINK_KEYS, f"validation {label}")
    _exact_object(
        evidence["extracted_assignment"],
        ASSIGNMENT_KEYS,
        "validation assignment",
    )
    for label in ("onnx", "vnnlib"):
        _validate_input_binding(evidence[label], label)
    response = _validate_machine_response(value["response"])
    earns_credit = response["result"] in CREDIT_RESULTS
    if (
        value.get("official_result") != response["result"]
        or value.get("rationale") != response["message"]
        or value.get("score_credit") is not earns_credit
        or value.get("classification") != ("valid" if earns_credit else "invalid")
    ):
        raise ReplayError("validation classification differs from worker response")
    expected_inputs = {
        "onnx": {
            "sha256": evidence["onnx"]["sha256"],
            "size_bytes": evidence["onnx"]["size_bytes"],
        },
        "vnnlib": {
            "sha256": evidence["vnnlib"]["sha256"],
            "size_bytes": evidence["vnnlib"]["size_bytes"],
        },
        "counterexample": {
            "sha256": evidence["extracted_assignment"]["sha256"],
            "size_bytes": evidence["extracted_assignment"]["size_bytes"],
        },
    }
    request_binding = {
        "protocol": WORKER_PROTOCOL,
        "abs_tolerance": COUNTEREXAMPLE_ATOL,
        "rel_tolerance": COUNTEREXAMPLE_RTOL,
        **expected_inputs,
    }
    expected_receipt = {
        "protocol": WORKER_PROTOCOL,
        "request_sha256": _canonical_sha256(request_binding),
        **expected_inputs,
        "response_sha256": _canonical_sha256(response),
        "native_dependencies_sha256": _canonical_sha256(PINNED_NATIVE_DEPENDENCIES),
    }
    if worker_receipt != expected_receipt:
        raise ReplayError("validation worker receipt differs from sidecar evidence")


def _canonical_sidecar(metadata_path: Path) -> Path:
    return metadata_path.with_name(
        f"{metadata_path.stem}.vnncomp2025-zero-tol-validation.json"
    )


def _write_immutable(path: Path, data: bytes, root: Path) -> None:
    if path.parent.is_symlink():
        raise ReplayError("validation sidecar parent must not be a symlink")
    try:
        parent = path.parent.resolve(strict=True)
        parent.relative_to(root)
    except (OSError, ValueError) as error:
        raise ReplayError("validation sidecar parent escapes artifact root") from error
    if path.is_symlink():
        raise FileExistsError(f"refusing to replace validation sidecar: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace validation sidecar: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _revalidate_archive(archive: ArchiveEvidence) -> None:
    for label, evidence in (
        ("metadata", archive.metadata_file),
        ("raw result", archive.result_file),
        ("start manifest", archive.start_file),
        ("original ONNX", archive.onnx.original),
        ("sealed ONNX", archive.onnx.sealed),
        ("original VNN-LIB", archive.vnnlib.original),
        ("sealed VNN-LIB", archive.vnnlib.sealed),
    ):
        _same_file(evidence, label)
    try:
        regular.revalidate_official_results(archive.official)
        regular.revalidate_official_benchmark(archive.benchmark)
        for label, inputs in (
            ("onnx", archive.onnx),
            ("vnnlib", archive.vnnlib),
        ):
            current, payload = regular.authoritative_benchmark_input(
                benchmark=archive.benchmark,
                category=archive.metadata["category"],
                declared_name=inputs.authoritative.declared_name,
                label=label,
            )
            if current != inputs.authoritative or payload != inputs.payload:
                raise ReplayError(
                    f"authoritative {label} changed during official replay"
                )
    except regular.EvidenceError as error:
        raise ReplayError(str(error)) from error


def replay_archive(
    *,
    metadata_path: Path,
    artifact_root: Path,
    benchmark_root: Path,
    official_results: Path,
    runtime_root: Path = PINNED_RUNTIME_ROOT,
    timeout_seconds: int = 600,
) -> Path:
    """Execute and durably publish one exact 2025 ZERO-TOL replay."""
    if timeout_seconds <= 0:
        raise ReplayError("checker timeout must be positive")
    if runtime_root != PINNED_RUNTIME_ROOT or runtime_root.is_symlink():
        raise ReplayError(
            "checker runtime argument must be the exact pinned retained path"
        )
    archive = _load_archive(
        metadata_path=metadata_path,
        artifact_root=artifact_root,
        benchmark_root=benchmark_root,
        official_results=official_results,
    )
    sidecar_path = _canonical_sidecar(archive.metadata_path)
    if sidecar_path.exists() or sidecar_path.is_symlink():
        raise FileExistsError(f"refusing to replace validation sidecar: {sidecar_path}")
    runtime_root = runtime_root.resolve(strict=True)
    benchmark_link = runtime_root / "vnncomp2025_benchmarks"
    if benchmark_link.resolve(strict=True) != archive.benchmark.repository_root:
        raise ReplayError(
            "retained runtime benchmark link differs from the pinned repository"
        )
    checker = _checker_identity(official_results, runtime_root)
    bound_replay = replay_bound_payloads(
        onnx_payload=archive.onnx.payload,
        vnnlib_payload=archive.vnnlib.payload,
        assignment_bytes=archive.assignment_bytes,
        runtime_root=runtime_root,
        timeout_seconds=timeout_seconds,
    )

    _revalidate_archive(archive)
    if _checker_identity(official_results, runtime_root) != checker:
        raise ReplayError("official checker identity changed during replay")
    sidecar = _build_sidecar(
        archive=archive,
        checker=checker,
        harness=bound_replay["harness"],
        worker_receipt=bound_replay["worker_receipt"],
        runtime=bound_replay["runtime"],
        response=bound_replay["response"],
    )
    data = (
        json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    _write_immutable(sidecar_path, data, archive.artifact_root)
    return sidecar_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--official-results", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=PINNED_RUNTIME_ROOT)
    parser.add_argument("--checker-timeout", type=int, default=600)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        sidecar = replay_archive(
            metadata_path=args.metadata,
            artifact_root=args.artifact_root,
            benchmark_root=args.benchmark_root,
            official_results=args.official_results,
            runtime_root=args.runtime_root,
            timeout_seconds=args.checker_timeout,
        )
    except (
        FileExistsError,
        OSError,
        ReplayError,
        ValueError,
    ) as error:
        print(f"exact 2025 counterexample replay failed: {error}", file=sys.stderr)
        return 2
    print(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
