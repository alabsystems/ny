#!/usr/bin/env python3
"""Replay an immutable NY SAT artifact with the pinned VNN-COMP checker.

This tool is intentionally fail-closed.  It validates the archived result,
model, property, metadata, and run-start manifest before starting the official
checker, validates them again before publishing evidence, and creates a
separate validation sidecar with O_EXCL.  It never rewrites the measurement
archive or translates a solver witness into another assignment syntax.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

PINNED_CHECKER_COMMIT = "b0ae71109ad0fe89661d5989405dc533bc3a9ee7"
PINNED_VNNLIB_PYTHON_COMMIT = "12c3f30dce67c7391ceb774be96b604b405c11f0"
CPU_PROVIDER = "CPUExecutionProvider"
SUPPORTED_METADATA_SCHEMAS = {
    "ny_measurement_result_v1",
    "ny_measurement_result_v2",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReplayError(RuntimeError):
    """The evidence or checker environment failed a closed-world check."""


@dataclass(frozen=True)
class FileEvidence:
    path: Path
    sha256: str
    size_bytes: int
    fingerprint: tuple[int, int, int, int, int]


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
    model_file: FileEvidence
    property_file: FileEvidence
    vnnlib_version: str


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _stable_file_evidence(path: Path) -> FileEvidence:
    path = path.resolve(strict=True)
    before = _fingerprint(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = _fingerprint(path)
    if before != after:
        raise ReplayError(f"file changed while it was hashed: {path}")
    return FileEvidence(path, digest.hexdigest(), after[2], after)


def _stable_read(path: Path) -> tuple[bytes, FileEvidence]:
    path = path.resolve(strict=True)
    before = _fingerprint(path)
    data = path.read_bytes()
    after = _fingerprint(path)
    if before != after:
        raise ReplayError(f"file changed while it was read: {path}")
    return data, FileEvidence(path, _sha256(data), len(data), after)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReplayError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _relative_artifact(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReplayError(f"missing {label} artifact path")
    if "\\" in value or "\0" in value:
        raise ReplayError(f"unsafe {label} artifact path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ReplayError(f"unsafe {label} artifact path: {value!r}")
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ReplayError(
            f"{label} artifact escapes the artifact root: {value!r}"
        ) from error
    return resolved


def _absolute_input(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReplayError(f"missing {label} resolved path")
    declared = Path(value)
    if not declared.is_absolute():
        raise ReplayError(f"{label} resolved path is not absolute: {value!r}")
    try:
        resolved = declared.resolve(strict=True)
    except OSError as error:
        raise ReplayError(f"{label} input is unavailable: {declared}") from error
    if resolved != declared:
        raise ReplayError(f"{label} resolved path is not canonical: {declared}")
    return resolved


def _verify_declared_file(
    path: Path,
    record: object,
    label: str,
) -> FileEvidence:
    if not isinstance(record, dict):
        raise ReplayError(f"missing {label} evidence record")
    evidence = _stable_file_evidence(path)
    expected_digest = _require_sha256(record.get("sha256"), f"{label} SHA-256")
    if evidence.sha256 != expected_digest:
        raise ReplayError(
            f"{label} SHA-256 mismatch: metadata={expected_digest}, actual={evidence.sha256}"
        )
    expected_size = record.get("size_bytes")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ReplayError(f"missing or invalid {label} size")
    if evidence.size_bytes != expected_size:
        raise ReplayError(
            f"{label} size mismatch: metadata={expected_size}, actual={evidence.size_bytes}"
        )
    return evidence


def _infer_vnnlib_version(property_path: Path) -> str:
    parts = property_path.parts
    for index, part in enumerate(parts):
        if part == "vnnlib" and index > 0 and parts[index - 1] in ("1.0", "2.0"):
            return parts[index - 1]
    # VNN-COMP 2025 and the VNN-LIB 1.0 side of the 2026 corpus have legacy
    # layouts without a version directory in some local setup checkouts.
    return "1.0"


def _extract_assignment(result: bytes) -> bytes:
    lines = result.splitlines(keepends=True)
    if not lines or lines[0].strip().lower() != b"sat":
        raise ReplayError("raw result does not start with a standalone SAT verdict")
    if len(lines) == 1:
        raise ReplayError("raw SAT result has no assignment")
    assignment = b"".join(lines[1:])
    if not assignment.strip():
        raise ReplayError("raw SAT result has an empty assignment")
    try:
        assignment.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReplayError("raw SAT assignment is not UTF-8") from error
    return assignment


def _load_archive(
    metadata_path: Path,
    artifact_root: Path,
    requested_version: str,
) -> ArchiveEvidence:
    artifact_root = artifact_root.resolve(strict=True)
    if not artifact_root.is_dir():
        raise ReplayError(f"artifact root is not a directory: {artifact_root}")
    metadata_path = metadata_path.resolve(strict=True)
    try:
        metadata_path.relative_to(artifact_root)
    except ValueError as error:
        raise ReplayError("metadata file is outside the artifact root") from error

    metadata_bytes, metadata_file = _stable_read(metadata_path)
    try:
        metadata = json.loads(metadata_bytes)
    except json.JSONDecodeError as error:
        raise ReplayError(f"metadata is not valid JSON: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ReplayError("metadata JSON must be an object")
    if metadata.get("schema") not in SUPPORTED_METADATA_SCHEMAS:
        raise ReplayError(
            f"unsupported result metadata schema: {metadata.get('schema')!r}"
        )
    if metadata.get("solver_verdict") != "sat":
        raise ReplayError("only metadata for a SAT result can be replayed")
    if metadata.get("witness_present") is not True:
        raise ReplayError("SAT metadata does not assert that a witness is present")
    validation = metadata.get("counterexample_validation")
    if not isinstance(validation, dict) or validation.get("status") != "not_checked":
        raise ReplayError("counterexample metadata is not in the not_checked state")

    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ReplayError("metadata has no run ID")

    result_path = _relative_artifact(
        artifact_root, metadata.get("result_artifact"), "raw result"
    )
    result_bytes, result_file = _stable_read(result_path)
    result_digest = metadata.get("raw_result_sha256", metadata.get("result_sha256"))
    expected_result = _require_sha256(result_digest, "raw result SHA-256")
    if result_file.sha256 != expected_result:
        raise ReplayError(
            f"raw result SHA-256 mismatch: metadata={expected_result}, actual={result_file.sha256}"
        )
    secondary_digest = metadata.get("result_sha256")
    if (
        secondary_digest is not None
        and _require_sha256(secondary_digest, "result SHA-256") != result_file.sha256
    ):
        raise ReplayError("result_sha256 does not identify the archived result")
    assignment = _extract_assignment(result_bytes)

    start_path = _relative_artifact(
        artifact_root, metadata.get("start_manifest"), "start manifest"
    )
    start_bytes, start_file = _stable_read(start_path)
    expected_start = _require_sha256(
        metadata.get("start_manifest_sha256"), "start-manifest SHA-256"
    )
    if start_file.sha256 != expected_start:
        raise ReplayError(
            f"start-manifest SHA-256 mismatch: metadata={expected_start}, actual={start_file.sha256}"
        )
    try:
        start = json.loads(start_bytes)
    except json.JSONDecodeError as error:
        raise ReplayError("start manifest is not valid JSON") from error
    if not isinstance(start, dict) or start.get("schema") != "ny_measurement_start_v1":
        raise ReplayError("unsupported start-manifest schema")
    if start.get("run_id") != run_id:
        raise ReplayError("start-manifest run ID does not match result metadata")

    model_record = metadata.get("onnx")
    property_record = metadata.get("vnnlib")
    if not isinstance(model_record, dict) or not isinstance(property_record, dict):
        raise ReplayError("metadata has no complete model/property evidence")
    model_path = _absolute_input(model_record.get("resolved_path"), "ONNX")
    property_path = _absolute_input(property_record.get("resolved_path"), "VNN-LIB")
    model_file = _verify_declared_file(model_path, model_record, "ONNX")
    property_file = _verify_declared_file(property_path, property_record, "VNN-LIB")

    network_field = model_record.get("declared_path")
    if not isinstance(network_field, str) or not network_field:
        raise ReplayError("metadata has no declared ONNX path")
    if network_field.lstrip().startswith("["):
        raise ReplayError(
            "multi-network metadata binds only one model and cannot be replayed safely"
        )
    property_field = property_record.get("declared_path")
    if not isinstance(property_field, str) or not property_field:
        raise ReplayError("metadata has no declared VNN-LIB path")

    inferred_version = _infer_vnnlib_version(property_path)
    if requested_version == "auto":
        version = inferred_version
    else:
        version = requested_version
        if inferred_version == "2.0" and version != inferred_version:
            raise ReplayError(
                f"requested VNN-LIB {version}, but the archived path is version 2.0"
            )

    return ArchiveEvidence(
        artifact_root=artifact_root,
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_file=metadata_file,
        result_file=result_file,
        result_bytes=result_bytes,
        assignment_bytes=assignment,
        start_file=start_file,
        model_file=model_file,
        property_file=property_file,
        vnnlib_version=version,
    )


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReplayError(
            f"could not inspect Git repository {repo}: {error}"
        ) from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ReplayError(f"Git inspection failed for {repo}: {detail}")
    return result.stdout.decode("utf-8", "strict")


def _checker_identity(checker_repo: Path) -> dict[str, Any]:
    checker_repo = checker_repo.resolve(strict=True)
    commit = _git(checker_repo, "rev-parse", "HEAD").strip()
    if commit != PINNED_CHECKER_COMMIT:
        raise ReplayError(
            f"checker revision mismatch: expected {PINNED_CHECKER_COMMIT}, found {commit}"
        )
    status = _git(checker_repo, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ReplayError("pinned checker has tracked worktree changes")
    names = _git(
        checker_repo,
        "ls-files",
        "SCORING/*.py",
        "SCORING/requirements.txt",
    ).splitlines()
    if not names or "SCORING/requirements.txt" not in names:
        raise ReplayError("checker source inventory is incomplete")
    source_hashes: dict[str, str] = {}
    for name in sorted(names):
        source_hashes[name] = _stable_file_evidence(checker_repo / name).sha256
    return {
        "repository": "https://github.com/VNN-COMP/vnncomp2026_results",
        "commit": commit,
        "source_sha256": source_hashes,
    }


def _vnnlib_source_identity(source_repo: Path) -> dict[str, Any]:
    source_repo = source_repo.resolve(strict=True)
    commit = _git(source_repo, "rev-parse", "HEAD").strip()
    if commit != PINNED_VNNLIB_PYTHON_COMMIT:
        raise ReplayError(
            "VNNLIB-Python source revision mismatch: "
            f"expected {PINNED_VNNLIB_PYTHON_COMMIT}, found {commit}"
        )
    status = _git(source_repo, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ReplayError("VNNLIB-Python has tracked worktree changes")
    submodules = _git(source_repo, "submodule", "status", "--recursive").splitlines()
    if not submodules or any(line[:1] != " " for line in submodules):
        raise ReplayError(
            "VNNLIB-Python submodules are missing or not at pinned revisions"
        )
    return {
        "repository": "https://github.com/VNNLIB/VNNLIB-Python",
        "commit": commit,
        "submodules": [line.strip() for line in submodules],
    }


def _require_exact_venv(python: Path, expected_prefix: Path) -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix}))",
            ],
            capture_output=True,
            check=False,
            timeout=30,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReplayError(f"could not start checker Python: {error}") from error
    if result.returncode != 0:
        raise ReplayError("checker Python failed its identity probe")
    try:
        identity = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReplayError("checker Python returned an invalid identity") from error
    if Path(identity.get("prefix", "")).resolve() != expected_prefix.resolve():
        raise ReplayError(
            f"checker Python is not in the dedicated venv {expected_prefix}"
        )
    return {"executable": str(python), "prefix": str(expected_prefix.resolve())}


def _invoke_official_checker(
    *,
    evidence: ArchiveEvidence,
    checker_repo: Path,
    checker_python: Path,
    checker_venv: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    request = {
        "checker_repo": str(checker_repo.resolve()),
        "checker_venv": str(checker_venv.resolve()),
        "onnx_path": str(evidence.model_file.path),
        "vnnlib_path": str(evidence.property_file.path),
        "vnnlib_version": evidence.vnnlib_version,
        "abs_tolerance": 1e-4,
        "rel_tolerance": 0.0,
    }
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    with tempfile.TemporaryDirectory(prefix="ny-vnncomp-ce-replay-") as temporary:
        witness_path = Path(temporary) / "assignment.counterexample"
        witness_path.write_bytes(evidence.assignment_bytes)
        request["witness_path"] = str(witness_path)
        try:
            result = subprocess.run(
                [
                    str(checker_python),
                    str(Path(__file__).resolve()),
                    "--official-worker",
                ],
                input=json.dumps(request, sort_keys=True).encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
                cwd=temporary,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ReplayError(f"official checker process failed: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ReplayError(
            f"official checker exited with status {result.returncode}: {detail}"
        )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReplayError("official checker did not return one JSON result") from error
    if not isinstance(response, dict) or response.get("ok") is not True:
        raise ReplayError(
            f"official checker reported an infrastructure error: {response!r}"
        )
    if response.get("provider") != CPU_PROVIDER:
        raise ReplayError("official checker did not select CPUExecutionProvider")
    return response


def _same_evidence(before: FileEvidence, after: FileEvidence, label: str) -> None:
    if before != after:
        raise ReplayError(f"{label} evidence changed during official replay")


def _classification(result: str) -> tuple[str, str, bool]:
    if result == "correct":
        return "validated", "strictly_correct", True
    if result == "correct_up_to_tolerance":
        return "validated", "correct_up_to_tolerance", True
    if result == "unsupported":
        return "not_validated", "unsupported", False
    return "validated", "invalid", False


def _write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
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


def replay_archive(
    *,
    metadata_path: Path,
    artifact_root: Path,
    checker_repo: Path,
    checker_python: Path,
    checker_venv: Path,
    vnnlib_source: Path,
    vnnlib_version: str = "auto",
    sidecar_path: Path | None = None,
    timeout_seconds: int = 600,
    invoke: Callable[..., dict[str, Any]] = _invoke_official_checker,
) -> Path:
    """Validate one SAT archive and create its immutable replay sidecar."""
    if timeout_seconds <= 0:
        raise ReplayError("checker timeout must be positive")
    evidence = _load_archive(metadata_path, artifact_root, vnnlib_version)
    if sidecar_path is None:
        sidecar_path = evidence.metadata_path.with_name(
            f"{evidence.metadata_path.stem}.counterexample-validation.json"
        )
    if sidecar_path.exists() or sidecar_path.is_symlink():
        raise FileExistsError(f"refusing to replace validation sidecar: {sidecar_path}")
    checker = _checker_identity(checker_repo)
    vnnlib = _vnnlib_source_identity(vnnlib_source)
    python_identity = _require_exact_venv(checker_python, checker_venv)

    response = invoke(
        evidence=evidence,
        checker_repo=checker_repo,
        checker_python=checker_python,
        checker_venv=checker_venv,
        timeout_seconds=timeout_seconds,
    )
    result = response.get("result")
    rationale = response.get("rationale")
    if not isinstance(result, str) or not isinstance(rationale, str):
        raise ReplayError("official checker response has no result/rationale")
    status, classification, score_credit = _classification(result)

    # Bind the bytes the checker was intended to inspect and reject concurrent
    # mutation before publishing a durable classification.
    _same_evidence(
        evidence.metadata_file,
        _stable_file_evidence(evidence.metadata_file.path),
        "metadata",
    )
    _same_evidence(
        evidence.result_file,
        _stable_file_evidence(evidence.result_file.path),
        "raw result",
    )
    _same_evidence(
        evidence.start_file,
        _stable_file_evidence(evidence.start_file.path),
        "start manifest",
    )
    _same_evidence(
        evidence.model_file,
        _stable_file_evidence(evidence.model_file.path),
        "ONNX",
    )
    _same_evidence(
        evidence.property_file,
        _stable_file_evidence(evidence.property_file.path),
        "VNN-LIB",
    )
    if _checker_identity(checker_repo) != checker:
        raise ReplayError("official checker source changed during replay")
    if _vnnlib_source_identity(vnnlib_source) != vnnlib:
        raise ReplayError("VNNLIB-Python source changed during replay")

    sidecar = {
        "schema": "ny_counterexample_validation_v1",
        "schema_version": 1,
        "validated_at_utc": _utc_now(),
        "status": status,
        "classification": classification,
        "official_result": result,
        "rationale": rationale,
        "score_credit": score_credit,
        "establishes_strict_sat": result == "correct",
        "tolerances": {
            "input_absolute": 1e-4,
            "relative": 0.0,
            "output_absolute": 0.0,
        },
        "provider": CPU_PROVIDER,
        "checker": checker,
        "checker_runtime": {
            **python_identity,
            "dependency_versions": response.get("dependency_versions"),
            "installed_vnnlib_files_sha256": response.get(
                "installed_vnnlib_files_sha256"
            ),
            "available_onnxruntime_providers": response.get("available_providers"),
        },
        "vnnlib_python_source": vnnlib,
        "vnnlib_version": evidence.vnnlib_version,
        "measurement": {
            "run_id": evidence.metadata.get("run_id"),
            "category": evidence.metadata.get("category"),
            "instance_index": evidence.metadata.get("instance_index"),
        },
        "evidence": {
            "metadata": {
                "artifact": evidence.metadata_path.relative_to(
                    evidence.artifact_root
                ).as_posix(),
                "sha256": evidence.metadata_file.sha256,
                "size_bytes": evidence.metadata_file.size_bytes,
            },
            "raw_result": {
                "artifact": evidence.result_file.path.relative_to(
                    evidence.artifact_root
                ).as_posix(),
                "sha256": evidence.result_file.sha256,
                "size_bytes": evidence.result_file.size_bytes,
            },
            "extracted_assignment": {
                "sha256": _sha256(evidence.assignment_bytes),
                "size_bytes": len(evidence.assignment_bytes),
                "transformation": "removed_standalone_sat_verdict_line_only",
            },
            "start_manifest": {
                "artifact": evidence.start_file.path.relative_to(
                    evidence.artifact_root
                ).as_posix(),
                "sha256": evidence.start_file.sha256,
                "size_bytes": evidence.start_file.size_bytes,
            },
            "onnx": {
                "sha256": evidence.model_file.sha256,
                "size_bytes": evidence.model_file.size_bytes,
            },
            "vnnlib": {
                "sha256": evidence.property_file.sha256,
                "size_bytes": evidence.property_file.size_bytes,
            },
        },
    }
    data = json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _write_immutable(sidecar_path, data)
    return sidecar_path


def _requirements_versions(requirements_path: Path) -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ReplayError(f"checker requirement is not exactly pinned: {line!r}")
        name, expected = line.split("==", 1)
        observed = importlib.metadata.version(name)
        if observed != expected:
            raise ReplayError(
                f"checker dependency {name} mismatch: expected {expected}, found {observed}"
            )
        versions[name] = observed
    # These are transitive runtime requirements of the pinned wheels/source
    # build.  The organizer's requirements file does not constrain their exact
    # versions, so record rather than invent a stronger pin.
    for name in ("ml-dtypes", "pybind11", "setuptools", "typing-extensions"):
        versions[name] = importlib.metadata.version(name)
    return versions


def _installed_distribution_hashes(name: str) -> dict[str, str]:
    distribution = importlib.metadata.distribution(name)
    hashes: dict[str, str] = {}
    for entry in distribution.files or ():
        path = Path(distribution.locate_file(entry))
        if path.is_file() and (
            "vnnlib" in entry.parts or entry.name.endswith(".dist-info")
        ):
            hashes[str(entry)] = _stable_file_evidence(path).sha256
    if not hashes:
        raise ReplayError(f"could not inventory installed distribution {name}")
    return hashes


class _OfficialResult:
    CORRECT = "correct"
    CORRECT_UP_TO_TOLERANCE = "correct_up_to_tolerance"
    NO_CE = "no_ce"
    EXEC_DOESNT_MATCH = "exec_doesnt_match"
    SPEC_NOT_VIOLATED = "spec_not_violated"
    WRONG_SHAPE = "wrong_shape"
    MALFORMED_CE = "malformed_ce"
    UNSUPPORTED = "unsupported"


def _official_worker() -> int:
    """Internal boundary executed only by the dedicated checker interpreter."""
    try:
        request = json.loads(sys.stdin.buffer.read())
        if not isinstance(request, dict):
            raise ReplayError("worker request is not an object")
        checker_repo = Path(request["checker_repo"]).resolve(strict=True)
        expected_venv = Path(request["checker_venv"]).resolve(strict=True)
        if Path(sys.prefix).resolve() != expected_venv:
            raise ReplayError("worker is not running in the dedicated checker venv")
        scoring = checker_repo / "SCORING"
        sys.path.insert(0, str(scoring))

        versions = _requirements_versions(scoring / "requirements.txt")
        import onnxruntime as ort  # noqa: PLC0415

        if CPU_PROVIDER not in ort.get_available_providers():
            raise ReplayError("CPUExecutionProvider is unavailable")
        witness = Path(request["witness_path"]).resolve(strict=True)
        onnx_path = Path(request["onnx_path"]).resolve(strict=True)
        vnnlib_path = Path(request["vnnlib_path"]).resolve(strict=True)
        version = request["vnnlib_version"]
        abs_tolerance = float(request["abs_tolerance"])
        rel_tolerance = float(request["rel_tolerance"])

        if version == "2.0":
            from counterexamples_v2 import (  # noqa: PLC0415
                validate_vnnlib2_counterexample,
            )

            parts = vnnlib_path.parts
            indices = [index for index, part in enumerate(parts) if part == "vnnlib"]
            if not indices:
                raise ReplayError("VNN-LIB 2.0 property path has no vnnlib directory")
            benchmark_dir = Path(*parts[: indices[-1]])
            result, rationale = validate_vnnlib2_counterexample(
                benchmark_dir,
                str(onnx_path),
                str(vnnlib_path),
                str(witness),
                abs_tolerance,
                rel_tolerance,
                _OfficialResult,
                True,
            )
        elif version == "1.0":
            from counterexamples import get_ce_diff  # noqa: PLC0415

            result, rationale = get_ce_diff(
                str(onnx_path),
                str(vnnlib_path),
                str(witness),
                abs_tolerance,
                rel_tolerance,
            )
        else:
            raise ReplayError(f"unsupported VNN-LIB version: {version!r}")

        response = {
            "ok": True,
            "result": result,
            "rationale": rationale,
            "provider": CPU_PROVIDER,
            "available_providers": ort.get_available_providers(),
            "dependency_versions": versions,
            "installed_vnnlib_files_sha256": _installed_distribution_hashes("vnnlib"),
        }
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        return 0
    except Exception as error:  # worker must return a closed, machine-readable failure
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 2


def _build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--checker-repo",
        type=Path,
        default=repo / "external_tools" / "vnncomp2026_results",
    )
    parser.add_argument(
        "--checker-python",
        type=Path,
        default=Path("/home/ayates/.venvs/vnncomp-ce-2026/bin/python"),
    )
    parser.add_argument(
        "--checker-venv",
        type=Path,
        default=Path("/home/ayates/.venvs/vnncomp-ce-2026"),
    )
    parser.add_argument(
        "--vnnlib-source",
        type=Path,
        default=repo / "external_tools" / "VNNLIB-Python",
    )
    parser.add_argument(
        "--vnnlib-version", choices=("auto", "1.0", "2.0"), default="auto"
    )
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--checker-timeout", type=int, default=600)
    return parser


def main() -> int:
    if sys.argv[1:] == ["--official-worker"]:
        return _official_worker()
    args = _build_parser().parse_args()
    try:
        sidecar = replay_archive(
            metadata_path=args.metadata,
            artifact_root=args.artifact_root,
            checker_repo=args.checker_repo,
            checker_python=args.checker_python,
            checker_venv=args.checker_venv,
            vnnlib_source=args.vnnlib_source,
            vnnlib_version=args.vnnlib_version,
            sidecar_path=args.sidecar,
            timeout_seconds=args.checker_timeout,
        )
    except (ReplayError, FileExistsError, OSError, ValueError) as error:
        print(f"counterexample replay failed: {error}", file=sys.stderr)
        return 2
    print(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
