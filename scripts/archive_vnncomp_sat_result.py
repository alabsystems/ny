#!/usr/bin/env python3
"""Archive a complete VNN-COMP result before the sweep reuses its file.

The measurement sweep historically retained only the first result line. This
helper stores the complete raw bytes plus content hashes for the exact ONNX and
VNN-LIB inputs under an immutable, content-addressed instance directory. When
NY emitted a flight sidecar, its validated structured record is embedded in the
immutable row metadata as well. SAT results additionally fail closed unless the
raw result contains an assignment.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pathlib
import sys

# Sibling import: make the script directory importable first, exactly as
# replay_vnncomp2025_counterexample.py does. Without it the module loads when
# run as a script but not when a test imports it by path.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import _portable_file_lock as _file_lock  # noqa: E402
import ny_measurement_provenance as provenance  # noqa: E402
from ny_measurement_provenance import (  # noqa: E402
    EXECUTION_EVIDENCE_PROFILE,
    JOINT_EXECUTION_EVIDENCE_PROFILE,
    ExecutionPredicateError,
    ProvenanceError,
    flight_build_stamp_for_binary_build_id,
    parse_execution_predicate_requirement,
    validate_binary_build_id,
    validate_execution_predicate_requirement,
)


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
MAX_FLIGHT_RECORD_BYTES = 16 * 1024 * 1024
FLIGHT_SCHEMA_VERSION = 3
SUPPORTED_FLIGHT_SCHEMA_VERSIONS = frozenset({2, FLIGHT_SCHEMA_VERSION})
FLIGHT_RECORD_BASE_KEYS = frozenset(
    {
        "ambient_env",
        "ambient_env_schema",
        "backend_kind",
        "backend_summary",
        "build_id",
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
FLIGHT_RECORD_PARTIAL_KEY = "record_status"
FLIGHT_RECORD_STATUS_KEYS = frozenset(
    {"status", "reason", "flushed_at_secs"}
)
FLIGHT_HOST_KEYS = frozenset(
    {"cpu_model", "hostname", "logical_cores", "ram_bytes"}
)
FLIGHT_EVENT_KEYS = frozenset({"method", "status"})
FLIGHT_EVENT_OPTIONAL_KEYS = frozenset({"at_secs", "reason"})
FLIGHT_AMBIENT_WORKER_KEYS = frozenset(
    {"OMP_NUM_THREADS", "RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS"}
)
LEVER_RECEIPT_SCHEMA = "ny-levers/receipt/v3"
SUPPORTED_LEVER_RECEIPT_SCHEMAS = frozenset(
    {"ny-levers/receipt/v2", LEVER_RECEIPT_SCHEMA}
)
LEVER_SOURCES = frozenset(
    {"default", "config", "legacy_env", "legacy_env_rejected"}
)
LEVER_BUCKETS = frozenset({"default_on", "auto", "cli", "debug"})
LEVER_MOATS = frozenset({"none", "low", "high"})
LEVER_PROVENANCE = frozenset(
    {"value_neutral", "measured", "unmeasured", "guard"}
)
LEVER_NAME = re.compile(r"^NY_[A-Z0-9_]+$")
V1_INPUT_ASSIGNMENT = re.compile(
    r"\(\s*X_\d+\s+[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*\)"
)
V2_ASSIGNMENT_HEADER = re.compile(r"^(\S+)\s+(\S+)\s+\[([0-9,\s]*)\]$")


def _balanced_parentheses(text: str) -> bool:
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _structured_sat_assignment(lines: list[bytes]) -> bool:
    """Recognize a structurally complete VNN-LIB 1.x or 2.0 assignment."""
    try:
        payload = [line.decode("utf-8").strip() for line in lines if line.strip()]
    except UnicodeDecodeError:
        return False
    if not payload:
        return False

    # VNN-LIB 1.x: an outer s-expression containing at least one numeric X_i
    # pair. Full semantic checking is deliberately left to the official replay.
    legacy = "\n".join(payload)
    if legacy.startswith("("):
        return (
            _balanced_parentheses(legacy)
            and V1_INPUT_ASSIGNMENT.search(legacy) is not None
        )

    # VNN-LIB 2.0 section 5.3: one tensor header followed by exactly the
    # row-major scalar count implied by its shape, repeated to EOF.
    position = 0
    declarations = 0
    while position < len(payload):
        match = V2_ASSIGNMENT_HEADER.fullmatch(payload[position])
        if match is None:
            return False
        dimensions = match.group(3).strip()
        try:
            shape = (
                []
                if not dimensions
                else [int(value.strip()) for value in dimensions.split(",")]
            )
        except ValueError:
            return False
        if any(dimension <= 0 for dimension in shape):
            return False
        value_count = 1
        for dimension in shape:
            value_count *= dimension
        position += 1
        if position + value_count > len(payload):
            return False
        for value in payload[position : position + value_count]:
            if len(value.split()) != 1:
                return False
            try:
                float(value)
            except ValueError:
                if value.lower() not in {"true", "false"}:
                    return False
        position += value_count
        declarations += 1
    return declarations > 0


def _write_immutable(path: Path, data: bytes) -> None:
    """Create ``path`` or accept an identical retry; never replace evidence."""
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_fingerprint(stat: os.stat_result) -> dict[str, int]:
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _stable_identity(path: Path) -> dict[str, object]:
    """Capture a canonical regular file and reject resolution/content races."""
    candidate = path
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"measurement input does not exist: {candidate}") from error
    if not resolved.is_file():
        raise ValueError(f"measurement input is not a regular file: {resolved}")
    before = _stat_fingerprint(resolved.stat())
    digest = _sha256_file(resolved)
    after = _stat_fingerprint(resolved.stat())
    try:
        resolved_after = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"measurement input changed while captured: {candidate}"
        ) from error
    if before != after or resolved_after != resolved:
        raise ValueError(f"measurement input changed while captured: {candidate}")
    return {
        "declared_path": str(candidate),
        "resolved_path": str(resolved),
        "size_bytes": after["size_bytes"],
        "sha256": digest,
        "fingerprint": after,
    }


def _instance_directory(
    artifact_root: Path,
    category: str,
    instance_index: int,
    onnx: str,
    vnnlib: str,
) -> Path:
    identity = json.dumps(
        [category, instance_index, onnx, vnnlib],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    instance_digest = hashlib.sha256(identity).hexdigest()
    return artifact_root / category / f"{instance_index:05d}-{instance_digest[:16]}"


def _seal_bound_file(
    *,
    original: dict[str, object],
    destination: Path,
    artifact_root: Path,
) -> dict[str, object]:
    expected_digest = original.get("sha256")
    expected_fingerprint = original.get("fingerprint")
    source_value = original.get("resolved_path")
    if (
        not isinstance(expected_digest, str)
        or not isinstance(expected_fingerprint, dict)
        or not isinstance(source_value, str)
    ):
        raise ValueError("original input identity is incomplete")
    source = Path(source_value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        copied_digest = hashlib.sha256()
        try:
            with (
                source.open("rb") as input_file,
                os.fdopen(descriptor, "wb") as output_file,
            ):
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    copied_digest.update(chunk)
                    output_file.write(chunk)
                output_file.flush()
                os.fsync(output_file.fileno())
            if (
                copied_digest.hexdigest() != expected_digest
                or _stat_fingerprint(source.stat()) != expected_fingerprint
            ):
                raise ValueError(f"measurement input changed while sealed: {source}")
            destination.chmod(0o444)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    if destination.is_symlink() or not destination.is_file():
        raise ValueError(f"sealed input path is unsafe: {destination}")
    sealed = _stable_identity(destination)
    if sealed["sha256"] != expected_digest:
        raise ValueError(f"sealed input does not match original: {destination}")
    return {
        "artifact": destination.relative_to(artifact_root).as_posix(),
        "resolved_path": sealed["resolved_path"],
        "size_bytes": sealed["size_bytes"],
        "sha256": sealed["sha256"],
        "fingerprint": sealed["fingerprint"],
        "mode": "read_only",
    }


def _read_start_manifest(
    start_manifest: Path, artifact_root: Path, run_id: str
) -> tuple[dict[str, object], bytes, str, str]:
    if start_manifest.is_symlink():
        raise ValueError(f"start manifest must not be a symlink: {start_manifest}")
    start_manifest = start_manifest.resolve()
    if not start_manifest.is_file():
        raise ValueError(f"start manifest is not a regular file: {start_manifest}")
    before = _stat_fingerprint(start_manifest.stat())
    data = start_manifest.read_bytes()
    if _stat_fingerprint(start_manifest.stat()) != before:
        raise ValueError(f"start manifest changed while read: {start_manifest}")
    start = json.loads(data)
    if start.get("schema") != "ny_measurement_start_v1":
        raise ValueError(f"unsupported start manifest schema: {start_manifest}")
    if start.get("run_id") != run_id:
        raise ValueError(f"start manifest run ID does not match {run_id!r}")
    try:
        artifact = start_manifest.relative_to(artifact_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"start manifest is outside the artifact root: {start_manifest}"
        ) from error
    measurement = start.get("measurement")
    declared_root = (
        measurement.get("artifact_root") if isinstance(measurement, dict) else None
    )
    if (
        declared_root is not None
        and Path(str(declared_root)).resolve() != artifact_root
    ):
        raise ValueError("artifact root does not match the start manifest")
    return start, data, hashlib.sha256(data).hexdigest(), artifact


def _validate_bound_identity(
    *,
    identity: object,
    declared_path: Path | None,
    artifact_root: Path,
    sealed: bool,
) -> Path:
    if not isinstance(identity, dict):
        raise ValueError("preflight input identity is invalid")
    resolved_value = identity.get("resolved_path")
    digest = identity.get("sha256")
    fingerprint = identity.get("fingerprint")
    size_bytes = identity.get("size_bytes")
    if (
        not isinstance(resolved_value, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or not isinstance(fingerprint, dict)
        or type(size_bytes) is not int
        or size_bytes < 0
    ):
        raise ValueError("preflight input identity is incomplete")
    resolved = Path(resolved_value)
    if declared_path is not None:
        observed = _stable_identity(declared_path)
        if observed["declared_path"] != str(declared_path):
            raise ValueError("preflight declared input path changed")
    else:
        observed = _stable_identity(resolved)
    if (
        observed["resolved_path"] != resolved_value
        or observed["sha256"] != digest
        or observed["fingerprint"] != fingerprint
        or observed["size_bytes"] != size_bytes
    ):
        raise ValueError(f"preflight-bound input drifted: {resolved}")
    if sealed:
        artifact = identity.get("artifact")
        if (
            not isinstance(artifact, str)
            or (artifact_root / artifact).resolve() != resolved
        ):
            raise ValueError("sealed preflight artifact path is invalid")
        try:
            resolved.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError("sealed input is outside the artifact root") from error
    return resolved


def validate_input_preflight(
    *,
    preflight_manifest: Path,
    artifact_root: Path,
    run_id: str,
    category: str,
    instance_index: int,
    onnx: str,
    vnnlib: str,
    onnx_file: Path,
    vnnlib_file: Path,
    start_manifest: Path,
) -> tuple[dict[str, object], str]:
    """Rehash originals and seals and validate their immutable pre-run binding."""
    artifact_root = artifact_root.resolve()
    start_manifest = start_manifest.resolve()
    _, _, start_digest, start_artifact = _read_start_manifest(
        start_manifest, artifact_root, run_id
    )
    expected_dir = _instance_directory(
        artifact_root, category, instance_index, onnx, vnnlib
    )
    expected_path = expected_dir / f"{run_id}.preflight.json"
    if preflight_manifest.is_symlink():
        raise ValueError(
            f"preflight manifest must not be a symlink: {preflight_manifest}"
        )
    preflight_manifest = preflight_manifest.resolve()
    if preflight_manifest != expected_path:
        raise ValueError("preflight manifest is outside its bound instance path")
    if not preflight_manifest.is_file():
        raise ValueError(f"preflight manifest is unavailable: {preflight_manifest}")
    before = _stat_fingerprint(preflight_manifest.stat())
    data = preflight_manifest.read_bytes()
    if _stat_fingerprint(preflight_manifest.stat()) != before:
        raise ValueError("preflight manifest changed while read")
    preflight = json.loads(data)
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema") != "ny_measurement_input_preflight_v1"
        or preflight.get("run_id") != run_id
        or preflight.get("category") != category
        or preflight.get("instance_index") != instance_index
        or preflight.get("start_manifest") != start_artifact
        or preflight.get("start_manifest_sha256") != start_digest
    ):
        raise ValueError("preflight manifest identity does not match this row")
    inputs = preflight.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("preflight input section is invalid")
    for label, declared_name, declared_file in (
        ("onnx", onnx, onnx_file),
        ("vnnlib", vnnlib, vnnlib_file),
    ):
        value = inputs.get(label)
        if not isinstance(value, dict) or value.get("declared_name") != declared_name:
            raise ValueError(f"preflight {label} declaration does not match")
        original = value.get("original")
        if not isinstance(original, dict) or original.get("declared_path") != str(
            declared_file
        ):
            raise ValueError(f"preflight {label} original path does not match")
        _validate_bound_identity(
            identity=original,
            declared_path=declared_file,
            artifact_root=artifact_root,
            sealed=False,
        )
        sealed_path = _validate_bound_identity(
            identity=value.get("sealed"),
            declared_path=None,
            artifact_root=artifact_root,
            sealed=True,
        )
        if sealed_path.name != Path(original["resolved_path"]).name:
            raise ValueError(f"sealed {label} filename does not preserve the original")
    return preflight, hashlib.sha256(data).hexdigest()


def seal_inputs(
    *,
    artifact_root: Path,
    run_id: str,
    category: str,
    instance_index: int,
    onnx: str,
    vnnlib: str,
    onnx_file: Path,
    vnnlib_file: Path,
    start_manifest: Path,
) -> Path:
    """Seal a benchmark pair and create its immutable pre-execution manifest."""
    for label, component in (("run ID", run_id), ("category", category)):
        if SAFE_COMPONENT.fullmatch(component) is None:
            raise ValueError(f"unsafe {label}: {component!r}")
    if instance_index <= 0:
        raise ValueError("instance index must be positive")
    artifact_root = artifact_root.resolve()
    start_manifest = start_manifest.resolve()
    start, _, start_digest, start_artifact = _read_start_manifest(
        start_manifest, artifact_root, run_id
    )
    measurement = start.get("measurement")
    categories = (
        measurement.get("categories") if isinstance(measurement, dict) else None
    )
    if isinstance(categories, list) and category not in categories:
        raise ValueError(f"category is not selected by the start manifest: {category}")
    instance_dir = _instance_directory(
        artifact_root, category, instance_index, onnx, vnnlib
    )
    preflight_path = instance_dir / f"{run_id}.preflight.json"
    if preflight_path.exists():
        validate_input_preflight(
            preflight_manifest=preflight_path,
            artifact_root=artifact_root,
            run_id=run_id,
            category=category,
            instance_index=instance_index,
            onnx=onnx,
            vnnlib=vnnlib,
            onnx_file=onnx_file,
            vnnlib_file=vnnlib_file,
            start_manifest=start_manifest,
        )
        return preflight_path

    originals = {
        "onnx": _stable_identity(onnx_file),
        "vnnlib": _stable_identity(vnnlib_file),
    }
    run_seal_root = (
        start_manifest.parent / "sealed" / "inputs" / category / instance_dir.name
    )
    inputs: dict[str, object] = {}
    for label, declared_name in (("onnx", onnx), ("vnnlib", vnnlib)):
        original = originals[label]
        digest = original["sha256"]
        source_name = Path(str(original["resolved_path"])).name
        destination = run_seal_root / label / str(digest) / source_name
        sealed = _seal_bound_file(
            original=original,
            destination=destination,
            artifact_root=artifact_root,
        )
        inputs[label] = {
            "declared_name": declared_name,
            "original": original,
            "sealed": sealed,
        }
    payload = {
        "schema": "ny_measurement_input_preflight_v1",
        "run_id": run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "instance_index": instance_index,
        "start_manifest": start_artifact,
        "start_manifest_sha256": start_digest,
        "inputs": inputs,
    }
    _write_immutable(
        preflight_path,
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    validate_input_preflight(
        preflight_manifest=preflight_path,
        artifact_root=artifact_root,
        run_id=run_id,
        category=category,
        instance_index=instance_index,
        onnx=onnx,
        vnnlib=vnnlib,
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        start_manifest=start_manifest,
    )
    return preflight_path


def _cache_key(path: Path, fingerprint: dict[str, int]) -> str:
    identity = json.dumps(
        [str(path), fingerprint], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def _load_hash_cache(
    path: Path,
    *,
    run_id: str,
    start_manifest_digest: str,
) -> tuple[dict[str, object], bool]:
    if not path.exists():
        return (
            {
                "schema": "ny_measurement_input_hash_cache_v1",
                "run_id": run_id,
                "start_manifest_sha256": start_manifest_digest,
                "entries": {},
            },
            True,
        )
    if path.is_symlink():
        raise ValueError(f"input hash cache must not be a symlink: {path}")
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"input hash cache is unreadable: {path}: {error}") from error
    if cache.get("schema") != "ny_measurement_input_hash_cache_v1":
        raise ValueError(f"unsupported input hash cache schema: {path}")
    if cache.get("run_id") != run_id:
        raise ValueError(f"input hash cache run ID mismatch: {path}")
    if cache.get("start_manifest_sha256") != start_manifest_digest:
        raise ValueError(f"input hash cache start-manifest mismatch: {path}")
    if not isinstance(cache.get("entries"), dict):
        raise ValueError(f"input hash cache entries are invalid: {path}")
    return cache, False


def _write_hash_cache_atomic(path: Path, cache: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    data = json.dumps(cache, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory_descriptor)
            except OSError as error:
                if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
                    raise
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _input_evidence(
    path: Path,
    declared_path: str,
    cache: dict[str, object],
) -> tuple[dict[str, object], bool]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"measurement input is not a file: {path}")
    before = path.stat()
    fingerprint = _stat_fingerprint(before)
    key = _cache_key(path, fingerprint)
    entries = cache["entries"]
    assert isinstance(entries, dict)
    cached = entries.get(key)
    if isinstance(cached, dict):
        if cached.get("path") != str(path) or cached.get("fingerprint") != fingerprint:
            raise ValueError(f"input hash cache key collision: {path}")
        digest = cached.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"input hash cache digest is invalid: {path}")
        cache_hit = True
        dirty = False
    else:
        digest = _sha256_file(path)
        cache_hit = False
        dirty = True
    after = path.stat()
    if _stat_fingerprint(after) != fingerprint:
        raise ValueError(f"measurement input changed while it was hashed: {path}")
    if dirty:
        entries[key] = {
            "path": str(path),
            "fingerprint": fingerprint,
            "sha256": digest,
        }
    return (
        {
            "declared_path": declared_path,
            "resolved_path": str(path),
            "size_bytes": after.st_size,
            "sha256": digest,
            "hash_cache_key": key,
            "hash_cache_hit": cache_hit,
        },
        dirty,
    )


def _capture_input_evidence(
    *,
    cache_path: Path,
    run_id: str,
    start_manifest_digest: str,
    onnx_file: Path,
    onnx: str,
    vnnlib_file: Path,
    vnnlib: str,
) -> tuple[dict[str, object], dict[str, object]]:
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        _file_lock.lock_exclusive(lock.fileno())
        cache, dirty = _load_hash_cache(
            cache_path,
            run_id=run_id,
            start_manifest_digest=start_manifest_digest,
        )
        onnx_evidence, onnx_dirty = _input_evidence(onnx_file, onnx, cache)
        vnnlib_evidence, vnnlib_dirty = _input_evidence(vnnlib_file, vnnlib, cache)
        if dirty or onnx_dirty or vnnlib_dirty:
            _write_hash_cache_atomic(cache_path, cache)
        return onnx_evidence, vnnlib_evidence


def _config_inputs_identity(start: dict[str, object]) -> dict[str, object] | None:
    measurement = start.get("measurement")
    if measurement is None:
        return None
    if not isinstance(measurement, dict):
        raise ValueError("start manifest measurement section is invalid")
    config_inputs = measurement.get("config_inputs")
    if config_inputs is None:
        return None
    if not isinstance(config_inputs, dict):
        raise ValueError("start manifest config-input evidence is invalid")
    required = (
        "schema",
        "declared_path",
        "resolved_path",
        "entry_count",
        "manifest_sha256",
    )
    if any(key not in config_inputs for key in required):
        raise ValueError("start manifest config-input evidence is incomplete")
    digest = config_inputs["manifest_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("start manifest config-input digest is invalid")
    return {key: config_inputs[key] for key in required}


def _valid_lever_value(value: object) -> bool:
    """Whether ``value`` is one of the scalar shapes emitted by ny-levers."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if type(value) is int:
        return 0 <= value <= (1 << 64) - 1
    return type(value) is float and math.isfinite(value)


def _valid_v3_lever_state(
    value: object, *, ambient_env: object
) -> bool:
    """Validate the versioned, count-consistent Phase-0c lever envelope."""
    if not isinstance(value, dict) or not isinstance(value.get("status"), str):
        return False
    if value["status"] == "not_materialized":
        return set(value) == {"status"}
    if value["status"] == "invalid_config":
        return (
            set(value) == {"status", "reason"}
            and isinstance(value["reason"], str)
            and bool(value["reason"])
        )
    if value["status"] != "resolved" or set(value) != {"status", "receipt"}:
        return False

    receipt = value["receipt"]
    base_fields = {
        "schema",
        "lever_count",
        "env_present",
        "env_accepted",
        "env_rejected",
        "levers",
    }
    if not isinstance(receipt, dict):
        return False
    schema = receipt.get("schema")
    if schema not in SUPPORTED_LEVER_RECEIPT_SCHEMAS:
        return False
    expected_fields = base_fields | ({"env_ignored"} if schema == LEVER_RECEIPT_SCHEMA else set())
    if not isinstance(receipt, dict) or set(receipt) != expected_fields:
        return False
    count_fields = [
        "lever_count",
        "env_present",
        "env_accepted",
        "env_rejected",
    ]
    if schema == LEVER_RECEIPT_SCHEMA:
        count_fields.append("env_ignored")
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
    if schema == LEVER_RECEIPT_SCHEMA:
        optional_entry_fields.add("ignored_raw")
    ignored = 0
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
        has_ignored_raw = "ignored_raw" in entry
        env_backed = source in source_counts or has_ignored_raw
        if env_backed != (name in ambient_env):
            return False
        if has_ignored_raw:
            if (
                schema != LEVER_RECEIPT_SCHEMA
                or source not in {"default", "config"}
                or "rejected_raw" in entry
                or not isinstance(entry.get("ignored_raw"), str)
                or type(entry.get("env_utf8")) is not bool
                or entry["ignored_raw"] != ambient_env[name]
            ):
                return False
            ignored += 1
        elif source == "legacy_env":
            if set(entry) & {"rejected_raw", "ignored_raw"} or type(entry.get("env_utf8")) is not bool:
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
    expected_ignored = receipt.get("env_ignored", 0)
    return (
        receipt["env_accepted"] == accepted
        and receipt["env_rejected"] == rejected
        and expected_ignored == ignored
        and receipt["env_present"] == accepted + rejected + ignored
    )


def _valid_current_flight_shape(record: dict[str, object]) -> bool:
    """Require the closed v3 record/event envelope used for current credit."""
    expected_record_keys = set(FLIGHT_RECORD_BASE_KEYS)
    expected_record_keys.update(set(record) & FLIGHT_RECORD_OPTIONAL_KEYS)
    expected_record_keys.add("levers")
    if FLIGHT_RECORD_PARTIAL_KEY in record:
        expected_record_keys.add(FLIGHT_RECORD_PARTIAL_KEY)
    if set(record) != expected_record_keys:
        return False
    if FLIGHT_RECORD_PARTIAL_KEY in record:
        record_status = record.get(FLIGHT_RECORD_PARTIAL_KEY)
        if (
            not isinstance(record_status, dict)
            or set(record_status) != FLIGHT_RECORD_STATUS_KEYS
        ):
            return False
    host = record.get("host")
    if not isinstance(host, dict) or set(host) != FLIGHT_HOST_KEYS:
        return False
    events = record.get("events")
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict):
            return False
        expected_event_keys = set(FLIGHT_EVENT_KEYS)
        expected_event_keys.update(set(event) & FLIGHT_EVENT_OPTIONAL_KEYS)
        if set(event) != expected_event_keys:
            return False
    return True


def _finite_nonnegative_flight_time(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _valid_partial_v3_flight_lifecycle(record: dict[str, object]) -> bool:
    """Validate Rust's exact prefix record emitted before a hard stop.

    The listed nonterminal events remain truthful execution evidence.  What a
    partial record cannot claim is that an absent lane was skipped or that the
    verifier reached its ordinary terminal path, so the envelope and final
    ``run_incomplete`` event must agree exactly.
    """
    record_status = record.get(FLIGHT_RECORD_PARTIAL_KEY)
    if (
        not isinstance(record_status, dict)
        or set(record_status) != FLIGHT_RECORD_STATUS_KEYS
        or record_status.get("status") != "partial_hard_stop"
        or not isinstance(record_status.get("reason"), str)
        or not record_status["reason"]
        or not _finite_nonnegative_flight_time(
            record_status.get("flushed_at_secs")
        )
    ):
        return False
    events = record.get("events")
    if not isinstance(events, list) or not events:
        return False
    previous_at = -1.0
    for event in events[:-1]:
        if (
            not isinstance(event, dict)
            or event.get("method") in {"run_complete", "run_incomplete"}
            or event.get("status") not in {"ran", "skipped", "not_reached"}
            or "at_secs" not in event
            or not _finite_nonnegative_flight_time(event.get("at_secs"))
            or float(event["at_secs"]) < previous_at
        ):
            return False
        previous_at = float(event["at_secs"])
    terminal = events[-1]
    if not isinstance(terminal, dict):
        return False
    terminal_at = terminal.get("at_secs")
    return (
        terminal.get("method") == "run_incomplete"
        and terminal.get("status") == "incomplete"
        and terminal.get("reason") == record_status["reason"]
        and "at_secs" in terminal
        and _finite_nonnegative_flight_time(terminal_at)
        and float(terminal_at) >= previous_at
        and float(terminal_at) == float(record_status["flushed_at_secs"])
    )


def _capture_flight_record(
    path: Path | None,
    *,
    category: str,
    timeout_seconds: int,
    solver_verdict: str,
) -> dict[str, object]:
    """Capture a row-bound flight record, or state explicitly that none exists."""
    if path is None:
        return {"status": "not_requested"}
    if path.is_symlink():
        raise ValueError(f"flight record must not be a symlink: {path}")
    if not path.exists():
        return {"status": "missing"}
    if not path.is_file():
        raise ValueError(f"flight record is not a regular file: {path}")
    before = _stat_fingerprint(path.stat())
    if before["size_bytes"] > MAX_FLIGHT_RECORD_BYTES:
        raise ValueError(f"flight record is oversized: {path}")
    data = path.read_bytes()
    if _stat_fingerprint(path.stat()) != before:
        raise ValueError(f"flight record changed while captured: {path}")
    try:
        record = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"flight record is not valid JSON: {path}: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"flight record is not a JSON object: {path}")
    schema_version = record.get("schema_version")
    partial_v3 = (
        schema_version == FLIGHT_SCHEMA_VERSION
        and FLIGHT_RECORD_PARTIAL_KEY in record
    )
    schema_valid = (
        type(schema_version) is int
        and schema_version in SUPPORTED_FLIGHT_SCHEMA_VERSIONS
        and (
            schema_version == FLIGHT_SCHEMA_VERSION
            or FLIGHT_RECORD_PARTIAL_KEY not in record
        )
        and (
            schema_version != FLIGHT_SCHEMA_VERSION
            or (
                _valid_current_flight_shape(record)
                and (
                    not partial_v3
                    or _valid_partial_v3_flight_lifecycle(record)
                )
                and _valid_v3_lever_state(
                    record.get("levers"), ambient_env=record.get("ambient_env")
                )
            )
        )
    )
    ambient_env = record.get("ambient_env")
    events = record.get("events")
    if (
        not schema_valid
        or (
            schema_version == FLIGHT_SCHEMA_VERSION
            and (
                record.get("ambient_env_schema") != "ny-cli/ambient-env/v3"
                or not isinstance(record.get("build_id"), str)
                or not record.get("build_id")
            )
        )
        or not isinstance(record.get("backend_kind"), str)
        or not isinstance(record.get("backend_summary"), str)
        or not isinstance(record.get("host"), dict)
        or record.get("category") != category
        or type(record.get("budget_secs")) is not int
        or record.get("budget_secs") != timeout_seconds
        or not isinstance(ambient_env, dict)
        or not all(
            isinstance(name, str)
            and isinstance(value, str)
            and (name.startswith("NY_") or name in FLIGHT_AMBIENT_WORKER_KEYS)
            for name, value in ambient_env.items()
        )
        or not isinstance(events, list)
        or not all(
            isinstance(event, dict)
            and isinstance(event.get("method"), str)
            and event.get("status")
            in (
                {"ran", "skipped", "not_reached", "incomplete"}
                if partial_v3
                else {"ran", "skipped", "not_reached", "complete"}
            )
            and (event.get("reason") is None or isinstance(event.get("reason"), str))
            and (
                event.get("at_secs") is None
                or (
                    type(event.get("at_secs")) in {int, float} and event["at_secs"] >= 0
                )
            )
            for event in events
        )
    ):
        raise ValueError(f"flight record identity is invalid: {path}")
    if partial_v3:
        # `_valid_partial_v3_flight_lifecycle` above owns the terminal prefix
        # contract.  In particular, a partial timeout has no `run_complete`
        # verdict to compare with the result token.
        if solver_verdict != "timeout":
            raise ValueError(
                "partial hard-stop flight record requires the committed "
                f"timeout verdict, not {solver_verdict!r}: {path}"
            )
        return {
            "status": "captured",
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "record": record,
        }
    terminal = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("method") == "run_complete"
        and event.get("status") == "complete"
    ]
    if (
        len(terminal) != 1
        or terminal[0].get("reason") != solver_verdict
        or not events
        or events[-1] is not terminal[0]
    ):
        raise ValueError(
            f"flight record terminal verdict does not match {solver_verdict!r}: {path}"
        )
    return {
        "status": "captured",
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "record": record,
    }


def _parse_required_flight_event(raw: str) -> dict[str, object]:
    """Parse one current versioned execution predicate via the shared API."""
    try:
        return parse_execution_predicate_requirement(raw)
    except ExecutionPredicateError as error:
        raise ValueError(
            f"required flight event must be a current predicate: {error}"
        ) from error


def _enforce_required_flight_events(
    flight_record: dict[str, object], requirements: tuple[str, ...],
    *,
    expected_build_stamp: str | None = None,
    sealed_requirement: str | None = None,
    execution_assignment: object = None,
) -> None:
    """Reject a measurement whose requested treatment did not actually engage."""
    if not requirements:
        return
    for raw in requirements:
        _parse_required_flight_event(raw)
    if flight_record.get("status") != "captured":
        raise ValueError(
            "required flight event cannot be verified because the flight record "
            f"is {flight_record.get('status', 'invalid')!r}"
        )
    record = flight_record.get("record")
    if not isinstance(record, dict):
        raise ValueError("required flight event cannot be verified")
    if expected_build_stamp is not None and record.get("build_id") != expected_build_stamp:
        raise ValueError(
            "required flight event was emitted by a different solver build stamp"
        )
    for raw in requirements:
        try:
            validate_execution_predicate_requirement(
                record,
                raw,
                assignment=(
                    execution_assignment
                    if raw == sealed_requirement
                    else None
                ),
            )
        except ExecutionPredicateError as error:
            raise ValueError(
                f"required flight event did not engage for {raw!r}: {error}"
            ) from error


def _required_flight_events_from_start(start: dict[str, object]) -> tuple[str, ...]:
    """Recover the predicate required by the current evidence profile.

    Pre-profile manifests remain readable as historical artifacts, but they do
    not acquire current execution credit merely because their environment
    happens to contain the retired bare METHOD=STATUS transport.
    """
    measurement = start.get("measurement")
    profile = (
        measurement.get("execution_evidence_profile")
        if isinstance(measurement, dict)
        else None
    )
    if profile is None:
        return ()
    if profile not in {
        provenance.LEGACY_EXECUTION_EVIDENCE_PROFILE,
        provenance.LEGACY_ASSIGNMENT_EXECUTION_EVIDENCE_PROFILE,
        provenance.LEGACY_JOINT_ASSIGNMENT_EXECUTION_EVIDENCE_PROFILE,
        EXECUTION_EVIDENCE_PROFILE,
        JOINT_EXECUTION_EVIDENCE_PROFILE,
    }:
        raise ValueError("start manifest execution-evidence profile is unsupported")
    environment = start.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("start manifest environment section is invalid")
    values = environment.get("values")
    if not isinstance(values, dict):
        raise ValueError("start manifest environment values are invalid")
    raw = values.get("NY_MEASURE_REQUIRE_FLIGHT_EVENT")
    if not isinstance(raw, str):
        raise ValueError(
            "current execution-evidence profile requires a sealed execution "
            "predicate"
        )
    # Parse now so a malformed sealed manifest fails even if no flight record
    # was emitted. The provenance writer normally guarantees this shape.
    parsed = _parse_required_flight_event(raw)
    expected_requirement_schema = (
        provenance.JOINT_EXECUTION_PREDICATE_SCHEMA
        if profile in {
            provenance.LEGACY_JOINT_ASSIGNMENT_EXECUTION_EVIDENCE_PROFILE,
            JOINT_EXECUTION_EVIDENCE_PROFILE,
        }
        else provenance.EXECUTION_PREDICATE_SCHEMA
    )
    if parsed.get("schema") != expected_requirement_schema:
        raise ValueError(
            "start manifest execution predicate does not match its evidence profile"
        )
    typed_values = environment.get("typed_values")
    if not isinstance(typed_values, dict) or typed_values.get(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT"
    ) != parsed:
        raise ValueError(
            "start manifest required execution predicate is not bound to its "
            "current typed declaration"
        )
    return (raw,)


def _execution_predicate_assignment_from_start(
    start: dict[str, object],
) -> object:
    """Validate the exact causal assignment sealed by the start producer."""
    measurement = start.get("measurement")
    profile = (
        measurement.get("execution_evidence_profile")
        if isinstance(measurement, dict)
        else None
    )
    if profile in {None, provenance.LEGACY_EXECUTION_EVIDENCE_PROFILE}:
        return None
    assignment_schema_by_profile = {
        provenance.LEGACY_ASSIGNMENT_EXECUTION_EVIDENCE_PROFILE: (
            provenance.LEGACY_EXECUTION_PREDICATE_ASSIGNMENT_SCHEMA
        ),
        provenance.LEGACY_JOINT_ASSIGNMENT_EXECUTION_EVIDENCE_PROFILE: (
            provenance.LEGACY_JOINT_EXECUTION_PREDICATE_ASSIGNMENT_SCHEMA
        ),
        EXECUTION_EVIDENCE_PROFILE: (
            provenance.EXECUTION_PREDICATE_ASSIGNMENT_SCHEMA
        ),
        JOINT_EXECUTION_EVIDENCE_PROFILE: (
            provenance.JOINT_EXECUTION_PREDICATE_ASSIGNMENT_SCHEMA
        ),
    }
    expected_assignment_schema = assignment_schema_by_profile.get(profile)
    if expected_assignment_schema is None or not isinstance(measurement, dict):
        raise ValueError("start manifest execution-evidence profile is unsupported")
    try:
        search_space_contract = provenance.validate_start_search_space_contract(
            start, reprobe_sealed_binary=False
        )
        preset_docs = provenance.execution_predicate_preset_docs_from_start(start)
        validated = provenance.validate_execution_predicate_assignment(
            measurement.get("execution_predicate_assignment"),
            environment=start.get("environment"),
            search_space_contract=search_space_contract,
            preset_docs=preset_docs,
            preset_expected=preset_docs is not None,
        )
        assignment = measurement.get("execution_predicate_assignment")
        if (
            not isinstance(assignment, dict)
            or assignment.get("schema") != expected_assignment_schema
        ):
            raise ExecutionPredicateError(
                "execution predicate assignment does not match its evidence profile"
            )
        return validated
    except (ExecutionPredicateError, ProvenanceError) as error:
        raise ValueError(
            f"start manifest execution-predicate assignment is invalid: {error}"
        ) from error


def _current_build_stamp_from_start(start: dict[str, object]) -> str | None:
    """Bind current row execution to the full binary identity sealed at start."""
    measurement = start.get("measurement")
    profile = (
        measurement.get("execution_evidence_profile")
        if isinstance(measurement, dict)
        else None
    )
    if profile is None:
        return None
    if profile not in {
        provenance.LEGACY_EXECUTION_EVIDENCE_PROFILE,
        provenance.LEGACY_ASSIGNMENT_EXECUTION_EVIDENCE_PROFILE,
        provenance.LEGACY_JOINT_ASSIGNMENT_EXECUTION_EVIDENCE_PROFILE,
        EXECUTION_EVIDENCE_PROFILE,
        JOINT_EXECUTION_EVIDENCE_PROFILE,
    }:
        raise ValueError("start manifest execution-evidence profile is unsupported")
    solver = start.get("solver_binary")
    if not isinstance(solver, dict) or not isinstance(solver.get("sha256"), str):
        raise ValueError("current start manifest solver identity is incomplete")
    try:
        validate_binary_build_id(
            solver.get("build_id"),
            expected_executable_sha256=solver["sha256"],
            declared_features=solver.get("declared_build_features"),
        )
        return flight_build_stamp_for_binary_build_id(solver["build_id"])
    except ProvenanceError as error:
        raise ValueError(f"current start manifest build identity is invalid: {error}") from error


def archive_result(
    *,
    result_file: Path,
    solver_log_file: Path,
    artifact_root: Path,
    run_id: str,
    category: str,
    instance_index: int,
    onnx: str,
    vnnlib: str,
    onnx_file: Path,
    vnnlib_file: Path,
    solver_verdict: str,
    solver_exit_status: int,
    timeout_seconds: int,
    elapsed_seconds: int,
    source_csv: str,
    start_manifest: Path,
    preflight_manifest: Path,
    flight_file: Path | None = None,
    required_flight_events: tuple[str, ...] = (),
) -> Path:
    """Archive one complete raw result and return its result-artifact path."""
    for label, component in (("run ID", run_id), ("category", category)):
        if SAFE_COMPONENT.fullmatch(component) is None:
            raise ValueError(f"unsafe {label}: {component!r}")
    if SAFE_COMPONENT.fullmatch(solver_verdict) is None:
        raise ValueError(f"unsafe solver verdict: {solver_verdict!r}")
    if not 0 <= solver_exit_status <= 255:
        raise ValueError(
            f"solver exit status is outside the shell range: {solver_exit_status}"
        )
    solver_verdict = solver_verdict.lower()
    data = result_file.read_bytes()
    solver_log_data = solver_log_file.read_bytes()
    if (
        flight_file is not None
        and flight_file.absolute() != Path(f"{result_file}.flight.json").absolute()
    ):
        raise ValueError("flight record is not adjacent to its result scratch file")
    flight_record = _capture_flight_record(
        flight_file,
        category=category,
        timeout_seconds=timeout_seconds,
        solver_verdict=solver_verdict,
    )
    lines = data.splitlines()
    first_line = b"".join(lines[0].split()).lower() if lines else b""
    if solver_verdict == "sat" and first_line != b"sat":
        raise ValueError(f"SAT result file has first line {first_line!r}")
    if solver_verdict == "sat" and not _structured_sat_assignment(lines[1:]):
        raise ValueError(
            "SAT result does not contain a structured counterexample assignment"
        )
    if first_line and first_line.decode("utf-8", "replace") != solver_verdict:
        raise ValueError(
            f"raw result verdict {first_line!r} does not match {solver_verdict!r}"
        )

    artifact_root = artifact_root.resolve()
    start_manifest = start_manifest.resolve()
    try:
        start_manifest_artifact = start_manifest.relative_to(artifact_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"start manifest is outside the artifact root: {start_manifest}"
        ) from error
    start, start_manifest_data, start_manifest_digest, _ = _read_start_manifest(
        start_manifest, artifact_root, run_id
    )
    sealed_requirements = _required_flight_events_from_start(start)
    execution_assignment = _execution_predicate_assignment_from_start(start)
    expected_build_stamp = _current_build_stamp_from_start(start)
    all_requirements = tuple(
        dict.fromkeys((*sealed_requirements, *required_flight_events))
    )
    _enforce_required_flight_events(
        flight_record,
        all_requirements,
        expected_build_stamp=expected_build_stamp,
        sealed_requirement=(sealed_requirements[0] if sealed_requirements else None),
        execution_assignment=execution_assignment,
    )
    config_inputs = _config_inputs_identity(start)
    preflight, preflight_digest = validate_input_preflight(
        preflight_manifest=preflight_manifest,
        artifact_root=artifact_root,
        run_id=run_id,
        category=category,
        instance_index=instance_index,
        onnx=onnx,
        vnnlib=vnnlib,
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        start_manifest=start_manifest,
    )
    preflight_inputs = preflight["inputs"]
    assert isinstance(preflight_inputs, dict)
    cache_path = start_manifest.with_name("input_hash_cache.json")
    onnx_evidence, vnnlib_evidence = _capture_input_evidence(
        cache_path=cache_path,
        run_id=run_id,
        start_manifest_digest=start_manifest_digest,
        onnx_file=onnx_file,
        onnx=onnx,
        vnnlib_file=vnnlib_file,
        vnnlib=vnnlib,
    )
    for label, observed in (("onnx", onnx_evidence), ("vnnlib", vnnlib_evidence)):
        value = preflight_inputs[label]
        assert isinstance(value, dict)
        original = value["original"]
        assert isinstance(original, dict)
        if (
            observed["resolved_path"] != original["resolved_path"]
            or observed["sha256"] != original["sha256"]
            or observed["size_bytes"] != original["size_bytes"]
        ):
            raise ValueError(f"post-run {label} evidence differs from preflight")

    result_digest = hashlib.sha256(data).hexdigest()
    solver_log_digest = hashlib.sha256(solver_log_data).hexdigest()
    instance_dir = _instance_directory(
        artifact_root, category, instance_index, onnx, vnnlib
    )
    archived_result = instance_dir / f"{run_id}.results"
    archived_solver_log = instance_dir / f"{run_id}.solver.log"
    metadata_path = instance_dir / f"{run_id}.json"

    # An identical retry in the same run is idempotent. Keep the original
    # capture timestamp rather than regenerating metadata that would differ
    # only because the retry happened later.
    artifact_exists = (
        archived_result.exists(),
        archived_solver_log.exists(),
        metadata_path.exists(),
    )
    if any(artifact_exists) and not all(artifact_exists):
        raise FileExistsError(
            f"refusing incomplete pre-existing artifact set: {instance_dir}/{run_id}"
        )
    if all(artifact_exists):
        if archived_result.read_bytes() != data:
            raise FileExistsError(
                f"refusing to replace different artifact: {archived_result}"
            )
        if archived_solver_log.read_bytes() != solver_log_data:
            raise FileExistsError(
                f"refusing to replace different artifact: {archived_solver_log}"
            )
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata.get("result_sha256") != result_digest:
            raise ValueError(f"artifact metadata digest mismatch: {metadata_path}")
        existing_log = existing_metadata.get("solver_log")
        if (
            not isinstance(existing_log, dict)
            or existing_log.get("sha256") != solver_log_digest
        ):
            raise ValueError(f"artifact solver-log digest mismatch: {metadata_path}")
        if existing_metadata.get("start_manifest_sha256") != start_manifest_digest:
            raise ValueError(
                f"artifact start-manifest digest mismatch: {metadata_path}"
            )
        existing_preflight = existing_metadata.get("input_preflight")
        if (
            not isinstance(existing_preflight, dict)
            or existing_preflight.get("sha256") != preflight_digest
        ):
            raise ValueError(f"artifact preflight digest mismatch: {metadata_path}")
        if existing_metadata.get("solver_verdict") != solver_verdict:
            raise ValueError(f"artifact verdict mismatch: {metadata_path}")
        if existing_metadata.get("solver_exit_status") != solver_exit_status:
            raise ValueError(f"artifact solver exit-status mismatch: {metadata_path}")
        if (
            existing_metadata.get("flight_record", {"status": "not_requested"})
            != flight_record
        ):
            raise ValueError(f"artifact flight-record mismatch: {metadata_path}")
        existing_onnx = existing_metadata.get("onnx")
        existing_vnnlib = existing_metadata.get("vnnlib")
        if (
            not isinstance(existing_onnx, dict)
            or existing_onnx.get("sha256") != onnx_evidence["sha256"]
        ):
            raise ValueError(f"artifact ONNX digest mismatch: {metadata_path}")
        if (
            not isinstance(existing_vnnlib, dict)
            or existing_vnnlib.get("sha256") != vnnlib_evidence["sha256"]
        ):
            raise ValueError(f"artifact VNN-LIB digest mismatch: {metadata_path}")
        return archived_result

    metadata = {
        "schema": "ny_measurement_result_v2",
        "schema_version": 2,
        "run_id": run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "instance_index": instance_index,
        "onnx": onnx_evidence,
        "vnnlib": vnnlib_evidence,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": elapsed_seconds,
        "source_csv": source_csv,
        "solver_verdict": solver_verdict,
        "solver_exit_status": solver_exit_status,
        "witness_present": solver_verdict == "sat",
        "counterexample_validation": {
            "status": "not_checked" if solver_verdict == "sat" else "not_applicable",
            "checker": None,
        },
        "raw_result_sha256": result_digest,
        "result_sha256": result_digest,
        "result_artifact": archived_result.relative_to(artifact_root).as_posix(),
        "solver_log": {
            "artifact": archived_solver_log.relative_to(artifact_root).as_posix(),
            "sha256": solver_log_digest,
            "size_bytes": len(solver_log_data),
            "stream": "combined_stdout_stderr",
        },
        "flight_record": flight_record,
        "input_hash_cache": cache_path.relative_to(artifact_root).as_posix(),
        "input_preflight": {
            "artifact": preflight_manifest.resolve()
            .relative_to(artifact_root)
            .as_posix(),
            "sha256": preflight_digest,
            "schema": "ny_measurement_input_preflight_v1",
        },
        "execution_inputs": {
            label: value["sealed"]
            for label, value in preflight_inputs.items()
            if isinstance(value, dict)
        },
        "config_inputs": config_inputs,
        "execution_config_inputs": (
            start.get("measurement", {}).get("sealed_config_inputs")
            if isinstance(start.get("measurement"), dict)
            else None
        ),
        "start_manifest": start_manifest_artifact,
        "start_manifest_sha256": start_manifest_digest,
    }
    metadata_bytes = (
        json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )

    _write_immutable(archived_result, data)
    try:
        _write_immutable(archived_solver_log, solver_log_data)
        _write_immutable(metadata_path, metadata_bytes)
    except BaseException:
        # A result without its path/validation metadata is not acceptable new
        # evidence. Only remove the result when this call created it and no
        # immutable metadata could be paired with it.
        if not metadata_path.exists():
            archived_result.unlink(missing_ok=True)
            archived_solver_log.unlink(missing_ok=True)
        raise
    return archived_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--solver-log-file", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--instance-index", type=int, required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--vnnlib", required=True)
    parser.add_argument("--onnx-file", type=Path, required=True)
    parser.add_argument("--vnnlib-file", type=Path, required=True)
    parser.add_argument("--solver-verdict", required=True)
    parser.add_argument("--solver-exit-status", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--start-manifest", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--flight-file", type=Path)
    parser.add_argument(
        "--require-flight-event",
        action="append",
        default=[],
        metavar="VERSIONED_PREDICATE",
        help=(
            "fail before archiving unless the current lever receipt and strict "
            "runtime schema prove this versioned execution predicate; repeatable"
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    archived = archive_result(
        result_file=args.result_file,
        solver_log_file=args.solver_log_file,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        category=args.category,
        instance_index=args.instance_index,
        onnx=args.onnx,
        vnnlib=args.vnnlib,
        onnx_file=args.onnx_file,
        vnnlib_file=args.vnnlib_file,
        solver_verdict=args.solver_verdict,
        solver_exit_status=args.solver_exit_status,
        timeout_seconds=args.timeout_seconds,
        elapsed_seconds=args.elapsed_seconds,
        source_csv=args.source_csv,
        start_manifest=args.start_manifest,
        preflight_manifest=args.preflight_manifest,
        flight_file=args.flight_file,
        required_flight_events=tuple(args.require_flight_event),
    )
    print(archived)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
