#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Resolve and seal the alpha-beta-CROWN transfer baseline corpus.

This is intentionally a thin layer over ``ny_measurement_provenance.py``.
It owns the fixed-corpus and metric schemas; the existing helper remains the
authority for solver, source, dependency, benchmark, environment, and host
provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

CORPUS_SCHEMA = "ny_abcrown_transfer_corpus_v1"
BASELINE_SCHEMA = "ny_abcrown_transfer_baseline_v1"
ROW_SCHEMA = "ny_abcrown_transfer_row_v1"
SUPPLEMENTAL_SCHEMA = "ny_abcrown_transfer_supplemental_metrics_v1"
DEFAULT_MANIFEST = Path("benchmarks/abcrown_transfer_corpus_v1.json")

ENTRY_KINDS = frozenset({"vnncomp", "repository_pair", "repository_test"})
ENTRY_ROLES = frozenset({"diagnostic", "fixture", "sentinel", "suite"})
VERDICTS = frozenset({"verified", "falsified", "unknown", "timeout", "error"})
REQUIRED_COVERAGE_TAGS = frozenset(
    {
        "residual_dag",
        "tinyimagenet",
        "cgan_target",
        "cgan_sentinel",
        "vgg_target",
        "vgg_sentinel",
        "vit_nonlinear_dag",
        "pure_chain_conv",
        "cuts_active",
        "conjunctive_multiobjective",
        "input_split",
        "relu_split",
        "acas_cpu",
        "soundness_regression",
    }
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PHASE_RE = re.compile(
    r"^\[phase\] (?P<name>.+) t=(?P<seconds>[0-9]+(?:\.[0-9]+)?)s$"
)
FLOAT_PATTERN = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
FRONTIER_RE = re.compile(
    rf"^\[frontier\] d=(?P<depth>[0-9]+) "
    rf"worst=(?P<worst>{FLOAT_PATTERN}) "
    rf"domains=(?P<domains>[0-9]+) "
    rf"t=(?P<seconds>[0-9]+(?:\.[0-9]+)?)s$"
)

PASS_FIELDS = (
    "bound_passes",
    "gradient_passes",
    "gpu_calls",
    "synchronizations",
    "fallbacks",
    "productive_passes_discarded",
)
BATCHING_FIELDS = (
    "backoff_count",
)
QUEUE_FIELDS = (
    "bytes_per_domain",
    "copied_bytes",
    "allocations",
    "alpha_bytes",
    "retained_la_bytes",
    "dropped_la_bytes",
)
MEMORY_INTEGER_FIELDS = (
    "peak_host_bytes",
    "peak_device_bytes",
    "swap_bytes",
    "oom_count",
)
DEADLINE_INTEGER_FIELDS = ("nonfinite_refusals",)


class BaselineError(ValueError):
    """A corpus or observation cannot be bound unambiguously."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path: Path, payload: object) -> None:
    """Create a JSON artifact exactly once, matching NY evidence conventions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace immutable evidence: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_json_bytes(payload))
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _expect_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise BaselineError(f"{context} is missing keys: {', '.join(missing)}")
    if unknown:
        raise BaselineError(f"{context} has unknown keys: {', '.join(unknown)}")


def _safe_relative_path(raw: object, context: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise BaselineError(f"{context} must be a non-empty relative path")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or str(path) != raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BaselineError(f"{context} must be a normalized relative path: {raw!r}")
    return raw


def _string_list(raw: object, context: str) -> list[str]:
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(item, str) or not item for item in raw)
    ):
        raise BaselineError(f"{context} must be a non-empty list of strings")
    if len(set(raw)) != len(raw):
        raise BaselineError(f"{context} contains duplicates")
    return raw


def _positive_integer(raw: object, context: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise BaselineError(f"{context} must be a positive integer")
    return raw


def _validate_expected(raw: object, context: str) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise BaselineError(f"{context} must be an object")
    _expect_keys(
        raw,
        required={"model_sha256", "property_sha256"},
        optional={"expected_result"},
        context=context,
    )
    result: dict[str, str] = {}
    for key in ("model_sha256", "property_sha256"):
        value = raw[key]
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise BaselineError(f"{context}.{key} must be lowercase SHA-256")
        result[key] = value
    if "expected_result" in raw:
        expected_result = raw["expected_result"]
        if expected_result not in {"verified", "falsified"}:
            raise BaselineError(
                f"{context}.expected_result must be 'verified' or 'falsified'"
            )
        result["expected_result"] = expected_result
    return result


def _validate_source_identity(raw: dict[str, Any], context: str) -> None:
    """Validate an optional, inseparable benchmark repository/commit binding."""
    repository = raw.get("source_repository")
    commit = raw.get("source_commit")
    if repository is None and commit is None:
        return
    if repository is None or commit is None:
        raise BaselineError(
            f"{context}.source_repository and source_commit must be provided together"
        )
    if not isinstance(repository, str):
        raise BaselineError(f"{context}.source_repository must be an HTTPS URL")
    try:
        parsed = urlsplit(repository)
        port = parsed.port
    except ValueError:
        parsed = None
        port = None
    if (
        parsed is None
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or parsed.geturl() != repository
    ):
        raise BaselineError(
            f"{context}.source_repository must be a canonical HTTPS repository URL"
        )
    if not isinstance(commit, str) or GIT_COMMIT_RE.fullmatch(commit) is None:
        raise BaselineError(
            f"{context}.source_commit must be a lowercase 40-hex Git commit"
        )


def _validate_entry(raw: object, index: int) -> dict[str, Any]:
    context = f"entries[{index}]"
    if not isinstance(raw, dict):
        raise BaselineError(f"{context} must be an object")
    kind = raw.get("kind")
    common = {"id", "kind", "role", "tags", "timeout_seconds"}
    if kind == "vnncomp":
        _expect_keys(
            raw,
            required=common
            | {
                "category",
                "source_index",
                "model",
                "property",
                "preset",
                "expected",
            },
            optional={"notes", "source_repository", "source_commit"},
            context=context,
        )
    elif kind == "repository_pair":
        _expect_keys(
            raw,
            required=common | {"model", "property", "command", "expected"},
            optional={"notes"},
            context=context,
        )
    elif kind == "repository_test":
        _expect_keys(
            raw,
            required=common | {"artifacts", "command"},
            optional={"notes"},
            context=context,
        )
    else:
        raise BaselineError(
            f"{context}.kind must be one of {sorted(ENTRY_KINDS)}, got {kind!r}"
        )

    entry_id = raw["id"]
    if (
        not isinstance(entry_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9.-]*", entry_id) is None
    ):
        raise BaselineError(f"{context}.id is not a safe stable identifier")
    role = raw["role"]
    if role not in ENTRY_ROLES:
        raise BaselineError(f"{context}.role must be one of {sorted(ENTRY_ROLES)}")
    tags = _string_list(raw["tags"], f"{context}.tags")
    timeout = _positive_integer(raw["timeout_seconds"], f"{context}.timeout_seconds")

    result = dict(raw)
    result["tags"] = tags
    result["timeout_seconds"] = timeout
    if kind in {"vnncomp", "repository_pair"}:
        result["model"] = _safe_relative_path(raw["model"], f"{context}.model")
        result["property"] = _safe_relative_path(
            raw["property"], f"{context}.property"
        )
        result["expected"] = _validate_expected(
            raw["expected"], f"{context}.expected"
        )
    if kind == "vnncomp":
        category = raw["category"]
        if (
            not isinstance(category, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", category) is None
        ):
            raise BaselineError(f"{context}.category is unsafe")
        result["source_index"] = _positive_integer(
            raw["source_index"], f"{context}.source_index"
        )
        result["preset"] = _safe_relative_path(
            raw["preset"], f"{context}.preset"
        )
        _validate_source_identity(raw, context)
    else:
        result["command"] = _string_list(raw["command"], f"{context}.command")
    if kind == "repository_test":
        result["artifacts"] = [
            _safe_relative_path(item, f"{context}.artifacts")
            for item in _string_list(raw["artifacts"], f"{context}.artifacts")
        ]
    if "notes" in raw and (not isinstance(raw["notes"], str) or not raw["notes"]):
        raise BaselineError(f"{context}.notes must be a non-empty string")
    return result


def load_corpus_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        data = path.read_bytes()
        raw = json.loads(data)
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineError(f"cannot read corpus manifest {path}: {error}") from error
    if not isinstance(raw, dict):
        raise BaselineError("corpus manifest root must be an object")
    _expect_keys(
        raw,
        required={"schema", "name", "description", "entries"},
        optional=set(),
        context="corpus manifest",
    )
    if raw["schema"] != CORPUS_SCHEMA:
        raise BaselineError(
            f"unsupported corpus schema {raw['schema']!r}; expected {CORPUS_SCHEMA!r}"
        )
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise BaselineError("corpus manifest name must be non-empty")
    if not isinstance(raw["description"], str) or not raw["description"]:
        raise BaselineError("corpus manifest description must be non-empty")
    if not isinstance(raw["entries"], list) or not raw["entries"]:
        raise BaselineError("corpus manifest entries must be non-empty")
    entries = [
        _validate_entry(entry, index)
        for index, entry in enumerate(raw["entries"])
    ]
    ids = [entry["id"] for entry in entries]
    if len(set(ids)) != len(ids):
        raise BaselineError("corpus manifest entry IDs must be unique")
    observed_tags = {tag for entry in entries for tag in entry["tags"]}
    missing_tags = sorted(REQUIRED_COVERAGE_TAGS - observed_tags)
    if missing_tags:
        raise BaselineError(
            "corpus manifest does not cover required M0 surfaces: "
            + ", ".join(missing_tags)
        )
    return {**raw, "entries": entries}, data


def _file_identity(path: Path, *, declared_path: str) -> dict[str, Any]:
    return {
        "declared_path": declared_path,
        "resolved_path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _check_expected_identity(
    identity: dict[str, Any],
    expected_sha256: str,
    *,
    entry_id: str,
    label: str,
    errors: list[str],
) -> None:
    observed = identity["sha256"]
    if observed != expected_sha256:
        errors.append(
            f"{entry_id}: {label} SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed}"
        )


def _read_instances_row(path: Path, source_index: int) -> tuple[str, str, int]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for index, row in enumerate(csv.reader(handle), start=1):
                if index != source_index:
                    continue
                if len(row) < 3:
                    raise BaselineError(
                        f"{path} row {source_index} has fewer than three fields"
                    )
                try:
                    timeout = int(float(row[2].strip()))
                except ValueError as error:
                    raise BaselineError(
                        f"{path} row {source_index} has invalid timeout {row[2]!r}"
                    ) from error
                return row[0].strip(), row[1].strip(), timeout
    except OSError as error:
        raise BaselineError(f"cannot read {path}: {error}") from error
    raise BaselineError(f"{path} has no source row {source_index}")


def _resolve_vnncomp_entry(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    benchmark_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    category_root = benchmark_root / entry["category"]
    instances_path = category_root / "instances.csv"
    preset_path = repo_root / entry["preset"]
    if not preset_path.is_file():
        errors.append(f"{entry['id']}: missing tracked preset {entry['preset']}")
        preset_identity = None
    else:
        preset_identity = _file_identity(
            preset_path, declared_path=entry["preset"]
        )

    skip_reasons: list[str] = []
    files: dict[str, Any] = {"preset": preset_identity}
    if not instances_path.is_file():
        skip_reasons.append("benchmark_category_unavailable")
    else:
        try:
            observed_row = _read_instances_row(instances_path, entry["source_index"])
        except BaselineError as error:
            errors.append(f"{entry['id']}: {error}")
        else:
            expected_row = (
                entry["model"],
                entry["property"],
                entry["timeout_seconds"],
            )
            if observed_row != expected_row:
                errors.append(
                    f"{entry['id']}: instances.csv row {entry['source_index']} "
                    f"identity mismatch: expected {expected_row!r}, "
                    f"observed {observed_row!r}"
                )
            files["instances_csv"] = _file_identity(
                instances_path,
                declared_path=f"{entry['category']}/instances.csv",
            )

        for label in ("model", "property"):
            relative = entry[label]
            path = category_root / relative
            if not path.is_file():
                reason = f"{label}_not_materialized"
                if label == "model" and Path(f"{path}.gz").is_file():
                    reason = "model_only_available_compressed"
                skip_reasons.append(reason)
                files[label] = None
                continue
            identity = _file_identity(
                path,
                declared_path=f"{entry['category']}/{relative}",
            )
            files[label] = identity
            _check_expected_identity(
                identity,
                entry["expected"][f"{label}_sha256"],
                entry_id=entry["id"],
                label=label,
                errors=errors,
            )

    return (
        {
            **entry,
            "status": "skipped" if skip_reasons else "ready",
            "skip_reasons": sorted(set(skip_reasons)),
            "files": files,
        },
        errors,
    )


def _resolve_repository_entry(
    entry: dict[str, Any], *, repo_root: Path
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    files: dict[str, Any] = {}
    if entry["kind"] == "repository_pair":
        for label in ("model", "property"):
            relative = entry[label]
            path = repo_root / relative
            if not path.is_file():
                errors.append(
                    f"{entry['id']}: missing tracked repository {label} {relative}"
                )
                files[label] = None
                continue
            identity = _file_identity(path, declared_path=relative)
            files[label] = identity
            _check_expected_identity(
                identity,
                entry["expected"][f"{label}_sha256"],
                entry_id=entry["id"],
                label=label,
                errors=errors,
            )
    else:
        artifact_files: list[dict[str, Any]] = []
        for relative in entry["artifacts"]:
            path = repo_root / relative
            if not path.is_file():
                errors.append(
                    f"{entry['id']}: missing tracked repository artifact {relative}"
                )
                continue
            artifact_files.append(_file_identity(path, declared_path=relative))
        files["artifacts"] = artifact_files
    return (
        {
            **entry,
            "status": "ready" if not errors else "invalid",
            "skip_reasons": [],
            "files": files,
        },
        errors,
    )


def resolve_corpus(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    benchmark_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    benchmark_root = benchmark_root.resolve()
    resolved: list[dict[str, Any]] = []
    errors: list[str] = []
    for entry in manifest["entries"]:
        if entry["kind"] == "vnncomp":
            item, item_errors = _resolve_vnncomp_entry(
                entry,
                repo_root=repo_root,
                benchmark_root=benchmark_root,
            )
        else:
            item, item_errors = _resolve_repository_entry(
                entry, repo_root=repo_root
            )
        resolved.append(item)
        errors.extend(item_errors)
    counts = {
        status: sum(item["status"] == status for item in resolved)
        for status in ("ready", "skipped", "invalid")
    }
    return {
        "repo_root": str(repo_root),
        "benchmark_root": str(benchmark_root),
        "counts": counts,
        "entries": resolved,
        "validation_errors": errors,
    }


def metric_contract() -> dict[str, Any]:
    """Return the stable result-field contract recorded in every M0 baseline."""
    return {
        "schema": ROW_SCHEMA,
        "telemetry_environment": {"NY_PHASE_TELEMETRY": "1"},
        "outcome_fields": [
            "verdict",
            "solved_count",
            "raw_result",
            "wall_seconds",
            "domains_explored",
            "domains_per_second",
            "first_bab_seconds",
            "root_bound_vector",
            "active_objective_rows",
        ],
        "phase_fields": ["events", "intervals", "totals_seconds"],
        "frontier_fields": [
            "frames",
            "maximum_depth",
            "matched_depth_worst_margin",
        ],
        "pass_fields": list(PASS_FIELDS),
        "batching_fields": [*BATCHING_FIELDS, "batch_size_histogram"],
        "queue_fields": list(QUEUE_FIELDS),
        "memory_fields": [*MEMORY_INTEGER_FIELDS, "gpu_utilization_percent"],
        "deadline_fields": [
            "overrun_seconds",
            "watchdog_hit",
            *DEADLINE_INTEGER_FIELDS,
        ],
        "missing_value_policy": (
            "Unavailable counters remain explicit nulls and are listed in "
            "metrics.unavailable; they are never inferred as zero."
        ),
    }


def _provenance_module() -> ModuleType:
    path = Path(__file__).with_name("ny_measurement_provenance.py")
    spec = importlib.util.spec_from_file_location(
        "ny_measurement_provenance_for_transfer_baseline", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load provenance helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture_baseline(
    *,
    manifest_path: Path,
    repo_root: Path,
    benchmark_root: Path,
    binary: Path,
    run_id: str,
    output_dir: Path,
    scratch_dir: Path | None = None,
    provenance_module: ModuleType | None = None,
) -> Path:
    if os.environ.get("NY_PHASE_TELEMETRY") != "1":
        raise BaselineError(
            "capture requires NY_PHASE_TELEMETRY=1 so phase/frontier absence "
            "cannot be confused with an ungated run"
        )
    repo_root = repo_root.resolve()
    benchmark_root = (
        benchmark_root
        if benchmark_root.is_absolute()
        else repo_root / benchmark_root
    ).resolve()
    manifest_path = (
        manifest_path if manifest_path.is_absolute() else repo_root / manifest_path
    ).resolve()
    manifest, manifest_bytes = load_corpus_manifest(manifest_path)
    first_resolution = resolve_corpus(
        manifest, repo_root=repo_root, benchmark_root=benchmark_root
    )
    if first_resolution["validation_errors"]:
        raise BaselineError("; ".join(first_resolution["validation_errors"]))

    output_dir = (
        output_dir if output_dir.is_absolute() else repo_root / output_dir
    ).resolve()
    artifact_root = output_dir / "artifacts"
    if scratch_dir is None:
        scratch_dir = Path(tempfile.gettempdir()) / f"ny-abcrown-transfer-{run_id}"
    scratch_dir = scratch_dir.resolve()
    categories = list(
        dict.fromkeys(
            entry["category"]
            for entry in manifest["entries"]
            if entry["kind"] == "vnncomp"
        )
    )
    timeout_cap = max(entry["timeout_seconds"] for entry in manifest["entries"])
    provenance = provenance_module or _provenance_module()
    try:
        start_path = provenance.capture_start_manifest(
            repo_root=repo_root,
            binary=binary,
            benchmark_root=benchmark_root,
            artifact_root=artifact_root,
            run_id=run_id,
            output_dir=output_dir,
            scratch_dir=scratch_dir,
            result_file=scratch_dir / "result.txt",
            solver_log_file=scratch_dir / "solver.log",
            categories_raw=" ".join(categories),
            timeout_cap_seconds=timeout_cap,
            watchdog_grace_seconds=5,
            max_rows_per_category=0,
            instance_index=0,
            vnnlib_version="",
            sweep_script=Path(__file__),
            configs_dir=repo_root / "configs",
        )
    except Exception as error:
        provenance_error = getattr(provenance, "ProvenanceError", None)
        expected = isinstance(error, (ValueError, OSError)) or (
            isinstance(provenance_error, type)
            and isinstance(error, provenance_error)
        )
        if expected:
            raise BaselineError(
                f"measurement provenance capture failed: {error}"
            ) from error
        raise
    second_resolution = resolve_corpus(
        manifest, repo_root=repo_root, benchmark_root=benchmark_root
    )
    if second_resolution != first_resolution:
        raise BaselineError(
            "corpus assets changed while immutable provenance was captured"
        )
    start_path = Path(start_path).resolve()
    baseline_path = start_path.with_name("transfer-baseline.json")
    payload = {
        "schema": BASELINE_SCHEMA,
        "run_id": run_id,
        "captured_at_utc": _utc_now(),
        "measurement_start": {
            "path": str(start_path),
            "sha256": _sha256_file(start_path),
        },
        "corpus_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_bytes),
        },
        "metric_contract": metric_contract(),
        "resolution": first_resolution,
    }
    writer = getattr(provenance, "_write_immutable", None)
    serializer = getattr(provenance, "_json_bytes", None)
    if callable(writer) and callable(serializer):
        writer(baseline_path, serializer(payload))
    else:
        _write_immutable(baseline_path, payload)
    return baseline_path


def parse_telemetry(log_text: str) -> dict[str, Any]:
    phases: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        phase_match = PHASE_RE.fullmatch(line.strip())
        if phase_match is not None:
            seconds = float(phase_match.group("seconds"))
            if phases and seconds < phases[-1]["seconds"]:
                raise BaselineError("phase telemetry timestamps are not monotonic")
            phases.append(
                {
                    "name": phase_match.group("name"),
                    "seconds": seconds,
                }
            )
            continue
        frontier_match = FRONTIER_RE.fullmatch(line.strip())
        if frontier_match is None:
            continue
        seconds = float(frontier_match.group("seconds"))
        worst = float(frontier_match.group("worst"))
        if not math.isfinite(worst):
            raise BaselineError("frontier worst margin must be finite")
        if frontier and seconds < frontier[-1]["seconds"]:
            raise BaselineError("frontier telemetry timestamps are not monotonic")
        frontier.append(
            {
                "depth": int(frontier_match.group("depth")),
                "worst_margin": worst,
                "domains_cumulative": int(frontier_match.group("domains")),
                "seconds": seconds,
            }
        )
    intervals = [
        {
            "from": previous["name"],
            "to": current["name"],
            "seconds": current["seconds"] - previous["seconds"],
        }
        for previous, current in zip(phases, phases[1:])
    ]
    return {
        "phase": {"events": phases, "intervals": intervals},
        "frontier": {"frames": frontier},
    }


def _optional_nonnegative_number(raw: object, context: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BaselineError(f"{context} must be numeric or null")
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise BaselineError(f"{context} must be finite and nonnegative")
    return value


def _optional_nonnegative_integer(raw: object, context: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise BaselineError(f"{context} must be a nonnegative integer or null")
    return raw


def _validate_numeric_group(
    raw: object,
    *,
    fields: tuple[str, ...],
    context: str,
    integer: bool = True,
    extra_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    if raw is None:
        return dict.fromkeys(fields)
    if not isinstance(raw, dict):
        raise BaselineError(f"{context} must be an object")
    _expect_keys(
        raw,
        required=set(),
        optional=set(fields) | set(extra_fields),
        context=context,
    )
    validator = (
        _optional_nonnegative_integer
        if integer
        else _optional_nonnegative_number
    )
    return {
        field: validator(raw.get(field), f"{context}.{field}") for field in fields
    }


def load_supplemental_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        raw: dict[str, Any] = {"schema": SUPPLEMENTAL_SCHEMA}
    else:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BaselineError(
                f"cannot read supplemental metrics {path}: {error}"
            ) from error
        if not isinstance(loaded, dict):
            raise BaselineError("supplemental metrics root must be an object")
        raw = loaded
    _expect_keys(
        raw,
        required={"schema"},
        optional={
            "phase_totals_seconds",
            "root_bound_vector",
            "matched_depth_worst_margin",
            "domains_explored",
            "first_bab_seconds",
            "active_objective_rows",
            "passes",
            "batching",
            "queue",
            "memory",
            "deadline",
        },
        context="supplemental metrics",
    )
    if raw["schema"] != SUPPLEMENTAL_SCHEMA:
        raise BaselineError(
            f"unsupported supplemental schema {raw['schema']!r}; "
            f"expected {SUPPLEMENTAL_SCHEMA!r}"
        )

    phase_totals = raw.get("phase_totals_seconds", {})
    if not isinstance(phase_totals, dict):
        raise BaselineError("phase_totals_seconds must be an object")
    normalized_phase_totals: dict[str, float] = {}
    for name, value in phase_totals.items():
        if not isinstance(name, str) or not name:
            raise BaselineError("phase total names must be non-empty strings")
        normalized = _optional_nonnegative_number(
            value, f"phase_totals_seconds.{name}"
        )
        if normalized is None:
            raise BaselineError("phase total values cannot be null")
        normalized_phase_totals[name] = normalized

    vector = raw.get("root_bound_vector")
    if vector is not None:
        if not isinstance(vector, list):
            raise BaselineError("root_bound_vector must be an array or null")
        normalized_vector = []
        for index, value in enumerate(vector):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BaselineError(f"root_bound_vector[{index}] must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise BaselineError(f"root_bound_vector[{index}] must be finite")
            normalized_vector.append(numeric)
        vector = normalized_vector

    margins = raw.get("matched_depth_worst_margin", {})
    if not isinstance(margins, dict):
        raise BaselineError("matched_depth_worst_margin must be an object")
    normalized_margins: dict[str, float] = {}
    for depth, value in margins.items():
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", depth) is None:
            raise BaselineError(
                "matched_depth_worst_margin keys must be canonical depths"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BaselineError(
                f"matched_depth_worst_margin.{depth} must be numeric"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise BaselineError(
                f"matched_depth_worst_margin.{depth} must be finite"
            )
        normalized_margins[depth] = numeric

    batching = _validate_numeric_group(
        raw.get("batching"),
        fields=BATCHING_FIELDS,
        context="batching",
        extra_fields=("batch_size_histogram",),
    )
    histogram: dict[str, int] | None = None
    if isinstance(raw.get("batching"), dict):
        batching_raw = raw["batching"]
        histogram_raw = batching_raw.get("batch_size_histogram")
        if histogram_raw is not None:
            if not isinstance(histogram_raw, dict):
                raise BaselineError("batch_size_histogram must be an object")
            histogram = {}
            for batch_size, count in histogram_raw.items():
                if re.fullmatch(r"[1-9][0-9]*", batch_size) is None:
                    raise BaselineError(
                        "batch_size_histogram keys must be positive integers"
                    )
                histogram[batch_size] = _optional_nonnegative_integer(
                    count, f"batch_size_histogram.{batch_size}"
                )
                if histogram[batch_size] is None:
                    raise BaselineError("batch_size_histogram counts cannot be null")
    batching["batch_size_histogram"] = histogram

    memory = _validate_numeric_group(
        raw.get("memory"),
        fields=MEMORY_INTEGER_FIELDS,
        context="memory",
        extra_fields=("gpu_utilization_percent",),
    )
    gpu_utilization = None
    if isinstance(raw.get("memory"), dict):
        memory_raw = raw["memory"]
        gpu_utilization = _optional_nonnegative_number(
            memory_raw.get("gpu_utilization_percent"),
            "memory.gpu_utilization_percent",
        )
        if gpu_utilization is not None and gpu_utilization > 100:
            raise BaselineError("memory.gpu_utilization_percent must be <= 100")
    memory["gpu_utilization_percent"] = gpu_utilization

    deadline = _validate_numeric_group(
        raw.get("deadline"),
        fields=DEADLINE_INTEGER_FIELDS,
        context="deadline",
        extra_fields=("overrun_seconds", "watchdog_hit"),
    )
    overrun = None
    watchdog = None
    if isinstance(raw.get("deadline"), dict):
        deadline_raw = raw["deadline"]
        overrun = _optional_nonnegative_number(
            deadline_raw.get("overrun_seconds"), "deadline.overrun_seconds"
        )
        watchdog = deadline_raw.get("watchdog_hit")
        if watchdog is not None and not isinstance(watchdog, bool):
            raise BaselineError("deadline.watchdog_hit must be boolean or null")
    deadline["overrun_seconds"] = overrun
    deadline["watchdog_hit"] = watchdog

    return {
        "phase_totals_seconds": normalized_phase_totals,
        "root_bound_vector": vector,
        "matched_depth_worst_margin": normalized_margins,
        "domains_explored": _optional_nonnegative_integer(
            raw.get("domains_explored"), "domains_explored"
        ),
        "first_bab_seconds": _optional_nonnegative_number(
            raw.get("first_bab_seconds"), "first_bab_seconds"
        ),
        "active_objective_rows": _optional_nonnegative_integer(
            raw.get("active_objective_rows"), "active_objective_rows"
        ),
        "passes": _validate_numeric_group(
            raw.get("passes"), fields=PASS_FIELDS, context="passes"
        ),
        "batching": batching,
        "queue": _validate_numeric_group(
            raw.get("queue"), fields=QUEUE_FIELDS, context="queue"
        ),
        "memory": memory,
        "deadline": deadline,
    }


def _parse_result(path: Path) -> tuple[str, str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise BaselineError(f"cannot read result file {path}: {error}") from error
    first = raw.splitlines()[0].strip().lower() if raw.splitlines() else ""
    mapping = {
        "unsat": "verified",
        "verified": "verified",
        "holds": "verified",
        "sat": "falsified",
        "falsified": "falsified",
        "violated": "falsified",
        "unknown": "unknown",
        "timeout": "timeout",
        "error": "error",
    }
    verdict = mapping.get(first)
    if verdict not in VERDICTS:
        raise BaselineError(f"unsupported or empty result token {first!r}")
    return verdict, first


def _verify_file_identity(identity: object, context: str) -> None:
    if not isinstance(identity, dict):
        raise BaselineError(f"{context} identity is missing")
    path_raw = identity.get("resolved_path")
    expected = identity.get("sha256")
    if not isinstance(path_raw, str) or not isinstance(expected, str):
        raise BaselineError(f"{context} identity is malformed")
    path = Path(path_raw)
    if not path.is_file():
        raise BaselineError(f"{context} disappeared after baseline capture: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise BaselineError(
            f"{context} changed after baseline capture: expected {expected}, "
            f"observed {observed}"
        )


def _collect_unavailable(metrics: dict[str, Any]) -> list[str]:
    unavailable: list[str] = []

    def visit(value: object, prefix: str) -> None:
        if value is None:
            unavailable.append(prefix)
        elif isinstance(value, dict):
            if not value:
                unavailable.append(prefix)
            for key, child in value.items():
                if key == "unavailable":
                    continue
                visit(child, f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, list) and not value:
            unavailable.append(prefix)

    visit(metrics, "")
    return sorted(item for item in unavailable if item)


def record_row(
    *,
    baseline_path: Path,
    entry_id: str,
    log_path: Path,
    result_path: Path,
    wall_seconds: float,
    output_path: Path,
    supplemental_path: Path | None = None,
) -> Path:
    wall_seconds = _optional_nonnegative_number(wall_seconds, "wall_seconds")
    assert wall_seconds is not None
    try:
        baseline_bytes = baseline_path.read_bytes()
        baseline = json.loads(baseline_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineError(f"cannot read baseline {baseline_path}: {error}") from error
    if not isinstance(baseline, dict) or baseline.get("schema") != BASELINE_SCHEMA:
        raise BaselineError(f"{baseline_path} is not a {BASELINE_SCHEMA} artifact")
    start = baseline.get("measurement_start")
    if not isinstance(start, dict):
        raise BaselineError("baseline lacks measurement_start identity")
    start_path_raw = start.get("path")
    start_sha = start.get("sha256")
    if not isinstance(start_path_raw, str) or not isinstance(start_sha, str):
        raise BaselineError("baseline measurement_start identity is malformed")
    start_path = Path(start_path_raw)
    if not start_path.is_file() or _sha256_file(start_path) != start_sha:
        raise BaselineError("measurement start provenance is missing or changed")

    resolution = baseline.get("resolution")
    if not isinstance(resolution, dict) or not isinstance(
        resolution.get("entries"), list
    ):
        raise BaselineError("baseline resolution is malformed")
    matching = [
        entry for entry in resolution["entries"] if entry.get("id") == entry_id
    ]
    if len(matching) != 1:
        raise BaselineError(f"baseline has no unique entry {entry_id!r}")
    entry = matching[0]
    if entry.get("status") != "ready":
        raise BaselineError(f"cannot record non-ready entry {entry_id!r}")
    files = entry.get("files")
    if not isinstance(files, dict):
        raise BaselineError(f"entry {entry_id!r} lacks file identities")
    if entry["kind"] in {"vnncomp", "repository_pair"}:
        for label in ("model", "property"):
            _verify_file_identity(files.get(label), f"{entry_id}.{label}")
    if entry["kind"] == "vnncomp":
        _verify_file_identity(files.get("preset"), f"{entry_id}.preset")
        _verify_file_identity(
            files.get("instances_csv"), f"{entry_id}.instances_csv"
        )
    if entry["kind"] == "repository_test":
        artifacts = files.get("artifacts")
        if not isinstance(artifacts, list):
            raise BaselineError(f"{entry_id}.artifacts identity is malformed")
        for index, identity in enumerate(artifacts):
            _verify_file_identity(identity, f"{entry_id}.artifacts[{index}]")

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise BaselineError(f"cannot read solver log {log_path}: {error}") from error
    telemetry = parse_telemetry(log_text)
    supplement = load_supplemental_metrics(supplemental_path)
    verdict, raw_result = _parse_result(result_path)
    frames = telemetry["frontier"]["frames"]
    domains = supplement["domains_explored"]
    if domains is None and frames:
        domains = max(frame["domains_cumulative"] for frame in frames)
    if domains is not None and frames:
        observed = max(frame["domains_cumulative"] for frame in frames)
        if domains < observed:
            raise BaselineError(
                f"domains_explored {domains} is below frontier cumulative {observed}"
            )
    first_bab = supplement["first_bab_seconds"]
    if first_bab is None and frames:
        first_bab = frames[0]["seconds"]
    metrics = {
        "outcome": {
            "verdict": verdict,
            "solved_count": int(verdict in {"verified", "falsified"}),
            "raw_result": raw_result,
            "wall_seconds": wall_seconds,
            "domains_explored": domains,
            "domains_per_second": (
                domains / wall_seconds
                if domains is not None and wall_seconds > 0
                else None
            ),
            "first_bab_seconds": first_bab,
            "root_bound_vector": supplement["root_bound_vector"],
            "active_objective_rows": supplement["active_objective_rows"],
        },
        "phase": {
            **telemetry["phase"],
            "totals_seconds": supplement["phase_totals_seconds"],
        },
        "frontier": {
            **telemetry["frontier"],
            "maximum_depth": (
                max(frame["depth"] for frame in frames) if frames else None
            ),
            "matched_depth_worst_margin": supplement[
                "matched_depth_worst_margin"
            ],
        },
        "passes": supplement["passes"],
        "batching": supplement["batching"],
        "queue": supplement["queue"],
        "memory": supplement["memory"],
        "deadline": supplement["deadline"],
    }
    metrics["unavailable"] = _collect_unavailable(metrics)
    payload = {
        "schema": ROW_SCHEMA,
        "run_id": baseline["run_id"],
        "entry_id": entry_id,
        "recorded_at_utc": _utc_now(),
        "baseline": {
            "path": str(baseline_path.resolve()),
            "sha256": _sha256(baseline_bytes),
        },
        "artifacts": {
            "log": _file_identity(log_path, declared_path=str(log_path)),
            "result": _file_identity(result_path, declared_path=str(result_path)),
            "supplemental_metrics": (
                _file_identity(
                    supplemental_path, declared_path=str(supplemental_path)
                )
                if supplemental_path is not None
                else None
            ),
        },
        "metrics": metrics,
    }
    _write_immutable(output_path, payload)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate and resolve the fixed corpus"
    )
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    validate.add_argument("--repo-root", type=Path, default=Path("."))
    validate.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("benchmarks/vnncomp2025/benchmarks"),
    )
    validate.add_argument("--require-all-ready", action="store_true")
    validate.add_argument("--output", type=Path)

    capture = subparsers.add_parser(
        "capture", help="create immutable start provenance and corpus resolution"
    )
    capture.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    capture.add_argument("--repo-root", type=Path, default=Path("."))
    capture.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("benchmarks/vnncomp2025/benchmarks"),
    )
    capture.add_argument("--binary", type=Path, default=Path("target/release/ny"))
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--output-dir", type=Path)
    capture.add_argument("--scratch-dir", type=Path)

    record = subparsers.add_parser(
        "record", help="create one immutable canonical measurement row"
    )
    record.add_argument("--baseline", type=Path, required=True)
    record.add_argument("--entry-id", required=True)
    record.add_argument("--log", type=Path, required=True)
    record.add_argument("--result", type=Path, required=True)
    record.add_argument("--wall-seconds", type=float, required=True)
    record.add_argument("--supplemental-metrics", type=Path)
    record.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            repo_root = args.repo_root.resolve()
            manifest_path = (
                args.manifest
                if args.manifest.is_absolute()
                else repo_root / args.manifest
            )
            benchmark_root = (
                args.benchmark_root
                if args.benchmark_root.is_absolute()
                else repo_root / args.benchmark_root
            )
            manifest, manifest_bytes = load_corpus_manifest(manifest_path)
            resolution = resolve_corpus(
                manifest,
                repo_root=repo_root,
                benchmark_root=benchmark_root,
            )
            payload = {
                "schema": CORPUS_SCHEMA,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_bytes),
                "metric_contract": metric_contract(),
                "resolution": resolution,
            }
            if resolution["validation_errors"]:
                raise BaselineError("; ".join(resolution["validation_errors"]))
            if args.require_all_ready and resolution["counts"]["skipped"]:
                raise BaselineError(
                    f"{resolution['counts']['skipped']} corpus entries are skipped"
                )
            if args.output is not None:
                _write_immutable(args.output, payload)
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "capture":
            output_dir = args.output_dir or Path(
                f"reports/benchmarks/abcrown-transfer/{args.run_id}"
            )
            path = capture_baseline(
                manifest_path=args.manifest,
                repo_root=args.repo_root,
                benchmark_root=args.benchmark_root,
                binary=args.binary,
                run_id=args.run_id,
                output_dir=output_dir,
                scratch_dir=args.scratch_dir,
            )
            print(path)
            return 0
        if args.command == "record":
            path = record_row(
                baseline_path=args.baseline,
                entry_id=args.entry_id,
                log_path=args.log,
                result_path=args.result,
                wall_seconds=args.wall_seconds,
                supplemental_path=args.supplemental_metrics,
                output_path=args.output,
            )
            print(path)
            return 0
    except (BaselineError, FileExistsError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
