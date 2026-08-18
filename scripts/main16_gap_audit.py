#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Audit the VNN-COMP 2025 regular-16 gap without mixing evidence tiers.

The legacy section is an explicitly optimistic, frozen-denominator projection
over old measured CSVs.  The exact-current section is restricted to immutable
measurement completions captured at ``--exact-commit`` under the explicitly
listed ``--artifact-root`` directories.  SAT rows require an immutable replay
sidecar before they receive either credit or an incorrect-result penalty.
Exact-2025 SAT qualification additionally requires ``--benchmark-root`` so the
sidecar can be rebound to the pinned official Git payloads.

Examples::

  python3 scripts/main16_gap_audit.py \
    --official /data/vnncomp2025_results \
    --benchmark-root /data/vnncomp2025_benchmarks/benchmarks \
    --legacy-measured reports/measured \
    --artifact-root /data/ny-run/artifacts \
    --exact-commit "$(git rev-parse HEAD)"

  python3 scripts/main16_gap_audit.py ... --format json --json-out audit.json
  python3 scripts/main16_gap_audit.py ... --format csv --csv-out audit.csv

Exit status 1 means supplied exact-commit evidence was incomplete or ambiguous;
the emitted report remains useful, but every affected row is fail-closed to
unmeasured.  Invalid arguments or roots return 2.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ny_measurement_provenance as provenance  # noqa: E402
import ny_retroactive_scorecard as retro  # noqa: E402
import vnncomp_competitive_score as competitive  # noqa: E402

SCHEMA = "ny_main16_gap_audit_v1"
EXACT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
LEGACY_TIER = "legacy_optimistic_projection"
QUALIFIED_TIER = "exact_commit_sealed_evidence"
METRIC = "published_2025_zero_tol_frozen_denominator_projection"


class AuditError(RuntimeError):
    """The requested audit cannot be interpreted unambiguously."""


@dataclass(frozen=True)
class OfficialContext:
    reference_order: dict[str, list[tuple]]
    ground_truth: dict[str, dict[tuple, str]]
    winner_points: dict[str, int]


@dataclass(frozen=True)
class SealedRecord:
    artifact_root: Path
    run_id: str
    category: str
    instance_index: int
    instance: tuple
    verdict: str
    counterexample: competitive.CounterexampleResult | None
    sat_replay_state: str


def _round_score(value: float) -> float:
    return round(value, 9)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AuditError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _strict_json_loads(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=_json_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except AuditError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AuditError(f"{label} is not strict JSON") from error


def _require_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise AuditError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise AuditError(f"{label} is unavailable: {path}") from error
    if not resolved.is_dir():
        raise AuditError(f"{label} is not a directory: {resolved}")
    return resolved


def _stable_json(path: Path, label: str) -> tuple[dict[str, Any], str, int]:
    if path.is_symlink():
        raise AuditError(f"{label} must not be a symlink: {path}")
    try:
        data, digest, fingerprint = provenance._stable_file_bytes(path)
        value = _strict_json_loads(data, label)
    except (
        OSError,
        UnicodeDecodeError,
        provenance.ProvenanceError,
    ) as error:
        raise AuditError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object: {path}")
    return value, digest, int(fingerprint["size_bytes"])


def _artifact_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise AuditError(f"invalid {label} artifact path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise AuditError(f"unsafe {label} artifact path: {value!r}")
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise AuditError(f"{label} artifact must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise AuditError(
            f"{label} artifact escapes or is missing from root: {value!r}"
        ) from error
    if not resolved.is_file():
        raise AuditError(f"{label} artifact is not a regular file: {resolved}")
    return resolved


def _checked_artifact(
    root: Path,
    evidence: object,
    label: str,
) -> tuple[Path, bytes, str, int]:
    if not isinstance(evidence, dict):
        raise AuditError(f"missing {label} evidence record")
    path = _artifact_path(root, evidence.get("artifact"), label)
    try:
        data, digest, fingerprint = provenance._stable_file_bytes(path)
    except (OSError, provenance.ProvenanceError) as error:
        raise AuditError(f"could not hash {label} artifact {path}: {error}") from error
    size = int(fingerprint["size_bytes"])
    if evidence.get("sha256") != digest or evidence.get("size_bytes") != size:
        raise AuditError(
            f"{label} artifact does not match its completion evidence: {path}"
        )
    return path, data, digest, size


def load_official_context(root: Path) -> OfficialContext:
    root = _require_directory(root, "official result root")
    reference_csv = root / "alpha_beta_crown" / "results.csv"
    scoring = root / "SCORING-ZERO-TOL" / "latex"
    longtable = scoring / "longtable.tex"
    scored = scoring / "scored.tex"
    for path in (reference_csv, longtable, scored):
        if path.is_symlink() or not path.is_file():
            raise AuditError(
                f"required official artifact is missing or a symlink: {path}"
            )
    try:
        reference = retro.load_reference_instance_order(reference_csv)
        truth = retro.load_published_ground_truth(longtable, reference)
        winners = retro.load_published_winner_points(scored)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise AuditError(
            f"invalid official result artifacts under {root}: {error}"
        ) from error
    missing = [category for category in retro.REGULAR if not reference.get(category)]
    if missing:
        raise AuditError(
            "official reference has no scored instances for: " + ", ".join(missing)
        )
    return OfficialContext(reference, truth, winners)


def load_legacy_results(
    root: Path, official: OfficialContext
) -> dict[str, dict[tuple, str]]:
    root = _require_directory(root, "legacy measured root")
    out: dict[str, dict[tuple, str]] = defaultdict(dict)
    for category in retro.REGULAR:
        path = root / f"{category}.csv"
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise AuditError(f"legacy measured file is not a regular file: {path}")
        loaded = retro.load_tool_csv(path)
        unexpected_categories = sorted(set(loaded) - {category})
        if unexpected_categories:
            raise AuditError(
                f"legacy file {path} contains other categories: "
                + ", ".join(unexpected_categories)
            )
        reference = set(official.reference_order[category])
        unexpected_rows = sorted(set(loaded.get(category, {})) - reference)
        if unexpected_rows:
            raise AuditError(
                f"legacy file {path} contains rows outside the official occurrence order"
            )
        out[category].update(loaded.get(category, {}))
    return out


def _min_extra_credits(raw: int, winner_raw: int) -> int:
    return max(0, math.ceil((winner_raw - raw) / retro.POINTS_CORRECT))


def _legacy_section(
    results: dict[str, dict[tuple, str]],
    official: OfficialContext,
) -> dict[str, Any]:
    total, breakdown = retro.published_artifact_projection(
        results,
        official.reference_order,
        official.ground_truth,
        official.winner_points,
        ny_sat_status="correct-up-to-tolerance",
    )
    suites: list[dict[str, Any]] = []
    for category in retro.REGULAR:
        # `retro.CategoryProjection`, read by FIELD NAME on purpose: this value
        # has grown twice (the field-falsified moat, then the unconfirmable
        # split) and each growth silently broke a positional unpack here.
        projection = breakdown[category]
        raw = projection.raw
        score = projection.normalized
        winner = projection.official_winner_raw
        credited = projection.credited
        incorrect = projection.contradictions
        assumed_sats = projection.assumed_sat_count
        total_rows = len(official.reference_order[category])
        suites.append(
            {
                "suite": category,
                "official_instances": total_rows,
                "credited": credited,
                "incorrect": incorrect,
                "unmeasured": total_rows
                - credited
                - incorrect
                - projection.unconfirmable,
                # Decided ny rows the published field never decided: they score
                # 0 because nothing could ever contradict them. Surfaced so the
                # audit's `unmeasured` bucket does not silently absorb them.
                "unconfirmable": projection.unconfirmable,
                "assumed_sat_credits": assumed_sats,
                "raw_points": raw,
                "official_winner_raw_points": winner,
                "score": _round_score(score),
                "gap_to_100": _round_score(100.0 - score),
                "min_extra_credits_to_100": _min_extra_credits(raw, winner),
            }
        )
    return {
        "evidence_tier": LEGACY_TIER,
        "is_exact_current_evidence": False,
        "sat_policy": "assume_every_legacy_sat_correct_up_to_tolerance",
        "warning": (
            "NOT QUALIFIED: legacy CSVs do not bind source, binary, inputs, or "
            "retained SAT replay evidence"
        ),
        "score": _round_score(total),
        "gap_to_perfect": _round_score(1600.0 - total),
        "min_extra_credits_to_perfect": sum(
            row["min_extra_credits_to_100"] for row in suites
        ),
        "credited": sum(row["credited"] for row in suites),
        "incorrect": sum(row["incorrect"] for row in suites),
        "unmeasured": sum(row["unmeasured"] for row in suites),
        "suites": suites,
    }


def _validate_start(
    start_path: Path,
    root: Path,
    exact_commit: str,
) -> tuple[dict[str, Any], str, int]:
    start, digest, size = _stable_json(start_path, "start manifest")
    run_id = start.get("run_id")
    if start.get("schema") != "ny_measurement_start_v1":
        raise AuditError(f"unsupported start manifest schema: {start_path}")
    if (
        not isinstance(run_id, str)
        or provenance.SAFE_COMPONENT.fullmatch(run_id) is None
    ):
        raise AuditError(f"invalid run ID in start manifest: {start_path}")
    if start_path.parent.name != run_id:
        raise AuditError(f"run directory and start run ID differ: {start_path}")
    ny = start.get("ny")
    measurement = start.get("measurement")
    if not isinstance(ny, dict) or not isinstance(measurement, dict):
        raise AuditError(
            f"start manifest lacks NY or measurement identity: {start_path}"
        )
    if ny.get("commit") != exact_commit:
        raise AuditError(
            f"internal: start manifest is not for requested commit: {start_path}"
        )
    tracked_diff_format = ny.get("tracked_diff_format")
    if tracked_diff_format not in {None, "ny_tracked_worktree_evidence_v2"}:
        raise AuditError(f"unsupported NY tracked-diff evidence format: {start_path}")
    if tracked_diff_format == "ny_tracked_worktree_evidence_v2" and (
        ny.get("tracked_diff_sha256") != provenance._sha256(b"")
        or ny.get("tracked_worktree_paths") != []
    ):
        raise AuditError(f"invalid clean NY tracked-diff evidence: {start_path}")
    if (
        ny.get("clean") is not True
        or ny.get("status_porcelain_v1_z_entries") != []
        or ny.get("tracked_diff_bytes") != 0
        or ny.get("untracked_files") != []
    ):
        raise AuditError(
            f"exact-commit run was captured from a dirty NY worktree: {start_path}"
        )
    declared_root = measurement.get("artifact_root")
    if not isinstance(declared_root, str):
        raise AuditError(f"start manifest has no artifact root: {start_path}")
    try:
        declared = Path(declared_root).resolve(strict=True)
    except OSError as error:
        raise AuditError(
            f"declared artifact root is unavailable: {declared_root}"
        ) from error
    if declared != root:
        raise AuditError(
            f"start manifest artifact root {declared} does not match caller root {root}"
        )
    return start, digest, size


def _validate_replay_sidecar(
    *,
    root: Path,
    start_path: Path,
    start_digest: str,
    start_size: int,
    run_id: str,
    record: dict[str, Any],
    metadata_path: Path,
    metadata_digest: str,
    metadata_size: int,
    result_path: Path,
    result_data: bytes,
    result_digest: str,
    result_size: int,
    official_evidence: Any | None = None,
    benchmark_evidence: Any | None = None,
    replay_session: dict[str, Any] | None = None,
) -> tuple[competitive.CounterexampleResult | None, str]:
    exact_sidecar = metadata_path.with_name(
        f"{metadata_path.stem}.vnncomp2025-zero-tol-validation.json"
    )
    retired_sidecar = metadata_path.with_name(
        f"{metadata_path.stem}.counterexample-validation.json"
    )
    if not exact_sidecar.exists():
        if retired_sidecar.exists():
            return None, "retired_2026_replay_sidecar_ignored"
        return None, "missing_replay_sidecar"
    if official_evidence is None or benchmark_evidence is None:
        return None, "exact_2025_replay_requires_pinned_benchmark_context"

    # Lazy import avoids the module-import cycle: regular_bank_evidence uses
    # this audit's sealed completion parser.
    import regular_bank_evidence as regular  # noqa: PLC0415

    category = record.get("category")
    instance_index = record.get("instance_index")
    assert isinstance(category, str)
    assert type(instance_index) is int
    try:
        occurrence, _ = regular._load_occurrence(
            category=category,
            instance_index=instance_index,
            benchmark=benchmark_evidence,
            official=official_evidence,
        )
        authoritative_inputs = {
            label: regular.authoritative_benchmark_input(
                benchmark=benchmark_evidence,
                category=category,
                declared_name=(
                    occurrence.onnx if label == "onnx" else occurrence.vnnlib
                ),
                label=label,
            )[0]
            for label in ("onnx", "vnnlib")
        }
        binding = regular.validate_exact_2025_sat_replay(
            root=root,
            metadata_path=metadata_path,
            metadata_digest=metadata_digest,
            metadata_size=metadata_size,
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
            official=official_evidence,
            benchmark=benchmark_evidence,
            authoritative_inputs=authoritative_inputs,
            replay_session=replay_session,
        )
    except regular.EvidenceError as error:
        raise AuditError(f"exact 2025 SAT replay is invalid: {error}") from error
    result = binding["official_result"]
    mapping = {
        "correct": competitive.CounterexampleResult.CORRECT,
        "correct_up_to_tolerance": (
            competitive.CounterexampleResult.CORRECT_UP_TO_TOLERANCE
        ),
        "no_ce": competitive.CounterexampleResult.NO_COUNTEREXAMPLE,
        "exec_doesnt_match": competitive.CounterexampleResult.EXEC_DOESNT_MATCH,
        "wrong_shape": competitive.CounterexampleResult.EXEC_DOESNT_MATCH,
        "spec_not_violated": competitive.CounterexampleResult.SPEC_NOT_VIOLATED,
    }
    return mapping[str(result)], f"exact_2025_zero_tol:{result}"


def _validate_record(
    *,
    root: Path,
    start_path: Path,
    start_digest: str,
    start_size: int,
    run_id: str,
    record: object,
    official: OfficialContext,
    official_evidence: Any | None = None,
    benchmark_evidence: Any | None = None,
    replay_session: dict[str, Any] | None = None,
) -> SealedRecord:
    if not isinstance(record, dict):
        raise AuditError(f"run {run_id} completion contains a non-object record")
    category = record.get("category")
    index = record.get("instance_index")
    verdict = record.get("solver_verdict")
    if (
        not isinstance(category, str)
        or type(index) is not int
        or index <= 0
        or not isinstance(verdict, str)
        or verdict not in provenance.STANDARD_SOLVER_VERDICTS
    ):
        raise AuditError(f"run {run_id} completion record identity is invalid")
    if category not in retro.REGULAR:
        # Extended records are integrity-checked but are outside this scorecard.
        reference_instance: tuple = ()
    else:
        rows = official.reference_order[category]
        if index > len(rows):
            raise AuditError(
                f"run {run_id} record {category}:{index} exceeds official row count"
            )
        reference_instance = rows[index - 1]
        if (
            retro.key(str(record.get("onnx", "")), str(record.get("vnnlib", "")))
            != reference_instance[:2]
        ):
            raise AuditError(
                f"run {run_id} record {category}:{index} does not match official order"
            )

    metadata_path, metadata_data, metadata_digest, metadata_size = _checked_artifact(
        root, record.get("metadata"), "metadata"
    )
    result_path, result_data, result_digest, result_size = _checked_artifact(
        root, record.get("result"), "raw result"
    )
    _checked_artifact(root, record.get("solver_log"), "solver log")
    preflight_path, _, preflight_digest, preflight_size = _checked_artifact(
        root, record.get("preflight"), "input preflight"
    )
    preflight = record.get("preflight")
    assert isinstance(preflight, dict)
    if (
        preflight.get("artifact") != preflight_path.relative_to(root).as_posix()
        or preflight.get("sha256") != preflight_digest
        or preflight.get("size_bytes") != preflight_size
    ):
        raise AuditError(f"run {run_id} preflight completion link is invalid")
    preflight_inputs = preflight.get("inputs")
    if not isinstance(preflight_inputs, dict) or set(preflight_inputs) != {
        "onnx",
        "vnnlib",
    }:
        raise AuditError(f"run {run_id} preflight input evidence is incomplete")
    for label in ("onnx", "vnnlib"):
        value = preflight_inputs[label]
        if not isinstance(value, dict):
            raise AuditError(f"run {run_id} {label} preflight evidence is invalid")
        sealed = _artifact_path(root, value.get("sealed_artifact"), f"sealed {label}")
        _, sealed_digest, _ = provenance._stable_file_bytes(sealed)
        if (
            value.get("original_sha256") != sealed_digest
            or value.get("sealed_sha256") != sealed_digest
        ):
            raise AuditError(
                f"run {run_id} sealed {label} bytes do not match preflight"
            )

    try:
        metadata = _strict_json_loads(metadata_data, "measurement metadata")
    except AuditError as error:
        raise AuditError(
            f"run {run_id} metadata is invalid JSON: {metadata_path}"
        ) from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != "ny_measurement_result_v2"
    ):
        raise AuditError(
            f"run {run_id} metadata schema is unsupported: {metadata_path}"
        )
    expected_start_artifact = start_path.relative_to(root).as_posix()
    if (
        metadata.get("run_id") != run_id
        or metadata.get("category") != category
        or metadata.get("instance_index") != index
        or metadata.get("solver_verdict") != verdict
        or metadata.get("start_manifest") != expected_start_artifact
        or metadata.get("start_manifest_sha256") != start_digest
        or metadata.get("result_artifact") != result_path.relative_to(root).as_posix()
        or metadata.get("result_sha256") != result_digest
        or metadata.get("raw_result_sha256") != result_digest
    ):
        raise AuditError(f"run {run_id} metadata does not bind its completion record")
    result_lines = result_data.splitlines()
    first_line = (
        b"".join(result_lines[0].split()).decode("utf-8", "replace").lower()
        if result_lines
        else ""
    )
    if first_line != verdict and not (
        verdict == "timeout" and first_line in {"", "timeout"}
    ):
        raise AuditError(f"run {run_id} raw result verdict differs from metadata")
    if verdict == "sat" and not provenance._structured_sat_assignment(result_lines[1:]):
        raise AuditError(f"run {run_id} SAT result lacks a structured assignment")

    counterexample = None
    sat_state = "not_sat"
    if verdict == "sat":
        counterexample, sat_state = _validate_replay_sidecar(
            root=root,
            start_path=start_path,
            start_digest=start_digest,
            start_size=start_size,
            run_id=run_id,
            record=record,
            metadata_path=metadata_path,
            metadata_digest=metadata_digest,
            metadata_size=metadata_size,
            result_path=result_path,
            result_data=result_data,
            result_digest=result_digest,
            result_size=result_size,
            official_evidence=official_evidence,
            benchmark_evidence=benchmark_evidence,
            replay_session=replay_session,
        )
    return SealedRecord(
        root,
        run_id,
        category,
        index,
        reference_instance,
        verdict,
        counterexample,
        sat_state,
    )


def _validate_completion(
    *,
    root: Path,
    start_path: Path,
    start: dict[str, Any],
    start_digest: str,
    start_size: int,
    official: OfficialContext,
    official_evidence: Any | None = None,
    benchmark_evidence: Any | None = None,
    replay_session: dict[str, Any] | None = None,
) -> list[SealedRecord]:
    completion_path = start_path.with_name("completion.json")
    if not completion_path.exists():
        raise AuditError(f"exact-commit run has no completion: {start_path.parent}")
    completion, _, _ = _stable_json(completion_path, "completion manifest")
    run_id = start["run_id"]
    integrity = completion.get("integrity")
    if (
        completion.get("schema") != "ny_measurement_completion_v1"
        or completion.get("run_id") != run_id
        or completion.get("start_manifest") != "start.json"
        or completion.get("start_manifest_sha256") != start_digest
        or completion.get("exit_status") != 0
        or completion.get("completed_successfully") is not True
        or not isinstance(integrity, dict)
        or integrity.get("status") != "valid"
        or integrity.get("violations") != []
    ):
        raise AuditError(
            f"exact-commit completion is not successfully integrity-qualified: {completion_path}"
        )
    checks = integrity.get("checks")
    cuda_runtime = checks.get("cuda_runtime") if isinstance(checks, dict) else None
    if not isinstance(cuda_runtime, dict) or cuda_runtime.get("status") != "valid":
        raise AuditError(
            f"completion lacks valid CUDA runtime identity: {completion_path}"
        )
    run_evidence = checks.get("run_evidence") if isinstance(checks, dict) else None
    if not isinstance(run_evidence, dict) or run_evidence.get("status") != "valid":
        raise AuditError(f"completion lacks valid run evidence: {completion_path}")
    records = run_evidence.get("records")
    if not isinstance(records, list):
        raise AuditError(f"completion run evidence has no records: {completion_path}")
    count = len(records)
    for field in (
        "metadata_count",
        "result_count",
        "solver_log_count",
        "preflight_count",
        "validated_record_count",
        "csv_row_count",
    ):
        if run_evidence.get(field) != count:
            raise AuditError(
                f"completion evidence count {field} differs from records: {completion_path}"
            )
    if run_evidence.get("records_sha256") != provenance._identity_sha256(records):
        raise AuditError(f"completion record digest differs: {completion_path}")
    return [
        _validate_record(
            root=root,
            start_path=start_path,
            start_digest=start_digest,
            start_size=start_size,
            run_id=run_id,
            record=record,
            official=official,
            official_evidence=official_evidence,
            benchmark_evidence=benchmark_evidence,
            replay_session=replay_session,
        )
        for record in records
    ]


def load_sealed_records(
    roots: list[Path],
    exact_commit: str,
    official: OfficialContext,
    *,
    official_evidence: Any | None = None,
    benchmark_evidence: Any | None = None,
) -> tuple[list[SealedRecord], dict[str, Any], bool]:
    resolved_roots = [
        _require_directory(root, "measurement artifact root") for root in roots
    ]
    if len(set(resolved_roots)) != len(resolved_roots):
        raise AuditError("measurement artifact roots must be unique")
    for left_index, left in enumerate(resolved_roots):
        for right in resolved_roots[left_index + 1 :]:
            if left in right.parents or right in left.parents:
                raise AuditError("measurement artifact roots must not overlap")

    records: list[SealedRecord] = []
    rejected: list[dict[str, str]] = []
    ignored_other_commit = 0
    accepted_runs: list[str] = []
    seen_run_ids: set[str] = set()
    duplicate_run_ids: set[str] = set()
    replay_session: dict[str, Any] = {}
    for root in sorted(resolved_roots, key=str):
        runs_dir = root / "runs"
        if not runs_dir.exists():
            continue
        if runs_dir.is_symlink() or not runs_dir.is_dir():
            rejected.append(
                {"run": str(runs_dir), "reason": "runs path is not a directory"}
            )
            continue
        for run_dir in sorted(runs_dir.iterdir(), key=lambda path: path.name):
            if run_dir.is_symlink() or not run_dir.is_dir():
                rejected.append(
                    {"run": str(run_dir), "reason": "run entry is not a directory"}
                )
                continue
            start_path = run_dir / "start.json"
            if not start_path.exists():
                rejected.append({"run": str(run_dir), "reason": "missing start.json"})
                continue
            try:
                unscoped_start, _, _ = _stable_json(start_path, "start manifest")
                ny = unscoped_start.get("ny")
                commit = ny.get("commit") if isinstance(ny, dict) else None
                if commit != exact_commit:
                    if isinstance(commit, str) and EXACT_COMMIT_RE.fullmatch(commit):
                        ignored_other_commit += 1
                        continue
                    raise AuditError("start manifest has no exact NY commit")
                start, start_digest, start_size = _validate_start(
                    start_path, root, exact_commit
                )
                run_id = str(start["run_id"])
                if run_id in seen_run_ids:
                    duplicate_run_ids.add(run_id)
                run_records = _validate_completion(
                    root=root,
                    start_path=start_path,
                    start=start,
                    start_digest=start_digest,
                    start_size=start_size,
                    official=official,
                    official_evidence=official_evidence,
                    benchmark_evidence=benchmark_evidence,
                    replay_session=replay_session,
                )
                seen_run_ids.add(run_id)
                accepted_runs.append(run_id)
                records.extend(
                    record for record in run_records if record.category in retro.REGULAR
                )
            except AuditError as error:
                rejected.append({"run": str(run_dir), "reason": str(error)})

    rejected.extend(
        {
            "run": run_id,
            "reason": "duplicate exact-commit run ID across artifact roots",
        }
        for run_id in sorted(duplicate_run_ids)
    )
    grouped: dict[tuple[str, int], list[SealedRecord]] = defaultdict(list)
    for record in records:
        if record.run_id in duplicate_run_ids:
            continue
        grouped[(record.category, record.instance_index)].append(record)
    ambiguous = [
        {
            "suite": identity[0],
            "instance_index": identity[1],
            "run_ids": sorted(record.run_id for record in matches),
        }
        for identity, matches in sorted(grouped.items())
        if len(matches) != 1
    ]
    usable = [matches[0] for matches in grouped.values() if len(matches) == 1]
    incomplete = bool(rejected or ambiguous)
    audit = {
        "status": "incomplete_fail_closed" if incomplete else "valid",
        "accepted_run_ids": sorted(
            run_id for run_id in accepted_runs if run_id not in duplicate_run_ids
        ),
        "ignored_other_commit_runs": ignored_other_commit,
        "rejected_runs": rejected,
        "ambiguous_rows": ambiguous,
    }
    if "snapshot" in replay_session:
        import regular_bank_evidence as regular  # noqa: PLC0415

        try:
            regular.revalidate_replay_session(replay_session)
        except regular.EvidenceError as error:
            raise AuditError(str(error)) from error
    return (
        sorted(usable, key=lambda row: (row.category, row.instance_index)),
        audit,
        incomplete,
    )


def _score_record(record: SealedRecord, truth: str) -> int | None:
    if record.verdict in {"unknown", "timeout", "error"}:
        return None
    if record.verdict == "sat" and record.counterexample is None:
        return None
    if (
        record.verdict == "sat"
        and record.counterexample == competitive.CounterexampleResult.CORRECT
        and truth == "holds"
    ):
        # A strict new witness changes the published field truth and can
        # penalize incumbent holds results, changing the frozen denominator.
        # This projection cannot reproduce that dynamic organizer rescore.
        return None
    result = "holds" if record.verdict == "unsat" else "violated"
    target = competitive.InstanceResult(
        tool="ny",
        benchmark=record.category,
        instance=str(record.instance),
        result=result,
        counterexample=record.counterexample,
        ce_required=True,
    )
    field = [target]
    if result == "holds" and truth == "violated":
        field.append(
            competitive.InstanceResult(
                tool="published_ground_truth",
                benchmark=record.category,
                instance=str(record.instance),
                result="violated",
                counterexample=competitive.CounterexampleResult.CORRECT,
                ce_required=True,
            )
        )
    return competitive.score_instance(target, field)


def _qualified_section(
    records: list[SealedRecord],
    audit: dict[str, Any],
    exact_commit: str,
    roots: list[Path],
    official: OfficialContext,
) -> dict[str, Any]:
    by_suite: dict[str, list[SealedRecord]] = defaultdict(list)
    for record in records:
        by_suite[record.category].append(record)
    suites: list[dict[str, Any]] = []
    total_score = 0.0
    for category in retro.REGULAR:
        solved = incorrect = raw = unqualified_sat = sealed_nonsolve = 0
        for record in by_suite[category]:
            points = _score_record(
                record,
                official.ground_truth.get(category, {}).get(record.instance, "unknown"),
            )
            if points is None:
                sealed_nonsolve += 1
                if record.verdict == "sat":
                    unqualified_sat += 1
            elif points == retro.POINTS_CORRECT:
                solved += 1
                raw += points
            elif points == retro.PENALTY_INCORRECT:
                incorrect += 1
                raw += points
            else:
                raise AuditError(f"unexpected scorer result {points} for {category}")
        total_rows = len(official.reference_order[category])
        unmeasured = total_rows - solved - incorrect
        normalized = competitive.normalize_benchmark(
            {"published_winner": official.winner_points[category], "ny": raw}
        )["ny"]
        total_score += normalized
        suites.append(
            {
                "suite": category,
                "official_instances": total_rows,
                "qualified_solved": solved,
                "qualified_incorrect": incorrect,
                "unmeasured": unmeasured,
                "sealed_non_solve_or_unqualified": sealed_nonsolve,
                "unqualified_sat": unqualified_sat,
                "raw_points": raw,
                "official_winner_raw_points": official.winner_points[category],
                "score": _round_score(normalized),
                "gap_to_100": _round_score(100.0 - normalized),
                "min_extra_credits_to_100": _min_extra_credits(
                    raw, official.winner_points[category]
                ),
            }
        )
    return {
        "evidence_tier": QUALIFIED_TIER,
        "is_exact_current_evidence": True,
        "exact_commit": exact_commit,
        "artifact_roots": sorted(str(path.resolve()) for path in roots),
        "sat_policy": (
            "exact_2025_zero_tol_replay_required_2026_sidecars_unqualified"
        ),
        "metric_caveat": (
            "Frozen published denominators are retained; a new strictly correct SAT "
            "witness can change incumbent penalties and the official denominator"
        ),
        "qualification_audit": audit,
        "score": _round_score(total_score),
        "gap_to_perfect": _round_score(1600.0 - total_score),
        "min_extra_credits_to_perfect": sum(
            row["min_extra_credits_to_100"] for row in suites
        ),
        "qualified_solved": sum(row["qualified_solved"] for row in suites),
        "qualified_incorrect": sum(row["qualified_incorrect"] for row in suites),
        "unmeasured": sum(row["unmeasured"] for row in suites),
        "suites": suites,
    }


def build_audit(
    *,
    official_root: Path,
    legacy_measured_root: Path,
    artifact_roots: list[Path],
    exact_commit: str,
    benchmark_root: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    if EXACT_COMMIT_RE.fullmatch(exact_commit) is None:
        raise AuditError(
            "--exact-commit must be exactly 40 lowercase hexadecimal digits"
        )
    if not artifact_roots:
        raise AuditError("at least one --artifact-root is required")
    official_evidence = None
    benchmark_evidence = None
    if benchmark_root is not None:
        # Imported lazily to avoid the regular-bank validator's dependency on
        # this module while both modules are initialized.
        import regular_bank_evidence as regular  # noqa: PLC0415

        try:
            official_evidence = regular.validate_official_results(official_root)
            benchmark_evidence = regular.validate_official_benchmark(benchmark_root)
        except regular.EvidenceError as error:
            raise AuditError(
                f"pinned exact-2025 SAT replay context is invalid: {error}"
            ) from error
        official = official_evidence.context
    else:
        official = load_official_context(official_root)
    legacy = load_legacy_results(legacy_measured_root, official)
    sealed, qualification_audit, incomplete = load_sealed_records(
        artifact_roots,
        exact_commit,
        official,
        official_evidence=official_evidence,
        benchmark_evidence=benchmark_evidence,
    )
    report = {
        "schema": SCHEMA,
        "claim_scope": (
            "local_reproducible_internal_counterfactual_"
            "not_official_or_independently_attested"
        ),
        "metric": METRIC,
        "theoretical_perfect_score": 1600.0,
        "suite_order": list(retro.REGULAR),
        "legacy_projection": _legacy_section(legacy, official),
        "qualified_current": _qualified_section(
            sealed,
            qualification_audit,
            exact_commit,
            artifact_roots,
            official,
        ),
    }
    return report, incomplete


def render_csv(report: dict[str, Any]) -> str:
    legacy = report["legacy_projection"]
    qualified = report["qualified_current"]
    legacy_by_suite = {row["suite"]: row for row in legacy["suites"]}
    qualified_by_suite = {row["suite"]: row for row in qualified["suites"]}
    output = io.StringIO(newline="")
    fields = [
        "suite",
        "official_instances",
        "legacy_evidence_tier",
        "legacy_score",
        "legacy_gap",
        "legacy_min_extra_credits",
        "qualified_evidence_tier",
        "qualified_solved",
        "qualified_incorrect",
        "qualified_unmeasured",
        "qualified_score",
        "qualified_gap",
        "qualified_min_extra_credits",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for suite in report["suite_order"]:
        left = legacy_by_suite[suite]
        right = qualified_by_suite[suite]
        writer.writerow(
            {
                "suite": suite,
                "official_instances": left["official_instances"],
                "legacy_evidence_tier": legacy["evidence_tier"],
                "legacy_score": left["score"],
                "legacy_gap": left["gap_to_100"],
                "legacy_min_extra_credits": left["min_extra_credits_to_100"],
                "qualified_evidence_tier": qualified["evidence_tier"],
                "qualified_solved": right["qualified_solved"],
                "qualified_incorrect": right["qualified_incorrect"],
                "qualified_unmeasured": right["unmeasured"],
                "qualified_score": right["score"],
                "qualified_gap": right["gap_to_100"],
                "qualified_min_extra_credits": right["min_extra_credits_to_100"],
            }
        )
    writer.writerow(
        {
            "suite": "TOTAL",
            "official_instances": qualified["qualified_solved"]
            + qualified["qualified_incorrect"]
            + qualified["unmeasured"],
            "legacy_evidence_tier": legacy["evidence_tier"],
            "legacy_score": legacy["score"],
            "legacy_gap": legacy["gap_to_perfect"],
            "legacy_min_extra_credits": legacy["min_extra_credits_to_perfect"],
            "qualified_evidence_tier": qualified["evidence_tier"],
            "qualified_solved": qualified["qualified_solved"],
            "qualified_incorrect": qualified["qualified_incorrect"],
            "qualified_unmeasured": qualified["unmeasured"],
            "qualified_score": qualified["score"],
            "qualified_gap": qualified["gap_to_perfect"],
            "qualified_min_extra_credits": qualified["min_extra_credits_to_perfect"],
        }
    )
    return output.getvalue()


def render_table(report: dict[str, Any]) -> str:
    legacy = report["legacy_projection"]
    qualified = report["qualified_current"]
    legacy_by_suite = {row["suite"]: row for row in legacy["suites"]}
    qualified_by_suite = {row["suite"]: row for row in qualified["suites"]}
    lines = [
        "MAIN16 GAP AUDIT — EVIDENCE TIERS MUST NOT BE COMBINED",
        (
            "claim scope: local counterfactual; nonofficial; "
            "not independently attested"
        ),
        f"legacy:    {legacy['evidence_tier']} (NOT QUALIFIED)",
        f"qualified: {qualified['evidence_tier']} @ {qualified['exact_commit']}",
        "",
        (
            f"{'suite':24s} {'legacy':>9s} {'gap':>9s} {'need':>5s}  "
            f"{'q_ok':>5s} {'q_bad':>5s} {'q_none':>6s} {'q_score':>9s}"
        ),
    ]
    for suite in report["suite_order"]:
        left = legacy_by_suite[suite]
        right = qualified_by_suite[suite]
        lines.append(
            f"{suite:24s} {left['score']:9.3f} {left['gap_to_100']:9.3f} "
            f"{left['min_extra_credits_to_100']:5d}  "
            f"{right['qualified_solved']:5d} {right['qualified_incorrect']:5d} "
            f"{right['unmeasured']:6d} {right['score']:9.3f}"
        )
    lines.extend(
        [
            "-" * 91,
            (
                f"legacy optimistic: {legacy['score']:.6f}/1600; "
                f"gap={legacy['gap_to_perfect']:.6f}; "
                f"minimum extra credits={legacy['min_extra_credits_to_perfect']}"
            ),
            (
                f"exact-current:     {qualified['score']:.6f}/1600; "
                f"solved={qualified['qualified_solved']}; "
                f"incorrect={qualified['qualified_incorrect']}; "
                f"unmeasured={qualified['unmeasured']}"
            ),
            f"qualification: {qualified['qualification_audit']['status']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_output(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official", type=Path, required=True, help="VNN-COMP 2025 result repository"
    )
    parser.add_argument(
        "--legacy-measured",
        type=Path,
        required=True,
        help="legacy per-suite NY CSV directory",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        required=True,
        help="direct artifact root containing runs/<run-id>/start.json; repeatable",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        help=(
            "pinned VNN-COMP 2025 benchmarks/ tree; required for exact-2025 "
            "SAT replay qualification"
        ),
    )
    parser.add_argument(
        "--exact-commit",
        required=True,
        help="40-lowercase-hex NY commit required in qualified start manifests",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="stdout format",
    )
    parser.add_argument("--json-out", type=Path, help="also write deterministic JSON")
    parser.add_argument("--csv-out", type=Path, help="also write deterministic CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, incomplete = build_audit(
            official_root=args.official,
            legacy_measured_root=args.legacy_measured,
            artifact_roots=args.artifact_root,
            exact_commit=args.exact_commit,
            benchmark_root=args.benchmark_root,
        )
        json_data = _json_bytes(report)
        csv_data = render_csv(report).encode("utf-8")
        table_data = render_table(report).encode("utf-8")
        if args.json_out is not None:
            _write_output(args.json_out, json_data)
        if args.csv_out is not None:
            _write_output(args.csv_out, csv_data)
        sys.stdout.buffer.write(
            {"table": table_data, "json": json_data, "csv": csv_data}[args.format]
        )
        return 1 if incomplete else 0
    except (AuditError, OSError, ValueError) as error:
        print(f"main16 audit failed closed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
