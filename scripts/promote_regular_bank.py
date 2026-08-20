#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Promote one sealed regular-track result into the measured score bank.

Dry-run is the default.  ``--apply`` uses an index-first transaction so a
crash cannot leave an unindexed decided row.  Repeating the exact request
resumes an index-only transaction or succeeds as an already-applied no-op.
Every existing index entry is fully revalidated before a new row is planned.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import main16_gap_audit as gap  # noqa: E402
import regular_bank_evidence as evidence  # noqa: E402
import pathlib

# Sibling import: make the script directory importable first, exactly as
# replay_vnncomp2025_counterexample.py does. Without it the module loads when
# run as a script but not when a test imports it by path.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import _portable_file_lock as _file_lock  # noqa: E402


INDEX_SCHEMA = evidence.INDEX_SCHEMA
PLAN_SCHEMA = "ny_regular_bank_promotion_plan_v3"
BATCH_PLAN_SCHEMA = "ny_regular_bank_promotion_batch_plan_v2"
PRE_PROFILE_ENTRY_SCHEMA = evidence.PRE_PROFILE_ENTRY_SCHEMA
PRE_PROFILE_DYNAMIC_ENTRY_SCHEMA = evidence.PRE_PROFILE_DYNAMIC_ENTRY_SCHEMA
ENTRY_SCHEMA = evidence.ENTRY_SCHEMA
DYNAMIC_ENTRY_SCHEMA = evidence.DYNAMIC_ENTRY_SCHEMA
LEGACY_DECIDED_ROW_MIGRATION = evidence.LEGACY_DECIDED_ROW_MIGRATION
UNRESOLVED_LITERAL_VERDICTS = evidence.UNRESOLVED_LITERAL_VERDICTS
DECIDED_VERDICTS = evidence.DECIDED_VERDICTS
PromotionError = evidence.EvidenceError


@dataclass(frozen=True)
class PromotionRequest:
    artifact_root: Path
    run_id: str
    category: str
    instance_index: int
    benchmark_root: Path
    official_results: Path
    measured_dir: Path
    exact_commit: str
    evidence_index: Path | None = None
    migrate_legacy_decided_row: bool = False


@dataclass(frozen=True)
class PromotionPlan:
    request: PromotionRequest
    measured_path: Path
    measured_before: bytes
    measured_after: bytes
    index_path: Path
    index_before: bytes | None
    index_after: bytes
    summary: dict[str, Any]


@dataclass(frozen=True)
class BatchMeasuredUpdate:
    path: Path
    before: bytes
    after: bytes


@dataclass(frozen=True)
class BatchPromotionPlan:
    requests: tuple[PromotionRequest, ...]
    measured_updates: tuple[BatchMeasuredUpdate, ...]
    index_path: Path
    index_before: bytes | None
    index_after: bytes
    summaries: tuple[dict[str, Any], ...]


def _canonical_destination(path: Path, label: str) -> Path:
    if path.name in {"", ".", ".."}:
        raise PromotionError(f"{label} has no file name: {path}")
    parent = evidence.resolved_directory(path.absolute().parent, f"{label} parent")
    candidate = parent / path.name
    if candidate.is_symlink():
        raise PromotionError(f"{label} must not be a symlink: {candidate}")
    return candidate


def _render_replacement_line(fields: list[str], original: bytes) -> bytes:
    if original.endswith(b"\r\n"):
        ending = "\r\n"
    elif original.endswith(b"\n"):
        ending = "\n"
    elif original.endswith(b"\r"):
        ending = "\r"
    else:
        ending = ""
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator=ending).writerow(fields)
    return buffer.getvalue().encode("utf-8")


def _replace_bank_row(
    *,
    measured_path: Path,
    measured_data: bytes,
    target: evidence.ValidatedPromotionEvidence,
    required_before: list[str] | None = None,
    required_after: list[str] | None = None,
    migrate_legacy_decided_row: bool = False,
) -> tuple[bytes, list[str], list[str], str]:
    located = evidence.locate_bank_row(
        measured_path=measured_path,
        data=measured_data,
        category=target.category,
        occurrence=target.occurrence,
        official_reference=target.official.context.reference_order[target.category],
    )
    before = located.fields
    if required_before is not None and before != required_before:
        raise PromotionError("measured row differs from indexed dangling row")
    prior_verdict = before[4]
    old_verdict = prior_verdict.strip().lower()
    if migrate_legacy_decided_row:
        if prior_verdict not in DECIDED_VERDICTS:
            raise PromotionError(
                "legacy decided-row migration requires an existing canonical "
                "sat/unsat verdict; aggregate correct/incorrect markers and "
                "unresolved rows are ineligible"
            )
        if prior_verdict != target.verdict:
            raise PromotionError(
                "legacy decided-row verdict conflicts with reopened sealed "
                f"evidence ({prior_verdict!r} != {target.verdict!r})"
            )
    elif old_verdict not in UNRESOLVED_LITERAL_VERDICTS:
        raise PromotionError(
            f"refusing to overwrite decided/correct/incorrect row verdict {before[4]!r}"
        )
    after = before[:4] + [
        target.verdict,
        target.runtime_seconds,
        target.run_id,
    ]
    if required_after is not None and after != required_after:
        raise PromotionError("resumed bank row differs from indexed after row")
    physical_lines = measured_data.splitlines(keepends=True)
    updated = list(physical_lines)
    updated[located.line_index] = _render_replacement_line(
        after, physical_lines[located.line_index]
    )
    return b"".join(updated), before, after, old_verdict


def _migration_requested(request: PromotionRequest) -> bool:
    if type(request.migrate_legacy_decided_row) is not bool:
        raise PromotionError("migrate_legacy_decided_row must be an explicit boolean")
    return request.migrate_legacy_decided_row


def _entry_uses_legacy_decided_migration(
    entry: evidence.ValidatedIndexEntry,
) -> bool:
    measured = entry.entry.get("measured_csv")
    return (
        isinstance(measured, dict)
        and measured.get("migration") == LEGACY_DECIDED_ROW_MIGRATION
    )


def _require_matching_index_mode(
    *,
    request: PromotionRequest,
    existing: evidence.ValidatedIndexEntry,
    row_key: str,
) -> None:
    if _migration_requested(request) != _entry_uses_legacy_decided_migration(existing):
        raise PromotionError(
            f"evidence index row {row_key} was created with a different "
            "legacy decided-row migration mode"
        )


def _same_promotion_evidence(
    left: evidence.ValidatedPromotionEvidence,
    right: evidence.ValidatedPromotionEvidence,
) -> bool:
    return evidence.promotion_evidence_binding(
        left, allow_pre_profile_start=True
    ) == evidence.promotion_evidence_binding(right, allow_pre_profile_start=True)


def _summary(
    *,
    request: PromotionRequest,
    target: evidence.ValidatedPromotionEvidence,
    measured_path: Path,
    measured_before: bytes,
    measured_after: bytes,
    index_path: Path,
    row_key: str,
    action: str,
    old_verdict: str,
) -> dict[str, Any]:
    summary = {
        "schema": PLAN_SCHEMA,
        "claim_scope": evidence.CLAIM_SCOPE,
        "action": action,
        "artifact_root": str(target.artifact_root),
        "category": request.category,
        "completion_sha256": target.completion_sha256,
        "containment_profile": target.containment_profile,
        "evidence_index": str(index_path),
        "exact_commit": request.exact_commit,
        "instance_index": request.instance_index,
        "measured_csv": str(measured_path),
        "measured_sha256_after": evidence.sha256(measured_after),
        "measured_sha256_before": evidence.sha256(measured_before),
        "migrate_legacy_decided_row": request.migrate_legacy_decided_row,
        "old_verdict": old_verdict,
        "official_results_commit": evidence.OFFICIAL_RESULTS_COMMIT,
        "policy": target.policy,
        "published_truth": target.published_truth,
        "row_key": row_key,
        "run_id": request.run_id,
        "runtime_seconds": target.runtime_seconds,
        "verdict": target.verdict,
    }
    if target.organizer_rescore is not None:
        summary.update(
            {
                "effective_truth": "violated",
                "organizer_rescore_official_denominator": target.organizer_rescore[
                    "denominator"
                ],
                "organizer_rescore_sha256": (
                    evidence.provenance._identity_sha256(target.organizer_rescore)
                ),
            }
        )
    return summary


def build_plan(request: PromotionRequest) -> PromotionPlan:
    migrate_legacy_decided_row = _migration_requested(request)
    measured_dir = evidence.resolved_directory(
        request.measured_dir, "measured bank directory"
    )
    pinned_official = evidence.validate_official_results(request.official_results)
    pinned_benchmark = evidence.validate_official_benchmark(request.benchmark_root)
    replay_session: dict[str, Any] = {}
    target = evidence.validate_promotion_evidence(
        artifact_root=request.artifact_root,
        run_id=request.run_id,
        category=request.category,
        instance_index=request.instance_index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        exact_commit=request.exact_commit,
        pinned_official=pinned_official,
        pinned_benchmark=pinned_benchmark,
        replay_session=replay_session,
    )
    measured_path = evidence.resolved_regular_file(
        measured_dir / f"{request.category}.csv", "measured category CSV"
    )
    index_arg = (
        request.evidence_index
        if request.evidence_index is not None
        else measured_dir / "regular_evidence_index.json"
    )
    index_path = _canonical_destination(index_arg, "evidence index")
    if index_path == measured_path:
        raise PromotionError("evidence index and measured CSV must be different files")
    if index_path.exists() and os.path.samefile(index_path, measured_path):
        raise PromotionError(
            "evidence index and measured CSV must not be hard links to one file"
        )

    validated_index = evidence.validate_regular_evidence_index(
        evidence_index=index_path,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        measured_dir=measured_dir,
        allow_missing=True,
        pinned_official=pinned_official,
        pinned_benchmark=pinned_benchmark,
        replay_session=replay_session,
    )
    row_key = evidence.canonical_row_key(request.category, target.occurrence)
    by_key = {entry.row_key: entry for entry in validated_index.entries}
    existing = by_key.get(row_key)
    unrelated_dangling = [
        entry.row_key
        for entry in validated_index.dangling_entries
        if entry.row_key != row_key
    ]
    if unrelated_dangling:
        raise PromotionError(
            "evidence index contains an unrelated dangling transaction; "
            "resume it first: " + ", ".join(unrelated_dangling)
        )

    measured_before = evidence.stable_bytes(measured_path, "measured category CSV")
    index_before = validated_index.data
    index_after = index_before
    if existing is not None:
        if not _same_promotion_evidence(existing.evidence, target):
            raise PromotionError(
                f"evidence index row {row_key} is bound to different evidence"
            )
        _require_matching_index_mode(
            request=request,
            existing=existing,
            row_key=row_key,
        )
        if existing.bank_state == "applied":
            measured_after = measured_before
            if existing.legacy_entry:
                migrated = evidence.migrate_legacy_index_entry(
                    target,
                    legacy_entry=existing.entry,
                    measured_path=measured_path,
                    row_after=existing.bank_row,
                )
                index_value = copy.deepcopy(validated_index.value)
                entries = index_value.get("entries")
                assert isinstance(entries, dict)
                entries[row_key] = migrated
                index_after = gap._json_bytes(index_value)
                action = "migrate_applied_legacy_index"
                old_verdict = "legacy_unavailable"
            else:
                action = "already_applied"
                measured_binding = existing.entry["measured_csv"]
                old_verdict = (
                    measured_binding["row_before"][4]
                    if "row_before" in measured_binding
                    else "legacy_unavailable"
                )
        elif existing.bank_state == "dangling" and not existing.legacy_entry:
            measured_binding = existing.entry["measured_csv"]
            measured_after, _, _, old_verdict = _replace_bank_row(
                measured_path=measured_path,
                measured_data=measured_before,
                target=target,
                required_before=measured_binding["row_before"],
                required_after=measured_binding["row_after"],
                migrate_legacy_decided_row=migrate_legacy_decided_row,
            )
            action = "resume_dangling_index"
        else:
            raise PromotionError(
                f"evidence index row {row_key} has unsupported bank state"
            )
        assert index_after is not None
    else:
        measured_after, row_before, row_after, old_verdict = _replace_bank_row(
            measured_path=measured_path,
            measured_data=measured_before,
            target=target,
            migrate_legacy_decided_row=migrate_legacy_decided_row,
        )
        entry = evidence.make_index_entry(
            target,
            measured_path=measured_path,
            measured_before=measured_before,
            measured_after=measured_after,
            row_before=row_before,
            row_after=row_after,
            migrate_legacy_decided_row=migrate_legacy_decided_row,
        )
        index_value = copy.deepcopy(validated_index.value)
        entries = index_value.get("entries")
        assert isinstance(entries, dict)
        if row_key in entries:
            raise PromotionError(f"evidence index row appeared twice: {row_key}")
        entries[row_key] = entry
        index_after = gap._json_bytes(index_value)
        action = (
            "migrate_legacy_decided_regular_bank_row"
            if migrate_legacy_decided_row
            else "replace_unresolved_regular_bank_row"
        )

    assert index_after is not None
    # Close the validation/planning race for every source whose bytes are
    # captured in the plan.  apply_plan performs the corresponding locked CAS.
    if evidence.stable_bytes(measured_path, "measured category CSV") != measured_before:
        raise PromotionError("measured CSV changed while the promotion was planned")
    current_index = (
        evidence.stable_bytes(index_path, "evidence index")
        if index_path.exists()
        else None
    )
    if current_index != index_before:
        raise PromotionError("evidence index changed while the promotion was planned")
    final_index = evidence.validate_regular_evidence_index(
        evidence_index=index_path,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        measured_dir=measured_dir,
        allow_missing=True,
        pinned_official=pinned_official,
        pinned_benchmark=pinned_benchmark,
        replay_session=replay_session,
    )
    if final_index.data != index_before:
        raise PromotionError("evidence index changed during final revalidation")
    reopened_target = evidence.validate_promotion_evidence(
        artifact_root=request.artifact_root,
        run_id=request.run_id,
        category=request.category,
        instance_index=request.instance_index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        exact_commit=request.exact_commit,
        pinned_official=pinned_official,
        pinned_benchmark=pinned_benchmark,
        replay_session=replay_session,
    )
    if not _same_promotion_evidence(target, reopened_target):
        raise PromotionError("candidate evidence changed during final revalidation")
    evidence.revalidate_official_benchmark(pinned_benchmark)
    evidence.revalidate_official_results(pinned_official)
    evidence.revalidate_replay_session(replay_session)

    return PromotionPlan(
        request=request,
        measured_path=measured_path,
        measured_before=measured_before,
        measured_after=measured_after,
        index_path=index_path,
        index_before=index_before,
        index_after=index_after,
        summary=_summary(
            request=request,
            target=target,
            measured_path=measured_path,
            measured_before=measured_before,
            measured_after=measured_after,
            index_path=index_path,
            row_key=row_key,
            action=action,
            old_verdict=old_verdict,
        ),
    )


def build_batch_plan(requests: list[PromotionRequest]) -> BatchPromotionPlan:
    """Plan many promotions with one full index validation and one commit."""

    if not requests:
        raise PromotionError("batch promotion requires at least one request")
    for request in requests:
        _migration_requested(request)
    requested_occurrences = [
        (request.category, request.instance_index) for request in requests
    ]
    if len(requested_occurrences) != len(set(requested_occurrences)):
        raise PromotionError("batch contains duplicate requested occurrences")
    canonical: list[tuple[PromotionRequest, Path, Path, Path, Path]] = []
    for request in requests:
        measured_dir = evidence.resolved_directory(
            request.measured_dir, "measured bank directory"
        )
        benchmark_root = evidence.resolved_directory(
            request.benchmark_root, "benchmark root"
        )
        official_results = evidence.resolved_directory(
            request.official_results, "official result root"
        )
        index_arg = (
            request.evidence_index
            if request.evidence_index is not None
            else measured_dir / "regular_evidence_index.json"
        )
        index_path = _canonical_destination(index_arg, "evidence index")
        canonical.append(
            (
                request,
                measured_dir,
                benchmark_root,
                official_results,
                index_path,
            )
        )
    common = canonical[0][1:]
    if any(values[1:] != common for values in canonical[1:]):
        raise PromotionError(
            "every batch request must share the same measured directory, "
            "benchmark root, official results root, and evidence index"
        )
    measured_dir, benchmark_root, official_results, index_path = common
    pinned_official = evidence.validate_official_results(official_results)
    pinned_benchmark = evidence.validate_official_benchmark(benchmark_root)
    replay_session: dict[str, Any] = {}
    validated_index = evidence.validate_regular_evidence_index(
        evidence_index=index_path,
        benchmark_root=benchmark_root,
        official_results=official_results,
        measured_dir=measured_dir,
        allow_missing=True,
        pinned_official=pinned_official,
        pinned_benchmark=pinned_benchmark,
        replay_session=replay_session,
    )

    targets: list[tuple[PromotionRequest, evidence.ValidatedPromotionEvidence]] = []
    for request, *_ in canonical:
        target = evidence.validate_promotion_evidence(
            artifact_root=request.artifact_root,
            run_id=request.run_id,
            category=request.category,
            instance_index=request.instance_index,
            benchmark_root=benchmark_root,
            official_results=official_results,
            exact_commit=request.exact_commit,
            pinned_official=pinned_official,
            pinned_benchmark=pinned_benchmark,
            replay_session=replay_session,
        )
        targets.append((request, target))
    targets.sort(
        key=lambda item: (
            item[1].category,
            item[1].instance_index,
            item[1].run_id,
            str(item[1].artifact_root),
        )
    )
    row_keys = [
        evidence.canonical_row_key(target.category, target.occurrence)
        for _, target in targets
    ]
    if len(row_keys) != len(set(row_keys)):
        raise PromotionError("batch contains duplicate official row identities")
    run_keys = [(str(target.artifact_root), target.run_id) for _, target in targets]
    if len(run_keys) != len(set(run_keys)):
        raise PromotionError("batch contains duplicate sealed run identities")

    existing_by_key = {entry.row_key: entry for entry in validated_index.entries}
    requested_keys = set(row_keys)
    unrelated_dangling = sorted(
        entry.row_key
        for entry in validated_index.dangling_entries
        if entry.row_key not in requested_keys
    )
    if unrelated_dangling:
        raise PromotionError(
            "evidence index contains dangling transactions omitted from the "
            "batch: " + ", ".join(unrelated_dangling)
        )

    index_value = copy.deepcopy(validated_index.value)
    raw_entries = index_value.get("entries")
    assert isinstance(raw_entries, dict)
    measured_before: dict[Path, bytes] = {}
    measured_after: dict[Path, bytes] = {}
    pending_new: list[
        tuple[
            str,
            evidence.ValidatedPromotionEvidence,
            Path,
            list[str],
            list[str],
            bool,
        ]
    ] = []
    summary_inputs: list[
        tuple[
            PromotionRequest,
            evidence.ValidatedPromotionEvidence,
            Path,
            str,
            str,
        ]
    ] = []

    if len(targets) != len(row_keys):
        raise PromotionError("batch target and row-key counts do not match")
    for (request, target), row_key in zip(targets, row_keys):
        measured_path = evidence.resolved_regular_file(
            measured_dir / f"{target.category}.csv",
            "measured category CSV",
        )
        if measured_path == index_path or (
            index_path.exists() and os.path.samefile(index_path, measured_path)
        ):
            raise PromotionError(
                "evidence index and measured CSV must be different files"
            )
        if measured_path not in measured_before:
            original = evidence.stable_bytes(measured_path, "measured category CSV")
            measured_before[measured_path] = original
            measured_after[measured_path] = original
        current = measured_after[measured_path]
        existing = existing_by_key.get(row_key)
        if existing is not None:
            if not _same_promotion_evidence(existing.evidence, target):
                raise PromotionError(
                    f"evidence index row {row_key} is bound to different evidence"
                )
            _require_matching_index_mode(
                request=request,
                existing=existing,
                row_key=row_key,
            )
            if existing.bank_state == "applied":
                if existing.legacy_entry:
                    raw_entries[row_key] = evidence.migrate_legacy_index_entry(
                        target,
                        legacy_entry=existing.entry,
                        measured_path=measured_path,
                        row_after=existing.bank_row,
                    )
                    action = "migrate_applied_legacy_index"
                    old_verdict = "legacy_unavailable"
                else:
                    action = "already_applied"
                    binding = existing.entry["measured_csv"]
                    old_verdict = (
                        binding["row_before"][4]
                        if "row_before" in binding
                        else "legacy_unavailable"
                    )
            elif existing.bank_state == "dangling" and not existing.legacy_entry:
                binding = existing.entry["measured_csv"]
                current, _, _, old_verdict = _replace_bank_row(
                    measured_path=measured_path,
                    measured_data=current,
                    target=target,
                    required_before=binding["row_before"],
                    required_after=binding["row_after"],
                    migrate_legacy_decided_row=(request.migrate_legacy_decided_row),
                )
                measured_after[measured_path] = current
                action = "resume_dangling_index"
            else:
                raise PromotionError(
                    f"evidence index row {row_key} has unsupported bank state"
                )
        else:
            current, row_before, row_after, old_verdict = _replace_bank_row(
                measured_path=measured_path,
                measured_data=current,
                target=target,
                migrate_legacy_decided_row=(request.migrate_legacy_decided_row),
            )
            measured_after[measured_path] = current
            pending_new.append(
                (
                    row_key,
                    target,
                    measured_path,
                    row_before,
                    row_after,
                    request.migrate_legacy_decided_row,
                )
            )
            action = (
                "migrate_legacy_decided_regular_bank_row"
                if request.migrate_legacy_decided_row
                else "replace_unresolved_regular_bank_row"
            )
        summary_inputs.append((request, target, measured_path, action, old_verdict))

    for (
        row_key,
        target,
        measured_path,
        row_before,
        row_after,
        migrate_legacy_decided_row,
    ) in pending_new:
        raw_entries[row_key] = evidence.make_index_entry(
            target,
            measured_path=measured_path,
            measured_before=measured_before[measured_path],
            measured_after=measured_after[measured_path],
            row_before=row_before,
            row_after=row_after,
            migrate_legacy_decided_row=migrate_legacy_decided_row,
        )
    index_after = gap._json_bytes(index_value)
    index_before = validated_index.data

    # Final global reopen closes races across earlier candidates and the
    # already-indexed history without reintroducing per-row O(N²) validation.
    final_index = evidence.validate_regular_evidence_index(
        evidence_index=index_path,
        benchmark_root=benchmark_root,
        official_results=official_results,
        measured_dir=measured_dir,
        allow_missing=True,
        pinned_official=pinned_official,
        pinned_benchmark=pinned_benchmark,
        replay_session=replay_session,
    )
    if final_index.data != index_before:
        raise PromotionError("evidence index changed while the batch was planned")
    for request, original_target in targets:
        reopened = evidence.validate_promotion_evidence(
            artifact_root=request.artifact_root,
            run_id=request.run_id,
            category=request.category,
            instance_index=request.instance_index,
            benchmark_root=benchmark_root,
            official_results=official_results,
            exact_commit=request.exact_commit,
            pinned_official=pinned_official,
            pinned_benchmark=pinned_benchmark,
            replay_session=replay_session,
        )
        if not _same_promotion_evidence(original_target, reopened):
            raise PromotionError(
                "candidate evidence changed while the batch was planned"
            )
    for path, before in measured_before.items():
        if evidence.stable_bytes(path, "measured category CSV") != before:
            raise PromotionError("measured CSV changed while the batch was planned")
    evidence.revalidate_official_benchmark(pinned_benchmark)
    evidence.revalidate_official_results(pinned_official)
    evidence.revalidate_replay_session(replay_session)

    summaries = tuple(
        _summary(
            request=request,
            target=target,
            measured_path=measured_path,
            measured_before=measured_before[measured_path],
            measured_after=measured_after[measured_path],
            index_path=index_path,
            row_key=evidence.canonical_row_key(target.category, target.occurrence),
            action=action,
            old_verdict=old_verdict,
        )
        for request, target, measured_path, action, old_verdict in summary_inputs
    )
    updates = tuple(
        BatchMeasuredUpdate(path, measured_before[path], measured_after[path])
        for path in sorted(measured_before, key=str)
    )
    return BatchPromotionPlan(
        requests=tuple(request for request, _ in targets),
        measured_updates=updates,
        index_path=index_path,
        index_before=index_before,
        index_after=index_after,
        summaries=summaries,
    )


@contextmanager
def _directory_locks(paths: list[Path]) -> Iterator[None]:
    with ExitStack() as stack:
        for directory in sorted({path.parent.resolve() for path in paths}, key=str):
            # Sorted so every caller takes the locks in the same order: this
            # loop holds several at once, and an inconsistent order between
            # processes is a deadlock. `directory_lock` keeps the POSIX
            # descriptor lock verbatim and only differs on Windows, which
            # cannot lock a directory handle at all.
            stack.enter_context(_file_lock.directory_lock(directory))
        yield


def _stage_file(path: Path, data: bytes, mode: int) -> Path:
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.promote-",
        dir=path.parent,
    )
    temp = Path(raw_temp)
    try:
        os.fchmod(descriptor, stat.S_IMODE(mode))
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temp.unlink(missing_ok=True)
        raise
    return temp


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _optional_current_bytes(path: Path, label: str) -> bytes | None:
    if path.is_symlink():
        raise PromotionError(f"{label} must not be a symlink: {path}")
    if not path.exists():
        return None
    return evidence.stable_bytes(path, label)


def _revalidate_apply_requests(
    requests: tuple[PromotionRequest, ...],
    summaries: tuple[dict[str, Any], ...],
) -> None:
    """Reopen all external evidence immediately before bank/index mutation."""

    if not requests:
        return
    first = requests[0]
    measured_dir = evidence.resolved_directory(
        first.measured_dir, "measured bank directory"
    )
    benchmark_root = evidence.resolved_directory(first.benchmark_root, "benchmark root")
    official_results = evidence.resolved_directory(
        first.official_results, "official result root"
    )
    index_arg = (
        first.evidence_index
        if first.evidence_index is not None
        else measured_dir / "regular_evidence_index.json"
    )
    index_path = _canonical_destination(index_arg, "evidence index")
    for request in requests[1:]:
        candidate_index = _canonical_destination(
            (
                request.evidence_index
                if request.evidence_index is not None
                else evidence.resolved_directory(
                    request.measured_dir, "measured bank directory"
                )
                / "regular_evidence_index.json"
            ),
            "evidence index",
        )
        if (
            evidence.resolved_directory(request.measured_dir, "measured bank directory")
            != measured_dir
            or evidence.resolved_directory(request.benchmark_root, "benchmark root")
            != benchmark_root
            or evidence.resolved_directory(
                request.official_results, "official result root"
            )
            != official_results
            or candidate_index != index_path
        ):
            raise PromotionError("apply-time batch request context changed")

    pinned_official = evidence.validate_official_results(official_results)
    pinned_benchmark = evidence.validate_official_benchmark(benchmark_root)
    replay_session: dict[str, Any] = {}
    evidence.validate_regular_evidence_index(
        evidence_index=index_path,
        benchmark_root=benchmark_root,
        official_results=official_results,
        measured_dir=measured_dir,
        allow_missing=True,
        pinned_official=pinned_official,
        pinned_benchmark=pinned_benchmark,
        replay_session=replay_session,
    )
    summary_by_identity = {
        (
            str(summary["artifact_root"]),
            str(summary["run_id"]),
            str(summary["category"]),
            int(summary["instance_index"]),
        ): summary
        for summary in summaries
    }
    immutable_keys = {
        "artifact_root",
        "category",
        "completion_sha256",
        "exact_commit",
        "instance_index",
        "migrate_legacy_decided_row",
        "official_results_commit",
        "policy",
        "published_truth",
        "run_id",
        "runtime_seconds",
        "verdict",
    }
    for request in requests:
        target = evidence.validate_promotion_evidence(
            artifact_root=request.artifact_root,
            run_id=request.run_id,
            category=request.category,
            instance_index=request.instance_index,
            benchmark_root=benchmark_root,
            official_results=official_results,
            exact_commit=request.exact_commit,
            pinned_official=pinned_official,
            pinned_benchmark=pinned_benchmark,
            replay_session=replay_session,
        )
        identity = (
            str(target.artifact_root),
            target.run_id,
            target.category,
            target.instance_index,
        )
        expected = summary_by_identity.get(identity)
        if expected is None:
            raise PromotionError(
                "apply-time evidence is absent from the validated plan"
            )
        observed = {
            "artifact_root": str(target.artifact_root),
            "category": target.category,
            "completion_sha256": target.completion_sha256,
            "exact_commit": target.exact_commit,
            "instance_index": target.instance_index,
            "migrate_legacy_decided_row": (request.migrate_legacy_decided_row),
            "official_results_commit": evidence.OFFICIAL_RESULTS_COMMIT,
            "policy": target.policy,
            "published_truth": target.published_truth,
            "run_id": target.run_id,
            "runtime_seconds": target.runtime_seconds,
            "verdict": target.verdict,
        }
        if target.organizer_rescore is not None:
            observed.update(
                {
                    "effective_truth": "violated",
                    "organizer_rescore_official_denominator": target.organizer_rescore[
                        "denominator"
                    ],
                    "organizer_rescore_sha256": (
                        evidence.provenance._identity_sha256(target.organizer_rescore)
                    ),
                }
            )
        dynamic_keys = {
            "effective_truth",
            "organizer_rescore_official_denominator",
            "organizer_rescore_sha256",
        }
        expected_dynamic = dynamic_keys & set(expected)
        observed_dynamic = dynamic_keys & set(observed)
        if expected_dynamic != observed_dynamic:
            raise PromotionError("external rescore evidence changed after planning")
        compared_keys = immutable_keys | expected_dynamic
        if {key: observed[key] for key in compared_keys} != {
            key: expected[key] for key in compared_keys
        }:
            raise PromotionError("external evidence changed after planning")
    evidence.revalidate_official_benchmark(pinned_benchmark)
    evidence.revalidate_official_results(pinned_official)
    evidence.revalidate_replay_session(replay_session)


def _restore_optional(
    *,
    path: Path,
    prior: bytes | None,
    mode: int,
) -> None:
    if prior is None:
        path.unlink(missing_ok=True)
    else:
        rollback = _stage_file(path, prior, mode)
        os.replace(rollback, path)
    _fsync_directory(path.parent)


def apply_plan(plan: PromotionPlan) -> None:
    """Apply or resume the exact plan with compare-and-swap semantics."""

    with _directory_locks([plan.measured_path, plan.index_path]):
        measured_now = evidence.stable_bytes(
            plan.measured_path, "measured category CSV"
        )
        index_now = _optional_current_bytes(plan.index_path, "evidence index")
        if measured_now not in {plan.measured_before, plan.measured_after}:
            raise PromotionError(
                "measured CSV differs from both validated transaction states"
            )
        if index_now not in {plan.index_before, plan.index_after}:
            raise PromotionError(
                "evidence index differs from both validated transaction states"
            )
        index_needs_change = index_now != plan.index_after
        measured_needs_change = measured_now != plan.measured_after
        if (
            measured_now == plan.measured_after
            and plan.measured_before != plan.measured_after
            and index_now == plan.index_before
            and plan.index_before != plan.index_after
        ):
            raise PromotionError(
                "measured row is applied without its exact evidence-index state"
            )
        if not index_needs_change and not measured_needs_change:
            return

        _revalidate_apply_requests((plan.request,), (plan.summary,))
        measured_mode = plan.measured_path.stat().st_mode
        index_mode = (
            plan.index_path.stat().st_mode
            if plan.index_path.exists()
            else (stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        )
        index_temp: Path | None = None
        measured_temp: Path | None = None
        index_changed = False
        measured_changed = False
        try:
            if index_needs_change:
                index_temp = _stage_file(plan.index_path, plan.index_after, index_mode)
            if measured_needs_change:
                measured_temp = _stage_file(
                    plan.measured_path, plan.measured_after, measured_mode
                )
            if index_temp is not None:
                os.replace(index_temp, plan.index_path)
                index_changed = True
                _fsync_directory(plan.index_path.parent)
            if measured_temp is not None:
                os.replace(measured_temp, plan.measured_path)
                measured_changed = True
                _fsync_directory(plan.measured_path.parent)
        except BaseException:
            if index_temp is not None:
                index_temp.unlink(missing_ok=True)
            if measured_temp is not None:
                measured_temp.unlink(missing_ok=True)
            if measured_changed:
                _restore_optional(
                    path=plan.measured_path,
                    prior=measured_now,
                    mode=measured_mode,
                )
            if index_changed:
                _restore_optional(
                    path=plan.index_path,
                    prior=index_now,
                    mode=index_mode,
                )
            raise


def apply_batch_plan(plan: BatchPromotionPlan) -> None:
    """Atomically commit an index-first multi-CSV promotion transaction."""

    paths = [plan.index_path, *(update.path for update in plan.measured_updates)]
    with _directory_locks(paths):
        index_now = _optional_current_bytes(plan.index_path, "evidence index")
        measured_now = {
            update.path: evidence.stable_bytes(update.path, "measured category CSV")
            for update in plan.measured_updates
        }
        if index_now not in {plan.index_before, plan.index_after}:
            raise PromotionError(
                "evidence index differs from both validated batch states"
            )
        for update in plan.measured_updates:
            if measured_now[update.path] not in {update.before, update.after}:
                raise PromotionError(
                    f"measured CSV differs from both batch states: {update.path}"
                )
            if (
                measured_now[update.path] == update.after
                and update.before != update.after
                and index_now == plan.index_before
                and plan.index_before != plan.index_after
            ):
                raise PromotionError(
                    "measured batch row is applied without its exact "
                    "evidence-index state"
                )
        index_needs_change = index_now != plan.index_after
        changed_updates = [
            update
            for update in plan.measured_updates
            if measured_now[update.path] != update.after
        ]
        if not index_needs_change and not changed_updates:
            return

        _revalidate_apply_requests(plan.requests, plan.summaries)
        index_mode = (
            plan.index_path.stat().st_mode
            if plan.index_path.exists()
            else (stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        )
        measured_modes = {
            update.path: update.path.stat().st_mode for update in changed_updates
        }
        index_temp: Path | None = None
        measured_temps: dict[Path, Path] = {}
        index_changed = False
        measured_changed: list[Path] = []
        try:
            if index_needs_change:
                index_temp = _stage_file(plan.index_path, plan.index_after, index_mode)
            for update in changed_updates:
                measured_temps[update.path] = _stage_file(
                    update.path,
                    update.after,
                    measured_modes[update.path],
                )
            if index_temp is not None:
                os.replace(index_temp, plan.index_path)
                index_changed = True
                _fsync_directory(plan.index_path.parent)
            for update in changed_updates:
                temporary = measured_temps[update.path]
                os.replace(temporary, update.path)
                measured_changed.append(update.path)
                _fsync_directory(update.path.parent)
        except BaseException:
            if index_temp is not None:
                index_temp.unlink(missing_ok=True)
            for temporary in measured_temps.values():
                temporary.unlink(missing_ok=True)
            for path in reversed(measured_changed):
                _restore_optional(
                    path=path,
                    prior=measured_now[path],
                    mode=measured_modes[path],
                )
            if index_changed:
                _restore_optional(
                    path=plan.index_path,
                    prior=index_now,
                    mode=index_mode,
                )
            raise


def promote_batch(
    requests: list[PromotionRequest],
    *,
    apply: bool = False,
) -> dict[str, Any]:
    plan = build_batch_plan(requests)
    if apply:
        apply_batch_plan(plan)
    changed = (
        any(update.before != update.after for update in plan.measured_updates)
        or plan.index_before != plan.index_after
    )
    return {
        "schema": BATCH_PLAN_SCHEMA,
        "claim_scope": evidence.CLAIM_SCOPE,
        "applied": apply,
        "changed": apply and changed,
        "evidence_index": str(plan.index_path),
        "request_count": len(plan.requests),
        "measured_file_count": len(plan.measured_updates),
        "rows": list(plan.summaries),
    }


def promote(request: PromotionRequest, *, apply: bool = False) -> dict[str, Any]:
    plan = build_plan(request)
    if apply:
        apply_plan(plan)
    return {
        **plan.summary,
        "applied": apply,
        "changed": apply and plan.summary["action"] != "already_applied",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--instance-index", type=int, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--official-results", type=Path, required=True)
    parser.add_argument("--measured-dir", type=Path, required=True)
    parser.add_argument("--exact-commit", required=True)
    parser.add_argument(
        "--evidence-index",
        type=Path,
        help=(
            "index destination; defaults to <measured-dir>/regular_evidence_index.json"
        ),
    )
    parser.add_argument(
        "--migrate-legacy-decided-row",
        action="store_true",
        help=(
            "explicitly replace an unindexed legacy sat/unsat row only when "
            "its verdict exactly matches the sealed evidence"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the index-first atomic transaction (default: dry run)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = PromotionRequest(
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        category=args.category,
        instance_index=args.instance_index,
        benchmark_root=args.benchmark_root,
        official_results=args.official_results,
        measured_dir=args.measured_dir,
        exact_commit=args.exact_commit,
        evidence_index=args.evidence_index,
        migrate_legacy_decided_row=args.migrate_legacy_decided_row,
    )
    try:
        summary = promote(request, apply=args.apply)
    except (PromotionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
