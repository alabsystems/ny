#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckSpec:
    name: str
    case: str
    phase: str
    expected_status: str
    expected_parameter_count: int | None
    expected_estimated_cpu_peak_bytes: int | None
    expected_cpu_dense_budget_bytes: int | None
    baseline_seconds: float | None
    max_regression_ratio: float | None
    max_seconds: float | None
    expected_detail_substring: str | None
    source_artifact: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CheckSpec":
        return cls(
            name=str(raw["name"]),
            case=str(raw["case"]),
            phase=str(raw["phase"]),
            expected_status=str(raw["expected_status"]),
            expected_parameter_count=_optional_int(raw.get("expected_parameter_count")),
            expected_estimated_cpu_peak_bytes=_optional_int(
                raw.get("expected_estimated_cpu_peak_bytes")
            ),
            expected_cpu_dense_budget_bytes=_optional_int(
                raw.get("expected_cpu_dense_budget_bytes")
            ),
            baseline_seconds=_optional_float(raw.get("baseline_seconds")),
            max_regression_ratio=_optional_float(raw.get("max_regression_ratio")),
            max_seconds=_optional_float(raw.get("max_seconds")),
            expected_detail_substring=_optional_str(raw.get("expected_detail_substring")),
            source_artifact=_optional_str(raw.get("source_artifact")),
        )

    def allowed_seconds(self) -> float | None:
        limits: list[float] = []
        if self.baseline_seconds is not None and self.max_regression_ratio is not None:
            limits.append(self.baseline_seconds * self.max_regression_ratio)
        if self.max_seconds is not None:
            limits.append(self.max_seconds)
        if not limits:
            return None
        return min(limits)


@dataclass(frozen=True)
class ObservedRow:
    case: str
    phase: str
    seconds: float | None
    parameter_count: int
    estimated_cpu_peak_bytes: int
    cpu_dense_budget_bytes: int
    status: str
    detail: str

    @classmethod
    def from_csv_row(cls, raw: dict[str, str]) -> "ObservedRow":
        return cls(
            case=raw["case"],
            phase=raw["phase"],
            seconds=_parse_optional_seconds(raw.get("seconds", "")),
            parameter_count=_parse_required_int(raw.get("parameter_count", ""), "parameter_count"),
            estimated_cpu_peak_bytes=_parse_required_int(
                raw.get("estimated_cpu_peak_bytes", ""),
                "estimated_cpu_peak_bytes",
            ),
            cpu_dense_budget_bytes=_parse_required_int(
                raw.get("cpu_dense_budget_bytes", ""),
                "cpu_dense_budget_bytes",
            ),
            status=raw.get("status", "").strip(),
            detail=raw.get("detail", "").strip(),
        )


@dataclass(frozen=True)
class CandidateRow:
    path: Path
    observed: ObservedRow


@dataclass(frozen=True)
class SelectionResult:
    candidate: CandidateRow | None
    reasons: tuple[str, ...]
    selection_mode: str | None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _parse_optional_seconds(raw: str) -> float | None:
    value = raw.strip()
    if not value:
        return None
    return float(value)


def _parse_required_int(raw: str, field_name: str) -> int:
    value = raw.strip()
    if not value:
        raise ValueError(f"missing required integer field `{field_name}`")
    return int(value)


def load_policy(path: Path) -> tuple[dict[str, Any], list[CheckSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks_raw = payload.get("checks")
    if not isinstance(checks_raw, list) or not checks_raw:
        raise ValueError(f"{path} must contain a non-empty `checks` list")
    checks = [CheckSpec.from_dict(item) for item in checks_raw]
    return payload, checks


def load_candidate_rows(paths: list[Path]) -> dict[tuple[str, str], list[CandidateRow]]:
    rows: dict[tuple[str, str], list[CandidateRow]] = {}
    required = {
        "case",
        "phase",
        "seconds",
        "parameter_count",
        "estimated_cpu_peak_bytes",
        "cpu_dense_budget_bytes",
        "status",
        "detail",
    }
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} missing required CSV columns: {', '.join(sorted(missing))}"
                )
            for raw in reader:
                observed = ObservedRow.from_csv_row(raw)
                rows.setdefault((observed.case, observed.phase), []).append(
                    CandidateRow(path=path, observed=observed)
                )
    return rows


def path_matches_source(candidate: Path, source_artifact: str) -> bool:
    source_parts = Path(source_artifact).parts
    candidate_parts = candidate.parts
    if len(source_parts) > len(candidate_parts):
        return False
    return candidate_parts[-len(source_parts) :] == source_parts


def select_observed_row(
    spec: CheckSpec,
    candidates: list[CandidateRow],
    *,
    allow_sole_candidate_source_fallback: bool = True,
) -> SelectionResult:
    if not candidates:
        return SelectionResult(candidate=None, reasons=("missing_row",), selection_mode=None)

    if spec.source_artifact is not None:
        sourced = [
            candidate
            for candidate in candidates
            if path_matches_source(candidate.path, spec.source_artifact)
        ]
        if len(sourced) == 1:
            return SelectionResult(
                candidate=sourced[0],
                reasons=(),
                selection_mode="source_artifact_match",
            )
        if not sourced:
            if allow_sole_candidate_source_fallback and len(candidates) == 1:
                LOG.warning(
                    "%s: source_artifact %r not matched, using sole candidate %s",
                    spec.name,
                    spec.source_artifact,
                    candidates[0].path,
                )
                return SelectionResult(
                    candidate=candidates[0],
                    reasons=(),
                    selection_mode="source_artifact_sole_candidate_fallback",
                )
            return SelectionResult(
                candidate=None,
                reasons=("source_artifact_missing",),
                selection_mode=None,
            )
        if len(sourced) > 1:
            candidates = sourced

    if len(candidates) == 1:
        return SelectionResult(candidate=candidates[0], reasons=(), selection_mode="single_candidate")

    rows = [candidate.observed for candidate in candidates]
    if all(row == rows[0] for row in rows[1:]):
        return SelectionResult(
            candidate=candidates[0],
            reasons=(),
            selection_mode="identical_candidates",
        )
    return SelectionResult(candidate=None, reasons=("ambiguous_row",), selection_mode=None)


def metadata_mismatch_reasons(spec: CheckSpec, observed: ObservedRow) -> list[str]:
    reasons: list[str] = []
    if (
        spec.expected_parameter_count is not None
        and observed.parameter_count != spec.expected_parameter_count
    ):
        reasons.append("parameter_count_mismatch")
    if (
        spec.expected_estimated_cpu_peak_bytes is not None
        and observed.estimated_cpu_peak_bytes != spec.expected_estimated_cpu_peak_bytes
    ):
        reasons.append("estimated_cpu_peak_bytes_mismatch")
    if (
        spec.expected_cpu_dense_budget_bytes is not None
        and observed.cpu_dense_budget_bytes != spec.expected_cpu_dense_budget_bytes
    ):
        reasons.append("cpu_dense_budget_bytes_mismatch")
    return reasons


def observed_value_reasons(spec: CheckSpec, observed: ObservedRow) -> list[str]:
    reasons: list[str] = []
    if observed.status != spec.expected_status:
        reasons.append("status_mismatch")
    if (
        spec.expected_detail_substring is not None
        and spec.expected_detail_substring not in observed.detail
    ):
        reasons.append("detail_mismatch")
    if spec.expected_status != "measured":
        return reasons
    if observed.seconds is None:
        reasons.append("missing_seconds")
        return reasons
    if not math.isfinite(observed.seconds):
        reasons.append("non_finite_seconds")
        return reasons
    allowed = spec.allowed_seconds()
    if allowed is not None and observed.seconds > allowed:
        reasons.append("seconds_exceeded")
    return reasons


def refresh_contract_reasons(spec: CheckSpec, observed: ObservedRow) -> list[str]:
    reasons: list[str] = []
    if observed.status != spec.expected_status:
        reasons.append("status_mismatch")
    if (
        spec.expected_detail_substring is not None
        and spec.expected_detail_substring not in observed.detail
    ):
        reasons.append("detail_mismatch")
    if spec.expected_status == "measured":
        if observed.seconds is None:
            reasons.append("missing_seconds")
        elif not math.isfinite(observed.seconds):
            reasons.append("non_finite_seconds")
    return reasons


def evaluate_check(
    spec: CheckSpec,
    observed_rows: dict[tuple[str, str], list[CandidateRow]],
) -> dict[str, Any]:
    selection = select_observed_row(spec, observed_rows.get((spec.case, spec.phase), []))
    reasons = list(selection.reasons)
    selected = selection.candidate
    observed = None if selected is None else selected.observed

    if observed is not None:
        reasons.extend(metadata_mismatch_reasons(spec, observed))
        reasons.extend(observed_value_reasons(spec, observed))

    return {
        "name": spec.name,
        "case": spec.case,
        "phase": spec.phase,
        "expected_status": spec.expected_status,
        "expected_detail_substring": spec.expected_detail_substring,
        "expected_parameter_count": spec.expected_parameter_count,
        "expected_estimated_cpu_peak_bytes": spec.expected_estimated_cpu_peak_bytes,
        "expected_cpu_dense_budget_bytes": spec.expected_cpu_dense_budget_bytes,
        "baseline_seconds": spec.baseline_seconds,
        "max_regression_ratio": spec.max_regression_ratio,
        "max_seconds": spec.max_seconds,
        "allowed_seconds": spec.allowed_seconds(),
        "source_artifact": spec.source_artifact,
        "selection_mode": selection.selection_mode,
        "observed_candidate": None if selected is None else str(selected.path),
        "observed_status": None if observed is None else observed.status,
        "observed_seconds": None if observed is None else observed.seconds,
        "observed_detail": None if observed is None else observed.detail,
        "observed_parameter_count": None if observed is None else observed.parameter_count,
        "observed_estimated_cpu_peak_bytes": (
            None if observed is None else observed.estimated_cpu_peak_bytes
        ),
        "observed_cpu_dense_budget_bytes": (
            None if observed is None else observed.cpu_dense_budget_bytes
        ),
        "regression": bool(reasons),
        "reasons": reasons,
    }


def normalize_candidate_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        root = Path.cwd()
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def refresh_policy_payload(
    policy_payload: dict[str, Any],
    checks: list[CheckSpec],
    observed_rows: dict[tuple[str, str], list[CandidateRow]],
    *,
    root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_checks = policy_payload.get("checks")
    if not isinstance(raw_checks, list) or len(raw_checks) != len(checks):
        raise ValueError("policy payload/check spec mismatch")

    refreshed_payload = dict(policy_payload)
    refreshed_checks: list[dict[str, Any]] = []
    refresh_results: list[dict[str, Any]] = []

    # The length check above preserves strict pairing on Python 3.9, the
    # repository's documented minimum (where zip(strict=...) is unavailable).
    for raw_check, spec in zip(raw_checks, checks):
        updated_check = dict(raw_check)
        selection = select_observed_row(
            spec,
            observed_rows.get((spec.case, spec.phase), []),
            allow_sole_candidate_source_fallback=False,
        )
        reasons = list(selection.reasons)
        selected = selection.candidate
        observed = None if selected is None else selected.observed

        if observed is not None:
            reasons.extend(metadata_mismatch_reasons(spec, observed))
            reasons.extend(refresh_contract_reasons(spec, observed))
            if not reasons:
                updated_check["source_artifact"] = normalize_candidate_path(selected.path, root)
                if spec.expected_status == "measured":
                    updated_check["baseline_seconds"] = observed.seconds

        refreshed_checks.append(updated_check)
        refresh_results.append(
            {
                "name": spec.name,
                "observed_candidate": None if selected is None else str(selected.path),
                "observed_seconds": None if observed is None else observed.seconds,
                "selection_mode": selection.selection_mode,
                "reasons": reasons,
            }
        )

    refreshed_payload["checks"] = refreshed_checks
    return refreshed_payload, refresh_results


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_to_write = dict(payload)
    existing_content: str | None = None
    try:
        existing_content = path.read_text(encoding="utf-8")
    except OSError:
        existing_content = None

    if existing_content is not None:
        try:
            existing_payload = json.loads(existing_content)
        except json.JSONDecodeError:
            existing_payload = None
        if isinstance(existing_payload, dict):
            normalized_existing = dict(existing_payload)
            normalized_payload = dict(payload_to_write)
            existing_generated_at = normalized_existing.pop("generated_at", None)
            normalized_payload.pop("generated_at", None)
            if normalized_existing == normalized_payload:
                if isinstance(existing_generated_at, str):
                    payload_to_write["generated_at"] = existing_generated_at
                new_content = json.dumps(payload_to_write, indent=2, ensure_ascii=False) + "\n"
                if new_content == existing_content:
                    return

    path.write_text(
        json.dumps(payload_to_write, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
