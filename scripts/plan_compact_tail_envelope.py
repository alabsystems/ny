#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Emit a fail-closed, diagnostic-only compact-tail envelope experiment plan.

The planner consumes one NY ``lpopt dump v1``, its root-margin sidecar, and the
matching solver log.  It validates the exact prop1761 evidence behind the
Graph-MIP escalation gap, then emits a deterministic B0/B1/K2/K4/K8/K16
experiment matrix.  It never loads a GPU backend, constructs a solver model,
or runs a verifier.

The K variants mirror the sound relational contract already used by
``AyTailSharedInputReachabilityEnvelope``:

    lower_a[j] * x + lower_b[j] <= p[j] * z
                                      <= upper_a[j] * x + upper_b[j]

where every row is produced by prefix CROWN over the same certified input box
and coefficient error is folded outward into the biases before export.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "compact_tail_envelope_v1.json"
SCHEMA = "ny_compact_tail_envelope_plan_v1"
VARIANT_IDS = ("B0", "B1", "K2", "K4", "K8", "K16")
SHARED_SUPPORT_ROWS = (2, 4, 8, 16)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SPLIT_RE = re.compile(
    r"\[lpopt-split\]\s+depth=(\d+)\s+bind_obj=(\d+)\s+"
    r"bind_lb=([^\s]+)\s+premises=([^\s]+)"
)
PREMISE_RE = re.compile(r"([^:,\s]+):(\d+):([AI])")
LEAF_DECLINE_RE = re.compile(
    r"Graph-MIP leaf: declined \(free_binaries=(\d+)\s*>\s*"
    r"leaf\s+budget\s+(\d+),\s*depth=(\d+)\)"
)
FRONTIER_RE = re.compile(r"\[frontier\]\s+d=(\d+)\s+worst=([^\s]+)\s+domains=(\d+)")


class PlanError(ValueError):
    """The diagnostic plan cannot be constructed without ambiguity."""


@dataclass(frozen=True)
class FileIdentity:
    """Content identity exported without a host-specific absolute path."""

    filename: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class BoundsBlock:
    """One named finite interval tensor from an lpopt dump."""

    name: str
    shape: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @property
    def elements(self) -> int:
        return len(self.lower)

    def phase_census(self, delta: float = 0.0) -> PhaseCensus:
        if not math.isfinite(delta) or delta < 0.0:
            raise PlanError("phase-census delta must be finite and nonnegative")
        if len(self.lower) != len(self.upper):
            raise PlanError(f"bounds block {self.name!r} has mismatched bounds")
        stable_positive = 0
        stable_negative = 0
        unstable = 0
        for lower, upper in zip(self.lower, self.upper):
            lower -= delta
            upper += delta
            if lower >= 0.0:
                stable_positive += 1
            elif upper <= 0.0:
                stable_negative += 1
            else:
                unstable += 1
        return PhaseCensus(
            stable_positive=stable_positive,
            stable_negative=stable_negative,
            unstable=unstable,
        )

    def width_metrics(self) -> dict[str, float]:
        if len(self.lower) != len(self.upper):
            raise PlanError(f"bounds block {self.name!r} has mismatched bounds")
        widths = [
            upper - lower for lower, upper in zip(self.lower, self.upper)
        ]
        if not widths:
            raise PlanError(f"bounds block {self.name!r} is empty")
        return {
            "minimum": min(widths),
            "mean": math.fsum(widths) / len(widths),
            "maximum": max(widths),
        }


@dataclass(frozen=True)
class PhaseCensus:
    """Stable/unstable classification under one sound pre-activation box."""

    stable_positive: int
    stable_negative: int
    unstable: int

    @property
    def total(self) -> int:
        return self.stable_positive + self.stable_negative + self.unstable

    def as_dict(self) -> dict[str, int]:
        return {
            "stable_positive": self.stable_positive,
            "stable_negative": self.stable_negative,
            "unstable": self.unstable,
            "total": self.total,
        }


@dataclass(frozen=True)
class LpoptDump:
    """Strictly parsed ``ny lpopt dump v1`` evidence."""

    input_bounds: BoundsBlock
    relu_pre_nodes: tuple[tuple[str, str], ...]
    node_bounds: Mapping[str, BoundsBlock]


@dataclass(frozen=True)
class MarginRecord:
    objective_index: int
    lower: float
    upper: float
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective_index": self.objective_index,
            "lower": self.lower,
            "upper": self.upper,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class SplitRecord:
    depth: int
    objective_index: int
    lower: float
    premises: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True)
class SplitFrontier:
    """A complete fixed-depth Boolean partition from the live solver log."""

    depth: int
    objective_index: int
    selectors: tuple[tuple[str, int], ...]
    leaves: tuple[SplitRecord, ...]
    logged_worst: float
    leaf_decline_free_binaries: tuple[int, ...]
    leaf_budget: int

    @property
    def worst_lower(self) -> float:
        return min(leaf.lower for leaf in self.leaves)

    @property
    def best_lower(self) -> float:
        return max(leaf.lower for leaf in self.leaves)


@dataclass(frozen=True)
class ProjectedMetrics:
    """Conservative structural size before any exact-rational conversion."""

    columns: int
    rows: int
    nnz_upper_bound: int
    binaries: int
    support_bank_bytes_upper_bound: int
    added_columns: int
    added_rows: int
    added_nnz_upper_bound: int

    def as_dict(self) -> dict[str, int]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "nnz_upper_bound": self.nnz_upper_bound,
            "binaries": self.binaries,
            "support_bank_bytes_upper_bound": self.support_bank_bytes_upper_bound,
            "added_columns": self.added_columns,
            "added_rows": self.added_rows,
            "added_nnz_upper_bound": self.added_nnz_upper_bound,
        }


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(path: Path) -> FileIdentity:
    if not path.is_file():
        raise PlanError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return FileIdentity(filename=path.name, sha256=digest.hexdigest(), size_bytes=size)


def _require_sha256(identity: FileIdentity, expected: object, context: str) -> None:
    sealed = _parse_sha(expected, f"{context} sealed hash")
    if identity.sha256 != sealed:
        raise PlanError(f"{context} hash mismatch")


def _write_new_atomic(path: Path, data: bytes) -> None:
    """Atomically publish bytes without replacing an existing artifact."""

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        # A same-directory hard link is an atomic, no-clobber publication.
        os.link(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _exact_keys(payload: Mapping[str, Any], expected: set[str], context: str) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise PlanError(
            f"{context} fields differ: missing={missing}, unknown={unknown}"
        )


def _finite_number(raw: object, context: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise PlanError(f"{context} must be a finite number")
    value = float(raw)
    if not math.isfinite(value):
        raise PlanError(f"{context} must be finite")
    return value


def _positive_int(raw: object, context: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise PlanError(f"{context} must be a positive integer")
    return raw


def _parse_sha(raw: object, context: str) -> str:
    if not isinstance(raw, str) or SHA256_RE.fullmatch(raw) is None:
        raise PlanError(f"{context} must be a lowercase SHA-256")
    return raw


def load_manifest(path: Path) -> tuple[dict[str, Any], FileIdentity]:
    """Load and strictly validate the sealed experiment contract."""

    identity = _identity(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot load planner manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PlanError("planner manifest root must be an object")
    _exact_keys(
        payload,
        {
            "schema",
            "diagnostic_only",
            "execution_allowed",
            "authority",
            "target",
            "evidence_sha256",
            "contract",
            "proof_ladder",
            "acceptance",
        },
        "manifest",
    )
    if payload["schema"] != SCHEMA:
        raise PlanError(f"unsupported planner schema: {payload['schema']!r}")
    if (
        payload["diagnostic_only"] is not True
        or payload["execution_allowed"] is not False
        or payload["authority"] is not False
    ):
        raise PlanError(
            "planner manifest must be diagnostic-only, forbid execution, and carry no authority"
        )

    target = payload["target"]
    if not isinstance(target, dict):
        raise PlanError("target must be an object")
    _exact_keys(
        target,
        {
            "id",
            "model_sha256",
            "property_sha256",
            "objective_index",
            "threshold",
            "input_elements",
            "seam_node",
            "seam_elements",
            "relu_node",
            "output_node",
            "output_elements",
            "fixed_tree_selectors",
        },
        "target",
    )
    for key in ("id", "seam_node", "relu_node", "output_node"):
        if not isinstance(target[key], str) or not target[key]:
            raise PlanError(f"target.{key} must be a non-empty string")
    _parse_sha(target["model_sha256"], "target.model_sha256")
    _parse_sha(target["property_sha256"], "target.property_sha256")
    if isinstance(target["objective_index"], bool) or not isinstance(
        target["objective_index"], int
    ):
        raise PlanError("target.objective_index must be an integer")
    _finite_number(target["threshold"], "target.threshold")
    for key in ("input_elements", "seam_elements", "output_elements"):
        _positive_int(target[key], f"target.{key}")
    selectors = target["fixed_tree_selectors"]
    if (
        not isinstance(selectors, list)
        or len(selectors) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in selectors
        )
        or len(set(selectors)) != len(selectors)
    ):
        raise PlanError(
            "target.fixed_tree_selectors must be four distinct nonnegative integers"
        )

    evidence_sha256 = payload["evidence_sha256"]
    if not isinstance(evidence_sha256, dict):
        raise PlanError("evidence_sha256 must be an object")
    _exact_keys(
        evidence_sha256,
        {"lpopt_dump", "root_margins", "solver_log"},
        "evidence_sha256",
    )
    for key, value in evidence_sha256.items():
        _parse_sha(value, f"evidence_sha256.{key}")

    contract = payload["contract"]
    if not isinstance(contract, dict):
        raise PlanError("contract must be an object")
    _exact_keys(
        contract,
        {
            "variants",
            "shared_input_support_rows",
            "encode_delta",
            "max_tail_binaries",
            "max_input_elements",
            "max_seam_elements",
            "max_projected_columns",
            "max_projected_rows",
            "max_projected_nnz",
            "max_support_bank_bytes",
            "usize_bytes",
            "split_depth",
            "fixed_tree_leaves",
        },
        "contract",
    )
    if contract["variants"] != list(VARIANT_IDS):
        raise PlanError("contract variant order or identity changed")
    if contract["shared_input_support_rows"] != list(SHARED_SUPPORT_ROWS):
        raise PlanError("shared-input support schedule must be exactly [2, 4, 8, 16]")
    encode_delta = _finite_number(contract["encode_delta"], "contract.encode_delta")
    if encode_delta < 0.0:
        raise PlanError("contract.encode_delta must be nonnegative")
    for key in (
        "max_tail_binaries",
        "max_input_elements",
        "max_seam_elements",
        "max_projected_columns",
        "max_projected_rows",
        "max_projected_nnz",
        "max_support_bank_bytes",
        "usize_bytes",
        "split_depth",
        "fixed_tree_leaves",
    ):
        _positive_int(contract[key], f"contract.{key}")
    if contract["usize_bytes"] != 8:
        raise PlanError("the sealed bank-size estimate requires 64-bit usize")
    if contract["split_depth"] != len(selectors):
        raise PlanError("split depth must equal fixed-tree selector count")
    if contract["fixed_tree_leaves"] != 1 << contract["split_depth"]:
        raise PlanError("fixed-tree leaf count must cover every selector assignment")

    proof_ladder = payload["proof_ladder"]
    if not isinstance(proof_ladder, list) or not proof_ladder:
        raise PlanError("proof_ladder must be a non-empty list")
    expected_proof_ids = (
        "adaptive-five",
        "fixed-tree-cold",
        "fixed-tree-progressive",
        "exact-decision",
    )
    if [entry.get("id") for entry in proof_ladder if isinstance(entry, dict)] != list(
        expected_proof_ids
    ):
        raise PlanError("proof ladder order or identity changed")
    for index, entry in enumerate(proof_ladder):
        if not isinstance(entry, dict):
            raise PlanError(f"proof_ladder[{index}] must be an object")
        required = {
            "id",
            "route",
            "ny_api",
            "implementation_status",
            "time_cap_seconds",
            "warm_start",
        }
        if entry["id"] == "fixed-tree-progressive":
            required |= {
                "adapter_requirement",
                "root_probe_milliseconds",
                "prefix_milliseconds",
                "start_assignment",
            }
        _exact_keys(entry, required, f"proof_ladder[{index}]")
        if not isinstance(entry["route"], str) or not entry["route"]:
            raise PlanError(f"proof_ladder[{index}].route must be non-empty")
        if not isinstance(entry["ny_api"], str) or not entry["ny_api"]:
            raise PlanError(f"proof_ladder[{index}].ny_api must be non-empty")
        if (
            not isinstance(entry["implementation_status"], str)
            or not entry["implementation_status"]
        ):
            raise PlanError(
                f"proof_ladder[{index}].implementation_status must be non-empty"
            )
        if not isinstance(entry["warm_start"], str) or not entry["warm_start"]:
            raise PlanError(f"proof_ladder[{index}].warm_start must be non-empty")
        if _finite_number(entry["time_cap_seconds"], "proof time cap") <= 0.0:
            raise PlanError("proof time caps must be positive")
    expected_routes = {
        "adaptive-five": (
            "adaptive_five_leaf_comb_target_fsb",
            "certify_linear_lower_bound_at_with_ay_adaptive_five_leaf_comb_target_fsb_admission",
            "solver_api_ready_compact_tail_producer_executor_absent",
            "no_caller_start_ay_internal_probe_and_leaf_warm_reuse",
        ),
        "fixed-tree-cold": (
            "fixed_assignment_tree",
            "certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_admission",
            "solver_api_ready_compact_tail_producer_executor_absent",
            "ay_default_root_cold_then_gray_order_preceding_basis_warm_leaves",
        ),
        "fixed-tree-progressive": (
            "fixed_assignment_tree_progressive_warm_start",
            "certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_compact_progressive_admission",
            "executor_ready_live_prefix_crown_producer_absent",
            "root_probe_then_progressive_prefix",
        ),
        "exact-decision": (
            "certified_exact_decision",
            "certify_linear_lower_bound_at_with_ay_admission",
            "solver_api_ready_compact_tail_producer_executor_absent",
            "no_ny_incumbent_or_branch_advice_ay_internal_defaults",
        ),
    }
    for entry in proof_ladder:
        expected = expected_routes[entry["id"]]
        actual = (
            entry["route"],
            entry["ny_api"],
            entry["implementation_status"],
            entry["warm_start"],
        )
        if actual != expected:
            raise PlanError(f"proof_ladder route contract changed for {entry['id']}")
    progressive = proof_ladder[2]
    adapter_requirement = progressive["adapter_requirement"]
    if (
        not isinstance(adapter_requirement, str)
        or "RootProbeThenProgressivePrefix" not in adapter_requirement
        or "with_fixed_assignment_tree_warm_start" not in adapter_requirement
    ):
        raise PlanError("progressive proof must bind the implemented AY warm-start adapter")
    for key in ("root_probe_milliseconds", "prefix_milliseconds"):
        _positive_int(progressive[key], f"proof_ladder[2].{key}")
    start_assignment = progressive["start_assignment"]
    if (
        isinstance(start_assignment, bool)
        or not isinstance(start_assignment, int)
        or not 0 <= start_assignment < contract["fixed_tree_leaves"]
    ):
        raise PlanError("progressive start_assignment is outside the fixed tree")

    acceptance = payload["acceptance"]
    if not isinstance(acceptance, dict):
        raise PlanError("acceptance must be an object")
    _exact_keys(
        acceptance,
        {
            "max_prefix_bank_wall_seconds",
            "max_proof_wall_seconds",
            "max_peak_rss_bytes",
            "minimum_worst_leaf_improvement",
            "required_certified_lower",
            "required_certificate_scope",
        },
        "acceptance",
    )
    for key in (
        "max_prefix_bank_wall_seconds",
        "max_proof_wall_seconds",
        "minimum_worst_leaf_improvement",
    ):
        if _finite_number(acceptance[key], f"acceptance.{key}") <= 0.0:
            raise PlanError(f"acceptance.{key} must be positive")
    _positive_int(acceptance["max_peak_rss_bytes"], "acceptance.max_peak_rss_bytes")
    _finite_number(acceptance["required_certified_lower"], "required certified lower")
    if float(acceptance["required_certified_lower"]) != float(target["threshold"]):
        raise PlanError("required certified lower must equal the target threshold")
    if (
        acceptance["required_certificate_scope"]
        != "end_to_end_request_bound_prefix_plus_tail"
    ):
        raise PlanError("acceptance requires an end-to-end request-bound certificate")

    return payload, identity


def _parse_float_tokens(
    tokens: Sequence[str], expected: int, context: str
) -> tuple[float, ...]:
    if len(tokens) != expected:
        raise PlanError(f"{context} has {len(tokens)} values, expected {expected}")
    values: list[float] = []
    for index, token in enumerate(tokens):
        try:
            value = float(token)
        except ValueError as error:
            raise PlanError(f"{context}[{index}] is not numeric") from error
        if not math.isfinite(value):
            raise PlanError(f"{context}[{index}] is not finite")
        values.append(value)
    return tuple(values)


def _validate_shape(
    elements: int, shape: Sequence[int], context: str
) -> tuple[int, ...]:
    if not shape or any(value <= 0 for value in shape):
        raise PlanError(f"{context} shape must contain positive dimensions")
    if math.prod(shape) != elements:
        raise PlanError(f"{context} shape product does not equal {elements}")
    return tuple(shape)


def _bounds_block(
    name: str,
    elements: int,
    shape: Sequence[int],
    lower_tokens: Sequence[str],
    upper_tokens: Sequence[str],
) -> BoundsBlock:
    if elements <= 0:
        raise PlanError(f"{name} must have a positive element count")
    parsed_shape = _validate_shape(elements, shape, name)
    lower = _parse_float_tokens(lower_tokens, elements, f"{name}.lower")
    upper = _parse_float_tokens(upper_tokens, elements, f"{name}.upper")
    if len(lower) != len(upper):
        raise PlanError(f"{name} has mismatched bounds")
    for index, (lo, hi) in enumerate(zip(lower, upper)):
        if lo > hi:
            raise PlanError(f"{name} has inverted bound at element {index}")
    return BoundsBlock(name=name, shape=parsed_shape, lower=lower, upper=upper)


def parse_lpopt_dump(path: Path) -> LpoptDump:
    """Parse the complete dump; unknown, duplicate, or malformed records fail."""

    try:
        numbered = [
            (line_number, line.strip())
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            )
            if line.strip()
        ]
    except (OSError, UnicodeError) as error:
        raise PlanError(f"cannot read lpopt dump {path}: {error}") from error
    cursor = 0

    def take(context: str) -> tuple[int, list[str]]:
        nonlocal cursor
        if cursor >= len(numbered):
            raise PlanError(f"lpopt dump ended before {context}")
        line_number, line = numbered[cursor]
        cursor += 1
        return line_number, line.split()

    line_number, header = take("header")
    if line_number != 1 or header != ["#", "ny", "lpopt", "dump", "v1"]:
        raise PlanError("lpopt dump header must be exactly '# ny lpopt dump v1'")

    _, input_header = take("INPUT")
    if len(input_header) < 4 or input_header[0] != "INPUT":
        raise PlanError("lpopt dump INPUT header is malformed")
    try:
        input_elements = int(input_header[1])
        input_shape = [int(value) for value in input_header[2:]]
    except ValueError as error:
        raise PlanError("lpopt INPUT dimensions must be integers") from error
    _, input_lower = take("input lower bounds")
    _, input_upper = take("input upper bounds")
    if not input_lower or input_lower[0] != "L":
        raise PlanError("lpopt INPUT must be followed by L")
    if not input_upper or input_upper[0] != "U":
        raise PlanError("lpopt INPUT lower row must be followed by U")
    input_bounds = _bounds_block(
        "_input",
        input_elements,
        input_shape,
        input_lower[1:],
        input_upper[1:],
    )

    _, relu_header = take("RELUMAP")
    if len(relu_header) != 2 or relu_header[0] != "RELUMAP":
        raise PlanError("lpopt RELUMAP header is malformed")
    try:
        relu_count = int(relu_header[1])
    except ValueError as error:
        raise PlanError("lpopt RELUMAP count must be an integer") from error
    if relu_count <= 0:
        raise PlanError("lpopt RELUMAP must not be empty")
    relu_pre_nodes: list[tuple[str, str]] = []
    seen_relus: set[str] = set()
    for index in range(relu_count):
        _, mapping = take(f"RELUMAP entry {index}")
        if len(mapping) != 2 or not mapping[0] or not mapping[1]:
            raise PlanError(f"lpopt RELUMAP entry {index} is malformed")
        if mapping[0] in seen_relus:
            raise PlanError(f"lpopt RELUMAP repeats {mapping[0]!r}")
        seen_relus.add(mapping[0])
        relu_pre_nodes.append((mapping[0], mapping[1]))

    node_bounds: dict[str, BoundsBlock] = {}
    while cursor < len(numbered):
        _, node_header = take("NODE")
        if len(node_header) < 4 or node_header[0] != "NODE":
            raise PlanError(f"unknown lpopt record: {' '.join(node_header)!r}")
        name = node_header[1]
        if name in node_bounds:
            raise PlanError(f"lpopt dump repeats NODE {name!r}")
        try:
            elements = int(node_header[2])
            shape = [int(value) for value in node_header[3:]]
        except ValueError as error:
            raise PlanError(f"NODE {name!r} dimensions must be integers") from error
        _, lower = take(f"NODE {name} lower bounds")
        _, upper = take(f"NODE {name} upper bounds")
        if not lower or lower[0] != "L" or not upper or upper[0] != "U":
            raise PlanError(f"NODE {name!r} must be followed by one L and one U row")
        node_bounds[name] = _bounds_block(name, elements, shape, lower[1:], upper[1:])

    missing_pre_nodes = sorted(
        {pre for _, pre in relu_pre_nodes if pre not in node_bounds}
    )
    if missing_pre_nodes:
        raise PlanError(
            f"lpopt dump lacks ReLU pre-activation nodes: {missing_pre_nodes}"
        )
    return LpoptDump(
        input_bounds=input_bounds,
        relu_pre_nodes=tuple(relu_pre_nodes),
        node_bounds=node_bounds,
    )


def parse_margins(path: Path) -> tuple[MarginRecord, ...]:
    """Parse the root-margin sidecar with exact contiguous objective order."""

    records: list[MarginRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PlanError(f"cannot read margin sidecar {path}: {error}") from error
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        tokens = raw.split()
        if len(tokens) != 4:
            raise PlanError(f"margin line {line_number} must have four fields")
        try:
            objective_index = int(tokens[0])
            lower, upper, threshold = (float(value) for value in tokens[1:])
        except ValueError as error:
            raise PlanError(f"margin line {line_number} is malformed") from error
        if objective_index != len(records):
            raise PlanError("margin objective indices must be contiguous and ordered")
        if not all(math.isfinite(value) for value in (lower, upper, threshold)):
            raise PlanError(f"margin line {line_number} contains a non-finite value")
        if lower > upper:
            raise PlanError(f"margin line {line_number} has an inverted interval")
        records.append(MarginRecord(objective_index, lower, upper, threshold))
    if not records:
        raise PlanError("margin sidecar is empty")
    return tuple(records)


def _parse_premises(raw: str, context: str) -> tuple[tuple[str, int, str], ...]:
    fields = raw.split(",")
    premises: list[tuple[str, int, str]] = []
    for field in fields:
        match = PREMISE_RE.fullmatch(field)
        if match is None:
            raise PlanError(f"{context} contains malformed premise {field!r}")
        premises.append((match.group(1), int(match.group(2)), match.group(3)))
    keys = [(node, index) for node, index, _ in premises]
    if len(keys) != len(set(keys)):
        raise PlanError(f"{context} repeats a split selector")
    return tuple(premises)


def parse_split_frontier(
    path: Path,
    *,
    depth: int,
    objective_index: int,
    expected_relu: str,
    expected_selectors: Sequence[int],
    expected_leaf_budget: int,
) -> SplitFrontier:
    """Extract and prove completeness of one depth-``d`` Boolean frontier."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PlanError(f"cannot read solver log {path}: {error}") from error

    leaves: list[SplitRecord] = []
    declines: list[tuple[int, int]] = []
    frontiers: list[tuple[float, int]] = []
    for line_number, line in enumerate(lines, start=1):
        split_match = SPLIT_RE.search(line)
        if split_match is not None:
            split_depth = int(split_match.group(1))
            split_objective = int(split_match.group(2))
            try:
                lower = float(split_match.group(3))
            except ValueError as error:
                raise PlanError(
                    f"split line {line_number} has a malformed lower bound"
                ) from error
            if not math.isfinite(lower):
                raise PlanError(
                    f"split line {line_number} has a non-finite lower bound"
                )
            premises = _parse_premises(
                split_match.group(4), f"split line {line_number}"
            )
            if split_depth == depth and split_objective == objective_index:
                leaves.append(
                    SplitRecord(
                        depth=split_depth,
                        objective_index=split_objective,
                        lower=lower,
                        premises=premises,
                    )
                )

        decline_match = LEAF_DECLINE_RE.search(line)
        if decline_match is not None and int(decline_match.group(3)) == depth:
            declines.append((int(decline_match.group(1)), int(decline_match.group(2))))

        frontier_match = FRONTIER_RE.search(line)
        if frontier_match is not None and int(frontier_match.group(1)) == depth:
            try:
                worst = float(frontier_match.group(2))
            except ValueError as error:
                raise PlanError(
                    f"frontier line {line_number} has malformed worst bound"
                ) from error
            if not math.isfinite(worst):
                raise PlanError(
                    f"frontier line {line_number} has a non-finite worst bound"
                )
            frontiers.append((worst, int(frontier_match.group(3))))

    expected_leaves = 1 << depth
    if len(leaves) != expected_leaves:
        raise PlanError(
            f"solver log has {len(leaves)} matching split leaves, expected {expected_leaves}"
        )
    selectors = tuple((node, index) for node, index, _ in leaves[0].premises)
    expected_keys = tuple((expected_relu, index) for index in expected_selectors)
    if selectors != expected_keys:
        raise PlanError(f"frontier selector order changed: {selectors!r}")
    if len(selectors) != depth:
        raise PlanError("frontier premise count does not equal its depth")
    assignments: set[tuple[str, ...]] = set()
    for leaf in leaves:
        keys = tuple((node, index) for node, index, _ in leaf.premises)
        if keys != selectors:
            raise PlanError("split leaves do not share one selector order")
        assignments.add(tuple(state for _, _, state in leaf.premises))
    expected_assignments = set(itertools.product(("A", "I"), repeat=depth))
    if assignments != expected_assignments:
        raise PlanError("split leaves do not cover every Boolean selector assignment")

    if not declines:
        raise PlanError("solver log contains no matching Graph-MIP leaf decline")
    if any(budget != expected_leaf_budget for _, budget in declines):
        raise PlanError("live leaf budget differs from the sealed compact-tail budget")
    if any(free <= budget for free, budget in declines):
        raise PlanError("a logged Graph-MIP decline was not actually over budget")

    matching_frontiers = [
        (worst, domains) for worst, domains in frontiers if domains == expected_leaves
    ]
    if len(matching_frontiers) != 1:
        raise PlanError(
            "solver log must contain one matching complete frontier summary"
        )
    logged_worst, _ = matching_frontiers[0]
    computed_worst = min(leaf.lower for leaf in leaves)
    if not math.isclose(logged_worst, computed_worst, rel_tol=0.0, abs_tol=5.0e-5):
        raise PlanError("frontier worst bound disagrees with its split records")

    return SplitFrontier(
        depth=depth,
        objective_index=objective_index,
        selectors=selectors,
        leaves=tuple(leaves),
        logged_worst=logged_worst,
        leaf_decline_free_binaries=tuple(free for free, _ in declines),
        leaf_budget=expected_leaf_budget,
    )


def _full_relu_census(dump: LpoptDump) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total = 0
    for relu, pre in dump.relu_pre_nodes:
        census = dump.node_bounds[pre].phase_census()
        rows.append(
            {
                "relu_node": relu,
                "preactivation_node": pre,
                **census.as_dict(),
            }
        )
        total += census.unstable
    return rows, total


def _base_tail_metrics(
    seam_elements: int,
    output_elements: int,
    census: PhaseCensus,
) -> ProjectedMetrics:
    # Graph-MIP aliases stable-positive ReLU coordinates and gives each
    # stable-negative coordinate a fixed-zero column. Every unstable coordinate
    # contributes one post-ReLU column, one binary, three rows, and 7 row
    # coefficients. The final dense Linear still receives all seam coordinates.
    # Include one conservative output decision row so every proof arm fits the
    # projection.
    columns = (
        seam_elements + 2 * census.unstable + census.stable_negative + output_elements
    )
    rows = 3 * census.unstable + output_elements + 1
    nnz = 7 * census.unstable + output_elements * (1 + seam_elements) + output_elements
    return ProjectedMetrics(
        columns=columns,
        rows=rows,
        nnz_upper_bound=nnz,
        binaries=census.unstable,
        support_bank_bytes_upper_bound=0,
        added_columns=0,
        added_rows=0,
        added_nnz_upper_bound=0,
    )


def _variant_metrics(
    variant_id: str,
    *,
    base: ProjectedMetrics,
    input_elements: int,
    seam_elements: int,
    usize_bytes: int,
) -> ProjectedMetrics:
    if variant_id == "B0":
        return base
    if variant_id == "B1":
        added_rows = 1
        added_nnz = seam_elements
        # One dense p plus its certified scalar lower/upper bounds, all f32.
        bank_bytes = (seam_elements + 2) * 4
        return ProjectedMetrics(
            columns=base.columns,
            rows=base.rows + added_rows,
            nnz_upper_bound=base.nnz_upper_bound + added_nnz,
            binaries=base.binaries,
            support_bank_bytes_upper_bound=bank_bytes,
            added_columns=0,
            added_rows=added_rows,
            added_nnz_upper_bound=added_nnz,
        )
    if not variant_id.startswith("K"):
        raise PlanError(f"unknown compact-tail variant {variant_id!r}")
    try:
        supports = int(variant_id[1:])
    except ValueError as error:
        raise PlanError(f"malformed compact-tail variant {variant_id!r}") from error
    if supports not in SHARED_SUPPORT_ROWS:
        raise PlanError(f"unsupported shared-input support count {supports}")

    added_columns = input_elements
    added_rows = 2 * supports
    # Each lower/upper row may be dense in both z and the shared root x.
    added_nnz = added_rows * (seam_elements + input_elements)
    # directions + lower_a + lower_b + upper_a + upper_b are f32; support
    # indices are usize.  This matches AyTailSharedInputReachabilityBank.
    floats = supports * (seam_elements + 2 * input_elements + 2)
    bank_bytes = floats * 4 + supports * usize_bytes
    return ProjectedMetrics(
        columns=base.columns + added_columns,
        rows=base.rows + added_rows,
        nnz_upper_bound=base.nnz_upper_bound + added_nnz,
        binaries=base.binaries,
        support_bank_bytes_upper_bound=bank_bytes,
        added_columns=added_columns,
        added_rows=added_rows,
        added_nnz_upper_bound=added_nnz,
    )


def _check_projected_caps(
    variant_id: str, metrics: ProjectedMetrics, contract: Mapping[str, Any]
) -> None:
    checks = (
        ("columns", metrics.columns, contract["max_projected_columns"]),
        ("rows", metrics.rows, contract["max_projected_rows"]),
        ("nnz", metrics.nnz_upper_bound, contract["max_projected_nnz"]),
        ("binaries", metrics.binaries, contract["max_tail_binaries"]),
        (
            "support bank bytes",
            metrics.support_bank_bytes_upper_bound,
            contract["max_support_bank_bytes"],
        ),
    )
    for label, observed, limit in checks:
        if observed > limit:
            raise PlanError(
                f"variant {variant_id} projected {label} {observed} exceeds cap {limit}"
            )


def _validate_target_evidence(
    *,
    manifest: Mapping[str, Any],
    model: FileIdentity,
    property_file: FileIdentity,
    dump: LpoptDump,
    margins: Sequence[MarginRecord],
    frontier: SplitFrontier,
) -> tuple[BoundsBlock, BoundsBlock, PhaseCensus, list[dict[str, Any]], int]:
    target = manifest["target"]
    contract = manifest["contract"]
    if model.sha256 != target["model_sha256"]:
        raise PlanError(
            f"model hash mismatch: expected {target['model_sha256']}, observed {model.sha256}"
        )
    if property_file.sha256 != target["property_sha256"]:
        raise PlanError(
            "property hash mismatch: "
            f"expected {target['property_sha256']}, observed {property_file.sha256}"
        )
    if dump.input_bounds.elements != target["input_elements"]:
        raise PlanError("lpopt input dimension differs from the sealed target")
    if dump.input_bounds.elements > contract["max_input_elements"]:
        raise PlanError("lpopt input exceeds the generic shared-input resource cap")

    relu_map = dict(dump.relu_pre_nodes)
    if relu_map.get(target["relu_node"]) != target["seam_node"]:
        raise PlanError("target ReLU is not mapped to the declared seam pre-activation")
    seam = dump.node_bounds.get(target["seam_node"])
    output = dump.node_bounds.get(target["output_node"])
    if seam is None or output is None:
        raise PlanError("lpopt dump lacks the declared compact-tail seam or output")
    if seam.elements != target["seam_elements"]:
        raise PlanError("seam dimension differs from the sealed target")
    if output.elements != target["output_elements"]:
        raise PlanError("output dimension differs from the sealed target")
    if seam.elements > contract["max_seam_elements"]:
        raise PlanError("seam exceeds the generic compact-tail resource cap")

    seam_census = seam.phase_census()
    if seam_census.unstable == 0:
        raise PlanError(
            "compact tail has no unstable ReLU and does not need this experiment"
        )
    if seam_census.unstable > contract["max_tail_binaries"]:
        raise PlanError("compact tail itself exceeds the exact-binary cap")
    for selector in target["fixed_tree_selectors"]:
        if selector >= seam.elements:
            raise PlanError("fixed-tree selector is outside the seam")
        if not seam.lower[selector] < 0.0 < seam.upper[selector]:
            raise PlanError(
                f"fixed-tree selector {selector} is not unstable at the seam"
            )

    census_rows, full_unstable = _full_relu_census(dump)
    if full_unstable <= contract["max_tail_binaries"]:
        raise PlanError(
            "whole graph already fits the leaf cap; compact-tail gap is absent"
        )
    if not frontier.leaf_decline_free_binaries:
        raise PlanError("live log does not demonstrate the whole-graph leaf gap")

    objective_index = target["objective_index"]
    if not 0 <= objective_index < len(margins):
        raise PlanError("target objective is absent from the root-margin sidecar")
    margin = margins[objective_index]
    if margin.threshold != float(target["threshold"]):
        raise PlanError("root-margin threshold differs from the sealed target")
    if margin.lower >= margin.threshold:
        raise PlanError("target objective is already root-verified")
    if frontier.objective_index != objective_index:
        raise PlanError("split frontier binds a different objective")
    return seam, output, seam_census, census_rows, full_unstable


def build_plan(
    *,
    manifest_path: Path,
    model_path: Path,
    property_path: Path,
    lpopt_path: Path,
    margins_path: Path,
    solver_log_path: Path,
) -> dict[str, Any]:
    """Construct the sealed plan without performing any experiment work."""

    manifest, manifest_identity = load_manifest(manifest_path)
    model_identity = _identity(model_path)
    property_identity = _identity(property_path)
    lpopt_identity = _identity(lpopt_path)
    margins_identity = _identity(margins_path)
    solver_log_identity = _identity(solver_log_path)
    sealed_evidence = manifest["evidence_sha256"]
    _require_sha256(lpopt_identity, sealed_evidence["lpopt_dump"], "lpopt dump")
    _require_sha256(margins_identity, sealed_evidence["root_margins"], "root margins")
    _require_sha256(solver_log_identity, sealed_evidence["solver_log"], "solver log")

    dump = parse_lpopt_dump(lpopt_path)
    margins = parse_margins(margins_path)
    target = manifest["target"]
    contract = manifest["contract"]
    frontier = parse_split_frontier(
        solver_log_path,
        depth=contract["split_depth"],
        objective_index=target["objective_index"],
        expected_relu=target["relu_node"],
        expected_selectors=target["fixed_tree_selectors"],
        expected_leaf_budget=contract["max_tail_binaries"],
    )
    seam, output, seam_census, census_rows, full_unstable = _validate_target_evidence(
        manifest=manifest,
        model=model_identity,
        property_file=property_identity,
        dump=dump,
        margins=margins,
        frontier=frontier,
    )

    encoded_seam_census = seam.phase_census(delta=float(contract["encode_delta"]))
    base = _base_tail_metrics(seam.elements, output.elements, encoded_seam_census)
    variants: list[dict[str, Any]] = []
    variant_metrics: dict[str, ProjectedMetrics] = {}
    for variant_id in VARIANT_IDS:
        metrics = _variant_metrics(
            variant_id,
            base=base,
            input_elements=dump.input_bounds.elements,
            seam_elements=seam.elements,
            usize_bytes=contract["usize_bytes"],
        )
        _check_projected_caps(variant_id, metrics, contract)
        variant_metrics[variant_id] = metrics
        if variant_id == "B0":
            support_mode = "independent_seam_box"
            supports = 0
            retains_shared_input = False
        elif variant_id == "B1":
            support_mode = "scalar_prefix_crown_support"
            supports = 1
            retains_shared_input = False
        else:
            support_mode = "shared_input_prefix_crown_envelope"
            supports = int(variant_id[1:])
            retains_shared_input = True
        variants.append(
            {
                "id": variant_id,
                "support_mode": support_mode,
                "supports": supports,
                "retains_shared_input": retains_shared_input,
                "projected_metrics": metrics.as_dict(),
            }
        )

    proof_matrix = [
        {
            "id": f"{variant_id}/{proof['id']}",
            "variant": variant_id,
            "proof": proof["id"],
            "route": proof["route"],
            "ny_api": proof["ny_api"],
            "implementation_status": proof["implementation_status"],
            "time_cap_seconds": proof["time_cap_seconds"],
            "warm_start": proof["warm_start"],
        }
        for variant_id in VARIANT_IDS
        for proof in manifest["proof_ladder"]
    ]
    margin = margins[target["objective_index"]]
    k16 = variant_metrics["K16"]

    return {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "execution_allowed": False,
        "authority": False,
        "manifest": manifest_identity.as_dict(),
        "target": {
            **target,
            "model": model_identity.as_dict(),
            "property": property_identity.as_dict(),
        },
        "evidence": {
            "lpopt_dump": lpopt_identity.as_dict(),
            "root_margins": margins_identity.as_dict(),
            "solver_log": solver_log_identity.as_dict(),
        },
        "observations": {
            "input": {
                "elements": dump.input_bounds.elements,
                "shape": list(dump.input_bounds.shape),
                "width": dump.input_bounds.width_metrics(),
            },
            "root_margin": margin.as_dict(),
            "full_network_unstable_binaries": full_unstable,
            "relu_census": census_rows,
            "compact_tail": {
                "seam_node": seam.name,
                "seam_elements": seam.elements,
                "seam_width": seam.width_metrics(),
                "phase_census": seam_census.as_dict(),
                "encoded_phase_census": encoded_seam_census.as_dict(),
                "encode_delta": contract["encode_delta"],
                "output_node": output.name,
                "output_elements": output.elements,
            },
            "fixed_tree_frontier": {
                "depth": frontier.depth,
                "leaves": len(frontier.leaves),
                "objective_index": frontier.objective_index,
                "selectors": [
                    {"relu_node": node, "neuron_index": index}
                    for node, index in frontier.selectors
                ],
                "worst_lower": frontier.worst_lower,
                "best_lower": frontier.best_lower,
                "logged_worst": frontier.logged_worst,
                "leaf_decline_free_binaries": list(frontier.leaf_decline_free_binaries),
                "leaf_budget": frontier.leaf_budget,
            },
        },
        "future_executor_requirements": {
            "tail": (
                "the dark executor encodes the declared live seam box through exact "
                "Relu_57 and Gemm_58 and admits only certified lower-bound evidence"
            ),
            "scalar_support": (
                "a future executor may add B1's range row L <= p*z <= U only after "
                "prefix CROWN certifies both endpoints over the same root input box"
            ),
            "shared_input_support": (
                "the K16 executor encodes every row as "
                "lower_a*x+lower_b <= p*z <= upper_a*x+upper_b against one shared "
                "latent root input x"
            ),
            "coefficient_error": (
                "a future executor must fold every prefix-CROWN coefficient-error "
                "interval outward into lower_b/upper_b over the certified root box "
                "before export"
            ),
            "fixed_tree": (
                "the executor uses candidate selectors only after binary_keys resolves "
                "four distinct unfixed integer [0,1] columns; all 16 AY-certified "
                "LP leaves and independent NY replays are required, while every "
                "unselected indicator remains relaxed"
            ),
            "admission": (
                "the executor requires AY to exactify every weak/Farkas row "
                "and NY to independently replay all obligations; timeout, weak leaf, "
                "or malformed evidence must remain unknown"
            ),
            "request_binding": (
                "a future executor must bind the certified prefix rows, tail model, "
                "objective, threshold, graph, model, property, and root box into one "
                "end-to-end request before any verdict can gain authority"
            ),
        },
        "resource_contract": {
            **contract,
            "k16_projected_nnz": k16.nnz_upper_bound,
            "k16_support_bank_bytes": k16.support_bank_bytes_upper_bound,
            "current_imb_cgan_caps": {
                "algebra_reused": True,
                "max_latent_inputs": 16,
                "max_added_nnz": 131072,
                "max_bank_bytes": 262144,
                "directly_admits_cifar_root_input": False,
                "reason": (
                    "the existing IMB caps are cGAN-specific: a 3072-element "
                    "shared input exceeds max_latent_inputs, and K16's bank exceeds "
                    "the 256 KiB payload cap"
                ),
            },
        },
        "producer_request": {
            "seam_node": target["seam_node"],
            "relu_node": target["relu_node"],
            "output_node": target["output_node"],
            "maximum_supports": max(SHARED_SUPPORT_ROWS),
            "nested_support_prefixes": list(SHARED_SUPPORT_ROWS),
            "direction_policy": [
                "current objective tail-CROWN slope first",
                "phase-conditioned tail slopes in fixed-tree Gray order",
                "remaining alpha-checkpoint slopes in deterministic checkpoint order",
                "greedy rank-preserving selection with index tie-breaks",
            ],
            "requirements": [
                "all directions finite and nonzero",
                "every K bank has full row rank and is a prefix of K16",
                "one batched prefix CROWN call produces lower/upper input-affine rows",
                "coefficient error is folded over the exact certified root input box",
                "export f32 values by exact bit pattern, never rounded display text",
                "preserve the typed objective index and current lower bound in every request",
                "revalidate graph/model/property/seam identities before model construction",
            ],
        },
        "variants": variants,
        "proof_ladder": manifest["proof_ladder"],
        "experiment_matrix": proof_matrix,
        "acceptance": {
            **manifest["acceptance"],
            "experiment_promotion": (
                "promote an experiment only if a variant certifies the target lower "
                "bound or improves the worst live leaf by at least the sealed amount "
                "while staying under every wall/RSS/model-size cap"
            ),
            "verdict_authority": (
                "requires an end-to-end request-bound prefix-plus-tail certificate "
                "accepted through the named NY certified API; a conditional AY tail "
                "result or this diagnostic plan carries no verdict authority"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate prop1761 Graph-MIP gap evidence and emit a diagnostic-only "
            "compact-tail shared-input envelope experiment plan."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", dest="property_path", type=Path, required=True)
    parser.add_argument("--lpopt-dump", type=Path, required=True)
    parser.add_argument("--root-margins", type=Path, required=True)
    parser.add_argument("--solver-log", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="write canonical JSON here; omit to print it to stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = build_plan(
            manifest_path=args.manifest,
            model_path=args.model,
            property_path=args.property_path,
            lpopt_path=args.lpopt_dump,
            margins_path=args.root_margins,
            solver_log_path=args.solver_log,
        )
        encoded = _canonical_json(payload)
        if args.output is None:
            sys.stdout.buffer.write(encoded)
        else:
            # Diagnostic artifacts are evidence: never replace an earlier run.
            _write_new_atomic(args.output, encoded)
    except (OSError, PlanError) as error:
        print(f"compact-tail plan declined: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
