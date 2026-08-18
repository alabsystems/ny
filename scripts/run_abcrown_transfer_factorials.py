#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Run the staged alpha-beta-CROWN recipe-transfer factorials.

The committed experiment manifest names a base NY preset and a set of
single-field or bundled treatments.  This driver materializes every treatment
as a standalone YAML file, hashes the exact input and generated presets, and
delegates bounded execution to ``benchmark_vnncomp_preset_bounded.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = (
    REPO_ROOT / "configs" / "experiments" / "abcrown_transfer_factorials.yaml"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "abcrown_transfer_factorials"
BOUNDED_RUNNER = REPO_ROOT / "scripts" / "benchmark_vnncomp_preset_bounded.py"
EXPECTED_EFFECTIVE_CONFIG_SCHEMA = "ny_beta_crown_effective_treatment_v1"
EXPECTED_EXECUTION_OBSERVATIONS_SCHEMA = (
    "ny_beta_crown_execution_observations_v5"
)
SUPPORTED_MEASURED_RESULTS = frozenset(
    {"verified", "falsified", "unknown", "timeout"}
)
REQUIRED_EFFECTIVE_CONFIG_SECTIONS = frozenset(
    {
        "batch",
        "branching",
        "attack",
        "alpha_crown",
        "beta_crown",
        "clip",
        "root",
        "invprop",
        "route",
    }
)
REQUIRED_INVPROP_FIELDS = frozenset(
    {
        "enabled",
        "apply_output_constraints_to",
        "tighten_input_bounds",
        "best_of_oc_and_no_oc",
        "directly_optimize",
        "share_gammas",
        "per_layer_gammas",
        "optimize_gammas",
        "gamma_lr",
        "top_level_output_constraint_matrix",
        "serial_clause_rebinding",
        "split_lift_requested",
        "split_lift_effective_armed",
    }
)
HARNESS_OWNED_EXTRA_FLAGS = frozenset(
    {
        "--property",
        "-p",
        "--preset",
        "--timeout",
        "--json",
        "--max-domains",
        "--domain-batch-metrics-jsonl",
        "--input-split-metrics-jsonl",
    }
)
NY_RECEIPT_FIELDS = (
    "schema",
    "binary_sha256",
    "source_kind",
    "source_commit",
    "source_state_sha256",
    "cargo_lock_sha256",
    "ay_commit",
    "features",
    "toolchain_kind",
    "toolchain_sha256",
    "artifact_provenance_sha256",
)
NY_RECEIPT_SCHEMA = "ny-submission-binary-receipt-v1"
PROMOTION_RECEIPT_FEATURES = "mip,cuda"
REPO_HEAD_TIMEOUT_SECONDS = 10.0

EXACT_C_BOOLEAN_FIELDS = frozenset(
    {
        "observed",
        "selected_iteration_limit_conflict",
        "selected_compressed_conflict",
        "attribution_conflict",
        "iteration_count_conflict",
        "counter_overflow",
    }
)
EXACT_C_COUNTER_FIELDS = frozenset(
    {
        "selections",
        "layout_observations",
        "source_rows",
        "evaluated_rows",
        "precertified_rows",
        "compressed_selections",
        "compressed_layouts_finalized",
        "compressed_layouts_rolled_back",
        "compact_commits",
        "compact_reconstruction_succeeded",
        "compact_reconstruction_failed",
        "compact_binding_map_succeeded",
        "compact_binding_map_failed",
        "compact_alpha_candidates",
        "compact_alpha_published",
        "compact_alpha_dropped",
        "outcomes_observed",
        "refused_before_commit",
        "committed",
        "iteration_count_outcomes",
        "attempted_iterations",
        "accepted_iterations",
    }
)
EXACT_C_NULLABLE_BOOLEAN_FIELDS = frozenset({"selected_compressed"})
ROOT_SPEC_PRUNE_BOOLEAN_FIELDS = frozenset(
    {"observed", "attribution_conflict", "route_conflict", "counter_overflow"}
)
ROOT_SPEC_PRUNE_NULLABLE_BOOLEAN_FIELDS = frozenset({"configured"})
ROOT_SPEC_PRUNE_COUNTER_FIELDS = frozenset(
    {
        "route_observations",
        "plans_built",
        "applied",
        "layout_observations",
        "source_rows",
        "evaluated_rows",
        "precertified_rows",
        "all_pruned",
    }
)
INVPROP_BOOLEAN_FIELDS = frozenset(
    {"observed", "attribution_conflict", "counter_overflow"}
)
INVPROP_COUNTER_FIELDS = frozenset(
    {
        "clause_rebind_attempts",
        "clause_rebind_accepted",
        "clause_rebind_refused",
        "alpha_initializations",
        "gamma_steps_attempted",
        "gamma_steps_applied",
        "nonzero_output_seed_folds",
        "nonzero_evaluated_output_seed_folds",
    }
)
FRESH_DOMAIN_CLIP_BOOLEAN_FIELDS = frozenset(
    {"observed", "attribution_conflict", "route_conflict", "counter_overflow"}
)
FRESH_DOMAIN_CLIP_NULLABLE_BOOLEAN_FIELDS = frozenset(
    {"configured", "route_authorized"}
)
FRESH_DOMAIN_CLIP_COUNTER_FIELDS = frozenset(
    {
        "route_observations",
        "attempts",
        "applied",
        "all_clauses_refuted",
        "skipped",
        "tightened_dimensions",
    }
)
PATCHES_MATERIALIZATION_BOOLEAN_FIELDS = frozenset(
    {"observed", "attribution_conflict", "counter_overflow"}
)
PATCHES_MATERIALIZATION_COUNTER_FIELDS = frozenset(
    {
        "attempts",
        "succeeded",
        "refused",
        "finite_deadline_attempts",
        "no_deadline_attempts",
        "affine_geometry_attempts",
        "anchored_geometry_attempts",
        "conflicting_geometry_attempts",
        "input_coefficient_error_attempts",
        "coefficient_error_absent",
        "coefficient_error_materialized",
        "memory_refusals",
        "deadline_refusals",
        "semantic_refusals",
        "memory_receipt_outcomes",
        "nominal_required_bytes",
        "capacity_overage_bytes",
        "admitted_bytes",
        "budget_bytes",
    }
)
PATCHES_MATERIALIZATION_PURPOSES = (
    "latent_input_crossover",
    "network_input_terminal",
    "other",
)
PATCHES_MATERIALIZATION_PURPOSE_COUNTER_FIELDS = (
    "attempts",
    "succeeded",
    "refused",
)


def _discover_git_executable() -> Path | None:
    candidates = [Path("/usr/bin/git"), Path("/bin/git")]
    discovered = shutil.which("git")
    if discovered is not None:
        candidates.append(Path(discovered))
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    return None


GIT_EXECUTABLE = _discover_git_executable()


def _effective_value(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = document
    for component in path:
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _as_f32(value: Any) -> float:
    """Round a numeric manifest value exactly as the resolved Rust f32 field."""
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _expected_treatment_authentication(
    arm: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Map arm fields onto the stable effective-treatment projection.

    Fields without a unique representation are returned as unauthenticated and
    force the arm to fail closed until NY's projection is expanded.
    """
    checks: list[dict[str, Any]] = []
    unsupported: list[str] = []

    def add(source: str, path: tuple[str, ...], expected: Any) -> None:
        checks.append(
            {"source": source, "path": list(path), "expected": expected}
        )

    direct_overrides: dict[str, tuple[tuple[str, ...], str | None]] = {
        "general.device": (("route", "proof_backend"), "lower"),
        "general.conv_mode": (("route", "configured_conv_mode"), "lower"),
        "attack.pgd_restarts": (("attack", "pgd_restarts"), None),
        "bab.batch_size": (("batch", "configured_size"), None),
        "solver.build_batch_size": (("batch", "build_batch_size"), None),
        "bab.auto_enlarge_batch_size": (("batch", "auto_enlarge"), None),
        "bab.branching.candidates": (
            ("branching", "configured_candidates"),
            None,
        ),
        "bab.branching.reduceop": (
            ("branching", "configured_reduce_op"),
            None,
        ),
        "bab.branching.kfsb_multi": (
            ("branching", "kfsb_multi_configured"),
            None,
        ),
        "bab.branching.input_split.sb_coeff_thresh": (
            ("branching", "input_split_coeff_threshold"),
            "f32",
        ),
        "bab.branching.input_split.reorder_bab": (
            ("branching", "reorder_bab"),
            None,
        ),
        "bab.branching.input_split.adv_check": (
            ("branching", "input_split_adv_check"),
            None,
        ),
        "bab.alpha_crown.lr_alpha": (("alpha_crown", "learning_rate"), "f32"),
        "bab.alpha_crown.iterations": (("alpha_crown", "iterations"), None),
        "bab.beta_crown.lr_alpha": (
            ("beta_crown", "learning_rate_alpha"),
            "f32",
        ),
        "bab.beta_crown.lr_beta": (
            ("beta_crown", "learning_rate_beta"),
            "f32",
        ),
        "bab.beta_crown.iterations": (("beta_crown", "iterations"), None),
        "bab.clip.interm_domain": (("clip", "interm_domain"), None),
        "bab.clip.interm_topk": (("clip", "interm_topk"), None),
        "bab.clip.in_alpha_crown": (("clip", "in_alpha_crown"), None),
        "bab.clip.input_split_fresh_domain_clip": (
            ("clip", "input_split_fresh_domain_clip_configured"),
            None,
        ),
        "bab.interm_transfer": (("route", "intermediate_bound_transfer"), None),
        "bab.root_crown_interm_dense_head": (
            ("root", "dense_head_configured"),
            None,
        ),
        "bab.atomic_root_c_margin_iterations": (
            ("root", "atomic_root_c_margin_iterations"),
            None,
        ),
    }
    overrides = arm.get("overrides", {})
    if not isinstance(overrides, dict):
        return checks, ["overrides"]
    for field, value in overrides.items():
        if field in direct_overrides:
            path, normalization = direct_overrides[field]
            if normalization == "lower":
                expected = str(value).lower()
            elif normalization == "f32":
                expected = _as_f32(value)
            else:
                expected = value
            add(field, path, expected)
        elif field == "solver.bound_prop_method" and value == "crown":
            add(field, ("alpha_crown", "enabled"), False)
        elif field == "solver.bound_prop_method" and value == "forward+backward":
            add(field, ("route", "use_forward_bounds"), True)
        elif field == "bab.branching.method" and value in {"kfsb", "sb"}:
            add(
                field,
                ("branching", "heuristic"),
                "kfsb" if value == "kfsb" else "input_split",
            )
        elif field == "bab.branching.input_split.enable" and value is True:
            add(field, ("branching", "heuristic"), "input_split")
        elif field == "model.vgg_abcrown_treatment" and value is True:
            add(field, ("route", "vgg_abcrown_treatment_active"), True)
        elif field == "attack.pgd_order" and value in {
            "before",
            "input_bab",
            "skip",
            "none",
            "disabled",
        }:
            schedule = {
                "before": "upfront",
                "input_bab": "input_bab",
                "skip": "disabled",
                "none": "disabled",
                "disabled": "disabled",
            }[value]
            add(field, ("attack", "schedule"), schedule)
        else:
            unsupported.append(field)

    env_paths: dict[str, tuple[tuple[str, ...], str]] = {
        "NY_ALPHA_FINAL_BOUND_ONLY": (
            ("alpha_crown", "final_bound_only_env_armed"),
            "one",
        ),
        "NY_ADAPTIVE_MICROBATCH_CONTROLLER": (
            ("batch", "adaptive_microbatch_controller_armed"),
            "one",
        ),
        "NY_PACKED_GRAPH_ALPHA_QUEUE": (
            ("alpha_crown", "packed_graph_alpha_queue_env_armed"),
            "one",
        ),
        "NY_ROOT_SPARSE_INTERM_CROWN": (
            ("root", "sparse_effective_armed"),
            "one",
        ),
        "NY_ROOT_SPEC_PRUNE": (
            ("root", "root_spec_prune_requested"),
            "zero_or_one",
        ),
        "NY_INVPROP": (("invprop", "enabled"), "truthy"),
        "NY_INVPROP_OPTIMIZE": (("invprop", "optimize_gammas"), "truthy"),
        "NY_INVPROP_LR": (("invprop", "gamma_lr"), "positive_f32"),
        "NY_INVPROP_SPLIT_LIFT": (
            ("invprop", "split_lift_requested"),
            "split_truthy",
        ),
    }
    env = arm.get("env", {})
    if not isinstance(env, dict):
        unsupported.append("env")
    else:
        for field, value in env.items():
            if field not in env_paths:
                unsupported.append(field)
                continue
            path, normalization = env_paths[field]
            rendered = str(value).strip()
            if normalization == "one":
                if rendered != "1":
                    unsupported.append(field)
                    continue
                expected = True
            elif normalization == "zero_or_one":
                if rendered not in {"0", "1"}:
                    unsupported.append(field)
                    continue
                expected = rendered == "1"
            elif normalization == "truthy":
                expected = rendered.lower() in {"1", "true", "on", "yes"}
            elif normalization == "split_truthy":
                expected = rendered.lower() in {"1", "true", "on"}
            elif normalization == "positive_f32":
                try:
                    expected = _as_f32(rendered)
                except (TypeError, ValueError, OverflowError):
                    unsupported.append(field)
                    continue
                if expected <= 0.0:
                    unsupported.append(field)
                    continue
            else:
                raise AssertionError(f"unknown env normalization {normalization}")
            add(field, path, expected)
            if field == "NY_INVPROP_SPLIT_LIFT":
                # A requested treatment must actually be armed to qualify for
                # promotion. The current research gate has no production call
                # sites, so requesting it deliberately fails this check.
                add(
                    field,
                    ("invprop", "split_lift_effective_armed"),
                    expected,
                )
    return checks, sorted(set(unsupported))


def _expected_execution_evidence(arm: dict[str, Any]) -> dict[str, Any] | None:
    """Describe runtime evidence required to promote an algorithm treatment.

    This is keyed by the treatment-bearing manifest fields rather than arm or
    experiment names, so renamed and newly added factorials inherit the same
    fail-closed contract.
    """
    overrides = arm.get("overrides", {})
    if not isinstance(overrides, dict):
        return None
    expectations: list[dict[str, Any]] = []
    exact_c_field = "bab.atomic_root_c_margin_iterations"
    if exact_c_field in overrides:
        iteration_limit = overrides[exact_c_field]
        if (
            isinstance(iteration_limit, bool)
            or not isinstance(iteration_limit, int)
            or iteration_limit < 0
        ):
            raise ManifestError(
                f"{exact_c_field} must be a non-negative integer for execution evidence"
            )
        expectations.append(
            {
                "treatment": "exact_c",
                "expected_iteration_limit": iteration_limit,
            }
        )

    fresh_clip_field = "bab.clip.input_split_fresh_domain_clip"
    if fresh_clip_field in overrides:
        configured = overrides[fresh_clip_field]
        if not isinstance(configured, bool):
            raise ManifestError(
                f"{fresh_clip_field} must be boolean for execution evidence"
            )
        expectations.append(
            {
                "treatment": "fresh_domain_clip",
                "configured": configured,
            }
        )

    env = arm.get("env", {})
    if isinstance(env, dict):
        if "NY_ROOT_SPEC_PRUNE" in env:
            prune_value = str(env["NY_ROOT_SPEC_PRUNE"]).strip()
            if prune_value not in {"0", "1"}:
                raise ManifestError(
                    "NY_ROOT_SPEC_PRUNE must be exactly 0 or 1 for execution evidence"
                )
            expectations.append(
                {
                    "treatment": "root_spec_prune",
                    "configured": prune_value == "1",
                }
            )
        invprop_enabled = str(env.get("NY_INVPROP", "")).strip().lower() in {
            "1",
            "true",
            "on",
            "yes",
        }
        if invprop_enabled and "NY_INVPROP_OPTIMIZE" in env:
            optimize_value = str(env["NY_INVPROP_OPTIMIZE"]).strip().lower()
            if optimize_value in {"1", "true", "on", "yes"}:
                optimize_gammas = True
            elif optimize_value in {"0", "false", "off", "no"}:
                optimize_gammas = False
            else:
                raise ManifestError(
                    "NY_INVPROP_OPTIMIZE must be a boolean value for execution evidence"
                )
            expectations.append(
                {
                    "treatment": "invprop_gamma",
                    "optimize_gammas": optimize_gammas,
                }
            )
    if {expectation["treatment"] for expectation in expectations} == {
        "exact_c",
        "root_spec_prune",
    } and len(expectations) == 2:
        exact = next(
            expectation
            for expectation in expectations
            if expectation["treatment"] == "exact_c"
        )
        prune = next(
            expectation
            for expectation in expectations
            if expectation["treatment"] == "root_spec_prune"
        )
        return {
            "treatment": "exact_c_root_spec_prune",
            "expected_iteration_limit": exact["expected_iteration_limit"],
            "prune_configured": prune["configured"],
        }
    if len(expectations) > 1:
        treatments = ", ".join(
            expectation["treatment"] for expectation in expectations
        )
        raise ManifestError(
            "one factorial arm may not combine runtime-gated treatments: "
            f"{treatments}"
        )
    return expectations[0] if expectations else None


class ManifestError(ValueError):
    """The factorial manifest is malformed or refers to an invalid target."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _exact_c_multi_iteration_aggregate_issues(exact_c: dict[str, Any]) -> list[str]:
    """Validate the bounded v5 authentication aggregates."""
    issues: list[str] = []
    boolean_fields = (
        "multiplicative_weights_requested_conflict",
        "gradient_plan_num_specs_conflict",
        "gradient_row_count_conflict",
        "multi_iteration_evidence_conflict",
    )
    counter_fields = (
        "multi_iteration_evidence_outcomes",
        "multiplicative_weights_plan_dispatched_outcomes",
        "multiplicative_weights_effective_outcomes",
        "completed_proposals",
        "adaptive_plan_dispatches",
    )
    types_valid = True
    for field in boolean_fields:
        if not isinstance(exact_c.get(field), bool):
            issues.append(f"execution_observations.exact_c.{field} is not boolean")
            types_valid = False
    for field in counter_fields:
        if not _is_nonnegative_int(exact_c.get(field)):
            issues.append(
                f"execution_observations.exact_c.{field} is not a "
                "non-negative integer"
            )
            types_valid = False
    requested = exact_c.get("multiplicative_weights_requested")
    if requested is not None and not isinstance(requested, bool):
        issues.append(
            "execution_observations.exact_c.multiplicative_weights_requested "
            "is not null or boolean"
        )
        types_valid = False
    for field in ("gradient_plan_num_specs", "gradient_row_count"):
        value = exact_c.get(field)
        if value is not None and not _is_nonnegative_int(value):
            issues.append(
                f"execution_observations.exact_c.{field} is not null or a "
                "non-negative integer"
            )
            types_valid = False
    aggregate_fields = (
        "iteration_count_outcomes",
        "committed",
        "attempted_iterations",
        "accepted_iterations",
    )
    if not types_valid or not all(
        _is_nonnegative_int(exact_c.get(field)) for field in aggregate_fields
    ):
        return issues

    evidence = exact_c["multi_iteration_evidence_outcomes"]
    iteration_outcomes = exact_c["iteration_count_outcomes"]
    committed = exact_c["committed"]
    attempted = exact_c["attempted_iterations"]
    accepted = exact_c["accepted_iterations"]
    plan_outcomes = exact_c["multiplicative_weights_plan_dispatched_outcomes"]
    effective_outcomes = exact_c["multiplicative_weights_effective_outcomes"]
    completed = exact_c["completed_proposals"]
    adaptive = exact_c["adaptive_plan_dispatches"]
    num_specs = exact_c.get("gradient_plan_num_specs")
    row_count = exact_c.get("gradient_row_count")
    selected_limit = exact_c.get("selected_iteration_limit")
    selected_limit_conflict = exact_c.get("selected_iteration_limit_conflict")

    if evidence != iteration_outcomes:
        issues.append(
            "exact-C multi-iteration evidence outcomes does not equal "
            "iteration_count_outcomes"
        )
    if evidence > committed:
        issues.append("exact-C multi-iteration evidence outcomes exceeds commits")
    if completed > attempted:
        issues.append("exact-C completed proposals exceeds attempted iterations")
    if completed < max(0, attempted - evidence):
        issues.append(
            "exact-C completed proposals is below the per-outcome completion bound"
        )
    if accepted > completed:
        issues.append("exact-C accepted iterations exceeds completed proposals")
    if plan_outcomes > evidence:
        issues.append("exact-C MW plan outcomes exceeds evidence outcomes")
    if effective_outcomes > plan_outcomes:
        issues.append("exact-C MW effective outcomes exceeds MW plan outcomes")
    if adaptive > attempted:
        issues.append("exact-C adaptive plan dispatches exceeds attempted iterations")
    if (
        selected_limit_conflict is False
        and _is_nonnegative_int(selected_limit)
        and attempted > evidence * selected_limit
    ):
        issues.append(
            "exact-C attempted iterations exceeds selected iteration limits"
        )
    issues.extend(
        f"exact-C {field} is present"
        for field in boolean_fields
        if exact_c[field]
    )

    if evidence == 0:
        if attempted != 0 or accepted != 0:
            issues.append(
                "exact-C iteration counts exist without authenticated evidence"
            )
        if requested is not None:
            issues.append("exact-C MW request is present without authenticated evidence")
        if any(value != 0 for value in (plan_outcomes, effective_outcomes, completed, adaptive)):
            issues.append("exact-C gradient work exists without authenticated evidence")
        if num_specs is not None or row_count is not None:
            issues.append("exact-C gradient shape is present without authenticated evidence")
        return issues

    if requested is None:
        issues.append("exact-C MW request is absent with authenticated evidence")
    if row_count is None or row_count == 0:
        issues.append("exact-C gradient row count is absent or zero")
    if (attempted == 0) != (num_specs is None):
        issues.append(
            "exact-C gradient num_specs presence disagrees with attempted iterations"
        )
    if requested is True and row_count is not None:
        if (attempted > 0) != (plan_outcomes > 0):
            issues.append("exact-C MW plan outcomes disagree with attempted iterations")
        if attempted < plan_outcomes:
            issues.append("exact-C attempted iterations is below MW plan outcomes")
        if num_specs is not None and num_specs != row_count:
            issues.append("exact-C MW gradient num_specs does not equal row count")
        expected_adaptive = attempted - plan_outcomes if row_count > 1 else 0
        if adaptive != expected_adaptive:
            issues.append(
                "exact-C adaptive plan dispatches disagrees with attempted plans "
                "and MW plan outcomes"
            )
        if completed < adaptive:
            issues.append("exact-C completed proposals is below adaptive dispatches")
        if effective_outcomes > adaptive:
            issues.append("exact-C MW effective outcomes exceeds adaptive dispatches")
        if completed < 2 * effective_outcomes:
            issues.append(
                "exact-C completed proposals cannot authenticate MW effective outcomes"
            )
        if row_count == 1 and completed < attempted - plan_outcomes:
            issues.append(
                "exact-C one-row MW completions are below active-outcome bound"
            )
        if row_count > 1 and completed >= adaptive:
            completed_final_plans = completed - adaptive
            if completed_final_plans > plan_outcomes:
                issues.append(
                    "exact-C completed final MW plans exceeds MW plan outcomes"
                )
            if effective_outcomes == 0:
                if completed > plan_outcomes:
                    issues.append(
                        "exact-C completed proposals require an MW effective outcome"
                    )
            else:
                minimum_adaptive_for_effective = (
                    2 * effective_outcomes
                    - min(effective_outcomes, completed_final_plans)
                )
                if adaptive < minimum_adaptive_for_effective:
                    issues.append(
                        "exact-C adaptive plans cannot realize MW effective outcomes"
                    )
    elif requested is False:
        if plan_outcomes != 0 or effective_outcomes != 0 or adaptive != 0:
            issues.append("exact-C MW work occurred while MW was not requested")
        if num_specs not in (None, 1):
            issues.append("exact-C non-MW gradient num_specs is not one")
    return issues


def _execution_observations_structure_issues(document: Any) -> list[str]:
    """Validate the current runtime schema while permitting future sections."""
    issues: list[str] = []
    if not isinstance(document, dict):
        return ["execution_observations is not an object"]
    if document.get("schema") != EXPECTED_EXECUTION_OBSERVATIONS_SCHEMA:
        issues.append(
            "execution_observations schema is not "
            f"{EXPECTED_EXECUTION_OBSERVATIONS_SCHEMA!r}"
        )
    issues.extend(
        f"execution_observations.{field} is not boolean"
        for field in ("run_active", "recording_conflict")
        if not isinstance(document.get(field), bool)
    )
    if document.get("recording_conflict") is True:
        issues.append("execution observation recording conflict is present")

    exact_c = document.get("exact_c")
    if not isinstance(exact_c, dict):
        issues.append("execution_observations.exact_c is not an object")
    else:
        issues.extend(_exact_c_multi_iteration_aggregate_issues(exact_c))
        exact_types_valid = True
        for field in EXACT_C_BOOLEAN_FIELDS:
            if not isinstance(exact_c.get(field), bool):
                issues.append(f"execution_observations.exact_c.{field} is not boolean")
                exact_types_valid = False
        for field in EXACT_C_COUNTER_FIELDS:
            if not _is_nonnegative_int(exact_c.get(field)):
                issues.append(
                    f"execution_observations.exact_c.{field} is not a "
                    "non-negative integer"
                )
                exact_types_valid = False
        for field in EXACT_C_NULLABLE_BOOLEAN_FIELDS:
            value = exact_c.get(field)
            if value is not None and not isinstance(value, bool):
                issues.append(
                    f"execution_observations.exact_c.{field} is not null or boolean"
                )
                exact_types_valid = False
        selected_limit = exact_c.get("selected_iteration_limit")
        if selected_limit is not None and not _is_nonnegative_int(selected_limit):
            issues.append(
                "execution_observations.exact_c.selected_iteration_limit is not "
                "null or a non-negative integer"
            )
            exact_types_valid = False
        stop_reasons = exact_c.get("stop_reasons")
        stop_reasons_valid = isinstance(stop_reasons, dict) and all(
            isinstance(reason, str)
            and bool(reason)
            and _is_nonnegative_int(count)
            and count > 0
            for reason, count in (
                stop_reasons.items() if isinstance(stop_reasons, dict) else []
            )
        )
        if not stop_reasons_valid:
            issues.append(
                "execution_observations.exact_c.stop_reasons is not a map of "
                "non-empty strings to positive integers"
            )
        if exact_types_valid and stop_reasons_valid:
            selections = exact_c["selections"]
            outcomes = exact_c["outcomes_observed"]
            refused = exact_c["refused_before_commit"]
            committed = exact_c["committed"]
            counted_outcomes = exact_c["iteration_count_outcomes"]
            attempted = exact_c["attempted_iterations"]
            accepted = exact_c["accepted_iterations"]
            layout_observations = exact_c["layout_observations"]
            source_rows = exact_c["source_rows"]
            evaluated_rows = exact_c["evaluated_rows"]
            precertified_rows = exact_c["precertified_rows"]
            compressed = exact_c["compressed_selections"]
            finalized = exact_c["compressed_layouts_finalized"]
            rolled_back = exact_c["compressed_layouts_rolled_back"]
            compact_commits = exact_c["compact_commits"]
            reconstruction_succeeded = exact_c[
                "compact_reconstruction_succeeded"
            ]
            reconstruction_failed = exact_c[
                "compact_reconstruction_failed"
            ]
            binding_succeeded = exact_c["compact_binding_map_succeeded"]
            binding_failed = exact_c["compact_binding_map_failed"]
            alpha_candidates = exact_c["compact_alpha_candidates"]
            alpha_published = exact_c["compact_alpha_published"]
            alpha_dropped = exact_c["compact_alpha_dropped"]
            if selections != outcomes:
                issues.append("exact-C selections does not equal outcomes_observed")
            if refused + committed != outcomes:
                issues.append(
                    "exact-C refused_before_commit + committed does not equal "
                    "outcomes_observed"
                )
            if sum(stop_reasons.values()) != outcomes:
                issues.append(
                    "exact-C stop-reason total does not equal outcomes_observed"
                )
            if counted_outcomes > committed:
                issues.append(
                    "exact-C iteration_count_outcomes exceeds committed outcomes"
                )
            if accepted > attempted:
                issues.append(
                    "exact-C accepted_iterations exceeds attempted_iterations"
                )
            if counted_outcomes == 0 and (attempted != 0 or accepted != 0):
                issues.append(
                    "exact-C aggregate iteration counts exist without a counted outcome"
                )
            if layout_observations != selections:
                issues.append("exact-C layout observations does not equal selections")
            if evaluated_rows + precertified_rows != source_rows:
                issues.append(
                    "exact-C evaluated + precertified rows does not equal source rows"
                )
            if selections > 0 and source_rows == 0:
                issues.append("exact-C selected a zero-row layout")
            if compressed > selections:
                issues.append("exact-C compressed selections exceeds selections")
            if finalized + rolled_back != compressed:
                issues.append(
                    "exact-C finalized + rolled-back layouts does not equal "
                    "compressed selections"
                )
            if reconstruction_succeeded + reconstruction_failed != compact_commits:
                issues.append(
                    "exact-C reconstruction outcomes do not equal compact commits"
                )
            if binding_succeeded + binding_failed != compact_commits:
                issues.append(
                    "exact-C binding-map outcomes do not equal compact commits"
                )
            if compact_commits > committed or compact_commits > finalized:
                issues.append(
                    "exact-C compact commits exceed committed/finalized outcomes"
                )
            if alpha_published + alpha_dropped != alpha_candidates:
                issues.append(
                    "exact-C alpha publication outcomes do not equal candidates"
                )
            if alpha_candidates > compact_commits:
                issues.append("exact-C alpha candidates exceeds compact commits")
            if (
                alpha_published > reconstruction_succeeded
                or alpha_published > binding_succeeded
            ):
                issues.append(
                    "exact-C published alpha lacks successful reconstruction/binding"
                )
            if exact_c["counter_overflow"]:
                issues.append("exact-C counter overflow is present")
            if exact_c["attribution_conflict"]:
                issues.append("exact-C attribution conflict is present")
            if exact_c["selected_iteration_limit_conflict"]:
                issues.append("exact-C selected iteration limit conflict is present")
            if exact_c["selected_compressed_conflict"]:
                issues.append("exact-C selected compressed-layout conflict is present")
            if exact_c["iteration_count_conflict"]:
                issues.append("exact-C iteration-count conflict is present")
            activity_observed = selections > 0 or outcomes > 0
            if exact_c["observed"] != activity_observed:
                issues.append("exact-C observed flag disagrees with recorded activity")
            if exact_c["selected_iteration_limit_conflict"]:
                if selected_limit is not None:
                    issues.append(
                        "exact-C conflicting selected limits retain a selected limit"
                    )
            elif (selections > 0) != (selected_limit is not None):
                issues.append(
                    "exact-C selected iteration limit disagrees with selections"
                )
            selected_compressed = exact_c["selected_compressed"]
            if exact_c["selected_compressed_conflict"]:
                if selected_compressed is not None:
                    issues.append(
                        "exact-C conflicting compression selections retain a disposition"
                    )
            elif (selections > 0) != (selected_compressed is not None):
                issues.append(
                    "exact-C selected compression disposition disagrees with selections"
                )
            elif selected_compressed is True and compressed != selections:
                issues.append(
                    "exact-C common compressed disposition disagrees with selections"
                )
            elif selected_compressed is False and compressed != 0:
                issues.append(
                    "exact-C common full-layout disposition has compressed selections"
                )

    root_prune = document.get("root_spec_prune")
    if not isinstance(root_prune, dict):
        issues.append("execution_observations.root_spec_prune is not an object")
    else:
        prune_types_valid = True
        for field in ROOT_SPEC_PRUNE_BOOLEAN_FIELDS:
            if not isinstance(root_prune.get(field), bool):
                issues.append(
                    f"execution_observations.root_spec_prune.{field} is not boolean"
                )
                prune_types_valid = False
        for field in ROOT_SPEC_PRUNE_NULLABLE_BOOLEAN_FIELDS:
            value = root_prune.get(field)
            if value is not None and not isinstance(value, bool):
                issues.append(
                    "execution_observations.root_spec_prune."
                    f"{field} is not null or boolean"
                )
                prune_types_valid = False
        for field in ROOT_SPEC_PRUNE_COUNTER_FIELDS:
            if not _is_nonnegative_int(root_prune.get(field)):
                issues.append(
                    "execution_observations.root_spec_prune."
                    f"{field} is not a non-negative integer"
                )
                prune_types_valid = False
        if prune_types_valid:
            routes = root_prune["route_observations"]
            plans = root_prune["plans_built"]
            applied = root_prune["applied"]
            layouts = root_prune["layout_observations"]
            source_rows = root_prune["source_rows"]
            evaluated_rows = root_prune["evaluated_rows"]
            precertified_rows = root_prune["precertified_rows"]
            all_pruned = root_prune["all_pruned"]
            if root_prune["observed"] != (routes > 0):
                issues.append(
                    "root-spec prune observed flag disagrees with route observations"
                )
            if (routes > 0) != (root_prune["configured"] is not None):
                issues.append(
                    "root-spec prune configured state disagrees with route observations"
                )
            if root_prune["configured"] is not True and (plans > 0 or applied > 0):
                issues.append("root-spec prune work occurred while not configured")
            if plans > routes:
                issues.append("root-spec prune plans exceeds route observations")
            if applied > plans:
                issues.append("root-spec prune applied layouts exceeds plans")
            if layouts != applied:
                issues.append(
                    "root-spec prune layout observations does not equal applied layouts"
                )
            if evaluated_rows + precertified_rows != source_rows:
                issues.append(
                    "root-spec prune evaluated + precertified rows does not equal source rows"
                )
            if applied == 0 and (source_rows != 0 or evaluated_rows != 0 or precertified_rows != 0):
                issues.append("root-spec prune row totals exist without an applied layout")
            if applied > 0 and source_rows == 0:
                issues.append("root-spec prune applied a zero-row layout")
            if all_pruned > applied:
                issues.append("root-spec prune all-pruned count exceeds applied layouts")
            if root_prune["counter_overflow"]:
                issues.append("root-spec prune counter overflow is present")
            if root_prune["attribution_conflict"]:
                issues.append("root-spec prune attribution conflict is present")
            if root_prune["route_conflict"]:
                issues.append("root-spec prune route conflict is present")

            if isinstance(exact_c, dict) and all(
                _is_nonnegative_int(exact_c.get(field))
                for field in (
                    "compressed_layouts_finalized",
                    "source_rows",
                    "evaluated_rows",
                    "precertified_rows",
                )
            ):
                finalized = exact_c["compressed_layouts_finalized"]
                if finalized > applied:
                    issues.append(
                        "finalized exact-C layouts exceeds root-prune applications"
                    )
                if (
                    exact_c.get("selected_compressed") is True
                    and finalized > 0
                    and finalized == applied
                ):
                    for field in (
                        "source_rows",
                        "evaluated_rows",
                        "precertified_rows",
                    ):
                        if exact_c[field] != root_prune[field]:
                            issues.append(
                                "finalized exact-C/root-prune row totals disagree "
                                f"for {field}"
                            )

    invprop = document.get("invprop")
    if not isinstance(invprop, dict):
        issues.append("execution_observations.invprop is not an object")
    else:
        invprop_types_valid = True
        for field in INVPROP_BOOLEAN_FIELDS:
            if not isinstance(invprop.get(field), bool):
                issues.append(f"execution_observations.invprop.{field} is not boolean")
                invprop_types_valid = False
        for field in INVPROP_COUNTER_FIELDS:
            if not _is_nonnegative_int(invprop.get(field)):
                issues.append(
                    f"execution_observations.invprop.{field} is not a "
                    "non-negative integer"
                )
                invprop_types_valid = False
        if invprop_types_valid:
            attempts = invprop["clause_rebind_attempts"]
            accepted = invprop["clause_rebind_accepted"]
            refused = invprop["clause_rebind_refused"]
            gamma_attempted = invprop["gamma_steps_attempted"]
            gamma_applied = invprop["gamma_steps_applied"]
            nonzero_folds = invprop["nonzero_output_seed_folds"]
            evaluated_folds = invprop[
                "nonzero_evaluated_output_seed_folds"
            ]
            if accepted + refused != attempts:
                issues.append(
                    "INVPROP accepted + refused clause rebinds does not equal attempts"
                )
            if gamma_applied > gamma_attempted:
                issues.append("INVPROP applied gamma steps exceeds attempted steps")
            if evaluated_folds > nonzero_folds:
                issues.append(
                    "INVPROP evaluated output-seed folds exceeds total output-seed "
                    "folds"
                )
            if gamma_attempted == 0 and evaluated_folds != 0:
                issues.append(
                    "INVPROP evaluated output-seed folds exist without attempted "
                    "gamma steps"
                )
            if gamma_applied == 0 and evaluated_folds != 0:
                issues.append(
                    "INVPROP evaluated output-seed folds exist without an applied "
                    "gamma step"
                )
            if nonzero_folds == 0 and evaluated_folds != 0:
                issues.append(
                    "INVPROP evaluated output-seed folds exist without any "
                    "output-seed fold"
                )
            if invprop["counter_overflow"]:
                issues.append("INVPROP counter overflow is present")
            if invprop["attribution_conflict"]:
                issues.append("INVPROP attribution conflict is present")
            activity_observed = any(
                invprop[field] > 0 for field in INVPROP_COUNTER_FIELDS
            )
            if invprop["observed"] != activity_observed:
                issues.append("INVPROP observed flag disagrees with recorded activity")

    fresh_clip = document.get("fresh_domain_clip")
    if not isinstance(fresh_clip, dict):
        issues.append(
            "execution_observations.fresh_domain_clip is not an object"
        )
    else:
        fresh_types_valid = True
        for field in FRESH_DOMAIN_CLIP_BOOLEAN_FIELDS:
            if not isinstance(fresh_clip.get(field), bool):
                issues.append(
                    "execution_observations.fresh_domain_clip."
                    f"{field} is not boolean"
                )
                fresh_types_valid = False
        for field in FRESH_DOMAIN_CLIP_NULLABLE_BOOLEAN_FIELDS:
            value = fresh_clip.get(field)
            if value is not None and not isinstance(value, bool):
                issues.append(
                    "execution_observations.fresh_domain_clip."
                    f"{field} is not null or boolean"
                )
                fresh_types_valid = False
        for field in FRESH_DOMAIN_CLIP_COUNTER_FIELDS:
            if not _is_nonnegative_int(fresh_clip.get(field)):
                issues.append(
                    "execution_observations.fresh_domain_clip."
                    f"{field} is not a non-negative integer"
                )
                fresh_types_valid = False
        if fresh_types_valid:
            attempts = fresh_clip["attempts"]
            dispositions = (
                fresh_clip["applied"]
                + fresh_clip["all_clauses_refuted"]
                + fresh_clip["skipped"]
            )
            if dispositions != attempts:
                issues.append(
                    "fresh-domain clip disposition total does not equal attempts"
                )
            if attempts > 0 and fresh_clip["route_authorized"] is not True:
                issues.append(
                    "fresh-domain clip attempted work without route authorization"
                )
            if (
                fresh_clip["route_authorized"] is True
                and fresh_clip["configured"] is not True
            ):
                issues.append(
                    "fresh-domain clip route was authorized while not configured"
                )
            activity_observed = (
                fresh_clip["route_observations"] > 0 or attempts > 0
            )
            if fresh_clip["observed"] != activity_observed:
                issues.append(
                    "fresh-domain clip observed flag disagrees with recorded activity"
                )
            if fresh_clip["route_observations"] == 0 and (
                fresh_clip["configured"] is not None
                or fresh_clip["route_authorized"] is not None
            ):
                issues.append(
                    "fresh-domain clip route state exists without a route observation"
                )
            if fresh_clip["route_observations"] > 0 and (
                fresh_clip["configured"] is None
                or fresh_clip["route_authorized"] is None
            ):
                issues.append(
                    "fresh-domain clip route observation lacks resolved route state"
                )
            if (
                fresh_clip["applied"] == 0
                and fresh_clip["tightened_dimensions"] > 0
            ):
                issues.append(
                    "fresh-domain clip tightened dimensions without an applied outcome"
                )
            if fresh_clip["counter_overflow"]:
                issues.append("fresh-domain clip counter overflow is present")
            if fresh_clip["attribution_conflict"]:
                issues.append("fresh-domain clip attribution conflict is present")
            if fresh_clip["route_conflict"]:
                issues.append("fresh-domain clip route conflict is present")

    patches = document.get("patches_materialization")
    if not isinstance(patches, dict):
        issues.append(
            "execution_observations.patches_materialization is not an object"
        )
    else:
        patches_types_valid = True
        for field in PATCHES_MATERIALIZATION_BOOLEAN_FIELDS:
            if not isinstance(patches.get(field), bool):
                issues.append(
                    "execution_observations.patches_materialization."
                    f"{field} is not boolean"
                )
                patches_types_valid = False
        for field in PATCHES_MATERIALIZATION_COUNTER_FIELDS:
            if not _is_nonnegative_int(patches.get(field)):
                issues.append(
                    "execution_observations.patches_materialization."
                    f"{field} is not a non-negative integer"
                )
                patches_types_valid = False

        purposes_valid = True
        for purpose_name in PATCHES_MATERIALIZATION_PURPOSES:
            purpose = patches.get(purpose_name)
            if not isinstance(purpose, dict):
                issues.append(
                    "execution_observations.patches_materialization."
                    f"{purpose_name} is not an object"
                )
                purposes_valid = False
                continue
            for field in PATCHES_MATERIALIZATION_PURPOSE_COUNTER_FIELDS:
                if not _is_nonnegative_int(purpose.get(field)):
                    issues.append(
                        "execution_observations.patches_materialization."
                        f"{purpose_name}.{field} is not a non-negative integer"
                    )
                    purposes_valid = False

        if patches_types_valid and purposes_valid:
            attempts = patches["attempts"]
            succeeded = patches["succeeded"]
            refused = patches["refused"]
            purpose_attempts = sum(
                patches[name]["attempts"]
                for name in PATCHES_MATERIALIZATION_PURPOSES
            )
            purpose_succeeded = sum(
                patches[name]["succeeded"]
                for name in PATCHES_MATERIALIZATION_PURPOSES
            )
            purpose_refused = sum(
                patches[name]["refused"]
                for name in PATCHES_MATERIALIZATION_PURPOSES
            )
            if patches["observed"] != (attempts > 0):
                issues.append(
                    "patches materialization observed flag disagrees with attempts"
                )
            if succeeded + refused != attempts:
                issues.append(
                    "patches materialization outcomes do not equal attempts"
                )
            if purpose_attempts != attempts:
                issues.append(
                    "patches materialization purpose attempts do not equal attempts"
                )
            if purpose_succeeded != succeeded or purpose_refused != refused:
                issues.append(
                    "patches materialization purpose outcomes disagree with totals"
                )
            if (
                patches["finite_deadline_attempts"]
                + patches["no_deadline_attempts"]
                != attempts
            ):
                issues.append(
                    "patches materialization deadline dispositions do not equal attempts"
                )
            if (
                patches["affine_geometry_attempts"]
                + patches["anchored_geometry_attempts"]
                + patches["conflicting_geometry_attempts"]
                != attempts
            ):
                issues.append(
                    "patches materialization geometry dispositions do not equal attempts"
                )
            if patches["input_coefficient_error_attempts"] > attempts:
                issues.append(
                    "patches materialization coefficient-error inputs exceed attempts"
                )
            if (
                patches["memory_refusals"]
                + patches["deadline_refusals"]
                + patches["semantic_refusals"]
                != refused
            ):
                issues.append(
                    "patches materialization refusal dispositions do not equal refusals"
                )
            if (
                patches["coefficient_error_absent"]
                + patches["coefficient_error_materialized"]
                != succeeded
            ):
                issues.append(
                    "patches materialization coefficient-error outcomes do not equal successes"
                )
            if patches["memory_receipt_outcomes"] != succeeded:
                issues.append(
                    "patches materialization memory receipts do not equal successes"
                )
            if (
                patches["nominal_required_bytes"]
                + patches["capacity_overage_bytes"]
                != patches["admitted_bytes"]
            ):
                issues.append(
                    "patches materialization admitted-byte receipt does not balance"
                )
            if patches["admitted_bytes"] > patches["budget_bytes"]:
                issues.append("patches materialization admitted bytes exceed budget")
            if patches["conflicting_geometry_attempts"] > refused:
                issues.append(
                    "patches materialization conflicting geometries exceed refusals"
                )
            if patches["counter_overflow"]:
                issues.append("patches materialization counter overflow is present")
            if patches["attribution_conflict"]:
                issues.append(
                    "patches materialization attribution conflict is present"
                )
    return issues


def _exact_c_treatment_issues(
    exact_c: Any,
    expected_limit: int,
    *,
    require_compressed: bool | None,
) -> list[str]:
    issues: list[str] = []
    if not isinstance(exact_c, dict):
        return ["exact-C runtime observations are absent"]
    if expected_limit == 0:
        inactive = (
            exact_c.get("selected_iteration_limit") is None
            and exact_c.get("selected_compressed") is None
            and all(exact_c.get(field) is False for field in EXACT_C_BOOLEAN_FIELDS)
            and all(exact_c.get(field) == 0 for field in EXACT_C_COUNTER_FIELDS)
            and exact_c.get("stop_reasons") == {}
        )
        if not inactive:
            issues.append("exact-C OFF arm recorded exact-C execution")
        return issues

    selections = exact_c.get("selections")
    outcomes = exact_c.get("outcomes_observed")
    if exact_c.get("observed") is not True:
        issues.append("exact-C ON arm did not observe the typed route")
    if not _is_nonnegative_int(selections) or selections == 0:
        issues.append("exact-C ON arm has no route selection")
    if not _is_nonnegative_int(outcomes) or outcomes == 0:
        issues.append("exact-C ON arm has no route outcome")
    if (
        _is_nonnegative_int(selections)
        and _is_nonnegative_int(outcomes)
        and selections != outcomes
    ):
        issues.append("exact-C selections and outcomes are inconsistent")
    if exact_c.get("selected_iteration_limit") != expected_limit:
        issues.append(
            "exact-C selected iteration limit does not match the arm: "
            f"expected {expected_limit}, observed "
            f"{exact_c.get('selected_iteration_limit')!r}"
        )
    for conflict in (
        "selected_iteration_limit_conflict",
        "selected_compressed_conflict",
        "attribution_conflict",
        "iteration_count_conflict",
    ):
        if exact_c.get(conflict) is not False:
            issues.append(f"exact-C {conflict} is present")
    counted = exact_c.get("iteration_count_outcomes")
    if not _is_nonnegative_int(counted) or counted == 0:
        issues.append("exact-C ON arm has no exact iteration-count outcome")
    attempted_iterations = exact_c.get("attempted_iterations")
    if not _is_nonnegative_int(attempted_iterations) or attempted_iterations == 0:
        issues.append("exact-C ON arm executed no exact-C iteration")

    if require_compressed is False:
        if exact_c.get("selected_compressed") is not False:
            issues.append("exact-C prune-OFF arm did not select the full row layout")
        for field in (
            "compressed_selections",
            "compressed_layouts_finalized",
            "compressed_layouts_rolled_back",
            "compact_commits",
            "compact_reconstruction_succeeded",
            "compact_reconstruction_failed",
            "compact_binding_map_succeeded",
            "compact_binding_map_failed",
            "compact_alpha_candidates",
            "compact_alpha_published",
            "compact_alpha_dropped",
        ):
            if exact_c.get(field) != 0:
                issues.append(
                    f"exact-C prune-OFF arm recorded compact activity in {field}"
                )
    elif require_compressed is True:
        if exact_c.get("selected_compressed") is not True:
            issues.append("combined arm did not select compressed exact-C rows")
        for field, description in (
            ("compressed_selections", "compressed selection"),
            ("compressed_layouts_finalized", "finalized compressed layout"),
            ("compact_commits", "compact commit"),
            ("compact_reconstruction_succeeded", "successful reconstruction"),
            ("compact_binding_map_succeeded", "successful binding map"),
            ("compact_alpha_candidates", "selected compact alpha candidate"),
            ("compact_alpha_published", "published compact alpha candidate"),
        ):
            value = exact_c.get(field)
            if not _is_nonnegative_int(value) or value == 0:
                issues.append(f"combined arm has no {description}")
        for field, description in (
            ("compressed_layouts_rolled_back", "rolled-back compressed layout"),
            ("compact_reconstruction_failed", "failed reconstruction"),
            ("compact_binding_map_failed", "failed binding map"),
            ("compact_alpha_dropped", "dropped compact alpha candidate"),
        ):
            if exact_c.get(field) != 0:
                issues.append(f"combined arm recorded a {description}")
    return issues


def _root_spec_prune_treatment_issues(
    root_prune: Any, configured: bool
) -> list[str]:
    issues: list[str] = []
    if not isinstance(root_prune, dict):
        return ["root-spec prune runtime observations are absent"]
    if root_prune.get("observed") is not True:
        issues.append("root-spec prune arm did not observe the dispatcher route")
    routes = root_prune.get("route_observations")
    if not _is_nonnegative_int(routes) or routes == 0:
        issues.append("root-spec prune arm has no route observation")
    if root_prune.get("configured") is not configured:
        issues.append("root-spec prune configured state does not match the arm")
    for conflict in ("route_conflict", "attribution_conflict", "counter_overflow"):
        if root_prune.get(conflict) is not False:
            issues.append(f"root-spec prune {conflict} is present")

    if configured:
        for field, description in (
            ("plans_built", "validated plan"),
            ("applied", "applied layout"),
            ("layout_observations", "layout observation"),
            ("source_rows", "source row"),
            ("precertified_rows", "precertified row"),
        ):
            value = root_prune.get(field)
            if not _is_nonnegative_int(value) or value == 0:
                issues.append(f"root-spec prune ON arm has no {description}")
        source_rows = root_prune.get("source_rows")
        evaluated_rows = root_prune.get("evaluated_rows")
        if (
            _is_nonnegative_int(source_rows)
            and _is_nonnegative_int(evaluated_rows)
            and evaluated_rows >= source_rows
        ):
            issues.append("root-spec prune ON arm did not reduce evaluated rows")
    else:
        for field in (
            "plans_built",
            "applied",
            "layout_observations",
            "source_rows",
            "evaluated_rows",
            "precertified_rows",
            "all_pruned",
        ):
            if root_prune.get(field) != 0:
                issues.append(
                    f"root-spec prune OFF arm recorded pruning activity in {field}"
                )
    return issues


def _execution_treatment_issues(
    document: Any,
    expected: dict[str, Any],
    *,
    allow_unengaged_falsified_sentinel: bool = False,
    invprop_clause_rebind_required: bool | None = True,
) -> list[str]:
    issues = _execution_observations_structure_issues(document)
    if not isinstance(document, dict):
        return issues
    if document.get("run_active") is not True:
        issues.append("execution observation scope was not active at verdict emission")
    if document.get("recording_conflict") is not False:
        issues.append("execution observation recording conflict is present")

    treatment = expected["treatment"]
    if treatment == "exact_c":
        issues.extend(
            _exact_c_treatment_issues(
                document.get("exact_c"),
                expected["expected_iteration_limit"],
                require_compressed=None,
            )
        )
    elif treatment == "root_spec_prune":
        issues.extend(
            _root_spec_prune_treatment_issues(
                document.get("root_spec_prune"), expected["configured"]
            )
        )
    elif treatment == "exact_c_root_spec_prune":
        expected_limit = expected["expected_iteration_limit"]
        prune_configured = expected["prune_configured"]
        issues.extend(
            _exact_c_treatment_issues(
                document.get("exact_c"),
                expected_limit,
                require_compressed=(
                    prune_configured if expected_limit > 0 else None
                ),
            )
        )
        issues.extend(
            _root_spec_prune_treatment_issues(
                document.get("root_spec_prune"), prune_configured
            )
        )
    elif treatment == "invprop_gamma":
        invprop = document.get("invprop")
        if not isinstance(invprop, dict):
            return issues
        if invprop.get("observed") is not True:
            issues.append("INVPROP arm did not reach a runtime seam")
        if invprop.get("attribution_conflict") is not False:
            issues.append("INVPROP runtime attribution conflict is present")
        attempts = invprop.get("clause_rebind_attempts")
        accepted = invprop.get("clause_rebind_accepted")
        refused = invprop.get("clause_rebind_refused")
        if invprop_clause_rebind_required is True:
            if not _is_nonnegative_int(accepted) or accepted == 0:
                issues.append("INVPROP arm has no accepted clause rebind")
        elif invprop_clause_rebind_required is False:
            if attempts != 0 or accepted != 0 or refused != 0:
                issues.append(
                    "top-level conjunctive INVPROP arm recorded clause rebinding"
                )
        else:
            issues.append(
                "INVPROP clause-rebind requirement is unavailable from effective config"
            )
        initializations = invprop.get("alpha_initializations")
        if not _is_nonnegative_int(initializations) or initializations == 0:
            issues.append("INVPROP arm has no alpha initialization")
        attempted = invprop.get("gamma_steps_attempted")
        applied = invprop.get("gamma_steps_applied")
        nonzero_folds = invprop.get("nonzero_output_seed_folds")
        evaluated_folds = invprop.get(
            "nonzero_evaluated_output_seed_folds"
        )
        if expected["optimize_gammas"]:
            if not _is_nonnegative_int(attempted) or attempted == 0:
                issues.append("INVPROP gamma-ON arm has no attempted gamma step")
            if not _is_nonnegative_int(applied) or applied == 0:
                issues.append("INVPROP gamma-ON arm has no applied gamma step")
            if (
                _is_nonnegative_int(attempted)
                and _is_nonnegative_int(applied)
                and applied > attempted
            ):
                issues.append("INVPROP applied gamma steps exceeds attempted steps")
            if not _is_nonnegative_int(nonzero_folds) or nonzero_folds == 0:
                issues.append("INVPROP gamma-ON arm has no nonzero output-seed fold")
            if (
                not _is_nonnegative_int(evaluated_folds)
                or evaluated_folds == 0
            ):
                issues.append(
                    "INVPROP gamma-ON arm has no nonzero evaluated output-seed fold"
                )
        elif (
            attempted != 0
            or applied != 0
            or nonzero_folds != 0
            or evaluated_folds != 0
        ):
            issues.append("INVPROP gamma-OFF arm executed gamma optimization")
    elif treatment == "fresh_domain_clip":
        fresh_clip = document.get("fresh_domain_clip")
        if not isinstance(fresh_clip, dict):
            issues.append("fresh-domain clip runtime observations are absent")
            return list(dict.fromkeys(issues))
        engaged = (
            fresh_clip.get("observed") is True
            or (
                _is_nonnegative_int(fresh_clip.get("route_observations"))
                and fresh_clip["route_observations"] > 0
            )
            or (
                _is_nonnegative_int(fresh_clip.get("attempts"))
                and fresh_clip["attempts"] > 0
            )
        )
        if allow_unengaged_falsified_sentinel and not engaged:
            return list(dict.fromkeys(issues))
        configured = expected["configured"]
        if fresh_clip.get("observed") is not True:
            issues.append("fresh-domain clip arm did not observe the dispatcher route")
        if fresh_clip.get("attribution_conflict") is not False:
            issues.append("fresh-domain clip runtime attribution conflict is present")
        if fresh_clip.get("route_conflict") is not False:
            issues.append("fresh-domain clip route conflict is present")
        route_observations = fresh_clip.get("route_observations")
        if not _is_nonnegative_int(route_observations) or route_observations == 0:
            issues.append("fresh-domain clip arm has no route observation")
        if fresh_clip.get("configured") is not configured:
            issues.append(
                "fresh-domain clip configured state does not match the arm"
            )
        if fresh_clip.get("route_authorized") is not configured:
            issues.append(
                "fresh-domain clip route authorization does not match the arm"
            )
        attempts = fresh_clip.get("attempts")
        if configured:
            if not _is_nonnegative_int(attempts) or attempts == 0:
                issues.append("fresh-domain clip ON arm has no clipping attempt")
        elif (
            attempts != 0
            or fresh_clip.get("applied") != 0
            or fresh_clip.get("all_clauses_refuted") != 0
            or fresh_clip.get("skipped") != 0
            or fresh_clip.get("tightened_dimensions") != 0
        ):
            issues.append("fresh-domain clip OFF arm executed clipping work")
    else:
        issues.append(f"unknown execution-evidence treatment {treatment!r}")
    return list(dict.fromkeys(issues))


def _is_digest(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _resolve_repo_head(repo_root: Path = REPO_ROOT) -> str:
    """Resolve HEAD without inheriting Git, loader, shell, or PATH controls."""
    if GIT_EXECUTABLE is None:
        raise ManifestError("cannot resolve repository HEAD: Git is unavailable")
    try:
        git_sha256 = _sha256(GIT_EXECUTABLE)
    except OSError as error:
        raise ManifestError(
            f"cannot authenticate Git executable: {error}"
        ) from error
    control_env = {
        "PATH": os.defpath,
        "HOME": "/nonexistent/ny-factorial-git-home",
        "XDG_CONFIG_HOME": "/nonexistent/ny-factorial-git-xdg",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        process = subprocess.run(
            [str(GIT_EXECUTABLE), "rev-parse", "--verify", "HEAD"],
            cwd=repo_root,
            env=control_env,
            capture_output=True,
            text=True,
            timeout=REPO_HEAD_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ManifestError(f"cannot resolve repository HEAD: {error}") from error
    try:
        final_git_sha256 = _sha256(GIT_EXECUTABLE)
    except OSError as error:
        raise ManifestError(
            f"cannot revalidate Git executable: {error}"
        ) from error
    if final_git_sha256 != git_sha256:
        raise ManifestError("Git executable changed while resolving repository HEAD")
    head = (process.stdout or "").strip()
    if process.returncode != 0 or not _is_digest(head, 40):
        detail = (process.stderr or "").strip() or f"exit={process.returncode}"
        raise ManifestError(
            f"cannot establish one exact 40-hex repository HEAD: {detail}"
        )
    return head


def _relevant_parent_environment(process_env: dict[str, str]) -> dict[str, str]:
    """Match the bounded runner and NY flight recorder ambient-env contract."""
    return {
        name: value
        for name, value in process_env.items()
        if name.startswith("NY_") or name == "OMP_NUM_THREADS"
    }


def _valid_receipt_identity(identity: Any) -> bool:
    if not isinstance(identity, dict) or set(identity) != set(NY_RECEIPT_FIELDS):
        return False
    if any(not isinstance(identity[field], str) for field in NY_RECEIPT_FIELDS):
        return False
    if identity["schema"] != NY_RECEIPT_SCHEMA:
        return False
    if any(
        not _is_digest(identity[field])
        for field in ("binary_sha256", "source_state_sha256", "toolchain_sha256")
    ):
        return False
    if identity["source_kind"] not in {"git", "archive", "prebuilt"}:
        return False
    if not _is_digest(identity["source_commit"], 40):
        return False
    for field in ("cargo_lock_sha256", "artifact_provenance_sha256"):
        if identity[field] != "none" and not _is_digest(identity[field]):
            return False
    if identity["ay_commit"] != "none" and not _is_digest(
        identity["ay_commit"], 40
    ):
        return False
    if re.fullmatch(r"[a-z0-9]+(?:,[a-z0-9]+)*", identity["features"]) is None:
        return False
    return identity["toolchain_kind"] in {"rustc-vv", "trust-sealed"}


def _receipt_file_sha256(identity: dict[str, str]) -> str:
    raw = "".join(f"{field}={identity[field]}\n" for field in NY_RECEIPT_FIELDS)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_effective_config_evidence(output_csv: Path) -> list[dict[str, Any]]:
    """Read and integrity-check the observed treatment for every measured row."""
    if not output_csv.is_file():
        return []
    evidence: list[dict[str, Any]] = []
    try:
        with output_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ManifestError(f"cannot read bounded-runner output {output_csv}: {error}") from error

    for row_number, row in enumerate(rows, start=1):
        raw_source_index = row.get("source_index_zero_based", "")
        try:
            source_index = int(raw_source_index)
        except (TypeError, ValueError):
            source_index = None
        raw = row.get("effective_config_json", "")
        recorded_hash = row.get("effective_config_sha256", "")
        observed: dict[str, Any] | None = None
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ManifestError(
                    f"{output_csv} row {row_number} has malformed effective_config_json: {error}"
                ) from error
            if not isinstance(parsed, dict):
                raise ManifestError(
                    f"{output_csv} row {row_number} effective_config_json is not an object"
                )
            try:
                canonical = _canonical_json(parsed)
            except (TypeError, ValueError) as error:
                raise ManifestError(
                    f"{output_csv} row {row_number} has non-canonical "
                    f"effective_config_json: {error}"
                ) from error
            actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if recorded_hash != actual_hash:
                raise ManifestError(
                    f"{output_csv} row {row_number} effective-config hash mismatch: "
                    f"recorded={recorded_hash!r}, observed={actual_hash}"
                )
            observed = parsed
        elif recorded_hash:
            raise ManifestError(
                f"{output_csv} row {row_number} has an effective-config hash without JSON"
            )

        raw_execution = row.get("execution_observations_json", "") or ""
        execution_hash = row.get("execution_observations_sha256", "") or ""
        execution_observations: dict[str, Any] | None = None
        if raw_execution:
            try:
                parsed_execution = json.loads(raw_execution)
            except json.JSONDecodeError as error:
                raise ManifestError(
                    f"{output_csv} row {row_number} has malformed "
                    f"execution_observations_json: {error}"
                ) from error
            if not isinstance(parsed_execution, dict):
                raise ManifestError(
                    f"{output_csv} row {row_number} "
                    "execution_observations_json is not an object"
                )
            try:
                canonical_execution = _canonical_json(parsed_execution)
            except (TypeError, ValueError) as error:
                raise ManifestError(
                    f"{output_csv} row {row_number} has non-canonical "
                    f"execution_observations_json: {error}"
                ) from error
            if raw_execution != canonical_execution:
                raise ManifestError(
                    f"{output_csv} row {row_number} "
                    "execution_observations_json is not in canonical form"
                )
            try:
                encoded_execution = canonical_execution.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ManifestError(
                    f"{output_csv} row {row_number} has non-UTF-8 "
                    "execution_observations_json"
                ) from error
            actual_execution_hash = hashlib.sha256(encoded_execution).hexdigest()
            if execution_hash != actual_execution_hash:
                raise ManifestError(
                    f"{output_csv} row {row_number} execution-observations hash "
                    f"mismatch: recorded={execution_hash!r}, "
                    f"observed={actual_execution_hash}"
                )
            execution_observations = parsed_execution
        elif execution_hash:
            raise ManifestError(
                f"{output_csv} row {row_number} has an execution-observations "
                "hash without JSON"
            )

        raw_receipt = row.get("ny_receipt_json", "") or ""
        receipt_sha256 = row.get("ny_receipt_sha256", "") or ""
        receipt_identity: dict[str, str] | None = None
        if raw_receipt:
            try:
                parsed_receipt = json.loads(raw_receipt)
            except json.JSONDecodeError as error:
                raise ManifestError(
                    f"{output_csv} row {row_number} has malformed ny_receipt_json: "
                    f"{error}"
                ) from error
            if not _valid_receipt_identity(parsed_receipt):
                raise ManifestError(
                    f"{output_csv} row {row_number} has an invalid NY receipt identity"
                )
            assert isinstance(parsed_receipt, dict)
            actual_receipt_sha256 = _receipt_file_sha256(parsed_receipt)
            if receipt_sha256 != actual_receipt_sha256:
                raise ManifestError(
                    f"{output_csv} row {row_number} NY receipt hash mismatch: "
                    f"recorded={receipt_sha256!r}, observed={actual_receipt_sha256}"
                )
            receipt_identity = parsed_receipt
        elif receipt_sha256:
            raise ManifestError(
                f"{output_csv} row {row_number} has an NY receipt hash without JSON"
            )

        preset_sha256 = row.get("preset_sha256", "") or ""
        if preset_sha256 and not _is_digest(preset_sha256):
            raise ManifestError(
                f"{output_csv} row {row_number} has an invalid preset_sha256"
            )

        raw_parent_env = row.get("parent_env_json", "") or ""
        parent_env_sha256 = row.get("parent_env_sha256", "") or ""
        parent_env: dict[str, str] | None = None
        if raw_parent_env:
            try:
                parsed_parent_env = json.loads(raw_parent_env)
            except json.JSONDecodeError as error:
                raise ManifestError(
                    f"{output_csv} row {row_number} has malformed parent_env_json: "
                    f"{error}"
                ) from error
            if (
                not isinstance(parsed_parent_env, dict)
                or any(
                    not isinstance(name, str) or not isinstance(value, str)
                    for name, value in parsed_parent_env.items()
                )
                or any(
                    not name.startswith("NY_") and name != "OMP_NUM_THREADS"
                    for name in parsed_parent_env
                )
            ):
                raise ManifestError(
                    f"{output_csv} row {row_number} has an invalid parent environment"
                )
            canonical_parent_env = _canonical_json(parsed_parent_env)
            actual_parent_env_sha256 = hashlib.sha256(
                canonical_parent_env.encode("utf-8")
            ).hexdigest()
            if parent_env_sha256 != actual_parent_env_sha256:
                raise ManifestError(
                    f"{output_csv} row {row_number} parent-env hash mismatch: "
                    f"recorded={parent_env_sha256!r}, "
                    f"observed={actual_parent_env_sha256}"
                )
            parent_env = parsed_parent_env
        elif parent_env_sha256:
            raise ManifestError(
                f"{output_csv} row {row_number} has a parent-env hash without JSON"
            )
        evidence.append(
            {
                "model": row.get("model", ""),
                "property": row.get("property", ""),
                "source_index_zero_based": source_index,
                "result": row.get("result", ""),
                "ny_source": row.get("ny_source", ""),
                "ny_binary": row.get("ny_binary", ""),
                "ny_version": row.get("ny_version", ""),
                "ny_sha256": row.get("ny_sha256", ""),
                "ny_receipt": receipt_identity,
                "ny_receipt_sha256": receipt_sha256 or None,
                "preset_sha256": preset_sha256 or None,
                "parent_env": parent_env,
                "parent_env_sha256": parent_env_sha256 or None,
                "effective_config_sha256": recorded_hash or None,
                "effective_config": observed,
                "execution_observations_sha256": execution_hash or None,
                "execution_observations": execution_observations,
            }
        )
    return evidence


def _assess_effective_config_evidence(
    observed: list[dict[str, Any]],
    expected_bindings: Any,
    *,
    expected_treatment_checks: list[dict[str, Any]] | None = None,
    unauthenticated_treatment_fields: list[str] | None = None,
    expected_preset_sha256: str | None = None,
    expected_parent_env: dict[str, str] | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    """Compare measured rows with the immutable corpus binding, fail-closed."""
    issues: list[str] = []
    expected_treatment_checks = expected_treatment_checks or []
    unauthenticated_treatment_fields = unauthenticated_treatment_fields or []
    expected_identities: list[tuple[int, str, str]] | None = None
    if isinstance(expected_bindings, list):
        expected_identities = [
            (
                binding["source_index_zero_based"],
                Path(binding["model"]).name,
                Path(binding["property"]).name,
            )
            for binding in expected_bindings
        ]

    observed_identities = [
        (
            row.get("source_index_zero_based"),
            row.get("model", ""),
            row.get("property", ""),
        )
        for row in observed
    ]
    expected_count = len(expected_identities) if expected_identities is not None else None
    count_matches = (
        len(observed) == expected_count if expected_count is not None else None
    )
    identity_matches = (
        observed_identities == expected_identities
        if expected_identities is not None
        else None
    )
    expected_result_checks = 0
    result_mismatches: list[dict[str, Any]] = []
    if isinstance(expected_bindings, list):
        for row_index, binding in enumerate(expected_bindings):
            expected_result = binding.get("expected_result")
            if expected_result is None:
                continue
            expected_result_checks += 1
            row = observed[row_index] if row_index < len(observed) else None
            expected_identity = (
                binding["source_index_zero_based"],
                Path(binding["model"]).name,
                Path(binding["property"]).name,
            )
            observed_identity = (
                (
                    row.get("source_index_zero_based"),
                    row.get("model", ""),
                    row.get("property", ""),
                )
                if isinstance(row, dict)
                else None
            )
            observed_result = row.get("result") if isinstance(row, dict) else None
            if (
                observed_identity != expected_identity
                or observed_result != expected_result
            ):
                result_mismatches.append(
                    {
                        "row": row_index + 1,
                        "corpus_id": binding.get("corpus_id"),
                        "expected": expected_result,
                        "observed": observed_result,
                        "identity_matches": observed_identity == expected_identity,
                    }
                )
    expected_results_match = not result_mismatches
    effective_config_rows = sum(
        row.get("effective_config") is not None for row in observed
    )
    schema_rows = sum(
        isinstance(row.get("effective_config"), dict)
        and row["effective_config"].get("schema")
        == EXPECTED_EFFECTIVE_CONFIG_SCHEMA
        for row in observed
    )
    structured_rows = sum(
        isinstance(row.get("effective_config"), dict)
        and all(
            isinstance(row["effective_config"].get(section), dict)
            and bool(row["effective_config"][section])
            for section in REQUIRED_EFFECTIVE_CONFIG_SECTIONS
        )
        and REQUIRED_INVPROP_FIELDS.issubset(row["effective_config"]["invprop"])
        and isinstance(
            row["effective_config"]["root"].get(
                "atomic_root_c_margin_iterations"
            ),
            int,
        )
        and not isinstance(
            row["effective_config"]["root"][
                "atomic_root_c_margin_iterations"
            ],
            bool,
        )
        and row["effective_config"]["root"][
            "atomic_root_c_margin_iterations"
        ]
        >= 0
        and isinstance(
            row["effective_config"]["root"].get(
                "root_spec_prune_requested"
            ),
            bool,
        )
        and isinstance(
            row["effective_config"]["branching"].get(
                "input_split_adv_check"
            ),
            int,
        )
        and not isinstance(
            row["effective_config"]["branching"]["input_split_adv_check"],
            bool,
        )
        for row in observed
    )
    valid_result_rows = sum(
        row.get("result") in SUPPORTED_MEASURED_RESULTS for row in observed
    )
    valid_provenance_rows = sum(
        row.get("ny_source") in {"explicit", "shared-default"}
        and isinstance(row.get("ny_binary"), str)
        and bool(row["ny_binary"])
        and isinstance(row.get("ny_version"), str)
        and bool(row["ny_version"])
        and row["ny_version"] != "unknown"
        and isinstance(row.get("ny_sha256"), str)
        and len(row["ny_sha256"]) == 64
        and all(char in "0123456789abcdef" for char in row["ny_sha256"])
        for row in observed
    )
    provenance_identities = {
        (
            row.get("ny_source"),
            row.get("ny_binary"),
            row.get("ny_version"),
            row.get("ny_sha256"),
        )
        for row in observed
    }
    provenance_consistent = bool(observed) and len(provenance_identities) == 1

    valid_receipt_rows = 0
    promotion_feature_rows = 0
    receipt_identities: set[tuple[str, str]] = set()
    for row in observed:
        identity = row.get("ny_receipt")
        receipt_sha256 = row.get("ny_receipt_sha256")
        if (
            _valid_receipt_identity(identity)
            and isinstance(identity, dict)
            and identity["features"] == PROMOTION_RECEIPT_FEATURES
        ):
            promotion_feature_rows += 1
        if (
            _valid_receipt_identity(identity)
            and isinstance(identity, dict)
            and _is_digest(receipt_sha256)
            and receipt_sha256 == _receipt_file_sha256(identity)
            and identity["binary_sha256"] == row.get("ny_sha256")
            and identity["features"] == PROMOTION_RECEIPT_FEATURES
        ):
            valid_receipt_rows += 1
            receipt_identities.add((_canonical_json(identity), receipt_sha256))
    receipt_consistent = (
        bool(observed)
        and valid_receipt_rows == len(observed)
        and len(receipt_identities) == 1
    )
    receipt_source_commits = {
        row["ny_receipt"]["source_commit"]
        for row in observed
        if _valid_receipt_identity(row.get("ny_receipt"))
    }
    receipt_source_commit_matches_repo = (
        receipt_consistent
        and _is_digest(expected_source_commit, 40)
        and receipt_source_commits == {expected_source_commit}
    )

    valid_preset_rows = sum(_is_digest(row.get("preset_sha256")) for row in observed)
    preset_hashes = {
        row.get("preset_sha256")
        for row in observed
        if _is_digest(row.get("preset_sha256"))
    }
    preset_consistent = (
        bool(observed)
        and valid_preset_rows == len(observed)
        and len(preset_hashes) == 1
    )
    preset_matches_expected = (
        preset_consistent
        and (
            expected_preset_sha256 is None
            or preset_hashes == {expected_preset_sha256}
        )
    )

    valid_parent_env_rows = 0
    parent_env_identities: set[tuple[str, str]] = set()
    for row in observed:
        parent_env = row.get("parent_env")
        parent_env_sha256 = row.get("parent_env_sha256")
        if (
            isinstance(parent_env, dict)
            and all(
                isinstance(name, str)
                and isinstance(value, str)
                and (name.startswith("NY_") or name == "OMP_NUM_THREADS")
                for name, value in parent_env.items()
            )
            and _is_digest(parent_env_sha256)
        ):
            canonical_parent_env = _canonical_json(parent_env)
            actual_parent_env_sha256 = hashlib.sha256(
                canonical_parent_env.encode("utf-8")
            ).hexdigest()
            if parent_env_sha256 == actual_parent_env_sha256:
                valid_parent_env_rows += 1
                parent_env_identities.add(
                    (canonical_parent_env, parent_env_sha256)
                )
    parent_env_consistent = (
        bool(observed)
        and valid_parent_env_rows == len(observed)
        and len(parent_env_identities) == 1
    )
    expected_parent_env_json = (
        _canonical_json(expected_parent_env)
        if expected_parent_env is not None
        else None
    )
    parent_env_matches_expected = (
        parent_env_consistent
        and (
            expected_parent_env_json is None
            or {identity[0] for identity in parent_env_identities}
            == {expected_parent_env_json}
        )
    )

    effective_config_hashes = [
        row.get("effective_config_sha256") for row in observed
    ]
    valid_effective_config_hash_rows = sum(
        isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
        for digest in effective_config_hashes
    )
    effective_config_hash_consistent = (
        bool(observed)
        and valid_effective_config_hash_rows == len(observed)
        and len(set(effective_config_hashes)) == 1
    )
    treatment_mismatches: list[dict[str, Any]] = []
    for row_number, row in enumerate(observed, start=1):
        document = row.get("effective_config")
        for check in expected_treatment_checks:
            path = tuple(check["path"])
            actual = (
                _effective_value(document, path)
                if isinstance(document, dict)
                else None
            )
            if actual != check["expected"]:
                treatment_mismatches.append(
                    {
                        "row": row_number,
                        "source": check["source"],
                        "path": check["path"],
                        "expected": check["expected"],
                        "observed": actual,
                    }
                )
    expected_treatment_matches = bool(observed) and not treatment_mismatches

    if expected_count is not None and not count_matches:
        issues.append(
            f"expected {expected_count} measured row(s), observed {len(observed)}"
        )
    if expected_identities is not None and not identity_matches:
        issues.append(
            "observed source-index/model/property sequence differs from immutable bindings"
        )
    if not observed:
        issues.append("no measured rows were recorded")
    elif effective_config_rows != len(observed):
        issues.append(
            f"effective_config present for {effective_config_rows}/{len(observed)} row(s)"
        )
    if observed and schema_rows != len(observed):
        issues.append(
            f"effective_config schema is {EXPECTED_EFFECTIVE_CONFIG_SCHEMA!r} for "
            f"{schema_rows}/{len(observed)} row(s)"
        )
    if observed and structured_rows != len(observed):
        issues.append(
            "all required effective_config treatment sections are non-empty for "
            f"{structured_rows}/{len(observed)} row(s)"
        )
    if observed and valid_result_rows != len(observed):
        issues.append(
            f"supported measured verdict present for {valid_result_rows}/{len(observed)} row(s)"
        )
    if observed and valid_provenance_rows != len(observed):
        issues.append(
            f"complete NY binary provenance present for {valid_provenance_rows}/"
            f"{len(observed)} row(s)"
        )
    if observed and not provenance_consistent:
        issues.append("NY binary provenance differs between measured rows")
    if observed and valid_receipt_rows != len(observed):
        issues.append(
            "authenticated NY receipt/source identity present for "
            f"{valid_receipt_rows}/{len(observed)} row(s)"
        )
    elif observed and not receipt_consistent:
        issues.append("NY receipt/source identity differs between measured rows")
    if observed and not receipt_source_commit_matches_repo:
        issues.append(
            "NY receipt source_commit does not match the authenticated repository HEAD"
        )
    if observed and promotion_feature_rows != len(observed):
        issues.append(
            f"NY receipt features must be exactly {PROMOTION_RECEIPT_FEATURES!r}"
        )
    if observed and valid_preset_rows != len(observed):
        issues.append(
            f"preset SHA-256 present for {valid_preset_rows}/{len(observed)} row(s)"
        )
    elif observed and not preset_consistent:
        issues.append("preset SHA-256 differs between measured rows")
    elif observed and not preset_matches_expected:
        issues.append(
            "observed preset SHA-256 differs from the generated arm preset"
        )
    if observed and valid_parent_env_rows != len(observed):
        issues.append(
            "authenticated parent environment present for "
            f"{valid_parent_env_rows}/{len(observed)} row(s)"
        )
    elif observed and not parent_env_consistent:
        issues.append("parent environment differs between measured rows")
    elif observed and not parent_env_matches_expected:
        issues.append(
            "observed parent environment differs from the isolated arm environment"
        )
    if observed and valid_effective_config_hash_rows != len(observed):
        issues.append(
            "complete effective-config hashes present for "
            f"{valid_effective_config_hash_rows}/{len(observed)} row(s)"
        )
    elif observed and not effective_config_hash_consistent:
        issues.append("effective-config treatment hashes differ between measured rows")
    if treatment_mismatches:
        mismatch = treatment_mismatches[0]
        issues.append(
            f"effective treatment mismatch in row {mismatch['row']} for "
            f"{mismatch['source']}: expected {mismatch['expected']!r} at "
            f"{'.'.join(mismatch['path'])}, observed {mismatch['observed']!r}"
        )
    if unauthenticated_treatment_fields:
        issues.append(
            "effective treatment projection does not authenticate arm field(s): "
            + ", ".join(unauthenticated_treatment_fields)
        )
    if result_mismatches:
        mismatch = result_mismatches[0]
        issues.append(
            f"authenticated expected verdict mismatch in row {mismatch['row']} "
            f"for {mismatch['corpus_id']!r}: expected "
            f"{mismatch['expected']!r}, observed {mismatch['observed']!r}"
        )

    complete = (
        bool(observed)
        and effective_config_rows == len(observed)
        and schema_rows == len(observed)
        and structured_rows == len(observed)
        and valid_result_rows == len(observed)
        and valid_provenance_rows == len(observed)
        and provenance_consistent
        and receipt_consistent
        and receipt_source_commit_matches_repo
        and preset_matches_expected
        and parent_env_matches_expected
        and effective_config_hash_consistent
        and expected_treatment_matches
        and not unauthenticated_treatment_fields
        and expected_results_match
        and count_matches is not False
        and identity_matches is not False
    )
    return {
        "expected_effective_config_rows": expected_count,
        "observed_effective_config_rows": len(observed),
        "observed_effective_config_payload_rows": effective_config_rows,
        "observed_effective_config_schema_rows": schema_rows,
        "observed_effective_config_structured_rows": structured_rows,
        "observed_supported_result_rows": valid_result_rows,
        "observed_binary_provenance_rows": valid_provenance_rows,
        "observed_binary_provenance_consistent": provenance_consistent,
        "observed_authenticated_receipt_rows": valid_receipt_rows,
        "observed_authenticated_receipt_consistent": receipt_consistent,
        "observed_promotion_receipt_feature_rows": promotion_feature_rows,
        "observed_receipt_source_commit_matches_repo": (
            receipt_source_commit_matches_repo
        ),
        "observed_preset_sha256_rows": valid_preset_rows,
        "observed_preset_sha256_consistent": preset_consistent,
        "observed_preset_sha256_matches_expected": preset_matches_expected,
        "observed_parent_env_rows": valid_parent_env_rows,
        "observed_parent_env_consistent": parent_env_consistent,
        "observed_parent_env_matches_expected": parent_env_matches_expected,
        "observed_effective_config_hash_rows": valid_effective_config_hash_rows,
        "observed_effective_config_hash_consistent": effective_config_hash_consistent,
        "expected_treatment_checks": expected_treatment_checks,
        "observed_expected_treatment_matches": expected_treatment_matches,
        "treatment_mismatches": treatment_mismatches,
        "expected_result_checks": expected_result_checks,
        "observed_expected_results_match": expected_results_match,
        "result_mismatches": result_mismatches,
        "unauthenticated_treatment_fields": unauthenticated_treatment_fields,
        "observed_row_count_matches": count_matches,
        "observed_row_identity_matches": identity_matches,
        "observed_effective_config_complete": complete,
        "effective_config_evidence_issues": issues,
    }


def _runtime_observation_records(
    observed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project per-instance runtime evidence without implying hash equality."""
    return [
        {
            "model": row.get("model", ""),
            "property": row.get("property", ""),
            "source_index_zero_based": row.get("source_index_zero_based"),
            "execution_observations_sha256": row.get(
                "execution_observations_sha256"
            ),
            "execution_observations": row.get("execution_observations"),
        }
        for row in observed
    ]


def _authenticated_falsified_sentinel_exemption(
    row: dict[str, Any], binding: Any
) -> bool:
    """Bind a pre-dispatch exemption to one exact expected-falsified row."""
    if not isinstance(binding, dict) or binding.get("expected_result") != "falsified":
        return False
    expected_identity = (
        binding.get("source_index_zero_based"),
        Path(str(binding.get("model", ""))).name,
        Path(str(binding.get("property", ""))).name,
    )
    observed_identity = (
        row.get("source_index_zero_based"),
        row.get("model", ""),
        row.get("property", ""),
    )
    return observed_identity == expected_identity and row.get("result") == "falsified"


def _assess_execution_observation_evidence(
    observed: list[dict[str, Any]],
    expected: dict[str, Any] | None,
    expected_bindings: Any = None,
) -> dict[str, Any]:
    """Validate run-variable observations for algorithm-factorial promotion."""
    payload_rows = sum(
        isinstance(row.get("execution_observations"), dict) for row in observed
    )
    schema_rows = sum(
        isinstance(row.get("execution_observations"), dict)
        and row["execution_observations"].get("schema")
        == EXPECTED_EXECUTION_OBSERVATIONS_SCHEMA
        for row in observed
    )
    structured_rows = sum(
        not _execution_observations_structure_issues(
            row.get("execution_observations")
        )
        for row in observed
    )
    hash_rows = 0
    for row in observed:
        document = row.get("execution_observations")
        digest = row.get("execution_observations_sha256")
        if not isinstance(document, dict) or not _is_digest(digest):
            continue
        try:
            canonical = _canonical_json(document)
            encoded = canonical.encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError):
            continue
        if hashlib.sha256(encoded).hexdigest() == digest:
            hash_rows += 1
    mismatches: list[dict[str, Any]] = []
    if expected is not None:
        for row_number, row in enumerate(observed, start=1):
            binding = (
                expected_bindings[row_number - 1]
                if isinstance(expected_bindings, list)
                and row_number <= len(expected_bindings)
                else None
            )
            rebind_requirement: bool | None = True
            if expected.get("treatment") == "invprop_gamma" and isinstance(
                row.get("effective_config"), dict
            ):
                serial_rebinding = _effective_value(
                    row["effective_config"],
                    ("invprop", "serial_clause_rebinding"),
                )
                if serial_rebinding == (
                    "possible_but_unobserved_for_top_level_disjunctions"
                ):
                    rebind_requirement = True
                elif serial_rebinding == (
                    "not_applicable_for_top_level_conjunction"
                ):
                    rebind_requirement = False
                else:
                    rebind_requirement = None
            row_issues = _execution_treatment_issues(
                row.get("execution_observations"),
                expected,
                allow_unengaged_falsified_sentinel=(
                    expected.get("treatment") == "fresh_domain_clip"
                    and _authenticated_falsified_sentinel_exemption(row, binding)
                ),
                invprop_clause_rebind_required=rebind_requirement,
            )
            if row_issues:
                mismatches.append({"row": row_number, "issues": row_issues})
    treatment_rows = len(observed) - len(mismatches) if expected is not None else 0

    issues: list[str] = []
    if not observed:
        issues.append("no measured rows were recorded")
    elif payload_rows != len(observed):
        issues.append(
            "execution_observations present for "
            f"{payload_rows}/{len(observed)} row(s)"
        )
    if observed and schema_rows != len(observed):
        issues.append(
            f"execution_observations schema is "
            f"{EXPECTED_EXECUTION_OBSERVATIONS_SCHEMA!r} for "
            f"{schema_rows}/{len(observed)} row(s)"
        )
    if observed and structured_rows != len(observed):
        issues.append(
            "known execution-observation fields and invariants are valid for "
            f"{structured_rows}/{len(observed)} row(s)"
        )
    if observed and hash_rows != len(observed):
        issues.append(
            "complete execution-observation hashes present for "
            f"{hash_rows}/{len(observed)} row(s)"
        )
    if mismatches:
        first = mismatches[0]
        issues.append(
            f"runtime treatment evidence mismatch in row {first['row']}: "
            + "; ".join(first["issues"])
        )

    complete = (
        (
            bool(observed)
            and payload_rows == len(observed)
            and schema_rows == len(observed)
            and structured_rows == len(observed)
            and hash_rows == len(observed)
            and treatment_rows == len(observed)
        )
        if expected is not None
        else None
    )
    return {
        "expected_execution_evidence": expected,
        "observed_execution_observation_rows": len(observed),
        "observed_execution_observation_payload_rows": payload_rows,
        "observed_execution_observation_schema_rows": schema_rows,
        "observed_execution_observation_structured_rows": structured_rows,
        "observed_execution_observation_hash_rows": hash_rows,
        "observed_execution_treatment_rows": treatment_rows,
        "execution_observation_mismatches": mismatches,
        "observed_execution_evidence_complete": complete,
        "execution_observation_evidence_issues": issues,
    }


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    if not cleaned or cleaned in {".", ".."}:
        raise ManifestError(f"invalid artifact name: {value!r}")
    return cleaned


def _deep_set(document: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not parts or any(not part for part in parts):
        raise ManifestError(f"invalid override path: {dotted_key!r}")
    cursor: dict[str, Any] = document
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ManifestError(
                f"override {dotted_key!r} crosses non-mapping field {part!r}"
            )
        cursor = child
    cursor[parts[-1]] = value


def _normalize_root_path(document: dict[str, Any], base_preset: Path) -> None:
    """Keep a generated preset's benchmark root independent of its new location."""
    general = document.get("general")
    if not isinstance(general, dict):
        return
    raw = general.get("root_path")
    if not isinstance(raw, str):
        return
    root = Path(raw)
    if not root.is_absolute():
        general["root_path"] = str((base_preset.parent / root).resolve())


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot load manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be a mapping")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ManifestError("manifest must contain a non-empty experiments list")
    return payload


def _bind_corpus_indices(
    manifest: dict[str, Any], experiments: list[dict[str, Any]]
) -> None:
    """Resolve stable corpus IDs to unfiltered source rows and input identities."""
    raw_path = manifest.get("corpus_manifest")
    if not isinstance(raw_path, str) or not raw_path:
        return
    corpus_path = _resolve_repo_path(raw_path, "corpus_manifest")
    try:
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot load corpus manifest {corpus_path}: {error}") from error
    entries = corpus.get("entries") if isinstance(corpus, dict) else None
    if not isinstance(entries, list):
        raise ManifestError("corpus manifest must contain an entries list")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not entry["id"]
        ):
            raise ManifestError("corpus manifest contains a malformed entry")
        entry_id = entry["id"]
        if entry_id in by_id:
            raise ManifestError(f"duplicate corpus entry id: {entry_id}")
        by_id[entry_id] = entry

    for experiment in experiments:
        corpus_ids = experiment.get("corpus_ids")
        if corpus_ids is None:
            continue
        if "indices" in experiment:
            raise ManifestError(
                f"experiment {experiment['name']!r} cannot set both corpus_ids and indices"
            )
        if not isinstance(corpus_ids, list) or not corpus_ids:
            raise ManifestError(
                f"experiment {experiment['name']!r} corpus_ids must be a non-empty list"
            )
        if any(not isinstance(entry_id, str) or not entry_id for entry_id in corpus_ids):
            raise ManifestError(
                f"experiment {experiment['name']!r} corpus_ids must contain "
                "non-empty strings"
            )
        if len(set(corpus_ids)) != len(corpus_ids):
            raise ManifestError(
                f"experiment {experiment['name']!r} contains duplicate corpus_ids"
            )
        for arm in experiment.get("arms", []):
            if isinstance(arm, dict) and "indices" in arm:
                raise ManifestError(
                    f"experiment {experiment['name']!r} arm {arm.get('name')!r} cannot "
                    "override indices when corpus_ids provide immutable row bindings"
                )
        indices: list[int] = []
        row_bindings: list[dict[str, Any]] = []
        for entry_id in corpus_ids:
            if not isinstance(entry_id, str) or entry_id not in by_id:
                raise ManifestError(
                    f"experiment {experiment['name']!r} has unknown corpus id {entry_id!r}"
                )
            entry = by_id[entry_id]
            if entry.get("kind") != "vnncomp":
                raise ManifestError(f"corpus entry {entry_id!r} is not a VNN-COMP row")
            if entry.get("category") != experiment.get("category"):
                raise ManifestError(
                    f"corpus entry {entry_id!r} category does not match "
                    f"experiment {experiment['name']!r}"
                )
            source_index = entry.get("source_index")
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index < 1
            ):
                raise ManifestError(
                    f"corpus entry {entry_id!r} has invalid one-based source_index"
                )
            model = entry.get("model")
            property_name = entry.get("property")
            expected = entry.get("expected")
            if not isinstance(model, str) or not model:
                raise ManifestError(f"corpus entry {entry_id!r} has invalid model path")
            if not isinstance(property_name, str) or not property_name:
                raise ManifestError(f"corpus entry {entry_id!r} has invalid property path")
            for label, path_text, suffix in (
                ("model", model, ".onnx"),
                ("property", property_name, ".vnnlib"),
            ):
                path = Path(path_text)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.suffix != suffix
                ):
                    raise ManifestError(
                        f"corpus entry {entry_id!r} has invalid logical {label} path"
                    )
            timeout_seconds = entry.get("timeout_seconds")
            if (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, int)
                or timeout_seconds <= 0
            ):
                raise ManifestError(
                    f"corpus entry {entry_id!r} has invalid timeout_seconds"
                )
            if not isinstance(expected, dict):
                raise ManifestError(f"corpus entry {entry_id!r} has no expected hashes")
            model_sha256 = expected.get("model_sha256")
            property_sha256 = expected.get("property_sha256")
            expected_result = expected.get("expected_result")
            for label, value in (
                ("model_sha256", model_sha256),
                ("property_sha256", property_sha256),
            ):
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                ):
                    raise ManifestError(
                        f"corpus entry {entry_id!r} has invalid {label}"
                    )
            if expected_result is not None and expected_result not in {
                "verified",
                "falsified",
            }:
                raise ManifestError(
                    f"corpus entry {entry_id!r} has invalid expected_result"
                )
            indices.append(source_index - 1)
            binding = {
                "corpus_id": entry_id,
                "source_index_zero_based": source_index - 1,
                "model": model,
                "property": property_name,
                "timeout_seconds": timeout_seconds,
                "model_sha256": model_sha256,
                "property_sha256": property_sha256,
            }
            if expected_result is not None:
                binding["expected_result"] = expected_result
            row_bindings.append(binding)
        experiment["_resolved_indices"] = indices
        if len(set(indices)) != len(indices):
            raise ManifestError(
                f"experiment {experiment['name']!r} resolves multiple corpus IDs "
                "to the same source row"
            )
        experiment["_resolved_row_bindings"] = row_bindings
        experiment["_corpus_manifest"] = str(corpus_path)


def _resolve_repo_path(raw: str, field: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise ManifestError(f"{field} does not exist: {path}")
    return path


def _selected_experiments(
    manifest: dict[str, Any], selected: set[str]
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in manifest["experiments"]:
        if not isinstance(raw, dict):
            raise ManifestError("each experiment must be a mapping")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ManifestError("each experiment requires a non-empty name")
        if name in seen:
            raise ManifestError(f"duplicate experiment name: {name}")
        seen.add(name)
        if not selected or name in selected:
            experiments.append(raw)
    missing = selected - seen
    if missing:
        raise ManifestError(f"unknown experiment(s): {', '.join(sorted(missing))}")
    return experiments


def _declared_treatment_env_keys(experiments: list[dict[str, Any]]) -> set[str]:
    """Return every environment factor declared by the selected experiment set."""
    keys: set[str] = set()
    for experiment in experiments:
        arms = experiment.get("arms")
        if not isinstance(arms, list) or not arms:
            raise ManifestError(
                f"experiment {experiment.get('name')!r} requires non-empty arms"
            )
        for arm in arms:
            if not isinstance(arm, dict):
                raise ManifestError(
                    f"experiment {experiment.get('name')!r} has malformed arm"
                )
            env = arm.get("env", {})
            if not isinstance(env, dict):
                raise ManifestError(f"arm {arm.get('name')!r} env must be a mapping")
            for key in env:
                if not isinstance(key, str) or not key:
                    raise ManifestError(
                        f"arm {arm.get('name')!r} env keys must be non-empty strings"
                    )
                keys.add(key)
    return keys


def _arm_process_environment(
    base_env: dict[str, str],
    env_overrides: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Strip every inherited NY treatment, then apply this arm's overrides."""
    process_env = dict(base_env)
    scrubbed = sorted(key for key in process_env if key.startswith("NY_"))
    for key in scrubbed:
        process_env.pop(key)
    process_env.update(env_overrides)
    return process_env, scrubbed


def _materialize_arm(
    *,
    base_preset: Path,
    arm: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    try:
        document = yaml.safe_load(base_preset.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot load base preset {base_preset}: {error}") from error
    if not isinstance(document, dict):
        raise ManifestError(f"base preset must be a mapping: {base_preset}")

    overrides = arm.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ManifestError(f"arm {arm.get('name')!r} overrides must be a mapping")
    for dotted_key, value in overrides.items():
        if not isinstance(dotted_key, str):
            raise ManifestError("override keys must be strings")
        _deep_set(document, dotted_key, value)
    _normalize_root_path(document, base_preset)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "# Generated by scripts/run_abcrown_transfer_factorials.py\n"
        "# Do not edit; source and overrides are bound in execution.json.\n"
        + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return overrides


def _runner_command(
    *,
    experiment: dict[str, Any],
    arm: dict[str, Any],
    preset: Path,
    output_csv: Path,
    args: argparse.Namespace,
) -> list[str]:
    category = experiment.get("category")
    if not isinstance(category, str) or not category:
        raise ManifestError(f"experiment {experiment.get('name')!r} requires category")
    command = [
        sys.executable,
        str(BOUNDED_RUNNER),
        "--year",
        str(experiment.get("year", 2025)),
        "--category",
        category,
        "--preset",
        str(preset),
        "--output",
        str(output_csv),
        "--tag",
        str(arm["name"]),
        "--timeout-slack",
        str(args.timeout_slack),
        "--domain-batch-metrics-dir",
        str(output_csv.parent / "domain-batch-metrics"),
        "--raw-artifact-dir",
        str(output_csv.parent / "raw-attempts"),
        "--require-ny-receipt",
    ]
    indices = arm.get(
        "indices", experiment.get("_resolved_indices", experiment.get("indices"))
    )
    sample = arm.get("sample", experiment.get("sample"))
    if indices is not None:
        if isinstance(indices, list):
            indices = ",".join(str(index) for index in indices)
        command.extend(["--indices", str(indices)])
    elif sample is not None:
        command.extend(["--sample", str(sample)])
    if args.ny_binary:
        command.extend(["--ny-binary", str(Path(args.ny_binary).resolve())])
    if args.benchmark_root:
        command.extend(["--benchmark-root", str(Path(args.benchmark_root).resolve())])
    if args.max_domains is not None:
        command.extend(["--max-domains", str(args.max_domains)])
    if args.timeout_cap:
        command.extend(["--timeout-cap", str(args.timeout_cap)])
    if args.warmup_runs:
        command.extend(["--warmup-runs", str(args.warmup_runs)])
    if args.rerun_presearch:
        command.extend(["--rerun-presearch", str(args.rerun_presearch)])
    extra_args = arm.get("extra_args", [])
    if not isinstance(extra_args, list) or any(
        not isinstance(extra_arg, str) for extra_arg in extra_args
    ):
        raise ManifestError(f"arm {arm.get('name')!r} extra_args must be a string list")
    for extra_arg in extra_args:
        flag = extra_arg.split("=", 1)[0]
        if flag in HARNESS_OWNED_EXTRA_FLAGS or (
            flag.startswith("-p") and not flag.startswith("--")
        ):
            raise ManifestError(
                f"arm {arm.get('name')!r} extra_args cannot override "
                f"harness-owned flag {flag!r}"
            )
        # The value itself commonly starts with `--`; use argparse's attached
        # form so it cannot be reinterpreted as an option to this driver.
        command.append(f"--extra-arg={extra_arg}")
    for binding in experiment.get("_resolved_row_bindings", []):
        command.extend(
            [
                "--expected-row-binding",
                json.dumps(binding, sort_keys=True, separators=(",", ":")),
            ]
        )
    return command


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Run one named experiment (repeatable; default: all).",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help=(
            "Run only an arm with this name (repeatable). Combine with "
            "--experiment when arm names are experiment-specific."
        ),
    )
    parser.add_argument("--ny-binary")
    parser.add_argument(
        "--benchmark-root",
        help="Explicit VNN-COMP benchmark root containing category directories.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout-slack", type=int, default=5)
    parser.add_argument(
        "--timeout-cap",
        type=int,
        default=0,
        help=(
            "Cap official row timeouts for explicitly non-promotional pilot runs; "
            "0 keeps official budgets."
        ),
    )
    parser.add_argument("--max-domains", type=int)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--rerun-presearch", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize/hash arms and print commands without executing NY.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if (
        args.timeout_slack < 0
        or args.timeout_cap < 0
        or args.warmup_runs < 0
        or args.rerun_presearch < 0
    ):
        raise ManifestError("timeout/warmup/rerun values must be non-negative")
    manifest_path = _resolve_repo_path(args.manifest, "manifest")
    manifest = _load_manifest(manifest_path)
    all_experiments = _selected_experiments(manifest, set())
    experiments = _selected_experiments(manifest, set(args.experiment))
    # Validate every declared env key even for an unselected experiment. Arm
    # isolation below is stronger still: it strips every inherited NY_* key.
    _declared_treatment_env_keys(all_experiments)
    _bind_corpus_indices(manifest, experiments)
    repo_head = _resolve_repo_head(REPO_ROOT)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    execution: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "repo_head": repo_head,
        "dry_run": args.dry_run,
        "timeout_cap_seconds": args.timeout_cap or None,
        "benchmark_root": (
            str(Path(args.benchmark_root).resolve()) if args.benchmark_root else None
        ),
        "arms": [],
    }
    failed = False
    requested_arms = set(args.arm)
    seen_requested_arms: set[str] = set()
    for experiment in experiments:
        experiment_name = _safe_name(str(experiment["name"]))
        base_preset = _resolve_repo_path(
            str(experiment.get("base_preset", "")),
            f"{experiment_name}.base_preset",
        )
        arms = experiment.get("arms")
        if not isinstance(arms, list) or not arms:
            raise ManifestError(f"experiment {experiment_name} requires non-empty arms")
        for arm in arms:
            if not isinstance(arm, dict) or not isinstance(arm.get("name"), str):
                raise ManifestError(f"experiment {experiment_name} has malformed arm")
            if requested_arms and arm["name"] not in requested_arms:
                continue
            seen_requested_arms.add(arm["name"])
            arm_name = _safe_name(arm["name"])
            arm_dir = output_root / experiment_name / arm_name
            preset_path = arm_dir / "preset.yaml"
            overrides = _materialize_arm(
                base_preset=base_preset,
                arm=arm,
                destination=preset_path,
            )
            output_csv = arm_dir / "results.csv"
            if os.path.lexists(output_csv):
                raise ManifestError(
                    f"refusing pre-existing arm output CSV: {output_csv}"
                )
            command = _runner_command(
                experiment=experiment,
                arm=arm,
                preset=preset_path,
                output_csv=output_csv,
                args=args,
            )
            env_overrides = arm.get("env", {})
            if not isinstance(env_overrides, dict):
                raise ManifestError(f"arm {arm_name} env must be a mapping")
            env_overrides = {str(key): str(value) for key, value in env_overrides.items()}
            process_env, scrubbed_inherited_env = _arm_process_environment(
                dict(os.environ), env_overrides
            )
            generated_preset_sha256 = _sha256(preset_path)
            expected_parent_env = _relevant_parent_environment(process_env)
            (
                expected_treatment_checks,
                unauthenticated_treatment_fields,
            ) = _expected_treatment_authentication(arm)
            expected_execution_evidence = _expected_execution_evidence(arm)
            record: dict[str, Any] = {
                "experiment": experiment_name,
                "arm": arm_name,
                "base_preset": str(base_preset),
                "base_preset_sha256": _sha256(base_preset),
                "generated_preset": str(preset_path),
                "generated_preset_sha256": generated_preset_sha256,
                "expected_parent_env": expected_parent_env,
                "overrides": overrides,
                "env": env_overrides,
                "scrubbed_inherited_treatment_env_keys": scrubbed_inherited_env,
                "command": command,
                "output_csv": str(output_csv),
                "domain_batch_metrics_dir": str(arm_dir / "domain-batch-metrics"),
                "raw_artifact_dir": str(arm_dir / "raw-attempts"),
                "disposition": arm.get("disposition", "measure"),
                "corpus_ids": experiment.get("corpus_ids"),
                "resolved_zero_based_indices": experiment.get("_resolved_indices"),
                "resolved_index_semantics": (
                    "zero_based_unfiltered_instances_csv_data_rows"
                ),
                "resolved_row_bindings": experiment.get("_resolved_row_bindings"),
                "observed_effective_configs": [],
                "expected_effective_config_rows": (
                    len(experiment["_resolved_row_bindings"])
                    if isinstance(experiment.get("_resolved_row_bindings"), list)
                    else None
                ),
                "observed_effective_config_rows": 0,
                "observed_effective_config_payload_rows": 0,
                "observed_effective_config_schema_rows": 0,
                "observed_effective_config_structured_rows": 0,
                "observed_supported_result_rows": 0,
                "observed_binary_provenance_rows": 0,
                "observed_binary_provenance_consistent": None,
                "observed_authenticated_receipt_rows": 0,
                "observed_authenticated_receipt_consistent": None,
                "observed_promotion_receipt_feature_rows": 0,
                "observed_receipt_source_commit_matches_repo": None,
                "observed_preset_sha256_rows": 0,
                "observed_preset_sha256_consistent": None,
                "observed_preset_sha256_matches_expected": None,
                "observed_parent_env_rows": 0,
                "observed_parent_env_consistent": None,
                "observed_parent_env_matches_expected": None,
                "observed_effective_config_hash_rows": 0,
                "observed_effective_config_hash_consistent": None,
                "expected_treatment_checks": expected_treatment_checks,
                "observed_expected_treatment_matches": None,
                "treatment_mismatches": [],
                "expected_result_checks": 0,
                "observed_expected_results_match": None,
                "result_mismatches": [],
                "unauthenticated_treatment_fields": (
                    unauthenticated_treatment_fields
                ),
                "observed_row_count_matches": None,
                "observed_row_identity_matches": None,
                "observed_effective_config_complete": None,
                "effective_config_evidence_issues": [],
                "observed_execution_observations": [],
                "expected_execution_evidence": expected_execution_evidence,
                "observed_execution_observation_rows": 0,
                "observed_execution_observation_payload_rows": 0,
                "observed_execution_observation_schema_rows": 0,
                "observed_execution_observation_structured_rows": 0,
                "observed_execution_observation_hash_rows": 0,
                "observed_execution_treatment_rows": 0,
                "execution_observation_mismatches": [],
                "observed_execution_evidence_complete": None,
                "execution_observation_evidence_issues": [],
                "observed_promotion_evidence_complete": None,
            }
            print(" ".join(command))
            if not args.dry_run:
                started = time.monotonic()
                process = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=process_env,
                    check=False,
                )
                record["returncode"] = process.returncode
                record["elapsed_seconds"] = round(time.monotonic() - started, 6)
                try:
                    observed = _read_effective_config_evidence(output_csv)
                    assessment = _assess_effective_config_evidence(
                        observed,
                        experiment.get("_resolved_row_bindings"),
                        expected_treatment_checks=expected_treatment_checks,
                        unauthenticated_treatment_fields=(
                            unauthenticated_treatment_fields
                        ),
                        expected_preset_sha256=generated_preset_sha256,
                        expected_parent_env=expected_parent_env,
                        expected_source_commit=repo_head,
                    )
                    execution_assessment = (
                        _assess_execution_observation_evidence(
                            observed,
                            expected_execution_evidence,
                            experiment.get("_resolved_row_bindings"),
                        )
                    )
                    record["observed_effective_configs"] = observed
                    record["observed_execution_observations"] = (
                        _runtime_observation_records(observed)
                    )
                    record.update(assessment)
                    record.update(execution_assessment)
                    execution_complete = record[
                        "observed_execution_evidence_complete"
                    ]
                    record["observed_promotion_evidence_complete"] = bool(
                        record["observed_effective_config_complete"]
                        and (
                            expected_execution_evidence is None
                            or execution_complete is True
                        )
                    )
                except ManifestError as error:
                    record["observed_effective_config_complete"] = False
                    record["effective_config_evidence_issues"] = [str(error)]
                    if expected_execution_evidence is not None:
                        record["observed_execution_evidence_complete"] = False
                        record["execution_observation_evidence_issues"] = [
                            str(error)
                        ]
                    record["observed_promotion_evidence_complete"] = False
                evidence_failed = not record[
                    "observed_promotion_evidence_complete"
                ]
                if process.returncode == 0 and evidence_failed:
                    record["evidence_failure"] = (
                        "bounded runner exited successfully without complete, "
                        "identity-bound static and required runtime treatment evidence"
                    )
                failed |= process.returncode != 0 or evidence_failed
            execution["arms"].append(record)

    missing_arms = requested_arms - seen_requested_arms
    if missing_arms:
        raise ManifestError(f"unknown selected arm(s): {', '.join(sorted(missing_arms))}")

    execution_path = output_root / "execution.json"
    execution_path.write_text(
        json.dumps(execution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"execution record: {execution_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
