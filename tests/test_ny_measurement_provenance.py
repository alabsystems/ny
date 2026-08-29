# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ny_measurement_provenance.py"
ARCHIVE_SCRIPT = REPO_ROOT / "scripts" / "archive_vnncomp_sat_result.py"
AY_REV = "1560972ade2b04a702dfbd13a2de5444ea216009"
HOST_STATE_FIXTURE = {
    "schema": "ny_measurement_host_state_v1",
    "captured_at_utc": "2026-07-18T00:00:00.000000Z",
    "load_average": {},
    "processes": {},
    "gpu": {},
}
CONTAINMENT_FIXTURE = {
    "schema": "ny_measurement_containment_v1",
    "fixture": True,
}
CUDA_RUNTIME_FIXTURE = {
    "schema": "ny_measurement_cuda_runtime_v1",
    "status": "not_required",
    "reason": "cuda_build_feature_not_declared",
}
CGAN_GENERATOR_BRANCH_BOOLEAN_ENV = (
    "NY_BRANCH_STEM",
    "NY_BRANCH_STEM_PROBE",
    "NY_BRANCH_TRACE",
    "NY_MO_BAB_TRACE",
    "NY_UNSTABLE_COUNT",
)
CGAN_GENERATOR_BRANCH_ENV = (
    *CGAN_GENERATOR_BRANCH_BOOLEAN_ENV,
    "NY_BRANCH_STEM_K",
    "NY_BRANCH_STEM_NODES",
)
CURRENT_EXECUTION_PREDICATE = (
    "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
    "NY_PGD_FAST_GRAD.0=ran"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("ny_measurement_provenance", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


provenance = _load_module()
REAL_CAPTURE_MEASUREMENT_CONTAINMENT = provenance._capture_measurement_containment
REAL_CAPTURE_CUDA_RUNTIME_DEPENDENCY = (
    provenance._capture_cuda_runtime_dependency
)
REAL_CAPTURE_AND_SEAL_CUDA_RUNTIME_DEPENDENCY = (
    provenance._capture_and_seal_cuda_runtime_dependency
)

CURRENT_PREDICATE_AXIS_TOKENS = {
    "NY_BNN_ATTACK_CAP": "150",
    "NY_BNN_ATTACK_FRAC": "0.5",
    "NY_BNN_SIGN_SPACE": "0",
    "NY_BNN_SIGN_SPACE_MINIMAL_MOVE": "1",
    "NY_BNN_SIGN_SPACE_TRUST_REGION": "linf",
    "NY_BNN_STE_PGD": "0",
    "NY_BN_GPU_EXTRACT": "1",
    "NY_EPOCH_ATTEMPTS": "4",
    "NY_EPOCH_DEPTH": "3",
    "NY_EPOCH_RETAIN": "256",
    "NY_EPOCH_TIER0": "8",
    "NY_EPOCH_UNIVERSE": "128",
    "NY_MARGIN_ROW_PARALLEL": "4",
    "NY_MARGIN_ROW_ALPHA_COLS": "8",
    "NY_MARGIN_ROW_ALPHA_GUARD": "1",
    "NY_MARGIN_ROW_ALPHA_ITERS": "16",
    "NY_MARGIN_ROW_ALPHA_LR": "0.1",
    "NY_MARGIN_ROW_ALPHA_OPT": "1",
    "NY_MARGIN_ROW_ALPHA_SECS": "20",
    "NY_MARGIN_ROW_BACKWARD_INTERM": "1",
    "NY_MARGIN_ROW_BETA_PERCOL": "1",
    "NY_MARGIN_ROW_BETA_COLS": "4",
    "NY_MARGIN_ROW_BI_CHUNK": "64",
    "NY_MARGIN_ROW_BI_CHUNK_SECS": "2",
    "NY_MARGIN_ROW_BI_SECS": "30",
    "NY_MARGIN_ROW_BI_TOPK": "4096",
    "NY_MARGIN_ROW_COL_RETIRE": "1",
    "NY_MARGIN_ROW_CONCURRENT": "1",
    "NY_MARGIN_ROW_DOMAIN_ALPHA": "1",
    "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA": "0.5",
    "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS": "4",
    "NY_MARGIN_ROW_K_HEAD": "4",
    "NY_MARGIN_ROW_K_TRUNK": "16",
    "NY_MARGIN_ROW_K_ADAPT": "1",
    "NY_MARGIN_ROW_PARALLEL_PREBUILD": "1",
    "NY_MARGIN_ROW_ROOT_F32": "1",
    "NY_MARGIN_ROW_REBAKE": "1",
    "NY_MARGIN_ROW_REBAKE_ROUNDS": "2",
    "NY_MARGIN_ROW_REBAKE_SECS": "30",
    "NY_MARGIN_ROW_TABLEAU_F32": "1",
    "NY_MO_ALPHA_STEP": "1",
    "NY_MO_KFSB": "1",
    "NY_MO_KFSB_REDUCE": "max",
    "NY_PGD_EXACT_BATCHED": "1",
    "NY_PGD_EXPLOIT": "1",
    "NY_PGD_FAST_GRAD": "0",
    "NY_PGD_HINGE_TAU": "0.001",
    "NY_PGD_LANE_WINDOW": "100",
    "NY_PGD_LEGACY_LANES": "20",
    "NY_PGD_ODI_ETA": "0.5",
    "NY_PGD_ODI_STEPS": "8",
    "NY_PGD_VJP_BATCH": "1",
    "NY_PGD_VJP_K": "52",
    "NY_PGD_VJP_STEPS": "250",
    "NY_POSTBAB_RESERVE_SECS": "100",
    "NY_ROOT_JOINT_INTERM_ALPHA": "1",
    "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT": "1",
    "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP": "1",
    "NY_ROOT_JOINT_INTERM_ALPHA_ITERS": "12",
    "NY_ROOT_JOINT_INTERM_ALPHA_LR": "0.1",
    "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL": "8",
    "NY_ROOT_JOINT_INTERM_ALPHA_PURE_CHAIN": "1",
    "NY_ROOT_JOINT_INTERM_ALPHA_SECS": "20",
    "NY_ROOT_JOINT_MIN_REMAINING_SECS": "120",
    "NY_UPFRONT_ATTACK": "1",
    "NY_UPFRONT_ATTACK_AUTO_CAP": "4",
    "NY_UPFRONT_ATTACK_CAP": "8",
    "NY_UPFRONT_ATTACK_FRAC": "0.08",
    "NY_UPFRONT_ATTACK_PROBE_FIRST": "1",
}


@pytest.fixture(autouse=True)
def _stub_measurement_containment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NY_MEASURE_EXPECTED_CPUS", raising=False)
    monkeypatch.setattr(
        provenance,
        "_capture_measurement_containment",
        lambda: CONTAINMENT_FIXTURE,
    )
    monkeypatch.setattr(
        provenance,
        "_capture_cuda_runtime_dependency",
        lambda _binary, _environment, _features: CUDA_RUNTIME_FIXTURE,
    )
    monkeypatch.setattr(
        provenance,
        "_capture_and_seal_cuda_runtime_dependency",
        lambda _binary, _environment, _features, _run_dir: CUDA_RUNTIME_FIXTURE,
    )
    monkeypatch.setattr(
        provenance,
        "_recapture_cuda_runtime_from_start",
        lambda start: start["dependencies"]["cuda_runtime"],
    )


def _load_archive_module():
    spec = importlib.util.spec_from_file_location(
        "archive_vnncomp_sat_result_for_provenance_tests", ARCHIVE_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


archive = _load_archive_module()
import regular_bank_evidence as regular_evidence  # noqa: E402


# A stand-in for the real `ny` binary that can answer `--build-id`
# (#binary-identity). Provenance now refuses a solver that cannot name its own
# build, and cross-checks the reported `exe=` against the bytes it sealed — so
# the fixture hashes ITSELF at run time rather than embedding a digest that
# would have to be a fixed point of its own contents.
_FIXTURE_BUILD_STAMP = "2" * 64
_FIXTURE_BUILD_ID_FIELDS = (
    "version=0.3.4745 features=mip,cuda backend=cuda+wgpu"
    " markers=margin-row-gpu-eft-impl-v9"
    " commit=1111111111111111111111111111111111111111 dirty=false revision=4745"
    " source=clean profile=release target=x86_64-unknown-linux-gnu rustc=1.95.0"
    f" stamp={_FIXTURE_BUILD_STAMP}"
)
_FIXTURE_SEARCH_SPACE_PAYLOAD = {
    "schema": "ny-levers/search-space/v6",
    "axes": [
        {
            "name": "NY_PGD_FAST_GRAD",
            "domain": "bool",
            "values": ["0", "1"],
            "class": "VerdictAffecting",
            "deliver": "preset:attack.pgd_fast_grad",
            "why": "fixture execution predicate axis",
            "absence_baseline": {"kind": "declaration_default"},
            "category_prefixes": [],
            "preset_baseline_keys": ["attack.pgd_fast_grad"],
        }
    ],
    "edges": [],
    "unsafe": [],
    "instrument_only": [],
    "test_only": [],
    "declarations": {
        "NY_PGD_FAST_GRAD": {
            "kind": "Bool",
            "default": "1",
            "min": None,
            "max": None,
            "members": None,
        }
    },
}
def _fixture_axis(
    name: str,
    *,
    domain: str,
    values: list[str],
    kind: str,
    default: str,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "name": name,
            "domain": domain,
            "values": values,
            "class": "VerdictAffecting",
            "deliver": f"preset:fixture.{name.lower()}",
            "why": "closed predicate-assignment fixture",
            "absence_baseline": {"kind": "declaration_default"},
            "category_prefixes": [],
            "preset_baseline_keys": [f"fixture.{name.lower()}"],
        },
        {
            "kind": kind,
            "default": default,
            "min": minimum,
            "max": maximum,
            "members": None,
        },
    )


def _root_predicate_search_space_payload() -> dict[str, object]:
    definitions = (
        _fixture_axis(
            "NY_ROOT_JOINT_INTERM_ALPHA",
            domain="bool",
            values=["0", "1"],
            kind="Bool",
            default="0",
        ),
        _fixture_axis(
            "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
            domain="bool",
            values=["0", "1"],
            kind="Bool",
            default="0",
        ),
        _fixture_axis(
            "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL",
            domain="grid",
            values=["1", "4", "8"],
            kind="UsizeTrimmed",
            default="512",
        ),
        _fixture_axis(
            "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP",
            domain="bool",
            values=["0", "1"],
            kind="Bool",
            default="0",
        ),
        _fixture_axis(
            "NY_BN_GPU_EXTRACT",
            domain="bool",
            values=["0", "1"],
            kind="Bool",
            default="0",
        ),
        _fixture_axis(
            "NY_CPU_COMPUTE_ENGINE",
            domain="bool",
            values=["0", "1"],
            kind="Bool",
            default="1",
        ),
        _fixture_axis(
            "NY_ROOT_JOINT_INTERM_ALPHA_LR",
            domain="enum",
            values=["0.01", "0.05", "0.1", "0.2"],
            kind="F64ClosedTrimmed",
            default="0.1",
            minimum=-3.402_823_466_385_288_6e38,
            maximum=3.402_823_466_385_288_6e38,
        ),
        _fixture_axis(
            "NY_ROOT_JOINT_INTERM_ALPHA_ITERS",
            domain="grid",
            values=["1", "12", "50", "100"],
            kind="UsizeTrimmed",
            default="100",
        ),
        _fixture_axis(
            "NY_ROOT_JOINT_INTERM_ALPHA_SECS",
            domain="grid",
            values=["8", "20", "30", "60"],
            kind="U64Trimmed",
            default="30",
        ),
    )
    return {
        "schema": "ny-levers/search-space/v6",
        "axes": [axis for axis, _declaration in definitions],
        "edges": [],
        "unsafe": [],
        "instrument_only": [],
        "test_only": [],
        "declarations": {
            axis["name"]: declaration for axis, declaration in definitions
        },
    }


ROOT_LR_REQUEST = "0.05"
ROOT_LR_PREREQUISITES = {
    "NY_BN_GPU_EXTRACT": "1",
    "NY_CPU_COMPUTE_ENGINE": "0",
    "NY_ROOT_JOINT_INTERM_ALPHA": "1",
    "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT": "1",
    "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP": "1",
    "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL": "8",
}
ROOT_JOINT_REQUESTED = [
    ["NY_ROOT_JOINT_INTERM_ALPHA_ITERS", "12"],
    ["NY_ROOT_JOINT_INTERM_ALPHA_LR", ROOT_LR_REQUEST],
]
ROOT_JOINT_PREDICATE = (
    "ny-exec-v2.root-joint-interm-alpha-parameter-engaged."
    "NY_ROOT_JOINT_INTERM_ALPHA_ITERS.12,"
    f"NY_ROOT_JOINT_INTERM_ALPHA_LR.{ROOT_LR_REQUEST}=ran"
)
MARGIN_ROW_POLICY_JOINT_PREDICATE = (
    "ny-exec-v2.margin-row-policy-parameter-engaged."
    "NY_MARGIN_ROW_BETA_COLS.4,NY_MARGIN_ROW_BETA_PERCOL.1=ran"
)
MARGIN_ROW_POLICY_ALL_JOINT_PREDICATE = (
    "ny-exec-v2.margin-row-policy-parameter-engaged."
    "NY_MARGIN_ROW_BETA_COLS.4,NY_MARGIN_ROW_BETA_PERCOL.1,"
    "NY_MARGIN_ROW_COL_RETIRE.1,NY_MARGIN_ROW_PARALLEL_PREBUILD.1=ran"
)
MARGIN_ROW_BACKWARD_INTERM_ALL_JOINT_PREDICATE = (
    "ny-exec-v2.margin-row-backward-interm-parameter-engaged."
    "NY_MARGIN_ROW_BACKWARD_INTERM.1,NY_MARGIN_ROW_BI_CHUNK.64,"
    "NY_MARGIN_ROW_BI_CHUNK_SECS.2,NY_MARGIN_ROW_BI_SECS.30,"
    "NY_MARGIN_ROW_BI_TOPK.4096=ran"
)
MARGIN_ROW_ROOT_PRECISION_JOINT_PREDICATE = (
    "ny-exec-v2.margin-row-root-precision-parameter-engaged."
    "NY_MARGIN_ROW_ROOT_F32.1,NY_MARGIN_ROW_TABLEAU_F32.1=ran"
)
MARGIN_ROW_EPOCH_ALL_JOINT_PREDICATE = (
    "ny-exec-v2.margin-row-epoch-parameter-engaged."
    "NY_EPOCH_ATTEMPTS.4,NY_EPOCH_DEPTH.3,NY_EPOCH_RETAIN.256,"
    "NY_EPOCH_TIER0.8,NY_EPOCH_UNIVERSE.128=ran"
)
MARGIN_ROW_DOMAIN_ALPHA_ALL_JOINT_PREDICATE = (
    "ny-exec-v2.margin-row-domain-alpha-parameter-engaged."
    "NY_MARGIN_ROW_DOMAIN_ALPHA.1,NY_MARGIN_ROW_DOMAIN_ALPHA_ETA.0.5,"
    "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS.4=ran"
)
MARGIN_ROW_WIDTH_JOINT_PREDICATE = (
    "ny-exec-v2.margin-row-width-parameter-engaged."
    "NY_MARGIN_ROW_K_HEAD.4,NY_MARGIN_ROW_K_TRUNK.16=ran"
)
MARGIN_ROW_SCHEDULE_PREDICATE = (
    "ny-exec-v1.margin-row-schedule-parameter-engaged."
    "NY_MARGIN_ROW_CONCURRENT.1=ran"
)


def _margin_row_width_event(**overrides: str) -> dict[str, object]:
    fields = {
        "schema": "ny-margin-row-width-execution/v2",
        "phase": "terminal",
        "outcome": "unknown",
        "configured_k_head": "4",
        "contextual_k_head": "8",
        "configured_k_trunk": "16",
        "contextual_k_trunk": "8",
        "adaptive_enabled": "false",
        "contextual_adaptive": "false",
        "tier0_exact": "0",
        "effective_k_head_max": "4",
        "effective_k_trunk_max": "16",
        "effective_contextual_k_head_max": "8",
        "effective_contextual_k_trunk_max": "8",
        "outer_head_selection_calls": "3",
        "outer_head_available_max": "8",
        "outer_head_selected_total": "12",
        "outer_head_binding_scored_calls": "2",
        "outer_trunk_selection_calls": "2",
        "outer_trunk_available_max": "20",
        "outer_trunk_selected_total": "32",
        "outer_trunk_binding_scored_calls": "1",
        "nested_head_selection_calls": "7",
        "nested_trunk_selection_calls": "9",
        "configured_adaptive_threshold_opportunities": "0",
        "configured_adaptive_escalations": "0",
        "contextual_adaptive_threshold_opportunities": "0",
        "contextual_adaptive_escalations": "0",
        "policy_mismatch": "false",
    }
    fields.update(overrides)
    return {
        "method": "margin_row_width_execution",
        "status": "ran",
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _margin_row_schedule_event(
    *, status: str = "ran", **overrides: str
) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/margin-row-schedule-execution/v2",
        "phase": "allocation",
        "outer_route": "actionable",
        "configured_reserve_secs": "0",
        "contextual_reserve_secs": "45",
        "configured_adaptive_reserve": "false",
        "contextual_adaptive_reserve": "false",
        "configured_reserve_max_frac_bits": "none",
        "contextual_reserve_max_frac_bits": "none",
        "configured_release_frac_bits": str(
            struct.unpack(">Q", struct.pack(">d", 0.40))[0]
        ),
        "contextual_release_frac_bits": str(
            struct.unpack(">Q", struct.pack(">d", 0.40))[0]
        ),
        "configured_concurrent": "true",
        "contextual_concurrent": "false",
        "configured_early_exit": "true",
        "contextual_early_exit": "true",
        "configured_verifier_threads": "none",
        "contextual_verifier_threads": "none",
        "configured_concurrent_threads": "none",
        "contextual_concurrent_threads": "none",
        "host_parallelism": "10",
        "global_pool_route": "not_requested",
        "global_pool_requested_threads": "none",
        "global_pool_actual_threads": "none",
        "lane_pool_route": "shared_global",
        "lane_pool_threads": "0",
        "private_pool_route": "not_requested",
        "concurrent_route": "spawned",
        "concurrent_spawned": "true",
        "internal_before_secs": "95",
        "reserve_route": "not_applicable",
        "uncapped_reserve_secs": "0",
        "capped_reserve_secs": "0",
        "held_reserve_secs": "0",
        "internal_after_secs": "95",
        "sparse_root_requested": "false",
        "policy_mismatch": "false",
    }
    fields.update(overrides)
    return {
        "method": "margin_row_schedule_execution",
        "status": status,
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _margin_row_concurrent_overlap_event(
    *, status: str = "ran", **overrides: str
) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/margin-row-concurrent-overlap-execution/v1",
        "phase": "work",
        "worker_active": "true",
        "verifier_active": "true",
    }
    fields.update(overrides)
    return {
        "method": "margin_row_concurrent_overlap_execution",
        "status": status,
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.02,
    }


def _margin_row_k_adapt_on_event(**overrides: str) -> dict[str, object]:
    fields = {
        "configured_k_head": "8",
        "contextual_k_head": "8",
        "configured_k_trunk": "8",
        "contextual_k_trunk": "8",
        "adaptive_enabled": "true",
        "contextual_adaptive": "false",
        "effective_k_head_max": "16",
        "effective_k_trunk_max": "16",
        "effective_contextual_k_head_max": "8",
        "effective_contextual_k_trunk_max": "8",
        "outer_head_selection_calls": "2",
        "outer_head_available_max": "20",
        "outer_head_selected_total": "32",
        "outer_head_binding_scored_calls": "2",
        "outer_trunk_selection_calls": "2",
        "outer_trunk_available_max": "20",
        "outer_trunk_selected_total": "32",
        "outer_trunk_binding_scored_calls": "2",
        "configured_adaptive_threshold_opportunities": "2",
        "configured_adaptive_escalations": "1",
        "contextual_adaptive_threshold_opportunities": "0",
        "contextual_adaptive_escalations": "0",
    }
    fields.update(overrides)
    return _margin_row_width_event(**fields)


def _margin_row_k_adapt_off_event(**overrides: str) -> dict[str, object]:
    fields = {
        "configured_k_head": "8",
        "contextual_k_head": "8",
        "configured_k_trunk": "8",
        "contextual_k_trunk": "8",
        "adaptive_enabled": "false",
        "contextual_adaptive": "true",
        "effective_k_head_max": "8",
        "effective_k_trunk_max": "8",
        "effective_contextual_k_head_max": "16",
        "effective_contextual_k_trunk_max": "16",
        "outer_head_selection_calls": "2",
        "outer_head_available_max": "20",
        "outer_head_selected_total": "16",
        "outer_head_binding_scored_calls": "2",
        "outer_trunk_selection_calls": "2",
        "outer_trunk_available_max": "20",
        "outer_trunk_selected_total": "16",
        "outer_trunk_binding_scored_calls": "2",
        "configured_adaptive_threshold_opportunities": "0",
        "configured_adaptive_escalations": "0",
        "contextual_adaptive_threshold_opportunities": "2",
        "contextual_adaptive_escalations": "1",
    }
    fields.update(overrides)
    return _margin_row_width_event(**fields)


def _margin_row_policy_event(**overrides: str) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/margin-row-policy-execution/v1",
        "phase": "terminal",
        "outcome": "unknown",
        "beta_per_column": "true",
        "beta_columns": "4",
        "column_retire": "true",
        "parallel_prebuild": "true",
        "beta_per_column_evals": "3",
        "beta_columns_binding_evals": "2",
        "beta_columns_max_candidates": "7",
        "newly_retired_columns": "5",
        "inherited_retirement_evals": "4",
        "parallel_prebuild_batches": "2",
        "parallel_prebuild_domains": "4",
        "policy_mismatch": "false",
    }
    fields.update(overrides)
    return {
        "method": "margin_row_policy_execution",
        "status": "ran",
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _margin_row_backward_interm_event(
    **overrides: str,
) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/margin-row-backward-interm-execution/v1",
        "phase": "terminal",
        "outcome": "exhausted",
        "enabled": "true",
        "configured_secs": "30",
        "configured_chunk": "64",
        "configured_chunk_secs": "2",
        "effective_chunk_secs": "2",
        "configured_topk": "4096",
        "eligible_layers": "4",
        "suppressed_setup_layers": "0",
        "layers_started": "3",
        "chunks_completed": "6",
        "columns_completed": "512",
        "deadline_capped": "false",
        "budget_exhausted": "true",
        "chunk_secs_capped": "false",
        "chunk_counterfactual_chunks": "2",
        "chunk_ceiling_binding_chunks": "2",
        "fixed_width_chunks": "0",
        "adaptive_width_changes": "2",
        "topk_binding_layers": "2",
        "policy_mismatch": "false",
    }
    fields.update(overrides)
    return {
        "method": "margin_row_backward_interm_execution",
        "status": "ran",
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _margin_row_root_precision_event(
    **overrides: str,
) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/margin-row-root-precision-execution/v1",
        "phase": "terminal",
        "outcome": "completed",
        "configured_root_f32": "true",
        "configured_tableau_f32": "true",
        "outward_eligible": "true",
        "eligible_conv_ops": "4",
        "root_f32_conv_ops": "4",
        "root_f64_conv_ops": "0",
        "eligible_store_ops": "9",
        "tableau_f32_store_ops": "9",
        "tableau_f64_store_ops": "0",
        "tableau_refused": "false",
        "policy_mismatch": "false",
    }
    fields.update(overrides)
    return {
        "method": "margin_row_root_precision_execution",
        "status": "ran",
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _f64_bits(value: float) -> str:
    return str(struct.unpack(">Q", struct.pack(">d", value))[0])


def _margin_row_domain_alpha_event(
    **overrides: str,
) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/margin-row-domain-alpha-execution/v1",
        "phase": "terminal",
        "route": "joint",
        "outcome": "unknown",
        "configured_enabled": "true",
        "configured_iters": "4",
        "configured_eta_bits": _f64_bits(0.5),
        "policy_mismatch": "false",
        "gate_evals": "3",
        "movable_domains": "2",
        "movable_slots": "5",
        "warm_entries_applied": "2",
        "ascent_domains": "2",
        "outer_ascent_domains": "1",
        "proposals": "4",
        "trial_passes_completed": "3",
        "trial_pass_failures": "1",
        "finite_trial_scores": "2",
        "accepted_trials": "1",
        "carried_entries": "2",
        "iters_counterfactual_domains": "1",
        "eta_counterfactual_scored_domains": "1",
        "outer_finite_trial_scores": "2",
        "outer_iters_counterfactual_domains": "1",
        "outer_eta_counterfactual_scored_domains": "1",
    }
    fields.update(overrides)
    return {
        "method": "margin_row_domain_alpha_execution",
        "status": "ran",
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _bnn_falsification_policy_event(**overrides: str) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/bnn-falsification-policy/v1",
        "mip_compiled": "true",
        "sign_space_enabled": "false",
        "minimal_move": "false",
        "trust_region": "full_box",
        "ste_pgd_enabled": "true",
        "lower_sign_space_enabled": "true",
        "lower_minimal_move": "false",
        "lower_trust_region": "full_box",
        "lower_ste_pgd_enabled": "true",
    }
    fields.update(overrides)
    return {
        "method": "bnn_falsification_policy",
        "status": "ran",
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _bnn_falsification_execution_event(
    *, status: str | None = None, **overrides: str
) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/bnn-falsification-execution/v1",
        "lane": "sign_space",
        "route": "private",
        "outcome": "disarmed",
        "sign_space_enabled": "false",
        "minimal_move": "false",
        "trust_region": "full_box",
        "ste_pgd_enabled": "true",
        "work_started": "false",
        "budget_ms": "none",
        "opportunity_budget_ms": "none",
        "free_units": "0",
        "flips": "0",
        "work_units": "0",
    }
    fields.update(overrides)
    if "opportunity_budget_ms" not in overrides:
        if fields["budget_ms"] != "none":
            fields["opportunity_budget_ms"] = fields["budget_ms"]
        elif (
            fields["outcome"] == "disarmed"
            and fields["route"] in {"private", "committed_cap"}
        ):
            fields["opportunity_budget_ms"] = "20000"
    return {
        "method": "bnn_falsification_execution",
        "status": status or ("ran" if fields["work_started"] == "true" else "skipped"),
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.02,
    }


def _bnn_falsification_record(
    events: list[dict[str, object]],
    *,
    sign_space: tuple[object, str] = (False, "legacy_env"),
    minimal_move: tuple[object, str] = (False, "default"),
    trust_region: tuple[object, str] = (None, "default"),
    ste_pgd: tuple[object, str] = (True, "config"),
) -> dict[str, object]:
    values = {
        "NY_BNN_SIGN_SPACE": sign_space,
        "NY_BNN_SIGN_SPACE_MINIMAL_MOVE": minimal_move,
        "NY_BNN_SIGN_SPACE_TRUST_REGION": trust_region,
        "NY_BNN_STE_PGD": ste_pgd,
    }
    entries = []
    ambient = {}
    for name, (value, source) in sorted(values.items()):
        entry = {
            "name": name,
            "value": value,
            "source": source,
            "bucket": "debug",
            "moat": "high",
            "provenance": "unmeasured",
        }
        if source == "legacy_env":
            entry["env_utf8"] = True
            ambient[name] = (
                "1" if value is True else "0" if value is False else str(value)
            )
        entries.append(entry)
    env_entries = [entry for entry in entries if entry["source"] == "legacy_env"]
    return {
        "schema_version": 3,
        "ambient_env": ambient,
        "levers": {
            "status": "resolved",
            "receipt": {
                "schema": "ny-levers/receipt/v3",
                "lever_count": len(entries),
                "env_present": len(env_entries),
                "env_accepted": len(env_entries),
                "env_rejected": 0,
                "env_ignored": 0,
                "levers": entries,
            },
        },
        "events": events,
    }


def _margin_row_root_alpha_event(
    **overrides: str,
) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/margin-row-root-alpha-execution/v1",
        "phase": "terminal",
        "outcome": "unknown",
        "alpha_enabled": "true",
        "alpha_iters": "16",
        "alpha_secs_bits": _f64_bits(20.0),
        "alpha_lr_bits": _f64_bits(0.1),
        "alpha_cols": "8",
        "alpha_guard": "true",
        "rebake_enabled": "false",
        "rebake_rounds": "2",
        "rebake_secs_bits": _f64_bits(30.0),
        "eligible": "true",
        "standalone_suppressed": "0",
        "standalone_attempted": "1",
        "standalone_completed": "1",
        "standalone_effective_secs_bits": _f64_bits(20.0),
        "standalone_deadline_capped": "false",
        "standalone_budget_exhausted": "false",
        "standalone_iters_cap_exhausted": "true",
        "standalone_fail_classes": "10",
        "standalone_iters_attempted": "16",
        "standalone_candidate_steps": "16",
        "standalone_legacy_steps": "0",
        "standalone_adam_steps": "16",
        "standalone_passes": "17",
        "standalone_max_steer_cols": "8",
        "standalone_frozen_improvements": "1",
        "standalone_accepted": "true",
        "standalone_moved_neurons": "1",
        "standalone_guard_opportunities": "1",
        "standalone_guard_baseline_evals": "1",
        "standalone_guard_candidate_evals": "1",
        "standalone_guard_reversions": "0",
        "standalone_frozen_before_bits": _f64_bits(-1.0),
        "standalone_frozen_best_bits": _f64_bits(-0.5),
        "standalone_published_before_bits": _f64_bits(-1.0),
        "standalone_published_after_bits": _f64_bits(-0.5),
        "rebake_suppressed": "1",
        "rebake_attempted": "0",
        "rebake_completed": "0",
        "rebake_effective_secs_bits": "0",
        "rebake_deadline_capped": "false",
        "rebake_budget_exhausted": "false",
        "rebake_round_cap_exhausted": "false",
        "rebake_entry_unbaked_alpha": "false",
        "rebake_baseline_evals": "0",
        "rebake_rounds_started": "0",
        "rebake_alpha_completed": "0",
        "rebake_alpha_iters_attempted": "0",
        "rebake_alpha_candidate_steps": "0",
        "rebake_alpha_legacy_steps": "0",
        "rebake_alpha_adam_steps": "0",
        "rebake_alpha_passes": "0",
        "rebake_alpha_max_fail_classes": "0",
        "rebake_alpha_max_steer_cols": "0",
        "rebake_alpha_moved_neurons": "0",
        "rebake_rebuilds_completed": "0",
        "rebake_rounds_scored": "0",
        "rebake_rounds_accepted": "0",
        "rebake_fixed_point_stops": "0",
        "rebake_affordability_refusals": "0",
        "rebake_aborts": "0",
        "rebake_published_before_bits": "none",
        "rebake_published_after_bits": "none",
        "policy_mismatch": "false",
    }
    fields.update(overrides)
    return {
        "method": "margin_row_root_alpha_execution",
        "status": "ran",
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _margin_row_root_alpha_disabled_event() -> dict[str, object]:
    return _margin_row_root_alpha_event(
        alpha_enabled="false",
        alpha_iters="20",
        standalone_suppressed="1",
        standalone_attempted="0",
        standalone_completed="0",
        standalone_effective_secs_bits="0",
        standalone_iters_cap_exhausted="false",
        standalone_fail_classes="0",
        standalone_iters_attempted="0",
        standalone_candidate_steps="0",
        standalone_adam_steps="0",
        standalone_passes="0",
        standalone_max_steer_cols="0",
        standalone_frozen_improvements="0",
        standalone_accepted="false",
        standalone_moved_neurons="0",
        standalone_guard_opportunities="0",
        standalone_guard_baseline_evals="0",
        standalone_guard_candidate_evals="0",
        standalone_frozen_before_bits="none",
        standalone_frozen_best_bits="none",
        standalone_published_before_bits="none",
        standalone_published_after_bits="none",
    )


def _margin_row_epoch_event(**overrides: str) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/margin-row-epoch-execution/v1",
        "phase": "terminal",
        "route": "joint",
        "outcome": "unknown",
        "configured_depth": "3",
        "configured_attempts": "4",
        "configured_tier0": "8",
        "configured_universe": "128",
        "configured_retain": "256",
        "configured_parent_topk": "0",
        "configured_debug": "false",
        "policy_mismatch": "false",
        "tree_entries": "8",
        "configured_depth_admissions": "5",
        "configured_depth_nested_with_stats": "1",
        "adaptive_depth_admissions": "0",
        "attempt_cap_decisions": "7",
        "attempt_cap_counterfactual_differences": "2",
        "attempt_cap_refusals": "1",
        "generation_refusals": "0",
        "deadline_refusals": "1",
        "attempts_started": "5",
        "attempt_failures": "4",
        "attempts_closed": "1",
        "outer_full_attempts": "5",
        "outer_incremental_attempts": "0",
        "nested_full_attempts": "0",
        "nested_incremental_attempts": "0",
        "full_rebuilds_completed": "3",
        "full_rebuild_refusals": "2",
        "parent_refreshes_started": "0",
        "parent_refreshes_completed": "0",
        "parent_refreshes_accepted": "0",
        "parent_refresh_refusals": "0",
        "parent_refresh_columns": "0",
        "parent_refresh_tightened": "0",
        "parent_refresh_stabilized": "0",
        "nested_started": "3",
        "nested_runs_with_stats": "3",
        "nested_unsat": "1",
        "nested_unknown": "2",
        "nested_errors": "0",
        "closures": "1",
        "debug_emissions": "0",
        "outer_tier0_calls": "4",
        "outer_tier0_scored_calls": "3",
        "outer_tier0_exact_binding_scored_calls": "2",
        "outer_universe_counterfactual_scored_calls": "1",
        "outer_retain_counterfactual_scored_calls": "1",
        "nested_full_tier0_calls": "1",
        "nested_incremental_tier0_calls": "0",
    }
    fields.update(overrides)
    return {
        "method": "margin_row_epoch_execution",
        "status": "ran",
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _mo_kfsb_event(**overrides: str) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/mo-kfsb-execution/v1",
        "phase": "terminal",
        "observation_schema": "ny_beta_crown_mo_kfsb_observations_v1",
        "canonical_master": "exact_enable",
        "legacy_childsim": "false",
        "configured_master": "false",
        "configured_candidates": "7",
        "effective_candidates": "7",
        "configured_reduce_op": "max",
        "effective_reduce_op": "max",
        "heuristic_eligible": "true",
        "branch_opportunities": "1",
        "offered_domains": "1",
        "nonempty_domain_opportunities": "1",
        "candidate_eligible_opportunities": "1",
        "typed_select_opportunities": "0",
        "lower_master_opportunities": "0",
        "effective_master_opportunities": "1",
        "route_eligible_opportunities": "1",
        "selector_waves": "1",
        "selector_domains": "1",
        "committed_waves": "1",
        "committed_parents": "1",
        "committed_leaves": "0",
        "force_on_binding_waves": "1",
        "master_binding_refusals": "0",
    }
    fields.update(overrides)
    return {
        "method": "mo_kfsb_execution",
        "status": "ran" if fields["selector_waves"] != "0" else "skipped",
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _mo_alpha_step_event(
    *, status: str | None = None, **overrides: str
) -> dict[str, object]:
    fields = {
        "schema": "ny-cli/mo-alpha-step-execution/v1",
        "phase": "terminal",
        "observation_schema": "ny_beta_crown_mo_alpha_step_observations_v1",
        "observed": "true",
        "effective_alpha_step": "true",
        "effective_beta_spsa": "true",
        "schedule_period": "4",
        "alpha_round_remainder": "3",
        "optimizer_invocations": "1",
        "eligible_invocations": "1",
        "eligible_iterations": "15",
        "fourth_iteration_opportunities": "3",
        "alpha_rounds_scheduled": "3",
        "beta_rounds_scheduled": "12",
        "alpha_rounds_completed": "3",
        "alpha_rounds_refused": "0",
        "alpha_updates_applied": "3",
        "perturbed_coordinates": "96",
    }
    fields.update(overrides)
    if status is None:
        status = "ran" if fields["alpha_rounds_completed"] != "0" else "skipped"
    return {
        "method": "mo_alpha_step_execution",
        "status": status,
        "reason": " ".join(f"{name}={value}" for name, value in fields.items()),
        "at_secs": 0.01,
    }


def _execution_predicate_record(
    assignment: list[tuple[str, str, object]],
    event: dict[str, object],
) -> dict[str, object]:
    entries = [
        {
            "name": name,
            "value": value,
            "source": "legacy_env",
            "bucket": "debug",
            "moat": "high",
            "provenance": "unmeasured",
            "env_utf8": True,
        }
        for name, _token, value in assignment
    ]
    if event.get("method") == "margin_row_backward_interm_execution":
        reason = event.get("reason")
        assert isinstance(reason, str)
        fields = dict(token.split("=", 1) for token in reason.split(" "))
        context = {
            "NY_MARGIN_ROW_BACKWARD_INTERM": (
                fields["enabled"] == "true", fields["enabled"] == "false"
            ),
            "NY_MARGIN_ROW_BI_SECS": (
                float(fields["configured_secs"]), fields["configured_secs"] == "20"
            ),
            "NY_MARGIN_ROW_BI_CHUNK": (
                int(fields["configured_chunk"]), fields["configured_chunk"] == "256"
            ),
            "NY_MARGIN_ROW_BI_CHUNK_SECS": (
                float(fields["configured_chunk_secs"]),
                fields["configured_chunk_secs"] == "0",
            ),
            "NY_MARGIN_ROW_BI_TOPK": (
                int(fields["configured_topk"]), fields["configured_topk"] == "1024"
            ),
        }
        existing = {entry["name"] for entry in entries}
        for name, (value, is_default) in context.items():
            if name in existing:
                continue
            entries.append(
                {
                    "name": name,
                    "value": value,
                    "source": "default" if is_default else "config",
                    "bucket": "debug",
                    "moat": "high",
                    "provenance": "unmeasured",
                }
            )
        entries.sort(key=lambda entry: entry["name"])
    elif event.get("method") == "margin_row_domain_alpha_execution":
        reason = event.get("reason")
        assert isinstance(reason, str)
        fields = dict(token.split("=", 1) for token in reason.split(" "))
        eta = struct.unpack(
            ">d", struct.pack(">Q", int(fields["configured_eta_bits"]))
        )[0]
        context = {
            "NY_MARGIN_ROW_DOMAIN_ALPHA": (
                fields["configured_enabled"] == "true", False
            ),
            "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS": (
                int(fields["configured_iters"]), 2
            ),
            "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA": (eta, 0.25),
        }
        existing = {entry["name"] for entry in entries}
        for name, (value, default) in context.items():
            if name in existing:
                continue
            entries.append(
                {
                    "name": name,
                    "value": value,
                    "source": "default" if value == default else "config",
                    "bucket": "debug",
                    "moat": "high",
                    "provenance": "unmeasured",
                }
            )
        entries.sort(key=lambda entry: entry["name"])
    elif event.get("method") == "margin_row_epoch_execution":
        reason = event.get("reason")
        assert isinstance(reason, str)
        fields = dict(token.split("=", 1) for token in reason.split(" "))
        context = {
            "NY_EPOCH_DEPTH": (int(fields["configured_depth"]), 0),
            "NY_EPOCH_ATTEMPTS": (int(fields["configured_attempts"]), 2),
            "NY_EPOCH_TIER0": (int(fields["configured_tier0"]), 0),
            "NY_EPOCH_UNIVERSE": (int(fields["configured_universe"]), 64),
            "NY_EPOCH_RETAIN": (int(fields["configured_retain"]), 128),
            "NY_MARGIN_ROW_EPOCH_PARENT_TOPK": (
                int(fields["configured_parent_topk"]), 0
            ),
        }
        existing = {entry["name"] for entry in entries}
        for name, (value, default) in context.items():
            if name in existing:
                continue
            entries.append(
                {
                    "name": name,
                    "value": value,
                    "source": "default" if value == default else "config",
                    "bucket": "debug",
                    "moat": "high",
                    "provenance": "unmeasured",
                }
            )
        entries.sort(key=lambda entry: entry["name"])
    elif event.get("method") == "margin_row_width_execution":
        reason = event.get("reason")
        assert isinstance(reason, str)
        fields = dict(token.split("=", 1) for token in reason.split(" "))
        context = {
            "NY_MARGIN_ROW_K_HEAD": (int(fields["configured_k_head"]), 8),
            "NY_MARGIN_ROW_K_TRUNK": (int(fields["configured_k_trunk"]), 8),
            "NY_MARGIN_ROW_K_ADAPT": (
                True if fields["adaptive_enabled"] == "true" else None,
                None,
            ),
        }
        existing = {entry["name"] for entry in entries}
        for name, (value, default) in context.items():
            if name in existing:
                continue
            entries.append(
                {
                    "name": name,
                    "value": value,
                    "source": "default" if value == default else "config",
                    "bucket": "debug",
                    "moat": "high",
                    "provenance": "unmeasured",
                }
            )
        entries.sort(key=lambda entry: entry["name"])
    elif event.get("method") == "margin_row_root_alpha_execution":
        reason = event.get("reason")
        assert isinstance(reason, str)
        fields = dict(token.split("=", 1) for token in reason.split(" "))
        from_bits = lambda name: struct.unpack(
            ">d", struct.pack(">Q", int(fields[name]))
        )[0]
        context = {
            "NY_MARGIN_ROW_ALPHA_OPT": (
                fields["alpha_enabled"] == "true", False
            ),
            "NY_MARGIN_ROW_ALPHA_ITERS": (int(fields["alpha_iters"]), 20),
            "NY_MARGIN_ROW_ALPHA_SECS": (
                from_bits("alpha_secs_bits"), 20.0
            ),
            "NY_MARGIN_ROW_ALPHA_LR": (from_bits("alpha_lr_bits"), 0.1),
            "NY_MARGIN_ROW_ALPHA_COLS": (int(fields["alpha_cols"]), 8),
            "NY_MARGIN_ROW_ALPHA_GUARD": (
                fields["alpha_guard"] == "true", True
            ),
            "NY_MARGIN_ROW_REBAKE": (
                fields["rebake_enabled"] == "true", False
            ),
            "NY_MARGIN_ROW_REBAKE_ROUNDS": (
                int(fields["rebake_rounds"]), 2
            ),
            "NY_MARGIN_ROW_REBAKE_SECS": (
                from_bits("rebake_secs_bits"), 30.0
            ),
        }
        existing = {entry["name"] for entry in entries}
        for name, (value, default) in context.items():
            if name in existing:
                continue
            entries.append(
                {
                    "name": name,
                    "value": value,
                    "source": "default" if value == default else "config",
                    "bucket": "debug",
                    "moat": "high",
                    "provenance": "unmeasured",
                }
            )
        entries.sort(key=lambda entry: entry["name"])
    elif event.get("method") == "margin_row_root_precision_execution":
        reason = event.get("reason")
        assert isinstance(reason, str)
        fields = dict(token.split("=", 1) for token in reason.split(" "))
        context = {
            "NY_MARGIN_ROW_ROOT_F32": fields["configured_root_f32"] == "true",
            "NY_MARGIN_ROW_TABLEAU_F32": (
                fields["configured_tableau_f32"] == "true"
            ),
        }
        existing = {entry["name"] for entry in entries}
        for name, value in context.items():
            if name in existing:
                continue
            entries.append(
                {
                    "name": name,
                    "value": value,
                    "source": "config" if value else "default",
                    "bucket": "debug",
                    "moat": "high",
                    "provenance": "measured",
                }
            )
        entries.sort(key=lambda entry: entry["name"])
    env_entries = [entry for entry in entries if entry["source"] == "legacy_env"]
    return {
        "schema_version": 3,
        "ambient_env": {name: token for name, token, _value in assignment},
        "levers": {
            "status": "resolved",
            "receipt": {
                "schema": "ny-levers/receipt/v3",
                "lever_count": len(entries),
                "env_present": len(env_entries),
                "env_accepted": len(env_entries),
                "env_rejected": 0,
                "env_ignored": 0,
                "levers": entries,
            },
        },
        "events": [event],
    }


def _margin_row_schedule_record(
    events: list[dict[str, object]],
    *,
    reserve: tuple[object, str] = (0, "config"),
    adaptive: tuple[object, str] = (False, "default"),
    max_fraction: tuple[object, str] = (None, "default"),
    release_fraction: tuple[object, str] = (0.40, "default"),
    concurrent: tuple[object, str] = (True, "legacy_env"),
    early_exit: tuple[object, str] = (True, "default"),
    verifier_threads: tuple[object, str] = (None, "default"),
    concurrent_threads: tuple[object, str] = (None, "default"),
) -> dict[str, object]:
    configured = {
        "NY_MARGIN_ROW_RESERVE_SECS": reserve,
        "NY_MARGIN_ROW_ADAPTIVE_RESERVE": adaptive,
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC": max_fraction,
        "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC": release_fraction,
        "NY_MARGIN_ROW_CONCURRENT": concurrent,
        "NY_MARGIN_ROW_EARLY_EXIT": early_exit,
        "NY_VERIFIER_THREADS": verifier_threads,
        "NY_MARGIN_ROW_CONCURRENT_THREADS": concurrent_threads,
    }
    entries = []
    ambient = {}
    for name, (value, source) in sorted(configured.items()):
        entry = {
            "name": name,
            "value": value,
            "source": source,
            "bucket": "debug",
            "moat": "high",
            "provenance": "unmeasured",
        }
        if source == "legacy_env":
            entry["env_utf8"] = True
            ambient[name] = (
                "1" if value is True else "0" if value is False else str(value)
            )
        entries.append(entry)
    env_entries = [entry for entry in entries if entry["source"] == "legacy_env"]
    return {
        "schema_version": 3,
        "ambient_env": ambient,
        "levers": {
            "status": "resolved",
            "receipt": {
                "schema": "ny-levers/receipt/v3",
                "lever_count": len(entries),
                "env_present": len(env_entries),
                "env_accepted": len(env_entries),
                "env_rejected": 0,
                "env_ignored": 0,
                "levers": entries,
            },
        },
        "events": events,
    }


def _current_source_search_space_payload() -> dict[str, object]:
    """Render the current Rust mirror in the exact compiled-v5 JSON shape."""
    import ny_search  # noqa: PLC0415

    space = ny_search.parse_space_rs()
    declarations = ny_search.parse_lever_kinds()
    return {
        "schema": ny_search.SPACE_SCHEMA,
        "axes": [{**axis.as_json(), "why": axis.why} for axis in space.axes],
        "edges": [
            {
                "child": edge.child,
                "kind": edge.kind,
                "parent": edge.parent,
                "parents": edge.parents or None,
                "bound": edge.bound,
                "token": edge.token,
                "site": edge.site,
            }
            for edge in space.edges
        ],
        "unsafe": space.unsafe_names,
        "instrument_only": space.instrument_only,
        "test_only": space.test_only,
        "declarations": {
            name: {
                "kind": declaration["kind"],
                "default": declaration.get("default"),
                "min": declaration.get("min"),
                "max": declaration.get("max"),
                "members": declaration.get("members"),
            }
            for name, declaration in declarations.items()
        },
    }


def _schedule_v2_source_search_space_payload() -> dict[str, object]:
    """Model the declarations supplied by the independently staged Rust v2."""
    payload = _current_source_search_space_payload()
    unsafe = set(payload["unsafe"])
    declarations = payload["declarations"]
    additions = {
        "NY_MARGIN_ROW_EARLY_EXIT": {
            "kind": "TrimmedExactOne", "default": "1", "min": None,
            "max": None, "members": None,
        },
        "NY_VERIFIER_THREADS": {
            "kind": "UsizeTrimmedUnicodeOwned", "default": None, "min": None,
            "max": None, "members": None,
        },
        "NY_MARGIN_ROW_CONCURRENT_THREADS": {
            "kind": "UsizeOverride", "default": None, "min": 0,
            "max": None, "members": None,
        },
    }
    missing = set(additions).difference(declarations)
    # This branch commit is intentionally independently testable before the
    # Rust declaration commit is integrated. Once any staged declaration is
    # present, require all three and preserve the real payload so a wrong kind,
    # default, min, or Unsafe classification cannot be hidden by this fixture.
    if missing:
        assert missing == set(additions)
        declarations.update(additions)
        unsafe.update(additions)
        payload["unsafe"] = sorted(unsafe)
    return payload


def _fixture_solver_script(
    version_line: str = "ny 0.1.0-test",
    *,
    search_space_payload: dict[str, object] | None = None,
) -> str:
    search_space_json = json.dumps(
        search_space_payload or _FIXTURE_SEARCH_SPACE_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "#!/bin/sh\n"
        'if [ "$1" = "benchmarks" ] && [ "$2" = "space" ] '
        '&& [ "$3" = "--json" ]; then\n'
        "  cat <<'NY_FIXTURE_SEARCH_SPACE'\n"
        f"{search_space_json}\n"
        "NY_FIXTURE_SEARCH_SPACE\n"
        "  exit 0\n"
        "fi\n"
        'if [ "$1" = "--build-id" ]; then\n'
        '  if command -v sha256sum >/dev/null 2>&1; then\n'
        '    self=$(sha256sum "$0" | cut -d" " -f1)\n'
        f"    identity=$(printf 'NY_BUILD_ID_V1\\n{_FIXTURE_BUILD_STAMP}\\n%s\\n' "
        '      "$self" | sha256sum | cut -d" " -f1)\n'
        "  else\n"
        '    self=$(shasum -a 256 "$0" | cut -d" " -f1)\n'
        f"    identity=$(printf 'NY_BUILD_ID_V1\\n{_FIXTURE_BUILD_STAMP}\\n%s\\n' "
        '      "$self" | shasum -a 256 | cut -d" " -f1)\n'
        "  fi\n"
        f'  printf "NY_BUILD_ID_V1 {_FIXTURE_BUILD_ID_FIELDS}'
        ' exe=%s id=%s\\n" "$self" "$identity"\n'
        "  exit 0\n"
        "fi\n"
        f"echo '{version_line}'\n"
    )


def _run(*command: str, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _init_git_repo(path: Path) -> None:
    _run("git", "init", "-q", cwd=path)
    _run("git", "config", "user.name", "NY Test", cwd=path)
    _run("git", "config", "user.email", "ny-test@example.invalid", cwd=path)


def _measurement_repos(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "ny"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / ".gitignore").write_text(
        "/target/\n/reports/\n/benchmarks/\n", encoding="utf-8"
    )
    (repo / "rust-toolchain.toml").write_text(
        '[toolchain]\nchannel = "1.95.0"\ncomponents = ["rustfmt", "clippy"]\n',
        encoding="utf-8",
    )
    (repo / "Cargo.lock").write_text(
        "[[package]]\n"
        'name = "ay-test"\n'
        'version = "0.1.0"\n'
        f'source = "git+https://github.com/alabsystems/ay.git?rev={AY_REV}'
        f'#{AY_REV}"\n',
        encoding="utf-8",
    )
    scripts = repo / "scripts"
    scripts.mkdir()
    sweep = scripts / "measure_ny_scorecard.sh"
    sweep.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    binary = repo / "target" / "release" / "ny"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        _fixture_solver_script(
            search_space_payload=_current_source_search_space_payload()
        ),
        encoding="utf-8",
    )
    binary.chmod(0o755)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "fixture", cwd=repo)
    # Git records whole-second commit epochs, which can tick over between the
    # fixture binary write and commit. Model a just-built solver deterministically.
    binary.touch()

    benchmark = tmp_path / "vnncomp-benchmarks"
    benchmark.mkdir()
    _init_git_repo(benchmark)
    benchmark_root = benchmark / "benchmarks"
    benchmark_root.mkdir()
    (benchmark_root / "README").write_text("fixture\n", encoding="utf-8")
    _run("git", "add", ".", cwd=benchmark)
    _run("git", "commit", "-qm", "fixture", cwd=benchmark)
    _run(
        "git",
        "remote",
        "add",
        "origin",
        "https://user:supersecret@github.com/example/bench.git?token=hidden",
        cwd=benchmark,
    )
    return repo, benchmark_root


def _clear_ny_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if (
            key.startswith(("NY_", "AY_", "MIMALLOC_", "LD_"))
            or key.upper().startswith("MIMALLOC_")
            or key.startswith(provenance.UNREVIEWED_SOLVER_RUNTIME_ENV_PREFIXES)
            or key in provenance.UNREVIEWED_SOLVER_RUNTIME_ENV_EXACT
            or key
            in {
                "DYLD_FORCE_FLAT_NAMESPACE",
                "DYLD_INSERT_LIBRARIES",
                "DYLD_LIBRARY_PATH",
            }
        ):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("GPU_AVAILABLE", raising=False)
    monkeypatch.setenv("NY_BUILD_FEATURES", "mip,cuda")


def test_mo_kfsb_predicate_requires_a_causal_master_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="MO-kFSB predicate fixture",
    )
    raw = (
        "ny-exec-v1.mo-kfsb-parameter-engaged."
        "NY_MO_KFSB.1=ran"
    )
    monkeypatch.setenv("NY_MO_KFSB", "1")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    assert assignment["requested_assignment"] == [["NY_MO_KFSB", "1"]]
    assert assignment["comparison_assignment"] == []

    record = _execution_predicate_record(
        [("NY_MO_KFSB", "1", True)],
        _mo_kfsb_event(),
    )
    assert provenance.validate_execution_predicate_requirement(
        record,
        raw,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(raw)

    record["events"] = [
        _mo_kfsb_event(
            committed_waves="0",
            committed_parents="0",
            force_on_binding_waves="0",
        )
    ]
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MO_KFSB=1",
    ):
        provenance.validate_execution_predicate_requirement(
            record,
            raw,
            assignment=assignment,
        )


def test_mo_alpha_step_predicate_requires_completed_every_fourth_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="MO-alpha predicate fixture",
    )
    raw = (
        "ny-exec-v1.mo-alpha-step-parameter-engaged."
        "NY_MO_ALPHA_STEP.1=ran"
    )
    monkeypatch.setenv("NY_MO_ALPHA_STEP", "1")
    monkeypatch.setenv("NY_MO_BETA_SPSA", "1")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    assert assignment["requested_assignment"] == [["NY_MO_ALPHA_STEP", "1"]]
    assert assignment["prerequisites_assignment"] == [["NY_MO_BETA_SPSA", "1"]]

    record = _execution_predicate_record(
        [
            ("NY_MO_ALPHA_STEP", "1", True),
            ("NY_MO_BETA_SPSA", "1", True),
        ],
        _mo_alpha_step_event(),
    )
    assert provenance.validate_execution_predicate_requirement(
        record,
        raw,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(raw)


def test_mo_alpha_step_predicate_requires_sealed_beta_assignment() -> None:
    raw = (
        "ny-exec-v1.mo-alpha-step-parameter-engaged."
        "NY_MO_ALPHA_STEP.1=ran"
    )
    record = _execution_predicate_record(
        [("NY_MO_ALPHA_STEP", "1", True)],
        _mo_alpha_step_event(),
    )
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="requires a sealed beta-SPSA prerequisite assignment",
    ):
        provenance.validate_execution_predicate_requirement(record, raw)


def test_mo_alpha_step_predicate_rejects_forged_empty_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="MO-alpha forged prerequisite fixture",
    )
    raw = (
        "ny-exec-v1.mo-alpha-step-parameter-engaged."
        "NY_MO_ALPHA_STEP.1=ran"
    )
    monkeypatch.setenv("NY_MO_ALPHA_STEP", "1")
    monkeypatch.setenv("NY_MO_BETA_SPSA", "1")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    forged = dict(assignment)
    forged["prerequisites_assignment"] = []
    forged["comparison_assignment"] = []
    forged["full_assignment"] = [["NY_MO_ALPHA_STEP", "1"]]
    forged["receipt_source_assignment"] = [
        pair for pair in assignment["receipt_source_assignment"]
        if pair[0] == "NY_MO_ALPHA_STEP"
    ]
    record = _execution_predicate_record(
        [("NY_MO_ALPHA_STEP", "1", True)],
        _mo_alpha_step_event(),
    )
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="exact sealed NY_MO_BETA_SPSA=1 prerequisite",
    ):
        provenance.validate_execution_predicate_requirement(
            record,
            raw,
            assignment=forged,
        )


@pytest.mark.parametrize(
    "events",
    [
        [
            _mo_alpha_step_event(
                status="skipped",
                alpha_rounds_completed="0",
                alpha_rounds_refused="3",
                alpha_updates_applied="0",
            )
        ],
        [
            _mo_alpha_step_event(
                status="skipped",
                eligible_iterations="3",
                fourth_iteration_opportunities="0",
                alpha_rounds_scheduled="0",
                beta_rounds_scheduled="3",
                alpha_rounds_completed="0",
                alpha_updates_applied="0",
                perturbed_coordinates="0",
            )
        ],
        [_mo_alpha_step_event(effective_beta_spsa="false")],
        [_mo_alpha_step_event(schedule_period="5")],
        [
            _mo_alpha_step_event(
                eligible_iterations="11",
                beta_rounds_scheduled="8",
            )
        ],
        [_mo_alpha_step_event(perturbed_coordinates="2")],
        [_mo_alpha_step_event(perturbed_coordinates="97")],
        [_mo_alpha_step_event(), _mo_alpha_step_event()],
        [
            {
                **_mo_alpha_step_event(),
                "reason": _mo_alpha_step_event()["reason"].replace(
                    " eligible_iterations=15",
                    " eligible_iterations=015",
                ),
            }
        ],
    ],
    ids=[
        "all-refused",
        "before-fourth",
        "beta-disabled",
        "wrong-schedule",
        "impossible-fourth-count",
        "too-few-coordinates",
        "indivisible-single-invocation-coordinates",
        "duplicate",
        "noncanonical-counter",
    ],
)
def test_mo_alpha_step_predicate_rejects_inactive_or_poisoned_terminals(
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, object]],
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="MO-alpha rejection fixture",
    )
    raw = (
        "ny-exec-v1.mo-alpha-step-parameter-engaged."
        "NY_MO_ALPHA_STEP.1=ran"
    )
    monkeypatch.setenv("NY_MO_ALPHA_STEP", "1")
    monkeypatch.setenv("NY_MO_BETA_SPSA", "1")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    record = _execution_predicate_record(
        [
            ("NY_MO_ALPHA_STEP", "1", True),
            ("NY_MO_BETA_SPSA", "1", True),
        ],
        events[0],
    )
    record["events"] = events
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MO_ALPHA_STEP=1",
    ):
        provenance.validate_execution_predicate_requirement(
            record,
            raw,
            assignment=assignment,
        )


def test_mo_alpha_step_predicate_is_one_sided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="MO-alpha force-off fixture",
    )
    raw = (
        "ny-exec-v1.mo-alpha-step-parameter-engaged."
        "NY_MO_ALPHA_STEP.0=ran"
    )
    monkeypatch.setenv("NY_MO_ALPHA_STEP", "0")
    monkeypatch.setenv("NY_MO_BETA_SPSA", "1")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_MO_ALPHA_STEP=0 equals the DefaultSpec",
    ):
        provenance.build_execution_predicate_assignment(environment, contract)


def test_mo_kfsb_disable_predicate_uses_the_typed_preset_counterfactual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ny_search  # noqa: PLC0415

    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="MO-kFSB disable predicate fixture",
    )
    cifar = ny_search.load_preset_doc(
        REPO_ROOT / "configs" / "vnncomp25" / "cifar100_2024.yaml"
    )
    assert isinstance(cifar, dict)
    raw = (
        "ny-exec-v1.mo-kfsb-parameter-engaged."
        "NY_MO_KFSB.0=ran"
    )
    monkeypatch.setenv("NY_MO_KFSB", "0")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[cifar],
        preset_expected=True,
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[cifar],
        preset_expected=True,
    )
    assert assignment["requested_assignment"] == [["NY_MO_KFSB", "0"]]
    assert assignment["comparison_assignment"] == []

    event = _mo_kfsb_event(
        canonical_master="exact_disable",
        configured_master="true",
        lower_master_opportunities="1",
        effective_master_opportunities="0",
        selector_waves="0",
        selector_domains="0",
        committed_waves="0",
        committed_parents="0",
        force_on_binding_waves="0",
        master_binding_refusals="1",
    )
    record = _execution_predicate_record(
        [("NY_MO_KFSB", "0", False)],
        event,
    )
    assert provenance.validate_execution_predicate_requirement(
        record,
        raw,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(raw)


@pytest.mark.parametrize("reduce_op", ["max"])
def test_mo_kfsb_reduce_predicate_requires_exact_committed_selector_work(
    monkeypatch: pytest.MonkeyPatch,
    reduce_op: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="MO-kFSB reducer predicate fixture",
    )
    raw = (
        "ny-exec-v1.mo-kfsb-parameter-engaged."
        f"NY_MO_KFSB_REDUCE.{reduce_op}=ran"
    )
    # The child axis is executable only with its canonical master prerequisite.
    monkeypatch.setenv("NY_MO_KFSB", "1")
    monkeypatch.setenv("NY_MO_KFSB_REDUCE", reduce_op)
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    assert assignment["requested_assignment"] == [
        ["NY_MO_KFSB_REDUCE", reduce_op]
    ]
    assert assignment["prerequisites_assignment"] == [["NY_MO_KFSB", "1"]]

    record = _execution_predicate_record(
        [
            ("NY_MO_KFSB", "1", True),
            ("NY_MO_KFSB_REDUCE", reduce_op, reduce_op),
        ],
        _mo_kfsb_event(
            configured_reduce_op="min",
            effective_reduce_op=reduce_op,
        ),
    )
    assert provenance.validate_execution_predicate_requirement(
        record,
        raw,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(raw)


def test_mo_kfsb_reduce_predicate_accepts_typed_master_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ny_search  # noqa: PLC0415

    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="MO-kFSB typed-master reducer fixture",
    )
    cifar = ny_search.load_preset_doc(
        REPO_ROOT / "configs" / "vnncomp25" / "cifar100_2024.yaml"
    )
    assert isinstance(cifar, dict)
    raw = (
        "ny-exec-v1.mo-kfsb-parameter-engaged."
        "NY_MO_KFSB_REDUCE.min=ran"
    )
    # CIFAR's typed preset already arms the master and configures Max. Only the
    # reducer override is explicit, so the terminal must report canonical
    # master absence and the armed lower config while proving Max -> Min work.
    monkeypatch.setenv("NY_MO_KFSB_REDUCE", "min")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[cifar],
        preset_expected=True,
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[cifar],
        preset_expected=True,
    )
    assert assignment["receipt_source_assignment"] == [
        ["NY_MO_KFSB", "config"],
        ["NY_MO_KFSB_REDUCE", "legacy_env"],
    ]

    typed_master_event = _mo_kfsb_event(
        canonical_master="absent",
        configured_master="true",
        configured_reduce_op="max",
        effective_reduce_op="min",
        lower_master_opportunities="1",
        force_on_binding_waves="0",
    )
    record = _execution_predicate_record(
        [("NY_MO_KFSB_REDUCE", "min", "min")],
        typed_master_event,
    )
    receipt = record["levers"]["receipt"]
    receipt["levers"].append(
        {
            "name": "NY_MO_KFSB",
            "value": True,
            "source": "config",
            "bucket": "debug",
            "moat": "high",
            "provenance": "unmeasured",
        }
    )
    receipt["levers"].sort(key=lambda entry: entry["name"])
    receipt["lever_count"] = len(receipt["levers"])
    assert provenance.validate_execution_predicate_requirement(
        record,
        raw,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(raw)

    record["events"] = [
        _mo_kfsb_event(
            canonical_master="exact_enable",
            configured_master="true",
            configured_reduce_op="max",
            effective_reduce_op="min",
            lower_master_opportunities="1",
            force_on_binding_waves="0",
        )
    ]
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MO_KFSB_REDUCE=min",
    ):
        provenance.validate_execution_predicate_requirement(
            record,
            raw,
            assignment=assignment,
        )


def test_mo_kfsb_reduce_predicate_requires_sealed_master_assignment() -> None:
    raw = (
        "ny-exec-v1.mo-kfsb-parameter-engaged."
        "NY_MO_KFSB_REDUCE.max=ran"
    )
    record = _execution_predicate_record(
        [("NY_MO_KFSB_REDUCE", "max", "max")],
        _mo_kfsb_event(configured_reduce_op="min"),
    )
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="requires a sealed master assignment",
    ):
        provenance.validate_execution_predicate_requirement(record, raw)


@pytest.mark.parametrize(
    "events",
    [
        [
            _mo_kfsb_event(
                configured_reduce_op="max",
                effective_reduce_op="min",
            )
        ],
        [
            _mo_kfsb_event(
                configured_reduce_op="min",
                route_eligible_opportunities="0",
                selector_waves="0",
                selector_domains="0",
                committed_waves="0",
                committed_parents="0",
                force_on_binding_waves="0",
            )
        ],
        [
            _mo_kfsb_event(
                configured_reduce_op="min",
                committed_waves="0",
                committed_parents="0",
                force_on_binding_waves="0",
            )
        ],
        [
            _mo_kfsb_event(configured_reduce_op="min"),
            _mo_kfsb_event(configured_reduce_op="min"),
        ],
        [
            {
                **_mo_kfsb_event(configured_reduce_op="min"),
                "reason": _mo_kfsb_event(
                    configured_reduce_op="min"
                )["reason"].replace(
                    " selector_waves=1", " selector_waves=01"
                ),
            }
        ],
        [_mo_kfsb_event()],
        [
            _mo_kfsb_event(
                canonical_master="absent",
                configured_master="true",
                configured_reduce_op="min",
                lower_master_opportunities="1",
                force_on_binding_waves="0",
            )
        ],
    ],
    ids=[
        "mismatch",
        "no-selector",
        "no-commit",
        "duplicate",
        "malformed",
        "same-value-no-op",
        "master-prerequisite-mismatch",
    ],
)
def test_mo_kfsb_reduce_predicate_rejects_noncausal_or_poisoned_terminals(
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, object]],
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="MO-kFSB reducer rejection fixture",
    )
    raw = (
        "ny-exec-v1.mo-kfsb-parameter-engaged."
        "NY_MO_KFSB_REDUCE.max=ran"
    )
    monkeypatch.setenv("NY_MO_KFSB", "1")
    monkeypatch.setenv("NY_MO_KFSB_REDUCE", "max")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    record = _execution_predicate_record(
        [
            ("NY_MO_KFSB", "1", True),
            ("NY_MO_KFSB_REDUCE", "max", "max"),
        ],
        events[0],
    )
    record["events"] = events

    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MO_KFSB_REDUCE=max",
    ):
        provenance.validate_execution_predicate_requirement(
            record,
            raw,
            assignment=assignment,
        )


def test_margin_row_root_precision_joint_predicate_requires_both_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="root-precision joint fixture",
    )
    monkeypatch.setenv("NY_MARGIN_ROW_ROOT_F32", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_TABLEAU_F32", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        MARGIN_ROW_ROOT_PRECISION_JOINT_PREDICATE,
    )
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    requested = [
        ["NY_MARGIN_ROW_ROOT_F32", "1"],
        ["NY_MARGIN_ROW_TABLEAU_F32", "1"],
    ]
    assert assignment["requested_assignment"] == requested
    assert assignment["comparison_assignment"] == []
    assert assignment["runtime_context_assignment"] == requested
    assert assignment["runtime_context_source_assignment"] == [
        ["NY_MARGIN_ROW_ROOT_F32", "legacy_env"],
        ["NY_MARGIN_ROW_TABLEAU_F32", "legacy_env"],
    ]
    record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_ROOT_F32", "1", True),
            ("NY_MARGIN_ROW_TABLEAU_F32", "1", True),
        ],
        _margin_row_root_precision_event(),
    )
    assert provenance.validate_execution_predicate_requirement(
        record,
        MARGIN_ROW_ROOT_PRECISION_JOINT_PREDICATE,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(
        MARGIN_ROW_ROOT_PRECISION_JOINT_PREDICATE
    )

    record["events"] = [
        _margin_row_root_precision_event(
            eligible_store_ops="0", tableau_f32_store_ops="0"
        )
    ]
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MARGIN_ROW_TABLEAU_F32=1",
    ):
        provenance.validate_execution_predicate_requirement(
            record,
            MARGIN_ROW_ROOT_PRECISION_JOINT_PREDICATE,
            assignment=assignment,
        )


def test_margin_row_root_precision_uses_preset_counterfactuals_and_full_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ny_search  # noqa: PLC0415

    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="root-precision preset fixture",
    )
    cifar = ny_search.load_preset_doc(
        REPO_ROOT / "configs" / "vnncomp25" / "cifar100_2024.yaml"
    )
    tiny = ny_search.load_preset_doc(
        REPO_ROOT / "configs" / "vnncomp25" / "tinyimagenet_2024.yaml"
    )
    assert isinstance(cifar, dict)
    assert isinstance(tiny, dict)

    root_raw = (
        "ny-exec-v1.margin-row-root-precision-parameter-engaged."
        "NY_MARGIN_ROW_ROOT_F32.0=ran"
    )
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_ROOT_F32", "0")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", root_raw)
    root_environment = provenance._capture_environment(contract)
    root_sealed = provenance.build_execution_predicate_assignment(
        root_environment,
        contract,
        preset_docs=[tiny],
        preset_expected=True,
    )
    root_assignment = provenance.validate_execution_predicate_assignment(
        root_sealed,
        environment=root_environment,
        search_space_contract=contract,
        preset_docs=[tiny],
        preset_expected=True,
    )
    assert root_assignment["comparison_assignment"] == []
    assert dict(root_assignment["runtime_context_assignment"]) == {
        "NY_MARGIN_ROW_ROOT_F32": "0",
        "NY_MARGIN_ROW_TABLEAU_F32": "1",
    }
    assert dict(root_assignment["runtime_context_source_assignment"]) == {
        "NY_MARGIN_ROW_ROOT_F32": "legacy_env",
        "NY_MARGIN_ROW_TABLEAU_F32": "config",
    }
    root_event = _margin_row_root_precision_event(
        configured_root_f32="false",
        root_f32_conv_ops="0",
        root_f64_conv_ops="4",
        tableau_f32_store_ops="0",
        tableau_f64_store_ops="9",
        tableau_refused="true",
    )
    root_record = _execution_predicate_record(
        [("NY_MARGIN_ROW_ROOT_F32", "0", False)], root_event
    )
    assert provenance.validate_execution_predicate_requirement(
        root_record, root_raw, assignment=root_assignment
    ) == provenance.parse_execution_predicate_requirement(root_raw)

    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_ROOT_F32", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT", root_raw.replace(".0=ran", ".1=ran")
    )
    no_op_environment = provenance._capture_environment(contract)
    with pytest.raises(provenance.ProvenanceError, match="no-op run"):
        provenance.build_execution_predicate_assignment(
            no_op_environment,
            contract,
            preset_docs=[tiny],
            preset_expected=True,
        )

    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_ROOT_F32", "0")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", root_raw)
    mixed_environment = provenance._capture_environment(contract)
    with pytest.raises(provenance.ProvenanceError, match="split the measurement"):
        provenance.build_execution_predicate_assignment(
            mixed_environment,
            contract,
            preset_docs=[cifar, tiny],
            preset_expected=True,
        )


def test_margin_row_tableau_precision_seals_root_prerequisite_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ny_search  # noqa: PLC0415

    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="tableau-precision prerequisite fixture",
    )
    tiny = ny_search.load_preset_doc(
        REPO_ROOT / "configs" / "vnncomp25" / "tinyimagenet_2024.yaml"
    )
    assert isinstance(tiny, dict)
    raw = (
        "ny-exec-v1.margin-row-root-precision-parameter-engaged."
        "NY_MARGIN_ROW_TABLEAU_F32.0=ran"
    )
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_TABLEAU_F32", "0")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    incomplete_environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ProvenanceError,
        match="direct governed inputs do not equal",
    ):
        provenance.build_execution_predicate_assignment(
            incomplete_environment,
            contract,
            preset_docs=[tiny],
            preset_expected=True,
        )

    monkeypatch.setenv("NY_MARGIN_ROW_ROOT_F32", "1")
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[tiny],
        preset_expected=True,
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[tiny],
        preset_expected=True,
    )
    assert assignment["requested_assignment"] == [
        ["NY_MARGIN_ROW_TABLEAU_F32", "0"]
    ]
    assert assignment["comparison_assignment"] == [
        ["NY_MARGIN_ROW_ROOT_F32", "1"]
    ]
    assert dict(assignment["runtime_context_assignment"]) == {
        "NY_MARGIN_ROW_ROOT_F32": "1",
        "NY_MARGIN_ROW_TABLEAU_F32": "0",
    }
    event = _margin_row_root_precision_event(
        configured_tableau_f32="false",
        tableau_f32_store_ops="0",
        tableau_f64_store_ops="9",
    )
    record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_ROOT_F32", "1", True),
            ("NY_MARGIN_ROW_TABLEAU_F32", "0", False),
        ],
        event,
    )
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)

    root_entry = next(
        entry
        for entry in record["levers"]["receipt"]["levers"]
        if entry["name"] == "NY_MARGIN_ROW_ROOT_F32"
    )
    root_entry["value"] = False
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="terminal configuration differs from its exact receipt",
    ):
        provenance.validate_execution_predicate_requirement(
            record, raw, assignment=assignment
        )


@pytest.mark.parametrize(
    "key",
    [
        "DYLD_FORCE_FLAT_NAMESPACE",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "LD_AUDIT",
        "LD_PRELOAD",
    ],
)
def test_measurement_environment_rejects_dynamic_loader_injection(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "")

    with pytest.raises(provenance.ProvenanceError, match="dynamic-loader injection"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    [
        "__GLX_VENDOR_LIBRARY_NAME",
        "__NV_PRIME_RENDER_OFFLOAD",
        "__VK_LAYER_NV_optimus",
        "ACCELERATE_NEW_LAPACK",
        "ACO_DEBUG",
        "AMD_CU_MASK",
        "ANV_DEBUG",
        "BLIS_NUM_THREADS",
        "CANDLE_GRAD_DO_NOT_DETACH",
        "CARGO_HOME",
        "CUBLASLT_LOG_LEVEL",
        "CUDA_DEVICE_ORDER",
        "CUDA_LAUNCH_BLOCKING",
        "CUDNN_LOGLEVEL_DBG",
        "DISABLE_LAYER_NV_OPTIMUS_1",
        "DRI_PRIME",
        "DRI_PRIME_DEBUG",
        "GALLIUM_DRIVER",
        "GBM_BACKEND",
        "GCONV_PATH",
        "GLIBC_TUNABLES",
        "GOMP_CPU_AFFINITY",
        "GOTO_NUM_THREADS",
        "INTEL_NO_HW",
        "KMP_AFFINITY",
        "LC_CTYPE",
        "LIBGL_ALWAYS_SOFTWARE",
        "LP_NUM_THREADS",
        "LVP_DEBUG",
        "MALLOC_CONF",
        "MATMUL_NUM_THREADS",
        "MESA_LOADER_DRIVER_OVERRIDE",
        "MESA_VK_DEVICE_SELECT",
        "MKL_DYNAMIC",
        "NODEVICE_SELECT",
        "NOUVEAU_USE_ZINK",
        "NVBLAS_GPU_LIST",
        "NVIDIA_TF32_OVERRIDE",
        "NVRTC_APPEND_FLAGS",
        "NVVM_IR_VER_CHK",
        "OMP_DYNAMIC",
        "OPENBLAS_CORETYPE",
        "ORT_LOAD_CONFIG_FROM_MODEL",
        "PYTHONPATH",
        "RADV_DEBUG",
        "RAYON_RS_NUM_CPUS",
        "RUST_MIN_STACK",
        "RUSTUP_DIST_SERVER",
        "TEMP",
        "TEMPDIR",
        "VECLIB_MAXIMUM_THREADS",
        "VK_ICD_FILENAMES",
        "VULKAN_SDK",
        "WGPU_ADAPTER_NAME",
        "WGPU_BACKEND",
        "XDG_CACHE_HOME",
        "ZINK_DEBUG",
        "mimalloc_eager_commit",
        "MiMaLlOc_Reserve_Huge_Os_Pages",
    ],
)
def test_measurement_environment_rejects_unreviewed_solver_runtime_control(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "1")

    with pytest.raises(
        provenance.ProvenanceError,
        match="unreviewed solver-runtime environment controls",
    ):
        provenance._capture_environment()


def test_measurement_environment_records_reviewed_solver_runtime_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    controls = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": "0",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "3",
        "ORT_LOG_SEVERITY_LEVEL": "4",
        "RAYON_NUM_THREADS": "5",
    }
    for key, value in controls.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    assert {key: environment["values"][key] for key in controls} == controls


@pytest.mark.parametrize(
    "value",
    [
        "",
        ":/opt/cuda/lib64",
        "/opt/cuda/lib64:",
        "/opt/cuda/lib64::/usr/lib",
        "relative/cuda",
        "/opt/cuda/lib64:relative/cuda",
    ],
)
def test_measurement_environment_rejects_unsafe_ld_library_path(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("LD_LIBRARY_PATH", value)

    with pytest.raises(provenance.ProvenanceError, match="LD_LIBRARY_PATH"):
        provenance._capture_environment()


def test_measurement_environment_accepts_dedicated_cuda_ld_library_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    first = tmp_path / "cuda-driver"
    second = tmp_path / "cuda-math"
    first.mkdir()
    second.mkdir()
    (first / "libcuda.so.1").write_bytes(b"driver")
    (second / "libcublas.so.13").write_bytes(b"cublas")
    value = f"{first}:{second}"
    monkeypatch.setenv("LD_LIBRARY_PATH", value)

    environment = provenance._capture_environment()

    assert environment["values"]["LD_LIBRARY_PATH"] == value


def test_measurement_environment_rejects_non_cuda_loader_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    loader_dir = tmp_path / "loader"
    loader_dir.mkdir()
    (loader_dir / "libcuda.so.1").write_bytes(b"driver")
    (loader_dir / "libc.so.6").write_bytes(b"injection")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(loader_dir))

    with pytest.raises(provenance.ProvenanceError, match="unsafe entry"):
        provenance._capture_environment()


def test_measurement_environment_rejects_system_loader_preload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    preload = tmp_path / "ld.so.preload"
    preload.write_text("/tmp/inject.so\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "SYSTEM_LD_SO_PRELOAD", preload)

    with pytest.raises(provenance.ProvenanceError, match="ld.so.preload"):
        provenance._capture_environment()


def _cuda_runtime_object(role: str, path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    file_stat = resolved.stat()
    fingerprint = provenance._file_fingerprint(resolved)
    return {
        "role": role,
        "provider_symbol": f"fixture_{role}_symbol",
        "mapped_path": str(resolved),
        "resolved_path": str(resolved),
        "mapped_device_major": os.major(file_stat.st_dev),
        "mapped_device_minor": os.minor(file_stat.st_dev),
        "mapped_inode": file_stat.st_ino,
        "size_bytes": file_stat.st_size,
        "sha256": provenance._sha256_file(resolved),
        "fingerprint": fingerprint,
    }


def _cuda_runtime_report(objects: list[dict[str, object]]) -> dict[str, object]:
    roles = {str(item["role"]) for item in objects}
    if "nvrtc" in roles and "nvrtc_builtins" in roles:
        nvrtc_status = "loaded_with_builtins"
    elif "nvrtc" in roles:
        nvrtc_status = "loaded"
    elif "nvrtc_builtins" in roles:
        nvrtc_status = "builtins_loaded_without_nvrtc"
    else:
        nvrtc_status = "not_loaded_feature_disabled"
    return {
        "schema": provenance.CUDA_RUNTIME_INFO_SCHEMA,
        "device_name": "fixture GPU",
        "pageable_host_ptr": False,
        "pageable_memory_access": True,
        "pageable_access_uses_host_page_tables": False,
        "integrated_device": False,
        "ordinary_gemm_transport": "explicit-device-copy",
        "ordinary_gemm_transport_policy": "auto",
        "ordinary_gemm_transport_reason": "discrete-device",
        "explicit_device_copy": True,
        "discrete_mode": True,
        "deadline_f64_transport": "explicit-device-copy",
        "candidates": {
            "driver": ["libcuda.so", "libcuda.so.1"],
            "cublas": ["libcublas.so", "libcublas.so.13"],
            "cublas_lt": ["libcublasLt.so", "libcublasLt.so.13"],
            "nvrtc": ["libnvrtc.so", "libnvrtc.so.13"],
        },
        "objects": objects,
        "nvrtc_status": nvrtc_status,
    }


def _stub_cuda_runtime_probe(
    monkeypatch: pytest.MonkeyPatch,
    reports: list[dict[str, object]],
) -> None:
    remaining = iter(reports)

    def fake_run(command, **_kwargs):
        assert command[-1] == "--cuda-runtime-info"
        report = next(remaining)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(json.dumps(report) + "\n").encode(),
            stderr=b"",
        )

    monkeypatch.setattr(provenance, "_run", fake_run)


@pytest.mark.parametrize(
    ("ordinary_transport", "discrete_mode"),
    [
        (None, False),
        ("explicit-device-copy", False),
        ("unified-memory", True),
        ("unknown-transport", False),
    ],
)
def test_cuda_runtime_probe_rejects_missing_or_inconsistent_ordinary_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ordinary_transport: str | None,
    discrete_mode: bool,
) -> None:
    report = _cuda_runtime_report([])
    if ordinary_transport is None:
        report.pop("ordinary_gemm_transport")
    else:
        report["ordinary_gemm_transport"] = ordinary_transport
    report["discrete_mode"] = discrete_mode
    _stub_cuda_runtime_probe(monkeypatch, [report])

    with pytest.raises(provenance.ProvenanceError, match="qualification is incomplete"):
        provenance._cuda_runtime_probe(tmp_path / "ny")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ordinary_gemm_transport_policy", "unknown-policy"),
        ("ordinary_gemm_transport_reason", "unknown-reason"),
        ("explicit_device_copy", False),
        ("integrated_device", True),
        ("integrated_device", 0),
        ("pageable_access_uses_host_page_tables", True),
        ("ordinary_gemm_transport_reason", "explicit-transport-override"),
        ("deadline_f64_transport", "direct-host-page-tables"),
    ],
)
def test_cuda_runtime_probe_rejects_inconsistent_auto_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    report = _cuda_runtime_report([])
    report[field] = value
    _stub_cuda_runtime_probe(monkeypatch, [report])

    with pytest.raises(provenance.ProvenanceError, match="qualification is incomplete"):
        provenance._cuda_runtime_probe(tmp_path / "ny")


def test_cuda_runtime_identity_hashes_required_and_optional_mapped_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        role: tmp_path / name
        for role, name in {
            "driver": "libcuda.so.1",
            "cublas": "libcublas.so.13",
            "cublas_lt": "libcublasLt.so.13",
            "nvrtc": "libnvrtc.so.13",
            "nvrtc_builtins": "libnvrtc-builtins.so.13.2",
        }.items()
    }
    for role, path in paths.items():
        path.write_bytes(f"{role} fixture\n".encode())
    report = _cuda_runtime_report(
        [_cuda_runtime_object(role, path) for role, path in paths.items()]
    )
    _stub_cuda_runtime_probe(monkeypatch, [report, report])

    identity = provenance._capture_cuda_runtime_identity(tmp_path / "ny")

    assert identity["schema"] == provenance.MEASUREMENT_CUDA_RUNTIME_SCHEMA
    assert identity["status"] == "captured"
    assert identity["probe"]["nvrtc_status"] == "loaded_with_builtins"
    captured = {item["role"]: item for item in identity["objects"]}
    assert set(captured) == set(paths)
    for role, path in paths.items():
        assert captured[role]["sha256"] == provenance._sha256_file(path)
        assert captured[role]["fingerprint"]["inode"] == path.stat().st_ino


def test_cuda_runtime_identity_records_nvrtc_as_not_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        role: tmp_path / name
        for role, name in {
            "driver": "libcuda.so.1",
            "cublas": "libcublas.so.13",
            "cublas_lt": "libcublasLt.so.13",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"fixture\n")
    report = _cuda_runtime_report(
        [_cuda_runtime_object(role, path) for role, path in paths.items()]
    )
    _stub_cuda_runtime_probe(monkeypatch, [report, report])

    identity = provenance._capture_cuda_runtime_identity(tmp_path / "ny")

    assert identity["probe"]["nvrtc_status"] == "not_loaded_feature_disabled"
    assert {item["role"] for item in identity["objects"]} == set(paths)


def test_cuda_runtime_probe_rejects_missing_transitive_cublas_lt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = tmp_path / "libcuda.so.1"
    cublas = tmp_path / "libcublas.so.13"
    driver.write_bytes(b"driver\n")
    cublas.write_bytes(b"cublas\n")
    report = _cuda_runtime_report(
        [
            _cuda_runtime_object("driver", driver),
            _cuda_runtime_object("cublas", cublas),
        ]
    )
    _stub_cuda_runtime_probe(monkeypatch, [report])

    with pytest.raises(provenance.ProvenanceError, match="cublas_lt"):
        provenance._capture_cuda_runtime_identity(tmp_path / "ny")


def test_cuda_runtime_identity_rejects_mapped_inode_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        role: tmp_path / name
        for role, name in {
            "driver": "libcuda.so.1",
            "cublas": "libcublas.so.13",
            "cublas_lt": "libcublasLt.so.13",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"fixture\n")
    objects = [
        _cuda_runtime_object(role, path) for role, path in paths.items()
    ]
    objects[0]["mapped_inode"] = int(objects[0]["mapped_inode"]) + 1
    objects[0]["fingerprint"]["inode"] = int(
        objects[0]["fingerprint"]["inode"]
    ) + 1
    report = _cuda_runtime_report(objects)
    _stub_cuda_runtime_probe(monkeypatch, [report])

    with pytest.raises(provenance.ProvenanceError, match="changed before"):
        provenance._capture_cuda_runtime_identity(tmp_path / "ny")


def test_cuda_runtime_identity_rejects_selection_change_during_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        role: tmp_path / name
        for role, name in {
            "driver": "libcuda.so.1",
            "cublas": "libcublas.so.13",
            "cublas_lt": "libcublasLt.so.13",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"fixture\n")
    first = _cuda_runtime_report(
        [_cuda_runtime_object(role, path) for role, path in paths.items()]
    )
    second = json.loads(json.dumps(first))
    second["ordinary_gemm_transport_policy"] = "override-explicit-device-copy"
    second["ordinary_gemm_transport_reason"] = "explicit-transport-override"
    _stub_cuda_runtime_probe(monkeypatch, [first, second])

    with pytest.raises(provenance.ProvenanceError, match="selection changed"):
        provenance._capture_cuda_runtime_identity(tmp_path / "ny")


def _fake_containment_tree(
    tmp_path: Path,
    *,
    expected_cpus: int = 10,
) -> tuple[Path, Path, Path, Path, str]:
    assert expected_cpus in {10, 20}
    uid = os.getuid()
    cgroup_root = tmp_path / "cgroup"
    policy_membership = (
        f"/user.slice/user-{uid}.slice/user@{uid}.service/ny.slice/ny-build.slice"
    )
    membership = f"{policy_membership}/ny-safe-gpu-{uid}-1234-5678.service"
    policy = cgroup_root / policy_membership.lstrip("/")
    current = cgroup_root / membership.lstrip("/")
    current.mkdir(parents=True)
    (cgroup_root / "cgroup.controllers").write_text(
        "cpu memory pids\n", encoding="ascii"
    )
    policy_values = {
        "memory.high": str(provenance.EXPECTED_MEMORY_HIGH_BYTES),
        "memory.max": str(provenance.EXPECTED_MEMORY_MAX_BYTES),
        "memory.swap.max": str(provenance.EXPECTED_MEMORY_SWAP_MAX_BYTES),
        "pids.max": str(provenance.EXPECTED_PIDS_MAX),
        "cpu.max": f"{expected_cpus * 100_000} 100000",
    }
    for name, value in policy_values.items():
        parent_value = "max 100000" if name == "cpu.max" else "max"
        (policy / name).write_text(f"{parent_value}\n", encoding="ascii")
        (current / name).write_text(f"{value}\n", encoding="ascii")

    user_slice = cgroup_root / f"user.slice/user-{uid}.slice"
    for name, value in {
        "memory.high": "max",
        "memory.max": "max",
        "memory.swap.max": "max",
        "pids.max": "337805",
        "cpu.max": "max 100000",
    }.items():
        (user_slice / name).write_text(f"{value}\n", encoding="ascii")

    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text(f"0::{membership}\n", encoding="ascii")
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"39 29 0:33 / {cgroup_root} rw,nosuid,nodev - cgroup2 cgroup2 rw\n",
        encoding="ascii",
    )
    return proc_cgroup, proc_mountinfo, cgroup_root, current, policy_membership


def _stub_resource_limit_as(
    monkeypatch: pytest.MonkeyPatch,
    limits: tuple[int, int],
) -> list[None]:
    calls: list[None] = []

    def read_limit() -> tuple[int, int]:
        calls.append(None)
        return limits

    monkeypatch.setattr(provenance, "_resource_limit_as", read_limit)
    return calls


def _capture(
    repo: Path,
    benchmark_root: Path,
    *,
    run_id: str,
    configs_dir: Path | None = None,
    require_execution_predicate: bool = True,
) -> Path:
    injected = (
        require_execution_predicate
        and "NY_MEASURE_REQUIRE_FLIGHT_EVENT" not in os.environ
    )
    if injected:
        os.environ["NY_MEASURE_REQUIRE_FLIGHT_EVENT"] = CURRENT_EXECUTION_PREDICATE
    try:
        return provenance.capture_start_manifest(
            repo_root=repo,
            binary=Path("target/release/ny"),
            benchmark_root=benchmark_root,
            artifact_root=Path("reports/measured/artifacts"),
            run_id=run_id,
            output_dir=Path("reports/measured"),
            scratch_dir=repo.parent / "scratch",
            result_file=repo.parent / "scratch" / "result.txt",
            solver_log_file=repo.parent / "scratch" / "solver.log",
            categories_raw="demo other_demo",
            timeout_cap_seconds=120,
            watchdog_grace_seconds=30,
            max_rows_per_category=0,
            instance_index=0,
            vnnlib_version="",
            sweep_script=Path("scripts/measure_ny_scorecard.sh"),
            configs_dir=configs_dir,
        )
    finally:
        if injected:
            os.environ.pop("NY_MEASURE_REQUIRE_FLIGHT_EVENT", None)


def _seed_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
) -> tuple[Path, Path, Path, Path, Path]:
    repo, benchmark_root = _measurement_repos(tmp_path)
    onnx_file = benchmark_root / "model.onnx"
    vnnlib_file = benchmark_root / "property.vnnlib"
    onnx_file.write_bytes(b"model fixture\n")
    vnnlib_file.write_text("; property fixture\n", encoding="utf-8")
    _run("git", "add", ".", cwd=benchmark_root.parent)
    _run("git", "commit", "-qm", "add measurement inputs", cwd=benchmark_root.parent)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id=run_id)
    start_payload = json.loads(start.read_text(encoding="utf-8"))
    flight_build_stamp = provenance.flight_build_stamp_for_binary_build_id(
        start_payload["solver_binary"]["build_id"]
    )
    result_file = tmp_path / "result.txt"
    solver_log_file = tmp_path / "solver.log"
    flight_file = Path(f"{result_file}.flight.json")
    result_file.write_bytes(b"unsat\n")
    solver_log_file.write_bytes(b"solver log\n")
    flight_file.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "build_id": flight_build_stamp,
                "backend_kind": "cpu-only",
                "backend_summary": "fixture CPU backend",
                "host": {
                    "hostname": "fixture-host",
                    "cpu_model": "fixture-cpu",
                    "logical_cores": 1,
                    "ram_bytes": 1_073_741_824,
                },
                "category": "demo",
                "budget_secs": 120,
                "ambient_env_schema": "ny-cli/ambient-env/v3",
                "ambient_env": {},
                "levers": {
                    "status": "resolved",
                    "receipt": {
                        "schema": "ny-levers/receipt/v3",
                        "lever_count": 1,
                        "env_present": 0,
                        "env_accepted": 0,
                        "env_rejected": 0,
                        "env_ignored": 0,
                        "levers": [
                            {
                                "name": "NY_PGD_FAST_GRAD",
                                "value": False,
                                "source": "config",
                                "bucket": "debug",
                                "moat": "high",
                                "provenance": "unmeasured",
                            }
                        ],
                    },
                },
                "events": [
                    {
                        "method": "pgd_fast_gradient_execution",
                        "status": "ran",
                        "reason": (
                            "schema=ny-cli/pgd-fast-gradient-execution/v1 "
                            "phase=terminal route=conjunctive configured=false "
                            "decisions=1 fast_attempts=0 fast_selected=0 "
                            "certified_selected=1 spsa_selected=0"
                        ),
                        "at_secs": 0.01,
                    },
                    {
                        "method": "run_complete",
                        "status": "complete",
                        "reason": "unsat",
                        "at_secs": 9.0,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_root = repo / "reports/measured/artifacts"
    preflight_manifest = archive.seal_inputs(
        artifact_root=artifact_root,
        run_id=run_id,
        category="demo",
        instance_index=1,
        onnx="model.onnx",
        vnnlib="property.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        start_manifest=start,
    )
    archived_result = archive.archive_result(
        result_file=result_file,
        solver_log_file=solver_log_file,
        artifact_root=artifact_root,
        run_id=run_id,
        category="demo",
        instance_index=1,
        onnx="model.onnx",
        vnnlib="property.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        solver_verdict="unsat",
        solver_exit_status=0,
        timeout_seconds=120,
        elapsed_seconds=9,
        source_csv="reports/measured/demo.csv",
        start_manifest=start,
        preflight_manifest=preflight_manifest,
        flight_file=flight_file,
    )
    csv_path = repo / "reports/measured/demo.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        f"demo,model.onnx,property.vnnlib,0,unsat,9,{run_id}\n",
        encoding="utf-8",
    )
    metadata_path = archived_result.parent / f"{run_id}.json"
    solver_log_path = archived_result.parent / f"{run_id}.solver.log"
    cache_path = start.with_name("input_hash_cache.json")
    return start, metadata_path, archived_result, solver_log_path, cache_path


def test_seal_file_refuses_content_matching_writable_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "sealed"
    contents = b"#!/bin/sh\nexit 0\n"
    source.write_bytes(contents)
    source.chmod(0o755)
    destination.write_bytes(contents)
    destination.chmod(0o755)

    with pytest.raises(provenance.ProvenanceError, match="sealed file is writable"):
        provenance._seal_file(source, destination, executable=True)

    assert destination.read_bytes() == contents
    assert destination.stat().st_mode & 0o777 == 0o755


def test_completion_rejects_post_start_writable_sealed_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id="writable-sealed-solver")
    start_record = json.loads(start.read_text(encoding="utf-8"))
    sealed_path = Path(start_record["solver_binary"]["sealed_execution"]["path"])
    sealed_path.chmod(0o755)

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    violations = record["integrity"]["violations"]

    assert record["completed_successfully"] is False
    assert record["integrity"]["checks"]["sealed_solver_binary"]["status"] == "invalid"
    assert any(
        item["code"] == "sealed_solver_binary_unavailable"
        and "sealed file is writable" in item["detail"]
        for item in violations
    )


@pytest.mark.parametrize(
    ("selector", "expected_cpus"),
    [(None, 10), ("20", 20)],
)
def test_containment_capture_resolves_and_types_selected_cpu_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str | None,
    expected_cpus: int,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, current, _policy_membership = (
        _fake_containment_tree(tmp_path, expected_cpus=expected_cpus)
    )
    if selector is not None:
        monkeypatch.setenv("NY_MEASURE_EXPECTED_CPUS", selector)
    rlimit_calls = _stub_resource_limit_as(
        monkeypatch,
        (
            provenance.EXPECTED_RLIMIT_AS_BYTES,
            provenance.EXPECTED_RLIMIT_AS_BYTES,
        ),
    )

    containment = REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
        proc_self_cgroup=proc_cgroup,
        proc_self_mountinfo=proc_mountinfo,
        cgroup_root=cgroup_root,
    )

    assert rlimit_calls == [None]
    assert containment["schema"] == "ny_measurement_containment_v1"
    assert containment["membership"].endswith(
        f"/ny-safe-gpu-{os.getuid()}-1234-5678.service"
    )
    assert containment["current_cgroup"] == str(current.resolve())
    assert containment["policy_cgroup"] == str(current.resolve())
    assert containment["effective"]["memory.high"]["value_bytes"] == 64 * 1024**3
    assert containment["effective"]["memory.max"]["value_bytes"] == 80 * 1024**3
    assert containment["effective"]["memory.swap.max"]["value_bytes"] == 8 * 1024**3
    assert containment["effective"]["pids.max"]["value"] == 4096
    assert containment["effective"]["cpu.max"]["equivalent_cpus"] == expected_cpus
    assert containment["policy"]["cpu.max"] == {
        "raw": f"{expected_cpus * 100_000} 100000",
        "quota_us": expected_cpus * 100_000,
        "period_us": 100_000,
        "equivalent_cpus": expected_cpus,
    }
    assert containment["rlimit_as"] == {
        "soft_bytes": 160 * 1024**3,
        "hard_bytes": 160 * 1024**3,
    }


@pytest.mark.parametrize(
    ("control", "wrong"),
    [
        ("memory.high", str(63 * 1024**3)),
        ("memory.max", str(79 * 1024**3)),
        ("memory.swap.max", str(7 * 1024**3)),
        ("pids.max", "4095"),
        ("cpu.max", "900000 100000"),
    ],
)
def test_containment_capture_rejects_policy_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    wrong: str,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, current, _policy_membership = (
        _fake_containment_tree(tmp_path)
    )
    (current / control).write_text(f"{wrong}\n", encoding="ascii")

    with pytest.raises(provenance.ProvenanceError, match="reviewed policy"):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=proc_mountinfo,
            cgroup_root=cgroup_root,
        )


def test_containment_capture_rejects_selected_cpu_policy_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, _current, _policy_membership = (
        _fake_containment_tree(tmp_path, expected_cpus=10)
    )
    monkeypatch.setenv("NY_MEASURE_EXPECTED_CPUS", "20")

    with pytest.raises(provenance.ProvenanceError, match="reviewed policy"):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=proc_mountinfo,
            cgroup_root=cgroup_root,
        )


@pytest.mark.parametrize(
    ("control", "wrong", "message"),
    [
        ("memory.max", str(70 * 1024**3), "effective memory.max"),
        ("pids.max", "2048", "effective pids.max"),
        ("cpu.max", "500000 100000", "effective cpu.max"),
        ("cpu.max", "forged", "malformed cpu.max"),
    ],
)
def test_containment_capture_rejects_tighter_or_malformed_ancestor_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    wrong: str,
    message: str,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, _current, policy_membership = (
        _fake_containment_tree(tmp_path)
    )
    policy = cgroup_root / policy_membership.lstrip("/")
    (policy / control).write_text(f"{wrong}\n", encoding="ascii")
    rlimit_calls = _stub_resource_limit_as(
        monkeypatch,
        (
            provenance.EXPECTED_RLIMIT_AS_BYTES,
            provenance.EXPECTED_RLIMIT_AS_BYTES,
        ),
    )

    with pytest.raises(provenance.ProvenanceError, match=message):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=proc_mountinfo,
            cgroup_root=cgroup_root,
        )
    assert rlimit_calls == [None]


@pytest.mark.parametrize(
    ("soft_delta", "hard_delta"),
    [(-1, 0), (0, -1)],
)
def test_containment_capture_rejects_either_rlimit_as_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    soft_delta: int,
    hard_delta: int,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, _current, _policy_membership = (
        _fake_containment_tree(tmp_path)
    )
    expected = provenance.EXPECTED_RLIMIT_AS_BYTES
    rlimit_calls = _stub_resource_limit_as(
        monkeypatch,
        (expected + soft_delta, expected + hard_delta),
    )

    with pytest.raises(provenance.ProvenanceError, match="RLIMIT_AS"):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=proc_mountinfo,
            cgroup_root=cgroup_root,
        )
    assert rlimit_calls == [None]


def test_containment_capture_rejects_wrong_membership_and_ambiguous_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_cgroup, proc_mountinfo, cgroup_root, _current, _policy_membership = (
        _fake_containment_tree(tmp_path)
    )
    proc_cgroup.write_text("0::/ny-build.slice/forged.scope\n", encoding="ascii")
    with pytest.raises(provenance.ProvenanceError, match="exact ny-safe-gpu"):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=proc_mountinfo,
            cgroup_root=cgroup_root,
        )

    _proc_cgroup, duplicate_mountinfo, _root, _current, _policy = (
        _fake_containment_tree(tmp_path / "duplicate")
    )
    duplicate_mountinfo.write_text(
        duplicate_mountinfo.read_text(encoding="ascii") * 2,
        encoding="ascii",
    )
    with pytest.raises(provenance.ProvenanceError, match="exactly one cgroup-v2"):
        REAL_CAPTURE_MEASUREMENT_CONTAINMENT(
            proc_self_cgroup=_proc_cgroup,
            proc_self_mountinfo=duplicate_mountinfo,
            cgroup_root=_root,
        )


def test_start_manifest_hashes_worktree_and_excludes_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    (repo / "notes.txt").write_text("untracked evidence\n", encoding="utf-8")
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MEASURE_CAP", "120")
    monkeypatch.setenv("NY_MARGIN_ROW_ADAPTIVE_RESERVE", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_DOMAIN_STACK", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_CLASSWISE", "1")
    monkeypatch.setenv("NY_ROOT_GEMM", "faer")
    monkeypatch.setenv("NY_WARMUP_ITERS", "03")
    monkeypatch.setenv("NY_CROWN_CUT_SEGMENT", "0")
    monkeypatch.setenv("NY_INPUT_SPLIT_WARM_PARALLEL", "1")
    monkeypatch.setenv("GITHUB_TOKEN", "must-never-appear")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest_path = _capture(repo, benchmark_root, run_id="run-one")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert regular_evidence.validate_start_schema_profile(manifest) == (
        regular_evidence.CURRENT_EXECUTION_START_PROFILE
    )

    assert manifest["schema"] == "ny_measurement_start_v1"
    assert manifest["ny"]["clean"] is False
    assert manifest["ny"]["untracked_files"] == [
        {
            "kind": "file",
            "path": "notes.txt",
            "sha256": provenance._sha256(b"untracked evidence\n"),
            "size_bytes": len(b"untracked evidence\n"),
        }
    ]
    assert len(manifest["ny"]["worktree_evidence_sha256"]) == 64
    assert manifest["solver_binary"]["sha256"] == provenance._sha256_file(
        repo / "target/release/ny"
    )
    assert manifest["solver_binary"]["declared_build_features"] == ["mip", "cuda"]
    assert manifest["dependencies"]["ay"]["git_revision"] == AY_REV
    assert manifest["dependencies"]["cuda_runtime"] == CUDA_RUNTIME_FIXTURE
    assert manifest["rust_toolchain"]["channel"] == "1.95.0"
    assert manifest["rust_toolchain"]["probe_tool"]["sha256"]
    assert manifest["provenance_tools"]["git"]["sha256"]
    assert manifest["benchmark"]["remotes"] == [
        {
            "name": "origin",
            "fetch_url": "https://github.com/example/bench.git",
        }
    ]
    assert manifest["measurement"]["timeout_cap_seconds"] == 120
    assert manifest["measurement"]["categories"] == ["demo", "other_demo"]
    assert manifest["measurement"]["execution_evidence_profile"] == (
        provenance.EXECUTION_EVIDENCE_PROFILE
    )
    assert manifest["environment"]["values"]["NY_MEASURE_CAP"] == "120"
    assert manifest["environment"]["values"]["NY_MARGIN_ROW_ADAPTIVE_RESERVE"] == "1"
    assert manifest["environment"]["values"]["NY_MARGIN_ROW_DOMAIN_STACK"] == "1"
    assert manifest["environment"]["values"]["NY_MARGIN_ROW_CLASSWISE"] == "1"
    assert manifest["environment"]["values"]["NY_ROOT_GEMM"] == "faer"
    assert manifest["environment"]["values"]["NY_WARMUP_ITERS"] == "03"
    assert manifest["environment"]["values"]["NY_CROWN_CUT_SEGMENT"] == "0"
    assert manifest["environment"]["values"]["NY_INPUT_SPLIT_WARM_PARALLEL"] == "1"
    assert manifest["environment"]["typed_values"] == {
        "NY_CROWN_CUT_SEGMENT": {
            "type": "nonnegative_integer",
            "value": 0,
        },
        "NY_MARGIN_ROW_ADAPTIVE_RESERVE": {
            "type": "boolean",
            "value": True,
        },
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT": {
            "schema": "ny-measure/execution-predicate-requirement/v1",
            "type": "execution_predicate_requirement",
            "predicate_id": "ny-exec-v1.pgd-fast-gradient-parameter-engaged",
            "axis": "NY_PGD_FAST_GRAD",
            "requested_value": "0",
            "claim": "parameter_engaged",
        },
        "NY_WARMUP_ITERS": {
            "type": "nonnegative_integer",
            "value": 3,
        },
    }
    assert manifest["host"]["containment"] == CONTAINMENT_FIXTURE
    assert manifest["host_state"] == HOST_STATE_FIXTURE
    assert b"must-never-appear" not in manifest_bytes
    assert b"supersecret" not in manifest_bytes
    assert b"token=hidden" not in manifest_bytes


def test_start_manifest_refuses_current_evidence_without_a_predicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)

    with pytest.raises(
        provenance.ProvenanceError,
        match="current measurement evidence requires a typed",
    ):
        _capture(
            repo,
            benchmark_root,
            run_id="missing-execution-predicate",
            require_execution_predicate=False,
        )


def test_pinned_toolchain_version_probe_must_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "rust-toolchain.toml").write_text(
        '[toolchain]\nchannel = "1.95.0"\n',
        encoding="utf-8",
    )

    def fail_probe(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=b"",
            stderr=b"missing pinned toolchain",
        )

    monkeypatch.setattr(provenance, "_run", fail_probe)
    true_executable = shutil.which("true")
    assert true_executable is not None, "true is required by this fixture"
    with pytest.raises(
        provenance.ProvenanceError,
        match="pinned rustc version probe failed",
    ):
        provenance._parse_toolchain(
            repo,
            declared_tool_path=true_executable,
            declared_tool_kind="rustc",
        )


def test_untracked_content_changes_worktree_evidence_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    note = repo / "notes.txt"
    note.write_text("one\n", encoding="utf-8")
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    first = json.loads(_capture(repo, benchmark_root, run_id="run-one").read_text())
    note.write_text("two\n", encoding="utf-8")
    second = json.loads(_capture(repo, benchmark_root, run_id="run-two").read_text())

    assert (
        first["ny"]["worktree_evidence_sha256"]
        != second["ny"]["worktree_evidence_sha256"]
    )


def test_clean_worktree_keeps_zero_length_tracked_diff_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(_capture(repo, benchmark_root, run_id="clean").read_text())

    for worktree in (manifest["ny"], manifest["benchmark"]):
        assert worktree["clean"] is True
        assert worktree["tracked_diff_format"] == "ny_tracked_worktree_evidence_v2"
        assert worktree["tracked_diff_bytes"] == 0
        assert worktree["tracked_diff_sha256"] == provenance._sha256(b"")
        assert worktree["tracked_worktree_paths"] == []


def test_tracked_worktree_evidence_hashes_files_symlinks_and_large_deletions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    benchmark_repo = benchmark_root.parent
    archive = benchmark_root / "archive.bin"
    config = benchmark_root / "config.txt"
    link = benchmark_root / "current"
    archive.write_bytes(b"x" * (2 * 1024 * 1024))
    config.write_text("committed\n", encoding="utf-8")
    link.symlink_to("old-target")
    _run("git", "add", ".", cwd=benchmark_repo)
    _run("git", "commit", "-qm", "add tracked states", cwd=benchmark_repo)

    archive.unlink()
    config.write_text("staged\n", encoding="utf-8")
    _run("git", "add", "benchmarks/config.txt", cwd=benchmark_repo)
    config.write_text("worktree\n", encoding="utf-8")
    config.chmod(0o755)
    link.unlink()
    link.symlink_to("new-target")

    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="tracked-states").read_text()
    )
    benchmark = manifest["benchmark"]
    entries = {entry["path"]: entry for entry in benchmark["tracked_worktree_paths"]}

    assert benchmark["clean"] is False
    assert benchmark["tracked_diff_format"] == "ny_tracked_worktree_evidence_v2"
    assert benchmark["tracked_diff_bytes"] < 16 * 1024
    assert entries["benchmarks/archive.bin"] == {
        "kind": "missing",
        "path": "benchmarks/archive.bin",
    }
    assert entries["benchmarks/config.txt"] == {
        "kind": "file",
        "mode": 0o755,
        "path": "benchmarks/config.txt",
        "sha256": provenance._sha256(b"worktree\n"),
        "size_bytes": len(b"worktree\n"),
    }
    target = os.fsencode("new-target")
    assert entries["benchmarks/current"] == {
        "kind": "symlink",
        "mode": 0o777,
        "path": "benchmarks/current",
        "sha256": provenance._sha256(target),
        "size_bytes": len(target),
    }


def test_tracked_worktree_path_capture_rejects_parent_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    with pytest.raises(provenance.ProvenanceError, match="unsafe tracked worktree"):
        provenance._tracked_path_state(repo, "../outside")


def test_tracked_worktree_capture_forces_filemode_detection(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    script = repo / "script.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    _run("git", "add", "script.sh", cwd=repo)
    _run("git", "commit", "-qm", "script", cwd=repo)
    _run("git", "config", "core.fileMode", "false", cwd=repo)
    script.chmod(0o755)

    evidence, entries = provenance._tracked_worktree_evidence(repo)

    assert evidence
    assert entries == [
        {
            "kind": "file",
            "mode": 0o755,
            "path": "script.sh",
            "sha256": provenance._sha256(b"#!/bin/sh\n"),
            "size_bytes": len(b"#!/bin/sh\n"),
        }
    ]


def test_staged_intermediate_change_alters_evidence_with_fixed_worktree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    path = repo / "state.txt"
    path.write_text("committed\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repo)
    _run("git", "commit", "-qm", "state", cwd=repo)

    path.write_text("staged-a\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repo)
    path.write_text("final-b\n", encoding="utf-8")
    first, first_entries = provenance._tracked_worktree_evidence(repo)

    path.write_text("staged-a2\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repo)
    path.write_text("final-b\n", encoding="utf-8")
    second, second_entries = provenance._tracked_worktree_evidence(repo)

    assert first_entries == second_entries
    assert first != second


def test_staged_add_delete_and_rename_bind_final_states(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    renamed = repo / "renamed.txt"
    deleted = repo / "deleted.txt"
    renamed.write_text("rename-content\n", encoding="utf-8")
    deleted.write_text("delete-content\n", encoding="utf-8")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "base", cwd=repo)
    renamed.rename(repo / "new-name.txt")
    deleted.unlink()
    (repo / "added.txt").write_text("added-content\n", encoding="utf-8")
    _run("git", "add", "-A", cwd=repo)

    evidence, entries = provenance._tracked_worktree_evidence(repo)
    by_path = {entry["path"]: entry for entry in entries}

    assert evidence
    assert set(by_path) == {
        "added.txt",
        "deleted.txt",
        "new-name.txt",
        "renamed.txt",
    }
    assert by_path["deleted.txt"]["kind"] == "missing"
    assert by_path["renamed.txt"]["kind"] == "missing"
    assert by_path["added.txt"]["sha256"] == provenance._sha256(b"added-content\n")
    assert by_path["new-name.txt"]["sha256"] == provenance._sha256(b"rename-content\n")


@pytest.mark.parametrize(
    "index_flag",
    ["--assume-unchanged", "--skip-worktree"],
)
def test_hidden_index_flags_fail_closed(
    tmp_path: Path,
    index_flag: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    path = repo / "hidden.txt"
    path.write_text("committed\n", encoding="utf-8")
    _run("git", "add", "hidden.txt", cwd=repo)
    _run("git", "commit", "-qm", "hidden", cwd=repo)
    _run("git", "update-index", index_flag, "hidden.txt", cwd=repo)
    path.write_text("hidden mutation\n", encoding="utf-8")

    with pytest.raises(provenance.ProvenanceError, match="unsupported Git"):
        provenance._tracked_worktree_evidence(repo)


def test_fsmonitor_valid_index_flag_parser_fails_closed() -> None:
    with pytest.raises(provenance.ProvenanceError, match="fsmonitor"):
        provenance._parse_index_flags(b"h tracked.txt\0", label="fsmonitor")


def test_git_environment_cannot_redirect_worktree_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    alternate = tmp_path / "alternate"
    for repo, content in ((target, "target\n"), (alternate, "alternate\n")):
        repo.mkdir()
        _init_git_repo(repo)
        (repo / "state.txt").write_text(content, encoding="utf-8")
        _run("git", "add", "state.txt", cwd=repo)
        _run("git", "commit", "-qm", "state", cwd=repo)
    (target / "state.txt").write_text("target-dirty\n", encoding="utf-8")
    monkeypatch.setenv("GIT_DIR", str(alternate / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(alternate))
    monkeypatch.setenv("GIT_INDEX_FILE", str(alternate / ".git" / "index"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(alternate))

    evidence, entries = provenance._tracked_worktree_evidence(target)

    assert evidence
    assert entries == [
        {
            "kind": "file",
            "mode": (target / "state.txt").stat().st_mode & 0o7777,
            "path": "state.txt",
            "sha256": provenance._sha256(b"target-dirty\n"),
            "size_bytes": len(b"target-dirty\n"),
        }
    ]


def test_replace_ref_cannot_hide_staged_and_worktree_change(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    path = repo / "state.txt"
    path.write_text("original\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repo)
    _run("git", "commit", "-qm", "original", cwd=repo)
    original = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    path.write_text("replacement\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repo)
    _run("git", "commit", "-qm", "replacement", cwd=repo)
    replacement = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch_ref = subprocess.run(
        ["git", "symbolic-ref", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run("git", "update-ref", branch_ref, original, cwd=repo)
    _run("git", "replace", original, replacement, cwd=repo)

    evidence, entries = provenance._tracked_worktree_evidence(repo)

    assert evidence
    assert entries == [
        {
            "kind": "file",
            "mode": path.stat().st_mode & 0o7777,
            "path": "state.txt",
            "sha256": provenance._sha256(b"replacement\n"),
            "size_bytes": len(b"replacement\n"),
        }
    ]


@pytest.mark.parametrize(
    ("stage_record", "message"),
    [
        (
            b"100644 " + b"a" * 40 + b" 2\tconflict.txt\0",
            "unmerged",
        ),
        (
            b"160000 " + b"a" * 40 + b" 0\tsubmodule\0",
            "Gitlink",
        ),
    ],
)
def test_unmerged_and_gitlink_index_entries_fail_closed(
    stage_record: bytes,
    message: str,
) -> None:
    with pytest.raises(provenance.ProvenanceError, match=message):
        provenance._parse_index_stage(stage_record, object_format="sha1")


def test_tracked_special_file_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    fifo = repo / "tracked.fifo"
    os.mkfifo(fifo)

    with pytest.raises(provenance.ProvenanceError, match="special entry"):
        provenance._tracked_path_state(repo, "tracked.fifo")


def test_ancestor_symlink_escape_is_never_followed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    _init_git_repo(repo)
    tracked = repo / "tracked"
    tracked.mkdir()
    (tracked / "secret").write_text("inside\n", encoding="utf-8")
    (outside / "secret").write_text("must-not-be-read\n", encoding="utf-8")
    tracked.rename(repo / "tracked-original")
    (repo / "tracked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(provenance.ProvenanceError, match="safely open"):
        provenance._tracked_path_state(repo, "tracked/secret")


def test_final_file_to_symlink_swap_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.write_text("must-not-be-read\n", encoding="utf-8")
    _init_git_repo(repo)
    path = repo / "state"
    path.write_text("inside\n", encoding="utf-8")
    real_open = provenance.os.open
    swapped = False

    def swap_before_open(
        candidate: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if candidate == b"state" and dir_fd is not None and not swapped:
            swapped = True
            path.rename(repo / "state-original")
            path.symlink_to(outside)
        return real_open(candidate, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(provenance.os, "open", swap_before_open)
    with pytest.raises(provenance.ProvenanceError, match="safely open"):
        provenance._tracked_path_state(repo, "state")


def _assert_byte_paths_are_deterministic(
    tmp_path: Path,
    raw_names: list[bytes],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    root_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, name in enumerate(raw_names):
            try:
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                    dir_fd=root_fd,
                )
            except OSError as error:
                if error.errno == errno.EILSEQ:
                    raise AssertionError(
                        "the selected filesystem cannot exercise the required "
                        "non-UTF-8 provenance contract"
                    ) from error
                raise
            try:
                os.write(descriptor, f"value-{index}\n".encode())
            finally:
                os.close(descriptor)
    finally:
        os.close(root_fd)

    first = provenance._tracked_path_states(repo, raw_names)
    second = provenance._tracked_path_states(repo, list(reversed(raw_names)))

    assert first == second
    assert [os.fsencode(entry["path"]) for entry in first] == sorted(raw_names)


def test_byte_paths_are_deterministic_and_not_normalized(tmp_path: Path) -> None:
    _assert_byte_paths_are_deterministic(
        tmp_path,
        [b"-leading", b"line\nbreak"],
    )


def external_non_utf8_byte_paths_are_deterministic_and_not_normalized(
    tmp_path: Path,
) -> None:
    _assert_byte_paths_are_deterministic(tmp_path, [b"nonutf8-\xff"])


def test_many_distinct_parents_keep_open_fds_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    paths: list[bytes] = []
    for index in range(512):
        relative = f"dir-{index:04d}"
        (repo / relative).mkdir()
        paths.append(os.fsencode(f"{relative}/missing"))

    real_open = provenance.os.open
    real_close = provenance.os.close
    live = 0
    max_live = 0

    def counted_open(*args: object, **kwargs: object) -> int:
        nonlocal live, max_live
        descriptor = real_open(*args, **kwargs)
        live += 1
        max_live = max(max_live, live)
        return descriptor

    def counted_close(descriptor: int) -> None:
        nonlocal live
        live -= 1
        real_close(descriptor)

    monkeypatch.setattr(provenance.os, "open", counted_open)
    monkeypatch.setattr(provenance.os, "close", counted_close)
    entries = provenance._tracked_path_states(repo, paths)

    assert len(entries) == len(paths)
    assert all(entry["kind"] == "missing" for entry in entries)
    assert max_live <= 2
    assert live == 0


def test_twenty_four_thousand_deletions_use_bounded_raw_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    paths = [f"archive/{index:05d}.gz".encode() for index in range(24_000)]
    oid = b"a" * 40
    zero = b"0" * 40
    raw = b"".join(
        b":100644 000000 " + oid + b" " + zero + b" D\0" + path + b"\0"
        for path in paths
    )
    index_stage = b"".join(b"100644 " + oid + b" 0\t" + path + b"\0" for path in paths)
    flags = b"".join(b"H " + path + b"\0" for path in paths)
    snapshot = {
        "head": oid,
        "object_format": "sha1",
        "index_stage": index_stage,
        "index_paths": paths,
        "flags_v": flags,
        "flags_f": flags,
    }
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        provenance,
        "_tracked_index_snapshot",
        lambda _repo: snapshot,
    )

    def fake_git(_repo: Path, *args: str, check: bool = True) -> bytes:
        del check
        commands.append(args)
        return raw if args[0] == "diff-files" else b""

    monkeypatch.setattr(provenance, "_git_evidence", fake_git)
    monkeypatch.setattr(
        provenance,
        "_tracked_path_states",
        lambda _repo, raw_paths: [
            {"path": os.fsdecode(path), "kind": "missing"}
            for path in sorted(set(raw_paths))
        ],
    )

    evidence, entries = provenance._tracked_worktree_evidence(repo)

    assert len(entries) == 24_000
    assert len(evidence) < 4 * 1024 * 1024
    assert all("--binary" not in command for command in commands)


def test_index_snapshot_change_during_capture_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    first = {
        "top_level": os.fsencode(repo),
        "head": b"a" * 40,
        "object_format": "sha1",
        "index_stage": b"",
        "index_paths": [],
        "flags_v": b"",
        "flags_f": b"",
    }
    second = {**first, "head": b"b" * 40}
    snapshots = iter((first, second))
    monkeypatch.setattr(
        provenance,
        "_tracked_index_snapshot",
        lambda _repo: next(snapshots),
    )
    monkeypatch.setattr(provenance, "_git_evidence", lambda *_args, **_kwargs: b"")

    with pytest.raises(provenance.ProvenanceError, match="HEAD or index changed"):
        provenance._tracked_worktree_evidence(repo)


def test_start_and_completion_records_are_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id="immutable-run")

    with pytest.raises(FileExistsError, match="immutable evidence"):
        _capture(repo, benchmark_root, run_id="immutable-run")

    completion = provenance.create_completion(start_manifest=start, exit_status=143)
    record = json.loads(completion.read_text(encoding="utf-8"))
    assert record["exit_status"] == 143
    assert record["completed_successfully"] is False
    assert record["start_manifest_sha256"] == provenance._sha256(start.read_bytes())
    assert record["integrity"]["status"] == "valid"
    assert record["integrity"]["violations"] == []
    assert record["host_state"] == HOST_STATE_FIXTURE
    with pytest.raises(FileExistsError, match="immutable evidence"):
        provenance.create_completion(start_manifest=start, exit_status=0)


def test_completion_rejects_containment_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id="containment-drift")
    monkeypatch.setattr(
        provenance,
        "_capture_measurement_containment",
        lambda: {
            "schema": "ny_measurement_containment_v1",
            "fixture": "drifted",
        },
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=143)
    record = json.loads(completion.read_text(encoding="utf-8"))
    violations = {item["code"] for item in record["integrity"]["violations"]}

    assert record["integrity"]["status"] == "invalid"
    assert record["integrity"]["checks"]["containment"]["status"] == "invalid"
    assert "containment_identity_mismatch" in violations


@pytest.mark.parametrize(
    "key",
    [
        "NY_CROWN_CUT_SEGMENT",
        "NY_IMB_OBJ",
        "NY_IMB_REGION_K",
        "NY_IMB_REPLAY_ONLY_LEAF",
        "NY_WARMUP_ITERS",
    ],
)
def test_typed_numeric_environment_rejects_non_numeric_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "not-a-number")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=key):
        _capture(repo, benchmark_root, run_id=f"invalid-{key.lower()}")


def test_positive_usize_environment_is_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    for key in (
        "NY_BUILD_MIN_FREE_KIB",
        "NY_CUDA_WIDE_MAX_BYTES",
        "NY_GPU_LOCK_WAIT_SECS",
        "NY_GPU_VMEM_LIMIT_KIB",
        "NY_MO_GPU_CHUNK",
    ):
        assert key not in environment["values"]
        assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NY_BUILD_MIN_FREE_KIB", "33554432", 33554432),
        ("NY_MO_GPU_CHUNK", "128", 128),
        ("NY_MO_GPU_CHUNK", "00128", 128),
        ("NY_CUDA_WIDE_MAX_BYTES", "2147483648", 2147483648),
        ("NY_CUDA_WIDE_MAX_BYTES", "0000000001", 1),
        ("NY_GPU_LOCK_WAIT_SECS", "03600", 3600),
        ("NY_GPU_VMEM_LIMIT_KIB", "167772160", 167772160),
    ],
)
def test_positive_usize_environment_preserves_raw_and_captures_typed_value(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "positive_integer",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "0", "+64", "-1", " 64", "64 ", "64.0"])
def test_mo_gpu_chunk_invalid_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_GPU_CHUNK", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MO_GPU_CHUNK"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    ["NY_MO_GPU_CHUNK", "NY_CUDA_WIDE_MAX_BYTES"],
)
def test_positive_usize_environment_accepts_native_maximum(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    usize_max = (sys.maxsize * 2) + 1
    monkeypatch.setenv(key, str(usize_max))

    environment = provenance._capture_environment()

    assert environment["typed_values"][key] == {
        "type": "positive_integer",
        "value": usize_max,
    }


@pytest.mark.parametrize(
    "key",
    ["NY_MO_GPU_CHUNK", "NY_CUDA_WIDE_MAX_BYTES"],
)
def test_positive_usize_environment_rejects_zero_and_overflow(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "0")
    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()

    usize_overflow = (sys.maxsize * 2) + 2
    monkeypatch.setenv(key, str(usize_overflow))
    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_mo_gpu_chunk_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_MO_GPU_CHUNK_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "64")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_guard_path_environment_is_captured_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    paths = {
        "NY_BUILD_DISK_PATH": "/private/build-target",
        "NY_GPU_LOCK_PATH": "/private/runtime/ny.lock",
    }
    for key, value in paths.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    for key, value in paths.items():
        assert environment["values"][key] == value
        assert key not in environment["typed_values"]


def test_noncuda_measure_override_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ALLOW_NONCUDA_MEASURE", "1")

    environment = provenance._capture_environment()

    assert environment["values"]["NY_ALLOW_NONCUDA_MEASURE"] == "1"
    assert environment["typed_values"]["NY_ALLOW_NONCUDA_MEASURE"] == {
        "type": "boolean",
        "value": True,
    }


@pytest.mark.parametrize("raw", ["", "true", "yes", "2"])
def test_noncuda_measure_override_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ALLOW_NONCUDA_MEASURE", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ALLOW_NONCUDA_MEASURE"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_attack_point_fast_kernels_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ATTACK_POINT_FAST_KERNELS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_ATTACK_POINT_FAST_KERNELS"] == raw
    assert environment["typed_values"]["NY_ATTACK_POINT_FAST_KERNELS"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_attack_point_fast_kernels_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ATTACK_POINT_FAST_KERNELS", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ATTACK_POINT_FAST_KERNELS"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_softmax_objective_envelope_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_SOFTMAX_OBJECTIVE_ENVELOPE", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_SOFTMAX_OBJECTIVE_ENVELOPE"] == raw
    assert environment["typed_values"]["NY_SOFTMAX_OBJECTIVE_ENVELOPE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_softmax_objective_envelope_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_SOFTMAX_OBJECTIVE_ENVELOPE", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_SOFTMAX_OBJECTIVE_ENVELOPE"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_root_alpha_phase_checkpoint_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_ALPHA_PHASE_CHECKPOINT", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_ROOT_ALPHA_PHASE_CHECKPOINT"] == raw
    assert environment["typed_values"]["NY_ROOT_ALPHA_PHASE_CHECKPOINT"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_root_alpha_phase_checkpoint_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_ALPHA_PHASE_CHECKPOINT", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ROOT_ALPHA_PHASE_CHECKPOINT"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cgan_truncated_cache_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_SERVE_TRUNCATED_CACHE", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CROWN_SERVE_TRUNCATED_CACHE"] == raw
    assert environment["typed_values"]["NY_CROWN_SERVE_TRUNCATED_CACHE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "true", "yes", "2"])
def test_cgan_truncated_cache_gate_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_SERVE_TRUNCATED_CACHE", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_CROWN_SERVE_TRUNCATED_CACHE",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cut_crown_resident_shadow_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUT_CROWN_RESIDENT_SHADOW", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CUT_CROWN_RESIDENT_SHADOW"] == raw
    assert environment["typed_values"]["NY_CUT_CROWN_RESIDENT_SHADOW"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "true", "yes", "2"])
def test_cut_crown_resident_shadow_gate_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUT_CROWN_RESIDENT_SHADOW", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_CUT_CROWN_RESIDENT_SHADOW",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cut_crown_m2_projected_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUT_CROWN_M2_PROJECTED", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CUT_CROWN_M2_PROJECTED"] == raw
    assert environment["typed_values"]["NY_CUT_CROWN_M2_PROJECTED"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "true", "yes", "2"])
def test_cut_crown_m2_projected_gate_rejects_noncanonical_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUT_CROWN_M2_PROJECTED", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_CUT_CROWN_M2_PROJECTED",
    ):
        provenance._capture_environment()


def test_mip_dump_capture_path_is_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_MIP_DUMP" not in environment["values"]
    assert "NY_MIP_DUMP" not in environment["typed_values"]


def test_mip_dump_capture_path_is_exactly_provenanced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    dump_path = tmp_path / "shared-decision-corpus"
    monkeypatch.setenv("NY_MIP_DUMP", str(dump_path))

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_DUMP"] == str(dump_path)
    assert environment["typed_values"]["NY_MIP_DUMP"] == {
        "type": "absolute_path",
        "value": str(dump_path),
    }


@pytest.mark.parametrize("raw", ["", "relative/mip-dump"])
def test_mip_dump_capture_path_rejects_non_absolute_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_DUMP", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MIP_DUMP"):
        provenance._capture_environment()


def test_mip_dump_unknown_neighbor_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_MIP_DUMP_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "/tmp/unreviewed")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_kfsb_winner_oracle_environment_is_captured_and_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    raw_values = {
        "NY_BRANCH_KFSB_CHILDSIM": "1",
        "NY_KFSB_LAYER_QUOTA": "0",
        "NY_MO_ADAPTIVE_DEPTH_SELECT": "1",
        "NY_MO_ADAPTIVE_DEPTH_SHADOW": "1",
        "NY_MO_KFSB": "1",
        "NY_MO_KFSB_CACHED_LA": "1",
        "NY_MO_KFSB_CERT_REUSE": "1",
        "NY_MO_KFSB_CHUNK": "00064",
        "NY_MO_KFSB_F64_SHADOW": "1",
        "NY_MO_KFSB_K": "07",
        "NY_MO_KFSB_PROBE": "1",
        "NY_MO_KFSB_REDUCE": "max",
        "NY_MO_KFSB_WINNER_PROBE": "1",
        "NY_MO_KFSB_WINNER_PROBE_DOMAINS": "03",
    }
    for key, value in raw_values.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    for key, value in raw_values.items():
        assert environment["values"][key] == value
    assert environment["typed_values"] == {
        "NY_MO_ADAPTIVE_DEPTH_SELECT": {"type": "boolean", "value": True},
        "NY_MO_ADAPTIVE_DEPTH_SHADOW": {"type": "boolean", "value": True},
        "NY_MO_KFSB": {"type": "boolean", "value": True},
        "NY_MO_KFSB_CERT_REUSE": {"type": "boolean", "value": True},
        "NY_MO_KFSB_F64_SHADOW": {"type": "boolean", "value": True},
        "NY_MO_KFSB_CHUNK": {"type": "positive_integer", "value": 64},
        "NY_MO_KFSB_K": {"type": "nonnegative_integer", "value": 7},
        "NY_MO_KFSB_REDUCE": {
            "type": "enum",
            "value": "max",
            "allowed_values": ["max", "min"],
        },
        "NY_MO_KFSB_WINNER_PROBE_DOMAINS": {
            "type": "positive_integer",
            "value": 3,
        },
    }


@pytest.mark.parametrize("raw", ["mean", "MAX", " min", "max ", ""])
def test_mo_kfsb_reduce_measurement_grammar_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_KFSB_REDUCE", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MO_KFSB_REDUCE"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("NY_MO_KFSB_K", "-1"),
        ("NY_MO_KFSB_CHUNK", "0"),
        ("NY_MO_KFSB_WINNER_PROBE_DOMAINS", "not-a-number"),
    ],
)
def test_kfsb_winner_oracle_invalid_numeric_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_kfsb_f64_shadow_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_KFSB_F64_SHADOW", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MO_KFSB_F64_SHADOW"] == raw
    assert environment["typed_values"]["NY_MO_KFSB_F64_SHADOW"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_kfsb_f64_shadow_gate_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_KFSB_F64_SHADOW", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MO_KFSB_F64_SHADOW"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_kfsb_cert_reuse_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_KFSB_CERT_REUSE", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MO_KFSB_CERT_REUSE"] == raw
    assert environment["typed_values"]["NY_MO_KFSB_CERT_REUSE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_kfsb_cert_reuse_gate_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_KFSB_CERT_REUSE", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MO_KFSB_CERT_REUSE"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    [
        "NY_MO_ADAPTIVE_DEPTH_SHADOW",
        "NY_MO_ADAPTIVE_DEPTH_SELECT",
        "NY_MO_ADAPTIVE_DEPTH_COMMIT",
    ],
)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_adaptive_depth_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize(
    "key",
    [
        "NY_MO_ADAPTIVE_DEPTH_SHADOW",
        "NY_MO_ADAPTIVE_DEPTH_SELECT",
        "NY_MO_ADAPTIVE_DEPTH_COMMIT",
    ],
)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_adaptive_depth_gate_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_rel_bab_deadline_multiplier_is_captured_raw_and_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_REL_BAB_DEADLINE_MULT", "+01.400e0")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="rel-bab-mult-raw").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_REL_BAB_DEADLINE_MULT"] == "+01.400e0"
    assert environment["typed_values"]["NY_REL_BAB_DEADLINE_MULT"] == {
        "type": "bounded_float",
        "value": 1.4,
        "minimum": 1.0,
        "maximum": 10.0,
    }


@pytest.mark.parametrize(
    "raw", ["not-a-number", "NaN", "inf", "0.999", "10.001", " 1.4 "]
)
def test_rel_bab_deadline_multiplier_invalid_value_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_REL_BAB_DEADLINE_MULT", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="NY_REL_BAB_DEADLINE_MULT"):
        _capture(repo, benchmark_root, run_id="rel-bab-mult-invalid")

    assert not (
        repo / "reports/measured/artifacts/runs/rel-bab-mult-invalid/start.json"
    ).exists()


def test_rel_bab_deadline_multiplier_unknown_neighbor_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_REL_BAB_DEADLINE_MULT_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1.4")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        _capture(repo, benchmark_root, run_id="rel-bab-mult-unknown")

    assert not (
        repo / "reports/measured/artifacts/runs/rel-bab-mult-unknown/start.json"
    ).exists()


def test_margin_row_schedule_family_is_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    for key in (
        "NY_MARGIN_ROW_RESERVE_SECS",
        "NY_MARGIN_ROW_ADAPTIVE_RESERVE",
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC",
        "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC",
        "NY_MARGIN_ROW_CONCURRENT",
    ):
        assert key not in environment["values"]
        assert key not in environment["typed_values"]


def test_upfront_attack_family_is_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    for key in (
        "NY_UPFRONT_ATTACK",
        "NY_UPFRONT_ATTACK_CAP",
        "NY_UPFRONT_ATTACK_AUTO_CAP",
        "NY_UPFRONT_ATTACK_FRAC",
        "NY_UPFRONT_ATTACK_PROBE_FIRST",
        "NY_BNN_ATTACK_CAP",
        "NY_BNN_ATTACK_FRAC",
    ):
        assert key not in environment["values"]
        assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    "key", ["NY_UPFRONT_ATTACK", "NY_UPFRONT_ATTACK_PROBE_FIRST"]
)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_upfront_attack_booleans_are_captured_exactly(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize(
    "key", ["NY_UPFRONT_ATTACK", "NY_UPFRONT_ATTACK_PROBE_FIRST"]
)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_upfront_attack_booleans_reject_fake_treatments(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    ["NY_UPFRONT_ATTACK_CAP", "NY_UPFRONT_ATTACK_AUTO_CAP", "NY_BNN_ATTACK_CAP"],
)
@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("12", 12)])
def test_upfront_attack_caps_are_captured_as_nonnegative_integers(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "nonnegative_integer",
        "value": expected,
    }


@pytest.mark.parametrize(
    "key",
    ["NY_UPFRONT_ATTACK_CAP", "NY_UPFRONT_ATTACK_AUTO_CAP", "NY_BNN_ATTACK_CAP"],
)
@pytest.mark.parametrize("raw", ["", "-1", "+1", " 12 ", "12.0", "nope"])
def test_upfront_attack_caps_reject_noncanonical_measurement_tokens(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "expected"), [("5e-324", 5e-324), ("0.08", 0.08), ("1", 1.0)]
)
@pytest.mark.parametrize("key", ["NY_UPFRONT_ATTACK_FRAC", "NY_BNN_ATTACK_FRAC"])
def test_upfront_attack_fraction_is_captured_in_its_exact_interval(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: float,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "bounded_float",
        "value": expected,
        "minimum": float.fromhex("0x0.0000000000001p-1022"),
        "maximum": 1.0,
    }


@pytest.mark.parametrize(
    "raw", ["", "0", "-0", "-1", "1.0001", "NaN", "inf", " 0.08 "]
)
@pytest.mark.parametrize("key", ["NY_UPFRONT_ATTACK_FRAC", "NY_BNN_ATTACK_FRAC"])
def test_upfront_attack_fraction_rejects_fake_treatments(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.25", 0.25),
        ("0.5", 0.5),
        ("0.0001", 0.0001),
        ("0.9999999999999999", 0.9999999999999999),
    ],
)
@pytest.mark.parametrize(
    "key",
    [
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC",
        "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC",
    ],
)
def test_margin_row_schedule_fraction_is_captured_raw_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: float,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "open_unit_decimal_fraction",
        "value": expected,
        "minimum_exclusive": 0.0,
        "maximum_exclusive": 1.0,
    }


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0",
        "0.0",
        "1",
        "1.0",
        "-0.25",
        "+0.25",
        ".25",
        "00.25",
        "0.250",
        "2.5e-1",
        "NaN",
        "nan",
        "inf",
        "Infinity",
        " 0.25 ",
        # Both Python and Rust's IEEE-754 f64 parser round this to 1.0, which
        # the runtime declines even though the source decimal is below one.
        "0.99999999999999999",
    ],
)
@pytest.mark.parametrize(
    "key",
    [
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC",
        "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC",
    ],
)
def test_margin_row_schedule_fraction_rejects_noncanonical_or_runtime_invalid(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("45", 45), ("82", 82)])
def test_margin_row_reserve_seconds_is_a_canonical_nonnegative_integer(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_RESERVE_SECS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MARGIN_ROW_RESERVE_SECS"] == raw
    assert environment["typed_values"]["NY_MARGIN_ROW_RESERVE_SECS"] == {
        "type": "nonnegative_integer",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "+1", "-1", "1.0", " 45 ", "bad"])
def test_margin_row_reserve_seconds_rejects_noncanonical_measurement_tokens(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_RESERVE_SECS", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MARGIN_ROW_RESERVE_SECS"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    ["NY_MARGIN_ROW_ADAPTIVE_RESERVE", "NY_MARGIN_ROW_CONCURRENT"],
)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_margin_row_schedule_booleans_are_captured_canonically(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize(
    "key",
    ["NY_MARGIN_ROW_ADAPTIVE_RESERVE", "NY_MARGIN_ROW_CONCURRENT"],
)
@pytest.mark.parametrize("raw", ["", "00", "+1", "true", "on", " 1 ", "1 "])
def test_margin_row_schedule_booleans_reject_compatibility_aliases(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    # In particular, CONCURRENT's runtime TrimmedExactOne grammar treats
    # padded 1 as true and every other Unicode spelling as false. Those legacy
    # aliases remain runtime-compatible but cannot name distinct measurements.
    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_margin_row_reserve_max_fraction_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_MARGIN_ROW_RESERVE_MAX_FRAC_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "0.25")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_ny_search_parses_and_canonicalizes_scheduler_layered_kinds(
    tmp_path: Path,
) -> None:
    import ny_search  # noqa: PLC0415

    (tmp_path / "schedule.rs").write_text(
        """
pub RESERVE = LeverDecl {
    name: "NY_FIXTURE_RESERVE",
    kind: LeverKind::U64Override,
    default: DefaultSpec::U64(45),
};
pub MAX_FRAC = LeverDecl {
    name: "NY_FIXTURE_MAX_FRAC",
    kind: LeverKind::F64OpenUnicodeOwned { min: 0.0, max: 1.0 },
    default: DefaultSpec::Unset,
};
pub RELEASE_FRAC = LeverDecl {
    name: "NY_FIXTURE_RELEASE_FRAC",
    kind: LeverKind::F64OpenOverride { min: 0.0, max: 1.0 },
    default: DefaultSpec::F64(0.40),
};
pub CONCURRENT = LeverDecl {
    name: "NY_FIXTURE_CONCURRENT",
    kind: LeverKind::TrimmedExactOne,
    default: DefaultSpec::Bool(false),
};
pub VERIFIER_THREADS = LeverDecl {
    name: "NY_FIXTURE_VERIFIER_THREADS",
    kind: LeverKind::UsizeTrimmedUnicodeOwned,
    default: DefaultSpec::Unset,
};
pub CONCURRENT_THREADS = LeverDecl {
    name: "NY_FIXTURE_CONCURRENT_THREADS",
    kind: LeverKind::UsizeOverride { min: 0 },
    default: DefaultSpec::Unset,
};
""",
        encoding="utf-8",
    )
    source = ny_search.parse_lever_kinds(tmp_path)
    assert source == {
        "NY_FIXTURE_RESERVE": {
            "kind": "U64Override",
            "default": "45",
            "file": "schedule.rs",
        },
        "NY_FIXTURE_MAX_FRAC": {
            "kind": "F64OpenUnicodeOwned",
            "default": None,
            "file": "schedule.rs",
            "min": 0.0,
            "max": 1.0,
        },
        "NY_FIXTURE_RELEASE_FRAC": {
            "kind": "F64OpenOverride",
            "default": "0.40",
            "file": "schedule.rs",
            "min": 0.0,
            "max": 1.0,
        },
        "NY_FIXTURE_CONCURRENT": {
            "kind": "TrimmedExactOne",
            "default": "0",
            "file": "schedule.rs",
        },
        "NY_FIXTURE_VERIFIER_THREADS": {
            "kind": "UsizeTrimmedUnicodeOwned",
            "default": None,
            "file": "schedule.rs",
        },
        "NY_FIXTURE_CONCURRENT_THREADS": {
            "kind": "UsizeOverride",
            "default": None,
            "file": "schedule.rs",
            "min": 0,
        },
    }

    compiled = ny_search._declarations_from_payload(
        {
            "NY_FIXTURE_RESERVE": {
                "kind": "U64Override",
                "default": "45",
                "min": None,
                "max": None,
                "members": None,
            },
            "NY_FIXTURE_MAX_FRAC": {
                "kind": "F64OpenUnicodeOwned",
                "default": None,
                "min": 0.0,
                "max": 1.0,
                "members": None,
            },
            "NY_FIXTURE_RELEASE_FRAC": {
                "kind": "F64OpenOverride",
                "default": "0.4",
                "min": 0.0,
                "max": 1.0,
                "members": None,
            },
            "NY_FIXTURE_CONCURRENT": {
                "kind": "TrimmedExactOne",
                "default": False,
                "min": None,
                "max": None,
                "members": None,
            },
            "NY_FIXTURE_VERIFIER_THREADS": {
                "kind": "UsizeTrimmedUnicodeOwned",
                "default": None,
                "min": None,
                "max": None,
                "members": None,
            },
            "NY_FIXTURE_CONCURRENT_THREADS": {
                "kind": "UsizeOverride",
                "default": None,
                "min": 0,
                "max": float((1 << (8 * struct.calcsize("P"))) - 1),
                "members": None,
            },
        }
    )
    assert compiled is not None
    assert compiled["NY_FIXTURE_RESERVE"]["kind"] == "U64Override"
    assert compiled["NY_FIXTURE_MAX_FRAC"] == {
        "kind": "F64OpenUnicodeOwned",
        "default": None,
        "min": 0.0,
        "max": 1.0,
    }
    assert compiled["NY_FIXTURE_RELEASE_FRAC"]["kind"] == "F64OpenOverride"
    assert compiled["NY_FIXTURE_CONCURRENT"] == {
        "kind": "TrimmedExactOne",
        "default": "0",
    }
    assert compiled["NY_FIXTURE_VERIFIER_THREADS"] == {
        "kind": "UsizeTrimmedUnicodeOwned",
        "default": None,
    }
    assert compiled["NY_FIXTURE_CONCURRENT_THREADS"] == {
        "kind": "UsizeOverride",
        "default": None,
        "min": 0,
        "max": float((1 << (8 * struct.calcsize("P"))) - 1),
    }

    reserve = source["NY_FIXTURE_RESERVE"]
    max_frac = source["NY_FIXTURE_MAX_FRAC"]
    release_frac = source["NY_FIXTURE_RELEASE_FRAC"]
    concurrent = source["NY_FIXTURE_CONCURRENT"]
    verifier_threads = source["NY_FIXTURE_VERIFIER_THREADS"]
    concurrent_threads = source["NY_FIXTURE_CONCURRENT_THREADS"]
    assert ny_search.value_admissible(reserve, "+45")
    assert not ny_search.value_admissible(reserve, " 45 ")
    assert ny_search.tokens_semantically_equal(reserve, "045", "45")
    for kind in (max_frac, release_frac):
        assert ny_search.value_admissible(kind, "+2.5e-1")
        assert not ny_search.value_admissible(kind, " 0.25 ")
        assert not ny_search.value_admissible(kind, "0")
        assert not ny_search.value_admissible(kind, "1")
        assert ny_search.tokens_semantically_equal(kind, "0.25", "2.5e-1")
    assert all(
        ny_search.value_admissible(concurrent, token)
        for token in ("1", " 1 ", "0", "true", "on", "", "bad")
    )
    assert ny_search.tokens_semantically_equal(concurrent, "1", "\u20031\u2003")
    assert ny_search.tokens_semantically_equal(concurrent, "0", "true")
    assert not ny_search.tokens_semantically_equal(concurrent, "1", "on")
    assert ny_search.value_admissible(verifier_threads, "  +0\n")
    assert not ny_search.value_admissible(verifier_threads, "malformed")
    assert ny_search.tokens_semantically_equal(verifier_threads, " 016 ", "16")
    assert ny_search.value_admissible(concurrent_threads, "+0")
    assert not ny_search.value_admissible(concurrent_threads, " 0 ")
    assert ny_search.tokens_semantically_equal(concurrent_threads, "+016", "16")

    reserve_axis = ny_search.Axis(
        "NY_FIXTURE_RESERVE", "grid", ["045", "60"],
        "VerdictAffecting", "env-only",
    )
    release_axis = ny_search.Axis(
        "NY_FIXTURE_RELEASE_FRAC", "enum", ["0.40", "0.50"],
        "VerdictAffecting", "env-only",
    )
    concurrent_axis = ny_search.Axis(
        "NY_FIXTURE_CONCURRENT", "bool", ["0", "1"],
        "VerdictAffecting", "env-only",
    )
    assert ny_search.admissible_values(
        reserve_axis, {reserve_axis.name: reserve}, False
    )[0] == ["60"]
    assert ny_search.admissible_values(
        release_axis, {release_axis.name: release_frac}, False
    )[0] == ["0.50"]
    assert ny_search.admissible_values(
        concurrent_axis, {concurrent_axis.name: concurrent}, False
    )[0] == ["1"]


def test_endgame_grace_is_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_ENDGAME_GRACE_SECS" not in environment["values"]
    assert "NY_ENDGAME_GRACE_SECS" not in environment["typed_values"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0.0), ("12", 12.0), ("+01.25e1", 12.5), ("30.0", 30.0)],
)
def test_endgame_grace_is_captured_raw_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: float,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ENDGAME_GRACE_SECS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_ENDGAME_GRACE_SECS"] == raw
    assert environment["typed_values"]["NY_ENDGAME_GRACE_SECS"] == {
        "type": "bounded_float",
        "value": expected,
        "minimum": 0.0,
        "maximum": 30.0,
    }


@pytest.mark.parametrize(
    "raw",
    ["", "not-a-number", "NaN", "inf", "-0.001", "30.001", " 12 ", "12s"],
)
def test_endgame_grace_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ENDGAME_GRACE_SECS", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ENDGAME_GRACE_SECS"):
        provenance._capture_environment()


def test_relational_mip_controls_are_captured_as_raw_launch_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    controls = {
        "NY_REL_DIFF_COUPLING": "1",
        "NY_REL_JOINT_RELU_CUTS": "1",
        "NY_REL_JOINT_RELU_CUTS_SUM": "0",
        "NY_REL_WHOLE_MIP_OBBT": "1",
        "NY_REL_WHOLE_MIP_OBBT_CHUNK": "08",
        "NY_REL_WHOLE_MIP_OBBT_COND": "1",
        "NY_REL_WHOLE_MIP_OBBT_COND_FRAC": "0.45",
        "NY_REL_WHOLE_MIP_OBBT_MAXN": "192",
        "NY_REL_WHOLE_MIP_OBBT_OUTER": "2",
        "NY_REL_WHOLE_MIP_OBBT_ROUNDS": "3",
        "NY_REL_WHOLE_MIP_OBBT_S": "12.5",
        "NY_REL_WHOLE_MIP_OBBT_WIDTH": "1000.0",
    }
    for key, value in controls.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    assert {key: environment["values"][key] for key in controls} == controls
    assert not controls.keys() & environment["typed_values"].keys()


@pytest.mark.parametrize(
    "key",
    ["NY_JOINT_MEAS_NCOLS", "NY_JOINT_MEAS_PERSOLVE_S", "NY_JOINT_MEAS_DEADLINE_S"],
)
def test_ignored_joint_cut_test_knobs_remain_outside_solver_measurement_authority(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "1")

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", False), ("0", False), ("1", True)],
)
def test_gpu_available_is_captured_raw_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("GPU_AVAILABLE", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="gpu-available-raw").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["GPU_AVAILABLE"] == raw
    assert environment["typed_values"]["GPU_AVAILABLE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["2", "00", "yes", "-1", " 1"])
def test_gpu_available_non_boolean_value_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("GPU_AVAILABLE", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="GPU_AVAILABLE"):
        _capture(repo, benchmark_root, run_id="gpu-available-invalid")

    assert not (
        repo / "reports/measured/artifacts/runs/gpu-available-invalid/start.json"
    ).exists()


def test_cuda_dgemm_triplet_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_CUDA_DGEMM_TRIPLET" not in environment["values"]
    assert "NY_CUDA_DGEMM_TRIPLET" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cuda_dgemm_triplet_is_captured_raw_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUDA_DGEMM_TRIPLET", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id=f"cuda-dgemm-triplet-{raw}").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_CUDA_DGEMM_TRIPLET"] == raw
    assert environment["typed_values"]["NY_CUDA_DGEMM_TRIPLET"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_cuda_dgemm_triplet_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUDA_DGEMM_TRIPLET", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_CUDA_DGEMM_TRIPLET"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("key", "values"),
    [
        (
            "NY_CUDA_GEMM_TRANSPORT",
            [
                "auto",
                "direct-host-page-tables",
                "unified-memory",
                "explicit-device-copy",
            ],
        ),
        ("NY_GPU_DENORM_PRESERVE", ["auto", "0", "1"]),
        ("NY_WGPU_CROWN", ["auto", "0", "1"]),
    ],
)
def test_gpu_profile_enums_are_captured_exactly(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    values: list[str],
) -> None:
    for raw in values:
        _clear_ny_environment(monkeypatch)
        monkeypatch.setenv(key, raw)
        environment = provenance._capture_environment()
        assert environment["values"][key] == raw
        assert environment["typed_values"][key] == {
            "type": "enum",
            "value": raw,
            "allowed_values": sorted(values),
        }


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("NY_CUDA_GEMM_TRANSPORT", "explicit"),
        ("NY_GPU_DENORM_PRESERVE", "true"),
        ("NY_WGPU_CROWN", "true"),
    ],
)
def test_gpu_profile_enums_reject_unreviewed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)
    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cuda_crown_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUDA_CROWN", raw)
    environment = provenance._capture_environment()
    assert environment["typed_values"]["NY_CUDA_CROWN"] == {
        "type": "boolean",
        "value": expected,
    }


def test_safenlp_short_grace_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert environment["allowlist_schema"] == "ny_measurement_environment_v1"
    assert "NY_SAFENLP_SHORT_GRACE" not in environment["values"]
    assert "NY_SAFENLP_SHORT_GRACE" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_safenlp_short_grace_is_captured_raw_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_SAFENLP_SHORT_GRACE", raw)

    environment = provenance._capture_environment()

    assert environment["allowlist_schema"] == "ny_measurement_environment_v1"
    assert environment["values"]["NY_SAFENLP_SHORT_GRACE"] == raw
    assert environment["typed_values"]["NY_SAFENLP_SHORT_GRACE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_safenlp_short_grace_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_SAFENLP_SHORT_GRACE", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_SAFENLP_SHORT_GRACE"):
        provenance._capture_environment()


def test_convtranspose_sound_f64_gpu_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_CONVTRANSPOSE_SOUND_F64_GPU" not in environment["values"]
    assert "NY_CONVTRANSPOSE_SOUND_F64_GPU" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_convtranspose_sound_f64_gpu_is_captured_raw_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CONVTRANSPOSE_SOUND_F64_GPU", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CONVTRANSPOSE_SOUND_F64_GPU"] == raw
    assert environment["typed_values"]["NY_CONVTRANSPOSE_SOUND_F64_GPU"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_convtranspose_sound_f64_gpu_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CONVTRANSPOSE_SOUND_F64_GPU", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_CONVTRANSPOSE_SOUND_F64_GPU"
    ):
        provenance._capture_environment()


def test_root_post_c_survivor_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_ROOT_POST_C_SURVIVOR" not in environment["values"]
    assert "NY_ROOT_POST_C_SURVIVOR" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_root_post_c_survivor_explicit_override_is_captured_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_POST_C_SURVIVOR", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id=f"root-post-c-survivor-{raw}").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_ROOT_POST_C_SURVIVOR"] == raw
    assert environment["typed_values"]["NY_ROOT_POST_C_SURVIVOR"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_root_post_c_survivor_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_POST_C_SURVIVOR", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ROOT_POST_C_SURVIVOR"):
        provenance._capture_environment()


def test_root_post_c_survivor_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_ROOT_POST_C_SURVIVOR_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


@pytest.mark.parametrize("key", ["NY_ALPHA_FINAL_BOUND_ONLY", "NY_BN_FOLD_EXT"])
def test_alpha_final_and_bn_fold_experiments_are_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert key not in environment["values"]
    assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NY_ALPHA_FINAL_BOUND_ONLY", "0", False),
        ("NY_ALPHA_FINAL_BOUND_ONLY", "1", True),
        ("NY_BN_FOLD_EXT", "0", False),
        ("NY_BN_FOLD_EXT", "1", True),
    ],
)
def test_alpha_final_and_bn_fold_explicit_overrides_are_immutably_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    run_id = f"{key.lower().replace('_', '-')}-{raw}"

    manifest_path = _capture(repo, benchmark_root, run_id=run_id)
    manifest_bytes = manifest_path.read_bytes()
    environment = json.loads(manifest_bytes)["environment"]
    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }

    monkeypatch.setenv(key, "1" if raw == "0" else "0")
    with pytest.raises(FileExistsError, match="immutable evidence"):
        _capture(repo, benchmark_root, run_id=run_id)
    assert manifest_path.read_bytes() == manifest_bytes


@pytest.mark.parametrize("key", ["NY_ALPHA_FINAL_BOUND_ONLY", "NY_BN_FOLD_EXT"])
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_alpha_final_and_bn_fold_malformed_values_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize("key", ["NY_ALPHA_FINAL_BOUND_ONLY", "NY_BN_FOLD_EXT"])
def test_alpha_final_and_bn_fold_unknown_neighbors_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = f"{key}_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_cgan_generator_branch_controls_are_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    for key in CGAN_GENERATOR_BRANCH_ENV:
        assert key not in environment["values"]
        assert key not in environment["typed_values"]


@pytest.mark.parametrize("key", CGAN_GENERATOR_BRANCH_BOOLEAN_ENV)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cgan_generator_branch_boolean_controls_are_captured_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


def test_cgan_generator_branch_scope_is_sealed_with_raw_node_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    raw_values = {
        "NY_BRANCH_STEM": "1",
        "NY_BRANCH_STEM_K": "0008",
        "NY_BRANCH_STEM_NODES": " Relu_6,Relu_9, Relu_12 ",
        "NY_BRANCH_STEM_PROBE": "0",
        "NY_BRANCH_TRACE": "1",
        "NY_MO_BAB_TRACE": "1",
        "NY_UNSTABLE_COUNT": "0",
    }
    for key, value in raw_values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="cgan-generator-branch").read_text()
    )
    environment = manifest["environment"]

    assert {
        key: environment["values"][key] for key in CGAN_GENERATOR_BRANCH_ENV
    } == raw_values
    assert environment["typed_values"] == {
        "NY_BRANCH_STEM": {"type": "boolean", "value": True},
        "NY_BRANCH_STEM_K": {"type": "positive_integer", "value": 8},
        "NY_BRANCH_STEM_PROBE": {"type": "boolean", "value": False},
        "NY_BRANCH_TRACE": {"type": "boolean", "value": True},
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT": {
            "schema": "ny-measure/execution-predicate-requirement/v1",
            "type": "execution_predicate_requirement",
            "predicate_id": "ny-exec-v1.pgd-fast-gradient-parameter-engaged",
            "axis": "NY_PGD_FAST_GRAD",
            "requested_value": "0",
            "claim": "parameter_engaged",
        },
        "NY_MO_BAB_TRACE": {"type": "boolean", "value": True},
        "NY_UNSTABLE_COUNT": {"type": "boolean", "value": False},
    }


@pytest.mark.parametrize("key", CGAN_GENERATOR_BRANCH_BOOLEAN_ENV)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_cgan_generator_branch_boolean_controls_reject_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0",
        "+8",
        "-1",
        " 8",
        "8 ",
        "8.0",
        str((sys.maxsize * 2) + 2),
    ],
)
def test_cgan_generator_branch_depth_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_BRANCH_STEM_K", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_BRANCH_STEM_K"):
        provenance._capture_environment()


@pytest.mark.parametrize("key", CGAN_GENERATOR_BRANCH_ENV)
def test_cgan_generator_branch_unknown_neighbors_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = f"{key}_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    [
        "NY_AY_BRANCH_HINTS",
        "NY_AY_MARGIN_REFRAME",
        "NY_AY_MILP_TALL_FLIP_CAP",
        "NY_AY_OBJECTIVE_FIRST_SAT",
        "NY_MIP_CERTIFIED_SHARED_TREE",
        "NY_MIP_SAFENLP_DIRECT_FIRST",
        "NY_MIP_SAFENLP_SHARED_PREFIX",
        "NY_MIP_SAFENLP_TARGET_FSB_PREFIX",
        "NY_MIP_SERIAL",
    ],
)
def test_ay_milp_canary_is_captured_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "1")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest_path = _capture(repo, benchmark_root, run_id=key.lower())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["environment"]["values"][key] == "1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", False), ("0", False), ("1", True)],
)
def test_ay_branch_hint_canary_is_sealed_as_a_strict_boolean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_BRANCH_HINTS", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="ay-branch-hints-typed").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_AY_BRANCH_HINTS"] == raw
    assert environment["typed_values"]["NY_AY_BRANCH_HINTS"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["2", "yes", "true", "-1", " 1"])
def test_ay_branch_hint_canary_malformed_measurement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_BRANCH_HINTS", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="NY_AY_BRANCH_HINTS"):
        _capture(repo, benchmark_root, run_id="ay-branch-hints-malformed")

    assert not (
        repo / "reports/measured/artifacts/runs/ay-branch-hints-malformed/start.json"
    ).exists()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_ay_margin_reframe_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_MARGIN_REFRAME", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_AY_MARGIN_REFRAME"] == raw
    assert environment["typed_values"]["NY_AY_MARGIN_REFRAME"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_ay_margin_reframe_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_MARGIN_REFRAME", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_AY_MARGIN_REFRAME"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_ay_objective_first_sat_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_OBJECTIVE_FIRST_SAT", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_AY_OBJECTIVE_FIRST_SAT"] == raw
    assert environment["typed_values"]["NY_AY_OBJECTIVE_FIRST_SAT"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_ay_objective_first_sat_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_OBJECTIVE_FIRST_SAT", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_AY_OBJECTIVE_FIRST_SAT"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_sequential_mip_serial_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SERIAL", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_SERIAL"] == raw
    assert environment["typed_values"]["NY_MIP_SERIAL"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_sequential_mip_serial_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SERIAL", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MIP_SERIAL"):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_mip_certified_shared_tree_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_CERTIFIED_SHARED_TREE", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_CERTIFIED_SHARED_TREE"] == raw
    assert environment["typed_values"]["NY_MIP_CERTIFIED_SHARED_TREE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_mip_certified_shared_tree_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_CERTIFIED_SHARED_TREE", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_MIP_CERTIFIED_SHARED_TREE"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_mip_safenlp_shared_prefix_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_SHARED_PREFIX", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_SAFENLP_SHARED_PREFIX"] == raw
    assert environment["typed_values"]["NY_MIP_SAFENLP_SHARED_PREFIX"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_mip_safenlp_shared_prefix_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_SHARED_PREFIX", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_MIP_SAFENLP_SHARED_PREFIX"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_mip_safenlp_target_fsb_prefix_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_TARGET_FSB_PREFIX", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_SAFENLP_TARGET_FSB_PREFIX"] == raw
    assert environment["typed_values"]["NY_MIP_SAFENLP_TARGET_FSB_PREFIX"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_mip_safenlp_target_fsb_prefix_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_TARGET_FSB_PREFIX", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_MIP_SAFENLP_TARGET_FSB_PREFIX"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_mip_safenlp_direct_first_is_sealed_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_DIRECT_FIRST", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MIP_SAFENLP_DIRECT_FIRST"] == raw
    assert environment["typed_values"]["NY_MIP_SAFENLP_DIRECT_FIRST"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "2", "true", "01", " 1", "1 "])
def test_mip_safenlp_direct_first_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MIP_SAFENLP_DIRECT_FIRST", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_MIP_SAFENLP_DIRECT_FIRST"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "value"),
    [("1", 1), ("05000", 5_000), ("60000", 60_000)],
)
def test_ay_node_warm_cap_is_sealed_as_a_bounded_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    value: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_NODE_WARM_CAP_MS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_AY_NODE_WARM_CAP_MS"] == raw
    assert environment["typed_values"]["NY_AY_NODE_WARM_CAP_MS"] == {
        "type": "bounded_positive_integer",
        "value": value,
        "minimum": 1,
        "maximum": 60_000,
    }


@pytest.mark.parametrize(
    "raw",
    ["", "0", "+5000", "-1", "60001", "5.0", " 5000", "5000 "],
)
def test_ay_node_warm_cap_malformed_measurement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_NODE_WARM_CAP_MS", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_AY_NODE_WARM_CAP_MS"):
        provenance._capture_environment()


def test_ay_node_warm_cap_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_NODE_WARM_CAP_MS_UNREVIEWED", "5000")

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_AY_NODE_WARM_CAP_MS_UNREVIEWED",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "value"),
    [("1", 1), ("0015", 15), ("3600", 3_600)],
)
def test_crown_ibp_collector_cap_is_sealed_as_a_bounded_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    value: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_IBP_COLLECTOR_CAP_SECS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CROWN_IBP_COLLECTOR_CAP_SECS"] == raw
    assert environment["typed_values"]["NY_CROWN_IBP_COLLECTOR_CAP_SECS"] == {
        "type": "bounded_positive_integer",
        "value": value,
        "minimum": 1,
        "maximum": 3_600,
    }


@pytest.mark.parametrize(
    "raw",
    ["", "0", "+15", "-1", "3601", "15.0", " 15", "15 "],
)
def test_crown_ibp_collector_cap_malformed_measurement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_IBP_COLLECTOR_CAP_SECS", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_CROWN_IBP_COLLECTOR_CAP_SECS"
    ):
        provenance._capture_environment()


def test_crown_chunk_aware_budget_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_CROWN_CHUNK_AWARE_BUDGET" not in environment["values"]
    assert "NY_CROWN_CHUNK_AWARE_BUDGET" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_crown_chunk_aware_budget_is_immutably_typed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_CHUNK_AWARE_BUDGET", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CROWN_CHUNK_AWARE_BUDGET"] == raw
    assert environment["typed_values"]["NY_CROWN_CHUNK_AWARE_BUDGET"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_crown_chunk_aware_budget_malformed_measurement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_CHUNK_AWARE_BUDGET", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_CROWN_CHUNK_AWARE_BUDGET"
    ):
        provenance._capture_environment()


def test_crown_chunk_aware_budget_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_CROWN_CHUNK_AWARE_BUDGET_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_objective_chunk_wave_policy_is_default_sequential_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_NO_CHUNK_WAVE_PAR" not in environment["values"]
    assert "NY_NO_CHUNK_WAVE_PAR" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_objective_chunk_wave_policy_is_immutably_typed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NO_CHUNK_WAVE_PAR", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_NO_CHUNK_WAVE_PAR"] == raw
    assert environment["typed_values"]["NY_NO_CHUNK_WAVE_PAR"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 0", "0 ", "+0", "１"])
def test_objective_chunk_wave_policy_malformed_measurement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NO_CHUNK_WAVE_PAR", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_NO_CHUNK_WAVE_PAR"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    [
        "NY_NO_CHUNK_ABORT",
        "NY_NO_CHUNK_GROW",
        "NY_CROWN_HONEST_PROVENANCE",
    ],
)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_coupled_crown_boolean_policy_is_immutably_typed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize(
    "key",
    [
        "NY_NO_CHUNK_ABORT",
        "NY_NO_CHUNK_GROW",
        "NY_CROWN_HONEST_PROVENANCE",
    ],
)
@pytest.mark.parametrize("raw", ["", "00", "true", " 0", "0 ", "+0", "１"])
def test_coupled_crown_boolean_policy_malformed_measurement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("32", 32), ("28800", 28_800)])
def test_objective_chunk_width_has_one_canonical_measurement_identity(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_OBJ_CHUNK", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CROWN_OBJ_CHUNK"] == raw
    assert environment["typed_values"]["NY_CROWN_OBJ_CHUNK"] == {
        "type": "nonnegative_integer",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", " 32", "32 ", "+32", "-1", "bad", "３２"])
def test_objective_chunk_width_rejects_alias_measurement_identities(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CROWN_OBJ_CHUNK", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_CROWN_OBJ_CHUNK"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("NY_CONV_SKIP_DEAD_F32", "1"),
        ("NY_COMPACT_TAIL_K16", "1"),
        ("NY_ALPHA_REFRESH_FRACTION", "0.125"),
        ("NY_PGD_DIAG", "1"),
        ("NY_PGD_EXACT_BATCHED", "0"),
        ("NY_PGD_GAMA", "1"),
        ("NY_PGD_GAMA_LAMBDA", "50"),
        ("NY_PGD_GAMA_LIN_FRAC", "0.25"),
        ("NY_PGD_VJP_BATCH", "0"),
        ("NY_IMB", "1"),
        ("NY_IMB_AY_REGION_PROOF", "affine"),
        ("NY_IMB_AY_TAIL_ADAPTIVE_FIVE_COMB", "1"),
        ("NY_IMB_BATCHED_REPLAY", "1"),
        ("NY_IMB_WIRE", "1"),
        ("NY_IMB_EARLY", "1"),
        ("NY_IMB_BUDGET_S", "300"),
        ("NY_IMB_LEAF_MODE", "crown_root"),
        ("NY_IMB_OBJ", "0"),
        ("NY_IMB_TAIL_ALPHA", "sample"),
        ("NY_IMB_TAIL_CERT_AY", "1"),
        ("NY_IMB_REGION_K", "2"),
        ("NY_IMB_REPLAY_ONLY", "1"),
        ("NY_IMB_REPLAY_ONLY_LEAF", "4"),
        ("NY_IMB_SELECTOR_K2_LIFT", "1"),
        ("NY_IMB_SELECTOR_K4_LIFT", "1"),
        ("NY_IMB_SELECTOR_RANGE_CRASH", "1"),
        ("NY_IMB_SELECTOR_SOLVE_PROFILE", "1"),
        ("NY_MIP_STABILITY_HINTS", "1"),
        ("NY_BAB_RESNET_REFOLD_GUARD", "0"),
        ("NY_PACKED_GRAPH_ALPHA_QUEUE", "1"),
        ("NY_CUDA_RESIDENT_PATCHES_ROOT", "1"),
        ("NY_ROOT_SKIP_ADAPTIVE_SPEC", "1"),
        ("NY_SEG_RESIDENT", "1"),
    ],
)
def test_dark_performance_canaries_are_captured_exactly(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, value)
    environment = provenance._capture_environment()

    assert environment["values"][key] == value


@pytest.mark.parametrize(
    ("key", "minimum", "maximum"),
    [
        ("NY_GAP_ATTRIBUTION_BUDGET_SECS", 1, 3_600),
        ("NY_GAP_ATTRIBUTION_ROWS", 1, 3),
    ],
)
def test_gap_attribution_integer_gates_are_bounded_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    minimum: int,
    maximum: int,
) -> None:
    for raw, expected in ((str(minimum), minimum), (str(maximum), maximum)):
        _clear_ny_environment(monkeypatch)
        monkeypatch.setenv(key, raw)

        environment = provenance._capture_environment()

        assert environment["values"][key] == raw
        assert environment["typed_values"][key] == {
            "type": "bounded_positive_integer",
            "value": expected,
            "minimum": minimum,
            "maximum": maximum,
        }


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("NY_GAP_ATTRIBUTION_BUDGET_SECS", "0"),
        ("NY_GAP_ATTRIBUTION_BUDGET_SECS", "3601"),
        ("NY_GAP_ATTRIBUTION_ROWS", "0"),
        ("NY_GAP_ATTRIBUTION_ROWS", "4"),
        ("NY_GAP_ATTRIBUTION_ROWS", "not-a-number"),
    ],
)
def test_gap_attribution_integer_gates_reject_unsealed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


EXACT_BOOLEAN_GATES = [
    "AY_LRA_WARM_SIMPLEX_STATE",
    "AY_MILP_NODE_PROP",
    "NY_ADAPTIVE_MICROBATCH_CONTROLLER",
    "NY_ALLOW_NONCUDA_MEASURE",
    "NY_ATTR_BRANCH",
    "NY_ATTR_BRANCH_DIAG",
    "NY_BAB_CLAUSE_LEARN",
    "NY_BAB_CLAUSE_REPLAY",
    "NY_BAB_RESNET_PARALLEL",
    "NY_BICCOS_BCP_SHADOW",
    "NY_BICCOS_Q_STAGE0",
    "NY_BICCOS_Q_STAGE1_REPLAY",
    "NY_COMPACT_TAIL_K16",
    "NY_CONE_REFRESH",
    "NY_CONSTRAINED_PATCHES_ALPHA_RELU_INPLACE",
    "NY_CONSTRAINED_PATCHES_BETA_SPARSE_CAPTURE",
    "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
    "NY_CROWN_IBP_SPARSE_RELU_ROWS",
    "NY_CUDA_RESIDENT_PATCHES_ROOT",
    "NY_DISABLE_CROWN_COLLECTION_CACHE",
    "NY_CUDA_WIDE",
    "NY_EFT_ERR",
    "NY_GAP_ATTRIBUTION",
    "NY_MARGIN_ROW_CONV_BWD_BLOCKED",
    "NY_IMB_AY_TAIL_ADAPTIVE_FIVE_COMB",
    "NY_IMB_SELECTOR_K2_LIFT",
    "NY_IMB_SELECTOR_K4_LIFT",
    "NY_IMB_SELECTOR_RANGE_CRASH",
    "NY_IMB_SELECTOR_SOLVE_PROFILE",
    "NY_INTERM_REFINE_PROBE",
    "NY_MO_BETA_BASELINE_ONLY",
    "NY_MIP_SAFENLP_DIRECT_FIRST",
    "NY_MIP_SAFENLP_SHARED_PREFIX",
    "NY_MIP_SAFENLP_TARGET_FSB_PREFIX",
    "NY_MO_BETA_BASELINE_FIRST",
    "NY_MO_CUDA_BETA_SPSA",
    "NY_MO_CUDA_BOUNDED_SHARED_EXECUTOR",
    "NY_MO_CUDA_FACTORY_ENGINE_HANDOFF",
    "NY_MO_STALL_OBBT_CANARY",
    "NY_PACKED_GRAPH_ALPHA_QUEUE",
    "NY_PATCHES_DEADLINE_FLAT_BIAS",
    "NY_PATCHES_DEADLINE_RELU",
    "NY_PATCHES_EAGER_ERR",
    "NY_PATCHES_EAGER_ERR_7D",
    "NY_PHASE_TELEMETRY",
    "NY_RESNET_ERR_MERGE",
    "NY_ROOT_ALPHA_CUDA_MARGIN_LR_BRACKET",
    "NY_ROOT_ALPHA_CUDA_MARGIN_MW",
    "NY_ROOT_ALPHA_CUDA_MARGIN_STEP",
    "NY_ROOT_ALPHA_CUDA_MARGIN_TOPK",
    "NY_ROOT_ALPHA_CUDA_ROWS",
    "NY_ROOT_ALPHA_GPU",
    "NY_ROOT_ALPHA_MARGIN",
    "NY_ROOT_ALPHA_MARGIN_GRADIENT",
    "NY_ROOT_CRITICAL_GPU_ALPHA",
    "NY_ROOT_CRITICAL_GPU_ALPHA_ACTIVE_SET",
    "NY_ROOT_CRITICAL_GPU_ALPHA_ACTIVE_SET_CASCADE",
    "NY_ROOT_CRITICAL_GPU_ALPHA_LR_BRACKET",
    "NY_ROOT_CRITICAL_GPU_SPEC",
    "NY_ROOT_CROWN_INTERM",
    "NY_ROOT_INTERM_CUDA_FACTORY",
    "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
    "NY_ROOT_JOINT_INTERM_ALPHA_PROBE",
    "NY_ROOT_OUTPUT_CONDITIONED_HEAD",
    "NY_ROOT_SPARSE_INTERM_CROWN",
    "NY_ROOT_SKIP_ADAPTIVE_SPEC",
    "NY_SELECTIVE_ROOT_ALPHA",
    "NY_SEG_RESIDENT",
    "NY_SKIP_DISJ_PGD",
    "NY_WIDE_ACTIVE_COMPACTION_TELEMETRY",
]


@pytest.mark.parametrize("key", EXACT_BOOLEAN_GATES)
def test_exact_gate_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert key not in environment["values"]
    assert key not in environment["typed_values"]


@pytest.mark.parametrize("key", EXACT_BOOLEAN_GATES)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_exact_gate_is_captured_raw_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("key", EXACT_BOOLEAN_GATES)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_exact_gate_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_ay_milp_experiment_controls_are_default_dark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    for key in ("AY_MILP_NODE_PROP",):
        assert key not in environment["values"]
        assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    "raw",
    ["0", "1024", "200000000"],
)
def test_retired_ay_milp_prop_arm_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("AY_MILP_PROP_ARM", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match=r"unrecorded NY_\*/AY_\*/MIMALLOC_\*.*AY_MILP_PROP_ARM",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "unknown_key",
    [
        "AY_MILP_PROP_ARM_UNREVIEWED",
        "AY_MILP_NODE_PROP_UNREVIEWED",
    ],
)
def test_ay_milp_experiment_unknown_neighbors_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    unknown_key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("AY_IDL_ENGINE", "0"),
        ("AY_MILP_FLIP_STALL", "120"),
    ],
)
def test_unreviewed_ay_pin_controls_remain_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    """A dependency bump must not silently grant inherited AY authority."""
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, value)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_root_crown_interm_secs_is_default_dark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_ROOT_CROWN_INTERM_SECS" not in environment["values"]
    assert "NY_ROOT_CROWN_INTERM_SECS" not in environment["typed_values"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0),
        ("00000060", 60),
        ("3600", 3_600),
    ],
)
def test_root_crown_interm_secs_is_captured_as_bounded_nonnegative_integer(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_CROWN_INTERM_SECS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_ROOT_CROWN_INTERM_SECS"] == raw
    assert environment["typed_values"]["NY_ROOT_CROWN_INTERM_SECS"] == {
        "type": "bounded_nonnegative_integer",
        "value": expected,
        "minimum": 0,
        "maximum": 3_600,
    }


@pytest.mark.parametrize(
    "raw",
    ["", "-1", "+1", " 0", "0 ", "1.0", "3601", "not-a-number"],
)
def test_root_crown_interm_secs_rejects_out_of_range_or_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_CROWN_INTERM_SECS", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ROOT_CROWN_INTERM_SECS"):
        provenance._capture_environment()


ROOT_INTERM_CANONICAL_CAPS = (
    ("NY_ROOT_CROWN_INTERM_MAXDIM", 0, 20_000),
    ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM", 0, 8_192),
    ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS", 0, 512),
    ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS", 0, 4),
    ("NY_ROOT_SPARSE_INTERM_CROWN_SECS", 0, 8),
)


@pytest.mark.parametrize(("key", "minimum", "maximum"), ROOT_INTERM_CANONICAL_CAPS)
def test_root_interm_canonical_cap_is_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    minimum: int,
    maximum: int,
) -> None:
    del minimum, maximum
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert key not in environment["values"]
    assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    ("key", "raw", "expected", "minimum", "maximum"),
    [
        ("NY_ROOT_CROWN_INTERM_MAXDIM", "0", 0, 0, 20_000),
        ("NY_ROOT_CROWN_INTERM_MAXDIM", "512", 512, 0, 20_000),
        ("NY_ROOT_CROWN_INTERM_MAXDIM", "20000", 20_000, 0, 20_000),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM", "0", 0, 0, 8_192),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM", "8192", 8_192, 0, 8_192),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS", "0", 0, 0, 512),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS", "512", 512, 0, 512),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS", "0", 0, 0, 4),
        ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS", "4", 4, 0, 4),
        ("NY_ROOT_SPARSE_INTERM_CROWN_SECS", "0", 0, 0, 8),
        ("NY_ROOT_SPARSE_INTERM_CROWN_SECS", "2", 2, 0, 8),
        ("NY_ROOT_SPARSE_INTERM_CROWN_SECS", "8", 8, 0, 8),
    ],
)
def test_root_interm_canonical_cap_is_captured_with_reviewed_bounds(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: int,
    minimum: int,
    maximum: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "bounded_canonical_nonnegative_integer",
        "value": expected,
        "minimum": minimum,
        "maximum": maximum,
    }


@pytest.mark.parametrize(("key", "minimum", "maximum"), ROOT_INTERM_CANONICAL_CAPS)
@pytest.mark.parametrize(
    "raw",
    ["", "00", "01", "-1", "+1", " 1", "1 ", "1.0", "not-a-number", "１"],
)
def test_root_interm_canonical_cap_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    minimum: int,
    maximum: int,
    raw: str,
) -> None:
    del minimum, maximum
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(("key", "minimum", "maximum"), ROOT_INTERM_CANONICAL_CAPS)
def test_root_interm_canonical_cap_rejects_values_above_reviewed_limit(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    minimum: int,
    maximum: int,
) -> None:
    del minimum
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, str(maximum + 1))

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    [
        "NY_IMB_AY_TAIL_ADAPTIVE_FIVE_COMB",
        "NY_IMB_BATCHED_REPLAY",
        "NY_IMB_REPLAY_ONLY",
        "NY_IMB_SELECTOR_K2_LIFT",
        "NY_IMB_SELECTOR_K4_LIFT",
        "NY_IMB_SELECTOR_RANGE_CRASH",
        "NY_IMB_SELECTOR_SOLVE_PROFILE",
        "NY_IMB_TAIL_CERT_AY",
    ],
)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_imb_exact_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize(
    "key",
    [
        "NY_IMB_AY_TAIL_ADAPTIVE_FIVE_COMB",
        "NY_IMB_BATCHED_REPLAY",
        "NY_IMB_REPLAY_ONLY",
        "NY_IMB_SELECTOR_K2_LIFT",
        "NY_IMB_SELECTOR_RANGE_CRASH",
        "NY_IMB_SELECTOR_SOLVE_PROFILE",
        "NY_IMB_TAIL_CERT_AY",
    ],
)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_imb_exact_gate_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize("raw", ["10", "20"])
def test_measurement_expected_cpus_is_captured_as_closed_enum(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MEASURE_EXPECTED_CPUS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MEASURE_EXPECTED_CPUS"] == raw
    assert environment["typed_values"]["NY_MEASURE_EXPECTED_CPUS"] == {
        "type": "enum",
        "value": raw,
        "allowed_values": ["10", "20"],
    }


@pytest.mark.parametrize(
    "raw",
    ["", "0", "010", "11", "21", "+20", "20 ", "ten"],
)
def test_measurement_expected_cpus_rejects_unreviewed_spellings(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MEASURE_EXPECTED_CPUS", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_MEASURE_EXPECTED_CPUS",
    ):
        provenance._capture_environment()


def test_required_flight_event_is_typed_and_never_enters_solver_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    raw = (
        "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
        "NY_PGD_FAST_GRAD.1=ran"
    )
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_MEASURE_REQUIRE_FLIGHT_EVENT"] == raw
    assert environment["typed_values"]["NY_MEASURE_REQUIRE_FLIGHT_EVENT"] == {
        "schema": "ny-measure/execution-predicate-requirement/v1",
        "type": "execution_predicate_requirement",
        "predicate_id": "ny-exec-v1.pgd-fast-gradient-parameter-engaged",
        "axis": "NY_PGD_FAST_GRAD",
        "requested_value": "1",
        "claim": "parameter_engaged",
    }
    assert "NY_PGD_FAST_GRAD" not in provenance.SOLVER_ENVIRONMENT_EXCLUDED_KEYS
    assert (
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT"
        in provenance.SOLVER_ENVIRONMENT_EXCLUDED_KEYS
    )


def test_current_execution_predicate_axes_are_allowlisted_and_typed() -> None:
    assert set(CURRENT_PREDICATE_AXIS_TOKENS) == set(
        provenance.EXECUTION_PREDICATE_AXES
    )
    assert provenance.EXECUTION_PREDICATE_AXES <= provenance.ENV_ALLOWLIST
    assert (
        provenance.EXECUTION_PREDICATE_AXES
        <= provenance._typed_environment_keys()
    )
    assert provenance.EXECUTION_PREDICATE_AXES.isdisjoint(
        provenance.SOLVER_ENVIRONMENT_EXCLUDED_KEYS
    )


def test_every_current_search_axis_has_closed_capture_and_delivery_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="current source ratchet"
    )
    axes = contract["axes"]
    assert isinstance(axes, dict)
    assert provenance.EXECUTION_PREDICATE_AXES <= set(axes)
    assert set(axes).isdisjoint(provenance.SOLVER_ENVIRONMENT_EXCLUDED_KEYS)
    identity = contract["identity"]
    assert isinstance(identity, dict)
    assert identity["axes"] == sorted(axes)
    assert identity["axis_count"] == len(axes)

    for name, axis in axes.items():
        assert isinstance(axis, dict)
        values = axis["values"]
        assert isinstance(values, list) and values
        monkeypatch.setenv(name, values[0])
        environment = provenance._capture_environment(contract)
        assert environment["values"][name] == values[0]
        assert name in environment["typed_values"]
        assert name not in provenance.SOLVER_ENVIRONMENT_EXCLUDED_KEYS
        monkeypatch.delenv(name)


def test_selected_registry_transports_only_closed_tokens_and_types_unfixed_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _root_predicate_search_space_payload(), source="root recipe fixture"
    )
    monkeypatch.setenv("NY_CPU_COMPUTE_ENGINE", "0")

    environment = provenance._capture_environment(contract)

    identity = contract["identity"]
    assert isinstance(identity, dict)
    assert environment["typed_values"]["NY_CPU_COMPUTE_ENGINE"] == {
        "type": "governed_search_axis",
        "value": "0",
        "declaration_kind": "Bool",
        "domain": "bool",
        "allowed_values": ["0", "1"],
        "registry_sha256": identity["sha256"],
    }

    monkeypatch.setenv("NY_CPU_COMPUTE_ENGINE", "false")
    with pytest.raises(
        provenance.ProvenanceError,
        match="outside the selected solver's closed search domain",
    ):
        provenance._capture_environment(contract)


def test_root_scalar_assignment_uses_only_the_generated_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _root_predicate_search_space_payload(), source="root recipe fixture"
    )
    for name, token in ROOT_LR_PREREQUISITES.items():
        monkeypatch.setenv(name, token)
    monkeypatch.setenv("NY_ROOT_JOINT_INTERM_ALPHA_LR", ROOT_LR_REQUEST)
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.root-joint-interm-alpha-parameter-engaged."
        f"NY_ROOT_JOINT_INTERM_ALPHA_LR.{ROOT_LR_REQUEST}=ran",
    )

    environment = provenance._capture_environment(contract)
    assignment = provenance.build_execution_predicate_assignment(
        environment, contract
    )

    requested = [["NY_ROOT_JOINT_INTERM_ALPHA_LR", ROOT_LR_REQUEST]]
    prerequisites = [list(pair) for pair in sorted(ROOT_LR_PREREQUISITES.items())]
    assert assignment["requested_assignment"] == requested
    assert assignment["prerequisites_assignment"] == prerequisites
    assert assignment["comparison_assignment"] == prerequisites
    assert assignment["full_assignment"] == sorted(requested + prerequisites)
    assert assignment["requested_delivery"] == "direct_environment"
    assert assignment["registry"] == contract["identity"]
    assert provenance.validate_execution_predicate_assignment(
        assignment,
        environment=environment,
        search_space_contract=contract,
    ) == {
        "predicate_id": (
            "ny-exec-v1.root-joint-interm-alpha-parameter-engaged"
        ),
        "requested_assignment": requested,
        "prerequisites_assignment": prerequisites,
        "full_assignment": sorted(requested + prerequisites),
        "comparison_assignment": prerequisites,
        "receipt_source_assignment": [
            [name, "legacy_env"]
            for name, _token in sorted(requested + prerequisites)
        ],
        "runtime_context_assignment": [],
        "runtime_context_source_assignment": [],
        "contrast_id": assignment["contrast_id"],
        "registry": contract["identity"],
    }


def test_same_family_joint_assignment_uses_one_generated_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _root_predicate_search_space_payload(), source="root joint recipe fixture"
    )
    for name, token in ROOT_LR_PREREQUISITES.items():
        monkeypatch.setenv(name, token)
    for name, token in ROOT_JOINT_REQUESTED:
        monkeypatch.setenv(name, token)
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", ROOT_JOINT_PREDICATE)

    environment = provenance._capture_environment(contract)
    assignment = provenance.build_execution_predicate_assignment(
        environment, contract
    )

    prerequisites = [list(pair) for pair in sorted(ROOT_LR_PREREQUISITES.items())]
    full = sorted(ROOT_JOINT_REQUESTED + prerequisites)
    assert environment["typed_values"]["NY_MEASURE_REQUIRE_FLIGHT_EVENT"] == {
        "schema": "ny-measure/execution-predicate-requirement/v2",
        "type": "execution_predicate_requirement",
        "predicate_id": (
            "ny-exec-v1.root-joint-interm-alpha-parameter-engaged"
        ),
        "requested_assignment": ROOT_JOINT_REQUESTED,
        "claim": "parameters_engaged",
    }
    assert assignment == {
        "schema": "ny-measure/execution-predicate-assignment/v4",
        "predicate_id": (
            "ny-exec-v1.root-joint-interm-alpha-parameter-engaged"
        ),
        "requested_deliveries": [
            ["NY_ROOT_JOINT_INTERM_ALPHA_ITERS", "direct_environment"],
            ["NY_ROOT_JOINT_INTERM_ALPHA_LR", "direct_environment"],
        ],
        "requested_assignment": ROOT_JOINT_REQUESTED,
        "prerequisites_assignment": prerequisites,
        "full_assignment": full,
        "comparison_assignment": prerequisites,
        "contrast_schema": "ny-search/contrast-design/v1",
        "contrast_id": assignment["contrast_id"],
        "registry": contract["identity"],
        "runtime_context_assignment": [],
        "runtime_context_source_assignment": [],
    }
    assert provenance.validate_execution_predicate_assignment(
        assignment,
        environment=environment,
        search_space_contract=contract,
    )["requested_assignment"] == ROOT_JOINT_REQUESTED


def test_legacy_assignment_schema_replays_only_its_historical_baseline_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _root_predicate_search_space_payload(), source="legacy assignment fixture"
    )
    for name, token in ROOT_LR_PREREQUISITES.items():
        monkeypatch.setenv(name, token)
    monkeypatch.setenv("NY_ROOT_JOINT_INTERM_ALPHA_LR", ROOT_LR_REQUEST)
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.root-joint-interm-alpha-parameter-engaged."
        f"NY_ROOT_JOINT_INTERM_ALPHA_LR.{ROOT_LR_REQUEST}=ran",
    )
    environment = provenance._capture_environment(contract)
    current = provenance.build_execution_predicate_assignment(environment, contract)
    assert current["schema"] == provenance.EXECUTION_PREDICATE_ASSIGNMENT_SCHEMA

    legacy = dict(current)
    legacy["schema"] = provenance.LEGACY_EXECUTION_PREDICATE_ASSIGNMENT_SCHEMA
    legacy.pop("runtime_context_assignment")
    legacy.pop("runtime_context_source_assignment")
    validated = provenance.validate_execution_predicate_assignment(
        legacy,
        environment=environment,
        search_space_contract=contract,
        # A reader may recover sealed configs from an old start. The legacy
        # schema must select its historical algorithm exactly, never guess by
        # trying both baseline interpretations.
        preset_docs=[{"unrelated": {"value": 1}}],
        preset_expected=True,
    )
    assert validated["runtime_context_assignment"] == []
    assert validated["runtime_context_source_assignment"] == []


def test_same_family_joint_assignment_rejects_mixed_delivery_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _root_predicate_search_space_payload(), source="root joint mixed fixture"
    )
    for name, token in ROOT_LR_PREREQUISITES.items():
        monkeypatch.setenv(name, token)
    monkeypatch.setenv("NY_ROOT_JOINT_INTERM_ALPHA_LR", ROOT_LR_REQUEST)
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", ROOT_JOINT_PREDICATE)

    environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ProvenanceError,
        match="cannot mix direct-environment and runtime-config",
    ):
        provenance.build_execution_predicate_assignment(environment, contract)


def test_margin_row_policy_scalar_assignment_uses_closed_family_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="margin policy scalar fixture"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_BETA", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_BETA_PERCOL", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.margin-row-policy-parameter-engaged."
        "NY_MARGIN_ROW_BETA_PERCOL.1=ran",
    )

    environment = provenance._capture_environment(contract)
    assignment = provenance.build_execution_predicate_assignment(
        environment, contract
    )

    assert assignment["requested_assignment"] == [
        ["NY_MARGIN_ROW_BETA_PERCOL", "1"]
    ]
    assert assignment["comparison_assignment"] == [["NY_MARGIN_ROW_BETA", "1"]]
    assert assignment["full_assignment"] == [
        ["NY_MARGIN_ROW_BETA", "1"],
        ["NY_MARGIN_ROW_BETA_PERCOL", "1"],
    ]


def test_cross_category_dependency_repair_seals_only_shared_live_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="cross-category dependency fixture",
    )
    monkeypatch.setenv("NY_MARGIN_ROW_PARALLEL_PREBUILD", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.margin-row-policy-parameter-engaged."
        "NY_MARGIN_ROW_PARALLEL_PREBUILD.1=ran",
    )
    incomplete_environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ProvenanceError,
        match="direct governed inputs do not equal",
    ):
        provenance.build_execution_predicate_assignment(
            incomplete_environment,
            contract,
            preset_docs=[
                {"margin_row": {"parallel": 0}},
                {"margin_row": {"parallel": 16}},
            ],
            preset_expected=True,
        )

    monkeypatch.setenv("NY_MARGIN_ROW_PARALLEL", "2")
    environment = provenance._capture_environment(contract)
    assignment = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[
            {"margin_row": {"parallel": 0}},
            {"margin_row": {"parallel": 16}},
        ],
        preset_expected=True,
    )
    assert assignment["requested_assignment"] == [
        ["NY_MARGIN_ROW_PARALLEL_PREBUILD", "1"]
    ]
    assert assignment["prerequisites_assignment"] == [
        ["NY_MARGIN_ROW_PARALLEL", "2"]
    ]
    assert assignment["comparison_assignment"] == [
        ["NY_MARGIN_ROW_PARALLEL", "2"]
    ]

    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_UPFRONT_ATTACK_AUTO_CAP", "6")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.upfront-parameter-engaged."
        "NY_UPFRONT_ATTACK_AUTO_CAP.6=ran",
    )
    impossible_environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ProvenanceError,
        match="cannot form an attributable same-family contrast",
    ):
        provenance.build_execution_predicate_assignment(
            impossible_environment,
            contract,
            preset_docs=[{}, {"attack": {"upfront_attack": False}}],
            preset_expected=True,
        )


def test_margin_row_policy_same_family_joint_seals_full_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="margin policy joint fixture"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_BETA", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_BETA_COLS", "4")
    monkeypatch.setenv("NY_MARGIN_ROW_BETA_PERCOL", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT", MARGIN_ROW_POLICY_JOINT_PREDICATE
    )

    environment = provenance._capture_environment(contract)
    assignment = provenance.build_execution_predicate_assignment(
        environment, contract
    )

    requested = [
        ["NY_MARGIN_ROW_BETA_COLS", "4"],
        ["NY_MARGIN_ROW_BETA_PERCOL", "1"],
    ]
    comparison = [["NY_MARGIN_ROW_BETA", "1"]]
    assert assignment["predicate_id"] == (
        "ny-exec-v1.margin-row-policy-parameter-engaged"
    )
    assert assignment["requested_assignment"] == requested
    assert assignment["comparison_assignment"] == comparison
    assert assignment["prerequisites_assignment"] == comparison
    assert assignment["full_assignment"] == sorted(requested + comparison)
    assert provenance.validate_execution_predicate_assignment(
        assignment,
        environment=environment,
        search_space_contract=contract,
    )["requested_assignment"] == requested


def test_margin_row_policy_scalar_predicate_requires_live_policy_work() -> None:
    raw = (
        "ny-exec-v1.margin-row-policy-parameter-engaged."
        "NY_MARGIN_ROW_BETA_PERCOL.1=ran"
    )
    record = _execution_predicate_record(
        [("NY_MARGIN_ROW_BETA_PERCOL", "1", True)],
        _margin_row_policy_event(),
    )

    assert provenance.validate_execution_predicate_requirement(record, raw) == (
        provenance.parse_execution_predicate_requirement(raw)
    )

    record["events"] = [
        _margin_row_policy_event(
            beta_per_column_evals="0",
            beta_columns_binding_evals="0",
            beta_columns_max_candidates="0",
        )
    ]
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MARGIN_ROW_BETA_PERCOL=1",
    ):
        provenance.validate_execution_predicate_requirement(record, raw)


def test_current_scalar_assignment_rejects_configured_same_family_hitchhiker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="margin policy receipt fixture"
    )
    raw = (
        "ny-exec-v1.margin-row-policy-parameter-engaged."
        "NY_MARGIN_ROW_BETA_PERCOL.1=ran"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_BETA", "1")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    event = _margin_row_policy_event(
        column_retire="false",
        parallel_prebuild="false",
        newly_retired_columns="0",
        inherited_retirement_evals="0",
        parallel_prebuild_batches="0",
        parallel_prebuild_domains="0",
    )
    record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_BETA", "1", True),
            ("NY_MARGIN_ROW_BETA_PERCOL", "1", True),
            ("NY_MARGIN_ROW_BETA_COLS", "4", 4),
        ],
        event,
    )
    entries = {
        entry["name"]: entry for entry in record["levers"]["receipt"]["levers"]
    }
    entries["NY_MARGIN_ROW_BETA_PERCOL"]["source"] = "config"
    entries["NY_MARGIN_ROW_BETA_COLS"]["source"] = "config"
    record["ambient_env"] = {"NY_MARGIN_ROW_BETA": "1"}
    receipt = record["levers"]["receipt"]
    receipt["env_present"] = 1
    receipt["env_accepted"] = 1

    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="unsealed configured axes: NY_MARGIN_ROW_BETA_COLS",
    ):
        provenance.validate_execution_predicate_requirement(
            record, raw, assignment=assignment
        )

    # An unrequested family member resolved from its declaration default is
    # contextual, not an undeclared treatment coordinate.
    entries["NY_MARGIN_ROW_BETA_COLS"]["source"] = "default"
    entries["NY_MARGIN_ROW_BETA_COLS"]["value"] = 8
    record["events"] = [
        _margin_row_policy_event(
            beta_columns="8",
            beta_columns_binding_evals="0",
            column_retire="false",
            parallel_prebuild="false",
            newly_retired_columns="0",
            inherited_retirement_evals="0",
            parallel_prebuild_batches="0",
            parallel_prebuild_domains="0",
        )
    ]
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)


def test_current_joint_assignment_rejects_configured_same_family_hitchhiker() -> None:
    raw = (
        "ny-exec-v2.exact-vjp-parameter-engaged."
        "NY_PGD_VJP_K.52,NY_PGD_VJP_STEPS.250=ran"
    )
    requested = [
        ["NY_PGD_VJP_K", "52"],
        ["NY_PGD_VJP_STEPS", "250"],
    ]
    assignment = {
        "predicate_id": "ny-exec-v1.exact-vjp-parameter-engaged",
        "requested_assignment": requested,
        "prerequisites_assignment": [],
        "full_assignment": requested,
        "comparison_assignment": [],
        "receipt_source_assignment": [
            ["NY_PGD_VJP_K", "legacy_env"],
            ["NY_PGD_VJP_STEPS", "legacy_env"],
        ],
        "runtime_context_assignment": [],
        "runtime_context_source_assignment": [],
        "contrast_id": "0123456789abcdef",
        "registry": {},
    }
    record = _execution_predicate_record(
        [
            ("NY_PGD_VJP_K", "52", 52),
            ("NY_PGD_VJP_STEPS", "250", 250),
            ("NY_PGD_ODI_STEPS", "4", 4),
        ],
        {"method": "unreached", "status": "skipped", "reason": "fixture"},
    )
    extra = next(
        entry
        for entry in record["levers"]["receipt"]["levers"]
        if entry["name"] == "NY_PGD_ODI_STEPS"
    )
    extra["source"] = "config"
    record["ambient_env"].pop("NY_PGD_ODI_STEPS")
    receipt = record["levers"]["receipt"]
    receipt["env_present"] = 2
    receipt["env_accepted"] = 2

    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="unsealed configured axes: NY_PGD_ODI_STEPS",
    ):
        provenance.validate_execution_predicate_requirement(
            record, raw, assignment=assignment
        )


def test_root_assignment_rejects_unsealed_noncreditable_family_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ny_search  # noqa: PLC0415

    raw = (
        "ny-exec-v1.root-joint-interm-alpha-parameter-engaged."
        f"NY_ROOT_JOINT_INTERM_ALPHA_LR.{ROOT_LR_REQUEST}=ran"
    )
    requested = [["NY_ROOT_JOINT_INTERM_ALPHA_LR", ROOT_LR_REQUEST]]
    prerequisites = [list(pair) for pair in sorted(ROOT_LR_PREREQUISITES.items())]
    full = sorted(requested + prerequisites)
    typed_values = {
        "NY_BN_GPU_EXTRACT": True,
        "NY_CPU_COMPUTE_ENGINE": False,
        "NY_ROOT_JOINT_INTERM_ALPHA": True,
        "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT": True,
        "NY_ROOT_JOINT_INTERM_ALPHA_FROZEN_STOP": True,
        "NY_ROOT_JOINT_INTERM_ALPHA_LR": float(ROOT_LR_REQUEST),
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL": 8,
    }
    record = _execution_predicate_record(
        [(name, token, typed_values[name]) for name, token in full],
        {"method": "unreached", "status": "skipped", "reason": "fixture"},
    )
    receipt = record["levers"]["receipt"]
    receipt["levers"].append(
        {
            "name": "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM",
            "value": 32_768,
            "source": "config",
            "bucket": "debug",
            "moat": "high",
            "provenance": "unmeasured",
        }
    )
    receipt["lever_count"] += 1
    assignment = {
        "predicate_id": "ny-exec-v1.root-joint-interm-alpha-parameter-engaged",
        "requested_assignment": requested,
        "prerequisites_assignment": prerequisites,
        "full_assignment": full,
        "comparison_assignment": prerequisites,
        "receipt_source_assignment": [
            [name, "legacy_env"] for name, _token in full
        ],
        "runtime_context_assignment": [],
        "runtime_context_source_assignment": [],
        "contrast_id": "0123456789abcdef",
        "registry": {},
    }

    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="unsealed configured axes: NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM",
    ):
        provenance.validate_execution_predicate_requirement(
            record, raw, assignment=assignment
        )

    # A declaration/default receipt is context, not a second configured
    # treatment coordinate. Isolate the closure check from the already-covered
    # root terminal validator so this control pins the source distinction.
    receipt["levers"][-1]["source"] = "default"
    receipt["levers"][-1]["value"] = None
    monkeypatch.setattr(
        ny_search,
        "_root_joint_interm_alpha_parameter_engagement",
        lambda wanted, _rows: {name for name, _token in wanted},
    )
    monkeypatch.setattr(
        ny_search,
        "_root_joint_interm_alpha_terminal_evidences",
        lambda _row: [{}],
    )
    monkeypatch.setattr(
        ny_search,
        "_root_joint_explicit_configuration_matches",
        lambda _event, _wanted: True,
    )
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)


def test_margin_row_policy_joint_predicate_requires_every_axis_live() -> None:
    assignment = [
        ("NY_MARGIN_ROW_BETA_COLS", "4", 4),
        ("NY_MARGIN_ROW_BETA_PERCOL", "1", True),
        ("NY_MARGIN_ROW_COL_RETIRE", "1", True),
        ("NY_MARGIN_ROW_PARALLEL_PREBUILD", "1", True),
    ]
    record = _execution_predicate_record(assignment, _margin_row_policy_event())

    assert provenance.validate_execution_predicate_requirement(
        record, MARGIN_ROW_POLICY_ALL_JOINT_PREDICATE
    ) == provenance.parse_execution_predicate_requirement(
        MARGIN_ROW_POLICY_ALL_JOINT_PREDICATE
    )

    record["events"] = [
        _margin_row_policy_event(
            parallel_prebuild_batches="1",
            parallel_prebuild_domains="1",
        )
    ]
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MARGIN_ROW_PARALLEL_PREBUILD=1",
    ):
        provenance.validate_execution_predicate_requirement(
            record, MARGIN_ROW_POLICY_ALL_JOINT_PREDICATE
        )


def test_margin_row_epoch_terminal_is_strict_route_split_and_outer_scored() -> None:
    import ny_search  # noqa: PLC0415

    assignment = [
        ("NY_EPOCH_ATTEMPTS", "4"),
        ("NY_EPOCH_DEPTH", "3"),
        ("NY_EPOCH_RETAIN", "256"),
        ("NY_EPOCH_TIER0", "8"),
        ("NY_EPOCH_UNIVERSE", "128"),
    ]
    row = {"flight": {"events": [_margin_row_epoch_event()]}}
    assert len(ny_search._margin_row_epoch_terminal_evidences(row)) == 1
    assert ny_search._margin_row_epoch_parameter_engagement(
        assignment, [row]
    ) == {name for name, _token in assignment}

    classwise = {"flight": {"events": [
        _margin_row_epoch_event(route="classwise")
    ]}}
    assert len(ny_search._margin_row_epoch_terminal_evidences(classwise)) == 1
    assert not ny_search._margin_row_epoch_parameter_engagement(
        assignment, [classwise]
    )

    incremental = _margin_row_epoch_event(
        configured_parent_topk="16",
        outer_full_attempts="0",
        outer_incremental_attempts="5",
        full_rebuilds_completed="0",
        full_rebuild_refusals="0",
        parent_refreshes_started="5",
        parent_refreshes_completed="4",
        parent_refreshes_accepted="3",
        parent_refresh_refusals="2",
        parent_refresh_columns="8",
        parent_refresh_tightened="3",
        parent_refresh_stabilized="1",
        nested_full_tier0_calls="0",
        nested_incremental_tier0_calls="1",
    )
    incremental_row = {"flight": {"events": [incremental]}}
    assert len(
        ny_search._margin_row_epoch_terminal_evidences(incremental_row)
    ) == 1
    assert not ny_search._margin_row_epoch_parameter_engagement(
        assignment, [incremental_row]
    )

    malformed = {"flight": {"events": [
        _margin_row_epoch_event(attempts_started="4")
    ]}}
    duplicate = {"flight": {"events": [
        _margin_row_epoch_event(), _margin_row_epoch_event()
    ]}}
    nested_only = {"flight": {"events": [
        _margin_row_epoch_event(
            outer_tier0_exact_binding_scored_calls="0"
        )
    ]}}
    assert not ny_search._margin_row_epoch_terminal_evidences(malformed)
    assert not ny_search._margin_row_epoch_terminal_evidences(duplicate)
    assert "NY_EPOCH_TIER0" not in (
        ny_search._margin_row_epoch_parameter_engagement(
            assignment, [nested_only]
        )
    )


def test_margin_row_width_terminal_is_exact_axis_local_and_outer_scored() -> None:
    import ny_search  # noqa: PLC0415

    wanted = [
        ("NY_MARGIN_ROW_K_HEAD", "4"),
        ("NY_MARGIN_ROW_K_TRUNK", "16"),
    ]
    row = {"flight": {"events": [_margin_row_width_event()]}}
    assert len(ny_search._margin_row_width_terminal_evidences(row)) == 1
    assert ny_search._margin_row_width_parameter_engagement(
        wanted, [row]
    ) == {name for name, _token in wanted}

    head_only = {"flight": {"events": [_margin_row_width_event(
        outer_trunk_binding_scored_calls="0",
    )]}}
    assert ny_search._margin_row_width_parameter_engagement(
        wanted, [head_only]
    ) == {"NY_MARGIN_ROW_K_HEAD"}

    trunk_only = {"flight": {"events": [_margin_row_width_event(
        outer_head_binding_scored_calls="0",
    )]}}
    assert ny_search._margin_row_width_parameter_engagement(
        wanted, [trunk_only]
    ) == {"NY_MARGIN_ROW_K_TRUNK"}

    nested_only = {"flight": {"events": [_margin_row_width_event(
        outer_head_selection_calls="0",
        outer_head_available_max="0",
        outer_head_selected_total="0",
        outer_head_binding_scored_calls="0",
        nested_head_selection_calls="11",
    )]}}
    assert "NY_MARGIN_ROW_K_HEAD" not in (
        ny_search._margin_row_width_parameter_engagement(wanted, [nested_only])
    )

    adaptive_lower_only = {"flight": {"events": [_margin_row_width_event(
        contextual_adaptive="true",
    )]}}
    # K also differs, but the lower policy can ratchet its operational width;
    # that selected-set difference is contaminated by the adaptive gate and
    # cannot authenticate a fixed-K treatment.
    assert len(
        ny_search._margin_row_width_terminal_evidences(adaptive_lower_only)
    ) == 1
    assert "NY_MARGIN_ROW_K_HEAD" not in (
        ny_search._margin_row_width_parameter_engagement(
            wanted, [adaptive_lower_only]
        )
    )


def test_margin_row_schedule_requires_clean_overlap_and_stats_terminal() -> None:
    import ny_search  # noqa: PLC0415

    wanted = [("NY_MARGIN_ROW_CONCURRENT", "1")]
    schedule = _margin_row_schedule_event()
    overlap = _margin_row_concurrent_overlap_event()
    width = _margin_row_width_event()
    row = {"flight": {"events": [schedule, overlap, width]}}
    assert len(ny_search._margin_row_schedule_terminal_evidences(row)) == 1
    assert len(ny_search._margin_row_concurrent_overlap_evidences(row)) == 1
    assert ny_search._margin_row_schedule_parameter_engagement(
        wanted, [row]
    ) == {"NY_MARGIN_ROW_CONCURRENT"}
    assert not ny_search._margin_row_schedule_parameter_engagement(
        [("NY_MARGIN_ROW_CONCURRENT", "0")], [row]
    )

    released = _margin_row_schedule_event(
        configured_reserve_secs="45",
        configured_adaptive_reserve="true",
        reserve_route="adaptive_released",
        sparse_root_requested="true",
    )
    released_row = {"flight": {"events": [released, overlap, width]}}
    assert len(
        ny_search._margin_row_schedule_terminal_evidences(released_row)
    ) == 1
    assert ny_search._margin_row_schedule_parameter_engagement(
        wanted, [released_row]
    ) == {"NY_MARGIN_ROW_CONCURRENT"}

    fixed = _margin_row_schedule_event(
        configured_reserve_secs="45",
        configured_adaptive_reserve="false",
        reserve_route="fixed",
        uncapped_reserve_secs="45",
        capped_reserve_secs="45",
        held_reserve_secs="45",
        internal_after_secs="50",
    )
    fixed_row = {"flight": {"events": [fixed, overlap, width]}}
    assert len(ny_search._margin_row_schedule_terminal_evidences(fixed_row)) == 1
    assert not ny_search._margin_row_schedule_parameter_engagement(
        wanted, [fixed_row]
    )

    # Rust's max(10) verifier floor makes this report held=0 despite a real
    # capped reserve and an extended internal window. It is valid telemetry,
    # but never a clean zero-reserve concurrent experiment.
    floor = _margin_row_schedule_event(
        configured_reserve_secs="1",
        configured_adaptive_reserve="false",
        internal_before_secs="5",
        reserve_route="fixed",
        uncapped_reserve_secs="1",
        capped_reserve_secs="1",
        held_reserve_secs="0",
        internal_after_secs="10",
    )
    floor_row = {"flight": {"events": [floor, overlap, width]}}
    assert len(ny_search._margin_row_schedule_terminal_evidences(floor_row)) == 1
    assert not ny_search._margin_row_schedule_parameter_engagement(
        wanted, [floor_row]
    )

    poisoned_width = _margin_row_width_event(policy_mismatch="true")
    poisoned_row = {
        "flight": {"events": [schedule, overlap, poisoned_width]}
    }
    assert len(
        ny_search._margin_row_width_terminal_evidences(poisoned_row)
    ) == 1
    assert not ny_search._margin_row_schedule_parameter_engagement(
        wanted, [poisoned_row]
    )

    impossible = _margin_row_schedule_event(configured_reserve_secs="45")
    impossible_row = {"flight": {"events": [impossible, overlap, width]}}
    # A standalone allocation receipt cannot know whether structural admission
    # later reached the stats-owning lane. The cross-event T1 binder can, and
    # must reject positive reserve + NotApplicable once it did.
    assert len(
        ny_search._margin_row_schedule_terminal_evidences(impossible_row)
    ) == 1
    assert not ny_search._margin_row_schedule_parameter_engagement(
        wanted, [impossible_row]
    )


@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({"configured_reserve_secs": "00"}, "ran"),
        ({"configured_release_frac_bits": str(0x7FF8000000000000)}, "ran"),
        ({"configured_concurrent": "false"}, "ran"),
        ({"reserve_route": "fixed"}, "ran"),
        ({"capped_reserve_secs": "1"}, "ran"),
        ({"internal_after_secs": "94", "held_reserve_secs": "1"}, "ran"),
        ({"sparse_root_requested": "true"}, "ran"),
        ({}, "skipped"),
    ],
)
def test_margin_row_schedule_rejects_impossible_arithmetic_routes_and_bits(
    overrides: dict[str, str], status: str
) -> None:
    import ny_search  # noqa: PLC0415

    event = _margin_row_schedule_event(status=status, **overrides)
    assert not ny_search._margin_row_schedule_terminal_evidences(
        {"flight": {"events": [event]}}
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "configured_verifier_threads": "4",
            "global_pool_route": "applied",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "4",
            "lane_pool_route": "verifier_remainder",
            "lane_pool_threads": "6",
            "private_pool_route": "built",
        },
        {
            "configured_verifier_threads": "4",
            "global_pool_route": "already_initialized",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "10",
            "lane_pool_route": "no_safe_remainder",
            "lane_pool_threads": "none",
            "private_pool_route": "not_requested",
            "concurrent_route": "no_safe_remainder",
            "concurrent_spawned": "false",
        },
        {
            "configured_verifier_threads": "4",
            "host_parallelism": "none",
            "global_pool_route": "applied",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "4",
            "lane_pool_route": "global_unavailable",
            "lane_pool_threads": "none",
            "private_pool_route": "not_requested",
            "concurrent_route": "global_unavailable",
            "concurrent_spawned": "false",
        },
        {
            "configured_concurrent_threads": "3",
            "lane_pool_route": "explicit_private",
            "lane_pool_threads": "3",
            "private_pool_route": "built",
        },
        {
            "configured_concurrent_threads": "3",
            "lane_pool_route": "explicit_private",
            "lane_pool_threads": "3",
            "private_pool_route": "build_failed",
            "concurrent_route": "private_pool_failed",
            "concurrent_spawned": "false",
        },
        {
            "configured_verifier_threads": "4",
            "configured_concurrent_threads": "0",
            "host_parallelism": "none",
            "global_pool_route": "applied",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "4",
        },
        {
            "configured_concurrent_threads": "2",
            "lane_pool_route": "explicit_private",
            "lane_pool_threads": "2",
            "private_pool_route": "built",
            "concurrent_route": "spawn_failed",
            "concurrent_spawned": "false",
        },
        {
            "outer_route": "proof_tail",
            "lane_pool_route": "not_consulted",
            "lane_pool_threads": "none",
            "concurrent_route": "outer_suppressed",
            "concurrent_spawned": "false",
        },
    ],
)
def test_margin_row_schedule_v2_accepts_exact_resource_routes(
    overrides: dict[str, str],
) -> None:
    import ny_search  # noqa: PLC0415

    event = _margin_row_schedule_event(**overrides)
    assert len(ny_search._margin_row_schedule_terminal_evidences(
        {"flight": {"events": [event]}}
    )) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "ny-cli/margin-row-schedule-execution/v1"},
        {"configured_verifier_threads": "+4"},
        {"configured_verifier_threads": "04"},
        {"configured_verifier_threads": str((sys.maxsize * 2) + 2)},
        {"host_parallelism": "0"},
        {"global_pool_requested_threads": "4"},
        {
            "configured_verifier_threads": "4",
            "global_pool_route": "applied",
            "global_pool_requested_threads": "3",
            "global_pool_actual_threads": "3",
        },
        {
            "configured_verifier_threads": "4",
            "global_pool_route": "applied",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "3",
        },
        {
            "configured_verifier_threads": "4",
            "global_pool_route": "already_initialized",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "none",
        },
        {"global_pool_route": "build_failed"},
        {
            "configured_verifier_threads": "4",
            "global_pool_route": "applied",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "4",
            "lane_pool_route": "verifier_remainder",
            "lane_pool_threads": "5",
            "private_pool_route": "built",
        },
        {
            "configured_verifier_threads": "4",
            "host_parallelism": "none",
            "global_pool_route": "applied",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "4",
            "lane_pool_route": "verifier_remainder",
            "lane_pool_threads": "6",
            "private_pool_route": "built",
        },
        {
            "configured_verifier_threads": "4",
            "global_pool_route": "applied",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "4",
            "lane_pool_route": "no_safe_remainder",
            "lane_pool_threads": "none",
            "concurrent_route": "no_safe_remainder",
            "concurrent_spawned": "false",
        },
        {
            "configured_verifier_threads": "4",
            "host_parallelism": "10",
            "global_pool_route": "applied",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "4",
            "lane_pool_route": "global_unavailable",
            "lane_pool_threads": "none",
            "concurrent_route": "global_unavailable",
            "concurrent_spawned": "false",
        },
        {"lane_pool_threads": "none"},
        {
            "configured_verifier_threads": "4",
            "global_pool_route": "applied",
            "global_pool_requested_threads": "4",
            "global_pool_actual_threads": "4",
        },
        {
            "configured_concurrent_threads": "3",
            "lane_pool_route": "explicit_private",
            "lane_pool_threads": "2",
            "private_pool_route": "built",
        },
        {
            "configured_concurrent_threads": "3",
            "lane_pool_route": "explicit_private",
            "lane_pool_threads": "3",
            "private_pool_route": "build_failed",
        },
        {
            "configured_concurrent_threads": "3",
            "lane_pool_route": "explicit_private",
            "lane_pool_threads": "3",
            "private_pool_route": "built",
            "concurrent_route": "private_pool_failed",
            "concurrent_spawned": "false",
        },
        {
            "lane_pool_route": "not_consulted",
            "lane_pool_threads": "0",
            "concurrent_route": "disabled",
            "concurrent_spawned": "false",
            "configured_concurrent": "false",
        },
    ],
)
def test_margin_row_schedule_v2_rejects_resource_contradictions(
    overrides: dict[str, str],
) -> None:
    import ny_search  # noqa: PLC0415

    event = _margin_row_schedule_event(**overrides)
    assert not ny_search._margin_row_schedule_terminal_evidences(
        {"flight": {"events": [event]}}
    )


def test_margin_row_schedule_rejects_reason_and_envelope_ambiguity() -> None:
    import ny_search  # noqa: PLC0415

    schedule = _margin_row_schedule_event()
    tokens = schedule["reason"].split(" ")
    malformed_schedule_events = []
    for changed in (
        tokens[:-1],
        [*tokens, "extra=1"],
        [tokens[1], tokens[0], *tokens[2:]],
    ):
        event = dict(schedule)
        event["reason"] = " ".join(changed)
        malformed_schedule_events.append(event)
    event = dict(schedule)
    event["reason"] = event["reason"].replace(" phase=", "  phase=", 1)
    malformed_schedule_events.append(event)
    for event in malformed_schedule_events:
        assert not ny_search._margin_row_schedule_terminal_evidences(
            {"flight": {"events": [event]}}
        )
    assert not ny_search._margin_row_schedule_terminal_evidences(
        {"flight": {"events": [schedule, schedule]}}
    )

    overlap = _margin_row_concurrent_overlap_event()
    width = _margin_row_width_event()
    wanted = [("NY_MARGIN_ROW_CONCURRENT", "1")]
    malformed_overlap = _margin_row_concurrent_overlap_event(
        worker_active="false"
    )
    envelopes = (
        [schedule, width],
        [schedule, malformed_overlap, width],
        [schedule, overlap, overlap, width],
        [overlap, schedule, width],
        [schedule, overlap, width, width],
    )
    for events in envelopes:
        assert not ny_search._margin_row_schedule_parameter_engagement(
            wanted, [{"flight": {"events": events}}]
        )


def test_margin_row_schedule_rows_cannot_stitch_different_sibling_policies() -> None:
    import ny_search  # noqa: PLC0415

    wanted = [("NY_MARGIN_ROW_CONCURRENT", "1")]
    baseline = {"flight": {"events": [
        _margin_row_schedule_event(),
        _margin_row_concurrent_overlap_event(),
        _margin_row_width_event(),
    ]}}
    alternate = {"flight": {"events": [
        _margin_row_schedule_event(
            configured_release_frac_bits=_f64_bits(0.5)
        ),
        _margin_row_concurrent_overlap_event(),
        _margin_row_width_event(),
    ]}}
    assert ny_search._margin_row_schedule_parameter_engagement(
        wanted, [baseline]
    ) == {"NY_MARGIN_ROW_CONCURRENT"}
    assert ny_search._margin_row_schedule_parameter_engagement(
        wanted, [alternate]
    ) == {"NY_MARGIN_ROW_CONCURRENT"}
    assert not ny_search._margin_row_schedule_parameter_engagement(
        wanted, [baseline, alternate]
    )

    applied = {"flight": {"events": [
        _margin_row_schedule_event(
            configured_verifier_threads="4",
            global_pool_route="applied",
            global_pool_requested_threads="4",
            global_pool_actual_threads="4",
            lane_pool_route="verifier_remainder",
            lane_pool_threads="6",
            private_pool_route="built",
        ),
        _margin_row_concurrent_overlap_event(),
        _margin_row_width_event(),
    ]}}
    already_initialized = {"flight": {"events": [
        _margin_row_schedule_event(
            configured_verifier_threads="4",
            global_pool_route="already_initialized",
            global_pool_requested_threads="4",
            global_pool_actual_threads="5",
            lane_pool_route="verifier_remainder",
            lane_pool_threads="5",
            private_pool_route="built",
        ),
        _margin_row_concurrent_overlap_event(),
        _margin_row_width_event(),
    ]}}
    assert ny_search._margin_row_schedule_parameter_engagement(
        wanted, [applied]
    ) == {"NY_MARGIN_ROW_CONCURRENT"}
    assert ny_search._margin_row_schedule_parameter_engagement(
        wanted, [already_initialized]
    ) == {"NY_MARGIN_ROW_CONCURRENT"}
    assert not ny_search._margin_row_schedule_parameter_engagement(
        wanted, [applied, already_initialized]
    )


def test_margin_row_schedule_v1_is_not_default_filled_into_current_t1() -> None:
    import ny_search  # noqa: PLC0415

    event = _margin_row_schedule_event()
    fields = dict(
        token.split("=", 1) for token in event["reason"].split(" ")
    )
    fields["schema"] = "ny-cli/margin-row-schedule-execution/v1"
    for name in (
        "configured_early_exit", "contextual_early_exit",
        "configured_verifier_threads", "contextual_verifier_threads",
        "configured_concurrent_threads", "contextual_concurrent_threads",
        "host_parallelism", "global_pool_route",
        "global_pool_requested_threads", "global_pool_actual_threads",
        "lane_pool_route", "lane_pool_threads", "private_pool_route",
    ):
        fields.pop(name)
    event["reason"] = " ".join(
        f"{name}={fields[name]}"
        for name in (
            "schema", "phase", "outer_route", "configured_reserve_secs",
            "contextual_reserve_secs", "configured_adaptive_reserve",
            "contextual_adaptive_reserve", "configured_reserve_max_frac_bits",
            "contextual_reserve_max_frac_bits", "configured_release_frac_bits",
            "contextual_release_frac_bits", "configured_concurrent",
            "contextual_concurrent", "concurrent_route", "concurrent_spawned",
            "internal_before_secs", "reserve_route", "uncapped_reserve_secs",
            "capped_reserve_secs", "held_reserve_secs", "internal_after_secs",
            "sparse_root_requested", "policy_mismatch",
        )
    )
    row = {"flight": {"events": [
        event,
        _margin_row_concurrent_overlap_event(),
        _margin_row_width_event(),
    ]}}
    assert not ny_search._margin_row_schedule_terminal_evidences(row)
    assert not ny_search._margin_row_schedule_parameter_engagement(
        [("NY_MARGIN_ROW_CONCURRENT", "1")], [row]
    )


def test_margin_row_schedule_assignment_binds_unsafe_preset_context_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _schedule_v2_source_search_space_payload(), source="schedule context fixture"
    )
    preset = {
        "margin_row": {
            "reserve_secs": 30,
            "adaptive_reserve": True,
            "reserve_max_frac": 0.25,
            "release_frac": 0.30,
            "early_exit": False,
            "verifier_threads": 4,
            "concurrent_threads": 2,
        }
    }
    monkeypatch.setenv("NY_MARGIN_ROW_CONCURRENT", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT", MARGIN_ROW_SCHEDULE_PREDICATE
    )
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[preset],
        preset_expected=True,
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[preset],
        preset_expected=True,
    )
    assert dict(assignment["runtime_context_assignment"]) == {
        "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC": repr(
            struct.unpack(">f", struct.pack(">f", 0.30))[0]
        ),
        "NY_MARGIN_ROW_ADAPTIVE_RESERVE": "1",
        "NY_MARGIN_ROW_CONCURRENT": "1",
        "NY_MARGIN_ROW_CONCURRENT_THREADS": "2",
        "NY_MARGIN_ROW_EARLY_EXIT": "0",
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC": "0.25",
        "NY_MARGIN_ROW_RESERVE_SECS": "30",
        "NY_VERIFIER_THREADS": "4",
    }
    assert dict(assignment["runtime_context_source_assignment"]) == {
        "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC": "config",
        "NY_MARGIN_ROW_ADAPTIVE_RESERVE": "config",
        "NY_MARGIN_ROW_CONCURRENT": "legacy_env",
        "NY_MARGIN_ROW_CONCURRENT_THREADS": "config",
        "NY_MARGIN_ROW_EARLY_EXIT": "config",
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC": "config",
        "NY_MARGIN_ROW_RESERVE_SECS": "config",
        "NY_VERIFIER_THREADS": "config",
    }

    release = struct.unpack(">f", struct.pack(">f", 0.30))[0]
    schedule = _margin_row_schedule_event(
        configured_reserve_secs="30",
        configured_adaptive_reserve="true",
        configured_reserve_max_frac_bits=_f64_bits(0.25),
        configured_release_frac_bits=_f64_bits(release),
        configured_early_exit="false",
        configured_verifier_threads="4",
        configured_concurrent_threads="2",
        global_pool_route="applied",
        global_pool_requested_threads="4",
        global_pool_actual_threads="4",
        lane_pool_route="explicit_private",
        lane_pool_threads="2",
        private_pool_route="built",
        reserve_route="adaptive_released",
        sparse_root_requested="true",
    )
    record = _margin_row_schedule_record(
        [
            schedule,
            _margin_row_concurrent_overlap_event(),
            _margin_row_width_event(),
        ],
        reserve=(30, "config"),
        adaptive=(True, "config"),
        max_fraction=(0.25, "config"),
        release_fraction=(release, "config"),
        early_exit=(False, "config"),
        verifier_threads=(4, "config"),
        concurrent_threads=(2, "config"),
    )
    assert provenance.validate_execution_predicate_requirement(
        record,
        MARGIN_ROW_SCHEDULE_PREDICATE,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(
        MARGIN_ROW_SCHEDULE_PREDICATE
    )

    # All seven unsafe preset siblings deliberately differ from one or more
    # declaration/lower values. They remain valid context, not hidden treatment
    # coordinates, because value and authority are sealed exactly.
    schedule_fields = dict(
        token.split("=", 1) for token in schedule["reason"].split(" ")
    )
    assert schedule_fields["configured_reserve_secs"] != schedule_fields[
        "contextual_reserve_secs"
    ]
    assert schedule_fields["configured_adaptive_reserve"] != schedule_fields[
        "contextual_adaptive_reserve"
    ]
    assert schedule_fields["configured_reserve_max_frac_bits"] != schedule_fields[
        "contextual_reserve_max_frac_bits"
    ]
    assert schedule_fields["configured_release_frac_bits"] != schedule_fields[
        "contextual_release_frac_bits"
    ]
    assert schedule_fields["configured_early_exit"] != schedule_fields[
        "contextual_early_exit"
    ]
    assert schedule_fields["configured_verifier_threads"] != schedule_fields[
        "contextual_verifier_threads"
    ]
    assert schedule_fields["configured_concurrent_threads"] != schedule_fields[
        "contextual_concurrent_threads"
    ]

    forged = json.loads(json.dumps(record))
    reserve_entry = next(
        entry
        for entry in forged["levers"]["receipt"]["levers"]
        if entry["name"] == "NY_MARGIN_ROW_RESERVE_SECS"
    )
    reserve_entry["value"] = 31
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="terminal configuration differs from its exact receipt",
    ):
        provenance.validate_execution_predicate_requirement(
            forged,
            MARGIN_ROW_SCHEDULE_PREDICATE,
            assignment=assignment,
        )

    for axis, forged_value in (
        ("NY_MARGIN_ROW_RESERVE_SECS", False),
        ("NY_MARGIN_ROW_ADAPTIVE_RESERVE", 1),
        ("NY_MARGIN_ROW_CONCURRENT", 1),
        ("NY_MARGIN_ROW_RESERVE_MAX_FRAC", False),
        ("NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC", 10**10000),
        ("NY_MARGIN_ROW_EARLY_EXIT", 0),
        ("NY_VERIFIER_THREADS", True),
        ("NY_MARGIN_ROW_CONCURRENT_THREADS", False),
    ):
        forged = json.loads(json.dumps(record))
        entry = next(
            candidate
            for candidate in forged["levers"]["receipt"]["levers"]
            if candidate["name"] == axis
        )
        entry["value"] = forged_value
        with pytest.raises(
            provenance.ExecutionPredicateError,
            match="terminal configuration differs from its exact receipt",
        ):
            provenance.validate_execution_predicate_requirement(
                forged,
                MARGIN_ROW_SCHEDULE_PREDICATE,
                assignment=assignment,
            )

    forged = json.loads(json.dumps(record))
    adaptive_entry = next(
        entry
        for entry in forged["levers"]["receipt"]["levers"]
        if entry["name"] == "NY_MARGIN_ROW_ADAPTIVE_RESERVE"
    )
    adaptive_entry["source"] = "default"
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="runtime authority 'default' differs from sealed authority 'config'",
    ):
        provenance.validate_execution_predicate_requirement(
            forged,
            MARGIN_ROW_SCHEDULE_PREDICATE,
            assignment=assignment,
        )

    forged = json.loads(json.dumps(record))
    forged_schedule = dict(forged["events"][0])
    forged_schedule["reason"] = forged_schedule["reason"].replace(
        "contextual_reserve_secs=45", "contextual_reserve_secs=44"
    )
    forged["events"][0] = forged_schedule
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="lower-authority policy differs",
    ):
        provenance.validate_execution_predicate_requirement(
            forged,
            MARGIN_ROW_SCHEDULE_PREDICATE,
            assignment=assignment,
        )

    for original, replacement in (
        ("contextual_early_exit=true", "contextual_early_exit=false"),
        ("contextual_verifier_threads=none", "contextual_verifier_threads=0"),
        (
            "contextual_concurrent_threads=none",
            "contextual_concurrent_threads=0",
        ),
    ):
        forged = json.loads(json.dumps(record))
        forged_schedule = dict(forged["events"][0])
        forged_schedule["reason"] = forged_schedule["reason"].replace(
            original, replacement
        )
        forged["events"][0] = forged_schedule
        with pytest.raises(
            provenance.ExecutionPredicateError,
            match="lower-authority policy differs",
        ):
            provenance.validate_execution_predicate_requirement(
                forged,
                MARGIN_ROW_SCHEDULE_PREDICATE,
                assignment=assignment,
            )


@pytest.mark.parametrize(
    "preset",
    [
        {"margin_row": {"reserve_secs": "0"}},
        {"margin_row": {"reserve_secs": False}},
        {"margin_row": {"adaptive_reserve": 1}},
        {"margin_row": {"reserve_max_frac": "0.25"}},
        {"margin_row": {"release_frac": []}},
        {"margin_row": {"early_exit": 1}},
        {"margin_row": {"verifier_threads": True}},
        {"margin_row": {"verifier_threads": -1}},
        {"margin_row": {"verifier_threads": (sys.maxsize * 2) + 2}},
        {"margin_row": {"concurrent_threads": "2"}},
        {"margin_row": {"concurrent_threads": True}},
        {"margin_row": []},
    ],
)
def test_margin_row_schedule_rejects_preset_values_rust_would_not_deserialize(
    monkeypatch: pytest.MonkeyPatch,
    preset: dict[str, object],
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _schedule_v2_source_search_space_payload(), source="schedule typed preset fixture"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_CONCURRENT", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT", MARGIN_ROW_SCHEDULE_PREDICATE
    )
    environment = provenance._capture_environment(contract)
    with pytest.raises(provenance.ProvenanceError, match="preset"):
        provenance.build_execution_predicate_assignment(
            environment,
            contract,
            preset_docs=[preset],
            preset_expected=True,
        )


def test_margin_row_schedule_typed_declined_fractions_restore_lower_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _schedule_v2_source_search_space_payload(), source="schedule declined preset fixture"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_CONCURRENT", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT", MARGIN_ROW_SCHEDULE_PREDICATE
    )
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[{
            "margin_row": {
                "reserve_secs": 0,
                "reserve_max_frac": 0.0,
                "release_frac": float("nan"),
            }
        }],
        preset_expected=True,
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[{
            "margin_row": {
                "reserve_secs": 0,
                "reserve_max_frac": 0.0,
                "release_frac": float("nan"),
            }
        }],
        preset_expected=True,
    )
    assert dict(assignment["runtime_context_assignment"])[
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC"
    ] == "none"
    assert dict(assignment["runtime_context_assignment"])[
        "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC"
    ] == "0.40"
    sources = dict(assignment["runtime_context_source_assignment"])
    assert sources["NY_MARGIN_ROW_RESERVE_MAX_FRAC"] == "default"
    assert sources["NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC"] == "default"


def test_margin_row_schedule_optional_usize_preserves_zero_vs_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _schedule_v2_source_search_space_payload(),
        source="schedule optional usize fixture",
    )
    preset = {"margin_row": {
        "reserve_secs": 0,
        "verifier_threads": 0,
        "concurrent_threads": 0,
    }}
    monkeypatch.setenv("NY_MARGIN_ROW_CONCURRENT", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT", MARGIN_ROW_SCHEDULE_PREDICATE
    )
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[preset],
        preset_expected=True,
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[preset],
        preset_expected=True,
    )
    context = dict(assignment["runtime_context_assignment"])
    sources = dict(assignment["runtime_context_source_assignment"])
    assert context["NY_VERIFIER_THREADS"] == "0"
    assert context["NY_MARGIN_ROW_CONCURRENT_THREADS"] == "0"
    assert sources["NY_VERIFIER_THREADS"] == "config"
    assert sources["NY_MARGIN_ROW_CONCURRENT_THREADS"] == "config"

    record = _margin_row_schedule_record(
        [
            _margin_row_schedule_event(
                configured_verifier_threads="0",
                configured_concurrent_threads="0",
            ),
            _margin_row_concurrent_overlap_event(),
            _margin_row_width_event(),
        ],
        verifier_threads=(0, "config"),
        concurrent_threads=(0, "config"),
    )
    assert provenance.validate_execution_predicate_requirement(
        record,
        MARGIN_ROW_SCHEDULE_PREDICATE,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(
        MARGIN_ROW_SCHEDULE_PREDICATE
    )


@pytest.mark.parametrize(
    ("axis", "token"),
    [
        ("NY_MARGIN_ROW_RESERVE_SECS", "30"),
        ("NY_MARGIN_ROW_ADAPTIVE_RESERVE", "1"),
        ("NY_MARGIN_ROW_RESERVE_MAX_FRAC", "0.25"),
        ("NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC", "0.3"),
        ("NY_MARGIN_ROW_EARLY_EXIT", "0"),
        ("NY_VERIFIER_THREADS", "4"),
        ("NY_MARGIN_ROW_CONCURRENT_THREADS", "2"),
    ],
)
def test_margin_row_schedule_refuses_unsafe_direct_assignment_hitchhikers(
    monkeypatch: pytest.MonkeyPatch,
    axis: str,
    token: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _schedule_v2_source_search_space_payload(), source="schedule unsafe fixture"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_CONCURRENT", "1")
    monkeypatch.setenv(axis, token)
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT", MARGIN_ROW_SCHEDULE_PREDICATE
    )
    environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ProvenanceError,
        match="unsearchable full-policy coordinates",
    ):
        provenance.build_execution_predicate_assignment(
            environment,
            contract,
            preset_docs=[{"margin_row": {"reserve_secs": 0}}],
            preset_expected=True,
        )


def test_margin_row_schedule_predicate_rejects_missing_duplicate_and_sibling_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _schedule_v2_source_search_space_payload(), source="schedule adversarial fixture"
    )
    preset = {"margin_row": {"reserve_secs": 0}}
    monkeypatch.setenv("NY_MARGIN_ROW_CONCURRENT", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT", MARGIN_ROW_SCHEDULE_PREDICATE
    )
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(
        environment, contract, preset_docs=[preset], preset_expected=True
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[preset],
        preset_expected=True,
    )

    def clean_record() -> dict[str, object]:
        return _margin_row_schedule_record([
            _margin_row_schedule_event(),
            _margin_row_concurrent_overlap_event(),
            _margin_row_width_event(),
        ])

    missing = clean_record()
    missing_entries = missing["levers"]["receipt"]["levers"]
    missing_entries[:] = [
        entry for entry in missing_entries
        if entry["name"] != "NY_MARGIN_ROW_RESERVE_MAX_FRAC"
    ]
    missing["levers"]["receipt"]["lever_count"] -= 1
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="must appear exactly once in the margin-row schedule receipt context",
    ):
        provenance.validate_execution_predicate_requirement(
            missing, MARGIN_ROW_SCHEDULE_PREDICATE, assignment=assignment
        )

    duplicate = clean_record()
    duplicate["events"].insert(1, _margin_row_concurrent_overlap_event())
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="did not prove live engagement",
    ):
        provenance.validate_execution_predicate_requirement(
            duplicate, MARGIN_ROW_SCHEDULE_PREDICATE, assignment=assignment
        )

    extra_width = clean_record()
    extra_width["events"].append(_margin_row_width_event())
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="did not prove live engagement",
    ):
        provenance.validate_execution_predicate_requirement(
            extra_width, MARGIN_ROW_SCHEDULE_PREDICATE, assignment=assignment
        )

    sibling = clean_record()
    sibling["events"][0] = _margin_row_schedule_event(
        configured_adaptive_reserve="true"
    )
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="terminal configuration differs from its exact receipt",
    ):
        provenance.validate_execution_predicate_requirement(
            sibling, MARGIN_ROW_SCHEDULE_PREDICATE, assignment=assignment
        )


def test_margin_row_k_adapt_credits_both_exact_contextual_directions() -> None:
    import ny_search  # noqa: PLC0415

    on_row = {"flight": {"events": [_margin_row_k_adapt_on_event()]}}
    off_row = {"flight": {"events": [_margin_row_k_adapt_off_event()]}}
    assert len(ny_search._margin_row_width_terminal_evidences(on_row)) == 1
    assert len(ny_search._margin_row_width_terminal_evidences(off_row)) == 1
    assert ny_search._margin_row_width_parameter_engagement(
        [("NY_MARGIN_ROW_K_ADAPT", "1")], [on_row]
    ) == {"NY_MARGIN_ROW_K_ADAPT"}
    assert ny_search._margin_row_width_parameter_engagement(
        [("NY_MARGIN_ROW_K_ADAPT", "0")], [off_row]
    ) == {"NY_MARGIN_ROW_K_ADAPT"}


@pytest.mark.parametrize(
    ("token", "event"),
    (
        (
            "1",
            _margin_row_k_adapt_on_event(
                effective_k_head_max="8",
                effective_k_trunk_max="8",
                outer_head_selected_total="16",
                outer_head_binding_scored_calls="0",
                outer_trunk_selected_total="16",
                outer_trunk_binding_scored_calls="0",
                configured_adaptive_threshold_opportunities="0",
                configured_adaptive_escalations="0",
            ),
        ),
        (
            "1",
            _margin_row_k_adapt_on_event(
                configured_k_head="32",
                contextual_k_head="32",
                configured_k_trunk="32",
                contextual_k_trunk="32",
                effective_k_head_max="32",
                effective_k_trunk_max="32",
                effective_contextual_k_head_max="32",
                effective_contextual_k_trunk_max="32",
                outer_head_selected_total="40",
                outer_head_binding_scored_calls="0",
                outer_trunk_selected_total="40",
                outer_trunk_binding_scored_calls="0",
                configured_adaptive_escalations="0",
            ),
        ),
        (
            "1",
            _margin_row_k_adapt_on_event(
                outer_head_binding_scored_calls="0",
                outer_trunk_binding_scored_calls="0",
            ),
        ),
        (
            "1",
            _margin_row_k_adapt_on_event(
                tier0_exact="8",
                outer_head_selection_calls="0",
                outer_head_available_max="0",
                outer_head_selected_total="0",
                outer_head_binding_scored_calls="0",
                outer_trunk_selection_calls="0",
                outer_trunk_available_max="0",
                outer_trunk_selected_total="0",
                outer_trunk_binding_scored_calls="0",
                configured_adaptive_threshold_opportunities="0",
                configured_adaptive_escalations="0",
            ),
        ),
        (
            "1",
            _margin_row_k_adapt_on_event(
                contextual_k_head="4",
                effective_contextual_k_head_max="4",
            ),
        ),
        (
            "0",
            _margin_row_k_adapt_off_event(
                outer_head_selection_calls="0",
                outer_head_available_max="0",
                outer_head_selected_total="0",
                outer_head_binding_scored_calls="0",
                outer_trunk_selection_calls="0",
                outer_trunk_available_max="0",
                outer_trunk_selected_total="0",
                outer_trunk_binding_scored_calls="0",
                nested_head_selection_calls="11",
                nested_trunk_selection_calls="11",
            ),
        ),
    ),
)
def test_margin_row_k_adapt_rejects_no_threshold_saturation_unscored_tier0_and_joint_k(
    token: str,
    event: dict[str, object],
) -> None:
    import ny_search  # noqa: PLC0415

    row = {"flight": {"events": [event]}}
    assert len(ny_search._margin_row_width_terminal_evidences(row)) == 1
    assert "NY_MARGIN_ROW_K_ADAPT" not in (
        ny_search._margin_row_width_parameter_engagement(
            [("NY_MARGIN_ROW_K_ADAPT", token)], [row]
        )
    )


def test_margin_row_k_adapt_cannot_borrow_engagement_from_an_inactive_row() -> None:
    import ny_search  # noqa: PLC0415

    active = {"flight": {"events": [_margin_row_k_adapt_on_event()]}}
    inactive = {"flight": {"events": [_margin_row_k_adapt_on_event(
        outer_head_binding_scored_calls="0",
        outer_trunk_binding_scored_calls="0",
    )]}}
    wanted = [("NY_MARGIN_ROW_K_ADAPT", "1")]
    assert ny_search._margin_row_width_parameter_engagement(
        wanted, [active, inactive]
    ) == {"NY_MARGIN_ROW_K_ADAPT"}
    assert not ny_search._every_row_parameter_engagement(
        ny_search._margin_row_width_parameter_engagement,
        wanted,
        [active, inactive],
    )


def test_margin_row_width_terminal_rejects_malformed_duplicate_and_conflict() -> None:
    import ny_search  # noqa: PLC0415

    valid = _margin_row_width_event()
    missing = _margin_row_width_event()
    missing["reason"] = str(missing["reason"]).replace(
        " policy_mismatch=false", ""
    )
    extra = _margin_row_width_event()
    extra["reason"] = str(extra["reason"]) + " extra=1"
    duplicate_field = _margin_row_width_event()
    duplicate_field["reason"] = (
        str(duplicate_field["reason"]) + " configured_k_head=4"
    )
    tabbed = _margin_row_width_event()
    tabbed["reason"] = str(tabbed["reason"]).replace(" ", "\t", 1)
    noncanonical = _margin_row_width_event()
    noncanonical["reason"] = str(noncanonical["reason"]).replace(
        "configured_k_head=4", "configured_k_head=04", 1
    )
    bad_schema = _margin_row_width_event()
    bad_schema["reason"] = str(bad_schema["reason"]).replace(
        "ny-margin-row-width-execution/v2",
        "ny-margin-row-width-execution/v1",
        1,
    )
    not_run = _margin_row_width_event(outcome="not-run")
    for event in (
        missing, extra, duplicate_field, tabbed, noncanonical, bad_schema,
        not_run,
    ):
        assert not ny_search._margin_row_width_terminal_evidences(
            {"flight": {"events": [event]}}
        )

    assert not ny_search._margin_row_width_terminal_evidences(
        {"flight": {"events": [valid, _margin_row_width_event()]}}
    )
    conflict = _margin_row_width_event(
        configured_k_head="5",
        effective_k_head_max="5",
    )
    assert not ny_search._margin_row_width_terminal_evidences(
        {"flight": {"events": [valid, conflict]}}
    )


@pytest.mark.parametrize(
    "event",
    (
        _margin_row_width_event(
            configured_adaptive_threshold_opportunities="1",
        ),
        _margin_row_k_adapt_on_event(
            effective_k_head_max="8",
            effective_k_trunk_max="8",
        ),
        _margin_row_k_adapt_on_event(
            configured_adaptive_escalations="0",
        ),
        _margin_row_k_adapt_on_event(
            configured_adaptive_escalations="3",
        ),
        _margin_row_k_adapt_on_event(
            effective_contextual_k_head_max="16",
        ),
        _margin_row_k_adapt_on_event(
            outer_head_available_max="8",
        ),
        _margin_row_k_adapt_on_event(
            outer_head_selected_total="41",
        ),
        _margin_row_k_adapt_on_event(
            configured_adaptive_threshold_opportunities=str(1 << 64),
        ),
        _margin_row_k_adapt_on_event(
            contextual_adaptive="true",
            effective_contextual_k_head_max="16",
            effective_contextual_k_trunk_max="16",
            outer_head_binding_scored_calls="0",
            outer_trunk_binding_scored_calls="0",
            contextual_adaptive_threshold_opportunities="1",
            contextual_adaptive_escalations="1",
        ),
    ),
)
def test_margin_row_width_v2_rejects_impossible_adaptive_aggregates(
    event: dict[str, object],
) -> None:
    import ny_search  # noqa: PLC0415

    assert not ny_search._margin_row_width_terminal_evidences(
        {"flight": {"events": [event]}}
    )


@pytest.mark.parametrize(
    "event",
    (
        _margin_row_width_event(
            contextual_k_head="4",
            outer_head_binding_scored_calls="0",
        ),
        _margin_row_width_event(tier0_exact="8"),
        _margin_row_width_event(
            adaptive_enabled="true",
            effective_k_head_max="16",
            configured_adaptive_threshold_opportunities="1",
            configured_adaptive_escalations="1",
        ),
        _margin_row_width_event(policy_mismatch="true"),
        _margin_row_width_event(
            effective_k_head_max="8",
        ),
        _margin_row_width_event(
            outer_head_selection_calls="0",
            outer_head_available_max="0",
            outer_head_selected_total="0",
            outer_head_binding_scored_calls="0",
        ),
    ),
)
def test_margin_row_width_rejects_noop_tier0_adaptive_or_inactive_head(
    event: dict[str, object],
) -> None:
    import ny_search  # noqa: PLC0415

    row = {"flight": {"events": [event]}}
    assert "NY_MARGIN_ROW_K_HEAD" not in (
        ny_search._margin_row_width_parameter_engagement(
            [("NY_MARGIN_ROW_K_HEAD", "4")], [row]
        )
    )


def test_margin_row_width_rows_and_axes_cannot_lend_binding_credit() -> None:
    import ny_search  # noqa: PLC0415

    wanted = [
        ("NY_MARGIN_ROW_K_HEAD", "4"),
        ("NY_MARGIN_ROW_K_TRUNK", "16"),
    ]
    head_row = {"flight": {"events": [_margin_row_width_event(
        outer_trunk_binding_scored_calls="0",
    )]}}
    trunk_row = {"flight": {"events": [_margin_row_width_event(
        outer_head_binding_scored_calls="0",
    )]}}

    # The family validator reports the union of live axes over a set; the T1
    # all-row wrapper must intersect singleton witnesses before score credit.
    assert ny_search._margin_row_width_parameter_engagement(
        wanted, [head_row, trunk_row]
    ) == {name for name, _token in wanted}
    assert not ny_search._every_row_parameter_engagement(
        ny_search._margin_row_width_parameter_engagement,
        wanted,
        [head_row, trunk_row],
    )

    both_row = {"flight": {"events": [_margin_row_width_event()]}}
    assert ny_search._every_row_parameter_engagement(
        ny_search._margin_row_width_parameter_engagement,
        wanted,
        [head_row, both_row],
    ) == {"NY_MARGIN_ROW_K_HEAD"}


def test_margin_row_width_execution_predicate_authenticates_live_source() -> None:
    raw = (
        "ny-exec-v1.margin-row-width-parameter-engaged."
        "NY_MARGIN_ROW_K_HEAD.4=ran"
    )
    record = _execution_predicate_record(
        [("NY_MARGIN_ROW_K_HEAD", "4", 4)],
        _margin_row_width_event(),
    )
    assert provenance.validate_execution_predicate_requirement(
        record, raw
    ) == provenance.parse_execution_predicate_requirement(raw)

    joint_record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_K_HEAD", "4", 4),
            ("NY_MARGIN_ROW_K_TRUNK", "16", 16),
        ],
        _margin_row_width_event(),
    )
    assert provenance.validate_execution_predicate_requirement(
        joint_record, MARGIN_ROW_WIDTH_JOINT_PREDICATE
    ) == provenance.parse_execution_predicate_requirement(
        MARGIN_ROW_WIDTH_JOINT_PREDICATE
    )

    adaptive_raw = (
        "ny-exec-v1.margin-row-width-parameter-engaged."
        "NY_MARGIN_ROW_K_ADAPT.1=ran"
    )
    adaptive_record = _execution_predicate_record(
        [("NY_MARGIN_ROW_K_ADAPT", "1", True)],
        _margin_row_k_adapt_on_event(),
    )
    assert provenance.validate_execution_predicate_requirement(
        adaptive_record, adaptive_raw
    ) == provenance.parse_execution_predicate_requirement(adaptive_raw)

    entry = next(
        item for item in record["levers"]["receipt"]["levers"]
        if item["name"] == "NY_MARGIN_ROW_K_HEAD"
    )
    entry["source"] = "config"
    record["levers"]["receipt"]["env_present"] -= 1
    record["levers"]["receipt"]["env_accepted"] -= 1
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="does not match its receipted child authority",
    ):
        provenance.validate_execution_predicate_requirement(record, raw)

    adaptive_entry = next(
        item for item in adaptive_record["levers"]["receipt"]["levers"]
        if item["name"] == "NY_MARGIN_ROW_K_ADAPT"
    )
    adaptive_entry["source"] = "config"
    adaptive_record["levers"]["receipt"]["env_present"] -= 1
    adaptive_record["levers"]["receipt"]["env_accepted"] -= 1
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="does not match its receipted child authority",
    ):
        provenance.validate_execution_predicate_requirement(
            adaptive_record, adaptive_raw
        )


def test_margin_row_width_assignment_seals_unrequested_sibling_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="margin width context fixture"
    )
    raw = (
        "ny-exec-v1.margin-row-width-parameter-engaged."
        "NY_MARGIN_ROW_K_HEAD.4=ran"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_K_HEAD", "4")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(
        environment, contract
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    assert assignment["runtime_context_assignment"] == [
        ["NY_MARGIN_ROW_K_ADAPT", "0"],
        ["NY_MARGIN_ROW_K_HEAD", "4"],
        ["NY_MARGIN_ROW_K_TRUNK", "8"],
    ]
    assert assignment["runtime_context_source_assignment"] == [
        ["NY_MARGIN_ROW_K_ADAPT", "default"],
        ["NY_MARGIN_ROW_K_HEAD", "legacy_env"],
        ["NY_MARGIN_ROW_K_TRUNK", "default"],
    ]

    event = _margin_row_width_event(
        configured_k_trunk="8",
        contextual_k_trunk="8",
        effective_k_trunk_max="8",
        outer_trunk_available_max="8",
        outer_trunk_selected_total="16",
        outer_trunk_binding_scored_calls="0",
    )
    record = _execution_predicate_record(
        [("NY_MARGIN_ROW_K_HEAD", "4", 4)], event
    )
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)

    forged = _margin_row_width_event(
        configured_k_trunk="16",
        contextual_k_trunk="8",
        effective_k_trunk_max="16",
    )
    forged_record = _execution_predicate_record(
        [("NY_MARGIN_ROW_K_HEAD", "4", 4)], forged
    )
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="terminal configuration differs from the sealed runtime context",
    ):
        provenance.validate_execution_predicate_requirement(
            forged_record, raw, assignment=assignment
        )


def test_margin_row_width_receipt_context_normalizes_typed_zero_sentinel() -> None:
    import ny_search  # noqa: PLC0415

    event = _margin_row_width_event(
        configured_k_trunk="8",
        contextual_k_trunk="8",
        effective_k_trunk_max="8",
        outer_trunk_available_max="8",
        outer_trunk_selected_total="16",
        outer_trunk_binding_scored_calls="0",
    )
    entries = [
        {
            "name": "NY_MARGIN_ROW_K_ADAPT",
            "value": None,
            "source": "default",
        },
        {
            "name": "NY_MARGIN_ROW_K_HEAD",
            "value": 4,
            "source": "legacy_env",
        },
        {
            "name": "NY_MARGIN_ROW_K_TRUNK",
            "value": 0,
            "source": "config",
        },
    ]
    assert provenance._margin_row_width_receipt_context_errors(
        ny_search,
        {"events": [event]},
        entries,
        [
            ["NY_MARGIN_ROW_K_ADAPT", "0"],
            ["NY_MARGIN_ROW_K_HEAD", "4"],
            ["NY_MARGIN_ROW_K_TRUNK", "0"],
        ],
        [
            ["NY_MARGIN_ROW_K_ADAPT", "default"],
            ["NY_MARGIN_ROW_K_HEAD", "legacy_env"],
            ["NY_MARGIN_ROW_K_TRUNK", "config"],
        ],
    ) == []

    for forged_source in ("config", "legacy_env"):
        entries[0]["source"] = forged_source
        forged_sources = [
            ["NY_MARGIN_ROW_K_ADAPT", forged_source],
            ["NY_MARGIN_ROW_K_HEAD", "legacy_env"],
            ["NY_MARGIN_ROW_K_TRUNK", "config"],
        ]
        assert any(
            "terminal configuration differs from its exact receipt" in error
            for error in provenance._margin_row_width_receipt_context_errors(
                ny_search,
                {"events": [event]},
                entries,
                [
                    ["NY_MARGIN_ROW_K_ADAPT", "0"],
                    ["NY_MARGIN_ROW_K_HEAD", "4"],
                    ["NY_MARGIN_ROW_K_TRUNK", "0"],
                ],
                forged_sources,
            )
        )
    entries[0]["source"] = "default"
    entries[0]["value"] = False
    assert any(
        "terminal configuration differs from its exact receipt" in error
        for error in provenance._margin_row_width_receipt_context_errors(
            ny_search,
            {"events": [event]},
            entries,
            [
                ["NY_MARGIN_ROW_K_ADAPT", "0"],
                ["NY_MARGIN_ROW_K_HEAD", "4"],
                ["NY_MARGIN_ROW_K_TRUNK", "0"],
            ],
            [
                ["NY_MARGIN_ROW_K_ADAPT", "default"],
                ["NY_MARGIN_ROW_K_HEAD", "legacy_env"],
                ["NY_MARGIN_ROW_K_TRUNK", "config"],
            ],
        )
    )
    entries[0]["value"] = None
    entries[2]["value"] = 8
    assert any(
        "sealed runtime context" in error
        for error in provenance._margin_row_width_receipt_context_errors(
            ny_search,
            {"events": [event]},
            entries,
            [
                ["NY_MARGIN_ROW_K_ADAPT", "0"],
                ["NY_MARGIN_ROW_K_HEAD", "4"],
                ["NY_MARGIN_ROW_K_TRUNK", "0"],
            ],
            [
                ["NY_MARGIN_ROW_K_ADAPT", "default"],
                ["NY_MARGIN_ROW_K_HEAD", "legacy_env"],
                ["NY_MARGIN_ROW_K_TRUNK", "config"],
            ],
        )
    )


def test_margin_row_k_adapt_receipt_context_binds_value_and_authority() -> None:
    import ny_search  # noqa: PLC0415

    event = _margin_row_k_adapt_on_event()
    entries = [
        {
            "name": "NY_MARGIN_ROW_K_ADAPT",
            "value": True,
            "source": "legacy_env",
        },
        {
            "name": "NY_MARGIN_ROW_K_HEAD",
            "value": 8,
            "source": "default",
        },
        {
            "name": "NY_MARGIN_ROW_K_TRUNK",
            "value": 8,
            "source": "default",
        },
    ]
    context = [
        ["NY_MARGIN_ROW_K_ADAPT", "1"],
        ["NY_MARGIN_ROW_K_HEAD", "8"],
        ["NY_MARGIN_ROW_K_TRUNK", "8"],
    ]
    sources = [
        ["NY_MARGIN_ROW_K_ADAPT", "legacy_env"],
        ["NY_MARGIN_ROW_K_HEAD", "default"],
        ["NY_MARGIN_ROW_K_TRUNK", "default"],
    ]
    assert provenance._margin_row_width_receipt_context_errors(
        ny_search,
        {"events": [event]},
        entries,
        context,
        sources,
    ) == []

    entries[0]["value"] = False
    assert any(
        "terminal configuration differs from its exact receipt" in error
        for error in provenance._margin_row_width_receipt_context_errors(
            ny_search, {"events": [event]}, entries, context, sources
        )
    )
    entries[0]["value"] = True
    entries[0]["source"] = "config"
    assert any(
        "differs from sealed authority" in error
        for error in provenance._margin_row_width_receipt_context_errors(
            ny_search, {"events": [event]}, entries, context, sources
        )
    )


def test_margin_row_epoch_joint_assignment_seals_unsafe_parent_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="epoch joint fixture"
    )
    requested = [
        ("NY_EPOCH_ATTEMPTS", "4", 4),
        ("NY_EPOCH_DEPTH", "3", 3),
        ("NY_EPOCH_RETAIN", "256", 256),
        ("NY_EPOCH_TIER0", "8", 8),
        ("NY_EPOCH_UNIVERSE", "128", 128),
    ]
    for name, token, _value in requested:
        monkeypatch.setenv(name, token)
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        MARGIN_ROW_EPOCH_ALL_JOINT_PREDICATE,
    )
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    assert dict(assignment["runtime_context_assignment"]) == {
        "NY_EPOCH_ATTEMPTS": "4",
        "NY_EPOCH_DEPTH": "3",
        "NY_EPOCH_RETAIN": "256",
        "NY_EPOCH_TIER0": "8",
        "NY_EPOCH_UNIVERSE": "128",
        "NY_MARGIN_ROW_EPOCH_PARENT_TOPK": "0",
    }
    assert dict(assignment["runtime_context_source_assignment"])[
        "NY_MARGIN_ROW_EPOCH_PARENT_TOPK"
    ] == "default"

    record = _execution_predicate_record(
        requested, _margin_row_epoch_event()
    )
    assert provenance.validate_execution_predicate_requirement(
        record,
        MARGIN_ROW_EPOCH_ALL_JOINT_PREDICATE,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(
        MARGIN_ROW_EPOCH_ALL_JOINT_PREDICATE
    )
    parent = next(
        entry
        for entry in record["levers"]["receipt"]["levers"]
        if entry["name"] == "NY_MARGIN_ROW_EPOCH_PARENT_TOPK"
    )
    parent["value"] = 16
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MARGIN_ROW_EPOCH_PARENT_TOPK=0 terminal configuration differs",
    ):
        provenance.validate_execution_predicate_requirement(
            record,
            MARGIN_ROW_EPOCH_ALL_JOINT_PREDICATE,
            assignment=assignment,
        )
    parent["value"] = 0
    parent["source"] = "config"
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="runtime authority 'config' differs from sealed authority 'default'",
    ):
        provenance.validate_execution_predicate_requirement(
            record,
            MARGIN_ROW_EPOCH_ALL_JOINT_PREDICATE,
            assignment=assignment,
        )


def test_margin_row_epoch_refuses_unsafe_parent_env_or_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="epoch unsafe fixture"
    )
    monkeypatch.setenv("NY_EPOCH_DEPTH", "3")
    monkeypatch.setenv("NY_MARGIN_ROW_EPOCH_PARENT_TOPK", "16")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.margin-row-epoch-parameter-engaged."
        "NY_EPOCH_DEPTH.3=ran",
    )
    environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ProvenanceError,
        match="unsearchable full-policy coordinates",
    ):
        provenance.build_execution_predicate_assignment(environment, contract)

    monkeypatch.delenv("NY_MARGIN_ROW_EPOCH_PARENT_TOPK")
    environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ProvenanceError,
        match="must remain sealed at 0",
    ):
        provenance.build_execution_predicate_assignment(
            environment,
            contract,
            preset_docs=[{"margin_row": {"epoch_parent_topk": 16}}],
            preset_expected=True,
        )


@pytest.mark.parametrize(
    ("axis", "preset_key", "baseline", "treatment"),
    [
        ("NY_EPOCH_DEPTH", "epoch_depth", 2, "3"),
        ("NY_EPOCH_ATTEMPTS", "epoch_attempts", 1, "4"),
        ("NY_EPOCH_TIER0", "epoch_tier0", 4, "8"),
        ("NY_EPOCH_UNIVERSE", "epoch_universe", 32, "128"),
        ("NY_EPOCH_RETAIN", "epoch_retain", 64, "256"),
    ],
)
def test_margin_row_epoch_refuses_a_nonhistorical_category_comparator(
    monkeypatch: pytest.MonkeyPatch,
    axis: str,
    preset_key: str,
    baseline: int,
    treatment: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="epoch comparator fixture"
    )
    monkeypatch.setenv(axis, treatment)
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.margin-row-epoch-parameter-engaged."
        f"{axis}.{treatment}=ran",
    )
    environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ProvenanceError,
        match="cannot earn T1 against category baseline",
    ):
        provenance.build_execution_predicate_assignment(
            environment,
            contract,
            preset_docs=[{"margin_row": {preset_key: baseline}}],
            preset_expected=True,
        )


def test_bnn_falsification_evidence_is_strict_row_local_and_causal() -> None:
    import ny_search  # noqa: PLC0415

    sibling_work = _bnn_falsification_execution_event(
        lane="ste_pgd",
        outcome="exhausted",
        work_started="true",
        budget_ms="20000",
        free_units="8",
        flips="2",
        work_units="9",
    )
    kill_events = [
        _bnn_falsification_policy_event(),
        _bnn_falsification_execution_event(),
        sibling_work,
    ]
    kill_row = {"flight": {"events": kill_events}}
    assert ny_search._bnn_falsification_parameter_engagement(
        [("NY_BNN_SIGN_SPACE", "0")], [kill_row]
    ) == {"NY_BNN_SIGN_SPACE"}

    # A disarm and sibling work must close in one row, never across instances.
    split_rows = [
        {"flight": {"events": kill_events[:2]}},
        {"flight": {"events": [kill_events[0], sibling_work]}},
    ]
    assert not ny_search._bnn_falsification_parameter_engagement(
        [("NY_BNN_SIGN_SPACE", "0")], split_rows
    )

    positive_policy = _bnn_falsification_policy_event(
        sign_space_enabled="true", lower_sign_space_enabled="false"
    )
    positive_work = _bnn_falsification_execution_event(
        sign_space_enabled="true",
        outcome="exhausted",
        work_started="true",
        budget_ms="30000",
        free_units="10",
        flips="3",
        work_units="12",
    )
    positive_row = {"flight": {"events": [positive_policy, positive_work]}}
    assert ny_search._bnn_falsification_parameter_engagement(
        [("NY_BNN_SIGN_SPACE", "1")], [positive_row]
    ) == {"NY_BNN_SIGN_SPACE"}
    max_budget_work = _bnn_falsification_execution_event(
        sign_space_enabled="true",
        outcome="exhausted",
        work_started="true",
        budget_ms="240000",
        free_units="10",
        flips="3",
        work_units="12",
    )
    assert ny_search._bnn_falsification_parameter_engagement(
        [("NY_BNN_SIGN_SPACE", "1")],
        [{"flight": {"events": [positive_policy, max_budget_work]}}],
    ) == {"NY_BNN_SIGN_SPACE"}
    positive_noop_policy = _bnn_falsification_policy_event(
        sign_space_enabled="true", lower_sign_space_enabled="true"
    )
    assert not ny_search._bnn_falsification_parameter_engagement(
        [("NY_BNN_SIGN_SPACE", "1")],
        [{"flight": {"events": [positive_noop_policy, positive_work]}}],
    )

    # Scheduled terminals are part of the current producer schema. A disabled
    # sibling is valid row evidence, but the scheduler route deliberately does
    # not prove a master kill because it can stop before consulting that lane.
    scheduled_row = {"flight": {"events": [
        _bnn_falsification_policy_event(lower_ste_pgd_enabled="false"),
        _bnn_falsification_execution_event(route="scheduled"),
        _bnn_falsification_execution_event(
            lane="ste_pgd",
            route="scheduled",
            outcome="exhausted",
            work_started="true",
            budget_ms="20000",
            free_units="8",
            flips="2",
            work_units="9",
        ),
    ]}}
    assert ny_search._bnn_falsification_evidence(scheduled_row) is not None
    assert ny_search._bnn_falsification_parameter_engagement(
        [("NY_BNN_STE_PGD", "1")], [scheduled_row]
    ) == {"NY_BNN_STE_PGD"}
    assert not ny_search._bnn_falsification_parameter_engagement(
        [("NY_BNN_SIGN_SPACE", "0")], [scheduled_row]
    )
    disarm_without_opportunity = {"flight": {"events": [
        _bnn_falsification_policy_event(),
        _bnn_falsification_execution_event(opportunity_budget_ms="none"),
        sibling_work,
    ]}}
    assert ny_search._bnn_falsification_evidence(
        disarm_without_opportunity
    ) is not None
    assert not ny_search._bnn_falsification_parameter_engagement(
        [("NY_BNN_SIGN_SPACE", "0")], [disarm_without_opportunity]
    )
    scheduled_no_budget = {"flight": {"events": [
        positive_policy,
        _bnn_falsification_execution_event(
            sign_space_enabled="true",
            route="scheduled",
            outcome="budget_unavailable",
        ),
    ]}}
    assert ny_search._bnn_falsification_evidence(scheduled_no_budget) is not None

    refinement_policy = _bnn_falsification_policy_event(
        sign_space_enabled="true", minimal_move="true", trust_region="linf"
    )
    refinement_work = _bnn_falsification_execution_event(
        sign_space_enabled="true",
        minimal_move="true",
        trust_region="linf",
        outcome="exhausted",
        work_started="true",
        budget_ms="30000",
        free_units="10",
        flips="3",
        work_units="12",
    )
    refinement_row = {
        "flight": {"events": [refinement_policy, refinement_work]}
    }
    refinements = [
        ("NY_BNN_SIGN_SPACE_MINIMAL_MOVE", "1"),
        ("NY_BNN_SIGN_SPACE_TRUST_REGION", "linf"),
    ]
    assert ny_search._bnn_falsification_parameter_engagement(
        refinements, [refinement_row]
    ) == {name for name, _token in refinements}
    refinement_noop_policy = _bnn_falsification_policy_event(
        sign_space_enabled="true",
        minimal_move="true",
        trust_region="linf",
        lower_minimal_move="true",
        lower_trust_region="linf",
    )
    assert not ny_search._bnn_falsification_parameter_engagement(
        refinements,
        [{"flight": {"events": [refinement_noop_policy, refinement_work]}}],
    )

    mip_off = {"flight": {"events": [
        _bnn_falsification_policy_event(mip_compiled="false"),
        _bnn_falsification_execution_event(),
        sibling_work,
    ]}}
    assert not ny_search._bnn_falsification_parameter_engagement(
        [("NY_BNN_SIGN_SPACE", "0")], [mip_off]
    )

    duplicate_policy = {"flight": {"events": [
        _bnn_falsification_policy_event(), *kill_events,
    ]}}
    duplicate_lane = {"flight": {"events": [
        *kill_events, _bnn_falsification_execution_event(),
    ]}}
    mismatch = _bnn_falsification_execution_event(sign_space_enabled="true")
    mismatched_row = {"flight": {"events": [
        _bnn_falsification_policy_event(), mismatch, sibling_work,
    ]}}
    malformed = _bnn_falsification_execution_event()
    malformed["reason"] += " extra=1"
    malformed_row = {"flight": {"events": [
        _bnn_falsification_policy_event(), malformed, sibling_work,
    ]}}
    raw_candidate = _bnn_falsification_execution_event(
        lane="ste_pgd",
        outcome="candidate",
        work_started="true",
        budget_ms="20000",
        free_units="8",
        flips="2",
        work_units="9",
    )
    raw_candidate_row = {"flight": {"events": [
        _bnn_falsification_policy_event(),
        _bnn_falsification_execution_event(),
        raw_candidate,
    ]}}
    zero_opportunity = {"flight": {"events": [
        _bnn_falsification_policy_event(),
        _bnn_falsification_execution_event(opportunity_budget_ms="0"),
        sibling_work,
    ]}}
    mismatched_budget = {"flight": {"events": [
        _bnn_falsification_policy_event(
            sign_space_enabled="true", lower_sign_space_enabled="false"
        ),
        _bnn_falsification_execution_event(
            sign_space_enabled="true",
            outcome="exhausted",
            work_started="true",
            budget_ms="20000",
            opportunity_budget_ms="20001",
            free_units="8",
            flips="2",
            work_units="9",
        ),
    ]}}
    refused_with_forged_work = {"flight": {"events": [
        positive_policy,
        _bnn_falsification_execution_event(
            sign_space_enabled="true",
            outcome="core_refused",
            work_started="true",
            budget_ms="20000",
            free_units="8",
            flips="2",
            work_units="9",
        ),
    ]}}
    solver_error_with_forged_work = {"flight": {"events": [
        positive_policy,
        _bnn_falsification_execution_event(
            sign_space_enabled="true",
            outcome="solver_error",
            work_started="true",
            budget_ms="20000",
            free_units="8",
            flips="2",
            work_units="9",
        ),
    ]}}
    execution_before_policy = {"flight": {"events": [
        _bnn_falsification_execution_event(),
        _bnn_falsification_policy_event(),
        sibling_work,
    ]}}
    disabled_budget_unavailable = {"flight": {"events": [
        _bnn_falsification_policy_event(),
        _bnn_falsification_execution_event(
            route="committed_cap",
            outcome="budget_unavailable",
            opportunity_budget_ms="none",
        ),
    ]}}
    scheduled_disarm_with_opportunity = {"flight": {"events": [
        _bnn_falsification_policy_event(),
        _bnn_falsification_execution_event(
            route="scheduled", opportunity_budget_ms="20000"
        ),
    ]}}
    below_floor_budget = {"flight": {"events": [
        positive_policy,
        _bnn_falsification_execution_event(
            sign_space_enabled="true",
            outcome="exhausted",
            work_started="true",
            budget_ms="19999",
            free_units="8",
            flips="2",
            work_units="9",
        ),
    ]}}
    above_cap_budget = {"flight": {"events": [
        positive_policy,
        _bnn_falsification_execution_event(
            sign_space_enabled="true",
            outcome="exhausted",
            work_started="true",
            budget_ms="240001",
            free_units="8",
            flips="2",
            work_units="9",
        ),
    ]}}
    saturated_budget = {"flight": {"events": [
        positive_policy,
        _bnn_falsification_execution_event(
            sign_space_enabled="true",
            outcome="exhausted",
            work_started="true",
            budget_ms=str((1 << 64) - 1),
            free_units="8",
            flips="2",
            work_units="9",
        ),
    ]}}
    for forged in (
        duplicate_policy,
        duplicate_lane,
        mismatched_row,
        malformed_row,
        raw_candidate_row,
        zero_opportunity,
        mismatched_budget,
        refused_with_forged_work,
        solver_error_with_forged_work,
        execution_before_policy,
        disabled_budget_unavailable,
        scheduled_disarm_with_opportunity,
        below_floor_budget,
        above_cap_budget,
        saturated_budget,
    ):
        assert ny_search._bnn_falsification_evidence(forged) is None
        assert not ny_search._bnn_falsification_parameter_engagement(
            [("NY_BNN_SIGN_SPACE", "0")], [forged]
        )


def test_bnn_falsification_provenance_binds_current_policy_receipt_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="BNN falsification fixture"
    )
    raw = (
        "ny-exec-v1.bnn-falsification-parameter-engaged."
        "NY_BNN_SIGN_SPACE.0=ran"
    )
    monkeypatch.setenv("NY_BNN_SIGN_SPACE", "0")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    traffic = {"attack": {"bnn_sign_space": True, "bnn_ste_pgd": True}}
    sealed = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[traffic],
        preset_expected=True,
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[traffic],
        preset_expected=True,
    )
    assert dict(assignment["runtime_context_assignment"]) == {
        "NY_BNN_SIGN_SPACE": "0",
        "NY_BNN_SIGN_SPACE_MINIMAL_MOVE": "0",
        "NY_BNN_SIGN_SPACE_TRUST_REGION": "full_box",
        "NY_BNN_STE_PGD": "1",
    }
    assert dict(assignment["runtime_context_source_assignment"]) == {
        "NY_BNN_SIGN_SPACE": "legacy_env",
        "NY_BNN_SIGN_SPACE_MINIMAL_MOVE": "default",
        "NY_BNN_SIGN_SPACE_TRUST_REGION": "default",
        "NY_BNN_STE_PGD": "config",
    }

    events = [
        _bnn_falsification_policy_event(),
        _bnn_falsification_execution_event(),
        _bnn_falsification_execution_event(
            lane="ste_pgd",
            outcome="exhausted",
            work_started="true",
            budget_ms="20000",
            free_units="8",
            flips="2",
            work_units="9",
        ),
    ]
    record = _bnn_falsification_record(events)
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)

    forger = json.loads(json.dumps(record))
    next(
        entry for entry in forger["levers"]["receipt"]["levers"]
        if entry["name"] == "NY_BNN_STE_PGD"
    )["source"] = "default"
    with pytest.raises(provenance.ExecutionPredicateError, match="runtime authority"):
        provenance.validate_execution_predicate_requirement(
            forger, raw, assignment=assignment
        )

    forger = json.loads(json.dumps(record))
    forger["events"].append(_bnn_falsification_policy_event())
    with pytest.raises(provenance.ExecutionPredicateError, match="singular current policy"):
        provenance.validate_execution_predicate_requirement(
            forger, raw, assignment=assignment
        )

    forger = json.loads(json.dumps(record))
    forger["events"][0] = _bnn_falsification_policy_event(
        lower_sign_space_enabled="false"
    )
    with pytest.raises(provenance.ExecutionPredicateError, match="shipped traffic lower"):
        provenance.validate_execution_predicate_requirement(
            forger, raw, assignment=assignment
        )


@pytest.mark.parametrize(
    ("axis", "token"),
    [
        ("NY_BNN_STE_PGD", "0"),
        ("NY_BNN_SIGN_SPACE_MINIMAL_MOVE", "1"),
        ("NY_BNN_SIGN_SPACE_TRUST_REGION", "linf"),
    ],
)
def test_bnn_falsification_provenance_accepts_every_remaining_axis(
    monkeypatch: pytest.MonkeyPatch,
    axis: str,
    token: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="BNN falsification fixture"
    )
    raw = (
        "ny-exec-v1.bnn-falsification-parameter-engaged."
        f"{axis}.{token}=ran"
    )
    monkeypatch.setenv(axis, token)
    if axis in {
        "NY_BNN_SIGN_SPACE_MINIMAL_MOVE",
        "NY_BNN_SIGN_SPACE_TRUST_REGION",
    }:
        monkeypatch.setenv("NY_BNN_SIGN_SPACE", "1")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    traffic = {"attack": {"bnn_sign_space": True, "bnn_ste_pgd": True}}
    sealed = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[traffic],
        preset_expected=True,
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[traffic],
        preset_expected=True,
    )

    if axis == "NY_BNN_STE_PGD":
        policy = _bnn_falsification_policy_event(
            sign_space_enabled="true", ste_pgd_enabled="false"
        )
        executions = [
            _bnn_falsification_execution_event(
                sign_space_enabled="true",
                ste_pgd_enabled="false",
                outcome="exhausted",
                work_started="true",
                budget_ms="20000",
                free_units="8",
                flips="2",
                work_units="9",
            ),
            _bnn_falsification_execution_event(
                lane="ste_pgd",
                sign_space_enabled="true",
                ste_pgd_enabled="false",
            ),
        ]
        record = _bnn_falsification_record(
            [policy, *executions],
            sign_space=(True, "config"),
            ste_pgd=(False, "legacy_env"),
        )
    else:
        minimal_move = axis == "NY_BNN_SIGN_SPACE_MINIMAL_MOVE"
        trust_region = token if axis == "NY_BNN_SIGN_SPACE_TRUST_REGION" else "full_box"
        policy = _bnn_falsification_policy_event(
            sign_space_enabled="true",
            minimal_move="true" if minimal_move else "false",
            trust_region=trust_region,
        )
        execution = _bnn_falsification_execution_event(
            sign_space_enabled="true",
            minimal_move="true" if minimal_move else "false",
            trust_region=trust_region,
            outcome="exhausted",
            work_started="true",
            budget_ms="20000",
            free_units="8",
            flips="2",
            work_units="9",
        )
        record = _bnn_falsification_record(
            [policy, execution],
            sign_space=(True, "legacy_env"),
            minimal_move=(
                True if minimal_move else False,
                "legacy_env" if minimal_move else "default",
            ),
            trust_region=(
                token if axis == "NY_BNN_SIGN_SPACE_TRUST_REGION" else None,
                "legacy_env"
                if axis == "NY_BNN_SIGN_SPACE_TRUST_REGION"
                else "default",
            ),
        )

    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)


def test_margin_row_domain_alpha_terminal_is_strict_outer_and_causal() -> None:
    import ny_search  # noqa: PLC0415

    assignment = [
        ("NY_MARGIN_ROW_DOMAIN_ALPHA", "1"),
        ("NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS", "4"),
        ("NY_MARGIN_ROW_DOMAIN_ALPHA_ETA", "0.5"),
    ]
    row = {"flight": {"events": [_margin_row_domain_alpha_event()]}}
    assert len(ny_search._margin_row_domain_alpha_terminal_evidences(row)) == 1
    assert ny_search._margin_row_domain_alpha_parameter_engagement(
        assignment, [row]
    ) == {name for name, _token in assignment}

    classwise = {"flight": {"events": [
        _margin_row_domain_alpha_event(route="classwise")
    ]}}
    nested_only = {"flight": {"events": [
        _margin_row_domain_alpha_event(
            outer_ascent_domains="0",
            outer_finite_trial_scores="0",
            outer_iters_counterfactual_domains="0",
            outer_eta_counterfactual_scored_domains="0",
        )
    ]}}
    zero_work = {
        name: "0" for name in ny_search._MARGIN_ROW_DOMAIN_ALPHA_WORK_FIELDS
    }
    configured_only = {"flight": {"events": [
        _margin_row_domain_alpha_event(**zero_work)
    ]}}
    failed_only = {"flight": {"events": [
        _margin_row_domain_alpha_event(
            gate_evals="1",
            movable_domains="1",
            movable_slots="1",
            warm_entries_applied="0",
            ascent_domains="1",
            outer_ascent_domains="1",
            proposals="1",
            trial_passes_completed="0",
            trial_pass_failures="1",
            finite_trial_scores="0",
            accepted_trials="0",
            carried_entries="0",
            iters_counterfactual_domains="0",
            eta_counterfactual_scored_domains="0",
            outer_finite_trial_scores="0",
            outer_iters_counterfactual_domains="0",
            outer_eta_counterfactual_scored_domains="0",
        )
    ]}}
    for diagnostic in (classwise, nested_only, configured_only, failed_only):
        assert len(
            ny_search._margin_row_domain_alpha_terminal_evidences(diagnostic)
        ) == 1
        assert not ny_search._margin_row_domain_alpha_parameter_engagement(
            assignment, [diagnostic]
        )

    malformed_events = [
        _margin_row_domain_alpha_event(policy_mismatch="true"),
        _margin_row_domain_alpha_event(configured_enabled="false"),
        _margin_row_domain_alpha_event(proposals="5"),
        _margin_row_domain_alpha_event(
            gate_evals="1",
            movable_domains="1",
            movable_slots="1",
            warm_entries_applied="0",
            ascent_domains="1",
            outer_ascent_domains="1",
            proposals="4",
            trial_passes_completed="2",
            trial_pass_failures="2",
            finite_trial_scores="1",
            accepted_trials="0",
            carried_entries="0",
            iters_counterfactual_domains="0",
            eta_counterfactual_scored_domains="0",
            outer_finite_trial_scores="1",
            outer_iters_counterfactual_domains="0",
            outer_eta_counterfactual_scored_domains="0",
        ),
        _margin_row_domain_alpha_event(
            proposals="2",
            trial_passes_completed="2",
            trial_pass_failures="0",
        ),
        _margin_row_domain_alpha_event(outer_finite_trial_scores="1"),
        _margin_row_domain_alpha_event(ascent_domains="3"),
        _margin_row_domain_alpha_event(outer_finite_trial_scores="3"),
        _margin_row_domain_alpha_event(
            outer_ascent_domains="0",
            outer_iters_counterfactual_domains="0",
            outer_eta_counterfactual_scored_domains="0",
        ),
        _margin_row_domain_alpha_event(configured_iters="04"),
        _margin_row_domain_alpha_event(
            configured_iters="2", iters_counterfactual_domains="1"
        ),
        _margin_row_domain_alpha_event(
            configured_eta_bits=_f64_bits(0.25),
            eta_counterfactual_scored_domains="1",
        ),
        _margin_row_domain_alpha_event(
            configured_eta_bits=_f64_bits(float("nan"))
        ),
    ]
    missing = _margin_row_domain_alpha_event()
    missing["reason"] = missing["reason"].replace(" carried_entries=2", "")
    extra = _margin_row_domain_alpha_event()
    extra["reason"] += " extra=1"
    duplicate_field = _margin_row_domain_alpha_event()
    duplicate_field["reason"] += " proposals=4"
    tabbed = _margin_row_domain_alpha_event()
    tabbed["reason"] = tabbed["reason"].replace(" phase=", "\tphase=")
    malformed_events.extend((missing, extra, duplicate_field, tabbed))
    for event in malformed_events:
        assert not ny_search._margin_row_domain_alpha_terminal_evidences(
            {"flight": {"events": [event]}}
        )
    assert not ny_search._margin_row_domain_alpha_terminal_evidences({
        "flight": {"events": [
            _margin_row_domain_alpha_event(),
            _margin_row_domain_alpha_event(),
        ]}
    })


def test_margin_row_domain_alpha_assignment_seals_policy_receipt_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="domain-alpha fixture"
    )
    raw = (
        "ny-exec-v1.margin-row-domain-alpha-parameter-engaged."
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS.4=ran"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_DOMAIN_ALPHA", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS", "4")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    assert assignment["requested_assignment"] == [
        ["NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS", "4"]
    ]
    assert assignment["comparison_assignment"] == [
        ["NY_MARGIN_ROW_DOMAIN_ALPHA", "1"]
    ]
    assert dict(assignment["runtime_context_assignment"]) == {
        "NY_MARGIN_ROW_DOMAIN_ALPHA": "1",
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA": "0.25",
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS": "4",
    }
    assert dict(assignment["runtime_context_source_assignment"]) == {
        "NY_MARGIN_ROW_DOMAIN_ALPHA": "legacy_env",
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA": "default",
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS": "legacy_env",
    }

    event = _margin_row_domain_alpha_event(
        configured_eta_bits=_f64_bits(0.25),
        eta_counterfactual_scored_domains="0",
        outer_eta_counterfactual_scored_domains="0",
    )
    record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_DOMAIN_ALPHA", "1", True),
            ("NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS", "4", 4),
        ],
        event,
    )
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)

    eta_entry = next(
        entry
        for entry in record["levers"]["receipt"]["levers"]
        if entry["name"] == "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA"
    )
    eta_entry["value"] = 0.5
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MARGIN_ROW_DOMAIN_ALPHA_ETA terminal configuration differs",
    ):
        provenance.validate_execution_predicate_requirement(
            record, raw, assignment=assignment
        )
    eta_entry["value"] = 0.25
    eta_entry["source"] = "config"
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="runtime authority 'config' differs from sealed authority 'default'",
    ):
        provenance.validate_execution_predicate_requirement(
            record, raw, assignment=assignment
        )


def test_margin_row_domain_alpha_joint_predicate_binds_exact_eta_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="domain-alpha joint fixture"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_DOMAIN_ALPHA", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS", "4")
    monkeypatch.setenv("NY_MARGIN_ROW_DOMAIN_ALPHA_ETA", "0.5")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        MARGIN_ROW_DOMAIN_ALPHA_ALL_JOINT_PREDICATE,
    )
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_DOMAIN_ALPHA", "1", True),
            ("NY_MARGIN_ROW_DOMAIN_ALPHA_ETA", "0.5", 0.5),
            ("NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS", "4", 4),
        ],
        _margin_row_domain_alpha_event(),
    )
    assert provenance.validate_execution_predicate_requirement(
        record,
        MARGIN_ROW_DOMAIN_ALPHA_ALL_JOINT_PREDICATE,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(
        MARGIN_ROW_DOMAIN_ALPHA_ALL_JOINT_PREDICATE
    )

    forged = _margin_row_domain_alpha_event(
        configured_eta_bits=_f64_bits(0.25),
        eta_counterfactual_scored_domains="0",
        outer_eta_counterfactual_scored_domains="0",
    )
    record["events"] = [forged]
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MARGIN_ROW_DOMAIN_ALPHA_ETA terminal configuration differs",
    ):
        provenance.validate_execution_predicate_requirement(
            record,
            MARGIN_ROW_DOMAIN_ALPHA_ALL_JOINT_PREDICATE,
            assignment=assignment,
        )


def test_margin_row_domain_alpha_uses_and_splits_sealed_preset_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="domain-alpha preset fixture",
    )
    raw = (
        "ny-exec-v1.margin-row-domain-alpha-parameter-engaged."
        "NY_MARGIN_ROW_DOMAIN_ALPHA.1=ran"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_DOMAIN_ALPHA", "1")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    preset = {"margin_row": {"domain_alpha_eta": 0.5}}
    sealed = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[preset],
        preset_expected=True,
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[preset],
        preset_expected=True,
    )
    assert dict(assignment["runtime_context_assignment"]) == {
        "NY_MARGIN_ROW_DOMAIN_ALPHA": "1",
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA": "0.5",
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS": "2",
    }
    assert dict(assignment["runtime_context_source_assignment"]) == {
        "NY_MARGIN_ROW_DOMAIN_ALPHA": "legacy_env",
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA": "config",
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS": "default",
    }
    record = _execution_predicate_record(
        [("NY_MARGIN_ROW_DOMAIN_ALPHA", "1", True)],
        _margin_row_domain_alpha_event(
            configured_iters="2",
            iters_counterfactual_domains="0",
            outer_iters_counterfactual_domains="0",
        ),
    )
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)

    with pytest.raises(provenance.ProvenanceError, match="split the measurement"):
        provenance.build_execution_predicate_assignment(
            environment,
            contract,
            preset_docs=[preset, {}],
            preset_expected=True,
        )


@pytest.mark.parametrize(
    ("axis", "preset_key", "baseline", "treatment"),
    [
        ("NY_MARGIN_ROW_DOMAIN_ALPHA", "domain_alpha", True, "1"),
        ("NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS", "domain_alpha_iters", 4, "1"),
        ("NY_MARGIN_ROW_DOMAIN_ALPHA_ETA", "domain_alpha_eta", 0.5, "0.125"),
    ],
)
def test_margin_row_domain_alpha_refuses_a_nonhistorical_category_comparator(
    monkeypatch: pytest.MonkeyPatch,
    axis: str,
    preset_key: str,
    baseline: object,
    treatment: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="domain-alpha comparator fixture",
    )
    monkeypatch.setenv("NY_MARGIN_ROW_DOMAIN_ALPHA", "1")
    monkeypatch.setenv(axis, treatment)
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.margin-row-domain-alpha-parameter-engaged."
        f"{axis}.{treatment}=ran",
    )
    environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ProvenanceError,
        match="cannot earn T1 against category baseline",
    ):
        provenance.build_execution_predicate_assignment(
            environment,
            contract,
            preset_docs=[{"margin_row": {preset_key: baseline}}],
            preset_expected=True,
        )


def test_margin_row_backward_interm_scalar_assignment_seals_master_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="backward-intermediate scalar fixture",
    )
    raw = (
        "ny-exec-v1.margin-row-backward-interm-parameter-engaged."
        "NY_MARGIN_ROW_BI_CHUNK.64=ran"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_BACKWARD_INTERM", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_BI_CHUNK", "64")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)

    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )

    assert assignment["requested_assignment"] == [["NY_MARGIN_ROW_BI_CHUNK", "64"]]
    assert assignment["comparison_assignment"] == [
        ["NY_MARGIN_ROW_BACKWARD_INTERM", "1"]
    ]
    assert assignment["full_assignment"] == [
        ["NY_MARGIN_ROW_BACKWARD_INTERM", "1"],
        ["NY_MARGIN_ROW_BI_CHUNK", "64"],
    ]
    record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_BACKWARD_INTERM", "1", True),
            ("NY_MARGIN_ROW_BI_CHUNK", "64", 64),
        ],
        _margin_row_backward_interm_event(
            configured_secs="20",
            configured_chunk_secs="0",
            effective_chunk_secs="0",
            configured_topk="1024",
            chunk_ceiling_binding_chunks="0",
            fixed_width_chunks="2",
            adaptive_width_changes="0",
            topk_binding_layers="0",
        ),
    )
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)


def test_margin_row_root_alpha_scalar_seals_full_policy_and_exact_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="root-alpha scalar fixture",
    )
    raw = (
        "ny-exec-v1.margin-row-root-alpha-parameter-engaged."
        "NY_MARGIN_ROW_ALPHA_ITERS.16=ran"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_ALPHA_OPT", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_ALPHA_ITERS", "16")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)

    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    assert assignment["requested_assignment"] == [
        ["NY_MARGIN_ROW_ALPHA_ITERS", "16"]
    ]
    assert assignment["comparison_assignment"] == [
        ["NY_MARGIN_ROW_ALPHA_OPT", "1"]
    ]
    assert dict(assignment["runtime_context_assignment"]) == {
        "NY_MARGIN_ROW_ALPHA_OPT": "1",
        "NY_MARGIN_ROW_ALPHA_ITERS": "16",
        "NY_MARGIN_ROW_ALPHA_SECS": "20.0",
        "NY_MARGIN_ROW_ALPHA_LR": "0.1",
        "NY_MARGIN_ROW_ALPHA_COLS": "8",
        "NY_MARGIN_ROW_ALPHA_GUARD": "1",
        "NY_MARGIN_ROW_REBAKE": "0",
        "NY_MARGIN_ROW_REBAKE_ROUNDS": "2",
        "NY_MARGIN_ROW_REBAKE_SECS": "30.0",
    }
    assert dict(assignment["runtime_context_source_assignment"]) == {
        "NY_MARGIN_ROW_ALPHA_OPT": "legacy_env",
        "NY_MARGIN_ROW_ALPHA_ITERS": "legacy_env",
        "NY_MARGIN_ROW_ALPHA_SECS": "default",
        "NY_MARGIN_ROW_ALPHA_LR": "default",
        "NY_MARGIN_ROW_ALPHA_COLS": "default",
        "NY_MARGIN_ROW_ALPHA_GUARD": "default",
        "NY_MARGIN_ROW_REBAKE": "default",
        "NY_MARGIN_ROW_REBAKE_ROUNDS": "default",
        "NY_MARGIN_ROW_REBAKE_SECS": "default",
    }

    record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_ALPHA_OPT", "1", True),
            ("NY_MARGIN_ROW_ALPHA_ITERS", "16", 16),
        ],
        _margin_row_root_alpha_event(),
    )
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)

    columns = next(
        entry
        for entry in record["levers"]["receipt"]["levers"]
        if entry["name"] == "NY_MARGIN_ROW_ALPHA_COLS"
    )
    columns["value"] = 4
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MARGIN_ROW_ALPHA_COLS terminal configuration differs",
    ):
        provenance.validate_execution_predicate_requirement(
            record, raw, assignment=assignment
        )


def test_margin_row_root_alpha_uses_preset_context_and_splits_mixed_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ny_search  # noqa: PLC0415

    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="root-alpha preset fixture",
    )
    cifar = ny_search.load_preset_doc(
        REPO_ROOT / "configs" / "vnncomp25" / "cifar100_2024.yaml"
    )
    tiny = ny_search.load_preset_doc(
        REPO_ROOT / "configs" / "vnncomp25" / "tinyimagenet_2024.yaml"
    )
    assert isinstance(cifar, dict)
    assert isinstance(tiny, dict)
    raw = (
        "ny-exec-v1.margin-row-root-alpha-parameter-engaged."
        "NY_MARGIN_ROW_ALPHA_OPT.0=ran"
    )

    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_ALPHA_OPT", "0")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(
        environment,
        contract,
        preset_docs=[cifar],
        preset_expected=True,
    )
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
        preset_docs=[cifar],
        preset_expected=True,
    )
    assert dict(assignment["runtime_context_assignment"])[
        "NY_MARGIN_ROW_ALPHA_OPT"
    ] == "0"
    assert dict(assignment["runtime_context_source_assignment"])[
        "NY_MARGIN_ROW_ALPHA_OPT"
    ] == "legacy_env"
    record = _execution_predicate_record(
        [("NY_MARGIN_ROW_ALPHA_OPT", "0", False)],
        _margin_row_root_alpha_disabled_event(),
    )
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)

    with pytest.raises(provenance.ProvenanceError, match="split the measurement"):
        provenance.build_execution_predicate_assignment(
            environment,
            contract,
            preset_docs=[cifar, tiny],
            preset_expected=True,
        )


def test_portable_preset_parser_matches_pyyaml_on_shipped_vnncomp_configs() -> None:
    # PyYAML is declared in requirements.txt (>=6.0), so its absence is a
    # PROVISIONING failure, not a reason to disappear. TEST_CONFORMANCE.md:
    # selecting a lane "makes a missing interpreter, package, extension, input,
    # or output a hard failure".
    try:
        import yaml  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - provisioning failure
        raise AssertionError(
            "PyYAML is a declared requirement of this lane (requirements.txt); "
            "install it into the interpreter running this suite "
            "(see the README virtualenv setup)"
        ) from error
    import ny_search  # noqa: PLC0415

    presets = sorted((REPO_ROOT / "configs").glob("vnncomp*/*.yaml"))
    assert presets
    for preset in presets:
        expected = yaml.safe_load(preset.read_text(encoding="utf-8"))
        assert ny_search.load_preset_doc_portable(preset) == expected, preset


def test_scorecard_isolated_python_loads_exact_preset_baseline_without_pyyaml() -> None:
    python = Path("/usr/bin/python3")
    # A hard requirement, not a skip: this test exists to prove the scorecard's
    # ISOLATED interpreter can load the preset baseline without PyYAML. Skipping
    # when that interpreter is missing silently deletes the only coverage of the
    # isolation contract, on exactly the hosts where it is least certain.
    assert python.is_file(), (
        "the scorecard's isolated interpreter /usr/bin/python3 is required by this "
        "lane and is missing; this test asserts the isolation contract and cannot "
        "be satisfied by another interpreter"
    )
    code = """
from pathlib import Path
import sys
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'scripts'))
import ny_measurement_provenance as provenance
import ny_search
docs = provenance._load_execution_preset_docs(
    root / 'configs' / 'vnncomp25', ['cifar100_2024']
)
assert ny_search.preset_effective_token(
    docs[0], 'preset:margin_row.backward_interm'
) == '1'
assert ny_search.preset_effective_token(
    docs[0], 'preset:margin_row.bi_secs'
) == '40'
print('portable-preset-context-ok')
"""
    result = subprocess.run(
        [str(python), "-E", "-s", "-S", "-B", "-c", code, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "portable-preset-context-ok"


def test_execution_preset_loader_resolves_suffix_and_rejects_ambiguous_yaml(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vnncomp25"
    root.mkdir()
    preset = root / "demo.yaml"
    preset.write_text(
        "margin_row:\n  backward_interm: true\n  bi_secs: 40\n",
        encoding="utf-8",
    )
    assert provenance._load_execution_preset_docs(root, ["demo_2024"]) == [
        {
            "margin_row": {
                "backward_interm": True,
                "bi_secs": 40,
            }
        }
    ]

    preset.write_text(
        "margin_row: &shared\n  backward_interm: true\n",
        encoding="utf-8",
    )
    with pytest.raises(provenance.ProvenanceError, match="parsed exactly"):
        provenance._load_execution_preset_docs(root, ["demo"])
    with pytest.raises(provenance.ProvenanceError, match="category is unsafe"):
        provenance._load_execution_preset_docs(root, ["../demo"])
    with pytest.raises(provenance.ProvenanceError, match="no preset"):
        provenance._load_execution_preset_docs(root, ["missing"])


def test_margin_row_backward_interm_uses_cifar_preset_baselines_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ny_search  # noqa: PLC0415

    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="backward-intermediate CIFAR preset fixture",
    )
    cifar = ny_search.load_preset_doc(
        REPO_ROOT / "configs" / "vnncomp25" / "cifar100_2024.yaml"
    )
    tiny = ny_search.load_preset_doc(
        REPO_ROOT / "configs" / "vnncomp25" / "tinyimagenet_2024.yaml"
    )
    assert isinstance(cifar, dict)
    assert isinstance(tiny, dict)

    def assignment(raw: str, values: dict[str, str], docs: list[dict[str, object]]):
        _clear_ny_environment(monkeypatch)
        for name, token in values.items():
            monkeypatch.setenv(name, token)
        monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
        environment = provenance._capture_environment(contract)
        sealed = provenance.build_execution_predicate_assignment(
            environment,
            contract,
            preset_docs=docs,
            preset_expected=True,
        )
        validated = provenance.validate_execution_predicate_assignment(
            sealed,
            environment=environment,
            search_space_contract=contract,
            preset_docs=docs,
            preset_expected=True,
        )
        return sealed, validated

    master_zero_raw = (
        "ny-exec-v1.margin-row-backward-interm-parameter-engaged."
        "NY_MARGIN_ROW_BACKWARD_INTERM.0=ran"
    )
    _sealed, master_zero = assignment(
        master_zero_raw,
        {"NY_MARGIN_ROW_BACKWARD_INTERM": "0"},
        [cifar],
    )
    assert master_zero["requested_assignment"] == [
        ["NY_MARGIN_ROW_BACKWARD_INTERM", "0"]
    ]
    disabled = _margin_row_backward_interm_event(
        outcome="disabled",
        enabled="false",
        configured_secs="40",
        configured_chunk="256",
        configured_chunk_secs="0",
        effective_chunk_secs="0",
        configured_topk="1024",
        suppressed_setup_layers="4",
        layers_started="0",
        chunks_completed="0",
        columns_completed="0",
        budget_exhausted="false",
        chunk_counterfactual_chunks="0",
        chunk_ceiling_binding_chunks="0",
        fixed_width_chunks="0",
        adaptive_width_changes="0",
        topk_binding_layers="0",
    )
    disabled_record = _execution_predicate_record(
        [("NY_MARGIN_ROW_BACKWARD_INTERM", "0", False)], disabled
    )
    assert provenance.validate_execution_predicate_requirement(
        disabled_record, master_zero_raw, assignment=master_zero
    ) == provenance.parse_execution_predicate_requirement(master_zero_raw)

    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_BACKWARD_INTERM", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        master_zero_raw.replace(".0=ran", ".1=ran"),
    )
    baseline_environment = provenance._capture_environment(contract)
    with pytest.raises(provenance.ProvenanceError, match="no-op run"):
        provenance.build_execution_predicate_assignment(
            baseline_environment,
            contract,
            preset_docs=[cifar],
            preset_expected=True,
        )

    secs_raw = (
        "ny-exec-v1.margin-row-backward-interm-parameter-engaged."
        "NY_MARGIN_ROW_BI_SECS.20=ran"
    )
    _secs_sealed, secs_assignment = assignment(
        secs_raw,
        {
            "NY_MARGIN_ROW_BACKWARD_INTERM": "1",
            "NY_MARGIN_ROW_BI_SECS": "20",
        },
        [cifar],
    )
    assert secs_assignment["comparison_assignment"] == [
        ["NY_MARGIN_ROW_BACKWARD_INTERM", "1"]
    ]
    secs_event = _margin_row_backward_interm_event(
        configured_secs="20",
        configured_chunk="256",
        configured_chunk_secs="0",
        effective_chunk_secs="0",
        configured_topk="1024",
        chunk_counterfactual_chunks="0",
        chunk_ceiling_binding_chunks="2",
        fixed_width_chunks="2",
        adaptive_width_changes="0",
        topk_binding_layers="0",
    )
    secs_record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_BACKWARD_INTERM", "1", True),
            ("NY_MARGIN_ROW_BI_SECS", "20", 20.0),
        ],
        secs_event,
    )
    assert provenance.validate_execution_predicate_requirement(
        secs_record, secs_raw, assignment=secs_assignment
    ) == provenance.parse_execution_predicate_requirement(secs_raw)

    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_BACKWARD_INTERM", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_BI_SECS", "20")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", secs_raw)
    mixed_environment = provenance._capture_environment(contract)
    with pytest.raises(provenance.ProvenanceError, match="split the measurement"):
        provenance.build_execution_predicate_assignment(
            mixed_environment,
            contract,
            preset_docs=[cifar, tiny],
            preset_expected=True,
        )

    chunk_raw = (
        "ny-exec-v1.margin-row-backward-interm-parameter-engaged."
        "NY_MARGIN_ROW_BI_CHUNK.64=ran"
    )
    _chunk_sealed, chunk_assignment = assignment(
        chunk_raw,
        {
            "NY_MARGIN_ROW_BACKWARD_INTERM": "1",
            "NY_MARGIN_ROW_BI_CHUNK": "64",
        },
        [cifar],
    )
    chunk_environment = provenance._capture_environment(contract)
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="differs from the regenerated contrast",
    ):
        provenance.validate_execution_predicate_assignment(
            _chunk_sealed,
            environment=chunk_environment,
            search_space_contract=contract,
        )
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="differs from the regenerated contrast",
    ):
        provenance.validate_execution_predicate_assignment(
            _chunk_sealed,
            environment=chunk_environment,
            search_space_contract=contract,
            preset_docs=[tiny],
            preset_expected=True,
        )
    chunk_event = _margin_row_backward_interm_event(
        configured_secs="40",
        configured_chunk="64",
        configured_chunk_secs="0",
        effective_chunk_secs="0",
        configured_topk="1024",
        chunk_ceiling_binding_chunks="0",
        fixed_width_chunks="2",
        adaptive_width_changes="0",
        topk_binding_layers="0",
    )
    chunk_record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_BACKWARD_INTERM", "1", True),
            ("NY_MARGIN_ROW_BI_CHUNK", "64", 64),
        ],
        chunk_event,
    )
    assert provenance.validate_execution_predicate_requirement(
        chunk_record, chunk_raw, assignment=chunk_assignment
    ) == provenance.parse_execution_predicate_requirement(chunk_raw)
    assert dict(chunk_assignment["runtime_context_assignment"]) == {
        "NY_MARGIN_ROW_BACKWARD_INTERM": "1",
        "NY_MARGIN_ROW_BI_CHUNK": "64",
        "NY_MARGIN_ROW_BI_CHUNK_SECS": "0.0",
        "NY_MARGIN_ROW_BI_SECS": "40",
        "NY_MARGIN_ROW_BI_TOPK": "1024",
    }
    assert dict(chunk_assignment["runtime_context_source_assignment"]) == {
        "NY_MARGIN_ROW_BACKWARD_INTERM": "legacy_env",
        "NY_MARGIN_ROW_BI_CHUNK": "legacy_env",
        "NY_MARGIN_ROW_BI_CHUNK_SECS": "default",
        "NY_MARGIN_ROW_BI_SECS": "config",
        "NY_MARGIN_ROW_BI_TOPK": "default",
    }
    secs_entry = next(
        entry
        for entry in chunk_record["levers"]["receipt"]["levers"]
        if entry["name"] == "NY_MARGIN_ROW_BI_SECS"
    )
    secs_entry["source"] = "default"
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="runtime authority 'default' differs from sealed authority 'config'",
    ):
        provenance.validate_execution_predicate_requirement(
            chunk_record, chunk_raw, assignment=chunk_assignment
        )
    secs_entry["source"] = "config"

    topk_raw = (
        "ny-exec-v1.margin-row-backward-interm-parameter-engaged."
        "NY_MARGIN_ROW_BI_TOPK.4096=ran"
    )
    _topk_sealed, topk_assignment = assignment(
        topk_raw,
        {
            "NY_MARGIN_ROW_BACKWARD_INTERM": "1",
            "NY_MARGIN_ROW_BI_TOPK": "4096",
        },
        [cifar],
    )
    topk_event = _margin_row_backward_interm_event(
        configured_secs="40",
        configured_chunk="256",
        configured_chunk_secs="0",
        effective_chunk_secs="0",
        configured_topk="4096",
        chunk_counterfactual_chunks="0",
        chunk_ceiling_binding_chunks="2",
        fixed_width_chunks="2",
        adaptive_width_changes="0",
    )
    topk_record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_BACKWARD_INTERM", "1", True),
            ("NY_MARGIN_ROW_BI_TOPK", "4096", 4096),
        ],
        topk_event,
    )
    assert provenance.validate_execution_predicate_requirement(
        topk_record, topk_raw, assignment=topk_assignment
    ) == provenance.parse_execution_predicate_requirement(topk_raw)


def test_margin_row_backward_interm_joint_assignment_requires_every_axis_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="backward-intermediate joint fixture",
    )
    tokens = {
        "NY_MARGIN_ROW_BACKWARD_INTERM": "1",
        "NY_MARGIN_ROW_BI_CHUNK": "64",
        "NY_MARGIN_ROW_BI_CHUNK_SECS": "2",
        "NY_MARGIN_ROW_BI_SECS": "30",
        "NY_MARGIN_ROW_BI_TOPK": "4096",
    }
    for name, token in tokens.items():
        monkeypatch.setenv(name, token)
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        MARGIN_ROW_BACKWARD_INTERM_ALL_JOINT_PREDICATE,
    )

    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )
    requested = [[name, token] for name, token in sorted(tokens.items())]
    assert assignment["requested_assignment"] == requested
    assert assignment["comparison_assignment"] == []
    assert assignment["full_assignment"] == requested

    record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_BACKWARD_INTERM", "1", True),
            ("NY_MARGIN_ROW_BI_CHUNK", "64", 64),
            ("NY_MARGIN_ROW_BI_CHUNK_SECS", "2", 2.0),
            ("NY_MARGIN_ROW_BI_SECS", "30", 30.0),
            ("NY_MARGIN_ROW_BI_TOPK", "4096", 4096),
        ],
        _margin_row_backward_interm_event(),
    )
    assert provenance.validate_execution_predicate_requirement(
        record,
        MARGIN_ROW_BACKWARD_INTERM_ALL_JOINT_PREDICATE,
        assignment=assignment,
    ) == provenance.parse_execution_predicate_requirement(
        MARGIN_ROW_BACKWARD_INTERM_ALL_JOINT_PREDICATE
    )

    record["events"] = [
        _margin_row_backward_interm_event(topk_binding_layers="0")
    ]
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="NY_MARGIN_ROW_BI_TOPK=4096",
    ):
        provenance.validate_execution_predicate_requirement(
            record,
            MARGIN_ROW_BACKWARD_INTERM_ALL_JOINT_PREDICATE,
            assignment=assignment,
        )


def test_margin_row_backward_interm_rejects_missing_or_mismatched_execution() -> None:
    raw = (
        "ny-exec-v1.margin-row-backward-interm-parameter-engaged."
        "NY_MARGIN_ROW_BI_SECS.30=ran"
    )
    record = _execution_predicate_record(
        [("NY_MARGIN_ROW_BI_SECS", "30", 30.0)],
        _margin_row_backward_interm_event(),
    )
    assert provenance.validate_execution_predicate_requirement(record, raw) == (
        provenance.parse_execution_predicate_requirement(raw)
    )

    record["events"] = []
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="no consistent current terminal receipt",
    ):
        provenance.validate_execution_predicate_requirement(record, raw)

    record["events"] = [
        _margin_row_backward_interm_event(configured_secs="20")
    ]
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="terminal configuration differs from its exact receipt",
    ):
        provenance.validate_execution_predicate_requirement(record, raw)

    record["events"] = [
        _margin_row_backward_interm_event(configured_secs="30"),
        _margin_row_backward_interm_event(configured_secs="20"),
    ]
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="no consistent current terminal receipt",
    ):
        provenance.validate_execution_predicate_requirement(record, raw)


def test_margin_row_backward_interm_authenticates_preset_context_and_rejects_env_hitchhiker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(),
        source="backward-intermediate receipt-closure fixture",
    )
    raw = (
        "ny-exec-v1.margin-row-backward-interm-parameter-engaged."
        "NY_MARGIN_ROW_BI_CHUNK.64=ran"
    )
    monkeypatch.setenv("NY_MARGIN_ROW_BACKWARD_INTERM", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_BI_CHUNK", "64")
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)
    environment = provenance._capture_environment(contract)
    sealed = provenance.build_execution_predicate_assignment(environment, contract)
    assignment = provenance.validate_execution_predicate_assignment(
        sealed,
        environment=environment,
        search_space_contract=contract,
    )

    record = _execution_predicate_record(
        [
            ("NY_MARGIN_ROW_BACKWARD_INTERM", "1", True),
            ("NY_MARGIN_ROW_BI_CHUNK", "64", 64),
        ],
        _margin_row_backward_interm_event(
            configured_secs="20",
            configured_chunk_secs="0",
            effective_chunk_secs="0",
            configured_topk="1024",
            chunk_ceiling_binding_chunks="0",
            fixed_width_chunks="2",
            adaptive_width_changes="0",
            topk_binding_layers="0",
        ),
    )
    assert provenance.validate_execution_predicate_requirement(
        record, raw, assignment=assignment
    ) == provenance.parse_execution_predicate_requirement(raw)

    extra = next(
        entry
        for entry in record["levers"]["receipt"]["levers"]
        if entry["name"] == "NY_MARGIN_ROW_BI_TOPK"
    )
    extra["source"] = "legacy_env"
    extra["env_utf8"] = True
    record["ambient_env"]["NY_MARGIN_ROW_BI_TOPK"] = "4096"
    receipt = record["levers"]["receipt"]
    receipt["env_present"] = 3
    receipt["env_accepted"] = 3

    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="runtime authority 'legacy_env' differs from sealed authority 'default'",
    ):
        provenance.validate_execution_predicate_requirement(
            record, raw, assignment=assignment
        )

    extra["value"] = 1024
    extra["source"] = "bogus_authority"
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="unknown resolution source 'bogus_authority'",
    ):
        provenance.validate_execution_predicate_requirement(
            record, raw, assignment=assignment
        )

    extra["source"] = "config"
    extra.pop("env_utf8")
    extra["value"] = 4096
    record["ambient_env"].pop("NY_MARGIN_ROW_BI_TOPK")
    receipt["env_present"] = 2
    receipt["env_accepted"] = 2
    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="terminal configuration differs from its exact receipt",
    ):
        provenance.validate_execution_predicate_requirement(
            record, raw, assignment=assignment
        )


@pytest.mark.parametrize(
    "raw",
    (
        # A joint spelling must contain at least two coordinates.
        "ny-exec-v2.root-joint-interm-alpha-parameter-engaged."
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS.12=ran",
        # Coordinates are an authenticated canonical list, not a set repaired later.
        "ny-exec-v2.root-joint-interm-alpha-parameter-engaged."
        "NY_ROOT_JOINT_INTERM_ALPHA_LR.0.05,"
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS.12=ran",
        "ny-exec-v2.root-joint-interm-alpha-parameter-engaged."
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS.12,"
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS.50=ran",
        # The transport names one family; foreign coordinates cannot hitchhike.
        "ny-exec-v2.root-joint-interm-alpha-parameter-engaged."
        "NY_PGD_FAST_GRAD.0,NY_ROOT_JOINT_INTERM_ALPHA_ITERS.12=ran",
    ),
)
def test_joint_execution_predicate_transport_rejects_noncanonical_or_cross_family(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_MEASURE_REQUIRE_FLIGHT_EVENT",
    ):
        provenance._capture_environment()


def test_same_family_hitchhiker_is_not_relabelled_as_a_prerequisite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _root_predicate_search_space_payload(), source="root recipe fixture"
    )
    for name, token in ROOT_LR_PREREQUISITES.items():
        monkeypatch.setenv(name, token)
    monkeypatch.setenv("NY_ROOT_JOINT_INTERM_ALPHA_LR", ROOT_LR_REQUEST)
    monkeypatch.setenv("NY_ROOT_JOINT_INTERM_ALPHA_SECS", "20")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.root-joint-interm-alpha-parameter-engaged."
        f"NY_ROOT_JOINT_INTERM_ALPHA_LR.{ROOT_LR_REQUEST}=ran",
    )
    environment = provenance._capture_environment(contract)

    with pytest.raises(
        provenance.ProvenanceError,
        match="direct governed inputs do not equal",
    ):
        provenance.build_execution_predicate_assignment(environment, contract)


def test_assignment_validator_detects_an_omitted_unrelated_registry_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _root_predicate_search_space_payload(), source="root recipe fixture"
    )
    for name, token in ROOT_LR_PREREQUISITES.items():
        monkeypatch.setenv(name, token)
    monkeypatch.setenv("NY_ROOT_JOINT_INTERM_ALPHA_LR", ROOT_LR_REQUEST)
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.root-joint-interm-alpha-parameter-engaged."
        f"NY_ROOT_JOINT_INTERM_ALPHA_LR.{ROOT_LR_REQUEST}=ran",
    )
    environment = provenance._capture_environment(contract)
    assignment = provenance.build_execution_predicate_assignment(
        environment, contract
    )
    environment["values"]["NY_ROOT_JOINT_INTERM_ALPHA_SECS"] = "20"
    environment["typed_values"]["NY_ROOT_JOINT_INTERM_ALPHA_SECS"] = {
        "type": "nonnegative_integer",
        "value": 20,
    }

    with pytest.raises(
        provenance.ExecutionPredicateError,
        match="direct governed inputs do not equal",
    ):
        provenance.validate_execution_predicate_assignment(
            assignment,
            environment=environment,
            search_space_contract=contract,
        )


def test_default_equal_scalar_is_rejected_before_scorecard_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _root_predicate_search_space_payload(), source="root recipe fixture"
    )
    monkeypatch.setenv("NY_ROOT_JOINT_INTERM_ALPHA_ITERS", "100")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.root-joint-interm-alpha-parameter-engaged."
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS.100=ran",
    )
    environment = provenance._capture_environment(contract)

    with pytest.raises(provenance.ProvenanceError, match="a no-op run"):
        provenance.build_execution_predicate_assignment(environment, contract)


def test_composite_root_master_is_not_a_one_axis_scorecard_treatment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    contract = provenance._search_space_contract_from_payload(
        _root_predicate_search_space_payload(), source="root recipe fixture"
    )
    monkeypatch.setenv("NY_ROOT_JOINT_INTERM_ALPHA", "1")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.root-joint-interm-alpha-parameter-engaged."
        "NY_ROOT_JOINT_INTERM_ALPHA.1=ran",
    )
    environment = provenance._capture_environment(contract)

    with pytest.raises(
        provenance.ProvenanceError,
        match="composite admission master",
    ):
        provenance.build_execution_predicate_assignment(environment, contract)


def test_search_space_contract_rejects_unknown_nested_schema_fields() -> None:
    payload = json.loads(json.dumps(_FIXTURE_SEARCH_SPACE_PAYLOAD))
    payload["declarations"]["NY_PGD_FAST_GRAD"]["unreviewed"] = True

    with pytest.raises(
        provenance.ProvenanceError,
        match="declaration has an invalid field set",
    ):
        provenance._search_space_contract_from_payload(
            payload, source="malformed fixture"
        )


@pytest.mark.parametrize(
    ("predicate_id", "axis"),
    sorted(
        (predicate_id, axis)
        for predicate_id, spec in provenance.EXECUTION_PREDICATE_SPECS.items()
        for axis in spec["axes"]
    ),
)
def test_every_current_predicate_axis_can_be_sealed_from_the_direct_environment(
    monkeypatch: pytest.MonkeyPatch,
    predicate_id: str,
    axis: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    token = CURRENT_PREDICATE_AXIS_TOKENS[axis]
    requirement = f"{predicate_id}.{axis}.{token}=ran"
    monkeypatch.setenv(axis, token)
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", requirement)

    environment = provenance._capture_environment()

    assert environment["values"][axis] == token
    assert axis in environment["typed_values"]
    assert environment["typed_values"]["NY_MEASURE_REQUIRE_FLIGHT_EVENT"] == (
        provenance.parse_execution_predicate_requirement(requirement)
    )


def test_direct_predicate_axis_reaches_the_sealed_solver_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_PGD_FAST_GRAD", "0")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT", CURRENT_EXECUTION_PREDICATE
    )
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="direct-predicate-axis").read_text()
    )

    assert manifest["environment"]["typed_values"]["NY_PGD_FAST_GRAD"] == {
        "type": "boolean",
        "value": False,
    }
    assert manifest["measurement"]["solver_environment"]["values"][
        "NY_PGD_FAST_GRAD"
    ] == "0"


def test_generated_root_recipe_reaches_the_actual_sealed_solver_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    binary = repo / "target/release/ny"
    binary.write_text(
        _fixture_solver_script(
            search_space_payload=_root_predicate_search_space_payload()
        ),
        encoding="utf-8",
    )
    binary.chmod(0o755)
    binary.touch()
    _clear_ny_environment(monkeypatch)
    for name, token in ROOT_LR_PREREQUISITES.items():
        monkeypatch.setenv(name, token)
    monkeypatch.setenv("NY_ROOT_JOINT_INTERM_ALPHA_LR", ROOT_LR_REQUEST)
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.root-joint-interm-alpha-parameter-engaged."
        f"NY_ROOT_JOINT_INTERM_ALPHA_LR.{ROOT_LR_REQUEST}=ran",
    )
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="root-recipe-delivery").read_text()
    )

    expected = {
        **ROOT_LR_PREREQUISITES,
        "NY_ROOT_JOINT_INTERM_ALPHA_LR": ROOT_LR_REQUEST,
    }
    solver_values = manifest["measurement"]["solver_environment"]["values"]
    assert {name: solver_values.get(name) for name in expected} == expected
    assert manifest["measurement"]["execution_predicate_assignment"][
        "full_assignment"
    ] == [list(pair) for pair in sorted(expected.items())]
    carried = provenance.validate_start_search_space_contract(manifest)
    assert carried == manifest["measurement"]["search_space_contract"]


def test_generated_joint_recipe_uses_v3_profile_and_seals_full_solver_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    binary = repo / "target/release/ny"
    binary.write_text(
        _fixture_solver_script(
            search_space_payload=_root_predicate_search_space_payload()
        ),
        encoding="utf-8",
    )
    binary.chmod(0o755)
    binary.touch()
    _clear_ny_environment(monkeypatch)
    for name, token in ROOT_LR_PREREQUISITES.items():
        monkeypatch.setenv(name, token)
    for name, token in ROOT_JOINT_REQUESTED:
        monkeypatch.setenv(name, token)
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", ROOT_JOINT_PREDICATE)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="root-joint-recipe-delivery").read_text()
    )

    expected = {
        **ROOT_LR_PREREQUISITES,
        **dict(ROOT_JOINT_REQUESTED),
    }
    assert manifest["measurement"]["execution_evidence_profile"] == (
        provenance.JOINT_EXECUTION_EVIDENCE_PROFILE
    )
    assert manifest["measurement"]["execution_predicate_assignment"]["schema"] == (
        provenance.JOINT_EXECUTION_PREDICATE_ASSIGNMENT_SCHEMA
    )
    solver_values = manifest["measurement"]["solver_environment"]["values"]
    assert {name: solver_values.get(name) for name in expected} == expected
    assert "NY_MEASURE_REQUIRE_FLIGHT_EVENT" not in solver_values


def test_predicate_axes_absent_from_selected_registry_cannot_hitchhike(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    binary = repo / "target/release/ny"
    binary.write_text(
        _fixture_solver_script(
            search_space_payload=_FIXTURE_SEARCH_SPACE_PAYLOAD
        ),
        encoding="utf-8",
    )
    binary.chmod(0o755)
    binary.touch()
    _clear_ny_environment(monkeypatch)
    for axis, token in CURRENT_PREDICATE_AXIS_TOKENS.items():
        monkeypatch.setenv(axis, token)
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT", CURRENT_EXECUTION_PREDICATE
    )
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    with pytest.raises(
        provenance.ProvenanceError,
        match="explicit but absent from the selected solver's governed registry",
    ):
        _capture(repo, benchmark_root, run_id="unpublished-predicate-axes")


def test_deprecated_rayon_worker_fallback_fails_before_solver_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    assert "RAYON_RS_NUM_CPUS" not in provenance.ENV_ALLOWLIST
    assert "RAYON_" in provenance.UNREVIEWED_SOLVER_RUNTIME_ENV_PREFIXES
    monkeypatch.setenv("RAYON_RS_NUM_CPUS", "4")

    with pytest.raises(
        provenance.ProvenanceError,
        match="RAYON_RS_NUM_CPUS",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("axis", "raw"),
    (
        ("NY_PGD_FAST_GRAD", "true"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_ITERS", "01"),
        ("NY_PGD_VJP_K", "0"),
        ("NY_PGD_VJP_STEPS", "01"),
        ("NY_PGD_ODI_ETA", "0"),
        ("NY_PGD_ODI_ETA", "1e-100"),
        ("NY_PGD_HINGE_TAU", "-0.1"),
        ("NY_ROOT_JOINT_INTERM_ALPHA_LR", "1e39"),
        ("NY_MARGIN_ROW_PARALLEL", "1"),
    ),
)
def test_current_predicate_axis_measurement_grammar_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    axis: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(axis, raw)

    with pytest.raises(provenance.ProvenanceError, match=axis):
        provenance._capture_environment()


@pytest.mark.parametrize("raw", ("", "01", "true", " 1", "1 "))
def test_cgan_input_leaf_rejects_nonexact_measurement_spellings(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CGAN_INPUT_LEAF", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_CGAN_INPUT_LEAF",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "raw",
    (
        "",
        "missing-separator",
        "cgan_input_leaf_summary=ran",
        "ny-exec-v1.unknown.NY_PGD_FAST_GRAD.1=ran",
        "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
        "NY_MARGIN_ROW_PARALLEL.1=ran",
        "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
        "NY_PGD_FAST_GRAD.1=skipped",
        "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
        "NY_PGD_FAST_GRAD.1=ran=ran",
    ),
)
def test_required_flight_event_rejects_malformed_contracts(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MEASURE_REQUIRE_FLIGHT_EVENT", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_MEASURE_REQUIRE_FLIGHT_EVENT",
    ):
        provenance._capture_environment()


def test_required_execution_predicate_must_match_direct_requested_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MARGIN_ROW_PARALLEL", "2")
    monkeypatch.setenv(
        "NY_MEASURE_REQUIRE_FLIGHT_EVENT",
        "ny-exec-v1.margin-row-frontier-parameter-engaged."
        "NY_MARGIN_ROW_PARALLEL.4=ran",
    )

    with pytest.raises(
        provenance.ProvenanceError,
        match="direct measurement environment",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("predicate_id", "axis", "requested"),
    (
        ("ny-exec-v1.upfront-parameter-engaged", "NY_UPFRONT_ATTACK_CAP", "8"),
        (
            "ny-exec-v1.margin-row-frontier-parameter-engaged",
            "NY_MARGIN_ROW_PARALLEL",
            "4",
        ),
        (
            "ny-exec-v1.margin-row-policy-parameter-engaged",
            "NY_MARGIN_ROW_BETA_COLS",
            "4",
        ),
        (
            "ny-exec-v1.margin-row-root-precision-parameter-engaged",
            "NY_MARGIN_ROW_ROOT_F32",
            "0",
        ),
        (
            "ny-exec-v1.margin-row-backward-interm-parameter-engaged",
            "NY_MARGIN_ROW_BI_CHUNK",
            "64",
        ),
        (
            "ny-exec-v1.margin-row-domain-alpha-parameter-engaged",
            "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS",
            "4",
        ),
        (
            "ny-exec-v1.margin-row-epoch-parameter-engaged",
            "NY_EPOCH_DEPTH",
            "3",
        ),
        (
            "ny-exec-v1.margin-row-width-parameter-engaged",
            "NY_MARGIN_ROW_K_HEAD",
            "4",
        ),
        (
            "ny-exec-v1.pgd-fast-gradient-parameter-engaged",
            "NY_PGD_FAST_GRAD",
            "1",
        ),
        (
            "ny-exec-v1.postbab-reserve-parameter-engaged",
            "NY_POSTBAB_RESERVE_SECS",
            "100",
        ),
        (
            "ny-exec-v1.mo-kfsb-parameter-engaged",
            "NY_MO_KFSB",
            "1",
        ),
        (
            "ny-exec-v1.mo-kfsb-parameter-engaged",
            "NY_MO_KFSB_REDUCE",
            "min",
        ),
        (
            "ny-exec-v1.mo-alpha-step-parameter-engaged",
            "NY_MO_ALPHA_STEP",
            "1",
        ),
        (
            "ny-exec-v1.root-joint-interm-alpha-parameter-engaged",
            "NY_ROOT_JOINT_INTERM_ALPHA_LR",
            "0.1",
        ),
        ("ny-exec-v1.exact-vjp-parameter-engaged", "NY_PGD_VJP_K", "52"),
    ),
)
def test_current_execution_predicate_families_have_closed_typed_ids(
    predicate_id: str, axis: str, requested: str
) -> None:
    parsed = provenance.parse_execution_predicate_requirement(
        f"{predicate_id}.{axis}.{requested}=ran"
    )

    assert parsed == {
        "schema": "ny-measure/execution-predicate-requirement/v1",
        "type": "execution_predicate_requirement",
        "predicate_id": predicate_id,
        "axis": axis,
        "requested_value": requested,
        "claim": "parameter_engaged",
    }


def test_execution_predicate_registry_covers_every_axis_specific_t1_contract(
) -> None:
    import ny_search  # noqa: PLC0415

    predicate_axes = {
        axis
        for spec in provenance.EXECUTION_PREDICATE_SPECS.values()
        for axis in spec["axes"]
    }
    assert predicate_axes == set(ny_search._AXIS_SPECIFIC_T1_CONTRACTS)
    assert all(
        callable(getattr(ny_search, spec["validator"], None))
        for spec in provenance.EXECUTION_PREDICATE_SPECS.values()
    )


def test_mo_alpha_predicate_seals_its_beta_spsa_prerequisite() -> None:
    import ny_search  # noqa: PLC0415

    spec = provenance.EXECUTION_PREDICATE_SPECS[
        "ny-exec-v1.mo-alpha-step-parameter-engaged"
    ]
    assert spec == {
        "validator": "_mo_alpha_step_parameter_engagement",
        "axes": frozenset({"NY_MO_ALPHA_STEP"}),
    }
    assert ny_search._AXIS_SPECIFIC_T1_CONTRACTS["NY_MO_ALPHA_STEP"] == (
        "ny-cli/mo-alpha-step-execution/v1"
    )
    assert "NY_MO_BETA_SPSA" not in ny_search._AXIS_SPECIFIC_T1_CONTRACTS


def test_backward_interm_predicate_registry_matches_its_closed_t1_family() -> None:
    import ny_search  # noqa: PLC0415

    axes = {
        "NY_MARGIN_ROW_BACKWARD_INTERM",
        "NY_MARGIN_ROW_BI_SECS",
        "NY_MARGIN_ROW_BI_CHUNK",
        "NY_MARGIN_ROW_BI_CHUNK_SECS",
        "NY_MARGIN_ROW_BI_TOPK",
    }
    spec = provenance.EXECUTION_PREDICATE_SPECS[
        "ny-exec-v1.margin-row-backward-interm-parameter-engaged"
    ]
    assert spec == {
        "validator": "_margin_row_backward_interm_parameter_engagement",
        "axes": frozenset(axes),
    }
    assert {
        axis: ny_search._AXIS_SPECIFIC_T1_CONTRACTS[axis] for axis in axes
    } == {
        axis: "ny-cli/margin-row-backward-interm-execution/v1"
        for axis in axes
    }


def test_root_precision_predicate_registry_matches_its_closed_t1_family() -> None:
    import ny_search  # noqa: PLC0415

    axes = {"NY_MARGIN_ROW_ROOT_F32", "NY_MARGIN_ROW_TABLEAU_F32"}
    spec = provenance.EXECUTION_PREDICATE_SPECS[
        "ny-exec-v1.margin-row-root-precision-parameter-engaged"
    ]
    assert spec == {
        "validator": "_margin_row_root_precision_parameter_engagement",
        "axes": frozenset(axes),
    }
    assert {
        axis: ny_search._AXIS_SPECIFIC_T1_CONTRACTS[axis] for axis in axes
    } == {
        axis: "ny-cli/margin-row-root-precision-execution/v1"
        for axis in axes
    }


def test_root_alpha_predicate_registry_matches_its_closed_t1_family() -> None:
    import ny_search  # noqa: PLC0415

    axes = {
        "NY_MARGIN_ROW_ALPHA_OPT",
        "NY_MARGIN_ROW_ALPHA_ITERS",
        "NY_MARGIN_ROW_ALPHA_SECS",
        "NY_MARGIN_ROW_ALPHA_LR",
        "NY_MARGIN_ROW_ALPHA_COLS",
        "NY_MARGIN_ROW_ALPHA_GUARD",
        "NY_MARGIN_ROW_REBAKE",
        "NY_MARGIN_ROW_REBAKE_ROUNDS",
        "NY_MARGIN_ROW_REBAKE_SECS",
    }
    spec = provenance.EXECUTION_PREDICATE_SPECS[
        "ny-exec-v1.margin-row-root-alpha-parameter-engaged"
    ]
    assert spec == {
        "validator": "_margin_row_root_alpha_parameter_engagement",
        "axes": frozenset(axes),
        "configured_axes": frozenset(axes),
    }
    assert {
        axis: ny_search._AXIS_SPECIFIC_T1_CONTRACTS[axis] for axis in axes
    } == {
        axis: "ny-cli/margin-row-root-alpha-execution/v1"
        for axis in axes
    }


def test_epoch_predicate_registry_seals_but_does_not_graduate_parent() -> None:
    import ny_search  # noqa: PLC0415

    axes = {
        "NY_EPOCH_DEPTH",
        "NY_EPOCH_ATTEMPTS",
        "NY_EPOCH_TIER0",
        "NY_EPOCH_UNIVERSE",
        "NY_EPOCH_RETAIN",
    }
    parent = "NY_MARGIN_ROW_EPOCH_PARENT_TOPK"
    spec = provenance.EXECUTION_PREDICATE_SPECS[
        "ny-exec-v1.margin-row-epoch-parameter-engaged"
    ]
    assert spec == {
        "validator": "_margin_row_epoch_parameter_engagement",
        "axes": frozenset(axes),
        "configured_axes": frozenset(axes | {parent}),
    }
    assert {
        axis: ny_search._AXIS_SPECIFIC_T1_CONTRACTS[axis] for axis in axes
    } == {
        axis: "ny-cli/margin-row-epoch-execution/v1" for axis in axes
    }
    assert parent not in ny_search._AXIS_SPECIFIC_T1_CONTRACTS
    assert parent not in provenance.EXECUTION_PREDICATE_AXES
    assert parent in provenance.ENV_ALLOWLIST
    assert parent in provenance.TYPED_CANONICAL_NONNEGATIVE_INTEGER_ENV


def test_width_predicate_registry_matches_its_closed_t1_family() -> None:
    import ny_search  # noqa: PLC0415

    axes = {
        "NY_MARGIN_ROW_K_HEAD",
        "NY_MARGIN_ROW_K_TRUNK",
        "NY_MARGIN_ROW_K_ADAPT",
    }
    spec = provenance.EXECUTION_PREDICATE_SPECS[
        "ny-exec-v1.margin-row-width-parameter-engaged"
    ]
    assert spec == {
        "validator": "_margin_row_width_parameter_engagement",
        "axes": frozenset(axes),
        "configured_axes": frozenset(axes),
    }
    assert {
        axis: ny_search._AXIS_SPECIFIC_T1_CONTRACTS[axis] for axis in axes
    } == {
        axis: "ny-margin-row-width-execution/v2" for axis in axes
    }
    assert axes <= provenance.ENV_ALLOWLIST
    assert {
        "NY_MARGIN_ROW_K_HEAD", "NY_MARGIN_ROW_K_TRUNK"
    } <= provenance.TYPED_CANONICAL_POSITIVE_INTEGER_ENV
    assert "NY_MARGIN_ROW_K_ADAPT" in provenance.TYPED_EXACT_BOOLEAN_ENV


def test_schedule_predicate_seals_unsafe_policy_but_graduates_only_concurrent(
) -> None:
    import ny_search  # noqa: PLC0415

    concurrent = "NY_MARGIN_ROW_CONCURRENT"
    unsafe_schedule_axes = {
        "NY_MARGIN_ROW_RESERVE_SECS",
        "NY_MARGIN_ROW_ADAPTIVE_RESERVE",
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC",
        "NY_MARGIN_ROW_ADAPTIVE_RELEASE_FRAC",
        "NY_MARGIN_ROW_EARLY_EXIT",
        "NY_VERIFIER_THREADS",
        "NY_MARGIN_ROW_CONCURRENT_THREADS",
    }
    spec = provenance.EXECUTION_PREDICATE_SPECS[
        "ny-exec-v1.margin-row-schedule-parameter-engaged"
    ]
    assert spec == {
        "validator": "_margin_row_schedule_parameter_engagement",
        "axes": frozenset({concurrent}),
        "configured_axes": frozenset(unsafe_schedule_axes | {concurrent}),
    }
    assert ny_search._AXIS_SPECIFIC_T1_CONTRACTS[concurrent] == (
        "ny-cli/margin-row-schedule-execution/v2+"
        "ny-cli/margin-row-concurrent-overlap-execution/v1+"
        "ny-margin-row-width-execution/v2"
    )
    assert unsafe_schedule_axes.isdisjoint(ny_search._AXIS_SPECIFIC_T1_CONTRACTS)
    assert unsafe_schedule_axes.isdisjoint(provenance.EXECUTION_PREDICATE_AXES)
    assert unsafe_schedule_axes | {concurrent} <= provenance.ENV_ALLOWLIST
    assert ny_search._MARGIN_ROW_SCHEDULE_FAMILY == frozenset({concurrent})
    assert ny_search._MARGIN_ROW_SCHEDULE_CONFIGURED_FAMILY == frozenset(
        unsafe_schedule_axes | {concurrent}
    )

    parsed = provenance.parse_execution_predicate_requirement(
        MARGIN_ROW_SCHEDULE_PREDICATE
    )
    assert parsed["predicate_id"] == (
        "ny-exec-v1.margin-row-schedule-parameter-engaged"
    )
    assert parsed["axis"] == concurrent
    assert parsed["requested_value"] == "1"

    space = ny_search.parse_space_rs()
    kinds = ny_search.parse_lever_kinds()
    kept, dropped = ny_search.admissible_values(
        space.by_name[concurrent],
        kinds,
        False,
        preset_doc={"margin_row": {"concurrent": True}},
        preset_known=True,
        preset_expected=True,
    )
    assert kept == []
    assert any(
        f"{concurrent}=0 has no row-local causal witness" in note
        for note in dropped
    )
    axis = space.by_name[concurrent]
    assert ny_search.runtime_applicability_finding(
        axis,
        ["custom_resnet"],
        preset_docs=[{"margin_row": {"reserve_secs": 0}}],
        preset_expected=True,
    ) is None
    assert "reserve-free" in ny_search.runtime_applicability_finding(
        axis,
        ["fixed_reserve"],
        preset_docs=[{"margin_row": {"reserve_secs": 45}}],
        preset_expected=True,
    )
    assert "cannot authenticate typed" in ny_search.runtime_applicability_finding(
        axis,
        ["forged"],
        preset_docs=[{"margin_row": {"reserve_secs": "0"}}],
        preset_expected=True,
    )
    assert "no-op" in ny_search.runtime_applicability_finding(
        axis,
        ["control", "already_on"],
        preset_docs=[
            {"margin_row": {"reserve_secs": 0, "concurrent": False}},
            {"margin_row": {"reserve_secs": 0, "concurrent": True}},
        ],
        preset_expected=True,
    )


def test_domain_alpha_predicate_registry_matches_its_closed_t1_family() -> None:
    import ny_search  # noqa: PLC0415

    axes = {
        "NY_MARGIN_ROW_DOMAIN_ALPHA",
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS",
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA",
    }
    spec = provenance.EXECUTION_PREDICATE_SPECS[
        "ny-exec-v1.margin-row-domain-alpha-parameter-engaged"
    ]
    assert spec == {
        "validator": "_margin_row_domain_alpha_parameter_engagement",
        "axes": frozenset(axes),
        "configured_axes": frozenset(axes),
    }
    assert {
        axis: ny_search._AXIS_SPECIFIC_T1_CONTRACTS[axis] for axis in axes
    } == {
        axis: "ny-cli/margin-row-domain-alpha-execution/v1"
        for axis in axes
    }
    assert "NY_MARGIN_ROW_DOMAIN_ALPHA" in provenance.TYPED_EXACT_BOOLEAN_ENV
    assert (
        "NY_MARGIN_ROW_DOMAIN_ALPHA_ITERS"
        in provenance.TYPED_CANONICAL_NONNEGATIVE_INTEGER_ENV
    )
    assert any(
        name == "NY_MARGIN_ROW_DOMAIN_ALPHA_ETA"
        for name, _minimum, _maximum in provenance.TYPED_BOUNDED_FLOAT_ENV
    )


def test_bnn_falsification_predicate_registry_matches_closed_policy_pair() -> None:
    import ny_search  # noqa: PLC0415

    axes = {
        "NY_BNN_SIGN_SPACE",
        "NY_BNN_SIGN_SPACE_MINIMAL_MOVE",
        "NY_BNN_SIGN_SPACE_TRUST_REGION",
        "NY_BNN_STE_PGD",
    }
    spec = provenance.EXECUTION_PREDICATE_SPECS[
        "ny-exec-v1.bnn-falsification-parameter-engaged"
    ]
    assert spec == {
        "validator": "_bnn_falsification_parameter_engagement",
        "axes": frozenset(axes),
        "configured_axes": frozenset(axes),
    }
    assert {
        axis: ny_search._AXIS_SPECIFIC_T1_CONTRACTS[axis] for axis in axes
    } == {
        axis: (
            "ny-cli/bnn-falsification-policy/v1+"
            "ny-cli/bnn-falsification-execution/v1"
        )
        for axis in axes
    }
    assert axes <= provenance.ENV_ALLOWLIST
    assert {
        "NY_BNN_SIGN_SPACE",
        "NY_BNN_SIGN_SPACE_MINIMAL_MOVE",
        "NY_BNN_STE_PGD",
    } <= provenance.TYPED_EXACT_BOOLEAN_ENV
    assert (
        "NY_BNN_SIGN_SPACE_TRUST_REGION",
        frozenset({"box", "tight", "linf"}),
    ) in provenance.TYPED_ENUM_ENV


def test_every_predicate_domain_token_is_encodable_in_joint_v2_transport() -> None:
    contract = provenance._search_space_contract_from_payload(
        _current_source_search_space_payload(), source="joint transport ratchet"
    )
    axes = contract["axes"]
    for axis in sorted(provenance.EXECUTION_PREDICATE_AXES):
        for token in axes[axis]["values"]:
            assert provenance.SAFE_COMPONENT.fullmatch(token), (axis, token)
            assert "," not in token and "=" not in token, (axis, token)


def test_scorecard_shell_execution_predicate_transport_accepts_only_ran() -> None:
    source = (REPO_ROOT / "scripts/measure_ny_scorecard.sh").read_text(
        encoding="utf-8"
    )
    start = source.index('case "$predicate_status" in')
    end = source.index("    esac", start)
    status_case = source[start:end]

    assert "\n      ran) ;;" in status_case
    assert all(
        status not in status_case
        for status in ("skipped", "not_reached", "complete")
    )
    requirement_case = source[
        source.index('case "$REQUIRED_FLIGHT_EVENT" in') : start
    ]
    assert "current measurement evidence requires" in requirement_case
    assert "'') ;;" not in requirement_case
    assert "*[!A-Za-z0-9_.,-]*" in requirement_case
    assert "ny-exec-v2.exact-vjp-parameter-engaged" in source


@pytest.mark.parametrize(
    "mode", ["affine", "reachability", "residual", "selector", "shared"]
)
def test_imb_ay_region_proof_mode_is_captured_as_closed_enum(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_IMB_AY_REGION_PROOF", mode)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_IMB_AY_REGION_PROOF"] == mode
    assert environment["typed_values"]["NY_IMB_AY_REGION_PROOF"] == {
        "type": "enum",
        "value": mode,
        "allowed_values": ["affine", "reachability", "residual", "selector", "shared"],
    }


@pytest.mark.parametrize(
    "raw",
    ["", "Affine", "scalar", "default", " affine", "affine ", "unknown"],
)
def test_imb_ay_region_proof_mode_rejects_unreviewed_spellings(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_IMB_AY_REGION_PROOF", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_IMB_AY_REGION_PROOF"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NY_IMB_OBJ", "0", 0),
        ("NY_IMB_OBJ", "001", 1),
        ("NY_IMB_REPLAY_ONLY_LEAF", "4", 4),
        ("NY_IMB_REPLAY_ONLY_LEAF", "0004", 4),
    ],
)
def test_imb_replay_selectors_are_captured_as_decimal_usize(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "nonnegative_integer",
        "value": expected,
    }


def test_boxlift_environment_is_captured_raw_and_unknown_neighbors_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    values = {
        "NY_BRANCH_RESCORE": "0",
        "NY_CLIP_INTERM_RESNET": "0",
        "NY_F64_LINEAGE_RECOVER": "1",
        "NY_INTERM_REFINE_SELECTIVE_TOPK": "020",
        "NY_PHASE_TELEMETRY": "1",
        "NY_ROOT_JOINT_INTERM_ALPHA": "1",
        "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT": "1",
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS": "12",
        "NY_ROOT_JOINT_INTERM_ALPHA_LR": "0.10",
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM": "02048",
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL": "8",
        "NY_ROOT_JOINT_INTERM_ALPHA_PROBE": "1",
        "NY_ROOT_JOINT_INTERM_ALPHA_SECS": "20",
        "NY_ROOT_JOINT_MIN_REMAINING_SECS": "0",
        "NY_ROOT_SPARSE_INTERM_CROWN": "1",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM": "8192",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS": "512",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS": "4",
        "NY_ROOT_SPARSE_INTERM_CROWN_SECS": "2",
        "NY_ROOT_SPEC_PRUNE": "1",
    }
    default_environment = provenance._capture_environment()
    for key in values:
        assert key not in default_environment["values"]

    for key, value in values.items():
        monkeypatch.setenv(key, value)
    environment = provenance._capture_environment()
    sparse_cap_bounds = {
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM": (0, 8_192),
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS": (0, 512),
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS": (0, 4),
        "NY_ROOT_SPARSE_INTERM_CROWN_SECS": (0, 8),
    }
    for key, value in values.items():
        assert environment["values"][key] == value
        if key in {
            "NY_PHASE_TELEMETRY",
            "NY_ROOT_JOINT_INTERM_ALPHA",
            "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
            "NY_ROOT_JOINT_INTERM_ALPHA_PROBE",
            "NY_ROOT_SPARSE_INTERM_CROWN",
        }:
            assert environment["typed_values"][key] == {
                "type": "boolean",
                "value": True,
            }
        elif key in sparse_cap_bounds:
            minimum, maximum = sparse_cap_bounds[key]
            assert environment["typed_values"][key] == {
                "type": "bounded_canonical_nonnegative_integer",
                "value": int(value),
                "minimum": minimum,
                "maximum": maximum,
            }
        elif key in {
            "NY_ROOT_JOINT_INTERM_ALPHA_ITERS",
            "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL",
            "NY_ROOT_JOINT_INTERM_ALPHA_SECS",
            "NY_ROOT_JOINT_MIN_REMAINING_SECS",
        }:
            assert environment["typed_values"][key] == {
                "type": "nonnegative_integer",
                "value": int(value),
            }
        elif key == "NY_ROOT_JOINT_INTERM_ALPHA_LR":
            assert environment["typed_values"][key] == {
                "type": "bounded_float",
                "value": float(value),
                "minimum": -3.402_823_466_385_288_6e38,
                "maximum": 3.402_823_466_385_288_6e38,
            }
        else:
            assert key not in environment["typed_values"]

    unknown_key = "NY_ROOT_JOINT_INTERM_ALPHA_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")
    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_exact_batched_route_switch_is_absent_by_default_and_typed_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    default_environment = provenance._capture_environment()
    assert "NY_PGD_EXACT_BATCHED" not in default_environment["values"]

    monkeypatch.setenv("NY_PGD_EXACT_BATCHED", "0")
    typed_environment = provenance._capture_environment()
    assert typed_environment["values"]["NY_PGD_EXACT_BATCHED"] == "0"
    assert typed_environment["typed_values"]["NY_PGD_EXACT_BATCHED"] == {
        "type": "boolean",
        "value": False,
    }

    monkeypatch.setenv("NY_PGD_EXACT_BATCHED", "00")
    with pytest.raises(
        provenance.ProvenanceError, match="NY_PGD_EXACT_BATCHED"
    ):
        provenance._capture_environment()


def test_spec_alpha_direct_is_captured_raw_and_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()
    assert "NY_SPEC_ALPHA_DIRECT" not in environment["values"]

    # Runtime activation is exact-string (`"1"` only). Preserve both active
    # and inactive spellings as raw launch evidence; do not normalize them.
    for raw in ("1", "0", "true"):
        monkeypatch.setenv("NY_SPEC_ALPHA_DIRECT", raw)
        environment = provenance._capture_environment()
        assert environment["values"]["NY_SPEC_ALPHA_DIRECT"] == raw
        assert "NY_SPEC_ALPHA_DIRECT" not in environment["typed_values"]


def test_rump_f64_engine_zero_is_captured_in_immutable_start_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="rump-f64-default").read_text()
    )
    assert "NY_RUMP_F64_ENGINE" not in default_manifest["environment"]["values"]

    # The runtime disables this engine only for exact "0". Preserve that launch
    # spelling as raw evidence instead of normalizing it into another behavior.
    monkeypatch.setenv("NY_RUMP_F64_ENGINE", "0")
    start_path = _capture(repo, benchmark_root, run_id="rump-f64-disabled")
    start_bytes = start_path.read_bytes()
    environment = json.loads(start_bytes)["environment"]
    assert environment["values"]["NY_RUMP_F64_ENGINE"] == "0"
    assert "NY_RUMP_F64_ENGINE" not in environment["typed_values"]

    monkeypatch.setenv("NY_RUMP_F64_ENGINE", "1")
    with pytest.raises(FileExistsError, match="immutable evidence"):
        _capture(repo, benchmark_root, run_id="rump-f64-disabled")
    assert start_path.read_bytes() == start_bytes


def test_postbab_ab_environment_is_absent_by_default_and_preserved_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    default_values = provenance._capture_environment()["values"]
    assert "NY_POSTBAB_BAB_SEEDS" not in default_values
    assert "NY_POSTBAB_BAB_SEEDS_K" not in default_values
    assert "NY_POSTBAB_FRONTIER_FASTLANE" not in default_values
    assert "NY_POSTBAB_FRONTIER_FASTLANE_SECS" not in default_values
    assert "NY_POSTBAB_RESERVE_SECS" not in default_values
    assert "NY_ACASXU_PROF" not in default_values

    # The runtime enables export only for exact live modes "1"/"2" (retired
    # mode "3" is off), trims/parses/clamps K, and enables profiling on any
    # present value. Preserve those research-control launch spellings. The now
    # governed reserve uses canonical typed measurement syntax so a spelling
    # that merely falls through Rust's trim/parser cannot masquerade as a
    # distinct optimizer treatment.
    monkeypatch.setenv("NY_POSTBAB_BAB_SEEDS", "1")
    monkeypatch.setenv("NY_POSTBAB_BAB_SEEDS_K", " 0256 ")
    monkeypatch.setenv("NY_POSTBAB_FRONTIER_FASTLANE", "1")
    monkeypatch.setenv("NY_POSTBAB_FRONTIER_FASTLANE_SECS", " 0010 ")
    monkeypatch.setenv("NY_POSTBAB_RESERVE_SECS", "100")
    monkeypatch.setenv("NY_ACASXU_PROF", "0")
    raw_environment = provenance._capture_environment()
    assert raw_environment["values"]["NY_POSTBAB_BAB_SEEDS"] == "1"
    assert raw_environment["values"]["NY_POSTBAB_BAB_SEEDS_K"] == " 0256 "
    assert raw_environment["values"]["NY_POSTBAB_FRONTIER_FASTLANE"] == "1"
    assert raw_environment["values"]["NY_POSTBAB_FRONTIER_FASTLANE_SECS"] == " 0010 "
    assert raw_environment["values"]["NY_POSTBAB_RESERVE_SECS"] == "100"
    assert raw_environment["values"]["NY_ACASXU_PROF"] == "0"
    assert "NY_POSTBAB_BAB_SEEDS" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_BAB_SEEDS_K" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_FRONTIER_FASTLANE" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_FRONTIER_FASTLANE_SECS" not in raw_environment["typed_values"]
    assert raw_environment["typed_values"]["NY_POSTBAB_RESERVE_SECS"] == {
        "type": "nonnegative_integer",
        "value": 100,
    }
    assert "NY_ACASXU_PROF" not in raw_environment["typed_values"]

    for raw in ("0100", " 0100 ", "+100"):
        monkeypatch.setenv("NY_POSTBAB_RESERVE_SECS", raw)
        with pytest.raises(
            provenance.ProvenanceError,
            match=(
                "NY_POSTBAB_RESERVE_SECS must be a canonical "
                "nonnegative integer"
            ),
        ):
            provenance._capture_environment()


def test_screen_schedule_environment_is_absent_by_default_and_preserved_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    keys = {
        "NY_SCREEN_WAVE_SIZE": " 0256 ",
        "NY_SCREEN_MVF_WAVE_SIZE": "0008",
        "NY_SCREEN_CELL_CHUNK": "0",
        "NY_SCREEN_MVF_CHUNK": "not-a-number",
        "NY_SCREEN_CROWN_MS": "5.5",
    }
    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="screen-schedule-default").read_text()
    )
    for key in keys:
        assert key not in default_manifest["environment"]["values"]

    for key, value in keys.items():
        monkeypatch.setenv(key, value)
    raw_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="screen-schedule-raw").read_text()
    )
    raw_environment = raw_manifest["environment"]
    for key, value in keys.items():
        assert raw_environment["values"][key] == value
        assert key not in raw_environment["typed_values"]


def test_start_binds_ay_executable_and_completion_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    ay_binary = tmp_path / "trusted-ay"
    ay_binary.write_text(
        f"#!/bin/sh\nprintf '%s\\n' 'ay fixture' 'build.commit={AY_REV}'\n",
        encoding="utf-8",
    )
    ay_binary.chmod(0o755)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY", str(ay_binary))
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    start = _capture(repo, benchmark_root, run_id="ay-drift")
    start_record = json.loads(start.read_text(encoding="utf-8"))
    executable = start_record["dependencies"]["ay"]["executable"]
    assert executable["declared_path"] == str(ay_binary)
    assert executable["resolved_path"] == str(ay_binary.resolve())
    assert executable["sha256"] == provenance._sha256_file(ay_binary)
    assert executable["build_commit"] == AY_REV
    assert f"build.commit={AY_REV}" in executable["version_stdout"]

    ay_binary.write_text(
        f"#!/bin/sh\nprintf '%s\\n' 'changed ay fixture' 'build.commit={AY_REV}'\n",
        encoding="utf-8",
    )
    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}
    assert record["integrity"]["status"] == "invalid"
    assert record["completed_successfully"] is False
    assert "ay_executable_identity_mismatch" in codes


@pytest.mark.parametrize(
    "version_output",
    [
        "ay fixture",
        "ay fixture\nbuild.commit=0123456789abcdef0123456789abcdef01234567",
        f"ay fixture\nbuild.commit={AY_REV}-dirty",
        f"ay fixture\nbuild.commit={AY_REV}\nbuild.commit={AY_REV}",
    ],
)
def test_start_rejects_ay_executable_not_built_from_exact_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_output: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    ay_binary = tmp_path / "unbound-ay"
    shell_lines = " ".join(repr(line) for line in version_output.splitlines())
    ay_binary.write_text(
        f"#!/bin/sh\nprintf '%s\\n' {shell_lines}\n",
        encoding="utf-8",
    )
    ay_binary.chmod(0o755)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY", str(ay_binary))
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    with pytest.raises(provenance.ProvenanceError, match=r"AY executable build\.commit"):
        _capture(repo, benchmark_root, run_id="ay-build-mismatch")


@pytest.mark.parametrize(
    ("changed_identity", "violation_code"),
    [
        ("solver", "solver_binary_sha256_mismatch"),
        ("ny_worktree", "ny_worktree_identity_mismatch"),
        ("benchmark", "benchmark_identity_mismatch"),
        ("config_inputs", "config_inputs_identity_mismatch"),
    ],
)
def test_completion_rejects_end_state_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_identity: str,
    violation_code: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    configs = tmp_path / "vnncomp25"
    configs.mkdir()
    config = configs / "demo.yaml"
    config.write_text("verifier:\n  timeout: 1\n", encoding="utf-8")
    (configs / "other_demo.yaml").write_text(
        "verifier:\n  timeout: 1\n", encoding="utf-8"
    )
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(
        repo,
        benchmark_root,
        run_id=f"drift-{changed_identity}",
        configs_dir=configs,
    )

    if changed_identity == "solver":
        (repo / "target/release/ny").write_text(
            "#!/bin/sh\necho 'changed ny'\n", encoding="utf-8"
        )
    elif changed_identity == "ny_worktree":
        (repo / "scripts/measure_ny_scorecard.sh").write_text(
            "#!/bin/bash\nexit 1\n", encoding="utf-8"
        )
    elif changed_identity == "benchmark":
        (benchmark_root / "README").write_text("changed fixture\n", encoding="utf-8")
    else:
        config.write_text("verifier:\n  timeout: 2\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}
    assert record["integrity"]["status"] == "invalid"
    assert record["completed_successfully"] is False
    assert violation_code in codes


def test_completion_rejects_cuda_runtime_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    monkeypatch.setattr(
        provenance,
        "_recapture_cuda_runtime_from_start",
        lambda _start: {
            **CUDA_RUNTIME_FIXTURE,
            "probe": {"fixture": False},
        },
    )
    start = _capture(repo, benchmark_root, run_id="cuda-runtime-drift")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "cuda_runtime_identity_mismatch" in codes
    assert record["integrity"]["checks"]["cuda_runtime"]["status"] == "invalid"


def test_cuda_runtime_dependency_is_explicitly_not_required_for_cpu_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "values": {
            "NY_ALLOW_NONCUDA_MEASURE": "1",
            "NY_NO_CUDA": "1",
        }
    }
    monkeypatch.setattr(
        provenance,
        "_capture_cuda_runtime_identity",
        lambda _binary: pytest.fail("CPU debug capture must not probe CUDA"),
    )

    identity = REAL_CAPTURE_CUDA_RUNTIME_DEPENDENCY(
        Path("/does/not/need/to/exist"),
        environment,
        ["mip", "cuda"],
    )

    assert identity == {
        "schema": provenance.MEASUREMENT_CUDA_RUNTIME_SCHEMA,
        "status": "not_required",
        "reason": "noncuda_measurement_explicitly_allowed",
    }


def test_cuda_runtime_cpu_debug_requires_exact_cpu_routing() -> None:
    with pytest.raises(provenance.ProvenanceError, match="NY_NO_CUDA=1"):
        REAL_CAPTURE_CUDA_RUNTIME_DEPENDENCY(
            Path("/does/not/need/to/exist"),
            {"values": {"NY_ALLOW_NONCUDA_MEASURE": "1"}},
            ["mip", "cuda"],
        )


def test_cuda_score_measurement_rejects_cpu_routing_override() -> None:
    with pytest.raises(provenance.ProvenanceError, match="NY_NO_CUDA is forbidden"):
        REAL_CAPTURE_CUDA_RUNTIME_DEPENDENCY(
            Path("/must/not/be/probed"),
            {"values": {"NY_NO_CUDA": "1"}},
            ["mip", "cuda"],
        )


def test_completion_rejects_malformed_input_hash_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id="bad-cache")
    cache = start.with_name("input_hash_cache.json")
    cache.write_text(
        json.dumps(
            {
                "schema": "ny_measurement_input_hash_cache_v1",
                "run_id": "bad-cache",
                "start_manifest_sha256": provenance._sha256(start.read_bytes()),
                "entries": {"not-a-content-address": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}
    assert record["integrity"]["status"] == "invalid"
    assert record["completed_successfully"] is False
    assert "input_hash_cache_entries_invalid" in codes


def test_completion_records_complete_run_artifact_bijection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, _metadata, _result, _log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id="complete-evidence",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    run_evidence = record["integrity"]["checks"]["run_evidence"]
    cache_check = record["integrity"]["checks"]["input_hash_cache"]

    assert record["completed_successfully"] is True
    assert record["integrity"]["checks"]["cuda_runtime"]["status"] == "valid"
    assert run_evidence["status"] == "valid"
    assert run_evidence["metadata_count"] == 1
    assert run_evidence["result_count"] == 1
    assert run_evidence["solver_log_count"] == 1
    assert run_evidence["csv_row_count"] == 1
    assert run_evidence["validated_record_count"] == 1
    assert cache_check["entry_count"] == 2
    assert cache_check["referenced_entry_count"] == 2
    assert cache_check["rehashed_entry_count"] == 2


@pytest.mark.parametrize(
    ("artifact", "violation_code"),
    [
        ("metadata", "run_result_artifact_mismatch"),
        ("result", "run_result_artifact_mismatch"),
        ("solver_log", "run_solver_log_artifact_mismatch"),
    ],
)
def test_completion_rejects_tampered_run_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    violation_code: str,
) -> None:
    start, metadata_path, result_path, solver_log_path, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"tampered-{artifact}",
    )
    if artifact == "metadata":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["result_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    elif artifact == "result":
        result_path.write_bytes(b"sat\n((X_0 0.0))\n")
    else:
        solver_log_path.write_bytes(b"tampered solver log\n")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert violation_code in codes


def test_completion_rejects_sat_without_structured_witness_after_consistent_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "sat-without-witness"
    start, metadata_path, result_path, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=run_id,
    )
    result_path.write_bytes(b"sat\n")
    result_digest = provenance._sha256_file(result_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "solver_verdict": "sat",
            "witness_present": True,
            "result_sha256": result_digest,
            "raw_result_sha256": result_digest,
            "counterexample_validation": {"status": "not_checked", "checker": None},
        }
    )
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    start_record = json.loads(start.read_text(encoding="utf-8"))
    csv_path = Path(start_record["measurement"]["output_dir"]) / "demo.csv"
    csv_path.write_text(
        f"demo,model.onnx,property.vnnlib,0,sat,9,{run_id}\n",
        encoding="utf-8",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_sat_witness_invalid" in codes


def test_completion_rejects_empty_unsat_after_consistent_digest_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, metadata_path, result_path, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id="empty-unsat",
    )
    result_path.write_bytes(b"")
    result_digest = provenance._sha256_file(result_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["result_sha256"] = result_digest
    metadata["raw_result_sha256"] = result_digest
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_result_verdict_mismatch" in codes


def test_postflight_sat_witness_parser_accepts_vnnlib1_and_vnnlib2() -> None:
    assert provenance._structured_sat_assignment([b"((X_0 0.25))"])
    assert provenance._structured_sat_assignment(
        [b"X float32 [1, 2]", b"0.25", b"-0.5"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solver_exit_status", 1),
        ("solver_verdict", "maybe"),
        ("timeout_seconds", "120"),
    ],
)
def test_completion_rejects_invalid_verdict_status_or_timeout_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    start, metadata_path, _result, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"invalid-{field}",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_metadata_fields_invalid" in codes


@pytest.mark.parametrize(
    ("mutation", "violation_code"),
    [
        ("result", "run_evidence_file_changed_during_completion"),
        ("csv", "run_evidence_file_changed_during_completion"),
        ("namespace", "run_artifact_namespace_changed_during_completion"),
    ],
)
def test_completion_rechecks_artifact_and_csv_snapshot_after_cache_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    violation_code: str,
) -> None:
    run_id = f"snapshot-{mutation}"
    start, metadata_path, result_path, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=run_id,
    )
    start_record = json.loads(start.read_text(encoding="utf-8"))
    csv_path = Path(start_record["measurement"]["output_dir"]) / "demo.csv"
    original_validate_cache = provenance._validate_input_hash_cache

    def mutate_after_cache(**kwargs):
        validated = original_validate_cache(**kwargs)
        if mutation == "result":
            result_path.write_bytes(b"unknown\n")
        elif mutation == "csv":
            csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n")
        else:
            orphan = metadata_path.parents[1] / "99999-race" / f"{run_id}.results"
            orphan.parent.mkdir()
            orphan.write_bytes(b"unsat\n")
        return validated

    monkeypatch.setattr(provenance, "_validate_input_hash_cache", mutate_after_cache)

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert violation_code in codes


@pytest.mark.parametrize("artifact", ["metadata", "result", "solver_log"])
def test_completion_rejects_deleted_or_orphaned_run_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    start, metadata_path, result_path, solver_log_path, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"deleted-{artifact}",
    )
    {
        "metadata": metadata_path,
        "result": result_path,
        "solver_log": solver_log_path,
    }[artifact].unlink()

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_artifact_trio_incomplete" in codes


def test_completion_rejects_extra_orphan_run_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "orphan-artifact"
    start, metadata_path, _result, _log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=run_id,
    )
    orphan = metadata_path.parents[1] / "99999-orphan" / f"{run_id}.results"
    orphan.parent.mkdir()
    orphan.write_bytes(b"unsat\n")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_artifact_trio_incomplete" in codes


@pytest.mark.parametrize("mutation", ["delete", "digest", "extra"])
def test_completion_rejects_missing_corrupt_or_unreferenced_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    start, _metadata, _result, _log, cache_path = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"cache-{mutation}",
    )
    if mutation == "delete":
        cache_path.unlink()
    else:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if mutation == "digest":
            first = next(iter(cache["entries"].values()))
            first["sha256"] = "0" * 64
        else:
            extra = tmp_path / "unreferenced-input"
            extra.write_bytes(b"unreferenced\n")
            fingerprint = provenance._file_fingerprint(extra)
            key = provenance._input_cache_key(str(extra.resolve()), fingerprint)
            cache["entries"][key] = {
                "path": str(extra.resolve()),
                "fingerprint": fingerprint,
                "sha256": provenance._sha256_file(extra),
            }
        cache_path.write_text(json.dumps(cache) + "\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    expected = {
        "delete": "input_hash_cache_missing_for_run_artifacts",
        "digest": "input_hash_cache_entry_sha256_mismatch",
        "extra": "input_hash_cache_entry_unreferenced",
    }
    assert expected[mutation] in codes


def test_completion_rejects_csv_row_without_matching_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, _metadata, _result, _log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id="csv-tamper",
    )
    start_record = json.loads(start.read_text(encoding="utf-8"))
    csv_path = Path(start_record["measurement"]["output_dir"]) / "demo.csv"
    csv_path.write_text(
        "demo,model.onnx,property.vnnlib,0,sat,9,csv-tamper\n",
        encoding="utf-8",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_csv_artifact_bijection_mismatch" in codes


@pytest.mark.parametrize(
    "unknown_key",
    [
        "NY_UNREVIEWED_SECRET",
        "NY_RUMP_F64_ENGINE_UNREVIEWED",
        "NY_MULTINEURON",
        "NY_MULTINEURON_STEM",
    ],
)
def test_unknown_ny_environment_fails_closed_without_writing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unknown_key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(unknown_key, "do-not-record")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        _capture(repo, benchmark_root, run_id="rejected-run")

    assert not (
        repo / "reports/measured/artifacts/runs/rejected-run/start.json"
    ).exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("BASH_ENV", "/tmp/forged-scorecard-environment"),
        ("BASH_FUNC_ulimit%%", "() { echo 167772160; }"),
    ],
)
def test_shell_injection_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, value)

    with pytest.raises(provenance.ProvenanceError, match="unsafe shell launch"):
        provenance._capture_environment()


def test_inherited_ay_environment_fails_closed_without_writing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("AY_MILP_FLIP_CAP_SECS", "3")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="AY_MILP_FLIP_CAP_SECS"):
        _capture(repo, benchmark_root, run_id="rejected-ay-run")

    assert not (
        repo / "reports/measured/artifacts/runs/rejected-ay-run/start.json"
    ).exists()


@pytest.mark.parametrize(
    "key", ["MIMALLOC_PURGE_DELAY", "MIMALLOC_FUTURE_UNREVIEWED_OPTION"]
)
def test_mimalloc_environment_fails_closed_without_writing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "0")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=key):
        _capture(repo, benchmark_root, run_id="rejected-mimalloc-run")

    assert not (
        repo / "reports/measured/artifacts/runs/rejected-mimalloc-run/start.json"
    ).exists()


def test_missing_build_feature_declaration_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.delenv("NY_BUILD_FEATURES")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(
        provenance.ProvenanceError, match="NY_BUILD_FEATURES is required"
    ):
        _capture(repo, benchmark_root, run_id="missing-features")

    assert not (
        repo / "reports/measured/artifacts/runs/missing-features/start.json"
    ).exists()


def test_rejects_broad_or_git_metadata_output_roots(tmp_path: Path) -> None:
    repo = tmp_path / "ny"
    repo.mkdir()
    (repo / ".git").mkdir()

    with pytest.raises(provenance.ProvenanceError, match="unsafe broad"):
        provenance._validate_mutation_root(repo, repo, "measurement output directory")
    with pytest.raises(provenance.ProvenanceError, match="Git metadata"):
        provenance._validate_mutation_root(
            repo / ".git" / "measurements",
            repo,
            "measurement output directory",
        )


def test_measurement_output_inside_repo_must_be_ignored_and_untracked(
    tmp_path: Path,
) -> None:
    repo, _benchmark_root = _measurement_repos(tmp_path)
    tracked_dir = repo / "reports/measured"
    tracked_dir.mkdir(parents=True)
    (tracked_dir / "demo.csv").write_text("legacy row\n", encoding="utf-8")
    _run("git", "add", "-f", "reports/measured/demo.csv", cwd=repo)
    _run("git", "commit", "-qm", "track canonical scorecard", cwd=repo)

    with pytest.raises(provenance.ProvenanceError, match="contains tracked NY paths"):
        provenance._validate_mutation_root(
            tracked_dir,
            repo,
            "measurement output directory",
        )
    with pytest.raises(provenance.ProvenanceError, match="must be Git-ignored"):
        provenance._validate_mutation_root(
            repo / "unignored-output",
            repo,
            "measurement output directory",
        )

    provenance._validate_mutation_root(
        repo / "reports/isolated-run",
        repo,
        "measurement output directory",
    )


def test_process_snapshot_is_sorted_bounded_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_args = "worker --token top-secret " + "x" * 500
    stdout = (
        "10 1 S 20 3.0 2.0 python python job.py --api-key=hidden\n"
        f"20 1 R 5 99.0 1.0 worker {long_args}\n"
        "30 1 S 10 3.0 4.0 curl curl https://user:pass@example.test/x\n"
    )
    monkeypatch.setattr(
        provenance,
        "_snapshot_command",
        lambda _command: {
            "status": "ok",
            "returncode": 0,
            "stdout": stdout,
            "stdout_truncated": False,
            "stderr": "",
            "stderr_truncated": False,
        },
    )

    snapshot = provenance._process_snapshot()

    assert [entry["pid"] for entry in snapshot["entries"]] == [20, 30, 10]
    encoded = json.dumps(snapshot)
    assert "top-secret" not in encoded
    assert "hidden" not in encoded
    assert "user:pass" not in encoded
    assert snapshot["entries"][0]["args_redacted"].endswith("…")
    assert len(snapshot["entries"][0]["args_redacted"]) == provenance.PROCESS_ARGS_LIMIT


def test_host_state_schema_is_stable_when_commands_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        provenance.os,
        "getloadavg",
        lambda: (_ for _ in ()).throw(OSError("unsupported")),
    )
    monkeypatch.setattr(
        provenance,
        "_utc_now",
        lambda: "2026-07-18T01:02:03.000000Z",
    )

    state = provenance._capture_host_state()

    assert state["schema"] == "ny_measurement_host_state_v1"
    assert state["captured_at_utc"] == "2026-07-18T01:02:03.000000Z"
    assert state["load_average"] == {
        "available": False,
        "one_minute": None,
        "five_minutes": None,
        "fifteen_minutes": None,
    }
    assert state["processes"]["status"] == "unavailable"
    assert state["processes"]["entries"] == []
    assert state["gpu"]["utilization"]["status"] == "unavailable"
    assert state["gpu"]["compute_processes"]["status"] == "unavailable"


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_nn4sys_phase_event_gate_is_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_1D_PHASE_EVENTS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_NN4SYS_1D_PHASE_EVENTS"] == raw
    assert environment["typed_values"]["NY_NN4SYS_1D_PHASE_EVENTS"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_nn4sys_phase_event_gate_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_1D_PHASE_EVENTS", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_NN4SYS_1D_PHASE_EVENTS"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "value"),
    [("1", 1), ("00256", 256), ("4096", 4096)],
)
def test_nn4sys_phase_event_cap_is_bounded_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    value: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS"] == raw
    assert environment["typed_values"]["NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS"] == {
        "type": "bounded_positive_integer",
        "value": value,
        "minimum": 1,
        "maximum": 4096,
    }


@pytest.mark.parametrize("raw", ["", "0", "4097", "-1", "+1", " 1", "1 "])
def test_nn4sys_phase_event_cap_rejects_out_of_range_or_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS", raw)

    with pytest.raises(
        provenance.ProvenanceError,
        match="NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS",
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_nn4sys_mvf_clip_gate_is_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_MVF_CLIP_DIAG", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_NN4SYS_MVF_CLIP_DIAG"] == raw
    assert environment["typed_values"]["NY_NN4SYS_MVF_CLIP_DIAG"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1"])
def test_nn4sys_mvf_clip_gate_rejects_nonexact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_NN4SYS_MVF_CLIP_DIAG", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_NN4SYS_MVF_CLIP_DIAG"
    ):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("name", "maximum"),
    [
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", 4096),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", 16384),
    ],
)
@pytest.mark.parametrize("raw", ["1", "00256"])
def test_nn4sys_mvf_clip_caps_are_typed_bounded_positive_integers(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    maximum: int,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(name, raw)

    environment = provenance._capture_environment()

    assert environment["values"][name] == raw
    assert environment["typed_values"][name] == {
        "type": "bounded_positive_integer",
        "value": int(raw),
        "minimum": 1,
        "maximum": maximum,
    }


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", ""),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", "0"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", "-1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", "+1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", " 1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", "1 "),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", "4097"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", ""),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", "0"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", "-1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", "+1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", " 1"),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", "1 "),
        ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", "16385"),
    ],
)
def test_nn4sys_mvf_clip_caps_reject_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(name, raw)

    with pytest.raises(provenance.ProvenanceError, match=name):
        provenance._capture_environment()


def test_smaller_profile_does_not_shrink_the_address_space_limit() -> None:
    """RLIMIT_AS is virtual address space, NOT the cgroup's physical cap.

    The CUDA driver and ONNX Runtime reserve tens of GiB of VA regardless of how
    much physical memory a host has. An earlier wsl24-20g profile attested
    RLIMIT_AS == memory.max, which starved those reservations: a cersyve instance
    that returns `sat` in 1s under an 80 GiB limit instead burned 106s and
    recorded `timeout` under a 20 GiB one, so the sweep produced all-timeout rows
    that looked like a real measurement.

    The address-space ceiling must remain common across physical profiles and
    strictly above even the largest profile's memory.max. Otherwise a virtual
    reservation becomes the allocation envelope before charged memory does.
    """
    profiles = provenance.CONTAINMENT_PROFILES
    assert set(profiles) == {"gb10-80g", "wsl24-20g"}
    assert provenance.EXPECTED_RLIMIT_AS_BYTES == 160 * 1024**3

    for name, profile in profiles.items():
        assert profile["rlimit_as_bytes"] == provenance.EXPECTED_RLIMIT_AS_BYTES, (
            f"{name}: RLIMIT_AS must stay at the guard ceiling; CUDA/ORT "
            f"reservations do not shrink with the host"
        )

    large = profiles["gb10-80g"]
    small = profiles["wsl24-20g"]
    assert large["memory_max_bytes"] < large["rlimit_as_bytes"]
    assert small["memory_max_bytes"] < small["rlimit_as_bytes"], (
        "wsl24-20g must contain physical memory more tightly than address space"
    )
    assert small["memory_high_bytes"] < small["memory_max_bytes"]


def test_stale_solver_binary_is_refused_before_measurement(tmp_path) -> None:
    """A binary older than the tree it will measure must stop the run.

    The manifest pinned the worktree HEAD and the binary sha256 independently,
    so sealing a binary built at commit A and then moving the worktree to
    commit B produced evidence that looked complete while measuring a
    mismatched pair. It happened: a sweep sealed configs from a commit adding
    the `bab.branching.input_split.sat_escape_branch` preset key while running
    a binary built before its reader existed. Every nn4sys instance
    fail-closed on the unrecognized key, banking 194 `unknown` rows that
    scored as a legitimate zero for the whole category.
    """
    import os

    repo = Path(__file__).resolve().parent.parent
    build_epoch = provenance._last_commit_epoch(repo, provenance._BUILD_INPUT_PATHS)
    config_epoch = provenance._last_commit_epoch(repo, provenance._BEHAVIOUR_INPUT_PATHS)
    assert build_epoch is not None and config_epoch is not None

    binary = tmp_path / "ny"
    binary.write_bytes(b"not a real solver")

    newest = max(build_epoch, config_epoch)
    os.utime(binary, (newest + 10, newest + 10))
    fresh = provenance._capture_build_coherence(repo, binary)
    assert fresh["binary_mtime_epoch"] == newest + 10
    assert fresh["build_inputs_last_commit_epoch"] == build_epoch

    oldest = min(build_epoch, config_epoch)
    os.utime(binary, (oldest - 10, oldest - 10))
    with pytest.raises(provenance.ProvenanceError) as excinfo:
        provenance._capture_build_coherence(repo, binary)
    assert "predates the worktree" in str(excinfo.value)
