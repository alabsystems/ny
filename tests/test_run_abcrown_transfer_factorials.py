# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_abcrown_transfer_factorials.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("abcrown_factorials", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _effective_config(module):
    return {
        "schema": module.EXPECTED_EFFECTIVE_CONFIG_SCHEMA,
        "batch": {"configured_size": 1},
        "branching": {"heuristic": "kfsb", "input_split_adv_check": -1},
        "attack": {"pgd_restarts": 10},
        "alpha_crown": {"enabled": True},
        "beta_crown": {"iterations": 8},
        "clip": {"interm_domain": True},
        "root": {
            "sparse_effective_armed": False,
            "atomic_root_c_margin_iterations": 0,
            "root_spec_prune_requested": False,
        },
        "invprop": {
            "enabled": True,
            "apply_output_constraints_to": ["all"],
            "tighten_input_bounds": False,
            "best_of_oc_and_no_oc": False,
            "directly_optimize": [],
            "share_gammas": False,
            "per_layer_gammas": False,
            "optimize_gammas": True,
            "gamma_lr": 0.5,
            "top_level_output_constraint_matrix": {
                "rows": 1,
                "columns": 2,
                "rhs_entries": 1,
                "is_conjunction": True,
            },
            "serial_clause_rebinding": "not-required",
            "split_lift_requested": False,
            "split_lift_effective_armed": False,
        },
        "route": {"model_kind": "graph"},
    }


def _execution_observations(
    module,
    *,
    exact_c_limit: int = 0,
    exact_c_compressed: bool = False,
    root_spec_prune: bool = False,
    invprop: bool = False,
    optimize_gammas: bool = False,
):
    exact_c_active = exact_c_limit > 0
    exact_source_rows = 4 if exact_c_active else 0
    exact_evaluated_rows = (
        2 if exact_c_active and exact_c_compressed else exact_source_rows
    )
    exact_precertified_rows = exact_source_rows - exact_evaluated_rows
    compact = exact_c_active and exact_c_compressed
    prune_source_rows = 4 if root_spec_prune else 0
    prune_evaluated_rows = 2 if root_spec_prune else 0
    prune_precertified_rows = prune_source_rows - prune_evaluated_rows
    return {
        "schema": module.EXPECTED_EXECUTION_OBSERVATIONS_SCHEMA,
        "run_active": True,
        "recording_conflict": False,
        "exact_c": {
            "observed": exact_c_active,
            "selections": int(exact_c_active),
            "selected_iteration_limit": exact_c_limit if exact_c_active else None,
            "selected_iteration_limit_conflict": False,
            "selected_compressed": exact_c_compressed if exact_c_active else None,
            "selected_compressed_conflict": False,
            "layout_observations": int(exact_c_active),
            "source_rows": exact_source_rows,
            "evaluated_rows": exact_evaluated_rows,
            "precertified_rows": exact_precertified_rows,
            "compressed_selections": int(compact),
            "compressed_layouts_finalized": int(compact),
            "compressed_layouts_rolled_back": 0,
            "compact_commits": int(compact),
            "compact_reconstruction_succeeded": int(compact),
            "compact_reconstruction_failed": 0,
            "compact_binding_map_succeeded": int(compact),
            "compact_binding_map_failed": 0,
            "compact_alpha_candidates": int(compact),
            "compact_alpha_published": int(compact),
            "compact_alpha_dropped": 0,
            "attribution_conflict": False,
            "counter_overflow": False,
            "outcomes_observed": int(exact_c_active),
            "refused_before_commit": 0,
            "committed": int(exact_c_active),
            "iteration_count_outcomes": int(exact_c_active),
            "iteration_count_conflict": False,
            "attempted_iterations": 3 if exact_c_active else 0,
            "accepted_iterations": 2 if exact_c_active else 0,
            "multi_iteration_evidence_outcomes": int(exact_c_active),
            "multiplicative_weights_requested": (
                False if exact_c_active else None
            ),
            "multiplicative_weights_requested_conflict": False,
            "multiplicative_weights_plan_dispatched_outcomes": 0,
            "multiplicative_weights_effective_outcomes": 0,
            "completed_proposals": 3 if exact_c_active else 0,
            "adaptive_plan_dispatches": 0,
            "gradient_plan_num_specs": 1 if exact_c_active else None,
            "gradient_plan_num_specs_conflict": False,
            "gradient_row_count": (
                exact_evaluated_rows if exact_c_active else None
            ),
            "gradient_row_count_conflict": False,
            "multi_iteration_evidence_conflict": False,
            "stop_reasons": {"iteration_limit": 1} if exact_c_active else {},
        },
        "root_spec_prune": {
            "observed": True,
            "attribution_conflict": False,
            "counter_overflow": False,
            "route_observations": 1,
            "configured": root_spec_prune,
            "route_conflict": False,
            "plans_built": int(root_spec_prune),
            "applied": int(root_spec_prune),
            "layout_observations": int(root_spec_prune),
            "source_rows": prune_source_rows,
            "evaluated_rows": prune_evaluated_rows,
            "precertified_rows": prune_precertified_rows,
            "all_pruned": 0,
        },
        "invprop": {
            "observed": invprop,
            "attribution_conflict": False,
            "counter_overflow": False,
            "clause_rebind_attempts": int(invprop),
            "clause_rebind_accepted": int(invprop),
            "clause_rebind_refused": 0,
            "alpha_initializations": int(invprop),
            "gamma_steps_attempted": int(invprop and optimize_gammas),
            "gamma_steps_applied": int(invprop and optimize_gammas),
            # Total folds include discarded SPSA probes. Evaluated folds prove
            # that a nonzero state also reached an authoritative loop iterate.
            "nonzero_output_seed_folds": int(invprop and optimize_gammas),
            "nonzero_evaluated_output_seed_folds": int(
                invprop and optimize_gammas
            ),
        },
        "fresh_domain_clip": {
            "observed": True,
            "attribution_conflict": False,
            "counter_overflow": False,
            "route_observations": 1,
            "configured": False,
            "route_authorized": False,
            "route_conflict": False,
            "attempts": 0,
            "applied": 0,
            "all_clauses_refuted": 0,
            "skipped": 0,
            "tightened_dimensions": 0,
        },
        "patches_materialization": {
            "observed": False,
            "attribution_conflict": False,
            "counter_overflow": False,
            "attempts": 0,
            "succeeded": 0,
            "refused": 0,
            "latent_input_crossover": {
                "attempts": 0,
                "succeeded": 0,
                "refused": 0,
            },
            "network_input_terminal": {
                "attempts": 0,
                "succeeded": 0,
                "refused": 0,
            },
            "other": {"attempts": 0, "succeeded": 0, "refused": 0},
            "finite_deadline_attempts": 0,
            "no_deadline_attempts": 0,
            "affine_geometry_attempts": 0,
            "anchored_geometry_attempts": 0,
            "conflicting_geometry_attempts": 0,
            "input_coefficient_error_attempts": 0,
            "coefficient_error_absent": 0,
            "coefficient_error_materialized": 0,
            "memory_refusals": 0,
            "deadline_refusals": 0,
            "semantic_refusals": 0,
            "memory_receipt_outcomes": 0,
            "nominal_required_bytes": 0,
            "capacity_overage_bytes": 0,
            "admitted_bytes": 0,
            "budget_bytes": 0,
        },
    }


def _execution_evidence(module, document):
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return {
        "execution_observations": document,
        "execution_observations_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }


def _receipt_identity():
    return {
        "schema": "ny-submission-binary-receipt-v1",
        "binary_sha256": "c" * 64,
        "source_kind": "git",
        "source_commit": "a" * 40,
        "source_state_sha256": "b" * 64,
        "cargo_lock_sha256": "d" * 64,
        "ay_commit": "e" * 40,
        "features": "mip,cuda",
        "toolchain_kind": "rustc-vv",
        "toolchain_sha256": "f" * 64,
        "artifact_provenance_sha256": "none",
    }


def _receipt_sha256():
    identity = _receipt_identity()
    fields = (
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
    raw = "".join(f"{field}={identity[field]}\n" for field in fields)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _binary_provenance():
    parent_env = {"NY_INVPROP": "1", "OMP_NUM_THREADS": "1"}
    parent_env_json = json.dumps(parent_env, sort_keys=True, separators=(",", ":"))
    return {
        "ny_source": "explicit",
        "ny_binary": "/sealed/ny",
        "ny_version": "ny 0.1.0",
        "ny_sha256": "c" * 64,
        "ny_receipt": _receipt_identity(),
        "ny_receipt_sha256": _receipt_sha256(),
        "preset_sha256": "9" * 64,
        "parent_env": parent_env,
        "parent_env_sha256": hashlib.sha256(
            parent_env_json.encode("utf-8")
        ).hexdigest(),
    }


def _csv_binary_provenance():
    parsed = _binary_provenance()
    return {
        "ny_source": parsed["ny_source"],
        "ny_binary": parsed["ny_binary"],
        "ny_version": parsed["ny_version"],
        "ny_sha256": parsed["ny_sha256"],
        "ny_receipt_json": json.dumps(
            parsed["ny_receipt"], sort_keys=True, separators=(",", ":")
        ),
        "ny_receipt_sha256": parsed["ny_receipt_sha256"],
        "preset_sha256": parsed["preset_sha256"],
        "parent_env_json": json.dumps(
            parsed["parent_env"], sort_keys=True, separators=(",", ":")
        ),
        "parent_env_sha256": parsed["parent_env_sha256"],
    }


def test_deep_set_builds_nested_mapping_and_rejects_scalar_crossing():
    module = _load_module()
    document = {}
    module._deep_set(document, "bab.branching.candidates", 7)
    assert document == {"bab": {"branching": {"candidates": 7}}}

    document = {"bab": 3}
    with pytest.raises(module.ManifestError, match="non-mapping"):
        module._deep_set(document, "bab.batch_size", 256)


def test_arm_environment_scrubs_declared_factors_before_applying_override():
    module = _load_module()
    experiments = [
        {
            "name": "fixture",
            "arms": [
                {"name": "baseline"},
                {
                    "name": "on",
                    "env": {"NY_ALPHA_FINAL_BOUND_ONLY": "1"},
                },
            ],
        }
    ]
    declared = module._declared_treatment_env_keys(experiments)
    assert declared == {"NY_ALPHA_FINAL_BOUND_ONLY"}

    baseline_env, baseline_scrubbed = module._arm_process_environment(
        {
            "PATH": "/bin",
            "OMP_NUM_THREADS": "4",
            "NY_ALPHA_FINAL_BOUND_ONLY": "1",
            "NY_BRANCH_LA": "1",
            "NY_INVPROP": "0",
            "NY_INVPROP_LR": "0.25",
            "NY_MO_KFSB": "1",
        },
        {},
    )
    assert baseline_env == {"PATH": "/bin", "OMP_NUM_THREADS": "4"}
    assert baseline_scrubbed == [
        "NY_ALPHA_FINAL_BOUND_ONLY",
        "NY_BRANCH_LA",
        "NY_INVPROP",
        "NY_INVPROP_LR",
        "NY_MO_KFSB",
    ]

    treatment_env, treatment_scrubbed = module._arm_process_environment(
        {"PATH": "/bin", "NY_ALPHA_FINAL_BOUND_ONLY": "stale"},
        {"NY_ALPHA_FINAL_BOUND_ONLY": "1"},
    )
    assert treatment_env == {
        "PATH": "/bin",
        "NY_ALPHA_FINAL_BOUND_ONLY": "1",
    }
    assert treatment_scrubbed == ["NY_ALPHA_FINAL_BOUND_ONLY"]


def test_selected_experiment_scrubs_treatments_declared_by_other_experiments(
    monkeypatch, tmp_path
):
    module = _load_module()
    base = tmp_path / "base.yaml"
    base.write_text("general: {}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {
                        "name": "selected",
                        "category": "fixture",
                        "base_preset": str(base),
                        "indices": [0],
                        "arms": [{"name": "baseline", "overrides": {}}],
                    },
                    {
                        "name": "other",
                        "category": "fixture",
                        "base_preset": str(base),
                        "indices": [0],
                        "arms": [
                            {
                                "name": "packed",
                                "env": {"NY_PACKED_GRAPH_ALPHA_QUEUE": "1"},
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NY_PACKED_GRAPH_ALPHA_QUEUE", "1")
    output = tmp_path / "output"

    assert (
        module.main(
            [
                "--manifest",
                str(manifest),
                "--experiment",
                "selected",
                "--dry-run",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
    assert len(execution["arms"]) == 1
    assert execution["arms"][0]["scrubbed_inherited_treatment_env_keys"] == [
        "NY_PACKED_GRAPH_ALPHA_QUEUE"
    ]


def test_preexisting_arm_output_is_refused_before_solver(monkeypatch, tmp_path):
    module = _load_module()
    base = tmp_path / "base.yaml"
    base.write_text("general: {}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {
                        "name": "fixture",
                        "category": "fixture",
                        "base_preset": str(base),
                        "indices": [0],
                        "arms": [{"name": "baseline", "overrides": {}}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    stale = output / "fixture" / "baseline" / "results.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale,untrusted\n", encoding="utf-8")
    monkeypatch.setattr(module, "_resolve_repo_head", lambda _root: "a" * 40)

    def unexpected_solver(*_args, **_kwargs):
        raise AssertionError("solver must not run with a pre-existing arm output")

    monkeypatch.setattr(module.subprocess, "run", unexpected_solver)

    with pytest.raises(module.ManifestError, match="pre-existing arm output CSV"):
        module.main(
            ["--manifest", str(manifest), "--output-dir", str(output)]
        )
    assert stale.read_text(encoding="utf-8") == "stale,untrusted\n"


def test_committed_manifest_has_no_unauthenticated_treatment_fields():
    module = _load_module()
    manifest = module._load_manifest(module.DEFAULT_MANIFEST)
    failures = {}
    for experiment in manifest["experiments"]:
        for arm in experiment["arms"]:
            _checks, unsupported = module._expected_treatment_authentication(arm)
            if unsupported:
                failures[f"{experiment['name']}/{arm['name']}"] = unsupported

    assert failures == {}


def test_repo_head_resolution_ignores_ambient_git_and_path_controls(
    monkeypatch, tmp_path
):
    module = _load_module()
    expected = module._resolve_repo_head(ROOT)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "attacker-tree"))
    monkeypatch.setenv("PATH", str(tmp_path / "attacker-bin"))
    monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "attacker.so"))

    assert module._resolve_repo_head(ROOT) == expected


def test_repo_head_resolution_rejects_non_commit_output(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="not-a-commit\n", stderr=""
        ),
    )

    with pytest.raises(module.ManifestError, match="40-hex"):
        module._resolve_repo_head(ROOT)


def test_materialized_preset_applies_overrides_and_absolutizes_root(tmp_path):
    module = _load_module()
    base_dir = tmp_path / "configs"
    base_dir.mkdir()
    base = base_dir / "base.yaml"
    base.write_text(
        "general:\n  root_path: ../benchmarks/category\n"
        "bab:\n  batch_size: 64\n",
        encoding="utf-8",
    )
    destination = tmp_path / "artifacts" / "preset.yaml"
    module._materialize_arm(
        base_preset=base,
        arm={"name": "treatment", "overrides": {"bab.batch_size": 256}},
        destination=destination,
    )
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert payload["bab"]["batch_size"] == 256
    assert payload["general"]["root_path"] == str(
        (base_dir / "../benchmarks/category").resolve()
    )


def test_effective_config_csv_evidence_round_trips_and_checks_hash(tmp_path):
    module = _load_module()
    output = tmp_path / "results.csv"
    effective = _effective_config(module)
    canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    execution = _execution_observations(module)
    execution["future_runtime_section"] = {"counter": 7}
    execution_canonical = json.dumps(
        execution, sort_keys=True, separators=(",", ":")
    )
    execution_digest = hashlib.sha256(
        execution_canonical.encode("utf-8")
    ).hexdigest()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "property",
                "source_index_zero_based",
                "result",
                "ny_source",
                "ny_binary",
                "ny_version",
                "ny_sha256",
                "ny_receipt_json",
                "ny_receipt_sha256",
                "preset_sha256",
                "parent_env_json",
                "parent_env_sha256",
                "effective_config_json",
                "effective_config_sha256",
                "execution_observations_json",
                "execution_observations_sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "model.onnx",
                "property": "prop.vnnlib",
                "source_index_zero_based": "7",
                "result": "verified",
                **_csv_binary_provenance(),
                "effective_config_json": canonical,
                "effective_config_sha256": digest,
                "execution_observations_json": execution_canonical,
                "execution_observations_sha256": execution_digest,
            }
        )

    evidence = module._read_effective_config_evidence(output)
    assert evidence == [
        {
            "model": "model.onnx",
            "property": "prop.vnnlib",
            "source_index_zero_based": 7,
            "result": "verified",
            **_binary_provenance(),
            "effective_config_sha256": digest,
            "effective_config": effective,
            "execution_observations_sha256": execution_digest,
            "execution_observations": execution,
        }
    ]
    assessment = module._assess_effective_config_evidence(
        evidence,
        [
            {
                "source_index_zero_based": 7,
                "model": "onnx/model.onnx",
                "property": "vnnlib/prop.vnnlib",
            }
        ],
        expected_source_commit="a" * 40,
    )
    assert assessment["observed_row_count_matches"] is True
    assert assessment["observed_row_identity_matches"] is True
    assert assessment["observed_binary_provenance_rows"] == 1
    assert assessment["observed_binary_provenance_consistent"] is True
    assert assessment["observed_effective_config_complete"] is True


@pytest.mark.parametrize("failure", ["noncanonical", "wrong_hash"])
def test_execution_observation_csv_requires_canonical_hash_bound_json(
    tmp_path, failure
):
    module = _load_module()
    output = tmp_path / "results.csv"
    execution = _execution_observations(module, exact_c_limit=4)
    canonical = json.dumps(execution, sort_keys=True, separators=(",", ":"))
    raw = json.dumps(execution, sort_keys=True) if failure == "noncanonical" else canonical
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if failure == "wrong_hash":
        digest = "0" * 64
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "execution_observations_json",
                "execution_observations_sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "execution_observations_json": raw,
                "execution_observations_sha256": digest,
            }
        )

    match = "canonical form" if failure == "noncanonical" else "hash mismatch"
    with pytest.raises(module.ManifestError, match=match):
        module._read_effective_config_evidence(output)


@pytest.mark.parametrize("iteration_limit", [0, 4])
def test_exact_c_runtime_gate_accepts_honest_off_and_on_evidence(iteration_limit):
    module = _load_module()
    document = _execution_observations(module, exact_c_limit=iteration_limit)
    # New runtime sections do not invalidate the current versioned contract.
    document["future_runtime_section"] = {"observed": True}
    observed = [_execution_evidence(module, document)]
    expected = {
        "treatment": "exact_c",
        "expected_iteration_limit": iteration_limit,
    }

    assessment = module._assess_execution_observation_evidence(observed, expected)

    assert assessment["observed_execution_observation_payload_rows"] == 1
    assert assessment["observed_execution_observation_schema_rows"] == 1
    assert assessment["observed_execution_observation_structured_rows"] == 1
    assert assessment["observed_execution_observation_hash_rows"] == 1
    assert assessment["observed_execution_treatment_rows"] == 1
    assert assessment["observed_execution_evidence_complete"] is True
    assert assessment["execution_observation_evidence_issues"] == []


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (
            lambda document: document["exact_c"].update(outcomes_observed=2),
            "selections and outcomes are inconsistent",
        ),
        (
            lambda document: document["exact_c"].update(stop_reasons={}),
            "stop-reason total",
        ),
        (
            lambda document: document["exact_c"].update(
                accepted_iterations=4
            ),
            "accepted_iterations exceeds attempted_iterations",
        ),
        (
            lambda document: document.update(recording_conflict=True),
            "recording conflict",
        ),
        (
            lambda document: document["exact_c"].update(counter_overflow=True),
            "counter overflow",
        ),
        (
            lambda document: document["exact_c"].update(
                attempted_iterations=0, accepted_iterations=0
            ),
            "executed no exact-C iteration",
        ),
        (
            lambda document: document["exact_c"].update(
                selected_iteration_limit=None,
                selected_iteration_limit_conflict=True,
            ),
            "selected iteration limit does not match",
        ),
    ],
)
def test_exact_c_runtime_gate_rejects_inconsistent_or_conflicted_evidence(
    mutation, expected_issue
):
    module = _load_module()
    document = _execution_observations(module, exact_c_limit=4)
    mutation(document)
    assessment = module._assess_execution_observation_evidence(
        [_execution_evidence(module, document)],
        {"treatment": "exact_c", "expected_iteration_limit": 4},
    )

    assert assessment["observed_execution_evidence_complete"] is False
    assert expected_issue in json.dumps(assessment["execution_observation_mismatches"])


def test_exact_c_off_runtime_gate_rejects_any_exact_c_execution():
    module = _load_module()
    document = _execution_observations(module, exact_c_limit=4)
    assessment = module._assess_execution_observation_evidence(
        [_execution_evidence(module, document)],
        {"treatment": "exact_c", "expected_iteration_limit": 0},
    )

    assert assessment["observed_execution_evidence_complete"] is False
    assert "OFF arm recorded exact-C execution" in json.dumps(
        assessment["execution_observation_mismatches"]
    )


@pytest.mark.parametrize(
    ("iteration_limit", "prune_configured", "compressed"),
    [
        (0, False, False),
        (4, False, False),
        (0, True, False),
        (4, True, True),
    ],
)
def test_exact_c_root_prune_factorial_requires_honest_runtime_composition(
    iteration_limit, prune_configured, compressed
):
    module = _load_module()
    document = _execution_observations(
        module,
        exact_c_limit=iteration_limit,
        exact_c_compressed=compressed,
        root_spec_prune=prune_configured,
    )
    expected = {
        "treatment": "exact_c_root_spec_prune",
        "expected_iteration_limit": iteration_limit,
        "prune_configured": prune_configured,
    }

    assessment = module._assess_execution_observation_evidence(
        [_execution_evidence(module, document)], expected
    )

    assert assessment["observed_execution_evidence_complete"] is True
    assert assessment["execution_observation_evidence_issues"] == []


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (
            lambda document: document["root_spec_prune"].update(
                applied=0,
                layout_observations=0,
                source_rows=0,
                evaluated_rows=0,
                precertified_rows=0,
            ),
            "no applied layout",
        ),
        (
            lambda document: document["exact_c"].update(
                selected_compressed=False,
                compressed_selections=0,
                compressed_layouts_finalized=0,
                compact_commits=0,
                compact_reconstruction_succeeded=0,
                compact_binding_map_succeeded=0,
                compact_alpha_candidates=0,
                compact_alpha_published=0,
                source_rows=4,
                evaluated_rows=4,
                precertified_rows=0,
            ),
            "did not select compressed exact-C rows",
        ),
        (
            lambda document: document["exact_c"].update(
                compact_reconstruction_succeeded=0,
                compact_reconstruction_failed=1,
                compact_alpha_published=0,
                compact_alpha_dropped=1,
            ),
            "failed reconstruction",
        ),
        (
            lambda document: document["exact_c"].update(
                compact_alpha_candidates=0,
                compact_alpha_published=0,
            ),
            "no selected compact alpha candidate",
        ),
        (
            lambda document: document["exact_c"].update(
                compact_alpha_published=0,
                compact_alpha_dropped=1,
            ),
            "dropped compact alpha candidate",
        ),
        (
            lambda document: document["root_spec_prune"].update(
                evaluated_rows=1,
                precertified_rows=3,
            ),
            "row totals disagree",
        ),
    ],
)
def test_exact_c_root_prune_combined_arm_fails_closed_on_missing_composition(
    mutation, expected_issue
):
    module = _load_module()
    document = _execution_observations(
        module,
        exact_c_limit=4,
        exact_c_compressed=True,
        root_spec_prune=True,
    )
    mutation(document)
    assessment = module._assess_execution_observation_evidence(
        [_execution_evidence(module, document)],
        {
            "treatment": "exact_c_root_spec_prune",
            "expected_iteration_limit": 4,
            "prune_configured": True,
        },
    )

    assert assessment["observed_execution_evidence_complete"] is False
    assert expected_issue in json.dumps(
        assessment["execution_observation_mismatches"]
    )


def test_exact_c_prune_off_arm_rejects_compressed_selection():
    module = _load_module()
    document = _execution_observations(
        module,
        exact_c_limit=4,
        exact_c_compressed=True,
        root_spec_prune=False,
    )
    assessment = module._assess_execution_observation_evidence(
        [_execution_evidence(module, document)],
        {
            "treatment": "exact_c_root_spec_prune",
            "expected_iteration_limit": 4,
            "prune_configured": False,
        },
    )

    assert assessment["observed_execution_evidence_complete"] is False
    assert "did not select the full row layout" in json.dumps(
        assessment["execution_observation_mismatches"]
    )


@pytest.mark.parametrize("optimize_gammas", [False, True])
def test_invprop_runtime_gate_accepts_off_and_evaluated_on_evidence(
    optimize_gammas
):
    module = _load_module()
    first = _execution_observations(
        module, invprop=True, optimize_gammas=optimize_gammas
    )
    second = copy.deepcopy(first)
    if optimize_gammas:
        # Runtime counts and hashes may vary by instance; they are not static
        # effective-treatment identity.
        second["invprop"]["gamma_steps_attempted"] = 2
    observed = [
        _execution_evidence(module, first),
        _execution_evidence(module, second),
    ]
    expected = {
        "treatment": "invprop_gamma",
        "optimize_gammas": optimize_gammas,
    }

    assessment = module._assess_execution_observation_evidence(observed, expected)

    assert len(
        {row["execution_observations_sha256"] for row in observed}
    ) == (2 if optimize_gammas else 1)
    assert assessment["observed_execution_observation_hash_rows"] == 2
    assert assessment["observed_execution_evidence_complete"] is True


def test_invprop_runtime_gate_rejects_probe_only_gamma_evidence():
    module = _load_module()
    document = _execution_observations(
        module, invprop=True, optimize_gammas=True
    )
    document["invprop"]["nonzero_evaluated_output_seed_folds"] = 0

    # This is an internally consistent event stream: the nonzero fold happened
    # only in a discarded optimizer probe. It must not promote the ON arm.
    assert module._execution_observations_structure_issues(document) == []
    assessment = module._assess_execution_observation_evidence(
        [_execution_evidence(module, document)],
        {"treatment": "invprop_gamma", "optimize_gammas": True},
    )

    assert assessment["observed_execution_evidence_complete"] is False
    assert "no nonzero evaluated output-seed fold" in json.dumps(
        assessment["execution_observation_mismatches"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (
            lambda invprop: invprop.pop(
                "nonzero_evaluated_output_seed_folds"
            ),
            "nonzero_evaluated_output_seed_folds is not a non-negative integer",
        ),
        (
            lambda invprop: invprop.update(
                nonzero_output_seed_folds=1,
                nonzero_evaluated_output_seed_folds=2,
            ),
            "evaluated output-seed folds exceeds total output-seed folds",
        ),
        (
            lambda invprop: invprop.update(
                gamma_steps_attempted=0,
                gamma_steps_applied=0,
                nonzero_output_seed_folds=1,
                nonzero_evaluated_output_seed_folds=1,
            ),
            "evaluated output-seed folds exist without attempted gamma steps",
        ),
        (
            lambda invprop: invprop.update(
                gamma_steps_applied=0,
                nonzero_output_seed_folds=1,
                nonzero_evaluated_output_seed_folds=1,
            ),
            "evaluated output-seed folds exist without an applied gamma step",
        ),
        (
            lambda invprop: invprop.update(
                nonzero_output_seed_folds=0,
                nonzero_evaluated_output_seed_folds=1,
            ),
            "evaluated output-seed folds exist without any output-seed fold",
        ),
    ],
)
def test_invprop_runtime_schema_rejects_invalid_evaluated_fold_causality(
    mutation, expected_issue
):
    module = _load_module()
    document = _execution_observations(
        module, invprop=True, optimize_gammas=True
    )
    mutation(document["invprop"])

    assert expected_issue in json.dumps(
        module._execution_observations_structure_issues(document)
    )


@pytest.mark.parametrize(
    "unsupported_schema",
    [
        "ny_beta_crown_execution_observations_v2",
        "ny_beta_crown_execution_observations_v3",
        "ny_beta_crown_execution_observations_v4",
        "ny_beta_crown_execution_observations_v99",
    ],
)
def test_stale_or_unknown_execution_observation_payload_is_rejected(
    unsupported_schema,
):
    module = _load_module()
    document = _execution_observations(module)
    document["schema"] = unsupported_schema

    assert "execution_observations_v5" in json.dumps(
        module._execution_observations_structure_issues(document)
    )


def test_execution_observation_v5_requires_bounded_multi_iteration_aggregates():
    module = _load_module()
    document = _execution_observations(module, exact_c_limit=4)
    document["exact_c"].pop("multi_iteration_evidence_outcomes")

    assert "multi_iteration_evidence_outcomes is not a non-negative integer" in json.dumps(
        module._execution_observations_structure_issues(document)
    )


def test_execution_observation_v5_requires_patches_materialization_section():
    module = _load_module()
    document = _execution_observations(module)
    document.pop("patches_materialization")

    assert (
        "execution_observations.patches_materialization is not an object"
        in module._execution_observations_structure_issues(document)
    )


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (
            lambda patches: patches.update(attempts=1),
            "patches materialization outcomes do not equal attempts",
        ),
        (
            lambda patches: patches.update(
                admitted_bytes=2, nominal_required_bytes=1, budget_bytes=2
            ),
            "patches materialization admitted-byte receipt does not balance",
        ),
        (
            lambda patches: patches.update(attribution_conflict=True),
            "patches materialization attribution conflict is present",
        ),
    ],
)
def test_execution_observation_v5_rejects_invalid_patches_receipts(
    mutation, expected_issue
):
    module = _load_module()
    document = _execution_observations(module)
    mutation(document["patches_materialization"])

    assert expected_issue in module._execution_observations_structure_issues(
        document
    )


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (
            lambda exact_c: exact_c.update(
                multiplicative_weights_requested=True,
                multiplicative_weights_plan_dispatched_outcomes=1,
                multiplicative_weights_effective_outcomes=1,
                adaptive_plan_dispatches=2,
                gradient_plan_num_specs=4,
                gradient_row_count=4,
            ),
            None,
        ),
        (
            lambda exact_c: exact_c.update(completed_proposals=4),
            "completed proposals exceeds attempted iterations",
        ),
        (
            lambda exact_c: exact_c.update(
                multiplicative_weights_requested=True,
                multiplicative_weights_plan_dispatched_outcomes=1,
                multiplicative_weights_effective_outcomes=1,
                adaptive_plan_dispatches=2,
                gradient_plan_num_specs=1,
            ),
            "MW gradient num_specs does not equal row count",
        ),
        (
            lambda exact_c: exact_c.update(
                multi_iteration_evidence_conflict=True
            ),
            "multi_iteration_evidence_conflict is present",
        ),
    ],
)
def test_execution_observation_v5_authenticates_bounded_gradient_aggregates(
    mutation, expected_issue
):
    module = _load_module()
    document = _execution_observations(module, exact_c_limit=4)
    mutation(document["exact_c"])

    issues = module._execution_observations_structure_issues(document)
    if expected_issue is None:
        assert issues == []
    else:
        assert expected_issue in json.dumps(issues)


@pytest.mark.parametrize(
    ("exact_c_limit", "mutation", "expected_issue"),
    [
        (
            0,
            lambda exact_c: exact_c.update(
                attempted_iterations=1,
                accepted_iterations=1,
                completed_proposals=1,
                selected_iteration_limit_conflict=True,
            ),
            "iteration counts exist without authenticated evidence",
        ),
        (
            4,
            lambda exact_c: exact_c.update(
                attempted_iterations=3,
                accepted_iterations=1,
                completed_proposals=1,
            ),
            "completed proposals is below the per-outcome completion bound",
        ),
        (
            4,
            lambda exact_c: exact_c.update(
                attempted_iterations=2,
                accepted_iterations=2,
                multiplicative_weights_requested=True,
                multiplicative_weights_plan_dispatched_outcomes=1,
                multiplicative_weights_effective_outcomes=0,
                completed_proposals=2,
                adaptive_plan_dispatches=1,
                gradient_plan_num_specs=4,
                gradient_row_count=4,
            ),
            "completed proposals require an MW effective outcome",
        ),
        (
            4,
            lambda exact_c: exact_c.update(
                committed=2,
                iteration_count_outcomes=2,
                attempted_iterations=2,
                accepted_iterations=0,
                multi_iteration_evidence_outcomes=2,
                multiplicative_weights_requested=True,
                multiplicative_weights_plan_dispatched_outcomes=1,
                multiplicative_weights_effective_outcomes=0,
                completed_proposals=0,
                adaptive_plan_dispatches=0,
                gradient_plan_num_specs=1,
                gradient_row_count=1,
            ),
            "one-row MW completions are below active-outcome bound",
        ),
    ],
)
def test_execution_observation_v5_rejects_impossible_aggregate_sequences(
    exact_c_limit, mutation, expected_issue
):
    module = _load_module()
    exact_c = _execution_observations(module, exact_c_limit=exact_c_limit)[
        "exact_c"
    ]
    mutation(exact_c)

    assert expected_issue in json.dumps(
        module._exact_c_multi_iteration_aggregate_issues(exact_c)
    )


def test_execution_observation_v5_accepts_one_row_mw_completion_boundary():
    module = _load_module()
    exact_c = _execution_observations(module, exact_c_limit=4)["exact_c"]
    exact_c.update(
        committed=2,
        iteration_count_outcomes=2,
        attempted_iterations=2,
        accepted_iterations=0,
        multi_iteration_evidence_outcomes=2,
        multiplicative_weights_requested=True,
        multiplicative_weights_plan_dispatched_outcomes=1,
        multiplicative_weights_effective_outcomes=0,
        completed_proposals=1,
        adaptive_plan_dispatches=0,
        gradient_plan_num_specs=1,
        gradient_row_count=1,
    )

    assert module._exact_c_multi_iteration_aggregate_issues(exact_c) == []


@pytest.mark.parametrize(
    ("attempted", "completed", "expected_issue"),
    [
        (8, 6, None),
        (9, 7, "attempted iterations exceeds selected iteration limits"),
    ],
)
def test_execution_observation_v5_bounds_attempts_by_selected_limit(
    attempted, completed, expected_issue
):
    module = _load_module()
    exact_c = _execution_observations(module, exact_c_limit=4)["exact_c"]
    exact_c.update(
        committed=2,
        iteration_count_outcomes=2,
        attempted_iterations=attempted,
        multi_iteration_evidence_outcomes=2,
        completed_proposals=completed,
    )

    issues = module._exact_c_multi_iteration_aggregate_issues(exact_c)
    if expected_issue is None:
        assert issues == []
    else:
        assert expected_issue in json.dumps(issues)


def test_execution_observation_v5_leaves_conflicted_limit_to_outer_validator():
    module = _load_module()
    document = _execution_observations(module, exact_c_limit=4)
    exact_c = document["exact_c"]
    exact_c.update(
        attempted_iterations=5,
        completed_proposals=4,
        selected_iteration_limit_conflict=True,
    )

    aggregate_issues = module._exact_c_multi_iteration_aggregate_issues(exact_c)
    assert "selected iteration limits" not in json.dumps(aggregate_issues)
    assert "selected iteration limit conflict is present" in json.dumps(
        module._execution_observations_structure_issues(document)
    )


@pytest.mark.parametrize(
    ("attempted", "completed", "adaptive", "effective"),
    [
        (5, 3, 2, 0),
        (6, 4, 3, 2),
        (5, 4, 2, 2),
    ],
)
def test_execution_observation_v5_accepts_exact_mw_aggregate_boundaries(
    attempted, completed, adaptive, effective
):
    module = _load_module()
    exact_c = _execution_observations(module, exact_c_limit=4)["exact_c"]
    exact_c.update(
        committed=3,
        iteration_count_outcomes=3,
        attempted_iterations=attempted,
        multi_iteration_evidence_outcomes=3,
        multiplicative_weights_requested=True,
        multiplicative_weights_plan_dispatched_outcomes=3,
        multiplicative_weights_effective_outcomes=effective,
        completed_proposals=completed,
        adaptive_plan_dispatches=adaptive,
        gradient_plan_num_specs=4,
        gradient_row_count=4,
    )

    assert module._exact_c_multi_iteration_aggregate_issues(exact_c) == []


def test_conjunctive_invprop_does_not_require_clause_rebinding():
    module = _load_module()
    document = _execution_observations(module, invprop=True)
    document["invprop"].update(
        clause_rebind_attempts=0,
        clause_rebind_accepted=0,
        clause_rebind_refused=0,
    )
    effective = _effective_config(module)
    effective["invprop"]["serial_clause_rebinding"] = (
        "not_applicable_for_top_level_conjunction"
    )
    row = {**_execution_evidence(module, document), "effective_config": effective}

    assessment = module._assess_execution_observation_evidence(
        [row],
        {"treatment": "invprop_gamma", "optimize_gammas": False},
    )

    assert assessment["observed_execution_evidence_complete"] is True


def test_invprop_unknown_property_shape_cannot_choose_rebind_exemption():
    module = _load_module()
    document = _execution_observations(module, invprop=True)
    effective = _effective_config(module)
    effective["invprop"]["serial_clause_rebinding"] = "property_shape_unavailable"
    row = {**_execution_evidence(module, document), "effective_config": effective}

    assessment = module._assess_execution_observation_evidence(
        [row],
        {"treatment": "invprop_gamma", "optimize_gammas": False},
    )

    assert assessment["observed_execution_evidence_complete"] is False
    assert "requirement is unavailable" in json.dumps(
        assessment["execution_observation_mismatches"]
    )


@pytest.mark.parametrize(
    ("optimize_gammas", "mutation", "expected_issue"),
    [
        (
            False,
            lambda invprop: invprop.update(clause_rebind_attempts=2),
            "accepted + refused clause rebinds",
        ),
        (
            False,
            lambda invprop: invprop.update(clause_rebind_accepted=0),
            "no accepted clause rebind",
        ),
        (
            False,
            lambda invprop: invprop.update(alpha_initializations=0),
            "no alpha initialization",
        ),
        (
            False,
            lambda invprop: invprop.update(
                gamma_steps_attempted=1, gamma_steps_applied=0
            ),
            "gamma-OFF arm executed",
        ),
        (
            True,
            lambda invprop: invprop.update(
                gamma_steps_attempted=0, gamma_steps_applied=0
            ),
            "no attempted gamma step",
        ),
        (
            True,
            lambda invprop: invprop.update(
                gamma_steps_attempted=1, gamma_steps_applied=2
            ),
            "applied gamma steps exceeds attempted",
        ),
        (
            True,
            lambda invprop: invprop.update(
                gamma_steps_attempted=1,
                gamma_steps_applied=0,
                nonzero_output_seed_folds=1,
                nonzero_evaluated_output_seed_folds=0,
            ),
            "no applied gamma step",
        ),
        (
            True,
            lambda invprop: invprop.update(
                gamma_steps_attempted=1,
                gamma_steps_applied=1,
                nonzero_output_seed_folds=0,
                nonzero_evaluated_output_seed_folds=0,
            ),
            "no nonzero output-seed fold",
        ),
        (
            True,
            lambda invprop: invprop.update(
                gamma_steps_attempted=1,
                gamma_steps_applied=1,
                nonzero_output_seed_folds=1,
                nonzero_evaluated_output_seed_folds=0,
            ),
            "no nonzero evaluated output-seed fold",
        ),
    ],
)
def test_invprop_runtime_gate_rejects_missing_or_inconsistent_events(
    optimize_gammas, mutation, expected_issue
):
    module = _load_module()
    document = _execution_observations(
        module, invprop=True, optimize_gammas=optimize_gammas
    )
    mutation(document["invprop"])
    assessment = module._assess_execution_observation_evidence(
        [_execution_evidence(module, document)],
        {
            "treatment": "invprop_gamma",
            "optimize_gammas": optimize_gammas,
        },
    )

    assert assessment["observed_execution_evidence_complete"] is False
    assert expected_issue in json.dumps(assessment["execution_observation_mismatches"])


def test_algorithm_runtime_gate_rejects_missing_payload_and_spoofed_hash():
    module = _load_module()
    expected = {"treatment": "exact_c", "expected_iteration_limit": 4}
    missing = module._assess_execution_observation_evidence([{}], expected)
    assert missing["observed_execution_evidence_complete"] is False
    assert missing["observed_execution_observation_payload_rows"] == 0

    document = _execution_observations(module, exact_c_limit=4)
    spoofed = module._assess_execution_observation_evidence(
        [
            {
                "execution_observations": document,
                "execution_observations_sha256": "0" * 64,
            }
        ],
        expected,
    )
    assert spoofed["observed_execution_evidence_complete"] is False
    assert spoofed["observed_execution_observation_hash_rows"] == 0


@pytest.mark.parametrize("configured", [False, True])
def test_fresh_domain_clip_runtime_gate_accepts_reached_off_and_attempted_on(
    configured
):
    module = _load_module()
    document = _execution_observations(module)
    fresh = document["fresh_domain_clip"]
    fresh["configured"] = configured
    fresh["route_authorized"] = configured
    if configured:
        fresh["attempts"] = 2
        fresh["skipped"] = 2
    assessment = module._assess_execution_observation_evidence(
        [_execution_evidence(module, document)],
        {"treatment": "fresh_domain_clip", "configured": configured},
    )

    assert assessment["observed_execution_evidence_complete"] is True


def test_fresh_domain_clip_runtime_gate_rejects_bad_dispositions_and_conflict():
    module = _load_module()
    document = _execution_observations(module)
    fresh = document["fresh_domain_clip"]
    fresh.update(
        configured=True,
        route_authorized=True,
        attempts=2,
        skipped=1,
        route_conflict=True,
    )
    assessment = module._assess_execution_observation_evidence(
        [_execution_evidence(module, document)],
        {"treatment": "fresh_domain_clip", "configured": True},
    )

    assert assessment["observed_execution_evidence_complete"] is False
    rendered = json.dumps(assessment["execution_observation_mismatches"])
    assert "disposition total" in rendered
    assert "route conflict" in rendered


def _unengaged_fresh_clip_observations(module):
    document = _execution_observations(module)
    document["fresh_domain_clip"].update(
        observed=False,
        route_observations=0,
        configured=None,
        route_authorized=None,
    )
    return document


def _sentinel_binding(*, expected_result="falsified"):
    binding = {
        "corpus_id": "sentinel",
        "source_index_zero_based": 34,
        "model": "onnx/model.onnx",
        "property": "vnnlib/sentinel.vnnlib",
    }
    if expected_result is not None:
        binding["expected_result"] = expected_result
    return binding


def _sentinel_runtime_row(module, result="falsified"):
    return {
        "source_index_zero_based": 34,
        "model": "model.onnx",
        "property": "sentinel.vnnlib",
        "result": result,
        **_execution_evidence(
            module, _unengaged_fresh_clip_observations(module)
        ),
    }


def test_authenticated_falsified_sentinel_may_finish_before_clip_dispatch():
    module = _load_module()
    assessment = module._assess_execution_observation_evidence(
        [_sentinel_runtime_row(module)],
        {"treatment": "fresh_domain_clip", "configured": True},
        [_sentinel_binding()],
    )

    assert assessment["observed_execution_evidence_complete"] is True


@pytest.mark.parametrize(
    ("result", "expected_result"),
    [("verified", "falsified"), ("falsified", None)],
)
def test_fresh_clip_sentinel_exemption_requires_bound_falsified_verdict(
    result, expected_result
):
    module = _load_module()
    assessment = module._assess_execution_observation_evidence(
        [_sentinel_runtime_row(module, result=result)],
        {"treatment": "fresh_domain_clip", "configured": True},
        [_sentinel_binding(expected_result=expected_result)],
    )

    assert assessment["observed_execution_evidence_complete"] is False
    assert "did not observe the dispatcher route" in json.dumps(
        assessment["execution_observation_mismatches"]
    )


def test_fresh_clip_target_without_execution_cannot_use_sentinel_exemption():
    module = _load_module()
    target_binding = {
        **_sentinel_binding(expected_result=None),
        "corpus_id": "unsat-target",
    }
    assessment = module._assess_execution_observation_evidence(
        [_sentinel_runtime_row(module, result="unknown")],
        {"treatment": "fresh_domain_clip", "configured": True},
        [target_binding],
    )

    assert assessment["observed_execution_evidence_complete"] is False
    assert "no clipping attempt" in json.dumps(
        assessment["execution_observation_mismatches"]
    )


def test_expected_sentinel_verdict_is_part_of_static_promotion_evidence():
    module = _load_module()
    effective = _effective_config(module)
    canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    row = {
        "source_index_zero_based": 34,
        "model": "model.onnx",
        "property": "sentinel.vnnlib",
        "result": "verified",
        **_binary_provenance(),
        "effective_config": effective,
        "effective_config_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }
    assessment = module._assess_effective_config_evidence(
        [row], [_sentinel_binding()], expected_source_commit="a" * 40
    )

    assert assessment["expected_result_checks"] == 1
    assert assessment["observed_expected_results_match"] is False
    assert assessment["result_mismatches"][0]["expected"] == "falsified"
    assert assessment["observed_effective_config_complete"] is False


def test_effective_config_assessment_rejects_wrong_arm_value_and_hash_drift():
    module = _load_module()
    first = _effective_config(module)
    second = _effective_config(module)
    second["batch"]["configured_size"] = 2

    def evidence(row: int, effective: dict) -> dict:
        canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"))
        return {
            "source_index_zero_based": row,
            "model": "model.onnx",
            "property": f"prop-{row}.vnnlib",
            "result": "verified",
            **_binary_provenance(),
            "effective_config_sha256": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
            "effective_config": effective,
        }

    expected_bindings = [
        {
            "source_index_zero_based": row,
            "model": "onnx/model.onnx",
            "property": f"vnnlib/prop-{row}.vnnlib",
        }
        for row in (1, 2)
    ]
    assessment = module._assess_effective_config_evidence(
        [evidence(1, first), evidence(2, second)],
        expected_bindings,
        expected_treatment_checks=[
            {
                "source": "bab.batch_size",
                "path": ["batch", "configured_size"],
                "expected": 1,
            }
        ],
        expected_source_commit="a" * 40,
    )

    assert assessment["observed_effective_config_hash_consistent"] is False
    assert assessment["observed_expected_treatment_matches"] is False
    assert assessment["treatment_mismatches"] == [
        {
            "row": 2,
            "source": "bab.batch_size",
            "path": ["batch", "configured_size"],
            "expected": 1,
            "observed": 2,
        }
    ]
    assert assessment["observed_effective_config_complete"] is False


def test_arm_authentication_projects_resolved_runtime_treatment_fields():
    module = _load_module()
    checks, unsupported = module._expected_treatment_authentication(
        {
            "overrides": {
                "bab.batch_size": 256,
                "general.conv_mode": "matrix",
                "attack.pgd_restarts": 100,
                "solver.build_batch_size": 512,
                "bab.branching.input_split.sb_coeff_thresh": 0.01,
                "bab.branching.input_split.reorder_bab": True,
                "bab.branching.input_split.adv_check": 0,
                "bab.root_crown_interm_dense_head": False,
                "bab.atomic_root_c_margin_iterations": 4,
                "model.vgg_abcrown_treatment": True,
                "attack.pgd_order": "input_bab",
            },
            "env": {
                "NY_ROOT_SPARSE_INTERM_CROWN": "1",
                "NY_ROOT_SPEC_PRUNE": "0",
                "NY_INVPROP": "1",
                "NY_INVPROP_OPTIMIZE": "0",
                "NY_INVPROP_LR": "0.25",
                "NY_INVPROP_SPLIT_LIFT": "true",
            },
        }
    )

    assert checks == [
        {
            "source": "bab.batch_size",
            "path": ["batch", "configured_size"],
            "expected": 256,
        },
        {
            "source": "general.conv_mode",
            "path": ["route", "configured_conv_mode"],
            "expected": "matrix",
        },
        {
            "source": "attack.pgd_restarts",
            "path": ["attack", "pgd_restarts"],
            "expected": 100,
        },
        {
            "source": "solver.build_batch_size",
            "path": ["batch", "build_batch_size"],
            "expected": 512,
        },
        {
            "source": "bab.branching.input_split.sb_coeff_thresh",
            "path": ["branching", "input_split_coeff_threshold"],
            "expected": module._as_f32(0.01),
        },
        {
            "source": "bab.branching.input_split.reorder_bab",
            "path": ["branching", "reorder_bab"],
            "expected": True,
        },
        {
            "source": "bab.branching.input_split.adv_check",
            "path": ["branching", "input_split_adv_check"],
            "expected": 0,
        },
        {
            "source": "bab.root_crown_interm_dense_head",
            "path": ["root", "dense_head_configured"],
            "expected": False,
        },
        {
            "source": "bab.atomic_root_c_margin_iterations",
            "path": ["root", "atomic_root_c_margin_iterations"],
            "expected": 4,
        },
        {
            "source": "model.vgg_abcrown_treatment",
            "path": ["route", "vgg_abcrown_treatment_active"],
            "expected": True,
        },
        {
            "source": "attack.pgd_order",
            "path": ["attack", "schedule"],
            "expected": "input_bab",
        },
        {
            "source": "NY_ROOT_SPARSE_INTERM_CROWN",
            "path": ["root", "sparse_effective_armed"],
            "expected": True,
        },
        {
            "source": "NY_ROOT_SPEC_PRUNE",
            "path": ["root", "root_spec_prune_requested"],
            "expected": False,
        },
        {
            "source": "NY_INVPROP",
            "path": ["invprop", "enabled"],
            "expected": True,
        },
        {
            "source": "NY_INVPROP_OPTIMIZE",
            "path": ["invprop", "optimize_gammas"],
            "expected": False,
        },
        {
            "source": "NY_INVPROP_LR",
            "path": ["invprop", "gamma_lr"],
            "expected": module._as_f32(0.25),
        },
        {
            "source": "NY_INVPROP_SPLIT_LIFT",
            "path": ["invprop", "split_lift_requested"],
            "expected": True,
        },
        {
            "source": "NY_INVPROP_SPLIT_LIFT",
            "path": ["invprop", "split_lift_effective_armed"],
            "expected": True,
        },
    ]
    assert unsupported == []


def test_arm_cannot_combine_independently_gated_runtime_treatments():
    module = _load_module()
    with pytest.raises(module.ManifestError, match="may not combine"):
        module._expected_execution_evidence(
            {
                "overrides": {
                    "bab.atomic_root_c_margin_iterations": 4,
                    "bab.clip.input_split_fresh_domain_clip": True,
                },
                "env": {
                    "NY_INVPROP": "1",
                    "NY_INVPROP_OPTIMIZE": "1",
                },
            }
        )


def test_requested_but_inert_split_lift_is_not_promotion_complete():
    module = _load_module()
    checks, unsupported = module._expected_treatment_authentication(
        {"overrides": {}, "env": {"NY_INVPROP_SPLIT_LIFT": "1"}}
    )
    assert unsupported == []
    effective = _effective_config(module)
    effective["invprop"]["split_lift_requested"] = True
    assert effective["invprop"]["split_lift_effective_armed"] is False
    canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    observed = {
        "source_index_zero_based": 0,
        "model": "model.onnx",
        "property": "prop.vnnlib",
        "result": "verified",
        **_binary_provenance(),
        "effective_config": effective,
        "effective_config_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }

    assessment = module._assess_effective_config_evidence(
        [observed],
        None,
        expected_treatment_checks=checks,
        expected_source_commit="a" * 40,
    )

    assert assessment["observed_expected_treatment_matches"] is False
    assert assessment["treatment_mismatches"] == [
        {
            "row": 1,
            "source": "NY_INVPROP_SPLIT_LIFT",
            "path": ["invprop", "split_lift_effective_armed"],
            "expected": True,
            "observed": False,
        }
    ]
    assert assessment["observed_effective_config_complete"] is False


def test_effective_config_assessment_rejects_truncation_identity_drift_and_missing_payload():
    module = _load_module()
    expected = [
        {
            "source_index_zero_based": 1,
            "model": "onnx/model.onnx",
            "property": "vnnlib/one.vnnlib",
        },
        {
            "source_index_zero_based": 3,
            "model": "onnx/model.onnx",
            "property": "vnnlib/two.vnnlib",
        },
    ]
    observed = [
        {
            "source_index_zero_based": 1,
            "model": "model.onnx",
            "property": "one.vnnlib",
            "result": "verified",
            **_binary_provenance(),
            "effective_config": _effective_config(module),
        }
    ]
    truncated = module._assess_effective_config_evidence(
        observed, expected, expected_source_commit="a" * 40
    )
    assert truncated["expected_effective_config_rows"] == 2
    assert truncated["observed_effective_config_rows"] == 1
    assert truncated["observed_row_count_matches"] is False
    assert truncated["observed_row_identity_matches"] is False
    assert truncated["observed_effective_config_complete"] is False

    observed.append(
        {
            "source_index_zero_based": 3,
            "model": "model.onnx",
            "property": "two.vnnlib",
            "result": "verified",
            **_binary_provenance(),
            "effective_config": None,
        }
    )
    missing = module._assess_effective_config_evidence(
        observed, expected, expected_source_commit="a" * 40
    )
    assert missing["observed_row_count_matches"] is True
    assert missing["observed_row_identity_matches"] is True
    assert missing["observed_effective_config_payload_rows"] == 1
    assert missing["observed_effective_config_complete"] is False


def test_effective_config_assessment_rejects_wrong_schema_and_error_result():
    module = _load_module()
    expected = [
        {
            "source_index_zero_based": 0,
            "model": "onnx/model.onnx",
            "property": "vnnlib/prop.vnnlib",
        }
    ]
    observed = [
        {
            "source_index_zero_based": 0,
            "model": "model.onnx",
            "property": "prop.vnnlib",
            "result": "error",
            **_binary_provenance(),
            "effective_config": {"schema": "untrusted_fixture_v1"},
        }
    ]

    assessment = module._assess_effective_config_evidence(
        observed, expected, expected_source_commit="a" * 40
    )
    assert assessment["observed_row_count_matches"] is True
    assert assessment["observed_row_identity_matches"] is True
    assert assessment["observed_effective_config_payload_rows"] == 1
    assert assessment["observed_effective_config_schema_rows"] == 0
    assert assessment["observed_effective_config_structured_rows"] == 0
    assert assessment["observed_supported_result_rows"] == 0
    assert assessment["observed_effective_config_complete"] is False


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("root", "atomic_root_c_margin_iterations"),
        ("root", "root_spec_prune_requested"),
        ("invprop", "split_lift_effective_armed"),
    ],
)
def test_effective_config_requires_exact_c_and_complete_invprop_state(
    section, field
):
    module = _load_module()
    effective = _effective_config(module)
    del effective[section][field]
    canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    observed = [
        {
            "source_index_zero_based": 0,
            "model": "model.onnx",
            "property": "prop.vnnlib",
            "result": "verified",
            **_binary_provenance(),
            "effective_config": effective,
            "effective_config_sha256": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
        }
    ]

    assessment = module._assess_effective_config_evidence(
        observed, None, expected_source_commit="a" * 40
    )

    assert assessment["observed_effective_config_structured_rows"] == 0
    assert assessment["observed_effective_config_complete"] is False


def test_promotion_evidence_rejects_receipt_preset_and_parent_env_drift():
    module = _load_module()
    effective = _effective_config(module)
    canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    base = {
        "source_index_zero_based": 0,
        "model": "model.onnx",
        "property": "prop.vnnlib",
        "result": "verified",
        **_binary_provenance(),
        "effective_config": effective,
        "effective_config_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }
    expected_parent_env = {"NY_INVPROP": "1", "OMP_NUM_THREADS": "1"}
    complete = module._assess_effective_config_evidence(
        [copy.deepcopy(base)],
        None,
        expected_preset_sha256="9" * 64,
        expected_parent_env=expected_parent_env,
        expected_source_commit="a" * 40,
    )
    assert complete["observed_effective_config_complete"] is True

    wrong_receipt = copy.deepcopy(base)
    wrong_receipt["ny_receipt"]["binary_sha256"] = "0" * 64
    receipt_assessment = module._assess_effective_config_evidence(
        [wrong_receipt],
        None,
        expected_preset_sha256="9" * 64,
        expected_parent_env=expected_parent_env,
        expected_source_commit="a" * 40,
    )
    assert receipt_assessment["observed_authenticated_receipt_rows"] == 0
    assert receipt_assessment["observed_effective_config_complete"] is False

    wrong_features = copy.deepcopy(base)
    wrong_features["ny_receipt"]["features"] = "mip"
    wrong_features["ny_receipt_sha256"] = module._receipt_file_sha256(
        wrong_features["ny_receipt"]
    )
    feature_assessment = module._assess_effective_config_evidence(
        [wrong_features],
        None,
        expected_preset_sha256="9" * 64,
        expected_parent_env=expected_parent_env,
        expected_source_commit="a" * 40,
    )
    assert feature_assessment["observed_promotion_receipt_feature_rows"] == 0
    assert feature_assessment["observed_effective_config_complete"] is False
    assert "features must be exactly 'mip,cuda'" in " ".join(
        feature_assessment["effective_config_evidence_issues"]
    )

    wrong_source = copy.deepcopy(base)
    wrong_source["ny_receipt"]["source_commit"] = "b" * 40
    wrong_source["ny_receipt_sha256"] = module._receipt_file_sha256(
        wrong_source["ny_receipt"]
    )
    source_assessment = module._assess_effective_config_evidence(
        [wrong_source],
        None,
        expected_preset_sha256="9" * 64,
        expected_parent_env=expected_parent_env,
        expected_source_commit="a" * 40,
    )
    assert source_assessment["observed_authenticated_receipt_rows"] == 1
    assert source_assessment["observed_receipt_source_commit_matches_repo"] is False
    assert source_assessment["observed_effective_config_complete"] is False

    wrong_preset = copy.deepcopy(base)
    wrong_preset["preset_sha256"] = "8" * 64
    preset_assessment = module._assess_effective_config_evidence(
        [wrong_preset],
        None,
        expected_preset_sha256="9" * 64,
        expected_parent_env=expected_parent_env,
        expected_source_commit="a" * 40,
    )
    assert preset_assessment["observed_preset_sha256_matches_expected"] is False
    assert preset_assessment["observed_effective_config_complete"] is False

    wrong_env = copy.deepcopy(base)
    wrong_env["parent_env"]["NY_INVPROP"] = "0"
    wrong_env_json = json.dumps(
        wrong_env["parent_env"], sort_keys=True, separators=(",", ":")
    )
    wrong_env["parent_env_sha256"] = hashlib.sha256(
        wrong_env_json.encode("utf-8")
    ).hexdigest()
    env_assessment = module._assess_effective_config_evidence(
        [wrong_env],
        None,
        expected_preset_sha256="9" * 64,
        expected_parent_env=expected_parent_env,
        expected_source_commit="a" * 40,
    )
    assert env_assessment["observed_parent_env_matches_expected"] is False
    assert env_assessment["observed_effective_config_complete"] is False


def test_corpus_bound_arm_cannot_override_indices(tmp_path):
    module = _load_module()
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "row-one",
                        "kind": "vnncomp",
                        "category": "fixture",
                        "source_index": 1,
                        "model": "onnx/model.onnx",
                        "property": "vnnlib/prop.vnnlib",
                        "timeout_seconds": 2,
                        "expected": {
                            "model_sha256": "a" * 64,
                            "property_sha256": "b" * 64,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    experiment = {
        "name": "fixture",
        "category": "fixture",
        "corpus_ids": ["row-one"],
        "arms": [{"name": "subset", "indices": [0]}],
    }
    manifest = {"corpus_manifest": str(corpus), "experiments": [experiment]}

    with pytest.raises(module.ManifestError, match="cannot override indices"):
        module._bind_corpus_indices(manifest, [experiment])


def test_corpus_binding_rejects_bool_indices_and_non_string_ids(tmp_path):
    module = _load_module()
    corpus = tmp_path / "corpus.json"
    entry = {
        "id": "row-one",
        "kind": "vnncomp",
        "category": "fixture",
        "source_index": True,
        "model": "onnx/model.onnx",
        "property": "vnnlib/prop.vnnlib",
        "timeout_seconds": 2,
        "expected": {
            "model_sha256": "a" * 64,
            "property_sha256": "b" * 64,
        },
    }
    corpus.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
    experiment = {
        "name": "fixture",
        "category": "fixture",
        "corpus_ids": ["row-one"],
        "arms": [{"name": "baseline"}],
    }
    manifest = {"corpus_manifest": str(corpus), "experiments": [experiment]}

    with pytest.raises(module.ManifestError, match="one-based source_index"):
        module._bind_corpus_indices(manifest, [experiment])

    experiment["corpus_ids"] = [["row-one"]]
    with pytest.raises(module.ManifestError, match="non-empty strings"):
        module._bind_corpus_indices(manifest, [experiment])


def test_corpus_binding_rejects_nondefinitive_expected_result(tmp_path):
    module = _load_module()
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "row-one",
                        "kind": "vnncomp",
                        "category": "fixture",
                        "source_index": 1,
                        "model": "onnx/model.onnx",
                        "property": "vnnlib/prop.vnnlib",
                        "timeout_seconds": 2,
                        "expected": {
                            "model_sha256": "a" * 64,
                            "property_sha256": "b" * 64,
                            "expected_result": "unknown",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    experiment = {
        "name": "fixture",
        "category": "fixture",
        "corpus_ids": ["row-one"],
        "arms": [{"name": "baseline"}],
    }

    with pytest.raises(module.ManifestError, match="invalid expected_result"):
        module._bind_corpus_indices(
            {"corpus_manifest": str(corpus), "experiments": [experiment]},
            [experiment],
        )


def test_factorial_extra_args_cannot_override_bounded_runner_authority(tmp_path):
    module = _load_module()
    args = module._parse_args([])
    experiment = {"name": "fixture", "category": "fixture", "indices": [0]}
    arm = {"name": "bad", "extra_args": ["--property=wrong.vnnlib"]}

    with pytest.raises(module.ManifestError, match="harness-owned flag"):
        module._runner_command(
            experiment=experiment,
            arm=arm,
            preset=tmp_path / "preset.yaml",
            output_csv=tmp_path / "results.csv",
            args=args,
        )

    arm = {"name": "allowed", "extra_args": ["--batch-size", "32"]}
    command = module._runner_command(
        experiment=experiment,
        arm=arm,
        preset=tmp_path / "preset.yaml",
        output_csv=tmp_path / "results.csv",
        args=args,
    )
    assert "--require-ny-receipt" in command
    assert "--extra-arg=--batch-size" in command
    assert "--extra-arg=32" in command


def test_successful_arm_fails_closed_on_truncated_effective_config_csv(
    monkeypatch, tmp_path
):
    module = _load_module()
    base = tmp_path / "base.yaml"
    base.write_text("general: {}\n", encoding="utf-8")
    corpus = tmp_path / "corpus.json"
    entries = []
    for source_index, property_name in [(1, "one.vnnlib"), (3, "two.vnnlib")]:
        entries.append(
            {
                "id": f"row-{source_index}",
                "kind": "vnncomp",
                "category": "fixture",
                "source_index": source_index,
                "model": "onnx/model.onnx",
                "property": f"vnnlib/{property_name}",
                "timeout_seconds": 2,
                "expected": {
                    "model_sha256": "a" * 64,
                    "property_sha256": "b" * 64,
                },
            }
        )
    corpus.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "corpus_manifest": str(corpus),
                "experiments": [
                    {
                        "name": "fixture",
                        "year": 2025,
                        "category": "fixture",
                        "base_preset": str(base),
                        "corpus_ids": ["row-1", "row-3"],
                        "arms": [{"name": "baseline", "overrides": {}}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    effective = _effective_config(module)
    canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def fake_run(command, **kwargs):
        if Path(command[0]).name == "git" and command[1:3] == [
            "rev-parse",
            "--verify",
        ]:
            return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n")
        output = Path(command[command.index("--output") + 1])
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "model",
                    "property",
                    "source_index_zero_based",
                    "result",
                    "ny_source",
                    "ny_binary",
                    "ny_version",
                    "ny_sha256",
                    "ny_receipt_json",
                    "ny_receipt_sha256",
                    "preset_sha256",
                    "parent_env_json",
                    "parent_env_sha256",
                    "effective_config_json",
                    "effective_config_sha256",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "model": "model.onnx",
                    "property": "one.vnnlib",
                    "source_index_zero_based": 0,
                    "result": "verified",
                    **_csv_binary_provenance(),
                    "effective_config_json": canonical,
                    "effective_config_sha256": digest,
                }
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output_root = tmp_path / "output"
    assert (
        module.main(
            ["--manifest", str(manifest), "--output-dir", str(output_root)]
        )
        == 1
    )
    execution = json.loads(
        (output_root / "execution.json").read_text(encoding="utf-8")
    )
    arm = execution["arms"][0]
    assert arm["returncode"] == 0
    assert arm["expected_effective_config_rows"] == 2
    assert arm["observed_effective_config_rows"] == 1
    assert arm["observed_row_count_matches"] is False
    assert arm["observed_row_identity_matches"] is False
    assert arm["observed_effective_config_complete"] is False
    assert "evidence_failure" in arm


@pytest.mark.parametrize("include_runtime", [False, True])
def test_algorithm_arm_promotion_requires_runtime_execution_evidence(
    monkeypatch, tmp_path, include_runtime
):
    module = _load_module()
    base = tmp_path / "base.yaml"
    base.write_text("general: {}\n", encoding="utf-8")
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "row-one",
                        "kind": "vnncomp",
                        "category": "fixture",
                        "source_index": 1,
                        "model": "onnx/model.onnx",
                        "property": "vnnlib/prop.vnnlib",
                        "timeout_seconds": 2,
                        "expected": {
                            "model_sha256": "a" * 64,
                            "property_sha256": "b" * 64,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "corpus_manifest": str(corpus),
                "experiments": [
                    {
                        "name": "exact-runtime-fixture",
                        "year": 2025,
                        "category": "fixture",
                        "base_preset": str(base),
                        "corpus_ids": ["row-one"],
                        "arms": [
                            {
                                "name": "exact-on",
                                "overrides": {
                                    "bab.atomic_root_c_margin_iterations": 4
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run(command, **kwargs):
        if Path(command[0]).name == "git":
            return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n")
        output = Path(command[command.index("--output") + 1])
        preset = Path(command[command.index("--preset") + 1])
        effective = _effective_config(module)
        effective["root"]["atomic_root_c_margin_iterations"] = 4
        effective_canonical = json.dumps(
            effective, sort_keys=True, separators=(",", ":")
        )
        process_parent_env = module._relevant_parent_environment(kwargs["env"])
        parent_env_json = json.dumps(
            process_parent_env, sort_keys=True, separators=(",", ":")
        )
        fieldnames = [
            "model",
            "property",
            "source_index_zero_based",
            "result",
            "ny_source",
            "ny_binary",
            "ny_version",
            "ny_sha256",
            "ny_receipt_json",
            "ny_receipt_sha256",
            "preset_sha256",
            "parent_env_json",
            "parent_env_sha256",
            "effective_config_json",
            "effective_config_sha256",
            "execution_observations_json",
            "execution_observations_sha256",
        ]
        row = {
            "model": "model.onnx",
            "property": "prop.vnnlib",
            "source_index_zero_based": 0,
            "result": "verified",
            **_csv_binary_provenance(),
            "preset_sha256": hashlib.sha256(preset.read_bytes()).hexdigest(),
            "parent_env_json": parent_env_json,
            "parent_env_sha256": hashlib.sha256(
                parent_env_json.encode("utf-8")
            ).hexdigest(),
            "effective_config_json": effective_canonical,
            "effective_config_sha256": hashlib.sha256(
                effective_canonical.encode("utf-8")
            ).hexdigest(),
        }
        if include_runtime:
            execution = _execution_observations(module, exact_c_limit=4)
            execution_canonical = json.dumps(
                execution, sort_keys=True, separators=(",", ":")
            )
            row.update(
                execution_observations_json=execution_canonical,
                execution_observations_sha256=hashlib.sha256(
                    execution_canonical.encode("utf-8")
                ).hexdigest(),
            )
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output_root = tmp_path / "output"
    returncode = module.main(
        ["--manifest", str(manifest), "--output-dir", str(output_root)]
    )
    execution = json.loads(
        (output_root / "execution.json").read_text(encoding="utf-8")
    )
    arm = execution["arms"][0]

    assert arm["observed_effective_config_complete"] is True
    assert arm["observed_execution_evidence_complete"] is include_runtime
    assert arm["observed_promotion_evidence_complete"] is include_runtime
    assert returncode == (0 if include_runtime else 1)
    assert ("evidence_failure" in arm) is (not include_runtime)


def test_committed_manifest_dry_run_materializes_every_arm(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    execution = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
    assert execution["dry_run"] is True
    assert len(execution["arms"]) == 46
    assert {arm["experiment"] for arm in execution["arms"]} == {
        "cgan_recipe",
        "cgan_nch1_prop1_backend_route",
        "cifar100_recipe",
        "cifar100_exact_c_iterations",
        "ml4acopf_row5_invprop_gamma",
        "ml4acopf_row52_invprop_gamma",
        "lsnc_fresh_domain_clip",
        "tinyimagenet_recipe",
        "vgg_recipe",
    }
    for arm in execution["arms"]:
        assert Path(arm["generated_preset"]).is_file()
        assert len(arm["generated_preset_sha256"]) == 64
        assert "--preset" in arm["command"]
        assert arm["corpus_ids"]
        assert arm["resolved_zero_based_indices"]
        assert arm["resolved_index_semantics"] == (
            "zero_based_unfiltered_instances_csv_data_rows"
        )
        assert len(arm["resolved_row_bindings"]) == len(arm["corpus_ids"])
        assert arm["command"].count("--expected-row-binding") == len(
            arm["corpus_ids"]
        )
        assert arm["observed_effective_configs"] == []
        assert arm["expected_effective_config_rows"] == len(arm["corpus_ids"])
        assert arm["observed_effective_config_rows"] == 0
        assert arm["observed_effective_config_schema_rows"] == 0
        assert arm["observed_effective_config_structured_rows"] == 0
        assert arm["observed_supported_result_rows"] == 0
        assert arm["observed_binary_provenance_rows"] == 0
        assert arm["observed_binary_provenance_consistent"] is None
        assert arm["observed_row_count_matches"] is None
        assert arm["observed_row_identity_matches"] is None
        assert arm["observed_effective_config_complete"] is None
        assert arm["expected_result_checks"] == 0
        assert arm["observed_expected_results_match"] is None
        assert arm["result_mismatches"] == []
        assert arm["observed_execution_observations"] == []
        assert arm["observed_execution_observation_rows"] == 0
        assert arm["observed_execution_observation_payload_rows"] == 0
        assert arm["observed_execution_observation_schema_rows"] == 0
        assert arm["observed_execution_observation_structured_rows"] == 0
        assert arm["observed_execution_observation_hash_rows"] == 0
        assert arm["observed_execution_treatment_rows"] == 0
        assert arm["observed_execution_evidence_complete"] is None
        assert arm["observed_promotion_evidence_complete"] is None
    skip_arms = [
        arm for arm in execution["arms"] if arm["arm"] == "skip_final_alpha_gradient"
    ]
    assert len(skip_arms) == 4
    assert all(
        arm["env"] == {"NY_ALPHA_FINAL_BOUND_ONLY": "1"}
        for arm in skip_arms
    )
    packed_arms = [
        arm for arm in execution["arms"] if arm["arm"] == "packed_graph_alpha_queue"
    ]
    assert len(packed_arms) == 2
    assert all(
        arm["env"] == {"NY_PACKED_GRAPH_ALPHA_QUEUE": "1"}
        for arm in packed_arms
    )
    adaptive_arms = [
        arm
        for arm in execution["arms"]
        if arm["arm"] == "adaptive_microbatch_controller"
    ]
    assert len(adaptive_arms) == 2
    assert all(
        arm["env"] == {"NY_ADAPTIVE_MICROBATCH_CONTROLLER": "1"}
        and arm["overrides"]["bab.auto_enlarge_batch_size"] is True
        for arm in adaptive_arms
    )
    adv_check_arms = [
        arm for arm in execution["arms"] if arm["arm"] == "adv_check_0"
    ]
    assert len(adv_check_arms) == 1
    assert adv_check_arms[0]["experiment"] == "cgan_recipe"
    assert adv_check_arms[0]["overrides"] == {
        "bab.branching.input_split.adv_check": 0
    }

    backend_canary = [
        arm
        for arm in execution["arms"]
        if arm["experiment"] == "cgan_nch1_prop1_backend_route"
    ]
    assert [arm["arm"] for arm in backend_canary] == [
        "cgan_nch1_prop1_explicit_cpu",
        "cgan_nch1_prop1_explicit_wgpu_request",
    ]
    assert all(
        arm["corpus_ids"] == ["cgan-nch1-prop1-backend-canary"]
        and arm["resolved_zero_based_indices"] == [1]
        for arm in backend_canary
    )
    assert backend_canary[0]["overrides"] == {"general.device": "cpu"}
    assert backend_canary[0]["env"] == {}
    assert backend_canary[1]["overrides"] == {"general.device": "wgpu"}
    assert backend_canary[1]["env"] == {}
    assert backend_canary[1]["disposition"] == (
        "default_off_quarantined_routing_diagnostic"
    )

    by_experiment = {}
    for arm in execution["arms"]:
        by_experiment.setdefault(
            arm["experiment"], arm["resolved_zero_based_indices"]
        )
    assert by_experiment == {
        "cgan_recipe": [6, 0],
        "cgan_nch1_prop1_backend_route": [1],
        "cifar100_recipe": [51, 13],
        "cifar100_exact_c_iterations": [51],
        "ml4acopf_row5_invprop_gamma": [4],
        "ml4acopf_row52_invprop_gamma": [51],
        "lsnc_fresh_domain_clip": [0, 34, 45],
        "tinyimagenet_recipe": [0, 2],
        "vgg_recipe": [1, 0],
    }

    exact_c_arms = [
        arm
        for arm in execution["arms"]
        if arm["experiment"] == "cifar100_exact_c_iterations"
    ]
    assert [arm["arm"] for arm in exact_c_arms] == [
        "exact_c_0_prune_0",
        "exact_c_4_prune_0",
        "exact_c_0_prune_1",
        "exact_c_4_prune_1",
    ]
    assert [
        arm["overrides"]["bab.atomic_root_c_margin_iterations"]
        for arm in exact_c_arms
    ] == [0, 4, 0, 4]
    assert [arm["env"]["NY_ROOT_SPEC_PRUNE"] for arm in exact_c_arms] == [
        "0",
        "0",
        "1",
        "1",
    ]
    assert [arm["expected_execution_evidence"] for arm in exact_c_arms] == [
        {
            "treatment": "exact_c_root_spec_prune",
            "expected_iteration_limit": 0,
            "prune_configured": False,
        },
        {
            "treatment": "exact_c_root_spec_prune",
            "expected_iteration_limit": 4,
            "prune_configured": False,
        },
        {
            "treatment": "exact_c_root_spec_prune",
            "expected_iteration_limit": 0,
            "prune_configured": True,
        },
        {
            "treatment": "exact_c_root_spec_prune",
            "expected_iteration_limit": 4,
            "prune_configured": True,
        },
    ]
    assert all(
        arm["corpus_ids"] == ["cifar100-medium-1761"]
        for arm in exact_c_arms
    )

    for experiment, corpus_id in (
        ("ml4acopf_row5_invprop_gamma", "vnncomp2025-ml4acopf-row5"),
        ("ml4acopf_row52_invprop_gamma", "vnncomp2025-ml4acopf-row52"),
    ):
        invprop_arms = [
            arm for arm in execution["arms"] if arm["experiment"] == experiment
        ]
        assert [arm["arm"] for arm in invprop_arms] == [
            "invprop_gamma_off",
            "invprop_gamma_on",
        ]
        assert [arm["env"]["NY_INVPROP_OPTIMIZE"] for arm in invprop_arms] == [
            "0",
            "1",
        ]
        assert [
            arm["expected_execution_evidence"] for arm in invprop_arms
        ] == [
            {"treatment": "invprop_gamma", "optimize_gammas": False},
            {"treatment": "invprop_gamma", "optimize_gammas": True},
        ]
        assert all(
            arm["corpus_ids"] == [corpus_id]
            and arm["env"]["NY_INVPROP"] == "1"
            and arm["env"]["NY_INVPROP_LR"] == "0.5"
            for arm in invprop_arms
        )

    fresh_clip_arms = [
        arm
        for arm in execution["arms"]
        if arm["experiment"] == "lsnc_fresh_domain_clip"
    ]
    assert [arm["arm"] for arm in fresh_clip_arms] == [
        "fresh_domain_clip_off",
        "fresh_domain_clip_on",
    ]
    assert [
        arm["expected_execution_evidence"] for arm in fresh_clip_arms
    ] == [
        {"treatment": "fresh_domain_clip", "configured": False},
        {"treatment": "fresh_domain_clip", "configured": True},
    ]
    assert all(
        arm["resolved_zero_based_indices"] == [0, 34, 45]
        and len(arm["resolved_row_bindings"]) == 3
        for arm in fresh_clip_arms
    )
    assert all(
        "expected_result" not in arm["resolved_row_bindings"][0]
        and [
            binding["expected_result"]
            for binding in arm["resolved_row_bindings"][1:]
        ]
        == ["falsified", "falsified"]
        for arm in fresh_clip_arms
    )
    assert sum(len(arm["resolved_row_bindings"]) for arm in fresh_clip_arms) == 6

    upstream_bundle = next(
        arm
        for arm in execution["arms"]
        if arm["experiment"] == "tinyimagenet_recipe"
        and arm["arm"] == "upstream_v25_bundle"
    )
    assert upstream_bundle["overrides"] == {
        "bab.batch_size": 256,
        "bab.branching.method": "kfsb",
        "bab.branching.candidates": 7,
        "bab.branching.reduceop": "max",
        "bab.branching.kfsb_multi": True,
        "bab.alpha_crown.lr_alpha": 0.25,
        "bab.alpha_crown.iterations": 20,
        "bab.beta_crown.lr_alpha": 0.1,
        "bab.beta_crown.lr_beta": 0.15,
        "bab.beta_crown.iterations": 8,
        "bab.clip.interm_domain": True,
        "bab.clip.interm_topk": 20,
        "bab.clip.in_alpha_crown": False,
    }
    generated_bundle = yaml.safe_load(
        Path(upstream_bundle["generated_preset"]).read_text(encoding="utf-8")
    )
    assert generated_bundle["bab"]["branching"]["method"] == "kfsb"
    assert generated_bundle["bab"]["branching"]["kfsb_multi"] is True
    assert generated_bundle["bab"]["alpha_crown"]["lr_alpha"] == 0.25
    assert generated_bundle["bab"]["clip"] == {
        "interm_domain": True,
        "interm_topk": 20,
        "in_alpha_crown": False,
    }


def test_explicit_benchmark_root_is_forwarded(tmp_path):
    benchmark_root = tmp_path / "corpus"
    benchmark_root.mkdir()
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--experiment",
            "cgan_recipe",
            "--benchmark-root",
            str(benchmark_root),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    execution = json.loads(
        (tmp_path / "out" / "execution.json").read_text(encoding="utf-8")
    )
    assert execution["benchmark_root"] == str(benchmark_root.resolve())
    for arm in execution["arms"]:
        index = arm["command"].index("--benchmark-root")
        assert arm["command"][index + 1] == str(benchmark_root.resolve())


def test_pilot_timeout_cap_is_forwarded_and_sealed(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--experiment",
            "vgg_recipe",
            "--timeout-cap",
            "17",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    execution = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
    assert execution["timeout_cap_seconds"] == 17
    for arm in execution["arms"]:
        index = arm["command"].index("--timeout-cap")
        assert arm["command"][index + 1] == "17"


def test_explicit_arm_selection_materializes_only_requested_pair(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--experiment",
            "vgg_recipe",
            "--arm",
            "baseline",
            "--arm",
            "abcrown_treatment",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    execution = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
    assert [(arm["experiment"], arm["arm"]) for arm in execution["arms"]] == [
        ("vgg_recipe", "baseline"),
        ("vgg_recipe", "abcrown_treatment"),
    ]
