# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.benchmark_vnncomp_preset_bounded as bounded
import scripts.benchmark_vnncomp_preset_bounded_results as bounded_results


def _write_placeholder_ny(tmp_path: Path) -> Path:
    ny_path = tmp_path / "fake_ny"
    ny_path.write_text("# placeholder\n", encoding="utf-8")
    ny_path.chmod(0o755)
    return ny_path


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_path = tmp_path / "model.onnx"
    property_path = tmp_path / "prop.vnnlib"
    preset_path = tmp_path / "preset.yaml"
    model_path.write_bytes(b"\x08\x01\x12\x03foo")
    property_path.write_text("", encoding="utf-8")
    preset_path.write_text("general:\n  root_path: .\n", encoding="utf-8")
    return model_path, property_path, preset_path


def _run_main(
    monkeypatch,
    tmp_path: Path,
    ny_path: Path,
    *extra_args: str,
) -> int:
    model_path, property_path, preset_path = _write_inputs(tmp_path)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports" / "benchmarks")
    monkeypatch.setattr(bounded, "NY_PREFLIGHT_TIMEOUT_SECS", 2.0)
    monkeypatch.setattr(
        bounded,
        "get_benchmark_instances",
        lambda year, category, **kwargs: [(model_path, property_path, 2)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--category",
            "fixturecat",
            "--preset",
            str(preset_path),
            "--ny-binary",
            str(ny_path),
            *extra_args,
        ],
    )
    return bounded.main()


def _receipt_identity(binary_sha256: str) -> dict[str, str]:
    return {
        "schema": bounded.NY_RECEIPT_SCHEMA,
        "binary_sha256": binary_sha256,
        "source_kind": "git",
        "source_commit": "a" * 40,
        "source_state_sha256": "b" * 64,
        "cargo_lock_sha256": "c" * 64,
        "ay_commit": "d" * 40,
        "features": "mip,cuda",
        "toolchain_kind": "rustc-vv",
        "toolchain_sha256": "e" * 64,
        "artifact_provenance_sha256": "none",
    }


def _execution_observations() -> dict:
    return {
        "schema": "ny_beta_crown_execution_observations_v5",
        "run_active": True,
        "recording_conflict": False,
        "exact_c": {
            "observed": False,
            "selections": 0,
            "selected_iteration_limit": None,
            "selected_iteration_limit_conflict": False,
            "selected_compressed": None,
            "selected_compressed_conflict": False,
            "layout_observations": 0,
            "source_rows": 0,
            "evaluated_rows": 0,
            "precertified_rows": 0,
            "compressed_selections": 0,
            "compressed_layouts_finalized": 0,
            "compressed_layouts_rolled_back": 0,
            "compact_commits": 0,
            "compact_reconstruction_succeeded": 0,
            "compact_reconstruction_failed": 0,
            "compact_binding_map_succeeded": 0,
            "compact_binding_map_failed": 0,
            "compact_alpha_candidates": 0,
            "compact_alpha_published": 0,
            "compact_alpha_dropped": 0,
            "attribution_conflict": False,
            "counter_overflow": False,
            "outcomes_observed": 0,
            "refused_before_commit": 0,
            "committed": 0,
            "iteration_count_outcomes": 0,
            "iteration_count_conflict": False,
            "attempted_iterations": 0,
            "accepted_iterations": 0,
            "multi_iteration_evidence_outcomes": 0,
            "multiplicative_weights_requested": None,
            "multiplicative_weights_requested_conflict": False,
            "multiplicative_weights_plan_dispatched_outcomes": 0,
            "multiplicative_weights_effective_outcomes": 0,
            "completed_proposals": 0,
            "adaptive_plan_dispatches": 0,
            "gradient_plan_num_specs": None,
            "gradient_plan_num_specs_conflict": False,
            "gradient_row_count": None,
            "gradient_row_count_conflict": False,
            "multi_iteration_evidence_conflict": False,
            "stop_reasons": {},
        },
        "root_spec_prune": {
            "observed": False,
            "attribution_conflict": False,
            "counter_overflow": False,
            "route_observations": 0,
            "configured": None,
            "route_conflict": False,
            "plans_built": 0,
            "applied": 0,
            "layout_observations": 0,
            "source_rows": 0,
            "evaluated_rows": 0,
            "precertified_rows": 0,
            "all_pruned": 0,
        },
        "invprop": {
            "observed": False,
            "attribution_conflict": False,
            "counter_overflow": False,
            "clause_rebind_attempts": 0,
            "clause_rebind_accepted": 0,
            "clause_rebind_refused": 0,
            "alpha_initializations": 0,
            "gamma_steps_attempted": 0,
            "gamma_steps_applied": 0,
            "nonzero_output_seed_folds": 0,
            "nonzero_evaluated_output_seed_folds": 0,
        },
        "fresh_domain_clip": {
            "observed": False,
            "attribution_conflict": False,
            "counter_overflow": False,
            "route_observations": 0,
            "configured": None,
            "route_authorized": None,
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


def _write_receipt(path: Path, identity: dict[str, str]) -> None:
    path.write_text(
        "".join(
            f"{field}={identity[field]}\n" for field in bounded.NY_RECEIPT_FIELDS
        ),
        encoding="utf-8",
    )


def test_bounded_preset_runner_rejects_unlaunchable_ny_binary(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 0))

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)

    exit_code = _run_main(monkeypatch, tmp_path, ny_path)

    assert exit_code == 2, f"Expected preflight failure exit code 2, got {exit_code}"
    assert "preflight timed out" in caplog.text.lower(), (
        f"Expected timeout log entry, got {caplog.text!r}"
    )
    reports_dir = tmp_path / "reports" / "benchmarks"
    assert not reports_dir.exists() or not list(reports_dir.glob("fixturecat_*.csv")), (
        "Preflight failure should not emit a benchmark CSV"
    )


def test_bounded_preset_runner_writes_csv_after_successful_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    for name in list(os.environ):
        if name.startswith("NY_") or name == "OMP_NUM_THREADS":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NY_INVPROP", "0")
    monkeypatch.setenv("NY_INVPROP_OPTIMIZE", "0")
    monkeypatch.setenv("NY_INVPROP_LR", "0.25")
    monkeypatch.setenv("NY_INVPROP_SPLIT_LIFT", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("UNRELATED_SECRET", "not-evidence")
    observed_process_envs: list[dict[str, str]] = []

    def fake_run(command, **kwargs):
        observed_process_envs.append(dict(kwargs["env"]))
        if command[-1] == "--version":
            # The run snapshot must remain stable after main() starts.
            os.environ["NY_INVPROP"] = "1"
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="ny 0.0.0\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "verified",
                    "domains_explored": 7,
                    "domains_verified": 3,
                    "max_depth_reached": 1,
                    "effective_config": {
                        "schema": "fixture_v1",
                        "route": {"model_kind": "graph"},
                    },
                    "execution_observations": _execution_observations(),
                }
            )
            + "\n",
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)

    exit_code = _run_main(monkeypatch, tmp_path, ny_path)

    assert exit_code == 0, f"Expected successful run exit code 0, got {exit_code}"
    report = next((tmp_path / "reports" / "benchmarks").glob("fixturecat_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert reader.fieldnames[:18] == [
            "model",
            "property",
            "timeout",
            "result",
            "elapsed",
            "domains",
            "domains_verified",
            "max_depth",
            "reason",
            "domain_batch_metrics_jsonl",
            "notes",
            "ny_source",
            "ny_binary",
            "ny_version",
            "ny_sha256",
            "source_index_zero_based",
            "effective_config_json",
            "effective_config_sha256",
        ]
        assert reader.fieldnames[-2:] == [
            "execution_observations_json",
            "execution_observations_sha256",
        ]
        rows = list(reader)
    assert len(rows) == 1, f"Expected one benchmark row, got {len(rows)}"
    assert rows[0]["result"] == "verified", (
        f"Expected verified result row, got {rows[0]!r}"
    )
    assert rows[0]["source_index_zero_based"] == "0"
    assert rows[0]["domains"] == "7", f"Expected 7 explored domains, got {rows[0]!r}"
    assert rows[0]["domains_verified"] == "3", (
        f"Expected 3 verified domains, got {rows[0]!r}"
    )
    assert rows[0]["max_depth"] == "1", (
        f"Expected max_depth 1 in CSV row, got {rows[0]!r}"
    )
    assert "ny_source" in rows[0], (
        f"Expected ny_source provenance column in CSV, got columns {list(rows[0].keys())}"
    )
    assert "ny_binary" in rows[0], (
        f"Expected ny_binary provenance column in CSV, got columns {list(rows[0].keys())}"
    )
    assert "ny_version" in rows[0], (
        f"Expected ny_version provenance column in CSV, got columns {list(rows[0].keys())}"
    )
    assert "ny_sha256" in rows[0], (
        f"Expected ny_sha256 provenance column in CSV, got columns {list(rows[0].keys())}"
    )
    assert rows[0]["ny_source"] == "explicit", (
        f"Expected ny_source=explicit when --ny-binary is set, got {rows[0]['ny_source']!r}"
    )
    expected_effective = json.dumps(
        {"schema": "fixture_v1", "route": {"model_kind": "graph"}},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert rows[0]["effective_config_json"] == expected_effective
    assert rows[0]["effective_config_sha256"] == hashlib.sha256(
        expected_effective.encode("utf-8")
    ).hexdigest()
    expected_execution = json.dumps(
        _execution_observations(), sort_keys=True, separators=(",", ":")
    )
    assert rows[0]["execution_observations_json"] == expected_execution
    assert rows[0]["execution_observations_sha256"] == hashlib.sha256(
        expected_execution.encode("utf-8")
    ).hexdigest()
    assert rows[0]["preset_sha256"] == hashlib.sha256(
        (tmp_path / "preset.yaml").read_bytes()
    ).hexdigest()
    parent_env = json.loads(rows[0]["parent_env_json"])
    assert parent_env == {
        "NY_INVPROP": "0",
        "NY_INVPROP_LR": "0.25",
        "NY_INVPROP_OPTIMIZE": "0",
        "NY_INVPROP_SPLIT_LIFT": "1",
        "OMP_NUM_THREADS": "2",
    }
    assert "UNRELATED_SECRET" not in parent_env
    assert rows[0]["parent_env_sha256"] == hashlib.sha256(
        rows[0]["parent_env_json"].encode("utf-8")
    ).hexdigest()
    assert observed_process_envs
    assert all(env["NY_INVPROP"] == "0" for env in observed_process_envs)


def test_required_receipt_is_staged_revalidated_and_recorded(
    monkeypatch, tmp_path: Path
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    binary_sha256 = hashlib.sha256(ny_path.read_bytes()).hexdigest()
    source_receipt = Path(f"{ny_path}.receipt")
    _write_receipt(source_receipt, _receipt_identity(binary_sha256))
    authenticated_paths: list[tuple[Path, Path]] = []

    def authenticate(staged_binary, staged_receipt, process_env):
        assert process_env is not os.environ
        assert staged_binary != ny_path
        assert staged_receipt != source_receipt
        assert staged_binary.read_bytes() == ny_path.read_bytes()
        evidence = bounded._parse_ny_receipt(staged_receipt)
        identity = json.loads(evidence.identity_json)
        assert identity["binary_sha256"] == hashlib.sha256(
            staged_binary.read_bytes()
        ).hexdigest()
        authenticated_paths.append((staged_binary, staged_receipt))
        return evidence

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","effective_config":{"schema":"fixture_v1"}}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded, "_authenticate_ny_receipt", authenticate)
    monkeypatch.setattr(bounded.subprocess, "run", fake_run)

    assert _run_main(monkeypatch, tmp_path, ny_path, "--require-ny-receipt") == 0
    assert len(authenticated_paths) == 2
    assert authenticated_paths[0] == authenticated_paths[1]
    report = next((tmp_path / "reports" / "benchmarks").glob("fixturecat_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["ny_receipt_json"]) == _receipt_identity(binary_sha256)
    assert row["ny_receipt_sha256"] == hashlib.sha256(
        source_receipt.read_bytes()
    ).hexdigest()


def test_required_receipt_missing_fails_before_solver_launch(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("solver must not launch without its required receipt")

    monkeypatch.setattr(bounded.subprocess, "run", unexpected_run)

    assert _run_main(monkeypatch, tmp_path, ny_path, "--require-ny-receipt") == 2
    assert "receipt" in caplog.text.lower()
    reports_dir = tmp_path / "reports" / "benchmarks"
    assert not reports_dir.exists()


def test_required_receipt_source_revalidation_failure_suppresses_csv(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    source_receipt = Path(f"{ny_path}.receipt")
    _write_receipt(
        source_receipt,
        _receipt_identity(hashlib.sha256(ny_path.read_bytes()).hexdigest()),
    )
    authentications = 0

    def authenticate(staged_binary, staged_receipt, process_env):
        nonlocal authentications
        authentications += 1
        if authentications == 2:
            raise RuntimeError("stale source identity")
        return bounded._parse_ny_receipt(staged_receipt)

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","effective_config":{"schema":"fixture_v1"}}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded, "_authenticate_ny_receipt", authenticate)
    monkeypatch.setattr(bounded.subprocess, "run", fake_run)

    assert _run_main(monkeypatch, tmp_path, ny_path, "--require-ny-receipt") == 1
    assert authentications == 2
    assert "stale source identity" in caplog.text
    assert not list((tmp_path / "reports" / "benchmarks").glob("*.csv"))


def test_real_receipt_helper_bridge_accepts_match_and_rejects_binary_drift(
    monkeypatch, tmp_path: Path
) -> None:
    helper = tmp_path / "vnncomp_scripts" / "submission_binary_receipt.sh"
    helper.parent.mkdir(parents=True)
    shutil.copy(
        Path(__file__).resolve().parents[1]
        / "vnncomp_scripts"
        / "submission_binary_receipt.sh",
        helper,
    )
    cargo_lock = tmp_path / "Cargo.lock"
    cargo_lock.write_text("version = 4\n", encoding="utf-8")
    lock_sha256 = hashlib.sha256(cargo_lock.read_bytes()).hexdigest()
    (tmp_path / ".ny-vnncomp-source.txt").write_text(
        "schema=ny-vnncomp-source-v1\n"
        f"ny_commit={'a' * 40}\n"
        f"cargo_lock_sha256={lock_sha256}\n",
        encoding="utf-8",
    )
    binary = tmp_path / "ny"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o500)
    original_binary = binary.read_bytes()
    control_env = {"PATH": os.defpath, "LC_ALL": "C"}
    source = subprocess.run(
        ["bash", str(helper), "identity", str(tmp_path)],
        cwd=tmp_path,
        env=control_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert source.returncode == 0, source.stderr
    source_identity = dict(
        line.split("=", 1) for line in source.stdout.strip().splitlines()
    )
    receipt_identity = {
        "schema": bounded.NY_RECEIPT_SCHEMA,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        **source_identity,
        "features": "mip,cuda",
        "toolchain_kind": "rustc-vv",
        "toolchain_sha256": "f" * 64,
        "artifact_provenance_sha256": "none",
    }
    receipt = tmp_path / "ny.receipt"
    _write_receipt(receipt, receipt_identity)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)

    evidence = bounded._authenticate_ny_receipt(
        binary,
        receipt,
        {"PATH": "/attacker/bin", "LD_PRELOAD": "/attacker/preload.so"},
    )
    assert json.loads(evidence.identity_json) == receipt_identity

    binary.chmod(0o700)
    with binary.open("ab") as handle:
        handle.write(b"# drift\n")
    binary.chmod(0o500)
    with pytest.raises(RuntimeError, match="stale/mismatched binary"):
        bounded._authenticate_ny_receipt(binary, receipt, control_env)

    binary.chmod(0o700)
    binary.write_bytes(original_binary)
    binary.chmod(0o500)
    (tmp_path / ".ny-vnncomp-source.txt").write_text(
        "schema=ny-vnncomp-source-v1\n"
        f"ny_commit={'b' * 40}\n"
        f"cargo_lock_sha256={lock_sha256}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="stale source identity"):
        bounded._authenticate_ny_receipt(binary, receipt, control_env)


def test_receipt_validation_environment_is_minimal_and_deterministic() -> None:
    hostile = {
        "PATH": "/attacker/bin",
        "LD_PRELOAD": "/attacker/preload.so",
        "DYLD_INSERT_LIBRARIES": "/attacker/dylib",
        "TMPDIR": "/attacker/tmp",
        "BASH_ENV": "/attacker/bash-env",
        "GIT_DIR": "/attacker/git",
        "NY_INVPROP": "1",
    }

    control = bounded._receipt_validation_environment(hostile)
    assert control["PATH"].split(os.pathsep)[0:2] == os.defpath.split(os.pathsep)
    assert control["LC_ALL"] == "C"
    assert control["GIT_CONFIG_GLOBAL"] == os.devnull
    assert control["GIT_CONFIG_NOSYSTEM"] == "1"
    assert not (
        {"LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "TMPDIR", "BASH_ENV"}
        & set(control)
    )


def test_preexisting_output_is_refused_before_any_solver_launch(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    output = tmp_path / "existing.csv"
    output.write_text("stale,untrusted\n", encoding="utf-8")

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("NY must not launch when output evidence already exists")

    monkeypatch.setattr(bounded.subprocess, "run", unexpected_run)

    assert _run_main(
        monkeypatch, tmp_path, ny_path, "--output", str(output)
    ) == 2
    assert output.read_text(encoding="utf-8") == "stale,untrusted\n"
    assert "refusing to reuse stale evidence" in caplog.text


def test_atomic_result_publication_cannot_clobber_racing_destination(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "evidence.csv"
    real_link = bounded_results.os.link

    def racing_link(source, destination, *args, **kwargs):
        Path(destination).write_text("racing,owner\n", encoding="utf-8")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(bounded_results.os, "link", racing_link)

    with pytest.raises(FileExistsError):
        bounded_results.write_results(output, [])
    assert output.read_text(encoding="utf-8") == "racing,owner\n"
    assert not list(tmp_path.glob(".evidence.csv.*.tmp"))


def test_atomic_result_publication_does_not_follow_a_racing_symlink(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "evidence.csv"

    def symlink_instead_of_link(source, destination, *args, **kwargs):
        assert not args
        assert not kwargs
        os.symlink(source, destination)

    monkeypatch.setattr(bounded_results.os, "link", symlink_instead_of_link)

    with pytest.raises(
        RuntimeError, match="published result path does not reference staged bytes"
    ):
        bounded_results.write_results(output, [])
    assert output.is_symlink()
    assert not list(tmp_path.glob(".evidence.csv.*.tmp"))


def test_year_selects_matching_default_preset_directory(
    monkeypatch, tmp_path: Path
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    model_path, property_path, _unused_preset = _write_inputs(tmp_path)
    preset_path = tmp_path / "configs" / "vnncomp26" / "fixturecat.yaml"
    preset_path.parent.mkdir(parents=True)
    preset_path.write_text("general:\n  root_path: .\n", encoding="utf-8")
    solver_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        solver_commands.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified"}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        bounded,
        "get_benchmark_instances",
        lambda year, category, **kwargs: [(model_path, property_path, 2)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--year",
            "2026",
            "--category",
            "fixturecat",
            "--ny-binary",
            str(ny_path),
            "--output",
            str(tmp_path / "result.csv"),
        ],
    )

    assert bounded.main() == 0
    assert len(solver_commands) == 1
    command = solver_commands[0]
    assert Path(command[command.index("--preset") + 1]) == preset_path


def test_explicit_relative_preset_is_resolved_once_against_caller_cwd(
    monkeypatch, tmp_path: Path
) -> None:
    caller_dir = tmp_path / "caller"
    repo_dir = tmp_path / "repo"
    caller_dir.mkdir()
    repo_dir.mkdir()
    ny_path = _write_placeholder_ny(caller_dir)
    model_path, property_path, _ = _write_inputs(caller_dir)
    caller_preset = caller_dir / "relative-preset.yaml"
    caller_preset.write_text("general:\n  root_path: caller\n", encoding="utf-8")
    (repo_dir / "relative-preset.yaml").write_text(
        "general:\n  root_path: wrong-repo-file\n", encoding="utf-8"
    )
    solver_presets: list[Path] = []

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        preset = Path(command[command.index("--preset") + 1])
        solver_presets.append(preset)
        assert preset.read_text(encoding="utf-8") == (
            "general:\n  root_path: caller\n"
        )
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout='{"status":"verified"}\n', stderr=""
        )

    monkeypatch.chdir(caller_dir)
    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    monkeypatch.setattr(bounded, "REPO_ROOT", repo_dir)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        bounded,
        "get_benchmark_instances",
        lambda year, category, **kwargs: [(model_path, property_path, 2)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--category",
            "fixturecat",
            "--preset",
            "relative-preset.yaml",
            "--ny-binary",
            str(ny_path),
            "--output",
            str(tmp_path / "results.csv"),
        ],
    )

    assert bounded.main() == 0
    assert solver_presets == [caller_preset]


def test_indices_address_unfiltered_instances_csv_rows(
    monkeypatch, tmp_path: Path
) -> None:
    """A missing earlier asset must not compact and retarget --indices."""
    ny_path = _write_placeholder_ny(tmp_path)
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text("general:\n  root_path: .\n", encoding="utf-8")
    category_dir = tmp_path / "corpus" / "fixturecat" / "1.0"
    (category_dir / "onnx").mkdir(parents=True)
    (category_dir / "vnnlib").mkdir()
    model_path = category_dir / "onnx" / "model.onnx"
    target_property = category_dir / "vnnlib" / "target.vnnlib"
    wrong_property = category_dir / "vnnlib" / "wrong.vnnlib"
    model_path.write_bytes(b"model")
    target_property.write_text("target", encoding="utf-8")
    wrong_property.write_text("wrong", encoding="utf-8")
    (category_dir / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/missing.vnnlib,2\n"
        "onnx/model.onnx,vnnlib/target.vnnlib,2\n"
        "onnx/model.onnx,vnnlib/wrong.vnnlib,2\n",
        encoding="utf-8",
    )
    solver_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        solver_commands.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","domains_explored":1}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports")
    expected_binding = json.dumps(
        {
            "corpus_id": "fixture-target",
            "source_index_zero_based": 1,
            "model": "onnx/model.onnx",
            "property": "vnnlib/target.vnnlib",
            "timeout_seconds": 2,
            "model_sha256": hashlib.sha256(b"model").hexdigest(),
            "property_sha256": hashlib.sha256(b"target").hexdigest(),
        }
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--year",
            "2026",
            "--category",
            "fixturecat",
            "--benchmark-root",
            str(tmp_path / "corpus"),
            "--preset",
            str(preset_path),
            "--ny-binary",
            str(ny_path),
            "--indices",
            "1",
            "--expected-row-binding",
            expected_binding,
            "--output",
            str(tmp_path / "result.csv"),
        ],
    )

    assert bounded.main() == 0
    assert len(solver_commands) == 1
    assert str(target_property) in solver_commands[0]
    assert str(wrong_property) not in solver_commands[0]
    with (tmp_path / "result.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["source_index_zero_based"] == "1"


def test_gzip_only_selected_property_is_staged_and_hash_bound(
    monkeypatch, tmp_path: Path
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text("general:\n  root_path: .\n", encoding="utf-8")
    category_dir = tmp_path / "corpus" / "fixturecat"
    (category_dir / "onnx").mkdir(parents=True)
    (category_dir / "vnnlib").mkdir()
    model_path = category_dir / "onnx" / "model.onnx"
    logical_property = category_dir / "vnnlib" / "target.vnnlib"
    model_path.write_bytes(b"model")
    with gzip.open(Path(f"{logical_property}.gz"), "wb") as handle:
        handle.write(b"target")
    (category_dir / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/target.vnnlib,2\n", encoding="utf-8"
    )
    solver_properties: list[Path] = []

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        property_arg = Path(command[command.index("--property") + 1])
        solver_properties.append(property_arg)
        assert property_arg.name == logical_property.name
        assert property_arg.parent.name == "vnnlib"
        assert property_arg.read_bytes() == b"target"
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","effective_config":{"schema":"fixture_v1"}}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports")
    binding = json.dumps(
        {
            "corpus_id": "fixture-target",
            "source_index_zero_based": 0,
            "model": "onnx/model.onnx",
            "property": "vnnlib/target.vnnlib",
            "timeout_seconds": 2,
            "model_sha256": hashlib.sha256(b"model").hexdigest(),
            "property_sha256": hashlib.sha256(b"target").hexdigest(),
        }
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--category",
            "fixturecat",
            "--benchmark-root",
            str(tmp_path / "corpus"),
            "--preset",
            str(preset_path),
            "--ny-binary",
            str(ny_path),
            "--indices",
            "0",
            "--expected-row-binding",
            binding,
            "--output",
            str(tmp_path / "result.csv"),
        ],
    )

    assert bounded.main() == 0
    assert len(solver_properties) == 1
    assert not logical_property.exists(), "staging must not mutate the benchmark corpus"


def test_gzip_only_selected_model_and_property_are_staged_and_hash_bound(
    monkeypatch, tmp_path: Path
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text("general:\n  root_path: .\n", encoding="utf-8")
    category_dir = tmp_path / "corpus" / "fixturecat"
    (category_dir / "onnx").mkdir(parents=True)
    (category_dir / "vnnlib").mkdir()
    logical_model = category_dir / "onnx" / "model.onnx"
    logical_property = category_dir / "vnnlib" / "target.vnnlib"
    with gzip.open(Path(f"{logical_model}.gz"), "wb") as handle:
        handle.write(b"compressed-model")
    with gzip.open(Path(f"{logical_property}.gz"), "wb") as handle:
        handle.write(b"compressed-property")
    (category_dir / "instances.csv").write_text(
        "onnx/model.onnx.gz,vnnlib/target.vnnlib.gz,2\n", encoding="utf-8"
    )

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        model_arg = Path(command[2])
        property_arg = Path(command[command.index("--property") + 1])
        assert model_arg.parent.name == "onnx"
        assert property_arg.parent.name == "vnnlib"
        assert model_arg.read_bytes() == b"compressed-model"
        assert property_arg.read_bytes() == b"compressed-property"
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","effective_config":{"schema":"fixture_v1"}}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports")
    binding = json.dumps(
        {
            "corpus_id": "fixture-target",
            "source_index_zero_based": 0,
            "model": "onnx/model.onnx",
            "property": "vnnlib/target.vnnlib",
            "timeout_seconds": 2,
            "model_sha256": hashlib.sha256(b"compressed-model").hexdigest(),
            "property_sha256": hashlib.sha256(b"compressed-property").hexdigest(),
        }
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--category",
            "fixturecat",
            "--benchmark-root",
            str(tmp_path / "corpus"),
            "--preset",
            str(preset_path),
            "--ny-binary",
            str(ny_path),
            "--indices",
            "0",
            "--expected-row-binding",
            binding,
            "--output",
            str(tmp_path / "result.csv"),
        ],
    )

    assert bounded.main() == 0
    assert not logical_model.exists(), "staging must not mutate the benchmark corpus"
    assert not logical_property.exists(), "staging must not mutate the benchmark corpus"


def test_nested_gzip_paths_preserve_complete_immutable_binding_suffix(
    tmp_path: Path,
) -> None:
    category_dir = tmp_path / "fixturecat" / "2.0"
    logical_model = category_dir / "onnx" / "medical" / "model.onnx"
    logical_property = (
        category_dir / "vnnlib" / "medical" / "properties" / "target.vnnlib"
    )
    logical_model.parent.mkdir(parents=True)
    logical_property.parent.mkdir(parents=True)
    with gzip.open(Path(f"{logical_model}.gz"), "wb") as handle:
        handle.write(b"nested-model")
    with gzip.open(Path(f"{logical_property}.gz"), "wb") as handle:
        handle.write(b"nested-property")

    selected, stage = bounded._stage_gzip_only_inputs(
        [(4, (logical_model, logical_property, 30))]
    )
    assert stage is not None
    try:
        _, (staged_model, staged_property, _) = selected[0]
        assert staged_model.parts[-3:] == ("onnx", "medical", "model.onnx")
        assert staged_property.parts[-4:] == (
            "vnnlib",
            "medical",
            "properties",
            "target.vnnlib",
        )
        binding = json.dumps(
            {
                "corpus_id": "nested-row",
                "source_index_zero_based": 4,
                "model": "onnx/medical/model.onnx",
                "property": "vnnlib/medical/properties/target.vnnlib",
                "timeout_seconds": 30,
                "model_sha256": hashlib.sha256(b"nested-model").hexdigest(),
                "property_sha256": hashlib.sha256(b"nested-property").hexdigest(),
            }
        )
        assert bounded._validate_expected_row_bindings(selected, [binding]) is None
    finally:
        stage.cleanup()


def test_archive_only_inputs_require_staging_aware_manifest_mode(
    tmp_path: Path,
) -> None:
    category_dir = tmp_path / "fixturecat"
    logical_model = category_dir / "onnx" / "model.onnx"
    logical_property = category_dir / "vnnlib" / "target.vnnlib"
    logical_model.parent.mkdir(parents=True)
    logical_property.parent.mkdir(parents=True)
    with gzip.open(Path(f"{logical_model}.gz"), "wb") as handle:
        handle.write(b"model")
    with gzip.open(Path(f"{logical_property}.gz"), "wb") as handle:
        handle.write(b"property")
    (category_dir / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/target.vnnlib,30\n", encoding="utf-8"
    )

    assert bounded.get_benchmark_instances(
        2026, "fixturecat", benchmark_root=tmp_path
    ) == []
    assert bounded.get_benchmark_instances(
        2026,
        "fixturecat",
        benchmark_root=tmp_path,
        preserve_source_rows=True,
    ) == [(logical_model, logical_property, 30)]


def test_version_manifest_fallback_is_limited_to_2026_and_later(
    tmp_path: Path,
) -> None:
    version_dir = tmp_path / "fixturecat" / "1.0"
    model = version_dir / "onnx" / "model.onnx"
    property_path = version_dir / "vnnlib" / "target.vnnlib"
    model.parent.mkdir(parents=True)
    property_path.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    property_path.write_bytes(b"property")
    (version_dir / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/target.vnnlib,30\n", encoding="utf-8"
    )

    assert bounded.get_benchmark_instances(
        2025, "fixturecat", benchmark_root=tmp_path
    ) == []
    assert bounded.get_benchmark_instances(
        2026, "fixturecat", benchmark_root=tmp_path
    ) == [(model, property_path, 30)]


def test_full_run_fails_closed_on_missing_late_manifest_input(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """A full run must not silently compact an incomplete category."""
    ny_path = _write_placeholder_ny(tmp_path)
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text("general:\n  root_path: .\n", encoding="utf-8")
    category_dir = tmp_path / "corpus" / "fixturecat" / "1.0"
    (category_dir / "onnx").mkdir(parents=True)
    (category_dir / "vnnlib").mkdir()
    (category_dir / "onnx" / "model.onnx").write_bytes(b"model")
    (category_dir / "vnnlib" / "present.vnnlib").write_bytes(b"present")
    (category_dir / "vnnlib" / "late.vnnlib").write_bytes(b"late")
    (category_dir / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/present.vnnlib,2\n"
        "onnx/missing-model.onnx,vnnlib/late.vnnlib,2\n",
        encoding="utf-8",
    )
    solver_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        solver_commands.append(command)
        raise AssertionError("no solver row may run after corpus preflight fails")

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--year",
            "2026",
            "--category",
            "fixturecat",
            "--benchmark-root",
            str(tmp_path / "corpus"),
            "--preset",
            str(preset_path),
            "--ny-binary",
            str(ny_path),
        ],
    )

    assert bounded.main() == 1
    assert not solver_commands
    assert "source row 1 has unavailable model input" in caplog.text


def test_json_verdict_parser_and_status_contract_fail_closed() -> None:
    output = (
        'diagnostic {"phase":"setup"}\n'
        '{"status":"unknown","reason":"a quoted } brace"}\n'
        '{"status":"verified","reason":"final { verdict"}\n'
    )
    assert bounded._parse_json_from_output(output) == {
        "status": "verified",
        "reason": "final { verdict",
    }
    assert bounded._validated_payload_status(
        {"status": "verified", "property_status": "safe"}
    ) == ("verified", None)
    status, error = bounded._validated_payload_status(
        {"status": "verified", "property_status": "unknown"}
    )
    assert status is None
    assert error is not None
    assert "status/property_status mismatch" in error
    status, error = bounded._validated_payload_status({"status": "finished"})
    assert status is None
    assert error is not None
    assert "unsupported status" in error
    assert bounded._definitive_status_matches_exit_code("unknown", 0)
    assert bounded._definitive_status_matches_exit_code("unknown", 2)
    assert bounded._normalize_status("potential_violation") == "unknown"
    assert not bounded._definitive_status_matches_exit_code("unknown", 1)
    assert not bounded._definitive_status_matches_exit_code("timeout", 2)
    payload, error = bounded._select_unambiguous_verdict_payload(
        '{"status":"verified","effective_config":{"schema":"one"}}',
        '{"status":"falsified","effective_config":{"schema":"two"}}',
    )
    assert payload is None
    assert error is not None
    assert "conflicting" in error
    payload, error = bounded._select_unambiguous_verdict_payload(
        '{"status":"verified"}',
        'diagnostic: truncated {"status":"falsified"',
    )
    assert payload is None
    assert error == "malformed JSON verdict marker in stderr"


def test_conflicting_stdout_stderr_verdicts_are_not_recorded(
    monkeypatch, tmp_path: Path
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","effective_config":{"schema":"one"}}\n',
            stderr='{"status":"falsified","effective_config":{"schema":"two"}}\n',
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    assert _run_main(monkeypatch, tmp_path, ny_path) == 0
    report = next((tmp_path / "reports" / "benchmarks").glob("fixturecat_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["result"] == "error"
    assert "conflicting stdout/stderr verdict statuses" in row["reason"]
    assert row["effective_config_json"] == ""


def test_execution_observations_are_hashed_separately_and_must_agree() -> None:
    effective = {"schema": "fixture_v1", "root": {"iterations": 4}}
    first_observations = _execution_observations()
    second_observations = json.loads(json.dumps(first_observations))
    second_observations["invprop"]["observed"] = True
    second_observations["invprop"]["alpha_initializations"] = 1

    first_payload = {
        "status": "verified",
        "effective_config": effective,
        "execution_observations": first_observations,
    }
    second_payload = {
        "status": "verified",
        "effective_config": effective,
        "execution_observations": second_observations,
    }
    first_effective = bounded._effective_config_evidence(first_payload)
    second_effective = bounded._effective_config_evidence(second_payload)
    first_execution = bounded._execution_observations_evidence(first_payload)
    second_execution = bounded._execution_observations_evidence(second_payload)

    assert first_effective == second_effective
    assert first_execution != second_execution
    assert first_execution[1] == hashlib.sha256(
        first_execution[0].encode("utf-8")
    ).hexdigest()

    payload, error = bounded._select_unambiguous_verdict_payload(
        json.dumps(first_payload), json.dumps(second_payload)
    )
    assert payload is None
    assert error == "conflicting stdout/stderr verdict field 'execution_observations'"


def test_extra_args_cannot_override_harness_authority(
    monkeypatch, caplog
) -> None:
    for argument in (
        "--timeout",
        "--timeout=0",
        "--property=wrong.vnnlib",
        "-pwrong.vnnlib",
        "--preset=wrong.yaml",
        "--json=false",
        "--max-domains=0",
        "--domain-batch-metrics-jsonl=/tmp/wrong.jsonl",
    ):
        error = bounded._validate_extra_args([argument])
        assert error is not None
        assert "harness-owned" in error
    assert bounded._validate_extra_args(
        ["--branching=input", "--batch-size", "32"]
    ) is None

    monkeypatch.setattr(
        bounded.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("authority validation must precede NY execution")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--category",
            "fixturecat",
            "--extra-arg=--timeout=0",
        ],
    )
    assert bounded.main() == 2
    assert "harness-owned flag '--timeout'" in caplog.text


def test_decisive_json_status_must_match_process_exit_code(
    monkeypatch, tmp_path: Path
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)

    def fake_run(command, **kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        return subprocess.CompletedProcess(
            args=command,
            returncode=4,
            stdout='{"status":"verified","effective_config":{"schema":"fixture_v1"}}\n',
            stderr="post-verdict operational failure",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    assert _run_main(monkeypatch, tmp_path, ny_path) == 0
    report = next((tmp_path / "reports" / "benchmarks").glob("fixturecat_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["result"] == "error"
    assert "status/exit mismatch" in rows[0]["reason"]
    assert rows[0]["effective_config_json"] == ""


def test_selected_source_row_with_missing_input_fails_closed(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text("general:\n  root_path: .\n", encoding="utf-8")
    category_dir = tmp_path / "corpus" / "fixturecat"
    (category_dir / "onnx").mkdir(parents=True)
    model_path = category_dir / "onnx" / "model.onnx"
    model_path.write_bytes(b"model")
    (category_dir / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/missing.vnnlib,2\n",
        encoding="utf-8",
    )
    solver_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        solver_commands.append(command)
        raise AssertionError("solver must not run for a missing selected input")

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--category",
            "fixturecat",
            "--benchmark-root",
            str(tmp_path / "corpus"),
            "--preset",
            str(preset_path),
            "--ny-binary",
            str(ny_path),
            "--indices",
            "0",
        ],
    )

    assert bounded.main() == 1
    assert not solver_commands
    assert "refusing to substitute" in caplog.text


def test_timeout_cap_is_explicitly_applied_to_pilot_command_and_watchdog(
    monkeypatch, tmp_path: Path
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    observed: list[tuple[list[str], int | None]] = []

    def fake_run(command, **kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        observed.append((list(command), kwargs.get("timeout")))
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"timeout","domains_explored":0}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    exit_code = _run_main_with_argv(
        monkeypatch,
        tmp_path,
        ny_path,
        ["--timeout-cap", "1", "--timeout-slack", "3"],
    )

    assert exit_code == 0
    assert len(observed) == 1
    command, watchdog = observed[0]
    timeout_index = command.index("--timeout")
    assert command[timeout_index + 1] == "1"
    assert watchdog == 4


def test_bounded_preset_runner_passes_domain_batch_metrics_sidecar(
    monkeypatch, tmp_path: Path
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    seen_commands: list[list[str]] = []
    metrics_dir = tmp_path / "domain_batch_metrics"

    def fake_run(command, **kwargs):
        seen_commands.append(list(command))
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="ny 0.0.0\n",
                stderr="",
            )
        metrics_path = None
        for index, value in enumerate(command):
            if value == "--domain-batch-metrics-jsonl":
                metrics_path = Path(command[index + 1])
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text('{"schema_version":"graph_domain_batch_metrics_v1"}\n', encoding="utf-8")
                break
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","domains_explored":7,"domains_verified":3,"max_depth_reached":1}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    model_path, property_path, preset_path = _write_inputs(tmp_path)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports" / "benchmarks")
    monkeypatch.setattr(bounded, "NY_PREFLIGHT_TIMEOUT_SECS", 2.0)
    monkeypatch.setattr(
        bounded,
        "get_benchmark_instances",
        lambda year, category, **kwargs: [(model_path, property_path, 2)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--category",
            "fixturecat",
            "--preset",
            str(preset_path),
            "--ny-binary",
            str(ny_path),
            "--domain-batch-metrics-dir",
            str(metrics_dir),
        ],
    )

    exit_code = bounded.main()

    assert exit_code == 0, f"Expected successful run exit code 0, got {exit_code}"
    domain_commands = [cmd for cmd in seen_commands if "--domain-batch-metrics-jsonl" in cmd]
    assert len(domain_commands) == 1, f"expected one benchmark command with domain-batch sidecar, got {seen_commands!r}"
    report = next((tmp_path / "reports" / "benchmarks").glob("fixturecat_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["domain_batch_metrics_jsonl"].endswith("fixturecat_idx0000.jsonl"), rows[0]


def test_resolve_ny_binary_prefers_shared_over_worker_local(tmp_path: Path) -> None:
    """Shared repo binaries must be preferred over worker-local (#4346)."""
    shared_release = tmp_path / "target" / "release" / "ny"
    worker_release = tmp_path / "target" / "worker_3" / "release" / "ny"
    shared_release.parent.mkdir(parents=True, exist_ok=True)
    worker_release.parent.mkdir(parents=True, exist_ok=True)
    shared_release.write_text("#!/bin/sh\necho ny 0.1.0\n")
    shared_release.chmod(0o755)
    worker_release.write_text("#!/bin/sh\necho ny 0.1.0-worker\n")
    worker_release.chmod(0o755)

    original_repo_root = bounded.REPO_ROOT
    try:
        bounded.REPO_ROOT = tmp_path
        old_env = os.environ.get("AI_WORKER_ID")
        os.environ["AI_WORKER_ID"] = "3"
        try:
            path, source = bounded._resolve_ny_binary(None)
        finally:
            if old_env is None:
                os.environ.pop("AI_WORKER_ID", None)
            else:
                os.environ["AI_WORKER_ID"] = old_env
    finally:
        bounded.REPO_ROOT = original_repo_root

    assert source == "shared-default", (
        f"Expected shared-default source when both exist, got {source!r}"
    )
    assert "worker_3" not in str(path), (
        f"Expected shared binary path, got worker-local: {path}"
    )


def test_explicit_relative_ny_binary_is_resolved_once_against_caller_cwd(
    monkeypatch, tmp_path: Path
) -> None:
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    binary = caller_dir / "bin" / "ny"
    binary.parent.mkdir()
    binary.write_bytes(b"caller-binary")
    monkeypatch.chdir(caller_dir)

    resolved, source = bounded._resolve_ny_binary("bin/ny")

    assert source == "explicit"
    assert resolved == binary
    assert resolved.is_absolute()
    monkeypatch.chdir(tmp_path)
    assert resolved.read_bytes() == b"caller-binary"


def test_private_binary_copy_binds_execution_after_selected_path_replacement(
    monkeypatch, tmp_path: Path
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    original_bytes = ny_path.read_bytes()
    solver_binary_paths: list[Path] = []

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        staged_binary = Path(command[0])
        solver_binary_paths.append(staged_binary)
        assert staged_binary != ny_path
        assert staged_binary.read_bytes() == original_bytes
        ny_path.write_bytes(b"replacement-after-provenance")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","effective_config":{"schema":"fixture_v1"}}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    assert _run_main(monkeypatch, tmp_path, ny_path) == 0
    assert len(solver_binary_paths) == 1
    report = next((tmp_path / "reports" / "benchmarks").glob("fixturecat_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["ny_binary"] == str(ny_path)
    assert row["ny_sha256"] == hashlib.sha256(original_bytes).hexdigest()


def test_private_binary_mutation_fails_before_evidence_is_written(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        staged_binary = Path(command[0])
        staged_binary.chmod(0o700)
        staged_binary.write_bytes(b"mutated-private-binary")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","effective_config":{"schema":"fixture_v1"}}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    assert _run_main(monkeypatch, tmp_path, ny_path) == 1
    reports_dir = tmp_path / "reports" / "benchmarks"
    assert not list(reports_dir.glob("fixturecat_*.csv"))
    assert "refusing to emit stale-provenance evidence" in caplog.text


@pytest.mark.parametrize("mutated_input", ["model", "property", "preset"])
def test_input_or_preset_mutation_fails_before_evidence_is_written(
    monkeypatch, tmp_path: Path, caplog, mutated_input: str
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    model_path, property_path, preset_path = _write_inputs(tmp_path)
    paths = {
        "model": model_path,
        "property": property_path,
        "preset": preset_path,
    }

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        paths[mutated_input].write_bytes(b"mutated-during-execution")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","effective_config":{"schema":"fixture_v1"}}\n',
            stderr="",
        )

    binding = json.dumps(
        {
            "corpus_id": "mutation-fixture",
            "source_index_zero_based": 0,
            "model": "model.onnx",
            "property": "prop.vnnlib",
            "timeout_seconds": 2,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "property_sha256": hashlib.sha256(
                property_path.read_bytes()
            ).hexdigest(),
        }
    )
    output = tmp_path / "result.csv"
    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        bounded,
        "get_benchmark_instances",
        lambda year, category, **kwargs: [(model_path, property_path, 2)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--category",
            "fixturecat",
            "--preset",
            str(preset_path),
            "--ny-binary",
            str(ny_path),
            "--expected-row-binding",
            binding,
            "--output",
            str(output),
        ],
    )

    assert bounded.main() == 1
    assert not output.exists()
    assert "refusing to emit evidence" in caplog.text


def _run_main_with_argv(monkeypatch, tmp_path: Path, ny_path: Path, extra_argv: list[str]) -> int:
    """Helper: run main() with custom argv extensions."""
    model_path, property_path, preset_path = _write_inputs(tmp_path)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports" / "benchmarks")
    monkeypatch.setattr(bounded, "NY_PREFLIGHT_TIMEOUT_SECS", 2.0)
    monkeypatch.setattr(
        bounded,
        "get_benchmark_instances",
        lambda year, category, **kwargs: [(model_path, property_path, 2)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--category",
            "fixturecat",
            "--preset",
            str(preset_path),
            "--ny-binary",
            str(ny_path),
        ]
        + extra_argv,
    )
    return bounded.main()


def test_warmup_runs_executes_untimed_warmups_before_measured(
    monkeypatch, tmp_path: Path,
) -> None:
    """With --warmup-runs=1, the runner should execute one warmup + one measured (#4412).

    Warmup elapsed time is excluded from the measurement, but both attempts
    must have the same finite external watchdog.
    """
    ny_path = _write_placeholder_ny(tmp_path)
    call_count = 0
    observed_timeouts: list[int | None] = []

    def fake_run(command, **kwargs):
        nonlocal call_count
        if "--version" in command:
            return subprocess.CompletedProcess(
                args=command, returncode=0,
                stdout="ny 0.0.0\n", stderr="",
            )
        call_count += 1
        observed_timeouts.append(kwargs.get("timeout"))
        return subprocess.CompletedProcess(
            args=command, returncode=0,
            stdout='{"status":"timeout","domains_explored":10,"domains_verified":0,"max_depth_reached":2}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    exit_code = _run_main_with_argv(monkeypatch, tmp_path, ny_path, ["--warmup-runs", "1"])

    assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
    assert call_count == 2, f"Expected 1 warmup + 1 measured = 2 beta-crown calls, got {call_count}"
    assert observed_timeouts == [7, 7], (
        f"Expected finite timeout+slack=2+5 for both attempts, got {observed_timeouts}"
    )
    report = next((tmp_path / "reports" / "benchmarks").glob("fixturecat_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["notes"] == "warmup_runs=1", f"Expected warmup provenance in notes, got {rows[0]['notes']!r}"


def test_rerun_presearch_retries_when_first_attempt_presearch(
    monkeypatch, tmp_path: Path,
) -> None:
    """With --rerun-presearch=1, a pre-search first attempt triggers a retry (#4412)."""
    ny_path = _write_placeholder_ny(tmp_path)
    attempt = 0

    def fake_run(command, **kwargs):
        nonlocal attempt
        if "--version" in command:
            return subprocess.CompletedProcess(
                args=command, returncode=0,
                stdout="ny 0.0.0\n", stderr="",
            )
        attempt += 1
        if attempt == 1:
            return subprocess.CompletedProcess(
                args=command, returncode=1,
                stdout="", stderr="Deadline exceeded: forward-linear deadline",
            )
        return subprocess.CompletedProcess(
            args=command, returncode=0,
            stdout='{"status":"timeout","domains_explored":15,"domains_verified":0,"max_depth_reached":3}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    exit_code = _run_main_with_argv(monkeypatch, tmp_path, ny_path, ["--rerun-presearch", "1"])

    assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
    assert attempt == 2, f"Expected 2 measured attempts (initial + 1 retry), got {attempt}"
    report = next((tmp_path / "reports" / "benchmarks").glob("fixturecat_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["result"] == "timeout", f"Expected final result from retry, got {rows[0]['result']!r}"
    assert rows[0]["domains"] == "15", f"Expected 15 domains from retry, got {rows[0]['domains']!r}"
    notes = rows[0]["notes"]
    assert "presearch_retry=1" in notes, f"Expected retry provenance in notes, got {notes!r}"
    assert "initial_result=error" in notes, f"Expected initial result in notes, got {notes!r}"
    assert "initial_domains=0" in notes, f"Expected initial domains in notes, got {notes!r}"


def test_legacy_path_no_warmup_no_rerun_matches_original(
    monkeypatch, tmp_path: Path,
) -> None:
    """With defaults (0/0), behavior matches original single-attempt flow (#4412)."""
    ny_path = _write_placeholder_ny(tmp_path)
    call_count = 0

    def fake_run(command, **kwargs):
        nonlocal call_count
        if "--version" in command:
            return subprocess.CompletedProcess(
                args=command, returncode=0,
                stdout="ny 0.0.0\n", stderr="",
            )
        call_count += 1
        return subprocess.CompletedProcess(
            args=command, returncode=0,
            stdout='{"status":"verified","domains_explored":7,"domains_verified":3,"max_depth_reached":1}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    exit_code = _run_main_with_argv(monkeypatch, tmp_path, ny_path, [])

    assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
    assert call_count == 1, f"Expected exactly 1 beta-crown call with no warmup/rerun, got {call_count}"
    report = next((tmp_path / "reports" / "benchmarks").glob("fixturecat_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["notes"] == "", f"Expected empty notes in legacy path, got {rows[0]['notes']!r}"


def test_raw_artifacts_retain_each_attempt_exactly(
    monkeypatch, tmp_path: Path,
) -> None:
    ny_path = _write_placeholder_ny(tmp_path)
    attempt = 0

    def fake_run(command, **kwargs):
        nonlocal attempt
        if "--version" in command:
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ny 0.0.0\n", stderr=""
            )
        attempt += 1
        if attempt == 1:
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="first stdout\n",
                stderr="first stderr\n",
            )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","domains_explored":1}\n',
            stderr="second stderr\n",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    artifact_root = tmp_path / "raw"
    exit_code = _run_main_with_argv(
        monkeypatch,
        tmp_path,
        ny_path,
        ["--rerun-presearch", "1", "--raw-artifact-dir", str(artifact_root)],
    )
    assert exit_code == 0
    row = artifact_root / "fixturecat_idx0000"
    first = row / "measured-01"
    second = row / "measured-02"
    assert (first / "stdout.log").read_text(encoding="utf-8") == "first stdout\n"
    assert (first / "stderr.log").read_text(encoding="utf-8") == "first stderr\n"
    assert (first / "result.txt").read_text(encoding="utf-8") == "error\n"
    assert (second / "result.txt").read_text(encoding="utf-8") == "verified\n"
    command = __import__("json").loads(
        (second / "command.json").read_text(encoding="utf-8")
    )
    assert command["returncode"] == 0
    assert command["external_timeout_seconds"] == 7


def test_rerun_presearch_sidecar_belongs_to_final_attempt(
    monkeypatch, tmp_path: Path,
) -> None:
    """Domain-batch sidecar should reflect the final measured attempt, not the discarded one (#4412)."""
    ny_path = _write_placeholder_ny(tmp_path)
    metrics_dir = tmp_path / "domain_batch_metrics"
    attempt = 0

    def fake_run(command, **kwargs):
        nonlocal attempt
        if "--version" in command:
            return subprocess.CompletedProcess(
                args=command, returncode=0,
                stdout="ny 0.0.0\n", stderr="",
            )
        attempt += 1
        metrics_path = None
        for idx, val in enumerate(command):
            if val == "--domain-batch-metrics-jsonl":
                metrics_path = Path(command[idx + 1])
                break
        if attempt == 1:
            if metrics_path:
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text('{"attempt":1,"presearch":true}\n', encoding="utf-8")
            return subprocess.CompletedProcess(
                args=command, returncode=1,
                stdout="", stderr="Deadline exceeded: forward-linear",
            )
        if metrics_path:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text('{"attempt":2,"domains":15}\n', encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command, returncode=0,
            stdout='{"status":"timeout","domains_explored":15,"domains_verified":0,"max_depth_reached":3}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)
    model_path, property_path, preset_path = _write_inputs(tmp_path)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports" / "benchmarks")
    monkeypatch.setattr(bounded, "NY_PREFLIGHT_TIMEOUT_SECS", 2.0)
    monkeypatch.setattr(
        bounded,
        "get_benchmark_instances",
        lambda year, category, **kwargs: [(model_path, property_path, 2)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--category",
            "fixturecat",
            "--preset",
            str(preset_path),
            "--ny-binary",
            str(ny_path),
            "--domain-batch-metrics-dir",
            str(metrics_dir),
            "--rerun-presearch",
            "1",
        ],
    )
    exit_code = bounded.main()

    assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
    sidecar = metrics_dir / "fixturecat_idx0000.jsonl"
    assert sidecar.exists(), "Sidecar should exist for the final accepted attempt"
    content = sidecar.read_text(encoding="utf-8")
    assert '"attempt":2' in content, (
        f"Sidecar should contain final attempt data, got {content!r}"
    )
    assert '"attempt":1' not in content, (
        f"Sidecar should NOT contain discarded attempt data, got {content!r}"
    )
