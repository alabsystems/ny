# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the CIFAR100 attack-axis qualification campaign."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/cifar100_attack_fast_kernels_qualify.py"
SPEC = importlib.util.spec_from_file_location("cifar_fast_qualify", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
qualify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualify
SPEC.loader.exec_module(qualify)

TEST_BACKEND_SUMMARY = "cpu-only [test]"
TEST_HOST = {
    "hostname": "qualification-test-host",
    "cpu_model": "test-cpu",
    "logical_cores": 8,
    "ram_bytes": 16 << 30,
}
TEST_REGIME_SHA256 = qualify.canonical_json_sha256(
    {
        "backend_kind": "cpu-only",
        "backend_summary": TEST_BACKEND_SUMMARY,
        "host": TEST_HOST,
    }
)


def _v3_lever_state() -> dict[str, object]:
    levers = [
        {
            "name": "NY_ALPHA_ZERO_YIELD_FRAC",
            "value": 0.25,
            "source": "config",
            "bucket": "debug",
            "moat": "high",
            "provenance": "measured",
        },
        {
            "name": "NY_PHASE_TELEMETRY",
            "value": True,
            "source": "legacy_env",
            "bucket": "debug",
            "moat": "low",
            "provenance": "unmeasured",
            "env_utf8": True,
        },
    ]
    return {
        "status": "resolved",
        "receipt": {
            "schema": "ny-levers/receipt/v2",
            "lever_count": len(levers),
            "env_present": 1,
            "env_accepted": 1,
            "env_rejected": 0,
            "levers": levers,
        },
    }


def _targets() -> list[qualify.Target]:
    targets = [
        qualify.Target(
            "gain",
            index,
            "CIFAR100_resnet_large.onnx",
            f"gain-{index}.vnnlib",
            "sat",
        )
        for index in range(1, qualify.EXPECTED_GAIN_ROWS + 1)
    ]
    targets.extend(
        qualify.Target(
            "guard",
            index,
            "CIFAR100_resnet_large.onnx",
            f"guard-{index}.vnnlib",
            "unsat",
        )
        for index in range(1, qualify.EXPECTED_GUARD_ROWS + 1)
    )
    return targets


def _row(
    target: qualify.Target,
    order_index: int,
    arm: str,
    status: str,
    *,
    authenticated: bool = True,
    trusted: bool = False,
    exit_code: int = 0,
    backend: str = "cpu-only",
    steps: int | None = None,
    regime: str = TEST_REGIME_SHA256,
    within_cutoff: bool = True,
    load_acceptable: bool = True,
) -> dict[str, str]:
    return {
        "cohort": target.cohort,
        "cohort_index": str(target.cohort_index),
        "order_index": str(order_index),
        "arm": arm,
        "onnx": target.onnx,
        "vnnlib": target.vnnlib,
        "ground_truth": target.ground_truth,
        "status": status,
        "wall_secs": "1.000000",
        "exit_code": str(exit_code),
        "attack_steps": "" if steps is None else str(steps),
        "arm_authenticated": str(authenticated).lower(),
        "trusted_upfront_sat": str(trusted).lower(),
        "backend_kind": backend,
        "regime_sha256": regime,
        "flight_publish_secs": "0.500000000",
        "flight_terminal_secs": "0.750000000",
        "within_official_cutoff": str(within_cutoff).lower(),
        "peak_load_per_core": "0.125000000",
        "load_acceptable": str(load_acceptable).lower(),
        "result_sha256": "1" * 64,
        "flight_sha256": "2" * 64,
        "log_sha256": "3" * 64,
    }


def _passing_rows() -> tuple[list[qualify.Target], list[dict[str, str]]]:
    targets = _targets()
    rows: list[dict[str, str]] = []
    for order_index, target in enumerate(targets, 1):
        if target.cohort == "gain":
            treatment_gains = target.cohort_index <= qualify.MINIMUM_PROMOTION_GAINS
            rows.append(_row(target, order_index, "off", "timeout"))
            rows.append(
                _row(
                    target,
                    order_index,
                    "on",
                    "sat" if treatment_gains else "timeout",
                    trusted=treatment_gains,
                    steps=1 if treatment_gains else None,
                )
            )
        else:
            rows.append(_row(target, order_index, "off", "unsat"))
            rows.append(_row(target, order_index, "on", "unsat"))
    return targets, rows


def _flight_payload(
    arm: str,
    status: str,
    *,
    axis: str = qualify.DEFAULT_AXIS,
    trusted: bool = False,
    terminal_secs: float = 0.75,
    host: dict[str, object] | None = None,
) -> dict[str, object]:
    events: list[dict[str, object]] = [
        {
            "method": "upfront_attack",
            "status": "ran",
            "reason": (
                "sat: trusted-oracle gate confirmed the upfront candidate"
                if trusted
                else "consulted; no confirmed candidate"
            ),
            "at_secs": 0.25,
        }
    ]
    events.extend(
        [
            {
                "method": "result_publish",
                "status": "ran",
                "reason": status,
                "at_secs": min(0.5, terminal_secs),
            },
            {
                "method": "run_complete",
                "status": "complete",
                "reason": status,
                "at_secs": terminal_secs,
            },
        ]
    )
    return {
        "schema_version": qualify.FLIGHT_SCHEMA_VERSION,
        "backend_kind": "cpu-only",
        "backend_summary": TEST_BACKEND_SUMMARY,
        "host": TEST_HOST if host is None else host,
        "load_avg_at_begin": [1.0, 1.0, 1.0],
        "load_avg_at_end": [1.0, 1.0, 1.0],
        "category": qualify.CATEGORY,
        "budget_secs": qualify.OFFICIAL_BUDGET_SECS,
        "ambient_env": qualify.forced_arm_environment(axis, arm),
        "levers": _v3_lever_state(),
        "events": events,
    }


def _write_arm_artifacts(
    output: Path,
    target: qualify.Target,
    order_index: int,
    arm: str,
    status: str,
    *,
    trusted: bool = False,
    exit_code: int = 0,
    steps: int | None = None,
) -> dict[str, str]:
    result_path, flight_path, log_path = qualify.artifact_paths(output, target, arm)
    result_body = f"{status}\n"
    if status == "sat":
        result_body += "(X_0 0.0)\n"
    result_path.write_text(result_body, encoding="utf-8")
    flight_path.write_text(
        json.dumps(_flight_payload(arm, status, trusted=trusted)) + "\n",
        encoding="utf-8",
    )
    log_body = "qualification fixture\n"
    if steps is not None:
        log_body += f"budget exhausted ({steps} gradient steps, 2 ORT evals)\n"
    log_path.write_text(log_body, encoding="utf-8")
    row = _row(
        target,
        order_index,
        arm,
        status,
        trusted=trusted,
        exit_code=exit_code,
        steps=steps,
    )
    evidence = qualify.flight_evidence(flight_path, arm, status)
    row.update(
        {
            "backend_kind": str(evidence["backend"]),
            "regime_sha256": str(evidence["regime_sha256"]),
            "flight_publish_secs": f"{float(evidence['publish_secs']):.9f}",
            "flight_terminal_secs": f"{float(evidence['terminal_secs']):.9f}",
            "within_official_cutoff": str(
                bool(evidence["within_official_cutoff"])
            ).lower(),
            "peak_load_per_core": f"{float(evidence['peak_load_per_core']):.9f}",
            "load_acceptable": str(bool(evidence["load_acceptable"])).lower(),
        }
    )
    row["result_sha256"] = qualify.sha256_file(result_path)
    row["flight_sha256"] = qualify.sha256_file(flight_path)
    row["log_sha256"] = qualify.sha256_file(log_path)
    return row


def test_alternating_arm_order_is_deterministic() -> None:
    assert qualify.arm_order(1) == ("off", "on")
    assert qualify.arm_order(2) == ("on", "off")
    assert qualify.arm_order(3) == ("off", "on")


def test_wrapper_vjp_axis_changes_only_exact_kill_switch_and_pins_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    off = qualify.forced_arm_environment(qualify.WRAPPER_VJP_AXIS, "off")
    on = qualify.forced_arm_environment(qualify.WRAPPER_VJP_AXIS, "on")
    assert {key for key in off if off[key] != on[key]} == {"NY_ORT_REFINE_VJP_BATCH"}
    assert off["NY_ORT_REFINE_VJP_BATCH"] == "0"
    assert on["NY_ORT_REFINE_VJP_BATCH"] == "1"
    assert off["NY_ORT_REFINE_VJP_K"] == str(qualify.WRAPPER_VJP_WIDTH)
    assert on["NY_ATTACK_POINT_FAST_KERNELS"] == "0"
    assert on["NY_ORT_REFINE_VJP_UNDER_MEMORY_LIMIT"] == "1"
    assert qualify.forced_arm_environment(qualify.FAST_KERNELS_AXIS, "on") == {
        "NY_ATTACK_POINT_FAST_KERNELS": "1",
        "NY_PHASE_TELEMETRY": "1",
        "OMP_NUM_THREADS": "1",
    }

    monkeypatch.setenv("NY_ORT_REFINE_VJP_BATCH", "ambient-garbage")
    monkeypatch.setenv("NY_ORT_REFINE_VJP_K", "999")
    monkeypatch.setenv("NY_ATTACK_POINT_FAST_KERNELS", "1")
    child = qualify.scrubbed_environment("off", qualify.WRAPPER_VJP_AXIS)
    assert {
        key: child[key]
        for key in qualify.forced_arm_environment(qualify.WRAPPER_VJP_AXIS, "off")
    } == off
    contract = qualify.compute_environment_contract(qualify.WRAPPER_VJP_AXIS)
    assert contract["experiment_axis"] == qualify.WRAPPER_VJP_AXIS
    assert contract["forced_by_arm"] == {"off": off, "on": on}


def test_wrapper_vjp_axis_authenticates_flight_and_terminal_log(
    tmp_path: Path,
) -> None:
    path = tmp_path / "flight.json"
    payload = _flight_payload("on", "timeout", axis=qualify.WRAPPER_VJP_AXIS)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        qualify.flight_evidence(path, "on", "timeout", qualify.WRAPPER_VJP_AXIS)[
            "authenticated"
        ]
        is True
    )

    payload["ambient_env"]["NY_ORT_REFINE_VJP_K"] = "63"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        qualify.flight_evidence(path, "on", "timeout", qualify.WRAPPER_VJP_AXIS)[
            "authenticated"
        ]
        is False
    )

    miss_log = (
        f"{qualify.WRAPPER_VJP_ARMED_MARKER} (K=64, hard cap 64, backend=cpu)\n"
        f"{qualify.WRAPPER_VJP_LOG_PREFIX} found no trusted violation "
        "(3 wave steps, 64 live lanes); continuing sequentially\n"
    )
    success_log = (
        f"{qualify.WRAPPER_VJP_ARMED_MARKER} (K=64, hard cap 64, backend=cpu)\n"
        f"{qualify.WRAPPER_VJP_SUCCESS_PREFIX} "
        "(lane 2, 4 wave steps, 256 restart-gradient steps, 8 ORT evals)\n"
    )
    assert qualify.wrapper_vjp_log_authenticated(miss_log, "on")
    assert qualify.parse_steps(miss_log, qualify.WRAPPER_VJP_AXIS) == 3
    assert qualify.wrapper_vjp_log_authenticated(success_log, "on")
    assert qualify.parse_steps(success_log, qualify.WRAPPER_VJP_AXIS) == 4
    assert not qualify.wrapper_vjp_log_authenticated(
        qualify.WRAPPER_VJP_ARMED_MARKER, "on"
    )
    zero_step_log = (
        f"{qualify.WRAPPER_VJP_ARMED_MARKER} (K=64, hard cap 64, backend=cpu)\n"
        f"{qualify.WRAPPER_VJP_LOG_PREFIX} found no trusted violation "
        "(0 wave steps, 64 live lanes); continuing sequentially\n"
    )
    assert not qualify.wrapper_vjp_log_authenticated(zero_step_log, "on")
    assert qualify.parse_steps(zero_step_log, qualify.WRAPPER_VJP_AXIS) == 0
    assert qualify.wrapper_vjp_log_authenticated("ordinary verifier log", "off")
    assert not qualify.wrapper_vjp_log_authenticated(miss_log, "off")

    declined_log = (
        f"{qualify.WRAPPER_VJP_DECLINED_PREFIX} (reason=no_accelerator)\n"
    )
    assert qualify.wrapper_vjp_decline_reasons(declined_log) == (
        "no_accelerator",
    )
    assert not qualify.wrapper_vjp_log_authenticated(declined_log, "on")
    assert not qualify.wrapper_vjp_log_authenticated(declined_log, "off")
    assert not qualify.wrapper_vjp_log_authenticated(
        declined_log + miss_log,
        "on",
    )


def test_attempt_scope_name_is_bound_only_to_sealed_identity() -> None:
    target = _targets()[0]
    launch_digest = "a" * 64
    observed = qualify.attempt_scope_unit_name(target, 1, "off", launch_digest)
    assert observed.startswith("ny-cifar-fast-")
    assert observed.endswith(".scope")
    assert len(observed.removeprefix("ny-cifar-fast-").removesuffix(".scope")) == 64
    assert qualify.attempt_scope_unit_name(target, 1, "off", launch_digest) == observed
    assert qualify.attempt_scope_unit_name(target, 1, "on", launch_digest) != observed
    assert qualify.attempt_scope_unit_name(target, 1, "off", "b" * 64) != observed


def test_scope_query_reads_exact_bounded_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_command: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed_command.extend(command)
        assert kwargs["timeout"] == qualify.SYSTEMD_SCOPE_QUERY_TIMEOUT_SECS
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "LoadState=loaded\n"
                "Result=oom-kill\n"
                "OOMKills=1\n"
                "MemoryPeak=4294967296\n"
                "MemoryMax=4294967296\n"
                "MemorySwapPeak=0\n"
                "MemorySwapMax=0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(qualify.subprocess, "run", fake_run)
    diagnostics = qualify.query_systemd_scope(
        "/usr/bin/systemctl", "ny-cifar-fast-fixture.scope"
    )
    assert diagnostics == {
        "LoadState": "loaded",
        "Result": "oom-kill",
        "OOMKills": "1",
        "MemoryPeak": "4294967296",
        "MemoryMax": "4294967296",
        "MemorySwapPeak": "0",
        "MemorySwapMax": "0",
    }
    assert observed_command[:5] == [
        "/usr/bin/systemctl",
        "--user",
        "show",
        "ny-cifar-fast-fixture.scope",
        "--no-pager",
    ]
    assert qualify.format_scope_failure(diagnostics).startswith("cause=cgroup-oom ")


def test_scope_query_accepts_missing_optional_properties_on_older_systemd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def older_systemd(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout="LoadState=loaded\nResult=exit-code\n",
            stderr="",
        )

    monkeypatch.setattr(qualify.subprocess, "run", older_systemd)
    diagnostics = qualify.query_systemd_scope(
        "/usr/bin/systemctl", "ny-cifar-fast-fixture.scope"
    )
    assert diagnostics == {"LoadState": "loaded", "Result": "exit-code"}
    assert not qualify.scope_diagnostics_prove_cgroup_oom(diagnostics)
    assert "oom_kills='<unavailable>'" in qualify.format_scope_failure(diagnostics)


def test_scope_query_rejects_missing_required_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_result(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout="LoadState=loaded\nOOMKills=0\n",
            stderr="",
        )

    monkeypatch.setattr(qualify.subprocess, "run", missing_result)
    assert (
        qualify.query_systemd_scope("/usr/bin/systemctl", "ny-cifar-fast-fixture.scope")
        is None
    )


def test_scope_oom_kills_unset_sentinel_does_not_prove_oom() -> None:
    diagnostics = {
        "LoadState": "loaded",
        "Result": "failed",
        "OOMKills": str(qualify.SYSTEMD_U64_UNSET),
    }
    assert not qualify.scope_diagnostics_prove_cgroup_oom(diagnostics)
    assert qualify.format_scope_failure(diagnostics).startswith("cause=child-failure ")
    assert qualify.scope_diagnostics_prove_cgroup_oom(
        {"LoadState": "loaded", "Result": "oom-kill"}
    )


def test_reset_failed_scope_is_bounded_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def timeout(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed.extend(command)
        assert kwargs["check"] is False
        assert kwargs["stdout"] == qualify.subprocess.DEVNULL
        assert kwargs["stderr"] == qualify.subprocess.DEVNULL
        assert kwargs["timeout"] == qualify.SYSTEMD_SCOPE_RESET_TIMEOUT_SECS
        raise qualify.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(qualify.subprocess, "run", timeout)
    qualify.reset_failed_systemd_scope(
        "/usr/bin/systemctl", "ny-cifar-fast-fixture.scope"
    )
    assert observed == [
        "/usr/bin/systemctl",
        "--user",
        "reset-failed",
        "ny-cifar-fast-fixture.scope",
    ]


def test_scope_query_unavailable_stays_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise OSError("fixture systemctl unavailable")

    monkeypatch.setattr(qualify.subprocess, "run", unavailable)
    assert (
        qualify.query_systemd_scope("/missing/systemctl", "ny-cifar-fast-fixture.scope")
        is None
    )
    assert (
        qualify.format_scope_failure(None)
        == "cause=child-failure scope_diagnostics=unavailable"
    )


def test_scope_query_rejects_systemctl_missing_unit_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_unit(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "LoadState=not-found\n"
                "Result=\n"
                "OOMKills=18446744073709551615\n"
                "MemoryPeak=[not set]\n"
                "MemoryMax=infinity\n"
                "MemorySwapPeak=[not set]\n"
                "MemorySwapMax=infinity\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(qualify.subprocess, "run", missing_unit)
    assert (
        qualify.query_systemd_scope(
            "/usr/bin/systemctl", "ny-cifar-fast-garbage-collected.scope"
        )
        is None
    )


def test_promotion_requires_nine_authenticated_trusted_gains_and_all_guards() -> None:
    targets, rows = _passing_rows()
    summary = qualify.summarize(rows, targets)
    assert summary["sat_gains"] == qualify.MINIMUM_PROMOTION_GAINS
    assert summary["proof_guards_closed_both_arms"] == qualify.EXPECTED_GUARD_ROWS
    assert summary["exact_expected_keys"] is True
    assert summary["global_or_default_on_authorized"] is False
    assert summary["promotion_pass"] is True


def test_one_proof_loss_blocks_promotion() -> None:
    targets, rows = _passing_rows()
    lost = next(
        row
        for row in rows
        if row["cohort"] == "guard"
        and row["cohort_index"] == "1"
        and row["arm"] == "on"
    )
    lost["status"] = "timeout"
    summary = qualify.summarize(rows, targets)
    assert summary["proof_losses"] == 1
    assert summary["promotion_pass"] is False


def test_untrusted_gained_treatment_blocks_promotion() -> None:
    targets, rows = _passing_rows()
    gained = next(
        row
        for row in rows
        if row["cohort"] == "gain" and row["cohort_index"] == "1" and row["arm"] == "on"
    )
    gained["trusted_upfront_sat"] = "false"
    summary = qualify.summarize(rows, targets)
    assert summary["gained_sat_rows_without_trusted_upfront_event"] == 1
    assert summary["promotion_pass"] is False


def test_gain_without_positive_fast_kernel_steps_blocks_promotion() -> None:
    targets, rows = _passing_rows()
    gained = next(
        row
        for row in rows
        if row["cohort"] == "gain" and row["cohort_index"] == "1" and row["arm"] == "on"
    )
    gained["attack_steps"] = "0"
    summary = qualify.summarize(rows, targets)
    assert summary["gains_without_positive_fast_kernel_evidence"] == 1
    assert summary["sat_gains"] == qualify.MINIMUM_PROMOTION_GAINS - 1
    assert summary["promotion_pass"] is False


def test_late_or_overloaded_row_blocks_promotion() -> None:
    targets, rows = _passing_rows()
    rows[0]["wall_secs"] = "100.000001"
    rows[0]["within_official_cutoff"] = "false"
    summary = qualify.summarize(rows, targets)
    assert summary["late_rows"] == 1
    assert summary["promotion_pass"] is False

    rows[0]["wall_secs"] = "1.000000"
    rows[0]["within_official_cutoff"] = "true"
    rows[0]["peak_load_per_core"] = "1.250000000"
    rows[0]["load_acceptable"] = "false"
    summary = qualify.summarize(rows, targets)
    assert summary["overloaded_rows"] == 1
    assert summary["promotion_pass"] is False


def test_nonzero_child_exit_blocks_promotion() -> None:
    targets, rows = _passing_rows()
    rows[0]["exit_code"] = "124"
    summary = qualify.summarize(rows, targets)
    assert summary["nonzero_exit_rows"] == 1
    assert summary["promotion_pass"] is False


def test_backend_pair_mismatch_blocks_promotion() -> None:
    targets, rows = _passing_rows()
    rows[0]["backend_kind"] = "cuda"
    summary = qualify.summarize(rows, targets)
    assert summary["backend_pair_mismatches"] == 1
    assert summary["promotion_pass"] is False


def test_host_backend_regime_mismatch_blocks_promotion() -> None:
    targets, rows = _passing_rows()
    rows[0]["regime_sha256"] = "a" * 64
    summary = qualify.summarize(rows, targets)
    assert summary["regime_pair_mismatches"] == 1
    assert summary["campaign_regimes"] == 2
    assert summary["promotion_pass"] is False


def test_summary_rejects_unexpected_and_duplicate_keys() -> None:
    targets, rows = _passing_rows()
    unexpected = dict(rows[0])
    unexpected["cohort_index"] = "99"
    with pytest.raises(qualify.QualificationError, match="unexpected campaign row"):
        qualify.summarize([unexpected, *rows[1:]], targets)
    with pytest.raises(qualify.QualificationError, match="duplicate campaign row"):
        qualify.summarize([*rows, dict(rows[0])], targets)


def test_results_snapshot_is_atomically_rebuilt(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    qualify.initialize_results(path)
    _, passing = _passing_rows()
    first, second = passing[:2]
    qualify.publish_results_snapshot(path, [first])
    qualify.publish_results_snapshot(path, [first, second])
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    assert rows == [first, second]


def test_persisted_rows_reject_torn_duplicate_and_tampered_evidence(
    tmp_path: Path,
) -> None:
    target = _targets()[0]
    results = tmp_path / "results.tsv"
    qualify.initialize_results(results)
    row = _write_arm_artifacts(tmp_path, target, 1, "off", "timeout")
    qualify.publish_results_snapshot(results, [row])
    assert qualify.load_persisted_rows(results, [target], tmp_path) == [row]

    qualify.publish_results_snapshot(results, [row, row])
    with pytest.raises(qualify.QualificationError, match="duplicate campaign row"):
        qualify.load_persisted_rows(results, [target], tmp_path)

    results.write_bytes(results.read_bytes()[:-1])
    with pytest.raises(qualify.QualificationError, match="torn result evidence"):
        qualify.load_persisted_rows(results, [target], tmp_path)


def test_atomic_row_fragment_repairs_a_torn_derived_snapshot(tmp_path: Path) -> None:
    target = _targets()[0]
    targets = [target]
    launch = _launch_fixture(tmp_path, targets)
    args = qualify.args_from_launch(launch)
    qualify.preseal_attempt(target, 1, "off", args, tmp_path, launch)
    row = _write_arm_artifacts(tmp_path, target, 1, "off", "timeout")
    qualify.commit_row_fragment(tmp_path, target, 1, "off", row)
    results = tmp_path / "results.tsv"
    results.write_bytes(b"torn")

    rows, incomplete = qualify.load_attempt_state(tmp_path, targets, launch)
    assert incomplete == []
    assert rows == [row]
    qualify.synchronize_results_snapshot(results, rows, allow_repair=True)
    assert qualify.load_persisted_rows(results, targets, tmp_path) == [row]


def test_persisted_rows_reject_artifact_hash_drift(tmp_path: Path) -> None:
    target = _targets()[0]
    results = tmp_path / "results.tsv"
    qualify.initialize_results(results)
    row = _write_arm_artifacts(tmp_path, target, 1, "off", "timeout")
    qualify.publish_results_snapshot(results, [row])
    _, _, log_path = qualify.artifact_paths(tmp_path, target, "off")
    log_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(qualify.QualificationError, match="artifact hash mismatch"):
        qualify.load_persisted_rows(results, [target], tmp_path)


def test_persisted_sat_requires_a_witness(tmp_path: Path) -> None:
    target = _targets()[0]
    results = tmp_path / "results.tsv"
    qualify.initialize_results(results)
    row = _write_arm_artifacts(tmp_path, target, 1, "on", "sat", trusted=True)
    result_path, _, _ = qualify.artifact_paths(tmp_path, target, "on")
    result_path.write_text("sat\n", encoding="utf-8")
    row["result_sha256"] = qualify.sha256_file(result_path)
    qualify.publish_results_snapshot(results, [row])
    with pytest.raises(qualify.QualificationError, match="no counterexample witness"):
        qualify.load_persisted_rows(results, [target], tmp_path)


def test_flight_authentication_binds_schema_category_budget_and_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "flight.json"
    payload = _flight_payload("on", "sat", trusted=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    evidence = qualify.flight_evidence(path, "on", "sat")
    assert evidence["authenticated"] is True
    assert evidence["trusted_sat"] is True
    assert evidence["backend"] == "cpu-only"
    assert evidence["regime_sha256"] == TEST_REGIME_SHA256
    assert evidence["publish_secs"] == 0.5
    assert evidence["terminal_secs"] == 0.75
    assert evidence["within_official_cutoff"] is True
    assert evidence["load_acceptable"] is True

    for field, bad_value in (
        ("schema_version", 1),
        ("category", "wrong"),
        ("budget_secs", 99),
    ):
        bad = dict(payload)
        bad[field] = bad_value
        path.write_text(json.dumps(bad), encoding="utf-8")
        assert qualify.flight_evidence(path, "on", "sat")["authenticated"] is False

    bad = dict(payload)
    bad["events"] = list(reversed(payload["events"]))
    path.write_text(json.dumps(bad), encoding="utf-8")
    assert qualify.flight_evidence(path, "on", "sat")["authenticated"] is False


def test_flight_authentication_accepts_legacy_v2_without_lever_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "flight.json"
    payload = _flight_payload("on", "sat", trusted=True)
    payload["schema_version"] = 2
    del payload["levers"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert qualify.flight_evidence(path, "on", "sat")["authenticated"] is True


@pytest.mark.parametrize(
    "malformation",
    (
        "not_materialized",
        "invalid_config",
        "wrong_receipt_schema",
        "count_type",
        "duplicate_name",
        "invalid_source",
        "invalid_source_type",
        "inconsistent_env_counts",
        "invalid_env_utf8",
        "empty_registry",
        "extra_entry_field",
        "env_source_missing_ambient",
        "config_with_env_evidence",
        "invalid_name",
        "unmeasured_default_on",
        "guard_auto",
    ),
)
def test_flight_authentication_rejects_malformed_v3_lever_state(
    tmp_path: Path, malformation: str
) -> None:
    path = tmp_path / "flight.json"
    payload = _flight_payload("on", "sat", trusted=True)
    receipt = payload["levers"]["receipt"]
    if malformation == "not_materialized":
        payload["levers"] = {"status": "not_materialized"}
    elif malformation == "invalid_config":
        payload["levers"] = {
            "status": "invalid_config",
            "reason": "typed preset was invalid",
        }
    elif malformation == "wrong_receipt_schema":
        receipt["schema"] = "ny-levers/receipt/v1"
    elif malformation == "count_type":
        receipt["env_present"] = True
    elif malformation == "duplicate_name":
        receipt["levers"][1]["name"] = receipt["levers"][0]["name"]
    elif malformation == "invalid_source":
        receipt["levers"][1]["source"] = "env"
    elif malformation == "invalid_source_type":
        receipt["levers"][1]["source"] = []
    elif malformation == "inconsistent_env_counts":
        receipt["env_accepted"] = 0
    elif malformation == "invalid_env_utf8":
        receipt["levers"][1]["env_utf8"] = "yes"
    elif malformation == "empty_registry":
        receipt["levers"] = []
        receipt["lever_count"] = 0
    elif malformation == "extra_entry_field":
        receipt["levers"][0]["unexpected"] = "not in receipt/v2"
    elif malformation == "env_source_missing_ambient":
        del payload["ambient_env"]["NY_PHASE_TELEMETRY"]
    elif malformation == "config_with_env_evidence":
        receipt["levers"][0]["env_utf8"] = True
    elif malformation == "invalid_name":
        receipt["levers"][0]["name"] = "NY_lowercase"
    elif malformation == "unmeasured_default_on":
        receipt["levers"][1]["bucket"] = "default_on"
    elif malformation == "guard_auto":
        receipt["levers"][0]["provenance"] = "guard"
        receipt["levers"][0]["bucket"] = "auto"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert qualify.flight_evidence(path, "on", "sat")["authenticated"] is False


def test_trusted_sat_requires_exact_gate_event_before_publish(tmp_path: Path) -> None:
    path = tmp_path / "flight.json"
    payload = _flight_payload("on", "sat", trusted=True)
    payload["events"][0]["reason"] = "trusted-oracle gate confirmed-ish"
    path.write_text(json.dumps(payload), encoding="utf-8")
    evidence = qualify.flight_evidence(path, "on", "sat")
    assert evidence["authenticated"] is True
    assert evidence["trusted_sat"] is False


def test_flight_requires_real_v3_host_load_and_official_timing(tmp_path: Path) -> None:
    path = tmp_path / "flight.json"
    payload = _flight_payload("on", "sat", trusted=True)
    del payload["host"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert qualify.flight_evidence(path, "on", "sat")["authenticated"] is False

    on_budget = _flight_payload(
        "on",
        "sat",
        trusted=True,
        terminal_secs=float(qualify.OFFICIAL_BUDGET_SECS),
    )
    path.write_text(json.dumps(on_budget), encoding="utf-8")
    assert qualify.flight_evidence(path, "on", "sat")["within_official_cutoff"] is True

    late = _flight_payload(
        "on",
        "sat",
        trusted=True,
        terminal_secs=qualify.OFFICIAL_BUDGET_SECS + 0.000001,
    )
    path.write_text(json.dumps(late), encoding="utf-8")
    evidence = qualify.flight_evidence(path, "on", "sat")
    assert evidence["authenticated"] is True
    assert evidence["within_official_cutoff"] is False

    malformed = _flight_payload("on", "sat", trusted=True)
    malformed["events"].insert(1, "not-an-event")
    path.write_text(json.dumps(malformed), encoding="utf-8")
    assert qualify.flight_evidence(path, "on", "sat")["authenticated"] is False


def test_compute_environment_scrubs_tuning_and_seals_device_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)
    baseline = qualify.compute_environment_contract()
    monkeypatch.setenv("MIMALLOC_PURGE_DELAY", "999")
    monkeypatch.setenv("RAYON_NUM_THREADS", "64")
    monkeypatch.setenv("SHLVL", "99")
    monkeypatch.setenv("_", "/different/shell/launcher")
    scrubbed = qualify.compute_environment_contract()
    assert (
        scrubbed["effective_child_environment_sha256"]
        == baseline["effective_child_environment_sha256"]
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    environment = qualify.scrubbed_environment("on")
    assert "MIMALLOC_PURGE_DELAY" not in environment
    assert "RAYON_NUM_THREADS" not in environment
    assert "SHLVL" not in environment
    assert "_" not in environment
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    contract = qualify.compute_environment_contract()
    assert contract["sealed_passthrough"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert (
        contract["effective_child_environment_sha256"]
        != baseline["effective_child_environment_sha256"]
    )

    monkeypatch.setenv("GLIBC_TUNABLES", "glibc.cpu.hwcaps=-AVX2")
    drifted = qualify.compute_environment_contract()
    assert (
        drifted["effective_child_environment_sha256"]
        != contract["effective_child_environment_sha256"]
    )


def test_target_evidence_round_trip_and_torn_rejection(tmp_path: Path) -> None:
    targets = _targets()
    path = tmp_path / "targets.csv"
    qualify.atomic_write_bytes(path, qualify.render_targets(targets))
    observed, raw = qualify.load_targets_file(path)
    assert observed == targets
    assert raw == qualify.render_targets(targets)
    path.write_bytes(raw[:-1])
    with pytest.raises(qualify.QualificationError, match="torn target evidence"):
        qualify.load_targets_file(path)


def test_canonical_observation_rejects_revision_and_asset_drift() -> None:
    observed = {
        "official_git_head": "1" * 40,
        "official_results": {"tool": {"size": 1, "sha256": "2" * 64}},
        "targets_csv_sha256": "3" * 64,
        "benchmark": {
            "git_head": "4" * 40,
            "instances": {"size": 2, "sha256": "5" * 64},
            "models": {
                "model.onnx": {
                    "asset": "model.onnx",
                    "size": 3,
                    "sha256": "6" * 64,
                }
            },
            "vnnlibs": {
                "property.vnnlib": {
                    "asset": "property.vnnlib",
                    "size": 4,
                    "sha256": "7" * 64,
                }
            },
        },
    }
    manifest = {
        "official_git_head": observed["official_git_head"],
        "benchmark_git_head": observed["benchmark"]["git_head"],
        "targets_csv_sha256": observed["targets_csv_sha256"],
        "official_results_manifest_sha256": qualify.canonical_json_sha256(
            observed["official_results"]
        ),
        "benchmark_manifest_sha256": qualify.canonical_json_sha256(
            observed["benchmark"]
        ),
        "canonical_selected_inputs_sha256": qualify.canonical_json_sha256(observed),
        "selected_vnnlib_count": 1,
        "selected_model": {
            "logical_name": "model.onnx",
            **observed["benchmark"]["models"]["model.onnx"],
        },
    }
    qualify.validate_canonical_observation(manifest, observed)
    drifted = json.loads(json.dumps(observed))
    drifted["official_git_head"] = "8" * 40
    with pytest.raises(qualify.QualificationError, match="official Git revision"):
        qualify.validate_canonical_observation(manifest, drifted)
    compressed = json.loads(json.dumps(observed))
    compressed["benchmark"]["vnnlibs"]["property.vnnlib"]["asset"] = (
        "property.vnnlib.gz"
    )
    with pytest.raises(qualify.QualificationError, match="selected VNNLIB assets"):
        qualify.validate_canonical_observation(manifest, compressed)


def test_canonical_manifest_requires_plain_model_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = json.loads(qualify.CANONICAL_INPUTS_PATH.read_text(encoding="utf-8"))
    manifest["selected_model"]["asset"] += ".gz"
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(qualify, "CANONICAL_INPUTS_PATH", path)
    with pytest.raises(qualify.QualificationError, match="selected-model names"):
        qualify.load_canonical_inputs_manifest()


def test_main_dry_run_validates_canonical_inputs_before_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _targets()[0]
    binary = tmp_path / "ny"
    binary.write_bytes(b"fixture")
    binary.chmod(0o755)
    args = SimpleNamespace(
        official_root=tmp_path,
        benchmark_root=tmp_path,
        binary=binary,
        out=tmp_path / "campaign",
        memory_max="4G",
        limit=1,
        dry_run=True,
        resume=False,
        allow_dirty=False,
        allow_unreceipted=False,
        systemd_run="/usr/bin/systemd-run",
        systemctl="/usr/bin/systemctl",
        timeout="/usr/bin/timeout",
    )
    monkeypatch.setattr(qualify, "parse_args", lambda: args)
    monkeypatch.setattr(
        qualify, "derive_selected_targets", lambda _args: ([target], ("fixture",))
    )
    monkeypatch.setattr(qualify, "repo_is_dirty", lambda: False)
    calls: list[str] = []
    immutable_inputs = {"schema": qualify.INPUTS_SCHEMA, "fixture": "sealed"}
    expected_digest = qualify.canonical_json_sha256(immutable_inputs)

    def validate_receipt(_args: object) -> bool:
        calls.append("receipt")
        return True

    def capture_inputs(
        _args: object,
        selected: list[qualify.Target],
        target_csv_sha256: str,
        receipt_valid: bool,
    ) -> dict[str, object]:
        calls.append("capture")
        assert selected == [target]
        assert target_csv_sha256 == qualify.sha256_bytes(
            qualify.render_targets([target])
        )
        assert receipt_valid is True
        return immutable_inputs

    def print_command(
        _target: qualify.Target,
        _order_index: int,
        _arm: str,
        _args: object,
        _output: Path,
        launch_inputs_sha256: str,
    ) -> None:
        calls.append("command")
        assert launch_inputs_sha256 == expected_digest
        assert launch_inputs_sha256 != "0" * 64

    monkeypatch.setattr(qualify, "validate_binary_receipt", validate_receipt)
    monkeypatch.setattr(qualify, "capture_immutable_inputs", capture_inputs)
    monkeypatch.setattr(qualify, "run_one", print_command)
    assert qualify.main() == 0
    assert calls == ["receipt", "capture", "command", "command"]


def test_find_launch_tool_canonicalizes_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "real-timeout"
    executable.write_bytes(b"fixture")
    executable.chmod(0o755)
    alias = tmp_path / "timeout"
    alias.symlink_to(executable)
    monkeypatch.setattr(qualify.shutil, "which", lambda _name: str(alias))

    assert qualify.find_launch_tool("timeout") == str(executable)


def test_launch_manifest_rejects_digest_and_input_drift(tmp_path: Path) -> None:
    inputs = {"schema": qualify.INPUTS_SCHEMA, "value": 1}
    path = tmp_path / "launch.json"
    qualify.atomic_write_json(path, qualify.build_launch_manifest(inputs))
    qualify.validate_launch_manifest(path, inputs)
    with pytest.raises(qualify.QualificationError, match="inputs drifted"):
        qualify.validate_launch_manifest(
            path, {"schema": qualify.INPUTS_SCHEMA, "value": 2}
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["immutable_inputs_sha256"] = "0" * 64
    qualify.atomic_write_json(path, payload)
    with pytest.raises(qualify.QualificationError, match="digest is invalid"):
        qualify.validate_launch_manifest(path, inputs)


def _launch_fixture(
    output: Path,
    targets: list[qualify.Target],
    *,
    axis: str = qualify.DEFAULT_AXIS,
) -> dict[str, object]:
    benchmark = output / "benchmark"
    (benchmark / "onnx").mkdir(parents=True)
    (benchmark / "vnnlib").mkdir()
    for target in targets:
        (benchmark / "onnx" / target.onnx).write_bytes(b"model")
        (benchmark / "vnnlib" / target.vnnlib).write_bytes(b"property")
    binary = output / "ny-fixture"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    inputs = {
        "schema": qualify.INPUTS_SCHEMA,
        "binary_receipt_valid": True,
        "ny_source": {"clean": True},
        "compute_environment": qualify.compute_environment_contract(axis),
        "binary": {"path": str(binary)},
        "benchmark": {"root": str(benchmark)},
        "launch_tools": {
            "systemd_run": {"path": "/usr/bin/systemd-run"},
            "systemctl": {"path": "/usr/bin/systemctl"},
            "timeout": {"path": "/usr/bin/timeout"},
        },
        "campaign": {
            "output": str(output),
            "memory_max": "4G",
            "experiment_axis": axis,
            "promotion_scope": qualify.PROMOTION_SCOPE,
            "official_budget_secs": qualify.OFFICIAL_BUDGET_SECS,
            "allow_dirty": False,
            "allow_unreceipted": False,
            "partial_limit": 1,
        },
    }
    launch = qualify.build_launch_manifest(inputs)
    qualify.atomic_write_bytes(output / "targets.csv", qualify.render_targets(targets))
    qualify.atomic_write_json(output / "launch.json", launch)
    qualify.initialize_results(output / "results.tsv")
    return launch


def _install_run_one_subprocess_fixture(
    monkeypatch: pytest.MonkeyPatch,
    args: Namespace,
    target: qualify.Target,
    output: Path,
    arm: str,
    launch_digest: str,
    *,
    exit_code: int,
    status: str,
    scope_result: str,
    log_body: bytes = b"fixture child completed\n",
) -> list[str]:
    result_path, flight_path, _log_path = qualify.artifact_paths(output, target, arm)
    scope_unit = qualify.attempt_scope_unit_name(target, 1, arm, launch_digest)
    events: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[0] == args.systemd_run:
            events.append("launch")
            assert command[command.index("--unit") + 1] == scope_unit
            result_path.write_text(f"{status}\n", encoding="utf-8")
            flight_path.write_text(
                json.dumps(_flight_payload(arm, status)) + "\n",
                encoding="utf-8",
            )
            log = kwargs["stdout"]
            assert hasattr(log, "write")
            log.write(log_body)
            return SimpleNamespace(returncode=exit_code)
        assert command[0] == args.systemctl
        assert command[3] == scope_unit
        if command[2] == "reset-failed":
            events.append("reset")
            assert kwargs["timeout"] == qualify.SYSTEMD_SCOPE_RESET_TIMEOUT_SECS
            return SimpleNamespace(returncode=0)
        assert command[2] == "show"
        events.append("diagnostics")
        assert kwargs["timeout"] == qualify.SYSTEMD_SCOPE_QUERY_TIMEOUT_SECS
        return SimpleNamespace(
            returncode=0,
            stdout=f"LoadState=loaded\nResult={scope_result}\n",
            stderr="",
        )

    monkeypatch.setattr(qualify.subprocess, "run", fake_run)
    return events


def test_run_one_happy_path_resets_scope_around_launch_and_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _targets()[0]
    launch = _launch_fixture(tmp_path, [target])
    args = qualify.args_from_launch(launch)
    args.dry_run = False
    launch_digest = str(launch["immutable_inputs_sha256"])
    events = _install_run_one_subprocess_fixture(
        monkeypatch,
        args,
        target,
        tmp_path,
        "off",
        launch_digest,
        exit_code=0,
        status="timeout",
        scope_result="success",
    )

    row = qualify.run_one(target, 1, "off", args, tmp_path, launch_digest)

    assert row["status"] == "timeout"
    assert row["exit_code"] == "0"
    assert row["arm_authenticated"] == "true"
    qualify.validate_row_artifacts(row, target, 1, "off", tmp_path)
    assert events == ["reset", "launch", "diagnostics", "reset"]


def test_run_one_nonzero_exit_with_valid_artifacts_still_returns_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _targets()[0]
    launch = _launch_fixture(tmp_path, [target])
    args = qualify.args_from_launch(launch)
    args.dry_run = False
    launch_digest = str(launch["immutable_inputs_sha256"])
    events = _install_run_one_subprocess_fixture(
        monkeypatch,
        args,
        target,
        tmp_path,
        "on",
        launch_digest,
        exit_code=17,
        status="timeout",
        scope_result="exit-code",
    )

    row = qualify.run_one(target, 1, "on", args, tmp_path, launch_digest)

    assert row["status"] == "timeout"
    assert row["exit_code"] == "17"
    assert row["arm_authenticated"] == "true"
    qualify.validate_row_artifacts(row, target, 1, "on", tmp_path)
    assert events == ["reset", "launch", "diagnostics", "reset"]


def test_run_one_wrapper_vjp_decline_is_explicitly_non_promotable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _targets()[0]
    launch = _launch_fixture(
        tmp_path,
        [target],
        axis=qualify.WRAPPER_VJP_AXIS,
    )
    args = qualify.args_from_launch(launch)
    args.dry_run = False
    launch_digest = str(launch["immutable_inputs_sha256"])
    decline = (
        f"{qualify.WRAPPER_VJP_DECLINED_PREFIX} "
        "(reason=no_accelerator)\n"
    ).encode()
    events = _install_run_one_subprocess_fixture(
        monkeypatch,
        args,
        target,
        tmp_path,
        "on",
        launch_digest,
        exit_code=0,
        status="timeout",
        scope_result="success",
        log_body=decline,
    )

    with pytest.raises(qualify.QualificationError) as caught:
        qualify.run_one(target, 1, "on", args, tmp_path, launch_digest)

    message = str(caught.value)
    assert "capability-inconclusive" in message
    assert "permanently non-promotable" in message
    assert "reason=no_accelerator" in message
    assert "no on-arm efficacy comparison is possible on this host" in message
    assert events == ["reset", "launch", "diagnostics", "reset"]


def test_run_one_reports_cgroup_oom_before_constructing_a_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _targets()[0]
    launch = _launch_fixture(tmp_path, [target])
    args = qualify.args_from_launch(launch)
    args.dry_run = False
    qualify.preseal_attempt(target, 1, "off", args, tmp_path, launch)
    result_path, flight_path, _log_path = qualify.artifact_paths(
        tmp_path, target, "off"
    )
    launch_digest = str(launch["immutable_inputs_sha256"])
    scope_unit = qualify.attempt_scope_unit_name(target, 1, "off", launch_digest)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[0] == args.systemd_run:
            assert command[command.index("--unit") + 1] == scope_unit
            result_path.write_text("unknown\n", encoding="utf-8")
            flight_path.write_text(
                json.dumps(_flight_payload("off", "unknown")) + "\n",
                encoding="utf-8",
            )
            log = kwargs["stdout"]
            assert hasattr(log, "write")
            log.write(b"fixture child killed\n")
            return SimpleNamespace(returncode=137)
        assert command[0] == args.systemctl
        assert command[3] == scope_unit
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "LoadState=loaded\n"
                "Result=oom-kill\n"
                "OOMKills=1\n"
                "MemoryPeak=4294967296\n"
                "MemoryMax=4294967296\n"
                "MemorySwapPeak=0\n"
                "MemorySwapMax=0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(qualify.subprocess, "run", fake_run)
    with pytest.raises(qualify.QualificationError) as caught:
        qualify.run_one(target, 1, "off", args, tmp_path, launch_digest)
    message = str(caught.value)
    assert "child failed before authenticated flight evidence" in message
    assert "exit_code=137" in message
    assert "missing_or_invalid_artifacts=none" in message
    assert "cause=cgroup-oom" in message
    assert "scope_result='oom-kill'" in message
    assert "oom_kills='1'" in message
    assert "memory_peak='4294967296'" in message
    assert "permanently non-promotable and will not be retried" in message
    attempt_path, row_path = qualify.attempt_paths(tmp_path, target, "off")
    assert attempt_path.is_file()
    assert not row_path.exists()
    assert flight_path.is_file()


def test_run_one_missing_artifacts_fails_closed_when_scope_query_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _targets()[0]
    launch = _launch_fixture(tmp_path, [target])
    args = qualify.args_from_launch(launch)
    args.dry_run = False
    qualify.preseal_attempt(target, 1, "on", args, tmp_path, launch)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if command[0] == args.systemd_run:
            return SimpleNamespace(returncode=0)
        assert command[0] == args.systemctl
        return SimpleNamespace(returncode=1, stdout="", stderr="not found")

    monkeypatch.setattr(qualify.subprocess, "run", fake_run)
    launch_digest = str(launch["immutable_inputs_sha256"])
    with pytest.raises(qualify.QualificationError) as caught:
        qualify.run_one(target, 1, "on", args, tmp_path, launch_digest)
    message = str(caught.value)
    assert "exit_code=0" in message
    assert "missing_or_invalid_artifacts=result,flight" in message
    assert "cause=child-failure scope_diagnostics=unavailable" in message
    assert "permanently non-promotable and will not be retried" in message
    attempt_path, row_path = qualify.attempt_paths(tmp_path, target, "on")
    assert attempt_path.is_file()
    assert not row_path.exists()


def test_completion_manifest_binds_campaign_files_and_artifacts(tmp_path: Path) -> None:
    target = _targets()[0]
    targets = [target]
    launch = _launch_fixture(tmp_path, targets)
    args = qualify.args_from_launch(launch)
    qualify.preseal_attempt(target, 1, "off", args, tmp_path, launch)
    off_row = _write_arm_artifacts(tmp_path, target, 1, "off", "timeout")
    qualify.commit_row_fragment(tmp_path, target, 1, "off", off_row)
    qualify.preseal_attempt(target, 1, "on", args, tmp_path, launch)
    on_row = _write_arm_artifacts(tmp_path, target, 1, "on", "timeout")
    qualify.commit_row_fragment(tmp_path, target, 1, "on", on_row)
    rows = [off_row, on_row]
    qualify.publish_results_snapshot(tmp_path / "results.tsv", rows)
    summary = qualify.summarize_with_launch_policy(rows, targets, launch)
    qualify.atomic_write_json(tmp_path / "summary.json", summary)
    completion = qualify.build_completion_manifest(
        tmp_path, launch, summary, rows, targets
    )
    qualify.atomic_write_json(tmp_path / "completion.json", completion)
    qualify.validate_completion_manifest(tmp_path, targets)
    stored = json.loads((tmp_path / "completion.json").read_text(encoding="utf-8"))
    assert stored["schema"] == qualify.COMPLETION_SCHEMA
    assert stored["files"]["results"]["sha256"] == qualify.sha256_file(
        tmp_path / "results.tsv"
    )
    assert (
        stored["artifacts"]["gain:1:off"]["flight"]["sha256"]
        == off_row["flight_sha256"]
    )
    fabricated = dict(summary)
    fabricated["sat_gains"] = 999
    qualify.atomic_write_json(tmp_path / "summary.json", fabricated)
    with pytest.raises(qualify.QualificationError, match="exact row recomputation"):
        qualify.validate_completion_manifest(tmp_path, targets)


def test_main_resume_refuses_a_presealed_incomplete_attempt_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _targets()[0]
    output = tmp_path / "campaign"
    binary = tmp_path / "ny"
    binary.write_bytes(b"fixture")
    binary.chmod(0o755)
    (tmp_path / "onnx").mkdir()
    (tmp_path / "vnnlib").mkdir()
    (tmp_path / "onnx" / target.onnx).write_bytes(b"model")
    (tmp_path / "vnnlib" / target.vnnlib).write_bytes(b"property")
    args = SimpleNamespace(
        official_root=tmp_path,
        benchmark_root=tmp_path,
        binary=binary,
        out=output,
        memory_max="4G",
        limit=1,
        dry_run=False,
        resume=False,
        allow_dirty=False,
        allow_unreceipted=False,
        systemd_run="/usr/bin/systemd-run",
        systemctl="/usr/bin/systemctl",
        timeout="/usr/bin/timeout",
    )
    monkeypatch.setattr(qualify, "parse_args", lambda: args)
    monkeypatch.setattr(
        qualify, "derive_selected_targets", lambda _args: ([target], ("fixture",))
    )
    monkeypatch.setattr(qualify, "repo_is_dirty", lambda: False)
    monkeypatch.setattr(qualify, "validate_binary_receipt", lambda _args: True)

    def fake_inputs(
        _args: object,
        targets: list[qualify.Target],
        target_csv_sha256: str,
        receipt_valid: bool,
    ) -> dict[str, object]:
        assert targets == [target]
        assert receipt_valid is True
        return {
            "schema": qualify.INPUTS_SCHEMA,
            "binary_receipt_valid": True,
            "ny_source": {"clean": True},
            "compute_environment": qualify.compute_environment_contract(),
            "binary": {"path": str(binary)},
            "benchmark": {"root": str(tmp_path)},
            "launch_tools": {
                "systemd_run": {"path": args.systemd_run},
                "systemctl": {"path": args.systemctl},
                "timeout": {"path": args.timeout},
            },
            "campaign": {
                "target_csv_sha256": target_csv_sha256,
                "output": str(output),
                "memory_max": "4G",
                "promotion_scope": qualify.PROMOTION_SCOPE,
                "official_budget_secs": qualify.OFFICIAL_BUDGET_SECS,
                "allow_dirty": False,
                "allow_unreceipted": False,
                "partial_limit": 1,
            },
        }

    monkeypatch.setattr(qualify, "capture_immutable_inputs", fake_inputs)
    attempts = {"off": 0, "on": 0}

    def fake_run_one(
        selected: qualify.Target,
        order_index: int,
        arm: str,
        _args: object,
        out: Path,
        _launch_inputs_sha256: str | None = None,
    ) -> dict[str, str]:
        attempts[arm] += 1
        if arm == "on" and attempts[arm] == 1:
            raise KeyboardInterrupt
        return _write_arm_artifacts(out, selected, order_index, arm, "timeout")

    monkeypatch.setattr(qualify, "run_one", fake_run_one)
    with pytest.raises(KeyboardInterrupt):
        qualify.main()
    persisted = qualify.load_persisted_rows(output / "results.tsv", [target], output)
    assert [row["arm"] for row in persisted] == ["off"]

    args.resume = True
    with pytest.raises(
        qualify.QualificationError, match="presealed incomplete attempt"
    ):
        qualify.main()
    assert attempts == {"off": 1, "on": 1}
    assert qualify.attempt_paths(output, target, "on")[0].is_file()
    assert not qualify.attempt_paths(output, target, "on")[1].exists()


def test_benchmark_asset_accepts_compressed_inputs(tmp_path: Path) -> None:
    compressed = tmp_path / "model.onnx.gz"
    compressed.write_bytes(b"fixture")
    assert qualify.benchmark_asset(tmp_path, "model.onnx") == compressed
