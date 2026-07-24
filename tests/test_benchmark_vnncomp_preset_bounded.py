# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import scripts.benchmark_vnncomp_preset_bounded as bounded


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


def _run_main(monkeypatch, tmp_path: Path, ny_path: Path) -> int:
    model_path, property_path, preset_path = _write_inputs(tmp_path)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports" / "benchmarks")
    monkeypatch.setattr(bounded, "NY_PREFLIGHT_TIMEOUT_SECS", 2.0)
    monkeypatch.setattr(
        bounded,
        "get_benchmark_instances",
        lambda year, category: [(model_path, property_path, 2)],
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
        ],
    )
    return bounded.main()


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

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="ny 0.0.0\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"status":"verified","domains_explored":7,"domains_verified":3,"max_depth_reached":1}\n',
            stderr="",
        )

    monkeypatch.setattr(bounded.subprocess, "run", fake_run)

    exit_code = _run_main(monkeypatch, tmp_path, ny_path)

    assert exit_code == 0, f"Expected successful run exit code 0, got {exit_code}"
    report = next((tmp_path / "reports" / "benchmarks").glob("fixturecat_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1, f"Expected one benchmark row, got {len(rows)}"
    assert rows[0]["result"] == "verified", (
        f"Expected verified result row, got {rows[0]!r}"
    )
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
        lambda year, category: [(model_path, property_path, 2)],
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
    import os

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


def _run_main_with_argv(monkeypatch, tmp_path: Path, ny_path: Path, extra_argv: list[str]) -> int:
    """Helper: run main() with custom argv extensions."""
    model_path, property_path, preset_path = _write_inputs(tmp_path)
    monkeypatch.setattr(bounded, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bounded, "REPORTS_DIR", tmp_path / "reports" / "benchmarks")
    monkeypatch.setattr(bounded, "NY_PREFLIGHT_TIMEOUT_SECS", 2.0)
    monkeypatch.setattr(
        bounded,
        "get_benchmark_instances",
        lambda year, category: [(model_path, property_path, 2)],
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

    The warmup attempt must pass timeout=None to subprocess.run (untimed),
    while the measured attempt passes timeout=timeout+timeout_slack.
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
    assert observed_timeouts == [None, 7], (
        f"Expected [None, 7] (untimed warmup, then timeout+slack=2+5), got {observed_timeouts}"
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
        lambda year, category: [(model_path, property_path, 2)],
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
