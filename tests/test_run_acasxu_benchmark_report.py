# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.run_acasxu_benchmark_report as report


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ny", "bench"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def _run_main(monkeypatch, tmp_path: Path, stdout: str) -> int:
    (tmp_path / "target/release").mkdir(parents=True, exist_ok=True)
    ny_path = tmp_path / "target/release/ny"
    ny_path.write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(report, "DEFAULT_NY_PATH", Path("target/release/ny"))
    monkeypatch.setattr(report, "REPORTS_DIR", Path("reports/benchmarks"))
    monkeypatch.setattr(report, "METRICS_DIR", Path("metrics/benchmarks"))
    monkeypatch.setattr(report, "_run", lambda cmd, cwd=None: _completed(stdout))
    monkeypatch.setattr(report, "_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(sys, "argv", ["prog", "--ny", str(ny_path)])

    return report.main()


def _load_report(tmp_path: Path) -> dict[str, object]:
    reports_dir = tmp_path / "reports/benchmarks"
    report_path = next(reports_dir.glob("acasxu_2021_*.json"))
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_run_benchmark_report_single_json(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "benchmark": "acasxu",
        "benchmark_year": 2021,
        "verified": 3,
        "total": 5,
        "pass_rate": 0.6,
        "avg_time_ms": 10.0,
        "timeout_count": 1,
        "error_count": 0,
    }
    exit_code = _run_main(monkeypatch, tmp_path, json.dumps(payload))
    assert exit_code == 0

    report_data = _load_report(tmp_path)
    assert report_data["verified"] == 3
    assert report_data["benchmark_year"] == 2021


def test_run_benchmark_report_trailing_output(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "benchmark": "acasxu",
        "benchmark_year": 2021,
        "verified": 7,
        "total": 9,
        "pass_rate": 0.777,
        "avg_time_ms": 11.0,
        "timeout_count": 0,
        "error_count": 0,
    }
    stdout = json.dumps(payload) + "\nBENCHMARK COMPLETE\n"
    exit_code = _run_main(monkeypatch, tmp_path, stdout)
    assert exit_code == 0

    report_data = _load_report(tmp_path)
    assert report_data["verified"] == 7
    assert report_data["total"] == 9


def test_run_benchmark_report_multi_json(monkeypatch, tmp_path: Path) -> None:
    payload_first = {
        "benchmark": "acasxu",
        "benchmark_year": 2021,
        "verified": 2,
        "total": 4,
        "pass_rate": 0.5,
        "avg_time_ms": 9.0,
        "timeout_count": 0,
        "error_count": 0,
    }
    payload_second = {
        "benchmark": "acasxu",
        "benchmark_year": 2021,
        "verified": 99,
        "total": 100,
        "pass_rate": 0.99,
        "avg_time_ms": 1.0,
        "timeout_count": 0,
        "error_count": 0,
    }
    stdout = json.dumps(payload_first) + "\n" + json.dumps(payload_second)
    exit_code = _run_main(monkeypatch, tmp_path, stdout)
    assert exit_code == 0

    report_data = _load_report(tmp_path)
    assert report_data["verified"] == 2
    assert report_data["total"] == 4
