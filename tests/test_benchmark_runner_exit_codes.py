# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import csv
import json
import os
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MALBEWARE_SCRIPT = REPO_ROOT / "scripts" / "benchmark_malbeware.sh"
CNN_SCRIPT = REPO_ROOT / "scripts" / "run_cnn_benchmark.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    path.chmod(0o755)


def _prepare_malbeware_fixture(
    tmp_path: Path, ny_source: str
) -> tuple[Path, dict[str, str]]:
    benchmark = tmp_path / "benchmarks" / "vnncomp2025" / "benchmarks" / "malbeware"
    benchmark.mkdir(parents=True)
    (benchmark / "instances.csv").write_text(
        "model.onnx,property.vnnlib,5\n",
        encoding="utf-8",
    )
    (benchmark / "model.onnx").write_bytes(b"model\n")
    (benchmark / "property.vnnlib").write_text("(assert true)\n", encoding="utf-8")

    preset = tmp_path / "configs" / "vnncomp25" / "malbeware.yaml"
    preset.parent.mkdir(parents=True)
    preset.write_text("solver: {}\n", encoding="utf-8")

    ny_binary = tmp_path / "fake-ny"
    _write_executable(ny_binary, ny_source)
    env = os.environ.copy()
    env["NY_BIN"] = str(ny_binary)
    return ny_binary, env


def _run_malbeware(
    tmp_path: Path, ny_source: str
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    _, env = _prepare_malbeware_fixture(tmp_path, ny_source)
    result = subprocess.run(
        ["bash", str(MALBEWARE_SCRIPT), "all", "--limit", "1"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    reports = list((tmp_path / "reports" / "benchmarks").glob("malbeware_*.csv"))
    assert len(reports) == 1, f"expected one report, found {reports}"
    with reports[0].open(newline="", encoding="utf-8") as report_file:
        rows = list(csv.DictReader(report_file))
    assert len(rows) == 1, f"expected one result row, got {rows}"
    return result, rows[0]


def _prepare_cnn_fixture(tmp_path: Path, ny_source: str) -> dict[str, str]:
    benchmark = (
        tmp_path / "benchmarks" / "vnncomp2021" / "benchmarks" / "cifar10_resnet"
    )
    model = benchmark / "onnx" / "resnet_2b.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model\n")
    prop = (
        benchmark
        / "vnnlib_properties_pgd_filtered"
        / "resnet2b_pgd_filtered"
        / "prop_0_eps_0.008.vnnlib"
    )
    prop.parent.mkdir(parents=True)
    prop.write_text("(assert true)\n", encoding="utf-8")

    ny_binary = tmp_path / "fake-ny"
    _write_executable(ny_binary, ny_source)

    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    _write_executable(
        fakebin / "git",
        """\
        #!/bin/sh
        if [ "$1" = "rev-parse" ] && [ "$2" = "--short" ] && [ "$3" = "HEAD" ]; then
            echo deadbeef
            exit 0
        fi
        echo "unexpected git arguments: $*" >&2
        exit 99
        """,
    )

    env = os.environ.copy()
    env.update(
        {
            "MAX_INSTANCES": "1",
            "METHOD": "alpha",
            "NY_BINARY": str(ny_binary),
            "PATH": f"{fakebin}:{env['PATH']}",
            "TIMEOUT": "5",
        }
    )
    return env


def _run_cnn(
    tmp_path: Path, ny_source: str
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    env = _prepare_cnn_fixture(tmp_path, ny_source)
    result = subprocess.run(
        ["bash", str(CNN_SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    reports = list(
        (tmp_path / "reports" / "benchmarks").glob("cifar10_resnet_2b_alpha_*.json")
    )
    assert len(reports) == 1, f"expected one report, found {reports}"
    return result, json.loads(reports[0].read_text(encoding="utf-8"))


def test_malbeware_accepts_documented_nonzero_verdict_exit(tmp_path: Path) -> None:
    result, row = _run_malbeware(
        tmp_path,
        """\
        #!/bin/sh
        echo "Status: VIOLATED"
        exit 1
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert row["result"] == "violated"
    assert row["exit_code"] == "1"


def test_malbeware_rejects_unexpected_solver_exit(tmp_path: Path) -> None:
    result, row = _run_malbeware(
        tmp_path,
        """\
        #!/bin/sh
        echo "Status: VERIFIED"
        exit 42
        """,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert row["result"] == "error"
    assert row["exit_code"] == "42"
    assert "ny exit=42 parsed=verified" in result.stdout


def test_cnn_accepts_documented_nonzero_verdict_exit(tmp_path: Path) -> None:
    result, report = _run_cnn(
        tmp_path,
        """\
        #!/bin/sh
        echo '{"status":"violated","property_status":"violated"}'
        exit 1
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert report["falsified"] == 1
    assert report["error"] == 0
    assert len(report["results"]) == 1
    row = report["results"][0]
    assert row["exit_code"] == 1
    assert row["instance"] == "prop_0"
    assert row["status"] == "falsified"
    assert row["time_s"] >= 0


def test_cnn_rejects_unexpected_solver_exit(tmp_path: Path) -> None:
    result, report = _run_cnn(
        tmp_path,
        """\
        #!/bin/sh
        echo '{"status":"verified","property_status":"safe"}'
        exit 42
        """,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert report["error"] == 1
    assert report["results"][0]["status"] == "error"
    assert report["results"][0]["exit_code"] == 42
    assert "ny exit=42 parsed=verified" in result.stderr
