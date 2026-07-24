# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import csv
import os
import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_vnncomp.sh"


def _write_category_fixture(tmp_path: Path, category: str) -> None:
    category_dir = tmp_path / "benchmarks" / "vnncomp2025" / "benchmarks" / category
    (category_dir / "onnx").mkdir(parents=True, exist_ok=True)
    (category_dir / "vnnlib").mkdir(parents=True, exist_ok=True)
    (category_dir / "onnx" / "model.onnx").write_bytes(b"\x08\x01\x12\x03foo")
    (category_dir / "vnnlib" / "prop.vnnlib").write_text("", encoding="utf-8")
    (category_dir / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/prop.vnnlib,2\n",
        encoding="utf-8",
    )


def _write_fake_ny(tmp_path: Path) -> Path:
    ny_path = tmp_path / "fake_ny.sh"
    ny_path.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            printf '%s\n' "$@" > "$(dirname "$0")/argv.txt"
            printf 'Status: VERIFIED\nDomains explored: 3\n'
            """
        ),
        encoding="utf-8",
    )
    ny_path.chmod(0o755)
    return ny_path


def _run_benchmark(tmp_path: Path, ny_path: Path, category: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BENCH_ROOT"] = str(tmp_path / "benchmarks" / "vnncomp2025" / "benchmarks")
    env["NY_BIN"] = str(ny_path)
    env["MAX_SIGNAL_RETRIES"] = "1"
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), category],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _load_single_result(tmp_path: Path, category: str) -> dict[str, str]:
    report = next((tmp_path / "reports" / "benchmarks").glob(f"{category}_*.csv"))
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1, rows
    return rows[0]


def test_benchmark_vnncomp_relusplitter_defaults_to_mip_4323(tmp_path: Path) -> None:
    category = "relusplitter"
    _write_category_fixture(tmp_path, category)
    ny_path = _write_fake_ny(tmp_path)

    result = _run_benchmark(tmp_path, ny_path, category)

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, category)
    assert row["status"] == "verified", row

    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert "--complete-verifier" in argv and "mip" in argv, (
        f"expected relusplitter to default to MIP complete verifier, got argv: {argv}"
    )
    assert "--mip-solver" not in argv, (
        f"the script must not inject --mip-solver (auto-escalation picks the solver), got argv: {argv}"
    )
    assert "Verifier: --complete-verifier mip" in result.stdout, (
        f"expected stdout banner to report relusplitter MIP auto-route, got: {result.stdout}"
    )
