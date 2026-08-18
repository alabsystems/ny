#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NVIDIA/Vulkan validation orchestrator manifest contract.

The shell orchestrator (scripts/run_nvidia_vulkan_validation.sh) is designed to
run on real NVIDIA hosts and shells out to cargo, benchmark scripts, etc. These
tests validate:
1. The manifest JSON schema contract that the orchestrator produces
2. The renderer's ability to consume each manifest variant
3. The shell script's syntactic validity

The real GPU path is host-dependent. A fake-command subprocess test still
exercises orchestration, artifact routing, command-path quoting, and JSON
serialization without requiring NVIDIA hardware.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "benchmark_reports"
    / "nvidia_vulkan_validation"
)
SCRIPT = REPO_ROOT / "scripts" / "run_nvidia_vulkan_validation.sh"
RENDERER = REPO_ROOT / "scripts" / "render_nvidia_vulkan_validation_report.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from render_nvidia_vulkan_validation_report import (
    _compute_verdict,
    compute_cersyve_totals,
    load_manifest,
    render_report,
)


class TestShellScriptSyntax:
    """The orchestrator shell script is syntactically valid."""

    def test_bash_n_passes(self):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


def _write_executable(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _run_with_fake_measurement_command(
    tmp_path: Path,
    cargo_source: str,
    *,
    benchmark_source: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    command_dir = tmp_path / "measurement commands"
    cargo = _write_executable(command_dir / "fake cargo", cargo_source)
    benchmark_called = tmp_path / "benchmark-called"
    benchmark = _write_executable(
        command_dir / "fake benchmark",
        benchmark_source
        or "#!/bin/bash\n: > \"${BENCHMARK_CALLED_FILE:?}\"\nexit 99\n",
    )
    output_dir = tmp_path / "validation output"
    output_dir.mkdir()
    (output_dir / "issue-4359-nvidia-vulkan-crown-backward.csv").write_text(
        "stale,data\n",
        encoding="utf-8",
    )
    (output_dir / "issue-4359-nvidia-vulkan-manifest.json").write_text(
        '{"stale": true}\n',
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "NVIDIA_SMI_CMD": "/bin/true",
        "CARGO_CMD": str(cargo),
        "CARGO_VERSION_CMD": str(cargo),
        "RUSTC_CMD": "/bin/true",
        "BENCHMARK_SCRIPT": str(benchmark),
        "BENCHMARK_CALLED_FILE": str(benchmark_called),
        "NY_BIN": "/bin/true",
    }

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--skip-reference",
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result, output_dir, benchmark_called


def test_orchestrator_writes_blocked_manifest_when_measurement_command_fails(
    tmp_path: Path,
) -> None:
    result, output_dir, benchmark_called = _run_with_fake_measurement_command(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/bash
            if [[ "${1:-}" == "-V" ]]; then
                printf 'cargo 1.0-test\\n'
                exit 0
            fi
            printf 'wgpu adapter: incomplete fixture (backend: Vulkan)\\n'
            exit 23
            """
        ),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    manifest = json.loads(
        (
            output_dir / "issue-4359-nvidia-vulkan-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["verdict"] == "blocked"
    assert manifest["blocker"] == (
        "measurement command failed (cargo exit 23, tee exit 0)"
    )
    assert manifest["reference_blocker"] == (
        "blocked: measurement command failed, skipped all downstream steps"
    )
    assert manifest["vulkan_confirmed"] is False
    assert manifest["compare_backends_cersyve_csv"] is None
    assert manifest["compare_backends_metaroom_csv"] is None
    assert not (
        output_dir / "issue-4359-nvidia-vulkan-crown-backward.csv"
    ).exists()
    assert not benchmark_called.exists()


def test_orchestrator_rejects_successful_measurement_without_csv(
    tmp_path: Path,
) -> None:
    result, output_dir, benchmark_called = _run_with_fake_measurement_command(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/bash
            if [[ "${1:-}" == "-V" ]]; then
                printf 'cargo 1.0-test\\n'
                exit 0
            fi
            printf 'wgpu adapter: CSV-free fixture (backend: Vulkan)\\n'
            """
        ),
    )

    assert result.returncode == 1, result.stdout + result.stderr
    manifest = json.loads(
        (
            output_dir / "issue-4359-nvidia-vulkan-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["verdict"] == "blocked"
    assert manifest["blocker"] == (
        "measurement command completed without a non-empty CSV artifact"
    )
    assert manifest["reference_blocker"] == (
        "blocked: measurement CSV missing or empty, skipped all downstream steps"
    )
    assert manifest["vulkan_confirmed"] is True
    assert manifest["adapter_line"] == (
        "wgpu adapter: CSV-free fixture (backend: Vulkan)"
    )
    assert not (
        output_dir / "issue-4359-nvidia-vulkan-crown-backward.csv"
    ).exists()
    assert not benchmark_called.exists()


def test_orchestrator_rejects_traversal_in_child_report_path(
    tmp_path: Path,
) -> None:
    result, output_dir, _benchmark_called = _run_with_fake_measurement_command(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/bash
            if [[ "${1:-}" == "-V" ]]; then
                printf 'cargo 1.0-test\\n'
                exit 0
            fi
            previous=""
            for argument in "$@"; do
                if [[ "$previous" == "--output" ]]; then
                    printf 'workload,value\\nfixture,1\\n' > "$argument"
                fi
                previous="$argument"
            done
            printf 'wgpu adapter: traversal fixture (backend: Vulkan)\\n'
            """
        ),
        benchmark_source=textwrap.dedent(
            """\
            #!/bin/bash
            set -euo pipefail
            category="$1"
            if [[ "$category" == "cersyve" ]]; then
                mkdir -p "${REPORT_DIR}/${category}_compare_backends_escape"
                printf 'escaped\\n' > "${REPORT_DIR}/../outside.csv"
                printf 'Report: %s\\n' \
                    "${REPORT_DIR}/${category}_compare_backends_escape/../../outside.csv"
            else
                output="${REPORT_DIR}/${category}_compare_backends_fixture.csv"
                printf 'category,backend,wall_seconds\\n' > "$output"
                printf 'Report: %s\\n' "$output"
            fi
            """
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(
        (
            output_dir / "issue-4359-nvidia-vulkan-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["verdict"] == "blocked"
    assert manifest["compare_backends_cersyve_csv"] is None
    assert manifest["compare_backends_metaroom_csv"].startswith(
        "metaroom_2023_compare_backends_"
    )
    assert "cersyve compare-backends command/report failed" in manifest["blocker"]
    assert "escaped or violated" in result.stderr


def test_orchestrator_collects_actual_compare_csvs_and_writes_valid_json(
    tmp_path: Path,
) -> None:
    command_dir = tmp_path / "commands with spaces"
    cargo = _write_executable(
        command_dir / "fake cargo",
        textwrap.dedent(
            """\
            #!/bin/bash
            set -euo pipefail
            if [[ "${1:-}" == "-V" ]]; then
                printf 'cargo 1.0-test\\n'
                exit 0
            fi
            previous=""
            for argument in "$@"; do
                if [[ "$previous" == "--output" ]]; then
                    printf 'workload,value\\nfixture,1\\n' > "$argument"
                fi
                previous="$argument"
            done
            printf 'wgpu adapter: Example "Quoted" \\\\ GPU (backend: Vulkan)\\n'
            """
        ),
    )
    rustc = _write_executable(
        command_dir / "fake rustc",
        "#!/bin/sh\nprintf 'rustc 1.0-test\\n'\n",
    )
    nvidia_smi = _write_executable(
        command_dir / "fake nvidia-smi",
        "#!/bin/sh\nprintf 'Example GPU, 1.0, 1 MiB\\n'\n",
    )
    benchmark = _write_executable(
        command_dir / "fake benchmark",
        textwrap.dedent(
            """\
            #!/bin/bash
            set -euo pipefail
            category="$1"
            output="${REPORT_DIR}/${category}_compare_backends_fixture_$$.csv"
            {
                printf 'category,backend,wall_seconds\\n'
                printf '%s,cpu,1.0\\n' "$category"
                printf '%s,wgpu,0.5\\n' "$category"
            } > "$output"
            printf 'Report: %s\\n' "$output"
            """
        ),
    )
    output_dir = tmp_path / "validation output"
    output_dir.mkdir()
    stale_report = output_dir / "cersyve_compare_backends_stale.csv"
    stale_report.write_text("stale\n", encoding="utf-8")
    # A future mtime would win the old glob/timestamp rediscovery even though
    # the child never named this file as its report.
    os.utime(stale_report, (2_000_000_000, 2_000_000_000))
    environment = {
        **os.environ,
        "NVIDIA_SMI_CMD": str(nvidia_smi),
        "CARGO_CMD": str(cargo),
        "CARGO_VERSION_CMD": str(cargo),
        "RUSTC_CMD": str(rustc),
        "BENCHMARK_SCRIPT": str(benchmark),
        "NY_BIN": "/bin/true",
    }

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--skip-reference",
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest_path = output_dir / "issue-4359-nvidia-vulkan-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["adapter_line"] == (
        'wgpu adapter: Example "Quoted" \\ GPU (backend: Vulkan)'
    )
    assert manifest["compare_backends_cersyve_csv"].startswith(
        "cersyve_compare_backends_"
    )
    assert manifest["compare_backends_cersyve_csv"] != stale_report.name
    assert manifest["compare_backends_metaroom_csv"].startswith(
        "metaroom_2023_compare_backends_"
    )
    assert (output_dir / manifest["compare_backends_cersyve_csv"]).is_file()
    assert (output_dir / manifest["compare_backends_metaroom_csv"]).is_file()
    assert manifest["blocker"] is None
    assert manifest["reference_blocker"] == "skipped: --skip-reference flag set"


class TestManifestContract:
    """The manifest JSON schema is consumed correctly by the renderer."""

    REQUIRED_KEYS = {
        "schema",
        "verdict",
        "blocker",
        "host_info_path",
        "measure_log_path",
        "measure_csv_path",
        "vulkan_confirmed",
        "adapter_line",
        "compare_backends_cersyve_csv",
        "compare_backends_metaroom_csv",
        "reference_blocker",
        "reference_cersyve_real_seconds",
    }

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "go_manifest.json",
            "nogo_manifest.json",
            "blocked_manifest.json",
            "blocked_reference_manifest.json",
        ],
    )
    def test_fixture_has_all_required_keys(self, fixture_name: str):
        with open(FIXTURES / fixture_name, encoding="utf-8") as f:
            data = json.load(f)
        missing = self.REQUIRED_KEYS - set(data.keys())
        assert not missing, f"Missing keys in {fixture_name}: {missing}"

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "go_manifest.json",
            "nogo_manifest.json",
            "blocked_manifest.json",
            "blocked_reference_manifest.json",
        ],
    )
    def test_fixture_has_correct_schema(self, fixture_name: str):
        manifest = load_manifest(FIXTURES / fixture_name)
        assert manifest["schema"] == "nvidia_vulkan_validation_manifest_v1", (
            f"Unexpected schema in {fixture_name}: {manifest['schema']}"
        )

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "go_manifest.json",
            "nogo_manifest.json",
            "blocked_manifest.json",
            "blocked_reference_manifest.json",
        ],
    )
    def test_renderer_produces_valid_markdown(self, fixture_name: str):
        manifest = load_manifest(FIXTURES / fixture_name)
        report = render_report(manifest, FIXTURES)
        assert "## Summary" in report, f"Missing Summary section in {fixture_name}"
        assert "## Verdict" in report, f"Missing Verdict section in {fixture_name}"
        assert len(report) > 100, f"Report too short for {fixture_name}: {len(report)} chars"

    def test_non_vulkan_manifest_produces_blocked_verdict(self):
        manifest = load_manifest(FIXTURES / "blocked_manifest.json")
        verdict = _compute_verdict(manifest, {})
        assert verdict == "blocked", f"Expected blocked, got {verdict!r}"

    def test_missing_reference_produces_blocked_verdict(self):
        manifest = load_manifest(FIXTURES / "blocked_reference_manifest.json")
        totals = compute_cersyve_totals([])
        verdict = _compute_verdict(manifest, totals)
        assert verdict == "blocked", f"Expected blocked, got {verdict!r}"

    def test_go_manifest_round_trip(self):
        """Manifest + CSV -> render -> verdict matches expected."""
        manifest = load_manifest(FIXTURES / "go_manifest.json")
        report = render_report(manifest, FIXTURES)
        assert "**go**" in report, "go manifest round-trip missing go verdict"

    def test_nogo_manifest_round_trip(self):
        manifest = load_manifest(FIXTURES / "nogo_manifest.json")
        report = render_report(manifest, FIXTURES)
        assert "**no-go**" in report, "nogo manifest round-trip missing no-go verdict"

    def test_blocked_manifest_round_trip(self):
        manifest = load_manifest(FIXTURES / "blocked_manifest.json")
        report = render_report(manifest, FIXTURES)
        assert "**blocked**" in report, "blocked manifest round-trip missing verdict"


class TestEndToEndCLI:
    """CLI integration: manifest -> renderer -> report file."""

    @pytest.mark.parametrize(
        "fixture_name,expected_verdict",
        [
            ("go_manifest.json", "**go**"),
            ("nogo_manifest.json", "**no-go**"),
            ("blocked_manifest.json", "**blocked**"),
            ("blocked_reference_manifest.json", "**blocked**"),
        ],
    )
    def test_cli_produces_expected_verdict(
        self, fixture_name: str, expected_verdict: str
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(RENDERER),
                    "--manifest",
                    str(FIXTURES / fixture_name),
                    "--output-dir",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            output = (
                Path(tmpdir) / "issue-4359-nvidia-vulkan-validation-current.md"
            )
            assert output.exists(), f"Expected report file at {output}"
            content = output.read_text()
            assert expected_verdict in content, (
                f"Expected {expected_verdict!r} in CLI output for {fixture_name}"
            )
