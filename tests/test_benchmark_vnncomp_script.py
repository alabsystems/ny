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


def _benchmark_root(tmp_path: Path, suite: str = "vnncomp2025") -> Path:
    return tmp_path / "benchmarks" / suite / "benchmarks"


def _write_category_fixture_rows(
    tmp_path: Path,
    category: str,
    rows: list[tuple[str, str, str]],
    *,
    suite: str = "vnncomp2025",
) -> None:
    category_dir = _benchmark_root(tmp_path, suite) / category
    instances_lines: list[str] = []

    for onnx_rel, vnnlib_rel, timeout in rows:
        model_path = category_dir / onnx_rel
        property_path = category_dir / vnnlib_rel
        model_path.parent.mkdir(parents=True, exist_ok=True)
        property_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            model_path.write_bytes(b"\x08\x01\x12\x03foo")
        if not property_path.exists():
            property_path.write_text("", encoding="utf-8")
        instances_lines.append(f"{onnx_rel},{vnnlib_rel},{timeout}")

    (category_dir / "instances.csv").write_text(
        "\n".join(instances_lines) + "\n",
        encoding="utf-8",
    )


def _write_category_fixture(tmp_path: Path, category: str, *, suite: str = "vnncomp2025") -> None:
    _write_category_fixture_rows(
        tmp_path,
        category,
        [("onnx/model.onnx", "vnnlib/prop.vnnlib", "2")],
        suite=suite,
    )


def _write_fake_ny(tmp_path: Path, body: str) -> Path:
    ny_path = tmp_path / "fake_ny.sh"
    ny_path.write_text(body, encoding="utf-8")
    ny_path.chmod(0o755)
    return ny_path


def _write_reference_csv(tmp_path: Path, category: str, *, result: str) -> Path:
    reference_dir = tmp_path / "reports" / "benchmarks" / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_path = reference_dir / f"{category}_alpha_beta_crown.csv"
    reference_path.write_text(
        "model,property,result\nonnx/model.onnx,vnnlib/prop.vnnlib,"
        + result
        + "\n",
        encoding="utf-8",
    )
    return reference_path


def _write_reference_manifest(
    tmp_path: Path,
    category: str,
    reference_path: Path | str,
) -> None:
    manifest_path = tmp_path / "reports" / "benchmarks" / "reference" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(reference_path, Path):
        output_path = reference_path.relative_to(tmp_path)
    else:
        output_path = reference_path
    manifest_path.write_text(
        textwrap.dedent(
            f"""\
            {{
              "categories": ["{category}"],
              "reference_files": {{
                "{category}": {{
                  "output_path": "{output_path}",
                  "instance_count": 1
                }}
              }}
            }}
            """
        ),
        encoding="utf-8",
    )


def _write_compare_backends_ny(
    tmp_path: Path,
    *,
    cpu_domains: str = "3",
    wgpu_domains: str = "5",
    fixed_domains: str | None = None,
) -> Path:
    if fixed_domains is not None:
        return _write_fake_ny(
            tmp_path,
            textwrap.dedent(
                f"""\
                #!/bin/sh
                printf 'Status: VERIFIED\\nDomains explored: {fixed_domains}\\n'
                """
            ),
        )

    return _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            f"""\
            #!/bin/sh
            backend="cpu"
            prev=""
            for arg in "$@"; do
                if [ "$prev" = "--backend" ]; then
                    backend="$arg"
                fi
                prev="$arg"
            done
            if [ "$backend" = "wgpu" ]; then
                printf 'Status: VERIFIED\\nDomains explored: {wgpu_domains}\\n'
            else
                printf 'Status: VERIFIED\\nDomains explored: {cpu_domains}\\n'
            fi
            """
        ),
    )


def _run_benchmark(
    tmp_path: Path,
    ny_path: Path,
    category: str,
    args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    benchmark_suite: str = "vnncomp2025",
    bench_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BENCH_ROOT"] = str(bench_root or _benchmark_root(tmp_path, benchmark_suite))
    env["NY_BIN"] = str(ny_path)
    env["MAX_SIGNAL_RETRIES"] = "1"
    if extra_env:
        env.update(extra_env)
    command = ["bash", str(SCRIPT_PATH), category]
    if args:
        command.extend(args)
    return subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _load_report_rows(tmp_path: Path, pattern: str) -> list[dict[str, str]]:
    matches = sorted((tmp_path / "reports" / "benchmarks").glob(pattern))
    assert matches, f"no reports matched {pattern}"
    report = matches[-1]
    with report.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_single_result(tmp_path: Path, category: str) -> dict[str, str]:
    rows = _load_report_rows(tmp_path, f"{category}_*.csv")
    assert len(rows) == 1, f"expected exactly one CSV row for {category}, got {len(rows)}"
    return rows[0]


def test_benchmark_vnncomp_emits_provenance_tags_in_notes(tmp_path: Path) -> None:
    """Provenance tags must appear in the notes field of every emitted row (#4346)."""
    category = "provcat"
    _write_category_fixture(tmp_path, category)
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            if [ "$1" = "--version" ]; then
                echo "ny 0.1.0-test"
                exit 0
            fi
            printf 'Status: VERIFIED\\nDomains explored: 1\\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category)
    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, category)
    notes = row.get("notes", "")
    assert "ny_source=" in notes, f"expected ny_source in notes, got {notes!r}"
    assert "ny_bin=" in notes, f"expected ny_bin in notes, got {notes!r}"
    assert "ny_version=ny 0.1.0-test" in notes, f"expected ny_version in notes, got {notes!r}"
    assert "ny_sha256=" in notes, f"expected ny_sha256 in notes, got {notes!r}"


def test_benchmark_vnncomp_explicit_ny_bin_records_explicit_source(tmp_path: Path) -> None:
    """When NY_BIN is set explicitly, ny_source=explicit must appear (#4346)."""
    category = "explicitcat"
    _write_category_fixture(tmp_path, category)
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            if [ "$1" = "--version" ]; then
                echo "ny 0.1.0-test"
                exit 0
            fi
            printf 'Status: VERIFIED\\nDomains explored: 1\\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category)
    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, category)
    notes = row.get("notes", "")
    assert "ny_source=explicit" in notes, (
        f"NY_BIN was set explicitly, expected ny_source=explicit, got {notes!r}"
    )


def test_benchmark_vnncomp_domain_batch_metrics_flag_links_sidecar(tmp_path: Path) -> None:
    category = "domainbatchcat"
    _write_category_fixture(tmp_path, category)
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            if [ "$1" = "--version" ]; then
                echo "ny 0.1.0-test"
                exit 0
            fi
            prev=""
            metrics=""
            for arg in "$@"; do
                if [ "$prev" = "--domain-batch-metrics-jsonl" ]; then
                    metrics="$arg"
                fi
                prev="$arg"
            done
            if [ -n "$metrics" ]; then
                mkdir -p "$(dirname "$metrics")"
                printf '{"schema_version":"graph_domain_batch_metrics_v1"}\n' > "$metrics"
            fi
            printf 'Status: VERIFIED\nDomains explored: 1\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category, ["--domain-batch-metrics"])

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, category)
    notes = row.get("notes", "")
    assert "domain_batch_metrics_jsonl=" in notes, (
        f"expected domain-batch metrics tag in notes, got {notes!r}"
    )
    metrics_rel = notes.split("domain_batch_metrics_jsonl=", 1)[1].split(";", 1)[0].strip()
    metrics_path = tmp_path / metrics_rel
    assert metrics_path.exists(), f"expected sidecar at {metrics_path}"


def test_benchmark_vnncomp_retries_signal_exit_and_records_verdict(tmp_path: Path) -> None:
    category = "retrycat"
    _write_category_fixture(tmp_path, category)
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            if [ "$1" = "--version" ]; then echo "ny 0.0.0-test"; exit 0; fi
            count_file="$(dirname "$0")/count.txt"
            count=0
            if [ -f "$count_file" ]; then
                count=$(cat "$count_file")
            fi
            count=$((count + 1))
            printf '%s' "$count" > "$count_file"
            if [ "$count" -eq 1 ]; then
                exit 143
            fi
            printf 'Status: VERIFIED\nDomains explored: 7\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category)
    combined_output = result.stdout + result.stderr

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, category)
    assert row["schema_version"] == "backend_benchmark_row_v1", row
    assert row["lane"] == "vnncomp_single_backend", row
    assert row["status"] == "verified", f"expected retry path to recover a verdict, got {row}"
    assert row["domains_explored"] == "7", f"expected domains from recovered verdict, got {row}"
    assert row["backend"] == "cpu", f"expected default backend to record cpu, got {row}"
    assert (
        "RETRY[default]: ny exited with code 143 before reporting a verdict" in combined_output
    ), f"expected retry note in output, got stdout={result.stdout!r} stderr={result.stderr!r}"


def test_benchmark_vnncomp_disjunctive_output_uses_final_domains_summary(tmp_path: Path) -> None:
    category = "disjunctivecat"
    _write_category_fixture(tmp_path, category)
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: UNKNOWN\n'
            printf '    Domains explored: 5\n'
            printf '    Domains explored: 9\n'
            printf 'Domains explored: 17\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category)
    combined_output = result.stdout + result.stderr

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, category)
    assert row["status"] == "unknown", f"expected disjunctive fixture to stay unknown, got {row}"
    assert row["domains_explored"] == "17", f"expected final aggregate domains count, got {row}"
    assert "unknown" in result.stdout and "17 domains" in result.stdout, (
        f"expected stdout summary to use final aggregate domains, got: {result.stdout}"
    )


def test_benchmark_vnncomp_signal_exit_without_verdict_counts_as_timeout(tmp_path: Path) -> None:
    category = "timeoutcat"
    _write_category_fixture(tmp_path, category)
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            if [ "$1" = "--version" ]; then echo "ny 0.0.0-test"; exit 0; fi
            exit 143
            """
        ),
    )

    result = _run_benchmark(
        tmp_path, ny_path, category,
        extra_env={"EXTERNAL_TIMEOUT_SLACK": "1", "WATCHDOG_TERM_GRACE": "1"},
    )
    combined_output = result.stdout + result.stderr

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, category)
    assert row["status"] == "timeout", (
        f"expected unrecovered signal exit to count as timeout, got {row}"
    )
    assert (
        "NOTE[default]: external watchdog enforced timeout" in combined_output
        or "NOTE[default]: counting exit code 143 without a verdict as timeout" in combined_output
    ), f"expected timeout note in output, got stdout={result.stdout!r} stderr={result.stderr!r}"


def test_benchmark_vnncomp_external_timeout_without_verdict_counts_as_timeout(
    tmp_path: Path,
) -> None:
    category = "watchdogcat"
    _write_category_fixture(tmp_path, category)
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            if [ "$1" = "--version" ]; then echo "ny 0.0.0-test"; exit 0; fi
            trap 'exit 143' TERM
            sleep 10
            """
        ),
    )

    result = _run_benchmark(
        tmp_path,
        ny_path,
        category,
        extra_env={
            "EXTERNAL_TIMEOUT_SLACK": "1",
            "WATCHDOG_TERM_GRACE": "1",
        },
    )
    combined_output = result.stdout + result.stderr

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, category)
    assert row["status"] == "timeout", f"expected watchdog timeout to count as timeout, got {row}"
    assert (
        "NOTE[default]: external watchdog enforced timeout after 3s without a verdict"
        in combined_output
    ), f"expected watchdog note in output, got stdout={result.stdout!r} stderr={result.stderr!r}"


def test_benchmark_vnncomp_rejects_text_placeholder_onnx_assets(tmp_path: Path) -> None:
    category = "placeholdercat"
    _write_category_fixture(tmp_path, category)
    (
        tmp_path
        / "benchmarks"
        / "vnncomp2025"
        / "benchmarks"
        / category
        / "onnx"
        / "model.onnx"
    ).write_text(
        "--2025-07-15 19:24:46-- https://example.invalid/download\n",
        encoding="utf-8",
    )
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: VERIFIED\nDomains explored: 99\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category)
    combined_output = result.stdout + result.stderr

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, category)
    assert row["status"] == "error", f"expected text placeholder asset to be rejected, got {row}"
    assert "benchmark ONNX asset is not binary" in combined_output, (
        f"expected explicit asset-integrity error, got stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_benchmark_vnncomp_forwards_backend_flag(tmp_path: Path) -> None:
    category = "backendcat"
    _write_category_fixture(tmp_path, category)
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf '%s\n' "$@" > "$(dirname "$0")/argv.txt"
            printf 'Status: VERIFIED\nDomains explored: 3\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category, args=["--backend", "wgpu"])

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, category)
    assert row["status"] == "verified", f"expected backend fixture to verify, got {row}"
    assert row["backend"] == "wgpu", f"expected backend field to record wgpu, got {row}"
    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8")
    assert "--backend" in argv and "wgpu" in argv, (
        f"expected backend flag to be forwarded to ny, got argv: {argv}"
    )
    assert "Backend: --backend wgpu" in result.stdout, (
        f"expected stdout banner to report backend selection, got: {result.stdout}"
    )


def test_benchmark_vnncomp_vit_forwards_heuristic_softmax_flag(tmp_path: Path) -> None:
    _write_category_fixture(tmp_path, "vit_2023")
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf '%s\n' "$@" > "$(dirname "$0")/argv.txt"
            printf 'Status: VERIFIED\nDomains explored: 5\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, "vit_2023")

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    row = _load_single_result(tmp_path, "vit_2023")
    assert row["status"] == "verified", f"expected vit fixture to verify, got {row}"
    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert "--allow-heuristic-softmax" in argv, (
        f"expected vit_2023 benchmark to forward heuristic softmax flag, got argv: {argv}"
    )
    assert "--branching" in argv and "input" in argv, (
        f"expected vit_2023 benchmark to default to input branching, got argv: {argv}"
    )
    assert "Category flags: --allow-heuristic-softmax" in result.stdout, (
        f"expected vit_2023 banner to report heuristic softmax flag, got: {result.stdout}"
    )
    assert "Branching: --branching input" in result.stdout, (
        f"expected vit_2023 banner to report input branching, got: {result.stdout}"
    )


def test_benchmark_vnncomp_vit_preserves_explicit_branching_override(
    tmp_path: Path,
) -> None:
    _write_category_fixture(tmp_path, "vit_2023")
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf '%s\n' "$@" > "$(dirname "$0")/argv.txt"
            printf 'Status: VERIFIED\nDomains explored: 5\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, "vit_2023", args=["--branching", "relu"])

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
    branching_values = [
        argv[index + 1]
        for index, value in enumerate(argv[:-1])
        if value == "--branching"
    ]
    assert branching_values == ["relu"], (
        f"expected explicit branching override to win over vit default, got argv: {argv}"
    )
    assert "Branching: --branching relu" in result.stdout, (
        f"expected stdout banner to report override branching, got: {result.stdout}"
    )


def test_benchmark_vnncomp_compare_backends_emits_two_normalized_rows(tmp_path: Path) -> None:
    category = "comparecat"
    _write_category_fixture(tmp_path, category)
    ny_path = _write_compare_backends_ny(tmp_path)

    result = _run_benchmark(tmp_path, ny_path, category, args=["--compare-backends"])

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    rows = _load_report_rows(tmp_path, f"{category}_compare_backends_*.csv")
    assert len(rows) == 2, f"expected one cpu row and one wgpu row, got {rows}"

    by_backend = {row["backend"]: row for row in rows}
    assert set(by_backend) == {"cpu", "wgpu"}, f"expected cpu/wgpu rows, got {rows}"
    assert by_backend["cpu"]["schema_version"] == "backend_benchmark_row_v1", rows
    assert by_backend["cpu"]["lane"] == "vnncomp_compare_backends", rows
    assert by_backend["cpu"]["subject_kind"] == "vnncomp_instance", rows
    assert by_backend["cpu"]["subject_id"] == by_backend["wgpu"]["subject_id"], rows
    assert by_backend["cpu"]["comparison_key"] == by_backend["wgpu"]["comparison_key"], rows
    assert (
        by_backend["cpu"]["subject_id"]
        == "vnncomp2025::comparecat::row=1::onnx/model.onnx::vnnlib/prop.vnnlib"
    ), rows
    assert by_backend["cpu"]["status"] == "verified", rows
    assert by_backend["wgpu"]["status"] == "verified", rows
    assert by_backend["cpu"]["domains_explored"] == "3", rows
    assert by_backend["wgpu"]["domains_explored"] == "5", rows
    assert by_backend["cpu"]["model_path"].endswith("comparecat/onnx/model.onnx"), rows
    assert by_backend["cpu"]["property_path"].endswith("comparecat/vnnlib/prop.vnnlib"), rows
    assert "Backend-only status divergence: 0" in result.stdout, result.stdout


def test_benchmark_vnncomp_compare_backends_distinguishes_duplicate_source_rows(
    tmp_path: Path,
) -> None:
    category = "satdup"
    _write_category_fixture_rows(
        tmp_path,
        category,
        [
            ("onnx/model.onnx", "vnnlib/prop.vnnlib", "2"),
            ("onnx/model.onnx", "vnnlib/prop.vnnlib", "2"),
        ],
    )
    ny_path = _write_compare_backends_ny(tmp_path)

    result = _run_benchmark(
        tmp_path,
        ny_path,
        category,
        args=["--compare-backends", "--limit", "2"],
    )

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    rows = _load_report_rows(tmp_path, f"{category}_compare_backends_*.csv")
    assert len(rows) == 4, rows

    keys = {row["comparison_key"] for row in rows}
    assert keys == {
        "vnncomp2025::satdup::row=1::onnx/model.onnx::vnnlib/prop.vnnlib",
        "vnncomp2025::satdup::row=2::onnx/model.onnx::vnnlib/prop.vnnlib",
    }, rows
    for key in keys:
        backends = {row["backend"] for row in rows if row["comparison_key"] == key}
        assert backends == {"cpu", "wgpu"}, rows


def test_benchmark_vnncomp_compare_backends_distinguishes_basename_collisions(
    tmp_path: Path,
) -> None:
    category = "safenlpmini"
    _write_category_fixture_rows(
        tmp_path,
        category,
        [
            ("onnx/a/model.onnx", "vnnlib/a/prop.vnnlib", "2"),
            ("onnx/b/model.onnx", "vnnlib/b/prop.vnnlib", "2"),
        ],
    )
    ny_path = _write_compare_backends_ny(tmp_path, fixed_domains="1")

    result = _run_benchmark(
        tmp_path,
        ny_path,
        category,
        args=["--compare-backends", "--limit", "2"],
    )

    assert result.returncode == 0, f"benchmark script failed: {result.stderr}"
    rows = _load_report_rows(tmp_path, f"{category}_compare_backends_*.csv")
    assert len({row["comparison_key"] for row in rows}) == 2, rows
    assert {
        row["comparison_key"]
        for row in rows
        if row["backend"] == "cpu"
    } == {
        "vnncomp2025::safenlpmini::row=1::onnx/a/model.onnx::vnnlib/a/prop.vnnlib",
        "vnncomp2025::safenlpmini::row=2::onnx/b/model.onnx::vnnlib/b/prop.vnnlib",
    }, rows


def test_benchmark_vnncomp_compare_backends_distinguishes_cross_suite_collisions(
    tmp_path: Path,
) -> None:
    category = "nn4sys"
    _write_category_fixture(tmp_path, category, suite="vnncomp2023")
    _write_category_fixture(tmp_path, category, suite="vnncomp2025")
    ny_path = _write_compare_backends_ny(tmp_path, fixed_domains="2")

    result_2023 = _run_benchmark(
        tmp_path,
        ny_path,
        category,
        args=["--compare-backends"],
        bench_root=_benchmark_root(tmp_path, "vnncomp2023"),
    )
    rows_2023 = _load_report_rows(tmp_path, f"{category}_compare_backends_*.csv")
    key_2023 = rows_2023[0]["comparison_key"]

    result_2025 = _run_benchmark(
        tmp_path,
        ny_path,
        category,
        args=["--compare-backends"],
        bench_root=_benchmark_root(tmp_path, "vnncomp2025"),
    )
    rows_2025 = _load_report_rows(tmp_path, f"{category}_compare_backends_*.csv")
    key_2025 = rows_2025[0]["comparison_key"]

    assert result_2023.returncode == 0, f"benchmark script failed: {result_2023.stderr}"
    assert result_2025.returncode == 0, f"benchmark script failed: {result_2025.stderr}"
    assert key_2023 == "vnncomp2023::nn4sys::row=1::onnx/model.onnx::vnnlib/prop.vnnlib", (
        f"expected 2023 suite key in comparison identity, got {key_2023!r}"
    )
    assert key_2025 == "vnncomp2025::nn4sys::row=1::onnx/model.onnx::vnnlib/prop.vnnlib", (
        f"expected 2025 suite key in comparison identity, got {key_2025!r}"
    )
    assert key_2023 != key_2025, (
        f"expected suite provenance to keep cross-suite rows distinct, got {key_2023!r} and {key_2025!r}"
    )


def test_benchmark_vnncomp_skips_unmanifested_reference_validation(
    tmp_path: Path,
) -> None:
    category = "syntheticref"
    _write_category_fixture(tmp_path, category)
    _write_reference_csv(tmp_path, category, result="verified")
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: VIOLATED\nDomains explored: 1\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category)

    assert result.returncode == 0, (
        "unmanifested reference CSVs must not trigger automatic soundness failures:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "Skipping auto-validation" in result.stdout, result.stdout
    assert "syntheticref_alpha_beta_crown.csv" in result.stdout, result.stdout


def test_benchmark_vnncomp_validates_manifest_backed_reference_csv(
    tmp_path: Path,
) -> None:
    """The tracked 0644 validator runs via bash in a normal checkout."""
    category = "manifestref"
    _write_category_fixture(tmp_path, category)
    reference_path = _write_reference_csv(tmp_path, category, result="verified")
    _write_reference_manifest(tmp_path, category, reference_path)
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: VIOLATED\nDomains explored: 1\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category)

    assert result.returncode == 1, (
        "manifest-backed reference CSV should remain authoritative for automatic validation:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "CRITICAL: onnx/model / vnnlib/prop" in result.stdout, result.stdout
    assert "potential soundness bug" in result.stdout, result.stdout


def test_benchmark_vnncomp_warns_when_reference_manifest_is_invalid(
    tmp_path: Path,
) -> None:
    category = "invalidmanifest"
    _write_category_fixture(tmp_path, category)
    _write_reference_csv(tmp_path, category, result="verified")
    manifest_path = tmp_path / "reports" / "benchmarks" / "reference" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{invalid json\n", encoding="utf-8")
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: VIOLATED\nDomains explored: 1\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category)

    assert result.returncode == 0, (
        "invalid manifest should warn and skip auto-validation instead of misclassifying the reference path:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "Reference manifest is unreadable or invalid" in result.stderr, result.stderr
    assert "Skipping auto-validation against" not in result.stdout, result.stdout


def test_benchmark_vnncomp_rejects_manifest_paths_outside_reference_dir(
    tmp_path: Path,
) -> None:
    category = "outofboundsmanifest"
    _write_category_fixture(tmp_path, category)
    rogue_reference = tmp_path / "reports" / "benchmarks" / "rogue.csv"
    rogue_reference.parent.mkdir(parents=True, exist_ok=True)
    rogue_reference.write_text(
        "model,property,result\nmodel,prop,verified\n",
        encoding="utf-8",
    )
    _write_reference_manifest(tmp_path, category, "reports/benchmarks/rogue.csv")
    ny_path = _write_fake_ny(
        tmp_path,
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'Status: VIOLATED\nDomains explored: 1\n'
            """
        ),
    )

    result = _run_benchmark(tmp_path, ny_path, category)

    assert result.returncode == 0, (
        "manifest-backed validation must reject output paths outside the reference directory:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "invalid output_path provenance" in result.stderr, result.stderr
    assert "potential soundness bug" not in result.stdout, result.stdout
