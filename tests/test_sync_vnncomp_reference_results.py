# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_vnncomp_reference_results.sh"
GENERATE_SCRIPT = REPO_ROOT / "scripts" / "generate_vnncomp_reference.sh"


def _write_harness_results(
    repo_root: Path,
    category: str,
    rows: list[tuple[str, str, str]],
) -> None:
    results_dir = repo_root / "alpha_beta_crown" / category
    results_dir.mkdir(parents=True, exist_ok=True)
    lines = ["category,onnx_path,vnnlib_path,prepare_runtime,result,runtime"]
    for onnx_path, vnnlib_path, result in rows:
        lines.append(f"{category},{onnx_path},{vnnlib_path},0.01,{result},1.0")
    (results_dir / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_sync_harness(tmp_path: Path) -> Path:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    sync_copy = scripts_dir / "sync_vnncomp_reference_results.sh"
    generate_copy = scripts_dir / "generate_vnncomp_reference.sh"
    shutil.copy2(SYNC_SCRIPT, sync_copy)
    shutil.copy2(GENERATE_SCRIPT, generate_copy)

    sync_copy.chmod(0o755)
    generate_copy.chmod(0o755)
    return sync_copy


def _init_git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "NY tests"],
        check=True,
    )
    (path / ".provenance-sentinel").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "fixture"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_sync_vnncomp_reference_results_generates_manifest_and_replaces_stale_csvs(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "vnncomp2025_results-ref"
    _write_harness_results(
        source_root,
        "acasxu_2023",
        [("models/a.onnx", "props/p0.vnnlib", "unsat")],
    )
    _write_harness_results(
        source_root,
        "malbeware",
        [
            ("models/b.onnx", "props/p1.vnnlib", "sat"),
            ("models/c.onnx.gz", "props/p2.vnnlib", "unknown"),
        ],
    )

    sync_script = _install_sync_harness(tmp_path)
    reference_dir = tmp_path / "reports" / "benchmarks" / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    stale_reference = reference_dir / "stale_alpha_beta_crown.csv"
    stale_reference.write_text("model,property,result\nstale,prop,verified\n", encoding="utf-8")
    _init_git_repo(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(sync_script),
            "--repo-root",
            str(source_root),
            "--tool",
            "alpha_beta_crown",
            "--year",
            "2025",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, f"sync script failed: {result.stderr}\n{result.stdout}"
    assert not stale_reference.exists(), "stale generated reference CSV should be replaced"

    acas_ref = reference_dir / "acasxu_2023_alpha_beta_crown.csv"
    mal_ref = reference_dir / "malbeware_alpha_beta_crown.csv"
    assert acas_ref.exists(), "expected ACAS reference CSV"
    assert mal_ref.exists(), "expected malbeware reference CSV"

    manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool"] == "alpha_beta_crown", manifest
    assert manifest["year"] == 2025, manifest
    assert manifest["categories"] == ["acasxu_2023", "malbeware"], manifest
    assert manifest["source_repo_root"] == str(source_root), manifest
    assert manifest["source_commit"] is None, manifest
    assert manifest["source_dirty"] is None, manifest
    assert manifest["reference_files"]["acasxu_2023"]["instance_count"] == 1, manifest
    assert manifest["reference_files"]["malbeware"]["instance_count"] == 2, manifest
    assert manifest["reference_files"]["malbeware"]["output_path"] == str(
        Path("reports/benchmarks/reference/malbeware_alpha_beta_crown.csv"),
    ), manifest
    # Benchmark-asset provenance fields (#4416)
    assert "benchmark_repo_root" in manifest, f"missing benchmark_repo_root: {manifest}"
    assert "benchmark_commit" in manifest, f"missing benchmark_commit: {manifest}"
    assert "benchmark_dirty" in manifest, f"missing benchmark_dirty: {manifest}"


def test_sync_benchmark_root_option_records_provenance(tmp_path: Path) -> None:
    """--benchmark-root option populates benchmark_repo_root in manifest."""
    source_root = tmp_path / "vnncomp2025_results-ref"
    benchmark_root = tmp_path / "vnncomp2025_benchmarks-ref"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    _init_git_repo(tmp_path)

    _write_harness_results(
        source_root,
        "acasxu_2023",
        [("models/a.onnx", "props/p0.vnnlib", "unsat")],
    )

    sync_script = _install_sync_harness(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(sync_script),
            "--repo-root",
            str(source_root),
            "--benchmark-root",
            str(benchmark_root),
            "--tool",
            "alpha_beta_crown",
            "--year",
            "2025",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, f"sync script failed: {result.stderr}\n{result.stdout}"

    reference_dir = tmp_path / "reports" / "benchmarks" / "reference"
    manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark_repo_root"] == str(benchmark_root), manifest
    # Not a git repo, so commit should be null
    assert manifest["benchmark_commit"] is None, manifest
    assert manifest["benchmark_dirty"] is None, manifest


def test_sync_records_commits_only_for_exact_checkout_roots(tmp_path: Path) -> None:
    source_root = tmp_path / "vnncomp2025_results-ref"
    benchmark_root = tmp_path / "vnncomp2025_benchmarks-ref"
    _write_harness_results(
        source_root,
        "acasxu_2023",
        [("models/a.onnx", "props/p0.vnnlib", "unsat")],
    )
    source_commit = _init_git_repo(source_root)
    benchmark_commit = _init_git_repo(benchmark_root)
    sync_script = _install_sync_harness(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(sync_script),
            "--repo-root",
            str(source_root),
            "--benchmark-root",
            str(benchmark_root),
            "--tool",
            "alpha_beta_crown",
            "--year",
            "2025",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, f"sync script failed: {result.stderr}\n{result.stdout}"
    manifest = json.loads(
        (tmp_path / "reports/benchmarks/reference/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["source_commit"] == source_commit, manifest
    assert manifest["source_dirty"] is False, manifest
    assert manifest["benchmark_commit"] == benchmark_commit, manifest
    assert manifest["benchmark_dirty"] is False, manifest


def test_sync_dirty_checkout_never_claims_an_exact_source_commit(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "vnncomp2025_results-ref"
    benchmark_root = tmp_path / "vnncomp2025_benchmarks-ref"
    _write_harness_results(
        source_root,
        "acasxu_2023",
        [("models/a.onnx", "props/p0.vnnlib", "unsat")],
    )
    _init_git_repo(source_root)
    benchmark_commit = _init_git_repo(benchmark_root)
    results_csv = source_root / "alpha_beta_crown/acasxu_2023/results.csv"
    results_csv.write_text(
        results_csv.read_text(encoding="utf-8")
        + "acasxu_2023,models/b.onnx,props/p1.vnnlib,0.01,sat,1.0\n",
        encoding="utf-8",
    )
    sync_script = _install_sync_harness(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(sync_script),
            "--repo-root",
            str(source_root),
            "--benchmark-root",
            str(benchmark_root),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (tmp_path / "reports/benchmarks/reference/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["source_commit"] is None, manifest
    assert manifest["source_dirty"] is True, manifest
    assert manifest["benchmark_commit"] == benchmark_commit, manifest
    assert manifest["benchmark_dirty"] is False, manifest


def test_sync_conversion_failure_preserves_complete_live_snapshot(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "vnncomp2025_results-ref"
    _write_harness_results(
        source_root,
        "a_valid",
        [("models/a.onnx", "props/p0.vnnlib", "unsat")],
    )
    broken_results = source_root / "alpha_beta_crown/z_broken/results.csv"
    broken_results.parent.mkdir(parents=True)
    broken_results.write_text(
        "category,onnx_path,vnnlib_path,prepare_runtime,result,runtime\n",
        encoding="utf-8",
    )
    sync_script = _install_sync_harness(tmp_path)
    reference_dir = tmp_path / "reports/benchmarks/reference"
    reference_dir.mkdir(parents=True)
    old_csv = reference_dir / "old_alpha_beta_crown.csv"
    old_manifest = reference_dir / "manifest.json"
    old_csv_bytes = b"model,property,result\nold,prop,verified\n"
    old_manifest_bytes = b'{"generation":"old-complete-snapshot"}\n'
    old_csv.write_bytes(old_csv_bytes)
    old_manifest.write_bytes(old_manifest_bytes)

    result = subprocess.run(
        [
            "bash",
            str(sync_script),
            "--repo-root",
            str(source_root),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode != 0, result.stdout
    assert old_csv.read_bytes() == old_csv_bytes
    assert old_manifest.read_bytes() == old_manifest_bytes
    assert sorted(path.name for path in reference_dir.iterdir()) == [
        "manifest.json",
        "old_alpha_beta_crown.csv",
    ]
