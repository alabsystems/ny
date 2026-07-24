# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "20260718T120000Z-test"
AY_REV = "1560972ade2b04a702dfbd13a2de5444ea216009"


def _run(*command: str, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _init_git_repo(path: Path) -> None:
    _run("git", "init", "-q", cwd=path)
    _run("git", "config", "user.name", "NY Test", cwd=path)
    _run("git", "config", "user.email", "ny-test@example.invalid", cwd=path)


def _fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "ny"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / ".gitignore").write_text(
        "/target/\n/reports/\n/benchmarks/\n", encoding="utf-8"
    )
    (repo / "rust-toolchain.toml").write_text(
        '[toolchain]\nchannel = "1.95.0"\ncomponents = ["rustfmt", "clippy"]\n',
        encoding="utf-8",
    )
    (repo / "Cargo.lock").write_text(
        "[[package]]\n"
        'name = "ay-test"\n'
        'version = "0.1.0"\n'
        f'source = "git+https://github.com/alabsystems/ay.git?rev={AY_REV}'
        f'#{AY_REV}"\n',
        encoding="utf-8",
    )
    scripts = repo / "scripts"
    scripts.mkdir()
    for name in (
        "archive_vnncomp_sat_result.py",
        "measure_ny_scorecard.sh",
        "ny_measurement_provenance.py",
        "seal_ny_measurement_inputs.py",
    ):
        shutil.copyfile(REPO_ROOT / "scripts" / name, scripts / name)

    binary = repo / "target/release/ny"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = --version ]; then\n'
        "  echo 'ny 0.1.0-test'\n"
        "  exit 0\n"
        "fi\n"
        'if [ "${1:-}" = --build-info ]; then\n'
        "  echo 'ny 0.1.0-test cuda=on mip=on'\n"
        "  exit 0\n"
        "fi\n"
        'echo "solver combined log for $5"\n'
        "printf 'solver argv:'\n"
        "printf ' <%s>' \"$@\"\n"
        "printf '\\n'\n"
        'case "$5" in\n'
        "  */violated.vnnlib) printf 'sat\\n((X_0 0.25))\\n' > \"$6\" ;;\n"
        "  *) printf 'unsat\\n' > \"$6\" ;;\n"
        "esac\n"
        'if [ "${MEASUREMENT_TEST_MUTATE_SOLVER:-}" = 1 ]; then\n'
        '  chmod u+w "$0"\n'
        "  printf '\\n# measurement drift\\n' >> \"$0\"\n"
        "fi\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "fixture", cwd=repo)

    benchmark = repo / "benchmarks/vnncomp2026"
    benchmark.mkdir(parents=True)
    _init_git_repo(benchmark)
    instance_dir = benchmark / "benchmarks/demo/2.0"
    instance_dir.mkdir(parents=True)
    (instance_dir / "instances.csv").write_text(
        "shared.onnx,violated.vnnlib,1\nshared.onnx,holds.vnnlib,1\n",
        encoding="utf-8",
    )
    (instance_dir / "shared.onnx").write_bytes(b"shared model")
    (instance_dir / "violated.vnnlib").write_text("; violated\n", encoding="utf-8")
    (instance_dir / "holds.vnnlib").write_text("; holds\n", encoding="utf-8")
    _run("git", "add", ".", cwd=benchmark)
    _run("git", "commit", "-qm", "fixture", cwd=benchmark)
    _run(
        "git",
        "remote",
        "add",
        "origin",
        "https://example.invalid/vnncomp-benchmarks.git",
        cwd=benchmark,
    )
    return repo


def _measurement_environment(
    repo: Path,
    tmp_path: Path,
    **overrides: str,
) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("NY_")
    }
    environment.update(
        {
            "GITHUB_TOKEN": "must-not-be-recorded",
            "NY_BUILD_FEATURES": "mip,cuda",
            "NY_MEASURE_CAP": "2",
            "NY_MEASURE_CATS": "demo",
            "NY_MEASURE_RUN_ID": RUN_ID,
            "NY_MARGIN_ROW_CLASSWISE": "1",
            "NY_ROOT": str(repo),
            "NY_SCRATCH": str(tmp_path / "scratch"),
        }
    )
    environment.update(overrides)
    return environment


def test_sweep_binds_rows_and_sat_artifact_to_completed_run(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(repo, tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    output_root = repo / f"reports/measured-runs/{RUN_ID}"
    artifact_root = output_root / "artifacts"
    csv_path = output_root / "demo.csv"
    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert rows == [
        ["demo", "shared.onnx", "violated.vnnlib", "0", "sat", "0", RUN_ID],
        ["demo", "shared.onnx", "holds.vnnlib", "0", "unsat", "0", RUN_ID],
    ]

    run_dir = artifact_root / f"runs/{RUN_ID}"
    start_path = run_dir / "start.json"
    completion_path = run_dir / "completion.json"
    start_bytes = start_path.read_bytes()
    start = json.loads(start_bytes)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert start["measurement"]["output_dir"] == str(output_root.resolve())
    assert not (repo / "reports/measured/demo.csv").exists()
    assert start["solver_binary"]["declared_build_features"] == ["mip", "cuda"]
    sealed_solver = start["solver_binary"]["sealed_execution"]
    assert sealed_solver["path"] != start["solver_binary"]["path"]
    assert (
        Path(sealed_solver["path"]).read_bytes()
        == Path(start["solver_binary"]["path"]).read_bytes()
    )
    assert start["measurement"]["csv_columns"][-1] == "run_id"
    assert start["measurement"]["config_inputs"] is None
    assert "--configs-dir" not in start["measurement"]["solver_command_template"]
    expected_root_gemm = (
        "faer"
        if platform.system() == "Linux"
        and platform.machine().lower() in {"aarch64", "arm64"}
        else "ndarray"
    )
    assert start["environment"]["values"]["NY_ROOT_GEMM"] == expected_root_gemm
    assert start["environment"]["values"]["NY_MARGIN_ROW_CLASSWISE"] == "1"
    assert b"must-not-be-recorded" not in start_bytes
    assert completion["exit_status"] == 0
    assert completion["completed_successfully"] is True
    assert completion["integrity"]["status"] == "valid"
    assert completion["integrity"]["violations"] == []
    run_evidence = completion["integrity"]["checks"]["run_evidence"]
    assert run_evidence["status"] == "valid"
    assert run_evidence["metadata_count"] == 2
    assert run_evidence["result_count"] == 2
    assert run_evidence["solver_log_count"] == 2
    assert run_evidence["preflight_count"] == 2
    assert run_evidence["csv_row_count"] == 2
    assert run_evidence["validated_record_count"] == 2
    assert len(run_evidence["records_sha256"]) == 64
    assert len(run_evidence["csv_evidence_sha256"]) == 64
    assert (
        completion["start_manifest_sha256"] == hashlib.sha256(start_bytes).hexdigest()
    )
    cache_path = run_dir / "input_hash_cache.json"
    assert completion["input_hash_cache"]["present"] is True
    assert completion["input_hash_cache"]["entry_count"] == 3
    assert (
        completion["input_hash_cache"]["sha256"]
        == hashlib.sha256(cache_path.read_bytes()).hexdigest()
    )

    metadata_paths = sorted((artifact_root / "demo").glob(f"*/{RUN_ID}.json"))
    assert len(metadata_paths) == 2
    metadata_by_verdict = {
        metadata["solver_verdict"]: (metadata_path, metadata)
        for metadata_path in metadata_paths
        for metadata in [json.loads(metadata_path.read_text(encoding="utf-8"))]
    }
    assert set(metadata_by_verdict) == {"sat", "unsat"}
    for metadata_path, metadata in metadata_by_verdict.values():
        assert metadata["start_manifest"] == f"runs/{RUN_ID}/start.json"
        assert (
            metadata["start_manifest_sha256"] == hashlib.sha256(start_bytes).hexdigest()
        )
        assert (
            metadata["raw_result_sha256"]
            == hashlib.sha256(
                metadata_path.with_suffix(".results").read_bytes()
            ).hexdigest()
        )
        solver_log_path = artifact_root / metadata["solver_log"]["artifact"]
        assert solver_log_path.read_bytes().startswith(b"solver combined log for ")
        for label in ("onnx", "vnnlib"):
            sealed_input = metadata["execution_inputs"][label]
            assert sealed_input["resolved_path"] != metadata[label]["resolved_path"]
            assert (
                Path(sealed_input["resolved_path"]).read_bytes()
                == Path(metadata[label]["resolved_path"]).read_bytes()
            )
            assert f" <{sealed_input['resolved_path']}>".encode() in (
                solver_log_path.read_bytes()
            )
        preflight = metadata["input_preflight"]
        assert (
            preflight["sha256"]
            == hashlib.sha256(
                (artifact_root / preflight["artifact"]).read_bytes()
            ).hexdigest()
        )
        assert b"--configs-dir" not in solver_log_path.read_bytes()
        assert metadata["config_inputs"] is None
        assert (
            metadata["solver_log"]["sha256"]
            == hashlib.sha256(solver_log_path.read_bytes()).hexdigest()
        )

    sat_path, sat_metadata = metadata_by_verdict["sat"]
    assert sat_path.with_suffix(".results").read_text(encoding="utf-8") == (
        "sat\n((X_0 0.25))\n"
    )
    assert sat_metadata["counterexample_validation"]["status"] == "not_checked"
    assert sat_metadata["onnx"]["sha256"] == hashlib.sha256(b"shared model").hexdigest()
    assert sat_metadata["onnx"]["hash_cache_hit"] is False
    unsat_path, unsat_metadata = metadata_by_verdict["unsat"]
    assert unsat_path.with_suffix(".results").read_bytes() == b"unsat\n"
    assert unsat_metadata["counterexample_validation"]["status"] == ("not_applicable")
    assert unsat_metadata["onnx"]["hash_cache_hit"] is True
    assert (
        unsat_metadata["vnnlib"]["sha256"] == hashlib.sha256(b"; holds\n").hexdigest()
    )


def test_explicit_solver_binary_is_allowed_captured_and_sealed(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    external_binary = tmp_path / "private-target/release/ny"
    external_binary.parent.mkdir(parents=True)
    shutil.copy2(repo / "target/release/ny", external_binary)
    external_binary.write_bytes(external_binary.read_bytes() + b"\n# external build\n")
    external_binary.chmod(0o755)
    isolated = repo / "reports/external-solver"
    run_id = "20260718T123000Z-external-solver"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_BIN=str(external_binary),
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    start = json.loads(
        (isolated / f"artifacts/runs/{run_id}/start.json").read_text(
            encoding="utf-8"
        )
    )
    assert start["environment"]["values"]["NY_MEASURE_BIN"] == str(external_binary)
    assert start["solver_binary"]["path"] == str(external_binary.resolve())
    sealed_binary = Path(start["solver_binary"]["sealed_execution"]["path"])
    assert sealed_binary != external_binary
    assert sealed_binary.read_bytes() == external_binary.read_bytes()


def test_integrity_failure_propagates_nonzero_from_completion_trap(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    isolated = repo / "reports/integrity-drift"
    run_id = "20260718T125000Z-integrity-drift"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            MEASUREMENT_TEST_MUTATE_SOLVER="1",
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1, f"{result.stdout}\n{result.stderr}"
    assert "completion integrity validation failed" in result.stderr
    completion = json.loads(
        (isolated / f"artifacts/runs/{run_id}/completion.json").read_text(
            encoding="utf-8"
        )
    )
    codes = {item["code"] for item in completion["integrity"]["violations"]}
    assert completion["exit_status"] == 0
    assert completion["completed_successfully"] is False
    assert completion["integrity"]["status"] == "invalid"
    assert "sealed_solver_binary_sha256_mismatch" in codes


def test_completion_write_failure_propagates_nonzero_from_trap(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    isolated = repo / "reports/completion-collision"
    run_id = "20260718T125500Z-completion-collision"
    run_dir = isolated / f"artifacts/runs/{run_id}"
    run_dir.mkdir(parents=True)
    completion_path = run_dir / "completion.json"
    completion_path.write_text("pre-existing evidence\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1, f"{result.stdout}\n{result.stderr}"
    assert "completion or integrity validation failed" in result.stderr
    assert completion_path.read_text(encoding="utf-8") == "pre-existing evidence\n"
    assert (run_dir / "start.json").is_file()


def test_isolated_output_reruns_legacy_row_and_honors_row_cap(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    legacy_dir = repo / "reports/measured"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "demo.csv").write_text(
        "demo,shared.onnx,violated.vnnlib,0,sat,0\n", encoding="utf-8"
    )
    isolated = repo / "reports/isolated-bank"
    run_id = "20260718T130000Z-isolated"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    with (isolated / "demo.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert rows == [["demo", "shared.onnx", "violated.vnnlib", "0", "sat", "0", run_id]]
    assert len((legacy_dir / "demo.csv").read_text(encoding="utf-8").splitlines()) == 1

    start = json.loads(
        (isolated / f"artifacts/runs/{run_id}/start.json").read_text(encoding="utf-8")
    )
    assert start["measurement"]["output_dir"] == str(isolated.resolve())
    assert start["measurement"]["max_rows_per_category"] == 1
    assert start["environment"]["values"]["NY_MEASURE_OUTPUT_DIR"] == str(isolated)
    artifacts = list((isolated / "artifacts/demo").glob(f"*/{run_id}.json"))
    assert len(artifacts) == 1


def test_exact_instance_selector_is_provenanced_and_runs_only_that_row(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    isolated = repo / "reports/selected-instance"
    run_id = "20260718T140000Z-selected"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_INSTANCE_INDEX="2",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
            NY_MEASURE_VNNLIB_VERSION="2.0",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    with (isolated / "demo.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert rows == [["demo", "shared.onnx", "holds.vnnlib", "0", "unsat", "0", run_id]]

    start = json.loads(
        (isolated / f"artifacts/runs/{run_id}/start.json").read_text(encoding="utf-8")
    )
    assert start["measurement"]["instance_index"] == 2
    assert start["measurement"]["vnnlib_version_selection"] == "2.0"
    assert start["environment"]["values"]["NY_MEASURE_INSTANCE_INDEX"] == "2"
    assert start["environment"]["values"]["NY_MEASURE_VNNLIB_VERSION"] == "2.0"
    metadata_paths = list((isolated / "artifacts/demo").glob(f"*/{run_id}.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    assert metadata["instance_index"] == 2


def test_multiple_vnnlib_versions_require_explicit_selection(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    benchmark = repo / "benchmarks/vnncomp2026/benchmarks/demo"
    shutil.copytree(benchmark / "2.0", benchmark / "1.0")
    isolated = repo / "reports/ambiguous-version"
    run_id = "20260718T150000Z-ambiguous"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert "multiple instances.csv files" in result.stderr
    assert "NY_MEASURE_VNNLIB_VERSION" in result.stderr
    assert not (isolated / "demo.csv").exists()

    completion = json.loads(
        (isolated / f"artifacts/runs/{run_id}/completion.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion["exit_status"] == 1
    assert completion["completed_successfully"] is False


def test_missing_input_fails_without_emitting_unarchived_csv_row(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    (repo / "benchmarks/vnncomp2026/benchmarks/demo/2.0/shared.onnx").unlink()
    isolated = repo / "reports/missing-input"
    run_id = "20260718T155000Z-missing"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "refusing to record an unarchived row with missing inputs" in result.stderr
    assert (isolated / "demo.csv").read_bytes() == b""
    completion = json.loads(
        (isolated / f"artifacts/runs/{run_id}/completion.json").read_text(
            encoding="utf-8"
        )
    )
    assert completion["exit_status"] == 1
    assert completion["completed_successfully"] is False


def test_external_configs_dir_is_passed_and_content_addressed(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    configs = tmp_path / "experiment-configs"
    preset_dir = configs / "vnncomp2025"
    preset_dir.mkdir(parents=True)
    preset = preset_dir / "demo.yaml"
    preset.write_text("verifier:\n  timeout: 2\n", encoding="utf-8")
    (configs / "README.txt").write_text("experiment A\n", encoding="utf-8")
    isolated = repo / "reports/external-configs"
    run_id = "20260718T160000Z-configs"

    result = subprocess.run(
        ["bash", "scripts/measure_ny_scorecard.sh"],
        cwd=repo,
        env=_measurement_environment(
            repo,
            tmp_path,
            NY_MEASURE_CONFIGS_DIR=str(configs),
            NY_MEASURE_MAX_ROWS_PER_CATEGORY="1",
            NY_MEASURE_OUTPUT_DIR=str(isolated),
            NY_MEASURE_RUN_ID=run_id,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    start_path = isolated / f"artifacts/runs/{run_id}/start.json"
    start = json.loads(start_path.read_text(encoding="utf-8"))
    config_inputs = start["measurement"]["config_inputs"]
    assert start["environment"]["values"]["NY_MEASURE_CONFIGS_DIR"] == str(configs)
    assert config_inputs["declared_path"] == str(configs)
    assert config_inputs["resolved_path"] == str(configs.resolve())
    assert config_inputs["entry_count"] == 3
    assert config_inputs["entries"] == [
        {
            "kind": "file",
            "path": "README.txt",
            "sha256": hashlib.sha256(b"experiment A\n").hexdigest(),
            "size_bytes": len(b"experiment A\n"),
        },
        {"kind": "directory", "path": "vnncomp2025"},
        {
            "kind": "file",
            "path": "vnncomp2025/demo.yaml",
            "sha256": hashlib.sha256(preset.read_bytes()).hexdigest(),
            "size_bytes": len(preset.read_bytes()),
        },
    ]
    manifest = {
        "schema": "ny_measurement_config_inputs_v1",
        "entries": config_inputs["entries"],
    }
    expected_manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    assert (
        config_inputs["manifest_sha256"]
        == hashlib.sha256(expected_manifest_bytes).hexdigest()
    )
    sealed_configs = start["measurement"]["sealed_config_inputs"]
    assert start["measurement"]["solver_command_template"][-2:] == [
        "--configs-dir",
        sealed_configs["resolved_path"],
    ]

    metadata_path = next((isolated / "artifacts/demo").glob(f"*/{run_id}.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["config_inputs"] == {
        key: config_inputs[key]
        for key in (
            "schema",
            "declared_path",
            "resolved_path",
            "entry_count",
            "manifest_sha256",
        )
    }
    solver_log = (
        isolated / "artifacts" / metadata["solver_log"]["artifact"]
    ).read_text(encoding="utf-8")
    assert f" <--configs-dir> <{sealed_configs['resolved_path']}>" in solver_log


def test_external_configs_dir_rejects_relative_or_missing_path(tmp_path: Path) -> None:
    for suffix, configs_dir, message in (
        ("relative", "configs", "must be an absolute path"),
        ("missing", str(tmp_path / "missing-configs"), "not an existing directory"),
    ):
        fixture_root = tmp_path / suffix
        fixture_root.mkdir()
        repo = _fixture_repo(fixture_root)
        isolated = repo / "reports/rejected-configs"
        run_id = f"20260718T170000Z-{suffix}"
        result = subprocess.run(
            ["bash", "scripts/measure_ny_scorecard.sh"],
            cwd=repo,
            env=_measurement_environment(
                repo,
                fixture_root,
                NY_MEASURE_CONFIGS_DIR=configs_dir,
                NY_MEASURE_OUTPUT_DIR=str(isolated),
                NY_MEASURE_RUN_ID=run_id,
            ),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 2
        assert message in result.stderr
        assert not isolated.exists()
