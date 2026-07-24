# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ny_measurement_provenance.py"
ARCHIVE_SCRIPT = REPO_ROOT / "scripts" / "archive_vnncomp_sat_result.py"
AY_REV = "1560972ade2b04a702dfbd13a2de5444ea216009"
HOST_STATE_FIXTURE = {
    "schema": "ny_measurement_host_state_v1",
    "captured_at_utc": "2026-07-18T00:00:00.000000Z",
    "load_average": {},
    "processes": {},
    "gpu": {},
}
CGAN_GENERATOR_BRANCH_BOOLEAN_ENV = (
    "NY_BRANCH_STEM",
    "NY_BRANCH_STEM_PROBE",
    "NY_BRANCH_TRACE",
    "NY_MO_BAB_TRACE",
    "NY_UNSTABLE_COUNT",
)
CGAN_GENERATOR_BRANCH_ENV = (
    *CGAN_GENERATOR_BRANCH_BOOLEAN_ENV,
    "NY_BRANCH_STEM_K",
    "NY_BRANCH_STEM_NODES",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("ny_measurement_provenance", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


provenance = _load_module()


def _load_archive_module():
    spec = importlib.util.spec_from_file_location(
        "archive_vnncomp_sat_result_for_provenance_tests", ARCHIVE_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


archive = _load_archive_module()


def _run(*command: str, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _init_git_repo(path: Path) -> None:
    _run("git", "init", "-q", cwd=path)
    _run("git", "config", "user.name", "NY Test", cwd=path)
    _run("git", "config", "user.email", "ny-test@example.invalid", cwd=path)


def _measurement_repos(tmp_path: Path) -> tuple[Path, Path]:
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
    sweep = scripts / "measure_ny_scorecard.sh"
    sweep.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    binary = repo / "target" / "release" / "ny"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho 'ny 0.1.0-test'\n", encoding="utf-8")
    binary.chmod(0o755)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "fixture", cwd=repo)

    benchmark = tmp_path / "vnncomp-benchmarks"
    benchmark.mkdir()
    _init_git_repo(benchmark)
    benchmark_root = benchmark / "benchmarks"
    benchmark_root.mkdir()
    (benchmark_root / "README").write_text("fixture\n", encoding="utf-8")
    _run("git", "add", ".", cwd=benchmark)
    _run("git", "commit", "-qm", "fixture", cwd=benchmark)
    _run(
        "git",
        "remote",
        "add",
        "origin",
        "https://user:supersecret@github.com/example/bench.git?token=hidden",
        cwd=benchmark,
    )
    return repo, benchmark_root


def _clear_ny_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(("NY_", "AY_", "MIMALLOC_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("GPU_AVAILABLE", raising=False)
    monkeypatch.setenv("NY_BUILD_FEATURES", "mip,cuda")


def _capture(
    repo: Path,
    benchmark_root: Path,
    *,
    run_id: str,
    configs_dir: Path | None = None,
) -> Path:
    return provenance.capture_start_manifest(
        repo_root=repo,
        binary=Path("target/release/ny"),
        benchmark_root=benchmark_root,
        artifact_root=Path("reports/measured/artifacts"),
        run_id=run_id,
        output_dir=Path("reports/measured"),
        scratch_dir=repo.parent / "scratch",
        result_file=repo.parent / "scratch" / "result.txt",
        solver_log_file=repo.parent / "scratch" / "solver.log",
        categories_raw="demo other_demo",
        timeout_cap_seconds=120,
        watchdog_grace_seconds=30,
        max_rows_per_category=0,
        instance_index=0,
        vnnlib_version="",
        sweep_script=Path("scripts/measure_ny_scorecard.sh"),
        configs_dir=configs_dir,
    )


def _seed_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
) -> tuple[Path, Path, Path, Path, Path]:
    repo, benchmark_root = _measurement_repos(tmp_path)
    onnx_file = benchmark_root / "model.onnx"
    vnnlib_file = benchmark_root / "property.vnnlib"
    onnx_file.write_bytes(b"model fixture\n")
    vnnlib_file.write_text("; property fixture\n", encoding="utf-8")
    _run("git", "add", ".", cwd=benchmark_root.parent)
    _run("git", "commit", "-qm", "add measurement inputs", cwd=benchmark_root.parent)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id=run_id)
    result_file = tmp_path / "result.txt"
    solver_log_file = tmp_path / "solver.log"
    result_file.write_bytes(b"unsat\n")
    solver_log_file.write_bytes(b"solver log\n")
    artifact_root = repo / "reports/measured/artifacts"
    preflight_manifest = archive.seal_inputs(
        artifact_root=artifact_root,
        run_id=run_id,
        category="demo",
        instance_index=1,
        onnx="model.onnx",
        vnnlib="property.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        start_manifest=start,
    )
    archived_result = archive.archive_result(
        result_file=result_file,
        solver_log_file=solver_log_file,
        artifact_root=artifact_root,
        run_id=run_id,
        category="demo",
        instance_index=1,
        onnx="model.onnx",
        vnnlib="property.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        solver_verdict="unsat",
        solver_exit_status=0,
        timeout_seconds=120,
        elapsed_seconds=9,
        source_csv="reports/measured/demo.csv",
        start_manifest=start,
        preflight_manifest=preflight_manifest,
    )
    csv_path = repo / "reports/measured/demo.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        f"demo,model.onnx,property.vnnlib,0,unsat,9,{run_id}\n",
        encoding="utf-8",
    )
    metadata_path = archived_result.parent / f"{run_id}.json"
    solver_log_path = archived_result.parent / f"{run_id}.solver.log"
    cache_path = start.with_name("input_hash_cache.json")
    return start, metadata_path, archived_result, solver_log_path, cache_path


def test_start_manifest_hashes_worktree_and_excludes_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    (repo / "notes.txt").write_text("untracked evidence\n", encoding="utf-8")
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MEASURE_CAP", "120")
    monkeypatch.setenv("NY_MARGIN_ROW_ADAPTIVE_RESERVE", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_DOMAIN_STACK", "1")
    monkeypatch.setenv("NY_MARGIN_ROW_CLASSWISE", "1")
    monkeypatch.setenv("NY_ROOT_GEMM", "faer")
    monkeypatch.setenv("NY_WARMUP_ITERS", "03")
    monkeypatch.setenv("NY_CROWN_CUT_SEGMENT", "0")
    monkeypatch.setenv("NY_INPUT_SPLIT_WARM_PARALLEL", "1")
    monkeypatch.setenv("GITHUB_TOKEN", "must-never-appear")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest_path = _capture(repo, benchmark_root, run_id="run-one")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)

    assert manifest["schema"] == "ny_measurement_start_v1"
    assert manifest["ny"]["clean"] is False
    assert manifest["ny"]["untracked_files"] == [
        {
            "kind": "file",
            "path": "notes.txt",
            "sha256": provenance._sha256(b"untracked evidence\n"),
            "size_bytes": len(b"untracked evidence\n"),
        }
    ]
    assert len(manifest["ny"]["worktree_evidence_sha256"]) == 64
    assert manifest["solver_binary"]["sha256"] == provenance._sha256_file(
        repo / "target/release/ny"
    )
    assert manifest["solver_binary"]["declared_build_features"] == ["mip", "cuda"]
    assert manifest["dependencies"]["ay"]["git_revision"] == AY_REV
    assert manifest["rust_toolchain"]["channel"] == "1.95.0"
    assert manifest["benchmark"]["remotes"] == [
        {
            "name": "origin",
            "fetch_url": "https://github.com/example/bench.git",
        }
    ]
    assert manifest["measurement"]["timeout_cap_seconds"] == 120
    assert manifest["measurement"]["categories"] == ["demo", "other_demo"]
    assert manifest["environment"]["values"]["NY_MEASURE_CAP"] == "120"
    assert manifest["environment"]["values"]["NY_MARGIN_ROW_ADAPTIVE_RESERVE"] == "1"
    assert manifest["environment"]["values"]["NY_MARGIN_ROW_DOMAIN_STACK"] == "1"
    assert manifest["environment"]["values"]["NY_MARGIN_ROW_CLASSWISE"] == "1"
    assert manifest["environment"]["values"]["NY_ROOT_GEMM"] == "faer"
    assert manifest["environment"]["values"]["NY_WARMUP_ITERS"] == "03"
    assert manifest["environment"]["values"]["NY_CROWN_CUT_SEGMENT"] == "0"
    assert manifest["environment"]["values"]["NY_INPUT_SPLIT_WARM_PARALLEL"] == "1"
    assert manifest["environment"]["typed_values"] == {
        "NY_CROWN_CUT_SEGMENT": {
            "type": "nonnegative_integer",
            "value": 0,
        },
        "NY_WARMUP_ITERS": {
            "type": "nonnegative_integer",
            "value": 3,
        },
    }
    assert manifest["host_state"] == HOST_STATE_FIXTURE
    assert b"must-never-appear" not in manifest_bytes
    assert b"supersecret" not in manifest_bytes
    assert b"token=hidden" not in manifest_bytes


def test_untracked_content_changes_worktree_evidence_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    note = repo / "notes.txt"
    note.write_text("one\n", encoding="utf-8")
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    first = json.loads(_capture(repo, benchmark_root, run_id="run-one").read_text())
    note.write_text("two\n", encoding="utf-8")
    second = json.loads(_capture(repo, benchmark_root, run_id="run-two").read_text())

    assert (
        first["ny"]["worktree_evidence_sha256"]
        != second["ny"]["worktree_evidence_sha256"]
    )


def test_start_and_completion_records_are_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id="immutable-run")

    with pytest.raises(FileExistsError, match="immutable evidence"):
        _capture(repo, benchmark_root, run_id="immutable-run")

    completion = provenance.create_completion(start_manifest=start, exit_status=143)
    record = json.loads(completion.read_text(encoding="utf-8"))
    assert record["exit_status"] == 143
    assert record["completed_successfully"] is False
    assert record["start_manifest_sha256"] == provenance._sha256(start.read_bytes())
    assert record["integrity"]["status"] == "valid"
    assert record["integrity"]["violations"] == []
    assert record["host_state"] == HOST_STATE_FIXTURE
    with pytest.raises(FileExistsError, match="immutable evidence"):
        provenance.create_completion(start_manifest=start, exit_status=0)


@pytest.mark.parametrize(
    "key",
    [
        "NY_CROWN_CUT_SEGMENT",
        "NY_IMB_OBJ",
        "NY_IMB_REGION_K",
        "NY_IMB_REPLAY_ONLY_LEAF",
        "NY_WARMUP_ITERS",
    ],
)
def test_typed_numeric_environment_rejects_non_numeric_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "not-a-number")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=key):
        _capture(repo, benchmark_root, run_id=f"invalid-{key.lower()}")


def test_positive_usize_environment_is_absent_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    for key in (
        "NY_BUILD_MIN_FREE_KIB",
        "NY_CUDA_WIDE_MAX_BYTES",
        "NY_GPU_LOCK_WAIT_SECS",
        "NY_GPU_VMEM_LIMIT_KIB",
        "NY_MO_GPU_CHUNK",
    ):
        assert key not in environment["values"]
        assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NY_BUILD_MIN_FREE_KIB", "33554432", 33554432),
        ("NY_MO_GPU_CHUNK", "128", 128),
        ("NY_MO_GPU_CHUNK", "00128", 128),
        ("NY_CUDA_WIDE_MAX_BYTES", "2147483648", 2147483648),
        ("NY_CUDA_WIDE_MAX_BYTES", "0000000001", 1),
        ("NY_GPU_LOCK_WAIT_SECS", "03600", 3600),
        ("NY_GPU_VMEM_LIMIT_KIB", "83886080", 83886080),
    ],
)
def test_positive_usize_environment_preserves_raw_and_captures_typed_value(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "positive_integer",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "0", "+64", "-1", " 64", "64 ", "64.0"])
def test_mo_gpu_chunk_invalid_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_MO_GPU_CHUNK", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_MO_GPU_CHUNK"):
        provenance._capture_environment()


@pytest.mark.parametrize("key", ["NY_MO_GPU_CHUNK", "NY_CUDA_WIDE_MAX_BYTES"])
def test_positive_usize_environment_accepts_native_maximum(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    usize_max = (sys.maxsize * 2) + 1
    monkeypatch.setenv(key, str(usize_max))

    environment = provenance._capture_environment()

    assert environment["typed_values"][key] == {
        "type": "positive_integer",
        "value": usize_max,
    }


@pytest.mark.parametrize("key", ["NY_MO_GPU_CHUNK", "NY_CUDA_WIDE_MAX_BYTES"])
def test_positive_usize_environment_rejects_zero_and_overflow(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "0")
    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()

    usize_overflow = (sys.maxsize * 2) + 2
    monkeypatch.setenv(key, str(usize_overflow))
    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_mo_gpu_chunk_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_MO_GPU_CHUNK_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "64")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_guard_path_environment_is_captured_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    paths = {
        "NY_BUILD_DISK_PATH": "/private/build-target",
        "NY_GPU_LOCK_PATH": "/private/runtime/ny.lock",
    }
    for key, value in paths.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    for key, value in paths.items():
        assert environment["values"][key] == value
        assert key not in environment["typed_values"]


def test_kfsb_winner_oracle_environment_is_captured_and_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    raw_values = {
        "NY_BRANCH_KFSB_CHILDSIM": "1",
        "NY_KFSB_LAYER_QUOTA": "0",
        "NY_MO_ADAPTIVE_DEPTH_SELECT": "1",
        "NY_MO_ADAPTIVE_DEPTH_SHADOW": "1",
        "NY_MO_KFSB": "1",
        "NY_MO_KFSB_CACHED_LA": "1",
        "NY_MO_KFSB_CHUNK": "00064",
        "NY_MO_KFSB_K": "07",
        "NY_MO_KFSB_PROBE": "1",
        "NY_MO_KFSB_REDUCE": "max",
        "NY_MO_KFSB_WINNER_PROBE": "1",
        "NY_MO_KFSB_WINNER_PROBE_DOMAINS": "03",
    }
    for key, value in raw_values.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    for key, value in raw_values.items():
        assert environment["values"][key] == value
    assert environment["typed_values"] == {
        "NY_MO_KFSB_CHUNK": {"type": "positive_integer", "value": 64},
        "NY_MO_KFSB_K": {"type": "nonnegative_integer", "value": 7},
        "NY_MO_KFSB_WINNER_PROBE_DOMAINS": {
            "type": "positive_integer",
            "value": 3,
        },
    }


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("NY_MO_KFSB_K", "-1"),
        ("NY_MO_KFSB_CHUNK", "0"),
        ("NY_MO_KFSB_WINNER_PROBE_DOMAINS", "not-a-number"),
    ],
)
def test_kfsb_winner_oracle_invalid_numeric_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


def test_rel_bab_deadline_multiplier_is_captured_raw_and_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_REL_BAB_DEADLINE_MULT", "+01.400e0")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="rel-bab-mult-raw").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_REL_BAB_DEADLINE_MULT"] == "+01.400e0"
    assert environment["typed_values"]["NY_REL_BAB_DEADLINE_MULT"] == {
        "type": "bounded_float",
        "value": 1.4,
        "minimum": 1.0,
        "maximum": 10.0,
    }


@pytest.mark.parametrize(
    "raw", ["not-a-number", "NaN", "inf", "0.999", "10.001", " 1.4 "]
)
def test_rel_bab_deadline_multiplier_invalid_value_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_REL_BAB_DEADLINE_MULT", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="NY_REL_BAB_DEADLINE_MULT"):
        _capture(repo, benchmark_root, run_id="rel-bab-mult-invalid")

    assert not (
        repo / "reports/measured/artifacts/runs/rel-bab-mult-invalid/start.json"
    ).exists()


def test_rel_bab_deadline_multiplier_unknown_neighbor_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_REL_BAB_DEADLINE_MULT_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1.4")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        _capture(repo, benchmark_root, run_id="rel-bab-mult-unknown")

    assert not (
        repo / "reports/measured/artifacts/runs/rel-bab-mult-unknown/start.json"
    ).exists()


def test_relational_mip_controls_are_captured_as_raw_launch_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    controls = {
        "NY_REL_DIFF_COUPLING": "1",
        "NY_REL_JOINT_RELU_CUTS": "1",
        "NY_REL_JOINT_RELU_CUTS_SUM": "0",
        "NY_REL_WHOLE_MIP_OBBT": "1",
        "NY_REL_WHOLE_MIP_OBBT_CHUNK": "08",
        "NY_REL_WHOLE_MIP_OBBT_COND": "1",
        "NY_REL_WHOLE_MIP_OBBT_COND_FRAC": "0.45",
        "NY_REL_WHOLE_MIP_OBBT_MAXN": "192",
        "NY_REL_WHOLE_MIP_OBBT_OUTER": "2",
        "NY_REL_WHOLE_MIP_OBBT_ROUNDS": "3",
        "NY_REL_WHOLE_MIP_OBBT_S": "12.5",
        "NY_REL_WHOLE_MIP_OBBT_WIDTH": "1000.0",
    }
    for key, value in controls.items():
        monkeypatch.setenv(key, value)

    environment = provenance._capture_environment()

    assert {key: environment["values"][key] for key in controls} == controls
    assert not controls.keys() & environment["typed_values"].keys()


@pytest.mark.parametrize(
    "key",
    ["NY_JOINT_MEAS_NCOLS", "NY_JOINT_MEAS_PERSOLVE_S", "NY_JOINT_MEAS_DEADLINE_S"],
)
def test_ignored_joint_cut_test_knobs_remain_outside_solver_measurement_authority(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "1")

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", False), ("0", False), ("1", True)],
)
def test_gpu_available_is_captured_raw_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("GPU_AVAILABLE", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="gpu-available-raw").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["GPU_AVAILABLE"] == raw
    assert environment["typed_values"]["GPU_AVAILABLE"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["2", "00", "yes", "-1", " 1"])
def test_gpu_available_non_boolean_value_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("GPU_AVAILABLE", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="GPU_AVAILABLE"):
        _capture(repo, benchmark_root, run_id="gpu-available-invalid")

    assert not (
        repo / "reports/measured/artifacts/runs/gpu-available-invalid/start.json"
    ).exists()


def test_cuda_dgemm_triplet_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_CUDA_DGEMM_TRIPLET" not in environment["values"]
    assert "NY_CUDA_DGEMM_TRIPLET" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cuda_dgemm_triplet_is_captured_raw_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUDA_DGEMM_TRIPLET", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id=f"cuda-dgemm-triplet-{raw}").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_CUDA_DGEMM_TRIPLET"] == raw
    assert environment["typed_values"]["NY_CUDA_DGEMM_TRIPLET"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_cuda_dgemm_triplet_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CUDA_DGEMM_TRIPLET", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_CUDA_DGEMM_TRIPLET"):
        provenance._capture_environment()


def test_convtranspose_sound_f64_gpu_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_CONVTRANSPOSE_SOUND_F64_GPU" not in environment["values"]
    assert "NY_CONVTRANSPOSE_SOUND_F64_GPU" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_convtranspose_sound_f64_gpu_is_captured_raw_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CONVTRANSPOSE_SOUND_F64_GPU", raw)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_CONVTRANSPOSE_SOUND_F64_GPU"] == raw
    assert environment["typed_values"]["NY_CONVTRANSPOSE_SOUND_F64_GPU"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_convtranspose_sound_f64_gpu_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_CONVTRANSPOSE_SOUND_F64_GPU", raw)

    with pytest.raises(
        provenance.ProvenanceError, match="NY_CONVTRANSPOSE_SOUND_F64_GPU"
    ):
        provenance._capture_environment()


def test_root_post_c_survivor_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert "NY_ROOT_POST_C_SURVIVOR" not in environment["values"]
    assert "NY_ROOT_POST_C_SURVIVOR" not in environment["typed_values"]


@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_root_post_c_survivor_explicit_override_is_captured_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_POST_C_SURVIVOR", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id=f"root-post-c-survivor-{raw}").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_ROOT_POST_C_SURVIVOR"] == raw
    assert environment["typed_values"]["NY_ROOT_POST_C_SURVIVOR"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_root_post_c_survivor_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_ROOT_POST_C_SURVIVOR", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_ROOT_POST_C_SURVIVOR"):
        provenance._capture_environment()


def test_root_post_c_survivor_unknown_neighbor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = "NY_ROOT_POST_C_SURVIVOR_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


@pytest.mark.parametrize("key", ["NY_ALPHA_FINAL_BOUND_ONLY", "NY_BN_FOLD_EXT"])
def test_alpha_final_and_bn_fold_experiments_are_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert key not in environment["values"]
    assert key not in environment["typed_values"]


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NY_ALPHA_FINAL_BOUND_ONLY", "0", False),
        ("NY_ALPHA_FINAL_BOUND_ONLY", "1", True),
        ("NY_BN_FOLD_EXT", "0", False),
        ("NY_BN_FOLD_EXT", "1", True),
    ],
)
def test_alpha_final_and_bn_fold_explicit_overrides_are_immutably_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    run_id = f"{key.lower().replace('_', '-')}-{raw}"

    manifest_path = _capture(repo, benchmark_root, run_id=run_id)
    manifest_bytes = manifest_path.read_bytes()
    environment = json.loads(manifest_bytes)["environment"]
    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }

    monkeypatch.setenv(key, "1" if raw == "0" else "0")
    with pytest.raises(FileExistsError, match="immutable evidence"):
        _capture(repo, benchmark_root, run_id=run_id)
    assert manifest_path.read_bytes() == manifest_bytes


@pytest.mark.parametrize("key", ["NY_ALPHA_FINAL_BOUND_ONLY", "NY_BN_FOLD_EXT"])
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_alpha_final_and_bn_fold_malformed_values_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize("key", ["NY_ALPHA_FINAL_BOUND_ONLY", "NY_BN_FOLD_EXT"])
def test_alpha_final_and_bn_fold_unknown_neighbors_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = f"{key}_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


def test_cgan_generator_branch_controls_are_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    for key in CGAN_GENERATOR_BRANCH_ENV:
        assert key not in environment["values"]
        assert key not in environment["typed_values"]


@pytest.mark.parametrize("key", CGAN_GENERATOR_BRANCH_BOOLEAN_ENV)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_cgan_generator_branch_boolean_controls_are_captured_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


def test_cgan_generator_branch_scope_is_sealed_with_raw_node_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    raw_values = {
        "NY_BRANCH_STEM": "1",
        "NY_BRANCH_STEM_K": "0008",
        "NY_BRANCH_STEM_NODES": " Relu_6,Relu_9, Relu_12 ",
        "NY_BRANCH_STEM_PROBE": "0",
        "NY_BRANCH_TRACE": "1",
        "NY_MO_BAB_TRACE": "1",
        "NY_UNSTABLE_COUNT": "0",
    }
    for key, value in raw_values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="cgan-generator-branch").read_text()
    )
    environment = manifest["environment"]

    assert {
        key: environment["values"][key] for key in CGAN_GENERATOR_BRANCH_ENV
    } == raw_values
    assert environment["typed_values"] == {
        "NY_BRANCH_STEM": {"type": "boolean", "value": True},
        "NY_BRANCH_STEM_K": {"type": "positive_integer", "value": 8},
        "NY_BRANCH_STEM_PROBE": {"type": "boolean", "value": False},
        "NY_BRANCH_TRACE": {"type": "boolean", "value": True},
        "NY_MO_BAB_TRACE": {"type": "boolean", "value": True},
        "NY_UNSTABLE_COUNT": {"type": "boolean", "value": False},
    }


@pytest.mark.parametrize("key", CGAN_GENERATOR_BRANCH_BOOLEAN_ENV)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_cgan_generator_branch_boolean_controls_reject_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0",
        "+8",
        "-1",
        " 8",
        "8 ",
        "8.0",
        str((sys.maxsize * 2) + 2),
    ],
)
def test_cgan_generator_branch_depth_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_BRANCH_STEM_K", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_BRANCH_STEM_K"):
        provenance._capture_environment()


@pytest.mark.parametrize("key", CGAN_GENERATOR_BRANCH_ENV)
def test_cgan_generator_branch_unknown_neighbors_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    unknown_key = f"{key}_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    ["NY_AY_BRANCH_HINTS", "NY_AY_MARGIN_REFRAME", "NY_AY_MILP_TALL_FLIP_CAP"],
)
def test_ay_milp_canary_is_captured_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "1")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest_path = _capture(repo, benchmark_root, run_id=key.lower())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["environment"]["values"][key] == "1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", False), ("0", False), ("1", True)],
)
def test_ay_branch_hint_canary_is_sealed_as_a_strict_boolean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_BRANCH_HINTS", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest = json.loads(
        _capture(repo, benchmark_root, run_id="ay-branch-hints-typed").read_text()
    )
    environment = manifest["environment"]
    assert environment["values"]["NY_AY_BRANCH_HINTS"] == raw
    assert environment["typed_values"]["NY_AY_BRANCH_HINTS"] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("raw", ["2", "yes", "true", "-1", " 1"])
def test_ay_branch_hint_canary_malformed_measurement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY_BRANCH_HINTS", raw)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="NY_AY_BRANCH_HINTS"):
        _capture(repo, benchmark_root, run_id="ay-branch-hints-malformed")

    assert not (
        repo / "reports/measured/artifacts/runs/ay-branch-hints-malformed/start.json"
    ).exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("NY_CONV_SKIP_DEAD_F32", "1"),
        ("NY_ALPHA_REFRESH_FRACTION", "0.125"),
        ("NY_PGD_DIAG", "1"),
        ("NY_PGD_EXACT_BATCHED", "0"),
        ("NY_PGD_GAMA", "1"),
        ("NY_PGD_GAMA_LAMBDA", "50"),
        ("NY_PGD_GAMA_LIN_FRAC", "0.25"),
        ("NY_PGD_VJP_BATCH", "0"),
        ("NY_IMB", "1"),
        ("NY_IMB_AY_REGION_PROOF", "affine"),
        ("NY_IMB_BATCHED_REPLAY", "1"),
        ("NY_IMB_WIRE", "1"),
        ("NY_IMB_EARLY", "1"),
        ("NY_IMB_BUDGET_S", "300"),
        ("NY_IMB_LEAF_MODE", "crown_root"),
        ("NY_IMB_OBJ", "0"),
        ("NY_IMB_TAIL_ALPHA", "sample"),
        ("NY_IMB_TAIL_CERT_AY", "1"),
        ("NY_IMB_REGION_K", "2"),
        ("NY_IMB_REPLAY_ONLY", "1"),
        ("NY_IMB_REPLAY_ONLY_LEAF", "4"),
        ("NY_MIP_STABILITY_HINTS", "1"),
        ("NY_BAB_RESNET_REFOLD_GUARD", "0"),
        ("NY_PACKED_GRAPH_ALPHA_QUEUE", "1"),
    ],
)
def test_dark_performance_canaries_are_captured_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    manifest_path = _capture(repo, benchmark_root, run_id=f"dark-{key.lower()}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["environment"]["values"][key] == value


TRANSFER_EXACT_BOOLEAN_GATES = [
    "NY_ADAPTIVE_MICROBATCH_CONTROLLER",
    "NY_PACKED_GRAPH_ALPHA_QUEUE",
    "NY_ROOT_ALPHA_GPU",
]


@pytest.mark.parametrize("key", TRANSFER_EXACT_BOOLEAN_GATES)
def test_transfer_exact_gate_is_default_dark_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()

    assert key not in environment["values"]
    assert key not in environment["typed_values"]


@pytest.mark.parametrize("key", TRANSFER_EXACT_BOOLEAN_GATES)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_transfer_exact_gate_is_captured_raw_and_typed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize("key", TRANSFER_EXACT_BOOLEAN_GATES)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_transfer_exact_gate_malformed_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize(
    "key",
    ["NY_IMB_BATCHED_REPLAY", "NY_IMB_REPLAY_ONLY", "NY_IMB_TAIL_CERT_AY"],
)
@pytest.mark.parametrize(("raw", "expected"), [("0", False), ("1", True)])
def test_imb_exact_gate_is_captured_as_exact_boolean(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: bool,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "boolean",
        "value": expected,
    }


@pytest.mark.parametrize(
    "key",
    ["NY_IMB_BATCHED_REPLAY", "NY_IMB_REPLAY_ONLY", "NY_IMB_TAIL_CERT_AY"],
)
@pytest.mark.parametrize("raw", ["", "00", "true", " 1", "1 ", "+1", "１"])
def test_imb_exact_gate_rejects_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    with pytest.raises(provenance.ProvenanceError, match=key):
        provenance._capture_environment()


@pytest.mark.parametrize("mode", ["affine", "reachability", "residual", "shared"])
def test_imb_ay_region_proof_mode_is_captured_as_closed_enum(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_IMB_AY_REGION_PROOF", mode)

    environment = provenance._capture_environment()

    assert environment["values"]["NY_IMB_AY_REGION_PROOF"] == mode
    assert environment["typed_values"]["NY_IMB_AY_REGION_PROOF"] == {
        "type": "enum",
        "value": mode,
        "allowed_values": ["affine", "reachability", "residual", "shared"],
    }


@pytest.mark.parametrize(
    "raw",
    ["", "Affine", "scalar", "default", " affine", "affine ", "unknown"],
)
def test_imb_ay_region_proof_mode_rejects_unreviewed_spellings(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_IMB_AY_REGION_PROOF", raw)

    with pytest.raises(provenance.ProvenanceError, match="NY_IMB_AY_REGION_PROOF"):
        provenance._capture_environment()


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NY_IMB_OBJ", "0", 0),
        ("NY_IMB_OBJ", "001", 1),
        ("NY_IMB_REPLAY_ONLY_LEAF", "4", 4),
        ("NY_IMB_REPLAY_ONLY_LEAF", "0004", 4),
    ],
)
def test_imb_replay_selectors_are_captured_as_decimal_usize(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    raw: str,
    expected: int,
) -> None:
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, raw)

    environment = provenance._capture_environment()

    assert environment["values"][key] == raw
    assert environment["typed_values"][key] == {
        "type": "nonnegative_integer",
        "value": expected,
    }


def test_boxlift_environment_is_sealed_raw_and_unknown_neighbors_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    values = {
        "NY_BRANCH_RESCORE": "0",
        "NY_CLIP_INTERM_RESNET": "0",
        "NY_F64_LINEAGE_RECOVER": "1",
        "NY_INTERM_REFINE_SELECTIVE_TOPK": "020",
        "NY_PHASE_TELEMETRY": "1",
        "NY_ROOT_JOINT_INTERM_ALPHA": "1",
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS": "012",
        "NY_ROOT_JOINT_INTERM_ALPHA_LR": "0.10",
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM": "02048",
        "NY_ROOT_JOINT_INTERM_ALPHA_SECS": "20.0",
        "NY_ROOT_SPARSE_INTERM_CROWN": "1",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM": "08192",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS": "0512",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS": "04",
        "NY_ROOT_SPARSE_INTERM_CROWN_SECS": "2.0",
        "NY_ROOT_SPEC_PRUNE": "1",
    }
    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="boxlift-default").read_text()
    )
    for key in values:
        assert key not in default_manifest["environment"]["values"]

    for key, value in values.items():
        monkeypatch.setenv(key, value)
    raw_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="boxlift-raw").read_text()
    )
    environment = raw_manifest["environment"]
    for key, value in values.items():
        assert environment["values"][key] == value
        assert key not in environment["typed_values"]

    unknown_key = "NY_ROOT_JOINT_INTERM_ALPHA_UNREVIEWED"
    monkeypatch.setenv(unknown_key, "1")
    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        _capture(repo, benchmark_root, run_id="boxlift-unknown")
    assert not (
        repo / "reports/measured/artifacts/runs/boxlift-unknown/start.json"
    ).exists()


def test_exact_batched_route_switch_is_absent_by_default_and_not_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="exact-batched-default").read_text()
    )
    assert "NY_PGD_EXACT_BATCHED" not in default_manifest["environment"]["values"]

    # The runtime selects generic batching only for the exact string "0". Like
    # adjacent PGD route switches, provenance preserves other spellings as raw
    # evidence instead of normalizing them into a behavior-changing value.
    monkeypatch.setenv("NY_PGD_EXACT_BATCHED", "00")
    raw_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="exact-batched-raw").read_text()
    )
    assert raw_manifest["environment"]["values"]["NY_PGD_EXACT_BATCHED"] == "00"
    assert "NY_PGD_EXACT_BATCHED" not in raw_manifest["environment"]["typed_values"]


def test_spec_alpha_direct_is_captured_raw_and_absent_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ny_environment(monkeypatch)

    environment = provenance._capture_environment()
    assert "NY_SPEC_ALPHA_DIRECT" not in environment["values"]

    # Runtime activation is exact-string (`"1"` only). Preserve both active
    # and inactive spellings as raw launch evidence; do not normalize them.
    for raw in ("1", "0", "true"):
        monkeypatch.setenv("NY_SPEC_ALPHA_DIRECT", raw)
        environment = provenance._capture_environment()
        assert environment["values"]["NY_SPEC_ALPHA_DIRECT"] == raw
        assert "NY_SPEC_ALPHA_DIRECT" not in environment["typed_values"]


def test_rump_f64_engine_zero_is_captured_in_immutable_start_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="rump-f64-default").read_text()
    )
    assert "NY_RUMP_F64_ENGINE" not in default_manifest["environment"]["values"]

    # The runtime disables this engine only for exact "0". Preserve that launch
    # spelling as raw evidence instead of normalizing it into another behavior.
    monkeypatch.setenv("NY_RUMP_F64_ENGINE", "0")
    start_path = _capture(repo, benchmark_root, run_id="rump-f64-disabled")
    start_bytes = start_path.read_bytes()
    environment = json.loads(start_bytes)["environment"]
    assert environment["values"]["NY_RUMP_F64_ENGINE"] == "0"
    assert "NY_RUMP_F64_ENGINE" not in environment["typed_values"]

    monkeypatch.setenv("NY_RUMP_F64_ENGINE", "1")
    with pytest.raises(FileExistsError, match="immutable evidence"):
        _capture(repo, benchmark_root, run_id="rump-f64-disabled")
    assert start_path.read_bytes() == start_bytes


def test_postbab_ab_environment_is_absent_by_default_and_preserved_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="postbab-seeds-default").read_text()
    )
    default_values = default_manifest["environment"]["values"]
    assert "NY_POSTBAB_BAB_SEEDS" not in default_values
    assert "NY_POSTBAB_BAB_SEEDS_K" not in default_values
    assert "NY_POSTBAB_FRONTIER_FASTLANE" not in default_values
    assert "NY_POSTBAB_FRONTIER_FASTLANE_SECS" not in default_values
    assert "NY_POSTBAB_RESERVE_SECS" not in default_values
    assert "NY_ACASXU_PROF" not in default_values

    # The runtime enables export only for exact "1", trims/parses/clamps K,
    # trims/parses the reserve, and enables profiling on any present value.
    # Preserve those launch spellings instead of pre-normalizing their behavior.
    monkeypatch.setenv("NY_POSTBAB_BAB_SEEDS", "1")
    monkeypatch.setenv("NY_POSTBAB_BAB_SEEDS_K", " 0256 ")
    monkeypatch.setenv("NY_POSTBAB_FRONTIER_FASTLANE", "1")
    monkeypatch.setenv("NY_POSTBAB_FRONTIER_FASTLANE_SECS", " 0010 ")
    monkeypatch.setenv("NY_POSTBAB_RESERVE_SECS", " 0100 ")
    monkeypatch.setenv("NY_ACASXU_PROF", "0")
    raw_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="postbab-seeds-raw").read_text()
    )
    raw_environment = raw_manifest["environment"]
    assert raw_environment["values"]["NY_POSTBAB_BAB_SEEDS"] == "1"
    assert raw_environment["values"]["NY_POSTBAB_BAB_SEEDS_K"] == " 0256 "
    assert raw_environment["values"]["NY_POSTBAB_FRONTIER_FASTLANE"] == "1"
    assert raw_environment["values"]["NY_POSTBAB_FRONTIER_FASTLANE_SECS"] == " 0010 "
    assert raw_environment["values"]["NY_POSTBAB_RESERVE_SECS"] == " 0100 "
    assert raw_environment["values"]["NY_ACASXU_PROF"] == "0"
    assert "NY_POSTBAB_BAB_SEEDS" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_BAB_SEEDS_K" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_FRONTIER_FASTLANE" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_FRONTIER_FASTLANE_SECS" not in raw_environment["typed_values"]
    assert "NY_POSTBAB_RESERVE_SECS" not in raw_environment["typed_values"]
    assert "NY_ACASXU_PROF" not in raw_environment["typed_values"]


def test_screen_schedule_environment_is_absent_by_default_and_preserved_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    keys = {
        "NY_SCREEN_WAVE_SIZE": " 0256 ",
        "NY_SCREEN_CELL_CHUNK": "0",
        "NY_SCREEN_MVF_CHUNK": "not-a-number",
        "NY_SCREEN_CROWN_MS": "5.5",
    }
    default_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="screen-schedule-default").read_text()
    )
    for key in keys:
        assert key not in default_manifest["environment"]["values"]

    for key, value in keys.items():
        monkeypatch.setenv(key, value)
    raw_manifest = json.loads(
        _capture(repo, benchmark_root, run_id="screen-schedule-raw").read_text()
    )
    raw_environment = raw_manifest["environment"]
    for key, value in keys.items():
        assert raw_environment["values"][key] == value
        assert key not in raw_environment["typed_values"]


def test_start_binds_ay_executable_and_completion_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    ay_binary = tmp_path / "trusted-ay"
    ay_binary.write_text("#!/bin/sh\necho 'ay fixture'\n", encoding="utf-8")
    ay_binary.chmod(0o755)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("NY_AY", str(ay_binary))
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)

    start = _capture(repo, benchmark_root, run_id="ay-drift")
    start_record = json.loads(start.read_text(encoding="utf-8"))
    executable = start_record["dependencies"]["ay"]["executable"]
    assert executable["declared_path"] == str(ay_binary)
    assert executable["resolved_path"] == str(ay_binary.resolve())
    assert executable["sha256"] == provenance._sha256_file(ay_binary)

    ay_binary.write_text("#!/bin/sh\necho 'changed ay fixture'\n", encoding="utf-8")
    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}
    assert record["integrity"]["status"] == "invalid"
    assert record["completed_successfully"] is False
    assert "ay_executable_identity_mismatch" in codes


@pytest.mark.parametrize(
    ("changed_identity", "violation_code"),
    [
        ("solver", "solver_binary_sha256_mismatch"),
        ("ny_worktree", "ny_worktree_identity_mismatch"),
        ("benchmark", "benchmark_identity_mismatch"),
        ("config_inputs", "config_inputs_identity_mismatch"),
    ],
)
def test_completion_rejects_end_state_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_identity: str,
    violation_code: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "demo.yaml"
    config.write_text("verifier:\n  timeout: 1\n", encoding="utf-8")
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(
        repo,
        benchmark_root,
        run_id=f"drift-{changed_identity}",
        configs_dir=configs,
    )

    if changed_identity == "solver":
        (repo / "target/release/ny").write_text(
            "#!/bin/sh\necho 'changed ny'\n", encoding="utf-8"
        )
    elif changed_identity == "ny_worktree":
        (repo / "scripts/measure_ny_scorecard.sh").write_text(
            "#!/bin/bash\nexit 1\n", encoding="utf-8"
        )
    elif changed_identity == "benchmark":
        (benchmark_root / "README").write_text("changed fixture\n", encoding="utf-8")
    else:
        config.write_text("verifier:\n  timeout: 2\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}
    assert record["integrity"]["status"] == "invalid"
    assert record["completed_successfully"] is False
    assert violation_code in codes


def test_completion_rejects_malformed_input_hash_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})
    monkeypatch.setattr(provenance, "_capture_host_state", lambda: HOST_STATE_FIXTURE)
    start = _capture(repo, benchmark_root, run_id="bad-cache")
    cache = start.with_name("input_hash_cache.json")
    cache.write_text(
        json.dumps(
            {
                "schema": "ny_measurement_input_hash_cache_v1",
                "run_id": "bad-cache",
                "start_manifest_sha256": provenance._sha256(start.read_bytes()),
                "entries": {"not-a-content-address": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}
    assert record["integrity"]["status"] == "invalid"
    assert record["completed_successfully"] is False
    assert "input_hash_cache_entries_invalid" in codes


def test_completion_records_complete_run_artifact_bijection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, _metadata, _result, _log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id="complete-evidence",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    run_evidence = record["integrity"]["checks"]["run_evidence"]
    cache_check = record["integrity"]["checks"]["input_hash_cache"]

    assert record["completed_successfully"] is True
    assert run_evidence["status"] == "valid"
    assert run_evidence["metadata_count"] == 1
    assert run_evidence["result_count"] == 1
    assert run_evidence["solver_log_count"] == 1
    assert run_evidence["csv_row_count"] == 1
    assert run_evidence["validated_record_count"] == 1
    assert cache_check["entry_count"] == 2
    assert cache_check["referenced_entry_count"] == 2
    assert cache_check["rehashed_entry_count"] == 2


@pytest.mark.parametrize(
    ("artifact", "violation_code"),
    [
        ("metadata", "run_result_artifact_mismatch"),
        ("result", "run_result_artifact_mismatch"),
        ("solver_log", "run_solver_log_artifact_mismatch"),
    ],
)
def test_completion_rejects_tampered_run_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    violation_code: str,
) -> None:
    start, metadata_path, result_path, solver_log_path, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"tampered-{artifact}",
    )
    if artifact == "metadata":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["result_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    elif artifact == "result":
        result_path.write_bytes(b"sat\n((X_0 0.0))\n")
    else:
        solver_log_path.write_bytes(b"tampered solver log\n")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert violation_code in codes


def test_completion_rejects_sat_without_structured_witness_after_consistent_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "sat-without-witness"
    start, metadata_path, result_path, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=run_id,
    )
    result_path.write_bytes(b"sat\n")
    result_digest = provenance._sha256_file(result_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "solver_verdict": "sat",
            "witness_present": True,
            "result_sha256": result_digest,
            "raw_result_sha256": result_digest,
            "counterexample_validation": {"status": "not_checked", "checker": None},
        }
    )
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    start_record = json.loads(start.read_text(encoding="utf-8"))
    csv_path = Path(start_record["measurement"]["output_dir"]) / "demo.csv"
    csv_path.write_text(
        f"demo,model.onnx,property.vnnlib,0,sat,9,{run_id}\n",
        encoding="utf-8",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_sat_witness_invalid" in codes


def test_completion_rejects_empty_unsat_after_consistent_digest_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, metadata_path, result_path, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id="empty-unsat",
    )
    result_path.write_bytes(b"")
    result_digest = provenance._sha256_file(result_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["result_sha256"] = result_digest
    metadata["raw_result_sha256"] = result_digest
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_result_verdict_mismatch" in codes


def test_postflight_sat_witness_parser_accepts_vnnlib1_and_vnnlib2() -> None:
    assert provenance._structured_sat_assignment([b"((X_0 0.25))"])
    assert provenance._structured_sat_assignment(
        [b"X float32 [1, 2]", b"0.25", b"-0.5"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solver_exit_status", 1),
        ("solver_verdict", "maybe"),
        ("timeout_seconds", "120"),
    ],
)
def test_completion_rejects_invalid_verdict_status_or_timeout_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    start, metadata_path, _result, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"invalid-{field}",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_metadata_fields_invalid" in codes


@pytest.mark.parametrize(
    ("mutation", "violation_code"),
    [
        ("result", "run_evidence_file_changed_during_completion"),
        ("csv", "run_evidence_file_changed_during_completion"),
        ("namespace", "run_artifact_namespace_changed_during_completion"),
    ],
)
def test_completion_rechecks_artifact_and_csv_snapshot_after_cache_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    violation_code: str,
) -> None:
    run_id = f"snapshot-{mutation}"
    start, metadata_path, result_path, _solver_log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=run_id,
    )
    start_record = json.loads(start.read_text(encoding="utf-8"))
    csv_path = Path(start_record["measurement"]["output_dir"]) / "demo.csv"
    original_validate_cache = provenance._validate_input_hash_cache

    def mutate_after_cache(**kwargs):
        validated = original_validate_cache(**kwargs)
        if mutation == "result":
            result_path.write_bytes(b"unknown\n")
        elif mutation == "csv":
            csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n")
        else:
            orphan = metadata_path.parents[1] / "99999-race" / f"{run_id}.results"
            orphan.parent.mkdir()
            orphan.write_bytes(b"unsat\n")
        return validated

    monkeypatch.setattr(provenance, "_validate_input_hash_cache", mutate_after_cache)

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert violation_code in codes


@pytest.mark.parametrize("artifact", ["metadata", "result", "solver_log"])
def test_completion_rejects_deleted_or_orphaned_run_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    start, metadata_path, result_path, solver_log_path, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"deleted-{artifact}",
    )
    {
        "metadata": metadata_path,
        "result": result_path,
        "solver_log": solver_log_path,
    }[artifact].unlink()

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_artifact_trio_incomplete" in codes


def test_completion_rejects_extra_orphan_run_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "orphan-artifact"
    start, metadata_path, _result, _log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=run_id,
    )
    orphan = metadata_path.parents[1] / "99999-orphan" / f"{run_id}.results"
    orphan.parent.mkdir()
    orphan.write_bytes(b"unsat\n")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_artifact_trio_incomplete" in codes


@pytest.mark.parametrize("mutation", ["delete", "digest", "extra"])
def test_completion_rejects_missing_corrupt_or_unreferenced_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    start, _metadata, _result, _log, cache_path = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id=f"cache-{mutation}",
    )
    if mutation == "delete":
        cache_path.unlink()
    else:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if mutation == "digest":
            first = next(iter(cache["entries"].values()))
            first["sha256"] = "0" * 64
        else:
            extra = tmp_path / "unreferenced-input"
            extra.write_bytes(b"unreferenced\n")
            fingerprint = provenance._file_fingerprint(extra)
            key = provenance._input_cache_key(str(extra.resolve()), fingerprint)
            cache["entries"][key] = {
                "path": str(extra.resolve()),
                "fingerprint": fingerprint,
                "sha256": provenance._sha256_file(extra),
            }
        cache_path.write_text(json.dumps(cache) + "\n", encoding="utf-8")

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    expected = {
        "delete": "input_hash_cache_missing_for_run_artifacts",
        "digest": "input_hash_cache_entry_sha256_mismatch",
        "extra": "input_hash_cache_entry_unreferenced",
    }
    assert expected[mutation] in codes


def test_completion_rejects_csv_row_without_matching_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, _metadata, _result, _log, _cache = _seed_complete_run(
        tmp_path,
        monkeypatch,
        run_id="csv-tamper",
    )
    start_record = json.loads(start.read_text(encoding="utf-8"))
    csv_path = Path(start_record["measurement"]["output_dir"]) / "demo.csv"
    csv_path.write_text(
        "demo,model.onnx,property.vnnlib,0,sat,9,csv-tamper\n",
        encoding="utf-8",
    )

    completion = provenance.create_completion(start_manifest=start, exit_status=0)
    record = json.loads(completion.read_text(encoding="utf-8"))
    codes = {item["code"] for item in record["integrity"]["violations"]}

    assert record["completed_successfully"] is False
    assert "run_csv_artifact_bijection_mismatch" in codes


@pytest.mark.parametrize(
    "unknown_key",
    [
        "NY_UNREVIEWED_SECRET",
        "NY_RUMP_F64_ENGINE_UNREVIEWED",
        "NY_MULTINEURON",
        "NY_MULTINEURON_STEM",
    ],
)
def test_unknown_ny_environment_fails_closed_without_writing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unknown_key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(unknown_key, "do-not-record")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=unknown_key):
        _capture(repo, benchmark_root, run_id="rejected-run")

    assert not (
        repo / "reports/measured/artifacts/runs/rejected-run/start.json"
    ).exists()


def test_inherited_ay_environment_fails_closed_without_writing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv("AY_MILP_FLIP_CAP_SECS", "3")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match="AY_MILP_FLIP_CAP_SECS"):
        _capture(repo, benchmark_root, run_id="rejected-ay-run")

    assert not (
        repo / "reports/measured/artifacts/runs/rejected-ay-run/start.json"
    ).exists()


@pytest.mark.parametrize(
    "key", ["MIMALLOC_PURGE_DELAY", "MIMALLOC_FUTURE_UNREVIEWED_OPTION"]
)
def test_mimalloc_environment_fails_closed_without_writing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.setenv(key, "0")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(provenance.ProvenanceError, match=key):
        _capture(repo, benchmark_root, run_id="rejected-mimalloc-run")

    assert not (
        repo / "reports/measured/artifacts/runs/rejected-mimalloc-run/start.json"
    ).exists()


def test_missing_build_feature_declaration_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, benchmark_root = _measurement_repos(tmp_path)
    _clear_ny_environment(monkeypatch)
    monkeypatch.delenv("NY_BUILD_FEATURES")
    monkeypatch.setattr(provenance, "_gpu_identity", lambda: {"fixture": True})

    with pytest.raises(
        provenance.ProvenanceError, match="NY_BUILD_FEATURES is required"
    ):
        _capture(repo, benchmark_root, run_id="missing-features")

    assert not (
        repo / "reports/measured/artifacts/runs/missing-features/start.json"
    ).exists()


def test_rejects_broad_or_git_metadata_output_roots(tmp_path: Path) -> None:
    repo = tmp_path / "ny"
    repo.mkdir()
    (repo / ".git").mkdir()

    with pytest.raises(provenance.ProvenanceError, match="unsafe broad"):
        provenance._validate_mutation_root(repo, repo, "measurement output directory")
    with pytest.raises(provenance.ProvenanceError, match="Git metadata"):
        provenance._validate_mutation_root(
            repo / ".git" / "measurements",
            repo,
            "measurement output directory",
        )


def test_measurement_output_inside_repo_must_be_ignored_and_untracked(
    tmp_path: Path,
) -> None:
    repo, _benchmark_root = _measurement_repos(tmp_path)
    tracked_dir = repo / "reports/measured"
    tracked_dir.mkdir(parents=True)
    (tracked_dir / "demo.csv").write_text("legacy row\n", encoding="utf-8")
    _run("git", "add", "-f", "reports/measured/demo.csv", cwd=repo)
    _run("git", "commit", "-qm", "track canonical scorecard", cwd=repo)

    with pytest.raises(provenance.ProvenanceError, match="contains tracked NY paths"):
        provenance._validate_mutation_root(
            tracked_dir,
            repo,
            "measurement output directory",
        )
    with pytest.raises(provenance.ProvenanceError, match="must be Git-ignored"):
        provenance._validate_mutation_root(
            repo / "unignored-output",
            repo,
            "measurement output directory",
        )

    provenance._validate_mutation_root(
        repo / "reports/isolated-run",
        repo,
        "measurement output directory",
    )


def test_process_snapshot_is_sorted_bounded_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_args = "worker --token top-secret " + "x" * 500
    stdout = (
        "10 1 S 20 3.0 2.0 python python job.py --api-key=hidden\n"
        f"20 1 R 5 99.0 1.0 worker {long_args}\n"
        "30 1 S 10 3.0 4.0 curl curl https://user:pass@example.test/x\n"
    )
    monkeypatch.setattr(
        provenance,
        "_snapshot_command",
        lambda _command: {
            "status": "ok",
            "returncode": 0,
            "stdout": stdout,
            "stdout_truncated": False,
            "stderr": "",
            "stderr_truncated": False,
        },
    )

    snapshot = provenance._process_snapshot()

    assert [entry["pid"] for entry in snapshot["entries"]] == [20, 30, 10]
    encoded = json.dumps(snapshot)
    assert "top-secret" not in encoded
    assert "hidden" not in encoded
    assert "user:pass" not in encoded
    assert snapshot["entries"][0]["args_redacted"].endswith("…")
    assert len(snapshot["entries"][0]["args_redacted"]) == provenance.PROCESS_ARGS_LIMIT


def test_host_state_schema_is_stable_when_commands_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        provenance.os,
        "getloadavg",
        lambda: (_ for _ in ()).throw(OSError("unsupported")),
    )
    monkeypatch.setattr(
        provenance,
        "_utc_now",
        lambda: "2026-07-18T01:02:03.000000Z",
    )

    state = provenance._capture_host_state()

    assert state["schema"] == "ny_measurement_host_state_v1"
    assert state["captured_at_utc"] == "2026-07-18T01:02:03.000000Z"
    assert state["load_average"] == {
        "available": False,
        "one_minute": None,
        "five_minutes": None,
        "fifteen_minutes": None,
    }
    assert state["processes"]["status"] == "unavailable"
    assert state["processes"]["entries"] == []
    assert state["gpu"]["utilization"]["status"] == "unavailable"
    assert state["gpu"]["compute_processes"]["status"] == "unavailable"
