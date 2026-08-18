# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "promote_regular_bank.py"
BATCH_SCRIPT = REPO / "scripts" / "promote_regular_bank_batch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("promote_regular_bank", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


promote = _load_module()


def _load_batch_module():
    spec = importlib.util.spec_from_file_location(
        "promote_regular_bank_batch", BATCH_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


promote_batch_cli = _load_batch_module()
provenance = promote.evidence.provenance
retro = promote.evidence.retro
import replay_vnncomp2025_counterexample as replay2025  # noqa: E402

CATEGORY = "cgan_2023"
ONNX = "onnx/model.onnx"
VNNLIB = "vnnlib/property.vnnlib"
RUN_ID = "sealed-fixture"
PRODUCTION_OFFICIAL_HASHES = dict(promote.evidence.OFFICIAL_ARTIFACT_SHA256)
PRODUCTION_RESCORE_HASHES = dict(promote.evidence.ORGANIZER_RESCORE_ARTIFACT_SHA256)
PRODUCTION_RESULTS_REPOSITORY_PINS = (
    promote.evidence.OFFICIAL_RESULTS_COMMIT,
    promote.evidence.OFFICIAL_RESULTS_TREE,
    promote.evidence.OFFICIAL_RESULTS_ORIGIN,
)
PRODUCTION_BENCHMARK_PINS = (
    promote.evidence.OFFICIAL_BENCHMARK_COMMIT,
    promote.evidence.OFFICIAL_BENCHMARK_TREE,
    promote.evidence.OFFICIAL_BENCHMARKS_TREE,
    promote.evidence.OFFICIAL_BENCHMARK_ORIGIN,
)


@pytest.fixture(autouse=True)
def _restore_official_hash_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    # Production evidence deliberately pins one exact Linux Git binary. Test
    # repositories exercise the same path-and-byte binding against the host's
    # real Git so this hermetic suite does not depend on that machine-local
    # evidence tool still existing at its sealed path.
    git_executable = shutil.which("git")
    assert git_executable is not None, "Git is required by promotion fixtures"
    git_path = Path(git_executable).resolve(strict=True)
    monkeypatch.setattr(promote.evidence, "PINNED_GIT_EXECUTABLE", git_path)
    monkeypatch.setattr(
        promote.evidence,
        "PINNED_GIT_SHA256",
        hashlib.sha256(git_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(promote.evidence, "_PINNED_GIT_FINGERPRINT", None)
    yield
    promote.evidence.OFFICIAL_ARTIFACT_SHA256 = dict(PRODUCTION_OFFICIAL_HASHES)
    promote.evidence.ORGANIZER_RESCORE_ARTIFACT_SHA256 = dict(PRODUCTION_RESCORE_HASHES)
    (
        promote.evidence.OFFICIAL_RESULTS_COMMIT,
        promote.evidence.OFFICIAL_RESULTS_TREE,
        promote.evidence.OFFICIAL_RESULTS_ORIGIN,
    ) = PRODUCTION_RESULTS_REPOSITORY_PINS
    (
        promote.evidence.OFFICIAL_BENCHMARK_COMMIT,
        promote.evidence.OFFICIAL_BENCHMARK_TREE,
        promote.evidence.OFFICIAL_BENCHMARKS_TREE,
        promote.evidence.OFFICIAL_BENCHMARK_ORIGIN,
    ) = PRODUCTION_BENCHMARK_PINS


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> tuple[str, int]:
    data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha(data), len(data)


def _evidence(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "artifact": path.relative_to(root).as_posix(),
        "sha256": _sha(data),
        "size_bytes": len(data),
    }


def _official_fixture(tmp_path: Path, *, target_truth: str) -> Path:
    root = tmp_path / "official"
    reference = root / "alpha_beta_crown" / "results.csv"
    reference.parent.mkdir(parents=True)
    lines = []
    longtable = []
    scored = []
    for category in retro.REGULAR:
        onnx = ONNX if category == CATEGORY else f"onnx/{category}.onnx"
        vnnlib = VNNLIB if category == CATEGORY else f"vnnlib/{category}.vnnlib"
        truth = target_truth if category == CATEGORY else "unsat"
        lines.append(f"{category},{onnx},{vnnlib},0,{truth},1\n")
        display = category.replace("_", " ")
        longtable.append(f"2025 {display} & 0 & \\textsc{{{truth}}}\n")
        scored.extend(
            [
                f"% Category 2025_{category} fixture\n",
                "0 & tool & x & x & x & x & 10 & x \\\\\n",
            ]
        )
    reference.write_text("".join(lines), encoding="utf-8")
    latex = root / "SCORING-ZERO-TOL" / "latex"
    latex.mkdir(parents=True)
    (latex / "longtable.tex").write_text("".join(longtable), encoding="utf-8")
    (latex / "scored.tex").write_text("".join(scored), encoding="utf-8")
    promote.evidence.OFFICIAL_ARTIFACT_SHA256 = {
        relative: _sha((root / relative).read_bytes())
        for relative in promote.evidence.OFFICIAL_ARTIFACT_SHA256
    }
    return root


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            str(promote.evidence.PINNED_GIT_EXECUTABLE),
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        check=False,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if not key.startswith("GIT_")
            },
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return result.stdout.decode("utf-8").strip()


def _commit_repo(repo: Path, *, origin: str | None = None) -> tuple[str, str]:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Evidence Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    if origin is not None:
        _git(repo, "remote", "add", "origin", origin)
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "canonical evidence fixture",
    )
    return _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD^{tree}")


def _file_identity(path: Path) -> dict[str, object]:
    digest, fingerprint = provenance._stable_file_hash(path)
    return {
        "declared_path": str(path.resolve()),
        "resolved_path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": fingerprint["size_bytes"],
        "fingerprint": fingerprint,
    }


def _sealed_input(
    *,
    source: Path,
    destination: Path,
    root: Path,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    destination.chmod(0o444)
    digest, fingerprint = provenance._stable_file_hash(destination)
    return {
        "artifact": destination.relative_to(root).as_posix(),
        "fingerprint": fingerprint,
        "mode": "read_only",
        "resolved_path": str(destination.resolve()),
        "sha256": digest,
        "size_bytes": fingerprint["size_bytes"],
    }


def _fixture(
    tmp_path: Path,
    *,
    verdict: str = "unsat",
    target_truth: str = "unsat",
    elapsed: int = 3,
    timeout: int = 10,
    dirty: bool = False,
    record_onnx: str = ONNX,
    measured_verdict: str = "timeout",
    measured_rows: int = 1,
    retain_sat_replay: bool = False,
    replay_result: str = "correct",
    ambiguous_onnx_payload: bool = False,
    source_extra_files: dict[str, bytes] | None = None,
    containment_profile: str | None = "gb10-80g",
    current_flight: bool = False,
) -> dict[str, Path | promote.PromotionRequest]:
    benchmark_repo = tmp_path / "benchmark-repo"
    benchmark = benchmark_repo / "benchmarks"
    instances = benchmark / CATEGORY / "instances.csv"
    instances.parent.mkdir(parents=True)
    instances.write_text(f"{ONNX},{VNNLIB},{timeout}\n", encoding="utf-8")
    model_path = benchmark / CATEGORY / ONNX
    property_path = benchmark / CATEGORY / VNNLIB
    model_path.parent.mkdir(parents=True)
    property_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model\n")
    if ambiguous_onnx_payload:
        model_path.with_name(f"{model_path.name}.gz").write_bytes(
            gzip.compress(model_path.read_bytes(), mtime=0)
        )
    property_path.write_bytes(b"property\n")
    fixture_origin = "https://example.invalid/vnncomp2025_benchmarks"
    benchmark_commit, benchmark_tree = _commit_repo(
        benchmark_repo, origin=fixture_origin
    )
    promote.evidence.OFFICIAL_BENCHMARK_COMMIT = benchmark_commit
    promote.evidence.OFFICIAL_BENCHMARK_TREE = benchmark_tree
    promote.evidence.OFFICIAL_BENCHMARKS_TREE = _git(
        benchmark_repo, "rev-parse", "HEAD:benchmarks"
    )
    promote.evidence.OFFICIAL_BENCHMARK_ORIGIN = fixture_origin
    official = _official_fixture(tmp_path, target_truth=target_truth)

    source_repo = tmp_path / "source"
    configs = source_repo / "configs"
    configs.mkdir(parents=True)
    (configs / "fixture.yaml").write_text("fixture: true\n", encoding="utf-8")
    solver_path = source_repo / "solver"
    solver_path.write_bytes(b"canonical solver fixture\n")
    solver_path.chmod(0o755)
    for relative, data in (source_extra_files or {}).items():
        extra = source_repo.joinpath(*relative.split("/"))
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(data)
    exact_commit, _ = _commit_repo(source_repo)
    if current_flight:
        behaviour_epoch = provenance._last_commit_epoch(
            source_repo, provenance._BEHAVIOUR_INPUT_PATHS
        )
        assert behaviour_epoch is not None
        os.utime(solver_path, None)
        assert int(solver_path.stat().st_mtime) >= behaviour_epoch
    full_archive = Path(f"{source_repo}.full-source.tar.gz")
    _git(
        source_repo,
        "archive",
        "--format=tar.gz",
        f"--output={full_archive}",
        exact_commit,
    )

    output = tmp_path / "measurement-output"
    output.mkdir()
    source_csv = output / f"{CATEGORY}.csv"
    source_row = [
        CATEGORY,
        record_onnx,
        VNNLIB,
        "0",
        verdict,
        str(elapsed),
        RUN_ID,
    ]
    with source_csv.open("w", encoding="utf-8", newline="") as destination:
        csv.writer(destination, lineterminator="\n").writerow(source_row)

    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "runs" / RUN_ID
    start_path = run_dir / "start.json"
    run_dir.mkdir(parents=True)
    solver_digest, solver_fingerprint = provenance._stable_file_hash(solver_path)
    sealed_solver = provenance._seal_file(
        solver_path,
        run_dir / "sealed" / "solver" / solver_digest / "solver",
        executable=True,
        expected_sha256=solver_digest,
        expected_fingerprint=solver_fingerprint,
    )
    ay_executable = _file_identity(solver_path)
    sealed_ay = provenance._seal_file(
        solver_path,
        run_dir / "sealed" / "ay" / solver_digest / "ay",
        executable=True,
        expected_sha256=solver_digest,
        expected_fingerprint=solver_fingerprint,
    )
    config_inputs = provenance._capture_config_inputs(configs.resolve())
    sealed_config_inputs = provenance._seal_config_inputs(config_inputs, run_dir)
    ny_identity = provenance._capture_worktree(source_repo)
    if dirty:
        ny_identity = {
            **ny_identity,
            "clean": False,
            "status_porcelain_v1_z_entries": [" M dirty"],
            "tracked_diff_bytes": 1,
        }
    benchmark_identity = provenance._capture_benchmark(benchmark.resolve())
    with provenance._bound_git_executable(str(promote.evidence.PINNED_GIT_EXECUTABLE)):
        git_identity = provenance._capture_git_executable(source_repo)
    measurement = {
        "artifact_root": str(artifact_root.resolve()),
        "benchmark_root": str(benchmark.resolve()),
        "categories": [CATEGORY],
        "categories_raw": CATEGORY,
        "config_inputs": config_inputs,
        "csv_columns": [
            "category",
            "onnx",
            "vnnlib",
            "prepare_seconds",
            "result",
            "runtime_seconds",
            "run_id",
        ],
        "instance_index": 1,
        "max_rows_per_category": 1,
        "output_dir": str(output.resolve()),
        "result_file": str((tmp_path / "scratch" / "result").resolve()),
        "scratch_dir": str((tmp_path / "scratch").resolve()),
        "sealed_config_inputs": sealed_config_inputs,
        "solver_command_template": [
            sealed_solver["path"],
            "vnncomp",
            "v1",
            "<category>",
            "<onnx>",
            "<vnnlib>",
            str((tmp_path / "scratch" / "result").resolve()),
            "<capped_timeout_seconds>",
            "--configs-dir",
            sealed_config_inputs["declared_path"],
        ],
        "solver_environment": {
            "mode": "env-i-reviewed-record-v1",
            "values": {
                "NY_AY": sealed_ay["path"],
                "NY_NO_CUDA": "1",
                "PATH": "/usr/bin:/bin",
                "RUST_LOG": "error",
            },
        },
        "solver_environment_overrides": {
            "NY_AY": sealed_ay["path"],
            "NY_NO_CUDA": "1",
            "PATH": "/usr/bin:/bin",
            "RUST_LOG": "error",
        },
        "solver_environment_unsets": [],
        "solver_log_file": str((tmp_path / "scratch" / "solver.log").resolve()),
        "solver_output_capture": "combined_stdout_stderr_exact_bytes",
        "sweep_invocation": ["fixture"],
        "timeout_cap_seconds": timeout,
        "vnnlib_version_selection": None,
        "watchdog_grace_seconds": 1,
    }
    if current_flight:
        measurement.update(
            {
                "flight_record_file": f'{measurement["result_file"]}.flight.json',
                "flight_record_capture": (
                    promote.evidence.FLIGHT_RECORD_CAPTURE_POLICY
                ),
            }
        )
    containment: dict[str, object] = {"fixture": True}
    if containment_profile is not None:
        containment["containment_profile"] = containment_profile
    solver_binary = {
        "declared_build_features": [],
        "declared_build_features_raw": "",
        "fingerprint": solver_fingerprint,
        "path": str(solver_path.resolve()),
        "sealed_execution": sealed_solver,
        "sha256": solver_digest,
        "size_bytes": solver_fingerprint["size_bytes"],
        "version_returncode": 0,
        "version_stderr": "",
        "version_stdout": "fixture",
    }
    if current_flight:
        solver_binary["build_coherence"] = provenance._capture_build_coherence(
            source_repo, solver_path
        )
    start = {
        "schema": "ny_measurement_start_v1",
        "run_id": RUN_ID,
        "started_at_utc": "2026-07-31T00:00:00Z",
        "ny": ny_identity,
        "benchmark": benchmark_identity,
        "solver_binary": solver_binary,
        "dependencies": {
            "ay": {
                "executable": ay_executable,
                "sealed_executable": sealed_ay,
            },
            "cuda_runtime": {
                "schema": "ny_measurement_cuda_runtime_v1",
                "status": "not_required",
                "reason": "noncuda_measurement_explicitly_allowed",
            },
        },
        "provenance_tools": {"git": git_identity},
        "rust_toolchain": {"fixture": True},
        "measurement": measurement,
        "environment": {},
        "host": {"containment": containment},
        "host_state": {},
    }
    start_digest, start_size = _write_json(start_path, start)

    instance = artifact_root / CATEGORY / "00001-fixture"
    sealed_inputs = {
        "onnx": _sealed_input(
            source=model_path,
            destination=run_dir / "sealed" / "inputs" / "model.onnx",
            root=artifact_root,
        ),
        "vnnlib": _sealed_input(
            source=property_path,
            destination=run_dir / "sealed" / "inputs" / "property.vnnlib",
            root=artifact_root,
        ),
    }
    originals = {
        "onnx": _file_identity(model_path),
        "vnnlib": _file_identity(property_path),
    }
    cache_entries: dict[str, object] = {}
    cache_keys: list[str] = []
    for label in ("onnx", "vnnlib"):
        original = originals[label]
        cache_key = provenance._input_cache_key(
            str(original["resolved_path"]),
            original["fingerprint"],
        )
        cache_keys.append(cache_key)
        cache_entries[cache_key] = {
            "fingerprint": original["fingerprint"],
            "path": original["resolved_path"],
            "sha256": original["sha256"],
        }
    cache_path = run_dir / "input_hash_cache.json"
    _write_json(
        cache_path,
        {
            "schema": "ny_measurement_input_hash_cache_v1",
            "run_id": RUN_ID,
            "start_manifest_sha256": start_digest,
            "updated_at_utc": "2026-07-31T00:00:01Z",
            "entries": cache_entries,
        },
    )
    preflight_path = instance / f"{RUN_ID}.preflight.json"
    preflight_inputs = {
        "onnx": {
            "declared_name": record_onnx,
            "original": originals["onnx"],
            "sealed": sealed_inputs["onnx"],
        },
        "vnnlib": {
            "declared_name": VNNLIB,
            "original": originals["vnnlib"],
            "sealed": sealed_inputs["vnnlib"],
        },
    }
    preflight = {
        "schema": "ny_measurement_input_preflight_v1",
        "run_id": RUN_ID,
        "captured_at_utc": "2026-07-31T00:00:02Z",
        "category": CATEGORY,
        "instance_index": 1,
        "start_manifest": start_path.relative_to(artifact_root).as_posix(),
        "start_manifest_sha256": start_digest,
        "inputs": preflight_inputs,
    }
    _write_json(preflight_path, preflight)

    result_path = instance / f"{RUN_ID}.results"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if verdict == "sat":
        result_path.write_bytes(b"sat\n((X_0 0.5))\n")
    else:
        result_path.write_bytes(f"{verdict}\n".encode())
    log_path = instance / f"{RUN_ID}.solver.log"
    log_path.write_bytes(b"solver log\n")
    metadata_path = instance / f"{RUN_ID}.json"
    result_evidence = _evidence(result_path, artifact_root)
    log_evidence = _evidence(log_path, artifact_root)
    preflight_evidence = _evidence(preflight_path, artifact_root)
    metadata_inputs: dict[str, object] = {}
    for label, declared in (("onnx", record_onnx), ("vnnlib", VNNLIB)):
        original = originals[label]
        metadata_inputs[label] = {
            "declared_path": declared,
            "hash_cache_hit": False,
            "hash_cache_key": cache_keys[0 if label == "onnx" else 1],
            "resolved_path": original["resolved_path"],
            "sha256": original["sha256"],
            "size_bytes": original["size_bytes"],
        }
    metadata = {
        "schema": "ny_measurement_result_v2",
        "schema_version": 2,
        "run_id": RUN_ID,
        "captured_at_utc": "2026-07-31T00:00:03Z",
        "category": CATEGORY,
        "instance_index": 1,
        "onnx": metadata_inputs["onnx"],
        "vnnlib": metadata_inputs["vnnlib"],
        "execution_inputs": sealed_inputs,
        "config_inputs": provenance._expected_metadata_config_identity(start),
        "execution_config_inputs": sealed_config_inputs,
        "input_hash_cache": cache_path.relative_to(artifact_root).as_posix(),
        "input_preflight": {
            "artifact": preflight_evidence["artifact"],
            "schema": "ny_measurement_input_preflight_v1",
            "sha256": preflight_evidence["sha256"],
        },
        "solver_verdict": verdict,
        "solver_exit_status": 0,
        "timeout_seconds": timeout,
        "elapsed_seconds": elapsed,
        "source_csv": str(source_csv.resolve()),
        "start_manifest": start_path.relative_to(artifact_root).as_posix(),
        "start_manifest_sha256": start_digest,
        "result_artifact": result_evidence["artifact"],
        "result_sha256": result_evidence["sha256"],
        "raw_result_sha256": result_evidence["sha256"],
        "solver_log": {
            **log_evidence,
            "stream": "combined_stdout_stderr",
        },
        "counterexample_validation": {
            "checker": None,
            "status": "not_checked" if verdict == "sat" else "not_applicable",
        },
        "witness_present": verdict == "sat",
    }
    if current_flight:
        ambient = {
            name: value
            for name, value in measurement["solver_environment"]["values"].items()
            if name.startswith("NY_") or name == "OMP_NUM_THREADS"
        }
        flight_record = {
            "schema_version": 3,
            "backend_kind": "cpu-only",
            "backend_summary": "fixture CPU backend",
            "host": {
                "hostname": "fixture-host",
                "cpu_model": "fixture-cpu",
                "logical_cores": 1,
                "ram_bytes": 1_073_741_824,
            },
            "load_avg_at_begin": [0.0, 0.0, 0.0],
            "load_avg_at_end": [0.0, 0.0, 0.0],
            "category": CATEGORY,
            "budget_secs": timeout,
            "ambient_env": ambient,
            "levers": {"status": "not_materialized"},
            "events": [
                {"method": "fixture", "status": "ran", "at_secs": 0.0},
                {
                    "method": "run_complete",
                    "status": "complete",
                    "reason": verdict,
                    "at_secs": float(elapsed),
                },
            ],
        }
        flight_bytes = json.dumps(
            flight_record,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        metadata["flight_record"] = {
            "status": "captured",
            "source_sha256": _sha(flight_bytes),
            "size_bytes": len(flight_bytes),
            "record": flight_record,
        }
    _write_json(metadata_path, metadata)
    preflight_summary = {
        label: {
            "original_sha256": originals[label]["sha256"],
            "sealed_artifact": sealed_inputs[label]["artifact"],
            "sealed_sha256": sealed_inputs[label]["sha256"],
        }
        for label in ("onnx", "vnnlib")
    }
    record = {
        "category": CATEGORY,
        "instance_index": 1,
        "onnx": record_onnx,
        "vnnlib": VNNLIB,
        "solver_verdict": verdict,
        "solver_exit_status": 0,
        "timeout_seconds": timeout,
        "elapsed_seconds": elapsed,
        "input_hash_cache_keys": cache_keys,
        "metadata": _evidence(metadata_path, artifact_root),
        "result": result_evidence,
        "solver_log": log_evidence,
        "preflight": {
            **preflight_evidence,
            "inputs": preflight_summary,
        },
    }
    if verdict == "sat" and retain_sat_replay:
        sidecar = {
            "schema": "ny_counterexample_validation_v1",
            "schema_version": 1,
            "status": "validated",
            "classification": "strictly_correct",
            "official_result": replay_result,
        }
        _write_json(
            metadata_path.with_name(
                f"{metadata_path.stem}.counterexample-validation.json"
            ),
            sidecar,
        )

    csv_data = source_csv.read_bytes()
    csv_evidence = [
        {
            "path": str(source_csv.resolve()),
            "sha256": _sha(csv_data),
            "size_bytes": len(csv_data),
            "current_run_row_count": 1,
            "current_run_rows_sha256": provenance._identity_sha256([source_row]),
        }
    ]
    run_evidence = {
        "schema": "ny_measurement_run_evidence_v1",
        "status": "valid",
        "produced_rows": True,
        "metadata_count": 1,
        "result_count": 1,
        "solver_log_count": 1,
        "preflight_count": 1,
        "validated_record_count": 1,
        "csv_row_count": 1,
        "records": [record],
        "records_sha256": provenance._identity_sha256([record]),
        "csv_evidence": csv_evidence,
        "csv_evidence_sha256": provenance._identity_sha256(csv_evidence),
        "input_hash_cache_entry_count": 2,
        "referenced_input_hash_cache_entry_count": 2,
    }
    completion_path = run_dir / "completion.json"
    identity_sources = {
        "benchmark": start["benchmark"],
        "config_inputs": measurement["config_inputs"],
        "containment": start["host"]["containment"],
        "cuda_runtime": start["dependencies"]["cuda_runtime"],
        "git_executable": git_identity,
        "git_executable_post": git_identity,
        "ny_worktree": start["ny"],
        "rust_toolchain": start["rust_toolchain"],
        "sealed_config_inputs": measurement["sealed_config_inputs"],
    }
    checks = {
        name: {
            "expected_identity_sha256": provenance._identity_sha256(identity),
            "observed_identity_sha256": provenance._identity_sha256(identity),
            "status": "valid",
        }
        for name, identity in identity_sources.items()
    }
    checks["solver_binary"] = {
        "expected_fingerprint": solver_fingerprint,
        "expected_sha256": solver_digest,
        "observed_fingerprint": solver_fingerprint,
        "observed_sha256": solver_digest,
        "path": str(solver_path.resolve()),
        "resolved_path": str(solver_path.resolve()),
        "status": "valid",
    }
    checks["sealed_solver_binary"] = {
        "expected_fingerprint": sealed_solver["fingerprint"],
        "expected_sha256": sealed_solver["sha256"],
        "observed_fingerprint": sealed_solver["fingerprint"],
        "observed_sha256": sealed_solver["sha256"],
        "path": sealed_solver["path"],
        "status": "valid",
    }
    checks["ay_executable"] = {
        "expected_identity_sha256": provenance._identity_sha256(ay_executable),
        "observed_identity_sha256": provenance._identity_sha256(ay_executable),
        "resolved_path": ay_executable["resolved_path"],
        "status": "valid",
    }
    checks["sealed_ay_executable"] = {
        "expected_fingerprint": sealed_ay["fingerprint"],
        "expected_sha256": sealed_ay["sha256"],
        "observed_fingerprint": sealed_ay["fingerprint"],
        "observed_sha256": sealed_ay["sha256"],
        "path": sealed_ay["path"],
        "status": "valid",
    }
    checks["run_evidence"] = run_evidence
    rehashed_cache_entries = [
        {
            "key": key,
            "path": entry["path"],
            "sha256": entry["sha256"],
            "size_bytes": entry["fingerprint"]["size_bytes"],
        }
        for key, entry in sorted(cache_entries.items())
    ]
    checks["input_hash_cache"] = {
        "status": "valid",
        "sha256": _sha(cache_path.read_bytes()),
        "entry_count": 2,
        "referenced_entry_count": 2,
        "rehashed_entry_count": 2,
        "entries_sha256": provenance._identity_sha256(rehashed_cache_entries),
    }
    _write_json(
        completion_path,
        {
            "schema": "ny_measurement_completion_v1",
            "run_id": RUN_ID,
            "ended_at_utc": "2026-07-31T00:00:04Z",
            "exit_status": 0,
            "completed_successfully": True,
            "start_manifest": "start.json",
            "start_manifest_sha256": start_digest,
            "input_hash_cache": {
                "artifact": "input_hash_cache.json",
                "entry_count": 2,
                "present": True,
                "sha256": _sha(cache_path.read_bytes()),
                "size_bytes": len(cache_path.read_bytes()),
            },
            "integrity": {
                "schema": "ny_measurement_completion_integrity_v1",
                "status": "valid",
                "violations": [],
                "checks": checks,
            },
            "host_state": {},
        },
    )

    measured = tmp_path / "measured"
    measured.mkdir()
    measured_path = measured / f"{CATEGORY}.csv"
    if measured_rows == 0:
        measured_path.write_bytes(
            b"other,onnx/other.onnx,vnnlib/other.vnnlib,0,timeout,10\r\n"
        )
    else:
        measured_path.write_bytes(
            b"other,onnx/other.onnx,vnnlib/other.vnnlib,0,timeout,10\r\n"
            + (
                f"{CATEGORY},{ONNX},{VNNLIB},0,{measured_verdict},10\r\n".encode()
                * measured_rows
            )
        )
    index = tmp_path / "evidence-index.json"
    request = promote.PromotionRequest(
        artifact_root=artifact_root,
        run_id=RUN_ID,
        category=CATEGORY,
        instance_index=1,
        benchmark_root=benchmark,
        official_results=official,
        measured_dir=measured,
        exact_commit=exact_commit,
        evidence_index=index,
    )
    return {
        "request": request,
        "result": result_path,
        "completion": completion_path,
        "start": start_path,
        "metadata": metadata_path,
        "preflight": preflight_path,
        "cache": cache_path,
        "instances": instances,
        "model": model_path,
        "property": property_path,
        "source_archive": full_archive,
        "measured": measured_path,
        "index": index,
    }


def _install_exact_2025_sidecar(
    fixture: dict[str, Path | promote.PromotionRequest],
    monkeypatch: pytest.MonkeyPatch,
    *,
    official_result: str,
    harness_runner_sha256: str | None = None,
) -> tuple[dict[str, object], Path]:
    # Another test module deliberately loads the runner under the same public
    # module name.  Keep the lazy consumer import bound to this fixture's
    # monkeypatched module so collection order cannot bypass the test doubles.
    monkeypatch.setitem(
        sys.modules,
        "replay_vnncomp2025_counterexample",
        replay2025,
    )
    request = fixture["request"]
    metadata = fixture["metadata"]
    result = fixture["result"]
    start = fixture["start"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(metadata, Path)
    assert isinstance(result, Path)
    assert isinstance(start, Path)

    official = promote.evidence.validate_official_results(request.official_results)
    benchmark = promote.evidence.validate_official_benchmark(request.benchmark_root)
    occurrence, _ = promote.evidence._load_occurrence(
        category=request.category,
        instance_index=request.instance_index,
        benchmark=benchmark,
        official=official,
    )
    authoritative = {
        label: promote.evidence.authoritative_benchmark_input(
            benchmark=benchmark,
            category=request.category,
            declared_name=(occurrence.onnx if label == "onnx" else occurrence.vnnlib),
            label=label,
        )[0]
        for label in ("onnx", "vnnlib")
    }
    assignment = replay2025._extract_assignment(result.read_bytes())
    checker = {
        "repository": replay2025.OFFICIAL_RESULTS_REPOSITORY,
        "commit": replay2025.OFFICIAL_RESULTS_COMMIT,
        "source_sha256": dict(replay2025.OFFICIAL_SOURCE_SHA256),
    }
    harness = {
        "runner_sha256": (
            harness_runner_sha256 or promote.evidence.PINNED_REPLAY_RUNNER_SHA256
        ),
        "worker_sha256": promote.evidence.PINNED_REPLAY_WORKER_SHA256,
        "protocol": replay2025.WORKER_PROTOCOL,
        "import_roots": [
            str(replay2025.PINNED_RUNTIME_ROOT / replay2025.PINNED_SCORING_RELATIVE),
            str(
                replay2025.PINNED_RUNTIME_ROOT
                / replay2025.PINNED_SITE_PACKAGES_RELATIVE
            ),
            str(replay2025.PINNED_RUNTIME_ROOT / replay2025.PINNED_STDLIB_RELATIVE),
        ],
    }
    runtime = {
        "python_executable": "/retained/python",
        "python_sha256": "1" * 64,
        "python_version": replay2025.PINNED_PYTHON_VERSION,
        "venv": str(replay2025.PINNED_RUNTIME_ROOT),
        "execution_scope": "host_bound_local_replay",
        "requirements_sha256": "2" * 64,
        "installed_versions": {},
        "onnxruntime_version": "1.16.3",
        "provider": replay2025.CPU_PROVIDER,
        "stdlib_manifest_sha256": "3" * 64,
        "site_packages_manifest_sha256": "4" * 64,
        "scoring_tree_manifest_sha256": "5" * 64,
        "native_dependencies": replay2025.PINNED_NATIVE_DEPENDENCIES,
        "ort_pybind_upstream_sha256": "6" * 64,
        "ort_pybind_patched_sha256": "7" * 64,
        "execstack_patch": {
            "tool": "patchelf",
            "tool_version": "0.18.0",
            "operation": "--clear-execstack",
            "changed_byte_count": 1,
            "before_gnu_stack": "RWE",
            "after_gnu_stack": "RW",
        },
    }
    response = {
        "result": official_result,
        "message": (
            "L-inf norm difference between onnx execution and CE file output: "
            "0.0 (rel error: 0.0); fixture"
        ),
        "diff": 0.0,
        "rel_error": 0.0,
    }
    input_receipts = {
        "onnx": replay2025._payload_receipt(b"model\n"),
        "vnnlib": replay2025._payload_receipt(b"property\n"),
        "counterexample": replay2025._payload_receipt(assignment),
    }
    request_binding = {
        "protocol": replay2025.WORKER_PROTOCOL,
        "abs_tolerance": replay2025.COUNTEREXAMPLE_ATOL,
        "rel_tolerance": replay2025.COUNTEREXAMPLE_RTOL,
        **input_receipts,
    }
    worker_receipt = {
        "protocol": replay2025.WORKER_PROTOCOL,
        "request_sha256": replay2025._canonical_sha256(request_binding),
        **input_receipts,
        "response_sha256": replay2025._canonical_sha256(response),
        "native_dependencies_sha256": replay2025._canonical_sha256(
            replay2025.PINNED_NATIVE_DEPENDENCIES
        ),
    }
    observed: dict[str, object] = {
        "harness": harness,
        "runtime": runtime,
        "response": response,
        "worker_receipt": worker_receipt,
    }
    monkeypatch.setattr(
        replay2025,
        "capture_replay_snapshot",
        lambda: {"harness": harness, "runtime": runtime},
    )
    monkeypatch.setattr(
        replay2025,
        "revalidate_replay_snapshot",
        lambda _snapshot: None,
    )
    monkeypatch.setattr(
        replay2025,
        "_checker_identity",
        lambda _official, _runtime: checker,
    )
    monkeypatch.setattr(
        replay2025,
        "replay_bound_payloads",
        lambda **_kwargs: observed,
    )

    root = request.artifact_root.resolve()
    sidecar = {
        "schema": replay2025.SCHEMA,
        "schema_version": replay2025.SCHEMA_VERSION,
        "validated_at_utc": "2026-07-31T12:00:00.000000Z",
        "status": "validated",
        "classification": "valid",
        "official_result": official_result,
        "rationale": response["message"],
        "score_credit": True,
        "scoring_year": 2025,
        "settings": {
            "ignore_ce_y": False,
            "counterexample_atol": 1e-4,
            "counterexample_rtol": 1e-3,
            "scoring_zero_tolerance": True,
        },
        "checker": checker,
        **observed,
        "measurement": {
            "run_id": request.run_id,
            "category": request.category,
            "instance_index": request.instance_index,
        },
        "evidence": {
            "metadata": _evidence(metadata, root),
            "raw_result": _evidence(result, root),
            "extracted_assignment": {
                "sha256": _sha(assignment),
                "size_bytes": len(assignment),
                "transformation": "removed_standalone_sat_verdict_line_only",
            },
            "start_manifest": _evidence(start, root),
            "onnx": {
                "sha256": authoritative["onnx"].sha256,
                "size_bytes": authoritative["onnx"].size_bytes,
                "official_git_path": authoritative["onnx"].git_path,
                "official_git_blob": authoritative["onnx"].git_blob,
            },
            "vnnlib": {
                "sha256": authoritative["vnnlib"].sha256,
                "size_bytes": authoritative["vnnlib"].size_bytes,
                "official_git_path": authoritative["vnnlib"].git_path,
                "official_git_blob": authoritative["vnnlib"].git_blob,
            },
        },
    }
    sidecar_path = metadata.with_name(
        f"{metadata.stem}.vnncomp2025-zero-tol-validation.json"
    )
    _write_json(sidecar_path, sidecar)
    return observed, sidecar_path


def _install_dynamic_organizer_corpus(
    fixture: dict[str, Path | promote.PromotionRequest],
    *,
    raw_results: dict[str, str],
    ce_results: dict[str, str],
    baseline_points: dict[str, int],
    true_result: str = "unsat",
) -> None:
    """Install a complete one-row organizer corpus and pin its Git tree."""

    request = fixture["request"]
    assert isinstance(request, promote.PromotionRequest)
    official = request.official_results
    participants = promote.evidence.ORGANIZER_PARTICIPANTS
    assert set(raw_results) == set(participants)
    assert set(baseline_points) == set(participants)
    assert set(ce_results) == {
        tool for tool, result in raw_results.items() if result == "sat"
    }
    assert "correct" not in ce_results.values()

    reference = official / "alpha_beta_crown" / "results.csv"
    reference_rows = list(
        csv.reader(reference.read_text(encoding="utf-8").splitlines())
    )
    for row in reference_rows:
        if row[0] == CATEGORY:
            row[4] = raw_results["alpha_beta_crown"]
    with reference.open("w", encoding="utf-8", newline="") as destination:
        csv.writer(destination, lineterminator="\n").writerows(reference_rows)

    for tool in participants:
        path = official / tool / "results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if tool == "alpha_beta_crown":
            continue
        path.write_text(
            f"{CATEGORY},{ONNX},{VNNLIB},0,{raw_results[tool]},1\n",
            encoding="utf-8",
        )

    latex_names = {
        "alpha_beta_crown": "$\\alpha$-$\\beta$-CROWN",
        "cora": "CORA",
        "neuralsat": "NeuralSAT",
        "nnenum": "nnenum",
        "nnv": "NNV",
        "pyrat": "PyRAT",
        "sobolbox": "SobolBox",
    }
    scored = official / "SCORING-ZERO-TOL" / "latex" / "scored.tex"
    lines = scored.read_text(encoding="utf-8").splitlines(keepends=True)
    marker = f"% Category 2025_{CATEGORY} fixture\n"
    marker_index = lines.index(marker)
    replacement = [marker]
    ranked = sorted(participants, key=lambda tool: baseline_points[tool], reverse=True)
    for rank, tool in enumerate(ranked, start=1):
        result = raw_results[tool]
        verified = int(result == "unsat")
        falsified = int(result == "sat")
        penalty = int(
            result == "sat"
            and ce_results[tool] not in {"correct", "correct_up_to_tolerance"}
        )
        replacement.append(
            f"{rank} & {latex_names[tool]} & {verified} & {falsified} & 0 & "
            f"{penalty} & {baseline_points[tool]} & 0 \\\\\n"
        )
    lines[marker_index : marker_index + 2] = replacement
    scored.write_text("".join(lines), encoding="utf-8")

    scoring = official / "SCORING-ZERO-TOL"
    for name, data in {
        "process_results.py": b"pinned organizer scoring semantics fixture\n",
        "settings.py": b"pinned organizer settings fixture\n",
        "counterexamples.py": b"pinned organizer checker fixture\n",
    }.items():
        (scoring / name).write_bytes(data)

    old_scores: dict[str, int] = {}
    for tool in participants:
        result = raw_results[tool]
        if result in {"timeout", "unknown", "error"}:
            old_scores[tool] = 0
        elif result == "unsat":
            old_scores[tool] = 10
        elif ce_results[tool] in {"correct", "correct_up_to_tolerance"}:
            old_scores[tool] = 10
        else:
            old_scores[tool] = -150
    log_lines = [
        f"Category 2025_{CATEGORY}:\n",
        f"Category 2025_{CATEGORY} has 1 (from alpha_beta_crown)\n",
        f"{len(participants)} participating tools: {list(participants)!r}\n",
    ]
    if ce_results:
        log_lines.append(f"were violated counterexamples valid?: {ce_results!r}\n")
    log_lines.extend(["Row: ['fixture']\n", f"True Result: {true_result}\n"])
    log_lines.extend(
        (
            f"0: {tool} score: {old_scores[tool]}, is_ver: False, "
            "is_fals: False, is_fastest: False\n"
        )
        for tool in participants
    )
    log_lines.extend(
        [
            f"Category 2025_{CATEGORY}:\n",
            *(f"{tool}: {baseline_points[tool]} (0.0%)\n" for tool in participants),
        ]
    )
    (scoring / "results.txt").write_text("".join(log_lines), encoding="utf-8")

    promote.evidence.OFFICIAL_ARTIFACT_SHA256 = {
        relative: _sha((official / relative).read_bytes())
        for relative in promote.evidence.OFFICIAL_ARTIFACT_SHA256
    }
    promote.evidence.ORGANIZER_RESCORE_ARTIFACT_SHA256 = {
        relative: _sha((official / relative).read_bytes())
        for relative in promote.evidence.ORGANIZER_RESCORE_ARTIFACT_SHA256
    }
    origin = "https://example.invalid/vnncomp2025_results"
    commit, tree = _commit_repo(official, origin=origin)
    promote.evidence.OFFICIAL_RESULTS_COMMIT = commit
    promote.evidence.OFFICIAL_RESULTS_TREE = tree
    promote.evidence.OFFICIAL_RESULTS_ORIGIN = origin


def test_tampered_raw_result_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = fixture["result"]
    assert isinstance(result, Path)
    result.write_bytes(b"sat\n")

    with pytest.raises(promote.PromotionError, match="does not match"):
        promote.build_plan(fixture["request"])


def test_nonvalid_completion_check_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    completion = fixture["completion"]
    assert isinstance(completion, Path)
    value = json.loads(completion.read_text(encoding="utf-8"))
    value["integrity"]["checks"]["cuda_runtime"]["status"] = "invalid"
    _write_json(completion, value)

    with pytest.raises(
        promote.PromotionError, match="does not bind its start identity"
    ):
        promote.build_plan(fixture["request"])


def test_dirty_start_provenance_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, dirty=True)

    with pytest.raises(promote.PromotionError, match="dirty NY worktree"):
        promote.build_plan(fixture["request"])


def test_canonical_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, record_onnx="onnx/not-the-official-model.onnx")

    with pytest.raises(promote.PromotionError, match="official order"):
        promote.build_plan(fixture["request"])


def test_raw_path_suffix_collision_does_not_match_official_row(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        record_onnx="shadow/onnx/model.onnx",
    )

    with pytest.raises(promote.PromotionError, match="canonical identity"):
        promote.build_plan(fixture["request"])


@pytest.mark.parametrize(
    "replacement",
    [
        f"./{ONNX},{VNNLIB},10\n",
        f"{ONNX},{VNNLIB},11\n",
    ],
)
def test_instances_worktree_must_equal_exact_pinned_row_bytes(
    tmp_path: Path,
    replacement: str,
) -> None:
    fixture = _fixture(tmp_path)
    instances = fixture["instances"]
    assert isinstance(instances, Path)
    instances.write_text(replacement, encoding="utf-8")

    with pytest.raises(promote.PromotionError, match="pinned commit"):
        promote.build_plan(fixture["request"])


def test_ambiguous_authoritative_payload_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, ambiguous_onnx_payload=True)

    with pytest.raises(promote.PromotionError, match="missing or ambiguous"):
        promote.build_plan(fixture["request"])


def _install_small_retained_payload_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    promote.evidence.PinnedOfficialBenchmark,
    dict[str, bytes],
    Path,
]:
    evidence = promote.evidence
    root = (tmp_path / "retained-large-models").resolve()
    logical = "benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx"
    other_logical = "benchmarks/vggnet16_2022/onnx/vgg16-7.onnx"
    payload = b"small retained official payload\n"
    compressed = gzip.compress(payload, mtime=0)
    setup_data = b"official setup fixture\n"
    setup = {
        "git_blob": "a" * 40,
        "git_path": "setup.sh",
        "sha256": _sha(setup_data),
    }
    payloads = {
        logical: {
            "compressed_sha256": _sha(compressed),
            "compressed_size_bytes": len(compressed),
            "compression": "gzip",
            "payload_sha256": _sha(payload),
            "payload_size_bytes": len(payload),
            "retained_artifact": (
                "cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx.gz"
            ),
            "source_relative_path": (
                "cgan_2023/seed_896832480/onnx/"
                "cGAN_imgSz32_nCh_3_small_transformer.onnx.gz"
            ),
        },
        other_logical: {
            "compressed_sha256": _sha(compressed),
            "compressed_size_bytes": len(compressed),
            "compression": "gzip",
            "payload_sha256": _sha(payload),
            "payload_size_bytes": len(payload),
            "retained_artifact": "vggnet16_2023/onnx/vgg16-7.onnx.gz",
            "source_relative_path": (
                "vggnet16_2023/seed_896832480/onnx/vgg16-7.onnx.gz"
            ),
        },
    }
    manifest = {
        "official_benchmark": {
            "commit": evidence.OFFICIAL_BENCHMARK_COMMIT,
            "origin": evidence.OFFICIAL_BENCHMARK_ORIGIN,
            "setup": setup,
        },
        "payloads": payloads,
        "schema": evidence.LARGE_MODEL_MANIFEST_SCHEMA,
        "source": {
            "base_url": "https://fixture.invalid/webdav",
            "selected_seed": "896832480",
            "share_id": "fixture-share",
        },
    }
    retained_files = {
        payloads[key]["retained_artifact"]: compressed for key in payloads
    }
    for relative, data in retained_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o444)
    manifest_digest, manifest_size = _write_json(root / "manifest.json", manifest)
    (root / "manifest.json").chmod(0o444)
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    root.chmod(0o555)

    monkeypatch.setattr(evidence, "PINNED_LARGE_MODEL_ROOT", root)
    monkeypatch.setattr(
        evidence,
        "PINNED_LARGE_MODEL_MANIFEST_SHA256",
        manifest_digest,
    )
    monkeypatch.setattr(
        evidence,
        "PINNED_LARGE_MODEL_MANIFEST_SIZE",
        manifest_size,
    )
    monkeypatch.setattr(evidence, "EXPECTED_LARGE_MODEL_MANIFEST", manifest)

    repository = tmp_path / "benchmark-repository"
    repository.mkdir()
    benchmark_root = repository / "benchmarks"
    benchmark_root.mkdir()
    benchmark = evidence.PinnedOfficialBenchmark(
        benchmark_root=benchmark_root,
        repository_root=repository,
        identity={
            "commit": evidence.OFFICIAL_BENCHMARK_COMMIT,
            "origin": evidence.OFFICIAL_BENCHMARK_ORIGIN,
        },
    )

    def missing_payload_git_blob(
        _benchmark: promote.evidence.PinnedOfficialBenchmark,
        git_path: str,
    ) -> tuple[str, bytes] | None:
        if git_path == "setup.sh":
            return setup["git_blob"], setup_data
        return None

    monkeypatch.setattr(evidence, "_git_blob", missing_payload_git_blob)
    return benchmark, {logical: payload, other_logical: payload}, root


def _make_retained_fixture_removable(root: Path) -> None:
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o755)
    root.chmod(0o755)


@pytest.mark.parametrize(
    "logical",
    [
        ("benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx"),
        "benchmarks/vggnet16_2022/onnx/vgg16-7.onnx",
    ],
)
def test_missing_git_payload_uses_only_hard_pinned_official_setup_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    logical: str,
) -> None:
    benchmark, payloads, root = _install_small_retained_payload_fixture(
        tmp_path, monkeypatch
    )
    category = logical.split("/", 2)[1]
    try:
        authoritative, observed = promote.evidence.authoritative_benchmark_input(
            benchmark=benchmark,
            category=category,
            declared_name=logical.removeprefix(f"benchmarks/{category}/"),
            label="onnx",
        )
        index_binding = promote.evidence._benchmark_binding(
            SimpleNamespace(
                benchmark_occurrence={"instance_index": 20},
                authoritative_inputs={"onnx": authoritative},
            )
        )
        replay_binding = promote.evidence._exact_replay_input_binding(
            authoritative,
            label="onnx",
        )
        replay2025._validate_input_binding(replay_binding, "onnx")
    finally:
        _make_retained_fixture_removable(root)

    assert observed == payloads[logical]
    assert authoritative.git_path is None
    assert authoritative.git_blob is None
    source = authoritative.retained_setup_payload
    assert source is not None
    assert source["logical_path"] == logical
    assert source["official_setup"]["git_path"] == "setup.sh"
    assert source["official_setup"]["git_blob"] == "a" * 40
    assert source["manifest"]["sha256"] == (
        promote.evidence.PINNED_LARGE_MODEL_MANIFEST_SHA256
    )
    assert index_binding["inputs"]["onnx"]["source_kind"] == (
        "official_setup_retained_payload"
    )
    assert index_binding["inputs"]["onnx"]["retained_setup_payload"] == source
    assert set(replay_binding) == replay2025.RETAINED_INPUT_BINDING_KEYS
    assert replay_binding["official_retained_setup_payload"] == source


def test_git_payload_takes_precedence_over_retained_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = promote.evidence
    repository = tmp_path / "repository"
    benchmark_root = repository / "benchmarks"
    benchmark_root.mkdir(parents=True)
    benchmark = evidence.PinnedOfficialBenchmark(
        benchmark_root,
        repository,
        {
            "commit": evidence.OFFICIAL_BENCHMARK_COMMIT,
            "origin": evidence.OFFICIAL_BENCHMARK_ORIGIN,
        },
    )
    git_payload = b"committed Git payload"
    logical = "benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx"

    monkeypatch.setattr(
        evidence,
        "PINNED_LARGE_MODEL_ROOT",
        tmp_path / "must-not-be-opened",
    )
    monkeypatch.setattr(
        evidence,
        "_git_blob",
        lambda _benchmark, path: ("b" * 40, git_payload) if path == logical else None,
    )

    authoritative, observed = evidence.authoritative_benchmark_input(
        benchmark=benchmark,
        category="cgan_2023",
        declared_name=logical.removeprefix("benchmarks/cgan_2023/"),
        label="onnx",
    )

    assert observed == git_payload
    assert authoritative.git_path == logical
    assert authoritative.retained_setup_payload is None


def test_ambiguous_git_payload_never_falls_back_to_retained_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = promote.evidence
    repository = tmp_path / "repository"
    benchmark_root = repository / "benchmarks"
    benchmark_root.mkdir(parents=True)
    benchmark = evidence.PinnedOfficialBenchmark(
        benchmark_root,
        repository,
        {
            "commit": evidence.OFFICIAL_BENCHMARK_COMMIT,
            "origin": evidence.OFFICIAL_BENCHMARK_ORIGIN,
        },
    )
    logical = "benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx"
    monkeypatch.setattr(
        evidence,
        "PINNED_LARGE_MODEL_ROOT",
        tmp_path / "must-not-be-opened",
    )
    monkeypatch.setattr(
        evidence,
        "_git_blob",
        lambda _benchmark, path: (
            ("b" * 40, b"payload") if path in {logical, f"{logical}.gz"} else None
        ),
    )

    with pytest.raises(promote.PromotionError, match=r"found 2"):
        evidence.authoritative_benchmark_input(
            benchmark=benchmark,
            category="cgan_2023",
            declared_name=logical.removeprefix("benchmarks/cgan_2023/"),
            label="onnx",
        )


def test_nonallowlisted_missing_git_payload_never_opens_retained_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = promote.evidence
    repository = tmp_path / "repository"
    benchmark_root = repository / "benchmarks"
    benchmark_root.mkdir(parents=True)
    benchmark = evidence.PinnedOfficialBenchmark(
        benchmark_root,
        repository,
        {
            "commit": evidence.OFFICIAL_BENCHMARK_COMMIT,
            "origin": evidence.OFFICIAL_BENCHMARK_ORIGIN,
        },
    )
    monkeypatch.setattr(
        evidence,
        "PINNED_LARGE_MODEL_ROOT",
        tmp_path / "must-not-be-opened",
    )
    monkeypatch.setattr(evidence, "_git_blob", lambda _benchmark, _path: None)

    with pytest.raises(promote.PromotionError, match=r"found 0"):
        evidence.authoritative_benchmark_input(
            benchmark=benchmark,
            category="cgan_2023",
            declared_name="onnx/not_allowlisted.onnx",
            label="onnx",
        )


def test_retained_fallback_rejects_mutable_inventory_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark, payloads, root = _install_small_retained_payload_fixture(
        tmp_path, monkeypatch
    )
    logical = next(
        path for path in payloads if path.startswith("benchmarks/cgan_2023/")
    )
    target = root / "cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx.gz"
    target.chmod(0o644)
    try:
        with pytest.raises(
            promote.PromotionError,
            match="inventory or immutable modes differ",
        ):
            promote.evidence.authoritative_benchmark_input(
                benchmark=benchmark,
                category="cgan_2023",
                declared_name=logical.removeprefix("benchmarks/cgan_2023/"),
                label="onnx",
            )
    finally:
        _make_retained_fixture_removable(root)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_retained_fallback_rejects_linked_payload_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_kind: str,
) -> None:
    benchmark, payloads, root = _install_small_retained_payload_fixture(
        tmp_path, monkeypatch
    )
    logical = next(
        path for path in payloads if path.startswith("benchmarks/cgan_2023/")
    )
    target = root / "cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx.gz"
    parent = target.parent
    external = tmp_path / f"{link_kind}-target.gz"
    external.write_bytes(target.read_bytes())
    parent.chmod(0o755)
    target.unlink()
    if link_kind == "symlink":
        target.symlink_to(external)
    else:
        os.link(external, target)
        target.chmod(0o444)
    parent.chmod(0o555)
    try:
        with pytest.raises(
            promote.PromotionError,
            match="symlink|hard-linked",
        ):
            promote.evidence.authoritative_benchmark_input(
                benchmark=benchmark,
                category="cgan_2023",
                declared_name=logical.removeprefix("benchmarks/cgan_2023/"),
                label="onnx",
            )
    finally:
        _make_retained_fixture_removable(root)


def test_retained_payload_cache_requires_final_source_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark, payloads, root = _install_small_retained_payload_fixture(
        tmp_path, monkeypatch
    )
    logical = next(
        path for path in payloads if path.startswith("benchmarks/cgan_2023/")
    )
    declared = logical.removeprefix("benchmarks/cgan_2023/")
    session: dict[str, object] = {
        "authoritative_benchmark": benchmark,
        "authoritative_payload_cache": {},
    }
    cache = session["authoritative_payload_cache"]
    assert isinstance(cache, dict)
    _, captured = promote.evidence.authoritative_benchmark_input(
        benchmark=benchmark,
        category="cgan_2023",
        declared_name=declared,
        label="onnx",
        payload_cache=cache,
    )
    target = root / "cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx.gz"
    parent = target.parent
    parent.chmod(0o755)
    target.chmod(0o644)
    target.write_bytes(gzip.compress(b"mutated payload", mtime=0))
    target.chmod(0o444)
    parent.chmod(0o555)
    try:
        _, cached = promote.evidence.authoritative_benchmark_input(
            benchmark=benchmark,
            category="cgan_2023",
            declared_name=declared,
            label="onnx",
            payload_cache=cache,
        )
        assert cached == captured
        with pytest.raises(
            promote.PromotionError,
            match="payload changed during validation",
        ):
            promote.evidence.revalidate_replay_session(session)
    finally:
        _make_retained_fixture_removable(root)


@pytest.mark.parametrize(
    ("data", "compression"),
    [
        (gzip.compress(b"payload", mtime=0) + b"trailing", "gzip"),
        (
            gzip.compress(b"payload", mtime=0) + gzip.compress(b"second", mtime=0),
            "gzip",
        ),
        (lzma.compress(b"payload") + b"trailing", "xz"),
        (lzma.compress(b"payload") + lzma.compress(b"second"), "xz"),
    ],
)
def test_authoritative_payload_decompression_rejects_extra_streams(
    data: bytes,
    compression: str,
) -> None:
    with pytest.raises(promote.PromotionError, match="multi-|trailing"):
        promote.evidence._strict_decompress(
            data,
            compression=compression,
            label="fixture",
        )


def test_pinned_git_uses_a_minimal_noninjectable_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_PRELOAD", "/tmp/injected.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/injected")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/tmp/attacker.gitconfig")
    monkeypatch.setenv("BASH_ENV", "/tmp/attacker.sh")

    environment = promote.evidence._git_environment()

    assert set(environment) == {
        "PATH",
        "HOME",
        "XDG_CONFIG_HOME",
        "LANG",
        "LC_ALL",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_TERMINAL_PROMPT",
    }
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert not any(key.startswith(("LD_", "DYLD_")) for key in environment)


def test_source_archive_trailing_bytes_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    archive = fixture["source_archive"]
    assert isinstance(archive, Path)
    archive.write_bytes(archive.read_bytes() + b"unbound trailing bytes")

    with pytest.raises(promote.PromotionError, match="multi-member, or trailing"):
        promote.build_plan(fixture["request"])


def test_source_archive_accepts_exact_long_tracked_git_path(
    tmp_path: Path,
) -> None:
    long_path = (
        "reports/measured-ext/evidence/relusplitter/"
        "oval21-benchmark_cifar_wide_kw-img773-eps0.0026143790849673205-"
        "36053ce57d0db2bb719fba653871d0e3.validation.json"
    )
    fixture = _fixture(
        tmp_path,
        source_extra_files={long_path: b'{"classification":"valid"}\n'},
    )

    plan = promote.build_plan(fixture["request"])

    assert plan.summary["action"] == "replace_unresolved_regular_bank_row"


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/member",
        "../escape",
        "nested/../../escape",
        "./noncanonical",
        "nested//noncanonical",
        "nested/./noncanonical",
        "windows\\separator",
        "nul\0member",
    ],
)
def test_source_archive_path_rejects_noncanonical_or_escaping_names(
    path: str,
) -> None:
    assert promote.evidence._safe_source_archive_path(path) is False


def test_source_archive_pax_path_must_be_necessary_exact_and_unadorned() -> None:
    commit = "a" * 40
    long_name = "tracked/" + "x" * 101
    member = tarfile.TarInfo(long_name)
    member.pax_headers = {"comment": commit, "path": long_name}
    assert promote.evidence._canonical_source_member_pax(
        member,
        commit=commit,
    )

    member.pax_headers = {"comment": commit, "path": "../escape"}
    assert not promote.evidence._canonical_source_member_pax(
        member,
        commit=commit,
    )
    member.pax_headers = {
        "comment": commit,
        "path": long_name,
        "unbound": "value",
    }
    assert not promote.evidence._canonical_source_member_pax(
        member,
        commit=commit,
    )

    short_member = tarfile.TarInfo("tracked/short")
    short_member.pax_headers = {
        "comment": commit,
        "path": short_member.name,
    }
    assert not promote.evidence._canonical_source_member_pax(
        short_member,
        commit=commit,
    )


def test_source_archive_extra_directory_member_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    archive = fixture["source_archive"]
    request = fixture["request"]
    assert isinstance(archive, Path)
    assert isinstance(request, promote.PromotionRequest)
    rewritten = archive.with_suffix(".rewritten.tar.gz")
    with (
        tarfile.open(archive, mode="r:gz") as source,
        tarfile.open(
            rewritten,
            mode="w:gz",
            pax_headers={"comment": request.exact_commit},
        ) as destination,
    ):
        for member in source:
            destination.addfile(
                member,
                source.extractfile(member) if member.isfile() else None,
            )
        hidden = tarfile.TarInfo("unbound-hidden-directory")
        hidden.type = tarfile.DIRTYPE
        hidden.mode = 0o775
        destination.addfile(hidden)
    archive.write_bytes(rewritten.read_bytes())

    with pytest.raises(promote.PromotionError, match="directory inventory"):
        promote.build_plan(request)


def test_cache_referenced_keys_must_equal_rehashed_entries(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    completion = fixture["completion"]
    assert isinstance(completion, Path)
    value = json.loads(completion.read_text(encoding="utf-8"))
    run_evidence = value["integrity"]["checks"]["run_evidence"]
    record = run_evidence["records"][0]
    record["input_hash_cache_keys"] = record["input_hash_cache_keys"][:1]
    run_evidence["records_sha256"] = provenance._identity_sha256(
        run_evidence["records"]
    )
    run_evidence["referenced_input_hash_cache_entry_count"] = 1
    value["integrity"]["checks"]["input_hash_cache"]["referenced_entry_count"] = 1
    _write_json(completion, value)

    with pytest.raises(promote.PromotionError, match="exact input-hash-cache"):
        promote.build_plan(fixture["request"])


def test_metadata_cache_keys_cannot_be_swapped_between_inputs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    metadata = fixture["metadata"]
    completion = fixture["completion"]
    request = fixture["request"]
    assert isinstance(metadata, Path)
    assert isinstance(completion, Path)
    assert isinstance(request, promote.PromotionRequest)

    metadata_value = json.loads(metadata.read_text(encoding="utf-8"))
    onnx_key = metadata_value["onnx"]["hash_cache_key"]
    vnnlib_key = metadata_value["vnnlib"]["hash_cache_key"]
    metadata_value["onnx"]["hash_cache_key"] = vnnlib_key
    metadata_value["vnnlib"]["hash_cache_key"] = onnx_key
    metadata_digest, metadata_size = _write_json(metadata, metadata_value)

    completion_value = json.loads(completion.read_text(encoding="utf-8"))
    run_evidence = completion_value["integrity"]["checks"]["run_evidence"]
    record = run_evidence["records"][0]
    record["metadata"]["sha256"] = metadata_digest
    record["metadata"]["size_bytes"] = metadata_size
    record["input_hash_cache_keys"] = [vnnlib_key, onnx_key]
    run_evidence["records_sha256"] = provenance._identity_sha256(
        run_evidence["records"]
    )
    _write_json(completion, completion_value)

    with pytest.raises(promote.PromotionError, match="cache key is invalid"):
        promote.build_plan(request)


def test_completion_check_body_cannot_be_truncated(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    completion = fixture["completion"]
    assert isinstance(completion, Path)
    value = json.loads(completion.read_text(encoding="utf-8"))
    del value["integrity"]["checks"]["solver_binary"]["resolved_path"]
    _write_json(completion, value)

    with pytest.raises(promote.PromotionError, match="canonical required fields"):
        promote.build_plan(fixture["request"])


def test_execution_template_cannot_substitute_an_unsealed_solver(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    start = fixture["start"]
    assert isinstance(start, Path)
    value = json.loads(start.read_text(encoding="utf-8"))
    value["measurement"]["solver_command_template"][0] = "/bin/true"
    _write_json(start, value)

    with pytest.raises(promote.PromotionError, match="exactly bind the sealed solver"):
        promote.build_plan(fixture["request"])


def test_over_budget_decision_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, elapsed=11, timeout=10)

    with pytest.raises(promote.PromotionError, match="over-budget"):
        promote.build_plan(fixture["request"])


@pytest.mark.parametrize("measured_rows", [0, 2])
def test_missing_or_duplicate_bank_row_fails_closed(
    tmp_path: Path, measured_rows: int
) -> None:
    fixture = _fixture(tmp_path, measured_rows=measured_rows)

    with pytest.raises(promote.PromotionError, match="duplicate or missing"):
        promote.build_plan(fixture["request"])


@pytest.mark.parametrize("decided", ["sat", "unsat", "correct", "incorrect"])
def test_decided_or_classified_row_is_never_overwritten(
    tmp_path: Path, decided: str
) -> None:
    fixture = _fixture(tmp_path, measured_verdict=decided)

    with pytest.raises(promote.PromotionError, match="refusing to overwrite"):
        promote.build_plan(fixture["request"])


def test_legacy_decided_row_migration_preserves_prior_row_and_is_idempotent(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, measured_verdict="unsat")
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    request = replace(request, migrate_legacy_decided_row=True)
    prior_row = [CATEGORY, ONNX, VNNLIB, "0", "unsat", "10"]

    plan = promote.build_plan(request)

    assert plan.summary["action"] == ("migrate_legacy_decided_regular_bank_row")
    assert plan.summary["migrate_legacy_decided_row"] is True
    value = json.loads(plan.index_after)
    binding = next(iter(value["entries"].values()))["measured_csv"]
    assert binding["migration"] == promote.LEGACY_DECIDED_ROW_MIGRATION
    assert binding["row_before"] == prior_row
    assert binding["row_before_sha256"] == provenance._identity_sha256(prior_row)

    promote.apply_plan(plan)
    validated = promote.evidence.validate_regular_evidence_index(
        evidence_index=index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        measured_dir=request.measured_dir,
    )
    assert len(validated.creditable_entries) == 1
    assert validated.creditable_entries[0].bank_state == "applied"
    applied_bank = measured.read_bytes()
    applied_index = index.read_bytes()

    repeated = promote.promote(request, apply=True)

    assert repeated["action"] == "already_applied"
    assert repeated["changed"] is False
    assert measured.read_bytes() == applied_bank
    assert index.read_bytes() == applied_index
    with pytest.raises(promote.PromotionError, match="different.*migration mode"):
        promote.build_plan(replace(request, migrate_legacy_decided_row=False))


def test_legacy_decided_row_migration_accepts_replayed_matching_sat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        verdict="sat",
        target_truth="sat",
        measured_verdict="sat",
    )
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )
    request = fixture["request"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(index, Path)
    request = replace(request, migrate_legacy_decided_row=True)

    plan = promote.build_plan(request)

    assert plan.summary["verdict"] == "sat"
    assert plan.summary["action"] == ("migrate_legacy_decided_regular_bank_row")
    promote.apply_plan(plan)
    validated = promote.evidence.validate_regular_evidence_index(
        evidence_index=index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        measured_dir=request.measured_dir,
    )
    assert len(validated.creditable_entries) == 1
    assert validated.creditable_entries[0].evidence.verdict == "sat"


def test_legacy_decided_row_migration_rejects_conflicting_verdict(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, measured_verdict="sat")
    request = fixture["request"]
    assert isinstance(request, promote.PromotionRequest)

    with pytest.raises(promote.PromotionError, match="verdict conflicts"):
        promote.build_plan(replace(request, migrate_legacy_decided_row=True))


@pytest.mark.parametrize("prior_verdict", ["timeout", "correct", "incorrect"])
def test_legacy_decided_row_migration_rejects_ineligible_prior_markers(
    tmp_path: Path, prior_verdict: str
) -> None:
    fixture = _fixture(tmp_path, measured_verdict=prior_verdict)
    request = fixture["request"]
    assert isinstance(request, promote.PromotionRequest)

    with pytest.raises(
        promote.PromotionError,
        match="aggregate correct/incorrect markers and unresolved rows are ineligible",
    ):
        promote.build_plan(replace(request, migrate_legacy_decided_row=True))


def test_legacy_decided_row_migration_requires_boolean_opt_in(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, measured_verdict="unsat")
    request = fixture["request"]
    assert isinstance(request, promote.PromotionRequest)

    with pytest.raises(promote.PromotionError, match="explicit boolean"):
        promote.build_plan(
            replace(request, migrate_legacy_decided_row=1)  # type: ignore[arg-type]
        )


def test_legacy_decided_row_migration_cannot_reclassify_indexed_row(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    assert isinstance(request, promote.PromotionRequest)
    promote.promote(request, apply=True)

    with pytest.raises(promote.PromotionError, match="different.*migration mode"):
        promote.build_plan(replace(request, migrate_legacy_decided_row=True))


def test_sat_without_retained_replay_evidence_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="sat")

    with pytest.raises(promote.PromotionError, match="replay"):
        promote.build_plan(fixture["request"])


@pytest.mark.parametrize(
    ("official_result", "target_truth", "expected_policy"),
    [
        (
            "correct",
            "sat",
            "exact_2025_zero_tol_replay_correct_v1",
        ),
        (
            "correct_up_to_tolerance",
            "unsat",
            "exact_2025_zero_tol_replay_correct_up_to_tolerance_v1",
        ),
    ],
)
def test_exact_2025_sat_replay_is_independently_replayed_before_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    official_result: str,
    target_truth: str,
    expected_policy: str,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth=target_truth)
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result=official_result,
    )

    plan = promote.build_plan(fixture["request"])

    assert plan.summary["verdict"] == "sat"
    assert plan.summary["policy"] == expected_policy


def test_current_start_and_embedded_flight_sat_replay_are_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        verdict="sat",
        target_truth="sat",
        current_flight=True,
    )
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )

    plan = promote.build_plan(fixture["request"])

    assert plan.summary["verdict"] == "sat"
    assert plan.summary["policy"] == "exact_2025_zero_tol_replay_correct_v1"


def test_strict_exact_2025_sat_enters_only_with_complete_dynamic_rescore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="unsat")
    participants = promote.evidence.ORGANIZER_PARTICIPANTS
    raw_results = dict.fromkeys(participants, "unknown")
    raw_results["alpha_beta_crown"] = "unsat"
    baseline = {
        "alpha_beta_crown": 100,
        "cora": 90,
        "neuralsat": 80,
        "nnenum": 70,
        "nnv": 60,
        "pyrat": 50,
        "sobolbox": 40,
    }
    _install_dynamic_organizer_corpus(
        fixture,
        raw_results=raw_results,
        ce_results={},
        baseline_points=baseline,
    )
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )

    plan = promote.build_plan(fixture["request"])

    assert plan.summary["policy"] == (
        "exact_2025_zero_tol_replay_correct_dynamic_rescore_v1"
    )
    assert plan.summary["published_truth"] == "holds"
    assert plan.summary["effective_truth"] == "violated"
    assert plan.summary["organizer_rescore_official_denominator"] == {
        "published_official_points": 100,
        "rescored_official_points": 90,
        "candidate_instance_points": 10,
    }
    index = json.loads(plan.index_after)
    entry = next(iter(index["entries"].values()))
    assert entry["entry_schema"] == promote.DYNAMIC_ENTRY_SCHEMA
    rescore = entry["organizer_rescore"]
    alpha = rescore["participants"]["alpha_beta_crown"]
    assert alpha["published_instance_outcome"] == "correct"
    assert alpha["published_instance_points"] == 10
    assert alpha["rescored_instance_outcome"] == "penalty"
    assert alpha["rescored_instance_points"] == -150
    assert alpha["score_delta"] == -160
    assert alpha["rescored_category_points"] == -60

    promote.apply_plan(plan)
    validated = promote.evidence.validate_regular_evidence_index(
        evidence_index=plan.index_path,
        benchmark_root=plan.request.benchmark_root,
        official_results=plan.request.official_results,
        measured_dir=plan.request.measured_dir,
    )
    assert len(validated.creditable_entries) == 1
    applied = validated.creditable_entries[0]
    assert applied.bank_state == "applied"
    assert applied.entry["entry_schema"] == promote.DYNAMIC_ENTRY_SCHEMA
    assert applied.evidence.organizer_rescore == rescore


def test_dynamic_v7_insertion_preserves_unrelated_v6_value_and_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_fixture = _fixture(tmp_path / "frozen")
    frozen_plan = promote.build_plan(frozen_fixture["request"])
    frozen_index = json.loads(frozen_plan.index_after)
    v6_entry = next(iter(frozen_index["entries"].values()))
    assert v6_entry["entry_schema"] == promote.ENTRY_SCHEMA
    v6_bytes = promote.gap._json_bytes(v6_entry)

    dynamic_fixture = _fixture(
        tmp_path / "dynamic",
        verdict="sat",
        target_truth="unsat",
    )
    participants = promote.evidence.ORGANIZER_PARTICIPANTS
    raw_results = dict.fromkeys(participants, "unknown")
    raw_results["alpha_beta_crown"] = "unsat"
    _install_dynamic_organizer_corpus(
        dynamic_fixture,
        raw_results=raw_results,
        ce_results={},
        baseline_points=dict.fromkeys(participants, 100),
    )
    _install_exact_2025_sidecar(
        dynamic_fixture,
        monkeypatch,
        official_result="correct",
    )
    index_path = dynamic_fixture["index"]
    request = dynamic_fixture["request"]
    assert isinstance(index_path, Path)
    assert isinstance(request, promote.PromotionRequest)
    unrelated_key = json.dumps(
        [CATEGORY, "onnx/unrelated.onnx", "vnnlib/unrelated.vnnlib", 0],
        separators=(",", ":"),
    )
    seeded_value = {
        "entries": {unrelated_key: v6_entry},
        "schema": promote.INDEX_SCHEMA,
    }
    seeded_data = promote.gap._json_bytes(seeded_value)
    index_path.write_bytes(seeded_data)
    validated_index = SimpleNamespace(
        data=seeded_data,
        value=seeded_value,
        entries=(),
        dangling_entries=(),
    )
    monkeypatch.setattr(
        promote.evidence,
        "validate_regular_evidence_index",
        lambda **_kwargs: validated_index,
    )

    plan = promote.build_plan(request)

    planned_value = json.loads(plan.index_after)
    retained = planned_value["entries"][unrelated_key]
    assert retained == v6_entry
    assert promote.gap._json_bytes(retained) == v6_bytes
    dynamic_entries = [
        entry for key, entry in planned_value["entries"].items() if key != unrelated_key
    ]
    assert len(dynamic_entries) == 1
    assert dynamic_entries[0]["entry_schema"] == promote.DYNAMIC_ENTRY_SCHEMA


def test_strict_truth_change_fails_closed_when_rescore_corpus_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="unsat")
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )

    with pytest.raises(
        promote.PromotionError,
        match="Git inspection failed|organizer rescore artifact",
    ):
        promote.build_plan(fixture["request"])


def test_dynamic_rescore_tolerance_only_sat_field_keeps_incumbent_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="unsat")
    raw_results = {
        "alpha_beta_crown": "sat",
        "cora": "sat",
        "neuralsat": "sat",
        "nnenum": "sat",
        "nnv": "timeout",
        "pyrat": "sat",
        "sobolbox": "unknown",
    }
    ce_results = {
        tool: "correct_up_to_tolerance"
        for tool, result in raw_results.items()
        if result == "sat"
    }
    baseline = {
        "alpha_beta_crown": 1860,
        "cora": 1830,
        "neuralsat": 1860,
        "nnenum": 1860,
        "nnv": 1100,
        "pyrat": 1850,
        "sobolbox": 1460,
    }
    _install_dynamic_organizer_corpus(
        fixture,
        raw_results=raw_results,
        ce_results=ce_results,
        baseline_points=baseline,
    )
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )

    plan = promote.build_plan(fixture["request"])

    value = json.loads(plan.index_after)
    rescore = next(iter(value["entries"].values()))["organizer_rescore"]
    assert all(
        payload["score_delta"] == 0 for payload in rescore["participants"].values()
    )
    assert rescore["denominator"] == {
        "published_official_points": 1860,
        "rescored_official_points": 1860,
        "candidate_instance_points": 10,
    }


def test_dynamic_rescore_rejects_raw_organizer_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="unsat")
    participants = promote.evidence.ORGANIZER_PARTICIPANTS
    _install_dynamic_organizer_corpus(
        fixture,
        raw_results=dict.fromkeys(participants, "unknown"),
        ce_results={},
        baseline_points=dict.fromkeys(participants, 10),
    )
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )
    request = fixture["request"]
    assert isinstance(request, promote.PromotionRequest)
    raw = request.official_results / "cora" / "results.csv"
    raw.write_bytes(raw.read_bytes() + b"\n")

    with pytest.raises(
        promote.PromotionError,
        match="worktree is not clean|identity mismatch",
    ):
        promote.build_plan(request)


def test_dynamic_rescore_rejects_indeterminate_organizer_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="unsat")
    participants = promote.evidence.ORGANIZER_PARTICIPANTS
    _install_dynamic_organizer_corpus(
        fixture,
        raw_results=dict.fromkeys(participants, "unknown"),
        ce_results={},
        baseline_points=dict.fromkeys(participants, 10),
        true_result="-",
    )
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )

    with pytest.raises(
        promote.PromotionError,
        match="target truth classification is indeterminate",
    ):
        promote.build_plan(fixture["request"])


def test_dynamic_rescore_index_provenance_tampering_is_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="unsat")
    participants = promote.evidence.ORGANIZER_PARTICIPANTS
    _install_dynamic_organizer_corpus(
        fixture,
        raw_results=dict.fromkeys(participants, "unknown"),
        ce_results={},
        baseline_points=dict.fromkeys(participants, 10),
    )
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )
    request = fixture["request"]
    measured = fixture["measured"]
    index_path = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index_path, Path)
    plan = promote.build_plan(request)
    measured.write_bytes(plan.measured_after)
    value = json.loads(plan.index_after)
    entry = next(iter(value["entries"].values()))
    entry["organizer_rescore"]["denominator"]["rescored_official_points"] += 1
    _write_json(index_path, value)

    with pytest.raises(promote.PromotionError, match="differs from reopened evidence"):
        promote.evidence.validate_regular_evidence_index(
            evidence_index=index_path,
            benchmark_root=request.benchmark_root,
            official_results=request.official_results,
            measured_dir=request.measured_dir,
        )


def test_exact_2025_sat_rejects_forged_replay_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="sat")
    observed, _ = _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )
    forged_observed = json.loads(json.dumps(observed))
    forged_observed["response"]["message"] = "forged independent response"
    monkeypatch.setattr(
        replay2025,
        "replay_bound_payloads",
        lambda **_kwargs: forged_observed,
    )

    with pytest.raises(
        promote.PromotionError,
        match="differs from independent exact 2025 bound replay",
    ):
        promote.build_plan(fixture["request"])


def test_exact_2025_sat_rejects_matched_unpinned_harness_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="sat")
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
        harness_runner_sha256="f" * 64,
    )

    with pytest.raises(
        promote.PromotionError,
        match="pinned exact 2025 replay harness hashes differ",
    ):
        promote.build_plan(fixture["request"])


def test_exact_2025_sat_rejects_independent_replay_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="sat")
    _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )

    def _fail_replay(**_kwargs: object) -> dict[str, object]:
        raise replay2025.ReplayError("fixture worker refused the request")

    monkeypatch.setattr(replay2025, "replay_bound_payloads", _fail_replay)
    with pytest.raises(
        promote.PromotionError,
        match="independent exact 2025 counterexample replay failed",
    ):
        promote.build_plan(fixture["request"])


def test_exact_2025_sat_rejects_forged_worker_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, verdict="sat", target_truth="sat")
    observed, _ = _install_exact_2025_sidecar(
        fixture,
        monkeypatch,
        official_result="correct",
    )
    forged_observed = json.loads(json.dumps(observed))
    forged_observed["worker_receipt"]["response_sha256"] = "f" * 64
    monkeypatch.setattr(
        replay2025,
        "replay_bound_payloads",
        lambda **_kwargs: forged_observed,
    )

    with pytest.raises(
        promote.PromotionError,
        match="differs from independent exact 2025 bound replay",
    ):
        promote.build_plan(fixture["request"])


def test_2026_sat_sidecar_is_not_authoritative_2025_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        verdict="sat",
        target_truth="sat",
        retain_sat_replay=True,
    )

    with pytest.raises(promote.PromotionError, match="exact 2025"):
        promote.build_plan(fixture["request"])


@pytest.mark.parametrize(
    ("replay_result", "policy"),
    [
        ("correct", "retained_sat_replay_correct_v1"),
        (
            "correct_up_to_tolerance",
            "retained_sat_replay_correct_up_to_tolerance_v1",
        ),
    ],
)
def test_2026_sat_sidecar_cannot_supersede_published_holds_truth(
    tmp_path: Path,
    replay_result: str,
    policy: str,
) -> None:
    fixture = _fixture(
        tmp_path,
        verdict="sat",
        target_truth="unsat",
        retain_sat_replay=True,
        replay_result=replay_result,
    )

    del policy
    with pytest.raises(promote.PromotionError, match="exact 2025"):
        promote.build_plan(fixture["request"])


def test_decision_must_align_with_published_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, verdict="unsat", target_truth="sat")

    with pytest.raises(promote.PromotionError, match="requires published holds truth"):
        promote.build_plan(fixture["request"])


def test_missing_published_truth_fails_closed_with_supported_scope(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, verdict="unsat", target_truth="-")

    with pytest.raises(
        promote.PromotionError,
        match=(
            "promotion is supported only when the pinned published truth is "
            "holds or violated"
        ),
    ):
        promote.build_plan(fixture["request"])


def test_malformed_existing_evidence_index_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    index = fixture["index"]
    assert isinstance(index, Path)
    _write_json(
        index,
        {
            "schema": promote.INDEX_SCHEMA,
            "entries": {"not-a-canonical-row-key": {}},
        },
    )

    with pytest.raises(promote.PromotionError, match="invalid row key"):
        promote.build_plan(fixture["request"])


def test_legacy_decided_row_opt_in_does_not_bypass_invalid_index(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, measured_verdict="unsat")
    request = fixture["request"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(index, Path)
    _write_json(
        index,
        {
            "schema": promote.INDEX_SCHEMA,
            "entries": {"not-a-canonical-row-key": {}},
        },
    )

    with pytest.raises(promote.PromotionError, match="invalid row key"):
        promote.build_plan(replace(request, migrate_legacy_decided_row=True))


def test_legacy_decided_row_opt_in_does_not_bypass_unrelated_dangling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, measured_verdict="unsat")
    request = fixture["request"]
    assert isinstance(request, promote.PromotionRequest)
    validated_index = SimpleNamespace(
        data=None,
        value={"schema": promote.INDEX_SCHEMA, "entries": {}},
        entries=(),
        dangling_entries=(SimpleNamespace(row_key="unrelated-row"),),
    )
    monkeypatch.setattr(
        promote.evidence,
        "validate_regular_evidence_index",
        lambda **_kwargs: validated_index,
    )

    with pytest.raises(promote.PromotionError, match="unrelated dangling"):
        promote.build_plan(replace(request, migrate_legacy_decided_row=True))


def test_successful_unsat_is_dry_run_by_default_then_applies_atomically(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    completion = fixture["completion"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    assert isinstance(completion, Path)
    before = measured.read_bytes()
    dry = promote.promote(request)

    assert dry["applied"] is False
    assert dry["old_verdict"] == "timeout"
    assert dry["verdict"] == "unsat"
    assert dry["policy"] == "sealed_unsat_plus_published_holds_v1"
    assert measured.read_bytes() == before
    assert not index.exists()

    applied = promote.promote(request, apply=True)

    assert applied["applied"] is True
    assert applied["changed"] is True
    lines = measured.read_bytes().splitlines(keepends=True)
    assert lines[0] == before.splitlines(keepends=True)[0]
    rows = list(csv.reader(measured.read_text(encoding="utf-8").splitlines()))
    assert rows[1] == [CATEGORY, ONNX, VNNLIB, "0", "unsat", "3", RUN_ID]
    index_value = json.loads(index.read_text(encoding="utf-8"))
    row_key = dry["row_key"]
    entry = index_value["entries"][row_key]
    assert entry["entry_schema"] == promote.ENTRY_SCHEMA
    assert entry["artifact_root"] == str(request.artifact_root.resolve())
    assert entry["run_id"] == RUN_ID
    assert entry["completion"]["sha256"] == _sha(completion.read_bytes())
    assert entry["measured_csv"]["sha256_after"] == _sha(measured.read_bytes())
    assert entry["measured_csv"]["row_before"][4] == "timeout"
    assert entry["measured_csv"]["row_after"] == rows[1]
    assert entry["official_results"]["release_commit"] == (
        promote.evidence.OFFICIAL_RESULTS_COMMIT
    )


@pytest.mark.parametrize("containment_profile", ["gb10-80g", "wsl24-20g"])
def test_new_entry_seals_named_containment_profile(
    tmp_path: Path,
    containment_profile: str,
) -> None:
    fixture = _fixture(tmp_path, containment_profile=containment_profile)

    plan = promote.build_plan(fixture["request"])

    entry = next(iter(json.loads(plan.index_after)["entries"].values()))
    assert entry["entry_schema"] == promote.ENTRY_SCHEMA
    assert entry["containment_profile"] == containment_profile
    assert plan.summary["containment_profile"] == containment_profile


def test_new_entry_rejects_start_without_named_containment_profile(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, containment_profile=None)

    with pytest.raises(
        promote.PromotionError,
        match=r"start\.host\.containment\.containment_profile",
    ):
        promote.build_plan(fixture["request"])


def test_new_entry_rejects_unsupported_named_containment_profile(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, containment_profile="wsl24-20g-alias")

    with pytest.raises(promote.PromotionError, match="profile is unsupported"):
        promote.build_plan(fixture["request"])


@pytest.mark.parametrize("mutation", ["missing", "changed", "extra"])
def test_profiled_entry_containment_binding_tampering_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path, containment_profile="gb10-80g")
    request = fixture["request"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(index, Path)
    promote.promote(request, apply=True)
    value = json.loads(index.read_text(encoding="utf-8"))
    entry = next(iter(value["entries"].values()))
    if mutation == "missing":
        entry.pop("containment_profile")
        expected = "unsupported entry fields"
    elif mutation == "changed":
        entry["containment_profile"] = "wsl24-20g"
        expected = "reopened evidence"
    else:
        entry["containment_profile_alias"] = "gb10-80g"
        expected = "unsupported entry fields"
    _write_json(index, value)

    with pytest.raises(promote.PromotionError, match=expected):
        promote.evidence.validate_regular_evidence_index(
            evidence_index=index,
            benchmark_root=request.benchmark_root,
            official_results=request.official_results,
            measured_dir=request.measured_dir,
        )


def _apply_pre_profile_entry(
    fixture: dict[str, Path | promote.PromotionRequest],
    target: promote.evidence.ValidatedPromotionEvidence,
    *,
    version: int,
) -> promote.evidence.ValidatedEvidenceIndex:
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    measured_before = measured.read_bytes()
    measured_after, row_before, row_after, _ = promote._replace_bank_row(
        measured_path=measured,
        measured_data=measured_before,
        target=target,
    )
    entry = promote.evidence._static_entry_payload(target, version=version)
    entry["measured_csv"] = {
        "path": str(measured.resolve()),
        "row_after": row_after,
        "row_after_sha256": provenance._identity_sha256(row_after),
        "row_before": row_before,
        "row_before_sha256": provenance._identity_sha256(row_before),
        "sha256_after": _sha(measured_after),
        "sha256_before": _sha(measured_before),
    }
    measured.write_bytes(measured_after)
    _write_json(
        index,
        {
            "entries": {
                promote.evidence.canonical_row_key(
                    request.category, target.occurrence
                ): entry
            },
            "schema": promote.INDEX_SCHEMA,
        },
    )
    return promote.evidence.validate_regular_evidence_index(
        evidence_index=index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        measured_dir=request.measured_dir,
    )


def test_pre_profile_v4_entry_remains_fully_reopenable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, containment_profile=None)
    request = fixture["request"]
    assert isinstance(request, promote.PromotionRequest)
    target = promote.evidence.validate_promotion_evidence(
        artifact_root=request.artifact_root,
        run_id=request.run_id,
        category=request.category,
        instance_index=request.instance_index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        exact_commit=request.exact_commit,
    )
    validated = _apply_pre_profile_entry(fixture, target, version=4)

    assert len(validated.creditable_entries) == 1
    assert validated.creditable_entries[0].entry["entry_schema"] == (
        promote.evidence.PRE_PROFILE_ENTRY_SCHEMA
    )
    assert validated.creditable_entries[0].evidence.containment_profile is None


def test_pre_profile_v5_dynamic_entry_remains_fully_reopenable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        verdict="sat",
        target_truth="unsat",
        containment_profile=None,
    )
    participants = promote.evidence.ORGANIZER_PARTICIPANTS
    raw_results = dict.fromkeys(participants, "unknown")
    raw_results["alpha_beta_crown"] = "unsat"
    _install_dynamic_organizer_corpus(
        fixture,
        raw_results=raw_results,
        ce_results={},
        baseline_points=dict.fromkeys(participants, 100),
    )
    _install_exact_2025_sidecar(fixture, monkeypatch, official_result="correct")
    request = fixture["request"]
    assert isinstance(request, promote.PromotionRequest)
    target = promote.evidence.validate_promotion_evidence(
        artifact_root=request.artifact_root,
        run_id=request.run_id,
        category=request.category,
        instance_index=request.instance_index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        exact_commit=request.exact_commit,
    )
    assert target.organizer_rescore is not None

    validated = _apply_pre_profile_entry(fixture, target, version=5)

    assert len(validated.creditable_entries) == 1
    assert validated.creditable_entries[0].entry["entry_schema"] == (
        promote.evidence.PRE_PROFILE_DYNAMIC_ENTRY_SCHEMA
    )
    assert validated.creditable_entries[0].evidence.containment_profile is None


def test_single_request_batch_uses_atomic_index_first_transaction(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)

    plan = promote.build_batch_plan([request])
    assert len(plan.summaries) == 1
    assert plan.summaries[0]["action"] == "replace_unresolved_regular_bank_row"
    promote.apply_batch_plan(plan)

    assert b",unsat,3,sealed-fixture" in measured.read_bytes()
    assert json.loads(index.read_text(encoding="utf-8"))["entries"]


def test_single_request_batch_migrates_matching_legacy_decided_row(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, measured_verdict="unsat")
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    request = replace(request, migrate_legacy_decided_row=True)

    plan = promote.build_batch_plan([request])

    assert plan.summaries[0]["action"] == ("migrate_legacy_decided_regular_bank_row")
    promote.apply_batch_plan(plan)
    entry = next(
        iter(json.loads(index.read_text(encoding="utf-8"))["entries"].values())
    )
    assert entry["measured_csv"]["migration"] == (promote.LEGACY_DECIDED_ROW_MIGRATION)
    assert b",unsat,3,sealed-fixture" in measured.read_bytes()


def test_batch_request_v2_requires_explicit_boolean_migration_field(
    tmp_path: Path,
) -> None:
    raw = {
        "artifact_root": "/fixture/artifacts",
        "run_id": RUN_ID,
        "category": CATEGORY,
        "instance_index": 1,
        "benchmark_root": "/fixture/benchmarks",
        "official_results": "/fixture/results",
        "measured_dir": "/fixture/measured",
        "exact_commit": "a" * 40,
        "evidence_index": None,
        "migrate_legacy_decided_row": True,
    }
    request_path = tmp_path / "request.json"
    _write_json(
        request_path,
        {
            "schema": promote_batch_cli.REQUEST_SCHEMA,
            "requests": [raw],
        },
    )

    loaded = promote_batch_cli._load_requests(request_path)

    assert len(loaded) == 1
    assert loaded[0].migrate_legacy_decided_row is True

    invalid = dict(raw)
    invalid["migrate_legacy_decided_row"] = 1
    _write_json(
        request_path,
        {
            "schema": promote_batch_cli.REQUEST_SCHEMA,
            "requests": [invalid],
        },
    )
    with pytest.raises(promote.PromotionError, match="invalid field types"):
        promote_batch_cli._load_requests(request_path)


def test_batch_request_v1_remains_non_migrating_and_rejects_opt_in_field(
    tmp_path: Path,
) -> None:
    raw = {
        "artifact_root": "/fixture/artifacts",
        "run_id": RUN_ID,
        "category": CATEGORY,
        "instance_index": 1,
        "benchmark_root": "/fixture/benchmarks",
        "official_results": "/fixture/results",
        "measured_dir": "/fixture/measured",
        "exact_commit": "a" * 40,
        "evidence_index": None,
    }
    request_path = tmp_path / "request.json"
    _write_json(
        request_path,
        {
            "schema": promote_batch_cli.PREVIOUS_REQUEST_SCHEMA,
            "requests": [raw],
        },
    )

    loaded = promote_batch_cli._load_requests(request_path)

    assert loaded[0].migrate_legacy_decided_row is False
    raw["migrate_legacy_decided_row"] = True
    _write_json(
        request_path,
        {
            "schema": promote_batch_cli.PREVIOUS_REQUEST_SCHEMA,
            "requests": [raw],
        },
    )
    with pytest.raises(promote.PromotionError, match="canonical fields"):
        promote_batch_cli._load_requests(request_path)


def test_batch_apply_reopens_external_evidence_after_plan_creation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    result = fixture["result"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(result, Path)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    measured_before = measured.read_bytes()
    plan = promote.build_batch_plan([request])
    result.write_bytes(b"sat\n((X_0 0.0))\n")

    with pytest.raises(promote.PromotionError, match="does not match"):
        promote.apply_batch_plan(plan)

    assert measured.read_bytes() == measured_before
    assert not index.exists()


def test_batch_rejects_duplicate_occurrences_before_writes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    before = measured.read_bytes()

    with pytest.raises(promote.PromotionError, match="duplicate requested"):
        promote.build_batch_plan([request, request])

    assert measured.read_bytes() == before
    assert not index.exists()


def test_batch_partial_index_first_state_resumes_all_remaining_files(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index.json"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"first-before\n")
    second.write_bytes(b"second-before\n")
    plan = promote.BatchPromotionPlan(
        requests=(),
        measured_updates=(
            promote.BatchMeasuredUpdate(first, b"first-before\n", b"first-after\n"),
            promote.BatchMeasuredUpdate(second, b"second-before\n", b"second-after\n"),
        ),
        index_path=index,
        index_before=None,
        index_after=b'{"entries":{},"schema":"fixture"}\n',
        summaries=(),
    )
    # Model a process death after the durable index write and first CSV rename.
    index.write_bytes(plan.index_after)
    first.write_bytes(b"first-after\n")

    promote.apply_batch_plan(plan)

    assert index.read_bytes() == plan.index_after
    assert first.read_bytes() == b"first-after\n"
    assert second.read_bytes() == b"second-after\n"


@pytest.mark.parametrize(
    ("fault_target", "fault_call"),
    [
        ("replace", 1),
        ("replace", 2),
        ("replace", 3),
        ("directory_fsync", 1),
        ("directory_fsync", 2),
        ("directory_fsync", 3),
    ],
)
def test_batch_apply_faults_roll_back_every_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_target: str,
    fault_call: int,
) -> None:
    index = tmp_path / "index.json"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"first-before\n")
    second.write_bytes(b"second-before\n")
    plan = promote.BatchPromotionPlan(
        requests=(),
        measured_updates=(
            promote.BatchMeasuredUpdate(first, b"first-before\n", b"first-after\n"),
            promote.BatchMeasuredUpdate(second, b"second-before\n", b"second-after\n"),
        ),
        index_path=index,
        index_before=None,
        index_after=b'{"entries":{},"schema":"fixture"}\n',
        summaries=(),
    )
    if fault_target == "replace":
        original = promote.os.replace
        calls = 0

        def fail_once(source: Path, destination: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == fault_call:
                raise OSError("injected batch replace fault")
            original(source, destination)

        monkeypatch.setattr(promote.os, "replace", fail_once)
    else:
        original_fsync = promote._fsync_directory
        calls = 0

        def fail_fsync_once(directory: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == fault_call:
                raise OSError("injected batch fsync fault")
            original_fsync(directory)

        monkeypatch.setattr(promote, "_fsync_directory", fail_fsync_once)

    with pytest.raises(OSError, match="injected batch"):
        promote.apply_batch_plan(plan)

    assert not index.exists()
    assert first.read_bytes() == b"first-before\n"
    assert second.read_bytes() == b"second-before\n"
    assert not list(tmp_path.glob(".*.promote-*"))


def test_default_index_is_canonical_measured_directory_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    assert isinstance(request, promote.PromotionRequest)

    plan = promote.build_plan(replace(request, evidence_index=None))

    assert plan.index_path == (
        request.measured_dir.resolve() / "regular_evidence_index.json"
    )


def test_single_cli_exposes_explicit_legacy_decided_row_migration_flag() -> None:
    args = promote._parser().parse_args(
        [
            "--artifact-root",
            "/fixture/artifacts",
            "--run-id",
            RUN_ID,
            "--category",
            CATEGORY,
            "--instance-index",
            "1",
            "--benchmark-root",
            "/fixture/benchmarks",
            "--official-results",
            "/fixture/results",
            "--measured-dir",
            "/fixture/measured",
            "--exact-commit",
            "a" * 40,
            "--migrate-legacy-decided-row",
        ]
    )

    assert args.migrate_legacy_decided_row is True


def test_dangling_index_is_resumed_without_rewriting_history(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    original = promote.build_plan(request)

    index.write_bytes(original.index_after)
    retained_index = index.read_bytes()
    validated = promote.evidence.validate_regular_evidence_index(
        evidence_index=index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        measured_dir=request.measured_dir,
    )
    assert len(validated.dangling_entries) == 1
    assert validated.creditable_entries == ()
    resumed = promote.build_plan(request)

    assert resumed.summary["action"] == "resume_dangling_index"
    assert resumed.index_before == retained_index
    assert resumed.index_after == retained_index
    promote.apply_plan(resumed)
    assert index.read_bytes() == retained_index
    assert b",unsat,3,sealed-fixture" in measured.read_bytes()


def test_dangling_legacy_decided_row_migration_resumes_idempotently(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, measured_verdict="unsat")
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    request = replace(request, migrate_legacy_decided_row=True)
    original = promote.build_plan(request)
    index.write_bytes(original.index_after)
    retained_index = index.read_bytes()

    validated = promote.evidence.validate_regular_evidence_index(
        evidence_index=index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        measured_dir=request.measured_dir,
    )
    assert len(validated.dangling_entries) == 1

    resumed = promote.build_plan(request)

    assert resumed.summary["action"] == "resume_dangling_index"
    assert resumed.index_after == retained_index
    promote.apply_plan(resumed)
    assert index.read_bytes() == retained_index
    assert b",unsat,3,sealed-fixture" in measured.read_bytes()
    assert promote.build_plan(request).summary["action"] == "already_applied"


def test_exact_already_applied_request_is_idempotent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    promote.promote(request, apply=True)
    bank_before = measured.read_bytes()
    index_before = index.read_bytes()

    repeated = promote.promote(request, apply=True)

    assert repeated["action"] == "already_applied"
    assert repeated["applied"] is True
    assert repeated["changed"] is False
    assert measured.read_bytes() == bank_before
    assert index.read_bytes() == index_before


def test_same_plan_object_can_be_applied_twice_exactly(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    plan = promote.build_plan(request)

    promote.apply_plan(plan)
    bank_after = measured.read_bytes()
    index_after = index.read_bytes()
    promote.apply_plan(plan)

    assert measured.read_bytes() == bank_after
    assert index.read_bytes() == index_after


def test_apply_reopens_external_evidence_after_plan_creation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    result = fixture["result"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(result, Path)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    plan = promote.build_plan(request)
    measured_before = measured.read_bytes()
    result.write_bytes(b"sat\n((X_0 0.0))\n")

    with pytest.raises(promote.PromotionError, match="does not match"):
        promote.apply_plan(plan)

    assert measured.read_bytes() == measured_before
    assert not index.exists()


def test_apply_hard_fails_on_unrecognized_partial_state(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    plan = promote.build_plan(request)
    index.write_bytes(plan.index_after)
    measured.write_bytes(measured.read_bytes().replace(b",timeout,10", b",timeout,9"))

    with pytest.raises(promote.PromotionError, match="transaction states"):
        promote.apply_plan(plan)


@pytest.mark.parametrize(
    ("fault_target", "fault_call"),
    [
        ("replace", 1),
        ("replace", 2),
        ("directory_fsync", 1),
        ("directory_fsync", 2),
    ],
)
def test_apply_faults_restore_pretransaction_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_target: str,
    fault_call: int,
) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    plan = promote.build_plan(request)
    measured_before = measured.read_bytes()

    if fault_target == "replace":
        original = promote.os.replace
        calls = 0

        def fail_once(source: Path, destination: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == fault_call:
                raise OSError("injected os.replace fault")
            original(source, destination)

        monkeypatch.setattr(promote.os, "replace", fail_once)
    else:
        original_fsync = promote._fsync_directory
        calls = 0

        def fail_fsync_once(directory: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == fault_call:
                raise OSError("injected directory fsync fault")
            original_fsync(directory)

        monkeypatch.setattr(promote, "_fsync_directory", fail_fsync_once)

    with pytest.raises(OSError, match="injected"):
        promote.apply_plan(plan)

    assert measured.read_bytes() == measured_before
    assert not index.exists()
    assert not list(measured.parent.glob(".*.promote-*"))
    assert not list(index.parent.glob(".*.promote-*"))


def test_row_binding_survives_unrelated_later_bank_edit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    promote.promote(request, apply=True)
    value = json.loads(index.read_text(encoding="utf-8"))
    row_key = next(iter(value["entries"]))
    historical_after = value["entries"][row_key]["measured_csv"]["sha256_after"]
    data = measured.read_bytes().replace(
        b"other,onnx/other.onnx,vnnlib/other.vnnlib,0,timeout,10",
        b"other,onnx/other.onnx,vnnlib/other.vnnlib,0,timeout,11",
    )
    measured.write_bytes(data)
    assert _sha(data) != historical_after

    validated = promote.evidence.validate_regular_evidence_index(
        evidence_index=index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        measured_dir=request.measured_dir,
    )

    assert len(validated.entries) == 1
    assert validated.entries[0].bank_state == "applied"
    assert promote.build_plan(request).summary["action"] == "already_applied"


def test_legacy_entry_is_revalidated_without_current_file_hash_assumption(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    measured = fixture["measured"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(measured, Path)
    assert isinstance(index, Path)
    promote.promote(request, apply=True)
    value = json.loads(index.read_text(encoding="utf-8"))
    entry = next(iter(value["entries"].values()))
    entry.pop("entry_schema")
    entry.pop("official_benchmark")
    entry.pop("official_results")
    entry.pop("sat_replay")
    entry.pop("source_snapshot")
    entry.pop("containment_profile")
    entry["benchmark"] = {
        key: entry["benchmark"][key]
        for key in (
            "instance_index",
            "instances_csv",
            "instances_csv_sha256",
            "official_timeout_seconds",
            "onnx",
            "pair_occurrence",
            "vnnlib",
        )
    }
    historical_hashes = {
        key: entry["measured_csv"][key] for key in ("sha256_after", "sha256_before")
    }
    entry["measured_csv"] = {
        key: entry["measured_csv"][key]
        for key in ("path", "sha256_after", "sha256_before")
    }
    _write_json(index, value)
    measured.write_bytes(
        measured.read_bytes().replace(
            b"other,onnx/other.onnx,vnnlib/other.vnnlib,0,timeout,10",
            b"other,onnx/other.onnx,vnnlib/other.vnnlib,0,timeout,11",
        )
    )

    validated = promote.evidence.validate_regular_evidence_index(
        evidence_index=index,
        benchmark_root=request.benchmark_root,
        official_results=request.official_results,
        measured_dir=request.measured_dir,
    )

    assert validated.entries[0].legacy_entry is True
    assert validated.entries[0].bank_state == "applied"
    assert len(validated.creditable_entries) == 1

    migration = promote.build_plan(request)
    assert migration.summary["action"] == "migrate_applied_legacy_index"
    assert migration.measured_before == migration.measured_after
    promote.apply_plan(migration)
    migrated_value = json.loads(index.read_text(encoding="utf-8"))
    migrated = next(iter(migrated_value["entries"].values()))
    assert migrated["entry_schema"] == promote.ENTRY_SCHEMA
    assert "row_before" not in migrated["measured_csv"]
    assert {
        key: migrated["measured_csv"][key] for key in ("sha256_after", "sha256_before")
    } == historical_hashes
    assert promote.build_plan(request).summary["action"] == "already_applied"


def test_prior_entry_is_fully_reopened_and_tampering_fails(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(index, Path)
    promote.promote(request, apply=True)
    value = json.loads(index.read_text(encoding="utf-8"))
    row_key = next(iter(value["entries"]))
    value["entries"][row_key]["runtime_seconds"] = "4"
    _write_json(index, value)

    with pytest.raises(promote.PromotionError, match="reopened evidence"):
        promote.evidence.validate_regular_evidence_index(
            evidence_index=index,
            benchmark_root=request.benchmark_root,
            official_results=request.official_results,
            measured_dir=request.measured_dir,
        )


def test_read_only_validator_returns_nonzero_for_corruption(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path)
    request = fixture["request"]
    index = fixture["index"]
    assert isinstance(request, promote.PromotionRequest)
    assert isinstance(index, Path)
    promote.promote(request, apply=True)
    value = json.loads(index.read_text(encoding="utf-8"))
    row_key = next(iter(value["entries"]))
    value["entries"][row_key]["completion"]["sha256"] = "0" * 64
    _write_json(index, value)

    status = promote.evidence.main(
        [
            "--evidence-index",
            str(index),
            "--benchmark-root",
            str(request.benchmark_root),
            "--official-results",
            str(request.official_results),
            "--measured-dir",
            str(request.measured_dir),
        ]
    )

    assert status == 2
    assert "reopened evidence" in capsys.readouterr().err


def test_official_result_hash_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    expected = dict(promote.evidence.OFFICIAL_ARTIFACT_SHA256)
    first = next(iter(expected))
    expected[first] = "0" * 64
    monkeypatch.setattr(promote.evidence, "OFFICIAL_ARTIFACT_SHA256", expected)

    with pytest.raises(promote.PromotionError, match="identity mismatch"):
        promote.build_plan(fixture["request"])
