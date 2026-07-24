# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "abcrown_transfer_baseline.py"
MANIFEST = REPO_ROOT / "benchmarks" / "abcrown_transfer_corpus_v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("abcrown_transfer_baseline", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = _load_module()


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _identity(path: Path, declared_path: str) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "declared_path": declared_path,
        "resolved_path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": _digest(data),
    }


def test_checked_in_manifest_is_complete_and_missing_benchmarks_are_skip_safe(
    tmp_path: Path,
) -> None:
    manifest, manifest_bytes = baseline.load_corpus_manifest(MANIFEST)

    assert manifest["schema"] == baseline.CORPUS_SCHEMA
    assert len(manifest["entries"]) >= 12
    assert len(manifest_bytes) > 0
    observed_tags = {
        tag for entry in manifest["entries"] for tag in entry["tags"]
    }
    assert baseline.REQUIRED_COVERAGE_TAGS <= observed_tags

    resolution = baseline.resolve_corpus(
        manifest,
        repo_root=REPO_ROOT,
        benchmark_root=tmp_path / "benchmark-assets-not-installed",
    )

    assert resolution["validation_errors"] == []
    repository_entries = [
        entry
        for entry in resolution["entries"]
        if entry["kind"].startswith("repository_")
    ]
    vnncomp_entries = [
        entry for entry in resolution["entries"] if entry["kind"] == "vnncomp"
    ]
    assert repository_entries
    assert all(entry["status"] == "ready" for entry in repository_entries)
    assert all(entry["status"] == "skipped" for entry in vnncomp_entries)
    assert all(
        entry["skip_reasons"] == ["benchmark_category_unavailable"]
        for entry in vnncomp_entries
    )


def _fixture_vnncomp_entry(
    repo_root: Path,
    benchmark_root: Path,
) -> dict[str, object]:
    model = b"model bytes\n"
    prop = b"; property bytes\n"
    category = benchmark_root / "demo"
    (category / "onnx").mkdir(parents=True)
    (category / "vnnlib").mkdir()
    (category / "onnx/model.onnx").write_bytes(model)
    (category / "vnnlib/prop.vnnlib").write_bytes(prop)
    (category / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/prop.vnnlib,12\n",
        encoding="utf-8",
    )
    preset = repo_root / "configs/demo.yaml"
    preset.parent.mkdir(parents=True)
    preset.write_text("general: {}\n", encoding="utf-8")
    return {
        "id": "demo-row",
        "kind": "vnncomp",
        "role": "fixture",
        "tags": ["fixture"],
        "category": "demo",
        "source_index": 1,
        "model": "onnx/model.onnx",
        "property": "vnnlib/prop.vnnlib",
        "timeout_seconds": 12,
        "preset": "configs/demo.yaml",
        "expected": {
            "model_sha256": _digest(model),
            "property_sha256": _digest(prop),
        },
    }


def test_vnncomp_resolution_binds_row_and_content_identity(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    benchmark_root = tmp_path / "benchmarks"
    repo_root.mkdir()
    entry = _fixture_vnncomp_entry(repo_root, benchmark_root)

    resolved, errors = baseline._resolve_vnncomp_entry(
        entry,
        repo_root=repo_root,
        benchmark_root=benchmark_root,
    )

    assert errors == []
    assert resolved["status"] == "ready"
    assert resolved["skip_reasons"] == []
    assert resolved["files"]["model"]["sha256"] == entry["expected"]["model_sha256"]
    assert (
        resolved["files"]["property"]["sha256"]
        == entry["expected"]["property_sha256"]
    )
    assert resolved["files"]["instances_csv"]["sha256"]
    assert resolved["files"]["preset"]["sha256"]


def test_vnncomp_resolution_rejects_source_row_drift(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    benchmark_root = tmp_path / "benchmarks"
    repo_root.mkdir()
    entry = _fixture_vnncomp_entry(repo_root, benchmark_root)
    instances = benchmark_root / "demo/instances.csv"
    instances.write_text(
        "onnx/model.onnx,vnnlib/other.vnnlib,12\n",
        encoding="utf-8",
    )

    _resolved, errors = baseline._resolve_vnncomp_entry(
        entry,
        repo_root=repo_root,
        benchmark_root=benchmark_root,
    )

    assert len(errors) == 1
    assert "instances.csv row 1 identity mismatch" in errors[0]


def test_parse_telemetry_preserves_phase_and_frontier_frames() -> None:
    parsed = baseline.parse_telemetry(
        "\n".join(
            [
                "unrelated solver output",
                "[phase] root start t=0.0s",
                "[phase] root end t=1.5s",
                "[frontier] d=3 worst=-0.12500 domains=64 t=2.0s",
                "[frontier] d=4 worst=-1.2e-2 domains=96 t=2.5s",
            ]
        )
    )

    assert parsed["phase"]["events"] == [
        {"name": "root start", "seconds": 0.0},
        {"name": "root end", "seconds": 1.5},
    ]
    assert parsed["phase"]["intervals"] == [
        {"from": "root start", "to": "root end", "seconds": 1.5}
    ]
    assert parsed["frontier"]["frames"][-1] == {
        "depth": 4,
        "worst_margin": -0.012,
        "domains_cumulative": 96,
        "seconds": 2.5,
    }


def test_supplemental_metrics_reject_negative_or_unknown_counters(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema": baseline.SUPPLEMENTAL_SCHEMA,
                "passes": {"gradient_passes": -1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(baseline.BaselineError, match="gradient_passes"):
        baseline.load_supplemental_metrics(bad)

    bad.write_text(
        json.dumps(
            {
                "schema": baseline.SUPPLEMENTAL_SCHEMA,
                "memory": {"unreviewed_bytes": 1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(baseline.BaselineError, match="unknown keys"):
        baseline.load_supplemental_metrics(bad)


def _baseline_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    model = tmp_path / "model.onnx"
    prop = tmp_path / "property.vnnlib"
    preset = tmp_path / "preset.yaml"
    instances = tmp_path / "instances.csv"
    model.write_bytes(b"model\n")
    prop.write_bytes(b"property\n")
    preset.write_text("general: {}\n", encoding="utf-8")
    instances.write_text("model.onnx,property.vnnlib,10\n", encoding="utf-8")
    start = tmp_path / "start.json"
    start.write_text('{"schema":"ny_measurement_start_v1"}\n', encoding="utf-8")
    payload = {
        "schema": baseline.BASELINE_SCHEMA,
        "run_id": "fixture-run",
        "captured_at_utc": "2026-07-23T00:00:00.000000Z",
        "measurement_start": {
            "path": str(start.resolve()),
            "sha256": baseline._sha256_file(start),
        },
        "corpus_manifest": {
            "path": str(MANIFEST),
            "sha256": baseline._sha256_file(MANIFEST),
        },
        "metric_contract": baseline.metric_contract(),
        "resolution": {
            "entries": [
                {
                    "id": "fixture-row",
                    "kind": "vnncomp",
                    "status": "ready",
                    "files": {
                        "model": _identity(model, "model.onnx"),
                        "property": _identity(prop, "property.vnnlib"),
                        "preset": _identity(preset, "preset.yaml"),
                        "instances_csv": _identity(instances, "instances.csv"),
                    },
                }
            ]
        },
    }
    artifact = tmp_path / "transfer-baseline.json"
    baseline._write_immutable(artifact, payload)
    return artifact, model, prop, start


def test_record_row_is_immutable_and_keeps_unavailable_metrics_explicit(
    tmp_path: Path,
) -> None:
    baseline_path, _model, _prop, _start = _baseline_fixture(tmp_path)
    log = tmp_path / "solver.log"
    log.write_text(
        "[phase] root start t=0.0s\n"
        "[phase] graph-bab start t=1.0s\n"
        "[frontier] d=2 worst=-0.25 domains=40 t=1.5s\n",
        encoding="utf-8",
    )
    result = tmp_path / "result.txt"
    result.write_text("unsat\n", encoding="utf-8")
    supplemental = tmp_path / "metrics.json"
    supplemental.write_text(
        json.dumps(
            {
                "schema": baseline.SUPPLEMENTAL_SCHEMA,
                "phase_totals_seconds": {"root": 1.0, "bab": 4.0},
                "domains_explored": 50,
                "active_objective_rows": 9,
                "passes": {
                    "bound_passes": 7,
                    "gradient_passes": 3,
                },
                "batching": {
                    "backoff_count": 1,
                    "batch_size_histogram": {"16": 2, "32": 1},
                },
                "memory": {
                    "peak_host_bytes": 4096,
                    "gpu_utilization_percent": 75.5,
                },
                "deadline": {
                    "overrun_seconds": 0,
                    "watchdog_hit": False,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "row.json"

    baseline.record_row(
        baseline_path=baseline_path,
        entry_id="fixture-row",
        log_path=log,
        result_path=result,
        wall_seconds=5.0,
        supplemental_path=supplemental,
        output_path=output,
    )
    row = json.loads(output.read_text(encoding="utf-8"))

    assert row["schema"] == baseline.ROW_SCHEMA
    assert row["metrics"]["outcome"]["verdict"] == "verified"
    assert row["metrics"]["outcome"]["solved_count"] == 1
    assert row["metrics"]["outcome"]["domains_explored"] == 50
    assert row["metrics"]["outcome"]["domains_per_second"] == 10.0
    assert row["metrics"]["frontier"]["maximum_depth"] == 2
    assert row["metrics"]["passes"]["gradient_passes"] == 3
    assert row["metrics"]["passes"]["gpu_calls"] is None
    assert "passes.gpu_calls" in row["metrics"]["unavailable"]
    assert row["metrics"]["memory"]["peak_host_bytes"] == 4096
    assert row["metrics"]["batching"]["batch_size_histogram"] == {
        "16": 2,
        "32": 1,
    }

    with pytest.raises(FileExistsError, match="immutable evidence"):
        baseline.record_row(
            baseline_path=baseline_path,
            entry_id="fixture-row",
            log_path=log,
            result_path=result,
            wall_seconds=5.0,
            supplemental_path=supplemental,
            output_path=output,
        )


def test_record_row_rejects_asset_drift(tmp_path: Path) -> None:
    baseline_path, model, _prop, _start = _baseline_fixture(tmp_path)
    model.write_bytes(b"changed model\n")
    log = tmp_path / "solver.log"
    log.write_text("", encoding="utf-8")
    result = tmp_path / "result.txt"
    result.write_text("unknown\n", encoding="utf-8")

    with pytest.raises(baseline.BaselineError, match="changed after baseline"):
        baseline.record_row(
            baseline_path=baseline_path,
            entry_id="fixture-row",
            log_path=log,
            result_path=result,
            wall_seconds=1,
            supplemental_path=None,
            output_path=tmp_path / "row.json",
        )


def test_capture_reuses_measurement_provenance_and_binds_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_capture_start_manifest(**kwargs):
        calls.append(kwargs)
        path = (
            Path(kwargs["artifact_root"])
            / "runs"
            / str(kwargs["run_id"])
            / "start.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text('{"schema":"ny_measurement_start_v1"}\n', encoding="utf-8")
        return path

    def fake_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)

    fake_provenance = SimpleNamespace(
        capture_start_manifest=fake_capture_start_manifest,
        _write_immutable=fake_write,
        _json_bytes=baseline._json_bytes,
    )
    monkeypatch.setenv("NY_PHASE_TELEMETRY", "1")
    output_dir = tmp_path / "output"

    path = baseline.capture_baseline(
        manifest_path=MANIFEST,
        repo_root=REPO_ROOT,
        benchmark_root=tmp_path / "missing-benchmark",
        binary=Path("target/release/ny"),
        run_id="m0-fixture",
        output_dir=output_dir,
        provenance_module=fake_provenance,
    )

    assert len(calls) == 1
    assert calls[0]["categories_raw"]
    assert calls[0]["configs_dir"] == REPO_ROOT / "configs"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == baseline.BASELINE_SCHEMA
    assert payload["metric_contract"]["telemetry_environment"] == {
        "NY_PHASE_TELEMETRY": "1"
    }
    assert payload["resolution"]["counts"]["skipped"] > 0
    assert payload["measurement_start"]["sha256"]


def test_capture_requires_phase_telemetry_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NY_PHASE_TELEMETRY", raising=False)
    with pytest.raises(baseline.BaselineError, match="NY_PHASE_TELEMETRY=1"):
        baseline.capture_baseline(
            manifest_path=MANIFEST,
            repo_root=REPO_ROOT,
            benchmark_root=tmp_path,
            binary=Path("target/release/ny"),
            run_id="missing-gate",
            output_dir=tmp_path / "output",
            provenance_module=SimpleNamespace(),
        )


def test_capture_translates_provenance_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProvenanceError(RuntimeError):
        pass

    def fail_capture(**_kwargs):
        raise ProvenanceError("benchmark worktree could not be sealed")

    monkeypatch.setenv("NY_PHASE_TELEMETRY", "1")
    fake_provenance = SimpleNamespace(
        ProvenanceError=ProvenanceError,
        capture_start_manifest=fail_capture,
    )

    with pytest.raises(
        baseline.BaselineError,
        match="measurement provenance capture failed",
    ):
        baseline.capture_baseline(
            manifest_path=MANIFEST,
            repo_root=REPO_ROOT,
            benchmark_root=tmp_path,
            binary=Path("target/release/ny"),
            run_id="capture-failure",
            output_dir=tmp_path / "output",
            provenance_module=fake_provenance,
        )


def test_validate_cli_succeeds_with_explicit_skips(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "validate",
            "--repo-root",
            str(REPO_ROOT),
            "--benchmark-root",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["resolution"]["counts"]["skipped"] > 0
    assert payload["resolution"]["validation_errors"] == []
