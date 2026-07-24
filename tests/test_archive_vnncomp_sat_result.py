# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "archive_vnncomp_sat_result.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("archive_vnncomp_sat_result", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


archive = _load_module()


def _archive(
    tmp_path: Path,
    result_file: Path,
    *,
    solver_verdict: str = "sat",
    instance_index: int = 7,
) -> Path:
    artifact_root = tmp_path / "artifacts"
    start_manifest = artifact_root / "runs" / "20260718T120000Z-123" / "start.json"
    start_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not start_manifest.exists():
        start_manifest.write_text(
            json.dumps(
                {
                    "schema": "ny_measurement_start_v1",
                    "run_id": "20260718T120000Z-123",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    onnx_file = tmp_path / "a.onnx"
    vnnlib_file = tmp_path / "p.vnnlib"
    if not onnx_file.exists():
        onnx_file.write_bytes(b"onnx fixture")
    if not vnnlib_file.exists():
        vnnlib_file.write_text("; property fixture\n", encoding="utf-8")
    solver_log_file = tmp_path / "solver.log"
    if not solver_log_file.exists():
        solver_log_file.write_bytes(b"solver stdout\nsolver stderr\n")
    preflight_manifest = archive.seal_inputs(
        artifact_root=artifact_root,
        run_id="20260718T120000Z-123",
        category="acasxu_2023",
        instance_index=instance_index,
        onnx="onnx/a.onnx",
        vnnlib="vnnlib/p.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        start_manifest=start_manifest,
    )
    return archive.archive_result(
        result_file=result_file,
        solver_log_file=solver_log_file,
        artifact_root=artifact_root,
        run_id="20260718T120000Z-123",
        category="acasxu_2023",
        instance_index=instance_index,
        onnx="onnx/a.onnx",
        vnnlib="vnnlib/p.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        solver_verdict=solver_verdict,
        solver_exit_status=0,
        timeout_seconds=116,
        elapsed_seconds=9,
        source_csv="reports/measured/acasxu_2023.csv",
        start_manifest=start_manifest,
        preflight_manifest=preflight_manifest,
    )


def test_archives_complete_witness_and_not_checked_metadata(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    complete_result = b"sat\n((X_0 1.25)\n (Y_0 -2.0))\n"
    result_file.write_bytes(complete_result)

    archived = _archive(tmp_path, result_file)

    assert archived.read_bytes() == complete_result
    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["witness_present"] is True
    assert metadata["counterexample_validation"]["status"] == "not_checked"
    assert (
        metadata["result_sha256"] == archive.hashlib.sha256(complete_result).hexdigest()
    )
    assert metadata["raw_result_sha256"] == metadata["result_sha256"]
    assert (
        metadata["onnx"]["sha256"]
        == archive.hashlib.sha256(b"onnx fixture").hexdigest()
    )
    assert (
        metadata["vnnlib"]["sha256"]
        == archive.hashlib.sha256(b"; property fixture\n").hexdigest()
    )
    solver_log = archived.with_name(archived.name.replace(".results", ".solver.log"))
    assert solver_log.read_bytes() == b"solver stdout\nsolver stderr\n"
    assert (
        metadata["solver_log"]["sha256"]
        == archive.hashlib.sha256(solver_log.read_bytes()).hexdigest()
    )
    start_manifest = tmp_path / "artifacts/runs/20260718T120000Z-123/start.json"
    assert metadata["start_manifest"] == ("runs/20260718T120000Z-123/start.json")
    assert (
        metadata["start_manifest_sha256"]
        == archive.hashlib.sha256(start_manifest.read_bytes()).hexdigest()
    )


def test_archives_non_sat_result_and_marks_validation_not_applicable(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")

    archived = _archive(tmp_path, result_file, solver_verdict="unsat")

    assert archived.read_bytes() == b"unsat\n"
    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["solver_verdict"] == "unsat"
    assert metadata["witness_present"] is False
    assert metadata["counterexample_validation"]["status"] == "not_applicable"


def test_archives_empty_timeout_result_bytes(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_bytes(b"")

    archived = _archive(tmp_path, result_file, solver_verdict="timeout")

    assert archived.read_bytes() == b""
    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["raw_result_sha256"] == archive.hashlib.sha256(b"").hexdigest()


def test_preflight_rejects_transient_input_swap_and_restore(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    run_id = "20260718T120000Z-swap"
    start_manifest = artifact_root / "runs" / run_id / "start.json"
    start_manifest.parent.mkdir(parents=True)
    start_manifest.write_text(
        json.dumps({"schema": "ny_measurement_start_v1", "run_id": run_id}) + "\n",
        encoding="utf-8",
    )
    onnx_file = tmp_path / "a.onnx"
    vnnlib_file = tmp_path / "p.vnnlib"
    original = b"onnx fixture"
    onnx_file.write_bytes(original)
    vnnlib_file.write_text("; property fixture\n", encoding="utf-8")
    preflight = archive.seal_inputs(
        artifact_root=artifact_root,
        run_id=run_id,
        category="acasxu_2023",
        instance_index=1,
        onnx="a.onnx",
        vnnlib="p.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        start_manifest=start_manifest,
    )

    # Restore the exact bytes and size. The pre-run stat identity must still
    # expose the transient replacement instead of accepting end-state equality.
    onnx_file.write_bytes(b"ONNX fixture")
    onnx_file.write_bytes(original)

    with pytest.raises(ValueError, match="preflight-bound input drifted"):
        archive.validate_input_preflight(
            preflight_manifest=preflight,
            artifact_root=artifact_root,
            run_id=run_id,
            category="acasxu_2023",
            instance_index=1,
            onnx="a.onnx",
            vnnlib="p.vnnlib",
            onnx_file=onnx_file,
            vnnlib_file=vnnlib_file,
            start_manifest=start_manifest,
        )


def test_preflight_rejects_manifest_symlink(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_bytes(b"unsat\n")
    archived = _archive(tmp_path, result_file, solver_verdict="unsat")
    preflight = archived.with_suffix(".preflight.json")
    backing = preflight.with_name("preflight-backing.json")
    preflight.rename(backing)
    preflight.symlink_to(backing)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _archive(tmp_path, result_file, solver_verdict="unsat")


def test_rejects_sat_without_assignment(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="structured counterexample"):
        _archive(tmp_path, result_file)


def test_rejects_sat_with_arbitrary_nonempty_payload(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\nthis-is-not-an-assignment\n", encoding="utf-8")

    with pytest.raises(ValueError, match="structured counterexample"):
        _archive(tmp_path, result_file)


def test_archives_structurally_complete_vnnlib2_assignment(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    complete_result = "sat\nX float32 [1, 2]\n0.25\n-0.5\nY float32 [1, 1]\n1.75\n"
    result_file.write_text(complete_result, encoding="utf-8")

    archived = _archive(tmp_path, result_file)
    assert archived.read_text(encoding="utf-8") == complete_result


def test_rejects_truncated_vnnlib2_assignment(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\nX float32 [1, 2]\n0.25\n", encoding="utf-8")

    with pytest.raises(ValueError, match="structured counterexample"):
        _archive(tmp_path, result_file)


def test_never_replaces_different_evidence_for_same_run(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n((X_0 1.0))\n", encoding="utf-8")
    archived = _archive(tmp_path, result_file)
    result_file.write_text("sat\n((X_0 2.0))\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _archive(tmp_path, result_file)

    assert "1.0" in archived.read_text(encoding="utf-8")


def test_identical_retry_is_idempotent(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n((X_0 1.0))\n", encoding="utf-8")

    first = _archive(tmp_path, result_file)
    original_metadata = first.with_suffix(".json").read_bytes()
    second = _archive(tmp_path, result_file)

    assert second == first
    assert first.with_suffix(".json").read_bytes() == original_metadata


def test_never_replaces_different_solver_log_for_same_run(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n((X_0 1.0))\n", encoding="utf-8")
    archived = _archive(tmp_path, result_file)
    (tmp_path / "solver.log").write_text("different log\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _archive(tmp_path, result_file)

    archived_log = archived.with_name(archived.name.replace(".results", ".solver.log"))
    assert archived_log.read_bytes() == b"solver stdout\nsolver stderr\n"


def test_run_cache_hashes_repeated_inputs_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n((X_0 1.0))\n", encoding="utf-8")
    original = archive._sha256_file
    hashed_paths: list[Path] = []

    def counting_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return original(path)

    monkeypatch.setattr(archive, "_sha256_file", counting_sha256)
    _archive(tmp_path, result_file, instance_index=7)
    _archive(tmp_path, result_file, instance_index=8)

    assert [path.name for path in hashed_paths].count("a.onnx") >= 2
    assert [path.name for path in hashed_paths].count("p.vnnlib") >= 2
    second_metadata = next(
        (tmp_path / "artifacts/acasxu_2023").glob("00008-*/20260718T120000Z-123.json")
    )
    metadata = json.loads(second_metadata.read_text(encoding="utf-8"))
    assert metadata["onnx"]["hash_cache_hit"] is True
    assert metadata["vnnlib"]["hash_cache_hit"] is True


def test_run_cache_rehashes_same_size_changed_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n((X_0 1.0))\n", encoding="utf-8")
    original = archive._sha256_file
    hashed_paths: list[Path] = []

    def counting_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return original(path)

    monkeypatch.setattr(archive, "_sha256_file", counting_sha256)
    _archive(tmp_path, result_file, instance_index=7)
    model = tmp_path / "a.onnx"
    assert len(b"onnx fixture") == len(b"ONNX fixture")
    model.write_bytes(b"ONNX fixture")
    _archive(tmp_path, result_file, instance_index=8)

    assert [path.name for path in hashed_paths].count("a.onnx") >= 2
    cache = json.loads(
        (
            tmp_path / "artifacts/runs/20260718T120000Z-123/input_hash_cache.json"
        ).read_text(encoding="utf-8")
    )
    assert len(cache["entries"]) == 3
