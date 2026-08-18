# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "replay_vnncomp2025_counterexample.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "replay_vnncomp2025_counterexample", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = _load_module()

MESSAGE = (
    "L-inf norm difference between onnx execution and CE file output: "
    "5.5e-07 (rel error: 2.25E-06);(rel_limit: 0.001)\n"
    "Checking if spec was actually violated"
)


def _response(
    *,
    result: str = "correct",
    message: str = MESSAGE,
    diff: float = 5.5e-7,
    rel_error: float = 2.25e-6,
) -> dict[str, object]:
    return {
        "result": result,
        "message": message,
        "diff": diff,
        "rel_error": rel_error,
    }


def _file(path: Path):
    _, evidence = replay._stable_read(path, path.name)
    return evidence


def _dummy_archive(tmp_path: Path):
    root = tmp_path.resolve()
    metadata_path = root / "candidate.json"
    result_path = root / "candidate.results"
    start_path = root / "start.json"
    metadata_path.write_text("{}\n", encoding="utf-8")
    result_path.write_text("sat\n((X_0 0.0)\n(Y_0 0.0))\n", encoding="utf-8")
    start_path.write_text("{}\n", encoding="utf-8")
    assignment = b"((X_0 0.0)\n(Y_0 0.0))\n"
    onnx = SimpleNamespace(
        authoritative=SimpleNamespace(
            sha256="1" * 64,
            size_bytes=101,
            git_path="benchmarks/acasxu_2023/onnx/model.onnx.gz",
            git_blob="a" * 40,
            retained_setup_payload=None,
        )
    )
    vnnlib = SimpleNamespace(
        authoritative=SimpleNamespace(
            sha256="2" * 64,
            size_bytes=202,
            git_path="benchmarks/acasxu_2023/vnnlib/property.vnnlib",
            git_blob="b" * 40,
            retained_setup_payload=None,
        )
    )
    return SimpleNamespace(
        artifact_root=root,
        metadata_path=metadata_path,
        metadata={
            "run_id": "exact-run",
            "category": "acasxu_2023",
            "instance_index": 84,
        },
        metadata_file=_file(metadata_path),
        result_file=_file(result_path),
        assignment_bytes=assignment,
        start_file=_file(start_path),
        onnx=onnx,
        vnnlib=vnnlib,
    )


def _current_archive_fixture(tmp_path: Path) -> tuple[Path, Path, Path, object]:
    root = (tmp_path / "artifacts").resolve()
    benchmark_root = (tmp_path / "benchmarks").resolve()
    official_results = (tmp_path / "results").resolve()
    root.mkdir()
    benchmark_root.mkdir()
    official_results.mkdir()
    result_path = root / "candidate.results"
    result_path.write_bytes(b"sat\n((X_0 0.0)\n(Y_0 0.0))\n")
    result_digest = replay._sha256(result_path.read_bytes())
    result_file = str((tmp_path / "scratch" / "result.txt").resolve())
    environment_values = {"NY_NO_CUDA": "1", "PATH": "/usr/bin:/bin"}
    measurement = dict.fromkeys(replay.regular.MEASUREMENT_KEYS)
    measurement.update(
        {
            "benchmark_root": str(benchmark_root),
            "categories": ["cgan_2023"],
            "instance_index": 1,
            "result_file": result_file,
            "flight_record_file": f"{result_file}.flight.json",
            "flight_record_capture": replay.regular.FLIGHT_RECORD_CAPTURE_POLICY,
            "solver_environment": {
                "mode": "env-i-reviewed-record-v1",
                "values": environment_values,
            },
            "timeout_cap_seconds": 10,
        }
    )
    solver = dict.fromkeys(replay.regular.SOLVER_BINARY_KEYS)
    solver.update(
        {
            "fingerprint": {"mtime_ns": 3_000_000_000},
            "build_coherence": {
                "binary_mtime_epoch": 3,
                "build_inputs_last_commit_epoch": 1,
                "behaviour_inputs_last_commit_epoch": 2,
                "build_input_paths": list(replay.provenance._BUILD_INPUT_PATHS),
                "behaviour_input_paths": list(
                    replay.provenance._BEHAVIOUR_INPUT_PATHS
                ),
            },
        }
    )
    benchmark = dict.fromkeys(replay.regular.BENCHMARK_WORKTREE_KEYS)
    benchmark["benchmark_root"] = str(benchmark_root)
    start = dict.fromkeys(replay.regular.START_KEYS)
    start.update(
        {
            "schema": "ny_measurement_start_v1",
            "run_id": "current-flight",
            "measurement": measurement,
            "solver_binary": solver,
            "benchmark": benchmark,
        }
    )
    start_path = root / "start.json"
    start_path.write_text(
        json.dumps(start, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    start_digest = replay._sha256(start_path.read_bytes())
    flight_record = {
        "schema_version": 3,
        "backend_kind": "cpu-only",
        "backend_summary": "fixture CPU backend",
        "host": {
            "hostname": "fixture-host",
            "cpu_model": "fixture-cpu",
            "logical_cores": 1,
            "ram_bytes": 1024,
        },
        "load_avg_at_begin": [0.0, 0.0, 0.0],
        "load_avg_at_end": [0.0, 0.0, 0.0],
        "category": "cgan_2023",
        "budget_secs": 10,
        "ambient_env": {"NY_NO_CUDA": "1"},
        "levers": {"status": "not_materialized"},
        "events": [
            {"method": "fixture", "status": "ran", "at_secs": 0.0},
            {
                "method": "run_complete",
                "status": "complete",
                "reason": "sat",
                "at_secs": 3.0,
            },
        ],
    }
    flight_bytes = json.dumps(
        flight_record,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    metadata = dict.fromkeys(replay.regular.METADATA_KEYS)
    metadata.update(
        {
            "schema": "ny_measurement_result_v2",
            "schema_version": 2,
            "run_id": "current-flight",
            "category": "cgan_2023",
            "instance_index": 1,
            "solver_verdict": "sat",
            "solver_exit_status": 0,
            "witness_present": True,
            "counterexample_validation": {
                "checker": None,
                "status": "not_checked",
            },
            "timeout_seconds": 10,
            "elapsed_seconds": 3,
            "result_artifact": result_path.relative_to(root).as_posix(),
            "result_sha256": result_digest,
            "raw_result_sha256": result_digest,
            "start_manifest": start_path.relative_to(root).as_posix(),
            "start_manifest_sha256": start_digest,
            "flight_record": {
                "status": "captured",
                "source_sha256": replay._sha256(flight_bytes),
                "size_bytes": len(flight_bytes),
                "record": flight_record,
            },
        }
    )
    metadata_path = root / "candidate.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    occurrence = SimpleNamespace(
        category="cgan_2023",
        instance_index=1,
        onnx="onnx/model.onnx",
        vnnlib="vnnlib/property.vnnlib",
        timeout_seconds=Decimal(10),
    )
    return metadata_path, benchmark_root, official_results, occurrence


def _install_historical_v2_lever_receipt(
    metadata_path: Path,
) -> tuple[dict[str, object], dict[str, object], bytes]:
    """Rewrite the current fixture as the short-lived direct-receipt v2 shape."""

    start_path = metadata_path.parent / "start.json"
    start = json.loads(start_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    environment = start["measurement"]["solver_environment"]["values"]
    environment.update(
        {
            "NY_HISTORICAL_ACCEPTED": "1",
            "NY_HISTORICAL_REJECTED": "bogus",
        }
    )
    record = metadata["flight_record"]["record"]
    record["schema_version"] = 2
    record["ambient_env"].update(
        {
            "NY_HISTORICAL_ACCEPTED": "1",
            "NY_HISTORICAL_REJECTED": "bogus",
        }
    )
    record["levers"] = {
        "schema": replay.regular.FLIGHT_V2_LEVER_RECEIPT_SCHEMA,
        "lever_count": 3,
        "env_overridden": 1,
        "levers": [
            {
                "name": "NY_HISTORICAL_ACCEPTED",
                "value": True,
                "source": "env",
                "bucket": "debug",
                "moat": "low",
                "provenance": "unmeasured",
            },
            {
                "name": "NY_HISTORICAL_DEFAULT",
                "value": False,
                "source": "default",
                "bucket": "debug",
                "moat": "low",
                "provenance": "unmeasured",
            },
            {
                "name": "NY_HISTORICAL_REJECTED",
                "value": None,
                "source": "default",
                "bucket": "debug",
                "moat": "low",
                "provenance": "unmeasured",
                "rejected_raw": "bogus",
            },
        ],
    }
    source = replay.regular._flight_record_source_bytes(record)
    metadata["flight_record"]["source_sha256"] = replay._sha256(source)
    metadata["flight_record"]["size_bytes"] = len(source)
    start_data = (json.dumps(start, indent=2, sort_keys=True) + "\n").encode()
    start_path.write_bytes(start_data)
    metadata["start_manifest_sha256"] = replay._sha256(start_data)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return start, metadata, source


def _refresh_embedded_flight_identity(metadata: dict[str, object]) -> bytes:
    flight = metadata["flight_record"]
    assert isinstance(flight, dict)
    record = flight["record"]
    assert isinstance(record, dict)
    source = replay.regular._flight_record_source_bytes(record)
    flight["source_sha256"] = replay._sha256(source)
    flight["size_bytes"] = len(source)
    return source


def _stub_archive_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    benchmark_root: Path,
    official_results: Path,
    occurrence: object,
) -> None:
    benchmark = SimpleNamespace(benchmark_root=benchmark_root)
    official = SimpleNamespace(root=official_results)
    monkeypatch.setattr(
        replay.regular, "validate_official_results", lambda _root: official
    )
    monkeypatch.setattr(
        replay.regular, "validate_official_benchmark", lambda _root: benchmark
    )
    monkeypatch.setattr(
        replay.regular,
        "_load_occurrence",
        lambda **_kwargs: (occurrence, {}),
    )
    input_evidence = SimpleNamespace(authoritative=None, payload=b"")
    monkeypatch.setattr(
        replay,
        "_validate_declared_input",
        lambda **_kwargs: input_evidence,
    )


def _checker() -> dict[str, object]:
    return {
        "repository": replay.OFFICIAL_RESULTS_REPOSITORY,
        "commit": replay.OFFICIAL_RESULTS_COMMIT,
        "source_sha256": dict(replay.OFFICIAL_SOURCE_SHA256),
    }


def _runtime() -> dict[str, object]:
    return {
        "python_executable": str(
            replay.PINNED_RUNTIME_ROOT / replay.PINNED_PYTHON_RELATIVE
        ),
        "python_sha256": replay.PINNED_PYTHON_SHA256,
        "python_version": replay.PINNED_PYTHON_VERSION,
        "venv": str(replay.PINNED_RUNTIME_ROOT),
        "execution_scope": "host_bound_local_replay",
        "requirements_sha256": replay.OFFICIAL_SOURCE_SHA256[
            "SCORING-ZERO-TOL/requirements.txt"
        ],
        "installed_versions": dict(replay.PINNED_REQUIREMENTS),
        "onnxruntime_version": "1.16.3",
        "provider": replay.CPU_PROVIDER,
        "stdlib_manifest_sha256": replay.PINNED_STDLIB_MANIFEST_SHA256,
        "site_packages_manifest_sha256": (replay.PINNED_SITE_PACKAGES_MANIFEST_SHA256),
        "scoring_tree_manifest_sha256": replay.PINNED_SCORING_MANIFEST_SHA256,
        "native_dependencies": dict(replay.PINNED_NATIVE_DEPENDENCIES),
        "ort_pybind_upstream_sha256": replay.ORT_UPSTREAM_SHA256,
        "ort_pybind_patched_sha256": replay.ORT_PATCHED_SHA256,
        "execstack_patch": {
            "tool": "patchelf",
            "tool_version": "0.18.0",
            "operation": "--clear-execstack",
            "changed_byte_count": 1,
            "before_gnu_stack": "RWE",
            "after_gnu_stack": "RW",
        },
    }


def _harness() -> dict[str, object]:
    return {
        "runner_sha256": "3" * 64,
        "worker_sha256": replay.PINNED_WORKER_SHA256,
        "protocol": replay.WORKER_PROTOCOL,
        "import_roots": [
            str(replay.PINNED_RUNTIME_ROOT / replay.PINNED_SCORING_RELATIVE),
            str(replay.PINNED_RUNTIME_ROOT / replay.PINNED_SITE_PACKAGES_RELATIVE),
            str(replay.PINNED_RUNTIME_ROOT / replay.PINNED_STDLIB_RELATIVE),
        ],
    }


def _worker_receipt(
    response: dict[str, object] | None = None,
    *,
    input_receipts: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    response = response or _response()
    inputs = input_receipts or {
        "onnx": {"sha256": "1" * 64, "size_bytes": 101},
        "vnnlib": {"sha256": "2" * 64, "size_bytes": 202},
        "counterexample": {
            "sha256": replay._sha256(b"((X_0 0.0)\n(Y_0 0.0))\n"),
            "size_bytes": 22,
        },
    }
    request = {
        "protocol": replay.WORKER_PROTOCOL,
        "abs_tolerance": replay.COUNTEREXAMPLE_ATOL,
        "rel_tolerance": replay.COUNTEREXAMPLE_RTOL,
        **inputs,
    }
    return {
        "protocol": replay.WORKER_PROTOCOL,
        "request_sha256": replay._canonical_sha256(request),
        **inputs,
        "response_sha256": replay._canonical_sha256(response),
        "native_dependencies_sha256": replay._canonical_sha256(
            replay.PINNED_NATIVE_DEPENDENCIES
        ),
    }


def _elf_with_stack(flags: int) -> bytes:
    data = bytearray(64 + 56)
    data[:6] = b"\x7fELF\x02\x01"
    data[32:40] = (64).to_bytes(8, "little")
    data[54:56] = (56).to_bytes(2, "little")
    data[56:58] = (1).to_bytes(2, "little")
    data[64:68] = (0x6474E551).to_bytes(4, "little")
    data[68:72] = flags.to_bytes(4, "little")
    return bytes(data)


def test_parses_exact_official_metrics() -> None:
    assert replay._parse_official_metrics(MESSAGE) == (5.5e-7, 2.25e-6)
    assert replay._validate_machine_response(_response()) == _response()


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: value.update({"extra": "self-attestation"}), "exact canonical"),
        (lambda value: value.update({"diff": 1.0}), "response/message mismatch"),
        (
            lambda value: value.update({"rel_error": float("inf")}),
            "response/message mismatch",
        ),
        (lambda value: value.update({"result": "organizer_says_yes"}), "unknown"),
    ],
)
def test_machine_response_fails_closed(mutator, match: str) -> None:
    response = _response()
    mutator(response)
    with pytest.raises(replay.ReplayError, match=match):
        replay._validate_machine_response(response)


@pytest.mark.parametrize(
    "message",
    [
        "counterexample accepted without metrics",
        (
            "L-inf norm difference between onnx execution and CE file output: "
            "nan (rel error: 0);"
        ),
        (
            "L-inf norm difference between onnx execution and CE file output: "
            "0 (rel error: inf);"
        ),
        f"{MESSAGE}\n{MESSAGE}",
    ],
)
def test_metric_parser_rejects_missing_nonfinite_or_ambiguous(
    message: str,
) -> None:
    with pytest.raises(replay.ReplayError):
        replay._parse_official_metrics(message)


def test_sidecar_has_exact_promoter_contract(tmp_path: Path) -> None:
    archive = _dummy_archive(tmp_path)
    sidecar = replay._build_sidecar(
        archive=archive,
        checker=_checker(),
        harness=_harness(),
        worker_receipt=_worker_receipt(),
        runtime=_runtime(),
        response=_response(),
    )

    assert set(sidecar) == replay.TOP_KEYS
    assert set(sidecar["settings"]) == replay.SETTINGS_KEYS
    assert set(sidecar["checker"]) == replay.CHECKER_KEYS
    assert set(sidecar["harness"]) == replay.HARNESS_KEYS
    assert set(sidecar["worker_receipt"]) == replay.WORKER_RECEIPT_KEYS
    assert set(sidecar["runtime"]) == replay.RUNTIME_KEYS
    assert set(sidecar["runtime"]["execstack_patch"]) == replay.EXECSTACK_PATCH_KEYS
    assert set(sidecar["measurement"]) == replay.MEASUREMENT_KEYS
    assert set(sidecar["evidence"]) == replay.EVIDENCE_KEYS
    assert set(sidecar["response"]) == replay.RESPONSE_KEYS
    assert sidecar["schema"] == ("ny_vnncomp2025_zero_tol_counterexample_validation_v2")
    assert sidecar["official_result"] == "correct"
    assert sidecar["classification"] == "valid"
    assert sidecar["score_credit"] is True
    assert sidecar["runtime"]["execution_scope"] == "host_bound_local_replay"
    assert sidecar["settings"] == {
        "ignore_ce_y": False,
        "counterexample_atol": 1e-4,
        "counterexample_rtol": 1e-3,
        "scoring_zero_tolerance": True,
    }
    assert sidecar["evidence"]["onnx"]["official_git_blob"] == "a" * 40
    assert (
        sidecar["evidence"]["extracted_assignment"]["transformation"]
        == "removed_standalone_sat_verdict_line_only"
    )


def test_sidecar_binds_retained_setup_payload_without_false_git_identity(
    tmp_path: Path,
) -> None:
    archive = _dummy_archive(tmp_path)
    logical_path = (
        "benchmarks/cgan_2023/onnx/"
        "cGAN_imgSz32_nCh_3_small_transformer.onnx"
    )
    payload_binding = replay.regular.EXPECTED_LARGE_MODEL_MANIFEST["payloads"][
        logical_path
    ]
    retained_path = replay.regular.PINNED_LARGE_MODEL_ROOT.joinpath(
        *Path(payload_binding["retained_artifact"]).parts
    )
    source = replay.regular._retained_source_binding(
        root=replay.regular.PINNED_LARGE_MODEL_ROOT,
        manifest_path=replay.regular.PINNED_LARGE_MODEL_ROOT / "manifest.json",
        logical_path=logical_path,
        setup=replay.regular.EXPECTED_LARGE_MODEL_MANIFEST[
            "official_benchmark"
        ]["setup"],
        payload_binding=payload_binding,
        retained_path=retained_path,
    )
    archive.onnx.authoritative = SimpleNamespace(
        sha256=payload_binding["payload_sha256"],
        size_bytes=payload_binding["payload_size_bytes"],
        git_path=None,
        git_blob=None,
        retained_setup_payload=source,
    )
    input_receipts = {
        "onnx": {
            "sha256": payload_binding["payload_sha256"],
            "size_bytes": payload_binding["payload_size_bytes"],
        },
        "vnnlib": {"sha256": "2" * 64, "size_bytes": 202},
        "counterexample": {
            "sha256": replay._sha256(archive.assignment_bytes),
            "size_bytes": len(archive.assignment_bytes),
        },
    }

    sidecar = replay._build_sidecar(
        archive=archive,
        checker=_checker(),
        harness=_harness(),
        worker_receipt=_worker_receipt(input_receipts=input_receipts),
        runtime=_runtime(),
        response=_response(),
    )

    binding = sidecar["evidence"]["onnx"]
    assert set(binding) == replay.RETAINED_INPUT_BINDING_KEYS
    assert binding["official_retained_setup_payload"] == source
    assert "official_git_path" not in binding
    assert "official_git_blob" not in binding

    forged = dict(binding)
    forged_source = dict(source)
    forged_source["logical_path"] = (
        "benchmarks/cgan_2023/onnx/not-allowlisted.onnx"
    )
    forged["official_retained_setup_payload"] = forged_source
    with pytest.raises(replay.ReplayError, match="not allowlisted"):
        replay._validate_input_binding(forged, "onnx")


def test_noncredit_official_result_is_durably_invalid(tmp_path: Path) -> None:
    archive = _dummy_archive(tmp_path)
    sidecar = replay._build_sidecar(
        archive=archive,
        checker=_checker(),
        harness=_harness(),
        worker_receipt=_worker_receipt(_response(result="spec_not_violated")),
        runtime=_runtime(),
        response=_response(result="spec_not_violated"),
    )
    assert sidecar["status"] == "validated"
    assert sidecar["classification"] == "invalid"
    assert sidecar["score_credit"] is False


def test_immutable_writer_refuses_existing_file_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    destination = root / "validation.json"
    replay._write_immutable(destination, b"first\n", root)
    assert destination.read_bytes() == b"first\n"
    assert destination.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="refusing to replace"):
        replay._write_immutable(destination, b"second\n", root)
    assert destination.read_bytes() == b"first\n"

    target = root / "target.json"
    target.write_bytes(b"preserve\n")
    link = root / "link.json"
    link.symlink_to(target)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        replay._write_immutable(link, b"overwrite\n", root)
    assert target.read_bytes() == b"preserve\n"


def test_artifact_lookup_rejects_symlink_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "evidence").write_bytes(b"not archived")
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(replay.ReplayError, match="traverses a symlink"):
        replay._artifact_file(root, "redirect/evidence", "raw result")


def test_extract_assignment_removes_only_sat_line() -> None:
    assignment = b"((X_0 0.5)\n(Y_0 1.0))\n"
    assert replay._extract_assignment(b"sat\n" + assignment) == assignment
    with pytest.raises(replay.ReplayError, match="exact SAT"):
        replay._extract_assignment(b"sat extra\n" + assignment)
    with pytest.raises(replay.ReplayError, match="no submitted assignment"):
        replay._extract_assignment(b"sat\n")


def test_current_flight_archive_loads_for_exact_2025_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, benchmark_root, official_results, occurrence = (
        _current_archive_fixture(tmp_path)
    )
    _stub_archive_sources(
        monkeypatch,
        benchmark_root=benchmark_root,
        official_results=official_results,
        occurrence=occurrence,
    )

    archive = replay._load_archive(
        metadata_path=metadata,
        artifact_root=metadata.parent,
        benchmark_root=benchmark_root,
        official_results=official_results,
    )

    assert archive.metadata["flight_record"]["status"] == "captured"
    assert archive.start["measurement"]["flight_record_file"].endswith(
        ".flight.json"
    )


def test_closed_legacy_and_current_profiles_reject_partial_flight_schema(
    tmp_path: Path,
) -> None:
    metadata_path, _, _, _ = _current_archive_fixture(tmp_path)
    start = json.loads((metadata_path.parent / "start.json").read_text())
    metadata = json.loads(metadata_path.read_text())
    assert (
        replay.regular.validate_flight_record_binding(
            start=start, metadata=metadata
        )
        == replay.regular.FLIGHT_START_PROFILE
    )

    build_only_start = json.loads(json.dumps(start))
    del build_only_start["measurement"]["flight_record_file"]
    del build_only_start["measurement"]["flight_record_capture"]
    legacy_metadata = json.loads(json.dumps(metadata))
    del legacy_metadata["flight_record"]
    assert (
        replay.regular.validate_flight_record_binding(
            start=build_only_start, metadata=legacy_metadata
        )
        == replay.regular.BUILD_COHERENCE_START_PROFILE
    )

    pre_build_start = json.loads(json.dumps(build_only_start))
    del pre_build_start["solver_binary"]["build_coherence"]
    assert (
        replay.regular.validate_flight_record_binding(
            start=pre_build_start, metadata=legacy_metadata
        )
        == replay.regular.LEGACY_START_PROFILE
    )

    partial_start = json.loads(json.dumps(start))
    del partial_start["measurement"]["flight_record_capture"]
    with pytest.raises(replay.regular.EvidenceError, match="complete canonical"):
        replay.regular.validate_flight_record_binding(
            start=partial_start, metadata=metadata
        )
    with pytest.raises(replay.regular.EvidenceError, match="inconsistent"):
        replay.regular.validate_flight_record_binding(
            start=start, metadata=legacy_metadata
        )

    stale_start = json.loads(json.dumps(start))
    stale_start["solver_binary"]["build_coherence"]["binary_mtime_epoch"] = 1
    with pytest.raises(replay.regular.EvidenceError, match="build-coherence"):
        replay.regular.validate_flight_record_binding(
            start=stale_start, metadata=metadata
        )


def test_flight_record_tampering_fails_digest_and_semantic_bindings(
    tmp_path: Path,
) -> None:
    metadata_path, _, _, _ = _current_archive_fixture(tmp_path)
    start = json.loads((metadata_path.parent / "start.json").read_text())
    metadata = json.loads(metadata_path.read_text())

    digest_tamper = json.loads(json.dumps(metadata))
    digest_tamper["flight_record"]["record"]["backend_summary"] = "forged"
    with pytest.raises(replay.regular.EvidenceError, match="source bytes"):
        replay.regular.validate_flight_record_binding(
            start=start, metadata=digest_tamper
        )

    verdict_tamper = json.loads(json.dumps(metadata))
    flight = verdict_tamper["flight_record"]
    flight["record"]["events"][-1]["reason"] = "unsat"
    source = replay.regular._flight_record_source_bytes(flight["record"])
    flight["source_sha256"] = replay._sha256(source)
    flight["size_bytes"] = len(source)
    with pytest.raises(replay.regular.EvidenceError, match="terminal verdict"):
        replay.regular.validate_flight_record_binding(
            start=start, metadata=verdict_tamper
        )

    environment_tamper = json.loads(json.dumps(metadata))
    flight = environment_tamper["flight_record"]
    flight["record"]["ambient_env"]["NY_FORGED"] = "1"
    source = replay.regular._flight_record_source_bytes(flight["record"])
    flight["source_sha256"] = replay._sha256(source)
    flight["size_bytes"] = len(source)
    with pytest.raises(replay.regular.EvidenceError, match="sealed execution"):
        replay.regular.validate_flight_record_binding(
            start=start, metadata=environment_tamper
        )


def test_schema2_flight_source_bytes_match_serde_json_float_spelling(
    tmp_path: Path,
) -> None:
    metadata_path, _, _, _ = _current_archive_fixture(tmp_path)
    start = json.loads((metadata_path.parent / "start.json").read_text())
    metadata = json.loads(metadata_path.read_text())
    record = metadata["flight_record"]["record"]
    record["schema_version"] = 2
    del record["levers"]
    record["events"][0]["at_secs"] = 5.77e-6
    record["events"][1]["at_secs"] = 0.00001251

    source = replay.regular._flight_record_source_bytes(record)

    assert b'"at_secs": 5.77e-6' in source
    assert b'"at_secs": 0.00001251' in source
    assert b"e-06" not in source
    metadata["flight_record"]["source_sha256"] = replay._sha256(source)
    metadata["flight_record"]["size_bytes"] = len(source)
    assert (
        replay.regular.validate_flight_record_binding(
            start=start, metadata=metadata
        )
        == replay.regular.FLIGHT_START_PROFILE
    )


def test_historical_schema2_direct_lever_receipt_is_bound_byte_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata_path, benchmark_root, official_results, occurrence = (
        _current_archive_fixture(tmp_path)
    )
    start, metadata, source = _install_historical_v2_lever_receipt(
        metadata_path
    )
    record = metadata["flight_record"]["record"]
    expected_record = {
        "schema_version": record["schema_version"],
        "backend_kind": record["backend_kind"],
        "backend_summary": record["backend_summary"],
        "host": {
            key: record["host"][key]
            for key in ("hostname", "cpu_model", "logical_cores", "ram_bytes")
        },
        "load_avg_at_begin": record["load_avg_at_begin"],
        "load_avg_at_end": record["load_avg_at_end"],
        "category": record["category"],
        "budget_secs": record["budget_secs"],
        "ambient_env": dict(sorted(record["ambient_env"].items())),
        "levers": json.loads(json.dumps(record["levers"], sort_keys=True)),
        "events": [
            {
                key: event[key]
                for key in ("method", "status", "reason", "at_secs")
                if key in event
            }
            for event in record["events"]
        ],
    }
    expected_source = json.dumps(
        expected_record, ensure_ascii=False, indent=2, allow_nan=False
    ).encode()

    assert source == expected_source
    assert replay._sha256(source) == (
        "f02740c1c18eab395d13bec090130a0dfdbaf1855579c8547c0d082393697da9"
    )
    assert source.index(b'"ambient_env"') < source.index(b'"levers"')
    assert source.index(b'"env_overridden"') < source.index(b'"schema"')
    assert (
        replay.regular.validate_flight_record_binding(
            start=start, metadata=metadata
        )
        == replay.regular.FLIGHT_START_PROFILE
    )
    _stub_archive_sources(
        monkeypatch,
        benchmark_root=benchmark_root,
        official_results=official_results,
        occurrence=occurrence,
    )
    loaded = replay._load_archive(
        metadata_path=metadata_path,
        artifact_root=metadata_path.parent,
        benchmark_root=benchmark_root,
        official_results=official_results,
    )
    assert loaded.metadata["flight_record"]["record"]["levers"] == (
        metadata["flight_record"]["record"]["levers"]
    )


def test_flight_lever_schema_mixing_and_v2_receipt_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    metadata_path, _, _, _ = _current_archive_fixture(tmp_path)
    current_start = json.loads(
        (metadata_path.parent / "start.json").read_text()
    )
    current_metadata = json.loads(metadata_path.read_text())

    v2_with_v3_envelope = json.loads(json.dumps(current_metadata))
    v2_with_v3_envelope["flight_record"]["record"]["schema_version"] = 2
    _refresh_embedded_flight_identity(v2_with_v3_envelope)
    with pytest.raises(
        replay.regular.EvidenceError, match="v2 lever receipt"
    ):
        replay.regular.validate_flight_record_binding(
            start=current_start, metadata=v2_with_v3_envelope
        )

    historical_start, historical_metadata, _ = (
        _install_historical_v2_lever_receipt(metadata_path)
    )
    v3_with_v1_receipt = json.loads(json.dumps(historical_metadata))
    v3_with_v1_receipt["flight_record"]["record"]["schema_version"] = 3
    with pytest.raises(replay.regular.EvidenceError, match="lever receipt"):
        replay.regular.validate_flight_record_binding(
            start=historical_start, metadata=v3_with_v1_receipt
        )

    bad_count = json.loads(json.dumps(historical_metadata))
    bad_count["flight_record"]["record"]["levers"]["lever_count"] = 4
    _refresh_embedded_flight_identity(bad_count)
    with pytest.raises(
        replay.regular.EvidenceError, match="v2 lever receipt"
    ):
        replay.regular.validate_flight_record_binding(
            start=historical_start, metadata=bad_count
        )

    unbound_environment = json.loads(json.dumps(historical_metadata))
    unbound_environment["flight_record"]["record"]["levers"]["levers"][
        0
    ]["name"] = "NY_NOT_IN_AMBIENT_ENV"
    _refresh_embedded_flight_identity(unbound_environment)
    with pytest.raises(
        replay.regular.EvidenceError, match="v2 lever receipt"
    ):
        replay.regular.validate_flight_record_binding(
            start=historical_start, metadata=unbound_environment
        )

    mismatched_rejection = json.loads(json.dumps(historical_metadata))
    mismatched_rejection["flight_record"]["record"]["levers"]["levers"][
        2
    ]["rejected_raw"] = "forged"
    _refresh_embedded_flight_identity(mismatched_rejection)
    with pytest.raises(
        replay.regular.EvidenceError, match="v2 lever receipt"
    ):
        replay.regular.validate_flight_record_binding(
            start=historical_start, metadata=mismatched_rejection
        )


def test_current_profile_accepts_explicitly_missing_flight_record(
    tmp_path: Path,
) -> None:
    metadata_path, _, _, _ = _current_archive_fixture(tmp_path)
    start = json.loads((metadata_path.parent / "start.json").read_text())
    metadata = json.loads(metadata_path.read_text())
    metadata["flight_record"] = {"status": "missing"}

    assert (
        replay.regular.validate_flight_record_binding(
            start=start, metadata=metadata
        )
        == replay.regular.FLIGHT_START_PROFILE
    )


def test_gnu_stack_parser_observes_one_byte_execstack_change() -> None:
    upstream = _elf_with_stack(7)
    patched = _elf_with_stack(6)
    assert replay._gnu_stack_flags(upstream, "upstream") == "RWE"
    assert replay._gnu_stack_flags(patched, "patched") == "RW"
    assert len(upstream) == len(patched)
    assert sum(left != right for left, right in zip(upstream, patched)) == 1
    with pytest.raises(replay.ReplayError, match="ELF64"):
        replay._gnu_stack_flags(b"not an elf", "adversarial")


def test_strict_json_rejects_nan() -> None:
    with pytest.raises(replay.ReplayError, match="non-finite JSON"):
        replay._strict_json_response(b'{"diff": NaN}', "adversarial worker")
    assert math.isfinite(
        replay._strict_json_response(b'{"diff": 0.0}', "worker")["diff"]
    )


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(replay.ReplayError, match="duplicate JSON key"):
        replay._strict_json_response(
            b'{"result":"correct","result":"forged"}',
            "adversarial worker",
        )


def test_worker_receipt_binds_every_consumed_byte() -> None:
    onnx = b"authoritative onnx"
    vnnlib = b"authoritative vnnlib"
    assignment = b"((X_0 0.0)\n(Y_0 0.0))\n"
    response = _response()
    inputs = {
        "onnx": replay._payload_receipt(onnx),
        "vnnlib": replay._payload_receipt(vnnlib),
        "counterexample": replay._payload_receipt(assignment),
    }
    request = {
        "protocol": replay.WORKER_PROTOCOL,
        "abs_tolerance": replay.COUNTEREXAMPLE_ATOL,
        "rel_tolerance": replay.COUNTEREXAMPLE_RTOL,
        **inputs,
    }
    receipt = {
        "protocol": replay.WORKER_PROTOCOL,
        "request_sha256": replay._canonical_sha256(request),
        **inputs,
        "response_sha256": replay._canonical_sha256(response),
        "native_dependencies_sha256": replay._canonical_sha256(
            replay.PINNED_NATIVE_DEPENDENCIES
        ),
    }
    assert (
        replay._validate_worker_receipt(
            receipt,
            onnx_payload=onnx,
            vnnlib_payload=vnnlib,
            assignment_bytes=assignment,
            response=response,
            native_dependencies=replay.PINNED_NATIVE_DEPENDENCIES,
        )
        == receipt
    )
    forged = {
        **receipt,
        "onnx": {"sha256": "0" * 64, "size_bytes": len(onnx)},
    }
    with pytest.raises(replay.ReplayError, match="supplied bytes"):
        replay._validate_worker_receipt(
            forged,
            onnx_payload=onnx,
            vnnlib_payload=vnnlib,
            assignment_bytes=assignment,
            response=response,
            native_dependencies=replay.PINNED_NATIVE_DEPENDENCIES,
        )


def test_effective_timeout_accepts_capped_sat() -> None:
    assert replay._validate_effective_timing(
        metadata={"timeout_seconds": 30, "elapsed_seconds": 29},
        measurement={"timeout_cap_seconds": 30},
        official_timeout=Decimal(116),
    ) == Decimal(30)


def test_effective_timeout_rejects_uncapped_or_overbudget_sat() -> None:
    with pytest.raises(replay.ReplayError, match="effective official/capped"):
        replay._validate_effective_timing(
            metadata={"timeout_seconds": 116, "elapsed_seconds": 29},
            measurement={"timeout_cap_seconds": 30},
            official_timeout=Decimal(116),
        )
    with pytest.raises(replay.ReplayError, match="over-budget"):
        replay._validate_effective_timing(
            metadata={"timeout_seconds": 30, "elapsed_seconds": 31},
            measurement={"timeout_cap_seconds": 30},
            official_timeout=Decimal(116),
        )


def test_tree_manifest_rejects_writable_cache_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sealed"
    root.mkdir()
    payload = root / "module.py"
    payload.write_text("VALUE = 1\n", encoding="utf-8")
    payload.chmod(0o444)
    root.chmod(0o555)
    first = replay._tree_manifest_sha256(root, "fixture")
    assert first == replay._tree_manifest_sha256(root, "fixture")

    cache = root / "__pycache__"
    shadow = root / "shadow.py"
    try:
        root.chmod(0o755)
        cache.mkdir()
        cache.chmod(0o555)
        root.chmod(0o555)
        with pytest.raises(replay.ReplayError, match="unsafe cache"):
            replay._tree_manifest_sha256(root, "fixture")
        root.chmod(0o755)
        cache.rmdir()
        shadow.symlink_to(payload)
        root.chmod(0o555)
        with pytest.raises(replay.ReplayError, match="cache/symlink"):
            replay._tree_manifest_sha256(root, "fixture")
    finally:
        root.chmod(0o755)
        if shadow.is_symlink():
            shadow.unlink()
        if cache.exists():
            cache.chmod(0o755)
            cache.rmdir()
        payload.chmod(0o644)


def test_batch_snapshot_avoids_per_row_global_rehash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {"harness": _harness(), "runtime": _runtime()}
    onnx = b"onnx"
    vnnlib = b"vnnlib"
    assignment = b"((X_0 0.0)\n(Y_0 0.0))\n"
    response = _response()
    inputs = {
        "onnx": replay._payload_receipt(onnx),
        "vnnlib": replay._payload_receipt(vnnlib),
        "counterexample": replay._payload_receipt(assignment),
    }
    request = {
        "protocol": replay.WORKER_PROTOCOL,
        "abs_tolerance": replay.COUNTEREXAMPLE_ATOL,
        "rel_tolerance": replay.COUNTEREXAMPLE_RTOL,
        **inputs,
    }
    receipt = {
        "protocol": replay.WORKER_PROTOCOL,
        "request_sha256": replay._canonical_sha256(request),
        **inputs,
        "response_sha256": replay._canonical_sha256(response),
        "native_dependencies_sha256": replay._canonical_sha256(
            replay.PINNED_NATIVE_DEPENDENCIES
        ),
    }
    monkeypatch.setattr(
        replay,
        "_harness_identity",
        lambda _root: pytest.fail("batch row rehashed harness"),
    )
    monkeypatch.setattr(
        replay,
        "_runtime_identity",
        lambda _root: pytest.fail("batch row rehashed runtime"),
    )
    monkeypatch.setattr(
        replay,
        "_invoke_exact_worker",
        lambda **_kwargs: (response, receipt),
    )
    monkeypatch.setattr(replay, "_pinned_runtime_root", lambda root: root)
    observed = replay.replay_bound_payloads(
        onnx_payload=onnx,
        vnnlib_payload=vnnlib,
        assignment_bytes=assignment,
        snapshot=snapshot,
    )
    assert observed == {
        **snapshot,
        "response": response,
        "worker_receipt": receipt,
    }


def test_batch_snapshot_final_recheck_detects_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(replay, "_pinned_runtime_root", lambda root: root)
    monkeypatch.setattr(replay, "_harness_identity", lambda _root: _harness())
    monkeypatch.setattr(replay, "_runtime_identity", lambda _root: _runtime())
    snapshot = replay.capture_replay_snapshot()
    changed_runtime = _runtime()
    changed_runtime["python_sha256"] = "f" * 64
    monkeypatch.setattr(replay, "_runtime_identity", lambda _root: changed_runtime)
    with pytest.raises(replay.ReplayError, match="changed during batch"):
        replay.revalidate_replay_snapshot(snapshot)


def test_canonical_sidecar_suffix_is_not_legacy() -> None:
    path = replay._canonical_sidecar(Path("/archive/run.json"))
    assert path.name == "run.vnncomp2025-zero-tol-validation.json"
    assert "counterexample-validation" not in path.name


def external_retained_runtime_matches_all_exact_pins() -> None:
    assert replay.PINNED_RUNTIME_ROOT.exists(), (
        "retained exact 2025 checker runtime is not installed: "
        f"{replay.PINNED_RUNTIME_ROOT}"
    )
    runtime = replay._runtime_identity(replay.PINNED_RUNTIME_ROOT)
    assert runtime == _runtime()


def external_official_checker_sources_match_commit_and_retained_copy() -> None:
    official_results = Path("<home>/ay/benchmarks/vnncomp/2025/results")
    assert official_results.exists(), (
        "pinned official 2025 results checkout is not installed: "
        f"{official_results}"
    )
    checker = replay._checker_identity(
        official_results,
        replay.PINNED_RUNTIME_ROOT,
    )
    assert checker == _checker()


ACAS_MODEL = Path(
    "<home>/ay/benchmarks/vnncomp/2025/benchmarks/benchmarks/"
    "acasxu_2023/onnx/ACASXU_run2a_5_3_batch_2000.onnx"
)
ACAS_PROPERTY = Path(
    "<home>/ay/benchmarks/vnncomp/2025/benchmarks/benchmarks/"
    "acasxu_2023/vnnlib/prop_2.vnnlib"
)
RETAINED_RUNNER = replay.PINNED_RUNTIME_ROOT / replay.PINNED_RETAINED_RUNNER_RELATIVE


def external_consumer_safe_bound_replay_executes_retained_worker() -> None:
    missing = [
        path
        for path in (ACAS_MODEL, ACAS_PROPERTY, RETAINED_RUNNER)
        if not path.exists()
    ]
    assert not missing, f"exact retained replay integration inputs are missing: {missing!r}"
    assignment = b"""((X_0 0.6798577308654785)
(X_1 -0.0066666677594184875)
(X_2 -0.046666666865348816)
(X_3 0.45000001788139343)
(X_4 -0.45000001788139343)
(Y_0 0.02272483892738819)
(Y_1 0.022669341415166855)
(Y_2 -0.018353242427110672)
(Y_3 0.0222333911806345)
(Y_4 -0.016658682376146317))
"""
    result = replay.replay_bound_payloads(
        onnx_payload=ACAS_MODEL.read_bytes(),
        vnnlib_payload=ACAS_PROPERTY.read_bytes(),
        assignment_bytes=assignment,
        timeout_seconds=60,
    )
    assert result["response"]["result"] == "correct"
    assert result["worker_receipt"]["counterexample"] == (
        replay._payload_receipt(assignment)
    )
    assert result["harness"]["worker_sha256"] == replay.PINNED_WORKER_SHA256
