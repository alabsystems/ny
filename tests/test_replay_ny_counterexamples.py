# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "replay_ny_counterexamples.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("replay_ny_counterexamples", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_archive(
    tmp_path: Path,
    *,
    verdict: str = "sat",
    raw_result: bytes = b"sat\n((X_0 0.5)\n(Y_0 1.0))\n",
    onnx_source: Path | None = None,
    vnnlib_source: Path | None = None,
) -> tuple[Path, Path]:
    artifact_root = tmp_path / "artifacts"
    run_id = "20260718T120000Z-123"
    start = artifact_root / "runs" / run_id / "start.json"
    start.parent.mkdir(parents=True)
    start.write_text(
        json.dumps({"schema": "ny_measurement_start_v1", "run_id": run_id}) + "\n",
        encoding="utf-8",
    )
    if onnx_source is None:
        onnx_source = tmp_path / "model.onnx"
        onnx_source.write_bytes(b"onnx fixture")
    if vnnlib_source is None:
        vnnlib_source = tmp_path / "property.vnnlib"
        vnnlib_source.write_text("; vnnlib fixture\n", encoding="utf-8")
    onnx_source = onnx_source.resolve()
    vnnlib_source = vnnlib_source.resolve()

    instance = artifact_root / "acasxu_2023" / "00000-deadbeefdeadbeef"
    instance.mkdir(parents=True)
    result = instance / f"{run_id}.results"
    result.write_bytes(raw_result)
    metadata = instance / f"{run_id}.json"
    payload = {
        "schema": "ny_measurement_result_v2",
        "schema_version": 2,
        "run_id": run_id,
        "category": "acasxu_2023",
        "instance_index": 0,
        "solver_verdict": verdict,
        "witness_present": verdict == "sat",
        "counterexample_validation": {
            "status": "not_checked" if verdict == "sat" else "not_applicable",
            "checker": None,
        },
        "result_artifact": result.relative_to(artifact_root).as_posix(),
        "raw_result_sha256": _sha(result),
        "result_sha256": _sha(result),
        "start_manifest": start.relative_to(artifact_root).as_posix(),
        "start_manifest_sha256": _sha(start),
        "onnx": {
            "declared_path": "onnx/model.onnx",
            "resolved_path": str(onnx_source),
            "sha256": _sha(onnx_source),
            "size_bytes": onnx_source.stat().st_size,
        },
        "vnnlib": {
            "declared_path": "vnnlib/property.vnnlib",
            "resolved_path": str(vnnlib_source),
            "sha256": _sha(vnnlib_source),
            "size_bytes": vnnlib_source.stat().st_size,
        },
    }
    metadata.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return artifact_root, metadata


def _patch_identities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        replay,
        "_checker_identity",
        lambda _path: {"commit": replay.PINNED_CHECKER_COMMIT, "source_sha256": {}},
    )
    monkeypatch.setattr(
        replay,
        "_vnnlib_source_identity",
        lambda _path: {"commit": replay.PINNED_VNNLIB_PYTHON_COMMIT},
    )
    monkeypatch.setattr(
        replay,
        "_require_exact_venv",
        lambda python, prefix: {"executable": str(python), "prefix": str(prefix)},
    )


def _fake_response(result: str = "correct") -> dict[str, object]:
    return {
        "ok": True,
        "result": result,
        "rationale": "minimal official-checker boundary",
        "provider": replay.CPU_PROVIDER,
        "dependency_versions": {"onnxruntime": "test"},
        "installed_vnnlib_files_sha256": {"vnnlib/_core.so": "0" * 64},
        "available_providers": [replay.CPU_PROVIDER],
    }


def test_replays_full_assignment_and_writes_immutable_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root, metadata = _make_archive(tmp_path)
    _patch_identities(monkeypatch)
    observed: dict[str, bytes] = {}

    def invoke(**kwargs):
        evidence = kwargs["evidence"]
        observed["assignment"] = evidence.assignment_bytes
        observed["archive"] = evidence.result_bytes
        return _fake_response()

    sidecar = replay.replay_archive(
        metadata_path=metadata,
        artifact_root=artifact_root,
        checker_repo=tmp_path,
        checker_python=tmp_path / "venv/bin/python",
        checker_venv=tmp_path / "venv",
        vnnlib_source=tmp_path,
        invoke=invoke,
    )

    assert observed["archive"] == b"sat\n((X_0 0.5)\n(Y_0 1.0))\n"
    assert observed["assignment"] == b"((X_0 0.5)\n(Y_0 1.0))\n"
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["status"] == "validated"
    assert record["classification"] == "strictly_correct"
    assert record["provider"] == "CPUExecutionProvider"
    assert record["evidence"]["metadata"]["sha256"] == _sha(metadata)
    assert (
        record["evidence"]["raw_result"]["sha256"]
        == hashlib.sha256(observed["archive"]).hexdigest()
    )
    assert (
        record["evidence"]["extracted_assignment"]["sha256"]
        == hashlib.sha256(observed["assignment"]).hexdigest()
    )

    with pytest.raises(FileExistsError, match="refusing to replace"):
        replay.replay_archive(
            metadata_path=metadata,
            artifact_root=artifact_root,
            checker_repo=tmp_path,
            checker_python=tmp_path / "venv/bin/python",
            checker_venv=tmp_path / "venv",
            vnnlib_source=tmp_path,
            invoke=lambda **_kwargs: pytest.fail("checker must not run on retry"),
        )


def test_rejects_tampered_model_before_checker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root, metadata = _make_archive(tmp_path)
    _patch_identities(monkeypatch)
    (tmp_path / "model.onnx").write_bytes(b"tampered")

    with pytest.raises(replay.ReplayError, match="ONNX SHA-256 mismatch"):
        replay.replay_archive(
            metadata_path=metadata,
            artifact_root=artifact_root,
            checker_repo=tmp_path,
            checker_python=tmp_path / "venv/bin/python",
            checker_venv=tmp_path / "venv",
            vnnlib_source=tmp_path,
            invoke=lambda **_kwargs: pytest.fail("checker saw tampered evidence"),
        )


def test_rejects_non_sat_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root, metadata = _make_archive(
        tmp_path, verdict="unsat", raw_result=b"unsat\n"
    )
    _patch_identities(monkeypatch)

    with pytest.raises(replay.ReplayError, match="only metadata for a SAT"):
        replay.replay_archive(
            metadata_path=metadata,
            artifact_root=artifact_root,
            checker_repo=tmp_path,
            checker_python=tmp_path / "venv/bin/python",
            checker_venv=tmp_path / "venv",
            vnnlib_source=tmp_path,
            invoke=lambda **_kwargs: pytest.fail("checker saw non-SAT evidence"),
        )


def test_records_official_malformed_classification_without_rewriting_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = b"sat\n((X_f[0] 0.25)\n(Y_f[0] 1.0))\n"
    artifact_root, metadata = _make_archive(tmp_path, raw_result=raw)
    _patch_identities(monkeypatch)

    sidecar = replay.replay_archive(
        metadata_path=metadata,
        artifact_root=artifact_root,
        checker_repo=tmp_path,
        checker_python=tmp_path / "venv/bin/python",
        checker_venv=tmp_path / "venv",
        vnnlib_source=tmp_path,
        invoke=lambda **_kwargs: _fake_response("malformed_ce"),
    )

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["classification"] == "invalid"
    result_path = next(artifact_root.glob("acasxu_2023/*/*.results"))
    assert result_path.read_bytes() == raw


def test_real_pinned_official_v1_checker_smoke(tmp_path: Path) -> None:
    checker_repo = REPO_ROOT / "external_tools" / "vnncomp2026_results"
    vnnlib_source_repo = REPO_ROOT / "external_tools" / "VNNLIB-Python"
    checker_python = Path("/home/ayates/.venvs/vnncomp-ce-2026/bin/python")
    benchmark_link = REPO_ROOT / "external_tools" / "vnncomp2026_benchmarks"
    benchmark_repo = REPO_ROOT / "benchmarks" / "vnncomp2025"
    official_result = (
        REPO_ROOT
        / "external_tools/vnncomp2025_results/alpha_beta_crown/2025_acasxu_2023"
        / "ACASXU_run2a_2_9_batch_2000_prop_2.counterexample.gz"
    )
    prerequisites = (
        checker_repo / ".git",
        vnnlib_source_repo / ".git",
        checker_python,
        benchmark_link,
        official_result,
    )
    if not all(path.exists() for path in prerequisites):
        pytest.skip("pinned checker/corpus/runtime is not installed")
    try:
        if importlib.metadata.version("vnnlib") != "1.0.2":
            pytest.skip("vnnlib 1.0.2 is not installed in this pytest interpreter")
    except importlib.metadata.PackageNotFoundError:
        # The replay itself runs in the dedicated interpreter; the test runner
        # does not need the package, so continue when that interpreter has it.
        pass

    model = (
        benchmark_repo / "benchmarks/acasxu_2023/onnx/ACASXU_run2a_2_9_batch_2000.onnx"
    )
    prop = benchmark_repo / "benchmarks/acasxu_2023/vnnlib/prop_2.vnnlib"
    if not model.is_file() or not prop.is_file():
        pytest.skip("2025 ACAS Xu smoke inputs are not installed")
    raw = b"sat\n" + gzip.open(official_result, "rb").read()
    artifact_root, metadata = _make_archive(
        tmp_path,
        raw_result=raw,
        onnx_source=model,
        vnnlib_source=prop,
    )

    sidecar = replay.replay_archive(
        metadata_path=metadata,
        artifact_root=artifact_root,
        checker_repo=checker_repo,
        checker_python=checker_python,
        checker_venv=checker_python.parent.parent,
        vnnlib_source=vnnlib_source_repo,
        timeout_seconds=120,
    )

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["official_result"] in {"correct", "correct_up_to_tolerance"}
    assert record["provider"] == "CPUExecutionProvider"
    assert record["checker"]["commit"] == replay.PINNED_CHECKER_COMMIT
    assert record["checker_runtime"]["dependency_versions"]["vnnlib"] == "1.0.2"


def test_real_pinned_official_v2_checker_and_malformed_ny_syntax_smoke(
    tmp_path: Path,
) -> None:
    checker_repo = REPO_ROOT / "external_tools" / "vnncomp2026_results"
    vnnlib_source_repo = REPO_ROOT / "external_tools" / "VNNLIB-Python"
    checker_python = Path("/home/ayates/.venvs/vnncomp-ce-2026/bin/python")
    benchmark_link = REPO_ROOT / "external_tools" / "vnncomp2026_benchmarks"
    benchmark = REPO_ROOT / "benchmarks/vnncomp2026/benchmarks/test/2.0"
    model = benchmark / "onnx/test_nano.onnx"
    prop = benchmark / "vnnlib/test_nano.vnnlib"
    prerequisites = (
        checker_repo / ".git",
        vnnlib_source_repo / ".git",
        checker_python,
        benchmark_link,
        model,
        prop,
    )
    if not all(path.exists() for path in prerequisites):
        pytest.skip("pinned checker, VNN-LIB 2.0 corpus, or runtime is not installed")

    well_formed_root, well_formed_metadata = _make_archive(
        tmp_path / "well-formed",
        raw_result=(b"sat\nX float32 [1]\n-1.0\nY float32 [1]\n-1.0\n"),
        onnx_source=model,
        vnnlib_source=prop,
    )
    well_formed_sidecar = replay.replay_archive(
        metadata_path=well_formed_metadata,
        artifact_root=well_formed_root,
        checker_repo=checker_repo,
        checker_python=checker_python,
        checker_venv=checker_python.parent.parent,
        vnnlib_source=vnnlib_source_repo,
        timeout_seconds=120,
    )
    well_formed = json.loads(well_formed_sidecar.read_text(encoding="utf-8"))
    # test_nano's model cannot satisfy Y <= -1.  A syntactically valid
    # assignment therefore reaches ONNX CPU replay and is rejected on the
    # specification, proving this is more than a parser-only smoke.
    assert well_formed["official_result"] == "spec_not_violated"
    assert well_formed["classification"] == "invalid"
    assert well_formed["vnnlib_version"] == "2.0"

    # NY's current relational emitter uses legacy s-expressions such as
    # `(X_f[i] value)`.  The 2.0 checker must see those bytes unchanged and
    # classify them as malformed instead of a replay adapter silently
    # translating them into section-5.3 assignment syntax.
    malformed_raw = b"sat\n((X_f[0] -1.0)\n(Y_f[0] -1.0))\n"
    malformed_root, malformed_metadata = _make_archive(
        tmp_path / "malformed",
        raw_result=malformed_raw,
        onnx_source=model,
        vnnlib_source=prop,
    )
    malformed_sidecar = replay.replay_archive(
        metadata_path=malformed_metadata,
        artifact_root=malformed_root,
        checker_repo=checker_repo,
        checker_python=checker_python,
        checker_venv=checker_python.parent.parent,
        vnnlib_source=vnnlib_source_repo,
        timeout_seconds=120,
    )
    malformed = json.loads(malformed_sidecar.read_text(encoding="utf-8"))
    assert malformed["official_result"] == "malformed_ce"
    assert malformed["classification"] == "invalid"
    archived_result = next(malformed_root.glob("acasxu_2023/*/*.results"))
    assert archived_result.read_bytes() == malformed_raw


def test_patched_relational_assignment_format_is_strict_official_witness(
    tmp_path: Path,
) -> None:
    """Cross-check NY's new tensor serializer shape/order against the checker."""
    checker_repo = REPO_ROOT / "external_tools" / "vnncomp2026_results"
    checker_python = Path("/home/ayates/.venvs/vnncomp-ce-2026/bin/python")
    benchmark = (
        REPO_ROOT / "benchmarks/vnncomp2026/benchmarks/monotonic_acasxu_2026/2.0"
    )
    model = benchmark / "onnx/ACASXU_run2a_2_4_batch_2000.onnx"
    prop = benchmark / "vnnlib/instance_0.vnnlib"
    if not all(
        path.exists() for path in (checker_repo / ".git", checker_python, model, prop)
    ):
        pytest.skip("pinned checker, relational corpus, or runtime is not installed")

    # This is exactly the section-5.3 header/scalar order emitted by NY's
    # patched `relational_counterexample_vnnlib`: f input/output followed by g
    # input/output, using the declaration's real type and full tensor shapes.
    # The selected inputs strictly satisfy instance_0's unsafe relation under
    # CPU ONNX replay; serialized Y values are deliberately ignored by policy.
    assignment = """\
X_f real [1, 1, 1, 5]
0.5
-0.1
0.3
0.227272727
0.25
Y_f real [1, 5]
0
0
0
0
0
X_g real [1, 1, 1, 5]
-0.1
-0.1
0.3
0.227272727
0.25
Y_g real [1, 5]
0
0
0
0
0
"""
    witness = tmp_path / "ny-relational.counterexample"
    witness.write_text(assignment, encoding="utf-8")
    network_field = repr([("f", str(model)), ("g", str(model))])
    worker = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])
from counterexamples_v2 import validate_vnnlib2_counterexample

class Result:
    CORRECT = "correct"
    CORRECT_UP_TO_TOLERANCE = "correct_up_to_tolerance"
    NO_CE = "no_ce"
    EXEC_DOESNT_MATCH = "exec_doesnt_match"
    SPEC_NOT_VIOLATED = "spec_not_violated"
    WRONG_SHAPE = "wrong_shape"
    MALFORMED_CE = "malformed_ce"
    UNSUPPORTED = "unsupported"

result, rationale = validate_vnnlib2_counterexample(
    sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
    1e-4, 0.0, Result, True,
)
print(json.dumps({"result": result, "rationale": rationale}))
"""
    completed = subprocess.run(
        [
            str(checker_python),
            "-c",
            worker,
            str(checker_repo / "SCORING"),
            str(benchmark),
            network_field,
            str(prop),
            str(witness),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        env={"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    assert response["result"] == "correct", response["rationale"]
