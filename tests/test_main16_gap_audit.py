from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "main16_gap_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("main16_gap_audit", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> tuple[str, int]:
    data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha(data), len(data)


def _official_fixture(tmp_path: Path, *, first_truth: str = "unsat") -> Path:
    root = tmp_path / "official"
    reference = root / "alpha_beta_crown/results.csv"
    reference.parent.mkdir(parents=True)
    reference.write_text(
        "".join(
            f"{category},onnx/{category}.onnx,vnnlib/{category}.vnnlib,0,unsat,1\n"
            for category in audit.retro.REGULAR
        ),
        encoding="utf-8",
    )
    latex = root / "SCORING-ZERO-TOL/latex"
    latex.mkdir(parents=True)
    longtable_lines = []
    scored_lines = []
    for index, category in enumerate(audit.retro.REGULAR):
        result = first_truth if index == 0 else "unsat"
        display = category.replace("_", " ")
        longtable_lines.append(f"2025 {display} & 0 & \\textsc{{{result}}}\n")
        scored_lines.extend(
            [
                f"% Category 2025_{category} fixture\n",
                "0 & tool & x & x & x & x & 10 & x \\\\\n",
            ]
        )
    (latex / "longtable.tex").write_text("".join(longtable_lines), encoding="utf-8")
    (latex / "scored.tex").write_text("".join(scored_lines), encoding="utf-8")
    return root


def _legacy_fixture(tmp_path: Path, *, first_verdict: str = "unsat") -> Path:
    root = tmp_path / "legacy"
    root.mkdir()
    for index, category in enumerate(audit.retro.REGULAR):
        verdict = first_verdict if index == 0 else "unsat"
        (root / f"{category}.csv").write_text(
            f"{category},onnx/{category}.onnx,vnnlib/{category}.vnnlib,0,{verdict},1\n",
            encoding="utf-8",
        )
    return root


def _evidence(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "artifact": path.relative_to(root).as_posix(),
        "sha256": _sha(data),
        "size_bytes": len(data),
    }


def _sealed_run(
    root: Path,
    *,
    commit: str,
    run_id: str = "sealed-run",
    category: str | None = None,
    verdict: str = "unsat",
    completion: bool = True,
    replay_result: str | None = None,
) -> Path:
    category = category or audit.retro.REGULAR[0]
    run_dir = root / "runs" / run_id
    start_path = run_dir / "start.json"
    start = {
        "schema": "ny_measurement_start_v1",
        "run_id": run_id,
        "ny": {
            "commit": commit,
            "clean": True,
            "status_porcelain_v1_z_entries": [],
            "tracked_diff_bytes": 0,
            "untracked_files": [],
        },
        "measurement": {
            "artifact_root": str(root.resolve()),
            "categories": [category],
        },
    }
    start_digest, start_size = _write_json(start_path, start)

    instance = root / category / "00001-fixture"
    sealed_onnx = run_dir / "sealed/inputs/onnx/model.onnx"
    sealed_vnnlib = run_dir / "sealed/inputs/vnnlib/property.vnnlib"
    sealed_onnx.parent.mkdir(parents=True)
    sealed_vnnlib.parent.mkdir(parents=True)
    sealed_onnx.write_bytes(b"model\n")
    sealed_vnnlib.write_bytes(b"property\n")
    preflight_path = instance / f"{run_id}.preflight.json"
    preflight = {
        "schema": "fixture",
        "inputs": {
            "onnx": {
                "original_sha256": _sha(b"model\n"),
                "sealed_sha256": _sha(b"model\n"),
                "sealed_artifact": sealed_onnx.relative_to(root).as_posix(),
            },
            "vnnlib": {
                "original_sha256": _sha(b"property\n"),
                "sealed_sha256": _sha(b"property\n"),
                "sealed_artifact": sealed_vnnlib.relative_to(root).as_posix(),
            },
        },
    }
    _write_json(preflight_path, preflight)

    result_path = instance / f"{run_id}.result.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(
        b"sat\n((X_0 0.5))\n" if verdict == "sat" else f"{verdict}\n".encode()
    )
    log_path = instance / f"{run_id}.solver.log"
    log_path.write_bytes(b"solver log\n")
    metadata_path = instance / f"{run_id}.json"
    metadata = {
        "schema": "ny_measurement_result_v2",
        "run_id": run_id,
        "category": category,
        "instance_index": 1,
        "solver_verdict": verdict,
        "start_manifest": start_path.relative_to(root).as_posix(),
        "start_manifest_sha256": start_digest,
        "result_artifact": result_path.relative_to(root).as_posix(),
        "result_sha256": _sha(result_path.read_bytes()),
        "raw_result_sha256": _sha(result_path.read_bytes()),
    }
    _write_json(metadata_path, metadata)
    record = {
        "category": category,
        "instance_index": 1,
        "onnx": f"onnx/{category}.onnx",
        "vnnlib": f"vnnlib/{category}.vnnlib",
        "solver_verdict": verdict,
        "metadata": _evidence(metadata_path, root),
        "result": _evidence(result_path, root),
        "solver_log": _evidence(log_path, root),
        "preflight": {**_evidence(preflight_path, root), "inputs": preflight["inputs"]},
    }
    if replay_result is not None:
        status, classification, credit = audit.replay._classification(replay_result)
        sidecar = {
            "schema": "ny_counterexample_validation_v1",
            "schema_version": 1,
            "status": status,
            "classification": classification,
            "official_result": replay_result,
            "score_credit": credit,
            "provider": audit.replay.CPU_PROVIDER,
            "checker": {"commit": audit.replay.PINNED_CHECKER_COMMIT},
            "vnnlib_python_source": {
                "commit": audit.replay.PINNED_VNNLIB_PYTHON_COMMIT
            },
            "measurement": {
                "run_id": run_id,
                "category": category,
                "instance_index": 1,
            },
            "evidence": {
                "metadata": _evidence(metadata_path, root),
                "raw_result": _evidence(result_path, root),
                "start_manifest": {
                    "artifact": start_path.relative_to(root).as_posix(),
                    "sha256": start_digest,
                    "size_bytes": start_size,
                },
            },
        }
        _write_json(
            metadata_path.with_name(
                f"{metadata_path.stem}.counterexample-validation.json"
            ),
            sidecar,
        )
    if completion:
        run_evidence = {
            "status": "valid",
            "metadata_count": 1,
            "result_count": 1,
            "solver_log_count": 1,
            "preflight_count": 1,
            "validated_record_count": 1,
            "csv_row_count": 1,
            "records": [record],
            "records_sha256": audit.provenance._identity_sha256([record]),
        }
        _write_json(
            run_dir / "completion.json",
            {
                "schema": "ny_measurement_completion_v1",
                "run_id": run_id,
                "exit_status": 0,
                "completed_successfully": True,
                "start_manifest": "start.json",
                "start_manifest_sha256": start_digest,
                "integrity": {
                    "status": "valid",
                    "violations": [],
                    "checks": {"run_evidence": run_evidence},
                },
            },
        )
    return run_dir


def test_empty_exact_bank_cannot_inherit_legacy_projection(tmp_path: Path) -> None:
    official = _official_fixture(tmp_path)
    legacy = _legacy_fixture(tmp_path)
    artifacts = tmp_path / "empty-artifacts"
    artifacts.mkdir()
    commit = "1" * 40

    report, incomplete = audit.build_audit(
        official_root=official,
        legacy_measured_root=legacy,
        artifact_roots=[artifacts],
        exact_commit=commit,
    )

    assert incomplete is False
    assert report["legacy_projection"]["score"] == 1600.0
    assert report["legacy_projection"]["evidence_tier"] == audit.LEGACY_TIER
    qualified = report["qualified_current"]
    assert qualified["evidence_tier"] == audit.QUALIFIED_TIER
    assert qualified["score"] == 0.0
    assert qualified["qualified_solved"] == 0
    assert qualified["qualified_incorrect"] == 0
    assert qualified["unmeasured"] == 16
    assert "legacy_optimistic_projection" in audit.render_csv(report)
    assert "EVIDENCE TIERS MUST NOT BE COMBINED" in audit.render_table(report)


def test_only_requested_exact_commit_completion_scores(tmp_path: Path) -> None:
    official = _official_fixture(tmp_path)
    legacy = _legacy_fixture(tmp_path, first_verdict="timeout")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    commit = "2" * 40
    _sealed_run(artifacts, commit="3" * 40, run_id="other-commit")
    _sealed_run(artifacts, commit=commit, run_id="exact-commit")

    report, incomplete = audit.build_audit(
        official_root=official,
        legacy_measured_root=legacy,
        artifact_roots=[artifacts],
        exact_commit=commit,
    )

    assert incomplete is False
    first_legacy = report["legacy_projection"]["suites"][0]
    assert first_legacy["score"] == 0.0
    assert first_legacy["min_extra_credits_to_100"] == 1
    qualified = report["qualified_current"]
    assert qualified["qualified_solved"] == 1
    assert qualified["score"] == 100.0
    assert qualified["qualification_audit"]["ignored_other_commit_runs"] == 1


def test_sat_requires_replay_and_validated_invalid_is_penalized(tmp_path: Path) -> None:
    official = _official_fixture(tmp_path)
    legacy = _legacy_fixture(tmp_path)
    commit = "4" * 40

    missing = tmp_path / "missing-replay"
    missing.mkdir()
    _sealed_run(missing, commit=commit, verdict="sat")
    missing_report, incomplete = audit.build_audit(
        official_root=official,
        legacy_measured_root=legacy,
        artifact_roots=[missing],
        exact_commit=commit,
    )
    assert incomplete is False
    first = missing_report["qualified_current"]["suites"][0]
    assert first["qualified_solved"] == 0
    assert first["qualified_incorrect"] == 0
    assert first["unqualified_sat"] == 1

    invalid = tmp_path / "invalid-replay"
    invalid.mkdir()
    _sealed_run(invalid, commit=commit, verdict="sat", replay_result="no_ce")
    invalid_report, incomplete = audit.build_audit(
        official_root=official,
        legacy_measured_root=legacy,
        artifact_roots=[invalid],
        exact_commit=commit,
    )
    assert incomplete is False
    first = invalid_report["qualified_current"]["suites"][0]
    assert first["qualified_solved"] == 0
    assert first["qualified_incorrect"] == 1
    assert first["raw_points"] == -150
    assert first["score"] == 0.0


def test_missing_completion_is_reported_and_zeroed(tmp_path: Path) -> None:
    official = _official_fixture(tmp_path)
    legacy = _legacy_fixture(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    commit = "5" * 40
    _sealed_run(artifacts, commit=commit, completion=False)

    report, incomplete = audit.build_audit(
        official_root=official,
        legacy_measured_root=legacy,
        artifact_roots=[artifacts],
        exact_commit=commit,
    )

    assert incomplete is True
    qualified = report["qualified_current"]
    assert qualified["score"] == 0.0
    assert qualified["qualification_audit"]["status"] == "incomplete_fail_closed"
    assert (
        "no completion"
        in qualified["qualification_audit"]["rejected_runs"][0]["reason"]
    )


def test_tampered_sealed_result_is_reported_and_zeroed(tmp_path: Path) -> None:
    official = _official_fixture(tmp_path)
    legacy = _legacy_fixture(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    commit = "8" * 40
    _sealed_run(artifacts, commit=commit)
    result = next((artifacts / audit.retro.REGULAR[0]).rglob("*.result.txt"))
    result.write_text("sat\n((X_0 0.5))\n", encoding="utf-8")

    report, incomplete = audit.build_audit(
        official_root=official,
        legacy_measured_root=legacy,
        artifact_roots=[artifacts],
        exact_commit=commit,
    )

    assert incomplete is True
    qualified = report["qualified_current"]
    assert qualified["qualified_solved"] == 0
    assert (
        "raw result artifact does not match"
        in qualified["qualification_audit"]["rejected_runs"][0]["reason"]
    )


def test_duplicate_exact_rows_are_ambiguous_and_zeroed(tmp_path: Path) -> None:
    official = _official_fixture(tmp_path)
    legacy = _legacy_fixture(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    commit = "6" * 40
    _sealed_run(artifacts, commit=commit, run_id="first")
    _sealed_run(artifacts, commit=commit, run_id="second")

    report, incomplete = audit.build_audit(
        official_root=official,
        legacy_measured_root=legacy,
        artifact_roots=[artifacts],
        exact_commit=commit,
    )

    assert incomplete is True
    qualified = report["qualified_current"]
    assert qualified["qualified_solved"] == 0
    assert qualified["qualification_audit"]["ambiguous_rows"] == [
        {
            "suite": audit.retro.REGULAR[0],
            "instance_index": 1,
            "run_ids": ["first", "second"],
        }
    ]


def test_cli_json_and_side_outputs_are_deterministic(tmp_path: Path) -> None:
    official = _official_fixture(tmp_path)
    legacy = _legacy_fixture(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    json_out = tmp_path / "audit.json"
    csv_out = tmp_path / "audit.csv"
    command = [
        sys.executable,
        str(SCRIPT),
        "--official",
        str(official),
        "--legacy-measured",
        str(legacy),
        "--artifact-root",
        str(artifacts),
        "--exact-commit",
        "7" * 40,
        "--format",
        "json",
        "--json-out",
        str(json_out),
        "--csv-out",
        str(csv_out),
    ]
    first = subprocess.run(command, check=True, capture_output=True)
    first_json = json_out.read_bytes()
    first_csv = csv_out.read_bytes()
    second = subprocess.run(command, check=True, capture_output=True)

    assert first.stdout == second.stdout == first_json == json_out.read_bytes()
    assert first_csv == csv_out.read_bytes()
    assert len(first_csv.splitlines()) == 18
    parsed = json.loads(first.stdout)
    assert parsed["schema"] == audit.SCHEMA


def test_help_names_exact_commit_and_artifact_scope() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--exact-commit" in result.stdout
    assert "--artifact-root" in result.stdout
