# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ny_retroactive_scorecard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ny_retroactive_scorecard", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scorecard = _load_module()


def test_load_tool_csv_excludes_harness_test_rows(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    results.write_text(
        "cat,onnx/model.onnx,vnnlib/property.vnnlib,0,unsat,1\n"
        "cat,onnx/test_nano.onnx,vnnlib/test_nano.vnnlib,0,unsat,1\n"
        "cat,onnx/test_tiny.onnx,vnnlib/test_tiny.vnnlib,0,unsat,1\n",
        encoding="utf-8",
    )

    loaded = scorecard.load_tool_csv(results)

    assert len(loaded["cat"]) == 1


def test_load_tool_csv_preserves_duplicate_instance_occurrences(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    row = "sat_relu,onnx/sat_v6_c27.onnx,vnnlib/sat_v6_c27.vnnlib,0,sat,1\n"
    results.write_text(row + row, encoding="utf-8")

    loaded = scorecard.load_tool_csv(results)["sat_relu"]

    assert len(loaded) == 2
    assert {key[-1] for key in loaded} == {0, 1}
    assert set(loaded.values()) == {"violated"}


def test_load_tool_csv_tolerates_trailing_measurement_run_id(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    results.write_text(
        "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,1,20260718T120000Z-123\n",
        encoding="utf-8",
    )

    loaded = scorecard.load_tool_csv(results)

    assert list(loaded["acasxu_2023"].values()) == ["holds"]


def _write_instances(
    tmp_path: Path,
    category: str,
    rows: list[str],
    *,
    version: str | None = None,
) -> Path:
    category_dir = tmp_path / "benchmarks" / category
    if version is not None:
        category_dir /= version
    category_dir.mkdir(parents=True, exist_ok=True)
    instances = category_dir / "instances.csv"
    instances.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
    return tmp_path / "benchmarks"


def test_ny_measurement_rejects_over_budget_decision_from_exact_occurrence(
    tmp_path: Path,
) -> None:
    benchmark_root = _write_instances(
        tmp_path,
        "cora_2024",
        ["./onnx/cifar10-point.onnx,vnnlib/cifar10-img440.vnnlib,30"],
    )
    measured = tmp_path / "cora_2024.csv"
    measured.write_text(
        "cora_2024,./onnx/cifar10-point.onnx,"
        "vnnlib/cifar10-img440.vnnlib,prepared,sat,38.70,run-440\n",
        encoding="utf-8",
    )

    with pytest.raises(
        scorecard.MeasurementBudgetError,
        match=r"refusing over-budget violated credit at 38\.70s.*"
        r"official budget is 30s.*instances\.csv:1.*sha256=",
    ):
        scorecard.load_ny_measured_csv(measured, benchmark_root)


def test_ny_measurement_duplicate_pair_requires_exact_full_order(
    tmp_path: Path,
) -> None:
    benchmark_root = _write_instances(
        tmp_path,
        "sat_relu",
        [
            "onnx/duplicate.onnx,vnnlib/duplicate.vnnlib,30",
            "onnx/unique.onnx,vnnlib/unique.vnnlib,40",
            "onnx/duplicate.onnx,vnnlib/duplicate.vnnlib,60",
        ],
    )
    partial = tmp_path / "partial.csv"
    partial.write_text(
        "sat_relu,onnx/duplicate.onnx,vnnlib/duplicate.vnnlib,0,unsat,25,run-a\n",
        encoding="utf-8",
    )

    with pytest.raises(
        scorecard.MeasurementBudgetError,
        match=r"cannot bind .* exactly; 2 official occurrences",
    ):
        scorecard.load_ny_measured_csv(partial, benchmark_root)

    complete = tmp_path / "complete.csv"
    complete.write_text(
        "sat_relu,onnx/duplicate.onnx,vnnlib/duplicate.vnnlib,0,unsat,25,run-b\n"
        "sat_relu,onnx/unique.onnx,vnnlib/unique.vnnlib,0,sat,35,run-b\n"
        "sat_relu,onnx/duplicate.onnx,vnnlib/duplicate.vnnlib,0,unsat,55,run-b\n",
        encoding="utf-8",
    )
    loaded = scorecard.load_ny_measured_csv(complete, benchmark_root)["sat_relu"]

    duplicate_keys = [
        instance
        for instance in loaded
        if instance[:2]
        == scorecard.key("onnx/duplicate.onnx", "vnnlib/duplicate.vnnlib")
    ]
    assert {instance[-1] for instance in duplicate_keys} == {0, 1}
    assert set(loaded.values()) == {"holds", "violated"}


def test_ny_measurement_fails_closed_on_ambiguous_instance_lists(
    tmp_path: Path,
) -> None:
    benchmark_root = _write_instances(
        tmp_path,
        "acasxu_2023",
        ["onnx/a.onnx,vnnlib/a.vnnlib,116"],
        version="1.0",
    )
    _write_instances(
        tmp_path,
        "acasxu_2023",
        ["onnx/a.onnx,vnnlib/a.vnnlib,116"],
        version="2.0",
    )
    measured = tmp_path / "acasxu.csv"
    measured.write_text(
        "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,10\n",
        encoding="utf-8",
    )

    with pytest.raises(
        scorecard.MeasurementBudgetError,
        match=r"expected exactly one instances\.csv.*found 2",
    ):
        scorecard.load_ny_measured_csv(measured, benchmark_root)


def test_ny_measurement_prefers_official_top_level_list_over_nested_payload(
    tmp_path: Path,
) -> None:
    benchmark_root = _write_instances(
        tmp_path,
        "cora_2024",
        ["./onnx/a.onnx,vnnlib/a.vnnlib,30"],
    )
    nested = benchmark_root / "cora_2024" / "vnnlib"
    nested.mkdir()
    (nested / "instances.csv").write_text(
        "./nns/a.onnx,./benchmark-files/a.vnnlib,300\n",
        encoding="utf-8",
    )
    measured = tmp_path / "cora.csv"
    measured.write_text(
        "cora_2024,./onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,30\n",
        encoding="utf-8",
    )

    loaded = scorecard.load_ny_measured_csv(measured, benchmark_root)

    assert list(loaded["cora_2024"].values()) == ["holds"]


def test_ny_measurement_noncredit_timeout_can_exceed_budget(
    tmp_path: Path,
) -> None:
    benchmark_root = _write_instances(
        tmp_path,
        "cora_2024",
        ["onnx/a.onnx,vnnlib/a.vnnlib,30.0"],
    )
    measured = tmp_path / "cora.csv"
    measured.write_text(
        "cora_2024,onnx/a.onnx,vnnlib/a.vnnlib,0,timeout,45\n",
        encoding="utf-8",
    )

    loaded = scorecard.load_ny_measured_csv(measured, benchmark_root)

    assert list(loaded["cora_2024"].values()) == ["timeout"]


def test_ny_measurement_decimal_budget_has_exact_runtime_boundary(
    tmp_path: Path,
) -> None:
    benchmark_root = _write_instances(
        tmp_path,
        "cora_2024",
        ["onnx/a.onnx,vnnlib/a.vnnlib,30.5"],
    )
    measured = tmp_path / "cora.csv"
    measured.write_text(
        "cora_2024,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,30.500\n",
        encoding="utf-8",
    )

    loaded = scorecard.load_ny_measured_csv(measured, benchmark_root)

    assert list(loaded["cora_2024"].values()) == ["holds"]

    measured.write_text(
        "cora_2024,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,30.5000000001\n",
        encoding="utf-8",
    )
    with pytest.raises(
        scorecard.MeasurementBudgetError,
        match=r"refusing over-budget holds credit at 30\.5000000001s.*"
        r"official budget is 30\.5s",
    ):
        scorecard.load_ny_measured_csv(measured, benchmark_root)


def test_main_refuses_to_score_over_budget_ny_row(tmp_path: Path) -> None:
    benchmark_root = _write_instances(
        tmp_path,
        "cora_2024",
        ["onnx/a.onnx,vnnlib/a.vnnlib,30"],
    )
    measured_dir = tmp_path / "measured"
    measured_dir.mkdir()
    (measured_dir / "cora_2024.csv").write_text(
        "cora_2024,onnx/a.onnx,vnnlib/a.vnnlib,0,sat,38.70,run-a\n",
        encoding="utf-8",
    )
    official_dir = tmp_path / "official"
    official_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--official",
            str(official_dir),
            "--measured",
            str(measured_dir),
            "--benchmark-root",
            str(benchmark_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "NY measurement budget validation failed" in result.stderr
    assert "refusing over-budget violated credit at 38.70s" in result.stderr


def test_published_artifacts_map_ids_and_preserve_official_denominator(
    tmp_path: Path,
) -> None:
    reference_csv = tmp_path / "results.csv"
    reference_csv.write_text(
        "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,1\n"
        "acasxu_2023,onnx/test_nano.onnx,vnnlib/test_nano.vnnlib,0,unsat,1\n"
        "acasxu_2023,onnx/b.onnx,vnnlib/b.vnnlib,0,sat,1\n",
        encoding="utf-8",
    )
    longtable = tmp_path / "longtable.tex"
    longtable.write_text(
        "2025 Acasxu 2023 & 0 & ~\\textsc{unsat} & - \\\\\n"
        "2025 Acasxu 2023 & 1 & ~\\textsc{sat} & - \\\\\n",
        encoding="utf-8",
    )
    reference = scorecard.load_reference_instance_order(reference_csv)
    ground_truth = scorecard.load_published_ground_truth(longtable, reference)
    first, second = reference["acasxu_2023"]

    assert len(reference["acasxu_2023"]) == 2
    assert ground_truth["acasxu_2023"] == {
        first: "holds",
        second: "violated",
    }

    ny = {"acasxu_2023": {first: "holds", second: "violated"}}
    winner_points = dict.fromkeys(scorecard.REGULAR, 100)
    total, breakdown = scorecard.published_artifact_projection(
        ny,
        reference,
        ground_truth,
        winner_points,
        ny_sat_status="correct-up-to-tolerance",
    )

    # NY gets 20 raw but remains normalized by the published 100-point winner,
    # not a raw-verdict-recomputed denominator of 20.
    assert breakdown["acasxu_2023"][:3] == (20, 20.0, 100)
    assert total == 20.0


def test_official_artifact_mode_requires_explicit_ny_sat_assumption() -> None:
    # The requirement is enforced in main because argparse cannot express the
    # conditional cleanly; guard the accepted status vocabulary at unit level.
    with pytest.raises(ValueError, match="unsupported NY SAT status"):
        scorecard.published_artifact_projection(
            {},
            {},
            {},
            dict.fromkeys(scorecard.REGULAR, 1),
            ny_sat_status="strict-but-unchecked",
        )


def _ext_fixture(
    tmp_path: Path,
    *,
    track: str,
    gt_rows: list[tuple[str, str, bool]],
    win_score: int,
    ny_csv_rows: list[str],
) -> tuple[Path, Path]:
    """Synthetic SCORING-ZERO-TOL results.txt plus a ny extended measured dir."""
    lines = [f"Category 2025_{track}:", "  participating tools: ['alpha_beta_crown']"]
    for rid, true_result, valid_ce in gt_rows:
        marker = ", 'x (v)'" if valid_ce else ""
        lines.append(f"Row: ['{rid}'{marker}]")
        lines.append(f"True Result: {true_result}")
    lines.append(f"0: alpha_beta_crown score: {win_score},")
    results_txt = tmp_path / "results.txt"
    results_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ext_dir = tmp_path / "measured-ext"
    ext_dir.mkdir()
    (ext_dir / f"{track}.csv").write_text(
        "".join(row + "\n" for row in ny_csv_rows), encoding="utf-8"
    )
    return results_txt, ext_dir


def test_score_extended_folds_penalty_into_projection_and_reports_bad(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="lsnc_relu",
        gt_rows=[("net-prop_0", "sat", True), ("net-prop_1", "unsat", False)],
        win_score=20,
        ny_csv_rows=[
            "lsnc_relu,net.onnx,prop_0.vnnlib,unsat,1.0",
            "lsnc_relu,net.onnx,prop_1.vnnlib,unsat,1.0",
        ],
    )

    grand, bad = scorecard.score_extended(results_txt, ext_dir, 0.0)
    out = capsys.readouterr().out

    # rate 1/2 over N=2 extrapolates to +10 credit; the measured -150 penalty
    # must survive into proj_raw, and the normalized score floors at 0.
    row = next(line for line in out.splitlines() if line.startswith("lsnc_relu"))
    assert bad == 1
    assert grand == 0.0
    # proj_raw, proj_norm, upside_norm — no unconfirmable rows here, so the
    # upside column equals the defensible one.
    assert row.split()[-3:] == ["-140", "0.0", "0.0"]
    assert "*** SCORECARD FAILURE: 1 blocking extended verdict(s)" in out
    assert "(1 scored -150, 0 tolerance-CE contradiction(s)" in out


def test_score_extended_prints_measured_only_norm_beside_projection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gt_rows: list[tuple[str, str, bool]] = [("net-prop_0", "unsat", False)]
    gt_rows += [(f"net-prop_{i}", "-", False) for i in range(1, 10)]
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="lsnc_relu",
        gt_rows=gt_rows,
        win_score=200,
        ny_csv_rows=["lsnc_relu,net.onnx,prop_0.vnnlib,unsat,1.0"],
    )

    grand, bad = scorecard.score_extended(results_txt, ext_dir, 0.0)
    out = capsys.readouterr().out

    # 1/1 sampled ok extrapolates to 100 raw (norm 50.0); measured-only is
    # 100*10/200 = 5.0 and must be printed beside the projection.
    row = next(line for line in out.splitlines() if line.startswith("lsnc_relu"))
    assert bad == 0
    assert grand == 50.0
    assert row.split()[-5:] == ["10", "5.0", "100", "50.0", "50.0"]
    assert "EXTENDED-9 measured-only normalized" in out


def test_extended_sat_credit_warns_without_evidence_and_rejects_evidence_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="ml4acopf_2024",
        gt_rows=[("net-prop_0", "sat", True)],
        win_score=10,
        ny_csv_rows=["ml4acopf_2024,net.onnx,prop_0.vnnlib,sat,1.0"],
    )

    grand, bad = scorecard.score_extended(results_txt, ext_dir, 0.0)
    out = capsys.readouterr().out
    assert (grand, bad) == (100.0, 0)
    assert "WARNING: 1 extended sat row(s) credited WITHOUT validated evidence" in out
    assert "no-evidence: ml4acopf_2024 net-prop_0" in out

    with pytest.raises(scorecard.MeasurementBudgetError, match="unsupported"):
        scorecard.score_extended(
            results_txt, ext_dir, 0.0, require_evidence=True
        )

    # A filename that merely resembles old validate_bank output cannot restore
    # credit: no extended evidence schema currently binds its contents.
    evidence_dir = ext_dir / "evidence" / "ml4acopf_2024"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / f"prop_0-{'0' * 32}.validation.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(scorecard.MeasurementBudgetError, match="unsupported"):
        scorecard.score_extended(
            results_txt, ext_dir, 0.0, require_evidence=True
        )


def test_main_rejects_extended_with_official_artifacts() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--official-artifacts",
            "--ny-sat-status",
            "unvalidated",
            "--extended",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--extended cannot be combined with --official-artifacts" in result.stderr
    assert "ZERO-TOL denominators" in result.stderr


def test_main_regular_evidence_mode_fails_closed_when_index_is_missing(
    tmp_path: Path,
) -> None:
    measured = tmp_path / "measured"
    measured.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--require-evidence",
            "--official-artifacts",
            "--ny-sat-status",
            "correct-up-to-tolerance",
            "--measured",
            str(measured),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "NY regular evidence validation failed" in result.stderr
    assert "regular_evidence_index.json" in result.stderr
    assert "unavailable" in result.stderr


def test_regular_evidence_loader_credits_only_applied_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import regular_bank_evidence as bank_evidence  # noqa: PLC0415

    occurrence = SimpleNamespace(score_key=("onnx/a.onnx", "vnnlib/a.vnnlib", 0))
    sealed = SimpleNamespace(
        category="cora_2024",
        occurrence=occurrence,
        verdict="unsat",
    )
    indexed = SimpleNamespace(
        row_key='["cora_2024","onnx/a.onnx","vnnlib/a.vnnlib",0]',
        evidence=sealed,
    )
    validated = SimpleNamespace(
        creditable_entries=(indexed,),
        dangling_entries=(),
    )
    monkeypatch.setattr(
        bank_evidence,
        "validate_regular_evidence_index",
        lambda **_: validated,
    )

    loaded, returned, sat_results = scorecard.load_evidenced_ny_regular(
        evidence_index=tmp_path / "regular_evidence_index.json",
        measured_dir=tmp_path,
        benchmark_root=tmp_path,
        official_results=tmp_path,
    )

    assert returned is validated
    assert sat_results == {}
    assert loaded == {
        "cora_2024": {
            ("onnx/a.onnx", "vnnlib/a.vnnlib", 0): "holds"
        }
    }


def test_frozen_projection_rejects_strict_sat_that_flips_published_truth() -> None:
    instance = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    ny = {"acasxu_2023": {instance: "violated"}}
    reference = {"acasxu_2023": [instance]}
    truth = {"acasxu_2023": {instance: "holds"}}
    winner_points = dict.fromkeys(scorecard.REGULAR, 100)

    with pytest.raises(
        scorecard.MeasurementBudgetError,
        match="rescore map does not exactly cover strict truth changes",
    ):
        scorecard.published_artifact_projection(
            ny,
            reference,
            truth,
            winner_points,
            ny_sat_status="correct-up-to-tolerance",
            ny_sat_results={"acasxu_2023": {instance: "correct"}},
        )


def test_dynamic_projection_recomputes_penalties_and_denominator() -> None:
    instance = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    ny = {"acasxu_2023": {instance: "violated"}}
    reference = {"acasxu_2023": [instance]}
    truth = {"acasxu_2023": {instance: "holds"}}
    winner_points = dict.fromkeys(scorecard.REGULAR, 100)
    rescore = {
        "category": "acasxu_2023",
        "occurrence": list(instance),
        "truth": {
            "published": "holds",
            "rescored": "violated",
            "cause": "ny_strictly_correct_exact_2025_counterexample",
        },
        "participants": {
            "leader": {
                "published_category": {"points": 100},
                "score_delta": -160,
            },
            "runner_up": {
                "published_category": {"points": 90},
                "score_delta": 0,
            },
        },
        "denominator": {
            "published_official_points": 100,
            "rescored_official_points": 90,
            "candidate_instance_points": 10,
        },
    }

    total, breakdown = scorecard.published_artifact_projection(
        ny,
        reference,
        truth,
        winner_points,
        ny_sat_status="correct-up-to-tolerance",
        ny_sat_results={"acasxu_2023": {instance: "correct"}},
        organizer_rescores={"acasxu_2023": {instance: rescore}},
    )

    assert breakdown["acasxu_2023"][:4] == (
        10,
        pytest.approx(100 * 10 / 90),
        90,
        1,
    )
    assert total == pytest.approx(100 * 10 / 90)


def test_dynamic_projection_uses_full_ny_category_raw_as_field_denominator() -> None:
    changed = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    instances = [changed, *[(f"onnx/{index}.onnx", "vnnlib/p.vnnlib", 0) for index in range(11)]]
    ny = {"acasxu_2023": dict.fromkeys(instances, "violated")}
    reference = {"acasxu_2023": instances}
    truth = {
        "acasxu_2023": {
            changed: "holds",
            **dict.fromkeys(instances[1:], "violated"),
        }
    }
    winner_points = dict.fromkeys(scorecard.REGULAR, 100)
    sat_results = {"acasxu_2023": dict.fromkeys(instances, "correct")}
    rescore = {
        "category": "acasxu_2023",
        "occurrence": list(changed),
        "truth": {
            "published": "holds",
            "rescored": "violated",
            "cause": "ny_strictly_correct_exact_2025_counterexample",
        },
        "participants": {
            "incumbent": {
                "published_category": {"points": 100},
                "score_delta": 0,
            }
        },
        "denominator": {
            "published_official_points": 100,
            "rescored_official_points": 100,
            "candidate_instance_points": 10,
        },
    }

    total, breakdown = scorecard.published_artifact_projection(
        ny,
        reference,
        truth,
        winner_points,
        ny_sat_status="correct-up-to-tolerance",
        ny_sat_results=sat_results,
        organizer_rescores={"acasxu_2023": {changed: rescore}},
    )

    # NY has 120 raw points, exceeding the rescored official maximum of 100,
    # so the actual field denominator is 120 and NY self-normalizes to 100.
    assert breakdown["acasxu_2023"][:3] == (120, 100.0, 100)
    assert total == 100.0


def test_dynamic_projection_aggregates_two_same_category_rescores() -> None:
    first = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    second = ("onnx/b.onnx", "vnnlib/b.vnnlib", 0)
    instances = [first, second]
    participants = {
        "leader": {
            "published_category": {"points": 300},
            "score_delta": -160,
        },
        "runner_up": {
            "published_category": {"points": 100},
            "score_delta": 0,
        },
    }

    def rescore(instance: tuple[str, str, int]) -> dict[str, object]:
        return {
            "category": "acasxu_2023",
            "occurrence": list(instance),
            "truth": {
                "published": "holds",
                "rescored": "violated",
                "cause": "ny_strictly_correct_exact_2025_counterexample",
            },
            "participants": participants,
            "denominator": {
                "published_official_points": 300,
                "rescored_official_points": 140,
                "candidate_instance_points": 10,
            },
        }

    organizer_rescores = {
        "acasxu_2023": {
            first: rescore(first),
            second: rescore(second),
        }
    }
    total, breakdown = scorecard.published_artifact_projection(
        {"acasxu_2023": dict.fromkeys(instances, "violated")},
        {"acasxu_2023": instances},
        {"acasxu_2023": dict.fromkeys(instances, "holds")},
        dict.fromkeys(scorecard.REGULAR, 300),
        ny_sat_status="correct-up-to-tolerance",
        ny_sat_results={"acasxu_2023": dict.fromkeys(instances, "correct")},
        organizer_rescores=organizer_rescores,
    )
    before, after = scorecard.rescore_official_overall(
        {"acasxu_2023": {"leader": 300, "runner_up": 100}},
        organizer_rescores,
        entrant_points={"acasxu_2023": 20},
    )

    assert breakdown["acasxu_2023"][:3] == (20, 20.0, 100)
    assert total == 20.0
    assert before == {"leader": 100.0, "runner_up": pytest.approx(100 / 3)}
    assert after == {"leader": 0.0, "runner_up": 100.0}


def test_dynamic_official_leader_reference_is_recomputed_after_penalty() -> None:
    instance = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    published = {
        "acasxu_2023": {"leader": 100, "runner_up": 90},
        "cgan_2023": {"leader": 100, "runner_up": 90},
    }
    rescore = {
        "participants": {
            "leader": {
                "published_category": {"points": 100},
                "score_delta": -160,
            },
            "runner_up": {
                "published_category": {"points": 90},
                "score_delta": 0,
            },
        }
    }

    before, after = scorecard.rescore_official_overall(
        published,
        {"acasxu_2023": {instance: rescore}},
    )

    assert before == {"leader": 200.0, "runner_up": 180.0}
    assert after == {"leader": 100.0, "runner_up": 190.0}
    assert max(before, key=before.get) == "leader"
    assert max(after, key=after.get) == "runner_up"


def test_zero_delta_dynamic_rescore_leaves_official_reference_exactly_unchanged() -> None:
    instance = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    published = {
        "acasxu_2023": {
            "alpha_beta_crown": 1860,
            "cora": 1830,
            "neuralsat": 1860,
            "nnenum": 1860,
            "nnv": 1100,
            "pyrat": 1850,
            "sobolbox": 1460,
        }
    }
    rescore = {
        "participants": {
            tool: {
                "published_category": {"points": points},
                "score_delta": 0,
            }
            for tool, points in published["acasxu_2023"].items()
        }
    }

    before, after = scorecard.rescore_official_overall(
        published,
        {"acasxu_2023": {instance: rescore}},
    )

    assert before == after


def test_dynamic_official_reference_uses_ny_field_denominator() -> None:
    instance = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    published = {
        "acasxu_2023": {"leader": 100, "runner_up": 90},
        "cgan_2023": {"leader": 100, "runner_up": 90},
    }
    rescore = {
        "participants": {
            "leader": {
                "published_category": {"points": 100},
                "score_delta": 0,
            },
            "runner_up": {
                "published_category": {"points": 90},
                "score_delta": 0,
            },
        }
    }

    before, after = scorecard.rescore_official_overall(
        published,
        {"acasxu_2023": {instance: rescore}},
        entrant_points={"acasxu_2023": 120, "cgan_2023": 0},
    )

    assert before == {"leader": 200.0, "runner_up": 180.0}
    assert after == {
        "leader": pytest.approx(100 * 100 / 120 + 100),
        "runner_up": 165.0,
    }


def test_dynamic_official_reference_rejects_incomplete_entrant_points() -> None:
    with pytest.raises(
        scorecard.MeasurementBudgetError,
        match="entrant raw points do not bind every published category",
    ):
        scorecard.rescore_official_overall(
            {"acasxu_2023": {"leader": 100}},
            {},
            entrant_points={},
        )


def test_frozen_projection_accepts_strict_sat_when_truth_is_already_violated() -> None:
    instance = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    ny = {"acasxu_2023": {instance: "violated"}}
    reference = {"acasxu_2023": [instance]}
    truth = {"acasxu_2023": {instance: "violated"}}
    winner_points = dict.fromkeys(scorecard.REGULAR, 100)

    total, breakdown = scorecard.published_artifact_projection(
        ny,
        reference,
        truth,
        winner_points,
        ny_sat_status="correct-up-to-tolerance",
        ny_sat_results={"acasxu_2023": {instance: "correct"}},
    )

    assert breakdown["acasxu_2023"][:4] == (10, 10.0, 100, 1)
    assert total == 10.0


def test_evidence_report_prints_nonofficial_claim_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import regular_bank_evidence as bank_evidence  # noqa: PLC0415

    context = SimpleNamespace(
        reference_order={category: [] for category in scorecard.REGULAR},
        ground_truth={category: {} for category in scorecard.REGULAR},
        winner_points=dict.fromkeys(scorecard.REGULAR, 1),
    )
    pinned = SimpleNamespace(context=context)
    monkeypatch.setattr(
        bank_evidence,
        "revalidate_official_results",
        lambda _: None,
    )

    status = scorecard.report_published_artifact_projection(
        tmp_path,
        {},
        ny_sat_status="correct-up-to-tolerance",
        target=1566.9,
        # This fixture asserts that no row was field-falsified; the moat input is
        # required precisely so a fixture cannot skip it by omission.
        field_falsified={},
        evidence_qualified=True,
        pinned_official=pinned,
        ny_sat_results={},
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "local reproducible/internal counterfactual" in output
    assert "not official or independently attested" in output


def test_evidence_report_compares_ny_to_dynamic_official_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import regular_bank_evidence as bank_evidence  # noqa: PLC0415

    instance = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    reference = {category: [] for category in scorecard.REGULAR}
    reference["acasxu_2023"] = [instance]
    truth = {category: {} for category in scorecard.REGULAR}
    truth["acasxu_2023"] = {instance: "holds"}
    pinned = SimpleNamespace(
        context=SimpleNamespace(
            reference_order=reference,
            ground_truth=truth,
            winner_points=dict.fromkeys(scorecard.REGULAR, 100),
        )
    )
    rescore = {
        "category": "acasxu_2023",
        "occurrence": list(instance),
        "truth": {
            "published": "holds",
            "rescored": "violated",
            "cause": "ny_strictly_correct_exact_2025_counterexample",
        },
        "participants": {
            "leader": {
                "published_category": {"points": 100},
                "score_delta": -160,
            },
            "runner_up": {
                "published_category": {"points": 90},
                "score_delta": 0,
            },
        },
        "denominator": {
            "published_official_points": 100,
            "rescored_official_points": 90,
            "candidate_instance_points": 10,
        },
    }
    monkeypatch.setattr(bank_evidence, "revalidate_official_results", lambda _: None)
    monkeypatch.setattr(
        scorecard,
        "load_published_tool_points",
        lambda _: {
            "acasxu_2023": {"leader": 100, "runner_up": 90},
            "cgan_2023": {"leader": 100, "runner_up": 90},
        },
    )

    status = scorecard.report_published_artifact_projection(
        tmp_path,
        {"acasxu_2023": {instance: "violated"}},
        ny_sat_status="correct-up-to-tolerance",
        target=1566.9,
        field_falsified={},
        evidence_qualified=True,
        pinned_official=pinned,
        ny_sat_results={"acasxu_2023": {instance: "correct"}},
        organizer_rescores={"acasxu_2023": {instance: rescore}},
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "dynamically rescored official leader: runner_up 190.0" in output
    assert "Official leader transition: leader 200.0 -> runner_up 190.0" in output
    assert "published alpha-beta-CROWN reference: 1566.9" not in output


def test_evidence_report_renormalizes_officials_when_ny_sets_field_max(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import regular_bank_evidence as bank_evidence  # noqa: PLC0415

    changed = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    instances = [
        changed,
        *[
            (f"onnx/{index}.onnx", "vnnlib/p.vnnlib", 0)
            for index in range(11)
        ],
    ]
    reference = {category: [] for category in scorecard.REGULAR}
    reference["acasxu_2023"] = instances
    truth = {category: {} for category in scorecard.REGULAR}
    truth["acasxu_2023"] = {
        changed: "holds",
        **dict.fromkeys(instances[1:], "violated"),
    }
    pinned = SimpleNamespace(
        context=SimpleNamespace(
            reference_order=reference,
            ground_truth=truth,
            winner_points=dict.fromkeys(scorecard.REGULAR, 100),
        )
    )
    rescore = {
        "category": "acasxu_2023",
        "occurrence": list(changed),
        "truth": {
            "published": "holds",
            "rescored": "violated",
            "cause": "ny_strictly_correct_exact_2025_counterexample",
        },
        "participants": {
            "incumbent": {
                "published_category": {"points": 100},
                "score_delta": 0,
            }
        },
        "denominator": {
            "published_official_points": 100,
            "rescored_official_points": 100,
            "candidate_instance_points": 10,
        },
    }
    monkeypatch.setattr(bank_evidence, "revalidate_official_results", lambda _: None)
    monkeypatch.setattr(
        scorecard,
        "load_published_tool_points",
        lambda _: {"acasxu_2023": {"incumbent": 100}},
    )

    status = scorecard.report_published_artifact_projection(
        tmp_path,
        {"acasxu_2023": dict.fromkeys(instances, "violated")},
        ny_sat_status="correct-up-to-tolerance",
        target=1566.9,
        field_falsified={},
        evidence_qualified=True,
        pinned_official=pinned,
        ny_sat_results={"acasxu_2023": dict.fromkeys(instances, "correct")},
        organizer_rescores={"acasxu_2023": {changed: rescore}},
    )

    assert status == 0
    output = capsys.readouterr().out
    assert (
        "NY projection: 100.0   dynamically rescored official leader: "
        "incumbent 83.3"
    ) in output
    assert "Behind reference by: -16.7" in output


def test_regular_evidence_loader_rejects_dangling_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import regular_bank_evidence as bank_evidence  # noqa: PLC0415

    dangling = SimpleNamespace(row_key='["cora_2024","a","b",0]')
    monkeypatch.setattr(
        bank_evidence,
        "validate_regular_evidence_index",
        lambda **_: SimpleNamespace(
            creditable_entries=(),
            dangling_entries=(dangling,),
        ),
    )

    with pytest.raises(
        scorecard.MeasurementBudgetError,
        match="non-creditable dangling transaction",
    ):
        scorecard.load_evidenced_ny_regular(
            evidence_index=tmp_path / "regular_evidence_index.json",
            measured_dir=tmp_path,
            benchmark_root=tmp_path,
            official_results=tmp_path,
        )


@pytest.mark.parametrize(
    "detail",
    [
        "tampered completion artifact",
        "measured bank row does not match indexed after-row",
    ],
)
def test_regular_evidence_loader_surfaces_tamper_and_index_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detail: str,
) -> None:
    import regular_bank_evidence as bank_evidence  # noqa: PLC0415

    def fail(**_: object) -> object:
        raise bank_evidence.EvidenceError(detail)

    monkeypatch.setattr(
        bank_evidence,
        "validate_regular_evidence_index",
        fail,
    )

    with pytest.raises(scorecard.MeasurementBudgetError, match=detail):
        scorecard.load_evidenced_ny_regular(
            evidence_index=tmp_path / "regular_evidence_index.json",
            measured_dir=tmp_path,
            benchmark_root=tmp_path,
            official_results=tmp_path,
        )


def test_main_exits_nonzero_when_extended_bad_rows_exist(tmp_path: Path) -> None:
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="lsnc_relu",
        gt_rows=[("net-prop_0", "sat", True)],
        win_score=10,
        ny_csv_rows=["lsnc_relu,net.onnx,prop_0.vnnlib,unsat,1.0"],
    )
    official = tmp_path / "official"
    official.mkdir()
    measured = tmp_path / "measured"
    measured.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--official",
            str(official),
            "--measured",
            str(measured),
            "--extended",
            "--results-txt",
            str(results_txt),
            "--ext-measured",
            str(ext_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "*** SCORECARD FAILURE: 1 blocking extended verdict(s)" in result.stdout


def test_extended_moat_blocks_ny_unsat_on_a_tolerance_falsified_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The asymmetric-moat regression.

    ``true_result`` is not ground truth: official ``process_results.py`` only
    writes ``sat`` when some counterexample graded exactly CORRECT, so a row
    every tool falsified with a CORRECT_UP_TO_TOLERANCE counterexample — still
    accepted, still +10 to that tool — is recorded as ``unsat`` with the ``(v)``
    marker still on the row.  101 extended-track rows look like that.  Before
    the fix the ny-unsat branch tested only ``tr == 'sat'``, so ny scored +10 on
    such a row with the board still printing "MOAT: zero contradictions".
    """
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="cctsdb_yolo_2023",
        # row 0: field falsified it, but only with a tolerance-graded CE.
        # row 1: genuinely unsat, nobody falsified it — must stay clean.
        gt_rows=[("net-prop_0", "unsat", True), ("net-prop_1", "unsat", False)],
        win_score=20,
        ny_csv_rows=[
            "cctsdb_yolo_2023,net.onnx,prop_0.vnnlib,unsat,1.0",
            "cctsdb_yolo_2023,net.onnx,prop_1.vnnlib,unsat,1.0",
        ],
    )

    grand, blocking = scorecard.score_extended(results_txt, ext_dir, 0.0)
    out = capsys.readouterr().out

    # The official scorer pays +10 for both rows, so the printed score is
    # unchanged — what the fix adds is the block.
    row = next(line for line in out.splitlines() if line.startswith("cctsdb"))
    assert row.split()[-5:] == ["20", "100.0", "20", "100.0", "100.0"]
    assert grand == 100.0
    assert blocking == 1
    assert "MOAT: zero contradictions" not in out
    assert "ny-unsat vs field-falsified row" in out
    assert "net-prop_0" in out
    assert "net-prop_1" not in out
    assert "*** SCORECARD FAILURE: 1 blocking extended verdict(s)" in out


def test_extended_moat_leaves_ny_sat_on_a_tolerance_falsified_row_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The symmetric partner: agreeing with the field's CE is not a flag."""
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="cctsdb_yolo_2023",
        gt_rows=[("net-prop_0", "unsat", True)],
        win_score=10,
        ny_csv_rows=["cctsdb_yolo_2023,net.onnx,prop_0.vnnlib,sat,1.0"],
    )

    grand, blocking = scorecard.score_extended(results_txt, ext_dir, 0.0)
    out = capsys.readouterr().out

    assert (grand, blocking) == (100.0, 0)
    assert "MOAT: zero contradictions" in out
    assert "lone ny-sat vs all-holds field" not in out


# --- unconfirmable rows: no field ground truth, so nothing can contradict ny --


def test_extended_scorecard_gives_no_credit_for_unsat_without_field_truth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Task #86: an ny verdict on a row NOBODY solved used to score +10.

    ``True Result: -`` means no tool decided the instance: there is no ``(v)``,
    no ``(h)``, no counterexample, nothing that could ever contradict ny.  The
    moat is defined entirely in terms of that missing evidence, so it can never
    flag such a row — yet the old ny-unsat branch fell through to ``ok += 1;
    raw += POINTS_CORRECT``.  The credit is now reported as separate upside.
    """
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="vit_2023",
        # row 0: the field solved it and holds -> real, auditable credit.
        # row 1: nobody solved it -> unfalsifiable by construction.
        gt_rows=[("net-prop_0", "unsat", False), ("net-prop_1", "-", False)],
        win_score=20,
        ny_csv_rows=[
            "vit_2023,net.onnx,prop_0.vnnlib,unsat,1.0",
            "vit_2023,net.onnx,prop_1.vnnlib,unsat,1.0",
        ],
    )

    grand, blocking = scorecard.score_extended(results_txt, ext_dir, 0.0)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("vit_2023"))

    assert blocking == 0
    # Defensible: 1 of 2 sampled rows credited -> 10 raw, extrapolated to 10
    # over N=2, normalized 50.0 against the 20-point winner.
    # Upside: both rows paid -> 20 raw -> 100.0. Before the fix the defensible
    # column WAS the upside column.
    assert row.split()[-5:] == ["10", "50.0", "10", "50.0", "100.0"]
    assert grand == 50.0
    assert "UNCONFIRMABLE: 1 extended row(s) scored 0" in out
    assert "unconfirmable: vit_2023 net-prop_1 (ny unsat, field True Result '-')" in out
    assert "EXTENDED-9 UPSIDE normalized" in out


def test_extended_scorecard_gives_no_credit_for_sat_without_field_truth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``sat`` gets no exemption on the extended track.

    A counterexample is self-certifying evidence in principle, but no validated
    extended evidence index exists (``score_extended`` refuses
    ``require_evidence`` outright), so an extended witness is assumed valid and
    never checked.  Assumed-valid is not confirmed.
    """
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="relusplitter",
        gt_rows=[("net-prop_0", "-", False)],
        win_score=10,
        ny_csv_rows=["relusplitter,net.onnx,prop_0.vnnlib,sat,1.0"],
    )

    grand, blocking = scorecard.score_extended(results_txt, ext_dir, 0.0)
    out = capsys.readouterr().out

    assert (grand, blocking) == (0.0, 0)
    assert "UNCONFIRMABLE: 1 extended row(s) scored 0" in out
    assert (
        "unconfirmable: relusplitter net-prop_0 (ny sat, field True Result '-')" in out
    )
    # It is not double-counted as an unevidenced sat credit: it was not credited.
    assert "WARNING: 1 extended sat row(s) credited WITHOUT" not in out


def test_published_projection_gives_no_credit_for_unsat_without_field_truth(
    tmp_path: Path,
) -> None:
    """The regular twin of the same hole.

    ``longtable.tex`` prints no cell for an instance nobody solved, so the row
    arrives as ``truth == "unknown"`` — and the ``holds`` branch's ``else``
    paid it +10 exactly like a genuine hold.
    """
    reference_csv = tmp_path / "results.csv"
    reference_csv.write_text(
        "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,1\n"
        "acasxu_2023,onnx/b.onnx,vnnlib/b.vnnlib,0,timeout,1\n",
        encoding="utf-8",
    )
    longtable = tmp_path / "longtable.tex"
    longtable.write_text(
        "2025 Acasxu 2023 & 0 & ~\\textsc{unsat} & - \\\\\n", encoding="utf-8"
    )
    reference = scorecard.load_reference_instance_order(reference_csv)
    ground_truth = scorecard.load_published_ground_truth(longtable, reference)
    decided, undecided = reference["acasxu_2023"]
    assert undecided not in ground_truth["acasxu_2023"]

    total, breakdown = scorecard.published_artifact_projection(
        {"acasxu_2023": {decided: "holds", undecided: "holds"}},
        reference,
        ground_truth,
        dict.fromkeys(scorecard.REGULAR, 100),
        ny_sat_status="correct-up-to-tolerance",
    )
    projection = breakdown["acasxu_2023"]

    assert (projection.raw, projection.credited) == (10, 1)
    assert projection.unconfirmable == 1
    assert projection.normalized == 10.0
    assert projection.upside_normalized == 20.0
    assert total == 10.0


def test_published_projection_credits_a_strictly_correct_sat_without_field_truth(
    tmp_path: Path,
) -> None:
    """The one exemption, and its boundary.

    A counterexample the ORGANIZER CHECKER graded strictly ``correct`` confirms
    itself: it needs no field ground truth, so it stays creditable on a row
    nobody solved.  A merely ``correct_up_to_tolerance`` grade — the assumed
    status the legacy CSVs carry — does not.
    """
    reference_csv = tmp_path / "results.csv"
    reference_csv.write_text(
        "acasxu_2023,onnx/b.onnx,vnnlib/b.vnnlib,0,timeout,1\n", encoding="utf-8"
    )
    longtable = tmp_path / "longtable.tex"
    longtable.write_text("", encoding="utf-8")
    reference = scorecard.load_reference_instance_order(reference_csv)
    ground_truth = scorecard.load_published_ground_truth(longtable, reference)
    (undecided,) = reference["acasxu_2023"]

    def project(sat_grade: str) -> scorecard.CategoryProjection:
        _total, breakdown = scorecard.published_artifact_projection(
            {"acasxu_2023": {undecided: "violated"}},
            reference,
            ground_truth,
            dict.fromkeys(scorecard.REGULAR, 100),
            ny_sat_status="correct-up-to-tolerance",
            ny_sat_results={"acasxu_2023": {undecided: sat_grade}},
            organizer_rescores={},
        )
        return breakdown["acasxu_2023"]

    strict = project("correct")
    assert (strict.raw, strict.credited, strict.unconfirmable) == (10, 1, 0)

    tolerance = project("correct_up_to_tolerance")
    assert (tolerance.raw, tolerance.credited, tolerance.unconfirmable) == (0, 0, 1)


def test_raw_verdict_board_reports_unconfirmable_rows_it_silently_dropped(
    tmp_path: Path,
) -> None:
    """The default path already scored these 0 — but said nothing about them.

    That silence is how the same rows came to be paid +10 on the other two
    scoring paths without anyone noticing, so the count is now printed.
    """
    benchmark_root = _write_instances(
        tmp_path,
        "cora_2024",
        [
            "onnx/a.onnx,vnnlib/a.vnnlib,300",
            "onnx/b.onnx,vnnlib/b.vnnlib,300",
        ],
    )
    official_dir = tmp_path / "official"
    for tool in scorecard.OFFICIAL_TOOLS:
        (official_dir / tool).mkdir(parents=True)
        # Row a: the field holds it. Row b: every tool timed out, so the field
        # established nothing at all.
        (official_dir / tool / "results.csv").write_text(
            "cora_2024,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,1\n"
            "cora_2024,onnx/b.onnx,vnnlib/b.vnnlib,0,timeout,1\n",
            encoding="utf-8",
        )
    measured_dir = tmp_path / "measured"
    measured_dir.mkdir()
    (measured_dir / "cora_2024.csv").write_text(
        "cora_2024,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,10,run-a\n"
        "cora_2024,onnx/b.onnx,vnnlib/b.vnnlib,0,unsat,10,run-a\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--official",
            str(official_dir),
            "--measured",
            str(measured_dir),
            "--benchmark-root",
            str(benchmark_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "UNCONFIRMABLE: 1 regular row(s) scored 0" in result.stdout
    cora = next(
        line for line in result.stdout.splitlines() if line.startswith("cora_2024")
    )
    # ny_raw, ny_norm, win_raw, ny_ok, ny_bad, unconf
    assert cora.split()[1:7] == ["10", "100.0", "10", "1", "0", "1"]


def test_blind_spot_inventory_names_structurally_blind_categories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The moat is relative to the field; say where the field left nothing.

    A category in which no tool ever reported a violation cannot contradict an
    ny ``unsat`` on ANY of its rows, so "zero contradictions" is silent about
    it.  vit_2023 is exactly that shape in the real artifacts.
    """
    results_txt = tmp_path / "results.txt"
    results_txt.write_text(
        "\n".join(
            [
                "Category 2025_vit_2023:",
                "  participating tools: ['alpha_beta_crown']",
                "Row: ['net-prop_0', '1.0 (h)']",
                "True Result: unsat",
                "0: alpha_beta_crown score: 10, is_ver: True, is_fals: False,",
                "Row: ['net-prop_1', 'timeout']",
                "True Result: -",
                "1: alpha_beta_crown score: 0, is_ver: False, is_fals: False,",
                "Category 2025_soundnessbench:",
                "  participating tools: ['alpha_beta_crown']",
                "Row: ['net-prop_0', '1.0 (v)']",
                "True Result: sat",
                "0: alpha_beta_crown score: 10, is_ver: False, is_fals: True,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inventory = scorecard.field_evidence_inventory(
        results_txt, ["soundnessbench", "vit_2023"]
    )

    assert inventory["vit_2023"].rows == 2
    assert inventory["vit_2023"].violated_marker == 0
    assert inventory["vit_2023"].holds_only == 1
    assert inventory["vit_2023"].no_field_truth == 1
    assert inventory["vit_2023"].unsolved_by_anybody == 1
    assert inventory["vit_2023"].blind_rows == 2
    assert not inventory["vit_2023"].moat_can_see_a_wrong_unsat

    assert inventory["soundnessbench"].accepted_ce == 1
    assert inventory["soundnessbench"].blind_rows == 0
    assert inventory["soundnessbench"].moat_can_see_a_wrong_unsat

    scorecard.report_field_evidence_inventory(inventory)
    out = capsys.readouterr().out
    assert "STRUCTURALLY BLIND CATEGOR(IES)" in out
    assert "vit_2023" in out
    assert "soundnessbench" not in out.split("STRUCTURALLY BLIND")[1]
