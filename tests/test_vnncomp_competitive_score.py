# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the VNN-COMP 2025 regular-track scoring model."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "vnncomp_competitive_score.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "vnncomp_competitive_score",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass forward-references (e.g. the
    # ``-> InstanceResult`` self-type under ``from __future__ import
    # annotations``) resolve via ``sys.modules`` on Python 3.9.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()
InstanceResult = mod.InstanceResult
CounterexampleResult = mod.CounterexampleResult


# ---------------------------------------------------------------------------
# Per-instance scoring.
# ---------------------------------------------------------------------------


def _holds(tool, instance="i0"):
    return InstanceResult(tool=tool, benchmark="b", instance=instance, result="holds")


def _violated(tool, instance="i0", ce=CounterexampleResult.CORRECT, ce_required=True):
    return InstanceResult(
        tool=tool,
        benchmark="b",
        instance=instance,
        result="violated",
        counterexample=ce,
        ce_required=ce_required,
    )


def test_correct_holds_scores_ten():
    h = _holds("t1")
    assert mod.score_instance(h, [h]) == mod.POINTS_CORRECT == 10


def test_correct_violated_with_valid_ce_scores_ten():
    v = _violated("t1")
    assert mod.score_instance(v, [v]) == 10


def test_within_tolerance_ce_still_scores_ten_for_sat_tool():
    v = _violated("t1", ce=CounterexampleResult.CORRECT_UP_TO_TOLERANCE)
    assert mod.score_instance(v, [v]) == 10


def test_timeout_unknown_error_score_zero():
    for status in ("timeout", "unknown", "error"):
        r = InstanceResult(tool="t1", benchmark="b", instance="i0", result=status)
        assert mod.score_instance(r, [r]) == 0


def test_violated_without_required_ce_is_penalized():
    v = _violated("t1", ce=None, ce_required=True)
    assert mod.score_instance(v, [v]) == mod.PENALTY_INCORRECT == -150


def test_violated_with_invalid_ce_is_penalized():
    v = _violated("t1", ce=CounterexampleResult.SPEC_NOT_VIOLATED)
    assert mod.score_instance(v, [v]) == -150


def test_holds_falsified_by_other_tools_correct_ce_is_penalized():
    h = _holds("t1")
    v = _violated("t2", ce=CounterexampleResult.CORRECT)
    instance = [h, v]
    assert mod.score_instance(h, instance) == -150  # disagreeing holds is wrong
    assert mod.score_instance(v, instance) == 10  # the SAT tool is right


def test_holds_not_falsified_by_within_tolerance_ce():
    h = _holds("t1")
    v = _violated("t2", ce=CounterexampleResult.CORRECT_UP_TO_TOLERANCE)
    instance = [h, v]
    # A within-tolerance CE neither falsifies the UNSAT tool nor penalizes SAT.
    assert mod.score_instance(h, instance) == 10
    assert mod.score_instance(v, instance) == 10


# ---------------------------------------------------------------------------
# Per-benchmark raw scoring + normalization.
# ---------------------------------------------------------------------------


def test_score_benchmark_sums_per_instance_points():
    results = [
        _holds("t1", "i0"),
        _holds("t1", "i1"),
        InstanceResult(tool="t1", benchmark="b", instance="i2", result="timeout"),
        _holds("t2", "i0"),
    ]
    raw = mod.score_benchmark(results)
    assert raw == {"t1": 20, "t2": 10}


def test_normalize_winner_gets_exactly_100():
    norm = mod.normalize_benchmark({"t1": 30, "t2": 10, "t3": 0})
    assert norm["t1"] == 100.0
    assert norm["t2"] == pytest.approx(100.0 * 10 / 30)
    assert norm["t3"] == 0.0


def test_single_minus_150_zeros_a_benchmarks_normalized_score():
    # t1: two correct holds (+20) then one wrong holds (-150) -> net -130.
    # The -130 is floored to 0 after normalization; it cannot go negative.
    h0 = _holds("t1", "i0")
    h1 = _holds("t1", "i1")
    h_wrong = _holds("t1", "i2")
    v = _violated("t2", "i2", ce=CounterexampleResult.CORRECT)  # falsifies t1's i2
    # t2 wins i2 (+10), participates only there.
    results = [h0, h1, h_wrong, v]
    raw = mod.score_benchmark(results)
    assert raw["t1"] == 20 - 150  # -130
    assert raw["t2"] == 10
    norm = mod.normalize_benchmark(raw)
    # Winner is t2 with +10; t1 net-negative -> floored to 0.0.
    assert norm["t2"] == 100.0
    assert norm["t1"] == 0.0


def test_negative_or_zero_max_benchmark_contributes_zero_to_everyone():
    # Every tool net-negative: whole benchmark contributes 0.0 to all.
    raw = {"t1": -150, "t2": -300}
    norm = mod.normalize_benchmark(raw)
    assert norm == {"t1": 0.0, "t2": 0.0}
    # Zero-max (all tools 0): also 0.0 each.
    assert mod.normalize_benchmark({"t1": 0, "t2": 0}) == {"t1": 0.0, "t2": 0.0}


# ---------------------------------------------------------------------------
# Overall summation.
# ---------------------------------------------------------------------------


def test_overall_scoreboard_sums_normalized_per_benchmark():
    # Benchmark A: t1 wins (100), t2 half.
    bench_a = [
        _holds("t1", "a0"),
        _holds("t1", "a1"),
        _holds("t2", "a0"),
    ]
    # Benchmark B: t2 wins (100), t1 absent (0).
    bench_b = [
        _holds("t2", "b0"),
    ]
    all_benchmarks = {"A": bench_a, "B": bench_b}
    report = mod.build_report(all_benchmarks, mod.Tolerance.ZERO_TOL)
    # t1: 100 (A) + 0 (B) = 100; t2: 50 (A) + 100 (B) = 150.
    assert report.totals["t1"] == pytest.approx(100.0)
    assert report.totals["t2"] == pytest.approx(150.0)
    ranked = mod._ranked(report.totals)
    assert ranked[0][0] == "t2"


def test_overall_scoreboard_helper_matches_build_report():
    bench_a = [_holds("t1", "a0"), _holds("t2", "a0")]
    bench_b = [_holds("t1", "b0")]
    all_benchmarks = {"A": bench_a, "B": bench_b}
    totals = mod.overall_scoreboard(all_benchmarks, tolerance=mod.Tolerance.ZERO_TOL)
    # A: tie -> both 100. B: t1 only -> 100. t1 = 200, t2 = 100.
    assert totals["t1"] == pytest.approx(200.0)
    assert totals["t2"] == pytest.approx(100.0)


def test_skip_tools_excluded():
    results = [_holds("rover", "i0"), _holds("t1", "i0")]
    raw = mod.score_benchmark(results)
    assert "rover" not in raw
    assert raw["t1"] == 10


# ---------------------------------------------------------------------------
# Result normalization (CSV substitutions).
# ---------------------------------------------------------------------------


def test_normalize_result_substitutions():
    assert mod.normalize_result("unsat") == "holds"
    assert mod.normalize_result("sat") == "violated"
    assert mod.normalize_result("run_instance_timeout") == "timeout"
    assert mod.normalize_result("prepare_instance_timeout") == "timeout"
    assert mod.normalize_result("prepare_instance_error_5") == "unknown"
    assert mod.normalize_result("error_exit_code_1") == "error"
    assert mod.normalize_result("error_nonmaximal") == "unknown"
    assert mod.normalize_result("no_result_in_file") == "unknown"
    assert mod.normalize_result("") == "unknown"
    assert mod.normalize_result("HOLDS") == "holds"


# ---------------------------------------------------------------------------
# JSON ingest (ny's vnncomp_latest.json shape).
# ---------------------------------------------------------------------------


def test_load_ny_json_expands_aggregate_counts(tmp_path):
    payload = {
        "categories": {
            "acasxu_2023": {
                "verified": 3,
                "falsified": 2,
                "timeout": 1,
                "unknown": 0,
                "error": 0,
            },
        },
    }
    path = tmp_path / "vnncomp_latest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    benchmarks = mod.load_ny_json(str(path))
    assert set(benchmarks) == {"acasxu_2023"}
    rows = benchmarks["acasxu_2023"]
    # 3 holds + 2 violated + 1 timeout = 6 rows.
    assert len(rows) == 6
    raw = mod.score_benchmark(rows)
    # 3 correct holds + 2 correct violated = 5 * 10 = 50; timeout scores 0.
    assert raw["ny"] == 50


# ---------------------------------------------------------------------------
# CSV ingest (official per-tool layout).
# ---------------------------------------------------------------------------


def test_instance_key_is_subdir_aware():
    # safenlp_2024 ships medical/ and ruarobot/ with identical onnx and vnnlib
    # basenames: they must not share a bucket.
    assert mod._instance_key(
        "onnx/medical/perturbations_0.onnx",
        "vnnlib/medical/hyperrectangle_1215.vnnlib",
    ) != mod._instance_key(
        "onnx/ruarobot/perturbations_0.onnx",
        "vnnlib/ruarobot/hyperrectangle_1215.vnnlib",
    )
    # ny's relative paths and the official tools' absolute ones still agree on
    # the same instance, and flat benchmarks collapse to "onnx/x".
    assert mod._instance_key("onnx/a.onnx", "./vnnlib/p.vnnlib") == mod._instance_key(
        "/root/benchmarks/acasxu_2023/onnx/a.onnx",
        "/root/benchmarks/acasxu_2023/vnnlib/p.vnnlib",
    )


def test_correct_holds_not_falsified_by_ce_on_same_basename_in_other_subdir(tmp_path):
    # toolA's valid CE on ruarobot/1215 must not falsify toolB's correct holds
    # on the DIFFERENT medical/1215 instance: both answers are right (+10 each).
    (tmp_path / "toolA.csv").write_text(
        "safenlp_2024,onnx/ruarobot/perturbations_0.onnx,"
        "vnnlib/ruarobot/hyperrectangle_1215.vnnlib,0.01,sat,1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "toolB.csv").write_text(
        "safenlp_2024,onnx/medical/perturbations_0.onnx,"
        "vnnlib/medical/hyperrectangle_1215.vnnlib,0.01,unsat,1.0\n",
        encoding="utf-8",
    )
    rows = mod.load_results_dir(str(tmp_path))["safenlp_2024"]
    assert len({r.instance for r in rows}) == 2
    assert mod.score_benchmark(rows) == {"toolA": 10, "toolB": 10}


def test_load_results_dir_headerless_csv(tmp_path):
    # Headerless harness results.csv: category,onnx,vnnlib,prep,result,runtime.
    (tmp_path / "alpha.csv").write_text(
        "acasxu_2023,m/a.onnx,p/p0.vnnlib,0.01,unsat,1.0\n"
        "acasxu_2023,m/b.onnx,p/p1.vnnlib,0.01,sat,2.0\n",
        encoding="utf-8",
    )
    benchmarks = mod.load_results_dir(str(tmp_path))
    assert set(benchmarks) == {"acasxu_2023"}
    rows = benchmarks["acasxu_2023"]
    assert {r.result for r in rows} == {"holds", "violated"}
    assert all(r.tool == "alpha" for r in rows)
    raw = mod.score_benchmark(rows)
    assert raw["alpha"] == 20  # one holds + one validated violated.


def test_load_results_dir_tolerates_trailing_measurement_run_id(tmp_path):
    (tmp_path / "ny.csv").write_text(
        "acasxu_2023,m/a.onnx,p/p0.vnnlib,0.01,unsat,1.0,20260718T120000Z-123\n",
        encoding="utf-8",
    )

    rows = mod.load_results_dir(str(tmp_path))["acasxu_2023"]

    assert len(rows) == 1
    assert rows[0].result == "holds"
    assert rows[0].runtime == 1.0


def test_load_results_dir_excludes_harness_test_instances(tmp_path):
    (tmp_path / "alpha.csv").write_text(
        "acasxu_2023,m/a.onnx,p/p0.vnnlib,0.01,unsat,1.0\n"
        "acasxu_2023,m/test_nano.onnx,p/test_nano.vnnlib,0.01,unsat,1.0\n"
        "acasxu_2023,m/test_tiny.onnx,p/test_tiny.vnnlib,0.01,unsat,1.0\n",
        encoding="utf-8",
    )

    rows = mod.load_results_dir(str(tmp_path))["acasxu_2023"]

    assert len(rows) == 1
    assert rows[0].instance == "m/a.onnx|p/p0.vnnlib"


def test_load_results_dir_reads_official_nested_tool_layout(tmp_path):
    tool_dir = tmp_path / "alpha_beta_crown"
    tool_dir.mkdir()
    (tool_dir / "results.csv").write_text(
        "acasxu_2023,m/a.onnx,p/p0.vnnlib,0.01,unsat,1.0\n",
        encoding="utf-8",
    )
    scoring_dir = tmp_path / "SCORING-ZERO-TOL"
    scoring_dir.mkdir()
    (scoring_dir / "results.csv").write_text(
        "bogus,m/b.onnx,p/p1.vnnlib,0.01,unsat,1.0\n",
        encoding="utf-8",
    )

    benchmarks = mod.load_results_dir(str(tmp_path))

    assert set(benchmarks) == {"acasxu_2023"}
    assert benchmarks["acasxu_2023"][0].tool == "alpha_beta_crown"


# ---------------------------------------------------------------------------
# Reporting / CLI.
# ---------------------------------------------------------------------------


def test_single_tool_report_withholds_the_target_verdict():
    # One tool wins every benchmark it scores on by construction, so the total
    # is self-referential and must not be rendered as beating a rival's total.
    report = mod.build_report({"A": [_holds("ny", "a0")]}, mod.Tolerance.ZERO_TOL)
    assert report.totals == {"ny": 100.0}
    text = mod.format_report(report, target=mod.DEFAULT_TARGET)
    assert "Reference total" not in text
    assert "AHEAD" not in text
    assert "BEHIND" not in text
    assert "needs" not in text
    assert "Single-tool scoreboard" in text


def test_multi_tool_report_still_compares_against_the_target():
    report = mod.build_report(
        {"A": [_holds("ny", "a0"), _holds("t2", "a0")]},
        mod.Tolerance.ZERO_TOL,
    )
    text = mod.format_report(report, target=50.0)
    assert "Reference total: 50.0" in text
    assert "AHEAD by 50.0" in text
    assert "ny = 100.0" in text
    assert "not an organizer-validated official scoreboard" in text


def test_cli_refuses_to_merge_aggregate_json_into_a_results_dir_field(tmp_path):
    # Aggregate counts carry no instance identities, so ny's rows could never be
    # cross-checked against a rival's counterexample: refuse rather than seat ny
    # in a table it cannot be scored incorrect in.
    (tmp_path / "vnncomp_latest.json").write_text(
        json.dumps({"categories": {"acasxu_2023": {"verified": 2}}}),
        encoding="utf-8",
    )
    json_path = str(tmp_path / "vnncomp_latest.json")
    old_argv = sys.argv
    sys.argv = ["prog", "--json", json_path, "--results-dir", str(tmp_path)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            mod.main()
    finally:
        sys.argv = old_argv
    assert excinfo.value.code != 0


def test_load_inputs_reads_exactly_one_source(tmp_path):
    payload = {"categories": {"acasxu_2023": {"verified": 1, "falsified": 1}}}
    json_path = tmp_path / "vnncomp_latest.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    args = mod._build_parser().parse_args(["--json", str(json_path)])
    benchmarks = mod._load_inputs(args)
    assert set(benchmarks) == {"acasxu_2023"}
    assert {r.tool for r in benchmarks["acasxu_2023"]} == {"ny"}


def test_load_inputs_filters_extended_and_test_categories_from_regular_board(tmp_path):
    (tmp_path / "alpha.csv").write_text(
        "acasxu_2023,m/a.onnx,p/a.vnnlib,0,unsat,1\n"
        "vggnet16_2022,m/b.onnx,p/b.vnnlib,0,unsat,1\n"
        "test,m/test.onnx,p/test.vnnlib,0,unsat,1\n",
        encoding="utf-8",
    )
    args = mod._build_parser().parse_args(["--results-dir", str(tmp_path)])

    benchmarks = mod._load_inputs(args)

    assert set(benchmarks) == {"acasxu_2023"}
