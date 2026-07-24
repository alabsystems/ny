# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Tests for validate_vnncomp_results.sh and audit_vnncomp_counterexamples.py."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.vnnlib_parser import (
    VnnlibParseError,
    evaluate_output_property,
    parse_vnnlib_output_property,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "validate_vnncomp_results.sh"


# ---------------------------------------------------------------------------
# VNN-LIB parser tests
# ---------------------------------------------------------------------------

def test_parse_flat_conjunctive_prop2() -> None:
    """prop_2 style: multiple flat (assert (<= Y_i Y_0))."""
    text = """\
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(declare-const Y_4 Real)
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
(assert (<= Y_3 Y_0))
(assert (<= Y_4 Y_0))
"""
    prop = parse_vnnlib_output_property(text)
    assert not prop.disjunctive, "prop_2 should be conjunctive"
    assert len(prop.branches) == 1, f"expected 1 branch, got {len(prop.branches)}"
    assert len(prop.branches[0].clauses) == 4, (
        f"expected 4 clauses, got {len(prop.branches[0].clauses)}"
    )
    assert prop.output_vars == {"Y_0": 0, "Y_1": 1, "Y_2": 2, "Y_3": 3, "Y_4": 4}, (
        f"unexpected output_vars: {prop.output_vars}"
    )


def test_parse_flat_conjunctive_prop3() -> None:
    """prop_3 style: (assert (<= Y_0 Y_i))."""
    text = """\
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (<= Y_0 Y_1))
(assert (<= Y_0 Y_2))
"""
    prop = parse_vnnlib_output_property(text)
    assert not prop.disjunctive, "prop_3 should be conjunctive"
    assert len(prop.branches) == 1, f"expected 1 branch, got {len(prop.branches)}"
    assert len(prop.branches[0].clauses) == 2, (
        f"expected 2 clauses, got {len(prop.branches[0].clauses)}"
    )


def test_parse_disjunctive_prop8() -> None:
    """prop_8 style: (assert (or (and ...) (and ...) ...))."""
    text = """\
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(declare-const Y_4 Real)
(assert (or
    (and (<= Y_2 Y_0) (<= Y_2 Y_1))
    (and (<= Y_3 Y_0) (<= Y_3 Y_1))
    (and (<= Y_4 Y_0) (<= Y_4 Y_1))
))
"""
    prop = parse_vnnlib_output_property(text)
    assert prop.disjunctive, "prop_8 should be disjunctive"
    assert len(prop.branches) == 3, f"expected 3 branches, got {len(prop.branches)}"
    for i, branch in enumerate(prop.branches):
        assert len(branch.clauses) == 2, (
            f"branch {i} expected 2 clauses, got {len(branch.clauses)}"
        )


def test_parse_ignores_input_constraints() -> None:
    """Input bounds (X_i) should not appear in output property."""
    text = """\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 Y_1))
"""
    prop = parse_vnnlib_output_property(text)
    assert len(prop.branches) == 1, f"expected 1 branch, got {len(prop.branches)}"
    assert len(prop.branches[0].clauses) == 1, (
        f"expected 1 output clause, got {len(prop.branches[0].clauses)}"
    )
    assert prop.branches[0].clauses[0].lhs == "Y_0", (
        f"expected lhs=Y_0, got {prop.branches[0].clauses[0].lhs}"
    )


def test_unsupported_syntax_raises() -> None:
    """Unsupported syntax should raise VnnlibParseError."""
    text = """\
(declare-const Y_0 Real)
"""
    try:
        parse_vnnlib_output_property(text)
        assert False, "should have raised VnnlibParseError"
    except VnnlibParseError:
        pass


# ---------------------------------------------------------------------------
# Constraint evaluation tests
# ---------------------------------------------------------------------------

def test_evaluate_conjunctive_satisfied() -> None:
    """Conjunctive property satisfied when all clauses hold."""
    text = """\
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (<= Y_0 Y_1))
"""
    prop = parse_vnnlib_output_property(text)
    result = evaluate_output_property(prop, [1.0, 2.0])
    assert result is True, f"expected satisfied, got {result}"


def test_evaluate_conjunctive_violated() -> None:
    """Conjunctive property not satisfied when a clause fails."""
    text = """\
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (<= Y_0 Y_1))
"""
    prop = parse_vnnlib_output_property(text)
    result = evaluate_output_property(prop, [3.0, 2.0])
    assert result is False, f"expected not satisfied, got {result}"


def test_evaluate_disjunctive_one_branch_satisfied() -> None:
    """Disjunctive property satisfied when at least one branch holds."""
    text = """\
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (or
    (and (<= Y_0 Y_1))
    (and (<= Y_2 Y_0))
))
"""
    prop = parse_vnnlib_output_property(text)
    # Y_0=5, Y_1=3, Y_2=1 -> first branch: 5<=3 False, second: 1<=5 True
    result = evaluate_output_property(prop, [5.0, 3.0, 1.0])
    assert result is True, f"expected satisfied (second branch), got {result}"


def test_evaluate_disjunctive_no_branch_satisfied() -> None:
    """Disjunctive property not satisfied when no branch holds."""
    text = """\
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (or
    (and (<= Y_0 Y_1))
    (and (<= Y_2 Y_0))
))
"""
    prop = parse_vnnlib_output_property(text)
    # Y_0=5, Y_1=3, Y_2=10 -> first: 5<=3 False, second: 10<=5 False
    result = evaluate_output_property(prop, [5.0, 3.0, 10.0])
    assert result is False, f"expected not satisfied, got {result}"


def test_evaluate_prop2_style_coc_maximal() -> None:
    """prop_2: COC is maximal -> all Y_i <= Y_0."""
    text = """\
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(declare-const Y_4 Real)
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
(assert (<= Y_3 Y_0))
(assert (<= Y_4 Y_0))
"""
    prop = parse_vnnlib_output_property(text)
    # Y_0 is largest -> unsafe (COC maximal)
    result_sat = evaluate_output_property(prop, [10.0, 5.0, 3.0, 2.0, 1.0])
    assert result_sat is True, f"expected COC-maximal satisfied, got {result_sat}"
    # Y_0 is not largest -> safe
    result_unsat = evaluate_output_property(prop, [1.0, 5.0, 3.0, 2.0, 10.0])
    assert result_unsat is False, f"expected COC-not-maximal unsatisfied, got {result_unsat}"


# ---------------------------------------------------------------------------
# Validation script integration tests
# ---------------------------------------------------------------------------

def _write_ny_v1_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write a minimal backend_benchmark_row_v1 CSV."""
    lines = [
        "schema_version,lane,subject_kind,subject_id,comparison_key,category,workload,"
        "model_path,property_path,preset_path,backend,timeout_seconds,status,"
        "actual_method,wall_seconds,domains_explored,output_width_sum,"
        "profile_artifact_path,notes"
    ]
    for model, prop, status in rows:
        lines.append(
            f"backend_benchmark_row_v1,vnncomp_single_backend,vnncomp_instance,"
            f"id,key,cat,,{model},{prop},preset.yaml,cpu,116,{status},,1.0,0,,,"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_simple_ref_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write a simple 3-column reference CSV."""
    lines = ["model,property,result"]
    for model, prop, status in rows:
        lines.append(f"{model},{prop},{status}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_critical_mismatch_exits_1(tmp_path: Path) -> None:
    """Validation script exits 1 when critical mismatches exist."""
    ny_csv = tmp_path / "ny.csv"
    ref_csv = tmp_path / "ref.csv"
    _write_ny_v1_csv(ny_csv, [("model_a.onnx", "prop_1.vnnlib", "violated")])
    _write_simple_ref_csv(ref_csv, [("model_a.onnx", "prop_1.vnnlib", "verified")])

    result = subprocess.run(
        ["bash", str(VALIDATE_SCRIPT), str(ny_csv), str(ref_csv)],
        capture_output=True, text=True, timeout=30,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1, (
        f"expected exit 1, got {result.returncode}: {result.stdout}"
    )
    assert "CRITICAL" in result.stdout, (
        f"expected CRITICAL in output: {result.stdout}"
    )
    assert "requires replay classification" in result.stdout, (
        f"expected 'requires replay classification' in output: {result.stdout}"
    )


def test_no_critical_mismatch_exits_0(tmp_path: Path) -> None:
    """Validation script exits 0 when results agree."""
    ny_csv = tmp_path / "ny.csv"
    ref_csv = tmp_path / "ref.csv"
    _write_ny_v1_csv(ny_csv, [("model_a.onnx", "prop_1.vnnlib", "verified")])
    _write_simple_ref_csv(ref_csv, [("model_a.onnx", "prop_1.vnnlib", "verified")])

    result = subprocess.run(
        ["bash", str(VALIDATE_SCRIPT), str(ny_csv), str(ref_csv)],
        capture_output=True, text=True, timeout=30,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}: {result.stdout}"
    )
    assert "PASS" in result.stdout, f"expected PASS in output: {result.stdout}"


def test_same_basename_different_category_stays_distinct(tmp_path: Path) -> None:
    """Instances sharing a basename across categories must not share a key.

    safenlp ships onnx/medical/perturbations_0.onnx and
    onnx/ruarobot/perturbations_0.onnx with matching vnnlib names. A
    basename-only key scores ny's ruarobot verdict against medical's answer,
    hiding a verified<->violated flip.
    """
    ny_csv = tmp_path / "ny.csv"
    ref_csv = tmp_path / "ref.csv"
    _write_ny_v1_csv(ny_csv, [
        ("onnx/ruarobot/perturbations_0.onnx",
         "vnnlib/ruarobot/hyperrectangle_984.vnnlib", "verified"),
    ])
    _write_simple_ref_csv(ref_csv, [
        ("onnx/medical/perturbations_0.onnx",
         "vnnlib/medical/hyperrectangle_984.vnnlib", "verified"),
        ("onnx/ruarobot/perturbations_0.onnx",
         "vnnlib/ruarobot/hyperrectangle_984.vnnlib", "violated"),
    ])

    result = subprocess.run(
        ["bash", str(VALIDATE_SCRIPT), str(ny_csv), str(ref_csv)],
        capture_output=True, text=True, timeout=30,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1, (
        f"expected exit 1 (ruarobot verified vs ref violated), "
        f"got {result.returncode}: {result.stdout}"
    )
    assert "CRITICAL" in result.stdout, (
        f"expected CRITICAL in output: {result.stdout}"
    )


def test_differing_path_prefixes_still_match(tmp_path: Path) -> None:
    """Keys ignore the leading prefix, so a repo-relative ny path still matches
    a reference path rooted at the benchmark directory."""
    ny_csv = tmp_path / "ny.csv"
    ref_csv = tmp_path / "ref.csv"
    _write_ny_v1_csv(ny_csv, [
        ("benchmarks/vnncomp2024/benchmarks/safenlp/onnx/medical/perturbations_0.onnx",
         "benchmarks/vnncomp2024/benchmarks/safenlp/vnnlib/medical/hyperrectangle_984.vnnlib",
         "verified"),
    ])
    _write_simple_ref_csv(ref_csv, [
        ("./onnx/medical/perturbations_0.onnx",
         "./vnnlib/medical/hyperrectangle_984.vnnlib", "verified"),
    ])

    result = subprocess.run(
        ["bash", str(VALIDATE_SCRIPT), str(ny_csv), str(ref_csv)],
        capture_output=True, text=True, timeout=30,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}: {result.stdout}"
    )
    assert "Agree:              1" in result.stdout, (
        f"expected the instance to match despite prefixes: {result.stdout}"
    )


def test_ambiguous_key_fails_closed(tmp_path: Path) -> None:
    """A key carrying two verdicts is refused rather than resolved by guessing.

    ml4acopf_2023 and ml4acopf_2024 both hold onnx/14_ieee_ml4acopf.onnx with
    matching vnnlib names, so mixing benchmark versions in one run collides even
    with the directory in the key.
    """
    ny_csv = tmp_path / "ny.csv"
    ref_csv = tmp_path / "ref.csv"
    _write_ny_v1_csv(ny_csv, [
        ("ml4acopf_2024/onnx/14_ieee_ml4acopf.onnx",
         "ml4acopf_2024/vnnlib/14_ieee_prop1.vnnlib", "verified"),
    ])
    _write_simple_ref_csv(ref_csv, [
        ("ml4acopf_2023/onnx/14_ieee_ml4acopf.onnx",
         "ml4acopf_2023/vnnlib/14_ieee_prop1.vnnlib", "verified"),
        ("ml4acopf_2024/onnx/14_ieee_ml4acopf.onnx",
         "ml4acopf_2024/vnnlib/14_ieee_prop1.vnnlib", "violated"),
    ])

    result = subprocess.run(
        ["bash", str(VALIDATE_SCRIPT), str(ny_csv), str(ref_csv)],
        capture_output=True, text=True, timeout=30,
        cwd=str(tmp_path),
    )
    assert result.returncode == 2, (
        f"expected exit 2 for an ambiguous key, got {result.returncode}: "
        f"{result.stdout}{result.stderr}"
    )
    assert "PASS" not in result.stdout, (
        f"an ambiguous key must never report PASS: {result.stdout}"
    )


def test_classifier_artifact_path_printed_on_critical_mismatch(tmp_path: Path) -> None:
    """When critical mismatches exist and classifier script is available,
    the validation script should mention classifier invocation."""
    ny_csv = tmp_path / "ny.csv"
    ref_csv = tmp_path / "ref.csv"
    _write_ny_v1_csv(ny_csv, [("model_a.onnx", "prop_1.vnnlib", "violated")])
    _write_simple_ref_csv(ref_csv, [("model_a.onnx", "prop_1.vnnlib", "verified")])

    result = subprocess.run(
        ["bash", str(VALIDATE_SCRIPT), str(ny_csv), str(ref_csv)],
        capture_output=True, text=True, timeout=30,
        cwd=str(tmp_path),
    )
    # The classifier won't actually succeed (no ny binary), but the
    # validation script should still mention the classifier attempt
    assert "replay classification" in result.stdout, (
        f"expected 'replay classification' in output: {result.stdout}"
    )


def test_unsupported_vnnlib_fails_closed() -> None:
    """Unsupported VNN-LIB syntax returns replay_failed classification."""
    text = "(declare-const Y_0 Real)\n"
    try:
        parse_vnnlib_output_property(text)
        assert False, "should raise"
    except VnnlibParseError as exc:
        assert "no output constraints" in str(exc), (
            f"expected 'no output constraints' in error, got: {exc}"
        )
