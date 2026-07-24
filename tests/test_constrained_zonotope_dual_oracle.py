from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "constrained_zonotope_dual_oracle.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "constrained_zonotope_dual_oracle", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


oracle = _load_module()


def test_lambda_zero_matches_unconstrained_l1_concretization() -> None:
    assert oracle.lower_certificate("3/2", [2, "-1/4"], [], [], []) == Fraction(-3, 4)
    assert oracle.upper_certificate("3/2", [2, "-1/4"], [], [], []) == Fraction(15, 4)


def test_lower_multiplier_recovers_epsilon_nonnegative_constraint() -> None:
    # min epsilon subject to epsilon >= 0 is exactly zero.
    assert oracle.lower_certificate(0, [1], [[1]], [0], [0]) == -1
    assert oracle.lower_certificate(0, [1], [[1]], [0], [1]) == 0


def test_upper_multiplier_recovers_epsilon_nonpositive_constraint() -> None:
    # max epsilon subject to -epsilon >= 0 is exactly zero.
    assert oracle.upper_certificate(0, [1], [[-1]], [0], [0]) == 1
    assert oracle.upper_certificate(0, [1], [[-1]], [0], [1]) == 0


def test_ny_leq_predicate_wrappers_apply_the_required_sign_conversion() -> None:
    # NY stores -epsilon <= 0 for epsilon >= 0.
    assert oracle.lower_certificate_leq(0, [1], [[-1]], [0], [1]) == 0
    # NY stores epsilon <= 0 directly.
    assert oracle.upper_certificate_leq(0, [1], [[1]], [0], [1]) == 0


def test_every_feasible_rational_grid_point_is_inside_certificates() -> None:
    center = "1/3"
    coefficients = ["2/3", "-3/5"]
    constraints = [[1, 1], [-1, 0]]
    offsets = ["1/4", "1/2"]
    multipliers = ["2/7", "1/9"]
    lower = oracle.lower_certificate(
        center, coefficients, constraints, offsets, multipliers
    )
    upper = oracle.upper_certificate(
        center, coefficients, constraints, offsets, multipliers
    )
    grid = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]
    for epsilon_0 in grid:
        for epsilon_1 in grid:
            if epsilon_0 + epsilon_1 + Fraction(1, 4) < 0:
                continue
            if -epsilon_0 + Fraction(1, 2) < 0:
                continue
            concrete = (
                Fraction(center)
                + Fraction(2, 3) * epsilon_0
                - Fraction(3, 5) * epsilon_1
            )
            assert lower <= concrete <= upper


def test_negative_multiplier_and_shape_mismatch_are_rejected() -> None:
    with pytest.raises(oracle.CertificateError, match="nonnegative"):
        oracle.lower_certificate(0, [1], [[1]], [0], [-1])
    with pytest.raises(oracle.CertificateError, match="columns"):
        oracle.lower_certificate(0, [1, 2], [[1]], [0], [0])


def test_cli_preserves_exact_rational_result(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "direction": "lower",
                "constraint_sense": "leq_rhs",
                "center": "1/3",
                "coefficients": [1],
                "constraints": [[-1]],
                "offsets": [0],
                "multipliers": [1],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(candidate)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["exact"] == "1/3"
    assert result["numerator"] == 1
    assert result["denominator"] == 3


def test_cli_rejects_binary_json_float(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "direction": "lower",
                "constraint_sense": "geq_zero",
                "center": 0.1,
                "coefficients": [],
                "constraints": [],
                "offsets": [],
                "multipliers": [],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(candidate)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "integer or rational string" in completed.stderr


def test_cli_rejects_missing_arrays_without_traceback(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "direction": "lower",
                "constraint_sense": "geq_zero",
                "center": 0,
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(candidate)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "coefficients must be a JSON array" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_requires_explicit_constraint_sense(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "direction": "lower",
                "center": 0,
                "coefficients": [],
                "constraints": [],
                "offsets": [],
                "multipliers": [],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(candidate)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "constraint_sense" in completed.stderr
