from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "constrained_zonotope_remainder_oracle.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "constrained_zonotope_remainder_oracle", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


oracle = _load_module()


def _affine_fixture():
    weights = [[2, -3], ["1/2", 4]]
    center = [1, -2]
    generators = [[1, "1/3"], [-2, 1]]
    remainder = ["1/10", "1/5"]
    bias = ["1/2", -1]
    candidate_center = ["851/100", "-426/50"]
    candidate_generators = [
        ["8001/1000", "-14003/6000"],
        ["-15/2", "12501/3000"],
    ]
    center_budget = ["1/100", "1/50"]
    generator_budget = [["1/1000", "1/2000"], [0, "1/3000"]]
    return (
        weights,
        center,
        generators,
        remainder,
        bias,
        candidate_center,
        candidate_generators,
        center_budget,
        generator_budget,
    )


def test_canonical_unstable_relu_transform_has_exact_rows_and_remainder() -> None:
    transform = oracle.unstable_relu_transform(0, [1], 1, -2, 2)
    assert transform.slope == Fraction(1, 2)
    assert transform.delta == Fraction(1, 2)
    assert transform.output_center == Fraction(1, 2)
    assert transform.output_generators == (Fraction(1, 2), Fraction(1, 2))
    assert transform.output_remainder == Fraction(1, 2)
    assert transform.y_ge_x.coefficients == (Fraction(1, 2), Fraction(-1, 2))
    assert transform.y_ge_x.rhs == 1
    assert transform.y_ge_zero.coefficients == (
        Fraction(-1, 2),
        Fraction(-1, 2),
    )
    assert transform.y_ge_zero.rhs == 1


def test_relu_witness_contains_every_exact_grid_point() -> None:
    transform = oracle.unstable_relu_transform(0, [1], 1, -2, 2)
    grid = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]
    for alpha, error in itertools.product(grid, repeat=2):
        witness = oracle.check_relu_witness(transform, [alpha], error)
        assert -1 <= witness.eta <= 1
        assert witness.actual_output == max(Fraction(0), witness.actual_input)
        assert witness.nominal_output + witness.output_error == witness.actual_output


def test_relu_rows_pin_nonzero_center_and_multisymbol_signs() -> None:
    transform = oracle.unstable_relu_transform("1/3", ["2/3", "-1/4"], "1/5", -1, 2)
    assert transform.slope == Fraction(2, 3)
    assert transform.delta == Fraction(1, 3)
    assert transform.y_ge_x.coefficients == (
        Fraction(2, 9),
        Fraction(-1, 12),
        Fraction(-1, 3),
    )
    assert transform.y_ge_x.rhs == Fraction(13, 45)
    assert transform.y_ge_zero.coefficients == (
        Fraction(-4, 9),
        Fraction(1, 6),
        Fraction(-1, 3),
    )
    assert transform.y_ge_zero.rhs == Fraction(31, 45)
    grid = [Fraction(-1), Fraction(0), Fraction(1)]
    errors = [Fraction(-1, 5), Fraction(0), Fraction(1, 5)]
    for alpha_0, alpha_1, error in itertools.product(grid, grid, errors):
        oracle.check_relu_witness(transform, [alpha_0, alpha_1], error)


def test_outward_chosen_relu_band_is_accepted_and_underbudget_is_rejected() -> None:
    transform = oracle.relu_transform_with_parameters(
        0, [1], "1/4", -2, 2, "1/4", "3/4"
    )
    # max(-a*l, (1-a)*u) = max(1/2, 3/2), exactly 2*delta.
    assert 2 * transform.delta == Fraction(3, 2)
    for alpha, error in itertools.product(
        [Fraction(-1), Fraction(0), Fraction(1)],
        [Fraction(-1, 4), Fraction(0), Fraction(1, 4)],
    ):
        oracle.check_relu_witness(transform, [alpha], error)

    with pytest.raises(oracle.OracleCounterexampleError, match="band is too small"):
        oracle.relu_transform_with_parameters(0, [1], 1, -2, 2, "1/2", "49/100")


def test_zero_slack_relu_predicate_fails_at_exact_corner() -> None:
    transform = oracle.unstable_relu_transform(0, [1], 1, -2, 2)
    zero_slack = oracle.PredicateRow(
        name="zero_slack_y_ge_x",
        coefficients=transform.y_ge_x.coefficients,
        rhs=transform.y_ge_x.rhs - (1 - transform.slope) * transform.input_remainder,
    )
    with pytest.raises(oracle.OracleCounterexampleError, match=r"lhs=1 > rhs=1/2"):
        oracle.check_relu_witness(
            transform,
            [1],
            -1,
            predicate_rows=(zero_slack, transform.y_ge_zero),
        )


def test_relu_oracle_rejects_binary_float_and_out_of_box_samples() -> None:
    with pytest.raises(oracle.OracleInputError, match="rational string"):
        oracle.unstable_relu_transform(0.0, [1], 1, -2, 2)
    transform = oracle.unstable_relu_transform(0, [1], 1, -2, 2)
    with pytest.raises(oracle.OracleInputError, match="outside"):
        oracle.check_relu_witness(transform, ["1001/1000"], 0)
    with pytest.raises(oracle.OracleInputError, match="exceeds"):
        oracle.check_relu_witness(transform, [0], "1001/1000")


def test_affine_candidate_errors_are_lumped_into_output_remainder() -> None:
    fixture = _affine_fixture()
    result = oracle.certify_affine_propagation(*fixture)
    assert result.ideal_center == (Fraction(17, 2), Fraction(-17, 2))
    assert result.ideal_generators == (
        (Fraction(8), Fraction(-7, 3)),
        (Fraction(-15, 2), Fraction(25, 6)),
    )
    assert result.input_box_charge == (Fraction(4, 5), Fraction(17, 20))
    assert result.remainder == (Fraction(1623, 2000), Fraction(2611, 3000))


def test_affine_remainder_contains_all_exact_alpha_and_error_corners() -> None:
    fixture = _affine_fixture()
    result = oracle.certify_affine_propagation(*fixture)
    weights = tuple(tuple(Fraction(value) for value in row) for row in fixture[0])
    center = tuple(Fraction(value) for value in fixture[1])
    generators = tuple(tuple(Fraction(value) for value in row) for row in fixture[2])
    radii = tuple(Fraction(value) for value in fixture[3])
    bias = tuple(Fraction(value) for value in fixture[4])

    for alpha in itertools.product([Fraction(-1), Fraction(1)], repeat=2):
        for signs in itertools.product([Fraction(-1), Fraction(1)], repeat=2):
            input_value = tuple(
                center[index]
                + sum(generators[index][symbol] * alpha[symbol] for symbol in range(2))
                + signs[index] * radii[index]
                for index in range(2)
            )
            actual = tuple(
                sum(
                    weights[output][column] * input_value[column] for column in range(2)
                )
                + bias[output]
                for output in range(2)
            )
            nominal = tuple(
                result.center[output]
                + sum(
                    result.generators[output][symbol] * alpha[symbol]
                    for symbol in range(2)
                )
                for output in range(2)
            )
            for output in range(2):
                assert abs(actual[output] - nominal[output]) <= result.remainder[output]


def test_affine_oracle_rejects_underbudgeted_candidate_and_negative_budget() -> None:
    fixture = list(_affine_fixture())
    fixture[7] = ["1/200", "1/50"]
    with pytest.raises(oracle.OracleCounterexampleError, match="exceeds budget"):
        oracle.certify_affine_propagation(*fixture)

    fixture = list(_affine_fixture())
    fixture[8] = [["-1/1000", "1/2000"], [0, "1/3000"]]
    with pytest.raises(oracle.OracleInputError, match="nonnegative"):
        oracle.certify_affine_propagation(*fixture)


def test_gamma_sum_abs_budgets_are_exact_and_guard_assumptions() -> None:
    assert oracle.gamma(2, "1/16") == Fraction(1, 7)
    center_budget, generator_budget = oracle.affine_gamma_budgets(
        [[2, -3]], [1, -2], [[1], [-2]], ["1/2"], "1/16"
    )
    assert center_budget == (Fraction(51, 26),)
    assert generator_budget == ((Fraction(8, 7),),)
    with pytest.raises(oracle.OracleInputError, match="requires"):
        oracle.gamma(16, "1/16")


def test_directional_box_charge_and_leq_dual_cover_exact_grid() -> None:
    assert oracle.directional_box_charge([2, -3], ["1/4", "1/6"]) == 1
    lower = oracle.directional_certificate_leq(
        [0], [[1]], ["1/4"], [[-1]], [0], [2], [2], "lower"
    )
    upper = oracle.directional_certificate_leq(
        [0], [[1]], ["1/4"], [[-1]], [0], [2], [0], "upper"
    )
    assert lower.nominal_value == 0
    assert lower.box_charge == Fraction(1, 2)
    assert lower.value == Fraction(-1, 2)
    assert upper.value == Fraction(5, 2)
    for alpha in [Fraction(0), Fraction(1, 2), Fraction(1)]:
        for error in [Fraction(-1, 4), Fraction(0), Fraction(1, 4)]:
            concrete = 2 * (alpha + error)
            assert lower.value <= concrete <= upper.value


def test_directional_oracle_rejects_negative_remainder_or_multiplier() -> None:
    with pytest.raises(oracle.OracleInputError, match="nonnegative"):
        oracle.directional_box_charge([1], [-1])
    with pytest.raises(oracle.OracleInputError, match="nonnegative"):
        oracle.directional_certificate_leq(
            [0], [[1]], [0], [[1]], [0], [1], [-1], "lower"
        )


def test_cli_self_check_reports_exact_corner_rejection() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-check"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "ACCEPT"
    assert result["arithmetic"] == "exact-rational"
    assert result["relu_samples_checked"] == 9
    assert result["zero_slack_corner_rejected"] is True
    assert result["directional_box_charge"] == "1/2"


def test_cli_requires_explicit_self_check_without_traceback() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "requires --self-check" in completed.stderr
    assert "Traceback" not in completed.stderr
