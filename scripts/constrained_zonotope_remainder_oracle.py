#!/usr/bin/env python3
# ruff: noqa: UP007, UP045
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Exact-rational oracle for constrained-zonotope box remainders.

This is a proof-arithmetic oracle, not a production transformer or optimizer.
It checks three pieces of the proposed domain

    x in c + G*alpha + [-r, r],  alpha in [-1, 1]^m,  C*alpha <= d:

* one-neuron unstable-ReLU containment and its remainder-relaxed predicate rows;
* affine remainder propagation around candidate nominal coefficients; and
* the directional dual charge ``sum_i |q_i| r_i``.

All public arithmetic accepts integers, rational strings, or ``Fraction``.
Binary floats are rejected so a failed check is an exact counterexample rather
than a floating-point tolerance decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Optional, Union

RationalInput = Union[int, str, Fraction]


class OracleInputError(ValueError):
    """The proposed exact-rational oracle input is malformed."""


class OracleCounterexampleError(AssertionError):
    """An exact witness violates a candidate containment claim."""


def _fraction(value: RationalInput, label: str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise OracleInputError(f"{label} must be an integer or rational string")
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise OracleInputError(f"invalid rational {label}: {value!r}") from error


def _vector(values: Sequence[RationalInput], label: str) -> tuple[Fraction, ...]:
    if isinstance(values, (str, bytes)):
        raise OracleInputError(f"{label} must be a rational sequence")
    return tuple(
        _fraction(value, f"{label}[{index}]") for index, value in enumerate(values)
    )


def _nonnegative_vector(
    values: Sequence[RationalInput], label: str
) -> tuple[Fraction, ...]:
    result = _vector(values, label)
    negative = [index for index, value in enumerate(result) if value < 0]
    if negative:
        joined = ", ".join(str(index) for index in negative)
        raise OracleInputError(
            f"{label} must be nonnegative; negative entries: {joined}"
        )
    return result


def _matrix(
    rows: Sequence[Sequence[RationalInput]],
    label: str,
    *,
    expected_rows: Optional[int] = None,
    expected_columns: Optional[int] = None,
) -> tuple[tuple[Fraction, ...], ...]:
    if isinstance(rows, (str, bytes)):
        raise OracleInputError(f"{label} must be a matrix")
    if expected_rows is not None and len(rows) != expected_rows:
        raise OracleInputError(
            f"{label} has {len(rows)} rows; expected {expected_rows}"
        )
    result = tuple(_vector(row, f"{label}[{index}]") for index, row in enumerate(rows))
    if expected_columns is None and result:
        expected_columns = len(result[0])
    if expected_columns is not None:
        for index, row in enumerate(result):
            if len(row) != expected_columns:
                raise OracleInputError(
                    f"{label}[{index}] has {len(row)} columns; "
                    f"expected {expected_columns}"
                )
    return result


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


@dataclass(frozen=True)
class PredicateRow:
    """One exact row ``coefficients * symbols <= rhs``."""

    name: str
    coefficients: tuple[Fraction, ...]
    rhs: Fraction

    def lhs(self, symbols: Sequence[RationalInput]) -> Fraction:
        values = _vector(symbols, f"{self.name}.symbols")
        if len(values) != len(self.coefficients):
            raise OracleInputError(
                f"{self.name} received {len(values)} symbols; "
                f"expected {len(self.coefficients)}"
            )
        return _dot(self.coefficients, values)


@dataclass(frozen=True)
class ReLUTransform:
    """Exact real-arithmetic unstable-ReLU transform."""

    input_center: Fraction
    input_generators: tuple[Fraction, ...]
    input_remainder: Fraction
    lower: Fraction
    upper: Fraction
    slope: Fraction
    delta: Fraction
    output_center: Fraction
    output_generators: tuple[Fraction, ...]
    output_remainder: Fraction
    y_ge_x: PredicateRow
    y_ge_zero: PredicateRow


@dataclass(frozen=True)
class ReLUWitness:
    """Exact witness mapping one predecessor point through ReLU."""

    alpha: tuple[Fraction, ...]
    input_error: Fraction
    nominal_input: Fraction
    actual_input: Fraction
    eta: Fraction
    nominal_output: Fraction
    output_error: Fraction
    actual_output: Fraction


def unstable_relu_transform(
    center: RationalInput,
    generator_row: Sequence[RationalInput],
    remainder: RationalInput,
    lower: RationalInput,
    upper: RationalInput,
) -> ReLUTransform:
    """Construct the canonical exact one-neuron DeepZ transform.

    This computes ``a=u/(u-l)`` and ``delta=-u*l/(2*(u-l))`` in exact rational
    arithmetic, then delegates to :func:`relu_transform_with_parameters`.
    """
    lo = _fraction(lower, "lower")
    hi = _fraction(upper, "upper")
    if not lo < 0 < hi:
        raise OracleInputError("unstable ReLU requires lower < 0 < upper")
    slope = hi / (hi - lo)
    delta = -hi * lo / (2 * (hi - lo))
    return relu_transform_with_parameters(
        center,
        generator_row,
        remainder,
        lo,
        hi,
        slope,
        delta,
    )


def relu_transform_with_parameters(
    center: RationalInput,
    generator_row: Sequence[RationalInput],
    remainder: RationalInput,
    lower: RationalInput,
    upper: RationalInput,
    slope: RationalInput,
    delta: RationalInput,
) -> ReLUTransform:
    """Validate chosen ReLU coefficients and construct sound predicate rows.

    The new symbol ordering is ``[old alpha..., eta]``. Predicate rows use NY's
    storage convention ``C * symbols <= d``.  Any ``0 <= a <= 1`` is sound when

    ``2*delta >= max(-a*lower, (1-a)*upper)``.

    This form lets an implementation submit exact encodings of nominal f64
    coefficients and prove that an outward ``delta`` still covers the envelope.
    """
    c = _fraction(center, "center")
    generators = _vector(generator_row, "generator_row")
    radius = _fraction(remainder, "remainder")
    lo = _fraction(lower, "lower")
    hi = _fraction(upper, "upper")
    chosen_slope = _fraction(slope, "slope")
    chosen_delta = _fraction(delta, "delta")
    if radius < 0:
        raise OracleInputError("remainder must be nonnegative")
    if not lo < 0 < hi:
        raise OracleInputError("unstable ReLU requires lower < 0 < upper")
    if not 0 <= chosen_slope <= 1:
        raise OracleInputError("ReLU slope must be in [0, 1]")
    required_band = max(-chosen_slope * lo, (1 - chosen_slope) * hi)
    if chosen_delta <= 0 or 2 * chosen_delta < required_band:
        raise OracleCounterexampleError(
            "ReLU band is too small: "
            f"2*delta={2 * chosen_delta} < required={required_band}"
        )

    one_minus_slope = 1 - chosen_slope
    output_center = chosen_slope * c + chosen_delta
    output_generators = tuple(chosen_slope * value for value in generators) + (
        chosen_delta,
    )
    output_remainder = chosen_slope * radius

    # For an actual predecessor x = c + g*alpha + e and the witness eta below,
    # y_hat - x_hat >= -(1-a)r and y_hat >= -a*r.  These rows preserve every
    # actual predecessor point without pretending that the independent output
    # remainder itself satisfies a ReLU graph constraint.
    y_ge_x = PredicateRow(
        name="y_ge_x",
        coefficients=(
            tuple(one_minus_slope * value for value in generators) + (-chosen_delta,)
        ),
        rhs=chosen_delta + one_minus_slope * (radius - c),
    )
    y_ge_zero = PredicateRow(
        name="y_ge_zero",
        coefficients=tuple(-chosen_slope * value for value in generators)
        + (-chosen_delta,),
        rhs=chosen_slope * c + chosen_delta + chosen_slope * radius,
    )
    return ReLUTransform(
        input_center=c,
        input_generators=generators,
        input_remainder=radius,
        lower=lo,
        upper=hi,
        slope=chosen_slope,
        delta=chosen_delta,
        output_center=output_center,
        output_generators=output_generators,
        output_remainder=output_remainder,
        y_ge_x=y_ge_x,
        y_ge_zero=y_ge_zero,
    )


def check_relu_witness(
    transform: ReLUTransform,
    alpha: Sequence[RationalInput],
    input_error: RationalInput,
    *,
    predicate_rows: Optional[Sequence[PredicateRow]] = None,
) -> ReLUWitness:
    """Check one exact predecessor point and return its DeepZ witness.

    ``OracleCounterexampleError`` identifies a failed containment or predicate row.
    Out-of-domain samples are input errors rather than skipped cases.
    """
    symbols = _vector(alpha, "alpha")
    if len(symbols) != len(transform.input_generators):
        raise OracleInputError(
            f"alpha has {len(symbols)} entries; "
            f"expected {len(transform.input_generators)}"
        )
    outside_box = [index for index, value in enumerate(symbols) if abs(value) > 1]
    if outside_box:
        joined = ", ".join(str(index) for index in outside_box)
        raise OracleInputError(f"alpha entries outside [-1, 1]: {joined}")
    error = _fraction(input_error, "input_error")
    if abs(error) > transform.input_remainder:
        raise OracleInputError("input_error exceeds the certified remainder")

    nominal_input = transform.input_center + _dot(transform.input_generators, symbols)
    actual_input = nominal_input + error
    if not transform.lower <= actual_input <= transform.upper:
        raise OracleInputError(
            "sample lies outside the certified ReLU bounds: "
            f"{actual_input} not in [{transform.lower}, {transform.upper}]"
        )
    actual_output = max(Fraction(0), actual_input)
    eta = (
        actual_output - transform.slope * actual_input - transform.delta
    ) / transform.delta
    if not -1 <= eta <= 1:
        raise OracleCounterexampleError(
            f"DeepZ witness eta={eta} is outside [-1, 1] at x={actual_input}"
        )

    all_symbols = symbols + (eta,)
    nominal_output = transform.output_center + _dot(
        transform.output_generators, all_symbols
    )
    output_error = transform.slope * error
    represented_output = nominal_output + output_error
    if represented_output != actual_output:
        raise OracleCounterexampleError(
            "ReLU representation mismatch: "
            f"represented={represented_output}, actual={actual_output}"
        )
    if abs(output_error) > transform.output_remainder:
        raise OracleCounterexampleError(
            f"output error {output_error} exceeds radius {transform.output_remainder}"
        )

    rows = (
        (transform.y_ge_x, transform.y_ge_zero)
        if predicate_rows is None
        else tuple(predicate_rows)
    )
    for row in rows:
        lhs = row.lhs(all_symbols)
        if lhs > row.rhs:
            raise OracleCounterexampleError(
                f"predicate {row.name} fails at alpha={symbols}, e={error}, "
                f"eta={eta}: lhs={lhs} > rhs={row.rhs}"
            )
    return ReLUWitness(
        alpha=symbols,
        input_error=error,
        nominal_input=nominal_input,
        actual_input=actual_input,
        eta=eta,
        nominal_output=nominal_output,
        output_error=output_error,
        actual_output=actual_output,
    )


@dataclass(frozen=True)
class AffinePropagation:
    """A checked affine candidate plus its proof-safe output remainder."""

    center: tuple[Fraction, ...]
    generators: tuple[tuple[Fraction, ...], ...]
    remainder: tuple[Fraction, ...]
    ideal_center: tuple[Fraction, ...]
    ideal_generators: tuple[tuple[Fraction, ...], ...]
    input_box_charge: tuple[Fraction, ...]
    center_error_budget: tuple[Fraction, ...]
    generator_error_budget: tuple[tuple[Fraction, ...], ...]


def _affine_ideal(
    weights: Sequence[Sequence[RationalInput]],
    center: Sequence[RationalInput],
    generators: Sequence[Sequence[RationalInput]],
    remainder: Sequence[RationalInput],
    bias: Sequence[RationalInput],
) -> tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
    tuple[Fraction, ...],
]:
    input_center = _vector(center, "center")
    if not input_center:
        raise OracleInputError("affine input must have at least one coordinate")
    input_remainder = _nonnegative_vector(remainder, "remainder")
    if len(input_remainder) != len(input_center):
        raise OracleInputError(
            f"remainder has {len(input_remainder)} entries; "
            f"expected {len(input_center)}"
        )
    matrix = _matrix(weights, "weights", expected_columns=len(input_center))
    if not matrix:
        raise OracleInputError("affine output must have at least one coordinate")
    offsets = _vector(bias, "bias")
    if len(offsets) != len(matrix):
        raise OracleInputError(
            f"bias has {len(offsets)} entries; expected {len(matrix)}"
        )
    generator_matrix = _matrix(
        generators, "generators", expected_rows=len(input_center)
    )
    symbol_count = len(generator_matrix[0]) if generator_matrix else 0
    for index, row in enumerate(generator_matrix):
        if len(row) != symbol_count:
            raise OracleInputError(
                f"generators[{index}] has {len(row)} columns; expected {symbol_count}"
            )

    ideal_center = tuple(
        _dot(row, input_center) + offsets[index] for index, row in enumerate(matrix)
    )
    ideal_generators = tuple(
        tuple(
            sum(
                (
                    row[column] * generator_matrix[column][symbol]
                    for column in range(len(input_center))
                ),
                Fraction(0),
            )
            for symbol in range(symbol_count)
        )
        for row in matrix
    )
    input_box_charge = tuple(
        sum(
            (abs(value) * input_remainder[column] for column, value in enumerate(row)),
            Fraction(0),
        )
        for row in matrix
    )
    return (
        matrix,
        ideal_center,
        ideal_generators,
        input_box_charge,
        input_center,
    )


def certify_affine_propagation(
    weights: Sequence[Sequence[RationalInput]],
    center: Sequence[RationalInput],
    generators: Sequence[Sequence[RationalInput]],
    remainder: Sequence[RationalInput],
    bias: Sequence[RationalInput],
    candidate_center: Sequence[RationalInput],
    candidate_generators: Sequence[Sequence[RationalInput]],
    center_error_budget: Sequence[RationalInput],
    generator_error_budget: Sequence[Sequence[RationalInput]],
) -> AffinePropagation:
    """Validate nominal affine coefficients and construct their output remainder.

    If ``bar_c = W*c+b`` and ``bar_G = W*G``, supplied nonnegative budgets must
    cover ``|candidate_center-bar_c|`` and each coefficient of
    ``|candidate_generators-bar_G|`` exactly.  The result uses

        r'_i = sum_j |W_ij| r_j + rho_c_i + sum_k rho_G_ik.
    """
    (
        _matrix_weights,
        ideal_center,
        ideal_generators,
        input_box_charge,
        _input_center,
    ) = _affine_ideal(weights, center, generators, remainder, bias)
    output_count = len(ideal_center)
    symbol_count = len(ideal_generators[0]) if ideal_generators else 0

    nominal_center = _vector(candidate_center, "candidate_center")
    if len(nominal_center) != output_count:
        raise OracleInputError(
            f"candidate_center has {len(nominal_center)} entries; "
            f"expected {output_count}"
        )
    nominal_generators = _matrix(
        candidate_generators,
        "candidate_generators",
        expected_rows=output_count,
        expected_columns=symbol_count,
    )
    center_budget = _nonnegative_vector(center_error_budget, "center_error_budget")
    if len(center_budget) != output_count:
        raise OracleInputError(
            f"center_error_budget has {len(center_budget)} entries; "
            f"expected {output_count}"
        )
    generator_budget = _matrix(
        generator_error_budget,
        "generator_error_budget",
        expected_rows=output_count,
        expected_columns=symbol_count,
    )
    for row_index, row in enumerate(generator_budget):
        negative = [index for index, value in enumerate(row) if value < 0]
        if negative:
            joined = ", ".join(str(index) for index in negative)
            raise OracleInputError(
                "generator_error_budget must be nonnegative; "
                f"row {row_index} negative columns: {joined}"
            )

    for output, (nominal, ideal, budget) in enumerate(
        zip(nominal_center, ideal_center, center_budget)
    ):
        error = abs(nominal - ideal)
        if error > budget:
            raise OracleCounterexampleError(
                f"center coefficient {output} error {error} exceeds budget {budget}"
            )
    for output in range(output_count):
        for symbol in range(symbol_count):
            error = abs(
                nominal_generators[output][symbol] - ideal_generators[output][symbol]
            )
            budget = generator_budget[output][symbol]
            if error > budget:
                raise OracleCounterexampleError(
                    f"generator coefficient ({output}, {symbol}) error {error} "
                    f"exceeds budget {budget}"
                )

    output_remainder = tuple(
        input_box_charge[output]
        + center_budget[output]
        + sum(generator_budget[output], Fraction(0))
        for output in range(output_count)
    )
    return AffinePropagation(
        center=nominal_center,
        generators=nominal_generators,
        remainder=output_remainder,
        ideal_center=ideal_center,
        ideal_generators=ideal_generators,
        input_box_charge=input_box_charge,
        center_error_budget=center_budget,
        generator_error_budget=generator_budget,
    )


def gamma(operation_count: int, unit_roundoff: RationalInput) -> Fraction:
    """Return the exact textbook ``gamma_n = n*u/(1-n*u)`` value."""
    if isinstance(operation_count, bool) or not isinstance(operation_count, int):
        raise OracleInputError("operation_count must be an integer")
    if operation_count < 0:
        raise OracleInputError("operation_count must be nonnegative")
    rounding = _fraction(unit_roundoff, "unit_roundoff")
    if rounding < 0:
        raise OracleInputError("unit_roundoff must be nonnegative")
    product_value = operation_count * rounding
    if product_value >= 1:
        raise OracleInputError("gamma_n requires operation_count * unit_roundoff < 1")
    return product_value / (1 - product_value)


def affine_gamma_budgets(
    weights: Sequence[Sequence[RationalInput]],
    center: Sequence[RationalInput],
    generators: Sequence[Sequence[RationalInput]],
    bias: Sequence[RationalInput],
    unit_roundoff: RationalInput = "1/9007199254740992",
) -> tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]]:
    """Compute exact sum-absolute-value gamma budgets for naive f64 dot products.

    This only instantiates the standard normal-arithmetic model.  A production
    checker must separately exclude overflow/underflow/FTZ and match the actual
    FMA/reduction order, or use directed interval accumulation instead.
    """
    input_center = _vector(center, "center")
    if not input_center:
        raise OracleInputError("affine input must have at least one coordinate")
    matrix = _matrix(weights, "weights", expected_columns=len(input_center))
    offsets = _vector(bias, "bias")
    if len(offsets) != len(matrix):
        raise OracleInputError(
            f"bias has {len(offsets)} entries; expected {len(matrix)}"
        )
    generator_matrix = _matrix(
        generators, "generators", expected_rows=len(input_center)
    )
    symbol_count = len(generator_matrix[0]) if generator_matrix else 0
    for index, row in enumerate(generator_matrix):
        if len(row) != symbol_count:
            raise OracleInputError(
                f"generators[{index}] has {len(row)} columns; expected {symbol_count}"
            )
    rounding = _fraction(unit_roundoff, "unit_roundoff")
    center_gamma = gamma(len(input_center) + 1, rounding)
    generator_gamma = gamma(len(input_center), rounding)
    center_budget = tuple(
        center_gamma
        * (
            sum(
                (abs(value * input_center[column]) for column, value in enumerate(row)),
                Fraction(0),
            )
            + abs(offsets[output])
        )
        for output, row in enumerate(matrix)
    )
    generator_budget = tuple(
        tuple(
            generator_gamma
            * sum(
                (
                    abs(row[column] * generator_matrix[column][symbol])
                    for column in range(len(input_center))
                ),
                Fraction(0),
            )
            for symbol in range(symbol_count)
        )
        for row in matrix
    )
    return center_budget, generator_budget


@dataclass(frozen=True)
class DirectionalCertificate:
    """An exact constrained-zonotope directional certificate."""

    direction: str
    value: Fraction
    nominal_value: Fraction
    box_charge: Fraction


def directional_box_charge(
    direction: Sequence[RationalInput], remainder: Sequence[RationalInput]
) -> Fraction:
    """Return the box support ``sum_i |q_i| r_i`` exactly."""
    query = _vector(direction, "direction")
    radius = _nonnegative_vector(remainder, "remainder")
    if len(query) != len(radius):
        raise OracleInputError(
            f"direction has {len(query)} entries; expected {len(radius)}"
        )
    return sum(
        (abs(query[index]) * radius[index] for index in range(len(query))),
        Fraction(0),
    )


def directional_certificate_leq(
    center: Sequence[RationalInput],
    generators: Sequence[Sequence[RationalInput]],
    remainder: Sequence[RationalInput],
    constraints: Sequence[Sequence[RationalInput]],
    rhs: Sequence[RationalInput],
    direction: Sequence[RationalInput],
    multipliers: Sequence[RationalInput],
    bound: str,
) -> DirectionalCertificate:
    """Evaluate a lower or upper dual candidate for ``C*alpha <= rhs``."""
    c = _vector(center, "center")
    if not c:
        raise OracleInputError("directional state must have at least one coordinate")
    q = _vector(direction, "direction")
    radius = _nonnegative_vector(remainder, "remainder")
    if len(q) != len(c) or len(radius) != len(c):
        raise OracleInputError("center, remainder, and direction sizes must match")
    generator_matrix = _matrix(generators, "generators", expected_rows=len(c))
    symbol_count = len(generator_matrix[0]) if generator_matrix else 0
    for index, row in enumerate(generator_matrix):
        if len(row) != symbol_count:
            raise OracleInputError(
                f"generators[{index}] has {len(row)} columns; expected {symbol_count}"
            )
    predicate = _matrix(constraints, "constraints", expected_columns=symbol_count)
    predicate_rhs = _vector(rhs, "rhs")
    lambdas = _nonnegative_vector(multipliers, "multipliers")
    if len(predicate) != len(predicate_rhs) or len(predicate_rhs) != len(lambdas):
        raise OracleInputError(
            "constraints, rhs, and multipliers must have the same row count"
        )
    if bound not in {"lower", "upper"}:
        raise OracleInputError("bound must be 'lower' or 'upper'")

    scalar_center = _dot(c, q)
    scalar_generators = tuple(
        sum(
            (q[output] * generator_matrix[output][symbol] for output in range(len(c))),
            Fraction(0),
        )
        for symbol in range(symbol_count)
    )
    lambda_rhs = _dot(lambdas, predicate_rhs)
    lambda_constraints = tuple(
        sum(
            (lambdas[row] * predicate[row][symbol] for row in range(len(predicate))),
            Fraction(0),
        )
        for symbol in range(symbol_count)
    )
    charge = directional_box_charge(q, radius)
    if bound == "lower":
        nominal = (
            scalar_center
            - lambda_rhs
            - sum(
                (
                    abs(scalar_generators[symbol] + lambda_constraints[symbol])
                    for symbol in range(symbol_count)
                ),
                Fraction(0),
            )
        )
        value = nominal - charge
    else:
        nominal = (
            scalar_center
            + lambda_rhs
            + sum(
                (
                    abs(scalar_generators[symbol] - lambda_constraints[symbol])
                    for symbol in range(symbol_count)
                ),
                Fraction(0),
            )
        )
        value = nominal + charge
    return DirectionalCertificate(
        direction=bound,
        value=value,
        nominal_value=nominal,
        box_charge=charge,
    )


def run_self_check() -> dict[str, object]:
    """Run deterministic exact checks, including a required corner rejection."""
    transform = unstable_relu_transform(0, [1], 1, -2, 2)
    checked = 0
    grid = (Fraction(-1), Fraction(0), Fraction(1))
    for alpha, error in product(grid, repeat=2):
        check_relu_witness(transform, [alpha], error)
        checked += 1

    # Omitting the (1-a)r slack is unsound.  At alpha=1,e=-1 the actual input
    # is exactly the ReLU kink while the nominal y-x gap is -1/2.
    zero_slack = PredicateRow(
        name="candidate_y_ge_x_without_remainder_slack",
        coefficients=transform.y_ge_x.coefficients,
        rhs=transform.y_ge_x.rhs - (1 - transform.slope) * transform.input_remainder,
    )
    try:
        check_relu_witness(
            transform,
            [1],
            -1,
            predicate_rows=(zero_slack, transform.y_ge_zero),
        )
    except OracleCounterexampleError:
        rejected_corner = True
    else:
        raise OracleCounterexampleError("zero-slack ReLU row unexpectedly passed")

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
    affine = certify_affine_propagation(
        weights,
        center,
        generators,
        remainder,
        bias,
        candidate_center,
        candidate_generators,
        ["1/100", "1/50"],
        [["1/1000", "1/2000"], [0, "1/3000"]],
    )
    certificate = directional_certificate_leq(
        [0], [[1]], ["1/4"], [[-1]], [0], [2], [2], "lower"
    )
    if certificate.value != Fraction(-1, 2):
        raise OracleCounterexampleError(
            f"unexpected directional certificate {certificate.value}"
        )
    return {
        "status": "ACCEPT",
        "arithmetic": "exact-rational",
        "relu_samples_checked": checked,
        "zero_slack_corner_rejected": rejected_corner,
        "affine_remainder": [str(value) for value in affine.remainder],
        "directional_box_charge": str(certificate.box_charge),
        "directional_lower": str(certificate.value),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run deterministic exact proof-arithmetic checks",
    )
    args = parser.parse_args(argv)
    if not args.self_check:
        parser.error("this proof oracle currently requires --self-check")
    try:
        result = run_self_check()
    except (OracleInputError, OracleCounterexampleError) as error:
        parser.error(str(error))
    json.dump(result, fp=sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
