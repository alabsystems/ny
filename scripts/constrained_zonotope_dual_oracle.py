#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Exact-rational oracle for constrained-zonotope dual certificates.

This standalone tool is a sign-convention and test oracle, not a production
optimizer.  It evaluates a supplied nonnegative multiplier exactly with
``fractions.Fraction`` for the domain

    x = center + coefficients * epsilon
    epsilon in [-1, 1]^m
    constraints * epsilon + offsets >= 0.

JSON numbers must be integers or strings (for example ``"0.125"`` or ``"1/8"``);
binary JSON floats are rejected so the result remains exact. JSON input must
also name ``constraint_sense`` explicitly: ``"geq_zero"`` for
``A*epsilon + b >= 0`` or ``"leq_rhs"`` for NY's ``A*epsilon <= b`` storage.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Union

# Keep the standalone oracle importable on NY's Python 3.9 floor.  `TypeAlias`
# and runtime PEP 604 unions are not uniformly available there.
RationalInput = Union[int, str, Fraction]


class CertificateError(ValueError):
    """A candidate certificate is malformed."""


def _fraction(value: RationalInput, label: str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise CertificateError(f"{label} must be an integer or rational string")
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise CertificateError(f"invalid rational {label}: {value!r}") from error


def _validated(
    center: RationalInput,
    coefficients: list[RationalInput],
    constraints: list[list[RationalInput]],
    offsets: list[RationalInput],
    multipliers: list[RationalInput],
) -> tuple[
    Fraction, list[Fraction], list[list[Fraction]], list[Fraction], list[Fraction]
]:
    c = _fraction(center, "center")
    g = [_fraction(value, f"coefficients[{i}]") for i, value in enumerate(coefficients)]
    if len(constraints) != len(offsets) or len(offsets) != len(multipliers):
        raise CertificateError(
            "constraints, offsets, and multipliers must have the same row count"
        )

    matrix: list[list[Fraction]] = []
    for row_index, row in enumerate(constraints):
        if len(row) != len(g):
            raise CertificateError(
                f"constraints[{row_index}] has {len(row)} columns; expected {len(g)}"
            )
        matrix.append(
            [
                _fraction(value, f"constraints[{row_index}][{column_index}]")
                for column_index, value in enumerate(row)
            ]
        )
    b = [_fraction(value, f"offsets[{i}]") for i, value in enumerate(offsets)]
    lambdas = [
        _fraction(value, f"multipliers[{i}]") for i, value in enumerate(multipliers)
    ]
    negative = [i for i, value in enumerate(lambdas) if value < 0]
    if negative:
        raise CertificateError(
            "multipliers must be nonnegative; negative rows: "
            + ", ".join(str(i) for i in negative)
        )
    return c, g, matrix, b, lambdas


def lower_certificate(
    center: RationalInput,
    coefficients: list[RationalInput],
    constraints: list[list[RationalInput]],
    offsets: list[RationalInput],
    multipliers: list[RationalInput],
) -> Fraction:
    """Return an exact lower certificate for one output direction."""
    c, g, matrix, b, lambdas = _validated(
        center, coefficients, constraints, offsets, multipliers
    )
    adjusted = [
        coefficient
        - sum(
            (lambdas[row] * matrix[row][column] for row in range(len(matrix))),
            Fraction(0),
        )
        for column, coefficient in enumerate(g)
    ]
    offset_penalty = sum(
        # `_validated` established equal lengths above; plain `zip` retains
        # that invariant while remaining compatible with Python 3.9.
        (multiplier * offset for multiplier, offset in zip(lambdas, b)),
        Fraction(0),
    )
    return c - offset_penalty - sum(map(abs, adjusted), Fraction(0))


def upper_certificate(
    center: RationalInput,
    coefficients: list[RationalInput],
    constraints: list[list[RationalInput]],
    offsets: list[RationalInput],
    multipliers: list[RationalInput],
) -> Fraction:
    """Return an exact upper certificate for one output direction."""
    c, g, matrix, b, lambdas = _validated(
        center, coefficients, constraints, offsets, multipliers
    )
    adjusted = [
        coefficient
        + sum(
            (lambdas[row] * matrix[row][column] for row in range(len(matrix))),
            Fraction(0),
        )
        for column, coefficient in enumerate(g)
    ]
    offset_bonus = sum(
        # `_validated` established equal lengths above; plain `zip` retains
        # that invariant while remaining compatible with Python 3.9.
        (multiplier * offset for multiplier, offset in zip(lambdas, b)),
        Fraction(0),
    )
    return c + offset_bonus + sum(map(abs, adjusted), Fraction(0))


def _negated_constraints(
    constraints: list[list[RationalInput]],
) -> list[list[Fraction]]:
    return [
        [
            -_fraction(value, f"constraints[{row_index}][{column_index}]")
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(constraints)
    ]


def lower_certificate_leq(
    center: RationalInput,
    coefficients: list[RationalInput],
    constraints: list[list[RationalInput]],
    rhs: list[RationalInput],
    multipliers: list[RationalInput],
) -> Fraction:
    """Return a lower certificate for NY's ``constraints * epsilon <= rhs`` form."""
    return lower_certificate(
        center,
        coefficients,
        _negated_constraints(constraints),
        rhs,
        multipliers,
    )


def upper_certificate_leq(
    center: RationalInput,
    coefficients: list[RationalInput],
    constraints: list[list[RationalInput]],
    rhs: list[RationalInput],
    multipliers: list[RationalInput],
) -> Fraction:
    """Return an upper certificate for NY's ``constraints * epsilon <= rhs`` form."""
    return upper_certificate(
        center,
        coefficients,
        _negated_constraints(constraints),
        rhs,
        multipliers,
    )


def _result(value: Fraction) -> dict[str, object]:
    return {
        "exact": str(value),
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _json_list(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise CertificateError(f"{label} must be a JSON array")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="JSON certificate candidate (default: stdin)",
    )
    args = parser.parse_args(argv)
    try:
        if args.input is None:
            value = json.load(sys.stdin)
        else:
            value = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CertificateError("input must be a JSON object")
        direction = value.get("direction")
        constraint_sense = value.get("constraint_sense")
        function = {
            ("lower", "geq_zero"): lower_certificate,
            ("upper", "geq_zero"): upper_certificate,
            ("lower", "leq_rhs"): lower_certificate_leq,
            ("upper", "leq_rhs"): upper_certificate_leq,
        }.get((direction, constraint_sense))
        if function is None:
            raise CertificateError(
                "direction/constraint_sense must be lower|upper and geq_zero|leq_rhs"
            )
        coefficients = _json_list(value.get("coefficients"), "coefficients")
        constraints = _json_list(value.get("constraints"), "constraints")
        if any(not isinstance(row, list) for row in constraints):
            raise CertificateError("every constraints row must be a JSON array")
        offsets = _json_list(value.get("offsets"), "offsets")
        multipliers = _json_list(value.get("multipliers"), "multipliers")
        result = function(
            value.get("center"),
            coefficients,
            constraints,
            offsets,
            multipliers,
        )
    except (OSError, json.JSONDecodeError, CertificateError) as error:
        parser.error(str(error))
    json.dump(_result(result), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
