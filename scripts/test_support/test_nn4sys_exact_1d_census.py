#!/usr/bin/env python3
"""Focused tests for scripts/nn4sys_exact_1d_census.py."""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "nn4sys_exact_1d_census.py"
SPEC = importlib.util.spec_from_file_location("nn4sys_exact_1d_census", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
census = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = census
SPEC.loader.exec_module(census)


def clause(
    *,
    inputs: int,
    free_axis: int,
    lower: str = "0.0",
    upper: str = "0.9999",
    weights: dict[int, str] | None = None,
    output: str = "(<= Y_0 0.5)",
) -> str:
    weights = weights or {}
    terms: list[str] = []
    for axis in range(inputs):
        low = weights.get(axis, "0.0")
        high = low
        if axis == free_axis:
            low, high = lower, upper
        terms.extend([f"(>= X_{axis} {low})", f"(<= X_{axis} {high})"])
    terms.append(output)
    return "(and " + " ".join(terms) + ")\n"


def property_text(clauses: list[str], *, inputs: int) -> str:
    declarations = "".join(
        f"(declare-const X_{axis} Real)\n" for axis in range(inputs)
    )
    return (
        declarations
        + "(declare-const Y_0 Real)\n\n(assert (or\n"
        + "".join(clauses)
        + "))\n"
    )


class Nn4sysExactOneAxisCensusTest(unittest.TestCase):
    def test_feature_axis_has_degree_one_mul_and_positive_constant_divisor(self) -> None:
        target = census.Target(
            kind="nondual",
            cardinality=1,
            filename="cardinality_0_1_2048.vnnlib",
            model=census.NONDUAL_MODEL,
            inputs=154,
            expected_clauses=2,
        )
        # Middle-group weight axes are 55, 69, 83, 97, 111, 125.
        weights = {55: "1.0", 69: "1e0"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / target.filename
            path.write_text(
                property_text(
                    [
                        clause(inputs=154, free_axis=54, weights=weights),
                        clause(
                            inputs=154,
                            free_axis=68,
                            weights=weights,
                            output="(>= Y_0 0.7)",
                        ),
                    ],
                    inputs=154,
                ),
                encoding="utf-8",
            )
            result = census.census_property(
                path, target, deadline=time.monotonic() + 1
            )
        self.assertEqual(result.free_dimension_histogram, {1: 2})
        self.assertEqual(result.one_axis_candidates, 2)
        self.assertEqual(result.piecewise_affine_pre_sigmoid, 2)
        self.assertEqual(result.degree_le_one_at_mul, 2)
        self.assertEqual(result.mul_nonlinear_obstructions, 0)
        self.assertEqual(result.dynamic_divisor_obstructions, 0)
        self.assertEqual(result.peelable_sigmoid, 2)
        self.assertEqual(result.free_axes, (54, 68))
        self.assertEqual(str(result.divisor_min), "2.0")
        self.assertEqual(str(result.divisor_max), "2.0")

    def test_weight_axis_is_not_mistaken_for_piecewise_affine_core(self) -> None:
        target = census.Target(
            kind="nondual",
            cardinality=1,
            filename="cardinality_0_1_2048.vnnlib",
            model=census.NONDUAL_MODEL,
            inputs=154,
            expected_clauses=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / target.filename
            path.write_text(
                property_text(
                    [
                        clause(
                            inputs=154,
                            free_axis=55,
                            lower="1.0",
                            upper="2.0",
                        )
                    ],
                    inputs=154,
                ),
                encoding="utf-8",
            )
            result = census.census_property(
                path, target, deadline=time.monotonic() + 1
            )
        self.assertEqual(result.degree_le_one_at_mul, 1)
        self.assertEqual(result.piecewise_affine_pre_sigmoid, 0)
        self.assertEqual(result.dynamic_divisor_obstructions, 1)
        self.assertEqual(result.mul_nonlinear_obstructions, 0)

    def test_incomplete_input_surface_and_expired_deadline_fail_closed(self) -> None:
        target = census.Target(
            kind="nondual",
            cardinality=1,
            filename="cardinality_0_1_2048.vnnlib",
            model=census.NONDUAL_MODEL,
            inputs=154,
            expected_clauses=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / target.filename
            path.write_text(
                "(assert (or\n(and (>= X_0 0.0) (<= X_0 1.0) "
                "(<= Y_0 0.5))\n))\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(census.CensusError, "expected 154"):
                census.census_property(
                    path, target, deadline=time.monotonic() + 1
                )
            with self.assertRaisesRegex(census.CensusError, "deadline"):
                census.census_property(path, target, deadline=time.monotonic())

    def test_csv_total_preserves_obstruction_counts(self) -> None:
        target = census.Target(
            kind="nondual",
            cardinality=1,
            filename="fixture.vnnlib",
            model=census.NONDUAL_MODEL,
            inputs=154,
            expected_clauses=3,
        )
        row = census.PropertyCensus(
            target=target,
            observed_clauses=3,
            free_dimension_histogram=census.Counter({1: 2, 2: 1}),
            one_axis_candidates=2,
            piecewise_affine_pre_sigmoid=1,
            degree_le_one_at_mul=2,
            mul_nonlinear_obstructions=0,
            dynamic_divisor_obstructions=1,
            peelable_sigmoid=1,
            constant_output=0,
            free_axes=(54, 55),
            divisor_min=census.Decimal("1"),
            divisor_max=census.Decimal("3"),
        )
        output = io.StringIO()
        census.write_csv([row], output)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[-1].startswith("TOTAL,2,1,3,"))
        self.assertIn(",2,1,2,0,1,1,0,", lines[-1])


if __name__ == "__main__":
    unittest.main()
