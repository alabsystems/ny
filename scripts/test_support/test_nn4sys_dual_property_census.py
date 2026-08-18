#!/usr/bin/env python3
"""Focused tests for scripts/nn4sys_dual_property_census.py."""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "nn4sys_dual_property_census.py"
SPEC = importlib.util.spec_from_file_location("nn4sys_dual_property_census", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
census = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = census
SPEC.loader.exec_module(census)


def clause(bounds: list[tuple[str, str]]) -> str:
    terms = []
    for axis, (lower, upper) in enumerate(bounds):
        terms.extend([f"(>= X_{axis} {lower})", f"(<= X_{axis} {upper})"])
    terms.append("(<= Y_0 -1e-5)")
    return "(and " + " ".join(terms) + ")\n"


class Nn4sysDualPropertyCensusTest(unittest.TestCase):
    def test_exact_decimal_comparison_and_histogram(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cardinality_1_3_2048_dual.vnnlib"
            path.write_text(
                "(assert (or\n"
                + clause([("0.0", "0.0"), ("1e-1", "0.10")])
                + clause([("0.0", "0.9999"), ("2.0", "2.0")])
                + clause([("-1", "1"), ("0", "3")])
                + "))\n",
                encoding="utf-8",
            )
            result = census.census_property(path)
            self.assertEqual(result.observed_clauses, 3)
            # Decimal sees 1e-1 == 0.10 exactly.
            self.assertEqual(result.free_dimension_histogram, {0: 1, 1: 1, 2: 1})

    def test_mismatched_bounds_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cardinality_1_1_2048_dual.vnnlib"
            path.write_text(
                "(assert (or\n(and (>= X_0 0.0) (<= Y_0 -1e-5))\n))\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mismatched input bounds"):
                census.census_property(path)

    def test_csv_includes_reproducible_total(self) -> None:
        rows = [
            census.PropertyCensus(2, 2, census.Counter({1: 2})),
            census.PropertyCensus(3, 3, census.Counter({1: 1, 2: 2})),
        ]
        output = io.StringIO()
        census.write_csv(rows, output)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[-1].startswith("TOTAL,5,"))
        self.assertTrue(lines[-1].endswith(",60.000000"))


if __name__ == "__main__":
    unittest.main()
