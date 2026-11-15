#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Tiny exact-fixture checks for NY's multiplier-only numerical proposer."""
from argparse import Namespace
from pathlib import Path
import tempfile
import time
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import paired_lp_dual_proposal as proposal


def row(terms, rhs=0.0, pivot=None):
    terms = sorted(terms)
    return proposal.Row(np.asarray([j for j, _ in terms], dtype=np.int64),
                        np.asarray([v for _, v in terms], dtype=np.float64), rhs, pivot)


def options(steps=0):
    return Namespace(steps=steps, learning_rate=0.01, anneal=0.995, record_every=10)


class NativeDualTests(unittest.TestCase):
    def test_affine_elimination_and_recovery_exact_constant_difference(self):
        model = proposal.Model(np.asarray([-1.0, 1.0, 2.0]), np.asarray([1.0, 5.0, 6.0]),
            [row([(0, -2.0), (1, 1.0)], 3.0, 1), row([(0, -2.0), (2, 1.0)], 4.0, 2)], [], [(1, 2)])
        result = proposal.optimize(model, options(), time.monotonic() + 5.0)
        positive, negative = result["candidates"]
        self.assertEqual(positive["proposal_lower_bound"], -1.0)
        self.assertEqual(negative["proposal_lower_bound"], 1.0)
        self.assertEqual(positive["equality_multipliers"], [1.0, -1.0])
        self.assertEqual(negative["equality_multipliers"], [-1.0, 1.0])
        self.assertEqual(positive["inequality_multipliers"], [])

    def test_joint_plane_seed_preserves_identical_pair_cancellation(self):
        inequalities = [
            row([(3, 1.0)]), row([(1, -1.0), (3, 1.0)]), row([(1, 0.5), (3, -1.0)], -0.5),
            row([(4, 1.0)]), row([(2, -1.0), (4, 1.0)]), row([(2, 0.5), (4, -1.0)], -0.5),
            row([(3, 1.0), (4, -1.0)]), row([(3, -1.0), (4, 1.0)]),
        ]
        model = proposal.Model(np.asarray([-1.0, -1.0, -1.0, 0.0, 0.0]),
            np.asarray([1.0, 1.0, 1.0, 1.0, 1.0]),
            [row([(0, -1.0), (1, 1.0)], pivot=1), row([(0, -1.0), (2, 1.0)], pivot=2)],
            inequalities, [(3, 4)])
        result = proposal.optimize(model, options(20), time.monotonic() + 5.0)
        for candidate in result["candidates"]:
            self.assertEqual(candidate["proposal_lower_bound"], 0.0)
            self.assertTrue(all(v >= 0.0 for v in candidate["inequality_multipliers"]))
            # Check the emitted GEQ dual independently on this integer fixture:
            # every residual and weighted RHS is exactly zero.
            residual = np.zeros(5)
            f, g = model.outputs[0]
            residual[f] = candidate["sign"]
            residual[g] = -candidate["sign"]
            rhs = 0.0
            for multipliers, rows in [(candidate["equality_multipliers"], model.equalities),
                                       (candidate["inequality_multipliers"], model.inequalities)]:
                for multiplier, source in zip(multipliers, rows):
                    residual[source.columns] -= multiplier * source.coefficients
                    rhs += multiplier * source.rhs
            self.assertTrue(np.all(residual == 0.0))
            self.assertEqual(rhs, 0.0)

    def test_explicit_geq_and_authored_pivots_are_required(self):
        valid = "PAIRED_LP_GEQ_V1 3 2 0 1\n-1 1\n1 5\n2 6\n1 2 0 -2 1 1 3\n2 2 0 -2 2 1 4\n1 2\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.lp"
            path.write_text(valid)
            self.assertEqual(len(proposal.read_model(path).equalities), 2)
            for bad in [valid.replace("GEQ", "LEQ"), valid.replace("1 2 0 -2 1 1 3", "0 2 0 -2 1 1 3"),
                        valid.replace("-1 1", "nan 1", 1), valid + "extra\n"]:
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    proposal.read_model(path)

    def test_duplicate_source_columns_and_exact_cancellation_are_proposals_only(self):
        # Both towers are x. The first authored equality deliberately stores
        # -1e16*x - x + 1e16*x separately: naive floating accumulation loses
        # the -x term, while fsum preserves its exact coefficient on this
        # integer fixture. A shared-input paired plane also repeats x with
        # cancelling +/-1/2 coefficients. Four raw terms exceed nv=3 and are
        # valid because column count includes duplicates.
        fixture = """PAIRED_LP_GEQ_V1 3 2 2 1
-1 1
-1 1
-1 1
1 4 0 -10000000000000000 0 -1 0 10000000000000000 1 1 0
2 2 0 -1 2 1 0
4 0 0.5 0 -0.5 1 1 2 -1 0
4 0 -0.5 0 0.5 1 -1 2 1 0
1 2
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicates.lp"
            path.write_text(fixture)
            original_bytes = path.read_bytes()
            model = proposal.read_model(path)
            self.assertEqual(model.equalities[0].columns.tolist(), [0, 1])
            self.assertEqual(model.equalities[0].coefficients.tolist(), [-1.0, 1.0])
            self.assertEqual(model.inequalities[0].columns.tolist(), [1, 2])
            self.assertEqual(model.inequalities[0].coefficients.tolist(), [1.0, -1.0])
            self.assertEqual(model.terms_coalesced_or_zeroed, 6)
            result = proposal.optimize(model, options(), time.monotonic() + 5.0)
            self.assertEqual([c["proposal_lower_bound"] for c in result["candidates"]], [0.0, 0.0])
            self.assertEqual(result["authority"], "numerical_proposal_only")
            self.assertEqual(result["proposal_terms_coalesced_or_zeroed"], 6)
            # Source bytes and original coefficients are never rewritten.
            self.assertEqual(path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
