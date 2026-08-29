# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""The non-reproduction classifier must compare against NY's OWN deadline.

`ny vnncomp` never runs to the scored budget: `internal_timeout_secs` hands the
verifier `budget - max(5, budget // 20)` seconds and keeps the rest to publish
RESULTS_FILE. On a 20 s row that is 15 s — 75% of the budget — so a genuine
budget-wall timeout lands well below a fixed 85%-of-budget threshold and used to
be reported as `BOUND-REGRESSION(early-give-up)`, i.e. as a phantom repair
target. These tests pin both the mirrored deadline and the direction of the fix.
"""

# ruff: noqa: PT009

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_banked_rows", REPO_ROOT / "scripts" / "check_banked_rows.py"
)
check_banked_rows = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = check_banked_rows
_SPEC.loader.exec_module(check_banked_rows)

PRESET_LOG = f"{check_banked_rows.PRESET_MARKER} vnncomp25/x.yaml"


def _classify(elapsed: float, budget: float) -> str:
    reproduced, cause = check_banked_rows.classify(
        SimpleNamespace(verdict="unsat"), "timeout", PRESET_LOG, 0, elapsed, budget
    )
    assert not reproduced
    return cause


class InternalDeadlineTests(unittest.TestCase):
    def test_mirrors_the_binary_grace_formula(self) -> None:
        # crates/ny-cli/src/commands/vnncomp.rs::internal_timeout_secs
        for budget, expected in (
            (20, 15.0),
            (25, 20.0),
            (30, 25.0),
            (100, 95.0),
            (300, 285.0),
            (1200, 1140.0),
        ):
            with self.subTest(budget=budget):
                self.assertEqual(
                    check_banked_rows.ny_internal_solver_deadline(budget), expected
                )

    def test_tiny_budget_falls_back_to_the_whole_budget(self) -> None:
        # The binary keeps the whole budget rather than a zero/negative timeout.
        self.assertEqual(check_banked_rows.ny_internal_solver_deadline(3), 3.0)


class ShortBudgetClassificationTests(unittest.TestCase):
    def test_safenlp_internal_wall_is_a_budget_wall_not_an_early_give_up(self) -> None:
        # The measured safenlp population: 20 s budget, 15 s internal deadline,
        # observed 15.7-16.3 s wall. Under the old fixed 0.85*budget=17 s rule
        # every one of these was labelled early-give-up.
        for elapsed in (15.7, 15.9, 16.1, 16.3):
            with self.subTest(elapsed=elapsed):
                self.assertEqual(
                    _classify(elapsed, 20), check_banked_rows.CAUSE_BUDGET_WALL
                )

    def test_other_short_budget_categories_are_covered(self) -> None:
        # lsnc_relu 25 s, cora_2024/relusplitter/nn4sys/collins_rul 30 s.
        self.assertEqual(_classify(20.4, 25), check_banked_rows.CAUSE_BUDGET_WALL)
        self.assertEqual(_classify(25.3, 30), check_banked_rows.CAUSE_BUDGET_WALL)

    def test_a_real_early_give_up_is_still_reported(self) -> None:
        # Half of NY's own 15 s deadline: nothing hit a wall here.
        self.assertEqual(_classify(7.5, 20), check_banked_rows.CAUSE_EARLY_GIVE_UP)


class LongBudgetUnchangedTests(unittest.TestCase):
    def test_thresholds_are_unchanged_from_sixty_seconds_up(self) -> None:
        for budget in (60, 100, 150, 300, 600, 1200, 3600):
            with self.subTest(budget=budget):
                deadline = check_banked_rows.ny_internal_solver_deadline(budget)
                self.assertEqual(min(0.85 * budget, 0.95 * deadline), 0.85 * budget)

    def test_long_row_below_the_historical_threshold_stays_early_give_up(self) -> None:
        self.assertEqual(_classify(200.0, 300), check_banked_rows.CAUSE_EARLY_GIVE_UP)
        self.assertEqual(_classify(258.0, 300), check_banked_rows.CAUSE_BUDGET_WALL)

    def test_the_change_can_only_relax_never_tighten(self) -> None:
        for budget in range(2, 4000):
            deadline = check_banked_rows.ny_internal_solver_deadline(budget)
            self.assertLessEqual(min(0.85 * budget, 0.95 * deadline), 0.85 * budget)


if __name__ == "__main__":
    unittest.main()
