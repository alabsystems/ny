# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""The counterexample oracle's input-domain scan, and its size guard.

``scripts/extended_bank/vnnlib_ce.py`` refused **77 of the 182 nn4sys specs**
before this module existed, for two separate reasons:

* ``cardinality_0_1_128.vnnlib`` -> ``input constraints do not reference X_0``.
  The scan demanded an INPUT-ONLY assertion for every declared ``X_i``.  nn4sys
  states its input interval only inside a mixed disjunct --
  ``(assert (or (and (>= X_0 a) (<= X_0 b) (<= Y_0 c)) ...))`` -- so the file was
  rejected before onnxruntime ever ran.
* ``cardinality_0_1000_128.vnnlib`` -> ``top-level expression exceeds 100000
  tokens``.  A hard refusal, not a knob.

``vnnlib_ce`` VALIDATES SAT WITNESSES, and nn4sys banks 162 unsat / 33 timeout /
zero sat, so no banked row's points ever depended on it.  What the defect cost
was the ABILITY to grade -- the falsification audit could not put a refutation
of those rows in front of the trusted oracle, and no future ``sat`` there could
be banked at all.

These tests hold the repair to a strict line: it must widen what can be
**scanned**, and must not widen by one witness what counts as a
**counterexample**.  The zero-tolerance verdict still comes from evaluating
every assertion at the raw witness and the executed outputs.
"""

# Stdlib unittest so this runs without pytest-only plugins; ruff's pytest
# assertion rules therefore do not apply.
# ruff: noqa: PT009, PT027

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

# HARD imports, never a guarded one. `unittest.SkipTest` on an ImportError is
# exactly the conditionally-vanishing test the source policy prohibits: it
# reports green on a machine where the oracle never ran, which is the opposite
# of what a test asserting an oracle's behaviour is for. The provisioned
# interpreter (`NY_GATE_PYTHON`, requirements.txt) carries all three.
import numpy as np
import onnx
import onnxruntime  # noqa: F401
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "extended_bank"))
vnnlib_ce = importlib.import_module("vnnlib_ce")

NN4SYS_VNNLIB = (
    REPO_ROOT / "benchmarks" / "vnncomp2025" / "benchmarks" / "nn4sys" / "vnnlib"
)

# The nn4sys shape, minimised: two boxes, each stated only inside a mixed arm.
NESTED_DISJUNCT_SPEC = """
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (or
  (and (>= X_0 0.0) (<= X_0 1.0) (>= X_1 0.0) (<= X_1 1.0) (>= Y_0 0.75))
  (and (>= X_0 2.0) (<= X_0 3.0) (>= X_1 2.0) (<= X_1 3.0) (>= Y_0 9.0))
))
"""

# X_1 is bounded in ONE arm only, so the disjunction leaves it free.  This must
# STILL be refused: widening the scan must not turn "unbounded input" into
# "fine".
HALF_BOUNDED_SPEC = """
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (or
  (and (>= X_0 0.0) (<= X_0 1.0) (>= X_1 0.0) (<= X_1 1.0) (>= Y_0 0.75))
  (and (>= X_0 2.0) (<= X_0 3.0) (>= Y_0 9.0))
))
"""

# The classic shape: an input-only assertion per X_i, then a separate output
# assertion.  Nothing about this may change.
INPUT_ONLY_SPEC = """
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= Y_0 0.75))
"""


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class NestedInputDomainScanTests(unittest.TestCase):
    """``_scan_property`` must find an input bound wherever it is stated."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_input_bound_inside_a_disjunct_is_recognised(self) -> None:
        path = _write(self.tmp, "nested.vnnlib", NESTED_DISJUNCT_SPEC)
        summary = vnnlib_ce._scan_property(path)
        self.assertEqual(summary.input_count, 2)
        self.assertEqual(summary.output_count, 1)
        # No input-ONLY assertion exists; the domain lives in the mixed arm.
        self.assertEqual(summary.input_assertion_count, 0)
        self.assertEqual(summary.nested_domain_assertion_count, 1)

    def test_an_input_bounded_in_only_one_arm_is_still_refused(self) -> None:
        """The disjunction of the arms is what constrains the witness.

        An ``X_i`` pinned in one arm and free in another is free overall.  If
        the widened scan took a union instead of an intersection it would
        accept this property and then grade witnesses against an unbounded box.
        """
        path = _write(self.tmp, "half.vnnlib", HALF_BOUNDED_SPEC)
        with self.assertRaisesRegex(
            vnnlib_ce.ValidationError, r"input constraints do not reference X_1"
        ):
            vnnlib_ce._scan_property(path)

    def test_classic_input_only_specs_are_unchanged(self) -> None:
        path = _write(self.tmp, "plain.vnnlib", INPUT_ONLY_SPEC)
        summary = vnnlib_ce._scan_property(path)
        self.assertEqual(summary.input_assertion_count, 2)
        self.assertEqual(summary.nested_domain_assertion_count, 0)

    def test_constrained_inputs_is_a_superset_of_the_old_rule(self) -> None:
        """Whatever a Y-free assertion used to contribute, it still does.

        The old rule marked EVERY ``X_i`` mentioned by a Y-free assertion, even
        inside arithmetic (``(>= (+ X_0 X_1) 3)``).  The widened rule must not
        quietly drop those, or specs that scanned yesterday stop scanning.
        """
        node = [">=", ["+", "X_0", "X_1"], "3"]
        self.assertEqual(vnnlib_ce._constrained_inputs(node), {0, 1})

    def test_negation_contributes_nothing(self) -> None:
        node = ["not", ["and", [">=", "X_0", "0"], ["<=", "Y_0", "1"]]]
        self.assertEqual(vnnlib_ce._constrained_inputs(node), set())


class ExpressionTokenGuardTests(unittest.TestCase):
    """The size guard must be a knob, not a wall -- and still a real guard."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _wide_spec(self, disjuncts: int) -> Path:
        arms = "\n".join(
            f"  (and (>= X_0 {i}.0) (<= X_0 {i}.5) (>= Y_0 0.75))"
            for i in range(disjuncts)
        )
        return _write(
            self.tmp,
            "wide.vnnlib",
            "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n"
            f"(assert (or\n{arms}\n))\n",
        )

    def test_the_guard_still_fires_at_its_configured_bound(self) -> None:
        path = self._wide_spec(200)
        with self.assertRaisesRegex(
            vnnlib_ce.ValidationError, r"top-level expression exceeds 64 tokens"
        ):
            vnnlib_ce._scan_property(path, max_expression_tokens=64)

    def test_raising_the_bound_lets_the_same_spec_scan(self) -> None:
        path = self._wide_spec(200)
        summary = vnnlib_ce._scan_property(path, max_expression_tokens=1_000_000)
        self.assertEqual(summary.input_count, 1)
        self.assertEqual(summary.nested_domain_assertion_count, 1)

    def test_the_refusal_names_the_knob(self) -> None:
        path = self._wide_spec(200)
        with self.assertRaises(vnnlib_ce.ValidationError) as caught:
            vnnlib_ce._scan_property(path, max_expression_tokens=64)
        message = str(caught.exception)
        self.assertIn("NY_VNNLIB_MAX_EXPRESSION_TOKENS", message)
        self.assertIn("--max-expression-tokens", message)

    def test_environment_override_is_read_and_validated(self) -> None:
        env = vnnlib_ce.EXPRESSION_TOKENS_ENV
        previous = os.environ.get(env)
        try:
            os.environ[env] = "12345"
            self.assertEqual(vnnlib_ce.configured_expression_tokens(), 12345)
            for bad in ("nonsense", "0", "-5"):
                os.environ[env] = bad
                with self.assertRaises(vnnlib_ce.ValidationError):
                    vnnlib_ce.configured_expression_tokens()
            del os.environ[env]
            self.assertEqual(
                vnnlib_ce.configured_expression_tokens(),
                vnnlib_ce.DEFAULT_MAX_EXPRESSION_TOKENS,
            )
        finally:
            if previous is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = previous

    def test_a_non_positive_explicit_bound_is_refused(self) -> None:
        with self.assertRaises(vnnlib_ce.ValidationError):
            vnnlib_ce._expression_token_cap(0)


class ZeroToleranceStillRejectsBadWitnessesTests(unittest.TestCase):
    """The point of the repair is to GRADE, never to grade more leniently."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.onnx_path = self._identity_2d(self.tmp / "identity.onnx")
        self.vnnlib_path = _write(self.tmp, "nested.vnnlib", NESTED_DISJUNCT_SPEC)

    @staticmethod
    def _identity_2d(path: Path) -> Path:
        """y_0 = x_0 + x_1, so the two arms are reachable and distinguishable."""
        weight = numpy_helper.from_array(
            np.ones((2, 1), dtype=np.float32), name="weight"
        )
        bias = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="bias")
        node = helper.make_node("Gemm", ["input", "weight", "bias"], ["output"])
        graph = helper.make_graph(
            [node],
            "sum2",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
            [weight, bias],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_operatorsetid("", 13)]
        )
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        return path

    def test_a_genuine_in_box_counterexample_is_accepted(self) -> None:
        """x = (0.5, 0.5) is in arm 1's box and y = 1.0 >= 0.75."""
        in_box, is_counterexample, detail = vnnlib_ce.validate(
            self.onnx_path, self.vnnlib_path, {0: 0.5, 1: 0.5}
        )
        self.assertTrue(in_box, detail)
        self.assertTrue(is_counterexample, detail)

    def test_an_out_of_box_witness_is_rejected(self) -> None:
        """x = (5.0, 5.0) satisfies NO arm's input interval.

        y = 10.0 clears both output thresholds, so a validator that had lost
        its in-box gate to the widened scan would call this a counterexample.
        """
        in_box, is_counterexample, detail = vnnlib_ce.validate(
            self.onnx_path, self.vnnlib_path, {0: 5.0, 1: 5.0}
        )
        self.assertFalse(is_counterexample, detail)
        self.assertFalse(in_box, detail)
        self.assertIn("in_box=False", detail)

    def test_a_witness_that_crosses_arms_is_rejected(self) -> None:
        """x = (0.5, 0.5) with arm 2's threshold: y = 1.0 does not reach 9.0.

        x_0 in arm 1's box and x_1 in arm 1's box means only arm 1 can fire,
        and arm 1 needs y >= 0.75 -- which holds -- so instead take a point
        inside arm 2's box where y misses arm 2's threshold: x = (2.0, 2.0)
        gives y = 4.0 < 9.0, and (2.0, 2.0) is outside arm 1's box.
        """
        in_box, is_counterexample, detail = vnnlib_ce.validate(
            self.onnx_path, self.vnnlib_path, {0: 2.0, 1: 2.0}
        )
        self.assertTrue(in_box, detail)
        self.assertFalse(is_counterexample, detail)

    def test_an_incomplete_witness_is_rejected(self) -> None:
        in_box, is_counterexample, detail = vnnlib_ce.validate(
            self.onnx_path, self.vnnlib_path, {0: 0.5}
        )
        self.assertFalse(in_box)
        self.assertFalse(is_counterexample)
        self.assertIn("incomplete witness", detail)

    def test_the_detail_says_the_domain_gate_is_nested(self) -> None:
        _in_box, _is_ce, detail = vnnlib_ce.validate(
            self.onnx_path, self.vnnlib_path, {0: 0.5, 1: 0.5}
        )
        self.assertIn("nested_domain_assertions=1", detail)
        self.assertIn("projected_domain_results=", detail)


# ---------------------------------------------------------------------------
# The nn4sys corpus lane.
#
# These three contracts need 2.3 GB of checked-out VNN-LIB, which most machines
# do not have. They previously carried `unittest.SkipTest` on the corpus
# directory, which the source policy prohibits and rightly so: a lane that
# quietly vanishes is indistinguishable from a lane that ran and passed, and
# these are the ONLY contracts that pin the repair against the real specs it
# was written for.
#
# They are therefore plain `external_*` functions on the pattern
# `test_vnnlib_ce_tensor_specs.py` established: ordinary `pytest tests/` does
# not collect them at all, and selecting the lane
#
#     pytest tests/test_vnnlib_ce_nested_input_domain.py \
#         --override-ini 'python_functions=external_*'
#
# opts into the COMPLETE contract, including a hard failure when the corpus is
# absent. Not `unittest.TestCase` methods: pytest collects those by the `test`
# prefix through unittest's own collector, which does not honour
# `python_functions`, so there would be no way to select the lane separately.
# ---------------------------------------------------------------------------

SMALL_LIMIT_BYTES = 5_000_000
# The corpus caps are the point of the repair; a spec is scanned under a bound
# large enough to reach it, not under the shipped default that refused it.
CORPUS_CAP = 40_000_000


def _require_corpus() -> list[Path]:
    """Hard-fail, never skip, when the corpus this lane needs is absent."""
    assert NN4SYS_VNNLIB.is_dir(), (
        f"nn4sys corpus is not materialized at {NN4SYS_VNNLIB}; this external "
        f"lane requires it. Materialize the corpus or do not select the lane."
    )
    specs = sorted(NN4SYS_VNNLIB.glob("*.vnnlib"))
    assert specs, f"nn4sys corpus at {NN4SYS_VNNLIB} contains no .vnnlib files"
    return specs


def _require_instance(*paths: Path) -> None:
    for path in paths:
        assert path.is_file(), (
            f"nn4sys instance {path} is missing from an otherwise materialized "
            f"corpus; this external lane names it explicitly."
        )


def external_every_reachable_spec_scans() -> None:
    """Reading the whole family is 2.3 GB.

    The default lane covers every spec up to 5 MB (23 of the 77 refusals, both
    refusal reasons); `NY_VNNLIB_FULL_CORPUS=1` covers all 182.
    """
    specs = _require_corpus()
    full = os.environ.get("NY_VNNLIB_FULL_CORPUS") == "1"
    checked = 0
    nested = 0
    for path in specs:
        if not full and path.stat().st_size > SMALL_LIMIT_BYTES:
            continue
        summary = vnnlib_ce._scan_property(path, max_expression_tokens=CORPUS_CAP)
        assert summary.input_count > 0, path.name
        assert summary.output_count > 0, path.name
        nested += summary.nested_domain_assertion_count > 0
        checked += 1
    assert checked > (100 if full else 20), checked
    # Every cardinality/lindex spec states its domain inside a disjunct; if none
    # did, this contract would be passing for the wrong reason.
    assert nested > 0


def external_the_projection_gate_is_the_only_in_box_gate_on_a_real_row() -> None:
    """cardinality_0_1_128 has ZERO input-only assertions.

    The raw-domain accumulator is therefore EMPTY on it, and a validator with
    only that gate would call every witness in-box vacuously. This pins the real
    behaviour end to end on the shipped corpus: the arm's own declared point is
    in-box, and the same point shifted far outside every arm is REJECTED -- by
    the projection, since nothing else is looking.
    """
    _require_corpus()
    spec_path = NN4SYS_VNNLIB / "cardinality_0_1_128.vnnlib"
    onnx_path = NN4SYS_VNNLIB.parent / "onnx" / "mscn_128d.onnx"
    _require_instance(spec_path, onnx_path)

    summary = vnnlib_ce._scan_property(spec_path)
    assert summary.input_assertion_count == 0
    assert summary.nested_domain_assertion_count == 1

    root = next(e for e in vnnlib_ce._file_expressions(spec_path) if e[0] == "assert")[1]
    assert root[0] == "or"
    values = {
        int(atom[1][2:]): float(atom[2])
        for atom in root[1][1:]
        if atom[0] == ">=" and isinstance(atom[1], str) and atom[1].startswith("X_")
    }
    for index in range(summary.input_count):
        values.setdefault(index, 0.0)

    in_box, _is_ce, detail = vnnlib_ce.validate(onnx_path, spec_path, values)
    assert in_box, detail
    assert "raw_domain_results=[]" in detail
    assert "projected_domain_results=[True]" in detail

    far = {index: value + 1000.0 for index, value in values.items()}
    in_box_far, is_ce_far, detail_far = vnnlib_ce.validate(onnx_path, spec_path, far)
    assert not in_box_far, detail_far
    assert not is_ce_far, detail_far
    assert "projected_domain_results=[False]" in detail_far


def external_a_previously_refused_spec_is_refused_for_no_reason_at_all() -> None:
    """The two named reproducers from the defect report, by name."""
    _require_corpus()
    for name, previous in (
        ("cardinality_0_1_128.vnnlib", "input constraints do not reference X_0"),
        ("cardinality_0_1000_128.vnnlib", "exceeds 100000 tokens"),
    ):
        path = NN4SYS_VNNLIB / name
        _require_instance(path)
        summary = vnnlib_ce._scan_property(path, max_expression_tokens=CORPUS_CAP)
        assert summary.input_count > 0, f"{name} (previously: {previous})"


if __name__ == "__main__":
    # `unittest.main()` runs the hermetic classes only. The `external_*`
    # corpus lane is selected through pytest; see its header above.
    unittest.main()
