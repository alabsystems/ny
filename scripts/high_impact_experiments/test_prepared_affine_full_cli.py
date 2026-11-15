"""Observation controls only; fabricated records cannot authorize a proof."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .evidence import ExperimentError, pin
from .original_hull_full_cli import POSITIVE, prepare
from .prepared_affine_full_cli import (
    NATIVE_STATUSES, OCCURRENCES, ORDER, PREPARED_SOURCE, affine_preservation,
    cohort_summary, declared_pin, original_objective, raw_result_observation,
    require_native_arrival, require_result_format, route_resource_observation,
)
from .prepared_route_full_cli import PROOF_ARITHMETIC, inspect_process


class PreparedAffineCliObservation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ny-prepared-affine-observations-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stamp = "NY_BUILD_STAMP_V1 commit=synthetic stamp=synthetic"
        self.row = {"year": 2026, "source_index": 3,
                    "model": {"path": "/model", "sha256": "model"},
                    "property": {"path": "/property", "sha256": "property"}}
        self.capsule = {"profiles": {"off": {"file": {"path": "/original/off/safenlp_2024.yaml"}}}}
        self.objective = {"target": 0, "classes": [1], "unsafe_relation": "<=", "strict_unsafe": False}
        self.flight = {"schema_version": 3, "budget_secs": 20, "category": "safenlp_2024",
                       "build_id": self.stamp, "ambient_env": {"OMP_NUM_THREADS": "1"}, "events": [
                           {"method": "preset_base", "status": "ran", "reason": "/original/off/safenlp_2024.yaml"},
                           {"method": "prepared_root", "status": "ran", "reason": ""},
                           {"method": "run_complete", "status": "complete", "reason": "unsat", "at_secs": 1.0}]}
        self.process = {"returncode": 0, "timed_out": False, "interrupted": False,
                        "wall_seconds": 1.1, "timeout_seconds": 20}
        self.bundle = [{"kind": "header", "schema": "ny-proof-bundle/v2", "profile": "proof",
                        "lane": "vnncomp_v1", "build": self.stamp, "onnx": {"sha256": "model"},
                        "vnnlib": {"sha256": "property"}},
                       {"kind": "verdict", "source": PREPARED_SOURCE, "required": [1], "min_margin": 0.75},
                       {"kind": "published", "result": "unsat"}]
        self.status("RootProven")

    def status(self, status):
        reason = ('PreparedRouteReceipt { status: ' + status + ', requested_error_lane: "off", '
                  f'proof_arithmetic: "{PROOF_ARITHMETIC}", reason: "original-source-or-root-refused", '
                  'byte_cap: 1048576, work_allowance: 12345, work_reserved: 12000, '
                  'peak_leased_bytes: 4567, class_count: 1, max_expansions: 32 }')
        self.flight["events"][1].update(status="ran" if status in NATIVE_STATUSES else "skipped", reason=reason)

    def write(self, verdict="unsat"):
        (self.root / "solver-result").write_text(verdict + "\n")
        self.flight["events"][-1]["reason"] = verdict
        (self.root / "solver-result.flight.json").write_text(json.dumps(self.flight))
        raw = b"".join(json.dumps(r).encode() + b"\n" for r in self.bundle)
        end = {"kind": "end", "lines": len(self.bundle) + 1, "sha256": hashlib.sha256(raw).hexdigest()}
        (self.root / "solver-result.proof.jsonl").write_bytes(raw + json.dumps(end).encode() + b"\n")

    def observe(self):
        return inspect_process(self.root, self.row, True, self.capsule, self.process, self.stamp)

    def property_row(self, relation="<=", left=0, right=1):
        path = self.root / "original.vnnlib"
        text = ("\n".join(f"(declare-const X_{i} Real)" for i in range(30))
                + "\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n"
                + "\n".join(f"(assert (>= X_{i} -1))\n(assert (<= X_{i} 1))" for i in range(30))
                + f"\n(assert ({relation} Y_{left} Y_{right}))\n")
        path.write_text(text)
        return {**self.row, "property": pin(path)}

    def test_actual_root_and_tree_receipt_observation_requires_original_positive_objective(self):
        for status in ("RootProven", "TreeProven"):
            self.status(status)
            self.write()
            observed = self.observe()
            actual = require_native_arrival(observed, self.row, self.objective)
            self.assertTrue(actual["positive_published"])
            self.assertEqual(actual["status"], status)
            self.assertFalse(observed["independent_arithmetic_replay"])
            self.assertFalse(observed["performance_qualification"])
            self.assertFalse(observed["official_score_credit"])

    def test_native_refusal_is_failure_even_if_legacy_published_unsat(self):
        self.bundle[1]["source"] = "legacy_uncovered"
        for status in ("Refused", "Declined"):
            self.status(status)
            self.write()
            observed = self.observe()
            with self.assertRaisesRegex(ExperimentError, "native arrival refused"):
                require_native_arrival(observed, self.row, self.objective)
            resources = route_resource_observation(observed)
            self.assertEqual(resources["counters"]["work_allowance"], 12345)
            self.assertEqual(resources["counters"]["work_reserved"], 12000)
            self.assertTrue(resources["counters_are_not_measurements"])

    def test_native_open_qualifies_source_but_cannot_replace_any_of_four_positive_arrivals(self):
        self.status("Open")
        self.write("unknown")
        observed = self.observe()
        other = {**self.row, "source_index": 4}
        actual = require_native_arrival(observed, other, self.objective)
        self.assertTrue(actual["native_arrived"])
        self.assertFalse(observed["prepared"]["proof_arrived"])
        with self.assertRaisesRegex(ExperimentError, "stayed Open"):
            require_native_arrival(observed, self.row, self.objective)

    def test_upfront_sat_requires_real_sat_and_cannot_preempt_rational_positive(self):
        self.flight["events"].pop(1)
        self.write("sat")
        observed = self.observe()
        actual = require_native_arrival(observed, {**self.row, "source_index": 4}, self.objective)
        self.assertTrue(actual["upfront_sat"])
        self.assertFalse(actual["native_arrived"])
        with self.assertRaisesRegex(ExperimentError, "positive original did not reach"):
            require_native_arrival(observed, self.row, self.objective)
        self.write("unknown")
        with self.assertRaisesRegex(ExperimentError, "without upfront SAT"):
            self.observe()

    def test_foreign_class_or_fake_legacy_proof_source_cannot_qualify_positive(self):
        self.write()
        observed = self.observe()
        with self.assertRaisesRegex(ExperimentError, "ordered objective"):
            require_native_arrival(observed, self.row, {**self.objective, "target": 1, "classes": [0]})
        self.bundle[1]["source"] = "original_hull_replay"
        self.bundle[1]["required"] = [0]
        self.write()
        with self.assertRaisesRegex(ExperimentError, "source disagree"):
            self.observe()

    def test_original_complete_class_count_cannot_be_partial_or_inflated(self):
        self.write()
        observed = self.observe()
        for classes in ([0, 1], [], [True], ["1"], [-1]):
            with self.assertRaisesRegex(ExperimentError, "invalid complete original"):
                require_native_arrival(observed, self.row, {**self.objective, "classes": classes})
        observed["prepared"]["class_count"] = 2
        with self.assertRaisesRegex(ExperimentError, "class count"):
            require_native_arrival(observed, self.row, self.objective)

    def test_changed_caps_wall_failure_or_cancellation_cannot_qualify_published_bytes(self):
        self.write()
        for delta in ({"wall_seconds": 20.001}, {"timeout_seconds": 21}, {"returncode": 1},
                      {"timed_out": True}, {"interrupted": True}):
            with self.assertRaisesRegex(ExperimentError, "process cannot qualify"):
                inspect_process(self.root, self.row, True, self.capsule, {**self.process, **delta}, self.stamp)

    def test_changed_actual_source_hash_or_unpaid_error_lane_trace_refuses(self):
        self.bundle[0]["onnx"]["sha256"] = "foreign-original"
        self.write()
        with self.assertRaisesRegex(ExperimentError, "input hash"):
            self.observe()
        self.bundle[0]["onnx"]["sha256"] = "model"
        self.flight["events"][1]["reason"] = self.flight["events"][1]["reason"].replace(
            'requested_error_lane: "off"', 'requested_error_lane: "per-col"')
        self.write()
        with self.assertRaisesRegex(ExperimentError, "requested/error arithmetic"):
            self.observe()

    def test_raw_original_output_relation_and_strictness_map_to_complete_class_ids(self):
        for relation, left, right, target, adversary in (
            ("<=", 0, 1, 0, 1), ("<", 1, 0, 1, 0),
            (">=", 0, 1, 1, 0), (">", 1, 0, 0, 1),
        ):
            objective = original_objective(self.property_row(relation, left, right))
            self.assertEqual(objective["target"], target)
            self.assertEqual(objective["classes"], [adversary])
            self.assertEqual(objective["strict_unsafe"], relation in {"<", ">"})
            self.assertEqual(objective["coordinate_map"], "X_j=original-input[0,j]")
            self.assertEqual(objective["authority"], "observation_only")

    def test_original_missing_duplicate_extra_feature_or_multiple_output_scope_refuses(self):
        for changed in (lambda t: t.replace("(declare-const X_29 Real)", ""),
                        lambda t: t + "(declare-const X_0 Real)\n",
                        lambda t: t + "(declare-const X_30 Real)\n",
                        lambda t: t + "(assert (<= Y_1 Y_0))\n",
                        lambda t: t + "(assert (<= (+ Y_0 Y_1) 1))\n"):
            row = self.property_row()
            path = Path(row["property"]["path"])
            path.write_text(changed(path.read_text()))
            row["property"] = pin(path)
            with self.assertRaises(ExperimentError):
                original_objective(row)
        with self.assertRaisesRegex(ExperimentError, "repeats its output"):
            original_objective(self.property_row("<=", 0, 0))

    def test_changed_original_property_bytes_fail_before_objective_observation(self):
        row = self.property_row()
        Path(row["property"]["path"]).write_text("foreign objective")
        with self.assertRaisesRegex(ExperimentError, "property bytes changed"):
            original_objective(row)

    def success_cases(self):
        cases = []
        for year, index in OCCURRENCES:
            actual = {"native_arrived": True, "upfront_sat": False,
                      "positive_published": True if (year, index) in POSITIVE else None}
            arms = [{"arm": arm, "passed": True, "observation": {"verdict": "unsat"},
                     **({"native_arrival": actual} if arm == "candidate_default" else {})}
                    for arm in ORDER]
            cases.append({"year": year, "source_index": index, "arms": arms,
                          "preservation": affine_preservation(arms)})
        return cases

    def test_only_exact26_occurrences78_arms_and_all_four_positive_arrivals_pass(self):
        cases = self.success_cases()
        actual = cohort_summary(cases)
        self.assertTrue(actual["passed"])
        self.assertEqual(actual["solver_processes"], 78)
        self.assertEqual(actual["native_outcomes"], 26)
        self.assertEqual(actual["upfront_sat"], 0)
        for changed in (cases[:-1], [cases[0]] + cases[1:-1] + [cases[0]], list(reversed(cases))):
            self.assertFalse(cohort_summary(changed)["passed"])
        changed = copy.deepcopy(cases)
        changed[0]["source_index"] = True
        self.assertFalse(cohort_summary(changed)["passed"])

    def test_missing_arm_or_actual_arrival_and_any_one_positive_open_cannot_be_hidden(self):
        original = self.success_cases()
        changed = copy.deepcopy(original)
        changed[-1]["arms"].pop()
        self.assertFalse(cohort_summary(changed)["passed"])
        changed = copy.deepcopy(original)
        changed[-1]["arms"][-1] = copy.deepcopy(changed[-1]["arms"][0])
        self.assertFalse(cohort_summary(changed)["passed"])
        changed = copy.deepcopy(original)
        del changed[-1]["arms"][1]["native_arrival"]
        self.assertFalse(cohort_summary(changed)["passed"])
        for year, index in POSITIVE:
            changed = copy.deepcopy(original)
            case = next(c for c in changed if (c["year"], c["source_index"]) == (year, index))
            case["arms"][1]["native_arrival"]["positive_published"] = False
            self.assertFalse(cohort_summary(changed)["passed"])

    def test_failed_arrival_cannot_hide_contrary_settled_original_verdict(self):
        arms = [{"arm": arm, "passed": arm != "candidate_default",
                 "observation": {"verdict": "sat" if arm == "candidate_default" else "unsat"}}
                for arm in ORDER]
        result = affine_preservation(arms)
        self.assertFalse(result["passed"])
        self.assertTrue(result["contrary_decided_verdict"])
        self.assertTrue(any("contrary" in error for error in result["errors"]))

    def test_actual_opposite_result_survives_strict_flight_failure_before_observation_returns(self):
        # Actual final SAT bytes conflict with the claimed prepared proof;
        # strict observation rightly refuses, but the emergency must survive.
        self.write("sat")
        raw = raw_result_observation(self.root)
        self.assertEqual(raw["verdict"], "sat")
        self.assertEqual(raw["authority"], "diagnostic_only")
        with self.assertRaisesRegex(ExperimentError, "complete UNSAT"):
            self.observe()
        arms = [{"arm": "reference_before", "passed": True, "observation": {"verdict": "unsat"}},
                {"arm": "candidate_default", "passed": False, "raw_result": raw,
                 "error": "strict prepared trace refused"}]
        result = affine_preservation(arms)
        self.assertFalse(result["passed"])
        self.assertTrue(result["contrary_decided_verdict"])

    def test_duplicate_bare_result_tokens_and_non_sat_suffix_refuse_but_real_witness_lines_remain(self):
        path = self.root / "solver-result"
        for text in ("unsat\nsat\n", "unknown\nunknown\n", "timeout\nextra\n", "sat\nunsat\n"):
            path.write_text(text)
            record = {"raw_result": raw_result_observation(self.root)}
            with self.assertRaisesRegex(ExperimentError, "trailing bytes|duplicate bare"):
                require_result_format(record)
        self.write()
        path.write_text("unsat\nsat\n")
        # The shared actual inspector now also rejects this malformed body.
        # Raw diagnostic UNSAT survives for the separate soundness stop.
        record = {"raw_result": raw_result_observation(self.root)}
        with self.assertRaisesRegex(ExperimentError, "duplicate bare"):
            self.observe()
        self.assertEqual(record["raw_result"]["verdict"], "unsat")
        with self.assertRaisesRegex(ExperimentError, "duplicate bare"):
            require_result_format(record)
        path.write_text("sat\n(X_0 0.5)\n(Y_0 -0.25)\n")
        record = {"raw_result": raw_result_observation(self.root)}
        self.assertEqual(record["raw_result"]["verdict"], "sat")
        require_result_format(record)
        with self.assertRaisesRegex(ExperimentError, "could not be qualified"):
            require_result_format({"raw_result_error": "missing bytes"})

    def test_launch_pin_must_match_original_declared_bytes_after_identity_probe_interval(self):
        path = self.root / "original-asset"
        path.write_bytes(b"pinned original source/profile/csv")
        original = pin(path)
        self.assertEqual(declared_pin(original), original)
        path.write_bytes(b"changed during identity probes")
        with self.assertRaisesRegex(ExperimentError, "declaration changed before launch"):
            declared_pin(original)
        path.write_bytes(b"pinned original source/profile/csv")
        with self.assertRaisesRegex(ExperimentError, "declaration changed before launch"):
            declared_pin({**original, "size_bytes": original["size_bytes"] + 1})

    def test_mandatory_actual_original_capsule_and_all26_unmodified_objectives(self):
        base = Path("<home>/ny-experiments/high-impact/2026-10-01")
        capsule, rows = prepare(base / "safenlp-hull-strict20s-capsule-v2.json",
                                base / "safenlp-original-rational-triangle-controls-v1.json")
        self.assertEqual([(r["year"], r["source_index"]) for r in rows], OCCURRENCES)
        self.assertEqual(len(rows), 26)
        for row in rows:
            actual = original_objective(row)
            self.assertEqual(actual["input_count"], 30)
            self.assertEqual(len(actual["classes"]), 1)
            self.assertEqual(row["official_timeout_seconds"], 20)
        self.assertEqual(len({r["model"]["sha256"] for r in rows}), 2)
        self.assertIsNotNone(capsule["profiles"]["off"]["file"]["sha256"])


if __name__ == "__main__":
    unittest.main()
