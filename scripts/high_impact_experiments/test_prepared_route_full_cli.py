"""Adversarial observation-gate controls; fabricated JSON grants no proof."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from .evidence import ExperimentError, pin
from .prepared_route_full_cli import (
    ORDER, PREPARED_SOURCE, PROOF_ARITHMETIC, append_preservation_arm,
    arrival_controls, inspect_process, load_arrivals, paired_build_contract,
    prepared_trace, preservation, raw_result_observation, require_result_format,
    validate_original_arrival,
)


class PreparedRouteCliObservation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ny-prepared-cli-observations-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stamp = "NY_BUILD_STAMP_V1 commit=synthetic stamp=synthetic"
        self.row = {"model": {"path": "/model", "sha256": "model"},
                    "property": {"path": "/property", "sha256": "property"}}
        self.capsule = {"profiles": {"off": {"file": {"path": "/original/off/safenlp_2024.yaml"}}}}
        self.reason = ('PreparedRouteReceipt { status: RootProven, requested_error_lane: "off", '
                       f'proof_arithmetic: "{PROOF_ARITHMETIC}", class_count: 2 }}')
        self.flight = {"schema_version": 3, "budget_secs": 20, "category": "safenlp_2024",
                       "build_id": self.stamp, "ambient_env": {"OMP_NUM_THREADS": "1"}, "events": [
                           {"method": "preset_base", "status": "ran", "reason": "/original/off/safenlp_2024.yaml"},
                           {"method": "prepared_root", "status": "ran", "reason": self.reason},
                           {"method": "run_complete", "status": "complete", "reason": "unsat", "at_secs": 1.0}]}
        self.process = {"returncode": 0, "timed_out": False, "interrupted": False,
                        "wall_seconds": 1.1, "timeout_seconds": 20}
        self.bundle = [{"kind": "header", "schema": "ny-proof-bundle/v2", "profile": "proof",
                        "lane": "vnncomp_v1", "build": self.stamp, "onnx": {"sha256": "model"},
                        "vnnlib": {"sha256": "property"}},
                       {"kind": "verdict", "source": PREPARED_SOURCE, "required": [1, 0], "min_margin": 0.75},
                       {"kind": "published", "result": "unsat"}]

    def write(self):
        (self.root / "solver-result").write_text("unsat\n")
        path = self.root / "solver-result.flight.json"
        path.write_text(json.dumps(self.flight))
        raw = b"".join(json.dumps(r).encode() + b"\n" for r in self.bundle)
        end = {"kind": "end", "lines": len(self.bundle) + 1, "sha256": hashlib.sha256(raw).hexdigest()}
        (self.root / "solver-result.proof.jsonl").write_bytes(raw + json.dumps(end).encode() + b"\n")
        return path

    def inspect(self, expected=None):
        return inspect_process(self.root, self.row, True, self.capsule, self.process, self.stamp, expected)

    def test_trace_requires_actual_final_source_receipt_and_exact_order(self):
        self.write()
        result = self.inspect({"status": "RootProven", "classes": [1, 0]})
        self.assertEqual(result["final_source"], PREPARED_SOURCE)
        self.assertIsNone(result["legacy_fallback_attribution"])
        self.assertFalse(result["independent_arithmetic_replay"])
        with self.assertRaisesRegex(ExperimentError, "ordered objective"):
            self.inspect({"status": "RootProven", "classes": [0, 1]})
        self.bundle[1]["source"] = "legacy_uncovered"
        self.write()
        with self.assertRaisesRegex(ExperimentError, "source disagree"):
            self.inspect()

    def test_empty_partial_duplicate_nonfinite_and_nonpositive_class_receipts_refuse(self):
        for required, margin in [([], 1.0), ([1], 1.0), ([1, 1], 1.0), ([True, 0], 1.0),
                                 ([1, 0], 0.0), ([1, 0], -0.1), ([1, 0], "0.75")]:
            self.bundle[1].update(required=required, min_margin=margin)
            self.write()
            with self.assertRaisesRegex(ExperimentError, "complete original classes"):
                self.inspect()

    def test_late_aborted_changed_cap_and_failed_processes_cannot_qualify_bytes(self):
        self.write()
        for changed in [{"timed_out": True}, {"interrupted": True}, {"wall_seconds": 20.001},
                        {"returncode": 1}, {"timeout_seconds": 21}]:
            process = {**self.process, **changed}
            with self.assertRaisesRegex(ExperimentError, "process cannot qualify"):
                inspect_process(self.root, self.row, True, self.capsule, process, self.stamp)

    def test_foreign_original_hash_and_corrupt_final_digest_refuse(self):
        self.bundle[0]["onnx"]["sha256"] = "other-original"
        self.write()
        with self.assertRaisesRegex(ExperimentError, "input hash"):
            self.inspect()
        self.bundle[0]["onnx"]["sha256"] = "model"
        self.write()
        path = self.root / "solver-result.proof.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"model"', b'"other"'))
        with self.assertRaisesRegex(ExperimentError, "digest"):
            self.inspect()

    def test_reference_cannot_contain_new_entry_and_fallback_is_attributed_honestly(self):
        path = self.write()
        with self.assertRaisesRegex(ExperimentError, "reference already contains"):
            prepared_trace(path, False, "unsat")
        self.flight["events"][1].update(status="skipped", reason=self.reason.replace("RootProven", "Refused"))
        self.bundle[1]["source"] = "authorized"
        self.write()
        result = self.inspect()
        self.assertFalse(result["prepared"]["proof_arrived"])
        self.assertEqual(result["legacy_fallback_attribution"], "authorized")
        with self.assertRaisesRegex(ExperimentError, "positive arrival is absent"):
            self.inspect({"status": "RootProven", "classes": [1, 0]})

    def test_silent_missing_entry_only_prior_sat_can_preempt(self):
        self.flight["events"] = [self.flight["events"][0], self.flight["events"][-1]]
        path = self.write()
        with self.assertRaisesRegex(ExperimentError, "without upfront SAT"):
            prepared_trace(path, True, "unknown")
        self.assertEqual(prepared_trace(path, True, "sat")["attribution"], "preempted_by_prior_sat")

    def test_default_policy_renamed_or_inconsistent_flight_receipt_refuses(self):
        for reason in [self.reason.replace('"off"', '"envelope"'),
                       self.reason.replace(PROOF_ARITHMETIC, "nominal-only"),
                       self.reason.replace("RootProven", "Refused")]:
            self.flight["events"][1]["reason"] = reason
            path = self.write()
            with self.assertRaises(ExperimentError):
                prepared_trace(path, True, "unsat")
        self.flight["events"].append(copy.deepcopy(self.flight["events"][1]))
        with self.assertRaisesRegex(ExperimentError, "exactly once"):
            prepared_trace(self.write(), True, "unsat")

    def test_banked_verdict_loss_fails_and_sat_unsat_flip_stops(self):
        def records(tokens):
            return [{"arm": arm, "passed": True, "observation": {"verdict": token}}
                    for arm, token in zip(ORDER, tokens)]
        self.assertTrue(preservation(records(["unknown", "unsat", "unknown"]))["passed"])
        self.assertFalse(preservation(records(["unsat", "unknown", "unsat"]))["passed"])
        flip = preservation(records(["sat", "unsat", "sat"]))
        self.assertFalse(flip["passed"])
        self.assertTrue(flip["contrary_decided_verdict"])
        self.assertFalse(preservation(records(["sat", "sat", "sat"])[:2])["passed"])

    def test_serialized_arrival_controls_have_distinct_root_tree_and_bound_original_pins(self):
        rows = arrival_controls(self.root / "arrivals")
        self.assertEqual([r["expected"]["status"] for r in rows], ["RootProven", "TreeProven"])
        self.assertNotEqual(rows[0]["model"]["sha256"], rows[1]["model"]["sha256"])
        self.assertTrue(all(r["timeout_seconds"] == 20 for r in rows))
        self.assertTrue(all(r["origin"] == "new_serialized_integration_control_not_competition_instance" for r in rows))
        path = self.root / "arrivals" / "manifest.json"
        digest = pin(path)["sha256"]
        loaded = load_arrivals(path, digest)
        self.assertEqual(len(loaded), 2)
        with self.assertRaisesRegex(ExperimentError, "manifest changed"):
            load_arrivals(path, "wrong")
        Path(rows[0]["model"]["path"]).write_bytes(b"foreign-model")
        with self.assertRaisesRegex(ExperimentError, "asset pin changed"):
            load_arrivals(path, digest)

    def test_arrival_manifest_cannot_change_original_caps_or_suppress_origin(self):
        arrival_controls(self.root / "arrivals")
        path = self.root / "arrivals" / "manifest.json"
        original = json.loads(path.read_text())
        for key, changed in [("timeout_seconds", 21), ("origin", "competition_score_credit")]:
            manifest = copy.deepcopy(original)
            manifest["rows"][0][key] = changed
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentError, "origin and20s cap"):
                load_arrivals(path, pin(path)["sha256"])

    def test_manifest_bool_integer_equality_negative_duplicate_and_text_ids_refuse(self):
        arrival_controls(self.root / "arrivals")
        path = self.root / "arrivals" / "manifest.json"
        original = json.loads(path.read_text())
        for classes in [[True, 0], [-1, 0], [1, 1], ["1", 0], []]:
            manifest = copy.deepcopy(original)
            manifest["rows"][0]["expected"]["classes"] = classes
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentError, "actual root/tree proof and classes"):
                load_arrivals(path, pin(path)["sha256"])
        self.write()
        with self.assertRaisesRegex(ExperimentError, "invalid original class IDs"):
            self.inspect({"status": "RootProven", "classes": [True, 0]})

    def test_original_arrival_manifest_requires_exact_csv_bytes_occurrence_and_cap(self):
        rows = arrival_controls(self.root / "arrivals")
        row = rows[0]
        csv_path = self.root / "arrivals" / "instances.csv"
        model = Path(row["model"]["path"]).name
        property_file = Path(row["property"]["path"]).name
        csv_path.write_text(f"{model},{property_file},20\n")
        row.update(origin="original_competition_instance", source_csv=pin(csv_path), source_index=1)
        manifest = self.root / "supplied.json"
        def write():
            manifest.write_text(json.dumps({"schema": "ny.prepared-cli-arrivals.v1", "rows": [row]}))
            return pin(manifest)["sha256"]
        self.assertEqual(len(load_arrivals(manifest, write())), 1)
        source_pins = validate_original_arrival(row)
        self.assertEqual([p["path"] for p in source_pins],
                         [str(csv_path), row["model"]["path"], row["property"]["path"]])
        row["source_index"] = True
        with self.assertRaisesRegex(ExperimentError, "pinned official CSV occurrence"):
            load_arrivals(manifest, write())
        row["source_index"] = 1
        csv_path.write_text(f"{model},{property_file},100\n")
        row["source_csv"] = pin(csv_path)
        with self.assertRaisesRegex(ExperimentError, "official cap"):
            load_arrivals(manifest, write())
        csv_path.write_text(f"{model},{property_file},20\n")
        row["source_csv"] = pin(csv_path)
        foreign = self.root / "arrivals" / "foreign.onnx"
        foreign.write_bytes(b"foreign original")
        row["model"] = pin(foreign)
        with self.assertRaisesRegex(ExperimentError, "original CSV model/property bytes"):
            load_arrivals(manifest, write())

    def test_distinct_frozen_source_binary_and_identical_build_treatment_are_required(self):
        first = {"commit": "candidate", "binary_pin": {"sha256": "candidate-exe"},
                 "identity": {"features": "mip", "features_all": "mip,ny-mip", "profile": "debug",
                              "target": "same-target", "backend": "wgpu"}}
        second = copy.deepcopy(first)
        second.update(commit="reference", binary_pin={"sha256": "reference-exe"})
        paired_build_contract(first, second)
        for replacement in [{"commit": "candidate"}, {"binary_pin": {"sha256": "candidate-exe"}}]:
            with self.assertRaisesRegex(ExperimentError, "distinct frozen"):
                paired_build_contract(first, {**second, **replacement})
        for key in first["identity"]:
            changed = copy.deepcopy(second)
            changed["identity"][key] = "changed-treatment"
            with self.assertRaisesRegex(ExperimentError, "build treatment differs"):
                paired_build_contract(first, changed)

    def test_actual_strict_inspector_rejects_duplicate_tokens_and_non_sat_suffix(self):
        self.write()
        path = self.root / "solver-result"
        for text in ("unsat\nsat\n", "unsat\nextra\n", "unsat\n\n"):
            path.write_text(text)
            raw = raw_result_observation(self.root)
            self.assertEqual(raw["verdict"], "unsat")
            self.assertEqual(raw["authority"], "diagnostic_only")
            with self.assertRaisesRegex(ExperimentError, "duplicate bare|trailing bytes"):
                self.inspect()
            with self.assertRaisesRegex(ExperimentError, "duplicate bare|trailing bytes"):
                require_result_format({"raw_result": raw})
        path.write_text("sat\nunsat\n")
        with self.assertRaisesRegex(ExperimentError, "duplicate bare"):
            self.inspect()
        path.write_text("sat\n(X_0 0.5)\n(Y_0 -0.25)\n")
        # A real SAT witness body is preserved; full qualification still
        # requires the unchanged flight/identity/cap checks and SAT arrival.
        require_result_format({"raw_result": raw_result_observation(self.root)})

    def test_actual_opposite_token_survives_strict_flight_failure(self):
        self.write()
        (self.root / "solver-result").write_text("sat\n")
        raw = raw_result_observation(self.root)
        with self.assertRaises(ExperimentError):
            self.inspect()
        records = [
            {"arm": "reference_before", "passed": True, "observation": {"verdict": "unsat"}},
            {"arm": "candidate_default", "passed": False, "raw_result": raw,
             "error": "strict actual flight refused"},
        ]
        actual = preservation(records)
        self.assertFalse(actual["passed"])
        self.assertTrue(actual["contrary_decided_verdict"])
        self.assertTrue(any("contrary" in error for error in actual["errors"]))

    def test_actual_failed_second_arm_stops_before_third_solver_arm(self):
        self.write()
        (self.root / "solver-result").write_text("sat\n")
        raw = raw_result_observation(self.root)
        with self.assertRaises(ExperimentError):
            self.inspect()
        records, started = [], []
        with self.assertRaisesRegex(ExperimentError, "soundness audit"):
            for arm in ORDER:
                # This is the SAME append/stop used immediately after each
                # real process in main. Failed inspection keeps actual SAT
                # diagnostics but cannot qualify the candidate arm.
                started.append(arm)
                record = ({"arm": arm, "passed": False, "raw_result": raw}
                          if arm == "candidate_default" else
                          {"arm": arm, "passed": True, "observation": {"verdict": "unsat"}})
                append_preservation_arm(records, record)
        self.assertEqual(started, ["reference_before", "candidate_default"])
        self.assertEqual(len(records), 2)
        self.assertFalse(records[1]["passed"])

    def test_failed_arm_and_raw_diagnostics_never_qualify_but_keep_contrary_stop(self):
        records = [
            {"arm": arm, "passed": arm != "candidate_default",
             "raw_result": {"verdict": "sat" if arm == "candidate_default" else "unsat",
                            "authority": "diagnostic_only"},
             **({"observation": {"verdict": "unsat"}} if arm != "candidate_default" else {})}
            for arm in ORDER
        ]
        actual = preservation(records)
        self.assertFalse(actual["passed"])
        self.assertTrue(actual["contrary_decided_verdict"])
        for record in records:
            record["raw_result"]["verdict"] = "unsat"
        actual = preservation(records)
        self.assertFalse(actual["passed"])
        self.assertFalse(actual["contrary_decided_verdict"])

    def test_raw_result_diagnostic_refuses_missing_symlink_oversize_and_unknown_tokens(self):
        path = self.root / "solver-result"
        with self.assertRaisesRegex(ExperimentError, "missing/unbounded"):
            raw_result_observation(self.root)
        target = self.root / "outside-result"
        target.write_text("unsat\n")
        path.symlink_to(target)
        with self.assertRaisesRegex(ExperimentError, "missing/unbounded"):
            raw_result_observation(self.root)
        path.unlink()
        path.write_bytes(b"x" * ((1 << 20) + 1))
        with self.assertRaisesRegex(ExperimentError, "missing/unbounded"):
            raw_result_observation(self.root)
        path.write_text("error\n")
        with self.assertRaisesRegex(ExperimentError, "invalid raw"):
            raw_result_observation(self.root)

    def test_actual_bounded_result_read_rejects_growth_after_fd_admission(self):
        path = self.root / "solver-result"
        path.write_text("sat\n")
        requested = []
        actual_fdopen = os.fdopen

        class GrowingReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def fileno(self):
                return self.stream.fileno()

            def read(self, size):
                requested.append(size)
                # Replace no proof/witness authority: grow the real result
                # AFTER its opened-FD admission and BEFORE its actual read.
                with path.open("ab") as writer:
                    writer.write(b"x" * ((1 << 20) + 1))
                return self.stream.read(size)

        def fdopen(descriptor, mode):
            return GrowingReader(actual_fdopen(descriptor, mode))

        with mock.patch("os.fdopen", side_effect=fdopen):
            with self.assertRaisesRegex(ExperimentError, "unbounded|changed"):
                raw_result_observation(self.root)
        self.assertEqual(requested, [(1 << 20) + 1])


if __name__ == "__main__":
    unittest.main()
