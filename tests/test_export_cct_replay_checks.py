# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Hermetic receipt-adapter controls; no model or checker is executed."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "export_cct_replay_checks.py"
SPEC = importlib.util.spec_from_file_location("export_cct_replay_checks", SOURCE)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class ReplayExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "pilot" / "r0-row11-candidate"
        self.replay_dir = self.root / "batch" / "replays" / "r0-row11-candidate"
        self.run.mkdir(parents=True)
        self.replay_dir.mkdir(parents=True)
        self.commit = "1" * 40
        self.model = self.payload("original/2.0/onnx/wide.onnx", b"authored model bytes, not imported")
        self.prop = self.payload("original/2.0/vnnlib/wide/p.vnnlib", b"authored property bytes")
        self.image = self.payload("image/ny", b"authored image bytes, not executed")
        self.result = self.payload("pilot/r0-row11-candidate/result.txt", b"sat\n((X_0 0))\n")
        self.witness = self.payload("batch/replays/r0-row11-candidate/assignment.counterexample", b"((X_0 0))\n")
        self.process = {"returncode": 0, "timed_out": False, "interrupted": False,
                        "wall_seconds": 2.25, "timeout_seconds": 30}
        self.start = {
            "schema": "ny_cct_pilot_start_v2", "arm": "candidate",
            "source": {"head": self.commit, "status_sha256": adapter.EMPTY_SHA256,
                       "tracked_diff_sha256": adapter.EMPTY_SHA256},
            "binary": self.image,
            "build_identity": {"fields": {"commit": self.commit, "source": "clean", "dirty": "false",
                                          "exe": self.image["sha256"], "features": "mip,cuda", "profile": "release"}},
            "row": {"index_zero_based": 11, "timeout_seconds": 30, "model": self.model,
                    "property": self.prop, "model_declared": "onnx/wide.onnx", "property_declared": "vnnlib/wide/p.vnnlib"},
        }
        self.start["argv"] = [self.image["path"], "-v", "vnncomp", "v1", "challenging_certified_training_2026",
                              self.model["path"], self.prop["path"], self.result["path"], "30", "--configs-dir", "authored"]
        self.observation = {"arm": "candidate", "row": 11, "status": "completed", "protocol_verdict": "sat",
                            "drained": True, "process": self.process}
        self.write_json(self.run / "start.json", self.start)
        self.write_json(self.run / "observation.json", self.observation)
        self.write_json(self.run / "exec-start.json", {"argv": self.start["argv"]})
        request = {"onnx_path": self.model["path"], "vnnlib_path": self.prop["path"], "vnnlib_version": "2.0",
                   "abs_tolerance": 1e-4, "rel_tolerance": 1e-3, "witness_path": self.witness["path"]}
        self.write_json(self.replay_dir / "request.json", request)
        self.write_json(self.replay_dir / "process.json", self.process)
        sources = [self.payload("checker/" + name, b"authored pinned checker-source stand-in")
                   for name in ("cex_checks.py", "counterexamples.py", "counterexamples_v2.py")]
        self.replay = {
            "schema": "ny_cct_direct_official_sat_replay_v1", "official_score_credit": False,
            "independent_local_strict_sat": True, "original_result": self.result,
            "pilot_start": adapter.pin(self.run / "start.json"),
            "pilot_observation": adapter.pin(self.run / "observation.json"),
            "response": {"ok": True, "provider": "CPUExecutionProvider", "result": "correct",
                         "dependency_versions": {"python": "3.12.3", "onnx": "1.21.0", "onnxruntime": "1.26.0"}},
            "dependencies": {"ready": True, "checker_commit": adapter.CHECKER_COMMIT, "checker_sources": sources},
            "hardware_semantics": "authored receipt fixture; no native execution",
        }
        self.batch_path = self.root / "batch" / "result.json"
        self.csv_path = self.root / "results.csv"
        self.row = ["challenging_certified_training_2026", self.model["path"], self.prop["path"], "0", "sat", "2.25"]
        self.write_csv([self.row])
        self.plan = {"schema": "ny_cct_actual_sat_replay_batch_v2", "adapter_contract": adapter.RESEARCH_ADAPTER,
                     "old_replay_passes_transferred": 0, "source_commits": {"candidate": self.commit},
                     "images": {"candidate": self.image}}
        self.refresh()

    def payload(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return adapter.pin(path)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_csv(self, rows):
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)

    def refresh(self, result="correct"):
        self.replay["response"]["result"] = result
        self.replay["independent_local_strict_sat"] = result == "correct"
        self.write_json(self.replay_dir / "replay.json", self.replay)
        self.write_json(self.replay_dir / "checker.stdout", self.replay["response"])
        self.write_json(self.replay_dir / "dependency-identity.json", self.replay["dependencies"])
        command = {**self.process, "error": None}
        self.batch = {
            "schema": "ny_cct_actual_sat_replay_batch_result_v2", "all_completed_sat_discovered": True,
            "official_score_credit": False,
            "collection": {"adapter_contract": self.plan["adapter_contract"], "old_replay_passes_transferred": 0,
                           "source_commits": {"candidate": self.commit}},
            "adjudications": [{"arm": "candidate", "row": 11, "run_dir": str(self.run), "result": self.result,
                               "command": command, "replay": adapter.pin(self.replay_dir / "replay.json"),
                               "adjudication": result, "strict_correct": result == "correct",
                               "input_tolerance_only": result == "correct_up_to_tolerance"}],
        }
        self.write_json(self.root / "batch" / "plan.json", self.plan)
        self.write_json(self.batch_path, self.batch)

    def export(self, validation_only=False):
        return adapter.export_records(batch_result=self.batch_path, batch_sha256=adapter.pin(self.batch_path)["sha256"],
                                      results_csv=self.csv_path, csv_sha256=adapter.pin(self.csv_path)["sha256"],
                                      source_commit=self.commit, image_sha256=self.image["sha256"],
                                      arm="candidate", validation_only=validation_only)

    def test_complete_receipt_chain_binds_exact_csv_and_all_payload_pins(self):
        record, = self.export()
        normalized = self.row.copy()
        normalized[4] = "violated"
        self.assertEqual(record["results_row_identity"], adapter.row_identity(normalized))
        self.assertTrue(record["establishes_sat"])
        self.assertFalse(record["ny_evidence"]["validation_only"])
        for key in ("model", "property", "image", "original_result", "witness", "pilot_start", "pilot_observation"):
            self.assertEqual(adapter.verify_pin(record["ny_evidence"][key]), record["ny_evidence"][key])

    def test_tolerance_unsupported_missing_and_invalid_are_distinct(self):
        expected = {"correct_up_to_tolerance": ("correct_with_tolerance", True),
                    "unsupported": ("unsupported", False), "no_ce": ("missing", False),
                    "malformed_ce": ("malformed", False), "spec_not_violated": ("incorrect", False)}
        for result, (label, credit) in expected.items():
            with self.subTest(result=result):
                self.refresh(result)
                record, = self.export()
                self.assertEqual((record["classification"], record["score_credit"]), (label, credit))
                self.assertFalse(record["establishes_sat"])
        with self.assertRaises(adapter.EvidenceError):
            adapter.classification("checker_failure_unqualified")

    def test_actual_runtime_prepare_time_and_unique_csv_join_are_required(self):
        for column, value in ((3, "1"), (5, "2.25001"), (1, "onnx/another.onnx")):
            row = self.row.copy()
            row[column] = value
            self.write_csv([row])
            with self.subTest(column=column), self.assertRaises(adapter.EvidenceError):
                self.export()
        self.write_csv([self.row, self.row])
        with self.assertRaises(adapter.EvidenceError):
            self.export()

    def test_payload_or_replay_link_tampering_cannot_borrow_a_strict_grade(self):
        for pinned in (self.model, self.prop, self.image, self.result, self.witness):
            path = Path(pinned["path"])
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=path), self.assertRaises(adapter.EvidenceError):
                self.export()
            path.write_bytes(original)
        self.batch["adjudications"][0]["run_dir"] = str(self.root)
        self.write_json(self.batch_path, self.batch)
        with self.assertRaises(adapter.EvidenceError):
            self.export()

    def test_wrong_source_image_late_or_censored_original_run_refuses(self):
        for section, key, value in (("source", "head", "2" * 40), ("binary", "sha256", "0" * 64)):
            changed = copy.deepcopy(self.start)
            changed[section][key] = value
            with self.subTest(section=section), self.assertRaises(adapter.EvidenceError):
                adapter.original_run(changed, self.observation, "candidate", self.commit, self.image["sha256"])
        for mutation in ({"outer_timeout_censored": True}, {"drained": False}, {"process": {**self.process, "wall_seconds": 31}}):
            changed = {**self.observation, **mutation}
            with self.subTest(mutation=mutation), self.assertRaises(adapter.EvidenceError):
                adapter.original_run(self.start, changed, "candidate", self.commit, self.image["sha256"])

    def test_legacy_batch_can_only_be_exported_as_validation_only(self):
        self.plan["adapter_contract"] = "legacy_f3_adapter_validation_fixture"
        self.refresh()
        with self.assertRaises(adapter.EvidenceError):
            self.export()
        record, = self.export(validation_only=True)
        self.assertTrue(record["ny_evidence"]["validation_only"])

    def test_missing_replay_evidence_is_not_reclassified_as_a_missing_witness(self):
        del self.batch["adjudications"][0]["replay"]
        self.write_json(self.batch_path, self.batch)
        with self.assertRaisesRegex(adapter.EvidenceError, "unclassified"):
            self.export()

    def test_full_row_hash_and_occurrence_keys_preserve_safenlp_subdirectories(self):
        medical = ["safenlp_2024", "/a/onnx/medical/m.onnx", "/a/vnnlib/medical/p.vnnlib", "0", "violated", "1.0"]
        robot = ["safenlp_2024", "/a/onnx/ruarobot/m.onnx", "/a/vnnlib/ruarobot/p.vnnlib", "0", "violated", "1.0"]
        self.assertNotEqual(adapter.row_identity(medical), adapter.row_identity(robot))
        self.assertNotEqual(adapter.occurrence_key(medical), adapter.occurrence_key(robot))
        alias = ["2026_safenlp_2024", "other/onnx/./medical/m.onnx", "other/vnnlib/medical/p.vnnlib", "0", "violated", "1"]
        self.assertEqual(adapter.occurrence_key(medical), adapter.occurrence_key(alias))
        self.assertNotEqual(adapter.row_identity(medical), adapter.row_identity(alias))


if __name__ == "__main__":
    unittest.main()
