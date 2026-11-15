"""Adversarial full-CLI gate parsing controls; synthetic records grant no proof."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .evidence import ExperimentError
from .original_hull_full_cli import (
    ORDER, clean_environment, command_for, inspect_bundle, inspect_case,
    inspect_flight, preservation,
)


class FullCliGate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ny-hull-full-cli-gate-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stamp = "NY_BUILD_STAMP_V1 commit=synthetic stamp=synthetic"
        self.row = {"year": 2026, "source_index": 3, "occurrence": 1,
                    "model": {"path": "/original/model.onnx", "sha256": "model-hash"},
                    "property": {"path": "/original/property.vnnlib", "sha256": "property-hash"}}
        self.profiles = {mode: {"file": {"path": "/profiles/" + mode + "/safenlp_2024.yaml"},
                               "configs_root": "/profiles/" + mode} for mode in ("root", "off")}
        self.capsule = {"profiles": self.profiles}
        self.flight = {
            "schema_version": 3, "budget_secs": 20, "category": "safenlp_2024",
            "build_id": self.stamp, "ambient_env": {"OMP_NUM_THREADS": "1"},
            "events": [
                {"method": "preset_base", "status": "ran", "reason": self.profiles["root"]["file"]["path"]},
                {"method": "hull_resource", "status": "ran",
                 "reason": "outcome=verified origin=original-replay census=participating-hull-only"},
                {"method": "run_complete", "status": "complete", "reason": "unsat", "at_secs": 1.0},
            ],
        }
        self.process = {"returncode": 0, "timed_out": False, "interrupted": False,
                        "wall_seconds": 1.5, "timeout_seconds": 20}
        self.bundle = [
            {"kind": "header", "schema": "ny-proof-bundle/v2", "profile": "proof", "lane": "vnncomp_v1",
             "build": self.stamp, "onnx": {"sha256": "model-hash"}, "vnnlib": {"sha256": "property-hash"}},
            {"kind": "verdict", "source": "original_hull_replay", "required": [0], "min_margin": 0.1, "nodes": 0},
            {"kind": "published", "result": "unsat"},
        ]

    def write_bundle(self, records=None):
        records = self.bundle if records is None else records
        raw = b"".join(json.dumps(record).encode() + b"\n" for record in records)
        end = {"kind": "end", "lines": len(records) + 1, "sha256": hashlib.sha256(raw).hexdigest()}
        path = self.root / "solver-result.proof.jsonl"
        path.write_bytes(raw + json.dumps(end).encode() + b"\n")
        return path

    def write_flight(self):
        path = self.root / "solver-result.flight.json"
        path.write_text(json.dumps(self.flight))
        return path

    def test_full_cli_literal_argv_keeps_original_native_treatment(self):
        self.row["argv_templates"] = {
            mode: ["__FROZEN_NY_BINARY__", "vnncomp", "v1", "safenlp_2024",
                   self.row["model"]["path"], self.row["property"]["path"], "__EXCLUSIVE_RESULT_PATH__", "20",
                   "--configs-dir", self.profiles[mode]["configs_root"], "--proof-profile", "proof"]
            for mode in ("root", "off")}
        command = command_for(self.row, "root", Path("/frozen/binary"), Path("/new/result"), self.capsule)
        self.assertEqual(command[0], "/frozen/binary")
        self.assertEqual(command[1:4], ["vnncomp", "v1", "safenlp_2024"])
        self.assertEqual(command[7], "20")
        self.assertFalse(any(key.startswith("NY_") for key in clean_environment()))
        self.row["argv_templates"]["root"].extend(["--skip-shape-inference"])
        with self.assertRaisesRegex(ExperimentError, "argv changed"):
            command_for(self.row, "root", Path("/frozen/binary"), Path("/new/result"), self.capsule)

    def test_full_cli_final_bundle_checks_original_hashes_and_integrity(self):
        path = self.write_bundle()
        result = inspect_bundle(path, self.row, self.stamp)
        self.assertEqual(result["source"], "original_hull_replay")
        self.assertFalse(result["arithmetic_replayed_by_driver"])
        path.write_bytes(path.read_bytes().replace(b'"model-hash"', b'"alien-hash"'))
        with self.assertRaisesRegex(ExperimentError, "digest"):
            inspect_bundle(path, self.row, self.stamp)
        forged = copy.deepcopy(self.bundle)
        forged[0]["onnx"]["sha256"] = "alien-hash"
        self.write_bundle(forged)
        with self.assertRaisesRegex(ExperimentError, "input hash"):
            inspect_bundle(path, self.row, self.stamp)
        for margin in (-0.0, 0.0, -0.1, "0.1", True):
            forged = copy.deepcopy(self.bundle)
            forged[1]["min_margin"] = margin
            self.write_bundle(forged)
            with self.assertRaisesRegex(ExperimentError, "positive complete"):
                inspect_bundle(path, self.row, self.stamp)

    def test_full_cli_trace_cannot_replace_actual_receipt_or_missing_bundle(self):
        (self.root / "solver-result").write_text("unsat\n")
        self.write_flight()
        with self.assertRaisesRegex(ExperimentError, "missing.*bundle"):
            inspect_case(self.root, "root", self.row, self.capsule, self.process, self.stamp)
        forged = copy.deepcopy(self.bundle)
        forged[1]["source"] = "legacy_uncovered"
        self.write_bundle(forged)
        with self.assertRaisesRegex(ExperimentError, "source disagree"):
            inspect_case(self.root, "root", self.row, self.capsule, self.process, self.stamp)
        self.write_bundle()
        observation = inspect_case(self.root, "root", self.row, self.capsule, self.process, self.stamp)
        self.assertTrue(observation["positive_original_replay"])
        self.assertFalse(observation["official_score_credit"])

    def test_full_cli_late_killed_and_incomplete_processes_refuse_published_bytes(self):
        (self.root / "solver-result").write_text("unsat\n")
        self.write_flight()
        self.write_bundle()
        for changed in ({"timed_out": True}, {"interrupted": True}, {"wall_seconds": 20.01},
                        {"returncode": 1}, {"timeout_seconds": 21}):
            process = {**self.process, **changed}
            with self.assertRaisesRegex(ExperimentError, "process cannot qualify"):
                inspect_case(self.root, "root", self.row, self.capsule, process, self.stamp)

    def test_full_cli_typed_arrival_refuses_env_stale_stamp_and_foreign_preset(self):
        path = self.write_flight()
        inspect_flight(path, "root", Path(self.profiles["root"]["file"]["path"]), self.stamp, "unsat")
        for key, value in (("ambient_env", {"NY_SINGLE_HIDDEN_HULL": "root"}),
                           ("build_id", "other build"), ("budget_secs", 21),
                           ("record_status", {"status": "partial_hard_stop"})):
            saved = copy.deepcopy(self.flight)
            self.flight[key] = value
            self.write_flight()
            with self.assertRaises(ExperimentError):
                inspect_flight(path, "root", Path(self.profiles["root"]["file"]["path"]), self.stamp, "unsat")
            self.flight = saved
        self.write_flight()
        with self.assertRaisesRegex(ExperimentError, "base preset"):
            inspect_flight(path, "root", Path("/foreign/safenlp_2024.yaml"), self.stamp, "unsat")

    def test_full_cli_off_cannot_gain_original_replay_authority(self):
        self.flight["events"][0]["reason"] = self.profiles["off"]["file"]["path"]
        path = self.write_flight()
        with self.assertRaisesRegex(ExperimentError, "OFF acquired"):
            inspect_flight(path, "off_before", Path(self.profiles["off"]["file"]["path"]), self.stamp, "unsat")

    def test_full_cli_preserves_settled_off_and_stops_contrary_decisions(self):
        def records(tokens):
            return [{"arm": arm, "passed": True, "observation": {"verdict": token}}
                    for arm, token in zip(ORDER, tokens)]
        self.assertTrue(preservation(records(["unknown", "unsat", "unknown"]))["passed"])
        self.assertTrue(preservation(records(["sat", "sat", "sat"]))["passed"])
        self.assertFalse(preservation(records(["unsat", "unknown", "unsat"]))["passed"])
        contrary = preservation(records(["sat", "unsat", "sat"]))
        self.assertFalse(contrary["passed"])
        self.assertTrue(contrary["contrary_decided_verdict"])
        self.assertFalse(preservation(records(["unknown", "unknown", "unknown"])[:2])["passed"])

    def test_full_cli_independent_positive_control_cannot_pass_as_open(self):
        (self.root / "solver-result").write_text("unknown\n")
        self.flight["events"] = [self.flight["events"][0], self.flight["events"][-1]]
        self.flight["events"][-1]["reason"] = "unknown"
        self.write_flight()
        with self.assertRaisesRegex(ExperimentError, "independently positive"):
            inspect_case(self.root, "root", self.row, self.capsule, self.process, self.stamp)


if __name__ == "__main__":
    unittest.main()
