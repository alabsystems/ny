"""Fast adversarial orchestration checks, using synthetic subprocesses only."""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from .cohort import load_cohort
from .evidence import (ExperimentError, archive_directory, digest, encode,
                       immutable_json, pin, pin_sources, strict_json, unchanged)
from .process import run_command
from .receipts import native_ny_observations, telemetry
from .replay import (bundle_shape, expand_argv, official_metadata,
                     validate_command_receipt)
from .runner import environment, run_arm


class Harness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="ny-e2-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model.onnx"
        self.prop = self.root / "property.vnnlib"
        self.model.write_bytes(b"synthetic model, not a neural-network certificate")
        self.prop.write_text("synthetic property")

    def cohort(self, indices=None):
        csv = self.root / "instances.csv"
        csv.write_text("model.onnx,property.vnnlib,1\nmodel.onnx,property.vnnlib,1\nmodel.onnx,property.vnnlib,2\n")
        config = {"kind": "synthetic", "suite": "synthetic-control", "category": "fixture",
                  "instances_csv": str(csv), "input_root": str(self.root)}
        if indices is not None:
            config["source_indices"] = indices
        return load_cohort(config, self.root)

    def test_repeated_occurrences_and_selection_preserve_source_order(self):
        rows, _ = self.cohort([3, 2, 1])
        self.assertEqual([r["source_index"] for r in rows], [1, 2, 3])
        self.assertEqual([r["occurrence"] for r in rows], [1, 2, 1])
        self.assertNotEqual(rows[0]["source_index"], rows[1]["source_index"])

    def test_duplicate_source_index_selection_refuses(self):
        with self.assertRaisesRegex(ExperimentError, "distinct"):
            self.cohort([1, 1])

    def test_frozen_input_mutation_and_symlink_retarget_refuse(self):
        evidence = pin(self.model)
        self.model.write_bytes(b"different bytes")
        with self.assertRaisesRegex(ExperimentError, "changed"):
            unchanged(evidence)
        link = self.root / "input-link"
        link.symlink_to(self.model)
        linked = pin(link)
        link.unlink()
        link.symlink_to(self.prop)
        with self.assertRaisesRegex(ExperimentError, "changed"):
            unchanged(linked)

    def test_archive_is_external_and_o_excl_does_not_replace(self):
        source = self.root / "source"
        source.mkdir()
        with self.assertRaisesRegex(ExperimentError, "outside"):
            archive_directory(source / "archive", source)
        path = self.root / "immutable.json"
        immutable_json(path, {"value": 1})
        with self.assertRaises(FileExistsError):
            immutable_json(path, {"value": 2})
        self.assertEqual(strict_json(path), {"value": 1})

    def request_receipt(self, verdict="unsat"):
        request = {"verdict": verdict, "bindings": {"model_sha256": "a" * 64,
                   "property_sha256": "b" * 64, "artifacts": {"result": "c" * 64},
                   "source_index": 2, "occurrence": 2}}
        receipt = {"schema": "ny_experiment_replay_v1", "replayer": "independent-fixture",
                   "status": "checked", "basis": "original_model_property",
                   "verdict": verdict, "bindings": request["bindings"],
                   "coverage": {"complete": True, "checked_obligations": 4,
                                "remaining_obligations": 0},
                   "arithmetic": {"checked": True, "method": "exact_rational", "checked_operations": 10},
                   "witness": {"input_checked": True, "output_checked": True,
                               "strict_violation": True, "checked_coordinates": 2}}
        return request, receipt

    def test_void_and_zero_coverage_never_qualify(self):
        request, receipt = self.request_receipt()
        receipt["status"] = "void"
        with self.assertRaises(ExperimentError):
            validate_command_receipt(receipt, request, "independent-fixture")
        receipt["status"] = "checked"
        receipt["coverage"]["checked_obligations"] = 0
        with self.assertRaisesRegex(ExperimentError, "coverage"):
            validate_command_receipt(receipt, request, "independent-fixture")

    def test_replay_hash_occurrence_and_arithmetic_are_required(self):
        request, receipt = self.request_receipt()
        bad = copy.deepcopy(receipt)
        bad["bindings"]["occurrence"] = 1
        with self.assertRaisesRegex(ExperimentError, "bind"):
            validate_command_receipt(bad, request, "independent-fixture")
        receipt["arithmetic"]["checked_operations"] = 0
        with self.assertRaisesRegex(ExperimentError, "arithmetic"):
            validate_command_receipt(receipt, request, "independent-fixture")

    def test_checked_attestation_does_not_mint_trusted_certificate(self):
        request, receipt = self.request_receipt()
        observed = validate_command_receipt(receipt, request, "independent-fixture")
        self.assertEqual(observed["status"], "external_replay_attestation")
        self.assertIs(observed["trusted_certificate"], False)
        self.assertIs(observed["official_score_credit"], False)

    def test_sat_requires_strict_positive_witness_receipt(self):
        request, receipt = self.request_receipt("sat")
        receipt["witness"]["strict_violation"] = False
        with self.assertRaisesRegex(ExperimentError, "strict"):
            validate_command_receipt(receipt, request, "independent-fixture")
        receipt["witness"]["strict_violation"] = True
        receipt["witness"]["checked_coordinates"] = 0
        with self.assertRaises(ExperimentError):
            validate_command_receipt(receipt, request, "independent-fixture")

    def test_independent_receipt_must_bind_request_bytes(self):
        request, receipt = self.request_receipt()
        receipt["request_sha256"] = "0" * 64
        with self.assertRaisesRegex(ExperimentError, "request bytes"):
            validate_command_receipt(receipt, request, "independent-fixture", "1" * 64)

    def test_arbitrary_status_json_and_duplicate_keys_refuse(self):
        request, _ = self.request_receipt()
        with self.assertRaises(ExperimentError):
            validate_command_receipt({"status": "pass", "verdict": "unsat"}, request, "fixture")
        path = self.root / "duplicate.json"
        path.write_text('{"status":"void","status":"checked"}')
        with self.assertRaisesRegex(ExperimentError, "duplicate"):
            strict_json(path)

    def test_exact_argv_treats_shell_text_as_literal(self):
        sentinel = self.root / "should-not-exist"
        literal = f"$(touch {sentinel}); `touch {sentinel}`"
        argv = expand_argv(["{binary}", "-c", "import sys;print(sys.argv[1])", "{value}"],
                           {"binary": sys.executable, "value": literal})
        receipt = run_command(argv, cwd=self.root, env=environment({}),
                              stdout=self.root / "out", stderr=self.root / "err", timeout_seconds=2)
        self.assertEqual(receipt["returncode"], 0)
        self.assertEqual((self.root / "out").read_text().strip(), literal)
        self.assertFalse(sentinel.exists())

    def test_timeout_stops_child_group_and_retains_actual_cost(self):
        heartbeat = self.root / "heartbeat"
        child = "import time,sys\nf=open(sys.argv[1],'a',buffering=1)\nwhile True:\n f.write('x');time.sleep(.02)"
        parent = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);time.sleep(30)"
        receipt = run_command([sys.executable, "-c", parent, child, str(heartbeat)],
                              cwd=self.root, env=environment({}), stdout=self.root / "out",
                              stderr=self.root / "err", timeout_seconds=0.3, sample_seconds=0.04)
        self.assertTrue(receipt["timed_out"])
        self.assertLess(receipt["wall_seconds"], 2)
        size = heartbeat.stat().st_size
        time.sleep(0.1)
        self.assertEqual(heartbeat.stat().st_size, size)
        self.assertIsNotNone(receipt["resource_usage"])

    def test_missing_phases_are_unavailable_not_zero(self):
        receipt = telemetry(self.root / "absent.json", 1)
        self.assertEqual(receipt["status"], "unavailable")
        self.assertEqual(receipt["phases"]["loading"]["status"], "unavailable")
        self.assertNotIn("seconds", receipt["phases"]["loading"])

    def test_phase_exclusive_time_and_reuse_scope_receipts_are_checked(self):
        path = self.root / "phases.json"
        value = {"schema": "ny_experiment_phases_v1", "phases": {
            "loading": {"status": "measured", "seconds": 3, "clock": "exclusive_monotonic"}}, "methods": []}
        path.write_bytes(encode(value))
        with self.assertRaisesRegex(ExperimentError, "exceed"):
            telemetry(path, 1)
        method = {"name": "atlas", "admitted": 1, "attempted": 1, "completed": 1,
                  "refused": 0, "skipped": 0, "certified_improvements": 0, "actual_eliminations": 0,
                  "peak_bytes": 64, "lookup_hits": 1, "checked_scope_hits": 2,
                  "coefficient_reads": 4, "seconds": 0.1, "reasons": {}}
        value["phases"] = {}
        value["methods"] = [method]
        path.write_bytes(encode(value))
        with self.assertRaisesRegex(ExperimentError, "counts"):
            telemetry(path, 1)

    def test_native_support_receipt_keeps_actual_and_equivalent_work_distinct(self):
        path = self.root / "log"
        path.write_text("[frozen-dual] class=5 build_ms=120 secs=.4 nodes=2 breakpoint_calls=1 "
                        "support_compiled=1 support_events=2/8 support_reads=3/12 support_build_reads=30 "
                        "support_peak_bytes=128 verdict=open\n"
                        "[frozen-dual] armed=1 objectives=1 secs=.6 closed=0\n")
        result = native_ny_observations(path)
        self.assertEqual(len(result["per_class"]), 1)
        self.assertEqual(len(result["tier_summaries"]), 1)
        self.assertEqual(result["per_class"][0]["counts"]["support_reads"], {"actual": 3, "comparison": 12})
        self.assertIn("not_observed", result["dense_reads_meaning"])

    def test_proof_bundle_shape_never_establishes_arithmetic(self):
        row = {"model": pin(self.model), "property": pin(self.prop)}
        bundle = self.root / "bundle"
        records = [{"kind": "header", "schema": "ny-proof-bundle/v1",
                    "onnx": {"sha256": row["model"]["sha256"]},
                    "vnnlib": {"sha256": row["property"]["sha256"]}},
                   {"kind": "leaf", "margin": 1}, {"kind": "published", "result": "unsat"}]
        bundle.write_bytes(b"".join(encode(r).replace(b"\n", b" ") + b"\n" for r in records))
        shape = bundle_shape(bundle, row)
        self.assertEqual(shape["status"], "shape_only")
        self.assertFalse(shape["arithmetic_checked"])

    def toy_run(self, body: str):
        repo = self.root / "source"
        scripts = repo / "scripts"
        scripts.mkdir(parents=True)
        package = Path(__file__).resolve().parent
        (scripts / "high_impact_experiments").symlink_to(package, target_is_directory=True)
        (scripts / "high_impact_experiments.py").write_bytes((package.parent / "high_impact_experiments.py").read_bytes())
        box = scripts / "ny_box.sh"
        # Deliberately a synthetic box double, never an exposed runner option.
        box.write_text('#!/bin/sh\nshift\n[ "$1" = "--" ] || exit 64\nshift\nexec "$@"\n')
        box.chmod(0o700)
        solver_source = repo / "synthetic.py"
        solver_source.write_text(body)
        rows, _ = self.cohort([1])
        arm = {"name": "synthetic", "role": "baseline", "program": "external", "binary": sys.executable,
               "source_paths": [str(solver_source)], "environment": environment({}),
               "argv": ["{binary}", str(solver_source), "{property}", "{result}"]}
        arm_pin = {"name": arm["name"], "binary": pin(Path(sys.executable)),
                   "profile": None, "sources": pin_sources([solver_source])}
        frozen = {"rows": rows, "arms": [arm_pin], "source_tree": {"head": "synthetic"},
                  "identity_sha256": "d" * 64}
        archive = self.root / "archive"
        archive.mkdir()
        cache = archive / "cache"
        cache.mkdir()
        plan = {"arms": [arm], "sample_seconds": 0.04, "box_wait_allowance_seconds": 2}
        with patch("high_impact_experiments.runner.git_state", return_value={"head": "synthetic"}):
            return run_arm(row=rows[0], arm=arm, arm_pin=arm_pin, plan=plan, frozen=frozen,
                           archive=archive, repo=repo, condition="process_cold", repetition=0,
                           cache=cache, mode="correctness")

    def test_whole_runner_changed_input_invalidates_decided_claim(self):
        result = self.toy_run("import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text('mutated')\nPath(sys.argv[2]).write_text('unsat\\n')\n")
        self.assertEqual(result["status"], "void")
        self.assertIn("frozen file changed", result["reason"])
        self.assertFalse(result["trusted_certificate"])

    def test_whole_runner_unreplayed_unsat_is_unqualified(self):
        result = self.toy_run("import sys\nfrom pathlib import Path\nPath(sys.argv[2]).write_text('unsat\\n')\n")
        self.assertEqual(result["status"], "unqualified")
        self.assertEqual(result["replay"]["status"], "unsupported")
        self.assertFalse(result["official_score_credit"])
        self.assertEqual(result["phase_receipts"]["status"], "unavailable")

    def test_whole_runner_open_result_retains_cost_without_pass_claim(self):
        result = self.toy_run("import sys\nfrom pathlib import Path\nPath(sys.argv[2]).write_text('unknown\\n')\n")
        self.assertEqual(result["status"], "open")
        self.assertEqual(result["post_run_hash_check"], "stable")
        self.assertGreater(result["query_process"]["wall_seconds"], 0)
        self.assertIn("cache", result["query_process"])

    def test_environment_cannot_bypass_measurement_guards(self):
        for value in ({"NY_BOX_LOAD_MAX": "999"}, {"LD_PRELOAD": "/tmp/inject.so"}):
            with self.assertRaises(ExperimentError):
                environment(value)


if __name__ == "__main__":
    unittest.main()
