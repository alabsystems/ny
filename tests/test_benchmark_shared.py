# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import _shared


class RunNyVerifyTests(unittest.TestCase):
    def run_result(
        self, returncode: int, stdout: str, stderr: str = ""
    ) -> _shared.VerificationResult:
        completed = subprocess.CompletedProcess(
            args=["ny"],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
        with patch.object(_shared.subprocess, "run", return_value=completed):
            return _shared.run_ny_verify(
                Path("network.onnx"),
                Path("property.vnnlib"),
                timeout=1,
            )

    def test_accepts_matching_json_status_and_exit_code(self) -> None:
        verified = self.run_result(
            0, json.dumps({"property_status": "safe", "bounds": [[0.0, 1.0]]})
        )
        falsified = self.run_result(1, json.dumps({"status": "violated"}))
        potential = self.run_result(
            2, json.dumps({"status": "potential_violation"})
        )
        timeout = self.run_result(3, json.dumps({"status": "timeout"}))

        self.assertEqual(verified.status, "verified")
        self.assertEqual(verified.bounds, [[0.0, 1.0]])
        self.assertEqual(falsified.status, "falsified")
        self.assertEqual(potential.status, "unknown")
        self.assertEqual(timeout.status, "timeout")

    def test_rejects_json_verdict_exit_code_disagreement(self) -> None:
        result = self.run_result(
            42,
            json.dumps({"property_status": "safe"}),
            "synthetic verifier crash",
        )

        self.assertEqual(result.status, "error")
        self.assertIn("requires exit code 0", result.error_message or "")
        self.assertIn("exit code 42", result.error_message or "")
        self.assertIn("synthetic verifier crash", result.error_message or "")

    def test_rejects_unrecognized_json_status(self) -> None:
        result = self.run_result(0, json.dumps({"status": "finished"}))

        self.assertEqual(result.status, "error")
        self.assertIn("recognized verification status", result.error_message or "")

    def test_text_fallback_requires_exact_status_field(self) -> None:
        accepted = self.run_result(1, "Status: VIOLATED\n")
        rejected = self.run_result(0, "diagnostic: property was not verified\n")

        self.assertEqual(accepted.status, "falsified")
        self.assertEqual(rejected.status, "error")


class DirectInvocationTests(unittest.TestCase):
    def test_measure_clip_reduction_help_works_outside_repo(self) -> None:
        script = REPO_ROOT / "scripts" / "measure_clip_reduction.py"
        with tempfile.TemporaryDirectory() as working_dir:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Measure Clip-and-Verify", result.stdout)


if __name__ == "__main__":
    unittest.main()
