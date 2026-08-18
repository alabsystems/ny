"""Stdlib-only tests for the portable extended-bank validator."""

# Ruff's pytest-style assertion rules do not apply: this is intentionally a
# stdlib unittest module so it can run without pytest or benchmark dependencies.
# ruff: noqa: PT009, PT027, A002

from __future__ import annotations

import builtins
import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "extended_bank"
sys.path.insert(0, str(SCRIPT_DIR))
validate_bank = importlib.import_module("validate_bank")
vnnlib_ce = importlib.import_module("vnnlib_ce")


class SourceSchemaTests(unittest.TestCase):
    def test_legacy_five_column_row_uses_fourth_column_as_verdict(self) -> None:
        result = validate_bank.parse_source_row(
            ["track", "onnx/a.onnx", "vnnlib/a.vnnlib", "SAT", "1.25"], 7
        )
        self.assertEqual(result.schema, "extended_bank_v1")
        self.assertEqual(result.verdict, "sat")
        self.assertEqual(result.seconds, "1.25")

    def test_measured_rows_use_fifth_column_as_verdict(self) -> None:
        for prepared, row_length in (("prepared", 6), ("0", 7)):
            with self.subTest(prepared=prepared, row_length=row_length):
                row = [
                    "track",
                    "onnx/a.onnx",
                    "vnnlib/a.vnnlib",
                    prepared,
                    "UNSAT",
                    "2",
                ]
                if row_length == 7:
                    row.append("sealed-run")
                result = validate_bank.parse_source_row(row, 3)
                self.assertEqual(result.verdict, "unsat")
                self.assertNotEqual(result.verdict, prepared)
                self.assertEqual(
                    result.run_id, "sealed-run" if row_length == 7 else None
                )

    def test_truncated_measured_row_cannot_bank_prepared_as_verdict(self) -> None:
        with self.assertRaisesRegex(
            validate_bank.BankValidationError, "prepared flag appears in the verdict"
        ):
            validate_bank.parse_source_row(
                [
                    "track",
                    "onnx/a.onnx",
                    "vnnlib/a.vnnlib",
                    "prepared",
                    "sat",
                ],
                11,
            )

    def test_solved_duplicate_replaces_unknown(self) -> None:
        rows = [
            validate_bank.parse_source_row(
                ["track", "a.onnx", "a.vnnlib", "unknown", "1"], 1
            ),
            validate_bank.parse_source_row(
                ["track", "a.onnx", "a.vnnlib", "unsat", "2"], 2
            ),
            validate_bank.parse_source_row(
                ["track", "a.onnx", "a.vnnlib", "unsat", "3"], 3
            ),
        ]
        selected = validate_bank.select_best(rows)[("a.onnx", "a.vnnlib")]
        self.assertEqual(selected.verdict, "unsat")
        self.assertEqual(selected.seconds, "2")

    def test_conflicting_solved_duplicates_are_fatal(self) -> None:
        rows = [
            validate_bank.parse_source_row(
                ["track", "a.onnx", "a.vnnlib", "sat", "1"], 1
            ),
            validate_bank.parse_source_row(
                ["track", "a.onnx", "a.vnnlib", "unsat", "2"], 2
            ),
        ]
        with self.assertRaisesRegex(
            validate_bank.BankValidationError, "conflicting solved verdicts"
        ):
            validate_bank.select_best(rows)

    def test_prepared_verdict_and_seconds_fields_are_strict(self) -> None:
        bad_rows = [
            (
                ["track", "a.onnx", "a.vnnlib", "1", "sat", "1"],
                "unsupported prepared flag",
            ),
            (
                ["track", "a.onnx", "a.vnnlib", "maybe", "1"],
                "unsupported verdict",
            ),
            (
                ["track", "a.onnx", "a.vnnlib", "sat", "nan"],
                "strict ASCII numeric syntax",
            ),
            (
                ["track", "a.onnx", "a.vnnlib", "sat", "-1"],
                "finite and nonnegative",
            ),
            (
                ["track", "a.onnx", "a.vnnlib", "sat", ""],
                "must be nonempty",
            ),
            (
                ["track", "a.onnx", "a.vnnlib", "sat", "1_0"],
                "strict ASCII numeric syntax",
            ),
            (
                ["track", "a.onnx", "a.vnnlib", "sat", "\N{ARABIC-INDIC DIGIT ONE}"],
                "strict ASCII numeric syntax",
            ),
            (
                ["track", "a.onnx", "a.vnnlib", "sat", "1e999"],
                "finite and nonnegative",
            ),
        ]
        for row, message in bad_rows:
            with self.subTest(row=row):
                with self.assertRaisesRegex(validate_bank.BankValidationError, message):
                    validate_bank.parse_source_row(row, 4)

    def test_mixed_headerless_schemas_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "mixed.csv"
            source.write_text(
                "track,a.onnx,a.vnnlib,unknown,1\n"
                "track,b.onnx,b.vnnlib,0,unknown,2,run\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                validate_bank.BankValidationError, "mixed headerless CSV schemas"
            ):
                validate_bank.load_source_results(source, "track")

    def test_exact_supported_headers_bind_the_row_schema(self) -> None:
        cases = (
            (
                "track,onnx,vnnlib,verdict,seconds\n",
                "track,a.onnx,a.vnnlib,unknown,1\n",
                "extended_bank_v1",
            ),
            (
                "track,onnx,vnnlib,prepared,verdict,seconds\n",
                "track,a.onnx,a.vnnlib,0,unknown,1\n",
                "measured_v1",
            ),
            (
                "track,onnx,vnnlib,prepared,verdict,seconds,run_id\n",
                "track,a.onnx,a.vnnlib,0,unknown,1,run\n",
                "measured_v2",
            ),
            (
                "category,model,spec,status,time\n",
                "track,a.onnx,a.vnnlib,unknown,1\n",
                "extended_bank_v1",
            ),
            (
                "cat,onnx,vnnlib,prepared,verdict,secs\n",
                "track,a.onnx,a.vnnlib,0,unknown,1\n",
                "measured_v1",
            ),
        )
        for header, row, expected_schema in cases:
            with self.subTest(expected_schema=expected_schema):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "source.csv"
                    source.write_text(header + row, encoding="utf-8")
                    results = validate_bank.load_source_results(source, "track")
                self.assertEqual(
                    [result.schema for result in results], [expected_schema]
                )

    def test_declared_header_rejects_every_target_schema_mismatch(self) -> None:
        cases = (
            (
                "track,onnx,vnnlib,prepared,verdict,seconds\n",
                "track,a.onnx,a.vnnlib,sat,1\n",
            ),
            (
                "track,onnx,vnnlib,verdict,seconds\n",
                "track,a.onnx,a.vnnlib,0,sat,1\n",
            ),
            (
                "track,onnx,vnnlib,prepared,verdict,seconds,run_id\n",
                "track,a.onnx,a.vnnlib,0,sat,1\n",
            ),
        )
        for header, row in cases:
            with self.subTest(header=header.strip()):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "source.csv"
                    source.write_text(header + row, encoding="utf-8")
                    with self.assertRaisesRegex(
                        validate_bank.BankValidationError,
                        "does not match the declared",
                    ):
                        validate_bank.load_source_results(source, "track")

    def test_duplicate_or_late_header_is_rejected(self) -> None:
        header = "track,onnx,vnnlib,verdict,seconds\n"
        target_row = "track,a.onnx,a.vnnlib,unknown,1\n"
        sources = (
            header + header + target_row,
            target_row + header,
            "other,a.onnx,a.vnnlib,unknown,1\n" + header + target_row,
        )
        for contents in sources:
            with self.subTest(contents=contents):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "source.csv"
                    source.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(
                        validate_bank.BankValidationError,
                        "header must appear exactly once first",
                    ):
                        validate_bank.load_source_results(source, "track")

    def test_unsupported_or_ambiguous_header_aliases_are_rejected(self) -> None:
        headers = (
            "group,onnx,vnnlib,verdict,seconds",
            "track,network,vnnlib,verdict,seconds",
            "track,onnx,formula,verdict,seconds",
            "track,onnx,vnnlib,outcome,seconds",
            "track,onnx,vnnlib,seconds,verdict",
            "track,onnx,vnnlib,prepared,verdict,seconds,",
            "track,onnx,vnnlib,prepared,status,time,run",
        )
        for header in headers:
            with self.subTest(header=header):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "source.csv"
                    source.write_text(header + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        validate_bank.BankValidationError,
                        "unsupported or ambiguous CSV header",
                    ):
                        validate_bank.load_source_results(source, "track")

    def test_actual_alias_header_artifacts_remain_loadable(self) -> None:
        for track in ("lsnc_relu", "relusplitter"):
            source = REPO_ROOT / "reports/measured" / f"{track}.csv"
            with self.subTest(track=track):
                self.assertTrue(
                    source.is_file(),
                    f"tracked measured report artifact is missing: {source}",
                )
                results = validate_bank.load_source_results(source, track)
                self.assertTrue(results)
                # These two ledgers moved to the 7-column `measured_v2` header
                # when a re-sweep started recording run_id on the rows it
                # actually re-ran. The declared header must keep binding EVERY
                # row to one schema (that strictness is pinned by
                # test_declared_header_rejects_every_target_schema_mismatch), so
                # rows nobody re-ran carry the repository's existing
                # `inherited-unverified` token rather than an empty field.
                self.assertTrue(
                    all(result.schema == "measured_v2" for result in results)
                )
                self.assertTrue(
                    all(result.run_id for result in results),
                    "a measured_v2 ledger must not leave provenance blank",
                )

    def test_headerless_five_six_and_seven_column_schemas_remain_supported(
        self,
    ) -> None:
        contents = (
            "track,a.onnx,a.vnnlib,unknown,1\n",
            "track,a.onnx,a.vnnlib,0,unknown,1\n",
            "track,a.onnx,a.vnnlib,0,unknown,1,run\n",
            (
                "track,a.onnx,a.vnnlib,0,unknown,1\n"
                "track,b.onnx,b.vnnlib,0,unknown,2,run\n"
            ),
        )
        for rows in contents:
            with self.subTest(rows=rows):
                with tempfile.TemporaryDirectory() as temporary:
                    source = Path(temporary) / "source.csv"
                    source.write_text(rows, encoding="utf-8")
                    results = validate_bank.load_source_results(source, "track")
                self.assertTrue(results)


class PortabilityTests(unittest.TestCase):
    def test_defaults_are_derived_from_script_repository(self) -> None:
        parser = validate_bank.build_parser()
        args = validate_bank._resolve_cli(parser.parse_args(["track", "source.csv"]))
        self.assertEqual(validate_bank.REPO_ROOT, REPO_ROOT)
        self.assertEqual(args.ny_bin, REPO_ROOT / "target/release/ny")
        self.assertEqual(
            args.ay_bin,
            (REPO_ROOT.parent / "ay/target/release/ay").resolve(),
        )
        self.assertEqual(
            args.bench_root,
            (REPO_ROOT / "benchmarks/vnncomp2025/benchmarks").resolve(),
        )
        self.assertEqual(args.output, REPO_ROOT / "reports/measured-ext/track.csv")

    def test_default_ay_binary_resolves_a_symlinked_sibling_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            repo_root = parent / "ny"
            actual_ay = parent / "immutable-ay"
            repo_root.mkdir()
            (actual_ay / "target/release").mkdir(parents=True)
            try:
                (parent / "ay").symlink_to(actual_ay, target_is_directory=True)
            except OSError as error:
                self.fail(
                    "directory symlinks are required by the sibling-repository "
                    f"portability contract: {error}"
                )

            resolved = validate_bank._resolve_from_repo(
                None,
                repo_root,
                repo_root.parent / "ay/target/release/ay",
            )

            self.assertEqual(resolved, actual_ay.resolve() / "target/release/ay")

    def test_help_does_not_import_numpy_or_onnxruntime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            blocker = Path(temporary)
            (blocker / "numpy.py").write_text(
                "raise RuntimeError('numpy was imported')\n", encoding="utf-8"
            )
            (blocker / "onnxruntime.py").write_text(
                "raise RuntimeError('onnxruntime was imported')\n", encoding="utf-8"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(blocker)
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "validate_bank.py"), "--help"],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ny-bin", result.stdout)
        self.assertIn("--ay-bin", result.stdout)
        self.assertIn("--bench-root", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_run_uses_private_result_directory_and_removes_it(self) -> None:
        observed_result_path: Path | None = None

        class FakeProcess:
            def __init__(self, command, **_kwargs):
                nonlocal observed_result_path
                observed_result_path = Path(command[6])
                self.returncode = 0
                observed_result_path.write_bytes(b"sat\n((X_0 0.0))\n")

            def wait(self, timeout=None):
                del timeout
                return self.returncode

            def kill(self):
                self.returncode = -9

        with mock.patch.object(validate_bank.subprocess, "Popen", FakeProcess):
            result = validate_bank.run_ny(
                ny_bin=Path("/tmp/ny"),
                ay_bin=Path("/tmp/ay"),
                track="track",
                onnx_path=Path("/tmp/a.onnx"),
                vnnlib_path=Path("/tmp/a.vnnlib"),
                budget=1,
                environment={},
            )
        self.assertEqual(result.verdict, "sat")
        self.assertIsNotNone(observed_result_path)
        assert observed_result_path is not None
        self.assertFalse(observed_result_path.parent.exists())

    def test_validator_is_loaded_from_exact_sibling_not_sys_modules(self) -> None:
        sentinel = SimpleNamespace(__file__="/tmp/not-the-validator.py")
        previous = sys.modules.get("vnnlib_ce")
        sys.modules["vnnlib_ce"] = sentinel
        try:
            validator = validate_bank._load_validator()
        finally:
            if previous is None:
                sys.modules.pop("vnnlib_ce", None)
            else:
                sys.modules["vnnlib_ce"] = previous
        self.assertIsNot(validator, sentinel)
        self.assertEqual(
            Path(validator.__file__).resolve(),
            (SCRIPT_DIR / "vnnlib_ce.py").resolve(),
        )

    def test_track_component_must_start_alphanumeric(self) -> None:
        for track in (".", "..", "-track", "_track"):
            with self.subTest(track=track):
                self.assertIsNone(validate_bank.SAFE_COMPONENT.fullmatch(track))


class StreamingParserTests(unittest.TestCase):
    VALID_PROPERTY = (
        "(declare-const X_0 Real)\n"
        "(declare-const Y_0 Real)\n"
        "(assert (>= X_0 0))\n"
        "(assert (>= Y_0 0))\n"
    )

    def _requirements(self, source: str):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "property.vnnlib"
            path.write_text(source, encoding="utf-8")
            return vnnlib_ce.property_requirements(path)

    def test_unknown_empty_and_bare_top_level_forms_are_rejected(self) -> None:
        for bad_form in ("(check-sat)", "junk", "()", "(set-option foo bar)"):
            with self.subTest(bad_form=bad_form):
                with self.assertRaisesRegex(vnnlib_ce.ValidationError, "top-level"):
                    self._requirements(f"{bad_form}\n{self.VALID_PROPERTY}")

    def test_command_order_and_canonical_variable_names_are_strict(self) -> None:
        bad_properties = (
            self.VALID_PROPERTY.replace("X_0 Real", "X_00 Real", 1),
            self.VALID_PROPERTY.replace("(>= X_0 0)", "(>= X_00 0)", 1),
            self.VALID_PROPERTY.replace("Y_0 Real", "Y_00 Real", 1),
            self.VALID_PROPERTY.replace("(>= Y_0 0)", "(>= Y_00 0)", 1),
            "(assert true)\n" + self.VALID_PROPERTY,
            "(declare-const X_0 Real)\n(set-logic QF_LRA)\n"
            "(declare-const Y_0 Real)\n(assert (>= X_0 0))\n"
            "(assert (>= Y_0 0))\n",
            "(set-logic QF_LRA)\n(set-logic QF_LRA)\n" + self.VALID_PROPERTY,
            self.VALID_PROPERTY + "(declare-const Y_1 Real)\n",
        )
        for source in bad_properties:
            with self.subTest(source=source[:80]):
                with self.assertRaises(vnnlib_ce.ValidationError):
                    self._requirements(source)

        requirements = self._requirements("(set-logic QF_LRA)\n" + self.VALID_PROPERTY)
        self.assertEqual(requirements.input_count, 1)

    def test_numeric_atoms_use_vnnlib_ascii_grammar(self) -> None:
        for atom in ("1_0", "+1_0.5", "\N{ARABIC-INDIC DIGIT ONE}"):
            with self.subTest(atom=atom):
                with self.assertRaisesRegex(vnnlib_ce.ValidationError, "unknown"):
                    vnnlib_ce.evaluate(atom, {})

    def test_operator_arities_and_all_boolean_branches_are_fail_closed(self) -> None:
        for expression in (["+"], ["+", "1"], ["*"], ["*", "1"], ["and"], ["or"]):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(vnnlib_ce.ValidationError, "requires"):
                    vnnlib_ce.evaluate(expression, {})
        for expression in (
            ["and", "false", "unknown"],
            ["or", "true", "unknown"],
            ["or", ["and"], [">=", "Y_0", "0"]],
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(vnnlib_ce.ValidationError):
                    vnnlib_ce.evaluate(expression, {"Y_0": 0.0})

        self.assertTrue(vnnlib_ce.evaluate(["and", "true"], {}))
        self.assertFalse(vnnlib_ce.evaluate(["or", "false"], {}))

    def test_cli_assignment_parser_rejects_noncanonical_atoms(self) -> None:
        for source in ("((X_00 0))", "((X_0 1_0))", "((X_0 0) (X_0 1))"):
            with self.subTest(source=source):
                with self.assertRaises(vnnlib_ce.ValidationError):
                    vnnlib_ce._extract_cli_assignment(source)

    def test_giant_comment_is_read_in_bounded_chunks(self) -> None:
        class GiantCommentReader:
            def __init__(self) -> None:
                self.remaining = vnnlib_ce.SOURCE_CHUNK_CHARS * 8
                self.started = False
                self.finished = False
                self.maximum_request = 0

            def read(self, size: int) -> str:
                self.maximum_request = max(self.maximum_request, size)
                if self.remaining:
                    amount = min(size, self.remaining)
                    self.remaining -= amount
                    prefix = ";" if not self.started else ""
                    self.started = True
                    return prefix + "x" * (amount - len(prefix))
                if not self.finished:
                    self.finished = True
                    return "\n(assert true)\n"
                return ""

        source = GiantCommentReader()
        self.assertEqual(
            list(vnnlib_ce.tokenize(source)),
            ["(", "assert", "true", ")"],
        )
        self.assertLessEqual(source.maximum_request, vnnlib_ce.SOURCE_CHUNK_CHARS)

    def test_token_atom_and_nesting_caps_fail_closed(self) -> None:
        with self.assertRaisesRegex(vnnlib_ce.ValidationError, "atom exceeds"):
            list(vnnlib_ce.tokenize("a" * (vnnlib_ce.MAX_ATOM_CHARS + 1)))
        nested = "(" * (vnnlib_ce.MAX_EXPRESSION_DEPTH + 1)
        nested += "true"
        nested += ")" * (vnnlib_ce.MAX_EXPRESSION_DEPTH + 1)
        with self.assertRaisesRegex(vnnlib_ce.ValidationError, "nesting exceeds"):
            list(vnnlib_ce.parse_all(nested))
        with (
            mock.patch.object(vnnlib_ce, "MAX_EXPRESSION_TOKENS", 8),
            self.assertRaisesRegex(vnnlib_ce.ValidationError, "tokens"),
        ):
            list(vnnlib_ce.parse_all("(and true true true true true true true)"))

    def test_large_synthetic_property_has_bounded_peak_memory(self) -> None:
        assertion_count = 50_000
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large.vnnlib"
            with path.open("w", encoding="utf-8") as output:
                output.write("(declare-const X_0 Real)\n")
                output.write("(declare-const Y_0 Real)\n")
                for _ in range(assertion_count):
                    output.write("(assert (>= X_0 0))\n")
                output.write("(assert (>= Y_0 0))\n")
            tracemalloc.start()
            try:
                requirements = vnnlib_ce.property_requirements(path)
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        self.assertEqual(requirements.input_assertion_count, assertion_count)
        self.assertLess(peak, 4 * 1024 * 1024)

class FailClosedWitnessTests(unittest.TestCase):
    @staticmethod
    def _reject_heavy_imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"numpy", "onnxruntime"}:
            raise AssertionError(f"heavy dependency imported: {name}")
        return FailClosedWitnessTests._original_import(
            name, globals, locals, fromlist, level
        )

    _original_import = builtins.__import__

    @staticmethod
    def _fake_runtime(output_values: list[float]):
        class FakeArray:
            def __init__(self, values):
                self.values = list(values)

            def reshape(self, _shape):
                return self

            def flatten(self):
                return self

            def astype(self, _dtype):
                return self

            def __iter__(self):
                return iter(self.values)

            def __len__(self):
                return len(self.values)

            def __getitem__(self, index):
                return self.values[index]

        class FakeSession:
            def get_inputs(self):
                return [SimpleNamespace(name="input", shape=[1])]

            def run(self, *_args, **_kwargs):
                return [FakeArray(output_values)]

        fake_numpy = SimpleNamespace(
            array=lambda values, dtype=None: FakeArray(values),
            fromiter=lambda values, dtype=None, count=-1: FakeArray(values),
            float32=object(),
            float64=object(),
        )
        fake_ort = SimpleNamespace(
            InferenceSession=lambda *_args, **_kwargs: FakeSession()
        )
        return mock.patch.dict(
            sys.modules,
            {"numpy": fake_numpy, "onnxruntime": fake_ort},
        )

    def _property(self, directory: Path, output_assertion: bool = True) -> Path:
        suffix = "(assert (>= Y_0 0.0))\n" if output_assertion else ""
        path = directory / "property.vnnlib"
        path.write_text(
            "(declare-const X_0 Real)\n"
            "(declare-const X_1 Real)\n"
            "(declare-const Y_0 Real)\n"
            "(assert (>= X_0 -1.0))\n"
            "(assert (<= X_0 1.0))\n"
            "(assert (>= X_1 -1.0))\n"
            "(assert (<= X_1 1.0))\n" + suffix,
            encoding="utf-8",
        )
        return path

    def test_incomplete_input_assignment_rejects_before_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prop = self._property(Path(temporary))
            with mock.patch("builtins.__import__", self._reject_heavy_imports):
                in_box, is_counterexample, detail = vnnlib_ce.validate(
                    Path(temporary) / "unused.onnx", prop, {0: 0.0}
                )
        self.assertFalse(in_box)
        self.assertFalse(is_counterexample)
        self.assertIn("missing X_1", detail)

    def test_property_without_output_assertion_rejects_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prop = self._property(Path(temporary), output_assertion=False)
            with mock.patch("builtins.__import__", self._reject_heavy_imports):
                in_box, is_counterexample, detail = vnnlib_ce.validate(
                    Path(temporary) / "unused.onnx", prop, {0: 0.0, 1: 0.0}
                )
        self.assertFalse(in_box)
        self.assertFalse(is_counterexample)
        self.assertIn("no output-referencing assertions", detail)

    def test_duplicate_witness_assignment_is_explicit(self) -> None:
        witness = validate_bank.parse_witness(
            b"sat\n((X_0 0.0)\n(X_0 0.5)\n(X_1 0.0))\n"
        )
        self.assertEqual(witness.duplicate_indices, (0,))
        self.assertEqual(witness.duplicate_count, 1)

    def test_duplicate_witness_tracking_is_bounded(self) -> None:
        assignments = [f"(X_{index} 0)" for index in range(100)]
        assignments.extend(f"(X_{index} 1)" for index in range(100))
        witness = validate_bank.parse_witness(
            ("sat\n(" + "\n".join(assignments) + ")\n").encode()
        )
        self.assertEqual(witness.duplicate_count, 100)
        self.assertEqual(len(witness.duplicate_indices), 32)

    def test_witness_indices_and_numbers_use_canonical_ascii_grammar(self) -> None:
        for assignment in (
            b"(X_00 0)",
            b"(X_0 1_0)",
            "(X_\N{ARABIC-INDIC DIGIT ONE} 0)".encode(),
        ):
            with self.subTest(assignment=assignment):
                witness = validate_bank.parse_witness(b"sat\n(" + assignment + b")\n")
                self.assertIsNotNone(witness.parse_error)

    def test_nested_non_simple_input_assertion_is_evaluated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prop = Path(temporary) / "property.vnnlib"
            prop.write_text(
                "(declare-const X_0 Real)\n"
                "(declare-const X_1 Real)\n"
                "(declare-const Y_0 Real)\n"
                "(assert (and (>= X_0 0) (>= X_1 0) "
                "(<= (+ X_0 X_1) 0.5)))\n"
                "(assert (>= Y_0 0))\n",
                encoding="utf-8",
            )
            with mock.patch("builtins.__import__", self._reject_heavy_imports):
                in_box, is_counterexample, detail = vnnlib_ce.validate(
                    Path(temporary) / "unused.onnx", prop, {0: 0.4, 1: 0.4}
                )
        self.assertFalse(in_box)
        self.assertFalse(is_counterexample)
        self.assertIn("raw_domain_results=[False]", detail)

    def test_every_declared_input_must_have_an_input_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prop = Path(temporary) / "property.vnnlib"
            prop.write_text(
                "(declare-const X_0 Real)\n"
                "(declare-const X_1 Real)\n"
                "(declare-const Y_0 Real)\n"
                "(assert (>= X_0 0))\n"
                "(assert (>= Y_0 0))\n",
                encoding="utf-8",
            )
            with mock.patch("builtins.__import__", self._reject_heavy_imports):
                in_box, is_counterexample, detail = vnnlib_ce.validate(
                    Path(temporary) / "unused.onnx", prop, {0: 0.0, 1: 0.0}
                )
        self.assertFalse(in_box)
        self.assertFalse(is_counterexample)
        self.assertIn("input constraints do not reference X_1", detail)

    def test_raw_domain_gates_in_box_f32_executed_is_diagnostic_only(self) -> None:
        # Official zero-tol semantics (SCORING-ZERO-TOL/counterexamples.py):
        # input constraints are checked on the RAW parsed witness values; the
        # float32 cast happens only inside the ORT execution. A witness whose
        # raw value is in-box but whose f32 image crosses the bound must PASS
        # the in-box gate (previously it was rejected, which made specs pinning
        # inputs to non-f32-representable constants — cctsdb_yolo — impossible
        # to falsify). With the gate passed, validate() proceeds to the ORT
        # execution stage, which the heavy-import mock rejects here.
        with tempfile.TemporaryDirectory() as temporary:
            prop = Path(temporary) / "property.vnnlib"
            prop.write_text(
                "(declare-const X_0 Real)\n"
                "(declare-const Y_0 Real)\n"
                "(assert (>= X_0 0))\n"
                "(assert (<= X_0 1.00000008))\n"
                "(assert (>= Y_0 0))\n",
                encoding="utf-8",
            )
            with mock.patch("builtins.__import__", self._reject_heavy_imports):
                with self.assertRaises(AssertionError):
                    vnnlib_ce.validate(
                        Path(temporary) / "unused.onnx", prop, {0: 1.00000007}
                    )

    def test_mixed_xy_assertion_uses_the_raw_witness_input(self) -> None:
        # Official zero-tol semantics: the full-tree evaluation runs at
        # (RAW witness X, ORT-executed Y) — counterexamples.py evaluates
        # is_specification_vio on tuple(x_list). Raw 1.00000007 + 0.0 holds
        # under the 1.00000008 bound, so this IS a counterexample even though
        # the f32 image of X_0 (1.0000001) would cross the bound.
        with tempfile.TemporaryDirectory() as temporary:
            prop = Path(temporary) / "property.vnnlib"
            prop.write_text(
                "(declare-const X_0 Real)\n"
                "(declare-const Y_0 Real)\n"
                "(assert (>= X_0 0))\n"
                "(assert (<= X_0 2))\n"
                "(assert (<= (+ X_0 Y_0) 1.00000008))\n",
                encoding="utf-8",
            )
            with self._fake_runtime([0.0]):
                in_box, is_counterexample, detail = vnnlib_ce.validate(
                    Path(temporary) / "unused.onnx", prop, {0: 1.00000007}
                )
        self.assertTrue(in_box)
        self.assertTrue(is_counterexample)
        self.assertIn("output_results=[True]", detail)

    def test_nonfinite_onnx_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prop = Path(temporary) / "property.vnnlib"
            prop.write_text(
                "(declare-const X_0 Real)\n"
                "(declare-const Y_0 Real)\n"
                "(assert (>= X_0 0))\n"
                "(assert (<= X_0 1))\n"
                "(assert (not (>= Y_0 0)))\n",
                encoding="utf-8",
            )
            with (
                self._fake_runtime([float("nan")]),
                self.assertRaisesRegex(vnnlib_ce.ValidationError, "non-finite output"),
            ):
                vnnlib_ce.validate(Path(temporary) / "unused.onnx", prop, {0: 0.0})


class EvidenceTests(unittest.TestCase):
    @staticmethod
    def _fixture(root: Path, verdict: str = "sat") -> tuple[object, dict[str, Path]]:
        source = root / "source.csv"
        source.write_text(
            f"track,onnx/a.onnx,vnnlib/a.vnnlib,0,{verdict},4,run-id\n",
            encoding="utf-8",
        )
        ny_bin = root / "ny"
        ay_bin = root / "ay"
        ny_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        ay_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        ny_bin.chmod(0o755)
        ay_bin.chmod(0o755)
        benchmark_dir = root / "bench" / "track"
        (benchmark_dir / "onnx").mkdir(parents=True)
        (benchmark_dir / "vnnlib").mkdir()
        onnx = benchmark_dir / "onnx/a.onnx"
        vnnlib = benchmark_dir / "vnnlib/a.vnnlib"
        onnx.write_bytes(b"model")
        vnnlib.write_text("property", encoding="utf-8")
        args = validate_bank._resolve_cli(
            validate_bank.build_parser().parse_args(
                [
                    "track",
                    str(source),
                    "--repo-root",
                    str(root),
                    "--ny-bin",
                    str(ny_bin),
                    "--ay-bin",
                    str(ay_bin),
                    "--bench-root",
                    str(root / "bench"),
                    "--output",
                    str(root / "bank.csv"),
                ]
            )
        )
        return args, {
            "source": source,
            "ny": ny_bin,
            "ay": ay_bin,
            "onnx": onnx,
            "vnnlib": vnnlib,
        }

    def test_raw_sat_and_hashed_validation_record_are_retained_read_only(self) -> None:
        raw = b"sat\n((X_0 0.0))\n"
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path = validate_bank.retain_validation_evidence(
                evidence_root=Path(temporary) / "evidence",
                raw_result=raw,
                record={
                    "schema": "ny_extended_bank_validation_v1",
                    "banked_verdict": "sat",
                },
                instance_name="property.vnnlib",
            )
            record = json.loads(evidence_path.read_text(encoding="utf-8"))
            raw_path = evidence_path.parent / record["raw_result"]["artifact"]
            self.assertEqual(raw_path.read_bytes(), raw)
            self.assertEqual(
                record["raw_result"]["sha256"], hashlib.sha256(raw).hexdigest()
            )
            self.assertEqual(record["raw_result"]["size_bytes"], len(raw))
            self.assertEqual(raw_path.stat().st_mode & 0o222, 0)
            self.assertEqual(evidence_path.stat().st_mode & 0o222, 0)

            second = validate_bank.retain_validation_evidence(
                evidence_root=Path(temporary) / "evidence",
                raw_result=b"sat\n((X_0 0.5))\n",
                record={
                    "schema": "ny_extended_bank_validation_v1",
                    "banked_verdict": "sat",
                },
                instance_name="property.vnnlib",
            )
            self.assertNotEqual(second, evidence_path)
            self.assertEqual(raw_path.read_bytes(), raw)
            self.assertFalse(list((Path(temporary) / "evidence").glob(".tmp-*")))

    def test_writable_preexisting_raw_artifact_is_rejected(self) -> None:
        raw = b"sat\n((X_0 0.0))\n"
        with tempfile.TemporaryDirectory() as temporary:
            evidence_root = Path(temporary) / "evidence"
            evidence_root.mkdir()
            raw_path = evidence_root / f"{hashlib.sha256(raw).hexdigest()}.results"
            raw_path.write_bytes(raw)
            raw_path.chmod(0o666)
            with self.assertRaisesRegex(
                validate_bank.BankValidationError, "artifact is writable"
            ):
                validate_bank.retain_validation_evidence(
                    evidence_root=evidence_root,
                    raw_result=raw,
                    record={"schema": "ny_extended_bank_validation_v1"},
                    instance_name="property.vnnlib",
                )
            self.assertNotEqual(raw_path.stat().st_mode & 0o222, 0)
            self.assertFalse(list(evidence_root.glob("*.validation.json")))

    def test_banked_sat_sidecar_contains_versions_detail_and_bound_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            source.write_text(
                "track,onnx/a.onnx,vnnlib/a.vnnlib,0,sat,4,run-id\n",
                encoding="utf-8",
            )
            ny_bin = root / "ny"
            ny_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            ny_bin.chmod(0o755)
            benchmark_dir = root / "bench" / "track"
            (benchmark_dir / "onnx").mkdir(parents=True)
            (benchmark_dir / "vnnlib").mkdir()
            onnx_path = benchmark_dir / "onnx/a.onnx"
            property_path = benchmark_dir / "vnnlib/a.vnnlib"
            onnx_path.write_bytes(b"model")
            property_path.write_text(
                "(declare-const X_0 Real)\n"
                "(declare-const Y_0 Real)\n"
                "(assert (>= Y_0 0))\n",
                encoding="utf-8",
            )
            args = validate_bank._resolve_cli(
                validate_bank.build_parser().parse_args(
                    [
                        "track",
                        str(source),
                        "--repo-root",
                        str(root),
                        "--ny-bin",
                        str(ny_bin),
                        "--bench-root",
                        str(root / "bench"),
                        "--output",
                        str(root / "bank.csv"),
                    ]
                )
            )
            validator = SimpleNamespace(
                __file__=str(SCRIPT_DIR / "vnnlib_ce.py"),
                runtime_versions=lambda: {
                    "python": "test",
                    "validator": "full_assert_v2",
                    "numpy": "test",
                    "onnxruntime": "test",
                },
                property_requirements=lambda _path: SimpleNamespace(
                    input_indices=(0,),
                    input_assertion_count=1,
                    output_assertion_count=1,
                ),
                validate=lambda *_args: (
                    True,
                    True,
                    "complete_inputs=1 yasserts=1 all_hold=True",
                ),
            )
            run = validate_bank.NyRun(
                "sat", b"sat\n((X_0 0.0))\n", returncode=0, timed_out=False
            )
            with (
                mock.patch.object(
                    validate_bank, "_load_validator", return_value=validator
                ),
                mock.patch.object(validate_bank, "run_ny", return_value=run),
            ):
                validate_bank.bank(args)
            sidecars = list(args.evidence_root.glob("*.validation.json"))
            self.assertEqual(len(sidecars), 1)
            record = json.loads(sidecars[0].read_text(encoding="utf-8"))
            self.assertEqual(
                record["validation"]["runtime_versions"]["validator"],
                "full_assert_v2",
            )
            self.assertIn("complete_inputs=1", record["validation"]["detail"])
            self.assertEqual(
                record["instance"]["onnx"]["sha256"],
                hashlib.sha256(b"model").hexdigest(),
            )
            self.assertEqual(
                record["raw_result"]["sha256"],
                hashlib.sha256(run.raw_result).hexdigest(),
            )

    def test_non_sat_bank_preserves_semantics_without_loading_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            with source.open("w", encoding="utf-8", newline="") as output:
                csv.writer(output).writerow(
                    [
                        "track",
                        "onnx/a.onnx",
                        "vnnlib/a.vnnlib",
                        "0",
                        "unsat",
                        "4",
                        "run-id",
                    ]
                )
            ny_bin = root / "ny"
            ny_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            ny_bin.chmod(0o755)
            benchmark_dir = root / "bench" / "track"
            (benchmark_dir / "onnx").mkdir(parents=True)
            (benchmark_dir / "vnnlib").mkdir()
            (benchmark_dir / "onnx/a.onnx").write_bytes(b"model")
            (benchmark_dir / "vnnlib/a.vnnlib").write_text("property", encoding="utf-8")
            parser = validate_bank.build_parser()
            args = validate_bank._resolve_cli(
                parser.parse_args(
                    [
                        "track",
                        str(source),
                        "--repo-root",
                        str(root),
                        "--ny-bin",
                        str(ny_bin),
                        "--bench-root",
                        str(root / "bench"),
                        "--output",
                        str(root / "bank.csv"),
                    ]
                )
            )
            with mock.patch.object(
                validate_bank, "_load_validator", side_effect=AssertionError
            ):
                returncode = validate_bank.bank(args)
            with (root / "bank.csv").open(encoding="utf-8", newline="") as bank_file:
                rows = list(csv.reader(bank_file))
        self.assertEqual(returncode, 0)
        self.assertEqual(
            rows,
            [["track", "onnx/a.onnx", "vnnlib/a.vnnlib", "unsat", "4"]],
        )

    def test_invalid_reproduced_sat_writes_unknown_evidence_and_exits_nonzero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _paths = self._fixture(root)
            validator = SimpleNamespace(
                __file__=str(SCRIPT_DIR / "vnnlib_ce.py"),
                runtime_versions=lambda: {"validator": "full_assert_v2"},
                property_requirements=lambda _path: SimpleNamespace(
                    input_indices=(0,),
                    input_assertion_count=1,
                    output_assertion_count=1,
                ),
                validate=lambda *_args: (True, False, "output assertion is false"),
            )
            run = validate_bank.NyRun(
                "sat", b"sat\n((X_0 0.0))\n", returncode=0, timed_out=False
            )
            with (
                mock.patch.object(
                    validate_bank, "_load_validator", return_value=validator
                ),
                mock.patch.object(validate_bank, "run_ny", return_value=run),
            ):
                returncode = validate_bank.bank(args)
            with args.output.open(encoding="utf-8", newline="") as bank_file:
                rows = list(csv.reader(bank_file))
            sidecars = list(args.evidence_root.glob("*.validation.json"))
            self.assertEqual(len(sidecars), 1)
            evidence = json.loads(sidecars[0].read_text(encoding="utf-8"))
        self.assertEqual(returncode, 3)
        self.assertEqual(
            rows,
            [["track", "onnx/a.onnx", "vnnlib/a.vnnlib", "unknown", "4"]],
        )
        self.assertEqual(evidence["banked_verdict"], "unknown")
        self.assertFalse(evidence["validation"]["is_counterexample"])

    def test_empty_target_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            source.write_text("different,a.onnx,a.vnnlib,unknown,1\n", encoding="utf-8")
            ny = root / "ny"
            ny.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            ny.chmod(0o755)
            args = validate_bank._resolve_cli(
                validate_bank.build_parser().parse_args(
                    [
                        "track",
                        str(source),
                        "--repo-root",
                        str(root),
                        "--ny-bin",
                        str(ny),
                    ]
                )
            )
            with self.assertRaisesRegex(
                validate_bank.BankValidationError, "contains no rows"
            ):
                validate_bank.bank(args)

    def test_even_unsat_rows_require_contained_existing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, paths = self._fixture(root, verdict="unsat")
            outside = root / "outside.onnx"
            outside.write_bytes(b"outside")
            paths["onnx"].unlink()
            paths["onnx"].symlink_to(outside)
            with self.assertRaisesRegex(
                validate_bank.BankValidationError, "escapes its track directory"
            ):
                validate_bank.bank(args)
            with self.assertRaisesRegex(
                validate_bank.BankValidationError, "relative POSIX path"
            ):
                validate_bank._benchmark_file(
                    root / "bench/track", str(outside.resolve())
                )

    def test_source_model_property_ny_and_ay_changes_are_fatal(self) -> None:
        for target in ("source", "onnx", "vnnlib", "ny", "ay"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                args, paths = self._fixture(root)

                def mutate_then_return(*, _paths=paths, _target=target, **_kwargs):
                    path = _paths[_target]
                    path.write_bytes(path.read_bytes() + b"changed")
                    return validate_bank.NyRun(
                        "unknown", b"unknown\n", returncode=0, timed_out=False
                    )

                with (
                    mock.patch.object(
                        validate_bank, "run_ny", side_effect=mutate_then_return
                    ),
                    self.assertRaisesRegex(
                        validate_bank.BankValidationError, "changed during validation"
                    ),
                ):
                    validate_bank.bank(args)
                self.assertFalse(args.output.exists())

    def test_output_and_evidence_destinations_cannot_alias_bound_inputs(self) -> None:
        for destination in ("output", "evidence"):
            for target in ("source", "ny", "ay", "onnx", "vnnlib"):
                with (
                    self.subTest(destination=destination, target=target),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    root = Path(temporary)
                    args, paths = self._fixture(root, verdict="unsat")
                    original = paths[target].read_bytes()
                    if destination == "output":
                        args.output = paths[target]
                    else:
                        args.evidence_root = paths[target]
                    with self.assertRaisesRegex(
                        validate_bank.BankValidationError, "collides"
                    ):
                        validate_bank.bank(args)
                    self.assertEqual(paths[target].read_bytes(), original)

    def test_output_csv_cannot_be_inside_evidence_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence_root = Path(temporary) / "evidence"
            for output in (
                evidence_root,
                evidence_root / "bank.csv",
                evidence_root / "nested/bank.csv",
            ):
                with self.subTest(output=output):
                    with self.assertRaisesRegex(
                        validate_bank.BankValidationError, "evidence root"
                    ):
                        validate_bank._reject_destination_collisions(
                            output=output,
                            evidence_root=evidence_root,
                            inputs=(),
                        )

    def test_resolved_path_alias_cannot_hide_solved_verdict_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, paths = self._fixture(root)
            paths["source"].write_text(
                "track,onnx/a.onnx,vnnlib/a.vnnlib,sat,1\n"
                "track,onnx/./a.onnx,vnnlib/./a.vnnlib,unsat,2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                validate_bank.BankValidationError,
                "conflicting solved verdicts resolve to the same",
            ):
                validate_bank.bank(args)


class EnvironmentDependencyTests(unittest.TestCase):
    @staticmethod
    def _block_import(name: str):
        real_import_module = importlib.import_module

        def fake_import_module(target, package=None):
            if target == name:
                raise ModuleNotFoundError(f"No module named {name!r}")
            if target in validate_bank.VALIDATION_DEPENDENCIES:
                # Keep this dependency-order test hermetic: the repository's
                # base test requirements intentionally do not install ONNX or
                # ONNX Runtime, and an earlier absent package must not mask the
                # package selected by this subtest.
                return SimpleNamespace(__name__=target)
            return real_import_module(target, package)

        return mock.patch.object(
            validate_bank.importlib, "import_module", fake_import_module
        )

    def test_missing_dependency_is_an_environment_error_not_a_moat_breach(
        self,
    ) -> None:
        for missing in validate_bank.VALIDATION_DEPENDENCIES:
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                _args, paths = EvidenceTests._fixture(root)
                argv = [
                    "track",
                    str(paths["source"]),
                    "--repo-root",
                    str(root),
                    "--ny-bin",
                    str(paths["ny"]),
                    "--ay-bin",
                    str(paths["ay"]),
                    "--bench-root",
                    str(root / "bench"),
                    "--output",
                    str(root / "bank.csv"),
                ]
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    self._block_import(missing),
                    mock.patch.object(
                        validate_bank, "run_ny", side_effect=AssertionError
                    ),
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    returncode = validate_bank.main(argv)
                self.assertEqual(returncode, 4)
                self.assertIn("ENVIRONMENT ERROR", stderr.getvalue())
                self.assertIn(missing, stderr.getvalue())
                self.assertNotIn("MOAT", stdout.getvalue())
                self.assertFalse((root / "bank.csv").exists())

    def test_runtime_import_error_never_reaches_the_moat_breach_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _paths = EvidenceTests._fixture(root)
            validator = SimpleNamespace(
                __file__=str(SCRIPT_DIR / "vnnlib_ce.py"),
                runtime_versions=lambda: {"validator": "full_assert_v2"},
                property_requirements=lambda _path: SimpleNamespace(
                    input_indices=(0,),
                    input_assertion_count=1,
                    output_assertion_count=1,
                ),
                validate=mock.Mock(
                    side_effect=ModuleNotFoundError("No module named 'onnxruntime'")
                ),
            )
            run = validate_bank.NyRun(
                "sat", b"sat\n((X_0 0.0))\n", returncode=0, timed_out=False
            )
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    validate_bank, "_load_validator", return_value=validator
                ),
                mock.patch.object(validate_bank, "run_ny", return_value=run),
                contextlib.redirect_stdout(stdout),
                self.assertRaises(validate_bank.EnvironmentDependencyError),
            ):
                validate_bank.bank(args)
            self.assertNotIn("MOAT", stdout.getvalue())
            self.assertFalse(args.output.exists())

    def test_vnnlib_ce_cli_missing_dependency_has_distinct_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            blocker = Path(temporary)
            (blocker / "onnxruntime.py").write_text(
                "raise ImportError('onnxruntime is blocked for this test')\n",
                encoding="utf-8",
            )
            counterexample = blocker / "ce.txt"
            counterexample.write_text("((X_0 0.0))\n", encoding="utf-8")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(blocker)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "vnnlib_ce.py"),
                    str(blocker / "model.onnx"),
                    str(blocker / "property.vnnlib"),
                    str(counterexample),
                ],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("ENVIRONMENT ERROR", result.stderr)
        self.assertNotIn("MOAT", result.stdout)


if __name__ == "__main__":
    unittest.main()
