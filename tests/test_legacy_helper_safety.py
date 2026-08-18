# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import gzip
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from scripts import export_docling_to_onnx, ny_measured_sweep, run_abcrown_baseline
from scripts.extended_bank import validate_reference_ces


def _write_executable(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _reference_ce_fixture(
    tmp_path: Path, *, measured_verdict: str = "unsat"
) -> tuple[Path, Path, Path]:
    benchmark_root = tmp_path / "benchmarks"
    benchmark_dir = benchmark_root / "fixture"
    model = benchmark_dir / "onnx/model.onnx"
    specification = benchmark_dir / "vnnlib/prop.vnnlib"
    model.parent.mkdir(parents=True)
    specification.parent.mkdir(parents=True)
    model.write_bytes(b"\x08\x01")
    specification.write_text("", encoding="utf-8")
    (benchmark_dir / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/prop.vnnlib,10\n", encoding="utf-8"
    )
    measured = tmp_path / "measured.csv"
    measured.write_text(
        f"fixture,onnx/model.onnx,vnnlib/prop.vnnlib,prepared,{measured_verdict},1.0\n",
        encoding="utf-8",
    )
    reference_root = tmp_path / "reference"
    for tool in validate_reference_ces.TOOLS:
        (reference_root / tool / "2025_fixture").mkdir(parents=True)
    counterexample_dir = reference_root / "alpha_beta_crown/2025_fixture"
    with gzip.open(
        counterexample_dir / "model_prop.counterexample.gz",
        "wt",
        encoding="utf-8",
    ) as handle:
        handle.write("sat\n((X_0 0.5))\n")
    return reference_root, benchmark_root, measured


def test_reference_ce_audit_reports_validated_false_unsat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_root, benchmark_root, measured = _reference_ce_fixture(tmp_path)
    monkeypatch.setattr(
        validate_reference_ces.vnnlib_ce,
        "validate",
        lambda *_args: (True, True, "validated fixture"),
    )

    summary, exit_code = validate_reference_ces.run_audit(
        "fixture", reference_root, benchmark_root, measured
    )

    assert exit_code == 3
    assert summary["complete"] is True
    assert summary["breach_count"] == 1
    assert summary["breaches"] == [
        {
            "tool": "alpha_beta_crown",
            "onnx": "model",
            "vnnlib": "prop",
            "detail": "validated fixture",
        }
    ]


def test_reference_ce_audit_fails_closed_on_malformed_counterexample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_root, benchmark_root, measured = _reference_ce_fixture(tmp_path)
    counterexample = (
        reference_root / "alpha_beta_crown/2025_fixture/model_prop.counterexample.gz"
    )
    with gzip.open(counterexample, "wt", encoding="utf-8") as handle:
        handle.write("sat\n((X_0 0.5)\n(X_0 0.6))\n")
    monkeypatch.setattr(
        validate_reference_ces.vnnlib_ce,
        "validate",
        lambda *_args: pytest.fail("malformed assignment reached validation"),
    )

    summary, exit_code = validate_reference_ces.run_audit(
        "fixture", reference_root, benchmark_root, measured
    )

    assert exit_code == 2
    assert summary["complete"] is False
    assert summary["errors"] == 1
    assert "duplicate assignment" in summary["issues"][0]


@pytest.mark.parametrize(
    ("validator_result", "expected_detail"),
    [
        (
            (False, False, "invalid property structure: malformed assertion"),
            "invalid property structure",
        ),
        (
            (False, False, "invalid input assertion: unsupported expression"),
            "invalid input assertion",
        ),
        (
            (True, False, "invalid output assertion: unsupported expression"),
            "invalid output assertion",
        ),
        (
            (False, False, "incomplete witness: missing X_1"),
            "incomplete witness",
        ),
        (
            (False, False, "non-finite witness values: X_0"),
            "non-finite witness values",
        ),
    ],
)
def test_reference_ce_audit_treats_validator_errors_as_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validator_result: tuple[bool, bool, str],
    expected_detail: str,
) -> None:
    reference_root, benchmark_root, measured = _reference_ce_fixture(
        tmp_path, measured_verdict="sat"
    )
    monkeypatch.setattr(
        validate_reference_ces.vnnlib_ce,
        "validate",
        lambda *_args: validator_result,
    )

    summary, exit_code = validate_reference_ces.run_audit(
        "fixture", reference_root, benchmark_root, measured
    )

    assert exit_code == 2
    assert summary["complete"] is False
    assert summary["errors"] == 1
    assert summary["tools"]["alpha_beta_crown"]["err"] == 1
    assert expected_detail in summary["issues"][0]


def test_reference_ce_audit_requires_a_measured_verdict_for_valid_ce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_root, benchmark_root, measured = _reference_ce_fixture(
        tmp_path, measured_verdict="unknown"
    )
    measured.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        validate_reference_ces.vnnlib_ce,
        "validate",
        lambda *_args: (True, True, "validated fixture"),
    )

    summary, exit_code = validate_reference_ces.run_audit(
        "fixture", reference_root, benchmark_root, measured
    )

    assert exit_code == 2
    assert summary["complete"] is False
    assert "no NY measured verdict" in summary["issues"][0]


def test_reference_ce_audit_fails_closed_on_missing_tool_category_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_root, benchmark_root, measured = _reference_ce_fixture(
        tmp_path, measured_verdict="sat"
    )
    (reference_root / "pyrat/2025_fixture").rmdir()
    monkeypatch.setattr(
        validate_reference_ces.vnnlib_ce,
        "validate",
        lambda *_args: (True, True, "validated fixture"),
    )

    summary, exit_code = validate_reference_ces.run_audit(
        "fixture", reference_root, benchmark_root, measured
    )

    assert exit_code == 2
    assert summary["complete"] is False
    assert summary["tools"]["pyrat"]["directory_present"] is False
    assert any(
        "pyrat: result directory is missing" in issue for issue in summary["issues"]
    )


def test_reference_ce_cli_requires_explicit_portable_reference_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VNNCOMP_RESULTS_ROOT", raising=False)

    with pytest.raises(SystemExit) as error:
        validate_reference_ces.main(["fixture", "--repo-root", str(tmp_path)])

    assert error.value.code == 2


def _sweep_fixture(tmp_path: Path, fake_ny_source: str) -> tuple[Path, Path, Path]:
    corpus = tmp_path / "corpus"
    benchmark = corpus / "fixture"
    model = benchmark / "onnx/model.onnx"
    specification = benchmark / "vnnlib/prop.vnnlib"
    model.parent.mkdir(parents=True)
    specification.parent.mkdir(parents=True)
    model.write_bytes(b"\x08\x01")
    specification.write_text("", encoding="utf-8")
    (benchmark / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/prop.vnnlib,10.0\n"
        "onnx/model.onnx,vnnlib/prop.vnnlib,3\n",
        encoding="utf-8",
    )
    ny = _write_executable(tmp_path / "bin/ny", fake_ny_source)
    output = tmp_path / "measured"
    return corpus, ny, output


def test_measured_sweep_uses_unique_results_and_caps_official_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation_log = tmp_path / "invocations.csv"
    corpus, ny, output = _sweep_fixture(
        tmp_path,
        (
            "#!/bin/sh\n"
            'printf \'%s,%s\\n\' "$6" "$7" >> "$SWEEP_INVOCATION_LOG"\n'
            "sleep 0.05\n"
            "printf 'unsat\\n' > \"$6\"\n"
        ),
    )
    monkeypatch.setenv("SWEEP_INVOCATION_LOG", str(invocation_log))

    exit_code = ny_measured_sweep.main(
        [
            "fixture",
            "--corpus",
            str(corpus),
            "--ny",
            str(ny),
            "--out",
            str(output),
            "--timeout",
            "5",
            "--workers",
            "2",
        ]
    )

    assert exit_code == 0
    invocations = list(
        csv.reader(invocation_log.read_text(encoding="utf-8").splitlines())
    )
    assert len({row[0] for row in invocations}) == 2
    assert sorted(row[1] for row in invocations) == ["3", "5"]
    with (output / "fixture.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert [row[4] for row in rows] == ["unsat", "unsat"]


def test_measured_sweep_rejects_nonzero_process_even_with_stale_verdict(
    tmp_path: Path,
) -> None:
    corpus, ny, output = _sweep_fixture(
        tmp_path,
        "#!/bin/sh\nprintf 'unsat\\n' > \"$6\"\nexit 7\n",
    )

    exit_code = ny_measured_sweep.main(
        [
            "fixture",
            "--corpus",
            str(corpus),
            "--ny",
            str(ny),
            "--out",
            str(output),
            "--limit",
            "1",
            "--workers",
            "1",
        ]
    )

    assert exit_code == 1
    with (output / "fixture.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.reader(handle))
    assert row[4] == "error"


def test_measured_sweep_never_scores_a_decision_returned_during_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "fixture").mkdir(parents=True)
    ny = tmp_path / "ny"
    ny.write_bytes(b"fixture")
    monotonic_values = iter((100.0, 111.0))
    monkeypatch.setattr(
        ny_measured_sweep.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    def fake_run(command, **_kwargs):
        Path(command[6]).write_text("unsat\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ny_measured_sweep.subprocess, "run", fake_run)

    status, elapsed = ny_measured_sweep.run_instance(
        ny,
        corpus,
        "fixture",
        "onnx/model.onnx",
        "vnnlib/prop.vnnlib",
        timeout=10,
        watchdog_grace=15,
    )

    assert status == "timeout"
    assert elapsed == 11.0


def test_abcrown_status_parser_rejects_substrings_and_conflicts() -> None:
    assert run_abcrown_baseline._parse_status("Result: unsat\n") == "unsat"
    assert run_abcrown_baseline._parse_status("VERIFIED\n") == "unsat"
    assert run_abcrown_baseline._parse_status("not verified\n") is None
    assert run_abcrown_baseline._parse_status("sat\nunsat\n") is None
    assert (
        run_abcrown_baseline._parse_result_status(
            "sat\nunsat\n",
            "Result: unsat\n",
        )
        is None
    ), "stdout must not mask a conflicting requested results artifact"


def test_abcrown_never_counts_a_decision_returned_after_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier_dir = tmp_path / "abcrown/complete_verifier"
    vnnlib_dir = tmp_path / "benchmark/vnnlib"
    vnnlib_dir.mkdir(parents=True)
    (vnnlib_dir / "prop_0.vnnlib").write_text("", encoding="utf-8")
    config = run_abcrown_baseline.BaselineConfig(
        abcrown_dir=tmp_path / "abcrown",
        python=tmp_path / "python",
        benchmark_dir=tmp_path / "benchmark",
        model=tmp_path / "benchmark/model.onnx",
        vnnlib_dir=vnnlib_dir,
        abcrown_config=verifier_dir / "config.yaml",
        device="cpu",
        property_template="prop_{index}.vnnlib",
    )
    monotonic_values = iter((100.0, 111.0))
    monkeypatch.setattr(
        run_abcrown_baseline.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    class FakeProcess:
        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is True
            result_path = Path(command[command.index("--results_file") + 1])
            result_path.write_text("unsat\n", encoding="utf-8")
            self.pid = 12345
            self.returncode = 0

        def communicate(self, timeout=None):
            assert timeout == 70.0
            return "", ""

    monkeypatch.setattr(run_abcrown_baseline.subprocess, "Popen", FakeProcess)

    result = run_abcrown_baseline.run_instance(config, 0, timeout=10)

    assert result["status"] == "timeout"
    assert result["time"] == 11.0


def test_abcrown_timeout_terminates_and_reaps_the_worker_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier_dir = tmp_path / "abcrown/complete_verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "abcrown.py").write_text("# fixture\n", encoding="utf-8")
    vnnlib_dir = tmp_path / "benchmark/vnnlib"
    vnnlib_dir.mkdir(parents=True)
    (vnnlib_dir / "prop_0.vnnlib").write_text("", encoding="utf-8")
    pid_file = tmp_path / "worker-pids"
    fake_python = _write_executable(
        tmp_path / "fake-python",
        (
            "#!/bin/sh\n"
            "trap '' TERM\n"
            "sleep 30 &\n"
            "child=$!\n"
            "printf '%s %s\\n' \"$$\" \"$child\" > \"$ABCROWN_PID_FILE\"\n"
            "wait \"$child\"\n"
        ),
    )
    monkeypatch.setenv("ABCROWN_PID_FILE", str(pid_file))
    config = run_abcrown_baseline.BaselineConfig(
        abcrown_dir=tmp_path / "abcrown",
        python=fake_python,
        benchmark_dir=tmp_path / "benchmark",
        model=tmp_path / "benchmark/model.onnx",
        vnnlib_dir=vnnlib_dir,
        abcrown_config=verifier_dir / "config.yaml",
        device="cpu",
        property_template="prop_{index}.vnnlib",
    )

    result = run_abcrown_baseline.run_instance(
        config,
        0,
        timeout=1,
        watchdog_grace=0.1,
        termination_grace=0.2,
    )

    assert result["status"] == "timeout"
    pids = [int(value) for value in pid_file.read_text(encoding="utf-8").split()]
    assert len(pids) == 2
    remaining = set(pids)
    deadline = time.monotonic() + 2.0
    while remaining and time.monotonic() < deadline:
        for pid in tuple(remaining):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                remaining.remove(pid)
        if remaining:
            time.sleep(0.01)
    assert not remaining, f"timed-out verifier processes survived: {sorted(remaining)}"


def test_abcrown_baseline_uses_explicit_external_checkout_and_atomic_output(
    tmp_path: Path,
) -> None:
    abcrown = tmp_path / "alpha beta CROWN"
    verifier = abcrown / "complete_verifier"
    (verifier / "exp_configs/vnncomp21").mkdir(parents=True)
    (verifier / "abcrown.py").write_text("# fixture\n", encoding="utf-8")
    (verifier / "exp_configs/vnncomp21/cifar10-resnet.yaml").write_text(
        "fixture: true\n", encoding="utf-8"
    )
    fake_python = _write_executable(
        tmp_path / "bin/fake python",
        (
            "#!/bin/sh\n"
            'previous=""\n'
            'for argument in "$@"; do\n'
            '    if [ "$previous" = "--results_file" ]; then\n'
            "        printf 'unsat\\n' > \"$argument\"\n"
            "    fi\n"
            '    previous="$argument"\n'
            "done\n"
        ),
    )
    benchmark = tmp_path / "benchmark"
    (benchmark / "onnx").mkdir(parents=True)
    (benchmark / "vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered").mkdir(
        parents=True
    )
    (benchmark / "onnx/resnet_2b.onnx").write_bytes(b"\x08\x01")
    (
        benchmark / "vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/"
        "prop_0_eps_0.008.vnnlib"
    ).write_text("", encoding="utf-8")
    output = tmp_path / "result output/baseline.json"

    exit_code = run_abcrown_baseline.main(
        [
            "--abcrown-dir",
            str(abcrown),
            "--python",
            str(fake_python),
            "--benchmark-dir",
            str(benchmark),
            "--max-instances",
            "1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "abcrown_supplementary_baseline_v1"
    assert payload["research_only"] is True
    assert payload["results"][0]["status"] == "unsat"


def test_docling_missing_dependencies_never_runs_installer_or_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "DocumentFigureClassifier").mkdir()
    monkeypatch.setattr(
        export_docling_to_onnx,
        "_load_export_dependencies",
        lambda: (_ for _ in ()).throw(
            export_docling_to_onnx.ExportError("fixture missing dependencies")
        ),
    )
    monkeypatch.setattr(
        export_docling_to_onnx.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "unexpected package installer/subprocess"
        ),
    )

    exit_code = export_docling_to_onnx.main(
        [
            "--model",
            "DocumentFigureClassifier",
            "--models-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 2


def test_docling_optimum_export_uses_argument_vector_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model; touch SHOULD_NOT_EXIST"
    model_path.mkdir()
    observed: list[str] = []
    validated: list[Path] = []

    class FakeChecker:
        @staticmethod
        def check_model(_model) -> None:
            return None

    class FakeOnnx:
        checker = FakeChecker

        @staticmethod
        def load(path: str):
            validated.append(Path(path))
            return object()

    def fake_run(arguments, *, check):
        assert check is False
        observed.extend(arguments)
        output = Path(arguments[-1])
        output.mkdir()
        (output / "model.onnx").write_bytes(b"\x08\x01")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(export_docling_to_onnx.subprocess, "run", fake_run)

    output = export_docling_to_onnx.export_with_optimum(
        model_path, "optimum cli with spaces", FakeOnnx
    )

    assert output == model_path / "onnx"
    assert observed[:-1] == [
        "optimum cli with spaces",
        "export",
        "onnx",
        "--model",
        str(model_path),
    ]
    staged_output = Path(observed[-1])
    assert staged_output.name == "onnx"
    assert staged_output.parent.parent == model_path
    assert staged_output.parent.name.startswith(".onnx-export-")
    assert output.is_dir()
    assert [path.name for path in validated] == ["model.onnx"]
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_docling_optimum_refuses_stale_output_without_running_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    stale_output = model_path / "onnx"
    stale_output.mkdir(parents=True)
    (stale_output / "old.onnx").write_bytes(b"old")
    monkeypatch.setattr(
        export_docling_to_onnx.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("stale output must stop before export"),
    )

    with pytest.raises(
        export_docling_to_onnx.ExportError, match="refusing to overwrite"
    ):
        export_docling_to_onnx.export_with_optimum(model_path, "optimum-cli", object())

    assert (stale_output / "old.onnx").read_bytes() == b"old"


def test_docling_optimum_rejects_invalid_onnx_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    def fake_run(arguments, *, check):
        assert check is False
        staged_output = Path(arguments[-1])
        staged_output.mkdir()
        (staged_output / "invalid.onnx").write_bytes(b"not an ONNX model")
        return subprocess.CompletedProcess(arguments, 0)

    class RejectingOnnx:
        @staticmethod
        def load(_path: str):
            raise ValueError("invalid protobuf")

    monkeypatch.setattr(export_docling_to_onnx.subprocess, "run", fake_run)

    with pytest.raises(export_docling_to_onnx.ExportError, match="invalid ONNX export"):
        export_docling_to_onnx.export_with_optimum(
            model_path, "optimum-cli", RejectingOnnx
        )

    assert not (model_path / "onnx").exists()
    assert not list(model_path.glob(".onnx-export-*"))


def test_docling_vlm_export_requires_explicit_code_execution_opt_in(
    tmp_path: Path,
) -> None:
    with pytest.raises(export_docling_to_onnx.ExportError, match="trust-remote-code"):
        export_docling_to_onnx.export_vlm_encoder(
            tmp_path,
            object(),
            object(),
            trust_remote_code=False,
        )


def test_benchmark_diff_prints_a_runnable_docling_export_recipe() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts/benchmark_diff.sh"
    ).read_text(encoding="utf-8")

    assert (
        "export_docling_to_onnx.py --model granite-docling-258M --trust-remote-code"
    ) in script
    assert ("models/docling/granite-docling-258M/vision_encoder.onnx") in script
    assert "models/docling/granite-docling-258M/model.onnx" not in script
