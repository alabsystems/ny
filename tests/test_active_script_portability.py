# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0 OR MIT

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.extended_bank import validate_sat_rows


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "extended_bank" / "validate_sat_rows.py"
RESNET_SCRIPT = REPO_ROOT / "scripts" / "resnet_gpu_sweep.sh"
MEASURE_SCRIPT = REPO_ROOT / "scripts" / "vnncomp_sat_measure.sh"
ACTIVE_SCRIPTS = (VALIDATE_SCRIPT, RESNET_SCRIPT, MEASURE_SCRIPT)


def _write_executable(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_ny(tmp_path: Path) -> Path:
    return _write_executable(
        tmp_path / "bin" / "ny",
        textwrap.dedent(
            """\
            #!/bin/sh
            if [ "${FAKE_NY_EXIT:-0}" -ne 0 ]; then
                exit "$FAKE_NY_EXIT"
            fi
            printf '%s\\n' "${FAKE_NY_VERDICT-sat}" > "$6"
            """
        ),
    )


def _base_environment(tmp_path: Path, ny_bin: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "NY_ROOT": str(tmp_path),
            "NY_BIN": str(ny_bin),
            "TMPDIR": str(tmp_path / "tmp"),
        }
    )
    (tmp_path / "tmp").mkdir(exist_ok=True)
    return environment


def _write_vnncomp_fixture(tmp_path: Path) -> tuple[Path, Path]:
    category = "fixture"
    category_dir = (
        tmp_path / "benchmarks" / "vnncomp2025" / "benchmarks" / category
    )
    model = category_dir / "onnx" / "model.onnx"
    specification = category_dir / "vnnlib" / "property.vnnlib"
    model.parent.mkdir(parents=True)
    specification.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    specification.write_text("", encoding="utf-8")
    (category_dir / "instances.csv").write_text(
        "onnx/model.onnx,vnnlib/property.vnnlib,1\n", encoding="utf-8"
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("property.vnnlib,sat\n", encoding="utf-8")
    return labels, category_dir


def _write_resnet_fixture(tmp_path: Path) -> Path:
    category_dir = (
        tmp_path
        / "benchmarks"
        / "vnncomp2025"
        / "benchmarks"
        / "cifar100_2024"
    )
    model = category_dir / "onnx" / "CIFAR100_resnet_medium.onnx"
    specification = category_dir / "vnnlib" / "property.vnnlib"
    model.parent.mkdir(parents=True)
    specification.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    specification.write_text("", encoding="utf-8")
    labels = tmp_path / "resnet-labels.csv"
    labels.write_text("property.vnnlib,sat\n", encoding="utf-8")
    return labels


def test_active_scripts_have_no_machine_local_paths() -> None:
    for script in ACTIVE_SCRIPTS:
        source = script.read_text(encoding="utf-8")
        assert "/Users/" not in source, script
        assert "/private/tmp/" not in source, script


@pytest.mark.parametrize(
    (
        "valid",
        "out_of_box",
        "errored",
        "other",
        "not_reproduced",
        "expected_code",
        "expected_text",
    ),
    [
        (1, 0, 0, 0, 0, 0, "SOUND:"),
        (0, 0, 0, 0, 0, 1, "INCONCLUSIVE:"),
        (1, 1, 0, 0, 0, 1, "OUT-OF-BOX"),
        (1, 0, 1, 0, 0, 2, "ENVIRONMENT/ERROR:"),
        (0, 0, 1, 0, 0, 2, "ENVIRONMENT/ERROR:"),
        (1, 0, 0, 1, 0, 1, "INCONCLUSIVE:"),
        (1, 0, 0, 0, 1, 1, "INCONCLUSIVE:"),
    ],
)
def test_sat_audit_conclusion_fails_closed_on_unvalidated_reproduction(
    valid: int,
    out_of_box: int,
    errored: int,
    other: int,
    not_reproduced: int,
    expected_code: int,
    expected_text: str,
) -> None:
    conclusion, exit_code = validate_sat_rows._audit_conclusion(
        valid, out_of_box, errored, other, not_reproduced
    )
    assert exit_code == expected_code
    assert expected_text in conclusion


def test_sat_audit_witness_pattern_keeps_signed_exponents() -> None:
    matches = validate_sat_rows.RAW_ASSIGNMENT.findall(
        "sat\n((X_0 -1.5e+03)\n(X_1 2E-4))\n"
    )
    assert matches == [("0", "-1.5e+03"), ("1", "2E-4")]


def test_sat_audit_input_resolution_cannot_escape_benchmark_root(
    tmp_path: Path,
) -> None:
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    resolved = validate_sat_rows._resolve_benchmark_input(
        bench_dir, "../outside.onnx"
    )
    assert resolved != outside.resolve()
    resolved.relative_to(bench_dir.resolve())


def test_validate_sat_rows_derives_binary_from_overridden_root(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["NY_ROOT"] = str(tmp_path)
    environment.pop("NY_BIN", None)
    environment.pop("NY_AY", None)
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "fixture", "0", "0"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert str(tmp_path / "target" / "release" / "ny") in result.stderr


def test_validate_sat_rows_honors_explicit_ay_binary(tmp_path: Path) -> None:
    ny_bin = _fake_ny(tmp_path)
    missing_ay = tmp_path / "custom" / "ay"
    environment = _base_environment(tmp_path, ny_bin)
    environment["NY_AY"] = str(missing_ay)
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "fixture", "0", "0"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert f"AY binary is missing or not executable: {missing_ay}" in result.stderr


def test_validate_sat_rows_fails_closed_on_missing_row_input(tmp_path: Path) -> None:
    ny_bin = _fake_ny(tmp_path)
    ay_bin = _write_executable(tmp_path / "bin" / "ay", "#!/bin/sh\nexit 0\n")
    category_dir = (
        tmp_path / "benchmarks" / "vnncomp2025" / "benchmarks" / "fixture"
    )
    category_dir.mkdir(parents=True)
    (category_dir / "instances.csv").write_text(
        "onnx/missing.onnx,vnnlib/missing.vnnlib,1\n", encoding="utf-8"
    )
    measured = tmp_path / "reports" / "measured" / "fixture.csv"
    measured.parent.mkdir(parents=True)
    measured.write_text(
        "row,onnx/missing.onnx,vnnlib/missing.vnnlib,0,sat\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.csv"
    environment = _base_environment(tmp_path, ny_bin)
    environment["NY_AY"] = str(ay_bin)
    environment["NY_SAT_AUDIT_OUT"] = str(output)
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "fixture", "0", "1"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "ONNX model is missing" in result.stderr
    assert not output.exists()


def test_validate_sat_rows_process_error_is_inconclusive(tmp_path: Path) -> None:
    ny_bin = _fake_ny(tmp_path)
    ay_bin = _write_executable(tmp_path / "bin" / "ay", "#!/bin/sh\nexit 0\n")
    _labels, category_dir = _write_vnncomp_fixture(tmp_path)
    measured = tmp_path / "reports" / "measured" / "fixture.csv"
    measured.parent.mkdir(parents=True)
    measured.write_text(
        "row,onnx/model.onnx,vnnlib/property.vnnlib,0,sat\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.csv"
    environment = _base_environment(tmp_path, ny_bin)
    environment.update(
        {
            "FAKE_NY_EXIT": "3",
            "NY_AY": str(ay_bin),
            "NY_SAT_AUDIT_OUT": str(output),
        }
    )
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), category_dir.name, "0", "1"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert "not-reproduced=1" in result.stdout
    assert "INCONCLUSIVE:" in result.stdout
    assert "not-repro" in output.read_text(encoding="utf-8")


def test_validate_sat_rows_missing_runtime_dependency_aborts_before_rerun(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "ny-was-run"
    ny_bin = _write_executable(
        tmp_path / "bin" / "ny",
        f'#!/bin/sh\ntouch "{marker}"\nprintf \'sat\\n\' > "$6"\n',
    )
    ay_bin = _write_executable(tmp_path / "bin" / "ay", "#!/bin/sh\nexit 0\n")
    _labels, category_dir = _write_vnncomp_fixture(tmp_path)
    measured = tmp_path / "reports" / "measured" / "fixture.csv"
    measured.parent.mkdir(parents=True)
    measured.write_text(
        "row,onnx/model.onnx,vnnlib/property.vnnlib,0,sat\n",
        encoding="utf-8",
    )
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "numpy.py").write_text(
        "raise ImportError('numpy is blocked for this test')\n", encoding="utf-8"
    )
    output = tmp_path / "audit.csv"
    environment = _base_environment(tmp_path, ny_bin)
    environment.update(
        {
            "NY_AY": str(ay_bin),
            "NY_SAT_AUDIT_OUT": str(output),
            "PYTHONPATH": str(blocker),
        }
    )
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), category_dir.name, "0", "1"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "not importable" in result.stderr
    assert not marker.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    ("script", "arguments", "prefix"),
    [
        (RESNET_SCRIPT, ["0", "0"], "resnet_gpu_sweep"),
        (MEASURE_SCRIPT, ["fixture", "0", "0", "0"], "vnncomp_sat_measure"),
    ],
)
def test_shell_audits_fail_closed_when_derived_binary_is_missing(
    tmp_path: Path, script: Path, arguments: list[str], prefix: str
) -> None:
    environment = dict(os.environ)
    environment["NY_ROOT"] = str(tmp_path)
    environment.pop("NY_BIN", None)
    result = subprocess.run(
        ["bash", str(script), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr.startswith(f"{prefix}: error:")
    assert str(tmp_path / "target" / "release" / "ny") in result.stderr


@pytest.mark.parametrize("verdict, expected_code", [("sat", 0), ("unsat", 1)])
def test_resnet_sweep_uses_overrides_and_fails_on_wrong_verdict(
    tmp_path: Path, verdict: str, expected_code: int
) -> None:
    ny_bin = _fake_ny(tmp_path)
    labels = _write_resnet_fixture(tmp_path)
    environment = _base_environment(tmp_path, ny_bin)
    environment["FAKE_NY_VERDICT"] = verdict
    result = subprocess.run(
        ["bash", str(RESNET_SCRIPT), "0", "1", str(labels)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected_code, result.stderr
    assert "instances=1" in result.stdout
    assert f"wrong={int(verdict == 'unsat')}" in result.stdout


@pytest.mark.parametrize("verdict, expected_code", [("sat", 0), ("unsat", 1)])
def test_vnncomp_measure_uses_overrides_and_fails_on_wrong_verdict(
    tmp_path: Path, verdict: str, expected_code: int
) -> None:
    ny_bin = _fake_ny(tmp_path)
    labels, _category_dir = _write_vnncomp_fixture(tmp_path)
    environment = _base_environment(tmp_path, ny_bin)
    environment["FAKE_NY_VERDICT"] = verdict
    result = subprocess.run(
        [
            "bash",
            str(MEASURE_SCRIPT),
            "fixture",
            "0",
            "1",
            "0",
            str(labels),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected_code, result.stderr
    assert f"sat_solved={int(verdict == 'sat')}/1" in result.stdout
    assert f"WRONG={int(verdict == 'unsat')}" in result.stdout


@pytest.mark.parametrize(
    ("script", "arguments", "fixture"),
    [
        (RESNET_SCRIPT, ["0", "1"], _write_resnet_fixture),
        (MEASURE_SCRIPT, ["fixture", "0", "1", "0"], _write_vnncomp_fixture),
    ],
)
def test_shell_audits_fail_closed_on_ny_process_error(
    tmp_path: Path,
    script: Path,
    arguments: list[str],
    fixture,
) -> None:
    ny_bin = _fake_ny(tmp_path)
    fixture_result = fixture(tmp_path)
    labels = fixture_result[0] if isinstance(fixture_result, tuple) else fixture_result
    environment = _base_environment(tmp_path, ny_bin)
    environment["FAKE_NY_EXIT"] = "3"
    result = subprocess.run(
        ["bash", str(script), *arguments, str(labels)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert "errors=1" in result.stdout


@pytest.mark.parametrize("verdict", ["", "garbage"])
@pytest.mark.parametrize(
    ("script", "arguments", "fixture"),
    [
        (RESNET_SCRIPT, ["0", "1"], _write_resnet_fixture),
        (MEASURE_SCRIPT, ["fixture", "0", "1", "0"], _write_vnncomp_fixture),
    ],
)
def test_shell_audits_fail_closed_on_invalid_ny_verdict(
    tmp_path: Path,
    verdict: str,
    script: Path,
    arguments: list[str],
    fixture,
) -> None:
    ny_bin = _fake_ny(tmp_path)
    fixture_result = fixture(tmp_path)
    labels = fixture_result[0] if isinstance(fixture_result, tuple) else fixture_result
    environment = _base_environment(tmp_path, ny_bin)
    environment["FAKE_NY_VERDICT"] = verdict
    result = subprocess.run(
        ["bash", str(script), *arguments, str(labels)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert "errors=1" in result.stdout
