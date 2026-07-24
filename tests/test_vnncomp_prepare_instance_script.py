# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE_SCRIPT = REPO_ROOT / "vnncomp_scripts" / "prepare_instance.sh"


def _install_fixture(tmp_path: Path) -> tuple[Path, Path]:
    script = tmp_path / "vnncomp_scripts" / "prepare_instance.sh"
    script.parent.mkdir(parents=True)
    script.write_bytes(PREPARE_SCRIPT.read_bytes())
    script.chmod(0o755)

    marker = tmp_path / "ny-was-invoked"
    ny = tmp_path / "target" / "release" / "ny"
    ny.parent.mkdir(parents=True)
    ny.write_text(
        f"#!/bin/sh\ntouch '{marker}'\nexit 99\n",
        encoding="utf-8",
    )
    ny.chmod(0o755)
    return script, marker


def _run(script: Path, onnx_arg: str, vnnlib: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script), "v1", "fixture", onnx_arg, str(vnnlib)],
        cwd=script.parent.parent,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_prepare_validates_regular_paths_without_analyzing_instance(
    tmp_path: Path,
) -> None:
    script, marker = _install_fixture(tmp_path)
    onnx = tmp_path / "model.onnx"
    vnnlib = tmp_path / "property.vnnlib"
    onnx.touch()
    vnnlib.touch()

    result = _run(script, str(onnx), vnnlib)

    assert result.returncode == 0, result.stderr
    assert "no instance analysis performed" in result.stdout
    assert not marker.exists(), "prepare_instance must not invoke the verifier"


def test_prepare_accepts_vnnlib2_multi_network_literal(tmp_path: Path) -> None:
    script, marker = _install_fixture(tmp_path)
    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    vnnlib = tmp_path / "property.vnnlib"
    first.touch()
    second.touch()
    vnnlib.touch()
    onnx_arg = repr([("f", str(first)), ("g", str(second))])

    result = _run(script, onnx_arg, vnnlib)

    assert result.returncode == 0, result.stderr
    assert not marker.exists(), "prepare_instance must not invoke the verifier"


def test_prepare_rejects_missing_path_inside_multi_network_literal(
    tmp_path: Path,
) -> None:
    script, marker = _install_fixture(tmp_path)
    existing = tmp_path / "first.onnx"
    missing = tmp_path / "missing.onnx"
    vnnlib = tmp_path / "property.vnnlib"
    existing.touch()
    vnnlib.touch()
    onnx_arg = repr([("f", str(existing)), ("g", str(missing))])

    result = _run(script, onnx_arg, vnnlib)

    assert result.returncode != 0
    assert "ONNX file not found" in result.stderr
    assert not marker.exists(), "prepare_instance must not invoke the verifier"
