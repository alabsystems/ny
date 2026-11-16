# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

"""Hermetic AyProc command discovery and successful-version identity controls."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_ay_binary_matches_pin.py"
PIN = "1" * 40
OTHER = "2" * 40
VERSION = f"ay 0.28.0+build.1.{PIN}@2026-11-15T00:00:00Z"


@pytest.fixture
def checker(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("_ny_ay_binary_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "pinned_ay_rev", lambda: PIN)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--json"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "path").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    monkeypatch.delenv("NY_AY", raising=False)

    def no_native_command(*_args, **_kwargs):
        pytest.fail("test attempted an unmocked subprocess")

    monkeypatch.setattr(module.subprocess, "run", no_native_command)
    return module


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture only: must never execute\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def mock_version(monkeypatch, checker, binary, *, returncode=0, version=VERSION):
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments)
        assert arguments == [str(binary), "--version"]
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "timeout": 30,
            "check": False,
        }
        return subprocess.CompletedProcess(arguments, returncode, version + "\n", "")

    monkeypatch.setattr(checker.subprocess, "run", run)
    return calls


def report(checker, capsys):
    result = checker.main()
    return result, json.loads(capsys.readouterr().out)


def test_unset_override_and_missing_default_is_absent(checker, capsys):
    result, value = report(checker, capsys)
    assert result == 0
    assert value["status"] == "absent"
    assert value["ny_ay_requested"] is None
    assert value["binary"] is None


def test_empty_override_refuses_even_when_default_is_available(
    checker, monkeypatch, tmp_path, capsys
):
    executable(tmp_path / "path/ay")
    monkeypatch.setenv("NY_AY", "")
    result, value = report(checker, capsys)
    assert result == 1
    assert value["status"] == "error"
    assert value["ny_ay_requested"] == ""
    assert value["binary"] is None
    assert "$NY_AY=''" in value["detail"]


def test_unset_override_uses_path_not_unlisted_cwd(
    checker, monkeypatch, tmp_path, capsys
):
    executable(tmp_path / "ay")
    selected = executable(tmp_path / "path/ay")
    calls = mock_version(monkeypatch, checker, selected)
    result, value = report(checker, capsys)
    assert result == 0 and value["status"] == "ok"
    assert value["binary"] == str(selected)
    assert value["ny_ay_requested"] is None
    assert calls == [[str(selected), "--version"]]


def test_bare_override_uses_path_not_same_named_cwd_or_default(
    checker, monkeypatch, tmp_path, capsys
):
    executable(tmp_path / "custom-ay")
    executable(tmp_path / "path/ay")
    selected = executable(tmp_path / "path/custom-ay")
    monkeypatch.setenv("NY_AY", "custom-ay")
    mock_version(monkeypatch, checker, selected)
    result, value = report(checker, capsys)
    assert result == 0 and value["status"] == "ok"
    assert value["binary"] == str(selected)
    assert value["ny_ay_requested"] == "custom-ay"


def test_relative_override_with_separator_uses_invoking_cwd(
    checker, monkeypatch, tmp_path, capsys
):
    executable(tmp_path / "tools/ay")
    executable(tmp_path / "path/tools/ay")
    monkeypatch.setenv("NY_AY", "tools/ay")
    mock_version(monkeypatch, checker, "tools/ay")
    result, value = report(checker, capsys)
    assert result == 0 and value["status"] == "ok"
    assert value["binary"] == "tools/ay"
    assert value["ny_ay_requested"] == "tools/ay"


def test_missing_relative_override_does_not_search_path(
    checker, monkeypatch, tmp_path, capsys
):
    executable(tmp_path / "path/tools/ay")
    executable(tmp_path / "path/ay")
    monkeypatch.setenv("NY_AY", "tools/ay")
    result, value = report(checker, capsys)
    assert result == 1 and value["status"] == "error"
    assert value["binary"] is None


def test_absolute_override_uses_named_file(
    checker, monkeypatch, tmp_path, capsys
):
    selected = executable(tmp_path / "explicit/ay")
    executable(tmp_path / "path/ay")
    monkeypatch.setenv("NY_AY", str(selected))
    mock_version(monkeypatch, checker, selected)
    result, value = report(checker, capsys)
    assert result == 0 and value["status"] == "ok"
    assert value["binary"] == str(selected)


@pytest.mark.parametrize("requested", ["missing-ay", "ay --version", "~/ay", "$AY_BIN"])
def test_invalid_explicit_command_never_falls_back(
    checker, monkeypatch, tmp_path, capsys, requested
):
    executable(tmp_path / "path/ay")
    monkeypatch.setenv("AY_BIN", str(tmp_path / "path/ay"))
    monkeypatch.setenv("NY_AY", requested)
    result, value = report(checker, capsys)
    assert result == 1 and value["status"] == "error"
    assert value["ny_ay_requested"] == requested
    assert value["binary"] is None


@pytest.mark.parametrize("kind", ["nonexecutable", "directory", "broken_symlink"])
def test_explicit_unlaunchable_path_is_an_error(
    checker, monkeypatch, tmp_path, capsys, kind
):
    selected = tmp_path / "invalid"
    if kind == "nonexecutable":
        selected.write_text("not executable\n", encoding="utf-8")
        selected.chmod(0o644)
    elif kind == "directory":
        selected.mkdir()
    else:
        selected.symlink_to(tmp_path / "missing")
    executable(tmp_path / "path/ay")
    monkeypatch.setenv("NY_AY", str(selected))
    result, value = report(checker, capsys)
    assert result == 1 and value["status"] == "error"
    assert value["binary"] is None
    assert "does not resolve to an executable" in value["detail"]


def test_warn_only_preserves_invalid_override_status(checker, monkeypatch, capsys):
    monkeypatch.setenv("NY_AY", "")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--json", "--warn-only"])
    result, value = report(checker, capsys)
    assert result == 0
    assert value["status"] == "error" and value["ny_ay_requested"] == ""


def test_matching_sha_from_failed_version_command_does_not_pass(
    checker, monkeypatch, tmp_path, capsys
):
    selected = executable(tmp_path / "path/ay")
    mock_version(monkeypatch, checker, selected, returncode=7)
    result, value = report(checker, capsys)
    assert result == 1 and value["status"] == "unknown"
    assert value["binary_sha"] is None
    assert value["binary_version"] == VERSION
    assert "did not establish a successful build sha" in value["detail"]
    assert PIN in value["detail"]


@pytest.mark.parametrize(
    ("version", "status", "sha"),
    [
        ("ay 0.28.0 without revision", "unknown", None),
        (VERSION.replace(PIN, OTHER), "mismatch", OTHER),
    ],
)
def test_successful_version_still_requires_matching_pin(
    checker, monkeypatch, tmp_path, capsys, version, status, sha
):
    selected = executable(tmp_path / "path/ay")
    mock_version(monkeypatch, checker, selected, version=version)
    result, value = report(checker, capsys)
    assert result == 1 and value["status"] == status
    assert value["binary_sha"] == sha
    assert value["binary_version"] == version
