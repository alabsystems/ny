# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_biccos_mts_factorial.py"
MANIFEST = ROOT / "benchmarks" / "biccos_mts_factorial_v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("biccos_mts_planner", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_is_diagnostic_only_and_has_exact_factorial():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["diagnostic_only"] is True
    assert payload["execution_allowed"] is False
    assert payload["targets"] == [
        "cifar100-medium-1761",
        "cifar100-medium-2477",
        "tinyimagenet-medium-1126",
        "tinyimagenet-medium-7943",
    ]
    assert [arm["id"] for arm in payload["arms"]] == [
        "baseline",
        "mts-only",
        "cs-only",
        "all",
    ]
    assert [
        (
            arm["id"],
            arm["biccos"],
            arm["multi_tree"],
            arm["constraint_strengthening"],
        )
        for arm in payload["arms"]
    ] == [
        ("baseline", False, False, False),
        ("mts-only", True, True, False),
        ("cs-only", True, False, True),
        ("all", True, True, True),
    ]


def test_committed_corpus_rows_have_exact_indices_and_asset_hashes():
    payload = json.loads(
        (ROOT / "benchmarks" / "abcrown_transfer_corpus_v1.json").read_text(
            encoding="utf-8"
        )
    )
    entries = {entry["id"]: entry for entry in payload["entries"]}
    expected = {
        "cifar100-medium-1761": (
            52,
            "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4",
            "f2a5e14de263f19a36d06a2200197e111f3cb7467eaf0f524edb09e3253b667b",
        ),
        "cifar100-medium-2477": (
            14,
            "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4",
            "f7832a361605ea8187e1abd4956ff0b1e0c67bd1b426df938e31f094f5383635",
        ),
        "tinyimagenet-medium-1126": (
            1,
            "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776",
            "9497c3bdd8ade3804cfd9fe8d415941a18049753e31eede0adbf4260cdc280c1",
        ),
        "tinyimagenet-medium-7943": (
            3,
            "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776",
            "27992347f22d80750a97dd4b5c3d602732d0977c7681a70d2109e0a27c9ee466",
        ),
    }
    assert {
        entry_id: (
            entries[entry_id]["source_index"],
            entries[entry_id]["expected"]["model_sha256"],
            entries[entry_id]["expected"]["property_sha256"],
        )
        for entry_id in expected
    } == expected


def test_wrong_abc_pin_fails_closed(monkeypatch, tmp_path):
    module = _load_module()
    abc_repo = tmp_path / "abc"
    abc_repo.mkdir()
    observed = {
        ("rev-parse", "HEAD"): "0" * 40,
    }

    def fake_git(_repo, *arguments):
        return observed[arguments]

    monkeypatch.setattr(module, "_git", fake_git)
    with pytest.raises(module.PlanError, match="pin mismatch"):
        module._validate_clean_pin(abc_repo, "1" * 40, "2" * 40)


def test_dirty_abc_checkout_fails_closed(monkeypatch, tmp_path):
    module = _load_module()
    abc_repo = tmp_path / "abc"
    abc_repo.mkdir()

    def fake_git(_repo, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return "1" * 40
        if arguments[0] == "status":
            return " M complete_verifier/bab.py"
        raise AssertionError(arguments)

    monkeypatch.setattr(module, "_git", fake_git)
    with pytest.raises(module.PlanError, match="not clean"):
        module._validate_clean_pin(abc_repo, "1" * 40, "2" * 40)


def test_manifest_cannot_enable_execution(tmp_path):
    module = _load_module()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["execution_allowed"] = True
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(module.PlanError, match="forbid execution"):
        module._load_manifest(path)
