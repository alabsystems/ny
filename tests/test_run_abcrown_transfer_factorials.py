# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_abcrown_transfer_factorials.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("abcrown_factorials", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deep_set_builds_nested_mapping_and_rejects_scalar_crossing():
    module = _load_module()
    document = {}
    module._deep_set(document, "bab.branching.candidates", 7)
    assert document == {"bab": {"branching": {"candidates": 7}}}

    document = {"bab": 3}
    try:
        module._deep_set(document, "bab.batch_size", 256)
    except module.ManifestError as error:
        assert "non-mapping" in str(error)
    else:
        raise AssertionError("scalar crossing must be rejected")


def test_materialized_preset_applies_overrides_and_absolutizes_root(tmp_path):
    module = _load_module()
    base_dir = tmp_path / "configs"
    base_dir.mkdir()
    base = base_dir / "base.yaml"
    base.write_text(
        "general:\n  root_path: ../benchmarks/category\n"
        "bab:\n  batch_size: 64\n",
        encoding="utf-8",
    )
    destination = tmp_path / "artifacts" / "preset.yaml"
    module._materialize_arm(
        base_preset=base,
        arm={"name": "treatment", "overrides": {"bab.batch_size": 256}},
        destination=destination,
    )
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert payload["bab"]["batch_size"] == 256
    assert payload["general"]["root_path"] == str(
        (base_dir / "../benchmarks/category").resolve()
    )


def test_committed_manifest_dry_run_materializes_every_arm(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    execution = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
    assert execution["dry_run"] is True
    assert len(execution["arms"]) == 33
    assert {arm["experiment"] for arm in execution["arms"]} == {
        "cgan_recipe",
        "cifar100_recipe",
        "tinyimagenet_recipe",
        "vgg_recipe",
    }
    for arm in execution["arms"]:
        assert Path(arm["generated_preset"]).is_file()
        assert len(arm["generated_preset_sha256"]) == 64
        assert "--preset" in arm["command"]
        assert arm["corpus_ids"]
        assert arm["resolved_zero_based_indices"]
    skip_arms = [
        arm for arm in execution["arms"] if arm["arm"] == "skip_final_alpha_gradient"
    ]
    assert len(skip_arms) == 4
    assert all(
        arm["env"] == {"NY_ALPHA_FINAL_BOUND_ONLY": "1"}
        for arm in skip_arms
    )
    packed_arms = [
        arm for arm in execution["arms"] if arm["arm"] == "packed_graph_alpha_queue"
    ]
    assert len(packed_arms) == 2
    assert all(
        arm["env"] == {"NY_PACKED_GRAPH_ALPHA_QUEUE": "1"}
        for arm in packed_arms
    )
    adaptive_arms = [
        arm
        for arm in execution["arms"]
        if arm["arm"] == "adaptive_microbatch_controller"
    ]
    assert len(adaptive_arms) == 2
    assert all(
        arm["env"] == {"NY_ADAPTIVE_MICROBATCH_CONTROLLER": "1"}
        and arm["overrides"]["bab.auto_enlarge_batch_size"] is True
        for arm in adaptive_arms
    )

    by_experiment = {}
    for arm in execution["arms"]:
        by_experiment.setdefault(
            arm["experiment"], arm["resolved_zero_based_indices"]
        )
    assert by_experiment == {
        "cgan_recipe": [6, 0],
        "cifar100_recipe": [51, 13],
        "tinyimagenet_recipe": [0, 2],
        "vgg_recipe": [1, 0],
    }


def test_explicit_benchmark_root_is_forwarded(tmp_path):
    benchmark_root = tmp_path / "corpus"
    benchmark_root.mkdir()
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--experiment",
            "cgan_recipe",
            "--benchmark-root",
            str(benchmark_root),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    execution = json.loads(
        (tmp_path / "out" / "execution.json").read_text(encoding="utf-8")
    )
    assert execution["benchmark_root"] == str(benchmark_root.resolve())
    for arm in execution["arms"]:
        index = arm["command"].index("--benchmark-root")
        assert arm["command"][index + 1] == str(benchmark_root.resolve())


def test_pilot_timeout_cap_is_forwarded_and_sealed(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--experiment",
            "vgg_recipe",
            "--timeout-cap",
            "17",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    execution = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
    assert execution["timeout_cap_seconds"] == 17
    for arm in execution["arms"]:
        index = arm["command"].index("--timeout-cap")
        assert arm["command"][index + 1] == "17"


def test_explicit_arm_selection_materializes_only_requested_pair(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--experiment",
            "vgg_recipe",
            "--arm",
            "baseline",
            "--arm",
            "abcrown_treatment",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    execution = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
    assert [(arm["experiment"], arm["arm"]) for arm in execution["arms"]] == [
        ("vgg_recipe", "baseline"),
        ("vgg_recipe", "abcrown_treatment"),
    ]
