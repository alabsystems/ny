# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import copy
from pathlib import Path

import pytest

import scripts.cifar100_bound_parity as parity

ROOT_DUMP = """\
# ny lpopt dump v1
INPUT 2 2
L -1 -0.5
U 1 0.5
RELUMAP 2
Relu_A Pre_A
Relu_B Pre_B
NODE Pre_A 2 2
L -1 -2
U 3 2
NODE Pre_B 1 1 1
L -4
U 5
"""

CHILD_DUMP = """\
# ny lpopt child-bounds dump v1 depth=2 bind_obj=1 bind_lb=-0.25 premises=Relu_B:0:A,Relu_A:1:I
INPUT 2 2
L -1 -0.5
U 1 0.5
NODE Pre_A 2 2
L -1 -2
U 3 0
NODE Pre_B 1 1 1
L 0
U 5
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _abc_export() -> dict:
    return {
        "schema": parity.SCHEMA,
        "kind": "abc_root",
        "pins": {
            "alpha_beta_crown_git": parity.ABC_SHA,
            "auto_lirpa_git": parity.AUTOLIRPA_SHA,
            "config_sha256": parity.CIFAR100_CONFIG_SHA256,
            "onnx_sha256": parity.CIFAR100_MEDIUM_ONNX_SHA256,
            "vnnlib_sha256": parity.PROP1761_VNNLIB_SHA256,
        },
        "recipe": {
            "cuts_enabled": False,
            "biccos_enabled": False,
            "clip_interm_domain": {"enabled": True},
        },
        "input": {
            "shape": [2],
            "lower": [-1.0, -0.5],
            "upper": [1.0, 0.5],
        },
        "specification": {
            "rows": [[[0, 1.0], [1, -1.0]], [[0, 1.0], [2, -1.0]]],
            "rhs": [0.0, 0.0],
        },
        "root": {
            "initial_crown_margins": [0.2, -0.1],
            "effective_margins": [0.2, 0.05],
            "optimized_margin_finite_rows": 1,
            "split_layers": [
                {
                    "ordinal": 0,
                    "preactivation": "/a",
                    "shape": [2],
                    "lower": [-0.5, -1.0],
                    "upper": [2.0, 1.0],
                },
                {
                    "ordinal": 1,
                    "preactivation": "/b",
                    "shape": [1],
                    "lower": [-2.0],
                    "upper": [3.0],
                },
            ],
        },
    }


def test_parse_root_and_child_dumps(tmp_path: Path) -> None:
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", CHILD_DUMP))

    assert root.input_bounds.shape == (2,)
    assert root.relu_map == (("Relu_A", "Pre_A"), ("Relu_B", "Pre_B"))
    assert root.nodes["Pre_B"].shape == (1, 1)
    assert child.child is not None
    assert child.child.depth == 2
    assert child.child.binding_objective == 1
    assert child.child.premises == (
        parity.Premise("Relu_B", 0, True),
        parity.Premise("Relu_A", 1, False),
    )


def test_parser_rejects_unsound_or_ambiguous_records(tmp_path: Path) -> None:
    reversed_bounds = ROOT_DUMP.replace("L -1 -2\nU 3 2", "L 4 -2\nU 3 2")
    with pytest.raises(parity.ParityError, match="lower 4.0 exceeds upper 3.0"):
        parity.parse_ny_dump(_write(tmp_path / "reversed.dump", reversed_bounds))

    duplicate = CHILD_DUMP.replace("Relu_B:0:A,Relu_A:1:I", "Relu_B:0:A,Relu_B:0:I")
    with pytest.raises(parity.ParityError, match="duplicate split premise"):
        parity.parse_ny_dump(_write(tmp_path / "duplicate.dump", duplicate))


def test_compare_separates_initial_crown_from_alpha(tmp_path: Path) -> None:
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    margins_path = _write(tmp_path / "root.dump.margins", "0 0.1 1 0\n1 -0.3 1 0\n")
    result = parity.compare_artifacts(
        root,
        parity.parse_ny_margins(margins_path),
        _abc_export(),
        tolerance=1e-7,
    )

    assert result["input_max_abs_difference"] == 0.0
    assert result["pins"]["vnnlib_sha256"] == parity.PROP1761_VNNLIB_SHA256
    assert result["margins"]["ny_verified_count"] == 1
    assert result["margins"]["abc_initial_crown_verified_count"] == 1
    assert result["margins"]["abc_verified_count"] == 2
    row = result["margins"]["per_objective"][1]
    assert row["abc_initial_crown_minus_ny"] == pytest.approx(0.2)
    assert row["abc_minus_ny"] == pytest.approx(0.35)
    assert result["layers"][0]["abc_nested_fraction"] == 1.0
    assert result["layers"][0]["abc_over_ny_width"] == pytest.approx(0.5625)


def test_replay_plan_uses_official_history_sign_convention(tmp_path: Path) -> None:
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", CHILD_DUMP))
    plan = parity.build_replay_plan(root, child, _abc_export())

    active, inactive = plan["mapped_premises"]
    assert active == {
        "ny_relu": "Relu_B",
        "ny_preactivation": "Pre_B",
        "abc_layer_index": 1,
        "abc_preactivation": "/b",
        "neuron": 0,
        "state": "active",
        "abc_history_sign": 1.0,
        "abc_bound_clamp": "lower=0",
    }
    assert inactive["abc_layer_index"] == 0
    assert inactive["abc_history_sign"] == -1.0
    assert inactive["abc_bound_clamp"] == "upper=0"


def test_replay_plan_rejects_mismatched_or_unclamped_child(tmp_path: Path) -> None:
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))

    mismatched_input = CHILD_DUMP.replace("L -1 -0.5\nU 1 0.5", "L -0.9 -0.5\nU 1 0.5")
    child = parity.parse_ny_dump(_write(tmp_path / "mismatched.dump", mismatched_input))
    with pytest.raises(parity.ParityError, match="root/child input bounds differ"):
        parity.build_replay_plan(root, child, _abc_export())

    unclamped = CHILD_DUMP.replace("L 0\nU 5", "L -0.1\nU 5")
    child = parity.parse_ny_dump(_write(tmp_path / "unclamped.dump", unclamped))
    with pytest.raises(parity.ParityError, match="active lower endpoint"):
        parity.build_replay_plan(root, child, _abc_export())


def test_export_encoding_is_deterministic_and_pin_checked(tmp_path: Path) -> None:
    export = _abc_export()
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    assert parity.write_export(first, export) == parity.write_export(second, export)
    assert first.read_bytes() == second.read_bytes()
    assert parity.read_export(first) == export

    wrong_pin = copy.deepcopy(export)
    wrong_pin["pins"]["alpha_beta_crown_git"] = "0" * 40
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    margins = parity.parse_ny_margins(
        _write(tmp_path / "root.dump.margins", "0 0.1 1 0\n1 -0.3 1 0\n")
    )
    with pytest.raises(parity.ParityError, match="alpha_beta_crown_git"):
        parity.compare_artifacts(root, margins, wrong_pin, tolerance=1e-7)
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", CHILD_DUMP))
    with pytest.raises(parity.ParityError, match="alpha_beta_crown_git"):
        parity.build_replay_plan(root, child, wrong_pin)


def test_noncanonical_property_requires_exact_expected_hash(tmp_path: Path) -> None:
    export = _abc_export()
    alternate_hash = "a" * 64
    export["pins"]["vnnlib_sha256"] = alternate_hash
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    margins = parity.parse_ny_margins(
        _write(tmp_path / "root.dump.margins", "0 0.1 1 0\n1 -0.3 1 0\n")
    )
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", CHILD_DUMP))

    with pytest.raises(parity.ParityError, match="does not match the expected"):
        parity.compare_artifacts(root, margins, export, tolerance=1e-7)
    with pytest.raises(parity.ParityError, match="does not match the expected"):
        parity.build_replay_plan(root, child, export)

    parity.compare_artifacts(
        root,
        margins,
        export,
        tolerance=1e-7,
        expected_vnnlib_sha256=alternate_hash,
    )
    replay = parity.build_replay_plan(
        root,
        child,
        export,
        expected_vnnlib_sha256=alternate_hash,
    )
    assert replay["pins"]["vnnlib_sha256"] == alternate_hash


@pytest.mark.parametrize("bad_hash", ["", "A" * 64, "g" * 64, "0" * 63])
def test_property_hash_must_be_canonical_sha256(tmp_path: Path, bad_hash: str) -> None:
    export = _abc_export()
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    margins = parity.parse_ny_margins(
        _write(tmp_path / "root.dump.margins", "0 0.1 1 0\n1 -0.3 1 0\n")
    )
    with pytest.raises(parity.ParityError, match="lowercase SHA-256"):
        parity.compare_artifacts(
            root,
            margins,
            export,
            tolerance=1e-7,
            expected_vnnlib_sha256=bad_hash,
        )
