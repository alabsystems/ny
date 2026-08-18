# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

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

UNCLAMPED_CHILD_DUMP = """\
# ny lpopt child-bounds dump v1 depth=2 bind_obj=1 bind_lb=-0.25 premises=Relu_B:0:A,Relu_A:1:I
INPUT 2 2
L -1 -0.5
U 1 0.5
NODE Pre_A 2 2
L -1 -2
U 3 1.5
NODE Pre_B 1 1 1
L -3.5
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
        "ny_root_interval": [-4.0, 5.0],
        "ny_child_raw_interval": [0.0, 5.0],
        "abc_root_interval": [-2.0, 3.0],
    }
    assert inactive["abc_layer_index"] == 0
    assert inactive["abc_history_sign"] == -1.0
    assert inactive["abc_bound_clamp"] == "upper=0"


def test_replay_plan_accepts_unclamped_child_with_separate_history(
    tmp_path: Path,
) -> None:
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    child = parity.parse_ny_dump(
        _write(tmp_path / "unclamped.dump", UNCLAMPED_CHILD_DUMP)
    )
    plan = parity.build_replay_plan(root, child, _abc_export())

    active, inactive = plan["mapped_premises"]
    assert plan["ny_child"]["node_bounds_semantics"] == parity.NY_CHILD_BOUND_SEMANTICS
    assert active["ny_child_raw_interval"] == [-3.5, 5.0]
    assert active["abc_bound_clamp"] == "lower=0"
    assert inactive["ny_child_raw_interval"] == [-2.0, 1.5]
    assert inactive["abc_bound_clamp"] == "upper=0"


def test_replay_plan_rejects_mismatched_or_conflicting_child(
    tmp_path: Path,
) -> None:
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))

    mismatched_input = CHILD_DUMP.replace("L -1 -0.5\nU 1 0.5", "L -0.9 -0.5\nU 1 0.5")
    child = parity.parse_ny_dump(_write(tmp_path / "mismatched.dump", mismatched_input))
    with pytest.raises(parity.ParityError, match="root/child input bounds differ"):
        parity.build_replay_plan(root, child, _abc_export())

    conflicting = UNCLAMPED_CHILD_DUMP.replace("L -3.5\nU 5", "L -3.5\nU -0.1")
    child = parity.parse_ny_dump(_write(tmp_path / "conflicting.dump", conflicting))
    with pytest.raises(parity.ParityError, match="conflict with its active"):
        parity.build_replay_plan(root, child, _abc_export())


def test_replay_plan_rejects_stable_or_out_of_range_premises(
    tmp_path: Path,
) -> None:
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", UNCLAMPED_CHILD_DUMP))
    stable_root_dump = ROOT_DUMP.replace("L -4\nU 5", "L 0\nU 5")
    stable_root = parity.parse_ny_dump(
        _write(tmp_path / "stable-root.dump", stable_root_dump)
    )
    with pytest.raises(parity.ParityError, match="not unstable in the NY root"):
        parity.build_replay_plan(stable_root, child, _abc_export())

    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    stable_abc = _abc_export()
    stable_abc["root"]["split_layers"][1]["lower"][0] = 0.0
    with pytest.raises(
        parity.ParityError, match="official root coordinate that is not unstable"
    ):
        parity.build_replay_plan(root, child, stable_abc)

    out_of_range_dump = UNCLAMPED_CHILD_DUMP.replace(
        "premises=Relu_B:0:A,Relu_A:1:I",
        "premises=Relu_B:1:A,Relu_A:1:I",
    )
    out_of_range = parity.parse_ny_dump(
        _write(tmp_path / "out-of-range.dump", out_of_range_dump)
    )
    with pytest.raises(parity.ParityError, match="Relu_B:1 exceeds size 1"):
        parity.build_replay_plan(root, out_of_range, _abc_export())


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


def test_replay_plan_validation_fails_closed(tmp_path: Path) -> None:
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", CHILD_DUMP))
    export = _abc_export()
    plan = parity.build_replay_plan(root, child, export)
    parity.validate_replay_plan(plan, export)

    wrong_pin = copy.deepcopy(plan)
    wrong_pin["pins"]["vnnlib_sha256"] = "a" * 64
    with pytest.raises(parity.ParityError, match="pins do not exactly match"):
        parity.validate_replay_plan(wrong_pin, export)

    duplicate = copy.deepcopy(plan)
    duplicate["mapped_premises"][1].update(
        abc_layer_index=1,
        abc_preactivation="/b",
        neuron=0,
    )
    with pytest.raises(parity.ParityError, match="duplicates an ABC coordinate"):
        parity.validate_replay_plan(duplicate, export)

    inconsistent = copy.deepcopy(plan)
    inconsistent["mapped_premises"][0]["abc_history_sign"] = -1.0
    with pytest.raises(parity.ParityError, match="inconsistent state metadata"):
        parity.validate_replay_plan(inconsistent, export)

    wrong_bounds_semantics = copy.deepcopy(plan)
    wrong_bounds_semantics["ny_child"]["node_bounds_semantics"] = (
        "premise_endpoints_preclamped"
    )
    with pytest.raises(parity.ParityError, match="separate from split history"):
        parity.validate_replay_plan(wrong_bounds_semantics, export)

    stable_coordinate = copy.deepcopy(plan)
    stable_coordinate["mapped_premises"][0]["ny_root_interval"] = [0.0, 5.0]
    with pytest.raises(parity.ParityError, match="not NY-root unstable"):
        parity.validate_replay_plan(stable_coordinate, export)

    conflicting_child = copy.deepcopy(plan)
    conflicting_child["mapped_premises"][0]["ny_child_raw_interval"] = [-3.0, -0.1]
    with pytest.raises(parity.ParityError, match="conflicts with NY child bounds"):
        parity.validate_replay_plan(conflicting_child, export)

    false_lineage = copy.deepcopy(plan)
    false_lineage["ny_child"]["lineage_relationship"] = "parent_is_previous_dump"
    with pytest.raises(parity.ParityError, match="must not claim"):
        parity.validate_replay_plan(false_lineage, export)

    root_disguised_as_child = copy.deepcopy(plan)
    root_disguised_as_child["ny_child"]["depth"] = 0
    root_disguised_as_child["mapped_premises"] = []
    with pytest.raises(parity.ParityError, match="integer >= 1"):
        parity.validate_replay_plan(root_disguised_as_child, export)


def test_root_structure_validation_rejects_malformed_layer(tmp_path: Path) -> None:
    export = _abc_export()
    export["root"]["split_layers"][0]["shape"] = [3]
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", CHILD_DUMP))
    with pytest.raises(parity.ParityError, match="expected 3"):
        parity.build_replay_plan(root, child, export)


def test_history_preserves_premise_depth_not_group_count(tmp_path: Path) -> None:
    same_layer_child = CHILD_DUMP.replace(
        "premises=Relu_B:0:A,Relu_A:1:I",
        "premises=Relu_A:0:A,Relu_A:1:I",
    ).replace("L -1 -2\nU 3 0", "L 0 -2\nU 3 0")
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", same_layer_child))
    plan = parity.build_replay_plan(root, child, _abc_export())
    history = parity.build_abc_history(plan, ["/a", "/b"])

    assert plan["ny_child"]["depth"] == 2
    assert history["/a"] == ([0, 1], [1.0, -1.0], [0.0, 0.0], [], [])
    assert history["/b"] == ([], [], [], [], [])


def test_live_history_requires_unstable_clipper_coordinate(tmp_path: Path) -> None:
    class Scalar:
        def __init__(self, value: bool) -> None:
            self.value = value

        def item(self) -> bool:
            return self.value

    class FlatMask:
        shape = (1, 2)

        def __init__(self, values: list[bool]) -> None:
            self.values = values

        def __getitem__(self, key: tuple[int, int]) -> Scalar:
            return Scalar(self.values[key[1]])

    class Mask:
        shape = (1, 2)

        def __init__(self, values: list[bool]) -> None:
            self.values = values

        def reshape(self, *_shape: int) -> FlatMask:
            return FlatMask(self.values)

    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", CHILD_DUMP))
    plan = parity.build_replay_plan(root, child, _abc_export())
    masks = {"/a": Mask([True, True]), "/b": Mask([True, False])}
    mapping = {"/a": {0: 0, 1: 1}, "/b": {0: 0}}
    parity._validate_live_history_coordinates(plan, masks, mapping)

    masks["/b"] = Mask([False, False])
    with pytest.raises(parity.ParityError, match="not live-root unstable"):
        parity._validate_live_history_coordinates(plan, masks, mapping)

    masks["/b"] = Mask([True, False])
    del mapping["/a"][1]
    with pytest.raises(parity.ParityError, match="absent from"):
        parity._validate_live_history_coordinates(plan, masks, mapping)


def test_replay_history_clamps_are_applied_after_reconstruction(
    tmp_path: Path,
) -> None:
    class Scalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class Tensor:
        dtype = "float32"
        device = "cpu"
        requires_grad = False

        def __init__(self, values: list[float]) -> None:
            self.values = values
            self.shape = (1, len(values))

        def numel(self) -> int:
            return len(self.values)

        def is_floating_point(self) -> bool:
            return True

        def reshape(self, first: int, second: int) -> Tensor:
            assert (first, second) == (1, -1)
            return self

        def __getitem__(self, key: tuple[int, int]) -> Scalar:
            assert key[0] == 0
            return Scalar(self.values[key[1]])

        def __setitem__(self, key: tuple[int, int], value: float) -> None:
            assert key[0] == 0
            self.values[key[1]] = value

    class Torch:
        @staticmethod
        def is_tensor(value: object) -> bool:
            return isinstance(value, Tensor)

    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", UNCLAMPED_CHILD_DUMP))
    plan = parity.build_replay_plan(root, child, _abc_export())
    lower = {"/a": Tensor([-0.5, -1.0]), "/b": Tensor([-2.0])}
    upper = {"/a": Tensor([2.0, 1.0]), "/b": Tensor([3.0])}

    applied = parity._apply_replay_history_clamps(lower, upper, plan, Torch)

    assert applied == [
        {
            "abc_preactivation": "/b",
            "neuron": 0,
            "state": "active",
            "before": [-2.0, 3.0],
            "after": [0.0, 3.0],
        },
        {
            "abc_preactivation": "/a",
            "neuron": 1,
            "state": "inactive",
            "before": [-1.0, 1.0],
            "after": [-1.0, 0.0],
        },
    ]
    parity._require_replay_history_clamps(
        lower, upper, plan, Torch, context="test post-preprocess"
    )

    lower["/b"].values[0] = -2.0
    with pytest.raises(parity.ParityError, match="did not preserve exact active clamp"):
        parity._require_replay_history_clamps(
            lower, upper, plan, Torch, context="test post-preprocess"
        )

    lower["/b"].values[0] = 0.25
    with pytest.raises(parity.ParityError, match="cannot apply active"):
        parity._apply_replay_history_clamps(lower, upper, plan, Torch)


def test_sole_official_survivor_check_is_exact() -> None:
    parity._require_sole_official_survivor([95], 95)
    for survivors in ([], [50], [95, 96]):
        with pytest.raises(parity.ParityError, match="sole official root survivor"):
            parity._require_sole_official_survivor(survivors, 95)


def test_child_artifact_is_explicitly_non_authoritative_and_deterministic(
    tmp_path: Path,
) -> None:
    root = parity.parse_ny_dump(_write(tmp_path / "root.dump", ROOT_DUMP))
    child = parity.parse_ny_dump(_write(tmp_path / "child.dump", CHILD_DUMP))
    plan = parity.build_replay_plan(root, child, _abc_export())
    arm = {
        "beta_enabled": True,
        "clip_enabled": False,
        "beta_crown_iteration": 11,
        "beta_optimizer_step_budget": 10,
        "bab_iteration": 3,
    }
    artifact = parity.build_abc_child_artifact(
        plan, arm, {"final": {"lower_minus_rhs": [-0.2]}}
    )

    assert artifact["kind"] == "abc_child"
    assert artifact["diagnostic_only"] is True
    assert artifact["verifier_authority"] is False
    assert artifact["provenance"]["parent_alpha_beta_warm_start"] == "not_available"
    assert artifact["provenance"]["lineage_relationship"] == "not_encoded"
    assert (
        artifact["provenance"]["ny_child_node_bounds"]
        == parity.NY_CHILD_BOUND_SEMANTICS
    )
    assert artifact["provenance"]["split_history_application"].startswith(
        "explicit_validated_zero_clamp"
    )
    malformed_arm = dict(arm)
    del malformed_arm["bab_iteration"]
    malformed_arm["unrelated"] = 1
    with pytest.raises(parity.ParityError, match="arm metadata is incomplete"):
        parity.build_abc_child_artifact(plan, malformed_arm, {})
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    assert parity.write_export(first, artifact) == parity.write_export(second, artifact)
    assert first.read_bytes() == second.read_bytes()


def test_abc_child_cli_dispatches_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_export(args: object) -> int:
        captured["args"] = args
        return 17

    monkeypatch.setattr(parity, "export_abc_child", fake_export)
    result = parity.main(
        [
            "abc-child",
            "--abc-repo",
            "/abc",
            "--config",
            "/config",
            "--onnx",
            "/model",
            "--vnnlib",
            "/property",
            "--ny-root",
            "/root",
            "--ny-child",
            "/child",
            "--abc-export",
            "/export",
            "--beta",
            "on",
            "--clip",
            "off",
            "--bab-iteration",
            "4",
            "--output",
            "/output",
        ]
    )

    assert result == 17
    assert captured["args"].beta == "on"
    assert captured["args"].clip == "off"
    assert captured["args"].beta_optimizer_updates == 10
    assert captured["args"].bab_iteration == 4


def test_optimizer_step_budget_uses_nonzero_evaluation_count() -> None:
    assert parity.abc_iteration_for_optimizer_updates(0) == 1
    assert parity.abc_iteration_for_optimizer_updates(10) == 11
    with pytest.raises(parity.ParityError, match="integer >= 0"):
        parity.abc_iteration_for_optimizer_updates(-1)
    with pytest.raises(parity.ParityError, match="integer >= 0"):
        parity.abc_iteration_for_optimizer_updates(True)


def test_pinned_checkout_must_be_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(parity.subprocess, "run", fake_run)
    checkout = Path("/pinned")
    parity._require_clean_git_tree(checkout, "test dependency")
    assert calls == [
        (
            (["git", "status", "--porcelain=v1", "--untracked-files=all"],),
            {
                "cwd": checkout,
                "check": True,
                "capture_output": True,
                "text": True,
            },
        )
    ]

    monkeypatch.setattr(
        parity,
        "_git_status_porcelain",
        lambda _path: " M tracked.py\n?? untracked.py\n",
    )
    with pytest.raises(
        parity.ParityError,
        match=r"test dependency checkout must be clean.*M tracked\.py.*untracked\.py",
    ):
        parity._require_clean_git_tree(checkout, "test dependency")


def test_gitlink_and_runtime_module_identity_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "abc"
    package = checkout / "auto_LiRPA" / "auto_LiRPA"
    package.mkdir(parents=True)
    module_file = _write(package / "__init__.py", "")
    parity._require_module_under(
        SimpleNamespace(__file__=str(module_file)), package, "auto_LiRPA"
    )
    with pytest.raises(parity.ParityError, match="outside the pinned checkout"):
        parity._require_module_under(
            SimpleNamespace(__file__=str(tmp_path / "shadow.py")),
            package,
            "auto_LiRPA",
        )

    expected = parity.AUTOLIRPA_SHA
    monkeypatch.setattr(
        parity.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"160000 commit {expected}\tauto_LiRPA\n"
        ),
    )
    assert parity._gitlink_commit(checkout, "auto_LiRPA") == expected
    monkeypatch.setattr(
        parity.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"100644 blob {expected}\tauto_LiRPA\n"
        ),
    )
    with pytest.raises(parity.ParityError, match="not an exact gitlink"):
        parity._gitlink_commit(checkout, "auto_LiRPA")


def test_beta_initialization_fails_closed() -> None:
    parity._require_absent_beta_warm_start(None, "root")
    parity._require_singleton_absent_domain_beta([None])
    history = {
        "layer_a": ([], [], [], [], []),
        "layer_b": ([], [], [], [], []),
    }
    parity._require_empty_root_history(history, ["layer_a", "layer_b"])

    with pytest.raises(parity.ParityError, match="root.*beta warm start"):
        parity._require_absent_beta_warm_start({"layer": [0.0]}, "root")
    with pytest.raises(parity.ParityError, match="singleton domain.*beta warm start"):
        parity._require_singleton_absent_domain_beta([{"layer": [0.0]}])
    with pytest.raises(parity.ParityError, match="singleton domain.*beta warm start"):
        parity._require_singleton_absent_domain_beta([])
    history["layer_a"][0].append(7)
    with pytest.raises(parity.ParityError, match="contains split decisions"):
        parity._require_empty_root_history(history, ["layer_a", "layer_b"])
    with pytest.raises(parity.ParityError, match="does not match its split nodes"):
        parity._require_empty_root_history(
            {"layer_a": ([], [], [], [], [])}, ["layer_a", "layer_b"]
        )


def test_generated_beta_values_must_be_exact_zero() -> None:
    class Scalar:
        def __init__(self, value: bool) -> None:
            self.value = value

        def item(self) -> bool:
            return self.value

    class NonzeroMask:
        def __init__(self, value: bool) -> None:
            self.value = value

        def any(self) -> Scalar:
            return Scalar(self.value)

    class Tensor:
        def __init__(self, values: list[float]) -> None:
            self.values = values

        def numel(self) -> int:
            return len(self.values)

        def detach(self) -> Tensor:
            return self

        def __ne__(self, other: object) -> NonzeroMask:
            assert other == 0
            return NonzeroMask(any(value != 0 for value in self.values))

    class Torch:
        @staticmethod
        def is_tensor(value: object) -> bool:
            return isinstance(value, Tensor)

    zero = SimpleNamespace(
        _data={
            "layer_a": [SimpleNamespace(val=Tensor([0.0, 0.0]))],
            "layer_b": [SimpleNamespace(val=Tensor([]))],
        }
    )
    parity._require_zero_sparse_beta_values(zero, 2, Torch)

    nonzero = SimpleNamespace(
        _data={"layer_a": [SimpleNamespace(val=Tensor([0.0, 0.25]))]}
    )
    with pytest.raises(parity.ParityError, match="layer_a.*nonzero"):
        parity._require_zero_sparse_beta_values(nonzero, 2, Torch)
    with pytest.raises(parity.ParityError, match="expected 3, got 2"):
        parity._require_zero_sparse_beta_values(zero, 3, Torch)
