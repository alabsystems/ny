# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "plan_compact_tail_envelope.py"
MANIFEST = REPO_ROOT / "benchmarks" / "compact_tail_envelope_v1.json"


def _load_planner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ny_compact_tail_planner_test", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


planner = _load_planner()


def _floats(value: float, count: int) -> str:
    return " ".join(str(value) for _ in range(count))


def _seam_bounds(
    *, stable_selector: int | None = None
) -> tuple[list[float], list[float]]:
    unstable_indices = set(range(41)) | {45, 93, 96}
    remaining = [index for index in range(100) if index not in unstable_indices]
    positive_indices = set(remaining[:37])
    lower = []
    upper = []
    for index in range(100):
        if index in unstable_indices:
            lower.append(-1.0)
            upper.append(1.0)
        elif index in positive_indices:
            lower.append(0.25)
            upper.append(2.0)
        else:
            lower.append(-2.0)
            upper.append(-0.25)
    if stable_selector is not None:
        lower[stable_selector], upper[stable_selector] = 0.25, 2.0
    return lower, upper


def _lpopt_text(*, stable_selector: int | None = None, suffix: str = "") -> str:
    seam_lower, seam_upper = _seam_bounds(stable_selector=stable_selector)
    relu_lower = [max(value, 0.0) for value in seam_lower]
    relu_upper = [max(value, 0.0) for value in seam_upper]
    return "\n".join(
        [
            "# ny lpopt dump v1",
            "INPUT 3072 3 32 32",
            f"L {_floats(-1.0, 3072)}",
            f"U {_floats(1.0, 3072)}",
            "RELUMAP 2",
            "Relu_early Pre_early",
            "Relu_57 Gemm_56",
            "NODE Pre_early 100 1 100",
            f"L {_floats(-1.0, 100)}",
            f"U {_floats(1.0, 100)}",
            "NODE Gemm_56 100 1 100",
            "L " + " ".join(str(value) for value in seam_lower),
            "U " + " ".join(str(value) for value in seam_upper),
            "NODE Relu_57 100 1 100",
            "L " + " ".join(str(value) for value in relu_lower),
            "U " + " ".join(str(value) for value in relu_upper),
            "NODE Gemm_58 100 1 100",
            f"L {_floats(-4.0, 100)}",
            f"U {_floats(4.0, 100)}",
            suffix,
            "",
        ]
    )


def _margins_text() -> str:
    rows = []
    for objective in range(99):
        if objective == 95:
            rows.append("95 -2.5667536 4.67645 0")
        else:
            rows.append(f"{objective} 1 2 0")
    return "\n".join(rows) + "\n"


def _solver_log_text(*, omit_assignment: tuple[str, ...] | None = None) -> str:
    selectors = (45, 1, 93, 96)
    lines = []
    lowers = []
    for ordinal, assignment in enumerate(itertools.product(("A", "I"), repeat=4)):
        if assignment == omit_assignment:
            continue
        lower = -1.0 - ordinal / 100.0
        lowers.append(lower)
        assert len(selectors) == len(assignment)
        premises = ",".join(
            f"Relu_57:{selector}:{state}"
            for selector, state in zip(selectors, assignment)
        )
        lines.append(
            f"[lpopt-split] depth=4 bind_obj=95 bind_lb={lower:.2f} premises={premises}"
        )
    lines.append(f"[frontier] d=4 worst={min(lowers):.2f} domains=16 t=50.1s")
    lines.extend(
        [
            "Graph-MIP leaf: declined (free_binaries=1136 > leaf  budget 96, depth=4)",
            "Graph-MIP leaf: declined (free_binaries=1132 > leaf  budget 96, depth=4)",
        ]
    )
    return "\n".join(lines) + "\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    stable_selector: int | None = None,
    omit_assignment: tuple[str, ...] | None = None,
    lpopt_suffix: str = "",
) -> dict[str, Path]:
    model = tmp_path / "model.onnx"
    property_path = tmp_path / "property.vnnlib"
    lpopt = tmp_path / "lpopt.dump"
    margins = tmp_path / "lpopt.dump.margins"
    solver_log = tmp_path / "solver.log"
    manifest_path = tmp_path / "manifest.json"
    model.write_bytes(b"sealed-model")
    property_path.write_bytes(b"sealed-property")
    lpopt.write_text(
        _lpopt_text(stable_selector=stable_selector, suffix=lpopt_suffix),
        encoding="utf-8",
    )
    margins.write_text(_margins_text(), encoding="utf-8")
    solver_log.write_text(
        _solver_log_text(omit_assignment=omit_assignment), encoding="utf-8"
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["target"]["model_sha256"] = _sha256(model.read_bytes())
    manifest["target"]["property_sha256"] = _sha256(property_path.read_bytes())
    manifest["evidence_sha256"] = {
        "lpopt_dump": _sha256(lpopt.read_bytes()),
        "root_margins": _sha256(margins.read_bytes()),
        "solver_log": _sha256(solver_log.read_bytes()),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "manifest_path": manifest_path,
        "model_path": model,
        "property_path": property_path,
        "lpopt_path": lpopt,
        "margins_path": margins,
        "solver_log_path": solver_log,
    }


def test_sealed_manifest_is_diagnostic_only_and_matches_imb_support_schedule() -> None:
    payload, _ = planner.load_manifest(MANIFEST)
    assert payload["diagnostic_only"] is True
    assert payload["execution_allowed"] is False
    assert payload["authority"] is False
    assert payload["contract"]["variants"] == ["B0", "B1", "K2", "K4", "K8", "K16"]
    assert payload["contract"]["shared_input_support_rows"] == [2, 4, 8, 16]
    assert payload["target"]["fixed_tree_selectors"] == [45, 1, 93, 96]
    assert payload["proof_ladder"][2]["implementation_status"] == (
        "executor_ready_live_prefix_crown_producer_absent"
    )
    assert payload["proof_ladder"][2]["ny_api"] == (
        "certify_linear_lower_bound_at_with_ay_fixed_assignment_tree_compact_progressive_admission"
    )
    assert payload["proof_ladder"][1]["warm_start"] == (
        "ay_default_root_cold_then_gray_order_preceding_basis_warm_leaves"
    )
    assert payload["proof_ladder"][3]["ny_api"] == (
        "certify_linear_lower_bound_at_with_ay_admission"
    )


def test_bounds_block_methods_reject_mismatched_bounds() -> None:
    block = planner.BoundsBlock(
        name="mismatched",
        shape=(2,),
        lower=(-1.0, 0.0),
        upper=(1.0,),
    )

    with pytest.raises(planner.PlanError, match="mismatched bounds"):
        block.phase_census()
    with pytest.raises(planner.PlanError, match="mismatched bounds"):
        block.width_metrics()


def test_plan_is_deterministic_and_projects_the_live_gap(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = planner.build_plan(**paths)
    second = planner.build_plan(**paths)
    assert planner._canonical_json(first) == planner._canonical_json(second)
    assert first["authority"] is False

    assert first["observations"]["full_network_unstable_binaries"] == 144
    assert first["observations"]["compact_tail"]["phase_census"] == {
        "stable_positive": 37,
        "stable_negative": 19,
        "unstable": 44,
        "total": 100,
    }
    frontier = first["observations"]["fixed_tree_frontier"]
    assert frontier["leaves"] == 16
    assert [entry["neuron_index"] for entry in frontier["selectors"]] == [45, 1, 93, 96]
    assert frontier["worst_lower"] == pytest.approx(-1.15)

    variants = {entry["id"]: entry for entry in first["variants"]}
    assert list(variants) == ["B0", "B1", "K2", "K4", "K8", "K16"]
    assert variants["B0"]["projected_metrics"]["binaries"] == 44
    assert variants["B0"]["projected_metrics"]["columns"] == 307
    assert variants["B0"]["projected_metrics"]["rows"] == 233
    assert variants["B0"]["projected_metrics"]["nnz_upper_bound"] == 10508
    assert variants["B1"]["projected_metrics"]["support_bank_bytes_upper_bound"] == 408
    assert variants["K16"]["projected_metrics"] == {
        "columns": 3379,
        "rows": 265,
        "nnz_upper_bound": 112012,
        "binaries": 44,
        "support_bank_bytes_upper_bound": 399872,
        "added_columns": 3072,
        "added_rows": 32,
        "added_nnz_upper_bound": 101504,
    }
    assert (
        first["resource_contract"]["current_imb_cgan_caps"][
            "directly_admits_cifar_root_input"
        ]
        is False
    )
    assert len(first["experiment_matrix"]) == 24
    assert first["acceptance"]["required_certificate_scope"] == (
        "end_to_end_request_bound_prefix_plus_tail"
    )
    assert "soundness_contract" not in first
    assert "future_executor_requirements" in first


def test_incomplete_fixed_tree_frontier_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, omit_assignment=("I", "I", "I", "I"))
    with pytest.raises(planner.PlanError, match="15 matching split leaves"):
        planner.build_plan(**paths)


def test_fixed_tree_selector_must_be_unstable_at_the_seam(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, stable_selector=45)
    with pytest.raises(planner.PlanError, match="selector 45 is not unstable"):
        planner.build_plan(**paths)


def test_unknown_lpopt_record_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, lpopt_suffix="SURPRISE 1 2 3")
    with pytest.raises(planner.PlanError, match="unknown lpopt record"):
        planner.build_plan(**paths)


def test_model_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    manifest["target"]["model_sha256"] = "0" * 64
    paths["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(planner.PlanError, match="model hash mismatch"):
        planner.build_plan(**paths)


@pytest.mark.parametrize(
    ("path_key", "expected_message"),
    [
        ("lpopt_path", "lpopt dump hash mismatch"),
        ("margins_path", "root margins hash mismatch"),
        ("solver_log_path", "solver log hash mismatch"),
    ],
)
def test_mutated_evidence_identity_fails_before_structural_parse(
    tmp_path: Path,
    path_key: str,
    expected_message: str,
) -> None:
    paths = _fixture(tmp_path)
    evidence_path = paths[path_key]
    evidence_path.write_bytes(evidence_path.read_bytes() + b"\n")
    with pytest.raises(planner.PlanError, match=expected_message):
        planner.build_plan(**paths)


def test_structurally_valid_evidence_swap_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    swapped = tmp_path / "other-solver.log"
    swapped.write_text(_solver_log_text() + "\n", encoding="utf-8")
    paths["solver_log_path"] = swapped
    with pytest.raises(planner.PlanError, match="solver log hash mismatch"):
        planner.build_plan(**paths)


def test_manifest_cannot_enable_execution_or_expand_support_schedule(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    base = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))

    enabled = copy.deepcopy(base)
    enabled["execution_allowed"] = True
    paths["manifest_path"].write_text(json.dumps(enabled), encoding="utf-8")
    with pytest.raises(planner.PlanError, match="forbid execution"):
        planner.build_plan(**paths)

    authoritative = copy.deepcopy(base)
    authoritative["authority"] = True
    paths["manifest_path"].write_text(json.dumps(authoritative), encoding="utf-8")
    with pytest.raises(planner.PlanError, match="carry no authority"):
        planner.build_plan(**paths)

    expanded = copy.deepcopy(base)
    expanded["contract"]["shared_input_support_rows"].append(32)
    paths["manifest_path"].write_text(json.dumps(expanded), encoding="utf-8")
    with pytest.raises(planner.PlanError, match="support schedule"):
        planner.build_plan(**paths)


def test_cli_output_is_canonical_and_never_overwrites_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "plan.json"
    argv = [
        "--manifest",
        str(paths["manifest_path"]),
        "--model",
        str(paths["model_path"]),
        "--property",
        str(paths["property_path"]),
        "--lpopt-dump",
        str(paths["lpopt_path"]),
        "--root-margins",
        str(paths["margins_path"]),
        "--solver-log",
        str(paths["solver_log_path"]),
        "--output",
        str(output),
    ]
    assert planner.main(argv) == 0
    first = output.read_bytes()
    assert first == planner._canonical_json(json.loads(first))

    assert planner.main(argv) == 2
    assert output.read_bytes() == first
    assert "File exists" in capsys.readouterr().err
