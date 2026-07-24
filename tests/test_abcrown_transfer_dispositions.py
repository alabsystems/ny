# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "benchmarks" / "abcrown_transfer_dispositions_v1.json"

EXPECTED_IDS = [
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
    "C1",
    "C2",
    "C3",
    "D1",
    "D2",
    "D3",
    "E1",
    "E2",
    "F1",
    "F2",
    "F3",
]

EXPECTED_MILESTONES = {
    "A1": "M1",
    "A2": "M1",
    "A3": "M1",
    "B1": "M1",
    "B2": "M2",
    "B3": "M2",
    "C1": "M3",
    "C2": "M3",
    "C3": "M3",
    "D1": "M4",
    "D2": "M4",
    "D3": "M4",
    "E1": "M5",
    "E2": "M5",
    "F1": "M1",
    "F2": "M6",
    "F3": "M7",
}

EXPECTED_STATUSES = {
    "A1": "retained_default_off",
    "A2": "retained_default_off",
    "A3": "retained_default_off",
    "B1": "retained_default_off",
    "B2": "promoted_implementation",
    "B3": "retained_default_off",
    "C1": "recorded_kill",
    "C2": "retained_default_off",
    "C3": "retained_default_off",
    "D1": "recorded_kill",
    "D2": "retained_default_off",
    "D3": "retained_default_off",
    "E1": "retained_default_off",
    "E2": "recorded_kill",
    "F1": "promoted_implementation",
    "F2": "promoted_implementation",
    "F3": "retained_default_off",
}

ALLOWED_STATUSES = {
    "promoted_implementation",
    "retained_default_off",
    "recorded_kill",
}

REQUIRED_DISPOSITION_FIELDS = {
    "id",
    "status",
    "default_changed",
    "rollback_or_fallback",
    "rationale",
    "evidence_paths",
}


def load_ledger() -> dict:
    with LEDGER_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def assert_nonempty_text(value: object, field: str, record_id: str) -> None:
    assert isinstance(value, str), f"{record_id}.{field} must be a string"
    assert value.strip(), f"{record_id}.{field} must not be empty"


def assert_evidence_paths(paths: object, record_id: str) -> None:
    assert isinstance(paths, list), f"{record_id}.evidence_paths must be a list"
    assert paths, f"{record_id}.evidence_paths must not be empty"

    for value in paths:
        assert_nonempty_text(value, "evidence_paths[]", record_id)
        portable = PurePosixPath(value)
        assert not portable.is_absolute(), (
            f"{record_id} evidence path must be repo-relative: {value}"
        )
        assert ".." not in portable.parts, (
            f"{record_id} evidence path must not traverse upward: {value}"
        )
        assert (REPO_ROOT / value).exists(), (
            f"{record_id} evidence path does not exist: {value}"
        )


def assert_disposition_record(record: object, *, require_milestone: bool) -> None:
    assert isinstance(record, dict), "each disposition must be an object"
    record_id = record.get("id", "<missing>")
    assert REQUIRED_DISPOSITION_FIELDS <= record.keys(), (
        f"{record_id} is missing required disposition fields"
    )
    assert_nonempty_text(record["id"], "id", str(record_id))
    assert record["status"] in ALLOWED_STATUSES, (
        f"{record_id}.status is not an allowed disposition"
    )
    assert record["default_changed"] is False, (
        f"{record_id} must not record a verifier default change"
    )
    assert_nonempty_text(
        record["rollback_or_fallback"], "rollback_or_fallback", record["id"]
    )
    assert_nonempty_text(record["rationale"], "rationale", record["id"])
    assert_evidence_paths(record["evidence_paths"], record["id"])

    if require_milestone:
        assert "milestone" in record, f"{record_id}.milestone is required"
        assert re.fullmatch(r"M[1-7]", record["milestone"]), (
            f"{record_id}.milestone must be M1 through M7"
        )


def test_ledger_schema_and_exact_gap_ids() -> None:
    ledger = load_ledger()
    assert ledger["schema"] == "ny_abcrown_transfer_dispositions_v1"
    assert ledger["allowed_statuses"] == [
        "promoted_implementation",
        "retained_default_off",
        "recorded_kill",
    ]
    assert ledger["source_plan"] == (
        "docs/ALPHA_BETA_CROWN_PERFORMANCE_TRANSFER_PLAN_2026-07-22.md"
    )
    assert (REPO_ROOT / ledger["source_plan"]).exists()
    assert isinstance(ledger["items"], list)
    assert [item["id"] for item in ledger["items"]] == EXPECTED_IDS


def test_top_level_dispositions_are_complete_and_fixed() -> None:
    ledger = load_ledger()
    for item in ledger["items"]:
        assert_disposition_record(item, require_milestone=True)
        assert_nonempty_text(item.get("title"), "title", item["id"])
        assert item["milestone"] == EXPECTED_MILESTONES[item["id"]]
        assert item["status"] == EXPECTED_STATUSES[item["id"]]


def test_subarm_dispositions_are_complete_and_unique() -> None:
    ledger = load_ledger()
    for item in ledger["items"]:
        subarms = item.get("subarms", [])
        assert isinstance(subarms, list), f"{item['id']}.subarms must be a list"
        subarm_ids = [subarm.get("id") for subarm in subarms]
        assert len(subarm_ids) == len(set(subarm_ids)), (
            f"{item['id']} contains duplicate subarm IDs"
        )
        for subarm in subarms:
            assert_disposition_record(subarm, require_milestone=False)


def test_required_composite_gap_subarms() -> None:
    items = {item["id"]: item for item in load_ledger()["items"]}
    actual = {
        gap_id: {subarm["id"]: subarm["status"] for subarm in items[gap_id]["subarms"]}
        for gap_id in ("E2", "F2", "F3")
    }
    assert actual == {
        "E2": {
            "pure_chain": "recorded_kill",
            "conjunctive_multi_objective": "recorded_kill",
            "per_disjunct_alpha": "retained_default_off",
            "active_cuts": "recorded_kill",
        },
        "F2": {
            "fused_wide_beta_capture": "promoted_implementation",
            "analytic_replacement_of_spsa_supplements": "recorded_kill",
            "independent_upper_analytic_chain": "recorded_kill",
            "residual_reshape_broadcast_edges": "recorded_kill",
            "fused_wide_alpha_capture": "retained_default_off",
        },
        "F3": {
            "full_biccos_multi_tree": "recorded_kill",
            "hybrid_input_activation_branching": "recorded_kill",
            "activation_space_bab_attack": "recorded_kill",
            "selective_input_split_alpha_refinement": "retained_default_off",
        },
    }
