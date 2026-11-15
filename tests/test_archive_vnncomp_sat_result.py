# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "archive_vnncomp_sat_result.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("archive_vnncomp_sat_result", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


archive = _load_module()
import ny_measurement_provenance as provenance  # noqa: E402

FAST_GRAD_PREDICATE = (
    "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
    "NY_PGD_FAST_GRAD.0=ran"
)
MARGIN_ROW_PREDICATE = (
    "ny-exec-v1.margin-row-frontier-parameter-engaged."
    "NY_MARGIN_ROW_PARALLEL.4=ran"
)
EXACT_VJP_JOINT_PREDICATE = (
    "ny-exec-v2.exact-vjp-parameter-engaged."
    "NY_PGD_VJP_K.52,NY_PGD_VJP_STEPS.250=ran"
)
FIXTURE_SEARCH_SPACE_PAYLOAD = {
    "schema": "ny-levers/search-space/v6",
    "axes": [
        {
            "name": "NY_PGD_FAST_GRAD",
            "domain": "bool",
            "values": ["0", "1"],
            "class": "VerdictAffecting",
            "deliver": "preset:attack.pgd_fast_grad",
            "why": "fixture execution-predicate axis",
            "absence_baseline": {"kind": "declaration_default"},
            "category_prefixes": [],
            "preset_baseline_keys": ["attack.pgd_fast_grad"],
        }
    ],
    "edges": [],
    "unsafe": [],
    "instrument_only": [],
    "test_only": [],
    "declarations": {
        "NY_PGD_FAST_GRAD": {
            "kind": "Bool",
            "default": "1",
            "min": None,
            "max": None,
            "members": None,
        }
    },
}
FIXTURE_JOINT_SEARCH_SPACE_PAYLOAD = {
    "schema": "ny-levers/search-space/v6",
    "axes": [
        {
            "name": name,
            "domain": "bool",
            "values": ["0", "1"],
            "class": "VerdictAffecting",
            "deliver": f"preset:attack.{name.lower()}",
            "why": "fixture same-family joint prerequisite axis",
            "absence_baseline": {"kind": "declaration_default"},
            "category_prefixes": [],
            "preset_baseline_keys": [f"attack.{name.lower()}"],
        }
        for name in ("NY_PGD_EXACT_BATCHED", "NY_PGD_VJP_BATCH")
    ] + [
        {
            "name": name,
            "domain": "grid",
            "values": values,
            "class": "VerdictAffecting",
            "deliver": f"preset:attack.{name.lower()}",
            "why": "fixture same-family joint execution-predicate axis",
            "absence_baseline": {"kind": "contextual"},
            "category_prefixes": [],
            "preset_baseline_keys": [f"attack.{name.lower()}"],
        }
        for name, values in (
            ("NY_PGD_VJP_K", ["8", "52", "128"]),
            ("NY_PGD_VJP_STEPS", ["100", "250", "1000"]),
        )
    ],
    "edges": [
        {
            "child": child,
            "kind": "armed",
            "parent": parent,
            "parents": None,
            "bound": None,
            "token": None,
            "site": "archive same-family joint fixture",
        }
        for child, parent in (
            ("NY_PGD_VJP_K", "NY_PGD_VJP_BATCH"),
            ("NY_PGD_VJP_STEPS", "NY_PGD_VJP_BATCH"),
            ("NY_PGD_VJP_STEPS", "NY_PGD_EXACT_BATCHED"),
        )
    ],
    "unsafe": [],
    "instrument_only": [],
    "test_only": [],
    "declarations": {
        **{
            name: {
                "kind": "Bool",
                "default": "0",
                "min": None,
                "max": None,
                "members": None,
            }
            for name in ("NY_PGD_EXACT_BATCHED", "NY_PGD_VJP_BATCH")
        },
        **{
        name: {
            "kind": "UsizeTrimmed",
            "default": None,
            "min": None,
            "max": None,
            "members": None,
        }
        for name in ("NY_PGD_VJP_K", "NY_PGD_VJP_STEPS")
        },
    },
}


def _fixture_binary_build_id(executable_sha256: str = "2" * 64) -> dict[str, object]:
    fields = {
        "version": "0.3.1",
        "features": "none",
        "backend": "wgpu",
        "markers": "margin-row-gpu-eft-impl-v9",
        "commit": "a" * 40,
        "dirty": "false",
        "revision": "1",
        "source": "clean",
        "profile": "release",
        "target": "fixture-target",
        "rustc": "1.90.0",
        "stamp": "1" * 64,
        "exe": executable_sha256,
    }
    fields["id"] = provenance.hashlib.sha256(
        f"{provenance.BUILD_ID_SCHEMA}\n{fields['stamp']}\n{fields['exe']}\n".encode(
            "ascii"
        )
    ).hexdigest()
    line = " ".join(
        [provenance.BUILD_ID_SCHEMA]
        + [
            f"{field}={fields[field]}"
            for field in provenance.BUILD_ID_CANONICAL_FIELDS
        ]
    )
    return {"schema": provenance.BUILD_ID_SCHEMA, "line": line, "fields": fields}


FIXTURE_BINARY_BUILD_ID = _fixture_binary_build_id()
FIXTURE_BUILD_STAMP = provenance.flight_build_stamp_for_binary_build_id(
    FIXTURE_BINARY_BUILD_ID
)


def _v3_lever_state() -> dict[str, object]:
    levers = [
        {
            "name": "NY_ALPHA_ZERO_YIELD_FRAC",
            "value": 0.25,
            "source": "config",
            "bucket": "debug",
            "moat": "high",
            "provenance": "measured",
        },
        {
            "name": "NY_PHASE_TELEMETRY",
            "value": True,
            "source": "legacy_env",
            "bucket": "debug",
            "moat": "low",
            "provenance": "unmeasured",
            "env_utf8": True,
        },
    ]
    return {
        "status": "resolved",
        "receipt": {
            "schema": "ny-levers/receipt/v3",
            "lever_count": len(levers),
            "env_present": 1,
            "env_accepted": 1,
            "env_rejected": 0,
            "env_ignored": 0,
            "levers": levers,
        },
    }


def test_lever_receipt_v3_ignored_evidence_and_v2_compatibility() -> None:
    ambient = {"NY_PHASE_TELEMETRY": "1"}
    current = _v3_lever_state()
    assert archive._valid_v3_lever_state(current, ambient_env=ambient)

    legacy = json.loads(json.dumps(current))
    legacy["receipt"]["schema"] = "ny-levers/receipt/v2"
    del legacy["receipt"]["env_ignored"]
    assert archive._valid_v3_lever_state(legacy, ambient_env=ambient)

    ignored = json.loads(json.dumps(current))
    entry = ignored["receipt"]["levers"][0]
    entry["name"] = "NY_UPFRONT_ATTACK"
    entry["value"] = False
    entry["ignored_raw"] = "true"
    entry["env_utf8"] = True
    ignored["receipt"]["env_present"] = 2
    ignored["receipt"]["env_ignored"] = 1
    ignored_ambient = {**ambient, "NY_UPFRONT_ATTACK": "true"}
    assert archive._valid_v3_lever_state(ignored, ambient_env=ignored_ambient)

    malformed = []
    bad = json.loads(json.dumps(ignored))
    bad["receipt"]["env_ignored"] = 0
    malformed.append((bad, ignored_ambient))
    malformed.append((ignored, ambient))
    malformed.append((ignored, {**ignored_ambient, "NY_UPFRONT_ATTACK": "false"}))
    bad = json.loads(json.dumps(ignored))
    del bad["receipt"]["levers"][0]["env_utf8"]
    malformed.append((bad, ignored_ambient))
    bad = json.loads(json.dumps(ignored))
    bad["receipt"]["levers"][0]["rejected_raw"] = "true"
    malformed.append((bad, ignored_ambient))
    bad = json.loads(json.dumps(ignored))
    bad["receipt"]["schema"] = "ny-levers/receipt/v2"
    del bad["receipt"]["env_ignored"]
    malformed.append((bad, ignored_ambient))
    for bad_state, bad_ambient in malformed:
        assert not archive._valid_v3_lever_state(bad_state, ambient_env=bad_ambient)


@pytest.mark.parametrize(
    "name", ["NY_MARGIN_ROW_HEAD_GROUPS", "NY_MARGIN_ROW_HEAD_DIRECTIONAL_CUTS"]
)
@pytest.mark.parametrize("enabled", [False, True])
def test_owner_directed_default_provenance_has_exact_scope(
    name: str, enabled: bool
) -> None:
    state = _v3_lever_state()
    entry = state["receipt"]["levers"][0]
    entry.update(
        name=name, value=enabled, bucket="default_on", moat="high",
        provenance="owner_directed",
    )
    ambient = {"NY_PHASE_TELEMETRY": "1"}
    assert archive._valid_v3_lever_state(state, ambient_env=ambient)
    for field, value in [
        ("name", "NY_UNAUTHORIZED_DEFAULT"), ("bucket", "debug"),
        ("moat", "low"), ("value", 1), ("provenance", "unmeasured"),
    ]:
        invalid = json.loads(json.dumps(state))
        invalid["receipt"]["levers"][0][field] = value
        assert not archive._valid_v3_lever_state(invalid, ambient_env=ambient)


def test_historical_v1_receipt_refuses_owner_directed_defaults() -> None:
    import regular_bank_evidence as bank_evidence

    receipt = {
        "schema": "ny-levers/receipt/v1",
        "lever_count": 1,
        "env_overridden": 0,
        "levers": [{
            "name": "NY_MARGIN_ROW_HEAD_GROUPS", "value": True,
            "source": "default", "bucket": "default_on", "moat": "high",
            "provenance": "guard",
        }],
    }
    assert bank_evidence._valid_v2_lever_receipt(receipt, ambient_env={})
    receipt["levers"][0]["provenance"] = "owner_directed"
    assert not bank_evidence._valid_v2_lever_receipt(receipt, ambient_env={})


def _v3_flight_record() -> dict[str, object]:
    return {
        "schema_version": 3,
        "build_id": FIXTURE_BUILD_STAMP,
        "backend_kind": "cpu-only",
        "backend_summary": "fixture",
        "host": {
            "hostname": "fixture-host",
            "cpu_model": "fixture-cpu",
            "logical_cores": 1,
            "ram_bytes": 1_073_741_824,
        },
        "load_avg_at_begin": [0.0, 0.0, 0.0],
        "load_avg_at_end": [0.0, 0.0, 0.0],
        "category": "acasxu_2023",
        "budget_secs": 116,
        "ambient_env_schema": "ny-cli/ambient-env/v3",
        "ambient_env": {
            "NY_PHASE_TELEMETRY": "1",
            "OMP_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "2",
            "RAYON_RS_NUM_CPUS": "3",
        },
        "levers": _v3_lever_state(),
        "events": [
            {
                "method": "run_complete",
                "status": "complete",
                "reason": "unsat",
                "at_secs": 1.25,
            }
        ],
    }


def _v3_partial_flight_record() -> dict[str, object]:
    completed = _v3_flight_record()
    reason = (
        "in-process watchdog hard stop at budget+grace; timeout committed"
    )
    flushed_at = 121.0
    completed["events"] = [
        {
            "method": "internal_timeout_route",
            "status": "ran",
            "at_secs": 0.01,
        },
        {
            "method": "run_incomplete",
            "status": "incomplete",
            "reason": reason,
            "at_secs": flushed_at,
        },
    ]
    # Match FlightRecord's Rust struct order: record_status is serialized
    # immediately after schema_version and before build_id.
    return {
        "schema_version": completed.pop("schema_version"),
        "record_status": {
            "status": "partial_hard_stop",
            "reason": reason,
            "flushed_at_secs": flushed_at,
        },
        **completed,
    }


def test_current_v3_flight_shape_matches_the_rust_completed_record_contract() -> None:
    record = _v3_flight_record()
    expected_keys = {
        "ambient_env",
        "ambient_env_schema",
        "backend_kind",
        "backend_summary",
        "budget_secs",
        "build_id",
        "category",
        "events",
        "host",
        "levers",
        "load_avg_at_begin",
        "load_avg_at_end",
        "schema_version",
    }
    assert set(record) == expected_keys
    assert archive._valid_current_flight_shape(record)

    for missing in ("ambient_env_schema", "build_id"):
        malformed = json.loads(json.dumps(record))
        del malformed[missing]
        assert not archive._valid_current_flight_shape(malformed)

    malformed = json.loads(json.dumps(record))
    malformed["unexpected"] = True
    assert not archive._valid_current_flight_shape(malformed)


def test_current_v3_partial_shape_matches_the_rust_hard_stop_contract() -> None:
    record = _v3_partial_flight_record()
    assert list(record)[:3] == ["schema_version", "record_status", "build_id"]
    assert set(record["record_status"]) == {
        "status",
        "reason",
        "flushed_at_secs",
    }
    assert archive._valid_current_flight_shape(record)
    assert archive._valid_partial_v3_flight_lifecycle(record)


def _add_requested_lever(
    record: dict[str, object],
    *,
    name: str,
    token: str,
    value: object,
    source: str = "legacy_env",
) -> None:
    ambient = record["ambient_env"]
    assert isinstance(ambient, dict)
    if source == "legacy_env":
        ambient[name] = token
    state = record["levers"]
    assert isinstance(state, dict)
    receipt = state["receipt"]
    assert isinstance(receipt, dict)
    levers = receipt["levers"]
    assert isinstance(levers, list)
    entry = {
        "name": name,
        "value": value,
        "source": source,
        "bucket": "debug",
        "moat": "high",
        "provenance": "unmeasured",
    }
    if source == "legacy_env":
        entry["env_utf8"] = True
    levers.append(entry)
    receipt["lever_count"] += 1
    if source == "legacy_env":
        receipt["env_present"] += 1
        receipt["env_accepted"] += 1


def _add_fast_gradient_engagement(
    record: dict[str, object],
    *,
    phase: str = "terminal",
    schema: str = "ny-cli/pgd-fast-gradient-execution/v1",
    decisions: int = 1,
    source: str = "legacy_env",
) -> None:
    _add_requested_lever(
        record,
        name="NY_PGD_FAST_GRAD",
        token="0",
        value=False,
        source=source,
    )
    status = "ran" if decisions > 0 else "skipped"
    reason = " ".join(
        (
            f"schema={schema}",
            f"phase={phase}",
            "route=conjunctive",
            "configured=false",
            f"decisions={decisions}",
            "fast_attempts=0",
            "fast_selected=0",
            f"certified_selected={decisions}",
            "spsa_selected=0",
        )
    )
    events = record["events"]
    assert isinstance(events, list)
    events.insert(
        0,
        {
            "method": "pgd_fast_gradient_execution",
            "status": status,
            "reason": reason,
            "at_secs": 0.01,
        },
    )


def _add_margin_row_engagement(
    record: dict[str, object], *, counterfactual_differences: int
) -> None:
    _add_requested_lever(
        record, name="NY_MARGIN_ROW_PARALLEL", token="4", value=4
    )
    contextual = 2 if counterfactual_differences else 4
    reason = " ".join(
        (
            "schema=ny-cli/margin-row-frontier-execution/v1",
            "phase=terminal",
            "route=joint",
            "outcome=unknown",
            "tree_started=true",
            "configured=4",
            f"contextual={contextual}",
            "steps=1",
            f"counterfactual_differences={counterfactual_differences}",
            "max_open=4",
            "max_batch=4",
        )
    )
    events = record["events"]
    assert isinstance(events, list)
    events.insert(
        0,
        {
            "method": "margin_row_frontier_execution",
            "status": "ran",
            "reason": reason,
            "at_secs": 0.01,
        },
    )


def _add_exact_vjp_joint_delivery(record: dict[str, object]) -> None:
    _add_requested_lever(
        record, name="NY_PGD_EXACT_BATCHED", token="1", value=True
    )
    _add_requested_lever(
        record, name="NY_PGD_VJP_BATCH", token="1", value=True
    )
    _add_requested_lever(
        record, name="NY_PGD_VJP_K", token="52", value=52
    )
    _add_requested_lever(
        record, name="NY_PGD_VJP_STEPS", token="250", value=250
    )


def _current_predicate_start() -> dict[str, object]:
    environment = {
        "values": {
            "NY_MEASURE_REQUIRE_FLIGHT_EVENT": FAST_GRAD_PREDICATE,
            "NY_PGD_FAST_GRAD": "0",
        },
        "typed_values": {
            "NY_MEASURE_REQUIRE_FLIGHT_EVENT": archive._parse_required_flight_event(
                FAST_GRAD_PREDICATE
            ),
            "NY_PGD_FAST_GRAD": {"type": "boolean", "value": False},
        },
    }
    search_space_contract = provenance._search_space_contract_from_payload(
        FIXTURE_SEARCH_SPACE_PAYLOAD, source="archive fixture"
    )
    return {
        "schema": "ny_measurement_start_v1",
        "run_id": "20260718T120000Z-123",
        "measurement": {
            "execution_evidence_profile": archive.EXECUTION_EVIDENCE_PROFILE,
            "search_space_contract": search_space_contract,
            "execution_predicate_assignment": (
                provenance.build_execution_predicate_assignment(
                    environment, search_space_contract
                )
            ),
        },
        "solver_binary": {
            "sha256": "2" * 64,
            "declared_build_features": [],
            "build_id": FIXTURE_BINARY_BUILD_ID,
        },
        "environment": environment,
    }


def _current_joint_predicate_start() -> dict[str, object]:
    parsed = archive._parse_required_flight_event(EXACT_VJP_JOINT_PREDICATE)
    environment = {
        "values": {
            "NY_MEASURE_REQUIRE_FLIGHT_EVENT": EXACT_VJP_JOINT_PREDICATE,
            "NY_PGD_EXACT_BATCHED": "1",
            "NY_PGD_VJP_BATCH": "1",
            "NY_PGD_VJP_K": "52",
            "NY_PGD_VJP_STEPS": "250",
        },
        "typed_values": {
            "NY_MEASURE_REQUIRE_FLIGHT_EVENT": parsed,
            "NY_PGD_EXACT_BATCHED": {"type": "boolean", "value": True},
            "NY_PGD_VJP_BATCH": {"type": "boolean", "value": True},
            "NY_PGD_VJP_K": {"type": "nonnegative_integer", "value": 52},
            "NY_PGD_VJP_STEPS": {
                "type": "nonnegative_integer",
                "value": 250,
            },
        },
    }
    search_space_contract = provenance._search_space_contract_from_payload(
        FIXTURE_JOINT_SEARCH_SPACE_PAYLOAD, source="archive joint fixture"
    )
    return {
        "schema": "ny_measurement_start_v1",
        "run_id": "20260718T120000Z-123",
        "measurement": {
            "execution_evidence_profile": (
                provenance.JOINT_EXECUTION_EVIDENCE_PROFILE
            ),
            "search_space_contract": search_space_contract,
            "execution_predicate_assignment": (
                provenance.build_execution_predicate_assignment(
                    environment, search_space_contract
                )
            ),
        },
        "solver_binary": {
            "sha256": "2" * 64,
            "declared_build_features": [],
            "build_id": FIXTURE_BINARY_BUILD_ID,
        },
        "environment": environment,
    }


def _archive(
    tmp_path: Path,
    result_file: Path,
    *,
    solver_verdict: str = "sat",
    instance_index: int = 7,
    flight_file: Path | None = None,
    required_flight_events: tuple[str, ...] = (),
    manifest_required_flight_event: str | None = None,
) -> Path:
    artifact_root = tmp_path / "artifacts"
    start_manifest = artifact_root / "runs" / "20260718T120000Z-123" / "start.json"
    start_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not start_manifest.exists():
        start = {"schema": "ny_measurement_start_v1", "run_id": "20260718T120000Z-123"}
        if manifest_required_flight_event is not None:
            if manifest_required_flight_event != FAST_GRAD_PREDICATE:
                raise AssertionError("archive fixture only models its current treatment")
            start = _current_predicate_start()
        start_manifest.write_text(json.dumps(start) + "\n", encoding="utf-8")
    onnx_file = tmp_path / "a.onnx"
    vnnlib_file = tmp_path / "p.vnnlib"
    if not onnx_file.exists():
        onnx_file.write_bytes(b"onnx fixture")
    if not vnnlib_file.exists():
        vnnlib_file.write_text("; property fixture\n", encoding="utf-8")
    solver_log_file = tmp_path / "solver.log"
    if not solver_log_file.exists():
        solver_log_file.write_bytes(b"solver stdout\nsolver stderr\n")
    preflight_manifest = archive.seal_inputs(
        artifact_root=artifact_root,
        run_id="20260718T120000Z-123",
        category="acasxu_2023",
        instance_index=instance_index,
        onnx="onnx/a.onnx",
        vnnlib="vnnlib/p.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        start_manifest=start_manifest,
    )
    return archive.archive_result(
        result_file=result_file,
        solver_log_file=solver_log_file,
        artifact_root=artifact_root,
        run_id="20260718T120000Z-123",
        category="acasxu_2023",
        instance_index=instance_index,
        onnx="onnx/a.onnx",
        vnnlib="vnnlib/p.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        solver_verdict=solver_verdict,
        solver_exit_status=0,
        timeout_seconds=116,
        elapsed_seconds=9,
        source_csv="reports/measured/acasxu_2023.csv",
        start_manifest=start_manifest,
        preflight_manifest=preflight_manifest,
        flight_file=flight_file,
        required_flight_events=required_flight_events,
    )


def test_archives_complete_witness_and_not_checked_metadata(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    complete_result = b"sat\n((X_0 1.25)\n (Y_0 -2.0))\n"
    result_file.write_bytes(complete_result)

    archived = _archive(tmp_path, result_file)

    assert archived.read_bytes() == complete_result
    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["witness_present"] is True
    assert metadata["counterexample_validation"]["status"] == "not_checked"
    assert (
        metadata["result_sha256"] == archive.hashlib.sha256(complete_result).hexdigest()
    )
    assert metadata["raw_result_sha256"] == metadata["result_sha256"]
    assert (
        metadata["onnx"]["sha256"]
        == archive.hashlib.sha256(b"onnx fixture").hexdigest()
    )
    assert (
        metadata["vnnlib"]["sha256"]
        == archive.hashlib.sha256(b"; property fixture\n").hexdigest()
    )
    solver_log = archived.with_name(archived.name.replace(".results", ".solver.log"))
    assert solver_log.read_bytes() == b"solver stdout\nsolver stderr\n"
    assert (
        metadata["solver_log"]["sha256"]
        == archive.hashlib.sha256(solver_log.read_bytes()).hexdigest()
    )
    start_manifest = tmp_path / "artifacts/runs/20260718T120000Z-123/start.json"
    assert metadata["start_manifest"] == ("runs/20260718T120000Z-123/start.json")
    assert (
        metadata["start_manifest_sha256"]
        == archive.hashlib.sha256(start_manifest.read_bytes()).hexdigest()
    )


def test_archives_non_sat_result_and_marks_validation_not_applicable(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")

    archived = _archive(tmp_path, result_file, solver_verdict="unsat")

    assert archived.read_bytes() == b"unsat\n"
    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["solver_verdict"] == "unsat"
    assert metadata["witness_present"] is False
    assert metadata["counterexample_validation"]["status"] == "not_applicable"


def test_embeds_a_row_bound_flight_record_in_immutable_metadata(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = {
        "schema_version": 2,
        "backend_kind": "cpu-only",
        "backend_summary": "fixture",
        "host": {},
        "category": "acasxu_2023",
        "budget_secs": 116,
        "ambient_env": {"NY_INVPROP": "1", "OMP_NUM_THREADS": "1"},
        "events": [
            {
                "method": "run_complete",
                "status": "complete",
                "reason": "unsat",
                "at_secs": 1.25,
            }
        ],
    }
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    archived = _archive(
        tmp_path,
        result_file,
        solver_verdict="unsat",
        flight_file=flight_file,
    )

    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    captured = metadata["flight_record"]
    assert captured["status"] == "captured"
    assert captured["record"] == record
    assert captured["size_bytes"] == len(flight_file.read_bytes())
    assert (
        captured["source_sha256"]
        == archive.hashlib.sha256(flight_file.read_bytes()).hexdigest()
    )


def test_embeds_a_valid_v3_flight_record_with_resolved_levers(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    archived = _archive(
        tmp_path,
        result_file,
        solver_verdict="unsat",
        flight_file=flight_file,
    )

    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["flight_record"]["record"] == record


def test_archives_a_real_current_partial_hard_stop_timeout_record(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("timeout\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_partial_flight_record()
    # Rust's watchdog uses serde_json::to_string_pretty without a trailing
    # newline.  Preserve that producer shape in this compatibility test.
    flight_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    archived = _archive(
        tmp_path,
        result_file,
        solver_verdict="timeout",
        flight_file=flight_file,
    )

    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    captured = metadata["flight_record"]
    assert captured["record"] == record
    assert captured["source_sha256"] == archive.hashlib.sha256(
        flight_file.read_bytes()
    ).hexdigest()


def test_partial_prefix_can_prove_a_closed_axis_execution_predicate(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("timeout\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_partial_flight_record()
    _add_fast_gradient_engagement(record)
    flight_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    archived = _archive(
        tmp_path,
        result_file,
        solver_verdict="timeout",
        flight_file=flight_file,
        manifest_required_flight_event=FAST_GRAD_PREDICATE,
    )

    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["flight_record"]["record"]["events"][-1]["status"] == (
        "incomplete"
    )


@pytest.mark.parametrize("solver_verdict", ("sat", "unknown"))
def test_partial_hard_stop_cannot_bind_a_non_timeout_result(
    tmp_path: Path, solver_verdict: str
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_bytes(
        b"sat\n((X_0 0.5))\n" if solver_verdict == "sat" else b"unknown\n"
    )
    flight_file = tmp_path / "result.txt.flight.json"
    flight_file.write_text(
        json.dumps(_v3_partial_flight_record(), indent=2), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="requires the committed timeout verdict"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict=solver_verdict,
            flight_file=flight_file,
        )


@pytest.mark.parametrize(
    "malformation",
    (
        "missing_status_field",
        "extra_status_field",
        "wrong_status",
        "empty_reason",
        "terminal_reason_mismatch",
        "terminal_time_mismatch",
        "terminal_complete",
        "terminal_not_last",
        "duplicate_incomplete",
        "negative_flush_time",
    ),
)
def test_rejects_malformed_current_partial_hard_stop_records(
    tmp_path: Path, malformation: str
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("timeout\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_partial_flight_record()
    status = record["record_status"]
    events = record["events"]
    assert isinstance(status, dict)
    assert isinstance(events, list)
    if malformation == "missing_status_field":
        del status["reason"]
    elif malformation == "extra_status_field":
        status["future"] = True
    elif malformation == "wrong_status":
        status["status"] = "partial"
    elif malformation == "empty_reason":
        status["reason"] = ""
        events[-1]["reason"] = ""
    elif malformation == "terminal_reason_mismatch":
        events[-1]["reason"] = "different hard stop"
    elif malformation == "terminal_time_mismatch":
        events[-1]["at_secs"] = 120.0
    elif malformation == "terminal_complete":
        events[-1] = {
            "method": "run_complete",
            "status": "complete",
            "reason": "timeout",
            "at_secs": 121.0,
        }
    elif malformation == "terminal_not_last":
        events.append(
            {"method": "late_event", "status": "ran", "at_secs": 121.0}
        )
    elif malformation == "duplicate_incomplete":
        events.insert(-1, json.loads(json.dumps(events[-1])))
    elif malformation == "negative_flush_time":
        status["flushed_at_secs"] = -1.0
        events[-1]["at_secs"] = -1.0
    flight_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="flight record identity is invalid"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="timeout",
            flight_file=flight_file,
        )


def test_current_v3_flight_rejects_an_unreviewed_non_ny_ambient_key(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    record["ambient_env"]["RAYON_UNREVIEWED"] = "1"
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="flight record identity is invalid"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
        )


def test_required_flight_event_is_enforced_before_archiving(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    _add_fast_gradient_engagement(record, source="config")
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    archived = _archive(
        tmp_path,
        result_file,
        solver_verdict="unsat",
        flight_file=flight_file,
        required_flight_events=(FAST_GRAD_PREDICATE,),
    )

    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["flight_record"]["record"] == record


def test_required_flight_event_rejects_a_dark_treatment(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    flight_file.write_text(json.dumps(_v3_flight_record()), encoding="utf-8")

    with pytest.raises(ValueError, match="did not engage"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
            required_flight_events=(FAST_GRAD_PREDICATE,),
        )


def test_sealed_start_manifest_enforces_flight_engagement(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    flight_file.write_text(json.dumps(_v3_flight_record()), encoding="utf-8")

    with pytest.raises(ValueError, match="did not engage"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
            manifest_required_flight_event=FAST_GRAD_PREDICATE,
        )


def test_current_execution_profile_binds_the_flight_to_the_sealed_binary(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    _add_fast_gradient_engagement(record)
    record["build_id"] = "NY_BUILD_STAMP_V1 version=other stamp=other"
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="different solver build stamp"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
            manifest_required_flight_event=FAST_GRAD_PREDICATE,
        )


def test_sealed_execution_predicate_rejects_a_stale_typed_declaration() -> None:
    start = {
        "measurement": {
            "execution_evidence_profile": archive.EXECUTION_EVIDENCE_PROFILE
        },
        "environment": {
            "values": {"NY_MEASURE_REQUIRE_FLIGHT_EVENT": FAST_GRAD_PREDICATE},
            "typed_values": {
                "NY_MEASURE_REQUIRE_FLIGHT_EVENT": {
                    "type": "flight_event_requirement",
                    "method": "pgd_fast_gradient_execution",
                    "status": "ran",
                }
            },
        }
    }

    with pytest.raises(ValueError, match="current typed declaration"):
        archive._required_flight_events_from_start(start)


def test_current_execution_profile_rejects_an_omitted_predicate() -> None:
    start = {
        "measurement": {
            "execution_evidence_profile": archive.EXECUTION_EVIDENCE_PROFILE
        },
        "environment": {"values": {}, "typed_values": {}},
    }

    with pytest.raises(ValueError, match="requires a sealed execution predicate"):
        archive._required_flight_events_from_start(start)


def test_current_execution_assignment_and_registry_have_exact_closed_shapes() -> None:
    start = _current_predicate_start()
    measurement = start["measurement"]
    assert isinstance(measurement, dict)
    assert archive._execution_predicate_assignment_from_start(start)[
        "requested_assignment"
    ] == [["NY_PGD_FAST_GRAD", "0"]]

    malformed = []
    missing_assignment_field = json.loads(json.dumps(start))
    del missing_assignment_field["measurement"]["execution_predicate_assignment"][
        "contrast_id"
    ]
    malformed.append(missing_assignment_field)
    extra_assignment_field = json.loads(json.dumps(start))
    extra_assignment_field["measurement"]["execution_predicate_assignment"][
        "future"
    ] = True
    malformed.append(extra_assignment_field)
    invented_contrast = json.loads(json.dumps(start))
    invented_contrast["measurement"]["execution_predicate_assignment"][
        "contrast_id"
    ] = "0" * 16
    malformed.append(invented_contrast)
    missing_contract_field = json.loads(json.dumps(start))
    del missing_contract_field["measurement"]["search_space_contract"]["identity"]
    malformed.append(missing_contract_field)
    extra_contract_field = json.loads(json.dumps(start))
    extra_contract_field["measurement"]["search_space_contract"]["future"] = True
    malformed.append(extra_contract_field)

    for candidate in malformed:
        with pytest.raises(ValueError, match="assignment is invalid"):
            archive._execution_predicate_assignment_from_start(candidate)

    downgraded = json.loads(json.dumps(start))
    downgraded_assignment = downgraded["measurement"][
        "execution_predicate_assignment"
    ]
    downgraded_assignment["schema"] = (
        provenance.LEGACY_EXECUTION_PREDICATE_ASSIGNMENT_SCHEMA
    )
    del downgraded_assignment["runtime_context_assignment"]
    del downgraded_assignment["runtime_context_source_assignment"]
    with pytest.raises(ValueError, match="does not match its evidence profile"):
        archive._execution_predicate_assignment_from_start(downgraded)

    historical = json.loads(json.dumps(downgraded))
    historical["measurement"]["execution_evidence_profile"] = (
        provenance.LEGACY_ASSIGNMENT_EXECUTION_EVIDENCE_PROFILE
    )
    assert archive._execution_predicate_assignment_from_start(historical)[
        "requested_assignment"
    ] == [["NY_PGD_FAST_GRAD", "0"]]

    relabelled = json.loads(json.dumps(start))
    relabelled["measurement"]["execution_evidence_profile"] = (
        provenance.LEGACY_ASSIGNMENT_EXECUTION_EVIDENCE_PROFILE
    )
    with pytest.raises(ValueError, match="does not match its evidence profile"):
        archive._execution_predicate_assignment_from_start(relabelled)


def test_same_family_joint_assignment_and_profile_are_closed() -> None:
    start = _current_joint_predicate_start()

    assert archive._required_flight_events_from_start(start) == (
        EXACT_VJP_JOINT_PREDICATE,
    )
    assignment = archive._execution_predicate_assignment_from_start(start)
    assert assignment["requested_assignment"] == [
        ["NY_PGD_VJP_K", "52"],
        ["NY_PGD_VJP_STEPS", "250"],
    ]
    assert assignment["comparison_assignment"] == [
        ["NY_PGD_EXACT_BATCHED", "1"],
        ["NY_PGD_VJP_BATCH", "1"],
    ]
    assert assignment["full_assignment"] == [
        ["NY_PGD_EXACT_BATCHED", "1"],
        ["NY_PGD_VJP_BATCH", "1"],
        ["NY_PGD_VJP_K", "52"],
        ["NY_PGD_VJP_STEPS", "250"],
    ]

    mismatched = json.loads(json.dumps(start))
    mismatched["measurement"]["execution_evidence_profile"] = (
        provenance.EXECUTION_EVIDENCE_PROFILE
    )
    with pytest.raises(ValueError, match="does not match its evidence profile"):
        archive._required_flight_events_from_start(mismatched)


def test_same_family_joint_event_requires_every_axis_engaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ny_search  # noqa: PLC0415

    start = _current_joint_predicate_start()
    requirement = archive._required_flight_events_from_start(start)[0]
    assignment = archive._execution_predicate_assignment_from_start(start)
    record = _v3_flight_record()
    _add_exact_vjp_joint_delivery(record)
    flight_record = {"status": "captured", "record": record}
    requested_names = {
        "NY_PGD_VJP_K",
        "NY_PGD_VJP_STEPS",
    }
    monkeypatch.setattr(
        ny_search,
        "_exact_vjp_parameter_engagement",
        lambda _wanted, _rows: requested_names,
    )

    archive._enforce_required_flight_events(
        flight_record,
        (requirement,),
        sealed_requirement=requirement,
        execution_assignment=assignment,
    )

    monkeypatch.setattr(
        ny_search,
        "_exact_vjp_parameter_engagement",
        lambda _wanted, _rows: {"NY_PGD_VJP_K"},
    )
    with pytest.raises(
        ValueError,
        match="NY_PGD_VJP_STEPS=250",
    ):
        archive._enforce_required_flight_events(
            flight_record,
            (requirement,),
            sealed_requirement=requirement,
            execution_assignment=assignment,
        )


def test_same_family_joint_event_requires_every_repaired_prerequisite_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ny_search  # noqa: PLC0415

    start = _current_joint_predicate_start()
    requirement = archive._required_flight_events_from_start(start)[0]
    assignment = archive._execution_predicate_assignment_from_start(start)
    record = _v3_flight_record()
    _add_exact_vjp_joint_delivery(record)
    receipt = record["levers"]["receipt"]
    receipt["levers"] = [
        entry
        for entry in receipt["levers"]
        if entry.get("name") != "NY_PGD_VJP_BATCH"
    ]
    receipt["lever_count"] -= 1
    receipt["env_present"] -= 1
    receipt["env_accepted"] -= 1
    ambient = record["ambient_env"]
    del ambient["NY_PGD_VJP_BATCH"]
    monkeypatch.setattr(
        ny_search,
        "_exact_vjp_parameter_engagement",
        lambda _wanted, _rows: {"NY_PGD_VJP_K", "NY_PGD_VJP_STEPS"},
    )

    with pytest.raises(
        ValueError,
        match="NY_PGD_VJP_BATCH=1 must appear exactly once",
    ):
        archive._enforce_required_flight_events(
            {"status": "captured", "record": record},
            (requirement,),
            sealed_requirement=requirement,
            execution_assignment=assignment,
        )


def test_scalar_sealed_direct_delivery_rejects_a_config_sourced_receipt() -> None:
    start = _current_predicate_start()
    requirement = archive._required_flight_events_from_start(start)[0]
    assignment = archive._execution_predicate_assignment_from_start(start)
    record = _v3_flight_record()
    _add_fast_gradient_engagement(record, source="config")

    with pytest.raises(ValueError, match="sealed config or child-environment"):
        archive._enforce_required_flight_events(
            {"status": "captured", "record": record},
            (requirement,),
            sealed_requirement=requirement,
            execution_assignment=assignment,
        )


def test_joint_sealed_direct_delivery_rejects_one_config_sourced_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ny_search  # noqa: PLC0415

    start = _current_joint_predicate_start()
    requirement = archive._required_flight_events_from_start(start)[0]
    assignment = archive._execution_predicate_assignment_from_start(start)
    record = _v3_flight_record()
    _add_exact_vjp_joint_delivery(record)
    ambient = record["ambient_env"]
    del ambient["NY_PGD_VJP_STEPS"]
    receipt = record["levers"]["receipt"]
    entry = next(
        item
        for item in receipt["levers"]
        if item["name"] == "NY_PGD_VJP_STEPS"
    )
    entry["source"] = "config"
    del entry["env_utf8"]
    receipt["env_present"] -= 1
    receipt["env_accepted"] -= 1
    monkeypatch.setattr(
        ny_search,
        "_exact_vjp_parameter_engagement",
        lambda _wanted, _rows: {"NY_PGD_VJP_K", "NY_PGD_VJP_STEPS"},
    )

    with pytest.raises(ValueError, match="sealed config or child-environment"):
        archive._enforce_required_flight_events(
            {"status": "captured", "record": record},
            (requirement,),
            sealed_requirement=requirement,
            execution_assignment=assignment,
        )


def test_required_flight_event_rejects_a_missing_record(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("timeout\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be verified"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="timeout",
            flight_file=tmp_path / "result.txt.flight.json",
            required_flight_events=(FAST_GRAD_PREDICATE,),
        )


@pytest.mark.parametrize(
    "requirement",
    (
        "",
        "missing-separator",
        "cgan_input_leaf_summary=ran",
        "ny-exec-v1.unknown.NY_PGD_FAST_GRAD.1=ran",
        "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
        "NY_MARGIN_ROW_PARALLEL.1=ran",
        "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
        "NY_PGD_FAST_GRAD.1=skipped",
        "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
        "NY_PGD_FAST_GRAD.1=ran=ran",
    ),
)
def test_required_flight_event_rejects_malformed_contracts(
    tmp_path: Path, requirement: str
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    flight_file.write_text(json.dumps(_v3_flight_record()), encoding="utf-8")

    with pytest.raises(ValueError, match="required flight event must be"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
            required_flight_events=(requirement,),
        )


@pytest.mark.parametrize(
    ("phase", "schema", "decisions"),
    (
        ("progress", "ny-cli/pgd-fast-gradient-execution/v1", 1),
        ("terminal", "ny-cli/pgd-fast-gradient-execution/v0", 1),
        ("terminal", "ny-cli/pgd-fast-gradient-execution/v1", 0),
    ),
)
def test_required_execution_predicate_rejects_nonterminal_malformed_or_no_work(
    tmp_path: Path, phase: str, schema: str, decisions: int
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    _add_fast_gradient_engagement(
        record, phase=phase, schema=schema, decisions=decisions
    )
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="did not prove live engagement"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
            required_flight_events=(FAST_GRAD_PREDICATE,),
        )


def test_required_execution_predicate_rejects_zero_counterfactual(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    _add_margin_row_engagement(record, counterfactual_differences=0)
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="did not prove live engagement"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
            required_flight_events=(MARGIN_ROW_PREDICATE,),
        )


def test_required_execution_predicate_rejects_mismatched_requested_value(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    _add_fast_gradient_engagement(record)
    flight_file.write_text(json.dumps(record), encoding="utf-8")
    mismatched = (
        "ny-exec-v1.pgd-fast-gradient-parameter-engaged."
        "NY_PGD_FAST_GRAD.1=ran"
    )

    with pytest.raises(ValueError, match="not accepted exactly"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
            required_flight_events=(mismatched,),
        )


@pytest.mark.parametrize("drift_location", ("record", "predicate_event"))
def test_current_v3_flight_schema_rejects_unknown_fields_before_credit(
    tmp_path: Path, drift_location: str
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    _add_fast_gradient_engagement(record)
    if drift_location == "record":
        record["future_field"] = "unversioned"
    else:
        events = record["events"]
        assert isinstance(events, list)
        events[0]["future_field"] = "unversioned"
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="flight record identity is invalid"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
            required_flight_events=(FAST_GRAD_PREDICATE,),
        )


def test_legacy_flight_event_remains_readable_but_cannot_earn_current_credit(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    events = record["events"]
    assert isinstance(events, list)
    events.insert(
        0,
        {
            "method": "cgan_input_leaf_summary",
            "status": "ran",
            "reason": "effective=true",
            "at_secs": 0.01,
        },
    )
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    archived = _archive(
        tmp_path,
        result_file,
        solver_verdict="unsat",
        flight_file=flight_file,
    )
    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["flight_record"]["record"] == record

    with pytest.raises(ValueError, match="unknown or legacy"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
            required_flight_events=("cgan_input_leaf_summary=ran",),
        )


def test_embeds_an_early_v3_flight_record_before_levers_materialize(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    record["levers"] = {"status": "not_materialized"}
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    archived = _archive(
        tmp_path,
        result_file,
        solver_verdict="unsat",
        flight_file=flight_file,
    )

    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["flight_record"]["record"] == record


def test_embeds_a_v3_flight_record_with_explicit_invalid_config(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unknown\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    record["levers"] = {
        "status": "invalid_config",
        "reason": "alpha_zero_yield_frac must be finite and in range",
    }
    record["events"][-1]["reason"] = "unknown"
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    archived = _archive(
        tmp_path,
        result_file,
        solver_verdict="unknown",
        flight_file=flight_file,
    )

    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["flight_record"]["record"] == record


@pytest.mark.parametrize(
    "malformation",
    (
        "missing_state",
        "unknown_status",
        "count_type",
        "duplicate_name",
        "invalid_source",
        "invalid_source_type",
        "inconsistent_env_counts",
        "invalid_env_utf8",
        "empty_registry",
        "extra_entry_field",
        "env_source_missing_ambient",
        "config_with_env_evidence",
        "invalid_name",
        "unmeasured_default_on",
        "guard_auto",
        "empty_invalid_reason",
    ),
)
def test_rejects_malformed_v3_lever_state(
    tmp_path: Path, malformation: str
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    record = _v3_flight_record()
    receipt = record["levers"]["receipt"]
    if malformation == "missing_state":
        del record["levers"]
    elif malformation == "unknown_status":
        record["levers"] = {"status": "pending"}
    elif malformation == "count_type":
        receipt["lever_count"] = True
    elif malformation == "duplicate_name":
        receipt["levers"][1]["name"] = receipt["levers"][0]["name"]
    elif malformation == "invalid_source":
        receipt["levers"][1]["source"] = "env"
    elif malformation == "invalid_source_type":
        receipt["levers"][1]["source"] = []
    elif malformation == "inconsistent_env_counts":
        receipt["env_present"] = 0
    elif malformation == "invalid_env_utf8":
        receipt["levers"][1]["env_utf8"] = "yes"
    elif malformation == "empty_registry":
        receipt["levers"] = []
        receipt["lever_count"] = 0
        receipt["env_present"] = 0
        receipt["env_accepted"] = 0
    elif malformation == "extra_entry_field":
        receipt["levers"][0]["unexpected"] = "not in receipt/v3"
    elif malformation == "env_source_missing_ambient":
        del record["ambient_env"]["NY_PHASE_TELEMETRY"]
    elif malformation == "config_with_env_evidence":
        receipt["levers"][0]["env_utf8"] = True
    elif malformation == "invalid_name":
        receipt["levers"][0]["name"] = "NY_lowercase"
    elif malformation == "unmeasured_default_on":
        receipt["levers"][1]["bucket"] = "default_on"
    elif malformation == "guard_auto":
        receipt["levers"][0]["provenance"] = "guard"
        receipt["levers"][0]["bucket"] = "auto"
    elif malformation == "empty_invalid_reason":
        record["levers"] = {"status": "invalid_config", "reason": ""}
    flight_file.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="flight record identity is invalid"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
        )


def test_rejects_a_flight_record_for_a_different_terminal_verdict(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("unsat\n", encoding="utf-8")
    flight_file = tmp_path / "result.txt.flight.json"
    flight_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend_kind": "cpu-only",
                "backend_summary": "fixture",
                "host": {},
                "category": "acasxu_2023",
                "budget_secs": 116,
                "ambient_env": {},
                "events": [
                    {
                        "method": "run_complete",
                        "status": "complete",
                        "reason": "sat",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="terminal verdict"):
        _archive(
            tmp_path,
            result_file,
            solver_verdict="unsat",
            flight_file=flight_file,
        )


def test_archives_empty_timeout_result_bytes(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_bytes(b"")
    flight_file = tmp_path / "result.txt.flight.json"

    archived = _archive(
        tmp_path,
        result_file,
        solver_verdict="timeout",
        flight_file=flight_file,
    )

    assert archived.read_bytes() == b""
    metadata = json.loads(archived.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["raw_result_sha256"] == archive.hashlib.sha256(b"").hexdigest()
    assert metadata["flight_record"] == {"status": "missing"}


def test_preflight_rejects_transient_input_swap_and_restore(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    run_id = "20260718T120000Z-swap"
    start_manifest = artifact_root / "runs" / run_id / "start.json"
    start_manifest.parent.mkdir(parents=True)
    start_manifest.write_text(
        json.dumps({"schema": "ny_measurement_start_v1", "run_id": run_id}) + "\n",
        encoding="utf-8",
    )
    onnx_file = tmp_path / "a.onnx"
    vnnlib_file = tmp_path / "p.vnnlib"
    original = b"onnx fixture"
    onnx_file.write_bytes(original)
    vnnlib_file.write_text("; property fixture\n", encoding="utf-8")
    preflight = archive.seal_inputs(
        artifact_root=artifact_root,
        run_id=run_id,
        category="acasxu_2023",
        instance_index=1,
        onnx="a.onnx",
        vnnlib="p.vnnlib",
        onnx_file=onnx_file,
        vnnlib_file=vnnlib_file,
        start_manifest=start_manifest,
    )

    # Restore the exact bytes and size. The pre-run stat identity must still
    # expose the transient replacement instead of accepting end-state equality.
    onnx_file.write_bytes(b"ONNX fixture")
    onnx_file.write_bytes(original)

    with pytest.raises(ValueError, match="preflight-bound input drifted"):
        archive.validate_input_preflight(
            preflight_manifest=preflight,
            artifact_root=artifact_root,
            run_id=run_id,
            category="acasxu_2023",
            instance_index=1,
            onnx="a.onnx",
            vnnlib="p.vnnlib",
            onnx_file=onnx_file,
            vnnlib_file=vnnlib_file,
            start_manifest=start_manifest,
        )


def test_preflight_rejects_manifest_symlink(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_bytes(b"unsat\n")
    archived = _archive(tmp_path, result_file, solver_verdict="unsat")
    preflight = archived.with_suffix(".preflight.json")
    backing = preflight.with_name("preflight-backing.json")
    preflight.rename(backing)
    preflight.symlink_to(backing)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _archive(tmp_path, result_file, solver_verdict="unsat")


def test_rejects_sat_without_assignment(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="structured counterexample"):
        _archive(tmp_path, result_file)


def test_rejects_sat_with_arbitrary_nonempty_payload(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\nthis-is-not-an-assignment\n", encoding="utf-8")

    with pytest.raises(ValueError, match="structured counterexample"):
        _archive(tmp_path, result_file)


def test_archives_structurally_complete_vnnlib2_assignment(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    complete_result = "sat\nX float32 [1, 2]\n0.25\n-0.5\nY float32 [1, 1]\n1.75\n"
    result_file.write_text(complete_result, encoding="utf-8")

    archived = _archive(tmp_path, result_file)
    assert archived.read_text(encoding="utf-8") == complete_result


def test_rejects_truncated_vnnlib2_assignment(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\nX float32 [1, 2]\n0.25\n", encoding="utf-8")

    with pytest.raises(ValueError, match="structured counterexample"):
        _archive(tmp_path, result_file)


def test_never_replaces_different_evidence_for_same_run(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n((X_0 1.0))\n", encoding="utf-8")
    archived = _archive(tmp_path, result_file)
    result_file.write_text("sat\n((X_0 2.0))\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _archive(tmp_path, result_file)

    assert "1.0" in archived.read_text(encoding="utf-8")


def test_identical_retry_is_idempotent(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n((X_0 1.0))\n", encoding="utf-8")

    first = _archive(tmp_path, result_file)
    original_metadata = first.with_suffix(".json").read_bytes()
    second = _archive(tmp_path, result_file)

    assert second == first
    assert first.with_suffix(".json").read_bytes() == original_metadata


def test_never_replaces_different_solver_log_for_same_run(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n((X_0 1.0))\n", encoding="utf-8")
    archived = _archive(tmp_path, result_file)
    (tmp_path / "solver.log").write_text("different log\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _archive(tmp_path, result_file)

    archived_log = archived.with_name(archived.name.replace(".results", ".solver.log"))
    assert archived_log.read_bytes() == b"solver stdout\nsolver stderr\n"


def test_run_cache_hashes_repeated_inputs_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n((X_0 1.0))\n", encoding="utf-8")
    original = archive._sha256_file
    hashed_paths: list[Path] = []

    def counting_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return original(path)

    monkeypatch.setattr(archive, "_sha256_file", counting_sha256)
    _archive(tmp_path, result_file, instance_index=7)
    _archive(tmp_path, result_file, instance_index=8)

    assert [path.name for path in hashed_paths].count("a.onnx") >= 2
    assert [path.name for path in hashed_paths].count("p.vnnlib") >= 2
    second_metadata = next(
        (tmp_path / "artifacts/acasxu_2023").glob("00008-*/20260718T120000Z-123.json")
    )
    metadata = json.loads(second_metadata.read_text(encoding="utf-8"))
    assert metadata["onnx"]["hash_cache_hit"] is True
    assert metadata["vnnlib"]["hash_cache_hit"] is True


def test_run_cache_rehashes_same_size_changed_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_file = tmp_path / "result.txt"
    result_file.write_text("sat\n((X_0 1.0))\n", encoding="utf-8")
    original = archive._sha256_file
    hashed_paths: list[Path] = []

    def counting_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return original(path)

    monkeypatch.setattr(archive, "_sha256_file", counting_sha256)
    _archive(tmp_path, result_file, instance_index=7)
    model = tmp_path / "a.onnx"
    assert len(b"onnx fixture") == len(b"ONNX fixture")
    model.write_bytes(b"ONNX fixture")
    _archive(tmp_path, result_file, instance_index=8)

    assert [path.name for path in hashed_paths].count("a.onnx") >= 2
    cache = json.loads(
        (
            tmp_path / "artifacts/runs/20260718T120000Z-123/input_hash_cache.json"
        ).read_text(encoding="utf-8")
    )
    assert len(cache["entries"]) == 3
