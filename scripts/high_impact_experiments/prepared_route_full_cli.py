#!/usr/bin/env python3
"""Frozen automatic prepared-route arrival and original SafeNLP preservation.

The 26 original occurrences run reference/default/reference, all at the same
official whole-process20s cap and pinned OFF hull profile. These MatMul/Add
originals can honestly exercise prepared refusal and old fallback. They cannot
stand in for prepared proof arrival. Separate, explicitly labeled serialized
integration controls must reach the actual default entry and final private
receipt source. No parser observation creates proof authority or independent
arithmetic replay, performance credit, RSS qualification or competition score.

Run only after the parent has built both clean frozen sources. This driver
does not compile, change old fixtures/presets, skip native inference, disable
watchdogs, or add NY_* controls. A supplied arrival manifest can replace the
new integration controls with independently pinned original model/property
files. Every process, including identity probes, has a cap no greater than20s.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from high_impact_experiments.evidence import (
    ExperimentError, immutable, immutable_json, pin, seal_copy, strict_json, unchanged,
)
from high_impact_experiments.original_hull_full_cli import (
    clean_environment, command_for, finite, inspect_bundle, inspect_flight,
    prepare, sha, source_commit,
)
from high_impact_experiments.process import run_command

ORDER = ("reference_before", "candidate_default", "reference_after")
PREPARED_SOURCE = "original_prepared_root"
PROOF_ARITHMETIC = "outward-per-column-four-plane"
STATUSES = {"Declined", "Refused", "Open", "RootProven", "TreeProven"}


def complete_class_ids(value: object) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(not isinstance(c, bool) and isinstance(c, int) and c >= 0 for c in value)
            and len(set(value)) == len(value))


def prepared_trace(path: Path, candidate: bool, verdict: str) -> dict:
    events = strict_json(path)["events"]
    trace = [e for e in events if e.get("method") == "prepared_root"]
    if not candidate:
        if trace:
            raise ExperimentError("reference already contains prepared automatic route")
        return {"attribution": "reference", "events": [], "proof_arrived": False}
    if not trace:
        if verdict != "sat":
            raise ExperimentError("candidate missed automatic prepared entry without upfront SAT")
        return {"attribution": "preempted_by_prior_sat", "events": [], "proof_arrived": False}
    if len(trace) != 1:
        raise ExperimentError("automatic prepared entry must be recorded exactly once")
    event = trace[0]
    reason = event.get("reason")
    if event.get("status") not in {"ran", "skipped"} or not isinstance(reason, str):
        raise ExperimentError("invalid prepared entry flight status/reason")
    if reason == "armed original classwise policy":
        return {"attribution": "policy_declined", "events": trace, "proof_arrived": False}
    match = re.search(r"\bstatus: (\w+)\b", reason)
    if not reason.startswith("PreparedRouteReceipt {") or not match or match[1] not in STATUSES:
        raise ExperimentError("unrecognized pinned prepared route receipt")
    status = match[1]
    if 'requested_error_lane: "off"' not in reason or f'proof_arithmetic: "{PROOF_ARITHMETIC}"' not in reason:
        raise ExperimentError("default requested/error arithmetic policy did not arrive")
    proof = status in {"RootProven", "TreeProven"}
    if event["status"] != ("ran" if status in {"Open", "RootProven", "TreeProven"} else "skipped"):
        raise ExperimentError("prepared typed status and actual flight status disagree")
    count = re.search(r"\bclass_count: (\d+)\b", reason)
    if proof and (verdict != "unsat" or not count or int(count[1]) == 0):
        raise ExperimentError("prepared positive entry did not publish complete UNSAT")
    attribution = {"Declined": "policy_or_resource_declined", "Refused": "source_or_root_refused",
                   "Open": "actual_open_fallback", "RootProven": "prepared_root_proof",
                   "TreeProven": "prepared_tree_proof"}[status]
    return {"attribution": attribution, "events": trace, "proof_arrived": proof,
            "status": status, "class_count": int(count[1]) if count else 0,
            "counters_are_not_measurements": True}


def raw_result_observation(directory: Path) -> dict:
    """Keep the bounded actual token even if identity/clock checks later fail.

    These bytes are diagnostic only. They cannot qualify a solver arm, witness
    or proof, and malformed bodies remain failures under the strict inspector.
    """
    path = directory / "solver-result"
    limit = 1 << 20

    def fingerprint(info: os.stat_result) -> tuple[int, ...]:
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ExperimentError("missing/unbounded raw result diagnostic")
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    try:
        original = fingerprint(path.lstat())
        # NOFOLLOW rejects a leaf symlink swapped in after lstat; NONBLOCK
        # prevents a replaced FIFO from blocking before the regular-fd check.
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = fingerprint(os.fstat(stream.fileno()))
            if before != original:
                raise ExperimentError("raw result replaced before bounded read")
            raw = stream.read(limit + 1)
            after = fingerprint(os.fstat(stream.fileno()))
        if len(raw) > limit:
            raise ExperimentError("missing/unbounded raw result diagnostic")
        if before != after or after != fingerprint(path.lstat()) or len(raw) != before[2]:
            raise ExperimentError("raw result changed during bounded read")
    except OSError as error:
        raise ExperimentError("missing/unbounded raw result diagnostic") from error
    identity = {"path": str(path.absolute()), "declared_path": str(path.absolute()),
                "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
                "fingerprint": list(before)}
    lines = raw.decode("utf-8").splitlines()
    token = lines[0] if lines else ""
    if token not in {"sat", "unsat", "unknown", "timeout"}:
        raise ExperimentError("invalid raw result diagnostic")
    format_error = None
    if token != "sat" and len(lines) != 1:
        format_error = "unexpected trailing bytes after original decided result"
    if any(line.strip() in {"sat", "unsat", "unknown", "timeout", "error"} for line in lines[1:]):
        format_error = "duplicate bare result token in original result"
    return {"verdict": token, "identity": identity, "format_error": format_error,
            "authority": "diagnostic_only"}


def require_result_format(record: dict) -> None:
    raw = record.get("raw_result")
    if not isinstance(raw, dict):
        raise ExperimentError("actual result body could not be qualified: "
                              + record.get("raw_result_error", "missing"))
    if raw.get("format_error") is not None:
        raise ExperimentError(raw["format_error"])


def inspect_process(directory: Path, row: dict, candidate: bool, capsule: dict,
                    process: dict, stamp: str, expected: dict | None = None) -> dict:
    if (process.get("returncode") != 0 or process.get("timed_out") or process.get("interrupted")
            or not finite(process.get("wall_seconds")) or not 0 <= process["wall_seconds"] <= 20
            or process.get("timeout_seconds") != 20):
        raise ExperimentError("late/failed full CLI process cannot qualify")
    raw = raw_result_observation(directory)
    require_result_format({"raw_result": raw})
    verdict = raw["verdict"]
    profile = Path(capsule["profiles"]["off"]["file"]["path"])
    # Reuse the unchanged strict original identity/clock/native/preset checks.
    # OFF hull is the original reference profile, not a prepared disable flag.
    flight = inspect_flight(directory / "solver-result.flight.json", "off_before", profile, stamp, verdict)
    prepared = prepared_trace(directory / "solver-result.flight.json", candidate, verdict)
    bundle = inspect_bundle(directory / "solver-result.proof.jsonl", row, stamp) if verdict == "unsat" else None
    source = bundle["source"] if bundle else None
    if prepared["proof_arrived"] != (source == PREPARED_SOURCE):
        raise ExperimentError("automatic prepared trace and actual final receipt source disagree")
    if source == PREPARED_SOURCE:
        required = bundle["required"]
        if (not complete_class_ids(required)
                or len(required) != prepared["class_count"]
                or not finite(bundle["min_margin"]) or bundle["min_margin"] <= 0):
            raise ExperimentError("prepared final receipt lacks finite positive complete original classes")
    if expected is not None:
        if not complete_class_ids(expected.get("classes")):
            raise ExperimentError("expected ordered objective contains invalid original class IDs")
        if not prepared["proof_arrived"] or prepared.get("status") != expected["status"]:
            raise ExperimentError("required default prepared root/tree positive arrival is absent")
        if bundle["required"] != expected["classes"]:
            raise ExperimentError("actual prepared receipt changed original ordered objective")
    return {"verdict": verdict, "prepared": prepared, "flight": flight, "bundle": bundle,
            "final_source": source, "legacy_fallback_attribution": None if prepared["proof_arrived"] else source,
            "official_score_credit": False, "independent_arithmetic_replay": False,
            "performance_qualification": False}


def preservation(records: list[dict]) -> dict:
    # Keep a contrary decided token visible even if that arm's later strict
    # identity/clock/arrival checks failed, or a third arm has not run yet.
    decided = set()
    for record in records:
        for key in ("observation", "raw_result"):
            observed = record.get(key)
            if isinstance(observed, dict) and observed.get("verdict") in {"sat", "unsat"}:
                decided.add(observed["verdict"])
    contrary = len(decided) > 1
    errors = ["contrary decided verdict: stop for source/witness/proof audit"] if contrary else []
    arms = {record["arm"]: record for record in records}
    if len(records) != 3 or set(arms) != set(ORDER) or not all(r.get("passed") for r in records):
        return {"passed": False, "contrary_decided_verdict": contrary,
                "errors": errors + ["missing or failed full CLI preservation arm"]}
    verdicts = {arm: arms[arm]["observation"]["verdict"] for arm in ORDER}
    for arm in ("reference_before", "reference_after"):
        if verdicts[arm] in {"sat", "unsat"} and verdicts["candidate_default"] != verdicts[arm]:
            errors.append("candidate failed to preserve settled " + arm)
    return {"passed": not errors, "contrary_decided_verdict": contrary, "errors": errors,
            "verdicts": verdicts, "reference_repeat_stability_diagnostic":
            verdicts["reference_before"] == verdicts["reference_after"]}


def append_preservation_arm(records: list[dict], record: dict) -> dict:
    """Record a completed actual arm and stop before launching the next one."""
    records.append(record)
    result = preservation(records)
    if result["contrary_decided_verdict"]:
        raise ExperimentError("contrary original decided verdict; stop for soundness audit")
    return result


# Minimal protobuf serialization of the SAME mathematical integration controls
# used by the genuine Rust original-file producer tests. No ONNX rewriting or
# native loading bypass occurs: actual vnncomp reads these immutable new files.
def varint(value: int) -> bytes:
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def integer(field: int, value: int) -> bytes:
    return varint(field << 3) + varint(value)


def blob(field: int, value: bytes | str) -> bytes:
    value = value.encode() if isinstance(value, str) else value
    return varint((field << 3) | 2) + varint(len(value)) + value


def tensor(name: str, dims: list[int], values: list[float]) -> bytes:
    return (b"".join(integer(1, d) for d in dims) + integer(2, 1)
            + blob(4, struct.pack("<" + "f" * len(values), *values)) + blob(8, name))


def info(name: str, dims: list[int]) -> bytes:
    shape = b"".join(blob(1, integer(1, d)) for d in dims)
    return blob(1, name) + blob(2, blob(1, integer(1, 1) + blob(2, shape)))


def node(kind: str, inputs: list[str], output: str, transpose: bool = False) -> bytes:
    value = b"".join(blob(1, name) for name in inputs) + blob(2, output) + blob(4, kind)
    if transpose:
        value += blob(5, blob(1, "transB") + integer(3, 1) + integer(20, 2))
    return value


def integration_model(tree: bool) -> bytes:
    nodes = [node("Conv", ["input", "wc", "bc"], "conv"), node("Relu", ["conv"], "trunk"),
             node("Flatten", ["trunk"], "flat"), node("Gemm", ["flat", "w1", "b1"], "pre", True),
             node("Relu", ["pre"], "head"), node("Gemm", ["head", "w2", "b2"], "output", True)]
    weights = [0.0, 0.0, 0.0, 0.0, 1.0, -1.0] if tree else [0.0] * 6
    tensors = [tensor("wc", [1, 1, 1, 1], [1.0]), tensor("bc", [1], [0.0]),
               tensor("w1", [2, 1], [1.0, 1.0]), tensor("b1", [2], [-1.0, -1.0]),
               tensor("w2", [3, 2], weights), tensor("b2", [3], [0.0, 0.0, 0.75])]
    graph = (b"".join(blob(1, n) for n in nodes) + blob(2, "prepared-integration-control")
             + b"".join(blob(5, t) for t in tensors) + blob(11, info("input", [1, 1, 1, 1]))
             + blob(12, info("output", [1, 3])))
    return integer(1, 8) + blob(7, graph) + blob(8, integer(2, 13))


def arrival_controls(directory: Path) -> list[dict]:
    directory.mkdir()
    property_bytes = ("(declare-const X_0 Real)\n(declare-const Y_0 Real)\n"
                      "(declare-const Y_1 Real)\n(declare-const Y_2 Real)\n"
                      "(assert (>= X_0 0))\n(assert (<= X_0 3))\n"
                      "(assert (or (>= Y_1 Y_2) (< Y_2 Y_0)))\n").encode()
    rows = []
    for name, tree in (("constant-root", False), ("duplicate-head-tree", True)):
        model_path, property_path = directory / (name + ".onnx"), directory / (name + ".vnnlib")
        model_pin = immutable(model_path, integration_model(tree))
        property_pin = immutable(property_path, property_bytes)
        rows.append({"name": name, "model": model_pin, "property": property_pin,
                     "origin": "new_serialized_integration_control_not_competition_instance",
                     "timeout_seconds": 20, "expected": {"status": "TreeProven" if tree else "RootProven",
                                                            "classes": [1, 0]}})
    immutable_json(directory / "manifest.json", {"schema": "ny.prepared-cli-arrivals.v1", "rows": rows,
                    "independent_arithmetic_replay": False, "official_score_credit": False})
    return rows


def load_arrivals(path: Path, expected_sha: str) -> list[dict]:
    if sha(path) != expected_sha:
        raise ExperimentError("supplied arrival manifest changed")
    manifest = strict_json(path)
    rows = manifest.get("rows")
    if manifest.get("schema") != "ny.prepared-cli-arrivals.v1" or not isinstance(rows, list) or not rows:
        raise ExperimentError("missing required positive arrival manifest")
    for row in rows:
        if row.get("timeout_seconds") != 20 or row.get("origin") not in {
                "original_competition_instance", "new_serialized_integration_control_not_competition_instance"}:
            raise ExperimentError("arrival control must retain explicit origin and20s cap")
        expected = row.get("expected", {})
        if not isinstance(expected, dict):
            raise ExperimentError("positive arrival observation must be an object")
        classes = expected.get("classes")
        if expected.get("status") not in {"RootProven", "TreeProven"} or not complete_class_ids(classes):
            raise ExperimentError("positive arrival must require actual root/tree proof and classes")
        for key in ("model", "property"):
            actual = pin(Path(row[key]["path"]))
            if actual["sha256"] != row[key]["sha256"] or actual["size_bytes"] != row[key]["size_bytes"]:
                raise ExperimentError("arrival original asset pin changed")
        if row["origin"] == "original_competition_instance":
            validate_original_arrival(row)
    return rows


def validate_original_arrival(row: dict) -> list[dict]:
    """Check exact CSV occurrence/bytes/cap; not a numerical certification."""
    declared = row.get("source_csv")
    index = row.get("source_index")
    if (not isinstance(declared, dict) or isinstance(index, bool) or not isinstance(index, int) or index < 1):
        raise ExperimentError("original arrival requires pinned official CSV occurrence")
    path = Path(declared["path"])
    actual = pin(path)
    if actual["sha256"] != declared["sha256"] or actual["size_bytes"] != declared["size_bytes"]:
        raise ExperimentError("original arrival CSV changed")
    with path.open() as stream:
        records = list(csv.reader(stream))
    if index > len(records) or len(records[index - 1]) != 3:
        raise ExperimentError("original arrival CSV occurrence is absent")
    model, property_file, timeout = records[index - 1]
    if float(timeout) != 20:
        raise ExperimentError("original arrival may not replace its official cap with20s")
    pins = [actual]
    for key, relative in (("model", model), ("property", property_file)):
        source_pin = pin(path.parent / relative)
        if source_pin["sha256"] != row[key]["sha256"] or source_pin["size_bytes"] != row[key]["size_bytes"]:
            raise ExperimentError("arrival changed original CSV model/property bytes")
        pins.append(source_pin)
    return pins


def frozen_solver(source: Path, original_path: Path, directory: Path, env: dict) -> dict:
    commit = source_commit(source)
    directory.mkdir()
    binary = directory / "frozen-solver"
    original = pin(original_path.resolve(strict=True))
    if not os.access(original["path"], os.X_OK):
        raise ExperimentError("frozen source executable is not runnable")
    _, _ = seal_copy(Path(original["path"]), binary)
    binary.chmod(0o555)
    sealed = pin(binary)
    process = run_command([str(binary), "--build-id", "--json"], cwd=source, env=env,
                          stdout=directory / "build-id.json", stderr=directory / "build-id.stderr", timeout_seconds=20)
    identity = strict_json(directory / "build-id.json")
    if (process["returncode"] != 0 or process["timed_out"] or process["interrupted"]
            or process["wall_seconds"] > 20 or identity.get("schema") != "NY_BUILD_ID_V1"
            or identity.get("commit") != commit or identity.get("dirty") != "false"
            or identity.get("source_digest") != "clean" or identity.get("exe_sha256") != original["sha256"]
            or "mip" not in identity.get("features", "").split(",")):
        raise ExperimentError("frozen source/binary/MIP build identity mismatch")
    process = run_command([str(binary), "--build-stamp"], cwd=source, env=env,
                          stdout=directory / "build-stamp.txt", stderr=directory / "build-stamp.stderr", timeout_seconds=20)
    stamp = (directory / "build-stamp.txt").read_text().strip()
    if (process["returncode"] != 0 or process["timed_out"] or process["interrupted"] or process["wall_seconds"] > 20
            or not stamp.startswith("NY_BUILD_STAMP_V1 ") or "\n" in stamp
            or "stamp=" + identity["stamp_sha256"] not in stamp.split()):
        raise ExperimentError("bound free build stamp mismatch")
    unchanged(original)
    return {"source": str(source), "commit": commit, "binary": str(binary),
            "identity": identity, "stamp": stamp, "binary_pin": sealed}


def paired_build_contract(candidate: dict, reference: dict) -> None:
    if candidate["commit"] == reference["commit"] or candidate["binary_pin"]["sha256"] == reference["binary_pin"]["sha256"]:
        raise ExperimentError("prepared default must compare distinct frozen candidate/reference")
    for key in ("features", "features_all", "profile", "target", "backend"):
        if candidate["identity"].get(key) != reference["identity"].get(key):
            raise ExperimentError("paired frozen build treatment differs: " + key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "binary", "reference-source", "reference-binary", "capsule", "rational-controls", "archive"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arrival-manifest", type=Path)
    parser.add_argument("--arrival-sha256")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if bool(args.arrival_manifest) != bool(args.arrival_sha256):
        raise ExperimentError("arrival manifest and independent expected SHA must be supplied together")
    capsule_path, rational_path = args.capsule.resolve(strict=True), args.rational_controls.resolve(strict=True)
    capsule, rows = prepare(capsule_path, rational_path)
    supplied = load_arrivals(args.arrival_manifest.resolve(strict=True), args.arrival_sha256) if args.arrival_manifest else None
    if args.prepare_only:
        print("PREPARED_ROUTE_FULL_CLI_PREPARATION_VALID cases=26 arms=3 arrival_controls="
              + str(len(supplied) if supplied else 2) + " execution=false", flush=True)
        return 0
    source, reference = args.source.resolve(strict=True), args.reference_source.resolve(strict=True)
    archive = args.archive.resolve()
    if archive in {source, reference} or source in archive.parents or reference in archive.parents:
        raise ExperimentError("exclusive archive must be outside both frozen sources")
    archive.mkdir(parents=True, exist_ok=False)
    env = clean_environment()
    candidate = frozen_solver(source, args.binary, archive / "candidate", env)
    control = frozen_solver(reference, args.reference_binary, archive / "reference", env)
    paired_build_contract(candidate, control)
    arrivals = supplied if supplied is not None else arrival_controls(archive / "arrival-assets")
    here = Path(__file__).resolve()
    pins = [pin(capsule_path), pin(rational_path), pin(here), candidate["binary_pin"], control["binary_pin"]]
    for name in ("cohort.py", "evidence.py", "process.py", "original_hull_full_cli.py"):
        pins.append(pin(here.parent / name))
    pins.append(pin(Path(capsule["profiles"]["off"]["file"]["path"])))
    for cohort in capsule["cohorts"]:
        pins.append(pin(Path(cohort["instances_csv_pin"]["path"])))
    if args.arrival_manifest:
        pins.append(pin(args.arrival_manifest.resolve(strict=True)))
    else:
        pins.append(pin(archive / "arrival-assets" / "manifest.json"))
    for row in rows + arrivals:
        pins.extend(pin(Path(row[key]["path"])) for key in ("model", "property"))
        if row.get("origin") == "original_competition_instance":
            # Byte-identical copied assets may be supplied. Keep both their
            # actual CLI bytes and the declared original CSV paths unchanged.
            pins.extend(validate_original_arrival(row))
    immutable_json(archive / "launch.json", {"schema": "ny.prepared-route-full-cli.launch.v1",
        "candidate": candidate, "reference": control, "pins": pins, "order": ORDER, "environment": env,
        "official_timeout_seconds": 20, "original_occurrences": 26, "arrival_controls": arrivals,
        "native_inference_override": None, "independent_arithmetic_replay": False,
        "performance_qualification": False, "legacy_phases_joined_to_prepared_ledger": False})

    def unchanged_inputs() -> None:
        for solver in (candidate, control):
            if source_commit(Path(solver["source"])) != solver["commit"]:
                raise ExperimentError("frozen source changed during full CLI gate")
        for record in pins:
            unchanged(record)

    def execute(row: dict, arm: str, directory: Path, expected: dict | None = None) -> dict:
        unchanged_inputs()
        solver = candidate if arm in {"candidate_default", "arrival"} else control
        directory.mkdir()
        if expected is None:
            command = command_for(row, "off_before", Path(solver["binary"]), directory / "solver-result", capsule)
        else:
            command = [solver["binary"], "vnncomp", "v1", "safenlp_2024", row["model"]["path"],
                       row["property"]["path"], str(directory / "solver-result"), "20", "--configs-dir",
                       capsule["profiles"]["off"]["configs_root"], "--proof-profile", "proof"]
        immutable_json(directory / "request.json", {"argv": command, "row": row, "environment": env,
                       "timeout_seconds": 20, "actual_full_cli": True, "expected_observation": expected})
        print(f"PREPARED_ROUTE_FULL_CLI_START arm={arm} directory={directory.name}", flush=True)
        process = run_command(command, cwd=Path(solver["source"]), env=env,
                              stdout=directory / "stdout.log", stderr=directory / "stderr.log", timeout_seconds=20)
        record = {"arm": arm, "process": process, "passed": False}
        try:
            record["raw_result"] = raw_result_observation(directory)
        except (ExperimentError, OSError, ValueError) as error:
            record["raw_result_error"] = str(error)
        try:
            unchanged_inputs()
            record["observation"] = inspect_process(directory, row, solver is candidate, capsule,
                                                   process, solver["stamp"], expected)
            record["passed"] = True
        except (ExperimentError, OSError, ValueError) as error:
            record["error"] = str(error)
        record["artifacts"] = [pin(path) for path in sorted(directory.iterdir()) if path.is_file()]
        immutable_json(directory / "receipt.json", record)
        print(f"PREPARED_ROUTE_FULL_CLI_DONE arm={arm} directory={directory.name} passed={record['passed']}", flush=True)
        if process["interrupted"]:
            raise KeyboardInterrupt
        return record

    cases, positive = [], []
    try:
        for row in rows:
            case = {"year": row["year"], "source_index": row["source_index"], "occurrence": row["occurrence"], "arms": []}
            for arm in ORDER:
                record = execute(row, arm, archive / f"{row['year']}-{row['source_index']}-{arm}")
                try:
                    append_preservation_arm(case["arms"], record)
                except ExperimentError:
                    case["preservation"] = preservation(case["arms"])
                    cases.append(case)
                    raise
            case["preservation"] = preservation(case["arms"])
            cases.append(case)
            if case["preservation"]["contrary_decided_verdict"]:
                raise ExperimentError("contrary original decided verdict; stop for soundness audit")
        for i, row in enumerate(arrivals):
            positive.append(execute(row, "arrival", archive / f"arrival-{i}", row["expected"]))
        unchanged_inputs()
        for record in [r for c in cases for r in c["arms"]] + positive:
            for artifact in record["artifacts"]:
                unchanged(artifact)
        preserved = len(cases) == 26 and all(c["preservation"]["passed"] for c in cases)
        arrived = len(positive) == len(arrivals) and bool(positive) and all(r["passed"] for r in positive)
        passed = preserved and arrived
        immutable_json(archive / "report.json", {"schema": "ny.prepared-route-full-cli.report.v1",
            "status": "PASS" if passed else "FAIL", "candidate": candidate, "reference": control,
            "cases": cases, "arrival_results": positive, "complete_occurrence_coverage": len(cases) == 26,
            "preservation_passed": preserved, "required_actual_prepared_arrival_passed": arrived,
            "original_models_may_decline_prepared_source": True, "actual_full_cli": True,
            "official_score_credit": False, "independent_arithmetic_replay": False,
            "performance_qualification": False, "loader_and_legacy_memory_qualified": False})
        print(f"PREPARED_ROUTE_FULL_CLI_COHORT_DONE rc={0 if passed else 1} cases={len(cases)} "
              f"processes={sum(len(c['arms']) for c in cases) + len(positive)}", flush=True)
        return 0 if passed else 1
    except BaseException as error:
        immutable_json(archive / "abort.json", {"status": "FAIL", "error": type(error).__name__ + ": " + str(error),
                       "completed_cases": cases, "arrival_results": positive})
        raise


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    try:
        raise SystemExit(main())
    except (ExperimentError, OSError, ValueError) as error:
        print("PREPARED_ROUTE_FULL_CLI_ABORT " + str(error), flush=True)
        raise SystemExit(1)
