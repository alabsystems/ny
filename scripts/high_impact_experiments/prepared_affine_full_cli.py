#!/usr/bin/env python3
"""Require native prepared arrival on all26 original SafeNLP occurrences.

The exact original assets run reference/default/reference (78 solver
processes), using the unchanged official20s OFF hull profile. All four
independently rational-positive originals must publish a genuine prepared
Root or Tree receipt. Every other candidate must reach an owned native Open
or Proven outcome, or be preempted by an actual earlier SAT. Source/resource
refusal is recorded as a failed arrival, never counted as native engagement.

Run after the parent builds clean frozen sources. This runner does not build,
rewrite assets, change caps/presets, skip native inference, supply NY_* flags,
or add synthetic arrival controls. JSON checks are observational and cannot
authorize a verdict. A PASS supplies original route correctness/arrival
evidence, not extra competition solves, independent arithmetic replay, or a
runtime/RSS qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from high_impact_experiments.evidence import (
    ExperimentError, immutable_json, pin, stable_read, unchanged,
)
from high_impact_experiments.original_hull_full_cli import (
    CAPSULE_SHA, POSITIVE, RATIONAL_SHA, clean_environment, command_for, prepare, source_commit,
)
from high_impact_experiments.prepared_route_full_cli import (
    ORDER, PREPARED_SOURCE, complete_class_ids, frozen_solver, inspect_process,
    paired_build_contract, preservation, raw_result_observation, require_result_format,
)
from high_impact_experiments.process import run_command

NATIVE_STATUSES = {"Open", "RootProven", "TreeProven"}
OCCURRENCES = ([(2025, i) for i in (1, 2, 3, 4, 5, 8, 9, 10, 28, 33, 76, 94)]
               + [(2026, i) for i in (1, 2, 3, 4, 5, 17, 20, 29, 31, 38, 43, 80, 146, 578)])
DECLARATION = re.compile(r"\(\s*declare-const\s+([XY]_[0-9]+)\s+Real\s*\)")
OUTPUT = re.compile(r"\(\s*assert\s*\(\s*(<=|>=|<|>)\s+(Y_[01])\s+(Y_[01])\s*\)\s*\)")


def original_objective(row: dict) -> dict:
    """Observe the pinned binary objective; this is not a proof constructor."""
    path = Path(row["property"]["path"])
    raw, actual = stable_read(path)
    declared = row["property"]
    if (actual["sha256"] != declared["sha256"]
            or actual["size_bytes"] != declared["size_bytes"] or len(raw) > 1 << 20):
        raise ExperimentError("original affine property bytes changed or exceeded grammar surface")
    text = re.sub(r";[^\n]*", "", raw.decode("utf-8"))
    names = DECLARATION.findall(text)
    if Counter(names) != Counter([f"X_{i}" for i in range(30)] + ["Y_0", "Y_1"]):
        raise ExperimentError("original native row requires exact30 input/two output declarations")
    without_declarations = DECLARATION.sub("", text)
    clauses = list(OUTPUT.finditer(without_declarations))
    if len(clauses) != 1:
        raise ExperimentError("original affine objective must be one complete binary output clause")
    clause = clauses[0]
    relation, left, right = clause.groups()
    if left == right:
        raise ExperimentError("original affine objective repeats its output class")
    remainder = without_declarations[:clause.start()] + without_declarations[clause.end():]
    if re.search(r"\bY_[0-9]+\b", remainder):
        raise ExperimentError("original affine objective has an additional output expression")
    target, adversary = (left, right) if relation in {"<=", "<"} else (right, left)
    return {"target": int(target[2:]), "classes": [int(adversary[2:])],
            "unsafe_relation": relation, "strict_unsafe": relation in {"<", ">"},
            "coordinate_map": "X_j=original-input[0,j]", "input_count": 30,
            "authority": "observation_only"}


def route_resource_observation(observation: dict) -> dict:
    """Retain actual phase receipts to diagnose a refused original arrival."""
    prepared = observation["prepared"]
    events = prepared["events"]
    if not events:
        return {"attribution": prepared["attribution"], "receipt": None,
                "counters_are_not_measurements": True}
    reason = events[0]["reason"]
    counters = {}
    for key in ("byte_cap", "work_allowance", "work_reserved", "peak_leased_bytes",
                "class_count", "max_expansions"):
        matches = re.findall(r"\b" + key + r": (\d+)\b", reason)
        if len(matches) == 1:
            counters[key] = int(matches[0])
    return {"attribution": prepared["attribution"], "receipt": reason, "counters": counters,
            "counters_are_not_measurements": True}


def declared_pin(record: dict) -> dict:
    """Compare the fresh stable pin with the original declared bytes."""
    actual = pin(Path(record["path"]))
    if (actual["sha256"] != record["sha256"]
            or actual["size_bytes"] != record.get("size_bytes", actual["size_bytes"])):
        raise ExperimentError("original declaration changed before launch pin: " + record["path"])
    return actual


def require_native_arrival(observation: dict, row: dict, objective: dict) -> dict:
    target, classes = objective.get("target"), objective.get("classes")
    if (isinstance(target, bool) or not isinstance(target, int) or target not in (0, 1)
            or not complete_class_ids(classes) or classes != [1 - target]):
        raise ExperimentError("invalid complete original binary objective observation")
    prepared = observation["prepared"]
    positive = (row["year"], row["source_index"]) in POSITIVE
    if prepared["attribution"] == "preempted_by_prior_sat":
        if positive:
            raise ExperimentError("independently positive original did not reach prepared publication")
        if observation["verdict"] != "sat" or observation["bundle"] is not None:
            raise ExperimentError("upfront SAT attribution must retain actual SAT result")
        return {"native_arrived": False, "upfront_sat": True, "positive_required": False}
    status = prepared.get("status")
    if status not in NATIVE_STATUSES:
        raise ExperimentError("original native arrival refused: " + prepared["attribution"])
    if prepared.get("class_count") != len(objective["classes"]):
        raise ExperimentError("native source changed its complete original class count")
    if prepared["proof_arrived"]:
        bundle = observation["bundle"]
        if (observation["verdict"] != "unsat" or observation["final_source"] != PREPARED_SOURCE
                or not isinstance(bundle, dict) or not complete_class_ids(bundle.get("required"))
                or bundle["required"] != objective["classes"]):
            raise ExperimentError("native prepared receipt changed complete original ordered objective")
    elif positive:
        raise ExperimentError("independently positive original stayed Open instead of prepared publication")
    return {"native_arrived": True, "upfront_sat": False, "status": status,
            "positive_required": positive, "positive_published": prepared["proof_arrived"] if positive else None}


def affine_preservation(records: list[dict]) -> dict:
    # A failed arrival check must never hide an already observed opposite
    # settled verdict. Stop even if another artifact/cap check also failed.
    decided = ({r.get("observation", {}).get("verdict") for r in records}
               | {r.get("raw_result", {}).get("verdict") for r in records}) & {"sat", "unsat"}
    result = preservation(records)
    if len(decided) > 1:
        result["passed"] = False
        result["contrary_decided_verdict"] = True
        result["errors"].append("contrary decided verdict: stop for source/witness/proof audit")
    return result


def cohort_summary(cases: list[dict]) -> dict:
    keys = [(case.get("year"), case.get("source_index")) for case in cases]
    complete = keys == OCCURRENCES and all(
        isinstance(year, int) and not isinstance(year, bool)
        and isinstance(index, int) and not isinstance(index, bool) for year, index in keys)
    records = [record for case in cases for record in case["arms"]]
    count = len(records)
    passed = complete and count == 78 and all(affine_preservation(case["arms"])["passed"] for case in cases)
    candidate = [r for r in records if r["arm"] == "candidate_default"]
    positives = [case for case in cases if (case["year"], case["source_index"]) in POSITIVE]
    arrived = (len(candidate) == 26 and all(r.get("passed") and "native_arrival" in r for r in candidate))
    four = len(positives) == 4 and all(any(
        r["arm"] == "candidate_default" and r.get("passed")
        and r.get("native_arrival", {}).get("positive_published") is True
        for r in case["arms"]) for case in positives)
    return {"passed": passed and arrived and four, "complete_occurrence_coverage": complete,
            "solver_processes": count, "preservation_passed": passed,
            "all26_original_native_or_upfront_sat_passed": arrived,
            "four_original_positive_prepared_publications_passed": four,
            "native_outcomes": sum(r.get("native_arrival", {}).get("native_arrived") is True for r in candidate),
            "upfront_sat": sum(r.get("native_arrival", {}).get("upfront_sat") is True for r in candidate),
            "failed_candidate_arrivals": [
                {"year": case["year"], "source_index": case["source_index"],
                 "error": r.get("error"), "resources": r.get("route_resources"),
                 "diagnostic_lines": r.get("prepared_diagnostic_lines", [])}
                for case in cases for r in case["arms"]
                if r["arm"] == "candidate_default" and not r.get("passed")]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "binary", "reference-source", "reference-binary", "capsule", "rational-controls", "archive"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    capsule_path, rational_path = args.capsule.resolve(strict=True), args.rational_controls.resolve(strict=True)
    capsule, rows = prepare(capsule_path, rational_path)
    if [(row["year"], row["source_index"]) for row in rows] != OCCURRENCES:
        raise ExperimentError("original affine gate occurrence coverage changed")
    objectives = [original_objective(row) for row in rows]
    if args.prepare_only:
        print("PREPARED_AFFINE_FULL_CLI_PREPARATION_VALID cases=26 arms=3 positives=4 execution=false", flush=True)
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
    here = Path(__file__).resolve()
    pins = [declared_pin({"path": str(capsule_path), "sha256": CAPSULE_SHA}),
            declared_pin({"path": str(rational_path), "sha256": RATIONAL_SHA}), pin(here),
            candidate["binary_pin"], control["binary_pin"]]
    for name in ("cohort.py", "evidence.py", "process.py", "original_hull_full_cli.py", "prepared_route_full_cli.py"):
        pins.append(pin(here.parent / name))
    pins.append(declared_pin(capsule["profiles"]["off"]["file"]))
    for cohort in capsule["cohorts"]:
        pins.append(declared_pin(cohort["instances_csv_pin"]))
    for row in rows:
        pins.extend(declared_pin(row[key]) for key in ("model", "property"))
    immutable_json(archive / "launch.json", {"schema": "ny.prepared-affine-full-cli.launch.v1",
        "candidate": candidate, "reference": control, "pins": pins, "order": ORDER, "environment": env,
        "official_timeout_seconds": 20, "original_occurrences": 26, "objectives": objectives,
        "required_positive_occurrences": sorted(POSITIVE), "native_inference_override": None,
        "synthetic_arrivals": 0, "identity_probe_processes": 4, "independent_arithmetic_replay": False,
        "performance_qualification": False, "legacy_phases_joined_to_prepared_ledger": False})

    def unchanged_inputs() -> None:
        for solver in (candidate, control):
            if source_commit(Path(solver["source"])) != solver["commit"]:
                raise ExperimentError("frozen source changed during original affine gate")
        for record in pins:
            unchanged(record)

    def execute(row: dict, objective: dict, arm: str, directory: Path) -> dict:
        unchanged_inputs()
        solver = candidate if arm == "candidate_default" else control
        directory.mkdir()
        command = command_for(row, "off_before", Path(solver["binary"]), directory / "solver-result", capsule)
        immutable_json(directory / "request.json", {"argv": command, "row": row, "environment": env,
            "timeout_seconds": 20, "actual_full_cli": True, "original_objective_observation": objective,
            "positive_prepared_publication_required": (row["year"], row["source_index"]) in POSITIVE})
        print(f"PREPARED_AFFINE_FULL_CLI_START arm={arm} directory={directory.name}", flush=True)
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
                process, solver["stamp"])
            if solver is candidate:
                record["route_resources"] = route_resource_observation(record["observation"])
            require_result_format(record)
            if solver is candidate:
                record["native_arrival"] = require_native_arrival(record["observation"], row, objective)
            record["passed"] = True
        except (ExperimentError, OSError, ValueError) as error:
            record["error"] = str(error)
        stderr = directory / "stderr.log"
        if stderr.is_file() and not stderr.is_symlink() and stderr.stat().st_size <= 16 << 20:
            record["prepared_diagnostic_lines"] = [line[:2048] for line in stderr.read_text(errors="replace").splitlines()
                if line.startswith("[prepared-route]")][:64]
        record["artifacts"] = [pin(path) for path in sorted(directory.iterdir()) if path.is_file()]
        immutable_json(directory / "receipt.json", record)
        print(f"PREPARED_AFFINE_FULL_CLI_DONE arm={arm} directory={directory.name} passed={record['passed']}", flush=True)
        if process["interrupted"]:
            raise KeyboardInterrupt
        return record

    cases = []
    try:
        for row, objective in zip(rows, objectives):
            case = {"year": row["year"], "source_index": row["source_index"], "occurrence": row["occurrence"],
                    "objective": objective, "arms": []}
            for arm in ORDER:
                case["arms"].append(execute(row, objective, arm, archive / f"{row['year']}-{row['source_index']}-{arm}"))
                early = affine_preservation(case["arms"])
                if early["contrary_decided_verdict"]:
                    case["preservation"] = early
                    cases.append(case)
                    raise ExperimentError("contrary original decided verdict; stop for soundness audit")
            case["preservation"] = affine_preservation(case["arms"])
            cases.append(case)
            if case["preservation"]["contrary_decided_verdict"]:
                raise ExperimentError("contrary original decided verdict; stop for soundness audit")
        unchanged_inputs()
        for record in [r for case in cases for r in case["arms"]]:
            for artifact in record["artifacts"]:
                unchanged(artifact)
        summary = cohort_summary(cases)
        immutable_json(archive / "report.json", {"schema": "ny.prepared-affine-full-cli.report.v1",
            "status": "PASS" if summary["passed"] else "FAIL", "candidate": candidate, "reference": control,
            "cases": cases, "summary": summary, "actual_full_cli": True, "synthetic_arrivals": 0,
            "original_models_may_decline_prepared_source": False, "official_score_credit": False,
            "independent_arithmetic_replay": False, "performance_qualification": False,
            "loader_and_legacy_memory_qualified": False})
        print(f"PREPARED_AFFINE_FULL_CLI_COHORT_DONE rc={0 if summary['passed'] else 1} cases={len(cases)} "
              f"processes={summary['solver_processes']}", flush=True)
        return 0 if summary["passed"] else 1
    except BaseException as error:
        immutable_json(archive / "abort.json", {"status": "FAIL", "error": type(error).__name__ + ": " + str(error),
                       "completed_cases": cases})
        raise


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    try:
        raise SystemExit(main())
    except (ExperimentError, OSError, ValueError) as error:
        print("PREPARED_AFFINE_FULL_CLI_ABORT " + str(error), flush=True)
        raise SystemExit(1)
