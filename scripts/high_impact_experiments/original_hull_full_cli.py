#!/usr/bin/env python3
"""Qualify the original26 SafeNLP cohort through the actual scored CLI.

Run under the parent's ny_box correctness hold with a previously built, frozen
binary. This driver never compiles, changes presets, skips native inference or
adds NY_* controls. Each OFF/Root/OFF process has the unchanged official20s cap,
including startup, native loading, preparation, checking, replay and publication.
Flight and proof-bundle records trace an actual private replay receipt through
the writer; their JSON is observational evidence, never certificate authority.
Loader/legacy memory census and independent replay of other decisions remain
separate obligations. A PASS is an original full-route correctness/arrival gate,
not a competition score or a speed/memory qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from high_impact_experiments.cohort import load_cohort
from high_impact_experiments.evidence import (
    ExperimentError, immutable_json, ny_build_source, pin, stable_read, strict_json, unchanged,
)
from high_impact_experiments.process import run_command

CAPSULE_SHA = "45ed7de826c06ab870d76a9d696001b7aafbd0c75f9834986f5163a7bee9c5c6"
RATIONAL_SHA = "2e2a06acfc61782c267ab3407b991e023c14e88c728f0f071a46fd01a264b776"
POSITIVE = {(2025, 5), (2025, 9), (2026, 2), (2026, 3)}
ORDER = ("off_before", "root", "off_after")
TOKENS = {"sat", "unsat", "unknown", "timeout", "error"}


def clean_environment() -> dict[str, str]:
    # PATH is only needed by native loader helpers/watchdog support. No inherited
    # credentials, NY levers, shape policy or external-watchdog bypass arrives.
    return {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C",
            "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1"}


def sha(path: Path) -> str:
    return pin(path)["sha256"]


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def source_commit(source: Path) -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source).decode().strip()
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=source):
        raise ExperimentError("full CLI gate requires a clean tracked source snapshot")
    if ny_build_source(source) != {"dirty": "false", "source": "clean"}:
        raise ExperimentError("uncommitted compiled source cannot qualify a frozen CLI")
    return head


def prepare(capsule_path: Path, rational_path: Path) -> tuple[dict, list[dict]]:
    if sha(capsule_path) != CAPSULE_SHA or sha(rational_path) != RATIONAL_SHA:
        raise ExperimentError("original capsule/rational controls changed")
    capsule = strict_json(capsule_path)
    if capsule.get("schema") != "ny.original-safenlp-hull-strict20s-capsule.v2":
        raise ExperimentError("unsupported original capsule")
    declared = capsule.get("rows", [])
    joined = []
    for config in capsule["cohorts"]:
        unchanged_pin = config["instances_csv_pin"]
        if sha(Path(unchanged_pin["path"])) != unchanged_pin["sha256"]:
            raise ExperimentError("original official CSV changed")
        rows, _ = load_cohort(config, capsule_path.parent)
        year = int(config["suite"].removeprefix("vnncomp"))
        for row in rows:
            joined.append((year, config["source_commit"], row))
    if len(joined) != 26 or len(declared) != 26:
        raise ExperimentError("all26 original occurrences are mandatory")
    for (year, commit, actual), frozen in zip(joined, declared):
        for name in ("source_index", "occurrence", "source_csv_sha256", "declared_model", "declared_property"):
            if actual[name] != frozen[name]:
                raise ExperimentError("original occurrence mismatch: " + name)
        if year != frozen["year"] or commit != frozen["source_commit"]:
            raise ExperimentError("original year/commit changed")
        if actual["timeout_seconds"] != 20 or frozen["official_timeout_seconds"] != 20:
            raise ExperimentError("original official20s cap changed")
        for name in ("model", "property"):
            for key in ("path", "sha256", "size_bytes"):
                if actual[name][key] != frozen[name][key]:
                    raise ExperimentError("original asset join changed: " + name)
            if sha(Path(frozen[name]["path"])) != frozen[name]["sha256"]:
                raise ExperimentError("original asset bytes changed")
    # The profile pair changes exactly one typed line; all other native-loader,
    # attack, phase and flush policies are the original frozen treatment.
    profiles = capsule["profiles"]
    for arm in ("root", "off"):
        record = profiles[arm]["file"]
        if sha(Path(record["path"])) != record["sha256"]:
            raise ExperimentError("original typed profile changed")
    root_bytes = Path(profiles["root"]["file"]["path"]).read_bytes()
    off_bytes = Path(profiles["off"]["file"]["path"]).read_bytes()
    if root_bytes.count(b"  single_hidden_hull: root\n") != 1 or root_bytes.replace(
            b"  single_hidden_hull: root\n", b"  single_hidden_hull: off\n") != off_bytes:
        raise ExperimentError("OFF/Root profiles differ beyond typed hull mode")
    rational = strict_json(rational_path)
    positives = {(r["year"], r["instance_index"] + 1): r for r in rational["results"]
                 if r["bound_sign"] == "positive"}
    if set(positives) != POSITIVE:
        raise ExperimentError("independent rational-positive controls changed")
    for row in declared:
        control = positives.get((row["year"], row["source_index"]))
        if control and (control["onnx_sha256"] != row["model"]["sha256"]
                        or control["vnnlib_sha256"] != row["property"]["sha256"]
                        or int(control["exact_lower_numerator"]) <= 0
                        or int(control["exact_lower_denominator"]) <= 0):
            raise ExperimentError("rational original positive binding/sign changed")
    return capsule, declared


def command_for(row: dict, arm: str, binary: Path, result: Path, capsule: dict) -> list[str]:
    mode = "root" if arm == "root" else "off"
    expected = ["__FROZEN_NY_BINARY__", "vnncomp", "v1", "safenlp_2024",
                row["model"]["path"], row["property"]["path"], "__EXCLUSIVE_RESULT_PATH__", "20",
                "--configs-dir", capsule["profiles"][mode]["configs_root"], "--proof-profile", "proof"]
    if row["argv_templates"][mode] != expected:
        raise ExperimentError("original literal full-CLI argv changed")
    return [str(binary) if arg == "__FROZEN_NY_BINARY__" else str(result)
            if arg == "__EXCLUSIVE_RESULT_PATH__" else arg for arg in expected]


def inspect_bundle(path: Path, row: dict, stamp: str) -> dict:
    """Validate final stream integrity/source trace, not replay its mathematics."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 << 20:
        raise ExperimentError("missing/unbounded final proof bundle")
    raw, _ = stable_read(path)
    lines = raw.splitlines(keepends=True)
    if len(lines) < 4 or any(len(line) > 1 << 20 or not line.endswith(b"\n") for line in lines):
        raise ExperimentError("incomplete/unbounded proof stream")
    def reject_constant(value):
        raise ExperimentError("nonfinite proof JSON value: " + value)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExperimentError("duplicate proof JSON field: " + key)
            result[key] = value
        return result

    records = [json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique) for line in lines]
    if any(not isinstance(record, dict) for record in records):
        raise ExperimentError("proof record must be an object")
    end = records[-1]
    if (end.get("kind") != "end" or end.get("lines") != len(lines)
            or end.get("sha256") != hashlib.sha256(b"".join(lines[:-1])).hexdigest()):
        raise ExperimentError("proof stream final digest/count changed")
    header = records[0]
    if (header.get("kind") != "header" or header.get("schema") != "ny-proof-bundle/v2"
            or header.get("build") != stamp or header.get("profile") != "proof"
            or header.get("lane") != "vnncomp_v1"):
        raise ExperimentError("unexpected final proof header")
    for source, key in (("onnx", "model"), ("vnnlib", "property")):
        if not isinstance(header.get(source), dict) or header[source].get("sha256") != row[key]["sha256"]:
            raise ExperimentError("published proof original input hash mismatch")
    published = [record for record in records if record.get("kind") == "published"]
    verdicts = [record for record in records if record.get("kind") == "verdict"]
    if published != [{"kind": "published", "result": "unsat"}] or len(verdicts) != 1:
        raise ExperimentError("final bundle must describe exactly one published UNSAT")
    verdict = verdicts[0]
    if not isinstance(verdict.get("source"), str):
        raise ExperimentError("published source missing")
    if verdict["source"] == "original_hull_replay":
        # Required `[0]` is this adapter's sole complete original unsafe clause.
        margin = verdict.get("min_margin")
        if verdict.get("required") != [0] or not finite(margin) or margin <= 0:
            raise ExperimentError("original replay source lacks positive complete objective")
    return {"source": verdict["source"], "min_margin": verdict.get("min_margin"),
            "required": verdict.get("required"), "stream_sha256": sha(path),
            "arithmetic_replayed_by_driver": False}


def inspect_flight(path: Path, arm: str, profile: Path, stamp: str, verdict: str) -> dict:
    flight = strict_json(path)
    if (flight.get("schema_version") != 3 or flight.get("budget_secs") != 20
            or flight.get("category") != "safenlp_2024" or flight.get("build_id") != stamp
            or "record_status" in flight):
        raise ExperimentError("wrong/incomplete original full-CLI flight identity")
    ambient = flight.get("ambient_env")
    if not isinstance(ambient, dict) or any(name.startswith("NY_") for name in ambient):
        raise ExperimentError("full-CLI arrival must use typed preset with zero NY_* controls")
    events = flight.get("events")
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ExperimentError("missing flight event trace")
    complete = [event for event in events if event.get("method") == "run_complete"]
    if (len(complete) != 1 or complete[0].get("status") != "complete"
            or complete[0].get("reason") != verdict
            or not finite(complete[0].get("at_secs")) or not 0 <= complete[0]["at_secs"] <= 20):
        raise ExperimentError("original publication did not complete under the same20s clock")
    if not any(event.get("method") == "preset_base" and event.get("status") == "ran"
               and event.get("reason") == str(profile) for event in events):
        raise ExperimentError("original typed base preset did not arrive")
    hull = [event for event in events if event.get("method") == "hull_resource"]
    arrived = any(event.get("status") == "ran" and event.get("reason") ==
                  "outcome=verified origin=original-replay census=participating-hull-only" for event in hull)
    if arm != "root" and hull:
        raise ExperimentError("OFF acquired new hull resource authority")
    if arrived and verdict != "unsat":
        raise ExperimentError("positive replay trace did not reach actual UNSAT publication")
    return {"positive_original_replay_arrived": arrived, "events": hull,
            "preset_source": "typed_original_profile", "published_at_seconds": complete[0]["at_secs"]}


def inspect_case(directory: Path, arm: str, row: dict, capsule: dict,
                 process: dict, stamp: str) -> dict:
    if (process.get("returncode") != 0 or process.get("timed_out") or process.get("interrupted")
            or not finite(process.get("wall_seconds")) or not 0 <= process["wall_seconds"] <= 20
            or process.get("timeout_seconds") != 20):
        raise ExperimentError("late/failed/aborted full-CLI process cannot qualify")
    path = directory / "solver-result"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1 << 20:
        raise ExperimentError("missing/unbounded actual result artifact")
    text = stable_read(path)[0].decode()
    verdict = text.splitlines()[0] if text.splitlines() else ""
    if verdict not in TOKENS or verdict == "error":
        raise ExperimentError("invalid/error full-CLI result")
    profile = Path(capsule["profiles"]["root" if arm == "root" else "off"]["file"]["path"])
    flight = inspect_flight(directory / "solver-result.flight.json", arm, profile, stamp, verdict)
    bundle_path = directory / "solver-result.proof.jsonl"
    bundle = inspect_bundle(bundle_path, row, stamp) if verdict == "unsat" else None
    source = None if bundle is None else bundle["source"]
    arrived = flight["positive_original_replay_arrived"]
    if arrived != (source == "original_hull_replay"):
        raise ExperimentError("captured replay and actual receipt publication source disagree")
    if arm != "root" and source == "original_hull_replay":
        raise ExperimentError("OFF retained an original hull receipt")
    if arm == "root" and (row["year"], row["source_index"]) in POSITIVE and not arrived:
        raise ExperimentError("independently positive original must traverse full CLI to replay receipt")
    return {"verdict": verdict, "flight": flight, "bundle": bundle,
            "positive_original_replay": arrived, "full_cli_completed": True,
            "official_score_credit": False}


def preservation(records: list[dict]) -> dict:
    by_arm = {record["arm"]: record for record in records}
    errors = []
    if set(by_arm) != set(ORDER) or len(records) != 3:
        return {"passed": False, "errors": ["incomplete OFF/Root/OFF coverage"], "contrary_decided_verdict": False}
    if not all(record.get("passed") is True for record in records):
        return {"passed": False, "errors": ["one full-CLI arm did not qualify"], "contrary_decided_verdict": False}
    tokens = {arm: by_arm[arm]["observation"]["verdict"] for arm in ORDER}
    decided = {token for token in tokens.values() if token in {"sat", "unsat"}}
    contrary = len(decided) > 1
    if contrary:
        errors.append("contrary decided verdicts: stop for source/witness/proof audit")
    for arm in ("off_before", "off_after"):
        if tokens[arm] in {"sat", "unsat"} and tokens["root"] != tokens[arm]:
            errors.append("Root failed to preserve settled " + arm + " decision")
    return {"passed": not errors, "errors": errors, "contrary_decided_verdict": contrary,
            "verdicts": tokens, "OFF_stability_diagnostic": tokens["off_before"] == tokens["off_after"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--rational-controls", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    capsule_path, rational_path = args.capsule.resolve(strict=True), args.rational_controls.resolve(strict=True)
    capsule, rows = prepare(capsule_path, rational_path)
    if args.prepare_only:
        print("ORIGINAL_HULL_FULL_CLI_PREPARATION_VALID cases=26 arms=3 execution=false", flush=True)
        return 0
    commit = source_commit(source)
    archive = args.archive.resolve()
    if archive == source or source in archive.parents:
        raise ExperimentError("exclusive full-CLI archive must be outside source")
    archive.mkdir(parents=True, exist_ok=False)
    original = pin(args.binary.resolve(strict=True))
    if not os.access(original["path"], os.X_OK):
        raise ExperimentError("frozen full-CLI binary is not executable")
    binary = archive / "frozen-solver"
    shutil.copyfile(original["path"], binary)
    binary.chmod(0o555)
    unchanged(original)
    if sha(binary) != original["sha256"]:
        raise ExperimentError("binary changed during immutable copy")
    env = clean_environment()
    build_identity = run_command([str(binary), "--build-id", "--json"], cwd=source, env=env,
                                 stdout=archive / "build-id.json", stderr=archive / "build-id.stderr", timeout_seconds=30)
    identity = strict_json(archive / "build-id.json")
    if (build_identity["returncode"] != 0 or build_identity["timed_out"] or build_identity["interrupted"]
            or identity.get("schema") != "NY_BUILD_ID_V1" or identity.get("commit") != commit
            or identity.get("dirty") != "false" or identity.get("source_digest") != "clean"
            or identity.get("exe_sha256") != original["sha256"]
            or "mip" not in identity.get("features", "").split(",")):
        raise ExperimentError("full-CLI executable/source/mip build identity mismatch")
    stamp_process = run_command([str(binary), "--build-stamp"], cwd=source, env=env,
                                stdout=archive / "build-stamp.txt", stderr=archive / "build-stamp.stderr", timeout_seconds=30)
    stamp = (archive / "build-stamp.txt").read_text().strip()
    if (stamp_process["returncode"] != 0 or stamp_process["timed_out"] or stamp_process["interrupted"] or "\n" in stamp
            or not stamp.startswith("NY_BUILD_STAMP_V1 ")
            or "stamp=" + identity["stamp_sha256"] not in stamp.split()):
        raise ExperimentError("binary free stamp does not match bound executable identity")
    pins = [pin(capsule_path), pin(rational_path), pin(binary), pin(Path(__file__).resolve())]
    for arm in ("root", "off"):
        pins.append(pin(Path(capsule["profiles"][arm]["file"]["path"])))
    for row in rows:
        pins.extend(pin(Path(row[key]["path"])) for key in ("model", "property"))
    immutable_json(archive / "launch.json", {"schema": "ny.original-safenlp-full-cli.launch.v1",
        "source_commit": commit, "identity": identity, "pins": pins, "environment": env,
        "order": ORDER, "official_timeout_seconds": 20, "cases": 26,
        "shape_inference_override": None, "performance_qualification": False,
        "memory_census": "participating-hull-only;loader-and-legacy-engines-unqualified"})

    def unchanged_inputs() -> None:
        if source_commit(source) != commit:
            raise ExperimentError("full-CLI source snapshot changed")
        for record in pins:
            unchanged(record)

    cases = []
    try:
        for row in rows:
            case = {"year": row["year"], "source_index": row["source_index"], "occurrence": row["occurrence"], "arms": []}
            for arm in ORDER:
                unchanged_inputs()
                directory = archive / f"{row['year']}-{row['source_index']}-{arm}"
                directory.mkdir()
                command = command_for(row, arm, binary, directory / "solver-result", capsule)
                immutable_json(directory / "request.json", {"argv": command, "environment": env,
                    "row": row, "timeout_seconds": 20, "actual_full_cli": True})
                print(f"ORIGINAL_HULL_FULL_CLI_START year={row['year']} source_index={row['source_index']} arm={arm}", flush=True)
                process = run_command(command, cwd=source, env=env, stdout=directory / "stdout.log",
                                      stderr=directory / "stderr.log", timeout_seconds=20)
                record = {"arm": arm, "process": process, "passed": False}
                try:
                    unchanged_inputs()
                    record["observation"] = inspect_case(directory, arm, row, capsule, process, stamp)
                    record["passed"] = True
                except (ExperimentError, OSError, ValueError) as error:
                    record["error"] = str(error)
                record["artifacts"] = [pin(path) for path in sorted(directory.iterdir()) if path.is_file()]
                immutable_json(directory / "receipt.json", record)
                case["arms"].append(record)
                print(f"ORIGINAL_HULL_FULL_CLI_DONE year={row['year']} source_index={row['source_index']} arm={arm} passed={record['passed']}", flush=True)
                if process["interrupted"]:
                    raise KeyboardInterrupt
            case["preservation"] = preservation(case["arms"])
            cases.append(case)
            if case["preservation"]["contrary_decided_verdict"]:
                raise ExperimentError("contrary original decided verdicts; stop cohort for soundness audit")
        unchanged_inputs()
        for case in cases:
            for record in case["arms"]:
                for artifact in record["artifacts"]:
                    unchanged(artifact)
        passed = len(cases) == 26 and all(case["preservation"]["passed"] for case in cases)
        report = {"schema": "ny.original-safenlp-full-cli.report.v1", "source_commit": commit,
                  "status": "PASS" if passed else "FAIL", "cases": cases,
                  "complete_occurrence_coverage": len(cases) == 26, "actual_full_cli": True,
                  "expected_positive_replay_arrivals": 4, "official_score_credit": False,
                  "independent_arithmetic_replay_by_driver": False, "performance_qualification": False,
                  "loader_and_legacy_memory_qualified": False}
        immutable_json(archive / "report.json", report)
        print(f"ORIGINAL_HULL_FULL_CLI_COHORT_DONE rc={0 if passed else 1} cases={len(cases)} processes={sum(len(c['arms']) for c in cases)}", flush=True)
        return 0 if passed else 1
    except BaseException as error:
        immutable_json(archive / "abort.json", {"status": "FAIL", "error": type(error).__name__ + ": " + str(error), "completed_cases": cases})
        raise


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    try:
        raise SystemExit(main())
    except (ExperimentError, OSError, ValueError, subprocess.SubprocessError) as error:
        print("ORIGINAL_HULL_FULL_CLI_ABORT " + str(error), flush=True)
        raise SystemExit(1)
