#!/usr/bin/env python3
"""Run every pinned SafeNLP resource occurrence with its original process cap.

This is source-contract correctness evidence. It does not measure speed, prove
the compiler, run the full competition route, or mint independent certificates.
Compilation finishes before any original 20-second instance process begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

CAPSULE_SHA = "45ed7de826c06ab870d76a9d696001b7aafbd0c75f9834986f5163a7bee9c5c6"
ARCHIVE_ROOT = Path("<home>/ny-experiments/high-impact/2026-10-01")
CONTROL = "hull_resource_cli_original_safenlp_strict20s"
POSITIVE_CONTROLS = {(2025, 5), (2025, 9), (2026, 2), (2026, 3)}
ACTIVE_ARCHIVE = None
ACTIVE_STAGE = "preparation"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            result.update(block)
    return result.hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def source_identity(source: Path) -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source).decode().strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=source
    )
    if dirty:
        raise RuntimeError("source-contract gate requires a clean tracked source")
    return commit


def stop_group(process: subprocess.Popen) -> int:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def run_process(command: list[str], source: Path, env: dict,
                stdout: Path, stderr: Path, cap: float | None) -> dict:
    timed_out, interrupted = False, False
    started = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(command, cwd=source, env=env, stdout=out,
                                   stderr=err, start_new_session=True)
        try:
            remaining = None if cap is None else max(0.0, cap - (time.monotonic() - started))
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = stop_group(process)
        except KeyboardInterrupt:
            interrupted = True
            returncode = stop_group(process)
        except BaseException:
            stop_group(process)
            raise
    elapsed = time.monotonic() - started
    return {"returncode": returncode, "timed_out": timed_out, "interrupted": interrupted,
            "process_wall_seconds_diagnostic_only": elapsed,
            "process_cap_seconds": cap,
            "within_process_cap": cap is None or elapsed <= cap}


def main() -> int:
    global ACTIVE_ARCHIVE, ACTIVE_STAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true",
                        help="validate original hashes/official join without building or executing")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    capsule_path = args.capsule.resolve(strict=True)
    archive = args.archive.resolve()
    if not archive.is_relative_to(ARCHIVE_ROOT) or archive == ARCHIVE_ROOT:
        raise RuntimeError("compiled original probe requires its exclusive archive root")
    if digest(capsule_path) != CAPSULE_SHA:
        raise RuntimeError("original 26-occurrence capsule identity mismatch")
    capsule = json.loads(capsule_path.read_text())
    if capsule["schema"] != "ny.original-safenlp-hull-strict20s-capsule.v2":
        raise RuntimeError("unsupported original capsule")
    declared = capsule["rows"]
    if len(declared) != 26:
        raise RuntimeError("complete original cohort requires all 26 occurrences")
    # Reuse the production cohort join, including official CSV multiplicities
    # and budgets. Capsule status strings are never a correctness oracle.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "scripts"))
    from high_impact_experiments.cohort import load_cohort

    joined = []
    for config in capsule["cohorts"]:
        csv_pin = config["instances_csv_pin"]
        if digest(Path(csv_pin["path"])) != csv_pin["sha256"]:
            raise RuntimeError("original instance CSV identity mismatch")
        rows, _ = load_cohort(config, capsule_path.parent)
        year = int(config["suite"].removeprefix("vnncomp"))
        for row in rows:
            joined.append((year, config["source_commit"], row))
    if len(joined) != 26:
        raise RuntimeError("official occurrence join lost coverage")
    for (year, commit, actual), frozen in zip(joined, declared):
        for key in ("source_index", "occurrence", "source_csv_sha256",
                    "declared_model", "declared_property"):
            if actual[key] != frozen[key]:
                raise RuntimeError("original occurrence join mismatch: " + key)
        if year != frozen["year"] or commit != frozen["source_commit"]:
            raise RuntimeError("original suite/source changed")
        if actual["timeout_seconds"] != 20 or frozen["official_timeout_seconds"] != 20:
            raise RuntimeError("original official process budget changed")
        for key in ("model", "property"):
            if any(actual[key][item] != frozen[key][item]
                   for item in ("path", "sha256", "size_bytes")):
                raise RuntimeError("original asset join changed: " + key)
            asset = Path(actual[key]["path"])
            if digest(asset) != actual[key]["sha256"]:
                raise RuntimeError("original asset bytes changed: " + key)
    root_profile = capsule["profiles"]["root"]["file"]
    if digest(Path(root_profile["path"])) != root_profile["sha256"]:
        raise RuntimeError("typed original Root preset changed")
    if args.prepare_only:
        print("ORIGINAL_HULL_PREPARATION_VALID cases=26 execution=false", flush=True)
        return 0
    commit = source_identity(source)
    archive.mkdir(parents=True, exist_ok=False)
    ACTIVE_ARCHIVE = archive
    runner_path = Path(__file__).resolve()
    runner_sha = digest(runner_path)
    source_pins = {str(path.relative_to(source)): digest(path) for path in (
        runner_path, source / "scripts/high_impact_experiments/cohort.py",
        source / "scripts/high_impact_experiments/evidence.py",
    )}

    def unchanged_inputs() -> bool:
        return (source_identity(source) == commit and
                all(digest(source / path) == sha for path, sha in source_pins.items()))

    write_new(archive / "launch.json", {
        "source_commit": commit, "runner_sha256": runner_sha,
        "capsule_sha256": CAPSULE_SHA, "performance_qualification": False,
        "source_inputs": source_pins,
    })
    env = {key: value for key, value in os.environ.items() if not key.startswith("NY_")}
    env["CARGO_TERM_COLOR"] = "never"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    build_command = [
        "targo", "--unverified", "test", "--locked", "--offline", "-p", "ny-cli",
        "--no-default-features", "--features", "mip,external-vnncomp",
        "--bin", "ny", "--no-run", "--message-format=json",
    ]
    print("ORIGINAL_HULL_BUILD_START source=" + commit, flush=True)
    build_stdout = archive / "build.stdout.jsonl"
    build_stderr = archive / "build.stderr.log"
    ACTIVE_STAGE = "build"
    build = run_process(build_command, source, env, build_stdout, build_stderr, None)
    executables = []
    for line in build_stdout.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if (event.get("reason") == "compiler-artifact"
                and event.get("target", {}).get("name") == "ny"
                and event.get("profile", {}).get("test") is True
                and event.get("executable")
                and {"mip", "external-vnncomp"} <= set(event.get("features", []))):
            executables.append(event)
    build_receipt = {
        "source_commit": commit, "command": build_command, **build,
        "stdout_sha256": digest(build_stdout), "stderr_sha256": digest(build_stderr),
        "compiler_proof": False,
        "artifacts": executables,
    }
    if (build["returncode"] != 0 or build["interrupted"] or len(executables) != 1
            or not unchanged_inputs()):
        write_new(archive / "build.json", build_receipt)
        raise RuntimeError("original external-feature test binary compilation failed")
    binary = archive / "hull-resource-control"
    original_executable = Path(executables[0]["executable"])
    original_sha = digest(original_executable)
    shutil.copyfile(original_executable, binary)
    binary.chmod(0o555)
    if digest(binary) != original_sha:
        raise RuntimeError("compiled binary changed while pinning")
    build_receipt["binary_sha256"] = original_sha
    write_new(archive / "build.json", build_receipt)
    cases = []
    for frozen in declared:
        if not unchanged_inputs() or digest(binary) != original_sha:
            raise RuntimeError("gate source or pinned executable changed")
        year, index = frozen["year"], frozen["source_index"]
        ACTIVE_STAGE = "case:" + str(year) + ":" + str(index)
        directory = archive / (str(year) + "-" + str(index))
        directory.mkdir()
        observation_path = directory / "observation.json"
        request = dict(frozen["resource_request_template"])
        request["receipt_path"] = str(observation_path)
        request_path = directory / "request.json"
        write_new(request_path, request)
        request_sha = digest(request_path)
        case_env = dict(env)
        case_env["NY_HULL_RESOURCE_ORIGINAL_REQUEST"] = str(request_path)
        command = [str(binary), CONTROL, "--test-threads=1", "--show-output"]
        stdout = directory / "stdout.log"
        stderr = directory / "stderr.log"
        print(f"ORIGINAL_HULL_CASE_START year={year} source_index={index}", flush=True)
        process = run_process(command, source, case_env, stdout, stderr, 20)
        text = stdout.read_text(errors="replace")
        named = bool(re.search(r"^test \S*" + CONTROL + r" \.\.\. ok$", text, re.MULTILINE))
        identity_unchanged = unchanged_inputs() and digest(binary) == original_sha
        passed = (process["returncode"] == 0 and not process["timed_out"]
                  and not process["interrupted"] and process["within_process_cap"] and named
                  and identity_unchanged
                  and "test result: ok. 1 passed; 0 failed; 0 ignored;" in text
                  and digest(request_path) == request_sha and observation_path.is_file())
        observation_errors = []
        try:
            observation = json.loads(observation_path.read_text()) if observation_path.is_file() else {}
            if not isinstance(observation, dict):
                raise ValueError("observation must be an object")
        except (OSError, ValueError) as error:
            observation_errors.append(str(error))
            observation = {}
            passed = False
        expected = {
            "schema": "ny.original-safenlp-hull-resource-observation.v1",
            "request_sha256": request_sha, "timeout_seconds": 20,
            "year": year, "source_index": index,
            "source_commit": frozen["source_commit"],
            "source_csv_sha256": frozen["source_csv_sha256"],
            "occurrence": frozen["occurrence"],
            "model_sha256": frozen["model"]["sha256"],
            "property_sha256": frozen["property"]["sha256"],
            "preset_sha256": root_profile["sha256"],
            "production_equivalent_full_route": False, "full_lp_engaged": False,
            "external_independent_arithmetic_replay": False,
        }
        passed = passed and all(observation.get(key) == value for key, value in expected.items())
        rust_elapsed = observation.get("elapsed_seconds")
        passed = passed and (isinstance(rust_elapsed, (int, float))
                             and not isinstance(rust_elapsed, bool)
                             and math.isfinite(rust_elapsed) and 0 <= rust_elapsed <= 20)
        positive = observation.get("observation") == "strict_positive_original_replay"
        if positive:
            lower = observation.get("checked_lower_bound")
            passed = passed and (isinstance(lower, (int, float)) and not isinstance(lower, bool)
                                 and math.isfinite(lower) and lower > 0
                                 and observation.get("private_proof_origin") == "OriginalReplay"
                                 and isinstance(observation.get("scope"), str)
                                 and bool(observation["scope"])
                                 and observation.get("unsafe_alternatives_fully_closed") is True)
        else:
            passed = passed and (observation.get("observation") == "open_or_refused"
                                 and observation.get("private_proof_origin") is None
                                 and observation.get("scope") is None
                                 and observation.get("checked_lower_bound") is None
                                 and observation.get("unsafe_alternatives_fully_closed") is False)
        if (year, index) in POSITIVE_CONTROLS:
            passed = passed and positive
        record = {
            "year": year, "source_index": index, "command": command,
            **process, "identity_unchanged": identity_unchanged,
            "named_control_passed": named, "source_contract_passed": passed,
            "observation_errors": observation_errors,
            "positive_original_replay": positive, "qualified_competition_decision": False,
            "request_sha256": request_sha, "binary_sha256": original_sha,
            "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr),
            "observation_sha256": digest(observation_path) if observation_path.is_file() else None,
        }
        write_new(directory / "result.json", record)
        cases.append(record)
        print(f"ORIGINAL_HULL_CASE_DONE year={year} source_index={index} passed={passed}", flush=True)
        if process["interrupted"]:
            break
    ACTIVE_STAGE = "final-publication"
    identities_valid = unchanged_inputs() and digest(binary) == original_sha
    logs_valid = all(
        digest(archive / (str(case["year"]) + "-" + str(case["source_index"])) / name)
        == case[key]
        for case in cases for name, key in (("stdout.log", "stdout_sha256"),
                                           ("stderr.log", "stderr_sha256"),
                                           ("request.json", "request_sha256"))
    )
    observations_valid = all(
        case["observation_sha256"] is None or
        digest(archive / (str(case["year"]) + "-" + str(case["source_index"]))
               / "observation.json") == case["observation_sha256"]
        for case in cases
    )
    passed = (len(cases) == 26 and identities_valid and logs_valid
              and observations_valid
              and all(case["source_contract_passed"] for case in cases))
    write_new(archive / "report.json", {
        "schema": "ny.original_safenlp_resource_cohort.report.v1", "source_commit": commit,
        "status": "PASS" if passed else "FAIL", "cases": cases,
        "capsule_sha256": CAPSULE_SHA, "binary_sha256": original_sha,
        "runner_sha256": runner_sha, "identity_unchanged": identities_valid,
        "logs_unchanged": logs_valid, "complete_occurrence_coverage": len(cases) == 26,
        "observations_unchanged": observations_valid,
        "independent_arithmetic_replay": False, "performance_qualification": False,
        "full_competition_route": False,
    })
    print("ORIGINAL_HULL_COHORT_DONE rc=" + ("0" if passed else "1")
          + " cases=" + str(len(cases)), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        result = main()
    except BaseException as error:
        if ACTIVE_ARCHIVE is not None:
            pins = {}
            for path in ACTIVE_ARCHIVE.rglob("*"):
                if path.is_file():
                    pins[str(path.relative_to(ACTIVE_ARCHIVE))] = digest(path)
            write_new(ACTIVE_ARCHIVE / "abort.json", {
                "schema": "ny.original_safenlp_resource_cohort.abort.v1",
                "stage": ACTIVE_STAGE, "error": type(error).__name__ + ": " + str(error),
                "status": "FAIL", "artifact_sha256": pins,
            })
        print("ORIGINAL_HULL_COHORT_ABORT " + type(error).__name__ + ": " + str(error), flush=True)
        result = 1
    raise SystemExit(result)
