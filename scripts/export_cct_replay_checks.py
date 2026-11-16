#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Export existing CCT replay receipts as row-bound --ny-ce-checks records.

This reads and hashes evidence; it never invokes a solver, imports a checker, or
replays a model. The supplied CSV must describe the actual pilot runs, including
their observed wall times and zero prepare time. Old/control replay batches are
accepted only with --validation-only, which the scorer refuses for NY credit.
These local CPU replay records do not claim official competition execution or
authenticate historical model provenance beyond their explicit payload pins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any


CHECKER_COMMIT = "eea564f5d0a8751d8321a409214db14d03f452b8"
RESEARCH_ADAPTER = "ny_research_actual_sat_replay_adapter_v2"
OLD_F3 = "f3a24a21a07233f16d92e1e390cb4ce93d6659c2"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
RESULTS = {
    "correct": ("correct", True, True),
    "correct_up_to_tolerance": ("correct_with_tolerance", False, True),
    "no_ce": ("missing", False, False),
    "malformed_ce": ("malformed", False, False),
    "unsupported": ("unsupported", False, False),
    "exec_doesnt_match": ("incorrect", False, False),
    "spec_not_violated": ("incorrect", False, False),
    "wrong_shape": ("incorrect", False, False),
}


class EvidenceError(ValueError):
    """A requested row lacks a complete, consistent existing evidence chain."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha(value: Any, length: int = 64) -> str:
    require(isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None,
            f"expected a lowercase {length}-hex digest")
    return value


def pin(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), f"not a regular evidence file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(all(getattr(before, key) == getattr(after, key) for key in fields),
            f"evidence changed while hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": after.st_size}


def verify_pin(expected: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(expected, dict), "missing evidence pin")
    current = pin(Path(expected["path"]))
    size = expected.get("bytes", expected.get("size_bytes"))
    require(isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            "evidence pin has no byte size")
    require(current["sha256"] == sha(expected["sha256"]) and current["bytes"] == size,
            f"evidence bytes differ: {current['path']}")
    return current


def load(path: Path) -> dict[str, Any]:
    require(path.stat().st_size <= 8 << 20, f"oversized metadata: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected metadata object: {path}")
    return value


def row_identity(row: list[str]) -> str:
    require(len(row) == 6 and all(isinstance(field, str) for field in row),
            "result row must have exactly six string fields")
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalized_suffix(field: str, anchor: str) -> str:
    parts: list[str] = []
    for part in field.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            require(bool(parts) and parts[-1] != "..", "path escapes its lexical root")
            parts.pop()
        else:
            parts.append(part)
    if anchor in parts:
        parts = parts[len(parts) - 1 - parts[::-1].index(anchor):]
    return "/".join(parts)


def occurrence_key(row: list[str]) -> tuple[str, str, str]:
    category = row[0].removeprefix("2026_")
    return (category, normalized_suffix(row[1], "onnx"), normalized_suffix(row[2], "vnnlib"))


def read_rows(path: Path) -> list[tuple[list[str], int]]:
    rows: list[tuple[list[str], int]] = []
    seen: dict[tuple[str, str, str], int] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for index, row in enumerate(csv.reader(stream)):
            if index == 0 and row == ["category", "onnx", "vnnlib", "prepare_time", "result", "runtime"]:
                continue
            require(len(row) == 6, f"CSV row {index + 1} is not six columns")
            key = occurrence_key(row)
            occurrence = seen.get(key, 0)
            seen[key] = occurrence + 1
            if row[4].strip().lower() in {"sat", "violated"}:
                normalized = row.copy()
                normalized[4] = "violated"
                rows.append((normalized, occurrence))
    return rows


def completed(process: dict[str, Any], budget: float | None = None) -> None:
    require(process.get("returncode") == 0 and process.get("timed_out") is False
            and process.get("interrupted") is False, "process did not complete successfully")
    seconds = process.get("wall_seconds")
    require(isinstance(seconds, (float, int)) and not isinstance(seconds, bool)
            and math.isfinite(seconds) and seconds >= 0, "invalid observed wall time")
    if budget is not None:
        require(seconds <= budget and process.get("timeout_seconds") == budget,
                "SAT did not finish within its original budget")


def classification(result: str) -> dict[str, Any]:
    require(result in RESULTS, f"unknown or unavailable checker outcome: {result}")
    label, establishes_sat, score_credit = RESULTS[result]
    return {"result": result, "classification": label,
            "establishes_sat": establishes_sat, "score_credit": score_credit}


def original_run(start: dict[str, Any], observation: dict[str, Any], arm: str,
                 source_commit: str, image_sha256: str) -> None:
    require(start.get("schema") == "ny_cct_pilot_start_v2" and start.get("arm") == arm,
            "wrong original pilot start/arm")
    fields = start["build_identity"]["fields"]
    require(start["source"]["head"] == fields["commit"] == source_commit
            and fields["source"] == "clean" and fields["dirty"] == "false"
            and start["source"]["status_sha256"] == EMPTY_SHA256
            and start["source"]["tracked_diff_sha256"] == EMPTY_SHA256,
            "original source identity is not the requested clean source")
    require(start["binary"]["sha256"] == fields["exe"] == image_sha256
            and fields["features"] == "mip,cuda" and fields["profile"] == "release",
            "original image/build identity differs")
    require(observation.get("arm") == arm and observation.get("row") == start["row"]["index_zero_based"]
            and observation.get("status") == "completed" and observation.get("protocol_verdict") == "sat"
            and observation.get("drained") is True
            and observation.get("outer_timeout_censored", False) is False,
            "original result was not a completed, uncensored, drained SAT")
    budget = start["row"]["timeout_seconds"]
    require(isinstance(budget, (int, float)) and not isinstance(budget, bool)
            and math.isfinite(budget) and budget > 0, "invalid original timeout")
    completed(observation["process"], budget)


def matching_row(rows: list[tuple[list[str], int]], start: dict[str, Any],
                 observation: dict[str, Any]) -> tuple[list[str], int]:
    original = start["row"]
    # Derive every field from the original execution. There was no prepare
    # phase in this supported pilot. Do not transplant a historical bank row,
    # rewrite model/property paths, or borrow a runtime from another attempt.
    canonical = ["challenging_certified_training_2026", original["model"]["path"],
                 original["property"]["path"], "0", "violated",
                 str(observation["process"]["wall_seconds"])]
    matches = [(row, occurrence) for row, occurrence in rows if row == canonical]
    require(len(matches) == 1, "CSV SAT row differs from the canonical original execution row")
    return matches[0]


def export_records(*, batch_result: Path, batch_sha256: str, results_csv: Path,
                   csv_sha256: str, source_commit: str, image_sha256: str,
                   arm: str, validation_only: bool) -> list[dict[str, Any]]:
    sha(source_commit, 40)
    sha(image_sha256)
    batch_pin, csv_pin = pin(batch_result), pin(results_csv)
    require(batch_pin["sha256"] == sha(batch_sha256), "reviewed batch-result digest differs")
    require(csv_pin["sha256"] == sha(csv_sha256), "reviewed results CSV digest differs")
    batch = load(batch_result)
    require(batch.get("schema") == "ny_cct_actual_sat_replay_batch_result_v2"
            and batch.get("all_completed_sat_discovered") is True
            and batch.get("official_score_credit") is False, "unsupported/incomplete replay batch")
    plan_path = batch_result.parent / "plan.json"
    plan = load(plan_path)
    require(plan.get("schema") == "ny_cct_actual_sat_replay_batch_v2", "wrong batch plan")
    if not validation_only:
        collection = batch["collection"]
        require(arm == "candidate" and source_commit != OLD_F3
                and plan.get("adapter_contract") == collection.get("adapter_contract") == RESEARCH_ADAPTER
                and plan.get("old_replay_passes_transferred") == collection.get("old_replay_passes_transferred") == 0
                and plan["source_commits"][arm] == collection["source_commits"][arm] == source_commit
                and plan["images"][arm]["sha256"] == image_sha256,
                "old/control/other-source batch requires explicit --validation-only")
    rows = read_rows(results_csv)
    require(bool(rows), "no SAT rows to bind")
    items = [item for item in batch["adjudications"] if item.get("arm") == arm]
    require(len(items) == len(rows), "replay batch must exactly cover this CSV's SAT rows")
    records = []
    used_rows: set[tuple[str, int]] = set()
    used_runs: set[str] = set()
    for item in items:
        run_dir = Path(item["run_dir"]).resolve(strict=True)
        require(str(run_dir) not in used_runs, "duplicate original run")
        used_runs.add(str(run_dir))
        completed(item["command"])
        require(item["command"].get("error") is None, "replay wrapper failed")
        require(isinstance(item.get("replay"), dict),
                "missing replay evidence is unclassified, not an explicit no_ce outcome")
        replay_pin = verify_pin(item["replay"])
        replay_path = Path(replay_pin["path"])
        replay = load(replay_path)
        require(replay.get("schema") == "ny_cct_direct_official_sat_replay_v1"
                and replay.get("official_score_credit") is False, "unsupported direct replay schema")
        start_pin, observation_pin = verify_pin(replay["pilot_start"]), verify_pin(replay["pilot_observation"])
        require(Path(start_pin["path"]) == run_dir / "start.json"
                and Path(observation_pin["path"]) == run_dir / "observation.json", "replay points to another run")
        start, observation = load(Path(start_pin["path"])), load(Path(observation_pin["path"]))
        original_run(start, observation, arm, source_commit, image_sha256)
        require(item["row"] == start["row"]["index_zero_based"], "adjudication row differs")
        original = verify_pin(replay["original_result"])
        require(original == verify_pin(item["result"]) and Path(original["path"]) == run_dir / "result.txt",
                "replay result is not the actual original result")
        request_path = replay_path.parent / "request.json"
        request = load(request_path)
        require(request["onnx_path"] == start["row"]["model"]["path"]
                and request["vnnlib_path"] == start["row"]["property"]["path"]
                and request["vnnlib_version"] == "2.0"
                and request["abs_tolerance"] == 1e-4 and request["rel_tolerance"] == 1e-3,
                "checker request differs from original source/property/policy")
        witness = pin(Path(request["witness_path"]))
        raw = Path(original["path"]).read_bytes().splitlines(keepends=True)
        require(raw and raw[0].strip().lower() == b"sat"
                and Path(witness["path"]).read_bytes() == b"".join(raw[1:]),
                "witness is not the byte-identical original assignment")
        completed(load(replay_path.parent / "process.json"))
        response = replay["response"]
        response_path = replay_path.parent / "checker.stdout"
        require(load(response_path) == response, "raw checker response differs from replay receipt")
        require(response.get("ok") is True and response.get("provider") == "CPUExecutionProvider",
                "checker response refused")
        deps = replay["dependencies"]
        dependency_path = replay_path.parent / "dependency-identity.json"
        require(load(dependency_path) == deps, "checker dependency receipt differs")
        require(deps.get("ready") is True and deps.get("checker_commit") == CHECKER_COMMIT,
                "wrong/unready checker source")
        checker_pins = [verify_pin(p) for p in deps["checker_sources"]
                        if Path(p["path"]).name in {"cex_checks.py", "counterexamples.py", "counterexamples_v2.py"}]
        require(len(checker_pins) == 3, "incomplete checker authority source pins")
        result = response["result"]
        grade = classification(result)
        require(item["adjudication"] == result and item["strict_correct"] == grade["establishes_sat"]
                and item["input_tolerance_only"] == (result == "correct_up_to_tolerance")
                and replay["independent_local_strict_sat"] == grade["establishes_sat"],
                "batch/direct checker classifications disagree")
        row, occurrence = matching_row(rows, start, observation)
        identity = row_identity(row)
        require((identity, occurrence) not in used_rows, "two replays claim the same CSV row")
        used_rows.add((identity, occurrence))
        model, prop, image = (verify_pin(start["row"]["model"]),
                              verify_pin(start["row"]["property"]), verify_pin(start["binary"]))
        argv = start["argv"]
        require(argv[:9] == [start["binary"]["path"], "-v", "vnncomp", "v1",
                             row[0].removeprefix("2026_"), model["path"], prop["path"],
                             original["path"], str(start["row"]["timeout_seconds"])],
                "original solver argv differs from CSV/source/budget")
        exec_start = load(run_dir / "exec-start.json")
        require(exec_start["argv"] == argv, "actual exec argv differs")
        versions = response["dependency_versions"]
        require(all(isinstance(versions.get(key), str) and versions[key] for key in ("python", "onnx", "onnxruntime")),
                "missing checker dependency versions")
        records.append({
            "schema_version": 1, "checker_version": "vnncomp2026-cex-1",
            "provider": "CPUExecutionProvider", "dependency_versions": {key: versions[key] for key in ("python", "onnx", "onnxruntime")},
            "benchmark": {"category": "2026_" + row[0].removeprefix("2026_"), "version": "2.0",
                          "network_field": row[1], "property_field": row[2]},
            "results_row_identity": identity, "witness_path": witness["path"],
            "witness_sha256": witness["sha256"], **grade, "rationale": response.get("rationale", ""),
            "ny_evidence": {
                "schema": "ny_vnncomp2026_row_replay_evidence_v1", "tool": "ny",
                "run_id": str(run_dir), "validation_only": validation_only,
                "source_commit": source_commit, "occurrence": occurrence,
                "results_csv": csv_pin, "model": model, "property": prop, "image": image,
                "original_result": original, "witness": witness, "replay": replay_pin,
                "replay_request": pin(request_path), "batch_result": batch_pin,
                "batch_plan": pin(plan_path), "pilot_start": start_pin, "pilot_observation": observation_pin,
                "replay_process": pin(replay_path.parent / "process.json"),
                "replay_stdout": pin(response_path), "replay_dependencies": pin(dependency_path),
                "exec_start": pin(run_dir / "exec-start.json"), "checker_sources": checker_pins,
                "original_timeout_seconds": start["row"]["timeout_seconds"],
                "original_wall_seconds": observation["process"]["wall_seconds"],
                "official_score_credit": False, "hardware_semantics": replay.get("hardware_semantics", ""),
            },
        })
    require(len(used_rows) == len(rows), "not every CSV SAT row has one bound replay")
    require(pin(batch_result) == batch_pin and pin(results_csv) == csv_pin, "export inputs changed")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-result", required=True, type=Path)
    parser.add_argument("--batch-sha256", required=True)
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--csv-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--arm", choices=("candidate", "baseline"), default="candidate")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "output directory already exists")
    records = export_records(batch_result=args.batch_result, batch_sha256=args.batch_sha256,
                             results_csv=args.results_csv, csv_sha256=args.csv_sha256,
                             source_commit=args.source_commit, image_sha256=args.image_sha256,
                             arm=args.arm, validation_only=args.validation_only)
    args.output_dir.mkdir(mode=0o700)
    for record in records:
        name = f"{record['results_row_identity']}-{record['ny_evidence']['occurrence']}.counterexample.check.json"
        with (args.output_dir / name).open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps({"checks": len(records), "validation_only": args.validation_only,
                      "source_commit": args.source_commit, "official_score_credit": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
