"""Independent replay adapters; orchestration never creates a proof token."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

from .evidence import (ExperimentError, bounded_artifact, count, immutable_json,
                       pin, strict_json, unchanged)
from .process import run_command


def expand_argv(template: list[str], variables: dict[str, str]) -> list[str]:
    if not isinstance(template, list) or not template or any(not isinstance(t, str) for t in template):
        raise ExperimentError("argv must be a nonempty list of strings")
    result = []
    for token in template:
        names = re.findall(r"\{([^{}]*)\}", token)
        if any(name not in variables for name in names) or token.count("{") != len(names) or token.count("}") != len(names):
            raise ExperimentError(f"unsupported argv placeholder: {token}")
        for name in names:
            token = token.replace("{" + name + "}", variables[name])
        if "\0" in token:
            raise ExperimentError("NUL in argv")
        result.append(token)
    return result


def bundle_shape(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Inspect the existing hint schema; shape is explicitly not arithmetic replay."""
    import json
    leaves = 0
    last = None
    try:
        with path.open("rb") as stream:
            header = json.loads(stream.readline(1024 * 1024))
            for line in stream:
                if len(line) > 1024 * 1024:
                    raise ExperimentError("proof bundle record exceeds 1 MiB")
                last = json.loads(line)
                if not isinstance(last, dict):
                    raise ExperimentError("malformed proof bundle record")
                leaves += last.get("kind") == "leaf"
    except (ValueError, UnicodeError) as error:
        raise ExperimentError("malformed proof bundle") from error
    if not isinstance(header, dict):
        raise ExperimentError("empty/malformed proof bundle")
    if header.get("kind") != "header" or header.get("schema") != "ny-proof-bundle/v1":
        raise ExperimentError("unsupported proof bundle schema")
    for src, target in (("onnx", "model"), ("vnnlib", "property")):
        if not isinstance(header.get(src), dict) or header[src].get("sha256") != row[target]["sha256"]:
            raise ExperimentError("proof bundle input binding mismatch")
    if last != {"kind": "published", "result": "unsat"}:
        raise ExperimentError("proof bundle was not finalized for UNSAT")
    return {"schema": "ny-proof-bundle/v1", "leaves": leaves,
            "status": "shape_only", "arithmetic_checked": False,
            "reason": "no_builtin_complete_unsat_arithmetic_replayer"}


def official_metadata(run_dir: Path, run_id: str, row: dict[str, Any],
                      start: dict[str, Any], result: dict[str, Any], seconds: float) -> Path:
    payload = {
        "schema": "ny_measurement_result_v2", "schema_version": 2,
        "run_id": run_id, "category": row["category"],
        # The archived measurement schema's index is zero based. Preserve both.
        "instance_index": row["source_index"] - 1,
        "official_source_index": row["source_index"], "occurrence": row["occurrence"],
        "solver_verdict": "sat", "witness_present": True,
        "counterexample_validation": {"status": "not_checked", "checker": None},
        "result_artifact": "result", "raw_result_sha256": result["sha256"],
        "result_sha256": result["sha256"], "start_manifest": "start.json",
        "start_manifest_sha256": start["sha256"],
        "timeout_seconds": row["timeout_seconds"], "elapsed_seconds": seconds,
        "source_csv": row["source_csv_sha256"],
    }
    for key, source in (("onnx", "model"), ("vnnlib", "property")):
        payload[key] = {"declared_path": row["declared_" + source],
                        "resolved_path": row[source]["path"],
                        "sha256": row[source]["sha256"], "size_bytes": row[source]["size_bytes"]}
    path = run_dir / "sat-metadata.json"
    immutable_json(path, payload)
    return path


def validate_command_receipt(raw: dict[str, Any], request: dict[str, Any],
                             name: str, request_sha256: str | None = None) -> dict[str, Any]:
    if raw.get("schema") != "ny_experiment_replay_v1" or raw.get("replayer") != name:
        raise ExperimentError("unknown replay receipt schema or replayer")
    if raw.get("bindings") != request["bindings"] or raw.get("verdict") != request["verdict"]:
        raise ExperimentError("independent replay did not bind exact original inputs/artifacts/run")
    if request_sha256 is not None and raw.get("request_sha256") != request_sha256:
        raise ExperimentError("independent replay did not bind its immutable request bytes")
    if raw.get("status") != "checked" or raw.get("basis") != "original_model_property":
        raise ExperimentError("void/unsupported/refused replay is not checked evidence")
    coverage = raw.get("coverage")
    if not isinstance(coverage, dict):
        raise ExperimentError("replay requires positive coverage")
    checked = count(coverage.get("checked_obligations"), "checked_obligations")
    remaining = count(coverage.get("remaining_obligations"), "remaining_obligations")
    if not checked or remaining or coverage.get("complete") is not True:
        raise ExperimentError("zero or incomplete replay coverage")
    if raw["verdict"] == "sat":
        witness = raw.get("witness")
        if (not isinstance(witness, dict) or witness.get("input_checked") is not True
                or witness.get("output_checked") is not True or witness.get("strict_violation") is not True
                or count(witness.get("checked_coordinates"), "checked_coordinates") == 0):
            raise ExperimentError("SAT replay requires a positive strict original witness receipt")
    elif raw["verdict"] == "unsat":
        arithmetic = raw.get("arithmetic")
        if (not isinstance(arithmetic, dict) or arithmetic.get("checked") is not True
                or arithmetic.get("method") not in {"directed_interval", "exact_rational"}
                or count(arithmetic.get("checked_operations"), "checked_operations") == 0):
            raise ExperimentError("UNSAT requires independently replayed arithmetic and complete coverage")
    else:
        raise ExperimentError("only decided claims admit replay")
    # This validates an attestation's contract, not the replayer's mathematics.
    # It deliberately produces no trusted certificate or official score credit.
    return {"status": "external_replay_attestation", "replayer": name,
            "coverage": coverage, "authority": "experimental_external_checker",
            "official_score_credit": False, "trusted_certificate": False}


def replay_claim(*, config: dict[str, Any] | None, run_dir: Path, run_id: str,
                 row: dict[str, Any], start: dict[str, Any], solver: dict[str, Any],
                 verdict: str, artifacts: dict[str, dict[str, Any]], repo: Path,
                 environment: dict[str, str], box: Path) -> dict[str, Any]:
    if not config:
        return {"status": "unsupported", "reason": "no_independent_replay_adapter",
                "trusted_certificate": False, "official_score_credit": False}
    adapter = config["adapter"]
    timeout = config["timeout_seconds"]
    bindings = {"run_id": run_id, "start_sha256": start["sha256"],
                "solver_binary_sha256": solver["sha256"],
                "model_sha256": row["model"]["sha256"],
                "property_sha256": row["property"]["sha256"],
                "source_csv_sha256": row["source_csv_sha256"],
                "source_index": row["source_index"], "occurrence": row["occurrence"],
                "replayer_identity_sha256": config["identity_sha256"],
                "artifacts": {key: value["sha256"] for key, value in artifacts.items()}}
    request = {"schema": "ny_experiment_replay_request_v1", "verdict": verdict,
               "bindings": bindings, "model": row["model"]["path"],
               "property": row["property"]["path"],
               "artifacts": {key: value["path"] for key, value in artifacts.items()},
               "authority": "original_model_property; solver telemetry is not evidence"}
    request_path = run_dir / "replay-request.json"
    request_pin = immutable_json(request_path, request)
    response_path = run_dir / "replay-response.json"
    if response_path.exists() or response_path.is_symlink():
        raise ExperimentError("replay response existed before independent invocation")
    shape = None
    if "proof_bundle" in artifacts:
        shape = bundle_shape(Path(artifacts["proof_bundle"]["path"]), row)
    if adapter == "official_sat":
        if verdict != "sat":
            return {"status": "unsupported", "reason": "official_sat_adapter_has_no_unsat_replay",
                    "bundle": shape, "trusted_certificate": False, "official_score_credit": False}
        metadata = official_metadata(run_dir, run_id, row, start, artifacts["result"],
                                     config["solver_wall_seconds"])
        argv = [str(Path(sys.executable).resolve()), str(repo / "scripts/replay_ny_counterexamples.py"),
                "--metadata", str(metadata), "--artifact-root", str(run_dir),
                "--checker-repo", config["checker_repo"], "--checker-python", config["checker_python"],
                "--checker-venv", config["checker_venv"], "--vnnlib-source", config["vnnlib_source"],
                "--vnnlib-version", config.get("vnnlib_version", "auto"),
                "--sidecar", str(response_path), "--checker-timeout", str(int(timeout))]
    elif adapter == "independent_command":
        argv = expand_argv(config["argv"], {"replayer": config["binary"],
                          "request": str(request_path), "response": str(response_path),
                          "run_dir": str(run_dir)})
        if argv[0] != config["binary"]:
            raise ExperimentError("independent replay must execute its pinned binary")
    else:
        raise ExperimentError(f"unsupported replay adapter: {adapter}")
    # Replay is an independently costed correctness probe. It cannot compete
    # concurrently with another experiment's measurement on this machine.
    command = [str(box), "build", "--", *argv]
    process = run_command(command, cwd=repo, env=environment,
                          stdout=run_dir / "replay.stdout", stderr=run_dir / "replay.stderr",
                          timeout_seconds=timeout)
    if process["timed_out"] or process["interrupted"] or process["returncode"] != 0 or process["wall_seconds"] > timeout:
        return {"status": "void", "reason": "replay_process_not_completed", "process": process,
                "official_score_credit": False, "trusted_certificate": False}
    if not response_path.exists():
        raise ExperimentError("replayer completed without a receipt")
    raw = strict_json(bounded_artifact(run_dir, "replay-response.json"))
    unchanged(request_pin)
    for value in artifacts.values():
        unchanged(value)
    if adapter == "official_sat":
        evidence = raw.get("evidence", {})
        if (raw.get("schema") != "ny_counterexample_validation_v1"
                or raw.get("official_result") != "correct" or raw.get("establishes_strict_sat") is not True
                or evidence.get("raw_result", {}).get("sha256") != artifacts["result"]["sha256"]
                or evidence.get("onnx", {}).get("sha256") != row["model"]["sha256"]
                or evidence.get("vnnlib", {}).get("sha256") != row["property"]["sha256"]
                or evidence.get("start_manifest", {}).get("sha256") != start["sha256"]
                or raw.get("measurement", {}).get("run_id") != run_id
                or evidence.get("extracted_assignment", {}).get("size_bytes", 0) <= 0):
            raise ExperimentError("official replay did not establish a strict bound witness")
        result = {"status": "strict_sat_replayed", "replayer": "pinned_vnncomp_counterexample_checker",
                  "coverage": {"checked_witnesses": 1, "input_and_output": True},
                  "official_score_credit": False, "trusted_certificate": False,
                  "note": "checker evidence only; official scoring/referee policy is separate"}
    else:
        result = validate_command_receipt(raw, request, config["name"], request_pin["sha256"])
    result.update({"artifact": pin(response_path), "process": process, "bundle": shape,
                   "request": request_pin})
    return result
