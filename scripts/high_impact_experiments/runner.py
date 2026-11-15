"""One frozen cohort, four explicit arms, immutable receipts, independent replay."""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from . import SCHEMA
from .cohort import load_cohort
from .evidence import (ExperimentError, archive_directory, count, digest, encode,
                       git_state, immutable, immutable_json, pin, pin_sources,
                       ny_build_source, positive, seal_copy, stable_read, strict_json, unchanged)
from .process import run_command
from .receipts import native_ny_observations, proposed_verdict, telemetry
from .replay import expand_argv, replay_claim

ROLES = {"baseline", "mechanism", "comparator", "combined"}
FORBIDDEN_ENV = {"LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "PYTHONPATH", "PYTHONSTARTUP"}


def environment(values: Any) -> dict[str, str]:
    if not isinstance(values, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                            or "\0" in k + v or "=" in k for k, v in values.items()):
        raise ExperimentError("environment must be an explicit string map")
    if any(k.startswith("NY_BOX_") or k == "NY_SKIP_CREDENTIAL_CHECK" or k in FORBIDDEN_ENV for k in values):
        raise ExperimentError("environment may not bypass the box or inject executable code")
    # Never inherit secret credentials or undeclared NY route settings.
    return {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", **values}


def resolve(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExperimentError("nonempty path required")
    return (base / value).resolve(strict=True)


def validate_plan(raw: dict[str, Any], base: Path, repo: Path) -> dict[str, Any]:
    if raw.get("schema") != "ny_high_impact_plan_v1":
        raise ExperimentError("unsupported experiment plan schema")
    arms = raw.get("arms")
    if not isinstance(arms, list) or len(arms) != 4 or {a.get("role") for a in arms if isinstance(a, dict)} != ROLES:
        raise ExperimentError("plan must contain exactly baseline, mechanism, comparator, combined arms")
    names = set()
    for arm in arms:
        name = arm.get("name")
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) or name in names:
            raise ExperimentError("arm names must be unique safe path components")
        names.add(name)
        arm["binary"] = str(resolve(base, arm["binary"]))
        if not os.access(arm["binary"], os.X_OK):
            raise ExperimentError(f"arm binary is not executable: {arm['binary']}")
        if not isinstance(arm.get("argv"), list) or not arm["argv"] or arm["argv"][0] != "{binary}":
            raise ExperimentError("arm argv must execute the explicitly pinned {binary}")
        arm["environment"] = environment(arm.get("environment", {}))
        if arm.get("program") not in {"ny", "external"}:
            raise ExperimentError("arm.program must name ny or external")
        if arm["program"] == "ny":
            if not isinstance(arm.get("build_features"), list) or any(not isinstance(v, str) for v in arm["build_features"]):
                raise ExperimentError("ny arms must declare build_features for binary cross-check")
            # Compile inputs are automatic, in addition to declared mechanism
            # sources and profile/runtime dependency roots.
            arm["source_paths"] = [*arm.get("source_paths", []),
                *[str(repo / p) for p in ("crates", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]]
            if arm.get("profile"):
                if "{profile}" not in arm["argv"]:
                    raise ExperimentError("NY profile must arrive through an explicit {profile} argv token")
            else:
                arm["configs_root"] = str(resolve(base, arm.get("configs_root", str(repo / "configs"))))
                if "{configs_root}" not in arm["argv"] or "--configs-dir" not in arm["argv"]:
                    raise ExperimentError("NY scored arms must explicitly bind --configs-dir {configs_root}")
                arm["source_paths"].append(arm["configs_root"])
        elif not arm.get("build_receipt"):
            raise ExperimentError("external arms require a pinned build/version provenance receipt")
        for key in ("profile", "build_receipt"):
            if arm.get(key):
                arm[key] = str(resolve(base, arm[key]))
        replay = arm.get("replay")
        if replay:
            if not isinstance(replay, dict) or replay.get("adapter") not in {"official_sat", "independent_command"}:
                raise ExperimentError("unsupported replay adapter")
            replay["timeout_seconds"] = positive(replay.get("timeout_seconds"), "replay timeout")
            if replay["adapter"] == "official_sat":
                for key in ("checker_repo", "checker_python", "checker_venv", "vnnlib_source"):
                    replay[key] = str(resolve(base, replay[key]))
                if replay["timeout_seconds"] < 1:
                    raise ExperimentError("official checker timeout must be at least one second")
            else:
                if not isinstance(replay.get("name"), str) or not replay["name"]:
                    raise ExperimentError("independent replay requires a name")
                replay["binary"] = str(resolve(base, replay["binary"]))
                if pin(Path(replay["binary"]))["sha256"] == pin(Path(arm["binary"]))["sha256"]:
                    raise ExperimentError("independent replayer cannot be the solver binary")
                if not replay.get("source_paths"):
                    raise ExperimentError("independent replay requires pinned source_paths")
                replay["source_paths"] = [str(resolve(base, p)) for p in replay["source_paths"]]
        if not arm.get("source_paths"):
            raise ExperimentError("each arm requires explicit source_paths")
        arm["source_paths"] = [str(resolve(base, p)) for p in arm["source_paths"]]
    conditions = raw.get("conditions", ["process_cold"])
    if not isinstance(conditions, list) or not conditions or len(set(conditions)) != len(conditions) or set(conditions) - {"process_cold", "artifact_cache_warm"}:
        raise ExperimentError("only process_cold and artifact_cache_warm are implemented; resident-model warm is unsupported")
    raw["conditions"] = conditions
    raw["repetitions"] = count(raw.get("repetitions", 1), "repetitions")
    if not 1 <= raw["repetitions"] <= 100:
        raise ExperimentError("repetitions must be between 1 and 100")
    raw["sample_seconds"] = positive(raw.get("sample_seconds", 0.1), "sample_seconds")
    if raw["sample_seconds"] < 0.02:
        raise ExperimentError("sample_seconds must be at least 0.02")
    raw["box_wait_allowance_seconds"] = positive(raw.get("box_wait_allowance_seconds", 6300), "box wait allowance")
    return raw


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExperimentError(f"cannot load provenance adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def freeze(plan_path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = strict_json(plan_path)
    plan = validate_plan(raw, plan_path.resolve().parent, repo)
    rows, cohort = load_cohort(plan["cohort"], plan_path.resolve().parent)
    module_dir = Path(__file__).resolve().parent
    toolkit = [*module_dir.glob("*.py"), module_dir.parent / "high_impact_experiments.py",
               repo / "scripts/ny_box.sh", repo / "scripts/replay_ny_counterexamples.py",
               repo / "scripts/ny_measurement_provenance.py", repo / "scripts/_portable_file_lock.py"]
    evidence: dict[str, Any] = {"schema": SCHEMA, "plan": pin(plan_path),
                                "source_tree": git_state(repo), "source_root": str(repo),
                                "cohort": cohort, "rows": rows, "arms": [],
                                "toolkit": pin_sources(toolkit), "python": pin(Path(sys.executable)),
                                "runtime": {"python": sys.version, "platform": platform.platform()}}
    for arm in plan["arms"]:
        rec = {"name": arm["name"], "role": arm["role"], "binary": pin(Path(arm["binary"])),
               "sources": pin_sources([Path(p) for p in arm["source_paths"]]),
               "profile": pin(Path(arm["profile"])) if arm.get("profile") else None,
               "build_receipt": pin(Path(arm["build_receipt"])) if arm.get("build_receipt") else None,
               "environment": arm["environment"], "argv_template": arm["argv"], "replay": None}
        replay = arm.get("replay")
        if replay:
            if replay["adapter"] == "official_sat":
                adapter = _load_module(repo / "scripts/replay_ny_counterexamples.py", "_e2_official_replay")
                checker = adapter._checker_identity(Path(replay["checker_repo"]))
                vnnlib = adapter._vnnlib_source_identity(Path(replay["vnnlib_source"]))
                rec["replay"] = {"adapter": "official_sat", "checker": checker, "vnnlib": vnnlib,
                                 "python": pin(Path(replay["checker_python"])),
                                 "source_paths": [pin(Path(replay["checker_repo"]) / p)
                                                  for p in checker["source_sha256"]]}
            else:
                rec["replay"] = {"adapter": "independent_command", "name": replay["name"],
                                 "binary": pin(Path(replay["binary"])),
                                 "source_paths": pin_sources([Path(p) for p in replay["source_paths"]])}
        evidence["arms"].append(rec)
    evidence["identity_sha256"] = digest(encode(evidence))
    return plan, evidence


def all_pins(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "sha256", "fingerprint", "size_bytes"} <= value.keys():
            return [value]
        return [p for child in value.values() for p in all_pins(child)]
    if isinstance(value, list):
        return [p for child in value for p in all_pins(child)]
    return []


def check_frozen(evidence: dict[str, Any], plan: dict[str, Any], repo: Path) -> None:
    for record in all_pins(evidence):
        unchanged(record)
    if git_state(repo) != evidence["source_tree"]:
        raise ExperimentError("source tree changed after freeze")
    # Detect added/removed source files, not only mutation of known members.
    for arm, record in zip(plan["arms"], evidence["arms"]):
        if pin_sources([Path(p) for p in arm["source_paths"]]) != record["sources"]:
            raise ExperimentError("arm source inventory changed after freeze")
        replay = arm.get("replay")
        if replay and replay["adapter"] == "official_sat":
            adapter = _load_module(repo / "scripts/replay_ny_counterexamples.py", "_e2_official_replay")
            if (adapter._checker_identity(Path(replay["checker_repo"])) != record["replay"]["checker"]
                    or adapter._vnnlib_source_identity(Path(replay["vnnlib_source"])) != record["replay"]["vnnlib"]):
                raise ExperimentError("official replay source changed after freeze")
        elif replay:
            if pin_sources([Path(p) for p in replay["source_paths"]]) != record["replay"]["source_paths"]:
                raise ExperimentError("independent replay source inventory changed")


def capture_builds(plan: dict[str, Any], evidence: dict[str, Any], archive: Path, repo: Path) -> None:
    provenance = None
    for arm, record in zip(plan["arms"], evidence["arms"]):
        if arm["program"] != "ny":
            receipt = strict_json(Path(arm["build_receipt"]))
            if (receipt.get("schema") != "ny_external_program_identity_v1"
                    or receipt.get("binary_sha256") != record["binary"]["sha256"]
                    or not isinstance(receipt.get("version"), str) or not receipt["version"]
                    or receipt.get("source_files_sha256") != digest(encode(record["sources"]))):
                raise ExperimentError("external build/version receipt does not bind pinned binary/source bytes")
            record["build_identity"] = receipt
            continue
        if provenance is None:
            provenance = _load_module(repo / "scripts/ny_measurement_provenance.py", "_e2_provenance")
        out, err = archive / f"{arm['name']}.build-id.stdout", archive / f"{arm['name']}.build-id.stderr"
        receipt = run_command([str(repo / "scripts/ny_box.sh"), "build", "--", arm["binary"], "--build-id"],
                              cwd=repo, env=arm["environment"], stdout=out, stderr=err,
                              timeout_seconds=plan["box_wait_allowance_seconds"] + 180)
        if receipt["timed_out"] or receipt["interrupted"] or receipt["returncode"] != 0:
            raise ExperimentError("NY build identity correctness probe refused or timed out")
        lines = out.read_text().strip().splitlines()
        if not lines:
            raise ExperimentError("NY binary returned no build identity")
        line = lines[0]
        pieces = line.split()
        fields: dict[str, str] = {}
        for item in pieces[1:]:
            key, sep, value = item.partition("=")
            if not sep or key in fields:
                raise ExperimentError("malformed NY build identity")
            fields[key] = value
        identity = {"schema": pieces[0], "line": line, "fields": fields}
        checked = provenance.validate_binary_build_id(identity,
                    expected_executable_sha256=record["binary"]["sha256"],
                    declared_features=arm["build_features"])
        if checked["commit"] != evidence["source_tree"]["head"]:
            raise ExperimentError("NY binary commit does not match the frozen source tree")
        expected_source = ny_build_source(repo)
        if any(checked[key] != expected for key, expected in expected_source.items()):
            raise ExperimentError("NY binary tracked source stamp does not match the frozen build inputs")
        record["build_identity"] = identity
        record["build_coherence"] = provenance._capture_build_coherence(repo, Path(arm["binary"]))
        record["build_identity_process"] = receipt
        record["build_identity_stdout"] = pin(out)
        record["build_identity_stderr"] = pin(err)


def seal_solver_artifacts(run_dir: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    result: dict[str, dict[str, Any]] = {}
    mutable: list[dict[str, Any]] = []
    for source, target, key in (("solver-result", "result", "result"),
                                ("solver-result.proof.jsonl", "proof-bundle.jsonl", "proof_bundle"),
                                ("solver-phases", "phases.json", "phases")):
        path = run_dir / source
        if path.exists():
            if path.is_symlink():
                raise ExperimentError("solver artifact may not be a symlink")
            result[key], original = seal_copy(path, run_dir / target)
            mutable.append(original)
    return result, mutable


def run_arm(*, row: dict[str, Any], arm: dict[str, Any], arm_pin: dict[str, Any],
            plan: dict[str, Any], frozen: dict[str, Any], archive: Path, repo: Path,
            condition: str, repetition: int, cache: Path, mode: str) -> dict[str, Any]:
    run_id = f"r{row['source_index']:06d}-{arm['name']}-{condition}-{repetition:03d}"
    run_dir = archive / run_id
    run_dir.mkdir(mode=0o700)
    result: dict[str, Any] = {"schema": "ny_experiment_row_v1", "run_id": run_id,
        "role": arm["role"], "arm": arm["name"], "row": row,
        "condition": condition, "repetition": repetition,
        "warmup": condition == "artifact_cache_warm" and repetition == 0,
        "measurement_lane": "ny_box_measure" if mode == "measure" else "ny_box_build_correctness_only",
        "official_score_credit": False, "trusted_certificate": False,
        "frozen_identity_sha256": frozen["identity_sha256"]}
    if "unsupported" in row:
        result.update({"status": "refused", "reason": row["unsupported"], "solver_claim": "not_run"})
        immutable_json(run_dir / "receipt.json", result)
        return result
    started = time.monotonic()
    artifacts: dict[str, dict[str, Any]] = {}
    originals: list[dict[str, Any]] = []
    try:
        check_frozen(frozen, plan, repo)
        variables = {"binary": arm["binary"], "model": row["model"]["path"],
                     "property": row["property"]["path"], "timeout": str(row["timeout_seconds"]),
                     "result": str(run_dir / "solver-result"), "phases": str(run_dir / "solver-phases"),
                     "run_dir": str(run_dir), "cache_dir": str(cache),
                     "profile": arm.get("profile", ""), "instance_index": str(row["source_index"])}
        variables.update({"category": row["category"], "suite": row["suite"],
                          "configs_root": arm.get("configs_root", "")})
        argv = expand_argv(arm["argv"], variables)
        start = {"schema": "ny_measurement_start_v1", "run_id": run_id,
                 "experiment": result, "frozen_identity_sha256": frozen["identity_sha256"],
                 "argv": argv, "environment": arm["environment"], "binary": arm_pin["binary"],
                 "profile": arm_pin["profile"], "source_tree": frozen["source_tree"]}
        start_pin = immutable_json(run_dir / "start.json", start)
        request = {"argv": argv, "cwd": str(repo), "environment": arm["environment"],
                   "run_dir": str(run_dir), "cache_dir": str(cache),
                   "timeout_seconds": row["timeout_seconds"], "sample_seconds": plan["sample_seconds"]}
        request_path = run_dir / "worker-request.json"
        request_pin = immutable_json(request_path, request)
        command = [str(repo / "scripts/ny_box.sh"), "measure" if mode == "measure" else "build", "--",
                   str(Path(sys.executable).resolve()), str(repo / "scripts/high_impact_experiments.py"),
                   "__worker", str(request_path)]
        wrapper = run_command(command, cwd=repo, env=environment({}),
                              stdout=run_dir / "box.stdout", stderr=run_dir / "box.stderr",
                              timeout_seconds=row["timeout_seconds"] + plan["box_wait_allowance_seconds"])
        result["box_process"] = wrapper
        unchanged(request_pin)
        if not (run_dir / "process.json").exists():
            result.update({"status": "refused", "reason": "box_did_not_complete_worker",
                           "solver_claim": "not_observed"})
        else:
            process = strict_json(run_dir / "process.json")
            if process.get("schema") != "ny_experiment_process_v1" or process.get("argv") != argv:
                raise ExperimentError("worker process receipt does not bind exact argv")
            result["query_process"] = process
            result["box_wait_and_orchestration_seconds"] = max(0.0, wrapper["wall_seconds"] - process["wall_seconds"])
            artifacts, originals = seal_solver_artifacts(run_dir)
            claim = proposed_verdict(run_dir / "result")
            result["solver_claim"] = claim
            result["artifacts"] = artifacts
            result["phase_receipts"] = telemetry(run_dir / "phases.json", process["wall_seconds"])
            if arm["program"] == "ny":
                result["native_method_observations"] = native_ny_observations(run_dir / "solver.stderr")
            if (wrapper["timed_out"] or wrapper["interrupted"] or process["timed_out"]
                    or process["interrupted"] or process["wall_seconds"] > row["timeout_seconds"]
                    or process["returncode"] != 0 or wrapper["returncode"] != 0):
                result.update({"status": "void", "reason": "whole_query_deadline_or_execution_not_completed"})
            elif claim in {"sat", "unsat"}:
                replay = dict(arm.get("replay") or {}) or None
                if replay:
                    replay["solver_wall_seconds"] = process["wall_seconds"]
                    replay["identity_sha256"] = digest(encode(arm_pin["replay"]))
                result["replay"] = replay_claim(config=replay, run_dir=run_dir, run_id=run_id,
                    row=row, start=start_pin, solver=arm_pin["binary"], verdict=claim,
                    artifacts=artifacts, repo=repo, environment=environment({}),
                    box=repo / "scripts/ny_box.sh")
                replay_status = result["replay"]["status"]
                result["status"] = "replay_observed" if replay_status in {"strict_sat_replayed", "external_replay_attestation"} else "unqualified"
                if result["status"] == "unqualified":
                    result["reason"] = "decided_claim_has_no_completed_independent_replay"
            elif claim in {"unknown", "timeout"}:
                result["status"] = "open"
            else:
                result.update({"status": "void", "reason": "missing_malformed_or_error_solver_result"})
        for record in [*artifacts.values(), *originals]:
            unchanged(record)
        check_frozen(frozen, plan, repo)
        result["post_run_hash_check"] = "stable"
    except (ExperimentError, OSError, ValueError, KeyError, RuntimeError) as error:
        result.update({"status": "void", "reason": str(error), "post_run_hash_check": "not_established"})
    result["whole_arm_seconds_including_replay_and_hashing"] = time.monotonic() - started
    immutable_json(run_dir / "receipt.json", result)
    return result


def execute(plan_path: Path, archive_path: Path, *, repo: Path, mode: str) -> dict[str, Any]:
    if mode not in {"measure", "correctness"}:
        raise ExperimentError("mode must be measure or correctness")
    plan, frozen = freeze(plan_path, repo)
    archive = archive_directory(archive_path, repo)
    # Freeze before any execution. This preflight record remains if provenance
    # refuses a stale binary, making the refusal reviewable instead of a zero.
    immutable_json(archive / "preflight.json", frozen)
    try:
        capture_builds(plan, frozen, archive, repo)
        check_frozen(frozen, plan, repo)
    except (ExperimentError, OSError, ValueError, RuntimeError) as error:
        summary = {"schema": SCHEMA, "status": "refused", "reason": str(error),
                   "official_score_credit": False, "runs": []}
        immutable_json(archive / "summary.json", summary)
        return summary
    frozen["identity_sha256"] = digest(encode({k: v for k, v in frozen.items() if k != "identity_sha256"}))
    immutable_json(archive / "frozen.json", frozen)
    runs = []
    for row in frozen["rows"]:
        for condition in plan["conditions"]:
            for repetition in range(plan["repetitions"] + (condition == "artifact_cache_warm")):
                for arm, arm_pin in zip(plan["arms"], frozen["arms"]):
                    cache_name = f"cache-r{row['source_index']:06d}-{arm['name']}-{condition}"
                    if condition == "process_cold":
                        cache_name += f"-{repetition:03d}"
                    cache = archive / cache_name
                    if not cache.exists():
                        cache.mkdir(mode=0o700)
                    runs.append(run_arm(row=row, arm=arm, arm_pin=arm_pin, plan=plan, frozen=frozen,
                                        archive=archive, repo=repo, condition=condition,
                                        repetition=repetition, cache=cache, mode=mode))
    # A final source/input change invalidates earlier runs too, not just the
    # last observation. Per-run immutable receipts are retained; the summary
    # is the authoritative orchestration disposition for the entire archive.
    try:
        check_frozen(frozen, plan, repo)
        stable = True
        reason = None
    except (ExperimentError, OSError, ValueError, RuntimeError) as error:
        stable, reason = False, str(error)
    counts: dict[str, int] = {}
    for run in runs:
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    summary = {"schema": SCHEMA, "status": "completed" if stable else "void",
               "reason": reason, "post_run_hash_check": "stable" if stable else "changed",
               "mode": mode, "cohort_kind": plan["cohort"]["kind"], "counts": counts,
               "official_score_credit": False, "trusted_certificate": False,
               "frozen_manifest": pin(archive / "frozen.json"),
               "runs": [{"run_id": run["run_id"], "status": run["status"],
                         "receipt": pin(archive / run["run_id"] / "receipt.json")} for run in runs],
               "limitations": ["OS page cache coldness is not established",
                               "resident-model warm protocol is not implemented",
                               "VRAM has no bound sampler; disk counters are OS blocks",
                               "external replayer attestations are not trusted certificates",
                               "proof bundle shape alone cannot qualify UNSAT",
                               "dynamic runtime dependencies require explicit source/profile pins"]}
    immutable_json(archive / "summary.json", summary)
    return summary
