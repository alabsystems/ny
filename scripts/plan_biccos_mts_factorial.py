#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Emit a fail-closed, diagnostic-only alpha-beta-CROWN BICCOS plan.

This planner never launches alpha-beta-CROWN. It validates the exact upstream
Git pins, upstream configuration bytes, and sealed NY transfer-corpus assets,
then emits a deterministic command matrix for an operator to review.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "biccos_mts_factorial_v1.json"
DEFAULT_ABC_REPO = Path("<home>/alpha-beta-CROWN-reference")
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "vnncomp2025" / "benchmarks"
SCHEMA = "ny_biccos_mts_factorial_plan_v1"
TARGET_IDS = (
    "cifar100-medium-1761",
    "cifar100-medium-2477",
    "tinyimagenet-medium-1126",
    "tinyimagenet-medium-7943",
)
ARM_IDS = ("baseline", "mts-only", "cs-only", "all")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class PlanError(ValueError):
    """The diagnostic plan cannot be constructed without ambiguity."""


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, declared_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise PlanError(f"required file is missing: {path}")
    return {
        "declared_path": declared_path,
        "resolved_path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _safe_relative_path(raw: object, context: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise PlanError(f"{context} must be a non-empty relative path")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or str(path) != raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PlanError(f"{context} must be a normalized relative path")
    return raw


def _git(repo: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise PlanError(f"git {' '.join(arguments)} failed for {repo}: {detail}")
    return process.stdout.strip()


def _validate_clean_pin(
    abc_repo: Path, expected_commit: str, expected_auto_lirpa: str
) -> dict[str, Any]:
    if not abc_repo.is_dir():
        raise PlanError(f"alpha-beta-CROWN repository is missing: {abc_repo}")
    observed_commit = _git(abc_repo, "rev-parse", "HEAD")
    if observed_commit != expected_commit:
        raise PlanError(
            "alpha-beta-CROWN pin mismatch: "
            f"expected {expected_commit}, observed {observed_commit}"
        )
    status = _git(
        abc_repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise PlanError("alpha-beta-CROWN repository is not clean")

    auto_lirpa = abc_repo / "auto_LiRPA"
    observed_auto_lirpa = _git(auto_lirpa, "rev-parse", "HEAD")
    if observed_auto_lirpa != expected_auto_lirpa:
        raise PlanError(
            "auto_LiRPA pin mismatch: "
            f"expected {expected_auto_lirpa}, observed {observed_auto_lirpa}"
        )
    auto_status = _git(auto_lirpa, "status", "--porcelain", "--untracked-files=all")
    if auto_status:
        raise PlanError("auto_LiRPA repository is not clean")

    gitlink = _git(abc_repo, "ls-tree", "HEAD", "auto_LiRPA").split()
    if len(gitlink) != 4 or gitlink[:2] != ["160000", "commit"]:
        raise PlanError("alpha-beta-CROWN auto_LiRPA gitlink is malformed")
    if gitlink[2] != expected_auto_lirpa:
        raise PlanError(
            "alpha-beta-CROWN gitlink mismatch: "
            f"expected {expected_auto_lirpa}, observed {gitlink[2]}"
        )
    return {
        "abcrown_commit": observed_commit,
        "auto_lirpa_commit": observed_auto_lirpa,
        "auto_lirpa_gitlink": gitlink[2],
        "clean": True,
    }


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        data = path.read_bytes()
        payload = json.loads(data)
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot load planner manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PlanError("planner manifest root must be an object")
    required = {
        "schema",
        "diagnostic_only",
        "execution_allowed",
        "abcrown",
        "corpus_manifest",
        "targets",
        "category_config_directories",
        "arms",
    }
    if set(payload) != required:
        raise PlanError("planner manifest has missing or unknown top-level fields")
    if payload["schema"] != SCHEMA:
        raise PlanError(f"unsupported planner schema: {payload['schema']!r}")
    if payload["diagnostic_only"] is not True:
        raise PlanError("planner manifest must be diagnostic-only")
    if payload["execution_allowed"] is not False:
        raise PlanError("planner manifest must forbid execution")
    if payload["targets"] != list(TARGET_IDS):
        raise PlanError("planner target order or identity changed")
    if not isinstance(payload["arms"], list):
        raise PlanError("planner arms must be a list")
    if [arm.get("id") for arm in payload["arms"] if isinstance(arm, dict)] != list(
        ARM_IDS
    ):
        raise PlanError("planner arm order or identity changed")
    return payload, data


def _load_baseline_module() -> ModuleType:
    path = REPO_ROOT / "scripts" / "abcrown_transfer_baseline.py"
    spec = importlib.util.spec_from_file_location(
        "ny_abcrown_transfer_baseline_for_biccos", path
    )
    if spec is None or spec.loader is None:
        raise PlanError(f"cannot load transfer corpus helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_targets(
    manifest: dict[str, Any],
    *,
    benchmark_root: Path,
    baseline_module: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus_relative = _safe_relative_path(
        manifest["corpus_manifest"], "corpus_manifest"
    )
    corpus_path = REPO_ROOT / corpus_relative
    try:
        corpus, corpus_bytes = baseline_module.load_corpus_manifest(corpus_path)
    except (ValueError, OSError) as error:
        raise PlanError(f"cannot validate transfer corpus: {error}") from error
    entries = {entry["id"]: entry for entry in corpus["entries"]}
    resolved_targets: list[dict[str, Any]] = []
    for target_id in TARGET_IDS:
        entry = entries.get(target_id)
        if entry is None:
            raise PlanError(f"transfer corpus is missing target {target_id}")
        resolved, errors = baseline_module._resolve_vnncomp_entry(
            entry,
            repo_root=REPO_ROOT,
            benchmark_root=benchmark_root,
        )
        if errors:
            raise PlanError("; ".join(errors))
        if resolved["status"] != "ready":
            reasons = ", ".join(resolved["skip_reasons"])
            raise PlanError(f"{target_id} is not sealed and ready: {reasons}")
        files = resolved["files"]
        resolved_targets.append(
            {
                "category": resolved["category"],
                "id": target_id,
                "model": files["model"],
                "property": files["property"],
                "instances_csv": files["instances_csv"],
                "source_index_one_based": resolved["source_index"],
                "timeout_seconds": resolved["timeout_seconds"],
            }
        )
    return (
        resolved_targets,
        {
            "declared_path": corpus_relative,
            "resolved_path": str(corpus_path.resolve()),
            "sha256": _sha256_bytes(corpus_bytes),
            "size_bytes": len(corpus_bytes),
        },
    )


def _nested(document: dict[str, Any], *path: str) -> object:
    value: object = document
    for field in path:
        if not isinstance(value, dict) or field not in value:
            return None
        value = value[field]
    return value


def _validate_arm_semantics(arm: dict[str, Any], document: dict[str, Any]) -> None:
    biccos = _nested(document, "bab", "cut", "biccos")
    biccos_enabled = isinstance(biccos, dict) and biccos.get("enabled") is True
    mts = biccos.get("multi_tree_branching") if isinstance(biccos, dict) else None
    mts_enabled = isinstance(mts, dict) and mts.get("enabled") is True
    restore_best = isinstance(mts, dict) and mts.get("restore_best_tree") is True
    explicit_cs = (
        biccos.get("constraint_strengthening") if isinstance(biccos, dict) else None
    )
    expected_biccos = arm["biccos"]
    expected_mts = arm["multi_tree"]
    expected_cs = arm["constraint_strengthening"]
    if biccos_enabled is not expected_biccos:
        raise PlanError(f"arm {arm['id']} BICCOS semantics do not match its label")
    if mts_enabled is not expected_mts:
        raise PlanError(f"arm {arm['id']} multi-tree semantics do not match its label")
    if expected_mts and not restore_best:
        raise PlanError(f"arm {arm['id']} must restore the best binary tree")
    # Upstream defaults constraint strengthening to true when BICCOS is enabled.
    observed_cs = biccos_enabled and explicit_cs is not False
    if observed_cs is not expected_cs:
        raise PlanError(
            f"arm {arm['id']} constraint-strengthening semantics do not match"
        )


def _resolve_arms(manifest: dict[str, Any], *, abc_repo: Path) -> list[dict[str, Any]]:
    directories = manifest["category_config_directories"]
    if not isinstance(directories, dict) or set(directories) != {
        "cifar100_2024",
        "tinyimagenet_2024",
    }:
        raise PlanError("planner config categories changed")
    arms: list[dict[str, Any]] = []
    for raw_arm in manifest["arms"]:
        if not isinstance(raw_arm, dict):
            raise PlanError("planner arm must be an object")
        required = {
            "id",
            "config",
            "config_sha256",
            "biccos",
            "multi_tree",
            "constraint_strengthening",
        }
        if set(raw_arm) != required:
            raise PlanError(f"arm {raw_arm.get('id')!r} has invalid fields")
        for field in ("biccos", "multi_tree", "constraint_strengthening"):
            if not isinstance(raw_arm[field], bool):
                raise PlanError(f"arm {raw_arm['id']} {field} must be boolean")
        config_name = _safe_relative_path(raw_arm["config"], "arm config")
        expected_hashes = raw_arm["config_sha256"]
        if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
            directories
        ):
            raise PlanError(f"arm {raw_arm['id']} config hashes are incomplete")
        configs: dict[str, Any] = {}
        for category in sorted(directories):
            directory = _safe_relative_path(
                directories[category], f"config directory for {category}"
            )
            relative = f"{directory}/{config_name}"
            identity = _identity(abc_repo / relative, declared_path=relative)
            expected = expected_hashes[category]
            if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
                raise PlanError(f"arm {raw_arm['id']} has an invalid config SHA-256")
            if identity["sha256"] != expected:
                raise PlanError(
                    f"arm {raw_arm['id']} config hash mismatch for {category}: "
                    f"expected {expected}, observed {identity['sha256']}"
                )
            try:
                document = yaml.safe_load(
                    Path(identity["resolved_path"]).read_text(encoding="utf-8")
                )
            except (OSError, yaml.YAMLError) as error:
                raise PlanError(
                    f"cannot parse upstream config {relative}: {error}"
                ) from error
            if not isinstance(document, dict):
                raise PlanError(f"upstream config is not a mapping: {relative}")
            _validate_arm_semantics(raw_arm, document)
            configs[category] = identity
        arms.append(
            {
                "id": raw_arm["id"],
                "biccos": raw_arm["biccos"],
                "multi_tree": raw_arm["multi_tree"],
                "constraint_strengthening": raw_arm["constraint_strengthening"],
                "configs": configs,
            }
        )
    return arms


def build_plan(
    *,
    manifest_path: Path,
    abc_repo: Path,
    benchmark_root: Path,
    python_path: Path | None = None,
    gpu_guard: Path | None = None,
    baseline_module: ModuleType | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    abc_repo = abc_repo.resolve()
    benchmark_root = benchmark_root.resolve()
    manifest, manifest_bytes = _load_manifest(manifest_path)
    abc = manifest["abcrown"]
    if not isinstance(abc, dict) or set(abc) != {
        "commit",
        "auto_lirpa_commit",
        "entrypoint",
    }:
        raise PlanError("planner ABC pin object is malformed")
    for field in ("commit", "auto_lirpa_commit"):
        if not isinstance(abc[field], str) or GIT_SHA_RE.fullmatch(abc[field]) is None:
            raise PlanError(f"abcrown.{field} must be a full lowercase Git SHA")

    git_identity = _validate_clean_pin(
        abc_repo, abc["commit"], abc["auto_lirpa_commit"]
    )
    entrypoint_relative = _safe_relative_path(abc["entrypoint"], "abcrown.entrypoint")
    entrypoint = _identity(
        abc_repo / entrypoint_relative, declared_path=entrypoint_relative
    )
    python_path = (python_path or abc_repo / ".venv" / "bin" / "python").resolve()
    python_identity = _identity(python_path, declared_path=".venv/bin/python")
    if gpu_guard is None:
        discovered = shutil.which("ny-safe-gpu-run")
        if discovered is None:
            raise PlanError("ny-safe-gpu-run is required to emit guarded commands")
        gpu_guard = Path(discovered)
    gpu_guard_identity = _identity(gpu_guard.resolve(), declared_path="ny-safe-gpu-run")

    targets, corpus_identity = _resolve_targets(
        manifest,
        benchmark_root=benchmark_root,
        baseline_module=baseline_module or _load_baseline_module(),
    )
    arms = _resolve_arms(manifest, abc_repo=abc_repo)
    runs: list[dict[str, Any]] = []
    arms_by_id = {arm["id"]: arm for arm in arms}
    for target in targets:
        for arm_id in ARM_IDS:
            arm = arms_by_id[arm_id]
            config = arm["configs"][target["category"]]
            argv = [
                gpu_guard_identity["resolved_path"],
                python_identity["resolved_path"],
                entrypoint["resolved_path"],
                "--config",
                config["resolved_path"],
                "--onnx_path",
                target["model"]["resolved_path"],
                "--vnnlib_path",
                target["property"]["resolved_path"],
                "--timeout",
                str(target["timeout_seconds"]),
            ]
            runs.append(
                {
                    "arm_id": arm_id,
                    "argv": argv,
                    "config_sha256": config["sha256"],
                    "run_id": f"{target['id']}::{arm_id}",
                    "target_id": target["id"],
                }
            )
    return {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "execution_allowed": False,
        "operator_action": "review_and_execute_commands_manually",
        "manifest": {
            "resolved_path": str(manifest_path),
            "sha256": _sha256_bytes(manifest_bytes),
            "size_bytes": len(manifest_bytes),
        },
        "abcrown": {
            **git_identity,
            "repository": str(abc_repo),
            "entrypoint": entrypoint,
            "python": python_identity,
            "gpu_guard": gpu_guard_identity,
        },
        "corpus": corpus_identity,
        "benchmark_root": str(benchmark_root),
        "targets": targets,
        "arms": arms,
        "runs": runs,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="emit a sealed diagnostic-only ABC BICCOS factorial plan"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--abc-repo", type=Path, default=DEFAULT_ABC_REPO)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--gpu-guard", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        plan = build_plan(
            manifest_path=args.manifest,
            abc_repo=args.abc_repo,
            benchmark_root=args.benchmark_root,
            python_path=args.python,
            gpu_guard=args.gpu_guard,
        )
        rendered = _canonical_json(plan)
        if args.output is None:
            sys.stdout.buffer.write(rendered)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(rendered)
    except (OSError, PlanError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
