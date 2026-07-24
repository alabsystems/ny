#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Run the staged alpha-beta-CROWN recipe-transfer factorials.

The committed experiment manifest names a base NY preset and a set of
single-field or bundled treatments.  This driver materializes every treatment
as a standalone YAML file, hashes the exact input and generated presets, and
delegates bounded execution to ``benchmark_vnncomp_preset_bounded.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = (
    REPO_ROOT / "configs" / "experiments" / "abcrown_transfer_factorials.yaml"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "abcrown_transfer_factorials"
BOUNDED_RUNNER = REPO_ROOT / "scripts" / "benchmark_vnncomp_preset_bounded.py"


class ManifestError(ValueError):
    """The factorial manifest is malformed or refers to an invalid target."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    if not cleaned or cleaned in {".", ".."}:
        raise ManifestError(f"invalid artifact name: {value!r}")
    return cleaned


def _deep_set(document: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not parts or any(not part for part in parts):
        raise ManifestError(f"invalid override path: {dotted_key!r}")
    cursor: dict[str, Any] = document
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ManifestError(
                f"override {dotted_key!r} crosses non-mapping field {part!r}"
            )
        cursor = child
    cursor[parts[-1]] = value


def _normalize_root_path(document: dict[str, Any], base_preset: Path) -> None:
    """Keep a generated preset's benchmark root independent of its new location."""
    general = document.get("general")
    if not isinstance(general, dict):
        return
    raw = general.get("root_path")
    if not isinstance(raw, str):
        return
    root = Path(raw)
    if not root.is_absolute():
        general["root_path"] = str((base_preset.parent / root).resolve())


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot load manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be a mapping")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ManifestError("manifest must contain a non-empty experiments list")
    return payload


def _bind_corpus_indices(
    manifest: dict[str, Any], experiments: list[dict[str, Any]]
) -> None:
    """Resolve stable corpus IDs to the bounded runner's zero-based indices."""
    raw_path = manifest.get("corpus_manifest")
    if not isinstance(raw_path, str) or not raw_path:
        return
    corpus_path = _resolve_repo_path(raw_path, "corpus_manifest")
    try:
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot load corpus manifest {corpus_path}: {error}") from error
    entries = corpus.get("entries") if isinstance(corpus, dict) else None
    if not isinstance(entries, list):
        raise ManifestError("corpus manifest must contain an entries list")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ManifestError("corpus manifest contains a malformed entry")
        entry_id = entry["id"]
        if entry_id in by_id:
            raise ManifestError(f"duplicate corpus entry id: {entry_id}")
        by_id[entry_id] = entry

    for experiment in experiments:
        corpus_ids = experiment.get("corpus_ids")
        if corpus_ids is None:
            continue
        if "indices" in experiment:
            raise ManifestError(
                f"experiment {experiment['name']!r} cannot set both corpus_ids and indices"
            )
        if not isinstance(corpus_ids, list) or not corpus_ids:
            raise ManifestError(
                f"experiment {experiment['name']!r} corpus_ids must be a non-empty list"
            )
        indices: list[int] = []
        for entry_id in corpus_ids:
            if not isinstance(entry_id, str) or entry_id not in by_id:
                raise ManifestError(
                    f"experiment {experiment['name']!r} has unknown corpus id {entry_id!r}"
                )
            entry = by_id[entry_id]
            if entry.get("kind") != "vnncomp":
                raise ManifestError(f"corpus entry {entry_id!r} is not a VNN-COMP row")
            if entry.get("category") != experiment.get("category"):
                raise ManifestError(
                    f"corpus entry {entry_id!r} category does not match "
                    f"experiment {experiment['name']!r}"
                )
            source_index = entry.get("source_index")
            if not isinstance(source_index, int) or source_index < 1:
                raise ManifestError(
                    f"corpus entry {entry_id!r} has invalid one-based source_index"
                )
            indices.append(source_index - 1)
        experiment["_resolved_indices"] = indices
        experiment["_corpus_manifest"] = str(corpus_path)


def _resolve_repo_path(raw: str, field: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise ManifestError(f"{field} does not exist: {path}")
    return path


def _selected_experiments(
    manifest: dict[str, Any], selected: set[str]
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in manifest["experiments"]:
        if not isinstance(raw, dict):
            raise ManifestError("each experiment must be a mapping")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ManifestError("each experiment requires a non-empty name")
        if name in seen:
            raise ManifestError(f"duplicate experiment name: {name}")
        seen.add(name)
        if not selected or name in selected:
            experiments.append(raw)
    missing = selected - seen
    if missing:
        raise ManifestError(f"unknown experiment(s): {', '.join(sorted(missing))}")
    return experiments


def _materialize_arm(
    *,
    base_preset: Path,
    arm: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    try:
        document = yaml.safe_load(base_preset.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot load base preset {base_preset}: {error}") from error
    if not isinstance(document, dict):
        raise ManifestError(f"base preset must be a mapping: {base_preset}")

    overrides = arm.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ManifestError(f"arm {arm.get('name')!r} overrides must be a mapping")
    for dotted_key, value in overrides.items():
        if not isinstance(dotted_key, str):
            raise ManifestError("override keys must be strings")
        _deep_set(document, dotted_key, value)
    _normalize_root_path(document, base_preset)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "# Generated by scripts/run_abcrown_transfer_factorials.py\n"
        "# Do not edit; source and overrides are bound in execution.json.\n"
        + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return overrides


def _runner_command(
    *,
    experiment: dict[str, Any],
    arm: dict[str, Any],
    preset: Path,
    output_csv: Path,
    args: argparse.Namespace,
) -> list[str]:
    category = experiment.get("category")
    if not isinstance(category, str) or not category:
        raise ManifestError(f"experiment {experiment.get('name')!r} requires category")
    command = [
        sys.executable,
        str(BOUNDED_RUNNER),
        "--year",
        str(experiment.get("year", 2025)),
        "--category",
        category,
        "--preset",
        str(preset),
        "--output",
        str(output_csv),
        "--tag",
        str(arm["name"]),
        "--timeout-slack",
        str(args.timeout_slack),
        "--domain-batch-metrics-dir",
        str(output_csv.parent / "domain-batch-metrics"),
        "--raw-artifact-dir",
        str(output_csv.parent / "raw-attempts"),
    ]
    indices = arm.get(
        "indices", experiment.get("_resolved_indices", experiment.get("indices"))
    )
    sample = arm.get("sample", experiment.get("sample"))
    if indices is not None:
        if isinstance(indices, list):
            indices = ",".join(str(index) for index in indices)
        command.extend(["--indices", str(indices)])
    elif sample is not None:
        command.extend(["--sample", str(sample)])
    if args.ny_binary:
        command.extend(["--ny-binary", str(Path(args.ny_binary).resolve())])
    if args.benchmark_root:
        command.extend(["--benchmark-root", str(Path(args.benchmark_root).resolve())])
    if args.max_domains is not None:
        command.extend(["--max-domains", str(args.max_domains)])
    if args.timeout_cap:
        command.extend(["--timeout-cap", str(args.timeout_cap)])
    if args.warmup_runs:
        command.extend(["--warmup-runs", str(args.warmup_runs)])
    if args.rerun_presearch:
        command.extend(["--rerun-presearch", str(args.rerun_presearch)])
    for extra_arg in arm.get("extra_args", []):
        command.extend(["--extra-arg", str(extra_arg)])
    return command


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Run one named experiment (repeatable; default: all).",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help=(
            "Run only an arm with this name (repeatable). Combine with "
            "--experiment when arm names are experiment-specific."
        ),
    )
    parser.add_argument("--ny-binary")
    parser.add_argument(
        "--benchmark-root",
        help="Explicit VNN-COMP benchmark root containing category directories.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout-slack", type=int, default=5)
    parser.add_argument(
        "--timeout-cap",
        type=int,
        default=0,
        help=(
            "Cap official row timeouts for explicitly non-promotional pilot runs; "
            "0 keeps official budgets."
        ),
    )
    parser.add_argument("--max-domains", type=int)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--rerun-presearch", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize/hash arms and print commands without executing NY.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if (
        args.timeout_slack < 0
        or args.timeout_cap < 0
        or args.warmup_runs < 0
        or args.rerun_presearch < 0
    ):
        raise ManifestError("timeout/warmup/rerun values must be non-negative")
    manifest_path = _resolve_repo_path(args.manifest, "manifest")
    manifest = _load_manifest(manifest_path)
    experiments = _selected_experiments(manifest, set(args.experiment))
    _bind_corpus_indices(manifest, experiments)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / stamp).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    execution: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "repo_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "dry_run": args.dry_run,
        "timeout_cap_seconds": args.timeout_cap or None,
        "benchmark_root": (
            str(Path(args.benchmark_root).resolve()) if args.benchmark_root else None
        ),
        "arms": [],
    }
    failed = False
    requested_arms = set(args.arm)
    seen_requested_arms: set[str] = set()
    for experiment in experiments:
        experiment_name = _safe_name(str(experiment["name"]))
        base_preset = _resolve_repo_path(
            str(experiment.get("base_preset", "")),
            f"{experiment_name}.base_preset",
        )
        arms = experiment.get("arms")
        if not isinstance(arms, list) or not arms:
            raise ManifestError(f"experiment {experiment_name} requires non-empty arms")
        for arm in arms:
            if not isinstance(arm, dict) or not isinstance(arm.get("name"), str):
                raise ManifestError(f"experiment {experiment_name} has malformed arm")
            if requested_arms and arm["name"] not in requested_arms:
                continue
            seen_requested_arms.add(arm["name"])
            arm_name = _safe_name(arm["name"])
            arm_dir = output_root / experiment_name / arm_name
            preset_path = arm_dir / "preset.yaml"
            overrides = _materialize_arm(
                base_preset=base_preset,
                arm=arm,
                destination=preset_path,
            )
            output_csv = arm_dir / "results.csv"
            command = _runner_command(
                experiment=experiment,
                arm=arm,
                preset=preset_path,
                output_csv=output_csv,
                args=args,
            )
            env_overrides = arm.get("env", {})
            if not isinstance(env_overrides, dict):
                raise ManifestError(f"arm {arm_name} env must be a mapping")
            env_overrides = {str(key): str(value) for key, value in env_overrides.items()}
            record: dict[str, Any] = {
                "experiment": experiment_name,
                "arm": arm_name,
                "base_preset": str(base_preset),
                "base_preset_sha256": _sha256(base_preset),
                "generated_preset": str(preset_path),
                "generated_preset_sha256": _sha256(preset_path),
                "overrides": overrides,
                "env": env_overrides,
                "command": command,
                "output_csv": str(output_csv),
                "domain_batch_metrics_dir": str(arm_dir / "domain-batch-metrics"),
                "raw_artifact_dir": str(arm_dir / "raw-attempts"),
                "disposition": arm.get("disposition", "measure"),
                "corpus_ids": experiment.get("corpus_ids"),
                "resolved_zero_based_indices": experiment.get("_resolved_indices"),
            }
            print(" ".join(command))
            if not args.dry_run:
                started = time.monotonic()
                process_env = os.environ.copy()
                process_env.update(env_overrides)
                process = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=process_env,
                    check=False,
                )
                record["returncode"] = process.returncode
                record["elapsed_seconds"] = round(time.monotonic() - started, 6)
                failed |= process.returncode != 0
            execution["arms"].append(record)

    missing_arms = requested_arms - seen_requested_arms
    if missing_arms:
        raise ManifestError(f"unknown selected arm(s): {', '.join(sorted(missing_arms))}")

    execution_path = output_root / "execution.json"
    execution_path.write_text(
        json.dumps(execution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"execution record: {execution_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
