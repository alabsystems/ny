#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Run preset-driven VNN-COMP beta-crown benchmarks with an external timeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import TypeVar

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports" / "benchmarks"
NY_PREFLIGHT_TIMEOUT_SECS = 5.0
T = TypeVar("T")
LOG = logging.getLogger(__name__)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks._shared import get_benchmark_instances
from scripts.benchmark_vnncomp_preset_bounded_results import (
    BenchmarkResult,
    NyProvenance,
    default_output_path,
    write_results,
)


def _resolve_ny_binary(explicit: str | None) -> tuple[Path, str]:
    """Resolve the ny binary path and return (path, source).

    Source is one of: "explicit", "shared-default".
    Policy (#4346): shared repo binaries are preferred over worker-local.
    Worker-local binaries require explicit --ny-binary to be used.
    """
    if explicit:
        return Path(explicit), "explicit"

    candidates: list[Path] = [
        REPO_ROOT / "target" / "release" / "ny",
        REPO_ROOT / "target" / "debug" / "ny",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate, "shared-default"

    raise FileNotFoundError(
        "No ny binary found. Build ny-cli or pass --ny-binary explicitly."
    )


def _compute_provenance(ny_binary: Path, source: str) -> NyProvenance:
    """Compute binary provenance metadata for CSV recording (#4346)."""
    try:
        version_result = subprocess.run(
            [str(ny_binary), "--version"],
            capture_output=True, text=True, timeout=5.0, check=False,
        )
        version = (version_result.stdout or "").strip() or "unknown"
    except (subprocess.TimeoutExpired, OSError):
        version = "unknown"

    try:
        sha256 = hashlib.sha256(ny_binary.read_bytes()).hexdigest()
    except OSError:
        sha256 = "unknown"

    return NyProvenance(
        source=source,
        binary=str(ny_binary),
        version=version,
        sha256=sha256,
    )


def _preflight_ny_binary(ny_binary: Path, timeout_secs: float | None = None) -> None:
    """Fail fast when the chosen binary cannot answer `--version`."""
    timeout_secs = NY_PREFLIGHT_TIMEOUT_SECS if timeout_secs is None else timeout_secs
    command = [str(ny_binary), "--version"]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True,
            timeout=timeout_secs, cwd=REPO_ROOT, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Ny binary preflight timed out after {timeout_secs:.1f}s "
            f"while running `--version`: {ny_binary}. "
            "Use a clean rebuilt binary or pass --ny-binary explicitly."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Ny binary preflight could not execute {ny_binary}: {exc}"
        ) from exc
    if process.returncode != 0:
        detail = (process.stderr or "").strip() or (process.stdout or "").strip() or f"exit_code={process.returncode}"
        raise RuntimeError(
            f"Ny binary preflight failed while running `--version`: {ny_binary} ({detail})"
        )


def _parse_json_from_output(text: str) -> dict | None:
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)

    return None


def _normalize_status(raw: str) -> str:
    status = raw.lower()
    if status == "safe":
        return "verified"
    if status == "violated":
        return "falsified"
    return status


def _sample_evenly(items: list[T], sample_size: int) -> list[tuple[int, T]]:
    if sample_size >= len(items):
        return list(enumerate(items))

    step = len(items) / sample_size
    sampled: list[tuple[int, T]] = []
    for sample_index in range(sample_size):
        item_index = int(sample_index * step)
        sampled.append((item_index, items[item_index]))
    return sampled


def _select_instances(
    instances: list[tuple[Path, Path, int]],
    sample: int,
    indices: list[int] | None,
) -> list[tuple[int, tuple[Path, Path, int]]]:
    if indices:
        selected: list[tuple[int, tuple[Path, Path, int]]] = []
        for index in indices:
            if index < 0 or index >= len(instances):
                raise ValueError(f"Index {index} out of range for {len(instances)} instances")
            selected.append((index, instances[index]))
        return selected

    if sample > 0:
        return _sample_evenly(instances, sample)

    return list(enumerate(instances))


def _build_command(
    ny_binary: Path, model_path: Path, property_path: Path,
    preset_path: Path, timeout: int, max_domains: int | None,
    domain_batch_metrics_jsonl: Path | None, extra_args: list[str],
) -> list[str]:
    cmd = [str(ny_binary), "beta-crown", str(model_path),
           "--property", str(property_path), "--preset", str(preset_path),
           "--timeout", str(timeout), "--json"]
    if max_domains is not None:
        cmd.extend(["--max-domains", str(max_domains)])
    if domain_batch_metrics_jsonl is not None:
        cmd.extend(["--domain-batch-metrics-jsonl", str(domain_batch_metrics_jsonl)])
    cmd.extend(extra_args)
    return cmd


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _write_attempt_artifacts(
    artifact_dir: Path | None,
    *,
    command: list[str],
    stdout: str,
    stderr: str,
    result: BenchmarkResult,
    elapsed: float,
    returncode: int | None,
    external_timeout: int | None,
) -> None:
    """Retain complete decoded process output and attempt metadata when requested."""
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(REPO_ROOT),
                "external_timeout_seconds": external_timeout,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (artifact_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    (artifact_dir / "result.txt").write_text(result.result + "\n", encoding="utf-8")


def _run_single_attempt(
    ny_binary: Path,
    preset_path: Path,
    model_path: Path,
    property_path: Path,
    timeout: int,
    timeout_slack: int | None,
    max_domains: int | None,
    domain_batch_metrics_jsonl: Path | None,
    extra_args: list[str],
    artifact_dir: Path | None = None,
) -> BenchmarkResult:
    command = _build_command(
        ny_binary=ny_binary, model_path=model_path,
        property_path=property_path, preset_path=preset_path,
        timeout=timeout, max_domains=max_domains,
        domain_batch_metrics_jsonl=domain_batch_metrics_jsonl,
        extra_args=extra_args,
    )
    sidecar_str = str(domain_batch_metrics_jsonl or "")
    common = dict(model=model_path.name, property=property_path.name,
                  timeout=timeout, domain_batch_metrics_jsonl=sidecar_str)
    external_timeout = None if timeout_slack is None else timeout + timeout_slack
    start = time.time()
    try:
        process = subprocess.run(
            command, capture_output=True, text=True,
            timeout=external_timeout, cwd=REPO_ROOT,
        )
        elapsed = time.time() - start
        stdout = process.stdout or ""
        stderr = process.stderr or ""
        returncode: int | None = process.returncode
    except subprocess.TimeoutExpired as error:
        elapsed = time.time() - start
        stdout = _timeout_output(error.stdout)
        stderr = _timeout_output(error.stderr)
        result = BenchmarkResult(
            **common, result="timeout_ext", elapsed=elapsed,
            domains_explored=0, domains_verified=0, max_depth_reached=0,
            reason=f"external_timeout_{external_timeout}s",
        )
        _write_attempt_artifacts(
            artifact_dir,
            command=command,
            stdout=stdout,
            stderr=stderr,
            result=result,
            elapsed=elapsed,
            returncode=None,
            external_timeout=external_timeout,
        )
        return result
    payload = _parse_json_from_output(stdout) or _parse_json_from_output(stderr)
    if payload:
        status = _normalize_status(
            payload.get("property_status", payload.get("status", "unknown")))
        result = BenchmarkResult(
            **common, result=status, elapsed=elapsed,
            domains_explored=int(payload.get("domains_explored", 0) or 0),
            domains_verified=int(payload.get("domains_verified", 0) or 0),
            max_depth_reached=int(payload.get("max_depth_reached", 0) or 0),
            reason=str(payload.get("reason", "")),
        )
    else:
        output_tail = (stdout + stderr).strip().splitlines()
        reason = output_tail[-1][:200] if output_tail else f"exit_code={process.returncode}"
        result = BenchmarkResult(
            **common, result="error", elapsed=elapsed,
            domains_explored=0, domains_verified=0, max_depth_reached=0,
            reason=reason,
        )
    _write_attempt_artifacts(
        artifact_dir,
        command=command,
        stdout=stdout,
        stderr=stderr,
        result=result,
        elapsed=elapsed,
        returncode=returncode,
        external_timeout=external_timeout,
    )
    return result


def _is_presearch_result(result: BenchmarkResult) -> bool:
    """A result is "pre-search" when it ended before BaB search started.

    Predicate from #4412 design: domains_explored == 0 and status is not
    verified or falsified.
    """
    if result.result in ("verified", "falsified"):
        return False
    return result.domains_explored == 0


def _build_retry_notes(
    warmup_runs: int, retry_count: int, initial: BenchmarkResult,
) -> str:
    """Build provenance notes for a retried pre-search row (#4412 Packet C)."""
    parts: list[str] = []
    if warmup_runs > 0:
        parts.append(f"warmup_runs={warmup_runs}")
    parts.append(f"measured_attempts={1 + retry_count}")
    parts.append(f"presearch_retry={retry_count}")
    parts.append(f"initial_result={initial.result}")
    parts.append(f"initial_domains={initial.domains_explored}")
    if initial.reason:
        parts.append(f"initial_reason={initial.reason[:100]}")
    return "; ".join(parts)


def _run_instance(
    ny_binary: Path,
    preset_path: Path,
    model_path: Path,
    property_path: Path,
    timeout: int,
    timeout_slack: int,
    max_domains: int | None,
    domain_batch_metrics_jsonl: Path | None,
    extra_args: list[str],
    warmup_runs: int = 0,
    rerun_presearch: int = 0,
    raw_artifact_dir: Path | None = None,
) -> BenchmarkResult:
    """Run one instance with optional warmup and pre-search rerun policy (#4412)."""
    base = dict(
        ny_binary=ny_binary, preset_path=preset_path,
        model_path=model_path, property_path=property_path,
        timeout=timeout, max_domains=max_domains, extra_args=extra_args,
    )
    for warmup_idx in range(warmup_runs):
        LOG.info("    warmup %d/%d", warmup_idx + 1, warmup_runs)
        _run_single_attempt(
            **base,
            timeout_slack=None,
            domain_batch_metrics_jsonl=None,
            artifact_dir=(
                raw_artifact_dir / f"warmup-{warmup_idx + 1:02d}"
                if raw_artifact_dir is not None
                else None
            ),
        )

    result = _run_single_attempt(
        **base,
        timeout_slack=timeout_slack,
        domain_batch_metrics_jsonl=domain_batch_metrics_jsonl,
        artifact_dir=(
            raw_artifact_dir / "measured-01"
            if raw_artifact_dir is not None
            else None
        ),
    )

    if rerun_presearch > 0 and _is_presearch_result(result):
        initial_result = result
        for retry_idx in range(rerun_presearch):
            LOG.info(
                "    presearch retry %d/%d (initial: %s, domains=%d)",
                retry_idx + 1, rerun_presearch,
                initial_result.result, initial_result.domains_explored,
            )
            if domain_batch_metrics_jsonl is not None:
                sidecar = Path(domain_batch_metrics_jsonl)
                if sidecar.exists():
                    sidecar.unlink()
            result = _run_single_attempt(
                **base,
                timeout_slack=timeout_slack,
                domain_batch_metrics_jsonl=domain_batch_metrics_jsonl,
                artifact_dir=(
                    raw_artifact_dir / f"measured-{retry_idx + 2:02d}"
                    if raw_artifact_dir is not None
                    else None
                ),
            )
            if not _is_presearch_result(result):
                break
        result.notes = _build_retry_notes(warmup_runs, retry_idx + 1, initial_result)
    elif warmup_runs > 0:
        result.notes = f"warmup_runs={warmup_runs}"

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preset-driven VNN-COMP beta-crown benchmarks with an external timeout."
    )
    parser.add_argument("--year", type=int, default=2025, help="VNN-COMP year (default: 2025)")
    parser.add_argument("--category", required=True, help="Benchmark category name")
    parser.add_argument(
        "--benchmark-root",
        default="",
        help=(
            "Explicit directory containing category folders. "
            "Default: benchmarks/vnncomp<year>/benchmarks in this checkout"
        ),
    )
    parser.add_argument(
        "--preset",
        default="",
        help="Preset YAML path (default: configs/vnncomp25/<category>.yaml)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Evenly sample N instances instead of running the full category",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated instance indices to run instead of the full category",
    )
    parser.add_argument(
        "--timeout-slack",
        type=int,
        default=5,
        help="Extra seconds before the external timeout kills ny (default: 5)",
    )
    parser.add_argument(
        "--timeout-cap",
        type=int,
        default=0,
        help=(
            "Cap each official instance timeout for non-promotional pilot runs "
            "(default: 0, use the official timeout)"
        ),
    )
    parser.add_argument(
        "--max-domains",
        type=int,
        default=-1,
        help="Optional max-domains override (default: use preset/CLI defaults)",
    )
    parser.add_argument(
        "--ny-binary",
        default="",
        help="Path to ny binary (default: shared repo release/debug)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix tag for the output CSV name",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Explicit output CSV path",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra beta-crown CLI argument to append (repeatable)",
    )
    parser.add_argument(
        "--domain-batch-metrics-dir",
        default="",
        help="Directory for per-run graph domain-batch JSONL sidecars",
    )
    parser.add_argument(
        "--raw-artifact-dir",
        default="",
        help=(
            "Directory in which to retain complete stdout, stderr, command, "
            "timing, and result output for every warmup/measured attempt"
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Untimed row-local warmup runs before the first measured attempt (default: 0)",
    )
    parser.add_argument(
        "--rerun-presearch",
        type=int,
        default=0,
        help="Max extra measured attempts when the current attempt ends before search (default: 0)",
    )
    return parser.parse_args()


def _validate_inputs(
    ny_binary: Path, preset_path: Path,
) -> NyProvenance | None:
    """Validate binary and preset, return provenance or None on error."""
    if not preset_path.exists():
        LOG.error("Preset not found: %s", preset_path)
        return None
    if not ny_binary.exists():
        LOG.error("Ny binary not found: %s", ny_binary)
        return None
    try:
        _preflight_ny_binary(ny_binary)
    except RuntimeError as err:
        LOG.error("%s", err)
        return None
    return NyProvenance(source="", binary="", version="", sha256="")


def main() -> int:
    args = parse_args()
    if args.timeout_cap < 0:
        raise ValueError("--timeout-cap must be non-negative")
    ny_binary, ny_source = _resolve_ny_binary(args.ny_binary or None)
    preset_path = Path(args.preset) if args.preset else REPO_ROOT / "configs" / "vnncomp25" / f"{args.category}.yaml"

    sentinel = _validate_inputs(ny_binary, preset_path)
    if sentinel is None:
        return 2

    provenance = _compute_provenance(ny_binary, ny_source)
    LOG.info(
        "Binary provenance: source=%s, sha256=%s..., version=%s",
        provenance.source, provenance.sha256[:16], provenance.version,
    )

    benchmark_root = Path(args.benchmark_root).resolve() if args.benchmark_root else None
    if benchmark_root is None:
        # Preserve the historical two-argument call for callers/tests that
        # replace this helper while exercising the bounded runner.
        instances = get_benchmark_instances(args.year, args.category)
    else:
        instances = get_benchmark_instances(
            args.year, args.category, benchmark_root=benchmark_root
        )
    if not instances:
        LOG.error("No instances found for %s %s", args.year, args.category)
        return 1

    indices = [int(part) for part in args.indices.split(",") if part.strip()] or None
    selected = _select_instances(instances, sample=args.sample, indices=indices)
    max_domains = args.max_domains if args.max_domains >= 0 else None
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(REPORTS_DIR, args.category, args.tag or None)
    )
    domain_batch_metrics_dir = (
        Path(args.domain_batch_metrics_dir)
        if args.domain_batch_metrics_dir
        else None
    )
    if domain_batch_metrics_dir is not None:
        domain_batch_metrics_dir.mkdir(parents=True, exist_ok=True)
    raw_artifact_dir = Path(args.raw_artifact_dir) if args.raw_artifact_dir else None
    if raw_artifact_dir is not None:
        raw_artifact_dir.mkdir(parents=True, exist_ok=False)

    LOG.info(
        "Running %d/%d %s instances with %s and preset %s",
        len(selected), len(instances), args.category, ny_binary, preset_path,
    )

    results: list[BenchmarkResult] = []
    counts: dict[str, int] = {}
    for position, (index, (model_path, property_path, timeout)) in enumerate(selected, start=1):
        effective_timeout = (
            min(timeout, args.timeout_cap) if args.timeout_cap > 0 else timeout
        )
        LOG.info(
            "[%d/%d] idx=%d %s / %s (timeout=%ss%s)",
            position, len(selected), index,
            model_path.name, property_path.name, timeout,
            (
                f", pilot_cap={effective_timeout}s"
                if effective_timeout != timeout
                else ""
            ),
        )
        domain_batch_metrics_jsonl = None
        if domain_batch_metrics_dir is not None:
            domain_batch_metrics_jsonl = domain_batch_metrics_dir / f"{args.category}_idx{index:04d}.jsonl"
        row_artifact_dir = (
            raw_artifact_dir / f"{args.category}_idx{index:04d}"
            if raw_artifact_dir is not None
            else None
        )
        result = _run_instance(
            ny_binary=ny_binary, preset_path=preset_path,
            model_path=model_path, property_path=property_path,
            timeout=effective_timeout, timeout_slack=args.timeout_slack,
            max_domains=max_domains,
            domain_batch_metrics_jsonl=domain_batch_metrics_jsonl,
            extra_args=args.extra_arg,
            warmup_runs=args.warmup_runs,
            rerun_presearch=args.rerun_presearch,
            raw_artifact_dir=row_artifact_dir,
        )
        counts[result.result] = counts.get(result.result, 0) + 1
        LOG.info(
            "  -> %s (%.2fs, domains=%d, verified=%d)",
            result.result, result.elapsed,
            result.domains_explored, result.domains_verified,
        )
        results.append(result)

    write_results(output_path, results, provenance)
    LOG.info("Saved results to %s", output_path)
    LOG.info(json.dumps({"counts": counts, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
