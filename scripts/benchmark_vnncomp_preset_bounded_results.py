# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BenchmarkResult:
    model: str
    property: str
    timeout: int
    result: str
    elapsed: float
    domains_explored: int
    domains_verified: int
    max_depth_reached: int
    reason: str
    domain_batch_metrics_jsonl: str
    notes: str = ""
    source_index_zero_based: int = -1
    effective_config_json: str = ""
    effective_config_sha256: str = ""
    execution_observations_json: str = ""
    execution_observations_sha256: str = ""


@dataclass
class NyProvenance:
    source: str  # "explicit", "shared-default"
    binary: str
    version: str
    sha256: str
    receipt_json: str = ""
    receipt_sha256: str = ""


@dataclass
class RunProvenance:
    preset_sha256: str
    parent_env_json: str
    parent_env_sha256: str


CSV_HEADER = [
    "model",
    "property",
    "timeout",
    "result",
    "elapsed",
    "domains",
    "domains_verified",
    "max_depth",
    "reason",
    "domain_batch_metrics_jsonl",
    "notes",
    "ny_source",
    "ny_binary",
    "ny_version",
    "ny_sha256",
    # Keep the original positional CSV contract above intact. New identity and
    # treatment-evidence fields are append-only for legacy non-DictReader
    # consumers.
    "source_index_zero_based",
    "effective_config_json",
    "effective_config_sha256",
    "ny_receipt_json",
    "ny_receipt_sha256",
    "preset_sha256",
    "parent_env_json",
    "parent_env_sha256",
    "execution_observations_json",
    "execution_observations_sha256",
]


def default_output_path(reports_dir: Path, category: str, tag: str | None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    return reports_dir / f"{category}_{timestamp}{suffix}.csv"


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_results(
    path: Path,
    results: list[BenchmarkResult],
    provenance: NyProvenance | None = None,
    run_provenance: RunProvenance | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    staged_identity: tuple[int, int] | None = None
    published_identity: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            staged_stat = os.fstat(handle.fileno())
            staged_identity = (staged_stat.st_dev, staged_stat.st_ino)
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            for result in results:
                writer.writerow(
                    [
                        result.model,
                        result.property,
                        result.timeout,
                        result.result,
                        f"{result.elapsed:.2f}",
                        result.domains_explored,
                        result.domains_verified,
                        result.max_depth_reached,
                        result.reason,
                        result.domain_batch_metrics_jsonl,
                        result.notes,
                        provenance.source if provenance else "",
                        provenance.binary if provenance else "",
                        provenance.version if provenance else "",
                        provenance.sha256 if provenance else "",
                        result.source_index_zero_based,
                        result.effective_config_json,
                        result.effective_config_sha256,
                        provenance.receipt_json if provenance else "",
                        provenance.receipt_sha256 if provenance else "",
                        run_provenance.preset_sha256 if run_provenance else "",
                        run_provenance.parent_env_json if run_provenance else "",
                        run_provenance.parent_env_sha256 if run_provenance else "",
                        result.execution_observations_json,
                        result.execution_observations_sha256,
                    ]
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_stat = temporary.stat()
        if (temporary_stat.st_dev, temporary_stat.st_ino) != staged_identity:
            raise RuntimeError("staged result path changed before publication")
        os.link(temporary, path)
        published_identity = staged_identity
        published_stat = os.stat(path, follow_symlinks=False)
        if (published_stat.st_dev, published_stat.st_ino) != published_identity:
            raise RuntimeError("published result path does not reference staged bytes")
        _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if published_identity is not None:
            try:
                published_stat = os.stat(path, follow_symlinks=False)
                if (published_stat.st_dev, published_stat.st_ino) == published_identity:
                    path.unlink()
                    _fsync_directory(path.parent)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)
        raise
