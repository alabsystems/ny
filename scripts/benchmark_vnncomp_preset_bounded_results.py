# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import csv
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


@dataclass
class NyProvenance:
    source: str  # "explicit", "shared-default"
    binary: str
    version: str
    sha256: str


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
]


def default_output_path(reports_dir: Path, category: str, tag: str | None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    return reports_dir / f"{category}_{timestamp}{suffix}.csv"


def write_results(
    path: Path,
    results: list[BenchmarkResult],
    provenance: NyProvenance | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
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
                ]
            )
