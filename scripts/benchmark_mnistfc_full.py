#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Full mnistfc VNN-COMP 2021 benchmark.

Runs all 89 instances across 3 models (256x2, 256x4, 256x6) with
alpha-CROWN and CROWN methods, recording results to JSON.

Part of #3290: Complete mnistfc benchmark coverage.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add benchmarks/ to path for _shared import
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from _shared import (
    NY_BINARY,
    get_benchmark_instances,
    run_ny_verify,
)

log = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).parent.parent / "reports" / "benchmarks"

STATUS_SYMBOLS = {"verified": "V", "falsified": "F", "unknown": "?", "timeout": "T", "error": "E"}
COUNTER_KEYS = ("total", "verified", "falsified", "unknown", "timeout", "error")


def _run_model_instances(insts: list[tuple], model_name: str) -> dict:
    """Run alpha-CROWN on all instances for a single model."""
    log.info("=" * 60)
    log.info("Model: %s (%d instances)", model_name, len(insts))

    model_results: dict = {k: 0 for k in COUNTER_KEYS}
    model_results.update({"total_time": 0.0, "instances": []})

    for i, (net, prop, timeout) in enumerate(insts):
        effective_timeout = min(timeout, 60)
        result = run_ny_verify(net, prop, timeout=effective_timeout, method="alpha")
        sym = STATUS_SYMBOLS.get(result.status, "?")
        log.info("  [%2d/%d] %s: %s (%.2fs)", i + 1, len(insts), prop.stem, sym, result.time_seconds)

        model_results["total"] += 1
        model_results[result.status] += 1
        model_results["total_time"] += result.time_seconds
        model_results["instances"].append({
            "property": prop.name, "status": result.status, "time": round(result.time_seconds, 3),
        })

    total = model_results["total"]
    if total > 0:
        model_results["verified_rate"] = round(model_results["verified"] / total * 100, 1)
        model_results["avg_time"] = round(model_results["total_time"] / total, 3)
    log.info("  Summary: %d/%d verified (%.1f%%)", model_results["verified"], total, model_results.get("verified_rate", 0))
    return model_results


def run_mnistfc_full() -> dict:
    """Run full mnistfc benchmark with alpha-CROWN across all 3 models."""
    instances = get_benchmark_instances(2021, "mnistfc")
    log.info("mnistfc: %d instances found", len(instances))
    if not instances:
        log.error("No mnistfc instances found. Run benchmarks/download_benchmarks.sh first.")
        sys.exit(1)

    models: dict[str, list] = {}
    for net, prop, timeout in instances:
        models.setdefault(net.stem, []).append((net, prop, timeout))

    all_results = {
        "benchmark": "mnistfc", "year": 2021, "method": "alpha",
        "binary": str(NY_BINARY), "timestamp": datetime.now().isoformat(),
        "models": {}, "totals": {k: 0 for k in COUNTER_KEYS} | {"total_time": 0.0},
    }

    for model_name, insts in sorted(models.items()):
        mr = _run_model_instances(insts, model_name)
        all_results["models"][model_name] = mr
        for key in COUNTER_KEYS:
            all_results["totals"][key] += mr[key]
        all_results["totals"]["total_time"] += mr["total_time"]

    totals = all_results["totals"]
    if totals["total"] > 0:
        totals["verified_rate"] = round(totals["verified"] / totals["total"] * 100, 1)
        totals["avg_time"] = round(totals["total_time"] / totals["total"], 3)
    return all_results


def run_cifar10_resnet_2b_alpha() -> dict:
    """Re-benchmark cifar10_resnet_2b with alpha-CROWN (Approach B)."""
    instances = get_benchmark_instances(2021, "cifar10_resnet")
    log.info("cifar10_resnet: %d instances found", len(instances))
    if not instances:
        log.warning("No cifar10_resnet instances found.")
        return {}

    resnet_2b = [(n, p, t) for n, p, t in instances if "2b" in n.stem]
    log.info("resnet_2b: %d instances", len(resnet_2b))

    results: dict = {
        "benchmark": "cifar10_resnet_2b", "year": 2021, "method": "alpha",
        "binary": str(NY_BINARY), "timestamp": datetime.now().isoformat(),
        "total": 0, "verified": 0, "unknown": 0, "error": 0, "timeout": 0,
        "total_time": 0.0, "instances": [],
    }

    for i, (net, prop, timeout) in enumerate(resnet_2b):
        effective_timeout = min(timeout, 60)
        result = run_ny_verify(net, prop, timeout=effective_timeout, method="alpha")
        sym = STATUS_SYMBOLS.get(result.status, "E")
        log.info("  [%2d/%d] %s: %s (%.2fs)", i + 1, len(resnet_2b), prop.stem, sym, result.time_seconds)

        results["total"] += 1
        results[result.status] = results.get(result.status, 0) + 1
        results["total_time"] += result.time_seconds
        results["instances"].append({
            "property": prop.name, "status": result.status, "time": round(result.time_seconds, 3),
        })

    if results["total"] > 0:
        results["verified_rate"] = round(results["verified"] / results["total"] * 100, 1)
        results["avg_time"] = round(results["total_time"] / results["total"], 3)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("Binary: %s (exists=%s)", NY_BINARY, NY_BINARY.exists())
    if not NY_BINARY.exists():
        log.error("ny binary not found. Run: cargo build -p ny-cli --release")
        sys.exit(1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mnistfc_results = run_mnistfc_full()
    mnistfc_file = REPORT_DIR / f"mnistfc_alpha_{timestamp}.json"
    with open(mnistfc_file, "w") as f:
        json.dump(mnistfc_results, f, indent=2)
    log.info("Results saved: %s", mnistfc_file)

    cifar_results = run_cifar10_resnet_2b_alpha()
    if cifar_results:
        cifar_file = REPORT_DIR / f"cifar10_resnet_2b_alpha_approachB_{timestamp}.json"
        with open(cifar_file, "w") as f:
            json.dump(cifar_results, f, indent=2)
        log.info("Results saved: %s", cifar_file)

    totals = mnistfc_results.get("totals", {})
    log.info("mnistfc: %d/%d (%.1f%%)", totals.get("verified", 0), totals.get("total", 0), totals.get("verified_rate", 0))
    if cifar_results:
        log.info("cifar10_resnet_2b: %d/%d (%.1f%%)", cifar_results.get("verified", 0), cifar_results.get("total", 0), cifar_results.get("verified_rate", 0))


if __name__ == "__main__":
    main()
