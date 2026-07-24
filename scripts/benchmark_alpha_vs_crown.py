#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Compare alpha-CROWN vs CROWN on mnistfc 256x2 and cifar10_resnet_2b.

Measures the bound tightness improvement from alpha optimization (Approach B).

Part of #3290: Benchmark coverage.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from _shared import NY_BINARY, get_benchmark_instances, run_ny_verify

log = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).parent.parent / "reports" / "benchmarks"


def compare_methods(
    year: int, benchmark: str, model_filter: str | None = None, max_instances: int = 30
) -> dict:
    """Run both CROWN and alpha-CROWN, compare verified counts."""
    instances = get_benchmark_instances(year, benchmark)
    if model_filter:
        instances = [(n, p, t) for n, p, t in instances if model_filter in n.stem]
    instances = instances[:max_instances]
    log.info("%s (%s): %d instances", benchmark, model_filter or "all", len(instances))

    crown_results = []
    alpha_results = []

    for i, (net, prop, timeout) in enumerate(instances):
        effective_timeout = min(timeout, 60)
        crown = run_ny_verify(net, prop, timeout=effective_timeout, method="crown")
        alpha = run_ny_verify(net, prop, timeout=effective_timeout, method="alpha")
        crown_results.append(crown)
        alpha_results.append(alpha)

        cs = "V" if crown.status == "verified" else "?"
        as_ = "V" if alpha.status == "verified" else "?"
        note = ""
        if alpha.status == "verified" and crown.status != "verified":
            note = " [ALPHA WIN]"
        log.info("  [%2d/%d] %s: CROWN=%s alpha=%s%s", i + 1, len(instances), prop.stem, cs, as_, note)

    crown_v = sum(1 for r in crown_results if r.status == "verified")
    alpha_v = sum(1 for r in alpha_results if r.status == "verified")
    crown_avg = sum(r.time_seconds for r in crown_results) / max(len(crown_results), 1)
    alpha_avg = sum(r.time_seconds for r in alpha_results) / max(len(alpha_results), 1)

    log.info("  CROWN: %d/%d, alpha: %d/%d, delta: %+d", crown_v, len(instances), alpha_v, len(instances), alpha_v - crown_v)

    return {
        "benchmark": benchmark, "model_filter": model_filter, "total": len(instances),
        "crown_verified": crown_v, "alpha_verified": alpha_v, "delta": alpha_v - crown_v,
        "crown_avg_time": round(crown_avg, 3), "alpha_avg_time": round(alpha_avg, 3),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("Binary: %s", NY_BINARY)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = [
        compare_methods(2021, "mnistfc", "256x2"),
        compare_methods(2021, "mnistfc", "256x4"),
        compare_methods(2021, "cifar10_resnet", "2b", max_instances=48),
    ]

    log.info("=" * 60)
    for r in results:
        log.info("  %s (%s): CROWN=%d/%d, alpha=%d/%d, delta=%+d",
                 r["benchmark"], r["model_filter"], r["crown_verified"], r["total"],
                 r["alpha_verified"], r["total"], r["delta"])

    report_file = REPORT_DIR / f"alpha_vs_crown_comparison_{timestamp}.json"
    with open(report_file, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)
    log.info("Saved: %s", report_file)


if __name__ == "__main__":
    main()
