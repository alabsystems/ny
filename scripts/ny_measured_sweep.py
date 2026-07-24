#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
"""Run ny on a VNN-COMP benchmark's instances and emit an official-format results CSV.

Produces rows `category,onnx,vnnlib,prepared,result,runtime` (the column shape
scripts/vnncomp_competitive_score.py and the official vnncomp2025_results
pipeline consume), so a measured ny sweep can be scored against the 1566.9 bar.

Usage:
  scripts/ny_measured_sweep.py <category> [--timeout 60] [--limit 0] [--workers 4]
                               [--corpus benchmarks/vnncomp2025/benchmarks]
                               [--ny target/release/ny] [--out reports/measured]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_instance(ny: Path, corpus: Path, cat: str, onnx: str, vnnlib: str,
                 timeout: int) -> tuple[str, float]:
    """Run one instance; return (result_token, wallclock_seconds)."""
    bench_dir = corpus / cat
    res_file = Path(f"/tmp/ny_sweep_{abs(hash((onnx, vnnlib)))}.txt")
    cmd = [str(ny), "vnncomp", "v1", cat, onnx, vnnlib, str(res_file), str(timeout)]
    start = time.time()
    try:
        # Hard wall-clock backstop = competition budget + grace (mirrors run_instance.sh).
        subprocess.run(cmd, cwd=str(bench_dir), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=timeout + 15, check=False)
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - start
    elapsed = time.time() - start
    try:
        token = res_file.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        token = "unknown"
    finally:
        res_file.unlink(missing_ok=True)
    return token or "unknown", elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("category")
    ap.add_argument("--timeout", type=int, default=60, help="per-instance budget (s)")
    ap.add_argument("--limit", type=int, default=0, help="cap #instances (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corpus", default="benchmarks/vnncomp2025/benchmarks")
    ap.add_argument("--ny", default="target/release/ny")
    ap.add_argument("--out", default="reports/measured")
    args = ap.parse_args()

    corpus = (REPO / args.corpus).resolve()
    ny = (REPO / args.ny).resolve()
    inst_csv = corpus / args.category / "instances.csv"
    if not inst_csv.is_file():
        print(f"no instances.csv for {args.category} at {inst_csv}", file=sys.stderr)
        return 2

    rows = []
    with inst_csv.open(encoding="utf-8") as fh:
        for parts in csv.reader(fh):
            if len(parts) < 2 or not parts[0].strip():
                continue
            # Per-instance official budget (column 3) when --timeout 0; else the cap.
            inst_to = args.timeout
            if args.timeout == 0 and len(parts) >= 3 and parts[2].strip():
                try:
                    inst_to = int(float(parts[2].strip()))
                except ValueError:
                    inst_to = 100
            rows.append((parts[0].strip(), parts[1].strip(), inst_to))
    if args.limit > 0:
        rows = rows[: args.limit]

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.category}.csv"

    print(f"[{args.category}] {len(rows)} instances, timeout={args.timeout}s, "
          f"workers={args.workers} -> {out_csv}", file=sys.stderr)

    results: dict[int, tuple[str, float]] = {}
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_instance, ny, corpus, args.category, o, v, to): i
                for i, (o, v, to) in enumerate(rows)}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            tok, rt = results[i]
            print(f"  [{done}/{len(rows)}] {rows[i][1].split('/')[-1]:40s} {tok:9s} {rt:7.2f}s",
                  file=sys.stderr)

    counts: dict[str, int] = {}
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for i, (onnx, vnnlib, _to) in enumerate(rows):
            tok, rt = results[i]
            counts[tok] = counts.get(tok, 0) + 1
            w.writerow([args.category, onnx, vnnlib, "prepared", tok, f"{rt:.2f}"])

    wall = time.time() - t0
    print(f"[{args.category}] done in {wall:.1f}s  counts={counts}", file=sys.stderr)
    print(f"  unsat={counts.get('unsat',0)} sat={counts.get('sat',0)} "
          f"timeout={counts.get('timeout',0)} unknown={counts.get('unknown',0)} "
          f"error={counts.get('error',0)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
