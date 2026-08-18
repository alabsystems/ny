#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Check banked rows against their official per-instance budgets, both directions.

FAILS (exit 1) on OVER-budget credited verdicts — a row credited `sat`/`unsat`
with a runtime exceeding its budget is a scoring INFLATION; at the real budget it
would have been a `timeout` worth 0.

REPORTS (exit 0) UNDER-budget timeouts — a row banked `timeout`/`unknown` after
using well under its budget was never given its full official time, so the bank
may UNDERSTATE ny. These score 0 either way, so they are not a correctness bug;
they are re-measure candidates.

SWEPT 2026-08-12, and the answer is NO POINTS ARE HIDDEN THERE. 178 such rows
exist; every scored cluster was re-run at FULL official budget and **76/76
still failed**:
  safenlp_2024   66/66 timeout at 20 s (banked at 15 s)
  nn4sys          6/6  timeout at 540-800 s (banked at ~110 s)
  cgan_2023       2/2  timeout at 1200 s (banked at 652 s / 1039 s)
  metaroom_2023   1/1  timeout at 210 s (banked at 55 s)
  tinyimagenet    1    banked `unknown` after 1.0 s of 100 s looked like a live
                       refusal; re-running gives an ordinary timeout, so the 1 s
                       entry is a stale bank artifact. Both score 0.
Only the extended-track clusters (lsnc_relu 69, ml4acopf 15) remain unswept, and
they are not in the scored 16.

The scorecard trusts `reports/measured/*.csv`. A row credited `sat`/`unsat` with a
runtime EXCEEDING its official per-instance budget (field 3 of the benchmark's
`instances.csv`) is a scoring INFLATION: at the real budget it would have been a
`timeout` worth 0. Timeouts measured PAST budget are fine and are ignored — a
105 s timeout is certainly a 30 s timeout.

Budgets are PER INSTANCE, not per benchmark (nn4sys alone spans 20-800 s), which
is why this compares row by row. Exit 1 on any violation.

Run: python3 scripts/check_bank_budget_parity.py
"""
import csv
import os
import sys

BENCH_ROOT = "benchmarks/vnncomp2025/benchmarks"
BANK = "reports/measured"


def budgets(bench):
    path = os.path.join(BENCH_ROOT, bench, "instances.csv")
    if not os.path.isfile(path):
        return None
    out = {}
    for row in csv.reader(open(path)):
        if len(row) < 3:
            continue
        try:
            out[(os.path.basename(row[0].strip()), os.path.basename(row[1].strip()))] = float(
                row[2].strip()
            )
        except ValueError:
            continue
    return out


def main():
    violations = []
    undermeasured = []
    checked = skipped = 0
    for fn in sorted(os.listdir(BANK)):
        if not fn.endswith(".csv"):
            continue
        bench = fn[:-4]
        bud = budgets(bench)
        if bud is None:
            skipped += 1
            continue
        for row in csv.reader(open(os.path.join(BANK, fn))):
            if len(row) < 6:
                continue
            try:
                secs = float(row[5])
            except ValueError:
                continue
            key = (os.path.basename(row[1].strip()), os.path.basename(row[2].strip()))
            official = bud.get(key)
            if official is None:
                continue
            if row[4].strip() not in ("sat", "unsat"):
                # Unsolved: flag if it never got most of its official time.
                if secs < official * 0.9:
                    undermeasured.append((bench, secs, official, key[1]))
                continue
            checked += 1
            # 0.5 s slack absorbs harness timing jitter, not a real overrun.
            if secs > official + 0.5:
                violations.append((bench, row[4].strip(), secs, official, key[1]))
    for v in violations:
        print(f"OVER-BUDGET CREDIT {v[0]}: {v[1]} at {v[2]}s but budget {v[3]}s — {v[4]}")
    print(
        f"checked {checked} credited row(s); {len(violations)} over budget; "
        f"{skipped} bank file(s) had no matching instances.csv"
    )
    if undermeasured:
        per = {}
        for b, _s, _o, _v in undermeasured:
            per[b] = per.get(b, 0) + 1
        print(
            f"NOTE {len(undermeasured)} unsolved row(s) were measured with <90% of their "
            "official budget (score 0 either way; re-measure candidates): "
            + ", ".join(f"{b}={n}" for b, n in sorted(per.items()))
        )
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
