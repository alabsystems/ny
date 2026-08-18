#!/usr/bin/env python3
"""cifar100_2024 scoreboard: ny (measured) vs the official VNN-COMP 2025 field.

Emits (a) per-tool verdict counts, (b) the winnable set (someone=unsat, ny=timeout),
(c) the leverage set (nobody unsat, ny not unsat), split medium/large.
"""
import csv
import os
import sys
from collections import Counter, defaultdict

# Official per-tool results, e.g. a checkout of
#   https://github.com/VNN-COMP/vnncomp2025_results
# (sparse: <tool>/2025_cifar100_2024/results.csv).
VR = sys.argv[1] if len(sys.argv) > 1 else "external_tools/vnncomp2025_results"
NY = sys.argv[2] if len(sys.argv) > 2 else "reports/measured/cifar100_2024.csv"
TOOLS = ["alpha_beta_crown", "neuralsat", "cora", "pyrat", "nnv", "rover"]


def key(onnx: str, vnnlib: str) -> tuple[str, str]:
    return (os.path.basename(onnx), os.path.basename(vnnlib))


def load(path: str) -> dict[tuple[str, str], tuple[str, float]]:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 5:
                continue
            onnx, vnnlib, verdict = row[1], row[2], row[4].strip()
            try:
                secs = float(row[5]) if len(row) > 5 else float("nan")
            except ValueError:
                secs = float("nan")
            out[key(onnx, vnnlib)] = (verdict, secs)
    return out


ny = load(NY)
official = {t: load(f"{VR}/{t}/2025_cifar100_2024/results.csv") for t in TOOLS}

# Restrict to the official instance set (drops ny's test_nano row).
instances = sorted(official["alpha_beta_crown"])
print(f"official instances: {len(instances)}\n")

print(f"{'tool':<20} {'unsat':>6} {'sat':>5} {'solved':>7} {'timeout':>8} {'other':>6}")
print("-" * 58)
for tool in TOOLS + ["ny"]:
    src = ny if tool == "ny" else official[tool]
    c = Counter(src.get(i, ("missing", 0))[0] for i in instances)
    solved = c["unsat"] + c["sat"]
    other = len(instances) - solved - c["timeout"]
    print(f"{tool:<20} {c['unsat']:>6} {c['sat']:>5} {solved:>7} {c['timeout']:>8} {other:>6}")

# A tool that reports `unsat` on a row another tool FALSIFIES is contradicting a
# witness-backed sat, i.e. emitting wrong verdicts (-150 each on the scored
# board). Such a tool's raw solve count is not a score, so it must not set the
# normalization baseline. In 2025 this excludes nnv, whose 190 "unsat" includes
# all 29 rows alpha-beta-CROWN falsifies.
falsified = {
    i for i in instances if any(official[t].get(i, ("", 0))[0] == "sat" for t in TOOLS)
}
contradictions = {
    t: sum(1 for i in falsified if official[t].get(i, ("", 0))[0] == "unsat") for t in TOOLS
}
sound = [t for t in TOOLS if contradictions[t] == 0]
for t in TOOLS:
    if contradictions[t]:
        print(f"  ! {t} contradicts {contradictions[t]} witness-backed sat rows — excluded from the baseline")

solved_by = {t: sum(1 for i in instances if official[t].get(i, ("", 0))[0] in ("unsat", "sat")) for t in TOOLS}
best = max(((solved_by[t], t) for t in sound), default=(0, "none"))
ny_solved = sum(1 for i in instances if ny.get(i, ("", 0))[0] in ("unsat", "sat"))
print(f"\nfield best (sound tools only): {best[1]} = {best[0]} solved; ny = {ny_solved}")
if best[0]:
    print(f"ny normalized (solve-count ratio): {100.0 * ny_solved / best[0]:.1f}")

# ---- the winnable set: some official tool proves unsat, ny does not solve ----
def net(k):
    return "large" if "large" in k[0] else "medium"


winnable = defaultdict(list)
leverage = defaultdict(list)
for i in instances:
    ny_v = ny.get(i, ("missing", 0))[0]
    # Only SOUND tools may define the winnable set — otherwise nnv's contradicted
    # unsats would inflate it from 60 to 150.
    provers = {t: official[t][i] for t in sound if official[t].get(i, ("", 0))[0] == "unsat"}
    if provers and ny_v != "unsat":
        winnable[net(i)].append((i, ny_v, provers))
    elif not provers and ny_v not in ("unsat", "sat"):
        # nobody proves it and ny does not solve it: the surpass pool
        if all(official[t].get(i, ("", 0))[0] != "sat" for t in TOOLS):
            leverage[net(i)].append((i, ny_v))

print(f"\nWINNABLE (>=1 official tool unsat, ny not unsat): "
      f"{sum(len(v) for v in winnable.values())}  "
      + "  ".join(f"{k}={len(v)}" for k, v in sorted(winnable.items())))
print(f"LEVERAGE (nobody solves, ny does not solve):       "
      f"{sum(len(v) for v in leverage.values())}  "
      + "  ".join(f"{k}={len(v)}" for k, v in sorted(leverage.items())))

# abc solve times on the winnable rows -> how "easy" they are
abc_secs = sorted(
    p["alpha_beta_crown"][1]
    for rows in winnable.values()
    for _, _, p in rows
    if "alpha_beta_crown" in p
)
if abc_secs:
    n = len(abc_secs)
    print(f"\nabc solve time on winnable rows (n={n}): "
          f"min={abc_secs[0]:.1f}s median={abc_secs[n // 2]:.1f}s max={abc_secs[-1]:.1f}s "
          f"under20s={sum(1 for s in abc_secs if s < 20)}")

out = "cifar100_winnable.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["net", "onnx", "vnnlib", "ny_verdict", "abc_verdict", "abc_secs", "other_provers"])
    for kind in ("medium", "large"):
        for (onnx, vnnlib), ny_v, provers in sorted(winnable[kind], key=lambda r: r[2].get("alpha_beta_crown", ("", 1e9))[1]):
            abc = provers.get("alpha_beta_crown")
            w.writerow([
                kind, onnx, vnnlib, ny_v,
                "unsat" if abc else official["alpha_beta_crown"].get((onnx, vnnlib), ("", 0))[0],
                f"{abc[1]:.2f}" if abc else "",
                ";".join(sorted(t for t in provers if t != "alpha_beta_crown")),
            ])
print(f"\nwrote {out}")
