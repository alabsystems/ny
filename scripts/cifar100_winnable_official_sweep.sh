#!/usr/bin/env bash
# Official-budget sweep of the cifar100 winnable-60 with the root-alpha margin
# fix armed. THIS IS THE MEASUREMENT THAT DECIDES WHETHER THE FIX SCORES.
#
# Context: docs/CIFAR100_ROOT_ALPHA_DEGRADES_SPEC_BOUNDS_2026-07-26.md
#   - The root alpha ascent maximized SUM of raw logit lower bounds, not the 99
#     margin specs (§1-2). NY_ROOT_ALPHA_MARGIN=1 ranks iterates by the margin
#     hinge instead; 8649e38e fixed the wiring that made it inert on this path.
#   - On a CUDA-less Mac that lifted mean root-verified 69.9 -> 78.1 of 99 across
#     32 rows and produced the first-ever winnable-60 conversion (prop_idx_7500)
#     — but only at ~15x the scored budget, so it does NOT establish score.
#   - Bound VALUES are hardware-independent; wall times are not. Hence this
#     script must run on GB10-class hardware at the official 100s budget.
#
# Usage:
#   scripts/cifar100_winnable_official_sweep.sh <official-results-dir> [outdir]
#
# <official-results-dir> is a checkout of https://github.com/VNN-COMP/vnncomp2025_results
# (needs <tool>/2025_cifar100_2024/results.csv).
#
# Emits per row: verdict, wall, root-verified, binding bound — plus a MOAT CHECK
# that fails loudly on any verdict contradicting the official field.
set -uo pipefail

NY_ROOT=${NY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
RESULTS=${1:?usage: $0 <official-results-dir> [outdir]}
OUT=${2:-$NY_ROOT/reports/measured-runs/cifar100-winnable-official-$(date -u +%Y%m%dT%H%M%SZ)}
BENCH=$NY_ROOT/benchmarks/vnncomp2025/benchmarks/cifar100_2024
BIN=${BIN:-$NY_ROOT/target/release/ny}
BUDGET=${BUDGET:-100}          # the OFFICIAL per-instance budget. Do not raise.
ARMS=${ARMS:-"baseline margin"}

mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "no binary at $BIN" >&2; exit 1; }
[ -d "$BENCH/vnnlib" ] || { echo "no benchmark at $BENCH" >&2; exit 1; }

echo "ny      : $(git -C "$NY_ROOT" rev-parse --short HEAD) ($(git -C "$NY_ROOT" status --porcelain | wc -l | tr -d ' ') dirty)"
echo "binary  : $(shasum -a 256 "$BIN" | cut -c1-16)"
echo "budget  : ${BUDGET}s   arms: $ARMS"
echo "out     : $OUT"

# --- derive the winnable set from the official results (sound tools only) -----
python3 - "$RESULTS" "$NY_ROOT" "$OUT" <<'PY'
import csv, os, sys
res, ny_root, out = sys.argv[1], sys.argv[2], sys.argv[3]
TOOLS = ["alpha_beta_crown", "neuralsat", "cora", "pyrat", "nnv", "rover"]
def load(p):
    d = {}
    if not os.path.exists(p): return d
    for r in csv.reader(open(p, newline="")):
        if len(r) >= 5:
            d[(os.path.basename(r[1]), os.path.basename(r[2]))] = r[4].strip()
    return d
official = {t: load(f"{res}/{t}/2025_cifar100_2024/results.csv") for t in TOOLS}
ny = load(f"{ny_root}/reports/measured/cifar100_2024.csv")
inst = list(official["alpha_beta_crown"])
# A tool contradicting a witness-backed sat is emitting wrong verdicts; it must
# not define the target set (2025: nnv contradicts 19).
falsified = {i for i in inst if any(official[t].get(i) == "sat" for t in TOOLS)}
sound = [t for t in TOOLS
         if not any(official[t].get(i) == "unsat" for i in falsified)]
rows = [i for i in inst
        if any(official[t].get(i) == "unsat" for t in sound) and ny.get(i) != "unsat"]
# Order matters for a multi-hour sweep: MEDIUM before LARGE (large roots are an
# order of magnitude looser -- 0/99 verified vs 96/99, see the doc §17), and
# within each, easiest-for-abc first. That way the run yields signal early and
# can be stopped at any point with the most informative rows already done.
_abc = {}
for r in csv.reader(open(f"{res}/alpha_beta_crown/2025_cifar100_2024/results.csv", newline="")):
    if len(r) >= 6:
        try: _abc[(os.path.basename(r[1]), os.path.basename(r[2]))] = float(r[5])
        except ValueError: pass
rows.sort(key=lambda i: ("large" in i[0], _abc.get(i, 1e9)))
with open(f"{out}/targets.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    for onnx, vnnlib in rows:
        gt = "unsat" if any(official[t].get((onnx, vnnlib)) == "unsat" for t in sound) else "?"
        w.writerow([onnx, vnnlib, gt])
print(f"winnable target rows: {len(rows)}  (sound tools: {','.join(sound)})")
PY

# --- run ---------------------------------------------------------------------
: > "$OUT/results.tsv"
while IFS=, read -r onnx vnnlib gt; do
  [ -f "$BENCH/vnnlib/$vnnlib" ] || { echo "MISSING $vnnlib" >&2; continue; }
  for arm in $ARMS; do
    log="$OUT/${vnnlib%%_eps*}.$arm.stderr"
    # Real prefix assignments only. NEVER pass multi-var env through an unquoted
    # shell parameter: zsh does not word-split, so the first gate silently gets a
    # malformed value and fails closed. That cost this campaign a false finding.
    # NOTE: coreutils `timeout` does not exist on stock macOS (defect class D3:
    # it silently 127s every row and the sweep "completes" with empty verdicts).
    # perl alarm+exec is the portable hang guard; ny self-limits at $BUDGET.
    if [ "$arm" = margin ]; then
      NY_ROOT_ALPHA_MARGIN=1 NY_PHASE_TELEMETRY=1 \
      perl -e 'alarm shift @ARGV; exec @ARGV' $((BUDGET + 60)) "$BIN" -v vnncomp v1 cifar100_2024 \
        "$BENCH/onnx/$onnx" "$BENCH/vnnlib/$vnnlib" "$OUT/$vnnlib.$arm.txt" "$BUDGET" \
        --configs-dir "$NY_ROOT/configs" >/dev/null 2>"$log"
    elif [ "$arm" = pernodefloor ]; then
      # PREALPHA_PER_NODE_BUDGET_LEVER experiment (b): raise the CROWN-IBP
      # collection's per-node floor so large-dim nodes (Conv_11 class) keep
      # their genuine CROWN intermediates instead of PerNodeDeadlineExceeded
      # -> IBP fallback. budget_policy.rs:89-116.
      NY_PER_NODE_FLOOR_SECS=12 NY_PHASE_TELEMETRY=1 \
      perl -e 'alarm shift @ARGV; exec @ARGV' $((BUDGET + 60)) "$BIN" -v vnncomp v1 cifar100_2024 \
        "$BENCH/onnx/$onnx" "$BENCH/vnnlib/$vnnlib" "$OUT/$vnnlib.$arm.txt" "$BUDGET" \
        --configs-dir "$NY_ROOT/configs" >/dev/null 2>"$log"
    else
      NY_PHASE_TELEMETRY=1 \
      perl -e 'alarm shift @ARGV; exec @ARGV' $((BUDGET + 60)) "$BIN" -v vnncomp v1 cifar100_2024 \
        "$BENCH/onnx/$onnx" "$BENCH/vnnlib/$vnnlib" "$OUT/$vnnlib.$arm.txt" "$BUDGET" \
        --configs-dir "$NY_ROOT/configs" >/dev/null 2>"$log"
    fi
    python3 - "$log" "$vnnlib" "$arm" "$gt" "$(cat "$OUT/$vnnlib.$arm.txt" 2>/dev/null)" \
      >> "$OUT/results.tsv" <<'PY'
import re, sys
strip = re.compile(r'\x1b\[[0-9;]*m')
t = [strip.sub('', l) for l in open(sys.argv[1], errors="replace")]
lo = [float(m.group(1)) for l in t if (m := re.search(r'obj\[\d+\]: bounds=\[(-?[0-9.eE+-]+),', l))]
ver = sum(1 for l in t if 'verified=true' in l)
# Null-run guard: on the margin arm the gate MUST report PRESENT. If it does not,
# the arm did not exercise the feature and its number is meaningless.
gate = "n/a"
for l in t:
    if 'gate ARMED' in l:
        gate = "PRESENT" if "PRESENT" in l else "ABSENT"
print(f"{sys.argv[2]}\t{sys.argv[3]}\t{sys.argv[5]}\t{sys.argv[4]}\t{ver}\t{len(lo)}\t"
      f"{min(lo) if lo else float('nan'):.4f}\t{gate}")
PY
    tail -1 "$OUT/results.tsv"
  done
done < "$OUT/targets.csv"

# --- moat check + summary ----------------------------------------------------
echo
python3 - "$OUT/results.tsv" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1]), delimiter="\t"))
bad = [r for r in rows if r[2] == "sat" and r[3] == "unsat"]
for arm in sorted({r[1] for r in rows}):
    a = [r for r in rows if r[1] == arm]
    solved = sum(1 for r in a if r[2] in ("unsat", "sat"))
    inert = sum(1 for r in a if arm == "margin" and r[7] != "PRESENT")
    # Rows that errored or timed out early can carry non-numeric placeholders
    # ("n/a"/"nan") in the root-verified column; count them as 0 verified rather
    # than aborting the whole summary (the per-row TSV keeps the raw value).
    def _rootv(r):
        try:
            return int(r[4])
        except (ValueError, IndexError):
            return 0
    print(f"{arm:<10} solved {solved}/{len(a)}   mean root-verified "
          f"{sum(_rootv(r) for r in a)/max(len(a),1):.1f}"
          + (f"   !! {inert} rows where the gate was not PRESENT" if inert else ""))
print()
if bad:
    print(f"*** MOAT BREACH: {len(bad)} rows report unsat where the official field has a "
          f"witness-backed sat ***")
    for r in bad: print("   ", r[0])
    sys.exit(2)
print("MOAT OK: no verdict contradicts the official field.")
PY
