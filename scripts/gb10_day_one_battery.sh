#!/bin/bash
# GB10 DAY-ONE BATTERY (#gb10-day-one) — turnkey measurement sequence for the
# CUDA competition host. Written 2026-08-02 on the Metal host, where every
# experiment below is blocked ONLY by hardware class (all machinery landed and
# adversarially verified; see docs/GB10_DAY_ONE_BATTERY_2026-08-02.md for the
# per-experiment predictions, gates, and points model).
#
# Usage: bash scripts/gb10_day_one_battery.sh [outdir]
# Requires: --features cuda,mip release build; benchmarks fetched; quiet box.
set -uo pipefail
NY_ROOT=${NY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${1:-$NY_ROOT/reports/measured-runs/gb10-day-one-$(date -u +%Y%m%dT%H%M%SZ)}
BIN=${BIN:-$NY_ROOT/target/release/ny}
mkdir -p "$OUT"
cd "$NY_ROOT"

echo "== 0. Provenance =="
git rev-parse --short HEAD | tee "$OUT/rev.txt"
"$BIN" --build-info | tee -a "$OUT/rev.txt"   # must show cuda=on mip=on
nvidia-smi -L | tee -a "$OUT/rev.txt" 2>/dev/null || echo "WARN: no nvidia-smi"

echo "== 1. Rate probes (the whole battery keys off these) =="
# ny vnncomp plan prints: fl rate probe + rule-6 window decision + provenance.
B=$NY_ROOT/benchmarks/vnncomp2025/benchmarks/cifar100_2024
"$BIN" vnncomp plan cifar100_2024 \
  "$B/onnx/CIFAR100_resnet_medium.onnx" \
  "$B/vnnlib/CIFAR100_resnet_medium_prop_idx_7500_sidx_40_eps_0.0039.vnnlib" 100 \
  --configs-dir ./configs | tee "$OUT/plan_cifar100.txt"
# GATE 1: rule 6 widens (needs measured rate >= 17.48 GMAC/s; cuBLAS Dgemm or
# Sgemm-with-seam should clear it). If it declines here, STOP and investigate
# the probe before burning sweep time.

echo "== 2. FL f64-vs-f32 A/B at official 100s (census target rows) =="
# Prediction (FL_FIRST_MEASUREMENT + CONVWALL_PANEL_VERDICT): f32 seam puts FL
# at ~5-15s; BaB starts from margins -16..-20 (vs -77). Rows ordered by census.
for row in prop_idx_7500_sidx_40 prop_idx_3343_sidx_1406 prop_idx_9502_sidx_7197 \
           prop_idx_815_sidx_1902 prop_idx_7641_sidx_1041; do
  for arm in f64 f32; do
    env_extra=""
    # DISARMED 2026-08-03 pending S1 rung-1 conformance (#fl-value-gpu-tier).
    # This was the ONLY place in the repo that armed NY_FORWARD_LINEAR_F32, and
    # arming it routes forward-linear VALUE GEMMs through FlValueGemmDevice — a
    # live wgpu f32 engine registered in the production binary (main.rs:410)
    # whose results feed published bounds via certified_f32_gemm_deadline_gpu
    # (forward_linear/image.rs:465).
    #
    # The call site charges gamma_{K+4}^f32 * S for those values, which assumes
    # u = 2^-24 on the device. Nothing verifies that assumption on the actual
    # silicon: FlValueGemmDevice runs no adapter conformance self-test, its
    # guard returns Err and lets the caller fall silently to CPU (per-call
    # recovery, the exact shape ladder.rs forbids), and it publishes alone
    # rather than as a differential union. Structurally the charge is right —
    # the chunker never splits the contraction dimension — but "right by
    # construction, unchecked on hardware" is not the bar for a verdict path.
    #
    # Re-arm only once the device passes the S1 ladder's rung-1 self-test.
    # Until then the f32 arm measures nothing we would be willing to bank.
    if [ "$arm" = f32 ]; then
      echo "  (f32 arm skipped: FlValueGemmDevice awaits S1 rung-1 conformance)"
      continue
    fi
    env NY_PHASE_TELEMETRY=1 $env_extra "$BIN" -v vnncomp v1 cifar100_2024 \
      "$B/onnx/CIFAR100_resnet_medium.onnx" \
      "$B/vnnlib/CIFAR100_resnet_medium_${row}_eps_0.0039.vnnlib" \
      "$OUT/c100_${row}.${arm}.txt" 100 --configs-dir ./configs \
      >/dev/null 2>"$OUT/c100_${row}.${arm}.stderr"
    echo "cifar100 $row/$arm: $(head -1 "$OUT/c100_${row}.${arm}.txt" 2>/dev/null || echo none)"
  done
done
# GATE 2: >=1 unsat => the cifar100 block is opening; run the full winnable-60
# with the winning arm (step 5). f32-vs-f64 margin delta must stay ~<4/row
# (the measured seam width cost) — larger means an f32 soundness/width surprise.

echo "== 3. tinyimagenet block at official 100s (throughput class test) =="
T=$NY_ROOT/benchmarks/vnncomp2025/benchmarks/tinyimagenet_2024
# The 3 rows that convert at 9x on an M5 Max CPU (easiest third exemplars) +
# the 2 that did not (tail sentinels).
for row in prop_idx_8472_sidx_7499 prop_idx_3677_sidx_3123 prop_idx_9586_sidx_1970 \
           prop_idx_8651_sidx_6574 prop_idx_7538_sidx_5076; do
  env NY_PHASE_TELEMETRY=1 "$BIN" -v vnncomp v1 tinyimagenet_2024 \
    "$T/onnx/TinyImageNet_resnet_medium.onnx" \
    "$T/vnnlib/TinyImageNet_resnet_medium_${row}_eps_0.0039.vnnlib" \
    "$OUT/tin_${row}.txt" 100 --configs-dir ./configs \
    >/dev/null 2>"$OUT/tin_${row}.stderr"
  echo "tinyimagenet $row: $(head -1 "$OUT/tin_${row}.txt" 2>/dev/null || echo none)"
done
# GATE 3: the 3 easy-third rows converting at 100s here => the block's easiest
# third (~23 rows, ~230 normalized points gross) banks on a full category sweep.

echo "== 4. nn4sys dual pool (leaf-throughput class test) =="
# Metal: 22 leaves/s vs needed 120-200 (SATURATION_ESCAPE doc). GPU BaB is the
# hypothesis. 240-clause first (cheapest discriminator).
N=$NY_ROOT/benchmarks/vnncomp2025/benchmarks/nn4sys_2023
if [ -d "$N" ]; then
  grep -m2 "dual" "$N/instances.csv" 2>/dev/null | while IFS=, read -r onnx vnnlib budget; do
    env NY_PHASE_TELEMETRY=1 "$BIN" -v vnncomp v1 nn4sys_2023 \
      "$N/$onnx" "$N/$vnnlib" "$OUT/nn4_$(basename "$vnnlib").txt" "${budget%.*}" \
      --configs-dir ./configs >/dev/null 2>"$OUT/nn4_$(basename "$vnnlib").stderr"
    echo "nn4sys $(basename "$vnnlib"): $(head -1 "$OUT/nn4_$(basename "$vnnlib").txt" 2>/dev/null || echo none)"
  done
fi

echo "== 5. On any gate pass: full category sweeps (points banking) =="
echo "cifar100: ARMS=baseline BUDGET=100 bash scripts/cifar100_winnable_official_sweep.sh <official-results-dir> $OUT/c100-sweep"
echo "tinyimagenet/nn4sys: use the categories' official instance lists at official budgets; MOAT-check everything."
echo "Battery complete. Score everything with: ny benchmarks score (bank-linter on)."
