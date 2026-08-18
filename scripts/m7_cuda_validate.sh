#!/usr/bin/env bash
# M7 CUDA f64-GPU CROWN — validation + A/B runbook (run on a real NVIDIA box).
#
# Purpose: take M7 from "typechecks on Mac" to "validated, zero-wrong-verdict, and
# measured on the deep-resnet scored rows". The MOAT firewall (§3) is a HARD GATE:
# if it finds a single verdict flip or a non-conservative bound, the script ABORTS
# and M7 must NOT be enabled in any verdict-emitting binary.
#
# Prereqs: NVIDIA GPU (A100/GB10/Grace-class ideal), CUDA 13 runtime, this repo.
# Usage:   bash scripts/m7_cuda_validate.sh 2>&1 | tee m7_validation.log
set -uo pipefail

# This runbook deliberately builds and executes long-lived GPU jobs. Re-enter
# it through the installed host guard before touching Cargo or CUDA so every
# invocation gets the 160-GiB address-space ceiling, host OOM backstop, lazy CUDA
# loading, bounded build/test parallelism, and the cross-agent GPU lock. The
# marker requests one re-entry, but is never accepted as proof of containment.
# The child must independently observe the exact validated RLIMIT_AS and the
# ny-build systemd slice. Missing/forged guard infrastructure is a hard failure:
# an uncontained validation is not evidence.
if ! command -v ny-safe-gpu-run >/dev/null 2>&1; then
  echo "ERROR: ny-safe-gpu-run is required for M7 validation; refusing an unguarded GPU run." >&2
  exit 2
fi
current_vmem_soft_kib="$(builtin ulimit -Sv)"
current_vmem_hard_kib="$(builtin ulimit -Hv)"
guard_attested=0
if [ "${current_vmem_soft_kib}" = "167772160" ] \
  && [ "${current_vmem_hard_kib}" = "167772160" ] \
  && grep -q '/ny-build.slice/' /proc/self/cgroup 2>/dev/null; then
  guard_attested=1
fi
if [ "${guard_attested}" != "1" ]; then
  if [ "${NY_M7_SAFE_GPU_WRAPPED:-0}" = "1" ]; then
    echo "ERROR: NY_M7_SAFE_GPU_WRAPPED was set without the required 160-GiB/slice attestation." >&2
    exit 2
  fi
  export NY_M7_SAFE_GPU_WRAPPED=1
  exec ny-safe-gpu-run bash "$(realpath "$0")" "$@"
fi

cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
FAIL=0
say() { printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }
ok()  { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
bad() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=1; }

# ─────────────────────────────────────────────────────────────────────────────
say "1. PREFLIGHT — CUDA present"
if command -v nvidia-smi >/dev/null 2>&1 \
  && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
  ok "nvidia-smi"
else
  echo "  nvidia-smi unavailable or NVML failed; deferring to the Driver API device tests"
fi
shopt -s nullglob
cublas_candidates=(
  /usr/lib/*-linux-gnu/libcublas.so*
  /usr/local/cuda*/lib64/libcublas.so*
  /usr/local/cuda*/targets/*/lib/libcublas.so*
)
shopt -u nullglob
if [ "${#cublas_candidates[@]}" -gt 0 ]; then
  printf '%s\n' "${cublas_candidates[0]}"
  ok "libcublas found"
else
  echo "  (libcublas via dlopen at runtime; ok if driver present)"
fi

# ─────────────────────────────────────────────────────────────────────────────
say "2. BUILD --features cuda + DEVICE TESTS"
# Exercise the actual submission builder rather than an almost-equivalent raw
# Cargo command. It enforces the mip,cuda tier and configures this host's
# relocated OpenSSL plus any required AArch64 compiler feature flags.
if NY_ALLOW_DEGRADED_BUILD=0 NY_REQUIRE_MIP=1 \
  vnncomp_scripts/build_submission_binary.sh; then
  ok "release build (mip,cuda)"
else
  bad "build failed"
  exit 1
fi
# `ny-cuda` no longer has an ignored hardware tier. Its explicit admission
# seam hard-fails when this selected NVIDIA validation host cannot qualify, so
# one Cargo invocation covers both hermetic contracts and live device probes.
# The `cuda` feature lives on ny-cli (`cuda = ["dep:ny-cuda"]`); the ny-cuda
# crate itself intentionally has no feature by that name.
cargo test --locked --release -p ny-cuda -- --nocapture \
  2>&1 | tee /tmp/m7_devtests.log | grep -E "test result|device|FAILED"
device_test_status="${PIPESTATUS[0]}"
if [ "${device_test_status}" -ne 0 ] || ! grep -q "test result: ok" /tmp/m7_devtests.log; then
  bad "ny-cuda device tests did not all pass (cargo status ${device_test_status})"
elif ! grep -q "Sgemm+Dgemm known-answer probes BIT-EXACT" /tmp/m7_devtests.log; then
  bad "ny-cuda device tests did not exercise the on-device IEEE known-answer path"
elif ! grep -Eq \
  'host_ptr_zero_copy = true \(pageable path EXERCISED\)|deadline_f64_transport=explicit-device-copy \(device-copy path EXERCISED\)' \
  /tmp/m7_devtests.log; then
  bad "ny-cuda device tests did not exercise the f64 GEMM device path"
else
  ok "ny-cuda device tests (IEEE known-answer + f64 GEMM path exercised)"
fi

BIN="$REPO/target/release/ny"
BM="benchmarks/vnncomp2025/benchmarks"

# ─────────────────────────────────────────────────────────────────────────────
say "3. MOAT FIREWALL — differential M7 vs trusted CPU on KNOWN-answer rows (HARD GATE)"
# For each row with a known ground-truth verdict in reports/measured, run TWICE:
#   arm CPU  : NY_NO_CUDA=1        (trusted faer f64 CPU engine — today's proven path)
#   arm M7   : cuda engine live    (sound_f64_gemm seam auto-installed by --features cuda)
# Assert: M7 verdict == CPU verdict == known ground truth. ANY flip => ABORT.
# Corpus: sample of decided (unsat + sat) rows across categories that exercise the
# f64 A·W seam (deep conv nets). Extend CORPUS freely; more coverage = stronger moat.
CORPUS=$(
  for cat in cifar100_2024 tinyimagenet_2024 collins_rul_cnn_2022 metaroom_2023; do
    awk -F, -v c="$cat" 'NR>1 && ($5=="unsat"||$5=="sat"){print c","$2","$3","$5}' "reports/measured/$cat.csv" 2>/dev/null | head -8
  done
)
NCHK=0; NFLIP=0; NINCONCLUSIVE=0
while IFS=, read -r cat onnx vn truth; do
  [ -z "${cat:-}" ] && continue
  o="$BM/$cat/$onnx"; v="$BM/$cat/$vn"
  [ -f "$o" ] || o="$BM/$cat/$(dirname "$onnx")/$(basename "$onnx")"
  [ -f "$v" ] && [ -f "$o" ] || { [ -f "$o.gz" ] && gunzip -kf "$o.gz"; [ -f "$v.gz" ] && gunzip -kf "$v.gz"; }
  bud=$(grep -F "$(basename "$vn")" "$BM/$cat/instances.csv" 2>/dev/null | head -1 | awk -F, '{print $3}' | tr -dc 0-9); [ -z "$bud" ] && bud=100
  NY_NO_CUDA=1 OMP_NUM_THREADS=1 "$BIN" vnncomp v1 "$cat" "$o" "$v" /tmp/cpu.txt "$bud" >/dev/null 2>&1; vcpu=$(head -1 /tmp/cpu.txt 2>/dev/null)
  OMP_NUM_THREADS=1          "$BIN" vnncomp v1 "$cat" "$o" "$v" /tmp/m7.txt  "$bud" >/dev/null 2>&1; vm7=$(head -1 /tmp/m7.txt 2>/dev/null)
  NCHK=$((NCHK+1))
  if ! { [ "$vm7" = sat ] || [ "$vm7" = unsat ]; } \
    || ! { [ "$vcpu" = sat ] || [ "$vcpu" = unsat ]; }; then
    bad "MOAT INCONCLUSIVE $cat/$(basename "$vn"): M7=$vm7 CPU=$vcpu truth=$truth"
    NINCONCLUSIVE=$((NINCONCLUSIVE+1))
  elif [ "$vm7" != "$truth" ]; then bad "MOAT VIOLATION $cat/$(basename "$vn"): M7=$vm7 truth=$truth"; NFLIP=$((NFLIP+1))
  elif [ "$vm7" != "$vcpu" ]; then bad "M7/CPU DISAGREE $cat/$(basename "$vn"): M7=$vm7 CPU=$vcpu"; NFLIP=$((NFLIP+1))
  else printf '  ok  %s/%s  M7=%s CPU=%s truth=%s\n' "$cat" "$(basename "$vn")" "$vm7" "$vcpu" "$truth"; fi
done <<< "$CORPUS"
if [ "$NFLIP" -eq 0 ] && [ "$NINCONCLUSIVE" -eq 0 ] && [ "$NCHK" -gt 0 ]; then
  ok "MOAT firewall: $NCHK decided rows, 0 flips/violations"
else
  bad "MOAT firewall: $NFLIP violations, $NINCONCLUSIVE inconclusive in $NCHK rows"
fi

if [ "$FAIL" -ne 0 ]; then
  say "ABORT — a gate failed. M7 must NOT be enabled in a verdict-emitting binary."
  say "Do not proceed to §4. Investigate the failures above (log: m7_validation.log)."
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
say "4. THROUGHPUT A/B — does M7 convert the deep-resnet timeout rows? (only runs if §3 clean)"
# The payoff test: cifar100/tinyimagenet rows NY times out on today. M7 arm enables
# the resident sound CROWN route (NY_CUDA_CROWN) + f64-GEMM seam at the official budget.
OUT=/tmp/m7_ab.csv; echo "cat,vnnlib,arm,verdict,wall_s" > "$OUT"
for cat in cifar100_2024 tinyimagenet_2024; do
  awk -F, 'NR>1 && $5=="timeout"{print $2","$3}' "reports/measured/$cat.csv" 2>/dev/null | head -10 | while IFS=, read -r onnx vn; do
    o="$BM/$cat/$onnx"; v="$BM/$cat/$vn"; [ -f "$o" ] || { [ -f "$o.gz" ] && gunzip -kf "$o.gz"; }; [ -f "$v" ] || { [ -f "$v.gz" ] && gunzip -kf "$v.gz"; }
    bud=$(grep -F "$(basename "$vn")" "$BM/$cat/instances.csv" 2>/dev/null | head -1 | awk -F, '{print $3}' | tr -dc 0-9); [ -z "$bud" ] && bud=100
    for arm in "base:NY_NO_CUDA=1" "m7:NY_CUDA_CROWN=1"; do
      lbl=${arm%%:*}; env=${arm#*:}
      s=$(date +%s.%N)
      env "$env" OMP_NUM_THREADS=1 "$BIN" vnncomp v1 "$cat" "$o" "$v" /tmp/ab.txt "$bud" >/dev/null 2>&1
      e=$(date +%s.%N)
      echo "$cat,$(basename "$vn"),$lbl,$(head -1 /tmp/ab.txt),$(awk "BEGIN{printf \"%.1f\",$e-$s}")" >> "$OUT"
    done
    r=$(basename "$vn"); echo "  $cat/$r base=$(awk -F, -v r="$r" '$2==r&&$3=="base"{print $4}' "$OUT") m7=$(awk -F, -v r="$r" '$2==r&&$3=="m7"{print $4}' "$OUT")"
  done
done
CONV=$(awk -F, 'NR>1 && $3=="m7" && ($4=="unsat"||$4=="sat")' "$OUT" | wc -l | tr -d ' ')
say "RESULT: M7 converted $CONV previously-timeout deep-resnet rows. Full table: $OUT"
echo "If CONV>0 with §3 clean: re-run the MOAT firewall on those specific rows, ORT/CPU-cross-confirm,"
echo "then bank to reports/measured + scorecard. If CONV=0: M7 is sound but not yet net-faster —"
echo "profile the serial-Dgemm path and wire cublasDgemmStridedBatched (M7 remaining perf lever)."
