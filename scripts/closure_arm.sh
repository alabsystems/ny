#!/usr/bin/env bash
# #closure-adaptive measurement runner. One arm over a list of rows, official
# 100 s budget, scored invocation. Re-verifies binary provenance before EVERY
# row (this tree is shared and the binary was swapped mid-study twice).
#
#   scripts/closure_arm.sh <arm-tag> <expected-exe-sha> <row>...
#
# A row is `<net>:<vnnlib-basename-fragment>`, e.g. `medium:idx_4169_sidx_9758`.
# Environment for the arm is inherited, so the caller sets NY_MARGIN_ROW_*.
set -uo pipefail

TAG="$1"; shift
WANT_EXE="$1"; shift
OUT="${NY_CLOSURE_OUT:?set NY_CLOSURE_OUT}"
B=benchmarks/vnncomp2025/benchmarks/cifar100_2024
mkdir -p "$OUT"

for row in "$@"; do
  net="${row%%:*}"; frag="${row#*:}"
  case "$net" in
    medium) onnx="$B/onnx/CIFAR100_resnet_medium.onnx"; pre="CIFAR100_resnet_medium" ;;
    large)  onnx="$B/onnx/CIFAR100_resnet_large.onnx";  pre="CIFAR100_resnet_large" ;;
    *) echo "unknown net $net"; exit 2 ;;
  esac
  vnnlib="$B/vnnlib/${pre}_prop_${frag}_eps_0.0039.vnnlib"
  [ -f "$vnnlib" ] || { echo "MISSING $vnnlib"; exit 2; }

  # R8 binary provenance, re-checked per row.
  got=$(./target/release/ny --build-id | tr ' ' '\n' | grep '^exe=' | cut -d= -f2)
  if [ "$got" != "$WANT_EXE" ]; then
    echo "BINARY SWAPPED before $frag: exe=$got want=$WANT_EXE — ABORT"
    exit 3
  fi

  log="$OUT/${TAG}__${frag}.log"
  res="$OUT/${TAG}__${frag}.txt"
  t0=$(date +%s)
  ./target/release/ny vnncomp v1 cifar100_2024 "$onnx" "$vnnlib" "$res" 100 \
      --configs-dir configs > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  # Verify the log's own INSTANCE line names the row we asked for (the
  # wrong-ROW trap: a sibling index reads as a conversion).
  if ! grep -q "$frag" "$log"; then
    echo "  !! $frag: log does not name this instance — VOID"
  fi
  printf '%-10s %-22s %-8s rc=%s %ss\n' "$TAG" "$frag" "$(cat "$res" 2>/dev/null | head -1)" "$rc" "$((t1-t0))"
done
