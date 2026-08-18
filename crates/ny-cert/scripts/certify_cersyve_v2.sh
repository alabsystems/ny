#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Cersyve v2: certificate-backed one-step verdicts for the 5 safe-forever
# finetune systems (docs/MEASURED_CERSYVE_SAFE_FOREVER.md, roadmap item 11).
#
# For each of the 10 con/inv queries this runs certify_onnx's DAG-aware exact
# pipeline (NYCERT_CONJ=1): a complete exact branch-and-bound over the vnnlib
# box where every leaf refutes one conjunct of the unsafe region
# {Y_0 <= 0 AND Y_1 >= 0} with an exact-rational Farkas certificate. Each
# leaf_<id>.farkas.json is self-checked by ny-cert's in-tree mirror and emitted
# as Clean external-certificate JSON. tree.json is NY's orchestration manifest,
# not a Clean external certificate.
#
# It first runs the exact-forward PARITY gate (loader correctness) per net,
# then certifies, then — if CLEAN_DIR is set — asks Clean's exact pinned CLI to
# re-check the leaf-local scalar contradictions. Clean does not consume
# tree.json, so it does NOT check tree coverage or source-target composition.
# No Clean source is copied.
#
# Usage:
#   BENCH=/path/to/vnncomp2025_benchmarks/cersyve \
#   [CLEAN_DIR=/path/to/clean] \
#   ./certify_cersyve_v2.sh <out_dir>
set -euo pipefail

# Certifying run: a stale NYCERT_CONJ_SCREEN=1 export would turn every certify
# call into an f64-screen dry run with no certificates.
unset NYCERT_CONJ_SCREEN

NY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCH="${BENCH:?set BENCH to the cersyve benchmark dir (onnx/ + vnnlib/)}"
OUT="${1:?usage: certify_cersyve_v2.sh <out_dir>}"
mkdir -p "$OUT"

BIN="$NY_DIR/target/release/certify_onnx"
[[ -x "$BIN" ]] || (cd "$NY_DIR" && cargo build --release -p ny-cert --bin certify_onnx)

SYSTEMS=(double_integrator lane_keep pendulum point_mass unicycle)

echo "=== Parity gate (exact loader vs f32 ONNX forward) ==="
for s in "${SYSTEMS[@]}"; do for l in con inv; do
  echo -n "${s}_${l}: "
  NYCERT_PARITY=1 "$BIN" "$BENCH/onnx/${s}_finetune_${l}.onnx" \
    "$BENCH/vnnlib/prop_${s}.vnnlib" /dev/null 2>/dev/null | grep PARITY_OK
done; done

echo "=== Certify (exact BaB, per-leaf Farkas certs, self-checked) ==="
for s in "${SYSTEMS[@]}"; do for l in con inv; do
  echo "--- ${s}_${l}"
  # NYCERT_INTERM_ROUND=1: sound outward 53-bit dyadic rounding of intermediate
  # CROWN bounds — caps bignum growth on the deep inv nets (see crown_deep.rs).
  # NYCERT_JOBS: concurrent arena-isolated exact leaf workers (throughput only).
  NYCERT_CONJ=1 NYCERT_TIGHT=1 NYCERT_INTERM_ROUND=1 \
  NYCERT_JOBS="${NYCERT_JOBS:-8}" NYCERT_MAX_LEAVES=200000 "$BIN" \
    "$BENCH/onnx/${s}_finetune_${l}.onnx" "$BENCH/vnnlib/prop_${s}.vnnlib" \
    "$OUT/${s}_${l}" 2>/dev/null | grep -E "CONJ_RESULT|CONJ_FAILED"
done; done

if [[ -n "${CLEAN_DIR:-}" ]]; then
  echo "=== Re-check leaf-local contradictions with Clean's external-cert verifier ==="
  CLEAN_WORK="$(mktemp -d)"
  trap 'rm -rf "$CLEAN_WORK"' EXIT
  source "$(dirname "${BASH_SOURCE[0]}")/_clean_pinned.sh"
  prepare_pinned_clean "$CLEAN_WORK/clean"
  for s in "${SYSTEMS[@]}"; do for l in con inv; do
    BATCH="$CLEAN_WORK/${s}_${l}.json"
    python3 - "$OUT/${s}_${l}" "$BATCH" <<'PY'
import json
import sys
from pathlib import Path

files = sorted(Path(sys.argv[1]).glob("leaf_*.farkas.json"))
if not files:
    raise SystemExit(f"no leaf certificates found under {sys.argv[1]}")
certificates = [json.loads(path.read_text(encoding="utf-8")) for path in files]
Path(sys.argv[2]).write_text(json.dumps(certificates) + "\n", encoding="utf-8")
print(f"Prepared {len(certificates)} certificates from {sys.argv[1]}")
PY
    "$CLEAN_BIN" cert verify-external-batch "$BATCH" --threads 1
  done; done
fi
echo "Cersyve v2 certification complete."
