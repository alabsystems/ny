#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Offline smoke test of the scorecard validation plumbing.
#
# This script does NOT run the verifier and does NOT recompute any verdict.
# The authoritative offline recomputation is the Rust regression test
# `offline_scorecard_matches_reference` (crate ny-onnx), which re-derives every
# verdict in `tests/fixtures/offline_scorecard_reference.csv` with the real
# engine and would catch a verified<->violated flip.
#
# What this script checks is `scripts/validate_vnncomp_results.sh` itself:
#   1. The committed reference verdicts, echoed through the validator, are
#      accepted (agreement path works; exit 0).
#   2. A deliberately flipped verdict is flagged CRITICAL (detection path
#      works; exit 1). Without this negative control the check could never
#      fail and would certify nothing.
#
# Usage:
#   scripts/check_offline_scorecard.sh
#
# Exit codes:
#   0 = validator plumbing OK (agreement accepted, injected flip detected)
#   1 = validator plumbing broken
#   2 = missing reference/validator

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REFERENCE="${REPO_ROOT}/tests/fixtures/offline_scorecard_reference.csv"
VALIDATOR="${SCRIPT_DIR}/validate_vnncomp_results.sh"

[[ -f "$REFERENCE" ]] || { echo "ERROR: missing reference: $REFERENCE" >&2; exit 2; }
[[ -f "$VALIDATOR" ]] || { echo "ERROR: missing validator: $VALIDATOR" >&2; exit 2; }

AGREE_CSV="$(mktemp)"
FLIP_CSV="$(mktemp)"
trap 'rm -f "$AGREE_CSV" "$FLIP_CSV"' EXIT

# Build two "ny"-format CSVs (model,property,timeout,result,elapsed) from the
# committed reference: one echoing the reference verdicts verbatim, and one
# with the first verdict flipped verified<->violated.
echo "model,property,timeout,result,elapsed" > "$AGREE_CSV"
echo "model,property,timeout,result,elapsed" > "$FLIP_CSV"
flipped=0
while IFS=',' read -r model property result; do
    [[ -z "$model" ]] && continue
    echo "${model},${property},10,${result},0.0" >> "$AGREE_CSV"
    if [[ "$flipped" -eq 0 ]]; then
        case "$result" in
            verified) flip="violated" ;;
            violated) flip="verified" ;;
            *)        flip="$result" ;;
        esac
        if [[ "$flip" != "$result" ]]; then
            flipped=1
        fi
        echo "${model},${property},10,${flip},0.0" >> "$FLIP_CSV"
    else
        echo "${model},${property},10,${result},0.0" >> "$FLIP_CSV"
    fi
done < <(tail -n +2 "$REFERENCE")

if [[ "$flipped" -eq 0 ]]; then
    echo "ERROR: no verified/violated row in $REFERENCE to flip; cannot exercise detection." >&2
    exit 1
fi

echo "[1/2] Reference verdicts echoed through the validator (expect agreement, exit 0)..."
if ! bash "$VALIDATOR" "$AGREE_CSV" "$REFERENCE" --format simple; then
    echo "FAIL: validator rejected the committed reference against itself." >&2
    exit 1
fi

echo ""
echo "[2/2] Injected verified<->violated flip (expect CRITICAL, exit 1)..."
rc=0
bash "$VALIDATOR" "$FLIP_CSV" "$REFERENCE" --format simple || rc=$?
if [[ "$rc" -ne 1 ]]; then
    echo "FAIL: validator did not flag the injected flip as CRITICAL (exit $rc)." >&2
    exit 1
fi

echo ""
echo "OK: validator plumbing detects a verified<->violated flip."
echo "NOTE: no verdicts were recomputed here; the authoritative offline check is"
echo "      the Rust test offline_scorecard_matches_reference (crate ny-onnx)."
