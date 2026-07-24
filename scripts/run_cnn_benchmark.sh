#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# CNN benchmark runner for VNN-COMP CIFAR-10 ResNet category.
# Measures verification rate with alpha-CROWN (Patches mode) and beta-CROWN (BaB).
# Part of #3290: First CNN benchmark measurement.

set -euo pipefail

NY="${NY_BINARY:-target/release/ny}"
BENCHMARK_DIR="benchmarks/vnncomp2021/benchmarks/cifar10_resnet"
MODEL_VARIANT="${MODEL_VARIANT:-resnet_2b}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
TIMEOUT="${TIMEOUT:-60}"
METHOD="${METHOD:-alpha}"
PGD_ATTACK="${PGD_ATTACK:-0}"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Model-specific configuration
MODEL="$BENCHMARK_DIR/onnx/${MODEL_VARIANT}.onnx"
case "$MODEL_VARIANT" in
    resnet_2b) PROP_DIR="resnet2b_pgd_filtered"; EPS="0.008" ;;
    resnet_4b) PROP_DIR="resnet4b_pgd_filtered"; EPS="0.004" ;;
    *) echo "ERROR: Unknown model variant: $MODEL_VARIANT"; exit 1 ;;
esac
OUTPUT_JSON="reports/benchmarks/cifar10_${MODEL_VARIANT}_${METHOD}_$(date -u +%Y%m%d_%H%M%S).json"

echo "=== CIFAR-10 ${MODEL_VARIANT} Benchmark ==="
echo "Method: $METHOD"
echo "Timeout: ${TIMEOUT}s"
echo "Max instances: $MAX_INSTANCES"
echo "Binary: $NY"
echo "Model: $MODEL_VARIANT (eps=$EPS)"
echo "Start: $TIMESTAMP"
echo ""

if [ ! -f "$NY" ]; then
    echo "ERROR: Binary not found: $NY"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Run: benchmarks/download_benchmarks.sh"
    exit 1
fi

# Collect results
VERIFIED=0
FALSIFIED=0
UNKNOWN=0
TIMEOUT_COUNT=0
ERROR_COUNT=0
TOTAL=0
RESULTS_JSON="["

for i in $(seq 0 $((MAX_INSTANCES - 1))); do
    PROP="$BENCHMARK_DIR/vnnlib_properties_pgd_filtered/${PROP_DIR}/prop_${i}_eps_${EPS}.vnnlib"
    if [ ! -f "$PROP" ]; then
        echo "  SKIP: prop_${i} not found"
        continue
    fi

    START_NS=$(python3 --no-wrapper -c "import time; print(int(time.time_ns()))")

    if [ "$METHOD" = "beta" ]; then
        PGD_FLAGS=""
        if [ "$PGD_ATTACK" = "1" ]; then
            PGD_FLAGS="--pgd-attack"
        fi
        OUTPUT=$("$NY" beta-crown "$MODEL" \
            --property "$PROP" \
            --timeout "$TIMEOUT" \
            --branching input \
            $PGD_FLAGS \
            2>&1) || true
    else
        OUTPUT=$("$NY" verify "$MODEL" \
            --property "$PROP" \
            --method "$METHOD" \
            --timeout "$TIMEOUT" \
            --json \
            2>&1) || true
    fi

    END_NS=$(python3 --no-wrapper -c "import time; print(int(time.time_ns()))")
    ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
    ELAPSED_S=$(python3 --no-wrapper -c "print(f'{${ELAPSED_MS}/1000:.2f}')")

    # Parse status from JSON output.
    # The verify command outputs "property_status" (safe/unknown/violated) and "status" (always "verified"
    # meaning the process completed). The beta-crown command outputs plain text "Status: VERIFIED/UNKNOWN".
    # Use property_status for JSON output; fall back to text parsing for beta-crown.
    STATUS="error"
    if echo "$OUTPUT" | grep -qi '"property_status".*"safe"'; then
        STATUS="verified"
    elif echo "$OUTPUT" | grep -qi '"property_status".*"violated"'; then
        STATUS="falsified"
    elif echo "$OUTPUT" | grep -qi '"property_status".*"unknown"'; then
        STATUS="unknown"
    elif echo "$OUTPUT" | grep -qi 'Status: VERIFIED'; then
        STATUS="verified"
    elif echo "$OUTPUT" | grep -qi 'Status: UNKNOWN'; then
        STATUS="unknown"
    elif echo "$OUTPUT" | grep -qi 'PotentialViolation\|Status: FALSIFIED'; then
        STATUS="falsified"
    elif echo "$OUTPUT" | grep -qi 'TIMEOUT\|timed out'; then
        STATUS="timeout"
    fi

    echo "  prop_${i}: ${STATUS} (${ELAPSED_S}s)"

    TOTAL=$((TOTAL + 1))
    case "$STATUS" in
        verified) VERIFIED=$((VERIFIED + 1)) ;;
        falsified) FALSIFIED=$((FALSIFIED + 1)) ;;
        unknown) UNKNOWN=$((UNKNOWN + 1)) ;;
        timeout) TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1)) ;;
        *) ERROR_COUNT=$((ERROR_COUNT + 1)) ;;
    esac

    # Build JSON entry
    if [ "$TOTAL" -gt 1 ]; then
        RESULTS_JSON="${RESULTS_JSON},"
    fi
    RESULTS_JSON="${RESULTS_JSON}{\"instance\":\"prop_${i}\",\"status\":\"${STATUS}\",\"time_s\":${ELAPSED_S}}"
done

RESULTS_JSON="${RESULTS_JSON}]"

echo ""
echo "=== SUMMARY ==="
echo "Total: $TOTAL"
echo "Verified: $VERIFIED ($((VERIFIED * 100 / TOTAL))%)"
echo "Falsified: $FALSIFIED ($((FALSIFIED * 100 / TOTAL))%)"
echo "Unknown: $UNKNOWN ($((UNKNOWN * 100 / TOTAL))%)"
echo "Timeout: $TIMEOUT_COUNT ($((TIMEOUT_COUNT * 100 / TOTAL))%)"
echo "Error: $ERROR_COUNT ($((ERROR_COUNT * 100 / TOTAL))%)"
echo "Solved: $((VERIFIED + FALSIFIED)) / $TOTAL ($((( VERIFIED + FALSIFIED ) * 100 / TOTAL))%)"
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Write JSON report
mkdir -p reports/benchmarks
cat > "$OUTPUT_JSON" << ENDJSON
{
  "benchmark": "cifar10_${MODEL_VARIANT}",
  "category": "vnncomp2021/cifar10_resnet",
  "model": "${MODEL_VARIANT}.onnx",
  "method": "$METHOD",
  "timeout_s": $TIMEOUT,
  "timestamp": "$TIMESTAMP",
  "commit": "$(git rev-parse --short HEAD)",
  "binary": "$NY",
  "instances_run": $TOTAL,
  "instances_total": $TOTAL,
  "verified": $VERIFIED,
  "falsified": $FALSIFIED,
  "unknown": $UNKNOWN,
  "timeout": $TIMEOUT_COUNT,
  "error": $ERROR_COUNT,
  "solved_rate": $(python3 --no-wrapper -c "print(f'{($VERIFIED + $FALSIFIED) / $TOTAL:.4f}' if $TOTAL > 0 else '0')"),
  "results": $RESULTS_JSON
}
ENDJSON

echo ""
echo "Results written to: $OUTPUT_JSON"
