#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark ny on VNN-COMP 2025 malbeware category.
# Runs all 150 instances (or a filtered subset) and reports results.
#
# Usage:
#   scripts/benchmark_malbeware.sh [all|linear-25|4-25|16-25] [--start-at N] [--limit N]
#
# Examples:
#   scripts/benchmark_malbeware.sh linear-25 --limit 10
#   scripts/benchmark_malbeware.sh 16-25 --start-at 11 --limit 5

set -euo pipefail

BENCH_DIR="benchmarks/vnncomp2025/benchmarks/malbeware"
NY_BIN="${NY_BIN:-./target/release/ny}"
PRESET="configs/vnncomp25/malbeware.yaml"
REPORT_DIR="reports/benchmarks"
FILTER="all"  # all, linear-25, 4-25, 16-25
START_AT=1
LIMIT=0

if [[ $# -gt 0 ]] && [[ "$1" != --* ]]; then
    FILTER="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-at) START_AT="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -x "$NY_BIN" ]]; then
    echo "Ny binary not found or not executable: $NY_BIN"
    exit 1
fi
if [[ "$START_AT" -lt 1 ]]; then
    echo "--start-at must be >= 1"
    exit 1
fi
if [[ "$LIMIT" -lt 0 ]]; then
    echo "--limit must be >= 0"
    exit 1
fi

mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$REPORT_DIR/malbeware_${FILTER}_${TIMESTAMP}.csv"
TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

echo "model,property,timeout,result,elapsed" > "$REPORT"

FILTERED_INDEX=0
TOTAL=0
VERIFIED=0
VIOLATED=0
UNKNOWN=0
ERROR=0

while IFS=',' read -r onnx vnnlib timeout; do
    # Strip carriage returns (instances.csv may have Windows line endings)
    onnx="${onnx//$'\r'/}"
    vnnlib="${vnnlib//$'\r'/}"
    timeout="${timeout//$'\r'/}"

    # Apply filter
    if [[ "$FILTER" != "all" ]] && [[ "$onnx" != *"$FILTER"* ]]; then
        continue
    fi

    FILTERED_INDEX=$((FILTERED_INDEX + 1))
    if [[ "$FILTERED_INDEX" -lt "$START_AT" ]]; then
        continue
    fi
    if [[ "$LIMIT" -gt 0 ]] && [[ "$TOTAL" -ge "$LIMIT" ]]; then
        break
    fi
    TOTAL=$((TOTAL + 1))
    ONNX_PATH="$BENCH_DIR/$onnx"
    VNNLIB_PATH="$BENCH_DIR/$vnnlib"

    # Decompress if needed
    if [[ ! -f "$ONNX_PATH" ]] && [[ -f "${ONNX_PATH}.gz" ]]; then
        gzip -dk "${ONNX_PATH}.gz"
    fi

    echo -n "[$TOTAL @ $FILTERED_INDEX] $(basename "$onnx") / $(basename "$vnnlib" .vnnlib) (${timeout}s)... "

    START_TIME=$(python3 -c "import time; print(time.time())")

    # Write output to temp file to avoid pipe buffer issues with large outputs
    "$NY_BIN" beta-crown "$ONNX_PATH" \
        --property "$VNNLIB_PATH" \
        --preset "$PRESET" \
        --timeout "$timeout" > "$TMPOUT" 2>&1 || true

    END_TIME=$(python3 -c "import time; print(time.time())")
    ELAPSED=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")

    # Parse result from temp file
    if grep -q "Status: VERIFIED" "$TMPOUT"; then
        if python3 -c "exit(0 if float('$ELAPSED') <= float('$timeout') else 1)"; then
            RESULT="verified"
            VERIFIED=$((VERIFIED + 1))
        else
            RESULT="timeout"
            UNKNOWN=$((UNKNOWN + 1))
        fi
    elif grep -q "Status: VIOLATED" "$TMPOUT"; then
        RESULT="violated"
        VIOLATED=$((VIOLATED + 1))
    elif grep -q "Status: UNKNOWN" "$TMPOUT"; then
        RESULT="unknown"
        UNKNOWN=$((UNKNOWN + 1))
    elif grep -q "Timed out" "$TMPOUT"; then
        RESULT="timeout"
        UNKNOWN=$((UNKNOWN + 1))
    else
        RESULT="error"
        ERROR=$((ERROR + 1))
        echo ""
        echo "  DEBUG: $(tail -5 "$TMPOUT")"
    fi

    echo "$RESULT (${ELAPSED}s)"
    echo "$(basename "$onnx"),$(basename "$vnnlib"),${timeout},${RESULT},${ELAPSED}" >> "$REPORT"

done < "$BENCH_DIR/instances.csv"

echo ""
echo "=== malbeware ($FILTER) Results ==="
echo "Start at: $START_AT"
echo "Limit: ${LIMIT:-0}"
echo "Total: $TOTAL"
echo "Verified: $VERIFIED"
echo "Violated: $VIOLATED"
echo "Unknown/Timeout: $UNKNOWN"
echo "Error: $ERROR"
echo "Score: $((VERIFIED + VIOLATED))/$TOTAL"
echo ""
echo "Report: $REPORT"
