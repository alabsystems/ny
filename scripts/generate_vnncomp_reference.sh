#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Generate VNN-COMP reference result files from known competition outcomes.
#
# Usage:
#   scripts/generate_vnncomp_reference.sh <category> <default_result>
#   scripts/generate_vnncomp_reference.sh --from-harness <harness_csv> <tool_name>
#
# Modes:
#   1) Generate from instances.csv with a uniform result (e.g., all "verified"):
#      scripts/generate_vnncomp_reference.sh acasxu_2023 verified
#
#   2) Convert VNN-COMP harness CSV to reference format:
#      scripts/generate_vnncomp_reference.sh --from-harness results.csv alpha_beta_crown
#
# Output: reports/benchmarks/reference/<category>_<tool>.csv

set -euo pipefail

BENCH_ROOT="${BENCH_ROOT:-benchmarks/vnncomp2025/benchmarks}"
REF_DIR="reports/benchmarks/reference"
mkdir -p "$REF_DIR"

if [[ "${1:-}" == "--from-harness" ]]; then
    # Mode 2: Convert VNN-COMP harness CSV
    HARNESS_CSV="${2:?Usage: $0 --from-harness <harness_csv> <tool_name>}"
    TOOL_NAME="${3:?Usage: $0 --from-harness <harness_csv> <tool_name>}"

    [[ -f "$HARNESS_CSV" ]] || { echo "ERROR: File not found: $HARNESS_CSV" >&2; exit 1; }

    # Group by category and create per-category reference files
    # Harness format: category,onnx_path,vnnlib_path,prepare_runtime,result,runtime
    # Use file-existence check instead of PREV_CAT tracking to handle
    # unsorted/interleaved categories without data loss
    SEEN_FILES=""
    while IFS=',' read -r cat onnx vnnlib prep result runtime; do
        [[ "$cat" == "category" ]] && continue  # skip header if present
        cat="${cat//$'\r'/}"
        result="${result//$'\r'/}"
        onnx_base="${onnx##*/}"; onnx_base="${onnx_base%.onnx}"; onnx_base="${onnx_base%.onnx.gz}"
        vnnlib_base="${vnnlib##*/}"; vnnlib_base="${vnnlib_base%.vnnlib}"

        OUTFILE="$REF_DIR/${cat}_${TOOL_NAME}.csv"
        if [[ "$SEEN_FILES" != *"|$OUTFILE|"* ]]; then
            echo "model,property,result" > "$OUTFILE"
            SEEN_FILES="${SEEN_FILES}|$OUTFILE|"
        fi
        echo "$onnx_base,$vnnlib_base,$result" >> "$OUTFILE"
    done < "$HARNESS_CSV"

    echo "Generated reference files in $REF_DIR/ from $HARNESS_CSV"
else
    # Mode 1: Generate uniform result from instances.csv
    CATEGORY="${1:?Usage: $0 <category> <default_result>}"
    DEFAULT_RESULT="${2:?Usage: $0 <category> <default_result>}"
    TOOL_NAME="${3:-alpha_beta_crown}"  # optional 3rd arg

    INSTANCES="$BENCH_ROOT/$CATEGORY/instances.csv"
    [[ -f "$INSTANCES" ]] || { echo "ERROR: Not found: $INSTANCES" >&2; exit 1; }

    OUTFILE="$REF_DIR/${CATEGORY}_${TOOL_NAME}.csv"
    echo "model,property,result" > "$OUTFILE"

    COUNT=0
    while IFS=',' read -r onnx vnnlib timeout; do
        onnx="${onnx//$'\r'/}"
        vnnlib="${vnnlib//$'\r'/}"
        onnx_base="${onnx##*/}"; onnx_base="${onnx_base%.onnx}"; onnx_base="${onnx_base%.onnx.gz}"
        vnnlib_base="${vnnlib##*/}"; vnnlib_base="${vnnlib_base%.vnnlib}"
        echo "$onnx_base,$vnnlib_base,$DEFAULT_RESULT" >> "$OUTFILE"
        COUNT=$((COUNT + 1))
    done < "$INSTANCES"

    echo "Generated: $OUTFILE ($COUNT instances, all=$DEFAULT_RESULT)"
fi
