#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Run VNN-COMP benchmarks across the current runnable default categories.
# Wraps scripts/benchmark_vnncomp.sh for each category and aggregates results.
#
# Usage:
#   scripts/benchmark_vnncomp_all.sh [--categories "cat1 cat2"] [--limit N] [--dry-run] [--start-from CAT] [--year YEAR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

YEAR="${YEAR:-2025}"
BENCH_ROOT="${BENCH_ROOT:-benchmarks/vnncomp${YEAR}/benchmarks}"
REPORT_DIR="reports/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)_$$
LIMIT_FLAG=""
DRY_RUN=false
START_FROM=""
CUSTOM_CATEGORIES=false

# Categories with preset configs (tuned for competition)
#
# Keep the default full-run set to categories with current-head runnable evidence.
# Runtime-limited probes remain available via --categories, but should not be part
# of the default bench sweep while they still hit 0-domain watchdog timeouts.
PRESET_CATEGORIES="malbeware sat_relu cersyve lsnc_relu relusplitter collins_rul_cnn_2022 safenlp_2024 cora_2024 dist_shift_2023 metaroom_2023 linearizenn_2024 nn4sys ml4acopf_2024 cgan_2023 cifar100_2024"
# Additional categories (default settings unless benchmark_vnncomp.sh adds per-category flags)
ADDITIONAL_CATEGORIES="acasxu_2023 tllverifybench_2023"
# Formerly-skipped categories, un-skipped 2026-05-30 after real-model probes
# (vnncomp2025 benchmarks) confirmed every one now produces a SOUND verdict — no
# crashes/errors. Status from direct stock-wgpu probes on real ONNX:
#   collins_aerospace_benchmark : SOLVES (sat, ~34s)
#   vggnet16_2022               : SOLVES (sat, ~597s, within 1200s budget)
#   vit_2023                    : valid verdict (unknown) — ViT attention IBP unblocked (O(n^2) matmul-IBP fix)
#   traffic_signs_recognition_2023 : valid verdict (timeout)
#   soundnessbench              : valid verdict (timeout/unknown); PGD time-budget bounded
#   tinyimagenet_2024           : valid verdict (unknown); loads past Conv shape-infer fix
#   cctsdb_yolo_2023            : SOLVES via cell enumeration (#cctsdb Phase C,
#                                 2026-07-04): unsat ~66s / sat ~10-15s dev-build
#                                 probes, sat witnesses ORT-gate confirmed
#   yolo_2023                   : valid verdict (unknown)
# Remaining work to raise solve-rate (NOT a soundness/crash issue): graph
# conv-layer IBP/alpha-CROWN node-collection is slow on conv-heavy DAGs
# (yolo/tinyimagenet).
RUNTIME_LIMITED_CATEGORIES="vggnet16_2022 collins_aerospace_benchmark vit_2023 traffic_signs_recognition_2023 soundnessbench tinyimagenet_2024 cctsdb_yolo_2023 yolo_2023"
# Default: all runnable categories
CATEGORIES="$PRESET_CATEGORIES $ADDITIONAL_CATEGORIES $RUNTIME_LIMITED_CATEGORIES"

# Skipped categories with reasons (reported in summary)
# Stored as newline-separated "name:reason" pairs (bash 3 compatible)
# All reasons must be specific and evidence-based, not placeholder labels.
#
# 2026-05-30: SKIPPED_LIST emptied. Every previously-skipped category was probed
# on its real vnncomp2025 ONNX model and now produces a SOUND verdict (no
# crash/error); the formerly-skipped set moved to RUNTIME_LIMITED_CATEGORIES
# above and now runs as part of the default sweep. Per-category status is
# documented there. The verifier is sound, so running these can only add points
# (solved instances) — unsolved instances return unknown/timeout, never a
# penalized wrong verdict.
SKIPPED_LIST=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --categories) CATEGORIES="$2"; CUSTOM_CATEGORIES=true; shift 2 ;;
        --limit) LIMIT_FLAG="--limit $2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --start-from) START_FROM="$2"; shift 2 ;;
        --year) YEAR="$2"; BENCH_ROOT="benchmarks/vnncomp${YEAR}/benchmarks"; shift 2 ;;
        --help|-h)
            echo "Usage: scripts/benchmark_vnncomp_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --categories \"cat1 cat2\"  Run specific categories (default: current runnable set)"
            echo "  --limit N                 Limit instances per category"
            echo "  --dry-run                 List categories and instance counts without running"
            echo "  --start-from CAT          Skip categories before CAT"
            echo "  --year YEAR               Benchmark year (default: 2025)"
            echo "  --help                    Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$REPORT_DIR"

GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
NY_VERSION=$(git describe --tags --dirty 2>/dev/null || echo "dev")

echo "========================================"
echo "VNN-COMP Benchmark Runner"
echo "========================================"
echo "Year:       $YEAR"
echo "Commit:     $GIT_COMMIT"
echo "Version:    $NY_VERSION"
echo "Categories: $(echo $CATEGORIES | wc -w | tr -d ' ')"
echo "Limit:      ${LIMIT_FLAG:-none}"
echo "Dry run:    $DRY_RUN"
echo ""

# Dry-run mode: list categories and instance counts
if $DRY_RUN; then
    echo "=== Supported Categories ==="
    for category in $CATEGORIES; do
        CSV="$BENCH_ROOT/$category/instances.csv"
        if [[ -f "$CSV" ]]; then
            COUNT=$(wc -l < "$CSV" | tr -d ' ')
            PRESET="none"
            if [[ -f "configs/vnncomp${YEAR: -2}/$category.yaml" ]]; then
                PRESET="configs/vnncomp${YEAR: -2}/$category.yaml"
            fi
            printf "  %-30s %4d instances  preset: %s\n" "$category" "$COUNT" "$PRESET"
        else
            printf "  %-30s      NOT FOUND\n" "$category"
        fi
    done
    echo ""
    echo "=== Skipped Categories ==="
    echo "$SKIPPED_LIST" | while IFS=':' read -r name reason; do
        printf "  %-30s %s\n" "$name" "$reason"
    done | sort
    exit 0
fi

# Determine run scope: full vs partial
# A full run uses the default category set with no subsetting flags.
TRACKER_YEAR="${TRACKER_YEAR:-2025}"

HAS_LIMIT=false
HAS_START_FROM=false
if [[ -n "$LIMIT_FLAG" ]]; then
    HAS_LIMIT=true
fi
if [[ -n "$START_FROM" ]]; then
    HAS_START_FROM=true
fi

# Track per-category CSV reports for aggregation
CSV_FILES=()
# Track failed categories as newline-separated "name:exit_code:reason" triples
FAILED_LIST=""
STARTED=false

OVERALL_START=$(python3 -c "import time; print(time.time())")

for category in $CATEGORIES; do
    # Handle --start-from
    if [[ -n "$START_FROM" ]] && ! $STARTED; then
        if [[ "$category" == "$START_FROM" ]]; then
            STARTED=true
        else
            echo "Skipping $category (before --start-from $START_FROM)"
            continue
        fi
    fi

    CSV="$BENCH_ROOT/$category/instances.csv"
    if [[ ! -f "$CSV" ]]; then
        echo "WARNING: $category — no instances.csv found, skipping"
        continue
    fi

    echo ""
    echo "########################################"
    echo "# Category: $category"
    echo "########################################"

    # Capture child output to parse the exact Report: path
    CHILD_LOG=$(mktemp)
    CHILD_EXIT=0
    # shellcheck disable=SC2086
    bash "$SCRIPT_DIR/benchmark_vnncomp.sh" "$category" $LIMIT_FLAG 2>&1 | tee "$CHILD_LOG" || CHILD_EXIT=${PIPESTATUS[0]}

    if [[ $CHILD_EXIT -ne 0 ]]; then
        echo "WARNING: $category exited with error (exit $CHILD_EXIT), continuing"
        FAILED_LIST="${FAILED_LIST}${category}:${CHILD_EXIT}:non-zero exit code
"
        rm -f "$CHILD_LOG"
        continue
    fi

    # Parse the exact report path from child output instead of globbing
    REPORT_PATH=$(grep -m1 '^Report: ' "$CHILD_LOG" | sed 's/^Report: //')
    rm -f "$CHILD_LOG"

    if [[ -z "$REPORT_PATH" ]]; then
        echo "WARNING: $category — no Report: line in output, skipping"
        FAILED_LIST="${FAILED_LIST}${category}:0:no report path in output
"
        continue
    fi

    # Verify the parsed report path exists and is inside reports/benchmarks/
    case "$REPORT_PATH" in
        "$REPORT_DIR"/* | reports/benchmarks/*)
            if [[ -f "$REPORT_PATH" ]]; then
                CSV_FILES+=("$REPORT_PATH")
            else
                echo "WARNING: $category — report path does not exist: $REPORT_PATH"
                FAILED_LIST="${FAILED_LIST}${category}:0:report file missing
"
            fi
            ;;
        *)
            echo "WARNING: $category — report path outside expected directory: $REPORT_PATH"
            FAILED_LIST="${FAILED_LIST}${category}:0:report path outside reports/benchmarks/
"
            ;;
    esac
done

OVERALL_END=$(python3 -c "import time; print(time.time())")
WALL_TIME=$(python3 -c "print(f'{$OVERALL_END - $OVERALL_START:.1f}')")

echo ""
echo "========================================"
echo "All categories complete. Wall time: ${WALL_TIME}s"
echo "========================================"

# Build skipped JSON for aggregation
SKIPPED_JSON=$(python3 -c "
import json
skipped = {}
for line in '''$SKIPPED_LIST'''.strip().split('\n'):
    name, reason = line.split(':', 1)
    skipped[name.strip()] = reason.strip()
print(json.dumps(skipped))
")

# Build failed JSON for aggregation
FAILED_JSON=$(python3 -c "
import json
failed = {}
for line in '''$FAILED_LIST'''.strip().split('\n'):
    if not line.strip():
        continue
    parts = line.split(':', 2)
    if len(parts) == 3:
        failed[parts[0].strip()] = {'exit_code': int(parts[1].strip()), 'reason': parts[2].strip()}
print(json.dumps(failed))
")

# Determine run_scope
RUN_SCOPE="full"
if $HAS_LIMIT || $HAS_START_FROM || $CUSTOM_CATEGORIES; then
    RUN_SCOPE="partial"
fi

# Aggregate results into JSON summary
SUMMARY="$REPORT_DIR/vnncomp_summary_${TIMESTAMP}.json"
AGGREGATE_ARGS=(
    --output "$SUMMARY"
    --year "$YEAR"
    --commit "$GIT_COMMIT"
    --version "$NY_VERSION"
    --wall-time "$WALL_TIME"
    --skipped "$SKIPPED_JSON"
    --failed "$FAILED_JSON"
    --run-scope "$RUN_SCOPE"
    --tracker-year "$TRACKER_YEAR"
)
if [[ "$RUN_SCOPE" == "full" ]] && [[ "$YEAR" == "$TRACKER_YEAR" ]] && [[ -z "$FAILED_LIST" ]]; then
    AGGREGATE_ARGS+=(--publish-metrics)
fi
if [[ ${#CSV_FILES[@]} -gt 0 ]]; then
    AGGREGATE_ARGS+=("${CSV_FILES[@]}")
fi

python3 "$SCRIPT_DIR/aggregate_vnncomp_results.py" "${AGGREGATE_ARGS[@]}"
if [[ "$RUN_SCOPE" == "full" ]] && [[ "$YEAR" == "$TRACKER_YEAR" ]] && [[ -z "$FAILED_LIST" ]]; then
    python3 "$SCRIPT_DIR/refresh_vnncomp_current_status.py"
fi
echo ""
echo "Summary: $SUMMARY"
echo "Run scope: $RUN_SCOPE"
if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    echo "Note: no successful category CSVs were aggregated; summary contains failed/skipped metadata only."
fi
echo ""
python3 -c "
import json
with open('$SUMMARY') as f:
    s = json.load(f)
print(f\"Total: {s['total_instances']} instances across {s['categories_attempted']} categories\")
print(f\"Score: {s['total_score']}/{s['total_instances']} ({s['overall_solve_rate']:.1f}%)\")
print(f\"Skipped: {len(s.get('skipped', {}))} categories\")
print(f\"Failed: {len(s.get('failed', {}))} categories\")
print(f\"Publication: {s.get('publication_scope', 'unknown')}\")
"
