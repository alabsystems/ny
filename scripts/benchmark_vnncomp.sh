#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark ny on any VNN-COMP category.
# Reads instances.csv and runs beta-crown on each instance, reporting results.
#
# Usage:
#   scripts/benchmark_vnncomp.sh <category> [--pgd] [--branching input] [--backend wgpu] [--compare-backends] [--domain-batch-metrics] [--no-preset] [--complete-verifier mip] [--start-at N] [--limit N]
#
# Examples:
#   scripts/benchmark_vnncomp.sh malbeware --pgd
#   scripts/benchmark_vnncomp.sh cersyve --pgd --branching input
#   scripts/benchmark_vnncomp.sh sat_relu --complete-verifier mip --start-at 2 --limit 10
#   scripts/benchmark_vnncomp.sh soundnessbench --backend wgpu --limit 1
#   scripts/benchmark_vnncomp.sh cersyve --compare-backends --start-at 3 --limit 2
#   scripts/benchmark_vnncomp.sh ml4acopf_2024 --no-preset --branching input --limit 8

set -euo pipefail

CATEGORY="${1:?Usage: benchmark_vnncomp.sh <category> [--pgd] [--branching METHOD] [--backend BACKEND] [--compare-backends] [--domain-batch-metrics] [--no-preset] [--start-at N] [--limit N]}"
shift

BENCH_ROOT="${BENCH_ROOT:-benchmarks/vnncomp2025/benchmarks}"
BENCH_DIR="$BENCH_ROOT/$CATEGORY"
# Track whether the caller explicitly set NY_BIN for provenance tagging (#4346)
NY_BIN_EXPLICIT=""
if [[ -n "${NY_BIN:-}" ]]; then
    NY_BIN_EXPLICIT="explicit"
fi
NY_BIN="${NY_BIN:-./target/release/ny}"
PRESET_DIR="configs/vnncomp25"
REPORT_DIR="reports/benchmarks"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_SIGNAL_RETRIES="${MAX_SIGNAL_RETRIES:-1}"
EXTERNAL_TIMEOUT_SLACK="${EXTERNAL_TIMEOUT_SLACK:-5}"
WATCHDOG_TERM_GRACE="${WATCHDOG_TERM_GRACE:-2}"
NO_PRESET=false
COMPARE_BACKENDS=false
DOMAIN_BATCH_METRICS=false
CATEGORY_EXTRA_FLAGS=""
CATEGORY_DEFAULT_BRANCHING=""
DOMAIN_BATCH_METRICS_ROOT=""
DOMAIN_BATCH_METRICS_FLAG=""
LAST_DOMAIN_BATCH_METRICS_JSONL=""

# Parse optional flags
PGD_FLAG=""
BRANCHING_FLAG=""
BACKEND_FLAG=""
VERIFIER_FLAG=""
START_AT=1
LIMIT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pgd) PGD_FLAG="--pgd-attack"; shift ;;
        --branching) BRANCHING_FLAG="--branching $2"; shift 2 ;;
        --backend) BACKEND_FLAG="--backend $2"; shift 2 ;;
        --compare-backends) COMPARE_BACKENDS=true; shift ;;
        --domain-batch-metrics) DOMAIN_BATCH_METRICS=true; shift ;;
        --no-preset) NO_PRESET=true; shift ;;
        --complete-verifier) VERIFIER_FLAG="--complete-verifier $2"; shift 2 ;;
        --mip-solver) VERIFIER_FLAG="$VERIFIER_FLAG --mip-solver $2"; shift 2 ;;
        --start-at) START_AT="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$CATEGORY" in
    acasxu_2023)
        # ACAS-Xu: alpha-CROWN regresses from 99.8% to 0% verified because
        # frozen root alpha bounds don't tighten for sub-domains (#3453).
        # Reference uses bound_prop_method=crown (plain CROWN, not alpha-CROWN).
        CATEGORY_EXTRA_FLAGS="--no-alpha"
        ;;
    vit_2023)
        # ViT attention uses internal Softmax nodes, so this category needs the
        # same heuristic relaxation already supported by `ny verify`.
        # Its transformer residual graph is also a DAG, so beta-crown must use
        # input splitting unless the caller overrides branching explicitly.
        CATEGORY_EXTRA_FLAGS="--allow-heuristic-softmax"
        CATEGORY_DEFAULT_BRANCHING="input"
        ;;
esac

if [[ -z "$BRANCHING_FLAG" ]] && [[ -n "$CATEGORY_DEFAULT_BRANCHING" ]]; then
    BRANCHING_FLAG="--branching $CATEGORY_DEFAULT_BRANCHING"
fi

# Category-default MIP routing: categories that the reference alpha-beta-CROWN
# routes to complete_verifier=mip. Only applied when caller hasn't specified
# --complete-verifier explicitly. (#3218, #2569)
if [[ -z "$VERIFIER_FLAG" ]]; then
    case "$CATEGORY" in
        sat_relu|malbeware|safenlp_2024|relusplitter)
            VERIFIER_FLAG="--complete-verifier mip"
            ;;
    esac
fi

# Validate
if [[ ! -d "$BENCH_DIR" ]]; then
    echo "Category not found: $BENCH_DIR"
    exit 1
fi
if [[ ! -f "$BENCH_DIR/instances.csv" ]]; then
    echo "No instances.csv in $BENCH_DIR"
    exit 1
fi
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
if [[ "$MAX_SIGNAL_RETRIES" -lt 0 ]]; then
    echo "MAX_SIGNAL_RETRIES must be >= 0"
    exit 1
fi
if [[ "$EXTERNAL_TIMEOUT_SLACK" -lt 0 ]]; then
    echo "EXTERNAL_TIMEOUT_SLACK must be >= 0"
    exit 1
fi
if [[ "$WATCHDOG_TERM_GRACE" -lt 0 ]]; then
    echo "WATCHDOG_TERM_GRACE must be >= 0"
    exit 1
fi
if [[ "$COMPARE_BACKENDS" == "true" && -n "$BACKEND_FLAG" ]]; then
    echo "--compare-backends cannot be combined with --backend"
    exit 1
fi

# Auto-detect preset unless explicitly disabled.
PRESET=""
PRESET_PATH=""
if [[ "$NO_PRESET" == "false" && -f "$PRESET_DIR/$CATEGORY.yaml" ]]; then
    PRESET_PATH="$PRESET_DIR/$CATEGORY.yaml"
    PRESET="--preset $PRESET_PATH"
fi

mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)_$$
if [[ "$COMPARE_BACKENDS" == "true" ]]; then
    REPORT="$REPORT_DIR/${CATEGORY}_compare_backends_${TIMESTAMP}.csv"
else
    REPORT="$REPORT_DIR/${CATEGORY}_${TIMESTAMP}.csv"
fi
if [[ "$DOMAIN_BATCH_METRICS" == "true" ]]; then
    DOMAIN_BATCH_METRICS_ROOT="$REPORT_DIR/domain_batch_metrics/${CATEGORY}_${TIMESTAMP}"
    mkdir -p "$DOMAIN_BATCH_METRICS_ROOT"
fi
TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

WATCHDOG_TIMEOUT_HIT=0
WATCHDOG_TIMEOUT_LIMIT=0
LAST_RESULT=""
LAST_ELAPSED=""
LAST_DOMAINS=""
LAST_ACTUAL_METHOD=""
# shellcheck source=scripts/benchmark_vnncomp_helpers.sh
source "$SCRIPT_DIR/benchmark_vnncomp_helpers.sh"

# Compute binary provenance once at startup (#4346)
compute_ny_provenance "$NY_BIN" "$NY_BIN_EXPLICIT"
PROVENANCE_NOTES=$(format_provenance_tags)
# Stamp compare-backends invocations with a compare_run_id (#4383)
if [[ "$COMPARE_BACKENDS" == "true" ]]; then
    COMPARE_RUN_ID="${CATEGORY}_${TIMESTAMP}"
    PROVENANCE_NOTES="${PROVENANCE_NOTES}; compare_run_id=${COMPARE_RUN_ID}"
fi
BENCHMARK_SUITE_KEY=$(benchmark_suite_key "$BENCH_ROOT")

prepare_domain_batch_metrics_for_run() {
    local source_index="$1"
    local backend_name="$2"
    local backend_suffix="${backend_name:-default}"

    LAST_DOMAIN_BATCH_METRICS_JSONL=""
    DOMAIN_BATCH_METRICS_FLAG=""
    if [[ "$DOMAIN_BATCH_METRICS" != "true" ]]; then
        return 0
    fi

    local metrics_path="$DOMAIN_BATCH_METRICS_ROOT/${CATEGORY}_row${source_index}_${backend_suffix}.jsonl"
    LAST_DOMAIN_BATCH_METRICS_JSONL=$(to_repo_relative_path "$metrics_path")
    DOMAIN_BATCH_METRICS_FLAG="--domain-batch-metrics-jsonl $metrics_path"
}

notes_with_domain_batch_metrics() {
    local notes="$1"
    local metrics_path="$2"

    if [[ -n "$metrics_path" ]]; then
        printf '%s; domain_batch_metrics_jsonl=%s' "$notes" "$metrics_path"
    else
        printf '%s' "$notes"
    fi
}

record_single_result() {
    local result="$1"
    local elapsed="$2"

    TOTAL_TIME=$(add_elapsed "$TOTAL_TIME" "$elapsed")

    case "$result" in
        verified) VERIFIED=$((VERIFIED + 1)) ;;
        violated) VIOLATED=$((VIOLATED + 1)) ;;
        unknown|timeout) UNKNOWN=$((UNKNOWN + 1)) ;;
        error) ERROR=$((ERROR + 1)) ;;
    esac
}

record_compare_result() {
    local backend="$1"
    local result="$2"
    local elapsed="$3"

    case "$backend" in
        cpu)
            CPU_TOTAL_TIME=$(add_elapsed "$CPU_TOTAL_TIME" "$elapsed")
            case "$result" in
                verified) CPU_VERIFIED=$((CPU_VERIFIED + 1)) ;;
                violated) CPU_VIOLATED=$((CPU_VIOLATED + 1)) ;;
                unknown|timeout) CPU_UNKNOWN=$((CPU_UNKNOWN + 1)) ;;
                error) CPU_ERROR=$((CPU_ERROR + 1)) ;;
            esac
            ;;
        wgpu)
            WGPU_TOTAL_TIME=$(add_elapsed "$WGPU_TOTAL_TIME" "$elapsed")
            case "$result" in
                verified) WGPU_VERIFIED=$((WGPU_VERIFIED + 1)) ;;
                violated) WGPU_VIOLATED=$((WGPU_VIOLATED + 1)) ;;
                unknown|timeout) WGPU_UNKNOWN=$((WGPU_UNKNOWN + 1)) ;;
                error) WGPU_ERROR=$((WGPU_ERROR + 1)) ;;
            esac
            ;;
    esac
}

write_backend_benchmark_header "$REPORT"
echo "=== Benchmarking: $CATEGORY ==="
echo "Bench dir: $BENCH_DIR"
if [[ "$NO_PRESET" == "true" ]]; then
    echo "Preset: disabled (--no-preset)"
else
    echo "Preset: ${PRESET_PATH:-none}"
fi
echo "PGD: ${PGD_FLAG:-disabled}"
echo "Branching: ${BRANCHING_FLAG:-default}"
if [[ "$COMPARE_BACKENDS" == "true" ]]; then
    echo "Backend: compare cpu vs wgpu"
else
    echo "Backend: ${BACKEND_FLAG:-default}"
fi
echo "Verifier: ${VERIFIER_FLAG:-bab}"
echo "Category flags: ${CATEGORY_EXTRA_FLAGS:-none}"
echo "Domain-batch metrics: ${DOMAIN_BATCH_METRICS}"
echo "Start at: $START_AT"
echo "Limit: ${LIMIT:-0}"
echo "Binary: $NY_BIN (source=$NY_SOURCE, sha256=${NY_SHA256:0:16}...)"
echo ""

SOURCE_INDEX=0
TOTAL=0
VERIFIED=0
VIOLATED=0
UNKNOWN=0
ERROR=0
TOTAL_TIME=0
CPU_VERIFIED=0
CPU_VIOLATED=0
CPU_UNKNOWN=0
CPU_ERROR=0
CPU_TOTAL_TIME=0
WGPU_VERIFIED=0
WGPU_VIOLATED=0
WGPU_UNKNOWN=0
WGPU_ERROR=0
WGPU_TOTAL_TIME=0
DIVERGED=0

while IFS=',' read -r onnx vnnlib timeout; do
    # Strip carriage returns
    onnx="${onnx//$'\r'/}"
    vnnlib="${vnnlib//$'\r'/}"
    timeout="${timeout//$'\r'/}"

    SOURCE_INDEX=$((SOURCE_INDEX + 1))
    if [[ "$SOURCE_INDEX" -lt "$START_AT" ]]; then
        continue
    fi
    if [[ "$LIMIT" -gt 0 ]] && [[ "$TOTAL" -ge "$LIMIT" ]]; then
        break
    fi
    TOTAL=$((TOTAL + 1))

    ONNX_PATH="$BENCH_DIR/$onnx"
    VNNLIB_PATH="$BENCH_DIR/$vnnlib"
    SUBJECT_ID="$(benchmark_row_identity "$BENCHMARK_SUITE_KEY" "$CATEGORY" "$SOURCE_INDEX" "$onnx" "$vnnlib")"
    COMPARISON_KEY="$SUBJECT_ID"
    MODEL_PATH_REL="$(to_repo_relative_path "$ONNX_PATH")"
    PROPERTY_PATH_REL="$(to_repo_relative_path "$VNNLIB_PATH")"

    # Decompress if needed
    if [[ ! -f "$ONNX_PATH" ]] && [[ -f "${ONNX_PATH}.gz" ]]; then
        gzip -dk "${ONNX_PATH}.gz"
    fi

    echo -n "[$TOTAL @ $SOURCE_INDEX] $(basename "$onnx") / $(basename "$vnnlib" .vnnlib) (${timeout}s)... "
    if [[ "$COMPARE_BACKENDS" == "true" ]]; then
        prepare_domain_batch_metrics_for_run "$SOURCE_INDEX" "cpu"
        run_benchmark_instance "$onnx" "$vnnlib" "$timeout" "cpu" "--backend cpu"
        CPU_RESULT="$LAST_RESULT"
        CPU_ELAPSED="$LAST_ELAPSED"
        CPU_DOMAINS="$LAST_DOMAINS"
        CPU_ACTUAL_METHOD="$LAST_ACTUAL_METHOD"
        CPU_NOTES=$(notes_with_domain_batch_metrics "$PROVENANCE_NOTES" "$LAST_DOMAIN_BATCH_METRICS_JSONL")
        record_compare_result "cpu" "$CPU_RESULT" "$CPU_ELAPSED"

        prepare_domain_batch_metrics_for_run "$SOURCE_INDEX" "wgpu"
        run_benchmark_instance "$onnx" "$vnnlib" "$timeout" "wgpu" "--backend wgpu"
        WGPU_RESULT="$LAST_RESULT"
        WGPU_ELAPSED="$LAST_ELAPSED"
        WGPU_DOMAINS="$LAST_DOMAINS"
        WGPU_ACTUAL_METHOD="$LAST_ACTUAL_METHOD"
        WGPU_NOTES=$(notes_with_domain_batch_metrics "$PROVENANCE_NOTES" "$LAST_DOMAIN_BATCH_METRICS_JSONL")
        record_compare_result "wgpu" "$WGPU_RESULT" "$WGPU_ELAPSED"

        DELTA_SECONDS=$(python3 -c "print(f'{float(\"$WGPU_ELAPSED\") - float(\"$CPU_ELAPSED\"):.2f}')")
        STATUS_DIVERGED="no"
        if [[ "$CPU_RESULT" != "$WGPU_RESULT" ]]; then
            STATUS_DIVERGED="yes"
            DIVERGED=$((DIVERGED + 1))
        fi

        echo "cpu=$CPU_RESULT (${CPU_ELAPSED}s, ${CPU_DOMAINS} domains); wgpu=$WGPU_RESULT (${WGPU_ELAPSED}s, ${WGPU_DOMAINS} domains); delta=${DELTA_SECONDS}s"
        append_backend_benchmark_row \
            "$REPORT" \
            "backend_benchmark_row_v1" \
            "vnncomp_compare_backends" \
            "vnncomp_instance" \
            "$SUBJECT_ID" \
            "$COMPARISON_KEY" \
            "$CATEGORY" \
            "" \
            "$MODEL_PATH_REL" \
            "$PROPERTY_PATH_REL" \
            "$PRESET_PATH" \
            "cpu" \
            "$timeout" \
            "$CPU_RESULT" \
            "$CPU_ACTUAL_METHOD" \
            "$CPU_ELAPSED" \
            "$CPU_DOMAINS" \
            "" \
            "" \
            "$CPU_NOTES"
        append_backend_benchmark_row \
            "$REPORT" \
            "backend_benchmark_row_v1" \
            "vnncomp_compare_backends" \
            "vnncomp_instance" \
            "$SUBJECT_ID" \
            "$COMPARISON_KEY" \
            "$CATEGORY" \
            "" \
            "$MODEL_PATH_REL" \
            "$PROPERTY_PATH_REL" \
            "$PRESET_PATH" \
            "wgpu" \
            "$timeout" \
            "$WGPU_RESULT" \
            "$WGPU_ACTUAL_METHOD" \
            "$WGPU_ELAPSED" \
            "$WGPU_DOMAINS" \
            "" \
            "" \
            "$WGPU_NOTES"
    else
        prepare_domain_batch_metrics_for_run "$SOURCE_INDEX" "${BACKEND_FLAG#--backend }"
        run_benchmark_instance "$onnx" "$vnnlib" "$timeout" "default" "$BACKEND_FLAG"
        RESULT="$LAST_RESULT"
        ELAPSED="$LAST_ELAPSED"
        DOMAINS="$LAST_DOMAINS"
        ACTUAL_METHOD="$LAST_ACTUAL_METHOD"
        BACKEND_NAME="${BACKEND_FLAG#--backend }"
        if [[ -z "$BACKEND_NAME" ]]; then
            BACKEND_NAME="cpu"
        fi
        NOTES=$(notes_with_domain_batch_metrics "$PROVENANCE_NOTES" "$LAST_DOMAIN_BATCH_METRICS_JSONL")
        record_single_result "$RESULT" "$ELAPSED"

        echo "$RESULT (${ELAPSED}s, ${DOMAINS} domains)"
        append_backend_benchmark_row \
            "$REPORT" \
            "backend_benchmark_row_v1" \
            "vnncomp_single_backend" \
            "vnncomp_instance" \
            "$SUBJECT_ID" \
            "$COMPARISON_KEY" \
            "$CATEGORY" \
            "" \
            "$MODEL_PATH_REL" \
            "$PROPERTY_PATH_REL" \
            "$PRESET_PATH" \
            "$BACKEND_NAME" \
            "$timeout" \
            "$RESULT" \
            "$ACTUAL_METHOD" \
            "$ELAPSED" \
            "$DOMAINS" \
            "" \
            "" \
            "$NOTES"
    fi

done < "$BENCH_DIR/instances.csv"

echo ""
if [[ "$COMPARE_BACKENDS" == "true" ]]; then
    CPU_SCORE=$((CPU_VERIFIED + CPU_VIOLATED))
    WGPU_SCORE=$((WGPU_VERIFIED + WGPU_VIOLATED))
    echo "=== $CATEGORY Backend Comparison ==="
    echo "Total instances: $TOTAL"
    echo "CPU solved: $CPU_SCORE/$TOTAL"
    echo "CPU verified/violated: $CPU_VERIFIED / $CPU_VIOLATED"
    echo "CPU unknown-or-timeout/error: $CPU_UNKNOWN / $CPU_ERROR"
    echo "WGPU solved: $WGPU_SCORE/$TOTAL"
    echo "WGPU verified/violated: $WGPU_VERIFIED / $WGPU_VIOLATED"
    echo "WGPU unknown-or-timeout/error: $WGPU_UNKNOWN / $WGPU_ERROR"
    echo "Backend-only status divergence: $DIVERGED"
    echo "CPU wall time: ${CPU_TOTAL_TIME}s"
    echo "WGPU wall time: ${WGPU_TOTAL_TIME}s"
    printf '\nReport: %s\n' "$REPORT"
    if [[ "$DIVERGED" -gt 0 ]]; then
        printf '\n*** WARNING: backend-only status divergence observed; route to a soundness issue before treating this as performance data. ***\n'
    fi
else
    SCORE=$((VERIFIED + VIOLATED)); printf '\n=== %s Results ===\n' "$CATEGORY"
    echo "Total: $TOTAL"
    echo "Verified (UNSAT): $VERIFIED"
    echo "Violated (SAT): $VIOLATED"
    echo "Unknown/Timeout: $UNKNOWN"
    echo "Error: $ERROR"
    if [[ "$TOTAL" -gt 0 ]]; then
        echo "Score: $SCORE/$TOTAL ($(python3 -c "print(f'{$SCORE/$TOTAL*100:.1f}%')"))"
    else
        echo "Score: 0/0 (no instances)"
    fi
    echo "Total wall time: ${TOTAL_TIME}s"; printf '\nReport: %s\n' "$REPORT"

    # Auto-validate against reference if available
    REF_DIR="reports/benchmarks/reference"
    REF_MANIFEST="$REF_DIR/manifest.json"
    REF_FILE=""
    REF_LOOKUP_STATUS=0
    if [[ -f "$REF_MANIFEST" ]]; then
        if REF_FILE="$(reference_manifest_output_path "$REF_MANIFEST" "$CATEGORY" "$REF_DIR")"; then
            [[ -f "$REF_FILE" ]] || { echo "WARNING: Reference manifest entry for $CATEGORY points to missing file: $REF_FILE" >&2; REF_LOOKUP_STATUS=5; REF_FILE=""; }
        else
            REF_LOOKUP_STATUS=$?
            case "$REF_LOOKUP_STATUS" in
                2) echo "WARNING: Reference manifest is unreadable or invalid: $REF_MANIFEST" >&2 ;;
                4) echo "WARNING: Reference manifest entry for $CATEGORY has invalid output_path provenance; expected $REF_DIR/*.csv" >&2 ;;
                3) ;;
                *) echo "WARNING: Reference manifest lookup failed for $CATEGORY (exit $REF_LOOKUP_STATUS): $REF_MANIFEST" >&2 ;;
            esac
        fi
    fi
    if [[ -n "$REF_FILE" ]]; then
        printf '\n=== Result Validation ===\n'
        VALIDATION_EXIT=0
        bash "$SCRIPT_DIR/validate_vnncomp_results.sh" "$REPORT" "$REF_FILE" \
            || VALIDATION_EXIT=$?
        if [[ "$VALIDATION_EXIT" -eq 1 ]]; then
            echo ""
            echo "*** CRITICAL: Result validation found disagreements — potential soundness bug ***"
            exit 1
        elif [[ "$VALIDATION_EXIT" -ge 2 ]]; then
            echo "WARNING: Validation script error (exit $VALIDATION_EXIT)" >&2
        fi
    elif [[ ! -f "$REF_MANIFEST" || "$REF_LOOKUP_STATUS" -eq 3 ]]; then
        UNMANIFESTED_REF=""
        for f in "$REF_DIR/${CATEGORY}_"*.csv; do [[ -f "$f" ]] && { UNMANIFESTED_REF="$f"; break; }; done
        if [[ -n "${UNMANIFESTED_REF:-}" ]]; then
            printf '\n=== Result Validation ===\n'
            echo "NOTE: Skipping auto-validation against $UNMANIFESTED_REF because it is not declared in $REF_MANIFEST."
            echo "NOTE: Only manifest-backed reference CSVs are treated as authoritative for automatic soundness validation."
        fi
    fi
fi
