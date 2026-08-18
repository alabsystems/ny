#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NY_BIN="${NY_BIN:-./target/release/ny}"
MAX_SIGNAL_RETRIES="${MAX_SIGNAL_RETRIES:-1}"
EXTERNAL_TIMEOUT_SLACK="${EXTERNAL_TIMEOUT_SLACK:-5}"
WATCHDOG_TERM_GRACE="${WATCHDOG_TERM_GRACE:-2}"
SAMPLE_BIN="${SAMPLE_BIN:-sample}"
SAMPLE_INTERVAL_MILLIS="${SAMPLE_INTERVAL_MILLIS:-10}"
SAMPLE_STUB_TEXT="${SAMPLE_STUB_TEXT:-}"

CATEGORY=""
MODEL=""
PROPERTY=""
PRESET_PATH=""
BACKEND=""
TIMEOUT=""
REPORT_PATH=""
OUTPUT_PATH=""
SAMPLE_EARLY_SECONDS="20"
SAMPLE_LATE_SECONDS="140"
SAMPLE_DURATION="10"
NOTES=""
BENCHMARK_SUITE=""
SOURCE_INDEX=""

usage() {
    cat <<'EOF'
Usage: profile_vnncomp_row.sh --category NAME --model PATH --property PATH --preset PATH --backend NAME --timeout SECONDS --report-path PATH --output PATH [--sample-early-seconds N] [--sample-late-seconds N] [--sample-duration N] [--notes TEXT]
       [--benchmark-suite ID] [--source-index N]
EOF
}

exact_alignment_identity_path() {
    local category="$1"
    local path="$2"
    local flag_name="$3"
    local normalized_path

    normalized_path=$(to_repo_relative_path "$path")
    normalized_path="${normalized_path#./}"

    case "$normalized_path" in
        onnx/*|vnnlib/*)
            printf '%s\n' "$normalized_path"
            ;;
        "$category"/*)
            printf '%s\n' "${normalized_path#"$category"/}"
            ;;
        *"/$category/"*)
            printf '%s\n' "${normalized_path##*"/$category/"}"
            ;;
        *)
            echo "Exact row alignment requires ${flag_name} to live under category '$category': $normalized_path" >&2
            return 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --category) CATEGORY="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --property) PROPERTY="$2"; shift 2 ;;
        --preset) PRESET_PATH="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --report-path) REPORT_PATH="$2"; shift 2 ;;
        --output) OUTPUT_PATH="$2"; shift 2 ;;
        --sample-early-seconds) SAMPLE_EARLY_SECONDS="$2"; shift 2 ;;
        --sample-late-seconds) SAMPLE_LATE_SECONDS="$2"; shift 2 ;;
        --sample-duration) SAMPLE_DURATION="$2"; shift 2 ;;
        --notes) NOTES="$2"; shift 2 ;;
        --benchmark-suite) BENCHMARK_SUITE="$2"; shift 2 ;;
        --source-index) SOURCE_INDEX="$2"; shift 2 ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

for required in CATEGORY MODEL PROPERTY PRESET_PATH BACKEND TIMEOUT REPORT_PATH OUTPUT_PATH; do
    if [[ -z "${!required}" ]]; then
        echo "Missing required argument: ${required,,}" >&2
        usage >&2
        exit 1
    fi
done

if ! python3 - \
    "$TIMEOUT" \
    "$SAMPLE_EARLY_SECONDS" \
    "$SAMPLE_LATE_SECONDS" \
    "$SAMPLE_DURATION" <<'PY'
import math
import re
import sys

decimal = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")
if re.fullmatch(r"[1-9][0-9]*", sys.argv[1]) is None:
    print(
        f"--timeout must be a positive integer, got: {sys.argv[1]!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)
values = {
    "--sample-early-seconds": (sys.argv[2], 0.0, "non-negative"),
    "--sample-late-seconds": (sys.argv[3], 0.0, "non-negative"),
    "--sample-duration": (sys.argv[4], 0.0, "positive"),
}
for option, (raw, minimum, description) in values.items():
    if decimal.fullmatch(raw) is None:
        print(
            f"{option} must be a finite {description} decimal number, got: {raw!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    value = float(raw)
    valid = math.isfinite(value) and (
        value >= minimum if option != "--sample-duration" else value > minimum
    )
    if not valid:
        print(
            f"{option} must be a finite {description} decimal number, got: {raw!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)
PY
then
    exit 1
fi

# shellcheck source=scripts/benchmark_vnncomp_helpers.sh
source "$SCRIPT_DIR/benchmark_vnncomp_helpers.sh"

if [[ -n "$SOURCE_INDEX" ]] && [[ ! "$SOURCE_INDEX" =~ ^[1-9][0-9]*$ ]]; then
    echo "--source-index must be a positive integer, got: $SOURCE_INDEX" >&2
    exit 1
fi
if [[ ! -x "$NY_BIN" ]]; then
    echo "Ny binary not found or not executable: $NY_BIN" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL" >&2
    exit 1
fi
if [[ ! -f "$PROPERTY" ]]; then
    echo "Property not found: $PROPERTY" >&2
    exit 1
fi
if [[ ! -f "$PRESET_PATH" ]]; then
    echo "Preset not found: $PRESET_PATH" >&2
    exit 1
fi
if [[ -z "$SAMPLE_STUB_TEXT" ]] && ! command -v "$SAMPLE_BIN" >/dev/null 2>&1; then
    echo "Profiler binary not found: $SAMPLE_BIN" >&2
    exit 1
fi

MODEL_PATH_REL="$(to_repo_relative_path "$MODEL")"
PROPERTY_PATH_REL="$(to_repo_relative_path "$PROPERTY")"
PRESET_PATH_REL="$(to_repo_relative_path "$PRESET_PATH")"
REPORT_PATH_REL="$(to_repo_relative_path "$REPORT_PATH")"
BENCHMARK_SUITE_KEY="${BENCHMARK_SUITE:-$(benchmark_suite_key_from_path "$MODEL_PATH_REL")}"
IDENTITY_MODEL_PATH="$MODEL_PATH_REL"
IDENTITY_PROPERTY_PATH="$PROPERTY_PATH_REL"

if [[ -n "$BENCHMARK_SUITE" && -n "$SOURCE_INDEX" ]]; then
    IDENTITY_MODEL_PATH="$(exact_alignment_identity_path "$CATEGORY" "$MODEL_PATH_REL" "--model")"
    IDENTITY_PROPERTY_PATH="$(exact_alignment_identity_path "$CATEGORY" "$PROPERTY_PATH_REL" "--property")"
fi

SUBJECT_ID="$(benchmark_row_identity "$BENCHMARK_SUITE_KEY" "$CATEGORY" "$SOURCE_INDEX" "$IDENTITY_MODEL_PATH" "$IDENTITY_PROPERTY_PATH")"
COMPARISON_KEY="$SUBJECT_ID"

mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$REPORT_PATH")"

PROFILE_RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
PROFILE_TMP_ROOT="${TMPDIR:-/tmp}"
PROFILE_TMP_PREFIX="${PROFILE_TMP_ROOT%/}/profile_vnncomp_row_${PROFILE_RUN_ID}"
TMPOUT="$(mktemp "${PROFILE_TMP_PREFIX}.stdout.XXXXXX")"
EARLY_SAMPLE_PATH="$(
    mktemp "${PROFILE_TMP_PREFIX}_early.sample.XXXXXX"
)"
LATE_SAMPLE_PATH="$(
    mktemp "${PROFILE_TMP_PREFIX}_late.sample.XXXXXX"
)"
EARLY_SAMPLE_CAPTURED=0
LATE_SAMPLE_CAPTURED=0

cleanup_profile_run() {
    if [[ -n "${WATCHDOG_PID:-}" ]]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    if process_is_live "${NY_PID:-}"; then
        kill -TERM "$NY_PID" 2>/dev/null || true
        sleep "${WATCHDOG_TERM_GRACE:-1}"
        kill -KILL "$NY_PID" 2>/dev/null || true
        wait "$NY_PID" 2>/dev/null || true
    fi
    if [[ -n "${WATCHDOG_MARKER:-}" ]]; then
        rm -f "$WATCHDOG_MARKER"
    fi
    if [[ "${EARLY_SAMPLE_CAPTURED:-0}" -eq 0 ]]; then
        rm -f "${EARLY_SAMPLE_PATH:-}"
    fi
    if [[ "${LATE_SAMPLE_CAPTURED:-0}" -eq 0 ]]; then
        rm -f "${LATE_SAMPLE_PATH:-}"
    fi
}
trap cleanup_profile_run EXIT

WATCHDOG_TIMEOUT_HIT=0
WATCHDOG_TIMEOUT_LIMIT=0
LAST_RESULT=""
LAST_ELAPSED=""
LAST_DOMAINS=""
LAST_ACTUAL_METHOD=""
LAST_EXIT_CODE=0
PGD_FLAG=""
BRANCHING_FLAG=""
CATEGORY_EXTRA_FLAGS=""
VERIFIER_FLAG=""
DOMAIN_BATCH_METRICS_ARGS=()
PRESET_ARGS=(--preset "$PRESET_PATH")
ONNX_PATH="$MODEL"
VNNLIB_PATH="$PROPERTY"

validate_onnx_asset "$ONNX_PATH"

capture_sample_output() {
    local target_pid="$1"
    local output_path="$2"
    local sample_timeout
    local elapsed="0.0"
    local sample_pid
    local sample_exit

    if [[ -n "$SAMPLE_STUB_TEXT" ]]; then
        printf '%s\n' "$SAMPLE_STUB_TEXT" > "$output_path"
        return 0
    fi

    sample_timeout=$(python3 -c \
        'import sys; print(max(float(sys.argv[1]) + 1.0, 2.0))' \
        "$SAMPLE_DURATION")

    set +e
    "$SAMPLE_BIN" "$target_pid" "$SAMPLE_DURATION" "$SAMPLE_INTERVAL_MILLIS" -mayDie -file "$output_path" &
    sample_pid=$!

    while process_is_live "$sample_pid"; do
        if ! python3 -c \
            'import sys; raise SystemExit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)' \
            "$elapsed" "$sample_timeout"; then
            kill -TERM "$sample_pid" 2>/dev/null || true
            sleep 0.2
            kill -KILL "$sample_pid" 2>/dev/null || true
            wait "$sample_pid" 2>/dev/null || true
            set -e
            return 124
        fi
        sleep 0.1
        elapsed=$(python3 -c \
            'import sys; print(f"{float(sys.argv[1]) + 0.1:.1f}")' \
            "$elapsed")
    done

    wait "$sample_pid"
    sample_exit=$?
    set -e
    return "$sample_exit"
}

write_backend_benchmark_header "$OUTPUT_PATH"

start_time=$(python3 -c 'import time; print(time.time())')
start_ny_with_watchdog "--backend $BACKEND" "$TIMEOUT"

sleep "$SAMPLE_EARLY_SECONDS"
if process_is_live "$NY_PID"; then
    if capture_sample_output "$NY_PID" "$EARLY_SAMPLE_PATH"; then
        EARLY_SAMPLE_CAPTURED=1
    fi
fi

sleep "$SAMPLE_LATE_SECONDS"
if process_is_live "$NY_PID"; then
    if capture_sample_output "$NY_PID" "$LATE_SAMPLE_PATH"; then
        LATE_SAMPLE_CAPTURED=1
    fi
fi

finish_ny_with_watchdog
end_time=$(python3 -c 'import time; print(time.time())')
LAST_ELAPSED=$(python3 -c \
    'import sys; print(f"{float(sys.argv[2]) - float(sys.argv[1]):.2f}")' \
    "$start_time" "$end_time")
parse_benchmark_result "$TIMEOUT" "$LAST_ELAPSED" "$BACKEND"

sample_notes="samples=none"
if [[ "$EARLY_SAMPLE_CAPTURED" -eq 1 && "$LATE_SAMPLE_CAPTURED" -eq 1 ]]; then
    sample_notes="samples=early,late"
elif [[ "$EARLY_SAMPLE_CAPTURED" -eq 1 ]]; then
    sample_notes="samples=early-only"
elif [[ "$LATE_SAMPLE_CAPTURED" -eq 1 ]]; then
    sample_notes="samples=late-only"
fi
if [[ -n "$NOTES" ]]; then
    sample_notes="$sample_notes; $NOTES"
fi

append_backend_benchmark_row \
    "$OUTPUT_PATH" \
    "backend_benchmark_row_v1" \
    "metaroom_host_profile" \
    "vnncomp_instance" \
    "$SUBJECT_ID" \
    "$COMPARISON_KEY" \
    "$CATEGORY" \
    "" \
    "$MODEL_PATH_REL" \
    "$PROPERTY_PATH_REL" \
    "$PRESET_PATH_REL" \
    "$BACKEND" \
    "$TIMEOUT" \
    "$LAST_RESULT" \
    "$LAST_ACTUAL_METHOD" \
    "$LAST_ELAPSED" \
    "$LAST_DOMAINS" \
    "" \
    "$REPORT_PATH_REL" \
    "$sample_notes"

printf 'profile_csv=%s\n' "$OUTPUT_PATH"
printf 'stdout_log=%s\n' "$TMPOUT"
if [[ "$EARLY_SAMPLE_CAPTURED" -eq 1 ]]; then
    printf 'early_sample=%s\n' "$EARLY_SAMPLE_PATH"
else
    printf 'early_sample=\n'
fi
if [[ "$LATE_SAMPLE_CAPTURED" -eq 1 ]]; then
    printf 'late_sample=%s\n' "$LATE_SAMPLE_PATH"
else
    printf 'late_sample=\n'
fi
