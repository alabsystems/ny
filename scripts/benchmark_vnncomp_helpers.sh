#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

validate_onnx_asset() {
    local onnx_path="$1"
    local file_desc

    if [[ ! -f "$onnx_path" ]]; then
        printf 'Error: benchmark ONNX asset missing: %s\n' "$onnx_path" > "$TMPOUT"
        return 1
    fi

    file_desc=$(LC_ALL=C file -bL "$onnx_path" 2>/dev/null || true)
    case "$file_desc" in
        *text*|*empty*)
            printf 'Error: benchmark ONNX asset is not binary (%s): %s\n' "$file_desc" "$onnx_path" > "$TMPOUT"
            return 1
            ;;
    esac

    return 0
}

process_is_live() {
    local pid="$1"
    local state

    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi

    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')
    if [[ -z "$state" ]]; then
        return 1
    fi

    case "$state" in
        Z*|*Z*)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

run_ny_with_watchdog() {
    local backend_flag="$1"
    local timeout="$2"

    start_ny_with_watchdog "$backend_flag" "$timeout"
    finish_ny_with_watchdog
    return "$LAST_EXIT_CODE"
}

start_ny_with_watchdog() {
    local backend_flag="$1"
    local timeout="${2%.*}"  # Truncate float to integer (e.g., 480.0 → 480)
    local watchdog_timeout=$((timeout + EXTERNAL_TIMEOUT_SLACK))
    local watchdog_polls=$((watchdog_timeout * 5))

    LAST_EXIT_CODE=0
    WATCHDOG_TIMEOUT_HIT=0
    WATCHDOG_TIMEOUT_LIMIT=$watchdog_timeout
    WATCHDOG_MARKER=$(mktemp)
    rm -f "$WATCHDOG_MARKER"

    set +e
    # shellcheck disable=SC2086
    "$NY_BIN" beta-crown "$ONNX_PATH" \
        --property "$VNNLIB_PATH" \
        $PRESET \
        $PGD_FLAG \
        $CATEGORY_EXTRA_FLAGS \
        $BRANCHING_FLAG \
        $backend_flag \
        $DOMAIN_BATCH_METRICS_FLAG \
        $VERIFIER_FLAG \
        --timeout "$timeout" > "$TMPOUT" 2>&1 &
    NY_PID=$!

    (
        local poll
        trap 'exit 0' TERM INT
        for ((poll = 0; poll < watchdog_polls; poll++)); do
            sleep 0.2
        done
        if process_is_live "$NY_PID"; then
            echo "$watchdog_timeout" > "$WATCHDOG_MARKER"
            kill -TERM "$NY_PID" 2>/dev/null || true
            sleep "$WATCHDOG_TERM_GRACE"
            if process_is_live "$NY_PID"; then
                kill -KILL "$NY_PID" 2>/dev/null || true
            fi
        fi
    ) &
    WATCHDOG_PID=$!
}

finish_ny_with_watchdog() {
    local exit_code
    local had_errexit=0

    case "$-" in
        *e*)
            had_errexit=1
            set +e
            ;;
    esac
    wait "$NY_PID"
    exit_code=$?
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null
    if [[ "$had_errexit" -eq 1 ]]; then
        set -e
    fi

    if [[ -s "$WATCHDOG_MARKER" ]] \
        && ! grep -q "Status:" "$TMPOUT" \
        && ! grep -qi "Timed out" "$TMPOUT" \
        && ! grep -q "Error:" "$TMPOUT"; then
        WATCHDOG_TIMEOUT_HIT=1
        printf '\nTimed out after external watchdog (%ss)\n' "$WATCHDOG_TIMEOUT_LIMIT" >> "$TMPOUT"
        exit_code=124
    fi

    rm -f "$WATCHDOG_MARKER"
    WATCHDOG_MARKER=""
    WATCHDOG_PID=""
    NY_PID=""
    LAST_EXIT_CODE="$exit_code"
    return 0
}

add_elapsed() {
    python3 -c "print(f'{float(\"$1\") + float(\"$2\"):.2f}')"
}

to_repo_relative_path() {
    local path="$1"

    case "$path" in
        "$PWD"/*)
            printf '%s\n' "${path#"$PWD"/}"
            ;;
        *)
            printf '%s\n' "$path"
            ;;
    esac
}

benchmark_suite_key() {
    local bench_root="$1"
    local normalized_root

    normalized_root=$(to_repo_relative_path "$bench_root")
    normalized_root="${normalized_root#./}"
    normalized_root="${normalized_root%/}"

    case "$normalized_root" in
        */benchmarks)
            local parent="${normalized_root%/benchmarks}"
            local suite="${parent##*/}"
            if [[ -n "$suite" ]]; then
                printf '%s\n' "$suite"
                return 0
            fi
            ;;
    esac

    printf '%s\n' "$normalized_root"
}

reference_manifest_output_path() {
    local manifest_path="$1"
    local category="$2"
    local allowed_prefix="${3:-reports/benchmarks/reference}"

    python3 - "$manifest_path" "$category" "$allowed_prefix" <<'PY'
import json
import os
import sys
from pathlib import Path

MANIFEST_INVALID = 2
CATEGORY_MISSING = 3
OUTPUT_PATH_INVALID = 4

manifest_path = Path(sys.argv[1])
category = sys.argv[2]
allowed_prefix = os.path.normpath(sys.argv[3]).rstrip(os.sep)

try:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
except (FileNotFoundError, OSError, json.JSONDecodeError):
    sys.exit(MANIFEST_INVALID)

reference_files = data.get("reference_files")
if not isinstance(reference_files, dict):
    sys.exit(MANIFEST_INVALID)

entry = reference_files.get(category)
if not isinstance(entry, dict):
    sys.exit(CATEGORY_MISSING)

output_path = entry.get("output_path")
if not isinstance(output_path, str) or not output_path:
    sys.exit(OUTPUT_PATH_INVALID)

normalized_output_path = os.path.normpath(output_path)
if os.path.isabs(normalized_output_path):
    sys.exit(OUTPUT_PATH_INVALID)
if normalized_output_path == allowed_prefix or not normalized_output_path.startswith(
    allowed_prefix + os.sep
):
    sys.exit(OUTPUT_PATH_INVALID)

print(normalized_output_path)
PY
}

benchmark_suite_key_from_path() {
    local path="$1"
    local normalized_path

    normalized_path=$(to_repo_relative_path "$path")
    normalized_path="${normalized_path#./}"

    case "$normalized_path" in
        benchmarks/*/benchmarks/*)
            normalized_path="${normalized_path#benchmarks/}"
            printf '%s\n' "${normalized_path%%/*}"
            ;;
        *)
            printf '%s\n' "$(dirname "$normalized_path")"
            ;;
    esac
}

benchmark_path_relative_to_category() {
    local category="$1"
    local path="$2"
    local repo_relative_path

    repo_relative_path=$(to_repo_relative_path "$path")
    repo_relative_path="${repo_relative_path#./}"

    case "$repo_relative_path" in
        "$category"/*)
            printf '%s\n' "${repo_relative_path#"$category"/}"
            ;;
        *"/$category/"*)
            printf '%s\n' "${repo_relative_path##*"/$category/"}"
            ;;
        *)
            printf '%s\n' "$repo_relative_path"
            ;;
    esac
}

# --- Binary provenance helpers (#4346) ---

# Compute the ny binary provenance: source classification, version, sha256.
# Sets globals: NY_SOURCE, NY_VERSION, NY_SHA256
compute_ny_provenance() {
    local ny_bin="$1"
    local explicit_env="${2:-}"  # "explicit" if caller set NY_BIN explicitly

    if [[ -n "$explicit_env" ]]; then
        NY_SOURCE="explicit"
    elif [[ "$ny_bin" == *"/worker_"*"/release/"* || "$ny_bin" == *"/worker_"*"/debug/"* ]]; then
        NY_SOURCE="worker-local"
    else
        NY_SOURCE="shared-default"
    fi

    NY_VERSION=$("$ny_bin" --version 2>/dev/null || echo "unknown")
    NY_SHA256=$(shasum -a 256 "$ny_bin" 2>/dev/null | awk '{print $1}' || echo "unknown")
}

# Format provenance tags for appending to the notes field.
# Usage: format_provenance_tags → "ny_source=...; ny_bin=...; ny_version=...; ny_sha256=..."
format_provenance_tags() {
    printf 'ny_source=%s; ny_bin=%s; ny_version=%s; ny_sha256=%s' \
        "$NY_SOURCE" "$NY_BIN" "$NY_VERSION" "$NY_SHA256"
}

# Append provenance tags to an existing notes string.
# Usage: append_provenance_to_notes "existing notes" → "existing notes; ny_source=..."
append_provenance_to_notes() {
    local existing_notes="$1"
    local provenance
    provenance=$(format_provenance_tags)

    if [[ -n "$existing_notes" ]]; then
        printf '%s; %s' "$existing_notes" "$provenance"
    else
        printf '%s' "$provenance"
    fi
}

benchmark_row_identity() {
    local suite_key="$1"
    local category="$2"
    local source_index="${3:-}"
    local onnx_rel="$4"
    local vnnlib_rel="$5"

    if [[ -n "$source_index" ]]; then
        printf '%s::%s::row=%s::%s::%s\n' \
            "$suite_key" \
            "$category" \
            "$source_index" \
            "$onnx_rel" \
            "$vnnlib_rel"
    else
        printf '%s::%s::%s::%s\n' \
            "$suite_key" \
            "$category" \
            "$onnx_rel" \
            "$vnnlib_rel"
    fi
}

benchmark_subject_id() {
    benchmark_row_identity "$@"
}

csv_escape_field() {
    local value="$1"

    value="${value//$'\r'/}"
    value="${value//\"/\"\"}"

    if [[ "$value" == *','* || "$value" == *'"'* || "$value" == *$'\n'* ]]; then
        printf '"%s"' "$value"
    else
        printf '%s' "$value"
    fi
}

write_csv_row() {
    local first=1
    local field

    for field in "$@"; do
        if [[ "$first" -eq 1 ]]; then
            first=0
        else
            printf ','
        fi
        csv_escape_field "$field"
    done
    printf '\n'
}

write_backend_benchmark_header() {
    local report="$1"

    write_csv_row \
        "schema_version" \
        "lane" \
        "subject_kind" \
        "subject_id" \
        "comparison_key" \
        "category" \
        "workload" \
        "model_path" \
        "property_path" \
        "preset_path" \
        "backend" \
        "timeout_seconds" \
        "status" \
        "actual_method" \
        "wall_seconds" \
        "domains_explored" \
        "output_width_sum" \
        "profile_artifact_path" \
        "notes" > "$report"
}

append_backend_benchmark_row() {
    local report="$1"
    shift

    write_csv_row "$@" >> "$report"
}

run_benchmark_attempts() {
    local timeout="$1"
    local backend_name="$2"
    local backend_flag="$3"
    local attempt=0

    LAST_EXIT_CODE=0
    if ! validate_onnx_asset "$ONNX_PATH"; then
        LAST_EXIT_CODE=1
        return
    fi

    while true; do
        attempt=$((attempt + 1))

        set +e
        run_ny_with_watchdog "$backend_flag" "$timeout"
        LAST_EXIT_CODE=$?
        set -e

        if grep -q "Status:" "$TMPOUT" || grep -qi "Timed out" "$TMPOUT" || grep -q "Error:" "$TMPOUT"; then
            return
        fi

        if [[ "$attempt" -gt "$MAX_SIGNAL_RETRIES" ]]; then
            return
        fi

        case "$LAST_EXIT_CODE" in
            124|137|143)
                echo "" >&2
                echo "  RETRY[$backend_name]: ny exited with code $LAST_EXIT_CODE before reporting a verdict (attempt $attempt/$((MAX_SIGNAL_RETRIES + 1)))" >&2
                ;;
            *)
                return
                ;;
        esac
    done
}

parse_benchmark_result() {
    local timeout="$1"
    local elapsed="$2"
    local backend_name="$3"
    local domains
    local actual_method

    domains=$(sed -n 's/.*Domains explored: \([0-9]*\).*/\1/p' "$TMPOUT" 2>/dev/null | tail -n1)
    LAST_DOMAINS="${domains:-0}"
    actual_method=$(sed -n 's/.*Actual method: \(.*\)$/\1/p' "$TMPOUT" 2>/dev/null | tail -n1)
    LAST_ACTUAL_METHOD="${actual_method:-}"

    if grep -q "Status: VERIFIED" "$TMPOUT"; then
        if python3 -c "exit(0 if float('$elapsed') <= float('$timeout') else 1)"; then
            LAST_RESULT="verified"
        else
            LAST_RESULT="timeout"
        fi
    elif grep -q "Status: VIOLATED" "$TMPOUT"; then
        LAST_RESULT="violated"
    elif grep -q "Status: POTENTIAL VIOLATION" "$TMPOUT" || grep -q "Status: UNKNOWN" "$TMPOUT"; then
        LAST_RESULT="unknown"
    elif grep -q "Status: TIMEOUT" "$TMPOUT" || grep -qi "Timed out" "$TMPOUT"; then
        LAST_RESULT="timeout"
    elif [[ "$LAST_EXIT_CODE" -eq 124 || "$LAST_EXIT_CODE" -eq 137 || "$LAST_EXIT_CODE" -eq 143 ]]; then
        LAST_RESULT="timeout"
        echo "" >&2
        echo "  NOTE[$backend_name]: counting exit code $LAST_EXIT_CODE without a verdict as timeout" >&2
    elif grep -q "Error:" "$TMPOUT"; then
        LAST_RESULT="error"
        echo "" >&2
        echo "  ERROR[$backend_name]: $(grep 'Error:' "$TMPOUT" | head -1)" >&2
    else
        LAST_RESULT="error"
        echo "" >&2
        echo "  DEBUG[$backend_name](exit=$LAST_EXIT_CODE): $(tail -3 "$TMPOUT")" >&2
    fi

    if [[ "$WATCHDOG_TIMEOUT_HIT" -eq 1 ]]; then
        echo "" >&2
        echo "  NOTE[$backend_name]: external watchdog enforced timeout after ${WATCHDOG_TIMEOUT_LIMIT}s without a verdict" >&2
    fi
}

run_benchmark_instance() {
    local onnx="$1"
    local vnnlib="$2"
    local timeout="$3"
    local backend_name="$4"
    local backend_flag="$5"
    local start_time
    local end_time

    ONNX_PATH="$BENCH_DIR/$onnx"
    VNNLIB_PATH="$BENCH_DIR/$vnnlib"

    start_time=$(python3 -c "import time; print(time.time())")
    run_benchmark_attempts "$timeout" "$backend_name" "$backend_flag"
    end_time=$(python3 -c "import time; print(time.time())")

    LAST_ELAPSED=$(python3 -c "print(f'{$end_time - $start_time:.2f}')")
    parse_benchmark_result "$timeout" "$LAST_ELAPSED" "$backend_name"
}
