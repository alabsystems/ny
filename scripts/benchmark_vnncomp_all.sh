#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Run VNN-COMP benchmarks across the current runnable default categories.
# Wraps scripts/benchmark_vnncomp.sh for each category and aggregates results.
#
# Usage:
#   scripts/benchmark_vnncomp_all.sh [--categories "cat1 cat2"] [--limit N] [--dry-run] [--start-from CAT] [--year YEAR] [--competition-wrapper|--diagnostic-beta-crown]

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
EXECUTION_SURFACE_EXPLICIT=false
COMPETITION_WRAPPER=false
CATEGORIES=""
SKIPPED_LIST=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --categories) CATEGORIES="$2"; CUSTOM_CATEGORIES=true; shift 2 ;;
        --limit) LIMIT_FLAG="--limit $2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --start-from) START_FROM="$2"; shift 2 ;;
        --year) YEAR="$2"; BENCH_ROOT="benchmarks/vnncomp${YEAR}/benchmarks"; shift 2 ;;
        --competition-wrapper) COMPETITION_WRAPPER=true; EXECUTION_SURFACE_EXPLICIT=true; shift ;;
        --diagnostic-beta-crown) COMPETITION_WRAPPER=false; EXECUTION_SURFACE_EXPLICIT=true; shift ;;
        --help|-h)
            echo "Usage: scripts/benchmark_vnncomp_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --categories \"cat1 cat2\"  Run specific categories (default: current runnable set)"
            echo "  --limit N                 Limit instances per category"
            echo "  --dry-run                 List categories and instance counts without running"
            echo "  --start-from CAT          Skip categories before CAT"
            echo "  --year YEAR               Benchmark year (default: 2025)"
            echo "  --competition-wrapper     Use the scored ny vnncomp protocol"
            echo "  --diagnostic-beta-crown   Use the legacy direct beta-crown diagnostic surface"
            echo "  --help                    Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Select a bounded, current-head default for the requested competition year.
# The attempted and skipped sets partition that year's official categories.
# Explicit --categories remains an opt-in probe and is filtered out of skipped
# metadata below.
case "$YEAR" in
    2025)
        DEFAULT_CATEGORIES="malbeware sat_relu cersyve lsnc_relu relusplitter collins_rul_cnn_2022 safenlp_2024 cora_2024 dist_shift_2023 metaroom_2023 linearizenn_2024 nn4sys ml4acopf_2024 cgan_2023 acasxu_2023 tllverifybench_2023"
        SKIPPED_LIST="vggnet16_2022:large-model CROWN probe is runtime-limited (measured hundreds of seconds per row)
collins_aerospace_benchmark:large YOLO-family probe is outside the bounded default sweep
vit_2023:representative transformer rows remain initial-bound/runtime limited
traffic_signs_recognition_2023:current-head probe exhausted its budget after entering search
soundnessbench:current-head probe remains PGD/runtime limited
cifar100_2024:large ResNet rows require the dedicated long-running GPU lane
tinyimagenet_2024:large ResNet rows remain initial-bound/runtime limited
cctsdb_yolo_2023:cell-enumeration probes are intentionally opt-in because rows can take about a minute
yolo_2023:current-head TinyYOLO rows remain initial-bound/runtime limited"
        ;;
    2026)
        # Official 2026 selection: 24 regular + 6 extended categories.
        # The bounded default runs the 17 current-head single-network categories;
        # every other selected category remains an explicit opt-in below.
        DEFAULT_CATEGORIES="malbeware sat_relu cersyve lsnc_relu relusplitter_2026 collins_rul_cnn_2022 safenlp_2024 cora_2024 dist_shift_2023 metaroom_2023 linearizenn_2024 nn4sys ml4acopf_2024 cgan2026 acasxu_2023 tllverifybench_2023 adaptive_cruise_control_non_linear_2026"
        SKIPPED_LIST="cctsdb_yolo_2023:cell-enumeration probes are intentionally opt-in because rows can take about a minute
challenging_certified_training_2026:wide certified-training CNN rows require the dedicated long-running GPU lane
cifar100_2024:large ResNet rows require the dedicated long-running GPU lane
collins_aerospace_benchmark:large YOLO-family probe is outside the bounded default sweep
isomorphic_acasxu_2026:native relational support exists, but the full official-budget sweep remains dedicated and opt-in
monotonic_acasxu_2026:native relational support exists, but the full official-budget sweep remains dedicated and opt-in
smart_turn_multimodal_2026:the current native wrapper does not support the complete VNN-LIB 2.0 multimodal property and operator surface
soundnessbench_2026:current-head probe needs an official-budget dedicated lane
tinyimagenet_2024:large ResNet rows remain initial-bound/runtime limited
traffic_signs_recognition_2023:current-head probe exhausted its budget after entering search
vggnet16_2022:large-model CROWN probe is runtime-limited (measured hundreds of seconds per row)
vit_2023:representative transformer rows remain initial-bound/runtime limited
yolo_2023:current-head TinyYOLO rows remain initial-bound/runtime limited"
        ;;
    *)
        if ! $CUSTOM_CATEGORIES; then
            echo "No default category set for VNN-COMP $YEAR; pass --categories explicitly" >&2
            exit 2
        fi
        DEFAULT_CATEGORIES=""
        SKIPPED_LIST=""
        ;;
esac

if [[ "$YEAR" == "2026" && "$EXECUTION_SURFACE_EXPLICIT" == "false" ]]; then
    # A 2026 summary is a potential score claim, so exercise the same native
    # wrapper, typed preset routing, timeout reserve, and trusted SAT gate as a
    # submission. Direct beta-crown remains explicitly available for diagnostic
    # backend/strategy work and remains the 2025 compatibility default.
    COMPETITION_WRAPPER=true
fi

if ! $CUSTOM_CATEGORIES; then
    CATEGORIES="$DEFAULT_CATEGORIES"
fi

# Resolve the exact benchmark version selected for 2026. All regular categories
# offer VNN-LIB 1.0. Four extended categories are 2.0-only.
benchmark_version() {
    local category="$1"
    if [[ "$YEAR" != "2026" ]]; then
        return 0
    fi
    case "$category" in
        adaptive_cruise_control_non_linear_2026|isomorphic_acasxu_2026|monotonic_acasxu_2026|smart_turn_multimodal_2026)
            printf '%s\n' "2.0"
            ;;
        *)
            printf '%s\n' "1.0"
            ;;
    esac
}

resolve_category_dir() {
    local category="$1"
    local category_root="$BENCH_ROOT/$category"
    local version

    version=$(benchmark_version "$category")
    if [[ -n "$version" && -f "$category_root/$version/instances.csv" ]]; then
        printf '%s\n' "$category_root/$version"
        return 0
    fi
    if [[ "$YEAR" == "2026" ]]; then
        # Do not silently substitute an unversioned/stale local copy when the
        # official 2026 selection names an exact VNN-LIB version.
        return 1
    fi
    if [[ -f "$category_root/instances.csv" ]]; then
        printf '%s\n' "$category_root"
        return 0
    fi
    return 1
}

# Mirror the native VNN-COMP resolver: prefer this year's exact/base preset,
# then carry a 2025 preset forward when 2026 has no override.
resolve_preset_path() {
    local category="$1"
    local base_category
    local dir
    local candidate
    local dirs

    base_category=$(printf '%s\n' "$category" | sed 's/_20[0-9][0-9]$//')
    if [[ "$YEAR" == "2026" ]]; then
        dirs="configs/vnncomp26 configs/vnncomp25"
    else
        dirs="configs/vnncomp25"
    fi
    for dir in $dirs; do
        for candidate in "$category" "$base_category"; do
            if [[ -f "$dir/$candidate.yaml" ]]; then
                printf '%s\n' "$dir/$candidate.yaml"
                return 0
            fi
            if [[ "$candidate" == "$category" ]]; then
                continue
            fi
        done
    done
    return 1
}

# A category cannot be both attempted and skipped. This matters for explicit
# runtime-limited probes, whose results would otherwise be double-counted by
# summary/dashboard consumers.
FILTERED_SKIPPED_LIST=""
while IFS=':' read -r name reason; do
    [[ -n "$name" ]] || continue
    if [[ " $CATEGORIES " == *" $name "* ]]; then
        continue
    fi
    FILTERED_SKIPPED_LIST="${FILTERED_SKIPPED_LIST}${name}:${reason}
"
done <<< "$SKIPPED_LIST"
SKIPPED_LIST="${FILTERED_SKIPPED_LIST%$'\n'}"

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
if [[ "$COMPETITION_WRAPPER" == "true" ]]; then
    echo "Surface:    ny vnncomp competition wrapper (score projections remain modeled-only; organizer results not bound)"
    COMPETITION_WRAPPER_FLAG="--competition-wrapper"
else
    echo "Surface:    beta-crown diagnostic (not eligible for score claims)"
    COMPETITION_WRAPPER_FLAG=""
fi
echo ""

# Dry-run mode: list categories and instance counts
if $DRY_RUN; then
    echo "=== Supported Categories ==="
    for category in $CATEGORIES; do
        CATEGORY_DIR=$(resolve_category_dir "$category" || true)
        if [[ -n "$CATEGORY_DIR" ]]; then
            CSV="$CATEGORY_DIR/instances.csv"
            COUNT=$(wc -l < "$CSV" | tr -d ' ')
            PRESET=$(resolve_preset_path "$category" || printf '%s\n' "none")
            printf "  %-30s %4d instances  preset: %s\n" "$category" "$COUNT" "$PRESET"
        else
            printf "  %-30s      NOT FOUND\n" "$category"
        fi
    done
    echo ""
    echo "=== Skipped Categories ==="
    echo "$SKIPPED_LIST" | while IFS=':' read -r name reason; do
        [[ -n "$name" ]] || continue
        printf "  %-30s %s\n" "$name" "$reason"
    done | sort
    exit 0
fi

# Determine run scope: full vs partial
# A full run uses the default category set with no subsetting flags.
TRACKER_YEAR="${TRACKER_YEAR:-$YEAR}"

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

    CATEGORY_DIR=$(resolve_category_dir "$category" || true)
    if [[ -z "$CATEGORY_DIR" ]]; then
        echo "WARNING: $category — no instances.csv found, skipping"
        FAILED_LIST="${FAILED_LIST}${category}:0:no instances.csv found
"
        continue
    fi
    CSV="$CATEGORY_DIR/instances.csv"
    PRESET_PATH_OVERRIDE=$(resolve_preset_path "$category" || true)

    echo ""
    echo "########################################"
    echo "# Category: $category"
    echo "########################################"

    # Capture child output to parse the exact Report: path
    CHILD_LOG=$(mktemp)
    CHILD_EXIT=0
    # shellcheck disable=SC2086
    BENCH_ROOT="$BENCH_ROOT" \
    BENCH_DIR="$CATEGORY_DIR" \
    PRESET_PATH_OVERRIDE="$PRESET_PATH_OVERRIDE" \
        bash "$SCRIPT_DIR/benchmark_vnncomp.sh" "$category" $LIMIT_FLAG $COMPETITION_WRAPPER_FLAG 2>&1 \
        | tee "$CHILD_LOG" || CHILD_EXIT=${PIPESTATUS[0]}

    if [[ $CHILD_EXIT -ne 0 ]]; then
        echo "WARNING: $category exited with error (exit $CHILD_EXIT), continuing"
        FAILED_LIST="${FAILED_LIST}${category}:${CHILD_EXIT}:non-zero exit code
"
        rm -f "$CHILD_LOG"
        continue
    fi

    # Parse the exact report path from child output instead of globbing
    REPORT_PATH=$(grep -m1 '^Report: ' "$CHILD_LOG" | sed 's/^Report: //' || true)
    rm -f "$CHILD_LOG"

    if [[ -z "$REPORT_PATH" ]]; then
        echo "WARNING: $category — no Report: line in output, skipping"
        FAILED_LIST="${FAILED_LIST}${category}:0:no report path in output
"
        continue
    fi

    # Resolve symlinks and `..` before containment testing. A lexical prefix
    # check accepts paths such as reports/benchmarks/../escaped.csv.
    REPORT_CANONICAL=""
    if REPORT_CANONICAL=$(python3 - "$REPORT_PATH" "$REPORT_DIR" <<'PY'
import sys
from pathlib import Path

try:
    candidate = Path(sys.argv[1]).resolve()
    report_root = Path(sys.argv[2]).resolve(strict=True)
    candidate.relative_to(report_root)
except (OSError, RuntimeError, ValueError):
    raise SystemExit(3)
if not candidate.is_file():
    raise SystemExit(2)
print(candidate)
PY
    ); then
        CSV_FILES+=("$REPORT_CANONICAL")
    else
        REPORT_VALIDATION_EXIT=$?
        if [[ "$REPORT_VALIDATION_EXIT" -eq 2 ]]; then
            echo "WARNING: $category — report path does not exist: $REPORT_PATH"
            FAILED_LIST="${FAILED_LIST}${category}:0:report file missing
"
        else
            echo "WARNING: $category — report path outside expected directory: $REPORT_PATH"
            FAILED_LIST="${FAILED_LIST}${category}:0:report path outside reports/benchmarks/
"
        fi
    fi
done

OVERALL_END=$(python3 -c "import time; print(time.time())")
WALL_TIME=$(python3 -c "print(f'{$OVERALL_END - $OVERALL_START:.1f}')")

echo ""
echo "========================================"
echo "All categories complete. Wall time: ${WALL_TIME}s"
echo "========================================"

# Build skipped JSON for aggregation
SKIPPED_JSON=$(NY_BENCH_SKIPPED_LIST="$SKIPPED_LIST" python3 -c '
import json
import os
skipped = {}
for line in os.environ["NY_BENCH_SKIPPED_LIST"].splitlines():
    if not line.strip():
        continue
    name, reason = line.split(":", 1)
    skipped[name.strip()] = reason.strip()
print(json.dumps(skipped))
')

# Build failed JSON for aggregation
FAILED_JSON=$(NY_BENCH_FAILED_LIST="$FAILED_LIST" python3 -c '
import json
import os
failed = {}
for line in os.environ["NY_BENCH_FAILED_LIST"].splitlines():
    if not line.strip():
        continue
    parts = line.split(":", 2)
    if len(parts) == 3:
        failed[parts[0].strip()] = {
            "exit_code": int(parts[1].strip()),
            "reason": parts[2].strip(),
        }
print(json.dumps(failed))
')

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
