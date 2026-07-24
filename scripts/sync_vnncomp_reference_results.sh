#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Sync official VNN-COMP harness results into per-category reference CSVs.
#
# Usage:
#   bash scripts/sync_vnncomp_reference_results.sh [--repo-root PATH] [--tool TOOL] [--year YEAR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_REPO_ROOT="~/vnncomp2025_results-ref"
BENCHMARK_ROOT="~/vnncomp2025_benchmarks-ref"
TOOL="alpha_beta_crown"
YEAR=2025
REF_DIR="reports/benchmarks/reference"
MANIFEST_PATH="$REF_DIR/manifest.json"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --repo-root PATH       Source vnncomp*_results checkout (default: ~/vnncomp2025_results-ref)"
    echo "  --benchmark-root PATH  Benchmark assets checkout (default: ~/vnncomp2025_benchmarks-ref)"
    echo "  --tool TOOL            Tool lane to sync (default: alpha_beta_crown)"
    echo "  --year YEAR            Benchmark year recorded in manifest (default: 2025)"
    echo "  --help                 Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-root) SOURCE_REPO_ROOT="$2"; shift 2 ;;
        --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
        --tool) TOOL="$2"; shift 2 ;;
        --year) YEAR="$2"; shift 2 ;;
        --help|-h)
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

SOURCE_REPO_ROOT="${SOURCE_REPO_ROOT/#\~/$HOME}"
BENCHMARK_ROOT="${BENCHMARK_ROOT/#\~/$HOME}"
SOURCE_TOOL_DIR="$SOURCE_REPO_ROOT/$TOOL"

if [[ ! -d "$SOURCE_TOOL_DIR" ]]; then
    echo "ERROR: source tool directory not found: $SOURCE_TOOL_DIR" >&2
    exit 1
fi

mkdir -p "$REF_DIR"
rm -f "$REF_DIR"/*_"$TOOL".csv

TMP_RECORDS=$(mktemp)
trap 'rm -f "$TMP_RECORDS"' EXIT

shopt -s nullglob
RESULT_FILES=("$SOURCE_TOOL_DIR"/*/results.csv)
shopt -u nullglob

if [[ ${#RESULT_FILES[@]} -eq 0 ]]; then
    echo "ERROR: no results.csv files found under $SOURCE_TOOL_DIR" >&2
    exit 1
fi

for results_csv in "${RESULT_FILES[@]}"; do
    category="$(basename "$(dirname "$results_csv")")"
    "$SCRIPT_DIR/generate_vnncomp_reference.sh" --from-harness "$results_csv" "$TOOL" >/dev/null

    output_path="$REF_DIR/${category}_${TOOL}.csv"
    if [[ ! -f "$output_path" ]]; then
        echo "ERROR: converter did not produce expected output: $output_path" >&2
        exit 1
    fi

    instance_count=$(( $(wc -l < "$output_path") - 1 ))
    printf '%s\t%s\t%s\t%s\n' \
        "$category" \
        "$results_csv" \
        "$output_path" \
        "$instance_count" >> "$TMP_RECORDS"
    echo "Synced ${category} -> ${output_path} (${instance_count} instances)"
done

SOURCE_COMMIT="null"
if git -C "$SOURCE_REPO_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    SOURCE_COMMIT="$(git -C "$SOURCE_REPO_ROOT" rev-parse HEAD)"
fi

BENCHMARK_COMMIT="null"
if [[ -d "$BENCHMARK_ROOT" ]] && git -C "$BENCHMARK_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    BENCHMARK_COMMIT="$(git -C "$BENCHMARK_ROOT" rev-parse HEAD)"
fi

python3 - "$TMP_RECORDS" "$MANIFEST_PATH" "$SOURCE_REPO_ROOT" "$SOURCE_COMMIT" "$TOOL" "$YEAR" "$BENCHMARK_ROOT" "$BENCHMARK_COMMIT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

records_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
source_repo_root = sys.argv[3]
source_commit = None if sys.argv[4] == "null" else sys.argv[4]
tool = sys.argv[5]
year = int(sys.argv[6])
benchmark_repo_root = sys.argv[7]
benchmark_commit = None if sys.argv[8] == "null" else sys.argv[8]

records = []
for line in records_path.read_text(encoding="utf-8").splitlines():
    category, source_path, output_path, instance_count = line.split("\t")
    records.append(
        {
            "category": category,
            "source_path": source_path,
            "output_path": output_path,
            "instance_count": int(instance_count),
        }
    )

records.sort(key=lambda record: record["category"])
payload = {
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source_repo_root": source_repo_root,
    "source_commit": source_commit,
    "benchmark_repo_root": benchmark_repo_root,
    "benchmark_commit": benchmark_commit,
    "tool": tool,
    "year": year,
    "categories": [record["category"] for record in records],
    "reference_files": {
        record["category"]: {
            "source_path": record["source_path"],
            "output_path": record["output_path"],
            "instance_count": record["instance_count"],
        }
        for record in records
    },
}

manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "Manifest: $MANIFEST_PATH"
