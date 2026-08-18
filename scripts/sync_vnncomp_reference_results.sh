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

SOURCE_REPO_ROOT="$HOME/vnncomp2025_results-ref"
BENCHMARK_ROOT="$HOME/vnncomp2025_benchmarks-ref"
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

if [[ ! "$TOOL" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "ERROR: invalid tool component: $TOOL" >&2
    exit 1
fi
if [[ ! "$YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "ERROR: year must be four ASCII digits: $YEAR" >&2
    exit 1
fi
if [[ ! -d "$SOURCE_REPO_ROOT" ]]; then
    echo "ERROR: source repository root not found: $SOURCE_REPO_ROOT" >&2
    exit 1
fi
SOURCE_REPO_ROOT="$(cd "$SOURCE_REPO_ROOT" && pwd -P)"
if [[ -d "$BENCHMARK_ROOT" ]]; then
    BENCHMARK_ROOT="$(cd "$BENCHMARK_ROOT" && pwd -P)"
fi
SOURCE_TOOL_DIR="$SOURCE_REPO_ROOT/$TOOL"
if [[ ! -d "$SOURCE_TOOL_DIR" ]]; then
    echo "ERROR: source tool directory not found: $SOURCE_TOOL_DIR" >&2
    exit 1
fi

shopt -s nullglob
RESULT_FILES=("$SOURCE_TOOL_DIR"/*/results.csv)
shopt -u nullglob

if [[ ${#RESULT_FILES[@]} -eq 0 ]]; then
    echo "ERROR: no results.csv files found under $SOURCE_TOOL_DIR" >&2
    exit 1
fi

REF_PARENT="$(dirname "$REF_DIR")"
mkdir -p "$REF_PARENT"
if [[ -L "$REF_DIR" ]]; then
    echo "ERROR: refusing to replace symlinked reference directory: $REF_DIR" >&2
    exit 1
fi

# The converter writes a hard-coded reports/benchmarks/reference path. Run it
# from a private same-filesystem work tree, validate the entire candidate
# snapshot there, and only then replace the live directory.
STAGING_ROOT=$(mktemp -d "$REF_PARENT/.reference-sync.XXXXXX")
STAGING_WORK="$STAGING_ROOT/work"
STAGED_REF_DIR="$STAGING_WORK/$REF_DIR"
BACKUP_ROOT=""
PUBLICATION_BACKED_UP=0
mkdir -p "$STAGED_REF_DIR"
if [[ -d "$REF_DIR" ]]; then
    cp -a "$REF_DIR"/. "$STAGED_REF_DIR"/
fi
rm -f "$STAGED_REF_DIR"/*_"$TOOL".csv "$STAGED_REF_DIR/manifest.json"

TMP_RECORDS="$STAGING_ROOT/records.tsv"
: > "$TMP_RECORDS"

cleanup_sync() {
    local exit_code=$?
    if [[ "$PUBLICATION_BACKED_UP" -eq 1 && ! -e "$REF_DIR" ]] \
        && [[ -d "${BACKUP_ROOT:-}/reference" ]]; then
        mv "$BACKUP_ROOT/reference" "$REF_DIR" || true
    fi
    if [[ -n "${STAGING_ROOT:-}" && -d "$STAGING_ROOT" ]]; then
        rm -rf -- "$STAGING_ROOT"
    fi
    if [[ -n "${BACKUP_ROOT:-}" && -d "$BACKUP_ROOT" ]]; then
        rm -rf -- "$BACKUP_ROOT"
    fi
    return "$exit_code"
}
trap cleanup_sync EXIT

for results_csv in "${RESULT_FILES[@]}"; do
    category="$(basename "$(dirname "$results_csv")")"
    if [[ ! "$category" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        echo "ERROR: invalid category component: $category" >&2
        exit 1
    fi
    (
        cd "$STAGING_WORK"
        "$SCRIPT_DIR/generate_vnncomp_reference.sh" \
            --from-harness "$results_csv" "$TOOL" >/dev/null
    )

    staged_output="$STAGED_REF_DIR/${category}_${TOOL}.csv"
    output_path="$REF_DIR/${category}_${TOOL}.csv"
    if [[ ! -f "$staged_output" || -L "$staged_output" ]]; then
        echo "ERROR: converter did not produce expected output: $output_path" >&2
        exit 1
    fi

    instance_count=$(( $(wc -l < "$staged_output") - 1 ))
    printf '%s\t%s\t%s\t%s\n' \
        "$category" \
        "$results_csv" \
        "$output_path" \
        "$instance_count" >> "$TMP_RECORDS"
    echo "Synced ${category} -> ${output_path} (${instance_count} instances)"
done

exact_checkout_provenance() {
    local candidate="$1"
    shift
    local candidate_root
    local git_root
    local status
    local required
    local relative

    [[ -d "$candidate" ]] || {
        printf 'null\tnull\n'
        return
    }
    candidate_root="$(cd "$candidate" && pwd -P)"
    git_root="$(git -C "$candidate" rev-parse --show-toplevel 2>/dev/null)" || {
        printf 'null\tnull\n'
        return
    }
    git_root="$(cd "$git_root" && pwd -P)"
    if [[ "$candidate_root" != "$git_root" ]]; then
        printf 'null\tnull\n'
        return
    fi

    status="$(git -C "$candidate" status --porcelain --untracked-files=all 2>/dev/null)" || {
        printf 'null\tnull\n'
        return
    }
    if [[ -n "$status" ]]; then
        printf 'null\ttrue\n'
        return
    fi
    for required in "$@"; do
        case "$required" in
            "$candidate_root"/*)
                relative="${required#"$candidate_root"/}"
                ;;
            *)
                printf 'null\ttrue\n'
                return
                ;;
        esac
        if ! git -C "$candidate" ls-files --error-unmatch -- "$relative" >/dev/null 2>&1; then
            printf 'null\ttrue\n'
            return
        fi
    done

    if git -C "$candidate" rev-parse --verify HEAD >/dev/null 2>&1; then
        printf '%s\tfalse\n' "$(git -C "$candidate" rev-parse --verify HEAD)"
    else
        printf 'null\tnull\n'
    fi
}

# `git -C nested/path rev-parse HEAD` walks upward into an enclosing checkout.
# Only attribute a commit when the supplied root is itself that checkout's
# top-level directory, clean, and (for results) tracks every consumed artifact.
# Dirty/non-checkout trees get a null commit plus an explicit dirty state.
SOURCE_PROVENANCE="$(exact_checkout_provenance "$SOURCE_REPO_ROOT" "${RESULT_FILES[@]}")"
IFS=$'\t' read -r SOURCE_COMMIT SOURCE_DIRTY <<< "$SOURCE_PROVENANCE"
BENCHMARK_PROVENANCE="$(exact_checkout_provenance "$BENCHMARK_ROOT")"
IFS=$'\t' read -r BENCHMARK_COMMIT BENCHMARK_DIRTY <<< "$BENCHMARK_PROVENANCE"

STAGED_MANIFEST="$STAGED_REF_DIR/manifest.json"
python3 - \
    "$TMP_RECORDS" \
    "$STAGED_MANIFEST" \
    "$STAGED_REF_DIR" \
    "$SOURCE_REPO_ROOT" \
    "$SOURCE_COMMIT" \
    "$SOURCE_DIRTY" \
    "$TOOL" \
    "$YEAR" \
    "$BENCHMARK_ROOT" \
    "$BENCHMARK_COMMIT" \
    "$BENCHMARK_DIRTY" <<'PY'
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

records_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
staged_ref_dir = Path(sys.argv[3])
source_repo_root = sys.argv[4]
source_commit = None if sys.argv[5] == "null" else sys.argv[5]
source_dirty_text = sys.argv[6]
tool = sys.argv[7]
year = int(sys.argv[8])
benchmark_repo_root = sys.argv[9]
benchmark_commit = None if sys.argv[10] == "null" else sys.argv[10]
benchmark_dirty_text = sys.argv[11]


def optional_bool(value):
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    raise SystemExit(f"invalid checkout dirty state: {value!r}")


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
if not records:
    raise SystemExit("reference snapshot contains no categories")
categories = [record["category"] for record in records]
if len(categories) != len(set(categories)):
    raise SystemExit("reference snapshot contains duplicate categories")

expected_names = {f"{category}_{tool}.csv" for category in categories}
actual_names = {path.name for path in staged_ref_dir.glob(f"*_{tool}.csv")}
if actual_names != expected_names:
    raise SystemExit(
        "reference snapshot file set mismatch: "
        f"missing={sorted(expected_names - actual_names)} "
        f"extra={sorted(actual_names - expected_names)}"
    )
for record in records:
    csv_path = staged_ref_dir / Path(record["output_path"]).name
    if csv_path.is_symlink() or not csv_path.is_file():
        raise SystemExit(f"invalid staged reference file: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    if not rows or rows[0] != ["model", "property", "result"]:
        raise SystemExit(f"invalid reference CSV header: {csv_path}")
    if len(rows) == 1 or any(len(row) != 3 or not all(row) for row in rows[1:]):
        raise SystemExit(f"invalid or empty reference CSV body: {csv_path}")
    actual_count = len(rows) - 1
    if actual_count != record["instance_count"]:
        raise SystemExit(
            f"reference CSV count mismatch for {csv_path}: "
            f"{actual_count} != {record['instance_count']}"
        )

payload = {
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source_repo_root": source_repo_root,
    "source_commit": source_commit,
    "source_dirty": optional_bool(source_dirty_text),
    "benchmark_repo_root": benchmark_repo_root,
    "benchmark_commit": benchmark_commit,
    "benchmark_dirty": optional_bool(benchmark_dirty_text),
    "tool": tool,
    "year": year,
    "categories": categories,
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

# Publish only after every conversion and the complete snapshot validation
# succeed. The same-filesystem directory rename prevents partial CSV sets; if
# the second rename fails, the EXIT trap restores the prior snapshot.
BACKUP_ROOT=$(mktemp -d "$REF_PARENT/.reference-backup.XXXXXX")
if [[ -d "$REF_DIR" ]]; then
    mv "$REF_DIR" "$BACKUP_ROOT/reference"
    PUBLICATION_BACKED_UP=1
fi
if ! mv "$STAGED_REF_DIR" "$REF_DIR"; then
    echo "ERROR: failed to publish staged reference snapshot" >&2
    exit 1
fi
PUBLICATION_BACKED_UP=0
rm -rf -- "$BACKUP_ROOT"
BACKUP_ROOT=""

echo "Manifest: $MANIFEST_PATH"
