#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Validate ny results against VNN-COMP reference answers.
# Detects incorrect results (verified vs violated mismatch) which
# cost -150 points under 2025 VNN-COMP scoring.
#
# Usage:
#   scripts/validate_vnncomp_results.sh <ny_csv> <reference_csv> [--format ny|harness]
#
# CSV formats:
#   ny:   model,property,timeout,result,elapsed[,domains]
#   ny_v1: backend_benchmark_row_v1 rows from benchmark_vnncomp.sh single-backend mode
#   harness: category,onnx_path,vnnlib_path,prepare_runtime,result,runtime
#
# Result normalization:
#   unsat -> verified    sat -> violated
#   verified -> verified violated -> violated
#   All others (timeout, unknown, error, ...) -> unknown
#
# Instance keys are <dir>/<stem> for both the model and the property, e.g.
# medical/perturbations_0 | medical/hyperrectangle_984. The trailing directory is
# part of the key because benchmarks reuse basenames across categories.
#
# Exit codes:
#   0 = no disagreements (may have coverage gaps)
#   1 = CRITICAL disagreements found (potential soundness bug)
#   2 = invalid arguments, file not found, or ambiguous instance keys

set -euo pipefail

die() { echo "ERROR: $1" >&2; exit 2; }

usage() {
    echo "Usage: $0 <ny_csv> <reference_csv> [--format ny|harness]"
    echo ""
    echo "Compares ny results against reference answers."
    echo "Reports agreements, critical disagreements, and coverage gaps."
    exit 2
}

# Normalize result string to canonical form
normalize_result() {
    local r="$1"
    case "$r" in
        unsat|verified|holds)   echo "verified" ;;
        sat|violated|falsified) echo "violated" ;;
        *)                      echo "unknown" ;;
    esac
}

# Reduce a path to its last directory plus file name, discarding the leading
# prefix so that differently-rooted CSVs still line up. The directory is retained
# because benchmarks reuse basenames across categories — safenlp ships both
# onnx/medical/perturbations_0.onnx and onnx/ruarobot/perturbations_0.onnx, with
# matching vnnlib names — and a basename-only key scores one instance's verdict
# against the other instance's answer.
path_tail() {
    local p="$1"
    local base="${p##*/}"
    local dir="${p%/*}"
    if [[ "$dir" == "$p" || -z "$dir" ]]; then
        echo "$base"
    else
        echo "${dir##*/}/$base"
    fi
}

# Extract model key: strip .onnx/.onnx.gz extension, keep <dir>/<stem>
model_key() {
    local m="$1"
    m="${m%.gz}"
    m="${m%.onnx}"
    path_tail "$m"
}

# Extract property key: strip .vnnlib/.vnnlib.gz extension, keep <dir>/<stem>
property_key() {
    local p="$1"
    p="${p%.gz}"
    p="${p%.vnnlib}"
    path_tail "$p"
}

# List keys that carry more than one verdict within a single key file. Such a key
# cannot be compared: whichever row a lookup happens to reach decides agreement.
ambiguous_keys() {
    awk -F'|' '
        { key = $1 "|" $2 }
        !(key in seen) { seen[key] = $3; next }
        seen[key] != $3 && !(key in flagged) {
            flagged[key] = 1
            print key "  (" seen[key] " vs " $3 ")"
        }
    ' "$1"
}

# Parse arguments
[[ $# -lt 2 ]] && usage
NY_CSV="$1"
REF_CSV="$2"
shift 2
REF_FORMAT="auto"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --format) REF_FORMAT="$2"; shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -f "$NY_CSV" ]] || die "File not found: $NY_CSV"
[[ -f "$REF_CSV" ]] || die "File not found: $REF_CSV"

# Auto-detect reference format from header
if [[ "$REF_FORMAT" == "auto" ]]; then
    HEADER=$(head -1 "$REF_CSV")
    NCOLS=$(echo "$HEADER" | awk -F',' '{print NF}')
    if [[ "$NCOLS" -eq 3 ]] && echo "$HEADER" | grep -q "^model,property,result"; then
        REF_FORMAT="simple"  # model,property,result (3 columns)
    elif echo "$HEADER" | grep -q "^schema_version,"; then
        REF_FORMAT="ny_v1"
    elif echo "$HEADER" | grep -q "^model,property"; then
        REF_FORMAT="ny"   # model,property,timeout,result,elapsed[,domains]
    else
        REF_FORMAT="harness"
    fi
fi

REPORT_DIR="reports/benchmarks/result_validation"
mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$REPORT_DIR/validation_${TIMESTAMP}.txt"

# Build associative arrays via temp files
NY_TMPF=$(mktemp)
REF_TMPF=$(mktemp)
trap 'rm -f "$NY_TMPF" "$REF_TMPF"' EXIT

NY_HEADER=$(head -1 "$NY_CSV" | tr -d '\r')
if echo "$NY_HEADER" | grep -q "^schema_version,"; then
    tail -n +2 "$NY_CSV" | tr -d '\r' | while IFS=',' read -r _ lane _ _ _ _ _ model_path property_path _ _ _ status _; do
        if [[ "$lane" != "vnncomp_single_backend" ]]; then
            continue
        fi
        m=$(model_key "$model_path")
        p=$(property_key "$property_path")
        r=$(normalize_result "$status")
        echo "$m|$p|$r"
    done | sort > "$NY_TMPF"
else
    tail -n +2 "$NY_CSV" | tr -d '\r' | while IFS=',' read -r col1 col2 col3 col4 rest; do
        m=$(model_key "$col1")
        p=$(property_key "$col2")
        r=$(normalize_result "$col4")
        echo "$m|$p|$r"
    done | sort > "$NY_TMPF"
fi

# Parse reference results
if [[ "$REF_FORMAT" == "simple" ]]; then
    # Simple 3-column: model,property,result
    tail -n +2 "$REF_CSV" | tr -d '\r' | while IFS=',' read -r col1 col2 col3; do
        m=$(model_key "$col1")
        p=$(property_key "$col2")
        r=$(normalize_result "$col3")
        echo "$m|$p|$r"
    done | sort > "$REF_TMPF"
elif [[ "$REF_FORMAT" == "ny" ]]; then
    # Ny format: model,property,timeout,result,...
    tail -n +2 "$REF_CSV" | tr -d '\r' | while IFS=',' read -r col1 col2 col3 col4 rest; do
        m=$(model_key "$col1")
        p=$(property_key "$col2")
        r=$(normalize_result "$col4")
        echo "$m|$p|$r"
    done | sort > "$REF_TMPF"
elif [[ "$REF_FORMAT" == "ny_v1" ]]; then
    # backend_benchmark_row_v1 single-backend rows
    tail -n +2 "$REF_CSV" | tr -d '\r' | while IFS=',' read -r _ lane _ _ _ _ _ model_path property_path _ _ _ status _; do
        if [[ "$lane" != "vnncomp_single_backend" ]]; then
            continue
        fi
        m=$(model_key "$model_path")
        p=$(property_key "$property_path")
        r=$(normalize_result "$status")
        echo "$m|$p|$r"
    done | sort > "$REF_TMPF"
elif [[ "$REF_FORMAT" == "harness" ]]; then
    # VNN-COMP harness: category,onnx_path,vnnlib_path,prepare_runtime,result,runtime
    tail -n +2 "$REF_CSV" | tr -d '\r' | while IFS=',' read -r _ onnx vnnlib _ result _; do
        m=$(model_key "$onnx")
        p=$(property_key "$vnnlib")
        r=$(normalize_result "$result")
        echo "$m|$p|$r"
    done | sort > "$REF_TMPF"
else
    die "Unknown format: $REF_FORMAT (expected simple, ny, ny_v1, or harness)"
fi

# Refuse to score anything once a key is ambiguous. Two instances can share a
# <dir>/<stem> key when a run mixes benchmark versions (ml4acopf_2023 and
# ml4acopf_2024 both hold onnx/14_ieee_ml4acopf.onnx with matching vnnlib names);
# if their verdicts differ, a lookup would silently pick one and could score a
# real verified↔violated flip as agreement. Validate one benchmark at a time.
NY_AMBIGUOUS=$(ambiguous_keys "$NY_TMPF")
REF_AMBIGUOUS=$(ambiguous_keys "$REF_TMPF")
if [[ -n "$NY_AMBIGUOUS" || -n "$REF_AMBIGUOUS" ]]; then
    echo "ERROR: conflicting verdicts share an instance key; no comparison is trustworthy." >&2
    if [[ -n "$NY_AMBIGUOUS" ]]; then
        echo "  ny ($NY_CSV):" >&2
        printf '    %s\n' "${NY_AMBIGUOUS//$'\n'/$'\n    '}" >&2
    fi
    if [[ -n "$REF_AMBIGUOUS" ]]; then
        echo "  reference ($REF_CSV):" >&2
        printf '    %s\n' "${REF_AMBIGUOUS//$'\n'/$'\n    '}" >&2
    fi
    exit 2
fi

# Compare results: write report to file, then display + check exit code
AGREE=0
DISAGREE_CRITICAL=0
DISAGREE_MILD=0
COVERAGE_GAP=0
NY_ONLY=0
TOTAL_MATCHED=0

exec 3>"$REPORT"

tee_line() { echo "$1"; echo "$1" >&3; }

tee_line "=== VNN-COMP Result Validation ==="
tee_line "Ny results: $NY_CSV ($(wc -l < "$NY_TMPF") instances)"
tee_line "Reference:     $REF_CSV ($(wc -l < "$REF_TMPF") instances)"
tee_line "Ref format:    $REF_FORMAT"
tee_line ""
tee_line "--- Per-instance comparison ---"

# For each ny result, find matching reference. Taking the first match is safe only
# because the ambiguity guard above has already rejected keys carrying more than
# one verdict, so every remaining match for a key reports the same answer.
while IFS='|' read -r g_model g_prop g_result; do
    ref_result=$(awk -F'|' -v m="${g_model}" -v p="${g_prop}" '$1==m && $2==p {print $3; exit}' "$REF_TMPF")

    if [[ -z "$ref_result" ]]; then
        NY_ONLY=$((NY_ONLY + 1))
        continue
    fi

    TOTAL_MATCHED=$((TOTAL_MATCHED + 1))

    if [[ "$g_result" == "$ref_result" ]]; then
        AGREE=$((AGREE + 1))
    elif [[ "$g_result" == "verified" && "$ref_result" == "violated" ]] || \
         [[ "$g_result" == "violated" && "$ref_result" == "verified" ]]; then
        DISAGREE_CRITICAL=$((DISAGREE_CRITICAL + 1))
        tee_line "CRITICAL: $g_model / $g_prop — ny=$g_result ref=$ref_result"
    elif [[ "$g_result" == "unknown" && "$ref_result" != "unknown" ]]; then
        COVERAGE_GAP=$((COVERAGE_GAP + 1))
    else
        DISAGREE_MILD=$((DISAGREE_MILD + 1))
        tee_line "MISMATCH: $g_model / $g_prop — ny=$g_result ref=$ref_result"
    fi
done < "$NY_TMPF"

# Count reference-only instances
REF_ONLY=0
while IFS='|' read -r r_model r_prop _; do
    if ! awk -F'|' -v m="${r_model}" -v p="${r_prop}" '$1==m && $2==p {found=1; exit} END {exit !found}' "$NY_TMPF"; then
        REF_ONLY=$((REF_ONLY + 1))
    fi
done < "$REF_TMPF"

tee_line ""
tee_line "--- Summary ---"
tee_line "Matched instances: $TOTAL_MATCHED"
tee_line "  Agree:              $AGREE"
tee_line "  CRITICAL mismatch:  $DISAGREE_CRITICAL  (verified↔violated — requires replay classification)"
tee_line "  Mild mismatch:      $DISAGREE_MILD  (e.g., verified vs unknown)"
tee_line "  Coverage gap:       $COVERAGE_GAP  (ny=unknown, ref has answer)"
tee_line "Ny-only:         $NY_ONLY  (no reference for comparison)"
tee_line "Reference-only:     $REF_ONLY  (ny didn't run these)"
tee_line ""

if [[ "$DISAGREE_CRITICAL" -gt 0 ]]; then
    tee_line "*** FAIL: $DISAGREE_CRITICAL critical disagreement(s) detected ***"
    tee_line "*** Each requires replay classification — run audit_vnncomp_counterexamples.py ***"

    # Invoke classifier when ny CSV is backend_benchmark_row_v1 format
    if echo "$NY_HEADER" | grep -q "^schema_version,"; then
        CLASSIFIER_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/audit_vnncomp_counterexamples.py"
        CLASSIFIER_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        CLASSIFIER_JSON="$REPORT_DIR/classifier_${CLASSIFIER_TIMESTAMP}.json"
        if [[ -f "$CLASSIFIER_SCRIPT" ]] && command -v python3 >/dev/null 2>&1; then
            tee_line ""
            tee_line "Running replay classifier..."
            CLASSIFIER_STATUS=0
            if python3 "$CLASSIFIER_SCRIPT" \
                    --ny-csv "$NY_CSV" \
                    --reference-csv "$REF_CSV" \
                    --ny-binary "${NY_BINARY:-./target/release/ny}" \
                    --output-json "$CLASSIFIER_JSON" \
                    --timeout "${CLASSIFIER_TIMEOUT:-120}" \
                    2>&1 | while IFS= read -r line; do tee_line "$line"; done
            then
                CLASSIFIER_STATUS=0
            else
                CLASSIFIER_STATUS=$?
            fi
            case "$CLASSIFIER_STATUS" in
                0) ;;
                1)
                    tee_line "Classifier status: unresolved replay failure(s) (exit 1)"
                    ;;
                *)
                    tee_line "Classifier status: execution failed (exit $CLASSIFIER_STATUS)"
                    ;;
            esac
            if [[ -f "$CLASSIFIER_JSON" ]]; then
                tee_line "Classifier artifact: $CLASSIFIER_JSON"
            fi
        fi
    fi
else
    tee_line "PASS: No critical disagreements."
    if [[ "$COVERAGE_GAP" -gt 0 ]]; then
        tee_line "NOTE: $COVERAGE_GAP instance(s) where reference solved but ny returned unknown."
    fi
fi

exec 3>&-
echo ""
echo "Report saved: $REPORT"

if [[ "$DISAGREE_CRITICAL" -gt 0 ]]; then
    exit 1
fi
exit 0
