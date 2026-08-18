#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Orchestrator for #4359 NVIDIA/Vulkan validation.
# Captures host facts, runs ny-gpu workloads, runs compare-backends,
# optionally runs alpha-beta-CROWN, and writes a JSON manifest.
#
# Usage:
#   scripts/run_nvidia_vulkan_validation.sh [--skip-reference] [--output-dir DIR]
#
# Environment overrides (for testing):
#   NVIDIA_SMI_CMD      - command for nvidia-smi (default: nvidia-smi)
#   CARGO_CMD           - command for cargo (default: cargo)
#   NY_BIN           - path to ny binary (default: ./target/release/ny)
#   BENCHMARK_SCRIPT    - path to benchmark_vnncomp.sh (default: scripts/benchmark_vnncomp.sh)
#   RUSTC_CMD           - command for rustc version check (default: rustc)
#   CARGO_VERSION_CMD   - command for cargo version check (default: cargo)
#   TIME_CMD            - command for /usr/bin/time (default: /usr/bin/time)
#   ABCROWN_DIR         - path to an alpha-beta-CROWN complete_verifier checkout
#                         (github.com/Verified-Intelligence/alpha-beta-CROWN);
#                         required unless --skip-reference is passed
#   PYTHON_CMD          - Python command for alpha-beta-CROWN (default: python3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configurable commands (overridable for testing)
NVIDIA_SMI_CMD="${NVIDIA_SMI_CMD:-nvidia-smi}"
CARGO_CMD="${CARGO_CMD:-cargo}"
NY_BIN="${NY_BIN:-./target/release/ny}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-${REPO_ROOT}/scripts/benchmark_vnncomp.sh}"
TIME_CMD="${TIME_CMD:-/usr/bin/time}"
ABCROWN_DIR="${ABCROWN_DIR:-}"
PYTHON_CMD="${PYTHON_CMD:-python3}"

# Parse arguments
SKIP_REFERENCE=false
OUTPUT_DIR="${REPO_ROOT}/reports/benchmarks"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-reference) SKIP_REFERENCE=true; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ABCROWN_DIR has no default: the reference comparator needs an explicit checkout.
if [[ "${SKIP_REFERENCE}" != "true" && -z "${ABCROWN_DIR}" ]]; then
    echo "ERROR: ABCROWN_DIR is not set. Point it at an alpha-beta-CROWN complete_verifier" >&2
    echo "       checkout (github.com/Verified-Intelligence/alpha-beta-CROWN), or pass --skip-reference." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

HOST_INFO_FILE="${OUTPUT_DIR}/issue-4359-nvidia-vulkan-host-info.txt"
MEASURE_LOG="${OUTPUT_DIR}/issue-4359-nvidia-vulkan-measure.log"
MEASURE_CSV="${OUTPUT_DIR}/issue-4359-nvidia-vulkan-crown-backward.csv"
MANIFEST_FILE="${OUTPUT_DIR}/issue-4359-nvidia-vulkan-manifest.json"
CERSYVE_CSV=""
METAROOM_CSV=""
REFERENCE_BLOCKER=""
REFERENCE_REAL_SECONDS=""
BLOCKER=""
VERDICT="blocked"
CHILD_LOG=""
VULKAN_OK=false
ADAPTER_LINE=""

cleanup_validation_tempfiles() {
    if [[ -n "${CHILD_LOG:-}" ]]; then
        rm -f "$CHILD_LOG"
    fi
}
trap cleanup_validation_tempfiles EXIT

write_validation_manifest() {
    python3 -I - \
        "$MANIFEST_FILE" \
        "$VERDICT" \
        "$BLOCKER" \
        "$VULKAN_OK" \
        "$ADAPTER_LINE" \
        "$CERSYVE_CSV" \
        "$METAROOM_CSV" \
        "$REFERENCE_BLOCKER" \
        "$REFERENCE_REAL_SECONDS" \
        "$HOST_INFO_FILE" \
        "$MEASURE_LOG" \
        "$MEASURE_CSV" <<'PY'
import json
import math
import os
import sys
import tempfile
from pathlib import Path

(
    manifest_path,
    verdict,
    blocker,
    vulkan_ok,
    adapter_line,
    cersyve_csv,
    metaroom_csv,
    reference_blocker,
    reference_seconds,
    host_info,
    measure_log,
    measure_csv,
) = sys.argv[1:]

reference_value = None
if reference_seconds:
    try:
        reference_value = float(reference_seconds)
    except ValueError as error:
        raise SystemExit(f"invalid reference timing {reference_seconds!r}: {error}")
    if not math.isfinite(reference_value) or reference_value < 0:
        raise SystemExit(f"invalid reference timing {reference_seconds!r}")


def artifact_name(value):
    return Path(value).name if value else None


payload = {
    "schema": "nvidia_vulkan_validation_manifest_v1",
    "verdict": verdict,
    "blocker": blocker or None,
    "host_info_path": artifact_name(host_info),
    "measure_log_path": artifact_name(measure_log),
    "measure_csv_path": artifact_name(measure_csv),
    "vulkan_confirmed": vulkan_ok == "true",
    "adapter_line": adapter_line,
    "compare_backends_cersyve_csv": artifact_name(cersyve_csv),
    "compare_backends_metaroom_csv": artifact_name(metaroom_csv),
    "reference_blocker": reference_blocker or None,
    "reference_cersyve_real_seconds": reference_value,
}

destination = Path(manifest_path)
temporary_name = ""
try:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_name = temporary.name
        json.dump(payload, temporary, indent=2, ensure_ascii=False)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary_name, destination)
except BaseException:
    if temporary_name:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    raise
PY
}

run_compare_backends() {
    local category="$1"
    local start_at="$2"
    local limit="$3"
    local child_exit=0
    local report_count
    local report_path

    CHILD_LOG=$(mktemp "${OUTPUT_DIR}/.${category}.compare.XXXXXX")
    NY_BIN="${NY_BIN}" REPORT_DIR="${OUTPUT_DIR}" \
        "${BENCHMARK_SCRIPT}" "$category" --compare-backends \
        --start-at "$start_at" --limit "$limit" 2>&1 \
        | tee "$CHILD_LOG" || child_exit=${PIPESTATUS[0]}
    if [[ "$child_exit" -ne 0 ]]; then
        rm -f "$CHILD_LOG"
        CHILD_LOG=""
        return "$child_exit"
    fi

    report_count=$(grep -c '^Report: ' "$CHILD_LOG" || true)
    if [[ "$report_count" -ne 1 ]]; then
        echo "ERROR: ${category} emitted ${report_count} exact Report: lines" >&2
        rm -f "$CHILD_LOG"
        CHILD_LOG=""
        return 1
    fi
    report_path=$(grep -m1 '^Report: ' "$CHILD_LOG" | sed 's/^Report: //')
    rm -f "$CHILD_LOG"
    CHILD_LOG=""

    # Resolve symlinks and traversal before accepting the child artifact. The
    # canonical file must be a direct child of this run's output directory and
    # retain the exact category report basename contract.
    if ! report_path=$(python3 -I - "$report_path" "$OUTPUT_DIR" "$category" <<'PY'
import sys
from pathlib import Path

candidate_text, output_text, category = sys.argv[1:]
try:
    candidate = Path(candidate_text).resolve(strict=True)
    output = Path(output_text).resolve(strict=True)
    relative = candidate.relative_to(output)
except (OSError, RuntimeError, ValueError):
    raise SystemExit(1)

prefix = f"{category}_compare_backends_"
if (
    relative.parent != Path(".")
    or not candidate.is_file()
    or not relative.name.startswith(prefix)
    or not relative.name.endswith(".csv")
    or len(relative.name) <= len(prefix) + len(".csv")
):
    raise SystemExit(1)
print(candidate)
PY
    ); then
        echo "ERROR: ${category} report escaped or violated the validation output contract: ${report_path}" >&2
        return 1
    fi
    COMPARE_REPORT="$report_path"
}

echo "=== NVIDIA/Vulkan Validation Orchestrator ==="
echo "Output directory: ${OUTPUT_DIR}"

# --- Step 1: Capture host facts ---
echo ""
echo "--- Step 1: Capturing host facts ---"
{
    echo "=== Host Facts ==="
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo ""
    echo "--- uname ---"
    uname -a
    echo ""
    echo "--- rustc ---"
    "${RUSTC_CMD:-rustc}" -Vv 2>&1 || echo "rustc not found"
    echo ""
    echo "--- cargo ---"
    "${CARGO_VERSION_CMD:-cargo}" -V 2>&1 || echo "cargo not found"
    echo ""
    echo "--- nvidia-smi ---"
    "${NVIDIA_SMI_CMD}" --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 || echo "nvidia-smi not available"
    echo ""
    echo "--- nvidia-smi full ---"
    "${NVIDIA_SMI_CMD}" 2>&1 || echo "nvidia-smi not available"
} > "$HOST_INFO_FILE" 2>&1
echo "Wrote: ${HOST_INFO_FILE}"

# --- Step 2: Run measure_crown_backward_workloads ---
echo ""
echo "--- Step 2: Running measure_crown_backward_workloads --graph-engine-only ---"
if ! rm -f -- "${MEASURE_CSV}"; then
    BLOCKER="could not clear the prior measurement CSV artifact"
    REFERENCE_BLOCKER="blocked: measurement setup failed, skipped all downstream steps"
    write_validation_manifest
    echo "Wrote: ${MANIFEST_FILE}"
    exit 1
fi

MEASURE_PIPE_STATUSES=()
if RUST_LOG=ny_gpu=info,ny_propagate=info \
    "${CARGO_CMD}" run -p ny-gpu --release --example measure_crown_backward_workloads -- \
    --graph-engine-only \
    --output "${MEASURE_CSV}" \
    2>&1 | tee "${MEASURE_LOG}"; then
    :
else
    MEASURE_PIPE_STATUSES=("${PIPESTATUS[@]}")
    BLOCKER="measurement command failed (cargo exit ${MEASURE_PIPE_STATUSES[0]:-unknown}, tee exit ${MEASURE_PIPE_STATUSES[1]:-unknown})"
    REFERENCE_BLOCKER="blocked: measurement command failed, skipped all downstream steps"
    write_validation_manifest
    echo "Wrote: ${MANIFEST_FILE}"
    exit 1
fi

# Fail-closed: verify Vulkan backend
if grep -q 'backend: Vulkan' "${MEASURE_LOG}"; then
    VULKAN_OK=true
    echo "PASS: wgpu backend is Vulkan"
else
    echo "FAIL: wgpu backend is NOT Vulkan (or not detected in log)"
fi

if ADAPTER_LINE=$(grep -m1 'wgpu adapter:' "${MEASURE_LOG}"); then
    echo "Adapter: ${ADAPTER_LINE}"
else
    ADAPTER_LINE=""
    echo "WARNING: no 'wgpu adapter:' line found in measurement log"
fi

if [[ ! -s "${MEASURE_CSV}" ]]; then
    BLOCKER="measurement command completed without a non-empty CSV artifact"
    REFERENCE_BLOCKER="blocked: measurement CSV missing or empty, skipped all downstream steps"
    write_validation_manifest
    echo "Wrote: ${MANIFEST_FILE}"
    exit 1
fi

if [[ "${VULKAN_OK}" != "true" ]]; then
    echo ""
    echo "FATAL: Vulkan backend not confirmed. Writing blocked manifest."
    BLOCKER="wgpu did not select Vulkan backend — check driver stack"
    REFERENCE_BLOCKER="blocked: Vulkan not confirmed, skipped all downstream steps"
    write_validation_manifest
    echo "Wrote: ${MANIFEST_FILE}"
    exit 1
fi

# --- Step 3: Run compare-backends ---
echo ""
echo "--- Step 3: Running compare-backends (cersyve + metaroom_2023) ---"

COMPARE_REPORT=""
if run_compare_backends cersyve 3 2; then
    CERSYVE_CSV="$COMPARE_REPORT"
else
    BLOCKER="cersyve compare-backends command/report failed"
fi

COMPARE_REPORT=""
if run_compare_backends metaroom_2023 10 1; then
    METAROOM_CSV="$COMPARE_REPORT"
else
    if [[ -n "$BLOCKER" ]]; then
        BLOCKER="${BLOCKER}; metaroom_2023 compare-backends command/report failed"
    else
        BLOCKER="metaroom_2023 compare-backends command/report failed"
    fi
fi

echo "Cersyve CSV: ${CERSYVE_CSV:-not found}"
echo "Metaroom CSV: ${METAROOM_CSV:-not found}"
if [[ -z "${CERSYVE_CSV}" || -z "${METAROOM_CSV}" ]]; then
    if [[ -z "$BLOCKER" ]]; then
        BLOCKER="compare-backends completed without both expected CSV artifacts"
    fi
fi

# --- Step 4: Optional reference comparator ---
if [[ "${SKIP_REFERENCE}" == "true" ]]; then
    REFERENCE_BLOCKER="skipped: --skip-reference flag set"
    echo ""
    echo "--- Step 4: Skipping reference comparator (--skip-reference) ---"
elif [[ ! -d "${ABCROWN_DIR}" ]]; then
    REFERENCE_BLOCKER="blocked: alpha-beta-CROWN directory not found at ${ABCROWN_DIR}"
    echo ""
    echo "--- Step 4: Reference comparator blocked (directory not found) ---"
else
    echo ""
    echo "--- Step 4: Running alpha-beta-CROWN reference (cersyve, start=2, end=4) ---"
    REFERENCE_OUTPUT="${OUTPUT_DIR}/issue-4359-abcrown-cersyve-reference.log"
    if "${TIME_CMD}" -p "${PYTHON_CMD}" "${ABCROWN_DIR}/abcrown.py" \
        --config "${ABCROWN_DIR}/exp_configs/vnncomp25/cersyve.yaml" \
        --start 2 --end 4 \
        > "${REFERENCE_OUTPUT}" 2>&1; then
        # Parse 'real' time from /usr/bin/time -p output (last line with 'real')
        if grep -q '^real ' "${REFERENCE_OUTPUT}"; then
            REFERENCE_REAL_SECONDS=$(grep '^real ' "${REFERENCE_OUTPUT}" | tail -1 | awk '{print $2}')
            echo "Reference real time: ${REFERENCE_REAL_SECONDS}s"
        else
            REFERENCE_BLOCKER="blocked: /usr/bin/time -p did not produce 'real' output"
        fi
    else
        REFERENCE_BLOCKER="blocked: alpha-beta-CROWN exited with error"
    fi
fi

# --- Step 5: Write manifest ---
echo ""
echo "--- Step 5: Writing manifest ---"

# Determine verdict state
VERDICT="blocked"
if [[ -n "${BLOCKER}" || -n "${REFERENCE_BLOCKER}" ]]; then
    VERDICT="blocked"
elif [[ -n "${REFERENCE_REAL_SECONDS}" ]]; then
    # We have both ny and reference data — verdict will be computed by renderer
    VERDICT="pending"
fi

write_validation_manifest

echo "Wrote: ${MANIFEST_FILE}"
echo ""
echo "=== Orchestrator complete ==="
echo "Next: python3 scripts/render_nvidia_vulkan_validation_report.py --manifest ${MANIFEST_FILE} --output-dir ${OUTPUT_DIR}"
