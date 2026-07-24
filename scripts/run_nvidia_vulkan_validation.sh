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
#   PYTHON_CMD          - python command (default: python)

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
PYTHON_CMD="${PYTHON_CMD:-python}"

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

HOST_INFO_FILE="${OUTPUT_DIR}/issue-4359-nvidia-vulkan-host-info.txt"
MEASURE_LOG="${OUTPUT_DIR}/issue-4359-nvidia-vulkan-measure.log"
MEASURE_CSV="${OUTPUT_DIR}/issue-4359-nvidia-vulkan-crown-backward.csv"
MANIFEST_FILE="${OUTPUT_DIR}/issue-4359-nvidia-vulkan-manifest.json"

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
    ${RUSTC_CMD:-rustc} -Vv 2>&1 || echo "rustc not found"
    echo ""
    echo "--- cargo ---"
    ${CARGO_VERSION_CMD:-cargo} -V 2>&1 || echo "cargo not found"
    echo ""
    echo "--- nvidia-smi ---"
    ${NVIDIA_SMI_CMD} --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 || echo "nvidia-smi not available"
    echo ""
    echo "--- nvidia-smi full ---"
    ${NVIDIA_SMI_CMD} 2>&1 || echo "nvidia-smi not available"
} > "$HOST_INFO_FILE" 2>&1
echo "Wrote: ${HOST_INFO_FILE}"

# --- Step 2: Run measure_crown_backward_workloads ---
echo ""
echo "--- Step 2: Running measure_crown_backward_workloads --graph-engine-only ---"
RUST_LOG=ny_gpu=info,ny_propagate=info \
    ${CARGO_CMD} run -p ny-gpu --release --example measure_crown_backward_workloads -- \
    --graph-engine-only \
    --output "${MEASURE_CSV}" \
    2>&1 | tee "${MEASURE_LOG}"

# Fail-closed: verify Vulkan backend
VULKAN_OK=false
if grep -q 'backend: Vulkan' "${MEASURE_LOG}"; then
    VULKAN_OK=true
    echo "PASS: wgpu backend is Vulkan"
else
    echo "FAIL: wgpu backend is NOT Vulkan (or not detected in log)"
fi

ADAPTER_LINE=""
if grep -q 'wgpu adapter:' "${MEASURE_LOG}"; then
    ADAPTER_LINE=$(grep 'wgpu adapter:' "${MEASURE_LOG}" | head -1)
    echo "Adapter: ${ADAPTER_LINE}"
else
    echo "WARNING: no 'wgpu adapter:' line found in measurement log"
fi

if [[ "${VULKAN_OK}" != "true" ]]; then
    echo ""
    echo "FATAL: Vulkan backend not confirmed. Writing blocked manifest."
    cat > "$MANIFEST_FILE" <<MANIFEST_EOF
{
    "schema": "nvidia_vulkan_validation_manifest_v1",
    "verdict": "blocked",
    "blocker": "wgpu did not select Vulkan backend — check driver stack",
    "host_info_path": "$(basename "$HOST_INFO_FILE")",
    "measure_log_path": "$(basename "$MEASURE_LOG")",
    "measure_csv_path": "$(basename "$MEASURE_CSV")",
    "vulkan_confirmed": false,
    "adapter_line": "${ADAPTER_LINE}",
    "compare_backends_cersyve_csv": null,
    "compare_backends_metaroom_csv": null,
    "reference_blocker": "blocked: Vulkan not confirmed, skipped all downstream steps",
    "reference_cersyve_real_seconds": null
}
MANIFEST_EOF
    echo "Wrote: ${MANIFEST_FILE}"
    exit 1
fi

# --- Step 3: Run compare-backends ---
echo ""
echo "--- Step 3: Running compare-backends (cersyve + metaroom_2023) ---"

# Find the compare-backends CSV outputs by timestamp
BEFORE_TS=$(date +%s)

NY_BIN="${NY_BIN}" "${BENCHMARK_SCRIPT}" cersyve --compare-backends --start-at 3 --limit 2

# Find the cersyve CSV (most recent comparecat file)
CERSYVE_CSV=""
for f in "${REPO_ROOT}"/reports/benchmarks/comparecat_compare_backends_*.csv; do
    if [[ -f "$f" ]]; then
        FILE_TS=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo 0)
        if [[ "$FILE_TS" -ge "$BEFORE_TS" ]]; then
            CERSYVE_CSV="$f"
        fi
    fi
done

BEFORE_TS2=$(date +%s)
NY_BIN="${NY_BIN}" "${BENCHMARK_SCRIPT}" metaroom_2023 --compare-backends --start-at 10 --limit 1

METAROOM_CSV=""
for f in "${REPO_ROOT}"/reports/benchmarks/comparecat_compare_backends_*.csv; do
    if [[ -f "$f" ]]; then
        FILE_TS=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo 0)
        if [[ "$FILE_TS" -ge "$BEFORE_TS2" ]]; then
            METAROOM_CSV="$f"
        fi
    fi
done

echo "Cersyve CSV: ${CERSYVE_CSV:-not found}"
echo "Metaroom CSV: ${METAROOM_CSV:-not found}"

# --- Step 4: Optional reference comparator ---
REFERENCE_BLOCKER=""
REFERENCE_REAL_SECONDS=""

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
    if ${TIME_CMD} -p ${PYTHON_CMD} "${ABCROWN_DIR}/abcrown.py" \
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
if [[ -n "${REFERENCE_BLOCKER}" ]]; then
    VERDICT="blocked"
elif [[ -n "${REFERENCE_REAL_SECONDS}" ]]; then
    # We have both ny and reference data — verdict will be computed by renderer
    VERDICT="pending"
fi

cat > "$MANIFEST_FILE" <<MANIFEST_EOF
{
    "schema": "nvidia_vulkan_validation_manifest_v1",
    "verdict": "${VERDICT}",
    "blocker": $(if [[ -n "${REFERENCE_BLOCKER}" ]]; then echo "\"${REFERENCE_BLOCKER}\""; else echo "null"; fi),
    "host_info_path": "$(basename "$HOST_INFO_FILE")",
    "measure_log_path": "$(basename "$MEASURE_LOG")",
    "measure_csv_path": "$(basename "$MEASURE_CSV")",
    "vulkan_confirmed": true,
    "adapter_line": "${ADAPTER_LINE}",
    "compare_backends_cersyve_csv": $(if [[ -n "${CERSYVE_CSV}" ]]; then echo "\"$(basename "$CERSYVE_CSV")\""; else echo "null"; fi),
    "compare_backends_metaroom_csv": $(if [[ -n "${METAROOM_CSV}" ]]; then echo "\"$(basename "$METAROOM_CSV")\""; else echo "null"; fi),
    "reference_blocker": $(if [[ -n "${REFERENCE_BLOCKER}" ]]; then echo "\"${REFERENCE_BLOCKER}\""; else echo "null"; fi),
    "reference_cersyve_real_seconds": $(if [[ -n "${REFERENCE_REAL_SECONDS}" ]]; then echo "${REFERENCE_REAL_SECONDS}"; else echo "null"; fi)
}
MANIFEST_EOF

echo "Wrote: ${MANIFEST_FILE}"
echo ""
echo "=== Orchestrator complete ==="
echo "Next: python3 scripts/render_nvidia_vulkan_validation_report.py --manifest ${MANIFEST_FILE} --output-dir ${OUTPUT_DIR}"
