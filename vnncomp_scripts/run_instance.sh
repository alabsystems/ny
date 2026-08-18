#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# VNN-COMP run_instance.sh - thin wrapper over the native `ny vnncomp` subcommand.
#
# Arguments: v1 CATEGORY ONNX_FILE VNNLIB_FILE RESULTS_FILE TIMEOUT
#
# Preset auto-loading, internal timeout tiering, β-CROWN invocation (with the AUTO
# branching/backend/complete-verifier/PGD defaults), result translation
# (unsat/sat/timeout/unknown/error) and counterexample-witness output now live INSIDE
# the binary (see crates/ny-cli/src/commands/vnncomp.rs). This script only locates the
# binary and execs it under an OS-level wall-clock backstop (timeout/gtimeout at
# TIMEOUT+10s); exit 124 from that backstop is still scored as a slow run, never an
# error, because `ny vnncomp` has already written a sound verdict to RESULTS_FILE
# before its own internal deadline fires.

set -u

# Locate the tool directory (repo root = parent of vnncomp_scripts/).
TOOL_DIR=$(dirname "$(dirname "$(realpath "$0")")")
RECEIPT_HELPER="${TOOL_DIR}/vnncomp_scripts/submission_binary_receipt.sh"

# Resolve the ny binary. An explicit NY_BIN is an intentional developer
# override and remains compatible with arbitrary mock/external executables; say
# loudly that it bypasses repository freshness checks. Automatic release/debug
# selection is different: it is the organizer/scoring path and must carry a
# receipt matching the exact binary bytes and current NY source identity.
explicit_ny_bin=0
if [ -n "${NY_BIN:-}" ]; then
    explicit_ny_bin=1
    if [ ! -f "${NY_BIN}" ] || [ ! -x "${NY_BIN}" ]; then
        [ -n "${5:-}" ] && echo "error" > "$5"
        echo "Error: explicit NY_BIN is not a regular executable: ${NY_BIN}" >&2
        exit 1
    fi
    echo "NOTE: using explicit NY_BIN override; automatic NY provenance receipt validation is bypassed: ${NY_BIN}" >&2
elif [ -f "${TOOL_DIR}/target/release/ny" ]; then
    NY_BIN="${TOOL_DIR}/target/release/ny"
elif [ -f "${TOOL_DIR}/target/debug/ny" ]; then
    NY_BIN="${TOOL_DIR}/target/debug/ny"
else
    # No binary => genuine error. Write it to RESULTS_FILE (arg 5) if we have one.
    [ -n "${5:-}" ] && echo "error" > "$5"
    echo "Error: ny binary not found. Run './vnncomp_scripts/build_submission_binary.sh' first." >&2
    exit 1
fi

if [ "${explicit_ny_bin}" = "0" ]; then
    if [ -L "${RECEIPT_HELPER}" ] || [ ! -f "${RECEIPT_HELPER}" ]; then
        [ -n "${5:-}" ] && echo "error" > "$5"
        echo "Error: NY receipt validator is missing: ${RECEIPT_HELPER}" >&2
        exit 1
    fi
    if ! bash "${RECEIPT_HELPER}" validate "${NY_BIN}" "${TOOL_DIR}" >&2; then
        [ -n "${5:-}" ] && echo "error" > "$5"
        echo "Error: refusing stale or unproven automatic NY binary ${NY_BIN}." >&2
        echo "  Rebuild with './vnncomp_scripts/build_submission_binary.sh'." >&2
        echo "  Developers may set NY_BIN explicitly to opt out for a controlled A/B." >&2
        exit 1
    fi
fi

# OS-level wall-clock backstop = scored budget + 10s. Only fires if ny's own internal
# deadline somehow fails to; ny vnncomp has already written a verdict by then.
#
# The scored budget ($6) can be fractional in the competition CSVs (metaroom_2023
# is 210.0, traffic_signs_recognition_2023 is 480.0). bash $((...)) is integer-only
# on every platform, so a decimal point aborts the arithmetic; under `set -u` that
# leaves WALL_TIMEOUT unset and the exec line below fails BEFORE ny launches, so no
# verdict is ever written (0 points across those categories). Floor to an integer
# for the OS backstop only; the raw fractional budget is still forwarded verbatim to
# `ny vnncomp "$@"`, which floors it internally (parse_budget_secs).
WALL_BUDGET=${6:-0}
WALL_BUDGET=${WALL_BUDGET%%.*}
WALL_TIMEOUT=$(( ${WALL_BUDGET:-0} + 10 ))

# ny uses faer/ndarray matrixmultiply (not OpenBLAS); keep OMP single-threaded so it
# doesn't fight rayon's pool. Rayon itself uses all cores (correctness, not repro).
export OMP_NUM_THREADS=1

# NY_MARGIN_ROW_CONV_BWD_BLOCKED / NY_MARGIN_ROW_PARALLEL are deliberately NOT
# exported: 1 has been the compiled default since 7b004fba (parallel frontier,
# 07-18) and 2eaa6b13 (cache-blocked backward conv, 07-19), so exporting =1 was
# a bit-exact no-op (MEASURED: margin_row/tests.rs asserts unset == "1" for both
# blocked_backward_enabled_from_env and margin_row_frontier_from_env). The =0
# serial kill switches in ny-propagate still work for developer A/Bs.

# safenlp razor-thin SAT lane (#safenlp-upfront-attack, measured 2026-07-20):
# force the upfront gradient falsification lane for safenlp only. Its 8 razor-thin
# SAT rows fall to ~8 gradient steps in the upfront slice, which frees the whole
# internal budget for the long-BaB UNSAT proofs (preset max_domains 50000) —
# without this, SAT-vs-UNSAT is a measured zero-sum on the 20s budget. Category-
# scoped: globally forcing the lane would tax every benchmark's budget. Attack-
# only (every witness passes the unchanged trusted-ORT gate) => moat-safe.
case "${2:-}" in
    safenlp_2024)
        export NY_UPFRONT_ATTACK=1
        ;;
esac

# GPU capability hint (#vnncomp-gpu-available-lost): prepare_instance.sh only
# validates inputs and locates the binary, and in any case its process environment
# would be gone by the time the harness runs this script. Do NOT probe via tool
# presence here: nvcc/nvidia-smi does not imply a usable GPU (for example, a CUDA
# toolkit on a CPU-only box whose only Vulkan device is a software rasterizer), and
# an exported 1 would override the binary's adapter-level probe. `ny vnncomp`
# self-probes for a HARDWARE adapter when the var is unset; an explicit caller-set
# GPU_AVAILABLE (0 or 1) still wins.

if command -v timeout >/dev/null 2>&1; then
    exec timeout "${WALL_TIMEOUT}" "${NY_BIN}" vnncomp "$@"
elif command -v gtimeout >/dev/null 2>&1; then
    exec gtimeout "${WALL_TIMEOUT}" "${NY_BIN}" vnncomp "$@"
else
    exec "${NY_BIN}" vnncomp "$@"
fi
