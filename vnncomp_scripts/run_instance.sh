#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# VNN-COMP run_instance.sh - thin wrapper over the native `ny vnncomp` subcommand.
#
# Arguments: v1 CATEGORY ONNX_FILE VNNLIB_FILE RESULTS_FILE TIMEOUT
#
# The binary owns presets, verification, deadlines and result output. This wrapper
# locates and authenticates it, then adds a timeout/gtimeout backstop when available.
# A backstop exit does not guarantee a completed result file.

set -u

# Locate the tool directory (repo root = parent of vnncomp_scripts/).
TOOL_DIR=$(dirname "$(dirname "$(realpath "$0")")")
RECEIPT_HELPER="${TOOL_DIR}/vnncomp_scripts/submission_binary_receipt.sh"

# Explicit NY_BIN overrides bypass source receipts. Automatic release/debug
# selection requires a receipt binding the binary bytes to the current source.
explicit_ny_bin=0
if [ -n "${NY_BIN:-}" ]; then
    explicit_ny_bin=1
    if [ ! -f "${NY_BIN}" ] || [ ! -x "${NY_BIN}" ]; then
        [ -n "${5:-}" ] && echo "error" > "$5"
        echo "Error: explicit NY_BIN is not a regular executable: ${NY_BIN}" >&2
        exit 1
    fi
    ny_bin="${NY_BIN}"
    echo "NOTE: using explicit NY_BIN override; automatic NY provenance receipt validation is bypassed: ${ny_bin}" >&2
elif [ -f "${TOOL_DIR}/target/release/ny" ]; then
    ny_bin="${TOOL_DIR}/target/release/ny"
elif [ -f "${TOOL_DIR}/target/debug/ny" ]; then
    ny_bin="${TOOL_DIR}/target/debug/ny"
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
    if ! bash "${RECEIPT_HELPER}" validate "${ny_bin}" "${TOOL_DIR}" >&2; then
        [ -n "${5:-}" ] && echo "error" > "$5"
        echo "Error: refusing stale or unproven automatic NY binary ${ny_bin}." >&2
        echo "  Rebuild with './vnncomp_scripts/build_submission_binary.sh'." >&2
        echo "  Developers may set NY_BIN explicitly to opt out for a controlled A/B." >&2
        exit 1
    fi
fi

# A sweep can pin `ny --build-stamp` via NY_EXPECT_BUILD_ID to refuse a changed
# build between rows. The receipt alone permits a rebuild of the current source.
# This optional check reads the compiled stamp rather than hashing the executable.
if [ -n "${NY_EXPECT_BUILD_ID:-}" ]; then
    NY_IDENTITY_LINE="$("${ny_bin}" --build-stamp 2>/dev/null | head -1)"
    case "${NY_IDENTITY_LINE}" in
        NY_BUILD_STAMP_V1\ *) ;;
        *) NY_IDENTITY_LINE="NY_BUILD_STAMP_V1 unavailable (binary predates #binary-identity)" ;;
    esac
    echo "${NY_IDENTITY_LINE}" >&2
    if [ "${NY_EXPECT_BUILD_ID}" != "${NY_IDENTITY_LINE}" ]; then
        [ -n "${5:-}" ] && echo "error" > "$5"
        echo "Error: the ny binary is not the one this measurement was sealed against." >&2
        echo "  expected: ${NY_EXPECT_BUILD_ID}" >&2
        echo "  found:    ${NY_IDENTITY_LINE}" >&2
        echo "  binary:   ${ny_bin}" >&2
        echo "  Rows measured by two different verifiers are not comparable. Refusing." >&2
        exit 1
    fi
fi

# The OS backstop uses floor(budget) + 10s because Bash arithmetic is integer-only.
# Forward the original budget unchanged; ny parses its own verification deadline.
WALL_BUDGET=${6:-0}
WALL_BUDGET=${WALL_BUDGET%%.*}
WALL_TIMEOUT=$(( ${WALL_BUDGET:-0} + 10 ))

# Limit OpenMP threads independently of the verifier's Rayon pool.
export OMP_NUM_THREADS=1

# The root wrapper retains these inputs for binary selection and identity checks;
# remove them before starting the verifier under its closed solver environment.
if [ "${VNNCOMP_SCORED_ENV_CLOSED:-0}" = "1" ]; then
    unset NY_BIN NY_EXPECT_BUILD_ID VNNCOMP_SCORED_ENV_CLOSED
fi

# Leave GPU detection to the binary's adapter probe; installed tools do not prove
# device availability. An explicit caller-set GPU_AVAILABLE still takes precedence.

if command -v timeout >/dev/null 2>&1; then
    exec timeout "${WALL_TIMEOUT}" "${ny_bin}" vnncomp "$@"
elif command -v gtimeout >/dev/null 2>&1; then
    exec gtimeout "${WALL_TIMEOUT}" "${ny_bin}" vnncomp "$@"
else
    exec "${ny_bin}" vnncomp "$@"
fi
