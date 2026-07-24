#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# VNN-COMP prepare_instance.sh - called once per instance before verification
#
# Arguments: v1 CATEGORY ONNX_FILE VNNLIB_FILE
#
# Per VNN-COMP 2026 rules, this script:
# - Receives instance parameters
# - May perform preprocessing, but MUST NOT analyze/verify the instance
# - Returns 0 on success, non-zero to skip this instance

TOOL_NAME=ny
VERSION_STRING=v1

# Check arguments
if [ "$1" != "${VERSION_STRING}" ]; then
    echo "Expected first argument (version string) '${VERSION_STRING}', got '$1'"
    exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4

# Validate the ONNX argument. VNN-LIB 2.0 relational benchmarks pass a Python
# literal list of `(network-name, path)` pairs in the single ONNX_FILE slot,
# rather than one filesystem path. Parse that representation with
# `ast.literal_eval` (never shell `eval`) and validate every referenced file.
if [ ! -f "${ONNX_FILE}" ]; then
    if ! python3 - "${ONNX_FILE}" <<'PY'
import ast
import os
import sys

try:
    networks = ast.literal_eval(sys.argv[1])
except (SyntaxError, ValueError) as exc:
    raise SystemExit(f"invalid multi-network ONNX argument: {exc}")

if not isinstance(networks, (list, tuple)) or not networks:
    raise SystemExit("multi-network ONNX argument must be a non-empty list")

for entry in networks:
    if (
        not isinstance(entry, (list, tuple))
        or len(entry) != 2
        or not isinstance(entry[0], str)
        or not isinstance(entry[1], str)
    ):
        raise SystemExit(f"invalid multi-network entry: {entry!r}")
    if not os.path.isfile(entry[1]):
        raise SystemExit(f"ONNX file not found: {entry[1]}")
PY
    then
        echo "Error: ONNX file or multi-network list not found/valid: ${ONNX_FILE}"
        exit 1
    fi
fi

if [ ! -f "${VNNLIB_FILE}" ]; then
    echo "Error: VNNLIB file not found: ${VNNLIB_FILE}"
    exit 1
fi

echo "Preparing ${TOOL_NAME} for benchmark instance in category '${CATEGORY}' with onnx file '${ONNX_FILE}' and vnnlib file '${VNNLIB_FILE}'"

# Find tool directory (one level up from this script)
TOOL_DIR=$(dirname "$(dirname "$(realpath "$0")")")
echo "TOOL_DIR is ${TOOL_DIR}"

# Set ny binary path: prefer the release build, fall back to a debug build
# for local workflows.
if [ -f "${TOOL_DIR}/target/release/ny" ]; then
    NY_BIN="${TOOL_DIR}/target/release/ny"
elif [ -f "${TOOL_DIR}/target/debug/ny" ]; then
    NY_BIN="${TOOL_DIR}/target/debug/ny"
else
    echo "Error: ny binary not found. Run './vnncomp_scripts/build_submission_binary.sh' first."
    exit 1
fi
export NY_BIN
echo "Using ny binary: ${NY_BIN}"

# Deliberately do not invoke `ny verify` here. The official contract permits
# conversion/compilation but explicitly forbids analysis in prepare_instance;
# all model-dependent verification work belongs in run_instance.sh.

echo "Preparation finished (no instance analysis performed)."
exit 0
