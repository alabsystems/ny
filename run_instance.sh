#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Root-level VNN-COMP run_instance.sh wrapper
# Delegates to vnncomp_scripts/run_instance.sh
#
# Arguments: v1 CATEGORY ONNX_FILE VNNLIB_FILE RESULTS_FILE TIMEOUT

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# The organizer-facing entrypoint is environment-closed for solver policy.
# Search/evaluation children already remove every NY_* override; scored runs
# must do the same or a caller's shell can silently replace the measured preset.
# Keep the two wrapper control-plane values long enough for the delegated helper
# to locate/authenticate the binary. The helper removes them before starting ny.
scored_ny_bin_present=${NY_BIN+x}
scored_ny_bin=${NY_BIN-}
scored_expect_build_id_present=${NY_EXPECT_BUILD_ID+x}
scored_expect_build_id=${NY_EXPECT_BUILD_ID-}

# AY experiments include presence-gated switches, while NY_* contains both the
# typed registry and the still-legacy parameter frontier. Namespace scrubbing is
# deliberately future-proof: adding a knob cannot reopen scored ambient authority.
for solver_env_prefix in AY_ NY_ MIMALLOC_; do
    while IFS= read -r solver_env; do
        unset "$solver_env"
    done < <(compgen -e -- "$solver_env_prefix")
done
unset RAYON_NUM_THREADS RAYON_RS_NUM_CPUS

if [ -n "$scored_ny_bin_present" ]; then
    export NY_BIN="$scored_ny_bin"
fi
if [ -n "$scored_expect_build_id_present" ]; then
    export NY_EXPECT_BUILD_ID="$scored_expect_build_id"
fi
export VNNCOMP_SCORED_ENV_CLOSED=1

exec "${SCRIPT_DIR}/vnncomp_scripts/run_instance.sh" "$@"
