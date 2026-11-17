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

# Close the organizer-facing solver environment. Retain binary selection and
# identity inputs until the delegated helper checks them, then removes them.
scored_ny_bin_present=${NY_BIN+x}
scored_ny_bin=${NY_BIN-}
scored_expect_build_id_present=${NY_EXPECT_BUILD_ID+x}
scored_expect_build_id=${NY_EXPECT_BUILD_ID-}

# Scrub whole namespaces so new solver switches cannot change scored runs.
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
