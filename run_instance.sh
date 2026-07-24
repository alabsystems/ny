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

# Competition runs must not inherit AY's developer-only lane overrides, query
# capture sink, or mimalloc runtime tuning. Keep newly pinned AY experiments
# explicit here: two are presence-gated, so even a caller's "0" can arm them.
# The allocator accepts an open-ended MIMALLOC_* namespace, so sanitize every
# currently exported member rather than maintaining a stale list. Local A/B and
# capture workflows use vnncomp_scripts/run_instance.sh (or invoke ny directly),
# both of which deliberately preserve these variables.
unset AY_MILP_SMT AY_MILP_GUB_CLIQUE AY_MILP_STAB_ORBIT \
    AY_MILP_COVER_MINIMAL AY_DUMP_QUERY_DIR
while IFS= read -r mimalloc_env; do
    unset "$mimalloc_env"
done < <(compgen -e -- MIMALLOC_)

exec "${SCRIPT_DIR}/vnncomp_scripts/run_instance.sh" "$@"
