#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Root-level VNN-COMP prepare_instance.sh wrapper
# Delegates to vnncomp_scripts/prepare_instance.sh
#
# Arguments: v1 CATEGORY ONNX_FILE VNNLIB_FILE

SCRIPT_DIR=$(dirname "$(realpath "$0")")
exec "${SCRIPT_DIR}/vnncomp_scripts/prepare_instance.sh" "$@"
