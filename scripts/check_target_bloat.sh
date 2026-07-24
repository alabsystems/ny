#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Diagnostic: detect bloated target/*/debug/deps/ directories that can cause
# dyld stall on macOS (ny#4182).
#
# Usage: ./scripts/check_target_bloat.sh [--threshold BYTES]
#
# Default threshold: 5MB (5242880 bytes) of directory metadata.
# Healthy deps/ dirs are ~300-400KB. Bloated ones reach 15-28MB.

set -euo pipefail

THRESHOLD="${1:-5242880}"  # 5MB default
TARGET_DIR="${2:-target}"
BLOATED=0

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "No target/ directory found."
    exit 0
fi

echo "Checking target directory bloat (threshold: $(( THRESHOLD / 1024 / 1024 ))MB)..."
echo ""

for deps_dir in "$TARGET_DIR"/*/debug/deps; do
    [[ -d "$deps_dir" ]] || continue
    dir_size=$(stat -f "%z" "$deps_dir" 2>/dev/null || echo "0")
    dir_name=$(echo "$deps_dir" | sed "s|$TARGET_DIR/||")
    size_mb=$(( dir_size / 1024 / 1024 ))

    if [[ "$dir_size" -gt "$THRESHOLD" ]]; then
        echo "  BLOATED: $dir_name (${size_mb}MB directory metadata)"
        BLOATED=1
    else
        size_kb=$(( dir_size / 1024 ))
        echo "  OK:      $dir_name (${size_kb}KB)"
    fi
done

echo ""
if [[ "$BLOATED" -eq 1 ]]; then
    echo "Bloated directories found. Workaround:"
    echo "  # Stop workers first, then:"
    echo "  for d in $TARGET_DIR/worker_*/debug/deps $TARGET_DIR/prover_*/debug/deps; do"
    echo "    [ -d \"\$d\" ] && rm -rf \"\$d\""
    echo "  done"
    exit 1
else
    echo "All target directories are healthy."
    exit 0
fi
