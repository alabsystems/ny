#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Build the UNTRUSTED search kernels' committed images (ENGINE_PROGRAM §1.9):
#
#   crates/ny-cuda/kernels/fd_search.cu    ->  fd_search.fatbin    (M4 level search)
#   crates/ny-cuda/kernels/wave_search.cu  ->  wave_search.fatbin  (M6 wave search)
#       SASS sm_86 (A10G) + SASS sm_121 (GB10) + PTX compute_80 (any other
#       device of compute capability >= 8.0; the driver JITs it)
#
# and rewrite search_kernels.manifest (format 2: per image its source and
# image sha256 and its entry points; the nvcc version and the flags, shared).
# ny-cuda embeds both images (no build.rs, no runtime compilation), and its
# manifest ratchet fails whenever a source, an image or the manifest drift
# apart — so after editing a .cu: run this, commit the sources, the images and
# the manifest together. The images load independently: a wave image that
# fails to load or to self-test refuses the wave search only.
#
# Each image is deterministic for a given nvcc (two builds, any cwd, same
# bytes), so a reviewer with the recorded nvcc can reproduce the sha256.
#
# Usage: scripts/build_search_kernels.sh [--check]
#   --check  rebuild into temp files and fail (exit 1) if either differs from
#            its committed image (requires the manifest's nvcc version).
#
# Discipline (CLAUDE.md): this is a CPU compile; run it at nice 19 and never
# while the shared box is measuring.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KDIR="$ROOT/crates/ny-cuda/kernels"
MANIFEST="search_kernels.manifest"
# name  source  image  entry points (the order ny-cuda's loaders destructure)
IMAGES=(
    "fd fd_search.cu fd_search.fatbin fd_level_warm,fd_compact,fd_level_points,fd_level_finish,fd_math_kat"
    "wave wave_search.cu wave_search.fatbin wave_init,wave_backward,wave_concretize,wave_forward,wave_adam,wave_babsr"
)

NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" >/dev/null 2>&1; then
    echo "build_search_kernels: nvcc not found (set NVCC=...)" >&2
    exit 2
fi

FLAGS=(
    -std=c++17
    --fatbin
    --fmad=false
    --ftz=false
    --prec-div=true
    --prec-sqrt=true
    -gencode arch=compute_86,code=sm_86
    -gencode arch=compute_121,code=sm_121
    -gencode arch=compute_80,code=compute_80
)

NVCC_VERSION="$("$NVCC" --version | grep -E '^Cuda compilation tools' | head -1)"
if [[ -z "$NVCC_VERSION" ]]; then
    echo "build_search_kernels: cannot read the nvcc version" >&2
    exit 2
fi

cd "$KDIR"
TMPS=()
cleanup() { rm -f "${TMPS[@]}"; }
trap cleanup EXIT

STALE=0
declare -A BUILT
for spec in "${IMAGES[@]}"; do
    read -r name src out _ <<<"$spec"
    tmp="$(mktemp "${TMPDIR:-/tmp}/${name}_search.XXXXXX.fatbin")"
    TMPS+=("$tmp")
    nice -n 19 "$NVCC" "${FLAGS[@]}" -o "$tmp" "$src"
    BUILT[$name]="$tmp"
    if [[ "${1:-}" == "--check" ]]; then
        if cmp -s "$tmp" "$out"; then
            echo "build_search_kernels: $out is current ($NVCC_VERSION)"
        else
            echo "build_search_kernels: $out is STALE for $src under $NVCC_VERSION" >&2
            STALE=1
        fi
    fi
done
if [[ "${1:-}" == "--check" ]]; then
    exit "$STALE"
fi

{
    echo "# search_kernels.manifest — WRITTEN by scripts/build_search_kernels.sh; do not"
    echo "# edit by hand. Checked by ny-cuda's manifest ratchet"
    echo "# (search_kernel_manifest_matches_the_committed_sources)."
    echo "format=2"
    for spec in "${IMAGES[@]}"; do
        read -r name src out kernels <<<"$spec"
        cp "${BUILT[$name]}" "$out"
        chmod 0644 "$out"
        prefix=""
        [[ "$name" == "fd" ]] || prefix="${name}_"
        echo "${prefix}source=$src"
        echo "${prefix}source_sha256=$(sha256sum "$src" | cut -d' ' -f1)"
        echo "${prefix}image=$out"
        echo "${prefix}image_sha256=$(sha256sum "$out" | cut -d' ' -f1)"
        echo "${prefix}kernels=$kernels"
    done
    echo "nvcc=$NVCC_VERSION"
    echo "flags=${FLAGS[*]}"
} >"$MANIFEST.tmp"
mv "$MANIFEST.tmp" "$MANIFEST"

echo "build_search_kernels: wrote $(for spec in "${IMAGES[@]}"; do read -r _ _ out _ <<<"$spec"; printf '%s ' "$out"; done)and $MANIFEST"
