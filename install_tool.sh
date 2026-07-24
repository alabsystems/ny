#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Root-level VNN-COMP install_tool.sh
#
# Arguments: v1
#
# Optional fast path: when the submission package contains a CI-built Linux
# x86_64 binary at dist/bin/ny-x86_64-linux.xz, it is checksum-verified,
# checked against its GNU glibc 2.39 runtime floor, unpacked, and sanity-run —
# no compiler, Rust toolchain, or network needed on the evaluation image.
# Fallback: bootstrap build deps + rustup and build from source. AY remains an
# exact internal Git dependency, so this path requires caller-provided read
# access to the pinned AY revision; the installer never persists credentials.

set -euo pipefail

VERSION_STRING=v1
if [ "${1:-}" != "${VERSION_STRING}" ]; then
    echo "Expected first argument (version string) '${VERSION_STRING}', got '${1:-}'" >&2
    exit 1
fi

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PREBUILT="${SCRIPT_DIR}/dist/bin/ny-x86_64-linux.xz"
TARGET_DIR="${SCRIPT_DIR}/target/release"
PREBUILT_MIN_GLIBC_MAJOR=2
# Floor matches docs/VNNCOMP_2026_TRUST_LINUX_BUILD.md: the CI binary links the
# ort prebuilt, which requires glibc >= 2.39 (Ubuntu 24.04 eval box provides it).
PREBUILT_MIN_GLIBC_MINOR=39

detect_glibc_version() {
    local version_output=""

    # getconf is the least ambiguous interface on glibc hosts. Keep an ldd
    # fallback for minimal images that do not package getconf.
    if command -v getconf >/dev/null 2>&1 \
        && version_output=$(getconf GNU_LIBC_VERSION 2>/dev/null) \
        && [[ "${version_output}" =~ ^glibc[[:space:]]+([0-9]+\.[0-9]+) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    if command -v ldd >/dev/null 2>&1 \
        && version_output=$(ldd --version 2>&1) \
        && [[ "${version_output}" =~ ([Gg][Ll][Ii][Bb][Cc]|GNU[[:space:]]+libc)[^0-9]*([0-9]+\.[0-9]+) ]]; then
        printf '%s\n' "${BASH_REMATCH[2]}"
        return 0
    fi
    return 1
}

glibc_supports_prebuilt() {
    local version="${1}"
    local major
    local minor

    if [[ ! "${version}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    major=$((10#${BASH_REMATCH[1]}))
    minor=$((10#${BASH_REMATCH[2]}))
    ((major > PREBUILT_MIN_GLIBC_MAJOR \
        || (major == PREBUILT_MIN_GLIBC_MAJOR && minor >= PREBUILT_MIN_GLIBC_MINOR)))
}

# --- Fast path: packaged prebuilt binary (Linux x86_64 only) -----------------
if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ] && [ -f "${PREBUILT}" ]; then
    echo "Verifying prebuilt submission binary from ${PREBUILT}..."
    if [ -f "${PREBUILT}.sha256" ]; then
        (cd "$(dirname "${PREBUILT}")" && sha256sum -c "$(basename "${PREBUILT}").sha256")
    else
        echo "ERROR: refusing unchecked prebuilt binary; missing ${PREBUILT}.sha256" >&2
        exit 1
    fi

    detected_glibc=$(detect_glibc_version || true)
    if [ -z "${detected_glibc}" ]; then
        echo "WARNING: unable to confirm GNU glibc >= ${PREBUILT_MIN_GLIBC_MAJOR}.${PREBUILT_MIN_GLIBC_MINOR} for the prebuilt binary." >&2
        echo "  Falling back to a source build for this host." >&2
    elif ! glibc_supports_prebuilt "${detected_glibc}"; then
        echo "WARNING: prebuilt ny requires GNU glibc >= ${PREBUILT_MIN_GLIBC_MAJOR}.${PREBUILT_MIN_GLIBC_MINOR}; detected ${detected_glibc}." >&2
        echo "  Falling back to a source build for this host." >&2
    else
        echo "Installing prebuilt submission binary (GNU glibc ${detected_glibc})..."
        mkdir -p "${TARGET_DIR}"
        staged_prebuilt=$(mktemp "${TARGET_DIR}/.ny-prebuilt.XXXXXX")
        trap 'rm -f -- "${staged_prebuilt}"' EXIT
        if xz -dc "${PREBUILT}" > "${staged_prebuilt}" \
            && chmod +x "${staged_prebuilt}" \
            && "${staged_prebuilt}" --version; then
            mv -f -- "${staged_prebuilt}" "${TARGET_DIR}/ny"
            trap - EXIT
            echo "Prebuilt ny installed at ${TARGET_DIR}/ny."
            exit 0
        fi
        rm -f -- "${staged_prebuilt}"
        trap - EXIT
        echo "WARNING: prebuilt ny failed its sanity run; falling back to a source build." >&2
    fi
elif [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    echo "WARNING: optional prebuilt binary is absent: ${PREBUILT}" >&2
    echo "  Falling back to a networked source build. AY is exact Git-pinned and is not" >&2
    echo "  vendored; authenticated AY read access, crates.io/ORT downloads, and native" >&2
    echo "  build prerequisites must be available. The competition build fails" >&2
    echo "  closed instead of installing a materially weaker non-MIP binary." >&2
fi

# --- Fallback: build from source on a fresh image ---------------------------
# The eval AMI ships no compiler or Rust toolchain; bootstrap both. Failures
# here are non-fatal per-step so a partially-provisioned image still proceeds
# to the build, which reports precisely what is missing.
if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    if [ "$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
    ${SUDO} apt-get update -y || true
    ${SUDO} apt-get install -y build-essential pkg-config libssl-dev git curl xz-utils || true
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "Installing rustup (toolchain pinned by rust-toolchain.toml)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
fi
if [ -f "${HOME}/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "${HOME}/.cargo/env"
fi

# AY stays revision-pinned in Cargo.lock. Deliberately do not inspect the
# checkout URL or write credential-bearing rewrites into the user's global Git
# config; callers of the source fallback must provide process-scoped access.

exec "${SCRIPT_DIR}/vnncomp_scripts/build_submission_binary.sh"
