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
# x86_64 binary at dist/bin/ny-x86_64-linux.xz, its complete provenance is
# independently verified, it is checked against its GNU glibc 2.39 runtime
# floor, and it is sanity-run — no compiler, Rust toolchain, or network needed
# on the evaluation image.
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
PREBUILT_CHECKSUM="${PREBUILT}.sha256"
PREBUILT_PROVENANCE="${SCRIPT_DIR}/dist/bin/ny-x86_64-linux.provenance.txt"
PREBUILT_VERIFIER="${SCRIPT_DIR}/vnncomp_scripts/verify_prebuilt.py"
TARGET_DIR="${SCRIPT_DIR}/target/release"
TARGET_BINARY="${TARGET_DIR}/ny"
TARGET_RECEIPT="${TARGET_BINARY}.receipt"
RECEIPT_HELPER="${SCRIPT_DIR}/vnncomp_scripts/submission_binary_receipt.sh"
PREBUILT_MIN_GLIBC_MAJOR=2
# Floor matches docs/VNNCOMP_2026_TRUST_LINUX_BUILD.md: the CI binary links the
# ort prebuilt, which requires glibc >= 2.39 (Ubuntu 24.04 eval box provides it).
# The same floor gates the SOURCE build: ort-sys downloads that same prebuilt
# static ONNX Runtime archive, and on Ubuntu 22.04-era toolchains
# (gcc-11/binutils 2.38) the final link fails with undefined references to
# onnxruntime internals. The sealed release builder is locked to Ubuntu 24.04;
# an ordinary Ubuntu 26.04 build may import GLIBC_2.43 and is not a compatible
# substitute for the Ubuntu 24.04 evaluation artifact.
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
    if [ ! -f "${PREBUILT_CHECKSUM}" ]; then
        echo "ERROR: refusing unchecked prebuilt binary; missing ${PREBUILT_CHECKSUM}" >&2
        exit 1
    fi
    if [ ! -f "${PREBUILT_PROVENANCE}" ]; then
        echo "ERROR: refusing unproven prebuilt binary; missing ${PREBUILT_PROVENANCE}" >&2
        exit 1
    fi
    if [ ! -f "${PREBUILT_VERIFIER}" ]; then
        echo "ERROR: refusing prebuilt binary without verifier ${PREBUILT_VERIFIER}" >&2
        exit 1
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 is required to verify the packaged prebuilt" >&2
        exit 1
    fi

    mkdir -p "${TARGET_DIR}"
    prebuilt_staging_dir=$(mktemp -d "${TARGET_DIR}/.ny-prebuilt.XXXXXX")
    staged_prebuilt="${prebuilt_staging_dir}/ny"
    staged_receipt="${prebuilt_staging_dir}/ny.receipt"
    cleanup_prebuilt_staging() {
        rm -f -- "${staged_prebuilt}" "${staged_receipt}"
        rmdir -- "${prebuilt_staging_dir}" 2>/dev/null || true
    }
    trap cleanup_prebuilt_staging EXIT
    if ! python3 -I "${PREBUILT_VERIFIER}" \
        --repo-root "${SCRIPT_DIR}" \
        --archive "${PREBUILT}" \
        --checksum "${PREBUILT_CHECKSUM}" \
        --provenance "${PREBUILT_PROVENANCE}" \
        --output "${staged_prebuilt}"; then
        echo "ERROR: packaged prebuilt failed provenance validation; refusing source fallback" >&2
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
        if chmod +x "${staged_prebuilt}" \
            && "${staged_prebuilt}" --version; then
            if [ -L "${RECEIPT_HELPER}" ] || [ ! -f "${RECEIPT_HELPER}" ]; then
                echo "ERROR: submission receipt helper is missing: ${RECEIPT_HELPER}" >&2
                exit 1
            fi
            if ! bash "${RECEIPT_HELPER}" create-prebuilt \
                "${staged_prebuilt}" \
                "${SCRIPT_DIR}" \
                "${PREBUILT_PROVENANCE}" \
                "${staged_receipt}"; then
                echo "ERROR: refusing a prebuilt without a matching sealed runtime receipt." >&2
                exit 1
            fi
            # Publish the receipt last. An interruption between these renames
            # leaves a new binary with an absent/old mismatching receipt, which
            # run_instance.sh rejects instead of silently scoring stale bytes.
            mv -f -- "${staged_prebuilt}" "${TARGET_BINARY}"
            mv -f -- "${staged_receipt}" "${TARGET_RECEIPT}"
            rmdir -- "${prebuilt_staging_dir}"
            trap - EXIT
            echo "Prebuilt ny installed at ${TARGET_BINARY}."
            echo "Runtime receipt installed at ${TARGET_RECEIPT}."
            exit 0
        fi
        echo "WARNING: prebuilt ny failed its sanity run; falling back to a source build." >&2
    fi
    cleanup_prebuilt_staging
    trap - EXIT
elif [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    echo "WARNING: optional prebuilt binary is absent: ${PREBUILT}" >&2
    echo "  Falling back to a networked source build. AY is exact Git-pinned and is not" >&2
    echo "  vendored; authenticated AY read access, crates.io/ORT downloads, and native" >&2
    echo "  build prerequisites must be available. The competition build fails" >&2
    echo "  closed instead of installing a materially weaker non-MIP binary." >&2
fi

# --- Fallback: build from source on a fresh image ---------------------------
# The eval AMI ships no compiler or Rust toolchain; bootstrap both. Package
# provisioning is part of the source-build contract, so fail immediately when
# it cannot complete instead of continuing with a partially provisioned image.
if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    if [ "$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
    ${SUDO} apt-get update -y
    ${SUDO} apt-get install -y build-essential pkg-config git curl python3 xz-utils
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

# Trust-pin diagnostic: rust-toolchain.toml pins the Trust toolchain
# (channel = "trust"), a locally linked rustup toolchain, and .cargo/config.toml
# carries its `-Ztrust-verify=off` opt-out, which stock rustc rejects at flag
# parse. On a host without a linked `trust` toolchain this source build cannot
# succeed: rustup fails fast at the first cargo invocation ("toolchain 'trust'
# is not installed") — before the long build, not 30 minutes into it. That
# fail-fast is deliberate (publish/DECISIONS.md, trust-flip entry): the
# supported installation is the validated prebuilt triplet above; the source
# fallback remains only for hosts that carry the Trust toolchain.
if ! rustup toolchain list 2>/dev/null | grep -q '^trust'; then
    echo "WARNING: rust-toolchain.toml pins the locally linked Trust toolchain" >&2
    echo "  (channel = \"trust\") and no rustup toolchain named 'trust' is linked on" >&2
    echo "  this host. The source fallback cannot build here; install via the" >&2
    echo "  validated prebuilt triplet instead. Proceeding so rustup can fail" >&2
    echo "  closed with its own error..." >&2
fi

# Toolchain-era diagnostic before the (long) build: source builds on hosts
# older than the Ubuntu 24.04 toolchain era are known to fail at the FINAL
# link (see the floor comment above), i.e. only after the whole workspace has
# compiled. Warn now rather than 30 minutes from now. Warn-only: a backported
# newer toolchain on an old glibc may still work, and the build itself stays
# the authoritative fail-closed gate.
detected_glibc=$(detect_glibc_version || true)
if [ -n "${detected_glibc}" ] && ! glibc_supports_prebuilt "${detected_glibc}"; then
    echo "WARNING: GNU glibc ${detected_glibc} detected (< ${PREBUILT_MIN_GLIBC_MAJOR}.${PREBUILT_MIN_GLIBC_MINOR} — pre-Ubuntu-24.04 toolchain era)." >&2
    echo "  Source builds are known to FAIL at the final link on Ubuntu 22.04-era" >&2
    echo "  toolchains (gcc-11/binutils 2.38): ort-sys downloads a prebuilt static" >&2
    echo "  ONNX Runtime built with a newer GCC, and the link dies with undefined" >&2
    echo "  references to onnxruntime internals. Build on Ubuntu >= 24.04 (the" >&2
    echo "  VNN-COMP eval AMI), or set ORT_LIB_LOCATION to a locally built" >&2
    echo "  ONNX Runtime. Proceeding anyway in case this host's toolchain is newer" >&2
    echo "  than its glibc suggests..." >&2
fi

exec "${SCRIPT_DIR}/vnncomp_scripts/build_submission_binary.sh"
