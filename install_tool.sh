#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Root-level VNN-COMP install_tool.sh
#
# Arguments: v1
#
# Optional fast path: when the submission package contains a prebuilt Linux
# x86_64 binary at dist/bin/ny-x86_64-linux.xz, its complete provenance is
# independently verified, it is checked against its GNU glibc 2.39 runtime
# floor, and it is sanity-run — no compiler, Rust toolchain, or network needed
# on the evaluation image.
# Release artifact: otherwise, when dist/bin/<host-target>/ holds a
# scripts/release/make_submission.sh artifact (MANIFEST, ny.xz, ny.sha256), it
# is verified, identity-checked and self-tested before it is installed; any
# failure exits non-zero (docs/RELEASE.md).
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
# Floor matches docs/VNNCOMP_2026_TRUST_LINUX_BUILD.md: the prebuilt binary links the
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

# --- Release artifact: dist/bin/<target>/ (scripts/release/make_submission.sh) --
# A DECLARED-lane build (`targo --unverified`, native or -Zbuild-std cross; it
# never carries the sealed record). It is used only when the sealed
# ny-x86_64-linux.* triplet above is absent and the directory for THIS host's
# target exists. Then nothing runs before vnncomp_scripts/verify_release.py has
# bound every byte to MANIFEST and MANIFEST to this checkout; the binary must
# then name itself exactly as MANIFEST does (`--build-id`, `--build-stamp`) and
# decide both install self-test instances correctly
# (vnncomp_scripts/selftest/: |x| <= 1 on [-1, 1], one unsat, one sat).
# Any failure exits non-zero: a present but unproven artifact is a packaging
# error, never a reason to try a source build. Only an older-glibc host falls
# through to the source build, exactly as for the sealed triplet.
release_host_target() {
    case "$(uname -s)/$(uname -m)" in
        Linux/x86_64) echo x86_64-unknown-linux-gnu ;;
        Linux/aarch64 | Linux/arm64) echo aarch64-unknown-linux-gnu ;;
        *) return 1 ;;
    esac
}

glibc_at_least() {
    local detected="$1" required="$2"
    [[ "${detected}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || return 1
    [[ "${required}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || return 1
    [ "$(printf '%s\n%s\n' "${required}" "${detected}" | sort -V | head -1)" = "${required}" ]
}

release_fail() {
    echo "ERROR: release artifact ${RELEASE_DIR}: $*" >&2
    echo "  Refusing to install it, and refusing a source-build fallback: a present but" >&2
    echo "  unproven artifact is a packaging error (see docs/RELEASE.md)." >&2
    exit 1
}

release_manifest_value() {
    local key="$1" line
    while IFS= read -r line; do
        case "${line}" in "${key}="*) printf '%s\n' "${line#*=}"; return 0 ;; esac
    done < "${RELEASE_DIR}/MANIFEST"
    return 1
}

# Run the staged binary with the scored wrapper's environment closure: no
# caller AY_*/NY_*/MIMALLOC_* variable reaches the self-test.
release_run_staged() {
    (
        for prefix in AY_ NY_ MIMALLOC_; do
            while IFS= read -r name; do unset "${name}"; done < <(compgen -e -- "${prefix}")
        done
        unset RAYON_NUM_THREADS RAYON_RS_NUM_CPUS
        if command -v timeout >/dev/null 2>&1; then
            exec timeout 900 "$@"
        fi
        exec "$@"
    )
}

cleanup_release_staging() {
    if [ -n "${RELEASE_STAGING:-}" ]; then
        # The self-test writes selftest_<prop>.{res,log} and ny's own sidecars
        # (selftest_<prop>.res.flight.json, ...): every file it can create is
        # named selftest_*, so this removes exactly the staging files, never more.
        rm -f -- "${RELEASE_STAGING}/ny" "${RELEASE_STAGING}/ny.receipt" "${RELEASE_STAGING}"/selftest_*
        rmdir -- "${RELEASE_STAGING}" 2>/dev/null || true
    fi
}

# Called in a `||` list, so `set -e` does not apply inside: every step below
# checks its own status.
install_release_artifact() {
    local staged staged_receipt verify_line detected required line
    local manifest_id manifest_stamp found_id found_stamp prop answer log
    command -v python3 >/dev/null 2>&1 || release_fail "python3 is required to verify it"
    [ -f "${RELEASE_VERIFIER}" ] && [ ! -L "${RELEASE_VERIFIER}" ] \
        || release_fail "the verifier ${RELEASE_VERIFIER} is missing"
    echo "Verifying release artifact ${RELEASE_DIR} (target ${RELEASE_TARGET})..."
    mkdir -p "${TARGET_DIR}" || release_fail "cannot create ${TARGET_DIR}"
    RELEASE_STAGING=$(mktemp -d "${TARGET_DIR}/.ny-release.XXXXXX") \
        || release_fail "cannot create a staging directory under ${TARGET_DIR}"
    trap cleanup_release_staging EXIT
    staged="${RELEASE_STAGING}/ny"
    staged_receipt="${RELEASE_STAGING}/ny.receipt"
    verify_line=$(python3 -I "${RELEASE_VERIFIER}" --repo-root "${SCRIPT_DIR}" \
        --target "${RELEASE_TARGET}" --output "${staged}") \
        || release_fail "verify_release.py refused it (reason above)"
    echo "${verify_line}"

    required=$(release_manifest_value max_glibc) || release_fail "MANIFEST has no max_glibc"
    detected=$(detect_glibc_version || true)
    if [ -z "${detected}" ] || ! glibc_at_least "${detected}" "${required}"; then
        echo "WARNING: the release binary needs GNU glibc >= ${required}; detected '${detected:-unknown}'." >&2
        echo "  Falling back to a source build for this host." >&2
        cleanup_release_staging
        trap - EXIT
        return 1
    fi

    chmod 0755 "${staged}" || release_fail "cannot mark the staged binary executable"
    line=$(release_run_staged "${staged}" --version 2>/dev/null | head -n 1) \
        || release_fail "the binary does not run on this host (--version failed)"
    echo "Release binary: ${line} (GNU glibc ${detected} >= ${required})"
    manifest_id=$(release_manifest_value build_id) || release_fail "MANIFEST has no build_id"
    manifest_stamp=$(release_manifest_value build_stamp) || release_fail "MANIFEST has no build_stamp"
    found_id=$(release_run_staged "${staged}" --build-id 2>/dev/null | grep -m 1 '^NY_BUILD_ID_V1 ' || true)
    [ "${found_id}" = "${manifest_id}" ] \
        || release_fail "the binary identifies itself as '${found_id:-<nothing>}', MANIFEST says '${manifest_id}'"
    found_stamp=$(release_run_staged "${staged}" --build-stamp 2>/dev/null | grep -m 1 '^NY_BUILD_STAMP_V1 ' || true)
    [ "${found_stamp}" = "${manifest_stamp}" ] \
        || release_fail "the binary's build stamp '${found_stamp:-<nothing>}' is not MANIFEST's '${manifest_stamp}'"
    echo "Release binary identity matches MANIFEST: ${found_id}"

    for prop in unsat sat; do
        log="${RELEASE_STAGING}/selftest_${prop}.log"
        release_run_staged "${staged}" vnncomp v1 test "${RELEASE_SELFTEST}/abs_relu.onnx" \
            "${RELEASE_SELFTEST}/abs_relu_${prop}.vnnlib" "${RELEASE_STAGING}/selftest_${prop}.res" 60 \
            > "${log}" 2>&1 || true
        answer=$(head -n 1 "${RELEASE_STAGING}/selftest_${prop}.res" 2>/dev/null | tr -d '[:space:]' || true)
        if [ "${answer}" != "${prop}" ]; then
            tail -n 20 "${log}" >&2 || true
            case "${answer}" in
                sat | unsat) release_fail "self-test WRONG ANSWER: abs_relu_${prop} answered ${answer}" ;;
                *) release_fail "self-test abs_relu_${prop} did not decide (answered '${answer:-<nothing>}')" ;;
            esac
        fi
        echo "Release self-test abs_relu_${prop}: ${answer}"
    done

    [ -f "${RECEIPT_HELPER}" ] && [ ! -L "${RECEIPT_HELPER}" ] \
        || release_fail "the receipt helper ${RECEIPT_HELPER} is missing"
    bash "${RECEIPT_HELPER}" create-release "${staged}" "${SCRIPT_DIR}" \
        "${RELEASE_DIR}/MANIFEST" "${staged_receipt}" \
        || release_fail "could not bind a runtime receipt to it"
    # Receipt last: an interruption leaves a binary whose receipt is absent or
    # stale, which run_instance.sh rejects instead of scoring unproven bytes.
    mv -f -- "${staged}" "${TARGET_BINARY}" || release_fail "cannot install ${TARGET_BINARY}"
    mv -f -- "${staged_receipt}" "${TARGET_RECEIPT}" || release_fail "cannot install ${TARGET_RECEIPT}"
    cleanup_release_staging
    trap - EXIT
    echo "NY_RELEASE_INSTALL_OK target=${RELEASE_TARGET} binary=${TARGET_BINARY} ${found_id}"
    exit 0
}

RELEASE_TARGET=$(release_host_target || true)
RELEASE_DIR="${SCRIPT_DIR}/dist/bin/${RELEASE_TARGET:-unsupported-host}"
RELEASE_VERIFIER="${SCRIPT_DIR}/vnncomp_scripts/verify_release.py"
RELEASE_SELFTEST="${SCRIPT_DIR}/vnncomp_scripts/selftest"
if [ -n "${RELEASE_TARGET}" ] && { [ -e "${RELEASE_DIR}" ] || [ -L "${RELEASE_DIR}" ]; }; then
    if [ -f "${PREBUILT}" ]; then
        echo "NOTE: ${RELEASE_DIR} is ignored: the sealed ${PREBUILT} takes precedence." >&2
    else
        install_release_artifact || true
    fi
fi

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
# (channel = "trust"), a locally linked rustup toolchain, and the source build
# runs its targo in the unverified lane (vnncomp_scripts/build_submission_binary.sh).
# On a host without a linked `trust` toolchain this source build cannot
# succeed, and it fails fast at the first tool invocation — before the long
# build, not 30 minutes into it. That fail-fast is deliberate
# (publish/DECISIONS.md, trust-flip entry): the supported installation is the
# validated prebuilt triplet above; the source fallback remains only for hosts
# that carry the Trust toolchain. Where it is linked, its bin directory goes on
# PATH so the builder finds targo/trustc.
if ! rustup toolchain list 2>/dev/null | grep -q '^trust'; then
    echo "WARNING: rust-toolchain.toml pins the locally linked Trust toolchain" >&2
    echo "  (channel = \"trust\") and no rustup toolchain named 'trust' is linked on" >&2
    echo "  this host. The source fallback cannot build here; install via the" >&2
    echo "  validated prebuilt triplet instead. Proceeding so the build can fail" >&2
    echo "  closed with its own error..." >&2
elif trust_targo="$(rustup which targo --toolchain trust 2>/dev/null)"; then
    PATH="$(dirname "${trust_targo}"):${PATH}"
    export PATH
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
