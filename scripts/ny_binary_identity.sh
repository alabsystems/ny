#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# ny_binary_identity.sh — the one command to run BEFORE any measurement.
#
# Prints the canonical binary identity and, optionally, refuses to proceed
# unless the binary is the feature tier you meant to measure.
#
# WHY THIS EXISTS
#   Four times this campaign a measurement was attributed to the wrong binary.
#   The sharpest case: a 330-row re-sweep sealed executable.sha256=cfe490f1…
#   with compute_backend="cuda …", then invoked target/release/ny BY MUTABLE
#   PATH for every row. Somebody rebuilt; the file became e3d81cc2…, mip-only,
#   wgpu. 21 rows read as regressions that were a different verifier. Both
#   binaries answered `ny 0.3.0`, because [workspace.package] version had not
#   moved for the whole campaign.
#
#   `ny --build-id` fixes the binary's half: the version is now derived
#   (<major>.<minor>.<git rev-list count>, plus +dirty.<commit>.<diff digest>),
#   qualified by the cargo feature tier, backend class, and impl markers, and
#   bound to the sha256 of the file's exact bytes. This script is the
#   harness's half.
#
# USAGE
#   scripts/ny_binary_identity.sh [--bin PATH] [--json] [--quiet]
#                                 [--expect-scored | --expect-features TIER]
#                                 [--expect-id LINE]
#
#   --bin PATH          binary to identify (default: $NY_BIN, else
#                       target/release/ny, else target/debug/ny)
#   --json              emit the machine-readable form instead of the line
#   --quiet             print nothing on success; exit status is the answer
#   --expect-features   exit 2 unless the compiled tier is exactly TIER
#   --expect-scored     shorthand for the SUBMISSION tier. Authority is
#                       vnncomp_scripts/build_submission_binary.sh:527, which
#                       tries "mip,cuda" FIRST and calls it "the required
#                       competition feature tier"; the lower tiers below it are
#                       reachable only under NY_ALLOW_DEGRADED_BUILD=1.
#                       vnncomp_scripts/submission_binary_receipt.sh hard-codes
#                       features=mip,cuda for the prebuilt package, and
#                       scripts/measure_ny_scorecard.sh refuses to measure
#                       unless --build-info says cuda=on mip=on.
#                       Override with NY_SCORED_FEATURES for a host that cannot
#                       run CUDA at all (this GB10 is wgpu/Vulkan, so its local
#                       lane is NY_SCORED_FEATURES=mip) — an override is a
#                       DECLARATION that the run is not scored-tier evidence.
#   --expect-id LINE    exit 2 unless the canonical line matches exactly.
#                       This is the mid-run check: capture the line once, pass
#                       it back before every row.
#   --stamp             use the FREE `ny --build-stamp` line (compiled fields
#                       only) instead of `ny --build-id` (which also hashes the
#                       executable). Use this for a per-row check inside a
#                       shell harness; use the default full form once, before
#                       the run, when you want the content address too.
#
# EXIT
#   0  identity printed and every expectation held
#   1  usage error, missing binary, or the binary could not self-identify
#   2  an expectation failed (wrong feature tier, or the identity changed)

set -uo pipefail

SCORED_FEATURES="${NY_SCORED_FEATURES:-mip,cuda}"

die() {
    echo "ERROR: ny binary identity: $*" >&2
    exit 1
}

reject() {
    echo "ERROR: ny binary identity: $*" >&2
    exit 2
}

sha256_file() {
    local path="$1" output
    if command -v sha256sum >/dev/null 2>&1; then
        output="$(sha256sum -- "${path}")" || return 1
        printf '%s\n' "${output%% *}"
    elif command -v shasum >/dev/null 2>&1; then
        output="$(shasum -a 256 -- "${path}")" || return 1
        printf '%s\n' "${output%% *}"
    elif command -v openssl >/dev/null 2>&1; then
        output="$(openssl dgst -sha256 "${path}")" || return 1
        printf '%s\n' "${output##* }"
    else
        return 1
    fi
}

# Read one `key=value` field out of the canonical single line.
identity_field() {
    local line="$1" key="$2" token
    for token in ${line}; do
        case "${token}" in
            "${key}="*) printf '%s\n' "${token#"${key}"=}"; return 0 ;;
        esac
    done
    return 1
}

BIN=""
JSON=0
QUIET=0
STAMP_ONLY=0
EXPECT_FEATURES=""
EXPECT_ID=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --bin) [ "$#" -ge 2 ] || die "--bin needs a path"; BIN="$2"; shift 2 ;;
        --json) JSON=1; shift ;;
        --quiet) QUIET=1; shift ;;
        --stamp) STAMP_ONLY=1; shift ;;
        --expect-scored) EXPECT_FEATURES="${SCORED_FEATURES}"; shift ;;
        --expect-features)
            [ "$#" -ge 2 ] || die "--expect-features needs a tier, e.g. mip,cuda"
            EXPECT_FEATURES="$2"; shift 2 ;;
        --expect-id)
            [ "$#" -ge 2 ] || die "--expect-id needs a canonical identity line"
            EXPECT_ID="$2"; shift 2 ;;
        -h|--help) sed -n '3,52p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

if [ -z "${BIN}" ]; then
    TOOL_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
    if [ -n "${NY_BIN:-}" ]; then
        BIN="${NY_BIN}"
    elif [ -f "${TOOL_DIR}/target/release/ny" ]; then
        BIN="${TOOL_DIR}/target/release/ny"
    elif [ -f "${TOOL_DIR}/target/debug/ny" ]; then
        BIN="${TOOL_DIR}/target/debug/ny"
    else
        die "no ny binary found; build one or pass --bin PATH"
    fi
fi

[ -f "${BIN}" ] || die "not a regular file: ${BIN}"
[ -x "${BIN}" ] || die "not executable: ${BIN}"

if [ "${STAMP_ONLY}" = "1" ]; then
    MODE_FLAG="--build-stamp"
    MODE_PREFIX="NY_BUILD_STAMP_V1"
else
    MODE_FLAG="--build-id"
    MODE_PREFIX="NY_BUILD_ID_V1"
fi

LINE="$("${BIN}" "${MODE_FLAG}" 2>/dev/null | head -1)" || true
case "${LINE}" in
    "${MODE_PREFIX} "*) ;;
    *)
        die "${BIN} does not support ${MODE_FLAG}.
  This binary predates #binary-identity, so nothing it produces can name its
  own build. Rebuild before measuring:
    cargo build --release -p ny-cli --features ${SCORED_FEATURES}"
        ;;
esac

# The binary reports the sha256 of the file it was loaded from; hash the file
# INDEPENDENTLY and require agreement. This is what makes the id
# content-addressed rather than self-asserted: a wrapper script, a shim, or a
# hardlink to different bytes is caught here. The free stamp makes no such
# claim, so there is nothing to cross-check in --stamp mode.
if [ "${STAMP_ONLY}" != "1" ]; then
    REPORTED_EXE="$(identity_field "${LINE}" exe)" \
        || die "canonical identity line has no exe= field: ${LINE}"
    ACTUAL_EXE="$(sha256_file "${BIN}")" \
        || die "sha256sum, shasum, or openssl is required to verify ${BIN}"
    if [ "${REPORTED_EXE}" != "${ACTUAL_EXE}" ]; then
        reject "the binary's self-reported bytes are not the bytes at its path.
  path:     ${BIN}
  reported: ${REPORTED_EXE}
  hashed:   ${ACTUAL_EXE}
  Something is between the path and the program that ran."
    fi
fi

if [ "${QUIET}" != "1" ]; then
    if [ "${JSON}" = "1" ]; then
        "${BIN}" "${MODE_FLAG}" --json || die "could not render JSON identity"
    else
        printf '%s\n' "${LINE}"
    fi
fi

if [ -n "${EXPECT_FEATURES}" ]; then
    FOUND_FEATURES="$(identity_field "${LINE}" features)" \
        || die "canonical identity line has no features= field"
    if [ "${FOUND_FEATURES}" != "${EXPECT_FEATURES}" ]; then
        FOUND_VERSION="$(identity_field "${LINE}" version)" || FOUND_VERSION="unknown"
        reject "wrong cargo feature tier for this measurement.
  expected: ${EXPECT_FEATURES}
  found:    ${FOUND_FEATURES}   (ny ${FOUND_VERSION})
  binary:   ${BIN}
  Rebuild with: cargo build --release -p ny-cli --features ${EXPECT_FEATURES}
  (or set NY_SCORED_FEATURES to declare a deliberately non-scored lane)"
    fi
fi

if [ -n "${EXPECT_ID}" ] && [ "${EXPECT_ID}" != "${LINE}" ]; then
    reject "the binary changed since the identity was captured.
  expected: ${EXPECT_ID}
  found:    ${LINE}
  Rows measured on either side of this change are not comparable."
fi

exit 0
