#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Create and validate the sidecar that makes target/release/ny an authenticated
# build output instead of an unversioned cache slot.  Keep this helper in shell:
# the organizer's prebuilt fast path must not gain a Python runtime dependency.

set -euo pipefail

RECEIPT_SCHEMA="ny-submission-binary-receipt-v1"
SOURCE_SCHEMA="ny-vnncomp-source-v1"
PREBUILT_SCHEMA="ny-vnncomp-prebuilt-v1"
RELEASE_SCHEMA="ny-release-manifest-v1"
RECEIPT_MAX_BYTES=8192

receipt_fail() {
    echo "ERROR: NY submission binary receipt: $*" >&2
    return 1
}

receipt_is_lower_hex() {
    local value="$1"
    local length="$2"
    [[ "${#value}" -eq "${length}" && "${value}" =~ ^[0-9a-f]+$ ]]
}

receipt_require_digest() {
    receipt_is_lower_hex "$1" 64 \
        || receipt_fail "$2 must be exactly 64 lowercase hexadecimal characters"
}

receipt_sha256_file() {
    local path="$1"
    local output digest

    if command -v sha256sum >/dev/null 2>&1; then
        output="$(sha256sum -- "${path}")" \
            || receipt_fail "could not hash ${path}"
        digest="${output%% *}"
    elif command -v shasum >/dev/null 2>&1; then
        output="$(shasum -a 256 -- "${path}")" \
            || receipt_fail "could not hash ${path}"
        digest="${output%% *}"
    elif command -v openssl >/dev/null 2>&1; then
        output="$(openssl dgst -sha256 "${path}")" \
            || receipt_fail "could not hash ${path}"
        digest="${output##* }"
    else
        receipt_fail "sha256sum, shasum, or openssl is required"
        return 1
    fi
    digest="$(printf '%s' "${digest}" | tr '[:upper:]' '[:lower:]')"
    receipt_require_digest "${digest}" "SHA-256 output" || return 1
    printf '%s\n' "${digest}"
}

receipt_require_regular_file() {
    local path="$1"
    local label="$2"
    if [ -L "${path}" ] || [ ! -f "${path}" ]; then
        receipt_fail "${label} must be a regular non-symlink file: ${path}"
        return 1
    fi
}

receipt_require_executable() {
    local path="$1"
    receipt_require_regular_file "${path}" "binary" || return 1
    if [ ! -x "${path}" ]; then
        receipt_fail "binary is not executable: ${path}"
        return 1
    fi
}

# Prints one exact AY revision when Cargo.lock contains canonical AY Git
# packages, or "none" for a workspace with no AY dependency (used by tiny
# hermetic builder fixtures).  Multiple or non-canonical pins fail closed.
receipt_ay_commit() {
    local lock_path="$1"
    local line requested resolved
    local observed=""
    local count=0
    local ay_source_re='^source = "git\+https://github\.com/alabsystems/ay\.git\?rev=([0-9a-f]{40})#([0-9a-f]{40})"$'

    if [ ! -f "${lock_path}" ]; then
        printf 'none\n'
        return 0
    fi
    while IFS= read -r line; do
        case "${line}" in
            *alabsystems/ay*) ;;
            *) continue ;;
        esac
        count=$((count + 1))
        if [[ ! "${line}" =~ ${ay_source_re} ]]; then
            receipt_fail "non-canonical AY Cargo.lock source: ${line}"
            return 1
        fi
        requested="${BASH_REMATCH[1]}"
        resolved="${BASH_REMATCH[2]}"
        if [ "${requested}" != "${resolved}" ]; then
            receipt_fail "AY requested revision ${requested} resolved to ${resolved}"
            return 1
        fi
        if [ -n "${observed}" ] && [ "${observed}" != "${resolved}" ]; then
            receipt_fail "Cargo.lock contains multiple AY revisions"
            return 1
        fi
        observed="${resolved}"
    done < "${lock_path}"

    if [ "${count}" -eq 0 ]; then
        printf 'none\n'
    else
        printf '%s\n' "${observed}"
    fi
}

receipt_parse_source_marker() {
    local marker="$1"
    local line line_number=0
    local schema="" commit="" lock_sha=""

    receipt_require_regular_file "${marker}" "archive source marker" || return 1
    while IFS= read -r line; do
        line_number=$((line_number + 1))
        case "${line_number}" in
            1) schema="${line#schema=}"; [ "${line}" = "schema=${schema}" ] ;;
            2) commit="${line#ny_commit=}"; [ "${line}" = "ny_commit=${commit}" ] ;;
            3) lock_sha="${line#cargo_lock_sha256=}"; [ "${line}" = "cargo_lock_sha256=${lock_sha}" ] ;;
            *)
                receipt_fail "archive source marker has unexpected line ${line_number}"
                return 1
                ;;
        esac || {
            receipt_fail "archive source marker has malformed line ${line_number}"
            return 1
        }
    done < "${marker}"
    if [ "${line_number}" -ne 3 ] || [ "${schema}" != "${SOURCE_SCHEMA}" ]; then
        receipt_fail "archive source marker has the wrong schema or field count"
        return 1
    fi
    receipt_is_lower_hex "${commit}" 40 \
        || {
            receipt_fail "archive ny_commit must be exactly 40 lowercase hexadecimal characters"
            return 1
        }
    receipt_require_digest "${lock_sha}" "archive cargo_lock_sha256" || return 1
    NY_MARKER_COMMIT="${commit}"
    NY_MARKER_LOCK_SHA256="${lock_sha}"
}

# Populate NY_SOURCE_KIND/COMMIT/STATE/LOCK/AY for the source tree from which a
# local Cargo artifact was built.  In a checkout, STATE is the SHA-256 of the
# complete tracked diff against HEAD, so changing source without committing it
# invalidates a prior receipt too.  Extracted archives use the packager-injected
# marker and bind its exact bytes.
receipt_load_source_identity() {
    local tool_dir="$1"
    local canonical_tool git_root head
    local lock_path="${tool_dir}/Cargo.lock"
    local marker="${tool_dir}/.ny-vnncomp-source.txt"
    local diff_file="" untracked_file="" untracked_path="" untracked_hash="" executable_bit=""
    local untracked_error=""

    canonical_tool="$(realpath "${tool_dir}")"
    git_root="$(git -C "${canonical_tool}" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "${git_root}" ] \
        && [ "$(realpath "${git_root}")" = "${canonical_tool}" ]; then
        head="$(git -C "${canonical_tool}" rev-parse --verify HEAD 2>/dev/null || true)"
        if ! receipt_is_lower_hex "${head}" 40; then
            receipt_fail "could not resolve an exact 40-hex NY HEAD"
            return 1
        fi
        diff_file="$(mktemp "${TMPDIR:-/tmp}/ny-source-diff.XXXXXX")" \
            || {
                receipt_fail "could not stage the NY source-state digest"
                return 1
            }
        if ! git -C "${canonical_tool}" diff --binary --no-ext-diff HEAD -- > "${diff_file}"; then
            rm -f -- "${diff_file}"
            receipt_fail "could not capture the tracked NY source diff"
            return 1
        fi
        untracked_file="$(mktemp "${TMPDIR:-/tmp}/ny-source-untracked.XXXXXX")" \
            || {
                rm -f -- "${diff_file}"
                receipt_fail "could not stage the untracked NY source list"
                return 1
            }
        if ! git -C "${canonical_tool}" ls-files \
            --others --exclude-standard -z -- > "${untracked_file}"; then
            rm -f -- "${diff_file}" "${untracked_file}"
            receipt_fail "could not capture untracked NY source files"
            return 1
        fi
        # A tracked diff can reference a newly created module, build script, or
        # include file. Bind both its path and raw bytes so changing such an
        # untracked input after the build invalidates the receipt as well.
        while IFS= read -r -d '' untracked_path; do
            if ! untracked_hash="$(
                git -C "${canonical_tool}" hash-object --no-filters -- "${untracked_path}"
            )"; then
                untracked_error="could not hash untracked source ${untracked_path}"
                break
            fi
            if ! receipt_is_lower_hex "${untracked_hash}" 40; then
                untracked_error="Git returned a malformed untracked source hash"
                break
            fi
            if [ -x "${canonical_tool}/${untracked_path}" ]; then
                executable_bit=1
            else
                executable_bit=0
            fi
            printf 'untracked\0%s\0%s\0%s\n' \
                "${untracked_path}" \
                "${untracked_hash}" \
                "${executable_bit}" >> "${diff_file}"
        done < "${untracked_file}"
        rm -f -- "${untracked_file}"
        if [ -n "${untracked_error}" ]; then
            rm -f -- "${diff_file}"
            receipt_fail "${untracked_error}"
            return 1
        fi
        NY_SOURCE_KIND="git"
        NY_SOURCE_COMMIT="${head}"
        NY_SOURCE_STATE_SHA256="$(receipt_sha256_file "${diff_file}")" || {
            rm -f -- "${diff_file}"
            return 1
        }
        rm -f -- "${diff_file}"
    else
        if [ ! -f "${marker}" ] || [ -L "${marker}" ]; then
            receipt_fail "no exact NY source identity: expected a repository-root Git HEAD or ${marker}"
            return 1
        fi
        receipt_parse_source_marker "${marker}" || return 1
        NY_SOURCE_KIND="archive"
        NY_SOURCE_COMMIT="${NY_MARKER_COMMIT}"
        NY_SOURCE_STATE_SHA256="$(receipt_sha256_file "${marker}")" || return 1
    fi

    if [ -f "${lock_path}" ] && [ ! -L "${lock_path}" ]; then
        NY_SOURCE_LOCK_SHA256="$(receipt_sha256_file "${lock_path}")" || return 1
    else
        NY_SOURCE_LOCK_SHA256="none"
    fi
    NY_SOURCE_AY_COMMIT="$(receipt_ay_commit "${lock_path}")" || return 1

    if [ "${NY_SOURCE_KIND}" = "archive" ]; then
        if [ "${NY_SOURCE_LOCK_SHA256}" != "${NY_MARKER_LOCK_SHA256}" ]; then
            receipt_fail "archive Cargo.lock does not match its packaged source marker"
            return 1
        fi
    fi
}

receipt_parse_prebuilt_manifest() {
    local manifest="$1"
    local line line_number=0 key value
    local seen="|"

    NY_PREBUILT_SCHEMA=""
    NY_PREBUILT_TARGET=""
    NY_PREBUILT_FEATURES=""
    NY_PREBUILT_TRUST_COMMIT=""
    NY_PREBUILT_BOOTSTRAP=""
    NY_PREBUILT_GATE_STATUS=""
    NY_PREBUILT_GATE_RECEIPT=""
    NY_PREBUILT_GATE_COMMANDS=""
    NY_PREBUILT_GATE_LOG=""
    NY_PREBUILT_TRUSTC=""
    NY_PREBUILT_TRUSTC_VERSION=""
    NY_PREBUILT_NY_COMMIT=""
    NY_PREBUILT_LOCK=""
    NY_PREBUILT_AY=""
    NY_PREBUILT_BUILDER=""
    NY_PREBUILT_ORT=""
    NY_PREBUILT_BINARY=""
    NY_PREBUILT_PACKAGE=""

    receipt_require_regular_file "${manifest}" "prebuilt provenance" || return 1
    while IFS= read -r line; do
        line_number=$((line_number + 1))
        case "${line}" in
            *=*)
                key="${line%%=*}"
                value="${line#*=}"
                ;;
            *)
                receipt_fail "prebuilt provenance has malformed line ${line_number}"
                return 1
                ;;
        esac
        if [ -z "${value}" ] \
            || [ "${value}" != "$(printf '%s' "${value}" | tr -d '[:space:]')" ] \
            || [[ "${value}" == *"="* ]]; then
            receipt_fail "prebuilt provenance has invalid value for ${key}"
            return 1
        fi
        case "${seen}" in
            *"|${key}|"*)
                receipt_fail "prebuilt provenance has duplicate key ${key}"
                return 1
                ;;
        esac
        seen="${seen}${key}|"
        case "${key}" in
            schema) NY_PREBUILT_SCHEMA="${value}" ;;
            target) NY_PREBUILT_TARGET="${value}" ;;
            features) NY_PREBUILT_FEATURES="${value}" ;;
            trust_commit) NY_PREBUILT_TRUST_COMMIT="${value}" ;;
            trust_bootstrap_mode) NY_PREBUILT_BOOTSTRAP="${value}" ;;
            trust_gate_status) NY_PREBUILT_GATE_STATUS="${value}" ;;
            trust_gate_receipt_sha256) NY_PREBUILT_GATE_RECEIPT="${value}" ;;
            trust_gate_commands_sha256) NY_PREBUILT_GATE_COMMANDS="${value}" ;;
            trust_gate_log_sha256) NY_PREBUILT_GATE_LOG="${value}" ;;
            trustc_sha256) NY_PREBUILT_TRUSTC="${value}" ;;
            trustc_version_sha256) NY_PREBUILT_TRUSTC_VERSION="${value}" ;;
            ny_commit) NY_PREBUILT_NY_COMMIT="${value}" ;;
            cargo_lock_sha256) NY_PREBUILT_LOCK="${value}" ;;
            ay_lock_commit) NY_PREBUILT_AY="${value}" ;;
            builder_script_sha256) NY_PREBUILT_BUILDER="${value}" ;;
            onnxruntime_static_sha256) NY_PREBUILT_ORT="${value}" ;;
            binary_sha256) NY_PREBUILT_BINARY="${value}" ;;
            package_sha256) NY_PREBUILT_PACKAGE="${value}" ;;
            *)
                receipt_fail "prebuilt provenance has unknown key ${key}"
                return 1
                ;;
        esac
    done < "${manifest}"
    if [ "${line_number}" -ne 18 ] \
        || [ "${NY_PREBUILT_SCHEMA}" != "${PREBUILT_SCHEMA}" ] \
        || [ "${NY_PREBUILT_TARGET}" != "x86_64-unknown-linux-gnu" ] \
        || [ "${NY_PREBUILT_FEATURES}" != "mip,cuda" ] \
        || [ "${NY_PREBUILT_BOOTSTRAP}" != "seed" ] \
        || [ "${NY_PREBUILT_GATE_STATUS}" != "passed" ]; then
        receipt_fail "prebuilt provenance has the wrong schema, target, feature tier, or gate status"
        return 1
    fi
    receipt_is_lower_hex "${NY_PREBUILT_TRUST_COMMIT}" 40 || {
        receipt_fail "prebuilt trust_commit is malformed"
        return 1
    }
    receipt_is_lower_hex "${NY_PREBUILT_NY_COMMIT}" 40 || {
        receipt_fail "prebuilt ny_commit is malformed"
        return 1
    }
    receipt_is_lower_hex "${NY_PREBUILT_AY}" 40 || {
        receipt_fail "prebuilt ay_lock_commit is malformed"
        return 1
    }
    local digest
    for digest in \
        "${NY_PREBUILT_GATE_RECEIPT}" \
        "${NY_PREBUILT_GATE_COMMANDS}" \
        "${NY_PREBUILT_GATE_LOG}" \
        "${NY_PREBUILT_TRUSTC}" \
        "${NY_PREBUILT_TRUSTC_VERSION}" \
        "${NY_PREBUILT_LOCK}" \
        "${NY_PREBUILT_BUILDER}" \
        "${NY_PREBUILT_ORT}" \
        "${NY_PREBUILT_BINARY}" \
        "${NY_PREBUILT_PACKAGE}"; do
        receipt_require_digest "${digest}" "prebuilt provenance digest" || return 1
    done
}

receipt_parse() {
    local receipt="$1"
    local line line_number=0 bytes

    receipt_require_regular_file "${receipt}" "receipt" || return 1
    bytes="$(wc -c < "${receipt}" | tr -d '[:space:]')"
    if [[ ! "${bytes}" =~ ^[0-9]+$ ]] || [ "${bytes}" -gt "${RECEIPT_MAX_BYTES}" ]; then
        receipt_fail "receipt is oversized or unreadable: ${receipt}"
        return 1
    fi
    while IFS= read -r line; do
        line_number=$((line_number + 1))
        case "${line_number}" in
            1) NY_RECEIPT_SCHEMA="${line#schema=}"; [ "${line}" = "schema=${NY_RECEIPT_SCHEMA}" ] ;;
            2) NY_RECEIPT_BINARY="${line#binary_sha256=}"; [ "${line}" = "binary_sha256=${NY_RECEIPT_BINARY}" ] ;;
            3) NY_RECEIPT_SOURCE_KIND="${line#source_kind=}"; [ "${line}" = "source_kind=${NY_RECEIPT_SOURCE_KIND}" ] ;;
            4) NY_RECEIPT_SOURCE_COMMIT="${line#source_commit=}"; [ "${line}" = "source_commit=${NY_RECEIPT_SOURCE_COMMIT}" ] ;;
            5) NY_RECEIPT_SOURCE_STATE="${line#source_state_sha256=}"; [ "${line}" = "source_state_sha256=${NY_RECEIPT_SOURCE_STATE}" ] ;;
            6) NY_RECEIPT_LOCK="${line#cargo_lock_sha256=}"; [ "${line}" = "cargo_lock_sha256=${NY_RECEIPT_LOCK}" ] ;;
            7) NY_RECEIPT_AY="${line#ay_commit=}"; [ "${line}" = "ay_commit=${NY_RECEIPT_AY}" ] ;;
            8) NY_RECEIPT_FEATURES="${line#features=}"; [ "${line}" = "features=${NY_RECEIPT_FEATURES}" ] ;;
            9) NY_RECEIPT_TOOLCHAIN_KIND="${line#toolchain_kind=}"; [ "${line}" = "toolchain_kind=${NY_RECEIPT_TOOLCHAIN_KIND}" ] ;;
            10) NY_RECEIPT_TOOLCHAIN="${line#toolchain_sha256=}"; [ "${line}" = "toolchain_sha256=${NY_RECEIPT_TOOLCHAIN}" ] ;;
            11) NY_RECEIPT_ARTIFACT_PROVENANCE="${line#artifact_provenance_sha256=}"; [ "${line}" = "artifact_provenance_sha256=${NY_RECEIPT_ARTIFACT_PROVENANCE}" ] ;;
            *)
                receipt_fail "receipt has unexpected line ${line_number}"
                return 1
                ;;
        esac || {
            receipt_fail "receipt has malformed line ${line_number}"
            return 1
        }
    done < "${receipt}"
    if [ "${line_number}" -ne 11 ] || [ "${NY_RECEIPT_SCHEMA}" != "${RECEIPT_SCHEMA}" ]; then
        receipt_fail "receipt has the wrong schema or field count"
        return 1
    fi
    receipt_require_digest "${NY_RECEIPT_BINARY}" "binary_sha256" || return 1
    receipt_is_lower_hex "${NY_RECEIPT_SOURCE_COMMIT}" 40 || {
        receipt_fail "source_commit must be exactly 40 lowercase hexadecimal characters"
        return 1
    }
    receipt_require_digest "${NY_RECEIPT_SOURCE_STATE}" "source_state_sha256" || return 1
    if [ "${NY_RECEIPT_LOCK}" != "none" ]; then
        receipt_require_digest "${NY_RECEIPT_LOCK}" "cargo_lock_sha256" || return 1
    fi
    if [ "${NY_RECEIPT_AY}" != "none" ] && ! receipt_is_lower_hex "${NY_RECEIPT_AY}" 40; then
        receipt_fail "ay_commit must be one exact commit or none"
        return 1
    fi
    case "${NY_RECEIPT_SOURCE_KIND}" in git|archive|prebuilt|release) ;; *)
        receipt_fail "unknown source_kind ${NY_RECEIPT_SOURCE_KIND}"
        return 1
    esac
    case "${NY_RECEIPT_TOOLCHAIN_KIND}" in rustc-vv|trust-sealed|trust-unverified) ;; *)
        receipt_fail "unknown toolchain_kind ${NY_RECEIPT_TOOLCHAIN_KIND}"
        return 1
    esac
    [[ "${NY_RECEIPT_FEATURES}" =~ ^[a-z0-9]+(,[a-z0-9]+)*$ ]] || {
        receipt_fail "features is not canonical"
        return 1
    }
    receipt_require_digest "${NY_RECEIPT_TOOLCHAIN}" "toolchain_sha256" || return 1
    if [ "${NY_RECEIPT_ARTIFACT_PROVENANCE}" != "none" ]; then
        receipt_require_digest \
            "${NY_RECEIPT_ARTIFACT_PROVENANCE}" \
            "artifact_provenance_sha256" || return 1
    fi
}

receipt_publish() {
    local receipt="$1"
    local binary_sha="$2"
    local source_kind="$3"
    local source_commit="$4"
    local source_state="$5"
    local lock_sha="$6"
    local ay_commit="$7"
    local features="$8"
    local toolchain_kind="$9"
    local toolchain_sha="${10}"
    local artifact_provenance="${11}"
    local receipt_dir receipt_name temporary=""

    receipt_dir="$(dirname "${receipt}")"
    receipt_name="$(basename "${receipt}")"
    if [ -L "${receipt_dir}" ] || [ ! -d "${receipt_dir}" ]; then
        receipt_fail "receipt parent must be a real directory: ${receipt_dir}"
        return 1
    fi
    temporary="$(mktemp "${receipt_dir}/.${receipt_name}.XXXXXX")" || {
        receipt_fail "could not allocate a receipt staging file"
        return 1
    }
    if ! {
        printf 'schema=%s\n' "${RECEIPT_SCHEMA}"
        printf 'binary_sha256=%s\n' "${binary_sha}"
        printf 'source_kind=%s\n' "${source_kind}"
        printf 'source_commit=%s\n' "${source_commit}"
        printf 'source_state_sha256=%s\n' "${source_state}"
        printf 'cargo_lock_sha256=%s\n' "${lock_sha}"
        printf 'ay_commit=%s\n' "${ay_commit}"
        printf 'features=%s\n' "${features}"
        printf 'toolchain_kind=%s\n' "${toolchain_kind}"
        printf 'toolchain_sha256=%s\n' "${toolchain_sha}"
        printf 'artifact_provenance_sha256=%s\n' "${artifact_provenance}"
    } > "${temporary}"; then
        rm -f -- "${temporary}"
        receipt_fail "could not write the staged receipt"
        return 1
    fi
    chmod 0644 "${temporary}" || {
        rm -f -- "${temporary}"
        receipt_fail "could not set receipt permissions"
        return 1
    }
    if ! mv -f -- "${temporary}" "${receipt}"; then
        rm -f -- "${temporary}"
        receipt_fail "could not atomically publish ${receipt}"
        return 1
    fi
}

receipt_create_local() {
    local binary="$1"
    local tool_dir="$2"
    local features="$3"
    local receipt="${4:-${binary}.receipt}"
    local binary_sha rustc_command toolchain_file toolchain_sha

    receipt_require_executable "${binary}" || return 1
    [[ "${features}" =~ ^[a-z0-9]+(,[a-z0-9]+)*$ ]] || {
        receipt_fail "local build features are not canonical: ${features}"
        return 1
    }
    receipt_load_source_identity "${tool_dir}" || return 1
    binary_sha="$(receipt_sha256_file "${binary}")" || return 1

    # The Trust toolchain's compiler (owner directive 2026-09-28).
    rustc_command="${RUSTC:-trustc}"
    if ! command -v "${rustc_command}" >/dev/null 2>&1; then
        receipt_fail "could not locate the compiler used for receipt identity: ${rustc_command}"
        return 1
    fi
    toolchain_file="$(mktemp "${TMPDIR:-/tmp}/ny-rustc-vv.XXXXXX")" || {
        receipt_fail "could not stage rustc identity"
        return 1
    }
    if ! "${rustc_command}" -vV > "${toolchain_file}"; then
        rm -f -- "${toolchain_file}"
        receipt_fail "${rustc_command} -vV failed"
        return 1
    fi
    toolchain_sha="$(receipt_sha256_file "${toolchain_file}")" || {
        rm -f -- "${toolchain_file}"
        return 1
    }
    rm -f -- "${toolchain_file}"

    receipt_publish \
        "${receipt}" \
        "${binary_sha}" \
        "${NY_SOURCE_KIND}" \
        "${NY_SOURCE_COMMIT}" \
        "${NY_SOURCE_STATE_SHA256}" \
        "${NY_SOURCE_LOCK_SHA256}" \
        "${NY_SOURCE_AY_COMMIT}" \
        "${features}" \
        "rustc-vv" \
        "${toolchain_sha}" \
        "none"
}

receipt_create_prebuilt() {
    local binary="$1"
    local tool_dir="$2"
    local manifest="$3"
    local receipt="${4:-${binary}.receipt}"
    local binary_sha manifest_sha lock_sha ay_commit git_root marker

    receipt_require_executable "${binary}" || return 1
    receipt_parse_prebuilt_manifest "${manifest}" || return 1
    binary_sha="$(receipt_sha256_file "${binary}")" || return 1
    if [ "${binary_sha}" != "${NY_PREBUILT_BINARY}" ]; then
        receipt_fail "installed prebuilt bytes do not match binary_sha256"
        return 1
    fi
    lock_sha="$(receipt_sha256_file "${tool_dir}/Cargo.lock")" || return 1
    if [ "${lock_sha}" != "${NY_PREBUILT_LOCK}" ]; then
        receipt_fail "prebuilt Cargo.lock identity does not match the extracted package"
        return 1
    fi
    ay_commit="$(receipt_ay_commit "${tool_dir}/Cargo.lock")" || return 1
    if [ "${ay_commit}" != "${NY_PREBUILT_AY}" ]; then
        receipt_fail "prebuilt AY identity does not match Cargo.lock"
        return 1
    fi

    git_root="$(git -C "${tool_dir}" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "${git_root}" ] \
        && [ "$(realpath "${git_root}")" = "$(realpath "${tool_dir}")" ]; then
        if [ "$(git -C "${tool_dir}" rev-parse --verify HEAD)" != "${NY_PREBUILT_NY_COMMIT}" ]; then
            receipt_fail "prebuilt ny_commit does not match the checkout HEAD"
            return 1
        fi
    else
        marker="${tool_dir}/.ny-vnncomp-source.txt"
        if [ -f "${marker}" ] && [ ! -L "${marker}" ]; then
            receipt_parse_source_marker "${marker}" || return 1
            if [ "${NY_MARKER_COMMIT}" != "${NY_PREBUILT_NY_COMMIT}" ] \
                || [ "${NY_MARKER_LOCK_SHA256}" != "${NY_PREBUILT_LOCK}" ]; then
                receipt_fail "prebuilt provenance does not match the archive source marker"
                return 1
            fi
        fi
        # Older validated submission archives predate the source marker. Their
        # strict prebuilt manifest still binds ny_commit and remains accepted.
    fi

    manifest_sha="$(receipt_sha256_file "${manifest}")" || return 1
    receipt_publish \
        "${receipt}" \
        "${binary_sha}" \
        "prebuilt" \
        "${NY_PREBUILT_NY_COMMIT}" \
        "${manifest_sha}" \
        "${NY_PREBUILT_LOCK}" \
        "${NY_PREBUILT_AY}" \
        "${NY_PREBUILT_FEATURES}" \
        "trust-sealed" \
        "${NY_PREBUILT_TRUSTC}" \
        "${manifest_sha}"
}

# --- release artifacts: dist/bin/<target>/MANIFEST (scripts/release/make_submission.sh)
# vnncomp_scripts/verify_release.py is the full admission check; this parser
# reads only the identities a receipt binds, but it is just as strict about
# them: exact schema, one value per key, canonical hex.
receipt_parse_release_manifest() {
    local manifest="$1"
    local line key value seen="|"

    NY_RELEASE_SCHEMA=""
    NY_RELEASE_TARGET=""
    NY_RELEASE_FEATURES=""
    NY_RELEASE_NY_COMMIT=""
    NY_RELEASE_NY_COMMIT_TREE=""
    NY_RELEASE_NY_COMMIT_OBJECT=""
    NY_RELEASE_PATHS=""
    NY_RELEASE_LOCK=""
    NY_RELEASE_AY=""
    NY_RELEASE_BINARY=""
    NY_RELEASE_TRUSTC=""

    receipt_require_regular_file "${manifest}" "release MANIFEST" || return 1
    while IFS= read -r line; do
        case "${line}" in
            *=*) key="${line%%=*}"; value="${line#*=}" ;;
            *)
                receipt_fail "release MANIFEST has a line that is not key=value"
                return 1
                ;;
        esac
        case "${key}" in
            schema|target|features|ny_commit|ny_commit_tree|ny_commit_object|release_paths|cargo_lock_sha256|ay_lock_commit|binary_sha256|trustc_sha256) ;;
            *) continue ;;
        esac
        case "${seen}" in
            *"|${key}|"*)
                receipt_fail "release MANIFEST has duplicate key ${key}"
                return 1
                ;;
        esac
        seen="${seen}${key}|"
        case "${key}" in
            schema) NY_RELEASE_SCHEMA="${value}" ;;
            target) NY_RELEASE_TARGET="${value}" ;;
            features) NY_RELEASE_FEATURES="${value}" ;;
            ny_commit) NY_RELEASE_NY_COMMIT="${value}" ;;
            ny_commit_tree) NY_RELEASE_NY_COMMIT_TREE="${value}" ;;
            ny_commit_object) NY_RELEASE_NY_COMMIT_OBJECT="${value}" ;;
            release_paths) NY_RELEASE_PATHS="${value}" ;;
            cargo_lock_sha256) NY_RELEASE_LOCK="${value}" ;;
            ay_lock_commit) NY_RELEASE_AY="${value}" ;;
            binary_sha256) NY_RELEASE_BINARY="${value}" ;;
            trustc_sha256) NY_RELEASE_TRUSTC="${value}" ;;
        esac
    done < "${manifest}"
    if [ "${NY_RELEASE_SCHEMA}" != "${RELEASE_SCHEMA}" ] || [ "${NY_RELEASE_FEATURES}" != "mip,cuda" ]; then
        receipt_fail "release MANIFEST has the wrong schema or feature tier"
        return 1
    fi
    case "${NY_RELEASE_TARGET}" in
        x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu) ;;
        *)
            receipt_fail "release MANIFEST names an unsupported target"
            return 1
            ;;
    esac
    receipt_is_lower_hex "${NY_RELEASE_NY_COMMIT}" 40 && receipt_is_lower_hex "${NY_RELEASE_AY}" 40 \
        && receipt_is_lower_hex "${NY_RELEASE_NY_COMMIT_TREE}" 40 || {
        receipt_fail "release MANIFEST ny_commit/ny_commit_tree/ay_lock_commit is malformed"
        return 1
    }
    if ! [[ "${NY_RELEASE_NY_COMMIT_OBJECT}" =~ ^[A-Za-z0-9+/]+={0,2}$ ]]; then
        receipt_fail "release MANIFEST ny_commit_object is missing or not base64"
        return 1
    fi
    if [ "${NY_RELEASE_PATHS}" != "$(IFS=,; printf '%s' "${RELEASE_PATHS[*]}")" ]; then
        receipt_fail "release MANIFEST release_paths is not exactly the release files this receipt removes"
        return 1
    fi
    receipt_require_digest "${NY_RELEASE_LOCK}" "release cargo_lock_sha256" || return 1
    receipt_require_digest "${NY_RELEASE_BINARY}" "release binary_sha256" || return 1
    receipt_require_digest "${NY_RELEASE_TRUSTC}" "release trustc_sha256" || return 1
}

# The same package roots vnncomp_scripts/verify_release.py binds at install
# time (its PACKAGE_ROOTS; release_pipeline_black_box.rs pins the two lists
# equal): an untracked, non-ignored file under one of them is an input
# ny_commit does not cover -- the binary reads configs/ presets at run time.
RELEASE_PACKAGE_ROOTS=(
    ".cargo"
    "benchmarks/download_benchmarks.sh"
    "Cargo.lock"
    "Cargo.toml"
    "LICENSE"
    "README.md"
    "_typos.toml"
    "clippy.toml"
    "configs"
    "crates"
    "install_tool.sh"
    "prepare_instance.sh"
    "requirements.txt"
    "rust-toolchain.toml"
    "run_instance.sh"
    "scripts/vnncomp_coverage.py"
    "scripts/release"
    "vnncomp_scripts"
)

# Every path a release commit may add on top of ny_commit, for every target:
# the same list as vnncomp_scripts/verify_release.py RELEASE_PATHS and the
# `release_paths` make_submission.sh records (release_pipeline_black_box.rs
# pins the three equal).
RELEASE_PATHS=(
    "dist/bin/aarch64-unknown-linux-gnu/MANIFEST"
    "dist/bin/aarch64-unknown-linux-gnu/ny.sha256"
    "dist/bin/aarch64-unknown-linux-gnu/ny.xz"
    "dist/bin/x86_64-unknown-linux-gnu/MANIFEST"
    "dist/bin/x86_64-unknown-linux-gnu/ny.sha256"
    "dist/bin/x86_64-unknown-linux-gnu/ny.xz"
)

# `git` on the release checkout DIR (canonical, from realpath): read-only, and
# with DIR -- exactly DIR, never `*` -- as safe.directory, so a clone owned by
# another user than the one running the install or the instance is read rather
# than refused as "dubious ownership"; every comparison is the one the owner
# would run. git matches the entry against the worktree, so a linked
# worktree's gitdir (a gitfile target) is opened under it too. This trusts
# nothing new but makes the clone's own .git live for these calls (its config
# -- a core.fsmonitor hook there runs as the caller --, index flags,
# refs/replace, shallow file): whoever can write that .git can already edit
# this script and run_instance.sh, which the same caller runs from the same
# clone. The binding is of committed content under git's own clone state, never
# a defence against the clone's owner (docs/RELEASE.md).
receipt_git() { # dir git-arguments...
    local dir="$1"
    shift
    git -c "safe.directory=${dir}" --no-optional-locks -C "${dir}" "$@"
}

receipt_is_release_path() {
    local candidate
    for candidate in "${RELEASE_PATHS[@]}"; do
        [ "${candidate}" = "$1" ] && return 0
    done
    return 1
}

receipt_release_path_below() {
    local candidate
    for candidate in "${RELEASE_PATHS[@]}"; do
        [[ "${candidate}" == "$1"/* ]] && return 0
    done
    return 1
}

# Prints the id of TREE (at path PREFIX) with every RELEASE_PATHS file removed,
# or "-" when nothing is left (git never stores an empty directory, so an
# emptied one goes too). Only a regular file (100644/100755) at a release path
# is removed: a symlink, directory or submodule there stays, so the comparison
# refuses it. Computable in a shallow clone: it reads only HEAD's own trees and
# hashes the rebuilt ones without writing (`hash-object` without -w). The same
# computation as verify_release.py tree_without_release_paths().
receipt_tree_without_release_paths() { # dir tree prefix
    local dir="$1" tree="$2" prefix="$3"
    local listing="" staged="" record meta name mode kind oid path stripped escaped index
    local changed=0 problem=""
    listing="$(mktemp "${TMPDIR:-/tmp}/ny-release-tree.XXXXXX")" || {
        receipt_fail "could not stage a tree listing"
        return 1
    }
    staged="$(mktemp "${TMPDIR:-/tmp}/ny-release-tree.XXXXXX")" || {
        rm -f -- "${listing}"
        receipt_fail "could not stage a tree"
        return 1
    }
    if ! receipt_git "${dir}" ls-tree --full-tree -z "${tree}" > "${listing}"; then
        rm -f -- "${listing}" "${staged}"
        receipt_fail "could not list tree ${tree}"
        return 1
    fi
    while IFS= read -r -d '' record; do
        meta="${record%%$'\t'*}"
        name="${record#*$'\t'}"
        read -r mode kind oid <<<"${meta}"
        if [ "${meta}" = "${record}" ] || [ -z "${name}" ] || ! receipt_is_lower_hex "${oid}" 40 \
            || [ "${meta}" != "${mode} ${kind} ${oid}" ]; then
            problem="unexpected git ls-tree record under ${prefix:-<root>}"
            break
        fi
        path="${prefix}${name}"
        if [ "${kind}" = blob ] && { [ "${mode}" = 100644 ] || [ "${mode}" = 100755 ]; } \
            && receipt_is_release_path "${path}"; then
            changed=1
            continue
        fi
        if [ "${kind}" = tree ] && receipt_release_path_below "${path}"; then
            if ! stripped="$(receipt_tree_without_release_paths "${dir}" "${oid}" "${path}/")"; then
                problem="could not strip the release files below ${path}"
                break
            fi
            [ "${stripped}" = "${oid}" ] || changed=1
            [ "${stripped}" = "-" ] && continue
            oid="${stripped}"
        fi
        escaped=""
        for ((index = 0; index < 40; index += 2)); do
            escaped+="\\x${oid:index:2}"
        done
        # shellcheck disable=SC2059 # the format is only \xHH escapes of a checked 40-hex id
        if ! { printf '%s %s\0' "${mode#0}" "${name}"; printf "${escaped}"; } >> "${staged}"; then
            problem="could not stage tree ${tree}"
            break
        fi
    done < "${listing}"
    rm -f -- "${listing}"
    if [ -n "${problem}" ]; then
        rm -f -- "${staged}"
        receipt_fail "${problem}"
        return 1
    fi
    if [ "${changed}" -eq 0 ]; then
        rm -f -- "${staged}"
        printf '%s\n' "${tree}"
        return 0
    fi
    if [ ! -s "${staged}" ]; then
        rm -f -- "${staged}"
        printf -- '-\n'
        return 0
    fi
    if ! stripped="$(receipt_git "${dir}" hash-object -t tree --stdin < "${staged}")" \
        || ! receipt_is_lower_hex "${stripped}" 40; then
        rm -f -- "${staged}"
        receipt_fail "could not hash the tree rebuilt from ${tree}"
        return 1
    fi
    rm -f -- "${staged}"
    printf '%s\n' "${stripped}"
}

# MANIFEST's ny_commit_object is ny_commit's raw commit object (base64): its git
# SHA-1 must be ny_commit -- the commit the binary embeds -- and its first
# header must name ny_commit_tree. That binds the recorded tree id to ny_commit
# without any history, so a shallow clone whose MANIFEST re-points
# ny_commit_tree at an edited source tree is refused as a full clone refuses
# it. The same check as verify_release.py verify_commit_object().
receipt_release_commit_object_binds() { # dir ny_commit ny_commit_tree object-base64
    local dir="$1" ny_commit="$2" ny_commit_tree="$3" encoded="$4"
    local decoded hashed header="" size
    decoded="$(mktemp "${TMPDIR:-/tmp}/ny-release-commit.XXXXXX")" || {
        receipt_fail "could not stage the ny_commit object"
        return 1
    }
    if ! printf '%s' "${encoded}" | base64 -d > "${decoded}" 2>/dev/null; then
        rm -f -- "${decoded}"
        receipt_fail "release MANIFEST ny_commit_object does not decode"
        return 1
    fi
    size="$(wc -c < "${decoded}")"
    if [ "${size}" -gt 32768 ]; then
        rm -f -- "${decoded}"
        receipt_fail "release MANIFEST ny_commit_object is larger than 32768 bytes"
        return 1
    fi
    hashed="$(receipt_git "${dir}" hash-object --literally -t commit --stdin < "${decoded}" 2>/dev/null || true)"
    IFS= read -r header < "${decoded}" || true
    rm -f -- "${decoded}"
    if [ "${hashed}" != "${ny_commit}" ]; then
        receipt_fail "release MANIFEST ny_commit_object hashes to ${hashed:-<nothing>}, not ny_commit ${ny_commit}"
        return 1
    fi
    if [ "${header}" != "tree ${ny_commit_tree}" ]; then
        receipt_fail "ny_commit ${ny_commit} names '${header}', not release ny_commit_tree ${ny_commit_tree}"
        return 1
    fi
}

# The checkout must be ny_commit's source plus at most the release files, and it
# must still be the UNMODIFIED checkout verify_release.py admitted at install
# time: no tracked edit relative to HEAD and no untracked file under a package
# root. Without that, editing a preset under configs/ after the install would
# change what every scored run does while this receipt still validated.
# "ny_commit's source" is bound by content in every case -- HEAD's tree with the
# release files removed is MANIFEST's ny_commit_tree, which MANIFEST's
# ny_commit_object proves is ny_commit's own tree -- so a SHALLOW clone of the
# release commit (no ny_commit object) binds as a full one does; when ny_commit
# IS present it must also have that tree and, unless HEAD is ny_commit, be an
# ancestor of HEAD that differs only in release files.
# Read-only (--no-optional-locks: no index refresh). An extracted archive is
# bound by its packaged source marker instead.
receipt_release_source_binding() {
    local tool_dir="$1" ny_commit="$2" ny_commit_tree="$3" ny_commit_object="$4"
    local canonical git_root head head_tree commit_tree source_tree path marker stray diff_rc=0 present=0
    canonical="$(realpath "${tool_dir}")" || {
        receipt_fail "could not resolve ${tool_dir}"
        return 1
    }
    git_root="$(receipt_git "${canonical}" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "${git_root}" ] && [ "$(realpath "${git_root}")" = "${canonical}" ]; then
        receipt_git "${canonical}" diff --quiet HEAD -- >/dev/null 2>&1 || diff_rc=$?
        if [ "${diff_rc}" -ne 0 ]; then
            receipt_fail "tracked files are modified relative to HEAD (git diff rc=${diff_rc}); the release binds an unmodified checkout"
            return 1
        fi
        if ! stray="$(receipt_git "${canonical}" ls-files --others --exclude-standard \
            -- "${RELEASE_PACKAGE_ROOTS[@]}" 2>/dev/null)"; then
            receipt_fail "could not list untracked files under the release package roots"
            return 1
        fi
        if [ -n "${stray}" ]; then
            receipt_fail "untracked files under the release package roots are not covered by ny_commit: $(printf '%s\n' "${stray}" | head -5 | paste -sd' ' -)"
            return 1
        fi
        head="$(receipt_git "${canonical}" rev-parse --verify 'HEAD^{commit}' 2>/dev/null || true)"
        if ! receipt_is_lower_hex "${head}" 40; then
            receipt_fail "could not resolve the checkout HEAD"
            return 1
        fi
        if receipt_git "${canonical}" cat-file -e "${ny_commit}^{commit}" 2>/dev/null; then
            present=1
            commit_tree="$(receipt_git "${canonical}" rev-parse --verify "${ny_commit}^{tree}" 2>/dev/null || true)"
            if [ "${commit_tree}" != "${ny_commit_tree}" ]; then
                receipt_fail "release MANIFEST ny_commit_tree ${ny_commit_tree} is not the tree of ny_commit ${ny_commit} (${commit_tree:-unresolved})"
                return 1
            fi
        fi
        if [ "${head}" = "${ny_commit}" ]; then
            :
        elif [ "${present}" = 1 ] \
            && receipt_git "${canonical}" merge-base --is-ancestor "${ny_commit}" "${head}" 2>/dev/null; then
            while IFS= read -r -d '' path; do
                receipt_is_release_path "${path}" && continue
                receipt_fail "HEAD differs from release ny_commit outside the release files: ${path}"
                return 1
            done < <(receipt_git "${canonical}" diff --name-only -z "${ny_commit}" "${head}" --)
        elif [ "$(receipt_git "${canonical}" rev-parse --is-shallow-repository 2>/dev/null)" = true ]; then
            # `git clone --depth 1` of the release commit: history above HEAD is
            # cut, so ancestry is unknowable; the tree check below binds it.
            :
        else
            receipt_fail "release ny_commit ${ny_commit} is neither HEAD nor an ancestor of HEAD ${head}"
            return 1
        fi
        head_tree="$(receipt_git "${canonical}" rev-parse --verify 'HEAD^{tree}' 2>/dev/null || true)"
        if ! receipt_is_lower_hex "${head_tree}" 40; then
            receipt_fail "could not resolve the checkout HEAD tree"
            return 1
        fi
        source_tree="$(receipt_tree_without_release_paths "${canonical}" "${head_tree}" "")" || return 1
        if [ "${source_tree}" != "${ny_commit_tree}" ]; then
            receipt_fail "HEAD ${head}'s tree without the release files is ${source_tree}, not release ny_commit_tree ${ny_commit_tree}: the checkout is not ny_commit's source plus release files"
            return 1
        fi
        receipt_release_commit_object_binds "${canonical}" "${ny_commit}" "${ny_commit_tree}" "${ny_commit_object}" \
            || return 1
        return 0
    fi
    marker="${tool_dir}/.ny-vnncomp-source.txt"
    if [ ! -f "${marker}" ] || [ -L "${marker}" ]; then
        receipt_fail "no Git checkout and no ${marker}: the release source identity is unknown"
        return 1
    fi
    receipt_parse_source_marker "${marker}" || return 1
    if [ "${NY_MARKER_COMMIT}" != "${ny_commit}" ]; then
        receipt_fail "the archive source marker names ${NY_MARKER_COMMIT}, the release names ${ny_commit}"
        return 1
    fi
}

receipt_release_identity_matches_checkout() {
    local tool_dir="$1"
    local lock_sha ay_commit
    lock_sha="$(receipt_sha256_file "${tool_dir}/Cargo.lock")" || return 1
    ay_commit="$(receipt_ay_commit "${tool_dir}/Cargo.lock")" || return 1
    if [ "${lock_sha}" != "${NY_RELEASE_LOCK}" ] || [ "${ay_commit}" != "${NY_RELEASE_AY}" ]; then
        receipt_fail "release Cargo.lock/AY identity does not match this checkout"
        return 1
    fi
    receipt_release_source_binding "${tool_dir}" "${NY_RELEASE_NY_COMMIT}" "${NY_RELEASE_NY_COMMIT_TREE}" \
        "${NY_RELEASE_NY_COMMIT_OBJECT}"
}

receipt_create_release() {
    local binary="$1"
    local tool_dir="$2"
    local manifest="$3"
    local receipt="${4:-${binary}.receipt}"
    local binary_sha manifest_sha

    receipt_require_executable "${binary}" || return 1
    receipt_parse_release_manifest "${manifest}" || return 1
    if [ "$(realpath "$(dirname "${manifest}")")" != "$(realpath "${tool_dir}/dist/bin/${NY_RELEASE_TARGET}")" ]; then
        receipt_fail "release MANIFEST must be ${tool_dir}/dist/bin/${NY_RELEASE_TARGET}/MANIFEST"
        return 1
    fi
    binary_sha="$(receipt_sha256_file "${binary}")" || return 1
    if [ "${binary_sha}" != "${NY_RELEASE_BINARY}" ]; then
        receipt_fail "the release binary's bytes do not match binary_sha256"
        return 1
    fi
    receipt_release_identity_matches_checkout "${tool_dir}" || return 1
    manifest_sha="$(receipt_sha256_file "${manifest}")" || return 1
    receipt_publish \
        "${receipt}" \
        "${binary_sha}" \
        "release" \
        "${NY_RELEASE_NY_COMMIT}" \
        "${manifest_sha}" \
        "${NY_RELEASE_LOCK}" \
        "${NY_RELEASE_AY}" \
        "${NY_RELEASE_FEATURES}" \
        "trust-unverified" \
        "${NY_RELEASE_TRUSTC}" \
        "${manifest_sha}"
}

receipt_validate() {
    local binary="$1"
    local tool_dir="$2"
    local receipt="${3:-${binary}.receipt}"
    local binary_sha manifest manifest_sha lock_sha ay_commit git_root marker candidate

    receipt_require_executable "${binary}" || return 1
    receipt_parse "${receipt}" || return 1
    binary_sha="$(receipt_sha256_file "${binary}")" || return 1
    if [ "${binary_sha}" != "${NY_RECEIPT_BINARY}" ]; then
        receipt_fail "stale/mismatched binary: expected ${NY_RECEIPT_BINARY}, found ${binary_sha}"
        return 1
    fi

    case "${NY_RECEIPT_SOURCE_KIND}" in
        git|archive)
            receipt_load_source_identity "${tool_dir}" || return 1
            if [ "${NY_SOURCE_KIND}" != "${NY_RECEIPT_SOURCE_KIND}" ] \
                || [ "${NY_SOURCE_COMMIT}" != "${NY_RECEIPT_SOURCE_COMMIT}" ] \
                || [ "${NY_SOURCE_STATE_SHA256}" != "${NY_RECEIPT_SOURCE_STATE}" ] \
                || [ "${NY_SOURCE_LOCK_SHA256}" != "${NY_RECEIPT_LOCK}" ] \
                || [ "${NY_SOURCE_AY_COMMIT}" != "${NY_RECEIPT_AY}" ]; then
                receipt_fail "stale source identity: binary receipt does not match the current NY source state"
                return 1
            fi
            ;;
        prebuilt)
            manifest="${tool_dir}/dist/bin/ny-x86_64-linux.provenance.txt"
            receipt_parse_prebuilt_manifest "${manifest}" || return 1
            manifest_sha="$(receipt_sha256_file "${manifest}")" || return 1
            if [ "${manifest_sha}" != "${NY_RECEIPT_ARTIFACT_PROVENANCE}" ] \
                || [ "${manifest_sha}" != "${NY_RECEIPT_SOURCE_STATE}" ] \
                || [ "${NY_PREBUILT_BINARY}" != "${NY_RECEIPT_BINARY}" ] \
                || [ "${NY_PREBUILT_NY_COMMIT}" != "${NY_RECEIPT_SOURCE_COMMIT}" ] \
                || [ "${NY_PREBUILT_LOCK}" != "${NY_RECEIPT_LOCK}" ] \
                || [ "${NY_PREBUILT_AY}" != "${NY_RECEIPT_AY}" ] \
                || [ "${NY_PREBUILT_FEATURES}" != "${NY_RECEIPT_FEATURES}" ] \
                || [ "${NY_PREBUILT_TRUSTC}" != "${NY_RECEIPT_TOOLCHAIN}" ]; then
                receipt_fail "prebuilt receipt does not match its sealed provenance manifest"
                return 1
            fi
            lock_sha="$(receipt_sha256_file "${tool_dir}/Cargo.lock")" || return 1
            ay_commit="$(receipt_ay_commit "${tool_dir}/Cargo.lock")" || return 1
            if [ "${lock_sha}" != "${NY_RECEIPT_LOCK}" ] \
                || [ "${ay_commit}" != "${NY_RECEIPT_AY}" ]; then
                receipt_fail "prebuilt receipt does not match the extracted Cargo.lock/AY identity"
                return 1
            fi
            git_root="$(git -C "${tool_dir}" rev-parse --show-toplevel 2>/dev/null || true)"
            if [ -n "${git_root}" ] \
                && [ "$(realpath "${git_root}")" = "$(realpath "${tool_dir}")" ]; then
                if [ "$(git -C "${tool_dir}" rev-parse --verify HEAD)" != "${NY_RECEIPT_SOURCE_COMMIT}" ]; then
                    receipt_fail "prebuilt receipt source_commit does not match checkout HEAD"
                    return 1
                fi
            else
                marker="${tool_dir}/.ny-vnncomp-source.txt"
                if [ -f "${marker}" ] && [ ! -L "${marker}" ]; then
                    receipt_parse_source_marker "${marker}" || return 1
                    if [ "${NY_MARKER_COMMIT}" != "${NY_RECEIPT_SOURCE_COMMIT}" ] \
                        || [ "${NY_MARKER_LOCK_SHA256}" != "${NY_RECEIPT_LOCK}" ]; then
                        receipt_fail "prebuilt receipt does not match the archive source marker"
                        return 1
                    fi
                fi
            fi
            ;;
        release)
            # The receipt names its MANIFEST by content: exactly one
            # dist/bin/<target>/MANIFEST must still hash to it.
            manifest=""
            for candidate in "${tool_dir}"/dist/bin/*/MANIFEST; do
                [ -f "${candidate}" ] && [ ! -L "${candidate}" ] || continue
                if [ "$(receipt_sha256_file "${candidate}")" = "${NY_RECEIPT_ARTIFACT_PROVENANCE}" ]; then
                    if [ -n "${manifest}" ]; then
                        receipt_fail "more than one release MANIFEST matches this receipt"
                        return 1
                    fi
                    manifest="${candidate}"
                fi
            done
            if [ -z "${manifest}" ]; then
                receipt_fail "the release MANIFEST this receipt was bound to is gone or has changed"
                return 1
            fi
            receipt_parse_release_manifest "${manifest}" || return 1
            if [ "${NY_RECEIPT_ARTIFACT_PROVENANCE}" != "${NY_RECEIPT_SOURCE_STATE}" ] \
                || [ "${NY_RELEASE_BINARY}" != "${NY_RECEIPT_BINARY}" ] \
                || [ "${NY_RELEASE_NY_COMMIT}" != "${NY_RECEIPT_SOURCE_COMMIT}" ] \
                || [ "${NY_RELEASE_LOCK}" != "${NY_RECEIPT_LOCK}" ] \
                || [ "${NY_RELEASE_AY}" != "${NY_RECEIPT_AY}" ] \
                || [ "${NY_RELEASE_FEATURES}" != "${NY_RECEIPT_FEATURES}" ] \
                || [ "${NY_RELEASE_TRUSTC}" != "${NY_RECEIPT_TOOLCHAIN}" ] \
                || [ "${NY_RECEIPT_TOOLCHAIN_KIND}" != "trust-unverified" ]; then
                receipt_fail "release receipt does not match its MANIFEST"
                return 1
            fi
            receipt_release_identity_matches_checkout "${tool_dir}" || return 1
            ;;
    esac

    printf 'NY binary receipt OK: sha256=%s source=%s@%s features=%s toolchain=%s:%s\n' \
        "${NY_RECEIPT_BINARY}" \
        "${NY_RECEIPT_SOURCE_KIND}" \
        "${NY_RECEIPT_SOURCE_COMMIT}" \
        "${NY_RECEIPT_FEATURES}" \
        "${NY_RECEIPT_TOOLCHAIN_KIND}" \
        "${NY_RECEIPT_TOOLCHAIN}"
}

usage() {
    echo "usage:" >&2
    echo "  $0 identity TOOL_DIR" >&2
    echo "  $0 create-local BINARY TOOL_DIR FEATURES [RECEIPT]" >&2
    echo "  $0 create-prebuilt BINARY TOOL_DIR MANIFEST [RECEIPT]" >&2
    echo "  $0 create-release BINARY TOOL_DIR MANIFEST [RECEIPT]" >&2
    echo "  $0 validate BINARY TOOL_DIR [RECEIPT]" >&2
    exit 2
}

case "${1:-}" in
    identity)
        [ "$#" -eq 2 ] || usage
        receipt_load_source_identity "$2"
        printf 'source_kind=%s\nsource_commit=%s\nsource_state_sha256=%s\ncargo_lock_sha256=%s\nay_commit=%s\n' \
            "${NY_SOURCE_KIND}" \
            "${NY_SOURCE_COMMIT}" \
            "${NY_SOURCE_STATE_SHA256}" \
            "${NY_SOURCE_LOCK_SHA256}" \
            "${NY_SOURCE_AY_COMMIT}"
        ;;
    create-local)
        [ "$#" -eq 4 ] || [ "$#" -eq 5 ] || usage
        receipt_create_local "$2" "$3" "$4" "${5:-${2}.receipt}"
        ;;
    create-prebuilt)
        [ "$#" -eq 4 ] || [ "$#" -eq 5 ] || usage
        receipt_create_prebuilt "$2" "$3" "$4" "${5:-${2}.receipt}"
        ;;
    create-release)
        [ "$#" -eq 4 ] || [ "$#" -eq 5 ] || usage
        receipt_create_release "$2" "$3" "$4" "${5:-${2}.receipt}"
        ;;
    validate)
        [ "$#" -eq 3 ] || [ "$#" -eq 4 ] || usage
        receipt_validate "$2" "$3" "${4:-${2}.receipt}"
        ;;
    *)
        usage
        ;;
esac
