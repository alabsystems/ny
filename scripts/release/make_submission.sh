#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# make_submission.sh -- the ONE release command for a VNN-COMP submission binary.
#
# From a CLEAN checkout at one commit it builds the competition binary for each
# requested target with the Trust toolchain in the unverified lane
# (`rustup run trust targo --unverified build --locked --offline --release
# -p ny-cli --bin ny --features mip,cuda` -- built WITH Trust, not verified BY
# it), strips it, makes the binary identify itself, runs the install self-test
# on it, and writes
#
#     dist/bin/<target>/ny          stripped executable (local use; not committed)
#     dist/bin/<target>/ny.xz       the shippable form (GitHub refuses >100 MB files)
#     dist/bin/<target>/ny.sha256   "<sha256>  ny"
#     dist/bin/<target>/MANIFEST    strict key=value record, schema ny-release-manifest-v1
#
# and then runs the installer's own verifier (vnncomp_scripts/verify_release.py)
# over what it wrote. `install_tool.sh` repeats that verification on the
# evaluation host before it trusts a byte. See docs/RELEASE.md.
#
# Targets and lanes (DECLARED in the manifest as build_lane, never relabelled):
#   aarch64-unknown-linux-gnu on an aarch64 host   build_lane=native
#   x86_64-unknown-linux-gnu  on an x86_64 host    build_lane=native
#   x86_64-unknown-linux-gnu  on an aarch64 host   build_lane=cross-build-std
#       -- the path that produced ny-pins/ny_x86_64_96e234a02 (2026-09-28):
#       Trust's own x86 codegen, std rebuilt from the compiler's commit with
#       -Zbuild-std, clang-18 + lld against an Ubuntu 24.04 (glibc 2.39)
#       sysroot. Prepared once by scripts/release/prepare_x86_cross_env.sh.
#   Neither lane is the seed-mode sealed builder (scripts/vnncomp_trust_linux_build.sh);
#   the binary keeps the UNSEALED provenance record and verify_release.py refuses
#   a sealed one.
#
# Discipline (CLAUDE.md): run it inside the box lock,
#     NY_BOX_WAIT_LOCK=21600 NY_BOX_WAIT_QUIET=21600 \
#       scripts/ny_box.sh build -- scripts/release/make_submission.sh --commit <sha>
# Step 0 (`--preflight-only`) takes no lock and fails closed in seconds.
#
# Output: one `MAKE_SUBMISSION_TARGET ...` line per target and a final
#     MAKE_SUBMISSION_DONE rc=<worst> WORST=<worst> <target>=<rc>... elapsed=<s>
# whose rc is the worst step rc (never the rc of a trailing echo).

set -uo pipefail

SELF="$(realpath "${BASH_SOURCE[0]}")"
SRC="$(git -C "$(dirname "${SELF}")" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "ERROR: make_submission.sh must live inside a Git checkout" >&2
    exit 2
}
T0=$(date +%s)
RUNID=$(date -u +%Y%m%dT%H%M%SZ)

COMMIT=""
TARGETS="x86_64-unknown-linux-gnu,aarch64-unknown-linux-gnu"
OUT_DIR="${SRC}/dist/bin"
TARGET_BASE="${SRC}/target/ship"
LOG_DIR=""
CROSS_ENV="${HOME}/.cache/ny-release/x86_64-cross"
QEMU="${HOME}/x86emu/qemu11u/usr/bin/qemu-x86_64"
QEMU_SYSROOT="${HOME}/x86emu/sysroot"
QEMU_CPU="Haswell-v4"
ORT_CACHES=()
PREFLIGHT_ONLY=0
REPLACE=0
WARM_ORT=0

usage() {
    cat <<EOF
usage: scripts/release/make_submission.sh [options]
  --commit SHA          the commit to release (default: HEAD); HEAD must equal it
  --targets LIST        comma list (default: ${TARGETS})
  --out-dir DIR         artifact root (default: <repo>/dist/bin)
  --target-dir DIR      cargo target dir base, one subdir per target (default: <repo>/target/ship)
  --log-dir DIR         build logs (default: <repo>/dist/logs/<commit9>-<runid>)
  --cross-env DIR       prepared x86_64 cross environment (default: ${CROSS_ENV})
  --qemu PATH           qemu-x86_64 for x86_64 self-identification on a non-x86 host
  --qemu-sysroot DIR    x86_64 runtime sysroot for qemu -L
  --ort-cache DIR       extra ONNX Runtime cache root to search (repeatable)
  --warm-ort            download the pinned ONNX Runtime archive when no cache has it (network)
  --replace             move an existing dist/bin/<target> aside instead of refusing
  --preflight-only      run step 0 and stop (no lock needed, no build)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --commit) COMMIT="${2:?}"; shift 2 ;;
        --targets) TARGETS="${2:?}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
        --target-dir) TARGET_BASE="${2:?}"; shift 2 ;;
        --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
        --cross-env) CROSS_ENV="${2:?}"; shift 2 ;;
        --qemu) QEMU="${2:?}"; shift 2 ;;
        --qemu-sysroot) QEMU_SYSROOT="${2:?}"; shift 2 ;;
        --ort-cache) ORT_CACHES+=("${2:?}"); shift 2 ;;
        --warm-ort) WARM_ORT=1; shift ;;
        --replace) REPLACE=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "${SRC}" || exit 2
FAIL=0
fail0() { echo "PREFLIGHT_FAIL: $*"; FAIL=1; }
ok0() { echo "preflight ok: $*"; }
sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
sha_text() { printf '%s' "$1" | sha256sum | awk '{print $1}'; }

# BEGIN release-build-hygiene (sourced on its own by
# crates/ny-test-utils/tests/release_pipeline_black_box.rs: functions and the
# release-path list only)
#
# Nothing from the caller may steer the build or the self-identification: no
# solver policy, no sealed-provenance claim, no compiler, flag or profile
# override. Cargo reads EVERY `CARGO_*` name as configuration: an explicit list
# of the spellings someone thought of misses the others
# (CARGO_BUILD_RUSTC_WRAPPER is RUSTC_WRAPPER by another name, and a wrapper
# injects flags that the build log's `Running` line, which the post-build
# check reads, never shows; CARGO_PROFILE_RELEASE_* rewrites the release
# profile). cc-rs reads the C/C++ compiler and flag families, ort-sys reads
# ORT_*, AY's build scripts read AY_*, and the scored closure removes
# MIMALLOC_*/RAYON_* before the self-test the verifier runs under. Only
# CARGO_HOME (where the locked sources live) and the box's own NY_BOX_* knobs
# survive; this script sets every value the build needs itself.
scrub_build_environment() {
    local name
    for name in $(compgen -e); do
        case "${name}" in
            NY_BOX_* | CARGO_HOME) ;;
            NY_* | AY_* | MIMALLOC_* | CARGO_* | __CARGO_* | ORT_* \
                | RUSTFLAGS | RUSTDOCFLAGS | RUSTC | RUSTC_WRAPPER | RUSTC_WORKSPACE_WRAPPER \
                | RUSTC_BOOTSTRAP | RUSTUP_TOOLCHAIN | RAYON_NUM_THREADS | RAYON_RS_NUM_CPUS \
                | PKG_CONFIG_ALLOW_CROSS | CRATE_CC_NO_DEFAULTS \
                | CC | CXX | AR | RANLIB | CFLAGS | CXXFLAGS | CPPFLAGS | LDFLAGS \
                | CC_* | CXX_* | AR_* | CFLAGS_* | CXXFLAGS_* | CPPFLAGS_* | LDFLAGS_* \
                | TARGET_CC | TARGET_CXX | TARGET_AR | TARGET_CFLAGS | TARGET_CXXFLAGS \
                | HOST_CC | HOST_CXX | HOST_AR | HOST_CFLAGS | HOST_CXXFLAGS)
                unset "${name}"
                ;;
        esac
    done
}

# Cargo also merges configuration FILES: $CARGO_HOME/config.toml and the
# .cargo/config.toml of every ANCESTOR of the working directory. Only the
# checkout's own .cargo/config.toml is part of ny_commit, so a release worktree
# nested inside another checkout, or a user-level config with rustflags or a
# [profile.release] table, would build different code under the same identity.
# Prints every configuration file outside the checkout that cargo would read.
external_cargo_configs() { # checkout-root
    local dir name home="${CARGO_HOME:-${HOME}/.cargo}"
    {
        for name in config config.toml; do
            [ -e "${home}/${name}" ] && printf '%s\n' "${home}/${name}"
        done
        dir="$(dirname "$1")"
        while :; do
            for name in config config.toml; do
                [ -e "${dir%/}/.cargo/${name}" ] && printf '%s\n' "${dir%/}/.cargo/${name}"
            done
            [ "${dir}" = / ] && break
            dir="$(dirname "${dir}")"
        done
    } | sort -u
}

# A git-sourced dependency (AY, today the only one) compiles from cargo's own
# checkout under $CARGO_HOME/git/checkouts, which cargo never re-verifies once
# `.cargo-ok` exists, and this script stamps AY as AY_SOURCE_GIT_DIRTY=false.
# So the checkout must BE the locked commit with no tracked edit. Read-only
# (--no-optional-locks: no index refresh in the shared cargo cache). Prints the
# checkout root, or the reason it is refused.
git_dependency_checkout_state() { # package-dir locked-commit
    local top head
    top="$(git --no-optional-locks -C "$1" rev-parse --show-toplevel 2>/dev/null)" \
        || { echo "${1} is not inside a git checkout"; return 1; }
    head="$(git --no-optional-locks -C "${top}" rev-parse --verify 'HEAD^{commit}' 2>/dev/null)"
    [ "${head}" = "$2" ] || { echo "${top} is at '${head:-no HEAD}', not the locked ${2}"; return 1; }
    git --no-optional-locks -C "${top}" diff --quiet HEAD -- 2>/dev/null \
        || { echo "${top} has tracked files modified relative to ${2}"; return 1; }
    printf '%s\n' "${top}"
}

# Every path a release commit may add on top of ny_commit, for every target (an
# operator may commit both targets' files). Recorded in MANIFEST as
# `release_paths`; vnncomp_scripts/verify_release.py RELEASE_PATHS and
# submission_binary_receipt.sh RELEASE_PATHS are the same list
# (release_pipeline_black_box.rs pins the three equal).
RELEASE_PATHS=(
    "dist/bin/aarch64-unknown-linux-gnu/MANIFEST"
    "dist/bin/aarch64-unknown-linux-gnu/ny.sha256"
    "dist/bin/aarch64-unknown-linux-gnu/ny.xz"
    "dist/bin/x86_64-unknown-linux-gnu/MANIFEST"
    "dist/bin/x86_64-unknown-linux-gnu/ny.sha256"
    "dist/bin/x86_64-unknown-linux-gnu/ny.xz"
)

# The tree id MANIFEST records as `ny_commit_tree`. The installer and the
# runtime receipt compare it with HEAD's tree minus RELEASE_PATHS, which needs
# no history, so a SHALLOW clone of the release commit (`git clone --depth 1`,
# no ny_commit object) is bound by content exactly as a full clone is. A commit
# that already TRACKS a release path cannot be bound that way (the comparison
# removes it from HEAD but it stays in the recorded tree), so it is refused here
# instead of at install time. Run in the checkout; prints the tree id, or the
# reason it is refused.
release_commit_tree() { # commit
    local tree tracked
    tree="$(git rev-parse --verify "$1^{tree}" 2>/dev/null)" && [[ "${tree}" =~ ^[0-9a-f]{40}$ ]] \
        || { echo "cannot resolve the tree of $1"; return 1; }
    tracked="$(git ls-tree -r --full-tree --name-only "$1" -- "${RELEASE_PATHS[@]}" 2>/dev/null)" \
        || { echo "cannot list the release paths of $1"; return 1; }
    [ -z "${tracked}" ] \
        || { echo "$1 already tracks release files: $(printf '%s\n' "${tracked}" | paste -sd' ' -)"; return 1; }
    printf '%s\n' "${tree}"
}

# MANIFEST `ny_commit_object`: the commit's raw commit object, base64 on one
# line. Its git SHA-1 is the commit the binary embeds and its first header names
# `ny_commit_tree`, so the installer binds that tree id to ny_commit with no
# history (a shallow clone lacks the ny_commit object itself). At most 32768
# bytes, the verifier's bound, so a MANIFEST stays inside its 64 KiB. Run in the
# checkout; prints the encoding, or the reason it is refused.
release_commit_object() { # commit
    local size encoded
    size="$(git cat-file -s "$1^{commit}" 2>/dev/null)" && [[ "${size}" =~ ^[0-9]+$ ]] \
        || { echo "cannot read the commit object of $1"; return 1; }
    [ "${size}" -le 32768 ] \
        || { echo "the commit object of $1 is ${size} bytes, over the 32768-byte MANIFEST bound (shorten its message)"; return 1; }
    encoded="$(git cat-file commit "$1^{commit}" | base64 -w0)" && [ -n "${encoded}" ] \
        || { echo "cannot encode the commit object of $1"; return 1; }
    printf '%s\n' "${encoded}"
}
# END release-build-hygiene

scrub_build_environment

echo "make_submission: runid=${RUNID} src=${SRC} host=$(uname -m) pid=$$"

# ------------------------------------------------------------------ step 0 --
for tool in git rustup sha256sum readelf xz python3 timeout awk sed tar find base64; do
    command -v "${tool}" >/dev/null 2>&1 && ok0 "tool ${tool}" || fail0 "tool ${tool} missing"
done

HEAD="$(git rev-parse --verify HEAD)"
[ -n "${COMMIT}" ] || COMMIT="${HEAD}"
COMMIT="$(git rev-parse --verify "${COMMIT}^{commit}" 2>/dev/null || echo "${COMMIT}")"
[ "${HEAD}" = "${COMMIT}" ] && ok0 "HEAD is ${COMMIT}" \
    || fail0 "HEAD ${HEAD} is not the release commit ${COMMIT} (check it out in a clean worktree)"
if NY_COMMIT_TREE="$(release_commit_tree "${COMMIT}")"; then
    ok0 "ny_commit_tree ${NY_COMMIT_TREE} (what a shallow clone of the release commit is bound to)"
else
    fail0 "${NY_COMMIT_TREE}"
fi
if NY_COMMIT_OBJECT="$(release_commit_object "${COMMIT}")"; then
    ok0 "ny_commit_object (${#NY_COMMIT_OBJECT} base64 bytes: binds ny_commit_tree to ${COMMIT:0:9} with no history)"
else
    fail0 "${NY_COMMIT_OBJECT}"
fi
RELEASE_PATHS_CSV="$(IFS=,; printf '%s' "${RELEASE_PATHS[*]}")"
DIRT="$(git status --porcelain=v1 --untracked-files=all)"
if [ -z "${DIRT}" ]; then
    ok0 "checkout is clean (tracked and untracked)"
else
    fail0 "checkout is dirty; a release is built only from a clean tree:"
    printf '%s\n' "${DIRT}" | head -20 | sed 's/^/    /'
fi
LOCK_SHA256="$(sha_file Cargo.lock)"
AY_PAIRS="$(grep -oE 'alabsystems/ay\.git\?rev=[0-9a-f]{40}#[0-9a-f]{40}' Cargo.lock \
    | sed -E 's/.*rev=([0-9a-f]{40})#([0-9a-f]{40})/\1 \2/' | sort -u)"
if [ "$(printf '%s\n' "${AY_PAIRS}" | grep -c .)" = 1 ] \
    && [ "$(printf '%s\n' "${AY_PAIRS}" | awk '{print ($1==$2)}')" = 1 ]; then
    AY_COMMIT="$(printf '%s\n' "${AY_PAIRS}" | awk '{print $1}')"
    ok0 "AY lock commit ${AY_COMMIT}"
else
    AY_COMMIT=""
    fail0 "Cargo.lock does not pin exactly one canonical AY revision"
fi

TRUSTC_VV="$(rustup run trust rustc -vV 2>&1)" || fail0 "rustup run trust rustc -vV failed: ${TRUSTC_VV}"
TRUSTC_COMMIT="$(printf '%s\n' "${TRUSTC_VV}" | awk '/^commit-hash:/{print $2}')"
BUILD_HOST="$(printf '%s\n' "${TRUSTC_VV}" | awk '/^host:/{print $2}')"
SYSROOT="$(rustup run trust rustc --print sysroot 2>/dev/null)"
TRUSTC_BIN="${SYSROOT}/bin/trustc"
[ -x "${TRUSTC_BIN}" ] || TRUSTC_BIN="${SYSROOT}/bin/rustc"
RUSTLIB_BIN="${SYSROOT}/lib/rustlib/${BUILD_HOST}/bin"
TARGO_VERSION="$(rustup run trust targo --version 2>&1)"
[[ "${TRUSTC_COMMIT}" =~ ^[0-9a-f]{40}$ ]] && ok0 "trustc ${TRUSTC_COMMIT} host=${BUILD_HOST} sysroot=${SYSROOT}" \
    || fail0 "cannot read the Trust compiler commit"
[ -x "${TRUSTC_BIN}" ] && ok0 "trustc binary ${TRUSTC_BIN}" || fail0 "no trustc binary under ${SYSROOT}/bin"
case "${TARGO_VERSION}" in targo\ *) ok0 "frontend: ${TARGO_VERSION}" ;; *) fail0 "targo: ${TARGO_VERSION}" ;; esac
for tool in llvm-strip llvm-ar; do
    [ -x "${RUSTLIB_BIN}/${tool}" ] && ok0 "${tool} from the Trust toolchain" || fail0 "${RUSTLIB_BIN}/${tool} missing"
done
METADATA="$(rustup run trust targo metadata --locked --offline --format-version 1 2>"${TMPDIR:-/tmp}/ny-release-metadata.$$.err")"
if [ -n "${METADATA}" ]; then
    ok0 "targo metadata --locked --offline resolves this tree"
else
    fail0 "targo metadata --locked --offline failed: $(grep -m3 error "${TMPDIR:-/tmp}/ny-release-metadata.$$.err")"
fi
rm -f -- "${TMPDIR:-/tmp}/ny-release-metadata.$$.err"

EXTERNAL_CARGO_CONFIGS="$(external_cargo_configs "${SRC}")"
if [ -z "${EXTERNAL_CARGO_CONFIGS}" ]; then
    ok0 "no cargo configuration outside the checkout (CARGO_HOME, ancestors of ${SRC})"
else
    fail0 "cargo would merge configuration from outside the checkout: $(printf '%s\n' "${EXTERNAL_CARGO_CONFIGS}" | paste -sd' ' -)"
fi
GIT_DEPENDENCIES="$(printf '%s' "${METADATA}" | python3 -c 'import json, os, sys
for package in json.load(sys.stdin)["packages"]:
    source = package.get("source") or ""
    if source.startswith("git+"):
        print(os.path.dirname(package["manifest_path"]), source.rsplit("#", 1)[-1])' 2>/dev/null)"
if [ -z "${GIT_DEPENDENCIES}" ] && grep -q '^source = "git+' Cargo.lock; then
    fail0 "Cargo.lock has git-sourced packages but the metadata names none"
fi
declare -A GIT_CHECKOUT_SEEN=()
while read -r dependency_dir locked_commit; do
    [ -n "${dependency_dir}" ] || continue
    if state="$(git_dependency_checkout_state "${dependency_dir}" "${locked_commit}")"; then
        if [ -z "${GIT_CHECKOUT_SEEN[${state}]:-}" ]; then
            GIT_CHECKOUT_SEEN[${state}]=1
            ok0 "git dependency checkout ${state} is the locked ${locked_commit:0:9}, no tracked edits"
        fi
    else
        fail0 "git dependency checkout refused: ${state}"
    fi
done <<<"${GIT_DEPENDENCIES}"

# ONNX Runtime: ort-sys pins one archive hash per target in its dist.tsv and
# extracts into dfbin/<target>/<hash>/ only AFTER verifying that hash.
ORT_SYS_DIR="$(printf '%s' "${METADATA}" | python3 -c 'import json,sys,os
d=json.load(sys.stdin)
p=[x for x in d["packages"] if x["name"]=="ort-sys"]
print(os.path.dirname(p[0]["manifest_path"]) if len(p)==1 else "")' 2>/dev/null)"
ORT_DIST="${ORT_SYS_DIR}/build/download/dist.tsv"
[ -f "${ORT_DIST}" ] && ok0 "ort-sys dist table ${ORT_DIST}" || fail0 "ort-sys dist.tsv not found (${ORT_DIST})"
ORT_CACHES+=("${HOME}/.cache/ort.pyke.io" "${CROSS_ENV}/ort-cache")
ort_hash() { awk -F'\t' -v t="$1" '$1==t && $2=="none"{print $4}' "${ORT_DIST}" 2>/dev/null; }
ort_lib() { # target -> path of the verified static library, or empty
    local hash cache
    hash="$(ort_hash "$1")"
    [ -n "${hash}" ] || return 0
    for cache in "${ORT_CACHES[@]}"; do
        if [ -f "${cache}/dfbin/$1/${hash}/libonnxruntime.a" ]; then
            printf '%s\n' "${cache}/dfbin/$1/${hash}/libonnxruntime.a"
            return 0
        fi
    done
}

IFS=',' read -r -a TARGET_LIST <<<"${TARGETS}"
CROSS_NEEDED=0
for target in "${TARGET_LIST[@]}"; do
    case "${target}" in
        x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu) ;;
        *) fail0 "unsupported target ${target}"; continue ;;
    esac
    if [ "${target}" != "${BUILD_HOST}" ]; then
        [ "${target}" = x86_64-unknown-linux-gnu ] && [ "${BUILD_HOST}" = aarch64-unknown-linux-gnu ] \
            && CROSS_NEEDED=1 || fail0 "no declared lane builds ${target} on ${BUILD_HOST}"
    fi
    lib="$(ort_lib "${target}")"
    if [ -n "${lib}" ]; then
        ok0 "ONNX Runtime for ${target}: ${lib} (dist hash $(ort_hash "${target}"))"
    elif [ "${WARM_ORT}" = 1 ]; then
        echo "preflight: ONNX Runtime for ${target} will be downloaded by the --warm-ort step"
    else
        fail0 "no verified ONNX Runtime archive for ${target} (hash $(ort_hash "${target}")) under: ${ORT_CACHES[*]}; pass --warm-ort or --ort-cache"
    fi
    if [ -e "${OUT_DIR}/${target}" ] && [ "${REPLACE}" != 1 ]; then
        fail0 "${OUT_DIR}/${target} exists; pass --replace to move it aside"
    fi
done

if [ "${CROSS_NEEDED}" = 1 ]; then
    XC="${CROSS_ENV}/bin/x86_64-linux-gnu-clang"
    XCXX="${CROSS_ENV}/bin/x86_64-linux-gnu-clang++"
    XSYSROOT="${CROSS_ENV}/xsysroot"
    STD_TAR="${CROSS_ENV}/trust-src-${TRUSTC_COMMIT:0:9}-library+trust-memchr.tar"
    STD_SRC="${CROSS_ENV}/trust-src-${TRUSTC_COMMIT:0:9}"
    TRUST_SRC_REPO="$(readlink -f "${SYSROOT}/lib/rustlib/rustc-src/rust" 2>/dev/null)"
    for wrapper in "${XC}" "${XCXX}"; do
        if [ -x "${wrapper}" ] && grep -q -- "--sysroot=${XSYSROOT} " "${wrapper}"; then
            ok0 "cross wrapper ${wrapper} targets ${XSYSROOT}"
        else
            fail0 "cross wrapper ${wrapper} missing or not bound to ${XSYSROOT} (run scripts/release/prepare_x86_cross_env.sh)"
        fi
    done
    for p in usr/lib/x86_64-linux-gnu/crt1.o usr/lib/x86_64-linux-gnu/Scrt1.o usr/lib/x86_64-linux-gnu/crti.o \
             usr/lib/x86_64-linux-gnu/libc.so usr/lib/x86_64-linux-gnu/libc_nonshared.a \
             usr/include/stdio.h usr/include/c++/13/iostream usr/lib/gcc/x86_64-linux-gnu/13/libgcc.a \
             usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.so; do
        [ -e "${XSYSROOT}/${p}" ] || fail0 "cross sysroot lacks ${p}"
    done
    if (cd "${CROSS_ENV}" && sha256sum -c --quiet debs/SHA256SUMS) >/dev/null 2>&1; then
        CROSS_SYSROOT_SHA256="$(sha_file "${CROSS_ENV}/debs/SHA256SUMS")"
        ok0 "cross sysroot debs match debs/SHA256SUMS (${CROSS_SYSROOT_SHA256})"
    else
        fail0 "cross sysroot debs do not match ${CROSS_ENV}/debs/SHA256SUMS"
    fi
    STD_WANT="$(awk '{print $1}' "${STD_TAR}.sha256" 2>/dev/null)"
    STD_GOT="$(git -C "${TRUST_SRC_REPO}" archive --format=tar "${TRUSTC_COMMIT}" library crates/trust-memchr 2>/dev/null | sha256sum | awk '{print $1}')"
    if [ -n "${STD_WANT}" ] && [ "${STD_WANT}" = "${STD_GOT}" ] && [ "$(sha_file "${STD_TAR}")" = "${STD_WANT}" ]; then
        ok0 "std source = git archive of the compiler's own commit ${TRUSTC_COMMIT:0:9} (${STD_GOT})"
    else
        fail0 "std source archive is not reproducible from ${TRUST_SRC_REPO} at ${TRUSTC_COMMIT} (want=${STD_WANT:-none} got=${STD_GOT})"
    fi
    n="$(tar -df "${STD_TAR}" -C "${STD_SRC}" 2>&1 | grep -v -c -E ': (Uid|Gid) differs$')"
    [ "${n}" = 0 ] && ok0 "extracted std tree matches its archive" || fail0 "extracted std tree differs from its archive (${n})"
    LLD="$(command -v ld.lld || true)"
    if [ -n "${LLD}" ]; then
        LINKER_SHA256="$(sha_file "$(readlink -f "${LLD}")")"
        ok0 "ld.lld ${LLD} -> $(readlink -f "${LLD}") ($("${LLD}" --version | head -1))"
    else
        fail0 "ld.lld is not on PATH (the cross wrappers link with -fuse-ld=lld)"
    fi
    [ -x "${QEMU}" ] && [ -d "${QEMU_SYSROOT}" ] && ok0 "qemu ${QEMU} ($("${QEMU}" --version | head -1)) sysroot ${QEMU_SYSROOT}" \
        || fail0 "qemu-x86_64 or its sysroot missing (${QEMU}, ${QEMU_SYSROOT}); needed to make the x86_64 binary identify itself"
    # Link probe: the C++ wrapper + lld + sysroot produce an x86_64 program that
    # runs under the same qemu the self-identification uses.
    probe_dir="$(mktemp -d "${TMPDIR:-/tmp}/ny-release-probe.XXXXXX")"
    printf '#include <cstdio>\n#include <string>\nint main() { std::string s("x86 link ok"); std::puts(s.c_str()); return 0; }\n' \
        >"${probe_dir}/probe.cc"
    if "${XCXX}" -O2 "${probe_dir}/probe.cc" -o "${probe_dir}/probe" 2>"${probe_dir}/link.err" \
        && [ "$(timeout 60 "${QEMU}" -cpu "${QEMU_CPU}" -L "${QEMU_SYSROOT}" "${probe_dir}/probe" 2>&1)" = "x86 link ok" ]; then
        ok0 "link probe: C++ via ${XCXX} runs under qemu ${QEMU_CPU}"
    else
        fail0 "link probe failed: $(head -3 "${probe_dir}/link.err" 2>/dev/null)"
    fi
    rm -f -- "${probe_dir}/probe.cc" "${probe_dir}/probe" "${probe_dir}/link.err"
    rmdir -- "${probe_dir}" 2>/dev/null || true
fi

# The cross lane's compiler/linker/std environment (empty for a native target).
cross_env_for() { # target -> prints NAME=VALUE lines
    [ "$1" = "${BUILD_HOST}" ] && return 0
    printf '%s\n' "CC_x86_64_unknown_linux_gnu=${XC}" "CXX_x86_64_unknown_linux_gnu=${XCXX}" \
        "AR_x86_64_unknown_linux_gnu=${RUSTLIB_BIN}/llvm-ar" \
        "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=${XC}" \
        "__CARGO_TESTS_ONLY_SRC_ROOT=${STD_SRC}/library"
}

# The resolver on the SAME command line, environment and flags each build will
# use (--unit-graph: resolution only, nothing compiles). A tree this toolchain
# cannot resolve fails here in seconds, not after the lock wait.
if [ "${FAIL}" = 0 ]; then
    for target in "${TARGET_LIST[@]}"; do
        extra=()
        [ "${target}" != "${BUILD_HOST}" ] && extra=(--target "${target}" -Zbuild-std=std,panic_abort)
        mapfile -t xenv < <(cross_env_for "${target}")
        units="$(env CARGO_TARGET_DIR="${TARGET_BASE}/${target}" CARGO_NET_OFFLINE=true "${xenv[@]}" \
            rustup run trust targo --unverified build --locked --offline --release -p ny-cli --bin ny \
            --features mip,cuda "${extra[@]}" -Zunstable-options --unit-graph 2>/dev/null \
            | python3 -c 'import json, sys
units = json.load(sys.stdin)["units"]
names = {u["target"]["name"] for u in units}
need = {"ny", "ort_sys", "ay_milp"} | ({"std"} if sys.argv[1] == "cross" else set())
print(len(units), int(need <= names))' "$([ "${target}" = "${BUILD_HOST}" ] && echo native || echo cross)" 2>/dev/null)"
        if [ "${units##* }" = 1 ]; then
            ok0 "${target}: unit graph resolves offline+locked (${units%% *} units)"
        else
            fail0 "${target}: the release command line does not resolve (unit graph: '${units}')"
        fi
    done
fi

avail="$(df -BG --output=avail "${SRC}" | tail -1 | tr -dc 0-9)"
[ "${avail:-0}" -ge 60 ] && ok0 "disk free ${avail}G" || fail0 "disk free ${avail}G < 60G"
MAKE_SUBMISSION_SHA256="$(sha_file "${SELF}")"
[ "$(git show "${COMMIT}:scripts/release/make_submission.sh" 2>/dev/null | sha256sum | awk '{print $1}')" = "${MAKE_SUBMISSION_SHA256}" ] \
    && ok0 "this script is the committed scripts/release/make_submission.sh" \
    || fail0 "the running make_submission.sh differs from the one committed at ${COMMIT}"

if [ "${FAIL}" != 0 ]; then
    echo "MAKE_SUBMISSION_DONE rc=3 WORST=3 preflight=FAIL elapsed=$(( $(date +%s) - T0 ))"
    exit 3
fi
echo "MAKE_SUBMISSION_PREFLIGHT_OK commit=${COMMIT} targets=${TARGETS} trustc=${TRUSTC_COMMIT:0:9}"
if [ "${PREFLIGHT_ONLY}" = 1 ]; then
    echo "MAKE_SUBMISSION_DONE rc=0 WORST=0 preflight=ok build=skipped elapsed=$(( $(date +%s) - T0 ))"
    exit 0
fi

# ------------------------------------------------------------- builds ------
[ -n "${LOG_DIR}" ] || LOG_DIR="${SRC}/dist/logs/${COMMIT:0:9}-${RUNID}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${TARGET_BASE}" || exit 2
export AY_SOURCE_GIT_COMMIT="${AY_COMMIT}" AY_SOURCE_GIT_DIRTY=false
export CARGO_NET_OFFLINE=true
TRUSTC_SHA256="$(sha_file "${TRUSTC_BIN}")"
TRUSTC_VERSION_SHA256="$(sha_text "${TRUSTC_VV}")"
TARGO_VERSION_SHA256="$(sha_text "${TARGO_VERSION}")"
SELFTEST="${SRC}/vnncomp_scripts/selftest"

warm_ort() { # target -- the builder's own credential-free warm-up, into the cross env cache
    local target="$1" cache="${CROSS_ENV}/ort-cache" log="${LOG_DIR}/${target}.ortwarm.log"
    local -a extra=() xenv=()
    [ "${target}" != "${BUILD_HOST}" ] && extra=(--target "${target}" -Zbuild-std=std,panic_abort)
    mapfile -t xenv < <(cross_env_for "${target}")
    mkdir -p "${cache}"
    echo "=== [$(date -u +%H:%M:%S)] ${target}: ONNX Runtime warm-up (network) -> ${cache}"
    env -u CARGO_NET_OFFLINE ORT_CACHE_DIR="${cache}" CARGO_TARGET_DIR="${TARGET_BASE}/${target}-ortwarm" \
        "${xenv[@]}" rustup run trust targo --unverified build --locked --release -p ny-onnx "${extra[@]}" \
        >"${log}" 2>&1
}

run_binary() { # target binary args... -- run natively or under qemu-user
    local target="$1" binary="$2"
    shift 2
    if [ "${target}" = "${BUILD_HOST}" ]; then
        timeout 600 "${binary}" "$@"
    else
        timeout 1200 "${QEMU}" -cpu "${QEMU_CPU}" -L "${QEMU_SYSROOT}" "${binary}" "$@"
    fi
}

field() { # line key -> value of key= in a space-separated identity line
    printf '%s\n' "$1" | tr ' ' '\n' | awk -F= -v k="$2" '$1==k{sub(/^[^=]*=/,""); print; exit}'
}

build_target() { # target -> 0 or the failing step's rc
    local target="$1" lane td bin log stage lib ort_sha
    local -a extra=()
    local -a xenv=()
    log="${LOG_DIR}/${target}.build.log"
    td="${TARGET_BASE}/${target}"
    if [ "${target}" = "${BUILD_HOST}" ]; then
        lane=native
        bin="${td}/release/ny"
    else
        lane=cross-build-std
        bin="${td}/${target}/release/ny"
        extra=(--target "${target}" -Zbuild-std=std,panic_abort)
    fi
    mapfile -t xenv < <(cross_env_for "${target}")
    lib="$(ort_lib "${target}")"
    if [ -z "${lib}" ]; then
        warm_ort "${target}" || { echo "  ${target}: ONNX Runtime warm-up failed (${LOG_DIR}/${target}.ortwarm.log)"; return 21; }
        lib="$(ort_lib "${target}")"
        [ -n "${lib}" ] || { echo "  ${target}: warm-up left no verified archive"; return 21; }
    fi
    ort_sha="$(sha_file "${lib}")"

    echo "=== [$(date -u +%H:%M:%S)] ${target}: build lane=${lane} target-dir=${td} log=${log}"
    env CARGO_TARGET_DIR="${td}" ORT_LIB_PATH="$(dirname "${lib}")" "${xenv[@]}" \
        rustup run trust targo --unverified build --locked --offline --release \
        -p ny-cli --bin ny --features mip,cuda "${extra[@]}" -v >"${log}" 2>&1
    local rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "  ${target}: build rc=${rc}; error lines:"
        grep -a -n -E '^(error|fatal)|ld\.lld: error' "${log}" | head -40 | sed 's/^/    /'
        tail -20 "${log}" | sed 's/^/    | /'
        return 30
    fi
    # The ny crate's final rustc line: unverified lane, no machine coupling.
    local ny_line last_verify
    ny_line="$(grep -a -E 'Running `[^`]*rustc --crate-name ny ' "${log}" | tail -1)"
    last_verify="$(printf '%s\n' "${ny_line}" | grep -oE -- '-Ztrust-verify=[a-z]+' | tail -1)"
    if [ "${last_verify}" != "-Ztrust-verify=off" ] || printf '%s\n' "${ny_line}" | grep -q -- 'target-cpu'; then
        echo "  ${target}: the ny rustc line is not the unverified, machine-neutral build (${last_verify:-no -Ztrust-verify})"
        return 31
    fi
    if [ "$(git rev-parse HEAD)" != "${COMMIT}" ] || [ -n "$(git status --porcelain=v1 --untracked-files=all)" ] \
        || [ "$(sha_file Cargo.lock)" != "${LOCK_SHA256}" ]; then
        echo "  ${target}: the checkout changed during the build"
        return 32
    fi
    [ -s "${bin}" ] || { echo "  ${target}: no binary at ${bin}"; return 33; }
    echo "  ${target}: build ok ($(grep -a -c '^ *Compiling ' "${log}") units compiled)"

    stage="${OUT_DIR}/.staging-${target}-${RUNID}"
    mkdir "${stage}" || return 40
    cp -- "${bin}" "${stage}/ny" && "${RUSTLIB_BIN}/llvm-strip" "${stage}/ny" || { echo "  ${target}: strip failed"; return 40; }
    chmod 0755 "${stage}/ny"

    local machine needed gl gx want_machine bin_sha
    want_machine="Advanced Micro Devices X86-64"
    [ "${target}" = aarch64-unknown-linux-gnu ] && want_machine="AArch64"
    machine="$(LC_ALL=C readelf -h "${stage}/ny" | awk -F: '/Machine:/{gsub(/^ +/,"",$2); print $2}')"
    [ "${machine}" = "${want_machine}" ] || { echo "  ${target}: ELF machine ${machine} != ${want_machine}"; return 41; }
    needed="$(LC_ALL=C readelf -d "${stage}/ny" | awk '/\(NEEDED\)/{gsub(/[][]/,"",$NF); print $NF}' | paste -sd, -)"
    case ",${needed}," in *onnxruntime*|*libcuda*|*libcudart*|*libcublas*|*libnvrtc*|*libvulkan*)
        echo "  ${target}: unexpected load-time dependency in NEEDED=${needed}"; return 42 ;;
    esac
    gl="$(LC_ALL=C readelf -V "${stage}/ny" | grep -oE 'GLIBC_[0-9]+\.[0-9]+(\.[0-9]+)?' | sed 's/GLIBC_//' | sort -Vu | tail -1)"
    gx="$(LC_ALL=C readelf -V "${stage}/ny" | grep -oE 'GLIBCXX_[0-9]+\.[0-9]+(\.[0-9]+)?' | sed 's/GLIBCXX_//' | sort -Vu | tail -1)"
    [ -n "${gx}" ] || gx=none
    # Ubuntu 24.04 (the evaluation image) ceiling: glibc 2.39, libstdc++ 3.4.33.
    [ "$(printf '%s\n' "${gl}" 2.39 | sort -V | tail -1)" = 2.39 ] || { echo "  ${target}: needs GLIBC_${gl} > 2.39"; return 43; }
    [ "${gx}" = none ] || [ "$(printf '%s\n' "${gx}" 3.4.33 | sort -V | tail -1)" = 3.4.33 ] \
        || { echo "  ${target}: needs GLIBCXX_${gx} > 3.4.33"; return 44; }
    bin_sha="$(sha_file "${stage}/ny")"

    # Self-identification: the binary states who it is; nothing is inferred.
    local version build_id build_stamp provenance
    version="$(run_binary "${target}" "${stage}/ny" --version 2>/dev/null | head -1)"
    build_id="$(run_binary "${target}" "${stage}/ny" --build-id 2>/dev/null | grep -m1 '^NY_BUILD_ID_V1 ')"
    build_stamp="$(run_binary "${target}" "${stage}/ny" --build-stamp 2>/dev/null | grep -m1 '^NY_BUILD_STAMP_V1 ')"
    provenance="$(run_binary "${target}" "${stage}/ny" __vnncomp-build-provenance 2>/dev/null | head -1)"
    echo "  ${target}: ${version}"
    echo "  ${target}: ${build_id}"
    local k want
    for k in commit:"${COMMIT}" dirty:false source:clean features:mip,cuda profile:release target:"${target}" exe:"${bin_sha}"; do
        want="${k#*:}"
        [ "$(field "${build_id}" "${k%%:*}")" = "${want}" ] \
            || { echo "  ${target}: build_id ${k%%:*}=$(field "${build_id}" "${k%%:*}") != ${want}"; return 50; }
    done
    [ "$(field "${build_stamp}" stamp)" = "$(field "${build_id}" stamp)" ] \
        || { echo "  ${target}: build_stamp disagrees with build_id"; return 51; }
    [ "${provenance}" = "ny.vnncomp.unsealed.v1|status=unsealed|" ] \
        || { echo "  ${target}: unexpected build provenance record '${provenance}'"; return 52; }
    if LC_ALL=C grep -a -c -F 'ny.vnncomp.build.v1|status=sealed|' "${stage}/ny" >/dev/null; then
        echo "  ${target}: a sealed provenance record is embedded; this lane never seals"; return 53
    fi
    local ay_embedded
    ay_embedded="$(LC_ALL=C grep -a -oE 'build\.commit=[0-9a-f]{40}(-dirty)?' "${stage}/ny" | cut -d= -f2 | sort -u | paste -sd' ' -)"
    [ "${ay_embedded}" = "${AY_COMMIT}" ] || { echo "  ${target}: embedded AY ${ay_embedded} != ${AY_COMMIT}"; return 54; }

    # The install self-test, at build time, on the exact bytes that ship.
    local prop res out
    for prop in unsat sat; do
        out="${stage}/selftest_${prop}.res"
        run_binary "${target}" "${stage}/ny" vnncomp v1 test "${SELFTEST}/abs_relu.onnx" \
            "${SELFTEST}/abs_relu_${prop}.vnnlib" "${out}" 60 >"${LOG_DIR}/${target}.selftest_${prop}.log" 2>&1
        res="$(head -1 "${out}" 2>/dev/null | tr -d '[:space:]')"
        rm -f -- "${out}" "${out}.flight.json"
        if [ "${res}" != "${prop}" ]; then
            echo "  ${target}: self-test ${prop} answered '${res:-<none>}' (log ${LOG_DIR}/${target}.selftest_${prop}.log)"
            [ "${res}" = sat ] || [ "${res}" = unsat ] && { echo "  EMERGENCY: a decided self-test answer is WRONG"; return 99; }
            return 55
        fi
        echo "  ${target}: self-test ${prop} -> ${res}"
    done

    # CUDA kernel images: the committed fatbins must be embedded byte-for-byte.
    local fatbins
    fatbins="$(python3 - "${SRC}/crates/ny-cuda/kernels" "${stage}/ny" <<'PY'
import hashlib, pathlib, sys
kernels, binary = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_bytes()
fields = dict(l.split("=", 1) for l in (kernels / "search_kernels.manifest").read_text().splitlines()
              if l and not l.startswith("#") and "=" in l)
out = []
for image, digest in (("image", "image_sha256"), ("wave_image", "wave_image_sha256")):
    data = (kernels / fields[image]).read_bytes()
    if hashlib.sha256(data).hexdigest() != fields[digest] or binary.find(data) < 0:
        sys.exit(f"{fields[image]}: committed bytes do not match the manifest or are not embedded")
    out.append(f"{fields[image]}:{fields[digest]}")
print(",".join(out))
PY
)" || { echo "  ${target}: CUDA fatbin check failed"; return 56; }
    echo "  ${target}: CUDA fatbins embedded: ${fatbins}"

    xz -9 -T0 -c -- "${stage}/ny" >"${stage}/ny.xz" && xz -t -- "${stage}/ny.xz" || { echo "  ${target}: xz failed"; return 57; }
    printf '%s  ny\n' "${bin_sha}" >"${stage}/ny.sha256"
    local std_source="prebuilt" sysroot_sha="none" linker_sha="none" identified="native"
    if [ "${lane}" = cross-build-std ]; then
        std_source="build-std:${STD_GOT}"
        sysroot_sha="${CROSS_SYSROOT_SHA256}"
        linker_sha="${LINKER_SHA256}"
        identified="qemu-user:$(basename "${QEMU}"):${QEMU_CPU}"
    fi
    {
        printf 'schema=ny-release-manifest-v1\n'
        printf 'target=%s\n' "${target}"
        printf 'build_lane=%s\n' "${lane}"
        printf 'build_host=%s\n' "${BUILD_HOST}"
        printf 'features=mip,cuda\n'
        printf 'ny_commit=%s\n' "${COMMIT}"
        printf 'ny_tree=clean\n'
        printf 'ny_commit_tree=%s\n' "${NY_COMMIT_TREE}"
        printf 'ny_commit_object=%s\n' "${NY_COMMIT_OBJECT}"
        printf 'release_paths=%s\n' "${RELEASE_PATHS_CSV}"
        printf 'cargo_lock_sha256=%s\n' "${LOCK_SHA256}"
        printf 'ay_lock_commit=%s\n' "${AY_COMMIT}"
        printf 'trustc_commit=%s\n' "${TRUSTC_COMMIT}"
        printf 'trustc_version_sha256=%s\n' "${TRUSTC_VERSION_SHA256}"
        printf 'trustc_sha256=%s\n' "${TRUSTC_SHA256}"
        printf 'targo_version_sha256=%s\n' "${TARGO_VERSION_SHA256}"
        printf 'std_source=%s\n' "${std_source}"
        printf 'cross_sysroot_sha256=%s\n' "${sysroot_sha}"
        printf 'linker_sha256=%s\n' "${linker_sha}"
        printf 'onnxruntime_static_sha256=%s\n' "${ort_sha}"
        printf 'binary_sha256=%s\n' "${bin_sha}"
        printf 'binary_size=%s\n' "$(stat -c %s "${stage}/ny")"
        printf 'package=ny.xz\n'
        printf 'package_sha256=%s\n' "$(sha_file "${stage}/ny.xz")"
        printf 'build_id=%s\n' "${build_id}"
        printf 'build_stamp=%s\n' "${build_stamp}"
        printf 'vnncomp_build_provenance=%s\n' "${provenance}"
        printf 'needed=%s\n' "${needed}"
        printf 'max_glibc=%s\n' "${gl}"
        printf 'max_glibcxx=%s\n' "${gx}"
        printf 'cuda_runtime=dlopen\n'
        printf 'cuda_fatbin_manifest_sha256=%s\n' "$(sha_file "${SRC}/crates/ny-cuda/kernels/search_kernels.manifest")"
        printf 'cuda_fatbins=%s\n' "${fatbins}"
        printf 'self_identified_by=%s\n' "${identified}"
        printf 'make_submission_sha256=%s\n' "${MAKE_SUBMISSION_SHA256}"
    } >"${stage}/MANIFEST"

    if [ -e "${OUT_DIR}/${target}" ]; then
        mkdir -p "${OUT_DIR}/.superseded"
        mv -- "${OUT_DIR}/${target}" "${OUT_DIR}/.superseded/${target}.${RUNID}" || return 60
        echo "  ${target}: previous artifact moved to ${OUT_DIR}/.superseded/${target}.${RUNID}"
    fi
    mv -- "${stage}" "${OUT_DIR}/${target}" || return 60

    # The installer's own verifier over exactly what was written (only when the
    # artifact sits where install_tool.sh will look for it).
    if [ "$(realpath "${OUT_DIR}")" = "$(realpath "${SRC}/dist/bin")" ]; then
        local check="${OUT_DIR}/.verify-${target}-${RUNID}"
        if ! python3 -I "${SRC}/vnncomp_scripts/verify_release.py" --repo-root "${SRC}" \
                --target "${target}" --output "${check}"; then
            rm -f -- "${check}"
            mkdir -p "${OUT_DIR}/.rejected"
            mv -- "${OUT_DIR}/${target}" "${OUT_DIR}/.rejected/${target}.${RUNID}"
            echo "  ${target}: verify_release.py REFUSED the artifact; moved to ${OUT_DIR}/.rejected/"
            return 61
        fi
        rm -f -- "${check}"
    fi
    echo "MAKE_SUBMISSION_TARGET target=${target} rc=0 lane=${lane} sha256=${bin_sha} package_sha256=$(sha_file "${OUT_DIR}/${target}/ny.xz") manifest_sha256=$(sha_file "${OUT_DIR}/${target}/MANIFEST") max_glibc=${gl} selftest=unsat:ok,sat:ok"
    return 0
}

WORST=0
SUMMARY=""
for target in "${TARGET_LIST[@]}"; do
    build_target "${target}"
    rc=$?
    [ "${rc}" -eq 0 ] || echo "MAKE_SUBMISSION_TARGET target=${target} rc=${rc}"
    SUMMARY="${SUMMARY} ${target%%-*}=${rc}"
    [ "${rc}" -gt "${WORST}" ] && WORST="${rc}"
done
echo "MAKE_SUBMISSION_DONE rc=${WORST} WORST=${WORST}${SUMMARY} commit=${COMMIT} logs=${LOG_DIR} elapsed=$(( $(date +%s) - T0 ))"
exit "${WORST}"
