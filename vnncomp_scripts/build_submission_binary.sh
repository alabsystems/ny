#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

TOOL_DIR=$(dirname "$(dirname "$(realpath "$0")")")
STABLE_TARGET_DIR="${TOOL_DIR}/target/release"
STABLE_SUBMISSION_BIN="${STABLE_TARGET_DIR}/ny"
STABLE_SUBMISSION_RECEIPT="${STABLE_SUBMISSION_BIN}.receipt"
RECEIPT_HELPER="${TOOL_DIR}/vnncomp_scripts/submission_binary_receipt.sh"

cd "${TOOL_DIR}"

if [ -L "${RECEIPT_HELPER}" ] || [ ! -f "${RECEIPT_HELPER}" ]; then
    echo "ERROR: submission receipt helper is missing or is a symlink: ${RECEIPT_HELPER}" >&2
    exit 1
fi
# Capture before Cargo starts and compare again after it finishes.  The helper
# binds HEAD plus the complete tracked diff (or the packager's archive marker),
# Cargo.lock, and AY, so a source/lock update during a long build cannot receive
# a receipt for the wrong bytes.
SOURCE_IDENTITY_BEFORE="$(bash "${RECEIPT_HELPER}" identity "${TOOL_DIR}")"

# Canonicalize a path whose final components may not exist yet. GNU
# `realpath -m` provides that operation on Linux but is unavailable in the
# BSD realpath shipped by macOS, where this builder and its integration tests
# also run. Python is already a hard prerequisite for Cargo artifact
# authentication below; isolated `os.path.realpath` has the required portable
# existing-prefix/symlink semantics without importing user site hooks.
canonicalize_missing_path() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: Python 3 is required to canonicalize Cargo staging paths." >&2
        return 1
    fi
    python3 -I - "$1" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
}

# Read one Cargo config target without embedding a Python here-document inside
# a quoted command substitution. macOS ships Bash 3.2, whose legacy parser can
# mis-tokenize complex here-document bodies in that position even though newer
# Bash accepts them.
read_cargo_build_target() {
    python3 -I - "$1" <<'PY'
import sys

try:
    import tomllib
except ImportError:
    print("__NY_UNRESOLVED_CARGO_TARGET__")
    raise SystemExit

unresolved = "__NY_UNRESOLVED_CARGO_TARGET__"
try:
    with open(sys.argv[1], "rb") as config_file:
        config = tomllib.load(config_file)
except (OSError, tomllib.TOMLDecodeError):
    print(unresolved)
    raise SystemExit

# Cargo's config-include facility has its own recursive merge semantics.  The
# build must not guess at the effective target without evaluating that graph.
if "include" in config:
    print(unresolved)
    raise SystemExit

build = config.get("build")
if build is None:
    raise SystemExit
if not isinstance(build, dict):
    print(unresolved)
    raise SystemExit

target = build.get("target")
if target is None:
    raise SystemExit
if not isinstance(target, str) or not target:
    print(unresolved)
    raise SystemExit
print(target)
PY
}

# An explicit target directory remains the staging *parent*, but Cargo never
# writes directly into it. In particular, the default parent is also the
# parent of the published target/release/ny alias: a private direct child keeps
# a failed or ambiguous Cargo invocation physically unable to alter that alias.
if [ -n "${CARGO_TARGET_DIR:-}" ]; then
    BUILD_TARGET_BASE="$(canonicalize_missing_path "${CARGO_TARGET_DIR}")"
elif [ -n "${CARGO_BUILD_TARGET_DIR:-}" ]; then
    BUILD_TARGET_BASE="$(canonicalize_missing_path "${CARGO_BUILD_TARGET_DIR}")"
elif [ -n "${AI_WORKER_ID:-}" ]; then
    if [[ ! "${AI_WORKER_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
        || [ "${AI_WORKER_ID}" = "." ] \
        || [ "${AI_WORKER_ID}" = ".." ]; then
        echo "ERROR: AI_WORKER_ID contains unsafe path characters." >&2
        exit 2
    fi
    BUILD_TARGET_BASE="${TOOL_DIR}/target/worker_${AI_WORKER_ID}"
else
    BUILD_TARGET_BASE="${TOOL_DIR}/target"
fi
BUILD_TARGET_BASE="$(canonicalize_missing_path "${BUILD_TARGET_BASE}")"
BUILD_TARGET_DIR=""
BUILD_STAGING_DIRS=()

cleanup_submission_staging() {
    local original_status="$1"
    local staging_dir
    trap - EXIT HUP INT TERM
    # Bash 3.2 treats an empty declared array as unbound under `set -u` when
    # expanded directly. The `+` guard keeps early-failure cleanup quiet while
    # preserving element boundaries once staging directories exist.
    for staging_dir in ${BUILD_STAGING_DIRS[@]+"${BUILD_STAGING_DIRS[@]}"}; do
        if [[ -d "${staging_dir}" \
            && ! -L "${staging_dir}" \
            && "$(dirname -- "${staging_dir}")" = "${BUILD_TARGET_BASE}" \
            && "$(basename -- "${staging_dir}")" == .ny-submission-build.* ]]; then
            rm -rf -- "${staging_dir}" || true
        fi
    done
    exit "${original_status}"
}
trap 'cleanup_submission_staging $?' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Building ny release binary..."

# --- Build prerequisite diagnostics (fail loudly, never silently) ---
# The `mip` feature is pure Rust (ay-milp); no CMake/libclang needed. A C
# compiler and pkg-config are still used by native dep crates (psm/stacker,
# zstd-sys, onig_sys, liblzma-sys). ort-sys fetches its static ONNX Runtime
# archive over rustls, so a source build needs network access but no system
# OpenSSL headers.
missing=""
for tool in cargo cc pkg-config python3; do
    command -v "${tool}" >/dev/null 2>&1 || missing="${missing} ${tool}"
done
if [ -n "${missing}" ]; then
    echo "WARNING: build prerequisites appear missing:${missing}" >&2
    echo "  (cargo=Rust toolchain; cc=C compiler; pkg-config=native dependency discovery; python3=Cargo artifact authentication)" >&2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python 3 is required to authenticate Cargo's emitted ny artifact." >&2
    exit 1
fi

# Do not bake one developer host's memory ceiling into the repository-wide
# Cargo configuration: that silently throttles organizer builds too.  The
# sealed release builder is the memory-heavy path, so derive a process-local
# default here unless the operator already selected CARGO_BUILD_JOBS.  Release
# rustc/LTO workers have measured near 8 GiB each; reserve 8 GiB for the OS and
# cap by the online CPU count.  Explicit CARGO_BUILD_JOBS remains authoritative.
if [ -z "${CARGO_BUILD_JOBS:-}" ]; then
    CARGO_BUILD_JOBS="$(python3 -I - <<'PY'
import os
import resource
from pathlib import Path

gib = 1024 ** 3
cpus = max(1, os.cpu_count() or 1)
memory_limits = []
try:
    memory_limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
except (AttributeError, OSError, TypeError, ValueError):
    pass

try:
    soft_as, hard_as = resource.getrlimit(resource.RLIMIT_AS)
    memory_limits.extend(
        limit
        for limit in (soft_as, hard_as)
        if limit > 0 and limit != resource.RLIM_INFINITY
    )
except (AttributeError, OSError, ValueError):
    pass

# The common cgroup-v2 namespace layout exposes the effective process limit at
# /sys/fs/cgroup/<membership>/memory.max. Scan its ancestors too: a parent may
# be tighter than the immediate service. Failure to resolve this optional host
# interface falls back to physical/RLIMIT memory, never to zero workers.
try:
    membership = next(
        line.split("::", 1)[1].strip()
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    current = (Path("/sys/fs/cgroup") / membership.lstrip("/")).resolve()
    cgroup_root = Path("/sys/fs/cgroup").resolve()
    while current == cgroup_root or cgroup_root in current.parents:
        raw = (current / "memory.max").read_text().strip()
        if raw != "max":
            memory_limits.append(int(raw))
        if current == cgroup_root:
            break
        current = current.parent
except (OSError, StopIteration, ValueError):
    pass

positive_limits = [limit for limit in memory_limits if limit > 0]
if positive_limits:
    memory_jobs = max(1, (min(positive_limits) - 8 * gib) // (8 * gib))
    cpus = min(cpus, memory_jobs)
print(max(1, cpus))
PY
)"
    export CARGO_BUILD_JOBS
    echo "Cargo release build jobs: ${CARGO_BUILD_JOBS} (CPU/memory-derived default)"
else
    echo "Cargo release build jobs: ${CARGO_BUILD_JOBS} (operator override)"
fi

# Toolchain-era check: ort-sys downloads a prebuilt static ONNX Runtime archive
# built with a newer GCC than Ubuntu 22.04 ships. On gcc-11/binutils 2.38 era
# hosts (glibc < 2.39) the FINAL link fails with undefined references to
# onnxruntime internals (e.g. MLTypeCallDispatcher<...Float8E4M3FN...>) —
# after the entire workspace has compiled. Known good: Ubuntu 24.04 and 26.04
# (the VNN-COMP 2026 eval AMI is Ubuntu Server 24.04). Warn-only: the build
# below remains the authoritative fail-closed gate.
if [ "$(uname -s)" = "Linux" ]; then
    glibc_ver=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}') || glibc_ver=""
    if [[ "${glibc_ver}" =~ ^([0-9]+)\.([0-9]+)$ ]] \
        && ((10#${BASH_REMATCH[1]} < 2 || (10#${BASH_REMATCH[1]} == 2 && 10#${BASH_REMATCH[2]} < 39))); then
        ld_ver=$(ld --version 2>/dev/null | head -n 1) || ld_ver=""
        echo "WARNING: glibc ${glibc_ver} host (< 2.39 — pre-Ubuntu-24.04 toolchain era; ld: ${ld_ver:-unknown})." >&2
        echo "  The release link is expected to FAIL with undefined onnxruntime symbols" >&2
        echo "  from libort_sys-*.rlib on this toolchain. Build on Ubuntu >= 24.04, or" >&2
        echo "  set ORT_LIB_LOCATION to a locally built ONNX Runtime." >&2
    fi
fi

# LLVM 22 validates AArch64 inline assembly feature requirements. gemm-f16
# contains FP16 vector instructions behind runtime dispatch but does not mark
# every helper with #[target_feature], so a generic AArch64 build is rejected at
# assembly time even on a capable host. Enable fp16 only when every reported CPU
# feature line exposes Advanced-SIMD half precision; never emit it blindly into
# a portable/cross build.
rust_host="$(rustc -vV 2>/dev/null | awk '$1 == "host:" { print $2 }' || true)"
effective_build_target="${CARGO_BUILD_TARGET:-}"
if [ -z "${effective_build_target}" ]; then
    # Cargo may source build.target from user or repository configuration. A
    # configured value is authoritative. Mirror Cargo's discovery order:
    # CARGO_HOME is lowest priority, then ancestor configurations from `/` to
    # the current workspace. Within one directory, legacy `.cargo/config`
    # wins when both names exist. Unfamiliar forms (arrays or includes we do
    # not evaluate) deliberately cannot equal the host, so no host-only flag
    # can leak into an unresolved cross build.
    cargo_config_files=()
    cargo_home="${CARGO_HOME:-${HOME:-}/.cargo}"
    if [ -f "${cargo_home}/config" ]; then
        cargo_config_files+=("${cargo_home}/config")
    elif [ -f "${cargo_home}/config.toml" ]; then
        cargo_config_files+=("${cargo_home}/config.toml")
    fi

    cargo_search_dirs=()
    cargo_search_dir="${TOOL_DIR}"
    while true; do
        cargo_search_dirs+=("${cargo_search_dir}")
        [ "${cargo_search_dir}" = "/" ] && break
        cargo_search_dir="$(dirname "${cargo_search_dir}")"
    done
    for ((cargo_dir_index=${#cargo_search_dirs[@]} - 1; cargo_dir_index >= 0; cargo_dir_index--)); do
        cargo_search_dir="${cargo_search_dirs[cargo_dir_index]}"
        if [ -f "${cargo_search_dir}/.cargo/config" ]; then
            cargo_config_files+=("${cargo_search_dir}/.cargo/config")
        elif [ -f "${cargo_search_dir}/.cargo/config.toml" ]; then
            cargo_config_files+=("${cargo_search_dir}/.cargo/config.toml")
        fi
    done

    for cargo_config in ${cargo_config_files[@]+"${cargo_config_files[@]}"}; do
        # Cargo configuration is TOML, where `"build".target` is a dotted key
        # but `"build.target"` is one unrelated quoted key.  Do not normalize
        # quotes or whitespace with a text parser: doing so can turn a lower-
        # priority cross target into an apparent host target.  Python's
        # standard TOML parser preserves those boundaries.  If it is absent,
        # the file is not valid TOML, includes another file, or uses a target
        # shape other than one non-empty string, fail closed by choosing a
        # sentinel that can never equal rustc's host triple.
        if command -v python3 >/dev/null 2>&1; then
            configured_target="$(read_cargo_build_target "${cargo_config}")"
        else
            configured_target="__NY_UNRESOLVED_CARGO_TARGET__"
        fi
        if [ -n "${configured_target}" ]; then
            effective_build_target="${configured_target}"
        fi
    done
fi
effective_build_target="${effective_build_target:-${rust_host}}"
if [ "${effective_build_target}" = "host-tuple" ]; then
    # Cargo 1.95 treats this documented spelling as the compiler's host
    # triple. Normalize it before deciding whether a host-only ISA flag is
    # safe; it still remains an explicitly targeted build for artifact layout.
    effective_build_target="${rust_host}"
fi
if [ "${rust_host}" = "aarch64-unknown-linux-gnu" ] \
    && [ "${effective_build_target}" = "${rust_host}" ] \
    && [ -r /proc/cpuinfo ] \
    && awk '
        /^Features[[:space:]]*:/ {
            seen = 1
            if ($0 !~ /(^|[[:space:]])asimdhp([[:space:]]|$)/) missing = 1
        }
        END { exit !(seen && !missing) }
    ' /proc/cpuinfo \
    && ! rustc --print cfg 2>/dev/null | grep -q '^target_feature="fp16"$'; then
    case " ${RUSTFLAGS:-} " in
        *"+fp16"*) ;;
        *)
            export RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }-C target-feature=+fp16"
            echo "Enabling AArch64 fp16 for this capable host (LLVM inline-assembly requirement)."
            ;;
    esac
fi

# Build with HiGHS/MIP + the CUDA cuBLAS engine. Competition packages require
# the full `mip,cuda` tier by default: silently accepting a weaker binary makes
# installation look successful while materially reducing scored coverage.
# Developers can explicitly opt into the old best-effort fallback ladder with
# `NY_ALLOW_DEGRADED_BUILD=1`.
# `cuda` routes the sound f64 CROWN GEMMs through cuBLAS (measured 2.1-2.46x on
# CROWN-bound instances) and is SAFE to include everywhere: cudarc dlopens
# CUDA/cuBLAS at runtime (no build-time CUDA toolkit or linking needed), the
# engine factory is lazy, and on CUDA-less hosts ny logs a note and uses the
# same sound CPU f64 path as a no-cuda build (NY_NO_CUDA=1 also disables it).
# `mip` builds the exact revision-pinned AY engine from its canonical Git
# source. A clean source build therefore needs process-scoped AY read access;
# credential-free evaluation should use the validated prebuilt triplet. NY
# remains sound without MIP, but that is a development configuration, not a
# competition-ready release.
SOURCE_BIN=""
artifact_gate_failed=0
last_cargo_status=1

create_submission_staging() {
    local observed_base staging_dir

    if ! mkdir -p "${BUILD_TARGET_BASE}"; then
        echo "ERROR: could not create Cargo staging parent: ${BUILD_TARGET_BASE}" >&2
        return 1
    fi
    observed_base="$(realpath "${BUILD_TARGET_BASE}")"
    if [ "${observed_base}" != "${BUILD_TARGET_BASE}" ]; then
        echo "ERROR: Cargo staging parent changed during setup." >&2
        return 1
    fi
    if ! staging_dir="$(mktemp -d "${BUILD_TARGET_BASE}/.ny-submission-build.XXXXXX")"; then
        echo "ERROR: could not create private Cargo staging directory." >&2
        return 1
    fi
    staging_dir="$(realpath "${staging_dir}")"
    BUILD_STAGING_DIRS+=("${staging_dir}")
    if [[ ! -d "${staging_dir}" \
        || -L "${staging_dir}" \
        || "$(dirname -- "${staging_dir}")" != "${BUILD_TARGET_BASE}" \
        || "$(basename -- "${staging_dir}")" != .ny-submission-build.* ]]; then
        echo "ERROR: Cargo staging directory failed containment validation." >&2
        return 1
    fi

    BUILD_TARGET_DIR="${staging_dir}"
    export CARGO_TARGET_DIR="${BUILD_TARGET_DIR}"
    export CARGO_BUILD_TARGET_DIR="${BUILD_TARGET_DIR}"
}

# Run one Cargo tier and select its ny binary solely from this invocation's
# machine-readable compiler-artifact event. An explicit build.target changes
# Cargo's layout to target/TRIPLE/release/ny, even when TRIPLE is the host. A
# fixed target/release/ny lookup can therefore fail after a good build or,
# worse, silently ship a stale binary from an earlier untargeted invocation.
run_submission_cargo_build() {
    local features="$1"
    local cargo_messages cargo_status selected_executable parse_status
    local -a cargo_arguments

    artifact_gate_failed=0
    if ! create_submission_staging; then
        artifact_gate_failed=1
        return 1
    fi
    if ! cargo_messages="$(mktemp "${BUILD_TARGET_DIR}/.ny-cargo-messages.XXXXXX")"; then
        echo "ERROR: could not create the invocation-scoped Cargo message log." >&2
        artifact_gate_failed=1
        return 1
    fi

    cargo_arguments=(
        build
        --locked
        --release
        -p ny-cli
        --target-dir "${BUILD_TARGET_DIR}"
        --message-format=json-render-diagnostics
    )
    if [ -n "${features}" ]; then
        cargo_arguments+=(--features "${features}")
    fi

    # Cargo keeps ordinary progress and rendered diagnostics on stderr. Its
    # stdout is now a JSON-lines protocol used only for artifact provenance.
    if cargo "${cargo_arguments[@]}" > "${cargo_messages}"; then
        cargo_status=0
    else
        cargo_status=$?
        last_cargo_status="${cargo_status}"
        rm -f "${cargo_messages}"
        return "${cargo_status}"
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: Python 3 is required to authenticate Cargo's emitted ny artifact." >&2
        rm -f "${cargo_messages}"
        artifact_gate_failed=1
        return 1
    fi

    selected_executable="$(python3 -I - \
        "${cargo_messages}" \
        "${TOOL_DIR}/crates/ny-cli/Cargo.toml" \
        "${BUILD_TARGET_DIR}" <<'PY'
import json
import os
import stat
import sys


def reject(message: str) -> None:
    print(f"ERROR: Cargo artifact provenance check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


messages_path, expected_manifest, expected_target_dir = sys.argv[1:]
expected_manifest = os.path.realpath(expected_manifest)
expected_target_dir = os.path.realpath(expected_target_dir)
candidates = []
build_finished = []

try:
    with open(messages_path, encoding="utf-8") as messages:
        for line_number, line in enumerate(messages, 1):
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as error:
                reject(f"line {line_number} is not valid Cargo JSON: {error}")
            if not isinstance(message, dict):
                reject(f"line {line_number} is not a JSON object")

            reason = message.get("reason")
            if reason == "build-finished":
                build_finished.append(message.get("success") is True)
                continue
            if reason != "compiler-artifact":
                continue

            target = message.get("target")
            if not isinstance(target, dict):
                continue
            kind = target.get("kind")
            if target.get("name") != "ny" or not isinstance(kind, list) or "bin" not in kind:
                continue
            manifest = message.get("manifest_path")
            if not isinstance(manifest, str) or os.path.realpath(manifest) != expected_manifest:
                continue
            if message.get("fresh") is not False:
                reject("the ny compiler-artifact was not freshly built in this invocation")
            executable = message.get("executable")
            filenames = message.get("filenames")
            if (
                not isinstance(executable, str)
                or not executable
                or "\n" in executable
                or not isinstance(filenames, list)
                or executable not in filenames
            ):
                reject("the ny compiler-artifact event has no unambiguous executable")
            candidates.append(executable)
except OSError as error:
    reject(f"could not read Cargo messages: {error}")

if build_finished != [True]:
    reject(f"expected one successful build-finished event, got {build_finished!r}")
if len(candidates) != 1:
    reject(f"expected exactly one ny executable, got {len(candidates)}")

executable = candidates[0]
real_executable = os.path.realpath(executable)
try:
    inside_target_dir = os.path.commonpath([real_executable, expected_target_dir]) == expected_target_dir
except ValueError:
    inside_target_dir = False
if not inside_target_dir:
    reject(f"ny executable escaped CARGO_TARGET_DIR: {executable}")
try:
    executable_stat = os.lstat(executable)
except OSError as error:
    reject(f"could not stat ny executable: {error}")
if not stat.S_ISREG(executable_stat.st_mode) or executable_stat.st_mode & 0o111 == 0:
    reject(f"ny executable is missing or not executable: {executable}")
print(executable)
PY
)"
    parse_status=$?
    rm -f "${cargo_messages}"
    if [ "${parse_status}" -ne 0 ] || [ -z "${selected_executable}" ]; then
        artifact_gate_failed=1
        return 1
    fi

    SOURCE_BIN="${selected_executable}"
    last_cargo_status=0
    return 0
}

built=""
if run_submission_cargo_build "mip,cuda"; then
    built="mip,cuda"
elif [ "${artifact_gate_failed}" = "1" ]; then
    exit 1
elif [ "${NY_ALLOW_DEGRADED_BUILD:-0}" != "1" ]; then
    echo "ERROR: required competition feature tier 'mip,cuda' failed to build." >&2
    echo "  Verify access to the pinned AY revision, native build prerequisites, and" >&2
    echo "  crates.io/ORT network access, or package the validated x86_64 prebuilt." >&2
    echo "  For a non-competition development build only, set NY_ALLOW_DEGRADED_BUILD=1." >&2
    exit "${last_cargo_status}"
else
    echo "WARNING: NY_ALLOW_DEGRADED_BUILD=1; trying non-competition fallback tiers." >&2
fi

if [ -z "${built}" ]; then
    # In explicitly degraded development builds, retain the oracle ahead of
    # CUDA-only acceleration: MIP carries relational and graph-MIP coverage.
    for features in "mip" "cuda" ""; do
        if [ -n "${features}" ]; then
            echo "Building ny with --features ${features}..."
            if run_submission_cargo_build "${features}"; then
                built="${features}"
                break
            fi
            if [ "${artifact_gate_failed}" = "1" ]; then
                exit 1
            fi
            echo "WARNING: '--features ${features}' build failed; trying the next fallback tier." >&2
        else
            echo "WARNING: falling back to the bare (no-feature) build so ny still runs and scores." >&2
            if run_submission_cargo_build ""; then
                built="(none)"
            else
                bare_build_status=$?
                if [ "${artifact_gate_failed}" = "1" ]; then
                    exit 1
                fi
                exit "${bare_build_status}"
            fi
        fi
    done
fi
echo "Built ny with features: ${built}"

# LOUD signal on a missing MIP feature. The MILP/ay-milp oracle backs the
# ~360 banked relational UNSATs (the NY_RELATIONAL_UNSAT gate authorizes unsat
# only when `prove_system_infeasible` succeeds — mip-gated) plus graph-mip
# completeness. The required AY revision is Git-pinned, so a missing-MIP build
# points to an explicit developer override, AY access, or another build
# prerequisite failure. Leave a machine-readable marker for post-build
# inspection.
case "${built}" in
    *mip*) has_mip=1 ;;
    *) has_mip=0 ;;
esac
if [ "${has_mip}" != "1" ]; then
    {
        echo "############################################################"
        echo "## CRITICAL: shipped ny binary has NO 'mip' feature (${built})"
        echo "## → the relational unsat oracle is DEAD (~360 pts score unknown)"
        echo "## → AY is Git-pinned; inspect the degraded-build override, AY access,"
        echo "##   and the native/network build prerequisites."
        echo "## Fix: ship a successful mip,cuda source build or a checked prebuilt."
        echo "############################################################"
    } >&2
    echo "NO_MIP built=${built}" > "$(dirname "$0")/../.SUBMISSION_FEATURE_ALERT"
    # Opt-in strict mode: fail the build rather than ship a no-oracle binary.
    if [ "${NY_REQUIRE_MIP:-0}" = "1" ]; then
        echo "NY_REQUIRE_MIP=1 set → refusing to ship a no-mip binary." >&2
        exit 3
    fi
else
    rm -f "$(dirname "$0")/../.SUBMISSION_FEATURE_ALERT" 2>/dev/null || true
fi

if [ -z "${SOURCE_BIN}" ] || [ ! -x "${SOURCE_BIN}" ]; then
    echo "Error: Cargo did not authenticate an executable ny binary from this invocation."
    exit 1
fi

# Publish through directory file descriptors. The temporary regular file is
# fully copied and fsynced in target/release before one rename replaces `ny`.
# O_NOFOLLOW rejects symlinked directory components and source artifacts;
# rename replaces a pre-existing final symlink itself rather than writing
# through it. Until that final syscall succeeds, the prior alias is untouched.
if ! python3 -I - "${SOURCE_BIN}" "${TOOL_DIR}" <<'PY'
import os
import secrets
import stat
import sys


def publish(source_path: str, tool_dir: str) -> None:
    required_flags = ["O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"]
    if any(not hasattr(os, flag) for flag in required_flags):
        raise OSError("platform lacks symlink-safe directory-open flags")

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    source_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    root_fd = target_fd = release_fd = source_fd = temporary_fd = -1
    temporary_name = ""
    published = False

    def open_or_create_directory(parent_fd: int, name: str) -> int:
        created = False
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        child_fd = os.open(name, directory_flags, dir_fd=parent_fd)
        if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
            os.close(child_fd)
            raise OSError(f"publication component is not a directory: {name}")
        if created:
            os.fchmod(child_fd, 0o755)
        return child_fd

    try:
        root_fd = os.open(tool_dir, directory_flags)
        target_fd = open_or_create_directory(root_fd, "target")
        release_fd = open_or_create_directory(target_fd, "release")

        source_fd = os.open(source_path, source_flags)
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_mode & 0o111 == 0:
            raise OSError("authenticated Cargo source is not a regular executable")

        for _ in range(32):
            temporary_name = f".ny-publish-{os.getpid()}-{secrets.token_hex(8)}"
            try:
                temporary_fd = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=release_fd,
                )
                break
            except FileExistsError:
                continue
        if temporary_fd < 0:
            raise OSError("could not allocate a unique publication file")

        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(temporary_fd, view)
                if written <= 0:
                    raise OSError("short write while publishing ny")
                view = view[written:]
            copied += len(chunk)
        if copied != source_stat.st_size:
            raise OSError("Cargo source changed while it was being published")

        os.fchmod(temporary_fd, source_stat.st_mode & 0o777)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        os.replace(
            temporary_name,
            "ny",
            src_dir_fd=release_fd,
            dst_dir_fd=release_fd,
        )
        published = True
    finally:
        if temporary_fd >= 0:
            try:
                os.close(temporary_fd)
            except OSError:
                pass
        if temporary_name and not published and release_fd >= 0:
            try:
                os.unlink(temporary_name, dir_fd=release_fd)
            except OSError:
                pass
        for descriptor in (source_fd, release_fd, target_fd, root_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


try:
    publish(sys.argv[1], sys.argv[2])
except (OSError, ValueError) as error:
    print(f"ERROR: atomic ny publication failed: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
then
    exit 1
fi

SOURCE_IDENTITY_AFTER="$(bash "${RECEIPT_HELPER}" identity "${TOOL_DIR}")"
if [ "${SOURCE_IDENTITY_AFTER}" != "${SOURCE_IDENTITY_BEFORE}" ]; then
    echo "ERROR: NY source identity changed during the submission build." >&2
    echo "  The binary was published without a new receipt and will fail closed." >&2
    exit 1
fi
receipt_features="${built}"
if [ "${receipt_features}" = "(none)" ]; then
    receipt_features="none"
fi
# Receipt publication is the final readiness sentinel.  If this step is
# interrupted, a prior receipt either remains and mismatches the new binary or
# is absent; run_instance.sh rejects both states.
if ! bash "${RECEIPT_HELPER}" create-local \
    "${STABLE_SUBMISSION_BIN}" \
    "${TOOL_DIR}" \
    "${receipt_features}" \
    "${STABLE_SUBMISSION_RECEIPT}"; then
    echo "ERROR: submission binary was published but its freshness receipt was not." >&2
    exit 1
fi

echo "Built submission binary:"
echo "  source: ${SOURCE_BIN}"
echo "  alias:  ${STABLE_SUBMISSION_BIN}"
echo "  receipt: ${STABLE_SUBMISSION_RECEIPT}"
