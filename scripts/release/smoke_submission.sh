#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# smoke_submission.sh -- release-candidate smoke test of a packaged binary
# through the OFFICIAL VNN-COMP contract.
#
#   1. clone the repository at the release commit into a fresh directory (what
#      the organizers do) and drop in exactly the committed release files
#      dist/bin/<target>/{MANIFEST,ny.xz,ny.sha256};
#   2. ./install_tool.sh v1 -- must take the release path, verify, self-test,
#      receipt and install target/release/ny;
#   3. for every row: the organizers' own run_single_instance.sh (which runs
#      prepare_instance.sh then run_instance.sh and appends the official
#      `category,onnx,vnnlib,prepare_time,result,run_time` line to results.csv);
#   4. assert, per row, from observables rather than inference:
#        * verdict vs the bank (sat<->unsat is an EMERGENCY, rc 42),
#        * the run log's NY_BUILD_STAMP_V1 line == MANIFEST build_stamp,
#        * the verifier process received NO NY_* / AY_* variable while the
#          caller's canaries NY_SMOKE_SCRUB_CANARY / AY_SMOKE_SCRUB_CANARY did
#          reach run_instance.sh (so the absence is not an empty grep),
#      and once: installed bytes == MANIFEST binary_sha256, `ny --build-id` ==
#      MANIFEST build_id, no `export NY_` line in vnncomp_scripts/run_instance.sh
#      and none in the root wrapper beyond its NY_BIN/NY_EXPECT_BUILD_ID pass-through.
#
# A foreign target (x86_64 on this aarch64 host) runs under qemu-user through
# scripts/release/exec_shim.c, a process-scoped binfmt_misc (registering the
# real one needs root). Its stated limit: ny's own self-exec children fail
# under qemu-user and take their in-process fallbacks. Budgets are the official
# per-instance timeouts times --emulation-factor (default 40 under emulation, 1
# natively), recorded in every row.
#
# A guard directory first on PATH refuses apt-get/sudo/curl/rustup/cargo/targo,
# so an installer that fell back to a source build fails loudly here instead of
# touching the host.
#
# It is a measurement: run it under the box lock,
#   NY_BOX_WAIT_LOCK=21600 NY_BOX_WAIT_QUIET=21600 scripts/ny_box.sh measure -- \
#     scripts/release/smoke_submission.sh --target x86_64-unknown-linux-gnu ...
# Final line:
#   SMOKE_DONE rc=<worst> WORST=<worst> target=... rows=N match=M emergency=E ...

set -uo pipefail

SELF="$(realpath "${BASH_SOURCE[0]}")"
SRC="$(git -C "$(dirname "${SELF}")" rev-parse --show-toplevel)"
T0=$(date +%s)
RUNID=$(date -u +%Y%m%dT%H%M%SZ)

TARGET=""
ARTIFACTS="${SRC}/dist/bin"
COMMIT=""
# The object store every worktree shares (a linked worktree is not cloneable by path).
REPO="$(git -C "${SRC}" rev-parse --path-format=absolute --git-common-dir)"
ROWS="${SRC}/scripts/release/smoke_rows.tsv"
# Benchmarks are gitignored: they live only in the main tree, next to the shared .git.
BENCH_ROOT="$(dirname "${REPO}")/benchmarks"
HARNESS=""
OUT=""
FACTOR=""
QEMU="${HOME}/x86emu/qemu11u/usr/bin/qemu-x86_64"
QEMU_SYSROOT="${HOME}/x86emu/sysroot"
QEMU_CPU="Haswell-v4"

while [ $# -gt 0 ]; do
    case "$1" in
        --target) TARGET="${2:?}"; shift 2 ;;
        --artifacts) ARTIFACTS="${2:?}"; shift 2 ;;
        --commit) COMMIT="${2:?}"; shift 2 ;;
        --repo) REPO="${2:?}"; shift 2 ;;
        --rows) ROWS="${2:?}"; shift 2 ;;
        --bench-root) BENCH_ROOT="${2:?}"; shift 2 ;;
        --harness) HARNESS="${2:?}"; shift 2 ;;
        --out) OUT="${2:?}"; shift 2 ;;
        --emulation-factor) FACTOR="${2:?}"; shift 2 ;;
        --qemu) QEMU="${2:?}"; shift 2 ;;
        --qemu-sysroot) QEMU_SYSROOT="${2:?}"; shift 2 ;;
        --qemu-cpu) QEMU_CPU="${2:?}"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

FAIL=0
fail0() { echo "SMOKE_PREFLIGHT_FAIL: $*"; FAIL=1; }
ok0() { echo "smoke preflight ok: $*"; }
sha_file() { sha256sum -- "$1" | awk '{print $1}'; }
mvalue() { awk -v k="$1" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }' "$2"; }

HOST_TARGET="$(uname -m)-unknown-linux-gnu"
[ -n "${TARGET}" ] || TARGET="${HOST_TARGET}"
[ -n "${HARNESS}" ] || HARNESS="${BENCH_ROOT}/vnncomp2025/run_single_instance.sh"
[ -n "${OUT}" ] || OUT="${SRC}/dist/smoke/${TARGET}-${RUNID}"
EMULATED=0
[ "${TARGET}" != "${HOST_TARGET}" ] && EMULATED=1
if [ -z "${FACTOR}" ]; then
    FACTOR=1
    [ "${EMULATED}" = 1 ] && FACTOR=40
fi
ART="${ARTIFACTS}/${TARGET}"

echo "smoke: runid=${RUNID} target=${TARGET} host=${HOST_TARGET} emulated=${EMULATED} factor=${FACTOR} out=${OUT}"
for tool in git bc cc sha256sum timeout python3 awk; do
    command -v "${tool}" >/dev/null 2>&1 && ok0 "tool ${tool}" || fail0 "tool ${tool} missing"
done
[ -f "${HARNESS}" ] && ok0 "official harness ${HARNESS} sha256=$(sha_file "${HARNESS}")" || fail0 "official harness ${HARNESS} missing"
for f in MANIFEST ny.xz ny.sha256; do
    [ -f "${ART}/${f}" ] && [ ! -L "${ART}/${f}" ] || fail0 "release file ${ART}/${f} missing"
done
BIN_SHA="$(mvalue binary_sha256 "${ART}/MANIFEST" 2>/dev/null)"
BUILD_ID="$(mvalue build_id "${ART}/MANIFEST" 2>/dev/null)"
BUILD_STAMP="$(mvalue build_stamp "${ART}/MANIFEST" 2>/dev/null)"
NY_COMMIT="$(mvalue ny_commit "${ART}/MANIFEST" 2>/dev/null)"
[ -n "${COMMIT}" ] || COMMIT="${NY_COMMIT}"
[ "${COMMIT}" = "${NY_COMMIT}" ] && [ -n "${COMMIT}" ] && ok0 "MANIFEST ny_commit ${NY_COMMIT}" \
    || fail0 "MANIFEST ny_commit '${NY_COMMIT}' is not the smoke commit '${COMMIT}'"
[ -n "${BIN_SHA}" ] && [ -n "${BUILD_ID}" ] && [ -n "${BUILD_STAMP}" ] && ok0 "MANIFEST identity: ${BUILD_STAMP}" \
    || fail0 "MANIFEST lacks binary_sha256/build_id/build_stamp"
git -C "${REPO}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null && ok0 "${REPO} has ${COMMIT}" || fail0 "${REPO} lacks ${COMMIT}"
if [ "${EMULATED}" = 1 ]; then
    [ "${TARGET}" = x86_64-unknown-linux-gnu ] || fail0 "no emulation rig for ${TARGET}"
    [ -x "${QEMU}" ] && [ -d "${QEMU_SYSROOT}" ] && ok0 "qemu $("${QEMU}" --version | head -1), -cpu ${QEMU_CPU}, -L ${QEMU_SYSROOT}" \
        || fail0 "qemu ${QEMU} / sysroot ${QEMU_SYSROOT} missing"
fi
[ -e "${OUT}" ] && fail0 "${OUT} exists (one directory per smoke run)"

# Rows: re-derive each expected verdict from its declared source.
declare -a ROW_ID ROW_DIR ROW_CAT ROW_ONNX ROW_VNNLIB ROW_EXPECT ROW_BUDGET
while IFS=$'\t' read -r id dir cat onnx vnnlib expect source; do
    [ -z "${id}" ] || [ "${id:0:1}" = "#" ] && continue
    base="${BENCH_ROOT}/${dir}"
    [ -f "${base}/${onnx}" ] && [ -f "${base}/${vnnlib}" ] || { fail0 "${id}: ${base}/${onnx} or ${vnnlib} missing"; continue; }
    budget="$(awk -F, -v o="${onnx}" -v v="${vnnlib}" '$1==o && $2==v {gsub(/[[:space:]]/,"",$3); print $3; exit}' "${base}/instances.csv")"
    [ -n "${budget}" ] || { fail0 "${id}: no official budget in ${base}/instances.csv"; continue; }
    case "${source}" in
        readme:*)
            grep -qF -- "${source#readme:}" "${base}/README.md" 2>/dev/null \
                || { fail0 "${id}: ${base}/README.md does not state '${source#readme:}'"; continue; }
            ;;
        *)
            banked="$(awk -F, -v c="${cat}" -v o="${onnx}" -v v="${vnnlib}" '$1==c && $2==o && $3==v {print $5}' "${SRC}/${source}" | sort -u)"
            [ "${banked}" = "${expect}" ] || { fail0 "${id}: bank ${source} says '${banked}', rows file says '${expect}'"; continue; }
            ;;
    esac
    ROW_ID+=("${id}"); ROW_DIR+=("${base}"); ROW_CAT+=("${cat}"); ROW_ONNX+=("${onnx}")
    ROW_VNNLIB+=("${vnnlib}"); ROW_EXPECT+=("${expect}"); ROW_BUDGET+=("${budget}")
done < "${ROWS}"
[ "${#ROW_ID[@]}" -gt 0 ] && ok0 "${#ROW_ID[@]} rows, expected verdicts re-derived from their sources" || fail0 "no usable rows"

if [ "${FAIL}" != 0 ]; then
    echo "SMOKE_DONE rc=3 WORST=3 target=${TARGET} preflight=FAIL elapsed=$(( $(date +%s) - T0 ))"
    exit 3
fi

mkdir -p "${OUT}/guard" "${OUT}/rows" || exit 2
SHIM="${OUT}/exec_shim.so"
cc -shared -fPIC -O2 -Wall -Wextra -Werror -o "${SHIM}" "${SRC}/scripts/release/exec_shim.c" -ldl \
    || { echo "SMOKE_DONE rc=4 WORST=4 target=${TARGET} shim=build-failed"; exit 4; }
for tool in apt-get sudo curl rustup cargo targo; do
    printf '#!/bin/sh\necho "SMOKE_GUARD: refused %s $*" >&2\nexit 97\n' "${tool}" >"${OUT}/guard/${tool}"
    chmod +x "${OUT}/guard/${tool}"
done

# ---- 1. the organizers' clone + the committed release files ----------------
TOOL="$(realpath -m "${OUT}/tool")"
git clone --quiet --shared --no-checkout "${REPO}" "${TOOL}" && git -C "${TOOL}" checkout --quiet --detach "${COMMIT}" \
    || { echo "SMOKE_DONE rc=5 WORST=5 target=${TARGET} clone=failed"; exit 5; }
mkdir -p "${TOOL}/dist/bin/${TARGET}"
cp -- "${ART}/MANIFEST" "${ART}/ny.xz" "${ART}/ny.sha256" "${TOOL}/dist/bin/${TARGET}/"
# Static half of the environment claim. The script that execs ny
# (vnncomp_scripts/run_instance.sh) must export no NY_* name at all. The root
# wrapper may re-export only its two documented control-plane pass-throughs,
# NY_BIN and NY_EXPECT_BUILD_ID (and only when the caller set them); the helper
# unsets both before starting ny. The exec-boundary log below is the dynamic,
# authoritative half.
EXPORT_NY="$(grep -cE '^[[:space:]]*export[[:space:]]+NY_' "${TOOL}/vnncomp_scripts/run_instance.sh")"
EXPORT_NY_WRAPPER="$(grep -E '^[[:space:]]*export[[:space:]]+NY_' "${TOOL}/run_instance.sh" \
    | grep -cvE '^[[:space:]]*export[[:space:]]+(NY_BIN|NY_EXPECT_BUILD_ID)="\$scored_(ny_bin|expect_build_id)"$')"
EXPORT_NY=$((EXPORT_NY + EXPORT_NY_WRAPPER))
echo "smoke: clone ${TOOL} @ ${COMMIT}; undeclared 'export NY_' lines in the run_instance.sh pair: ${EXPORT_NY}; status: [$(git -C "${TOOL}" status --porcelain=v1 | tr '\n' ' ')]"

# ---- the rig environment ------------------------------------------------------
export PATH="${OUT}/guard:${PATH}"
export LD_PRELOAD="${SHIM}"
export X86SHIM_SUBJECT="${TOOL}/target/release/ny"
export NY_SMOKE_SCRUB_CANARY=1 AY_SMOKE_SCRUB_CANARY=1
if [ "${EMULATED}" = 1 ]; then
    export X86SHIM_QEMU="${QEMU}" X86SHIM_QEMU_CPU="${QEMU_CPU}" X86SHIM_QEMU_SYSROOT="${QEMU_SYSROOT}"
    export X86SHIM_UNAME_MACHINE=x86_64
fi

# ---- 2. install_tool.sh v1 ---------------------------------------------------
export X86SHIM_LOG="${OUT}/install.exec.log"
( cd "${TOOL}" && timeout 3600 ./install_tool.sh v1 ) >"${OUT}/install.log" 2>&1
INSTALL_RC=$?
INSTALL_OK=$(grep -c '^NY_RELEASE_INSTALL_OK ' "${OUT}/install.log")
INSTALLED_SHA="$(sha_file "${TOOL}/target/release/ny" 2>/dev/null)"
RECEIPT_KIND="$(mvalue source_kind "${TOOL}/target/release/ny.receipt" 2>/dev/null)"
echo "smoke: install rc=${INSTALL_RC} release_ok_lines=${INSTALL_OK} installed_sha256=${INSTALLED_SHA:-none} receipt_kind=${RECEIPT_KIND:-none}"
grep -E '^(NY_RELEASE_VERIFY_OK|NY_RELEASE_INSTALL_OK|Release |ERROR|WARNING|SMOKE_GUARD)' "${OUT}/install.log" | sed 's/^/  install| /'
INSTALL_STATUS=ok
if [ "${INSTALL_RC}" -ne 0 ] || [ "${INSTALL_OK}" -ne 1 ] || [ "${INSTALLED_SHA}" != "${BIN_SHA}" ] || [ "${RECEIPT_KIND}" != release ]; then
    INSTALL_STATUS=FAIL
fi

export X86SHIM_LOG="${OUT}/buildid.exec.log"
FOUND_ID="$(timeout 1200 "${TOOL}/target/release/ny" --build-id 2>/dev/null | grep -m1 '^NY_BUILD_ID_V1 ')"
BUILD_ID_STATUS=ok
[ "${FOUND_ID}" = "${BUILD_ID}" ] || BUILD_ID_STATUS=MISMATCH
echo "smoke: installed --build-id ${BUILD_ID_STATUS}: ${FOUND_ID:-<none>}"

# ---- 3. rows through the official harness --------------------------------------
RESULTS="${OUT}/results.csv"
: >"${RESULTS}"
TSV="${OUT}/rows.tsv"
printf 'id\tcategory\tonnx\tvnnlib\texpected\tresult\tclass\tofficial_budget\tharness_budget\twall\tstamp_lines\tny_execs\tny_env_violations\tcanary_seen\tlog\n' >"${TSV}"
N=0; MATCH=0; EMERG=0; INCONC=0; ENV_BAD=0; CANARY_MISS=0; STAMP_BAD=0; HARNESS_BAD=0
if [ "${INSTALL_STATUS}" = ok ]; then
    for i in "${!ROW_ID[@]}"; do
        id="${ROW_ID[$i]}"; rd="${OUT}/rows/${id}"; mkdir -p "${rd}"
        hb=$(( ${ROW_BUDGET[$i]%%.*} * FACTOR ))
        export X86SHIM_LOG="${rd}/exec.log"
        before=$(grep -c . "${RESULTS}")
        t0=$(date +%s.%N)
        ( cd "${rd}" && bash "${HARNESS}" v1 "${TOOL}" "${ROW_CAT[$i]}" "${ROW_DIR[$i]}/${ROW_ONNX[$i]}" \
            "${ROW_DIR[$i]}/${ROW_VNNLIB[$i]}" "${hb}" "${RESULTS}" "${rd}/counterexample.txt" ) >"${rd}/run.log" 2>&1
        wall=$(echo "$(date +%s.%N) - ${t0}" | bc)
        N=$((N + 1))
        after=$(grep -c . "${RESULTS}")
        result=""
        if [ "${after}" -eq $((before + 1)) ]; then
            result="$(tail -1 "${RESULTS}" | awk -F, '{print $5}')"
        else
            HARNESS_BAD=$((HARNESS_BAD + 1))
        fi
        expect="${ROW_EXPECT[$i]}"
        case "${expect}:${result}" in
            unsat:unsat|sat:sat) class=MATCH; MATCH=$((MATCH + 1)) ;;
            unsat:sat|sat:unsat)
                class=EMERGENCY; EMERG=$((EMERG + 1))
                echo "EMERGENCY: ${id} ${ROW_CAT[$i]} ${ROW_ONNX[$i]} ${ROW_VNNLIB[$i]} expected ${expect}, the packaged binary answered ${result} -- STOP AND REPORT"
                ;;
            *) class=INCONCLUSIVE; INCONC=$((INCONC + 1)) ;;
        esac
        stamp_lines=$(grep -a -c -x -F -- "${BUILD_STAMP}" "${rd}/run.log")
        [ "${stamp_lines}" -ge 1 ] || STAMP_BAD=$((STAMP_BAD + 1))
        subject_execs=$(awk -v p="${X86SHIM_SUBJECT}" '$0 ~ /^X86SHIM / { for (f = 1; f <= NF; f++) if ($f == "path=" p) { n++ } } END { print n + 0 }' "${rd}/exec.log" 2>/dev/null)
        violations=$(awk -v p="${X86SHIM_SUBJECT}" '$0 ~ /^X86SHIM / { s = 0; for (f = 1; f <= NF; f++) { if ($f == "path=" p) s = 1 }
            if (s) for (f = 1; f <= NF; f++) if (($f ~ /^env_ny=/ && $f != "env_ny=-") || ($f ~ /^env_ay=/ && $f != "env_ay=-") || $f == "ld_preload=1") n++ } END { print n + 0 }' "${rd}/exec.log" 2>/dev/null)
        canary=$(grep -E 'path=[^ ]*/run_instance\.sh ' "${rd}/exec.log" 2>/dev/null | grep -c 'NY_SMOKE_SCRUB_CANARY')
        [ "${subject_execs}" -ge 1 ] && [ "${violations}" -eq 0 ] || ENV_BAD=$((ENV_BAD + 1))
        [ "${canary}" -ge 1 ] || CANARY_MISS=$((CANARY_MISS + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${id}" "${ROW_CAT[$i]}" "${ROW_ONNX[$i]}" \
            "${ROW_VNNLIB[$i]}" "${expect}" "${result:-none}" "${class}" "${ROW_BUDGET[$i]}" "${hb}" "${wall}" \
            "${stamp_lines}" "${subject_execs}" "${violations}" "${canary}" "${rd}/run.log" >>"${TSV}"
        echo "smoke row ${id} ${ROW_CAT[$i]} ${ROW_ONNX[$i]##*/} ${ROW_VNNLIB[$i]##*/}: expected=${expect} result=${result:-none} ${class} wall=${wall}s budget=${hb}s stamp_lines=${stamp_lines} ny_execs=${subject_execs} env_violations=${violations} canary=${canary}"
    done
fi

WORST=0
bump() { [ "$1" -gt "${WORST}" ] && WORST="$1"; }
[ "${INCONC}" -gt 0 ] && bump 3
[ "${HARNESS_BAD}" -gt 0 ] && bump 5
[ "${STAMP_BAD}" -gt 0 ] && bump 11
[ "${BUILD_ID_STATUS}" != ok ] && bump 12
[ "${CANARY_MISS}" -gt 0 ] && bump 13
[ "${ENV_BAD}" -gt 0 ] || [ "${EXPORT_NY}" -ne 0 ] && bump 14
[ "${INSTALL_STATUS}" != ok ] && bump 20
[ "${EMERG}" -gt 0 ] && bump 42
echo "SMOKE_DONE rc=${WORST} WORST=${WORST} target=${TARGET} emulated=${EMULATED} factor=${FACTOR} rows=${N} match=${MATCH} emergency=${EMERG} inconclusive=${INCONC} harness_errors=${HARNESS_BAD} env_violation_rows=${ENV_BAD} canary_missing_rows=${CANARY_MISS} stamp_mismatch_rows=${STAMP_BAD} export_ny_lines=${EXPORT_NY} install=${INSTALL_STATUS} build_id=${BUILD_ID_STATUS} binary_sha256=${BIN_SHA} results=${RESULTS} elapsed=$(( $(date +%s) - T0 ))"
exit "${WORST}"
