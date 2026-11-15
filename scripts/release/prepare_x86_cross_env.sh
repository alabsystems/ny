#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# prepare_x86_cross_env.sh -- one-time, user-space preparation of the x86_64
# cross environment that scripts/release/make_submission.sh uses to build the
# x86_64-unknown-linux-gnu competition binary on an aarch64 Linux host.
#
# This is the recipe that produced ny-pins/ny_x86_64_96e234a02 on 2026-09-28
# (scratchpad/x86build/prep.sh), parametrised by directory and by the linked
# Trust toolchain instead of hard-coded. It invents no toolchain: the compiler is
# the linked `trust` toolchain's own x86 backend; the C/C++ side is the host's
# clang-18 driving lld; the target libraries are Ubuntu 24.04 (noble) amd64
# packages, the same glibc 2.39 the evaluation image runs.
#
#   <dir>/debs/            4 noble-updates amd64 -dev debs, each SHA-256-pinned and,
#                          while the signed index still lists it, checked against the
#                          Packages index, whose own SHA-256 is checked against the
#                          gpgv-verified InRelease
#   <dir>/xsysroot/        a COPY of the runtime sysroot (--runtime-sysroot) with
#                          those debs extracted into it
#   <dir>/trust-src-<c9>/  `git archive` of the compiler's own commit (library +
#                          crates/trust-memchr): the std that -Zbuild-std compiles
#   <dir>/bin/             clang-18 cross wrappers bound to <dir>/xsysroot
#
# No sudo, no system package install, no writes to the Trust checkout. It is
# idempotent and never deletes: an existing xsysroot or std tree is kept as is
# (make_submission.sh re-verifies both before every build).
#
# usage: scripts/release/prepare_x86_cross_env.sh [--dir DIR] [--runtime-sysroot DIR]

set -uo pipefail

DIR="${HOME}/.cache/ny-release/x86_64-cross"
RUNTIME_SYSROOT="${HOME}/x86emu/sysroot"
while [ $# -gt 0 ]; do
    case "$1" in
        --dir) DIR="${2:?}"; shift 2 ;;
        --runtime-sysroot) RUNTIME_SYSROOT="${2:?}"; shift 2 ;;
        *) echo "usage: $0 [--dir DIR] [--runtime-sysroot DIR]" >&2; exit 2 ;;
    esac
done

die() { echo "PREP_FAIL: $*" >&2; exit 1; }
for tool in curl gpgv xz dpkg-deb sha256sum git tar clang-18 clang++-18 rustup; do
    command -v "${tool}" >/dev/null 2>&1 || die "missing tool ${tool}"
done
[ -r /usr/share/keyrings/ubuntu-archive-keyring.gpg ] || die "missing /usr/share/keyrings/ubuntu-archive-keyring.gpg"
[ -e "${RUNTIME_SYSROOT}/usr/lib/x86_64-linux-gnu/libc.so.6" ] \
    || die "runtime sysroot ${RUNTIME_SYSROOT} has no x86_64 libc.so.6 (see scratchpad/x86replay/setup_env.sh)"

TRUSTC_VV="$(rustup run trust rustc -vV)" || die "rustup run trust rustc -vV"
TRUST_COMMIT="$(printf '%s\n' "${TRUSTC_VV}" | awk '/^commit-hash:/{print $2}')"
SYSROOT="$(rustup run trust rustc --print sysroot)"
TRUST_SRC_REPO="$(readlink -f "${SYSROOT}/lib/rustlib/rustc-src/rust")"
[[ "${TRUST_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "cannot read the Trust compiler commit"
git -C "${TRUST_SRC_REPO}" cat-file -e "${TRUST_COMMIT}^{commit}" 2>/dev/null \
    || die "the Trust source ${TRUST_SRC_REPO} does not contain ${TRUST_COMMIT}"

BASE=http://archive.ubuntu.com/ubuntu
SUITE=noble-updates
# Each deb is pinned by SHA-256. The pins are the digests the 2026-09-28 prep
# verified against the then-current gpgv-signed index; noble-updates moves on
# (linux-libc-dev 6.8.0-142 had left the index by 2026-10-01 while staying in
# the pool), so today's index is checked when it still lists a deb and the pin
# alone decides when it does not. A deb matching neither is refused.
DEBS=(
    "pool/main/g/glibc/libc6-dev_2.39-0ubuntu8.9_amd64.deb e13d5fcc1b2a86f75bca8e0026a8e39f24fe97ca86e92be79991f8697eb1306f"
    "pool/main/l/linux/linux-libc-dev_6.8.0-142.142_amd64.deb 937db1a88a4fa2ea97fd4eab89f2cd9d077290f6a26bcd27b1a6d24fa3d706b6"
    "pool/main/g/gcc-13/libgcc-13-dev_13.3.0-6ubuntu2~24.04.1_amd64.deb cd689db2691edaa10f37329307292796bb599e722e0505c79e14caaa1fe9a93a"
    "pool/main/g/gcc-13/libstdc++-13-dev_13.3.0-6ubuntu2~24.04.1_amd64.deb ee5633e863e19c3381ed97842ce35ed32ede96a3d1ae4e94c051d3036fe21347"
)

mkdir -p "${DIR}/debs" "${DIR}/index" "${DIR}/bin" || die "cannot create ${DIR}"
cd "${DIR}" || die "cannot enter ${DIR}"

echo "== index: InRelease (gpgv) -> Packages.xz sha256"
curl -sfo index/InRelease "${BASE}/dists/${SUITE}/InRelease" || die "InRelease download"
gpgv --keyring /usr/share/keyrings/ubuntu-archive-keyring.gpg index/InRelease >index/gpgv.log 2>&1 \
    || { cat index/gpgv.log; die "InRelease signature"; }
curl -sfo index/Packages.xz "${BASE}/dists/${SUITE}/main/binary-amd64/Packages.xz" || die "Packages.xz download"
want="$(awk '/^SHA256:/{s=1;next} /^[A-Za-z]/{s=0} s && $3=="main/binary-amd64/Packages.xz"{print $1}' index/InRelease)"
[ -n "${want}" ] && [ "${want}" = "$(sha256sum index/Packages.xz | awk '{print $1}')" ] \
    || die "Packages.xz does not match the signed InRelease"
xz -dc index/Packages.xz >index/Packages || die "Packages decompress"

echo "== debs: download + sha256 against the pin and, while it lists them, the signed index"
for entry in "${DEBS[@]}"; do
    f="${entry% *}"
    pin="${entry##* }"
    b="$(basename "${f}")"
    [ -s "debs/${b}" ] || curl -sfo "debs/${b}" "${BASE}/${f}" || die "download ${b}"
    got="$(sha256sum "debs/${b}" | awk '{print $1}')"
    indexed="$(awk -v fn="${f}" '/^Filename: /{cur=$2} /^SHA256: /{ if (cur==fn) print $2 }' index/Packages)"
    echo "${b} pin=${pin} index=${indexed:-<no longer listed>} got=${got}"
    [ "${got}" = "${pin}" ] || die "sha256 of ${b} is not its pin"
    [ -z "${indexed}" ] || [ "${indexed}" = "${got}" ] || die "the signed index disagrees with the pin for ${b}"
done
sha256sum debs/*.deb >debs/SHA256SUMS

echo "== xsysroot: copy the runtime sysroot, extract the dev debs"
if [ -e xsysroot ]; then
    echo "xsysroot exists; kept (make_submission.sh re-verifies it)"
else
    mkdir xsysroot && cp -a "${RUNTIME_SYSROOT}/." xsysroot/ || die "sysroot copy"
    for d in debs/*.deb; do dpkg-deb -x "${d}" xsysroot || die "dpkg-deb -x ${d}"; done
fi

echo "== std source at the compiler's own commit ${TRUST_COMMIT}"
# library/Cargo.toml patches memchr to ../crates/trust-memchr -- the only path
# that escapes library/ -- so the archive carries that crate too.
SRC_DIR="trust-src-${TRUST_COMMIT:0:9}"
TAR="${SRC_DIR}-library+trust-memchr.tar"
if [ ! -e "${TAR}" ]; then
    git -C "${TRUST_SRC_REPO}" archive --format=tar "${TRUST_COMMIT}" library crates/trust-memchr >"${TAR}" \
        || die "git archive"
    sha256sum "${TAR}" >"${TAR}.sha256"
fi
if [ ! -e "${SRC_DIR}" ]; then
    mkdir -p "${SRC_DIR}" && tar -xf "${TAR}" -C "${SRC_DIR}" || die "tar -x"
fi

echo "== clang-18 cross wrappers"
cat >bin/x86_64-linux-gnu-clang <<EOF
#!/bin/sh
# clang-18 driving lld (ld.lld on PATH) against the x86_64 cross sysroot.
exec /usr/bin/clang-18 --target=x86_64-linux-gnu --sysroot=${DIR}/xsysroot --gcc-toolchain=${DIR}/xsysroot/usr -fuse-ld=lld -Qunused-arguments "\$@"
EOF
sed 's#^exec /usr/bin/clang-18 #exec /usr/bin/clang++-18 #' bin/x86_64-linux-gnu-clang >bin/x86_64-linux-gnu-clang++
chmod +x bin/x86_64-linux-gnu-clang bin/x86_64-linux-gnu-clang++
command -v ld.lld >/dev/null 2>&1 || echo "WARNING: ld.lld is not on PATH; the wrappers link with -fuse-ld=lld"
echo "PREP_OK dir=${DIR} trust=${TRUST_COMMIT:0:9} debs_sha256sums=$(sha256sum debs/SHA256SUMS | awk '{print $1}')"
