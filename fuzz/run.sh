#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Build one fuzz target on the Trust toolchain and run it under libFuzzer.
# This is the Trust spelling of `cargo +nightly fuzz run <target> -- <args>`,
# which stays the recipe on a host without the Trust toolchain.
#
# Why not `targo fuzz run`: cargo-fuzz spawns a bare `cargo` from PATH
# (`Command::new("cargo")`), not `$CARGO`, so under targo the fuzz build would
# run on whatever stock `cargo` PATH resolves rather than on a Targo lane. This
# script does cargo-fuzz's build step itself on the explicit UNVERIFIED lane
# (fuzz binaries are test harnesses and carry no proof claim), with the same
# instrumentation flags `cargo fuzz build -s none` uses. No sanitizer: the
# Trust aarch64-apple-darwin sysroot ships no ASan runtime. The Trust compiler
# accepts the nightly-only flags cargo-fuzz needs, so no nightly is required.
#
# Usage: fuzz/run.sh <target> [libFuzzer args...]
#   fuzz/run.sh bound_propagation_soundness -max_total_time=60
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: fuzz/run.sh <target> [libFuzzer args...]" >&2
    exit 2
fi
target=$1
shift

here=$(cd "$(dirname "$0")" && pwd -P)
: "${TARGO:=targo}"
triple=$(trustc -vV 2>/dev/null | sed -n 's/^host: //p')
triple=${triple:-aarch64-apple-darwin}

# --target keeps these flags off build scripts and proc macros, as cargo-fuzz does.
RUSTFLAGS="${RUSTFLAGS:-} -Cpasses=sancov-module \
-Cllvm-args=-sanitizer-coverage-level=4 \
-Cllvm-args=-sanitizer-coverage-inline-8bit-counters \
-Cllvm-args=-sanitizer-coverage-pc-table \
-Cllvm-args=-sanitizer-coverage-trace-compares \
--cfg fuzzing -Cdebug-assertions -Coverflow-checks" \
    "$TARGO" --unverified build --manifest-path "$here/Cargo.toml" \
    --release --target "$triple" --bin "$target"

# Same corpus and crash-artifact layout cargo-fuzz uses (fuzz/.gitignore).
corpus="$here/corpus/$target"
artifacts="$here/artifacts/$target"
mkdir -p "$corpus" "$artifacts"
exec "$here/target/$triple/release/$target" -artifact_prefix="$artifacts/" "$corpus" "$@"
