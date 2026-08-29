#!/usr/bin/env bash
# gate.sh — the first-party drift gate.
#
# There is no CI in this repository: every push-storm re-drifts the lint gate
# and the packaging invariants, and each audit has re-zeroed them by hand.
# This script is the one local command that runs every drift-prone check with
# a per-check scoreboard (PASS/FAIL lines, final summary, nonzero exit on any
# FAIL).
#
# Checks:
#   a  Rust hygiene gate   targo fmt --all -- --check, Cargo-owned source policy,
#                          ny-levers declarations/direct-literal ratchet, and
#                          migration inventory (no Python interpreter), then
#                          targo tippy --locked --workspace --exclude ny-python
#                          --all-targets -- -D warnings, minus the crates in
#                          CLIPPY_SKIPPED_CRATES (printed as SKIPPED + reason),
#                          plus the shipped ny-cli `mip,cuda` feature tier
#   b  ny-cert tests        cargo test --locked -p ny-cert --all-targets
#   c  ny CLI contracts     focused submission/cert, measured-delivery,
#                           dependency-pin, layered-flight-receipt, and
#                           alpha-zero-yield contract tests
#   d  ny-propagate units   cargo test --locked -p ny-propagate --lib collection
#                           && cargo test --locked -p ny-propagate --lib conv2d
#   e  harness pytest       env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1
#                           $NY_GATE_PYTHON -s -m pytest tests -q
#   f  submission           regenerate a tarball to a temp path (debug ny
#      invariants           binary, --no-build) and assert the packaging
#                           soundness invariants (see check_f below)
#   g  Python coherence     pytest plugin tests plus Cargo-owned black-box
#                           contracts for repository Python tooling and policy
#   h  preset capability    cargo test --locked -p ny-cli --bin ny
#      (STATIC, seconds)     preset::backend_capability_tests — every shipped
#                            preset's declared `device:` is HONOURED at runtime or
#                            covered by a dated, measured waiver in
#                            configs/backend_capability_waivers.yaml. Catches a
#                            fail-closed backend gate (e.g. the WGPU proof
#                            quarantine, 1ede1d30) that silently downgrades 16
#                            presets to the CPU verifier.
#   i  preset model load    cargo test --locked --release -p ny-cli
#                            --features external-vnncomp --bin ny
#      (STATIC, ~1 min)      preset::model_load_smoke_tests — every model a
#                            shipped preset points at still LOADS, and every
#                            benchmark root declares its BANKED NORMALIZED in
#                            configs/preset_score_at_risk.yaml so a failure names
#                            the points it deletes. Catches a fail-closed loader
#                            gate (e.g. the Cast-dtype gate, 25dee0c5) that
#                            zeroes a banked category by making its models
#                            unloadable — eight such gates have shipped.
#                            ABSENT BENCHMARK DATA IS A HARD FAILURE, not a skip:
#                            a pass with no data checked NOTHING, and that
#                            vacuous green is how a real regression gets
#                            misfiled as inherited. Symlink the year roots from
#                            a checkout that has the benchmark data; this lane
#                            deliberately cannot be disabled.
#                            NY_PRESET_LOAD_SMOKE=all loads every model instead of
#                            one per architecture family.
#   j  soundness oracles    cargo test --locked --release -p ny-propagate --lib,
#      (--release, ~4 min)   three filters: margin_row::tests:: (the RootEval.dj
#                            ADVERSARIAL ENCLOSURE oracles — the certified
#                            per-class lower bound never overshoots the true
#                            feasible margin at any sampled point, i.e. no
#                            false-UNSAT), plus wide_alpha_true::tests:: and
#                            interm_refine::. Check [d] runs the collection and
#                            conv2d slices ONLY, so before this lane existed the
#                            gate COMPILED the moat's own oracles on every run
#                            and never executed one of them.
#                            --release is mandatory: a debug ny-propagate test
#                            binary dies on this host inside gemm-common (pulled
#                            in by faer) on fullfp16 instructions the target
#                            rejects, and the oracles' sampling budgets are
#                            minutes of arithmetic at debug speed anyway.
#                            Every filter carries a MINIMUM test count, because
#                            `cargo test <filter>` exits 0 when the filter
#                            matches NOTHING — the same vacuous green that check
#                            [i] refuses for absent benchmark data.
#   k  ny-api suite          cargo test --locked -p ny-api --features cert
#                            --all-targets (--all-targets EXCLUDES doctests, which
#                            the `trust` toolchain cannot run) —
#      (seconds)             the integration tests that were SILENTLY
#                            SKIPPED. Every [[test]] target carries
#                            required-features, and cargo skips an unmet test
#                            target with NO warning and exit 0, so before
#                            `default = ["cert"]` landed this crate ran 26 unit
#                            tests, skipped all 8 integration binaries, and
#                            reported success. --features is passed EXPLICITLY
#                            so the lane does not depend on the default key
#                            staying put. Carries a MINIMUM count and asserts
#                            two cert-tier binaries BY NAME, so "the targets
#                            built but the cert tier went dark" fails closed.
#   l  activation-math       cargo test --locked -p ny-propagate --lib
#      certified-bound pins  silu:: plus cargo test --locked
#      (unmeasured)          -p ny-relaxation --all-targets. SiLU's certified
#                            IBP lower bound is `next_down_f32` of SiLU
#                            evaluated at a critical point found by NEWTON'S
#                            METHOD IN f32 with a central-difference second
#                            derivative (silu/math.rs). ONE f32 ULP of backoff
#                            is the entire margin, so soundness rested on
#                            Newton having converged and NOTHING asserted it.
#                            The `newton_pin` modules pin it against an
#                            mpmath-derived minimum (dps 60), so unlike the
#                            assertions they replaced they cannot drift with
#                            the code under test. Before this lane NO SiLU test
#                            ran in the gate at all, and `ny-relaxation` — the
#                            mirror crate that exists to detect drift against
#                            ny-propagate — appeared ZERO times in this script.
#                            Both slices carry a MINIMUM count and assert all
#                            four pins BY ANCHORED NAME in BOTH crates.
#
# NOT in this gate (needs verification runs, minutes to hours):
#   scripts/check_banked_rows.py --bin <release ny> — replays a per-category
#   sample of banked SOLVED ledger rows through `ny vnncomp` and names the CAUSE
#   of each non-reproduction. Run it before banking and after any fail-closed
#   gate lands.
#
# Flags:
#   --fast        run the bounded edit-loop lane (checks a-h; skips i-n)
#   --only <x>    run a single check by letter (a-n)
#   -h | --help   usage
# Environment:
#   NY_GATE_PYTHON  Python interpreter provisioned from requirements.txt
#                   (default: ./.venv/bin/python or ./.venv/Scripts/python.exe
#                   when present, else python3)
#   NY_SKIP_CREDENTIAL_CHECK=1
#                   skip the github.com credential precondition. It normally
#                   costs 0.027s and no network when no token is set, and it
#                   refuses when a token IS set and does not authenticate —
#                   because every cargo lane then dies at dependency
#                   resolution (crates/ny-mip pulls ay-milp from a private
#                   repo) and reports VOID as FAIL. Skipping does not fix it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Parse control-only arguments before resolving any gate prerequisite. Help is
# repository documentation: it must remain available on a machine that has not
# installed the pinned Trust frontend yet. Every executing mode continues into
# the fail-closed frontend check below.
usage() {
  # Print the header block: every line from line 2 up to the first non-comment
  # line. This follows the header as checks and flags are added.
  awk 'NR >= 2 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

FAST=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --fast) FAST=1 ;;
    --only)
      [ $# -ge 2 ] || {
        echo "gate.sh: --only requires a check letter a-n" >&2
        exit 2
      }
      shift
      ONLY="$1"
      ;;
    --only=*) ONLY="${1#--only=}" ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "gate.sh: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done
if [ -n "$ONLY" ]; then
  case "$ONLY" in
    a|b|c|d|e|f|g|h|i|j|k|l|m|n) ;;
    *)
      echo "gate.sh: --only takes a single check letter a-n, got: '$ONLY'" >&2
      exit 2
      ;;
  esac
fi

# Prefer the repository virtualenv the README tells you to create. Defaulting
# straight to `python3` meant checks [e] and [g] failed on any host whose system
# interpreter lacks the requirements — including one with no `pip` or `ensurepip`
# at all, where they had been red long enough to read as permanent. Requiring
# every caller to export NY_GATE_PYTHON put a working gate behind a step nothing
# enforces. An explicit NY_GATE_PYTHON still wins, so CI and multi-venv setups
# are unaffected.
# A venv's interpreter is `bin/python` on Unix and `Scripts/python.exe` on
# Windows. Probing only the first meant this auto-detection — added precisely so
# the gate works without every caller exporting NY_GATE_PYTHON — could never fire
# under Git Bash/MSYS, and [e]/[g] stayed red on Windows with a correctly
# provisioned .venv sitting in the repo root.
# The `else python3` half of the documented default (see the Environment block
# above) was never implemented: with no `.venv` and no exported
# NY_GATE_PYTHON, this loop set NOTHING and the variable stayed unset. Under
# `set -u` the three lanes that read it then died at their FIRST line —
#
#   scripts/gate.sh: line 348: NY_GATE_PYTHON: unbound variable
#
# — checks [e], [f] and [g], each failing in 0s with a message about a shell
# variable rather than about Python. Worse than either alternative: it bypasses
# the importable-check below, whose error names the actual remedy ("Use the
# README virtualenv setup, install requirements.txt there"). Falling back to
# `python3` restores the documented contract AND routes a genuinely
# under-provisioned host into that message instead of a crash.
if [ -z "${NY_GATE_PYTHON:-}" ]; then
  for candidate in "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/.venv/Scripts/python.exe"; do
    if [ -x "$candidate" ]; then
      NY_GATE_PYTHON="$candidate"
      break
    fi
  done
fi
: "${NY_GATE_PYTHON:=python3}"

# Crates excluded from the Clippy portion of the Rust hygiene gate (check a),
# with reasons.
# Probed 2026-07-20 under exactly:
#   targo tippy --locked --workspace --exclude ny-python --all-targets -- -D warnings
# Re-probe with that command and prune this list as the drift is paid down;
# every entry here is un-linted debt, not a permanent carve-out.
# Empty since the 2026-07-20 env-wall migration completed: every crate's raw
# set_var/remove_var sites were routed through the blessed lock helpers
# (ny-test-utils env choke point; ny-mip ay_env), so the whole workspace runs
# under check [a]'s -D warnings. Add an entry ("crate|reason") ONLY for a
# crate with a known, dated, deliberately-unfixed warning — never to hide
# fresh drift.
CLIPPY_SKIPPED_CRATES=()

# ---------------------------------------------------------------------------
# Rust frontend resolution — REQUIRED, never a silent fallback.
#
# on a toolchain whose sysroot ships
# Trust-branded frontends ONLY: `targo-fmt` and `targo-tippy`. `cargo-fmt` and
# `cargo-clippy` are deliberately NOT installed there — they survive as private
# compat protocols, never as installed aliases. So `cargo fmt` and
# `cargo clippy` cannot run in this repository at all:
#
#   error: 'cargo-fmt' is not installed for the custom toolchain 'trust'.
#
# Measured 2026-08-27: check [a] failed in 0s on its FIRST command for exactly
# that reason, which meant the entire Rust hygiene lane — fmt, source policy,
# the lever ratchet, and both Clippy sweeps — never executed. It was not
# reporting drift; it was reporting that it could not look. Real fmt drift was
# sitting behind it (crates/ny-api and others).
#
# Resolve the pinned compiler through rustup explicitly. Calling ambient
# `rustc --print sysroot` is insufficient: `/opt/homebrew/bin` can precede the
# rustup proxies on PATH, in which case the command silently names stock Rust
# even though rust-toolchain.toml pins `trust`. That made every executing gate
# mode fail before its first check on a correctly linked Trust host.
NY_TOOLCHAIN_CHANNEL="$(awk -F'"' '/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"/ { print $2; exit }' rust-toolchain.toml)"
NY_RUSTUP="${NY_RUSTUP:-}"
if [ -z "$NY_RUSTUP" ]; then
  NY_RUSTUP="$(command -v rustup 2>/dev/null || true)"
fi
if [ -z "$NY_RUSTUP" ]; then
  for candidate in "${CARGO_HOME:-$HOME/.cargo}/bin/rustup" "${CARGO_HOME:-$HOME/.cargo}/bin/rustup.exe"; do
    if [ -x "$candidate" ]; then
      NY_RUSTUP="$candidate"
      break
    fi
  done
fi
NY_PINNED_RUSTC=""
if [ -n "$NY_RUSTUP" ] && [ -x "$NY_RUSTUP" ] && [ -n "$NY_TOOLCHAIN_CHANNEL" ]; then
  NY_PINNED_RUSTC="$("$NY_RUSTUP" which rustc --toolchain "$NY_TOOLCHAIN_CHANNEL" 2>/dev/null || true)"
fi
NY_RUST_SYSROOT=""
if [ -n "$NY_PINNED_RUSTC" ] && [ -x "$NY_PINNED_RUSTC" ]; then
  NY_RUST_SYSROOT="$("$NY_PINNED_RUSTC" --print sysroot 2>/dev/null || true)"
fi
TARGO=""
CARGO=""
if [ -n "$NY_RUST_SYSROOT" ]; then
  for candidate in "$NY_RUST_SYSROOT/bin/targo" "$NY_RUST_SYSROOT/bin/targo.exe"; do
    if [ -x "$candidate" ]; then
      TARGO="$candidate"
      break
    fi
  done
  for candidate in "$NY_RUST_SYSROOT/bin/cargo" "$NY_RUST_SYSROOT/bin/cargo.exe"; do
    if [ -x "$candidate" ]; then
      CARGO="$candidate"
      break
    fi
  done
fi
if [ -z "$TARGO" ] || [ -z "$CARGO" ]; then
  echo "gate.sh: cannot resolve the pinned Trust frontend." >&2
  echo "  rust-toolchain.toml pins channel = \"${NY_TOOLCHAIN_CHANNEL:-<unparsed>}\"; rustup must resolve a sysroot with targo and cargo frontends." >&2
  echo "  rustup: ${NY_RUSTUP:-<none>}" >&2
  echo "  pinned rustc: ${NY_PINNED_RUSTC:-<none>}" >&2
  echo "  resolved sysroot: ${NY_RUST_SYSROOT:-<none>}" >&2
  echo "  refusing to fall back to stock cargo — it would lint a different toolchain." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# GitHub credential precondition — REQUIRED, and deliberately NOT a lane.
#
# crates/ny-mip pulls `ay-milp` from a private GitHub repo, and the pinned rev
# is not in the local cargo git db, so every cargo command in this workspace
# needs a working github.com credential before it can resolve dependencies. An
# invalid GITHUB_TOKEN/GH_TOKEN in the environment overrides the working gh
# keyring credential and makes that fetch fail — measured 2026-08-28:
# `git ls-remote` on that URL returns rc=128 "Invalid username or token" with
# the token set, and rc=0 with it unset.
#
# On 2026-08-28 that cost a 41-MINUTE gate run which produced no test results
# at all: cargo died at dependency resolution, so ZERO tests ran and every
# assertion in every lane was VOID — while the scoreboard reported plain FAILs
# that read like a broken tree.
#
# WHY HERE AND NOT AS LANE [l]: lanes are skippable by --fast and --only, and
# this precondition governs every lane that shells out to cargo (a,b,c,d,f,h,
# i,j,k). A skippable check cannot protect `--only j`. Placing it here also
# preserves the ordering the two frontend tests in tests/test_gate_help.py
# assert: it runs strictly AFTER the frontend refusal above, so
# test_executing_gate_still_requires_the_trust_frontend still sees its own
# message and exit 2.
#
# Cost when healthy: 0.025-0.027s over 3 runs and ZERO network — with no token
# set the check returns on its fast path without probing. Only a token that IS
# set is probed (0.356-0.479s failing, 0.624-0.910s succeeding; 8s watchdog).
#
# Exit 3 from the checker means "authenticates: no" and is mapped to gate.sh's
# existing precondition exit code 2. Offline is a WARNING (exit 0) inside the
# checker, so a developer with no network still runs everything that does not
# need it.
if [ "${NY_SKIP_CREDENTIAL_CHECK:-0}" != 1 ]; then
  CREDENTIAL_CHECK="$REPO_ROOT/scripts/check_github_credential.sh"
  # Fail CLOSED on a missing checker rather than silently proceeding. Without
  # this, deleting the script turns the precondition into a no-op that still
  # "passes": bash exits 127, which is not 3, so the refusal below never fires.
  # That is the same shape as `cargo test <filter>` exiting 0 on a filter that
  # matched nothing — a check that cannot run must not read as a check that
  # passed.
  if [ ! -r "$CREDENTIAL_CHECK" ]; then
    echo "gate.sh: $CREDENTIAL_CHECK is missing or unreadable." >&2
    echo "  It is a tracked file; its absence is repository drift, not a local choice." >&2
    exit 2
  fi
  credential_rc=0
  bash "$CREDENTIAL_CHECK" --context gate || credential_rc=$?
  if [ "$credential_rc" -eq 3 ]; then
    echo "gate.sh: refusing to run — the GitHub credential above cannot fetch ay-milp." >&2
    echo "  Every cargo lane would die at dependency resolution and report VOID as FAIL." >&2
    echo "  Set NY_SKIP_CREDENTIAL_CHECK=1 to run anyway (it will not help)." >&2
    exit 2
  fi
fi

# h is a pure config-vs-code coherence check that finishes in seconds, so it
# belongs in --fast: the two gates that silently zeroed a banked category were
# both landed by someone who would have run the fast gate, not the full one.
#
# d, e, f and g joined it on 2026-08-19 for the same reason, with four fresh
# instances. A full-gate run that day found: fmt drift in two crates; a new
# Python test carrying a prohibited `pytest.mark.skipif` that silently deleted
# seven counterexample contracts; that same file missing from the migration
# manifest; `ny-falsify` a workspace member absent from docs/PACKAGES.md; and
# two claims-of-record docs still pinning an `ay-milp` revision Cargo.lock had
# moved off. Every one was caught by a check that ALREADY EXISTED, and every
# one reached main anyway — because the checks that catch them were not in the
# lane anybody can afford to run before a push.
#
# The cost of closing that hole is 126s: d(44) + e(82) + f(0) + g(0), taking
# --fast from ~95s to ~221s. What stays out is i(297s) and j(2669s) — j alone
# is 84% of the full gate's wall clock, and THAT is what makes the full gate
# unrunnable in an edit loop. Adding a cheap check here is not weakening the
# full gate; it is putting the cheap half where it will actually be run.
FAST_CHECKS="a b c d e f g h"

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ny-gate.XXXXXX")"
trap 'rm -rf "$LOG_DIR"' EXIT

RESULTS=()
FAILURES=0
PASSES=0
SKIPS=0

selected() {
  local letter="$1"
  if [ -n "$ONLY" ]; then
    [ "$letter" = "$ONLY" ]
  elif [ "$FAST" -eq 1 ]; then
    case " $FAST_CHECKS " in *" $letter "*) return 0 ;; *) return 1 ;; esac
  else
    return 0
  fi
}

run_check() {
  local letter="$1" name="$2" fn="$3"
  if ! selected "$letter"; then
    local why="--fast"
    [ -n "$ONLY" ] && why="--only $ONLY"
    RESULTS+=("SKIP  [$letter] $name  ($why)")
    SKIPS=$((SKIPS + 1))
    return 0
  fi
  local log="$LOG_DIR/check-$letter.log"
  echo ""
  echo "=== [$letter] $name ==="
  local start rc end
  start="$(date +%s)"
  set +e
  ( set -euo pipefail; "$fn" ) 2>&1 | tee "$log"
  rc=$?
  set -e
  end="$(date +%s)"
  if [ "$rc" -eq 0 ]; then
    RESULTS+=("PASS  [$letter] $name  ($((end - start))s)")
    PASSES=$((PASSES + 1))
    echo "--- [$letter] PASS ($((end - start))s) ---"
  else
    RESULTS+=("FAIL  [$letter] $name  ($((end - start))s)  log: $log")
    FAILURES=$((FAILURES + 1))
    echo "--- [$letter] FAIL ($((end - start))s) ---"
  fi
}

# ---------------------------------------------------------------------------
# a) Rust formatting + Clippy drift gate
# ---------------------------------------------------------------------------
check_a() {
  # `tippy`, not `clippy`: the pinned Trust sysroot installs `targo-tippy` and
  # keeps `cargo-clippy`/`clippy-driver` as private compat protocols that are
  # never installed aliases, so a `clippy` subcommand does not resolve here.
  local args=(tippy --locked --workspace --exclude ny-python --all-targets)
  local entry crate reason
  # ${arr[@]+...} guard: macOS bash 3.2 under `set -u` treats an EMPTY array
  # expansion as unbound (same idiom as the RESULTS loop below).
  for entry in ${CLIPPY_SKIPPED_CRATES[@]+"${CLIPPY_SKIPPED_CRATES[@]}"}; do
    crate="${entry%%|*}"
    reason="${entry#*|}"
    echo "SKIPPED crate $crate: $reason"
    args+=(--exclude "$crate")
  done
  echo "+ targo fmt --all -- --check"
  "$TARGO" fmt --all -- --check
  echo "+ cargo test --locked -j 1 -p ny-test-utils --test source_policy"
  "$CARGO" test --locked -j 1 -p ny-test-utils --test source_policy
  echo "+ cargo test --locked -j 1 -p ny-test-utils --test python_correctness_migration_manifest"
  "$CARGO" test --locked -j 1 -p ny-test-utils --test python_correctness_migration_manifest
  echo "+ cargo test --locked -j 1 -p ny-levers"
  "$CARGO" test --locked -j 1 -p ny-levers
  echo "+ targo ${args[*]} -- -D warnings"
  "$TARGO" "${args[@]}" -- -D warnings
  # The VNN-COMP executable is built with this exact non-default feature pair.
  # Workspace Clippy alone does not enable either feature and therefore cannot
  # catch integration drift in the scored MIP/CUDA surface.
  echo "+ targo tippy --locked -p ny-cli --bin ny --features mip,cuda -- -D warnings"
  "$TARGO" tippy --locked -p ny-cli --bin ny --features mip,cuda -- -D warnings
}

# ---------------------------------------------------------------------------
# b) ny-cert test suite
# ---------------------------------------------------------------------------
check_b() {
  echo "+ cargo test --locked -p ny-cert --all-targets"
  "$CARGO" test --locked -p ny-cert --all-targets
}

# ---------------------------------------------------------------------------
# c) ny CLI focused contracts: submission, cert, delivered gates, dependency
#    unity, layered flight receipts, and the measured-but-unshipped
#    alpha-zero-yield seam.
# ---------------------------------------------------------------------------
check_c() {
  echo "+ cargo test --locked -p ny-cli --bin ny vnncomp_submit"
  "$CARGO" test --locked -p ny-cli --bin ny vnncomp_submit
  echo "+ cargo test --locked -p ny-cli --bin ny cert"
  "$CARGO" test --locked -p ny-cli --bin ny cert
  echo "+ cargo test --locked -p ny-cli --bin ny alpha_zero_yield"
  "$CARGO" test --locked -p ny-cli --bin ny alpha_zero_yield
  echo "+ cargo test --locked -p ny-propagate --lib alpha_zero_yield"
  "$CARGO" test --locked -p ny-propagate --lib alpha_zero_yield
  echo "+ cargo test --locked -p ny-cli --test measured_gate_delivery"
  "$CARGO" test --locked -p ny-cli --test measured_gate_delivery
  echo "+ cargo test --locked -p ny-cli --test ay_pin_unity"
  "$CARGO" test --locked -p ny-cli --test ay_pin_unity
  echo "+ cargo test --locked -p ny-cli --test flight_lever_receipt"
  "$CARGO" test --locked -p ny-cli --test flight_lever_receipt
}

# ---------------------------------------------------------------------------
# d) ny-propagate unit slices
# ---------------------------------------------------------------------------
check_d() {
  echo "+ cargo test --locked -p ny-propagate --lib collection"
  "$CARGO" test --locked -p ny-propagate --lib collection
  # The conv2d slice contains scoped tests for the process-global
  # NY_DENSE_BUDGET_MB knob. Run this slice serially so a deliberate
  # zero-budget scope cannot alter an unrelated test's allocation policy.
  echo "+ cargo test --locked -p ny-propagate --lib conv2d -- --test-threads=1"
  "$CARGO" test --locked -p ny-propagate --lib conv2d -- --test-threads=1
}

# ---------------------------------------------------------------------------
# e) transitional top-level Python/shell harness pytest
# ---------------------------------------------------------------------------
check_e() {
  local gate_python="$NY_GATE_PYTHON"
  if ! env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$gate_python" -s -c 'import numpy, onnx, onnxruntime, pytest, sys, yaml; __import__("tomllib" if sys.version_info >= (3, 11) else "tomli")' \
    >/dev/null 2>&1; then
    echo "ERROR: Python tooling dependencies are not importable in isolated mode by $gate_python." >&2
    echo "Use the README virtualenv setup, install requirements.txt there, and set NY_GATE_PYTHON to that interpreter." >&2
    return 1
  fi
  echo "+ env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 $gate_python -s -m pytest tests -q"
  (
    unset NY_GATE_PYTHON
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
      "$gate_python" -s -m pytest tests -q
  )
}

# ---------------------------------------------------------------------------
# f) submission-packaging invariants (the pack-time soundness-of-packaging
#    gate). Regenerates a real tarball to a temp path and asserts, on the
#    archived bytes:
#      f1  0 crates/*/proptest-regressions and 0 crates/*/corpus entries
#          (the EXCLUDE_GLOBS actually applied, and stayed crates/*-scoped)
#      f2  no internal AY/NY/Trust/Clean source copy is tracked or packaged. NY
#          consumes AY from its exact Git revision; NY-owned code and unrelated
#          third-party vendor content remain permitted.
#      f3  vendor-manifest scan: every file listed in every packaged
#          vendor/**/.cargo-checksum.json is present in the archive (a
#          missing one aborts cargo's offline directory-source build).
# ---------------------------------------------------------------------------
check_f() {
  local target_dir bin
  target_dir="$("$CARGO" metadata --locked --format-version 1 --no-deps | "$NY_GATE_PYTHON" -c \
    'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
  bin="$target_dir/debug/ny"
  if [ ! -x "$bin" ]; then
    echo "debug ny binary absent — building it"
    echo "+ cargo build --locked -p ny-cli --bin ny"
    "$CARGO" build --locked -p ny-cli --bin ny
  else
    echo "using existing debug binary: $bin"
  fi

  local tmpdir tarball listing
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/ny-gate-sub.XXXXXX")"
  # Each check runs in its own subshell (see run_check), so this EXIT trap is
  # scoped to check_f and fires on error exits as well as normal returns.
  # Expand tmpdir now, not at trap time.
  # shellcheck disable=SC2064
  trap "rm -rf '$tmpdir'" EXIT
  tarball="$tmpdir/gate-submission.tar.gz"
  echo "+ $bin vnncomp-submit --no-build --output $tarball"
  "$bin" vnncomp-submit --no-build --output "$tarball"

  listing="$tmpdir/listing.txt"
  tar -tzf "$tarball" | sed -e 's|^\./||' -e 's|/$||' >"$listing"
  echo "archive entries: $(wc -l <"$listing" | tr -d ' ')"

  local ok=0

  # f1: dev-data excludes actually applied.
  local n_pt n_corpus
  n_pt="$(grep -cE '^crates/[^/]+/proptest-regressions(/|$)' "$listing" || true)"
  n_corpus="$(grep -cE '^crates/[^/]+/corpus(/|$)' "$listing" || true)"
  if [ "$n_pt" -eq 0 ] && [ "$n_corpus" -eq 0 ]; then
    echo "  ok    f1: 0 crates/*/proptest-regressions, 0 crates/*/corpus entries"
  else
    echo "  FAIL  f1: crates/*/proptest-regressions entries: $n_pt, crates/*/corpus entries: $n_corpus (want 0/0)"
    ok=1
  fi

  # f2: first-party repositories must never be copied into NY. This includes
  # Clean, the legacy crates/trust-spec mirror, and the generic vendor/
  # build_support path that held unused byte-identical AY helpers. Exact path
  # boundaries leave nested helpers in legitimate third-party crates and
  # NY-owned shared corpus paths alone.
  local internal_re='^(vendor/(ay|ny|trust|clean|build_support)|crates/(ay|ny|trust|clean|trust-spec)|crates/ny-cert/proofs/lean/(Crownproof|\.lake/packages/crownproof))(/|$)'
  local n_internal tracked_internal
  n_internal="$(grep -cE "$internal_re" "$listing" || true)"
  tracked_internal="$(git ls-files | while IFS= read -r path; do
    if [ -e "$path" ] || [ -L "$path" ]; then
      printf '%s\n' "$path"
    fi
  done | grep -E "$internal_re" || true)"
  if [ "$n_internal" -eq 0 ] && [ -z "$tracked_internal" ]; then
    echo "  ok    f2: no tracked or packaged AY/NY/Trust/Clean source copy"
  else
    echo "  FAIL  f2: internal repository source must not be copied into NY (archive entries: $n_internal)"
    if [ -n "$tracked_internal" ]; then
      echo "        tracked internal-copy paths remain:"
      printf '%s\n' "$tracked_internal" | sed 's/^/          /' | head -20
    fi
    ok=1
  fi

  # f3: vendor-manifest scan on the archived bytes.
  if "$NY_GATE_PYTHON" - "$tarball" <<'PY'
import json
import sys
import tarfile

path = sys.argv[1]
with tarfile.open(path, "r:gz") as tar:
    members = tar.getmembers()
    def norm(name):
        return name[2:].rstrip("/") if name.startswith("./") else name.rstrip("/")
    names = {norm(m.name) for m in members}
    manifests = [
        m for m in members
        if norm(m.name).startswith("vendor/")
        and norm(m.name).endswith("/.cargo-checksum.json")
    ]
    missing = []
    for member in manifests:
        crate_dir = norm(member.name)[: -len("/.cargo-checksum.json")]
        data = json.load(tar.extractfile(member))
        for rel in data["files"]:
            if f"{crate_dir}/{rel}" not in names:
                missing.append(f"{crate_dir}/{rel}")
    print(f"  vendor-manifest scan: {len(manifests)} .cargo-checksum.json manifest(s) in archive")
    if missing:
        print(f"  FAIL  f3: archive drops {len(missing)} checksummed vendored file(s) — offline build would abort:")
        for entry in sorted(missing)[:20]:
            print(f"          {entry}")
        sys.exit(1)
PY
  then
    echo "  ok    f3: every vendor/**/.cargo-checksum.json entry present in the archive"
  else
    ok=1
  fi

  return "$ok"
}

# ---------------------------------------------------------------------------
# g) Python package/tooling coherence
# ---------------------------------------------------------------------------
check_g() {
  local gate_python="$NY_GATE_PYTHON"
  if ! env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$gate_python" -s -c 'import numpy, onnx, onnxruntime, pytest, sys, yaml; __import__("tomllib" if sys.version_info >= (3, 11) else "tomli")' \
    >/dev/null 2>&1; then
    echo "ERROR: Python tooling dependencies are not importable in isolated mode by $gate_python." >&2
    echo "Use the README virtualenv setup, install requirements.txt there, and set NY_GATE_PYTHON to that interpreter." >&2
    return 1
  fi
  echo "+ (cd crates/ny-python && env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 $gate_python -s -m pytest -p ny_pytest.plugin ny_pytest/tests -q)"
  (
    unset NY_GATE_PYTHON
    cd crates/ny-python
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
      "$gate_python" -s -m pytest -p ny_pytest.plugin ny_pytest/tests -q
  )
  echo "+ NY_TEST_PYTHON=$gate_python cargo test --locked -p ny-test-utils --features python-tools --tests"
  (
    unset NY_GATE_PYTHON
    NY_TEST_PYTHON="$gate_python" \
      "$CARGO" test --locked -p ny-test-utils --features python-tools --tests
  )
}

# ---------------------------------------------------------------------------
# h) STATIC preset capability guard: declared device honoured or waived
# ---------------------------------------------------------------------------
check_h() {
  echo "+ cargo test --locked -p ny-cli --bin ny preset::backend_capability_tests"
  "$CARGO" test --locked -p ny-cli --bin ny preset::backend_capability_tests
}

# ---------------------------------------------------------------------------
# i) STATIC preset model-load smoke: every shipped preset's models still load
#
#    --release on purpose: this decodes ~1 GB of ONNX protobuf, which is minutes
#    in a debug build and seconds in release.
#
#    This check FAILS when the benchmark repositories are absent. That is
#    deliberate: it used to pass vacuously, and a vacuous pass is
#    indistinguishable from "everything loads" in a gate log. Do not "fix" a
#    red here by exporting NY_PRESET_LOAD_SMOKE=off in this script — that
#    reintroduces the exact hole. Make the data reachable instead.
# ---------------------------------------------------------------------------
check_i() {
  echo "+ cargo test --locked --release -p ny-cli --features external-vnncomp --bin ny preset::model_load_smoke_tests -- --nocapture"
  "$CARGO" test --locked --release -p ny-cli --features external-vnncomp --bin ny preset::model_load_smoke_tests -- --nocapture
}

# ---------------------------------------------------------------------------
# j) ny-propagate SOUNDNESS ORACLES (--release)
#
#    Check [d] runs the collection and conv2d slices only, so until this lane
#    existed the gate BUILT the moat's own oracles on every full run and never
#    executed one of them. These are the tests that catch a false-UNSAT: the
#    RootEval.dj adversarial enclosure oracles assert the certified per-class
#    lower bound `dj[k]` never exceeds the true feasible margin `Y_t(x)-Y_j(x)`
#    at any sampled point, over hundreds of random and near-cancellation
#    twin-nets; wide_alpha_true and interm_refine cover the batched wide-alpha
#    and intermediate-refinement seams that feed it.
#
#    --release on purpose, twice over: a debug ny-propagate test binary dies on
#    this host inside gemm-common (pulled in by faer) on fullfp16 instructions
#    the target rejects, and the oracles' sampling budgets are minutes of
#    arithmetic at debug speed.
#
#    ADDED COST: ~4 min. The 172 selected tests run serially (RUST_TEST_THREADS
#    is 1 in .cargo/config.toml) on top of the release ny-propagate lib test
#    binary, which is a relink on a warm target directory and several minutes on
#    a cold one. That is why this stays out of --fast, which is meant to finish
#    in seconds.
#
#    Each filter carries a MINIMUM test count. `cargo test <filter>` exits 0
#    when the filter matches NOTHING, so a rename or a module move would
#    otherwise leave this lane green while running zero oracles — the same
#    vacuous green check [i] refuses for absent benchmark data. Lower a floor
#    only alongside the deletion that justifies it, never to clear a red.
#
#    OBSERVED COUNTS (the floors below are deliberately lower; these are what
#    the tree actually has, so a reader can tell growth from breakage):
#      2026-08-16, default features:  54 / 16 / 102
#      2026-08-28, default features:  98 / 16 / 104   <- first execution of this
#        lane on aarch64-unknown-linux-gnu. The enclosure slice nearly DOUBLED
#        (54 -> 98) because oracles were added, which is the good direction; the
#        floors 50 / 14 / 95 are still met with room. Recorded because a reader
#        diffing 98 against a lone stale "54" would think something moved.
#
#    WHAT THIS LANE COST, since the header above says "~4 min": measured 2561 s
#    end to end, of which ~2537 s was the COLD release build of the test binary
#    (`-C lto -C codegen-units=1` over a ~187 MB rlib) and 24.1 s was the oracles
#    themselves. Slices 2 and 3 are instant because they reuse slice 1's binary.
#
#    WHAT THE ORACLES REPORTED on that first run: ~45.2M sampled (class, point)
#    enclosure checks, ZERO overshoots — the certified per-class lower bound
#    never exceeded the true feasible margin in any adversarial regime the
#    oracles construct. Every `worst_over` in a verdict-bearing mode was
#    NEGATIVE, i.e. strictly inside the true margin.
# ---------------------------------------------------------------------------
run_propagate_oracle_slice() {
  local label="$1" floor="$2" filter="$3"
  local slice_log="$LOG_DIR/check-j-$label.log" ran
  echo "+ cargo test --locked --release -p ny-propagate --lib $filter"
  # `if !` so the pipeline's failure is handled here instead of aborting the
  # check subshell: every slice reports, the way check_f's f1/f2/f3 do.
  if ! "$CARGO" test --locked --release -p ny-propagate --lib "$filter" 2>&1 | tee "$slice_log"; then
    echo "  FAIL  j-$label: cargo test failed for filter '$filter'"
    return 1
  fi
  ran="$(awk '/^test result: ok\. [0-9]+ passed/ { print $4; exit }' "$slice_log")"
  if [ -z "$ran" ] || [ "$ran" -lt "$floor" ]; then
    echo "  FAIL  j-$label: filter '$filter' ran ${ran:-0} test(s), floor is $floor — the filter no longer names those oracles, so this lane proved nothing"
    return 1
  fi
  echo "  ok    j-$label: $ran test(s) passed via '$filter' (floor $floor)"
  return 0
}

check_j() {
  local ok=0
  run_propagate_oracle_slice enclosure 50 margin_row::tests:: || ok=1
  # The three floors above are on each FILTER'S TOTAL. `margin_row::tests::`
  # matches 98 tests of which only ~13 are the dj/box enclosure oracles, so
  # deleting an oracle could be masked by the ~85 siblings and the floor of 50
  # would still be met. These two slices pin the oracle subsets DIRECTLY. They
  # ADD a constraint — they do not replace or lower the floor of 50 above.
  #
  # Floors are MEASURED, not chosen: `-- --list` on 2026-08-28 reports exactly 3
  # and 6. They are set AT the observed count rather than below it because these
  # are the load-bearing oracles: any deletion must be a deliberate act that
  # lowers the floor in the same commit, which is the same rule the header states
  # for the totals.
  run_propagate_oracle_slice enclosure-oracle-mods 3 margin_row::tests::enclosure_oracle_ || ok=1
  run_propagate_oracle_slice enclosure-oracle-fns 6 margin_row::tests::oracle_ || ok=1
  run_propagate_oracle_slice wide-alpha-true 14 wide_alpha_true::tests:: || ok=1
  run_propagate_oracle_slice interm-refine 95 interm_refine:: || ok=1
  return "$ok"
}

MODE="full"
[ "$FAST" -eq 1 ] && MODE="fast (--fast)"
[ -n "$ONLY" ] && MODE="single check (--only $ONLY)"
echo "ny drift gate — mode: $MODE — repo: $REPO_ROOT"

# ---------------------------------------------------------------------------
# k) ny-api suite — the integration tests that were SILENTLY SKIPPED
# ---------------------------------------------------------------------------
#    Every [[test]] target in crates/ny-api carries `required-features`, and
#    cargo skips an unmet test target with NO warning and exit 0. Before
#    `default = ["cert"]` landed, `cargo test -p ny-api` compiled the lib, ran
#    26 unfeatured unit tests, skipped all 8 integration binaries (49 tests),
#    and reported success — "matched nothing reads as pass", shipped as build
#    configuration.
#
#    `--features cert` is passed EXPLICITLY so this lane does not depend on the
#    default key staying put: if someone drops or narrows `default`, the lane
#    keeps running the suite instead of silently going hollow again.
#
#    Counts RE-DERIVED on the merged tree 2026-08-27 (the pre-merge figures were
#    stale: upstream added 4 test targets while this lane was being written).
#    12 [[test]] targets, 76 #[test] fns under tests/; `cert` arms 65 of them.
#    gpu_facade_contract's 11 need --features gpu and a real device.
#
#    The floor and the BY-NAME assertion exist for the same reason as lane [j]:
#    a count alone cannot tell "the suite ran" from "only the 26 old unfeatured
#    unit tests ran". Counted 2026-08-27. Lower a floor only alongside the
#    deletion that justifies it, never to clear a red.
check_k() {
  # DISTINCT from run_check's own "$LOG_DIR/check-k.log": run_check tees this
  # function's output to that exact path while it runs, so grepping it here
  # races the outer tee's truncation and the by-name assertions read an empty
  # or partial file. Lane [j] uses check-j-$label.log for the same reason.
  local log="$LOG_DIR/check-k-api.log" floor=55 ran rc
  # --all-targets mirrors lane [b] and is load-bearing, not cosmetic: it runs
  # lib + integration targets but EXCLUDES doctests. Without it cargo exits 1 on
  # `rustdoc is not installed for the custom toolchain 'trust'` even when every
  # test passed, which would make this lane permanently red on this host for a
  # reason that has nothing to do with ny-api.
  echo "+ cargo test --locked -p ny-api --features cert --all-targets"
  # "$CARGO", not bare `cargo`: gate.sh resolves $CARGO from the sysroot that
  # rust-toolchain.toml pins (see :273-291). A bare `cargo` resolves from PATH
  # and can be a DIFFERENT toolchain than every other lane runs against, which
  # would make this lane's result incomparable with theirs.
  "$CARGO" test --locked -p ny-api --features cert --all-targets 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}

  # Branch on the ok/FAILED token, never on the count alone: `ok. 0 passed` is
  # VOID (matched nothing) while `FAILED. 0 passed; N failed` is a real failure.
  if grep -qE '^test result: FAILED' "$log"; then
    echo "  FAIL  k: ny-api reported a real test failure"
    return 1
  fi
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL  k: cargo test -p ny-api exited $rc"
    return 1
  fi

  # Count INTEGRATION tests only. Summing every `ok. N passed` line made this
  # floor HOLLOW: ny-api's lib carries 200 #[test] fns, so the lib block alone
  # cleared a floor of 65 with every integration binary dark — precisely the
  # condition this lane exists to detect. awk tracks which binary is reporting
  # and ignores the `Running unittests src/lib.rs` block.
  ran=$(awk '
    /^[[:space:]]+Running unittests/ { in_int = 0; next }
    /^[[:space:]]+Running tests\// { in_int = 1; next }
    in_int && /^test result: ok\. [0-9]+ passed/ { for (i = 1; i <= NF; i++) if ($i == "ok.") { n += $(i+1); break } }
    END { print n + 0 }
  ' "$log")
  # A zero count is VOID, not pass: it means no integration binary reported at
  # all. Rejected explicitly rather than relying on the floor, so that lowering
  # the floor some day cannot silently re-admit a hollow run.
  if [ "$ran" -eq 0 ]; then
    echo "  FAIL  k: ny-api ran ZERO integration tests — VOID, not a pass"
    echo "        (the lib block is excluded from this count by design)"
    return 1
  fi
  if [ "$ran" -lt "$floor" ]; then
    echo "  FAIL  k: ny-api ran $ran tests, below the floor of $floor"
    echo "        A count under the floor means test targets were SKIPPED for"
    echo "        unmet required-features — the exact defect this lane guards."
    return 1
  fi

  # BY NAME: one test from each feature tier, so "the targets built but the
  # cert tier went dark" fails closed instead of passing on propagate alone.
  local missing=0 t
  for t in lipschitz_sound facade_expansion_contract; do
    if ! grep -qE "^\s+Running tests/$t\.rs" "$log"; then
      echo "  FAIL  k: test binary '$t' never ran (required-features unmet?)"
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] || return 1

  echo "  ok    k: ny-api ran $ran INTEGRATION tests (floor $floor), cert-tier binaries present"
  echo "        NOTE: gpu_facade_contract (11 tests) needs --features gpu and a"
  echo "              real device; it is NOT covered here. Known, named gap."
}

# ---------------------------------------------------------------------------
# l) activation-math certified-bound pins (SiLU: ny-propagate + ny-relaxation)
# ---------------------------------------------------------------------------
#    `silu_critical_point` (crates/ny-propagate/src/layers/activations/silu/
#    math.rs) finds SiLU's argmin by NEWTON'S METHOD IN f32, using a CENTRAL
#    DIFFERENCE for the second derivative at eps = 1.0e-5_f32. `silu_min_max`
#    then evaluates SiLU there and applies `next_down_f32` — ONE f32 ULP of
#    backoff, the entire safety margin — to publish a CERTIFIED LOWER BOUND.
#    The bound is therefore sound only if Newton converged, and nothing
#    asserted that until the `newton_pin` modules landed. Those pins are
#    zero-tolerance and compare against an mpmath-derived minimum (dps 60), so
#    unlike the two assertions they replaced they cannot drift with the code
#    they check.
#
#    Before this lane, NO SiLU test ran in the gate at all. Census of every
#    `cargo test` in this file: [b] is -p ny-cert, [c] -p ny-cli, [d] is
#    -p ny-propagate --lib filtered to `collection` and `conv2d`, [j] is
#    -p ny-propagate --lib filtered to `margin_row::tests::`,
#    `wide_alpha_true::tests::` and `interm_refine::`, [k] is -p ny-api. None
#    of them reaches `silu::`. `ny-relaxation` appeared ZERO times in this
#    script, so the mirror crate that exists precisely to detect drift against
#    ny-propagate (tests/drift.rs) was itself never run. This lane closes both
#    gaps, which is why it runs the whole mirror crate and not just its pins.
#
#    --all-targets on ny-relaxation for the same reason as [b] and [k]: it runs
#    lib + tests/drift.rs + tests/soundness.rs but EXCLUDES doctests, which the
#    pinned `trust` toolchain has no rustdoc to run. Without it this lane would
#    be permanently red for a reason that has nothing to do with SiLU.
#
#    Debug, not --release: [d] already runs debug `-p ny-propagate --lib`
#    slices inside --fast, so that binary is known to work on this host, and
#    these pins are scalar arithmetic with no gemm on the path. The silu slice
#    also contains one `#[cfg(debug_assertions)]` test, which a --release run
#    would silently drop below the floor.
#
#    NOT in FAST_CHECKS: this lane's wall clock has never been measured — it
#    was authored without box access. Move it into FAST_CHECKS once somebody
#    has timed it, not before; an unmeasured cost is how --fast stops being
#    fast.
#
#    The floors are STATIC counts, derived 2026-08-28 by enumerating `#[test]`
#    in the sources rather than by observing a run:
#      ny-propagate `silu::` — 27 existing (24 top-level in silu/tests.rs plus
#        3 inside its single `proptest!` block; one of the 24 is the
#        `#[cfg(debug_assertions)]` chord-precondition case) + 4 new pins = 31.
#      ny-relaxation --all-targets — 10 lib unit tests (exp 3, log 3,
#        softmax 2, rounding 1, safe_math 1) + drift.rs 11 + soundness.rs 11
#        + 4 new pins = 36.
#    The first execution of this lane is what confirms them. A floor that is
#    too HIGH fails closed and its FAIL line prints the real number to correct
#    it with; a floor is only ever LOWERED alongside the deletion that
#    justifies it, never to clear a red.
run_activation_pin_slice() {
  local label="$1" floor="$2" slice_log="$3"
  shift 3
  local ran rc=0
  echo "+ cargo $*"
  # `if !` rather than a bare pipeline: run_check runs this inside
  # `set -euo pipefail`, where a failing pipeline aborts the subshell before
  # any of the diagnostics below can print. Lane [j] uses the same form.
  if ! "$CARGO" "$@" 2>&1 | tee "$slice_log"; then
    rc=1
  fi

  # Branch on the ok/FAILED token FIRST and never on the count alone:
  # `test result: ok. 0 passed` is VOID (the selection matched nothing) while
  # `test result: FAILED. 0 passed; N failed` is a real failure. A detector in
  # this campaign inverted exactly this and filed a genuine red test as a
  # tooling artifact.
  if grep -qE '^test result: FAILED' "$slice_log"; then
    echo "  FAIL  l-$label: a test in this slice actually FAILED"
    return 1
  fi
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL  l-$label: cargo exited non-zero with no FAILED test result"
    return 1
  fi

  # One awk rather than `grep -oE | grep -oE | awk`: that pipeline exits
  # non-zero when nothing matches, and under `pipefail` the command
  # substitution would abort the subshell before the floor message could
  # print. awk sums across every binary --all-targets ran and always exits 0.
  ran="$(awk '/^test result: ok\. [0-9]+ passed/ { n += $4 } END { print n + 0 }' "$slice_log")"
  if [ "$ran" -lt "$floor" ]; then
    echo "  FAIL  l-$label: ran $ran test(s), floor is $floor"
    echo "        Under the floor means the filter or target selection no"
    echo "        longer reaches these pins, so this lane proved NOTHING."
    return 1
  fi
  echo "  ok    l-$label: $ran test(s) passed (floor $floor)"
  return 0
}

check_l() {
  # Own log paths, for lane [k]'s reason: run_check tees this function's output
  # to "$LOG_DIR/check-l.log" WHILE it runs, so grepping that path here races
  # the outer tee's truncation and the by-name assertions read a partial file.
  local prop_log="$LOG_DIR/check-l-silu-propagate.log"
  local relax_log="$LOG_DIR/check-l-silu-relaxation.log"
  local ok=0

  run_activation_pin_slice propagate 31 "$prop_log" \
    test --locked -p ny-propagate --lib silu:: || ok=1
  run_activation_pin_slice relaxation 36 "$relax_log" \
    test --locked -p ny-relaxation --all-targets || ok=1

  # BY NAME, ANCHORED on the full module path and on the trailing ` ... ok`.
  # A bare `silu` substring also matches the silu_chord/silu_eval tests, which
  # say nothing about the Newton premise, and an unanchored name is how "elu"
  # matched "celu" earlier in this campaign. All four pins must have RUN and
  # PASSED in BOTH crates: ny-relaxation mirrors ny-propagate's math, so a pin
  # that exists in only one of them is itself the drift this asserts against.
  local missing=0 t
  for t in certified_minimum_never_exceeds_true_silu_minimum \
    newton_lands_inside_the_certified_robustness_radius \
    no_f32_lies_between_the_critical_point_and_the_true_argmin \
    critical_point_is_the_nearest_f32_to_the_true_argmin; do
    if ! grep -qE "^test layers::activations::silu::math::newton_pin::$t \.\.\. ok\$" \
      "$prop_log"; then
      echo "  FAIL  l: ny-propagate pin '$t' did not run and pass"
      missing=1
    fi
    if ! grep -qE "^test silu::math::newton_pin::$t \.\.\. ok\$" "$relax_log"; then
      echo "  FAIL  l: ny-relaxation mirror pin '$t' did not run and pass"
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] || ok=1

  if [ "$ok" -eq 0 ]; then
    echo "  ok    l: all 4 certified-bound pins ran and passed in both crates"
  fi
  return "$ok"
}

# ---------------------------------------------------------------------------
# m) ny-api GPU-facade tier — the 11 tests lane [k] can only NAME, not run
# ---------------------------------------------------------------------------
#    gpu_facade_contract.rs is gated `required-features = ["gpu"]`, and the
#    crate default is ["cert"], so lane [k] cannot reach it. The default is
#    correct and must not change: `gpu = ["propagate", "dep:ny-gpu",
#    "ny-gpu/wgpu"]` would put a real ny-gpu + wgpu edge on every plain
#    `cargo build -p ny-api`.
#
#    But the tests DO NOT need a device. The file's own header (lines 21-22)
#    states it "runs identically on a host with no adapter"; every test names
#    types and checks constructor signatures, and the live ladder is a
#    SEPARATE #[ignore]d unit test in ny_api::gpu. So the gap lane [k] names is
#    closable here at the cost of one extra compile, not a hardware
#    requirement. Before this lane those 11 tests had never run in the gate.
#
#    Same discipline as [k]: branch on the ok/FAILED token before any count,
#    count INTEGRATION binaries only (the lib block would swamp any floor),
#    reject a zero count as VOID, assert the gpu binary BY NAME so "it built
#    but the gpu tier went dark" fails closed, and use a distinct log filename
#    because run_check tees to check-m.log while this function reads it.
check_m() {
  local log="$LOG_DIR/check-m-api-gpu.log" floor=11 ran rc
  echo "+ cargo test --locked -p ny-api --features cert,gpu --all-targets"
  # "$CARGO" for the same reason as lane [k]: the pinned sysroot frontend, not
  # whatever `cargo` PATH happens to resolve.
  "$CARGO" test --locked -p ny-api --features cert,gpu --all-targets 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}

  if grep -qE '^test result: FAILED' "$log"; then
    echo "  FAIL  m: ny-api reported a real test failure under cert,gpu"
    return 1
  fi
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL  m: cargo test -p ny-api --features cert,gpu exited $rc"
    return 1
  fi

  # gpu_facade_contract only. A total across binaries would let the other tiers
  # cover a dark gpu binary, which is the exact failure this lane exists for.
  ran=$(awk '
    /^[[:space:]]+Running tests\/gpu_facade_contract\.rs/ { seen = 1; next }
    seen && /^test result: ok\. [0-9]+ passed/ { for (i = 1; i <= NF; i++) if ($i == "ok.") { print $(i+1); exit } }
  ' "$log")
  ran=${ran:-0}

  if [ "$ran" -eq 0 ]; then
    echo "  FAIL  m: gpu_facade_contract ran ZERO tests — VOID, not a pass"
    echo "        (either the binary never ran, or required-features went unmet)"
    return 1
  fi
  if [ "$ran" -lt "$floor" ]; then
    echo "  FAIL  m: gpu_facade_contract ran $ran tests, below the floor of $floor"
    return 1
  fi
  echo "  ok    m: gpu_facade_contract ran $ran tests (floor $floor)"
}

# ---------------------------------------------------------------------------
# n) score-at-risk drift — the yaml that prices a model-load refusal
# ---------------------------------------------------------------------------
#    configs/preset_score_at_risk.yaml told the model-load guard that vit_2023
#    was worth 16.8 normalized points for SIXTEEN DAYS after a proof retraction
#    took its bank to zero solved rows. Nothing noticed: the only consumer
#    (crates/ny-cli/src/preset/model_load_smoke_tests.rs) asserts merely
#    0.0 <= banked_normalized <= 100.0, which 16.8 satisfied throughout.
#
#    scripts/check_score_at_risk_drift.py closes that. Every check it runs is
#    DENOMINATOR-FREE, so it needs no external_tools/vnncomp2025_results
#    checkout (gitignored, absent here) — recomputing a member from the bank
#    would need the winner denominators, but CHECKING these invariants does not:
#      1. census anchored to the SCORER's own REGULAR/EXTENDED lists, not to the
#         yaml's own rows — comparing the yaml against itself cannot detect an
#         omission;
#      2. each total equals the sum of its members, de-duplicated (the file
#         carries 31 roots for 25 scored benchmarks because cifar100/safenlp
#         appear under alternate years, so a naive sum over-counts by 136.0);
#      3. any SCORED benchmark whose bank has zero sat/unsat rows must be priced
#         0.0, since raw=0 forces normalized 0.0 under both branches of the
#         scorer's meas_norm. This is the vit_2023 failure exactly, and it is
#         verified to fire on that scenario alone with the totals made
#         self-consistent.
#    It exits 2 (VOID) rather than 0 when it cannot check — no bank files found,
#    or a bank file with no recognisable verdict column.
check_n() {
  local log="$LOG_DIR/check-n-score-drift.log" rc
  echo "+ python3 scripts/check_score_at_risk_drift.py"
  "$NY_GATE_PYTHON" scripts/check_score_at_risk_drift.py 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  case "$rc" in
    0) : ;;
    2) echo "  FAIL  n: drift check could not run (VOID) — it verified NOTHING"; return 1 ;;
    *) echo "  FAIL  n: score-at-risk drift detected (see above)"; return 1 ;;
  esac
  grep -q '^PASS:' "$log" || {
    echo "  FAIL  n: exit 0 without a PASS line — treating as VOID"; return 1; }
  echo "  ok    n: score-at-risk yaml agrees with the bank and the scorer's lists"
}

run_check a "Rust hygiene gate (fmt + source policy + lever ratchet + workspace/scored-tier Clippy)" check_a
run_check b "ny-cert test suite (--all-targets)" check_b
run_check c "focused submission, cert, lever, and dependency contracts" check_c
run_check d "ny-propagate unit slices (collection + conv2d)" check_d
run_check e "transitional Python/shell harness pytest (no skips permitted)" check_e
run_check f "submission-packaging invariants" check_f
run_check g "Python package/version coherence" check_g
run_check h "preset capability guard (declared device honoured or waived)" check_h
run_check i "preset model-load smoke (every shipped preset's models load)" check_i
run_check j "ny-propagate soundness oracles (RootEval.dj enclosure, wide-alpha-true, interm-refine)" check_j
run_check k "ny-api suite (integration tests formerly skipped for unmet required-features)" check_k
run_check l "activation-math certified-bound pins (ny-propagate silu + ny-relaxation mirror)" check_l
run_check m "ny-api GPU-facade tier (11 tests lane [k] cannot reach)" check_m
run_check n "score-at-risk drift (yaml vs bank vs scorer lists)" check_n

echo ""
echo "==================== GATE SCOREBOARD ===================="
for line in ${RESULTS[@]+"${RESULTS[@]}"}; do
  echo "$line"
done
echo "---------------------------------------------------------"
if [ "$FAILURES" -gt 0 ]; then
  echo "GATE: FAIL ($PASSES pass, $FAILURES fail, $SKIPS skipped)"
  exit 1
fi
echo "GATE: PASS ($PASSES pass, $FAILURES fail, $SKIPS skipped)"
