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
#   a  Rust hygiene gate   cargo fmt --all -- --check, Cargo-owned source policy,
#                          ny-levers declarations/direct-literal ratchet, and
#                          migration inventory (no Python interpreter), then
#                          cargo clippy --locked --workspace --exclude ny-python
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
#
# NOT in this gate (needs verification runs, minutes to hours):
#   scripts/check_banked_rows.py --bin <release ny> — replays a per-category
#   sample of banked SOLVED ledger rows through `ny vnncomp` and names the CAUSE
#   of each non-reproduction. Run it before banking and after any fail-closed
#   gate lands.
#
# Flags:
#   --fast        skip the heavy suites and packaging (runs a, b, c, h only)
#   --only <x>    run a single check by letter (a-j)
#   -h | --help   usage
# Environment:
#   NY_GATE_PYTHON  Python interpreter provisioned from requirements.txt
#                   (default: ./.venv/bin/python or ./.venv/Scripts/python.exe
#                   when present, else python3)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
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
if [ -z "${NY_GATE_PYTHON:-}" ]; then
  for candidate in "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/.venv/Scripts/python.exe"; do
    if [ -x "$candidate" ]; then
      NY_GATE_PYTHON="$candidate"
      break
    fi
  done
fi

# Crates excluded from the Clippy portion of the Rust hygiene gate (check a),
# with reasons.
# Probed 2026-07-20 under exactly:
#   cargo clippy --locked --workspace --exclude ny-python --all-targets -- -D warnings
# Re-probe with that command and prune this list as the drift is paid down;
# every entry here is un-linted debt, not a permanent carve-out.
# Empty since the 2026-07-20 env-wall migration completed: every crate's raw
# set_var/remove_var sites were routed through the blessed lock helpers
# (ny-test-utils env choke point; ny-mip ay_env), so the whole workspace runs
# under check [a]'s -D warnings. Add an entry ("crate|reason") ONLY for a
# crate with a known, dated, deliberately-unfixed warning — never to hide
# fresh drift.
CLIPPY_SKIPPED_CRATES=()

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

usage() {
  # Print the header block: every line from line 2 up to the first non-comment
  # line. The hardcoded end line this replaced was already one short of the
  # header and silently truncated the help text every time a check was added.
  awk 'NR >= 2 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

FAST=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --fast) FAST=1 ;;
    --only)
      shift
      ONLY="${1:-}"
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
    a|b|c|d|e|f|g|h|i|j) ;;
    *)
      echo "gate.sh: --only takes a single check letter a-j, got: '$ONLY'" >&2
      exit 2
      ;;
  esac
fi

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
  local args=(clippy --locked --workspace --exclude ny-python --all-targets)
  local entry crate reason
  # ${arr[@]+...} guard: macOS bash 3.2 under `set -u` treats an EMPTY array
  # expansion as unbound (same idiom as the RESULTS loop below).
  for entry in ${CLIPPY_SKIPPED_CRATES[@]+"${CLIPPY_SKIPPED_CRATES[@]}"}; do
    crate="${entry%%|*}"
    reason="${entry#*|}"
    echo "SKIPPED crate $crate: $reason"
    args+=(--exclude "$crate")
  done
  echo "+ cargo fmt --all -- --check"
  cargo fmt --all -- --check
  echo "+ cargo test --locked -j 1 -p ny-test-utils --test source_policy"
  cargo test --locked -j 1 -p ny-test-utils --test source_policy
  echo "+ cargo test --locked -j 1 -p ny-test-utils --test python_correctness_migration_manifest"
  cargo test --locked -j 1 -p ny-test-utils --test python_correctness_migration_manifest
  echo "+ cargo test --locked -j 1 -p ny-levers"
  cargo test --locked -j 1 -p ny-levers
  echo "+ cargo ${args[*]} -- -D warnings"
  cargo "${args[@]}" -- -D warnings
  # The VNN-COMP executable is built with this exact non-default feature pair.
  # Workspace Clippy alone does not enable either feature and therefore cannot
  # catch integration drift in the scored MIP/CUDA surface.
  echo "+ cargo clippy --locked -p ny-cli --bin ny --features mip,cuda -- -D warnings"
  cargo clippy --locked -p ny-cli --bin ny --features mip,cuda -- -D warnings
}

# ---------------------------------------------------------------------------
# b) ny-cert test suite
# ---------------------------------------------------------------------------
check_b() {
  echo "+ cargo test --locked -p ny-cert --all-targets"
  cargo test --locked -p ny-cert --all-targets
}

# ---------------------------------------------------------------------------
# c) ny CLI focused contracts: submission, cert, delivered gates, dependency
#    unity, layered flight receipts, and the measured-but-unshipped
#    alpha-zero-yield seam.
# ---------------------------------------------------------------------------
check_c() {
  echo "+ cargo test --locked -p ny-cli --bin ny vnncomp_submit"
  cargo test --locked -p ny-cli --bin ny vnncomp_submit
  echo "+ cargo test --locked -p ny-cli --bin ny cert"
  cargo test --locked -p ny-cli --bin ny cert
  echo "+ cargo test --locked -p ny-cli --bin ny alpha_zero_yield"
  cargo test --locked -p ny-cli --bin ny alpha_zero_yield
  echo "+ cargo test --locked -p ny-propagate --lib alpha_zero_yield"
  cargo test --locked -p ny-propagate --lib alpha_zero_yield
  echo "+ cargo test --locked -p ny-cli --test measured_gate_delivery"
  cargo test --locked -p ny-cli --test measured_gate_delivery
  echo "+ cargo test --locked -p ny-cli --test ay_pin_unity"
  cargo test --locked -p ny-cli --test ay_pin_unity
  echo "+ cargo test --locked -p ny-cli --test flight_lever_receipt"
  cargo test --locked -p ny-cli --test flight_lever_receipt
}

# ---------------------------------------------------------------------------
# d) ny-propagate unit slices
# ---------------------------------------------------------------------------
check_d() {
  echo "+ cargo test --locked -p ny-propagate --lib collection"
  cargo test --locked -p ny-propagate --lib collection
  # The conv2d slice contains scoped tests for the process-global
  # NY_DENSE_BUDGET_MB knob. Run this slice serially so a deliberate
  # zero-budget scope cannot alter an unrelated test's allocation policy.
  echo "+ cargo test --locked -p ny-propagate --lib conv2d -- --test-threads=1"
  cargo test --locked -p ny-propagate --lib conv2d -- --test-threads=1
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
  target_dir="$(cargo metadata --locked --format-version 1 --no-deps | "$NY_GATE_PYTHON" -c \
    'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
  bin="$target_dir/debug/ny"
  if [ ! -x "$bin" ]; then
    echo "debug ny binary absent — building it"
    echo "+ cargo build --locked -p ny-cli --bin ny"
    cargo build --locked -p ny-cli --bin ny
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
      cargo test --locked -p ny-test-utils --features python-tools --tests
  )
}

# ---------------------------------------------------------------------------
# h) STATIC preset capability guard: declared device honoured or waived
# ---------------------------------------------------------------------------
check_h() {
  echo "+ cargo test --locked -p ny-cli --bin ny preset::backend_capability_tests"
  cargo test --locked -p ny-cli --bin ny preset::backend_capability_tests
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
  cargo test --locked --release -p ny-cli --features external-vnncomp --bin ny preset::model_load_smoke_tests -- --nocapture
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
#    vacuous green check [i] refuses for absent benchmark data. Counted
#    2026-08-16 under default features: 54 / 16 / 102. Lower a floor only
#    alongside the deletion that justifies it, never to clear a red.
# ---------------------------------------------------------------------------
run_propagate_oracle_slice() {
  local label="$1" floor="$2" filter="$3"
  local slice_log="$LOG_DIR/check-j-$label.log" ran
  echo "+ cargo test --locked --release -p ny-propagate --lib $filter"
  # `if !` so the pipeline's failure is handled here instead of aborting the
  # check subshell: every slice reports, the way check_f's f1/f2/f3 do.
  if ! cargo test --locked --release -p ny-propagate --lib "$filter" 2>&1 | tee "$slice_log"; then
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
  run_propagate_oracle_slice wide-alpha-true 14 wide_alpha_true::tests:: || ok=1
  run_propagate_oracle_slice interm-refine 95 interm_refine:: || ok=1
  return "$ok"
}

MODE="full"
[ "$FAST" -eq 1 ] && MODE="fast (--fast)"
[ -n "$ONLY" ] && MODE="single check (--only $ONLY)"
echo "ny drift gate — mode: $MODE — repo: $REPO_ROOT"

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
