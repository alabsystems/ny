#!/usr/bin/env bash
# gate.sh — local code, tooling, packaging and proof checks.
#
# Checks:
#   a  Rust hygiene         format, source policy, migration inventory, levers,
#                           workspace Tippy and the CLI mip,cuda feature tier
#   b  ny-cert              hermetic tests plus pinned external Lean grounding
#                           (missing Lake dependencies make the lane VOID)
#   c  CLI contracts        submission, certificates, dependency pins, delivery
#                           and flight receipts
#   d  propagation          collection and conv2d unit tests
#   e  harness              repository Python/shell tooling tests
#   f  submission           package a debug binary and check tarball invariants
#   g  Python coherence     pytest plugin and Cargo-owned tooling contracts
#   h  preset capability    backend declarations versus measured waivers
#   i  preset model load    shipped models and score-at-risk declarations;
#                           missing benchmark data is VOID, never a pass
#   j  soundness oracles    release-mode margin-row, wide-alpha and intermediate
#                           bound tests, plus the optimized calibration probe
#   k  API                  cert-enabled unit and integration tests
#   l  activation bounds    SiLU minimum pins and relaxation enclosure oracles
#   m  GPU API              cert,gpu facade contracts
#   n  score-risk drift     census and bank consistency across both years
#   o  Trust verification   ny-cert advisory library/binary checks; refuted rows
#                           fail, unknown rows are reported separately, and an
#                           unmeasured prover or missing authority is VOID
#
# Flags:
#   --fast        run the bounded edit-loop lane (checks a-h; skips i-o)
#   --only <x>    run a single check by letter (a-o)
#   -h | --help   usage (requires no installed compiler)
#
# Environment:
#   NY_GATE_PYTHON  provisioned interpreter; defaults to the repository .venv
#                   (Unix or Windows layout), then python3
#   NY_RUSTUP       explicit rustup executable for pinned frontend resolution
#   NY_SKIP_CREDENTIAL_CHECK=1  bypass the GitHub credential precondition
#
# PASS means the entire lane ran successfully. FAIL means a checked contract
# failed. VOID means a required host dependency, dataset, prover floor or
# authority was unavailable; available parts still run, and any FAIL wins.
# Exit codes: 0 all selected lanes PASS; 1 any FAIL; 2 invalid arguments or
# refused precondition; 3 no FAIL but at least one VOID. Unselected lanes SKIP.
#
# Banked-verdict replay is separate: scripts/check_banked_rows.py --bin <ny>.
# Run it before banking results and after changing a fail-closed solver gate.
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
        echo "gate.sh: --only requires a check letter a-o" >&2
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
    a|b|c|d|e|f|g|h|i|j|k|l|m|n|o) ;;
    *)
      echo "gate.sh: --only takes a single check letter a-o, got: '$ONLY'" >&2
      exit 2
      ;;
  esac
fi

# An explicit interpreter wins; otherwise prefer the provisioned local venv.
if [ -z "${NY_GATE_PYTHON:-}" ]; then
  for candidate in "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/.venv/Scripts/python.exe"; do
    if [ -x "$candidate" ]; then
      NY_GATE_PYTHON="$candidate"
      break
    fi
  done
fi
: "${NY_GATE_PYTHON:=python3}"

# Temporary lint exclusions are "crate|dated reason" entries. Keep empty
# unless debt is explicitly accepted; never exclude a crate to hide new drift.
CLIPPY_SKIPPED_CRATES=()

# Resolve the pinned Trust sysroot, honoring an explicit rustup selection.
# Native Trust installations expose trustc directly instead of a rustup link.
# Every command uses that sysroot's targo: unverified builds/tests, fmt/tippy,
# and the explicit advisory Trust lane. Stock frontends are not a fallback.
NY_TOOLCHAIN_CHANNEL="$(awk -F'"' '/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"/ { print $2; exit }' rust-toolchain.toml)"
NY_RUSTUP="${NY_RUSTUP:-}"
NY_NATIVE_TRUSTC=""
if [ -z "$NY_RUSTUP" ] && [ "$NY_TOOLCHAIN_CHANNEL" = trust ]; then
  NY_NATIVE_TRUSTC="$(command -v trustc 2>/dev/null || true)"
fi
NY_PINNED_RUSTC=""
NY_RUST_SYSROOT=""
if [ -n "$NY_NATIVE_TRUSTC" ]; then
  NY_PINNED_RUSTC="$NY_NATIVE_TRUSTC"
  # An installed but broken native compiler must refuse, rather than select a
  # different compiler silently. Require its Trust identity before its sysroot.
  if "$NY_PINNED_RUSTC" -vV 2>/dev/null | awk '$1 == "trust:" && $2 ~ /^[0-9]+\./ { found = 1 } END { exit !found }'; then
    NY_RUST_SYSROOT="$("$NY_PINNED_RUSTC" --print sysroot 2>/dev/null || true)"
  fi
else
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
  if [ -n "$NY_RUSTUP" ] && [ -x "$NY_RUSTUP" ] && [ -n "$NY_TOOLCHAIN_CHANNEL" ]; then
    NY_PINNED_RUSTC="$("$NY_RUSTUP" which rustc --toolchain "$NY_TOOLCHAIN_CHANNEL" 2>/dev/null || true)"
  fi
  if [ -n "$NY_PINNED_RUSTC" ] && [ -x "$NY_PINNED_RUSTC" ]; then
    NY_RUST_SYSROOT="$("$NY_PINNED_RUSTC" --print sysroot 2>/dev/null || true)"
  fi
fi
TARGO=""
if [ -n "$NY_RUST_SYSROOT" ]; then
  for candidate in "$NY_RUST_SYSROOT/bin/targo" "$NY_RUST_SYSROOT/bin/targo.exe"; do
    if [ -x "$candidate" ]; then
      TARGO="$candidate"
      break
    fi
  done
fi
if [ -z "$TARGO" ]; then
  echo "gate.sh: cannot resolve the pinned Trust frontend." >&2
  echo "  rust-toolchain.toml pins channel = \"${NY_TOOLCHAIN_CHANNEL:-<unparsed>}\"; native Trust or rustup must resolve a sysroot with a targo frontend." >&2
  echo "  rustup: ${NY_RUSTUP:-<none>}" >&2
  echo "  pinned rustc: ${NY_PINNED_RUSTC:-<none>}" >&2
  echo "  resolved sysroot: ${NY_RUST_SYSROOT:-<none>}" >&2
  echo "  refusing to fall back to stock cargo — it would build and lint with a different toolchain." >&2
  exit 2
fi

# Check credentials before expensive work: an invalid environment token can
# prevent fetching the private ay-milp dependency. No token uses a no-network
# fast path; a rejected token refuses execution. Frontend resolution comes first.
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

# Keep inexpensive drift checks in the edit loop; dataset-heavy and release
# oracle lanes remain available in the full gate and through --only.
FAST_CHECKS="a b c d e f g h"

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ny-gate.XXXXXX")"
trap 'rm -rf "$LOG_DIR"' EXIT

RESULTS=()
FAILURES=0
PASSES=0
SKIPS=0
VOIDS=0

# A lane that cannot run its whole contract on this host records each missing
# prerequisite with lane_void and returns GATE_VOID_RC. VOID needs BOTH: the
# return code alone could be any tool's own exit 3 (pytest uses 3 for an
# internal error), so a 3 with no recorded reason is a FAIL, and so is a
# recorded reason with any other code.
GATE_VOID_RC=3
CHECK_VOID_FILE=""
lane_void() {
  printf '%s\n' "$*" >>"$CHECK_VOID_FILE"
  echo "  VOID  $*"
}

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
  CHECK_VOID_FILE="$LOG_DIR/check-$letter.void"
  : >"$CHECK_VOID_FILE"
  echo ""
  echo "=== [$letter] $name ==="
  local start rc end
  start="$(date +%s)"
  set +e
  ( set -euo pipefail; "$fn" ) 2>&1 | tee "$log"
  rc=$?
  set -e
  end="$(date +%s)"
  if [ "$rc" -eq 0 ] && [ ! -s "$CHECK_VOID_FILE" ]; then
    RESULTS+=("PASS  [$letter] $name  ($((end - start))s)")
    PASSES=$((PASSES + 1))
    echo "--- [$letter] PASS ($((end - start))s) ---"
  elif [ "$rc" -eq "$GATE_VOID_RC" ] && [ -s "$CHECK_VOID_FILE" ]; then
    local reason more
    reason="$(head -n 1 "$CHECK_VOID_FILE")"
    more=$(($(wc -l <"$CHECK_VOID_FILE") - 1))
    if [ "$more" -gt 0 ]; then
      reason="$reason (+$more more)"
    fi
    RESULTS+=("VOID  [$letter] $name  ($((end - start))s)  NOT CHECKED: $reason  log: $log")
    VOIDS=$((VOIDS + 1))
    echo "--- [$letter] VOID ($((end - start))s) ---"
  else
    RESULTS+=("FAIL  [$letter] $name  ($((end - start))s)  log: $log")
    FAILURES=$((FAILURES + 1))
    echo "--- [$letter] FAIL ($((end - start))s) ---"
  fi
}

# ---------------------------------------------------------------------------
# a) Rust formatting + Clippy drift gate
# ---------------------------------------------------------------------------
#    EVERY STEP RUNS, AND EACH FAILING STEP IS NAMED. The lane used to stop at
#    its first failure, so one red hid every red behind it and each fix
#    uncovered the next. Measured 2026-10-08 on origin/main f0de3bf47: the
#    lane reported one failure (fmt drift in three files) while three more sat
#    unreported behind it — the raw NY_* read ratchet (+4), the source frontier
#    (31 unreviewed sites) and workspace Clippy — and had to be found by
#    running each step by hand. The ny-levers run uses --no-fail-fast for the
#    same reason one level down: without it Cargo stops at the first failing
#    test target, and the ratchet hid the frontier.
check_a() {
  # `tippy`, not `clippy`: the pinned Trust sysroot installs `targo-tippy` and
  # keeps `cargo-clippy`/`clippy-driver` as private compat protocols that are
  # never installed aliases, so a `clippy` subcommand does not resolve here.
  local args=(tippy --keep-going --locked --workspace --exclude ny-python --all-targets)
  local entry crate reason failed=""
  # ${arr[@]+...} guard: macOS bash 3.2 under `set -u` treats an EMPTY array
  # expansion as unbound (same idiom as the RESULTS loop below).
  for entry in ${CLIPPY_SKIPPED_CRATES[@]+"${CLIPPY_SKIPPED_CRATES[@]}"}; do
    crate="${entry%%|*}"
    reason="${entry#*|}"
    echo "SKIPPED crate $crate: $reason"
    args+=(--exclude "$crate")
  done
  echo "+ targo fmt --all -- --check"
  "$TARGO" fmt --all -- --check || failed="$failed fmt"
  echo "+ targo --unverified test --locked -j 1 -p ny-test-utils --test source_policy"
  "$TARGO" --unverified test --locked -j 1 -p ny-test-utils --test source_policy ||
    failed="$failed source-policy"
  echo "+ targo --unverified test --locked -j 1 -p ny-test-utils --test python_correctness_migration_manifest"
  "$TARGO" --unverified test --locked -j 1 -p ny-test-utils --test python_correctness_migration_manifest ||
    failed="$failed migration-inventory"
  echo "+ targo --unverified test --locked --no-fail-fast -j 1 -p ny-levers"
  "$TARGO" --unverified test --locked --no-fail-fast -j 1 -p ny-levers ||
    failed="$failed ny-levers"
  echo "+ targo ${args[*]} -- -D warnings"
  "$TARGO" "${args[@]}" -- -D warnings || failed="$failed tippy-workspace"
  # The VNN-COMP executable is built with this exact non-default feature pair.
  # Workspace Clippy alone does not enable either feature and therefore cannot
  # catch integration drift in the scored MIP/CUDA surface.
  echo "+ targo tippy --keep-going --locked -p ny-cli --bin ny --features mip,cuda -- -D warnings"
  "$TARGO" tippy --keep-going --locked -p ny-cli --bin ny --features mip,cuda -- -D warnings ||
    failed="$failed tippy-ny-cli-mip-cuda"
  if [ -n "$failed" ]; then
    echo "  FAIL  a: failing step(s):$failed — every step ran; each failure is reported above"
    return 1
  fi
  echo "  ok    a: fmt, source policy, migration inventory, ny-levers and both Clippy tiers passed"
}

# ---------------------------------------------------------------------------
# b) ny-cert test suite
# ---------------------------------------------------------------------------
#    Two halves. The hermetic suite (default features) needs nothing but the
#    tree. GROUNDING — every #[trust::cite(crownproof::X)] resolves to a
#    declared, sorry-free theorem in the exact pinned Clean dependency, and the
#    proof-carrying status of check_farkas against it — reads the downloaded
#    Lake checkout under crates/ny-cert/proofs/lean/.lake/packages, so it is the
#    `external-lean` lane's `pinned_corpus` slice (TEST_CONFORMANCE.md: the
#    default lane must not need a downloaded corpus). Until 2026-10-04 those
#    checks sat in the default lane, so every host without the checkout (any
#    fresh worktree since 8511be9a3 replaced the vendored copy with the Lake
#    dependency) reported six "test failures" that were one missing download.
#
#    Absent checkout: the hermetic half still runs (and can FAIL), then the
#    lane is VOID naming both paths — never a pass, never a test failure.
#    Present: the slice runs with a count floor (a filter matching nothing is
#    VOID, not green) and FAILS on a missing/drifted pin or an ungrounded cite.
LANE_B_PINNED_FLOOR=4
check_b() {
  echo "+ targo --unverified test --locked -p ny-cert --all-targets"
  "$TARGO" --unverified test --locked -p ny-cert --all-targets

  local lean_root="crates/ny-cert/proofs/lean" missing=0 path
  for path in "$lean_root/.lake/packages/crownproof/crown-proofs/lean/Crownproof" \
    "$lean_root/.lake/packages/mathlib/Mathlib"; do
    if [ ! -d "$path" ]; then
      lane_void "pinned Lake checkout absent: $REPO_ROOT/$path (run \`lake update\` in $lean_root) — #[trust::cite] grounding NOT checked"
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    return "$GATE_VOID_RC"
  fi

  local log="$LOG_DIR/check-b-pinned-corpus.log" rc ran
  echo "+ targo --unverified test --locked -p ny-cert --features external-lean --lib pinned_corpus"
  set +e
  "$TARGO" --unverified test --locked -p ny-cert --features external-lean --lib pinned_corpus 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  if grep -qE '^test result: FAILED' "$log" || [ "$rc" -ne 0 ]; then
    echo "  FAIL  b: the pinned-corpus grounding slice failed (exit $rc)"
    return 1
  fi
  ran="$(awk '/^test result: ok\. [0-9]+ passed/ { n += $4 } END { print n + 0 }' "$log")"
  if [ "$ran" -lt "$LANE_B_PINNED_FLOOR" ]; then
    echo "  FAIL  b: the pinned_corpus filter ran $ran test(s), floor is $LANE_B_PINNED_FLOOR — the grounding slice proved nothing"
    return 1
  fi
  echo "  ok    b: hermetic suite and $ran pinned-corpus grounding test(s) passed"
}

# ---------------------------------------------------------------------------
# c) ny CLI focused contracts: submission, cert, delivered gates, dependency
#    unity, layered flight receipts, and the measured-but-unshipped
#    alpha-zero-yield seam.
# ---------------------------------------------------------------------------
check_c() {
  echo "+ targo --unverified test --locked -p ny-cli --bin ny vnncomp_submit"
  "$TARGO" --unverified test --locked -p ny-cli --bin ny vnncomp_submit
  echo "+ targo --unverified test --locked -p ny-cli --bin ny cert"
  "$TARGO" --unverified test --locked -p ny-cli --bin ny cert
  echo "+ targo --unverified test --locked -p ny-cli --bin ny alpha_zero_yield"
  "$TARGO" --unverified test --locked -p ny-cli --bin ny alpha_zero_yield
  echo "+ targo --unverified test --locked -p ny-propagate --lib alpha_zero_yield"
  "$TARGO" --unverified test --locked -p ny-propagate --lib alpha_zero_yield
  echo "+ targo --unverified test --locked -p ny-cli --test measured_gate_delivery"
  "$TARGO" --unverified test --locked -p ny-cli --test measured_gate_delivery
  echo "+ targo --unverified test --locked -p ny-cli --test ay_pin_unity"
  "$TARGO" --unverified test --locked -p ny-cli --test ay_pin_unity
  echo "+ targo --unverified test --locked -p ny-cli --test flight_lever_receipt"
  "$TARGO" --unverified test --locked -p ny-cli --test flight_lever_receipt
}

# ---------------------------------------------------------------------------
# d) ny-propagate unit slices
# ---------------------------------------------------------------------------
check_d() {
  echo "+ targo --unverified test --locked -p ny-propagate --lib collection"
  "$TARGO" --unverified test --locked -p ny-propagate --lib collection
  # The conv2d slice contains scoped tests for the process-global
  # NY_DENSE_BUDGET_MB knob. Run this slice serially so a deliberate
  # zero-budget scope cannot alter an unrelated test's allocation policy.
  echo "+ targo --unverified test --locked -p ny-propagate --lib conv2d -- --test-threads=1"
  "$TARGO" --unverified test --locked -p ny-propagate --lib conv2d -- --test-threads=1
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
  target_dir="$("$TARGO" metadata --locked --format-version 1 --no-deps | "$NY_GATE_PYTHON" -c \
    'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
  bin="$target_dir/debug/ny"
  if [ ! -x "$bin" ]; then
    echo "debug ny binary absent — building it"
    echo "+ targo --unverified build --locked -p ny-cli --bin ny"
    "$TARGO" --unverified build --locked -p ny-cli --bin ny
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
#    BOTH HALVES RUN, AND EVERY CARGO TEST TARGET REPORTS. The lane used to
#    stop at the plugin suite's first failure, so the Cargo-owned contracts
#    behind it never ran while that failure stood. Measured 2026-10-08: the
#    stale Python version (red since ac4cf1483, 2026-09-07) hid
#    repository_policy_black_box, which had been failing since 5ccc91508
#    (2026-09-30) added ny-proofcheck without a layer or a package-inventory
#    entry; fixing the version uncovered it. --no-fail-fast for the same
#    reason one level down: Cargo otherwise stops at the first failing test
#    target (python_tooling_black_box sorts before repository_policy_black_box).
check_g() {
  local gate_python="$NY_GATE_PYTHON" failed=""
  if ! env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$gate_python" -s -c 'import numpy, onnx, onnxruntime, pytest, sys, yaml; __import__("tomllib" if sys.version_info >= (3, 11) else "tomli")' \
    >/dev/null 2>&1; then
    echo "ERROR: Python tooling dependencies are not importable in isolated mode by $gate_python." >&2
    echo "Use the README virtualenv setup, install requirements.txt there, and set NY_GATE_PYTHON to that interpreter." >&2
    return 1
  fi
  echo "+ (cd crates/ny-python && env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 $gate_python -s -m pytest -p ny_pytest.plugin ny_pytest/tests -q)"
  # `cd … &&`: a subshell used as an `if` condition runs with errexit off, so
  # a bare `cd` that failed would let pytest run in the wrong directory.
  if ! (
    unset NY_GATE_PYTHON
    cd crates/ny-python &&
      env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
        "$gate_python" -s -m pytest -p ny_pytest.plugin ny_pytest/tests -q
  ); then
    echo "  FAIL  g: the pytest plugin suite failed"
    failed="$failed plugin-suite"
  fi
  echo "+ NY_TEST_PYTHON=$gate_python targo --unverified test --locked --no-fail-fast -p ny-test-utils --features python-tools --tests"
  if ! (
    unset NY_GATE_PYTHON
    NY_TEST_PYTHON="$gate_python" \
      "$TARGO" --unverified test --locked --no-fail-fast -p ny-test-utils --features python-tools --tests
  ); then
    echo "  FAIL  g: a Cargo-owned Python tooling or repository-policy contract failed"
    failed="$failed cargo-contracts"
  fi
  if [ -n "$failed" ]; then
    echo "  FAIL  g: failing half(s):$failed — both ran; each failure is reported above"
    return 1
  fi
  echo "  ok    g: the plugin suite and every Cargo-owned Python tooling and policy contract passed"
}

# ---------------------------------------------------------------------------
# h) STATIC preset capability guard: declared device honoured or waived
# ---------------------------------------------------------------------------
check_h() {
  echo "+ targo --unverified test --locked -p ny-cli --bin ny preset::backend_capability_tests"
  "$TARGO" --unverified test --locked -p ny-cli --bin ny preset::backend_capability_tests
}

# ---------------------------------------------------------------------------
# i) STATIC preset model-load smoke: every shipped preset's models still load
#
#    --release on purpose: this decodes ~1 GB of ONNX protobuf, which is minutes
#    in a debug build and seconds in release.
#
#    This check never PASSES when benchmark data is absent. That is
#    deliberate: it used to pass vacuously, and a vacuous pass is
#    indistinguishable from "everything loads" in a gate log. Do not "fix" a
#    red here by exporting NY_PRESET_LOAD_SMOKE=off in this script — that
#    reintroduces the exact hole. Make the data reachable instead.
#
#    Two runs of the same test binary, so the result says WHICH failure it is:
#      1. the guard run: every model that IS on the box loads, the stake
#         registry, the helpers. The guard test itself FAILS when inputs are
#         absent, but only AFTER its loads: it prints its `model file(s)
#         checked` line once every assertion about the present models has held,
#         then panics BENCHMARK DATA IS NOT ON THIS BOX. So a guard run whose
#         ONLY failed test is the guard, with that line and that panic, is the
#         absent-data outcome; any other failure here is a FAIL;
#      2. shipped_preset_benchmark_data_is_present alone — its failure names
#         every missing path, and the lane is VOID with each one.
#    The two must agree (both pass, or both report absent data); a disagreement
#    is a FAIL. A guard that PASSES over absent data and leaves the failing to
#    the second run (the first draft of this split, caught in review before it
#    reached main) exits 0 having loaded nothing when it is selected by name on
#    a box with no data.
#    Before the split a sparse checkout (this Mac's vnncomp2025 materializes 5
#    of 26 categories) panicked "PRESET POINTS AT A MISSING CATEGORY" on the
#    first absent category and the guard loaded nothing at all.
LANE_I_GUARD_FLOOR=5
LANE_I_GUARD_TEST="preset::model_load_smoke_tests::every_shipped_preset_model_still_loads"

# Did the guard run in log $1 fail for absent data ONLY? All four must hold:
# one result line, FAILED with exactly one failed test and the rest of the
# floor passed; the failure list names the guard and nothing else; the guard
# printed its loads-held line; and its panic is the absent-data one.
lane_i_guard_failed_only_on_absent_data() {
  local log="$1" passed failed_names
  [ "$(grep -cE '^test result: ' "$log")" -eq 1 ] || return 1
  passed="$(sed -nE 's/^test result: FAILED\. ([0-9]+) passed; 1 failed; .*/\1/p' "$log")"
  [ -n "$passed" ] && [ "$passed" -ge $((LANE_I_GUARD_FLOOR - 1)) ] || return 1
  failed_names="$(sed -n '/^failures:$/,/^test result: /p' "$log" | sed -nE 's/^    ([A-Za-z0-9_:]+)$/\1/p' | LC_ALL=C sort -u)"
  [ "$failed_names" = "$LANE_I_GUARD_TEST" ] || return 1
  # Not anchored to the line start: with one test thread libtest prints
  # `test <name> ... ` and the test's own stderr continues that line, so on a
  # box with NO data (nothing loaded, nothing printed before it) this is the
  # first thing the guard writes and it follows the test's name.
  grep -qE 'preset load smoke: [0-9]+ model file\(s\) checked across [0-9]+ preset\(s\)' "$log" || return 1
  grep -q 'BENCHMARK DATA IS NOT ON THIS BOX' "$log"
}

check_i() {
  local guard_log="$LOG_DIR/check-i-guard.log" data_log="$LOG_DIR/check-i-data.log" rc ran guard_absent=0
  local presence="preset::model_load_smoke_tests::shipped_preset_benchmark_data_is_present"
  echo "+ targo --unverified test --locked --release -p ny-cli --features external-vnncomp --bin ny preset::model_load_smoke_tests -- --nocapture --skip $presence"
  set +e
  "$TARGO" --unverified test --locked --release -p ny-cli --features external-vnncomp --bin ny preset::model_load_smoke_tests -- --nocapture --skip "$presence" 2>&1 | tee "$guard_log"
  rc=${PIPESTATUS[0]}
  set -e
  if grep -qE '^test result: FAILED' "$guard_log" || [ "$rc" -ne 0 ]; then
    if [ "$rc" -ne 0 ] && lane_i_guard_failed_only_on_absent_data "$guard_log"; then
      guard_absent=1
      echo "  ....  i: every present model loaded; the guard failed ONLY for inputs this box lacks — naming them next"
    else
      echo "  FAIL  i: the model-load guard failed (exit $rc) — see the panic above"
      return 1
    fi
  else
    ran="$(awk '/^test result: ok\. [0-9]+ passed/ { n += $4 } END { print n + 0 }' "$guard_log")"
    if [ "$ran" -lt "$LANE_I_GUARD_FLOOR" ]; then
      echo "  FAIL  i: the guard filter ran $ran test(s), floor is $LANE_I_GUARD_FLOOR — it proved nothing"
      return 1
    fi
  fi

  echo "+ targo --unverified test --locked --release -p ny-cli --features external-vnncomp --bin ny $presence -- --nocapture --exact"
  set +e
  "$TARGO" --unverified test --locked --release -p ny-cli --features external-vnncomp --bin ny "$presence" -- --nocapture --exact 2>&1 | tee "$data_log"
  rc=${PIPESTATUS[0]}
  set -e
  if grep -qE '^test result: ok\. 1 passed' "$data_log" && [ "$rc" -eq 0 ]; then
    if [ "$guard_absent" -ne 0 ]; then
      echo "  FAIL  i: the guard failed for absent benchmark data but the data test found none absent — the two halves disagree"
      return 1
    fi
    echo "  ok    i: $ran guard test(s) passed and every preset's benchmark data is present"
    return 0
  fi
  if grep -qE '^test result: FAILED\. 0 passed; 1 failed' "$data_log" &&
    grep -q 'BENCHMARK DATA IS NOT ON THIS BOX' "$data_log"; then
    if [ "$guard_absent" -eq 0 ]; then
      echo "  FAIL  i: the data test reports absent benchmark data but the guard PASSED — a guard that passes over inputs it did not load is the vacuous green this lane refuses"
      return 1
    fi
    local path
    while IFS= read -r path; do
      lane_void "benchmark data absent: $path"
    done < <(sed -nE 's/^[[:space:]]+missing: (.*)$/\1/p' "$data_log" | LC_ALL=C sort -u)
    if [ ! -s "$CHECK_VOID_FILE" ]; then
      lane_void "benchmark data absent (the data test named no path; see $data_log)"
    fi
    echo "  VOID  i: every present model loads; the inputs above were NOT checked"
    return "$GATE_VOID_RC"
  fi
  echo "  FAIL  i: the data-presence test did not run cleanly (exit $rc) — neither a pass nor a data report"
  return 1
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
  echo "+ targo --unverified test --locked --release -p ny-propagate --lib $filter"
  # `if !` so the pipeline's failure is handled here instead of aborting the
  # check subshell: every slice reports, the way check_f's f1/f2/f3 do.
  if ! "$TARGO" --unverified test --locked --release -p ny-propagate --lib "$filter" 2>&1 | tee "$slice_log"; then
    echo "  FAIL  j-$label: targo --unverified test failed for filter '$filter'"
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
  # Not an oracle: the OPTIMIZED arm of a build-kind pin. An unoptimized build
  # of ny-propagate's unit tests takes the shipped forward-linear rate without
  # probing (#fl-calib-debug-tests, docs/DECISIONS.md D24), so no debug lane
  # builds the probe's fixture any more, and this lane holds the gate's only
  # release build of those tests. The pin counts probe reps (at least one must
  # start in an optimized build); it reads no clock. Floor 1 = the one test,
  # by its full name. It reuses the binary the slices above built and costs
  # the probe itself: three reps, each bounded at 5 s.
  run_propagate_oracle_slice fl-probe-pin 1 forward_linear::tests::calibration_probe_runs_in_optimized_builds_only || ok=1
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
  echo "+ targo --unverified test --locked -p ny-api --features cert --all-targets"
  # "$TARGO", not a bare PATH lookup: gate.sh resolves $TARGO from the sysroot
  # that rust-toolchain.toml pins (see the frontend resolution above). A PATH
  # lookup can be a DIFFERENT toolchain than every other lane runs against, which
  # would make this lane's result incomparable with theirs.
  "$TARGO" --unverified test --locked -p ny-api --features cert --all-targets 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}

  # Branch on the ok/FAILED token, never on the count alone: `ok. 0 passed` is
  # VOID (matched nothing) while `FAILED. 0 passed; N failed` is a real failure.
  if grep -qE '^test result: FAILED' "$log"; then
    echo "  FAIL  k: ny-api reported a real test failure"
    return 1
  fi
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL  k: targo --unverified test -p ny-api exited $rc"
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
# l) activation-math certified-bound pins (SiLU) + scalar relaxation envelopes
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
#    of them reaches `silu::`.
#
#    The second slice, `tests::relaxation_envelope::`, checks the envelope of
#    every production scalar relaxation (relu, abs, exp, pow2, sqrt, log, silu,
#    gelu erf/tanh) against the function it bounds: dense sampling at a few-ULP
#    tolerance, and an adversarial grid (subnormals, thresholds, branch points,
#    extremes) at near-zero slack against independent f64 references, with
#    sqrt's alpha-CROWN tangent point swept across and outside [l, u]. Until
#    2026-09-29 this lane ran those checks against `ny-relaxation`, a scalar
#    mirror kept only for the Kani harnesses, plus the drift test binding the
#    mirror to production and asserting production's soundness on that grid.
#    The mirror and the harnesses were deleted (owner ruling 2026-09-29: prune
#    the Kani harnesses), and both halves were ported to the production crate.
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
#    The floors are STATIC counts, derived by enumerating `#[test]` in the
#    sources rather than by observing a run:
#      ny-propagate `silu::` (2026-08-28) — 27 existing (24 top-level in
#        silu/tests.rs plus 3 inside its single `proptest!` block; one of the
#        24 is the `#[cfg(debug_assertions)]` chord-precondition case) + 4 new
#        pins = 31.
#      ny-propagate `tests::relaxation_envelope::` (2026-09-29) — 19, and a
#        run on m29 passed exactly 19.
#    The first execution of this lane is what confirms them. A floor that is
#    too HIGH fails closed and its FAIL line prints the real number to correct
#    it with; a floor is only ever LOWERED alongside the deletion that
#    justifies it, never to clear a red.
run_activation_pin_slice() {
  local label="$1" floor="$2" slice_log="$3"
  shift 3
  local ran rc=0
  echo "+ targo --unverified $*"
  # `if !` rather than a bare pipeline: run_check runs this inside
  # `set -euo pipefail`, where a failing pipeline aborts the subshell before
  # any of the diagnostics below can print. Lane [j] uses the same form.
  if ! "$TARGO" --unverified "$@" 2>&1 | tee "$slice_log"; then
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
    echo "  FAIL  l-$label: targo exited non-zero with no FAILED test result"
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
  local envelope_log="$LOG_DIR/check-l-relaxation-envelope.log"
  local ok=0

  run_activation_pin_slice propagate 31 "$prop_log" \
    test --locked -p ny-propagate --lib silu:: || ok=1
  run_activation_pin_slice envelope 19 "$envelope_log" \
    test --locked -p ny-propagate --lib tests::relaxation_envelope:: || ok=1

  # BY NAME, ANCHORED on the full module path and on the trailing ` ... ok`.
  # A bare `silu` substring also matches the silu_chord/silu_eval tests, which
  # say nothing about the Newton premise, and an unanchored name is how "elu"
  # matched "celu" earlier in this campaign. All four pins must have RUN and
  # PASSED.
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
  done
  [ "$missing" -eq 0 ] || ok=1

  if [ "$ok" -eq 0 ]; then
    echo "  ok    l: all 4 certified-bound pins and the envelope slice passed"
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
  echo "+ targo --unverified test --locked -p ny-api --features cert,gpu --all-targets"
  # "$TARGO" for the same reason as lane [k]: the pinned sysroot frontend, not
  # whatever PATH happens to resolve.
  # set +e around the pipeline, as in [o]: under run_check's `set -euo
  # pipefail` a failing targo would abort here, before PIPESTATUS is read and
  # before the FAIL lines below could say why.
  set +e
  "$TARGO" --unverified test --locked -p ny-api --features cert,gpu --all-targets 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  if grep -qE '^test result: FAILED' "$log"; then
    echo "  FAIL  m: ny-api reported a real test failure under cert,gpu"
    return 1
  fi
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL  m: targo --unverified test -p ny-api --features cert,gpu exited $rc"
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
#         self-consistent. The bank is the file in the ONE directory the scorer
#         reads for that row's track (its --measured default for a regular
#         row, --ext-measured for an extended one), anchored to the scorer like
#         the lists in 1: a same-named file in another field's ledger
#         (reports/measured-2026) or in the other track's directory is not the
#         bank, a stake above 0.0 with no bank is drift (the scorer credits an
#         unmeasured benchmark 0), and a recorded `command` in the yaml that
#         names another directory is drift rather than an instruction.
#    It exits 2 (VOID) rather than 0 when it cannot check — a bank directory
#    missing, no bank files found — and reports a bank file with no
#    recognisable verdict column as a failure. Either way lane [n] FAILS: the
#    banks are tracked files, so their absence is repository drift, not a
#    host condition (which is what the gate's own VOID status means).
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

# ---------------------------------------------------------------------------
# o) ny-cert verified lane (advisory ramp) — `targo trust check --allow-l0-gaps`
# ---------------------------------------------------------------------------
#    Owner direction 2026-10-01: move ALab builds onto Trust's verified lane.
#    The advisory ramp FAILS on any FAILED (refuted) Level 0 row and REPORTS
#    unknown rows without failing. ny-cert's library reached 0 failed rows at
#    ny 189fcf0d7 (2026-10-01) and its binaries (certify_onnx,
#    certify_onnx_diff, frontier, lrat_to_lean) at the fix/ny-cert-bins-rows
#    commit (126 -> 0); this lane holds both there.
#
#    Library, measured on m29, 2026-10-01, targo 321aaeda7 (seal 9192), on ny
#    origin/main f7fe28ed1: 249 s cold (bare command, exit 0, Gate:
#    CONDITIONAL-PASS, 309 proved, 0 failed, 0 runtime-checked, 6 assumed,
#    10 mandated, 561 unknown / 886 total) and 276 s through `gate.sh --only o`
#    on the warm tree (307 proved, 0 failed, 562 unknown / 885 total).
#    Binaries, measured on m29, 2026-10-01, same targo: 110-119 s per run,
#    Gate: CONDITIONAL-PASS, 101-102 proved, 0 failed, 114 mandated,
#    1017-1018 unknown / 1233 total across four runs of the phase-1 tree.
#    After the review fixes (matrix entry cap), measured 2026-10-02, same
#    targo, two runs each, identical both times: `--bins` 100 proved,
#    0 failed, 114 mandated, 1018 unknown / 1232 total (110 s); per binary
#    (the lane's runs): certify_onnx 65 proved, 90 mandated, 711 unknown /
#    866 total (106 s); certify_onnx_diff 23 proved, 19 mandated,
#    251 unknown / 293 total (45 s); frontier 7 proved, 2 mandated,
#    26 unknown / 35 total (14 s); lrat_to_lean 6 proved, 3 mandated,
#    29 unknown / 38 total (18 s). All 0 failed.
#    Verification is NOT cached between runs, so a warm run takes as long as
#    a cold one. Both the proved/unknown split AND the obligation total move
#    between runs (solver timeouts), so neither is a stable pin; each run
#    gates on its verdict, its failed count, and its own proved floor.
#    NOT in FAST_CHECKS: six minutes is not an edit-loop cost.
#
#    A green here is NOT "proved": the unknown, assumed, mandated,
#    runtime-checked and contract-panic rows are conditions of the pass, and
#    the ok lines echo targo's whole Gate lines rather than summarising any
#    of them away. Only the strict lane (`targo trust check`, no
#    --allow-l0-gaps) or --certify makes them failures; that is the
#    destination, not today.
#
#    ONE RUN PER TARGET, NOT ONE OVERALL. `--lib --bins` in one invocation
#    would print one combined Gate line, and a single floor over the sum
#    cannot see a target go dark: the library alone clears a combined floor.
#    The same holds one level down: under one `--bins` floor of 80 against
#    101-102 measured, certify_onnx_diff (22 proved) could vanish entirely,
#    or frontier and lrat_to_lean together (13), and the lane stayed green.
#    So the library and EACH binary get their own run, their own
#    exactly-one-Gate-line rule and their own proved floor, and the lane
#    passes only if every run does. All runs always execute (one failure
#    does not hide another target's verdict). The binary list is checked
#    against crates/ny-cert/src/bin, so a new binary without a floor fails
#    the lane instead of going ungated.
#
#    --locked, like every other Cargo-resolving lane here and the ramp in
#    publish/config.sh: the gate must verify the dependency versions the
#    tracked Cargo.lock names, and must never rewrite it.
#
#    Fail-closed parsing, same discipline as [j]/[k]/[l]: targo's exit code
#    alone is not trusted. Each run must print EXACTLY ONE `Gate:` line (none
#    is VOID: wrong frontend, a cargo masquerading as targo, a lane that
#    verified nothing; several is VOID: a pass line could hide a failing one).
#    Its verdict word must be PASS or CONDITIONAL-PASS (FAIL, INCONCLUSIVE,
#    VOID or anything unrecognised fails even at 0 failed rows and exit 0).
#    The failed count must be 0, the total non-zero, and the proved count at
#    least that target's floor: a solver that went dark turns every row
#    unknown, which is 0 failed and must not read as green.
#    Own log paths, for lane [k]'s reason (run_check tees to check-o.log while
#    this function reads).
#
#    The floors sit below the measured counts (lib 307-309; certify_onnx 65,
#    certify_onnx_diff 23, frontier 7, lrat_to_lean 6) to absorb timeout
#    jitter (certify_onnx_diff's timeouts moved 0-55 between runs) and small
#    ny-cert edits, and each is above zero, so any one binary going dark
#    fails; lower one only with a fresh measurement recorded here, never to
#    turn the lane green.
#
#    A PROVED COUNT IS A PROPERTY OF (SOURCE, PROVER), so the floors are kept
#    PER PROVER, keyed by the targo commit `targo --version` prints. Every
#    number above was measured under m29's targo 321aaeda7. The m21 Mac's pinned
#    Trust is 6606d3bdf (2026-09-25), a different line (115 commits of 321aaeda7
#    are not in it, and it classifies rows as runtime-checked/assumed that
#    321aaeda7 reported otherwise). On the SAME ny-cert input — nothing under
#    crates/ny-cert, crates/ny-contracts, Cargo.lock, Cargo.toml,
#    rust-toolchain.toml or .cargo changed between 5614ac9d3, where these
#    floors were set, and 228dab0ba — 6606d3bdf proves frontier 1-2 of 33 and
#    lrat_to_lean 2-5 of 40 in every run, so 321aaeda7's floors 5 and 4 failed
#    on every m21 gate while no proof had been lost from the source. Comparing
#    counts across provers is meaningless in BOTH directions (a prover that
#    proves more would equally hide a lost proof), so a prover with no
#    measured set is not compared at all: the lane still runs every target and
#    still FAILS a refuted row, a missing or extra Gate line, a non-pass
#    verdict, a non-zero exit or zero proved, and then ends VOID naming the
#    prover. To add a set, run `gate.sh --only o` at least three times on a
#    tree whose ny-cert input matches a recorded measurement, record every
#    count here, and set each floor below the lowest.
#
#    6606d3bdf, measured on m21 (M5 Max), on the ny-cert Rust input of
#    5614ac9d3 (unchanged through 7f4564cfe: main has added only Lean proofs
#    under crates/ny-cert since), eight `gate.sh` lane [o] runs — seven on
#    2026-10-02..04 and the last on 2026-10-08 at pure origin/main 7f4564cfe —
#    each beside other ledger jobs at load 20-45 (the 218/43 run overlapped a
#    second gate), every run 0 failed (proved / total):
#      lib               256/1027 250/1026 280/1028 282/1030 274/1027 218/1024 266/1025 271/1025
#      certify_onnx       43/785   60/804   69/893   69/893   66/788   43/785   65/805   69/835
#      certify_onnx_diff  17/183   25/289   26/289   26/289   24/289   21/183   23/289   23/289
#      frontier            1/33     2/33     2/33     2/33     2/33     2/33     2/33     2/33
#      lrat_to_lean        2/40     3/40     3/40     5/40     2/40     3/40     3/40     3/40
#    Floors sit below the lowest run; frontier's lowest is 1, so its floor is
#    1 (it still fails a binary gone dark).
LANE_O_FLOOR_SETS='
321aaeda7 lib:250 certify_onnx:50 certify_onnx_diff:15 frontier:5 lrat_to_lean:4
6606d3bdf lib:200 certify_onnx:35 certify_onnx_diff:13 frontier:1 lrat_to_lean:1
'

# The start of targo's refusal when it cannot provision its trustd memory
# authority (targo-trust/src/pipeline/run.rs: both variants begin with it and
# continue "at <socket>: <reason>; refusing to launch a split-authority Cargo
# fan-out"). lane_o_run reports that refusal as VOID.
LANE_O_NO_AUTHORITY='could not establish the selected same-sysroot trustd memory authority'

# The prover this gate runs: the commit in `targo --version`'s first line
# (`targo 1.99.0-dev (6606d3bdf 2026-09-25) (...)`), or empty.
lane_o_prover() {
  "$TARGO" --version 2>/dev/null |
    sed -nE '1s/^targo [^ ]+ \(([0-9a-f]{7,40})[ )].*$/\1/p'
}

# The floor set recorded for prover $1 (`target:floor ...`), or empty. A key
# matches when either spelling of the commit is a prefix of the other.
lane_o_floor_set() {
  local prover="$1" key rest
  [ "${#prover}" -ge 7 ] || return 0
  while read -r key rest; do
    [ -n "$key" ] || continue
    if [ "${prover#"$key"}" != "$prover" ] || [ "${key#"$prover"}" != "$key" ]; then
      printf '%s\n' "$rest"
      return 0
    fi
  done <<EOF_SETS
$LANE_O_FLOOR_SETS
EOF_SETS
}

# The proved floor for target $2 (`lib` or a binary name) in floor set $1.
lane_o_floor() {
  local entry
  for entry in $1; do
    if [ "${entry%%:*}" = "$2" ]; then
      printf '%s\n' "${entry#*:}"
      return 0
    fi
  done
}

# One verified-lane run over one ny-cert target selection (`--lib`, or
# `--bin <name>`), labelled in every message as `o/<label>`.
lane_o_run() {
  local label="$1" floor="$2"
  shift 2
  local log="$LOG_DIR/check-o-cert-verified-${label//[^A-Za-z0-9_-]/-}.log" rc gate_lines gate_line n_gate verdict failed total proved
  echo "+ targo trust check --allow-l0-gaps --locked -p ny-cert $*"
  # set +e around the pipeline: run_check runs this under `set -euo pipefail`,
  # where a non-zero targo would abort the subshell before PIPESTATUS is read
  # and before the diagnostics below could name the cause.
  set +e
  "$TARGO" trust check --allow-l0-gaps --locked -p ny-cert "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  gate_lines="$(grep -E '^[[:space:]]*Gate: ' "$log" || true)"
  if [ -z "$gate_lines" ]; then
    # targo's own refusal to START the verified lane, as opposed to a run that
    # ended without a verdict: it could not get this host's trustd memory
    # authority (another Trust build's trustd holds the socket, the sibling
    # trustd is missing, or its crash epoch is not CLEAN) and compiled and
    # verified nothing. That is a state of the HOST, which no change to this
    # tree can cause or cure, so it is VOID with targo's reason — measured
    # 2026-10-04, when a Trust stage-1 build's trustd held the socket and
    # every target of this lane read as five FAILs. Only this exact refusal
    # with a non-zero exit qualifies; any other silence (a compile error, a
    # wrong frontend, a crash) stays a FAIL.
    if [ "$rc" -ne 0 ] && grep -qF "$LANE_O_NO_AUTHORITY" "$log" &&
      grep -qF 'refusing to launch a split-authority Cargo fan-out' "$log"; then
      local socket why
      socket="$(sed -nE "s/.*$LANE_O_NO_AUTHORITY at ([^ ]+): .*/\\1/p" "$log" | sed -n 1p)"
      why="$(grep -m 1 -E '^trustd: fatal: ' "$log" || true)"
      if [ -z "$why" ]; then
        why="$(sed -nE "s/.*$LANE_O_NO_AUTHORITY at [^ ]+: (.*)/\\1/p" "$log" | sed -n 1p | cut -c1-160)"
      fi
      lane_void "lane [o] $label: the verified lane did not start — targo could not establish this host's trustd memory authority at ${socket:-<unparsed socket>} (${why:-no reason given}); NOT verified. Follow targo's text above (wait for the other trustd to retire; never signal it) and re-run"
      return "$GATE_VOID_RC"
    fi
    echo "  FAIL  o/$label: no 'Gate:' line — targo trust did not report a verdict (VOID)"
    return 1
  fi
  n_gate="$(printf '%s\n' "$gate_lines" | wc -l | tr -d ' ')"
  if [ "$n_gate" -ne 1 ]; then
    echo "  FAIL  o/$label: $n_gate 'Gate:' lines, expected exactly 1 (VOID):"
    printf '%s\n' "$gate_lines"
    return 1
  fi
  gate_line="$gate_lines"
  verdict="$(printf '%s\n' "$gate_line" | sed -nE 's/^[[:space:]]*Gate: ([A-Z-]+)([[:space:]].*)?$/\1/p')"
  failed="$(printf '%s\n' "$gate_line" | sed -nE 's/.*[(, ]([0-9]+) failed[,)].*/\1/p')"
  total="$(printf '%s\n' "$gate_line" | sed -nE 's/.*\/ ([0-9]+) total\).*/\1/p')"
  proved="$(printf '%s\n' "$gate_line" | sed -nE 's/.*\(([0-9]+) proved[,)].*/\1/p')"
  if [ -z "$verdict" ] || [ -z "$failed" ] || [ -z "$total" ] || [ -z "$proved" ]; then
    echo "  FAIL  o/$label: unparseable Gate line (VOID): $gate_line"
    return 1
  fi
  if [ "$failed" -ne 0 ]; then
    echo "  FAIL  o/$label: $failed FAILED verified-lane row(s): $gate_line"
    return 1
  fi
  case "$verdict" in
    PASS|CONDITIONAL-PASS) : ;;
    *) echo "  FAIL  o/$label: Gate verdict is $verdict, not PASS/CONDITIONAL-PASS: $gate_line"; return 1 ;;
  esac
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL  o/$label: targo trust check exited $rc with 0 failed rows: $gate_line"
    return 1
  fi
  if [ "$total" -eq 0 ]; then
    echo "  FAIL  o/$label: 0 obligations in total — the lane verified NOTHING (VOID)"
    return 1
  fi
  # `[ 5 -lt 2OO ]` is a bash error, which an `if` reads as FALSE: a mistyped
  # or missing floor would wave every proved count through. Checked here, after
  # the run, so the target's refuted rows and verdict are still reported above.
  case "$floor" in
    ''|*[!0-9]*|0*)
      echo "  FAIL  o/$label: proved floor '$floor' is not a positive integer, so $proved proved of $total was compared with nothing — fix LANE_O_FLOOR_SETS"
      return 1
      ;;
  esac
  if [ "$proved" -lt "$floor" ]; then
    echo "  FAIL  o/$label: only $proved proved of $total, below the floor of $floor (solver dark or obligations lost): $gate_line"
    return 1
  fi
  echo "  ok    o/$label: 0 failed, $proved proved (floor $floor); every other row is a CONDITION, not proved:"
  echo "        $(printf '%s\n' "$gate_line" | sed -E 's/^[[:space:]]+//')"
}

# The ny-cert binaries Cargo builds: src/bin/<name>.rs and
# src/bin/<name>/main.rs (the crate declares no [[bin]] tables), sorted.
lane_o_actual_bins() {
  local f
  for f in crates/ny-cert/src/bin/*.rs crates/ny-cert/src/bin/*/main.rs; do
    [ -e "$f" ] || continue
    case "$f" in
      */main.rs) f="${f%/main.rs}" ;;
      *) f="${f%.rs}" ;;
    esac
    printf '%s\n' "${f##*/}"
  done | LC_ALL=C sort
}

check_o() {
  local rc=0 void=0 step prover floors entry key rest name floor expected actual malformed duplicated
  prover="$(lane_o_prover)"
  floors="$(lane_o_floor_set "$prover")"
  if [ -n "$floors" ]; then
    echo "  prover: targo $prover — proved floors: $floors"
  else
    echo "  prover: targo ${prover:-<unparsed>} — NO proved floors are recorded for it; every other check runs, the floors cannot"
  fi

  # Every recorded set must name exactly the binaries Cargo builds, so a new
  # binary cannot go ungated under any prover — and must be WELL-FORMED: the
  # library's own floor, each target exactly once, every floor a positive
  # integer. A set that omitted `lib:` used to gate the library at the
  # unmeasured-prover fallback of 1 (planted 2026-10-08: 5 proved of a typical
  # 270 read as PASS), and a mistyped floor (`lib:2OO`) made the comparison a
  # bash error that an `if` reads as false. Main's single required
  # LANE_O_LIB_PROVED_FLOOR had neither state; the table must not add them.
  actual="$(lane_o_actual_bins)"
  while read -r key rest; do
    [ -n "$key" ] || continue
    expected="$(for entry in $rest; do
      [ "${entry%%:*}" = lib ] || printf '%s\n' "${entry%%:*}"
    done | LC_ALL=C sort)"
    if [ "$expected" != "$actual" ]; then
      echo "  FAIL  o/bins: prover $key's floor set does not match crates/ny-cert/src/bin —"
      echo "        lane: $(printf '%s' "$expected" | tr '\n' ' ')"
      echo "        tree: $(printf '%s' "$actual" | tr '\n' ' ')"
      echo "        a binary without its own proved floor would go ungated; fix LANE_O_FLOOR_SETS"
      rc=1
    fi
    malformed=""
    case " $rest " in
      *" lib:"*) : ;;
      *) malformed="$malformed [no lib: floor]" ;;
    esac
    for entry in $rest; do
      case "$entry" in
        *:*:*|:*|*:) malformed="$malformed [$entry]" ;;
        *:*)
          case "${entry#*:}" in
            *[!0-9]*|0*) malformed="$malformed [$entry]" ;;
          esac
          ;;
        *) malformed="$malformed [$entry]" ;;
      esac
    done
    duplicated="$(for entry in $rest; do printf '%s\n' "${entry%%:*}"; done | LC_ALL=C sort | uniq -d | tr '\n' ' ')"
    if [ -n "$duplicated" ]; then
      malformed="$malformed [more than one floor for: ${duplicated% }]"
    fi
    if [ -n "$malformed" ]; then
      echo "  FAIL  o/floors: prover $key's floor set is malformed:$malformed"
      echo "        every set needs lib:<n> and <binary>:<n> once each, n a positive integer; fix LANE_O_FLOOR_SETS"
      rc=1
    fi
  done <<EOF_SETS
$LANE_O_FLOOR_SETS
EOF_SETS

  # Under an unmeasured prover each run still needs a non-zero proved count:
  # zero proved over hundreds of obligations is a dark solver under any prover.
  # A run that targo refused to start (no trustd memory authority on this
  # host) returns GATE_VOID_RC with its reason recorded; every other non-zero
  # is a FAIL, and a FAIL on any target wins over a VOID on another.
  # The fallback floor of 1 is the UNMEASURED prover's only. Under a recorded
  # prover a target with no floor of its own gets the empty floor, which
  # lane_o_run refuses (the set check above has already named it).
  local unmeasured_floor=""
  [ -n "$floors" ] || unmeasured_floor=1
  floor="$(lane_o_floor "$floors" lib)"
  step=0
  lane_o_run lib "${floor:-$unmeasured_floor}" --lib || step=$?
  if [ "$step" -eq "$GATE_VOID_RC" ]; then void=1; elif [ "$step" -ne 0 ]; then rc=1; fi
  for name in $actual; do
    floor="$(lane_o_floor "$floors" "$name")"
    step=0
    lane_o_run "bin:$name" "${floor:-$unmeasured_floor}" --bin "$name" || step=$?
    if [ "$step" -eq "$GATE_VOID_RC" ]; then void=1; elif [ "$step" -ne 0 ]; then rc=1; fi
  done
  if [ "$rc" -ne 0 ]; then
    return "$rc"
  fi
  if [ -z "$floors" ]; then
    lane_void "lane [o] proved floors: none measured under prover targo ${prover:-<unparsed>} (recorded: $(printf '%s\n' "$LANE_O_FLOOR_SETS" | awk 'NF { printf "%s ", $1 }')) — 0 refuted rows, but proved counts NOT compared"
    void=1
  fi
  if [ "$void" -ne 0 ]; then
    return "$GATE_VOID_RC"
  fi
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
run_check l "activation-math certified-bound pins (silu) + scalar relaxation envelopes" check_l
run_check m "ny-api GPU-facade tier (11 tests lane [k] cannot reach)" check_m
run_check n "score-at-risk drift (yaml vs bank vs scorer lists)" check_n
run_check o "ny-cert verified lane, advisory ramp (targo trust check --allow-l0-gaps --lib, then --bin per binary)" check_o

echo ""
echo "==================== GATE SCOREBOARD ===================="
for line in ${RESULTS[@]+"${RESULTS[@]}"}; do
  echo "$line"
done
echo "---------------------------------------------------------"
if [ "$FAILURES" -gt 0 ]; then
  echo "GATE: FAIL ($PASSES pass, $FAILURES fail, $VOIDS void, $SKIPS skipped)"
  exit 1
fi
if [ "$VOIDS" -gt 0 ]; then
  echo "GATE: VOID ($PASSES pass, $FAILURES fail, $VOIDS void, $SKIPS skipped) — the VOID lanes did not check what they name; this is not a pass"
  exit "$GATE_VOID_RC"
fi
echo "GATE: PASS ($PASSES pass, $FAILURES fail, $VOIDS void, $SKIPS skipped)"
