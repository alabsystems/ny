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
#   a  clippy drift gate    cargo clippy --workspace --exclude ny-python
#                           --all-targets -- -D warnings, minus the crates in
#                           CLIPPY_SKIPPED_CRATES (printed as SKIPPED + reason)
#   b  ny-cert tests        cargo test -p ny-cert --all-targets
#   c  ny CLI unit tests    cargo test -p ny-cli --bin ny vnncomp_submit
#                           && cargo test -p ny-cli --bin ny cert
#   d  ny-propagate units   cargo test -p ny-propagate --lib collection
#                           && cargo test -p ny-propagate --lib conv2d
#   e  harness pytest       python3 -m pytest tests/test_run_instance_preset_resolution.py -q
#   f  submission           regenerate a tarball to a temp path (debug ny
#      invariants           binary, --no-build) and assert the packaging
#                           soundness invariants (see check_f below)
#
# Flags:
#   --fast        skip the heavy suites and packaging (runs a, b, c only)
#   --only <x>    run a single check by letter (a-f)
#   -h | --help   usage
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Crates excluded from the clippy drift gate (check a), with reasons.
# Probed 2026-07-20 under exactly:
#   cargo clippy --workspace --exclude ny-python --all-targets -- -D warnings
# Re-probe with that command and prune this list as the drift is paid down;
# every entry here is un-linted debt, not a permanent carve-out.
# Empty since the 2026-07-20 env-wall migration completed: every crate's raw
# set_var/remove_var sites were routed through the blessed lock helpers
# (ny-test-utils env choke point; ny-mip ay_env), so the whole workspace runs
# under check [a]'s -D warnings. Add an entry ("crate|reason") ONLY for a
# crate with a known, dated, deliberately-unfixed warning — never to hide
# fresh drift.
CLIPPY_SKIPPED_CRATES=()

ALL_CHECKS="a b c d e f"
FAST_CHECKS="a b c"

usage() {
  sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
    a|b|c|d|e|f) ;;
    *)
      echo "gate.sh: --only takes a single check letter a-f, got: '$ONLY'" >&2
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
# a) clippy drift gate
# ---------------------------------------------------------------------------
check_a() {
  local args=(clippy --workspace --exclude ny-python --all-targets)
  local entry crate reason
  # ${arr[@]+...} guard: macOS bash 3.2 under `set -u` treats an EMPTY array
  # expansion as unbound (same idiom as the RESULTS loop below).
  for entry in ${CLIPPY_SKIPPED_CRATES[@]+"${CLIPPY_SKIPPED_CRATES[@]}"}; do
    crate="${entry%%|*}"
    reason="${entry#*|}"
    echo "SKIPPED crate $crate: $reason"
    args+=(--exclude "$crate")
  done
  echo "+ cargo ${args[*]} -- -D warnings"
  cargo "${args[@]}" -- -D warnings
}

# ---------------------------------------------------------------------------
# b) ny-cert test suite
# ---------------------------------------------------------------------------
check_b() {
  echo "+ cargo test -p ny-cert --all-targets"
  cargo test -p ny-cert --all-targets
}

# ---------------------------------------------------------------------------
# c) ny CLI unit tests: submission packer + cert
# ---------------------------------------------------------------------------
check_c() {
  echo "+ cargo test -p ny-cli --bin ny vnncomp_submit"
  cargo test -p ny-cli --bin ny vnncomp_submit
  echo "+ cargo test -p ny-cli --bin ny cert"
  cargo test -p ny-cli --bin ny cert
}

# ---------------------------------------------------------------------------
# d) ny-propagate unit slices
# ---------------------------------------------------------------------------
check_d() {
  echo "+ cargo test -p ny-propagate --lib collection"
  cargo test -p ny-propagate --lib collection
  # The conv2d slice contains scoped tests for the process-global
  # NY_DENSE_BUDGET_MB knob. Run this slice serially so a deliberate
  # zero-budget scope cannot alter an unrelated test's allocation policy.
  echo "+ cargo test -p ny-propagate --lib conv2d -- --test-threads=1"
  cargo test -p ny-propagate --lib conv2d -- --test-threads=1
}

# ---------------------------------------------------------------------------
# e) harness preset-resolution pytest
# ---------------------------------------------------------------------------
check_e() {
  echo "+ python3 -m pytest tests/test_run_instance_preset_resolution.py -q"
  python3 -m pytest tests/test_run_instance_preset_resolution.py -q
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
  target_dir="$(cargo metadata --format-version 1 --no-deps | python3 -c \
    'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
  bin="$target_dir/debug/ny"
  if [ ! -x "$bin" ]; then
    echo "debug ny binary absent — building it"
    echo "+ cargo build -p ny-cli --bin ny"
    cargo build -p ny-cli --bin ny
  else
    echo "using existing debug binary: $bin"
  fi

  local tmpdir tarball listing
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/ny-gate-sub.XXXXXX")"
  # Each check runs in its own subshell (see run_check), so this EXIT trap is
  # scoped to check_f and fires on error exits as well as normal returns.
  # shellcheck disable=SC2064 -- expand tmpdir now, not at trap time
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
  if python3 - "$tarball" <<'PY'
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

MODE="full"
[ "$FAST" -eq 1 ] && MODE="fast (--fast)"
[ -n "$ONLY" ] && MODE="single check (--only $ONLY)"
echo "ny drift gate — mode: $MODE — repo: $REPO_ROOT"

run_check a "clippy drift gate (workspace, -D warnings)" check_a
run_check b "ny-cert test suite (--all-targets)" check_b
run_check c "ny CLI unit tests (vnncomp_submit + cert)" check_c
run_check d "ny-propagate unit slices (collection + conv2d)" check_d
run_check e "harness preset-resolution pytest" check_e
run_check f "submission-packaging invariants" check_f

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
