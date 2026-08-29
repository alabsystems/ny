#!/usr/bin/env bash
# check_github_credential.sh — refuse in seconds when a broken GitHub token in
# the environment will void the whole workspace, instead of discovering it at
# cargo dependency resolution minutes later.
#
# WHY THIS EXISTS
# ---------------
# Measured on this host 2026-08-28, every line below reproduced literally.
#
# `crates/ny-mip/Cargo.toml` depends on `ay-milp` from a PRIVATE repository:
#
#     ay-milp = { version = "0.18.0",
#                 git = "https://github.com/alabsystems/ay.git", rev = "0c987ad3..." }
#
# so EVERY crate in the workspace needs that fetch to succeed before a single
# test compiles. When it fails, cargo exits at dependency resolution and ZERO
# tests run — a 41-minute gate run produced no test results at all and reported
# every assertion as VOID. The failure looks like a broken tree; it is a broken
# credential.
#
# THE CHAIN, each link verified here rather than assumed:
#
#   1. `~/.gitconfig` routes github.com to `gh`, and RESETS every other helper:
#
#          [credential "https://github.com"]
#              helper =
#              helper = !/opt/homebrew/bin/gh auth git-credential
#
#      An empty `helper =` is git's list-reset directive, so the osxkeychain
#      helper inherited from the Command Line Tools gitconfig is discarded.
#      GIT_TRACE=1 on `git ls-remote` confirms it — one helper spawns:
#          trace: run_command: '/opt/homebrew/bin/gh auth git-credential get'
#      and osxkeychain spawns ZERO times. NOTE FOR THE NEXT READER: the shadowing
#      is NOT git-helper-over-keychain, as was first assumed. It happens INSIDE
#      gh, between the env token and gh's own keyring:
#          with token:    X Failed to log in to github.com using token (GITHUB_TOKEN)
#                         ✓ Logged in to github.com account ... (keyring)
#                         - Active account: false
#          without token: ✓ Logged in ... (keyring)   - Active account: true
#      The remedy is the same either way; the reason is not, and the wrong
#      reason sends the next person to edit the wrong file.
#
#   2. `.cargo/config.toml` sets `[net] git-fetch-with-cli = true`, so cargo
#      shells out to the git CLI and inherits all of the above.
#
#   3. The cache is COLD, which is what makes this total rather than
#      intermittent. In ~/.cargo/git/db/ay-c43f9a767e4727fc the pinned rev is
#      absent — and the path is PROVEN RESOLVED with a positive control, because
#      a zero from a typo and a zero from an absent object look identical:
#          rev-parse --is-bare-repository  -> true
#          count-objects -v                -> in-pack: 109966, packs: 24
#          cat-file -t 0c987ad3<pinned>     -> fatal: could not get object info
#          cat-file -t <a rev that IS here> -> commit
#      No vendor/ dir, no source replacement, no `offline` key, CARGO_NET_OFFLINE
#      unset. So a network fetch is MANDATORY.
#
#   4. That fetch fails instantly and reversibly:
#          ls-remote, token set    -> rc=128, "Invalid username or token."
#          ls-remote, token unset  -> rc=0,   45eb9771...  HEAD
#
# TWO THINGS THAT SHAPE THIS SCRIPT
#
#   (a) GH_TOKEN breaks it identically AND OUTRANKS GITHUB_TOKEN. Proven with a
#       fake non-secret value: GH_TOKEN=ghp_000... alone gives the same rc=128,
#       and with both set `gh auth status` names GH_TOKEN as the failing one. A
#       check that inspects only GITHUB_TOKEN is a HALF-CHECK. In-repo precedent
#       agrees: scripts/vnncomp_trust_linux_build.sh unsets GH_PAT, GH_TOKEN and
#       GITHUB_TOKEN together.
#
#   (b) EXIT CODE ALONE CANNOT CLASSIFY. Both failure shapes are rc=128:
#          offline      rc=128  fatal: unable to access ...: Could not resolve host: ...
#          invalid cred rc=128  remote: Invalid username or token.
#       So classification branches on the MESSAGE, never on the exit code. This
#       is what lets "no network" stay a WARNING while "bad credential" is a
#       hard refusal.
#
# SECRET HYGIENE IS PART OF THE CONTRACT, not a nicety: a bad shell expansion
# leaked a token into a transcript during this investigation. This script reads
# presence with ${VAR+x} and length with ${#VAR}; it NEVER expands a token value
# into a command line, a log line, or an error, and it pipes all probe output
# through redact() first — git can echo a URL-embedded credential, and `gh auth
# status` prints a `gho_`-prefixed prefix of its own.
#
# EXIT CODES
#   0  healthy, or a warning (offline / indeterminate) — callers proceed
#   3  authentication failed — callers must REFUSE
#   2  usage error
#
# KNOWN, NAMED GAP: the fast path returns 0 without probing when no token is
# set, so an expired KEYRING credential is not detected. That is deliberate —
# the healthy case must cost nothing, and this script targets the documented
# env-token failure. A no-token host resolves via the keyring in 0.624-0.910s
# (3 runs), so the fast path is not hiding a common failure.

set -uo pipefail

SELF="check_github_credential.sh"

# Redact anything token-shaped before it reaches a terminal. Two families:
# GitHub's own prefixes (github_pat_/ghp_/gho_/ghu_/ghs_/ghr_), and a
# credential embedded in a URL as https://user:secret@host, which git prints
# verbatim in some failure messages.
redact() {
  sed -E \
    -e 's/(github_pat|ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]+/<redacted>/g' \
    -e 's#(https?://)[^/[:space:]@]+@#\1<redacted>@#g'
}

# ---------------------------------------------------------------------------
# classify — the whole decision, as a PURE function of (rc, message text).
#
# Kept free of I/O so --self-test can drive it with the canned strings captured
# during the investigation, with no network at all. Echoes exactly one of:
# ok | auth-failed | offline | indeterminate
#
# ORDERING IS DELIBERATE. Network patterns are tested BEFORE auth patterns, and
# only UNAMBIGUOUS ones are listed. The tie-break question is "which mistake do
# we prefer?": wrongly refusing an offline developer blocks all work and gets
# the check switched off, while wrongly warning on a bad credential costs the
# box slot we already pay today. So ambiguity resolves toward the warning.
# `unable to access` is deliberately in NEITHER list — it appears in offline
# messages and in HTTP-error messages alike, so it falls through to
# indeterminate, which also warns.
classify() {
  local rc="$1" text="$2"

  [ "$rc" -eq 0 ] && { printf 'ok'; return 0; }

  # 124 is this script's own watchdog verdict (see run_bounded). A probe that
  # hangs is a network problem, not a credential verdict.
  [ "$rc" -eq 124 ] && { printf 'offline'; return 0; }

  case "$text" in
    *"Could not resolve host"*|\
    *"Temporary failure in name resolution"*|\
    *"Failed to connect"*|\
    *"Connection timed out"*|\
    *"Connection refused"*|\
    *"Network is unreachable"*|\
    *"Operation timed out"*|\
    *"gnutls_handshake"*|\
    *"SSL connect error"*)
      printf 'offline'; return 0 ;;
  esac

  case "$text" in
    *"Invalid username or token"*|\
    *"Authentication failed"*|\
    *"could not read Username"*|\
    *"terminal prompts disabled"*|\
    *"Support for password authentication was removed"*|\
    *"The requested URL returned error: 401"*|\
    *"The requested URL returned error: 403"*)
      printf 'auth-failed'; return 0 ;;
  esac

  printf 'indeterminate'
}

# ---------------------------------------------------------------------------
# run_bounded <seconds> <command...> — portable watchdog.
#
# Does NOT use coreutils `timeout`: gate.sh documents Windows Git Bash support
# and `timeout` here is homebrew coreutils, not a guarantee. Uses a background
# job plus a background killer that leaves a marker file, so a timeout is
# reported as rc=124 UNAMBIGUOUSLY. Reading it off the signal exit status would
# not do: git's own failure status is 128, and 128 is exactly the value a naive
# "rc >= 128 means signalled" test would collide with.
#
# Writes combined output to $PROBE_OUT.
run_bounded() {
  local limit="$1"; shift
  local mark="$PROBE_OUT.timeout"
  rm -f "$mark"
  : >"$PROBE_OUT"

  "$@" >"$PROBE_OUT" 2>&1 &
  local job=$!
  (
    # `sleep` then check: if the job is still alive at the deadline, mark and
    # kill. TERM first so git can unwind, KILL as the backstop.
    sleep "$limit"
    if kill -0 "$job" 2>/dev/null; then
      : >"$mark"
      kill -TERM "$job" 2>/dev/null || true
      sleep 1
      kill -KILL "$job" 2>/dev/null || true
    fi
  ) >/dev/null 2>&1 &
  local killer=$!

  local rc=0
  # stderr is suppressed around the reaps ONLY. When the watchdog kills the
  # probe, bash announces the reap itself — "line N: 12345 Terminated: 15 ..." —
  # which is shell bookkeeping, not a finding, and printing it next to a
  # credential message reads like a crash. The probe's OWN output is captured in
  # $PROBE_OUT and is untouched by this.
  { wait "$job" || rc=$?; } 2>/dev/null
  # The killer is either finished or parked in `sleep`; terminating it keeps the
  # common path from waiting out the full limit.
  { kill -TERM "$killer" 2>/dev/null || true; } 2>/dev/null
  { wait "$killer" || true; } 2>/dev/null

  if [ -f "$mark" ]; then
    rm -f "$mark"
    return 124
  fi
  return "$rc"
}

# ---------------------------------------------------------------------------
# Token inspection. PRESENCE and LENGTH only — a value never leaves this scope.
#
# gh consults GH_TOKEN before GITHUB_TOKEN (measured: with both set, `gh auth
# status` names GH_TOKEN as the failing one), so report them in that order.
token_len() {
  local value="${!1-}"
  printf '%s' "${#value}"
}

ACTIVE_VAR=""
describe_tokens() {
  local name
  for name in GH_TOKEN GITHUB_TOKEN; do
    if [ -n "${!name+x}" ] && [ -n "${!name-}" ]; then
      printf '  %-13s set (%s chars)\n' "$name:" "$(token_len "$name")"
      [ -z "$ACTIVE_VAR" ] && ACTIVE_VAR="$name"
    else
      printf '  %-13s unset\n' "$name:"
    fi
  done
}

# Echo the probe's own words, redacted and prefixed. Silent when the probe left
# nothing — a watchdog kill produces no output, and an empty indented block
# under a WARNING line reads as truncated output rather than as "no output".
emit_probe_text() {
  # NOT `[ -n "${1//[[:space:]]/}" ]`: bash pattern substitution over a large
  # capture is quadratic, and a chatty git stderr made this hang >48s. `case`
  # short-circuits on the first non-space byte.
  case "$1" in *[![:space:]]*) ;; *) return 0 ;; esac
  printf '%s\n' "$1" | redact | sed "s/^/$PREFIX   /" >&2
}

remedy() {
  echo "  REMEDY: unset ${ACTIVE_VAR:-GITHUB_TOKEN} (and the other of GH_TOKEN/GITHUB_TOKEN)," >&2
  echo "          or refresh the PAT if you actually need it. The keyring credential" >&2
  echo "          underneath already works: with both popped, ls-remote returned rc=0." >&2
  echo "          One-off:  env -u GITHUB_TOKEN -u GH_TOKEN <your command>" >&2
  echo "  WHY:    ~/.gitconfig routes github.com to \`gh auth git-credential\` and RESETS" >&2
  echo "          every other helper, so gh is the sole credential source; gh prefers the" >&2
  echo "          env token over its keyring. osxkeychain is not in the chain at all." >&2
  echo "          .cargo/config.toml sets git-fetch-with-cli=true, so cargo inherits this," >&2
  echo "          and crates/ny-mip's private ay-milp git dep makes it fatal WORKSPACE-WIDE." >&2
}

# ---------------------------------------------------------------------------
# --self-test — drives classify() with the literal strings captured on this host
# during the 2026-08-28 investigation. NO NETWORK. Prints a case count so a
# run that matched nothing is distinguishable from a run that passed, the same
# discipline gate.sh lanes [j] and [k] apply to their test filters.
self_test() {
  local failures=0 cases=0
  check_case() {
    local label="$1" rc="$2" want="$3" text="$4" got
    cases=$((cases + 1))
    got="$(classify "$rc" "$text")"
    if [ "$got" != "$want" ]; then
      echo "  SELFTEST FAIL $label: want=$want got=$got" >&2
      failures=$((failures + 1))
    fi
  }

  # Literal ls-remote output, token set (rc=128).
  check_case invalid-token 128 auth-failed \
"remote: Invalid username or token. Password authentication is not supported for Git operations.
fatal: Authentication failed for 'https://github.com/alabsystems/ay.git/'"
  # Literal ls-remote output, unresolvable host (rc=128 — SAME rc as above).
  check_case offline-dns 128 offline \
"fatal: unable to access 'https://nonexistent.invalid/x/y.git/': Could not resolve host: nonexistent.invalid"
  # Literal ls-remote output, tokens popped (rc=0).
  check_case healthy 0 ok \
"45eb97713346d8d8164b77a335057a166ae881cc	HEAD"
  # run_bounded's watchdog verdict.
  check_case watchdog-timeout 124 offline ""
  # GIT_TERMINAL_PROMPT=0 with no usable credential.
  check_case no-credential 128 auth-failed \
"fatal: could not read Username for 'https://github.com': terminal prompts disabled"
  check_case http-403 128 auth-failed \
"fatal: unable to access 'https://github.com/x/y.git/': The requested URL returned error: 403"
  check_case offline-connect 128 offline \
"fatal: unable to access 'https://github.com/x/y.git/': Failed to connect to github.com port 443: Network is unreachable"
  # An rc=0 verdict must not be overridden by scary-looking text on stderr.
  check_case ok-beats-text 0 ok "remote: Invalid username or token."
  # Unrecognised failure text warns rather than refusing — the ambiguity
  # tie-break documented above classify().
  check_case unknown-text 128 indeterminate \
"fatal: unable to access 'https://github.com/x/y.git/': something entirely new"

  if [ "$failures" -ne 0 ]; then
    echo "SELFTEST FAILED $failures of $cases cases" >&2
    return 1
  fi
  echo "SELFTEST ok $cases cases"
}

# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTEXT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --self-test) self_test; exit $? ;;
    # `shift 2` when only one argument remains is a NO-OP that returns non-zero
    # (there is no `set -e` here), and the `continue` then re-reads the same
    # argument forever. Measured 2026-08-28: `--context` as the final argument
    # HUNG until killed (rc=124 under a 5s timeout). Require the value.
    --context)
      [ $# -ge 2 ] || { echo "$SELF: --context requires a value" >&2; exit 2; }
      CONTEXT="$2"; shift 2; continue ;;
    -h|--help)
      echo "usage: $SELF [--context <label>] [--self-test]"
      echo "  exit 0 = ok or warning (offline/indeterminate); exit 3 = authentication failed"
      exit 0
      ;;
    *) echo "$SELF: unknown argument: $1" >&2; exit 2 ;;
  esac
done

PREFIX="[credential]"
[ -n "$CONTEXT" ] && PREFIX="[credential:$CONTEXT]"

have_token=0
for name in GH_TOKEN GITHUB_TOKEN; do
  if [ -n "${!name+x}" ] && [ -n "${!name-}" ]; then
    have_token=1
    [ -z "$ACTIVE_VAR" ] && ACTIVE_VAR="$name"
  fi
done

# FAST PATH: the healthy case pays NOTHING. No token means nothing can shadow
# the keyring, so there is no probe and no network round trip.
if [ "$have_token" -eq 0 ]; then
  echo "$PREFIX GH_TOKEN/GITHUB_TOKEN unset — no env token can shadow the gh keyring. ok"
  exit 0
fi

# Derive the probe URL from the real dependency rather than pinning a second
# copy of it here: crates/ny-mip/Cargo.toml is what cargo actually fetches, and
# a hardcoded duplicate would go stale the next time the dep moves. (It already
# moved once during this investigation: v0.15.0/rev 6228729e at an older HEAD
# vs v0.18.0/rev 0c987ad3 at the current one.)
MIP_MANIFEST="$REPO_ROOT/crates/ny-mip/Cargo.toml"
PROBE_URL=""
if [ -r "$MIP_MANIFEST" ]; then
  PROBE_URL="$(awk -F'git = "' '/^ay-milp[[:space:]]*=/ { split($2, p, "\""); print p[1]; exit }' \
    "$MIP_MANIFEST" 2>/dev/null || true)"
fi

echo "$PREFIX an environment token is set; probing whether it authenticates."
describe_tokens

if [ -z "$PROBE_URL" ]; then
  # Cannot probe, so cannot claim the credential is bad. Warn: refusing here
  # would block the whole gate over an unparsed manifest.
  echo "$PREFIX WARNING: could not derive the ay-milp git URL from $MIP_MANIFEST." >&2
  echo "$PREFIX Skipping the probe. If cargo dies at dependency resolution, this is why." >&2
  exit 0
fi

PROBE_OUT="$(mktemp "${TMPDIR:-/tmp}/ny-credential.XXXXXX")"
trap 'rm -f "$PROBE_OUT" "$PROBE_OUT.timeout"' EXIT

# 8s cap against a probe measured over 3 runs each on this host 2026-08-28:
# 0.356-0.479s failing, 0.624-0.910s succeeding. That is ~9x headroom on the
# slowest sample, and bounded so a black-holed network cannot hang the gate.
# GIT_TERMINAL_PROMPT=0 is what stops git from parking on an interactive
# username prompt when the helper yields nothing. GIT_ASKPASS is belt-and-braces
# for the case where a GUI askpass would otherwise pop up; pointing it at a path
# that does not exist is harmless — measured 2026-08-28, GIT_ASKPASS=/nonexistent
# gives byte-identical output to omitting it — so this stays safe on a host
# without /usr/bin/false.
PROBE_LIMIT="${NY_CREDENTIAL_PROBE_TIMEOUT:-8}"
run_bounded "$PROBE_LIMIT" env GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/usr/bin/false \
  git ls-remote "$PROBE_URL" HEAD
probe_rc=$?
probe_text="$(cat "$PROBE_OUT" 2>/dev/null || true)"

verdict="$(classify "$probe_rc" "$probe_text")"

case "$verdict" in
  ok)
    echo "$PREFIX ${ACTIVE_VAR} authenticates against ${PROBE_URL}. ok"
    exit 0
    ;;
  offline)
    # Requirement, not an oversight: no network is a WARNING. A developer on a
    # plane must still be able to run everything that does not need the network.
    echo "$PREFIX WARNING: no network reachable, so the token could not be validated." >&2
    emit_probe_text "$probe_text"
    echo "$PREFIX Proceeding. A cargo step needing the ay-milp fetch will still fail." >&2
    exit 0
    ;;
  auth-failed)
    echo "$PREFIX REFUSING: ${ACTIVE_VAR} is set and DOES NOT AUTHENTICATE." >&2
    emit_probe_text "$probe_text"
    echo "$PREFIX Every cargo command in this workspace will die at dependency" >&2
    echo "$PREFIX resolution — crates/ny-mip pulls ay-milp from that private repo —" >&2
    echo "$PREFIX so ZERO tests would run and every assertion would be VOID." >&2
    remedy
    exit 3
    ;;
  *)
    echo "$PREFIX WARNING: probe failed in a way this check does not recognise." >&2
    emit_probe_text "$probe_text"
    echo "$PREFIX Proceeding, but a cargo failure at dependency resolution is likely." >&2
    exit 0
    ;;
esac
