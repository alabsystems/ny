#!/usr/bin/env bash
# ny_box.sh — mutual exclusion + quiescence gate for the measurement box.
#
# WHY THIS EXISTS
# ---------------
# On 2026-08-22 four agents worked the same box concurrently. A scored 100 s
# cifar100 closure measurement ran while another agent held `cargo build
# --release`; 1-min loadavg was 9.62 against a documented precondition of < 3.
#
# That combination does not merely add noise, it manufactures a specific WRONG
# ANSWER. cifar100's failing rows are wallclock-bound at ~1.15 s/expansion, and
# the closure metric is a COUNT (expansions, closed domains, open-ratio). Steal
# cores and the run does strictly less work in the same wall, so the contaminated
# arm reports fewer expansions and fewer closures — which reads exactly like
# "the mechanism did nothing". The campaign has already banked one vacuous null
# on that mechanism for a different reason (it was never armed on this lane).
# Two vacuous nulls on the same lever would retire a live mechanism for good.
#
# Verdict A/Bs are much more robust to this; RATE and COUNT measurements are not.
# So the rule is not "be careful", it is "hold the lock".
#
# USAGE
#   scripts/ny_box.sh measure [--budget N] -- <command...>
#       Exclusive lock + quiescence wait + load recording. For any scored run.
#   scripts/ny_box.sh build -- <command...>
#       Exclusive lock, NO quiescence wait (a build need not wait for idle, it
#       only needs to not overlap a measurement).
#   scripts/ny_box.sh status
#       Who holds the lock, current load, live ny processes.
#
# When the command names cargo/targo, a github.com credential check runs BEFORE
# the lock wait (see below) and refuses if an env token cannot fetch ay-milp.
# NY_SKIP_CREDENTIAL_CHECK=1 disables it.
#
# The lock is advisory and process-scoped via flock(1), or the same kernel
# flock through Python's fcntl module where the flock CLI is unavailable. It is
# released automatically if the holder dies, so a crashed agent cannot wedge
# the box.
set -uo pipefail

LOCK_DIR="${NY_BOX_LOCK_DIR:-/tmp/claude-1000/ny-box}"
LOCK="$LOCK_DIR/box.lock"
HOLDER="$LOCK_DIR/holder"
mkdir -p "$LOCK_DIR" 2>/dev/null || true

# Max seconds to wait for the lock. A measurement sweep can legitimately hold the
# box for a long time, so this is generous; exceeding it is an error, not a
# reason to barge in.
WAIT_LOCK="${NY_BOX_WAIT_LOCK:-5400}"
# Max seconds to wait for quiescence once the lock is held. If the box will not
# settle we FAIL rather than measure hot — a refused run is recoverable, a
# silently contaminated one is not.
WAIT_QUIET="${NY_BOX_WAIT_QUIET:-900}"
# Seconds of quiescence wait permitted WHILE HOLDING the lock. The long wait now
# happens BEFORE the lock (see wait_for_quiescence), so this is only a short
# grace to absorb a blip between the pre-lock check and the lock acquisition.
# Keeping it small is what stops a never-quiescent host from starving builds.
QUIET_GRACE="${NY_BOX_QUIET_GRACE:-120}"
LOAD_MAX="${NY_BOX_LOAD_MAX:-3}"
# How "busy" is judged once no builder and no ny process is live. `cpu` (the
# default on Linux) measures busy CPU-cores DIRECTLY over a short window from
# /proc/stat deltas (user+nice+system+irq+softirq+steal+iowait, i.e. iowait counts
# as busy); `loadavg` is the historical 1-minute average. The 1-minute average is
# a LAGGING proxy: after any 20-thread ny row it stays above LOAD_MAX for 2-4
# minutes while the CPU is ~98% idle. Measured 2026-09-28: W6B's 2-12 s rows
# started 5 minutes apart with the box idle, so ~95% of its queue time was
# waiting for a number to decay. The process checks above are the real guard
# against contention; the busy-core window measures the same quantity loadavg
# approximates, without the lag. Hosts without /proc/stat fall back to loadavg.
QUIET_PROBE="${NY_BOX_QUIET_PROBE:-cpu}"
QUIET_WINDOW="${NY_BOX_QUIET_WINDOW:-5}"
PROC_STAT="${NY_BOX_PROC_STAT:-/proc/stat}"   # test hook: a file (or FIFO) standing in for /proc/stat

log() { printf '[ny-box] %s\n' "$*" >&2; }

load_snapshot() {
  if [ -r /proc/loadavg ]; then
    cat /proc/loadavg
    return
  fi
  if command -v sysctl >/dev/null 2>&1; then
    local snapshot; snapshot=$(sysctl -n vm.loadavg 2>/dev/null || true)
    if [ -n "$snapshot" ]; then
      printf '%s\n' "$snapshot" | tr -d '{}' | awk '{$1=$1; print}'
      return
    fi
  fi
  uptime
}

load1() {
  if [ -r /proc/loadavg ]; then
    cut -d' ' -f1 /proc/loadavg
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n vm.loadavg 2>/dev/null | tr -d '{}' | awk '{print $1}'
  else
    uptime | sed 's/.*load averages*://' | awk '{print $1}' | tr -d ','
  fi
}
load1_int() { load1 | cut -d. -f1; }

# Busy CPU-cores x100 over QUIET_WINDOW seconds, from two /proc/stat samples.
# Prints nothing (caller falls back to loadavg) when the aggregate line cannot be
# read twice. iowait and steal count as busy: a box stalled on I/O is not quiet.
cpu_stat_sample() { # -> "busy total" jiffies from the aggregate 'cpu ' line
  awk '$1=="cpu" {b=$2+$3+$4+$6+$7+$8+$9; t=b+$5; print b, t; exit}' "$PROC_STAT" 2>/dev/null
}
busy_cores_x100() {
  local a b n
  a=$(cpu_stat_sample) || return 0
  [ -n "$a" ] || return 0
  sleep "$QUIET_WINDOW"
  b=$(cpu_stat_sample) || return 0
  [ -n "$b" ] || return 0
  n=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)
  awk -v a="$a" -v b="$b" -v n="$n" 'BEGIN {
    split(a, x, " "); split(b, y, " ");
    dt = y[2] - x[2]; db = y[1] - x[1];
    if (dt <= 0) exit;
    printf "%d\n", (db / dt) * n * 100 }'
}

timestamp() {
  date -Iseconds 2>/dev/null || date -u '+%Y-%m-%dT%H:%M:%SZ'
}

lock_is_free() {
  if command -v flock >/dev/null 2>&1; then
    ( flock -n 9 || exit 1 ) 9>>"$LOCK"
    return
  fi
  python3 - "$LOCK" <<'PY'
import fcntl
import sys

with open(sys.argv[1], "a+b") as lock:
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(1)
PY
}

# Foreign compute = anything that competes for cores which we did not start.
#
# NEVER use `-f` here: it matches this very script's own command line, a
# self-match trap that has cost this campaign three sessions (exit 144).
#
# But `-x ny` was ALSO wrong, and wrong in the direction that silently corrupts
# measurements. Every agent that follows the binary-provenance rule copies
# `target/release/ny` to a pinned name -- `ny_pin` is the one in use today --
# precisely because a concurrent `cargo build` would otherwise swap the binary
# mid-run-set. `-x` is an EXACT name match, so it sees none of them. Caught
# 2026-08-24 with `ny_pin` burning 714% CPU while this gate reported the box
# quiet and `measure` proceeded. The careful agents were exactly the invisible
# ones.
#
# Without -f, pgrep matches the pattern as a regex against the process NAME
# only, so `^ny` catches ny/ny_pin/ny-anything and cannot match this script
# (whose comm is `bash`).
ny_procs() { pgrep '^ny' 2>/dev/null | wc -l; }

# Name the offender, not just the category. A `measure` caller holds the lock
# while it waits here, so a BARE `cargo` (one not itself taking this lock) can
# burn the caller's whole quiescence allowance and force an exit 75 — the box
# stays locked and idle for 15 minutes and nothing is measured. That livelock
# was observed 2026-08-24. The waiting agent cannot fix it without knowing WHICH
# process to chase, so print pid + command line and let it escalate.
# BUILDER PROCESS NAMES. `rust-toolchain.toml` pins `channel = "trust"`, and every
# build in this repo is `rustup run trust targo --unverified ...`, which spawns
# `targo` and `trustc` -- NEVER `cargo`, NEVER `rustc`. Checking only the latter
# two made this gate blind to every build the campaign actually runs: measured
# 2026-09-15 at loadavg 17.58 with 20 `trustc` live, `pgrep -x cargo` and
# `pgrep -x rustc` BOTH returned 0. The only remaining guard was the 1-minute
# LOAD_MAX average, which lags, so a `measure` could be admitted onto a box that
# was about to be saturated -- and wall-clock IS the verdict on wall-limited
# benchmarks (84% of banked tinyimagenet proofs finish in the last 10% of budget).
# The verified-lane driver (`targo-trust`) and tippy's driver (`tippy-driver`)
# are Trust build processes too; cargo/rustc stay for any other toolchain.
# `-x` is an EXACT name match, so each name must be listed; never `-f` (the
# self-match trap that cost three sessions, exit 144).
BUILD_PROCS="targo targo-trust trustc tippy-driver cargo rustc"
builder_pids() { local p; for p in $BUILD_PROCS; do pgrep -x "$p" 2>/dev/null; done; }
builder_name() {
  local p
  for p in $BUILD_PROCS; do
    pgrep -x "$p" >/dev/null 2>&1 && { echo "$p"; return; }
  done
}

offenders() {
  local pids; pids=$(builder_pids; pgrep '^ny')
  [ -z "$pids" ] && return
  # shellcheck disable=SC2086
  ps -o pid=,ppid=,etime=,args= -p $pids 2>/dev/null | cut -c1-150 | sed 's/^/      /'
}

busy_reason() {
  local b; b=$(builder_name)
  [ -n "$b" ] && { echo "$b running"; return; }
  local n; n=$(ny_procs)
  [ "$n" -gt 0 ] && { echo "$n ny process(es) already live (incl. pinned copies)"; return; }
  if [ "$QUIET_PROBE" = cpu ]; then
    local c; c=$(busy_cores_x100)
    if [ -n "$c" ]; then
      [ "$c" -ge $(( LOAD_MAX * 100 )) ] && { echo "cpu busy $(( c / 100 )).$(printf '%02d' $(( c % 100 ))) cores >= $LOAD_MAX over ${QUIET_WINDOW}s (loadavg $(load1))"; return; }
      echo ""; return
    fi
  fi
  local l; l=$(load1_int)
  [ "$l" -ge "$LOAD_MAX" ] && { echo "loadavg $(load1) >= $LOAD_MAX"; return; }
  echo ""
}

cmd_status() {
  echo "lock:  $LOCK"
  if [ -f "$HOLDER" ]; then echo "held:  $(cat "$HOLDER" 2>/dev/null)"; else echo "held:  (no holder file)"; fi
  # Probe without blocking, in a subshell so the lock is dropped immediately if
  # we do get it: reporting status must never be able to take the box.
  if lock_is_free; then echo "state: FREE"; else echo "state: BUSY"; fi
  echo "load:  $(load_snapshot)"
  if [ "$QUIET_PROBE" = cpu ]; then echo "cpu:   $(busy_cores_x100 | awk '{printf "%.2f busy cores", $1/100}') over ${QUIET_WINDOW}s (probe=cpu, LOAD_MAX=$LOAD_MAX)"; fi
  echo "ny:    $(ny_procs) process(es) (matches pinned copies: ny, ny_pin, ...)"
  echo "build: $(builder_pids | wc -l) build process(es) ($BUILD_PROCS)"
}

original_args=("$@")
mode="${1:-}"; shift || true
case "$mode" in
  status) cmd_status; exit 0 ;;
  measure|build) ;;
  *) log "usage: ny_box.sh {measure|build|status} [--budget N] -- <command...>"; exit 2 ;;
esac

budget=""
while [ $# -gt 0 ]; do
  case "$1" in
    --budget) budget="${2:-}"; shift 2 ;;
    --) shift; break ;;
    *) log "unexpected argument before --: $1"; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { log "no command given after --"; exit 2; }

# ---------------------------------------------------------------------------
# GitHub credential precondition — BEFORE the lock, never after it.
#
# CLAUDE.md's own rule: "a 200 ms check must never be discovered after a
# 1912-second lock wait." An invalid GITHUB_TOKEN/GH_TOKEN in the environment
# overrides the working gh keyring credential, so `cargo` cannot fetch the
# private `ay-milp` git dependency that crates/ny-mip pins, and EVERY cargo
# command in this workspace dies at dependency resolution with zero tests run.
# Measured 2026-08-28: rc=128 "Invalid username or token" with the token set,
# rc=0 with it unset; the probe costs 0.356-0.479s over 3 runs and the no-token
# fast path costs 0.025-0.027s. Discovering that after waiting out a 57-row
# sweep costs the whole slot, and the failure reads as a broken tree.
#
# SCOPED TO CARGO/TARGO ON PURPOSE. `measure -- ./ny_pin …` needs no credential
# at all, and adding a network probe in front of every measurement would put a
# new failure mode (and new latency) on the path this script exists to protect.
# So the check runs only when argv actually names a build tool.
#
# Guarded on NY_BOX_FCNTL_LOCKED so the macOS python re-exec below — which runs
# this same script a second time with the lock already held — does not probe
# twice.
if [ "${NY_BOX_FCNTL_LOCKED:-0}" != 1 ] && [ "${NY_SKIP_CREDENTIAL_CHECK:-0}" != 1 ]; then
  needs_credential=0
  for word in "$@"; do
    # ${word##*/} strips any directory so a FULL-PATH invocation still matches:
    # `/opt/homebrew/bin/cargo` reduces to `cargo`. Without the strip the exact
    # arms below miss it entirely — verified 2026-08-28 by mutation, which is
    # also how the first version of this comment was caught being wrong. It had
    # claimed the strip was there to stop `~/.cargo/bin/ny_pin` being refused;
    # it is not, because the arms are exact or space-delimited and `.cargo`
    # inside a path never matched either way. The pinned-binary case is safe on
    # its own merits, and is pinned by a test regardless.
    case "${word##*/}" in
      cargo|cargo.exe|targo|targo.exe) needs_credential=1; break ;;
      # Runners in this campaign are written as `-- bash -c '…cargo…'`, where
      # the whole script is ONE argv word, so a basename match alone misses
      # them. CLAUDE.md records exactly such a runner: three steps that each
      # exited 127 on `cargo: command not found` inside an inner `bash -c`.
      "cargo "*|"targo "*|*" cargo "*|*" targo "*)
        needs_credential=1; break ;;
    esac
  done
  if [ "$needs_credential" -eq 1 ]; then
    NY_BOX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    credential_checker="$NY_BOX_ROOT/scripts/check_github_credential.sh"
    if [ -r "$credential_checker" ]; then
      credential_rc=0
      bash "$credential_checker" --context ny-box >&2 || credential_rc=$?
      if [ "$credential_rc" -eq 3 ]; then
        log "REFUSING before taking the box lock: the GitHub credential above"
        log "cannot fetch ay-milp, so this cargo command would burn the slot and"
        log "then die at dependency resolution with zero output."
        log "Set NY_SKIP_CREDENTIAL_CHECK=1 to proceed anyway (it will not help)."
        exit 2
      fi
    else
      # Deliberately a WARNING here where gate.sh refuses. gate.sh IS the drift
      # gate, so a missing tracked file is its business; ny_box.sh's job is
      # mutual exclusion, and refusing a build over a missing ADVISORY helper
      # would add a new way to lose box access without preventing any
      # contamination. What must not happen is losing the protection SILENTLY —
      # that is the no-op-that-reads-as-a-pass shape — so say so out loud.
      log "WARNING: $credential_checker is missing; running the credential"
      log "precondition was skipped. A bad GITHUB_TOKEN will now surface only"
      log "after the lock wait, at cargo dependency resolution."
    fi
  fi
fi

# macOS has the same advisory flock syscall but no flock(1). Acquire it in a
# tiny Python bootstrap, mark the descriptor inheritable, and exec this script
# again. The second invocation keeps that SAME descriptor for its entire
# lifetime, so cancellation/crash releases the lock exactly like flock(1).
if ! command -v flock >/dev/null 2>&1 && [ "${NY_BOX_FCNTL_LOCKED:-0}" != 1 ]; then
  exec python3 - "$LOCK" "$WAIT_LOCK" "$HOLDER" "$0" "${original_args[@]}" <<'PY'
import fcntl
import os
import sys
import time

lock_path, wait_raw, holder, script, *args = sys.argv[1:]
deadline = time.monotonic() + float(wait_raw)
lock = open(lock_path, "a+b", buffering=0)
while True:
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        if time.monotonic() >= deadline:
            try:
                with open(holder, encoding="utf-8", errors="replace") as source:
                    current = source.read().strip() or "unknown"
            except OSError:
                current = "unknown"
            print(
                "[ny-box] TIMEOUT after %ss waiting for the box lock.\n"
                "[ny-box] current holder: %s\n"
                "[ny-box] This is a REFUSAL, not a suggestion to run anyway. Re-queue your run."
                % (wait_raw, current),
                file=sys.stderr,
            )
            raise SystemExit(75)
        time.sleep(0.1)

# Bash reserves low descriptors while opening a script. Duplicate the lock to
# the same explicit descriptor the flock(1) path uses; the inner invocation
# knows not to reopen it, and every command it starts inherits the live lock.
lock_fd = 9
if lock.fileno() != lock_fd:
    os.dup2(lock.fileno(), lock_fd, inheritable=True)
else:
    os.set_inheritable(lock_fd, True)
environment = os.environ.copy()
environment["NY_BOX_FCNTL_LOCKED"] = "1"
environment["NY_BOX_FCNTL_LOCK_FD"] = str(lock_fd)
os.execvpe(script, [script, *args], environment)
PY
fi

if [ "${NY_BOX_FCNTL_LOCKED:-0}" = 1 ]; then
  # A redirection-free write probe: reopening FD 9 would replace and release
  # the inherited lock before the command starts.
  : >&9 2>/dev/null || { log "inherited fcntl lock descriptor is closed"; exit 75; }
else
  exec 9>>"$LOCK" || { log "cannot open lock $LOCK"; exit 1; }
fi

# Wait for quiescence. `budget` seconds; returns 0 when quiet, 1 on timeout.
# `held` is 1 when we already own the lock (changes only the diagnostics).
wait_for_quiescence() {
  local budget="$1" held="$2" q0 reason el
  q0=$(date +%s)
  while :; do
    reason="$(busy_reason)"
    [ -z "$reason" ] && return 0
    el=$(( $(date +%s) - q0 ))
    [ "$el" -ge "$budget" ] && { printf '%s' "$reason" >"$LOCK_DIR/.last_reason"; return 1; }
    if [ $(( el % 60 )) -lt 15 ]; then
      if [ "$held" = 1 ]; then
        log "waiting for quiescence WHILE HOLDING the lock (${el}s of ${budget}s grace): $reason"
      else
        log "waiting for quiescence BEFORE taking the lock (${el}s of ${budget}s): $reason"
      fi
      offenders >&2
    fi
    sleep 15
  done
}

# #box-quiesce-before-lock. A `measure` used to take the lock and THEN wait here
# for up to WAIT_QUIET. On a host with a permanent foreign load floor that wait
# never succeeds, so the caller held the box for its entire allowance, measured
# nothing, and exited 75 -- starving every queued `build` behind it. Two such
# holders (3h08m and 2h22m, nothing written) were cleared by hand on 2026-08-27.
# The long wait now happens with NO lock held, so a run that can never quiesce
# can never block anyone else.
if [ "$mode" = "measure" ]; then
  if ! wait_for_quiescence "$WAIT_QUIET" 0; then
    reason="$(cat "$LOCK_DIR/.last_reason" 2>/dev/null || echo unknown)"
    log "REFUSING TO MEASURE: box not quiescent after ${WAIT_QUIET}s ($reason)."
    log "No lock was taken, so nothing was starved. Re-queue your run."
    offenders >&2
    exit 75
  fi
fi

t0=$(date +%s)
if command -v flock >/dev/null 2>&1; then
  if ! flock -w "$WAIT_LOCK" 9; then
    log "TIMEOUT after ${WAIT_LOCK}s waiting for the box lock."
    log "current holder: $(cat "$HOLDER" 2>/dev/null || echo unknown)"
    log "This is a REFUSAL, not a suggestion to run anyway. Re-queue your run."
    exit 75   # EX_TEMPFAIL
  fi
elif [ "${NY_BOX_FCNTL_LOCKED:-0}" != 1 ] || [ -z "${NY_BOX_FCNTL_LOCK_FD:-}" ]; then
  log "no flock(1) or inherited fcntl lock is available; refusing to run"
  exit 75
fi
waited=$(( $(date +%s) - t0 ))
printf 'pid=%s mode=%s since=%s cmd=%s\n' "$$" "$mode" "$(timestamp)" "$*" >"$HOLDER"
trap 'rm -f "$HOLDER" 2>/dev/null || true' EXIT
[ "$waited" -gt 5 ] && log "acquired lock after ${waited}s wait"

if [ "$mode" = "measure" ]; then
  # Short in-lock re-verify only: the box was quiet moments ago, before we took
  # the lock. This absorbs a blip (e.g. the previous holder's last process
  # exiting) without ever holding the box through a long, hopeless wait.
  if ! wait_for_quiescence "$QUIET_GRACE" 1; then
    reason="$(cat "$LOCK_DIR/.last_reason" 2>/dev/null || echo unknown)"
    log "REFUSING TO MEASURE: box went busy again within the ${QUIET_GRACE}s in-lock grace ($reason)."
    log "Releasing the lock immediately rather than holding it through a long wait."
    offenders >&2
    exit 75
  fi
fi

# READ THESE TWO CORRECTLY. `load_before` is the meaningful one: it is sampled
# after quiescence, so it describes the box the run STARTED on. `load_after` is
# NOT a contention signal — ny is a rayon-parallel solver and drives loadavg to
# 12-20 all by itself, so a run legitimately reports load_before=2.5 ->
# load_after=19.7 with nothing else on the machine. Judging a run contaminated
# because load_after is high would throw away clean measurements.
load_before="$(load_snapshot)"
log "START mode=$mode load_before='$load_before'${budget:+ budget=${budget}s}"
log "cmd: $*"

s0=$(date +%s)
"$@"
rc=$?
elapsed=$(( $(date +%s) - s0 ))
load_after="$(load_snapshot)"

log "END rc=$rc elapsed=${elapsed}s load_after='$load_after'"
# Emit a machine-readable provenance line the caller can grep into its results
# table. A scored run whose log lacks this line was not box-locked and should be
# treated as unscored — that is the whole point of printing it.
printf '[ny-box-provenance] mode=%s rc=%d elapsed=%d load_before=%s load_after=%s lock_wait=%d quiesced=%s\n' \
  "$mode" "$rc" "$elapsed" "$(echo "$load_before" | cut -d' ' -f1)" \
  "$(echo "$load_after" | cut -d' ' -f1)" "$waited" \
  "$([ "$mode" = measure ] && echo yes || echo n/a)" >&2
exit $rc
