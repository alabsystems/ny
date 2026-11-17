#!/usr/bin/env bash
# ny_box.sh — mutual exclusion + quiescence gate for the measurement box.
#
# Builds and measurements share one advisory lock. Contention changes how much
# work a wall-limited solver can finish, so measurements also require quiescence.
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
# released when the last process holding an inherited lock descriptor closes it.
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
# After checking for live builders and ny processes, Linux measures busy cores
# over a /proc/stat window, counting iowait and steal as busy. Other hosts fall
# back to the one-minute load average; NY_BOX_QUIET_PROBE=loadavg selects it too.
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

# Match process names, including pinned ny copies. `-x ny` misses those copies;
# `-f` can match this script's own arguments and refuse its measurement.
ny_procs() { pgrep '^ny' 2>/dev/null | wc -l; }

# Check Trust drivers explicitly; stock compiler names remain for other
# toolchains. Report each matching PID and command to identify contention.
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

# Check build-tool credentials before waiting for the lock: a bad env token
# can prevent fetching the pinned AY dependency. Solver-only commands skip the
# probe, as does the macOS re-exec that already holds the lock.
if [ "${NY_BOX_FCNTL_LOCKED:-0}" != 1 ] && [ "${NY_SKIP_CREDENTIAL_CHECK:-0}" != 1 ]; then
  needs_credential=0
  for word in "$@"; do
    # Strip directories so full-path build-tool invocations also match.
    case "${word##*/}" in
      cargo|cargo.exe|targo|targo.exe) needs_credential=1; break ;;
      # Also recognize build-tool commands inside a shell-script argument.
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
      # Missing advisory helpers warn; gate.sh enforces tracked-file integrity.
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
