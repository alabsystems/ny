#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
#
# M0 runner TEMPLATE (ENGINE_PROGRAM section 2 M0 items 2-5, section 1.7 "Banking and moat protocol").
#
# Runs every row of an m0-spec/v1 spec -- MOAT rows (scripts/m0/moat_rows.py, both years), NEGCTL rows
# (negative_controls.py), KNIFE rows (knife_edge.py), AB rows (ab_arms.py), or a merge of them -- as
# `ny vnncomp v1 ... --proof-profile competition|proof` under the competition's own OS backstop
# (`timeout budget+10`, as vnncomp_scripts/run_instance.sh), one box acquisition per run through
# `scripts/ny_box.sh measure`, stdout+stderr captured PER RUN, and every run judged in python from its log AND
# its results file (scripts/m0/m0_judge.py, pinned by scripts/m0/m0_selftest.py):
#
#   * a stated verdict (log or results file) contradicting field truth, a banked sat<->unsat flip, sat and unsat
#     in one run, a tier or sat-precedence alarm, a sat<->unsat difference between any two runs of an instance
#     (any tag, VOID or not), or a Proof verdict that Competition does not reproduce in >= 2 of 3 reps
#     (section 1.7 mode identity) = SOUNDNESS EMERGENCY: summary, `M0MOAT_DONE rc=3 WORST=3 ...`, exit 3; a
#     relaunch into an M0_OUT whose record holds an emergency refuses (rc 3) before measuring anything;
#   * a run that cannot prove what it ran (wrong instance or preset, no `[proof-profile] mode=<p> source=cli`,
#     tier line absent, attacks not confirmed off in the flight record, no `Result:` line) = VOID, never a pass.
#
# Modes (the spec is the second argument in every mode):
#   --check   SPEC   step 0 only, to stdout: tools, pin, markers, zero NY_*, cgroup, spec, per-row resolver.
#                    No box, no measurement. Run it before --launch.
#   --launch  SPEC   step 0, then snapshot this script + m0lib.py + m0_judge.py + the spec into
#                    $M0_OUT/harness_<stamp>/ (bash reads a running script by offset: the snapshot is what
#                    runs, so this file may be edited afterwards) and start it under `systemd-run --user`
#                    (its own cgroup: terminal-scope teardown cannot kill it), with PATH/HOME/RUSTUP_HOME/
#                    CARGO_HOME, WorkingDirectory=$NY and BOTH box waits (NY_BOX_WAIT_LOCK and
#                    NY_BOX_WAIT_QUIET = 21600 s). Verifies MainPID, the unit cgroup and the environ.
#   --run     SPEC   the job itself (what the unit executes); log: $M0_OUT/m0moat.log (append).
#   --summary SPEC   re-judge nothing; print the summary of $M0_OUT/judged.tsv.
#
# Environment (required for --check/--launch/--run):
#   M0_PIN=<home>/ny-pins/ny_<...>   the gated release pin (built in an isolated worktree)
#   M0_PIN_SHA16=<16 hex>                  sha256 prefix recorded by the gate's PINNED line
#   M0_PIN_COMMIT=<40 hex>                 the commit that --build-id must name
#   M0_OUT=<home>/ny/scratchpad/...  output dir (created; per-run raw/ res/ and judged.tsv live here)
# Optional: M0_FEATURES (default mip,cuda), M0_LABEL (unit name), M0_MEMORY_MAX (>= 96G only: below it the
#   CROWN dense clamp moves and verdicts change -- CLAUDE.md), M0_NY (default <home>/ny),
#   M0_STOP_FILE (a path: when it exists before a run, the runner stops with `M0MOAT_DONE rc=3 ... reason=stop-file`
#   -- a caller running several specs at once points it at its own EMERGENCY record, so an emergency in one
#   stops the others before their next run).
#   M0_OUT_ROOT (default $M0_NY/scratchpad): the root M0_OUT must live under -- a gitignored or out-of-tree
#   directory, never a tracked path. The seal pipeline (scripts/seal/) points it at its own output root.
#
# Exit / DONE line: `M0MOAT_DONE rc=<r> WORST=<r> ...` propagates the REAL rc: 3 emergency, 5 VOID or not-run
# runs, 4 a banked verdict lost, 0 clean; 9x = a failed precondition (nothing measured).
#
# Portability: this file must run under bash 3.2 (macOS /bin/bash), as ny's other shell tools do, because
# m0_selftest.py drives it with whatever `bash` the host has (`--bash /bin/bash` selects the stock one, and the
# m0_ test in ny-test-utils' python_tooling_black_box runs exactly that; source_policy's m0_ tests scan this file
# lexically, which is what catches a bash-4 construct on Linux). No bash-4-only construct: no `mapfile`/
# `readarray`, no associative array (bash 3.2 rejects `declare -A` and the assignment then runs on an INDEXED
# array whose string subscripts evaluate arithmetically -- docs/DEFECT_REGISTER_2026-07-31.md D1
# `#bash32-declare-A`), no `${v,,}`/`${v^^}`, no negative subscripts, no `&>>`/`|&`, and every array that can
# be empty is expanded as `${a[@]+"${a[@]}"}` (bash 3.2 reports an empty array as unbound under `set -u`).
# Timestamps come from `date +FORMAT` fields both GNU and BSD date accept (`date -Is` is GNU-only), reshaped
# to what `date -Is` printed (`2026-10-04T12:00:00-07:00`), so the log format did not change on Linux: call
# now_iso, never `date -I...` (source_policy refuses that spelling here too).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NY=${M0_NY:-<home>/ny}
MODE=${1:-}
SPEC=${2:-}
OUT=${M0_OUT:-}
FEATURES=${M0_FEATURES:-mip,cuda}
JUDGE="$HERE/m0_judge.py"
MEM_MIN=103079215104   # 96 GiB
VERDICT_ENV_RE='^(NY_|NYCERT_|AY_MILP_|RAYON_|PROBE_|GPU_AVAILABLE=|CUDA_VISIBLE_DEVICES=)'

now_iso() { # GNU `date -Is` output, built from fields BSD date also has: %z is +hhmm, -Is prints +hh:mm
  local t head
  t=$(date +%Y-%m-%dT%H:%M:%S%z); head=${t%??}
  echo "$head:${t#$head}"
}
die() { echo "M0MOAT_DONE rc=$1 WORST=$1 reason=$2 $(now_iso)"; exit "$1"; }

need_env() { # [summary]: only the spec and M0_OUT are needed to re-read a finished run
  [ -n "$SPEC" ] && [ -f "$SPEC" ] || die 2 "usage: $0 --check|--launch|--run|--summary <spec.tsv>"
  SPEC=$(readlink -f "$SPEC")
  JUDGED=${OUT:+$OUT/judged.tsv}
  if [ "${1:-}" = summary ]; then [ -n "$OUT" ] || die 2 "M0_OUT unset"; return 0; fi
  [ -n "${M0_PIN:-}" ] || die 2 "M0_PIN unset"
  [[ "${M0_PIN_SHA16:-}" =~ ^[0-9a-f]{16}$ ]] || die 2 "M0_PIN_SHA16 must be 16 hex"
  [[ "${M0_PIN_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 2 "M0_PIN_COMMIT must be 40 hex"
  [ -n "$OUT" ] || die 2 "M0_OUT unset"
  OUT_ROOT=${M0_OUT_ROOT:-$NY/scratchpad}
  case "$OUT_ROOT" in /*) ;; *) die 2 "M0_OUT_ROOT must be absolute: $OUT_ROOT";; esac
  case "$OUT" in "${OUT_ROOT%/}"/*) ;; *) die 2 "M0_OUT must live under ${OUT_ROOT%/}/ (gitignored or out of tree)";; esac
  BIN=$OUT/probe_m0   # a name no one's `pgrep '^ny'` quiescence gate matches
  JUDGED=$OUT/judged.tsv
}

resolve_pin() {
  [ -f "$M0_PIN" ] && [ -x "$M0_PIN" ] || die 91 "pin-missing:$M0_PIN"
  PIN_SHA=$(sha256sum "$M0_PIN" | cut -d' ' -f1)
  [ "${PIN_SHA:0:16}" = "$M0_PIN_SHA16" ] || die 91 "pin-sha:${PIN_SHA:0:16}!=$M0_PIN_SHA16"
  mkdir -p "$OUT" || die 98 "mkdir:$OUT"
  if [ ! -x "$BIN" ] || [ "$(sha256sum "$BIN" | cut -d' ' -f1)" != "$PIN_SHA" ]; then
    cp "$M0_PIN" "$BIN" || die 91 "copy"
  fi
  [ "$(sha256sum "$BIN" | cut -d' ' -f1)" = "$PIN_SHA" ] || die 91 "copy-sha"
  printf '%s  %s\n' "$PIN_SHA" "$M0_PIN" > "$BIN.sha256"
  local bid
  bid=$("$BIN" --build-id 2>/dev/null </dev/null)
  echo "  PIN $M0_PIN -> $BIN sha256=$PIN_SHA"
  echo "  BUILD-ID $bid"
  case "$bid" in *"commit=$M0_PIN_COMMIT "*) ;; *) die 91 "build-id-commit";; esac
  case "$bid" in *"features=$FEATURES "*"dirty=false"*"exe=$PIN_SHA"*) ;; *) die 91 "build-id(features/dirty/exe)";; esac
}

mem_check() { # ny's governor reads the TIGHTEST memory.max on its cgroup path, ancestors included, and RLIMIT_AS
  # (crown_memory.rs process_memory_headroom): a cap anywhere below 96 GiB moves the CROWN dense clamp (CLAUDE.md)
  local cg mm p lv os
  os=$(uname -s)
  if [ "$os" != Linux ]; then
    # Off Linux the host has no cgroup hierarchy to read, and the dense budget's headroom term is None there
    # (crown_memory.rs observe_memory_terms; CLAUDE.md: "Off Linux the headroom is None"), so no cgroup cap can
    # move the dense clamp and the walk does not apply. RLIMIT_AS does apply: since c95ed6abc ny's strict
    # envelope (crown_memory.rs process_memory_envelope, #darwin-memory-envelope) reads it on macOS too, and
    # ny-cli's prepared-root route and vnncomp driver, the input-split batcher and the margin-row root box
    # consume that envelope. A finite limit therefore changes what ny does on this host as well, and it is
    # refused below as on Linux.
    echo "  cgroup: none on $os (no cgroup hierarchy to walk; RLIMIT_AS is checked on every host)"
    lv=$(ulimit -v)
    [ "$lv" = unlimited ] || die 86 "rlimit-as-set:$lv(KiB; refused on every host)"
    return 0
  fi
  cg=$(cut -d: -f3 /proc/self/cgroup 2>/dev/null | head -1)
  mm=$(cat "/sys/fs/cgroup$cg/memory.max" 2>/dev/null || echo '?')
  echo "  cgroup: $(head -1 /proc/self/cgroup 2>/dev/null) memory.max=$mm"
  case "$mm" in
    max) ;;
    ''|*[!0-9]*) die 86 "cgroup-memory.max-unreadable:$mm" ;;
    *) [ "$mm" -ge "$MEM_MIN" ] || die 86 "cgroup-memory-cap-below-96G:$mm" ;;
  esac
  p=$(dirname "$cg")
  while [ -n "$p" ] && [ "$p" != / ] && [ "$p" != . ]; do
    lv=$(cat "/sys/fs/cgroup$p/memory.max" 2>/dev/null || echo max)   # absent: no memory controller, no cap
    case "$lv" in
      max) ;;
      *[!0-9]*) die 86 "cgroup-ancestor-memory.max-unreadable:$p:$lv" ;;
      *) [ "$lv" -ge "$MEM_MIN" ] || die 86 "cgroup-ancestor-memory-cap-below-96G:$p:$lv" ;;
    esac
    p=$(dirname "$p")
  done
  lv=$(ulimit -v)
  [ "$lv" = unlimited ] || die 86 "rlimit-as-set:$lv(KiB; ny's governor reads it)"
}

plan_check() { # tag cat onnx vnnlib budget cfg preset -> 0 ok, 1 mismatch, 2 unverifiable (plan could not load)
  local T=$1 C=$2 O=$3 V=$4 B=$5 CFG=$6 PRE=$7 out p v b
  out=$(nice -n 19 "$BIN" vnncomp plan "$C" "$O" "$V" "$B" --configs-dir "$CFG" 2>&1 </dev/null)
  p=$(printf '%s\n' "$out" | awk '/^preset:/ && !f {print $2; f=1}')
  v=$(printf '%s\n' "$out" | awk '/^vnnlib:/ && !f {print $2; f=1}')
  b=$(printf '%s\n' "$out" | awk '/scored_budget_secs/ && !f {print $3; f=1}')
  if [ -z "$p$v$b" ] && [ "$(grep -c '^Error' <<<"$out")" -ge 1 ]; then
    echo "  PLAN-UNVERIFIABLE $T: $(grep -m1 '^Error' <<<"$out" | cut -c1-160) (the run's own Loading preset: line decides)"
    return 2
  fi
  if { [ "$PRE" = "-" ] || [ "$p" = "$CFG/$PRE" ]; } && [ "$v" = "$V" ] && [ "$b" = "$B" ]; then return 0; fi
  echo "  PLAN-FAIL $T: preset want=$CFG/$PRE got=${p:-<none>} | vnnlib ok=$([ "$v" = "$V" ] && echo y || echo n) | budget want=$B got=${b:-<none>}"
  return 1
}

step0() {
  local t mk c n=0 ms="" pc=0 pu=0 key
  for t in sha256sum strings python3 awk grep nice head cut readlink timeout; do
    command -v "$t" >/dev/null || die 127 "tool:$t"
  done
  python3 -c 'import yaml' 2>/dev/null || die 127 "python3-yaml"
  [ -x "$NY/scripts/ny_box.sh" ] || die 93 "ny_box.sh missing"
  [ -f "$JUDGE" ] && [ -f "$HERE/m0lib.py" ] || die 93 "harness files missing under $HERE"
  resolve_pin
  strings -a "$BIN" > "$OUT/pin_strings.txt" || die 92 "strings"
  while IFS= read -r mk; do
    [ -n "$mk" ] || continue
    c=$(grep -cF -- "$mk" "$OUT/pin_strings.txt"); c=${c:-0}
    [ "$c" -ge 1 ] || die 92 "marker-absent-from-pin:$mk"
    ms="$ms $c"; n=$((n+1))
  done < <(python3 "$JUDGE" markers --spec "$SPEC" </dev/null)
  [ $n -ge 5 ] || die 92 "marker-list-empty(the positive control failed)"
  echo "  markers ok ($n fragments, counts:$ms)"
  cd "$NY" || die 93 "cd $NY"
  # zero NY_* (run_instance.sh exports none), and none of the non-NY_ variables ny reads that change what it
  # does: GPU_AVAILABLE overrides the adapter probe, RAYON_* the margin-row thread default, AY_MILP_*/NYCERT_*
  # solver knobs, CUDA_VISIBLE_DEVICES the device set. The box's own NY_BOX_* knobs are stripped per run below.
  if env | grep -E "$VERDICT_ENV_RE" | grep -vE '^NY_BOX_'; then
    die 95 "verdict-env: the verifier must run with zero NY_* and none of GPU_AVAILABLE/RAYON_*/AY_MILP_*/NYCERT_*/CUDA_VISIBLE_DEVICES/PROBE_*"
  fi
  mem_check
  python3 "$JUDGE" validate --spec "$SPEC" </dev/null || die 97 "spec"
  if [ -f "$JUDGED" ]; then # a relaunch (resume) must never keep measuring past a recorded emergency
    python3 "$JUDGE" prior --spec "$SPEC" --judged "$JUDGED" </dev/null; local pr=$?
    [ $pr -eq 0 ] || { [ $pr -eq 3 ] && die 3 "prior-emergency(read $JUDGED; resolve it before any relaunch)"; die 96 "prior-check rc=$pr"; }
  fi
  # one newline-delimited list of the (category, configs dir, preset) keys already resolver-checked: bash 3.2
  # has no associative array (see the header), and a key holds `|` and `/`, which an indexed subscript would
  # evaluate as arithmetic
  local nl=$'\n' done_plan=""
  for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r T C O V B CFG PRE RC RP <<<"$line"
    # the resolver's answer depends on (category, configs dir) only: one representative row per pair keeps
    # step 0 (which runs OUTSIDE the box lock) from loading hundreds of models; every run's own
    # `Loading preset:` and instance lines are still checked by the judge
    key="$C|$CFG|$PRE"
    case "$nl$done_plan" in *"$nl$key$nl"*) continue ;; esac
    done_plan="$done_plan$key$nl"
    plan_check "$T" "$C" "$O" "$V" "$B" "$CFG" "$PRE"
    case $? in 0) pc=$((pc+1));; 2) pu=$((pu+1));; *) die 94 "plan:$T";; esac
  done
  local runs budget
  runs=$(awk -F'\t' '{n+=$8+$9} END{print n+0}' <(printf '%s\n' "${ROWS[@]}"))
  budget=$(awk -F'\t' '{s+=($8+$9)*$5} END{printf "%.1f", s/3600}' <(printf '%s\n' "${ROWS[@]}"))
  echo "  plans ok ($pc resolver checks over (category, configs dir), $pu unverifiable by plan); ${#ROWS[@]} rows, $runs runs, <= ${budget} h of budget"
}

load_rows() { # one array element per spec row; a read loop, not bash 4's mapfile (see the header)
  local row
  ROWS=()
  while IFS= read -r row || [ -n "$row" ]; do ROWS+=("$row"); done < <(python3 "$JUDGE" rows --spec "$SPEC" </dev/null)
  [ "${#ROWS[@]}" -ge 1 ] || die 97 "spec has no rows"
}

summary() { python3 "$JUDGE" summary --spec "$SPEC" --judged "$JUDGED" </dev/null; }

launch() {
  echo "=== M0MOAT launch $(now_iso)"
  step0
  local stamp snap unit pid cg envf v
  stamp=$(date +%Y%m%d-%H%M%S)
  snap=$OUT/harness_$stamp
  [ -e "$snap" ] && die 98 "snapshot-exists:$snap"
  mkdir -p "$snap" && cp "$HERE/m0lib.py" "$HERE/m0_judge.py" "$HERE/moat_both_profiles.sh" "$snap/" \
    && cp "$SPEC" "$snap/spec.tsv" || die 98 "snapshot"
  unit="m0moat-${M0_LABEL:-run}-$stamp"
  local -a props=(--property=WorkingDirectory="$NY")
  if [ -n "${M0_MEMORY_MAX:-}" ]; then props+=(--property=MemoryMax="$M0_MEMORY_MAX"); fi
  local -a envs=(--setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
                 --setenv=CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" --setenv=NY_BOX_WAIT_LOCK=21600
                 --setenv=NY_BOX_WAIT_QUIET=21600 --setenv=NY_BOX_QUIET_GRACE=300)
  for v in M0_PIN M0_PIN_SHA16 M0_PIN_COMMIT M0_OUT M0_FEATURES M0_NY M0_LABEL M0_STOP_FILE M0_OUT_ROOT; do
    if [ -n "${!v:-}" ]; then envs+=(--setenv="$v=${!v}"); fi
  done
  # no --collect: a collected unit vanishes with its status
  systemd-run --user --unit="$unit" "${props[@]}" "${envs[@]}" /bin/bash "$snap/moat_both_profiles.sh" --run "$snap/spec.tsv" \
    || die 99 "systemd-run"
  pid=0
  for _ in $(seq 1 40); do
    pid=$(systemctl --user show -p MainPID --value "$unit" 2>/dev/null); [ "${pid:-0}" != 0 ] && break
    sleep 0.25
  done
  if [ "${pid:-0}" = 0 ]; then
    echo "  unit $unit: $(systemctl --user show -p ActiveState -p Result -p ExecMainStatus "$unit" 2>/dev/null | tr '\n' ' ')"
    tail -5 "$OUT/m0moat.log" 2>/dev/null
    die 99 "unit-has-no-MainPID:$unit(exited already? read $OUT/m0moat.log)"
  fi
  cg=$(systemctl --user show -p ControlGroup --value "$unit" 2>/dev/null)
  case "$cg" in *"/app.slice/$unit.service") ;; *) die 99 "unit-cgroup:$cg";; esac
  envf=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null)
  grep -q '^NY_BOX_WAIT_LOCK=21600$' <<<"$envf" && grep -q '^NY_BOX_WAIT_QUIET=21600$' <<<"$envf" \
    || die 99 "unit-environ-lacks-box-waits(pid=$pid)"
  grep -q '^PATH=.*\.cargo/bin' <<<"$envf" || echo "  NOTE unit PATH has no .cargo/bin (the runner needs none; recorded)"
  echo "M0MOAT_LAUNCHED unit=$unit pid=$pid cgroup=$cg log=$OUT/m0moat.log snapshot=$snap"
}

run() { # stdout/stderr already appended to $OUT/m0moat.log by the --run branch
  echo "=== M0MOAT heartbeat start $(now_iso) pid=$$ unit=${INVOCATION_ID:-none} cwd=$(pwd) spec=$SPEC"
  export NY_BOX_WAIT_LOCK=${NY_BOX_WAIT_LOCK:-21600} NY_BOX_WAIT_QUIET=${NY_BOX_WAIT_QUIET:-21600} \
         NY_BOX_QUIET_GRACE=${NY_BOX_QUIET_GRACE:-300}
  step0
  ( while sleep 600; do echo "  HB $(now_iso) pid=$$ current=$(cat "$OUT/.current" 2>/dev/null || echo none)"; done ) &
  local hb=$! N=0 REFUSEDN=0 T C O V B CFG PRE RC RP P K RAW RES R JL JR k line v f aside n
  # every NY_BOX_* knob is the box's, never the verifier's: strip them all from the measured process (its flight
  # record snapshots every NY_* it saw, and the judge VOIDs a run whose snapshot is not exactly OMP_NUM_THREADS=1)
  local -a UNSET=()
  for v in $(compgen -e | grep -E '^NY_BOX_'); do UNSET+=(-u "$v"); done
  trap 'kill '"$hb"' 2>/dev/null' EXIT
  echo "  step0 ok; pin=${M0_PIN##*/} sha256=${PIN_SHA:0:16} NY_BOX_WAIT_LOCK=$NY_BOX_WAIT_LOCK NY_BOX_WAIT_QUIET=$NY_BOX_WAIT_QUIET"
  for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r T C O V B CFG PRE RC RP <<<"$line"
    for k in $(seq 1 $(( RC > RP ? RC : RP ))); do
      for P in competition proof; do
        if [ "$P" = competition ] && [ "$k" -gt "$RC" ]; then continue; fi
        if [ "$P" = proof ] && [ "$k" -gt "$RP" ]; then continue; fi
        RAW=$OUT/raw/$T.$P.r$k.raw; RES=$OUT/res/$T.$P.r$k.res
        if [ -f "$RAW.judged" ]; then continue; fi          # resume: this run was already judged
        if [ -n "${M0_STOP_FILE:-}" ] && [ -e "$M0_STOP_FILE" ]; then
          echo "  ###### STOP FILE $M0_STOP_FILE exists (an emergency recorded elsewhere) -- STOPPING before $T $P r$k ######"
          summary; echo "M0MOAT_DONE rc=3 WORST=3 STOPPED_AT=$T/$P/r$k reason=stop-file $(now_iso)"; exit 3
        fi
        [ "$(sha256sum "$BIN" | cut -d' ' -f1)" = "$PIN_SHA" ] || { summary; die 91 "pin-changed-before:$T"; }
        N=$((N+1)); echo "$T $P r$k $(now_iso)" > "$OUT/.current"
        # an interrupted earlier attempt of this run left files behind: a stale results file or flight record
        # would be judged as this run's (a flight check passing on the previous run's record). Move them aside
        # (never delete) so the judge reads only what THIS run wrote. The aside name is made unique by probing,
        # not by `date +%N` (BSD date has no %N on older hosts, and a repeated name would overwrite evidence).
        for f in "$RAW" "$RES" "$RES.flight.json"; do
          if [ -e "$f" ]; then
            aside=$f.stale.$(date +%s); n=0
            while [ -e "$aside.$n" ]; do n=$((n+1)); done
            mv -- "$f" "$aside.$n" || { summary; die 98 "stale-move:$f"; }
          fi
        done
        echo "  START $T $P r$k [$C] budget=${B}s backstop=$((B + 10))s $(now_iso)"
        R=75
        for attempt in 1 2 3; do
          # the competition's own OS wall-clock backstop (vnncomp_scripts/run_instance.sh: `timeout budget+10`):
          # an overrun is killed exactly where competition kills it, so its scored verdict is what the results
          # file holds at that instant (judged from --res; rc 124), and a hung run cannot hold the box forever
          "$NY/scripts/ny_box.sh" measure -- env ${UNSET[@]+"${UNSET[@]}"} \
            OMP_NUM_THREADS=1 timeout "$((B + 10))" "$BIN" vnncomp v1 "$C" "$O" "$V" "$RES" "$B" --configs-dir "$CFG" \
            --proof-profile "$P" >"$RAW" 2>&1 </dev/null; R=$?
          if [ $R -eq 75 ] || grep -q 'REFUSING TO MEASURE' "$RAW"; then
            echo "  REFUSED (box) $T $P r$k attempt $attempt rc=$R"; cp "$RAW" "$RAW.refused$attempt"; R=75; continue
          fi
          break
        done
        if [ $R -eq 75 ]; then REFUSEDN=$((REFUSEDN+1)); echo "  NOT-RUN $T $P r$k (box refused 3x; resume re-runs it)"; continue; fi
        JL=$(python3 "$JUDGE" judge --spec "$SPEC" --tag "$T" --profile "$P" --rep "$k" --raw "$RAW" \
               --flight "$RES.flight.json" --res "$RES" --box-rc "$R" --judged "$JUDGED" </dev/null); JR=$?
        [ $JR -eq 0 ] || [ $JR -eq 3 ] || [ $JR -eq 10 ] || { summary; die 96 "judge-error:$T rc=$JR"; }
        : > "$RAW.judged"
        echo "  RUN $JL | box_rc=$R"
        if [ $JR -eq 3 ]; then
          echo "  ###### $T $P r$k: SOUNDNESS EMERGENCY -- STOPPING ######"
          summary; echo "M0MOAT_DONE rc=3 WORST=3 STOPPED_AT=$T/$P/r$k reason=emergency $(now_iso)"; exit 3
        fi
        if ! python3 "$JUDGE" profiles --spec "$SPEC" --judged "$JUDGED" --tag "$T" </dev/null; then
          echo "  ###### $T: CROSS-RUN EMERGENCY (flip or profile split) -- STOPPING ######"
          summary; echo "M0MOAT_DONE rc=3 WORST=3 STOPPED_AT=$T reason=profile-emergency $(now_iso)"; exit 3
        fi
      done
    done
  done
  local S SR
  S=$(summary); SR=$?
  echo "$S"
  local last; last=$(grep -E '^M0_SUMMARY ' <<<"$S" | tail -1)
  echo "M0MOAT_DONE rc=$SR WORST=$SR ran=$N box_refused=$REFUSEDN ${last#M0_SUMMARY } pin=${M0_PIN##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60)) $(now_iso)"
  exit "$SR"
}

case "$MODE" in
  --check) need_env; load_rows; echo "=== M0MOAT step-0 check only $(now_iso)"; step0; echo "=== M0MOAT step-0 PASSED"; exit 0 ;;
  --launch) need_env; load_rows; launch; exit 0 ;;
  --run) need_env; mkdir -p "$OUT/raw" "$OUT/res" || die 98 "mkdir:$OUT"
         exec >>"$OUT/m0moat.log" 2>&1; load_rows; run ;;
  --summary) need_env summary; summary; exit $? ;;
  *) echo "usage: $0 --check|--launch|--run|--summary <spec.tsv>   (see the header)"; exit 2 ;;
esac
