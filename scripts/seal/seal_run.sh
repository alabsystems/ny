#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
#
# The SEAL runner (scripts/seal/README.md). Orchestration only: every decision is made in scripts/seal/seal.py +
# seallib.py (pinned by seal_selftest.py), and every measured run goes through the m0 runner
# (scripts/m0/moat_both_profiles.sh: one `scripts/ny_box.sh measure` hold per run, the competition backstop
# `timeout budget+10`, stdout+stderr kept PER RUN, the m0 judge, EMERGENCY stop, resume, M0_STOP_FILE).
#
#   --check    --stage S [--shard k/n]   step 0 only (fail closed, no box, no measurement)
#   --snapshot --stage S [--shard k/n]   step 0 + harness snapshot + print the exact systemd-run line (no launch)
#   --launch   --stage S [--shard k/n]   step 0 + snapshot + `systemd-run --user` (own cgroup, MemoryMax=96G,
#                                        WorkingDirectory=$SEAL_NY, PATH/HOME/RUSTUP_HOME/CARGO_HOME, both box waits
#                                        21600 s) + verified MainPID / cgroup / MemoryMax / environ
#   --run      --stage S [--shard k/n]   the job itself (what the unit executes); log $SEAL_OUT/seal_<S>[_kofn].log
#   --summary  --stage S                 re-derive the stage verdict from its records (writes nothing)
#
# Stages (plan order S5 -> S7 -> S6/S8; S6 and S8 refuse to start until S5 PASSED and S7 PASSED/PARTIAL on the same
# pin -- the plan carries one pin, PIN_LOCK binds every stage of one SEAL_OUT to it):
#   S5 arrival (overlay-only arming, zero NY_*, the observable on >= N rows else VOID)
#   S7 negative controls + knife-edge (attacks off, truly-SAT rows: ANY unsat = EMERGENCY, stop every shard)
#   S6 stratified A/B (armed vs capability-absent control, official budgets, >= 3 reps on marginal rows)
#   S8 both-profile both-year moat (0 flips; Proof verdicts reproduced in >= 2/3 Competition reps); in a CERTIFIED plan
#      (plan --s8-mode certified) S8 is phase A: every row ONCE in Competition, armed
#   S8C the certified follow-ups (scripts/seal/certlib.py), after every S8 shard's phase A is clean: (1) the x86 replay of
#      every armed-sat witness (one `ny_box.sh build` hold; a WRONG SAT is an EMERGENCY and stops before any re-proof),
#      (2) one Proof-profile re-run of every armed-unsat row the ledger does not cover (m0 runner, S8R_ tags), (3)
#      `ny proof-check` on each published bundle (one `ny_box.sh build` hold per check; `refuted` = EMERGENCY), (4) the
#      classic 3c+1p sub-moat of every re-proof the checker did not ACCEPT (m0 runner, S8F_ tags), (5) the S8 judge.
#      Every phase resumes: a written follow-up spec must re-derive, a judged run or a ran check is kept.
#
# Environment: SEAL_OUT (required: the plan directory written by `seal.py plan`). Optional: SEAL_NY (<home>/ny:
#   the main tree with scripts/ny_box.sh and benchmarks/), SEAL_LABEL (unit name), SEAL_MEMORY_MAX (96G; below it the
#   CROWN dense clamp moves and verdicts change -- CLAUDE.md -- so a smaller value is refused).
#
# DONE line (anchor a chain on `^SEAL_DONE rc=0 `, and count markers first: the log is appended):
#   SEAL_DONE rc=<stage rc> WORST=<max(stage rc, m0 rc)> stage=<S> shard=<k/n> status=<..> ... m0exit=<r> ...
#   rc 0 PASS; 4 PARTIAL; 5 VOID / incomplete / not-run; 3 EMERGENCY (EMERGENCY file written: every shard stops);
#   9x a failed precondition (nothing measured); 10 an empty shard (nothing to do).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
M0=$(cd "$HERE/../m0" && pwd)
PY="$HERE/seal.py"
NY=${SEAL_NY:-<home>/ny}
OUT=${SEAL_OUT:-}
MEMORY_MAX=${SEAL_MEMORY_MAX:-96G}
MEM_MIN=103079215104   # 96 GiB
VERDICT_ENV_RE='^(NY_|NYCERT_|AY_MILP_|RAYON_|PROBE_|GPU_AVAILABLE=|CUDA_VISIBLE_DEVICES=)'
BOX_ENV_OK_RE='^NY_BOX_(WAIT_LOCK|WAIT_QUIET|QUIET_GRACE)='
MODE=""; STAGE=""; SHARD="1/1"
while [ $# -gt 0 ]; do
  case "$1" in
    --check|--snapshot|--launch|--run|--summary) MODE=$1 ;;
    --stage) STAGE=${2:-}; shift ;;
    --shard) SHARD=${2:-}; shift ;;
    *) echo "usage: $0 --check|--snapshot|--launch|--run|--summary --stage S5|S6|S7|S8|S8C [--shard k/n]"; exit 2 ;;
  esac
  shift
done
SK=${SHARD/\//of}
TAG="$STAGE"; [ "$SHARD" != 1/1 ] && TAG="${STAGE}_$SK"

done_line() { # rc worst extra...
  local rc=$1 worst=$2; shift 2
  echo "SEAL_DONE rc=$rc WORST=$worst stage=${STAGE:-none} shard=$SHARD $* $(date -Is)"
}
die() { # the reason may quote another tool's DONE line or exit status: neutralise its rc=/WORST= (ONE rc= per line)
  local why=${2//rc=/rc:}; why=${why//WORST=/WORST:}
  done_line "$1" "$1" "reason=$why"; exit "$1"; }
py() { python3 "$PY" "$@" </dev/null; }

need_common() {
  [ -n "$OUT" ] || die 2 "SEAL_OUT-unset"
  case "$OUT" in /*) ;; *) die 2 "SEAL_OUT-must-be-absolute:$OUT";; esac
  [ -f "$OUT/plan.json" ] || die 2 "no-plan:$OUT/plan.json(run seal.py plan first)"
  case "$STAGE" in S5|S6|S7|S8|S8C) ;; *) die 2 "--stage S5|S6|S7|S8|S8C required";; esac
  [[ "$SHARD" =~ ^[1-9][0-9]*/[1-9][0-9]*$ ]] || die 2 "bad-shard:$SHARD"
  [ "$STAGE" != S8C ] || [ "$SHARD" = 1/1 ] || die 2 "S8C is one job (no shards)"
  case "$MEMORY_MAX" in 96G|9[7-9]G|[1-9][0-9][0-9]G) ;; *) die 2 "SEAL_MEMORY_MAX=$MEMORY_MAX(below 96G the CROWN dense clamp moves)";; esac
}

emergency_gate() {
  if [ -e "$OUT/EMERGENCY" ]; then
    echo "  EMERGENCY record present: $OUT/EMERGENCY"; sed 's/^/    /' "$OUT/EMERGENCY" | head -20
    die 3 "prior-emergency(triage it; nothing runs while $OUT/EMERGENCY exists)"
  fi
}

record_emergency() { printf '%s stage=%s shard=%s %s\n' "$(date -Is)" "$STAGE" "$SHARD" "$*" >> "$OUT/EMERGENCY"; }

pin_info() {
  local line
  while IFS= read -r line; do
    case "$line" in PIN_*=*|CAPABILITY=*|ROUTER_SHADOW=*|CERT_*=*) printf -v "${line%%=*}" '%s' "${line#*=}" ;; esac
  done < <(py pin-info --out "$OUT")
  [ -n "${PIN_PATH:-}" ] && [ -n "${PIN_SHA:-}" ] && [ -n "${PIN_COMMIT:-}" ] || die 97 "plan-pin-unreadable"
}

resolve_pin() {
  [ -f "$PIN_PATH" ] || die 91 "pin-missing:$PIN_PATH"
  [ "$(sha256sum "$PIN_PATH" | cut -d' ' -f1)" = "$PIN_SHA" ] || die 91 "pin-sha-differs-from-the-plan:$PIN_PATH"
  PROBE=$OUT/probe_seal   # a byte-identical copy: another agent's `pgrep '^ny'` quiescence gate never sees it
  if [ ! -x "$PROBE" ] || [ "$(sha256sum "$PROBE" | cut -d' ' -f1)" != "$PIN_SHA" ]; then
    cp "$PIN_PATH" "$PROBE.tmp" && chmod 755 "$PROBE.tmp" && mv "$PROBE.tmp" "$PROBE" || die 91 "copy"
  fi
  [ "$(sha256sum "$PROBE" | cut -d' ' -f1)" = "$PIN_SHA" ] || die 91 "copy-sha"
  local bid; bid=$("$PROBE" --build-id 2>/dev/null </dev/null)
  echo "  PIN $PIN_PATH sha256=$PIN_SHA"
  echo "  BUILD-ID $bid"
  case "$bid" in *"commit=$PIN_COMMIT "*) ;; *) die 91 "build-id-commit";; esac
  case "$bid" in *"features=$PIN_FEATURES "*"dirty=false"*"exe=$PIN_SHA"*) ;; *) die 91 "build-id(features/dirty/exe)";; esac
  local lock=$OUT/PIN_LOCK have
  if [ -f "$lock" ]; then
    have=$(awk '{print $1; exit}' "$lock")
    [ "$have" = "$PIN_SHA" ] || die 91 "pin-lock:$OUT-was-measured-with-${have:0:16}(one-pin-per-SEAL_OUT)"
  fi
}

pin_lock_write() { [ -f "$OUT/PIN_LOCK" ] || printf '%s %s %s %s\n' "$PIN_SHA" "$PIN_COMMIT" "$PIN_PATH" "$(date -Is)" > "$OUT/PIN_LOCK" || die 98 "pin-lock-write"; }

mem_check() { # ny's governor reads the TIGHTEST memory.max on its cgroup path, ancestors included, and RLIMIT_AS
  local cg mm p lv
  cg=$(cut -d: -f3 /proc/self/cgroup 2>/dev/null | head -1)
  mm=$(cat "/sys/fs/cgroup$cg/memory.max" 2>/dev/null || echo max)
  case "$mm" in max) ;; *[!0-9]*) die 86 "cgroup-memory.max-unreadable:$mm" ;; *) [ "$mm" -ge "$MEM_MIN" ] || die 86 "cgroup-memory-cap-below-96G:$mm" ;; esac
  MEM_EFFECTIVE=$mm   # the tightest cap on the path ("max" when none): what the dense-budget clamp actually reads
  p=$(dirname "$cg")
  while [ -n "$p" ] && [ "$p" != / ] && [ "$p" != . ]; do
    lv=$(cat "/sys/fs/cgroup$p/memory.max" 2>/dev/null || echo max)
    case "$lv" in max) ;; *[!0-9]*) die 86 "cgroup-ancestor-memory.max-unreadable:$p" ;; *) [ "$lv" -ge "$MEM_MIN" ] || die 86 "cgroup-ancestor-memory-cap-below-96G:$p:$lv" ;; esac
    case "$lv" in max) ;; *) { [ "$MEM_EFFECTIVE" = max ] || [ "$lv" -lt "$MEM_EFFECTIVE" ]; } && MEM_EFFECTIVE=$lv ;; esac
    p=$(dirname "$p")
  done
  lv=$(ulimit -v); [ "$lv" = unlimited ] || die 86 "rlimit-as-set:$lv"
}

mem_lock_check() { # --run only: ONE effective memory cap per SEAL_OUT. The cap's VALUE is load-bearing (CLAUDE.md:
  # headroom/8 clamps the CROWN dense budget, Dense->Patches), so an A/B or moat whose runs straddle two caps (a
  # relaunch with another SEAL_MEMORY_MAX, a --run outside the unit) is confounded; the first --run binds it
  local lock=$OUT/MEM_LOCK have
  if [ -f "$lock" ]; then
    have=$(awk '{print $1; exit}' "$lock")
    [ "$have" = "$MEM_EFFECTIVE" ] || die 86 "mem-lock:$OUT-was-measured-under-memory.max=$have,-this-run-is-under-$MEM_EFFECTIVE(one-cap-per-SEAL_OUT)"
  else
    printf '%s %s %s\n' "$MEM_EFFECTIVE" "$MEMORY_MAX" "$(date -Is)" > "$lock" || die 98 "mem-lock-write"
  fi
  echo "  memory cap bound: memory.max=$MEM_EFFECTIVE ($lock)"
}

markers_check() {
  local mk c n=0
  strings -a "$PROBE" > "$OUT/pin_strings.txt" || die 92 "strings"
  while IFS= read -r mk; do
    [ -n "$mk" ] || continue
    c=$(grep -cF -- "$mk" "$OUT/pin_strings.txt"); c=${c:-0}
    [ "$c" -ge 1 ] || die 92 "marker-absent-from-pin:$mk"
    n=$((n+1))
  done < <(py markers --out "$OUT")
  [ $n -ge 1 ] || die 92 "marker-list-empty"
  echo "  markers ok ($n capability fragments in the pin's strings)"
}

loader_probe() { # every overlay yaml the stage loads, through the pin's own preset loader; plus a NEGATIVE control
  local y n=0 dir=$OUT/step0_$TAG f frag
  mkdir -p "$dir"
  while IFS= read -r y; do
    [ -n "$y" ] || continue
    ( cd "$dir" && nice -n 19 timeout 120 "$PROBE" beta-crown /nonexistent_seal.onnx --preset "$y" ) > "$dir/loader.txt" 2>&1
    [ "$(grep -c 'Model loading failed: File not found: /nonexistent_seal.onnx' "$dir/loader.txt")" -ge 1 ] \
      || { echo "  LOADER-FAIL $y: $(grep -m2 -E '^Error|unknown|Caused by' "$dir/loader.txt" | tr '\n' ' ' | cut -c1-240)"; die 94 "loader:$y"; }
    n=$((n+1))
  done < <(py loader-list --out "$OUT" --stage "$STAGE")
  [ $n -ge 1 ] || die 94 "loader-list-empty"
  IFS=$'\t' read -r f frag < <(py loader-bad --out "$OUT" --stage "$STAGE" --dest-dir "$dir")
  [ -n "${f:-}" ] && [ -f "$f" ] || die 94 "loader-negative-control-not-written"
  ( cd "$dir" && nice -n 19 timeout 120 "$PROBE" beta-crown /nonexistent_seal.onnx --preset "$f" ) > "$f.txt" 2>&1
  [ "$(grep -c 'Model loading failed: File not found' "$f.txt")" -eq 0 ] \
    || die 94 "loader-negative-control-PASSED(the loader accepted an unknown key: its acceptance of the overlay proves nothing)"
  [ "$(grep -cF -- "$frag" "$f.txt")" -ge 1 ] \
    || die 94 "loader-negative-control-refused-for-another-reason(want '$frag'; read $f.txt)"
  echo "  loader ok: $n overlay yaml(s) load through the pin's preset loader; the unknown-key control is refused by name"
}

resolver_control() { # the m0 step 0 checks the pin's own resolver on the overlay dir; this proves the check can FAIL
  local c o v b p e out
  IFS=$'\t' read -r c o v b p e < <(py resolver-control --out "$OUT" --stage "$STAGE" --dest-dir "$OUT/step0_$TAG")
  [ -n "${e:-}" ] || die 94 "resolver-control-row"
  out=$( (cd "$OUT/step0_$TAG" && nice -n 19 timeout 300 "$PROBE" vnncomp plan "$c" "$o" "$v" "$b" --configs-dir "$e") 2>&1 </dev/null)
  if [ "$p" != - ] && [ "$(grep -cF -- "$p" <<<"$out")" -ge 1 ]; then
    die 94 "resolver-negative-control-resolved-the-overlay-from-an-empty-dir(the resolver check discriminates nothing)"
  fi
  echo "  resolver negative control ok: an empty configs dir does not resolve $(basename "$p")"
}

m0_env() { echo "M0_NY=$NY M0_PIN=$PIN_PATH M0_PIN_SHA16=${PIN_SHA:0:16} M0_PIN_COMMIT=$PIN_COMMIT M0_FEATURES=$PIN_FEATURES M0_STOP_FILE=$OUT/EMERGENCY M0_OUT_ROOT=$OUT"; }

step0() {
  local t rc
  echo "=== SEAL step0 stage=$STAGE shard=$SHARD out=$OUT $(date -Is)"
  for t in python3 sha256sum strings awk grep timeout nice git cut sed; do command -v "$t" >/dev/null || die 127 "tool:$t"; done
  python3 -c 'import yaml' 2>/dev/null || die 127 "python3-yaml"
  [ -x "$NY/scripts/ny_box.sh" ] || die 93 "ny_box.sh-missing:$NY"
  [ -f "$M0/moat_both_profiles.sh" ] && [ -f "$M0/m0lib.py" ] || die 93 "m0-harness-missing:$M0"
  emergency_gate
  local s0=$OUT/step0_py_$TAG.txt
  py step0 --out "$OUT" --stage "$STAGE" --shard "$SHARD" > "$s0" 2>&1; rc=$?
  sed 's/^/  /' "$s0"
  SPEC=$(grep -m1 '^SPEC=' "$s0" | cut -d= -f2-)
  M0OUT=$(grep -m1 '^M0OUT=' "$s0" | cut -d= -f2-)
  case "$rc" in
    0) ;;
    10) done_line 10 10 "status=EMPTY-SHARD reason=no-rows-in-this-shard"; exit 10 ;;
    3) die 3 "prior-emergency" ;;
    *) die 97 "step0-py(rc:$rc)" ;;
  esac
  [ -n "$SPEC" ] && [ -f "$SPEC" ] && [ -n "$M0OUT" ] || die 97 "step0-py-gave-no-spec"
  if env | grep -E "$VERDICT_ENV_RE" | grep -vE '^NY_BOX_'; then
    die 95 "verdict-env(the verifier must run with zero NY_* and none of GPU_AVAILABLE/RAYON_*/AY_MILP_*/NYCERT_*/CUDA_VISIBLE_DEVICES/PROBE_*)"
  fi
  # the box's own knobs other than the two waits and the grace change WHAT a measurement means (a raised load cap
  # or a test /proc/stat measures on a hot box; another lock dir or an inherited fcntl lock measures outside the
  # shared lock), which confounds an A/B as surely as a verdict lever (CLAUDE.md: load-max confounds fixed-order arms)
  if env | grep -E '^NY_BOX_' | grep -vE "$BOX_ENV_OK_RE"; then
    die 95 "box-env(only NY_BOX_WAIT_LOCK/NY_BOX_WAIT_QUIET/NY_BOX_QUIET_GRACE may reach ny_box.sh; a relaxed quiescence or lock confounds every arm)"
  fi
  mem_check
  pin_info
  resolve_pin
  markers_check
  loader_probe
  resolver_control
  mkdir -p "$M0OUT"
  env $(m0_env) M0_OUT="$M0OUT" bash "$M0/moat_both_profiles.sh" --check "$SPEC" > "$OUT/step0_m0_$TAG.txt" 2>&1; rc=$?
  grep -E 'PLAN-|markers ok|plans ok|M0MOAT_DONE' "$OUT/step0_m0_$TAG.txt" | sed 's/^/  m0: /' | tail -6
  [ $rc -eq 0 ] || die "$rc" "m0-step0(read $OUT/step0_m0_$TAG.txt)"
  echo "=== SEAL step0 PASSED stage=$STAGE shard=$SHARD"
}

step0_cert() { # S8C: the certified follow-ups' step 0 (fail closed, no box)
  local t rc s0=$OUT/step0_py_$TAG.txt n
  echo "=== SEAL step0 stage=$STAGE out=$OUT $(date -Is)"
  for t in python3 sha256sum strings awk grep timeout nice git cut sed; do command -v "$t" >/dev/null || die 127 "tool:$t"; done
  python3 -c 'import yaml' 2>/dev/null || die 127 "python3-yaml"
  [ -x "$NY/scripts/ny_box.sh" ] || die 93 "ny_box.sh-missing:$NY"
  [ -f "$M0/moat_both_profiles.sh" ] && [ -f "$M0/m0lib.py" ] || die 93 "m0-harness-missing:$M0"
  [ -f "$HERE/x86_replay.py" ] && [ -f "$HERE/certlib.py" ] || die 93 "certified-harness-missing:$HERE"
  emergency_gate
  py cert-step0 --out "$OUT" > "$s0" 2>&1; rc=$?
  sed 's/^/  /' "$s0"
  [ $rc -eq 0 ] || die 97 "cert-step0(rc:$rc; read $s0)"
  REPROOF_N=$(grep -m1 '^REPROOF=' "$s0" | cut -d= -f2); REPLAY_N=$(grep -m1 '^REPLAY=' "$s0" | cut -d= -f2)
  [[ "${REPROOF_N:-}" =~ ^[0-9]+$ ]] && [[ "${REPLAY_N:-}" =~ ^[0-9]+$ ]] || die 97 "cert-step0-gave-no-counts"
  if env | grep -E "$VERDICT_ENV_RE" | grep -vE '^NY_BOX_'; then
    die 95 "verdict-env(the verifier must run with zero NY_* and none of GPU_AVAILABLE/RAYON_*/AY_MILP_*/NYCERT_*/CUDA_VISIBLE_DEVICES/PROBE_*)"
  fi
  if env | grep -E '^NY_BOX_' | grep -vE "$BOX_ENV_OK_RE"; then
    die 95 "box-env(only NY_BOX_WAIT_LOCK/NY_BOX_WAIT_QUIET/NY_BOX_QUIET_GRACE may reach ny_box.sh)"
  fi
  mem_check
  pin_info
  [[ "${CERT_BUDGET_S:-}" =~ ^[0-9]+$ ]] && [[ "${CERT_TIMEOUT_S:-}" =~ ^[0-9]+$ ]] || die 97 "plan-is-not-certified(no checker budget)"
  resolve_pin
  markers_check
  # the pin must carry the checker: its own usage line, from the pin copy (a pin without `proof-check` would make
  # every re-proof fall back to the classic sub-moat -- correct, but a silent 4x cost; refuse up front)
  n=$( (cd "$OUT" && timeout 60 "$PROBE" proof-check --help) 2>&1 </dev/null | grep -c 'proof-check \[OPTIONS\] <BUNDLE>')
  [ "${n:-0}" -ge 1 ] || die 92 "the pin has no proof-check subcommand"
  echo "  checker ok: the pin's proof-check (budget ${CERT_BUDGET_S}s, timeout ${CERT_TIMEOUT_S}s); re-proofs=$REPROOF_N replays=$REPLAY_N"
  echo "=== SEAL step0 PASSED stage=$STAGE"
}

m0_phase() { # spec m0out label -> runs it through the m0 runner (step 0 first); exits on an emergency
  local spec=$1 mout=$2 label=$3 base new last R
  mkdir -p "$mout"
  env $(m0_env) M0_OUT="$mout" bash "$M0/moat_both_profiles.sh" --check "$spec" > "$OUT/step0_m0_${TAG}_$label.txt" 2>&1; R=$?
  [ $R -eq 0 ] || die "$R" "m0-step0-$label(read $OUT/step0_m0_${TAG}_$label.txt)"
  base=$(grep -c '^M0MOAT_DONE ' "$mout/m0moat.log" 2>/dev/null); base=${base:-0}
  echo "  [$label] m0 runner: spec=$spec out=$mout (DONE markers before: $base) $(date -Is)"
  env $(m0_env) M0_OUT="$mout" bash "$M0/moat_both_profiles.sh" --run "$spec"; R=$?
  new=$(grep -c '^M0MOAT_DONE ' "$mout/m0moat.log" 2>/dev/null); new=${new:-0}
  [ "$new" -gt "$base" ] || die 96 "m0-runner-wrote-no-NEW-DONE-line($label m0exit:$R)"
  last=$(grep '^M0MOAT_DONE ' "$mout/m0moat.log" | tail -1)
  echo "  [$label] m0: $last"
  case "$last" in "M0MOAT_DONE rc=$R "*) ;; *) die 96 "m0-DONE-disagrees-with-its-exit($label $R)";; esac
  if [ "$R" -eq 3 ]; then
    [[ "$last" == *" reason=stop-file "* ]] || record_emergency "m0-runner-stopped($label): ${last#M0MOAT_DONE } (read $mout/m0moat.log)"
    echo "  ###### SOUNDNESS EMERGENCY in the certified $label phase: $OUT/EMERGENCY written; nothing more runs ######"
    done_line 3 3 "status=EMERGENCY phase=$label m0exit=$R pin=${PIN_PATH##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60))"
    exit 3
  fi
  case $R in 0|4|5) ;; *) die "$R" "m0-precondition-$label(read $mout/m0moat.log)";; esac
  M0_LAST_RC=$R
}

derive() { # phase -> SPEC_D ROWS_D M0OUT_D
  local f=$OUT/S8C/derive_$1.txt rc
  mkdir -p "$OUT/S8C"
  py cert-derive --out "$OUT" --phase "$1" > "$f" 2>&1; rc=$?
  sed 's/^/  /' "$f"
  [ $rc -eq 0 ] || die 97 "cert-derive-$1(rc:$rc; read $f)"
  SPEC_D=$(grep -m1 '^SPEC=' "$f" | cut -d= -f2-); ROWS_D=$(grep -m1 '^ROWS=' "$f" | cut -d= -f2)
  M0OUT_D=$(grep -m1 '^M0OUT=' "$f" | cut -d= -f2-)
  [ -n "$SPEC_D" ] && [[ "$ROWS_D" =~ ^[0-9]+$ ]] && [ -n "$M0OUT_D" ] || die 97 "cert-derive-$1-gave-no-spec"
}

run_certify() {
  local R last tag bnd onnx vnn rep logp n=0 J JR s f worst=0 cdir=$OUT/S8C
  mkdir -p "$cdir/check"
  # (1) every armed-sat witness on the organisers' checker under emulated x86: a WRONG SAT stops before any re-proof
  if [ "$REPLAY_N" -gt 0 ]; then
    echo "  [replay] $REPLAY_N armed-sat row(s): x86_replay.py batch under one ny_box.sh build hold $(date -Is)"
    "$NY/scripts/ny_box.sh" build -- nice -n 10 python3 "$HERE/x86_replay.py" batch --out "$OUT" </dev/null >> "$cdir/x86_batch.log" 2>&1; R=$?
    last=$(grep '^X86_REPLAY_DONE ' "$cdir/x86_batch.log" | tail -1)
    echo "  [replay] ${last:-<no X86_REPLAY_DONE line>} (box rc=$R; log $cdir/x86_batch.log)"
    case "$last" in
      "X86_REPLAY_DONE rc=3 "*)
        record_emergency "x86 replay WRONG SAT: ${last#X86_REPLAY_DONE } (read $cdir/x86_batch.log)"
        echo "  ###### WRONG SAT: the organisers' checker rejects an armed-sat witness on x86 -- EMERGENCY ######"
        done_line 3 3 "status=EMERGENCY phase=replay pin=${PIN_PATH##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60))"
        exit 3 ;;
      "X86_REPLAY_DONE rc=0 "*) ;;
      *) worst=5; echo "  [replay] NOT every replay passed: the S8 stays VOID until they do (re-run S8C)" ;;
    esac
  fi
  # (2) one Proof-profile re-run of every armed-unsat row the ledger does not cover
  derive reproof
  if [ "$ROWS_D" -gt 0 ]; then m0_phase "$SPEC_D" "$M0OUT_D" reproof; fi
  # (3) ny proof-check on every published bundle (resume: a check that ran is kept)
  while IFS=$'\t' read -r tag bnd onnx vnn rep logp; do
    [ -n "$tag" ] || continue
    n=$((n+1))
    echo "  [check] $tag: proof-check $(basename "$bnd") $(date -Is)"
    "$NY/scripts/ny_box.sh" build -- nice -n 10 timeout "$CERT_TIMEOUT_S" "$PROBE" proof-check "$bnd" --onnx "$onnx" \
      --vnnlib "$vnn" --report "$rep" --budget-s "$CERT_BUDGET_S" > "$logp" 2>&1 </dev/null; R=$?
    if [ $R -eq 75 ] || [ "$(grep -c '^\[ny-box\] .*REFUS' "$logp")" -ge 1 ]; then
      # the box refused (lock wait expired): the check did NOT run -- it stays pending (a re-run of S8C runs it),
      # never a fallback that would spend a classic sub-moat on a row the checker never saw
      mv -- "$logp" "$logp.refused.$(date +%s%N)"
      echo "  [check] $tag: the box refused (rc=$R): the check stays pending"
      worst=5
      continue
    fi
    echo "CERT_CHECK_RAN rc=$R" >> "$logp"
    echo "  [check] $tag: rc=$R $(grep -m1 '^PROOF_CHECK_DONE ' "$logp" | cut -c1-200)"
  done < <(py cert-check-list --out "$OUT")
  py cert-scan --out "$OUT" > "$cdir/scan.txt" 2>&1; R=$?
  sed 's/^/  [scan] /' "$cdir/scan.txt"
  if [ $R -eq 3 ]; then
    record_emergency "certified re-proof: $(grep -m3 '^  EMERGENCY ' "$cdir/scan.txt" | tr '\n' ' ')"
    echo "  ###### a re-proof is an EMERGENCY (refuted, or the Proof profile states sat) -- EMERGENCY FILE WRITTEN ######"
    done_line 3 3 "status=EMERGENCY phase=check pin=${PIN_PATH##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60))"
    exit 3
  fi
  [ $R -eq 0 ] || die 96 "cert-scan(rc:$R)"
  # (4) the classic sub-moat of every re-proof the checker did not ACCEPT
  if grep -q ' pending=0 ' "$cdir/scan.txt" || grep -q ' pending=0$' "$cdir/scan.txt"; then
    derive classic
    if [ "$ROWS_D" -gt 0 ]; then m0_phase "$SPEC_D" "$M0OUT_D" classic; fi
  else
    echo "  [classic] re-proofs still pending (a Proof run not OK, or a check not run): the classic set waits"
    worst=5
  fi
  # (5) the S8 verdict
  J=$(py judge-stage --out "$OUT" --stage S8 --harness "$(dirname "$HERE")"); JR=$?
  echo "$J"
  s=$(grep '^SEAL_STAGE ' <<<"$J" | tail -1)
  [ -n "$s" ] && [[ "$JR" =~ ^(0|3|4|5)$ ]] || die 96 "judge-stage(rc:$JR)"
  f=$(sed -E 's/^SEAL_STAGE //; s/(^| )stage=[^ ]+//; s/^ +//' <<<"$s")
  if [ $JR -eq 3 ]; then
    record_emergency "$(grep -m3 '^  EMERGENCY ' <<<"$J" | tr '\n' ' ')"
    done_line 3 3 "${f/status=[A-Z]*/status=EMERGENCY} pin=${PIN_PATH##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60))"
    exit 3
  fi
  [ $JR -gt $worst ] && worst=$JR
  done_line "$JR" "$worst" "$f pin=${PIN_PATH##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60))"
  exit "$JR"
}

snapshot_and_unit() { # -> SNAP, UNIT, CMD (array)
  local stamp; stamp=$(date +%Y%m%d-%H%M%S)
  SNAP=$OUT/harness_${TAG}_$stamp
  [ -e "$SNAP" ] && die 98 "snapshot-exists:$SNAP"
  mkdir -p "$SNAP/seal" "$SNAP/m0" && cp "$HERE"/*.py "$HERE"/*.sh "$SNAP/seal/" && cp "$M0"/*.py "$M0"/*.sh "$SNAP/m0/" \
    || die 98 "snapshot"
  UNIT="seal-${SEAL_LABEL:-$CAPABILITY}-${TAG//_/-}-$stamp"
  CMD=(systemd-run --user --unit="$UNIT" --property=WorkingDirectory="$NY" --property=MemoryMax="$MEMORY_MAX"
       --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
       --setenv=CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" --setenv=NY_BOX_WAIT_LOCK=21600 --setenv=NY_BOX_WAIT_QUIET=21600
       --setenv=NY_BOX_QUIET_GRACE=300 --setenv=SEAL_OUT="$OUT" --setenv=SEAL_NY="$NY" --setenv=SEAL_MEMORY_MAX="$MEMORY_MAX")
  [ -n "${SEAL_LABEL:-}" ] && CMD+=(--setenv=SEAL_LABEL="$SEAL_LABEL")
  CMD+=(/bin/bash "$SNAP/seal/seal_run.sh" --run --stage "$STAGE" --shard "$SHARD")
}

launch() {
  if [ "$STAGE" = S8C ]; then step0_cert; else step0; fi
  snapshot_and_unit
  echo "  SYSTEMD-RUN: $(printf '%q ' "${CMD[@]}")"
  # no --collect: a collected unit vanishes with its status
  "${CMD[@]}" || die 99 "systemd-run"
  local pid=0 cg envf mm
  for _ in $(seq 1 40); do pid=$(systemctl --user show -p MainPID --value "$UNIT" 2>/dev/null); [ "${pid:-0}" != 0 ] && break; sleep 0.25; done
  [ "${pid:-0}" != 0 ] || { tail -5 "$OUT/seal_$TAG.log" 2>/dev/null; die 99 "unit-has-no-MainPID:$UNIT(exited already? read $OUT/seal_$TAG.log)"; }
  cg=$(systemctl --user show -p ControlGroup --value "$UNIT" 2>/dev/null)
  case "$cg" in *"/app.slice/$UNIT.service") ;; *) die 99 "unit-cgroup:$cg";; esac
  mm=$(systemctl --user show -p MemoryMax --value "$UNIT" 2>/dev/null)
  [ "${mm:-0}" = "$(numfmt --from=iec "$MEMORY_MAX" 2>/dev/null || echo "$MEM_MIN")" ] || echo "  NOTE unit MemoryMax=$mm (wanted $MEMORY_MAX)"
  envf=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null)
  grep -q '^NY_BOX_WAIT_LOCK=21600$' <<<"$envf" && grep -q '^NY_BOX_WAIT_QUIET=21600$' <<<"$envf" || die 99 "unit-environ-lacks-box-waits(pid=$pid)"
  grep -q "^SEAL_OUT=$OUT\$" <<<"$envf" || die 99 "unit-environ-lacks-SEAL_OUT(pid=$pid)"
  echo "SEAL_LAUNCHED unit=$UNIT pid=$pid cgroup=$cg memory_max=$mm log=$OUT/seal_$TAG.log snapshot=$SNAP"
}

heartbeat() {
  ( while sleep 600; do echo "  HB $(date -Is) pid=$$ stage=$STAGE shard=$SHARD current=$(cat "${M0OUT:-$OUT/S8C}/.current" "$OUT/S8C/run_reproof/.current" "$OUT/S8C/run_classic/.current" 2>/dev/null | tail -1 || echo -)"; done ) &
  HB=$!
  trap 'kill $HB 2>/dev/null' EXIT
}

stage_field() { grep -oE " $1=[^ ]+" <<<"$2" | head -1 | cut -d= -f2; }

run_stage() {
  local base new last R J JR s worst f
  base=$(grep -c '^M0MOAT_DONE ' "$M0OUT/m0moat.log" 2>/dev/null); base=${base:-0}
  echo "  m0 runner: spec=$SPEC out=$M0OUT (DONE markers before: $base) $(date -Is)"
  env $(m0_env) M0_OUT="$M0OUT" bash "$M0/moat_both_profiles.sh" --run "$SPEC"; R=$?
  new=$(grep -c '^M0MOAT_DONE ' "$M0OUT/m0moat.log" 2>/dev/null); new=${new:-0}
  [ "$new" -gt "$base" ] || die 96 "m0-runner-wrote-no-NEW-DONE-line(m0exit:$R)"
  last=$(grep '^M0MOAT_DONE ' "$M0OUT/m0moat.log" | tail -1)
  echo "  m0: $last"
  case "$last" in "M0MOAT_DONE rc=$R "*) ;; *) die 96 "m0-DONE-disagrees-with-its-exit($R)";; esac
  if [ "$R" -eq 3 ] && [[ "$last" == *" reason=stop-file "* ]]; then
    echo "  ###### $OUT/EMERGENCY appeared during this stage: stopped before the next run ######"
    done_line 3 3 "status=STOPPED stopped_by=$OUT/EMERGENCY m0exit=$R elapsed_min=$((SECONDS/60))"; exit 3
  fi
  local emerg_recorded=0
  if [ "$R" -eq 3 ]; then
    # the m0 runner STOPPED on an emergency: record it BEFORE judging, so every other shard stops at its next run
    # even when the judge below fails (review: a judge that crashed after m0 exit 3 died 96 with no EMERGENCY file,
    # and every other shard kept measuring)
    record_emergency "m0-runner-stopped: ${last#M0MOAT_DONE } (read $M0OUT/m0moat.log)"
    emerg_recorded=1
    echo "  ###### m0 runner stopped on a SOUNDNESS EMERGENCY: $OUT/EMERGENCY written before judging ######"
  fi
  case $R in 0|3|4|5) ;; *) die "$R" "m0-precondition(read $M0OUT/m0moat.log)";; esac
  J=$(py judge-stage --out "$OUT" --stage "$STAGE" --harness "$(dirname "$HERE")"); JR=$?
  echo "$J"
  s=$(grep '^SEAL_STAGE ' <<<"$J" | tail -1)
  if [ -z "$s" ] || ! [[ "$JR" =~ ^(0|3|4|5)$ ]]; then
    if [ "$emerg_recorded" = 1 ]; then
      done_line 3 3 "status=EMERGENCY judge=failed(rc:$JR) m0exit=$R pin=${PIN_PATH##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60))"
      exit 3
    fi
    [ -n "$s" ] || die 96 "judge-stage-printed-no-SEAL_STAGE-line(rc:$JR)"
    die 96 "judge-stage-error(rc:$JR)"
  fi
  f=$(sed -E 's/^SEAL_STAGE //; s/(^| )stage=[^ ]+//; s/^ +//' <<<"$s")   # the DONE line names the stage once
  worst=$(( JR > R ? JR : R ))
  if [ $R -eq 3 ] || [ $JR -eq 3 ]; then
    [ "$emerg_recorded" = 1 ] || record_emergency "$(grep -m3 '^  EMERGENCY ' <<<"$J" | tr '\n' ' ') m0exit=$R"
    echo "  ###### SOUNDNESS EMERGENCY in $STAGE -- EMERGENCY FILE WRITTEN, EVERY SHARD STOPS ######"
    done_line 3 3 "${f/status=[A-Z]*/status=EMERGENCY} m0exit=$R pin=${PIN_PATH##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60))"
    exit 3
  fi
  done_line "$JR" "$worst" "$f m0exit=$R pin=${PIN_PATH##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60))"
  exit "$JR"
}

case "$MODE" in
  --check|--snapshot|--launch|--run|--summary) need_common ;;
  *) echo "usage: $0 --check|--snapshot|--launch|--run|--summary --stage S5|S6|S7|S8 [--shard k/n]"; exit 2 ;;
esac
S0=step0; [ "$STAGE" = S8C ] && S0=step0_cert
case "$MODE" in
  --check) $S0; exit 0 ;;
  --snapshot) $S0; snapshot_and_unit; echo "  SNAPSHOT $SNAP"; echo "SYSTEMD-RUN $(printf '%q ' "${CMD[@]}")"; exit 0 ;;
  --launch) launch; exit 0 ;;
  --summary) py judge-stage --out "$OUT" --stage "${STAGE/S8C/S8}" --no-write; exit $? ;;
  --run)
    exec >>"$OUT/seal_$TAG.log" 2>&1
    cd "$NY" || die 93 "cd:$NY"
    echo "=== SEAL heartbeat start $(date -Is) pid=$$ stage=$STAGE shard=$SHARD unit=${INVOCATION_ID:-none} cwd=$(pwd) out=$OUT"
    $S0
    mem_lock_check
    pin_lock_write
    export NY_BOX_WAIT_LOCK=${NY_BOX_WAIT_LOCK:-21600} NY_BOX_WAIT_QUIET=${NY_BOX_WAIT_QUIET:-21600} NY_BOX_QUIET_GRACE=${NY_BOX_QUIET_GRACE:-300}
    echo "  step0 ok; pin=${PIN_PATH##*/} sha256=${PIN_SHA:0:16} NY_BOX_WAIT_LOCK=$NY_BOX_WAIT_LOCK NY_BOX_WAIT_QUIET=$NY_BOX_WAIT_QUIET"
    heartbeat
    if [ "$STAGE" = S8C ]; then run_certify; else run_stage; fi ;;
esac
