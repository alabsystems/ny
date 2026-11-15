#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
#
# The runner of the KEYLESS CERTIFIED moat (scripts/seal/keyless_certified.py: every decision is made there; this
# file only orchestrates). Every measured run goes through the m0 runner (scripts/m0/moat_both_profiles.sh: one
# `scripts/ny_box.sh measure` hold per run, the competition backstop `timeout budget+10`, stdout+stderr kept PER
# RUN, the m0 judge, M0_STOP_FILE); every x86 replay and every proof-check runs under one `ny_box.sh build` hold.
#
#   --check   --shard k/n   step 0 only (fail closed, no box, no measurement)
#   --launch  --shard k/n   step 0 + harness snapshot + `systemd-run --user` (own cgroup, MemoryMax=$KC_MEMORY_MAX,
#                           WorkingDirectory=$KC_NY, PATH/HOME/RUSTUP_HOME/CARGO_HOME, both box waits 21600 s and the
#                           quiet grace 300 s) + verified MainPID / cgroup / MemoryMax / environ
#   --run     --shard k/n   the job itself; log $KC_OUT/kc_<k>of<n>.log (appended)
#   --summary               the cumulative judgement of every batch judged so far (writes nothing)
#   --sample <abs path>     (with any mode) AMENDMENT 2: run the STRATIFIED SAMPLE written by `keyless_certified.py
#                           sample` (<KC_OUT>/<name>.json, sha-locked, bound to plan.json's sha256) instead of the full
#                           plan: the batch list becomes the sample's carried (judged) batches + its new batches. Every
#                           judge, EMERGENCY, replay, re-proof, check, classic and global cross-tag rule is unchanged
#                           (the global check still reads EVERY plan row). --launch names the unit
#                           kc-<cap>-<sample name>-<k>of<n>-<stamp> and passes --sample to the unit.
#
# Per batch (the plan's order): A (phase A, m0) -> replay (x86, sats) -> reproof (m0, Proof) -> check (proof-check)
# -> scan (+ the route-B2 rule) -> classic (m0, 3c+1p of every non-ACCEPT) -> judge-batch (BATCH.json, and the
# global cross-tag INSTANCE-FLIP). A batch with a final BATCH.json is skipped on a relaunch (resume).
#
# Environment: KC_OUT (required: the plan directory written by `keyless_certified.py plan`). Optional: KC_NY
# (<home>/ny: ny_box.sh and benchmarks/), KC_MEMORY_MAX (110G; below 96G the CROWN dense clamp moves).
#
# EMERGENCY (truth contradiction, bank flip, alarm, a WRONG SAT on x86, refuted, a Proof sat, a route-B2 breach, a
# cross-tag instance flip): $KC_OUT/EMERGENCY is written and every shard stops before its next run.
# DONE line (anchor on `^KC_DONE rc=`, count markers first: the log is appended):
#   KC_DONE rc=<worst> WORST=<worst> shard=<k/n> status=<..> batches=<n> judged=<k> ...
#   rc 0 every batch PASS; 4 a lost banked verdict; 5 VOID / pending somewhere; 3 EMERGENCY; 9x a precondition.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
M0=$(cd "$HERE/../m0" && pwd)
PY="$HERE/keyless_certified.py"
NY=${KC_NY:-<home>/ny}
OUT=${KC_OUT:-}
MEMORY_MAX=${KC_MEMORY_MAX:-110G}
MEM_MIN=103079215104   # 96 GiB
VERDICT_ENV_RE='^(NY_|NYCERT_|AY_MILP_|RAYON_|PROBE_|GPU_AVAILABLE=|CUDA_VISIBLE_DEVICES=)'
BOX_ENV_OK_RE='^NY_BOX_(WAIT_LOCK|WAIT_QUIET|QUIET_GRACE)='
MODE=""; SHARD="1/1"; SAMPLE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check|--launch|--run|--summary) MODE=$1 ;;
    --shard) SHARD=${2:-}; shift ;;
    --sample) SAMPLE=${2:-}; shift ;;
    *) echo "usage: $0 --check|--launch|--run|--summary [--shard k/n] [--sample <abs sample.json>]"; exit 2 ;;
  esac
  shift
done
SARGS=()
[ -n "$SAMPLE" ] && SARGS=(--sample "$SAMPLE")
SK=${SHARD/\//of}
WORST=0

done_line() { local rc=$1 worst=$2; shift 2; echo "KC_DONE rc=$rc WORST=$worst shard=$SHARD $* $(date -Is)"; }
die() { local why=${2//rc=/rc:}; why=${why//WORST=/WORST:}; done_line "$1" "$1" "reason=$why"; exit "$1"; }
py() { python3 "$PY" "$@" "${SARGS[@]}" </dev/null; }   # every py subcommand the runner calls takes --sample
record_emergency() { printf '%s shard=%s %s\n' "$(date -Is)" "$SHARD" "$*" >> "$OUT/EMERGENCY"; }

need_common() {
  [ -n "$OUT" ] || die 2 "KC_OUT-unset"
  case "$OUT" in /*) ;; *) die 2 "KC_OUT-must-be-absolute:$OUT";; esac
  [ -f "$OUT/plan.json" ] || die 2 "no-plan:$OUT/plan.json"
  [[ "$SHARD" =~ ^[1-9][0-9]*/[1-9][0-9]*$ ]] || die 2 "bad-shard:$SHARD"
  case "$MEMORY_MAX" in 9[6-9]G|[1-9][0-9][0-9]G) ;; *) die 2 "KC_MEMORY_MAX=$MEMORY_MAX(below 96G the CROWN dense clamp moves)";; esac
  if [ -n "$SAMPLE" ]; then
    case "$SAMPLE" in /*) ;; *) die 2 "sample-must-be-absolute:$SAMPLE";; esac
    [ -f "$SAMPLE" ] && [ -f "$SAMPLE.sha256" ] || die 2 "no-sample:$SAMPLE(+.sha256)"
    [ "$(dirname "$SAMPLE")" = "$OUT" ] || die 2 "sample-not-in-KC_OUT:$SAMPLE"
  fi
}

emergency_gate() {
  if [ -e "$OUT/EMERGENCY" ]; then
    echo "  EMERGENCY record present: $OUT/EMERGENCY"; sed 's/^/    /' "$OUT/EMERGENCY" | head -20
    done_line 3 3 "status=STOPPED stopped_by=$OUT/EMERGENCY elapsed_min=$((SECONDS/60))"; exit 3
  fi
}

mem_check() { # ny's governor reads the TIGHTEST memory.max on its cgroup path, ancestors included, and RLIMIT_AS
  local cg mm p lv
  cg=$(cut -d: -f3 /proc/self/cgroup 2>/dev/null | head -1)
  mm=$(cat "/sys/fs/cgroup$cg/memory.max" 2>/dev/null || echo max)
  case "$mm" in max) ;; *[!0-9]*) die 86 "cgroup-memory.max-unreadable:$mm" ;; *) [ "$mm" -ge "$MEM_MIN" ] || die 86 "cgroup-memory-cap-below-96G:$mm" ;; esac
  MEM_EFFECTIVE=$mm
  p=$(dirname "$cg")
  while [ -n "$p" ] && [ "$p" != / ] && [ "$p" != . ]; do
    lv=$(cat "/sys/fs/cgroup$p/memory.max" 2>/dev/null || echo max)
    case "$lv" in max) ;; *[!0-9]*) die 86 "cgroup-ancestor-memory.max-unreadable:$p" ;; *) [ "$lv" -ge "$MEM_MIN" ] || die 86 "cgroup-ancestor-memory-cap-below-96G:$p:$lv" ;; esac
    case "$lv" in max) ;; *) { [ "$MEM_EFFECTIVE" = max ] || [ "$lv" -lt "$MEM_EFFECTIVE" ]; } && MEM_EFFECTIVE=$lv ;; esac
    p=$(dirname "$p")
  done
  lv=$(ulimit -v); [ "$lv" = unlimited ] || die 86 "rlimit-as-set:$lv"
}

mem_lock_check() { # ONE effective memory cap per KC_OUT (the cap's VALUE moves the CROWN dense clamp)
  local lock=$OUT/MEM_LOCK have
  if [ -f "$lock" ]; then
    have=$(awk '{print $1; exit}' "$lock")
    [ "$have" = "$MEM_EFFECTIVE" ] || die 86 "mem-lock:$OUT-was-measured-under-memory.max=$have,-this-run-is-under-$MEM_EFFECTIVE"
  else
    printf '%s %s %s\n' "$MEM_EFFECTIVE" "$MEMORY_MAX" "$(date -Is)" > "$lock" || die 98 "mem-lock-write"
  fi
  echo "  memory cap bound: memory.max=$MEM_EFFECTIVE ($lock)"
}

m0_env() { echo "M0_NY=$NY M0_PIN=$PIN M0_PIN_SHA16=${PIN_SHA:0:16} M0_PIN_COMMIT=$PIN_COMMIT M0_FEATURES=$PIN_FEATURES M0_STOP_FILE=$OUT/EMERGENCY M0_OUT_ROOT=$OUT"; }

step0() {
  local t rc s0=$OUT/step0_py_$SK.txt neg c o v b e p out n
  echo "=== KC step0 shard=$SHARD out=$OUT $(date -Is)"
  for t in python3 sha256sum strings awk grep timeout nice git cut sed ln systemctl; do command -v "$t" >/dev/null || die 127 "tool:$t"; done
  python3 -c 'import yaml' 2>/dev/null || die 127 "python3-yaml"
  [ -x "$NY/scripts/ny_box.sh" ] || die 93 "ny_box.sh-missing:$NY"
  [ -f "$M0/moat_both_profiles.sh" ] && [ -f "$M0/m0lib.py" ] && [ -f "$M0/m0_judge.py" ] || die 93 "m0-harness-missing:$M0"
  [ -f "$HERE/x86_replay.py" ] && [ -f "$HERE/certlib.py" ] && [ -f "$HERE/seallib.py" ] && [ -f "$HERE/keyless_moat_plan.py" ] || die 93 "seal-harness-missing:$HERE"
  [ -e "$OUT/EMERGENCY" ] && { echo "  EMERGENCY record present: $OUT/EMERGENCY"; die 3 "prior-emergency"; }
  py step0 --out "$OUT" --shard "$SK" > "$s0" 2>&1; rc=$?
  sed 's/^/  /' "$s0"
  [ $rc -eq 0 ] || die 97 "step0-py(rc:$rc; read $s0)"
  UNION=$(grep -m1 '^UNION=' "$s0" | cut -d= -f2-); NBATCH=$(grep -m1 '^BATCHES=' "$s0" | cut -d= -f2)
  PIN=$(grep -m1 '^PIN=' "$s0" | cut -d= -f2-); PIN_SHA=$(grep -m1 '^PIN_SHA=' "$s0" | cut -d= -f2)
  PIN_COMMIT=$(grep -m1 '^PIN_COMMIT=' "$s0" | cut -d= -f2); PIN_FEATURES=$(grep -m1 '^PIN_FEATURES=' "$s0" | cut -d= -f2)
  CERT_BUDGET_S=$(grep -m1 '^CERT_BUDGET_S=' "$s0" | cut -d= -f2); CERT_TIMEOUT_S=$(grep -m1 '^CERT_TIMEOUT_S=' "$s0" | cut -d= -f2)
  [ -n "$UNION" ] && [ -f "$UNION" ] && [[ "${NBATCH:-}" =~ ^[0-9]+$ ]] && [[ "${PIN_SHA:-}" =~ ^[0-9a-f]{64}$ ]] \
    && [[ "${PIN_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] && [[ "${CERT_BUDGET_S:-}" =~ ^[0-9]+$ ]] && [[ "${CERT_TIMEOUT_S:-}" =~ ^[0-9]+$ ]] \
    || die 97 "step0-py-gave-no-plan-facts"
  if env | grep -E "$VERDICT_ENV_RE" | grep -vE '^NY_BOX_'; then
    die 95 "verdict-env(the verifier must run with zero NY_* and none of GPU_AVAILABLE/RAYON_*/AY_MILP_*/NYCERT_*/CUDA_VISIBLE_DEVICES/PROBE_*)"
  fi
  if env | grep -E '^NY_BOX_' | grep -vE "$BOX_ENV_OK_RE"; then
    die 95 "box-env(only NY_BOX_WAIT_LOCK/NY_BOX_WAIT_QUIET/NY_BOX_QUIET_GRACE may reach ny_box.sh)"
  fi
  mem_check
  # the pin: sha256 = the plan's, a byte-identical renamed copy (no `pgrep '^ny'` gate sees it), --build-id
  [ -f "$PIN" ] || die 91 "pin-missing:$PIN"
  [ "$(sha256sum "$PIN" | cut -d' ' -f1)" = "$PIN_SHA" ] || die 91 "pin-sha-differs-from-the-plan"
  PROBE=$OUT/probe_kc
  if [ ! -x "$PROBE" ] || [ "$(sha256sum "$PROBE" | cut -d' ' -f1)" != "$PIN_SHA" ]; then
    cp "$PIN" "$PROBE.tmp" && chmod 755 "$PROBE.tmp" && mv "$PROBE.tmp" "$PROBE" || die 91 "copy"
  fi
  [ "$(sha256sum "$PROBE" | cut -d' ' -f1)" = "$PIN_SHA" ] || die 91 "copy-sha"
  local bid; bid=$("$PROBE" --build-id 2>/dev/null </dev/null)
  echo "  PIN $PIN sha256=$PIN_SHA"
  echo "  BUILD-ID $bid"
  case "$bid" in *"commit=$PIN_COMMIT "*) ;; *) die 91 "build-id-commit";; esac
  case "$bid" in *"features=$PIN_FEATURES "*"dirty=false"*"exe=$PIN_SHA"*) ;; *) die 91 "build-id(features/dirty/exe)";; esac
  if [ -f "$OUT/PIN_LOCK" ]; then [ "$(awk '{print $1; exit}' "$OUT/PIN_LOCK")" = "$PIN_SHA" ] || die 91 "pin-lock(one pin per KC_OUT)"; fi
  # the checker the follow-ups need, from the pin copy
  n=$( (cd "$OUT" && timeout 60 "$PROBE" proof-check --help) 2>&1 </dev/null | grep -c 'proof-check \[OPTIONS\] <BUNDLE>')
  [ "${n:-0}" -ge 1 ] || die 92 "the pin has no proof-check subcommand"
  # the resolver check can FAIL: an empty configs dir must not resolve the plan's preset
  neg=$(grep -m1 '^RESOLVER_NEG=' "$s0" | cut -d= -f2-)
  if [ -n "$neg" ]; then
    IFS=$'\t' read -r c o v b e p <<<"$neg"
    out=$( (cd "$e" && nice -n 19 timeout 300 "$PROBE" vnncomp plan "$c" "$o" "$v" "$b" --configs-dir "$e") 2>&1 </dev/null)
    [ "$(grep -cF -- "$p" <<<"$out")" -eq 0 ] || die 94 "resolver-negative-control-resolved-the-preset-from-an-empty-dir"
    echo "  resolver negative control ok: an empty configs dir does not resolve $(basename "$p")"
  fi
  # the m0 runner's own step 0 over the shard's whole spec: tools, pin, markers, zero NY_*, cgroup, the spec, and
  # the pin's own resolver per (category, configs dir) -- the exact path every run takes
  mkdir -p "$OUT/step0_m0_$SK"
  [ -e "$OUT/step0_m0_$SK/probe_m0" ] || ln "$PROBE" "$OUT/step0_m0_$SK/probe_m0" || die 98 "link-probe"
  env $(m0_env) M0_OUT="$OUT/step0_m0_$SK" bash "$M0/moat_both_profiles.sh" --check "$UNION" > "$OUT/step0_m0_$SK.txt" 2>&1; rc=$?
  grep -E 'PLAN-|markers ok|plans ok|M0MOAT_DONE|BUILD-ID' "$OUT/step0_m0_$SK.txt" | sed 's/^/  m0: /' | tail -8
  [ $rc -eq 0 ] || die "$rc" "m0-step0(read $OUT/step0_m0_$SK.txt)"
  echo "=== KC step0 PASSED shard=$SHARD batches=$NBATCH"
}

m0_phase() { # spec m0out label: one m0 runner pass (it resumes by its own markers); exits on an emergency
  local spec=$1 mout=$2 label=$3 base new last R
  mkdir -p "$mout"
  [ -e "$mout/probe_m0" ] || ln "$PROBE" "$mout/probe_m0" || die 98 "link-probe:$mout"
  base=$(grep -c '^M0MOAT_DONE ' "$mout/m0moat.log" 2>/dev/null); base=${base:-0}
  echo "  [$label] m0 runner: spec=$spec out=$mout (DONE markers before: $base) $(date -Is)"
  env $(m0_env) M0_OUT="$mout" bash "$M0/moat_both_profiles.sh" --run "$spec"; R=$?
  new=$(grep -c '^M0MOAT_DONE ' "$mout/m0moat.log" 2>/dev/null); new=${new:-0}
  [ "$new" -gt "$base" ] || die 96 "m0-runner-wrote-no-NEW-DONE-line($label m0exit:$R)"
  last=$(grep '^M0MOAT_DONE ' "$mout/m0moat.log" | tail -1)
  echo "  [$label] m0: ${last//rc=/rc:}"
  case "$last" in "M0MOAT_DONE rc=$R "*) ;; *) die 96 "m0-DONE-disagrees-with-its-exit($label $R)";; esac
  if [ "$R" -eq 3 ]; then
    if [[ "$last" == *" reason=stop-file "* ]]; then
      done_line 3 3 "status=STOPPED stopped_by=$OUT/EMERGENCY phase=$label elapsed_min=$((SECONDS/60))"; exit 3
    fi
    record_emergency "m0-runner-stopped($label): ${last#M0MOAT_DONE } (read $mout/m0moat.log)"
    echo "  ###### SOUNDNESS EMERGENCY in $label: $OUT/EMERGENCY written; nothing more runs ######"
    py judge-batch --out "$OUT" --shard "$SK" --batch "$CUR" > "$OUT/run/$SK/$CUR/judge_emergency.txt" 2>&1
    done_line 3 3 "status=EMERGENCY phase=$label m0exit=$R batch=$CUR pin=${PIN##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60))"
    exit 3
  fi
  case $R in 0|4|5) ;; *) die "$R" "m0-precondition-$label(read $mout/m0moat.log)";; esac
}

kv() { grep -m1 "^$1=" "$2" | cut -d= -f2-; }

run_batch() { # batch-id spec
  local b=$1 spec=$2 base=$OUT/run/$SK/$1 f R last tag bnd onnx vnn rep logp J JR s
  CUR=$b
  mkdir -p "$base/check"
  if [ -f "$base/BATCH.json" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("final") and d.get("rc") in (0, 4) else 1)' "$base/BATCH.json"; then
    echo "  [$b] final BATCH.json kept (resume): $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["status"], len(d["rows"]))' "$base/BATCH.json")"
    return 0
  fi
  emergency_gate
  echo "=== [$b] batch start $(date -Is) spec=$spec"
  echo "$b A $(date -Is)" > "$OUT/run/$SK/.phase"
  m0_phase "$spec" "$OUT/run/$SK/A" "A:$b"
  f=$base/reproof.txt
  py reproof --out "$OUT" --shard "$SK" --batch "$b" > "$f" 2>&1; R=$?
  sed 's/^/  [reproof] /' "$f"
  [ $R -eq 0 ] || [ $R -eq 3 ] || die 97 "reproof-derive-$b(rc:$R)"
  if [ $R -eq 3 ]; then record_emergency "phase-A emergency class in $b"; done_line 3 3 "status=EMERGENCY phase=A batch=$b"; exit 3; fi
  # (1) every phase-A sat on the organisers' checker under emulated x86 (a WRONG SAT stops before any re-proof)
  if [ "$(kv SAT "$f")" -gt 0 ]; then
    echo "$b replay $(date -Is)" > "$OUT/run/$SK/.phase"
    echo "  [replay] $(kv SAT "$f") sat row(s): one ny_box.sh build hold $(date -Is)"
    "$NY/scripts/ny_box.sh" build -- nice -n 10 python3 "$PY" replay --out "$OUT" --shard "$SK" --batch "$b" "${SARGS[@]}" </dev/null >> "$base/x86.log" 2>&1; R=$?
    last=$(grep '^X86_REPLAY_DONE ' "$base/x86.log" | tail -1)
    echo "  [replay] ${last//rc=/rc:} (box exit $R; $base/x86.log)"
    case "$last" in
      "X86_REPLAY_DONE rc=3 "*)
        record_emergency "x86 replay WRONG SAT in $b: ${last#X86_REPLAY_DONE } (read $base/x86.log)"
        echo "  ###### WRONG SAT: the organisers' checker rejects a sat witness on x86 -- EMERGENCY ######"
        py judge-batch --out "$OUT" --shard "$SK" --batch "$b" > "$base/judge_emergency.txt" 2>&1
        done_line 3 3 "status=EMERGENCY phase=replay batch=$b elapsed_min=$((SECONDS/60))"; exit 3 ;;
      "X86_REPLAY_DONE rc=0 "*) ;;
      *) echo "  [replay] not every replay passed: the batch stays VOID until they do (a relaunch re-runs them)" ;;
    esac
  fi
  # (2) one Proof-profile re-run of every phase-A unsat
  if [ "$(kv ROWS "$f")" -gt 0 ]; then
    echo "$b reproof $(date -Is)" > "$OUT/run/$SK/.phase"
    m0_phase "$(kv SPEC "$f")" "$OUT/run/$SK/R" "R:$b"
  fi
  # (3) ny proof-check on every published bundle (resume: a check that ran is kept)
  while IFS=$'\t' read -r tag bnd onnx vnn rep logp; do
    [ -n "$tag" ] || continue
    echo "$b check $tag $(date -Is)" > "$OUT/run/$SK/.phase"
    echo "  [check] $tag: proof-check $(basename "$bnd") $(date -Is)"
    "$NY/scripts/ny_box.sh" build -- nice -n 10 timeout "$CERT_TIMEOUT_S" "$PROBE" proof-check "$bnd" --onnx "$onnx" \
      --vnnlib "$vnn" --report "$rep" --budget-s "$CERT_BUDGET_S" > "$logp" 2>&1 </dev/null; R=$?
    if [ $R -eq 75 ] || [ "$(grep -c '^\[ny-box\] .*REFUS' "$logp")" -ge 1 ]; then
      mv -- "$logp" "$logp.refused.$(date +%s%N)"
      echo "  [check] $tag: the box refused (exit $R): the check stays pending"
      continue
    fi
    echo "CERT_CHECK_RAN rc=$R" >> "$logp"
    echo "  [check] $tag: exit $R $(grep -m1 '^PROOF_CHECK_DONE ' "$logp" | cut -c1-200 | sed 's/rc=/rc:/g')"
  done < <(py check-list --out "$OUT" --shard "$SK" --batch "$b")
  # (4) the re-proof outcomes (+ the route-B2 rule); the classic set once they are final
  f=$base/scan.txt
  py scan --out "$OUT" --shard "$SK" --batch "$b" > "$f" 2>&1; R=$?
  sed 's/^/  [scan] /' "$f"
  if [ $R -eq 3 ]; then
    record_emergency "certified re-proof EMERGENCY in $b: $(grep -m3 '^  emergency\|^  EMERGENCY' "$f" | tr '\n' ' ')"
    echo "  ###### a re-proof is an EMERGENCY (refuted / Proof sat / route-B2) -- EMERGENCY FILE WRITTEN ######"
    py judge-batch --out "$OUT" --shard "$SK" --batch "$b" > "$base/judge_emergency.txt" 2>&1
    done_line 3 3 "status=EMERGENCY phase=check batch=$b elapsed_min=$((SECONDS/60))"; exit 3
  fi
  [ $R -eq 0 ] || [ $R -eq 5 ] || die 96 "scan-$b(rc:$R)"
  if [ $R -eq 0 ] && [ "$(kv FROWS "$f")" -gt 0 ]; then
    echo "$b classic $(date -Is)" > "$OUT/run/$SK/.phase"
    m0_phase "$(kv FSPEC "$f")" "$OUT/run/$SK/F" "F:$b"
  fi
  # (5) the batch verdict (+ the global cross-tag instance check)
  J=$(py judge-batch --out "$OUT" --shard "$SK" --batch "$b"); JR=$?
  printf '%s\n' "$J" | sed 's/^/  [judge] /'
  s=$(grep '^BATCH_DONE ' <<<"$J" | tail -1)
  [ -n "$s" ] && [[ "$JR" =~ ^(0|3|4|5)$ ]] || die 96 "judge-batch-$b(rc:$JR)"
  if [ $JR -eq 3 ]; then
    record_emergency "batch $b: $(grep -m3 '^  EMERGENCY ' <<<"$J" | tr '\n' ' ')"
    echo "  ###### EMERGENCY in batch $b -- EMERGENCY FILE WRITTEN, EVERY SHARD STOPS ######"
    done_line 3 3 "status=EMERGENCY phase=judge batch=$b elapsed_min=$((SECONDS/60))"; exit 3
  fi
  [ $JR -gt $WORST ] && WORST=$JR
  JUDGED=$((JUDGED+1))
  echo "=== [$b] batch done $(date -Is) ${s//rc=/rc:}"
}

heartbeat() {
  ( while sleep 600; do echo "  HB $(date -Is) pid=$$ shard=$SHARD phase=$(cat "$OUT/run/$SK/.phase" 2>/dev/null || echo -) current=$(cat "$OUT/run/$SK/A/.current" "$OUT/run/$SK/R/.current" "$OUT/run/$SK/F/.current" 2>/dev/null | sort -k4 | tail -1 || echo -)"; done ) &
  HB=$!
  trap 'kill $HB 2>/dev/null' EXIT
}

launch() {
  step0
  local stamp snap unit pid cg mm envf
  stamp=$(date +%Y%m%d-%H%M%S)
  snap=$OUT/harness_${SK}_$stamp
  [ -e "$snap" ] && die 98 "snapshot-exists:$snap"
  mkdir -p "$snap/seal/x86_controls" "$snap/m0" && cp "$HERE"/*.py "$HERE"/*.sh "$snap/seal/" \
    && cp "$HERE"/x86_controls/* "$snap/seal/x86_controls/" && cp "$M0"/*.py "$M0"/*.sh "$snap/m0/" || die 98 "snapshot"
  ( cd "$HERE" && git rev-parse HEAD 2>/dev/null && git status --porcelain -- . ../m0 2>/dev/null ) > "$snap/GIT_STATE" 2>&1
  sha256sum "$snap"/seal/* "$snap"/m0/* > "$snap/SHA256SUMS" 2>/dev/null
  local label; label=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["capability"] + ("-smoke" if d.get("smoke") else ""))' "$OUT/plan.json") || die 97 "plan-label"
  if [ -n "$SAMPLE" ]; then
    label="$label-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$SAMPLE")" || die 97 "sample-label"
  fi
  unit="kc-${label}-${SK}-$stamp"
  local -a CMD=(systemd-run --user --unit="$unit" --property=WorkingDirectory="$NY" --property=MemoryMax="$MEMORY_MAX"
       --setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
       --setenv=CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" --setenv=NY_BOX_WAIT_LOCK=21600 --setenv=NY_BOX_WAIT_QUIET=21600
       --setenv=NY_BOX_QUIET_GRACE=300 --setenv=KC_OUT="$OUT" --setenv=KC_NY="$NY" --setenv=KC_MEMORY_MAX="$MEMORY_MAX"
       /bin/bash "$snap/seal/keyless_certified_run.sh" --run --shard "$SHARD" "${SARGS[@]}")
  echo "  SYSTEMD-RUN: $(printf '%q ' "${CMD[@]}")"
  "${CMD[@]}" || die 99 "systemd-run"
  pid=0
  for _ in $(seq 1 40); do pid=$(systemctl --user show -p MainPID --value "$unit" 2>/dev/null); [ "${pid:-0}" != 0 ] && break; sleep 0.25; done
  [ "${pid:-0}" != 0 ] || { tail -5 "$OUT/kc_$SK.log" 2>/dev/null; die 99 "unit-has-no-MainPID:$unit"; }
  cg=$(systemctl --user show -p ControlGroup --value "$unit" 2>/dev/null)
  case "$cg" in *"/app.slice/$unit.service") ;; *) die 99 "unit-cgroup:$cg";; esac
  mm=$(systemctl --user show -p MemoryMax --value "$unit" 2>/dev/null)
  [ "${mm:-0}" = "$(numfmt --from=iec "$MEMORY_MAX" 2>/dev/null)" ] || die 99 "unit-MemoryMax=$mm(wanted $MEMORY_MAX)"
  envf=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null)
  grep -q '^NY_BOX_WAIT_LOCK=21600$' <<<"$envf" && grep -q '^NY_BOX_WAIT_QUIET=21600$' <<<"$envf" \
    && grep -q '^NY_BOX_QUIET_GRACE=300$' <<<"$envf" || die 99 "unit-environ-lacks-box-waits(pid=$pid)"
  grep -q "^KC_OUT=$OUT\$" <<<"$envf" || die 99 "unit-environ-lacks-KC_OUT(pid=$pid)"
  grep -q '^PATH=.*\.cargo/bin' <<<"$envf" || echo "  NOTE unit PATH has no .cargo/bin (the runner needs none)"
  grep -q '^HOME=' <<<"$envf" || die 99 "unit-environ-lacks-HOME"
  echo "KC_LAUNCHED unit=$unit pid=$pid cgroup=$cg memory_max=$mm log=$OUT/kc_$SK.log snapshot=$snap sample=${SAMPLE:--}"
}

case "$MODE" in
  --check|--launch|--run|--summary) need_common ;;
  *) echo "usage: $0 --check|--launch|--run|--summary [--shard k/n]"; exit 2 ;;
esac
case "$MODE" in
  --check) step0; exit 0 ;;
  --launch) launch; exit 0 ;;
  --summary) py summary --out "$OUT"; exit $? ;;
  --run)
    exec >>"$OUT/kc_$SK.log" 2>&1
    cd "$NY" || die 93 "cd:$NY"
    echo "=== KC heartbeat start $(date -Is) pid=$$ shard=$SHARD unit=${INVOCATION_ID:-none} cwd=$(pwd) out=$OUT sample=${SAMPLE:--}"
    step0
    mem_lock_check
    [ -f "$OUT/PIN_LOCK" ] || printf '%s %s %s %s\n' "$PIN_SHA" "$PIN_COMMIT" "$PIN" "$(date -Is)" > "$OUT/PIN_LOCK" || die 98 "pin-lock-write"
    export NY_BOX_WAIT_LOCK=${NY_BOX_WAIT_LOCK:-21600} NY_BOX_WAIT_QUIET=${NY_BOX_WAIT_QUIET:-21600} NY_BOX_QUIET_GRACE=${NY_BOX_QUIET_GRACE:-300}
    echo "  step0 ok; pin=${PIN##*/} sha256=${PIN_SHA:0:16} NY_BOX_WAIT_LOCK=$NY_BOX_WAIT_LOCK NY_BOX_WAIT_QUIET=$NY_BOX_WAIT_QUIET NY_BOX_QUIET_GRACE=$NY_BOX_QUIET_GRACE"
    heartbeat
    JUDGED=0; CUR=-
    mapfile -t BATCHES < <(py batches --out "$OUT" --shard "$SK")
    [ "${#BATCHES[@]}" -eq "$NBATCH" ] || die 97 "batch-list(${#BATCHES[@]}!=$NBATCH)"
    for line in "${BATCHES[@]}"; do
      IFS=$'\t' read -r bid bspec brows <<<"$line"
      run_batch "$bid" "$bspec"
    done
    S=$(py summary --out "$OUT"); SR=$?
    printf '%s\n' "$S" | sed 's/^/  [summary] /' | tail -60
    last=$(grep '^KC_SUMMARY ' <<<"$S" | tail -1)
    [ $SR -gt $WORST ] && WORST=$SR
    st=PASS; [ $WORST -eq 4 ] && st=PARTIAL; [ $WORST -eq 5 ] && st=VOID; [ $WORST -eq 3 ] && st=EMERGENCY
    done_line "$WORST" "$WORST" "status=$st planned_batches=$NBATCH judged_now=$JUDGED ${last#KC_SUMMARY rc=[0-9] } pin=${PIN##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60))"
    exit "$WORST" ;;
esac
