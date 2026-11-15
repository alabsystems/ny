#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
#
# #phase-cost census RUNNER. Runs every row of a costcensus-spec/v1 plan (scripts/costcensus/costcensus.py plan) as
# `ny vnncomp v1 ... --proof-profile competition` under the competition's own OS backstop (`timeout budget+10`, as
# vnncomp_scripts/run_instance.sh) and `OMP_NUM_THREADS=1` (run_instance.sh's only export), ONE box acquisition
# per row through `scripts/ny_box.sh measure`, stdout+stderr kept PER ROW (raw/<tag>.log), the results file and its
# flight record kept per row (res/<tag>.txt, res/<tag>.txt.flight.json), then aggregates with costcensus.py.
#
# Modes (CC_OUT/plan.tsv is the plan; put it there before --check):
#   --check    step 0 only, to stdout: tools, pin (sha, build-id commit/dirty/exe), census markers in the pin,
#              zero NY_*, cgroup memory, the plan, and the per-row preset resolver (`vnncomp plan`) from the run's
#              own cwd and configs dir. No box, no measurement.
#   --launch   step 0, snapshot this script + costcensus.py + m0lib.py + plan into CC_OUT/harness_<stamp>/ (bash
#              reads a running script by offset; the snapshot is what runs), start it under `systemd-run --user`
#              (own cgroup, MemoryMax=96G where the CROWN dense clamp is inert -- CLAUDE.md), with PATH/HOME,
#              WorkingDirectory=$NY and BOTH box waits at 21600 s. Verifies MainPID, the unit cgroup, the environ.
#   --run      the job (what the unit executes); log: CC_OUT/census.log (append).
#   --aggregate re-judge every row and rewrite the report (no box).
#
# Environment: CC_PIN (release binary), CC_PIN_SHA16, CC_PIN_COMMIT (40 hex), CC_OUT (under CC_OUT_ROOT),
#   CC_CONFIGS (the configs dir of the pin's commit). Optional: CC_FEATURES (mip,cuda), CC_NY (<home>/ny),
#   CC_OUT_ROOT (/tmp/claude-1000), CC_LABEL, CC_REPORT (CC_OUT/PHASE_COST_REPORT.md), CC_GUARD_NS (the measured
#   release per-enter guard cost: adds the census-overhead bound to the report), CC_CHUNK_SECS (pack rows into one
#   box hold up to this worst-case wall; 0 = one hold per row).
#
# DONE line: `CENSUS_DONE rc=<r> WORST=<r> ran=<n> refused=<n> agg_rc=<r> ...` propagates the REAL rc: 3 a soundness
# emergency (a banked decided verdict contradicted), 5 VOID rows, 6 nothing usable, 0 clean, 9x precondition.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NY=${CC_NY:-<home>/ny}
MODE=${1:-}
OUT=${CC_OUT:-}
FEATURES=${CC_FEATURES:-mip,cuda}
AGG="$HERE/costcensus.py"
MEM_MIN=103079215104   # 96 GiB
VERDICT_ENV_RE='^(NY_|NYCERT_|AY_MILP_|RAYON_|PROBE_|GPU_AVAILABLE=|CUDA_VISIBLE_DEVICES=)'

die() { echo "CENSUS_DONE rc=$1 WORST=$1 reason=$2 $(date -Is)"; exit "$1"; }

need_env() {
  [ -n "$OUT" ] || die 2 "CC_OUT unset"
  local root=${CC_OUT_ROOT:-/tmp/claude-1000}
  case "$OUT" in "${root%/}"/*) ;; *) die 2 "CC_OUT must live under ${root%/}/ (scratch, never a tracked path)";; esac
  [ "$MODE" = --aggregate ] && { [ -n "${CC_PIN_COMMIT:-}" ] || die 2 "CC_PIN_COMMIT unset"; return 0; }
  [ -n "${CC_PIN:-}" ] || die 2 "CC_PIN unset"
  [[ "${CC_PIN_SHA16:-}" =~ ^[0-9a-f]{16}$ ]] || die 2 "CC_PIN_SHA16 must be 16 hex"
  [[ "${CC_PIN_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die 2 "CC_PIN_COMMIT must be 40 hex"
  [ -n "${CC_CONFIGS:-}" ] && [ -d "$CC_CONFIGS" ] || die 2 "CC_CONFIGS must be a configs dir"
  BIN=$OUT/probe_census   # a name no `pgrep '^ny'` quiescence gate matches
  SPEC=$OUT/plan.tsv
}

resolve_pin() {
  [ -f "$CC_PIN" ] && [ -x "$CC_PIN" ] || die 91 "pin-missing:$CC_PIN"
  PIN_SHA=$(sha256sum "$CC_PIN" | cut -d' ' -f1)
  [ "${PIN_SHA:0:16}" = "$CC_PIN_SHA16" ] || die 91 "pin-sha:${PIN_SHA:0:16}!=$CC_PIN_SHA16"
  if [ ! -x "$BIN" ] || [ "$(sha256sum "$BIN" | cut -d' ' -f1)" != "$PIN_SHA" ]; then cp "$CC_PIN" "$BIN" || die 91 copy; fi
  [ "$(sha256sum "$BIN" | cut -d' ' -f1)" = "$PIN_SHA" ] || die 91 "copy-sha"
  printf '%s  %s\n' "$PIN_SHA" "$CC_PIN" > "$BIN.sha256"
  local bid; bid=$("$BIN" --build-id 2>/dev/null </dev/null)
  echo "  PIN $CC_PIN -> $BIN sha256=$PIN_SHA"
  echo "  BUILD-ID $bid"
  case "$bid" in *"commit=$CC_PIN_COMMIT "*) ;; *) die 91 "build-id-commit";; esac
  case "$bid" in *"features=$FEATURES "*"dirty=false"*"exe=$PIN_SHA"*) ;; *) die 91 "build-id(features/dirty/exe)";; esac
}

mem_check() { # ny's governor reads the tightest memory.max on its cgroup path; below 96 GiB the dense clamp moves
  local cg mm p lv
  cg=$(cut -d: -f3 /proc/self/cgroup 2>/dev/null | head -1)
  mm=$(cat "/sys/fs/cgroup$cg/memory.max" 2>/dev/null || echo '?')
  echo "  cgroup: $(head -1 /proc/self/cgroup 2>/dev/null) memory.max=$mm"
  case "$mm" in max) ;; ''|*[!0-9]*) die 86 "memory.max-unreadable:$mm";; *) [ "$mm" -ge "$MEM_MIN" ] || die 86 "cap-below-96G:$mm";; esac
  p=$(dirname "$cg")
  while [ -n "$p" ] && [ "$p" != / ] && [ "$p" != . ]; do
    lv=$(cat "/sys/fs/cgroup$p/memory.max" 2>/dev/null || echo max)
    case "$lv" in max) ;; *[!0-9]*) die 86 "ancestor-unreadable:$p";; *) [ "$lv" -ge "$MEM_MIN" ] || die 86 "ancestor-cap-below-96G:$p:$lv";; esac
    p=$(dirname "$p")
  done
  [ "$(ulimit -v)" = unlimited ] || die 86 "rlimit-as-set"
}

load_rows() { # tag category onnx vnnlib budget, one per line, from the plan (validated by the aggregator's reader)
  mapfile -t ROWS < <(python3 - "$SPEC" "$AGG" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cc", sys.argv[2]); cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
for r in cc.read_spec(sys.argv[1]):
    print("\t".join(str(r[k]) for k in ("tag", "category", "onnx", "vnnlib", "budget")))
PY
)
  [ "${#ROWS[@]}" -ge 1 ] || die 97 "plan has no rows (or is not a costcensus-spec/v1 plan)"
}

step0() {
  local t c mk T C O V B out p b
  for t in sha256sum strings python3 awk grep timeout nice; do command -v "$t" >/dev/null || die 127 "tool:$t"; done
  [ -x "$NY/scripts/ny_box.sh" ] || die 93 "ny_box.sh missing"
  [ -f "$AGG" ] || die 93 "costcensus.py missing under $HERE"
  resolve_pin
  strings -a "$BIN" > "$OUT/pin_strings.txt" || die 92 strings
  # the census's own observables must be IN the pin (grep -c reads to EOF: no SIGPIPE under pipefail)
  for mk in 'ny-core/phase-cost/v1' '[phase-cost] level=' 'branch_prescore' 'domains_verified'; do
    c=$(grep -cF -- "$mk" "$OUT/pin_strings.txt"); c=${c:-0}
    [ "$c" -ge 1 ] || die 92 "marker-absent-from-pin:$mk"
  done
  echo "  census markers present in the pin"
  cd "$NY" || die 93 "cd $NY"
  if env | grep -E "$VERDICT_ENV_RE" | grep -vE '^NY_BOX_'; then die 95 "verdict-env: the verifier must run with zero NY_*"; fi
  mem_check
  load_rows
  # the preset resolver, exactly as the run calls it (same cwd, same --configs-dir), one row per category
  declare -A seen=()
  for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r T C O V B <<<"$line"
    [ -n "${seen[$C]:-}" ] && continue; seen[$C]=1
    out=$(nice -n 19 "$BIN" vnncomp plan "$C" "$O" "$V" "$B" --configs-dir "$CC_CONFIGS" 2>&1 </dev/null)
    p=$(printf '%s\n' "$out" | awk '/^preset:/ && !f {print $2; f=1}')
    b=$(printf '%s\n' "$out" | awk '/scored_budget_secs/ && !f {print $3; f=1}')
    if [ -z "$p$b" ] && [ "$(grep -c '^Error' <<<"$out")" -ge 1 ]; then
      echo "  PLAN-UNVERIFIABLE $C: $(grep -m1 '^Error' <<<"$out" | cut -c1-140) (the run's own Loading preset: line decides)"
      continue
    fi
    [ "$b" = "$B" ] || die 94 "plan-budget:$C:want=$B:got=${b:-none}"
    # `preset: none (built-in defaults)` is a category without a preset; any file must come from CC_CONFIGS
    case "$p" in ""|none|"$CC_CONFIGS"/*) ;; *) die 94 "plan-preset-outside-configs:$C:$p";; esac
    echo "  plan ok $C preset=${p:-none} budget=$b"
  done
  echo "  ${#ROWS[@]} rows; worst case $(awk -F'\t' '{s+=$5+10} END{printf "%.2f", s/3600}' <(printf '%s\n' "${ROWS[@]}")) h of budget"
}

aggregate() {
  local rep=${CC_REPORT:-$OUT/PHASE_COST_REPORT.md}
  python3 "$AGG" aggregate --out-dir "$OUT" --expect-commit "$CC_PIN_COMMIT" --report "$rep" --json "$OUT/aggregate.json" \
    --title "${CC_TITLE:-2026 stratum}" --meta "pin=${CC_PIN:-}" --meta "pin_sha256=${PIN_SHA:-$(cut -d' ' -f1 "$OUT/probe_census.sha256" 2>/dev/null)}" \
    --meta "commit=$CC_PIN_COMMIT" --meta "stratum=$OUT/plan.tsv" --meta "configs_dir=${CC_CONFIGS:-}" \
    --meta "profile=competition, OMP_NUM_THREADS=1, zero NY_*, timeout budget+10, ny_box.sh measure per $([ "${CC_CHUNK_SECS:-0}" -gt 0 ] && echo "chunk of <= ${CC_CHUNK_SECS}s worst case" || echo row)" \
    --meta "host=$(hostname) $(nproc)c" --meta "generated=$(date -Is)" ${CC_GUARD_NS:+--guard-ns "$CC_GUARD_NS"} </dev/null
}

launch() {
  echo "=== CENSUS launch $(date -Is)"
  step0
  local stamp snap unit pid cg envf v
  stamp=$(date +%Y%m%d-%H%M%S); snap=$OUT/harness_$stamp
  [ -e "$snap" ] && die 98 "snapshot-exists:$snap"
  mkdir -p "$snap" && cp "$HERE/census_run.sh" "$AGG" "$snap/" && cp "$SPEC" "$snap/plan.tsv.snapshot" || die 98 snapshot
  unit="costcensus-${CC_LABEL:-run}-$stamp"
  local -a props=(--property=WorkingDirectory="$NY" --property=MemoryMax=96G)
  local -a envs=(--setenv=PATH="$PATH" --setenv=HOME="$HOME" --setenv=NY_BOX_WAIT_LOCK=21600
                 --setenv=NY_BOX_WAIT_QUIET=21600 --setenv=NY_BOX_QUIET_GRACE=300)
  for v in CC_PIN CC_PIN_SHA16 CC_PIN_COMMIT CC_OUT CC_CONFIGS CC_FEATURES CC_NY CC_OUT_ROOT CC_LABEL CC_REPORT CC_TITLE CC_GUARD_NS CC_CHUNK_SECS; do
    if [ -n "${!v:-}" ]; then envs+=(--setenv="$v=${!v}"); fi
  done
  systemd-run --user --unit="$unit" "${props[@]}" "${envs[@]}" /bin/bash "$snap/census_run.sh" --run || die 99 systemd-run
  pid=0
  for _ in $(seq 1 40); do pid=$(systemctl --user show -p MainPID --value "$unit" 2>/dev/null); [ "${pid:-0}" != 0 ] && break; sleep 0.25; done
  [ "${pid:-0}" != 0 ] || die 99 "unit-has-no-MainPID:$unit(read $OUT/census.log)"
  cg=$(systemctl --user show -p ControlGroup --value "$unit" 2>/dev/null)
  case "$cg" in *"/app.slice/$unit.service") ;; *) die 99 "unit-cgroup:$cg";; esac
  envf=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null)
  grep -c '^NY_BOX_WAIT_LOCK=21600$' <<<"$envf" >/dev/null && grep -c '^NY_BOX_WAIT_QUIET=21600$' <<<"$envf" >/dev/null \
    || die 99 "unit-environ-lacks-box-waits(pid=$pid)"
  echo "CENSUS_LAUNCHED unit=$unit pid=$pid cgroup=$cg log=$OUT/census.log snapshot=$snap"
}

run() {
  echo "=== CENSUS heartbeat start $(date -Is) pid=$$ unit=${INVOCATION_ID:-none} cwd=$(pwd)"
  export NY_BOX_WAIT_LOCK=${NY_BOX_WAIT_LOCK:-21600} NY_BOX_WAIT_QUIET=${NY_BOX_WAIT_QUIET:-21600} NY_BOX_QUIET_GRACE=${NY_BOX_QUIET_GRACE:-300}
  step0
  ( while sleep 600; do echo "  HB $(date -Is) pid=$$ current=$(cat "$OUT/.current" 2>/dev/null || echo none)"; done ) &
  local hb=$! N=0 REFUSEDN=0 T C O V B RAW RES R f v
  local -a UNSET=()
  for v in $(compgen -e | grep -E '^NY_BOX_'); do UNSET+=(-u "$v"); done   # the box's knobs never reach the verifier
  trap 'kill '"$hb"' 2>/dev/null' EXIT
  echo "  step0 ok; pin=${CC_PIN##*/} sha256=${PIN_SHA:0:16} rows=${#ROWS[@]}"
  # CC_CHUNK_SECS > 0 packs consecutive rows into ONE box hold whose worst case (sum of budget+10) stays under it:
  # on a contended box (flock is not FIFO) one acquisition per row can cost more wall than the rows themselves.
  # Each row still gets its own process, backstop, log, results file and flight record; the box's own lines go to
  # chunks/<k>.box, never into a row's log.
  local CHUNK=${CC_CHUNK_SECS:-0} K=0 ACC=0 line
  local -a CUR=()
  mkdir -p "$OUT/chunks" || die 98 "mkdir chunks"
  flush() {
    [ ${#CUR[@]} -ge 1 ] || return 0
    K=$((K+1)); local cs=$OUT/chunks/chunk_$K.sh crc=$OUT/chunks/chunk_$K.rc box=$OUT/chunks/chunk_$K.box ln T C O V B RAW RES f attempt R
    { echo '#!/bin/bash'; echo 'set -u'
      for ln in "${CUR[@]}"; do
        IFS=$'\t' read -r T C O V B <<<"$ln"
        RAW=$OUT/raw/$T.log; RES=$OUT/res/$T.txt
        printf 'echo "ROW_START %s $(date -Is)" >>%q\n' "$T" "$crc"
        printf 'env %s OMP_NUM_THREADS=1 timeout %d %q vnncomp v1 %q %q %q %q %d --configs-dir %q --proof-profile competition >%q 2>&1 </dev/null; echo "ROW_RC %s $? $(date -Is)" >>%q\n' \
          "${UNSET[*]}" "$((B + 10))" "$BIN" "$C" "$O" "$V" "$RES" "$B" "$CC_CONFIGS" "$RAW" "$T" "$crc"
      done
      echo 'exit 0'; } > "$cs" || die 98 "chunk-script:$K"
    for ln in "${CUR[@]}"; do   # never judge a previous attempt's files as this run's
      IFS=$'\t' read -r T C O V B <<<"$ln"
      for f in "$OUT/raw/$T.log" "$OUT/res/$T.txt" "$OUT/res/$T.txt.flight.json"; do
        if [ -e "$f" ]; then mv -- "$f" "$f.stale.$(date +%s%N)" || die 98 "stale-move:$f"; fi
      done
    done
    [ "$(sha256sum "$BIN" | cut -d' ' -f1)" = "$PIN_SHA" ] || die 91 "pin-changed-before-chunk:$K"
    echo "chunk $K (${#CUR[@]} rows, worst ${ACC}s) $(date -Is)" > "$OUT/.current"
    echo "  CHUNK $K START rows=${#CUR[@]} worst=${ACC}s $(date -Is)"
    R=75
    for attempt in 1 2 3; do
      "$NY/scripts/ny_box.sh" measure -- bash "$cs" >"$box" 2>&1 </dev/null; R=$?
      if [ $R -eq 75 ] || [ "$(grep -c 'REFUSING TO MEASURE' "$box")" -ge 1 ]; then
        echo "  REFUSED (box) chunk $K attempt $attempt rc=$R"; cp "$box" "$box.refused$attempt"; R=75; continue
      fi
      break
    done
    for ln in "${CUR[@]}"; do
      IFS=$'\t' read -r T C O V B <<<"$ln"
      RAW=$OUT/raw/$T.log
      if [ $R -eq 75 ] || [ "$(grep -c "^ROW_RC $T " "$crc" 2>/dev/null)" -lt 1 ]; then
        REFUSEDN=$((REFUSEDN+1)); echo "  NOT-RUN $T (chunk $K box_rc=$R; a resume re-runs it)"; continue
      fi
      N=$((N+1)); : > "$RAW.done"
      echo "  END $T row_rc=$(grep "^ROW_RC $T " "$crc" | tail -1 | cut -d' ' -f3) result=$(grep -m1 -oE '^Result: [a-z]+' "$RAW" || echo none) $(grep -m1 -E '^\[phase-cost\] ' "$RAW" | cut -c1-160)"
    done
    echo "  CHUNK $K END box_rc=$R $(grep -E '^\[ny-box\] (acquired|END)' "$box" | tr '\n' ' ' | cut -c1-200)"
    CUR=(); ACC=0
  }
  for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r T C O V B <<<"$line"
    if [ -f "$OUT/raw/$T.log.done" ]; then continue; fi        # resume
    if [ ${#CUR[@]} -ge 1 ] && { [ "$CHUNK" -le 0 ] || [ $((ACC + B + 10)) -gt "$CHUNK" ]; }; then flush; fi
    CUR+=("$line"); ACC=$((ACC + B + 10))
  done
  flush
  local AR; aggregate; AR=$?
  local W=$AR; [ $REFUSEDN -gt 0 ] && [ $W -eq 0 ] && W=5
  echo "CENSUS_DONE rc=$W WORST=$W ran=$N refused=$REFUSEDN agg_rc=$AR pin=${CC_PIN##*/} sha256=${PIN_SHA:0:16} elapsed_min=$((SECONDS/60)) $(date -Is)"
  exit "$W"
}

case "$MODE" in
  --check) need_env; echo "=== CENSUS step-0 check only $(date -Is)"; step0; echo "=== CENSUS step-0 PASSED"; exit 0 ;;
  --launch) need_env; launch; exit 0 ;;
  --run) need_env; mkdir -p "$OUT/raw" "$OUT/res" || die 98 "mkdir:$OUT"; exec >>"$OUT/census.log" 2>&1; run ;;
  --aggregate) need_env; aggregate; exit $? ;;
  *) echo "usage: $0 --check|--launch|--run|--aggregate   (see the header)"; exit 2 ;;
esac
