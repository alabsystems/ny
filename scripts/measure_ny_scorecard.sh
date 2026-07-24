#!/bin/bash
# Resumable NY scorecard measurement: runs the REAL competition entry point
# (`ny vnncomp v1`) over each regular-track benchmark's instances.csv at the
# official per-instance timeout, writing an isolated per-run evidence directory
# (reports/measured-runs/<run-id> by default) in the official results.csv format
# plus a trailing provenance field
# (category,onnx,vnnlib,prep,RESULT,time,run_id). Legacy six-column rows remain
# valid and all scorers ignore the optional trailing field.
# Smallest benchmarks first so each completes (and scores accurately) soonest.
# Resumable: skips instances already recorded. GPU-serial (one instance at a time).
set -u
# Repo root: auto-derive from this script's location; override with NY_ROOT.
cd "${NY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}" || exit 1
# An explicit binary lets guarded/private CARGO_TARGET_DIR builds enter the
# provenance seal directly, without copying or symlinking them into the shared
# repository target directory. The start manifest resolves and hashes either
# form before execution, so an empty/bad override still fails closed.
BIN="${NY_MEASURE_BIN:-target/release/ny}"
# Materialize the compile-time root-GEMM default before provenance capture so
# an unset performance knob is still explicit in every immutable measurement.
# Preserve any caller value verbatim: Rust handles explicit faer/ndarray and
# conservatively falls back for unknown or empty values.
if [ "${NY_ROOT_GEMM+x}" != x ]; then
  case "$(uname -s 2>/dev/null):$(uname -m 2>/dev/null)" in
    Linux:aarch64|Linux:arm64) export NY_ROOT_GEMM=faer ;;
    *) export NY_ROOT_GEMM=ndarray ;;
  esac
fi
# CNF-recovery route (cnf_route.rs) needs the ay CDCL binary: auto-discover if unset.
# Without NY_AY (or ay on PATH) the route SILENTLY skips and sat_relu unsats grind in MIP.
if [ -z "${NY_AY:-}" ]; then
  for c in "$HOME/ay/target/release/ay" "$(command -v ay 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] && { export NY_AY="$c"; break; }
  done
fi
# Benchmark root: newest local mirror first; override with NY_BROOT.
BROOT="${NY_BROOT:-benchmarks/vnncomp2026/benchmarks}"
[ -d "$BROOT" ] || BROOT=benchmarks/vnncomp2025/benchmarks
CONFIGS_DIR="${NY_MEASURE_CONFIGS_DIR:-}"
# NY normally auto-discovers the repository configs tree from its executable.
# Make that implicit input explicit so it can be sealed before execution.
if [ -z "$CONFIGS_DIR" ] && [ -d "$PWD/configs" ]; then
  CONFIGS_DIR="$PWD/configs"
fi
PROVENANCE_CONFIG_ARGS=()
if [ -n "$CONFIGS_DIR" ]; then
  case "$CONFIGS_DIR" in
    /*) ;;
    *) echo "ERROR: NY_MEASURE_CONFIGS_DIR must be an absolute path: $CONFIGS_DIR" >&2; exit 2 ;;
  esac
  [ -d "$CONFIGS_DIR" ] || {
    echo "ERROR: NY_MEASURE_CONFIGS_DIR is not an existing directory: $CONFIGS_DIR" >&2
    exit 2
  }
  PROVENANCE_CONFIG_ARGS=(--configs-dir "$CONFIGS_DIR")
fi
RUN_ID="${NY_MEASURE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
case "$RUN_ID" in
  *[!A-Za-z0-9_.-]*|'') echo "ERROR: unsafe NY_MEASURE_RUN_ID: $RUN_ID" >&2; exit 2 ;;
esac
OUT="${NY_MEASURE_OUTPUT_DIR:-reports/measured-runs/$RUN_ID}"
[ -n "$OUT" ] || { echo "ERROR: NY_MEASURE_OUTPUT_DIR is empty" >&2; exit 2; }
ARTIFACT_ROOT="${NY_MEASURE_ARTIFACTS:-$OUT/artifacts}"
SCRATCH="${NY_SCRATCH:-${TMPDIR:-/tmp}/ny_measure_scratch}"
mkdir -p "$SCRATCH" || { echo "ERROR: cannot create scratch directory: $SCRATCH" >&2; exit 1; }
RF="$SCRATCH/ny_vnncomp_result.txt"
LOGF="$SCRATCH/ny_vnncomp_output.log"

# Likely-fast benchmarks first (small nets NY verifies quickly -> NY's core
# standing across many benchmarks becomes visible soonest), hard conv/GAN nets
# last. Override order with NY_MEASURE_CATS.
CATS="${NY_MEASURE_CATS:-cersyve tllverifybench_2023 collins_rul_cnn_2022 linearizenn_2024 dist_shift_2023 soundnessbench sat_relu acasxu_2023 metaroom_2023 nn4sys malbeware cgan_2023 cora_2024 cifar100_2024 tinyimagenet_2024 safenlp_2024}"
CAP="${NY_MEASURE_CAP:-120}"
case "$CAP" in
  *[!0-9]*|'') echo "ERROR: NY_MEASURE_CAP must be a positive integer: $CAP" >&2; exit 2 ;;
  0) echo "ERROR: NY_MEASURE_CAP must be greater than zero" >&2; exit 2 ;;
esac
WATCHDOG_GRACE=30
MAX_ROWS_PER_CATEGORY="${NY_MEASURE_MAX_ROWS_PER_CATEGORY:-0}"
case "$MAX_ROWS_PER_CATEGORY" in
  *[!0-9]*|'') echo "ERROR: NY_MEASURE_MAX_ROWS_PER_CATEGORY must be a positive integer when set" >&2; exit 2 ;;
esac
if [ -n "${NY_MEASURE_MAX_ROWS_PER_CATEGORY:-}" ] && [ "$MAX_ROWS_PER_CATEGORY" -eq 0 ]; then
  echo "ERROR: NY_MEASURE_MAX_ROWS_PER_CATEGORY must be greater than zero" >&2
  exit 2
fi
INSTANCE_INDEX="${NY_MEASURE_INSTANCE_INDEX:-0}"
case "$INSTANCE_INDEX" in
  *[!0-9]*|'') echo "ERROR: NY_MEASURE_INSTANCE_INDEX must be a positive integer when set" >&2; exit 2 ;;
esac
if [ -n "${NY_MEASURE_INSTANCE_INDEX:-}" ] && [ "$INSTANCE_INDEX" -eq 0 ]; then
  echo "ERROR: NY_MEASURE_INSTANCE_INDEX must be greater than zero" >&2
  exit 2
fi
VNNLIB_VERSION="${NY_MEASURE_VNNLIB_VERSION:-}"
case "$VNNLIB_VERSION" in
  ''|1.0|2.0) ;;
  *) echo "ERROR: NY_MEASURE_VNNLIB_VERSION must be 1.0 or 2.0" >&2; exit 2 ;;
esac

# Capture the immutable run evidence before any result row can be appended.
# This is deliberately fail-closed: a missing binary, unstable dirty worktree,
# unpinned dependency, unknown NY_* environment knob, or incomplete benchmark
# checkout leaves the measured bank untouched.
# Bash 3.2 treats an empty array expansion as unset under `set -u`. The +
# form below expands to zero arguments when no external config was selected.
START_MANIFEST=$(python3 scripts/ny_measurement_provenance.py start \
  --repo-root . \
  --binary "$BIN" \
  --benchmark-root "$BROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --scratch-dir "$SCRATCH" \
  --result-file "$RF" \
  --solver-log-file "$LOGF" \
  --categories "$CATS" \
  --timeout-cap-seconds "$CAP" \
  --watchdog-grace-seconds "$WATCHDOG_GRACE" \
  --max-rows-per-category "$MAX_ROWS_PER_CATEGORY" \
  --instance-index "$INSTANCE_INDEX" \
  --vnnlib-version "$VNNLIB_VERSION" \
  ${PROVENANCE_CONFIG_ARGS[@]+"${PROVENANCE_CONFIG_ARGS[@]}"} \
  --sweep-script "$0") || {
    echo "ERROR: refusing to measure without complete immutable start provenance" >&2
    exit 1
  }
[ -f "$START_MANIFEST" ] || {
  echo "ERROR: provenance helper did not create its reported start manifest" >&2
  exit 1
}
# Execute only the run-local copies that were sealed into the start manifest.
# Originals remain bound too and are rehashed by the completion postflight.
BIN=$(python3 -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["solver_binary"]["sealed_execution"]["path"])' \
  "$START_MANIFEST") || {
    echo "ERROR: could not recover the provenance-bound solver path" >&2
    exit 1
  }
[ -x "$BIN" ] || {
  echo "ERROR: provenance-bound solver is not executable: $BIN" >&2
  exit 1
}
# A plain build (no cuda/mip) silently measures 0/99 on cifar100/tinyimagenet
# (docs/CIFAR100_CUDA_BOOTSTRAP_BREAKTHROUGH_2026-07-15.md), so probe the sealed
# binary's compiled features and refuse to measure without both. Binaries too
# old to know --build-info exit nonzero and fail this gate the same way.
BUILD_INFO=$("$BIN" --build-info 2>/dev/null) || BUILD_INFO=""
if [ "${NY_ALLOW_NONCUDA_MEASURE:-0}" != 1 ]; then
  case "$BUILD_INFO" in
    *"cuda=on"*"mip=on"*) ;;
    *)
      echo "ERROR: solver binary was not built with the cuda+mip competition features" >&2
      echo "  --build-info reported: ${BUILD_INFO:-<unavailable>}" >&2
      echo "  Rebuild with: cargo build --release -p ny-cli --features cuda,mip" >&2
      echo "  (set NY_ALLOW_NONCUDA_MEASURE=1 only for CPU-track debugging)" >&2
      exit 2
      ;;
  esac
fi
SEALED_AY=$(python3 -c \
  'import json, sys; value=json.load(open(sys.argv[1], encoding="utf-8"))["dependencies"]["ay"]["sealed_executable"]; print("" if value is None else value["path"])' \
  "$START_MANIFEST") || {
    echo "ERROR: could not recover the provenance-bound AY path" >&2
    exit 1
  }
if [ -n "$SEALED_AY" ]; then
  export NY_AY="$SEALED_AY"
fi
CONFIGS_DIR=$(python3 -c \
  'import json, sys; value=json.load(open(sys.argv[1], encoding="utf-8"))["measurement"]["sealed_config_inputs"]; print("" if value is None else value["resolved_path"])' \
  "$START_MANIFEST") || {
    echo "ERROR: could not recover the provenance-bound config path" >&2
    exit 1
  }

_record_completion() {
  local rc=$?
  local completion_rc=0
  trap - EXIT
  trap '' HUP INT TERM
  python3 scripts/ny_measurement_provenance.py complete \
    --start-manifest "$START_MANIFEST" --exit-status "$rc" >/dev/null || completion_rc=$?
  if [ "$completion_rc" -ne 0 ]; then
    echo "ERROR: immutable measurement completion or integrity validation failed" >&2
    if [ "$rc" -eq 0 ]; then rc=1; fi
  fi
  exit "$rc"
}

_measurement_signal() {
  local signal="$1"
  local rc=1
  case "$signal" in
    HUP) rc=129 ;;
    INT) rc=130 ;;
    TERM) rc=143 ;;
  esac
  trap - "$signal"
  exit "$rc"
}

trap _record_completion EXIT
trap '_measurement_signal HUP' HUP
trap '_measurement_signal INT' INT
trap '_measurement_signal TERM' TERM

mkdir -p "$OUT" || { echo "ERROR: cannot create output directory: $OUT" >&2; exit 1; }

# Portable per-instance watchdog: macOS ships no GNU `timeout` (the Linux original
# used it), so a bare `timeout ... ny` fails "command not found" and records every
# instance as timeout/0s. Prefer gtimeout/timeout when present; else background the
# run and hard-kill it after the budget. `env` is SIP-safe here (it does not strip
# DYLD when exec'ing our own non-protected binary), so only the watchdog needs help.
_run_to() {
  local secs="$1"; shift
  if command -v gtimeout >/dev/null 2>&1; then gtimeout "$secs" "$@"; return $?; fi
  if command -v timeout  >/dev/null 2>&1; then timeout  "$secs" "$@"; return $?; fi
  "$@" & local pid=$!
  ( sleep "$secs"; kill -9 "$pid" 2>/dev/null ) & local wd=$!
  wait "$pid" 2>/dev/null; local rc=$?
  kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
  return $rc
}

for cat in $CATS; do
  if [ -n "$VNNLIB_VERSION" ]; then
    search_root="$BROOT/$cat/$VNNLIB_VERSION"
    search_depth=1
  else
    search_root="$BROOT/$cat"
    search_depth=2
  fi
  instances=$(find "$search_root" -maxdepth "$search_depth" -type f -name instances.csv 2>/dev/null | LC_ALL=C sort)
  instance_list_count=$(printf '%s\n' "$instances" | awk 'NF { count++ } END { print count + 0 }')
  if [ "$instance_list_count" -eq 0 ]; then
    if [ "$INSTANCE_INDEX" -gt 0 ] || [ -n "$VNNLIB_VERSION" ]; then
      echo "ERROR: selected category $cat has no instances.csv" >&2
      exit 1
    fi
    echo "SKIP $cat (no instances.csv)"
    continue
  fi
  if [ "$instance_list_count" -gt 1 ]; then
    echo "ERROR: $cat has multiple instances.csv files; set NY_MEASURE_VNNLIB_VERSION to 1.0 or 2.0" >&2
    printf '%s\n' "$instances" >&2
    exit 1
  fi
  inst="$instances"
  bdir=$(dirname "$inst")
  csv="$OUT/$cat.csv"
  touch "$csv" || { echo "ERROR: cannot create measurement CSV: $csv" >&2; exit 1; }
  # Do not inject the shared test_nano/test_tiny overhead probes. The official
  # result processor explicitly filters those harness checks before scoring.
  total=$(wc -l < "$inst")
  if [ "$INSTANCE_INDEX" -gt "$total" ]; then
    echo "ERROR: selected instance index $INSTANCE_INDEX exceeds $cat row count $total" >&2
    exit 1
  fi
  echo "=== $cat ($total instances) $(date +%H:%M:%S) ==="
  n=0
  run_rows=0
  while IFS=, read -r onnx vnnlib timeout; do
    n=$((n+1))
    [ -z "$onnx" ] && continue
    if [ "$INSTANCE_INDEX" -gt 0 ] && [ "$n" -ne "$INSTANCE_INDEX" ]; then
      continue
    fi
    # Resume occurrence-by-occurrence. Some official instance lists contain an
    # identical (onnx,vnnlib) pair more than once; a simple grep collapses those
    # scored rows. Compare the pair's occurrence count through this input row to
    # the number already recorded instead.
    wanted_occurrence=$(awk -F, -v o="$onnx" -v v="$vnnlib" -v upto="$n" \
      'NR <= upto && $1 == o && $2 == v { count++ } END { print count + 0 }' "$inst")
    recorded_occurrence=$(awk -F, -v o="$onnx" -v v="$vnnlib" \
      '$2 == o && $3 == v { count++ } END { print count + 0 }' "$csv")
    if [ "$recorded_occurrence" -ge "$wanted_occurrence" ]; then continue; fi
    if [ "$MAX_ROWS_PER_CATEGORY" -gt 0 ] && [ "$run_rows" -ge "$MAX_ROWS_PER_CATEGORY" ]; then
      break
    fi
    op="$bdir/$onnx"; vp="$bdir/$vnnlib"
    [ -f "$op" ] && [ -f "$vp" ] || {
      echo "ERROR: refusing to record an unarchived row with missing inputs: $cat row $n" >&2
      exit 1
    }
    ROW_BINDING=$(PYTHONDONTWRITEBYTECODE=1 python3 scripts/seal_ny_measurement_inputs.py \
      --artifact-root "$ARTIFACT_ROOT" \
      --run-id "$RUN_ID" \
      --category "$cat" \
      --instance-index "$n" \
      --onnx "$onnx" \
      --vnnlib "$vnnlib" \
      --onnx-file "$op" \
      --vnnlib-file "$vp" \
      --start-manifest "$START_MANIFEST") || {
        echo "ERROR: refusing to execute without immutable pre-run input binding" >&2
        exit 1
      }
    PREFLIGHT_MANIFEST=$(python3 -c \
      'import json, sys; print(json.loads(sys.argv[1])["preflight_manifest"])' \
      "$ROW_BINDING") || exit 1
    SEALED_OP=$(python3 -c \
      'import json, sys; print(json.loads(sys.argv[1])["onnx_file"])' \
      "$ROW_BINDING") || exit 1
    SEALED_VP=$(python3 -c \
      'import json, sys; print(json.loads(sys.argv[1])["vnnlib_file"])' \
      "$ROW_BINDING") || exit 1
    to=${timeout%%[!0-9]*}; [ -z "$to" ] && to=100
    # Conservative acceleration: cap NY's timeout (NY solves fast or not at all; the
    # extra competition budget rarely converts a timeout). Gives a LOWER-BOUND score
    # fast. Override with NY_MEASURE_CAP; the manifest records the exact cap.
    [ "$to" -gt "$CAP" ] && to=$CAP
    : > "$RF" || { echo "ERROR: cannot clear result scratch file: $RF" >&2; exit 1; }
    : > "$LOGF" || { echo "ERROR: cannot clear solver log: $LOGF" >&2; exit 1; }
    t0=$SECONDS
    if [ -n "$CONFIGS_DIR" ]; then
      _run_to $((to+WATCHDOG_GRACE)) env RUST_LOG=error "$BIN" vnncomp v1 "$cat" "$SEALED_OP" "$SEALED_VP" "$RF" "$to" --configs-dir "$CONFIGS_DIR" >"$LOGF" 2>&1
    else
      _run_to $((to+WATCHDOG_GRACE)) env RUST_LOG=error "$BIN" vnncomp v1 "$cat" "$SEALED_OP" "$SEALED_VP" "$RF" "$to" >"$LOGF" 2>&1
    fi
    solver_rc=$?
    el=$((SECONDS-t0))
    res=$(head -1 "$RF" 2>/dev/null | tr -d '\r\n ' )
    [ -z "$res" ] && res=timeout
    # RF is reused on the next instance. Before any non-missing row is recorded,
    # preserve its complete raw bytes and bind them to the exact ONNX/VNN-LIB
    # hashes and start manifest. SAT remains stricter: the helper rejects it if
    # the raw result has no counterexample assignment.
    python3 scripts/archive_vnncomp_sat_result.py \
      --result-file "$RF" \
      --solver-log-file "$LOGF" \
      --artifact-root "$ARTIFACT_ROOT" \
      --run-id "$RUN_ID" \
      --category "$cat" \
      --instance-index "$n" \
      --onnx "$onnx" \
      --vnnlib "$vnnlib" \
      --onnx-file "$op" \
      --vnnlib-file "$vp" \
      --solver-verdict "$res" \
      --solver-exit-status "$solver_rc" \
      --timeout-seconds "$to" \
      --elapsed-seconds "$el" \
      --source-csv "$csv" \
      --start-manifest "$START_MANIFEST" \
      --preflight-manifest "$PREFLIGHT_MANIFEST" >/dev/null || {
        if [ "$res" = sat ]; then
          echo "ERROR: refusing to record SAT without its complete witness artifact" >&2
        else
          echo "ERROR: refusing to record result without complete immutable row evidence" >&2
        fi
        exit 1
      }
    echo "$cat,$onnx,$vnnlib,0,$res,$el,$RUN_ID" >> "$csv" || exit 1
    run_rows=$((run_rows+1))
  done < "$inst"
  echo "  done $cat: $(wc -l < "$csv")/$total recorded"
done
echo "=== ALL BENCHMARKS SWEPT $(date +%H:%M:%S) ==="
