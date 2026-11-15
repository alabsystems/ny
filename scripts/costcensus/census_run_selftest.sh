#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end selftest of census_run.sh on FAKES (no ny binary, no box, no cargo): a fake `ny_box.sh` that runs its
# command, a fake pin that answers --build-id / `vnncomp plan` / `vnncomp v1` and writes a phase_cost flight record.
# Checks: a chunked run (several rows per box hold) judges every row OK and aggregates; a resume runs nothing again;
# a box that refuses every hold yields a non-zero DONE (never a silent pass); a leaked NY_* refuses before measuring;
# a pin without the census markers refuses at step 0; CC_OUT outside the scratch root refuses.
# Last line: CENSUS_RUN_SELFTEST_DONE rc=<0|1> passed=<n> failed=<n>.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${CC_OUT_ROOT:-/tmp/claude-1000}
mkdir -p "$ROOT" || exit 1
W=$(mktemp -d "$ROOT/census_run_selftest.XXXXXX") || exit 1
P=0; FL=0
ok() { echo "test_$1 ... ok"; P=$((P+1)); }
bad() { echo "test_$1 ... FAILED ($2)"; FL=$((FL+1)); }

mkdir -p "$W/ny/scripts" "$W/ny2/scripts" "$W/cfg/vnncomp26" "$W/bench"
cat > "$W/ny/scripts/ny_box.sh" <<'EOF'
#!/bin/bash
while [ "$1" != "--" ]; do shift; done; shift
echo "[ny-box] acquired lock after 0s wait"; "$@"; rc=$?; echo "[ny-box] END rc=$rc elapsed=0s"; exit $rc
EOF
printf '#!/bin/bash\necho "REFUSING TO MEASURE: hot"\nexit 75\n' > "$W/ny2/scripts/ny_box.sh"
chmod +x "$W/ny/scripts/ny_box.sh" "$W/ny2/scripts/ny_box.sh"
COMMIT=cccccccccccccccccccccccccccccccccccccccc
cat > "$W/fakeny.sh" <<EOF
#!/bin/bash
SELF=\$(readlink -f "\$0"); SHA=\$(sha256sum "\$SELF" | cut -d' ' -f1)
case "\$1" in
  --build-id) echo "NY_BUILD_ID_V1 version=0 features=mip,cuda backend=cpu commit=$COMMIT dirty=false exe=\$SHA"; exit 0;;
  vnncomp)
    if [ "\$2" = plan ]; then echo "preset:   $W/cfg/vnncomp26/\$3.yaml"; echo "  scored_budget_secs  = \$6"; exit 0; fi
    cat=\$3; res=\$6; b=\$7
    echo "Running VNN-COMP instance: category='\$cat' onnx='\$4' vnnlib='\$5' results='\$res' timeout=\${b}s"
    echo "[proof-profile] mode=competition source=cli"
    echo "unsat" > "\$res"
    python3 - "\$res" "\$cat" <<'PY'
import json, os, sys
res, cat = sys.argv[1], sys.argv[2]
amb = {k: v for k, v in os.environ.items() if k.startswith("NY_") or k == "OMP_NUM_THREADS"}
ph = [{"phase": "bab_loop", "wall_s": 1.5, "thread_s": 1.5, "self_s": 1.0, "enters": 3, "units": 0, "unit": "domains"},
      {"phase": "child_bound", "wall_s": 0.5, "thread_s": 0.5, "self_s": 0.5, "enters": 30, "units": 30, "unit": "domains"}]
rec = {"schema_version": 3, "build_id": "NY_BUILD_STAMP_V1 commit=$COMMIT dirty=false", "category": cat,
       "ambient_env": amb, "events": [{"method": "run_complete", "status": "complete", "reason": "unsat", "at_secs": 2.0}],
       "phase_cost": {"schema": "ny-core/phase-cost/v1", "level": "coarse", "elapsed_s": 2.0, "rss_samples": 2, "phases": ph,
                      "edges": [{"parent": "root", "child": "bab_loop", "thread_s": 1.5},
                                {"parent": "bab_loop", "child": "child_bound", "thread_s": 0.5}],
                      "counters": {"domains_explored": 30, "domains_verified": 10}}}
json.dump(rec, open(res + ".flight.json", "w"))
PY
    echo "[phase-cost] level=coarse elapsed_s=2.00 phases=2 enters=33 top=bab_loop:1.50"
    echo "Result: unsat"; exit 0;;
esac
exit 2
# markers: ny-core/phase-cost/v1 [phase-cost] level= branch_prescore domains_verified
EOF
chmod +x "$W/fakeny.sh"
cp "$W/fakeny.sh" "$W/fake_nomarkers.sh"; sed -i '$d' "$W/fake_nomarkers.sh"
SHA=$(sha256sum "$W/fakeny.sh" | cut -c1-16)
mkspec() { # dir rows...
  local d=$1; shift; mkdir -p "$d"
  { echo "# costcensus-spec/v1"; printf 'tag\tyear\tcategory\tindex\tonnx\tvnnlib\tbudget\tbank_verdict\tbank_wall\tstratum\n'
    local i=0 c; for c in "$@"; do i=$((i+1)); printf '%s__i%d\t2026\t%s\t%d\t%s/m.onnx\t%s/p%d.vnnlib\t%d\tunsat\t1.0\tsolved\n' \
      "$c" "$i" "$c" "$i" "$W/bench" "$W/bench" "$i" $((20 * i)); done; } > "$d/plan.tsv"
}
run() { # out ny extra-env... -- mode
  local out=$1 nydir=$2; shift 2
  env -u NY_BOX_WAIT_LOCK -u NY_BOX_WAIT_QUIET CC_OUT_ROOT="$ROOT" CC_NY="$nydir" CC_PIN="${PIN:-$W/fakeny.sh}" \
    CC_PIN_SHA16="${PINSHA:-$SHA}" CC_PIN_COMMIT=$COMMIT CC_OUT="$out" CC_CONFIGS="$W/cfg" "$@"
}

mkspec "$W/o1" cat_a cat_b cat_c cat_d
(cd /tmp && run "$W/o1" "$W/ny" CC_CHUNK_SECS=100 CC_GUARD_NS=100 bash "$HERE/census_run.sh" --run); r=$?
last=$(grep -E '^CENSUS_DONE' "$W/o1/census.log" | tail -1); nch=$(ls "$W/o1/chunks"/*.sh 2>/dev/null | wc -l)
if [ $r -eq 0 ] && [[ "$last" == *"rc=0 "*"ran=4 "* ]] && [ "$nch" -eq 3 ] \
   && [ "$(grep -c 'COSTCENSUS_AGG_DONE rc=0 rows=4 ok=4' "$W/o1/census.log")" -eq 1 ]; then ok chunked_run_judges_every_row
else bad chunked_run_judges_every_row "rc=$r chunks=$nch last=$last"; fi
if [ "$(grep -c 'NY_BOX' "$W/o1/res/cat_a__i1.txt.flight.json")" -eq 0 ] && [ "$(grep -c '"OMP_NUM_THREADS": "1"' "$W/o1/res/cat_a__i1.txt.flight.json")" -eq 1 ]; then
  ok box_knobs_never_reach_the_verifier; else bad box_knobs_never_reach_the_verifier "ambient leaked"; fi
(cd /tmp && run "$W/o1" "$W/ny" CC_CHUNK_SECS=100 bash "$HERE/census_run.sh" --run); r=$?
last=$(grep -E '^CENSUS_DONE' "$W/o1/census.log" | tail -1)
if [ $r -eq 0 ] && [[ "$last" == *"ran=0 "* ]]; then ok a_resume_runs_nothing_again; else bad a_resume_runs_nothing_again "$last"; fi
mkspec "$W/o2" cat_a cat_b
(cd /tmp && run "$W/o2" "$W/ny2" CC_CHUNK_SECS=0 bash "$HERE/census_run.sh" --run); r=$?
last=$(grep -E '^CENSUS_DONE' "$W/o2/census.log" | tail -1)
if [ $r -ne 0 ] && [[ "$last" == *"refused=2 "* ]]; then ok a_refusing_box_is_never_a_pass; else bad a_refusing_box_is_never_a_pass "rc=$r $last"; fi
mkspec "$W/o3" cat_a
out=$(cd /tmp && run "$W/o3" "$W/ny" NY_LEAK=1 bash "$HERE/census_run.sh" --check 2>&1); r=$?
if [ $r -eq 95 ]; then ok a_leaked_ny_env_refuses_before_measuring; else bad a_leaked_ny_env_refuses_before_measuring "rc=$r"; fi
out=$(cd /tmp && PIN="$W/fake_nomarkers.sh" PINSHA=$(sha256sum "$W/fake_nomarkers.sh" | cut -c1-16) run "$W/o3" "$W/ny" bash "$HERE/census_run.sh" --check 2>&1); r=$?
if [ $r -eq 92 ]; then ok a_pin_without_the_census_refuses_at_step0; else bad a_pin_without_the_census_refuses_at_step0 "rc=$r"; fi
out=$(CC_OUT=<home>/out bash "$HERE/census_run.sh" --check 2>&1); r=$?
if [ $r -eq 2 ]; then ok an_out_dir_outside_scratch_refuses; else bad an_out_dir_outside_scratch_refuses "rc=$r"; fi
RC=$([ $FL -eq 0 ] && [ $P -gt 0 ] && echo 0 || echo 1)
echo "CENSUS_RUN_SELFTEST_DONE rc=$RC passed=$P failed=$FL work=$W"
exit "$RC"
