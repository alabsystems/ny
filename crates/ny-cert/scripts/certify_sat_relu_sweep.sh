#!/usr/bin/env bash
# certify_sat_relu_sweep.sh — reproduce the sat_relu UNSAT certification
# pipeline end-to-end for ONE named benchmark instance:
#
#     artifact (ny CNF recovery)  ->  ay LRAT refutation  ->  Lean transcript
#     ->  Lean kernel check (plain `decide`, no `native_decide`)
#
# Usage:
#     certify_sat_relu_sweep.sh <instance> [<bench_dir>]
#
#     <instance>   e.g. unsat_v30_c38   (an unsat_* row of instances.csv)
#     <bench_dir>  sat_relu benchmark root (default: /tmp/vc25/benchmarks/sat_relu)
#
# Environment:
#     NY   path to the `ny` binary   (default: <repo>/target/release/ny)
#     AY   path to the `ay` binary   (required)
#
# Pipeline, step by step (this is exactly how the 45 modules under
# proofs/lean/NyProof/SatReluSweep/ were produced):
#
#  1. Run `ny vnncomp v1 sat_relu <onnx> <vnnlib> <result> 100` with an
#     ISOLATED $TMPDIR.  ny's cnf_route driver detects the SAT-encoded ReLU
#     gadget, recovers the CNF literal-for-literal, and writes it to
#     $TMPDIR/ny_cnf_<pid>_<nanos>.cnf before handing it to ay.  The private
#     TMPDIR makes the (pid/nanos-unique) file trivially findable.
#  2. Re-solve the snapshotted DIMACS with `ay solve --proof x.lrat
#     --proof-format lrat`.  ay emits LRAT natively and self-verifies the
#     proof ("verify-proof: ... verified"); we still do not trust it — the
#     Lean kernel re-checks everything.
#  3. `lrat_to_lean --fast` transcribes CNF+LRAT into a self-contained Lean
#     module (Formula literal + RStep list + `by decide` replay).  The tool
#     is syntax-only plumbing and FAILS CLOSED on anything it does not
#     understand (RAT steps with negative hints, unknown ids, missing empty
#     clause).
#  4. `lake build NyProof.SatReluSweep.<Name>` replays the refutation in
#     the Lean kernel and composes `checkRefutationFast_sound` with
#     `SatReluVerdict.safe_of_unsat`, printing the axiom manifest for
#     `check_ok` / `instance_unsat` / `instance_safe`.  The trust base must
#     be exactly [propext, Classical.choice, Quot.sound] — `native_decide`
#     (Lean.ofReduceBool) never appears.
#
# The whole-sweep target is `lake build NyProof.SatReluSweepAll`.

set -euo pipefail

INST=${1:?usage: certify_sat_relu_sweep.sh <instance e.g. unsat_v30_c38> [<bench_dir>]}
BENCH=${2:-/tmp/vc25/benchmarks/sat_relu}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NY_CERT_DIR=$(dirname "$SCRIPT_DIR")                       # crates/ny-cert
REPO=$(cd "$NY_CERT_DIR/../.." && pwd)                     # repo root
LEAN_DIR=$NY_CERT_DIR/proofs/lean

NY=${NY:-$REPO/target/release/ny}
AY=${AY:?error: set AY to the path of the ay binary (e.g. <ay checkout>/target/release/ay)}

if [[ ! "$INST" =~ ^unsat_v([0-9]+)_c([0-9]+)$ ]]; then
    echo "error: instance '$INST' does not match unsat_v<N>_c<M>" >&2; exit 1
fi
V=${BASH_REMATCH[1]}; C=${BASH_REMATCH[2]}
NAME="V${V}C${C}"                       # Lean file / lake module leaf name
MOD="SatReluSweep_v${V}c${C}"           # namespace under Crownproof

ONNX=$BENCH/onnx/$INST.onnx
VNNLIB=$BENCH/vnnlib/$INST.vnnlib
[[ -f $ONNX && -f $VNNLIB ]] || { echo "error: $ONNX / $VNNLIB missing" >&2; exit 1; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/sat_relu_sweep.XXXXXX")
echo "== work dir: $WORK"

# -- 1. ny: recover the CNF (isolated TMPDIR so the temp DIMACS is findable) --
mkdir -p "$WORK/nytmp"
echo "== [1/4] ny vnncomp v1 sat_relu $INST"
TMPDIR=$WORK/nytmp NY_AY=$AY "$NY" vnncomp v1 sat_relu \
    "$ONNX" "$VNNLIB" "$WORK/result.txt" 100 > "$WORK/ny.log" 2>&1
RES=$(cat "$WORK/result.txt")
[[ $RES == unsat ]] || { echo "error: ny result '$RES' != unsat (see $WORK/ny.log)" >&2; exit 1; }
CNFS=("$WORK"/nytmp/ny_cnf_*.cnf)
[[ ${#CNFS[@]} -eq 1 ]] || { echo "error: expected exactly 1 recovered CNF, got ${#CNFS[@]}" >&2; exit 1; }
cp "${CNFS[0]}" "$WORK/$INST.cnf"
echo "   recovered $(head -1 "$WORK/$INST.cnf")"

# -- 2. ay: LRAT refutation of the snapshotted CNF -----------------------------
echo "== [2/4] ay solve --proof-format lrat"
"$AY" solve "$WORK/$INST.cnf" --proof "$WORK/$INST.lrat" --proof-format lrat \
    > "$WORK/ay.log" 2>&1 || true
grep -q '^s UNSATISFIABLE' "$WORK/ay.log" || { echo "error: ay did not report UNSAT" >&2; exit 1; }
grep -q 'verify-proof: .* verified' "$WORK/ay.log" || { echo "error: ay proof self-check failed" >&2; exit 1; }
echo "   $(grep -c '' "$WORK/$INST.lrat") LRAT lines, self-verified by ay"

# -- 3. lrat_to_lean: syntax-only transcription (fails closed) ----------------
echo "== [3/4] lrat_to_lean --fast -> NyProof/SatReluSweep/$NAME.lean"
mkdir -p "$LEAN_DIR/NyProof/SatReluSweep"
cargo run --quiet --release -p ny-cert --bin lrat_to_lean -- \
    "$WORK/$INST.cnf" "$WORK/$INST.lrat" \
    "$LEAN_DIR/NyProof/SatReluSweep/$NAME.lean" "$MOD" --fast

# -- 4. Lean kernel check ------------------------------------------------------
echo "== [4/4] lake build NyProof.SatReluSweep.$NAME"
export PATH="$HOME/.elan/bin:$PATH"
cd "$LEAN_DIR"
lake build "NyProof.SatReluSweep.$NAME"
echo "== certified: Crownproof.$MOD.instance_safe (axioms above must be"
echo "   exactly [propext, Classical.choice, Quot.sound])"
