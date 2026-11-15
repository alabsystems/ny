#!/bin/bash
# Regenerate the two censuses this campaign selects work from.
#
# WHY THIS EXISTS: censuses DECAY. On 2026-08-31 upstream landings alone moved
# every near-threshold cifar100 row by +0.001..+0.008 in three hours — the same
# magnitude as the best mechanism available — and one row (idx_1798) went from
# 0/6 to 5/8 convertible with NO config change. A census more than a few
# upstream commits old understates every row.
#
#   FRONTIER census — distance to closure for UNSOLVED rows. Answers "is this
#     target in reach?" Both of this campaign's mechanism-driven conversions came
#     from its near-threshold band, and it predicted convertibility in exact rank
#     order (the three converted rows were the three nearest).
#   MARGIN census — slack to the wall for PROVING rows. Answers "what will this
#     cost?" A lever costing X seconds endangers exactly the rows with margin < X.
#     It retrospectively named both proofs `interm_alpha: 5` destroyed.
#
# These populations are DISJOINT, which is why a frontier measurement alone
# cannot see a wall-costing lever's real price.
#
# Usage:  scripts/refresh_censuses.sh <benchmark> [budget_s]
#   e.g.  scripts/refresh_censuses.sh cifar100_2024 100
#
# Runs the SHIPPED config with zero NY_* — what competition would score.
set -uo pipefail
BM="${1:?usage: refresh_censuses.sh <benchmark> [budget_s]}"
BUD="${2:-100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/benchmarks/vnncomp2025/benchmarks/$BM"
CSV="$ROOT/reports/measured/$BM.csv"
OUT="${CENSUS_OUT:-/tmp/census_$BM_$$}"
mkdir -p "$OUT"
[ -f "$CSV" ] || { echo "no bank at $CSV"; exit 2; }
[ -d "$BENCH" ] || { echo "no benchmark dir at $BENCH"; exit 2; }

PIN="$OUT/ny_pin"
cp -f "$ROOT/target/release/ny" "$PIN" || { echo "build first: ny_box.sh build -- targo --unverified build --release -p ny-cli --bin ny"; exit 2; }
echo "[pin] $(sha256sum "$PIN" | cut -d' ' -f1)"

# unsolved rows -> frontier census; solved (unsat) rows -> margin census
awk -F, '$5=="timeout"||$5=="unknown"{o=$2;v=$3;sub(/^\.\//,"",o);sub(/^\.\//,"",v);print o"\t"v"\tpool"}' "$CSV" >  "$OUT/work.tsv"
awk -F, '$5=="unsat" && $3!~/test_nano/{o=$2;v=$3;sub(/^\.\//,"",o);sub(/^\.\//,"",v);print o"\t"v"\tmoat"}' "$CSV" >> "$OUT/work.tsv"
echo "[work] $(grep -c 'pool$' "$OUT/work.tsv") unsolved + $(grep -c 'moat$' "$OUT/work.tsv") proving rows"

printf "row\tphase\tverdict\twall\tworst\n" > "$OUT/raw.tsv"
while IFS=$'\t' read -r onnx vn phase; do
  [ -n "$vn" ] || continue
  row=$(basename "$vn" .vnnlib | sed 's/.*prop_//; s/_eps.*//')
  log="$OUT/$row.err"
  # zero NY_*: the shipped preset is the only thing arming anything
  env $(env | sed -n 's/^\(NY_[A-Za-z0-9_]*\)=.*/-u \1/p' | tr '\n' ' ') \
      NY_CONFIGS_DIR="$ROOT/configs" OMP_NUM_THREADS=1 NY_MARGIN_ROW_TRACE=1 \
      "$PIN" vnncomp v1 "$BM" "$BENCH/$onnx" "$BENCH/$vn" "$OUT/$row.res" "$BUD" \
      > "$OUT/$row.out" 2> "$log"
  R=$(grep -ah "Result:" "$log" "$OUT/$row.out" | tail -1 | sed 's/.*Result:[[:space:]]*//' | awk '{print $1}')
  W=$(grep -ah "wall_s=" "$log" | tail -1 | grep -o "wall_s=[0-9.]*" | cut -d= -f2)
  WO=$(grep -a "frontier worst" "$log" | tail -1 | sed -n 's/.*frontier worst[[:space:]]*\([-0-9.e+]*\).*/\1/p')
  grep -qa "^Error:" "$log" && R="VOID"   # an errored run is VOID, never a verdict
  printf "%s\t%s\t%s\t%s\t%s\n" "$row" "$phase" "${R:-NONE}" "${W:-NA}" "${WO:-NA}" >> "$OUT/raw.tsv"
  echo "  $row [$phase] -> ${R:-NONE} wall=${W:-NA} worst=${WO:-NA}"
  rm -f "$OUT/$row.out"
done < "$OUT/work.tsv"

# FRONTIER census: header OUTSIDE the sort (piping it through sort buries it
# mid-file and any NR>1 extraction then silently drops the NEAREST row)
{ printf "row\tdistance_to_closure\tverdict\n"
  awk -F'\t' 'NR>1&&$2=="pool"&&$5!="NA"&&$5!=""{print $1"\t"$5"\t"$3}' "$OUT/raw.tsv" | sort -k2 -g -r
} > "$OUT/frontier_census.tsv"

# MARGIN census: budget - wall for rows that still prove
{ printf "row\twall\tmargin_s\trisk\n"
  awk -F'\t' -v B="$BUD" 'NR>1&&$2=="moat"&&$3=="unsat"&&$4!="NA"{m=B-$4;
       printf "%s\t%s\t%.1f\t%s\n",$1,$4,m,(m<10?"AT-RISK":(m<20?"thin":"safe"))}' "$OUT/raw.tsv" | sort -k3 -g
} > "$OUT/margin_census.tsv"

echo
echo "=== FRONTIER: nearest 10 unsolved (select targets from here, NOT banked wall) ==="
tail -n +2 "$OUT/frontier_census.tsv" | head -10 | sed 's/^/  /'
echo "=== MARGIN: rows a wall-costing lever would endanger (margin < 20s) ==="
awk -F'\t' 'NR>1&&$3+0<20' "$OUT/margin_census.tsv" | sed 's/^/  /'
echo "=== MOAT ==="
awk -F'\t' 'NR>1&&$2=="moat"{n++; if($3!="unsat")b++} END{printf "  %d proving rows, %d no longer prove\n",n,b+0}' "$OUT/raw.tsv"
echo
echo "written: $OUT/frontier_census.tsv  $OUT/margin_census.tsv"
