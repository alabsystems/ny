# Offline priors v1: the cost-model pipeline (#router-r7)

ROUTER_PROGRAM §5.5 / §10 R7. This directory builds the ONE compiled prior table the router may read,
`crates/ny-core/src/router/priors_v1.tsv`, and the anonymised unit manifest its Rust gates re-derive it from,
`crates/ny-core/tests/data/priors_v1_rows.tsv`. Nothing here runs cargo, an ny binary or the box.

## What is learned, and what is never learned

A **cost model per phase**: `log2(seconds) = intercept[cell] + slope * log2(params)` with an upper quantile
`+ spread_k * spread[cell]`. It predicts TIME. It never predicts a verdict, a route, a lane's success or a
category. The generalization audit (2026-09-28) refuted static-feature auto-selection because size features
fingerprint the corpus (`n_params` alone isolates 24/24 scored categories); so:

- the **key** is two operator bits, `conv` (Conv / ConvTranspose) and `nonpwl` (Sigmoid / Tanh / Softmax /
  var x var product / normalisation), never a size, budget, input dim or operator-signature id;
- the continuous size enters through **one exponent shared by every cell** of a phase;
- Layer-B success curves are **uniform** (they may fit only generic-board rows, and none exist).

## The gates (each recorded in the table; the reader refuses a table that breaks one)

| gate | rule | where re-checked |
|---|---|---|
| G1 | cells are `*` or `conv=<0|1>;nonpwl=<0|1>`; work is `params` | `PriorTable::validate`, `priors_outputs_are_not_keyed_by_name` |
| G2 | every shipped cell spans >= 2 family-level leakage groups, dominant share <= 0.6; failing cells collapse to `*` | `priors_cells_span_two_families_within_purity` |
| G3 | per-cell-majority classifier on the key, trained on one model-hash fold and scored on the other (family-balanced), beats the majority baseline by <= 0.20 — with a positive control: 1-NN on the raw size beats it by > 0.25 | `priors_key_cannot_identify_family_beyond_bound_on_held_out_models` |
| G4 | no shipped cell holds one family | evidence `isolating=0` |
| G5 | 0 <= shared slope <= 2, identical across a phase's cells | `validate` (refuses per-cell slopes) |
| G6 | leave-one-family-out overfit gap (median held-out − full error, log2) <= 0.5, and LOGO does not lose to a constant null | `priors_logo_holds_out_each_family_entirely` |
| G7 | the phase passes G2–G6 again in EVERY leave-one-family-out refit | same test; `a_pass_that_depends_on_one_family_is_refused_by_g7` |

## The training set (v1)

- **Source**: every `<results>.flight.json` receipt under `<home>/ny/scratchpad` whose identity is
  recoverable from a raw log's `Running VNN-COMP instance:` line (2014 identities), on the GB10 (20 logical
  cores, ~122 GiB; the Mac's receipts are excluded), with 1-min load <= 6 at the receipt's start (232
  dropped), joined BY MODEL to `scratchpad/vbs/general/instance_features.tsv` (`n_params_init` = the plan
  resolver's initializer element count). Census: `data/priors_v1_census.json`.
- **Committed**: `data/priors_v1_trainset.tsv` (2441 observations from 1665 receipts, with category, family,
  model md5 and the receipt path), so `check` reproduces the table without the scratchpad.
- **Phases**: `attack.upfront` (upfront attack start to close, kept only when it ended exhausted before 0.9 x
  its own cap: 492 obs / 9 families), `verifier.setup` (margin-row allocation hand-off to the verifier's
  first internal event: 1661 / 18), `root.bound` (crown precheck seconds per clause, kept when every clause
  verified or the pass ended before a quarter of the budget: 288 / 7). Censored observations are DROPPED.
- **Families**: GENERALIZATION_DESIGN §9.1 md5 groups plus the K4.3 merges (acasxu triple, cgan pair,
  soundnessbench pair, cifar100 + tinyimagenet + cct, yolo pair, cora + relusplitter pair).

## v1 result

| phase | ships | why |
|---|---|---|
| `verifier.setup` | yes | 18 families, slope 0.271, cells `conv=0;nonpwl=0` / `conv=0;nonpwl=1` / `conv=1;nonpwl=0`; LOGO gap 0.036, 10 W / 8 L vs null; key excess 0.107; G7 18 of 18 |
| `attack.upfront` | **no** | passes G2–G6 on the full data (gap 0.372) but only 3 of 9 leave-one-family-out refits still pass: its pass is single families' models (G7) |
| `root.bound` | **no** | LOGO gap 3.56, loses to the null 1 W / 6 L, key excess 0.267, G7 0 of 7 |

So v1 **prices no first rung**: every binding needs a refused phase (`attack.upfront_apgd` ->
`attack.upfront`; the verifier routes -> `verifier.setup + root.bound`). The router reads the table under
`optimizer.priors: true` and logs the `verifier.setup` prediction; the planning consumers (CR-C contracts,
CR-A's rung fit) are wired and tested on synthetic tables and views, and inert on the compiled one.

## Commands

```
python3 scripts/router_priors/build_priors.py collect --roots <home>/ny/scratchpad \
    --features <home>/ny/scratchpad/vbs/general/instance_features.tsv \
    --out scripts/router_priors/data/priors_v1_trainset.tsv --census scripts/router_priors/data/priors_v1_census.json
python3 scripts/router_priors/build_priors.py fit --rows scripts/router_priors/data/priors_v1_trainset.tsv \
    --table crates/ny-core/src/router/priors_v1.tsv --fixture crates/ny-core/tests/data/priors_v1_rows.tsv
python3 scripts/router_priors/build_priors.py check --rows ... --table ... --fixture ...   # exit 1 on drift
python3 scripts/router_priors/priors_selftest.py   # PRIORS_SELFTEST_DONE rc=0 ...
python3 scripts/router_priors/runtime_key_crosscheck.py \
    --features <home>/ny/scratchpad/vbs/general/instance_features.tsv --benchmarks <home>/ny/benchmarks
    # KEY_CROSSCHECK_DONE rc=0 models=187 agree=187 key_mismatch=0 params_mismatch=0 missing=0 (2026-10-02)
```

## The training key vs the runtime key (review, 2026-10-02)

The pipeline keys a unit from the static features table (live-node op flags); the router keys a row from the
plan resolver's protobuf scan (every node; operand constness from initializers only).
`runtime_key_crosscheck.py` re-implements the Rust scan + `op_kinds` (a second implementation, so it can
drift; the authoritative check needs a binary) and finds the two keys AND the parameter counts equal on all
187 training models. The DEFINITIONS still differ off-corpus, which matters for unseen rows (the GEN-R and
sealed-holdout point): a Mul / MatMul / Gemm / Div whose constant comes from a `Constant` NODE is bilinear
(`nonpwl=1`) at runtime and static in the features table; Group/InstanceNormalization is `nonpwl` at runtime
only; the features table ignores dead nodes. A prior refit on new families should key its units with the
runtime definition.

## What the anti-fingerprint bound covers (review, 2026-10-02)

G3 bounds a classifier on the KEY. The prior's full input (key + size) and so its logged per-row prediction
DO identify the family on held-out models (MEASURED on this training set: 1-NN on (key, log2 size) beats the
majority baseline by 0.411 / 0.415 / 0.425 for attack.upfront / verifier.setup / root.bound; 1-NN on the
logged verifier.setup median by ~0.35; bound 0.20). A size-bearing cost model cannot pass a classifier-on-
inputs ratchet. Its guard is capacity (one exponent, at most one intercept per closed-vocabulary cell) checked
out of family by G6/G7, pinned by `priors_outputs_carry_the_size_fingerprint_so_the_guard_is_capacity`.

`fit` is the only writer of the table; never hand-edit it (the `check` command and
`the_fixture_is_anonymous_and_the_committed_table_has_no_drift` both catch it, and the table's sha256 is
part of the version token every plan line and receipt carries).
