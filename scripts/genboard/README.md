# GEN-R generic board harness (#router-r7)

ROUTER_PROGRAM §9.8 (the headline generality metric) and §10 R7. **Built and selftested; never run on the
box.** Python 3 + PyYAML only (reuses `scripts/m0/m0lib.py` for instances, banks, field truth and the preset
resolver mirror, and `scripts/router_priors/priorslib.py` for families and the prior refit).

## The board

- **GEN-R** — the pinned binary with a configs dir holding ONLY the generic default: one byte-identical yaml
  per category (so the resolver finds it) whose only keys are `router.mode`, `optimizer.mode`,
  `optimizer.priors`. No category yaml, no per-category arming. `genboard.py configs` writes it and
  `verify_configs` refuses any dir with two distinct texts, a non-generic key, or a category that resolves to
  nothing (AUTO DEFAULTS would run instead of GEN-R).
- **TUNED** — the SAME pin with the shipped `configs/`, run by the same runner (`--arm tuned`). A bank made by
  another pin is accepted only with `--tuned-bank` and every report line says `NOT-SAME-PIN`.
- **Headline**: GEN-R / TUNED raw per year (target >= 0.95), rows gained / lost, the per-family sign test.

## Protocol (§9.7–9.9, GENERALIZATION_DESIGN K4)

| step | how |
|---|---|
| screen | `plan --frac F --seed S`: seeded random order within each category, never instances.csv order |
| run | `run`: each category in TWO `ny_box.sh measure` holds (first 10 seeded rows, then the rest); every row driven directly (`<pin> vnncomp v1 ...`, stdout+stderr per row), never `ny benchmarks run` (it discards `[router]` lines) |
| VOID | no result token; the log's INSTANCE line is not this row (category, NETWORK and property: one property file is shared by many networks); the preset (`preset_base` in the sidecar) was not loaded from this arm's configs dir; GEN-R / fold without a `[router] v1` banner or without `[router] done`; `optimizer.priors` armed but no `[router] priors` line. A VOID row still records what the binary PUBLISHED (`published` column) |
| dark arm | the first 10 seeded rows all VOID: `OUTDIR/DARK`, the category stops with NO capability verdict (never a stop-loss) |
| stop-loss | after 10 seeded rows, a category's GEN-R arm stops if it decided <= 50% of what TUNED decided there |
| emergency | any sat <-> unsat contradiction of what an arm PUBLISHED -- a VOID row included -- with TUNED (same-pin judged table, else the bank) or the field (`run --truth-2026/--truth-2025`, checked in `run`, not only at `score`); TUNED against the field too: `OUTDIR/EMERGENCY`, exit 3, nothing further runs |
| statistics | PER YEAR (the twin year is reproducibility only, never pooled): exact McNemar per category with discordant rows collapsed to their MODEL cluster (row-level counts reported, not used); exact sign test across family groups (`sign_threshold`: 17 of 23, 21 of 29) |
| rule-existence LOGO | `logo`: per family G, every rule whose evidence (`rule_evidence.tsv`) is a subset of {G} is disabled in G's fold configs, run with `run --arm fold` (a fold dir may add only rule switches set to false); a family no rule depends on is reported as "GEN-R row IS the held-out row", never a hidden zero gap; the prior is REFIT without G and any planning DECISION change on G's rows (priced vs unpriced, rung fits vs not, or undecidable: a per-clause phase without a clause count) is flagged `needs_heldout_binary`; when the generic default arms the prior, `logo` REFUSES to run without `--trainset` and `--features` |
| prior / host census | every judged row records the prior's state (`read`, `host_gated`, `unavailable`, `-`) and the banner's host class; `run` / `judge` print them in their DONE line |
| sealed holdout | `seal` hashes the holdout entry list once (never overwritten), `verify-seal` refuses an edited list, `plan --holdout-entries` excludes sealed rows, `spend` appends to a ledger and refuses a second look without `--allow-respend` |

## Preconditions `run` checks before any lock wait (fail closed)

Zero `NY_*` other than `NY_BOX_*`; the pin is an executable file (its sha256 goes in `RUN_META.json`); a
GEN-R configs dir passes `verify_configs`, a TUNED dir resolves a preset for every category; every row's
files exist; an existing OUTDIR belongs to the same plan and pin (resume skips captured rows).

## Commands

```
python3 scripts/genboard/genboard.py configs --out /tmp/.../genr_cfg
python3 scripts/genboard/genboard.py plan --year 2026 --seed r7 --frac 0.25 --out plan26.tsv
python3 scripts/genboard/genboard.py run --plan plan26.tsv --configs genr_cfg --pin <pinned copy> --out gen26 --arm gen-r
python3 scripts/genboard/genboard.py run --plan plan26.tsv --configs <home>/ny/configs --pin <same pin> --out tuned26 --arm tuned
python3 scripts/genboard/genboard.py score --plan plan26.tsv --gen gen26 --tuned tuned26 --report score26.json
python3 scripts/genboard/genboard.py logo --plan plan26.tsv --base-configs genr_cfg --out logo26 \
    --features <home>/ny/scratchpad/vbs/general/instance_features.tsv
python3 scripts/genboard/genboard.py run --plan plan26.tsv --configs logo26/fold_resnet_ct --pin <same pin> --out fold26_resnet_ct --arm fold --categories cifar100_2024,tinyimagenet_2024,challenging_certified_training_2026
python3 scripts/genboard/genboard_selftest.py      # GENBOARD_SELFTEST_DONE rc=0 ran=25 passed=25 failed=0
```

Each command ends with `GENBOARD_<CMD>_DONE rc=<rc> ...` carrying its own counts.

## Gaps (honest)

- Never run on a real row. A full board is 300–450 box-hours (GC K3.6); the §9.8 screen (25%) is the first
  affordable step. GEN arms should run under a memory cap (yaml memory keys such as ml4acopf's queue cap
  and cgan's chunk governors are absent on GEN-R) or exclude cgan, and say so.
- **The cap's value decides whether the prior is read (review, MEASURED from the source).** The router's host
  class reads the cgroup limit (`router/facts.rs::host_envelope`: `mem = min(total, memory.max)`), and the
  GB10 class needs 100..140 GiB (`HostClass::classify`). Under ROUTER_PROGRAM §9.8's `MemoryMax=96G` the GB10
  classifies as `other`, so the GB10-measured prior prices NOTHING (`unpriced=<cap>:host:other`) and every
  other host-class-keyed router decision changes with it. To measure the prior on the GB10 cap at
  `MemoryMax=100G`..`140G` (100G still keeps the dense-budget clamp inert: 100/8 >= the 12 GiB ceiling) and
  confirm `prior_read=` / `hosts=gb10:` in the DONE line.
- `needs_heldout_binary` folds need a binary built with the held-out table (the table is compiled in by
  design): the harness reports them; building that binary is a manual step.
- The sealed external holdout corpus (VNN-COMP 2021–22 families absent from 2025/26) is not on disk; the
  seal / spend machinery is tested on synthetic entries only. There is NO command that runs sealed rows yet:
  `plan --holdout-entries` excludes them and `run` refuses a plan carrying them, and `spend` only appends a
  ledger line (it does not re-verify the entry list). When the corpus lands, a sealed run must be a plan built
  from `verify-seal`-ed entries, scored once, with the spend line written BEFORE the run.
- The rule-evidence table is hand-derived from ROUTER_PROGRAM §5.2 / §10.3–10.7 (§10.7: R6's four O-TRIAL
  rules, added at the ENG12 merge); a rule whose evidence is wrongly marked `structural` would never be
  disabled by its fold. The selftest derives the acting-rule list from the delivered `router.rules.*` /
  `optimizer.rules.*` / `optimizer.priors` keys in space.rs, so a rule with no row fails it.
