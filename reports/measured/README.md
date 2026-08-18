# Measurement audit status

All promoted evidence and score outputs described here have claim scope
`local_reproducible_internal_counterfactual_not_official_or_independently_attested`.
They are internal, locally reproducible evidence—not organizer certification,
an official result, a cryptographic signature, or third-party attestation.
UNSAT rows retain solver output but no independently checked proof; exact SAT
replay is deliberately host/path-bound to its retained runtime.

## Latest checked-in aggregate snapshot

The most recent aggregate ledger checked into this repository is
[`configs/preset_score_at_risk.yaml`](../../configs/preset_score_at_risk.yaml),
measured 2026-08-07 at `5382ddb8`: regular-16 **1394.5**, extended-9 **477.6**,
and full-field-25 **1872.1**. Against the exact scorer-derived 2025 regular
winner, 1566.894737 ([pinned official category table](https://github.com/VNN-COMP/vnncomp2025_results/blob/ea89fbc2518b6729f17c96eeec22c56c88e496a9/SCORING-ZERO-TOL/latex/scored.tex)),
that snapshot is approximately **172.4 points behind**.

This is a ledger snapshot, not an exact live score. Later bank mutations do not
automatically update it; rerun the scorer and refresh its provenance before
calling a new aggregate current. Dated measurements below remain historical.

## READ FIRST — a row here is RE-VERIFIED or it is INHERITED

A ledger row is only as good as the last time somebody ran it. Three
fail-closed soundness gates landed after most of these rows were banked, and
nothing re-checked them for months; two banked categories zeroed unnoticed.
The rows in this bank are therefore not uniform:

* **RE-VERIFIED at HEAD** — replayed through `ny vnncomp v1` with the
  category's own preset, at the official per-instance budget, on a `--release
  --features cuda,mip` build, one instance at a time, with the log confirmed to
  carry `Loading preset:`. The complete per-row record (banked verdict, observed
  verdict, return code, wall, 1-minute load average, and the commit it was
  measured at) is in
  `reports/measured-runs/lane4-honest-correction-2026-08-03/corrections.csv`.
* **INHERITED** — nobody has re-run it since it was banked. An inherited
  `sat`/`unsat` is a CLAIM, not a measurement. Do not quote it as current
  capability without a re-run.

`reports/measured-runs/lane4-honest-correction-2026-08-03/README.md` carries the
per-category counts, the reason the correction covers some categories and not
others, and the resulting score.

### The distinction is IN THE CSV, not only in this file

Until 2026-08-03 that distinction lived only here and in a sidecar. That is the
wrong place for it for two reasons. The first is that a reader forms a belief
about a row while looking at the ROW, and nothing at that point said whether
anybody had ever re-run it. The second is worse: **the sidecars are not in
git.** `.gitignore` un-ignores only `reports/measured/*.csv`,
`reports/measured/README.md`, `reports/measured-ext/` and
`reports/measured-2026/*.csv` — everything under `reports/measured-runs/` is
ignored. So the `corrections.csv` this file cites as the per-row provenance
record exists on one machine's disk and in no clone. Provenance that a fresh
checkout cannot see is not provenance.

Both scorers already accept the seven-column `measured-7` schema
(`category,onnx,vnnlib,prepare_status,verdict,seconds,run_id`) and treat the
seventh field as a free-form provenance label, so the label now travels with the
row:

| seventh column | meaning |
|---|---|
| `laneB-20260803-db3fb2c7` | re-measured at `db3fb2c7`, official budget, serial, preset-loaded |
| `lane4-20260803-3ab7022e` | re-measured by the honest correction at `3ab7022e` |
| `inherited-unverified` | **nobody has re-run this row since it was banked** |

An `inherited-unverified` `sat`/`unsat` is a CLAIM. Treat it as capability only
after a re-run. The tag is provenance only — neither
`scripts/ny_retroactive_scorecard.py` nor `ny benchmarks score` gives or
withholds a single point on the strength of it, so tagging cannot flatter the
score.

## Systemic audit, 2026-08-04 (`l1audit-20260804-d1282d23`)

At `d1282d23`, **2081 of 2420 banked decided rows (86%) carried no
re-verification tag at all**, and those rows carried **1598.8 of the 1871.5
FULL-FIELD normalized points**. The systemic audit re-ran 1940 of them through
`ny vnncomp v1` with the category preset, at the exact official per-instance
budget, serially, on a `--release --features cuda,mip` build at `d1282d23`
(binary outside the tree, `--configs-dir <worktree>/configs`, every log
confirmed to carry `Loading preset:`).

**Result: 1907/1940 reproduced first time. 31 rows are confirmed dead. 3 rows
that failed the first re-run turned out to be alive.** FULL-FIELD 1871.5 ->
**1847.9** (REGULAR-16 1392.3 -> 1370.3, EXTENDED-9 479.1 -> 477.6), 2197
modeled matches, ZERO contradictions, extended moat still symmetric.

| category | decided banked | reproduced | dead | normalized before -> after |
|---|---|---|---|---|
| acasxu_2023 | 183 | 182 | 1 | 98.4 -> 97.8 |
| cctsdb_yolo_2023 | 39 | 39 | 0 | 100.0 |
| cersyve | 12 | 12 | 0 | 100.0 |
| cgan_2023 | 12 | 12 | 0 | 57.1 |
| collins_aerospace_benchmark | 6 | 6 | 0 | 100.0 |
| collins_rul_cnn_2022 | 62 | 54 | 8 | 100.0 -> 87.1 |
| cora_2024 | 151 | 148 | 3 | 98.7 -> 96.7 |
| lsnc_relu | 11 | 11 | 0 | 13.8 |
| malbeware | 150 | 142 | 8 | 100.0 -> 94.7 |
| ml4acopf_2024 | 44 | 43 | 1 | 69.8 -> 68.3 |
| nn4sys | 162 | 161 | 1 | 83.5 -> 83.0 |
| relusplitter | 34 | 34 | 0 | 19.9 |
| safenlp_2024 | 976 | 968 | 8 | 90.4 -> 89.6 |
| soundnessbench | 50 | 50 | 0 | 100.0 |
| traffic_signs_recognition_2023 | 39 | 39 | 0 | 90.7 |
| vit_2023 | 9 | 9 | 0 | 7.2 |

### A SINGLE FAILED RE-RUN IS NOT A DEAD ROW

Three rows failed the sweep and then reproduced on repeat, at a quieter box:

| row | sweep | repeats |
|---|---|---|
| `cersyve point_mass_finetune_inv / prop_point_mass` | timeout @ 97.4 s | **unsat @ 56.5 s, 2 of 3** |
| `soundnessbench model.onnx / model_19` | timeout @ 143.4 s | **sat @ 15.4 s, 3 of 3** |
| `nn4sys mscn_128d_dual / cardinality_1_10450_128_dual` | unknown @ 187.7 s | **unsat @ 156.4 s, 3 of 3** |

`soundnessbench model_19` is a row of the MANDATED 50/50-sat gate. Had the
single-shot sweep been banked, this audit would have reported that gate broken
when it is intact. **Every downgrade in this bank therefore required the row to
fail the sweep AND all three repeats**; the rule is implemented, not advisory.
The complete repeat record (per attempt: verdict, return code, wall, 1-minute
load average) is committed at
`reports/measured-ext/evidence/lane1-audit-20260804/repeats.csv`, and the full
1940-row sweep record at `.../per-row.csv`.

### Cause of every confirmed death

| cause | rows | shape |
|---|---|---|
| `BOUND-REGRESSION(budget-wall)` | 23 | runs to the wire and yields nothing |
| `BOUND-REGRESSION(early-give-up)` | 8 | **stops at ~16 s of a 20 s budget** (all `safenlp_2024 perturbations_0`) |
| load failure / crash / backend substitution | 0 | no row died from any of these |

No verdict FLIPPED: zero `sat` <-> `unsat` disagreements in 1940 rows.

Two of the dead blocks are per-model, not scattered, which is what makes them a
repair list rather than noise:

* `collins_rul_cnn_2022` — all 8 deaths are `NN_rul_full_window_20/40`, banked
  at 41-197 s and now dead against 300/600/1200 s budgets.
* `malbeware` — all 8 deaths are `malware_malimg_family_scaled_{4,16}-25`,
  banked at **0-1 s** and now dead against a 100 s budget.

The 8 `safenlp_2024` deaths are a different defect: the verifier ABANDONS at
~16 s of its 20 s budget on rows it used to close at 15-17 s, so a fifth of the
budget is left unused.

257 of the 1940 re-runs ran on a SUBSTITUTED backend (preset asks for `wgpu`,
the binary runs `cpu` under the sticky overflow-sentinel quarantine); 245 of
them reproduced anyway and 11 of the 31 confirmed deaths are among the rest.
The substitution is not by itself fatal, but no
`cora_2024`, `malbeware` or `soundnessbench` row is currently a measurement of
the configuration its preset declares.

### What this audit did NOT cover

Never re-run, still `inherited-unverified`, still scored:

| category | decided rows unmeasured | normalized points resting on them |
|---|---|---|
| `dist_shift_2023` | 72 | 100.0 |
| `linearizenn_2024` | 59 | 98.3 |
| `tllverifybench_2023` | 32 | 100.0 |
| `vggnet16_2022` (ext) | 14 | 77.8 |

**177 decided rows / ~376 normalized points remain unmeasured.** They were
de-prioritised deliberately: ranked by exposure per row-second (points at risk
divided by the optimistic re-run cost) they are the four worst buys on the
board — `linearizenn_2024` alone costs 6.7 h of wall for 98.3 points, against
`cersyve` at 46 s for 100. Do not read this bank's 2032 re-verified rows as
"the bank is clean".

### The extended bank CANNOT record provenance at all

`reports/measured-ext/*.csv` is `extended_bank_v1`
(`track,onnx,vnnlib,verdict,seconds`) — five columns, no `run_id` slot.
`scripts/extended_bank/validate_bank.py:parse_source_row` reads a 6-column row
as `measured_v1` and then rejects it because column 3 holds a verdict, not a
prepared flag. So **195 extended decided rows carrying 477.6 normalized points
have nowhere to say whether anybody has ever re-run them**, and this file's
earlier claim that `traffic_signs_recognition_2023` carries the tag is FALSE —
that ledger is five-column and untagged like the rest. The disambiguation is
available whenever someone wants it (`VERDICTS` and `PREPARED_TOKENS` are
disjoint, so a 6-column `extended_bank_v2` is unambiguous); until then, extended
provenance lives only in
`reports/measured-ext/evidence/lane1-audit-20260804/`.

Categories carrying the seventh column today: every regular category this audit
touched, plus `tinyimagenet_2024`, `cifar100_2024` and `metaroom_2023`.
Untagged six-column ledgers (`dist_shift_2023`, `linearizenn_2024`,
`tllverifybench_2023`) are exactly the ones no sweep has reached yet; adding the
column as they are re-measured is the intended direction of travel.

Two bounds follow from the shape of the re-measurement and both matter:

* the re-measured set is an **upper bound** — only rows already banked
  `sat`/`unsat` were re-run, and not even all of those, so inherited rows may
  also have died; and
* it is a **lower bound** — no row that HEAD might NEWLY solve was ever tried.

That is exactly why the correction never wholesale-replaces a ledger with a
partial sweep: doing so would invent verdicts for rows nobody measured.

## THERE ARE TWO BANKS — score both

Results live in **two** directories, and neither is a superset of the other:

* `reports/measured/` — this directory, primarily the regular-16 categories.
* `reports/measured-ext/` — the extended-track categories, plus per-row
  `evidence/*.validation.json`. It is the ONLY bank for
  `traffic_signs_recognition_2023`, `vggnet16_2022` and `vit_2023`.

Scoring only this directory reports those three as ZERO and under-counted the
extended track by **182 normalized points** (303.4 instead of 485.6). Always pass
both, and pass the budget years so recorded runtimes are checked against the
official per-instance timeouts:

```sh
ny benchmarks score --results reports/measured --results reports/measured-ext \
  --official /tmp/vnncomp2025_results --budget-year 2025 --budget-year 2026 \
  --witnesses unvalidated --track extended
```

`ny benchmarks score` now names any scored category that has zero measured rows,
because a missing bank is indistinguishable from a real zero in the totals. It
also refuses to merge two banks that disagree on the same instance rather than
silently picking one — that refusal is what surfaced two stale rows here
(`lsnc_relu quadrotor2d_state_34`, `relusplitter mnist-net_256x6 prop_9_0.03`),
in both cases with `reports/measured/` behind the extended bank.

Pick the witness model deliberately: `--witnesses unvalidated` reproduces the
PUBLISHED 2025 board (validated — it returns alpha-beta-CROWN 1566.9 and PyRAT
1228.4) and is the right model for "where would NY have placed";
`--witnesses assumed-valid` (the default) charges -150 for an UNSAT contradicted
by a SAT witness and is the right model for judging NY's own soundness. The two
differ by 18 points on the current bank.

The checked-in CSVs that predate the provenance protocol are legacy raw-verdict
snapshots. They remain useful for regression triage and headroom estimates, but
they are **not auditable as an official VNN-COMP score**. In particular, their
old six-column rows have no run ID and their SAT witnesses were not retained.

New rows produced by `scripts/measure_ny_scorecard.sh` automate the local
evidence-capture portion of the protocol:

- Before any row is appended, an immutable
  `artifacts/runs/<run_id>/start.json` records the UTC start, NY commit and Git
  status, a digest of the complete tracked dirty diff, hashes of every
  non-ignored untracked file, the solver-binary SHA-256, declared build
  features (required through `NY_BUILD_FEATURES`), pinned AY revision, pinned
  Rust toolchain and observed `rustc` identity.
- The same record binds the benchmark repository path, sanitized remote,
  commit and status; exact category order, timeout cap, command template and
  output paths; a fixed non-secret environment allowlist; and the OS, kernel,
  CPU, memory, GPU/driver and resource-limit identity.
- Every new CSV row has a seventh `run_id` column. The six official positional
  fields are unchanged, so legacy rows and existing scorers remain compatible.
- Every non-missing row is appended only after its complete raw result file is
  archived immutably together with the solver's exact combined stdout/stderr
  bytes. Its JSON metadata records both artifact SHA-256 values, exact ONNX and
  VNN-LIB SHA-256/size, and linked start-manifest SHA-256. These per-row input
  hashes bind post-setup assets that may be intentionally ignored by the
  benchmark Git repository. A missing-input summary row cannot have those
  artifacts and remains the explicit exception.
- Large models reused by many properties are read and hashed only once per run.
  A locked, atomically replaced run-local cache keys hashes by resolved path,
  device/inode, size, nanosecond mtime and ctime; any fingerprint change forces
  a rehash and retains the earlier version as separate evidence. The immutable
  completion record binds the final cache SHA-256 and entry count.
- SAT capture is stricter: the raw result must include a counterexample
  assignment, and its validation status starts at `not_checked`. Non-SAT rows
  use the explicit `not_applicable` counterexample-validation status.
- The shell EXIT trap makes a best-effort immutable
  `artifacts/runs/<run_id>/completion.json` with the UTC end and shell exit
  status, including ordinary interruption statuses. SIGKILL or machine loss
  cannot run a userspace trap; a missing completion record therefore marks an
  incomplete run rather than silently implying success.

Start capture is fail-closed. It rejects a missing binary or benchmark
checkout, an unstable worktree, an ambiguous AY pin, a missing build-feature
declaration, a reused run ID, and any set `NY_*` variable not in the reviewed
fixed allowlist. Environment values
outside that allowlist (including tokens and passwords) are never copied into
the manifest. Set `NY_BUILD_FEATURES` to the exact comma-separated Cargo feature
list used for the measured binary; capture refuses to start without it. The
binary hash remains the authoritative artifact identity.

Capture alone does **not** turn a SAT artifact into an organizer-accepted
counterexample. Every new SAT metadata file initially says
`counterexample_validation.status = not_checked`. Validate one retained SAT
artifact with:

```bash
python3 scripts/replay_vnncomp2025_counterexample.py \
  --artifact-root /absolute/path/to/artifacts \
  --metadata /absolute/path/to/artifacts/<category>/<instance>/<run_id>.json \
  --benchmark-root /absolute/path/to/vnncomp2025_benchmarks/benchmarks \
  --official-results /absolute/path/to/vnncomp2025_results
```

The exact replay is fail-closed. It requires the retained
`<home>/ny-vnncomp2025-checker-exact-20260731T074000Z` runtime (or that
same exact path passed explicitly), the pinned 2025 results and benchmark Git
identities, the exact official requirements, Python 3.11.15,
`onnxruntime==1.16.3`, and `CPUExecutionProvider`. Its isolated retained worker
runs with `-I -S -B`, and the evidence binds the checker source, worker/harness,
import-tree manifests, native-library closure, settings
(`IGNORE_CE_Y=False`, `ATOL=1e-4`, `RTOL=1e-3`), authoritative input source,
raw assignment, metadata, result, and start manifest. Sidecar schema v2 uses a
strict source union: an input binds either its official Git path/blob or its
complete retained official-setup payload manifest, never null or mislabelled
Git fields. The pinned runner SHA-256 is
`c8d20b67304d0bc52e74ae0a0d279ed10198107379aad670681891b774d372d2`;
the worker SHA-256 is
`001d1ac6af69e61fa108fe60d4589a54a850ad6bf1b7bd72ef5a428bc1410c63`.
Replay verifies those identities again after execution and creates a separate
`*.vnncomp2025-zero-tol-validation.json` sidecar with `O_EXCL`; it never changes
the raw result or metadata.

The older `scripts/replay_ny_counterexamples.py` checks against a 2026 checker
and writes `*.counterexample-validation.json`. Those sidecars remain useful
diagnostics but are explicitly non-creditable for the 2025 ZERO-TOL bank.

The only extraction is removal of the standalone leading `sat` protocol line;
all remaining assignment bytes reach the official checker unchanged. This is
important for VNN-LIB 2.0: pre-fix NY relational outputs use legacy
`(X_f[i] value)` s-expressions, which the official section-5.3 parser correctly
classifies as `malformed_ce`. NY now serializes declaration-ordered tensor
headers and scalar values; the patched dual-network format replays as a strict
official monotonic-ACAS witness. Replay does not translate old bytes, so any
affected relational measurements still require a rerun.

Before publishing an official-style ZERO-TOL or SMALL-TOL rank, replay every
retained witness and consume the immutable classifications in scorer input. A
zero count of raw-verdict contradictions is not a soundness certificate.

The old files may also contain `test_nano`/`test_tiny` harness rows. All NY
score utilities exclude those rows because the official 2025 result processor
excluded them. `scripts/ny_retroactive_scorecard.py` therefore continues to
label unvalidated input a **raw-verdict counterfactual**, not an official
scoreboard.

## Starting a clean auditable bank

By default, each invocation chooses a run ID and writes to the isolated
`reports/measured-runs/<run_id>` directory. `reports/measured` is a legacy
tracked bank, not the default. Set `NY_MEASURE_RUN_ID` for a deliberate ID or
`NY_MEASURE_OUTPUT_DIR` for a different bank; the resolved output path is
recorded in the start manifest. A custom bank can resume-skip matching rows
already present there, occurrence by occurrence. Broad targets such as the
filesystem root, the NY repository root or an ancestor, and `.git` are rejected
before measurement.

The named containment profile controls charged memory, not virtual address
reservations. `gb10-80g` retains `memory.high=64 GiB` and `memory.max=80 GiB`;
`wsl24-20g` retains `16 GiB` and `20 GiB`. Both use the same finite soft/hard
`RLIMIT_AS=160 GiB` (`ulimit -v 167772160` KiB). CUDA/ONNX Runtime reserves
about 53.5 GiB before useful work, and a sealed CIFAR100 idx7641 run reached
79.67 GiB of VA while its cgroup peak was only 24.36 GiB. The former 80 GiB
address-space cap was therefore an allocation-path cliff, not a physical-memory
backstop. At 160 GiB the measured run has 80.33 GiB of VA headroom while the
cgroup has 55.64 GiB of charged-memory headroom, so the cgroup is authoritative.
The finite limit remains defense in depth against runaway per-process mappings.
Every manifest records the exact soft and hard values; historical 80 GiB
manifests remain historical facts and must not be rewritten.

For a provenance-complete one-row-per-category protocol smoke, use an isolated
bank and the optional positive row cap. From the repository root, put
`scripts/` on `PATH` so the required `ny-safe-gpu-run` containment guard is
discoverable:

```bash
export PATH="$PWD/scripts:$PATH"
NY_BUILD_FEATURES=mip,cuda \
NY_MEASURE_OUTPUT_DIR=reports/measured-runs/20260718T120000Z \
NY_MEASURE_MAX_ROWS_PER_CATEGORY=1 \
scripts/measure_ny_scorecard.sh
```

Omit `NY_MEASURE_MAX_ROWS_PER_CATEGORY` for an unlimited sweep. The effective
limit (or `null` for unlimited) and output root are both manifest-bound.

## Retained official-setup large models

The pinned benchmark Git commit declares 20 model uses whose payload bytes are
installed by `setup.sh` rather than stored as Git blobs: cGAN rows 20–21 use
`benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx`, and all
18 vggnet16 rows use
`benchmarks/vggnet16_2022/onnx/vgg16-7.onnx`. These are the only two logical
paths eligible for the offline fallback. Any Git candidate takes precedence,
multiple Git candidates remain an error, and a missing non-allowlisted path
fails closed.

The payload snapshot is fixed at
`<home>/ny-vnncomp2025-large-models-exact-20260731T083000Z`.
Its mode-0444 `manifest.json` has SHA-256
`f7243cb9fa4dbacee49d439233563cfa08da194b7775af4dfd6966390d7170aa`
and binds the official benchmark commit, `setup.sh` Git blob and SHA-256,
Sciebo share `RapAoed1dxG1PMs`, selected seed `896832480`, source-relative
paths, and compressed/decompressed hashes and sizes. Validation accepts the
fallback only when the pinned Git commit contains zero candidates; it checks
the exact no-link mode-0555/0444 inventory, strict single-member gzip bytes,
payload bytes, and a final source rehash. The remote share is a mutable
distribution mechanism, not an independent attestation; the retained
host-bound hashes are the authority for this local counterfactual.

Evidence validation never changes the benchmark worktree. Materialize the two
payloads with the separate dry-run-by-default command:

```bash
python3 scripts/materialize_vnncomp2025_large_models.py \
  --benchmark-root /absolute/path/to/vnncomp2025_benchmarks/benchmarks
# Review the two-file/symlink plan, then repeat with --apply.
```

Apply uses atomic no-overwrite file creation, verifies mode 0444 and exact
payload hashes, and creates only the official setup-script vgg symlink
`vggnet16_2022/onnx/vgg16-7.onnx` →
`../../vggnet16_2023/onnx/vgg16-7.onnx`. Existing different files or links
are never replaced. Until this step is applied, the cGAN 20–21 and vgg 1–18
worktree rows remain deliberately unmeasurable even though their retained
authority is available.

## Promoting a completed regular-track row

Measurement capture and SAT replay are necessary but do not themselves modify
the checked-in regular bank. Use the dry-run-by-default promoter for one exact
official occurrence:

```bash
python3 scripts/promote_regular_bank.py \
  --benchmark-root /absolute/path/to/vnncomp2025_benchmarks/benchmarks \
  --official-results /absolute/path/to/vnncomp2025_results \
  --measured-dir reports/measured \
  --artifact-root /absolute/path/to/isolated-run/artifacts \
  --run-id <sealed-run-id> \
  --category <category> \
  --instance-index <positive-one-based-official-row> \
  --exact-commit <40-hex-clean-source-commit>
```

Review the reported transaction, then repeat with `--apply`. Promotion
revalidates the clean source identity and its retained adjacent
`<source-repository>.full-source.tar.gz` Git archive member-for-member, along
with the immutable completion, exact input and result bytes, unique official
occurrence, timeout budget, published truth, and the pinned official-checker
replay for SAT. By default it replaces only a unique unresolved bank row. The
canonical `reports/measured/regular_evidence_index.json` is written first so an
interrupted transaction is detectable and exactly resumable; later promotions
do not invalidate earlier row bindings. Entry schema v4 records an exact
authoritative-source union for every ONNX/VNN-LIB input (`git_blob` or
`official_setup_retained_payload`) and binds the complete retained manifest
when that fallback is used. Entry schema v5 is reserved for a strictly correct
SAT witness that changes pinned published truth from `holds` to `violated`. It
additionally binds and recomputes the pinned organizer repository, every scored
participant row and counterexample classification, old/new instance points,
penalties, category totals, the entrant-aware category denominator, and the
official overall leader. Indeterminate truth and incomplete or inconsistent
rescoring fail closed. Legacy v1–v4 objects are reopened under their original
schemas and remain unchanged when a validated v5 transaction is added.

An unindexed legacy row that already contains `sat` or `unsat` requires the
explicit `--migrate-legacy-decided-row` flag. This mode accepts only a
canonical prior verdict that exactly equals the independently reopened sealed
verdict. It preserves the complete prior CSV row and its hash in the indexed
transaction; unresolved rows, `correct`/`incorrect` aggregate markers,
conflicting verdicts, and rows indexed under another mode fail closed.

For many rows, use the batch path rather than invoking the one-row promoter in
a loop. The batch validates the existing index once, shares one pinned
checker/runtime snapshot, replays and validates every candidate, performs one
final global revalidation, then commits one index-first transaction across all
affected CSVs:

```bash
python3 scripts/promote_regular_bank_batch.py \
  --requests /absolute/path/to/batch-request.json
# Review the dry-run JSON, then repeat with --apply.
```

The current request file has schema
`ny_regular_bank_promotion_batch_request_v2` and a nonempty `requests` array.
Each item contains the same named arguments as the one-row command plus
`"evidence_index": null` (or an explicit common path) and the required boolean
`"migrate_legacy_decided_row"`. Schema v1 remains accepted as a strictly
non-migrating request for compatibility. Duplicate/conflicting occurrences and
omitted pre-existing dangling transactions fail closed. A process death after
the durable index write leaves only indexed dangling rows; repeating the exact
batch completes the remaining CSV renames.

The shared validator is read-only:

```bash
python3 scripts/regular_bank_evidence.py \
  --benchmark-root /absolute/path/to/vnncomp2025_benchmarks/benchmarks \
  --official-results /absolute/path/to/vnncomp2025_results \
  --measured-dir reports/measured
```

For an evidence-qualified projection, pass the same pinned roots:

```bash
python3 scripts/ny_retroactive_scorecard.py \
  --official /absolute/path/to/vnncomp2025_results \
  --benchmark-root /absolute/path/to/vnncomp2025_benchmarks/benchmarks \
  --measured reports/measured \
  --official-artifacts \
  --ny-sat-status correct-up-to-tolerance \
  --require-evidence
```

Only applied, fully revalidated index entries receive regular-track credit.
Missing, dangling, corrupt, or mismatched evidence fails nonzero, and unindexed
legacy rows deliberately score zero. The large external artifact directories
and full-source archives are part of the evidence record and must be retained
at their indexed absolute paths.
