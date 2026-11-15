# VNN-COMP 2025 result bank

This directory holds `ny`'s locally measured verdicts on the VNN-COMP 2025
benchmarks: one CSV per category, measured at each instance's official time
limit.

`ny` was not an official VNN-COMP 2025 entrant. Everything here is local,
reproducible, internal evidence, not an official result, an organizer
certification, or a third-party attestation. `unsat` rows keep the solver's
output but no independently checked proof.

## Files

| Path | Contents |
| --- | --- |
| `<category>.csv` | One row per benchmark instance. |
| `sat_relu_sealed_20260726_e6d9211c.csv` | An immutable sealed `sat_relu` snapshot. The scorer merges it with `sat_relu.csv`. |
| `regular_evidence_index.json` | Rows bound to retained run artifacts. `scripts/ny_retroactive_scorecard.py --require-evidence` credits only these. |
| `board_refresh_2025_20260914/` | This bank scored on 2026-09-14 (`SUMMARY.md`, plus the scorer's JSON output). |
| `evidence/` | Supporting evidence for individual disputed rows. |
| `<category>_artifacts/` | Retained logs and provenance for the sweeps that re-measured that category. |
| `<category>_NOTES.md` | Internal per-category campaign notes (not part of the public snapshot). |

The bank covers all 16 regular-track categories and 7 extended-track
categories. `vit_2023` and `yolo_2023` have no rows here, so they score zero
on the extended board.

## Row format

Each row has the six official columns
`category,onnx,vnnlib,prepare_status,verdict,seconds`, optionally followed by a
seventh provenance label. Most files have no header line; the scorer skips a
`cat,onnx,vnnlib,…` header where one exists.

| Label | Meaning |
| --- | --- |
| a run tag, such as `l1audit-20260804-d1282d23` | Names the sweep that measured the row. A tag can itself mark a capped run (`cap110-NOT-a-budget-measurement-…`). |
| `inherited-unverified` | Not re-run since it was banked. Treat its verdict as a claim, not a measurement. |
| *(no seventh column)* | A legacy row that predates the provenance label. |

The label is provenance only; no scorer awards or withholds points because of
it. Every scorer drops the `test_nano` and `test_tiny` harness rows, as the
official 2025 result processor did.

## Scoring

```bash
ny benchmarks score --results reports/measured \
  --official /path/to/vnncomp2025_results \
  --year 2025 --budget-year 2025 --track regular   # or --track extended
```

The two tracks are separate leaderboards; never combine their totals. The
scorer merges every `*.csv` in each `--results` path and aborts the whole run
if two rows for one instance disagree; `reports/measured-ext` disagrees with
this bank on shared rows, so score this bank on its own.

`--budget-year` checks every recorded runtime against the official
per-instance budget: it drops solved rows that exceeded it and lists undecided
rows recorded under 90% of it. `--witnesses` decides how a `sat`
counterexample that contradicts another tool's `unsat` is treated:

- `adjudicated` (the default) uses the organizers' published adjudication
  where one exists, and otherwise lets a `sat` convict a contradicting `unsat`
  only when two or more participants claim it (a lone contested `sat` is
  reported as disputed and scores nobody);
- `assumed-valid` treats every `sat` as valid, which is the strict view for
  judging `ny`'s own soundness;
- `unvalidated` credits every claimed verdict, which reproduces the published
  2025 board.

## Measuring new rows

`scripts/measure_ny_scorecard.sh` runs `ny vnncomp v1` over the regular-track
categories (its benchmark root defaults to `benchmarks/vnncomp2026/benchmarks`
when that exists; set `NY_BROOT=benchmarks/vnncomp2025/benchmarks` for this
bank) and captures the evidence:

- before the first row, an immutable `artifacts/runs/<run_id>/start.json`
  recording the source commit and diff, the binary's hash and build features,
  the pinned `ay` revision, the toolchain, the benchmark checkout, and the host;
- for each row, the raw result file and the solver's exact output, stored
  immutably, and a seventh `run_id` column in the CSV;
- at exit, `artifacts/runs/<run_id>/completion.json`. A run without one is
  incomplete.

Start-up fails closed on a missing binary or benchmark checkout, a worktree
that changes while it is being recorded, an ambiguous `ay` pin, a missing
`NY_BUILD_FEATURES` (set it to the exact feature list of the binary under
test), a reused run ID, or any `NY_*` variable outside a reviewed allowlist.

Output goes to `reports/measured-runs/<run_id>` unless `NY_MEASURE_OUTPUT_DIR`
says otherwise. `NY_MEASURE_MAX_ROWS_PER_CATEGORY` caps rows per category for
a quick protocol check. `NY_MEASURE_CAP=<secs>` caps each row's budget at
min(official, `<secs>`) for a lower-bound sweep (the default, 0, means no
cap); read its `timeout` rows as not measured.

```bash
export PATH="$PWD/scripts:$PATH"    # puts the ny-safe-gpu-run guard on PATH
NY_BUILD_FEATURES=mip,cuda \
NY_MEASURE_OUTPUT_DIR=reports/measured-runs/20260718T120000Z \
NY_MEASURE_MAX_ROWS_PER_CATEGORY=1 \
scripts/measure_ny_scorecard.sh
```

Memory is contained by a named cgroup profile,
`NY_MEASURE_CONTAINMENT_PROFILE=gb10-80g` (the default) or `wsl24-20g`,
recorded in the start manifest. The sweep refuses to run unless
`ny-build.slice` carries that profile's limits exactly (units in
`scripts/systemd/`). A smaller profile gives a lower bound, not a comparable
result.

## Replaying counterexamples

A captured `sat` row starts as `counterexample_validation.status =
not_checked`. To validate it against the official 2025 checker at zero
tolerance:

```bash
python3 scripts/replay_vnncomp2025_counterexample.py \
  --artifact-root /path/to/artifacts \
  --metadata /path/to/artifacts/<category>/<instance>/<run_id>.json \
  --benchmark-root /path/to/vnncomp2025_benchmarks/benchmarks \
  --official-results /path/to/vnncomp2025_results
```

The replay is fail-closed. It needs the pinned checker runtime the script
names (`--runtime-root`), Python 3.11.15, and `onnxruntime==1.16.3` on
`CPUExecutionProvider`. It writes a separate
`*.vnncomp2025-zero-tol-validation.json` sidecar and never modifies the result
or its metadata. (`scripts/replay_ny_counterexamples.py` checks against the
2026 checker; its sidecars are diagnostic only for this bank.)

## Large models installed by `setup.sh`

Two model files in the 2025 benchmarks come from the benchmark's `setup.sh`
rather than Git: the cGAN small-transformer model and `vgg16-7.onnx`. Install
them from the retained, hash-pinned snapshot with:

```bash
python3 scripts/materialize_vnncomp2025_large_models.py \
  --benchmark-root /path/to/vnncomp2025_benchmarks/benchmarks
# Review the plan, then repeat with --apply.
```

Until they are installed, the rows that use them cannot be measured.

## Promoting rows into the bank

Measurement and replay do not change the checked-in bank. Promote one sealed
row with:

```bash
python3 scripts/promote_regular_bank.py \
  --benchmark-root /path/to/vnncomp2025_benchmarks/benchmarks \
  --official-results /path/to/vnncomp2025_results \
  --measured-dir reports/measured \
  --artifact-root /path/to/run/artifacts \
  --run-id <run-id> \
  --category <category> \
  --instance-index <1-based-official-row> \
  --exact-commit <40-hex-source-commit>
# Review the dry run, then repeat with --apply.
```

Promotion re-validates the source archive, the run's inputs and outputs, the
official occurrence and budget, the published truth, and (for `sat`) the
checker replay. It writes `regular_evidence_index.json` before touching any
CSV, so an interrupted promotion can be detected and resumed. Replacing a
legacy row that already holds `sat` or `unsat` needs
`--migrate-legacy-decided-row`, and is accepted only when the sealed verdict
matches it exactly.

For many rows, `scripts/promote_regular_bank_batch.py --requests FILE` does
the same in one validated transaction. `scripts/regular_bank_evidence.py`
re-checks the whole index without changing anything:

```bash
python3 scripts/regular_bank_evidence.py \
  --benchmark-root /path/to/vnncomp2025_benchmarks/benchmarks \
  --official-results /path/to/vnncomp2025_results \
  --measured-dir reports/measured
```

Missing, dangling, or mismatched evidence fails the check. The external
artifact directories and source archives are part of the evidence record and
must stay at their indexed paths.
