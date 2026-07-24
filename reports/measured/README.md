# Measurement audit status

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
scripts/replay_ny_counterexamples.py \
  --artifact-root /absolute/path/to/artifacts \
  --metadata /absolute/path/to/artifacts/<category>/<instance>/<run_id>.json
```

The replay is fail-closed. It accepts only SAT metadata, verifies the immutable
metadata, raw result, run-start manifest, ONNX, and VNN-LIB hashes before
execution, invokes `external_tools/vnncomp2026_results` at pinned revision
`b0ae7110` in `~/.venvs/vnncomp-ce-2026`, requires every dependency version in
the official `SCORING/requirements.txt` (including VNNLIB-Python 1.0.2), and
uses `CPUExecutionProvider` only. It verifies the evidence and checker source a
second time after replay, then creates a separate
`*.counterexample-validation.json` sidecar with `O_EXCL`. It never changes the
raw result or its metadata. The sidecar binds the strict/tolerance/invalid
classification, rationale, checker commit and source hashes, installed runtime,
and every input/result/manifest hash.

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

The default output remains `reports/measured` for backward compatibility and
therefore resume-skips matching legacy rows. Set `NY_MEASURE_OUTPUT_DIR` to a
new directory when producing an evidence-complete rerun; its resolved path is
recorded in the start manifest and its resume decisions are isolated from the
legacy CSVs. Broad targets such as the filesystem root, the NY repository root
or an ancestor, and `.git` are rejected before measurement.

For a provenance-complete one-row-per-category protocol smoke, use an isolated
bank and the optional positive row cap:

```bash
NY_BUILD_FEATURES=mip,cuda \
NY_MEASURE_OUTPUT_DIR=reports/measured-runs/20260718T120000Z \
NY_MEASURE_MAX_ROWS_PER_CATEGORY=1 \
scripts/measure_ny_scorecard.sh
```

Omit `NY_MEASURE_MAX_ROWS_PER_CATEGORY` for an unlimited sweep. The effective
limit (or `null` for unlimited) and output root are both manifest-bound.
