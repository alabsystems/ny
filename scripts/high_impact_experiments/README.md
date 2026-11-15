# High-impact experiment runner (E2)

This is experimental orchestration, with **no certificate constructor and no
official scoring authority**. It freezes a cohort and four named route arms,
runs exact argv through the measurement box, archives costs and method receipts,
then invokes a separate named replayer for each decided claim. Missing, void,
refused, unsupported, and zero-coverage replay cannot qualify a decision.

## Run

```sh
python3 -B scripts/high_impact_experiments.py check /absolute/path/plan.json
python3 -B scripts/high_impact_experiments.py run /absolute/path/plan.json \
  --archive /external/existing-parent/new-exclusive-archive --mode measure
```

`check` hashes inputs and sources; it executes no solver. `measure` runs queries
through the fixed repository `scripts/ny_box.sh measure`. `correctness` uses
`ny_box.sh build` and labels every receipt as a correctness probe, so its costs
cannot be mistaken for fair measurements. Binary identity probes and independent
replays use the build lane. Wrapper timeout includes the box wait allowance;
solver timeout starts immediately before subprocess startup and includes the
entire query. Signal cleanup may add a recorded 0.4-second grace. Arms must not
daemonize or escape their process group.

All arms run serially. Official CSV source order is retained, including repeated
identical rows. `source_indices` is a selection of distinct **one-based occurrence
indices**; the input list never changes source order. A repeated CSV row remains
a separate run. Multi-network rows are retained as explicit unsupported refusals.
Synthetic controls require `cohort.kind: synthetic` and stay labelled separately.
The caller supplies the official CSV; its exact bytes, suite, category, source
index, occurrence ordinal, model/property bytes, and timeout bind every run.

Archives must be outside the source worktree. A new directory is created
exclusively. Runner-owned JSON, logs, sealed artifacts and replay requests use
`O_EXCL`; an existing path refuses. Solver-owned result/hint files are untrusted
outputs, sealed into separate immutable files after execution. Frozen source,
input, binary, profile, replayer and raw artifact identities are checked after
each run and at experiment completion. Changes void the relevant run and/or
the entire summary; the earlier immutable receipts remain available for audit.
File hashing and sealing stream in 1-MiB chunks, including large checkpoints.

## Plan contract

The plan is JSON, schema `ny_high_impact_plan_v1`. Required fields:

| Field | Meaning |
| --- | --- |
| `cohort` | `kind`, `suite`, `category`, `instances_csv`, `input_root`; optional `source_indices` |
| `arms` | Exactly four objects with unique safe `name` and roles `baseline`, `mechanism`, `comparator`, `combined` |
| arm `binary`, `argv`, `program` | Executable path, string array starting with `{binary}`, `ny` or `external` |
| arm `source_paths` | Explicit source/dependency files or directories; membership and bytes are frozen |
| arm `environment` | Explicit string map; no inherited NY knobs or credentials; no box bypass/code injection |
| arm `profile` | Optional explicitly passed profile file; dependencies belong in `source_paths` |
| arm `configs_root` | NY scored preset root, explicitly passed as `--configs-dir {configs_root}` |
| arm `build_features` | Required for NY, e.g. `[]` or `["mip", "cuda"]`; checked against the actual binary |
| arm `build_receipt` | Required for external programs; pinned version/source/binary identity (below) |
| arm `replay` | Optional independent replay adapter; absence leaves all decided claims unqualified |
| `conditions` | Subset of `process_cold`, `artifact_cache_warm`; default `process_cold` |
| `repetitions` | 1–100; default 1; warm condition additionally records repetition 0 as warmup |
| `box_wait_allowance_seconds` | Separate wrapper admission/wait allowance; default 6300 |
| `sample_seconds` | At least 0.02; default 0.1 |

Paths are relative to the plan directory unless absolute. The runner freezes its
own source, Python binary, wrapper and provenance/replay helpers. For NY it also
freezes the complete `crates/`, Cargo manifests/lock and toolchain pin, plus the
selected preset root. Existing `ny --build-id` is replayed through the box:
executable SHA-256, compiled features, commit, tracked source dirty digest and
the existing build-coherence check must agree. Untracked compiled sources refuse
until a source snapshot can be bound by the existing build stamp. External build
receipts are operator provenance, never mathematical proof.

For example, a NY arm can use:

```json
{
  "name": "mechanism",
  "role": "mechanism",
  "program": "ny",
  "binary": "/sealed/build/ny",
  "build_features": [],
  "configs_root": "/frozen/profiles/mechanism/configs",
  "source_paths": ["/frozen/runtime-dependency-manifests"],
  "environment": {},
  "argv": ["{binary}", "vnncomp", "v1", "{category}", "{model}",
           "{property}", "{result}", "{timeout}", "--configs-dir", "{configs_root}",
           "--proof-profile", "proof"],
  "replay": {
    "adapter": "official_sat",
    "checker_repo": "/pinned/vnncomp2026_results",
    "checker_python": "/pinned/venv/bin/python",
    "checker_venv": "/pinned/venv",
    "vnnlib_source": "/pinned/VNNLIB-Python",
    "timeout_seconds": 600
  }
}
```

Declare other arms explicitly; the runner does not invent comparison settings.
Allowed argv placeholders are `binary`, `model`, `property`, `timeout`, `result`,
`phases`, `run_dir`, `cache_dir`, `profile`, `configs_root`, `instance_index`,
`category`, `suite`. Replacement creates literal argv elements; no shell, eval,
glob, command substitution, or quoting interpretation is used. An explicit
profile must appear as a `{profile}` token. Scored NY arms must explicitly pass
their frozen config root instead of relying on nearest-directory inference.

An external identity receipt is schema `ny_external_program_identity_v1`, with
`version` (nonempty string), `binary_sha256`, and `source_files_sha256`. The last
field is `digest(encode(pin_sources(source_paths)))` from the package's evidence
API. Use the same exact source roots and host snapshot in the arm. Executable and
sources are independently rehashed; this receipt does not prove how they were built.

## Replay contracts

### Official SAT

`official_sat` reuses `scripts/replay_ny_counterexamples.py` and its existing
`ny_measurement_start_v1`, `ny_measurement_result_v2`, and
`ny_counterexample_validation_v1` contracts. The official checker revision,
source inventory, VNNLIB revision/submodules, Python and witness bytes are bound.
Only `official_result: correct` with a strict input/output witness and matching
original input/result/start hashes yields `strict_sat_replayed`.
`correct_up_to_tolerance`, missing assignment, unsupported, no CE or process
timeout remain unqualified/void. The runner does not rewrite witness syntax.

The organizer checker supports SAT, not UNSAT. `ny-proof-bundle/v1` is recognized
and its original-input hashes/finalized trailer can be inspected. That is
**shape inspection only**. The repository currently has no complete general
independent arithmetic replayer for these hint bundles; this artifact route is
explicitly unsupported for automatic UNSAT qualification.

### Pinned independent command

`independent_command` requires `name`, `binary`, `source_paths`, `argv` and
`timeout_seconds`. Its binary must differ from the solver binary. Its argv starts
with `{replayer}` and uses `{request}` and `{response}` (also `{run_dir}`) as
literal paths. The response must not exist before invocation. The immutable
request schema is `ny_experiment_replay_request_v1`, with original model/property
paths, sealed artifacts, proposed verdict, and exact `bindings` including run,
source occurrence, solver/start/artifact hashes and replayer identity hash.

The response contract is:

```json
{
  "schema": "ny_experiment_replay_v1",
  "replayer": "registered-independent-lane-name",
  "request_sha256": "sha256-of-the-exact-request-file",
  "status": "checked",
  "basis": "original_model_property",
  "verdict": "unsat",
  "bindings": "COPY THE REQUEST BINDINGS OBJECT EXACTLY",
  "coverage": {"complete": true, "checked_obligations": 1, "remaining_obligations": 0},
  "arithmetic": {"checked": true, "method": "exact_rational", "checked_operations": 1}
}
```

The illustrative `bindings` string above must actually be the request's JSON
object. Coverage and operations must reflect independent checked work. UNSAT
requires positive complete coverage plus `directed_interval` or `exact_rational`
arithmetic. SAT instead requires `witness` with `input_checked`, `output_checked`,
`strict_violation` all true and positive `checked_coordinates`. Refusal, void,
zero coverage, incomplete cover, mismatched request/hash/occurrence or exit status
alone cannot qualify a claim. A positive generic receipt is labelled
`external_replay_attestation`; it **does not** become a trusted certificate.
Auditing/approving the named replayer's mathematics is separate from orchestration.
Official scoring credit is always false in these experiment archives.

## Costs and method engagement

Each query records whole solver wall time, box wait/orchestration, independent
replay cost, and whole arm cost including all hash checks. `wait4` captures child
CPU/max RSS/OS disk blocks. Linux additionally samples live group RSS; macOS
samples the direct child's RSS. These scopes are explicit. Block counts are
never mislabeled bytes. Cache snapshots record logical bytes/files and their
scan cost. VRAM is explicitly unavailable without a bound device sampler.

Each condition starts a new process. `process_cold` gives each run a new private
artifact cache directory; it does **not** establish cold OS page cache.
`artifact_cache_warm` repeats with the same private cache and charges/archives
the warmup. The solver must explicitly use `{cache_dir}` to engage this cache;
the before/after census is evidence of actual occupancy. A resident-model warm
protocol is not implemented and cannot be selected. Global/model cache state is
not inferred from a second subprocess.

If the solver writes `{phases}`, the runner accepts schema
`ny_experiment_phases_v1`, with `phases` and `methods`. Known phases are loading,
lowering, preprocessing, root_construction, intermediate_refresh, proposal_search,
certification, branching, witness_validation, output. A measured phase needs
`status: measured`, finite nonnegative `seconds`, `clock: exclusive_monotonic`
(optional integer `peak_bytes`). Sum of exclusive phase times cannot exceed the
whole query. Missing phases are explicitly `unavailable`, never invented zeros.

Each method requires `name`; integer admitted/attempted/completed/refused/skipped,
certified_improvements/actual_eliminations, peak_bytes, lookup_hits,
checked_scope_hits and coefficient_reads; finite nonnegative seconds; and a
counted `reasons` map for refused/skipped calls. Count relations are checked.
Lookup hits alone do not count as scope reuse. Method receipts are untrusted
observations and never close proof obligations.

The existing production `[frozen-dual]` log format is ingested without changing
Rust. Per-class Q1 actual grouped columns/events/reads/build reads/bytes remain
distinct from equivalent dense-work allowances and from the aggregate tier
summary. Rounded build/search costs remain diagnostics, not invented exclusive
phase timings. Actual per-phase telemetry and detailed native refusal reasons
remain unavailable on solver routes that do not emit the sidecar contract.

`receipt.json` is per run; `summary.json` binds the final frozen manifest and each
receipt. An earlier run can look locally stable while the final summary is void
after later input/source changes: consumers must inspect the summary too. The
launcher returns nonzero on experiment refusal/void or any refused, void or
unqualified run. Completed open results are reported as open, never pass.

## Lightweight checks

```sh
cd scripts
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest high_impact_experiments.test_harness -v
```

These use synthetic process/box doubles only. They test source occurrence
preservation, duplicate selection refusal, immutable external paths, changed
inputs and symlink targets, exact argv, process-group deadlines, replay hash and
positive arithmetic/witness coverage, void replay, missing phases, scope hit
counts and the production Q1 observation parser. They are not method A/Bs or
competition measurements.
