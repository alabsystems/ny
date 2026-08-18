<div align="center">
  <h1>ny</h1>
  <p><strong>Prove neural-network properties. Find counterexamples. Check certificates.</strong></p>
  <p>A neural-network verifier in Rust. Given a model and a property, <code>ny</code>
  returns a proof, a concrete counterexample, <code>unknown</code>, or a timeout
  — and on eligible proofs emits an exact-rational certificate an external checker can replay.</p>
  <p>
    <a href="#quick-start">Quick start</a> ·
    <a href="#check-nys-answers-without-trusting-ny">Check the answers</a> ·
    <a href="#use-ny">Use ny</a> ·
    <a href="#vnn-comp-benchmark-workflow">VNN-COMP</a> ·
    <a href="#methods-and-coverage">Methods</a> ·
    <a href="#gpu">GPU</a> ·
    <a href="#python">Python</a> ·
    <a href="#results">Results</a> ·
    <a href="https://github.com/alabsystems/ny/issues">Issues</a>
  </p>
</div>

## What is ny?

`ny` checks whether a neural network can ever violate a property — for example
"no perturbation of this image within `eps` changes its classification," or
"this control network keeps the system inside its safe region." You supply the
model and the property; `ny` returns one of:

| Answer | Meaning | Artifact |
| --- | --- | --- |
| **verified** | The property holds across the whole input region. | On eligible paths, an exact-rational certificate for an external checker. |
| **falsified** | The property can be violated. | A concrete input, re-executed through the model to confirm the violation. |
| **unknown** | `ny` could not prove or refute it within its methods or limits. | The reason (unsupported operator, exhausted domain budget). |
| **timeout** | `ny` reached the requested time limit before settling the property. | The elapsed time and partial-result context when available. |

It implements interval bound propagation (IBP), CROWN, α-CROWN, and β-CROWN
branch-and-bound, with a PGD attack for finding counterexamples. Arithmetic on
the verdict path carries certified rounding-error bounds, and every
counterexample is validated by concrete re-execution before it is reported.

Models: ONNX, NNet, PyTorch, SafeTensors, CoreML, GGUF. Properties: VNN-LIB.
Written in Rust.

> **Status:** `ny` is pre-1.0 research software under active development. All
> solving happens on the first-party [`ay`](https://github.com/alabsystems/ay)
> engine — by repo policy, no third-party LP/MIP/SMT/SAT solver appears on any
> verdict path. The public surface is intentionally small while the internals
> stabilize; the measurement methodology and rerun protocol are in
> [`reports/measured/README.md`](reports/measured/README.md), and the current
> score/cGAN/WGPU evidence boundary is in
> the internal current-state note (`docs/CURRENT_STATE_2026-08-10.md`, not part of the public snapshot).

## Quick start

Build `ny` from source with [`rustup`](https://rustup.rs/) and a native C/C++
build toolchain. The
repository pins the supported Rust version in `rust-toolchain.toml`; `rustup`
installs it automatically. On Ubuntu, install the native compiler, linker, and
C++ standard-library development files plus the checkout/download tools with:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl gzip pkg-config python3 unzip xz-utils
```

Then build the CLI:

```bash
git clone https://github.com/alabsystems/ny
cd ny
cargo build --locked --release -p ny-cli    # → target/release/ny
export PATH="$PWD/target/release:$PATH"
```

On Linux this needs an Ubuntu 24.04-era native toolchain or newer: the default
build downloads a prebuilt static ONNX Runtime compiled with a newer GCC than
Ubuntu 22.04 ships, so on 22.04 (gcc-11/binutils 2.38) the final link fails with
undefined references to ONNX Runtime internals. Ubuntu 24.04 and 26.04 are known
good once the C/C++ prerequisites above are installed. On an older host, build
inside a 24.04+ container or point `ORT_LIB_LOCATION` at a locally built ONNX
Runtime. The default runtime download uses rustls, so a system OpenSSL
development package is not required.

`crossing_relu.nnet` is a tiny network that computes `f(x) = |x|`. Its shipped
property asks `ny` to prove the output stays below a bound for *every* input in
`[-1, 1]`. Prove the safe version:

```bash
ny beta-crown tests/models/crossing_relu.nnet -p tests/models/crossing_relu_safe.vnnlib
```

```
--- Result ---
Status: VERIFIED
All inputs produce output < 1.9999999
```

`ny` proves the bound and exits `0`. On eligible verified models, `beta-crown`
also writes a certificate beside the model by default (`<model-stem>.cert.json`).
Pass `--no-certificate` when you only want the verdict.

Now ask for a bound the network *can* exceed — `|x|` reaches `0.5` inside the
box, so the property is false:

```bash
ny beta-crown tests/models/crossing_relu.nnet -p tests/models/crossing_relu_unsafe.vnnlib
```

```
--- Result ---
Status: VIOLATED
Found counterexample where output >= 0.49999997
Counterexample input: [0.508...]
```

`ny` returns `falsified` and exits `1`. The counterexample is not a guess: `ny`
runs it back through the model and confirms the violation before reporting it.

`verify` and `beta-crown` exit `0` proved, `1` falsified, `2` unknown, `3`
timeout, and `4` for an invalid invocation or operational error. `ny --help`
lists the supported command surface. The explicit `ny verify --allow-unknown`
CI override maps an `unknown` result to `0`; it never masks a timeout or an
operational error.

New to verification? `ny tutorial` is an interactive walk through the same
ideas — `ny tutorial demo` runs this exact check for you.

## Check ny's answers without trusting ny

A **falsified** verdict is self-evidencing: the reported input is re-executed
through the model and shown to violate the property before `ny` prints it. You
can re-run it yourself in any inference engine.

A **verified** verdict is backed, on eligible paths, by a proof you can replay
without trusting `ny`. For a sequential fully-connected ReLU network (up to 8192
hidden neurons), `ny` re-derives the CROWN bound in exact rational arithmetic
and writes it as an entailment + Farkas certificate:

```bash
ny beta-crown tests/models/crossing_relu.nnet \
  -p tests/models/crossing_relu_safe.vnnlib \
  --emit-certificate crossing_relu.cert.json
```

The certificate is self-checked before it is written, then stands on its own. It
records the claim, the exact-rational Farkas multipliers, and the linear
premises that discharge the unsafe region — no floating point, no reference to
`ny`'s solver state:

```json
{
  "claim": "exact CROWN proves 'Y_0 < 2' over the whole input box, emptying the unsafe region",
  "entailment": { "multipliers": ["0", "0", "1/2", "..."], "premises": ["..."] }
}
```

`ny`'s external kernel checker, **Clean**, replays these certificates
end-to-end. The [`ay` solver](https://github.com/alabsystems/ay) and
[Clean](https://github.com/alabsystems/clean) are published alongside NY.
The Lean proof layer is an NY-owned `NyProof` overlay over the exact mapped
Clean Lake dependency, never a copied Clean source tree. When it is enabled, certificate emission is on by default;
`--competition-mode` (set automatically by `ny vnncomp`) turns it off to
maximize verify-rate and never weakens a verdict's soundness.

## Use ny

`ny` auto-detects model formats and exposes a command per task:

```bash
ny beta-crown model.onnx -p prop.vnnlib            # complete β-CROWN branch-and-bound
ny verify model.onnx -p prop.vnnlib --method alpha # single-method bounds (ibp | crown | alpha)
ny vnncomp v1 acasxu_2023 model.onnx prop.vnnlib out.txt 116   # one VNN-COMP instance, competition protocol
ny inspect model.onnx                              # model structure
ny lipschitz model.onnx                            # sound certified global Lipschitz bound
ny diff a.onnx b.onnx                              # where two models diverge (porting)
ny compare a.onnx b.onnx                           # compare propagated output bounds
ny sensitivity model.onnx                          # which layers amplify input noise
ny quantize-check model.onnx                       # float16 / int8 quantization safety
ny profile-bounds model.onnx                       # bound-width growth through the network
ny weights info --file model.onnx                  # inspect stored weights
ny export --size tiny --output export_whisper.py   # generate a Whisper→ONNX Python exporter
ny gt verify model.onnx spec.gt.json --property dominates \
  --input-bounds="-1,1"                             # geometric ground truth
```

`ny verify` and `ny beta-crown` write their verdict to stdout and encode it in
the exit code; `ny vnncomp` writes the verdict to the competition results file.

### VNN-COMP benchmark workflow

The `ny benchmarks` commands acquire a corpus, inventory it, run local sweeps,
compare result banks, and check for path-dependent behavior. Run acquisition
and status commands from this checkout (or one of its subdirectories):

```bash
ny benchmarks download 2025 2026       # positional years
ny benchmarks download --all           # 2021-2026, including optional 2022
ny benchmarks status --year 2026
ny benchmarks status --year 2025 --year 2026 --strict
```

`download` runs the upstream setup scripts and can consume substantial network
bandwidth and disk space. Existing year directories are normally reused; this
is not a promise to update an existing checkout to the latest upstream commit.
`status` is an asset inventory (models, properties, and instance CSVs), not a
parse, execution, or correctness test. `--strict` fails when a category present
in the selected local corpus is missing required model/property assets.

Run one category, then resume the same output safely:

```bash
ny benchmarks run \
  --year 2026 \
  --vnnlib-version 2.0 \
  --category adaptive_cruise_control_non_linear_2026 \
  --output reports/sweeps/acc-2026.csv

ny benchmarks run \
  --year 2026 \
  --vnnlib-version 2.0 \
  --category adaptive_cruise_control_non_linear_2026 \
  --output reports/sweeps/acc-2026.csv \
  --resume
```

Flat corpora and categories with exactly one instance list need no version
option. If a checkout contains parallel `1.0/instances.csv` and
`2.0/instances.csv` lists, the runner refuses to guess: pass
`--vnnlib-version 1.0` or `--vnnlib-version 2.0`. The selection applies to
every requested category, so use separate category selections when a corpus
mixes versions.

The primary output uses the official six-column aggregate-results order:
`category,network,property,prepare_time,result,run_time`. The direct runner has
no separate preparation phase, so it records `prepare_time` as zero. A
neighboring `acc-2026.metadata.jsonl` is the authoritative, atomically replaced
state: it records stable duplicate occurrences, applied/official budgets, cap
provenance, and bounded diagnostics. The official CSV is regenerated from that
state in canonical order, so a crash between publications is repaired by the
next `--resume`. Prior `error` rows are retried at the same occurrence instead
of being silently skipped; summaries and exit status cover the entire bank,
including rows from earlier invocations.

`acc-2026.manifest.json` pins the schema/year, requested VNN-LIB version,
timeout-cap policy, `instances.csv` paths and hashes, configuration-tree hash,
executable hash/build provenance, and target OS/architecture. Resume refuses
incompatible state before running; changing any of those inputs requires a new
output bank.
Model/property asset bytes, hardware, drivers, and the wider process
environment are not hashed by this manifest and still belong in an audited
external run record.

An existing output bank is protected by default: use `--resume` only for a
compatible continuation, or `--overwrite` to discard all three artifacts and
start a new bank explicitly.

Each instance runs in an isolated child with an outer `budget + 30s` cleanup
watchdog. The extra 30 seconds does not extend scoring: a decided child whose
measured run time exceeds its official budget is recorded as `timeout`. A
failed child becomes an `error` row and the sweep continues so later instances
are still measured, but the command exits nonzero after preserving state.

By default the budget comes from each category's `instances.csv`.
`--timeout-cap` can only lower it; any capped sweep is a lower bound, not
an official result. `--limit` is a partial sweep, `--dry-run` writes no result
bank, and `--json` changes the stdout summary rather than suppressing the
CSV/metadata/manifest outputs. Matching the official instance budgets alone
does not make even an uncapped full local sweep competition-comparable:
hardware,
environment, software revision, preparation logs, inputs, and SAT witnesses
must be preserved and independently replayed before making an audited result
claim.

Compare the current bank with a historical reference and, optionally, a
multi-tool results field:

```bash
ny benchmarks score \
  --results reports/sweeps/acc-2026.csv \
  --baseline reports/sweeps/acc-2026-before.csv

ny benchmarks score \
  --results reports/measured \
  --official /path/to/vnncomp2026_results \
  --year 2026 \
  --track regular \
  --json
```

As of 2026-08-08, the official VNN-COMP 2026 aggregate results are public at
results commit
[`eea564f5`](https://github.com/VNN-COMP/vnncomp2026_results/tree/eea564f5d0a8751d8321a409214db14d03f452b8).
The pinned [regular table](https://github.com/VNN-COMP/vnncomp2026_results/blob/eea564f5d0a8751d8321a409214db14d03f452b8/SCORING/latex/total_regular.tex)
reports alpha-beta-CROWN at 2344.0 and Vibecheck at 2278.6; the pinned
[extended table](https://github.com/VNN-COMP/vnncomp2026_results/blob/eea564f5d0a8751d8321a409214db14d03f452b8/SCORING/latex/total_extended.tex)
reports alpha-beta-CROWN at 600.0. Those artifacts make the public organizer
field usable with `--official`. They do **not** establish an NY score: NY was
not an official entrant and `reports/measured-2026/` is not a complete 2026
result bank. Partial local rows and 2025 carry-overs must not be extrapolated
across resampled 2026 properties.

The comparison uses the union of current and reference identities, so a row
that disappeared from the current bank is still a regression. The baseline is
only a historical reference, never ground truth; a `sat`↔`unsat` flip is a
soundness alarm requiring evidence review. Without `--official`, the command
reports exact row/solve deltas and deliberately invents no normalized score.

With `--official`, `--year 2025` or `--year 2026` is required. The explicit
edition selects that release's category membership; it is never inferred
because several category ids occur in both years but changed tracks. The scorer
models the VNN-COMP raw-point rules: a correct decided result earns +10, an
incorrect result earns −150, and an undecided result earns zero; there is no
time bonus. `--track regular` (the default) and `--track extended` keep the two
published category sets separate. The scorer then applies winner-relative
per-benchmark normalization, clipped at zero, for selected-track categories
represented in the supplied field. Only normalized instance identities present
in that field earn modeled points; extra current rows remain visible in
solve-count comparisons but cannot inflate the score.

This is still a supplied-field/category model, not official Regular or Extended
track scoring. The imported CSV reader requires finite nonnegative recorded
times, but it does not load each version's `instances.csv` to force over-budget
decisions to timeout. It structurally normalizes relational model lists and
path suffixes; it does not reproduce the organizer's cross-version positional
alignment checks. Raw CSVs also omit organizer witness-validation outcomes, so
the model assumes every `sat` row carries a strictly valid counterexample. It
can therefore penalize a contradicted `unsat`, but cannot detect an invalid SAT
witness from CSV alone. The JSON fields `score_year`, `score_track`,
`modeled_score_caveat`, and `officially_scored_benchmarks` make the selected
edition and those limits machine-visible.

Finally, check that decided verdicts do not depend on recognizable paths:

```bash
ny benchmarks reseed \
  --year 2026 \
  --vnnlib-version 2.0 \
  --category adaptive_cruise_control_non_linear_2026 \
  --limit 10
```

`reseed` runs each original and a byte-identical twin under fresh,
content-derived model/property names (including relational and `.onnx.gz`
models). It is a rename metamorphism, not regeneration of the underlying
dataset or seed. A pass requires at least one pair with matching decided
`sat`/`unsat` verdicts and no timeout, unknown, child error, missing asset, or
disagreement; matching inconclusive results do not pass.

For audited VNN-COMP counterexamples in NY's flat result format, recompute only
the `Y_*` values with the requested ONNX Runtime version and provider while
preserving every submitted `X_*` token:

```bash
python scripts/canonicalize_vnncomp_ce_y.py \
  --onnx model.onnx \
  --result original.results \
  --output canonical.results \
  --vnnlib property.vnnlib \
  --scoring-dir external_tools/vnncomp2026_results/SCORING \
  --receipt canonical.receipt.json \
  --require-onnxruntime 1.16.3 \
  --require-provider CPUExecutionProvider
```

The helper accepts a standalone `sat`, then contiguous `X_0...X_n` entries
followed by contiguous `Y_0...Y_m` entries. It requires exactly one ONNX model
input and exactly one model output, then flattens the sole output tensor; it is
not a converter for VNN-LIB 2.0 tensor or relational assignments. The command
pins the requested ONNX Runtime version and provider, installs that provider as
the session's only execution provider, and never overwrites an existing output
or receipt. A mandatory tol=0 replay gate (official SCORING checker at
abs_tol=0/rel_tol=0, plus a written-witness violation check on the canonical
`Y_*` values) refuses any canonicalization whose witness stops violating the
property at zero tolerance; the strict verdict and checker file hashes are
sealed into the receipt. The JSON receipt hashes the model, source result, canonical result,
preserved numeric `X_*` token stream, and recomputed `Y_*` token stream.
Python, NumPy, ONNX Runtime, provider, shape, and dtype metadata are recorded in
the receipt but, except for the required ONNX Runtime version/provider, are not
independently pinned or hashed. The hardened atomic-publication path requires a
POSIX host that exposes `O_DIRECTORY` and `O_NOFOLLOW`.

## Methods and coverage

| Stage | What `ny` does | Evidence it produces |
| --- | --- | --- |
| Incomplete bounds | IBP, CROWN, α-CROWN linear bound propagation | Certified enclosures with rounding-error bounds |
| Complete verification | β-CROWN branch-and-bound (input and ReLU splitting) | `verified`, or a counterexample |
| SAT-ReLU recovery | Bit-exact gadget decompilation to CNF; in-process `ay-sat` | SAT witness re-executed through the graph; UNSAT admitted only after ResolutionDag RUP replay |
| Counterexamples | PGD attack, then concrete re-execution | The input, re-run through the model |
| Certificates | Exact-rational entailment + Farkas (eligible FC-ReLU nets) | `<model-stem>.cert.json`, re-checkable by an external kernel |
| Escalation | Exact SMT via `ay` (`ny gt verify --escalate smt`) | An `ay` verdict on the escalated query |

### VNN-LIB coverage

The ordinary property path supports VNN-LIB 1.0 scalar declarations and affine
constraints, plus partial VNN-LIB 2.0 tensor declarations, indexed variables,
network wrappers, and relational forms. Unsupported syntax fails closed.
Strict output comparisons (`Y < ...` and `Y > ...`) retain their strict
semantics. Strict affine input-domain atoms are currently rejected because the
ordinary input-box representation is closed; competition entry points return
`unknown` instead of treating an excluded endpoint as inclusive.

For VNN-LIB 2.0 formulas rejected specifically because their arithmetic is
nonlinear, the competition path has a dedicated fallback for tensor variables,
`+`, `-`, `*`, `/`, comparisons (`<`, `<=`, `>`, `>=`, `=`, `==`, `!=`), and
`and`/`or`/`not`. Source decimal literals retain their exact rational meaning.
A `sat` result requires the complete unrelaxed formula to hold at a concrete
input, trusted ONNX Runtime replay, and a definite true result from the sound
f64 output enclosure; false, undecided, or unsupported evaluation returns
`unknown`. This fallback does not reinterpret an otherwise-affine strict input
atom as nonlinear arithmetic.

The nonlinear `unsat` lane is intentionally narrower: every input coordinate
must have a finite simple bound, the graph must support sound outward-rounded
f64 interval propagation, and input-space branch-and-bound is limited to at
most four scalar inputs. It reports `unsat` only after every subbox is
definitely refuted against the original formula. An unsupported operator,
division interval crossing zero, unresolved boundary, higher-dimensional box,
resource limit, or exhausted budget fails closed to `unknown`. This is targeted
nonlinear coverage, not a complete implementation of arbitrary VNN-LIB 2.0
arithmetic.

Whisper support currently includes model loading and component extraction.
Whisper block compatibility calls execute CPU graph IBP only. Unsupported GPU,
zonotope, CROWN, and overflow-control requests fail closed without returning
substitute bounds. Sequential verification is unavailable, and Whisper CLI
verdict commands fail closed without producing a verdict. The CROWN-named
block API also fails closed. Compatibility detail records set
`stage_metrics_available = false` when their stage-looking widths merely alias
the final graph-IBP width. Heuristic LayerNorm forward-mode requests are
rejected rather than returned as untagged proof-looking bounds.

`ny export` writes a Python exporter; it does not download or convert a model
inside the Rust process. Install its dependencies and run the generated script:

```bash
python -m pip install torch openai-whisper onnx onnxscript
ny export --size tiny --output export_whisper.py
python export_whisper.py
```

Supported sizes are `tiny`, `base`, `small`, `medium`, and `large`. The script
exports a CPU encoder with the model-declared mel-bin and fixed audio-context
dimensions, validates the resulting ONNX file, and leaves only the batch axis
dynamic. Current Torch exporters write tensor weights to a `.data` sidecar so
even the large encoder stays below ONNX protobuf's 2 GiB message limit; keep
that sidecar beside the `.onnx` file. NY's file loader resolves these relative
external tensor slices, while its in-memory byte loader intentionally rejects
models that have no filesystem origin. For external-data models, NY relies on
the exporter's authored shapes instead of letting ONNX Runtime reopen sidecars
outside NY's filesystem capability. The resulting model is for loading and
block analysis; Whisper CLI verification remains fail-closed as described
above.

Decoder transformer support currently covers loading, structure inspection,
and heuristic causal-self-attention/MLP graph artifacts. Cross-attention
extraction fails closed until the graph layer has a sound multi-input contract.
CPU, GPU, single-block, and sequential decoder verification compatibility APIs
fail closed without bounds or details until their inferred attention topology,
projection conventions, masks, normalization attributes, and cross-attention
composition are proven equivalent to the loaded graph.

The LP/MIP lane (`ny-mip`, in-process `ay-milp`) is available behind the `mip`
feature. The exact SMT escalation path runs on an `ay` binary located at
runtime through `$NY_AY` / `$PATH`.

An `mip` build also contains a category-typed, profile-limited, default-dark
cGAN input-leaf oracle selected only by `NY_CGAN_INPUT_LEAF=1` for
`cgan_2023`. It authenticates the authored imgSz32 nCh1/nCh3 profiles and
matching scalar-band properties rather than accepting the whole category. Its
final `Relu_23` strict M17/M20 portfolio may discharge an authenticated input-split
leaf; M24 remains counterfactual telemetry. The deeper
`Relu_20 -> BatchNormalization_21 -> Conv_22` pullback is implemented as
dormant library/validation machinery but is deliberately not requested by the
production leaf constructor, so its two measurements are `NotRequested` and
have no verdict effect. Re-enabling it safely requires a second pass after the
historical tail; retaining optional `Relu_20` state across mandatory work would
change finite-cap completion. See the
the internal current-state note.
The current cGAN bank has 22 rows (12 SAT, 7 timeout, 2 unknown, and the
`test_nano` UNSAT). Its 2026-08-11 parity-budget refresh changed no verdict;
the bank commit reports a 1390.8 regular standing and closes the
budget-starvation thesis. That commit-local standing does not replace the older
fully pinned aggregate ledger described below, and the bank records no solve
attributable to this default-dark leaf route.

The supported graph BaB frontiers can budget their estimated resident domain
payload directly or through a preset. Coverage includes the ordinary and
precomputed ReLU-split heaps plus GPU `DomainList` ReLU- and input-split routes:

```bash
ny beta-crown model.onnx -p prop.vnnlib --max-queue-bytes 2147483648
```

```yaml
bab:
  max_queue_bytes: 2147483648
```

This setting sums NY's estimated live domain payload; it is not a process-RSS
or allocator limit. `0` (the default) means unlimited. A queue always retains
at least one domain, so one oversized domain may exceed the configured budget.
If enforcement evicts any unverified domain, the final verdict is `unknown`,
never a proof based on an incomplete search. Grouped-disjunctive `DomainList`
storage rejects a nonzero byte cap until its row sidecar has a complete census.

## GPU

The default build includes a cross-platform wgpu compute backend (Metal,
Vulkan, DX12):

```bash
ny beta-crown model.onnx -p prop.vnnlib --backend wgpu
```

Verdict-bearing WGPU authority is explicit and request-scoped. An ordinary
`WgpuDevice` is always unarmed. `WgpuVerdictRequest::new()` is the typed public
request: construction of a verdict device creates one adapter context, runs
the five live qualification rungs eagerly, and produces a complete
`WgpuVerdictReport`. A successful device retains that report; a refusal
preserves it in the typed qualification error. Only success lets the public
`ComputeDevice` proof adapter expose CROWN. IBP, DAG, ordinary `GemmEngine`,
and every other proof accessor remain closed.

`--backend wgpu` asks for that qualified CROWN route. If device construction or
any rung refuses, the CLI fails closed to the sound CPU implementation. It
emits the unconditional `NY-HARNESS: BACKEND-OVERRIDE` marker and an
authenticated `ProofBackendReceipt` recording the request source,
qualification/failure detail, effective backend, fallback, adapter, and proof
provenance; JSON mode carries the same receipt. There is no CLI opt-out that
allows an unqualified GPU bound to decide a verdict. Building with `--features
cuda` adds a dynamically loaded native CUDA/cuBLAS accelerator for supported
non-WGPU routes.

Current Apple status is measured rather than inferred: on the 2026-08-11 Apple
M5 Max/Metal run the live ladder passed 3/5 (IEEE f32, host EFT, and sentinel
taint) but failed EFT primitives and gradual underflow. The typed request
therefore refused and the hermetic proof A/B fell back to byte-identical CPU
output. See the
the internal WGPU Metal qualification record.

One narrow WGPU exception is verdict-safe without opening that general
authority: `FlValueGemmDevice` may compute only forward-linear value products
under the `NY_FORWARD_LINEAR_F32` treatment. The caller charges the complete
f32 accumulation/FTZ error separately, the certified error-base products stay
on CPU f64, and non-finite or finite-sentinel results cause typed fallback. It
cannot be used as an ordinary `GemmEngine` or CROWN backend.

The raw `WgpuDevice` CROWN authority gate opened in the 2026-08-11 UTC B0
review. U1 and U3 are discharged. U4 is also discharged: the AUTO/default
route carries out-of-band words through the applicable twins, on-device row
state, host folds, resident Conv, and ResNet segment composition; C2 protects
the error-lowering EFT combine and the armed C1 preflight refuses absent or
tainted rows. Host fused Conv and segment-resident device streams still
typed-refuse instead of dropping word state; other unsupported configurations
also refuse. U5/U6 are discharged by adversarial device oracles for activation
Lipschitz/intercept propagation and concretize enclosure/fail-closed identity.
The measured GB10/Vulkan + DenormPreserve ladder is 5/5. There is no ambient
self-arm environment grant: only the exact typed verdict request plus all five
live probes can qualify the narrow public CROWN accessor, and every failure
remains closed. The canonical current ledger is the the WGPU verdict-authority section of the internal current-state note.

Ordinary `GemmEngine` methods on `WgpuDevice`, including `gemm_f32` and
`conv_transpose_2d`, remain typed refusals. Its CROWN accessor is the narrow
exception: after the explicit request and successful cached qualification it
may expose the sound resident backward. Separate crate-internal inherent
kernels remain available to device tests and diagnostics. Complete Clip does
not trust a device suggestion as a bound: private host replay must reproduce
it in the exact clip context before a context-bound affine-provenance token can
be minted. Device code cannot construct that token, so malformed, stale, or
cross-context suggestions remain non-authoritative and fall back safely.

Falsification is exempt and stays accelerated. Attack steering is
verdict-neutral by construction — its floats only choose where to look next,
and every candidate still passes the unchanged admission gates — so it keeps
whichever accelerator the host has (the shared CUDA engine on an NVIDIA box,
WGPU otherwise) even when the proof lane is forced to CPU.

## Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make PYTHON="$VIRTUAL_ENV/bin/python" test-python-tooling

# If the first line fails with "ensurepip is not available" (Debian/Ubuntu ship
# venv separately) and you cannot `apt install python3-venv`, uv needs neither:
#   uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt

# Optional: build and test the Python bindings.
python -m pip install maturin
cd crates/ny-python
maturin develop --release    # Python ≥ 3.9
```

The `ny` module provides `verify()` and `verify_torch()` (IBP/CROWN/α/β, on ONNX
files or in-memory `torch.nn.Module`s), model diffing and equivalence checks
across ONNX/SafeTensors/PyTorch/GGUF/CoreML, quantization-safety checks, bound
profiling, and a pytest plugin (`ny_pytest`) with `assert_verified`,
`assert_equivalent`, fixtures, and `--ny-*` options. Fully typed (`py.typed` +
`.pyi` stubs).

## Results

Most checked-in VNN-COMP 2025 CSV rows are legacy six-column raw-verdict
measurements without retained SAT witnesses. Two newer seven-column UNSAT rows
(`linearizenn_2024` and `cora_2024`) have immutable external run artifacts and
are bound by `reports/measured/regular_evidence_index.json`; the read-only
validator reopens their manifests, hashes, official occurrence, budget, and
published truth before the scorecard can credit them. The older seven-column
`sat_relu` file identifies a sealed 100/100 local run, but its
manifest/log/input/witness bundle is absent from this checkout, so the CSV
alone remains neither revalidated score evidence nor an official rank.
See [`reports/measured/README.md`](reports/measured/README.md) for the immutable
rerun, witness-replay, completion-gated promotion, and `--require-evidence`
scorecard protocol. Earlier competition plans in this tree are experiment
notebooks, not current score evidence.

The latest pinned aggregate ledger is the 2026-08-07 snapshot in
`configs/preset_score_at_risk.yaml` (regular-16 1394.5, extended-9 477.6,
combined 1872.1), not an official result. Several category banks changed after
that snapshot, so it is also not an exact aggregate for current `main`; no
later fully refreshed aggregate and provenance block is checked in. NY has no
exact 2026 score because `reports/measured-2026/` contains only the `cersyve`
and `monotonic_acasxu_2026` category banks, not a complete regular or extended
track. The the internal current-state note separates
pinned snapshots, later row evidence, and implemented-but-unmeasured features.

The checked-in historical `sat_relu` proof corpus additionally LRAT/Lean-replays
all 50 UNSAT rows from its recorded sweep. The current runtime path does not
invoke Lean per request: it independently regenerates an in-memory AY
ResolutionDag and RUP-replays it before admitting `verified`. On `cersyve`, 5
of 9 control systems are certified safe for an unbounded horizon by a
kernel-checked induction theorem over `ny`'s one-step verdicts.

## Testing and proofs

`ny` is covered by thousands of Rust tests, property-based tests, and four fuzz
targets, one of which asserts that computed bounds enclose concrete executions.
Kani proof harnesses verify the `ny-relaxation` scalar-relaxation mirror
(`proofs/kani/`; binding to the production `ny-propagate` paths is in progress —
see the status note in `crates/ny-relaxation/src/lib.rs`), Lean proofs cover the
certificate lemmas (`crates/ny-cert/proofs/lean/`), and a model-checked TLA+
specification covers the documented CNF/cell/MIP/BaB verdict-admission
abstraction (`specs/tla/`). Direct relational, TLL, fractional-head, and
post-BaB margin-row lanes are outside that model.

Cargo is the canonical correctness entry point. The complete hermetic workspace
lane is:

```bash
cargo check --workspace --exclude ny-python
cargo test --locked --workspace --exclude ny-python
```

The default suite is hermetic and contains no ignored tests. Python-binding,
real-corpus, external-checker, GPU, and measurement lanes are explicit and fail
fast when selected without their prerequisites; see
the internal `TEST_CONFORMANCE.md` note (not part of the public snapshot) for the commands and policy. The
ordered solver/test program is tracked in the
the internal 1–6 priority roadmap; it
deliberately adds no hosted CI.

The toolchain is pinned by `rust-toolchain.toml`.

## Workspace

| Layer | Crates |
| --- | --- |
| Core types | `ny-core`, `ny-tensor` |
| Runtime contracts and configuration | `ny-contracts`, `ny-levers` |
| Model ingestion | `ny-load`, `ny-build`, `ny-onnx`, `ny-trace-bridge` |
| Verification engines | `ny-propagate`, `ny-gpu`, `ny-cuda`, `ny-mip` |
| Ground truth | `ny-groundtruth` |
| Certificates | `ny-cert`, `ny-relaxation` |
| Interfaces | `ny-cli`, `ny-api`, `ny-python` |
| Dev tooling | `ny-test-utils`, `ny-lint-guard` |

## Developing and contributing

Contribution basics (build, test, licensing, reporting) are in
[`CONTRIBUTING.md`](CONTRIBUTING.md). See [`SECURITY.md`](SECURITY.md) for
private vulnerability reports and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for
community expectations.

## License

Apache License 2.0. See [`LICENSE`](LICENSE). `ny` was created by Andrew Yates.
