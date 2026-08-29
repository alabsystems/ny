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

> **Status:** `ny` is pre-1.0 research software under active development, and
> its public surface is intentionally small while the internals stabilize. All
> solving happens on the first-party [`ay`](https://github.com/alabsystems/ay)
> engine — by repo policy, no third-party LP/MIP/SMT/SAT solver appears on any
> verdict path. Measurement methodology and the rerun protocol are in
> [`reports/measured/README.md`](reports/measured/README.md); the current
> score/cGAN/WGPU evidence boundary is in the internal current-state note
> (`docs/CURRENT_STATE_2026-08-10.md`, not part of the public snapshot).

## Quick start

Build `ny` from source with [`rustup`](https://rustup.rs/) and a native C/C++
build toolchain. The repository pins the supported Rust version in
`rust-toolchain.toml`; `rustup` installs it automatically.

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl gzip pkg-config python3 unzip xz-utils

git clone https://github.com/alabsystems/ny
cd ny
cargo build --locked --release -p ny-cli    # → target/release/ny
export PATH="$PWD/target/release:$PATH"
```

On Linux this needs an Ubuntu 24.04-era native toolchain or newer (24.04 and
26.04 are known good): the build downloads a prebuilt static ONNX Runtime
compiled with a newer GCC than 22.04 ships, so on 22.04 (gcc-11/binutils 2.38)
the final link fails with undefined references to ONNX Runtime internals. On an
older host, build inside a 24.04+ container or point `ORT_LIB_LOCATION` at a
locally built ONNX Runtime. The runtime download uses rustls, so no system
OpenSSL development package is needed.

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
timeout, and `4` for an invalid invocation or operational error; the explicit
`ny verify --allow-unknown` CI override maps `unknown` to `0` and never masks a
timeout or an operational error. `ny --version` reports a build-derived version
(`<major>.<minor>.<commit count>`, plus a dirty suffix) rather than the frozen
manifest version, and `ny --build-id` prints the full content-addressed build
identity.

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

`ny` auto-detects model formats and exposes a command per task. `ny --help` and
`ny <command> --help` are the authoritative flag reference.

```bash
ny beta-crown model.onnx -p prop.vnnlib            # complete β-CROWN branch-and-bound
ny verify model.onnx -p prop.vnnlib --method alpha # single-method bounds (ibp | crown | alpha;
                                                   #   beta is sequential-only)
ny vnncomp v1 acasxu_2023 model.onnx prop.vnnlib out.txt 116   # one VNN-COMP instance, competition protocol
ny vnncomp plan acasxu_2023 model.onnx prop.vnnlib 116         # the resolved run plan, without running it
ny coverage                                        # soundness class of every ingestable operator
ny claims verify                                   # re-derive recorded empirical claims from pinned artifacts
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
the exit code; `ny vnncomp` writes it to the competition results file.

`ny coverage` prints the trace-bridge soundness manifest: all 123 ingestable
operators classified `exact`, `sound`, `sound_but_loose`, or `unsupported` by a
wildcard-free `match`, so a new operator fails to compile until it is
deliberately classified. It is the soundness *taxonomy*, not the porting status —
`translate()` still fails closed on any classified-but-unported operator.

`ny claims verify` reads a claims ledger, re-derives each recorded empirical
claim from the content-pinned artifacts it was made from, and prints `TRUE` /
`FALSE` / `STALE` / `VACUOUS` / `UNGROUNDED`. It also ships as the leaf
`ny-claims` binary, which links in seconds instead of relinking the whole CLI.

### VNN-COMP benchmark workflow

`ny benchmarks` acquires a corpus, inventories it, runs local sweeps, compares
result banks, and checks for path-dependent behavior. Run it from this checkout
or a subdirectory; `--help` on each subcommand is the authoritative reference.

```bash
ny benchmarks download 2025 2026                    # positional years; --all adds historical ones
ny benchmarks status --year 2026 --strict           # asset inventory, not a parse or a run
ny benchmarks run --year 2026 --vnnlib-version 2.0 \
  --category adaptive_cruise_control_non_linear_2026 \
  --output reports/sweeps/acc-2026.csv [--resume]
ny benchmarks score --results reports/measured-2026 --baseline previous.csv
ny benchmarks reseed --year 2026 --category adaptive_cruise_control_non_linear_2026 --limit 10
```

`download` runs the upstream setup scripts, can consume substantial bandwidth
and disk, and reuses rather than updates existing year directories. `status`
inventories models, properties, and instance CSVs; `--strict` fails when a
locally present category is missing required assets. `--vnnlib-version` is
needed only where a checkout holds parallel `1.0/` and `2.0/` instance lists —
the runner refuses to guess — and applies to every requested category.

**`run` writes three paired artifacts.** The CSV uses the official six-column
order `category,network,property,prepare_time,result,run_time`, with
`prepare_time` zero because this runner has no separate preparation phase.
`<name>.metadata.jsonl` is the authoritative, atomically replaced row state, and
the CSV is regenerated from it in canonical order, so a crash between
publications is repaired by the next `--resume`. `<name>.manifest.json` pins the
schema/year, VNN-LIB version, timeout-cap policy, `instances.csv` hashes,
configuration-tree hash, executable hash and build provenance, applied lever arm,
and target OS/architecture, and refuses a resume whose provenance no longer
matches — but asset bytes, hardware, drivers, and the process environment are
*not* hashed and still belong in an audited external run record. An existing bank
is protected by default: `--resume` continues a compatible one, `--overwrite`
discards all three.

Each instance runs in an isolated child under an outer `budget + 30s` cleanup
watchdog; the grace never extends scoring, so a decided child that exceeds its
official budget is recorded as `timeout`. A failed child becomes an `error` row
and the sweep continues, but the command exits nonzero after preserving state,
and `--resume` retries those rows rather than skipping them. Budgets otherwise
come from each category's `instances.csv`.

| `run` flag | Effect |
| --- | --- |
| `--timeout-cap N` | Only *lowers* each `instances.csv` budget; any capped sweep is a lower bound, not an official result. |
| `--instances FILE` | Runs exactly the named `category,onnx,vnnlib` rows. A row the corpus lacks is an error, never a skip. |
| `--lever NAME=VALUE` | Applies a validated arm to every child *as data* — recorded per row and sealed into the manifest, so a resume under a different arm refuses. Inert and `Unsafe` axes are rejected rather than measured; `ny benchmarks space` prints the compiled search space. |
| `--cache off\|read\|read-write` | Verdict cache keyed on binary, configuration, category, model, property, budget, arm, backend, host, and ambient lever set. A served row is marked `from_cache`, so a replay is never read as evidence that a child ran. |

`--limit N` takes a prefix of each category, `--dry-run` writes no bank, and
`--json` changes the stdout summary without suppressing the three artifacts.
Matching the official budgets does not make even an uncapped full local sweep
competition-comparable: hardware, environment, software revision, preparation
logs, inputs, and SAT witnesses must all be preserved and independently replayed
before an audited result claim.

**`score`** compares a bank against a baseline over the union of both identity
sets, so a row that disappeared is still a regression. The baseline is a
historical reference, never ground truth, and a `sat`↔`unsat` flip is a
soundness alarm requiring evidence review. Without `--official` it reports exact
row/solve deltas and invents no normalized score.

```bash
ny benchmarks score --results reports/measured-2026 \
  --official /path/to/vnncomp2026_results \
  --year 2026 --track regular --budget-year 2026 --json
```

With `--official`, `--year 2025` or `--year 2026` is required — membership is
never inferred, because several category ids occur in both editions but changed
tracks. The scorer models the VNN-COMP raw-point rules (+10 correct decided,
−150 incorrect, 0 undecided; no time bonus), keeps `--track regular` (default)
and `--track extended` separate, and applies winner-relative per-benchmark
normalization clipped at zero over the categories the supplied field represents;
only identities present in that field earn modeled points. `--budget-year YEAR`
loads that edition's per-instance budgets and drops any solved row recorded above
its own budget as a phantom point — omit it and runtimes go unchecked, which the
report states. `--witnesses assumed-valid` (default) treats every `sat` as a
strictly valid counterexample, so an `unsat` it contradicts takes the real −150;
`unvalidated` credits every claimed verdict and only reproduces a published
board's arithmetic.

The result is a supplied-field model, not official Regular or Extended scoring:
it does not reproduce the organizer's cross-version positional alignment, and raw
CSVs omit organizer witness validation, so it can penalize a contradicted `unsat`
but cannot detect an invalid SAT witness. `score_year`, `score_track`,
`modeled_score_caveat`, and `officially_scored_benchmarks` make the edition and
those limits machine-visible in `--json`. The official 2026 results — public at
[`VNN-COMP/vnncomp2026_results`](https://github.com/VNN-COMP/vnncomp2026_results),
pinned here at
[`eea564f5`](https://github.com/VNN-COMP/vnncomp2026_results/tree/eea564f5d0a8751d8321a409214db14d03f452b8)
— are what make `--official` usable; that field establishes no NY score.

**`reseed`** runs each original and a byte-identical twin under fresh,
content-derived model and property names (including relational and `.onnx.gz`
models) — a rename metamorphism, not regeneration of the dataset or its seed. A
pass needs at least one pair with matching decided `sat`/`unsat` verdicts and no
timeout, unknown, child error, missing asset, or disagreement; matching
inconclusive results do not pass.

**Counterexample canonicalization.** For audited VNN-COMP counterexamples in
NY's flat result format, recompute only the `Y_*` values under a pinned ONNX
Runtime while preserving every submitted `X_*` token:

```bash
python scripts/canonicalize_vnncomp_ce_y.py \
  --onnx model.onnx --result original.results --output canonical.results \
  --vnnlib property.vnnlib \
  --scoring-dir /path/to/vnncomp2026_results/SCORING \
  --receipt canonical.receipt.json \
  --require-onnxruntime 1.16.3 --require-provider CPUExecutionProvider
```

It handles a standalone `sat` over exactly one model input and one model output —
not VNN-LIB 2.0 tensor or relational assignments — pins the session to the
requested provider, and never overwrites an existing output or receipt. A
mandatory tol=0 gate (the official SCORING checker at `abs_tol=0`/`rel_tol=0`,
plus a written-witness violation check on the canonical `Y_*`) refuses any
canonicalization whose witness stops violating the property at zero tolerance,
and the receipt seals that strict verdict, the checker hashes, and hashes of the
model, both results, and both token streams. Atomic publication requires a POSIX
host exposing `O_DIRECTORY` and `O_NOFOLLOW`.

## Methods and coverage

| Stage | What `ny` does | Evidence it produces |
| --- | --- | --- |
| Incomplete bounds | IBP, CROWN, α-CROWN linear bound propagation | Certified enclosures with rounding-error bounds |
| Complete verification | β-CROWN branch-and-bound (input and ReLU splitting) | `verified`, or a counterexample |
| SAT-ReLU recovery | Bit-exact gadget decompilation to CNF; in-process `ay-sat` | SAT witness re-executed through the graph; UNSAT admitted only after ResolutionDag RUP replay |
| Counterexamples | PGD attack, then concrete re-execution | The input, re-run through the model |
| Certificates | Exact-rational entailment + Farkas (eligible FC-ReLU nets) | `<model-stem>.cert.json`, re-checkable by an external kernel |
| Escalation | Exact SMT via `ay` (`ny gt verify --escalate smt`) | An `ay` verdict on the escalated query |

`ny coverage` prints the per-operator soundness classification behind the first
two rows.

### VNN-LIB coverage

The ordinary property path supports VNN-LIB 1.0 scalar declarations and affine
constraints, plus partial VNN-LIB 2.0 tensor declarations, indexed variables,
network wrappers, and relational forms; unsupported syntax fails closed. Strict
output comparisons (`Y < ...`, `Y > ...`) keep their strict semantics, while
strict affine input-domain atoms are rejected because the ordinary input-box
representation is closed — competition entry points return `unknown` rather than
treat an excluded endpoint as inclusive.

For VNN-LIB 2.0 formulas rejected specifically because their arithmetic is
nonlinear, the competition path has a dedicated fallback covering tensor
variables, `+ - * /`, the comparisons `< <= > >= = == !=`, and `and`/`or`/`not`,
with source decimals retaining their exact rational meaning. A `sat` requires the
complete unrelaxed formula to hold at a concrete input, trusted ONNX Runtime
replay, and a definite true result from the sound f64 output enclosure; anything
else returns `unknown`, and the fallback never reinterprets an otherwise-affine
strict input atom as nonlinear arithmetic. The nonlinear `unsat` lane is narrower
still: every input coordinate needs a finite simple bound, the graph must support
sound outward-rounded f64 interval propagation, input-space branch-and-bound is
limited to at most four scalar inputs, and `unsat` is reported only after every
subbox is definitely refuted against the original formula. An unsupported
operator, a division interval crossing zero, an unresolved boundary, a
higher-dimensional box, a resource limit, or an exhausted budget fails closed to
`unknown`. This is targeted coverage, not a complete implementation of VNN-LIB
2.0 arithmetic.

### Transformer models

Transformer support stops short of verdicts. Whisper coverage is model loading
and component extraction; block compatibility calls execute CPU graph IBP only,
and GPU, zonotope, CROWN, sequential, and overflow-control requests fail closed
rather than return substitute bounds — as do the Whisper CLI verdict commands and
heuristic LayerNorm forward-mode requests. Decoder support likewise covers
loading, structure inspection, and heuristic causal-self-attention/MLP graph
artifacts; cross-attention extraction and every decoder verification
compatibility API fail closed until the inferred topology, projections, masks,
normalization attributes, and cross-attention composition are proven equivalent
to the loaded graph.

`ny export` writes a Python exporter; it does not download or convert a model
inside the Rust process. Install its dependencies and run the generated script:

```bash
python -m pip install torch openai-whisper onnx onnxscript
ny export --size tiny --output export_whisper.py
python export_whisper.py
```

Sizes are `tiny`, `base`, `small`, `medium`, and `large`. The script exports a
CPU encoder with the model-declared mel-bin and fixed audio-context dimensions,
validates the resulting ONNX file, and leaves only the batch axis dynamic. Torch
writes tensor weights to a `.data` sidecar so even the large encoder stays below
ONNX protobuf's 2 GiB message limit; keep it beside the `.onnx` file. NY's file
loader resolves those relative external tensor slices using the exporter's
authored shapes, while its in-memory byte loader deliberately rejects models with
no filesystem origin.

### Optional lanes and memory budgets

The LP/MIP lane (`ny-mip`, in-process `ay-milp`) is available behind the `mip`
feature; exact SMT escalation runs on an `ay` binary located at runtime through
`$NY_AY` / `$PATH`. An `mip` build also carries a default-dark cGAN input-leaf
oracle, armed only by `NY_CGAN_INPUT_LEAF=1` and only for authenticated
`cgan_2023` imgSz32 nCh1/nCh3 profiles with matching scalar-band properties; its
deeper pullback is dormant machinery the production leaf constructor never
requests, and the 22-row `cgan_2023` bank (12 `sat`, 7 `timeout`, 3 `unsat`)
records no solve attributable to this route.

The supported graph BaB frontiers — the ordinary and precomputed ReLU-split
heaps plus the GPU `DomainList` ReLU- and input-split routes — can budget their
estimated resident domain payload directly or through a preset:

```bash
ny beta-crown model.onnx -p prop.vnnlib --max-queue-bytes 2147483648   # preset key: bab.max_queue_bytes
```

This sums NY's estimated live domain payload; it is not a process-RSS or
allocator limit, and `0` (the default) means unlimited. A queue always retains at
least one domain, so a single oversized domain may exceed the budget. If
enforcement evicts any unverified domain the verdict is `unknown`, never a proof
from an incomplete search. Grouped-disjunctive `DomainList` storage rejects a
nonzero byte cap until its row sidecar has a complete census.

## GPU

The default build includes a cross-platform wgpu compute backend (Metal,
Vulkan, DX12):

```bash
ny beta-crown model.onnx -p prop.vnnlib --backend wgpu
```

Verdict-bearing WGPU authority is explicit and request-scoped. An ordinary
`WgpuDevice` is unarmed, and its ordinary `GemmEngine` methods (`gemm_f32`,
`conv_transpose_2d`, …) are typed refusals. `WgpuVerdictRequest::new()` is the
one typed public request: it creates a single adapter context, eagerly runs five
live qualification rungs, and produces a `WgpuVerdictReport` — retained on
success, preserved inside the typed error on refusal. Only success lets the
public `ComputeDevice` proof adapter expose CROWN; IBP, DAG, and every other
proof accessor stay closed. There is no ambient self-arm environment grant and
no CLI opt-out that lets an unqualified GPU bound decide a verdict.

`--backend wgpu` asks for that qualified route. If device construction or any
rung refuses, the CLI fails closed to the sound CPU implementation, emits the
unconditional `NY-HARNESS: BACKEND-OVERRIDE` marker, and publishes an
authenticated `ProofBackendReceipt` naming the request source, qualification or
failure detail, effective backend, fallback, adapter, and proof provenance (JSON
mode carries the same receipt). `--features cuda` adds a dynamically loaded
native CUDA/cuBLAS accelerator for supported non-WGPU routes.

Qualification is measured per adapter, never assumed. On the 2026-08-11 Apple
M5 Max/Metal run the ladder passed 3/5 (IEEE f32, host EFT, sentinel taint) but
failed EFT primitives and gradual underflow, so the request refused and the
hermetic proof A/B fell back to byte-identical CPU output. The measured
GB10/Vulkan + DenormPreserve ladder is 5/5, and on that adapter the raw
`WgpuDevice` CROWN gate is open, with an armed preflight that refuses absent or
tainted rows; unsupported configurations typed-refuse rather than drop word
state. The internal current-state note and WGPU Metal qualification record are
the canonical ledger for both.

Two narrow seams sit outside that authority. `FlValueGemmDevice` computes only
forward-linear value products under the `NY_FORWARD_LINEAR_F32` treatment: the
caller charges the complete f32 accumulation/FTZ error separately, certified
error-base products stay on CPU f64, and non-finite or sentinel results fall
back; it can never act as an ordinary `GemmEngine` or CROWN backend. And Complete
Clip treats a device result as a mere suggestion until private host replay
reproduces it in the exact clip context and mints a context-bound
affine-provenance token that device code cannot construct.

Falsification is exempt and stays accelerated. Attack steering is
verdict-neutral by construction — its floats only choose where to look next, and
every candidate still passes the unchanged admission gates — so it keeps
whichever accelerator the host has (CUDA on an NVIDIA box, WGPU otherwise) even
when the proof lane is forced to CPU.

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
`assert_equivalent`, `assert_bounds`, fixtures, and `--ny-*` options. Fully
typed (`py.typed` + `.pyi` stubs).

## Results

**NY has never been an official VNN-COMP entrant.** Everything below is local,
internally reproducible measurement — not an official result, an organizer
certification, or a third-party attestation.
[`reports/measured/README.md`](reports/measured/README.md) carries the immutable
rerun, witness-replay, completion-gated promotion, and `--require-evidence`
scorecard protocol.

**2025 corpus.** `reports/measured/` is the bank. Rows carry a seventh
provenance column: a run tag means the row was replayed at its official
per-instance budget on a recorded build, while `inherited-unverified` means
nobody has re-run it since it was banked — a claim, not a measurement. Four
legacy files (`cgan_2023`, `dist_shift_2023`, `nn4sys`, `sat_relu`) remain
six-column and untagged. Twenty-eight rows across six categories are bound to
immutable external run artifacts by
`reports/measured/regular_evidence_index.json`, and a read-only validator
reopens their manifests, hashes, official occurrence, budget, and published truth
before the scorecard may credit them. The sealed `sat_relu` file names a 100/100
local run whose manifest/log/input/witness bundle is not in this checkout, so
that CSV alone is neither revalidated evidence nor a rank. The latest checked-in
aggregate ledger here is the 2026-08-11 snapshot at `0cb3005a` in
`configs/preset_score_at_risk.yaml`, and that file now carries a STALE banner
over two of its three totals. Only extended-9 is current: **482.8**
(corrected 2026-08-27, was 499.6). Regular-16 **1406.8** is marked stale and
drifting upward as banks convert, and full-field-25 **1906.4** is stale by
construction — 1406.8 + 482.8 is 1889.6. Do not quote those two as current;
no refreshed aggregate with its provenance block is checked in.

**2026 corpus.** Banks live in `reports/measured-2026/` (internal; not part of
the public snapshot), where a family counts as banked only when every row of its
`instances.csv` ran to a protocol verdict token at the official per-instance
budget. Its generated coverage census is the index of record and currently
reports complete banks for **19 of 30** scored categories. No refreshed
normalized total for that census is checked in — the most recent board rollup
predates three of those banks — so no current 2026 score should be quoted from
this tree.

Two proof-level results stand on their own artifacts rather than a scoreboard.
The checked-in historical `sat_relu` proof corpus LRAT/Lean-replays all 49 unique
UNSAT instances of its recorded sweep (the benchmark's `instances.csv` lists 50
UNSAT rows, one duplicated); the runtime path does not invoke Lean per request —
it independently regenerates an in-memory AY ResolutionDag and RUP-replays it
before admitting `verified`. And on `cersyve`, 5 of 9 finetune control systems
are certified safe for an unbounded horizon by a kernel-checked induction theorem
over `ny`'s one-step `con`/`inv` verdicts.

## Testing and proofs

`ny` is covered by thousands of Rust tests, property-based tests, and four fuzz
targets, one of which asserts that computed bounds enclose concrete executions.
Kani harnesses verify the `ny-relaxation` scalar-relaxation mirror
(`proofs/kani/`), and differential tests in `crates/ny-relaxation/tests/drift.rs`
bind those mirrors to the production `ny-propagate` paths over an adversarial
grid — every mirrored pair except the centered-normalization helpers, which
production inlines and which therefore remain unbound. Lean proofs cover the
certificate lemmas (`crates/ny-cert/proofs/lean/`; see
[`crates/ny-cert/proofs/README.md`](crates/ny-cert/proofs/README.md) for what is
proved and the one bridge lemma still taken as an axiom). Model-checked TLA+
specifications (`specs/tla/`) cover the documented CNF/cell/MIP/BaB
verdict-admission abstraction and the environment-lock contract; direct
relational, TLL, fractional-head, and post-BaB margin-row lanes are outside the
admission model.

Cargo is the canonical correctness entry point; the complete hermetic lane is:

```bash
cargo check --workspace --exclude ny-python
cargo test --locked --workspace --exclude ny-python
```

That suite is hermetic and contains no ignored tests — `#[ignore]`,
`pytest.skip`, and their relatives are prohibited in first-party source and
enforced by a lexical scanner. Python-binding, real-corpus, external-checker,
GPU, and measurement lanes are explicit and fail fast when selected without their
prerequisites; the internal `TEST_CONFORMANCE.md` note (not part of the public
snapshot) has the commands and policy. There is deliberately no hosted CI, and
the toolchain is pinned by `rust-toolchain.toml`.

## Workspace

| Layer | Crates |
| --- | --- |
| Core types | `ny-core`, `ny-tensor` |
| Runtime contracts and configuration | `ny-contracts`, `ny-levers` |
| Model ingestion | `ny-load`, `ny-build`, `ny-onnx`, `ny-trace-bridge` |
| Verification engines | `ny-propagate`, `ny-gpu`, `ny-gpu-passthrough`, `ny-cuda`, `ny-accelerate`, `ny-mip` |
| Falsification | `ny-falsify` |
| Ground truth | `ny-groundtruth` |
| Certificates | `ny-cert`, `ny-relaxation` |
| Evidence ledger | `ny-claims` |
| Interfaces | `ny-cli`, `ny-api`, `ny-python` |
| Dev tooling | `ny-test-utils`, `ny-lint-guard` |

## Developing and contributing

Contribution basics (build, test, licensing, reporting) are in
[`CONTRIBUTING.md`](CONTRIBUTING.md). See [`SECURITY.md`](SECURITY.md) for
private vulnerability reports and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for
community expectations.

## License

Apache License 2.0. See [`LICENSE`](LICENSE). `ny` was created by Andrew Yates.
