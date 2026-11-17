<div align="center">
  <h1>ny</h1>
  <p><strong>Prove neural-network properties. Find counterexamples. Check certificates.</strong></p>
  <p>
    <a href="#quick-start">Quick start</a> ·
    <a href="#checking-the-answers">Checking the answers</a> ·
    <a href="#commands">Commands</a> ·
    <a href="#vnn-comp">VNN-COMP</a> ·
    <a href="#methods">Methods</a> ·
    <a href="#gpu">GPU</a> ·
    <a href="#python">Python</a> ·
    <a href="https://github.com/alabsystems/ny/issues">Issues</a>
  </p>
</div>

`ny` is a neural-network verifier written in Rust. Give it a model and a
property, and it answers with one of:

| Answer | Meaning | Evidence |
| --- | --- | --- |
| **verified** | The property holds on the whole input region. | An optional exact-rational certificate on eligible networks. |
| **falsified** | The property can be violated. | A concrete input, re-run through the model to confirm the violation. |
| **unknown** | `ny` could neither prove nor refute it. | The reason, such as an unsupported operator or an exhausted search budget. |
| **timeout** | The time limit ran out first. | Elapsed time and partial progress, where available. |

It implements interval bound propagation (IBP), CROWN, α-CROWN, and β-CROWN
branch-and-bound, plus a PGD attack for finding counterexamples. Arithmetic on
the verdict path carries certified rounding-error bounds.

- **Models:** ONNX, NNet, PyTorch, SafeTensors, CoreML, GGUF.
- **Properties:** VNN-LIB 1.0 and a subset of VNN-LIB 2.0.
- **Solvers:** first-party [`ay`](https://github.com/alabsystems/ay) engines
  handle LP, MIP, SMT and SAT; NY implements bound propagation natively.
  No third-party solver sits on a verdict path.

> **Status:** pre-1.0 research software. Interfaces may change without notice.

## Quick start

`ny` needs Git, a C/C++ build toolchain, and a current Trust installation.
`rust-toolchain.toml` selects the local `trust` alias.
[aterm](https://github.com/alabsystems/aterm#install) manages Trust with
`aterm pkg install trust`. Run `aterm help rust` in the checkout to check
which compiler this host uses.

```bash
git clone https://github.com/alabsystems/ny
cd ny
targo --unverified build --locked --release -p ny-cli    # → target/release/ny
export PATH="$PWD/target/release:$PATH"
```

`targo --unverified` builds without a compiler proof. For competition
measurements, enable `--features mip`; add `cuda` for the CUDA lane. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for build profiles and Linux requirements.

In Git checkouts, `ny --version` uses the manifest's major/minor and the commit
count as its patch, with a suffix for tracked build-relevant edits. Counts can
coincide across branches; `ny --build-id` reports the commit, build stamp, and
executable SHA-256 for reproducible measurements.

`tests/models/crossing_relu.nnet` computes `f(x) = |x|`. Prove that its output
never reaches 2 for any input in `[-1, 1]`:

```bash
ny beta-crown tests/models/crossing_relu.nnet -p tests/models/crossing_relu_safe.vnnlib
```

```
--- Result ---
Status: VERIFIED
All inputs produce output < 2
```

Now ask whether the output can reach 0.5. `ny` returns a counterexample it has
re-run through the model (the input may vary):

```bash
ny beta-crown tests/models/crossing_relu.nnet -p tests/models/crossing_relu_unsafe.vnnlib
```

```
--- Result ---
Status: VIOLATED
Found counterexample where output >= 0.5
Counterexample input: [0.50806665]
```

`verify` and `beta-crown` report the answer in their exit code:

| Exit code | Meaning |
| --- | --- |
| 0 | verified |
| 1 | falsified |
| 2 | unknown |
| 3 | timeout |
| 4 | invalid invocation or operational error |

For CI, `ny verify --allow-unknown` maps ordinary `unknown` results to 0.
A refused counterexample still exits 2; timeouts and errors keep their codes.

New to verification? `ny tutorial` offers short lessons, and
`ny tutorial demo` runs the example above.

## Checking the answers

**Counterexamples** are re-executed through the model before `ny` reports them.

**Certificates** are optional for sequential, fully connected ReLU networks
with up to 8192 hidden neurons. `ny` attempts an exact-rational CROWN proof
and self-checks its entailment and Farkas certificates before writing them:

```bash
ny beta-crown tests/models/crossing_relu.nnet \
  -p tests/models/crossing_relu_safe.vnnlib \
  --emit-certificate crossing_relu.cert.json
```

A certificate contains only exact rationals and linear premises, with no
floating point and no solver state. The external kernel checker
[Clean](https://github.com/alabsystems/clean) replays it end to end. Lean
proofs of the certificate lemmas live in
[`crates/ny-cert/proofs/`](crates/ny-cert/proofs/README.md), whose README
states what is proved and what is assumed.

By default `beta-crown` attempts `<model-stem>.cert.json` next to the model.
Emission is best effort: unsupported properties, inconclusive derivation,
budget limits, or write failures can leave a verified verdict without a
sidecar. `--no-certificate` disables optional emission without changing the
verdict; `--require-certificate` takes precedence and restricts verdicts to
proofs the external format can represent.
`ny vnncomp` runs under the `competition` proof profile, which skips the
certificate and samples downgrade-only audits less often; the certifiers that
decide a verdict run identically in both profiles.

## Commands

| Command | Purpose |
| --- | --- |
| `ny beta-crown MODEL -p PROP` | Complete verification by β-CROWN branch-and-bound. |
| `ny verify MODEL -p PROP --method M` | Check with `ibp`, `crown`, `alpha` (default), or `beta` (`.nnet` only); use `beta-crown` for ONNX graphs. |
| `ny vnncomp v1 CATEGORY MODEL PROP OUT TIMEOUT` | Run one VNN-COMP instance under the competition protocol. |
| `ny vnncomp plan CATEGORY MODEL PROP TIMEOUT` | Print the resolved run plan without running it. |
| `ny benchmarks …` | Download, inventory, sweep, and score VNN-COMP corpora. |
| `ny coverage` | List the trace bridge's operator soundness taxonomy. |
| `ny claims verify` | Re-check recorded empirical claims against their pinned artifacts. |
| `ny proof-check …` | Independently audit a proof bundle; its output never changes a solver verdict. |
| `ny inspect MODEL` | Show model structure. |
| `ny lipschitz MODEL` | Compute a certified global Lipschitz bound. |
| `ny diff A B` | Find where two models diverge (useful when porting). |
| `ny compare A B` | Compare two models' propagated output bounds. |
| `ny sensitivity MODEL` | Find which layers amplify input noise. |
| `ny quantize-check MODEL` | Check float16 and int8 quantization safety. |
| `ny profile-bounds MODEL` | Show how bound width grows through the network. |
| `ny weights info --file MODEL` | Inspect stored weights. |
| `ny gt verify …` | Check geometric ground-truth properties. |
| `ny export --size SIZE --output FILE` | Generate a Python script that exports a Whisper encoder to ONNX. |
| `ny tutorial` | Short verification lessons and a runnable demo. |

`ny --help` and `ny <command> --help` describe commands and public options.

`ny coverage` classifies trace operators as `exact`, `sound`,
`sound_but_loose`, or `unsupported`. These describe possible relaxations;
classified operators can still lack an implemented translation and be refused.

## VNN-COMP

`ny vnncomp v1` runs one instance exactly as the competition harness does and
writes the result file; [`vnncomp_scripts/`](vnncomp_scripts/README.md) holds
the harness scripts. `ny benchmarks` works on whole corpora:

```bash
ny benchmarks download 2025 2026                    # fetch corpora (large)
ny benchmarks status --year 2026 --strict           # check that assets are present
ny benchmarks run --year 2026 --category CATEGORY \
  --vnnlib-version 2.0 --output reports/sweeps/run.csv # add --resume to continue
ny benchmarks score --results BANK_DIR --official RESULTS_CHECKOUT --year 2026 \
  --budget-year 2026 --ny-ce-checks REPLAY_CHECKS --ny-ce-source SOURCE_COMMIT
ny benchmarks reseed --year 2026 --category CATEGORY --vnnlib-version 2.0 --limit 10
```

- **`run`** runs each instance in its own process at the official
  per-instance budget from `instances.csv`. It writes a CSV in the official
  column order, a `.metadata.jsonl` row log, and a `.manifest.json` that pins
  the build, configuration, and inputs; `--resume` refuses a bank whose
  provenance has changed. `--timeout-cap` can only lower budgets, so a capped
  sweep is a lower bound, not a result. Where a category ships both `1.0/` and
  `2.0/` instance lists, pass `--vnnlib-version`; the runner will not guess.
- **`score`** compares a bank with a baseline. With `--official` it models the
  VNN-COMP rules (+10 per correct verdict, −150 per incorrect one, no time
  bonus, normalized to each category's winner), and `--witnesses` chooses how
  disputed counterexamples count. It is a local model of the rules, not an
  official score. For 2026, adjudicated mode uses published per-run checks;
  strict SAT witnesses can penalize UNSAT, while tolerant acceptance credits
  only its own SAT. NY SAT rows need source-bound `--ny-ce-checks`; missing
  checks remain unqualified and receive zero. The NY replay reader and
  [exporter](scripts/export_cct_replay_checks.py) currently accept CCT pilot
  receipts only; see the exporter's `--help` for inputs.
  With `--year 2026 --budget-year 2026`, field timeout checks subtract each
  tool's fastest decided solve from its runtimes, using the complete
  `TOOL/results.csv` including harness rows, as in the
  [published processor](https://github.com/VNN-COMP/vnncomp2026_results/blob/eea564f5d0a8751d8321a409214db14d03f452b8/SCORING/process_results.py).
  NY keeps its strict original wall-budget check. Rule modeling does not
  establish corpus identity or cross-version equivalence. The CSV reader rejects
  published `-1` preparation-failure runtimes, so the complete official 2026
  checkout still fails before scoring.
- **`reseed`** re-runs instances under renamed copies of their files and
  checks that the verdicts match.

`scripts/canonicalize_vnncomp_ce_y.py` recomputes a counterexample's output
values under a pinned ONNX Runtime, preserving every input numeric token, and
refuses the result if the witness no longer violates the property at zero
tolerance.

`ny` has not been an official VNN-COMP entrant. The
[2025 result bank](reports/measured/README.md) documents local measurements at
the official time limits, replay, and scoring. These results are not official
or independently attested.

The November 17 head-confirmation cache update
passed 35 selected tests and a CUDA CLI check. Its matched
Tiny177 pilot
produced four accepted timeouts. The candidate's diagnostic run completed 86 of
199 confirmation rows; head composition remained unfinished. No extra solve or
statistical end-to-end speedup is established.

## Methods

For SAT-encodable ReLU networks, `ay-sat` solves a bit-exact CNF encoding
in-process. Witnesses are replayed through the network; UNSAT requires a
re-checked resolution proof.

`ny gt verify --escalate smt` can escalate to exact SMT through `ay` (resolved
via `$NY_AY`, `$PATH`, then the linked `trust` toolchain). SAT witnesses are
checked in exact rational arithmetic; emitted Alethe proofs can be replayed
externally.

**VNN-LIB.** `ny` supports VNN-LIB 1.0 scalar declarations and affine
constraints, and part of VNN-LIB 2.0: tensor declarations, indexed variables,
network wrappers, and relational forms. Unsupported syntax is rejected, never
guessed at. Strict input-domain constraints are rejected, because the input
region is a closed box; outside the nonlinear fallback below, the competition
entry point returns `unknown` for them. VNN-LIB 2.0 properties with nonlinear
arithmetic get a narrow fallback: `sat` needs the full formula, strict atoms
included, to hold at a concrete input replayed through ONNX Runtime, `unsat` is
limited to at most four bounded scalar inputs, and everything else is
`unknown`.

**Transformers.** `ny` loads and inspects Whisper and decoder models and
extracts their components, but it does not yet verify them: those requests
fail closed instead of returning unproven bounds. To export a Whisper encoder
to ONNX:

```bash
python -m pip install torch openai-whisper onnx onnxscript
ny export --size tiny --output export_whisper.py   # tiny, base, small, medium, or large
python export_whisper.py
```

Keep the `.data` weight file the script writes next to the `.onnx` file.

**Optional features.**

- `--features mip` adds the LP/MIP lane (`ny-mip`, running `ay-milp`
  in-process).
- `--features cuda` adds a dynamically loaded CUDA/cuBLAS accelerator.
- `--max-queue-bytes N` (preset key `bab.max_queue_bytes`) budgets estimated
  graph branch-and-bound queue payload, not total process memory. Zero disables
  it; the highest-priority domain is retained even if it exceeds the budget.
  Evicting an unexplored domain prevents a `verified` result.

## GPU

The default build includes a wgpu backend (Metal, Vulkan, DX12):

```bash
ny beta-crown model.onnx -p prop.vnnlib --backend wgpu
```

For proof work, wgpu first qualifies the adapter's f32 arithmetic, error-free
transformations, and subnormal preservation. If that path fails, it tries a
qualified path that bounds subnormal-flush error. If neither qualifies, `ny`
uses the CPU, prints `NY-HARNESS: BACKEND-OVERRIDE`, and records why. There is
no switch to skip qualification.

Counterexample search can use GPU arithmetic without proof qualification to
propose inputs. Backend requests and availability still apply, and every
candidate passes the same CPU checks.

With `--features cuda`, admitted compact or near-dense root convolutions can use
an independently qualified CUDA engine. The competition entry point prewarms it
before instance deadlines; finite-deadline calls use only an existing engine.
Unsupported devices keep CPU contraction. `NY_NO_CUDA=1` disables this path.
See the [arithmetic and failure contract](crates/ny-cuda/kernels/packed_root_README.txt).

## Python

General benchmark tooling uses the packages in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make PYTHON="$VIRTUAL_ENV/bin/python" test-python-tooling
```

Competition counterexample replay additionally requires the
[replay script's pinned checker environment](scripts/replay_ny_counterexamples.py).

If `venv` is unavailable (Debian and Ubuntu ship it separately), use
`uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`.

The Python bindings (Python 3.9 or newer) build with maturin:

```bash
python -m pip install maturin
cd crates/ny-python
maturin develop --release
```

The `ny` module provides `verify()` and `verify_torch()` (IBP, CROWN, α- and
β-CROWN on ONNX files or in-memory `torch.nn.Module`s). Pass
`output_bounds=[(lower, upper), ...]` to state a property; omitting it computes
bounds without checking a property. It also provides model diffing,
equivalence and quantization checks, bound profiling, type stubs, and the
`ny_pytest` plugin with `assert_verified`, `assert_equivalent`, and `assert_bounds`.

## Testing

```bash
targo --unverified check --workspace --exclude ny-python
targo --unverified test --locked --workspace --exclude ny-python
```

The default suite needs no benchmark downloads. Two legacy ACAS controls
return early when the optional 2023 corpus is absent, providing no coverage.
Explicit ignore and skip directives are rejected by a source-policy test.
The workspace test command includes real GPU controls: `ny-cli` enables
`ny-gpu/gpu-tests` through a dev dependency, so it needs a suitable adapter.
Python-binding, benchmark-corpus and external-checker lanes have separate
prerequisites; see `TEST_CONFORMANCE.md`.

Beyond unit and property tests:

- four fuzz targets in [`fuzz/`](fuzz), one of which checks that computed
  bounds contain concrete executions;
- the Lean proofs in [`crates/ny-cert/proofs/`](crates/ny-cert/proofs/README.md);
- TLA+ models of verdict admission in [`specs/tla/`](specs/tla).

## Workspace

| Layer | Crates |
| --- | --- |
| Core types | `ny-core`, `ny-tensor` |
| Contract annotations and configuration | `ny-contracts`, `ny-levers` |
| Model ingestion | `ny-load`, `ny-build`, `ny-onnx`, `ny-trace-bridge` |
| Verification engines | `ny-propagate`, `ny-gpu`, `ny-gpu-passthrough`, `ny-cuda`, `ny-accelerate`, `ny-mip`, `ny-bbe` |
| Falsification | `ny-falsify` |
| Ground truth | `ny-groundtruth` |
| Certificates and offline proof checking | `ny-cert`, `ny-proofcheck` |
| Evidence ledger | `ny-claims` |
| Interfaces | `ny-cli`, `ny-api`, `ny-python` |
| Development tooling | `ny-test-utils`, `ny-lint-guard` |

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for building, testing, and licensing,
[`SECURITY.md`](SECURITY.md) for reporting vulnerabilities, and
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community expectations.

## License

Apache License 2.0; see [`LICENSE`](LICENSE). `ny` was created by Andrew Yates.
