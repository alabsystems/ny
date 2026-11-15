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
| **verified** | The property holds on the whole input region. | On eligible networks, an exact-rational certificate that an external checker can replay. |
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

`ny` needs Git, a C/C++ build toolchain, and the Trust toolchain named in
`rust-toolchain.toml`. [aterm](https://github.com/alabsystems/aterm#install)
manages Trust with `aterm pkg install trust`. Run `aterm help rust` in the
checkout to check which compiler this host uses.

```bash
git clone https://github.com/alabsystems/ny
cd ny
targo --unverified build --locked --release -p ny-cli    # → target/release/ny
export PATH="$PWD/target/release:$PATH"
```

`targo --unverified` builds without a compiler proof. For competition
measurements, enable `--features mip`; add `cuda` for the CUDA lane. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for build profiles and Linux requirements.

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

Now ask whether the output can reach 0.5. It can, so `ny` returns a
counterexample, having first re-run it through the model:

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

For CI, `ny verify --allow-unknown` maps `unknown` to 0. It never masks a
timeout or an error.

New to verification? `ny tutorial` is an interactive introduction, and
`ny tutorial demo` runs the example above.

## Checking the answers

**Counterexamples** are re-executed through the model before `ny` reports
them, and you can replay them in any inference engine.

**Proofs** come with a certificate on eligible networks: sequential,
fully connected ReLU networks with up to 8192 hidden neurons. `ny` re-derives
the CROWN bound in exact rational arithmetic, writes it as an entailment plus
a Farkas certificate, and self-checks the certificate before writing it:

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

On eligible networks `beta-crown` writes `<model-stem>.cert.json` next to the
model by default. Pass `--no-certificate` to skip it. The certificate is
derived after the verdict, so skipping it never changes the verdict.
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
| `ny tutorial` | Interactive introduction to verification. |

`ny --help` and `ny <command> --help` list every flag.

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
  --output reports/sweeps/run.csv                   # add --resume to continue
ny benchmarks score --results BANK_DIR --official RESULTS_CHECKOUT --year 2026
ny benchmarks reseed --year 2026 --category CATEGORY --limit 10
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
  official score.
- **`reseed`** re-runs instances under renamed copies of their files and
  checks that the verdicts match.

`scripts/canonicalize_vnncomp_ce_y.py` recomputes a counterexample's output
values under a pinned ONNX Runtime, keeping its inputs byte for byte, and
refuses the result if the witness no longer violates the property at zero
tolerance.

`ny` has not been an official VNN-COMP entrant. It is measured locally on past
VNN-COMP benchmarks at the official time limits. The result bank, its
provenance rules, and the scripts that re-run and score it are described in
[`reports/measured/`](reports/measured/README.md). These are local
measurements, not official or independently attested results.

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
fail closed instead of returning unproven bounds. `ny export` writes a Python
script that exports a Whisper encoder to ONNX:

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
- `--max-queue-bytes N` (preset key `bab.max_queue_bytes`) caps the memory of
  the branch-and-bound queue. If the cap forces an unexplored domain to be
  dropped, the answer is `unknown`, never `verified`.

## GPU

The default build includes a wgpu backend (Metal, Vulkan, DX12):

```bash
ny beta-crown model.onnx -p prop.vnnlib --backend wgpu
```

A GPU contributes to a proof only after it passes qualification checks on
that adapter, covering IEEE f32 semantics, error-free transformations, and
subnormal handling. If any check fails, `ny` falls back to the CPU, prints
`NY-HARNESS: BACKEND-OVERRIDE`, and records why. There is no switch to skip
qualification.

Counterexample search is exempt. GPU arithmetic there only steers where to
look, and every candidate still passes the same CPU checks, so the search uses
whatever accelerator the host has.

## Python

The benchmark and replay scripts need the packages in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make PYTHON="$VIRTUAL_ENV/bin/python" test-python-tooling
```

If `venv` is unavailable (Debian and Ubuntu ship it separately), use
`uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`.

The Python bindings (Python 3.9 or newer) build with maturin:

```bash
python -m pip install maturin
cd crates/ny-python
maturin develop --release
```

The `ny` module provides `verify()` and `verify_torch()` (IBP, CROWN, α- and
β-CROWN on ONNX files or in-memory `torch.nn.Module`s), model diffing and
equivalence checks, quantization checks, bound profiling, and a pytest plugin,
`ny_pytest`, with `assert_verified`, `assert_equivalent`, and `assert_bounds`.
It ships type stubs.

## Testing

```bash
targo --unverified check --workspace --exclude ny-python
targo --unverified test --locked --workspace --exclude ny-python
```

The suite is hermetic and must not contain ignored or skipped tests; a
source-policy test rejects them. GPU, Python-binding, real-benchmark, and
external-checker tests are separate opt-in lanes that fail fast when their
prerequisites are missing.

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
