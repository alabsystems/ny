<div align="center">
  <h1>ny</h1>
  <p><strong>Prove neural-network properties. Find counterexamples. Check certificates.</strong></p>
  <p>A neural-network verifier in Rust. Given a model and a property, <code>ny</code>
  returns a proof, a concrete counterexample, or <code>unknown</code> — and on
  eligible proofs emits an exact-rational certificate an external checker can replay.</p>
  <p>
    <a href="#quick-start">Quick start</a> ·
    <a href="#check-nys-answers-without-trusting-ny">Check the answers</a> ·
    <a href="#use-ny">Use ny</a> ·
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
| **unknown** | `ny` could not prove or refute it within its methods or limits. | The reason (timeout, unsupported operator, exhausted domain budget). |

It implements interval bound propagation (IBP), CROWN, α-CROWN, and β-CROWN
branch-and-bound, with a PGD attack for finding counterexamples. Arithmetic on
the verdict path carries certified rounding-error bounds, and every
counterexample is validated by concrete re-execution before it is reported.

Models: ONNX, NNet, PyTorch, SafeTensors, CoreML, GGUF. Properties: VNN-LIB.
Written in Rust. Apache-2.0.

> **Status:** `ny` is pre-1.0 research software under active development. All
> solving happens on the first-party [`ay`](https://github.com/alabsystems/ay)
> engine — by repo policy, no third-party LP/MIP/SMT/SAT solver appears on any
> verdict path. The public surface is intentionally small while the internals
> stabilize; the measurement methodology and rerun protocol are in
> [`reports/measured/README.md`](reports/measured/README.md).

## Quick start

Build `ny` from source with a current stable Rust toolchain:

```bash
git clone https://github.com/alabsystems/ny
cd ny
cargo build --release -p ny-cli    # → target/release/ny
export PATH="$PWD/target/release:$PATH"
```

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

`ny` proves the bound and exits `0`. Now ask for a bound the network *can*
exceed — `|x|` reaches `0.5` inside the box, so the property is false:

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
timeout. `ny --help` lists every command.

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
ny verify model.onnx -p prop.vnnlib --method alpha # single-method bounds (ibp | crown | alpha | beta)
ny vnncomp v1 acasxu_2023 model.onnx prop.vnnlib out.txt 116   # one VNN-COMP instance, competition protocol
ny inspect model.onnx                              # model structure
ny lipschitz model.onnx                            # sound certified global Lipschitz bound
ny diff a.onnx b.onnx                              # where two models diverge (porting)
ny compare a.onnx b.onnx                           # equivalence between two models
ny sensitivity model.onnx                          # which layers amplify input noise
ny quantize-check model.onnx                       # float16 / int8 quantization safety
ny profile-bounds model.onnx                       # bound-width growth through the network
ny weights model.onnx                              # inspect and diff weights
ny gt verify spec.gt.json model.onnx               # verify against a geometric ground-truth spec
```

`ny verify` and `ny beta-crown` write their verdict to stdout and encode it in
the exit code; `ny vnncomp` writes the verdict to the competition results file.

## Methods and coverage

| Stage | What `ny` does | Evidence it produces |
| --- | --- | --- |
| Incomplete bounds | IBP, CROWN, α-CROWN linear bound propagation | Certified enclosures with rounding-error bounds |
| Complete verification | β-CROWN branch-and-bound (input and ReLU splitting) | `verified`, or a counterexample |
| Counterexamples | PGD attack, then concrete re-execution | The input, re-run through the model |
| Certificates | Exact-rational entailment + Farkas (eligible FC-ReLU nets) | `<model>.cert.json`, re-checkable by an external kernel |
| Escalation | Exact SMT via `ay` (`ny gt verify --escalate smt`) | An `ay` verdict on the escalated query |

The LP/MIP lane (`ny-mip`, in-process `ay-milp`) is available behind the `mip`
feature. The exact SMT escalation path runs on an `ay` binary located at
runtime through `$NY_AY` / `$PATH`.

## GPU

The default build includes a cross-platform wgpu compute backend (Metal,
Vulkan, DX12):

```bash
ny beta-crown model.onnx -p prop.vnnlib --backend wgpu
```

GPU bounds are computed as certified f32 enclosures, and every adapter passes an
IEEE-754 conformance self-check at startup. Building with `--features cuda` adds
a native cuBLAS engine that runs the f64 CROWN matrix products on NVIDIA GPUs,
loading CUDA dynamically at runtime.

## Python

```bash
cd crates/ny-python
maturin develop    # Python ≥ 3.9
```

The `ny` module provides `verify()` and `verify_torch()` (IBP/CROWN/α/β, on ONNX
files or in-memory `torch.nn.Module`s), model diffing and equivalence checks
across ONNX/SafeTensors/PyTorch/GGUF/CoreML, quantization-safety checks, bound
profiling, and a pytest plugin (`ny_pytest`) with `assert_verified`,
`assert_equivalent`, fixtures, and `--ny-*` options. Fully typed (`py.typed` +
`.pyi` stubs).

## Results

The checked-in VNN-COMP 2025 CSVs are legacy raw-verdict measurements without
retained SAT witnesses. They support regression analysis and explicitly
conditional score projections — not an official rank, and not a claim that every
verdict was organizer-validated. The audited projections, published incumbent
totals, and exact evidence gaps are recorded in an internal claims-of-record
audit; the immutable rerun and witness-replay protocol is documented in
[`reports/measured/README.md`](reports/measured/README.md). Earlier competition
plans in this tree are experiment notebooks, not current score evidence.

On `sat_relu`, all 50 UNSAT verdicts are additionally kernel-certified: the SAT
solver's LRAT refutations are replayed by a Lean proof checker. On `cersyve`, 5
of 9 control systems are certified safe for an unbounded horizon by a
kernel-checked induction theorem over `ny`'s one-step verdicts.

## Testing and proofs

`ny` is covered by ~11,800 Rust tests, property-based tests, and four fuzz
targets, one of which asserts that computed bounds enclose concrete executions.
Kani proof harnesses verify the `ny-relaxation` scalar-relaxation mirror
(`proofs/kani/`; binding to the production `ny-propagate` paths is in progress —
see the status note in `crates/ny-relaxation/src/lib.rs`), Lean proofs cover the
certificate lemmas (`crates/ny-cert/proofs/lean/`), and a model-checked TLA+
specification covers verdict admission (`specs/tla/`).

The default Rust workspace resolves and checks with:

```bash
cargo check --workspace --exclude ny-python
cargo test
make test-python                       # Python bindings
```

The toolchain is pinned by `rust-toolchain.toml`.

## Workspace

| Layer | Crates |
| --- | --- |
| Core types | `ny-core`, `ny-tensor` |
| Model ingestion | `ny-load`, `ny-build`, `ny-onnx`, `ny-trace-bridge` |
| Verification engines | `ny-propagate`, `ny-gpu`, `ny-cuda`, `ny-mip` |
| Ground truth | `ny-groundtruth` |
| Certificates | `ny-cert`, `ny-relaxation` |
| Interfaces | `ny-cli`, `ny-api`, `ny-python` |
| Dev tooling | `ny-test-utils`, `ny-lint-guard`, `ny-contracts` |

## Developing and contributing

Contribution basics (build, test, licensing, reporting) are in
[`CONTRIBUTING.md`](CONTRIBUTING.md). See [`SECURITY.md`](SECURITY.md) for
private vulnerability reports and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for
community expectations.

## License

Apache License 2.0. See [`LICENSE`](LICENSE). `ny` was created by Andrew Yates.
