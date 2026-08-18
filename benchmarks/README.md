# Benchmarks

Benchmark scripts exercise the `ny` CLI against small public fixtures and
VNN-COMP-style workloads. Keep benchmark output out of git; commit only scripts,
input manifests, and compact fixtures needed for regression coverage.

Build the CLI first:

```bash
cargo build --release -p ny-cli
```

Correctness regressions belong in the Cargo-owned Rust test suite. The Python
files in this directory are explicitly selected measurement or cross-tool
diagnostics; pytest does not treat them as correctness coverage. A selected
diagnostic fails nonzero if its release binary, corpus, or third-party
comparator is unavailable.

Run a scoped diagnostic, for example:

```bash
python benchmarks/compare_crown_simple.py
```

The Auto-LiRPA ACAS-Xu and attention probes are likewise explicit:

```bash
python benchmarks/probe_autolirpa_acasxu.py
python benchmarks/compare_attention.py
```

The VNN-COMP corpus lanes are explicit diagnostics rather than default pytest
tests because their public corpora are downloaded separately. After running
`benchmarks/download_benchmarks.sh`, invoke them directly:

```bash
pytest benchmarks/acasxu_diagnostic.py -v --ny-benchmark-timeout=10
pytest benchmarks/vnncomp_diagnostic.py -v --ny-benchmark-method=beta
```

The fixed alpha-beta-CROWN transfer corpus and its immutable M0 capture
workflow are documented in
`docs/ABCROWN_TRANSFER_BASELINE_M0.md` (internal docs, not part of this snapshot).

The BICCOS multi-tree transfer planner validates the pinned alpha-beta-CROWN
and auto_LiRPA checkouts, the four sealed CIFAR100/TinyImageNet rows, and the
exact upstream factorial configs before emitting commands:

```bash
python3 scripts/plan_biccos_mts_factorial.py \
  --benchmark-root /absolute/path/to/vnncomp2025/benchmarks \
  --output /tmp/biccos-mts-plan.json
```

It is diagnostic-only: it has no execution mode and never starts a GPU job.

The compact-tail envelope planner seals the prop1761 Graph-MIP gap and emits
the B0/B1/K2/K4/K8/K16 experiment matrix for the
`Gemm_56 -> Relu_57 -> Gemm_58` tail:

```bash
python3 scripts/plan_compact_tail_envelope.py \
  --model /path/to/CIFAR100_resnet_medium.onnx \
  --property /path/to/CIFAR100_resnet_medium_prop_idx_1761_sidx_3933_eps_0.0039.vnnlib \
  --lpopt-dump /path/to/lpopt.dump \
  --root-margins /path/to/lpopt.dump.margins \
  --solver-log /path/to/solver.log
```

It requires exact sealed hashes for the model, property, and every evidence
file; checks that the log contains all 16 four-selector assignments; reproduces
the full/compact binary census; and exports conservative model/payload metrics.
Log coverage is diagnostic only; it is not proof authority. The planner has no
solver or execution mode, and malformed, mismatched, over-cap, or incomplete
evidence is rejected.

The separate compact K16 executor is dark unless both
`NY_IMB_TAIL_CERT_AY=1` and `NY_COMPACT_TAIL_K16=1` are exact-set. It accepts
only a live, opaque prefix-CROWN envelope produced inside `ny-propagate`; this
planner's JSON and log records cannot construct that type or authorize a
verdict. The current remaining gap is that live CIFAR100 prefix-bank producer.
