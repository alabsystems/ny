# Benchmarks

Benchmark scripts exercise the `ny` CLI against small public fixtures and
VNN-COMP-style workloads. Keep benchmark output out of git; commit only scripts,
input manifests, and compact fixtures needed for regression coverage.

Build the CLI first:

```bash
cargo build --release -p ny-cli
```

Then run a scoped benchmark script, for example:

```bash
python benchmarks/test_crown_simple.py
```

The fixed alpha-beta-CROWN transfer corpus and its immutable M0 capture
workflow are documented in
`docs/ABCROWN_TRANSFER_BASELINE_M0.md` (internal docs, not part of this snapshot).
