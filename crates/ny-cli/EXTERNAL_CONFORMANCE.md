# ny-cli external conformance

The default test suite is hermetic. Tests that authenticate downloaded models,
official properties, or an external solver live in explicit feature lanes. A
selected lane treats missing inputs as a failure; it never reports success
after checking nothing.

## VNN-COMP corpus lane

Run the complete CLI corpus lane with the MIP-only qualifications enabled:

```sh
cargo test -p ny-cli --features 'mip external-vnncomp'
```

Download the VNN-COMP benchmark repositories first. The cGAN tests use
`benchmarks/vnncomp2025/benchmarks/cgan_2023` by default; set
`NY_CGAN_NCH1_ROOT` only when that category is staged elsewhere. The preset
load guard needs every benchmark root referenced by a shipped preset. Set
`NY_PRESET_LOAD_SMOKE=all` to load every model instead of one representative
per architecture family; there is intentionally no runtime-off setting.

The TLL source-authentication qualifications require explicit authored inputs:

- `NY_TLL_IDENTITY_FIXTURE=/path/to/model.onnx` (or `.onnx.gz`)
- `NY_TLL_PROPERTY_FIXTURE_ROOT=/path/to/vnnlib-directory`
- `NY_TLL_END_TO_END_MODEL=/path/to/model.onnx`
- `NY_TLL_END_TO_END_PROPERTY=/path/to/property.vnnlib`

Graph-MIP solve-regression tests require a certified result. A sound `None`
fallback is valid production behavior but is not a passing solve test. To
collect a per-clause table where timeouts are useful data, use the explicit
research command instead:

```sh
cargo run -p ny-cli --release --features mip -- \
  vnncomp research graph-mip mscn-per-clause-128d \
  --bench-dir /path/to/nn4sys/1.0
```

`NY_GRAPH_MIP_TEST_BUDGET` sets that harness's per-clause budget in seconds.

## External ay lane

```sh
cargo test -p ny-cli --features external-ay
```

This lane requires the revision-pinned `ay` executable through `NY_AY` or
`PATH`. Missing or mismatched solver prerequisites are failures.
