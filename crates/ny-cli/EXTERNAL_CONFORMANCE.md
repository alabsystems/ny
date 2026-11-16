# ny-cli external conformance

The default test suite is hermetic. Tests that authenticate downloaded models,
official properties, or an external solver live in explicit feature lanes. A
selected lane treats missing inputs as a failure; it never reports success
after checking nothing.

## VNN-COMP corpus lane

Run the complete CLI corpus lane with the MIP-only qualifications enabled:

```sh
targo --unverified test --release -p ny-cli --features 'mip external-vnncomp'
```

Run it in release: the #mr-ibp-seed CNN7 real-ONNX gates
(`commands/margin_row_bab/ibp_seed_real_tests.rs`) build full root tableaux and
refuse a debug build as VOID, and the `proof_check_real` integration target
drives the release `ny` binary on banked rows (it also needs python3 with
onnxruntime and is meant to run under `scripts/ny_box.sh measure`). The lane
also carries `f64_tail_census` and the dist_shift real-model admission probe.
The prepared-source byte/path equivalence test requires the original
`benchmarks/vnncomp2025/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx`;
it checks the model's pinned SHA-256 before comparing both extraction paths.

Download the VNN-COMP benchmark repositories first
(`benchmarks/download_benchmarks.sh`). The cGAN tests use
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
targo --unverified run -p ny-cli --release --features mip -- \
  vnncomp-research graph-mip mscn-per-clause-128d \
  --bench-dir /path/to/nn4sys/1.0
```

`NY_GRAPH_MIP_TEST_BUDGET` sets that harness's per-clause budget in seconds
(default 10).

## External ay lane

```sh
targo --unverified test -p ny-cli --features external-ay
```

This lane requires the revision-pinned `ay` executable through `NY_AY` or
`PATH`. Missing or mismatched solver prerequisites are failures.
