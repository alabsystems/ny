# ny-gpu

`ny-gpu` holds accelerated bound propagation: Rayon CPU kernels and, under the
default `wgpu` feature, WebGPU compute (CROWN backward, IBP forward) plus GPU
benchmark support. `ny-cli` builds it with default features; `ny-onnx` and
`ny-python` disable them and forward their own `wgpu` feature.

Keep device setup, shader dispatch, and GPU benchmark support inside this crate.
Higher-level verification policy belongs in `ny-propagate` or `ny-cli`.

Real-device tests compile only under `gpu-tests`:

```sh
targo --unverified test -p ny-gpu --features gpu-tests
```

The tests whose subject is uncharged verdict authority need an adapter that
keeps f32 subnormals and exact error-free transforms (Vulkan with the SPIR-V
DenormPreserve injection); they are the `gpu-tests-uncharged` lane, which
implies `gpu-tests`. On a pure-flush adapter such as Apple/Metal that lane
fails each test naming the refused rung (see `TEST_CONFORMANCE.md`):

```sh
targo --unverified test -p ny-gpu --features gpu-tests-uncharged
```
