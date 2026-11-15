# GPU cost-calibration profiles

A cost-calibration profile gives conservative throughput floors for one GPU:
per op family, the minimum effective FLOPs/ns and bytes/ns and a launch
overhead, plus a workspace slack in bytes. `estimate_model_timing`
(`crates/ny-onnx/src/cost/timing.rs`) combines a profile with the static cost
estimate into latency and peak-memory bounds; from the CLI, run
`ny inspect <model.onnx> --cost --timing-profile <profile.json>`.

`m4-max-metal-conservative.json` is the Apple M4 Max (Metal) profile loaded by
the `ny-onnx` cost tests (`crates/ny-onnx/src/cost/tests.rs`). To adapt it for
another device, measure sustained per-family throughput and set each floor
below what the hardware always achieves. The tests check the profile's fields
and that every rate is positive and finite; nothing checks that a floor is
conservative.
