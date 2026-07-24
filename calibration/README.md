# GPU cost-calibration profiles

A cost-calibration profile records conservative per-op-family throughput floors
for a specific GPU (minimum effective FLOPs/ns, bytes/ns, and launch overhead),
against which the analytic cost model is checked. `m4-max-metal-conservative.json`
is a conservative Apple M4 Max (Metal) envelope consumed by the `ny-onnx`
cost-model tests. To adapt it for another machine, measure your device's
sustained per-family throughput and set the floors safely below what the
hardware always achieves — the tests only require the profile to be a
conservative under-estimate.
