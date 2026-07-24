<!--
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
SPDX-License-Identifier: Apache-2.0
-->

# GPU Backend Delta: cersyve + metaroom_2023

**Issue:** #4282  
**Epic:** #4258  
**Date:** 2026-03-21

## Summary

Compare CPU vs WGPU backend on cersyve and metaroom_2023 categories.

## Commands

**Compare backends:**

```bash
python3 scripts/benchmark_vnncomp.sh --compare-backends --categories cersyve,metaroom_2023
```

## Artifacts

- **cersyve CSV:** `reports/benchmarks/cersyve_compare_backends.csv`
- **metaroom CSV:** `reports/benchmarks/metaroom_2023_compare_backends.csv`

## Row Identity

| Comparison Key | Category | Subject Kind | Backend | Status |
|---|---|---|---|---|
| cersyve::row=1::m.onnx::p.vnnlib | cersyve | vnncomp_instance | cpu | verified |
| cersyve::row=1::m.onnx::p.vnnlib | cersyve | vnncomp_instance | wgpu | verified |
| metaroom::row=1::n.onnx::q.vnnlib | metaroom_2023 | vnncomp_instance | cpu | verified |
| metaroom::row=1::n.onnx::q.vnnlib | metaroom_2023 | vnncomp_instance | wgpu | verified |

## Derived Comparison

### Category Overview

| Category | Samples | CPU Solved | WGPU Solved | CPU Wall | WGPU Wall | Delta | Speedup | Delta % | Divergences |
|---|---|---|---|---|---|---|---|---|---|
| cersyve | 1 | 1 | 1 | 1.20 | 0.80 | -0.40 | 1.50x | +33.3% | 0 |
| metaroom_2023 | 1 | 1 | 1 | 2.50 | 1.80 | -0.70 | 1.39x | +28.0% | 0 |

### Per-Instance Detail

| Comparison Key | CPU Status | CPU Seconds | CPU Domains | CPU Width | WGPU Status | WGPU Seconds | WGPU Domains | WGPU Width | Delta | Speedup | Delta % | Width Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cersyve::row=1::m.onnx::p.vnnlib | verified | 1.20 | 5 | 0.00 | verified | 0.80 | 5 | 0.00 | -0.40 | 1.50x | +33.3% | 1.00 |
| metaroom::row=1::n.onnx::q.vnnlib | verified | 2.50 | 10 | 0.01 | verified | 1.80 | 10 | 0.01 | -0.70 | 1.39x | +28.0% | 1.00 |

## Divergence Gate

No status divergences observed between CPU and WGPU backends.

### Observations

- WGPU shows modest speedup on both categories.
- Bound widths are identical across backends.

## Verdict

**PASS**: No divergences. WGPU backend is safe to use for these categories.
