<!--
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
SPDX-License-Identifier: Apache-2.0
-->

# Issue `#4291` Metaroom Host Profile Current Report

**Issue:** #4291  
**Epic:** #4258  
**Date:** 2026-03-21

## Summary

The direct row-10 WGPU profile timed out after `221.70s` with `0` explored domains.

## Commands

**Profile wrapper invocation:**

```bash
scripts/profile_vnncomp_row.sh --backend wgpu --timeout 210
```

## Artifacts

- **CSV artifact:** `reports/benchmarks/issue-4291-metaroom-host-profile_20260321_032249.csv`

## Row Identity

- `schema_version`: `backend_benchmark_row_v1`
- `lane`: `metaroom_host_profile`
- `subject_kind`: `vnncomp_instance`
- `subject_id`: `metaroom_2023::model.onnx::spec.vnnlib`
- `comparison_key`: `metaroom_2023::model.onnx::spec.vnnlib`
- `model_path`: `benchmarks/model.onnx`
- `property_path`: `benchmarks/spec.vnnlib`
- `preset_path`: `configs/preset.yaml`
- `backend`: `wgpu`
- `timeout_seconds`: `210`
- `status`: `timeout`
- `actual_method`: `empty (no final method line before timeout)`
- `wall_seconds`: `221.70`
- `domains_explored`: `0`
- `profile_artifact_path`: `reports/benchmarks/issue-4291-metaroom-host-profile-current.md`

## Profile Findings

Final verifier facts:

- `status`: `timeout`
- `wall_seconds`: `221.70`
- `domains_explored`: `0`
- `actual_method`: not emitted

Early sample dominant stack families:

- `try_disjunctive_sampling_attack` -> `evaluate_model`
- `propagate_ibp_core`

Late sample dominant stack families:

- `compute_graph_bab_bootstrap`
- `propagate_crown_to_node_core`

Hotspot-to-issue mapping:

- Late-sample bootstrap work starts with `#2237`'s remaining scratch.

## Divergence Gate

No backend-only divergence observed.

## Verdict

Pre-BaB warmup timeout.
