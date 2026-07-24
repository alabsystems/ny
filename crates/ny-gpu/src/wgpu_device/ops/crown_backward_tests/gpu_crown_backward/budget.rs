// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::timing::{
    assert_results_close, build_metaroom_like_layers, build_soundnessbench_like_layers,
    run_manual_spec_batches,
};
use super::*;
use crate::wgpu_device::estimate_crown_backward_peak_bytes;
use crate::wgpu_device::ops::crown_memory_estimate::max_specs_per_budget;

// Blessed env-mutation choke point (clippy env wall): replaces the previous
// local ScopedEnvVar duplicate; these tests are serialized by
// `gpu_test_serial_guard`.
use ny_test_utils::env::ScopedEnvVar;

#[test]
fn test_gpu_crown_memory_budget_batching_matches_manual_chunks_3515() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    let (layers, input_dim, num_specs) = build_metaroom_like_layers();
    let full_estimate = estimate_crown_backward_peak_bytes(&layers, num_specs);
    let single_estimate = estimate_crown_backward_peak_bytes(&layers, 1);
    let budget_bytes = (full_estimate / 4).max(single_estimate);
    // Use the production binary-search helper instead of a local linear scan (#3515 Phase 3).
    let budget_batch = max_specs_per_budget(&layers, num_specs, budget_bytes);
    assert!(
        budget_batch > 0,
        "budget should admit at least one spec batch"
    );
    assert!(
        budget_batch < num_specs,
        "budget should force batching for metaroom-like workload"
    );

    let budget_mb = budget_bytes.div_ceil(1024 * 1024);
    let _budget = ScopedEnvVar::set("NY_GPU_MEMORY_BUDGET_MB", &budget_mb.to_string());
    let spec = identity_spec(num_specs);
    let inp_l = vec![-0.25f32; input_dim];
    let inp_u = vec![0.25f32; input_dim];

    let auto_batched = device
        .crown_backward_gpu(&layers, &spec, num_specs, &inp_l, &inp_u)
        .expect("budget-batched run should succeed");
    device
        .clear_crown_working_set()
        .expect("clear CROWN working set between budget batch checks");
    let manual_chunks = run_manual_spec_batches(
        &device,
        &layers,
        &spec,
        num_specs,
        budget_batch,
        &inp_l,
        &inp_u,
    );

    assert_results_close(
        &auto_batched,
        &manual_chunks,
        1e-4,
        "budget-driven spec batching",
    );
}

#[test]
fn test_gpu_crown_memory_budget_fallback_3515() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    let _budget = ScopedEnvVar::set("NY_GPU_MEMORY_BUDGET_MB", "1");
    let (layers, input_dim, num_specs) = build_soundnessbench_like_layers();
    let spec = identity_spec(num_specs);
    let inp_l = vec![-0.01f32; input_dim];
    let inp_u = vec![0.01f32; input_dim];

    let result = device.crown_backward_gpu(&layers, &spec, num_specs, &inp_l, &inp_u);
    assert!(
        result.is_err(),
        "1 MB budget should reject even a single soundnessbench spec batch",
    );
    let err = if let Err(err) = result {
        err
    } else {
        return;
    };
    assert!(
        matches!(err, ny_core::NyError::GpuMemoryExceeded { .. }),
        "expected GpuMemoryExceeded, got {err:?}",
    );
    if let ny_core::NyError::GpuMemoryExceeded {
        required_bytes,
        budget_bytes,
    } = err
    {
        assert!(
            required_bytes > budget_bytes,
            "expected required_bytes > budget_bytes, got required={required_bytes} budget={budget_bytes}",
        );
        assert_eq!(
            budget_bytes,
            1024 * 1024,
            "expected the 1 MB budget override to propagate to the error, got {budget_bytes}",
        );
    }
}

/// Verify that the public CROWN working-set cleanup hook can be called between
/// model runs without affecting correctness (#3515 Phase 2 readiness).
#[test]
fn test_crown_resource_release_between_models() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // Run a small backward pass to populate pool + plan cache.
    let layers = vec![GpuCrownLayer::Linear {
        weight: vec![1.0, 0.0, 0.0, 1.0].into(),
        bias: Some(vec![0.0, 0.0].into()),
        out_features: 2,
        in_features: 2,
    }];
    let spec = identity_spec(2);
    let inp_l = vec![-1.0f32; 2];
    let inp_u = vec![1.0f32; 2];
    let first = device
        .crown_backward_gpu(&layers, &spec, 2, &inp_l, &inp_u)
        .expect("first run");

    // Release all CROWN resources through the public runner hook.
    device
        .clear_crown_working_set()
        .expect("clear CROWN working set");

    // Re-run: pool + plan rebuilt from scratch, results should match.
    let second = device
        .crown_backward_gpu(&layers, &spec, 2, &inp_l, &inp_u)
        .expect("second run after release");

    assert_results_close(&first, &second, 1e-6, "resource release roundtrip");
}
