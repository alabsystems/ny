// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Plan-cache lifetime regression tests (#perf-plan-cache stale-alias fix).
//!
//! The CROWN plan cache keys static weights by **`Arc` pointer identity**
//! (crown_plan_key.rs). That is only sound while the keyed allocations are
//! alive: if a model's weight `Arc`s are dropped, the allocator can hand the
//! same address to a *different* model's weights, colliding with the stale key
//! and serving the old plan — whose old weights are baked into its staging
//! buffer — for the new model. This produced wrong (not merely loose) bounds:
//! `test_crown_backward_dual_alpha_crossing` observed GPU=-3.3 vs CPU=-0.325
//! when run after tests whose same-sized weight `Arc`s had been freed.
//!
//! The fix makes every cached `PreparedCrownPlan` hold keep-alive clones of
//! its weight `Arc`s (`static_weight_arcs`), so no keyed address can be
//! recycled while its key is live. These tests pin that lifetime contract.

use super::*;
use std::sync::Arc;

/// The cached plan must keep the model's weight/bias `Arc`s alive after the
/// caller drops its layer list — otherwise the pointer-identity cache key can
/// alias a recycled allocation (the stale-plan bug above).
#[test]
fn crown_plan_cache_keeps_static_weight_arcs_alive() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    // Fresh cache so counts below are attributable to this test's plan.
    device
        .clear_crown_plan_cache()
        .expect("clear plan cache before lifetime check");

    let weight: Arc<[f32]> = vec![0.7f32, -0.3, 0.2, 0.6].into();
    let bias: Arc<[f32]> = vec![0.05f32, -0.04].into();
    let layers = vec![GpuCrownLayer::Linear {
        weight: weight.clone(),
        bias: Some(bias.clone()),
        out_features: 2,
        in_features: 2,
        cert_err: Default::default(),
    }];
    let spec = vec![1.0f32, 0.0, 0.0, 1.0];
    device
        .crown_backward_gpu(&layers, &spec, 2, &[-1.0, -1.0], &[1.0, 1.0])
        .expect("backward should succeed");
    drop(layers);

    // Test copy + the cached plan's keep-alive clone (at least).
    assert!(
        Arc::strong_count(&weight) >= 2,
        "cached plan must retain the weight Arc (got strong_count={}); \
         without it the pointer-identity plan key can alias a recycled \
         allocation and serve a stale plan with the OLD model's weights",
        Arc::strong_count(&weight)
    );
    assert!(
        Arc::strong_count(&bias) >= 2,
        "cached plan must retain the bias Arc (got strong_count={})",
        Arc::strong_count(&bias)
    );

    // Clearing the cache must release the retained Arcs (no leak across models).
    device
        .clear_crown_plan_cache()
        .expect("clear plan cache after lifetime check");
    assert_eq!(
        Arc::strong_count(&weight),
        1,
        "clear_crown_plan_cache must release the retained weight Arc"
    );
    assert_eq!(
        Arc::strong_count(&bias),
        1,
        "clear_crown_plan_cache must release the retained bias Arc"
    );
}

/// End-to-end alias regression: run model A, drop it, then run model B with
/// identical topology/shapes but different weights, many times. With the
/// keep-alive fix the B results must always match B's own fresh-device result
/// (before the fix, B could silently reuse A's cached plan when the allocator
/// recycled A's weight addresses).
#[test]
fn crown_plan_cache_no_stale_alias_across_models() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    device
        .clear_crown_plan_cache()
        .expect("clear plan cache before alias check");

    let spec = vec![1.0f32, 0.0, 0.0, 1.0];
    let inp_l = [-1.0f32, -1.0];
    let inp_u = [1.0f32, 1.0];

    let build = |w: &[f32], b: &[f32]| -> Vec<GpuCrownLayer> {
        vec![GpuCrownLayer::Linear {
            weight: w.to_vec().into(),
            bias: Some(b.to_vec().into()),
            out_features: 2,
            in_features: 2,
            cert_err: Default::default(),
        }]
    };

    // Reference result for model B on a clean cache.
    let b_layers = build(&[0.5, -0.3, 0.2, 0.4], &[0.1, -0.1]);
    let b_ref = device
        .crown_backward_gpu(&b_layers, &spec, 2, &inp_l, &inp_u)
        .expect("B reference");
    drop(b_layers);
    device
        .clear_crown_plan_cache()
        .expect("clear plan cache after B reference");

    // Interleave: build+run A, drop A's Arcs, build+run B into the SAME-sized
    // allocations (the allocator is free to recycle A's addresses), compare.
    for i in 0..16 {
        let a_layers = build(&[1.0, 0.0, 0.0, 1.0], &[1.0, 1.0]);
        device
            .crown_backward_gpu(&a_layers, &spec, 2, &inp_l, &inp_u)
            .expect("A run");
        drop(a_layers); // A's weight Arcs die; addresses may be recycled...

        let b_layers = build(&[0.5, -0.3, 0.2, 0.4], &[0.1, -0.1]);
        let b_run = device
            .crown_backward_gpu(&b_layers, &spec, 2, &inp_l, &inp_u)
            .expect("B run");
        for j in 0..2 {
            assert!(
                (b_run.lower_bounds[j] - b_ref.lower_bounds[j]).abs() <= 1e-6
                    && (b_run.upper_bounds[j] - b_ref.upper_bounds[j]).abs() <= 1e-6,
                "iteration {i}: model B served stale bounds (lower[{j}]={} vs ref {}, \
                 upper[{j}]={} vs ref {}) — plan-cache pointer aliasing",
                b_run.lower_bounds[j],
                b_ref.lower_bounds[j],
                b_run.upper_bounds[j],
                b_ref.upper_bounds[j]
            );
        }
        drop(b_run);
    }

    device
        .clear_crown_plan_cache()
        .expect("clear plan cache after alias check");
}
