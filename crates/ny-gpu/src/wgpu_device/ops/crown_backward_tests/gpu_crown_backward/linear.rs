// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU CROWN backward tests for Linear + activation networks.

use super::*;
use rayon::prelude::*;

/// Build GpuCrownLayer list for Linear1->ReLU->Linear2 (backward order).
fn build_layers(
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    inp_l: &[f32],
    inp_u: &[f32],
) -> Vec<GpuCrownLayer> {
    let (pre_l, pre_u) = ibp_forward_linear(&w1, &b1, inp_l, inp_u, hidden, in_dim);
    let (ls, us, li, ui) = relu_slopes(&pre_l, &pre_u);
    vec![
        GpuCrownLayer::Linear {
            weight: w2.into(),
            bias: Some(b2.into()),
            out_features: out_dim,
            in_features: hidden,
            cert_err: Default::default(),
        },
        GpuCrownLayer::Activation {
            lower_slope: ls,
            upper_slope: us,
            lower_intercept: li,
            upper_intercept: ui,
            num_neurons: hidden,
        },
        GpuCrownLayer::Linear {
            weight: w1.into(),
            bias: Some(b1.into()),
            out_features: hidden,
            in_features: in_dim,
            cert_err: Default::default(),
        },
    ]
}

/// Build GpuCrownLayer list with caller-supplied slopes/intercepts (not ReLU).
///
/// Tests the GPU activation backward shader with arbitrary slope/intercept
/// patterns, catching bugs that ReLU's restricted {0,1} slope domain would miss.
fn build_layers_general_activation(
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    lower_slope: Vec<f32>,
    upper_slope: Vec<f32>,
    lower_intercept: Vec<f32>,
    upper_intercept: Vec<f32>,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
) -> Vec<GpuCrownLayer> {
    vec![
        GpuCrownLayer::Linear {
            weight: w2.into(),
            bias: Some(b2.into()),
            out_features: out_dim,
            in_features: hidden,
            cert_err: Default::default(),
        },
        GpuCrownLayer::Activation {
            lower_slope,
            upper_slope,
            lower_intercept,
            upper_intercept,
            num_neurons: hidden,
        },
        GpuCrownLayer::Linear {
            weight: w1.into(),
            bias: Some(b1.into()),
            out_features: hidden,
            in_features: in_dim,
            cert_err: Default::default(),
        },
    ]
}

// ---- Deterministic Tests ----

#[test]
fn test_crown_backward_gpu_deterministic() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    let layers = build_layers(
        vec![0.3f32, -0.2, 0.1, 0.5, -0.4, 0.6], // w1: 3x2
        vec![0.0f32, 0.1, -0.1],                 // b1
        vec![0.5f32, -0.3, 0.1, 0.2, 0.4, -0.5], // w2: 2x3
        vec![0.1f32, -0.1],                      // b2
        2,
        3,
        2, // in, hidden, out
        &[-1.0, -1.0],
        &[1.0, 1.0],
    );
    assert_gpu_matches_cpu(&device, &layers, 2, &[-1.0, -1.0], &[1.0, 1.0], 1e-4);
}

#[test]
fn test_crown_backward_gpu_hidden_larger_than_output() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();
    // hidden_dim (10) >> out_dim (2): the exact case 3446 caught.
    let w1: Vec<f32> = (0..20).map(|i| 0.1 * (i as f32 - 5.0)).collect();
    let w2: Vec<f32> = (0..20).map(|i| 0.1 * (i as f32 - 5.0)).collect();
    let layers = build_layers(
        w1,
        vec![0.0f32; 10],
        w2,
        vec![0.0f32; 2],
        2,
        10,
        2,
        &[-1.0, -1.0],
        &[1.0, 1.0],
    );
    assert_gpu_matches_cpu(&device, &layers, 2, &[-1.0, -1.0], &[1.0, 1.0], 1e-3);
}

/// Deterministic test with sigmoid-like slopes/intercepts.
///
/// Sigmoid relaxation produces slopes in (0, 0.25] and non-zero intercepts.
/// This catches GPU bugs where the shader mishandles fractional slopes or
/// non-zero intercepts that ReLU never produces.
#[test]
fn test_crown_backward_gpu_sigmoid_like_activation() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let w1 = vec![0.3f32, -0.2, 0.1, 0.5, -0.4, 0.6];
    let b1 = vec![0.0, 0.1, -0.1];
    let w2 = vec![0.5, -0.3, 0.1, 0.2, 0.4, -0.5];
    let b2 = vec![0.1, -0.1];

    // Sigmoid-like slopes: all in (0, 0.25], non-zero intercepts
    let ls = vec![0.15, 0.20, 0.10];
    let us = vec![0.25, 0.18, 0.22];
    let li = vec![-0.05, 0.02, -0.03];
    let ui = vec![0.08, -0.01, 0.06];

    let layers = build_layers_general_activation(w1, b1, w2, b2, ls, us, li, ui, 2, 3, 2);
    assert_gpu_matches_cpu(&device, &layers, 2, &[-1.0, -1.0], &[1.0, 1.0], 1e-4);
}

/// Deterministic test with mixed negative slopes (GELU-like unstable region).
///
/// GELU can produce negative lower slopes in its unstable region. This tests
/// that the GPU shader correctly handles sign flips when slopes are negative.
#[test]
fn test_crown_backward_gpu_negative_slope_activation() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let w1 = vec![0.5f32, -0.3, 0.2, -0.1, 0.4, 0.6];
    let b1 = vec![0.1, -0.2, 0.0];
    let w2 = vec![-0.4, 0.3, 0.2, 0.1, -0.5, 0.4];
    let b2 = vec![0.0, 0.05];

    // GELU-like: negative lower slopes, positive upper slopes
    let ls = vec![-0.1, 0.0, -0.05];
    let us = vec![0.8, 1.0, 0.6];
    let li = vec![0.02, 0.0, 0.01];
    let ui = vec![0.1, 0.0, 0.15];

    let layers = build_layers_general_activation(w1, b1, w2, b2, ls, us, li, ui, 2, 3, 2);
    assert_gpu_matches_cpu(&device, &layers, 2, &[-0.5, -0.5], &[0.5, 0.5], 1e-4);
}

// ---- Proptests ----

// Random Linear->ReLU->Linear networks, GPU matches CPU reference.
proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(50) })]
    #[test]
    fn proptest_crown_backward_gpu_vs_cpu(
        in_dim in 2usize..=6,
        hidden_dim in 2usize..=8,
        out_dim in 1usize..=4,
        w1_seed in proptest::collection::vec(-2.0f32..2.0, 48),
        w2_seed in proptest::collection::vec(-2.0f32..2.0, 48),
        b1_seed in proptest::collection::vec(-1.0f32..1.0, 8),
        b2_seed in proptest::collection::vec(-1.0f32..1.0, 4),
        inp_center in proptest::collection::vec(-2.0f32..2.0, 6),
        inp_radius in proptest::collection::vec(0.01f32..1.0, 6),
    ) {
        let _gpu_serial = gpu_test_serial_guard();
        let device = require_device();

        let w1: Vec<f32> = (0..hidden_dim * in_dim).map(|i| w1_seed[i % w1_seed.len()]).collect();
        let w2: Vec<f32> = (0..out_dim * hidden_dim).map(|i| w2_seed[i % w2_seed.len()]).collect();
        let b1: Vec<f32> = (0..hidden_dim).map(|i| b1_seed[i % b1_seed.len()]).collect();
        let b2: Vec<f32> = (0..out_dim).map(|i| b2_seed[i % b2_seed.len()]).collect();
        let inp_l: Vec<f32> = (0..in_dim).map(|i| inp_center[i % 6] - inp_radius[i % 6]).collect();
        let inp_u: Vec<f32> = (0..in_dim).map(|i| inp_center[i % 6] + inp_radius[i % 6]).collect();

        let layers = build_layers(w1, b1, w2, b2, in_dim, hidden_dim, out_dim, &inp_l, &inp_u);
        let spec = identity_spec(out_dim);

        let gpu = device.crown_backward_gpu(&layers, &spec, out_dim, &inp_l, &inp_u)
            .map_err(|e| TestCaseError::fail(format!("GPU error: {e}")))?;
        let (cpu_l, cpu_u) = cpu_crown_backward(&layers, &spec, out_dim, &inp_l, &inp_u);

        let eps = 1e-2;
        for i in 0..out_dim {
            let dl = (gpu.lower_bounds[i] - cpu_l[i]).abs();
            let du = (gpu.upper_bounds[i] - cpu_u[i]).abs();
            prop_assert!(dl <= eps, "lower[{i}] GPU={} CPU={} diff={dl}", gpu.lower_bounds[i], cpu_l[i]);
            prop_assert!(du <= eps, "upper[{i}] GPU={} CPU={} diff={du}", gpu.upper_bounds[i], cpu_u[i]);
            prop_assert!(gpu.lower_bounds[i] <= gpu.upper_bounds[i] + eps,
                "spec {i}: lower {} > upper {}", gpu.lower_bounds[i], gpu.upper_bounds[i]);
        }
    }
}

// General activation slopes/intercepts (non-ReLU patterns).
//
// The GPU activation backward shader operates on arbitrary (slope, intercept)
// pairs. ReLU only exercises slopes in {0, 1} and intercepts in {0, -us*l}.
// This test generates slope/intercept values mimicking sigmoid, tanh, GELU,
// and other activations whose slopes span (0, 1) and intercepts can be
// non-zero, catching GPU bugs invisible to ReLU-only testing.
proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(50) })]
    #[test]
    fn proptest_crown_backward_gpu_general_activation_vs_cpu(
        in_dim in 2usize..=6,
        hidden_dim in 2usize..=8,
        out_dim in 1usize..=4,
        w1_seed in proptest::collection::vec(-1.0f32..1.0, 48),
        w2_seed in proptest::collection::vec(-1.0f32..1.0, 48),
        b1_seed in proptest::collection::vec(-0.5f32..0.5, 8),
        b2_seed in proptest::collection::vec(-0.5f32..0.5, 4),
        // Slopes in [0, 1] (valid linear relaxation range for monotone activations)
        ls_seed in proptest::collection::vec(0.0f32..1.0, 8),
        us_seed in proptest::collection::vec(0.0f32..1.0, 8),
        // Intercepts can be non-zero (e.g., sigmoid upper relaxation has positive intercept)
        li_seed in proptest::collection::vec(-0.5f32..0.5, 8),
        ui_seed in proptest::collection::vec(-0.5f32..0.5, 8),
        inp_center in proptest::collection::vec(-1.0f32..1.0, 6),
        inp_radius in proptest::collection::vec(0.01f32..0.5, 6),
    ) {
        let _gpu_serial = gpu_test_serial_guard();
        let device = require_device();

        let w1: Vec<f32> = (0..hidden_dim * in_dim).map(|i| w1_seed[i % w1_seed.len()]).collect();
        let w2: Vec<f32> = (0..out_dim * hidden_dim).map(|i| w2_seed[i % w2_seed.len()]).collect();
        let b1: Vec<f32> = (0..hidden_dim).map(|i| b1_seed[i % b1_seed.len()]).collect();
        let b2: Vec<f32> = (0..out_dim).map(|i| b2_seed[i % b2_seed.len()]).collect();
        // `ls_seed`/`us_seed` are drawn INDEPENDENTLY from 0.0..1.0, which
        // freely produces `lower_slope > upper_slope` — a degenerate relaxation
        // (the lower envelope crossing above the upper) that production never
        // supplies. The sound GPU walk is RIGHT to refuse it: random search
        // found `ls=[0.969,0.943,0.485] us=[0.257,0.0,0.114]` and the device
        // published the FALLBACK_BOUND degrade (a valid, useless lower bound)
        // while the CPU returned a number, so the closeness assertion failed on
        // input neither lane should be compared on. Order the pair instead of
        // weakening the assertion: `us` now lands in `[ls, 1]`.
        let ls: Vec<f32> = (0..hidden_dim).map(|i| ls_seed[i % ls_seed.len()]).collect();
        let us: Vec<f32> = (0..hidden_dim)
            .map(|i| {
                let l = ls[i];
                l + (1.0 - l) * us_seed[i % us_seed.len()]
            })
            .collect();
        // Same degeneracy in the intercept dimension: `li_seed`/`ui_seed` are
        // independent, so the generator produces `lower_intercept >
        // upper_intercept` (the recorded seed has li=[0.204,-0.359,-0.458] vs
        // ui=[0.147,-0.407,-0.500]) — again a crossed envelope production never
        // supplies, and again the device is right to publish the degrade.
        // Order the pair: `ui` lands at or above `li`.
        let li: Vec<f32> = (0..hidden_dim).map(|i| li_seed[i % li_seed.len()]).collect();
        let ui: Vec<f32> = (0..hidden_dim)
            .map(|i| li[i].max(ui_seed[i % ui_seed.len()]))
            .collect();
        let inp_l: Vec<f32> = (0..in_dim).map(|i| inp_center[i % 6] - inp_radius[i % 6]).collect();
        let inp_u: Vec<f32> = (0..in_dim).map(|i| inp_center[i % 6] + inp_radius[i % 6]).collect();

        let layers = build_layers_general_activation(
            w1, b1, w2, b2, ls, us, li, ui, in_dim, hidden_dim, out_dim,
        );
        let spec = identity_spec(out_dim);

        // Compute CPU reference first to check for inverted bounds.
        // Arbitrary slopes/intercepts may not form a valid activation relaxation,
        // which can produce lower > upper. The GPU concretize shader's inversion
        // guard returns (-1e10, +1e10) in this case — correct behavior for invalid
        // inputs, but not comparable to the raw CPU result. Skip these cases.
        let (cpu_l, cpu_u) = cpu_crown_backward(&layers, &spec, out_dim, &inp_l, &inp_u);
        for i in 0..out_dim {
            prop_assume!(cpu_l[i] <= cpu_u[i] + 1e-3);
        }

        let gpu = device.crown_backward_gpu(&layers, &spec, out_dim, &inp_l, &inp_u)
            .map_err(|e| TestCaseError::fail(format!("GPU general activation error: {e}")))?;

        // Tighter tolerance than ReLU tests: smaller weights (-1..1) and inputs
        // reduce accumulation error, making 1e-3 achievable for these sizes.
        let eps = 1e-3;
        for i in 0..out_dim {
            let dl = (gpu.lower_bounds[i] - cpu_l[i]).abs();
            let du = (gpu.upper_bounds[i] - cpu_u[i]).abs();
            prop_assert!(dl <= eps,
                "general_act lower[{i}] GPU={} CPU={} diff={dl}", gpu.lower_bounds[i], cpu_l[i]);
            prop_assert!(du <= eps,
                "general_act upper[{i}] GPU={} CPU={} diff={du}", gpu.upper_bounds[i], cpu_u[i]);
            prop_assert!(gpu.lower_bounds[i] <= gpu.upper_bounds[i] + eps,
                "general_act spec {i}: lower {} > upper {}",
                gpu.lower_bounds[i], gpu.upper_bounds[i]);
        }
    }
}

/// Regression: concurrent CROWN backward on one shared device must NOT abort and
/// must produce the same result as the sequential CPU reference.
///
/// This reproduces the live ACAS-Xu wgpu crash: a tiny 5-input fully-connected
/// net (Linear(5->8) -> ReLU -> Linear(8->5)) driven through the GPU CROWN
/// backward from many Rayon worker threads — exactly how BaB input-splitting
/// fans out subdomains. Before the fix, two concurrent calls mapped the *same*
/// shared cached-plan readback buffer and hit wgpu's
/// `assert_eq!(mapped_range, 0..0, "Buffer is already mapped")` panic
/// (`wgpu-28.0.0/src/api/buffer.rs:572`), aborting the process under
/// panic=abort. The `gpu_serialize` lock now serializes GPU submit+readback so
/// concurrent calls are safe and every result matches the single-threaded CPU
/// reference (soundness: GPU == CPU).
#[test]
fn test_crown_backward_gpu_concurrent_shared_device_no_panic_acasxu_5d() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let in_dim = 5usize;
    let hidden = 8usize;
    let out_dim = 5usize;

    // Deterministic small weights/biases (acasxu-scale magnitudes).
    let w1: Vec<f32> = (0..(hidden * in_dim))
        .map(|i| 0.05 * (i as f32 - (hidden * in_dim) as f32 / 2.0))
        .collect();
    let b1: Vec<f32> = (0..hidden).map(|i| 0.01 * i as f32 - 0.04).collect();
    let w2: Vec<f32> = (0..(out_dim * hidden))
        .map(|i| 0.03 * ((i % 7) as f32 - 3.0))
        .collect();
    let b2: Vec<f32> = (0..out_dim).map(|i| 0.02 * i as f32).collect();

    let inp_l = vec![-0.5f32; in_dim];
    let inp_u = vec![0.5f32; in_dim];

    let layers = build_layers(w1, b1, w2, b2, in_dim, hidden, out_dim, &inp_l, &inp_u);
    let spec = identity_spec(out_dim);

    // Single-threaded CPU reference verdict.
    let (cpu_l, cpu_u) = cpu_crown_backward(&layers, &spec, out_dim, &inp_l, &inp_u);

    // Fan out concurrent GPU CROWN backward calls on the SAME shared device,
    // mirroring Rayon BaB. None may panic; all must match the CPU reference.
    let results: Vec<GpuCrownResult> = (0..64)
        .into_par_iter()
        .map(|_| {
            device
                .crown_backward_gpu(&layers, &spec, out_dim, &inp_l, &inp_u)
                .expect("concurrent GPU CROWN backward must not abort or error")
        })
        .collect();

    let eps = 1e-3;
    for (run, gpu) in results.iter().enumerate() {
        for i in 0..out_dim {
            let dl = (gpu.lower_bounds[i] - cpu_l[i]).abs();
            let du = (gpu.upper_bounds[i] - cpu_u[i]).abs();
            assert!(
                dl <= eps,
                "run {run} lower[{i}] GPU={} CPU={} diff={dl}",
                gpu.lower_bounds[i],
                cpu_l[i]
            );
            assert!(
                du <= eps,
                "run {run} upper[{i}] GPU={} CPU={} diff={du}",
                gpu.upper_bounds[i],
                cpu_u[i]
            );
        }
    }
}
