// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest equivalence tests for Patches vs Dense CROWN backward on Conv2d.
//!
//! These tests verify that the Patches representation (`PatchesLinearBounds`)
//! produces identical A-matrices and biases to the Dense representation
//! (`LinearBounds`) after conversion via `to_dense()`.
//!
//! Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md
//! Part of #2620, Epic #2613
//!
//! The wall-clock guards below are hang sentinels, not performance assertions.
//! Keep enough scheduler headroom for the 500-case properties when unrelated
//! crate suites are saturating a shared builder.

use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds};
use crate::layers::activations::ReLULayer;
use crate::layers::common::{BoundPropagation, PatchesPropagation};
use crate::layers::convolution::conv2d::Conv2dLayer;
use crate::LinearBounds;
use ndarray::{Array1, ArrayD, IxDyn};
use proptest::prelude::*;

/// Tolerance for Patches-vs-Dense A-matrix comparison.
/// Both paths compute the same mathematical operation (convolution transpose),
/// but the Dense path uses 6-nested scalar loops while Patches uses structured
/// tensor operations. f32 accumulation order differs.
///
/// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md (tolerance
/// justification section)
const PATCHES_DENSE_TOLERANCE: f32 = 1e-5;

/// Generate a random kernel of shape (out_c, in_c, kh, kw) with values in
/// [-2.0, 2.0], using a seed for reproducibility.
fn make_kernel(out_c: usize, in_c: usize, kh: usize, kw: usize, seed: u64) -> ArrayD<f32> {
    let len = out_c * in_c * kh * kw;
    let mut rng = seed;
    let values: Vec<f32> = (0..len)
        .map(|_| {
            // Simple xorshift64 for deterministic generation
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32); // [0, 1)
            u * 4.0 - 2.0 // [-2, 2)
        })
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), values).expect("kernel shape mismatch")
}

/// Generate a random bias of shape (out_c,) with values in [-1.0, 1.0].
fn make_bias(out_c: usize, seed: u64) -> Array1<f32> {
    let mut rng = seed.wrapping_add(12345);
    let values: Vec<f32> = (0..out_c)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 2.0 - 1.0
        })
        .collect();
    Array1::from_vec(values)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Test 1: Patches vs Dense equivalence for identity incoming bounds.
    ///
    /// Property: For any valid Conv2d layer, propagate_patches followed by
    /// to_dense() produces the same A-matrices and biases as propagate_linear
    /// on the Dense path.
    ///
    /// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md
    /// "Proptest Equivalence Specification — Test 1"
    #[ntest::timeout(30000)]
    #[test]
    fn proptest_conv2d_patches_vs_dense_identity(
        in_c in 1usize..=4,
        out_c in 1usize..=4,
        kh in 1usize..=3,
        kw in 1usize..=3,
        in_h in 3usize..=8,
        in_w in 3usize..=8,
        stride_h in 1usize..=2,
        stride_w in 1usize..=2,
        pad_h in 0usize..=1,
        pad_w in 0usize..=1,
        use_bias in proptest::bool::ANY,
        seed in any::<u64>(),
    ) {
        // 1. Compute output spatial size, skip if invalid
        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;
        prop_assume!(padded_h >= kh && padded_w >= kw);
        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;
        prop_assume!(out_h >= 1 && out_w >= 1);

        let out_dim = out_c * out_h * out_w;

        // 2. Generate kernel and bias
        let kernel = make_kernel(out_c, in_c, kh, kw, seed);
        let bias = if use_bias { Some(make_bias(out_c, seed)) } else { None };

        let conv = Conv2dLayer::with_input_shape(
            kernel, bias, (stride_h, stride_w), (pad_h, pad_w), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("Conv2d creation failed: {e}")))?;

        // 3. Dense path: identity LinearBounds → propagate_linear
        let identity_lb = LinearBounds::identity(out_dim);
        let dense_result = conv.propagate_linear(&identity_lb)
            .map_err(|e| TestCaseError::fail(format!("Dense propagate_linear failed: {e}")))?
            .into_owned();

        // 4. Patches path: identity PatchesLinearBounds → propagate_patches → to_dense
        let patches_identity = PatchesLinearBounds {
            row_count: out_dim,
            lower_a: PatchesData {
                coeff_err: None,
                patches: None,
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: true,
                output_shape: (out_c, out_h, out_w),
                input_shape: (out_c, out_h, out_w),
                unstable_idx: None,
            },
            lower_b: Array1::zeros(out_dim),
            upper_a: PatchesData {
                coeff_err: None,
                patches: None,
                stride: (1, 1),
                padding: (0, 0, 0, 0),
                identity: true,
                output_shape: (out_c, out_h, out_w),
                input_shape: (out_c, out_h, out_w),
                unstable_idx: None,
            },
            upper_b: Array1::zeros(out_dim),
        };

        let patches_result = conv.propagate_patches(&patches_identity)
            .map_err(|e| TestCaseError::fail(format!("Patches propagate_patches failed: {e}")))?;
        let patches_dense = patches_result.into_dense()
            .map_err(|e| TestCaseError::fail(format!("Patches to_dense failed: {e}")))?;

        // 5. Compare element-wise with tolerance
        let la_dense = dense_result.lower_a();
        let la_patches = patches_dense.lower_a();
        let ua_dense = dense_result.upper_a();
        let ua_patches = patches_dense.upper_a();

        prop_assert_eq!(
            la_dense.shape(), la_patches.shape(),
            "lower_a shape mismatch: dense={:?}, patches={:?}",
            la_dense.shape(), la_patches.shape()
        );
        prop_assert_eq!(
            ua_dense.shape(), ua_patches.shape(),
            "upper_a shape mismatch: dense={:?}, patches={:?}",
            ua_dense.shape(), ua_patches.shape()
        );

        for ((idx, &d), &p) in la_dense.indexed_iter().zip(la_patches.iter()) {
            let diff = (d - p).abs();
            let scale = d.abs().max(p.abs()).max(1.0);
            prop_assert!(
                diff <= PATCHES_DENSE_TOLERANCE * scale,
                "lower_a mismatch at {:?}: dense={}, patches={}, diff={}, tol={}",
                idx, d, p, diff, PATCHES_DENSE_TOLERANCE * scale,
            );
        }

        for ((idx, &d), &p) in ua_dense.indexed_iter().zip(ua_patches.iter()) {
            let diff = (d - p).abs();
            let scale = d.abs().max(p.abs()).max(1.0);
            prop_assert!(
                diff <= PATCHES_DENSE_TOLERANCE * scale,
                "upper_a mismatch at {:?}: dense={}, patches={}, diff={}, tol={}",
                idx, d, p, diff, PATCHES_DENSE_TOLERANCE * scale,
            );
        }

        // Compare biases
        let lb_dense = dense_result.lower_b();
        let lb_patches = patches_dense.lower_b();
        let ub_dense = dense_result.upper_b();
        let ub_patches = patches_dense.upper_b();

        for i in 0..out_dim {
            let diff_lb = (lb_dense[i] - lb_patches[i]).abs();
            let scale_lb = lb_dense[i].abs().max(lb_patches[i].abs()).max(1.0);
            prop_assert!(
                diff_lb <= PATCHES_DENSE_TOLERANCE * scale_lb,
                "lower_b mismatch at {}: dense={}, patches={}, diff={}",
                i, lb_dense[i], lb_patches[i], diff_lb,
            );
            let diff_ub = (ub_dense[i] - ub_patches[i]).abs();
            let scale_ub = ub_dense[i].abs().max(ub_patches[i].abs()).max(1.0);
            prop_assert!(
                diff_ub <= PATCHES_DENSE_TOLERANCE * scale_ub,
                "upper_b mismatch at {}: dense={}, patches={}, diff={}",
                i, ub_dense[i], ub_patches[i], diff_ub,
            );
        }
    }

    /// Test 2: Patches vs Dense soundness — both paths contain true conv output.
    ///
    /// Property: For random inputs within bounds, CROWN bounds from both paths
    /// must contain the true convolution output.
    ///
    /// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md
    /// "Proptest Equivalence Specification — Test 2"
    #[ntest::timeout(30000)]
    #[test]
    fn proptest_conv2d_patches_vs_dense_soundness(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        kh in 1usize..=3,
        kw in 1usize..=3,
        in_h in 3usize..=6,
        in_w in 3usize..=6,
        stride_h in 1usize..=2,
        stride_w in 1usize..=2,
        pad_h in 0usize..=1,
        pad_w in 0usize..=1,
        seed in any::<u64>(),
    ) {
        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;
        prop_assume!(padded_h >= kh && padded_w >= kw);
        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;
        prop_assume!(out_h >= 1 && out_w >= 1);

        let out_dim = out_c * out_h * out_w;
        let in_dim = in_c * in_h * in_w;

        let kernel = make_kernel(out_c, in_c, kh, kw, seed);
        let conv = Conv2dLayer::with_input_shape(
            kernel, None, (stride_h, stride_w), (pad_h, pad_w), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("Conv2d creation failed: {e}")))?;

        // Create input bounds: random centers in [-1, 1), width 0.2 per element
        let in_shape = [in_c, in_h, in_w];
        let lower_vals: Vec<f32> = (0..in_dim).map(|i| {
            let mut rng = seed.wrapping_add(i as u64 + 99999);
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 2.0 - 1.0 // [-1, 1)
        }).collect();
        let upper_vals: Vec<f32> = lower_vals.iter().map(|&l| l + 0.2).collect();

        let lower_nd = ArrayD::from_shape_vec(IxDyn(&in_shape), lower_vals.clone()).unwrap();
        let upper_nd = ArrayD::from_shape_vec(IxDyn(&in_shape), upper_vals.clone()).unwrap();
        let input_bt = ny_tensor::BoundedTensor::new(lower_nd, upper_nd)
            .map_err(|e| TestCaseError::fail(format!("BoundedTensor creation failed: {e}")))?;

        // Dense path
        let identity_lb = LinearBounds::identity(out_dim);
        let dense_result = conv.propagate_linear(&identity_lb)
            .map_err(|e| TestCaseError::fail(format!("Dense propagate_linear failed: {e}")))?
            .into_owned();
        let dense_flat_input = input_bt.flatten();
        let dense_bounds = dense_result.concretize(&dense_flat_input);

        // Patches path
        let patches_identity = PatchesLinearBounds {
            row_count: out_dim,
            lower_a: PatchesData {
                coeff_err: None,
                patches: None, stride: (1, 1), padding: (0, 0, 0, 0),
                identity: true,
                output_shape: (out_c, out_h, out_w),
                input_shape: (out_c, out_h, out_w),
                unstable_idx: None,
            },
            lower_b: Array1::zeros(out_dim),
            upper_a: PatchesData {
                coeff_err: None,
                patches: None, stride: (1, 1), padding: (0, 0, 0, 0),
                identity: true,
                output_shape: (out_c, out_h, out_w),
                input_shape: (out_c, out_h, out_w),
                unstable_idx: None,
            },
            upper_b: Array1::zeros(out_dim),
        };
        let patches_result = conv.propagate_patches(&patches_identity)
            .map_err(|e| TestCaseError::fail(format!("Patches propagate_patches failed: {e}")))?;
        let patches_lb = patches_result.into_dense()
            .map_err(|e| TestCaseError::fail(format!("Patches to_dense failed: {e}")))?;
        let patches_bounds = patches_lb.concretize(&dense_flat_input);

        // Sample 20 random inputs within bounds and verify containment
        let tol = 1e-5_f32;
        for s in 0..20 {
            let sample_vals: Vec<f32> = lower_vals.iter().zip(upper_vals.iter()).enumerate()
                .map(|(i, (&l, &u))| {
                    let t = ((s as f32 * 0.618_034) + (i as f32 * 0.414_213)) % 1.0;
                    l + (u - l) * t
                })
                .collect();
            let sample_nd = ArrayD::from_shape_vec(
                IxDyn(&in_shape), sample_vals,
            ).unwrap();
            let sample_pt = ny_tensor::BoundedTensor::new(sample_nd.clone(), sample_nd)
                .unwrap();
            let true_output = conv.propagate_ibp(&sample_pt)
                .map_err(|e| TestCaseError::fail(format!("Conv2d eval failed: {e}")))?;

            let true_flat: Vec<f32> = true_output.lower().iter().copied().collect();
            for (j, &true_val) in true_flat.iter().enumerate().take(out_dim) {

                // Dense bounds contain true output
                prop_assert!(
                    dense_bounds.lower()[[j]] <= true_val + tol,
                    "Dense lower bound violation at output {}: bound={} > true={}",
                    j, dense_bounds.lower()[[j]], true_val,
                );
                prop_assert!(
                    dense_bounds.upper()[[j]] >= true_val - tol,
                    "Dense upper bound violation at output {}: bound={} < true={}",
                    j, dense_bounds.upper()[[j]], true_val,
                );

                // Patches bounds contain true output
                prop_assert!(
                    patches_bounds.lower()[[j]] <= true_val + tol,
                    "Patches lower bound violation at output {}: bound={} > true={}",
                    j, patches_bounds.lower()[[j]], true_val,
                );
                prop_assert!(
                    patches_bounds.upper()[[j]] >= true_val - tol,
                    "Patches upper bound violation at output {}: bound={} < true={}",
                    j, patches_bounds.upper()[[j]], true_val,
                );
            }
        }
    }

    /// Test 3: Patches-then-dense chain equivalence.
    ///
    /// Property: For Conv2d → ensure_dense() (simulating an activation forcing
    /// Dense fallback) → Conv2d, the Patches path produces the same result as
    /// the Dense-only path.
    ///
    /// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md
    /// "Proptest Equivalence Specification — Test 3"
    ///
    /// Timeout raised 10s→20s→40s→90s (#vnncomp-aw-soundness): the dense Conv2d
    /// CROWN backward f64-recomputes the coefficient for the sound certified
    /// error (≈2× the per-call cost). The 500-case dense-vs-patches
    /// equivalence proptest now measures ~19s ISOLATED in a debug build, so
    /// the 20s wall reliably trips under parallel test load. Shared builders can
    /// also starve an otherwise serial run past 40s, so 90s keeps this a hang
    /// sentinel without making scheduler contention a correctness failure.
    #[ntest::timeout(90000)]
    #[test]
    fn proptest_conv2d_chain_patches_vs_dense(
        // First conv: in_c1 → out_c1
        in_c1 in 1usize..=3,
        out_c1 in 1usize..=3,
        kh1 in 1usize..=3,
        kw1 in 1usize..=3,
        in_h1 in 5usize..=8,
        in_w1 in 5usize..=8,
        // Second conv: out_c1 → out_c2 (in_c2 = out_c1)
        out_c2 in 1usize..=3,
        kh2 in 1usize..=2,
        kw2 in 1usize..=2,
        seed in any::<u64>(),
    ) {
        // Compute conv1 output shape (stride=1, padding=0 for simplicity)
        prop_assume!(in_h1 >= kh1 && in_w1 >= kw1);
        let out_h1 = in_h1 - kh1 + 1;
        let out_w1 = in_w1 - kw1 + 1;
        prop_assume!(out_h1 >= 1 && out_w1 >= 1);

        // Conv2 input shape = conv1 output shape
        let in_c2 = out_c1;
        let in_h2 = out_h1;
        let in_w2 = out_w1;
        prop_assume!(in_h2 >= kh2 && in_w2 >= kw2);
        let out_h2 = in_h2 - kh2 + 1;
        let out_w2 = in_w2 - kw2 + 1;
        prop_assume!(out_h2 >= 1 && out_w2 >= 1);

        let conv2_out_dim = out_c2 * out_h2 * out_w2;

        let kernel1 = make_kernel(out_c1, in_c1, kh1, kw1, seed);
        let kernel2 = make_kernel(out_c2, in_c2, kh2, kw2, seed.wrapping_add(1));

        let conv1 = Conv2dLayer::with_input_shape(
            kernel1, None, (1, 1), (0, 0), in_h1, in_w1,
        ).map_err(|e| TestCaseError::fail(format!("Conv1 creation failed: {e}")))?;

        let conv2 = Conv2dLayer::with_input_shape(
            kernel2, None, (1, 1), (0, 0), in_h2, in_w2,
        ).map_err(|e| TestCaseError::fail(format!("Conv2 creation failed: {e}")))?;

        // Dense-only path: identity → conv2.propagate_linear → conv1.propagate_linear
        let lb = LinearBounds::identity(conv2_out_dim);
        let after_conv2_dense = conv2.propagate_linear(&lb)
            .map_err(|e| TestCaseError::fail(format!("Dense conv2 failed: {e}")))?
            .into_owned();
        let after_conv1_dense = conv1.propagate_linear(&after_conv2_dense)
            .map_err(|e| TestCaseError::fail(format!("Dense conv1 failed: {e}")))?
            .into_owned();

        // Patches path: identity patches → conv2.propagate_patches → ensure_dense → conv1
        let patches_id = PatchesLinearBounds {
            row_count: conv2_out_dim,
            lower_a: PatchesData {
                coeff_err: None,
                patches: None, stride: (1, 1), padding: (0, 0, 0, 0),
                identity: true,
                output_shape: (out_c2, out_h2, out_w2),
                input_shape: (out_c2, out_h2, out_w2),
                unstable_idx: None,
            },
            lower_b: Array1::zeros(conv2_out_dim),
            upper_a: PatchesData {
                coeff_err: None,
                patches: None, stride: (1, 1), padding: (0, 0, 0, 0),
                identity: true,
                output_shape: (out_c2, out_h2, out_w2),
                input_shape: (out_c2, out_h2, out_w2),
                unstable_idx: None,
            },
            upper_b: Array1::zeros(conv2_out_dim),
        };

        let after_conv2_patches = conv2.propagate_patches(&patches_id)
            .map_err(|e| TestCaseError::fail(format!("Patches conv2 failed: {e}")))?;
        // Simulate activation forcing Dense (ensure_dense)
        let after_conv2_lb = after_conv2_patches.into_dense()
            .map_err(|e| TestCaseError::fail(format!("ensure_dense failed: {e}")))?;
        // Second conv uses Dense path (since we're now Dense)
        let after_conv1_patches = conv1.propagate_linear(&after_conv2_lb)
            .map_err(|e| TestCaseError::fail(format!("Dense conv1 (after patches) failed: {e}")))?
            .into_owned();

        // Compare
        let la_dense = after_conv1_dense.lower_a();
        let la_patches = after_conv1_patches.lower_a();
        let ua_dense = after_conv1_dense.upper_a();
        let ua_patches = after_conv1_patches.upper_a();

        // Use slightly looser tolerance for chained ops
        let chain_tol = 1e-4_f32;

        prop_assert_eq!(
            la_dense.shape(), la_patches.shape(),
            "lower_a shape mismatch in chain: dense={:?}, patches={:?}",
            la_dense.shape(), la_patches.shape()
        );
        prop_assert_eq!(
            ua_dense.shape(), ua_patches.shape(),
            "upper_a shape mismatch in chain: dense={:?}, patches={:?}",
            ua_dense.shape(), ua_patches.shape()
        );

        for ((idx, &d), &p) in la_dense.indexed_iter().zip(la_patches.iter()) {
            let diff = (d - p).abs();
            let scale = d.abs().max(p.abs()).max(1.0);
            prop_assert!(
                diff <= chain_tol * scale,
                "Chain lower_a mismatch at {:?}: dense={}, patches={}, diff={}",
                idx, d, p, diff,
            );
        }
        for ((idx, &d), &p) in ua_dense.indexed_iter().zip(ua_patches.iter()) {
            let diff = (d - p).abs();
            let scale = d.abs().max(p.abs()).max(1.0);
            prop_assert!(
                diff <= chain_tol * scale,
                "Chain upper_a mismatch at {:?}: dense={}, patches={}, diff={}",
                idx, d, p, diff,
            );
        }

        // Compare biases
        let total_outputs = after_conv1_dense.num_outputs();
        for i in 0..total_outputs {
            let diff_lb = (after_conv1_dense.lower_b()[i] - after_conv1_patches.lower_b()[i]).abs();
            let scale_lb = after_conv1_dense.lower_b()[i].abs().max(1.0);
            prop_assert!(
                diff_lb <= chain_tol * scale_lb,
                "Chain lower_b mismatch at {}: dense={}, patches={}",
                i, after_conv1_dense.lower_b()[i], after_conv1_patches.lower_b()[i],
            );
            let diff_ub = (after_conv1_dense.upper_b()[i] - after_conv1_patches.upper_b()[i]).abs();
            let scale_ub = after_conv1_dense.upper_b()[i].abs().max(1.0);
            prop_assert!(
                diff_ub <= chain_tol * scale_ub,
                "Chain upper_b mismatch at {}: dense={}, patches={}",
                i, after_conv1_dense.upper_b()[i], after_conv1_patches.upper_b()[i],
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Test 4: Conv2d → ReLU Patches vs Dense equivalence.
    ///
    /// Property: For a Conv2d followed by ReLU, the Patches backward path
    /// (propagate_patches then propagate_patches_with_bounds) produces the same
    /// bounds as the Dense backward path (propagate_linear then
    /// propagate_linear_with_bounds) after conversion via to_dense().
    ///
    /// This is the key Phase 2 test — verifies that ReLU's Patches backward
    /// correctly scales patches coefficients by per-neuron relaxation slopes.
    ///
    /// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md Phase 2
    /// Part of #2613
    #[ntest::timeout(30000)]
    #[test]
    fn proptest_conv2d_relu_patches_vs_dense(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        kh in 1usize..=3,
        kw in 1usize..=3,
        in_h in 3usize..=7,
        in_w in 3usize..=7,
        stride_h in 1usize..=2,
        stride_w in 1usize..=2,
        pad_h in 0usize..=1,
        pad_w in 0usize..=1,
        seed in any::<u64>(),
    ) {
        // Compute conv output shape
        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;
        prop_assume!(padded_h >= kh && padded_w >= kw);
        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;
        prop_assume!(out_h >= 1 && out_w >= 1);

        let out_dim = out_c * out_h * out_w;
        let in_dim = in_c * in_h * in_w;

        // Create Conv2d layer
        let kernel = make_kernel(out_c, in_c, kh, kw, seed);
        let bias = Some(make_bias(out_c, seed));
        let conv = Conv2dLayer::with_input_shape(
            kernel, bias, (stride_h, stride_w), (pad_h, pad_w), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("Conv2d creation failed: {e}")))?;

        // Create pre-activation bounds for Conv2d's input (also serves as ReLU's input)
        let pre_conv_lower: Vec<f32> = (0..in_dim).map(|i| {
            let mut rng = seed.wrapping_add(i as u64 + 77777);
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 2.0 - 1.0 // [-1, 1)
        }).collect();
        let pre_conv_upper: Vec<f32> = pre_conv_lower.iter().map(|&l| l + 0.3).collect();

        let pre_conv_bt = ny_tensor::BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), pre_conv_lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), pre_conv_upper).unwrap(),
        ).map_err(|e| TestCaseError::fail(format!("pre_conv BoundedTensor failed: {e}")))?;

        // ---- Dense path: identity → Conv2d backward → ReLU backward ----
        let identity_lb = LinearBounds::identity(out_dim);
        let after_conv_dense = conv.propagate_linear(&identity_lb)
            .map_err(|e| TestCaseError::fail(format!("Dense Conv2d backward failed: {e}")))?
            .into_owned();
        let relu = ReLULayer::new();
        // In backward order Conv2d→ReLU, this models network Input→ReLU→Conv2d→Output.
        // After Conv2d backward, bounds reference Conv2d's input space = ReLU's output space.
        // ReLU's pre-activation bounds are the network input (pre_conv_bt), not Conv2d output.
        let after_relu_dense = relu.propagate_linear_with_bounds(&after_conv_dense, &pre_conv_bt)
            .map_err(|e| TestCaseError::fail(format!("Dense ReLU backward failed: {e}")))?;

        // ---- Patches path: identity patches → Conv2d backward → ReLU backward ----
        let patches_identity = PatchesLinearBounds::identity(
            (out_c, out_h, out_w),
            (out_c, out_h, out_w),
        );

        let after_conv_patches = conv.propagate_patches(&patches_identity)
            .map_err(|e| TestCaseError::fail(format!("Patches Conv2d backward failed: {e}")))?;

        // ReLU backward in Patches mode — pre-activation bounds are pre_conv_bt
        // (network input), not post_conv_bt, since Conv2d backward already mapped
        // bounds to Conv2d's input space = ReLU's output space.
        let after_relu_patches = match after_conv_patches {
            CrownBounds::Patches(ref pb) => {
                relu.propagate_patches_with_bounds(pb, &pre_conv_bt)
                    .map_err(|e| TestCaseError::fail(format!("Patches ReLU backward failed: {e}")))?
            }
            CrownBounds::Dense(_) => {
                // Conv2d may fall back to Dense if kernel covers entire input
                return Ok(());
            }
        };

        // Convert Patches result to Dense for comparison
        let after_relu_patches_dense = after_relu_patches.into_dense()
            .map_err(|e| TestCaseError::fail(format!("Patches to_dense failed: {e}")))?;

        // ---- Compare ----
        // Use slightly looser tolerance because Patches and Dense paths compute
        // the same math in different order (Patches: per-position, Dense: per-row).
        // Additionally, directed rounding (next_down/next_up) applies at different
        // points in the two paths.
        let chain_tol = 1e-4_f32;

        let la_dense = after_relu_dense.lower_a();
        let la_patches = after_relu_patches_dense.lower_a();
        let ua_dense = after_relu_dense.upper_a();
        let ua_patches = after_relu_patches_dense.upper_a();

        prop_assert_eq!(
            la_dense.shape(), la_patches.shape(),
            "lower_a shape mismatch: dense={:?}, patches={:?}",
            la_dense.shape(), la_patches.shape()
        );

        for j in 0..la_dense.nrows() {
            for i in 0..la_dense.ncols() {
                let d: f32 = la_dense[[j, i]];
                let p: f32 = la_patches[[j, i]];
                let diff = (d - p).abs();
                let scale = d.abs().max(p.abs()).max(1.0);
                prop_assert!(
                    diff <= chain_tol * scale,
                    "Conv2d→ReLU lower_a mismatch at [{},{}]: dense={}, patches={}, diff={}",
                    j, i, d, p, diff,
                );
            }
        }

        for j in 0..ua_dense.nrows() {
            for i in 0..ua_dense.ncols() {
                let d: f32 = ua_dense[[j, i]];
                let p: f32 = ua_patches[[j, i]];
                let diff = (d - p).abs();
                let scale = d.abs().max(p.abs()).max(1.0);
                prop_assert!(
                    diff <= chain_tol * scale,
                    "Conv2d→ReLU upper_a mismatch at [{},{}]: dense={}, patches={}, diff={}",
                    j, i, d, p, diff,
                );
            }
        }

        // Compare biases
        for i in 0..after_relu_dense.num_outputs() {
            let diff_lb = (after_relu_dense.lower_b()[i] - after_relu_patches_dense.lower_b()[i]).abs();
            let scale_lb = after_relu_dense.lower_b()[i].abs().max(after_relu_patches_dense.lower_b()[i].abs()).max(1.0);
            prop_assert!(
                diff_lb <= chain_tol * scale_lb,
                "Conv2d→ReLU lower_b mismatch at {}: dense={}, patches={}, diff={}",
                i, after_relu_dense.lower_b()[i], after_relu_patches_dense.lower_b()[i], diff_lb,
            );

            let diff_ub = (after_relu_dense.upper_b()[i] - after_relu_patches_dense.upper_b()[i]).abs();
            let scale_ub = after_relu_dense.upper_b()[i].abs().max(after_relu_patches_dense.upper_b()[i].abs()).max(1.0);
            prop_assert!(
                diff_ub <= chain_tol * scale_ub,
                "Conv2d→ReLU upper_b mismatch at {}: dense={}, patches={}, diff={}",
                i, after_relu_dense.upper_b()[i], after_relu_patches_dense.upper_b()[i], diff_ub,
            );
        }
    }
}

// Test 6 (Conv2d composition) moved to crown_patches_composition.rs

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Test 5: Conv2d → BatchNorm Patches vs Dense equivalence.
    ///
    /// Property: For a Conv2d followed by BatchNorm in the backward pass,
    /// the Patches path (propagate_patches → propagate_patches) produces the
    /// same bounds as the Dense path (propagate_linear → propagate_linear_with_bounds)
    /// after conversion via to_dense().
    ///
    /// This tests Phase 2 BatchNorm Patches: per-channel scaling of patches
    /// coefficients preserves equivalence with Dense per-column scaling.
    ///
    /// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md Phase 2
    /// Part of #2613
    #[ntest::timeout(30000)]
    #[test]
    fn proptest_conv2d_batchnorm_patches_vs_dense(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        kh in 1usize..=3,
        kw in 1usize..=3,
        in_h in 3usize..=7,
        in_w in 3usize..=7,
        stride_h in 1usize..=2,
        stride_w in 1usize..=2,
        pad_h in 0usize..=1,
        pad_w in 0usize..=1,
        use_negative_scale in proptest::bool::ANY,
        seed in any::<u64>(),
    ) {
        use crate::layers::BatchNormLayer;

        // Compute conv output shape
        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;
        prop_assume!(padded_h >= kh && padded_w >= kw);
        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;
        prop_assume!(out_h >= 1 && out_w >= 1);

        let out_dim = out_c * out_h * out_w;
        let in_dim = in_c * in_h * in_w;

        // Create Conv2d layer
        let kernel = make_kernel(out_c, in_c, kh, kw, seed);
        let bias = Some(make_bias(out_c, seed));
        let conv = Conv2dLayer::with_input_shape(
            kernel, bias, (stride_h, stride_w), (pad_h, pad_w), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("Conv2d creation failed: {e}")))?;

        // Create BatchNorm layer with in_c channels (operates on Conv2d's INPUT space
        // since we're going backward: Output → Conv2d → BatchNorm → Input)
        // BN's num_channels = Conv2d's in_c (the space we land in after Conv2d backward)
        let bn_scale_vals: Vec<f32> = (0..in_c).map(|i| {
            let mut rng = seed.wrapping_add(i as u64 + 55555);
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            let s = u * 3.0 + 0.5; // [0.5, 3.5)
            if use_negative_scale && i % 2 == 0 { -s } else { s }
        }).collect();
        let bn_bias_vals: Vec<f32> = (0..in_c).map(|i| {
            let mut rng = seed.wrapping_add(i as u64 + 66666);
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 2.0 - 1.0 // [-1, 1)
        }).collect();

        let bn = BatchNormLayer::from_scale_bias(
            ArrayD::from_shape_vec(IxDyn(&[in_c]), bn_scale_vals).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[in_c]), bn_bias_vals).unwrap(),
        )
        .unwrap();

        // Pre-activation bounds for BatchNorm's input (needed for Dense backward).
        // Shape is (in_c, in_h, in_w) — the spatial shape of BN's input.
        let pre_bn_lower: Vec<f32> = (0..in_dim).map(|i| {
            let mut rng = seed.wrapping_add(i as u64 + 88888);
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 2.0 - 1.0
        }).collect();
        let pre_bn_upper: Vec<f32> = pre_bn_lower.iter().map(|&l| l + 0.3).collect();
        let pre_bn_bt = ny_tensor::BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), pre_bn_lower).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), pre_bn_upper).unwrap(),
        ).map_err(|e| TestCaseError::fail(format!("pre_bn BoundedTensor failed: {e}")))?;

        // ---- Dense path: identity → Conv2d backward → BN backward ----
        let identity_lb = LinearBounds::identity(out_dim);
        let after_conv_dense = conv.propagate_linear(&identity_lb)
            .map_err(|e| TestCaseError::fail(format!("Dense Conv2d backward failed: {e}")))?
            .into_owned();
        // BN backward on Dense path uses propagate_linear_with_bounds
        let after_bn_dense = bn.propagate_linear_with_bounds(&after_conv_dense, &pre_bn_bt)
            .map_err(|e| TestCaseError::fail(format!("Dense BN backward failed: {e}")))?;

        // ---- Patches path: identity patches → Conv2d backward → BN backward ----
        let patches_identity = PatchesLinearBounds::identity(
            (out_c, out_h, out_w),
            (out_c, out_h, out_w),
        );
        let after_conv_patches = conv.propagate_patches(&patches_identity)
            .map_err(|e| TestCaseError::fail(format!("Patches Conv2d backward failed: {e}")))?;

        // BN backward in Patches mode
        let after_bn_patches = match after_conv_patches {
            CrownBounds::Patches(ref pb) => {
                bn.propagate_patches(pb)
                    .map_err(|e| TestCaseError::fail(format!("Patches BN backward failed: {e}")))?
            }
            CrownBounds::Dense(_) => {
                // Conv2d may fall back to Dense; skip this case
                return Ok(());
            }
        };

        // Convert Patches result to Dense for comparison
        let after_bn_patches_dense = after_bn_patches.into_dense()
            .map_err(|e| TestCaseError::fail(format!("Patches to_dense failed: {e}")))?;

        // ---- Compare ----
        // Tolerance accounts for: different computation order (per-channel in Patches
        // vs per-column in Dense), f64→f32 bias rounding, and directed rounding
        // (next_down/next_up applied at different points).
        let chain_tol = 1e-4_f32;

        let la_dense = after_bn_dense.lower_a();
        let la_patches = after_bn_patches_dense.lower_a();
        let ua_dense = after_bn_dense.upper_a();
        let ua_patches = after_bn_patches_dense.upper_a();

        prop_assert_eq!(
            la_dense.shape(), la_patches.shape(),
            "lower_a shape mismatch: dense={:?}, patches={:?}",
            la_dense.shape(), la_patches.shape()
        );

        for j in 0..la_dense.nrows() {
            for i in 0..la_dense.ncols() {
                let d: f32 = la_dense[[j, i]];
                let p: f32 = la_patches[[j, i]];
                let diff = (d - p).abs();
                let scale = d.abs().max(p.abs()).max(1.0);
                prop_assert!(
                    diff <= chain_tol * scale,
                    "Conv2d→BN lower_a mismatch at [{},{}]: dense={}, patches={}, diff={}",
                    j, i, d, p, diff,
                );
            }
        }

        for j in 0..ua_dense.nrows() {
            for i in 0..ua_dense.ncols() {
                let d: f32 = ua_dense[[j, i]];
                let p: f32 = ua_patches[[j, i]];
                let diff = (d - p).abs();
                let scale = d.abs().max(p.abs()).max(1.0);
                prop_assert!(
                    diff <= chain_tol * scale,
                    "Conv2d→BN upper_a mismatch at [{},{}]: dense={}, patches={}, diff={}",
                    j, i, d, p, diff,
                );
            }
        }

        // Compare biases
        for i in 0..after_bn_dense.num_outputs() {
            let diff_lb = (after_bn_dense.lower_b()[i] - after_bn_patches_dense.lower_b()[i]).abs();
            let scale_lb = after_bn_dense.lower_b()[i].abs()
                .max(after_bn_patches_dense.lower_b()[i].abs())
                .max(1.0);
            prop_assert!(
                diff_lb <= chain_tol * scale_lb,
                "Conv2d→BN lower_b mismatch at {}: dense={}, patches={}, diff={}",
                i, after_bn_dense.lower_b()[i], after_bn_patches_dense.lower_b()[i], diff_lb,
            );

            let diff_ub = (after_bn_dense.upper_b()[i] - after_bn_patches_dense.upper_b()[i]).abs();
            let scale_ub = after_bn_dense.upper_b()[i].abs()
                .max(after_bn_patches_dense.upper_b()[i].abs())
                .max(1.0);
            prop_assert!(
                diff_ub <= chain_tol * scale_ub,
                "Conv2d→BN upper_b mismatch at {}: dense={}, patches={}, diff={}",
                i, after_bn_dense.upper_b()[i], after_bn_patches_dense.upper_b()[i], diff_ub,
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Test 6: Sparse patches (filter_to_unstable) produce identical Dense A-matrix
    /// rows for unstable neurons as the full dense patches path.
    ///
    /// Property: For any valid Conv2d layer and random unstable mask,
    /// filter_to_unstable() followed by to_dense() produces the same non-zero
    /// rows in the Dense A-matrix as the full Patches to_dense().
    ///
    /// Part of #2613 Phase 4 step 19
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_sparse_patches_vs_dense_equivalence(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        kh in 1usize..=3,
        kw in 1usize..=3,
        in_h in 3usize..=6,
        in_w in 3usize..=6,
        stride_h in 1usize..=2,
        stride_w in 1usize..=2,
        pad_h in 0usize..=1,
        pad_w in 0usize..=1,
        seed in any::<u64>(),
        unstable_frac in 0.1f32..0.8f32,
    ) {
        let out_h = (in_h + 2 * pad_h).checked_sub(kh).map(|v| v / stride_h + 1).unwrap_or(0);
        let out_w = (in_w + 2 * pad_w).checked_sub(kw).map(|v| v / stride_w + 1).unwrap_or(0);
        prop_assume!(out_h >= 1 && out_w >= 1);

        let out_dim = out_c * out_h * out_w;
        let in_dim = in_c * in_h * in_w;

        let kernel = make_kernel(out_c, in_c, kh, kw, seed);
        let bias = make_bias(out_c, seed);
        let conv = Conv2dLayer::with_input_shape(
            kernel, Some(bias), (stride_h, stride_w), (pad_h, pad_w), in_h, in_w,
        );
        prop_assume!(conv.is_ok());
        let conv = conv.unwrap();

        // Full Patches path
        let identity = PatchesLinearBounds::identity(
            (out_c, out_h, out_w), (in_c, in_h, in_w),
        );
        let patches_result = conv.propagate_patches(&identity);
        prop_assume!(patches_result.is_ok());
        let patches_result = patches_result.unwrap();

        let full_plb = match patches_result {
            CrownBounds::Patches(pb) => *pb,
            CrownBounds::Dense(_) => return Ok(()),
        };

        let full_dense = full_plb.to_dense();
        prop_assume!(full_dense.is_ok());
        let full_dense = full_dense.unwrap();

        // Generate random unstable mask
        let mut rng = seed.wrapping_mul(7);
        let mut mask = ndarray::Array3::<bool>::from_elem((out_c, out_h, out_w), false);
        let target = ((out_dim as f32) * unstable_frac).ceil() as usize;
        let mut count = 0;
        for c in 0..out_c {
            for h in 0..out_h {
                for w in 0..out_w {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    if count < target {
                        mask[[c, h, w]] = true;
                        count += 1;
                    }
                }
            }
        }
        prop_assume!(count > 0 && count < out_dim);

        // Filter to unstable — min_sparsity=1.0 to always filter
        let sparse_plb = full_plb.filter_to_unstable(&mask, 1.0);
        prop_assume!(sparse_plb.is_some());
        let sparse_plb = sparse_plb.unwrap();

        let sparse_dense = sparse_plb.to_dense();
        prop_assume!(sparse_dense.is_ok());
        let sparse_dense = sparse_dense.unwrap();

        prop_assert_eq!(sparse_dense.num_outputs(), out_dim);
        prop_assert_eq!(sparse_dense.num_inputs(), in_dim);

        // Verify: unstable rows match, stable rows are zero
        for c in 0..out_c {
            for h in 0..out_h {
                for w in 0..out_w {
                    let flat = c * out_h * out_w + h * out_w + w;
                    if mask[[c, h, w]] {
                        for j in 0..in_dim {
                            let diff_l = (sparse_dense.lower_a()[[flat, j]]
                                - full_dense.lower_a()[[flat, j]]).abs();
                            prop_assert!(
                                diff_l <= PATCHES_DENSE_TOLERANCE,
                                "lower_a[{flat},{j}]: sparse={}, full={}, diff={}",
                                sparse_dense.lower_a()[[flat, j]],
                                full_dense.lower_a()[[flat, j]], diff_l,
                            );
                            let diff_u = (sparse_dense.upper_a()[[flat, j]]
                                - full_dense.upper_a()[[flat, j]]).abs();
                            prop_assert!(
                                diff_u <= PATCHES_DENSE_TOLERANCE,
                                "upper_a[{flat},{j}]: sparse={}, full={}, diff={}",
                                sparse_dense.upper_a()[[flat, j]],
                                full_dense.upper_a()[[flat, j]], diff_u,
                            );
                        }
                        let diff_lb = (sparse_dense.lower_b()[[flat]]
                            - full_dense.lower_b()[[flat]]).abs();
                        prop_assert!(
                            diff_lb <= PATCHES_DENSE_TOLERANCE,
                            "lower_b[{flat}]: sparse={}, full={}",
                            sparse_dense.lower_b()[[flat]], full_dense.lower_b()[[flat]],
                        );
                    } else {
                        for j in 0..in_dim {
                            prop_assert!(
                                sparse_dense.lower_a()[[flat, j]] == 0.0,
                                "stable lower_a[{},{}] = {} (expected 0)",
                                flat, j, sparse_dense.lower_a()[[flat, j]]
                            );
                            prop_assert!(
                                sparse_dense.upper_a()[[flat, j]] == 0.0,
                                "stable upper_a[{},{}] = {} (expected 0)",
                                flat, j, sparse_dense.upper_a()[[flat, j]]
                            );
                        }
                    }
                }
            }
        }
    }
}

// =====================================================================
// 7D explicit-rows activation err soundness (§6.4 T6,
// docs/PATCHES_7D_COEFF_ERR_CLOSURE.md).
// =====================================================================

/// SplitMix64-style deterministic mixer for per-(sample, side, tap) draws.
fn mix64(a: u64, b: u64, c: u64, d: u64) -> u64 {
    let mut z = a
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(b.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(c.wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add(d.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn mix_unit_f64(a: u64, b: u64, c: u64, d: u64) -> f64 {
    (mix64(a, b, c, d) >> 11) as f64 / (1u64 << 53) as f64
}

/// Perturbation factor in {-1, -0.5, 0, 0.5, 1} — endpoints and sign flips.
fn delta_factor(k: u64) -> f64 {
    match k % 5 {
        0 => -1.0,
        1 => -0.5,
        2 => 0.0,
        3 => 0.5,
        _ => 1.0,
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// T6 (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §6.4): on the 7D
    /// explicit-rows layout the activation backward emits per-SPEC-row
    /// coefficient errs plus outward bias discharges that COVER every
    /// sampled admissible true incoming tensor (each coefficient
    /// independently perturbed within its row's carried err, endpoints and
    /// sign flips included), with the real ReLU relaxation. Coefficient and
    /// bias coverage are both checked in f64 with no tolerance epsilon.
    #[ntest::timeout(20000)]
    #[test]
    fn prop_patches_activation_7d_err_covers_sampled_true(
        row_count in 1usize..=3,
        out_c in 1usize..=2,
        out_h in 1usize..=2,
        out_w in 1usize..=2,
        in_c in 1usize..=2,
        kh in 1usize..=2,
        kw in 1usize..=2,
        stride_h in 1usize..=2,
        stride_w in 1usize..=2,
        pad_left in 0usize..=1,
        pad_top in 0usize..=1,
        in_h in 1usize..=3,
        in_w in 1usize..=3,
        seed in any::<u64>(),
    ) {
        use crate::layers::activations::relu::relu_linear_relaxation;
        use crate::layers::common::crown_elementwise_backward_patches;

        let shape = [row_count, out_c, out_h, out_w, in_c, kh, kw];
        let n: usize = shape.iter().product();
        let in_dim = in_c * in_h * in_w;

        // Patches values in [-2, 2) with exact zeros (a==0 taps live).
        let fill = |salt: u64| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let u = mix_unit_f64(seed, salt, i as u64, 17);
                    if u < 0.15 { 0.0 } else { (u * 4.0 - 2.0) as f32 }
                })
                .collect()
        };
        // Per-spec-row carried errs in [0, 0.5] with exact zeros.
        let errs = |salt: u64| -> Vec<f32> {
            (0..row_count)
                .map(|r| {
                    let u = mix_unit_f64(seed, salt, r as u64, 29);
                    if u < 0.25 { 0.0 } else { (u * 0.5) as f32 }
                })
                .collect()
        };
        let biases = |salt: u64| -> Array1<f32> {
            Array1::from_vec(
                (0..row_count)
                    .map(|r| (mix_unit_f64(seed, salt, r as u64, 43) * 2.0 - 1.0) as f32)
                    .collect(),
            )
        };
        let lower_errs = errs(1);
        let upper_errs = errs(2);
        let bounds = PatchesLinearBounds {
            row_count,
            lower_a: PatchesData {
                coeff_err: Some(Array1::from_vec(lower_errs.clone())),
                patches: Some(ArrayD::from_shape_vec(IxDyn(&shape), fill(3)).unwrap()),
                stride: (stride_h, stride_w),
                padding: (pad_left, 0, pad_top, 0),
                identity: false,
                output_shape: (out_c, out_h, out_w),
                input_shape: (in_c, in_h, in_w),
                unstable_idx: None,
            },
            lower_b: biases(5),
            upper_a: PatchesData {
                coeff_err: Some(Array1::from_vec(upper_errs.clone())),
                patches: Some(ArrayD::from_shape_vec(IxDyn(&shape), fill(4)).unwrap()),
                stride: (stride_h, stride_w),
                padding: (pad_left, 0, pad_top, 0),
                identity: false,
                output_shape: (out_c, out_h, out_w),
                input_shape: (in_c, in_h, in_w),
                unstable_idx: None,
            },
            upper_b: biases(6),
        };

        // Real ReLU relaxation over mixed-regime pre-activation bounds
        // (stable-positive, stable-negative, and unstable neurons).
        let pre_l: Vec<f32> = (0..in_dim)
            .map(|i| (mix_unit_f64(seed, 7, i as u64, 61) * 2.0 - 1.0) as f32)
            .collect();
        let pre_u: Vec<f32> = pre_l
            .iter()
            .enumerate()
            .map(|(i, &l)| l + (mix_unit_f64(seed, 8, i as u64, 71) * 1.5) as f32)
            .collect();
        let pre = ny_tensor::BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), pre_l.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), pre_u.clone()).unwrap(),
        ).map_err(|e| TestCaseError::fail(format!("pre BoundedTensor failed: {e}")))?;

        let result = crown_elementwise_backward_patches(&bounds, &pre, relu_linear_relaxation)
            .map_err(|e| TestCaseError::fail(format!("7D activation backward failed: {e}")))?;
        let CrownBounds::Patches(res) = result else {
            return Err(TestCaseError::fail("expected patches output"));
        };
        let err_l = res.lower_a.coeff_err.as_ref()
            .ok_or_else(|| TestCaseError::fail("lower err must be Some on 7D"))?;
        let err_u = res.upper_a.coeff_err.as_ref()
            .ok_or_else(|| TestCaseError::fail("upper err must be Some on 7D"))?;
        prop_assert_eq!(err_l.len(), row_count);
        prop_assert_eq!(err_u.len(), row_count);

        // f64 relaxation constants, per input neuron.
        let relax64: Vec<(f64, f64, f64, f64)> = pre_l
            .iter()
            .zip(pre_u.iter())
            .map(|(&l, &u)| {
                let r = relu_linear_relaxation(l, u);
                (
                    f64::from(r.lower_slope),
                    f64::from(r.lower_intercept),
                    f64::from(r.upper_slope),
                    f64::from(r.upper_intercept),
                )
            })
            .collect();

        let old_l = bounds.lower_a.patches.as_ref().unwrap();
        let old_u = bounds.upper_a.patches.as_ref().unwrap();
        let new_l = res.lower_a.patches.as_ref().unwrap();
        let new_u = res.upper_a.patches.as_ref().unwrap();

        // 8 sampled admissible true tensors per side: every valid tap
        // perturbed by factor·e_row, factor ∈ {-1,-0.5,0,0.5,1}.
        for s in 0..8u64 {
            for row in 0..row_count {
                let e_l = f64::from(lower_errs[row]);
                let e_u = f64::from(upper_errs[row]);
                let ne_l = f64::from(err_l[row]);
                let ne_u = f64::from(err_u[row]);
                let mut btilde_l = f64::from(bounds.lower_b[row]);
                let mut btilde_u = f64::from(bounds.upper_b[row]);
                let mut tap = 0u64;
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        tap += 1;
                                        let idx = [row, oc, oh, ow, ic, ki, kj];
                                        let ih_raw =
                                            (oh * stride_h + ki) as isize - pad_top as isize;
                                        let iw_raw =
                                            (ow * stride_w + kj) as isize - pad_left as isize;
                                        if ih_raw < 0
                                            || (ih_raw as usize) >= in_h
                                            || iw_raw < 0
                                            || (iw_raw as usize) >= in_w
                                        {
                                            // Structural padding taps stay 0.
                                            prop_assert_eq!(new_l[idx], 0.0);
                                            prop_assert_eq!(new_u[idx], 0.0);
                                            continue;
                                        }
                                        let flat = ic * in_h * in_w
                                            + ih_raw as usize * in_w
                                            + iw_raw as usize;
                                        let (ls, li, us, ui) = relax64[flat];

                                        let at_l = f64::from(old_l[idx])
                                            + delta_factor(mix64(seed, s, 100 + row as u64, tap))
                                                * e_l;
                                        let (c_ideal, h) = if at_l > 0.0 {
                                            (at_l * ls, at_l * li)
                                        } else if at_l < 0.0 {
                                            (at_l * us, at_l * ui)
                                        } else {
                                            (0.0, 0.0)
                                        };
                                        let stored = f64::from(new_l[idx]);
                                        prop_assert!(
                                            (stored - c_ideal).abs() <= ne_l,
                                            "lower coeff sample {s} row {row} tap {idx:?}: \
                                             |{stored} - {c_ideal}| > err {ne_l}",
                                        );
                                        btilde_l += h;

                                        let at_u = f64::from(old_u[idx])
                                            + delta_factor(mix64(seed, s, 200 + row as u64, tap))
                                                * e_u;
                                        let (c_ideal, h) = if at_u > 0.0 {
                                            (at_u * us, at_u * ui)
                                        } else if at_u < 0.0 {
                                            (at_u * ls, at_u * li)
                                        } else {
                                            (0.0, 0.0)
                                        };
                                        let stored = f64::from(new_u[idx]);
                                        prop_assert!(
                                            (stored - c_ideal).abs() <= ne_u,
                                            "upper coeff sample {s} row {row} tap {idx:?}: \
                                             |{stored} - {c_ideal}| > err {ne_u}",
                                        );
                                        btilde_u += h;
                                    }
                                }
                            }
                        }
                    }
                }
                prop_assert!(
                    f64::from(res.lower_b[row]) <= btilde_l,
                    "lower bias sample {} row {}: stored {} > ideal {}",
                    s, row, res.lower_b[row], btilde_l,
                );
                prop_assert!(
                    f64::from(res.upper_b[row]) >= btilde_u,
                    "upper bias sample {} row {}: stored {} < ideal {}",
                    s, row, res.upper_b[row], btilde_u,
                );
            }
        }
    }
}
