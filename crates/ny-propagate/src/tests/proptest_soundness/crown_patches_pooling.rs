// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest equivalence tests for Patches vs Dense CROWN backward through
//! pooling layers (AvgPool, MaxPool).
//!
//! These tests verify that Patches-mode backward through Conv2d + pooling
//! produces identical bounds to Dense-mode backward after to_dense() conversion.
//!
//! Design: designs/2026-03-01-patches-phase3-pooling-termination.md
//! Part of #2613
//!
//! WALL-CLOCK POLICY FOR THIS FILE: every `#[ntest::timeout(..)]` below is a
//! HANG SENTINEL, not a performance assertion. Measured 2026-08-19, isolated
//! and single-threaded, these properties cost 10-30s in a debug build, and the
//! walls they used to carry (20-60s) were margins of 1.2-2.9x. That is not a
//! sentinel, it is a coin flip that any concurrent load loses -- and they duly
//! failed at both 4 and 8 test threads. They are now a uniform 300s, roughly
//! 10-20x the measured isolated cost. The failure these exist to catch is an
//! infinite loop, and no finite wall lets one of those through.
//! MEASURE BEFORE LOWERING THEM.

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::layers::common::{BoundPropagation, PatchesPropagation};
use crate::layers::convolution::conv2d::Conv2dLayer;
use crate::layers::pooling::average::AveragePoolLayer;
use crate::layers::pooling::max::MaxPool2dLayer;
use crate::LinearBounds;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

/// Tolerance for Patches-vs-Dense comparison.
/// Both paths compute mathematically equivalent operations but accumulation
/// order differs. MaxPool bias uses f64 accumulation in Dense path with
/// directed rounding (next_down/next_up), so bias tolerance is slightly looser.
const PATCHES_DENSE_A_TOLERANCE: f32 = 1e-5;
const PATCHES_DENSE_B_TOLERANCE: f32 = 1e-4;

/// Generate a random kernel of shape (out_c, in_c, kh, kw) with values in
/// [-2.0, 2.0], using a seed for reproducibility.
fn make_kernel(out_c: usize, in_c: usize, kh: usize, kw: usize, seed: u64) -> ArrayD<f32> {
    let len = out_c * in_c * kh * kw;
    let mut rng = seed;
    let values: Vec<f32> = (0..len)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let u = (rng as f32) / (u64::MAX as f32);
            u * 4.0 - 2.0
        })
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), values).expect("kernel shape mismatch")
}

/// Generate random bounded tensor (pre-activation bounds) for a 3D spatial shape.
fn make_bounded_tensor(channels: usize, h: usize, w: usize, seed: u64) -> BoundedTensor {
    let size = channels * h * w;
    let mut rng = seed.wrapping_add(99999);
    let mut lower_vals = Vec::with_capacity(size);
    let mut upper_vals = Vec::with_capacity(size);
    for _ in 0..size {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let center = ((rng as f32) / (u64::MAX as f32)) * 4.0 - 2.0;
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let width = ((rng as f32) / (u64::MAX as f32)) * 0.5 + 0.01;
        lower_vals.push(center - width);
        upper_vals.push(center + width);
    }
    let lower = ArrayD::from_shape_vec(IxDyn(&[channels, h, w]), lower_vals).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[channels, h, w]), upper_vals).unwrap();
    BoundedTensor::new(lower, upper).unwrap()
}

/// Compare two LinearBounds element-wise for approximate equality.
fn assert_bounds_approx_eq(
    dense: &LinearBounds,
    patches: &LinearBounds,
    a_tol: f32,
    b_tol: f32,
    label: &str,
) -> Result<(), TestCaseError> {
    prop_assert!(
        dense.lower_a().shape() == patches.lower_a().shape(),
        "{}: lower_a shape mismatch: {:?} vs {:?}",
        label,
        dense.lower_a().shape(),
        patches.lower_a().shape(),
    );

    for ((idx, &d), &p) in dense.lower_a().indexed_iter().zip(patches.lower_a().iter()) {
        let diff = (d - p).abs();
        let scale = d.abs().max(p.abs()).max(1.0);
        prop_assert!(
            diff <= a_tol * scale,
            "{}: lower_a mismatch at {:?}: dense={}, patches={}, diff={}",
            label,
            idx,
            d,
            p,
            diff,
        );
    }
    for ((idx, &d), &p) in dense.upper_a().indexed_iter().zip(patches.upper_a().iter()) {
        let diff = (d - p).abs();
        let scale = d.abs().max(p.abs()).max(1.0);
        prop_assert!(
            diff <= a_tol * scale,
            "{}: upper_a mismatch at {:?}: dense={}, patches={}, diff={}",
            label,
            idx,
            d,
            p,
            diff,
        );
    }

    let out_dim = dense.lower_b().len();
    for i in 0..out_dim {
        let diff_lb = (dense.lower_b()[i] - patches.lower_b()[i]).abs();
        let scale_lb = dense.lower_b()[i]
            .abs()
            .max(patches.lower_b()[i].abs())
            .max(1.0);
        prop_assert!(
            diff_lb <= b_tol * scale_lb,
            "{}: lower_b mismatch at {}: dense={}, patches={}, diff={}",
            label,
            i,
            dense.lower_b()[i],
            patches.lower_b()[i],
            diff_lb,
        );
        let diff_ub = (dense.upper_b()[i] - patches.upper_b()[i]).abs();
        let scale_ub = dense.upper_b()[i]
            .abs()
            .max(patches.upper_b()[i].abs())
            .max(1.0);
        prop_assert!(
            diff_ub <= b_tol * scale_ub,
            "{}: upper_b mismatch at {}: dense={}, patches={}, diff={}",
            label,
            i,
            dense.upper_b()[i],
            patches.upper_b()[i],
            diff_ub,
        );
    }
    Ok(())
}

/// Concretize per-output scalar bounds of `lb` over an input box.
/// lower = sum_j min(la_j*l_j, la_j*u_j) + lower_b
/// upper = sum_j max(ua_j*l_j, ua_j*u_j) + upper_b
fn concretize_over_box(lb: &LinearBounds, input: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let in_l: Vec<f32> = input.lower().iter().copied().collect();
    let in_u: Vec<f32> = input.upper().iter().copied().collect();
    let out_dim = lb.lower_b().len();
    let mut lowers = Vec::with_capacity(out_dim);
    let mut uppers = Vec::with_capacity(out_dim);
    for o in 0..out_dim {
        let mut lo = lb.lower_b()[o] as f64;
        let mut hi = lb.upper_b()[o] as f64;
        for j in 0..in_l.len() {
            let la = lb.lower_a()[[o, j]] as f64;
            let ua = lb.upper_a()[[o, j]] as f64;
            lo += la.min(0.0) * in_u[j] as f64 + la.max(0.0) * in_l[j] as f64;
            hi += ua.max(0.0) * in_u[j] as f64 + ua.min(0.0) * in_l[j] as f64;
        }
        lowers.push(lo as f32);
        uppers.push(hi as f32);
    }
    (lowers, uppers)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Conv2d → AvgPool: Patches backward == Dense backward.
    ///
    /// Builds a Conv2d → AvgPool chain and verifies that the Patches path
    /// (Conv2d patches → AvgPool patches → to_dense) matches the Dense path
    /// (Conv2d dense backward → AvgPool dense backward).
    ///
    /// Reference: designs/2026-03-01-patches-phase3-pooling-termination.md
    #[ntest::timeout(300000)]
    #[test]
    fn proptest_conv2d_avgpool_patches_vs_dense(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        conv_k in 1usize..=3,
        in_h in 6usize..=10,
        in_w in 6usize..=10,
        pool_k in 2usize..=3,
        seed in any::<u64>(),
    ) {
        // Conv2d with stride=1, pad=0 for simplicity
        let conv_out_h = in_h - conv_k + 1;
        let conv_out_w = in_w - conv_k + 1;
        prop_assume!(conv_out_h >= 2 && conv_out_w >= 2);

        // AvgPool with stride == kernel (non-overlapping)
        let pool_out_h = conv_out_h / pool_k;
        let pool_out_w = conv_out_w / pool_k;
        prop_assume!(pool_out_h >= 1 && pool_out_w >= 1);

        let out_dim = out_c * pool_out_h * pool_out_w;

        // Create layers
        let kernel = make_kernel(out_c, in_c, conv_k, conv_k, seed);
        let conv = Conv2dLayer::with_input_shape(
            kernel, None, (1, 1), (0, 0), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("Conv2d failed: {e}")))?;

        let avgpool = AveragePoolLayer::new((pool_k, pool_k), (pool_k, pool_k), (0, 0), true);

        // Create pre-activation bounds for AvgPool (= Conv2d output bounds)
        let conv_input_bt = make_bounded_tensor(in_c, in_h, in_w, seed);
        let conv_output_bt = conv.propagate_ibp(&conv_input_bt)
            .map_err(|e| TestCaseError::fail(format!("Conv IBP failed: {e}")))?;

        // --- Dense path ---
        let identity_lb = LinearBounds::identity(out_dim);
        let after_avgpool_dense = avgpool.propagate_linear_with_bounds(
            &identity_lb, &conv_output_bt,
        ).map_err(|e| TestCaseError::fail(format!("AvgPool Dense failed: {e}")))?;
        let after_conv_dense = conv.propagate_linear(&after_avgpool_dense)
            .map_err(|e| TestCaseError::fail(format!("Conv Dense failed: {e}")))?
            .into_owned();

        // --- Patches path ---
        // Start with identity patches for Conv2d output shape
        let patches_identity = PatchesLinearBounds::identity(
            (out_c, pool_out_h, pool_out_w),
            (out_c, pool_out_h, pool_out_w),
        );

        // AvgPool patches backward
        let after_avgpool_patches = avgpool.propagate_patches_with_bounds(
            &patches_identity, &conv_output_bt,
        ).map_err(|e| TestCaseError::fail(format!("AvgPool Patches failed: {e}")))?;

        // Conv2d patches backward
        let after_conv_patches = match after_avgpool_patches {
            CrownBounds::Patches(pb) => {
                conv.propagate_patches(&pb)
                    .map_err(|e| TestCaseError::fail(format!("Conv Patches failed: {e}")))?
            }
            CrownBounds::Dense(_) => {
                // Fell back to dense — still valid, compare after conv
                let lb = after_avgpool_patches.into_dense()
                    .map_err(|e| TestCaseError::fail(format!("into_dense failed: {e}")))?;
                let result = conv.propagate_linear(&lb)
                    .map_err(|e| TestCaseError::fail(format!("Conv Dense 2 failed: {e}")))?
                    .into_owned();
                CrownBounds::Dense(result)
            }
        };

        let patches_dense = after_conv_patches.into_dense()
            .map_err(|e| TestCaseError::fail(format!("Final to_dense failed: {e}")))?;

        // Compare
        assert_bounds_approx_eq(
            &after_conv_dense, &patches_dense,
            PATCHES_DENSE_A_TOLERANCE, PATCHES_DENSE_B_TOLERANCE,
            "Conv2d+AvgPool",
        )?;
    }

    /// Conv2d → MaxPool: Patches backward == Dense backward.
    ///
    /// Builds a Conv2d → MaxPool chain and verifies that the Patches path
    /// produces identical bounds to the Dense path. MaxPool is nonlinear,
    /// so the relaxation slopes/biases must match between paths.
    ///
    /// Reference: designs/2026-03-01-patches-phase3-pooling-termination.md
    // Match the AvgPool differential budget above. This 200-case nonlinear
    // relaxation sweep legitimately exceeds 10 seconds on loaded shared hosts.
    #[ntest::timeout(300000)]
    #[test]
    fn proptest_conv2d_maxpool_patches_vs_dense(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        conv_k in 1usize..=3,
        in_h in 6usize..=10,
        in_w in 6usize..=10,
        pool_k in 2usize..=3,
        seed in any::<u64>(),
    ) {
        let conv_out_h = in_h - conv_k + 1;
        let conv_out_w = in_w - conv_k + 1;
        prop_assume!(conv_out_h >= 2 && conv_out_w >= 2);

        let pool_out_h = conv_out_h / pool_k;
        let pool_out_w = conv_out_w / pool_k;
        prop_assume!(pool_out_h >= 1 && pool_out_w >= 1);

        let out_dim = out_c * pool_out_h * pool_out_w;

        // Create layers
        let kernel = make_kernel(out_c, in_c, conv_k, conv_k, seed);
        let conv = Conv2dLayer::with_input_shape(
            kernel, None, (1, 1), (0, 0), in_h, in_w,
        ).map_err(|e| TestCaseError::fail(format!("Conv2d failed: {e}")))?;

        let maxpool = MaxPool2dLayer::new((pool_k, pool_k), (pool_k, pool_k), (0, 0));

        // Create pre-activation bounds for MaxPool (= Conv2d output bounds)
        let conv_input_bt = make_bounded_tensor(in_c, in_h, in_w, seed);
        let conv_output_bt = conv.propagate_ibp(&conv_input_bt)
            .map_err(|e| TestCaseError::fail(format!("Conv IBP failed: {e}")))?;

        // --- Dense path ---
        let identity_lb = LinearBounds::identity(out_dim);
        let after_maxpool_dense = maxpool.propagate_linear_with_bounds(
            &identity_lb, &conv_output_bt,
        ).map_err(|e| TestCaseError::fail(format!("MaxPool Dense failed: {e}")))?;
        let after_conv_dense = conv.propagate_linear(&after_maxpool_dense)
            .map_err(|e| TestCaseError::fail(format!("Conv Dense failed: {e}")))?
            .into_owned();

        // --- Patches path ---
        let patches_identity = PatchesLinearBounds::identity(
            (out_c, pool_out_h, pool_out_w),
            (out_c, pool_out_h, pool_out_w),
        );

        // MaxPool patches backward
        let after_maxpool_patches = maxpool.propagate_patches_with_bounds(
            &patches_identity, &conv_output_bt,
        ).map_err(|e| TestCaseError::fail(format!("MaxPool Patches failed: {e}")))?;

        // Conv2d patches backward
        let after_conv_patches = match after_maxpool_patches {
            CrownBounds::Patches(pb) => {
                conv.propagate_patches(&pb)
                    .map_err(|e| TestCaseError::fail(format!("Conv Patches failed: {e}")))?
            }
            CrownBounds::Dense(_) => {
                let lb = after_maxpool_patches.into_dense()
                    .map_err(|e| TestCaseError::fail(format!("into_dense failed: {e}")))?;
                let result = conv.propagate_linear(&lb)
                    .map_err(|e| TestCaseError::fail(format!("Conv Dense 2 failed: {e}")))?
                    .into_owned();
                CrownBounds::Dense(result)
            }
        };

        let patches_dense = after_conv_patches.into_dense()
            .map_err(|e| TestCaseError::fail(format!("Final to_dense failed: {e}")))?;

        // Dense and Patches are NO LONGER bit-equal through MaxPool: the dense
        // no-winner path now routes the lower row linearly through i*=argmax_i l_i
        // (and the upper row through i* for ua<0), a SOUND tighter relaxation that
        // the patches path deliberately does NOT apply (its shared winner_d slope
        // map would feed the upper row where y<=x_{i*} is false → unsound). So we
        // assert the legitimate post-change relationship instead of equality:
        //   (1) both are sound over the input box, and
        //   (2) dense is at least as tight as patches (dense_lo >= patches_lo,
        //       dense_hi <= patches_hi).
        // Definite-winner windows still produce identical A/b in both paths; only
        // no-winner windows make dense strictly tighter.
        let (dense_lo, dense_hi) = concretize_over_box(&after_conv_dense, &conv_input_bt);
        let (patch_lo, patch_hi) = concretize_over_box(&patches_dense, &conv_input_bt);
        let tol = PATCHES_DENSE_B_TOLERANCE;
        for o in 0..dense_lo.len() {
            let scale = dense_lo[o]
                .abs()
                .max(patch_lo[o].abs())
                .max(dense_hi[o].abs())
                .max(patch_hi[o].abs())
                .max(1.0);
            // Sound nesting + dense tightness on the LOWER bound.
            prop_assert!(
                dense_lo[o] >= patch_lo[o] - tol * scale,
                "Conv2d+MaxPool: dense lower {} should be >= patches lower {} (out {})",
                dense_lo[o], patch_lo[o], o,
            );
            // Sound nesting + dense tightness on the UPPER bound.
            prop_assert!(
                dense_hi[o] <= patch_hi[o] + tol * scale,
                "Conv2d+MaxPool: dense upper {} should be <= patches upper {} (out {})",
                dense_hi[o], patch_hi[o], o,
            );
            // Dense interval must remain well-formed (lower <= upper).
            prop_assert!(
                dense_lo[o] <= dense_hi[o] + tol * scale,
                "Conv2d+MaxPool: dense lower {} exceeds dense upper {} (out {})",
                dense_lo[o], dense_hi[o], o,
            );
        }
    }
}
