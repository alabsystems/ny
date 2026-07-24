// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{array, Array3, Array4};
use proptest::prelude::*;

use crate::multi_norm::MultiNormBounds;

mod assertion_hygiene;

#[test]
fn test_concretize_l2_single_word() {
    let lw = Array4::from_shape_vec((1, 1, 2, 1), vec![3.0, 4.0]).unwrap();
    let uw = lw.clone();
    let lb = Array3::from_shape_vec((1, 1, 1), vec![1.0]).unwrap();
    let ub = lb.clone();
    let bounds = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub).unwrap();
    let concretized = bounds.concretize().unwrap();
    assert_eq!(concretized.lower()[[0, 0, 0]], -4.0);
    assert_eq!(concretized.upper()[[0, 0, 0]], 6.0);
}

#[test]
fn test_concretize_l1_dual_is_linf() {
    let lw = Array4::from_shape_vec((1, 1, 3, 1), vec![1.0, -5.0, 2.0]).unwrap();
    let uw = lw.clone();
    let lb = Array3::from_shape_vec((1, 1, 1), vec![0.0]).unwrap();
    let ub = lb.clone();
    let bounds = MultiNormBounds::new(1.0, 2.0, 1, lw, lb, uw, ub).unwrap();
    let concretized = bounds.concretize().unwrap();
    assert_eq!(concretized.lower()[[0, 0, 0]], -10.0);
    assert_eq!(concretized.upper()[[0, 0, 0]], 10.0);
}

#[test]
fn test_concretize_splits_perturbed_words() {
    let lw = Array4::from_shape_vec((1, 1, 4, 1), vec![3.0, 4.0, 0.0, 5.0]).unwrap();
    let uw = lw.clone();
    let lb = Array3::from_shape_vec((1, 1, 1), vec![1.0]).unwrap();
    let ub = lb.clone();
    let bounds = MultiNormBounds::new(2.0, 1.0, 2, lw, lb, uw, ub).unwrap();
    let concretized = bounds.concretize().unwrap();
    // Two chunks: [3,4] => 5, [0,5] => 5, total = 10.
    assert_eq!(concretized.lower()[[0, 0, 0]], -9.0);
    assert_eq!(concretized.upper()[[0, 0, 0]], 11.0);
}

#[test]
fn test_add_bias() {
    let lw = Array4::zeros((1, 1, 1, 1));
    let uw = lw.clone();
    let lb = Array3::from_shape_vec((1, 1, 1), vec![2.0]).unwrap();
    let ub = lb.clone();
    let bounds = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub).unwrap();
    let bias = array![[[1.5]]];
    let shifted = bounds.add_bias(&bias).unwrap();
    assert_eq!(shifted.lb[[0, 0, 0]], 3.5);
    assert_eq!(shifted.ub[[0, 0, 0]], 3.5);
}

#[test]
fn test_transpose_len_out_swaps_axes() {
    let lw = Array4::from_shape_vec((1, 2, 1, 3), (0..6).map(|v| v as f32).collect()).unwrap();
    let uw = lw.clone();
    let lb = Array3::from_shape_vec((1, 2, 3), (0..6).map(|v| v as f32).collect()).unwrap();
    let ub = lb.clone();
    let bounds = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub).unwrap();
    let transposed = bounds.transpose_len_out().unwrap();
    assert_eq!(transposed.lw.shape(), &[1, 3, 1, 2]);
    assert_eq!(transposed.lb.shape(), &[1, 3, 2]);
    assert_eq!(transposed.lw[[0, 2, 0, 1]], bounds.lw[[0, 1, 0, 2]]);
    assert_eq!(transposed.lb[[0, 2, 1]], bounds.lb[[0, 1, 2]]);
}

#[test]
fn test_scale_negative_swaps_bounds() {
    let lw = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0]).unwrap();
    let uw = Array4::from_shape_vec((1, 1, 1, 1), vec![2.0]).unwrap();
    let lb = Array3::from_shape_vec((1, 1, 1), vec![3.0]).unwrap();
    let ub = Array3::from_shape_vec((1, 1, 1), vec![4.0]).unwrap();
    let bounds = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub).unwrap();
    let scaled = bounds.scale(-2.0);
    assert_eq!(scaled.lw[[0, 0, 0, 0]], -4.0);
    assert_eq!(scaled.uw[[0, 0, 0, 0]], -2.0);
    assert_eq!(scaled.lb[[0, 0, 0]], -8.0);
    assert_eq!(scaled.ub[[0, 0, 0]], -6.0);
}

/// Regression: scale(NaN) must return conservative fallback bounds, not NaN arrays.
/// Prior to this fix, NaN fell through to the negative branch producing all-NaN.
#[test]
fn test_scale_nan_returns_conservative_fallback() {
    let lw = Array4::from_shape_vec((1, 1, 1, 1), vec![1.0])
        .expect("invariant: shape matches element count");
    let uw = Array4::from_shape_vec((1, 1, 1, 1), vec![2.0])
        .expect("invariant: shape matches element count");
    let lb = Array3::from_shape_vec((1, 1, 1), vec![3.0])
        .expect("invariant: shape matches element count");
    let ub = Array3::from_shape_vec((1, 1, 1), vec![4.0])
        .expect("invariant: shape matches element count");
    let bounds = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub)
        .expect("invariant: valid MultiNormBounds construction");

    let scaled = bounds.scale(f32::NAN);

    // Weights must be zeroed (no input dependence).
    assert_eq!(scaled.lw[[0, 0, 0, 0]], 0.0, "NaN scale: lw must be zero");
    assert_eq!(scaled.uw[[0, 0, 0, 0]], 0.0, "NaN scale: uw must be zero");

    // Biases must be conservative fallback bounds.
    assert!(
        scaled.lb[[0, 0, 0]] < 0.0,
        "NaN scale: lb must be negative fallback"
    );
    assert!(
        scaled.ub[[0, 0, 0]] > 0.0,
        "NaN scale: ub must be positive fallback"
    );
    assert!(
        scaled.lb[[0, 0, 0]].is_finite(),
        "NaN scale: lb must be finite"
    );
    assert!(
        scaled.ub[[0, 0, 0]].is_finite(),
        "NaN scale: ub must be finite"
    );
    assert!(
        scaled.lb[[0, 0, 0]] <= scaled.ub[[0, 0, 0]],
        "NaN scale: bounds must not be inverted"
    );
}

#[test]
fn test_from_input_embeddings_single_word() {
    let embeddings = array![[[1.0, -2.0], [0.5, 0.25], [-1.5, 3.0]]];
    let bounds = MultiNormBounds::from_input_embeddings(&embeddings, 2.0, 0.5, 1, &[1]).unwrap();
    let concretized = bounds.concretize().unwrap();
    // Perturbed word (index 1): +/- eps per dimension.
    assert_eq!(concretized.lower()[[0, 1, 0]], 0.0);
    assert_eq!(concretized.upper()[[0, 1, 0]], 1.0);
    assert_eq!(concretized.lower()[[0, 1, 1]], -0.25);
    assert_eq!(concretized.upper()[[0, 1, 1]], 0.75);
    // Unperturbed words remain unchanged.
    assert_eq!(concretized.lower()[[0, 0, 0]], 1.0);
    assert_eq!(concretized.upper()[[0, 0, 0]], 1.0);
    assert_eq!(concretized.lower()[[0, 2, 1]], 3.0);
    assert_eq!(concretized.upper()[[0, 2, 1]], 3.0);
}

#[test]
fn test_from_input_embeddings_multi_word() {
    let embeddings = array![[[2.0, 0.0], [1.0, -1.0], [0.0, 4.0]]];
    let bounds =
        MultiNormBounds::from_input_embeddings(&embeddings, 2.0, 0.25, 2, &[0, 2]).unwrap();
    let concretized = bounds.concretize().unwrap();
    // Word 0 perturbed.
    assert_eq!(concretized.lower()[[0, 0, 0]], 1.75);
    assert_eq!(concretized.upper()[[0, 0, 0]], 2.25);
    assert_eq!(concretized.lower()[[0, 0, 1]], -0.25);
    assert_eq!(concretized.upper()[[0, 0, 1]], 0.25);
    // Word 2 perturbed.
    assert_eq!(concretized.lower()[[0, 2, 0]], -0.25);
    assert_eq!(concretized.upper()[[0, 2, 0]], 0.25);
    assert_eq!(concretized.lower()[[0, 2, 1]], 3.75);
    assert_eq!(concretized.upper()[[0, 2, 1]], 4.25);
    // Word 1 unperturbed.
    assert_eq!(concretized.lower()[[0, 1, 0]], 1.0);
    assert_eq!(concretized.upper()[[0, 1, 0]], 1.0);
    assert_eq!(concretized.lower()[[0, 1, 1]], -1.0);
    assert_eq!(concretized.upper()[[0, 1, 1]], -1.0);
}

#[test]
fn test_matmul_batched_matches_unbatched() {
    let lw = Array4::from_shape_vec((2, 1, 1, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let uw = lw.clone();
    let lb = Array3::from_shape_vec((2, 1, 2), vec![0.5, 1.0, 1.5, 2.0]).unwrap();
    let ub = lb.clone();
    let bounds = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub).unwrap();
    let weight = array![[1.0, -1.0], [2.0, 0.5]];
    let weight_batched = array![[[1.0, -1.0], [2.0, 0.5]], [[1.0, -1.0], [2.0, 0.5]]];
    let unbatched = bounds.matmul(&weight).unwrap();
    let batched = bounds.matmul_batched(&weight_batched).unwrap();
    assert_eq!(batched.lw, unbatched.lw);
    assert_eq!(batched.uw, unbatched.uw);
    assert_eq!(batched.lb, unbatched.lb);
    assert_eq!(batched.ub, unbatched.ub);
}

#[test]
fn test_dot_product_dim_in_one_matches_interval_product() {
    let lw = Array4::zeros((1, 1, 1, 1));
    let uw = lw.clone();
    let lb = Array3::from_shape_vec((1, 1, 1), vec![1.0]).unwrap();
    let ub = Array3::from_shape_vec((1, 1, 1), vec![2.0]).unwrap();
    let a = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub).unwrap();

    let lw_b = Array4::zeros((1, 1, 1, 1));
    let uw_b = lw_b.clone();
    let lb_b = Array3::from_shape_vec((1, 1, 1), vec![3.0]).unwrap();
    let ub_b = Array3::from_shape_vec((1, 1, 1), vec![4.0]).unwrap();
    let b = MultiNormBounds::new(2.0, 1.0, 1, lw_b, lb_b, uw_b, ub_b).unwrap();

    let out = a.dot_product(&b).unwrap();
    // dot_product() uses concretize_sound() internally (#2239), which widens
    // intermediate bounds by 1 ULP via directed rounding. The resulting bias
    // bounds must *contain* the true interval product [1,2]×[3,4] = [3,8]
    // but may be slightly wider than exact.
    assert!(
        out.lb[[0, 0, 0]] <= 3.0,
        "lower bound {} must not exceed true product lower 3.0",
        out.lb[[0, 0, 0]]
    );
    assert!(
        out.ub[[0, 0, 0]] >= 8.0,
        "upper bound {} must not be below true product upper 8.0",
        out.ub[[0, 0, 0]]
    );
    // Bounds should be tight (within a few ULPs of exact).
    assert!(
        (out.lb[[0, 0, 0]] - 3.0).abs() < 1e-4,
        "lower bound {} too far from exact 3.0",
        out.lb[[0, 0, 0]]
    );
    assert!(
        (out.ub[[0, 0, 0]] - 8.0).abs() < 1e-4,
        "upper bound {} too far from exact 8.0",
        out.ub[[0, 0, 0]]
    );
}

#[test]
fn test_mul_elementwise_concrete_matches_product() {
    let lw = Array4::zeros((1, 1, 1, 1));
    let uw = lw.clone();
    let lb = Array3::from_shape_vec((1, 1, 1), vec![2.0]).unwrap();
    let ub = lb.clone();
    let a = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub).unwrap();

    let lw_b = Array4::zeros((1, 1, 1, 1));
    let uw_b = lw_b.clone();
    let lb_b = Array3::from_shape_vec((1, 1, 1), vec![3.0]).unwrap();
    let ub_b = lb_b.clone();
    let b = MultiNormBounds::new(2.0, 1.0, 1, lw_b, lb_b, uw_b, ub_b).unwrap();

    let out = a.mul_elementwise(&b).unwrap();
    assert_eq!(out.lb[[0, 0, 0]], 6.0);
    assert_eq!(out.ub[[0, 0, 0]], 6.0);
}

/// Regression #3116: NaN weights must propagate through matmul, not silently become zero.
///
/// Prior to this fix, `v.max(0.0)` / `v.min(0.0)` followed IEEE 754-2008 and
/// returned 0.0 for NaN inputs, silently dropping the weight's contribution and
/// producing unsound (too-tight) bounds. After the fix, `nan_propagating_max_zero`
/// / `nan_propagating_min_zero` preserve NaN, which propagates through the dot
/// product to produce NaN in the output bounds — a detectable failure.
#[test]
fn test_matmul_nan_weight_propagates_not_silently_zeroed_3116() {
    // 1×1×1×2 bounds (batch=1, length=1, dim_in=1, dim_out=2).
    let lw = Array4::from_shape_vec((1, 1, 1, 2), vec![1.0, 1.0])
        .expect("invariant: shape matches element count");
    let uw = lw.clone();
    let lb = Array3::from_shape_vec((1, 1, 2), vec![1.0, 1.0])
        .expect("invariant: shape matches element count");
    let ub = lb.clone();
    let bounds = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub)
        .expect("invariant: valid MultiNormBounds construction");

    // Weight matrix 2×2 with one NaN entry.
    let weight = array![[1.0, 0.0], [f32::NAN, 1.0]];
    let result = bounds
        .matmul(&weight)
        .expect("matmul should not error on NaN weights");

    // The NaN weight must propagate into the output bounds, NOT silently become zero.
    // Column 0 of the weight has [1.0, NaN], so the first output neuron's bounds
    // must contain NaN (from the NaN weight's contribution).
    assert!(
        result.lb[[0, 0, 0]].is_nan(),
        "lb[0] should be NaN due to NaN weight, got {}",
        result.lb[[0, 0, 0]]
    );
    assert!(
        result.ub[[0, 0, 0]].is_nan(),
        "ub[0] should be NaN due to NaN weight, got {}",
        result.ub[[0, 0, 0]]
    );

    // Column 1 of the weight is [0.0, 1.0] — all finite, so output should be finite.
    assert!(
        !result.lb[[0, 0, 1]].is_nan(),
        "lb[1] should be finite (no NaN weights in column 1)"
    );
    assert!(
        !result.ub[[0, 0, 1]].is_nan(),
        "ub[1] should be finite (no NaN weights in column 1)"
    );
}

/// Regression #3116: NaN weights must propagate through matmul_batched.
#[test]
fn test_matmul_batched_nan_weight_propagates_3116() {
    let lw = Array4::from_shape_vec((1, 1, 1, 2), vec![1.0, 1.0])
        .expect("invariant: shape matches element count");
    let uw = lw.clone();
    let lb = Array3::from_shape_vec((1, 1, 2), vec![1.0, 1.0])
        .expect("invariant: shape matches element count");
    let ub = lb.clone();
    let bounds = MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub)
        .expect("invariant: valid MultiNormBounds construction");

    // Batched weight: (batch=1, 2, 2) with one NaN entry.
    let weight = Array3::from_shape_vec((1, 2, 2), vec![1.0, 0.0, f32::NAN, 1.0])
        .expect("invariant: shape matches element count");
    let result = bounds
        .matmul_batched(&weight)
        .expect("matmul_batched should not error on NaN weights");

    // NaN in column 0 must propagate.
    assert!(
        result.lb[[0, 0, 0]].is_nan(),
        "batched lb[0] should be NaN due to NaN weight, got {}",
        result.lb[[0, 0, 0]]
    );
    assert!(
        result.ub[[0, 0, 0]].is_nan(),
        "batched ub[0] should be NaN due to NaN weight, got {}",
        result.ub[[0, 0, 0]]
    );
}

/// concretize_sound() must widen bounds by at least 1 ULP vs concretize() (#2287).
#[test]
fn test_concretize_sound_widens_bounds_by_directed_rounding_2287() {
    // Set up non-trivial norm: lw != uw so bounds differ.
    let lw = Array4::from_shape_vec((1, 1, 2, 1), vec![3.0, 4.0]).expect("invariant: static shape");
    let uw = Array4::from_shape_vec((1, 1, 2, 1), vec![1.0, 2.0]).expect("invariant: static shape");
    let lb = Array3::from_shape_vec((1, 1, 1), vec![1.0]).expect("invariant: static shape");
    let ub = Array3::from_shape_vec((1, 1, 1), vec![2.0]).expect("invariant: static shape");
    let bounds =
        MultiNormBounds::new(2.0, 1.0, 1, lw, lb, uw, ub).expect("invariant: valid bounds");

    let plain = bounds.concretize().expect("invariant: concretize succeeds");
    let sound = bounds
        .concretize_sound()
        .expect("invariant: concretize_sound succeeds");

    // Sound lower must be <= plain lower (widened toward -inf).
    assert!(
        sound.lower()[[0, 0, 0]] <= plain.lower()[[0, 0, 0]],
        "sound lower {} should be <= plain lower {}",
        sound.lower()[[0, 0, 0]],
        plain.lower()[[0, 0, 0]]
    );
    // Sound upper must be >= plain upper (widened toward +inf).
    assert!(
        sound.upper()[[0, 0, 0]] >= plain.upper()[[0, 0, 0]],
        "sound upper {} should be >= plain upper {}",
        sound.upper()[[0, 0, 0]],
        plain.upper()[[0, 0, 0]]
    );
    // Bounds must be valid (not inverted).
    assert!(
        sound.lower()[[0, 0, 0]] <= sound.upper()[[0, 0, 0]],
        "sound bounds should not be inverted: [{}, {}]",
        sound.lower()[[0, 0, 0]],
        sound.upper()[[0, 0, 0]]
    );
}

/// concretize_sound() must produce finite, non-inverted bounds even with
/// extreme epsilon values that stress the dual-norm computation (#2353).
#[test]
fn test_concretize_sound_extreme_eps_produces_valid_bounds_2353() {
    // Large eps * large norm can push lower below upper. new_repaired(Widen)
    // catches any such case (#3423).
    let lw =
        Array4::from_shape_vec((1, 1, 2, 1), vec![1e15, 1e15]).expect("invariant: static shape");
    let uw =
        Array4::from_shape_vec((1, 1, 2, 1), vec![1e15, 1e15]).expect("invariant: static shape");
    let lb = Array3::from_shape_vec((1, 1, 1), vec![0.0]).expect("invariant: static shape");
    let ub = Array3::from_shape_vec((1, 1, 1), vec![1.0]).expect("invariant: static shape");
    // eps = 1e15, norm ≈ sqrt(2)*1e15 ≈ 1.41e15.
    // lower = 0 - 1e15 * 1.41e15 ≈ -1.41e30, upper = 1 + 1e15 * 1.41e15 ≈ 1.41e30.
    let bounds =
        MultiNormBounds::new(2.0, 1e15, 1, lw, lb, uw, ub).expect("invariant: valid bounds");
    let sound = bounds
        .concretize_sound()
        .expect("invariant: concretize_sound succeeds");

    // Must be valid (not inverted, not NaN).
    let lo = sound.lower()[[0, 0, 0]];
    let hi = sound.upper()[[0, 0, 0]];
    assert!(!lo.is_nan(), "lower bound should not be NaN");
    assert!(!hi.is_nan(), "upper bound should not be NaN");
    assert!(lo <= hi, "bounds should not be inverted: [{lo}, {hi}]");
}

// concretize_sound() with f64 accumulators must produce bounds that contain
// the f64-precision concretization result for high-dimensional inputs (#2364).
//
// The old f32-accumulator norm had O(dim_in * eps_f32) error which for
// dim_in=768 could be ~4.6e-5 relative — thousands of ULPs, far exceeding
// the 1-ULP directed rounding budget.
proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]
    #[test]
    fn proptest_concretize_sound_contains_f64_result_high_dim(
        dim_in in 128_usize..=1024,
        seed in 0u64..1000,
    ) {
        // Generate pseudorandom weights using a simple LCG seeded by `seed`.
        let mut rng_state = seed;
        let mut next_f32 = || -> f32 {
            // LCG: Numerical Recipes parameters.
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to [-1.0, 1.0].
            let bits = ((rng_state >> 33) as u32) & 0x7FFFFF;
            let frac = bits as f32 / 0x7FFFFF as f32;
            frac * 2.0 - 1.0
        };
        let weights: Vec<f32> = (0..dim_in).map(|_| next_f32()).collect();
        let lw = Array4::from_shape_vec((1, 1, dim_in, 1), weights.clone())
            .expect("invariant: static shape");
        let uw = lw.clone();
        let lb = Array3::from_shape_vec((1, 1, 1), vec![0.5])
            .expect("invariant: static shape");
        let ub = Array3::from_shape_vec((1, 1, 1), vec![1.5])
            .expect("invariant: static shape");
        let bounds = MultiNormBounds::new(2.0, 0.1, 1, lw, lb, uw, ub)
            .expect("invariant: valid bounds");
        let sound = bounds.concretize_sound()
            .expect("invariant: concretize_sound succeeds");

        // Compute the f64-precision "true" concretization.
        let norm_f64: f64 = weights.iter()
            .map(|&w| (w as f64).abs().powi(2))
            .sum::<f64>()
            .sqrt();
        let true_lower = 0.5_f64 - 0.1_f64 * norm_f64;
        let true_upper = 1.5_f64 + 0.1_f64 * norm_f64;

        let lo = sound.lower()[[0, 0, 0]];
        let hi = sound.upper()[[0, 0, 0]];

        prop_assert!(
            (lo as f64) <= true_lower,
            "sound lower {} exceeds f64 true lower {} (dim_in={}, seed={})",
            lo, true_lower, dim_in, seed
        );
        prop_assert!(
            (hi as f64) >= true_upper,
            "sound upper {} below f64 true upper {} (dim_in={}, seed={})",
            hi, true_upper, dim_in, seed
        );
        prop_assert!(!lo.is_nan(), "lower is NaN");
        prop_assert!(!hi.is_nan(), "upper is NaN");
        prop_assert!(lo <= hi, "bounds inverted: [{lo}, {hi}]");
    }
}
