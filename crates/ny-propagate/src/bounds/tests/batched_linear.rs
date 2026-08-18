// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for BatchedLinearBounds.

use super::checked_bounds;
use crate::bounds::BatchedLinearBounds;
use ndarray::{array, ArrayD, IxDyn};
use ny_core::NyError;
use std::{
    mem::size_of,
    time::{Duration, Instant},
};

#[ntest::timeout(5000)]
#[test]
fn test_batched_linear_bounds_identity_1d() {
    let bounds = BatchedLinearBounds::identity(&[4]).unwrap();

    assert_eq!(bounds.input_shape, vec![4]);
    assert_eq!(bounds.output_shape, vec![4]);
    assert_eq!(bounds.in_dim(), 4);
    assert_eq!(bounds.out_dim(), 4);

    // Check it's an identity
    let a_shape = bounds.lower_a.shape();
    assert_eq!(a_shape, &[4, 4]);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_linear_bounds_identity_2d() {
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();

    assert_eq!(bounds.input_shape, vec![2, 3]);
    assert_eq!(bounds.output_shape, vec![2, 3]);
    assert_eq!(bounds.in_dim(), 3);
    assert_eq!(bounds.out_dim(), 3);

    // Shape: [batch=2, out=3, in=3]
    let a_shape = bounds.lower_a.shape();
    assert_eq!(a_shape, &[2, 3, 3]);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_linear_bounds_identity_empty() {
    let bounds = BatchedLinearBounds::identity(&[]).unwrap();

    // Should create minimal bounds
    assert_eq!(bounds.input_shape, vec![1]);
    assert_eq!(bounds.output_shape, vec![1]);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_linear_bounds_concretize_identity() {
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();

    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("bounds shape mismatch"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("bounds shape mismatch"),
    );

    let output = bounds.concretize(&input).unwrap();

    // Identity concretize with directed rounding (#2391): lower bounds are
    // rounded down (next_down_f32) and upper bounds rounded up (next_up_f32), so
    // we check soundness containment (lb <= true_lower, ub >= true_upper) within
    // 1 ULP. The BLAS path now accumulates the dot product in f64 (operands cast
    // f32->f64, products exact) and rounds only the single final f64->f32 cast,
    // so for an identity row (a single `1.0 * x` term, computed exactly) the gap
    // is EXACTLY the one directed-rounding ULP — recovering the original 1-ULP
    // tightness that the earlier f32-BLAS + envelope path relaxed.
    let true_lower = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let true_upper = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let out_lower = output.lower().as_slice().unwrap();
    let out_upper = output.upper().as_slice().unwrap();
    for i in 0..6 {
        assert!(
            out_lower[i] <= true_lower[i],
            "lower[{i}] = {} should be <= {}",
            out_lower[i],
            true_lower[i]
        );
        assert!(
            out_upper[i] >= true_upper[i],
            "upper[{i}] = {} should be >= {}",
            out_upper[i],
            true_upper[i]
        );
        // Gap is at most 1 ULP from the single directed f64->f32 cast. The f64
        // dot of an identity row is exact (single non-zero entry = 1.0).
        let lo_ulps =
            (true_lower[i].to_bits() as i64 - out_lower[i].to_bits() as i64).unsigned_abs();
        let hi_ulps =
            (out_upper[i].to_bits() as i64 - true_upper[i].to_bits() as i64).unsigned_abs();
        assert!(lo_ulps <= 1, "lower[{i}] off by {lo_ulps} ULPs (max 1)");
        assert!(hi_ulps <= 1, "upper[{i}] off by {hi_ulps} ULPs (max 1)");
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_linear_bounds_identity_for_attention_valid() {
    // Valid attention shape: [batch=1, heads=2, seq=4, seq=4]
    let bounds = BatchedLinearBounds::identity_for_attention(&[1, 2, 4, 4]);

    assert!(bounds.is_some());
    let bounds = bounds.unwrap();

    // Flattened size = 4*4 = 16
    // A shape: [1, 2, 16, 16]
    assert_eq!(bounds.lower_a.shape(), &[1, 2, 16, 16]);
    assert_eq!(bounds.input_shape, vec![1, 2, 16]);
    assert_eq!(bounds.output_shape, vec![1, 2, 16]);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_linear_bounds_identity_for_attention_non_square() {
    // Non-square: seq_out != seq_in
    let bounds = BatchedLinearBounds::identity_for_attention(&[1, 2, 4, 5]);
    assert!(bounds.is_none());
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_linear_bounds_identity_for_attention_wrong_dims() {
    // Not 4D
    let bounds = BatchedLinearBounds::identity_for_attention(&[2, 4, 4]);
    assert!(bounds.is_none());
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_linear_bounds_identity_for_attention_too_large() {
    // seq=65 means flat_size=4225 > 4096 limit
    let bounds = BatchedLinearBounds::identity_for_attention(&[1, 1, 65, 65]);
    assert!(bounds.is_none());
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_linear_bounds_compose_identity() {
    // Composing two identities should give identity
    let a = BatchedLinearBounds::identity(&[3]).unwrap();
    let b = BatchedLinearBounds::identity(&[3]).unwrap();

    let composed = a.compose(&b).unwrap();

    // Check shapes
    assert_eq!(composed.input_shape, a.input_shape);
    assert_eq!(composed.output_shape, b.output_shape);

    // Composed identity should still be identity (within directed rounding tolerance).
    // compose() uses f64 accumulators with directed rounding (next_down_f32 for lower,
    // next_up_f32 for upper), so results may differ from exact values by at most 1 ULP.
    // For soundness: lower <= exact <= upper.
    let input = checked_bounds(
        array![1.0_f32, 2.0, 3.0].into_dyn(),
        array![4.0_f32, 5.0, 6.0].into_dyn(),
    );

    let output = composed.concretize(&input).unwrap();
    let lower = output.lower().as_slice().unwrap();
    let upper = output.upper().as_slice().unwrap();
    for (i, (&lo, &hi)) in lower.iter().zip(upper.iter()).enumerate() {
        let exact_lo = (i + 1) as f32;
        let exact_hi = (i + 4) as f32;
        assert!(lo <= exact_lo, "lower[{i}]={lo} > exact {exact_lo}");
        assert!(hi >= exact_hi, "upper[{i}]={hi} < exact {exact_hi}");
        assert!(
            (lo - exact_lo).abs() < 1e-5,
            "lower[{i}]={lo} too far from {exact_lo}"
        );
        assert!(
            (hi - exact_hi).abs() < 1e-5,
            "upper[{i}]={hi} too far from {exact_hi}"
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_linear_bounds_compose_dimension_mismatch() {
    let a = BatchedLinearBounds::identity(&[3]).unwrap();
    let b = BatchedLinearBounds::identity(&[4]).unwrap(); // Different dim

    let result = a.compose(&b);
    assert!(result.is_err());
}

/// Regression test: BatchedLinearBounds::concretize_sound must return bounds
/// at least as wide as concretize() (widened by 1 ULP toward -inf/+inf).
/// This is the batched counterpart of test_concretize_sound_widens_bounds in linear.rs.
#[ntest::timeout(5000)]
#[test]
fn test_batched_concretize_sound_widens_bounds() {
    // Build a non-trivial batched linear bound with mixed signs.
    // Shape: [batch=2, out=2, in=3] for A, [batch=2, out=2] for b.
    let lower_a = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 3]),
        vec![
            1.0, -0.5, 0.3, // batch 0, row 0
            -0.2, 0.8, -0.1, // batch 0, row 1
            0.7, 0.0, -0.4, // batch 1, row 0
            -0.6, 0.3, 0.9, // batch 1, row 1
        ],
    )
    .unwrap();
    let upper_a = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 3]),
        vec![
            1.2, -0.3, 0.5, // batch 0, row 0
            0.0, 1.0, 0.1, // batch 0, row 1
            0.9, 0.2, -0.2, // batch 1, row 0
            -0.4, 0.5, 1.1, // batch 1, row 1
        ],
    )
    .unwrap();
    let lower_b = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.1, -0.2, 0.05, -0.15]).unwrap();
    let upper_b = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.3, 0.1, 0.25, 0.05]).unwrap();

    let bounds = BatchedLinearBounds::from_parts_unchecked(
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        vec![2, 3],
        vec![2, 2],
    );

    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.5, -0.5, -0.8, 0.3, -0.3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 1.5, 0.5, 0.9, 1.2, 0.7]).unwrap(),
    );

    let normal = bounds.concretize(&input).unwrap();
    let sound = bounds.concretize_sound(&input).unwrap();

    let n = normal.lower().len();
    for i in 0..n {
        assert!(
            sound.lower().as_slice().unwrap()[i] <= normal.lower().as_slice().unwrap()[i],
            "sound lower[{i}]={} should be <= normal lower[{i}]={}",
            sound.lower().as_slice().unwrap()[i],
            normal.lower().as_slice().unwrap()[i],
        );
        assert!(
            sound.upper().as_slice().unwrap()[i] >= normal.upper().as_slice().unwrap()[i],
            "sound upper[{i}]={} should be >= normal upper[{i}]={}",
            sound.upper().as_slice().unwrap()[i],
            normal.upper().as_slice().unwrap()[i],
        );
    }
}

/// Concretize_sound on identity bounds should produce bounds strictly wider than
/// the input (since directed rounding widens by 1 ULP even on exact identity).
#[ntest::timeout(5000)]
#[test]
fn test_batched_concretize_sound_identity_widens() {
    let bounds = BatchedLinearBounds::identity(&[3]).unwrap();
    let input = checked_bounds(
        array![1.0_f32, 2.0, 3.0].into_dyn(),
        array![4.0_f32, 5.0, 6.0].into_dyn(),
    );

    let normal = bounds.concretize(&input).unwrap();
    let sound = bounds.concretize_sound(&input).unwrap();

    // Identity concretize gives exact input bounds; sound version must be strictly wider
    // (strict inequality catches regressions where rounding is accidentally a no-op).
    for i in 0..3 {
        assert!(
            sound.lower().as_slice().unwrap()[i] < normal.lower().as_slice().unwrap()[i],
            "sound lower[{i}]={} must be strictly < normal lower[{i}]={}",
            sound.lower().as_slice().unwrap()[i],
            normal.lower().as_slice().unwrap()[i],
        );
        assert!(
            sound.upper().as_slice().unwrap()[i] > normal.upper().as_slice().unwrap()[i],
            "sound upper[{i}]={} must be strictly > normal upper[{i}]={}",
            sound.upper().as_slice().unwrap()[i],
            normal.upper().as_slice().unwrap()[i],
        );
    }
}

/// Regression #2214/#2220: batched concretize_sound with n=4096 uses BLAS DGEMV
/// (f64 accumulation of f32-cast operands). Verify lower <= upper, and that the
/// result is TIGHT — within a handful of result-ULPs of the EXACT-product f64
/// reference — even though alternating-sign coefficients maximize cancellation.
///
/// This is the tightness-recovery follow-up to commit 0b03a2e: f64 accumulation
/// makes the BLAS dot exact up to a single directed f32 cast, so the previous
/// conservative `gamma_{2n+2}*S` envelope (vacuously large under cancellation)
/// is gone. The test also asserts the gap is orders of magnitude below that old
/// envelope, proving the over-widening was eliminated.
#[ntest::timeout(60000)]
#[test]
fn test_batched_concretize_sound_n4096_blas_precision() {
    let n = 4096;
    let scale = 1.0 / (n as f32).sqrt();
    let a_data: Vec<f32> = (0..n)
        .map(|j| {
            let sign = if j % 2 == 0 { 1.0f32 } else { -1.0 };
            sign * (1.0 + j as f32 * 1e-5) * scale
        })
        .collect();
    let x_data: Vec<f32> = (0..n).map(|j| 0.5 + j as f32 * 1e-4).collect();

    // EXACT-product f64 reference: each product `a_j * x_j` of two f32 values is
    // exact in f64, so this is the true mathematical value of the (degenerate)
    // linear bound — the basis of the tightness check. (The f64-accumulate BLAS
    // path computes exactly this, up to its sub-f64-ULP summation residual and
    // the single directed f32 cast.)
    let ref_sum: f64 = a_data
        .iter()
        .zip(&x_data)
        .map(|(&a, &x)| a as f64 * x as f64)
        .sum();
    let ref_f32 = ref_sum as f32;

    let lower_a = ArrayD::from_shape_vec(IxDyn(&[1, n]), a_data).unwrap();
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        lower_a.clone(),
        ArrayD::zeros(IxDyn(&[1])),
        lower_a,
        ArrayD::zeros(IxDyn(&[1])),
        vec![n],
        vec![1],
    );
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[n]), x_data.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), x_data).unwrap(),
    );

    let sound = bounds.concretize_sound(&input).unwrap();
    let sl = sound.lower().as_slice().unwrap()[0];
    let su = sound.upper().as_slice().unwrap()[0];

    // Basic interval property: lower <= upper.
    assert!(sl <= su, "lower {sl} > upper {su}");

    // TIGHTNESS RECOVERY: the f64-accumulate BLAS path computes the EXACT-product
    // f64 dot and rounds only the single f64->f32 cast (then `concretize_sound`
    // adds one more directed ULP). With alternating-sign coefficients the result
    // is tiny but `S = sum|a_j*x_j|` is huge; the OLD f32-BLAS path widened by
    // `gamma_{2n+2}*S` (~ n*2^-24 * S), which is vacuously large here. The f64
    // path instead stays within a HANDFUL of result-ULPs of the exact reference
    // — i.e. the tightness the envelope path destroyed is recovered. Tolerance =
    // a few ULPs of the result plus the sub-f64-ULP dot residual (`~ n*S*2^-53`).
    let s: f64 = a_data_abs_dot_x_for_residual(n, scale);
    let dot_residual = (n as f64) * s * f64::EPSILON;
    let result_ulp = (ref_f32.abs()).max(f32::MIN_POSITIVE) * f32::EPSILON;
    let blas_tol = 8.0 * result_ulp + 4.0 * dot_residual as f32;
    assert!(
        (sl - ref_f32).abs() <= blas_tol,
        "lower {sl} too far from EXACT f64 ref {ref_f32} (gap: {}, tol: {blas_tol})",
        (sl - ref_f32).abs()
    );
    assert!(
        (su - ref_f32).abs() <= blas_tol,
        "upper {su} too far from EXACT f64 ref {ref_f32} (gap: {}, tol: {blas_tol})",
        (su - ref_f32).abs()
    );
    // Confirm the f64 path is DRAMATICALLY tighter than the rejected envelope:
    // the old `gamma_{2n+2}*S` widening would have been orders of magnitude
    // larger than the actual gap here.
    let u = 0.5 * f64::from(f32::EPSILON);
    let old_envelope = ((2.0 * n as f64 + 2.0) * u) / (1.0 - (2.0 * n as f64 + 2.0) * u) * s;
    assert!(
        (sl - ref_f32).abs() as f64 <= 0.01 * old_envelope,
        "f64 path not meaningfully tighter than the old envelope: gap={}, old_env={old_envelope}",
        (sl - ref_f32).abs()
    );
}

/// Recompute `S = sum_j |a_j * x_j|` (f64) for the n4096 degenerate-box test,
/// used to size the sub-f64-ULP dot residual and to compare against the rejected
/// `gamma*S` envelope. Mirrors the `a_data`/`x_data` construction above.
fn a_data_abs_dot_x_for_residual(n: usize, scale: f32) -> f64 {
    (0..n)
        .map(|j| {
            let a = (1.0 + j as f32 * 1e-5) * scale; // |a_j| (sign dropped)
            let x = 0.5 + j as f32 * 1e-4;
            (a as f64) * (x as f64)
        })
        .sum()
}

/// Soundness invariant: `concretize_sound` bounds must contain the true linear
/// function value `f(x) = A @ x + b` for any x inside the input interval.
///
/// This tests the core soundness property that makes CROWN verification valid:
/// for all x in [x_l, x_u], lower_bound <= f(x) <= upper_bound.
///
/// We test with a known linear function and verify the property at multiple
/// sample points (corners and midpoint of the input interval).
#[ntest::timeout(10000)]
#[test]
fn test_batched_concretize_sound_contains_true_function_values() {
    // Build a small linear function: f(x) = A @ x + b
    // 2 outputs, 3 inputs.
    // lower_a and upper_a are the same here (exact linear function, not relaxation),
    // so concretized lower should be <= f(x) and upper should be >= f(x) for all x in range.
    let a_data = vec![
        1.5_f32, -0.7, 0.3, // output 0 coefficients
        -0.4, 1.2, -0.8, // output 1 coefficients
    ];
    let b_data = vec![0.1_f32, -0.2]; // biases

    let lower_a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), a_data.clone()).unwrap();
    let upper_a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), a_data).unwrap();
    let lower_b = ArrayD::from_shape_vec(IxDyn(&[2]), b_data.clone()).unwrap();
    let upper_b = ArrayD::from_shape_vec(IxDyn(&[2]), b_data).unwrap();

    let bounds = BatchedLinearBounds::new(lower_a, lower_b, upper_a, upper_b, vec![3], vec![2])
        .expect("valid bounds construction");

    // Input interval: x in [-1.0, 2.0] x [0.5, 1.5] x [-0.5, 0.5]
    let x_l = vec![-1.0_f32, 0.5, -0.5];
    let x_u = vec![2.0_f32, 1.5, 0.5];

    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[3]), x_l.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), x_u.clone()).unwrap(),
    );

    let sound = bounds
        .concretize_sound(&input)
        .expect("concretize_sound should succeed");

    let sl = sound.lower().as_slice().unwrap();
    let su = sound.upper().as_slice().unwrap();

    // Sample points inside the input interval.
    let a_rows = [vec![1.5_f32, -0.7, 0.3], vec![-0.4_f32, 1.2, -0.8]];
    let b_vals = [0.1_f32, -0.2];

    let sample_points: Vec<Vec<f32>> = vec![
        x_l,                   // lower corner
        x_u,                   // upper corner
        vec![0.5, 1.0, 0.0],   // midpoint
        vec![-1.0, 1.5, 0.5],  // mixed corner 1
        vec![2.0, 0.5, -0.5],  // mixed corner 2
        vec![0.0, 0.75, 0.25], // interior point
    ];

    for x in &sample_points {
        for out_idx in 0..2 {
            let f_x: f32 = a_rows[out_idx]
                .iter()
                .zip(x.iter())
                .map(|(&a, &xi)| a * xi)
                .sum::<f32>()
                + b_vals[out_idx];

            assert!(
                sl[out_idx] <= f_x,
                "soundness violation: lower[{out_idx}]={} > f(x)={f_x} at x={x:?}",
                sl[out_idx]
            );
            assert!(
                su[out_idx] >= f_x,
                "soundness violation: upper[{out_idx}]={} < f(x)={f_x} at x={x:?}",
                su[out_idx]
            );
        }
    }
}

/// Soundness: `concretize_sound` with separate lower_a and upper_a (relaxation)
/// must still contain the range of the true function for all x in the interval.
///
/// This models the real CROWN use case where lower_a != upper_a because the
/// linear relaxation of a nonlinear activation produces different slopes for
/// the lower and upper bound functions.
#[ntest::timeout(10000)]
#[test]
fn test_batched_concretize_sound_relaxation_contains_both_bound_functions() {
    // lower bound function: f_L(x) = lower_a @ x + lower_b
    // upper bound function: f_U(x) = upper_a @ x + upper_b
    // concretize_sound must produce:
    //   result.lower <= min_{x in [x_l,x_u]} f_L(x)
    //   result.upper >= max_{x in [x_l,x_u]} f_U(x)
    let lower_a = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.5_f32, -0.3]).unwrap();
    let upper_a = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.8_f32, 0.2]).unwrap();
    let lower_b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.1_f32]).unwrap();
    let upper_b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.2_f32]).unwrap();

    let bounds = BatchedLinearBounds::new(lower_a, lower_b, upper_a, upper_b, vec![2], vec![1])
        .expect("valid bounds construction");

    // Input: x in [-1, 1] x [-2, 2]
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 2.0]).unwrap(),
    );

    let sound = bounds
        .concretize_sound(&input)
        .expect("concretize_sound should succeed");

    let sl = sound.lower().as_slice().unwrap()[0];
    let su = sound.upper().as_slice().unwrap()[0];

    // Compute the actual minimum of f_L and maximum of f_U over the input box
    // by evaluating at all 4 corners.
    let corners: Vec<(f32, f32)> = vec![(-1.0, -2.0), (-1.0, 2.0), (1.0, -2.0), (1.0, 2.0)];

    let mut min_fl = f32::INFINITY;
    let mut max_fu = f32::NEG_INFINITY;
    for &(x0, x1) in &corners {
        let fl = 0.5 * x0 + (-0.3) * x1 + (-0.1);
        let fu = 0.8 * x0 + 0.2 * x1 + 0.2;
        min_fl = min_fl.min(fl);
        max_fu = max_fu.max(fu);
    }

    assert!(
        sl <= min_fl,
        "sound lower {sl} must be <= min(f_L) = {min_fl}"
    );
    assert!(
        su >= max_fu,
        "sound upper {su} must be >= max(f_U) = {max_fu}"
    );
    // Also verify the basic well-formedness.
    assert!(sl <= su, "lower {sl} must be <= upper {su}");
}

/// Soundness on the SCALAR fallback under cancellation: one ±Inf coefficient
/// anywhere in the batch fails `all_finite_for_blas` and routes the WHOLE batch
/// through `concretize_scalar_posneg` — including well-behaved co-tenant rows.
/// That path must form each coefficient×input product in f64: with large
/// mixed-sign coefficients and a near-degenerate box (|term| ≫ |result|), f32
/// round-to-nearest products bias the accumulated sum INWARD by far more than
/// the final result-magnitude ULP widening can cover, yielding a "sound" lower
/// bound above the true function value.
#[ntest::timeout(10000)]
#[test]
fn test_batched_concretize_sound_scalar_fallback_cancellation_sound() {
    let c: f32 = 1.0e7;
    let x0: f32 = 1.000_065_6;
    let x1: f32 = 1.000_003_1;

    // Row 0: heavy cancellation. Row 1: Inf coefficient (legal conservative
    // guard) that forces the batch off the BLAS path.
    let lower_a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![c, -c, f32::INFINITY, 0.0]).unwrap();
    let upper_a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![c, -c, 0.0, 0.0]).unwrap();
    let lower_b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let upper_b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let bounds = BatchedLinearBounds::new(lower_a, lower_b, upper_a, upper_b, vec![2], vec![2])
        .expect("valid bounds construction");

    // Degenerate (single-point) box: the true row-0 value is exactly
    // f(x) = c*x0 - c*x1, computed in f64 (f32 operands promote exactly).
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![x0, x1]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![x0, x1]).unwrap(),
    );
    let truth = (c as f64) * (x0 as f64) - (c as f64) * (x1 as f64);

    let sound = bounds
        .concretize_sound(&input)
        .expect("concretize_sound should succeed");
    let sl = sound.lower().as_slice().unwrap();
    let su = sound.upper().as_slice().unwrap();

    // Row 0 must remain finite (no over-degradation of well-behaved rows) and
    // must contain the true value despite the cancellation.
    assert!(sl[0].is_finite(), "row 0 lower must stay finite: {}", sl[0]);
    assert!(
        (sl[0] as f64) <= truth,
        "soundness violation on scalar fallback: lower {} > true value {truth}",
        sl[0]
    );
    assert!(
        (su[0] as f64) >= truth,
        "soundness violation on scalar fallback: upper {} < true value {truth}",
        su[0]
    );

    // Row 1 carries an Inf guard coefficient on the lower side: that direction
    // degrades to the sound saturating bound rather than a confident value.
    assert_eq!(
        sl[1],
        f32::NEG_INFINITY,
        "Inf-guard row lower must degrade to -inf"
    );
    assert!(su[1].is_finite(), "row 1 upper is well-behaved (zero row)");
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_exact_transport_sentinel_degrades_and_stays_sticky() {
    use ny_core::CROWN_COEFF_MAX;

    let sentinel = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![CROWN_COEFF_MAX]).unwrap();
    let zero_bias = ArrayD::zeros(IxDyn(&[1]));
    let guarded = BatchedLinearBounds::new(
        sentinel.clone(),
        zero_bias.clone(),
        sentinel,
        zero_bias.clone(),
        vec![1],
        vec![1],
    )
    .expect("shape-valid sentinel test bound");
    let input = checked_bounds(array![0.0_f32].into_dyn(), array![0.0_f32].into_dyn());
    let concrete = guarded
        .concretize_sound(&input)
        .expect("sentinel concretization degrades instead of failing");
    assert_eq!(concrete.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(concrete.upper()[[0]], f32::INFINITY);

    // Exact zero composition must not cancel the finite transport sentinel.
    let zero_a = ArrayD::zeros(IxDyn(&[1, 1]));
    let zero = BatchedLinearBounds::new(
        zero_a.clone(),
        zero_bias.clone(),
        zero_a,
        zero_bias,
        vec![1],
        vec![1],
    )
    .expect("zero outer bound");
    let composed = guarded
        .compose(&zero)
        .expect("unsafe coefficient composition must conservatively degrade");
    assert_eq!(composed.lower_a[[0, 0]], f32::NEG_INFINITY);
    assert_eq!(composed.upper_a[[0, 0]], f32::INFINITY);
}

/// A ±Inf conservative guard coefficient (legal per `BatchedLinearBounds::new`,
/// produced by `compose`) must degrade its bound direction to the sound
/// saturating value on the scalar fallback. The pos/neg split would otherwise
/// multiply the guard by an input endpoint whose sign or exact zero flips or
/// silently drops the poison: a -inf lower guard times a negative upper
/// endpoint yields +inf (then repaired by SWAP into a confident finite lower),
/// and times an exact-zero endpoint contributes 0 (guard vanishes entirely).
#[ntest::timeout(10000)]
#[test]
fn test_batched_concretize_scalar_inf_guard_coefficient_degrades_row() {
    // Lower guard, negative box on the guarded coordinate.
    let lower_a = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::NEG_INFINITY, 1.0]).unwrap();
    let upper_a = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.0]).unwrap();
    let lower_b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let upper_b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let bounds = BatchedLinearBounds::new(lower_a, lower_b, upper_a, upper_b, vec![2], vec![1])
        .expect("valid bounds construction");

    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-2.0_f32, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, 1.0]).unwrap(),
    );
    let sound = bounds
        .concretize_sound(&input)
        .expect("concretize_sound should succeed");
    let sl = sound.lower().as_slice().unwrap()[0];
    let su = sound.upper().as_slice().unwrap()[0];

    assert_eq!(
        sl,
        f32::NEG_INFINITY,
        "lower guard coefficient must degrade the lower bound to -inf, got {sl}"
    );
    // The upper direction is independent and well-behaved:
    // max of f_U = x0 + x1 over the box is -1 + 1 = 0.
    assert!(su.is_finite() && su >= 0.0, "upper must stay sound: {su}");

    // Upper guard, same negative box: +inf * negative endpoint would drive the
    // upper bound to -inf (below every true value) without the degradation.
    let lower_a2 = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 1.0]).unwrap();
    let upper_a2 = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::INFINITY, 1.0]).unwrap();
    let lower_b2 = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let upper_b2 = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let bounds2 =
        BatchedLinearBounds::new(lower_a2, lower_b2, upper_a2, upper_b2, vec![2], vec![1])
            .expect("valid bounds construction");
    let input2 = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-2.0_f32, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, 1.0]).unwrap(),
    );
    let sound2 = bounds2
        .concretize_sound(&input2)
        .expect("concretize_sound should succeed");
    let sl2 = sound2.lower().as_slice().unwrap()[0];
    let su2 = sound2.upper().as_slice().unwrap()[0];

    assert_eq!(
        su2,
        f32::INFINITY,
        "upper guard coefficient must degrade the upper bound to +inf, got {su2}"
    );
    // min of f_L = x0 + x1 over the box is -2 + 0 = -2.
    assert!(
        sl2.is_finite() && sl2 <= -2.0,
        "lower must stay sound: {sl2}"
    );
}

fn live_batched_deadline() -> Instant {
    Instant::now() + Duration::from_secs(5)
}

fn assert_same_endpoint_bits(expected: &ArrayD<f32>, actual: &ArrayD<f32>) {
    assert_eq!(expected.shape(), actual.shape());
    for (&expected, &actual) in expected.iter().zip(actual) {
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "endpoint mismatch: expected {expected:e}, actual {actual:e}"
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_memory_bytes_and_finite_deadline_live_parity() {
    let _env_lock = ny_test_utils::env::lock_env();
    let bounds = BatchedLinearBounds::identity(&[2, 3]).expect("batched identity");
    assert_eq!(
        bounds.memory_bytes(),
        48 * size_of::<f32>() + 4 * size_of::<usize>()
    );
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-2.0, -1.0, 0.5, 2.0, 3.0, 4.0]).unwrap(),
    );
    let legacy = bounds.concretize_sound(&input).expect("legacy concretize");
    let no_deadline = bounds
        .concretize_sound_with_deadline(&input, None)
        .expect("no-deadline wrapper");
    assert_same_endpoint_bits(legacy.lower(), no_deadline.lower());
    assert_same_endpoint_bits(legacy.upper(), no_deadline.upper());
    let finite = bounds
        .concretize_sound_with_deadline(&input, Some(live_batched_deadline()))
        .expect("finite-deadline concretize");
    assert_same_endpoint_bits(legacy.lower(), finite.lower());
    assert_same_endpoint_bits(legacy.upper(), finite.upper());

    let mut with_error = bounds;
    with_error.set_coeff_err(
        ArrayD::zeros(IxDyn(&[2, 3, 3])),
        ArrayD::zeros(IxDyn(&[2, 3, 3])),
    );
    assert_eq!(
        with_error.memory_bytes(),
        84 * size_of::<f32>() + 4 * size_of::<usize>()
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_finite_deadline_broadcast_and_flat_attention_parity() {
    let _env_lock = ny_test_utils::env::lock_env();
    let broadcast_a =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.5]).unwrap();
    let broadcast_b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.25, 0.75]).unwrap();
    let broadcast = BatchedLinearBounds::new(
        broadcast_a.clone(),
        broadcast_b.clone(),
        broadcast_a,
        broadcast_b,
        vec![2, 3],
        vec![2, 2],
    )
    .unwrap();
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-2.0, 1.0, 2.0, 3.0, -4.0, 5.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 2.0, 3.0, 4.0, -3.0, 6.0]).unwrap(),
    );
    let broadcast_legacy = broadcast.concretize_sound(&input).unwrap();
    let broadcast_finite = broadcast
        .concretize_sound_with_deadline(&input, Some(live_batched_deadline()))
        .unwrap();
    assert_eq!(broadcast_finite.shape(), &[2, 2]);
    for (&legacy, &finite) in broadcast_legacy
        .lower()
        .iter()
        .zip(broadcast_finite.lower())
    {
        assert!(
            finite <= legacy,
            "finite lower {finite:e} must enclose legacy {legacy:e}"
        );
    }
    for (&legacy, &finite) in broadcast_legacy
        .upper()
        .iter()
        .zip(broadcast_finite.upper())
    {
        assert!(
            finite >= legacy,
            "finite upper {finite:e} must enclose legacy {legacy:e}"
        );
    }

    let flat_a = ArrayD::from_shape_vec(
        IxDyn(&[2, 6]),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    )
    .unwrap();
    let flat_b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap();
    let flat = BatchedLinearBounds::new(
        flat_a.clone(),
        flat_b.clone(),
        flat_a,
        flat_b,
        vec![2, 3],
        vec![2],
    )
    .unwrap();
    let flat_legacy = flat.concretize_sound(&input).unwrap();
    let flat_finite = flat
        .concretize_sound_with_deadline(&input, Some(live_batched_deadline()))
        .unwrap();
    assert_eq!(flat_finite.shape(), &[2]);
    for (&legacy, &finite) in flat_legacy.lower().iter().zip(flat_finite.lower()) {
        assert!(
            finite <= legacy,
            "finite lower {finite:e} must enclose legacy {legacy:e}"
        );
    }
    for (&legacy, &finite) in flat_legacy.upper().iter().zip(flat_finite.upper()) {
        assert!(
            finite >= legacy,
            "finite upper {finite:e} must enclose legacy {legacy:e}"
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_finite_deadline_carries_coefficient_error_and_normalizes_endpoints() {
    let _env_lock = ny_test_utils::env::lock_env();
    let a = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let mut bounds =
        BatchedLinearBounds::new(a.clone(), b.clone(), a, b, vec![1], vec![1]).unwrap();
    let input = checked_bounds(array![-2.0_f32].into_dyn(), array![3.0_f32].into_dyn());
    let exact = bounds
        .concretize_sound_with_deadline(&input, Some(live_batched_deadline()))
        .unwrap();
    bounds.set_coeff_err(
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.25]).unwrap(),
    );
    let carried = bounds
        .concretize_sound_with_deadline(&input, Some(live_batched_deadline()))
        .unwrap();
    assert!(carried.lower()[[0]] < exact.lower()[[0]]);
    assert!(carried.upper()[[0]] > exact.upper()[[0]]);
    assert!(carried.lower()[[0]] <= -2.5);
    assert!(carried.upper()[[0]] >= 3.75);

    let tiny = f32::from_bits(1);
    let tiny_a = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![-tiny]).unwrap();
    let zero_b = ArrayD::zeros(IxDyn(&[1]));
    let tiny_bounds = BatchedLinearBounds::new(
        tiny_a.clone(),
        zero_b.clone(),
        tiny_a,
        zero_b,
        vec![1],
        vec![1],
    )
    .unwrap();
    let point = checked_bounds(array![1.0_f32].into_dyn(), array![1.0_f32].into_dyn());
    let published = tiny_bounds
        .concretize_sound_with_deadline(&point, Some(live_batched_deadline()))
        .unwrap();
    for &value in published.lower().iter().chain(published.upper()) {
        let magnitude = value.to_bits() & 0x7fff_ffff;
        assert!(
            magnitude == 0 || magnitude >= f32::MIN_POSITIVE.to_bits(),
            "finite-authority endpoint must be zero, normal, or infinite: {value:e}"
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_finite_deadline_is_atomic_when_expired_or_refused_mid_reduction() {
    let _env_lock = ny_test_utils::env::lock_env();
    let bounds = BatchedLinearBounds::identity(&[1, 8]).unwrap();
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[1, 8]), vec![-1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 8]), vec![1.0; 8]).unwrap(),
    );
    let source_lower = bounds.lower_a.clone();
    let source_upper = bounds.upper_a.clone();
    let input_lower = input.lower().clone();
    let input_upper = input.upper().clone();

    let expired = bounds
        .concretize_sound_with_deadline(&input, Some(Instant::now()))
        .expect_err("already-expired deadline must be terminal");
    assert!(matches!(expired, NyError::DeadlineExceeded(_)));

    let midwork = bounds
        .concretize_sound_with_forced_deadline_for_test(
            &input,
            "during batched finite lower reduction",
        )
        .expect_err("forced reduction deadline must be terminal");
    match midwork {
        NyError::DeadlineExceeded(message) => {
            assert!(message.contains("during batched finite lower reduction"));
        }
        error => panic!("expected DeadlineExceeded, got {error:?}"),
    }
    assert_eq!(bounds.lower_a, source_lower);
    assert_eq!(bounds.upper_a, source_upper);
    assert_eq!(input.lower(), &input_lower);
    assert_eq!(input.upper(), &input_upper);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_finite_deadline_total_live_budget_exact_and_minus_one() {
    let bounds = BatchedLinearBounds::identity(&[1, 4]).unwrap();
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-1.0; 4]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap(),
    );
    let required = bounds
        .finite_concretize_required_bytes_for_test(&input)
        .expect("valid finite plan");
    bounds
        .concretize_sound_with_budget_for_test(&input, live_batched_deadline(), required)
        .expect("exact total-live budget must be admitted");
    let error = bounds
        .concretize_sound_with_budget_for_test(&input, live_batched_deadline(), required - 1)
        .expect_err("budget-minus-one must refuse before allocation");
    assert!(matches!(
        error,
        NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site: "batched finite concretization",
        } if required_bytes == required && budget_bytes == required - 1
    ));
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_finite_deadline_polls_final_validation_atomically() {
    let _env_lock = ny_test_utils::env::lock_env();
    let bounds = BatchedLinearBounds::identity(&[1, 4]).unwrap();
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-1.0; 4]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap(),
    );
    let source = bounds.lower_a.clone();
    let error = bounds
        .concretize_sound_with_forced_deadline_for_test(
            &input,
            "during batched bounded-tensor validation",
        )
        .expect_err("final validation refusal must remain terminal");
    assert!(matches!(error, NyError::DeadlineExceeded(_)));
    assert_eq!(bounds.lower_a, source);
}

#[ntest::timeout(5000)]
#[test]
fn test_batched_finite_deadline_accepts_vector_like_input_reshape() {
    let _env_lock = ny_test_utils::env::lock_env();
    let bounds = BatchedLinearBounds::identity(&[1, 4]).unwrap();
    let input = checked_bounds(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-3.0, -2.0, -1.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-2.0, -1.0, 0.0, 1.0]).unwrap(),
    );
    let finite = bounds
        .concretize_sound_with_deadline(&input, Some(live_batched_deadline()))
        .unwrap();
    assert_eq!(finite.shape(), &[1, 4]);
    for ((&actual_lower, &actual_upper), (&exact_lower, &exact_upper)) in finite
        .lower()
        .iter()
        .zip(finite.upper())
        .zip(input.lower().iter().zip(input.upper()))
    {
        assert!(
            actual_lower <= exact_lower && actual_upper >= exact_upper,
            "finite identity publication [{actual_lower:e}, {actual_upper:e}] must enclose the exact reshaped input [{exact_lower:e}, {exact_upper:e}]"
        );
        for endpoint in [actual_lower, actual_upper] {
            let magnitude = endpoint.to_bits() & 0x7fff_ffff;
            assert!(
                magnitude == 0 || magnitude >= f32::MIN_POSITIVE.to_bits(),
                "finite-authority endpoint must be zero, normal, or infinite: {endpoint:e}"
            );
        }
    }
}
