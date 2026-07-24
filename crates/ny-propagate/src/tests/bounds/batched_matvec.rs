// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_batched_matvec_degenerate_a() {
    // Kills mutant: replace || with && in line 733
    // When a has < 2 dims, should return 0-dim array
    use ndarray::ArrayD;
    let a = ArrayD::zeros(ndarray::IxDyn(&[3])); // 1D array
    let x = ArrayD::from_elem(ndarray::IxDyn(&[3]), 1.0f32);
    let result = batched_matvec(&a, &x, false).unwrap();
    assert!(result.shape().is_empty()); // 0-dimensional array
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_matvec_degenerate_x() {
    // Kills mutant: empty x check
    use ndarray::ArrayD;
    let a = ArrayD::zeros(ndarray::IxDyn(&[2, 3]));
    let x: ArrayD<f32> = ArrayD::zeros(ndarray::IxDyn(&[])); // 0-dim
    let result = batched_matvec(&a, &x, false).unwrap();
    assert!(result.shape().is_empty()); // 0-dimensional array
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_matvec_basic() {
    // Basic matrix-vector multiplication
    use ndarray::{array, ArrayD};
    let a: ArrayD<f32> = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let x: ArrayD<f32> = array![1.0, 1.0].into_dyn();
    let result = batched_matvec(&a, &x, false).unwrap();
    assert!((result[[0]] - 3.0).abs() < 1e-6);
    assert!((result[[1]] - 7.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_matvec_with_inf() {
    // Tests the inf path
    // Kills mutant: replace || with && in line 763-764
    use ndarray::{array, ArrayD};
    let a: ArrayD<f32> = array![[1.0, 0.0], [0.0, 1.0]].into_dyn();
    let x: ArrayD<f32> = array![f32::INFINITY, 2.0].into_dyn();
    let result = batched_matvec(&a, &x, false).unwrap();
    assert!(result[[0]].is_infinite() && result[[0]] > 0.0);
    assert!((result[[1]] - 2.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_matvec_nan_in_a() {
    // Test NaN handling in matrix - NaN propagates per safe_mul_for_bounds behavior
    use ndarray::{array, ArrayD};
    let a: ArrayD<f32> = array![[1.0, f32::NAN], [0.0, 1.0]].into_dyn();
    let x: ArrayD<f32> = array![1.0, 1.0].into_dyn();
    let result = batched_matvec(&a, &x, false).unwrap();
    // Row 0: 1.0*1.0 + NaN*1.0 = 1.0 + NaN = NaN (NaN propagates)
    // Row 1: 0.0*1.0 + 1.0*1.0 = 0.0 + 1.0 = 1.0 (no NaN)
    assert!(
        result[[0]].is_nan(),
        "Expected NaN when matrix contains NaN"
    );
    assert!(
        (result[[1]] - 1.0).abs() < 1e-6,
        "Row without NaN should compute normally"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_matvec_zero_times_inf() {
    // 0 * inf should be handled correctly (gives 0 in safe path)
    use ndarray::{array, ArrayD};
    let a: ArrayD<f32> = array![[0.0, 1.0], [1.0, 0.0]].into_dyn();
    let x: ArrayD<f32> = array![f32::INFINITY, 2.0].into_dyn();
    let result = batched_matvec(&a, &x, false).unwrap();
    // First row: 0*inf + 1*2 = 0 + 2 = 2 (safe mul handles 0*inf = 0)
    assert!((result[[0]] - 2.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_matvec_inf_minus_inf_is_conservative_inf() {
    // Kills mutant: replace || with && in batched_matvec has_inf_or_nan check
    use ndarray::{Array1, Array2};
    let a = Array2::from_shape_vec((1, 2), vec![f32::INFINITY, f32::NEG_INFINITY])
        .unwrap()
        .into_dyn();
    let x = Array1::from_vec(vec![1.0, 1.0]).into_dyn();
    let r = batched_matvec(&a, &x, false).unwrap();
    assert_eq!(r.shape(), &[1]);
    assert!(r[[0]].is_infinite() && r[[0]] > 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_matvec_inf_cancellation_lower_bound_gives_neg_inf() {
    // Verifies is_lower polarity: inf + (-inf) cancellation maps to -inf for lower bounds.
    // Counterpart to test_batched_matvec_inf_minus_inf_is_conservative_inf (upper bound).
    // Reference: safe_add_for_bounds_with_polarity uses the same convention.
    use ndarray::{Array1, Array2};
    let a = Array2::from_shape_vec((1, 2), vec![f32::INFINITY, f32::NEG_INFINITY])
        .unwrap()
        .into_dyn();
    let x = Array1::from_vec(vec![1.0, 1.0]).into_dyn();
    let r = batched_matvec(&a, &x, true).unwrap();
    assert_eq!(r.shape(), &[1]);
    assert!(
        r[[0]].is_infinite() && r[[0]] < 0.0,
        "Lower-bound inf cancellation should give -inf, got {}",
        r[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_matvec_f64_accumulation_tighter_than_f32() {
    // Verify that f64 accumulation produces results that DIFFER from naive f32
    // accumulation for high-dimensional dot products (dim=768, transformer scale).
    //
    // Error analysis: f32 accumulation of n terms has worst-case relative error
    // O(n * eps_f32) ≈ 768 * 2^-24 ≈ 4.6e-5. With f64 accumulation, the only
    // rounding occurs in the final f64-to-f32 cast, giving at most 0.5 ULP error.
    //
    // Input design: one large entry (1e6) followed by 767 small entries (1e-3).
    // The large entry makes the running f32 sum ~1e6; subsequent 1e-3 additions
    // are below f32 ULP at that magnitude (~0.0625) and get absorbed to zero.
    // In f64, the small entries accumulate correctly: 767 * 1e-3 = 0.767.
    // f32 result ≈ 1000000.0; f64 result ≈ 1000000.75 (as f32). Diff = 0.75.
    use ndarray::ArrayD;
    let n = 768;

    let a_data: Vec<f32> = vec![1.0f32; n];
    let mut x_data: Vec<f32> = vec![1e-3f32; n];
    x_data[0] = 1e6f32;

    let a = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, n]), a_data).unwrap();
    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), x_data).unwrap();

    // Compute f64 reference
    let a_slice = a.as_slice().unwrap();
    let x_slice = x.as_slice().unwrap();
    let mut f64_ref = 0.0f64;
    for j in 0..n {
        f64_ref += a_slice[j] as f64 * x_slice[j] as f64;
    }
    let f64_ref_f32 = f64_ref as f32;

    // Compute naive f32 reference (what we'd get without f64 accumulators)
    let mut f32_naive = 0.0f32;
    for j in 0..n {
        f32_naive += a_slice[j] * x_slice[j];
    }

    // Precondition: f32 and f64 must actually differ for this test to be meaningful.
    // f32 absorbs the small terms: 1e6 + 1e-3 = 1e6 in f32 (ULP at 1e6 is ~0.0625).
    // f64 accumulates them: 1e6 + 767*1e-3 ≈ 1000000.767 → 1000000.75 as f32.
    assert_ne!(
        f32_naive, f64_ref_f32,
        "Test precondition failed: f32 and f64 accumulation must produce different results. \
         Both gave {}. The test input does not exercise precision differences.",
        f32_naive,
    );

    let result = batched_matvec(&a, &x, false).unwrap();

    // The f64-accumulated result should match the f64 reference, NOT the f32 reference.
    assert_eq!(
        result[[0]],
        f64_ref_f32,
        "f64-accumulated batched_matvec should match f64 reference. \
         Got {}, expected {} (f64 ref: {}, f32 naive: {})",
        result[[0]],
        f64_ref_f32,
        f64_ref,
        f32_naive,
    );

    // The result should differ from naive f32 accumulation.
    assert_ne!(
        result[[0]],
        f32_naive,
        "batched_matvec result matches naive f32 — f64 accumulation may have regressed",
    );
}
