// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::NyError;

// ==================== Softmax Affine Tests ====================

#[test]
fn test_softmax_affine_concrete_1d() {
    // Test softmax on concrete (no error) zonotope
    // softmax([1,2,3]) = [0.09003, 0.24473, 0.66524]
    let values = arr1(&[1.0_f32, 2.0, 3.0]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    let result = z.softmax_affine(-1).unwrap();
    let center = result.center();

    // Compute expected softmax
    let e1 = 1.0_f32.exp();
    let e2 = 2.0_f32.exp();
    let e3 = 3.0_f32.exp();
    let sum = e1 + e2 + e3;
    let expected = [e1 / sum, e2 / sum, e3 / sum];

    for (i, &exp) in expected.iter().enumerate() {
        assert!(
            (center[i] - exp).abs() < 1e-5,
            "softmax[{}] = {}, got {}",
            i,
            exp,
            center[i]
        );
    }
}

#[test]
fn test_softmax_affine_uniform() {
    // Test softmax on uniform input: softmax([1,1,1]) = [1/3, 1/3, 1/3]
    let values = arr1(&[1.0_f32, 1.0, 1.0]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    let result = z.softmax_affine(-1).unwrap();
    let center = result.center();

    for i in 0..3 {
        assert!(
            (center[i] - 1.0 / 3.0).abs() < 1e-5,
            "uniform softmax[{}] should be 1/3, got {}",
            i,
            center[i]
        );
    }
}

#[test]
fn test_softmax_affine_2d() {
    // Test softmax on 2D input (seq_len=2, dim=3)
    let values = arr2(&[[1.0_f32, 2.0, 3.0], [0.0, 0.0, 0.0]]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    let result = z.softmax_affine(-1).unwrap();
    let center = result.center();

    // Row 0: softmax([1,2,3])
    let e1 = 1.0_f32.exp();
    let e2 = 2.0_f32.exp();
    let e3 = 3.0_f32.exp();
    let sum0 = e1 + e2 + e3;
    let expected0 = [e1 / sum0, e2 / sum0, e3 / sum0];

    // Row 1: softmax([0,0,0]) = [1/3, 1/3, 1/3]
    let expected1 = [1.0 / 3.0; 3];

    for i in 0..3 {
        assert!(
            (center[[0, i]] - expected0[i]).abs() < 1e-5,
            "softmax row0[{}] = {}, got {}",
            i,
            expected0[i],
            center[[0, i]]
        );
        assert!(
            (center[[1, i]] - expected1[i]).abs() < 1e-5,
            "softmax row1[{}] = {}, got {}",
            i,
            expected1[i],
            center[[1, i]]
        );
    }
}

#[test]
fn test_softmax_affine_4d_last_axis() {
    let values = ArrayD::from_shape_vec(
        IxDyn(&[1, 2, 2, 3]),
        vec![
            1.0_f32, 2.0, 3.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0,
        ],
    )
    .expect("valid 4D softmax input");
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);
    let initial_errors = z.n_error_terms;

    let result = z.softmax_affine(-1).unwrap();
    assert_eq!(result.shape(), &[1, 2, 2, 3]);
    assert_eq!(result.n_error_terms, initial_errors + 12);

    let center = result.center();

    let e1 = 1.0_f32.exp();
    let e2 = 2.0_f32.exp();
    let e3 = 3.0_f32.exp();
    let sum = e1 + e2 + e3;
    let expected = [e1 / sum, e2 / sum, e3 / sum];
    for (i, &exp) in expected.iter().enumerate() {
        assert!(
            (center[[0, 0, 0, i]] - exp).abs() < 1e-5,
            "4D softmax row[0,0,0,{}] = {}, got {}",
            i,
            exp,
            center[[0, 0, 0, i]]
        );
        assert!(
            (center[[0, 0, 1, i]] - 1.0 / 3.0).abs() < 1e-5,
            "uniform 4D softmax row[0,0,1,{}] should be 1/3, got {}",
            i,
            center[[0, 0, 1, i]]
        );
    }

    for head in 0..2 {
        for seq in 0..2 {
            let row_sum: f32 = (0..3).map(|d| center[[0, head, seq, d]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "4D softmax center row [0,{head},{seq},:] should sum to 1, got {row_sum}"
            );
        }
    }

    let bounds = result.to_bounded_tensor().unwrap();
    assert!(
        bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .all(|(l, u)| l <= u),
        "4D softmax bounds must remain ordered"
    );
}

#[test]
fn test_softmax_affine_sum_to_one_preserved() {
    // Softmax output sum should be approximately 1
    let values = arr1(&[1.0_f32, -1.0, 0.5, 2.0]);
    let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 0.1);

    let result = z.softmax_affine(-1).unwrap();
    let center = result.center();

    let sum: f32 = center.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax center should sum to 1, got {}",
        sum
    );
}

#[test]
fn test_softmax_affine_causal_masks_future_positions() {
    // Causal softmax must output exactly 0 for masked positions (j > i).
    let seq = 4_usize;
    let center: Vec<f32> = (0..seq * seq).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let values = ArrayD::from_shape_vec(vec![seq, seq], center).expect("shape ok");

    // Shared uncertainty across all entries; masked logits still vary, but should not affect outputs.
    let z = ZonotopeTensor::from_input_shared(&values, 0.25);

    let result = z.softmax_affine_causal(-1).unwrap();

    // Causal softmax adds one approximation error term per element (#2522).
    // n_attn_rows = prefix_size * seq_q = 1 * 4 = 4, seq_k = 4.
    assert_eq!(result.n_error_terms, z.n_error_terms + seq * seq);

    let center_out = result.center();
    for i in 0..seq {
        let mut row_sum = 0.0f32;
        for j in 0..seq {
            let v = center_out[[i, j]];
            if j > i {
                assert!(
                    v.abs() < 1e-6,
                    "masked causal softmax center should be 0 at ({},{}) got {}",
                    i,
                    j,
                    v
                );
            } else {
                row_sum += v;
            }
        }
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "causal softmax row {} center should sum to 1, got {}",
            i,
            row_sum
        );
    }

    let bounds = result.to_bounded_tensor().unwrap();
    for i in 0..seq {
        for j in (i + 1)..seq {
            assert!(
                bounds.upper()[[i, j]] <= 1e-6 && bounds.lower()[[i, j]] >= -1e-6,
                "masked causal softmax bounds should be 0 at ({},{}) got [{},{}]",
                i,
                j,
                bounds.lower()[[i, j]],
                bounds.upper()[[i, j]]
            );
        }
    }
}

#[test]
fn test_softmax_affine_causal_error_term_count_overflow_returns_error_3012() {
    // Keep coeff storage tiny and override only the logical element shape.
    // The checked multiplication must reject the metadata before coeff access.
    let zonotope = ZonotopeTensor {
        coeffs: ArrayD::zeros(IxDyn(&[1, 1, 1, 2])),
        n_error_terms: 0,
        element_shape: vec![usize::MAX, 1, 2],
    };

    let err = zonotope
        .softmax_affine_causal(-1)
        .expect_err("overflowing causal softmax error-term count should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("n_new_error_terms overflows")),
        "expected causal softmax overflow error, got: {err:?}"
    );
}

#[test]
fn test_softmax_affine_error_term_added() {
    // Verify that softmax adds one approximation error term per element (#2522).
    // Before #2522 this was a single shared error term; per-element independent
    // error symbols prevent false cancellation under downstream linear ops.
    let values = arr1(&[1.0_f32, 2.0, 3.0]);
    let dim = values.len();
    let z = ZonotopeTensor::from_input_shared(&values.into_dyn(), 0.1);
    let initial_errors = z.n_error_terms;

    let result = z.softmax_affine(-1).unwrap();

    assert_eq!(
        result.n_error_terms,
        initial_errors + dim,
        "softmax should add one approximation error term per element (dim={dim})"
    );
}

/// Regression test for #2473: softmax_affine uses max(radius_k) instead of sum(radius_k)
/// for the perturbation radius in the approximation error bound.
///
/// A zonotope with per-element error terms has worst-case perturbation radius
/// r = Σ_k ||a_k||_1 (sum over all error terms), NOT max_k ||a_k||_1.
/// Using max underestimates r, which makes the quadratic error bound 0.5 * r²
/// too small, potentially producing bounds that don't contain true concrete outputs.
///
/// This test creates a zonotope with multiple independent error terms and
/// evaluates softmax at points where individual ε_k take extreme (+1 or -1)
/// values in opposite directions. These are reachable points within the zonotope
/// that the max-radius formula cannot account for.
///
/// NOTE: With `from_input_elementwise`, each error term affects only one dimension,
/// so the Jacobian-transformed error coefficients create wide bounds that dominate
/// the approximation error. This test currently passes even with the buggy max-radius
/// because the bounds are loose enough. The bug is more likely to cause unsoundness
/// after several layers of propagation where error terms become correlated across
/// all dimensions. This test serves as a smoke-test foundation — after #2473 is
/// fixed (sum replaces max), it ensures the fix doesn't break corner containment.
#[test]
fn test_softmax_affine_elementwise_soundness_regression_2473() {
    // Create a 3-element zonotope with per-element error terms and LARGE epsilon.
    // This gives 3 error terms, each affecting one element.
    // center = [1.0, 2.0, 3.0], epsilon = 2.0
    //
    // The zonotope represents:
    //   {[1.0 + 2.0*ε₁, 2.0 + 2.0*ε₂, 3.0 + 2.0*ε₃] : ε_i ∈ [-1, 1]}
    //
    // Large epsilon is critical because the max-vs-sum error scales quadratically:
    //   max-based radius: r_max = max_k ||a_k||_1 = 2.0
    //   sum-based radius: r_sum = Σ_k ||a_k||_1 = 6.0
    //   Error underestimate: 0.5*(6.0² - 2.0²) = 0.5*(36-4) = 16.0
    let values = arr1(&[1.0_f32, 2.0, 3.0]);
    let z = ZonotopeTensor::from_input_elementwise(&values.into_dyn(), 2.0);
    assert_eq!(
        z.n_error_terms, 3,
        "elementwise creates one error term per element"
    );

    let result = z.softmax_affine(-1).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    fn softmax_3(x: [f32; 3]) -> [f32; 3] {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = e.iter().sum();
        [e[0] / sum, e[1] / sum, e[2] / sum]
    }

    // Test at all 8 corners of the zonotope (ε_i ∈ {-1, +1}).
    // These are concrete points inside the zonotope, so the zonotope bounds
    // MUST contain them for soundness.
    let eps = 2.0_f32;
    let center = [1.0_f32, 2.0, 3.0];
    for signs in 0..8u32 {
        let s0 = if signs & 1 != 0 { 1.0 } else { -1.0 };
        let s1 = if signs & 2 != 0 { 1.0 } else { -1.0 };
        let s2 = if signs & 4 != 0 { 1.0 } else { -1.0 };
        let point = [
            center[0] + eps * s0,
            center[1] + eps * s1,
            center[2] + eps * s2,
        ];
        let s = softmax_3(point);

        for (i, &si) in s.iter().enumerate() {
            assert!(
                bounds.lower()[i] <= si,
                "#2473 regression: zonotope softmax lower bound {} must be <= true softmax {} \
                 at corner ({}, {}, {}) for element {} (lower bound too tight — likely max-vs-sum radius bug)",
                bounds.lower()[i], si, s0, s1, s2, i
            );
            assert!(
                bounds.upper()[i] >= si,
                "#2473 regression: zonotope softmax upper bound {} must be >= true softmax {} \
                 at corner ({}, {}, {}) for element {} (upper bound too tight — likely max-vs-sum radius bug)",
                bounds.upper()[i], si, s0, s1, s2, i
            );
        }
    }
}

/// Targeted soundness test for #2473 using *correlated* error terms.
///
/// With `from_input_elementwise`, each error term only affects one dimension, so
/// Jacobian-transformed error coefficients already create wide bounds that mask the
/// max-vs-sum bug. Here we construct a zonotope with hand-crafted coefficients where
/// each error term affects ALL dimensions, making the total L1 radius much larger
/// than the max single-term L1 radius.
///
/// Zonotope: center = [0, 0, 0] with 3 error terms:
///   a_1 = [1, 1, 0]    → ||a_1||_1 = 2
///   a_2 = [0, 1, 1]    → ||a_2||_1 = 2
///   a_3 = [1, 0, 1]    → ||a_3||_1 = 2
///
/// max_k ||a_k||_1 = 2 (buggy radius)
/// Σ_k   ||a_k||_1 = 6 (correct radius)
///
/// Error bound ratio: (6/2)² = 9x underestimate with max.
/// This makes the buggy code produce much tighter (unsound) error bounds.
#[test]
fn test_softmax_affine_correlated_errors_soundness_2473() {
    // Build zonotope: shape (4, 3) = [center + 3 error terms, 3 elements]
    // center = [0, 0, 0]
    // a_1 = [1, 1, 0], a_2 = [0, 1, 1], a_3 = [1, 0, 1]
    let data: Vec<f32> = vec![
        0.0, 0.0, 0.0, // center
        1.0, 1.0, 0.0, // error term 1
        0.0, 1.0, 1.0, // error term 2
        1.0, 0.0, 1.0, // error term 3
    ];
    let coeffs = ArrayD::from_shape_vec(IxDyn(&[4, 3]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();
    assert_eq!(z.n_error_terms, 3);

    let result = z.softmax_affine(-1).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    fn softmax_3(x: [f32; 3]) -> [f32; 3] {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = e.iter().sum();
        [e[0] / sum, e[1] / sum, e[2] / sum]
    }

    // Test all 8 corners (ε_i ∈ {-1, +1}) of the zonotope.
    // At each corner: x = center + ε₁·a₁ + ε₂·a₂ + ε₃·a₃
    for signs in 0..8u32 {
        let e1 = if signs & 1 != 0 { 1.0f32 } else { -1.0 };
        let e2 = if signs & 2 != 0 { 1.0f32 } else { -1.0 };
        let e3 = if signs & 4 != 0 { 1.0f32 } else { -1.0 };
        let point = [
            0.0 + e1 * 1.0 + e2 * 0.0 + e3 * 1.0,
            0.0 + e1 * 1.0 + e2 * 1.0 + e3 * 0.0,
            0.0 + e1 * 0.0 + e2 * 1.0 + e3 * 1.0,
        ];
        let s = softmax_3(point);

        for (i, &si) in s.iter().enumerate() {
            assert!(
                bounds.lower()[i] <= si,
                "#2473 correlated: lower bound {} > true softmax {} at corner \
                 (ε₁={}, ε₂={}, ε₃={}) elem {} — unsound if max used instead of sum for radius",
                bounds.lower()[i],
                si,
                e1,
                e2,
                e3,
                i
            );
            assert!(
                bounds.upper()[i] >= si,
                "#2473 correlated: upper bound {} < true softmax {} at corner \
                 (ε₁={}, ε₂={}, ε₃={}) elem {} — unsound if max used instead of sum for radius",
                bounds.upper()[i],
                si,
                e1,
                e2,
                e3,
                i
            );
        }
    }
}

/// Strengthened regression test for #2473: detects max-vs-sum radius bug.
///
/// The original `test_softmax_affine_correlated_errors_soundness_2473` test passes
/// even with the buggy `max_radius` formula because at center `[0,0,0]` the
/// Jacobian-transformed error terms create bounds wide enough to absorb the
/// underestimated approximation error. (Issue #2512)
///
/// This test uses:
/// 1. **Asymmetric center** `[1, 0, 0]` where softmax = `[0.576, 0.212, 0.212]`.
///    The Jacobian is smaller for elements 1,2, reducing error term contributions.
/// 2. **Many small error terms** (10 terms, each L1 = 0.3) cycling through 3
///    directions: `[-0.15, 0, 0.15]`, `[-0.15, 0.15, 0]`, `[0, -0.15, 0.15]`.
///    This gives `r_max = 0.3` (buggy) vs `r_sum = 3.0` (correct), a 10x ratio
///    and a 100x ratio in the quadratic error bound.
///
/// With the buggy `max` formula, `approx_error = 0.5 * 0.3² = 0.045`.
/// With the correct `sum` formula, `approx_error = 0.5 * 3.0² = 4.5`.
///
/// The all-ε=+1 corner reaches point `[-0.05, 0, 1.05]` where
/// `softmax[2] ≈ 0.594`, but the buggy upper bound for element 2 is only `0.561`.
/// This is a definitive containment violation (margin = +0.034) that can only
/// be fixed by using `sum` instead of `max` for the perturbation radius.
#[test]
fn test_softmax_affine_correlated_errors_detects_max_vs_sum_2512() {
    // Center [1, 0, 0]: softmax ≈ [0.576, 0.212, 0.212]
    // 10 error terms cycling through 3 zero-sum directions.
    // Each term has L1 = 0.3.
    // r_max = 0.3 (buggy), r_sum = 3.0 (correct), ratio = 100x in error bound.
    let directions: [[f32; 3]; 3] = [
        [-0.15, 0.0, 0.15], // L1 = 0.3
        [-0.15, 0.15, 0.0], // L1 = 0.3
        [0.0, -0.15, 0.15], // L1 = 0.3
    ];

    // Build zonotope: shape (11, 3) = [center + 10 error terms, 3 elements]
    let mut data: Vec<f32> = vec![1.0, 0.0, 0.0]; // center
    for k in 0..10 {
        data.extend_from_slice(&directions[k % 3]);
    }
    let coeffs = ArrayD::from_shape_vec(IxDyn(&[11, 3]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();
    assert_eq!(z.n_error_terms, 10);

    let result = z.softmax_affine(-1).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    fn softmax_3(x: [f32; 3]) -> [f32; 3] {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = e.iter().sum();
        [e[0] / sum, e[1] / sum, e[2] / sum]
    }

    // Test all 1024 corners (ε_k ∈ {-1, +1}, k = 0..9).
    // The all-+1 corner is the known worst case for the max-vs-sum bug.
    let dirs: [[f32; 3]; 3] = [[-0.15, 0.0, 0.15], [-0.15, 0.15, 0.0], [0.0, -0.15, 0.15]];
    for bits in 0..1024u32 {
        let mut point = [1.0_f32, 0.0, 0.0];
        for k in 0..10 {
            let eps: f32 = if bits & (1 << k) != 0 { 1.0 } else { -1.0 };
            let d = &dirs[k % 3];
            for i in 0..3 {
                point[i] += eps * d[i];
            }
        }
        let s = softmax_3(point);

        for (i, &si) in s.iter().enumerate() {
            assert!(
                bounds.lower()[i] <= si,
                "#2512 strengthened: lower bound {} > true softmax {} at corner bits={:#06b} \
                 elem {} — max-vs-sum radius bug detected",
                bounds.lower()[i],
                si,
                bits,
                i
            );
            assert!(
                bounds.upper()[i] >= si,
                "#2512 strengthened: upper bound {} < true softmax {} at corner bits={:#06b} \
                 elem {} — max-vs-sum radius bug detected",
                bounds.upper()[i],
                si,
                bits,
                i
            );
        }
    }
}

/// Soundness test for #2473 on the 2D softmax path with correlated errors.
#[test]
fn test_softmax_affine_2d_correlated_errors_soundness_2473() {
    // Build 2D zonotope: shape (3, 2, 3) = [center + 2 errors, seq=2, dim=3]
    // Per-sequence-position softmax along dim=3.
    // center = [[0,0,0],[0,0,0]]
    // a_1 = [[1,1,0],[0.5,0.5,0]] — affects first two dims of each row
    // a_2 = [[0,1,1],[0,0.5,0.5]] — affects last two dims of each row
    let data: Vec<f32> = vec![
        // center (row 0, row 1)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // error term 1
        1.0, 1.0, 0.0, 0.5, 0.5, 0.0, // error term 2
        0.0, 1.0, 1.0, 0.0, 0.5, 0.5,
    ];
    let coeffs = ArrayD::from_shape_vec(IxDyn(&[3, 2, 3]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();
    assert_eq!(z.n_error_terms, 2);

    let result = z.softmax_affine(-1).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    fn softmax_3(x: [f32; 3]) -> [f32; 3] {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = e.iter().sum();
        [e[0] / sum, e[1] / sum, e[2] / sum]
    }

    // Test all 4 corners (ε_1, ε_2 ∈ {-1, +1})
    for signs in 0..4u32 {
        let e1 = if signs & 1 != 0 { 1.0f32 } else { -1.0 };
        let e2 = if signs & 2 != 0 { 1.0f32 } else { -1.0 };

        // Row 0: x = [e1*1 + e2*0, e1*1 + e2*1, e1*0 + e2*1] = [e1, e1+e2, e2]
        let row0 = [e1, e1 + e2, e2];
        let s0 = softmax_3(row0);

        // Row 1: x = [e1*0.5, e1*0.5+e2*0.5, e2*0.5]
        let row1 = [e1 * 0.5, e1 * 0.5 + e2 * 0.5, e2 * 0.5];
        let s1 = softmax_3(row1);

        for i in 0..3 {
            assert!(
                bounds.lower()[[0, i]] <= s0[i],
                "#2473 2D correlated: row 0 lower {} > true {} at (ε₁={}, ε₂={}) elem {}",
                bounds.lower()[[0, i]],
                s0[i],
                e1,
                e2,
                i
            );
            assert!(
                bounds.upper()[[0, i]] >= s0[i],
                "#2473 2D correlated: row 0 upper {} < true {} at (ε₁={}, ε₂={}) elem {}",
                bounds.upper()[[0, i]],
                s0[i],
                e1,
                e2,
                i
            );
            assert!(
                bounds.lower()[[1, i]] <= s1[i],
                "#2473 2D correlated: row 1 lower {} > true {} at (ε₁={}, ε₂={}) elem {}",
                bounds.lower()[[1, i]],
                s1[i],
                e1,
                e2,
                i
            );
            assert!(
                bounds.upper()[[1, i]] >= s1[i],
                "#2473 2D correlated: row 1 upper {} < true {} at (ε₁={}, ε₂={}) elem {}",
                bounds.upper()[[1, i]],
                s1[i],
                e1,
                e2,
                i
            );
        }
    }
}

/// Soundness test for #2473 on the causal softmax path with correlated errors.
#[test]
fn test_softmax_affine_causal_correlated_errors_soundness_2473() {
    // Build causal softmax zonotope: shape (3, 3, 3) = [center + 2 errors, seq_q=3, seq_k=3]
    // Causal mask: row i only attends to keys 0..=i.
    let data: Vec<f32> = vec![
        // center: 3x3 matrix
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        // error term 1: affects pairs of entries
        1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, // error term 2
        0.0, 1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0,
    ];
    let coeffs = ArrayD::from_shape_vec(IxDyn(&[3, 3, 3]), data).unwrap();
    let z = ZonotopeTensor::new(coeffs).unwrap();
    assert_eq!(z.n_error_terms, 2);

    let result = z.softmax_affine_causal(-1).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();

    fn softmax_prefix(x: &[f32]) -> Vec<f32> {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = e.iter().sum();
        e.iter().map(|v| v / sum).collect()
    }

    // Test all 4 corners
    for signs in 0..4u32 {
        let e1 = if signs & 1 != 0 { 1.0f32 } else { -1.0 };
        let e2 = if signs & 2 != 0 { 1.0f32 } else { -1.0 };

        // Compute concrete point: x[i,j] = center[i,j] + e1*a1[i,j] + e2*a2[i,j]
        let a1 = [1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0];
        let a2 = [0.0, 1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0];

        // For each query row, compute causal softmax over allowed prefix
        for qi in 0..3usize {
            let allowed = qi + 1;
            let row_start = qi * 3;
            let logits: Vec<f32> = (0..allowed)
                .map(|j| e1 * a1[row_start + j] + e2 * a2[row_start + j])
                .collect();
            let s = softmax_prefix(&logits);

            for (j, &sj) in s.iter().enumerate() {
                assert!(
                    bounds.lower()[[qi, j]] <= sj,
                    "#2473 causal correlated: lower {} > true {} at row {} col {} (ε₁={}, ε₂={})",
                    bounds.lower()[[qi, j]],
                    sj,
                    qi,
                    j,
                    e1,
                    e2
                );
                assert!(
                    bounds.upper()[[qi, j]] >= sj,
                    "#2473 causal correlated: upper {} < true {} at row {} col {} (ε₁={}, ε₂={})",
                    bounds.upper()[[qi, j]],
                    sj,
                    qi,
                    j,
                    e1,
                    e2
                );
            }

            // Masked positions must be 0
            for j in allowed..3 {
                assert!(
                    bounds.upper()[[qi, j]].abs() <= 1e-6,
                    "causal masked position ({},{}) should be 0, got bounds [{},{}]",
                    qi,
                    j,
                    bounds.lower()[[qi, j]],
                    bounds.upper()[[qi, j]]
                );
            }
        }
    }
}

// ==================== Per-element error cancellation regression tests (#2522) ====================

/// Regression test for #2522: softmax 1D approximation errors must be independent per element
/// so they cannot cancel under downstream [1, -1] linear projection.
#[test]
fn test_softmax_affine_1d_linear_projection_soundness_2522() {
    use ndarray::arr2;

    let values = arr1(&[0.5_f32, 0.5]);
    let z = ZonotopeTensor::from_input_elementwise(&values.into_dyn(), 1.0);

    let sm_z = z.softmax_affine(-1).unwrap();
    let dim = 2;
    assert_eq!(
        sm_z.n_error_terms(),
        z.n_error_terms() + dim,
        "softmax_affine 1D should add one error symbol per element (#2522)"
    );

    let weight = arr2(&[[1.0_f32, -1.0]]);
    let projected = sm_z.linear(&weight, None).unwrap();
    let bounds = projected.to_bounded_tensor().unwrap();

    fn softmax2(x: [f32; 2]) -> [f32; 2] {
        let mx = x[0].max(x[1]);
        let e0 = (x[0] - mx).exp();
        let e1 = (x[1] - mx).exp();
        let s = e0 + e1;
        [e0 / s, e1 / s]
    }

    let mut true_min = f32::INFINITY;
    let mut true_max = f32::NEG_INFINITY;
    for &e0 in &[-1.0_f32, 1.0] {
        for &e1 in &[-1.0_f32, 1.0] {
            let x = [0.5 + 1.0 * e0, 0.5 + 1.0 * e1];
            let sm = softmax2(x);
            let y = sm[0] - sm[1];
            true_min = true_min.min(y);
            true_max = true_max.max(y);
        }
    }

    assert!(
        bounds.lower()[0] <= true_min + 1e-4,
        "#2522 softmax 1D: lower bound {} should contain true min {} \
         (cancellation if shared error terms)",
        bounds.lower()[0],
        true_min
    );
    assert!(
        bounds.upper()[0] >= true_max - 1e-4,
        "#2522 softmax 1D: upper bound {} should contain true max {} \
         (cancellation if shared error terms)",
        bounds.upper()[0],
        true_max
    );
}

/// Regression test for #2522: softmax 2D approximation errors must be independent per element.
#[test]
fn test_softmax_affine_2d_linear_projection_soundness_2522() {
    use ndarray::arr2;

    let values = arr2(&[[0.5_f32, 0.5]]);
    let z = ZonotopeTensor::from_input_2d(&values, 1.0);

    let sm_z = z.softmax_affine(-1).unwrap();
    assert_eq!(
        sm_z.n_error_terms(),
        z.n_error_terms() + values.len(),
        "softmax_affine 2D should add one error symbol per element (#2522)"
    );

    let weight = arr2(&[[1.0_f32, -1.0]]);
    let projected = sm_z.linear(&weight, None).unwrap();
    let bounds = projected.to_bounded_tensor().unwrap();

    fn softmax2(x: [f32; 2]) -> [f32; 2] {
        let mx = x[0].max(x[1]);
        let e0 = (x[0] - mx).exp();
        let e1 = (x[1] - mx).exp();
        let s = e0 + e1;
        [e0 / s, e1 / s]
    }

    let mut true_min = f32::INFINITY;
    let mut true_max = f32::NEG_INFINITY;
    for &e0 in &[-1.0_f32, 1.0] {
        for &e1 in &[-1.0_f32, 1.0] {
            let x = [0.5 + 1.0 * e0, 0.5 + 1.0 * e1];
            let sm = softmax2(x);
            let y = sm[0] - sm[1];
            true_min = true_min.min(y);
            true_max = true_max.max(y);
        }
    }

    assert!(
        bounds.lower()[[0, 0]] <= true_min + 1e-4,
        "#2522 softmax 2D: lower bound {} should contain true min {}",
        bounds.lower()[[0, 0]],
        true_min
    );
    assert!(
        bounds.upper()[[0, 0]] >= true_max - 1e-4,
        "#2522 softmax 2D: upper bound {} should contain true max {}",
        bounds.upper()[[0, 0]],
        true_max
    );
}

/// Regression test for #2522: causal softmax approximation errors must be independent per element.
#[test]
fn test_softmax_affine_causal_linear_projection_soundness_2522() {
    use ndarray::arr2;

    // 2x2 attention matrix: row 0 sees only col 0, row 1 sees cols 0-1.
    let values = arr2(&[[0.5_f32, 0.5], [0.5, 0.5]]);
    let z = ZonotopeTensor::from_input_2d(&values, 1.0);

    let sm_z = z.softmax_affine_causal(-1).unwrap();
    let seq_q = 2;
    let seq_k = 2;
    assert_eq!(
        sm_z.n_error_terms(),
        z.n_error_terms() + seq_q * seq_k,
        "softmax_affine_causal should add per-element error symbols (#2522)"
    );

    // Test row 1 (sees cols 0-1): project through [1, -1] to detect cancellation.
    fn softmax2(x: [f32; 2]) -> [f32; 2] {
        let mx = x[0].max(x[1]);
        let e0 = (x[0] - mx).exp();
        let e1 = (x[1] - mx).exp();
        let s = e0 + e1;
        [e0 / s, e1 / s]
    }

    let bounds = sm_z.to_bounded_tensor().unwrap();

    // from_input_2d creates per-element error terms, so values[1,0] and values[1,1] vary independently.
    let mut true_min = f32::INFINITY;
    let mut true_max = f32::NEG_INFINITY;
    for &e_10 in &[-1.0_f32, 1.0] {
        for &e_11 in &[-1.0_f32, 1.0] {
            let x = [0.5 + 1.0 * e_10, 0.5 + 1.0 * e_11];
            let sm = softmax2(x);
            let y = sm[0] - sm[1]; // [1, -1] projection
            true_min = true_min.min(y);
            true_max = true_max.max(y);
        }
    }

    // Compute projected bounds for row 1: sm_z[1,0] - sm_z[1,1]
    let proj_lower = bounds.lower()[[1, 0]] - bounds.upper()[[1, 1]];
    let proj_upper = bounds.upper()[[1, 0]] - bounds.lower()[[1, 1]];

    assert!(
        proj_lower <= true_min + 1e-4,
        "#2522 causal softmax: projected lower {} should contain true min {}",
        proj_lower,
        true_min
    );
    assert!(
        proj_upper >= true_max - 1e-4,
        "#2522 causal softmax: projected upper {} should contain true max {}",
        proj_upper,
        true_max
    );
}

// NaN safety tests (#2676 Site 1) are in softmax_nan.rs to keep this file under 1000 lines.
