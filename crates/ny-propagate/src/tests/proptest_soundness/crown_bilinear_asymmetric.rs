// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Rectangular BilinearCrown soundness proptests.
//!
//! `crown_bilinear.rs` already covers square `transpose_b=true` soundness.
//! These tests close the remaining #3289 gap by exercising rectangular
//! `m != n` shapes and the `transpose_b=false` production path against
//! concrete matmul outputs.

use crate::layers::binary_ops::BilinearCrownLayer;
use crate::LinearBounds;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

const BILINEAR_TOLERANCE: f32 = 1e-3;

fn make_bt(lower: &[f32], upper: &[f32], shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

fn selector_patterns(len: usize) -> Vec<Vec<f32>> {
    vec![
        vec![0.0; len],
        vec![1.0; len],
        vec![0.5; len],
        (0..len)
            .map(|idx| if idx % 2 == 0 { 0.0 } else { 1.0 })
            .collect(),
        (0..len)
            .map(|idx| if idx % 2 == 0 { 1.0 } else { 0.0 })
            .collect(),
        (0..len)
            .map(|idx| match idx % 3 {
                0 => 0.25,
                1 => 0.75,
                _ => 0.5,
            })
            .collect(),
    ]
}

fn realize_interval(lower: &[f32], upper: &[f32], selector: &[f32]) -> Vec<f32> {
    lower
        .iter()
        .zip(upper.iter())
        .zip(selector.iter())
        .map(|((&lo, &hi), &t)| lo + t * (hi - lo))
        .collect()
}

fn matmul_flat(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    transpose_b: bool,
) -> Vec<f32> {
    let (m, k) = a_shape;
    let n = if transpose_b {
        assert_eq!(b_shape.1, k, "transpose_b=true requires b_shape.1 == k");
        b_shape.0
    } else {
        assert_eq!(b_shape.0, k, "transpose_b=false requires b_shape.0 == k");
        b_shape.1
    };

    let mut output = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for l in 0..k {
                let a_idx = i * k + l;
                let b_idx = if transpose_b { j * k + l } else { l * n + j };
                sum += a[a_idx] * b[b_idx];
            }
            output[i * n + j] = sum;
        }
    }
    output
}

fn assert_rectangular_bilinear_soundness(
    transpose_b: bool,
    q_lower: &[f32],
    q_upper: &[f32],
    q_shape: (usize, usize),
    k_lower: &[f32],
    k_upper: &[f32],
    k_shape: (usize, usize),
) -> Result<(), TestCaseError> {
    let q_bounds = make_bt(q_lower, q_upper, &[q_shape.0, q_shape.1]);
    let k_bounds = make_bt(k_lower, k_upper, &[k_shape.0, k_shape.1]);
    let output_size = q_shape.0 * if transpose_b { k_shape.0 } else { k_shape.1 };

    let layer = BilinearCrownLayer::new(transpose_b, None);
    let identity = LinearBounds::identity(output_size);
    let (bounds_q, bounds_k) = layer
        .propagate_linear_binary(&identity, &q_bounds, &k_bounds)
        .map_err(|err| TestCaseError::fail(format!("propagate_linear_binary failed: {err}")))?;

    let concrete_q = bounds_q.concretize(&q_bounds);
    let concrete_k = bounds_k.concretize(&k_bounds);
    let crown_lower: Vec<f32> = concrete_q
        .lower()
        .iter()
        .zip(concrete_k.lower().iter())
        .map(|(&q, &k)| q + k)
        .collect();
    let crown_upper: Vec<f32> = concrete_q
        .upper()
        .iter()
        .zip(concrete_k.upper().iter())
        .map(|(&q, &k)| q + k)
        .collect();

    let q_patterns = selector_patterns(q_lower.len());
    let k_patterns = selector_patterns(k_lower.len());

    for (q_pattern_idx, q_pattern) in q_patterns.iter().enumerate() {
        let q = realize_interval(q_lower, q_upper, q_pattern);
        for (k_pattern_idx, k_pattern) in k_patterns.iter().enumerate() {
            let k = realize_interval(k_lower, k_upper, k_pattern);
            let truth = matmul_flat(&q, q_shape, &k, k_shape, transpose_b);

            for (out_idx, &truth_val) in truth.iter().enumerate() {
                prop_assert!(
                    truth_val >= crown_lower[out_idx] - BILINEAR_TOLERANCE,
                    "lower violation at output {out_idx}: truth={truth_val} < lb={} \
                     (transpose_b={}, q_pattern={}, k_pattern={})",
                    crown_lower[out_idx],
                    transpose_b,
                    q_pattern_idx,
                    k_pattern_idx,
                );
                prop_assert!(
                    truth_val <= crown_upper[out_idx] + BILINEAR_TOLERANCE,
                    "upper violation at output {out_idx}: truth={truth_val} > ub={} \
                     (transpose_b={}, q_pattern={}, k_pattern={})",
                    crown_upper[out_idx],
                    transpose_b,
                    q_pattern_idx,
                    k_pattern_idx,
                );
            }
        }
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(120) })]

    /// Rectangular transpose-b soundness: Q=[m=2, k=2], K=[n=3, k=2], output=[2, 3].
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_bilinear_crown_asymmetric_transpose_true(
        (lq0, uq0) in super::valid_interval(2.0),
        (lq1, uq1) in super::valid_interval(2.0),
        (lq2, uq2) in super::valid_interval(2.0),
        (lq3, uq3) in super::valid_interval(2.0),
        (lk0, uk0) in super::valid_interval(2.0),
        (lk1, uk1) in super::valid_interval(2.0),
        (lk2, uk2) in super::valid_interval(2.0),
        (lk3, uk3) in super::valid_interval(2.0),
        (lk4, uk4) in super::valid_interval(2.0),
        (lk5, uk5) in super::valid_interval(2.0),
    ) {
        let q_lower = [lq0, lq1, lq2, lq3];
        let q_upper = [uq0, uq1, uq2, uq3];
        let k_lower = [lk0, lk1, lk2, lk3, lk4, lk5];
        let k_upper = [uk0, uk1, uk2, uk3, uk4, uk5];

        assert_rectangular_bilinear_soundness(
            true,
            &q_lower,
            &q_upper,
            (2, 2),
            &k_lower,
            &k_upper,
            (3, 2),
        )?;
    }

    /// Rectangular non-transposed soundness: Q=[m=2, k=2], K=[k=2, n=3], output=[2, 3].
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_bilinear_crown_asymmetric_transpose_false(
        (lq0, uq0) in super::valid_interval(2.0),
        (lq1, uq1) in super::valid_interval(2.0),
        (lq2, uq2) in super::valid_interval(2.0),
        (lq3, uq3) in super::valid_interval(2.0),
        (lk0, uk0) in super::valid_interval(2.0),
        (lk1, uk1) in super::valid_interval(2.0),
        (lk2, uk2) in super::valid_interval(2.0),
        (lk3, uk3) in super::valid_interval(2.0),
        (lk4, uk4) in super::valid_interval(2.0),
        (lk5, uk5) in super::valid_interval(2.0),
    ) {
        let q_lower = [lq0, lq1, lq2, lq3];
        let q_upper = [uq0, uq1, uq2, uq3];
        let k_lower = [lk0, lk1, lk2, lk3, lk4, lk5];
        let k_upper = [uk0, uk1, uk2, uk3, uk4, uk5];

        assert_rectangular_bilinear_soundness(
            false,
            &q_lower,
            &q_upper,
            (2, 2),
            &k_lower,
            &k_upper,
            (2, 3),
        )?;
    }
}
