// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::common::BoundPropagation;
use crate::{LinearBounds, LinearLayer};
use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use proptest::prelude::*;

use super::{sample_points, valid_interval, FP_TOLERANCE};

// =============================================================================
// LINEAR LAYER SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Linear layer IBP soundness: for any x in input bounds, Wx+b is in output bounds.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_linear_ibp_2x2(
        w11 in -5.0f32..5.0,
        w12 in -5.0f32..5.0,
        w21 in -5.0f32..5.0,
        w22 in -5.0f32..5.0,
        b1 in -5.0f32..5.0,
        b2 in -5.0f32..5.0,
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
    ) {
        let weight = arr2(&[[w11, w12], [w21, w22]]);
        let bias = arr1(&[b1, b2]);
        let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let output = linear.propagate_ibp(&input).unwrap();

        // Test multiple concrete points
        for x1 in sample_points(l1, u1, 5) {
            for x2 in sample_points(l2, u2, 5) {
                let x = arr1(&[x1, x2]);
                let y = weight.dot(&x) + &bias;

                for i in 0..2 {
                    prop_assert!(
                        output.lower()[[i]] - FP_TOLERANCE <= y[i] && y[i] <= output.upper()[[i]] + FP_TOLERANCE,
                        "Linear soundness violation at output {}: y=Wx+b where x=[{}, {}] gives y[{}]={}, not in [{}, {}]",
                        i, x1, x2, i, y[i], output.lower()[[i]], output.upper()[[i]]
                    );
                }
            }
        }
    }

    /// Linear layer with larger dimensions (5x3).
#[ntest::timeout(10000)]
    #[test]
    fn soundness_linear_ibp_5x3(
        weights in prop::collection::vec(-3.0f32..3.0, 15),  // 5*3 = 15 weights
        biases in prop::collection::vec(-3.0f32..3.0, 5),
        bounds in prop::collection::vec(valid_interval(5.0), 3),
    ) {
        // Reshape weights to 5x3 matrix
        let weight = Array2::from_shape_vec((5, 3), weights).unwrap();
        let bias = Array1::from_vec(biases);
        let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

        // Create input bounds
        let lower_vec: Vec<f32> = bounds.iter().map(|(l, _)| *l).collect();
        let upper_vec: Vec<f32> = bounds.iter().map(|(_, u)| *u).collect();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), lower_vec).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), upper_vec).unwrap()
        ).unwrap();

        let output = linear.propagate_ibp(&input).unwrap();

        // Test corner points and center
        for corner in 0..8 {  // 2^3 corners
            let mut x_vec = Vec::new();
            for (i, (l, u)) in bounds.iter().enumerate() {
                let use_upper = (corner >> i) & 1 == 1;
                x_vec.push(if use_upper { *u } else { *l });
            }
            let x = Array1::from_vec(x_vec);
            let y = weight.dot(&x) + &bias;

            for i in 0..5 {
                prop_assert!(
                    output.lower()[[i]] - FP_TOLERANCE <= y[i] && y[i] <= output.upper()[[i]] + FP_TOLERANCE,
                    "Linear 5x3 soundness violation at output {}: y[{}]={} not in [{}, {}]",
                    i, i, y[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }
}

/// Regression for #2183: Linear CROWN bias path must accumulate in f64 and
/// apply directed rounding when casting to f32.
/// Converted from proptest with `_case in 0u8..1` (zero randomization).
#[ntest::timeout(10000)]
#[test]
fn directed_rounding_linear_crown_bias_2183() {
    let out_features = 100usize;
    let linear = LinearLayer::new(
        Array2::zeros((out_features, 1)),
        Some(Array1::from_elem(out_features, 0.1_f32)),
    )
    .unwrap();

    let incoming = LinearBounds::new(
        Array2::from_elem((1, out_features), 1.0_f32),
        arr1(&[0.0_f32]),
        Array2::from_elem((1, out_features), 1.0_f32),
        arr1(&[0.0_f32]),
    )
    .unwrap();

    let result = linear
        .propagate_linear(&incoming)
        .expect("Linear CROWN failed")
        .into_owned();

    let true_f64: f64 = (0..out_features).map(|_| 0.1_f32 as f64).sum();
    let nearest = true_f64 as f32;
    let expected_lower = if nearest as f64 <= true_f64 {
        nearest
    } else {
        next_down_f32(nearest)
    };
    let expected_upper = if nearest as f64 >= true_f64 {
        nearest
    } else {
        next_up_f32(nearest)
    };

    let mut f32_sum = 0.0_f32;
    for _ in 0..out_features {
        f32_sum += 0.1_f32;
    }
    assert_ne!(
        f32_sum.to_bits(),
        (true_f64 as f32).to_bits(),
        "test setup must exercise f64 vs f32 accumulation divergence",
    );

    assert_eq!(
        result.lower_b[0].to_bits(),
        expected_lower.to_bits(),
        "Linear lower_b must be the tight directed f32 rounding of the f64 accumulation",
    );
    assert_eq!(
        result.upper_b[0].to_bits(),
        expected_upper.to_bits(),
        "Linear upper_b must be the tight directed f32 rounding of the f64 accumulation",
    );
    assert!(
        (result.lower_b[0] as f64) <= true_f64,
        "Linear lower_b must stay <= true f64 bias",
    );
    assert!(
        (result.upper_b[0] as f64) >= true_f64,
        "Linear upper_b must stay >= true f64 bias",
    );
}
