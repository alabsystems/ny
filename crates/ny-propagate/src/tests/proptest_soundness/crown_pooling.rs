// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward soundness proptests for pooling layers.
//!
//! AveragePool is linear so CROWN backward should give exact (sound) bounds.
//! MaxPool2d uses definite-winner routing or constant IBP-fallback bounds.
//!
//! Uses 1-channel 2x2 inputs (4 elements) with kernel 2x2 stride 2 to produce
//! a single scalar output, allowing exhaustive sampling of the input space via
//! nested loops over `sample_points`.
//!
//! Part of #40.

use crate::layers::pooling::{AveragePoolLayer, MaxPool2dLayer};
use crate::LinearBounds;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use proptest::prelude::*;

use super::sample_points;

/// Tolerance for pooling CROWN soundness checks.
const POOL_CROWN_TOLERANCE: f32 = 1e-4;

/// Concretize CROWN bounds at a concrete point for a single output.
fn concretize_at(result: &LinearBounds, x: &[f32], output_idx: usize) -> (f32, f32) {
    let mut lb = result.lower_b[output_idx];
    let mut ub = result.upper_b[output_idx];
    for (i, &xi) in x.iter().enumerate() {
        lb += result.lower_a[[output_idx, i]] * xi;
        ub += result.upper_a[[output_idx, i]] * xi;
    }
    (lb, ub)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    // =========================================================================
    // AveragePool CROWN backward soundness (identity bounds)
    // =========================================================================

    /// AveragePool is linear: CROWN backward should give exact bounds.
    /// 1-channel 2x2 input, kernel 2x2, stride 2 → 1 output (scalar average).
    /// Exhaustive 4-nested-loop sampling of input space.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_avgpool_crown_identity(
        l0 in -10.0f32..10.0, l1 in -10.0f32..10.0,
        l2 in -10.0f32..10.0, l3 in -10.0f32..10.0,
        d0 in 0.0f32..10.0, d1 in 0.0f32..10.0,
        d2 in 0.0f32..10.0, d3 in 0.0f32..10.0,
    ) {
        let lowers = [l0, l1, l2, l3];
        let uppers = [l0 + d0, l1 + d1, l2 + d2, l3 + d3];

        let lower_arr = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), lowers.to_vec()).unwrap();
        let upper_arr = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), uppers.to_vec()).unwrap();
        let pre_activation = BoundedTensor::new(lower_arr, upper_arr).unwrap();

        let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
        let identity = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&identity, &pre_activation).unwrap();

        // Exhaustive sampling: 10^4 = 10000 points
        let samples: Vec<Vec<f32>> = (0..4)
            .map(|i| sample_points(lowers[i], uppers[i], 10))
            .collect();

        for &x0 in &samples[0] {
            for &x1 in &samples[1] {
                for &x2 in &samples[2] {
                    for &x3 in &samples[3] {
                        let xs = [x0, x1, x2, x3];
                        let true_val = (x0 + x1 + x2 + x3) / 4.0;
                        let (lb, ub) = concretize_at(&result, &xs, 0);

                        let scale_tol = POOL_CROWN_TOLERANCE * true_val.abs().max(1.0);
                        prop_assert!(
                            lb <= true_val + scale_tol,
                            "AvgPool lower violated: lb={lb} > true={true_val}"
                        );
                        prop_assert!(
                            ub + scale_tol >= true_val,
                            "AvgPool upper violated: ub={ub} < true={true_val}"
                        );
                    }
                }
            }
        }
    }

    /// AveragePool CROWN with negative incoming coefficients.
    /// Tests coefficient sign propagation through the linear backward pass.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_avgpool_crown_negative_coeffs(
        l0 in -5.0f32..5.0, l1 in -5.0f32..5.0,
        l2 in -5.0f32..5.0, l3 in -5.0f32..5.0,
        d0 in 0.0f32..5.0, d1 in 0.0f32..5.0,
        d2 in 0.0f32..5.0, d3 in 0.0f32..5.0,
        c in -5.0f32..5.0,
    ) {
        prop_assume!(c.abs() > 0.01);

        let lowers = [l0, l1, l2, l3];
        let uppers = [l0 + d0, l1 + d1, l2 + d2, l3 + d3];

        let lower_arr = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), lowers.to_vec()).unwrap();
        let upper_arr = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), uppers.to_vec()).unwrap();
        let pre_activation = BoundedTensor::new(lower_arr, upper_arr).unwrap();

        let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 1), vec![c]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 1), vec![c]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer.propagate_linear_with_bounds(&incoming, &pre_activation).unwrap();

        let samples: Vec<Vec<f32>> = (0..4)
            .map(|i| sample_points(lowers[i], uppers[i], 10))
            .collect();

        for &x0 in &samples[0] {
            for &x1 in &samples[1] {
                for &x2 in &samples[2] {
                    for &x3 in &samples[3] {
                        let xs = [x0, x1, x2, x3];
                        let avg_val = (x0 + x1 + x2 + x3) / 4.0;
                        let true_val = c * avg_val;
                        let (lb, ub) = concretize_at(&result, &xs, 0);

                        let scale_tol = POOL_CROWN_TOLERANCE * true_val.abs().max(1.0);
                        prop_assert!(
                            lb <= true_val + scale_tol,
                            "AvgPool neg-coeff lower violated: lb={lb} > true={true_val} (c={c})"
                        );
                        prop_assert!(
                            ub + scale_tol >= true_val,
                            "AvgPool neg-coeff upper violated: ub={ub} < true={true_val} (c={c})"
                        );
                    }
                }
            }
        }
    }

    // =========================================================================
    // MaxPool2d CROWN backward soundness (identity bounds)
    // =========================================================================

    /// MaxPool2d with 1-channel 2x2 input, kernel 2x2, stride 2 → 1 output.
    /// Tests both definite-winner routing and IBP-fallback cases.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_maxpool_crown_identity(
        l0 in -10.0f32..10.0, l1 in -10.0f32..10.0,
        l2 in -10.0f32..10.0, l3 in -10.0f32..10.0,
        d0 in 0.0f32..10.0, d1 in 0.0f32..10.0,
        d2 in 0.0f32..10.0, d3 in 0.0f32..10.0,
    ) {
        let lowers = [l0, l1, l2, l3];
        let uppers = [l0 + d0, l1 + d1, l2 + d2, l3 + d3];

        let lower_arr = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), lowers.to_vec()).unwrap();
        let upper_arr = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), uppers.to_vec()).unwrap();
        let pre_activation = BoundedTensor::new(lower_arr, upper_arr).unwrap();

        let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
        let identity = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&identity, &pre_activation).unwrap();

        let samples: Vec<Vec<f32>> = (0..4)
            .map(|i| sample_points(lowers[i], uppers[i], 10))
            .collect();

        for &x0 in &samples[0] {
            for &x1 in &samples[1] {
                for &x2 in &samples[2] {
                    for &x3 in &samples[3] {
                        let xs = [x0, x1, x2, x3];
                        let true_val = x0.max(x1).max(x2).max(x3);
                        let (lb, ub) = concretize_at(&result, &xs, 0);

                        let scale_tol = POOL_CROWN_TOLERANCE * true_val.abs().max(1.0);
                        prop_assert!(
                            lb <= true_val + scale_tol,
                            "MaxPool lower violated: lb={lb} > true={true_val} at ({x0},{x1},{x2},{x3})"
                        );
                        prop_assert!(
                            ub + scale_tol >= true_val,
                            "MaxPool upper violated: ub={ub} < true={true_val} at ({x0},{x1},{x2},{x3})"
                        );
                    }
                }
            }
        }
    }

    // =========================================================================
    // MaxPool2d CROWN with definite winner
    // =========================================================================

    /// MaxPool2d where one input definitively dominates (lower > all others' upper).
    /// Gradient should flow entirely through the winner (identity routing).
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_maxpool_crown_definite_winner(
        base in -5.0f32..5.0,
        base_width in 0.0f32..1.0,
        margin in 0.1f32..5.0,
        win_width in 0.0f32..2.0,
    ) {
        let l_win = base + base_width + margin;
        let u_win = l_win + win_width;
        let u_base = base + base_width;

        prop_assume!(l_win > u_base);

        let lower_arr = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 2]), vec![l_win, base, base, base]
        ).unwrap();
        let upper_arr = ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 2]), vec![u_win, u_base, u_base, u_base]
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower_arr, upper_arr).unwrap();

        let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
        let identity = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&identity, &pre_activation).unwrap();

        // With definite winner at position 0, gradient flows through it entirely
        prop_assert!(
            (result.lower_a[[0, 0]] - 1.0).abs() < 1e-6,
            "Definite winner: lower_a[0,0] should be 1.0, got {}",
            result.lower_a[[0, 0]]
        );
        prop_assert!(
            (result.upper_a[[0, 0]] - 1.0).abs() < 1e-6,
            "Definite winner: upper_a[0,0] should be 1.0, got {}",
            result.upper_a[[0, 0]]
        );
        for i in 1..4 {
            prop_assert!(
                result.lower_a[[0, i]].abs() < 1e-6,
                "Definite winner: lower_a[0,{i}] should be 0.0, got {}",
                result.lower_a[[0, i]]
            );
            prop_assert!(
                result.upper_a[[0, i]].abs() < 1e-6,
                "Definite winner: upper_a[0,{i}] should be 0.0, got {}",
                result.upper_a[[0, i]]
            );
        }

        // Verify soundness at sampled points
        for &x in &sample_points(l_win, u_win, 20) {
            let true_val = x; // Winner is always the max
            let (lb, ub) = concretize_at(&result, &[x, base, base, base], 0);

            let scale_tol = POOL_CROWN_TOLERANCE * true_val.abs().max(1.0);
            prop_assert!(
                lb <= true_val + scale_tol,
                "MaxPool definite-winner lower violated: lb={lb} > true={true_val}"
            );
            prop_assert!(
                ub + scale_tol >= true_val,
                "MaxPool definite-winner upper violated: ub={ub} < true={true_val}"
            );
        }
    }

    // =========================================================================
    // MaxPool2d CROWN with negative incoming coefficients
    // =========================================================================

    /// MaxPool2d CROWN backward with negative incoming coefficients.
    /// Tests sign-switching in the IBP-fallback case.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_maxpool_crown_negative_coeffs(
        l0 in -5.0f32..5.0, l1 in -5.0f32..5.0,
        l2 in -5.0f32..5.0, l3 in -5.0f32..5.0,
        d0 in 0.0f32..5.0, d1 in 0.0f32..5.0,
        d2 in 0.0f32..5.0, d3 in 0.0f32..5.0,
        c in -5.0f32..5.0,
    ) {
        prop_assume!(c.abs() > 0.01);

        let lowers = [l0, l1, l2, l3];
        let uppers = [l0 + d0, l1 + d1, l2 + d2, l3 + d3];

        let lower_arr = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), lowers.to_vec()).unwrap();
        let upper_arr = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), uppers.to_vec()).unwrap();
        let pre_activation = BoundedTensor::new(lower_arr, upper_arr).unwrap();

        let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 1), vec![c]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 1), vec![c]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer.propagate_linear_with_bounds(&incoming, &pre_activation).unwrap();

        let samples: Vec<Vec<f32>> = (0..4)
            .map(|i| sample_points(lowers[i], uppers[i], 10))
            .collect();

        for &x0 in &samples[0] {
            for &x1 in &samples[1] {
                for &x2 in &samples[2] {
                    for &x3 in &samples[3] {
                        let xs = [x0, x1, x2, x3];
                        let max_val = x0.max(x1).max(x2).max(x3);
                        let true_val = c * max_val;
                        let (lb, ub) = concretize_at(&result, &xs, 0);

                        let scale_tol = POOL_CROWN_TOLERANCE * true_val.abs().max(1.0);
                        prop_assert!(
                            lb <= true_val + scale_tol,
                            "MaxPool neg-coeff lower violated: lb={lb} > true={true_val} (c={c})"
                        );
                        prop_assert!(
                            ub + scale_tol >= true_val,
                            "MaxPool neg-coeff upper violated: ub={ub} < true={true_val} (c={c})"
                        );
                    }
                }
            }
        }
    }
}

/// Regression for #2183: MaxPool CROWN fallback bias path must accumulate
/// constants in f64 and apply directed rounding on the final f32 cast.
/// Converted from proptest with `_case in 0u8..1` (zero randomization).
#[ntest::timeout(10000)]
#[test]
fn directed_rounding_maxpool_crown_bias_2183() {
    let layer = MaxPool2dLayer::new((2, 2), (1, 1), (0, 0));

    // 1x11x11 input with overlapping intervals ensures no definite winner in
    // each 2x2 window, so the constant fallback path is used.
    let pre = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 11, 11]), 0.1_f32),
        ArrayD::from_elem(IxDyn(&[1, 11, 11]), 0.2_f32),
    )
    .unwrap();

    // Output is 1x10x10 -> 100 pooled values. One outgoing row aggregates all
    // outputs to create a sensitive f64 accumulation in the CONSTANT-bias path.
    //
    // NOTE: after the sound dense no-winner LOWER relaxation, la>0 / ua<0 route
    // the lower/upper rows LINEARLY through i* (no constant accumulated). The
    // surviving constant-bias arms are la<0 (lower_b += la*max_upper) and ua>0
    // (upper_b += ua*max_upper). We drive THOSE arms here to keep exercising the
    // f64 directed-rounding guard (#2183): la=-1, ua=+1.
    let incoming = LinearBounds::new(
        Array2::from_elem((1, 100), -1.0_f32),
        Array1::zeros(1),
        Array2::from_elem((1, 100), 1.0_f32),
        Array1::zeros(1),
    )
    .unwrap();

    let result = layer
        .propagate_linear_with_bounds(&incoming, &pre)
        .expect("MaxPool CROWN failed");

    // la=-1 < 0 → lower_b += la * max_upper = -0.2 per window (constant path).
    // ua=+1 > 0 → upper_b += ua * max_upper = +0.2 per window (constant path).
    let true_lower_f64: f64 = (0..100).map(|_| -(0.2_f32 as f64)).sum();
    let true_upper_f64: f64 = (0..100).map(|_| 0.2_f32 as f64).sum();
    let expected_lower = next_down_f32(true_lower_f64 as f32);
    let expected_upper = next_up_f32(true_upper_f64 as f32);

    let mut f32_lower_sum = 0.0_f32;
    let mut f32_upper_sum = 0.0_f32;
    for _ in 0..100 {
        f32_lower_sum += -0.2_f32;
        f32_upper_sum += 0.2_f32;
    }
    assert_ne!(
        f32_lower_sum.to_bits(),
        (true_lower_f64 as f32).to_bits(),
        "test setup must exercise f64 vs f32 lower accumulation divergence",
    );
    assert_ne!(
        f32_upper_sum.to_bits(),
        (true_upper_f64 as f32).to_bits(),
        "test setup must exercise f64 vs f32 upper accumulation divergence",
    );

    assert_eq!(
        result.lower_b[0].to_bits(),
        expected_lower.to_bits(),
        "MaxPool lower_b must use next_down_f32 on f64 accumulation",
    );
    assert_eq!(
        result.upper_b[0].to_bits(),
        expected_upper.to_bits(),
        "MaxPool upper_b must use next_up_f32 on f64 accumulation",
    );
    assert!(
        (result.lower_b[0] as f64) <= true_lower_f64,
        "MaxPool lower_b must stay <= true f64 lower constant",
    );
    assert!(
        (result.upper_b[0] as f64) >= true_upper_f64,
        "MaxPool upper_b must stay >= true f64 upper constant",
    );
}
