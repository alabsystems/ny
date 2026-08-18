// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::common::BoundPropagation;
use crate::layers::BatchNormLayer;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{arr1, Array1, Array2, ArrayD, Axis, IxDyn};
use ny_core::{f64_to_f32_down, f64_to_f32_up};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{batchnorm, valid_interval};

// =============================================================================
// BATCHNORM SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// BatchNorm IBP soundness: for any x in [l, u], batchnorm(x) is in computed bounds.
    ///
    /// Uses inference mode (running statistics) rather than batch statistics.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_batchnorm_ibp_3d(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        // Running statistics - must be positive for variance
        mean0 in -2.0f32..2.0,
        mean1 in -2.0f32..2.0,
        mean2 in -2.0f32..2.0,
        var0 in 0.1f32..5.0,
        var1 in 0.1f32..5.0,
        var2 in 0.1f32..5.0,
        // Scale and shift
        gamma0 in 0.5f32..2.0,
        gamma1 in 0.5f32..2.0,
        gamma2 in 0.5f32..2.0,
        beta0 in -1.0f32..1.0,
        beta1 in -1.0f32..1.0,
        beta2 in -1.0f32..1.0,
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn()
        ).unwrap();

        let ny = Array1::from_vec(vec![gamma0, gamma1, gamma2]);
        let beta = Array1::from_vec(vec![beta0, beta1, beta2]);
        let running_mean = Array1::from_vec(vec![mean0, mean1, mean2]);
        let running_var = Array1::from_vec(vec![var0, var1, var2]);

        // Convert to ArrayD for BatchNormLayer constructor
        let ny_d = ny.clone().into_dyn();
        let beta_d = beta.clone().into_dyn();
        let mean_d = running_mean.clone().into_dyn();
        let var_d = running_var.clone().into_dyn();

        let bn_layer = BatchNormLayer::new(
            &ny_d,
            &beta_d,
            &mean_d,
            &var_d,
            1e-5
        ).unwrap();
        let output = bn_layer.propagate_ibp(&input).unwrap();

        // Test corner points
        let corners = vec![
            arr1(&[l0, l1, l2]),
            arr1(&[u0, u1, u2]),
            arr1(&[u0, l1, l2]),
            arr1(&[l0, u1, l2]),
            arr1(&[l0, l1, u2]),
            arr1(&[f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2)]),
        ];

        for x in corners {
            let bn_x = batchnorm(&x, &ny, &beta, &running_mean, &running_var, 1e-5);

            for i in 0..3 {
                let tol = 1e-6;
                prop_assert!(
                    output.lower()[[i]] - tol <= bn_x[i] && bn_x[i] <= output.upper()[[i]] + tol,
                    "BatchNorm soundness violation: batchnorm({:?})[{}]={} not in [{}, {}]",
                    x, i, bn_x[i], output.lower()[[i]], output.upper()[[i]]
                );
            }
        }
    }
}

/// Regression for #2183: BatchNorm CROWN bias path must accumulate in f64
/// and use directed rounding when casting back to f32.
/// Converted from proptest with `_case in 0u8..1` (zero randomization).
#[ntest::timeout(10000)]
#[test]
fn directed_rounding_batchnorm_crown_bias_2183() {
    let channels = 100usize;
    let layer = BatchNormLayer::from_scale_bias(
        ArrayD::from_elem(IxDyn(&[channels]), 1.0_f32),
        ArrayD::from_elem(IxDyn(&[channels]), 0.1_f32),
    )
    .unwrap();

    let pre = BoundedTensor::new(
        Array1::from_elem(channels, -1.0_f32).into_dyn(),
        Array1::from_elem(channels, 1.0_f32).into_dyn(),
    )
    .unwrap();

    let incoming = LinearBounds::new(
        Array2::from_elem((1, channels), 1.0_f32),
        arr1(&[0.0_f32]),
        Array2::from_elem((1, channels), 1.0_f32),
        arr1(&[0.0_f32]),
    )
    .unwrap();

    let result = layer
        .propagate_linear_with_bounds(&incoming, &pre)
        .expect("BatchNorm CROWN failed");

    let true_f64: f64 = (0..channels).map(|_| 0.1_f32 as f64).sum();
    let expected_lower = f64_to_f32_down(true_f64);
    let expected_upper = f64_to_f32_up(true_f64);

    let mut f32_sum = 0.0_f32;
    for _ in 0..channels {
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
        "BatchNorm lower_b must convert the certified f64 endpoint downward",
    );
    assert_eq!(
        result.upper_b[0].to_bits(),
        expected_upper.to_bits(),
        "BatchNorm upper_b must convert the certified f64 endpoint upward",
    );
    assert!(
        (result.lower_b[0] as f64) <= true_f64,
        "BatchNorm lower_b must stay <= true f64 bias",
    );
    assert!(
        (result.upper_b[0] as f64) >= true_f64,
        "BatchNorm upper_b must stay >= true f64 bias",
    );
}

// =============================================================================
// BATCHNORM BATCHED CROWN BACKWARD SOUNDNESS
// =============================================================================

/// Concretize batched CROWN linear bounds against input interval bounds.
fn concretize_batched_crown(
    result: &BatchedLinearBounds,
    pre_activation: &BoundedTensor,
) -> (Vec<f32>, Vec<f32>) {
    let concrete = result
        .concretize(pre_activation)
        .expect("concretize should not fail for valid bounds");
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// BatchNorm batched CROWN soundness with identity incoming bounds.
    ///
    /// For 1D input [C=3]: verifies that for any x in [l, u],
    /// batchnorm(x) lies within the CROWN-computed bounds.
    /// BatchNorm is exact affine, so tolerance is tight.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_batchnorm_batched_crown_1d_identity(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        gamma0 in 0.5f32..2.0,
        gamma1 in 0.5f32..2.0,
        gamma2 in 0.5f32..2.0,
        beta0 in -1.0f32..1.0,
        beta1 in -1.0f32..1.0,
        beta2 in -1.0f32..1.0,
        mean0 in -2.0f32..2.0,
        mean1 in -2.0f32..2.0,
        mean2 in -2.0f32..2.0,
        var0 in 0.1f32..5.0,
        var1 in 0.1f32..5.0,
        var2 in 0.1f32..5.0,
    ) {
        let num_ch = 3;
        let ny = Array1::from_vec(vec![gamma0, gamma1, gamma2]);
        let beta = Array1::from_vec(vec![beta0, beta1, beta2]);
        let mean = Array1::from_vec(vec![mean0, mean1, mean2]);
        let var = Array1::from_vec(vec![var0, var1, var2]);

        let layer = BatchNormLayer::new(
            &ny.clone().into_dyn(),
            &beta.clone().into_dyn(),
            &mean.clone().into_dyn(),
            &var.clone().into_dyn(),
            1e-5,
        ).unwrap();

        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn(),
        ).unwrap();

        // Identity batched bounds [num_ch, num_ch]
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[num_ch, num_ch]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[num_ch, num_ch]));
        for i in 0..num_ch {
            la[[i, i]] = 1.0;
            ua[[i, i]] = 1.0;
        }
        let lb = ArrayD::<f32>::zeros(IxDyn(&[num_ch]));
        let ub = ArrayD::<f32>::zeros(IxDyn(&[num_ch]));
        let bounds = BatchedLinearBounds::new(
            la, lb, ua, ub,
            vec![num_ch], vec![num_ch],
        ).unwrap();

        let result = layer
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| TestCaseError::fail(format!("BatchNorm batched CROWN failed: {e}")))?;

        let (crown_lower, crown_upper) = concretize_batched_crown(&result, &input);

        // Sample corner and midpoints
        let samples: Vec<[f32; 3]> = vec![
            [l0, l1, l2],
            [u0, u1, u2],
            [u0, l1, l2],
            [l0, u1, l2],
            [l0, l1, u2],
            [f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2)],
        ];

        for x_arr in &samples {
            let x = arr1(x_arr);
            let bn_x = batchnorm(&x, &ny, &beta, &mean, &var, 1e-5);

            for i in 0..num_ch {
                // Tightened 1e-4 → 1e-6 to match the IBP test: BatchNorm is exact
                // affine, and the CROWN path now folds the scale/bias precompute
                // error outward, so the bound must hold to ~1 ulp at these
                // magnitudes (#batchnorm-ibp-directed-rounding, CROWN counterpart).
                let tol = 1e-6;
                prop_assert!(
                    crown_lower[i] - tol <= bn_x[i],
                    "BatchNorm batched CROWN lower violation: crown_lower[{i}]={} > bn({:?})[{i}]={}",
                    crown_lower[i], x, bn_x[i],
                );
                prop_assert!(
                    bn_x[i] <= crown_upper[i] + tol,
                    "BatchNorm batched CROWN upper violation: bn({:?})[{i}]={} > crown_upper[{i}]={}",
                    x, bn_x[i], crown_upper[i],
                );
            }
        }
    }

    /// BatchNorm batched CROWN with negative scale channels.
    ///
    /// Negative scale flips the sign of coefficients during CROWN backward.
    /// Verifies soundness is maintained when some channels have negative scale.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_batchnorm_batched_crown_negative_scale(
        (l0, u0) in valid_interval(3.0),
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let num_ch = 3;
        // Negative scale for channel 1
        let scale = ArrayD::from_shape_vec(IxDyn(&[num_ch]), vec![2.0, -1.5, 0.5]).unwrap();
        let bias = ArrayD::from_shape_vec(IxDyn(&[num_ch]), vec![0.1, -0.3, 0.7]).unwrap();
        let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2]).into_dyn(),
            arr1(&[u0, u1, u2]).into_dyn(),
        ).unwrap();

        // Identity batched bounds
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[num_ch, num_ch]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[num_ch, num_ch]));
        for i in 0..num_ch {
            la[[i, i]] = 1.0;
            ua[[i, i]] = 1.0;
        }
        let lb = ArrayD::<f32>::zeros(IxDyn(&[num_ch]));
        let ub = ArrayD::<f32>::zeros(IxDyn(&[num_ch]));
        let bounds = BatchedLinearBounds::new(
            la, lb, ua, ub,
            vec![num_ch], vec![num_ch],
        ).unwrap();

        let result = layer
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| TestCaseError::fail(format!("BatchNorm batched CROWN failed: {e}")))?;

        let (crown_lower, crown_upper) = concretize_batched_crown(&result, &input);

        // from_scale_bias: y = scale * x + bias (mean=0, var=1, eps=0)
        let ny = arr1(&[2.0, -1.5, 0.5]);
        let beta = arr1(&[0.1, -0.3, 0.7]);
        let mean = arr1(&[0.0, 0.0, 0.0]);
        let var = arr1(&[1.0, 1.0, 1.0]);
        let eps = 0.0_f32;

        let samples: Vec<[f32; 3]> = vec![
            [l0, l1, l2],
            [u0, u1, u2],
            [u0, l1, u2],
            [l0, u1, l2],
            [f32::midpoint(l0, u0), f32::midpoint(l1, u1), f32::midpoint(l2, u2)],
        ];

        for x_arr in &samples {
            let x = arr1(x_arr);
            let bn_x = batchnorm(&x, &ny, &beta, &mean, &var, eps);

            for i in 0..num_ch {
                // Tightened 1e-4 → 1e-6 to match the IBP test: BatchNorm is exact
                // affine, and the CROWN path now folds the scale/bias precompute
                // error outward, so the bound must hold to ~1 ulp at these
                // magnitudes (#batchnorm-ibp-directed-rounding, CROWN counterpart).
                let tol = 1e-6;
                prop_assert!(
                    crown_lower[i] - tol <= bn_x[i],
                    "negscale lower[{i}]: crown_lower={} > bn_x={}",
                    crown_lower[i], bn_x[i],
                );
                prop_assert!(
                    bn_x[i] <= crown_upper[i] + tol,
                    "negscale upper[{i}]: bn_x={} > crown_upper={}",
                    bn_x[i], crown_upper[i],
                );
            }
        }
    }

    /// BatchNorm batched CROWN: batched vs scalar equivalence.
    ///
    /// Verifies that the batched CROWN backward path produces results
    /// equivalent to the scalar path within FP tolerance.
    #[ntest::timeout(10000)]
    #[test]
    fn batchnorm_batched_matches_scalar(
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        scale_vals in prop::collection::vec(0.1f32..3.0, 3),
        bias_vals in prop::collection::vec(-2.0f32..2.0, 3),
    ) {
        let num_ch = 3;
        let num_out = 2;

        let scale = ArrayD::from_shape_vec(IxDyn(&[num_ch]), scale_vals).unwrap();
        let bias = ArrayD::from_shape_vec(IxDyn(&[num_ch]), bias_vals).unwrap();
        let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

        // Pre-activation: 1D [3] input
        let pre = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[num_ch]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[num_ch]), 1.0_f32),
        ).unwrap();

        // Scalar bounds: [2, 3] matrix
        let scalar_bounds = LinearBounds::new(
            Array2::from_shape_vec((num_out, num_ch), lower_a_vals).unwrap(),
            Array1::from_vec(lower_b_vals),
            Array2::from_shape_vec((num_out, num_ch), upper_a_vals).unwrap(),
            Array1::from_vec(upper_b_vals),
        ).unwrap();
        let expected = layer.propagate_linear_with_bounds(&scalar_bounds, &pre)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;

        // Batched bounds: [2, 3] (same shape, just as ArrayD)
        let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
            scalar_bounds.lower_a.clone().into_dyn(),
            scalar_bounds.lower_b.clone().into_dyn(),
            scalar_bounds.upper_a.clone().into_dyn(),
            scalar_bounds.upper_b.into_dyn(),
            vec![num_ch],
            vec![num_out],
        );

        let actual = layer.propagate_linear_batched_with_bounds(&batched_bounds, &pre)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        let equiv_tol = 2e-5;

        for (idx, (&a, &e)) in actual.lower_a.iter().zip(expected.lower_a.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= equiv_tol,
                "lower_a mismatch at {idx}: batched={a}, scalar={e}"
            );
        }
        for (idx, (&a, &e)) in actual.upper_a.iter().zip(expected.upper_a.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= equiv_tol,
                "upper_a mismatch at {idx}: batched={a}, scalar={e}"
            );
        }
        for (idx, (&a, &e)) in actual.lower_b.iter().zip(expected.lower_b.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= equiv_tol,
                "lower_b mismatch at {idx}: batched={a}, scalar={e}"
            );
        }
        for (idx, (&a, &e)) in actual.upper_b.iter().zip(expected.upper_b.iter()).enumerate() {
            prop_assert!(
                (a - e).abs() <= equiv_tol,
                "upper_b mismatch at {idx}: batched={a}, scalar={e}"
            );
        }
    }

    /// BatchNorm batched CROWN parity WITH incoming certified coeff error.
    ///
    /// Regression moat for #cgan-conv-err-compose: the batched BatchNorm backward
    /// now PROPAGATES incoming coeff error as `e·|scale|` (plus its fresh per-coeff
    /// ULP term) and widens the bias by `e·|bias_i|` / the `(|a|+e)·w_err` margin,
    /// EXACTLY mirroring the scalar `crown_scalar.rs`. Before this change the
    /// batched path emitted only the fresh `|A_new|·u` term and relied on the
    /// dispatcher discharging the incoming err over BN's output box — so the two
    /// paths DIVERGED on any BN with incoming err. This asserts they now agree on
    /// all four bound fields AND on both propagated err matrices.
    ///
    /// The incoming err simulates what an upstream conv backward emits before it
    /// reaches BN (backward order: input → BN → conv; conv runs first and hands BN
    /// its fresh coeff err) — the real cGAN conv→BN stack scenario.
    #[ntest::timeout(10000)]
    #[test]
    fn batchnorm_batched_matches_scalar_with_incoming_err(
        lower_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        upper_a_vals in prop::collection::vec(-5.0f32..5.0, 6),
        lower_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        upper_b_vals in prop::collection::vec(-5.0f32..5.0, 2),
        scale_vals in prop::collection::vec(-3.0f32..3.0, 3),
        bias_vals in prop::collection::vec(-2.0f32..2.0, 3),
        lower_err_vals in prop::collection::vec(0.0f32..2.0, 6),
        upper_err_vals in prop::collection::vec(0.0f32..2.0, 6),
    ) {
        prop_assume!(scale_vals.iter().all(|s| s.abs() > 0.05));

        let num_ch = 3;
        let num_out = 2;

        let scale = ArrayD::from_shape_vec(IxDyn(&[num_ch]), scale_vals).unwrap();
        let bias = ArrayD::from_shape_vec(IxDyn(&[num_ch]), bias_vals).unwrap();
        let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

        let pre = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[num_ch]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[num_ch]), 1.0_f32),
        ).unwrap();

        // Scalar bounds [2, 3] WITH incoming coeff error.
        let mut scalar_bounds = LinearBounds::new(
            Array2::from_shape_vec((num_out, num_ch), lower_a_vals).unwrap(),
            Array1::from_vec(lower_b_vals),
            Array2::from_shape_vec((num_out, num_ch), upper_a_vals).unwrap(),
            Array1::from_vec(upper_b_vals),
        ).unwrap();
        scalar_bounds.set_coeff_err(
            Array2::from_shape_vec((num_out, num_ch), lower_err_vals.clone()).unwrap(),
            Array2::from_shape_vec((num_out, num_ch), upper_err_vals.clone()).unwrap(),
        );

        let expected = layer.propagate_linear_with_bounds(&scalar_bounds, &pre)
            .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;

        // Batched bounds carrying the SAME incoming coeff error.
        let mut batched_bounds = BatchedLinearBounds::from_parts_unchecked(
            scalar_bounds.lower_a.clone().into_dyn(),
            scalar_bounds.lower_b.clone().into_dyn(),
            scalar_bounds.upper_a.clone().into_dyn(),
            scalar_bounds.upper_b.clone().into_dyn(),
            vec![num_ch],
            vec![num_out],
        );
        batched_bounds.set_coeff_err(
            ArrayD::from_shape_vec(IxDyn(&[num_out, num_ch]), lower_err_vals).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[num_out, num_ch]), upper_err_vals).unwrap(),
        );

        let actual = layer.propagate_linear_batched_with_bounds(&batched_bounds, &pre)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;

        let equiv_tol = 2e-5;

        for (idx, (&a, &e)) in actual.lower_a.iter().zip(expected.lower_a.iter()).enumerate() {
            prop_assert!((a - e).abs() <= equiv_tol, "lower_a mismatch at {idx}: batched={a}, scalar={e}");
        }
        for (idx, (&a, &e)) in actual.upper_a.iter().zip(expected.upper_a.iter()).enumerate() {
            prop_assert!((a - e).abs() <= equiv_tol, "upper_a mismatch at {idx}: batched={a}, scalar={e}");
        }
        for (idx, (&a, &e)) in actual.lower_b.iter().zip(expected.lower_b.iter()).enumerate() {
            prop_assert!((a - e).abs() <= equiv_tol, "lower_b mismatch at {idx}: batched={a}, scalar={e}");
        }
        for (idx, (&a, &e)) in actual.upper_b.iter().zip(expected.upper_b.iter()).enumerate() {
            prop_assert!((a - e).abs() <= equiv_tol, "upper_b mismatch at {idx}: batched={a}, scalar={e}");
        }

        // The propagated coeff-error matrices must now MATCH (they diverged before).
        let exp_lerr = expected.lower_a_err().expect("scalar BN emits lower coeff err");
        let exp_uerr = expected.upper_a_err().expect("scalar BN emits upper coeff err");
        let act_lerr = actual.lower_a_err.as_ref().expect("batched BN emits lower coeff err");
        let act_uerr = actual.upper_a_err.as_ref().expect("batched BN emits upper coeff err");
        for (idx, (&a, &e)) in act_lerr.iter().zip(exp_lerr.iter()).enumerate() {
            prop_assert!((a - e).abs() <= equiv_tol, "lower_a_err mismatch at {idx}: batched={a}, scalar={e}");
        }
        for (idx, (&a, &e)) in act_uerr.iter().zip(exp_uerr.iter()).enumerate() {
            prop_assert!((a - e).abs() <= equiv_tol, "upper_a_err mismatch at {idx}: batched={a}, scalar={e}");
        }
    }

    /// BatchNorm batched CROWN parity WITH a leading batch/domain axis.
    ///
    /// Nit-lock for the review of #cgan-conv-err-compose: the earlier parity
    /// tests use a 2D A `[out, in]` with NO leading batch axis, so the batch-axis
    /// aliasing risk (per-channel `[in_dim]` arrays broadcasting on the trailing
    /// axis across independent domains) was only covered indirectly. This stacks
    /// `B` INDEPENDENT domains — each a distinct `[out, in]` matrix with its own
    /// incoming coeff err — into a `[B, out, in]` batched tensor, runs the batched
    /// backward ONCE, then asserts every slice `d` equals the scalar path on that
    /// slice. BatchNorm's affine map is domain-independent, so no domain may leak
    /// into another: slice `d` of the batched output must match scalar(domain d)
    /// on all four bound fields AND both propagated err matrices.
    #[ntest::timeout(10000)]
    #[test]
    fn batchnorm_batched_matches_scalar_leading_batch_axis(
        la in prop::collection::vec(-5.0f32..5.0, 12),
        ua in prop::collection::vec(-5.0f32..5.0, 12),
        lb in prop::collection::vec(-5.0f32..5.0, 4),
        ub in prop::collection::vec(-5.0f32..5.0, 4),
        scale_vals in prop::collection::vec(-3.0f32..3.0, 3),
        bias_vals in prop::collection::vec(-2.0f32..2.0, 3),
        le in prop::collection::vec(0.0f32..2.0, 12),
        ue in prop::collection::vec(0.0f32..2.0, 12),
    ) {
        prop_assume!(scale_vals.iter().all(|s| s.abs() > 0.05));

        let batch = 2usize;
        let num_out = 2usize;
        let num_ch = 3usize;
        let per = num_out * num_ch; // A/err block per domain
        let perb = num_out; // b block per domain

        let scale = ArrayD::from_shape_vec(IxDyn(&[num_ch]), scale_vals).unwrap();
        let bias = ArrayD::from_shape_vec(IxDyn(&[num_ch]), bias_vals).unwrap();
        let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

        let pre = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[num_ch]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[num_ch]), 1.0_f32),
        ).unwrap();

        // Batched bounds: [B, out, in], with per-domain incoming coeff err.
        let mut batched = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[batch, num_out, num_ch]), la.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, num_out]), lb.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, num_out, num_ch]), ua.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, num_out]), ub.clone()).unwrap(),
            vec![num_ch],
            vec![num_out],
        );
        batched.set_coeff_err(
            ArrayD::from_shape_vec(IxDyn(&[batch, num_out, num_ch]), le.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, num_out, num_ch]), ue.clone()).unwrap(),
        );

        let actual = layer.propagate_linear_batched_with_bounds(&batched, &pre)
            .map_err(|e| TestCaseError::fail(format!("batched CROWN failed: {e}")))?;
        let act_lerr = actual.lower_a_err.as_ref().expect("batched BN emits lower coeff err");
        let act_uerr = actual.upper_a_err.as_ref().expect("batched BN emits upper coeff err");

        let equiv_tol = 2e-5;

        for d in 0..batch {
            let a0 = d * per;
            let b0 = d * perb;
            // Scalar bounds for domain d's [out, in] slice, same incoming err.
            let mut scalar = LinearBounds::new(
                Array2::from_shape_vec((num_out, num_ch), la[a0..a0 + per].to_vec()).unwrap(),
                Array1::from_vec(lb[b0..b0 + perb].to_vec()),
                Array2::from_shape_vec((num_out, num_ch), ua[a0..a0 + per].to_vec()).unwrap(),
                Array1::from_vec(ub[b0..b0 + perb].to_vec()),
            ).unwrap();
            scalar.set_coeff_err(
                Array2::from_shape_vec((num_out, num_ch), le[a0..a0 + per].to_vec()).unwrap(),
                Array2::from_shape_vec((num_out, num_ch), ue[a0..a0 + per].to_vec()).unwrap(),
            );
            let expected = layer.propagate_linear_with_bounds(&scalar, &pre)
                .map_err(|e| TestCaseError::fail(format!("scalar CROWN failed: {e}")))?;

            // Batched slice d must equal scalar(domain d) — no cross-domain leak.
            let sl_la = actual.lower_a.index_axis(Axis(0), d);
            let sl_ua = actual.upper_a.index_axis(Axis(0), d);
            let sl_lb = actual.lower_b.index_axis(Axis(0), d);
            let sl_ub = actual.upper_b.index_axis(Axis(0), d);
            let sl_lerr = act_lerr.index_axis(Axis(0), d);
            let sl_uerr = act_uerr.index_axis(Axis(0), d);

            for (idx, (&a, &e)) in sl_la.iter().zip(expected.lower_a.iter()).enumerate() {
                prop_assert!((a - e).abs() <= equiv_tol, "d{d} lower_a[{idx}]: batched={a}, scalar={e}");
            }
            for (idx, (&a, &e)) in sl_ua.iter().zip(expected.upper_a.iter()).enumerate() {
                prop_assert!((a - e).abs() <= equiv_tol, "d{d} upper_a[{idx}]: batched={a}, scalar={e}");
            }
            for (idx, (&a, &e)) in sl_lb.iter().zip(expected.lower_b.iter()).enumerate() {
                prop_assert!((a - e).abs() <= equiv_tol, "d{d} lower_b[{idx}]: batched={a}, scalar={e}");
            }
            for (idx, (&a, &e)) in sl_ub.iter().zip(expected.upper_b.iter()).enumerate() {
                prop_assert!((a - e).abs() <= equiv_tol, "d{d} upper_b[{idx}]: batched={a}, scalar={e}");
            }
            let exp_lerr = expected.lower_a_err().expect("scalar BN emits lower coeff err");
            let exp_uerr = expected.upper_a_err().expect("scalar BN emits upper coeff err");
            for (idx, (&a, &e)) in sl_lerr.iter().zip(exp_lerr.iter()).enumerate() {
                prop_assert!((a - e).abs() <= equiv_tol, "d{d} lower_a_err[{idx}]: batched={a}, scalar={e}");
            }
            for (idx, (&a, &e)) in sl_uerr.iter().zip(exp_uerr.iter()).enumerate() {
                prop_assert!((a - e).abs() <= equiv_tol, "d{d} upper_a_err[{idx}]: batched={a}, scalar={e}");
            }
        }
    }
}
