// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for AdaIN1d bound propagation.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;

use super::types::AdaIN1dLayer;
use crate::layers::common::BoundPropagation;
use crate::layers::normalization::InstanceNorm1dLayer;

fn make_adain(num_channels: usize, style_gamma: &[f32], style_beta: &[f32]) -> AdaIN1dLayer {
    let inn = InstanceNorm1dLayer::new_default(num_channels, 1e-5).unwrap();
    AdaIN1dLayer::new(
        inn,
        Array1::from_vec(style_gamma.to_vec()),
        Array1::from_vec(style_beta.to_vec()),
    )
    .unwrap()
}

fn make_bounded_2d(lower: &[&[f32]], upper: &[&[f32]]) -> BoundedTensor {
    let c = lower.len();
    let t = lower[0].len();
    let lower_flat: Vec<f32> = lower.iter().flat_map(|r| r.iter().copied()).collect();
    let upper_flat: Vec<f32> = upper.iter().flat_map(|r| r.iter().copied()).collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[c, t]), lower_flat).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[c, t]), upper_flat).unwrap(),
    )
    .unwrap()
}

fn make_non_contiguous_bounded_2d(lower: &[&[f32]], upper: &[&[f32]]) -> BoundedTensor {
    let c = lower.len();
    let t = lower[0].len();
    let lower_flat: Vec<f32> = (0..t)
        .flat_map(|ti| (0..c).map(move |ch| lower[ch][ti]))
        .collect();
    let upper_flat: Vec<f32> = (0..t)
        .flat_map(|ti| (0..c).map(move |ch| upper[ch][ti]))
        .collect();

    let lower_arr = ArrayD::from_shape_vec(IxDyn(&[t, c]), lower_flat)
        .unwrap()
        .view()
        .reversed_axes()
        .to_owned();
    let upper_arr = ArrayD::from_shape_vec(IxDyn(&[t, c]), upper_flat)
        .unwrap()
        .view()
        .reversed_axes()
        .to_owned();

    assert_eq!(lower_arr.shape(), &[c, t]);
    assert_eq!(upper_arr.shape(), &[c, t]);

    BoundedTensor::new(lower_arr, upper_arr).unwrap()
}

// ---------- Construction tests ----------

#[test]
fn test_new_valid() {
    let inn = InstanceNorm1dLayer::new_default(3, 1e-5).unwrap();
    let result = AdaIN1dLayer::new(
        inn,
        Array1::from_vec(vec![1.0, 2.0, 3.0]),
        Array1::from_vec(vec![0.0, 0.5, 1.0]),
    );
    assert!(
        result.is_ok(),
        "AdaIN with matching channels should succeed"
    );
    assert_eq!(result.unwrap().num_channels(), 3);
}

#[test]
fn test_new_mismatched_style_ny_channels() {
    let inn = InstanceNorm1dLayer::new_default(3, 1e-5).unwrap();
    let result = AdaIN1dLayer::new(
        inn,
        Array1::from_vec(vec![1.0, 2.0]), // 2 != 3
        Array1::from_vec(vec![0.0, 0.5, 1.0]),
    );
    assert!(
        result.is_err(),
        "should reject ny channel mismatch (2 vs 3)"
    );
}

#[test]
fn test_new_mismatched_style_beta_channels() {
    let inn = InstanceNorm1dLayer::new_default(3, 1e-5).unwrap();
    let result = AdaIN1dLayer::new(
        inn,
        Array1::from_vec(vec![1.0, 2.0, 3.0]),
        Array1::from_vec(vec![0.0, 0.5]), // 2 != 3
    );
    assert!(
        result.is_err(),
        "should reject beta channel mismatch (2 vs 3)"
    );
}

#[test]
fn test_new_non_finite_style_gamma() {
    let inn = InstanceNorm1dLayer::new_default(2, 1e-5).unwrap();
    let result = AdaIN1dLayer::new(
        inn,
        Array1::from_vec(vec![f32::NAN, 1.0]),
        Array1::from_vec(vec![0.0, 0.0]),
    );
    assert!(result.is_err(), "should reject NaN in style ny");
}

#[test]
fn test_new_non_finite_style_beta() {
    let inn = InstanceNorm1dLayer::new_default(2, 1e-5).unwrap();
    let result = AdaIN1dLayer::new(
        inn,
        Array1::from_vec(vec![1.0, 1.0]),
        Array1::from_vec(vec![f32::INFINITY, 0.0]),
    );
    assert!(result.is_err(), "should reject infinity in style beta");
}

#[test]
fn test_effective_instance_norm_rejects_non_finite_effective_ny_3912() {
    let instance_norm = InstanceNorm1dLayer::new(
        Array1::from_vec(vec![f32::MAX, 1.0]),
        Array1::from_vec(vec![0.0, 0.0]),
        1e-5,
    )
    .unwrap();
    let adain = AdaIN1dLayer::new(
        instance_norm,
        Array1::from_vec(vec![2.0, 1.0]),
        Array1::from_vec(vec![0.0, 0.0]),
    )
    .unwrap();

    let err = adain
        .effective_instance_norm()
        .expect_err("effective InstanceNorm should reject infinite effective ny");
    assert!(
        matches!(err, NyError::InvalidSpec(ref message) if message.contains("effective InstanceNorm ny")),
        "expected InvalidSpec for non-finite effective ny, got: {err}"
    );
}

#[test]
fn test_effective_instance_norm_rejects_non_finite_effective_beta_3912() {
    let instance_norm = InstanceNorm1dLayer::new(
        Array1::from_vec(vec![1.0, 1.0]),
        Array1::from_vec(vec![f32::MAX, 0.0]),
        1e-5,
    )
    .unwrap();
    let adain = AdaIN1dLayer::new(
        instance_norm,
        Array1::from_vec(vec![2.0, 1.0]),
        Array1::from_vec(vec![0.0, 0.0]),
    )
    .unwrap();

    let err = adain
        .effective_instance_norm()
        .expect_err("effective InstanceNorm should reject infinite effective beta");
    assert!(
        matches!(err, NyError::InvalidSpec(ref message) if message.contains("effective InstanceNorm beta")),
        "expected InvalidSpec for non-finite effective beta, got: {err}"
    );
}

// ---------- IBP soundness tests ----------

#[test]
fn test_ibp_identity_style_matches_instance_norm() {
    let inn = InstanceNorm1dLayer::new_default(2, 1e-5).unwrap();
    let adain = AdaIN1dLayer::new_identity_style(inn.clone()).unwrap();

    let input = make_bounded_2d(
        &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]],
        &[&[1.5, 2.5, 3.5], &[4.5, 5.5, 6.5]],
    );

    let inn_result = inn.propagate_ibp(&input).unwrap();
    let adain_result = adain.propagate_ibp(&input).unwrap();

    for (a, b) in inn_result.lower().iter().zip(adain_result.lower().iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "IBP lower: InstanceNorm {a} vs AdaIN {b}"
        );
    }
    for (a, b) in inn_result.upper().iter().zip(adain_result.upper().iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "IBP upper: InstanceNorm {a} vs AdaIN {b}"
        );
    }
}

#[test]
fn test_ibp_soundness_contains_eval_samples() {
    // AdaIN with non-trivial style
    let layer = make_adain(2, &[2.0, 0.5], &[10.0, -5.0]);

    let input = make_bounded_2d(
        &[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]],
        &[&[2.0, 3.0, 4.0, 5.0], &[6.0, 7.0, 8.0, 9.0]],
    );

    let bounds = layer.propagate_ibp(&input).unwrap();

    // Sample 200 random points in the input interval and verify they're within bounds
    let c = 2_usize;
    let t = 4_usize;
    for seed in 0..200_u32 {
        let mut sample_vals = Vec::new();
        for ch in 0..c {
            for ti in 0..t {
                let lo = input.lower()[IxDyn(&[ch, ti])];
                let hi = input.upper()[IxDyn(&[ch, ti])];
                let frac = ((seed.wrapping_mul(2654435761) ^ (ch * t + ti) as u32) as f32)
                    / u32::MAX as f32;
                let val = lo + frac * (hi - lo);
                sample_vals.push(val);
            }
        }

        let sample_input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[c, t]), sample_vals.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[c, t]), sample_vals).unwrap(),
        )
        .unwrap();

        // Eval at the sample point (IBP on a point interval gives the exact eval)
        let sample_output = layer.propagate_ibp(&sample_input).unwrap();

        for ch in 0..c {
            for ti in 0..t {
                let idx = IxDyn(&[ch, ti]);
                let val = sample_output.lower()[&idx]; // point ≈ lower ≈ upper
                let bound_lo = bounds.lower()[&idx];
                let bound_hi = bounds.upper()[&idx];

                assert!(
                    val >= bound_lo - 1e-4 && val <= bound_hi + 1e-4,
                    "IBP soundness violation at [{ch},{ti}]: val={val}, bounds=[{bound_lo}, {bound_hi}]"
                );
            }
        }
    }
}

#[test]
fn test_ibp_negative_style_gamma() {
    // Negative style_gamma should flip bounds correctly
    let layer = make_adain(1, &[-2.0], &[0.0]);

    let input = make_bounded_2d(&[&[1.0, 2.0, 3.0]], &[&[2.0, 3.0, 4.0]]);

    let bounds = layer.propagate_ibp(&input).unwrap();

    // With negative ny, lower should be less than upper
    for i in 0..3 {
        let idx = IxDyn(&[0, i]);
        assert!(
            bounds.lower()[&idx] <= bounds.upper()[&idx],
            "Bounds inverted at position {i}: lower={} > upper={}",
            bounds.lower()[&idx],
            bounds.upper()[&idx]
        );
    }
}

#[test]
fn test_ibp_zero_style_gamma() {
    // Zero style_gamma: output should be constant = style_beta
    let layer = make_adain(1, &[0.0], &[7.0]);

    let input = make_bounded_2d(&[&[1.0, 2.0, 3.0]], &[&[5.0, 6.0, 7.0]]);

    let bounds = layer.propagate_ibp(&input).unwrap();

    for i in 0..3 {
        let idx = IxDyn(&[0, i]);
        // With zero ny, output = 0 * instnorm + 7 = 7
        assert!(
            (bounds.lower()[&idx] - 7.0).abs() < 1e-2 || bounds.lower()[&idx] <= 7.0,
            "With zero style_gamma, lower bound should be <= 7: got {}",
            bounds.lower()[&idx]
        );
        assert!(
            (bounds.upper()[&idx] - 7.0).abs() < 1e-2 || bounds.upper()[&idx] >= 7.0,
            "With zero style_gamma, upper bound should be >= 7: got {}",
            bounds.upper()[&idx]
        );
    }
}

#[test]
fn test_ibp_ternary_accepts_non_contiguous_style_bounds_4250() {
    let inn = InstanceNorm1dLayer::new_default(2, 1e-5).unwrap();
    let layer = AdaIN1dLayer::variable_style(inn).unwrap();

    let input = make_bounded_2d(
        &[&[-0.4, 0.1, 0.6], &[0.3, -0.2, 0.5]],
        &[&[0.2, 0.8, 1.1], &[0.9, 0.4, 1.2]],
    );
    let style_gamma = make_bounded_2d(
        &[&[0.8, 0.9, 1.0], &[1.1, 1.2, 1.3]],
        &[&[1.0, 1.1, 1.2], &[1.3, 1.4, 1.5]],
    );
    let style_beta = make_bounded_2d(
        &[&[-0.3, -0.2, -0.1], &[0.0, 0.1, 0.2]],
        &[&[-0.1, 0.0, 0.1], &[0.2, 0.3, 0.4]],
    );
    let style_ny_non_contig = make_non_contiguous_bounded_2d(
        &[&[0.8, 0.9, 1.0], &[1.1, 1.2, 1.3]],
        &[&[1.0, 1.1, 1.2], &[1.3, 1.4, 1.5]],
    );
    let style_beta_non_contig = make_non_contiguous_bounded_2d(
        &[&[-0.3, -0.2, -0.1], &[0.0, 0.1, 0.2]],
        &[&[-0.1, 0.0, 0.1], &[0.2, 0.3, 0.4]],
    );

    assert!(
        style_ny_non_contig.lower().as_slice().is_none(),
        "precondition: style ny lower should be non-contiguous"
    );
    assert!(
        style_beta_non_contig.lower().as_slice().is_none(),
        "precondition: style beta lower should be non-contiguous"
    );

    let expected = layer
        .propagate_ibp_ternary(&input, &style_gamma, &style_beta)
        .expect("contiguous variable-style AdaIN should succeed");
    let actual = layer
        .propagate_ibp_ternary(&input, &style_ny_non_contig, &style_beta_non_contig)
        .expect("non-contiguous style bounds should not degrade AdaIN ternary IBP");

    for (lhs, rhs) in actual.lower().iter().zip(expected.lower().iter()) {
        assert!(
            (lhs - rhs).abs() < 1e-6,
            "lower mismatch for non-contiguous style bounds: {lhs} vs {rhs}"
        );
    }
    for (lhs, rhs) in actual.upper().iter().zip(expected.upper().iter()) {
        assert!(
            (lhs - rhs).abs() < 1e-6,
            "upper mismatch for non-contiguous style bounds: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn test_ibp_rejects_non_finite_input() {
    // BoundedTensor::new itself validates finiteness, so NaN inputs
    // are rejected before they reach the layer.
    let result = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![f32::NAN, 1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
    );
    assert!(
        result.is_err(),
        "BoundedTensor should reject NaN lower bounds"
    );
}

#[test]
fn test_ibp_requires_2d_input() {
    let layer = make_adain(1, &[1.0], &[0.0]);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).unwrap(),
    )
    .unwrap();

    assert!(
        layer.propagate_ibp(&input).is_err(),
        "IBP should reject 1D input for AdaIN"
    );
}

// ---------- CROWN backward soundness tests ----------

/// CROWN backward soundness test with identity incoming bounds (Sampling mode).
///
/// Verifies that concretized CROWN output bounds contain the true
/// AdaIN1d evaluation at many sampled points within the input interval.
#[test]
fn test_crown_backward_soundness_identity_sampling() {
    use crate::layers::normalization::LayerNormCrownMode;
    use crate::LinearBounds;
    use ndarray::arr1;

    let layer =
        make_adain(2, &[2.0, 0.5], &[10.0, -5.0]).with_crown_mode(LayerNormCrownMode::Sampling);

    let num_channels = 2;
    let time_len = 3;
    let total = num_channels * time_len;

    let lower_vals = vec![-1.0_f32, -2.0, 0.0, 1.0, -1.0, 0.5];
    let upper_vals = vec![1.0_f32, 0.0, 2.0, 3.0, 1.0, 2.5];

    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), lower_vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), upper_vals.clone()).unwrap(),
    )
    .unwrap();

    let bounds = LinearBounds::identity(total);
    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("Sampling CROWN should succeed");

    // Concretize against flat input
    let input_flat = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower_vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper_vals.clone()).unwrap(),
    )
    .unwrap();
    let concrete = result.concretize(&input_flat);

    // Sample 200 points and verify soundness
    for s in 0..200_u32 {
        let mut sample = Vec::with_capacity(total);
        for i in 0..total {
            let t = ((s.wrapping_mul(2654435761) ^ (i as u32)).wrapping_mul(2654435761)) as f32
                / u32::MAX as f32;
            sample.push(lower_vals[i] + (upper_vals[i] - lower_vals[i]) * t);
        }

        let mut y_flat: Vec<f32> = Vec::with_capacity(total);
        for c in 0..num_channels {
            let start = c * time_len;
            let channel_input = arr1(&sample[start..start + time_len]);
            let y_channel = layer
                .eval_channel(&channel_input, c)
                .expect("eval should succeed");
            y_flat.extend(y_channel.iter());
        }

        for (i, &y_val) in y_flat.iter().enumerate().take(total) {
            assert!(
                concrete.lower()[[i]] <= y_val + 1e-3,
                "CROWN lower violated: dim {i}, sample {s}: lower {} > eval {}",
                concrete.lower()[[i]],
                y_val
            );
            assert!(
                concrete.upper()[[i]] >= y_val - 1e-3,
                "CROWN upper violated: dim {i}, sample {s}: upper {} < eval {}",
                concrete.upper()[[i]],
                y_flat[i]
            );
        }
    }
}

/// Helper: verify concretized bounds contain all grid-sampled composed outputs.
fn verify_composed_bounds(
    concrete: &BoundedTensor,
    layer: &AdaIN1dLayer,
    lower_vals: &[f32],
    upper_vals: &[f32],
    a_rows: &[[f32; 3]],
    tol: f32,
) {
    use ndarray::arr1;
    let steps = 8;
    for i0 in 0..=steps {
        for i1 in 0..=steps {
            for i2 in 0..=steps {
                let frac = |i: usize, idx: usize| {
                    lower_vals[idx] + (upper_vals[idx] - lower_vals[idx]) * i as f32 / steps as f32
                };
                let x = arr1(&[frac(i0, 0), frac(i1, 1), frac(i2, 2)]);
                let y = layer.eval_channel(&x, 0).unwrap();
                for (k, row) in a_rows.iter().enumerate() {
                    let composed = row[0] * y[0] + row[1] * y[1] + row[2] * y[2];
                    assert!(concrete.lower()[[k]] <= composed + tol, "lower dim {k}");
                    assert!(concrete.upper()[[k]] >= composed - tol, "upper dim {k}");
                }
            }
        }
    }
}

/// CROWN backward soundness with non-identity A-matrix for AdaIN1d.
#[test]
fn test_crown_backward_soundness_non_identity_coeff() {
    use crate::layers::normalization::LayerNormCrownMode;
    use crate::LinearBounds;
    use ndarray::Array2;

    let layer = make_adain(1, &[1.5], &[-1.0]).with_crown_mode(LayerNormCrownMode::Sampling);
    let lower_vals = vec![0.0_f32, 1.0, 2.0];
    let upper_vals = vec![0.5_f32, 1.5, 2.5];

    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), lower_vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), upper_vals.clone()).unwrap(),
    )
    .unwrap();

    // A = [1 -1 0; 0 1 -1; 0.5 0 0.5]
    let a = Array2::from_shape_vec(
        (3, 3),
        vec![1.0_f32, -1.0, 0.0, 0.0, 1.0, -1.0, 0.5, 0.0, 0.5],
    )
    .unwrap();
    let bounds = LinearBounds::new(a.clone(), Array1::zeros(3), a, Array1::zeros(3)).unwrap();

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .unwrap();
    let input_flat = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), lower_vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), upper_vals.clone()).unwrap(),
    )
    .unwrap();
    let concrete = result.concretize(&input_flat);

    verify_composed_bounds(
        &concrete,
        &layer,
        &lower_vals,
        &upper_vals,
        &[[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [0.5, 0.0, 0.5]],
        1e-2,
    );
}

/// AdaIN CROWN backward: identity style should match InstanceNorm CROWN.
///
/// When style_gamma = 1 and style_beta = 0, AdaIN is exactly InstanceNorm.
/// The CROWN linearization should produce identical results.
#[test]
fn test_crown_backward_identity_style_matches_instance_norm() {
    use crate::layers::normalization::LayerNormCrownMode;
    use crate::LinearBounds;

    let inn = InstanceNorm1dLayer::new_default(2, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);
    let adain =
        AdaIN1dLayer::new_identity_style(InstanceNorm1dLayer::new_default(2, 1e-5).unwrap())
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::Sampling);

    let total = 6; // C=2, T=3

    let lower_vals = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let upper_vals = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), lower_vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), upper_vals.clone()).unwrap(),
    )
    .unwrap();

    let bounds = LinearBounds::identity(total);

    let inn_result = inn
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("InstanceNorm CROWN should succeed");
    let adain_result = adain
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("AdaIN CROWN should succeed");

    // Concretize both
    let input_flat = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower_vals).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper_vals).unwrap(),
    )
    .unwrap();
    let inn_concrete = inn_result.concretize(&input_flat);
    let adain_concrete = adain_result.concretize(&input_flat);

    for i in 0..total {
        assert!(
            (inn_concrete.lower()[[i]] - adain_concrete.lower()[[i]]).abs() < 1e-4,
            "Identity style CROWN lower mismatch at dim {i}: inn {} vs adain {}",
            inn_concrete.lower()[[i]],
            adain_concrete.lower()[[i]]
        );
        assert!(
            (inn_concrete.upper()[[i]] - adain_concrete.upper()[[i]]).abs() < 1e-4,
            "Identity style CROWN upper mismatch at dim {i}: inn {} vs adain {}",
            inn_concrete.upper()[[i]],
            adain_concrete.upper()[[i]]
        );
    }
}

// ---------- Trait wiring tests ----------

#[test]
fn test_requires_pre_activation_bounds() {
    let layer = make_adain(1, &[1.0], &[0.0]);
    assert!(
        layer.requires_pre_activation_bounds(),
        "AdaIN requires pre-activation bounds"
    );
}

#[test]
fn test_propagate_linear_returns_error() {
    let layer = make_adain(1, &[1.0], &[0.0]);
    let dummy = crate::LinearBounds::identity(3);
    assert!(
        layer.propagate_linear(&dummy).is_err(),
        "propagate_linear should be unsupported for AdaIN"
    );
}

// ── NormLayer trait input validation tests (#3339) ───────────────────

/// Verify that eval() rejects input whose length is not divisible by num_channels.
/// Part of #3339.
#[test]
fn test_normlayer_eval_rejects_indivisible_input() {
    use super::super::trait_norm::NormLayer;
    let layer = make_adain(3, &[1.0, 2.0, 3.0], &[0.0, 0.5, 1.0]);
    let x = Array1::zeros(7); // 7 not divisible by 3
    let result = NormLayer::eval(&layer, &x);
    assert!(
        result.is_err(),
        "AdaIN eval should reject input of len 7 for 3 channels"
    );
}

/// Verify that jacobian() rejects input whose length is not divisible by num_channels.
/// Part of #3339.
#[test]
fn test_normlayer_jacobian_rejects_indivisible_input() {
    use super::super::trait_norm::NormLayer;
    let layer = make_adain(3, &[1.0, 2.0, 3.0], &[0.0, 0.5, 1.0]);
    let x = Array1::zeros(7);
    let result = NormLayer::jacobian(&layer, &x);
    assert!(
        result.is_err(),
        "AdaIN jacobian should reject input of len 7 for 3 channels"
    );
}

// CROWN NaN/Inf guard tests moved to tests_crown.rs (#3103) to keep file under 500-line limit.
