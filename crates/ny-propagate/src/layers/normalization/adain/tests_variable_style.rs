// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for variable-style AdaIN1d (ternary layer).
//!
//! Part of #4142. Covers:
//! - Variable-style constructor and arity predicates
//! - Fixed-style accessor rejection on variable-style
//! - Ternary IBP soundness (4-corner product hull)
//! - Ternary CROWN backward soundness
//! - Gate regression: variable-style excluded from batched CROWN

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;

use super::types::{AdaIN1dLayer, AdaINStyleMode};
use crate::layers::common::BoundPropagation;
use crate::layers::normalization::InstanceNorm1dLayer;
use crate::layers::Layer;

fn make_variable_adain(num_channels: usize) -> AdaIN1dLayer {
    let inn = InstanceNorm1dLayer::new_default(num_channels, 1e-5).unwrap();
    AdaIN1dLayer::variable_style(inn).unwrap()
}

fn make_fixed_adain(num_channels: usize, style_gamma: &[f32], style_beta: &[f32]) -> AdaIN1dLayer {
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

fn make_bounded_flat(lo: &[f32], hi: &[f32]) -> BoundedTensor {
    let n = lo.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, n]), lo.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, n]), hi.to_vec()).unwrap(),
    )
    .unwrap()
}

fn sample_hash(seed: u32, idx: usize, offset: u32) -> f32 {
    let h = seed
        .wrapping_mul(2654435761)
        .wrapping_add(offset)
        .wrapping_mul(2246822519)
        .wrapping_add(idx as u32);
    (h as f32) / u32::MAX as f32
}

fn sample_in_box(lo: &[f32], hi: &[f32], seed: u32, offset: u32) -> Vec<f32> {
    (0..lo.len())
        .map(|j| lo[j] + sample_hash(seed, j, offset) * (hi[j] - lo[j]))
        .collect()
}

fn eval_adain_true(layer: &AdaIN1dLayer, x: &[f32], g: &[f32], b: &[f32]) -> Vec<f32> {
    let x_point = make_bounded_flat(x, x);
    let z = layer.instance_norm.propagate_ibp(&x_point).unwrap();
    let n = x.len();
    (0..n)
        .map(|j| g[j] * z.lower()[IxDyn(&[0, j])] + b[j])
        .collect()
}

/// Concretize one row of ternary CROWN bounds over the joint (x, g, b) box.
fn concretize_ternary_row(
    row: usize,
    is_lower: bool,
    lbs: [&crate::LinearBounds; 3],
    bounds: &[(&[f32], &[f32]); 3],
    bias: f32,
) -> f32 {
    let n = bounds[0].0.len();
    let mut val = bias as f64;
    for (k, lb) in lbs.iter().enumerate() {
        let a_mat = if is_lower { lb.lower_a() } else { lb.upper_a() };
        let (lo, hi) = bounds[k];
        for j in 0..n {
            let a_val = a_mat[[row, j]] as f64;
            if is_lower {
                val += if a_val >= 0.0 {
                    a_val * lo[j] as f64
                } else {
                    a_val * hi[j] as f64
                };
            } else {
                val += if a_val >= 0.0 {
                    a_val * hi[j] as f64
                } else {
                    a_val * lo[j] as f64
                };
            }
        }
    }
    val as f32
}

// ---------- Construction & arity tests ----------

#[test]
fn test_variable_style_constructor() {
    let layer = make_variable_adain(3);
    assert_eq!(layer.num_channels(), 3);
    assert!(
        layer.requires_style_inputs(),
        "variable-style AdaIN should require style inputs"
    );
    assert!(matches!(layer.style_mode, AdaINStyleMode::Variable));
}

#[test]
fn test_fixed_style_does_not_require_style_inputs() {
    let layer = make_fixed_adain(2, &[1.0, 2.0], &[0.0, 0.5]);
    assert!(
        !layer.requires_style_inputs(),
        "fixed-style AdaIN should not require style inputs"
    );
    assert!(matches!(layer.style_mode, AdaINStyleMode::Fixed(_)));
}

#[test]
fn test_variable_style_rejects_style_ny_accessor() {
    let layer = make_variable_adain(2);
    let err = layer
        .style_gamma()
        .expect_err("variable-style has no embedded style_gamma");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

#[test]
fn test_variable_style_rejects_style_beta_accessor() {
    let layer = make_variable_adain(2);
    let err = layer
        .style_beta()
        .expect_err("variable-style has no embedded style_beta");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

#[test]
fn test_variable_style_effective_instance_norm_errors() {
    let layer = make_variable_adain(2);
    let err = layer
        .effective_instance_norm()
        .expect_err("variable-style cannot collapse to effective InstanceNorm");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

// ---------- Layer enum arity integration tests ----------

#[test]
fn test_layer_enum_is_ternary_variable_style() {
    let adain = make_variable_adain(2);
    let layer = Layer::AdaIN1d(adain);
    assert!(layer.is_ternary(), "variable-style AdaIN should be ternary");
    assert!(
        !layer.is_binary(),
        "variable-style AdaIN should not be binary"
    );
}

#[test]
fn test_layer_enum_is_not_ternary_fixed_style() {
    let adain = make_fixed_adain(2, &[1.0, 2.0], &[0.0, 0.5]);
    let layer = Layer::AdaIN1d(adain);
    assert!(
        !layer.is_ternary(),
        "fixed-style AdaIN should not be ternary"
    );
    assert!(!layer.is_binary(), "fixed-style AdaIN should not be binary");
}

#[test]
fn test_layer_enum_min_inputs_variable_style() {
    let adain = make_variable_adain(2);
    let layer = Layer::AdaIN1d(adain);
    assert_eq!(layer.min_inputs(), 3, "variable-style AdaIN needs 3 inputs");
}

#[test]
fn test_layer_enum_min_inputs_fixed_style() {
    let adain = make_fixed_adain(2, &[1.0, 2.0], &[0.0, 0.5]);
    let layer = Layer::AdaIN1d(adain);
    assert_eq!(layer.min_inputs(), 1, "fixed-style AdaIN needs 1 input");
}

// ---------- Gate regression: supports_batched_crown ----------

#[test]
fn test_supports_batched_crown_fixed_style_accepted() {
    let adain = make_fixed_adain(2, &[1.0, 2.0], &[0.0, 0.5]);
    let layer = Layer::AdaIN1d(adain);
    assert!(
        layer.supports_batched_crown(),
        "fixed-style AdaIN should support batched CROWN"
    );
}

#[test]
fn test_supports_batched_crown_variable_style_rejected() {
    let adain = make_variable_adain(2);
    let layer = Layer::AdaIN1d(adain);
    assert!(
        !layer.supports_batched_crown(),
        "variable-style AdaIN must NOT support batched CROWN"
    );
}

#[test]
fn test_supports_batched_crown_with_conv2d_excludes_all_adain() {
    // Both fixed and variable should be excluded from conv2d batched CROWN.
    let fixed = Layer::AdaIN1d(make_fixed_adain(2, &[1.0, 2.0], &[0.0, 0.5]));
    let variable = Layer::AdaIN1d(make_variable_adain(2));
    assert!(
        !fixed.supports_batched_crown_with_conv2d(),
        "fixed-style AdaIN excluded from conv2d batched CROWN"
    );
    assert!(
        !variable.supports_batched_crown_with_conv2d(),
        "variable-style AdaIN excluded from conv2d batched CROWN"
    );
}

// ---------- Ternary IBP soundness ----------

#[test]
fn test_ibp_ternary_rejects_fixed_style() {
    let layer = make_fixed_adain(2, &[1.0, 2.0], &[0.0, 0.5]);
    let x = make_bounded_2d(&[&[1.0, 2.0], &[3.0, 4.0]], &[&[2.0, 3.0], &[4.0, 5.0]]);
    let g = x.clone();
    let b = x.clone();
    let err = layer
        .propagate_ibp_ternary(&x, &g, &b)
        .expect_err("fixed-style should reject ternary IBP");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

#[test]
fn test_ibp_ternary_shape_mismatch_gamma() {
    let layer = make_variable_adain(2);
    let x = make_bounded_2d(&[&[1.0, 2.0], &[3.0, 4.0]], &[&[2.0, 3.0], &[4.0, 5.0]]);
    let g_wrong = make_bounded_2d(&[&[1.0]], &[&[2.0]]); // wrong shape
    let b = x.clone();
    assert!(layer.propagate_ibp_ternary(&x, &g_wrong, &b).is_err());
}

#[test]
fn test_ibp_ternary_shape_mismatch_beta() {
    let layer = make_variable_adain(2);
    let x = make_bounded_2d(&[&[1.0, 2.0], &[3.0, 4.0]], &[&[2.0, 3.0], &[4.0, 5.0]]);
    let g = x.clone();
    let b_wrong = make_bounded_2d(&[&[1.0]], &[&[2.0]]); // wrong shape
    assert!(layer.propagate_ibp_ternary(&x, &g, &b_wrong).is_err());
}

/// Ternary IBP soundness: sample random (x, g, b) points in the joint box and
/// verify the true output is within IBP bounds.
///
/// Uses C=2, T=2 with non-trivial style intervals per design spec.
#[test]
fn test_ibp_ternary_soundness_sampling() {
    let layer = make_variable_adain(2);

    let c = 2_usize;
    let t = 2_usize;

    // Input x bounds.
    let x_bounds = make_bounded_2d(&[&[1.0, 3.0], &[5.0, 7.0]], &[&[2.0, 4.0], &[6.0, 8.0]]);

    // Style ny bounds (non-trivial interval, includes negatives for ch 1).
    let g_bounds = make_bounded_2d(&[&[0.5, 0.5], &[-1.0, -1.0]], &[&[2.0, 2.0], &[0.5, 0.5]]);

    // Style beta bounds.
    let b_bounds = make_bounded_2d(&[&[-1.0, -1.0], &[0.0, 0.0]], &[&[1.0, 1.0], &[2.0, 2.0]]);

    let ibp_result = layer
        .propagate_ibp_ternary(&x_bounds, &g_bounds, &b_bounds)
        .expect("ternary IBP should succeed");

    // Sample 300 random (x, g, b) points and verify each is within bounds.
    for seed in 0..300_u32 {
        let mut x_vals = Vec::new();
        let mut g_vals = Vec::new();
        let mut b_vals = Vec::new();

        for ch in 0..c {
            for ti in 0..t {
                let hash = |offset: u32| -> f32 {
                    let h = seed
                        .wrapping_mul(2654435761)
                        .wrapping_add(offset)
                        .wrapping_mul(2246822519)
                        .wrapping_add((ch * t + ti) as u32);
                    (h as f32) / u32::MAX as f32
                };

                let xlo = x_bounds.lower()[IxDyn(&[ch, ti])];
                let xhi = x_bounds.upper()[IxDyn(&[ch, ti])];
                x_vals.push(xlo + hash(0) * (xhi - xlo));

                let glo = g_bounds.lower()[IxDyn(&[ch, ti])];
                let ghi = g_bounds.upper()[IxDyn(&[ch, ti])];
                g_vals.push(glo + hash(1) * (ghi - glo));

                let blo = b_bounds.lower()[IxDyn(&[ch, ti])];
                let bhi = b_bounds.upper()[IxDyn(&[ch, ti])];
                b_vals.push(blo + hash(2) * (bhi - blo));
            }
        }

        // Compute true output: y = g * InstanceNorm(x) + b.
        // Use IBP on point intervals for InstanceNorm eval.
        let x_point = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[c, t]), x_vals.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[c, t]), x_vals).unwrap(),
        )
        .unwrap();
        let z_point = layer.instance_norm.propagate_ibp(&x_point).unwrap();

        // Element-wise: y_i = g_i * z_i + b_i.
        for ch in 0..c {
            for ti in 0..t {
                let flat_idx = ch * t + ti;
                let z_val = z_point.lower()[IxDyn(&[ch, ti])];
                let y_val = g_vals[flat_idx] * z_val + b_vals[flat_idx];

                let bound_lo = ibp_result.lower()[IxDyn(&[ch, ti])];
                let bound_hi = ibp_result.upper()[IxDyn(&[ch, ti])];

                assert!(
                    y_val >= bound_lo - 1e-3 && y_val <= bound_hi + 1e-3,
                    "IBP ternary soundness violation: seed={seed}, [{ch},{ti}]: \
                     y={y_val}, bounds=[{bound_lo}, {bound_hi}]"
                );
            }
        }
    }
}

/// Ternary IBP with identity-equivalent style intervals [1,1] and [0,0]
/// should match fixed-style identity IBP.
#[test]
fn test_ibp_ternary_identity_style_matches_fixed() {
    let variable = make_variable_adain(2);
    let fixed =
        AdaIN1dLayer::new_identity_style(InstanceNorm1dLayer::new_default(2, 1e-5).unwrap())
            .unwrap();

    let x = make_bounded_2d(
        &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]],
        &[&[1.5, 2.5, 3.5], &[4.5, 5.5, 6.5]],
    );

    // Point-interval style = identity.
    let g_identity = make_bounded_2d(
        &[&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]],
        &[&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]],
    );
    let b_zero = make_bounded_2d(
        &[&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]],
        &[&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]],
    );

    let fixed_result = fixed.propagate_ibp(&x).unwrap();
    let variable_result = variable
        .propagate_ibp_ternary(&x, &g_identity, &b_zero)
        .unwrap();

    for i in 0..fixed_result.lower().len() {
        let f_lo = fixed_result.lower().as_slice().unwrap()[i];
        let v_lo = variable_result.lower().as_slice().unwrap()[i];
        let f_hi = fixed_result.upper().as_slice().unwrap()[i];
        let v_hi = variable_result.upper().as_slice().unwrap()[i];

        // Variable-style uses 4-corner hull which may be slightly wider
        // than the fixed-style sign-analysis path, but should still contain
        // the same points. Allow variable to be wider.
        assert!(
            v_lo <= f_lo + 1e-4,
            "Variable lower should be <= fixed lower: v={v_lo}, f={f_lo}"
        );
        assert!(
            v_hi >= f_hi - 1e-4,
            "Variable upper should be >= fixed upper: v={v_hi}, f={f_hi}"
        );
    }
}

// ---------- Ternary CROWN backward soundness ----------

/// Ternary CROWN soundness with identity A-matrix (C=1, T=3).
#[test]
fn test_crown_ternary_soundness_sampling() {
    use crate::LinearBounds;

    let layer = make_variable_adain(1);
    let n = 3;

    let x_lo = [0.0_f32, 1.0, 2.0];
    let x_hi = [0.5_f32, 1.5, 2.5];
    let g_lo = [0.5_f32, 0.5, 0.5];
    let g_hi = [2.0_f32, 2.0, 2.0];
    let b_lo = [-0.5_f32, -0.5, -0.5];
    let b_hi = [0.5_f32, 0.5, 0.5];

    let node_lb = LinearBounds::identity(n);
    let (per_input, bias_lower, bias_upper) = layer
        .propagate_crown_ternary(
            &node_lb,
            &make_bounded_flat(&x_lo, &x_hi),
            &make_bounded_flat(&g_lo, &g_hi),
            &make_bounded_flat(&b_lo, &b_hi),
        )
        .expect("ternary CROWN should succeed");

    assert_eq!(per_input.len(), 3);
    let lb_x = per_input[0].as_ref().unwrap();
    let lb_g = per_input[1].as_ref().unwrap();
    let lb_b = per_input[2].as_ref().unwrap();
    let lbs = [lb_x, lb_g, lb_b];
    let bounds = [
        (&x_lo[..], &x_hi[..]),
        (&g_lo[..], &g_hi[..]),
        (&b_lo[..], &b_hi[..]),
    ];

    for seed in 0..200_u32 {
        let x_s = sample_in_box(&x_lo, &x_hi, seed, 0);
        let g_s = sample_in_box(&g_lo, &g_hi, seed, 1);
        let b_s = sample_in_box(&b_lo, &b_hi, seed, 2);
        let y = eval_adain_true(&layer, &x_s, &g_s, &b_s);

        for j in 0..n {
            let lo = concretize_ternary_row(j, true, lbs, &bounds, bias_lower[j]);
            let hi = concretize_ternary_row(j, false, lbs, &bounds, bias_upper[j]);
            assert!(
                y[j] >= lo - 1e-2,
                "lower: seed={seed}, dim={j}: y={}, lo={lo}",
                y[j]
            );
            assert!(
                y[j] <= hi + 1e-2,
                "upper: seed={seed}, dim={j}: y={}, hi={hi}",
                y[j]
            );
        }
    }
}

/// Ternary CROWN with non-identity 2×3 A-matrix, stressing margin correction.
#[test]
fn test_crown_ternary_non_identity_a_soundness() {
    use crate::LinearBounds;
    use ndarray::Array2;

    let layer = make_variable_adain(1);
    let n = 3;
    let d = 2;

    let x_lo = [0.0_f32, 1.0, 2.0];
    let x_hi = [1.0_f32, 2.0, 3.0];
    let g_lo = [0.5_f32, 0.5, 0.5];
    let g_hi = [2.0_f32, 2.0, 2.0];
    let b_lo = [-1.0_f32, -1.0, -1.0];
    let b_hi = [1.0_f32, 1.0, 1.0];

    // A = [[1, -1, 0], [0.5, 0, 0.5]]
    let a = Array2::from_shape_vec((d, n), vec![1.0, -1.0, 0.0, 0.5, 0.0, 0.5]).unwrap();
    let node_lb = LinearBounds::new(a.clone(), Array1::zeros(d), a, Array1::zeros(d)).unwrap();

    let (per_input, bias_lower, bias_upper) = layer
        .propagate_crown_ternary(
            &node_lb,
            &make_bounded_flat(&x_lo, &x_hi),
            &make_bounded_flat(&g_lo, &g_hi),
            &make_bounded_flat(&b_lo, &b_hi),
        )
        .expect("ternary CROWN should succeed");

    let lb_x = per_input[0].as_ref().unwrap();
    let lb_g = per_input[1].as_ref().unwrap();
    let lb_b = per_input[2].as_ref().unwrap();
    let lbs = [lb_x, lb_g, lb_b];
    let bounds = [
        (&x_lo[..], &x_hi[..]),
        (&g_lo[..], &g_hi[..]),
        (&b_lo[..], &b_hi[..]),
    ];
    let a_rows: [[f32; 3]; 2] = [[1.0, -1.0, 0.0], [0.5, 0.0, 0.5]];

    for seed in 0..200_u32 {
        let x_s = sample_in_box(&x_lo, &x_hi, seed, 0);
        let g_s = sample_in_box(&g_lo, &g_hi, seed, 1);
        let b_s = sample_in_box(&b_lo, &b_hi, seed, 2);
        let y = eval_adain_true(&layer, &x_s, &g_s, &b_s);

        for row in 0..d {
            let composed: f32 = (0..n).map(|j| a_rows[row][j] * y[j]).sum();
            let lo = concretize_ternary_row(row, true, lbs, &bounds, bias_lower[row]);
            let hi = concretize_ternary_row(row, false, lbs, &bounds, bias_upper[row]);
            assert!(
                composed >= lo - 1e-2,
                "lower: seed={seed}, row={row}: c={composed}, lo={lo}"
            );
            assert!(
                composed <= hi + 1e-2,
                "upper: seed={seed}, row={row}: c={composed}, hi={hi}"
            );
        }
    }
}
