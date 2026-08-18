// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU CROWN backward tests for `ActivationReluDualAlpha` (#4313).
//!
//! Verifies the four-slice ABI (lower_pos_slope, cross_slope, upper_neg_slope,
//! cross_intercept) matches the CPU reference for dual-alpha ReLU relaxation.

use super::*;

/// CPU reference dual-alpha activation backward (#4313).
#[allow(clippy::too_many_arguments)] // test helper
pub(super) fn cpu_dual_alpha_activation_backward(
    a_l: &mut Vec<f32>,
    a_u: &mut Vec<f32>,
    b_l: &mut [f32],
    b_u: &mut [f32],
    lower_pos_slope: &[f32],
    cross_slope: &[f32],
    upper_neg_slope: &[f32],
    cross_intercept: &[f32],
    num_specs: usize,
    n: usize,
) {
    let mut new_l = vec![0.0f32; num_specs * n];
    let mut new_u = vec![0.0f32; num_specs * n];
    for s in 0..num_specs {
        let (mut lb, mut ub) = (0.0f32, 0.0f32);
        for j in 0..n {
            let idx = s * n + j;
            let (al, au) = (a_l[idx], a_u[idx]);
            // Lower: pos uses alpha_lower, neg uses chord
            if al >= 0.0 {
                new_l[idx] = al * lower_pos_slope[j];
            } else {
                new_l[idx] = al * cross_slope[j];
                lb += al * cross_intercept[j];
            }
            // Upper: pos uses chord, neg uses alpha_upper
            if au >= 0.0 {
                new_u[idx] = au * cross_slope[j];
                ub += au * cross_intercept[j];
            } else {
                new_u[idx] = au * upper_neg_slope[j];
            }
        }
        b_l[s] += lb;
        b_u[s] += ub;
    }
    *a_l = new_l;
    *a_u = new_u;
}

/// Build layers: Linear2 -> DualAlpha -> Linear1 (backward order).
fn build_dual_alpha_layers(
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    lower_pos_slope: Vec<f32>,
    cross_slope: Vec<f32>,
    upper_neg_slope: Vec<f32>,
    cross_intercept: Vec<f32>,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
) -> Vec<GpuCrownLayer> {
    vec![
        GpuCrownLayer::Linear {
            weight: w2.into(),
            bias: Some(b2.into()),
            out_features: out_dim,
            in_features: hidden,
            cert_err: Default::default(),
        },
        GpuCrownLayer::ActivationReluDualAlpha {
            lower_pos_slope,
            cross_slope,
            upper_neg_slope,
            cross_intercept,
            num_neurons: hidden,
        },
        GpuCrownLayer::Linear {
            weight: w1.into(),
            bias: Some(b1.into()),
            out_features: hidden,
            in_features: in_dim,
            cert_err: Default::default(),
        },
    ]
}

/// Deterministic: dual-alpha with crossing neurons (mixed pre-activation signs).
#[test]
fn test_crown_backward_dual_alpha_crossing() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // Pre-activation bounds: neuron 0 is crossing [-1, 2], neuron 1 is crossing [-0.5, 1.5]
    // Chord slope = u/(u-l), intercept = -chord*l
    let chord_0 = 2.0 / (2.0 - (-1.0)); // 2/3
    let intercept_0 = -chord_0 * (-1.0); // 2/3
    let chord_1 = 1.5 / (1.5 - (-0.5)); // 3/4
    let intercept_1 = -chord_1 * (-0.5); // 3/8

    // Alpha slopes: lower_pos (alpha_lower for a>=0), upper_neg (alpha_upper for a<0)
    // Choose non-trivial values to distinguish from symmetric case
    let lower_pos_slope = vec![0.8, 0.6]; // alpha_lower for positive A-coeff
    let upper_neg_slope = vec![0.3, 0.2]; // alpha_upper for negative A-coeff
    let cross_slope = vec![chord_0, chord_1];
    let cross_intercept = vec![intercept_0, intercept_1];

    let layers = build_dual_alpha_layers(
        vec![0.3, -0.2, 0.1, 0.5], // w1: 2x2
        vec![0.0, 0.1],            // b1
        vec![0.5, -0.3, 0.2, 0.4], // w2: 2x2
        vec![0.1, -0.1],           // b2
        lower_pos_slope,
        cross_slope,
        upper_neg_slope,
        cross_intercept,
        2,
        2,
        2,
    );
    assert_gpu_matches_cpu(&device, &layers, 2, &[-1.0, -1.0], &[1.0, 1.0], 1e-4);
}

/// Dual-alpha with all-positive pre-activation (only lower_pos_slope matters for lower).
#[test]
fn test_crown_backward_dual_alpha_all_positive() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // All neurons have pre_l >= 0 → chord = identity (slope 1, intercept 0)
    // lower_pos_slope = alpha_lower, upper_neg_slope shouldn't matter
    let layers = build_dual_alpha_layers(
        vec![1.0, 0.0, 0.0, 1.0],  // w1: identity
        vec![1.0, 1.0],            // b1: shifts input positive
        vec![1.0, 0.5, -0.5, 1.0], // w2
        vec![0.0, 0.0],            // b2
        vec![0.9, 0.7],            // lower_pos_slope
        vec![1.0, 1.0],            // cross_slope (chord = identity for positive)
        vec![0.0, 0.0],            // upper_neg_slope (unused when all positive)
        vec![0.0, 0.0],            // cross_intercept (zero for positive)
        2,
        2,
        2,
    );
    assert_gpu_matches_cpu(&device, &layers, 2, &[0.0, 0.0], &[1.0, 1.0], 1e-4);
}

/// Dual-alpha with all-negative pre-activation (only upper_neg_slope matters for upper).
#[test]
fn test_crown_backward_dual_alpha_all_negative() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    // All neurons have pre_u <= 0 → dead ReLU, chord slope=0
    let layers = build_dual_alpha_layers(
        vec![1.0, 0.0, 0.0, 1.0],  // w1: identity
        vec![-2.0, -2.0],          // b1: shifts input negative
        vec![1.0, 0.5, -0.5, 1.0], // w2
        vec![0.0, 0.0],            // b2
        vec![0.0, 0.0],            // lower_pos_slope (dead)
        vec![0.0, 0.0],            // cross_slope (dead)
        vec![0.0, 0.0],            // upper_neg_slope (dead)
        vec![0.0, 0.0],            // cross_intercept (dead)
        2,
        2,
        2,
    );
    assert_gpu_matches_cpu(&device, &layers, 2, &[-1.0, -1.0], &[0.0, 0.0], 1e-4);
}

/// Negative control: divergent alpha_lower vs alpha_upper should produce
/// different bounds than a symmetric Activation with the same cross_slope.
#[test]
fn test_crown_backward_dual_alpha_differs_from_symmetric() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let chord_0 = 2.0 / 3.0;
    let intercept_0 = 2.0 / 3.0;
    let chord_1 = 0.75;
    let intercept_1 = 0.375;

    // Dual-alpha with asymmetric slopes
    let dual_layers = build_dual_alpha_layers(
        vec![0.3, -0.2, 0.1, 0.5],
        vec![0.0, 0.1],
        vec![0.5, -0.3, 0.2, 0.4],
        vec![0.1, -0.1],
        vec![0.8, 0.6], // != chord
        vec![chord_0, chord_1],
        vec![0.3, 0.2], // != chord
        vec![intercept_0, intercept_1],
        2,
        2,
        2,
    );

    // Symmetric Activation with chord as both slopes
    let sym_layers = vec![
        GpuCrownLayer::Linear {
            weight: vec![0.5, -0.3, 0.2, 0.4].into(),
            bias: Some(vec![0.1, -0.1].into()),
            out_features: 2,
            in_features: 2,
            cert_err: Default::default(),
        },
        GpuCrownLayer::Activation {
            lower_slope: vec![chord_0, chord_1],
            upper_slope: vec![chord_0, chord_1],
            lower_intercept: vec![intercept_0, intercept_1],
            upper_intercept: vec![intercept_0, intercept_1],
            num_neurons: 2,
        },
        GpuCrownLayer::Linear {
            weight: vec![0.3, -0.2, 0.1, 0.5].into(),
            bias: Some(vec![0.0, 0.1].into()),
            out_features: 2,
            in_features: 2,
            cert_err: Default::default(),
        },
    ];

    let spec = identity_spec(2);
    let inp_l = [-1.0, -1.0];
    let inp_u = [1.0, 1.0];

    let dual = device
        .crown_backward_gpu(&dual_layers, &spec, 2, &inp_l, &inp_u)
        .expect("dual-alpha should succeed");
    let sym = device
        .crown_backward_gpu(&sym_layers, &spec, 2, &inp_l, &inp_u)
        .expect("symmetric should succeed");

    // At least one bound should differ (asymmetric alphas ≠ symmetric chord)
    let any_diff = dual
        .lower_bounds
        .iter()
        .zip(&sym.lower_bounds)
        .chain(dual.upper_bounds.iter().zip(&sym.upper_bounds))
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        any_diff,
        "dual-alpha bounds should differ from symmetric: dual_l={:?} sym_l={:?} dual_u={:?} sym_u={:?}",
        dual.lower_bounds, sym.lower_bounds, dual.upper_bounds, sym.upper_bounds
    );
}

/// Wider network: hidden=8, in=4, out=3 — tests multi-workgroup dispatch.
#[test]
fn test_crown_backward_dual_alpha_wider() {
    let _gpu_serial = gpu_test_serial_guard();
    let device = require_device();

    let hidden = 8;
    let in_dim = 4;
    let out_dim = 3;

    let w1: Vec<f32> = (0..hidden * in_dim)
        .map(|i| 0.1 * (i as f32 - 16.0))
        .collect();
    let b1 = vec![0.0f32; hidden];
    let w2: Vec<f32> = (0..out_dim * hidden)
        .map(|i| 0.1 * (i as f32 - 12.0))
        .collect();
    let b2 = vec![0.0f32; out_dim];

    // Mixed crossing: alternating slopes to exercise both branches
    let lower_pos_slope: Vec<f32> = (0..hidden).map(|i| 0.5 + 0.05 * i as f32).collect();
    let cross_slope: Vec<f32> = (0..hidden).map(|i| 0.3 + 0.08 * i as f32).collect();
    let upper_neg_slope: Vec<f32> = (0..hidden).map(|i| 0.1 + 0.04 * i as f32).collect();
    let cross_intercept: Vec<f32> = (0..hidden).map(|i| 0.05 * i as f32).collect();

    let layers = build_dual_alpha_layers(
        w1,
        b1,
        w2,
        b2,
        lower_pos_slope,
        cross_slope,
        upper_neg_slope,
        cross_intercept,
        in_dim,
        hidden,
        out_dim,
    );
    let inp_l: Vec<f32> = vec![-1.0; in_dim];
    let inp_u: Vec<f32> = vec![1.0; in_dim];
    assert_gpu_matches_cpu(&device, &layers, out_dim, &inp_l, &inp_u, 1e-3);
}
