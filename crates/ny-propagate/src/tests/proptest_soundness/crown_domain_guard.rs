// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain guard rejection tests for CROWN backward entry points (#3070).
//!
//! Most activations use `non_finite_domain_guard` and reject any non-finite
//! (NaN or ±Inf) pre-activation bounds with `NumericalInstability`.
//!
//! ReLU and LeakyReLU are the exception: their `*_linear_relaxation` functions
//! carry proven infinite-case branches, so they use `nan_only_domain_guard` —
//! NaN is still rejected, but ±Inf endpoints are accepted and produce a sound
//! relaxation. Those two layers are covered by
//! `test_domain_guard_relu_rejects_nan_accepts_inf` and
//! `test_domain_guard_leaky_relu_rejects_nan_accepts_inf`.
//!
//! Part of #3070, #1696.

use crate::layers::activations::{
    CeluLayer, ClipLayer, EluLayer, HardSigmoidLayer, HardSwishLayer, LeakyReLULayer, MishLayer,
    PReluLayer, ReLULayer, SeluLayer, ShrinkLayer, SiLULayer, SnakeLayer, SoftsignLayer,
    ThresholdedReluLayer,
};
use crate::layers::arithmetic::{AbsLayer, PowConstantLayer, SqrtLayer};
use crate::layers::misc::{CeilLayer, FloorLayer, RoundLayer, SignLayer};
use crate::layers::softmax::GELULayer;
use crate::layers::trigonometric::{
    ArctanLayer, CosLayer, SigmoidLayer, SinLayer, SoftplusLayer, TanLayer, TanhLayer,
};
use crate::LinearBounds;
use ndarray::arr1;
use ny_core::NyError;
use ny_tensor::BoundedTensor;

fn make_non_finite_preact(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new_unchecked(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn()).unwrap()
}

fn identity_bounds() -> LinearBounds {
    LinearBounds::identity(1)
}

/// Assert CROWN backward rejects non-finite pre-activation with NumericalInstability.
fn assert_rejects_non_finite(
    name: &str,
    propagate: impl Fn(&LinearBounds, &BoundedTensor) -> crate::Result<LinearBounds>,
) {
    let cases: &[(f32, f32, &str)] = &[
        (f32::NAN, 1.0, "lower=NaN"),
        (-1.0, f32::NAN, "upper=NaN"),
        (f32::NEG_INFINITY, 1.0, "lower=-inf"),
        (-1.0, f32::INFINITY, "upper=+inf"),
    ];
    let bounds = identity_bounds();
    for &(l, u, label) in cases {
        let pre = make_non_finite_preact(l, u);
        let result = propagate(&bounds, &pre);
        assert!(
            result.is_err(),
            "{name} ({label}): expected Err for non-finite pre-activation, got Ok"
        );
        match result.unwrap_err() {
            NyError::NumericalInstability(msg) => {
                assert!(
                    msg.contains("non-finite"),
                    "{name} ({label}): expected 'non-finite' in error, got: {msg}"
                );
            }
            other => {
                panic!("{name} ({label}): expected NumericalInstability, got: {other}");
            }
        }
    }
}

/// Assert CROWN backward rejects NaN pre-activation but ACCEPTS infinite endpoints
/// and returns a sound linear relaxation over the (possibly unbounded) domain.
///
/// For piecewise-linear activations (ReLU, LeakyReLU) the `*_linear_relaxation`
/// functions carry proven over-approximation branches for l=-inf and/or u=+inf, so
/// the CROWN backward path uses a NaN-only guard (`nan_only_domain_guard`). This
/// recovers a tight sound bound on unbounded inputs that the stricter
/// `non_finite_domain_guard` would discard via IBP fallback.
fn assert_rejects_nan_accepts_inf<F>(
    name: &str,
    f: F,
    propagate: impl Fn(&LinearBounds, &BoundedTensor) -> crate::Result<LinearBounds>,
) where
    F: Fn(f32) -> f32,
{
    // NaN endpoints must still be rejected — a NaN cannot be soundly bounded.
    for &(l, u, label) in &[(f32::NAN, 1.0, "lower=NaN"), (-1.0, f32::NAN, "upper=NaN")] {
        let pre = make_non_finite_preact(l, u);
        let result = propagate(&identity_bounds(), &pre);
        assert!(
            result.is_err(),
            "{name} ({label}): expected Err for NaN pre-activation, got Ok"
        );
        match result.unwrap_err() {
            NyError::NumericalInstability(msg) => assert!(
                msg.contains("NaN"),
                "{name} ({label}): expected 'NaN' in error, got: {msg}"
            ),
            other => panic!("{name} ({label}): expected NumericalInstability, got: {other}"),
        }
    }

    // Infinite endpoints must be ACCEPTED and produce a sound relaxation.
    // (l, u, finite-sample-grid endpoints to probe within the bounded part)
    let inf_cases: &[(f32, f32, f32, f32, &str)] = &[
        (f32::NEG_INFINITY, 2.0, -50.0, 2.0, "l=-inf,u=2"),
        (-2.0, f32::INFINITY, -2.0, 50.0, "l=-2,u=+inf"),
        (
            f32::NEG_INFINITY,
            f32::INFINITY,
            -50.0,
            50.0,
            "l=-inf,u=+inf",
        ),
    ];
    for &(l, u, sample_lo, sample_hi, label) in inf_cases {
        let pre = make_non_finite_preact(l, u);
        let result = propagate(&identity_bounds(), &pre).unwrap_or_else(|e| {
            panic!("{name} ({label}): expected Ok for infinite bounds, got {e}")
        });

        let ls = result.lower_a[[0, 0]];
        let li = result.lower_b[0];
        let us = result.upper_a[[0, 0]];
        let ui = result.upper_b[0];

        // NaN must never appear — that would be unsound corruption.
        assert!(
            !ls.is_nan() && !li.is_nan() && !us.is_nan() && !ui.is_nan(),
            "{name} ({label}): NaN in relaxation: ls={ls}, li={li}, us={us}, ui={ui}"
        );

        // Soundness over a finite probe grid covering the bounded part of the domain.
        // Unbounded directions are allowed to fall back to a ±Inf plane (still sound).
        for k in 0..=200 {
            let x = sample_lo + (sample_hi - sample_lo) * (k as f32 / 200.0);
            let fx = f(x);
            let lower = ls * x + li;
            let upper = us * x + ui;
            let tol = 1e-4 * fx.abs().max(1.0);
            assert!(
                lower <= fx + tol,
                "{name} ({label}) lower UNSOUND at x={x}: {lower} > f(x)={fx}"
            );
            assert!(
                upper + tol >= fx,
                "{name} ({label}) upper UNSOUND at x={x}: {upper} < f(x)={fx}"
            );
        }
    }
}

// S-shaped activations

#[test]
fn test_domain_guard_rejects_tanh() {
    let layer = TanhLayer::new();
    assert_rejects_non_finite("Tanh", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_sigmoid() {
    let layer = SigmoidLayer::new();
    assert_rejects_non_finite("Sigmoid", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_arctan() {
    let layer = ArctanLayer::new();
    assert_rejects_non_finite("Arctan", |b, p| layer.propagate_linear_with_bounds(b, p));
}

// Trigonometric / Softplus

#[test]
fn test_domain_guard_rejects_softplus() {
    let layer = SoftplusLayer::new();
    assert_rejects_non_finite("Softplus", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_sin() {
    let layer = SinLayer::new();
    assert_rejects_non_finite("Sin", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_cos() {
    let layer = CosLayer::new();
    assert_rejects_non_finite("Cos", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_tan() {
    let layer = TanLayer::new();
    assert_rejects_non_finite("Tan", |b, p| layer.propagate_linear_with_bounds(b, p));
}

// Arithmetic

#[test]
fn test_domain_guard_rejects_abs() {
    let layer = AbsLayer;
    assert_rejects_non_finite("Abs", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_pow2() {
    let layer = PowConstantLayer::square();
    assert_rejects_non_finite("PowConstant", |b, p| {
        layer.propagate_linear_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_sqrt() {
    let layer = SqrtLayer::new();
    assert_rejects_non_finite("Sqrt", |b, p| layer.propagate_linear_with_bounds(b, p));
}

// Softmax family

#[test]
fn test_domain_guard_rejects_gelu() {
    let layer = GELULayer::default();
    assert_rejects_non_finite("GELU", |b, p| layer.propagate_linear_with_bounds(b, p));
}

// Non-macro activations with manual domain_guard

/// ReLU uses a NaN-only guard: NaN is rejected, but infinite pre-activation bounds
/// are accepted because `relu_linear_relaxation` has proven infinite-case branches.
#[test]
fn test_domain_guard_relu_rejects_nan_accepts_inf() {
    let layer = ReLULayer::new();
    assert_rejects_nan_accepts_inf(
        "ReLU",
        |x| x.max(0.0),
        |b, p| layer.propagate_linear_with_bounds(b, p),
    );
}

#[test]
fn test_domain_guard_rejects_hardswish() {
    let layer = HardSwishLayer::new();
    assert_rejects_non_finite("HardSwish", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_mish() {
    let layer = MishLayer::new();
    assert_rejects_non_finite("Mish", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_softsign() {
    let layer = SoftsignLayer::new();
    assert_rejects_non_finite("Softsign", |b, p| layer.propagate_linear_with_bounds(b, p));
}

// --- Coverage completions: 14 layers that had domain guards but no tests ---
// Added by Prover tool_quality audit.

// Macro-generated activations with domain_guard

#[test]
fn test_domain_guard_rejects_silu() {
    let layer = SiLULayer::new();
    assert_rejects_non_finite("SiLU", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_elu() {
    let layer = EluLayer::default_alpha();
    assert_rejects_non_finite("ELU", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_celu() {
    let layer = CeluLayer::default_alpha();
    assert_rejects_non_finite("CELU", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_selu() {
    let layer = SeluLayer::new();
    assert_rejects_non_finite("SELU", |b, p| layer.propagate_linear_with_bounds(b, p));
}

/// LeakyReLU uses a NaN-only guard: NaN is rejected, but infinite pre-activation
/// bounds are accepted because `leaky_relu_linear_relaxation` has proven
/// infinite-case branches. Tested across alpha in (0,1), alpha<0, and alpha>1.
#[test]
fn test_domain_guard_leaky_relu_rejects_nan_accepts_inf() {
    for alpha in [0.01_f32, 0.5, -0.5, 2.0] {
        let layer = LeakyReLULayer::new(alpha);
        assert_rejects_nan_accepts_inf(
            "LeakyReLU",
            move |x| if x >= 0.0 { x } else { alpha * x },
            |b, p| layer.propagate_linear_with_bounds(b, p),
        );
    }
}

#[test]
fn test_domain_guard_rejects_hard_sigmoid() {
    let layer = HardSigmoidLayer::default();
    assert_rejects_non_finite("HardSigmoid", |b, p| {
        layer.propagate_linear_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_thresholded_relu() {
    let layer = ThresholdedReluLayer::default();
    assert_rejects_non_finite("ThresholdedRelu", |b, p| {
        layer.propagate_linear_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_snake() {
    let layer = SnakeLayer::default_frequency();
    assert_rejects_non_finite("Snake", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_prelu() {
    let layer = PReluLayer::from_scalar(0.25);
    assert_rejects_non_finite("PReLU", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_clip() {
    let layer = ClipLayer::new(-1.0, 1.0);
    assert_rejects_non_finite("Clip", |b, p| layer.propagate_linear_with_bounds(b, p));
}

// Piecewise constant layers

#[test]
fn test_domain_guard_rejects_floor() {
    let layer = FloorLayer::new();
    assert_rejects_non_finite("Floor", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_ceil() {
    let layer = CeilLayer::new();
    assert_rejects_non_finite("Ceil", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_round() {
    let layer = RoundLayer::new();
    assert_rejects_non_finite("Round", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_sign() {
    let layer = SignLayer::new();
    assert_rejects_non_finite("Sign", |b, p| layer.propagate_linear_with_bounds(b, p));
}

#[test]
fn test_domain_guard_rejects_shrink() {
    let layer = ShrinkLayer::default();
    assert_rejects_non_finite("Shrink", |b, p| layer.propagate_linear_with_bounds(b, p));
}
