// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward domain guard rejection tests (#3084).
//!
//! Mirrors `crown_domain_guard.rs` but exercises `propagate_linear_batched_with_bounds`
//! instead of `propagate_linear_with_bounds`. Layers with `non_finite_domain_guard`
//! reject any non-finite pre-activation on the batched path.
//!
//! ReLU and LeakyReLU use `nan_only_domain_guard` (proven infinite-case relaxation
//! branches), so they reject NaN but accept ±Inf and stay sound — covered by
//! `test_domain_guard_relu_batched_rejects_nan_accepts_inf` and
//! `test_domain_guard_leaky_relu_batched_rejects_nan_accepts_inf`.
//!
//! Part of #3084, #3070, #1696.

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
use crate::BatchedLinearBounds;
use ndarray::arr1;
use ny_core::NyError;
use ny_tensor::BoundedTensor;

fn make_non_finite_preact(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new_unchecked(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn()).unwrap()
}

fn batched_identity_bounds() -> BatchedLinearBounds {
    BatchedLinearBounds::identity(&[1]).unwrap()
}

/// Assert batched CROWN backward rejects non-finite pre-activation.
fn assert_rejects_non_finite_batched(
    name: &str,
    propagate: impl Fn(&BatchedLinearBounds, &BoundedTensor) -> crate::Result<BatchedLinearBounds>,
) {
    let cases: &[(f32, f32, &str)] = &[
        (f32::NAN, 1.0, "lower=NaN"),
        (-1.0, f32::NAN, "upper=NaN"),
        (f32::NEG_INFINITY, 1.0, "lower=-inf"),
        (-1.0, f32::INFINITY, "upper=+inf"),
    ];
    let bounds = batched_identity_bounds();
    for &(l, u, label) in cases {
        let pre = make_non_finite_preact(l, u);
        let result = propagate(&bounds, &pre);
        assert!(
            result.is_err(),
            "{name} batched ({label}): expected Err for non-finite pre-activation, got Ok"
        );
        match result.unwrap_err() {
            NyError::NumericalInstability(msg) => {
                assert!(
                    msg.contains("non-finite"),
                    "{name} batched ({label}): expected 'non-finite' in error, got: {msg}"
                );
            }
            other => {
                panic!("{name} batched ({label}): expected NumericalInstability, got: {other}");
            }
        }
    }
}

/// Assert batched CROWN backward rejects NaN but ACCEPTS infinite pre-activation
/// and returns a sound relaxation. For ReLU/LeakyReLU the batched path uses a
/// NaN-only guard backed by proven infinite-case relaxation branches.
fn assert_rejects_nan_accepts_inf_batched<F>(
    name: &str,
    f: F,
    propagate: impl Fn(&BatchedLinearBounds, &BoundedTensor) -> crate::Result<BatchedLinearBounds>,
) where
    F: Fn(f32) -> f32,
{
    for &(l, u, label) in &[(f32::NAN, 1.0, "lower=NaN"), (-1.0, f32::NAN, "upper=NaN")] {
        let pre = make_non_finite_preact(l, u);
        let result = propagate(&batched_identity_bounds(), &pre);
        assert!(
            result.is_err(),
            "{name} batched ({label}): expected Err for NaN, got Ok"
        );
        match result.unwrap_err() {
            NyError::NumericalInstability(msg) => assert!(
                msg.contains("NaN"),
                "{name} batched ({label}): expected 'NaN' in error, got: {msg}"
            ),
            other => {
                panic!("{name} batched ({label}): expected NumericalInstability, got: {other}")
            }
        }
    }

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
    for &(l, u, lo, hi, label) in inf_cases {
        let pre = make_non_finite_preact(l, u);
        let result = propagate(&batched_identity_bounds(), &pre).unwrap_or_else(|e| {
            panic!("{name} batched ({label}): expected Ok for infinite bounds, got {e}")
        });
        let ls = result.lower_a()[[0, 0]];
        let li = result.lower_b()[[0]];
        let us = result.upper_a()[[0, 0]];
        let ui = result.upper_b()[[0]];
        assert!(
            !ls.is_nan() && !li.is_nan() && !us.is_nan() && !ui.is_nan(),
            "{name} batched ({label}): NaN in relaxation"
        );
        for k in 0..=200 {
            let x = lo + (hi - lo) * (k as f32 / 200.0);
            let fx = f(x);
            let lower = ls * x + li;
            let upper = us * x + ui;
            let tol = 1e-4 * fx.abs().max(1.0);
            assert!(
                lower <= fx + tol,
                "{name} batched ({label}) lower UNSOUND at x={x}: {lower} > f={fx}"
            );
            assert!(
                upper + tol >= fx,
                "{name} batched ({label}) upper UNSOUND at x={x}: {upper} < f={fx}"
            );
        }
    }
}

// S-shaped activations (batched)

#[test]
fn test_domain_guard_rejects_tanh_batched() {
    let layer = TanhLayer::new();
    assert_rejects_non_finite_batched("Tanh", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_sigmoid_batched() {
    let layer = SigmoidLayer::new();
    assert_rejects_non_finite_batched("Sigmoid", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_arctan_batched() {
    let layer = ArctanLayer::new();
    assert_rejects_non_finite_batched("Arctan", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

// Trigonometric / Softplus (batched)

#[test]
fn test_domain_guard_rejects_softplus_batched() {
    let layer = SoftplusLayer::new();
    assert_rejects_non_finite_batched("Softplus", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_sin_batched() {
    let layer = SinLayer::new();
    assert_rejects_non_finite_batched("Sin", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_cos_batched() {
    let layer = CosLayer::new();
    assert_rejects_non_finite_batched("Cos", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_tan_batched() {
    let layer = TanLayer::new();
    assert_rejects_non_finite_batched("Tan", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

// Arithmetic (batched)

#[test]
fn test_domain_guard_rejects_abs_batched() {
    let layer = AbsLayer;
    assert_rejects_non_finite_batched("Abs", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_pow2_batched() {
    let layer = PowConstantLayer::square();
    assert_rejects_non_finite_batched("PowConstant", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_sqrt_batched() {
    let layer = SqrtLayer::new();
    assert_rejects_non_finite_batched("Sqrt", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

// Softmax family (batched)

#[test]
fn test_domain_guard_rejects_gelu_batched() {
    let layer = GELULayer::default();
    assert_rejects_non_finite_batched("GELU", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

// Non-macro activations with manual domain_guard (batched)

/// ReLU batched path uses a NaN-only guard: NaN rejected, infinite accepted + sound.
#[test]
fn test_domain_guard_relu_batched_rejects_nan_accepts_inf() {
    let layer = ReLULayer::new();
    assert_rejects_nan_accepts_inf_batched(
        "ReLU",
        |x| x.max(0.0),
        |b, p| layer.propagate_linear_batched_with_bounds(b, p),
    );
}

#[test]
fn test_domain_guard_rejects_hardswish_batched() {
    let layer = HardSwishLayer::new();
    assert_rejects_non_finite_batched("HardSwish", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_mish_batched() {
    let layer = MishLayer::new();
    assert_rejects_non_finite_batched("Mish", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_softsign_batched() {
    let layer = SoftsignLayer::new();
    assert_rejects_non_finite_batched("Softsign", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

// Macro-generated activations with domain_guard (batched)

#[test]
fn test_domain_guard_rejects_silu_batched() {
    let layer = SiLULayer::new();
    assert_rejects_non_finite_batched("SiLU", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_elu_batched() {
    let layer = EluLayer::default_alpha();
    assert_rejects_non_finite_batched("ELU", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_celu_batched() {
    let layer = CeluLayer::default_alpha();
    assert_rejects_non_finite_batched("CELU", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_selu_batched() {
    let layer = SeluLayer::new();
    assert_rejects_non_finite_batched("SELU", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

/// LeakyReLU batched path uses a NaN-only guard: NaN rejected, infinite accepted + sound.
#[test]
fn test_domain_guard_leaky_relu_batched_rejects_nan_accepts_inf() {
    for alpha in [0.01_f32, 0.5, -0.5, 2.0] {
        let layer = LeakyReLULayer::new(alpha);
        assert_rejects_nan_accepts_inf_batched(
            "LeakyReLU",
            move |x| if x >= 0.0 { x } else { alpha * x },
            |b, p| layer.propagate_linear_batched_with_bounds(b, p),
        );
    }
}

#[test]
fn test_domain_guard_rejects_hard_sigmoid_batched() {
    let layer = HardSigmoidLayer::default();
    assert_rejects_non_finite_batched("HardSigmoid", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_thresholded_relu_batched() {
    let layer = ThresholdedReluLayer::default();
    assert_rejects_non_finite_batched("ThresholdedRelu", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_snake_batched() {
    let layer = SnakeLayer::default_frequency();
    assert_rejects_non_finite_batched("Snake", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_prelu_batched() {
    let layer = PReluLayer::from_scalar(0.25);
    assert_rejects_non_finite_batched("PReLU", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_clip_batched() {
    let layer = ClipLayer::new(-1.0, 1.0);
    assert_rejects_non_finite_batched("Clip", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

// Piecewise constant layers (batched)

#[test]
fn test_domain_guard_rejects_floor_batched() {
    let layer = FloorLayer::new();
    assert_rejects_non_finite_batched("Floor", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_ceil_batched() {
    let layer = CeilLayer::new();
    assert_rejects_non_finite_batched("Ceil", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_round_batched() {
    let layer = RoundLayer::new();
    assert_rejects_non_finite_batched("Round", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_sign_batched() {
    let layer = SignLayer::new();
    assert_rejects_non_finite_batched("Sign", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}

#[test]
fn test_domain_guard_rejects_shrink_batched() {
    let layer = ShrinkLayer::default();
    assert_rejects_non_finite_batched("Shrink", |b, p| {
        layer.propagate_linear_batched_with_bounds(b, p)
    });
}
