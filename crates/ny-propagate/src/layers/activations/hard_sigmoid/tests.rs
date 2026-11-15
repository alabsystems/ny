// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;
use crate::LinearBounds;
use ndarray::{array, ArrayD, IxDyn};

#[test]
fn test_try_new_rejects_invalid_params_2551() {
    for alpha in [-0.2, f32::NAN, f32::INFINITY] {
        let err =
            HardSigmoidLayer::try_new(alpha, 0.5).expect_err("invalid alpha should be rejected");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    for beta in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err =
            HardSigmoidLayer::try_new(0.2, beta).expect_err("invalid beta should be rejected");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }
}

/// alpha = 0 makes HardSigmoid the constant y = clamp(beta, 0, 1) — a
/// well-defined function the CROWN relaxation already handles exactly, so the
/// constructor admits it (the gate is aligned across constructor, IBP, CROWN).
#[test]
fn test_try_new_accepts_zero_alpha_constant_function() {
    let layer = HardSigmoidLayer::try_new(0.0, 0.5).expect("alpha = 0 must be admitted");
    assert_eq!(layer.alpha, 0.0);
}

// ── Relaxation function tests ──────────────────────────────────────

#[test]
fn test_relaxation_entirely_zero() {
    // x < -beta/alpha = -2.5: y = 0
    let r = hard_sigmoid_linear_relaxation(-10.0, -3.0, 0.2, 0.5);
    assert!(r.lower_slope.abs() < 1e-6);
    assert!(r.lower_intercept.abs() < 1e-6);
    assert!(r.upper_slope.abs() < 1e-6);
    assert!(r.upper_intercept.abs() < 1e-6);
}

#[test]
fn test_relaxation_entirely_one() {
    // x > (1 - beta)/alpha = 2.5: y = 1
    let r = hard_sigmoid_linear_relaxation(3.0, 10.0, 0.2, 0.5);
    assert!(r.lower_slope.abs() < 1e-6);
    assert!((r.lower_intercept - 1.0).abs() < 1e-6);
    assert!(r.upper_slope.abs() < 1e-6);
    assert!((r.upper_intercept - 1.0).abs() < 1e-6);
}

#[test]
fn test_relaxation_entirely_linear() {
    // Within linear region: y = 0.2*x + 0.5
    // x_low = -2.5, x_high = 2.5
    let r = hard_sigmoid_linear_relaxation(-2.0, 2.0, 0.2, 0.5);
    assert!((r.lower_slope - 0.2).abs() < 1e-6);
    assert!((r.lower_intercept - 0.5).abs() < 1e-6);
    assert!((r.upper_slope - 0.2).abs() < 1e-6);
    assert!((r.upper_intercept - 0.5).abs() < 1e-6);
}

#[test]
fn test_relaxation_nan() {
    let r = hard_sigmoid_linear_relaxation(f32::NAN, 1.0, 0.2, 0.5);
    assert!(r.lower_slope.abs() < 1e-6);
    assert!(r.lower_intercept.abs() < 1e-6);
    assert!(r.upper_slope.abs() < 1e-6);
    assert!((r.upper_intercept - 1.0).abs() < 1e-6);
}

// ── Relaxation soundness ───────────────────────────────────────────

#[test]
fn test_relaxation_soundness_grid() {
    let intervals: &[(f32, f32)] = &[
        (-5.0, 5.0),   // crosses both boundaries
        (-3.0, 0.0),   // crosses lower only
        (0.0, 3.0),    // crosses upper only
        (-2.0, 2.0),   // entirely linear
        (-10.0, -3.0), // entirely zero
        (3.0, 10.0),   // entirely one
    ];

    for &(l, u) in intervals {
        let r = hard_sigmoid_linear_relaxation(l, u, 0.2, 0.5);
        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = hard_sigmoid_eval(0.2, 0.5, x);
            let lower_bound = r.lower_slope * x + r.lower_intercept;
            let upper_bound = r.upper_slope * x + r.upper_intercept;
            assert!(
                lower_bound <= y + 1e-5,
                "[{},{}] x={}: lb {} > y {}",
                l,
                u,
                x,
                lower_bound,
                y
            );
            assert!(
                upper_bound >= y - 1e-5,
                "[{},{}] x={}: ub {} < y {}",
                l,
                u,
                x,
                upper_bound,
                y
            );
        }
    }
}

// ── IBP tests ──────────────────────────────────────────────────────

#[test]
fn test_ibp_fully_saturated_zero() {
    let layer = HardSigmoidLayer::default_params();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3]), -10.0_f32),
        ArrayD::from_elem(IxDyn(&[3]), -5.0_f32),
    )
    .unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    for &v in result.lower().iter() {
        assert!(v.abs() < 1e-5, "fully below: lower should be 0");
    }
    for &v in result.upper().iter() {
        assert!(v.abs() < 1e-5, "fully below: upper should be 0");
    }
}

#[test]
fn test_ibp_fully_saturated_one() {
    let layer = HardSigmoidLayer::default_params();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3]), 5.0_f32),
        ArrayD::from_elem(IxDyn(&[3]), 10.0_f32),
    )
    .unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    for &v in result.lower().iter() {
        assert!((v - 1.0).abs() < 1e-5, "fully above: lower should be 1");
    }
    for &v in result.upper().iter() {
        assert!((v - 1.0).abs() < 1e-5, "fully above: upper should be 1");
    }
}

#[test]
fn test_ibp_crossing() {
    let layer = HardSigmoidLayer::default_params();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -5.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 5.0_f32),
    )
    .unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    for &v in result.lower().iter() {
        assert!(v.abs() < 1e-5, "lower should be 0 (hs(-5)=0)");
    }
    for &v in result.upper().iter() {
        assert!((v - 1.0).abs() < 1e-5, "upper should be 1 (hs(5)=1)");
    }
}

// ── CROWN backward tests ───────────────────────────────────────────

#[test]
fn test_crown_fully_linear_region() {
    let layer = HardSigmoidLayer::default_params();
    let pre = BoundedTensor::new(array![-2.0_f32].into_dyn(), array![2.0_f32].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    // Within linear region: slope=0.2, intercept=0.5
    assert!((result.lower_a[[0, 0]] - 0.2).abs() < 1e-5);
    assert!((result.lower_b[0] - 0.5).abs() < 1e-5);
    assert!((result.upper_a[[0, 0]] - 0.2).abs() < 1e-5);
    assert!((result.upper_b[0] - 0.5).abs() < 1e-5);
}

#[test]
fn test_crown_crossing_soundness() {
    let layer = HardSigmoidLayer::default_params();
    let l = -5.0_f32;
    let u = 5.0_f32;
    let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = hard_sigmoid_eval(0.2, 0.5, x);
        let lower_bound = la * x + lb;
        let upper_bound = ua * x + ub;
        assert!(
            lower_bound <= y + 1e-5,
            "lb {} > y {} at x={}",
            lower_bound,
            y,
            x
        );
        assert!(
            upper_bound >= y - 1e-5,
            "ub {} < y {} at x={}",
            upper_bound,
            y,
            x
        );
    }
}

// ── Division-by-zero guard regression tests ─────────────────────

/// Regression: alpha=0 caused division by zero computing x_low and x_high.
#[test]
fn test_relaxation_alpha_zero_no_div_by_zero() {
    let r = hard_sigmoid_linear_relaxation(-1.0, 1.0, 0.0, 0.5);
    // alpha=0 → constant function y = clip(0.5, 0, 1) = 0.5
    assert!(r.lower_slope.abs() < 1e-6, "slope should be 0");
    assert!(
        (r.lower_intercept - 0.5).abs() < 1e-6,
        "intercept should be 0.5"
    );
    assert!(r.upper_slope.abs() < 1e-6, "slope should be 0");
    assert!(
        (r.upper_intercept - 0.5).abs() < 1e-6,
        "intercept should be 0.5"
    );
}

/// Regression: alpha=0 with beta outside [0,1] should clamp.
#[test]
fn test_relaxation_alpha_zero_beta_clamps() {
    let r = hard_sigmoid_linear_relaxation(-1.0, 1.0, 0.0, 2.0);
    assert!((r.lower_intercept - 1.0).abs() < 1e-6, "beta=2 clamps to 1");
    assert!((r.upper_intercept - 1.0).abs() < 1e-6, "beta=2 clamps to 1");

    let r = hard_sigmoid_linear_relaxation(-1.0, 1.0, 0.0, -1.0);
    assert!(r.lower_intercept.abs() < 1e-6, "beta=-1 clamps to 0");
    assert!(r.upper_intercept.abs() < 1e-6, "beta=-1 clamps to 0");
}

/// Regression: point interval (l == u) caused division by zero in crossing branches.
#[test]
fn test_relaxation_point_interval() {
    let r = hard_sigmoid_linear_relaxation(1.0, 1.0, 0.2, 0.5);
    let y = hard_sigmoid_eval(0.2, 0.5, 1.0);
    assert!(
        r.lower_slope.abs() < 1e-6,
        "slope should be 0 for point interval"
    );
    assert!(
        (r.lower_intercept - y).abs() < 1e-6,
        "intercept should be exact value"
    );
    assert!(
        r.upper_slope.abs() < 1e-6,
        "slope should be 0 for point interval"
    );
    assert!(
        (r.upper_intercept - y).abs() < 1e-6,
        "intercept should be exact value"
    );
}

#[test]
fn test_propagate_linear_requires_preact() {
    let layer = HardSigmoidLayer::default_params();
    let bounds = LinearBounds::identity(1);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("requires pre-activation");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

// ── IBP guard regression tests ────────────────────────────

#[test]
fn test_ibp_invalid_params_rejected() {
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 1.0_f32),
    )
    .unwrap();
    // Negative alpha flips monotonicity → refused with an accurate message
    let err = HardSigmoidLayer {
        alpha: -0.2,
        beta: 0.5,
    }
    .propagate_ibp(&input)
    .expect_err("negative alpha must stay refused");
    match err {
        NyError::InvalidSpec(msg) => assert!(
            msg.contains("monotonicity"),
            "negative-alpha refusal must explain the monotonicity flip: {msg}"
        ),
        other => panic!("expected InvalidSpec, got {other:?}"),
    }
    // Zero alpha: constant function y = clamp(beta, 0, 1) — now admitted and
    // computed exactly (aligned with the CROWN relaxation's alpha = 0 case).
    let out = HardSigmoidLayer {
        alpha: 0.0,
        beta: 0.5,
    }
    .propagate_ibp(&input)
    .expect("alpha = 0 is a constant function and must propagate");
    assert_eq!(out.lower()[[0]], 0.5);
    assert_eq!(out.upper()[[0]], 0.5);
    // NaN alpha
    assert!(matches!(
        HardSigmoidLayer {
            alpha: f32::NAN,
            beta: 0.5
        }
        .propagate_ibp(&input),
        Err(NyError::InvalidSpec(_))
    ));
    // NaN beta
    assert!(matches!(
        HardSigmoidLayer {
            alpha: 0.2,
            beta: f32::NAN
        }
        .propagate_ibp(&input),
        Err(NyError::InvalidSpec(_))
    ));
}

#[test]
fn test_ibp_nan_input_rejected() {
    let layer = HardSigmoidLayer::default_params();
    // NaN in lower bounds fails closed
    let nan_lower = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[2]), f32::NAN),
        ArrayD::from_elem(IxDyn(&[2]), 1.0_f32),
    )
    .unwrap();
    assert!(matches!(
        layer.propagate_ibp(&nan_lower),
        Err(NyError::NumericalInstability(_))
    ));
    // NaN in upper bounds fails closed
    let nan_upper = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), f32::NAN),
    )
    .unwrap();
    assert!(matches!(
        layer.propagate_ibp(&nan_upper),
        Err(NyError::NumericalInstability(_))
    ));
}

/// Saturation flow (dc9da0fde class): ±inf input endpoints are the deep-stack
/// IBP overflow convention and must map to HardSigmoid's exact limits —
/// HardSigmoid(-inf) = 0, HardSigmoid(+inf) = 1 — instead of aborting.
/// Supersedes the Inf half of the pre-migration
/// test_ibp_non_finite_input_rejected.
#[test]
fn test_ibp_saturated_input_flows_with_exact_limits() {
    let layer = HardSigmoidLayer::default_params();

    // Fully saturated box [-inf, +inf] → exactly [0, 1].
    let input = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[2]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[2]), f32::INFINITY),
    )
    .unwrap();
    let out = layer
        .propagate_ibp(&input)
        .expect("saturated input must propagate, not error");
    for &v in out.lower().iter() {
        assert_eq!(v, 0.0, "HardSigmoid(-inf) limit is exactly 0");
    }
    for &v in out.upper().iter() {
        assert_eq!(v, 1.0, "HardSigmoid(+inf) limit is exactly 1");
    }

    // Half-saturated box [-inf, 0] → [0, eval(0)] = [0, beta].
    let input = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[1]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[1]), 0.0_f32),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.lower()[[0]], 0.0);
    assert!(
        (out.upper()[[0]] - 0.5).abs() < 1e-6,
        "eval(0) = beta = 0.5"
    );

    // alpha = 0 stays exact on a saturated box: constant clamp(beta, 0, 1)
    // (the monotone endpoint mapping would compute 0·inf = NaN — the
    // dedicated constant path must bypass it).
    let layer = HardSigmoidLayer::try_new(0.0, 2.0).unwrap();
    let input = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[1]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[1]), f32::INFINITY),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.lower()[[0]], 1.0, "beta = 2 clamps to 1");
    assert_eq!(out.upper()[[0]], 1.0);
}
