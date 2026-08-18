// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::LinearBounds;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use proptest::prelude::*;

#[test]
fn test_new_stores_scalar_alpha() {
    let layer = SnakeLayer::new(5.0).expect("test: valid Snake");
    assert_eq!(layer.alpha.len(), 1);
    assert!((layer.alpha[0] - 5.0).abs() < 1e-5);
}

#[test]
fn test_default_frequency() {
    let layer = SnakeLayer::default_frequency();
    assert_eq!(layer.alpha.len(), 1);
    assert!((layer.alpha[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_per_channel_constructor_rejects_empty_or_nonpositive() {
    assert!(
        SnakeLayer::per_channel(Array1::from_vec(vec![])).is_err(),
        "empty alpha must be rejected"
    );
    assert!(
        SnakeLayer::per_channel(Array1::from_vec(vec![1.0, 0.0])).is_err(),
        "zero alpha must be rejected"
    );
    assert!(
        SnakeLayer::per_channel(Array1::from_vec(vec![1.0, -0.5])).is_err(),
        "negative alpha must be rejected"
    );
    assert!(
        SnakeLayer::per_channel(Array1::from_vec(vec![1.0, f32::INFINITY])).is_err(),
        "infinite alpha must be rejected"
    );
    assert!(
        SnakeLayer::per_channel(Array1::from_vec(vec![1.0, f32::NAN])).is_err(),
        "NaN alpha must be rejected"
    );
}

#[test]
fn test_alpha_at_scalar_and_per_channel() {
    let scalar = SnakeLayer::new(5.0).expect("test: valid Snake");
    assert!((scalar.alpha_at(0) - 5.0).abs() < 1e-5);
    assert!((scalar.alpha_at(17) - 5.0).abs() < 1e-5);

    let per_channel =
        SnakeLayer::per_channel(Array1::from_vec(vec![0.5, 1.5, 3.0])).expect("valid alpha");
    assert!((per_channel.alpha_at(0) - 0.5).abs() < 1e-5);
    assert!((per_channel.alpha_at(1) - 1.5).abs() < 1e-5);
    assert!((per_channel.alpha_at(2) - 3.0).abs() < 1e-5);
    assert!((per_channel.alpha_at(3) - 0.5).abs() < 1e-5);
}

/// Verify IBP bounds contain the true function value at many sample points.
#[test]
fn test_ibp_soundness_grid() {
    let layer = SnakeLayer::new(5.0).expect("test: valid Snake");
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-10.0]).expect("test: valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0]).expect("test: valid shape"),
    )
    .expect("test: valid bounds");
    let out = layer.propagate_ibp(&input).expect("test: IBP propagation");

    // Since Snake is monotone, IBP bounds are exact: [f(l), f(u)]
    let expected_lower = snake_eval_f32(-10.0, 5.0);
    let expected_upper = snake_eval_f32(10.0, 5.0);
    assert!(
        (out.lower()[[0]] - expected_lower).abs() < 1e-4,
        "IBP lower {} != expected {}",
        out.lower()[[0]],
        expected_lower
    );
    assert!(
        (out.upper()[[0]] - expected_upper).abs() < 1e-4,
        "IBP upper {} != expected {}",
        out.upper()[[0]],
        expected_upper
    );

    // Verify bounds contain f(x) at grid points
    for i in 0..201 {
        let x = -10.0 + (i as f32) * 0.1;
        let y = snake_eval_f32(x, 5.0);
        assert!(
            out.lower()[[0]] <= y + 1e-4,
            "lower {} > eval {} at x={}",
            out.lower()[[0]],
            y,
            x
        );
        assert!(
            out.upper()[[0]] >= y - 1e-4,
            "upper {} < eval {} at x={}",
            out.upper()[[0]],
            y,
            x
        );
    }
}

/// Key test: verify that Snake IBP gives tight bounds for multi-period inputs,
/// unlike the Sin -> Pow composition which gives [-10, 110] for this case.
#[test]
fn test_ibp_multi_period_tight_bounds() {
    let layer = SnakeLayer::new(0.01).expect("test: valid Snake");
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-10.0]).expect("test: valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0]).expect("test: valid shape"),
    )
    .expect("test: valid bounds");
    let out = layer.propagate_ibp(&input).expect("test: IBP propagation");

    let bound_width = out.upper()[[0]] - out.lower()[[0]];
    assert!(
        bound_width < 25.0,
        "Snake IBP bound width {} should be within 2x of true range, not 120",
        bound_width
    );

    let true_lower = snake_eval_f32(-10.0, 0.01);
    let true_upper = snake_eval_f32(10.0, 0.01);
    let true_width = true_upper - true_lower;
    assert!(
        bound_width <= 2.0 * true_width + 1.0,
        "bound width {} > 2x true width {} (acceptance criteria)",
        bound_width,
        true_width
    );
}

/// Verify IBP with large alpha (many oscillations)
#[test]
fn test_ibp_large_alpha() {
    let layer = SnakeLayer::new(100.0).expect("test: valid Snake");
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-10.0]).expect("test: valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0]).expect("test: valid shape"),
    )
    .expect("test: valid bounds");
    let out = layer.propagate_ibp(&input).expect("test: IBP propagation");

    let bound_width = out.upper()[[0]] - out.lower()[[0]];
    assert!(
        bound_width < 20.5,
        "Large alpha bound width {} should be near 20.01",
        bound_width
    );
}

#[test]
fn test_per_channel_ibp_matches_elementwise_eval() {
    let layer =
        SnakeLayer::per_channel(Array1::from_vec(vec![0.5, 2.0, 4.0])).expect("valid alpha");
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, -1.0, 0.5]).expect("test: valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 1.5]).expect("test: valid shape"),
    )
    .expect("test: valid bounds");

    let out = layer.propagate_ibp(&input).expect("test: IBP propagation");
    let expected_lower = [-2.0, -1.0, 0.5];
    let expected_upper = [1.0, 2.0, 1.5];
    let alpha = [0.5, 2.0, 4.0];

    for i in 0..alpha.len() {
        let expected_l = snake_eval_f32(expected_lower[i], alpha[i]);
        let expected_u = snake_eval_f32(expected_upper[i], alpha[i]);
        assert!(
            (out.lower()[[i]] - expected_l).abs() < 1e-4,
            "dim {i}: lower {} != expected {}",
            out.lower()[[i]],
            expected_l
        );
        assert!(
            (out.upper()[[i]] - expected_u).abs() < 1e-4,
            "dim {i}: upper {} != expected {}",
            out.upper()[[i]],
            expected_u
        );
    }
}

/// Regression test for #4117: per-channel IBP with spatial dims [C, T].
///
/// The modulo-based alpha_at(idx % C) gives the wrong channel when T > 1
/// because flat index i corresponds to channel i/T, not i%C. The stride-based
/// alpha_for_flat(i, stride) fixes this.
#[test]
fn test_per_channel_ibp_with_spatial_dims_4117() {
    // Shape [C=2, T=3]: 2 channels, 3 time steps each
    let layer = SnakeLayer::per_channel(Array1::from_vec(vec![0.5, 3.0])).expect("valid alpha");
    let input = BoundedTensor::new(
        // Row-major [c0t0, c0t1, c0t2, c1t0, c1t1, c1t2]
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-2.0, -1.0, 0.0, -3.0, 1.0, 2.0])
            .expect("test: valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0])
            .expect("test: valid shape"),
    )
    .expect("test: valid bounds");

    let out = layer.propagate_ibp(&input).expect("IBP propagation");

    // Channel 0 (alpha=0.5): all time steps in first row
    for t in 0..3 {
        let expected_l = snake_eval_f32(input.lower()[[0, t]], 0.5);
        let expected_u = snake_eval_f32(input.upper()[[0, t]], 0.5);
        assert!(
            (out.lower()[[0, t]] - expected_l).abs() < 1e-4,
            "c0 t{t}: lower {} != expected {} (alpha should be 0.5)",
            out.lower()[[0, t]],
            expected_l
        );
        assert!(
            (out.upper()[[0, t]] - expected_u).abs() < 1e-4,
            "c0 t{t}: upper {} != expected {} (alpha should be 0.5)",
            out.upper()[[0, t]],
            expected_u
        );
    }

    // Channel 1 (alpha=3.0): all time steps in second row
    for t in 0..3 {
        let expected_l = snake_eval_f32(input.lower()[[1, t]], 3.0);
        let expected_u = snake_eval_f32(input.upper()[[1, t]], 3.0);
        assert!(
            (out.lower()[[1, t]] - expected_l).abs() < 1e-4,
            "c1 t{t}: lower {} != expected {} (alpha should be 3.0)",
            out.lower()[[1, t]],
            expected_l
        );
        assert!(
            (out.upper()[[1, t]] - expected_u).abs() < 1e-4,
            "c1 t{t}: upper {} != expected {} (alpha should be 3.0)",
            out.upper()[[1, t]],
            expected_u
        );
    }
}

/// Regression test for #4117: per-channel CROWN with spatial dims [C, T].
///
/// Verifies that CROWN backward correctly maps flat neuron index to channel
/// when the input has spatial dimensions beyond the channel axis.
#[test]
fn test_per_channel_crown_with_spatial_dims_4117() {
    // Shape [C=2, T=2]: 4 total elements, 2 channels
    let layer = SnakeLayer::per_channel(Array1::from_vec(vec![0.5, 3.0])).expect("valid alpha");
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, -0.5, -2.0, -1.5])
            .expect("test: valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.5, 2.0, 1.5])
            .expect("test: valid shape"),
    )
    .expect("test: valid bounds");
    let bounds = LinearBounds::identity(4);
    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("CROWN propagation");

    // Element (0,0) and (0,1) are channel 0 → alpha=0.5
    // Element (1,0) and (1,1) are channel 1 → alpha=3.0
    let relax_c0_e0 = snake_linear_relaxation(-1.0, 1.0, 0.5);
    let relax_c0_e1 = snake_linear_relaxation(-0.5, 0.5, 0.5);
    let relax_c1_e0 = snake_linear_relaxation(-2.0, 2.0, 3.0);
    let relax_c1_e1 = snake_linear_relaxation(-1.5, 1.5, 3.0);

    // Diagonal entries match channel-correct relaxation slopes
    assert!(
        (result.lower_a[[0, 0]] - relax_c0_e0.lower_slope).abs() < 1e-5,
        "c0 e0 lower slope: {} vs {}",
        result.lower_a[[0, 0]],
        relax_c0_e0.lower_slope
    );
    assert!(
        (result.lower_a[[1, 1]] - relax_c0_e1.lower_slope).abs() < 1e-5,
        "c0 e1 lower slope: {} vs {}",
        result.lower_a[[1, 1]],
        relax_c0_e1.lower_slope
    );
    assert!(
        (result.lower_a[[2, 2]] - relax_c1_e0.lower_slope).abs() < 1e-5,
        "c1 e0 lower slope: {} vs {}",
        result.lower_a[[2, 2]],
        relax_c1_e0.lower_slope
    );
    assert!(
        (result.lower_a[[3, 3]] - relax_c1_e1.lower_slope).abs() < 1e-5,
        "c1 e1 lower slope: {} vs {}",
        result.lower_a[[3, 3]],
        relax_c1_e1.lower_slope
    );

    // Off-diagonal entries should be zero (elementwise activation)
    assert!(result.lower_a[[0, 1]].abs() < 1e-6);
    assert!(result.lower_a[[1, 0]].abs() < 1e-6);
    assert!(result.lower_a[[2, 3]].abs() < 1e-6);
    assert!(result.lower_a[[3, 2]].abs() < 1e-6);
}

/// Verify CROWN relaxation soundness at many grid points.
#[test]
fn test_relaxation_soundness() {
    for &a in &[0.5, 1.0, 5.0, 10.0] {
        let r = snake_linear_relaxation(-3.0, 3.0, a);
        for i in 0..61 {
            let x = -3.0 + (i as f32) * 0.1;
            let y = snake_eval_f32(x, a);
            let lb = r.lower_slope * x + r.lower_intercept;
            let ub = r.upper_slope * x + r.upper_intercept;
            assert!(
                lb <= y + 1e-3,
                "a={}: lower {} > eval {} at x={}",
                a,
                lb,
                y,
                x
            );
            assert!(
                ub >= y - 1e-3,
                "a={}: upper {} < eval {} at x={}",
                a,
                ub,
                y,
                x
            );
        }
    }
}

/// Verify CROWN relaxation for multi-period inputs (the #3051 scenario).
#[test]
fn test_relaxation_multi_period_soundness() {
    let r = snake_linear_relaxation(-10.0, 10.0, 0.01);
    for i in 0..201 {
        let x = -10.0 + (i as f32) * 0.1;
        let y = snake_eval_f32(x, 0.01);
        let lb = r.lower_slope * x + r.lower_intercept;
        let ub = r.upper_slope * x + r.upper_intercept;
        assert!(
            lb <= y + 1e-2,
            "Multi-period lower {} > eval {} at x={}",
            lb,
            y,
            x
        );
        assert!(
            ub >= y - 1e-2,
            "Multi-period upper {} < eval {} at x={}",
            ub,
            y,
            x
        );
    }
}

#[test]
fn test_relaxation_nan_returns_sound() {
    let r = snake_linear_relaxation(f32::NAN, 1.0, 1.0);
    assert_eq!(r.lower_slope, 0.0);
    assert!(r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative());
    assert_eq!(r.upper_slope, 0.0);
    assert!(r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive());
}

#[test]
fn test_relaxation_point_interval() {
    let r = snake_linear_relaxation(1.0, 1.0, 5.0);
    let y = snake_eval_f32(1.0, 5.0);
    let lb = r.lower_slope * 1.0 + r.lower_intercept;
    let ub = r.upper_slope * 1.0 + r.upper_intercept;
    // Point interval: l == u, so relaxation should be exact (both bounds match eval).
    assert!(
        (lb - y).abs() < 1e-4,
        "Point interval lower {} far from eval {}",
        lb,
        y
    );
    assert!(
        (ub - y).abs() < 1e-4,
        "Point interval upper {} far from eval {}",
        ub,
        y
    );
}

#[test]
fn test_narrow_nonpoint_interval_does_not_use_tangent() {
    let l = 0.0_f32;
    let u = 1e-8_f32;
    let a = 1e8_f32;
    let r = snake_linear_relaxation(l, u, a);

    for &x in &[f64::from(l), 0.5 * f64::from(u), f64::from(u)] {
        let y = snake_eval_f64(x, f64::from(a));
        let lower = f64::from(r.lower_slope) * x + f64::from(r.lower_intercept);
        let upper = f64::from(r.upper_slope) * x + f64::from(r.upper_intercept);
        assert!(
            lower <= y,
            "narrow-interval lower envelope missed by {}",
            lower - y
        );
        assert!(
            upper >= y,
            "narrow-interval upper envelope missed by {}",
            y - upper
        );
    }
}

/// Regression test for #3083: infinite upper bound relaxation had slope/intercept swapped.
/// The lower bound was f(l)*x instead of constant f(l), which is unsound for large x.
#[test]
fn test_relaxation_infinite_upper_soundness_regression_3083() {
    let a = 1.0;
    let l = 2.0;
    let r = snake_linear_relaxation(l, f32::INFINITY, a);

    // Lower bound should be constant f(l), not f(l)*x
    assert!(
        r.lower_slope.abs() < 1e-6,
        "lower_slope should be 0 (constant bound), got {}",
        r.lower_slope
    );
    let fl = snake_eval_f32(l, a);
    assert!(
        r.lower_intercept <= fl,
        "lower_intercept {} should be <= f(l)={}",
        r.lower_intercept,
        fl
    );

    // Upper bound should be +inf
    assert_eq!(r.upper_slope, 0.0);
    assert!(r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive());

    // Verify soundness at large x: lower bound <= f(x)
    for &x in &[2.0, 5.0, 10.0, 100.0, 1000.0] {
        let y = snake_eval_f32(x, a);
        let lb = r.lower_slope * x + r.lower_intercept;
        assert!(
            lb <= y + 1e-3,
            "lower {} > f({})={} — the #3083 bug",
            lb,
            x,
            y
        );
    }
}

/// Regression test for #3083: verify infinite lower bound case is also sound.
#[test]
fn test_relaxation_infinite_lower_soundness() {
    let a = 1.0;
    let u = 2.0;
    let r = snake_linear_relaxation(f32::NEG_INFINITY, u, a);

    // Lower bound should be -inf (constant)
    assert_eq!(r.lower_slope, 0.0);
    assert!(r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative());

    // Upper bound should be constant f(u)
    assert!(
        r.upper_slope.abs() < 1e-6,
        "upper_slope should be 0 (constant bound), got {}",
        r.upper_slope
    );
    let fu = snake_eval_f32(u, a);
    assert!(
        r.upper_intercept >= fu,
        "upper_intercept {} should be >= f(u)={}",
        r.upper_intercept,
        fu
    );

    // Verify soundness at negative x: f(x) <= upper bound
    for &x in &[-1000.0, -100.0, -10.0, 0.0, 2.0] {
        let y = snake_eval_f32(x, a);
        let ub = r.upper_slope * x + r.upper_intercept;
        assert!(ub >= y - 1e-3, "upper {} < f({})={}", ub, x, y);
    }
}

/// Test both bounds infinite.
#[test]
fn test_relaxation_both_infinite() {
    let r = snake_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY, 1.0);
    assert_eq!(r.lower_slope, 0.0);
    assert!(r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative());
    assert_eq!(r.upper_slope, 0.0);
    assert!(r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive());
}

#[test]
fn test_relaxation_zero_alpha() {
    let r = snake_linear_relaxation(-1.0, 1.0, 0.0);
    assert!((r.lower_slope - 1.0).abs() < 0.01);
    assert!((r.upper_slope - 1.0).abs() < 0.01);
}

/// A nonzero alpha is never uniformly interchangeable with zero over the full
/// finite f32 domain.  At x ~= pi/(2a), the residual sin^2(a*x)/a is ~= 1/a
/// even when alpha is tiny.
#[test]
fn test_tiny_nonzero_alpha_relaxation_retains_global_residual() {
    for &a in &[1e-9_f32, 1e-12_f32, 1e-20_f32] {
        let x = (std::f64::consts::FRAC_PI_2 / f64::from(a)) as f32;
        let y = snake_eval_f64(f64::from(x), f64::from(a));
        assert!(
            y - f64::from(x) > 0.9 / f64::from(a),
            "test point must exercise the non-identity residual for a={a}"
        );

        let r = snake_linear_relaxation(0.0, x, a);
        let lower = f64::from(r.lower_slope) * f64::from(x) + f64::from(r.lower_intercept);
        let upper = f64::from(r.upper_slope) * f64::from(x) + f64::from(r.upper_intercept);
        assert!(
            lower <= y,
            "tiny-alpha lower envelope missed by {}",
            lower - y
        );
        assert!(
            upper >= y,
            "tiny-alpha upper envelope missed by {}",
            y - upper
        );
        assert!(r.upper_intercept > 0.0);
    }
}

#[test]
fn test_tiny_negative_alpha_internal_relaxation_is_symmetric_and_sound() {
    let a = -1e-9_f32;
    let x = (std::f64::consts::FRAC_PI_2 / f64::from(a.abs())) as f32;
    let y = snake_eval_f64(f64::from(x), f64::from(a));
    let r = snake_linear_relaxation(0.0, x, a);
    let lower = f64::from(r.lower_slope) * f64::from(x) + f64::from(r.lower_intercept);
    let upper = f64::from(r.upper_slope) * f64::from(x) + f64::from(r.upper_intercept);

    assert!(
        lower <= y,
        "negative-alpha lower envelope missed by {}",
        lower - y
    );
    assert!(
        upper >= y,
        "negative-alpha upper envelope missed by {}",
        y - upper
    );
    assert!(r.lower_intercept < 0.0);
    assert_eq!(r.upper_intercept, 0.0);
}

#[test]
fn test_relaxation_small_alpha_large_input_is_not_identity() {
    let a = 1.0e-12_f32;
    let l = 0.0_f32;
    let u = 3.084_070_9e22_f32;
    let r = snake_linear_relaxation(l, u, a);

    assert_eq!(r.lower_slope, 1.0);
    assert_eq!(r.lower_intercept, 0.0);
    assert_eq!(r.upper_slope, 1.0);
    assert!(r.upper_intercept >= (1.0 / f64::from(a)) as f32);

    for x in [l, 7.853_982e12_f32, u] {
        let y = snake_eval_f64(f64::from(x), f64::from(a));
        let lower = f64::from(r.lower_slope) * f64::from(x) + f64::from(r.lower_intercept);
        let upper = f64::from(r.upper_slope) * f64::from(x) + f64::from(r.upper_intercept);
        assert!(lower <= y, "small-alpha lower envelope failed at {x:e}");
        assert!(upper >= y, "small-alpha upper envelope failed at {x:e}");
    }
}

#[test]
fn test_relaxation_narrow_interval_high_frequency_is_not_a_point() {
    let a = 1.0e8_f32;
    let l = 0.0_f32;
    let u = 1.0e-8_f32;
    let r = snake_linear_relaxation(l, u, a);
    let y = snake_eval_f64(f64::from(u), f64::from(a));
    let upper = f64::from(r.upper_slope) * f64::from(u) + f64::from(r.upper_intercept);

    assert!(
        upper >= y,
        "high-frequency narrow interval upper envelope failed: {upper:e} < {y:e}"
    );
}

#[test]
fn test_propagate_linear_returns_error() {
    let layer = SnakeLayer::new(1.0).expect("test: valid Snake");
    let bounds = LinearBounds::new(
        Array2::eye(2),
        Array1::zeros(2),
        Array2::eye(2),
        Array1::zeros(2),
    )
    .expect("test: valid linear bounds");
    assert!(layer.propagate_linear(&bounds).is_err());
}

#[test]
fn test_requires_pre_activation_bounds() {
    let layer = SnakeLayer::new(1.0).expect("test: valid Snake");
    assert!(layer.requires_pre_activation_bounds());
}

#[test]
fn test_crown_backward_soundness() {
    let layer = SnakeLayer::new(2.0).expect("test: valid Snake");
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-3.0]).expect("test: valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).expect("test: valid shape"),
    )
    .expect("test: valid bounds");
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![1.0]).expect("test: valid shape"),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), vec![1.0]).expect("test: valid shape"),
        Array1::zeros(1),
    )
    .expect("test: valid linear bounds");
    let result = BoundPropagation::propagate_linear_with_bounds(&layer, &bounds, &pre_act)
        .expect("test: CROWN propagation");

    for i in 0..61 {
        let x = -3.0 + (i as f32) * 0.1;
        let y = snake_eval_f32(x, 2.0);
        let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(lb <= y + 1e-3, "CROWN lower {} > {} at x={}", lb, y, x);
        assert!(ub >= y - 1e-3, "CROWN upper {} < {} at x={}", ub, y, x);
    }
}

#[test]
fn test_per_channel_crown_relaxation_uses_channel_specific_alpha() {
    let layer =
        SnakeLayer::per_channel(Array1::from_vec(vec![0.5, 2.0])).expect("test: valid Snake");
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).expect("test: valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).expect("test: valid shape"),
    )
    .expect("test: valid bounds");
    let bounds = LinearBounds::identity(2);
    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("test: CROWN propagation");

    let relax0 = snake_linear_relaxation(-1.0, 1.0, 0.5);
    let relax1 = snake_linear_relaxation(-1.0, 1.0, 2.0);

    assert!((result.lower_a[[0, 0]] - relax0.lower_slope).abs() < 1e-5);
    assert!((result.upper_a[[0, 0]] - relax0.upper_slope).abs() < 1e-5);
    assert!((result.lower_b[0] - relax0.lower_intercept).abs() < 1e-5);
    assert!((result.upper_b[0] - relax0.upper_intercept).abs() < 1e-5);

    assert!((result.lower_a[[1, 1]] - relax1.lower_slope).abs() < 1e-5);
    assert!((result.upper_a[[1, 1]] - relax1.upper_slope).abs() < 1e-5);
    assert!((result.lower_b[1] - relax1.lower_intercept).abs() < 1e-5);
    assert!((result.upper_b[1] - relax1.upper_intercept).abs() < 1e-5);

    assert!(result.lower_a[[0, 1]].abs() < 1e-6);
    assert!(result.lower_a[[1, 0]].abs() < 1e-6);
    assert!(result.upper_a[[0, 1]].abs() < 1e-6);
    assert!(result.upper_a[[1, 0]].abs() < 1e-6);
}

#[test]
fn test_crown_backward_no_pre_activation_errors() {
    let layer = SnakeLayer::new(1.0).expect("test: valid Snake");
    let bounds = LinearBounds::new(
        Array2::eye(1),
        Array1::zeros(1),
        Array2::eye(1),
        Array1::zeros(1),
    )
    .expect("test: valid linear bounds");
    assert!(layer.propagate_crown_backward(&bounds, None).is_err());
}

/// Regression test for #3090: large a*width should not silently skip critical
/// points. The fix falls back to conservative sin² envelope bounds (±1/a) when
/// the critical point count exceeds MAX_PERIODIC_POINTS.
#[test]
fn test_relaxation_large_a_width_soundness_regression_3090() {
    // a=100 on [-50, 50]: ~3183 critical points per base, exceeds old 1000 cap
    for &(a, l, u) in &[
        (100.0_f32, -50.0_f32, 50.0_f32),
        (1000.0, -100.0, 100.0),
        (50.0, -100.0, 100.0),
    ] {
        let r = snake_linear_relaxation(l, u, a);

        // Verify soundness at dense grid of points
        let steps = 1000;
        for i in 0..=steps {
            let x = l + (u - l) * (i as f32) / (steps as f32);
            let y = snake_eval_f32(x, a);
            let lb = r.lower_slope * x + r.lower_intercept;
            let ub = r.upper_slope * x + r.upper_intercept;
            assert!(
                lb <= y + 1e-2,
                "a={a}: lower {lb} > eval {y} at x={x} (regression #3090)"
            );
            assert!(
                ub >= y - 1e-2,
                "a={a}: upper {ub} < eval {y} at x={x} (regression #3090)"
            );
        }
    }
}

/// Regression test for #3095: negative `a` causes unsound CROWN relaxation.
/// The constructor must reject `a <= 0` since Ziyin et al. 2020 defines
/// Snake only for positive frequency parameters.
#[test]
fn test_new_rejects_negative_a_3095() {
    assert!(
        SnakeLayer::new(-5.0).is_err(),
        "negative a should be rejected"
    );
    assert!(
        SnakeLayer::new(-0.001).is_err(),
        "small negative a should be rejected"
    );
    assert!(SnakeLayer::new(0.0).is_err(), "zero a should be rejected");
    assert!(
        SnakeLayer::new(f32::NEG_INFINITY).is_err(),
        "negative infinity should be rejected"
    );
    assert!(
        SnakeLayer::new(f32::INFINITY).is_err(),
        "positive infinity should be rejected"
    );
    assert!(SnakeLayer::new(f32::NAN).is_err(), "NaN should be rejected");
    // Positive values should still work
    assert!(
        SnakeLayer::new(0.001).is_ok(),
        "small positive a should be accepted"
    );
    assert!(
        SnakeLayer::new(100.0).is_ok(),
        "large positive a should be accepted"
    );
}

// ── Spatial per-channel IBP soundness proptest (#4117) ───────────────────
//
// Snake is monotonic (f'(x) = 1 + sin(2ax) >= 0), so IBP is exact:
// output = [snake(l, alpha_c), snake(u, alpha_c)]. The proptest verifies
// that the stride-based alpha_for_flat mapping selects the correct channel
// alpha for [C, T] inputs with C > 1 and T > 1.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    /// Property regression for #4117: per-channel alpha must stay constant
    /// across spatial positions within a channel for multi-element `[C, T]`
    /// inputs. The old modulo lookup (`idx % C`) violated this when `T > 1`.
    #[test]
    fn proptest_snake_ibp_spatial_per_channel_sound_4117(
        channels in 2usize..=8,
        spatial in 2usize..=16,
        alpha_base in 0.1f32..2.0f32,
        alpha_step in 0.05f32..0.5f32,
    ) {
        let alphas: Vec<f32> = (0..channels)
            .map(|c| alpha_base + alpha_step * c as f32)
            .collect();
        let layer = SnakeLayer::per_channel(Array1::from_vec(alphas.clone()))
            .expect("invariant: generated alpha vector has positive finite entries");

        let total = channels * spatial;
        let lower_flat: Vec<f32> = (0..total)
            .map(|i| -(1.0 + (i % spatial) as f32 * 0.3 + (i / spatial) as f32 * 0.2))
            .collect();
        let upper_flat: Vec<f32> = (0..total)
            .map(|i| 0.5 + (i % spatial) as f32 * 0.15)
            .collect();

        let input = BoundedTensor::new(
            Array2::from_shape_vec((channels, spatial), lower_flat.clone())
                .expect("invariant: shape matches channels * spatial")
                .into_dyn(),
            Array2::from_shape_vec((channels, spatial), upper_flat.clone())
                .expect("invariant: shape matches channels * spatial")
                .into_dyn(),
        )
        .expect("invariant: lower <= upper elementwise");

        let output = layer
            .propagate_ibp(&input)
            .expect("per-channel spatial inputs should propagate");
        let out_lower = output.lower().as_slice().expect("contiguous");
        let out_upper = output.upper().as_slice().expect("contiguous");

        for flat_idx in 0..total {
            let channel = flat_idx / spatial;
            let alpha = alphas[channel];
            let l = lower_flat[flat_idx];
            let u = upper_flat[flat_idx];
            // Snake is monotonic: exact IBP is [snake(l), snake(u)].
            let expected_l = snake_eval_f32(l, alpha);
            let expected_u = snake_eval_f32(u, alpha);

            prop_assert!(
                (out_lower[flat_idx] - expected_l).abs() < 1e-4,
                "flat_idx={flat_idx}, channel={channel}, alpha={alpha}: \
                 lower {} != snake(l={l})={expected_l}",
                out_lower[flat_idx]
            );
            prop_assert!(
                (out_upper[flat_idx] - expected_u).abs() < 1e-4,
                "flat_idx={flat_idx}, channel={channel}, alpha={alpha}: \
                 upper {} != snake(u={u})={expected_u}",
                out_upper[flat_idx]
            );
        }
    }
}

// ── Strict zero-tolerance CROWN relaxation proptest (#3292) ──────────────
//
// Pattern from #3285: f64-evaluated reference with zero tolerance catches
// f32 cancellation bugs invisible to magnitude-scaled tolerance tests.
// Uses snake_eval_f64 from the parent module as independent f64 reference.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Strict soundness proptest for Snake CROWN relaxation (a=1.0).
    /// Uses f64 reference (snake_eval_f64) with zero tolerance on 200-point grid.
    /// Ref: Ziyin et al. 2020 Snake activation, #3292.
    #[test]
    fn proptest_snake_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let a = 1.0_f32;
        let relax = snake_linear_relaxation(l, u, a);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = snake_eval_f64(x, a as f64);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "Snake lower bound UNSOUND at x={x}: {lower_val} > snake({x})={fx}, \
                 interval=[{l}, {u}], a={a}, gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "Snake upper bound UNSOUND at x={x}: {upper_val} < snake({x})={fx}, \
                 interval=[{l}, {u}], a={a}, gap={}", fx - upper_val
            );
        }
    }
}
