// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for arithmetic layer implementations.

use super::{
    abs_linear_relaxation, pow2_linear_relaxation, sqrt::sqrt_linear_relaxation_with_alpha,
    sqrt_linear_relaxation, AbsLayer, AddConstantLayer, DivConstantLayer, MulConstantLayer,
    PowConstantLayer, SqrtLayer, SubConstantLayer,
};
use crate::layers::activations::LinearRelaxation;
use crate::layers::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{array, Array2, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

/// Independent f64 sqrt reference for strict proptest. (#3292)
fn sqrt_f64_reference(x: f64) -> f64 {
    x.sqrt()
}

fn assert_close(actual: f32, expected: f32, tol: f32, label: impl std::fmt::Display) {
    assert!(
        (actual - expected).abs() < tol,
        "{label}: expected {expected}, got {actual}"
    );
}

fn assert_sqrt_envelope(l: f32, u: f32) {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = sqrt_linear_relaxation(l, u);
    let span = (u - l).abs();
    let samples = if span < 1e-8 { 1 } else { 100 };
    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let x = if i == 0 {
            l
        } else if i == samples {
            u
        } else {
            l + (u - l) * t
        };
        let fx = x.max(0.0).sqrt();
        let lower = ls * x + li;
        let upper = us * x + ui;
        let tol = (1e-6_f32).max(1e-4 * fx);
        assert!(
            lower <= fx + tol,
            "lower envelope violated for [{l}, {u}] at x={x}: {lower} > {fx}"
        );
        assert!(
            upper + tol >= fx,
            "upper envelope violated for [{l}, {u}] at x={x}: {upper} < {fx}"
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn sqrt_relaxation_envelope_basic() {
    let intervals = [
        (0.0, 0.0),
        (0.0, 1e-10),
        (0.0, 1.0),
        (0.1, 2.0),
        (-1.0, 0.0),
        (-1.0, 4.0),
        (-1000.0, 1.0),
        (-10.0, 1e-4),
        (1.0, 4.0),
        (1e-6, 1e-3),
        (2.0, 2.0),
    ];
    for (l, u) in intervals {
        assert_sqrt_envelope(l, u);
    }
}

#[ntest::timeout(5000)]
#[test]
fn sqrt_relaxation_invalid_bounds_fallback() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = sqrt_linear_relaxation(1.0, -1.0);
    assert_eq!(ls, 0.0);
    assert_eq!(us, 0.0);
    assert!(
        li.is_infinite() && li.is_sign_negative(),
        "invalid bounds lower intercept should be -inf, got {li}",
    );
    assert!(
        ui.is_infinite() && ui.is_sign_positive(),
        "invalid bounds upper intercept should be +inf, got {ui}",
    );
}

#[ntest::timeout(5000)]
#[test]
fn sqrt_relaxation_matches_alpha_at_chord_parallel_tangent() {
    // The default relaxation uses the chord-parallel (minimal-gap) tangent point
    // t* = ((sqrt(l)+sqrt(u))/2)^2, so it reproduces the with-alpha path fed t*
    // (not the loose tangent-at-u). The lower chord is independent of the tangent
    // point, so it matches the with-alpha path for any mid.
    for (l, u) in [(0.0_f32, 1.0_f32), (0.25, 2.5), (1.0, 4.0)] {
        let t_star = f32::midpoint((l.max(0.0)).sqrt(), u.max(0.0).sqrt()).powi(2);
        let default = sqrt_linear_relaxation(l, u);
        let alpha_at_tstar = sqrt_linear_relaxation_with_alpha(l, u, t_star);
        assert!(
            (default.lower_slope - alpha_at_tstar.lower_slope).abs() < 1e-7
                && (default.lower_intercept - alpha_at_tstar.lower_intercept).abs() < 1e-7
                && (default.upper_slope - alpha_at_tstar.upper_slope).abs() < 1e-7
                && (default.upper_intercept - alpha_at_tstar.upper_intercept).abs() < 1e-7,
            "default must reproduce the chord-parallel tangent relaxation for [{l}, {u}]"
        );

        // Lower chord is independent of the tangent point.
        let alpha_at_u = sqrt_linear_relaxation_with_alpha(l, u, u);
        assert!(
            (default.lower_slope - alpha_at_u.lower_slope).abs() < 1e-7
                && (default.lower_intercept - alpha_at_u.lower_intercept).abs() < 1e-7,
            "default lower chord must be independent of the upper tangent for [{l}, {u}]"
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn sqrt_relaxation_non_finite_inputs() {
    let cases = [
        (f32::NAN, 1.0),
        (0.0, f32::NAN),
        (f32::INFINITY, 1.0),
        (0.0, f32::INFINITY),
        (f32::NEG_INFINITY, 0.0),
        (0.0, f32::NEG_INFINITY),
    ];
    for (l, u) in cases {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = sqrt_linear_relaxation(l, u);
        assert_eq!(ls, 0.0);
        assert_eq!(us, 0.0);
        assert!(
            li.is_infinite() && li.is_sign_negative(),
            "non-finite sqrt lower intercept should be -inf for [{l}, {u}], got {li}",
        );
        assert!(
            ui.is_infinite() && ui.is_sign_positive(),
            "non-finite sqrt upper intercept should be +inf for [{l}, {u}], got {ui}",
        );
    }
}

#[test]
fn arithmetic_try_new_rejects_non_finite_params_4307() {
    let non_finite = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, f32::NAN]).unwrap();

    for err in [
        AddConstantLayer::try_new(non_finite.clone()).unwrap_err(),
        SubConstantLayer::try_new(non_finite.clone()).unwrap_err(),
        MulConstantLayer::try_new(non_finite.clone()).unwrap_err(),
        PowConstantLayer::try_new(f32::INFINITY).unwrap_err(),
    ] {
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    let div_non_finite = DivConstantLayer::try_new(non_finite).unwrap_err();
    assert!(matches!(div_non_finite, NyError::InvalidSpec(_)));

    let div_near_zero =
        DivConstantLayer::try_new(ArrayD::from_elem(IxDyn(&[]), 1.0e-12)).unwrap_err();
    assert!(matches!(div_near_zero, NyError::InvalidSpec(_)));
}

#[ntest::timeout(5000)]
#[test]
fn sqrt_relaxation_all_negative_interval() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = sqrt_linear_relaxation(-2.0, -0.5);
    assert_eq!(ls, 0.0);
    assert_eq!(us, 0.0);
    assert_eq!(li, 0.0);
    assert_eq!(ui, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn sqrt_relaxation_accepts_negative_pre_activation() {
    // #4113: ensure_nonnegative_bounds now returns Ok(()) for negative lower
    // bounds because the linear relaxation clamps l to max(l, 0) (line 143 in
    // sqrt.rs) and adjusts the upper intercept (lines 200-206). This preserves
    // CROWN tightness instead of falling back to IBP.
    let layer = SqrtLayer::new();
    let lower = array![-0.1].into_dyn();
    let upper = array![0.2].into_dyn();
    let pre_activation = BoundedTensor::new(lower, upper).expect("bounds construction");

    layer
        .ensure_nonnegative_bounds(&pre_activation)
        .expect("negative lower bound should be accepted (clamped by relaxation)");
}

#[ntest::timeout(5000)]
#[test]
fn sqrt_propagate_linear_requires_pre_activation_bounds() {
    let layer = SqrtLayer::new();
    let bounds = LinearBounds::identity(1);

    let err = layer
        .propagate_linear(&bounds)
        .expect_err("missing pre-activation bounds should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("requires pre-activation bounds"),
        "unexpected error message: {msg}"
    );
}

// ── Sqrt CROWN backward tests ────────────────────────────────────────

#[test]
fn test_sqrt_crown_backward_soundness() {
    let layer = SqrtLayer::new();
    let l = 0.5_f32;
    let u = 4.0_f32;
    let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = x.sqrt();
        let tol = (1e-4_f32).max(1e-3 * y);
        assert!(
            la * x + lb <= y + tol,
            "Sqrt CROWN lb violated at x={x}: {} > {y}",
            la * x + lb
        );
        assert!(
            ua * x + ub >= y - tol,
            "Sqrt CROWN ub violated at x={x}: {} < {y}",
            ua * x + ub
        );
    }
}

#[test]
fn test_sqrt_crown_backward_near_zero() {
    let layer = SqrtLayer::new();
    let l = 0.01_f32;
    let u = 1.0_f32;
    let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = x.sqrt();
        let tol = (1e-4_f32).max(1e-3 * y);
        assert!(
            result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + tol,
            "near-zero sqrt lb violated at x={x}"
        );
        assert!(
            result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - tol,
            "near-zero sqrt ub violated at x={x}"
        );
    }
}

#[test]
fn test_sqrt_crown_backward_multi_neuron() {
    let layer = SqrtLayer::new();
    let pre = BoundedTensor::new(
        array![0.1_f32, 1.0].into_dyn(),
        array![1.0_f32, 9.0].into_dyn(),
    )
    .unwrap();
    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    for neuron in 0..2 {
        let la = result.lower_a[[neuron, neuron]];
        let lb = result.lower_b[neuron];
        let ua = result.upper_a[[neuron, neuron]];
        let ub = result.upper_b[neuron];
        let lo = pre.lower()[neuron];
        let hi = pre.upper()[neuron];

        for k in 0..=20 {
            let x = lo + (hi - lo) * (k as f32 / 20.0);
            let y = x.sqrt();
            let tol = (1e-4_f32).max(1e-3 * y);
            assert!(
                la * x + lb <= y + tol,
                "neuron {neuron} lb violated at x={x}"
            );
            assert!(
                ua * x + ub >= y - tol,
                "neuron {neuron} ub violated at x={x}"
            );
        }
    }
}

#[test]
fn test_sqrt_crown_backward_clamps_negative_interval_4118() {
    let layer = SqrtLayer::new();
    let pre = BoundedTensor::new(array![-1.0_f32].into_dyn(), array![1.0_f32].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre)
        .expect("negative lower bound should clamp locally instead of erroring");

    assert!(
        result.lower_a.iter().all(|value| value.is_finite())
            && result.upper_a.iter().all(|value| value.is_finite())
            && result.lower_b.iter().all(|value| value.is_finite())
            && result.upper_b.iter().all(|value| value.is_finite()),
        "sqrt CROWN clamp regression should keep linear bounds finite, got {result:?}"
    );

    for k in 0..=20 {
        let x = k as f32 / 20.0;
        let y = x.sqrt();
        let lower = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let upper = result.upper_a[[0, 0]] * x + result.upper_b[0];
        let tol = (1e-4_f32).max(1e-3 * y);
        assert!(
            lower <= y + tol,
            "clamped sqrt lb violated at x={x}: {lower} > {y}"
        );
        assert!(
            upper >= y - tol,
            "clamped sqrt ub violated at x={x}: {upper} < {y}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// AddConstantLayer tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_add_constant_ibp_exact_shape() {
    // y = x + c where c = [1, 2, 3], x ∈ [[0,0,0], [4,5,6]]
    // Expected: y ∈ [[1,2,3], [5,7,9]]
    let layer = AddConstantLayer::new(array![1.0, 2.0, 3.0].into_dyn());
    let input = BoundedTensor::new(
        array![0.0, 0.0, 0.0].into_dyn(),
        array![4.0, 5.0, 6.0].into_dyn(),
    )
    .unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert_eq!(result.lower().as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    assert_eq!(result.upper().as_slice().unwrap(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_add_constant_ibp_negative_constant() {
    // y = x + (-5), x ∈ [2, 10]
    // Expected: y ∈ [-3, 5]
    let layer = AddConstantLayer::new(array![-5.0].into_dyn());
    let input = BoundedTensor::new(array![2.0].into_dyn(), array![10.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert_close(result.lower()[0], -3.0, 1e-6, "add constant lower[0]");
    assert_close(result.upper()[0], 5.0, 1e-6, "add constant upper[0]");
}

#[test]
fn test_add_constant_ibp_broadcast_cnn_bias() {
    // CNN bias: constant shape [2] added to input shape [2, 3, 3]
    let layer = AddConstantLayer::new(array![10.0, 20.0].into_dyn());
    let lower = ArrayD::zeros(IxDyn(&[2, 3, 3]));
    let upper = ArrayD::ones(IxDyn(&[2, 3, 3]));
    let input = BoundedTensor::new(lower, upper).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    // Channel 0 should be [10, 11], channel 1 should be [20, 21]
    assert_close(
        result.lower()[[0, 0, 0]],
        10.0,
        1e-6,
        "cnn bias lower[0,0,0]",
    );
    assert_close(
        result.upper()[[0, 0, 0]],
        11.0,
        1e-6,
        "cnn bias upper[0,0,0]",
    );
    assert_close(
        result.lower()[[1, 0, 0]],
        20.0,
        1e-6,
        "cnn bias lower[1,0,0]",
    );
    assert_close(
        result.upper()[[1, 0, 0]],
        21.0,
        1e-6,
        "cnn bias upper[1,0,0]",
    );
}

#[test]
fn test_add_constant_crown_backward_identity() {
    // CROWN backward: starting from identity bounds, y = x + c.
    // After substitution: A @ (x + c) + b = A @ x + (A @ c + b).
    // With identity A and zero b: new bias = I @ c = c.
    let c = 3.0_f32;
    let layer = AddConstantLayer::new(array![c].into_dyn());
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds).unwrap();

    // Coefficient matrices unchanged
    assert_eq!(result.lower_a[[0, 0]], 1.0);
    assert_eq!(result.upper_a[[0, 0]], 1.0);
    // Bias shifted by c
    assert_close(result.lower_b[0], c, 1e-6, "add constant lower_b[0]");
    assert_close(result.upper_b[0], c, 1e-6, "add constant upper_b[0]");
}

#[test]
fn test_add_constant_crown_backward_soundness() {
    // Verify CROWN bounds are sound: concretized bounds must contain true output.
    // y = x + 2.5, x ∈ [-1, 3]
    let c = 2.5_f32;
    let layer = AddConstantLayer::new(array![c].into_dyn());
    let input_bounds = BoundedTensor::new(array![-1.0].into_dyn(), array![3.0].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds).unwrap();
    let concrete = result.concretize(&input_bounds);
    // True output: [-1 + 2.5, 3 + 2.5] = [1.5, 5.5]
    assert!(
        concrete.lower()[0] <= 1.5 + 1e-5,
        "CROWN lower {} > true lower 1.5",
        concrete.lower()[0]
    );
    assert!(
        concrete.upper()[0] >= 5.5 - 1e-5,
        "CROWN upper {} < true upper 5.5",
        concrete.upper()[0]
    );
}

#[test]
fn test_add_constant_crown_backward_multi_output() {
    // 2D CROWN: A = [[1, 0], [0, 1]], y = x + [c1, c2]
    let layer = AddConstantLayer::new(array![1.0, -2.0].into_dyn());
    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear(&bounds).unwrap();
    assert_close(
        result.lower_b[0],
        1.0,
        1e-6,
        "add constant multi lower_b[0]",
    );
    assert_close(
        result.lower_b[1],
        -2.0,
        1e-6,
        "add constant multi lower_b[1]",
    );
    assert_close(
        result.upper_b[0],
        1.0,
        1e-6,
        "add constant multi upper_b[0]",
    );
    assert_close(
        result.upper_b[1],
        -2.0,
        1e-6,
        "add constant multi upper_b[1]",
    );
}

#[test]
fn test_add_constant_crown_backward_non_identity_a() {
    // Non-identity A: A = [[2, -1]], c = [3, 5]
    // Bias contribution: A @ c = 2*3 + (-1)*5 = 1
    let layer = AddConstantLayer::new(array![3.0, 5.0].into_dyn());
    let a = Array2::from_shape_vec((1, 2), vec![2.0, -1.0]).unwrap();
    let bounds = LinearBounds::new(a.clone(), array![0.0], a, array![0.0]).unwrap();
    let result = layer.propagate_linear(&bounds).unwrap();
    // Expected bias: 0 + (2*3 + (-1)*5) = 1
    assert_close(
        result.lower_b[0],
        1.0,
        1e-5,
        "add constant non-identity lower_b[0]",
    );
    assert_close(
        result.upper_b[0],
        1.0,
        1e-5,
        "add constant non-identity upper_b[0]",
    );
}

#[test]
fn test_add_constant_batched_crown() {
    // Batched CROWN: scalar c = 2.0, identity bounds on dim=3
    let layer = AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), 2.0));
    let bounds = BatchedLinearBounds::identity(&[3]).unwrap();
    let result = layer.propagate_linear_batched(&bounds).unwrap();
    // Each row of A has all ones along diagonal (sum = 1 for identity),
    // so bias contribution = c * 1 = 2.0 per output neuron.
    for i in 0..3 {
        assert!(
            (result.lower_b[i] - 2.0).abs() < 1e-6,
            "batched lower_b[{i}] = {}, expected 2.0",
            result.lower_b[i]
        );
        assert!(
            (result.upper_b[i] - 2.0).abs() < 1e-6,
            "batched upper_b[{i}] = {}, expected 2.0",
            result.upper_b[i]
        );
    }
}

#[test]
fn test_add_constant_batched_crown_handles_vector_constant() {
    // Batched CROWN supports vector constants via flatten_constant_to_in_dim.
    // For c = [1.0, 2.0] with identity A and zero b:
    // new_bias = A @ c + b = [1.0, 2.0]
    let layer = AddConstantLayer::new(array![1.0, 2.0].into_dyn());
    let bounds = BatchedLinearBounds::identity(&[2]).unwrap();
    let result = layer
        .propagate_linear_batched(&bounds)
        .expect("batched CROWN should handle vector AddConstant");
    assert_close(result.lower_b[[0]], 1.0, 1e-4, "batched add lower_b[0]");
    assert_close(result.lower_b[[1]], 2.0, 1e-4, "batched add lower_b[1]");
    assert_close(result.upper_b[[0]], 1.0, 1e-4, "batched add upper_b[0]");
    assert_close(result.upper_b[[1]], 2.0, 1e-4, "batched add upper_b[1]");
}

// ═══════════════════════════════════════════════════════════════════════
// SubConstantLayer tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_sub_constant_ibp_standard() {
    // y = x - 3, x ∈ [1, 5] => y ∈ [-2, 2]
    let layer = SubConstantLayer::scalar(3.0);
    let input = BoundedTensor::new(array![1.0].into_dyn(), array![5.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert_close(result.lower()[0], -2.0, 1e-6, "sub constant lower[0]");
    assert_close(result.upper()[0], 2.0, 1e-6, "sub constant upper[0]");
}

#[test]
fn test_sub_constant_ibp_reverse() {
    // y = 10 - x, x ∈ [3, 7] => y ∈ [3, 7]
    let layer = SubConstantLayer::new_reverse(array![10.0].into_dyn());
    let input = BoundedTensor::new(array![3.0].into_dyn(), array![7.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    // y_lower = 10 - 7 = 3, y_upper = 10 - 3 = 7
    assert_close(result.lower()[0], 3.0, 1e-6, "reverse sub lower[0]");
    assert_close(result.upper()[0], 7.0, 1e-6, "reverse sub upper[0]");
}

#[test]
fn test_sub_constant_ibp_reverse_bounds_swap() {
    // y = c - x swaps bounds: when x is large, y is small
    // y = 0 - x, x ∈ [-2, 5] => y ∈ [-5, 2]
    let layer = SubConstantLayer::new_reverse(array![0.0].into_dyn());
    let input = BoundedTensor::new(array![-2.0].into_dyn(), array![5.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert_close(result.lower()[0], -5.0, 1e-6, "reverse sub swap lower[0]");
    assert_close(result.upper()[0], 2.0, 1e-6, "reverse sub swap upper[0]");
}

#[test]
fn test_sub_constant_ibp_multi_element_broadcast() {
    // y = x - c, c = scalar 1.0, x has shape [3]
    let layer = SubConstantLayer::scalar(1.0);
    let input = BoundedTensor::new(
        array![2.0, 4.0, 6.0].into_dyn(),
        array![3.0, 5.0, 7.0].into_dyn(),
    )
    .unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    // IBP now rounds OUTWARD (directed rounding, #vnncomp-softmax-complex):
    // bounds must enclose the exact interval and stay within one ulp of it.
    for (got, exact) in result.lower().iter().zip([1.0f32, 3.0, 5.0]) {
        assert!(
            *got <= exact && exact - got < 1e-5,
            "lower {got} should enclose exact {exact} from below"
        );
    }
    for (got, exact) in result.upper().iter().zip([2.0f32, 4.0, 6.0]) {
        assert!(
            *got >= exact && got - exact < 1e-5,
            "upper {got} should enclose exact {exact} from above"
        );
    }
}

#[test]
fn test_sub_constant_crown_backward_standard() {
    // y = x - c, CROWN backward with identity: bias -= A @ c
    let layer = SubConstantLayer::scalar(5.0);
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds).unwrap();
    // Coefficients unchanged
    assert_eq!(result.lower_a[[0, 0]], 1.0);
    assert_eq!(result.upper_a[[0, 0]], 1.0);
    // Bias: 0 - 1*5 = -5
    assert!(
        (result.lower_b[0] - (-5.0)).abs() < 1e-5,
        "sub_constant lower_b should be -5.0, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.upper_b[0] - (-5.0)).abs() < 1e-5,
        "sub_constant upper_b should be -5.0, got {}",
        result.upper_b[0]
    );
}

#[test]
fn test_sub_constant_crown_backward_reverse() {
    // y = c - x, CROWN backward: coefficients negate, bias += A @ c
    let layer = SubConstantLayer::new_reverse(array![5.0].into_dyn());
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds).unwrap();
    // Coefficients negated
    assert_eq!(result.lower_a[[0, 0]], -1.0);
    assert_eq!(result.upper_a[[0, 0]], -1.0);
    // Bias: 0 + 1*5 = 5
    assert!(
        (result.lower_b[0] - 5.0).abs() < 1e-5,
        "sub_constant_reverse lower_b should be 5.0, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.upper_b[0] - 5.0).abs() < 1e-5,
        "sub_constant_reverse upper_b should be 5.0, got {}",
        result.upper_b[0]
    );
}

#[test]
fn test_sub_constant_crown_backward_soundness() {
    // Verify concretized CROWN bounds contain true output for both modes.
    let input_bounds = BoundedTensor::new(array![-1.0].into_dyn(), array![3.0].into_dyn()).unwrap();
    let c = 2.0_f32;

    // Standard: y = x - 2, true range: [-3, 1]
    let layer = SubConstantLayer::scalar(c);
    let id_bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&id_bounds).unwrap();
    let concrete = result.concretize(&input_bounds);
    assert!(
        concrete.lower()[0] <= -3.0 + 1e-5,
        "sub constant concrete lower {} should be <= -3.0",
        concrete.lower()[0]
    );
    assert!(
        concrete.upper()[0] >= 1.0 - 1e-5,
        "sub constant concrete upper {} should be >= 1.0",
        concrete.upper()[0]
    );

    // Reverse: y = 2 - x, true range: [-1, 3]
    let layer_rev = SubConstantLayer::new_reverse(array![c].into_dyn());
    let id_bounds_rev = LinearBounds::identity(1);
    let result_rev = layer_rev.propagate_linear(&id_bounds_rev).unwrap();
    let concrete_rev = result_rev.concretize(&input_bounds);
    assert!(
        concrete_rev.lower()[0] <= -1.0 + 1e-5,
        "reverse sub concrete lower {} should be <= -1.0",
        concrete_rev.lower()[0]
    );
    assert!(
        concrete_rev.upper()[0] >= 3.0 - 1e-5,
        "reverse sub concrete upper {} should be >= 3.0",
        concrete_rev.upper()[0]
    );
}

#[test]
fn test_sub_constant_batched_crown_standard() {
    let layer = SubConstantLayer::scalar(3.0);
    let bounds = BatchedLinearBounds::identity(&[2]).unwrap();
    let result = layer.propagate_linear_batched(&bounds).unwrap();
    for i in 0..2 {
        assert_close(
            result.lower_b[i],
            -3.0,
            1e-6,
            format!("batched sub lower_b[{i}]"),
        );
    }
}

#[test]
fn test_sub_constant_batched_crown_reverse() {
    let layer = SubConstantLayer::new_reverse(ArrayD::from_elem(IxDyn(&[]), 4.0));
    let bounds = BatchedLinearBounds::identity(&[2]).unwrap();
    let result = layer.propagate_linear_batched(&bounds).unwrap();
    // Coefficients negated
    for i in 0..2 {
        assert_eq!(result.lower_a[[i, i]], -1.0);
        assert_eq!(result.upper_a[[i, i]], -1.0);
        assert_close(
            result.lower_b[i],
            4.0,
            1e-6,
            format!("batched reverse sub lower_b[{i}]"),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MulConstantLayer tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_mul_constant_ibp_positive() {
    // y = x * 3, x ∈ [2, 5] => y ∈ [6, 15]
    let layer = MulConstantLayer::scalar(3.0);
    let input = BoundedTensor::new(array![2.0].into_dyn(), array![5.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert!(
        (result.lower()[0] - 6.0).abs() < 1e-6,
        "mul_constant(3) lower should be 6.0, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 15.0).abs() < 1e-6,
        "mul_constant(3) upper should be 15.0, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_mul_constant_ibp_negative_swaps_bounds() {
    // y = x * (-2), x ∈ [1, 4] => y ∈ [-8, -2]
    let layer = MulConstantLayer::scalar(-2.0);
    let input = BoundedTensor::new(array![1.0].into_dyn(), array![4.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert!(
        (result.lower()[0] - (-8.0)).abs() < 1e-6,
        "mul_constant(-2) lower should be -8.0, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - (-2.0)).abs() < 1e-6,
        "mul_constant(-2) upper should be -2.0, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_mul_constant_ibp_zero() {
    // y = x * 0, x ∈ [-100, 100] => y ∈ [0, 0]
    let layer = MulConstantLayer::scalar(0.0);
    let input = BoundedTensor::new(array![-100.0].into_dyn(), array![100.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert_eq!(result.lower()[0], 0.0);
    assert_eq!(result.upper()[0], 0.0);
}

#[test]
fn test_mul_constant_ibp_multi_element() {
    // y = x * [2, -3], x ∈ [[1, -1], [3, 2]]
    let layer = MulConstantLayer::new(array![2.0, -3.0].into_dyn());
    let input =
        BoundedTensor::new(array![1.0, -1.0].into_dyn(), array![3.0, 2.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    // Element 0: x*2, x ∈ [1,3] => [2, 6]
    assert_close(result.lower()[0], 2.0, 1e-6, "mul constant lower[0]");
    assert_close(result.upper()[0], 6.0, 1e-6, "mul constant upper[0]");
    // Element 1: x*(-3), x ∈ [-1,2] => [-6, 3] (swapped)
    assert_close(result.lower()[1], -6.0, 1e-6, "mul constant lower[1]");
    assert_close(result.upper()[1], 3.0, 1e-6, "mul constant upper[1]");
}

#[test]
fn test_mul_constant_crown_backward_scalar() {
    // y = x * c, CROWN: A @ (x * c) + b = (A * c) @ x + b
    // With identity A and c = 2: new_A = 2*I, bias unchanged
    let layer = MulConstantLayer::scalar(2.0);
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds).unwrap();
    assert_eq!(result.lower_a[[0, 0]], 2.0);
    assert_eq!(result.upper_a[[0, 0]], 2.0);
    assert_eq!(result.lower_b[0], 0.0);
    assert_eq!(result.upper_b[0], 0.0);
}

#[test]
fn test_mul_constant_crown_backward_negative() {
    // y = x * (-3), CROWN substitution (no swap): A → A * c
    let layer = MulConstantLayer::scalar(-3.0);
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds).unwrap();
    // No swap per CROWN substitution rule (designs/2026-01-29-crown-affine-negative-scale.md)
    assert_eq!(result.lower_a[[0, 0]], -3.0);
    assert_eq!(result.upper_a[[0, 0]], -3.0);
}

#[test]
fn test_mul_constant_crown_backward_soundness() {
    // Verify concretized CROWN bounds contain true output.
    // y = x * 2.5, x ∈ [-2, 3] => y ∈ [-5, 7.5]
    let layer = MulConstantLayer::scalar(2.5);
    let input_bounds = BoundedTensor::new(array![-2.0].into_dyn(), array![3.0].into_dyn()).unwrap();
    let id_bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&id_bounds).unwrap();
    let concrete = result.concretize(&input_bounds);
    assert!(
        concrete.lower()[0] <= -5.0 + 1e-5,
        "mul constant concrete lower {} should be <= -5.0",
        concrete.lower()[0]
    );
    assert!(
        concrete.upper()[0] >= 7.5 - 1e-5,
        "mul constant concrete upper {} should be >= 7.5",
        concrete.upper()[0]
    );
}

#[test]
fn test_mul_constant_crown_backward_neg_soundness() {
    // y = x * (-2), x ∈ [-1, 4] => y ∈ [-8, 2]
    let layer = MulConstantLayer::scalar(-2.0);
    let input_bounds = BoundedTensor::new(array![-1.0].into_dyn(), array![4.0].into_dyn()).unwrap();
    let id_bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&id_bounds).unwrap();
    let concrete = result.concretize(&input_bounds);
    assert!(
        concrete.lower()[0] <= -8.0 + 1e-5,
        "lower {} > true -8.0",
        concrete.lower()[0]
    );
    assert!(
        concrete.upper()[0] >= 2.0 - 1e-5,
        "upper {} < true 2.0",
        concrete.upper()[0]
    );
}

#[test]
fn test_mul_constant_batched_crown() {
    let layer = MulConstantLayer::scalar(0.5);
    let bounds = BatchedLinearBounds::identity(&[3]).unwrap();
    let result = layer.propagate_linear_batched(&bounds).unwrap();
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] - 0.5).abs() < 1e-6,
            "batched lower_a[{i},{i}] = {}, expected 0.5",
            result.lower_a[[i, i]]
        );
    }
}

#[test]
fn test_mul_constant_crown_backward_per_channel_broadcast_3896() {
    let layer = MulConstantLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 3.0]).unwrap(),
        vec![2, 3],
    );
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![
                -1.0, 0.5, 2.0, //
                -2.0, 1.0, 4.0,
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![
                1.0, 2.5, 3.0, //
                0.0, 2.0, 5.0,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let crown = layer
        .propagate_linear(&LinearBounds::identity(6))
        .expect("per-channel MulConstant CROWN backward should succeed")
        .concretize(&input_bounds);
    let ibp = layer
        .propagate_ibp(&input_bounds)
        .expect("MulConstant IBP should succeed");

    let crown_lower: Vec<f32> = crown.lower().iter().copied().collect();
    let crown_upper: Vec<f32> = crown.upper().iter().copied().collect();
    let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
    let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();
    for (index, (got, expected)) in crown_lower.iter().zip(ibp_lower.iter()).enumerate() {
        assert_close(*got, *expected, 1e-6, format!("mul crown lower[{index}]"));
    }
    for (index, (got, expected)) in crown_upper.iter().zip(ibp_upper.iter()).enumerate() {
        assert_close(*got, *expected, 1e-6, format!("mul crown upper[{index}]"));
    }
}

#[test]
fn test_mul_constant_batched_crown_per_channel_broadcast_3896() {
    let layer = MulConstantLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 3.0]).unwrap(),
        vec![2, 3],
    );
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();
    let result = layer
        .propagate_linear_batched(&bounds)
        .expect("per-channel MulConstant batched CROWN backward should succeed");

    for idx in 0..3 {
        assert_eq!(result.lower_a[[0, idx, idx]], 2.0);
        assert_eq!(result.upper_a[[0, idx, idx]], 2.0);
        assert_eq!(result.lower_a[[1, idx, idx]], 3.0);
        assert_eq!(result.upper_a[[1, idx, idx]], 3.0);
    }
}

#[test]
fn test_mul_constant_batched_crown_rejects_broadcast_expansion_3896() {
    let layer = MulConstantLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0, 3.0, 4.0, 5.0]).unwrap(),
        vec![1],
    );
    let bounds = BatchedLinearBounds::identity(&[4]).unwrap();
    let err = layer
        .propagate_linear_batched(&bounds)
        .expect_err("broadcast expansion should fall back instead of scaling batched CROWN");
    assert!(
        matches!(err, NyError::UnsupportedOp(ref message) if message.contains("broadcast expansion/reduction")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn test_mul_constant_crown_backward_per_channel_requires_input_shape_3896() {
    let layer =
        MulConstantLayer::new(ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 3.0]).unwrap());
    let err = layer
        .propagate_linear(&LinearBounds::identity(6))
        .expect_err("per-channel broadcast without input_shape should be rejected");
    assert!(
        matches!(err, NyError::UnsupportedOp(ref message) if message.contains("requires input_shape")),
        "unexpected error: {err:?}"
    );
}

/// Per-channel broadcast WITHOUT recorded input_shape must recover the layout
/// from the pre-activation shape on the `propagate_crown_backward` path and
/// produce coefficients identical to the recorded-input_shape path.
#[test]
fn test_mul_constant_crown_backward_runtime_shape_recovery() {
    let constant = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 3.0]).unwrap();
    let no_shape = MulConstantLayer::new(constant.clone());
    let with_shape = MulConstantLayer::with_input_shape(constant, vec![2, 3]);

    let pre_act = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 3]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32),
    )
    .unwrap();

    let recovered = no_shape
        .propagate_crown_backward(&LinearBounds::identity(6), Some(&pre_act))
        .expect("runtime-shape recovery should succeed with matching pre-activation");
    let recorded = with_shape
        .propagate_crown_backward(&LinearBounds::identity(6), Some(&pre_act))
        .expect("recorded input_shape path should succeed");

    assert_eq!(recovered.lower_a(), recorded.lower_a());
    assert_eq!(recovered.upper_a(), recorded.upper_a());
    assert_eq!(recovered.lower_b(), recorded.lower_b());
    assert_eq!(recovered.upper_b(), recorded.upper_b());

    // Without pre-activation the old conservative rejection must remain.
    let err = no_shape
        .propagate_crown_backward(&LinearBounds::identity(6), None)
        .expect_err("no input_shape and no pre-activation must stay rejected");
    assert!(
        matches!(err, NyError::UnsupportedOp(ref message) if message.contains("requires input_shape")),
        "unexpected error: {err:?}"
    );
}

/// A pre-activation whose element count disagrees with the incoming
/// coefficient columns (broadcast EXPANSION of x: input [1,3] * c [2,1] →
/// output [2,3]) must NOT take the runtime-shape recovery; column scaling
/// would be unsound there (each x element feeds several outputs).
#[test]
fn test_mul_constant_crown_backward_runtime_shape_rejects_expansion() {
    let layer =
        MulConstantLayer::new(ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 3.0]).unwrap());
    let pre_act = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 3]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1, 3]), 1.0f32),
    )
    .unwrap();
    // Incoming bounds live on the OUTPUT (6 elements); pre-activation has 3.
    let err = layer
        .propagate_crown_backward(&LinearBounds::identity(6), Some(&pre_act))
        .expect_err("expansion case must stay rejected");
    assert!(
        matches!(err, NyError::UnsupportedOp(ref message) if message.contains("requires input_shape")),
        "unexpected error: {err:?}"
    );
}

/// DivConstant delegates the same runtime-shape recovery (y = x * (1/c)).
#[test]
fn test_div_constant_crown_backward_runtime_shape_recovery() {
    let constant = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 4.0]).unwrap();
    let layer = DivConstantLayer::new(constant);
    let pre_act = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 3]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32),
    )
    .unwrap();
    let result = layer
        .propagate_crown_backward(&LinearBounds::identity(6), Some(&pre_act))
        .expect("DivConstant runtime-shape recovery should succeed");
    for j in 0..6 {
        let expected = if j < 3 { 0.5 } else { 0.25 };
        assert_close(
            result.lower_a()[[j, j]],
            expected,
            1e-6,
            format!("div lower_a[{j},{j}]"),
        );
        assert_close(
            result.upper_a()[[j, j]],
            expected,
            1e-6,
            format!("div upper_a[{j},{j}]"),
        );
    }
}

/// #3034: MulConstant CROWN backward with c=0.0 and Inf input coefficients
/// must produce zero output coefficients, not NaN from Inf * 0.0.
///
/// Uses direct struct construction because LinearBounds::new() rejects Inf
/// in coefficients. This test specifically needs Inf to verify the c==0
/// short-circuit avoids Inf * 0.0 = NaN.
#[test]
fn test_mul_constant_crown_backward_zero_with_inf_coeff_3034() {
    // Non-batched path: propagate_linear with c=0 and Inf in A-matrices.
    let layer = MulConstantLayer::scalar(0.0);
    // Construct LinearBounds via direct field initialization to bypass both
    // LinearBounds::new() (rejects Inf) and from_parts_unchecked() (debug_assert
    // rejects Inf). In production, Inf coefficients can accumulate during CROWN
    // backward propagation through multiple layers before reaching MulConstant.
    let bounds = LinearBounds {
        lower_a: Array2::from_shape_vec((1, 2), vec![f32::INFINITY, -1.0])
            .expect("invariant: shape valid"),
        lower_b: array![1.0],
        upper_a: Array2::from_shape_vec((1, 2), vec![f32::NEG_INFINITY, 2.0])
            .expect("invariant: shape valid"),
        upper_b: array![3.0],
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = layer
        .propagate_linear(&bounds)
        .expect("propagate_linear should succeed");

    // All coefficients must be exactly 0.0 (c==0 short-circuit avoids Inf*0=NaN).
    for j in 0..2 {
        assert_eq!(
            result.lower_a[[0, j]],
            0.0,
            "lower_a[0,{j}] should be 0.0, got {}",
            result.lower_a[[0, j]]
        );
        assert_eq!(
            result.upper_a[[0, j]],
            0.0,
            "upper_a[0,{j}] should be 0.0, got {}",
            result.upper_a[[0, j]]
        );
    }
    // Bias unchanged.
    assert_eq!(result.lower_b[0], 1.0);
    assert_eq!(result.upper_b[0], 3.0);
}

/// #3034: MulConstant batched CROWN backward with c=0.0 and Inf coefficients.
#[test]
fn test_mul_constant_batched_crown_zero_with_inf_coeff_3034() {
    use ndarray::ArrayD;

    let layer = MulConstantLayer::scalar(0.0);
    // Create batched bounds with Inf in coefficient matrices.
    let lower_a = ArrayD::from_shape_vec(
        IxDyn(&[2, 2]),
        vec![f32::INFINITY, -1.0, 0.0, f32::NEG_INFINITY],
    )
    .expect("invariant: shape valid");
    let upper_a = ArrayD::from_shape_vec(
        IxDyn(&[2, 2]),
        vec![f32::NEG_INFINITY, 2.0, 0.0, f32::INFINITY],
    )
    .expect("invariant: shape valid");
    let lower_b =
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).expect("invariant: shape valid");
    let upper_b =
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).expect("invariant: shape valid");

    let bounds = BatchedLinearBounds::new(lower_a, lower_b, upper_a, upper_b, vec![2], vec![2])
        .expect("invariant: batched bounds valid");

    let result = layer
        .propagate_linear_batched(&bounds)
        .expect("propagate_linear_batched should succeed");

    // All coefficients must be exactly 0.0, not NaN.
    for val in result.lower_a.iter() {
        assert_eq!(*val, 0.0, "lower_a element should be 0.0, got {val}");
    }
    for val in result.upper_a.iter() {
        assert_eq!(*val, 0.0, "upper_a element should be 0.0, got {val}");
    }
    // Bias unchanged.
    assert_eq!(result.lower_b[[0]], 1.0);
    assert_eq!(result.lower_b[[1]], 2.0);
    assert_eq!(result.upper_b[[0]], 3.0);
    assert_eq!(result.upper_b[[1]], 4.0);
}

/// #3273: MulConstant IBP with c=0.0 and Inf input bounds must not produce NaN.
/// Prior to fix, `Inf * 0.0 = NaN` poisoned output bounds.
#[test]
fn test_mul_constant_ibp_zero_with_inf_bounds_3273() {
    let layer = MulConstantLayer::scalar(0.0);
    let input = BoundedTensor::new_allow_infinite(
        array![f32::NEG_INFINITY].into_dyn(),
        array![f32::INFINITY].into_dyn(),
    )
    .unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    // x * 0 = 0 for all x, even x = ±Inf.
    assert_eq!(
        result.lower()[0],
        0.0,
        "lower should be 0.0, got {}",
        result.lower()[0]
    );
    assert_eq!(
        result.upper()[0],
        0.0,
        "upper should be 0.0, got {}",
        result.upper()[0]
    );
}

/// #3273: MulConstant IBP with c=0.0, multi-element tensor mixing Inf and finite bounds.
#[test]
fn test_mul_constant_ibp_zero_multi_element_with_inf_3273() {
    let layer = MulConstantLayer::new(array![0.0, 2.0, 0.0].into_dyn());
    let input = BoundedTensor::new_allow_infinite(
        array![f32::NEG_INFINITY, 1.0, -5.0].into_dyn(),
        array![f32::INFINITY, 3.0, f32::INFINITY].into_dyn(),
    )
    .unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    // Element 0: c=0, x ∈ [-Inf, Inf] => [0, 0]
    assert_eq!(result.lower()[0], 0.0);
    assert_eq!(result.upper()[0], 0.0);
    // Element 1: c=2, x ∈ [1, 3] => [2, 6]
    assert_close(result.lower()[1], 2.0, 1e-6, "mul zero-mixed lower[1]");
    assert_close(result.upper()[1], 6.0, 1e-6, "mul zero-mixed upper[1]");
    // Element 2: c=0, x ∈ [-5, Inf] => [0, 0]
    assert_eq!(result.lower()[2], 0.0);
    assert_eq!(result.upper()[2], 0.0);
}

// ═══════════════════════════════════════════════════════════════════════
// DivConstantLayer tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_div_constant_ibp_positive() {
    // y = x / 2, x ∈ [4, 10] => y ∈ [2, 5]
    let layer = DivConstantLayer::scalar(2.0);
    let input = BoundedTensor::new(array![4.0].into_dyn(), array![10.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert_close(result.lower()[0], 2.0, 1e-6, "div constant lower[0]");
    assert_close(result.upper()[0], 5.0, 1e-6, "div constant upper[0]");
}

#[test]
fn test_div_constant_ibp_negative_swaps() {
    // y = x / (-2), x ∈ [4, 10] => y ∈ [-5, -2]
    let layer = DivConstantLayer::scalar(-2.0);
    let input = BoundedTensor::new(array![4.0].into_dyn(), array![10.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert_close(result.lower()[0], -5.0, 1e-6, "negative div lower[0]");
    assert_close(result.upper()[0], -2.0, 1e-6, "negative div upper[0]");
}

#[test]
fn test_div_constant_ibp_rejects_near_zero() {
    let err = DivConstantLayer::try_scalar(1e-12).expect_err("near-zero divisor should fail");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[test]
fn test_div_constant_crown_delegates_to_mul_inverse() {
    // y = x / 4 is equivalent to y = x * 0.25
    // CROWN backward: A → A * (1/c) = A * 0.25
    let layer = DivConstantLayer::scalar(4.0);
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds).unwrap();
    assert_close(result.lower_a[[0, 0]], 0.25, 1e-6, "div lower_a[0,0]");
    assert_close(result.upper_a[[0, 0]], 0.25, 1e-6, "div upper_a[0,0]");
    assert_eq!(result.lower_b[0], 0.0);
}

#[test]
fn test_div_constant_crown_backward_soundness() {
    // y = x / 3, x ∈ [0, 9] => y ∈ [0, 3]
    let layer = DivConstantLayer::scalar(3.0);
    let input_bounds = BoundedTensor::new(array![0.0].into_dyn(), array![9.0].into_dyn()).unwrap();
    let id_bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&id_bounds).unwrap();
    let concrete = result.concretize(&input_bounds);
    assert!(
        concrete.lower()[0] <= 0.0 + 1e-5,
        "div constant concrete lower {} should be <= 0.0",
        concrete.lower()[0]
    );
    assert!(
        concrete.upper()[0] >= 3.0 - 1e-5,
        "div constant concrete upper {} should be >= 3.0",
        concrete.upper()[0]
    );
}

#[test]
fn test_div_constant_batched_crown() {
    let layer = DivConstantLayer::scalar(2.0);
    let bounds = BatchedLinearBounds::identity(&[2]).unwrap();
    let result = layer.propagate_linear_batched(&bounds).unwrap();
    for i in 0..2 {
        assert_close(
            result.lower_a[[i, i]],
            0.5,
            1e-6,
            format!("batched div lower_a[{i},{i}]"),
        );
    }
}

#[test]
fn test_div_constant_crown_backward_per_channel_broadcast_3896() {
    let layer = DivConstantLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 4.0]).unwrap(),
        vec![2, 3],
    );
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![
                -2.0, 1.0, 2.0, //
                -4.0, 0.0, 4.0,
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![
                2.0, 3.0, 6.0, //
                0.0, 8.0, 12.0,
            ],
        )
        .unwrap(),
    )
    .unwrap();

    let crown = layer
        .propagate_linear(&LinearBounds::identity(6))
        .expect("per-channel DivConstant CROWN backward should succeed")
        .concretize_sound(&input_bounds);
    let ibp = layer
        .propagate_ibp(&input_bounds)
        .expect("DivConstant IBP should succeed");

    let crown_lower: Vec<f32> = crown.lower().iter().copied().collect();
    let crown_upper: Vec<f32> = crown.upper().iter().copied().collect();
    let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
    let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();
    for (index, (got, expected)) in crown_lower.iter().zip(ibp_lower.iter()).enumerate() {
        assert_close(*got, *expected, 1e-5, format!("div crown lower[{index}]"));
    }
    for (index, (got, expected)) in crown_upper.iter().zip(ibp_upper.iter()).enumerate() {
        assert_close(*got, *expected, 1e-5, format!("div crown upper[{index}]"));
    }
}

#[test]
fn test_div_constant_batched_crown_per_channel_broadcast_3896() {
    let layer = DivConstantLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 4.0]).unwrap(),
        vec![2, 3],
    );
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();
    let result = layer
        .propagate_linear_batched(&bounds)
        .expect("per-channel DivConstant batched CROWN backward should succeed");

    for idx in 0..3 {
        assert_close(
            result.lower_a[[0, idx, idx]],
            0.5,
            1e-6,
            format!("div batched lower_a[0,{idx},{idx}]"),
        );
        assert_close(
            result.upper_a[[0, idx, idx]],
            0.5,
            1e-6,
            format!("div batched upper_a[0,{idx},{idx}]"),
        );
        assert_close(
            result.lower_a[[1, idx, idx]],
            0.25,
            1e-6,
            format!("div batched lower_a[1,{idx},{idx}]"),
        );
        assert_close(
            result.upper_a[[1, idx, idx]],
            0.25,
            1e-6,
            format!("div batched upper_a[1,{idx},{idx}]"),
        );
    }
}

#[test]
fn test_div_constant_batched_crown_rejects_broadcast_expansion_3896() {
    let layer = DivConstantLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![2.0, 4.0, 8.0, 16.0]).unwrap(),
        vec![1],
    );
    let bounds = BatchedLinearBounds::identity(&[4]).unwrap();
    let err = layer
        .propagate_linear_batched(&bounds)
        .expect_err("broadcast expansion should fall back instead of scaling batched CROWN");
    assert!(
        matches!(err, NyError::UnsupportedOp(ref message) if message.contains("broadcast expansion/reduction")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn test_div_constant_batched_crown_rejects_near_zero() {
    let err = DivConstantLayer::try_scalar(1e-12).expect_err("near-zero divisor should fail");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

// ═══════════════════════════════════════════════════════════════════════
// AbsLayer tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_abs_ibp_all_positive() {
    // x ∈ [1, 5] => |x| ∈ [1, 5]
    let layer = AbsLayer::new();
    let input = BoundedTensor::new(array![1.0].into_dyn(), array![5.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert!(
        (result.lower()[0] - 1.0).abs() < 1e-6,
        "abs(positive) lower should be 1.0, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 5.0).abs() < 1e-6,
        "abs(positive) upper should be 5.0, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_abs_ibp_all_negative() {
    // x ∈ [-5, -1] => |x| ∈ [1, 5]
    let layer = AbsLayer::new();
    let input = BoundedTensor::new(array![-5.0].into_dyn(), array![-1.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert!(
        (result.lower()[0] - 1.0).abs() < 1e-6,
        "abs(negative) lower should be 1.0, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 5.0).abs() < 1e-6,
        "abs(negative) upper should be 5.0, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_abs_ibp_crossing_zero() {
    // x ∈ [-3, 5] => |x| ∈ [0, 5]
    let layer = AbsLayer::new();
    let input = BoundedTensor::new(array![-3.0].into_dyn(), array![5.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert_eq!(result.lower()[0], 0.0);
    assert!(
        (result.upper()[0] - 5.0).abs() < 1e-6,
        "abs(crossing) upper should be 5.0, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_abs_ibp_crossing_negative_dominates() {
    // x ∈ [-7, 2] => |x| ∈ [0, 7]
    let layer = AbsLayer::new();
    let input = BoundedTensor::new(array![-7.0].into_dyn(), array![2.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert_eq!(result.lower()[0], 0.0);
    assert!(
        (result.upper()[0] - 7.0).abs() < 1e-6,
        "abs(neg_dominates) upper should be 7.0, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_abs_ibp_multi_element() {
    // Test all three cases simultaneously
    let layer = AbsLayer::new();
    let input = BoundedTensor::new(
        array![1.0, -5.0, -3.0].into_dyn(),
        array![3.0, -1.0, 2.0].into_dyn(),
    )
    .unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    // [1,3] positive: [1,3]
    assert!(
        (result.lower()[0] - 1.0).abs() < 1e-6,
        "abs multi[0] lower should be 1.0, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 3.0).abs() < 1e-6,
        "abs multi[0] upper should be 3.0, got {}",
        result.upper()[0]
    );
    // [-5,-1] negative: [1,5]
    assert!(
        (result.lower()[1] - 1.0).abs() < 1e-6,
        "abs multi[1] lower should be 1.0, got {}",
        result.lower()[1]
    );
    assert!(
        (result.upper()[1] - 5.0).abs() < 1e-6,
        "abs multi[1] upper should be 5.0, got {}",
        result.upper()[1]
    );
    // [-3,2] crossing: [0,3]
    assert_eq!(result.lower()[2], 0.0);
    assert!(
        (result.upper()[2] - 3.0).abs() < 1e-6,
        "abs multi[2] upper should be 3.0, got {}",
        result.upper()[2]
    );
}

fn assert_abs_envelope(l: f32, u: f32) {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = abs_linear_relaxation(l, u);
    let samples = 100;
    for k in 0..=samples {
        let t = k as f32 / samples as f32;
        let x = l + (u - l) * t;
        let fx = x.abs();
        let lower = ls * x + li;
        let upper = us * x + ui;
        let tol = (1e-5_f32).max(1e-4 * fx);
        assert!(
            lower <= fx + tol,
            "Abs lower envelope violated at [{l}, {u}], x={x}: {lower} > {fx}"
        );
        assert!(
            upper + tol >= fx,
            "Abs upper envelope violated at [{l}, {u}], x={x}: {upper} < {fx}"
        );
    }
}

#[test]
fn test_abs_relaxation_envelope() {
    let intervals = [
        (0.0, 1.0),
        (-1.0, 0.0),
        (1.0, 5.0),
        (-5.0, -1.0),
        (-3.0, 5.0),
        (-7.0, 2.0),
        (-1e-3, 1e-3),
        (0.0, 0.0),
        (-1.0, 1.0),
    ];
    for (l, u) in intervals {
        assert_abs_envelope(l, u);
    }
}

#[test]
fn test_abs_relaxation_non_finite() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = abs_linear_relaxation(f32::NAN, 1.0);
    assert_eq!(ls, 0.0);
    assert_eq!(us, 0.0);
    assert!(
        li.is_infinite() && li.is_sign_negative(),
        "NaN input: lower intercept should be -inf, got {li}"
    );
    assert!(
        ui.is_infinite() && ui.is_sign_positive(),
        "NaN input: upper intercept should be +inf, got {ui}"
    );

    let r2 = abs_linear_relaxation(0.0, f32::INFINITY);
    assert_eq!(r2.lower_slope, 0.0);
    assert_eq!(r2.upper_slope, 0.0);
    assert!(
        r2.lower_intercept.is_infinite() && r2.lower_intercept.is_sign_negative(),
        "inf input: lower intercept should be -inf, got {}",
        r2.lower_intercept
    );
    assert!(
        r2.upper_intercept.is_infinite() && r2.upper_intercept.is_sign_positive(),
        "inf input: upper intercept should be +inf, got {}",
        r2.upper_intercept
    );
}

#[test]
fn test_abs_crown_backward_soundness() {
    let layer = AbsLayer::new();
    // Test crossing interval: x ∈ [-2, 3]
    let pre = BoundedTensor::new(array![-2.0].into_dyn(), array![3.0].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = -2.0 + 5.0 * (k as f32 / 50.0);
        let y = x.abs();
        let tol = (1e-4_f32).max(1e-3 * y);
        assert!(
            la * x + lb <= y + tol,
            "Abs CROWN lb violated at x={x}: {} > {y}",
            la * x + lb
        );
        assert!(
            ua * x + ub >= y - tol,
            "Abs CROWN ub violated at x={x}: {} < {y}",
            ua * x + ub
        );
    }
}

#[test]
fn test_abs_crown_backward_all_positive() {
    let layer = AbsLayer::new();
    let pre = BoundedTensor::new(array![1.0].into_dyn(), array![5.0].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    // All positive: |x| = x, so slope = 1, intercept = 0
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < 1e-6,
        "abs(positive) lower_a should be 1.0, got {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_b[0]).abs() < 1e-6,
        "abs(positive) lower_b should be 0.0, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.upper_a[[0, 0]] - 1.0).abs() < 1e-6,
        "abs(positive) upper_a should be 1.0, got {}",
        result.upper_a[[0, 0]]
    );
    assert!(
        (result.upper_b[0]).abs() < 1e-6,
        "abs(positive) upper_b should be 0.0, got {}",
        result.upper_b[0]
    );
}

#[test]
fn test_abs_crown_backward_all_negative() {
    let layer = AbsLayer::new();
    let pre = BoundedTensor::new(array![-5.0].into_dyn(), array![-1.0].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    // All negative: |x| = -x, so slope = -1, intercept = 0
    assert!(
        (result.lower_a[[0, 0]] - (-1.0)).abs() < 1e-6,
        "abs(negative) lower_a should be -1.0, got {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_b[0]).abs() < 1e-6,
        "abs(negative) lower_b should be 0.0, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.upper_a[[0, 0]] - (-1.0)).abs() < 1e-6,
        "abs(negative) upper_a should be -1.0, got {}",
        result.upper_a[[0, 0]]
    );
    assert!(
        (result.upper_b[0]).abs() < 1e-6,
        "abs(negative) upper_b should be 0.0, got {}",
        result.upper_b[0]
    );
}

#[test]
fn test_abs_propagate_linear_without_bounds_errors() {
    let layer = AbsLayer::new();
    let bounds = LinearBounds::identity(1);
    assert!(
        layer.propagate_linear(&bounds).is_err(),
        "Abs CROWN should require pre-activation bounds"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// PowConstantLayer tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_pow_ibp_square_all_positive() {
    // y = x^2, x ∈ [2, 4] => y ∈ [4, 16]
    let layer = PowConstantLayer::square();
    let input = BoundedTensor::new(array![2.0].into_dyn(), array![4.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert!(
        (result.lower()[0] - 4.0).abs() < 1e-5,
        "x^2 positive lower should be 4.0, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 16.0).abs() < 1e-5,
        "x^2 positive upper should be 16.0, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_pow_ibp_square_all_negative() {
    // y = x^2, x ∈ [-4, -2] => y ∈ [4, 16]
    let layer = PowConstantLayer::square();
    let input = BoundedTensor::new(array![-4.0].into_dyn(), array![-2.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert!(
        (result.lower()[0] - 4.0).abs() < 1e-5,
        "x^2 negative lower should be 4.0, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 16.0).abs() < 1e-5,
        "x^2 negative upper should be 16.0, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_pow_ibp_square_crossing_zero() {
    // y = x^2, x ∈ [-3, 5] => y ∈ [0, 25] (min at 0)
    let layer = PowConstantLayer::square();
    let input = BoundedTensor::new(array![-3.0].into_dyn(), array![5.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert_eq!(result.lower()[0], 0.0);
    assert!(
        (result.upper()[0] - 25.0).abs() < 1e-5,
        "x^2 crossing upper should be 25.0, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_pow_ibp_cube_monotonic() {
    // y = x^3, x ∈ [-2, 3] => y ∈ [-8, 27] (monotonically increasing)
    let layer = PowConstantLayer::new(3.0);
    let input = BoundedTensor::new(array![-2.0].into_dyn(), array![3.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert!(
        (result.lower()[0] - (-8.0)).abs() < 1e-4,
        "x^3 lower should be -8.0, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 27.0).abs() < 1e-4,
        "x^3 upper should be 27.0, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_pow_ibp_negative_exponent_rejects_zero() {
    // y = x^(-1) with interval containing zero should fail
    let layer = PowConstantLayer::new(-1.0);
    let input = BoundedTensor::new(array![-1.0].into_dyn(), array![1.0].into_dyn()).unwrap();
    assert!(
        layer.propagate_ibp(&input).is_err(),
        "x^(-1) with zero-crossing interval should return error"
    );
}

#[test]
fn test_pow_ibp_negative_exponent_positive_interval() {
    // y = x^(-2), x ∈ [2, 4] => y ∈ [1/16, 1/4]
    let layer = PowConstantLayer::new(-2.0);
    let input = BoundedTensor::new(array![2.0].into_dyn(), array![4.0].into_dyn()).unwrap();
    let result = layer.propagate_ibp(&input).unwrap();
    assert!(
        (result.lower()[0] - 0.0625).abs() < 1e-5,
        "x^(-2) lower should be 0.0625, got {}",
        result.lower()[0]
    );
    assert!(
        (result.upper()[0] - 0.25).abs() < 1e-5,
        "x^(-2) upper should be 0.25, got {}",
        result.upper()[0]
    );
}

#[test]
fn test_pow_ibp_non_integer_rejects_negative() {
    // y = x^0.5 with negative inputs should fail
    let layer = PowConstantLayer::new(0.5);
    let input = BoundedTensor::new(array![-1.0].into_dyn(), array![1.0].into_dyn()).unwrap();
    assert!(
        layer.propagate_ibp(&input).is_err(),
        "x^0.5 with negative inputs should return error"
    );
}

fn assert_pow2_envelope(l: f32, u: f32) {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = pow2_linear_relaxation(l, u);
    let samples = 100;
    for k in 0..=samples {
        let t = k as f32 / samples as f32;
        let x = l + (u - l) * t;
        let fx = x * x;
        let lower = ls * x + li;
        let upper = us * x + ui;
        let tol = (1e-4_f32).max(1e-3 * fx);
        assert!(
            lower <= fx + tol,
            "Pow2 lower envelope violated at [{l}, {u}], x={x}: {lower} > {fx}"
        );
        assert!(
            upper + tol >= fx,
            "Pow2 upper envelope violated at [{l}, {u}], x={x}: {upper} < {fx}"
        );
    }
}

#[test]
fn test_pow2_relaxation_envelope() {
    let intervals = [
        (0.0, 1.0),
        (-1.0, 0.0),
        (1.0, 4.0),
        (-4.0, -1.0),
        (-3.0, 5.0),
        (-1.0, 1.0),
        (-1e-3, 1e-3),
        (2.0, 2.0),
        (0.0, 0.0),
    ];
    for (l, u) in intervals {
        assert_pow2_envelope(l, u);
    }
}

#[test]
fn test_pow2_relaxation_non_finite() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = pow2_linear_relaxation(f32::NAN, 1.0);
    assert_eq!(ls, 0.0);
    assert_eq!(us, 0.0);
    assert!(
        li.is_infinite() && li.is_sign_negative(),
        "pow2 NaN input: lower intercept should be -inf, got {li}"
    );
    assert!(
        ui.is_infinite() && ui.is_sign_positive(),
        "pow2 NaN input: upper intercept should be +inf, got {ui}"
    );
}

#[test]
fn test_pow_crown_backward_cube_nonnegative_soundness_4354() {
    let layer = PowConstantLayer::new(3.0);
    let pre = BoundedTensor::new(array![0.0].into_dyn(), array![2.0].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    for k in 0..=50 {
        let x = 2.0 * (k as f32 / 50.0);
        let y = x * x * x;
        let tol = (1e-4_f32).max(1e-3 * y.abs());
        assert!(
            result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + tol,
            "Pow3 CROWN lb violated at x={x}: {} > {y}",
            result.lower_a[[0, 0]] * x + result.lower_b[0]
        );
        assert!(
            result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - tol,
            "Pow3 CROWN ub violated at x={x}: {} < {y}",
            result.upper_a[[0, 0]] * x + result.upper_b[0]
        );
    }
}

#[test]
fn test_pow_crown_backward_cube_mixed_sign_still_errors_4354() {
    let layer = PowConstantLayer::new(3.0);
    let pre = BoundedTensor::new(array![-1.0].into_dyn(), array![2.0].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    assert!(
        layer.propagate_linear_with_bounds(&bounds, &pre).is_err(),
        "mixed-sign cubic intervals should remain unsupported"
    );
}

#[test]
fn test_pow_crown_backward_square_soundness() {
    let layer = PowConstantLayer::square();
    // x ∈ [-2, 3], x^2 is convex
    let pre = BoundedTensor::new(array![-2.0].into_dyn(), array![3.0].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = -2.0 + 5.0 * (k as f32 / 50.0);
        let y = x * x;
        let tol = (1e-4_f32).max(1e-3 * y);
        assert!(
            la * x + lb <= y + tol,
            "Pow2 CROWN lb violated at x={x}: {} > {y}",
            la * x + lb
        );
        assert!(
            ua * x + ub >= y - tol,
            "Pow2 CROWN ub violated at x={x}: {} < {y}",
            ua * x + ub
        );
    }
}

#[test]
fn test_pow_crown_backward_square_positive_interval() {
    let layer = PowConstantLayer::square();
    let pre = BoundedTensor::new(array![1.0].into_dyn(), array![4.0].into_dyn()).unwrap();
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

    for k in 0..=50 {
        let x = 1.0 + 3.0 * (k as f32 / 50.0);
        let y = x * x;
        let tol = (1e-4_f32).max(1e-3 * y);
        assert!(
            result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + tol,
            "Pow2 CROWN lb violated at x={x}"
        );
        assert!(
            result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - tol,
            "Pow2 CROWN ub violated at x={x}"
        );
    }
}

#[test]
fn test_pow_propagate_linear_without_bounds_errors() {
    let layer = PowConstantLayer::square();
    let bounds = LinearBounds::identity(1);
    assert!(
        layer.propagate_linear(&bounds).is_err(),
        "Pow CROWN should require pre-activation bounds"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// common.rs helper tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_extract_scalar_constant_scalar() {
    // A 0-d array with a single element should return that element.
    let arr = ArrayD::from_elem(IxDyn(&[]), 7.25_f32);
    let val = super::common::extract_scalar_constant_for_batched(&arr, "Test").unwrap();
    assert!(
        (val - 7.25).abs() < 1e-7,
        "scalar extract should return 7.25, got {val}"
    );
}

#[test]
fn test_extract_scalar_constant_1d_single() {
    // A 1-d array with shape [1] has len 1 — should succeed.
    let arr = ArrayD::from_elem(IxDyn(&[1]), -2.5_f32);
    let val = super::common::extract_scalar_constant_for_batched(&arr, "Test").unwrap();
    assert!(
        (val - (-2.5)).abs() < 1e-7,
        "1d single extract should return -2.5, got {val}"
    );
}

#[test]
fn test_extract_scalar_constant_rejects_multi_element() {
    // A 1-d array with shape [2] has len 2 — must be rejected.
    let arr = ArrayD::from_elem(IxDyn(&[2]), 1.0_f32);
    let err = super::common::extract_scalar_constant_for_batched(&arr, "FooLayer");
    assert!(err.is_err(), "multi-element array should be rejected");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("FooLayer") && msg.contains("scalar"),
        "Error should mention layer name and scalar constraint: {msg}"
    );
}

#[test]
fn test_extract_scalar_constant_rejects_2d() {
    // A 2×2 array has len 4 — must be rejected.
    let arr = ArrayD::from_elem(IxDyn(&[2, 2]), 0.0_f32);
    assert!(
        super::common::extract_scalar_constant_for_batched(&arr, "Bar").is_err(),
        "2d array should be rejected as non-scalar"
    );
}

#[test]
fn test_div_constant_eps_is_positive_and_small() {
    // DIV_CONSTANT_EPS should be a small positive guard value.
    // Bind to a runtime variable to avoid clippy::assertions_on_constants.
    let eps = super::common::DIV_CONSTANT_EPS;
    assert!(eps > 0.0, "DIV_CONSTANT_EPS must be positive, got {eps}");
    assert!(eps < 1e-6, "DIV_CONSTANT_EPS must be small, got {eps}");
}

// ── Strict zero-tolerance CROWN relaxation proptest (#3292) ──────────────
//
// Pattern from #3285: f64-evaluated reference with zero tolerance catches
// f32 cancellation bugs invisible to magnitude-scaled tolerance tests.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Strict soundness proptest for sqrt CROWN relaxation.
    /// Uses f64 reference (sqrt_f64_reference) with zero tolerance on 200-point grid.
    /// sqrt is concave on [0, +inf): lower bound = chord, upper bound = tangent.
    /// Domain restricted to non-negative l (sqrt undefined for x < 0).
    /// Ref: alpha-beta-CROWN auto_LiRPA sqrt relaxation, #3292.
    #[test]
    fn proptest_sqrt_relaxation_strict_soundness(
        l in 0.0f32..10.0,
        width in 0.01f32..20.0,
    ) {
        let u = l + width;
        let relax = sqrt_linear_relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = sqrt_f64_reference(x);

            let lower_val = ls as f64 * x + li as f64;
            prop_assert!(
                lower_val <= fx,
                "sqrt lower bound UNSOUND at x={x}: {lower_val} > sqrt({x})={fx}, \
                 interval=[{l}, {u}], gap={}", lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            prop_assert!(
                upper_val >= fx,
                "sqrt upper bound UNSOUND at x={x}: {upper_val} < sqrt({x})={fx}, \
                 interval=[{l}, {u}], gap={}", fx - upper_val
            );
        }
    }
}

// ===== SubConstant broadcast CROWN backward (#ml4acopf-genbab) =====
//
// The ml4acopf trigonometric threshold banks contain
// `Sub(x[1,P,1], thresholds[K]) -> [1,P,K]`: the variable input expands along
// the last axis while the constant broadcasts across the leading axes. The
// CROWN backward previously hard-errored with
// `ShapeMismatch { expected: [P*K], got: [K] }`, killing every GenBaB child
// propagation on the ml4acopf_2024 benchmark.

fn broadcast_c3_layer(reverse: bool) -> SubConstantLayer {
    let c = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
    if reverse {
        SubConstantLayer::new_reverse(c)
    } else {
        SubConstantLayer::new(c)
    }
}

fn broadcast_pre_x21() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 2, 1]), -1.0),
        ArrayD::from_elem(IxDyn(&[1, 2, 1]), 1.0),
    )
    .unwrap()
}

#[test]
fn test_subconstant_broadcast_crown_backward_reduces_columns_ml4acopf() {
    // y = broadcast(x[1,2,1]) - broadcast(c[3]) -> [1,2,3] (flat 6).
    let layer = broadcast_c3_layer(false);
    let a = Array2::from_shape_vec((1, 6), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let bounds =
        LinearBounds::new(a.clone(), ndarray::array![0.5], a, ndarray::array![0.5]).unwrap();
    let out = layer
        .propagate_crown_backward(&bounds, Some(&broadcast_pre_x21()))
        .expect("broadcast CROWN backward must succeed");
    // Column reduction: input col 0 <- outputs {0,1,2}: 1+2+3 = 6;
    // input col 1 <- outputs {3,4,5}: 4+5+6 = 15.
    assert_eq!(out.num_inputs(), 2);
    assert_close(out.lower_a()[[0, 0]], 6.0, 1e-5, "reduced lower_a col 0");
    assert_close(out.lower_a()[[0, 1]], 15.0, 1e-5, "reduced lower_a col 1");
    assert_close(out.upper_a()[[0, 0]], 6.0, 1e-5, "reduced upper_a col 0");
    // Bias: 0.5 - A @ bcast(c); bcast(c) = [10,20,30,10,20,30] so
    // A @ c = 10+40+90+40+100+180 = 460.
    assert_close(out.lower_b()[0], 0.5 - 460.0, 1e-2, "reduced lower_b");
    assert_close(out.upper_b()[0], 0.5 - 460.0, 1e-2, "reduced upper_b");
    // fan_in = 3 >= 2: certified scatter-add coefficient error must be attached.
    assert!(
        out.lower_a_err().is_some() && out.upper_a_err().is_some(),
        "certified coefficient error channel must be attached for fan_in >= 2"
    );
}

#[test]
fn test_subconstant_broadcast_crown_backward_reverse_negates() {
    // y = broadcast(c[3]) - broadcast(x[1,2,1]): A negates, bias adds A @ c.
    let layer = broadcast_c3_layer(true);
    let a = Array2::from_shape_vec((1, 6), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let bounds =
        LinearBounds::new(a.clone(), ndarray::array![0.5], a, ndarray::array![0.5]).unwrap();
    let out = layer
        .propagate_crown_backward(&bounds, Some(&broadcast_pre_x21()))
        .expect("reverse broadcast CROWN backward must succeed");
    assert_eq!(out.num_inputs(), 2);
    assert_close(out.lower_a()[[0, 0]], -6.0, 1e-5, "reverse lower_a col 0");
    assert_close(out.lower_a()[[0, 1]], -15.0, 1e-5, "reverse lower_a col 1");
    assert_close(out.lower_b()[0], 0.5 + 460.0, 1e-2, "reverse lower_b");
    assert_close(out.upper_b()[0], 0.5 + 460.0, 1e-2, "reverse upper_b");
}

#[test]
fn test_subconstant_broadcast_crown_backward_encloses_affine_truth() {
    // y = broadcast(x) - broadcast(c) is affine in x, so the reduced linear
    // bounds evaluated at every input-box corner must enclose the exact
    // A @ y(x) + b (directed rounding makes them outer, never inner).
    let layer = broadcast_c3_layer(false);
    let a = Array2::from_shape_vec(
        (2, 6),
        vec![
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, //
            0.5, 1.5, -2.5, 3.5, -4.5, 5.5,
        ],
    )
    .unwrap();
    let bias = ndarray::array![0.25, -0.75];
    let bounds = LinearBounds::new(a.clone(), bias.clone(), a.clone(), bias.clone()).unwrap();
    let lo = [-1.5f32, 0.5];
    let hi = [2.0f32, 1.0];
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), lo.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), hi.to_vec()).unwrap(),
    )
    .unwrap();
    let out = layer
        .propagate_crown_backward(&bounds, Some(&pre))
        .expect("broadcast CROWN backward must succeed");
    let c = [10.0f64, 20.0, 30.0];
    for corner in 0..4u32 {
        let x = [
            if corner & 1 == 0 { lo[0] } else { hi[0] } as f64,
            if corner & 2 == 0 { lo[1] } else { hi[1] } as f64,
        ];
        for row in 0..2 {
            let mut truth = bias[row] as f64;
            for i in 0..2 {
                for j in 0..3 {
                    truth += a[[row, i * 3 + j]] as f64 * (x[i] - c[j]);
                }
            }
            let mut lo_eval = out.lower_b()[row] as f64;
            let mut hi_eval = out.upper_b()[row] as f64;
            for i in 0..2 {
                lo_eval += out.lower_a()[[row, i]] as f64 * x[i];
                hi_eval += out.upper_a()[[row, i]] as f64 * x[i];
            }
            assert!(
                lo_eval <= truth + 1e-3 && truth - 1e-3 <= hi_eval,
                "corner {corner} row {row}: [{lo_eval}, {hi_eval}] must enclose {truth}"
            );
        }
    }
}

#[test]
fn test_subconstant_broadcast_crown_backward_ml4acopf_shape_regression() {
    // Exact failing configuration from ml4acopf GenBaB child propagation:
    // Sub(x[1,20,1], thresholds[18]) -> [1,20,18]; incoming A is [1, 360].
    // Previously: ShapeMismatch { expected: [360], got: [18] } ->
    // "GenBaB child propagation failed" on every child, BaB abandoned.
    let c = ArrayD::from_shape_fn(IxDyn(&[18]), |i| i[0] as f32 * 0.1);
    let layer = SubConstantLayer::new(c);
    let a = Array2::from_elem((1, 360), 0.25);
    let bounds =
        LinearBounds::new(a.clone(), ndarray::array![0.0], a, ndarray::array![0.0]).unwrap();
    let pre = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 20, 1]), -0.5),
        ArrayD::from_elem(IxDyn(&[1, 20, 1]), 0.5),
    )
    .unwrap();
    let out = layer
        .propagate_crown_backward(&bounds, Some(&pre))
        .expect("ml4acopf threshold-bank pattern must propagate");
    assert_eq!(out.num_inputs(), 20);
    // Each input column sums 18 coefficients of 0.25 = 4.5.
    for i in 0..20 {
        assert_close(out.lower_a()[[0, i]], 4.5, 1e-4, format!("col {i}"));
    }
}

#[test]
fn test_subconstant_crown_backward_elementwise_delegates_unchanged() {
    // Non-broadcast case must stay byte-identical to propagate_linear.
    let c = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, -2.0, 3.0, -4.0]).unwrap();
    let layer = SubConstantLayer::new(c);
    let a = Array2::from_shape_vec((2, 4), (0..8).map(|v| v as f32 - 3.5).collect()).unwrap();
    let bias = ndarray::array![0.5, -0.5];
    let bounds = LinearBounds::new(a.clone(), bias.clone(), a, bias).unwrap();
    let pre = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4]), -1.0),
        ArrayD::from_elem(IxDyn(&[4]), 1.0),
    )
    .unwrap();
    let via_crown = layer.propagate_crown_backward(&bounds, Some(&pre)).unwrap();
    let via_linear = layer.propagate_linear(&bounds).unwrap().into_owned();
    assert_eq!(via_crown.lower_a(), via_linear.lower_a());
    assert_eq!(via_crown.upper_a(), via_linear.upper_a());
    assert_eq!(via_crown.lower_b(), via_linear.lower_b());
    assert_eq!(via_crown.upper_b(), via_linear.upper_b());
}
