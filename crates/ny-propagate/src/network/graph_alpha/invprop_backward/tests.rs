// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::augment_bounds_with_constraints;
use super::best_bounds::take_best_linear_bounds;
use super::take_best_bounds;
use crate::bounds::LinearBounds;
use crate::invprop::{InvpropConfig, OutputConstraints};
use ndarray::{arr1, arr2, ArrayD};
use ny_tensor::BoundedTensor;

/// Check for infeasibility in bounds (lb > ub).
///
/// Returns true if any dimension has lower bound > upper bound or NaN,
/// indicating the constraint region is infeasible.
fn check_infeasibility(bounds: &BoundedTensor) -> bool {
    let flat = bounds.flatten();
    flat.lower()
        .iter()
        .zip(flat.upper().iter())
        .any(|(&l, &u)| l.is_nan() || u.is_nan() || l > u)
}

/// Mark bounds as infeasible using the canonical `(+inf, -inf)` sentinel.
fn mark_bounds_infeasible(bounds: &mut BoundedTensor) {
    bounds.mark_infeasible_all();
}

struct InvpropBackwardContext<'a> {
    config: &'a InvpropConfig,
}

impl<'a> InvpropBackwardContext<'a> {
    fn new(config: &'a InvpropConfig) -> Self {
        Self { config }
    }

    fn should_apply_to(&self, layer_name: &str, layer_type: &str) -> bool {
        self.config.should_apply_to(layer_name, layer_type)
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_take_best_bounds() {
    let lower_oc = ArrayD::from_shape_vec(vec![3], vec![0.1, 0.5, 0.3]).unwrap();
    let upper_oc = ArrayD::from_shape_vec(vec![3], vec![0.9, 0.8, 0.7]).unwrap();
    let bounds_oc = BoundedTensor::new(lower_oc, upper_oc).unwrap();

    let lower_no_oc = ArrayD::from_shape_vec(vec![3], vec![0.2, 0.4, 0.4]).unwrap();
    let upper_no_oc = ArrayD::from_shape_vec(vec![3], vec![0.85, 0.9, 0.65]).unwrap();
    let bounds_no_oc = BoundedTensor::new(lower_no_oc, upper_no_oc).unwrap();

    let best = take_best_bounds(&bounds_oc, &bounds_no_oc);

    assert!((best.lower()[[0]] - 0.2).abs() < 1e-6);
    assert!((best.lower()[[1]] - 0.5).abs() < 1e-6);
    assert!((best.lower()[[2]] - 0.4).abs() < 1e-6);

    assert!((best.upper()[[0]] - 0.85).abs() < 1e-6);
    assert!((best.upper()[[1]] - 0.8).abs() < 1e-6);
    assert!((best.upper()[[2]] - 0.65).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_take_best_bounds_shape_mismatch_returns_original() {
    let lower_oc = ArrayD::from_shape_vec(vec![2], vec![0.1, 0.5]).unwrap();
    let upper_oc = ArrayD::from_shape_vec(vec![2], vec![0.9, 0.8]).unwrap();
    let bounds_oc = BoundedTensor::new(lower_oc, upper_oc).unwrap();

    let lower_no_oc = ArrayD::from_shape_vec(vec![3], vec![0.2, 0.4, 0.4]).unwrap();
    let upper_no_oc = ArrayD::from_shape_vec(vec![3], vec![0.85, 0.9, 0.65]).unwrap();
    let bounds_no_oc = BoundedTensor::new(lower_no_oc, upper_no_oc).unwrap();

    let best = take_best_bounds(&bounds_oc, &bounds_no_oc);

    assert_eq!(best.lower(), bounds_oc.lower());
    assert_eq!(best.upper(), bounds_oc.upper());
}

/// Regression test for #1938 + #2655: merge of non-overlapping intervals should widen
/// to [-inf, +inf] rather than clamping to a degenerate point interval.
/// Element 1 here has oc=[0.6, 0.8] and no_oc=[0.4, 0.5], so merge produces
/// lower=0.6, upper=0.5 (inverted). `clamp_inverted_best_bounds` widens to [-inf, +inf].
#[ntest::timeout(10000)]
#[test]
fn test_take_best_bounds_crossing_intervals_widened_2655() {
    let lower_oc = ArrayD::from_shape_vec(vec![2], vec![0.1, 0.6]).unwrap();
    let upper_oc = ArrayD::from_shape_vec(vec![2], vec![0.9, 0.8]).unwrap();
    let bounds_oc = BoundedTensor::new(lower_oc, upper_oc).unwrap();

    let lower_no_oc = ArrayD::from_shape_vec(vec![2], vec![0.2, 0.4]).unwrap();
    let upper_no_oc = ArrayD::from_shape_vec(vec![2], vec![0.85, 0.5]).unwrap();
    let bounds_no_oc = BoundedTensor::new(lower_no_oc, upper_no_oc).unwrap();

    let best = take_best_bounds(&bounds_oc, &bounds_no_oc);

    assert!((best.lower()[[0]] - 0.2).abs() < 1e-6);
    assert!((best.upper()[[0]] - 0.85).abs() < 1e-6);

    assert!(
        best.lower()[[1]] <= best.upper()[[1]],
        "merge inversion must be widened: lower={}, upper={}",
        best.lower()[[1]],
        best.upper()[[1]]
    );
    assert_eq!(best.lower()[[1]], f32::NEG_INFINITY);
    assert_eq!(best.upper()[[1]], f32::INFINITY);
}

/// Regression test for #3093: NaN in `bounds_with_oc` should not be sticky.
/// IEEE 754 makes `other > NaN` false, so without explicit NaN guards the
/// NaN would persist even when `bounds_without_oc` has a finite value.
#[ntest::timeout(10000)]
#[test]
fn test_take_best_bounds_nan_replaced_by_finite_3093() {
    let lower_oc = ArrayD::from_shape_vec(vec![2], vec![0.3, f32::NAN]).unwrap();
    let upper_oc = ArrayD::from_shape_vec(vec![2], vec![f32::NAN, 0.8]).unwrap();
    let bounds_oc = BoundedTensor::new_unchecked(lower_oc, upper_oc).unwrap();

    let lower_no_oc = ArrayD::from_shape_vec(vec![2], vec![0.2, 0.4]).unwrap();
    let upper_no_oc = ArrayD::from_shape_vec(vec![2], vec![0.9, 0.7]).unwrap();
    let bounds_no_oc = BoundedTensor::new(lower_no_oc, upper_no_oc).unwrap();

    let best = take_best_bounds(&bounds_oc, &bounds_no_oc);

    assert!((best.lower()[[0]] - 0.3).abs() < 1e-6);
    assert!(
        (best.upper()[[0]] - 0.9).abs() < 1e-6,
        "NaN upper should be replaced by finite value, got {}",
        best.upper()[[0]]
    );

    assert!(
        (best.lower()[[1]] - 0.4).abs() < 1e-6,
        "NaN lower should be replaced by finite value, got {}",
        best.lower()[[1]]
    );
    assert!((best.upper()[[1]] - 0.7).abs() < 1e-6);

    assert!(
        best.lower().iter().all(|v| !v.is_nan()),
        "No NaN should remain in lower bounds"
    );
    assert!(
        best.upper().iter().all(|v| !v.is_nan()),
        "No NaN should remain in upper bounds"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_check_infeasibility() {
    let lower = ArrayD::from_shape_vec(vec![3], vec![0.1, 0.2, 0.3]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![3], vec![0.5, 0.6, 0.7]).unwrap();
    let feasible = BoundedTensor::new(lower, upper).unwrap();
    assert!(!check_infeasibility(&feasible));

    let lower_inf = ArrayD::from_shape_vec(vec![3], vec![0.1, 0.7, 0.3]).unwrap();
    let upper_inf = ArrayD::from_shape_vec(vec![3], vec![0.5, 0.6, 0.7]).unwrap();
    let infeasible = BoundedTensor::new_unchecked(lower_inf, upper_inf).unwrap();
    assert!(check_infeasibility(&infeasible));

    let lower_nan = ArrayD::from_shape_vec(vec![2], vec![0.1, f32::NAN]).unwrap();
    let upper_nan = ArrayD::from_shape_vec(vec![2], vec![0.5, 0.6]).unwrap();
    let nan_bounds = BoundedTensor::new_unchecked(lower_nan, upper_nan).unwrap();
    assert!(check_infeasibility(&nan_bounds));
}

/// Stage-0 oracle: the corrected assume-violation Lagrangian at the output seed.
///
/// VIOLATION semantics (`C y <= rhs`), so the sound fold is:
///   lower row i:  a += C^T gamma_l ,  b -= gamma_l . rhs
///   upper row i:  a -= C^T gamma_u ,  b += gamma_u . rhs
///
/// This FAILS on the pre-fix code, which dropped the `C^T gamma` A-matrix term
/// entirely and folded the bias with the INVERTED sign (`b += gamma.rhs`).
#[ntest::timeout(10000)]
#[test]
fn test_augment_seed_corrected_lagrangian() {
    let bounds = LinearBounds::identity(2);

    // Single constraint touching output coord 0: y0 <= 1 (violation region).
    let a_matrix = arr2(&[[1.0, 0.0]]);
    let rhs = arr1(&[1.0]);
    let constraints = OutputConstraints::new(a_matrix, rhs, true).unwrap();

    let gammas_lower = arr2(&[[0.5, 0.5]]);
    let gammas_upper = arr2(&[[0.5, 0.5]]);

    let augmented =
        augment_bounds_with_constraints(&bounds, &constraints, &gammas_lower, &gammas_upper);

    // Bias: corrected sign is lower -= gamma.rhs = -0.5 (pre-fix gave +0.5).
    assert!(
        (augmented.lower_b[0] - (-0.5)).abs() < 1e-6,
        "lower_b[0] = {}",
        augmented.lower_b[0]
    );
    assert!((augmented.lower_b[1] - (-0.5)).abs() < 1e-6);
    // Upper: b += gamma.rhs = +0.5.
    assert!((augmented.upper_b[0] - 0.5).abs() < 1e-6);
    assert!((augmented.upper_b[1] - 0.5).abs() < 1e-6);

    // A-matrix term MUST be folded (pre-fix left A = I). Row i, col 0 gets +/-gamma.
    assert!(
        (augmented.lower_a[[0, 0]] - 1.5).abs() < 1e-6,
        "lower_a[0,0] = {} (expected I + gamma = 1.5)",
        augmented.lower_a[[0, 0]]
    );
    assert!((augmented.lower_a[[1, 0]] - 0.5).abs() < 1e-6);
    assert!((augmented.upper_a[[0, 0]] - 0.5).abs() < 1e-6);
    assert!((augmented.upper_a[[1, 0]] - (-0.5)).abs() < 1e-6);
    // Untouched column stays identity.
    assert!((augmented.lower_a[[1, 1]] - 1.0).abs() < 1e-6);

    // Both certified-error matrices MUST be materialized once any A-delta is stored
    // (a None err silently skips the concretize outward penalty).
    assert!(
        augmented.lower_a_err.is_some() && augmented.upper_a_err.is_some(),
        "both err matrices must be materialized after folding an A-delta"
    );
}

/// Stage-0 oracle: `gamma == 0` is the identity map (INVPROP inertness).
#[ntest::timeout(10000)]
#[test]
fn test_augment_gamma_zero_is_identity() {
    let bounds = LinearBounds::identity(3);
    let a_matrix = arr2(&[[1.0, -1.0, 0.0]]);
    let rhs = arr1(&[0.0]);
    let constraints = OutputConstraints::new(a_matrix, rhs, true).unwrap();

    let zeros = arr2(&[[0.0, 0.0, 0.0]]);
    let augmented = augment_bounds_with_constraints(&bounds, &constraints, &zeros, &zeros);

    assert_eq!(augmented.lower_a, bounds.lower_a);
    assert_eq!(augmented.upper_a, bounds.upper_a);
    assert_eq!(augmented.lower_b, bounds.lower_b);
    assert_eq!(augmented.upper_b, bounds.upper_b);
    // No err attached when nothing was folded.
    assert!(augmented.lower_a_err.is_none() && augmented.upper_a_err.is_none());
}

/// Stage-0 oracle: non-conjunction constraints fail closed (core-level guard, not
/// only the CLI). A disjunctive violation must never be dualized as one conjunction.
#[ntest::timeout(10000)]
#[test]
fn test_augment_non_conjunction_fail_closed() {
    let bounds = LinearBounds::identity(2);
    let a_matrix = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let rhs = arr1(&[1.0, 1.0]);
    // is_conjunction = false => disjunction; must be a no-op.
    let constraints = OutputConstraints::new(a_matrix, rhs, false).unwrap();

    let gammas = arr2(&[[0.5, 0.5], [0.5, 0.5]]);
    let augmented = augment_bounds_with_constraints(&bounds, &constraints, &gammas, &gammas);

    assert_eq!(augmented.lower_b, bounds.lower_b);
    assert_eq!(augmented.lower_a, bounds.lower_a);
    assert!(augmented.lower_a_err.is_none());
}

/// Stage-0 oracle: the node-identity gate makes non-seed bounds a sound no-op, so
/// the historical per-layer / input-level augment call sites cannot fold raw C onto
/// unrelated columns (the dimensional-coincidence false-HOLD path).
#[ntest::timeout(10000)]
#[test]
fn test_augment_non_seed_is_noop() {
    // A non-identity 2x2 bound (rows = output coords, but A != I): must be skipped.
    let bounds = LinearBounds {
        lower_a: arr2(&[[2.0, 0.0], [0.0, 3.0]]),
        lower_b: arr1(&[0.0, 0.0]),
        upper_a: arr2(&[[2.0, 0.0], [0.0, 3.0]]),
        upper_b: arr1(&[0.0, 0.0]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let a_matrix = arr2(&[[1.0, 0.0]]);
    let rhs = arr1(&[1.0]);
    let constraints = OutputConstraints::new(a_matrix, rhs, true).unwrap();
    let gammas = arr2(&[[0.5, 0.5]]);

    let augmented = augment_bounds_with_constraints(&bounds, &constraints, &gammas, &gammas);

    assert_eq!(augmented.lower_a, bounds.lower_a);
    assert_eq!(augmented.lower_b, bounds.lower_b);
    assert_eq!(augmented.upper_b, bounds.upper_b);
}

#[ntest::timeout(10000)]
#[test]
fn test_take_best_linear_bounds() {
    let bounds_oc = LinearBounds {
        lower_a: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        lower_b: arr1(&[0.5, 0.3]),
        upper_a: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        upper_b: arr1(&[0.8, 0.9]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let bounds_no_oc = LinearBounds {
        lower_a: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        lower_b: arr1(&[0.4, 0.4]),
        upper_a: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        upper_b: arr1(&[0.85, 0.85]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let best = take_best_linear_bounds(&bounds_oc, &bounds_no_oc);

    assert_eq!(best.lower_b.len(), 2);
    assert_eq!(best.upper_b.len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_invprop_context_routing() {
    let config = InvpropConfig {
        enabled: true,
        apply_output_constraints_to: vec!["BoundReLU".to_string()],
        ..Default::default()
    };

    let context = InvpropBackwardContext::new(&config);

    assert!(context.should_apply_to("layer1", "BoundReLU"));
    assert!(context.should_apply_to("layer2", "BoundReLU"));

    assert!(!context.should_apply_to("layer1", "BoundLinear"));
    assert!(!context.should_apply_to("layer1", "BoundConv2d"));

    let disabled_config = InvpropConfig::default();
    assert!(!disabled_config.should_apply_to("layer1", "BoundReLU"));
}

#[ntest::timeout(10000)]
#[test]
fn test_invprop_context_all_layers() {
    let config = InvpropConfig {
        enabled: true,
        apply_output_constraints_to: vec!["all".to_string()],
        ..Default::default()
    };

    let context = InvpropBackwardContext::new(&config);

    assert!(context.should_apply_to("layer1", "BoundReLU"));
    assert!(context.should_apply_to("layer2", "BoundLinear"));
    assert!(context.should_apply_to("layer3", "BoundConv2d"));
}

#[ntest::timeout(10000)]
#[test]
fn test_invprop_context_by_name() {
    let config = InvpropConfig {
        enabled: true,
        apply_output_constraints_to: vec!["/input.7".to_string()],
        ..Default::default()
    };

    let context = InvpropBackwardContext::new(&config);

    assert!(context.should_apply_to("/input.7/relu", "BoundReLU"));
    assert!(context.should_apply_to("/input.7", "BoundLinear"));

    assert!(!context.should_apply_to("/input.5", "BoundReLU"));
    assert!(!context.should_apply_to("layer1", "BoundReLU"));
}

/// Stage-0 oracle: shared (broadcast) gammas, corrected sign.
/// lower bias delta = -(gamma_0.rhs_0 + gamma_1.rhs_1) = -(0.5*1 + 0.3*2) = -1.1.
#[ntest::timeout(10000)]
#[test]
fn test_augment_bounds_shared_gammas() {
    let bounds = LinearBounds::identity(3);

    let a_matrix = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let rhs = arr1(&[1.0, 2.0]);
    let constraints = OutputConstraints::new(a_matrix, rhs, true).unwrap();

    let gammas_lower = arr2(&[[0.5], [0.3]]);
    let gammas_upper = arr2(&[[0.5], [0.3]]);

    let augmented =
        augment_bounds_with_constraints(&bounds, &constraints, &gammas_lower, &gammas_upper);

    assert!(
        (augmented.lower_b[0] - (-1.1)).abs() < 1e-6,
        "got {}",
        augmented.lower_b[0]
    );
    assert!((augmented.lower_b[1] - (-1.1)).abs() < 1e-6);
    assert!((augmented.lower_b[2] - (-1.1)).abs() < 1e-6);
    // Upper mirrors with +.
    assert!((augmented.upper_b[0] - 1.1).abs() < 1e-6);
    // A-term folded on the two constrained columns.
    assert!((augmented.lower_a[[0, 0]] - 1.5).abs() < 1e-6);
    assert!((augmented.lower_a[[1, 1]] - 1.3).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_mark_bounds_infeasible() {
    let lower = ArrayD::from_shape_vec(vec![3], vec![0.1, 0.2, 0.3]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![3], vec![0.5, 0.6, 0.7]).unwrap();
    let mut bounds = BoundedTensor::new(lower, upper).unwrap();

    assert!(!check_infeasibility(&bounds));

    mark_bounds_infeasible(&mut bounds);

    assert!(check_infeasibility(&bounds));
    assert!(bounds.lower()[[0]] > bounds.upper()[[0]]);
}
