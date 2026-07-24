// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::complete_clip_engine::{add_certified_lower_bias, construct_constraints};
use super::prelude::*;
use crate::beta_crown::config::InputClipType;
use crate::complete_clip::complete_clip;
use crate::relaxed_clip::relaxed_clip;
use ndarray::{array, Array3};

/// Test-local oracle for defensive branches that well-formed engine calls do
/// not naturally construct (for example, a malformed `complete_clip()` result).
fn classify_complete_clip_result_like_engine(
    constrained_lb: ny_core::Result<ArrayD<f32>>,
    lbias: &Array2<f32>,
    thresholds: &Array2<f32>,
) -> bool {
    match constrained_lb {
        Ok(constrained_lb) => match constrained_lb.into_dimensionality::<ndarray::Ix2>() {
            Ok(constrained_lb_2d) => match add_certified_lower_bias(&constrained_lb_2d, lbias) {
                Ok(output_lb) => BetaCrownVerifier::any_verified(&output_lb, thresholds),
                Err(_) => false,
            },
            Err(_) => false,
        },
        Err(ny_core::NyError::InfeasibleDomain(_)) => true,
        Err(_) => false,
    }
}

struct CompleteClippingOkCase {
    verifier: BetaCrownVerifier,
    network: Network,
    input: BoundedTensor,
    new_l: ArrayD<f32>,
    new_u: ArrayD<f32>,
    l_a: Array3<f32>,
    lbias: Array2<f32>,
    thresholds: Array2<f32>,
}

fn build_complete_clipping_ok_case() -> CompleteClippingOkCase {
    let linear = LinearLayer::new(arr2(&[[1.0_f32], [-1.0_f32]]), Some(arr1(&[-0.4, 0.6])))
        .expect("linear layer should build");
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        ..Default::default()
    });
    let (_output_bounds, linear_bounds) = network
        .propagate_crown_with_linear(&input)
        .expect("linear network should produce CROWN bounds");
    let flat = input.flatten();
    let x_l = flat
        .lower()
        .to_owned()
        .into_shape_clone((1, 1))
        .expect("reshape x_l");
    let x_u = flat
        .upper()
        .to_owned()
        .into_shape_clone((1, 1))
        .expect("reshape x_u");
    let l_a = linear_bounds
        .lower_a()
        .clone()
        .into_shape_clone((1, 2, 1))
        .expect("reshape lA");
    let lbias = linear_bounds
        .lower_b()
        .clone()
        .into_shape_clone((1, 2))
        .expect("reshape lbias");
    let thresholds = Array2::zeros((1, 2));
    let (new_l, new_u) = relaxed_clip(
        &x_l.into_dyn(),
        &x_u.into_dyn(),
        &l_a.clone().into_dyn(),
        &lbias.clone().into_dyn(),
        &thresholds.clone().into_dyn(),
        1,
        true,
    )
    .expect("relaxed clipping should preserve feasibility for individually satisfiable specs");

    CompleteClippingOkCase {
        verifier,
        network,
        input,
        new_l,
        new_u,
        l_a,
        lbias,
        thresholds,
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clipping_wrong_rank_result_is_not_verified() {
    let lbias = arr2(&[[0.0, 0.0]]);
    let thresholds = arr2(&[[0.5, 0.5]]);
    let constrained_lb = ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), vec![0.6, 0.7]).unwrap();

    assert!(
        !classify_complete_clip_result_like_engine(Ok(constrained_lb), &lbias, &thresholds),
        "wrong-rank complete_clip output must conservatively return verified=false"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clip_constraint_bias_rounds_half_ulp_down() {
    // The exact difference is halfway between 1.0 and its predecessor.  RN
    // subtraction returns 1.0, which would strengthen the necessary
    // counterexample constraint and could exclude feasible counterexamples.
    let half_ulp_below_one = 2.0_f32.powi(-25);
    let l_a = Array3::zeros((1, 1, 1));
    let lbias = arr2(&[[1.0]]);
    let thresholds = arr2(&[[half_ulp_below_one]]);

    let (_, constr_b) = construct_constraints(&l_a, &lbias, &thresholds).unwrap();

    assert_eq!(constr_b[[0, 0]], ny_tensor::next_down_f32(1.0));
    assert!(
        f64::from(constr_b[[0, 0]]) <= f64::from(lbias[[0, 0]]) - f64::from(thresholds[[0, 0]])
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clip_certified_bias_merge_rounds_half_ulp_down() {
    // The exact sum is the same halfway value.  A certified lower endpoint
    // must store the predecessor rather than RN's upward choice of 1.0.
    let half_ulp_below_one = 2.0_f32.powi(-25);
    let constrained_lb = arr2(&[[1.0]]);
    let lbias = arr2(&[[-half_ulp_below_one]]);

    let output_lb = add_certified_lower_bias(&constrained_lb, &lbias).unwrap();

    assert_eq!(output_lb[[0, 0]], ny_tensor::next_down_f32(1.0));
    assert!(
        f64::from(output_lb[[0, 0]])
            <= f64::from(constrained_lb[[0, 0]]) + f64::from(lbias[[0, 0]])
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_complete_clipping_ok_result_uses_bias_for_verification() {
    let case = build_complete_clipping_ok_case();
    let flat = case.input.flatten();
    let x_l = flat
        .lower()
        .to_owned()
        .into_shape_clone((1, 1))
        .expect("reshape x_l");
    let x_u = flat
        .upper()
        .to_owned()
        .into_shape_clone((1, 1))
        .expect("reshape x_u");
    let dm_lb_pre = BetaCrownVerifier::concretize_dm_lb(&x_l, &x_u, &case.l_a, &case.lbias, true);
    assert!(
        !BetaCrownVerifier::any_verified(&dm_lb_pre, &case.thresholds),
        "pre-clipping bounds should not already verify the cross-constrained specs"
    );

    let dm_lb_post = BetaCrownVerifier::concretize_dm_lb_from_dyn(
        &case.new_l,
        &case.new_u,
        &case.l_a,
        &case.lbias,
        true,
    );
    assert!(
        !BetaCrownVerifier::any_verified(&dm_lb_post, &case.thresholds),
        "relaxed clipping should still leave verification to the complete-clipping branch"
    );

    let (constr_a, constr_b) =
        construct_constraints(&case.l_a, &case.lbias, &case.thresholds).unwrap();
    let constrained_result = complete_clip(
        &case
            .new_l
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("reshape clip_x_l")
            .into_dyn(),
        &case
            .new_u
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("reshape clip_x_u")
            .into_dyn(),
        &case.l_a.clone().into_dyn(),
        &constr_a.into_dyn(),
        &constr_b.into_dyn(),
        -1.0,
        true,
        1,
    );
    let constrained_lb = constrained_result
        .expect("production complete_clip path should return constrained lower bounds");
    let constrained_lb_2d = constrained_lb
        .into_dimensionality::<ndarray::Ix2>()
        .expect("complete_clip output should stay 2D");
    let output_lb = add_certified_lower_bias(&constrained_lb_2d, &case.lbias).unwrap();
    assert!(
        BetaCrownVerifier::any_verified(&output_lb, &case.thresholds),
        "complete clipping should verify only after adding lbias to the constrained result"
    );

    let outcome = case
        .verifier
        .apply_complete_clipping(&case.network, case.input.clone(), &[1], 0.0, None)
        .expect("complete clipping should return a conservative outcome");
    assert!(
        outcome.verified,
        "apply_complete_clipping should mark the cross-constrained domain verified"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clip_with_precomputed_specs_matches_multi_spec_complete_clipping() {
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    let mut multi_network = Network::new();
    multi_network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32], [-1.0_f32]]), Some(arr1(&[-0.4, 0.6])))
            .expect("multi-output linear layer should build"),
    ));
    let (_multi_output_bounds, linear_bounds) = multi_network
        .propagate_crown_with_linear(&input)
        .expect("multi-output network should produce CROWN linear bounds");

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Complete,
        relaxed_clip_iterations: 1,
        ..Default::default()
    });
    let tol = 1.0e-6_f32;

    let helper_outcome = verifier
        .complete_clip_with_precomputed_specs(&input, &[1], &linear_bounds, &[0.0, 0.0])
        .expect("full-spec precomputed complete clipping should succeed");
    let engine_outcome = verifier
        .apply_complete_clipping(&multi_network, input, &[1], 0.0, None)
        .expect("multi-spec complete clipping should succeed");

    assert_eq!(
        helper_outcome.verified, engine_outcome.verified,
        "full-spec precomputed helper should match the direct multi-spec complete-clipping path"
    );

    let helper_flat = helper_outcome.bounds.flatten();
    let engine_flat = engine_outcome.bounds.flatten();
    assert!(
        (helper_flat.lower()[[0]] - engine_flat.lower()[[0]]).abs() <= tol,
        "lower bound mismatch: helper={} engine={}",
        helper_flat.lower()[[0]],
        engine_flat.lower()[[0]]
    );
    assert!(
        (helper_flat.upper()[[0]] - engine_flat.upper()[[0]]).abs() <= tol,
        "upper bound mismatch: helper={} engine={}",
        helper_flat.upper()[[0]],
        engine_flat.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clipping_invalid_spec_error_is_not_verified() {
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![1.0, 0.0].into_dyn();
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0]].into_dyn();

    let lbias = arr2(&[[0.0]]);
    let thresholds = arr2(&[[0.0]]);
    let invalid_spec_result =
        complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1);

    assert!(
        matches!(&invalid_spec_result, Err(ny_core::NyError::InvalidSpec(_))),
        "test setup must exercise the InvalidSpec error path, got {:?}",
        invalid_spec_result
    );
    assert!(
        !classify_complete_clip_result_like_engine(invalid_spec_result, &lbias, &thresholds),
        "non-infeasibility errors must not mark the domain verified"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clipping_inverted_bounds_invalid_spec_is_not_verified() {
    let x_l = array![[1.0, 0.0]].into_dyn();
    let x_u = array![[0.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[0.0]].into_dyn();

    let lbias = arr2(&[[0.0]]);
    let thresholds = arr2(&[[0.0]]);
    let invalid_spec_result =
        complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1);

    assert!(
        matches!(&invalid_spec_result, Err(ny_core::NyError::InvalidSpec(_))),
        "test setup must exercise the inverted-bounds InvalidSpec path, got {:?}",
        invalid_spec_result
    );
    assert!(
        !classify_complete_clip_result_like_engine(invalid_spec_result, &lbias, &thresholds),
        "inverted-bounds InvalidSpec must not mark the domain verified"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clipping_infeasible_domain_marks_verified() {
    let x_l = array![[0.0, 0.0]].into_dyn();
    let x_u = array![[1.0, 1.0]].into_dyn();
    let objective = array![[[1.0, 0.0]]].into_dyn();
    let a_matrix = array![[[1.0, 0.0]]].into_dyn();
    let b_vector = array![[1.0]].into_dyn();

    let lbias = arr2(&[[0.0]]);
    let thresholds = arr2(&[[0.0]]);
    let infeasible_result =
        complete_clip(&x_l, &x_u, &objective, &a_matrix, &b_vector, -1.0, false, 1);

    assert!(
        matches!(
            &infeasible_result,
            Err(ny_core::NyError::InfeasibleDomain(_))
        ),
        "test setup must exercise the InfeasibleDomain path, got {:?}",
        infeasible_result
    );
    assert!(
        classify_complete_clip_result_like_engine(infeasible_result, &lbias, &thresholds),
        "InfeasibleDomain must mark the domain verified"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clipping_evaluate_result_adds_bias_before_verification() {
    let lbias = arr2(&[[0.2, 0.0]]);
    let thresholds = arr2(&[[0.5, 0.5]]);
    let constrained_lb = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.35, 0.1]).unwrap();

    assert!(
        classify_complete_clip_result_like_engine(Ok(constrained_lb), &lbias, &thresholds),
        "verification must use constrained lower bounds plus lbias before threshold comparison"
    );
}
