// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use std::sync::Arc;

use super::super::tensor_ext::{has_inverted_output_bounds, BoundedTensorExt};
use super::super::BetaCrownVerifier;
use super::optimize_loop::{patience_exhausted_after_iteration, update_best_output_bounds};
use crate::beta_crown::{BetaCrownConfig, BetaState, CutPool, DomainAlphaState, SplitHistory};
use crate::{Layer, LinearLayer, Network, ReLULayer};

const SAMPLE_TOLERANCE_NY: f32 = 1.0e-5;

/// Construct BoundedTensor from validated (lower <= upper, finite) slices.
fn make_bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let l = ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap();
    BoundedTensor::new(l, u).unwrap()
}

/// Construct BoundedTensor bypassing validation — allows NaN, Inf, inverted bounds.
fn make_bounds_unchecked(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let l = ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap();
    BoundedTensor::new_unchecked(l, u).unwrap()
}

fn simple_network_2769() -> Network {
    let linear1 = LinearLayer::new(arr2(&[[1.0, -1.0], [-1.0, 1.0]]), None).unwrap();
    let linear2 = LinearLayer::new(arr2(&[[1.0, 1.0]]), None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network
}

fn simple_network_output_2769(x0: f32, x1: f32) -> f32 {
    (x0 - x1).abs()
}

fn assert_simple_network_bounds_contain_samples_2769(bounds: &BoundedTensor) {
    let flat = bounds.flatten();
    let lower = flat.lower()[[0]];
    let upper = flat.upper()[[0]];

    for i in -5..=5 {
        let x0 = i as f32 / 10.0;
        for j in -5..=5 {
            let x1 = j as f32 / 10.0;
            let y = simple_network_output_2769(x0, x1);
            assert!(
                y >= lower - SAMPLE_TOLERANCE_NY && y <= upper + SAMPLE_TOLERANCE_NY,
                "sample ({x0}, {x1}) -> {y} must stay within [{lower}, {upper}]",
            );
        }
    }
}

#[test]
fn test_has_inverted_output_bounds_normal() {
    let bounds = make_bounds(&[-1.0, 0.0, -3.0], &[1.0, 2.0, 0.5]);
    assert!(!has_inverted_output_bounds(&bounds));
}

#[test]
fn test_has_inverted_output_bounds_tight() {
    let bounds = make_bounds(&[1.0, 2.0], &[1.0, 2.0]);
    assert!(!has_inverted_output_bounds(&bounds));
}

#[test]
fn test_has_inverted_output_bounds_inverted() {
    let bounds = make_bounds_unchecked(&[-1.0, 3.0, -3.0], &[1.0, 2.0, 0.5]);
    assert!(has_inverted_output_bounds(&bounds));
}

#[test]
fn test_has_inverted_output_bounds_inf_not_inverted() {
    let bounds = make_bounds_unchecked(&[f32::NEG_INFINITY, 0.0], &[f32::INFINITY, 1.0]);
    assert!(!has_inverted_output_bounds(&bounds));
}

#[test]
fn test_has_inverted_output_bounds_infeasible_sentinel() {
    let bounds = make_bounds_unchecked(
        &[f32::INFINITY, f32::INFINITY],
        &[f32::NEG_INFINITY, f32::NEG_INFINITY],
    );
    assert!(!has_inverted_output_bounds(&bounds));
}

#[test]
fn test_has_inverted_output_bounds_nan() {
    let bounds = make_bounds_unchecked(&[f32::NAN, 0.0], &[1.0, 1.0]);
    assert!(!has_inverted_output_bounds(&bounds));
}

#[test]
fn test_update_best_output_bounds_keeps_peak_after_overshoot_3072() {
    let mut best_lower = f32::NEG_INFINITY;
    let mut best_bounds = None;

    let first = make_bounds(&[-0.5], &[1.0]);
    let peak = make_bounds(&[0.25], &[0.5]);
    let overshoot = make_bounds(&[0.1], &[0.4]);

    assert_eq!(
        update_best_output_bounds(&mut best_lower, &mut best_bounds, &first),
        -0.5
    );
    assert_eq!(
        update_best_output_bounds(&mut best_lower, &mut best_bounds, &peak),
        0.25
    );
    assert_eq!(
        update_best_output_bounds(&mut best_lower, &mut best_bounds, &overshoot),
        0.1
    );

    let best = best_bounds.expect("peak iteration should remain selected after overshoot");
    assert_eq!(best_lower, 0.25);
    assert_eq!(best.lower()[[0]], 0.25);
    assert_eq!(best.upper()[[0]], 0.5);
}

#[test]
fn test_update_best_output_bounds_skips_non_finite_candidate_3072() {
    let mut best_lower = f32::NEG_INFINITY;
    let mut best_bounds = None;

    let finite = make_bounds(&[0.2], &[0.6]);
    let nan_candidate = make_bounds_unchecked(&[f32::NAN], &[0.7]);

    assert_eq!(
        update_best_output_bounds(&mut best_lower, &mut best_bounds, &finite),
        0.2
    );
    assert_eq!(
        update_best_output_bounds(&mut best_lower, &mut best_bounds, &nan_candidate),
        f32::NEG_INFINITY
    );

    let best = best_bounds.expect("non-finite candidate must not replace best finite bounds");
    assert_eq!(best_lower, 0.2);
    assert_eq!(best.lower()[[0]], 0.2);
    assert_eq!(best.upper()[[0]], 0.6);
}

#[test]
fn test_patience_exhausted_after_stalled_iterations_2418() {
    let mut no_improve_iters = 0usize;
    let patience = 2usize;

    assert!(
        !patience_exhausted_after_iteration(
            f32::NEG_INFINITY,
            0.25,
            &mut no_improve_iters,
            patience
        ),
        "first finite improvement should reset the patience counter"
    );
    assert_eq!(no_improve_iters, 0);

    assert!(
        !patience_exhausted_after_iteration(0.25, 0.25, &mut no_improve_iters, patience),
        "first stalled iteration should not stop before patience is exceeded"
    );
    assert_eq!(no_improve_iters, 1);

    assert!(
        !patience_exhausted_after_iteration(0.25, 0.25, &mut no_improve_iters, patience),
        "second stalled iteration should still be tolerated"
    );
    assert_eq!(no_improve_iters, 2);

    assert!(
        patience_exhausted_after_iteration(0.25, 0.25, &mut no_improve_iters, patience),
        "#2418: the next stalled iteration must trigger the break condition"
    );
    assert_eq!(no_improve_iters, 3);
}

#[test]
fn test_patience_exhausted_after_iteration_resets_on_improvement_2418() {
    let mut no_improve_iters = 2usize;
    assert!(
        !patience_exhausted_after_iteration(0.10, 0.20, &mut no_improve_iters, 2),
        "a better lower bound must clear the stalled-iteration counter"
    );
    assert_eq!(no_improve_iters, 0);
}

#[test]
fn test_optimize_joint_bounds_from_layer_contains_samples_2769() {
    let network = simple_network_2769();
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();
    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let cut_pool = CutPool::default();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        use_alpha_crown: true,
        beta_iterations: 5,
        ..Default::default()
    });

    let (_, parent_intermediate) = verifier
        .compute_bounds_capturing_intermediate(
            &network,
            &input,
            &history,
            &layer_bounds,
            &beta_state,
            &alpha_state,
            &cut_pool,
            None,
        )
        .expect("parent intermediate bounds should compute");

    let mut beta_state = BetaState::empty();
    let mut alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let mut cut_pool = CutPool::default();
    let (bounds, _) = verifier
        .optimize_joint_bounds_from_layer(
            &network,
            &input,
            &history,
            &layer_bounds,
            &mut beta_state,
            &mut alpha_state,
            &mut cut_pool,
            1,
            &parent_intermediate,
            None,
        )
        .expect("optimize_joint_bounds_from_layer should succeed");

    assert_simple_network_bounds_contain_samples_2769(&bounds);
}

fn deeper_network_3072() -> Network {
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[0.8, -0.2], [-0.3, 0.9]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    let w3 = arr2(&[[0.6, 0.4]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));
    network
}

fn input_bounds_3072() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap()
}

fn sorted_relu_keys_3072(alpha_state: &DomainAlphaState) -> Vec<(usize, usize)> {
    let mut relu_keys: Vec<(usize, usize)> = alpha_state.neurons.keys().copied().collect();
    relu_keys.sort_unstable();
    assert_eq!(relu_keys, vec![(1, 0), (1, 1), (3, 0), (3, 1)]);
    relu_keys
}

fn apply_alpha_assignment_3072(
    alpha_state: &mut DomainAlphaState,
    relu_keys: &[(usize, usize)],
    assignment: &[f32],
) {
    assert_eq!(relu_keys.len(), assignment.len());
    for (relu_key, value) in relu_keys.iter().zip(assignment.iter().copied()) {
        alpha_state
            .neurons
            .get_mut(relu_key)
            .expect("manual #3072 alpha assignment must target an unstable ReLU")
            .alpha = value;
    }
}

fn compute_bounds_for_alpha_assignment_3072(
    verifier: &BetaCrownVerifier,
    network: &Network,
    input: &BoundedTensor,
    history: &SplitHistory,
    layer_bounds: &[Arc<BoundedTensor>],
    assignment: &[f32],
) -> BoundedTensor {
    let mut alpha =
        DomainAlphaState::from_layer_bounds_and_constraints(network, layer_bounds, history);
    let relu_keys = sorted_relu_keys_3072(&alpha);
    apply_alpha_assignment_3072(&mut alpha, &relu_keys, assignment);
    let (bounds, _) = verifier
        .compute_bounds_capturing_intermediate(
            network,
            input,
            history,
            layer_bounds,
            &BetaState::empty(),
            &alpha,
            &CutPool::default(),
            None,
        )
        .unwrap();
    bounds
}

fn compute_bounds_from_layer_for_alpha_assignment_3072(
    verifier: &BetaCrownVerifier,
    network: &Network,
    input: &BoundedTensor,
    history: &SplitHistory,
    layer_bounds: &[Arc<BoundedTensor>],
    start_layer: usize,
    assignment: &[f32],
) -> BoundedTensor {
    let mut alpha =
        DomainAlphaState::from_layer_bounds_and_constraints(network, layer_bounds, history);
    let relu_keys = sorted_relu_keys_3072(&alpha);
    apply_alpha_assignment_3072(&mut alpha, &relu_keys, assignment);

    let alpha_ref =
        DomainAlphaState::from_layer_bounds_and_constraints(network, layer_bounds, history);
    let (_, parent_intermediate) = verifier
        .compute_bounds_capturing_intermediate(
            network,
            input,
            history,
            layer_bounds,
            &BetaState::empty(),
            &alpha_ref,
            &CutPool::default(),
            None,
        )
        .unwrap();

    let (bounds, _) = verifier
        .compute_bounds_from_layer(
            network,
            input,
            history,
            layer_bounds,
            &BetaState::empty(),
            &alpha,
            &CutPool::default(),
            start_layer,
            &parent_intermediate,
            None,
        )
        .unwrap();
    bounds
}

#[test]
fn test_update_best_output_bounds_keeps_real_network_peak_after_manual_alpha_drop_3072() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let network = deeper_network_3072();
    let input = input_bounds_3072();
    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();
    let history = SplitHistory::new();

    let degraded = compute_bounds_for_alpha_assignment_3072(
        &verifier,
        &network,
        &input,
        &history,
        &layer_bounds,
        &[1.0, 0.0, 1.0, 0.0],
    );
    let peak = compute_bounds_for_alpha_assignment_3072(
        &verifier,
        &network,
        &input,
        &history,
        &layer_bounds,
        &[0.0, 0.0, 0.0, 0.0],
    );

    assert!(
        peak.lower_scalar() > degraded.lower_scalar() + 1e-6,
        "manual #3072 full-path fixture must have a strictly tighter earlier bound: peak={}, degraded={}",
        peak.lower_scalar(),
        degraded.lower_scalar(),
    );

    let mut best_lower = f32::NEG_INFINITY;
    let mut best_bounds = None;
    update_best_output_bounds(&mut best_lower, &mut best_bounds, &degraded);
    update_best_output_bounds(&mut best_lower, &mut best_bounds, &peak);
    update_best_output_bounds(&mut best_lower, &mut best_bounds, &degraded);

    let best = best_bounds.expect("peak full-path bounds should remain selected");
    assert_eq!(best_lower, peak.lower_scalar());
    assert_eq!(best.lower()[[0]], peak.lower()[[0]]);
    assert_eq!(best.upper()[[0]], peak.upper()[[0]]);
}

#[test]
fn test_update_best_output_bounds_keeps_from_layer_peak_after_manual_alpha_drop_3072() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let network = deeper_network_3072();
    let input = input_bounds_3072();
    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();
    let history = SplitHistory::new();

    let degraded = compute_bounds_from_layer_for_alpha_assignment_3072(
        &verifier,
        &network,
        &input,
        &history,
        &layer_bounds,
        1,
        &[1.0, 1.0, 0.0, 0.0],
    );
    let peak = compute_bounds_from_layer_for_alpha_assignment_3072(
        &verifier,
        &network,
        &input,
        &history,
        &layer_bounds,
        1,
        &[0.0, 0.0, 0.0, 0.0],
    );

    assert!(
        peak.lower_scalar() > degraded.lower_scalar() + 1e-6,
        "manual #3072 from_layer fixture must have a strictly tighter earlier bound: peak={}, degraded={}",
        peak.lower_scalar(),
        degraded.lower_scalar(),
    );

    let mut best_lower = f32::NEG_INFINITY;
    let mut best_bounds = None;
    update_best_output_bounds(&mut best_lower, &mut best_bounds, &degraded);
    update_best_output_bounds(&mut best_lower, &mut best_bounds, &peak);
    update_best_output_bounds(&mut best_lower, &mut best_bounds, &degraded);

    let best = best_bounds.expect("peak from_layer bounds should remain selected");
    assert_eq!(best_lower, peak.lower_scalar());
    assert_eq!(best.lower()[[0]], peak.lower()[[0]]);
    assert_eq!(best.upper()[[0]], peak.upper()[[0]]);
}
