// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::super::BetaCrownVerifier;
use crate::beta_crown::{BetaCrownConfig, BetaState, CutPool, DomainAlphaState, SplitHistory};
use crate::{IntermediateLinearBounds, Layer, LinearLayer, Network, ReLULayer};

fn deeper_network_3089() -> Network {
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).expect("first linear layer fixture must be valid");
    let w2 = arr2(&[[0.8, -0.2], [-0.3, 0.9]]);
    let linear2 = LinearLayer::new(w2, None).expect("second linear layer fixture must be valid");
    let w3 = arr2(&[[0.6, 0.4]]);
    let linear3 = LinearLayer::new(w3, None).expect("third linear layer fixture must be valid");

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));
    network
}

fn input_bounds_3089() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn())
        .expect("fixture input bounds must be valid")
}

struct OptimizeLoopFixture3089 {
    verifier: BetaCrownVerifier,
    network: Network,
    input: BoundedTensor,
    layer_bounds: Vec<Arc<BoundedTensor>>,
    history: SplitHistory,
    beta_state: BetaState,
    alpha_state: DomainAlphaState,
    cut_pool: CutPool,
}

impl OptimizeLoopFixture3089 {
    fn new(config: BetaCrownConfig) -> Self {
        let verifier = BetaCrownVerifier::new(config);
        let network = deeper_network_3089();
        let input = input_bounds_3089();
        let layer_bounds: Vec<Arc<BoundedTensor>> = network
            .collect_ibp_bounds(&input)
            .expect("IBP bounds fixture must compute successfully")
            .into_iter()
            .map(Arc::new)
            .collect();
        let history = SplitHistory::new();
        let beta_state = BetaState::empty();
        let alpha_state =
            DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
        let cut_pool = CutPool::default();

        assert!(
            !alpha_state.is_empty(),
            "fixture must exercise optimize_loop with unstable alpha parameters present"
        );

        Self {
            verifier,
            network,
            input,
            layer_bounds,
            history,
            beta_state,
            alpha_state,
            cut_pool,
        }
    }
}

fn assert_bounds_match_3089(actual: &BoundedTensor, expected: &BoundedTensor) {
    assert_eq!(actual.lower().shape(), expected.lower().shape());
    assert_eq!(actual.upper().shape(), expected.upper().shape());

    for (idx, (actual_lower, expected_lower)) in actual
        .lower()
        .iter()
        .zip(expected.lower().iter())
        .enumerate()
    {
        assert!(
            same_bound_value_3089(*actual_lower, *expected_lower),
            "lower bound mismatch at index {idx}: actual={actual_lower}, expected={expected_lower}",
        );
    }

    for (idx, (actual_upper, expected_upper)) in actual
        .upper()
        .iter()
        .zip(expected.upper().iter())
        .enumerate()
    {
        assert!(
            same_bound_value_3089(*actual_upper, *expected_upper),
            "upper bound mismatch at index {idx}: actual={actual_upper}, expected={expected_upper}",
        );
    }
}

fn same_bound_value_3089(actual: f32, expected: f32) -> bool {
    if actual.is_finite() && expected.is_finite() {
        (actual - expected).abs() <= 1e-6
    } else {
        actual == expected || (actual.is_nan() && expected.is_nan())
    }
}

fn assert_intermediate_match_3089(
    actual: &IntermediateLinearBounds,
    expected: &IntermediateLinearBounds,
    input: &BoundedTensor,
    layer_bounds: &[Arc<BoundedTensor>],
) {
    assert_eq!(actual.start_layer(), expected.start_layer());
    assert_eq!(
        actual.bounds_at_layer().len(),
        expected.bounds_at_layer().len()
    );

    for layer_idx in 0..actual.bounds_at_layer().len() {
        let actual_layer = actual
            .get(layer_idx)
            .expect("actual intermediate layer should exist");
        let expected_layer = expected
            .get(layer_idx)
            .expect("expected intermediate layer should exist");
        let layer_input = if layer_idx == 0 {
            input
        } else {
            layer_bounds[layer_idx - 1].as_ref()
        };

        let actual_concrete = actual_layer.concretize_sound(layer_input);
        let expected_concrete = expected_layer.concretize_sound(layer_input);
        assert_bounds_match_3089(&actual_concrete, &expected_concrete);
    }
}

#[test]
fn test_optimize_joint_bounds_beta_iterations_zero_matches_direct_compute_3089() {
    let mut fixture = OptimizeLoopFixture3089::new(BetaCrownConfig {
        beta_iterations: 0,
        use_alpha_crown: true,
        ..Default::default()
    });

    let (expected_bounds, expected_intermediate) = fixture
        .verifier
        .compute_bounds_capturing_intermediate(
            &fixture.network,
            &fixture.input,
            &fixture.history,
            &fixture.layer_bounds,
            &fixture.beta_state,
            &fixture.alpha_state,
            &fixture.cut_pool,
            None,
        )
        .expect("full optimize-loop baseline bounds must compute");
    let (actual_bounds, actual_intermediate) = fixture
        .verifier
        .optimize_joint_bounds(
            &fixture.network,
            &fixture.input,
            &fixture.history,
            &fixture.layer_bounds,
            &mut fixture.beta_state,
            &mut fixture.alpha_state,
            &mut fixture.cut_pool,
            None,
        )
        .expect("beta_iterations == 0 path must match direct compute");

    assert_bounds_match_3089(&actual_bounds, &expected_bounds);
    assert_intermediate_match_3089(
        &actual_intermediate,
        &expected_intermediate,
        &fixture.input,
        &fixture.layer_bounds,
    );
}

#[test]
fn test_optimize_joint_bounds_from_layer_beta_iterations_zero_matches_partial_compute_3089() {
    let mut fixture = OptimizeLoopFixture3089::new(BetaCrownConfig {
        beta_iterations: 0,
        use_alpha_crown: true,
        ..Default::default()
    });
    let start_layer = 1;

    let (_, parent_intermediate) = fixture
        .verifier
        .compute_bounds_capturing_intermediate(
            &fixture.network,
            &fixture.input,
            &fixture.history,
            &fixture.layer_bounds,
            &fixture.beta_state,
            &fixture.alpha_state,
            &fixture.cut_pool,
            None,
        )
        .expect("parent intermediate bounds must compute");
    let (expected_bounds, expected_intermediate) = fixture
        .verifier
        .compute_bounds_from_layer(
            &fixture.network,
            &fixture.input,
            &fixture.history,
            &fixture.layer_bounds,
            &fixture.beta_state,
            &fixture.alpha_state,
            &fixture.cut_pool,
            start_layer,
            &parent_intermediate,
            None,
        )
        .expect("from-layer baseline bounds must compute");
    let (actual_bounds, actual_intermediate) = fixture
        .verifier
        .optimize_joint_bounds_from_layer(
            &fixture.network,
            &fixture.input,
            &fixture.history,
            &fixture.layer_bounds,
            &mut fixture.beta_state,
            &mut fixture.alpha_state,
            &mut fixture.cut_pool,
            start_layer,
            &parent_intermediate,
            None,
        )
        .expect("from-layer beta_iterations == 0 path must match partial compute");

    assert_bounds_match_3089(&actual_bounds, &expected_bounds);
    assert_intermediate_match_3089(
        &actual_intermediate,
        &expected_intermediate,
        &fixture.input,
        &fixture.layer_bounds,
    );
}

#[test]
fn test_optimize_joint_bounds_past_deadline_before_first_iteration_errors_3089() {
    let mut fixture = OptimizeLoopFixture3089::new(BetaCrownConfig {
        beta_iterations: 3,
        use_alpha_crown: true,
        ..Default::default()
    });
    fixture.verifier.config.alpha_config.deadline =
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());

    let err = fixture
        .verifier
        .optimize_joint_bounds(
            &fixture.network,
            &fixture.input,
            &fixture.history,
            &fixture.layer_bounds,
            &mut fixture.beta_state,
            &mut fixture.alpha_state,
            &mut fixture.cut_pool,
            None,
        )
        .expect_err("expired deadline before iter 0 must surface numerical instability");

    assert!(
        matches!(err, ny_core::NyError::NumericalInstability(ref msg) if msg.contains("No valid bounds computed during joint optimization")),
        "expected optimize_joint_bounds deadline error, got {err:?}"
    );
}

#[test]
fn test_optimize_joint_bounds_from_layer_past_deadline_before_first_iteration_errors_3089() {
    let mut fixture = OptimizeLoopFixture3089::new(BetaCrownConfig {
        beta_iterations: 3,
        use_alpha_crown: true,
        ..Default::default()
    });
    fixture.verifier.config.alpha_config.deadline =
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    let start_layer = 1;

    let (_, parent_intermediate) = fixture
        .verifier
        .compute_bounds_capturing_intermediate(
            &fixture.network,
            &fixture.input,
            &fixture.history,
            &fixture.layer_bounds,
            &fixture.beta_state,
            &fixture.alpha_state,
            &fixture.cut_pool,
            None,
        )
        .expect("from-layer parent intermediate bounds must compute");
    let err = fixture
        .verifier
        .optimize_joint_bounds_from_layer(
            &fixture.network,
            &fixture.input,
            &fixture.history,
            &fixture.layer_bounds,
            &mut fixture.beta_state,
            &mut fixture.alpha_state,
            &mut fixture.cut_pool,
            start_layer,
            &parent_intermediate,
            None,
        )
        .expect_err("expired deadline before iter 0 must surface numerical instability from-layer");

    assert!(
        matches!(err, ny_core::NyError::NumericalInstability(ref msg) if msg.contains("No valid bounds computed during joint optimization (from layer)")),
        "expected optimize_joint_bounds_from_layer deadline error, got {err:?}"
    );
}
