// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for domain-specific α state.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::arr1;
use ny_tensor::BoundedTensor;

use super::*;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::config::AdaptiveOptConfig;
use crate::{GraphNetwork, GraphNode, Layer, ReLULayer};

#[test]
fn domain_alpha_adam_step_sanitizes_nan_updates() {
    let mut state = DomainAlphaState::empty();
    state.neurons.insert((1, 0), AlphaNeuronState::new(0.25));
    state.neurons.get_mut(&(1, 0)).unwrap().grad = f32::NAN;

    let config = AdaptiveOptConfig::default();
    let _ = state.gradient_step_adam(&config, 1);

    let alpha = state.alpha(1, 0);
    assert!(
        alpha.is_finite(),
        "alpha must stay finite after NaN gradient"
    );
    assert!(
        (alpha - 0.5).abs() <= f32::EPSILON,
        "NaN updates must fall back to alpha=0.5, got {alpha}"
    );
}

#[test]
fn graph_from_parent_sanitizes_nan_alpha_warm_start() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.set_output("relu0");

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    let history = GraphSplitHistory::new();

    let mut parent = GraphDomainAlphaState::empty();
    parent.insert(
        "relu0".to_string(),
        0,
        AlphaNeuronState {
            alpha: f32::NAN,
            grad: 0.0,
            velocity: 0.0,
            adam_m: 0.0,
            adam_v: 0.0,
            adam_v_max: 0.0,
        },
    );

    let child =
        GraphDomainAlphaState::from_parent(&parent, &graph, &node_bounds, &history, &input_bounds);
    let alpha = child.alpha("relu0", 0);
    assert!(alpha.is_finite(), "child alpha must stay finite");
    assert!(
        (alpha - 0.5).abs() <= f32::EPSILON,
        "NaN warm-start must fall back to alpha=0.5, got {alpha}"
    );

    let alpha_array = child.build_alpha_array("relu0", &input_bounds);
    assert!(
        alpha_array[0].is_finite(),
        "alpha array must not contain NaN after warm-start"
    );
}

#[test]
fn graph_from_root_alpha_state_transfers_optimized_values() {
    // Build a graph with two ReLU nodes, each with 3 neurons.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["relu0".to_string()],
    ));
    graph.set_output("relu1");

    // Input bounds: 3 neurons, all unstable (l<0<u).
    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0, -2.0, -0.5]).into_dyn(),
        arr1(&[1.0, 0.5, 3.0]).into_dyn(),
    )
    .unwrap();
    // relu0 pre-activation bounds = input_bounds (since relu0 takes _input)
    // relu1 pre-activation bounds: 3 unstable neurons
    let relu1_bounds = BoundedTensor::new(
        arr1(&[-0.3, -1.0, -0.1]).into_dyn(),
        arr1(&[0.7, 2.0, 0.4]).into_dyn(),
    )
    .unwrap();
    let mut node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    node_bounds.insert("relu0".to_string(), Arc::new(relu1_bounds));

    let history = GraphSplitHistory::new();

    // Build a "root-level optimized" GraphAlphaState with specific alpha values.
    let mut root_alpha = crate::bounds::GraphAlphaState::new();
    root_alpha
        .alphas
        .insert("relu0".to_string(), arr1(&[0.7, 0.3, 0.9]));
    root_alpha
        .alphas_upper
        .insert("relu0".to_string(), arr1(&[0.2, 0.6, 0.4]));
    root_alpha
        .alphas
        .insert("relu1".to_string(), arr1(&[0.15, 0.85, 0.42]));
    root_alpha
        .alphas_upper
        .insert("relu1".to_string(), arr1(&[0.9, 0.1, 0.75]));

    let state = GraphDomainAlphaState::from_root_alpha_state(
        &root_alpha,
        &graph,
        &node_bounds,
        &history,
        &input_bounds,
    );

    assert_eq!(state.len(), 6, "expected 6 unstable neurons");
    for (node_name, expected_lower, expected_upper) in [
        ("relu0", [0.7_f32, 0.3, 0.9], [0.2_f32, 0.6, 0.4]),
        ("relu1", [0.15_f32, 0.85, 0.42], [0.9_f32, 0.1, 0.75]),
    ] {
        for (idx, (&lower, &upper)) in expected_lower.iter().zip(expected_upper.iter()).enumerate()
        {
            let actual_lower = state.alpha(node_name, idx);
            let actual_upper = state.alpha_upper(node_name, idx);
            assert!(
                (actual_lower - lower).abs() < 1e-6,
                "{node_name} lower neuron {idx} should be {lower}, got {actual_lower}"
            );
            assert!(
                (actual_upper - upper).abs() < 1e-6,
                "{node_name} upper neuron {idx} should be {upper}, got {actual_upper}"
            );
        }
    }
}

/// #hard-six α-inherit-expand: with a CHANNEL-ONLY root alpha (the
/// `full_conv_alpha: false` warmup, e.g. cifar100_2024 — length-C arrays +
/// `spatial_shapes`), the historical `from_root_alpha_state` mis-indexes the
/// first C flat neurons with OTHER channels' α and silently keeps the
/// heuristic for flat index ≥ C. Under `NY_ALPHA_INHERIT_EXPAND=1` the
/// channel array is spatially broadcast so every unstable neuron seeds from
/// ITS OWN channel's warmup-optimized α. Gate unset stays byte-identical.
#[test]
fn graph_from_root_alpha_state_channel_only_expand_gate_4404() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.set_output("relu0");

    // Conv-shaped pre-activation [C=2, H=2, W=2]: 8 neurons, all unstable
    // with u > -l so the per-neuron heuristic α is 1.0 everywhere (distinct
    // from the channel values below).
    let shape = ndarray::IxDyn(&[2, 2, 2]);
    let input_bounds = BoundedTensor::new(
        ndarray::ArrayD::from_elem(shape.clone(), -1.0_f32),
        ndarray::ArrayD::from_elem(shape, 2.0_f32),
    )
    .unwrap();
    let node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    let history = GraphSplitHistory::new();

    // Channel-only root alpha: length C=2, spatial_shapes records [2, 2, 2].
    let mut root_alpha = crate::bounds::GraphAlphaState::new();
    root_alpha
        .alphas
        .insert("relu0".to_string(), arr1(&[0.7_f32, 0.3]));
    root_alpha
        .alphas_upper
        .insert("relu0".to_string(), arr1(&[0.6_f32, 0.4]));
    root_alpha
        .spatial_shapes
        .insert("relu0".to_string(), vec![2, 2, 2]);

    // Gate OFF (historical, byte-identical): flat idx 0/1 take alpha_arr[0/1]
    // (idx 1 is channel 0's second spatial position but receives CHANNEL 1's
    // α — the documented mis-index); idx >= C keep the heuristic (1.0).
    let state_off = GraphDomainAlphaState::from_root_alpha_state(
        &root_alpha,
        &graph,
        &node_bounds,
        &history,
        &input_bounds,
    );
    assert_eq!(state_off.len(), 8, "expected 8 unstable neurons");
    assert!((state_off.alpha("relu0", 0) - 0.7).abs() < 1e-6);
    assert!((state_off.alpha("relu0", 1) - 0.3).abs() < 1e-6);
    for idx in 2..8 {
        let a = state_off.alpha("relu0", idx);
        assert!(
            (a - 1.0).abs() < 1e-6,
            "gate off: neuron {idx} must keep the heuristic 1.0, got {a}"
        );
    }

    // Gate ON: flat idx i seeds from channel (i / spatial)'s α.
    let state_on =
        crate::tests::with_serialized_env_vars(&[("NY_ALPHA_INHERIT_EXPAND", "1")], || {
            GraphDomainAlphaState::from_root_alpha_state(
                &root_alpha,
                &graph,
                &node_bounds,
                &history,
                &input_bounds,
            )
        });
    for idx in 0..8 {
        let expected = if idx < 4 { 0.7 } else { 0.3 };
        let expected_upper = if idx < 4 { 0.6 } else { 0.4 };
        let a = state_on.alpha("relu0", idx);
        let au = state_on.alpha_upper("relu0", idx);
        assert!(
            (a - expected).abs() < 1e-6,
            "gate on: neuron {idx} must seed from its channel's α {expected}, got {a}"
        );
        assert!(
            (au - expected_upper).abs() < 1e-6,
            "gate on: neuron {idx} upper must seed from its channel's α {expected_upper}, got {au}"
        );
    }
}

#[test]
fn graph_from_root_alpha_state_sanitizes_nan() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.set_output("relu0");

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0, -2.0]).into_dyn(), arr1(&[1.0, 0.5]).into_dyn()).unwrap();
    let node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    let history = GraphSplitHistory::new();

    // Root alpha with NaN and Inf values.
    let mut root_alpha = crate::bounds::GraphAlphaState::new();
    root_alpha
        .alphas
        .insert("relu0".to_string(), arr1(&[f32::NAN, f32::INFINITY]));

    let state = GraphDomainAlphaState::from_root_alpha_state(
        &root_alpha,
        &graph,
        &node_bounds,
        &history,
        &input_bounds,
    );

    let a0 = state.alpha("relu0", 0);
    assert!(a0.is_finite(), "NaN root alpha must be sanitized, got {a0}");
    assert!(
        (a0 - 0.5).abs() <= f32::EPSILON,
        "NaN root alpha must fall back to 0.5, got {a0}"
    );
    let a1 = state.alpha("relu0", 1);
    assert!(a1.is_finite(), "Inf root alpha must be clamped, got {a1}");
    assert!(
        (a1 - 1.0).abs() <= f32::EPSILON,
        "Inf root alpha must clamp to 1.0, got {a1}"
    );
}

#[test]
fn graph_from_root_alpha_state_skips_stable_neurons() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.set_output("relu0");

    // Neuron 0: unstable (-1, 1), Neuron 1: stable positive (0.5, 2.0),
    // Neuron 2: stable negative (-3.0, -0.1)
    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0, 0.5, -3.0]).into_dyn(),
        arr1(&[1.0, 2.0, -0.1]).into_dyn(),
    )
    .unwrap();
    let node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    let history = GraphSplitHistory::new();

    let mut root_alpha = crate::bounds::GraphAlphaState::new();
    root_alpha
        .alphas
        .insert("relu0".to_string(), arr1(&[0.65, 0.33, 0.77]));

    let state = GraphDomainAlphaState::from_root_alpha_state(
        &root_alpha,
        &graph,
        &node_bounds,
        &history,
        &input_bounds,
    );

    // Only neuron 0 is unstable — neurons 1 and 2 are stable.
    assert_eq!(state.len(), 1, "only 1 unstable neuron expected");
    let a0 = state.alpha("relu0", 0);
    assert!(
        (a0 - 0.65).abs() < 1e-6,
        "unstable neuron should get root alpha 0.65, got {a0}"
    );
    // Stable neurons should not be in the state at all.
    assert!(
        state.neuron("relu0", 1).is_none(),
        "stable positive neuron must not be in state"
    );
    assert!(
        state.neuron("relu0", 2).is_none(),
        "stable negative neuron must not be in state"
    );
}

#[test]
fn graph_from_root_alpha_state_handles_missing_node() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.set_output("relu0");

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    let history = GraphSplitHistory::new();

    // Root alpha has NO entry for relu0 — should fall back to heuristic.
    let root_alpha = crate::bounds::GraphAlphaState::new();

    let state = GraphDomainAlphaState::from_root_alpha_state(
        &root_alpha,
        &graph,
        &node_bounds,
        &history,
        &input_bounds,
    );

    assert_eq!(state.len(), 1, "1 unstable neuron expected");
    // Heuristic: u(1.0) > -l(1.0) is false (equal), so alpha = 0.0
    let a0 = state.alpha("relu0", 0);
    assert!(
        (a0 - 0.0).abs() <= f32::EPSILON,
        "missing root alpha should use heuristic (0.0 since u == -l), got {a0}"
    );
}

#[test]
fn graph_from_root_alpha_state_handles_short_alpha_array() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.set_output("relu0");

    // 3 unstable neurons
    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0, -2.0, -0.5]).into_dyn(),
        arr1(&[1.0, 0.5, 3.0]).into_dyn(),
    )
    .unwrap();
    let node_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    let history = GraphSplitHistory::new();

    // Root alpha only has 1 value — shorter than the 3 neurons.
    let mut root_alpha = crate::bounds::GraphAlphaState::new();
    root_alpha.alphas.insert("relu0".to_string(), arr1(&[0.42]));

    let state = GraphDomainAlphaState::from_root_alpha_state(
        &root_alpha,
        &graph,
        &node_bounds,
        &history,
        &input_bounds,
    );

    assert_eq!(state.len(), 3, "3 unstable neurons expected");
    // Neuron 0 gets root alpha.
    let a0 = state.alpha("relu0", 0);
    assert!(
        (a0 - 0.42).abs() < 1e-6,
        "neuron 0 should get root alpha 0.42, got {a0}"
    );
    // Neurons 1 and 2 should keep heuristic values (index >= alpha array length).
    // Neuron 1: l=-2.0, u=0.5 → u(0.5) > -l(2.0)? No → heuristic = 0.0
    let a1 = state.alpha("relu0", 1);
    assert!(
        (a1 - 0.0).abs() <= f32::EPSILON,
        "neuron 1 should use heuristic (0.0), got {a1}"
    );
    // Neuron 2: l=-0.5, u=3.0 → u(3.0) > -l(0.5)? Yes → heuristic = 1.0
    let a2 = state.alpha("relu0", 2);
    assert!(
        (a2 - 1.0).abs() <= f32::EPSILON,
        "neuron 2 should use heuristic (1.0), got {a2}"
    );
}

#[test]
fn graph_build_alpha_upper_array_uses_upper_path_values() {
    let mut state = GraphDomainAlphaState::empty();
    state.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.2));
    state.insert("relu0".to_string(), 1, AlphaNeuronState::new(0.8));
    state
        .upper_neurons_mut()
        .get_mut("relu0")
        .expect("upper map should mirror insert")
        .get_mut(&1)
        .expect("upper neuron 1 should exist")
        .set_alpha(0.35);

    let pre_activation = BoundedTensor::new(
        arr1(&[-1.0, -0.5, 0.2]).into_dyn(),
        arr1(&[2.0, 1.0, 1.2]).into_dyn(),
    )
    .unwrap();

    let lower = state.build_alpha_array("relu0", &pre_activation);
    let upper = state.build_alpha_upper_array("relu0", &pre_activation);

    assert!(
        (lower[1] - 0.8).abs() < 1e-6,
        "lower-path alpha should use lower map, got {}",
        lower[1]
    );
    assert!(
        (upper[1] - 0.35).abs() < 1e-6,
        "upper-path alpha should use upper map, got {}",
        upper[1]
    );
    assert!(
        (upper[2] - 1.0).abs() < 1e-6,
        "stable positive neuron should still use slope 1.0, got {}",
        upper[2]
    );
}

#[test]
fn graph_domain_alpha_adam_step_basic() {
    // Test that a single Adam step moves alpha in the gradient direction,
    // returns the correct max gradient magnitude, and updates optimizer state.
    let mut state = GraphDomainAlphaState::empty();
    state.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.5));
    state.insert("relu0".to_string(), 1, AlphaNeuronState::new(0.3));

    // Set gradients: positive for neuron 0, negative for neuron 1
    state
        .neuron_mut("relu0", 0)
        .expect("neuron 0 just inserted")
        .grad = 1.0;
    state
        .neuron_mut("relu0", 1)
        .expect("neuron 1 just inserted")
        .grad = -0.5;

    let config = AdaptiveOptConfig::default();
    let max_grad = state.gradient_step_adam(&config, 1);

    // max_grad should reflect the larger raw gradient magnitude (#2416).
    // grad for neuron 0 = 1.0, grad for neuron 1 = -0.5 (no clipping, grad_clip=0.0).
    // So max_grad = max(|1.0|, |-0.5|) = 1.0
    assert!(
        (max_grad - 1.0).abs() < 1e-5,
        "max_grad should be ~1.0 (raw gradient of larger neuron), got {max_grad}"
    );

    // Neuron 0 (positive grad) should increase alpha
    let n0 = state
        .neuron("relu0", 0)
        .expect("neuron 0 should exist after adam step");
    assert!(
        n0.alpha > 0.5,
        "positive gradient should increase alpha, got {}",
        n0.alpha
    );
    // Verify optimizer state was updated (not just alpha)
    // adam_m = (1 - beta1) * grad = 0.1 * 1.0 = 0.1
    assert!(
        (n0.adam_m - 0.1).abs() < 1e-6,
        "adam_m should be 0.1, got {}",
        n0.adam_m
    );
    // adam_v = (1 - beta2) * grad^2 = 0.001 * 1.0 = 0.001
    assert!(
        (n0.adam_v - 0.001).abs() < 1e-6,
        "adam_v should be 0.001, got {}",
        n0.adam_v
    );

    // Neuron 1 (negative grad) should decrease alpha
    let n1 = state
        .neuron("relu0", 1)
        .expect("neuron 1 should exist after adam step");
    assert!(
        n1.alpha < 0.3,
        "negative gradient should decrease alpha, got {}",
        n1.alpha
    );
    // adam_m for negative gradient: 0.1 * (-0.5) = -0.05
    assert!(
        (n1.adam_m - (-0.05)).abs() < 1e-6,
        "adam_m should be -0.05, got {}",
        n1.adam_m
    );

    // Both alphas must remain in [0, 1]
    assert!(
        (0.0..=1.0).contains(&n0.alpha),
        "alpha must be in [0,1], got {}",
        n0.alpha
    );
    assert!(
        (0.0..=1.0).contains(&n1.alpha),
        "alpha must be in [0,1], got {}",
        n1.alpha
    );
}

#[test]
fn graph_domain_alpha_adam_grad_clipping() {
    // Verify gradient clipping by checking that the internal first moment (adam_m)
    // is bounded. Adam normalizes m_hat/sqrt(v_hat), so the alpha update is
    // approximately lr*sign(grad) regardless of magnitude on step 1. The meaningful
    // effect of clipping is on the optimizer state (adam_m, adam_v), not the first
    // step's alpha change.
    let mut state = GraphDomainAlphaState::empty();
    state.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.5));

    // Set a very large gradient
    state
        .neuron_mut("relu0", 0)
        .expect("neuron just inserted")
        .grad = 1000.0;

    let config = AdaptiveOptConfig {
        grad_clip: 1.0,
        ..AdaptiveOptConfig::default()
    };

    let max_grad_clipped = state.gradient_step_adam(&config, 1);
    let n_clipped = state
        .neuron("relu0", 0)
        .expect("neuron should exist after adam step");
    let a0 = n_clipped.alpha;

    // With grad clipped to 1.0, the update should be modest
    assert!(
        a0.is_finite(),
        "alpha must be finite after clipped gradient"
    );
    assert!(
        (0.0..=1.0).contains(&a0),
        "alpha must be in [0,1], got {a0}"
    );

    // adam_m should reflect the CLIPPED gradient, not the original 1000.0
    // adam_m = (1-beta1) * clipped_grad = 0.1 * 1.0 = 0.1
    assert!(
        (n_clipped.adam_m - 0.1).abs() < 1e-5,
        "adam_m should reflect clipped gradient (0.1), got {}",
        n_clipped.adam_m
    );
    // adam_v should reflect clipped_grad^2 = 0.001 * 1.0 = 0.001
    assert!(
        (n_clipped.adam_v - 0.001).abs() < 1e-6,
        "adam_v should reflect clipped gradient squared (0.001), got {}",
        n_clipped.adam_v
    );

    // Compare with unclipped: adam_m should be 1000x larger
    let mut state2 = GraphDomainAlphaState::empty();
    state2.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.5));
    state2
        .neuron_mut("relu0", 0)
        .expect("neuron just inserted")
        .grad = 1000.0;
    let config2 = AdaptiveOptConfig {
        grad_clip: 0.0, // disabled
        ..AdaptiveOptConfig::default()
    };
    let max_grad_unclipped = state2.gradient_step_adam(&config2, 1);
    let n_unclipped = state2
        .neuron("relu0", 0)
        .expect("neuron should exist after adam step");

    // Unclipped adam_m = 0.1 * 1000 = 100 — 1000x larger
    assert!(
        n_unclipped.adam_m > n_clipped.adam_m * 500.0,
        "unclipped adam_m ({}) should be ~1000x larger than clipped ({})",
        n_unclipped.adam_m,
        n_clipped.adam_m
    );

    // max_grad (raw gradient, #2416) should also differ substantially
    assert!(
        max_grad_unclipped > max_grad_clipped * 500.0,
        "unclipped max_grad ({max_grad_unclipped}) should be ~1000x larger than clipped ({max_grad_clipped})"
    );
}

#[test]
fn graph_domain_alpha_adam_amsgrad() {
    let mut state = GraphDomainAlphaState::empty();
    state.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.5));

    let config = AdaptiveOptConfig {
        amsgrad: true,
        ..AdaptiveOptConfig::default()
    };

    // Step 1: large gradient
    state
        .neuron_mut("relu0", 0)
        .expect("neuron just inserted")
        .grad = 5.0;
    let _ = state.gradient_step_adam(&config, 1);
    let v_max_after_step1 = state
        .neuron("relu0", 0)
        .expect("neuron should exist after adam step")
        .adam_v_max;
    assert!(
        v_max_after_step1 > 0.0,
        "v_max should be positive after first step"
    );

    // Step 2: small gradient — v_max should NOT decrease (AMSGrad property)
    state
        .neuron_mut("relu0", 0)
        .expect("neuron should exist after step 1")
        .grad = 0.01;
    let _ = state.gradient_step_adam(&config, 2);
    let v_max_after_step2 = state
        .neuron("relu0", 0)
        .expect("neuron should exist after adam step")
        .adam_v_max;
    assert!(
        v_max_after_step2 >= v_max_after_step1,
        "AMSGrad: v_max must not decrease (was {v_max_after_step1}, now {v_max_after_step2})"
    );
}

#[test]
fn graph_domain_alpha_adam_nan_gradient() {
    let mut state = GraphDomainAlphaState::empty();
    state.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.5));
    state
        .neuron_mut("relu0", 0)
        .expect("neuron just inserted")
        .grad = f32::NAN;

    let config = AdaptiveOptConfig::default();
    let _ = state.gradient_step_adam(&config, 1);

    let a0 = state.alpha("relu0", 0);
    assert!(
        a0.is_finite(),
        "alpha must remain finite after NaN gradient, got {a0}"
    );
    assert!(
        (a0 - 0.5).abs() <= f32::EPSILON,
        "NaN gradient should produce NaN update → sanitize_alpha → 0.5, got {a0}"
    );
}

#[test]
fn domain_alpha_get_alpha_clamps_above_one() {
    // Acceptance criterion: set alpha to 1.5, verify get_alpha returns 1.0.
    // DomainAlphaState::get_alpha must sanitize the raw field value.
    let mut state = DomainAlphaState::empty();
    state.neurons.insert(
        (1, 0),
        AlphaNeuronState {
            alpha: 1.5, // Bypasses set_alpha — raw field write
            grad: 0.0,
            velocity: 0.0,
            adam_m: 0.0,
            adam_v: 0.0,
            adam_v_max: 0.0,
        },
    );

    let alpha = state.alpha(1, 0);
    assert!(
        (alpha - 1.0).abs() <= f32::EPSILON,
        "alpha > 1.0 must be clamped to 1.0, got {alpha}"
    );
}

#[test]
fn domain_alpha_get_alpha_sanitizes_nan() {
    // Acceptance criterion: set alpha to NaN, verify get_alpha returns 0.5.
    // sanitize_alpha maps NaN to 0.5 (the midpoint of [0, 1]).
    let mut state = DomainAlphaState::empty();
    state.neurons.insert(
        (1, 0),
        AlphaNeuronState {
            alpha: f32::NAN, // Bypasses set_alpha — raw field write
            grad: 0.0,
            velocity: 0.0,
            adam_m: 0.0,
            adam_v: 0.0,
            adam_v_max: 0.0,
        },
    );

    let alpha = state.alpha(1, 0);
    assert!(
        alpha.is_finite(),
        "NaN alpha must be sanitized to finite, got {alpha}"
    );
    assert!(
        (alpha - 0.5).abs() <= f32::EPSILON,
        "NaN alpha must be sanitized to 0.5, got {alpha}"
    );
}

#[test]
fn domain_alpha_get_alpha_clamps_below_zero() {
    // Negative alpha is unsound for different reasons (slope < 0).
    // sanitize_alpha clamps to [0, 1], so -0.5 → 0.0.
    let mut state = DomainAlphaState::empty();
    state.neurons.insert(
        (1, 0),
        AlphaNeuronState {
            alpha: -0.5,
            grad: 0.0,
            velocity: 0.0,
            adam_m: 0.0,
            adam_v: 0.0,
            adam_v_max: 0.0,
        },
    );

    let alpha = state.alpha(1, 0);
    assert!(
        (alpha - 0.0).abs() <= f32::EPSILON,
        "alpha < 0.0 must be clamped to 0.0, got {alpha}"
    );
}

#[test]
fn graph_domain_alpha_adam_no_bias_correction() {
    let mut state = GraphDomainAlphaState::empty();
    state.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.5));
    state
        .neuron_mut("relu0", 0)
        .expect("neuron just inserted")
        .grad = 1.0;

    let config = AdaptiveOptConfig {
        bias_correction: false,
        ..AdaptiveOptConfig::default()
    };

    let max_grad = state.gradient_step_adam(&config, 1);

    // max_grad now returns raw gradient magnitude, not m_hat (#2416).
    // Raw gradient = 1.0, so max_grad = 1.0 regardless of bias_correction.
    assert!(
        (max_grad - 1.0).abs() < 1e-5,
        "max_grad should be ~1.0 (raw gradient magnitude), got {max_grad}"
    );

    let a0 = state.alpha("relu0", 0);
    assert!(
        a0.is_finite() && (0.0..=1.0).contains(&a0),
        "alpha must be valid, got {a0}"
    );
}

/// Regression: beta1=1.0 caused bias_correction1=0, division by zero in DomainAlphaState (#2556).
/// Mirrors the fix in bounds/alpha.rs (#2315).
#[test]
fn domain_alpha_adam_beta1_one_no_div_by_zero() {
    let mut state = DomainAlphaState::empty();
    state.neurons.insert((1, 0), AlphaNeuronState::new(0.5));
    state
        .neurons
        .get_mut(&(1, 0))
        .expect("neuron just inserted")
        .grad = 0.1;

    let config = AdaptiveOptConfig {
        beta1: 1.0,
        bias_correction: true,
        ..AdaptiveOptConfig::default()
    };
    let max_grad = state.gradient_step_adam(&config, 1);

    let alpha = state.alpha(1, 0);
    assert!(
        alpha.is_finite(),
        "alpha must be finite with beta1=1.0, got {alpha}"
    );
    assert!(
        (0.0..=1.0).contains(&alpha),
        "alpha must be in [0, 1], got {alpha}"
    );
    assert!(
        max_grad.is_finite(),
        "max_grad must be finite with beta1=1.0, got {max_grad}"
    );

    // Verify optimizer state is also finite (not Inf from division by zero)
    let n = &state.neurons[&(1, 0)];
    assert!(
        n.adam_m.is_finite(),
        "adam_m must be finite, got {}",
        n.adam_m
    );
}

/// Regression: beta2=1.0 caused bias_correction2=0, division by zero in DomainAlphaState (#2556).
#[test]
fn domain_alpha_adam_beta2_one_no_div_by_zero() {
    let mut state = DomainAlphaState::empty();
    state.neurons.insert((1, 0), AlphaNeuronState::new(0.5));
    state
        .neurons
        .get_mut(&(1, 0))
        .expect("neuron just inserted")
        .grad = 0.1;

    let config = AdaptiveOptConfig {
        beta2: 1.0,
        bias_correction: true,
        ..AdaptiveOptConfig::default()
    };
    let max_grad = state.gradient_step_adam(&config, 1);

    let alpha = state.alpha(1, 0);
    assert!(
        alpha.is_finite(),
        "alpha must be finite with beta2=1.0, got {alpha}"
    );
    assert!(
        (0.0..=1.0).contains(&alpha),
        "alpha must be in [0, 1], got {alpha}"
    );
    assert!(
        max_grad.is_finite(),
        "max_grad must be finite with beta2=1.0, got {max_grad}"
    );

    // Verify optimizer state is also finite (not NaN from 0/0 division)
    let n = &state.neurons[&(1, 0)];
    assert!(
        n.adam_v.is_finite(),
        "adam_v must be finite, got {}",
        n.adam_v
    );
}

/// Regression: beta1=1.0 caused bias_correction1=0, division by zero in GraphDomainAlphaState (#2556).
#[test]
fn graph_domain_alpha_adam_beta1_one_no_div_by_zero() {
    let mut state = GraphDomainAlphaState::empty();
    state.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.5));
    state
        .neuron_mut("relu0", 0)
        .expect("neuron just inserted")
        .grad = 0.1;

    let config = AdaptiveOptConfig {
        beta1: 1.0,
        bias_correction: true,
        ..AdaptiveOptConfig::default()
    };
    let max_grad = state.gradient_step_adam(&config, 1);

    let a0 = state.alpha("relu0", 0);
    assert!(
        a0.is_finite(),
        "alpha must be finite with beta1=1.0, got {a0}"
    );
    assert!(
        (0.0..=1.0).contains(&a0),
        "alpha must be in [0, 1], got {a0}"
    );
    assert!(
        max_grad.is_finite(),
        "max_grad must be finite with beta1=1.0, got {max_grad}"
    );

    // Verify optimizer state is finite (not Inf from division by zero)
    let n = state
        .neuron("relu0", 0)
        .expect("neuron should exist after adam step");
    assert!(
        n.adam_m.is_finite(),
        "adam_m must be finite, got {}",
        n.adam_m
    );
}

/// Regression test for #2939: standard (non-Adam) gradient_step convergence
/// tracking must use nan_propagating_max, not f32::max, to surface NaN gradients.
///
/// Before the fix, `f32::max(0.0, NaN) = 0.0` silently dropped NaN, making the
/// convergence check report 0.0 even when gradients were corrupt.
#[test]
fn domain_alpha_gradient_step_nan_convergence_tracking_2939() {
    let mut state = DomainAlphaState::empty();
    state.neurons.insert((1, 0), AlphaNeuronState::new(0.5));
    state
        .neurons
        .get_mut(&(1, 0))
        .expect("neuron just inserted")
        .grad = f32::NAN;

    let max_grad = state.gradient_step(0.01, 0.9);

    // max_grad must be NaN (nan_propagating_max propagates NaN)
    assert!(
        max_grad.is_nan(),
        "max_grad must be NaN when gradient is NaN, got {max_grad}"
    );

    // Alpha must be finite (sanitize_alpha catches NaN→0.5)
    let alpha = state.alpha(1, 0);
    assert!(
        alpha.is_finite(),
        "alpha must be finite after NaN gradient, got {alpha}"
    );

    // Velocity must be reset to 0.0 (NaN guard from #2608)
    let n = &state.neurons[&(1, 0)];
    assert_eq!(
        n.velocity, 0.0,
        "NaN-corrupted velocity must be reset to 0.0, got {}",
        n.velocity
    );
}

/// Regression test for #2939: after velocity NaN recovery, subsequent gradient
/// steps must work normally (corruption is not permanent).
#[test]
fn domain_alpha_gradient_step_nan_recovery_then_normal_2939() {
    let mut state = DomainAlphaState::empty();
    state.neurons.insert((1, 0), AlphaNeuronState::new(0.5));

    // Step 1: NaN gradient corrupts velocity
    state
        .neurons
        .get_mut(&(1, 0))
        .expect("neuron just inserted")
        .grad = f32::NAN;
    let _ = state.gradient_step(0.01, 0.9);

    // After recovery: velocity=0.0, alpha=0.5 (sanitized)
    let n = &state.neurons[&(1, 0)];
    assert_eq!(n.velocity, 0.0);

    // Step 2: valid gradient — should update normally
    state
        .neurons
        .get_mut(&(1, 0))
        .expect("neuron just inserted")
        .grad = 1.0;
    let max_grad = state.gradient_step(0.1, 0.9);

    assert!(
        max_grad.is_finite(),
        "max_grad should be finite with valid gradient, got {max_grad}"
    );
    let n = &state.neurons[&(1, 0)];
    assert!(
        n.velocity.is_finite(),
        "velocity should be finite after recovery, got {}",
        n.velocity
    );
}

/// Regression: beta2=1.0 caused bias_correction2=0, division by zero in GraphDomainAlphaState (#2556).
#[test]
fn graph_domain_alpha_adam_beta2_one_no_div_by_zero() {
    let mut state = GraphDomainAlphaState::empty();
    state.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.5));
    state
        .neuron_mut("relu0", 0)
        .expect("neuron just inserted")
        .grad = 0.1;

    let config = AdaptiveOptConfig {
        beta2: 1.0,
        bias_correction: true,
        ..AdaptiveOptConfig::default()
    };
    let max_grad = state.gradient_step_adam(&config, 1);

    let a0 = state.alpha("relu0", 0);
    assert!(
        a0.is_finite(),
        "alpha must be finite with beta2=1.0, got {a0}"
    );
    assert!(
        (0.0..=1.0).contains(&a0),
        "alpha must be in [0, 1], got {a0}"
    );
    assert!(
        max_grad.is_finite(),
        "max_grad must be finite with beta2=1.0, got {max_grad}"
    );

    // Verify optimizer state is finite (not NaN from 0/0 division)
    let n = state
        .neuron("relu0", 0)
        .expect("neuron should exist after adam step");
    assert!(
        n.adam_v.is_finite(),
        "adam_v must be finite, got {}",
        n.adam_v
    );
}

/// #hard-six unshared-α: a persisted (lower-stepped) snapshot must restore
/// the `lower == upper` invariant via `sync_upper_from_lower`, so children
/// prepped from it extract plain `Activation` layers (not the wide-lane
/// unbatchable `ActivationReluDualAlpha`). Also: neurons absent from the
/// lower map keep their upper value, and the synced value stays clamped.
#[test]
fn graph_domain_alpha_sync_upper_from_lower_restores_invariant() {
    let mut state = GraphDomainAlphaState::empty();
    state.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.5));
    state.insert("relu0".to_string(), 1, AlphaNeuronState::new(0.25));

    // Simulate the wide ascent: step the LOWER path only (the upper map
    // receives zero gradients and stays put) — the exact divergence the
    // persistence lane snapshots.
    state
        .neuron_mut("relu0", 0)
        .expect("neuron 0 inserted")
        .grad = -1.0; // ascent direction: alpha moves up
    let _ = state.gradient_step_adam(&AdaptiveOptConfig::default(), 1);
    let lower0 = state.alpha("relu0", 0);
    assert_ne!(
        lower0,
        state.alpha_upper("relu0", 0),
        "precondition: lower-only step must diverge the pair"
    );

    // Extra upper-only entry: must survive the sync untouched.
    state
        .upper_neurons_mut()
        .get_mut("relu0")
        .expect("node map exists")
        .insert(7, AlphaNeuronState::new(0.75));

    state.sync_upper_from_lower();

    assert_eq!(
        state.alpha_upper("relu0", 0),
        lower0,
        "stepped neuron: upper must equal the stepped lower"
    );
    assert_eq!(
        state.alpha_upper("relu0", 1),
        state.alpha("relu0", 1),
        "unstepped neuron: pair must remain equal"
    );
    assert_eq!(
        state.alpha_upper("relu0", 7),
        0.75,
        "upper-only neuron must be untouched"
    );
    for m in state.upper_neurons().values() {
        for n in m.values() {
            let a = n.alpha();
            assert!((0.0..=1.0).contains(&a), "upper alpha clamped, got {a}");
        }
    }
}
