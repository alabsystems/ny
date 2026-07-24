// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for GraphAlphaState.

use super::checked_bounds;
use crate::bounds::GraphAlphaState;
use ndarray::array;

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_new() {
    let state = GraphAlphaState::new();

    assert!(state.alphas.is_empty());
    assert!(state.unstable_mask.is_empty());
    assert!(state.velocity.is_empty());
    assert_eq!(state.num_unstable(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_default() {
    let state = GraphAlphaState::default();
    assert!(state.alphas.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_add_relu_node_all_positive() {
    let mut state = GraphAlphaState::new();

    let bounds = checked_bounds(
        array![1.0_f32, 2.0].into_dyn(),
        array![3.0_f32, 4.0].into_dyn(),
    );

    state.add_relu_node("relu1", &bounds, false).unwrap();

    assert_eq!(state.num_unstable(), 0);
    assert_eq!(
        state.alpha("relu1").unwrap().as_slice().unwrap(),
        &[1.0, 1.0]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_add_relu_node_mixed() {
    let mut state = GraphAlphaState::new();

    let bounds = checked_bounds(
        array![1.0_f32, -3.0, -1.0].into_dyn(), // positive, negative, crossing
        array![2.0_f32, -1.0, 2.0].into_dyn(),
    );

    state.add_relu_node("relu1", &bounds, false).unwrap();

    assert_eq!(state.num_unstable(), 1);

    let alpha = state.alpha("relu1").unwrap();
    assert_eq!(alpha[0], 1.0); // Positive
    assert_eq!(alpha[1], 0.0); // Negative
    assert_eq!(alpha[2], 1.0); // Crossing with u=2 > -l=1
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_add_multiple_nodes() {
    let mut state = GraphAlphaState::new();

    state
        .add_relu_node(
            "relu1",
            &checked_bounds(array![-1.0_f32].into_dyn(), array![1.0_f32].into_dyn()),
            false,
        )
        .unwrap();

    state
        .add_relu_node(
            "relu2",
            &checked_bounds(
                array![-1.0_f32, -1.0].into_dyn(),
                array![1.0_f32, 1.0].into_dyn(),
            ),
            false,
        )
        .unwrap();

    assert_eq!(state.num_unstable(), 3); // 1 + 2

    let nodes: Vec<&str> = state.relu_nodes().collect();
    assert_eq!(nodes.len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_get_alpha_missing() {
    let state = GraphAlphaState::new();
    assert!(state.alpha("nonexistent").is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_update() {
    let mut state = GraphAlphaState::new();

    // Use asymmetric bounds where u > -l, so alpha initializes to 1
    state
        .add_relu_node(
            "relu1",
            &checked_bounds(
                array![-1.0_f32, -1.0].into_dyn(),
                array![2.0_f32, 2.0].into_dyn(), // u=2 > -l=1
            ),
            false,
        )
        .unwrap();

    // Both neurons unstable with alpha=1 (u > -l)
    assert_eq!(state.alpha("relu1").unwrap()[0], 1.0);
    assert_eq!(state.alpha("relu1").unwrap()[1], 1.0);

    let gradient = array![0.5_f32, 0.5_f32];
    state.update("relu1", &gradient, 0.1, 0.0);

    let alpha = state.alpha("relu1").unwrap();
    // alpha -= lr * gradient = 1.0 - 0.1 * 0.5 = 0.95
    assert!((alpha[0] - 0.95).abs() < 1e-6);
    assert!((alpha[1] - 0.95).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_update_missing_node() {
    let mut state = GraphAlphaState::new();

    // Should not panic, just silently return
    let gradient = array![1.0_f32];
    state.update("nonexistent", &gradient, 0.1, 0.0);

    assert_eq!(state.num_unstable(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_update_with_momentum() {
    let mut state = GraphAlphaState::new();

    // Use asymmetric bounds where u > -l, so alpha initializes to 1
    state
        .add_relu_node(
            "relu1",
            &checked_bounds(
                array![-1.0_f32].into_dyn(),
                array![2.0_f32].into_dyn(), // u=2 > -l=1
            ),
            false,
        )
        .unwrap();

    // Alpha starts at 1.0 (u > -l)
    assert_eq!(state.alpha("relu1").unwrap()[0], 1.0);

    // First update
    let gradient = array![0.5_f32];
    state.update("relu1", &gradient, 0.1, 0.9);

    // vel = 0.9 * 0 - 0.1 * 0.5 = -0.05
    // alpha = 1.0 + (-0.05) = 0.95
    let alpha1 = state.alpha("relu1").unwrap()[0];
    assert!((alpha1 - 0.95).abs() < 1e-6);

    // Second update - momentum should accumulate
    state.update("relu1", &gradient, 0.1, 0.9);

    // vel = 0.9 * (-0.05) - 0.1 * 0.5 = -0.045 - 0.05 = -0.095
    // alpha = 0.95 + (-0.095) = 0.855
    let alpha2 = state.alpha("relu1").unwrap()[0];
    assert!((alpha2 - 0.855).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_relu_nodes_iterator() {
    let mut state = GraphAlphaState::new();

    state
        .add_relu_node(
            "a_relu",
            &checked_bounds(array![-1.0_f32].into_dyn(), array![1.0_f32].into_dyn()),
            false,
        )
        .unwrap();
    state
        .add_relu_node(
            "b_relu",
            &checked_bounds(array![-1.0_f32].into_dyn(), array![1.0_f32].into_dyn()),
            false,
        )
        .unwrap();

    let mut nodes: Vec<&str> = state.relu_nodes().collect();
    nodes.sort_unstable();

    assert_eq!(nodes, vec!["a_relu", "b_relu"]);
}

/// #4404: channel_only_alpha=true reduces [C,H,W] alpha to [C] by worst-case per channel.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_channel_only_alpha_reduces_to_channels_4404() {
    let mut state = GraphAlphaState::new();

    // Shape [2, 2, 2] = 2 channels, 2x2 spatial.
    // Channel 0: lower=[-1,-2, 1, 1], upper=[3, 1, 2, 2] — but per channel:
    //   ch0 spatial: lower=[-1,-2], upper=[3,1] → min_lower=-2, max_upper=3 → unstable
    //   ch1 spatial: lower=[1, 1], upper=[2, 2] → min_lower=1, max_upper=2 → stable (positive)
    let lower = ndarray::arr1(&[-1.0_f32, -2.0, 1.0, 1.0]);
    let upper = ndarray::arr1(&[3.0_f32, 1.0, 2.0, 2.0]);
    let bounds = checked_bounds(
        lower
            .into_shape_with_order(ndarray::IxDyn(&[2, 2, 1]))
            .unwrap(),
        upper
            .into_shape_with_order(ndarray::IxDyn(&[2, 2, 1]))
            .unwrap(),
    );

    state.add_relu_node("conv_relu", &bounds, true).unwrap();

    // Alpha should have length 2 (channels), not 4 (neurons)
    let alpha = state.alpha("conv_relu").unwrap();
    assert_eq!(alpha.len(), 2, "channel-only alpha should have C elements");

    // Channel 0 is unstable: max_upper=3 > -min_lower=2, so alpha=1
    assert_eq!(alpha[0], 1.0, "ch0 unstable, u>-l => alpha=1");
    // Channel 1 is positive (stable): alpha=1 (positive region)
    assert_eq!(alpha[1], 1.0, "ch1 all positive => alpha=1");

    // spatial_shapes should record the original shape
    assert_eq!(
        state.spatial_shape("conv_relu"),
        Some([2, 2, 1].as_slice()),
        "spatial_shape should record original [C,H,W]"
    );
}

/// #4404: channel_only_alpha=false (full_conv_alpha=true default) keeps per-neuron alpha.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_full_conv_alpha_identity_4404() {
    let mut state_full = GraphAlphaState::new();
    let mut state_channel = GraphAlphaState::new();

    let lower = ndarray::arr1(&[-1.0_f32, -2.0, 1.0, 1.0]);
    let upper = ndarray::arr1(&[3.0_f32, 1.0, 2.0, 2.0]);
    let bounds = checked_bounds(
        lower
            .into_shape_with_order(ndarray::IxDyn(&[2, 2, 1]))
            .unwrap(),
        upper
            .into_shape_with_order(ndarray::IxDyn(&[2, 2, 1]))
            .unwrap(),
    );

    state_full.add_relu_node("relu", &bounds, false).unwrap();
    state_channel.add_relu_node("relu", &bounds, true).unwrap();

    // Full: 4 alpha elements. Channel: 2 alpha elements.
    assert_eq!(state_full.alpha("relu").unwrap().len(), 4);
    assert_eq!(state_channel.alpha("relu").unwrap().len(), 2);

    // Full has no spatial_shape entry
    assert!(state_full.spatial_shape("relu").is_none());
    // Channel has spatial_shape entry
    assert!(state_channel.spatial_shape("relu").is_some());
}

/// #4404: expand_alpha broadcasts [C] to [C*H*W] correctly.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_expand_alpha_broadcast_4404() {
    let mut state = GraphAlphaState::new();

    // 3 channels, 2x1 spatial = 6 neurons
    let lower = ndarray::Array::from_elem(ndarray::IxDyn(&[3, 2, 1]), -1.0_f32);
    let upper = ndarray::Array::from_elem(ndarray::IxDyn(&[3, 2, 1]), 2.0_f32);
    let bounds = checked_bounds(lower, upper);

    state.add_relu_node("relu", &bounds, true).unwrap();
    let alpha = state.alpha("relu").unwrap();
    assert_eq!(alpha.len(), 3); // C=3

    let expanded = state.expand_alpha("relu", alpha);
    assert_eq!(expanded.len(), 6); // C*H*W = 3*2*1
                                   // Each channel's alpha is repeated across spatial positions
    assert_eq!(expanded[0], alpha[0]); // ch0, pos0
    assert_eq!(expanded[1], alpha[0]); // ch0, pos1
    assert_eq!(expanded[2], alpha[1]); // ch1, pos0
    assert_eq!(expanded[3], alpha[1]); // ch1, pos1
    assert_eq!(expanded[4], alpha[2]); // ch2, pos0
    assert_eq!(expanded[5], alpha[2]); // ch2, pos1
}

/// #4404: reduce_gradient sums spatial dims per channel.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_reduce_gradient_sums_spatial_4404() {
    let mut state = GraphAlphaState::new();

    // 2 channels, 3x1 spatial = 6 neurons
    let lower = ndarray::Array::from_elem(ndarray::IxDyn(&[2, 3, 1]), -1.0_f32);
    let upper = ndarray::Array::from_elem(ndarray::IxDyn(&[2, 3, 1]), 2.0_f32);
    let bounds = checked_bounds(lower, upper);

    state.add_relu_node("relu", &bounds, true).unwrap();

    // Full gradient: [1,2,3, 4,5,6] for 2 channels x 3 spatial
    let gradient = ndarray::array![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let reduced = state.reduce_gradient("relu", &gradient);
    assert_eq!(reduced.len(), 2);
    assert!((reduced[0] - 6.0).abs() < 1e-6, "ch0: 1+2+3=6");
    assert!((reduced[1] - 15.0).abs() < 1e-6, "ch1: 4+5+6=15");
}

/// #4404: reduce_gradient is a no-op for already channel-sized gradients.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_reduce_gradient_channel_sized_noop_4404() {
    let mut state = GraphAlphaState::new();

    let lower = ndarray::Array::from_elem(ndarray::IxDyn(&[2, 3, 1]), -1.0_f32);
    let upper = ndarray::Array::from_elem(ndarray::IxDyn(&[2, 3, 1]), 2.0_f32);
    let bounds = checked_bounds(lower, upper);

    state.add_relu_node("relu", &bounds, true).unwrap();

    let channel_grad = ndarray::array![1.25_f32, -0.75];
    let reduced = state.reduce_gradient("relu", &channel_grad);
    assert_eq!(reduced, channel_grad);
}

/// #4404: reduce_gradient returns zeros when gradient length mismatches C*H*W.
/// This happens when AnalyticChain fails and returns a 0-length or wrong-length gradient.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_reduce_gradient_length_mismatch_returns_zeros_4404() {
    let mut state = GraphAlphaState::new();

    // 2 channels, 3x1 spatial = 6 neurons expected
    let lower = ndarray::Array::from_elem(ndarray::IxDyn(&[2, 3, 1]), -1.0_f32);
    let upper = ndarray::Array::from_elem(ndarray::IxDyn(&[2, 3, 1]), 2.0_f32);
    let bounds = checked_bounds(lower, upper);

    state.add_relu_node("relu", &bounds, true).unwrap();

    // Zero-length gradient (AnalyticChain missing A + pre-ReLU bounds)
    let empty_grad = ndarray::Array1::<f32>::zeros(0);
    let reduced = state.reduce_gradient("relu", &empty_grad);
    assert_eq!(reduced.len(), 2, "should return per-channel zeros");
    assert!((reduced[0]).abs() < 1e-10);
    assert!((reduced[1]).abs() < 1e-10);

    // Wrong-length gradient (partial intermediate storage)
    let short_grad = ndarray::array![1.0_f32, 2.0, 3.0];
    let reduced = state.reduce_gradient("relu", &short_grad);
    assert_eq!(
        reduced.len(),
        2,
        "should return per-channel zeros on length mismatch"
    );
    assert!((reduced[0]).abs() < 1e-10);
}

/// #4404: expand/reduce round-trip preserves gradient signal.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_expand_reduce_roundtrip_4404() {
    let mut state = GraphAlphaState::new();

    // 2 channels, 2x2 spatial
    let lower = ndarray::Array::from_elem(ndarray::IxDyn(&[2, 2, 2]), -1.0_f32);
    let upper = ndarray::Array::from_elem(ndarray::IxDyn(&[2, 2, 2]), 2.0_f32);
    let bounds = checked_bounds(lower, upper);

    state.add_relu_node("relu", &bounds, true).unwrap();
    let alpha = state.alpha("relu").unwrap().clone();

    // Expand then reduce: reduce(expand(alpha)) should give alpha * spatial_size
    let expanded = state.expand_alpha("relu", &alpha);
    let reduced = state.reduce_gradient("relu", &expanded);
    // Each channel alpha is repeated 4 times, so sum = alpha * 4
    assert!((reduced[0] - alpha[0] * 4.0).abs() < 1e-6);
    assert!((reduced[1] - alpha[1] * 4.0).abs() < 1e-6);
}

/// #4404: no-op expand/reduce for non-channel-only nodes.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_expand_noop_for_full_alpha_4404() {
    let mut state = GraphAlphaState::new();

    let bounds = checked_bounds(
        ndarray::array![-1.0_f32, -2.0, 1.0].into_dyn(),
        ndarray::array![2.0_f32, 1.0, 3.0].into_dyn(),
    );

    state.add_relu_node("relu", &bounds, false).unwrap();
    let alpha = state.alpha("relu").unwrap();

    // expand_alpha should be identity (clone) for non-channel-only nodes
    let expanded = state.expand_alpha("relu", alpha);
    assert_eq!(expanded, *alpha);

    // reduce_gradient should also be identity
    let grad = ndarray::array![0.5_f32, 1.0, 0.3];
    let reduced = state.reduce_gradient("relu", &grad);
    assert_eq!(reduced, grad);
}

/// #4404: channel_only_alpha with ndim < 3 falls back to full alpha.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_channel_only_skipped_for_1d_4404() {
    let mut state = GraphAlphaState::new();

    // 1D input (FC layer output): channel_only_alpha=true should still use full alpha
    let bounds = checked_bounds(
        ndarray::array![-1.0_f32, -2.0].into_dyn(),
        ndarray::array![2.0_f32, 1.0].into_dyn(),
    );

    state.add_relu_node("relu", &bounds, true).unwrap();

    // Should be full alpha (2 elements), not channel-only
    assert_eq!(state.alpha("relu").unwrap().len(), 2);
    assert!(state.spatial_shape("relu").is_none());
}

/// #4404: channel-only alpha + reduce_gradient + update_adam pipeline test.
/// Verifies that the optimizer actually updates channel-only alpha when gradients
/// are per-neuron (the exact bug fixed in W3 iter 121 — previously update_adam
/// silently skipped all conv alpha updates due to length mismatch).
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_channel_only_optimizer_pipeline_4404() {
    let mut state = GraphAlphaState::new();

    // 2 channels, 3x1 spatial = 6 neurons
    let lower = ndarray::Array::from_elem(ndarray::IxDyn(&[2, 3, 1]), -1.0_f32);
    let upper = ndarray::Array::from_elem(ndarray::IxDyn(&[2, 3, 1]), 2.0_f32);
    let bounds = checked_bounds(lower, upper);

    state.add_relu_node("relu", &bounds, true).unwrap();
    let alpha_before = state.alpha("relu").unwrap().clone();
    assert_eq!(alpha_before.len(), 2); // channel-only

    // Simulate a per-neuron gradient (length 6) from backward pass
    let full_gradient = ndarray::array![0.5_f32, 0.3, 0.2, 0.4, 0.6, 0.1];

    // Reduce to per-channel, then update
    let reduced = state.reduce_gradient("relu", &full_gradient);
    assert_eq!(reduced.len(), 2);

    let params = crate::bounds::alpha::AdamParams::new(0.01, 1);
    state.update_adam("relu", &reduced, &params);

    let alpha_after = state.alpha("relu").unwrap();
    // Alpha MUST have changed (this was the bug: silent skip due to length mismatch)
    assert_ne!(
        alpha_after.as_slice().unwrap(),
        alpha_before.as_slice().unwrap(),
        "channel-only alpha must be updated after reduce_gradient + update_adam"
    );
}

/// Regression test for #1937: GraphAlphaState gradient-length mismatch must not panic.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_state_update_gradient_length_mismatch_no_panic_1937() {
    let mut state = GraphAlphaState::new();

    // 4 unstable neurons
    state
        .add_relu_node(
            "relu1",
            &checked_bounds(
                array![-1.0_f32, -0.5, -2.0, -0.3].into_dyn(),
                array![1.0_f32, 0.5, 2.0, 0.3].into_dyn(),
            ),
            false,
        )
        .unwrap();
    assert_eq!(state.alpha("relu1").unwrap().len(), 4);
    let alpha_before = state.alpha("relu1").unwrap().clone();

    // Wrong length gradient (1 vs 4) — the #1937 bug scenario
    let wrong_gradient = array![0.5_f32];
    state.update("relu1", &wrong_gradient, 0.1, 0.0);

    // Alpha unchanged — guard skipped the update
    assert_eq!(*state.alpha("relu1").unwrap(), alpha_before);

    // Also test Adam path
    let params = crate::bounds::alpha::AdamParams::new(0.01, 1);
    state.update_adam("relu1", &wrong_gradient, &params);
    assert_eq!(*state.alpha("relu1").unwrap(), alpha_before);
}
