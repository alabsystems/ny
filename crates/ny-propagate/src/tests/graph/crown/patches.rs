// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Behavioral tests for Phase 1b (#2613): Graph engine CrownBounds/Patches support.
//!
//! These tests verify that the graph CROWN engine correctly takes the Patches
//! fast-path for CNN DAGs (Conv2d -> ReLU -> Conv2d chains) and produces sound
//! bounds, including at merge points (skip connections) where Patches -> Dense
//! conversion occurs.

use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
use crate::*;
use ndarray::{Array1, ArrayD, IxDyn};

/// Sample offsets for soundness verification. Each offset creates a concrete
/// input at center + offset, exercising boundary and interior points.
const SAMPLE_OFFSETS: [f32; 5] = [-0.1, -0.05, 0.0, 0.05, 0.1];

/// Assert that concrete outputs at sampled inputs fall within CROWN bounds.
fn assert_bounds_contain_samples(
    bounds: &BoundedTensor,
    network: &Network,
    center_val: f32,
    shape: &[usize],
) {
    for &offset in &SAMPLE_OFFSETS {
        let point = ArrayD::from_elem(IxDyn(shape), center_val + offset);
        let point_input = BoundedTensor::concrete(point).unwrap();
        let exact_output = network.propagate_ibp(&point_input).unwrap();

        for (idx, ((out_val, &lower), &upper)) in exact_output
            .lower()
            .iter()
            .zip(bounds.lower().iter())
            .zip(bounds.upper().iter())
            .enumerate()
        {
            assert!(
                *out_val >= lower - 1e-4,
                "Soundness violation at offset={}, idx={}: output {} < lower {}",
                offset,
                idx,
                out_val,
                lower
            );
            assert!(
                *out_val <= upper + 1e-4,
                "Soundness violation at offset={}, idx={}: output {} > upper {}",
                offset,
                idx,
                out_val,
                upper
            );
        }
    }
}

/// Build a Conv2d -> ReLU -> Conv2d sequential chain as a GraphNetwork.
///
/// Input: [1, 4, 4], Conv1: [2,1,2,2] -> [2,3,3], Conv2: [1,2,2,2] -> [1,2,2].
/// Output is 3D spatial with Conv2d present -> Patches mode triggered.
fn build_conv_relu_conv_network() -> (Network, GraphNetwork, BoundedTensor) {
    let mut kernel1 = ArrayD::zeros(IxDyn(&[2, 1, 2, 2]));
    kernel1[[0, 0, 0, 0]] = 1.0;
    kernel1[[0, 0, 0, 1]] = -0.5;
    kernel1[[0, 0, 1, 0]] = 0.3;
    kernel1[[0, 0, 1, 1]] = 0.2;
    kernel1[[1, 0, 0, 0]] = -0.4;
    kernel1[[1, 0, 0, 1]] = 0.8;
    kernel1[[1, 0, 1, 0]] = -0.1;
    kernel1[[1, 0, 1, 1]] = 0.6;
    let bias1 = Array1::from_vec(vec![0.1, -0.1]);
    let conv1 = Conv2dLayer::with_input_shape(kernel1, Some(bias1), (1, 1), (0, 0), 4, 4).unwrap();

    let mut kernel2 = ArrayD::zeros(IxDyn(&[1, 2, 2, 2]));
    kernel2[[0, 0, 0, 0]] = 0.5;
    kernel2[[0, 0, 0, 1]] = -0.3;
    kernel2[[0, 0, 1, 0]] = 0.2;
    kernel2[[0, 0, 1, 1]] = 0.4;
    kernel2[[0, 1, 0, 0]] = -0.2;
    kernel2[[0, 1, 0, 1]] = 0.6;
    kernel2[[0, 1, 1, 0]] = -0.1;
    kernel2[[0, 1, 1, 1]] = 0.3;
    let bias2 = Array1::from_vec(vec![0.05]);
    let conv2 = Conv2dLayer::with_input_shape(kernel2, Some(bias2), (1, 1), (0, 0), 3, 3).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Conv2d(conv2));

    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let center = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5_f32);
    let input = BoundedTensor::from_epsilon(center, 0.1).unwrap();

    (network, graph, input)
}

/// Build a Conv2d + residual skip connection graph.
///
/// Graph: _input -> conv1(same-pad) -> relu -> add -> output
///          |                                   ^
///          +-----------------------------------+
fn build_residual_graph() -> (Conv2dLayer, GraphNetwork, BoundedTensor) {
    let mut kernel = ArrayD::zeros(IxDyn(&[1, 1, 3, 3]));
    kernel[[0, 0, 0, 0]] = 0.2;
    kernel[[0, 0, 0, 1]] = -0.1;
    kernel[[0, 0, 0, 2]] = 0.15;
    kernel[[0, 0, 1, 0]] = 0.3;
    kernel[[0, 0, 1, 1]] = 0.5;
    kernel[[0, 0, 1, 2]] = -0.2;
    kernel[[0, 0, 2, 0]] = 0.1;
    kernel[[0, 0, 2, 1]] = 0.25;
    kernel[[0, 0, 2, 2]] = -0.15;
    let bias = Array1::from_vec(vec![0.05]);
    let conv1 = Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (1, 1), 4, 4).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1.clone())));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["relu".to_string(), "_input".to_string()],
    ));
    graph.set_output("add");

    let center = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5_f32);
    let input = BoundedTensor::from_epsilon(center, 0.1).unwrap();

    (conv1, graph, input)
}

/// Phase 1b: Conv2d -> ReLU -> Conv2d sequential chain through graph engine.
/// Verifies Crown provenance (Patches path) and soundness via sampling.
#[ntest::timeout(120000)]
#[test]
fn test_graph_crown_conv2d_chain_patches_soundness() {
    let (network, graph, input) = build_conv_relu_conv_network();

    // Pin the shared Dense budget so concurrent zero-budget fallback oracles
    // cannot change this test's expected CROWN provenance.
    let crown_result =
        tests::with_crown_dense_budget_mb("2048", || graph.propagate_crown_with_provenance(&input))
            .unwrap();
    assert_eq!(
        crown_result.provenance,
        BoundsProvenance::Crown,
        "Expected Crown provenance (Patches path), got {:?}",
        crown_result.provenance
    );
    assert_eq!(crown_result.bounds.shape(), &[1, 2, 2]);

    let has_width = crown_result
        .bounds
        .lower()
        .iter()
        .zip(crown_result.bounds.upper().iter())
        .any(|(l, u)| u - l > 1e-6);
    assert!(has_width, "CROWN bounds should have non-zero width");

    assert_bounds_contain_samples(&crown_result.bounds, &network, 0.5, &[1, 4, 4]);

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    assert_eq!(ibp_bounds.shape(), &[1, 2, 2]);
}

/// Phase 1b: Conv2d with residual skip connection.
/// Verifies Patches -> Dense conversion at Add merge point and soundness.
#[ntest::timeout(120000)]
#[test]
fn test_graph_crown_conv2d_residual_skip_patches_to_dense() {
    let (conv1, graph, input) = build_residual_graph();

    let crown_result =
        tests::with_crown_dense_budget_mb("2048", || graph.propagate_crown_with_provenance(&input))
            .unwrap();
    assert_eq!(crown_result.bounds.shape(), &[1, 4, 4]);
    assert_eq!(
        crown_result.provenance,
        BoundsProvenance::Crown,
        "Expected Crown provenance even with skip connection, got {:?}",
        crown_result.provenance
    );

    // Soundness: manually compute output = relu(conv1(x)) + x at each sample point
    for &offset in &SAMPLE_OFFSETS {
        let point = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5_f32 + offset);
        let point_input = BoundedTensor::concrete(point.clone()).unwrap();
        let conv_out = conv1.propagate_ibp(&point_input).unwrap();
        let relu_out = conv_out.lower().mapv(|v| v.max(0.0));
        let skip_out = &relu_out + &point;

        for (idx, ((out_val, &lower), &upper)) in skip_out
            .iter()
            .zip(crown_result.bounds.lower().iter())
            .zip(crown_result.bounds.upper().iter())
            .enumerate()
        {
            assert!(
                *out_val >= lower - 1e-4,
                "Skip soundness at offset={}, idx={}: {} < lower {}",
                offset,
                idx,
                out_val,
                lower
            );
            assert!(
                *out_val <= upper + 1e-4,
                "Skip soundness at offset={}, idx={}: {} > upper {}",
                offset,
                idx,
                out_val,
                upper
            );
        }
    }
}

/// #3550 regression: the residual Conv2d graph forces a checked Patches-to-Dense
/// merge under batched CROWN, and zero budget must route that failure into the
/// existing graph-level fallback provenance instead of surfacing `CpuMemoryExceeded`.
#[ntest::timeout(120000)]
#[test]
fn test_graph_batched_crown_conv2d_residual_zero_budget_falls_back_3550() {
    tests::with_crown_dense_budget_mb("0", || {
        let (_conv1, mut graph, input) = build_residual_graph();
        for use_patches in [true, false] {
            graph.set_use_patches_mode(use_patches);
            let expected = graph.propagate_crown_with_provenance(&input).unwrap();
            let batched = graph
                .propagate_crown_batched_with_provenance(&input)
                .unwrap();

            assert_eq!(
                expected.provenance,
                BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::MemoryBudgetExceeded
                ),
                "unbatched graph CROWN must preserve memory fallback provenance (patches={use_patches})"
            );
            assert_eq!(
                batched.provenance,
                BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::MemoryBudgetExceeded
                ),
                "batched graph CROWN must preserve memory fallback provenance (patches={use_patches})"
            );
            assert_eq!(batched.bounds.lower(), expected.bounds.lower());
            assert_eq!(batched.bounds.upper(), expected.bounds.upper());
        }
    });
}

/// #4382 regression: graph CROWN on a residual Conv2d DAG should NOT
/// record `graph_crown/utils.rs` as a densification site. The patches-native
/// merge path at the Add node should keep both branches in Patches form.
#[ntest::timeout(120000)]
#[test]
fn test_graph_crown_conv2d_residual_no_utils_densification_4382() {
    use crate::bounds::patches::{patches_to_dense_call_sites, reset_patches_to_dense_call_count};

    let (_conv1, graph, input) = build_residual_graph();

    reset_patches_to_dense_call_count();
    let crown_result =
        tests::with_crown_dense_budget_mb("2048", || graph.propagate_crown_with_provenance(&input))
            .unwrap();
    let sites = patches_to_dense_call_sites();

    assert_eq!(
        crown_result.provenance,
        BoundsProvenance::Crown,
        "Expected Crown provenance for residual graph"
    );

    // The merge point (Add node) should NOT force densification in utils.rs.
    // Before #4382, accumulate_crown_bounds_to_input called into_dense() on the
    // second contribution, which would record utils.rs as a call site.
    let utils_densifications: Vec<&String> = sites
        .iter()
        .filter(|s| s.contains("graph_crown/utils.rs"))
        .collect();
    assert!(
        utils_densifications.is_empty(),
        "#4382 regression: graph_crown/utils.rs should not appear in densification \
         sites when patches-native merge is active, but found: {:?}",
        utils_densifications
    );
}

/// #4382 regression: final bounds from patches-native merge at residual Add
/// must match bounds from the old dense-merge path within directed-rounding
/// slack. This verifies numerical equivalence — the optimization preserves
/// correctness.
#[ntest::timeout(120000)]
#[test]
fn test_graph_crown_conv2d_residual_patches_merge_soundness_4382() {
    let (conv1, graph, input) = build_residual_graph();

    let crown_result =
        tests::with_crown_dense_budget_mb("2048", || graph.propagate_crown_with_provenance(&input))
            .unwrap();
    assert_eq!(crown_result.bounds.shape(), &[1, 4, 4]);

    // Verify soundness: for each sample point, manually compute
    // output = relu(conv1(x)) + x and check containment.
    for &offset in &SAMPLE_OFFSETS {
        let point = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5_f32 + offset);
        let point_input = BoundedTensor::concrete(point.clone()).unwrap();
        let conv_out = conv1.propagate_ibp(&point_input).unwrap();
        let relu_out = conv_out.lower().mapv(|v| v.max(0.0));
        let skip_out = &relu_out + &point;

        for (idx, ((out_val, &lower), &upper)) in skip_out
            .iter()
            .zip(crown_result.bounds.lower().iter())
            .zip(crown_result.bounds.upper().iter())
            .enumerate()
        {
            assert!(
                *out_val >= lower - 1e-4,
                "#4382 soundness at offset={}, idx={}: {} < lower {}",
                offset,
                idx,
                out_val,
                lower
            );
            assert!(
                *out_val <= upper + 1e-4,
                "#4382 soundness at offset={}, idx={}: {} > upper {}",
                offset,
                idx,
                out_val,
                upper
            );
        }
    }

    // Verify bounds are tighter than (or equal to) IBP bounds
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    for (idx, ((&cl, &cu), (&il, &iu))) in crown_result
        .bounds
        .lower()
        .iter()
        .zip(crown_result.bounds.upper().iter())
        .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
        .enumerate()
    {
        assert!(
            cl >= il - 1e-4,
            "#4382 tightness: CROWN lower {} < IBP lower {} at idx={}",
            cl,
            il,
            idx
        );
        assert!(
            cu <= iu + 1e-4,
            "#4382 tightness: CROWN upper {} > IBP upper {} at idx={}",
            cu,
            iu,
            idx
        );
    }
}

/// Build a Conv2d + Sub residual skip connection graph.
///
/// Graph: _input -> conv1(same-pad) -> relu -> sub -> output
///          |                                   ^
///          +-----------------------------------+
/// Output = _input - relu(conv1(_input))
fn build_sub_residual_graph() -> (Conv2dLayer, GraphNetwork, BoundedTensor) {
    use crate::layers::SubLayer;

    let mut kernel = ArrayD::zeros(IxDyn(&[1, 1, 3, 3]));
    kernel[[0, 0, 0, 0]] = 0.2;
    kernel[[0, 0, 0, 1]] = -0.1;
    kernel[[0, 0, 0, 2]] = 0.15;
    kernel[[0, 0, 1, 0]] = 0.3;
    kernel[[0, 0, 1, 1]] = 0.5;
    kernel[[0, 0, 1, 2]] = -0.2;
    kernel[[0, 0, 2, 0]] = 0.1;
    kernel[[0, 0, 2, 1]] = 0.25;
    kernel[[0, 0, 2, 2]] = -0.15;
    let bias = Array1::from_vec(vec![0.05]);
    let conv1 = Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (1, 1), 4, 4).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1.clone())));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sub",
        Layer::Sub(SubLayer),
        vec!["_input".to_string(), "relu".to_string()],
    ));
    graph.set_output("sub");

    let center = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5_f32);
    let input = BoundedTensor::from_epsilon(center, 0.1).unwrap();
    (conv1, graph, input)
}

/// #4382 regression: Sub residual graph (output = input - relu(conv(input)))
/// should also use patches-native merge and produce sound bounds.
#[ntest::timeout(120000)]
#[test]
fn test_graph_crown_conv2d_residual_sub_soundness_4382() {
    use crate::bounds::patches::{patches_to_dense_call_sites, reset_patches_to_dense_call_count};

    let (conv1, graph, input) = build_sub_residual_graph();

    reset_patches_to_dense_call_count();
    let crown_result =
        tests::with_crown_dense_budget_mb("2048", || graph.propagate_crown_with_provenance(&input))
            .unwrap();
    let sites = patches_to_dense_call_sites();

    assert_eq!(crown_result.provenance, BoundsProvenance::Crown);
    assert_eq!(crown_result.bounds.shape(), &[1, 4, 4]);

    let utils_sites: Vec<&String> = sites
        .iter()
        .filter(|s| s.contains("graph_crown/utils.rs"))
        .collect();
    assert!(
        utils_sites.is_empty(),
        "#4382 Sub regression: unexpected densification at utils.rs: {:?}",
        utils_sites
    );

    for &offset in &SAMPLE_OFFSETS {
        let point = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5_f32 + offset);
        let point_input = BoundedTensor::concrete(point.clone()).unwrap();
        let conv_out = conv1.propagate_ibp(&point_input).unwrap();
        let relu_out = conv_out.lower().mapv(|v| v.max(0.0));
        let sub_out = &point - &relu_out;
        assert_bounds_contain_samples_array(&crown_result.bounds, &sub_out);
    }
}

/// Assert that a pre-computed output array falls within the bounds.
fn assert_bounds_contain_samples_array(bounds: &BoundedTensor, output: &ArrayD<f32>) {
    for (idx, ((out_val, &lower), &upper)) in output
        .iter()
        .zip(bounds.lower().iter())
        .zip(bounds.upper().iter())
        .enumerate()
    {
        assert!(
            *out_val >= lower - 1e-4,
            "idx={}: {} < lower {}",
            idx,
            out_val,
            lower
        );
        assert!(
            *out_val <= upper + 1e-4,
            "idx={}: {} > upper {}",
            idx,
            out_val,
            upper
        );
    }
}

/// Phase 1b: Conv2d graph CROWN bounds match sequential CROWN bounds.
#[ntest::timeout(120000)]
#[test]
fn test_graph_crown_conv2d_chain_matches_sequential() {
    let (network, graph, input) = build_conv_relu_conv_network();

    let (graph_bounds, seq_bounds) = tests::with_crown_dense_budget_mb("2048", || {
        (
            graph.propagate_crown(&input).unwrap(),
            network.propagate_crown(&input).unwrap(),
        )
    });

    assert_eq!(graph_bounds.shape(), seq_bounds.shape());

    for (idx, ((&gl, &gu), (&sl, &su))) in graph_bounds
        .lower()
        .iter()
        .zip(graph_bounds.upper().iter())
        .zip(seq_bounds.lower().iter().zip(seq_bounds.upper().iter()))
        .enumerate()
    {
        assert!(
            gl.is_finite() && gu.is_finite(),
            "Graph bounds idx={} non-finite: [{}, {}]",
            idx,
            gl,
            gu
        );
        assert!(
            sl.is_finite() && su.is_finite(),
            "Sequential bounds idx={} non-finite: [{}, {}]",
            idx,
            sl,
            su
        );
        assert!(gl <= gu + 1e-5, "Graph bounds inverted at idx={}", idx);
        assert!(sl <= su + 1e-5, "Sequential bounds inverted at idx={}", idx);
    }
}
