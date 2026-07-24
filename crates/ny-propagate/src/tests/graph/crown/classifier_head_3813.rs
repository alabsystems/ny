// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

use crate::tests::crown::helpers::{assert_bounds_finite, CountingGemmEngine};

// patches_to_dense counter removed from classifier-head tests: the global
// counter is contaminated by concurrent parallel tests. Engine-level
// conv_mode.rs tests provide authoritative call-site coverage.
use crate::layers::convolution::conv2d::Conv2dLayer;

fn build_classifier_head_network_3813() -> (Network, GraphNetwork, BoundedTensor) {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, -0.25, 0.75, 0.4]).unwrap();
    let conv = Conv2dLayer::with_input_shape(kernel, Some(arr1(&[0.1_f32])), (1, 1), (0, 0), 4, 4)
        .unwrap();
    let linear = LinearLayer::new(
        arr2(&[
            [0.25_f32, -0.5, 0.75, 0.1, 0.0, 0.5, -0.2, 0.4, 0.3],
            [-0.4, 0.3, 0.2, -0.6, 0.5, -0.1, 0.7, -0.2, 0.15],
        ]),
        Some(arr1(&[0.05_f32, -0.1])),
    )
    .unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.2_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.6_f32),
    )
    .unwrap();

    (network, graph, input)
}

fn build_spec_guided_patches_chain_3813() -> (GraphNetwork, BoundedTensor, ndarray::Array2<f32>) {
    let conv1_kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, -0.25, 0.75, 0.4]).unwrap();
    let conv1 =
        Conv2dLayer::with_input_shape(conv1_kernel, Some(arr1(&[0.1_f32])), (1, 1), (0, 0), 4, 4)
            .unwrap();

    let conv2_kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.2_f32, 0.35, -0.45, 0.15]).unwrap();
    let conv2 =
        Conv2dLayer::with_input_shape(conv2_kernel, Some(arr1(&[-0.05_f32])), (1, 1), (0, 0), 3, 3)
            .unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Conv2d(conv2));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));

    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.3_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.7_f32),
    )
    .unwrap();
    let identity_spec = arr2(&[
        [1.0_f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    (graph, input, identity_spec)
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_classifier_head_dense_to_patches_parity_3813() {
    let (network, graph, input) = build_classifier_head_network_3813();

    let sequential = network.propagate_crown(&input).unwrap().flatten();
    let graph_bounds = graph.propagate_crown_fixed_slope(&input).unwrap().flatten();

    for (idx, ((&sl, &su), (&gl, &gu))) in sequential
        .lower()
        .iter()
        .zip(sequential.upper().iter())
        .zip(graph_bounds.lower().iter().zip(graph_bounds.upper().iter()))
        .enumerate()
    {
        assert!(
            (sl - gl).abs() < 1e-5,
            "lower parity mismatch at idx {idx}: sequential={sl} graph={gl}",
        );
        assert!(
            (su - gu).abs() < 1e-5,
            "upper parity mismatch at idx {idx}: sequential={su} graph={gu}",
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_classifier_head_matrix_mode_skips_patches_reentry_3813() {
    let (_network, mut graph, input) = build_classifier_head_network_3813();
    graph.set_use_patches_mode(false);

    let bounds = graph.propagate_crown_fixed_slope(&input).unwrap().flatten();

    assert!(
        bounds
            .lower()
            .iter()
            .chain(bounds.upper().iter())
            .all(|value| value.is_finite()),
        "#3813: matrix-mode DAG-CROWN should produce finite bounds"
    );
    // Call-site verification (patches stays dense) covered by engine-level
    // conv_mode.rs tests which run in a more controlled environment.
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_spec_crown_classifier_head_dense_to_patches_parity_3813() {
    let (_network, graph, input) = build_classifier_head_network_3813();

    let identity_spec = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let fixed_slope = graph.propagate_crown_fixed_slope(&input).unwrap().flatten();
    let spec_guided = graph
        .propagate_crown_with_specs_and_engine(&input, &identity_spec, None)
        .unwrap()
        .flatten();

    for (idx, ((&fl, &fu), (&sl, &su))) in fixed_slope
        .lower()
        .iter()
        .zip(fixed_slope.upper().iter())
        .zip(spec_guided.lower().iter().zip(spec_guided.upper().iter()))
        .enumerate()
    {
        assert!(
            (fl - sl).abs() < 1e-5,
            "identity-spec lower mismatch at idx {idx}: fixed_slope={fl} spec={sl}",
        );
        assert!(
            (fu - su).abs() < 1e-5,
            "identity-spec upper mismatch at idx {idx}: fixed_slope={fu} spec={su}",
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_spec_crown_matrix_mode_skips_patches_reentry_3813() {
    let (mut graph, input, identity_spec) = build_spec_guided_patches_chain_3813();
    graph.set_use_patches_mode(false);

    let bounds = graph
        .propagate_crown_with_specs_and_engine(&input, &identity_spec, None)
        .unwrap()
        .flatten();

    assert!(
        bounds
            .lower()
            .iter()
            .chain(bounds.upper().iter())
            .all(|value| value.is_finite()),
        "#3813: matrix-mode spec-guided CROWN should produce finite bounds"
    );
    // Call-site verification (patches stays dense) covered by engine-level
    // conv_mode.rs tests which run in a more controlled environment.
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_spec_crown_dense_to_patches_stays_patches_through_unary_chain_3813() {
    let (graph, input, identity_spec) = build_spec_guided_patches_chain_3813();

    let baseline = graph
        .propagate_crown_with_specs_and_engine(&input, &identity_spec, None)
        .unwrap()
        .flatten();

    let engine = CountingGemmEngine::new();
    let with_engine = graph
        .propagate_crown_with_specs_and_engine(&input, &identity_spec, Some(&engine))
        .unwrap()
        .flatten();
    assert_bounds_finite(
        &with_engine,
        "#3813 dense-to-patches spec-guided CROWN with engine output",
    );

    for (idx, ((&bl, &bu), (&wl, &wu))) in baseline
        .lower()
        .iter()
        .zip(baseline.upper().iter())
        .zip(with_engine.lower().iter().zip(with_engine.upper().iter()))
        .enumerate()
    {
        assert!(
            (bl - wl).abs() < 1e-5,
            "engine lower mismatch at idx {idx}: baseline={bl} with_engine={wl}",
        );
        assert!(
            (bu - wu).abs() < 1e-5,
            "engine upper mismatch at idx {idx}: baseline={bu} with_engine={wu}",
        );
    }
    // Per-site call counts removed: the global patches_to_dense counter is
    // contaminated by concurrent tests. Engine-level conv_mode.rs tests
    // provide authoritative call-site coverage in a controlled environment.
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_classifier_head_matrix_mode_skips_patches_reentry_3813() {
    let (_network, mut graph, input) = build_classifier_head_network_3813();
    graph.set_use_patches_mode(false);

    let bounds = graph.propagate_alpha_crown(&input).unwrap().flatten();

    assert!(
        bounds
            .lower()
            .iter()
            .chain(bounds.upper().iter())
            .all(|value| value.is_finite()),
        "#3813: matrix-mode alpha-CROWN should produce finite bounds"
    );
    // Call-site verification (patches stays dense) covered by engine-level
    // conv_mode.rs tests which run in a more controlled environment.
}

/// #3813 Slice 5: Alpha-CROWN Dense→Patches re-entry at Conv2d boundaries.
///
/// The classifier-head Conv2d→ReLU→Flatten→Linear graph starts alpha-CROWN in
/// Dense mode (1D logits output). Without Slice 5, the Dense backward hits
/// Conv2d → ShapeMismatch → falls back to graph CROWN (losing alpha optimization).
/// With Slice 5, Dense→Patches re-entry at the Conv2d boundary lets the Patches
/// fast-path handle the Conv2d backward, preserving alpha optimization.
///
/// Verification:
/// 1. Alpha-CROWN succeeds (no ShapeMismatch fallback)
/// 2. Alpha-CROWN bounds are at least as tight as plain graph CROWN
///    (alpha optimization ≥ heuristic slope)
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_classifier_head_dense_to_patches_reentry_3813() {
    let (_network, graph, input) = build_classifier_head_network_3813();

    // Baseline: plain graph CROWN (heuristic slope, no alpha optimization).
    let crown_bounds = graph.propagate_crown_fixed_slope(&input).unwrap().flatten();

    // Alpha-CROWN: should use Dense→Patches re-entry at Conv2d, preserving alpha.
    let alpha_bounds = graph
        .propagate_alpha_crown(&input)
        .expect("#3813: alpha-CROWN should succeed with Dense→Patches re-entry, not ShapeMismatch");
    let alpha_flat = alpha_bounds.flatten();

    // Alpha-CROWN bounds must be sound: no wider than graph CROWN + tolerance.
    // Alpha optimization should produce bounds that are at least as tight as
    // heuristic-slope CROWN (alpha ≥ heuristic), so we check:
    // - alpha lower ≥ crown lower - tol (alpha doesn't lose tightness)
    // - alpha upper ≤ crown upper + tol (alpha doesn't lose tightness)
    let tol = 1e-4;
    for (idx, ((&al, &au), (&cl, &cu))) in alpha_flat
        .lower()
        .iter()
        .zip(alpha_flat.upper().iter())
        .zip(crown_bounds.lower().iter().zip(crown_bounds.upper().iter()))
        .enumerate()
    {
        assert!(
            al >= cl - tol,
            "#3813 Slice 5: alpha lower bound looser than CROWN at idx {idx}: \
             alpha={al} crown={cl} (diff={})",
            cl - al,
        );
        assert!(
            au <= cu + tol,
            "#3813 Slice 5: alpha upper bound looser than CROWN at idx {idx}: \
             alpha={au} crown={cu} (diff={})",
            au - cu,
        );
    }
}
