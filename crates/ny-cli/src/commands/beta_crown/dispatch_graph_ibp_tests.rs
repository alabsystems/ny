// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{arr1, ArrayD, IxDyn};
use ny_onnx::vnnlib::parse_vnnlib;
use ny_propagate::layers::Conv2dLayer;
use ny_propagate::{GraphNetwork, GraphNode, Layer, Network};
use std::time::Duration;

const TWO_POW_24: f32 = 16_777_216.0;

fn cancellation_graph(bias: f32) -> GraphNetwork {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2]), vec![TWO_POW_24, 1.0]).unwrap();
    let conv = Conv2dLayer::new(kernel, Some(arr1(&[bias])), (1, 1), (0, 0)).unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.set_output("conv");
    graph
}

fn cancellation_network() -> Network {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2]), vec![TWO_POW_24, 1.0]).unwrap();
    let conv = Conv2dLayer::new(kernel, Some(arr1(&[-TWO_POW_24])), (1, 1), (0, 0)).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network
}

fn point(first: f32, second: f32) -> BoundedTensor {
    let values = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![first, second]).unwrap();
    BoundedTensor::new(values.clone(), values).unwrap()
}

fn point_property(first: u8, second: u8) -> VnnLibSpec {
    parse_vnnlib(&format!(
        "(declare-const X_0 Real)\n\
         (declare-const X_1 Real)\n\
         (declare-const Y_0 Real)\n\
         (assert (>= X_0 {first}))\n\
         (assert (<= X_0 {first}))\n\
         (assert (>= X_1 {second}))\n\
         (assert (<= X_1 {second}))\n\
         (assert (or (>= Y_0 0.5) (<= Y_0 -2)))",
    ))
    .unwrap()
}

fn property_is_refuted(output: &BoundedTensor, property: &VnnLibSpec) -> bool {
    ibp_check_vnnlib_safe(
        output.lower().as_slice().unwrap(),
        output.upper().as_slice().unwrap(),
        property,
    )
}

fn encloses(output: &BoundedTensor, exact: f32) -> bool {
    output.len() == 1
        && output.lower().iter().all(|&value| value <= exact)
        && output.upper().iter().all(|&value| exact <= value)
}

#[test]
fn early_graph_ibp_cancellation_cannot_refute_the_violated_original_property() {
    let graph = cancellation_graph(-TWO_POW_24);
    let input = point(1.0, 1.0);
    let property = point_property(1, 1);
    // Exact integer oracle: 2^24 + 1 - 2^24 = 1 satisfies the first unsafe
    // clause. The old plain Conv stores the f32 sum before adding the bias:
    // 2^24 + 1 rounds to 2^24, then subtracting 2^24 produces a false point 0.
    // This genuine legacy call pins why its output cannot authorize the check.
    let plain = graph.propagate_ibp(&input).unwrap();
    assert!(!encloses(&plain, 1.0));
    assert!(property_is_refuted(&plain, &property));

    let output = early_graph_ibp_output(
        &graph,
        &input,
        Some(Instant::now() + Duration::from_secs(30)),
    )
    .unwrap();
    assert!(encloses(&output, 1.0));
    assert!(!property_is_refuted(&output, &property));
    assert_eq!(property.output_constraint_clauses.len(), 2);
}

#[test]
fn early_graph_ibp_uses_the_actual_input_and_graph_without_reusing_a_safe_box() {
    let graph = cancellation_graph(-TWO_POW_24);
    let live = Some(Instant::now() + Duration::from_secs(30));
    let other_input = early_graph_ibp_output(&graph, &point(1.0, 0.0), live).unwrap();
    assert!(encloses(&other_input, 0.0));
    assert!(property_is_refuted(&other_input, &point_property(1, 0)));
    let actual_input = early_graph_ibp_output(&graph, &point(1.0, 1.0), live).unwrap();
    assert!(encloses(&actual_input, 1.0));
    assert!(!property_is_refuted(&actual_input, &point_property(1, 1)));

    // Same public node name and shape, but a different original bias: the
    // complete property is safe at -1 and violated at +1. No caller map enters.
    let changed = cancellation_graph(-TWO_POW_24 - 2.0);
    let changed_output = early_graph_ibp_output(&changed, &point(1.0, 1.0), live).unwrap();
    assert!(encloses(&changed_output, -1.0));
    assert!(property_is_refuted(&changed_output, &point_property(1, 1)));
    let original_again = early_graph_ibp_output(&graph, &point(1.0, 1.0), live).unwrap();
    assert!(!property_is_refuted(&original_again, &point_property(1, 1)));
}

#[test]
fn early_graph_ibp_expiry_and_incomplete_source_never_publish_an_output() {
    let graph = cancellation_graph(-TWO_POW_24);
    let input = point(1.0, 1.0);
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("one millisecond before now is representable");
    assert!(matches!(
        early_graph_ibp_output(&graph, &input, Some(expired)),
        Err(ny_core::NyError::DeadlineExceeded(_))
    ));
    let wrong_shape = BoundedTensor::concrete(arr1(&[1.0_f32, 1.0]).into_dyn()).unwrap();
    assert!(early_graph_ibp_output(
        &graph,
        &wrong_shape,
        Some(Instant::now() + Duration::from_secs(30))
    )
    .is_err());
    let mut missing_output = graph;
    missing_output.set_output("absent");
    assert!(early_graph_ibp_output(
        &missing_output,
        &input,
        Some(Instant::now() + Duration::from_secs(30))
    )
    .is_err());
}

#[test]
fn early_sequential_ibp_cancellation_cannot_refute_the_violated_original_property() {
    let network = cancellation_network();
    let input = point(1.0, 1.0);
    let property = point_property(1, 1);
    // The plain sequential forward delegates straight to the same f32 Conv
    // trait implementation. The existing certified collector is a distinct API.
    let plain = network.propagate_ibp(&input).unwrap();
    assert!(!encloses(&plain, 1.0));
    assert!(property_is_refuted(&plain, &property));
    let output = early_sequential_ibp_output(
        &network,
        &input,
        Some(Instant::now() + Duration::from_secs(30)),
    )
    .unwrap();
    assert!(encloses(&output, 1.0));
    assert!(!property_is_refuted(&output, &property));
    assert_eq!(property.output_constraint_clauses.len(), 2);
}

#[test]
fn early_sequential_ibp_expiry_and_no_completed_output_never_publish_bounds() {
    let network = cancellation_network();
    let input = point(1.0, 1.0);
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("one millisecond before now is representable");
    let error = early_sequential_ibp_output(&network, &input, Some(expired))
        .expect_err("expired certified Conv must not supply early verdict bounds");
    assert!(matches!(
        error,
        ny_core::NyError::LayerError { layer_index: 0, source, .. }
            if source.is_deadline_exceeded()
    ));
    assert!(matches!(
        early_sequential_ibp_output(
            &Network::new(),
            &input,
            Some(Instant::now() + Duration::from_secs(30))
        ),
        Err(ny_core::NyError::InvalidSpec(_))
    ));
}
