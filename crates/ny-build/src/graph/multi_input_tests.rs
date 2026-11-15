// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Containment and equivalence tests for the multi-input concat-view adapter
//! (#multi-input-propagation).
//!
//! Every test here was watched fail before the mechanism (or, for the planted
//! violation, fails when the corruption is NOT rejected): the containment
//! harness must be shown ABLE to fail.

use super::super::builder::{build_graph_network, GraphBuildInputs};
use super::*;
use crate::{DataType, GraphNetworkOptions, LayerSpec, WeightStore};
use ndarray::arr1;
use ny_core::LayerType;
use ny_tensor::BoundedTensor;
use std::collections::{HashMap, HashSet};

fn tensor_spec(name: &str, shape: &[i64]) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
        dtype: DataType::Float32,
    }
}

fn layer_spec(name: &str, layer_type: LayerType, inputs: &[&str], outputs: &[&str]) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type,
        inputs: inputs.iter().copied().map(str::to_owned).collect(),
        outputs: outputs.iter().copied().map(str::to_owned).collect(),
        weights: None,
        attributes: HashMap::new(),
    }
}

fn armed_options() -> GraphNetworkOptions {
    GraphNetworkOptions {
        multi_input_concat: true,
        ..GraphNetworkOptions::default()
    }
}

fn box_tensor(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn())
        .expect("valid bounds for test box")
}

/// The two-input model used across the containment tests: inputs declared in
/// order `[a (2 elems), b (3 elems)]`, but the LAYER LIST uses `b` FIRST —
/// usage order differs from declaration order, so an implementation that
/// assigned slices by first use (or by anything other than declared input
/// order) routes `b`'s consumers to `a`'s coordinates and fails these tests.
fn two_input_build() -> (
    Vec<LayerSpec>,
    Vec<TensorSpec>,
    Vec<TensorSpec>,
    WeightStore,
) {
    let layers = vec![
        layer_spec("relu_b", LayerType::ReLU, &["b"], &["relu_b_out"]),
        layer_spec("relu_a", LayerType::ReLU, &["a"], &["relu_a_out"]),
    ];
    let inputs = vec![tensor_spec("a", &[1, 2]), tensor_spec("b", &[1, 3])];
    let outputs = vec![tensor_spec("relu_b_out", &[1, 3])];
    (layers, inputs, outputs, WeightStore::new())
}

fn build_two_input_graph(options: GraphNetworkOptions) -> GraphNetwork {
    let (layers, inputs, outputs, weights) = two_input_build();
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = HashMap::new();
    let data = GraphBuildInputs {
        layers: &layers,
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        tensor_producer: &tensor_producer,
        constant_tensors: &constant_tensors,
        tensor_shapes: &tensor_shapes,
    };
    build_graph_network(&data, options).expect("two-input graph builds")
}

/// Declared-order joint box: `a` gets asymmetric NEGATIVE coordinates,
/// `b` gets large positive ones, so any slice mix-up changes every number.
fn joint_box() -> BoundedTensor {
    box_tensor(
        &[-4.0, -3.0, 10.0, 20.0, 30.0],
        &[-2.0, -1.0, 11.0, 21.0, 31.0],
    )
}

#[test]
fn concat_layout_is_exact_declared_order_partition() {
    let inputs = [
        tensor_spec("a", &[1, 2]),
        tensor_spec("b", &[1, 3, 2]),
        tensor_spec("c", &[4]),
    ];
    let layout = concat_input_layout(&inputs).expect("static shapes lay out");
    let slots = layout.slots();
    assert_eq!(layout.total_len(), 2 + 6 + 4);
    assert_eq!(
        slots
            .iter()
            .map(|slot| (slot.name.as_str(), slot.offset, slot.len))
            .collect::<Vec<_>>(),
        vec![("a", 0, 2), ("b", 2, 6), ("c", 8, 4)],
        "slots must follow DECLARED order with contiguous offsets"
    );
    assert_eq!(slots[1].internal_shape, vec![3, 2], "batch dim squeezed");
    layout.verify_partition().expect("exact partition verifies");
}

#[test]
fn concat_layout_rejects_planted_off_by_one_slice() {
    // The planted violation: shift the second slot's offset by ONE coordinate.
    // The partition check MUST refuse — this is the proof the containment
    // harness can fail. Both the checker and the production emission path
    // (which re-verifies before emitting any node) are exercised.
    let inputs = [tensor_spec("a", &[1, 2]), tensor_spec("b", &[1, 3])];
    let mut layout = concat_input_layout(&inputs).expect("valid layout");
    layout
        .verify_partition()
        .expect("uncorrupted layout passes");

    layout.corrupt_slot_offset_for_test(1, 1);
    let error = layout
        .verify_partition()
        .expect_err("off-by-one slice offset must be rejected");
    assert!(
        error.to_string().contains("layout violation"),
        "unexpected error: {error}"
    );

    let mut graph = GraphNetwork::new();
    let mut tensor_to_node = HashMap::new();
    insert_input_view_nodes(&mut graph, &mut tensor_to_node, &layout)
        .expect_err("emission path must re-verify and refuse the corrupted layout");
    assert_eq!(
        graph.num_nodes(),
        0,
        "no view node may be emitted from a corrupted layout"
    );
}

#[test]
fn concat_layout_refuses_dynamic_non_batch_dims_and_duplicates() {
    let dynamic = [tensor_spec("a", &[1, 2]), tensor_spec("b", &[1, -1])];
    concat_input_layout(&dynamic).expect_err("dynamic non-batch dim must refuse");

    let duplicate = [tensor_spec("a", &[1, 2]), tensor_spec("a", &[1, 3])];
    concat_input_layout(&duplicate).expect_err("duplicate input name must refuse");

    let single = [tensor_spec("a", &[1, 2])];
    concat_input_layout(&single).expect_err("single input must stay on the legacy path");
}

#[test]
fn armed_two_input_ibp_routes_declared_order_slices() {
    // Usage order (b first) differs from declaration order ([a, b]).
    // The output is Relu(b); with the joint box laid out in DECLARED order the
    // output must be exactly b's box (all-positive, ReLU is identity there).
    // A usage-ordered or off-by-one layout would leak a's negative
    // coordinates into the result.
    let graph = build_two_input_graph(armed_options());
    let output = graph.propagate_ibp(&joint_box()).expect("IBP propagates");
    assert_eq!(output.lower().as_slice().unwrap(), &[10.0, 20.0, 30.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[11.0, 21.0, 31.0]);
}

#[test]
fn armed_two_input_view_bounds_are_exact_product_slices() {
    // CONTAINMENT: each view node's interval must be EXACTLY the per-input
    // slice of the joint box — bit-equal, no widening in either direction.
    let graph = build_two_input_graph(armed_options());
    let bounds = graph
        .collect_node_bounds_with_engine(&joint_box(), None)
        .expect("node bounds collect");

    let a_view = bounds.get("_input_view_0").expect("view node for input a");
    assert_eq!(a_view.lower().as_slice().unwrap(), &[-4.0, -3.0]);
    assert_eq!(a_view.upper().as_slice().unwrap(), &[-2.0, -1.0]);

    let b_view = bounds.get("_input_view_1").expect("view node for input b");
    assert_eq!(b_view.lower().as_slice().unwrap(), &[10.0, 20.0, 30.0]);
    assert_eq!(b_view.upper().as_slice().unwrap(), &[11.0, 21.0, 31.0]);

    // And the consumers are wired to the views, not to `_input`.
    assert_eq!(
        graph.node("relu_a").expect("relu_a").inputs(),
        &["_input_view_0".to_string()]
    );
    assert_eq!(
        graph.node("relu_b").expect("relu_b").inputs(),
        &["_input_view_1".to_string()]
    );
}

#[test]
fn armed_two_input_crown_ibp_matches_exact_slices() {
    // CROWN backward must terminate through the Slice views with the same
    // exact routing (selection matrices, no widening on this model).
    let graph = build_two_input_graph(armed_options());
    let result = graph
        .collect_crown_ibp_bounds_dag_with_status(&joint_box())
        .expect("CROWN-IBP collects");
    let out = result.bounds.get("relu_b").expect("output node bound");
    assert_eq!(out.lower().as_slice().unwrap(), &[10.0, 20.0, 30.0]);
    assert_eq!(out.upper().as_slice().unwrap(), &[11.0, 21.0, 31.0]);
}

#[test]
fn gate_off_two_input_build_is_byte_identical_legacy_aliasing() {
    // Default OFF: both inputs alias to `_input` exactly as before the
    // mechanism existed — no view nodes, consumers wired straight to the
    // single input node (which is why multi-input models fail closed today).
    let graph = build_two_input_graph(GraphNetworkOptions::default());
    assert!(
        graph
            .node_names()
            .iter()
            .all(|name| !name.starts_with("_input_view")),
        "gate off must emit no view nodes"
    );
    assert_eq!(
        graph.node("relu_a").expect("relu_a").inputs(),
        &["_input".to_string()]
    );
    assert_eq!(
        graph.node("relu_b").expect("relu_b").inputs(),
        &["_input".to_string()]
    );
}

#[test]
fn single_input_model_is_bit_identical_with_gate_on() {
    // The default path must not move: on a single-input model the armed and
    // unarmed builds must produce the same graph and BIT-IDENTICAL IBP and
    // CROWN-IBP bounds.
    let layers = vec![
        layer_spec("relu0", LayerType::ReLU, &["x"], &["relu0_out"]),
        layer_spec("relu1", LayerType::ReLU, &["relu0_out"], &["relu1_out"]),
    ];
    let inputs = vec![tensor_spec("x", &[1, 4])];
    let outputs = vec![tensor_spec("relu1_out", &[1, 4])];
    let weights = WeightStore::new();
    let tensor_producer = HashMap::new();
    let constant_tensors = HashSet::new();
    let tensor_shapes = HashMap::new();
    let build = |options| {
        let data = GraphBuildInputs {
            layers: &layers,
            inputs: &inputs,
            outputs: &outputs,
            weights: &weights,
            tensor_producer: &tensor_producer,
            constant_tensors: &constant_tensors,
            tensor_shapes: &tensor_shapes,
        };
        build_graph_network(&data, options).expect("single-input graph builds")
    };
    let off = build(GraphNetworkOptions::default());
    let on = build(armed_options());
    assert_eq!(off.node_names(), on.node_names());
    for name in off.node_names() {
        assert_eq!(
            off.node(name).expect("off node").inputs(),
            on.node(name).expect("on node").inputs(),
            "wiring must be identical for node {name}"
        );
    }

    let input = box_tensor(&[-1.0, -0.5, 0.25, 2.0], &[-0.5, 0.5, 1.0, 3.0]);
    let ibp_off = off.propagate_ibp(&input).expect("off IBP");
    let ibp_on = on.propagate_ibp(&input).expect("on IBP");
    assert_eq!(ibp_off.lower(), ibp_on.lower());
    assert_eq!(ibp_off.upper(), ibp_on.upper());

    let crown_off = off
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .expect("off CROWN-IBP");
    let crown_on = on
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .expect("on CROWN-IBP");
    for name in off.node_names() {
        let (Some(bound_off), Some(bound_on)) =
            (crown_off.bounds.get(name), crown_on.bounds.get(name))
        else {
            assert_eq!(
                crown_off.bounds.contains_key(name),
                crown_on.bounds.contains_key(name),
                "CROWN-IBP coverage must be identical for node {name}"
            );
            continue;
        };
        assert_eq!(bound_off.lower(), bound_on.lower(), "node {name}");
        assert_eq!(bound_off.upper(), bound_on.upper(), "node {name}");
    }
}
