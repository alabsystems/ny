// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the affine-extent slice shape pass (#cctsdb B2).

use std::collections::HashMap;

use super::augment_inferred_shapes_with_affine_slice_extents;
use crate::onnx_proto::{GraphProto, NodeProto};
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};

fn node(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

fn graph(nodes: Vec<NodeProto>) -> GraphProto {
    GraphProto {
        node: nodes,
        ..Default::default()
    }
}

fn insert_i64_const(weights: &mut WeightStore, name: &str, values: &[i64]) {
    weights.insert(
        name.to_string(),
        ArrayD::from_shape_vec(
            IxDyn(&[values.len()]),
            values.iter().map(|&v| v as f32).collect(),
        )
        .unwrap(),
    );
    weights.insert_integers(
        name.to_string(),
        ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap(),
    );
}

/// The cctsdb pattern: Slice(data, starts=[x], ends=[x+w]) with x an opaque
/// activation scalar => static extent w.
#[test]
fn affine_slice_extent_same_symbol() {
    // x (opaque, from a Cast) -> starts; Add(x, w=3) -> ends.
    let g = graph(vec![
        node("Cast", "cast", &["x_float"], &["x"]),
        node("Add", "add_w", &["x", "w"], &["x_plus_w"]),
        node(
            "Slice",
            "slice",
            &["data", "x", "x_plus_w", "axes", "steps"],
            &["sliced"],
        ),
    ]);
    let mut weights = WeightStore::new();
    insert_i64_const(&mut weights, "w", &[3]);
    insert_i64_const(&mut weights, "axes", &[2]);
    insert_i64_const(&mut weights, "steps", &[1]);
    let mut shapes: HashMap<String, Vec<i64>> =
        HashMap::from([("data".to_string(), vec![1, 3, 64, 64])]);

    let added = augment_inferred_shapes_with_affine_slice_extents(&g, &weights, &mut shapes);
    assert!(added);
    assert_eq!(shapes.get("sliced"), Some(&vec![1, 3, 3, 64]));
}

/// Chained slices resolve through the internal fixpoint: the second slice's
/// data shape comes from the first deduction.
#[test]
fn affine_slice_extent_chained() {
    let g = graph(vec![
        node("Add", "add_w1", &["x", "one"], &["x_plus_1"]),
        node("Add", "add_w2", &["y", "one"], &["y_plus_1"]),
        node(
            "Slice",
            "slice_a",
            &["data", "x", "x_plus_1", "axes2", "steps"],
            &["a"],
        ),
        node(
            "Slice",
            "slice_b",
            &["a", "y", "y_plus_1", "axes3", "steps"],
            &["b"],
        ),
    ]);
    let mut weights = WeightStore::new();
    insert_i64_const(&mut weights, "one", &[1]);
    insert_i64_const(&mut weights, "axes2", &[2]);
    insert_i64_const(&mut weights, "axes3", &[3]);
    insert_i64_const(&mut weights, "steps", &[1]);
    let mut shapes: HashMap<String, Vec<i64>> =
        HashMap::from([("data".to_string(), vec![1, 3, 64, 64])]);

    let added = augment_inferred_shapes_with_affine_slice_extents(&g, &weights, &mut shapes);
    assert!(added);
    assert_eq!(shapes.get("a"), Some(&vec![1, 3, 1, 64]));
    assert_eq!(shapes.get("b"), Some(&vec![1, 3, 1, 1]));
}

/// Different symbols on starts vs ends: extent is data-dependent, no fold.
#[test]
fn affine_slice_extent_different_symbols_rejected() {
    let g = graph(vec![node(
        "Slice",
        "slice",
        &["data", "x", "y", "axes", "steps"],
        &["sliced"],
    )]);
    let mut weights = WeightStore::new();
    insert_i64_const(&mut weights, "axes", &[0]);
    insert_i64_const(&mut weights, "steps", &[1]);
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::from([("data".to_string(), vec![64])]);

    let added = augment_inferred_shapes_with_affine_slice_extents(&g, &weights, &mut shapes);
    assert!(!added);
    assert!(!shapes.contains_key("sliced"));
}

/// Fully constant starts/ends are the existing machinery's job — skipped here.
#[test]
fn affine_slice_extent_all_const_skipped() {
    let g = graph(vec![node(
        "Slice",
        "slice",
        &["data", "s", "e", "axes", "steps"],
        &["sliced"],
    )]);
    let mut weights = WeightStore::new();
    insert_i64_const(&mut weights, "s", &[1]);
    insert_i64_const(&mut weights, "e", &[5]);
    insert_i64_const(&mut weights, "axes", &[0]);
    insert_i64_const(&mut weights, "steps", &[1]);
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::from([("data".to_string(), vec![64])]);

    let added = augment_inferred_shapes_with_affine_slice_extents(&g, &weights, &mut shapes);
    assert!(!added);
}

/// Non-unit steps change the extent formula: rejected.
#[test]
fn affine_slice_extent_non_unit_steps_rejected() {
    let g = graph(vec![
        node("Add", "add_w", &["x", "w"], &["x_plus_w"]),
        node(
            "Slice",
            "slice",
            &["data", "x", "x_plus_w", "axes", "steps"],
            &["sliced"],
        ),
    ]);
    let mut weights = WeightStore::new();
    insert_i64_const(&mut weights, "w", &[4]);
    insert_i64_const(&mut weights, "axes", &[0]);
    insert_i64_const(&mut weights, "steps", &[2]);
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::from([("data".to_string(), vec![64])]);

    let added = augment_inferred_shapes_with_affine_slice_extents(&g, &weights, &mut shapes);
    assert!(!added);
}

/// Extent larger than the axis is capped to the axis length (static-max).
#[test]
fn affine_slice_extent_capped_to_axis() {
    let g = graph(vec![
        node("Add", "add_w", &["x", "w"], &["x_plus_w"]),
        node(
            "Slice",
            "slice",
            &["data", "x", "x_plus_w", "axes", "steps"],
            &["sliced"],
        ),
    ]);
    let mut weights = WeightStore::new();
    insert_i64_const(&mut weights, "w", &[100]);
    insert_i64_const(&mut weights, "axes", &[0]);
    insert_i64_const(&mut weights, "steps", &[1]);
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::from([("data".to_string(), vec![64])]);

    let added = augment_inferred_shapes_with_affine_slice_extents(&g, &weights, &mut shapes);
    assert!(added);
    assert_eq!(shapes.get("sliced"), Some(&vec![64]));
}

/// Sub-based ends (`ends = x - (-w)`) also resolve: combine handles negation.
#[test]
fn affine_slice_extent_sub_form() {
    let g = graph(vec![
        node("Sub", "sub_w", &["x", "neg_w"], &["x_plus_w"]),
        node("Unsqueeze", "unsq", &["x_raw"], &["x"]),
        node(
            "Slice",
            "slice",
            &["data", "x", "x_plus_w", "axes", "steps"],
            &["sliced"],
        ),
    ]);
    let mut weights = WeightStore::new();
    insert_i64_const(&mut weights, "neg_w", &[-2]);
    insert_i64_const(&mut weights, "axes", &[0]);
    insert_i64_const(&mut weights, "steps", &[1]);
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::from([("data".to_string(), vec![64])]);

    let added = augment_inferred_shapes_with_affine_slice_extents(&g, &weights, &mut shapes);
    assert!(added);
    assert_eq!(shapes.get("sliced"), Some(&vec![2]));
}
