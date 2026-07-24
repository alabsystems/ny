// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ndarray::{array, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::batched_domain::{DomainMetadata, PickedDomains};

use super::tests::{make_relu_graph, make_simple_picked};
use super::{create_root_processed_domain, graph_domain_from_picked};

#[test]
fn test_create_root_processed_domain_basic() {
    let mut node_bounds = HashMap::new();
    let bounds = BoundedTensor::new(
        array![-1.0f32, 0.0].into_dyn(),
        array![1.0f32, 2.0].into_dyn(),
    )
    .unwrap();
    node_bounds.insert("layer0".to_string(), bounds);

    let input = BoundedTensor::new(
        array![0.0f32, -0.5].into_dyn(),
        array![1.0f32, 0.5].into_dyn(),
    )
    .unwrap();

    let layer_names = vec!["layer0".to_string()];
    let result = create_root_processed_domain(&node_bounds, &input, -1.0, 2.0, &layer_names)
        .expect("should create root domain");

    assert_eq!(result.input_lowers.shape()[0], 1);
    assert_eq!(result.input_uppers.shape()[0], 1);
    assert_eq!(result.global_lbs, vec![-1.0]);
    assert_eq!(result.global_ubs, vec![2.0]);
    assert_eq!(result.metadata.len(), 1);
    assert!(result.keep_mask[0]);

    let layer_lower = result.layer_lowers.get("layer0").unwrap();
    assert_eq!(layer_lower.shape(), &[1, 2]);
}

#[test]
fn test_create_root_processed_domain_empty_layers() {
    let node_bounds = HashMap::new();
    let input = BoundedTensor::new(array![0.0f32].into_dyn(), array![1.0f32].into_dyn()).unwrap();

    let result = create_root_processed_domain(&node_bounds, &input, 0.0, 1.0, &[])
        .expect("empty layers should succeed");
    assert!(result.layer_lowers.is_empty());
    assert_eq!(result.metadata.len(), 1);
}

#[test]
fn test_create_root_processed_domain_nan_bounds_rejected() {
    let node_bounds = HashMap::new();
    let input = BoundedTensor::new(array![0.0f32].into_dyn(), array![1.0f32].into_dyn()).unwrap();

    let err = create_root_processed_domain(&node_bounds, &input, f32::NAN, 1.0, &[]);
    assert!(err.is_err(), "NaN lower bound should be rejected");
}

#[test]
fn test_graph_domain_from_picked_basic() {
    let picked = make_simple_picked(
        &[0.0, -0.5],
        &[1.0, 0.5],
        &[-1.0, 0.0],
        &[1.0, 2.0],
        "pre_relu",
        -1.0,
        2.0,
    );
    let layer_names = vec!["pre_relu".to_string()];

    let domain = graph_domain_from_picked(0, &picked, &layer_names, false, None)
        .expect("basic extraction should succeed");

    assert_eq!(domain.lower_bound, -1.0);
    assert_eq!(domain.upper_bound, 2.0);
    assert_eq!(domain.depth, 0);
    assert!(domain.node_bounds.contains_key("pre_relu"));
}

#[test]
fn test_graph_domain_from_picked_out_of_bounds_idx() {
    let picked = make_simple_picked(&[0.0], &[1.0], &[-1.0], &[1.0], "layer0", 0.0, 1.0);
    let layer_names = vec!["layer0".to_string()];

    let err = graph_domain_from_picked(1, &picked, &layer_names, false, None);
    assert!(err.is_err(), "idx out of bounds should fail");
}

#[test]
fn test_graph_domain_from_picked_missing_layer() {
    let picked = make_simple_picked(&[0.0], &[1.0], &[-1.0], &[1.0], "layer0", 0.0, 1.0);
    let layer_names = vec!["nonexistent".to_string()];

    let err = graph_domain_from_picked(0, &picked, &layer_names, false, None);
    assert!(err.is_err(), "missing layer should produce error");
}

#[test]
fn test_graph_domain_from_picked_with_graph_alpha_init() {
    let picked = make_simple_picked(
        &[0.0, -0.5],
        &[1.0, 0.5],
        &[-1.0, 0.0],
        &[1.0, 2.0],
        "relu0",
        -1.0,
        2.0,
    );
    let layer_names = vec!["relu0".to_string()];
    let graph = make_relu_graph("relu0");

    let domain = graph_domain_from_picked(0, &picked, &layer_names, false, Some(&graph))
        .expect("extraction with graph should succeed");

    assert_eq!(domain.lower_bound, -1.0);
}

#[test]
fn test_select_input_split_dimension_empty_input_errors() {
    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 0]), vec![]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 0]), vec![]).unwrap(),
        global_lbs: vec![0.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata::root(0.0, 1.0).unwrap()],
    };

    let err = super::select_input_split_dimension(&picked, 0);
    assert!(err.is_err(), "empty input should fail");
}
