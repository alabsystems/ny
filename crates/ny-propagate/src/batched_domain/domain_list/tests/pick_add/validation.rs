// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_add_rejects_missing_configured_layer_bounds() {
    let config = create_test_config();
    let mut list = DomainList::new(config).unwrap();

    let processed = ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
        global_lbs: vec![0.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: 0.0,
            upper_bound: 1.0,
            depth: 0,
            constraints: Vec::new(),
            cached_la: None,
            needs_bounding: false,
            alpha_state: None,
            node_bounds_override: None,
        }],
        keep_mask: vec![true],
    };

    let err = list
        .add(processed)
        .expect_err("missing configured layer bounds must be rejected");
    assert!(
        err.to_string().contains("missing layer bounds"),
        "unexpected error: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_add_rejects_incomplete_configured_layer_bounds() {
    let config = create_test_config();
    let mut list = DomainList::new(config).unwrap();

    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-0.1, -0.2]).unwrap(),
    );
    layer_lowers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-0.3, -0.4]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.1, 0.2]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
        global_lbs: vec![0.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: 0.0,
            upper_bound: 1.0,
            depth: 0,
            constraints: Vec::new(),
            cached_la: None,
            needs_bounding: false,
            alpha_state: None,
            node_bounds_override: None,
        }],
        keep_mask: vec![true],
    };

    let err = list
        .add(processed)
        .expect_err("incomplete configured layer bounds must be rejected");
    assert!(
        err.to_string().contains("incomplete layer bounds"),
        "unexpected error: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_add_rejects_unconfigured_layer_bounds() {
    let config = create_test_config();
    let mut list = DomainList::new(config).unwrap();

    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-0.1, -0.2]).unwrap(),
    );
    layer_lowers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-0.3, -0.4]).unwrap(),
    );
    layer_lowers.insert(
        "relu_extra".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-0.5, -0.6]).unwrap(),
    );

    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.1, 0.2]).unwrap(),
    );
    layer_uppers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.3, 0.4]).unwrap(),
    );
    layer_uppers.insert(
        "relu_extra".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.5, 0.6]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
        global_lbs: vec![0.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: 0.0,
            upper_bound: 1.0,
            depth: 0,
            constraints: Vec::new(),
            cached_la: None,
            needs_bounding: false,
            alpha_state: None,
            node_bounds_override: None,
        }],
        keep_mask: vec![true],
    };

    let err = list
        .add(processed)
        .expect_err("unconfigured layer bounds must be rejected");
    assert!(
        err.to_string().contains("unconfigured layer"),
        "unexpected error: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_add_rejects_keep_mask_length_mismatch() {
    let config = create_test_config();
    let mut list = DomainList::new(config).unwrap();

    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.1, -0.2, -0.3, -0.4]).unwrap(),
    );
    layer_lowers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.5, -0.6, -0.7, -0.8]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
    );
    layer_uppers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 0.6, 0.7, 0.8]).unwrap(),
    );
    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap(),
        input_uppers: ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap(),
        global_lbs: vec![0.0, 0.1],
        global_ubs: vec![1.0, 1.1],
        metadata: vec![
            DomainMetadata {
                lower_bound: 0.0,
                upper_bound: 1.0,
                depth: 0,
                constraints: Vec::new(),
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: 0.1,
                upper_bound: 1.1,
                depth: 0,
                constraints: Vec::new(),
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
        ],
        keep_mask: vec![true],
    };

    let err = list
        .add(processed)
        .expect_err("keep_mask length mismatch must be rejected");
    assert!(
        err.to_string().contains("keep_mask length mismatch"),
        "unexpected error: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_domain_list_respects_per_layer_shapes() {
    let mut layer_shapes = HashMap::new();
    layer_shapes.insert("relu1".to_string(), vec![2]);
    layer_shapes.insert("relu2".to_string(), vec![1]);
    let config = DomainListConfig {
        traversal: TreeTraversal::DepthFirst,
        layer_names: vec!["relu1".to_string(), "relu2".to_string()],
        layer_shapes,
        input_shape: vec![4],
        initial_capacity: 4,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();

    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-0.5, -0.3]).unwrap(),
    );
    layer_lowers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![-0.2]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.5, 0.7]).unwrap(),
    );
    layer_uppers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.4]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0, 0.1, 0.2, 0.3]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 1.1, 1.2, 1.3]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![0.5],
        metadata: vec![DomainMetadata {
            lower_bound: -0.5,
            upper_bound: 0.5,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            alpha_state: None,
            node_bounds_override: None,
        }],
        keep_mask: vec![true],
    };

    list.add(processed).unwrap();
    let picked = list.pick_out(1).unwrap();

    let relu1 = picked.layer_lowers.get("relu1").unwrap();
    let relu2 = picked.layer_lowers.get("relu2").unwrap();
    assert_eq!(relu1.shape(), &[1, 2]);
    assert_eq!(relu2.shape(), &[1, 1]);
}
