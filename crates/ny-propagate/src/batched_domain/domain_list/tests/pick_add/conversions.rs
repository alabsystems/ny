// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_from_picked_domains_basic() {
    // Create a PickedDomains manually
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, -0.5, 0.0, -2.0, -1.5, -1.0]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 1.5, 2.0, 2.0, 2.5, 3.0]).unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 2,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 0.1, 0.2, 0.3]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 1.1, 1.2, 1.3]).unwrap(),
        global_lbs: vec![-0.5, -0.3],
        global_ubs: vec![0.5, 0.7],
        metadata: vec![
            DomainMetadata {
                lower_bound: -0.5,
                upper_bound: 0.5,
                depth: 0,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: -0.3,
                upper_bound: 0.7,
                depth: 1,
                constraints: vec![("relu0".to_string(), 1, true, None)],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
        ],
    };

    let batched = BatchedDomains::from_picked_domains(picked);

    assert_eq!(batched.batch_size(), 2);
    assert_eq!(batched.lower_bounds(), &[-0.5, -0.3]);
    assert_eq!(batched.upper_bounds(), &[0.5, 0.7]);
    assert_eq!(batched.depths(), &[0, 1]);

    // Verify layer bounds wrapped correctly
    let relu0_lower = batched.layer_lowers().get("relu0").unwrap();
    assert_eq!(relu0_lower.as_array().shape(), &[2, 3]);
    assert_eq!(relu0_lower.as_array()[[0, 0]], -1.0);
    assert_eq!(relu0_lower.as_array()[[1, 2]], -1.0);

    // Verify constraints preserved
    assert!(batched.constraints()[0].is_empty());
    assert_eq!(
        batched.constraints()[1],
        vec![("relu0".to_string(), 1, true, None)]
    );
}

/// Build a 2-domain PickedDomains fixture for interm_transfer tests.
/// Domain 0 layer "relu0": lb=[-1.0, -0.5, 0.5], ub=[1.0, 1.5, 2.0]
///   → neurons 0,1 unstable; neuron 2 stable-positive
fn make_two_domain_picked() -> PickedDomains {
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, -0.5, 0.5, -2.0, -1.5, -1.0]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 1.5, 2.0, 2.0, 2.5, 3.0]).unwrap(),
    );
    PickedDomains {
        batch_size: 2,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 0.1, 0.2, 0.3]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 1.1, 1.2, 1.3]).unwrap(),
        global_lbs: vec![-0.5, -0.3],
        global_ubs: vec![0.5, 0.7],
        metadata: vec![
            DomainMetadata {
                lower_bound: -0.5,
                upper_bound: 0.5,
                depth: 0,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: -0.3,
                upper_bound: 0.7,
                depth: 1,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
        ],
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_from_picked_domains_with_interm_transfer() {
    // Verify that from_picked_domains_with_options populates static bounds
    // and unstable masks when enable_interm_transfer=true. Fixes #1655.
    let picked = make_two_domain_picked();
    let options = BatchedDomainOptions {
        enable_interm_transfer: true,
    };
    let batched = BatchedDomains::from_picked_domains_with_options(picked, options);
    assert_eq!(batched.batch_size(), 2);

    // Static bounds should be populated from batch index 0
    let static_lowers = batched
        .static_layer_lowers()
        .expect("static_layer_lowers should be Some");
    let static_uppers = batched
        .static_layer_uppers()
        .expect("static_layer_uppers should be Some");
    let masks = batched
        .unstable_masks()
        .expect("unstable_masks should be Some");

    // Static bounds come from batch index 0: [-1.0, -0.5, 0.5] / [1.0, 1.5, 2.0]
    let sl = static_lowers.get("relu0").expect("relu0 static lower");
    let su = static_uppers.get("relu0").expect("relu0 static upper");
    assert_eq!(sl.as_array().shape(), &[3]);
    assert_eq!(sl.as_array()[[0]], -1.0);
    assert_eq!(sl.as_array()[[1]], -0.5);
    assert_eq!(sl.as_array()[[2]], 0.5);
    assert_eq!(su.as_array()[[0]], 1.0);
    assert_eq!(su.as_array()[[1]], 1.5);
    assert_eq!(su.as_array()[[2]], 2.0);

    // Unstable mask: neurons 0,1 unstable (lb<0<ub), neuron 2 stable (lb>0)
    let relu0_mask = masks.get("relu0").expect("relu0 unstable mask");
    assert_eq!(relu0_mask.shape(), &[3]);
    assert!(relu0_mask[[0]], "neuron 0 should be unstable");
    assert!(relu0_mask[[1]], "neuron 1 should be unstable");
    assert!(!relu0_mask[[2]], "neuron 2 should be stable (lb > 0)");

    // Sparse-to-dense: neurons 0 and 1 are unstable
    let indices = batched
        .sparse_to_dense_indices("relu0")
        .expect("sparse indices");
    assert_eq!(indices, vec![0, 1]);
    assert_eq!(batched.unstable_count("relu0"), Some(2));
}

#[ntest::timeout(10000)]
#[test]
fn test_from_picked_domains_without_interm_transfer() {
    // Verify that from_picked_domains (default options) does NOT populate static bounds.
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-1.0, 0.5]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.1]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.1]).unwrap(),
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
    };

    let batched = BatchedDomains::from_picked_domains(picked);

    assert_eq!(batched.batch_size(), 1);
    assert!(
        batched.static_layer_lowers().is_none(),
        "static bounds should be None with default options"
    );
    assert!(batched.static_layer_uppers().is_none());
    assert!(batched.unstable_masks().is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_from_picked_domains_empty() {
    let picked = PickedDomains {
        batch_size: 0,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::zeros(IxDyn(&[0, 2])),
        input_uppers: ArrayD::zeros(IxDyn(&[0, 2])),
        global_lbs: vec![],
        global_ubs: vec![],
        metadata: vec![],
    };

    let batched = BatchedDomains::from_picked_domains(picked);

    assert!(batched.is_empty());
    assert_eq!(batched.batch_size(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_processed_domains_from_batched_results() {
    let mut builder = BatchedDomainsBuilder::new(vec!["relu0".to_string()]);
    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, -0.5, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 1.5, 2.0]).unwrap(),
        ),
    );

    builder.add_domain(
        &layer_bounds,
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.1]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.1]).unwrap(),
        -0.5,
        0.5,
        0,
        vec![],
    );
    builder.add_domain(
        &layer_bounds,
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.2, 0.3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.2, 1.3]).unwrap(),
        -0.3,
        0.7,
        1,
        vec![("relu0".to_string(), 1, true, None)],
    );

    let batched = builder.build().unwrap();
    let new_lower_bounds = vec![-0.4, -0.2];
    let new_upper_bounds = vec![0.4, 0.6];

    let mut new_layer_lowers = HashMap::new();
    new_layer_lowers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-0.9, -0.4, 0.1, -0.8, -0.3, 0.2]).unwrap(),
    );
    let mut new_layer_uppers = HashMap::new();
    new_layer_uppers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.9, 1.4, 1.9, 0.8, 1.3, 1.8]).unwrap(),
    );

    let new_input_lowers =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.05, 0.15, 0.25, 0.35]).unwrap();
    let new_input_uppers =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.95, 1.05, 1.15, 1.25]).unwrap();

    let keep_mask = vec![true, false]; // Keep first, discard second

    let processed = ProcessedDomains::from_batched_results(
        &batched,
        new_lower_bounds.clone(),
        new_upper_bounds.clone(),
        new_layer_lowers,
        new_layer_uppers,
        new_input_lowers.clone(),
        new_input_uppers.clone(),
        keep_mask.clone(),
    )
    .expect("test: NaN-free bounds");

    // Verify metadata was constructed correctly
    assert_eq!(processed.metadata.len(), 2);
    assert_eq!(processed.metadata[0].lower_bound, -0.4);
    assert_eq!(processed.metadata[0].upper_bound, 0.4);
    assert_eq!(processed.metadata[0].depth, 0);
    assert!(processed.metadata[0].constraints.is_empty());

    assert_eq!(processed.metadata[1].lower_bound, -0.2);
    assert_eq!(processed.metadata[1].upper_bound, 0.6);
    assert_eq!(processed.metadata[1].depth, 1);
    assert_eq!(
        processed.metadata[1].constraints,
        vec![("relu0".to_string(), 1, true, None)]
    );

    // Verify bounds were stored
    assert_eq!(processed.global_lbs, new_lower_bounds);
    assert_eq!(processed.global_ubs, new_upper_bounds);
    assert_eq!(processed.input_lowers, new_input_lowers);
    assert_eq!(processed.input_uppers, new_input_uppers);

    // Verify keep_mask preserved
    assert_eq!(processed.keep_mask, keep_mask);
}

#[ntest::timeout(10000)]
#[test]
fn test_cached_linear_bounds_from_linear_bounds_map() {
    use crate::LinearBounds;
    use ndarray::{Array1, Array2};

    // Create a map of LinearBounds
    let mut linear_bounds_map: HashMap<String, LinearBounds> = HashMap::new();

    let lb1 = LinearBounds {
        lower_a: Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        lower_b: Array1::from_vec(vec![0.1, 0.2]),
        upper_a: Array2::from_shape_vec((2, 3), vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1]).unwrap(),
        upper_b: Array1::from_vec(vec![0.3, 0.4]),
        lower_a_err: None,
        upper_a_err: None,
    };
    linear_bounds_map.insert("relu0".to_string(), lb1);

    let lb2 = LinearBounds {
        lower_a: Array2::from_shape_vec((2, 2), vec![7.0, 8.0, 9.0, 10.0]).unwrap(),
        lower_b: Array1::from_vec(vec![0.5, 0.6]),
        upper_a: Array2::from_shape_vec((2, 2), vec![7.1, 8.1, 9.1, 10.1]).unwrap(),
        upper_b: Array1::from_vec(vec![0.7, 0.8]),
        lower_a_err: None,
        upper_a_err: None,
    };
    linear_bounds_map.insert("relu1".to_string(), lb2);

    // Convert to CachedLinearBounds
    let cached = CachedLinearBounds::from_linear_bounds_map(linear_bounds_map);

    // Verify the conversion
    assert_eq!(cached.len(), 2);
    assert!(!cached.is_empty());

    // Check relu0 matrices
    let relu0_lower = cached.lower_a.get("relu0").unwrap();
    assert_eq!(relu0_lower.shape(), &[2, 3]);
    assert_eq!(relu0_lower[[0, 0]], 1.0);
    assert_eq!(relu0_lower[[1, 2]], 6.0);

    let relu0_upper = cached.upper_a.get("relu0").unwrap();
    assert_eq!(relu0_upper.shape(), &[2, 3]);
    assert_eq!(relu0_upper[[0, 0]], 1.1);

    // Check relu1 matrices
    let relu1_lower = cached.lower_a.get("relu1").unwrap();
    assert_eq!(relu1_lower.shape(), &[2, 2]);
    assert_eq!(relu1_lower[[0, 0]], 7.0);
}
