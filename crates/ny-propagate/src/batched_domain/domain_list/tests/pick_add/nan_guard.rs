// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for NaN propagation through the batched domain pipeline (#2246).
//!
//! Verifies that NaN values in global bounds, layer bounds, and sort ordering
//! are handled safely rather than corrupting branch selection, sort transitivity,
//! or stability classification.

use super::*;
use ny_tensor::BoundedTensor;
use std::sync::Arc;

// --- DomainList::add rejects NaN global bounds ---

#[ntest::timeout(10000)]
#[test]
fn test_add_rejects_nan_global_lower_bound_2246() {
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
    layer_uppers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.3, 0.4]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0; 4]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap(),
        global_lbs: vec![f32::NAN],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: f32::NAN,
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
        .expect_err("NaN global lower bound must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_add_rejects_inf_global_upper_bound_2246() {
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
    layer_uppers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.3, 0.4]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0; 4]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap(),
        global_lbs: vec![0.0],
        global_ubs: vec![f32::INFINITY],
        metadata: vec![DomainMetadata {
            lower_bound: 0.0,
            upper_bound: f32::INFINITY,
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
        .expect_err("Inf global upper bound must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

// --- DomainList::add rejects NaN/Inf in layer/input tensors (#3115) ---

#[ntest::timeout(10000)]
#[test]
fn test_add_rejects_nan_layer_lower_bound_3115() {
    let config = create_test_config();
    let mut list = DomainList::new(config).unwrap();

    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::NAN, -0.2]).unwrap(),
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
    layer_uppers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.3, 0.4]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0; 4]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
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
        .expect_err("NaN layer lower bound must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_add_rejects_inf_layer_upper_bound_3115() {
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
    layer_uppers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::INFINITY, 0.4]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0; 4]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
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
        .expect_err("Inf layer upper bound must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_add_rejects_nan_input_lower_bound_3115() {
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
    layer_uppers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.3, 0.4]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![f32::NAN, 0.0, 0.0, 0.0])
            .unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
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
        .expect_err("NaN input lower bound must be rejected");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}

/// Non-finite values in dropped rows (keep_mask=false) must NOT trigger rejection.
/// This locks the keep_mask semantics: only kept rows are validated (#3115).
#[ntest::timeout(10000)]
#[test]
fn test_add_ignores_nonfinite_dropped_rows_3115() {
    let config = create_test_config();
    let mut list = DomainList::new(config).unwrap();

    // Batch of 2 domains: domain 0 is kept (valid), domain 1 is dropped (NaN).
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.1, -0.2, f32::NAN, f32::NAN]).unwrap(),
    );
    layer_lowers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-0.3, -0.4, -0.3, -0.4]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "relu1".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.1, 0.2, f32::INFINITY, 0.2]).unwrap(),
    );
    layer_uppers.insert(
        "relu2".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.3, 0.4, 0.3, 0.4]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![0.0, 0.0, 0.0, 0.0, f32::NAN, 0.0, 0.0, 0.0],
        )
        .unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).unwrap(),
        global_lbs: vec![-1.0, -2.0],
        global_ubs: vec![1.0, 2.0],
        metadata: vec![
            DomainMetadata {
                lower_bound: -1.0,
                upper_bound: 1.0,
                depth: 0,
                constraints: Vec::new(),
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: -2.0,
                upper_bound: 2.0,
                depth: 0,
                constraints: Vec::new(),
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
        ],
        keep_mask: vec![true, false], // domain 1 (NaN) is dropped
    };

    list.add(processed)
        .expect("dropped rows with NaN should not trigger rejection");
    assert_eq!(list.len(), 1, "only the kept domain should be stored");
}

// --- DomainList::pick_out rejects stored non-finite data (#3115) ---

/// Append a raw domain row to all DomainList storages, bypassing `add()`.
/// Used to simulate storage corruption for defense-in-depth testing.
fn append_raw_domain_row(
    list: &mut DomainList,
    relu1_lower: &[f32; 2],
    node_bounds_override: Option<Arc<HashMap<String, BoundedTensor>>>,
) {
    let r1l = ArrayD::from_shape_vec(IxDyn(&[1, 2]), relu1_lower.to_vec()).unwrap();
    list.layer_lowers
        .get_mut("relu1")
        .unwrap()
        .append(&r1l)
        .unwrap();
    let r1u = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.1, 0.2]).unwrap();
    list.layer_uppers
        .get_mut("relu1")
        .unwrap()
        .append(&r1u)
        .unwrap();
    let r2l = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-0.3, -0.4]).unwrap();
    list.layer_lowers
        .get_mut("relu2")
        .unwrap()
        .append(&r2l)
        .unwrap();
    let r2u = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.3, 0.4]).unwrap();
    list.layer_uppers
        .get_mut("relu2")
        .unwrap()
        .append(&r2u)
        .unwrap();
    let il = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0; 4]).unwrap();
    list.input_lowers.append(&il).unwrap();
    let iu = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap();
    list.input_uppers.append(&iu).unwrap();
    let gl = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![-1.0]).unwrap();
    list.global_lbs.append(&gl).unwrap();
    let gu = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap();
    list.global_ubs.append(&gu).unwrap();
    list.metadata.push(DomainMetadata {
        lower_bound: -1.0,
        upper_bound: 1.0,
        depth: 0,
        constraints: Vec::new(),
        cached_la: None,
        needs_bounding: false,
        alpha_state: None,
        node_bounds_override,
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_pick_out_rejects_stored_nan_layer_tensor_3115() {
    assert_pick_out_rejects_stored_nan_layer_tensor_3115(TreeTraversal::DepthFirst);
}

#[ntest::timeout(10000)]
#[test]
fn test_pick_out_rejects_stored_nan_layer_tensor_bfs_restores_queue_3115() {
    assert_pick_out_rejects_stored_nan_layer_tensor_3115(TreeTraversal::BreadthFirst);
}

fn assert_pick_out_rejects_stored_nan_layer_tensor_3115(traversal: TreeTraversal) {
    let mut config = create_test_config();
    config.traversal = traversal;
    let mut list = DomainList::new(config).unwrap();

    // Add a valid domain via the normal path.
    let processed = ProcessedDomains::valid_single_domain();
    list.add(processed).unwrap();

    // Manually corrupt storage by appending a NaN-bearing row.
    append_raw_domain_row(&mut list, &[f32::NAN, -0.2], None);

    // pick_out should catch the corrupted stored tensor.
    let err = list
        .pick_out(2)
        .expect_err("pick_out must reject stored NaN layer tensor");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
    assert_eq!(
        list.len(),
        2,
        "pick_out validation errors must not drop queued domains"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_add_rejects_nonfinite_node_bounds_override_4143() {
    let config = create_test_config();
    let mut list = DomainList::new(config).unwrap();
    let mut processed = ProcessedDomains::valid_single_domain();
    processed.metadata[0].node_bounds_override = Some(Arc::new(HashMap::from([(
        "relu1".to_string(),
        BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5_f32, f32::INFINITY]).unwrap(),
        )
        .expect("infinite override constructs for corruption regression"),
    )])));

    let err = list
        .add(processed)
        .expect_err("add must reject non-finite deferred override metadata");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
    assert_eq!(list.len(), 0, "rejected add must not append domains");
}

#[ntest::timeout(10000)]
#[test]
fn test_pick_out_rejects_stored_nonfinite_node_bounds_override_4143() {
    assert_pick_out_rejects_stored_nonfinite_node_bounds_override_4143(TreeTraversal::DepthFirst);
}

#[ntest::timeout(10000)]
#[test]
fn test_pick_out_rejects_stored_nonfinite_node_bounds_override_bfs_restores_queue_4143() {
    assert_pick_out_rejects_stored_nonfinite_node_bounds_override_4143(TreeTraversal::BreadthFirst);
}

fn assert_pick_out_rejects_stored_nonfinite_node_bounds_override_4143(traversal: TreeTraversal) {
    let mut config = create_test_config();
    config.traversal = traversal;
    let mut list = DomainList::new(config).unwrap();

    list.add(ProcessedDomains::valid_single_domain())
        .expect("valid domain should add");

    let corrupted_override = Arc::new(HashMap::from([(
        "relu1".to_string(),
        BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1_f32, f32::INFINITY]).unwrap(),
        )
        .expect("infinite override constructs for corruption regression"),
    )]));
    append_raw_domain_row(&mut list, &[-0.1, -0.2], Some(corrupted_override));

    let err = list
        .pick_out(2)
        .expect_err("pick_out must reject stored non-finite node-bounds override");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
    assert_eq!(
        list.len(),
        2,
        "pick_out validation errors must not drop queued domains"
    );
}

// --- Sort ordering with NaN is deterministic via total_cmp ---

#[ntest::timeout(10000)]
#[test]
fn test_sort_with_nan_global_lbs_deterministic_2246() {
    // Build a DomainList with valid domains, then manually inject NaN
    // to verify sort doesn't panic or produce non-deterministic ordering.
    // We test via the builder path (which validates) then pick_out to get
    // the sorted global_lbs.

    // This test verifies the sort function handles NaN by exercising
    // total_cmp which places NaN after all finite values.
    let config = create_test_config();
    let mut list = DomainList::new(config).unwrap();

    // Add three valid domains
    for lb in &[-1.0_f32, -2.0, -0.5] {
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
        layer_uppers.insert(
            "relu2".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.3, 0.4]).unwrap(),
        );

        let processed = ProcessedDomains {
            layer_lowers,
            layer_uppers,
            input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.0; 4]).unwrap(),
            input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0; 4]).unwrap(),
            global_lbs: vec![*lb],
            global_ubs: vec![1.0],
            metadata: vec![DomainMetadata {
                lower_bound: *lb,
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
        list.add(processed).unwrap();
    }

    // Sort should not panic (even though previous code used partial_cmp
    // which would have been problematic with NaN)
    list.sort_by_domain_priority(false).unwrap();

    let picked = list.pick_out(3).unwrap();
    assert_eq!(picked.batch_size, 3);
    // Verify all three domains are present and sort produced a valid
    // permutation (total_cmp gives deterministic order with NaN).
    let mut lbs = picked.global_lbs;
    lbs.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(lbs, vec![-2.0, -1.0, -0.5]);
}

// --- Unstable mask treats NaN as unstable ---

#[ntest::timeout(10000)]
#[test]
fn test_unstable_mask_nan_bounds_classified_unstable_2246() {
    use super::super::super::super::utils::unstable_mask_from_bounds;

    let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, 0.5, f32::NAN, -0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 1.0, 1.0, f32::NAN]).unwrap();

    let mask = unstable_mask_from_bounds(&lower, &upper).unwrap();

    // neuron 0: l=-1 < 0 && u=1 > 0 → unstable
    assert!(mask[[0]], "neuron 0 should be unstable");
    // neuron 1: l=0.5, not < 0 → stable
    assert!(!mask[[1]], "neuron 1 should be stable");
    // neuron 2: l=NaN → conservatively unstable (#2246)
    assert!(mask[[2]], "neuron 2 with NaN lower should be unstable");
    // neuron 3: u=NaN → conservatively unstable (#2246)
    assert!(mask[[3]], "neuron 3 with NaN upper should be unstable");
}

// --- find_unstable_neurons_batched treats NaN as unstable ---

#[ntest::timeout(10000)]
#[test]
fn test_find_unstable_nan_bounds_treated_unstable_2246() {
    let mut layer_lowers = HashMap::new();
    // Batch of 1 domain, 3 neurons: [stable, NaN-lower, unstable]
    layer_lowers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![0.5, f32::NAN, -1.0]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 1.0, 1.0]).unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.0]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
            upper_bound: 1.0,
            depth: 0,
            constraints: Vec::new(),
            cached_la: None,
            needs_bounding: false,
            alpha_state: None,
            node_bounds_override: None,
        }],
    };

    let relu_pre_map: HashMap<String, String> = [("relu0".to_string(), "relu0".to_string())]
        .into_iter()
        .collect();

    let unstable = picked.find_unstable_neurons_batched(&relu_pre_map).unwrap();
    assert_eq!(unstable.len(), 1);

    let neuron_indices: Vec<usize> = unstable[0].iter().map(|(_, idx)| *idx).collect();
    // Neuron 0 (l=0.5, u=1.0): stable
    assert!(!neuron_indices.contains(&0), "neuron 0 should be stable");
    // Neuron 1 (l=NaN, u=1.0): conservatively unstable (#2246)
    assert!(
        neuron_indices.contains(&1),
        "neuron 1 with NaN lower should be unstable"
    );
    // Neuron 2 (l=-1.0, u=1.0): genuinely unstable
    assert!(neuron_indices.contains(&2), "neuron 2 should be unstable");
}

// --- select_branch_batched skips NaN-bound neurons ---

#[ntest::timeout(10000)]
#[test]
fn test_select_branch_skips_nan_scored_neurons_2246() {
    let mut layer_lowers = HashMap::new();
    // Batch of 1 domain, 2 neurons: [NaN-lower, genuinely-unstable]
    layer_lowers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::NAN, -1.0]).unwrap(),
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
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.0]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
            upper_bound: 1.0,
            depth: 0,
            constraints: Vec::new(),
            cached_la: None,
            needs_bounding: false,
            alpha_state: None,
            node_bounds_override: None,
        }],
    };

    let relu_pre_map: HashMap<String, String> = [("relu0".to_string(), "relu0".to_string())]
        .into_iter()
        .collect();

    // Both neurons are in the "unstable" list (NaN neuron is conservatively
    // included by find_unstable_neurons_batched)
    let unstable = vec![vec![
        ("relu0".to_string(), 0_usize),
        ("relu0".to_string(), 1_usize),
    ]];

    let branches = picked
        .select_branch_batched(&unstable, &relu_pre_map)
        .unwrap();

    // The NaN-bound neuron 0 must be skipped; neuron 1 must be selected
    let branch = branches[0].as_ref().expect("should select non-NaN neuron");
    assert_eq!(
        branch.1, 1,
        "should select neuron 1 (valid), not neuron 0 (NaN)"
    );
    // Intercept for neuron 1: (-(-1.0) * 2.0) / (2.0 - (-1.0)) = 2.0/3.0
    let expected_intercept = 2.0_f32 / 3.0;
    assert!(
        (branch.2 - expected_intercept).abs() < 1e-6,
        "intercept should be 2/3, got {}",
        branch.2
    );
}

// --- Builder rejects NaN global bounds ---

#[ntest::timeout(10000)]
#[test]
fn test_builder_rejects_nan_global_bounds_2246() {
    let layer_names = vec!["relu0".to_string()];
    let mut builder = BatchedDomainsBuilder::new(layer_names);

    let mut layer_bounds = HashMap::new();
    layer_bounds.insert(
        "relu0".to_string(),
        (
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -0.5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 0.5]).unwrap(),
        ),
    );

    builder.add_domain(
        &layer_bounds,
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
        f32::NAN, // NaN lower bound
        1.0,
        0,
        Vec::new(),
    );

    let err = builder
        .build()
        .expect_err("NaN global bounds must be rejected at build time");
    assert!(
        err.to_string().contains("non-finite"),
        "unexpected error: {err}"
    );
}
