// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_domain_list_empty() {
    let config = create_test_config();
    let list = DomainList::new(config).unwrap();
    assert!(list.is_empty());
    assert_eq!(list.len(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_new_rejects_missing_layer_shape() {
    let config = DomainListConfig {
        traversal: TreeTraversal::DepthFirst,
        layer_names: vec!["relu0".to_string(), "missing_layer".to_string()],
        layer_shapes: {
            let mut m = HashMap::new();
            m.insert("relu0".to_string(), vec![4]);
            m // "missing_layer" not present
        },
        input_shape: vec![4],
        initial_capacity: 16,
        max_queue_size: 0,
    };
    let result = DomainList::new(config);
    match result {
        Ok(_) => panic!("expected error for missing layer shape"),
        Err(err) => assert!(
            err.to_string().contains("missing_layer"),
            "expected error about missing_layer, got: {err}"
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_pick_out_empty() {
    let config = create_test_config();
    let mut list = DomainList::new(config).unwrap();
    let picked = list.pick_out(5).unwrap();
    assert_eq!(picked.batch_size, 0);
    assert!(picked.metadata.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_filter_batch() {
    let array =
        ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let mask = vec![true, false, true, false];
    let filtered = filter_batch(&array, &mask).unwrap();
    assert_eq!(filtered.shape(), &[2, 2]);
    assert_eq!(filtered[[0, 0]], 1.0);
    assert_eq!(filtered[[1, 0]], 5.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_filter_batch_all_false() {
    let array = ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0; 8]).unwrap();
    let mask = vec![false, false, false, false];
    let filtered = filter_batch(&array, &mask).unwrap();
    assert_eq!(filtered.shape(), &[0, 2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_filter_batch_rejects_mask_length_mismatch() {
    let array = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let err = filter_batch(&array, &[true])
        .expect_err("mask length mismatch must be rejected to avoid silent truncation");
    assert!(
        err.to_string().contains("mask length mismatch"),
        "unexpected error: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pick_out_batched_empty() {
    let config = create_test_config();
    let mut list = DomainList::new(config).unwrap();
    let result = list
        .pick_out_batched(5, BatchedDomainOptions::default())
        .unwrap();
    assert_eq!(result.batch_size, 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_pick_out_batched_preserves_data() {
    // Create a DomainList with matching element shapes.
    let mut layer_shapes = HashMap::new();
    layer_shapes.insert("relu0".to_string(), vec![1]);
    let config = DomainListConfig {
        traversal: TreeTraversal::DepthFirst,
        layer_names: vec!["relu0".to_string()],
        layer_shapes,
        input_shape: vec![4], // Input shape matches the default test config
        initial_capacity: 16,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();

    // Add domains with matching element shape [1] for layer bounds
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![-0.5, -0.3]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "relu0".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0.5, 0.7]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        )
        .unwrap(),
        input_uppers: ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        )
        .unwrap(),
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
        keep_mask: vec![true, true],
    };

    list.add(processed).unwrap();
    assert_eq!(list.len(), 2);

    // Now use pick_out_batched
    let picked = list
        .pick_out_batched(2, BatchedDomainOptions::default())
        .unwrap();

    // Verify PickedDomains
    assert_eq!(picked.batch_size, 2);
    assert_eq!(picked.global_lbs.len(), 2);
    assert_eq!(picked.metadata.len(), 2);

    // Verify layer bounds shape: [batch=2, element=1]
    let relu0_lower = picked.layer_lowers.get("relu0").unwrap();
    assert_eq!(relu0_lower.shape(), &[2, 1]);

    // Verify input bounds shape: [batch=2, input=4]
    assert_eq!(picked.input_lowers.shape(), &[2, 4]);

    // Verify constraint history is preserved
    assert_eq!(picked.metadata[0].constraints.len(), 0);
    assert!(!picked.metadata[1].constraints.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_pick_out_batched_preserves_needs_bounding_metadata_3870() {
    let config = DomainListConfig {
        traversal: TreeTraversal::DepthFirst,
        layer_names: Vec::new(),
        layer_shapes: HashMap::new(),
        input_shape: vec![4],
        initial_capacity: 16,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();

    let processed = ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![0.0; 8]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).unwrap(),
        global_lbs: vec![-0.8, 0.3],
        global_ubs: vec![0.2, 0.9],
        metadata: vec![
            DomainMetadata {
                lower_bound: -0.8,
                upper_bound: 0.2,
                depth: 1,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: 0.3,
                upper_bound: 0.9,
                depth: 2,
                constraints: vec![],
                cached_la: None,
                needs_bounding: true,
                alpha_state: None,
                node_bounds_override: None,
            },
        ],
        keep_mask: vec![true, true],
    };

    list.add(processed).unwrap();
    list.sort_by_domain_priority(false).unwrap();

    let picked = list
        .pick_out_batched(2, BatchedDomainOptions::default())
        .unwrap();

    assert_eq!(picked.metadata.len(), 2);
    assert!(picked.metadata[0].needs_bounding());
    assert!(!picked.metadata[1].needs_bounding());
    assert_eq!(picked.global_lbs, vec![0.3, -0.8]);
    assert_eq!(picked.metadata[0].lower_bound(), 0.3);
    assert_eq!(picked.metadata[1].lower_bound(), -0.8);
}

#[ntest::timeout(10000)]
#[test]
fn test_sort_by_domain_priority_lower_mode_matches_cpu_heap_3870() {
    let config = DomainListConfig {
        traversal: TreeTraversal::BreadthFirst,
        layer_names: Vec::new(),
        layer_shapes: HashMap::new(),
        input_shape: vec![2],
        initial_capacity: 16,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();

    let processed = ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0; 6]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).unwrap(),
        global_lbs: vec![0.4, -0.9, -0.1],
        global_ubs: vec![1.1, 0.3, 0.8],
        metadata: vec![
            DomainMetadata {
                lower_bound: 0.4,
                upper_bound: 1.1,
                depth: 1,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: -0.9,
                upper_bound: 0.3,
                depth: 2,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: -0.1,
                upper_bound: 0.8,
                depth: 3,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
        ],
        keep_mask: vec![true, true, true],
    };

    list.add(processed).unwrap();
    list.sort_by_domain_priority(false).unwrap();

    let picked = list
        .pick_out_batched(3, BatchedDomainOptions::default())
        .unwrap();

    assert_eq!(
        picked.global_lbs,
        vec![-0.9, -0.1, 0.4],
        "lower-bound mode must pop the smallest lower bounds first, matching CPU domain_priority()",
    );
    assert_eq!(picked.metadata[0].depth(), 2);
    assert_eq!(picked.metadata[1].depth(), 3);
    assert_eq!(picked.metadata[2].depth(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_sort_by_domain_priority_upper_mode_matches_cpu_heap_3870() {
    let config = DomainListConfig {
        traversal: TreeTraversal::BreadthFirst,
        layer_names: Vec::new(),
        layer_shapes: HashMap::new(),
        input_shape: vec![2],
        initial_capacity: 16,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();

    let processed = ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0; 6]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).unwrap(),
        global_lbs: vec![-0.2, -0.8, 0.1],
        global_ubs: vec![0.5, 1.4, 0.9],
        metadata: vec![
            DomainMetadata {
                lower_bound: -0.2,
                upper_bound: 0.5,
                depth: 1,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: -0.8,
                upper_bound: 1.4,
                depth: 2,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: 0.1,
                upper_bound: 0.9,
                depth: 3,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
        ],
        keep_mask: vec![true, true, true],
    };

    list.add(processed).unwrap();
    list.sort_by_domain_priority(true).unwrap();

    let picked = list
        .pick_out_batched(3, BatchedDomainOptions::default())
        .unwrap();

    assert_eq!(
        picked.global_ubs,
        vec![1.4, 0.9, 0.5],
        "upper-bound mode must pop the largest upper bounds first, matching CPU domain_priority()",
    );
    assert_eq!(picked.metadata[0].depth(), 2);
    assert_eq!(picked.metadata[1].depth(), 3);
    assert_eq!(picked.metadata[2].depth(), 1);
}

/// Regression for #3870: cached_la (linear bounds for SB scoring) must survive
/// the DomainList add → sort → pick round-trip. If sorting drops or misaligns
/// cached_la, the GPU input-split SB scorer falls back to width-only splitting,
/// which selects the wrong split dimension on lsnc_relu-shaped models.
#[ntest::timeout(10000)]
#[test]
fn test_cached_linear_bounds_survive_sort_roundtrip_3870() {
    use crate::batched_domain::CachedLinearBounds;
    use crate::LinearBounds;
    use ndarray::{arr1 as a1, arr2 as a2};

    let config = DomainListConfig {
        traversal: TreeTraversal::BreadthFirst,
        layer_names: Vec::new(),
        layer_shapes: HashMap::new(),
        input_shape: vec![2],
        initial_capacity: 16,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();

    // Domain A: has linear bounds (simulates a root domain with CROWN coefficients)
    let linear_a = LinearBounds::new(
        a2(&[[1.0_f32, -0.5]]),
        a1(&[0.1_f32]),
        a2(&[[-0.3_f32, 0.8]]),
        a1(&[-0.05_f32]),
    )
    .expect("valid linear bounds");
    let mut cached_a = HashMap::new();
    cached_a.insert("__test_key__".to_string(), linear_a);
    let cached_la_a = CachedLinearBounds::from_linear_bounds_map(cached_a);

    // Domain B: no linear bounds (simulates a deferred child under reorder_bab)
    let processed = ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0; 4]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0; 4]).unwrap(),
        global_lbs: vec![-0.3, -0.8],
        global_ubs: vec![0.5, 0.2],
        metadata: vec![
            DomainMetadata {
                lower_bound: -0.3,
                upper_bound: 0.5,
                depth: 1,
                constraints: vec![],
                cached_la: Some(Arc::new(cached_la_a)),
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: -0.8,
                upper_bound: 0.2,
                depth: 2,
                constraints: vec![],
                cached_la: None,
                needs_bounding: true,
                alpha_state: None,
                node_bounds_override: None,
            },
        ],
        keep_mask: vec![true, true],
    };

    list.add(processed).unwrap();

    // Sort reorders domains: lb=-0.8 has higher priority (score = 0.8) than lb=-0.3 (score = 0.3)
    list.sort_by_domain_priority(false).unwrap();

    let picked = list
        .pick_out_batched(2, BatchedDomainOptions::default())
        .unwrap();

    assert_eq!(picked.metadata.len(), 2);

    // After sort, domain B (lb=-0.8) is first, domain A (lb=-0.3) is second
    assert_eq!(picked.global_lbs, vec![-0.8, -0.3]);

    // Domain B (now at index 0) should have no cached_la
    assert!(
        picked.metadata[0].cached_la().is_none(),
        "domain B should have no cached linear bounds"
    );
    assert!(
        picked.metadata[0].needs_bounding(),
        "domain B should still need bounding"
    );

    // Domain A (now at index 1) should still have its cached linear bounds
    let restored_la = picked.metadata[1]
        .cached_la()
        .as_ref()
        .expect("domain A must retain cached_la after sort")
        .linear_bounds("__test_key__")
        .expect("cached_la must contain the test key after sort round-trip");

    assert_eq!(
        restored_la.lower_a()[[0, 0]],
        1.0,
        "lower_a coefficient must survive sort round-trip"
    );
    assert_eq!(
        restored_la.lower_a()[[0, 1]],
        -0.5,
        "lower_a coefficient must survive sort round-trip"
    );
    assert_eq!(
        restored_la.upper_a()[[0, 1]],
        0.8,
        "upper_a coefficient must survive sort round-trip"
    );
    assert!(!picked.metadata[1].needs_bounding());
}

/// Regression for #4406: sort_by_domain_priority must succeed when the
/// underlying QueueTensorStorage has wrapped (BreadthFirst with add→pick→add
/// cycle). Before this fix, the sort path called `global_lbs.tensor()` which
/// rejects non-contiguous wrapped queue data. The fix derives sort scores from
/// DomainMetadata instead.
#[ntest::timeout(10000)]
#[test]
fn test_sort_by_domain_priority_wrapped_queue_lower_mode_4406() {
    // Use small initial_capacity so the queue wraps after pick + re-add.
    let config = DomainListConfig {
        traversal: TreeTraversal::BreadthFirst,
        layer_names: Vec::new(),
        layer_shapes: HashMap::new(),
        input_shape: vec![2],
        initial_capacity: 4,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();

    // Step 1: add 4 domains to fill the queue to capacity.
    let batch1 = ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![0.0; 8]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0; 8]).unwrap(),
        global_lbs: vec![-0.1, -0.8, -0.3, 0.2],
        global_ubs: vec![0.5, 0.3, 0.7, 1.0],
        metadata: vec![
            DomainMetadata {
                lower_bound: -0.1,
                upper_bound: 0.5,
                depth: 0,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: -0.8,
                upper_bound: 0.3,
                depth: 1,
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
            DomainMetadata {
                lower_bound: 0.2,
                upper_bound: 1.0,
                depth: 2,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
        ],
        keep_mask: vec![true, true, true, true],
    };
    list.add(batch1).unwrap();
    assert_eq!(list.len(), 4);

    // Step 2: pick 2 domains to advance the queue head past the start.
    let _picked = list
        .pick_out_batched(2, BatchedDomainOptions::default())
        .unwrap();
    assert_eq!(list.len(), 2);

    // Step 3: add 2 more domains. The queue storage now wraps around:
    // slots [0,1] are consumed, [2,3] still hold data, new data goes to [0,1].
    let batch2 = ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0; 4]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0; 4]).unwrap(),
        global_lbs: vec![-0.5, 0.4],
        global_ubs: vec![0.6, 1.2],
        metadata: vec![
            DomainMetadata {
                lower_bound: -0.5,
                upper_bound: 0.6,
                depth: 2,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
            DomainMetadata {
                lower_bound: 0.4,
                upper_bound: 1.2,
                depth: 3,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                alpha_state: None,
                node_bounds_override: None,
            },
        ],
        keep_mask: vec![true, true],
    };
    list.add(batch2).unwrap();
    assert_eq!(list.len(), 4);

    // Step 4: sort_by_domain_priority must succeed despite wrapped queue.
    // Before the #4406 fix this panicked with:
    //   "QueueTensorStorage::tensor() requires contiguous data"
    list.sort_by_domain_priority(false).unwrap();

    // Step 5: verify CPU lower-bound priority ordering is preserved.
    // Remaining domains after first pick: lb=-0.3, lb=0.2 (original batch1[2,3])
    // Plus batch2: lb=-0.5, lb=0.4
    // BFS lower-bound sort: smallest lb first → [-0.5, -0.3, 0.2, 0.4]
    let picked = list
        .pick_out_batched(4, BatchedDomainOptions::default())
        .unwrap();

    assert_eq!(picked.batch_size, 4);
    assert_eq!(
        picked.global_lbs,
        vec![-0.5, -0.3, 0.2, 0.4],
        "lower-bound mode must sort smallest lb first in BFS, even with wrapped queue (#4406)",
    );
    // Metadata lower_bound must match sorted order.
    let meta_lbs: Vec<f32> = picked.metadata.iter().map(|m| m.lower_bound()).collect();
    assert_eq!(meta_lbs, vec![-0.5, -0.3, 0.2, 0.4]);
}
