// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn metadata(lower_bound: f32, upper_bound: f32, depth: usize) -> DomainMetadata {
    DomainMetadata {
        lower_bound,
        upper_bound,
        depth,
        constraints: vec![],
        cached_la: None,
        needs_bounding: false,
        alpha_state: None,
        node_bounds_override: None,
    }
}

/// Regression for #4406: upper-bound priority sorting must also tolerate
/// wrapped queue storage. The production fix reads score scalars from
/// DomainMetadata in both lower-bound and upper-bound modes.
#[ntest::timeout(10000)]
#[test]
fn test_sort_by_domain_priority_wrapped_queue_upper_mode_4406() {
    let config = DomainListConfig {
        traversal: TreeTraversal::BreadthFirst,
        layer_names: Vec::new(),
        layer_shapes: HashMap::new(),
        input_shape: vec![2],
        initial_capacity: 4,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();

    let batch1 = ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![0.0; 8]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0; 8]).unwrap(),
        global_lbs: vec![-0.1, -0.8, -0.3, 0.2],
        global_ubs: vec![0.5, 0.3, 0.7, 1.0],
        metadata: vec![
            metadata(-0.1, 0.5, 0),
            metadata(-0.8, 0.3, 1),
            metadata(-0.3, 0.7, 1),
            metadata(0.2, 1.0, 2),
        ],
        keep_mask: vec![true, true, true, true],
    };
    list.add(batch1).unwrap();

    let _picked = list
        .pick_out_batched(2, BatchedDomainOptions::default())
        .unwrap();

    let batch2 = ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0; 4]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0; 4]).unwrap(),
        global_lbs: vec![-0.5, 0.4],
        global_ubs: vec![0.6, 1.2],
        metadata: vec![metadata(-0.5, 0.6, 2), metadata(0.4, 1.2, 3)],
        keep_mask: vec![true, true],
    };
    list.add(batch2).unwrap();

    list.sort_by_domain_priority(true).unwrap();

    let picked = list
        .pick_out_batched(4, BatchedDomainOptions::default())
        .unwrap();

    assert_eq!(picked.batch_size, 4);
    assert_eq!(
        picked.global_ubs,
        vec![1.2, 1.0, 0.7, 0.6],
        "upper-bound mode must sort largest ub first in BFS, even with wrapped queue (#4406)",
    );
    assert_eq!(
        picked.global_lbs,
        vec![0.4, 0.2, -0.3, -0.5],
        "wrapped-queue upper-mode sort must keep storage aligned across tensors and metadata (#4406)",
    );

    let meta_ubs: Vec<f32> = picked.metadata.iter().map(|m| m.upper_bound()).collect();
    assert_eq!(meta_ubs, vec![1.2, 1.0, 0.7, 0.6]);
}
