// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for queue-cap eviction (#2326): eviction must keep the most
//! promising domains, and every eviction must be recorded in
//! `evicted_count()` so the BaB loop can refuse to report Verified after
//! discarding unexplored search space.

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

fn layerless_config(max_queue_size: usize) -> DomainListConfig {
    DomainListConfig {
        traversal: TreeTraversal::BreadthFirst,
        layer_names: Vec::new(),
        layer_shapes: HashMap::new(),
        input_shape: vec![2],
        initial_capacity: 4,
        max_queue_size,
    }
}

fn processed_batch(lbs: Vec<f32>, ubs: Vec<f32>) -> ProcessedDomains {
    let n = lbs.len();
    assert_eq!(n, ubs.len());
    let metadata_vec: Vec<DomainMetadata> = lbs
        .iter()
        .zip(&ubs)
        .map(|(&lb, &ub)| metadata(lb, ub, 1))
        .collect();
    ProcessedDomains {
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[n, 2]), vec![0.0; n * 2]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[n, 2]), vec![1.0; n * 2]).unwrap(),
        global_lbs: lbs,
        global_ubs: ubs,
        metadata: metadata_vec,
        keep_mask: vec![true; n],
    }
}

/// Eviction keeps the lowest lower_bound domains (most promising in
/// verify-lower mode), removes the highest, and records the removals.
#[ntest::timeout(10000)]
#[test]
fn test_add_over_queue_cap_evicts_highest_lower_bound_and_records_count() {
    let mut list = DomainList::new(layerless_config(2)).unwrap();
    assert_eq!(list.evicted_count(), 0);

    list.add(processed_batch(
        vec![-0.8, 0.5, -0.1, -0.3],
        vec![0.2, 1.5, 0.9, 0.7],
    ))
    .unwrap();

    assert_eq!(list.len(), 2, "queue must be truncated to max_queue_size");
    assert_eq!(
        list.evicted_count(),
        2,
        "every evicted domain must be recorded so the BaB result cannot claim Verified"
    );

    let picked = list
        .pick_out_batched(2, BatchedDomainOptions::default())
        .unwrap();
    assert_eq!(
        picked.global_lbs,
        vec![-0.8, -0.3],
        "eviction must keep the lowest lower_bound domains in original order"
    );
}

/// `max_queue_size == 0` disables the cap entirely: nothing is evicted.
#[ntest::timeout(10000)]
#[test]
fn test_queue_cap_zero_disables_eviction() {
    let mut list = DomainList::new(layerless_config(0)).unwrap();

    list.add(processed_batch(
        vec![-0.8, 0.5, -0.1, -0.3],
        vec![0.2, 1.5, 0.9, 0.7],
    ))
    .unwrap();

    assert_eq!(list.len(), 4);
    assert_eq!(list.evicted_count(), 0);
}

/// The eviction count accumulates across `add` calls: any nonzero total
/// means the search space was truncated at some point in the run.
#[ntest::timeout(10000)]
#[test]
fn test_evicted_count_accumulates_across_adds() {
    let mut list = DomainList::new(layerless_config(1)).unwrap();

    list.add(processed_batch(vec![-0.8, -0.1], vec![0.2, 0.9]))
        .unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list.evicted_count(), 1);

    list.add(processed_batch(vec![-0.5, -0.2], vec![0.4, 0.6]))
        .unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list.evicted_count(), 3);
}
