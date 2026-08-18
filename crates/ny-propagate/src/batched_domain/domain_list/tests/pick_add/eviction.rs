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
    assert_eq!(lbs.len(), ubs.len());
    let metadata_vec: Vec<DomainMetadata> = lbs
        .iter()
        .zip(&ubs)
        .map(|(&lb, &ub)| metadata(lb, ub, 1))
        .collect();
    processed_metadata(metadata_vec)
}

fn processed_metadata(metadata_vec: Vec<DomainMetadata>) -> ProcessedDomains {
    let n = metadata_vec.len();
    let lbs = metadata_vec
        .iter()
        .map(DomainMetadata::lower_bound)
        .collect();
    let ubs = metadata_vec
        .iter()
        .map(DomainMetadata::upper_bound)
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
    list.configure_queue_eviction(0, false).unwrap();

    list.add(processed_batch(
        vec![-0.8, 0.5, -0.1, -0.3],
        vec![0.2, 1.5, 0.9, 0.7],
    ))
    .unwrap();

    assert_eq!(list.len(), 4);
    assert_eq!(list.evicted_count(), 0);
}

/// Byte enforcement is installed before add and shares the existing queue
/// compaction/latch path. Zero count cap still means unlimited by count.
#[ntest::timeout(10000)]
#[test]
fn test_byte_cap_evicts_on_add_and_records_count() {
    let mut list = DomainList::new(layerless_config(0)).unwrap();
    let two_rows = list.estimated_bytes_per_domain().saturating_mul(2);
    list.configure_queue_eviction(two_rows, false).unwrap();

    list.add(processed_batch(
        vec![-0.8, 0.5, -0.1, -0.3],
        vec![0.2, 1.5, 0.9, 0.7],
    ))
    .unwrap();

    assert_eq!(list.len(), 2);
    assert_eq!(list.evicted_count(), 2);
    assert!(list.estimated_resident_bytes() <= two_rows);
}

#[ntest::timeout(10000)]
#[test]
fn test_count_and_byte_caps_use_the_tighter_limit() {
    let mut count_tighter = DomainList::new(layerless_config(2)).unwrap();
    let row_bytes = count_tighter.estimated_bytes_per_domain();
    count_tighter
        .configure_queue_eviction(row_bytes.saturating_mul(4), false)
        .unwrap();
    count_tighter
        .add(processed_batch(
            vec![-0.8, 0.5, -0.1, -0.3, 0.2],
            vec![0.2, 1.5, 0.9, 0.7, 1.2],
        ))
        .unwrap();
    assert_eq!(count_tighter.len(), 2);

    let mut bytes_tighter = DomainList::new(layerless_config(5)).unwrap();
    let row_bytes = bytes_tighter.estimated_bytes_per_domain();
    bytes_tighter
        .configure_queue_eviction(row_bytes.saturating_mul(2), false)
        .unwrap();
    bytes_tighter
        .add(processed_batch(
            vec![-0.8, 0.5, -0.1, -0.3, 0.2],
            vec![0.2, 1.5, 0.9, 0.7, 1.2],
        ))
        .unwrap();
    assert_eq!(bytes_tighter.len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_byte_cap_below_one_row_keeps_one_domain_for_progress() {
    let mut list = DomainList::new(layerless_config(0)).unwrap();
    list.configure_queue_eviction(1, false).unwrap();
    list.add(processed_batch(vec![-0.8, 0.5, -0.1], vec![0.2, 1.5, 0.9]))
        .unwrap();

    assert_eq!(list.len(), 1);
    assert_eq!(list.evicted_count(), 2);
    assert!(list.estimated_resident_bytes() > 1);
}

/// The active verification sense controls eviction priority. Upper-bound
/// verification must retain the largest upper bounds, matching the CPU heap.
#[ntest::timeout(10000)]
#[test]
fn test_upper_mode_eviction_keeps_highest_upper_bounds() {
    let mut list = DomainList::new(layerless_config(2)).unwrap();
    list.configure_queue_eviction(0, true).unwrap();

    list.add(processed_batch(
        vec![-0.1, -0.8, -0.3, 0.2],
        vec![0.5, 3.0, 1.0, 2.0],
    ))
    .unwrap();

    let picked = list
        .pick_out_batched(2, BatchedDomainOptions::default())
        .unwrap();
    assert_eq!(picked.global_ubs, vec![3.0, 2.0]);
    assert_eq!(list.evicted_count(), 2);
}

/// Metadata grows with depth, cached state, and split histories. The byte cap
/// must be recomputed by each add rather than frozen from the root row.
#[ntest::timeout(10000)]
#[test]
fn test_byte_cap_recomputes_after_metadata_growth() {
    let mut list = DomainList::new(layerless_config(0)).unwrap();
    list.add(processed_batch(
        vec![-0.8, 0.5, -0.1, -0.3],
        vec![0.2, 1.5, 0.9, 0.7],
    ))
    .unwrap();
    let root_sized_budget = list.estimated_resident_bytes();
    list.configure_queue_eviction(root_sized_budget, false)
        .unwrap();
    assert_eq!(list.len(), 4);

    let large_history = |lower_bound: f32| DomainMetadata {
        lower_bound,
        upper_bound: 1.0,
        depth: 128,
        constraints: (0..128)
            .map(|index| {
                (
                    format!("very_long_relu_node_name_{index:04}_{}", "x".repeat(128)),
                    index,
                    index % 2 == 0,
                    None,
                )
            })
            .collect(),
        cached_la: None,
        needs_bounding: false,
        alpha_state: None,
        node_bounds_override: None,
    };
    list.add(processed_metadata(vec![
        large_history(-10.0),
        large_history(-9.0),
    ]))
    .unwrap();

    assert!(
        list.len() < 4,
        "heavier child metadata must tighten the resident frontier"
    );
    assert!(
        list.len() == 1 || list.estimated_resident_bytes() <= root_sized_budget,
        "the only permitted over-budget case is the one-domain progress floor"
    );
    assert!(list.evicted_count() > 0);
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
