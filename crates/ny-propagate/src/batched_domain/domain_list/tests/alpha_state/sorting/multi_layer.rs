// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn metadata_with_alpha(
    lower_bound: f32,
    upper_bound: f32,
    alpha_state: Option<crate::beta_crown::state::GraphDomainAlphaState>,
) -> DomainMetadata {
    DomainMetadata {
        lower_bound,
        upper_bound,
        depth: 1,
        constraints: vec![],
        cached_la: None,
        needs_bounding: false,
        node_bounds_override: None,
        alpha_state: alpha_state.map(Into::into),
    }
}

/// Test that alpha state with neurons spanning multiple ReLU layers survives
/// the add → sort → pick_out round-trip with all values correctly preserved.
///
/// Regression coverage: all prior alpha tests used a single ReLU layer ("relu0").
/// This test verifies that the storage mechanism correctly handles alpha from
/// multiple layers, ensuring HashMap keys (node_name, neuron_idx) don't collide
/// or lose entries across different layers.
#[ntest::timeout(10000)]
#[test]
fn test_multi_layer_alpha_state_roundtrip_through_sort() {
    use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};

    let mut layer_shapes = HashMap::new();
    layer_shapes.insert("relu0".to_string(), vec![1]);
    layer_shapes.insert("relu1".to_string(), vec![1]);
    let config = DomainListConfig {
        traversal: TreeTraversal::DepthFirst,
        layer_names: vec!["relu0".to_string(), "relu1".to_string()],
        layer_shapes,
        input_shape: vec![2],
        initial_capacity: 16,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();

    // Domain A: alpha values in both relu0 and relu1
    let mut alpha_state_a = GraphDomainAlphaState::empty();
    alpha_state_a.insert("relu0".to_string(), 0, AlphaNeuronState::new(0.73));
    alpha_state_a.insert("relu1".to_string(), 3, AlphaNeuronState::new(0.22));

    // Domain B: alpha values only in relu1 (different layer composition than A)
    let mut alpha_state_b = GraphDomainAlphaState::empty();
    alpha_state_b.insert("relu1".to_string(), 0, AlphaNeuronState::new(0.88));
    alpha_state_b.insert("relu1".to_string(), 1, AlphaNeuronState::new(0.15));

    // Domain C: no alpha state (baseline)
    let make_processed =
        |lb: f32, ub: f32, alpha: Option<GraphDomainAlphaState>| ProcessedDomains {
            layer_lowers: {
                let mut m = HashMap::new();
                m.insert(
                    "relu0".to_string(),
                    ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![lb]).unwrap(),
                );
                m.insert(
                    "relu1".to_string(),
                    ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![lb * 0.5]).unwrap(),
                );
                m
            },
            layer_uppers: {
                let mut m = HashMap::new();
                m.insert(
                    "relu0".to_string(),
                    ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![ub]).unwrap(),
                );
                m.insert(
                    "relu1".to_string(),
                    ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![ub * 0.5]).unwrap(),
                );
                m
            },
            input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.1]).unwrap(),
            input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.1]).unwrap(),
            global_lbs: vec![lb],
            global_ubs: vec![ub],
            metadata: vec![metadata_with_alpha(lb, ub, alpha)],
            keep_mask: vec![true],
        };

    // Add in reverse lb order so the CPU-priority sort must actually reorder.
    // Lower-bound mode stores C(-0.1), B(-0.5), A(-0.9) after sort, and DFS
    // pick_out pops from the end so A is processed first.
    list.add(make_processed(-0.1, 0.9, None)).unwrap(); // C
    list.add(make_processed(-0.5, 0.5, Some(alpha_state_b)))
        .unwrap(); // B
    list.add(make_processed(-0.9, 0.1, Some(alpha_state_a)))
        .unwrap(); // A

    assert_eq!(list.len(), 3);
    list.sort_by_domain_priority(false).unwrap();

    // Pick one at a time — lower-bound mode processes the smallest lower bound first.

    let first = list.pick_out(1).unwrap();
    let first_md = &first.metadata[0];
    assert!(
        (first_md.lower_bound - (-0.9)).abs() < 1e-6,
        "smallest lower bound should be picked first, got {}",
        first_md.lower_bound
    );
    let alpha_a_recovered = first_md
        .alpha_state()
        .expect("domain A should have alpha state");
    assert_eq!(
        alpha_a_recovered.len(),
        2,
        "domain A alpha should have 2 neurons (relu0 and relu1)"
    );
    let a_relu0 = alpha_a_recovered
        .neuron("relu0", 0)
        .expect("A should have (relu0, 0)");
    assert!(
        (a_relu0.alpha - 0.73).abs() < 1e-6,
        "A (relu0, 0) alpha should be 0.73, got {}",
        a_relu0.alpha
    );
    let a_relu1 = alpha_a_recovered
        .neuron("relu1", 3)
        .expect("A should have (relu1, 3)");
    assert!(
        (a_relu1.alpha - 0.22).abs() < 1e-6,
        "A (relu1, 3) alpha should be 0.22, got {}",
        a_relu1.alpha
    );

    let second = list.pick_out(1).unwrap();
    let second_md = &second.metadata[0];
    assert!(
        (second_md.lower_bound - (-0.5)).abs() < 1e-6,
        "middle lb should be picked second, got {}",
        second_md.lower_bound
    );
    let alpha_b_recovered = second_md
        .alpha_state()
        .expect("domain B should have alpha state");
    assert_eq!(
        alpha_b_recovered.len(),
        2,
        "domain B alpha should have 2 neurons (both in relu1)"
    );
    let b_n0 = alpha_b_recovered
        .neuron("relu1", 0)
        .expect("B should have (relu1, 0)");
    assert!(
        (b_n0.alpha - 0.88).abs() < 1e-6,
        "B (relu1, 0) alpha should be 0.88, got {}",
        b_n0.alpha
    );
    let b_n1 = alpha_b_recovered
        .neuron("relu1", 1)
        .expect("B should have (relu1, 1)");
    assert!(
        (b_n1.alpha - 0.15).abs() < 1e-6,
        "B (relu1, 1) alpha should be 0.15, got {}",
        b_n1.alpha
    );

    let third = list.pick_out(1).unwrap();
    let third_md = &third.metadata[0];
    assert!(
        (third_md.lower_bound - (-0.1)).abs() < 1e-6,
        "largest lower bound should be picked last, got {}",
        third_md.lower_bound
    );
    assert!(
        third_md.alpha_state().is_none(),
        "domain C should have no alpha state"
    );
}
