// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Regression test for #1845 handoff gap: sorting must preserve alpha metadata
/// alignment with each domain's bound record.
#[ntest::timeout(10000)]
#[test]
fn test_sort_preserves_alpha_state_alignment_1845() {
    use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};

    let mut layer_shapes = HashMap::new();
    layer_shapes.insert("relu0".to_string(), vec![1]);
    let config = DomainListConfig {
        traversal: TreeTraversal::DepthFirst,
        layer_names: vec!["relu0".to_string()],
        layer_shapes,
        input_shape: vec![2],
        initial_capacity: 16,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();
    let make_processed =
        |lower_bound: f32, upper_bound: f32, alpha_value: Option<f32>| -> ProcessedDomains {
            let alpha_state = alpha_value.map(|alpha| {
                let mut state = GraphDomainAlphaState::empty();
                state.insert("relu0".to_string(), 0, AlphaNeuronState::new(alpha));
                state
            });
            let mut metadata = DomainMetadata::root(lower_bound, upper_bound).unwrap();
            metadata.set_alpha_state(alpha_state);

            ProcessedDomains {
                layer_lowers: {
                    let mut m = HashMap::new();
                    m.insert(
                        "relu0".to_string(),
                        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![lower_bound]).unwrap(),
                    );
                    m
                },
                layer_uppers: {
                    let mut m = HashMap::new();
                    m.insert(
                        "relu0".to_string(),
                        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![upper_bound]).unwrap(),
                    );
                    m
                },
                input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap(),
                input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.0]).unwrap(),
                global_lbs: vec![lower_bound],
                global_ubs: vec![upper_bound],
                metadata: vec![metadata],
                keep_mask: vec![true],
            }
        }; // Insert in reverse order so CPU-priority sort must reorder before DFS pick_out.
    list.add(make_processed(-0.1, 0.7, Some(0.77))).unwrap();
    list.add(make_processed(-0.5, 0.8, None)).unwrap();
    list.add(make_processed(-0.9, 0.9, Some(0.11))).unwrap();
    assert_eq!(list.len(), 3);
    // Lower-bound mode mirrors CPU `domain_priority()`: DFS pick_out pops the most negative lower bound first.
    list.sort_by_domain_priority(false).unwrap();
    let first = list.pick_out(1).unwrap();
    let first_md = &first.metadata[0];
    assert!(
        (first_md.lower_bound - (-0.9)).abs() < 1e-6,
        "smallest lower bound should be picked first after sort"
    );
    let first_alpha = first_md
        .alpha_state()
        .and_then(|a| a.neuron("relu0", 0))
        .expect("first domain should retain alpha state");
    assert!(
        (first_alpha.alpha - 0.11).abs() < 1e-6,
        "first domain alpha should match pre-sort metadata"
    );
    let second = list.pick_out(1).unwrap();
    let second_md = &second.metadata[0];
    assert!(
        (second_md.lower_bound - (-0.5)).abs() < 1e-6,
        "middle bound should be picked second"
    );
    assert!(
        second_md.alpha_state().is_none(),
        "middle domain should preserve None alpha metadata"
    );

    let third = list.pick_out(1).unwrap();
    let third_md = &third.metadata[0];
    assert!(
        (third_md.lower_bound - (-0.1)).abs() < 1e-6,
        "largest lower bound should be picked last"
    );
    let third_alpha = third_md
        .alpha_state()
        .and_then(|a| a.neuron("relu0", 0))
        .expect("last domain should retain alpha state");
    assert!(
        (third_alpha.alpha - 0.77).abs() < 1e-6,
        "last domain alpha should match pre-sort metadata"
    );
}
