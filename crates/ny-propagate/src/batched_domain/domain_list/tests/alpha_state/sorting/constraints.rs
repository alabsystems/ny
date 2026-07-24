// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Verify that sort correctly reorders alpha metadata when domains also
/// have constraints. This is the "sort + constraints combined" gap:
/// the existing sort test uses `constraints: vec![]` and the constraint
/// tests don't sort. Real BaB domains have both.
#[test]
fn test_sort_preserves_alpha_with_constraints_combined() {
    use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};

    let mut layer_shapes = HashMap::new();
    layer_shapes.insert("relu0".to_string(), vec![2]);
    let config = DomainListConfig {
        traversal: TreeTraversal::DepthFirst,
        layer_names: vec!["relu0".to_string()],
        layer_shapes,
        input_shape: vec![2],
        initial_capacity: 16,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();
    let make_processed = |lower_bound: f32,
                          upper_bound: f32,
                          alpha: Option<f32>,
                          constraints: Vec<(String, usize, bool, Option<f32>)>|
     -> ProcessedDomains {
        let alpha_state = alpha.map(|a| {
            let mut state = GraphDomainAlphaState::empty();
            state.insert("relu0".to_string(), 0, AlphaNeuronState::new(a));
            state
        });

        let mut metadata = DomainMetadata::root(lower_bound, upper_bound).unwrap();
        metadata.depth = 1;
        metadata.constraints = constraints;
        metadata.set_alpha_state(alpha_state);

        ProcessedDomains {
            layer_lowers: {
                let mut m = HashMap::new();
                m.insert(
                    "relu0".to_string(),
                    ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![lower_bound, lower_bound]).unwrap(),
                );
                m
            },
            layer_uppers: {
                let mut m = HashMap::new();
                m.insert(
                    "relu0".to_string(),
                    ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![upper_bound, upper_bound]).unwrap(),
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
    };
    // Domain A: lb=-0.3, alpha=0.55, constraint on neuron 1 (inactive)
    list.add(make_processed(
        -0.3,
        0.6,
        Some(0.55),
        vec![("relu0".to_string(), 1, false, None)],
    ))
    .unwrap();
    // Domain B: lb=-0.8, alpha=0.22, constraint on neuron 0 (active)
    list.add(make_processed(
        -0.8,
        0.9,
        Some(0.22),
        vec![("relu0".to_string(), 0, true, None)],
    ))
    .unwrap();

    assert_eq!(list.len(), 2);
    list.sort_by_domain_priority(false).unwrap();

    // Lower-bound mode mirrors CPU `domain_priority()`: DFS pick_out pops the smallest lower bound first.
    let first = list.pick_out(1).unwrap();
    let first_md = &first.metadata[0];
    assert!(
        (first_md.lower_bound - (-0.8)).abs() < 1e-6,
        "smallest lower bound should be picked first: got {}",
        first_md.lower_bound
    );
    // Alpha state should be 0.22 (domain B).
    let first_alpha = first_md
        .alpha_state()
        .and_then(|a| a.neuron("relu0", 0))
        .expect("domain B should retain alpha");
    assert!(
        (first_alpha.alpha - 0.22).abs() < 1e-6,
        "domain B alpha should be 0.22, got {}",
        first_alpha.alpha
    );
    // Constraint should be on neuron 0 (domain B's constraint).
    assert_eq!(
        first_md.constraints.len(),
        1,
        "domain B should have 1 constraint"
    );
    assert_eq!(
        first_md.constraints[0].1, 0,
        "domain B constraint on neuron 0"
    );
    assert!(
        first_md.constraints[0].2,
        "domain B constraint should be active"
    );
    let second = list.pick_out(1).unwrap();
    let second_md = &second.metadata[0];
    assert!(
        (second_md.lower_bound - (-0.3)).abs() < 1e-6,
        "largest lower bound should be picked second: got {}",
        second_md.lower_bound
    );
    let second_alpha = second_md
        .alpha_state()
        .and_then(|a| a.neuron("relu0", 0))
        .expect("domain A should retain alpha");
    assert!(
        (second_alpha.alpha - 0.55).abs() < 1e-6,
        "domain A alpha should be 0.55, got {}",
        second_alpha.alpha
    );
    assert_eq!(
        second_md.constraints[0].1, 1,
        "domain A constraint on neuron 1"
    );
    assert!(
        !second_md.constraints[0].2,
        "domain A constraint should be inactive"
    );
}
