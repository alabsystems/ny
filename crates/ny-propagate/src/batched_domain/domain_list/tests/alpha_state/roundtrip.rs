// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn domain_metadata_with_alpha(
    lower_bound: f32,
    upper_bound: f32,
    depth: usize,
    constraints: Vec<(String, usize, bool, Option<f32>)>,
    alpha_state: Option<crate::beta_crown::state::GraphDomainAlphaState>,
) -> DomainMetadata {
    DomainMetadata {
        lower_bound,
        upper_bound,
        depth,
        constraints,
        cached_la: None,
        needs_bounding: false,
        node_bounds_override: None,
        alpha_state: alpha_state.map(Into::into),
    }
}

/// Test that alpha_state in DomainMetadata survives the add → pick_out round-trip.
///
/// This verifies that optimized alpha values persisted in DomainMetadata are
/// not lost when domains pass through the DomainList storage cycle.
/// Regression test for #1845 alpha persistence.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_roundtrip_through_domain_list_1845() {
    use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};

    let mut layer_shapes = HashMap::new();
    // Use a shape consistent with the alpha neuron indices used below (0 and 2).
    layer_shapes.insert("relu0".to_string(), vec![3]);
    let config = DomainListConfig {
        traversal: TreeTraversal::DepthFirst,
        layer_names: vec!["relu0".to_string()],
        layer_shapes,
        input_shape: vec![2],
        initial_capacity: 16,
        max_queue_size: 0,
    };
    let mut list = DomainList::new(config).unwrap();

    // Create alpha state with specific optimized values
    let mut alpha_state = GraphDomainAlphaState::empty();
    alpha_state.insert(
        "relu0".to_string(),
        0,
        AlphaNeuronState::new(0.73), // optimized alpha, not the heuristic 0/1
    );
    alpha_state.insert("relu0".to_string(), 2, AlphaNeuronState::new(0.42));

    // Domain with alpha state
    let processed_with_alpha = ProcessedDomains {
        layer_lowers: {
            let mut m = HashMap::new();
            m.insert(
                "relu0".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-0.5, -0.4, -0.3]).unwrap(),
            );
            m
        },
        layer_uppers: {
            let mut m = HashMap::new();
            m.insert(
                "relu0".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![0.5, 0.6, 0.7]).unwrap(),
            );
            m
        },
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.1]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.1]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![0.5],
        metadata: vec![domain_metadata_with_alpha(
            -0.5,
            0.5,
            2,
            vec![("relu0".to_string(), 1, true, None)],
            Some(alpha_state),
        )],
        keep_mask: vec![true],
    };

    // Domain without alpha state (as baseline comparison)
    let processed_no_alpha = ProcessedDomains {
        layer_lowers: {
            let mut m = HashMap::new();
            m.insert(
                "relu0".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-0.3, -0.2, -0.1]).unwrap(),
            );
            m
        },
        layer_uppers: {
            let mut m = HashMap::new();
            m.insert(
                "relu0".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![0.7, 0.8, 0.9]).unwrap(),
            );
            m
        },
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.2, 0.3]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.2, 1.3]).unwrap(),
        global_lbs: vec![-0.3],
        global_ubs: vec![0.7],
        metadata: vec![domain_metadata_with_alpha(-0.3, 0.7, 1, vec![], None)],
        keep_mask: vec![true],
    };

    list.add(processed_no_alpha).unwrap();
    list.add(processed_with_alpha).unwrap();
    assert_eq!(list.len(), 2);
    // Pick out both domains (DFS pops from end, so the alpha-state domain
    // was added last and should be picked first)
    let picked = list.pick_out(2).unwrap();
    assert_eq!(picked.batch_size, 2);
    let relu0_lower = picked
        .layer_lowers
        .get("relu0")
        .expect("picked domains should include relu0 lower bounds");
    let relu0_upper = picked
        .layer_uppers
        .get("relu0")
        .expect("picked domains should include relu0 upper bounds");
    assert_eq!(
        relu0_lower.shape(),
        &[2, 3],
        "relu0 lower bounds must stay batched"
    );
    assert_eq!(
        relu0_upper.shape(),
        &[2, 3],
        "relu0 upper bounds must stay batched"
    );

    // Find the domain that has alpha state
    let alpha_domain = picked
        .metadata
        .iter()
        .find(|m| m.alpha_state().is_some())
        .expect("should find domain with alpha_state after round-trip");
    assert!(
        (alpha_domain.lower_bound - (-0.5)).abs() < 1e-6,
        "alpha metadata must remain attached to the lb=-0.5 domain"
    );
    assert!(
        (alpha_domain.upper_bound - 0.5).abs() < 1e-6,
        "alpha metadata must remain attached to the ub=0.5 domain"
    );
    assert_eq!(
        alpha_domain.depth, 2,
        "alpha metadata must preserve original domain depth"
    );
    assert_eq!(
        alpha_domain.constraints,
        vec![("relu0".to_string(), 1, true, None)],
        "alpha metadata must preserve original constraints"
    );

    let recovered_alpha = alpha_domain.alpha_state().unwrap();
    let alpha_row = picked
        .metadata
        .iter()
        .position(|m| (m.lower_bound - (-0.5)).abs() < 1e-6)
        .expect("alpha domain row should be present in picked metadata");
    assert!(
        (relu0_lower[[alpha_row, 0]] - (-0.5)).abs() < 1e-6
            && (relu0_lower[[alpha_row, 1]] - (-0.4)).abs() < 1e-6
            && (relu0_lower[[alpha_row, 2]] - (-0.3)).abs() < 1e-6,
        "alpha domain row must keep relu0 lower tensor values"
    );
    assert!(
        (relu0_upper[[alpha_row, 0]] - 0.5).abs() < 1e-6
            && (relu0_upper[[alpha_row, 1]] - 0.6).abs() < 1e-6
            && (relu0_upper[[alpha_row, 2]] - 0.7).abs() < 1e-6,
        "alpha domain row must keep relu0 upper tensor values"
    );

    // Verify specific optimized alpha values survived the round-trip
    assert_eq!(
        recovered_alpha.len(),
        2,
        "alpha state should preserve both neuron entries"
    );
    let neuron_0 = recovered_alpha
        .neuron("relu0", 0)
        .expect("neuron (relu0, 0) should be in recovered alpha state");
    assert!(
        (neuron_0.alpha - 0.73).abs() < 1e-6,
        "alpha for (relu0, 0) should be 0.73, got {}",
        neuron_0.alpha
    );
    let neuron_2 = recovered_alpha
        .neuron("relu0", 2)
        .expect("neuron (relu0, 2) should be in recovered alpha state");
    assert!(
        (neuron_2.alpha - 0.42).abs() < 1e-6,
        "alpha for (relu0, 2) should be 0.42, got {}",
        neuron_2.alpha
    );

    // Verify the other domain has no alpha state
    let no_alpha_domain = picked
        .metadata
        .iter()
        .find(|m| m.alpha_state().is_none())
        .expect("should find domain without alpha_state");
    assert!(
        (no_alpha_domain.lower_bound - (-0.3)).abs() < 1e-6,
        "no-alpha metadata must remain attached to the lb=-0.3 domain"
    );
    assert!(
        (no_alpha_domain.upper_bound - 0.7).abs() < 1e-6,
        "no-alpha metadata must remain attached to the ub=0.7 domain"
    );
    assert_eq!(
        no_alpha_domain.depth, 1,
        "no-alpha metadata must preserve original domain depth"
    );
    assert!(
        no_alpha_domain.constraints.is_empty(),
        "no-alpha domain constraints must remain empty"
    );
    let no_alpha_row = picked
        .metadata
        .iter()
        .position(|m| (m.lower_bound - (-0.3)).abs() < 1e-6)
        .expect("no-alpha domain row should be present in picked metadata");
    assert!(
        (relu0_lower[[no_alpha_row, 0]] - (-0.3)).abs() < 1e-6
            && (relu0_lower[[no_alpha_row, 1]] - (-0.2)).abs() < 1e-6
            && (relu0_lower[[no_alpha_row, 2]] - (-0.1)).abs() < 1e-6,
        "no-alpha domain row must keep relu0 lower tensor values"
    );
    assert!(
        (relu0_upper[[no_alpha_row, 0]] - 0.7).abs() < 1e-6
            && (relu0_upper[[no_alpha_row, 1]] - 0.8).abs() < 1e-6
            && (relu0_upper[[no_alpha_row, 2]] - 0.9).abs() < 1e-6,
        "no-alpha domain row must keep relu0 upper tensor values"
    );
    assert!(
        no_alpha_domain.alpha_state().is_none(),
        "domain without alpha should remain None"
    );
}

/// Verify that `Some(empty_alpha_state)` round-trips correctly and is not
/// confused with `None`. This is a type-level distinction that matters for
/// warm-start initialization: `None` means "initialize from graph bounds",
/// while `Some(empty)` means "all neurons had their alpha pruned by constraints".
#[test]
fn test_empty_alpha_state_roundtrip() {
    use crate::beta_crown::state::GraphDomainAlphaState;

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

    // Domain with Some(empty alpha state) — all neurons constrained away.
    let empty_alpha = GraphDomainAlphaState::empty();
    let processed = ProcessedDomains {
        layer_lowers: {
            let mut m = HashMap::new();
            m.insert(
                "relu0".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![-0.5]).unwrap(),
            );
            m
        },
        layer_uppers: {
            let mut m = HashMap::new();
            m.insert(
                "relu0".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.5]).unwrap(),
            );
            m
        },
        input_lowers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap(),
        input_uppers: ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.0]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![0.5],
        metadata: vec![DomainMetadata {
            lower_bound: -0.5,
            upper_bound: 0.5,
            depth: 2,
            constraints: vec![("relu0".to_string(), 0, false, None)],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: Some(empty_alpha.into()),
        }],
        keep_mask: vec![true],
    };

    list.add(processed).unwrap();
    assert_eq!(list.len(), 1);

    let picked = list.pick_out(1).unwrap();
    let md = &picked.metadata[0];

    // The alpha_state should still be Some (not None), even though it's empty.
    assert!(
        md.alpha_state().is_some(),
        "empty alpha state should survive round-trip as Some, not be collapsed to None"
    );
    assert!(
        md.alpha_state().unwrap().is_empty(),
        "empty alpha state should have zero neurons after round-trip"
    );
}
