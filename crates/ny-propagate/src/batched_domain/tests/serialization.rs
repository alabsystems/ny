// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use crate::beta_crown::engine::graph::domain_conversion::history_from_constraints;
use crate::beta_crown::{
    GenBabConstraint, GraphBabDomain, GraphBetaState, GraphNeuronConstraint, GraphSplitHistory,
};
use ndarray::array;
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::sync::Arc;

/// Test constraint serialization roundtrip for mixed ReLU/GenBaB constraints.
///
/// Verifies that serialize_constraints (via BatchedDomains::from_graph_domains)
/// and history_from_constraints correctly roundtrip constraint tuples while
/// preserving order and split point values.
///
#[ntest::timeout(5000)]
#[test]
fn test_constraint_serialization_roundtrip_relu_only() {
    // Create history with only ReLU constraints
    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu0".to_string(),
        neuron_idx: 5,
        is_active: true,
        score: 0.5,
    });
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 3,
        is_active: false,
        score: 0.3,
    });
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu0".to_string(),
        neuron_idx: 10,
        is_active: true,
        score: 0.7,
    });

    let input_bounds =
        Arc::new(BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap());
    let mut node_bounds = HashMap::new();
    node_bounds.insert(
        "relu0".to_string(),
        Arc::new(BoundedTensor::new(array![-1.0].into_dyn(), array![1.0].into_dyn()).unwrap()),
    );

    let domain = GraphBabDomain {
        history,
        node_bounds,
        lower_bound: 0.0,
        upper_bound: 1.0,
        depth: 3,
        priority: 0.5,
        input_bounds,
        beta_state: GraphBetaState::default(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    // Serialize via BatchedDomains
    let domains = vec![&domain];
    let layer_names = vec!["relu0".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    // Reconstruct history from serialized constraints
    let constraints = &batched.constraints()[0];
    let reconstructed = history_from_constraints(constraints).unwrap();

    // Verify: same number of ReLU constraints
    assert_eq!(reconstructed.constraints.len(), 3);
    assert_eq!(reconstructed.genbab_constraints.len(), 0);
    assert_eq!(reconstructed.split_count, 3);

    // Verify: constraint order and values preserved
    assert_eq!(reconstructed.constraints[0].node_name, "relu0");
    assert_eq!(reconstructed.constraints[0].neuron_idx, 5);
    assert!(
        reconstructed.constraints[0].is_active,
        "constraint[0] relu0[5] should be active"
    );

    assert_eq!(reconstructed.constraints[1].node_name, "relu1");
    assert_eq!(reconstructed.constraints[1].neuron_idx, 3);
    assert!(
        !reconstructed.constraints[1].is_active,
        "constraint[1] relu1[3] should be inactive"
    );

    assert_eq!(reconstructed.constraints[2].node_name, "relu0");
    assert_eq!(reconstructed.constraints[2].neuron_idx, 10);
    assert!(
        reconstructed.constraints[2].is_active,
        "constraint[2] relu0[10] should be active"
    );
}

/// Test constraint serialization roundtrip for GenBaB-only constraints.
#[ntest::timeout(5000)]
#[test]
fn test_constraint_serialization_roundtrip_genbab_only() {
    // Create history with only GenBaB constraints
    let mut history = GraphSplitHistory::new();
    history.add_genbab_constraint(
        GenBabConstraint::new(
            "gelu0".to_string(),
            2,
            -0.5,
            true, // upper branch: x >= -0.5
            0.4,
        )
        .unwrap(),
    );
    history.add_genbab_constraint(
        GenBabConstraint::new(
            "sigmoid0".to_string(),
            0,
            0.0,
            false, // lower branch: x <= 0.0
            0.6,
        )
        .unwrap(),
    );
    history.add_genbab_constraint(
        GenBabConstraint::new("gelu0".to_string(), 5, 0.25, true, 0.8).unwrap(),
    );

    let input_bounds =
        Arc::new(BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap());
    let mut node_bounds = HashMap::new();
    node_bounds.insert(
        "gelu0".to_string(),
        Arc::new(BoundedTensor::new(array![-1.0].into_dyn(), array![1.0].into_dyn()).unwrap()),
    );

    let domain = GraphBabDomain {
        history,
        node_bounds,
        lower_bound: 0.0,
        upper_bound: 1.0,
        depth: 3,
        priority: 0.5,
        input_bounds,
        beta_state: GraphBetaState::default(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    // Serialize via BatchedDomains
    let domains = vec![&domain];
    let layer_names = vec!["gelu0".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    // Reconstruct history from serialized constraints
    let constraints = &batched.constraints()[0];
    let reconstructed = history_from_constraints(constraints).unwrap();

    // Verify: all GenBaB constraints (no ReLU)
    assert_eq!(reconstructed.constraints.len(), 0);
    assert_eq!(reconstructed.genbab_constraints.len(), 3);
    assert_eq!(reconstructed.split_count, 3);

    // Verify: constraint order, split points, and branch directions preserved
    let c0 = &reconstructed.genbab_constraints[0];
    assert_eq!(c0.node_name, "gelu0");
    assert_eq!(c0.neuron_idx, 2);
    assert!(
        (c0.split_point - (-0.5)).abs() < 1e-6,
        "c0 split_point: expected -0.5, got {}",
        c0.split_point
    );
    assert!(c0.is_upper_branch, "c0 gelu0[2] should be upper branch");

    let c1 = &reconstructed.genbab_constraints[1];
    assert_eq!(c1.node_name, "sigmoid0");
    assert_eq!(c1.neuron_idx, 0);
    assert!(
        (c1.split_point - 0.0).abs() < 1e-6,
        "c1 split_point: expected 0.0, got {}",
        c1.split_point
    );
    assert!(
        !c1.is_upper_branch,
        "c1 sigmoid0[0] should not be upper branch"
    );

    let c2 = &reconstructed.genbab_constraints[2];
    assert_eq!(c2.node_name, "gelu0");
    assert_eq!(c2.neuron_idx, 5);
    assert!(
        (c2.split_point - 0.25).abs() < 1e-6,
        "c2 split_point: expected 0.25, got {}",
        c2.split_point
    );
    assert!(c2.is_upper_branch, "c2 gelu0[5] should be upper branch");
}

/// Test constraint serialization roundtrip for mixed ReLU and GenBaB constraints.
///
/// This tests the interleaving logic in serialize_constraints: constraints must
/// be serialized in split order (using genbab_split_ids) and reconstructed correctly.
#[ntest::timeout(5000)]
#[test]
fn test_constraint_serialization_roundtrip_mixed() {
    // Create history with interleaved ReLU and GenBaB constraints
    // Split order: ReLU -> GenBaB -> ReLU -> GenBaB -> GenBaB
    let mut history = GraphSplitHistory::new();

    // Split 0: ReLU
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu0".to_string(),
        neuron_idx: 1,
        is_active: true,
        score: 0.1,
    });

    // Split 1: GenBaB
    history.add_genbab_constraint(
        GenBabConstraint::new("gelu0".to_string(), 2, -0.3, true, 0.2).unwrap(),
    );

    // Split 2: ReLU
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: false,
        score: 0.3,
    });

    // Split 3: GenBaB
    history.add_genbab_constraint(
        GenBabConstraint::new("sigmoid0".to_string(), 3, 0.5, false, 0.4).unwrap(),
    );

    // Split 4: GenBaB
    history.add_genbab_constraint(
        GenBabConstraint::new("gelu0".to_string(), 4, -0.1, true, 0.5).unwrap(),
    );

    assert_eq!(history.split_count, 5);
    assert_eq!(history.constraints.len(), 2);
    assert_eq!(history.genbab_constraints.len(), 3);

    let input_bounds =
        Arc::new(BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap());
    let mut node_bounds = HashMap::new();
    node_bounds.insert(
        "relu0".to_string(),
        Arc::new(BoundedTensor::new(array![-1.0].into_dyn(), array![1.0].into_dyn()).unwrap()),
    );

    let domain = GraphBabDomain {
        history,
        node_bounds,
        lower_bound: 0.0,
        upper_bound: 1.0,
        depth: 5,
        priority: 0.5,
        input_bounds,
        beta_state: GraphBetaState::default(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    // Serialize via BatchedDomains
    let domains = vec![&domain];
    let layer_names = vec!["relu0".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    // Verify serialized constraints are in correct order
    let constraints = &batched.constraints()[0];
    assert_eq!(constraints.len(), 5);

    // Split 0: ReLU relu0[1] active
    assert_eq!(constraints[0], ("relu0".to_string(), 1, true, None));
    // Split 1: GenBaB gelu0[2] at -0.3, upper branch
    assert_eq!(constraints[1].0, "gelu0");
    assert_eq!(constraints[1].1, 2);
    assert!(constraints[1].2, "split 1 gelu0[2] should be upper branch");
    assert!(
        (constraints[1].3.unwrap() - (-0.3)).abs() < 1e-6,
        "split 1 split_point: expected -0.3, got {}",
        constraints[1].3.unwrap()
    );
    // Split 2: ReLU relu1[0] inactive
    assert_eq!(constraints[2], ("relu1".to_string(), 0, false, None));
    // Split 3: GenBaB sigmoid0[3] at 0.5, lower branch
    assert_eq!(constraints[3].0, "sigmoid0");
    assert_eq!(constraints[3].1, 3);
    assert!(
        !constraints[3].2,
        "split 3 sigmoid0[3] should not be upper branch"
    );
    assert!(
        (constraints[3].3.unwrap() - 0.5).abs() < 1e-6,
        "split 3 split_point: expected 0.5, got {}",
        constraints[3].3.unwrap()
    );
    // Split 4: GenBaB gelu0[4] at -0.1, upper branch
    assert_eq!(constraints[4].0, "gelu0");
    assert_eq!(constraints[4].1, 4);
    assert!(constraints[4].2, "split 4 gelu0[4] should be upper branch");
    assert!(
        (constraints[4].3.unwrap() - (-0.1)).abs() < 1e-6,
        "split 4 split_point: expected -0.1, got {}",
        constraints[4].3.unwrap()
    );

    // Reconstruct history from serialized constraints
    let reconstructed = history_from_constraints(constraints).unwrap();

    // Verify counts
    assert_eq!(reconstructed.constraints.len(), 2);
    assert_eq!(reconstructed.genbab_constraints.len(), 3);
    assert_eq!(reconstructed.split_count, 5);

    // Verify ReLU constraints preserved
    assert_eq!(reconstructed.constraints[0].node_name, "relu0");
    assert_eq!(reconstructed.constraints[0].neuron_idx, 1);
    assert!(
        reconstructed.constraints[0].is_active,
        "reconstructed relu constraint[0] relu0[1] should be active"
    );

    assert_eq!(reconstructed.constraints[1].node_name, "relu1");
    assert_eq!(reconstructed.constraints[1].neuron_idx, 0);
    assert!(
        !reconstructed.constraints[1].is_active,
        "reconstructed relu constraint[1] relu1[0] should be inactive"
    );

    // Verify GenBaB constraints preserved with correct split points
    let g0 = &reconstructed.genbab_constraints[0];
    assert_eq!(g0.node_name, "gelu0");
    assert_eq!(g0.neuron_idx, 2);
    assert!(
        (g0.split_point - (-0.3)).abs() < 1e-6,
        "g0 split_point: expected -0.3, got {}",
        g0.split_point
    );
    assert!(g0.is_upper_branch, "g0 gelu0[2] should be upper branch");

    let g1 = &reconstructed.genbab_constraints[1];
    assert_eq!(g1.node_name, "sigmoid0");
    assert_eq!(g1.neuron_idx, 3);
    assert!(
        (g1.split_point - 0.5).abs() < 1e-6,
        "g1 split_point: expected 0.5, got {}",
        g1.split_point
    );
    assert!(
        !g1.is_upper_branch,
        "g1 sigmoid0[3] should not be upper branch"
    );

    let g2 = &reconstructed.genbab_constraints[2];
    assert_eq!(g2.node_name, "gelu0");
    assert_eq!(g2.neuron_idx, 4);
    assert!(
        (g2.split_point - (-0.1)).abs() < 1e-6,
        "g2 split_point: expected -0.1, got {}",
        g2.split_point
    );
    assert!(g2.is_upper_branch, "g2 gelu0[4] should be upper branch");
}

/// Test constraint serialization roundtrip for empty history.
/// Edge case: domain with no constraints should roundtrip correctly.
#[ntest::timeout(5000)]
#[test]
fn test_constraint_serialization_roundtrip_empty() {
    // Create empty history
    let history = GraphSplitHistory::new();
    assert_eq!(history.split_count, 0);
    assert_eq!(history.constraints.len(), 0);
    assert_eq!(history.genbab_constraints.len(), 0);

    let input_bounds =
        Arc::new(BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap());

    let domain = GraphBabDomain {
        history,
        node_bounds: HashMap::new(),
        lower_bound: 0.0,
        upper_bound: 1.0,
        depth: 0,
        priority: 0.5,
        input_bounds,
        beta_state: GraphBetaState::default(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    // Serialize via BatchedDomains
    let domains = vec![&domain];
    let layer_names: Vec<String> = vec![];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    // Reconstruct history from serialized constraints
    let constraints = &batched.constraints()[0];
    assert!(
        constraints.is_empty(),
        "empty history should serialize to empty constraints, got {} entries",
        constraints.len()
    );

    let reconstructed = history_from_constraints(constraints).unwrap();

    // Verify: empty history
    assert_eq!(reconstructed.constraints.len(), 0);
    assert_eq!(reconstructed.genbab_constraints.len(), 0);
    assert_eq!(reconstructed.split_count, 0);
}

/// Test constraint serialization roundtrip preserves genbab_split_ids ordering.
/// This verifies the interleaving logic uses split IDs correctly.
#[ntest::timeout(5000)]
#[test]
fn test_constraint_serialization_roundtrip_split_ids() {
    // Create history with known split IDs
    // ReLU at split 0, GenBaB at split 1, ReLU at split 2
    let mut history = GraphSplitHistory::new();

    // Split 0
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu0".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.1,
    });

    // Split 1 - GenBaB
    history.add_genbab_constraint(
        GenBabConstraint::new("gelu0".to_string(), 1, -0.5, true, 0.2).unwrap(),
    );

    // Split 2
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu0".to_string(),
        neuron_idx: 2,
        is_active: false,
        score: 0.3,
    });

    // Verify original split_ids
    assert_eq!(history.split_count, 3);
    assert_eq!(history.genbab_split_ids.len(), 1);
    assert_eq!(history.genbab_split_ids[0], 1); // GenBaB was split #1

    let input_bounds =
        Arc::new(BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap());

    let domain = GraphBabDomain {
        history,
        node_bounds: HashMap::new(),
        lower_bound: 0.0,
        upper_bound: 1.0,
        depth: 3,
        priority: 0.5,
        input_bounds,
        beta_state: GraphBetaState::default(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    // Serialize via BatchedDomains
    let domains = vec![&domain];
    let layer_names: Vec<String> = vec![];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    // Verify serialized order: ReLU, GenBaB, ReLU
    let constraints = &batched.constraints()[0];
    assert_eq!(constraints.len(), 3);
    assert_eq!(constraints[0], ("relu0".to_string(), 0, true, None));
    assert_eq!(constraints[1].0, "gelu0");
    assert!(
        constraints[1].3.is_some(),
        "GenBaB constraint at split 1 should have a split_point"
    );
    assert_eq!(constraints[2], ("relu0".to_string(), 2, false, None));

    // Reconstruct and verify split_ids are reconstructed correctly
    let reconstructed = history_from_constraints(constraints).unwrap();
    assert_eq!(reconstructed.split_count, 3);
    assert_eq!(reconstructed.constraints.len(), 2);
    assert_eq!(reconstructed.genbab_constraints.len(), 1);
    assert_eq!(reconstructed.genbab_split_ids.len(), 1);
    assert_eq!(reconstructed.genbab_split_ids[0], 1); // Should be reconstructed as split #1
}

/// Test constraint serialization behavior for range splits (multiple GenBaB per split).
///
/// **Known limitation:** ConstraintTuple format doesn't preserve split_id grouping.
/// Range splits (multiple constraints per split) get "flattened" - each constraint
/// becomes its own split on reconstruction. This is acceptable because:
/// 1. Constraint VALUES (node, neuron, split_point, direction) are preserved
/// 2. The split_count expansion doesn't affect correctness, only depth tracking
/// 3. If exact split_id preservation is needed, ConstraintTuple would need extension
#[ntest::timeout(5000)]
#[test]
fn test_constraint_serialization_range_splits_flatten() {
    // Create history with a range split (lower and upper bound on same neuron)
    let mut history = GraphSplitHistory::new();

    // Split 0: Range split - both constraints share the same split ID
    history.add_genbab_constraints_for_split([
        GenBabConstraint::new("gelu0".to_string(), 0, -0.5, true, 0.3).unwrap(), // lower bound: x >= -0.5
        GenBabConstraint::new("gelu0".to_string(), 0, 0.5, false, 0.4).unwrap(), // upper bound: x <= 0.5
    ]);

    // Split 1: Single GenBaB
    history.add_genbab_constraint(
        GenBabConstraint::new("sigmoid0".to_string(), 1, 0.0, true, 0.5).unwrap(),
    );

    // Verify original: 2 splits but 3 constraints
    assert_eq!(history.split_count, 2);
    assert_eq!(history.genbab_constraints.len(), 3);
    assert_eq!(history.genbab_split_ids.len(), 3);
    // First two constraints share split ID 0 (range split)
    assert_eq!(history.genbab_split_ids[0], 0);
    assert_eq!(history.genbab_split_ids[1], 0);
    // Third constraint has split ID 1
    assert_eq!(history.genbab_split_ids[2], 1);

    let input_bounds =
        Arc::new(BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap());

    let domain = GraphBabDomain {
        history,
        node_bounds: HashMap::new(),
        lower_bound: 0.0,
        upper_bound: 1.0,
        depth: 2,
        priority: 0.5,
        input_bounds,
        beta_state: GraphBetaState::default(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    // Serialize via BatchedDomains
    let domains = vec![&domain];
    let layer_names: Vec<String> = vec![];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    // All 3 GenBaB constraints should be serialized (values preserved)
    let constraints = &batched.constraints()[0];
    assert_eq!(constraints.len(), 3);

    // Verify constraint VALUES are preserved
    assert_eq!(constraints[0].0, "gelu0");
    assert_eq!(constraints[0].1, 0);
    assert!(
        (constraints[0].3.unwrap() - (-0.5)).abs() < 1e-6,
        "range[0] split_point: expected -0.5, got {}",
        constraints[0].3.unwrap()
    );
    assert!(
        constraints[0].2,
        "range[0] gelu0[0] should be upper branch (lower bound)"
    );

    assert_eq!(constraints[1].0, "gelu0");
    assert_eq!(constraints[1].1, 0);
    assert!(
        (constraints[1].3.unwrap() - 0.5).abs() < 1e-6,
        "range[1] split_point: expected 0.5, got {}",
        constraints[1].3.unwrap()
    );
    assert!(
        !constraints[1].2,
        "range[1] gelu0[0] should not be upper branch (upper bound)"
    );

    assert_eq!(constraints[2].0, "sigmoid0");
    assert_eq!(constraints[2].1, 1);
    assert!(
        (constraints[2].3.unwrap() - 0.0).abs() < 1e-6,
        "single[2] split_point: expected 0.0, got {}",
        constraints[2].3.unwrap()
    );
    assert!(
        constraints[2].2,
        "single[2] sigmoid0[1] should be upper branch"
    );

    // Reconstruct - range splits get flattened (known limitation)
    let reconstructed = history_from_constraints(constraints).unwrap();
    assert_eq!(reconstructed.genbab_constraints.len(), 3);
    assert_eq!(reconstructed.genbab_split_ids.len(), 3);

    // Each constraint now has its own split ID (flattened from original grouping)
    // This is the expected behavior given ConstraintTuple format limitations
    assert_eq!(reconstructed.genbab_split_ids[0], 0);
    assert_eq!(reconstructed.genbab_split_ids[1], 1); // Was 0 in original
    assert_eq!(reconstructed.genbab_split_ids[2], 2); // Was 1 in original

    // Split count = 3 (flattened from 2)
    assert_eq!(reconstructed.split_count, 3);

    // Verify constraint VALUES are still correct
    let c0 = &reconstructed.genbab_constraints[0];
    assert_eq!(c0.node_name, "gelu0");
    assert_eq!(c0.neuron_idx, 0);
    assert!(
        (c0.split_point - (-0.5)).abs() < 1e-6,
        "flattened c0 split_point: expected -0.5, got {}",
        c0.split_point
    );
    assert!(
        c0.is_upper_branch,
        "flattened c0 gelu0[0] should be upper branch"
    );

    let c1 = &reconstructed.genbab_constraints[1];
    assert_eq!(c1.node_name, "gelu0");
    assert_eq!(c1.neuron_idx, 0);
    assert!(
        (c1.split_point - 0.5).abs() < 1e-6,
        "flattened c1 split_point: expected 0.5, got {}",
        c1.split_point
    );
    assert!(
        !c1.is_upper_branch,
        "flattened c1 gelu0[0] should not be upper branch"
    );

    let c2 = &reconstructed.genbab_constraints[2];
    assert_eq!(c2.node_name, "sigmoid0");
    assert_eq!(c2.neuron_idx, 1);
    assert!(
        (c2.split_point - 0.0).abs() < 1e-6,
        "flattened c2 split_point: expected 0.0, got {}",
        c2.split_point
    );
    assert!(
        c2.is_upper_branch,
        "flattened c2 sigmoid0[1] should be upper branch"
    );
}

/// Regression test for #2248: serialize_constraints must return an error when
/// split_count is too low and constraints would be silently dropped.
#[ntest::timeout(5000)]
#[test]
fn test_serialize_constraints_rejects_wrong_split_count_2248() {
    // Create a valid history with 2 ReLU constraints (split_count=2)
    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu0".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu0".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });
    assert_eq!(history.split_count, 2);
    assert_eq!(history.constraints.len(), 2);

    // Corrupt split_count to be too low — this should now return an error
    history.split_count = 1;

    let input_bounds =
        Arc::new(BoundedTensor::new(array![0.0].into_dyn(), array![1.0].into_dyn()).unwrap());
    let node_bounds = HashMap::new();

    let domain = GraphBabDomain {
        history,
        node_bounds,
        lower_bound: 0.0,
        upper_bound: 1.0,
        depth: 0,
        priority: 0.0,
        input_bounds,
        beta_state: GraphBetaState::default(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    let domains = vec![&domain];
    let layer_names: Vec<String> = vec![];
    let err = BatchedDomains::from_graph_domains(&domains, &layer_names)
        .expect_err("mismatched split_count should be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("serialize_constraints: not all constraints consumed"),
        "unexpected error: {msg}"
    );
}
