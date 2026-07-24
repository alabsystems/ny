// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for beta_crown graph domain transitions.
//!
//! Tests split history monotonicity, domain depth invariants, and
//! threshold-mode switching behavior for GraphBabDomain and
//! MultiObjectiveGraphBabDomain.
//!
//! Coverage targets:
//! - `GraphSplitHistory::add_constraint`, `add_genbab_constraint`, `add_genbab_constraints_for_split`
//! - `GraphBabDomain::with_constraint`, `with_general_split`
//! - `MultiObjectiveGraphBabDomain::with_constraint`
//! - `beta_crown/domain.rs` `with_constraint` depth increment (`depth: self.depth + 1`)
//! - branching.rs:332-333 (split_count += 1)
//!
//! Part of #1959.

use super::gpu_bab::simple_graph_network;
use super::prelude::*;
use crate::beta_crown::branching::GenBabConstraint;
use crate::Result;

// =============================================================================
// Helpers
// =============================================================================

/// Build a root GraphBabDomain from the simple_graph_network with IBP bounds.
fn root_domain(verify_upper: bool) -> (GraphNetwork, BoundedTensor, GraphBabDomain) {
    let graph = simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Get IBP bounds for all nodes (HashMap<String, BoundedTensor>)
    let node_bounds = graph.collect_node_bounds(&input).unwrap();

    let root = GraphBabDomain::root(node_bounds, 0.0, 4.0, &input, verify_upper).unwrap();
    (graph, input, root)
}

/// Build a root MultiObjectiveGraphBabDomain from the simple graph.
fn root_multi_domain(
    verify_upper: bool,
) -> (
    GraphNetwork,
    BoundedTensor,
    MultiObjectiveGraphBabDomain,
    Vec<f32>,
) {
    let graph = simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let node_bounds = graph.collect_node_bounds(&input).unwrap();

    // Two objectives: one easy (threshold outside bounds), one hard
    let thresholds = vec![10.0, 0.5];
    let objective_bounds = vec![(0.0, 4.0), (0.0, 4.0)];

    let root = MultiObjectiveGraphBabDomain::root(
        node_bounds,
        objective_bounds,
        &input,
        &thresholds,
        verify_upper,
    )
    .unwrap();
    (graph, input, root, thresholds)
}

/// Create a valid ReLU constraint for relu1, neuron 0, active branch.
fn relu1_active() -> GraphNeuronConstraint {
    GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 1.0,
    }
}

/// Create a valid ReLU constraint for relu1, neuron 0, inactive branch.
fn relu1_inactive() -> GraphNeuronConstraint {
    GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: false,
        score: 1.0,
    }
}

/// Create a valid ReLU constraint for relu1, neuron 1, active branch.
fn relu1_n1_active() -> GraphNeuronConstraint {
    GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 1,
        is_active: true,
        score: 0.8,
    }
}

// =============================================================================
// Split History Monotonicity Tests
// =============================================================================

/// GraphSplitHistory::depth() increases by exactly 1 per add_constraint.
#[test]
fn test_split_history_depth_monotonic_relu() {
    let mut history = GraphSplitHistory::new();
    assert_eq!(history.depth(), 0, "empty history has depth 0");

    for i in 0..5 {
        history.add_constraint(GraphNeuronConstraint {
            node_name: format!("relu_{}", i),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        });
        assert_eq!(
            history.depth(),
            i + 1,
            "depth should be {} after {} ReLU constraint(s)",
            i + 1,
            i + 1
        );
    }
}

/// GraphSplitHistory::depth() increases by exactly 1 per add_genbab_constraint.
#[test]
fn test_split_history_depth_monotonic_genbab() {
    let mut history = GraphSplitHistory::new();

    for i in 0..5 {
        history.add_genbab_constraint(
            GenBabConstraint::new(format!("gelu_{}", i), 0, 0.5, true, 1.0).unwrap(),
        );
        assert_eq!(
            history.depth(),
            i + 1,
            "depth should be {} after {} GenBaB constraint(s)",
            i + 1,
            i + 1
        );
    }
}

/// add_genbab_constraints_for_split increments depth by exactly 1 for a range split
/// (multiple constraints share one split_id).
#[test]
fn test_split_history_range_split_single_depth_increment() {
    let mut history = GraphSplitHistory::new();
    history.add_constraint(relu1_active()); // depth = 1

    let lower = GenBabConstraint::new("gelu1".to_string(), 0, -1.0, true, 1.0).unwrap();
    let upper = GenBabConstraint::new("gelu1".to_string(), 0, 1.0, false, 1.0).unwrap();
    history.add_genbab_constraints_for_split(vec![lower, upper]); // depth = 2 (not 3)

    assert_eq!(
        history.depth(),
        2,
        "range split with 2 constraints should add only 1 to depth"
    );
    assert_eq!(
        history.genbab_constraints.len(),
        2,
        "both GenBaB constraints should be stored"
    );
    // Both constraints share the same split_id
    assert_eq!(
        history.genbab_split_ids[0], history.genbab_split_ids[1],
        "range split constraints should share the same split_id"
    );
}

/// constraints.len() and genbab_constraints.len() are monotonically non-decreasing.
#[test]
fn test_split_history_constraint_counts_monotonic() {
    let mut history = GraphSplitHistory::new();
    let mut prev_relu = 0;
    let mut prev_genbab = 0;

    for i in 0..3 {
        history.add_constraint(GraphNeuronConstraint {
            node_name: format!("relu_{}", i),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        });
        assert!(
            history.constraints.len() >= prev_relu,
            "ReLU constraint count must be non-decreasing"
        );
        prev_relu = history.constraints.len();
    }

    for i in 0..3 {
        history.add_genbab_constraint(
            GenBabConstraint::new(format!("gelu_{}", i), 0, 0.5, true, 1.0).unwrap(),
        );
        assert!(
            history.genbab_constraints.len() >= prev_genbab,
            "GenBaB constraint count must be non-decreasing"
        );
        prev_genbab = history.genbab_constraints.len();
    }

    assert_eq!(history.constraints.len(), 3);
    assert_eq!(history.genbab_constraints.len(), 3);
    assert_eq!(history.depth(), 6); // 3 ReLU + 3 GenBaB
}

/// with_constraint() creates a new history without mutating the original.
#[test]
fn test_split_history_with_constraint_immutable() {
    let history = GraphSplitHistory::new();
    let child = history.with_constraint(relu1_active());

    assert_eq!(history.depth(), 0, "parent should remain unchanged");
    assert_eq!(child.depth(), 1, "child should have depth 1");
}

/// Lookup cache stays in sync after multiple adds.
#[test]
fn test_split_history_lookup_cache_consistent() {
    let mut history = GraphSplitHistory::new();

    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 1.0,
    });
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.5,
    });
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu2".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.7,
    });

    assert_eq!(history.is_constrained("relu1", 0), Some(true));
    assert_eq!(history.is_constrained("relu1", 1), Some(false));
    assert_eq!(history.is_constrained("relu2", 0), Some(true));
    assert_eq!(history.is_constrained("relu2", 1), None); // unconstrained
    assert_eq!(history.is_constrained("relu3", 0), None); // no such node
}

// =============================================================================
// GraphBabDomain Transition Invariant Tests
// =============================================================================

/// Child domain depth = parent depth + 1 via with_constraint.
#[ntest::timeout(10000)]
#[test]
fn test_domain_depth_increments_with_constraint() -> Result<()> {
    let (graph, _input, root) = root_domain(true);
    assert_eq!(root.depth, 0, "root depth should be 0");

    let child = root
        .with_constraint(&graph, relu1_active(), true)?
        .expect("active constraint on unstable neuron should succeed");
    assert_eq!(child.depth, 1, "child depth should be parent + 1");

    // Chain: apply second constraint to child
    let grandchild = child
        .with_constraint(&graph, relu1_n1_active(), true)?
        .expect("second constraint on different neuron should succeed");
    assert_eq!(grandchild.depth, 2, "grandchild depth should be 2");
    Ok(())
}

/// Child domain depth = parent depth + 1 via with_general_split.
///
/// Uses a non-zero split point (0.5) to exercise the GenBaB path.
/// A split at 0.0 would be treated as a ReLU split by `is_relu_split()`.
#[ntest::timeout(10000)]
#[test]
fn test_domain_depth_increments_with_general_split() -> Result<()> {
    let (graph, _input, root) = root_domain(true);

    let split = NeuronSplit {
        layer: LayerRef::Name("relu1".to_string()),
        neuron_idx: 0,
        lower_bound: Some(0.5),
        upper_bound: None,
        score: 1.0,
        input_index: None,
        norm_inv_rms_window: None,
    };

    let child = root
        .with_general_split(&graph, split, true)?
        .expect("valid general split should produce child");
    assert_eq!(child.depth, 1, "general split child depth should be 1");

    // Verify that GenBaB history recorded the constraint (not ReLU path)
    assert_eq!(
        child.history.genbab_constraints.len(),
        1,
        "non-zero general split should add one GenBaB constraint to history"
    );
    assert_eq!(
        child.history.split_count, 1,
        "general split should increment split_count"
    );
    Ok(())
}

/// with_constraint returns None for a non-ReLU node.
#[ntest::timeout(10000)]
#[test]
fn test_constraint_on_non_relu_returns_none() -> Result<()> {
    let (graph, _input, root) = root_domain(true);

    let bad_constraint = GraphNeuronConstraint {
        node_name: "linear1".to_string(), // not a ReLU node
        neuron_idx: 0,
        is_active: true,
        score: 1.0,
    };

    assert!(
        root.with_constraint(&graph, bad_constraint, true)?
            .is_none(),
        "constraint on linear node should return None"
    );
    Ok(())
}

/// with_constraint returns None for neuron_idx out of bounds.
#[ntest::timeout(10000)]
#[test]
fn test_constraint_oob_neuron_returns_none() -> Result<()> {
    let (graph, _input, root) = root_domain(true);

    let oob_constraint = GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 999, // way out of bounds
        is_active: true,
        score: 1.0,
    };

    assert!(
        root.with_constraint(&graph, oob_constraint, true)?
            .is_none(),
        "out-of-bounds neuron index should return None"
    );
    Ok(())
}

/// with_constraint returns None for a nonexistent node name.
#[ntest::timeout(10000)]
#[test]
fn test_constraint_nonexistent_node_returns_none() -> Result<()> {
    let (graph, _input, root) = root_domain(true);

    let bad_constraint = GraphNeuronConstraint {
        node_name: "nonexistent_relu".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 1.0,
    };

    assert!(
        root.with_constraint(&graph, bad_constraint, true)?
            .is_none(),
        "nonexistent node name should return None"
    );
    Ok(())
}

/// Both branches (active and inactive) of an unstable neuron produce valid children.
#[ntest::timeout(10000)]
#[test]
fn test_both_branches_produce_children() -> Result<()> {
    let (graph, _input, root) = root_domain(true);

    let active_child = root.with_constraint(&graph, relu1_active(), true)?;
    let inactive_child = root.with_constraint(&graph, relu1_inactive(), true)?;

    assert!(
        active_child.is_some(),
        "active branch should produce child for unstable neuron"
    );
    assert!(
        inactive_child.is_some(),
        "inactive branch should produce child for unstable neuron"
    );

    let ac = active_child.unwrap();
    let ic = inactive_child.unwrap();

    // Both should have depth 1
    assert_eq!(ac.depth, 1);
    assert_eq!(ic.depth, 1);

    // History should reflect the constraint
    assert_eq!(ac.history.depth(), 1);
    assert_eq!(ic.history.depth(), 1);
    assert_eq!(ac.history.is_constrained("relu1", 0), Some(true));
    assert_eq!(ic.history.is_constrained("relu1", 0), Some(false));
    Ok(())
}

/// Domain split history depth matches domain depth along a chain.
#[ntest::timeout(10000)]
#[test]
fn test_history_depth_matches_domain_depth() -> Result<()> {
    let (graph, _input, root) = root_domain(true);
    assert_eq!(root.history.depth(), root.depth);

    let child = root.with_constraint(&graph, relu1_active(), true)?.unwrap();
    assert_eq!(
        child.history.depth(),
        child.depth,
        "history depth should equal domain depth after with_constraint"
    );

    let grandchild = child
        .with_constraint(&graph, relu1_n1_active(), true)?
        .unwrap();
    assert_eq!(
        grandchild.history.depth(),
        grandchild.depth,
        "history depth should track domain depth through chain"
    );
    Ok(())
}

// =============================================================================
// Threshold-Mode Switching Tests
// =============================================================================

/// verify_upper_bound mode does not affect whether with_constraint creates a child.
/// Only priority changes; child existence is mode-invariant.
#[ntest::timeout(10000)]
#[test]
fn test_threshold_mode_does_not_affect_child_creation() -> Result<()> {
    let (graph_t, _input_t, root_t) = root_domain(true);
    let (graph_f, _input_f, root_f) = root_domain(false);

    let constraint = relu1_active();

    let child_upper = root_t.with_constraint(&graph_t, constraint.clone(), true)?;
    let child_lower = root_f.with_constraint(&graph_f, constraint, false)?;

    assert_eq!(
        child_upper.is_some(),
        child_lower.is_some(),
        "child creation should not depend on verify_upper_bound mode"
    );

    let cu = child_upper.unwrap();
    let cl = child_lower.unwrap();

    // Depth and history must be identical
    assert_eq!(cu.depth, cl.depth, "depth must be mode-invariant");
    assert_eq!(
        cu.history.depth(),
        cl.history.depth(),
        "history depth must be mode-invariant"
    );

    // Node bounds should be identical (same constraint applied)
    assert_eq!(
        cu.node_bounds.len(),
        cl.node_bounds.len(),
        "node bounds count must be mode-invariant"
    );
    Ok(())
}

/// verify_upper_bound mode changes priority direction.
#[ntest::timeout(10000)]
#[test]
fn test_threshold_mode_flips_priority_direction() -> Result<()> {
    let (graph_t, _input_t, root_t) = root_domain(true);
    let (graph_f, _input_f, root_f) = root_domain(false);

    // For verify_upper: priority = upper_bound = 4.0
    // For !verify_upper: priority = -lower_bound = -0.0 = 0.0
    // These must differ to confirm mode actually changes priority.
    assert!(
        (root_t.priority - 4.0).abs() < 1e-6,
        "verify_upper=true root priority should be upper_bound (4.0), got {}",
        root_t.priority
    );
    assert!(
        root_f.priority.abs() < 1e-6,
        "verify_upper=false root priority should be -lower_bound (0.0), got {}",
        root_f.priority
    );
    assert!(
        (root_t.priority - root_f.priority).abs() > 1e-6,
        "priorities must differ between verify_upper modes"
    );

    let child_t = root_t
        .with_constraint(&graph_t, relu1_active(), true)?
        .unwrap();
    let child_f = root_f
        .with_constraint(&graph_f, relu1_active(), false)?
        .unwrap();

    // Priorities should differ (different formulas)
    // We just verify both are finite
    assert!(
        child_t.priority.is_finite(),
        "verify_upper child priority must be finite"
    );
    assert!(
        child_f.priority.is_finite(),
        "!verify_upper child priority must be finite"
    );
    Ok(())
}

// =============================================================================
// MultiObjectiveGraphBabDomain Transition Tests
// =============================================================================

/// Multi-objective root domain has correct initial state.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_root_domain_state() {
    let (_graph, _input, root, thresholds) = root_multi_domain(true);

    assert_eq!(root.depth, 0, "root depth should be 0");
    assert_eq!(root.history.depth(), 0, "root history should be empty");
    assert_eq!(
        root.objective_bounds.len(),
        2,
        "should have 2 objective bounds"
    );
    assert_eq!(root.verified.len(), 2, "should have 2 verified flags");

    // First objective: bounds (0,4), threshold 10.0, verify_upper=true
    // upper(4) < threshold(10)? Yes -> verified
    assert!(
        root.verified[0],
        "first objective should be verified (upper 4 < threshold 10)"
    );

    // Second objective: bounds (0,4), threshold 0.5, verify_upper=true
    // upper(4) < threshold(0.5)? No -> not verified
    assert!(
        !root.verified[1],
        "second objective should not be verified (upper 4 >= threshold 0.5)"
    );

    assert!(
        !root.all_verified(),
        "not all objectives verified when second is open"
    );
    assert_eq!(root.verified_count(), 1, "exactly one objective verified");

    // Ensure thresholds are used correctly
    assert_eq!(thresholds.len(), 2);
}

/// Multi-objective domain child has depth = parent + 1.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_depth_increments() -> Result<()> {
    let (graph, _input, root, thresholds) = root_multi_domain(true);

    let child = root
        .with_constraint(&graph, relu1_active(), true, &thresholds)?
        .expect("constraint on multi-objective root should produce child");

    assert_eq!(child.depth, 1, "multi-objective child depth should be 1");
    assert_eq!(
        child.history.depth(),
        1,
        "multi-objective child history depth should be 1"
    );
    Ok(())
}

/// Multi-objective with_constraint returns None for non-ReLU node.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_rejects_non_relu() -> Result<()> {
    let (graph, _input, root, thresholds) = root_multi_domain(true);

    let bad = GraphNeuronConstraint {
        node_name: "linear1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 1.0,
    };

    assert!(
        root.with_constraint(&graph, bad, true, &thresholds)?
            .is_none(),
        "multi-objective constraint on linear node should return None"
    );
    Ok(())
}

/// Multi-objective inherits verified status from parent.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_child_preserves_verified_count() -> Result<()> {
    let (graph, _input, root, thresholds) = root_multi_domain(true);

    let child = root
        .with_constraint(&graph, relu1_active(), true, &thresholds)?
        .unwrap();

    // Child inherits parent's verified vector (bounds are not re-checked in with_constraint,
    // they get updated during the next propagation pass)
    assert_eq!(
        child.verified.len(),
        root.verified.len(),
        "child should have same number of objectives"
    );
    assert_eq!(
        child.objective_bounds.len(),
        root.objective_bounds.len(),
        "child should inherit objective bounds vector length"
    );
    Ok(())
}

// =============================================================================
// SplitHistory <-> GraphSplitHistory Conversion
// =============================================================================

/// SplitHistory::to_graph_split_history preserves constraint count and depth.
#[test]
fn test_sequential_to_graph_history_preserves_depth() {
    let mut seq = SplitHistory::new();
    seq.add_constraint(NeuronConstraint::new(0, 0, true, 1.0).unwrap());
    seq.add_constraint(NeuronConstraint::new(2, 1, false, 0.5).unwrap());

    let graph_hist = seq.to_graph_split_history().expect("valid history");

    assert_eq!(
        graph_hist.depth(),
        seq.depth(),
        "converted graph history depth should match sequential depth"
    );
    assert_eq!(
        graph_hist.constraints.len(),
        seq.constraints.len(),
        "converted constraint count should match"
    );

    // Verify the naming convention: layer_idx -> "layer_{idx}"
    assert_eq!(
        graph_hist.is_constrained("layer_0", 0),
        Some(true),
        "layer 0 neuron 0 should be active"
    );
    assert_eq!(
        graph_hist.is_constrained("layer_2", 1),
        Some(false),
        "layer 2 neuron 1 should be inactive"
    );
}

// =============================================================================
// General Split (GenBaB) Domain Transition Tests
// =============================================================================

/// with_general_split returns None when split bounds are infeasible.
#[ntest::timeout(10000)]
#[test]
fn test_general_split_infeasible_returns_none() -> Result<()> {
    let (graph, _input, root) = root_domain(true);

    // Create a split with bounds that don't intersect the current neuron bounds.
    // relu1 pre-activation bounds for neuron 0 span a range crossing 0
    // (since input is [-1,1] and linear1 weight row 0 is [1,-1], output is [-2,2]).
    // Setting lower_bound > current upper should be infeasible.
    let split = NeuronSplit {
        layer: LayerRef::Name("relu1".to_string()),
        neuron_idx: 0,
        lower_bound: Some(100.0), // way above any possible pre-activation
        upper_bound: Some(200.0),
        score: 1.0,
        input_index: None,
        norm_inv_rms_window: None,
    };

    assert!(
        root.with_general_split(&graph, split, true)?.is_none(),
        "infeasible general split should return None"
    );
    Ok(())
}

/// with_general_split on a nonexistent node returns None.
#[ntest::timeout(10000)]
#[test]
fn test_general_split_nonexistent_node_returns_none() -> Result<()> {
    let (graph, _input, root) = root_domain(true);

    let split = NeuronSplit {
        layer: LayerRef::Name("nonexistent".to_string()),
        neuron_idx: 0,
        lower_bound: Some(0.0),
        upper_bound: None,
        score: 1.0,
        input_index: None,
        norm_inv_rms_window: None,
    };

    assert!(
        root.with_general_split(&graph, split, true)?.is_none(),
        "split on nonexistent node should return None"
    );
    Ok(())
}

/// with_general_split rejects NaN split bounds instead of silently absorbing them (#2954).
///
/// Before the fix, IEEE 754 `f32::max(current_l, NaN) = current_l` would silently
/// drop the NaN, producing a child domain with seemingly valid bounds. The fix uses
/// `nan_propagating_max`/`nan_propagating_min` plus an explicit NaN guard that
/// returns `NumericalInstability` before NaN can reach constraints or BoundedTensor.
/// This test targets the NETWORK_INPUT path (splitting on linear1).
#[ntest::timeout(10000)]
#[test]
fn test_general_split_nan_lower_bound_not_absorbed_2954() -> Result<()> {
    // Use simple_graph_network: input -> linear1 -> relu1 -> linear2
    // linear1's input is NETWORK_INPUT.
    // Splitting on linear1 constrains NETWORK_INPUT bounds directly, exercising
    // the code path where effective_l is written into child BoundedTensor.
    let graph = simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let root = GraphBabDomain::root(node_bounds, 0.0, 4.0, &input, true).unwrap();

    // Split on linear1, neuron 0, with NaN lower bound.
    // linear1's pre-activation is NETWORK_INPUT, so the NETWORK_INPUT branch
    // writes effective_l into child bounds — NaN from nan_propagating_max
    // causes BoundedTensor::new rejection.
    let split = NeuronSplit {
        layer: LayerRef::Name("linear1".to_string()),
        neuron_idx: 0,
        lower_bound: Some(f32::NAN), // NaN split bound
        upper_bound: None,
        score: 1.0,
        input_index: None,
        norm_inv_rms_window: None,
    };

    // With the fix: nan_propagating_max(current_l, NaN) = NaN, caught by
    // the explicit NaN guard before reaching BoundedTensor::new or constraints.
    // Before the fix: f32::max(current_l, NaN) = current_l, silently dropping NaN.
    let result = root.with_general_split(&graph, split, true);
    assert!(
        result.is_err(),
        "NaN split bound should be caught by NaN guard, got: {result:?}"
    );

    Ok(())
}

/// with_general_split rejects NaN upper split bound (#2954).
#[ntest::timeout(10000)]
#[test]
fn test_general_split_nan_upper_bound_not_absorbed_2954() -> Result<()> {
    let graph = simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let root = GraphBabDomain::root(node_bounds, 0.0, 4.0, &input, true).unwrap();

    // Split on linear1 with NaN upper bound — same NETWORK_INPUT path.
    let split = NeuronSplit {
        layer: LayerRef::Name("linear1".to_string()),
        neuron_idx: 0,
        lower_bound: None,
        upper_bound: Some(f32::NAN), // NaN upper split bound
        score: 1.0,
        input_index: None,
        norm_inv_rms_window: None,
    };

    let result = root.with_general_split(&graph, split, true);
    assert!(
        result.is_err(),
        "NaN upper split bound should be caught by NaN guard, got: {result:?}"
    );

    Ok(())
}

/// with_general_split rejects NaN split bounds on non-NETWORK_INPUT path (#2954).
///
/// When splitting on relu1, the pre-activation is linear1 (not NETWORK_INPUT),
/// so the code takes the else branch at line 362. Without the explicit NaN guard,
/// NaN would silently flow into GenBabConstraint objects. The guard catches NaN
/// before it reaches either code path.
#[ntest::timeout(10000)]
#[test]
fn test_general_split_nan_non_network_input_path_2954() -> Result<()> {
    let graph = simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let root = GraphBabDomain::root(node_bounds, 0.0, 4.0, &input, true).unwrap();

    // Split on relu1, neuron 0, with NaN lower bound.
    // relu1's pre-activation is linear1 (not NETWORK_INPUT), so this exercises
    // the non-NETWORK_INPUT code path where effective_l would flow into
    // GenBabConstraint without the explicit NaN guard.
    let split = NeuronSplit {
        layer: LayerRef::Name("relu1".to_string()),
        neuron_idx: 0,
        lower_bound: Some(f32::NAN),
        upper_bound: None,
        score: 1.0,
        input_index: None,
        norm_inv_rms_window: None,
    };

    let result = root.with_general_split(&graph, split, true);
    assert!(
        result.is_err(),
        "NaN split bound on non-NETWORK_INPUT path should be caught by NaN guard, got: {result:?}"
    );

    Ok(())
}
