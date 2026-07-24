// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for graph network β state.

use super::*;

fn make_entry(node_name: &str, neuron_idx: usize, value: f32, grad: f32) -> GraphBetaEntry {
    GraphBetaEntry {
        node_name: node_name.to_string(),
        neuron_idx,
        split_point: 0.0,
        value,
        sign: 1.0,
        grad,
        m: 0.0,
        v: 0.0,
        v_max: 0.0,
    }
}

// Build fixtures through the indexed constructor so lookup-sensitive tests
// exercise the same state shape as production domains.
fn make_state(entries: Vec<GraphBetaEntry>) -> GraphBetaState {
    GraphBetaState::from_entries(entries)
}

/// Regression test for #2980: from_history_with_warmup must sanitize NaN/Inf
/// in parent optimizer state fields (value, m, v, v_max) instead of copying
/// them through to the child domain.
#[test]
fn test_from_history_with_warmup_sanitizes_nan_parent_state_2980() {
    use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};

    // Create a parent state with NaN-corrupted optimizer fields.
    // This simulates NaN leaking into m/v/v_max through a code path that
    // the gradient_step NaN guard didn't catch.
    let parent_state = make_state(vec![GraphBetaEntry {
        node_name: "relu0".to_string(),
        neuron_idx: 0,
        split_point: 0.0,
        value: f32::NAN,
        sign: 1.0,
        grad: 0.0,
        m: f32::NAN,
        v: f32::INFINITY,
        v_max: f32::NEG_INFINITY,
    }]);

    // Create a history that matches the parent constraint
    let mut history = GraphSplitHistory::new();
    history.add_constraint(
        GraphNeuronConstraint::new("relu0".to_string(), 0, true, 1.0).expect("valid constraint"),
    );

    let child_state = GraphBetaState::from_history_with_warmup(&history, &parent_state, 0.0)
        .expect("warmup from valid history");

    assert_eq!(child_state.entries.len(), 1);
    let entry = &child_state.entries[0];

    // Value must be sanitized (NaN → 0.0 via constructor)
    assert!(
        entry.value().is_finite(),
        "NaN parent value must be sanitized, got {}",
        entry.value()
    );
    assert_eq!(entry.value(), 0.0);

    // Adam state must be sanitized
    assert!(entry.m.is_finite(), "NaN parent m must be sanitized to 0.0");
    assert_eq!(entry.m, 0.0);
    assert!(entry.v.is_finite(), "Inf parent v must be sanitized to 0.0");
    assert_eq!(entry.v, 0.0);
    assert!(
        entry.v_max.is_finite(),
        "-Inf parent v_max must be sanitized to 0.0"
    );
    assert_eq!(entry.v_max, 0.0);
}

/// Regression test for #2980: from_history_with_warmup preserves valid parent
/// optimizer state (m, v, v_max) when values are finite.
#[test]
fn test_from_history_with_warmup_preserves_valid_parent_state_2980() {
    use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};

    let parent_state = make_state(vec![GraphBetaEntry {
        node_name: "relu0".to_string(),
        neuron_idx: 0,
        split_point: 0.0,
        value: 0.75,
        sign: 1.0,
        grad: 0.5,
        m: 0.1,
        v: 0.02,
        v_max: 0.03,
    }]);

    let mut history = GraphSplitHistory::new();
    history.add_constraint(
        GraphNeuronConstraint::new("relu0".to_string(), 0, true, 1.0).expect("valid constraint"),
    );

    let child_state = GraphBetaState::from_history_with_warmup(&history, &parent_state, 0.0)
        .expect("warmup from valid history");

    let entry = &child_state.entries[0];

    // Valid parent state should be preserved
    assert_eq!(entry.value(), 0.75, "valid parent value preserved");
    assert_eq!(entry.m, 0.1, "valid parent m preserved");
    assert_eq!(entry.v, 0.02, "valid parent v preserved");
    assert_eq!(entry.v_max, 0.03, "valid parent v_max preserved");
    // Gradient should be reset for new optimization
    assert_eq!(entry.grad(), 0.0, "gradient reset for child domain");
}

/// Regression test for #2939: standard (non-Adam) gradient_step must recover
/// from NaN gradients instead of permanently corrupting graph beta values.
#[test]
fn test_gradient_step_nan_recovery_2939() {
    let mut state = make_state(vec![
        make_entry("relu0", 0, 0.5, 0.1),
        make_entry("relu0", 1, 0.3, f32::NAN),
    ]);

    let max_grad = state.gradient_step(0.01);

    // NaN-infected entry must be reset
    assert!(
        state.entries[1].value().is_finite(),
        "graph beta value must be finite after NaN gradient, got {}",
        state.entries[1].value()
    );
    assert_eq!(
        state.entries[1].value(),
        0.0,
        "NaN-infected graph beta must reset to 0.0"
    );
    assert_eq!(
        state.entries[1].grad(),
        0.0,
        "NaN-infected graph grad must reset to 0.0"
    );

    // Valid entry must still be updated normally
    assert!(
        state.entries[0].value() > 0.5,
        "valid graph beta should increase, got {}",
        state.entries[0].value()
    );

    // max_grad must be NaN (nan_propagating_max propagates NaN)
    assert!(
        max_grad.is_nan(),
        "max_grad should be NaN when any gradient is NaN, got {max_grad}"
    );
}

/// Regression test for #2939: after NaN recovery, the next gradient step must
/// work normally (corruption is not permanent).
#[test]
fn test_gradient_step_nan_recovery_then_normal_2939() {
    let mut state = make_state(vec![make_entry("relu0", 0, 0.5, f32::NAN)]);

    // Step 1: NaN corrupts, then recovery
    let _ = state.gradient_step(0.01);
    assert_eq!(state.entries[0].value(), 0.0);
    assert_eq!(state.entries[0].grad(), 0.0);

    // Step 2: valid gradient — should work normally
    state.entries[0].grad = 1.0;
    let max_grad = state.gradient_step(0.1);

    assert!(
        state.entries[0].value() > 0.0,
        "beta should increase after recovery, got {}",
        state.entries[0].value()
    );
    assert!(
        max_grad.is_finite(),
        "max_grad should be finite with valid gradient, got {max_grad}"
    );
}

/// Regression test for #2600: NaN-margin objectives must be selected as critical
/// (worst case) in multi-objective gradient computation.
///
/// IEEE 754: NaN < min_margin = false, so without the NEG_INFINITY guard,
/// NaN objectives are silently skipped and gradients optimize the wrong objective.
#[test]
fn test_multi_objective_nan_lb_selects_nan_as_critical_2600() {
    use crate::bounds::GraphAlphaCrownIntermediate;
    use ndarray::arr2;

    // State with one beta entry at relu0, neuron 0
    let mut state = make_state(vec![make_entry("relu0", 0, 0.5, 0.0)]);

    // A matrix at relu0: 2 outputs × 1 neuron = [[a00], [a10]]
    let mut intermediate = GraphAlphaCrownIntermediate::new();
    intermediate
        .a_at_relu
        .insert("relu0".to_string(), arr2(&[[1.0], [2.0]]));

    // Objective 0: lb=NaN (corrupted), objective 1: lb=-0.5 (valid)
    let obj_bounds: Vec<(f32, f32)> = vec![(f32::NAN, 1.0), (-0.5, 0.5)];
    // Objective coefficients: obj0 = [1.0, 0.0], obj1 = [0.0, 1.0]
    let objectives = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let thresholds = vec![0.0, 0.0];
    let verified_mask = vec![false, false];

    let max_grad = state.compute_analytical_gradients_multi_objective(
        &intermediate,
        &obj_bounds,
        &objectives,
        &thresholds,
        &verified_mask,
        false,
    );

    // Objective 0 (NaN) should be selected as critical.
    // With obj0 coefficients [1.0, 0.0] and A = [[1.0], [2.0]]:
    //   sensitivity = 1.0*1.0 + 0.0*2.0 = 1.0
    //   grad = -sign(1.0) * 1.0 = -1.0
    let grad = state.entries[0].grad();
    assert!(
        grad.is_finite(),
        "gradient should be finite (NaN objective selected, valid A matrix), got {grad}"
    );
    assert_eq!(
        grad, -1.0,
        "gradient should reflect objective 0 (NaN lb) as critical, not objective 1"
    );
    assert!(
        max_grad.is_finite(),
        "max_grad should be finite, got {max_grad}"
    );
}

/// Regression test for #2600: when only one objective has NaN lb and all others
/// are verified, the NaN objective must still be selected (not all-verified).
#[test]
fn test_multi_objective_nan_lb_not_masked_by_verified_2600() {
    use crate::bounds::GraphAlphaCrownIntermediate;
    use ndarray::arr2;

    let mut state = make_state(vec![make_entry("relu0", 0, 0.5, 0.0)]);

    let mut intermediate = GraphAlphaCrownIntermediate::new();
    intermediate
        .a_at_relu
        .insert("relu0".to_string(), arr2(&[[1.0], [2.0]]));

    // Objective 0: NaN (unverified), objective 1: verified
    let obj_bounds: Vec<(f32, f32)> = vec![(f32::NAN, 1.0), (0.5, 1.0)];
    let objectives = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let thresholds = vec![0.0, 0.0];
    let verified_mask = vec![false, true]; // obj1 is verified

    let _max_grad = state.compute_analytical_gradients_multi_objective(
        &intermediate,
        &obj_bounds,
        &objectives,
        &thresholds,
        &verified_mask,
        false,
    );

    // Even with obj1 verified, obj0 (NaN) should be selected as critical
    let grad = state.entries[0].grad();
    assert_eq!(
        grad, -1.0,
        "NaN objective must be selected when other objectives are verified"
    );
}

/// Regression test for #4306: when spec-guided intermediates already store one
/// row per objective, gradient computation must consume the selected row
/// directly instead of reapplying the objective coefficients.
#[test]
fn test_multi_objective_spec_rows_use_direct_row_sensitivity_4306() {
    use crate::bounds::GraphAlphaCrownIntermediate;
    use ndarray::arr2;

    let mut state = make_state(vec![make_entry("relu0", 0, 0.5, 0.0)]);

    let mut intermediate = GraphAlphaCrownIntermediate::new();
    intermediate
        .a_at_relu
        .insert("relu0".to_string(), arr2(&[[1.5], [-3.0]]));

    let obj_bounds: Vec<(f32, f32)> = vec![(0.25, 1.0), (-0.5, 0.5)];
    let thresholds = vec![0.0, 0.0];
    let verified_mask = vec![false, false];

    let max_grad = state.compute_analytical_gradients_multi_objective_spec_rows(
        &intermediate,
        &obj_bounds,
        &thresholds,
        &verified_mask,
        false,
    );

    assert_eq!(
        state.entries[0].grad(),
        3.0,
        "disjunctive mode should select the worst-margin objective row directly"
    );
    assert_eq!(max_grad, 3.0);
}

/// Regression test for #4306: conjunctive critical-objective selection should
/// continue using max beta sensitivity even when intermediates are spec rows.
#[test]
fn test_multi_objective_spec_rows_conjunctive_prefers_max_sensitivity_4306() {
    use crate::bounds::GraphAlphaCrownIntermediate;
    use ndarray::arr2;

    let mut state = make_state(vec![make_entry("relu0", 0, 0.5, 0.0)]);

    let mut intermediate = GraphAlphaCrownIntermediate::new();
    intermediate
        .a_at_relu
        .insert("relu0".to_string(), arr2(&[[0.2], [5.0]]));

    let obj_bounds: Vec<(f32, f32)> = vec![(0.8, 1.0), (-0.1, 0.2)];
    let thresholds = vec![0.0, 0.0];
    let verified_mask = vec![false, false];

    let max_grad = state.compute_analytical_gradients_multi_objective_spec_rows(
        &intermediate,
        &obj_bounds,
        &thresholds,
        &verified_mask,
        true,
    );

    assert_eq!(
        state.entries[0].grad(),
        -5.0,
        "conjunctive mode should follow the most beta-sensitive objective row"
    );
    assert_eq!(max_grad, 5.0);
}

/// Build a parent beta state with `n` constraints spread across nodes,
/// each with distinct optimized values and Adam state.
fn build_scaled_parent_state(n: usize) -> GraphBetaState {
    let entries = (0..n)
        .map(|i| GraphBetaEntry {
            node_name: format!("relu_{}", i / 50),
            neuron_idx: i % 50,
            split_point: 0.0,
            value: 0.1 + (i as f32) * 0.001,
            sign: 1.0,
            grad: 0.0,
            m: 0.01 * (i as f32),
            v: 0.001 * (i as f32),
            v_max: 0.002 * (i as f32),
        })
        .collect();
    GraphBetaState::from_entries(entries)
}

/// Build a split history matching the parent state pattern, plus one new constraint.
fn build_scaled_history(n: usize) -> crate::beta_crown::branching::GraphSplitHistory {
    use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
    let mut history = GraphSplitHistory::new();
    for i in 0..n {
        history.add_constraint(
            GraphNeuronConstraint::new(format!("relu_{}", i / 50), i % 50, true, 1.0)
                .expect("valid constraint"),
        );
    }
    // New constraint from the split that created this child domain
    history.add_constraint(
        GraphNeuronConstraint::new("relu_10".to_string(), 0, true, 1.0).expect("valid constraint"),
    );
    history
}

/// Performance regression: from_history_with_warmup at BaB-realistic scale.
///
/// Current implementation: O(C²) via linear scan (entry_for_constraint called
/// C times). Exercises 500 constraints — VNN-COMP realistic BaB depth.
/// Tracks: #2936, #2999
#[test]
fn test_from_history_with_warmup_scaling_500_constraints() {
    let n = 500;
    let parent_state = build_scaled_parent_state(n);
    let history = build_scaled_history(n);

    let child = GraphBetaState::from_history_with_warmup(&history, &parent_state, 0.0)
        .expect("warmup from valid history at scale");

    assert_eq!(child.entries.len(), n + 1);

    // First entry inherits parent value
    assert_eq!(child.entries[0].node_name, "relu_0");
    assert!((child.entries[0].value() - 0.1).abs() < 1e-6);

    // Mid entry (#249) inherits parent value and Adam state
    let mid = &child.entries[249];
    assert_eq!(mid.node_name, "relu_4");
    assert!((mid.value() - (0.1 + 249.0 * 0.001)).abs() < 1e-4);
    assert!((mid.m - 0.01 * 249.0).abs() < 1e-4);

    // New constraint gets default init
    let new_entry = &child.entries[n];
    assert_eq!(new_entry.node_name, "relu_10");
    assert_eq!(new_entry.value(), 0.0);
    assert_eq!(new_entry.m, 0.0);
    assert_eq!(new_entry.v, 0.0);
}

/// Create a beta entry with specific split_point and sign (extends make_entry).
fn make_entry_full(
    node_name: &str,
    neuron_idx: usize,
    value: f32,
    split_point: f32,
    sign: f32,
) -> GraphBetaEntry {
    GraphBetaEntry {
        node_name: node_name.to_string(),
        neuron_idx,
        split_point,
        value,
        sign,
        grad: 0.0,
        m: 0.0,
        v: 0.0,
        v_max: 0.0,
    }
}

/// Verifies entry_for_constraint distinguishes same-node constraints by
/// neuron_idx, split_point, and sign (the common BaB case). Tracks: #2936
#[test]
fn test_entry_for_constraint_same_node_different_neurons() {
    let state = make_state(vec![
        make_entry_full("relu_0", 0, 0.5, 0.0, 1.0),
        make_entry_full("relu_0", 1, 0.75, 0.0, 1.0),
        make_entry_full("relu_0", 2, 1.0, 0.5, -1.0),
    ]);

    // Exact match
    let found = state.entry_for_constraint("relu_0", 1, 0.0, 1.0);
    assert!(found.is_some(), "should find neuron 1");
    assert!((found.unwrap().value() - 0.75).abs() < 1e-6);

    // Mismatches: split_point, sign, node
    assert!(state.entry_for_constraint("relu_0", 0, 0.5, 1.0).is_none());
    assert!(state.entry_for_constraint("relu_0", 2, 0.5, 1.0).is_none());
    assert!(state.entry_for_constraint("relu_1", 0, 0.0, 1.0).is_none());

    // Match with negative sign and non-zero split_point
    let found = state.entry_for_constraint("relu_0", 2, 0.5, -1.0);
    assert!(found.is_some());
    assert!((found.unwrap().value() - 1.0).abs() < 1e-6);
}

#[test]
fn test_graph_beta_state_rebuilds_lookup_after_entries_push_2936() {
    use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};

    let mut history = GraphSplitHistory::new();
    history.add_constraint(
        GraphNeuronConstraint::new("relu0".to_string(), 0, true, 1.0).expect("valid constraint"),
    );
    let mut state =
        GraphBetaState::from_history_with_init(&history, 0.5).expect("indexed graph beta state");

    state
        .entries
        .push(make_entry_full("relu0", 0, 0.25, 0.5, -1.0));

    let stale_entries: Vec<_> = state.entries_for_node("relu0").collect();
    assert_eq!(
        stale_entries.len(),
        2,
        "stale lookup falls back to linear scan"
    );
    let stale_values: Vec<f32> = stale_entries.iter().map(|entry| entry.value()).collect();
    assert_eq!(stale_values, vec![0.5, 0.25]);
    assert!(state.has_node_entries("relu0"));

    let first = state
        .entry("relu0", 0)
        .expect("first entry remains addressable");
    assert!((first.value() - 0.5).abs() < 1e-6);
    assert_eq!(first.sign(), 1.0);

    let exact = state
        .entry_for_constraint("relu0", 0, 0.5, -1.0)
        .expect("new split-specific entry must be found");
    assert!((exact.value() - 0.25).abs() < 1e-6);

    let signed = state
        .signed_beta("relu0", 0)
        .expect("all matching entries should contribute");
    assert!((signed - 0.25).abs() < 1e-6);

    state.accumulate_grad("relu0", 0, 1.5);
    assert!((state.entries[0].grad() - 1.5).abs() < 1e-6);
    assert!((state.entries[1].grad() - 1.5).abs() < 1e-6);
}

/// Part of #2936: verify `has_node_entries` and `entries_for_node` produce
/// results consistent with the old linear scan pattern.
#[test]
fn test_has_node_entries_and_entries_for_node_2936() {
    let entries = vec![
        make_entry("relu0", 0, 0.5, 0.0),
        make_entry("relu0", 3, 0.3, 0.0),
        make_entry("relu1", 1, 0.7, 0.0),
        make_entry("relu0", 0, 0.2, 0.0), // duplicate (node, neuron) — different split
    ];
    let state = GraphBetaState::from_entries(entries);

    // has_node_entries: true for nodes with entries, false for absent nodes.
    assert!(state.has_node_entries("relu0"));
    assert!(state.has_node_entries("relu1"));
    assert!(!state.has_node_entries("relu2"));
    assert!(!state.has_node_entries(""));

    // entries_for_node: yields exactly the entries for that node.
    let relu0: Vec<_> = state.entries_for_node("relu0").collect();
    assert_eq!(
        relu0.len(),
        3,
        "relu0 has 3 entries (idx 0, 3, and duplicate 0)"
    );
    assert!(relu0.iter().all(|e| e.node_name == "relu0"));
    // Verify the indexed iterator preserves the original linear-scan order.
    let neuron_idxs: Vec<usize> = relu0.iter().map(|e| e.neuron_idx).collect();
    assert_eq!(neuron_idxs, vec![0, 3, 0]);
    let values: Vec<f32> = relu0.iter().map(|e| e.value()).collect();
    assert_eq!(values, vec![0.5, 0.3, 0.2]);

    let relu1: Vec<_> = state.entries_for_node("relu1").collect();
    assert_eq!(relu1.len(), 1);
    assert_eq!(relu1[0].neuron_idx, 1);

    let relu2: Vec<_> = state.entries_for_node("relu2").collect();
    assert!(relu2.is_empty(), "absent node yields empty iterator");
}

// ── Performance proofs: O(1) indexed lookups at BaB scale (#2936) ─────────────

/// Performance proof: GraphBetaState index covers all entries after construction,
/// guaranteeing O(1) `entry()` and `signed_beta()` lookups at BaB-realistic scale.
///
/// Without indexed lookups, each backward CROWN pass scans all B entries per ReLU
/// node → O(R*B) per pass. With the neuron_index, lookups are O(1).
///
/// Verifies: `lookup_index_fresh()` returns true after `from_entries()` at scale,
/// and every entry is reachable via the indexed path.
/// Tracks: #2936 Finding 1, Finding 2.
#[test]
fn test_graph_beta_state_index_fresh_at_scale_2936() {
    let n = 500;
    let state = build_scaled_parent_state(n);

    // Index must be fresh.
    assert!(
        state.lookup_index_fresh(),
        "lookup index must be fresh after from_entries with {} entries",
        n
    );
    assert_eq!(state.indexed_entries, state.entries.len());

    // Every entry must be reachable via indexed lookup.
    for entry in &state.entries {
        let found = state
            .entry(&entry.node_name, entry.neuron_idx)
            .expect("every entry must be found via indexed lookup");
        assert_eq!(found.node_name, entry.node_name);
        assert_eq!(found.neuron_idx, entry.neuron_idx);
    }

    // entries_for_node must cover all entries for each node.
    let num_nodes = n.div_ceil(50);
    for i in 0..num_nodes {
        let node_name = format!("relu_{i}");
        let indexed_count = state.entries_for_node(&node_name).count();
        let linear_count = state
            .entries
            .iter()
            .filter(|e| e.node_name == node_name)
            .count();
        assert_eq!(
            indexed_count, linear_count,
            "node {node_name}: indexed ({indexed_count}) must match linear scan ({linear_count})"
        );
    }
}

/// Performance proof: `from_history_with_warmup` produces a fresh index at scale,
/// so child domain lookups are O(1) from the first access.
///
/// Before the fix (#2936 Finding 2), `from_history_with_warmup` used O(B²) linear
/// scans via `entry_for_constraint` for each of B constraints. Now it uses the
/// neuron_index for O(1) per constraint.
///
/// Verifies: child state has fresh index and all parent values are correctly warmed.
/// Tracks: #2936 Finding 2.
#[test]
fn test_graph_beta_from_history_warmup_index_fresh_at_scale_2936() {
    let n = 500;
    let parent_state = build_scaled_parent_state(n);
    let history = build_scaled_history(n);

    let child = GraphBetaState::from_history_with_warmup(&history, &parent_state, 0.0)
        .expect("warmup at scale");

    // Child index must be fresh immediately.
    assert!(
        child.lookup_index_fresh(),
        "child state index must be fresh after from_history_with_warmup"
    );
    assert_eq!(child.indexed_entries, child.entries.len());

    // All parent-matching entries must have warmed values (not default 0.0).
    let warmed_count = child.entries.iter().filter(|e| e.value() > 0.0).count();
    assert_eq!(
        warmed_count, n,
        "all {n} parent-matching entries must have warmed values, got {warmed_count}"
    );

    // The new constraint (not in parent) must have default value.
    let new_entry = &child.entries[n];
    assert_eq!(
        new_entry.value(),
        0.0,
        "new constraint must have default 0.0"
    );
}

/// Performance proof: `signed_beta` uses indexed path at scale, correctly summing
/// multiple entries for the same (node, neuron) pair.
///
/// In BaB trees, the same neuron can be constrained multiple times at different
/// split points. `signed_beta` must find all of them via the index.
/// Tracks: #2936 Finding 3.
#[test]
fn test_graph_beta_signed_beta_indexed_multi_entry_2936() {
    // Create entries with duplicate (node, neuron) pairs but different signs.
    let entries = vec![
        make_entry_full("relu_0", 5, 0.3, 0.0, 1.0),
        make_entry_full("relu_0", 5, 0.2, 0.5, -1.0),
        make_entry_full("relu_0", 5, 0.1, 1.0, 1.0),
        make_entry("relu_1", 0, 0.5, 0.0),
    ];
    let state = make_state(entries);

    assert!(state.lookup_index_fresh());

    // signed_beta should sum: 0.3*1.0 + 0.2*(-1.0) + 0.1*1.0 = 0.2
    let signed = state
        .signed_beta("relu_0", 5)
        .expect("must find multi-entry neuron");
    assert!(
        (signed - 0.2).abs() < 1e-6,
        "signed_beta must sum all entries: expected 0.2, got {signed}"
    );

    // Single-entry neuron: 0.5 * 1.0 = 0.5
    let single = state
        .signed_beta("relu_1", 0)
        .expect("must find single-entry neuron");
    assert!((single - 0.5).abs() < 1e-6);

    // Absent neuron: None.
    assert!(state.signed_beta("relu_2", 0).is_none());
}

/// Summarize a beta entry for comparison: (node_name, neuron_idx, split, value, sign).
fn entry_summary(entry: &GraphBetaEntry) -> (String, usize, f32, f32, f32) {
    (
        entry.node_name().to_string(),
        entry.neuron_idx(),
        entry.split_point(),
        entry.value(),
        entry.sign(),
    )
}

fn make_four_entry_state() -> GraphBetaState {
    make_state(vec![
        make_entry("relu0", 0, 0.5, 0.0),
        make_entry("relu1", 1, 0.7, 0.0),
        make_entry("relu0", 3, 0.3, 0.0),
        make_entry_full("relu2", 0, 0.4, 0.25, -1.0),
    ])
}

#[test]
fn test_entries_for_node_fresh_lookup_matches_linear_scan_2936() {
    let state = make_four_entry_state();
    let expected: Vec<_> = state
        .entries
        .iter()
        .filter(|e| e.node_name == "relu0")
        .map(entry_summary)
        .collect();
    let actual: Vec<_> = state.entries_for_node("relu0").map(entry_summary).collect();
    assert_eq!(
        actual, expected,
        "fresh node lookup must match the old linear scan order"
    );
}

#[test]
fn test_entries_for_node_stale_index_falls_back_to_linear_scan_2936() {
    let mut state = make_four_entry_state();
    state
        .entries
        .push(make_entry_full("relu0", 2, 0.25, 0.5, -1.0));
    state
        .entries
        .push(make_entry_full("relu3", 0, 0.9, 0.0, 1.0));

    let expected: Vec<_> = state
        .entries
        .iter()
        .filter(|e| e.node_name == "relu0")
        .map(entry_summary)
        .collect();
    let actual: Vec<_> = state.entries_for_node("relu0").map(entry_summary).collect();
    assert_eq!(
        actual, expected,
        "stale node lookup must fall back to the old linear scan semantics"
    );
    assert!(state.has_node_entries("relu3"));
    assert_eq!(state.signed_beta("relu0", 2), Some(-0.25));
}

#[test]
fn test_entries_for_node_rebuild_preserves_order_2936() {
    let mut state = make_four_entry_state();
    state
        .entries
        .push(make_entry_full("relu0", 2, 0.25, 0.5, -1.0));
    state
        .entries
        .push(make_entry_full("relu3", 0, 0.9, 0.0, 1.0));

    let expected: Vec<_> = state
        .entries
        .iter()
        .filter(|e| e.node_name == "relu0")
        .map(entry_summary)
        .collect();

    // accumulate_grad triggers index rebuild
    state.accumulate_grad("relu0", 2, 1.25);
    let actual: Vec<_> = state.entries_for_node("relu0").map(entry_summary).collect();
    assert_eq!(
        actual, expected,
        "rebuilding node indexes must preserve the old linear scan order"
    );
    assert!((state.entries[4].grad() - 1.25).abs() < 1e-6);
}
