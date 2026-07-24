// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Assert properties of a split in a GeneralSplitHistory.
fn assert_general_split(
    general: &GeneralSplitHistory,
    idx: usize,
    layer_name: &str,
    has_lower: bool,
    has_upper: bool,
    branching_point: Option<f32>,
    expected_score: f32,
    label: &str,
) {
    let split = &general.splits()[idx];
    assert!(
        matches!(split.layer(), LayerRef::Name(name) if name == layer_name),
        "{label} should reference {layer_name}"
    );
    assert_eq!(
        split.lower_bound().is_some(),
        has_lower,
        "{label} lower_bound"
    );
    assert_eq!(
        split.upper_bound().is_some(),
        has_upper,
        "{label} upper_bound"
    );
    assert_eq!(split.branching_point(), branching_point);
    assert!(
        (split.score() - expected_score).abs() < 1e-6,
        "{label} score: expected {expected_score}, got {}",
        split.score()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_genbab_constraint_creation() {
    let constraint =
        GenBabConstraint::new("gelu_1".to_string(), 3, -0.5, true, 0.5).expect("finite constraint");
    assert_eq!(constraint.node_name, "gelu_1");
    assert_eq!(constraint.neuron_idx, 3);
    assert_eq!(constraint.split_point, -0.5);
    assert!(constraint.is_upper_branch, "is_upper_branch should be true");
    assert!(
        (constraint.score - 0.5).abs() < 1e-6,
        "score: expected 0.5, got {}",
        constraint.score
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_genbab_constraint_to_graph_neuron_constraint() {
    // Upper branch (x >= -0.5) → is_active = true
    let upper =
        GenBabConstraint::new("node".to_string(), 0, -0.5, true, 1.0).expect("finite constraint");
    let converted = upper
        .to_graph_neuron_constraint()
        .expect("valid conversion");
    assert!(
        converted.is_active(),
        "upper branch should convert to is_active=true"
    );
    assert_eq!(converted.node_name(), "node");
    assert_eq!(converted.neuron_idx(), 0);
    assert!(
        (converted.score() - 1.0).abs() < 1e-6,
        "upper branch score: expected 1.0, got {}",
        converted.score()
    );

    // Lower branch (x <= -0.5) → is_active = false
    let lower = GenBabConstraint::new("node".to_string(), 0, -0.5, false, -0.25)
        .expect("finite constraint");
    let converted = lower
        .to_graph_neuron_constraint()
        .expect("valid conversion");
    assert!(
        !converted.is_active(),
        "lower branch should convert to is_active=false"
    );
    assert!(
        (converted.score() + 0.25).abs() < 1e-6,
        "lower branch score: expected -0.25, got {}",
        converted.score()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_constraint_enum() {
    // ReLU constraint
    let relu = GraphConstraint::Relu(GraphNeuronConstraint {
        node_name: "relu_1".to_string(),
        neuron_idx: 5,
        is_active: true,
        score: 0.0,
    });
    assert_eq!(relu.node_name(), "relu_1");
    assert_eq!(relu.neuron_idx(), 5);
    assert_eq!(relu.beta_sign(), 1.0);
    assert!(
        relu.is_upper_branch(),
        "ReLU active constraint should be upper branch"
    );
    assert_eq!(relu.split_point(), 0.0);

    // GenBaB constraint
    let genbab = GraphConstraint::GenBab(
        GenBabConstraint::new("gelu_2".to_string(), 10, -0.3, false, 0.0)
            .expect("finite constraint"),
    );
    assert_eq!(genbab.node_name(), "gelu_2");
    assert_eq!(genbab.neuron_idx(), 10);
    assert_eq!(genbab.beta_sign(), -1.0);
    assert!(
        !genbab.is_upper_branch(),
        "GenBaB is_upper_branch=false should remain false"
    );
    assert_eq!(genbab.split_point(), -0.3);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_split_history_with_genbab() {
    let mut history = GraphSplitHistory::new();

    // Add ReLU constraint
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    // Add GenBaB constraint at non-zero split point
    history.add_genbab_constraint(
        GenBabConstraint::new("gelu_1".to_string(), 5, -0.5, true, 0.0).expect("finite constraint"),
    );

    // Check depth includes both
    assert_eq!(history.depth(), 2);

    // Check ReLU constraint lookup
    assert_eq!(history.is_constrained("relu_1", 0), Some(true));
    assert_eq!(history.is_constrained("gelu_1", 5), None); // Not a ReLU constraint

    // Check GenBaB constraint lookup
    assert_eq!(
        history.is_genbab_constrained("gelu_1", 5),
        Some((Some(-0.5), None))
    );
    assert_eq!(history.is_genbab_constrained("relu_1", 0), None); // Not a GenBaB constraint
}

#[ntest::timeout(10000)]
#[test]
fn test_last_branch_node_mixed_history() {
    // Empty history → None
    let history = GraphSplitHistory::new();
    assert_eq!(history.last_branch_node(), None);

    // ReLU only → last ReLU node
    let mut history = GraphSplitHistory::new();
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    assert_eq!(history.last_branch_node(), Some("relu_1"));

    // ReLU then GenBaB → GenBaB is most recent
    history.add_genbab_constraint(
        GenBabConstraint::new("gelu_1".to_string(), 5, -0.5, true, 0.0).expect("finite constraint"),
    );
    assert_eq!(history.last_branch_node(), Some("gelu_1"));

    // GenBaB then ReLU → ReLU is most recent
    let mut history2 = GraphSplitHistory::new();
    history2.add_genbab_constraint(
        GenBabConstraint::new("sigmoid_1".to_string(), 0, 0.5, false, 0.0)
            .expect("finite constraint"),
    );
    assert_eq!(history2.last_branch_node(), Some("sigmoid_1"));
    history2.add_constraint(GraphNeuronConstraint {
        node_name: "relu_2".to_string(),
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });
    assert_eq!(history2.last_branch_node(), Some("relu_2"));
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_split_history_iter_all() {
    let mut history = GraphSplitHistory::new();

    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    history.add_genbab_constraint(
        GenBabConstraint::new("gelu_1".to_string(), 1, -0.5, false, 0.0)
            .expect("finite constraint"),
    );

    let all: Vec<_> = history.iter_all().collect();
    assert_eq!(all.len(), 2);

    // First is ReLU
    match &all[0] {
        GraphConstraint::Relu(c) => {
            assert_eq!(c.node_name, "relu_1");
            assert!(c.is_active, "first iter_all entry (ReLU) should be active");
        }
        _ => panic!("Expected ReLU constraint first"),
    }

    // Second is GenBaB
    match &all[1] {
        GraphConstraint::GenBab(c) => {
            assert_eq!(c.node_name, "gelu_1");
            assert_eq!(c.split_point, -0.5);
            assert!(
                !c.is_upper_branch,
                "second iter_all entry (GenBaB) should not be upper branch"
            );
        }
        _ => panic!("Expected GenBaB constraint second"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_split_history_with_genbab_constraint() {
    let history = GraphSplitHistory::new();

    let constraint = GenBabConstraint::new("sigmoid_1".to_string(), 2, 0.5, true, 0.75)
        .expect("finite constraint");
    let new_history = history.with_genbab_constraint(constraint);

    assert_eq!(new_history.depth(), 1);
    assert!(
        new_history.has_genbab_constraints(),
        "history should have GenBaB constraints after adding one"
    );
    assert_eq!(
        new_history.is_genbab_constrained("sigmoid_1", 2),
        Some((Some(0.5), None))
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_split_history_to_graph_preserves_scores() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint::new(0, 1, true, 0.42).unwrap());
    history.add_constraint(NeuronConstraint::new(2, 3, false, -0.25).unwrap());

    let graph_history = history.to_graph_split_history().expect("valid history");
    assert_eq!(graph_history.constraints.len(), 2);

    let first = &graph_history.constraints[0];
    assert_eq!(first.node_name(), "layer_0");
    assert_eq!(first.neuron_idx(), 1);
    assert!(first.is_active(), "first constraint should be active");
    assert!(
        (first.score() - 0.42).abs() < 1e-6,
        "first score: expected 0.42, got {}",
        first.score()
    );

    let second = &graph_history.constraints[1];
    assert_eq!(second.node_name(), "layer_2");
    assert_eq!(second.neuron_idx(), 3);
    assert!(
        !second.is_active(),
        "second constraint should not be active"
    );
    assert!(
        (second.score() + 0.25).abs() < 1e-6,
        "second score: expected -0.25, got {}",
        second.score()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_general_split_history_from_graph_preserves_scores() {
    let mut graph_history = GraphSplitHistory::new();
    graph_history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_0".to_string(),
        neuron_idx: 1,
        is_active: true,
        score: 0.42,
    });
    graph_history.add_constraint(GraphNeuronConstraint {
        node_name: "relu_1".to_string(),
        neuron_idx: 2,
        is_active: false,
        score: -0.1,
    });
    graph_history.add_genbab_constraint(
        GenBabConstraint::new("gelu_1".to_string(), 3, -0.5, true, 0.77)
            .expect("finite constraint"),
    );

    let general = GeneralSplitHistory::from_graph_history(&graph_history).expect("valid history");
    assert_eq!(general.splits().len(), 3);

    assert_general_split(
        &general,
        0,
        "relu_0",
        true,
        false,
        Some(0.0),
        0.42,
        "active relu",
    );
    assert_general_split(
        &general,
        1,
        "relu_1",
        false,
        true,
        Some(0.0),
        -0.1,
        "inactive relu",
    );
    assert_general_split(
        &general,
        2,
        "gelu_1",
        true,
        false,
        Some(-0.5),
        0.77,
        "upper genbab",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_general_split_history_merges_genbab_range_split() {
    let mut graph_history = GraphSplitHistory::new();
    graph_history.add_genbab_constraints_for_split([
        GenBabConstraint::new("gelu_1".to_string(), 4, -0.25, true, 0.2)
            .expect("finite constraint"),
        GenBabConstraint::new("gelu_1".to_string(), 4, 0.75, false, 0.9)
            .expect("finite constraint"),
    ]);

    let general = GeneralSplitHistory::from_graph_history(&graph_history).expect("valid history");
    assert_eq!(general.splits().len(), 1);

    let split = &general.splits()[0];
    assert!(
        matches!(split.layer(), LayerRef::Name(name) if name == "gelu_1"),
        "merged range split should reference gelu_1"
    );
    assert_eq!(split.lower_bound(), Some(-0.25));
    assert_eq!(split.upper_bound(), Some(0.75));
    assert_eq!(split.branching_point(), None);
    assert!(
        (split.score() - 0.9).abs() < 1e-6,
        "merged range split score: expected 0.9, got {}",
        split.score()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_general_split_history_does_not_merge_different_split_ids() {
    let mut graph_history = GraphSplitHistory::new();
    graph_history.add_genbab_constraint(
        GenBabConstraint::new("gelu_1".to_string(), 4, -0.25, true, 0.2)
            .expect("finite constraint"),
    );
    graph_history.add_genbab_constraint(
        GenBabConstraint::new("gelu_1".to_string(), 4, 0.75, false, 0.9)
            .expect("finite constraint"),
    );

    let general = GeneralSplitHistory::from_graph_history(&graph_history).expect("valid history");
    assert_eq!(general.splits().len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_general_split_history_does_not_merge_different_neurons() {
    let mut graph_history = GraphSplitHistory::new();
    graph_history.add_genbab_constraints_for_split([
        GenBabConstraint::new("gelu_1".to_string(), 4, -0.25, true, 0.2)
            .expect("finite constraint"),
        GenBabConstraint::new("gelu_1".to_string(), 5, 0.75, false, 0.9)
            .expect("finite constraint"),
    ]);

    let general = GeneralSplitHistory::from_graph_history(&graph_history).expect("valid history");
    assert_eq!(general.splits().len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_nonzero_split_point_regression() {
    // Regression test: ensure non-zero split points are preserved correctly
    // This tests the fix for #205 (GenBaB splits were approximated as ReLU constraints)

    let mut history = GraphSplitHistory::new();

    // Add a GeLU split at -0.5 (the GeLU inflection region)
    let gelu_split =
        GenBabConstraint::new("gelu_1".to_string(), 3, -0.5, true, 1.2).expect("finite constraint");
    history.add_genbab_constraint(gelu_split);

    // Verify the split point is preserved (not approximated to 0)
    let (lower, upper) = history
        .is_genbab_constrained("gelu_1", 3)
        .expect("finite constraint");
    assert_eq!(
        lower,
        Some(-0.5),
        "Split point should be preserved as -0.5, not approximated"
    );
    assert!(upper.is_none(), "Upper branch flag should be preserved");

    // Add a sigmoid split at 0.0 (should still work as GenBaB, not converted to ReLU)
    let sigmoid_split = GenBabConstraint::new("sigmoid_1".to_string(), 0, 0.0, false, -0.3)
        .expect("finite constraint");
    history.add_genbab_constraint(sigmoid_split);

    let (lower, upper) = history
        .is_genbab_constrained("sigmoid_1", 0)
        .expect("finite constraint");
    assert!(
        lower.is_none(),
        "sigmoid lower branch should have lower=None"
    );
    assert_eq!(upper, Some(0.0));

    // Verify total count
    assert_eq!(history.genbab_constraints.len(), 2);
    assert_eq!(history.constraints.len(), 0); // No ReLU constraints
}

#[ntest::timeout(10000)]
#[test]
fn test_genbab_constraint_rejects_nan_split_point() {
    let result = GenBabConstraint::new("gelu_1".to_string(), 0, f32::NAN, true, 0.5);
    assert!(result.is_err(), "NaN split_point must be rejected");

    let result = GenBabConstraint::new("gelu_1".to_string(), 0, f32::INFINITY, true, 0.5);
    assert!(result.is_err(), "Inf split_point must be rejected");

    let result = GenBabConstraint::new("gelu_1".to_string(), 0, f32::NEG_INFINITY, true, 0.5);
    assert!(result.is_err(), "NEG_INFINITY split_point must be rejected");

    // Finite values should succeed
    let result = GenBabConstraint::new("gelu_1".to_string(), 0, 0.0, true, 0.5);
    assert!(result.is_ok(), "Finite split_point must be accepted");
}

/// Verify that pub(crate) accessor methods return the same values passed to new().
/// Fields are pub(crate) to prevent struct literal construction that bypasses NaN
/// validation in new(). (#3017)
#[ntest::timeout(10000)]
#[test]
fn test_genbab_constraint_accessors_3017() {
    let c = GenBabConstraint::new("sigmoid_0".to_string(), 7, -1.5, false, 2.0)
        .expect("finite constraint");
    assert_eq!(c.node_name(), "sigmoid_0");
    assert_eq!(c.neuron_idx(), 7);
    assert_eq!(c.split_point(), -1.5);
    assert!(
        !c.is_upper_branch(),
        "is_upper_branch() should return false"
    );
    assert!(
        (c.score() - 2.0).abs() < f32::EPSILON,
        "score() should return 2.0, got {}",
        c.score()
    );
}

/// Verify that NaN/Inf scores are rejected by GenBabConstraint::new(). (#3017)
#[ntest::timeout(10000)]
#[test]
fn test_genbab_constraint_rejects_nan_score_3017() {
    let nan_score = GenBabConstraint::new("node".to_string(), 0, 0.0, true, f32::NAN);
    assert!(nan_score.is_err(), "NaN score must be rejected");

    let inf_score = GenBabConstraint::new("node".to_string(), 0, 0.0, true, f32::INFINITY);
    assert!(inf_score.is_err(), "Inf score must be rejected");

    let neg_inf_score = GenBabConstraint::new("node".to_string(), 0, 0.0, true, f32::NEG_INFINITY);
    assert!(
        neg_inf_score.is_err(),
        "NEG_INFINITY score must be rejected"
    );

    // Finite score should succeed
    let ok = GenBabConstraint::new("node".to_string(), 0, 0.0, true, 1.5);
    assert!(ok.is_ok(), "Finite score must be accepted");
}

/// Verify that NaN/Inf point values are rejected by NeuronSplit::at_point(). (#3017)
#[ntest::timeout(10000)]
#[test]
fn test_neuron_split_rejects_nan_point_3017() {
    let nan_point = NeuronSplit::at_point(LayerRef::Index(0), 0, f32::NAN, true);
    assert!(nan_point.is_err(), "NaN point must be rejected");

    let inf_point = NeuronSplit::at_point(LayerRef::Index(0), 0, f32::INFINITY, false);
    assert!(inf_point.is_err(), "Inf point must be rejected");

    // Finite point should succeed
    let ok = NeuronSplit::at_point(LayerRef::Index(0), 0, -0.5, true);
    assert!(ok.is_ok(), "Finite point must be accepted");
}
