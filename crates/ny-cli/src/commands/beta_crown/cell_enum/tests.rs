// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the cell-enumeration driver: detection (qualifying and
//! non-qualifying) and aggregation semantics (indeterminate / partial coverage
//! can never yield unsat).

use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::layers::{AddConstantLayer, GatherLayer, SigmoidLayer, TruncLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};

use super::super::BetaCrownModel;
use super::*;

/// Spec: X_0 free in [0, 2], X_1 fixed 3.0; unsafe iff Y_0 <= threshold.
fn spec(threshold: f64) -> VnnLibSpec {
    let mut spec = VnnLibSpec::new();
    spec.num_inputs = 2;
    spec.num_outputs = 1;
    spec.input_bounds = vec![(0.0, 2.0), (3.0, 3.0)];
    spec.output_constraints = vec![OutputConstraint::LessEqConst(0, threshold)];
    spec.output_constraint_clauses = vec![vec![OutputConstraint::LessEqConst(0, threshold)]];
    spec.is_disjunction = true;
    spec
}

fn gather_indices(indices: &[i64]) -> GatherLayer {
    let arr = ArrayD::from_shape_vec(IxDyn(&[indices.len()]), indices.to_vec()).unwrap();
    GatherLayer::new(0, Some(arr), vec![indices.len()])
}

/// Qualifying graph: input[2] -> Gather[0] -> Trunc -> +10 -> output.
/// Y_0 = trunc(X_0) + 10, so Y_0 in {10, 11, 12} over the box.
fn trunc_gated_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "gather_x",
        Layer::Gather(gather_indices(&[0])),
    ));
    graph.add_node(GraphNode::new(
        "trunc_x",
        Layer::Trunc(TruncLayer),
        vec!["gather_x".to_string()],
    ));
    let bias = ArrayD::from_elem(IxDyn(&[1]), 10.0_f32);
    graph.add_node(GraphNode::new(
        "shift",
        Layer::AddConstant(AddConstantLayer::new(bias)),
        vec!["trunc_x".to_string()],
    ));
    graph.set_output("shift");
    graph
}

/// Non-qualifying graph: the free dim ALSO reaches the output without a Trunc.
fn leaky_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    // Gather selects the free dim, but its consumer is AddConstant, not Trunc.
    graph.add_node(GraphNode::from_input(
        "gather_x",
        Layer::Gather(gather_indices(&[0])),
    ));
    let bias = ArrayD::from_elem(IxDyn(&[1]), 10.0_f32);
    graph.add_node(GraphNode::new(
        "shift",
        Layer::AddConstant(AddConstantLayer::new(bias)),
        vec!["gather_x".to_string()],
    ));
    graph.set_output("shift");
    graph
}

/// Qualifying STRUCTURE but with an op the f64 cell evaluator does not
/// support after the trunc — every cell becomes indeterminate.
fn indeterminate_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "gather_x",
        Layer::Gather(gather_indices(&[0])),
    ));
    graph.add_node(GraphNode::new(
        "trunc_x",
        Layer::Trunc(TruncLayer),
        vec!["gather_x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer),
        vec!["trunc_x".to_string()],
    ));
    graph.set_output("sigmoid");
    graph
}

fn far_deadline() -> Instant {
    Instant::now() + Duration::from_mins(1)
}

/// `NY_NO_CELL_ENUM` is process-global; serialize every test that calls
/// `try_cell_enumeration` against the one that mutates it.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn qualifying_unsat_all_cells_safe() {
    let _env = ENV_LOCK.lock().unwrap();
    // Y_0 = trunc(X_0) + 10 >= 10 for all cells; unsafe iff Y_0 <= 0.5 -> UNSAT.
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let result = try_cell_enumeration(&model, &[2], &spec(0.5), far_deadline())
        .expect("qualifying spec must be decided");
    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "expected Verified, got {:?}",
        result.result
    );
    assert_eq!(result.domains_explored, 3, "3 trunc cells for [0, 2]");
}

#[test]
fn qualifying_sat_finds_witness_cell() {
    let _env = ENV_LOCK.lock().unwrap();
    // Unsafe iff Y_0 <= 10.5: cell v=0 gives Y_0 = 10 <= 10.5 -> SAT with witness.
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let result = try_cell_enumeration(&model, &[2], &spec(10.5), far_deadline())
        .expect("qualifying spec must be decided");
    match result.result {
        BabVerificationStatus::Violated {
            counterexample,
            output,
        } => {
            assert_eq!(counterexample.len(), 2);
            // The witness must lie in the violating cell [0, 1) and in the box.
            assert!(counterexample[0] >= 0.0 && counterexample[0] < 1.0);
            assert_eq!(counterexample[1], 3.0);
            assert_eq!(output.len(), 1);
            assert!(output[0] <= 10.5);
        }
        other => panic!("expected Violated, got {other:?}"),
    }
}

#[test]
fn non_qualifying_free_dim_not_trunc_gated_falls_through() {
    let _env = ENV_LOCK.lock().unwrap();
    // The free dim reaches the output without a Trunc: MUST fall through
    // (evaluating a representative would NOT cover the whole cell).
    let model = BetaCrownModel::Graph(Box::new(leaky_graph()));
    assert!(
        try_cell_enumeration(&model, &[2], &spec(0.5), far_deadline()).is_none(),
        "non-trunc-gated free dim must fall through to the normal pipeline"
    );
}

#[test]
fn negative_free_range_falls_through() {
    let _env = ENV_LOCK.lock().unwrap();
    // lo < 0: trunc cells are not [v, v+1) there; detection must fail closed.
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let mut negative = spec(0.5);
    negative.input_bounds[0] = (-1.0, 2.0);
    assert!(try_cell_enumeration(&model, &[2], &negative, far_deadline()).is_none());
}

#[test]
fn indeterminate_cells_never_yield_unsat() {
    let _env = ENV_LOCK.lock().unwrap();
    // The f64 evaluator cannot decide any cell (unsupported Sigmoid): the
    // driver must fall through (None), NEVER report Verified.
    let model = BetaCrownModel::Graph(Box::new(indeterminate_graph()));
    assert!(
        try_cell_enumeration(&model, &[2], &spec(0.5), far_deadline()).is_none(),
        "indeterminate cells must not produce a verdict"
    );
}

#[test]
fn expired_deadline_reports_timeout_not_unsat() {
    let _env = ENV_LOCK.lock().unwrap();
    // Partial coverage (deadline already passed): Timeout, never Verified.
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let result = try_cell_enumeration(&model, &[2], &spec(0.5), expired)
        .expect("expired deadline still yields an explicit Timeout");
    assert!(
        matches!(result.result, BabVerificationStatus::Timeout),
        "partial coverage must report Timeout, got {:?}",
        result.result
    );
}

#[test]
fn cell_budget_overflow_falls_through() {
    let _env = ENV_LOCK.lock().unwrap();
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let mut huge = spec(0.5);
    huge.input_bounds[0] = (0.0, 1.0e9);
    assert!(try_cell_enumeration(&model, &[2], &huge, far_deadline()).is_none());
}

#[test]
fn disable_flag_falls_through() {
    let _env = ENV_LOCK.lock().unwrap();
    // NY_NO_CELL_ENUM=1 disables the driver entirely. (Serialized + restored
    // via the blessed env choke point — clippy env wall.)
    let result = ny_test_utils::env::with_serialized_env_vars(&[("NY_NO_CELL_ENUM", "1")], || {
        let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
        try_cell_enumeration(&model, &[2], &spec(0.5), far_deadline())
    });
    assert!(result.is_none(), "disable flag must fall through");
}

#[test]
fn cell_representative_prefers_midpoint_and_respects_box() {
    assert_eq!(cell_representative(0.0, 0.0, 62.0), Some(0.5));
    assert_eq!(cell_representative(62.0, 0.0, 62.0), Some(62.0));
    // Clipped cell [0.7, 1): midpoint 0.5 is outside the box; v=0 too; falls
    // back to lo = 0.7 (still trunc == 0).
    assert_eq!(cell_representative(0.0, 0.7, 2.0), Some(0.7));
    // Degenerate: no representative in [2, 3) ∩ [0.0, 1.5].
    assert_eq!(cell_representative(2.0, 0.0, 1.5), None);
}

#[test]
fn box_checks_respect_clause_semantics() {
    // Disjunctive unsafe (Y0 <= 0.5) OR (Y0 >= 2.0), box [1.0, 1.5]: safe.
    let mut spec = VnnLibSpec::new();
    spec.num_outputs = 1;
    spec.is_disjunction = true;
    spec.output_constraint_clauses = vec![
        vec![OutputConstraint::LessEqConst(0, 0.5)],
        vec![OutputConstraint::GreaterEqConst(0, 2.0)],
    ];
    assert!(box_definitely_safe(&[1.0], &[1.5], &spec));
    assert!(!box_definitely_violated(&[1.0], &[1.5], &spec));
    // Box [0.2, 0.4] lies inside the first clause: definitely violated.
    assert!(box_definitely_violated(&[0.2], &[0.4], &spec));
    assert!(!box_definitely_safe(&[0.2], &[0.4], &spec));
    // Straddling box [0.4, 1.0]: neither.
    assert!(!box_definitely_safe(&[0.4], &[1.0], &spec));
    assert!(!box_definitely_violated(&[0.4], &[1.0], &spec));
}
