// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the cell-enumeration driver: detection (qualifying and
//! non-qualifying) and aggregation semantics (indeterminate / partial coverage
//! can never yield unsat).

use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::layers::{AddConstantLayer, GatherLayer, TanhLayer, TruncLayer};
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
        Layer::Trunc(TruncLayer::new()),
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
        Layer::Trunc(TruncLayer::new()),
        vec!["gather_x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "tanh",
        Layer::Tanh(TanhLayer),
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

fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    // A failed assertion must not turn every later test into a misleading
    // poisoned-lock failure. The environment restoration guard has already
    // run during unwinding, so recovering the test-only mutex is safe.
    ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn qualifying_unsat_all_cells_safe() {
    let _env = lock_env();
    // Y_0 = trunc(X_0) + 10 >= 10 for all cells; unsafe iff Y_0 <= 0.5 -> UNSAT.
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let result = try_cell_enumeration(&model, &[2], &spec(0.5), Some(far_deadline()))
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
    let _env = lock_env();
    // Unsafe iff Y_0 <= 10.5: cell v=0 gives Y_0 = 10 <= 10.5 -> SAT with witness.
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let result = try_cell_enumeration(&model, &[2], &spec(10.5), Some(far_deadline()))
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
    let _env = lock_env();
    // The free dim reaches the output without a Trunc: MUST fall through
    // (evaluating a representative would NOT cover the whole cell).
    let model = BetaCrownModel::Graph(Box::new(leaky_graph()));
    assert!(
        try_cell_enumeration(&model, &[2], &spec(0.5), Some(far_deadline())).is_none(),
        "non-trunc-gated free dim must fall through to the normal pipeline"
    );
}

#[test]
fn negative_free_range_falls_through() {
    let _env = lock_env();
    // lo < 0: trunc cells are not [v, v+1) there; detection must fail closed.
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let mut negative = spec(0.5);
    negative.input_bounds[0] = (-1.0, 2.0);
    assert!(try_cell_enumeration(&model, &[2], &negative, Some(far_deadline())).is_none());
}

#[test]
fn indeterminate_cells_never_yield_unsat() {
    let _env = lock_env();
    // The f64 evaluator cannot decide any cell (unsupported Tanh): the
    // driver must fall through (None), NEVER report Verified.
    let model = BetaCrownModel::Graph(Box::new(indeterminate_graph()));
    assert!(
        try_cell_enumeration(&model, &[2], &spec(0.5), Some(far_deadline())).is_none(),
        "indeterminate cells must not produce a verdict"
    );
}

#[test]
fn expired_deadline_reports_timeout_not_unsat() {
    let _env = lock_env();
    // Partial coverage (deadline already passed): Timeout, never Verified.
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let result = try_cell_enumeration(&model, &[2], &spec(0.5), Some(expired))
        .expect("expired deadline still yields an explicit Timeout");
    assert!(
        matches!(result.result, BabVerificationStatus::Timeout),
        "partial coverage must report Timeout, got {:?}",
        result.result
    );
}

#[test]
fn cell_budget_overflow_falls_through() {
    let _env = lock_env();
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let mut huge = spec(0.5);
    huge.input_bounds[0] = (0.0, 1.0e9);
    assert!(try_cell_enumeration(&model, &[2], &huge, Some(far_deadline())).is_none());
}

#[test]
fn disable_flag_falls_through() {
    let _env = lock_env();
    // NY_NO_CELL_ENUM=1 disables the driver entirely. (Serialized + restored
    // via the blessed env choke point — clippy env wall.)
    let result = ny_test_utils::env::with_serialized_env_vars(&[("NY_NO_CELL_ENUM", "1")], || {
        let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
        try_cell_enumeration(&model, &[2], &spec(0.5), Some(far_deadline()))
    });
    assert!(result.is_none(), "disable flag must fall through");
}

#[test]
fn unbounded_deadline_evaluates_the_complete_cell_cover() {
    let _env = lock_env();
    let model = BetaCrownModel::Graph(Box::new(trunc_gated_graph()));
    let result = try_cell_enumeration(&model, &[2], &spec(0.5), None)
        .expect("finite cell cover should complete without a deadline");
    assert!(matches!(result.result, BabVerificationStatus::Verified));
    assert_eq!(result.domains_explored, 3);
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

/// #cctsdb-witness-f32. `44d9e46d` required the f32 witness coordinate to
/// decode back INSIDE the exact f64 declared interval. For cctsdb_yolo_2023
/// every one of the 12288 pixels is pinned by an equality pair on an 8-digit
/// `k/255` decimal, and 45 of its 115 distinct literals have NO f32 inside
/// them — so that rule refused 100% of the category's rows before any forward
/// ran, costing all 28 banked sats.
#[test]
fn witness_f32_collapses_a_declared_point_with_no_f32_preimage() {
    // 50/255 as printed by the cctsdb spec generator. Neither neighbouring f32
    // decodes back into the degenerate interval [v, v].
    let v = 0.19607843_f64;
    assert_ne!(f32_to_f64_exact(f64_to_f32_down(v)), v);
    assert_ne!(f32_to_f64_exact(f64_to_f32_up(v)), v);
    assert!(f32_to_f64_exact(f64_to_f32_down(v)) < v);
    assert!(f32_to_f64_exact(f64_to_f32_up(v)) > v);

    // Declared point: collapse to the nearest f32 (what the f32 network is fed
    // regardless, and what `#witness-snap-declared` restores before emission).
    assert_eq!(witness_f32(v, v, v), Some(v as f32));
}

#[test]
fn witness_f32_prefers_a_contained_candidate_and_still_fails_closed_on_a_real_box() {
    // Exactly representable point: containment succeeds, no collapse needed.
    assert_eq!(witness_f32(0.5, 0.5, 0.5), Some(0.5_f32));
    // Wide box: the contained candidate is returned.
    let contained = witness_f32(0.19607843, 0.0, 1.0).expect("f32 inside [0, 1]");
    let decoded = f32_to_f64_exact(contained);
    assert!((0.0..=1.0).contains(&decoded));
    // NON-degenerate box too narrow to hold any f32 still fails closed: the
    // collapse is authorised for declared POINTS only.
    let lo = 0.19607843_f64;
    let hi = f32_to_f64_exact(f64_to_f32_up(lo)) - f64::EPSILON * 0.25;
    assert!(lo < hi, "constructed a non-degenerate sub-ULP box");
    assert_eq!(witness_f32(lo, lo, hi), None);
}

/// The free-dim cell-drift check is load-bearing (the f64 forward certified
/// the integer cell of the representative) and is NOT relaxed by the repair.
#[test]
fn witness_f32_free_dim_keeps_its_cell_and_the_strict_switch_restores_the_old_refusal() {
    // Free-dim representatives are v + 0.5 — always exact in f32.
    assert_eq!(witness_f32(31.5, 0.0, 62.0), Some(31.5_f32));
    assert_eq!(f32_to_f64_exact(31.5_f32).trunc(), 31.5_f64.trunc());

    // The kill switch reproduces the pre-repair (broken) refusal verbatim, and
    // the repaired arm recovers the witness — the whole 28-row delta, pinned.
    let v = 0.19607843_f64;
    assert_eq!(
        witness_f32_with(v, v, v, true),
        None,
        "strict arm must reproduce 44d9e46d"
    );
    assert_eq!(witness_f32_with(v, v, v, false), Some(v as f32));
}

#[test]
fn malformed_output_boxes_never_prove_a_cell_verdict() {
    let spec = spec(0.5);
    let malformed = [
        (&[][..], &[][..]),
        (&[1.0][..], &[][..]),
        (&[f64::NAN][..], &[2.0][..]),
        (&[1.0][..], &[f64::INFINITY][..]),
        (&[2.0][..], &[1.0][..]),
    ];
    for (lower, upper) in malformed {
        assert!(
            !box_definitely_safe(lower, upper, &spec),
            "malformed box {lower:?}..{upper:?} must not prove safety"
        );
        assert!(
            !box_definitely_violated(lower, upper, &spec),
            "malformed box {lower:?}..{upper:?} must not prove violation"
        );
    }
}

#[test]
fn nonfinite_output_constant_never_proves_a_cell_verdict() {
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let spec = spec(threshold);
        assert!(!box_definitely_safe(&[1.0], &[2.0], &spec));
        assert!(!box_definitely_violated(&[1.0], &[2.0], &spec));
        assert_eq!(scalar_box_safety_directions(&spec), (false, false));
    }
}

#[test]
fn scalar_safety_direction_is_structural_not_an_infinite_box_probe() {
    let less = spec(0.5);
    assert_eq!(scalar_box_safety_directions(&less), (true, false));

    let mut greater = spec(0.5);
    greater.output_constraints = vec![OutputConstraint::GreaterEqConst(0, 0.5)];
    greater.output_constraint_clauses = greater
        .output_constraints
        .iter()
        .cloned()
        .map(|constraint| vec![constraint])
        .collect();
    assert_eq!(scalar_box_safety_directions(&greater), (false, true));
}
