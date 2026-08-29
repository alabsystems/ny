// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Facade contract for the complete-MIP surface.
//!
//! Everything here is reached through `ny_api::complete::*` (plus
//! `ny_api::cert` for the Farkas checker) — there is no `ny_mip` import in this
//! file, which is exactly the property an embedded consumer needs. The
//! assertions are CONSTRUCTIVE: each export is built and driven, so a
//! re-export that survived as a name but lost the type it needs to be usable
//! still fails here.

#![allow(unused_qualifications)]

use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, ArrayD, IxDyn};

use ny_api::cert::check_farkas;
use ny_api::complete::{
    encode_feedforward, from_milp_text, obbt_relaxation_bounds, prove_infeasible_with_row_farkas,
    star_search_unsafe_conjunction, star_verify, to_milp_text, to_smtlib_decision,
    to_smtlib_minimize, verify_graph_complete_mip, Col, ColSpec, CompleteMipOptions,
    CompleteMipReport, FcReluChain, MilpProblem, MipBackend, MipConfig, MipEncoder,
    MipFeasibilityIngress, MipParts, MipResult, MipSolver, RelaxationObbt, Row, RowSide, RowSpec,
    Sense, SplitUnsatCache, Star, StarBudget, StarLayer, StarSpec, StarStats,
    StarUnsafeCandidateVerdict, StarUnsafeConjunction, StarUnsafeRow, StarVerdict,
    DEFAULT_PREACTIVATION_INFLATION,
};
use ny_api::graph::{GraphNetwork, GraphNode};
use ny_api::layers::{Layer, LinearLayer, ReLULayer};
use ny_api::{Bound, VerificationSpec};

/// `y = relu(x) + relu(-x) = |x|` as a `Linear -> ReLU -> Linear` chain.
fn abs_encoder() -> MipEncoder {
    encode_feedforward(
        &[vec![1.0, -1.0], vec![1.0, 1.0]],
        &[vec![0.0, 0.0], vec![0.0]],
        &[1, 2, 1],
        &[Bound::new(-1.0, 1.0)],
        &[vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)]],
    )
    .expect("a feed-forward ReLU net is encodable")
}

/// A tiny infeasible LP: `x >= 1` and `x <= 0`, no integer columns.
fn infeasible_lp() -> MilpProblem {
    let mut problem = MilpProblem::new();
    let x: Col = problem.add_col(0.0, -10.0, 10.0);
    let _lower: Row = problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);
    let _upper: Row = problem.add_row(f64::NEG_INFINITY, 0.0, [(x, 1.0)]);
    problem
}

#[test]
fn mip_config_field_types_are_settable_through_the_facade() {
    // The audit's exact complaint: `MipConfig` was exported while the types of
    // two of its fields were not, so neither knob could be set from outside.
    let config = MipConfig {
        backend: MipBackend::Ay,
        feasibility_ingress: MipFeasibilityIngress::Historical,
        timeout_secs: 15.0,
        parallel_split: 1,
        ..MipConfig::default()
    };
    assert_eq!(config.backend, MipBackend::Ay);
    assert_eq!(
        config.feasibility_ingress,
        MipFeasibilityIngress::Historical
    );
    assert_ne!(MipBackend::Ay, MipBackend::AyProc);
    assert_ne!(
        MipFeasibilityIngress::Historical,
        MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix
    );
    assert_ne!(
        MipFeasibilityIngress::RequireSafeNlpSharedBinaryPrefix,
        MipFeasibilityIngress::RequireSafeNlpMarkedMarginSharedBinaryPrefix
    );
}

#[test]
fn encode_and_solve_round_trip_through_the_facade() {
    // |x| <= 1 on [-1, 1], so `y >= 1.5` must be infeasible.
    let mut encoder = abs_encoder();
    encoder
        .constrain_output_geq_const(0, 1.5)
        .expect("output index 0 exists");
    assert_eq!(encoder.output_vars().len(), 1);

    let parts: MipParts = encoder.into_parts();
    assert_eq!(parts.input_vars.len(), 1);
    assert_eq!(parts.output_vars.len(), 1);
    assert_eq!(parts.binary_vars.len(), parts.binary_widths.len());
    assert!(parts.num_cols >= parts.input_vars.len() + parts.output_vars.len());

    let solver = MipSolver::new(
        parts,
        MipConfig {
            timeout_secs: 30.0,
            ..MipConfig::default()
        },
    );
    assert_eq!(solver.num_inputs(), 1);
    assert_eq!(solver.num_outputs(), 1);
    match solver.check_feasibility().expect("solve runs") {
        MipResult::Unsat { certified } => {
            let _: bool = certified;
        }
        other => panic!("|x| never reaches 1.5 on [-1, 1]: {other:?}"),
    }
}

#[test]
fn the_ir_is_a_first_class_facade_value() {
    let problem = infeasible_lp();
    assert_eq!(problem.num_cols(), 1);
    assert_eq!(problem.num_rows(), 2);

    let cols: &[ColSpec] = problem.cols();
    assert_eq!(cols[0].lb, -10.0);
    assert!(!cols[0].integer);
    let rows: &[RowSpec] = problem.rows();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].coeffs.len(), 1);

    // The dump lane round-trips the IR through text.
    let text = to_milp_text(&problem);
    let restored = from_milp_text(&text).expect("the dump round-trips");
    assert_eq!(restored.num_cols(), problem.num_cols());
    assert_eq!(restored.num_rows(), problem.num_rows());
}

#[test]
fn smtlib_export_lane_emits_both_forms() {
    let problem = infeasible_lp();

    let decision = to_smtlib_decision(&problem).expect("decision form emits");
    assert!(
        decision.contains("check-sat"),
        "the decision form is a check-sat query: {decision}"
    );

    let minimize = to_smtlib_minimize(&problem, Col(0)).expect("minimize form emits");
    assert!(
        minimize.contains("minimize"),
        "the optimization form carries the objective directive: {minimize}"
    );
    assert_ne!(
        decision, minimize,
        "the two forms must not collapse to the same byte stream"
    );
}

#[test]
fn obbt_tightening_lane_runs_and_never_widens() {
    // `x in [-10, 10]` with rows pinning it to `[1, 3]`.
    let mut problem = MilpProblem::new();
    let x = problem.add_col(0.0, -10.0, 10.0);
    problem.add_row(1.0, f64::INFINITY, [(x, 1.0)]);
    problem.add_row(f64::NEG_INFINITY, 3.0, [(x, 1.0)]);

    let obbt: RelaxationObbt = obbt_relaxation_bounds(
        &problem,
        &[x],
        2,
        Duration::from_secs(5),
        Instant::now() + Duration::from_secs(30),
        1,
    )
    .expect("OBBT runs on a continuous relaxation");

    assert!(!obbt.infeasible, "the LP is feasible");
    let (lower, upper) = obbt.bounds[0];
    assert!(
        lower >= -10.0 && upper <= 10.0,
        "OBBT may only tighten, never widen: [{lower}, {upper}]"
    );
    assert!(
        lower <= 1.0 && upper >= 3.0,
        "a rigorous bound must still enclose the true range [1, 3]: [{lower}, {upper}]"
    );
    assert!(obbt.rounds >= 1);
}

#[test]
fn farkas_producer_and_checker_are_both_on_the_facade() {
    // The producer: exact-rational infeasibility evidence for an LP.
    let evidence = prove_infeasible_with_row_farkas(&infeasible_lp(), 30.0)
        .expect("the Farkas lane runs")
        .expect("`x >= 1 and x <= 0` is infeasible, so evidence exists");
    assert!(
        evidence.len() >= 2,
        "both conflicting rows must appear in the certificate: {evidence:?}"
    );
    for (row, side, _multiplier) in &evidence {
        assert!(*row < 2, "row index {row} is out of range");
        assert!(matches!(side, RowSide::Lower | RowSide::Upper));
    }
    assert!(
        evidence.iter().any(|(_, side, _)| *side == RowSide::Lower)
            && evidence.iter().any(|(_, side, _)| *side == RowSide::Upper),
        "the conflict needs one side of each row: {evidence:?}"
    );

    // The producer declines an integer problem rather than guessing.
    let mut with_binary = infeasible_lp();
    with_binary.add_integer_col(0.0, 0.0, 1.0);
    assert!(
        prove_infeasible_with_row_farkas(&with_binary, 30.0)
            .expect("the lane runs")
            .is_none(),
        "the LP-only Farkas lane must fail closed on binaries"
    );

    // The matching checker is reachable from the same facade (it lives on the
    // certificate module, which is the correct pairing).
    let checker: fn(
        &ny_api::cert::FarkasCertificate,
    ) -> Result<ny_api::cert::Rat, ny_api::cert::CheckError> = check_farkas;
    let _ = checker;
}

#[test]
fn solver_side_types_are_nameable() {
    assert_ne!(Sense::Minimise, Sense::Maximise);
    let cache = SplitUnsatCache::default();
    let _ = &cache;

    // The witness-only probe lane: nameable, and by construction it has no
    // `Unsat` variant, so it can never authorize a verdict.
    fn assert_probe_is_verdict_free(probe: &ny_api::complete::OneSidedSatProbe) -> bool {
        matches!(probe, ny_api::complete::OneSidedSatProbe::Witness(_))
            || matches!(probe, ny_api::complete::OneSidedSatProbe::Declined(_))
    }
    let declined = ny_api::complete::OneSidedSatProbe::Declined(
        ny_api::complete::OneSidedSatDecline::UnsupportedBackend,
    );
    assert!(assert_probe_is_verdict_free(&declined));
    let witness = ny_api::complete::OneSidedSatWitness {
        objective: 0.0,
        output_values: vec![0.0],
        input_values: vec![0.0],
    };
    assert!(assert_probe_is_verdict_free(
        &ny_api::complete::OneSidedSatProbe::Witness(witness)
    ));
}

#[test]
fn star_search_lane_is_constructible_and_candidate_only() {
    let layers = vec![
        StarLayer::Gemm {
            weight: arr2(&[[1.0_f32], [-1.0]]),
            bias: Some(arr1(&[0.0, 0.0])),
        },
        StarLayer::Relu,
        StarLayer::Gemm {
            weight: arr2(&[[1.0_f32, 1.0]]),
            bias: Some(arr1(&[0.0])),
        },
    ];
    let input: Star = Star::from_input_box(&ArrayD::from_elem(IxDyn(&[1]), 0.0_f32), 1.0);
    let budget = StarBudget::new(256, 8, Instant::now() + Duration::from_secs(30));

    // `|x| > -1` holds everywhere on the box.
    let spec = StarSpec {
        rows: vec![(vec![1.0], -1.0)],
    };
    let (verdict, stats): (StarVerdict, StarStats) =
        star_verify(&layers, &input, &spec, &budget).expect("the star search runs");
    assert!(
        matches!(
            verdict,
            StarVerdict::CandidateVerified
                | StarVerdict::CandidateCounterexampleExists
                | StarVerdict::Unknown(_)
        ),
        "every star verdict is a candidate or an admission of ignorance: {verdict:?}"
    );
    assert!(stats.popped >= 1, "the search must have run: {stats:?}");

    // The unsafe-conjunction lane is a separate, also candidate-only verdict.
    let unsafe_region = StarUnsafeConjunction {
        rows: vec![StarUnsafeRow {
            coefficients: vec![1.0],
            threshold: -1.0,
            strict: false,
        }],
    };
    let (candidate, _stats): (StarUnsafeCandidateVerdict, StarStats) =
        star_search_unsafe_conjunction(&layers, &input, &unsafe_region, &budget)
            .expect("the unsafe-conjunction search runs");
    assert!(
        matches!(
            candidate,
            StarUnsafeCandidateVerdict::CandidateUnsafeClosureEmpty
                | StarUnsafeCandidateVerdict::CandidateUnsafeClosureNotProvedEmpty
                | StarUnsafeCandidateVerdict::Unknown(_)
        ),
        "the unsafe lane has no verdict-bearing variant: {candidate:?}"
    );
}

/// `Linear -> ReLU -> Linear -> ReLU -> Linear` computing exactly `0`, with two
/// second-layer neurons that share a pre-activation.
fn cancelling_chain() -> GraphNetwork {
    fn linear(weight: Vec<f32>, rows: usize, cols: usize, bias: Vec<f32>) -> LinearLayer {
        LinearLayer::new(
            ndarray::Array2::from_shape_vec((rows, cols), weight).expect("weight shape"),
            Some(arr1(&bias)),
        )
        .expect("valid linear layer")
    }

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(linear(vec![1.0, -1.0], 2, 1, vec![0.0, 0.0])),
    ));
    graph.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer::new()),
        vec!["l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "l2",
        Layer::Linear(linear(vec![1.0, 1.0, 1.0, 1.0], 2, 2, vec![-0.5, -0.5])),
        vec!["r1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "r2",
        Layer::ReLU(ReLULayer::new()),
        vec!["l2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "l3",
        Layer::Linear(linear(vec![1.0, -1.0], 1, 2, vec![0.0])),
        vec!["r2".to_string()],
    ));
    graph.set_output("l3");
    graph
}

#[test]
fn graph_terminal_reduces_and_verifies_through_the_facade() {
    let graph = cancelling_chain();
    let chain: FcReluChain =
        ny_api::complete::reduce_graph_to_fc_relu(&graph).expect("a fragment graph reduces");
    assert_eq!(chain.layer_dims(), &[1, 2, 2, 1]);
    assert_eq!(chain.num_linear_layers(), 3);
    assert_eq!(chain.input_dim(), 1);
    assert_eq!(chain.output_dim(), 1);
    assert_eq!(chain.linear_node_names().len(), 3);

    let options = CompleteMipOptions {
        mip: MipConfig {
            timeout_secs: 30.0,
            ..MipConfig::default()
        },
        preactivation_inflation: DEFAULT_PREACTIVATION_INFLATION,
    };
    assert_eq!(
        CompleteMipOptions::default().preactivation_inflation,
        DEFAULT_PREACTIVATION_INFLATION
    );

    let spec = VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(-0.01, 0.01)])
        .expect("valid spec");
    let report: CompleteMipReport =
        verify_graph_complete_mip(&graph, &spec, &options).expect("encodable graph");
    assert!(report.verified, "the MIP sees the cancellation: {report:?}");
    assert_eq!(report.disjuncts, 2);
    assert_eq!(report.disjuncts_refuted, 2);
    assert!(report.inconclusive.is_none());
}
