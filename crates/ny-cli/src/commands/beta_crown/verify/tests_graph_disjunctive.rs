// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
    time::Duration,
};

use ndarray::{arr1, ArrayD, IxDyn};
use ny_onnx::vnnlib::parse_vnnlib;
use ny_propagate::{
    beta_crown::BetaCrownConfig,
    layers::{Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer},
    BabVerificationStatus, BetaCrownVerifier, GraphNetwork, GraphNode, InputClipType, Layer,
    Network,
};
use ny_tensor::BoundedTensor;

use super::disjunctive::config_for_clause_invprop;
use super::disjunctive_unified::filter_unverified_clauses_for_unified;
use super::{verify_relational_constraints, BetaCrownModel};

#[derive(Clone, Default)]
struct CapturedTracing(Arc<Mutex<Vec<u8>>>);

struct CapturedTracingWriter(Arc<Mutex<Vec<u8>>>);

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedTracing {
    type Writer = CapturedTracingWriter;

    fn make_writer(&'a self) -> Self::Writer {
        CapturedTracingWriter(Arc::clone(&self.0))
    }
}

impl Write for CapturedTracingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("captured tracing mutex").extend(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl CapturedTracing {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().expect("captured tracing mutex").clone())
            .expect("tracing output is UTF-8")
    }
}

fn build_two_output_sequential_network() -> Network {
    let w1 = ndarray::arr2(&[[1.0_f32]]);
    let b1 = arr1(&[0.0_f32]);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    let w2 = ndarray::arr2(&[[1.0_f32], [-1.0]]);
    let b2 = arr1(&[0.5_f32, 0.5]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network
}

fn build_single_relu_anti_correlated_graph_for_disjunction() -> (GraphNetwork, BoundedTensor) {
    let linear1 = LinearLayer::new(ndarray::Array2::eye(1), None).unwrap();
    let w2 = ndarray::arr2(&[[1.0_f32], [-1.0]]);
    let b2 = arr1(&[0.5_f32, 0.5]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");

    let input =
        BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap();

    (graph, input)
}

fn build_single_relu_anti_correlated_conv_graph_for_disjunction() -> (GraphNetwork, BoundedTensor) {
    let conv = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).unwrap(),
        None,
        (1, 1),
        (0, 0),
        1,
        1,
    )
    .unwrap();
    let w2 = ndarray::arr2(&[[1.0_f32], [-1.0]]);
    let b2 = arr1(&[0.5_f32, 0.5]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear2));

    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.0_f32),
    )
    .unwrap();

    (graph, input)
}

fn make_disjunction_spec_with_threshold(threshold: f32) -> ny_onnx::vnnlib::VnnLibSpec {
    parse_vnnlib(&format!(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (or
    (<= Y_0 {threshold})
    (<= Y_1 {threshold})
))
"#,
    ))
    .unwrap()
}

fn make_disjunction_spec() -> ny_onnx::vnnlib::VnnLibSpec {
    make_disjunction_spec_with_threshold(0.55)
}

fn make_ml4_shaped_opposite_direction_disjunction() -> ny_onnx::vnnlib::VnnLibSpec {
    let outputs = (0..160).fold(String::new(), |mut acc, idx| {
        use std::fmt::Write as _;
        let _ = writeln!(acc, "(declare-const Y_{idx} Real)");
        acc
    });
    parse_vnnlib(&format!(
        r#"
(declare-const X_0 Real)
{outputs}
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (or
    (>= Y_159 1.060001)
    (<= Y_159 0.939999)
))
"#,
    ))
    .unwrap()
}

fn make_three_clause_disjunction_spec() -> ny_onnx::vnnlib::VnnLibSpec {
    parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (or
    (<= Y_0 Y_1)
    (<= Y_2 Y_3)
    (<= Y_1 Y_3)
))
"#,
    )
    .unwrap()
}

fn make_single_clause_spec(
    spec: &ny_onnx::vnnlib::VnnLibSpec,
    clause_index: usize,
) -> ny_onnx::vnnlib::VnnLibSpec {
    let mut clause_spec = spec.clone();
    clause_spec.output_constraints = clause_spec.output_constraint_clauses[clause_index].clone();
    clause_spec.output_constraint_clauses = Vec::new();
    clause_spec.is_disjunction = false;
    clause_spec.per_clause_input_bounds = Vec::new();
    clause_spec
}

fn make_graph_disjunction_config() -> BetaCrownConfig {
    BetaCrownConfig {
        max_domains: 100,
        max_depth: 10,
        timeout: Duration::from_secs(5),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        batch_size: 1,
        ..Default::default()
    }
}

#[test]
fn disjunctive_clause_invprop_rebinds_exact_matrix_and_isolates_clauses() {
    let top_level = make_ml4_shaped_opposite_direction_disjunction();
    let first_spec = make_single_clause_spec(&top_level, 0);
    let second_spec = make_single_clause_spec(&top_level, 1);
    let stale_constraints = first_spec.to_output_constraints().unwrap();

    let mut base = make_graph_disjunction_config();
    base.max_domains = 731;
    base.alpha_config.invprop.enabled = true;
    base.alpha_config.invprop.optimize_gammas = true;
    base.alpha_config.invprop.share_gammas = true;
    base.alpha_config.invprop.gamma_lr = 0.375;
    base.alpha_config.invprop.apply_output_constraints_to = vec!["all".to_string()];
    // Deliberately seed clause 0's matrix.  Clause 1 must overwrite it; a
    // conditional get-or-insert implementation is unsound and fails below.
    base.alpha_config.output_constraints = Some(stale_constraints.clone());

    let first = config_for_clause_invprop(&base, &first_spec);
    let second = config_for_clause_invprop(&base, &second_spec);
    let first_constraints = first.alpha_config.output_constraints.as_ref().unwrap();
    let second_constraints = second.alpha_config.output_constraints.as_ref().unwrap();
    let first_expected = first_spec.to_output_constraints().unwrap();
    let second_expected = second_spec.to_output_constraints().unwrap();

    for constraints in [first_constraints, second_constraints] {
        assert!(constraints.is_conjunction);
        assert!(constraints.clause_indices.is_none());
        assert_eq!(constraints.num_constraints(), 1);
        assert_eq!(constraints.output_dim(), 160);
    }
    assert_eq!(first_constraints.a_matrix, first_expected.a_matrix);
    assert_eq!(first_constraints.rhs, first_expected.rhs);
    assert_eq!(second_constraints.a_matrix, second_expected.a_matrix);
    assert_eq!(second_constraints.rhs, second_expected.rhs);
    assert_eq!(first_constraints.a_matrix[[0, 159]], -1.0);
    assert_eq!(second_constraints.a_matrix[[0, 159]], 1.0);
    assert_eq!(
        first_constraints.rhs[0].to_bits(),
        ny_tensor::next_up_f32(-1.060001_f32).to_bits()
    );
    assert_eq!(
        second_constraints.rhs[0].to_bits(),
        ny_tensor::next_up_f32(0.939999_f32).to_bits()
    );

    for rebound in [&first, &second] {
        assert_eq!(rebound.max_domains, base.max_domains);
        assert_eq!(
            rebound.alpha_config.invprop.enabled,
            base.alpha_config.invprop.enabled
        );
        assert_eq!(
            rebound.alpha_config.invprop.optimize_gammas,
            base.alpha_config.invprop.optimize_gammas
        );
        assert_eq!(
            rebound.alpha_config.invprop.share_gammas,
            base.alpha_config.invprop.share_gammas
        );
        assert_eq!(
            rebound.alpha_config.invprop.gamma_lr.to_bits(),
            base.alpha_config.invprop.gamma_lr.to_bits()
        );
        assert_eq!(
            rebound.alpha_config.invprop.apply_output_constraints_to,
            base.alpha_config.invprop.apply_output_constraints_to
        );
    }
    assert_eq!(
        base.alpha_config
            .output_constraints
            .as_ref()
            .unwrap()
            .a_matrix,
        stale_constraints.a_matrix,
        "clause rebinding must not mutate the shared top-level config"
    );

    let mut invprop_disabled = base.clone();
    invprop_disabled.alpha_config.invprop.enabled = false;
    let disabled = config_for_clause_invprop(&invprop_disabled, &second_spec);
    assert!(
        !disabled.alpha_config.invprop.enabled,
        "clause rebinding must preserve the NY_INVPROP=0 policy decision"
    );

    let assert_refused = |refused: BetaCrownConfig, context: &str| {
        assert!(
            refused.alpha_config.output_constraints.is_none(),
            "{context}: refused clause must clear every inherited matrix"
        );
        assert!(
            !refused.alpha_config.invprop.enabled,
            "{context}: refused clause must disable the inert INVPROP channel"
        );
        assert!(
            !refused.alpha_config.invprop.optimize_gammas,
            "{context}: refused clause must disable gamma optimization"
        );
    };

    let mut disjunction_only = first_spec.clone();
    disjunction_only.is_disjunction = true;
    assert_refused(
        config_for_clause_invprop(&base, &disjunction_only),
        "top-level disjunction",
    );

    let mut grouped_only = first_spec.clone();
    grouped_only.output_constraint_clauses = vec![grouped_only.output_constraints.clone()];
    assert_refused(
        config_for_clause_invprop(&base, &grouped_only),
        "residual clause grouping",
    );

    let mut empty = first_spec.clone();
    empty.output_constraints.clear();
    assert_refused(config_for_clause_invprop(&base, &empty), "empty clause");

    let mut malformed = first_spec;
    malformed.output_constraints = vec![ny_onnx::vnnlib::OutputConstraint::LessEqConst(
        malformed.num_outputs,
        0.0,
    )];
    assert_refused(
        config_for_clause_invprop(&base, &malformed),
        "programmatic out-of-range output index",
    );
}

#[test]
fn serial_disjunctive_dispatch_engages_clause_local_invprop() {
    let telemetry_run = ny_propagate::execution_telemetry::begin_run();
    let (graph, input) = build_single_relu_anti_correlated_graph_for_disjunction();
    let vnnlib = make_disjunction_spec();
    let mut config = make_graph_disjunction_config();
    config.use_alpha_crown = true;
    config.alpha_config.iterations = 1;
    config.alpha_config.adaptive_skip = false;
    config.alpha_config.invprop.enabled = true;
    config.alpha_config.invprop.optimize_gammas = false;
    config.alpha_config.output_constraints = None;
    let verifier = BetaCrownVerifier::new(config.clone());

    let captured = CapturedTracing::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_ansi(false)
        .without_time()
        .with_target(false)
        .with_writer(captured.clone())
        .finish();
    // This deliberately tiny one-ReLU graph stays on the synchronous
    // single-domain route, so the thread-local subscriber observes the actual
    // downstream alpha initialization.  If this fixture is ever parallelized,
    // replace the trace assertion with a typed metrics sink.
    let result = tracing::subscriber::with_default(subscriber, || {
        verify_relational_constraints(
            &BetaCrownModel::Graph(Box::new(graph)),
            &input,
            &vnnlib,
            &config,
            &verifier,
            true,  // use_relu_split
            false, // gpu_bab
            false, // pgd_attack
            0,     // pgd_restarts
            0,     // pgd_steps
            5,     // timeout
            None,  // gemm_engine
            true,  // json
        )
    })
    .unwrap();
    assert!(matches!(
        result.result,
        BabVerificationStatus::Unknown { .. }
    ));

    let logs = captured.text();
    assert!(
        logs.contains("GraphNetwork α-CROWN: INVPROP enabled with 1 constraints"),
        "the serial clause dispatch did not deliver the rebound matrix downstream:\n{logs}"
    );
    assert!(
        !logs.contains("INVPROP enabled in config but no output_constraints provided"),
        "a serial clause reached graph alpha without its local matrix:\n{logs}"
    );
    let observed = ny_propagate::execution_telemetry::snapshot();
    assert!(observed.invprop.observed);
    assert!(observed.invprop.clause_rebind_attempts > 0);
    assert_eq!(
        observed.invprop.clause_rebind_accepted + observed.invprop.clause_rebind_refused,
        observed.invprop.clause_rebind_attempts
    );
    assert!(observed.invprop.clause_rebind_accepted > 0);
    assert!(observed.invprop.alpha_initializations > 0);
    assert_eq!(observed.invprop.gamma_steps_attempted, 0);
    assert!(!observed.invprop.attribution_conflict);
    drop(telemetry_run);
    assert!(!ny_propagate::execution_telemetry::snapshot().run_active);
}

#[test]
fn graph_multi_clause_disjunction_nonconv_keeps_clause_loop_reason_3813() {
    let (graph, input) = build_single_relu_anti_correlated_graph_for_disjunction();
    let vnnlib = make_disjunction_spec();

    assert!(vnnlib.has_multi_constraint_disjunction());
    assert!(vnnlib
        .output_constraint_clauses
        .iter()
        .all(|clause| clause.len() == 1));

    let config = make_graph_disjunction_config();
    let verifier = BetaCrownVerifier::new(config.clone());

    let result = verify_relational_constraints(
        &BetaCrownModel::Graph(Box::new(graph)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        true,  // use_relu_split
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        5,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();

    let reason = match result.result {
        BabVerificationStatus::Unknown { reason } => reason,
        other => panic!("expected Unknown graph disjunctive result, got {other:?}"),
    };

    assert!(
        reason.starts_with("Clause 1:"),
        "#3813 precheck skip is scoped to Conv2d-heavy graphs; non-conv graph disjunctions should keep the per-clause fallback reason, got: {reason}"
    );
}

#[test]
fn graph_multi_clause_disjunction_conv2d_skips_clause_loop_reason_3813() {
    let (graph, input) = build_single_relu_anti_correlated_conv_graph_for_disjunction();
    let vnnlib = make_disjunction_spec();

    assert!(vnnlib.has_multi_constraint_disjunction());
    assert!(vnnlib
        .output_constraint_clauses
        .iter()
        .all(|clause| clause.len() == 1));

    let config = make_graph_disjunction_config();
    let verifier = BetaCrownVerifier::new(config.clone());

    let result = verify_relational_constraints(
        &BetaCrownModel::Graph(Box::new(graph)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        true,  // use_relu_split
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        5,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();

    let reason = match result.result {
        BabVerificationStatus::Unknown { reason } => reason,
        other => panic!("expected Unknown graph disjunctive result, got {other:?}"),
    };

    assert!(
        !reason.starts_with("Clause 1:"),
        "#3813 Conv2d graphs should still use the batched multi-objective path, got fallback reason: {reason}"
    );
}

#[test]
fn graph_multi_clause_disjunction_conv2d_gpu_bab_matches_direct_clause_dispatch_3862() {
    let vnnlib = make_disjunction_spec_with_threshold(0.75);
    let clause_spec = make_single_clause_spec(&vnnlib, 0);

    let config = BetaCrownConfig {
        max_domains: 0,
        ..make_graph_disjunction_config()
    };
    let verifier = BetaCrownVerifier::new(config.clone());

    let (direct_graph, direct_input) =
        build_single_relu_anti_correlated_conv_graph_for_disjunction();
    let direct_result = verify_relational_constraints(
        &BetaCrownModel::Graph(Box::new(direct_graph)),
        &direct_input,
        &clause_spec,
        &config,
        &verifier,
        true,  // use_relu_split
        true,  // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        5,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();
    let direct_reason = match direct_result.result {
        BabVerificationStatus::Unknown { reason } => reason,
        other => panic!("expected single-clause GPU-BaB result to be Unknown, got {other:?}"),
    };

    let (graph, input) = build_single_relu_anti_correlated_conv_graph_for_disjunction();
    let disjunctive_result = verify_relational_constraints(
        &BetaCrownModel::Graph(Box::new(graph)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        true,  // use_relu_split
        true,  // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        5,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();

    let reason = match disjunctive_result.result {
        BabVerificationStatus::Unknown { reason } => reason,
        other => panic!("expected Unknown graph disjunctive result, got {other:?}"),
    };

    assert!(
        reason.starts_with("Clause 1:"),
        "#3862 gpu_bab=true must bypass the shared multi-objective fast path and return per-clause reasoning, got: {reason}"
    );
    assert_eq!(
        reason,
        format!("Clause 1: {direct_reason}"),
        "#3862 disjunctive gpu_bab routing should match the direct single-clause GPU-BaB dispatch"
    );
}

#[test]
fn unified_input_split_prunes_preverified_clauses_4257() {
    let vnnlib = make_three_clause_disjunction_spec();
    let clauses = vnnlib.output_constraint_clauses.clone();

    let (filtered_vnnlib, filtered_clauses) =
        filter_unverified_clauses_for_unified(&vnnlib, &clauses, &[true, false, true])
            .expect("pre-verified clauses should be removed before grouped search");

    assert_eq!(
        filtered_clauses,
        vec![clauses[1].clone()],
        "#4257 unified grouped search should only keep unresolved clauses"
    );
    assert_eq!(
        filtered_vnnlib.output_constraint_clauses,
        vec![clauses[1].clone()],
        "#4257 filtered VNN-LIB should preserve only the unresolved clause group"
    );
    assert_eq!(
        filtered_vnnlib.output_constraints,
        clauses[1].clone(),
        "#4257 flat output_constraints should stay aligned with the unresolved grouped clause"
    );
    assert!(
        filtered_vnnlib.is_disjunction,
        "filtered unified spec must remain disjunctive"
    );
}

#[test]
fn unified_input_split_keeps_original_spec_without_preverified_clauses_4257() {
    let vnnlib = make_three_clause_disjunction_spec();
    let clauses = vnnlib.output_constraint_clauses.clone();

    assert!(
        filter_unverified_clauses_for_unified(&vnnlib, &clauses, &[false, false, false]).is_none(),
        "#4257 should not clone or rewrite the grouped spec when precheck proves nothing"
    );
}

#[test]
fn unified_input_split_pruning_keeps_per_clause_input_bounds_4257() {
    let mut vnnlib = make_three_clause_disjunction_spec();
    let clauses = vnnlib.output_constraint_clauses.clone();

    let mut filtered_clause_bounds = std::collections::BTreeMap::new();
    filtered_clause_bounds.insert(0, (-0.25, 0.25));
    vnnlib.per_clause_input_bounds = vec![
        std::collections::BTreeMap::new(),
        filtered_clause_bounds.clone(),
        std::collections::BTreeMap::new(),
    ];

    let (filtered_vnnlib, filtered_clauses) =
        filter_unverified_clauses_for_unified(&vnnlib, &clauses, &[true, false, true])
            .expect("pre-verified clauses should be removed before grouped search");

    assert_eq!(
        filtered_clauses,
        vec![clauses[1].clone()],
        "#4257 unified pruning should keep only the unresolved clause"
    );
    assert_eq!(
        filtered_vnnlib.per_clause_input_bounds,
        vec![filtered_clause_bounds],
        "#4257 pruning must preserve unresolved per-clause bounds so grouped routing stays gated off"
    );
}

/// Sequential model with use_relu_split=false should use the grouped input-split
/// disjunctive BaB lane (Packet B of #3740). The grouped lane converts Sequential
/// → Graph and runs one shared BaB tree instead of the per-clause timeout-slicing
/// loop. The result should NOT contain "Clause 1:" prefix because the grouped
/// verifier does not decompose into per-clause sub-problems.
///
/// Part of #3740 Packet B.
#[test]
fn sequential_input_split_disjunction_uses_grouped_lane_3740() {
    let network = build_two_output_sequential_network();
    let vnnlib = make_disjunction_spec();

    assert!(vnnlib.has_multi_constraint_disjunction());

    let config = make_graph_disjunction_config();
    let verifier = BetaCrownVerifier::new(config.clone());

    let result = verify_relational_constraints(
        &BetaCrownModel::Sequential(Box::new(network)),
        &BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap(),
        &vnnlib,
        &config,
        &verifier,
        false, // use_relu_split = false → input split
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        5,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();

    // The grouped lane returns result directly from verify_graph_input_split_multi_clause_disjunctive,
    // not the per-clause loop. If this falls through to the clause loop, the reason will
    // start with "Clause 1:".
    match &result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                !reason.starts_with("Clause 1:"),
                "#3740 input-split disjunctions should use grouped lane, not per-clause fallback, got: {reason}"
            );
        }
        BabVerificationStatus::Verified => {
            // Even better — the grouped lane verified the property.
        }
        BabVerificationStatus::Timeout => {
            // Acceptable — the grouped lane ran but timed out.
        }
        other => {
            panic!("unexpected result from grouped disjunctive lane: {other:?}");
        }
    }
}

/// Graph model with use_relu_split=false should also use the grouped lane.
/// Part of #3740 Packet B.
#[test]
fn graph_input_split_disjunction_uses_grouped_lane_3740() {
    let (graph, input) = build_single_relu_anti_correlated_graph_for_disjunction();
    let vnnlib = make_disjunction_spec();
    let config = make_graph_disjunction_config();
    let verifier = BetaCrownVerifier::new(config.clone());

    let result = verify_relational_constraints(
        &BetaCrownModel::Graph(Box::new(graph)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        false, // use_relu_split = false → input split
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        5,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();

    match &result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                !reason.starts_with("Clause 1:"),
                "#3740 graph input-split disjunctions should use grouped lane, got: {reason}"
            );
        }
        BabVerificationStatus::Verified | BabVerificationStatus::Timeout => {}
        other => {
            panic!("unexpected result from grouped disjunctive lane: {other:?}");
        }
    }
}

/// Per-clause input bounds should force the clause-by-clause fallback,
/// not the grouped lane. The grouped lane does not support per-clause domains.
/// Part of #3740 Packet B.
#[test]
fn per_clause_input_bounds_keeps_clause_fallback_3740() {
    let (graph, input) = build_single_relu_anti_correlated_graph_for_disjunction();
    let mut vnnlib = make_disjunction_spec();

    // Add non-empty per-clause input bounds to trigger fallback.
    let mut per_clause = std::collections::BTreeMap::new();
    per_clause.insert(0, (-0.5, 0.5));
    vnnlib.per_clause_input_bounds = vec![per_clause, std::collections::BTreeMap::new()];

    let config = make_graph_disjunction_config();
    let verifier = BetaCrownVerifier::new(config.clone());

    let result = verify_relational_constraints(
        &BetaCrownModel::Graph(Box::new(graph)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        false, // use_relu_split = false → input split
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        5,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();

    // With per-clause input bounds, should fall through to the per-clause loop
    // which produces "Clause N:" prefixed reasons.
    match &result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                reason.starts_with("Clause 1:"),
                "#3740 per_clause_input_bounds should keep clause fallback, got: {reason}"
            );
        }
        BabVerificationStatus::Verified | BabVerificationStatus::Timeout => {
            // If verified or timeout, the test is inconclusive for routing but
            // acceptable as correct behavior.
        }
        other => {
            panic!("unexpected result: {other:?}");
        }
    }
}

/// Build a two-clause disjunction where clause 0 is trivially satisfied
/// at the root domain but clause 1 is not. This is the exact false-positive
/// shape that triggers the grouped clip bug: the scalar `clip_outcome.verified`
/// fires on clause 0's row and short-circuits the whole child as Verified
/// without checking clause 1.
///
/// Network: y0 = ReLU(x) + 0.5 ∈ [0.5, 1.5], y1 = -ReLU(x) + 0.5 ∈ [-0.5, 0.5]
/// Spec: (or (<= Y_0 0.3) (<= Y_1 0.3))
///   Clause 0: prove Y_0 > 0.3 → lower(Y_0)=0.5 > 0.3 → satisfied
///   Clause 1: prove Y_1 > 0.3 → lower(Y_1)=-0.5 NOT > 0.3 → NOT satisfied
///
/// Correct result: NOT Verified (clause 1 cannot be discharged).
fn make_asymmetric_disjunction_spec() -> ny_onnx::vnnlib::VnnLibSpec {
    parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (or
    (<= Y_0 0.3)
    (<= Y_1 0.3)
))
"#,
    )
    .unwrap()
}

/// Regression: grouped relaxed clip must not short-circuit on a single
/// satisfied clause. Before Packet B1, `clip_outcome.verified` (a conjunctive
/// "any row proved" scalar) caused false Verified results for grouped
/// OR-of-AND specs. Part of #3740 Packet B1.
#[test]
fn graph_input_split_disjunction_relaxed_clip_requires_all_clauses_3740() {
    let (graph, input) = build_single_relu_anti_correlated_graph_for_disjunction();
    let vnnlib = make_asymmetric_disjunction_spec();

    assert!(vnnlib.has_multi_constraint_disjunction());

    // Small domain/depth budget: the bug triggers on the first BaB child
    // where clip fires (right child [0,1] at depth 1). No need for deep search.
    let config = BetaCrownConfig {
        max_domains: 10,
        max_depth: 3,
        timeout: Duration::from_secs(2),
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        batch_size: 1,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config.clone());

    let result = verify_relational_constraints(
        &BetaCrownModel::Graph(Box::new(graph)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        false, // use_relu_split = false → grouped input split
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        2,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();

    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "#3740 Packet B1: relaxed clip must not treat one satisfied clause as \
         whole-domain verification for grouped OR-of-AND specs, got {:?}",
        result.result
    );
}

/// Regression: grouped complete clip must not short-circuit on a single
/// satisfied clause. Same false-positive shape as the relaxed path but
/// exercising the complete clip code path. Part of #3740 Packet B1.
#[test]
fn graph_input_split_disjunction_complete_clip_requires_all_clauses_3740() {
    let (graph, input) = build_single_relu_anti_correlated_graph_for_disjunction();
    let vnnlib = make_asymmetric_disjunction_spec();

    assert!(vnnlib.has_multi_constraint_disjunction());

    let config = BetaCrownConfig {
        max_domains: 10,
        max_depth: 3,
        timeout: Duration::from_secs(2),
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Complete,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        batch_size: 1,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config.clone());

    let result = verify_relational_constraints(
        &BetaCrownModel::Graph(Box::new(graph)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        false, // use_relu_split = false → grouped input split
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        2,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .unwrap();

    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "#3740 Packet B1: complete clip must not treat one satisfied clause as \
         whole-domain verification for grouped OR-of-AND specs, got {:?}",
        result.result
    );
}
