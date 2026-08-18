// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::constraint_iter::{iterate_constraints, ConstraintIterConfig};
use super::{
    build_multi_objectives, config_for_sequential_conjunction_graph,
    should_upgrade_sequential_conjunction_to_graph, verify_relational_constraints, AggregationMode,
    BetaCrownModel,
};
use ndarray::{arr1, arr2};
use ny_core::Result as NyResult;
use ny_onnx::vnnlib::parse_vnnlib;
use ny_propagate::{
    beta_crown::{BetaCrownConfig, BranchingHeuristic},
    layers::LinearLayer,
    BabVerificationStatus, BetaCrownResult, BetaCrownVerifier, GraphDomainBatchMetricsSink,
    GraphDomainBatchRecord, InputSplitBatchRecord, InputSplitMetricsSink, Layer, Network,
};
use ny_tensor::BoundedTensor;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug)]
struct NoopInputSplitMetricsSink;

impl InputSplitMetricsSink for NoopInputSplitMetricsSink {
    fn record_batch_summary(&self, _record: &InputSplitBatchRecord) -> NyResult<()> {
        Ok(())
    }
}

#[derive(Debug)]
struct NoopGraphDomainBatchMetricsSink;

impl GraphDomainBatchMetricsSink for NoopGraphDomainBatchMetricsSink {
    fn record_batch_summary(&self, _record: &GraphDomainBatchRecord) -> NyResult<()> {
        Ok(())
    }
}

fn build_sequential_input_split_constant_conjunction_test_1923() -> (
    Network,
    ny_onnx::vnnlib::VnnLibSpec,
    BoundedTensor,
    BetaCrownConfig,
) {
    let linear = LinearLayer::new(
        arr2(&[[2.0_f32, 1.0_f32], [-4.0_f32, 5.0_f32]]),
        Some(arr1(&[-0.1_f32, 2.01_f32])),
    )
    .expect("two-output linear layer should build");

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let vnnlib = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 0.0))
(assert (<= Y_1 0.0))
"#,
    )
    .expect("joint-clause sequential spec should parse");

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("finite bounds");

    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        input_split_sb_margin_weight: 1.0,
        max_domains: 8,
        max_depth: 1,
        timeout: Duration::from_secs(1),
        use_alpha_crown: false,
        // Mirrors the late Sequential -> Graph capability mismatch from the
        // ACAS proof trace; the route adapter must make it graph-compatible.
        use_crown_ibp: true,
        enable_cuts: false,
        ..Default::default()
    };

    (network, vnnlib, input, config)
}

/// Sequential input-split conjunctive specs upgrade to the graph joint
/// multi-objective path, which decides jointly-impossible specs that
/// per-constraint decomposition cannot (each individual constraint remains
/// satisfiable, so per-constraint BaB returns Unknown forever).
///
/// MEASURED 2026-07-10: sat_relu unsat_v30_c38 (Y_0>=1 AND Y_1<=0) went
/// unsat 36.4s -> 2s via this upgrade. Same-LHS relational conjunctions now
/// use the same lane; the parity tests below pin their signed rows.
///
/// This fixture is impossible only jointly: each individual constraint remains
/// satisfiable somewhere in the input box, so only the joint per-subdomain
/// any-row semantics can prove it.
#[test]
fn sequential_input_split_constant_conjunction_upgrades_to_joint_graph() {
    let (network, vnnlib, input, config) =
        build_sequential_input_split_constant_conjunction_test_1923();
    assert_eq!(
        vnnlib.output_constraints.len(),
        2,
        "expected a 2-constraint conjunction"
    );
    assert!(
        super::sequential::normalize_same_lhs_reduction(&vnnlib).is_none(),
        "constant conjunction must NOT match the same-LHS max-diff reduction"
    );

    let verifier = BetaCrownVerifier::new(config.clone());
    let result = verify_relational_constraints(
        &BetaCrownModel::Sequential(Box::new(network)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        false, // use_relu_split=false -> input splitting
        false, // gpu_bab
        false, // pgd_attack
        0,     // pgd_restarts
        0,     // pgd_steps
        1,     // timeout
        None,  // gemm_engine
        true,  // json
    )
    .expect("sequential input-split conjunctive verification should complete");

    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "constant conjunction should upgrade to the graph joint lane and verify the \
         jointly-impossible spec (per-constraint decomposition cannot), got {:?}",
        result.result
    );
}

fn reduction_rows(
    family: &str,
    lhs: usize,
    rhs_indices: &[usize],
    num_outputs: usize,
) -> Vec<Vec<f32>> {
    rhs_indices
        .iter()
        .map(|&rhs| {
            let mut row = vec![0.0_f32; num_outputs];
            match family {
                "ge" => {
                    row[rhs] = 1.0;
                    row[lhs] = -1.0;
                }
                "le" => {
                    row[lhs] = 1.0;
                    row[rhs] = -1.0;
                }
                other => panic!("unexpected reduction family {other}"),
            }
            row
        })
        .collect()
}

/// ACAS-Xu same-LHS conjunctions must route to the graph multi-row lane, and
/// the rows built by the graph planner must exactly match the historical
/// signed-difference reduction. This pins both prop_2's flipped `ge` form and
/// prop_3/4's literal `le` form.
#[test]
fn same_lhs_reduction_routes_to_joint_graph_with_row_parity() {
    // prop_2 shape: flipped same-LHS (Y_1<=Y_0, Y_2<=Y_0 -> Y_0>=Y_1, Y_0>=Y_2).
    let prop2_shape = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
"#,
    )
    .expect("prop_2-shaped spec should parse");
    let (family, lhs, rhs) = super::sequential::normalize_same_lhs_reduction(&prop2_shape)
        .expect("prop_2 shape must match the same-LHS reduction");
    assert_eq!((family, lhs, rhs.as_slice()), ("ge", 0, &[1_usize, 2][..]));
    assert!(
        !should_upgrade_sequential_conjunction_to_graph(&prop2_shape, false),
        "reducible same-LHS prop_2 under INPUT split must stay on the sequential \
         lane: only that engine recomputes CROWN-IBP intermediates per domain \
         (faa66c38 dropped this and cost official prop_3/prop_4 their unsat)"
    );
    assert!(
        should_upgrade_sequential_conjunction_to_graph(&prop2_shape, true),
        "under ReLU split the same shape still takes the graph any-row lane"
    );
    let (prop2_objectives, prop2_thresholds) =
        build_multi_objectives(&prop2_shape).expect("prop_2 objectives should build");
    assert_eq!(
        prop2_objectives,
        reduction_rows(family, lhs, &rhs, prop2_shape.num_outputs),
        "graph prop_2 rows must preserve sequential signed-difference semantics"
    );
    assert_eq!(prop2_thresholds, vec![0.0_f32; rhs.len()]);

    // The 2026 ACAS track also ships the same property in VNN-LIB 2.0 tensor
    // syntax. Parser normalization must reach the identical routing and rows.
    let prop2_v2_shape = parse_vnnlib(
        r#"
(vnnlib-version <2.0>)
(declare-network N
    (declare-input X float32 [1, 1, 1, 1])
    (declare-output Y float32 [1, 3])
)
(assert (>= X[0,0,0,0] 0.0))
(assert (<= X[0,0,0,0] 1.0))
(assert (<= Y[0,1] Y[0,0]))
(assert (<= Y[0,2] Y[0,0]))
"#,
    )
    .expect("VNN-LIB 2.0 prop_2-shaped spec should parse");
    // Same routing as the 1.0 shape above: reducible same-LHS stays sequential
    // under input split, upgrades under ReLU split.
    assert!(!should_upgrade_sequential_conjunction_to_graph(
        &prop2_v2_shape,
        false
    ));
    assert!(should_upgrade_sequential_conjunction_to_graph(
        &prop2_v2_shape,
        true
    ));
    let (prop2_v2_objectives, prop2_v2_thresholds) =
        build_multi_objectives(&prop2_v2_shape).expect("VNN-LIB 2.0 objectives should build");
    assert_eq!(prop2_v2_objectives, prop2_objectives);
    assert_eq!(prop2_v2_thresholds, prop2_thresholds);

    // prop_3/4 shape: literal same-LHS LessEq family (Y_0<=Y_1, Y_0<=Y_2).
    let prop3_shape = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_0 Y_1))
(assert (<= Y_0 Y_2))
"#,
    )
    .expect("prop_3-shaped spec should parse");
    let (family, lhs, rhs) = super::sequential::normalize_same_lhs_reduction(&prop3_shape)
        .expect("prop_3 shape must match the same-LHS reduction");
    assert_eq!((family, lhs, rhs.as_slice()), ("le", 0, &[1_usize, 2][..]));
    assert!(
        !should_upgrade_sequential_conjunction_to_graph(&prop3_shape, false),
        "prop_3/prop_4 are THE regressed rows: reducible same-LHS under input \
         split must keep the sequential lane's per-domain CROWN-IBP intermediates"
    );
    assert!(
        should_upgrade_sequential_conjunction_to_graph(&prop3_shape, true),
        "under ReLU split the same shape still takes the graph any-row lane"
    );
    let (objectives, thresholds) =
        build_multi_objectives(&prop3_shape).expect("prop_3 objectives should build");
    assert_eq!(
        objectives,
        reduction_rows(family, lhs, &rhs, prop3_shape.num_outputs),
        "graph prop_3/4 rows must preserve sequential signed-difference semantics"
    );
    assert_eq!(thresholds, vec![0.0_f32; rhs.len()]);

    // Mixed relational LHS (no shared output) must NOT match: it takes the
    // graph joint upgrade instead.
    let mixed_shape = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_1 Y_0))
(assert (<= Y_0 Y_2))
"#,
    )
    .expect("mixed spec should parse");
    assert!(
        super::sequential::normalize_same_lhs_reduction(&mixed_shape).is_none(),
        "mixed-LHS relational conjunction must not match the same-LHS reduction"
    );
    assert!(
        should_upgrade_sequential_conjunction_to_graph(&mixed_shape, false),
        "mixed-LHS conjunction is NOT same-LHS reducible, so it has no sequential \
         lane to keep and must remain on the existing graph joint route"
    );
}

#[test]
fn same_lhs_input_split_graph_upgrade_uses_fast_bounded_bootstrap() {
    let same_lhs = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
"#,
    )
    .expect("same-LHS spec should parse");
    let incoming = BetaCrownConfig {
        use_alpha_crown: false,
        use_forward_bounds: false,
        use_crown_ibp: true,
        max_domains: 731,
        ..Default::default()
    };

    let input_split = config_for_sequential_conjunction_graph(&incoming, &same_lhs, false);
    assert!(
        input_split.use_forward_bounds,
        "converted same-LHS input splitting needs a fixed root map so graph spec-CROWN does not enter the per-target DAG collector"
    );
    assert!(
        !input_split.use_crown_ibp,
        "the Sequential crown preset must not be reinterpreted as graph per-node CROWN-IBP"
    );
    assert!(
        input_split.input_split_ibp_enhancement,
        "plain CROWN must refresh intermediate references on each input subdomain"
    );
    assert_eq!(input_split.max_domains, incoming.max_domains);

    let relu_split = config_for_sequential_conjunction_graph(&incoming, &same_lhs, true);
    assert_eq!(relu_split.use_forward_bounds, incoming.use_forward_bounds);
    assert_eq!(relu_split.use_crown_ibp, incoming.use_crown_ibp);
    assert_eq!(
        relu_split.input_split_ibp_enhancement,
        incoming.input_split_ibp_enhancement
    );

    let alpha = config_for_sequential_conjunction_graph(
        &BetaCrownConfig {
            use_alpha_crown: true,
            ..incoming.clone()
        },
        &same_lhs,
        false,
    );
    assert!(alpha.use_alpha_crown);
    assert_eq!(alpha.use_forward_bounds, incoming.use_forward_bounds);
    assert_eq!(alpha.use_crown_ibp, incoming.use_crown_ibp);
    assert_eq!(
        alpha.input_split_ibp_enhancement,
        incoming.input_split_ibp_enhancement
    );
}

fn verify_same_lhs_constant_outputs(y1_minus_y0: f32) -> BetaCrownResult {
    let linear = LinearLayer::new(
        arr2(&[[0.0_f32], [0.0_f32], [0.0_f32]]),
        Some(arr1(&[0.0_f32, y1_minus_y0, -1.0_f32])),
    )
    .expect("three-output constant linear layer should build");
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let vnnlib = parse_vnnlib(
        r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_1 Y_0))
(assert (<= Y_2 Y_0))
"#,
    )
    .expect("same-LHS constant-output spec should parse");
    // Restored routing: a reducible same-LHS conjunction under INPUT split stays
    // on the sequential lane (it keeps per-domain CROWN-IBP intermediates), so
    // the graph upgrade is declined here. The end-to-end verification below must
    // still complete and stay sound on whichever lane it lands.
    assert!(!should_upgrade_sequential_conjunction_to_graph(
        &vnnlib, false
    ));
    assert!(should_upgrade_sequential_conjunction_to_graph(
        &vnnlib, true
    ));

    let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite input box");
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 4,
        max_depth: 0,
        timeout: Duration::from_secs(2),
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_relaxed_clip: false,
        reorder_bab: false,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config.clone());
    verify_relational_constraints(
        &BetaCrownModel::Sequential(Box::new(network)),
        &input,
        &vnnlib,
        &config,
        &verifier,
        false,
        false,
        false,
        0,
        0,
        2,
        None,
        true,
    )
    .expect("same-LHS graph verification should complete")
}

/// A conjunctive box closes when any certified row is strictly positive, but
/// equality with the threshold must remain open. The equality arm is genuinely
/// SAT (`Y_1 == Y_0` and `Y_2 < Y_0` everywhere), so this also guards against a
/// false UNSAT introduced at the new routing boundary.
#[test]
fn same_lhs_graph_any_row_closure_is_strict_and_sat_safe() {
    let closed = verify_same_lhs_constant_outputs(0.25);
    assert!(
        matches!(closed.result, BabVerificationStatus::Verified),
        "one strictly positive certified row must close the root, got {:?}",
        closed.result
    );

    let equality_sat = verify_same_lhs_constant_outputs(0.0);
    assert!(
        !matches!(equality_sat.result, BabVerificationStatus::Verified),
        "a row equal to its threshold must not prove the genuinely SAT conjunction"
    );
}

#[test]
fn iterate_constraints_preserves_runtime_sinks_4398() {
    let constraints = vec![ny_onnx::vnnlib::OutputConstraint::GreaterEqConst(0, 0.0)];
    let parent_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        max_domains: 1,
        max_depth: 1,
        timeout: Duration::from_secs(2),
        ..Default::default()
    })
    .with_input_split_metrics_sink(Arc::new(NoopInputSplitMetricsSink))
    .with_graph_domain_batch_metrics_sink(Arc::new(NoopGraphDomainBatchMetricsSink));

    let iter_config = ConstraintIterConfig {
        aggregation: AggregationMode::Conjunctive,
        overall_timeout: Duration::from_secs(5),
        per_constraint_timeout: Duration::from_secs(2),
        min_timeout_ms: 10,
        total_constraint_count: 1,
        num_outputs: 1,
        base_config: BetaCrownConfig {
            max_domains: 1,
            max_depth: 1,
            timeout: Duration::from_secs(2),
            ..Default::default()
        },
        parent_verifier: Some(&parent_verifier),
        engine: None,
        json: true,
    };

    iterate_constraints(
        &constraints,
        &iter_config,
        |dispatch| {
            assert!(
                dispatch.verifier.input_split_metrics_sink_arc().is_some(),
                "per-constraint verifier should inherit input-split sink"
            );
            assert!(
                dispatch
                    .verifier
                    .graph_domain_batch_metrics_sink_arc()
                    .is_some(),
                "per-constraint verifier should inherit graph domain-batch sink"
            );
            Ok(BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: 1,
                domains_verified: 1,
                cuts_generated: 0,
                max_depth_reached: 0,
                time_elapsed: Duration::from_millis(1),
                output_bounds: None,
            })
        },
        None,
    )
    .expect("per-constraint iteration should succeed");
}
