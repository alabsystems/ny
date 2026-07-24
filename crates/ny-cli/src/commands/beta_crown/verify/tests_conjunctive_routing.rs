// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::constraint_iter::{iterate_constraints, ConstraintIterConfig};
use super::{verify_relational_constraints, AggregationMode, BetaCrownModel};
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

fn build_sequential_input_split_non_joint_test_1923() -> (
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
        use_crown_ibp: false,
        enable_cuts: false,
        ..Default::default()
    };

    (network, vnnlib, input, config)
}

/// Sequential input-split conjunctive specs WITHOUT the same-LHS max-diff
/// reduction (constant conjunctions — the sat_relu/lsnc shape) now upgrade to
/// the graph joint multi-objective path, which decides jointly-impossible
/// specs that per-constraint decomposition cannot (each individual constraint
/// remains satisfiable, so per-constraint BaB returns Unknown forever).
///
/// MEASURED 2026-07-10: sat_relu unsat_v30_c38 (Y_0>=1 AND Y_1<=0) went
/// unsat 36.4s -> 2s via this upgrade. The #1923 concern (rerouting ACAS-Xu
/// input-split models) is covered by the same-LHS carve-out asserted in
/// `same_lhs_reduction_pins_acasxu_shape_to_sequential_path_1923` below:
/// same-LHS relational conjunctions never reach this upgrade.
///
/// This fixture is impossible only jointly: each individual constraint remains
/// satisfiable somewhere in the input box, so the sequential per-constraint path
/// satisfiable, so only the joint per-subdomain any-row semantics can prove it.
#[test]
fn sequential_input_split_constant_conjunction_upgrades_to_joint_graph() {
    let (network, vnnlib, input, config) = build_sequential_input_split_non_joint_test_1923();
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

/// #1923 regression guard, re-expressed for the shape-aware dispatch: ACAS-Xu
/// style same-LHS relational conjunctions (prop_2: Y_1<=Y_0, Y_2<=Y_0, ...;
/// prop_3/4: literal same-LHS) must match the max-diff reduction predicate,
/// which pins them to the sequential pipeline (upfront PGD -> full-budget
/// max-diff BaB) instead of the graph joint upgrade. MEASURED 2026-07-10:
/// 4_2/prop_3 verifies in 672 max-diff domains (~4s BaB) while the graph
/// multi-objective input-split lane times out at 116s (>100k domains).
#[test]
fn same_lhs_reduction_pins_acasxu_shape_to_sequential_path_1923() {
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
