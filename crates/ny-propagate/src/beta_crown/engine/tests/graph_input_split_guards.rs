// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regressions for graph input-split unsplittable-domain guards.
//! Part of #2660.
//!
//! NaN handling divergence tests added for #1860 (R1/1776 finding):
//! Graph input_split hard-errors on non-finite CROWN output, unlike the GPU
//! path which gracefully degrades to Unknown.

use super::prelude::*;

fn input_split_verifier() -> BetaCrownVerifier {
    BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 32,
        max_depth: 8,
        timeout: Duration::from_secs(1),
        ..Default::default()
    })
}

fn single_input_identity_graph() -> GraphNetwork {
    let identity = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity layer should build");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("id", Layer::Linear(identity)));
    graph.set_output("id");
    graph
}

fn empty_input_bias_only_graph() -> GraphNetwork {
    let bias_only = LinearLayer::new(Array2::<f32>::zeros((1, 0)), Some(arr1(&[0.0_f32])))
        .expect("bias-only layer should build");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("bias_only", Layer::Linear(bias_only)));
    graph.set_output("bias_only");
    graph
}

fn sb_margin_multi_objective_graph() -> GraphNetwork {
    // Two-output linear graph crafted so:
    // - objective 0 is nearly verified and prefers splitting x0
    // - objective 1 is far from verified and has a larger x1 coefficient
    //
    // With margin weighting, the root split must move from x1 to x0.
    let linear = LinearLayer::new(
        arr2(&[[2.0_f32, 1.0_f32], [-4.0_f32, 5.0_f32]]),
        Some(arr1(&[-0.1_f32, 2.01_f32])),
    )
    .expect("two-output linear layer should build");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");
    graph
}

#[ntest::timeout(5000)]
#[test]
fn test_verify_graph_input_split_non_finite_bounds_returns_unknown_unsplittable() {
    let graph = single_input_identity_graph();
    let input = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .expect("infinite endpoint bounds should be allowed");
    let verifier = input_split_verifier();

    let result = verifier
        .verify_graph_input_split(&graph, &input, &[1.0], 0.0)
        .expect("graph input split should not error on unsplittable domain");

    match result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                reason.contains("Unsplittable domain"),
                "expected unsplittable reason, got: {reason}"
            );
        }
        other => panic!("expected Unknown for unsplittable non-finite bounds, got {other:?}"),
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_verify_graph_input_split_zero_width_bounds_returns_unknown_unsplittable() {
    let graph = single_input_identity_graph();
    let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[0.0_f32]).into_dyn())
        .expect("zero-width finite bounds are valid");
    let verifier = input_split_verifier();

    let result = verifier
        .verify_graph_input_split(&graph, &input, &[1.0], 0.0)
        .expect("graph input split should not error on zero-width domain");

    match result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                reason.contains("Unsplittable domain"),
                "expected unsplittable reason, got: {reason}"
            );
        }
        other => panic!("expected Unknown for unsplittable zero-width bounds, got {other:?}"),
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_verify_graph_input_split_empty_input_bounds_returns_unknown_unsplittable() {
    let graph = empty_input_bias_only_graph();
    let input = BoundedTensor::new(
        Array1::<f32>::from_vec(Vec::new()).into_dyn(),
        Array1::<f32>::from_vec(Vec::new()).into_dyn(),
    )
    .expect("empty finite bounds should be valid");
    let verifier = input_split_verifier();

    let result = verifier
        .verify_graph_input_split(&graph, &input, &[1.0], 0.0)
        .expect("graph input split should not error on empty input bounds");

    match result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                reason.contains("Unsplittable domain"),
                "expected unsplittable reason, got: {reason}"
            );
        }
        other => panic!("expected Unknown for unsplittable empty bounds, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_graph_input_split_margin_weight_changes_split_1074() {
    let graph = sb_margin_multi_objective_graph();
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("finite bounds");
    let objectives = vec![vec![1.0_f32, 0.0_f32], vec![0.0_f32, 1.0_f32]];
    let thresholds = [0.0_f32, 0.0_f32];

    let base_config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 8,
        max_depth: 1,
        timeout: Duration::from_secs(1),
        ..Default::default()
    };
    let verifier_with_margin = BetaCrownVerifier::new(BetaCrownConfig {
        input_split_sb_margin_weight: 1.0,
        ..base_config.clone()
    });
    let verifier_without_margin = BetaCrownVerifier::new(BetaCrownConfig {
        input_split_sb_margin_weight: 0.0,
        ..base_config
    });

    let with_margin = verifier_with_margin
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("margin-weighted multi-objective graph input split should not error");
    let without_margin = verifier_without_margin
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("unweighted multi-objective graph input split should not error");

    assert!(
        matches!(with_margin.result, BabVerificationStatus::Verified),
        "margin-weighted SB should split x0 and verify both children, got {:?}",
        with_margin.result
    );
    assert_eq!(
        with_margin.domains_explored, 1,
        "both child domains should verify immediately after the root split"
    );
    assert_eq!(
        with_margin.domains_verified, 2,
        "x0 split should verify the left child via objective 1 and the right child via objective 0"
    );

    assert!(
        matches!(without_margin.result, BabVerificationStatus::Unknown { .. }),
        "without margin weighting the verifier should keep splitting x1 and remain unresolved, got {:?}",
        without_margin.result
    );
    assert_eq!(
        without_margin.max_depth_reached, 1,
        "the unweighted path should leave one child unresolved at the depth limit"
    );
    assert_eq!(
        without_margin.domains_verified, 1,
        "x1 split should only verify the high-x1 child in this regression"
    );
}

// ---------------------------------------------------------------------------
// NaN handling regression: #1860 (R1/1776 finding)
// ---------------------------------------------------------------------------
//
// Original divergence (R1/1776): The graph path hard-errored on non-finite
// CROWN output via domain_priority() → Err(NumericalInstability), while the
// GPU path gracefully degraded to Unknown.
//
// Status: The graph path now gracefully degrades to Unknown, matching GPU.
// The tests below verify this resolved behavior.

/// Graph with extreme weights that overflow f32 during IBP fallback, producing
/// non-finite output bounds.
///
/// **Current behavior**: Returns `Ok(Unknown { ... })` — graceful degradation
/// matching the GPU path. The BaB loop exhausts the domain limit without
/// verifying any domains. The original #1860 divergence (graph hard-errored
/// while GPU degraded) has been resolved.
fn extreme_weight_overflow_graph() -> GraphNetwork {
    // Two linear layers with weights near f32 overflow: 1e20 × 1e20 = 1e40 > f32::MAX.
    // CROWN backward NaN guard fires → IBP fallback → IBP produces Inf bounds.
    let w1 = arr2(&[[1e20_f32, -1e20], [1e20, 1e20]]);
    let linear1 = LinearLayer::new(w1, None).expect("linear1");
    let w2 = arr2(&[[1e20_f32, 1e20]]);
    let linear2 = LinearLayer::new(w2, None).expect("linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_input_split_extreme_weights_degrades_gracefully_1860() {
    // Regression test for #1860: extreme weights that overflow f32 during
    // IBP propagation. The graph path now gracefully degrades to Unknown
    // (matching GPU path behavior), resolving the original #1860 divergence
    // where graph hard-errored while GPU returned Unknown.
    let graph = extreme_weight_overflow_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite bounds");
    let verifier = input_split_verifier();

    let result = verifier.verify_graph_input_split(&graph, &input, &[1.0], 0.0);

    // Current behavior: graph path gracefully degrades to Unknown (matching GPU path).
    // The original #1860 divergence was that graph hard-errored while GPU returned Unknown.
    // That divergence has been resolved — the graph path now also degrades gracefully.
    let bab_result = result
        .expect("graph input_split should gracefully degrade on extreme weights, not hard-error");
    match &bab_result.result {
        BabVerificationStatus::Unknown { .. } => {
            // Expected: BaB loop exhausts domain limit without verifying any domains.
            assert_eq!(
                bab_result.domains_verified, 0,
                "extreme weights should not produce verified domains"
            );
        }
        BabVerificationStatus::Verified => {
            panic!(
                "extreme overflow weights should not produce Verified — \
                 this indicates a soundness bug (Inf/NaN bounds accepted as valid)"
            );
        }
        other => {
            // Timeout, PotentialViolation — acceptable degradation outcomes.
            assert!(
                !matches!(other, BabVerificationStatus::Verified),
                "unexpected verified status with extreme weights: {other:?}"
            );
        }
    }
}

/// Graph with moderate weights that produce finite but very large bounds.
/// Ensures no panics on near-overflow weight combinations.
fn near_overflow_graph() -> GraphNetwork {
    // Weights below overflow threshold individually but IBP accumulation
    // across layers may produce very large values.
    let w1 = arr2(&[[1e10_f32, -1e10], [1e10, 1e10]]);
    let linear1 = LinearLayer::new(w1, None).expect("linear1");
    let w2 = arr2(&[[1e10_f32, 1e10]]);
    let linear2 = LinearLayer::new(w2, None).expect("linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_input_split_near_overflow_does_not_panic_1860() {
    // Near-overflow weights: 1e10 × 1e10 = 1e20 < f32::MAX ≈ 3.4e38.
    // CROWN should produce finite (but very large) bounds. The graph path
    // should handle this without panicking, even if bounds are extreme.
    let graph = near_overflow_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite bounds");
    let verifier = input_split_verifier();

    // Near-overflow weights (1e10 × 1e10 = 1e20 < f32::MAX ≈ 3.4e38) should
    // produce finite bounds through IBP. The graph path should handle this
    // without error, unlike the extreme_weights test above.
    let result = verifier.verify_graph_input_split(&graph, &input, &[1.0], 0.0);

    let bab_result = result.expect(
        "near-overflow weights (1e10) should not cause hard error — \
         accumulated IBP bounds (≈4e20) are well within f32 range",
    );
    // BaB loop exits at domains_explored >= max_domains.
    assert!(
        bab_result.domains_explored <= verifier.config.max_domains,
        "should respect domain limit: explored {} > max {}",
        bab_result.domains_explored,
        verifier.config.max_domains,
    );
}

// ---------------------------------------------------------------------------
// NaN threshold guard regression: #3646 (P1/1262 finding)
// ---------------------------------------------------------------------------
//
// The multi-objective path lacked NaN/Inf threshold guards that the
// single-objective path has in domain_is_verified_for_mode (beta_config.rs:538).
// NaN thresholds cause IEEE 754 comparisons to silently fail, making BaB run
// to exhaustion without concluding.

/// Multi-objective with NaN threshold must return NumericalInstability error,
/// not silently run BaB to exhaustion. Part of #3646.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_nan_threshold_returns_error_3646() {
    let graph = single_input_identity_graph();
    let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite bounds");
    let verifier = input_split_verifier();

    let objectives = vec![vec![1.0_f32]];
    let thresholds = [f32::NAN];

    let result = verifier.verify_graph_input_split_multi_objective_conjunctive(
        &graph,
        &input,
        &objectives,
        &thresholds,
        None,
        None,
    );

    match result {
        Err(ny_core::NyError::NumericalInstability(msg)) => {
            assert!(
                msg.contains("non-finite"),
                "error should mention non-finite threshold, got: {msg}"
            );
        }
        Err(other) => panic!("expected NumericalInstability for NaN threshold, got: {other}"),
        Ok(bab) => panic!(
            "NaN threshold should error, not return {:?} after {} domains",
            bab.result, bab.domains_explored
        ),
    }
}

/// Multi-objective with +Inf threshold must also return NumericalInstability.
/// Part of #3646.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_inf_threshold_returns_error_3646() {
    let graph = single_input_identity_graph();
    let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite bounds");
    let verifier = input_split_verifier();

    let objectives = vec![vec![1.0_f32]];
    let thresholds = [f32::INFINITY];

    let result = verifier.verify_graph_input_split_multi_objective_conjunctive(
        &graph,
        &input,
        &objectives,
        &thresholds,
        None,
        None,
    );

    match result {
        Err(ny_core::NyError::NumericalInstability(msg)) => {
            assert!(
                msg.contains("non-finite"),
                "error should mention non-finite threshold, got: {msg}"
            );
        }
        Err(other) => panic!("expected NumericalInstability for Inf threshold, got: {other}"),
        Ok(bab) => panic!(
            "Inf threshold should error, not return {:?} after {} domains",
            bab.result, bab.domains_explored
        ),
    }
}

/// Multi-objective with one finite and one NaN threshold: the NaN threshold
/// should be caught even when mixed with valid thresholds. Part of #3646.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_mixed_nan_threshold_returns_error_3646() {
    let linear =
        LinearLayer::new(arr2(&[[1.0_f32], [2.0_f32]]), None).expect("two-output linear layer");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");

    let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite bounds");
    let verifier = input_split_verifier();

    let objectives = vec![vec![1.0_f32], vec![2.0_f32]];
    let thresholds = [0.5_f32, f32::NAN]; // second threshold is NaN

    let result = verifier.verify_graph_input_split_multi_objective_conjunctive(
        &graph,
        &input,
        &objectives,
        &thresholds,
        None,
        None,
    );

    match result {
        Err(ny_core::NyError::NumericalInstability(msg)) => {
            assert!(
                msg.contains("threshold[1]"),
                "error should identify the NaN threshold index, got: {msg}"
            );
        }
        Err(other) => panic!("expected NumericalInstability for mixed NaN threshold, got: {other}"),
        Ok(bab) => panic!(
            "mixed NaN threshold should error, not return {:?}",
            bab.result
        ),
    }
}

// ---------------------------------------------------------------------------
// reorder_bab=true parity test: #3870
// ---------------------------------------------------------------------------
//
// W2 implemented reorder_bab in commit e670585b4 (#3870). The feature defers
// CROWN bounding to domain pop time, using parent bounds as priority estimates.
// P1 verified reorder_bab=false is unchanged via static analysis (commit 7bee17ac5),
// but no unit test exercises the reorder_bab=true code path.
//
// This test verifies behavioral parity: reorder_bab=true produces the same
// verification result as reorder_bab=false on a graph that requires splitting.

/// Graph with moderate weights requiring splitting to verify. Linear -> ReLU -> Linear,
/// 2 inputs -> 2 hidden -> 1 output. Input domain [-1, 1]^2 is too wide for root
/// bounds to verify, forcing BaB to split.
fn splittable_relu_graph() -> GraphNetwork {
    let w1 = arr2(&[[1.5_f32, -0.5], [-0.5, 1.5]]);
    let b1 = arr1(&[0.0_f32, 0.0]);
    let linear1 = LinearLayer::new(w1, Some(b1)).expect("linear1");

    let w2 = arr2(&[[1.0_f32, -1.0]]);
    let b2 = arr1(&[0.0_f32]);
    let linear2 = LinearLayer::new(w2, Some(b2)).expect("linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_reorder_bab_true_produces_equivalent_result_3870() {
    let graph = splittable_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite bounds");

    // Threshold chosen so root bounds don't verify immediately,
    // forcing BaB to split at least once.
    let threshold = -0.5;

    let default_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 8,
        batch_size: 1,
        timeout: Duration::from_secs(5),
        reorder_bab: false,
        ..Default::default()
    });

    let reorder_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 8,
        batch_size: 1,
        timeout: Duration::from_secs(5),
        reorder_bab: true,
        ..Default::default()
    });

    let default_result = default_verifier
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("default BaB should not error");
    let reorder_result = reorder_verifier
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("reorder BaB should not error");

    // Both paths must produce the same verification status.
    assert_eq!(
        std::mem::discriminant(&default_result.result),
        std::mem::discriminant(&reorder_result.result),
        "reorder_bab=true should produce same verification status as default: \
         default={:?}, reorder={:?}",
        default_result.result,
        reorder_result.result,
    );

    assert!(
        default_result.domains_explored >= 2,
        "control test must split at least once to exercise the non-root path, got {}",
        default_result.domains_explored,
    );
    assert!(
        reorder_result.domains_explored >= 2,
        "reorder_bab test must split at least once to exercise deferred bounding, got {}",
        reorder_result.domains_explored,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_reorder_bab_batched_queue_preserves_status_3870() {
    let graph = splittable_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite bounds");
    let threshold = -0.5;

    let scalar_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 8,
        batch_size: 1,
        timeout: Duration::from_secs(5),
        reorder_bab: true,
        ..Default::default()
    });
    let batched_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 8,
        batch_size: 4,
        timeout: Duration::from_secs(5),
        reorder_bab: true,
        ..Default::default()
    });

    let scalar_result = scalar_verifier
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("scalar reorder BaB should not error");
    let batched_result = batched_verifier
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("batched reorder BaB should not error");

    assert_eq!(
        std::mem::discriminant(&scalar_result.result),
        std::mem::discriminant(&batched_result.result),
        "batched reorder_bab should preserve verification status: scalar={:?}, batched={:?}",
        scalar_result.result,
        batched_result.result,
    );
    assert!(
        batched_result.domains_explored >= 2,
        "batched reorder_bab must still split at least once, got {}",
        batched_result.domains_explored,
    );
}

/// Reorder BaB on the identity graph should produce the same result as default.
/// The identity graph is verifiable at the root (no splitting needed), so this
/// tests the degenerate case where reorder_bab=true never creates needs_bounding
/// children.
#[ntest::timeout(10000)]
#[test]
fn test_reorder_bab_identity_graph_verifies_at_root_3870() {
    let graph = single_input_identity_graph();
    let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite bounds");

    // Identity graph: output = input ∈ [0, 1]. Threshold 2.0 means lower bound 0
    // is well below threshold, so the property min(f(x)) >= threshold is unverifiable.
    // Use threshold -1.0 so that root lower bound (0.0) >= -1.0 verifies immediately.
    let threshold = -1.0;

    let reorder_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 32,
        max_depth: 8,
        timeout: Duration::from_secs(1),
        reorder_bab: true,
        ..Default::default()
    });

    let result = reorder_verifier
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("reorder BaB on identity graph should not error");

    match &result.result {
        BabVerificationStatus::Verified => {
            // Root domain verified immediately — no splitting needed.
            assert_eq!(
                result.domains_verified, 1,
                "identity graph should verify exactly the root domain"
            );
        }
        other => panic!(
            "identity graph with easy threshold should verify, got {:?}",
            other
        ),
    }
}

// ---------------------------------------------------------------------------
// adv_check PGD probe during BaB: #3870
// ---------------------------------------------------------------------------
//
// adv_check runs a lightweight PGD probe on the current domain during BaB
// to detect SAT counterexamples early. This test verifies the probe fires
// and can find violations.

/// adv_check should find a concrete counterexample before BaB splits when the
/// root bounds are inconclusive but the property is genuinely violated.
#[ntest::timeout(10000)]
#[test]
fn test_adv_check_finds_root_counterexample_before_bab_3870() {
    let graph = single_input_identity_graph();
    let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite bounds");

    // Root bounds are [0, 1], so threshold=0.5 is inconclusive:
    // - not Verified because lower=0.0 is not > 0.5
    // - not PotentialViolation because upper=1.0 is not < 0.5
    //
    // With adv_check=0, the PGD probe runs before domains_explored is
    // incremented and should deterministically descend to x=0.0 on the
    // identity graph, producing an immediate PotentialViolation.
    let threshold = 0.5;

    let without_adv_check = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 2048,
        max_depth: 12,
        timeout: Duration::from_secs(2),
        adv_check: -1,
        ..Default::default()
    });
    let with_adv_check = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 2048,
        max_depth: 12,
        timeout: Duration::from_secs(2),
        adv_check: 0,
        ..Default::default()
    });

    let without_adv_result = without_adv_check
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("adv_check=-1 should not cause errors");
    let with_adv_result = with_adv_check
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("adv_check=0 should not cause errors");

    assert!(
        matches!(
            with_adv_result.result,
            BabVerificationStatus::PotentialViolation { .. }
        ),
        "adv_check should find a concrete root-domain counterexample, got {:?}",
        with_adv_result.result
    );
    assert_eq!(
        with_adv_result.domains_explored, 0,
        "adv_check should return before BaB counts the root domain"
    );

    assert!(
        matches!(
            without_adv_result.result,
            BabVerificationStatus::PotentialViolation { .. }
        ),
        "the same property should still be violated without adv_check, got {:?}",
        without_adv_result.result
    );
    assert!(
        without_adv_result.domains_explored >= 1,
        "without adv_check the verifier should need to process at least the root domain, got {}",
        without_adv_result.domains_explored
    );
}

/// adv_check=-1 (disabled) should produce identical results to no adv_check.
#[ntest::timeout(10000)]
#[test]
fn test_adv_check_disabled_produces_same_result_3870() {
    let graph = splittable_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite bounds");

    let threshold = -0.5;

    let no_adv_check = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 8,
        timeout: Duration::from_secs(5),
        adv_check: -1,
        ..Default::default()
    });

    let with_adv_check = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 8,
        timeout: Duration::from_secs(5),
        adv_check: 0,
        ..Default::default()
    });

    let result_disabled = no_adv_check
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("adv_check=-1 should not error");
    let result_enabled = with_adv_check
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("adv_check=0 should not error");

    // Both should verify (or both Unknown), but adv_check must not change
    // the soundness of the result.
    assert_eq!(
        std::mem::discriminant(&result_disabled.result),
        std::mem::discriminant(&result_enabled.result),
        "adv_check should not change verification outcome: \
         disabled={:?}, enabled={:?}",
        result_disabled.result,
        result_enabled.result,
    );
}

/// Per-sub-domain α refinement (input_split_alpha_iteration > 0) must stay sound
/// end-to-end with α-CROWN active: on a genuinely-verifiable property it must
/// reach `Verified` and never fabricate a `PotentialViolation`. The
/// `splittable_relu_graph` output range over [-1,1]^2 is exactly [-2, 2] (verified
/// by brute force), so threshold -2.5 is provably satisfied (output > -2.5 for all
/// inputs). The warm path's tighter per-sub-domain bounds may only resolve it with
/// FEWER domains than the frozen default — never producing an unsound verdict.
#[ntest::timeout(20000)]
#[test]
fn test_input_split_alpha_refinement_stays_sound_on_verifiable_property() {
    let graph = splittable_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite bounds");
    // True min output is -2.0, so output > -2.5 holds everywhere → Verified.
    let threshold = -2.5;

    let base = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: true,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 256,
        max_depth: 12,
        batch_size: 1,
        timeout: Duration::from_secs(10),
        reorder_bab: false,
        ..Default::default()
    };

    // Frozen default: single frozen-alpha pass per domain (iterations = 0).
    let frozen_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        input_split_alpha_iteration: 0,
        ..base.clone()
    });
    // Warm path: 5 warm-started SPSA iterations per sub-domain at lr 0.05.
    let warm_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        input_split_alpha_iteration: 5,
        input_split_lr_alpha: 0.05,
        ..base
    });

    let frozen_result = frozen_verifier
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("frozen-alpha input split should not error");
    let warm_result = warm_verifier
        .verify_graph_input_split(&graph, &input, &[1.0], threshold)
        .expect("warm-start input split should not error");

    // Soundness: the warm path must NEVER report a violation on a property that
    // is genuinely true. (A false PotentialViolation would be an unsound bug.)
    assert!(
        !matches!(
            warm_result.result,
            BabVerificationStatus::PotentialViolation { .. }
        ),
        "warm-start refinement fabricated a violation on a verifiable property: {:?}",
        warm_result.result,
    );
    // Both paths should verify this provable property; assert the warm path keeps
    // the frozen verdict (status discriminant matches).
    assert_eq!(
        std::mem::discriminant(&frozen_result.result),
        std::mem::discriminant(&warm_result.result),
        "per-sub-domain α refinement changed the verdict on a verifiable property: \
         frozen={:?}, warm={:?}",
        frozen_result.result,
        warm_result.result,
    );
    // Tighter per-domain bounds should prune at least as aggressively: warm must
    // not need MORE domains than the frozen default.
    assert!(
        warm_result.domains_explored <= frozen_result.domains_explored,
        "warm-start refinement should not increase domain count: warm={}, frozen={}",
        warm_result.domains_explored,
        frozen_result.domains_explored,
    );
}
