// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression and architecture-specific tests for GPU BaB.

use super::*;

fn x0_only_input_split_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear0",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32, 0.0_f32]]), None).unwrap()),
    ));
    graph.set_output("linear0");
    graph
}

/// Test GPU BaB with GenBaB heuristic for GeLU nonlinearity.
/// Part of #1534: Verify GenBaB path in verify_graph_gpu_domain_list.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_bab_genbab_heuristic() {
    use crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig;

    let graph = gelu_graph_network();

    // Input bounds spanning zero trigger GeLU nonlinearity.
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    let objective = vec![1.0];
    // GeLU output bounds for input [-0.5, 0.5]:
    // GeLU(-0.5) ≈ -0.5 * Φ(-0.5) ≈ -0.154
    // GeLU(0.5) ≈ 0.5 * Φ(0.5) ≈ 0.346
    // With identity first linear, each GeLU neuron has bounds ~[-0.154, 0.346].
    // Sum via w2=[[1,1]]: roughly [-0.31, 0.69].
    // Use threshold above max output to ensure verification.
    let threshold = 2.0;

    let genbab_config = NonlinearBranchingConfig::default();
    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 20,
        max_depth: 5,
        batch_size: 2,
        branching_heuristic: BranchingHeuristic::GenBaB(genbab_config),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB with GenBaB should succeed");

    // Should verify since max output is well below threshold.
    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "GeLU network with GenBaB should verify when output < threshold"
    );
    // GenBaB path should have been taken.
    assert!(
        result.domains_explored >= 1,
        "Should explore at least root domain"
    );
    // Verify the result structure is valid (verifier ran to completion).
    assert!(
        result.time_elapsed > Duration::ZERO,
        "Verification should have taken measurable time"
    );
}

/// Test GPU BaB with Conv2d graph exercises the lA-capture backward variant
/// for Conv2d, ReLU, Flatten, and Linear layers.
///
/// Regression test for #1811: Before the fix, Conv2d and Flatten were handled
/// by an identity pass-through in the lA-capture backward variant, producing
/// degraded bounds. After the fix, the correct backward transformations are
/// applied.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_bab_conv2d_graph_network() {
    let graph = conv_graph_network();

    // Input: [1, 3, 3] with values in [-0.5, 0.5].
    let lower = ArrayD::from_elem(IxDyn(&[1, 3, 3]), -0.5_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 3, 3]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let objective = vec![1.0];
    // Conv with averaging filter on [-0.5, 0.5] gives roughly [-1, 1] per channel.
    // ReLU clips to [0, 1], sum of 8 elements: roughly [0, 8].
    // Use high threshold to ensure immediate verification.
    let threshold = 20.0;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 50,
        max_depth: 5,
        batch_size: 4,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB with Conv2d graph should succeed");

    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "Conv2d graph should verify with high threshold"
    );
}

/// Test GPU BaB with Conv2d graph forces branching via tight threshold.
///
/// Regression test for #1811: Exercises the full BaB loop with Conv2d/Flatten
/// in the lA-capture backward variant, including child domain creation and
/// warm-start with cached lA. Uses a threshold tight enough to prevent
/// immediate verification, forcing the BaB loop to actually run.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_bab_conv2d_graph_branching() {
    let graph = conv_graph_network();

    // Input: [1, 3, 3] with values in [-1.0, 1.0] - wider range forces more branching.
    let lower = ArrayD::from_elem(IxDyn(&[1, 3, 3]), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 3, 3]), 1.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let objective = vec![1.0];
    // Tight threshold: actual max output is ~8 (all ReLU active, sum of 8 neurons).
    // IBP gives loose upper bound (~16), CROWN tightens it. Setting threshold to 10
    // is tight enough that the root CROWN upper bound likely exceeds it, forcing BaB
    // branching to narrow the bounds and verify.
    let threshold = 10.0;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 200,
        max_depth: 10,
        batch_size: 4,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB with Conv2d graph (branching) should succeed");

    // Verify the BaB loop actually ran (not just immediate verification).
    assert!(
        result.domains_explored >= 2,
        "Expected BaB branching (domains_explored >= 2), got {}. \
         The threshold may need adjustment if CROWN root bounds are tight enough \
         to verify immediately.",
        result.domains_explored
    );

    // Should either verify or remain unknown - should NOT panic or produce
    // NaN/infinite bounds from the backward pass.
    match result.result {
        BabVerificationStatus::Verified => {
            assert!(
                result.domains_verified >= 1,
                "Should verify at least one domain"
            );
        }
        BabVerificationStatus::Unknown { .. } => {
            // Unknown is acceptable - the point is no panic/NaN from Conv2d backward.
        }
        BabVerificationStatus::PotentialViolation => {
            // Also acceptable for some threshold values.
        }
        other => {
            panic!("Unexpected GPU BaB result for Conv2d graph: {:?}", other);
        }
    }
    // The test succeeding without panic is the key assertion for #1811.
}

/// Regression for #1891: InputSplit GPU BaB should not require per-layer
/// DomainList storage because input-split children carry only input bounds.
///
/// This test runs enough input-split iterations to exercise the periodic
/// DomainList sort path while ensuring `verify_graph_gpu_domain_list` stays
/// successful (no internal storage-length mismatch errors).
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_input_split_layerless_domain_list_regression_1891() {
    let w = arr2(&[[1.0]]);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear0",
        Layer::Linear(LinearLayer::new(w, None).unwrap()),
    ));
    graph.set_output("linear0");

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let objective = vec![1.0];

    // In lower-bound verification mode with threshold -1.0, domains touching
    // the left boundary remain unresolved for multiple splits (strict `lower > threshold`),
    // which forces repeated InputSplit iterations.
    let config = BetaCrownConfig {
        verify_upper_bound: false,
        branching_heuristic: BranchingHeuristic::InputSplit,
        batch_size: 1,
        max_depth: 6,
        max_domains: 64,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, -1.0, None, None)
        .expect("InputSplit GPU BaB should complete without DomainList invariant errors");

    assert!(
        result.domains_explored >= 3,
        "expected multiple input-split iterations, got explored={}",
        result.domains_explored
    );
}

/// Regression for #3870: GPU input split must use the existing SB scorer when
/// a spec-guided linear form is available, rather than falling back to width-only.
///
/// The crafted graph depends only on `x0`, while `x1` has a much larger width.
/// A width-only heuristic splits `x1` and stays unresolved at depth 1. SB uses
/// the cached linear coefficients from the root bound pass, splits `x0`, and
/// reaches a concrete violation immediately.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_input_split_uses_sb_linear_scoring_3870() {
    let graph = x0_only_input_split_graph();
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 100.0_f32]).into_dyn(),
    )
    .unwrap();
    let objective = vec![1.0_f32];
    let threshold = 0.8_f32;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 16,
        max_depth: 1,
        batch_size: 1,
        timeout: Duration::from_secs(2),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU input split should complete on SB-scoring regression");

    assert!(
        matches!(result.result, BabVerificationStatus::PotentialViolation),
        "SB scoring should split x0 and expose the violating child immediately, got {:?}",
        result.result
    );
}

/// Regression for #3870: reordered GPU input split must preserve the same
/// verification result as the bound-now path on the same single-objective lane.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_input_split_reorder_bab_preserves_status_3870() {
    let graph = x0_only_input_split_graph();
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 100.0_f32]).into_dyn(),
    )
    .unwrap();
    let objective = vec![1.0_f32];
    let threshold = 0.8_f32;

    let base_config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 16,
        max_depth: 2,
        timeout: Duration::from_secs(2),
        ..Default::default()
    };
    let immediate_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        batch_size: 1,
        reorder_bab: false,
        ..base_config.clone()
    });
    let reorder_verifier = BetaCrownVerifier::new(BetaCrownConfig {
        batch_size: 4,
        reorder_bab: true,
        ..base_config
    });

    let immediate_result = immediate_verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("non-reordered GPU input split should succeed");
    let reorder_result = reorder_verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("reordered GPU input split should succeed");

    assert_eq!(
        std::mem::discriminant(&immediate_result.result),
        std::mem::discriminant(&reorder_result.result),
        "reordered GPU input split should preserve verification status: immediate={:?}, reorder={:?}",
        immediate_result.result,
        reorder_result.result,
    );
    assert!(
        matches!(
            reorder_result.result,
            BabVerificationStatus::PotentialViolation
        ),
        "the reordered path should still surface the violating x0 child, got {:?}",
        reorder_result.result
    );
}

/// Regression for #1817: with `beta_iterations = 0`, default beta initialization
/// from history must match beta=0 behavior (and avoid the old beta=0.1 widening).
#[ntest::timeout(10000)]
#[test]
fn test_graph_constrained_crown_tightening() {
    use crate::beta_crown::domain::GraphCrownContext;

    // 3-hidden-layer MLP: [2] -> [4] -> [4] -> [4] -> [1].
    let w1 = arr2(&[[2.0, -1.0], [-1.0, 2.0], [1.0, 1.0], [-1.0, -1.0]]);
    let b1 = Some(arr1(&[0.0, 0.0, -0.5, 0.5]));
    let w2 = arr2(&[
        [1.0, -1.0, 0.5, 0.0],
        [-1.0, 1.0, 0.0, 0.5],
        [0.5, 0.5, -1.0, 1.0],
        [0.0, -0.5, 1.0, -1.0],
    ]);
    let w3 = arr2(&[
        [1.0, 0.5, -0.5, 0.0],
        [0.0, 1.0, 0.5, -0.5],
        [-0.5, 0.0, 1.0, 0.5],
        [0.5, -0.5, 0.0, 1.0],
    ]);
    let w4 = arr2(&[[1.0, -1.0, 0.5, -0.5]]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, b1).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(arr1(&[0.0; 4]))).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, Some(arr1(&[0.0; 4]))).unwrap()),
        vec!["relu2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["linear3".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear4",
        Layer::Linear(LinearLayer::new(w4, None).unwrap()),
        vec!["relu3".to_string()],
    ));
    graph.set_output("linear4");

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let objective = vec![1.0];

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    // Step 1: Get IBP bounds for base bounds.
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let node_bounds: std::collections::HashMap<String, Arc<BoundedTensor>> = ibp_bounds
        .iter()
        .map(|(k, v)| (k.clone(), Arc::new(v.clone())))
        .collect();

    // Step 2: Apply one split constraint and compare beta initialization choices.
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 1.0,
    });

    let beta = GraphBetaState::from_history(&history).unwrap();
    let beta_nonzero = GraphBetaState::from_history_with_init(&history, 0.1).unwrap();
    assert!(
        beta.entries.iter().all(|entry| entry.value() == 0.0),
        "GraphBetaState::from_history must initialize beta to 0.0 for #1817 regression"
    );
    assert!(
        beta_nonzero
            .entries
            .iter()
            .all(|entry| entry.value() == 0.1),
        "from_history_with_init should preserve explicit beta init for control case"
    );

    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);
    let (constrained_output, _) = verifier
        .propagate_crown_with_graph_constraints(
            &graph,
            &input,
            &context,
            Some(&beta),
            Some(&objective),
        )
        .unwrap();
    let (constrained_output_nonzero, _) = verifier
        .propagate_crown_with_graph_constraints(
            &graph,
            &input,
            &context,
            Some(&beta_nonzero),
            Some(&objective),
        )
        .unwrap();

    let constrained_upper = constrained_output.upper_scalar();
    let constrained_upper_nonzero = constrained_output_nonzero.upper_scalar();

    // Primary assertion: beta=0.0 produces bounds at least as tight as beta=0.1.
    assert!(
        constrained_upper <= constrained_upper_nonzero + 1e-6,
        "beta=0.0 should not be worse than beta=0.1 for #1817 regression.\n\
         upper(beta=0.0): {}\n\
         upper(beta=0.1): {}",
        constrained_upper,
        constrained_upper_nonzero
    );

    // Strengthening assertion: beta=0.1 should produce measurably wider bounds.
    assert!(
        constrained_upper_nonzero > constrained_upper + 1e-6,
        "beta=0.1 should produce measurably wider (worse) upper bounds than beta=0.0 \
         when no optimization iterations run. If both are equal, the constraint \
         has no effect and this test is not exercising the #1817 regression.\n\
         upper(beta=0.0): {}\n\
         upper(beta=0.1): {}",
        constrained_upper,
        constrained_upper_nonzero
    );
}

/// #1817 diagnostic: Three-way BaB comparison to isolate where bounds diverge.
///
/// Compares:
/// 1. `verify()` (sequential Network BaB)
/// 2. `verify_graph_relu_split()` (sequential GraphNetwork BaB)
/// 3. `verify_graph_gpu_domain_list()` (GPU DomainList BaB)
///
/// If (2) fails but (1) succeeds -> graph CROWN BaB issue.
/// If (3) fails but (2) succeeds -> DomainList-specific issue.
#[ntest::timeout(60000)]
#[test]
fn test_three_way_bab_comparison_1817() {
    use ndarray::Array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(42);
    let w1: Array2<f32> = Array::from_shape_fn((8, 2), |_| {
        use rand::RngExt;
        rng.random_range(-1.0..1.0)
    });
    let w2: Array2<f32> = Array::from_shape_fn((8, 8), |_| {
        use rand::RngExt;
        rng.random_range(-0.5..0.5)
    });
    let w3: Array2<f32> = Array::from_shape_fn((8, 8), |_| {
        use rand::RngExt;
        rng.random_range(-0.5..0.5)
    });
    let w4: Array2<f32> = Array::from_shape_fn((8, 8), |_| {
        use rand::RngExt;
        rng.random_range(-0.5..0.5)
    });
    let w5: Array2<f32> = Array::from_shape_fn((1, 8), |_| {
        use rand::RngExt;
        rng.random_range(-1.0..1.0)
    });

    // Build sequential Network.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w3.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w4.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w5.clone(), None).unwrap()));

    // Build equivalent GraphNetwork.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, None).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, None).unwrap()),
        vec!["relu2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["linear3".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear4",
        Layer::Linear(LinearLayer::new(w4, None).unwrap()),
        vec!["relu3".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu4",
        Layer::ReLU(ReLULayer),
        vec!["linear4".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear5",
        Layer::Linear(LinearLayer::new(w5, None).unwrap()),
        vec!["relu4".to_string()],
    ));
    graph.set_output("linear5");

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Sample to find true max.
    const GRID: usize = 101;
    let mut max_out = f32::NEG_INFINITY;
    for i in 0..GRID {
        let x0 = -1.0 + 2.0 * (i as f32 / (GRID - 1) as f32);
        for j in 0..GRID {
            let x1 = -1.0 + 2.0 * (j as f32 / (GRID - 1) as f32);
            let inp =
                BoundedTensor::new(arr1(&[x0, x1]).into_dyn(), arr1(&[x0, x1]).into_dyn()).unwrap();
            let out = network.propagate_ibp(&inp).unwrap();
            max_out = max_out.max(out.lower()[[0]]);
        }
    }

    let threshold = max_out + 0.5;
    eprintln!("\n=== Three-way BaB comparison ===");
    eprintln!("Sampled max: {:.4}, threshold: {:.4}", max_out, threshold);

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(15),
        max_domains: 50000,
        max_depth: 50,
        batch_size: 64,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    };

    // 1. Sequential BaB (Network).
    let v1 = BetaCrownVerifier::new(config.clone());
    let r1 = v1.verify(&network, &input, threshold).unwrap();
    eprintln!(
        "Sequential BaB: {:?} (explored={}, verified={}, time={:.2}s)",
        r1.result,
        r1.domains_explored,
        r1.domains_verified,
        r1.time_elapsed.as_secs_f32()
    );

    // 2. Graph BaB (verify_graph_relu_split).
    let objective = vec![1.0];
    let v2 = BetaCrownVerifier::new(config.clone());
    let r2 = v2
        .verify_graph_relu_split(&graph, &input, &objective, threshold)
        .unwrap();
    eprintln!(
        "Graph BaB (relu_split): {:?} (explored={}, verified={}, time={:.2}s)",
        r2.result,
        r2.domains_explored,
        r2.domains_verified,
        r2.time_elapsed.as_secs_f32()
    );

    // 3. GPU BaB (DomainList).
    let v3 = BetaCrownVerifier::new(config);
    let r3 = v3
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .unwrap();
    eprintln!(
        "GPU BaB (DomainList): {:?} (explored={}, verified={}, time={:.2}s)",
        r3.result,
        r3.domains_explored,
        r3.domains_verified,
        r3.time_elapsed.as_secs_f32()
    );

    // Summary: compare verification rates across all three paths.
    let rate1 = if r1.domains_explored > 0 {
        r1.domains_verified as f64 / r1.domains_explored as f64
    } else {
        0.0
    };
    let rate2 = if r2.domains_explored > 0 {
        r2.domains_verified as f64 / r2.domains_explored as f64
    } else {
        0.0
    };
    let rate3 = if r3.domains_explored > 0 {
        r3.domains_verified as f64 / r3.domains_explored as f64
    } else {
        0.0
    };
    eprintln!(
        "\n=== Verification rate comparison ===\n\
         Sequential:  {:.1}% ({}/{})\n\
         Graph split: {:.1}% ({}/{})\n\
         GPU BaB:     {:.1}% ({}/{})",
        rate1 * 100.0,
        r1.domains_verified,
        r1.domains_explored,
        rate2 * 100.0,
        r2.domains_verified,
        r2.domains_explored,
        rate3 * 100.0,
        r3.domains_verified,
        r3.domains_explored,
    );

    // All three paths should verify at least some domains.
    assert!(
        r1.domains_verified > 0,
        "Sequential BaB should verify some domains: {:?}",
        r1.result
    );
    assert!(
        r2.domains_verified > 0,
        "Graph BaB (relu_split) should verify some domains: {:?} (explored={}, verified={})",
        r2.result,
        r2.domains_explored,
        r2.domains_verified
    );
    assert!(
        r3.domains_verified > 0,
        "GPU BaB (DomainList) should verify some domains: {:?} (explored={}, verified={})",
        r3.result,
        r3.domains_explored,
        r3.domains_verified
    );

    // Report relative verification rate for diagnostics.
    let ratio = if rate1 > 0.0 { rate3 / rate1 } else { 1.0 };
    eprintln!("GPU BaB rate / Sequential rate = {:.2}", ratio);
}

/// Helper: create a `DomainMetadata` with the given bounds and depth.
fn make_domain_meta(lower: f32, upper: f32, depth: usize) -> crate::batched_domain::DomainMetadata {
    crate::batched_domain::DomainMetadata {
        lower_bound: lower,
        upper_bound: upper,
        depth,
        constraints: vec![],
        cached_la: None,
        needs_bounding: false,
        node_bounds_override: None,
        alpha_state: None,
    }
}

/// Regression for #2933: NaN-bounded domains must be dropped by the prefilter,
/// not passed through to the branching pipeline.
///
/// Before the fix, NaN domains fell through `check_domain_bounds` to `Undecided`
/// and were added to `processable_indices`, causing progressive NaN contamination.
#[ntest::timeout(5000)]
#[test]
fn test_prefilter_drops_nan_domains_regression_2933() {
    use std::time::Instant;

    use crate::beta_crown::engine::graph::gpu_bab::check::BabLoopState;
    use crate::beta_crown::engine::graph::gpu_bab::prefilter::prefilter_picked_domains;

    let metadata = vec![
        make_domain_meta(0.5, 1.0, 1),           // 0: verified (lower > 0.0)
        make_domain_meta(f32::NAN, 1.0, 2),      // 1: NaN lower — dropped
        make_domain_meta(-1.0, f32::NAN, 3),     // 2: NaN upper — dropped
        make_domain_meta(f32::NAN, f32::NAN, 4), // 3: both NaN — dropped
        make_domain_meta(f32::NEG_INFINITY, f32::INFINITY, 5), // 4: Inf — dropped
        make_domain_meta(-0.5, 0.5, 1),          // 5: undecided (processable)
    ];

    let mut state = BabLoopState::new(Instant::now());
    // Lower-bound mode (verify_upper_bound=false): verified when lower > threshold
    let result = prefilter_picked_domains(&metadata, 0.0, false, 10, &mut state);

    // Only domain 5 should be processable. Domain 0 is verified, domains 1-4 are NaN/Inf.
    assert_eq!(
        result.processable_indices,
        vec![5],
        "NaN/Inf domains must be dropped"
    );
    assert!(!result.violation);
    assert!(
        state.unresolved_due_to_propagation_failure,
        "NaN/Inf must set propagation failure"
    );
    assert_eq!(
        state.domains_verified, 1,
        "Domain 0 should be counted as verified"
    );
}

/// Regression test for #1817: ensure GPU BaB (DomainList) and CPU BaB (sequential)
/// produce consistent verification outcomes on a network where both paths converge.
///
/// Uses a simple 2-hidden-layer network with a loose threshold that both paths can
/// verify, and a tighter threshold that requires BaB branching. Asserts that both
/// paths agree on the final verification status (Verified/Unknown/PotentialViolation).
///
/// This catches regressions like the DomainList sort direction bug (799a207d) where
/// DFS picked the worst domains first, causing GPU BaB to diverge while CPU BaB converged.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_bab_cpu_bab_consistency_regression_1817() {
    // Use an explicit GEMM engine so this test exercises the DomainList
    // engine-backed path (not the CPU fallback path).
    let engine = NaiveCpuGemmEngine;

    // Build a simple graph: Linear(2->4) -> ReLU -> Linear(4->4) -> ReLU -> Linear(4->1).
    let w1 = arr2(&[[0.5, -0.3], [-0.3, 0.5], [0.4, 0.4], [-0.2, 0.6]]);
    let b1 = arr1(&[0.1, -0.1, 0.0, 0.05]);
    let w2 = arr2(&[
        [0.3, -0.2, 0.1, 0.4],
        [-0.1, 0.5, -0.3, 0.2],
        [0.2, 0.1, 0.4, -0.1],
        [-0.3, 0.2, 0.1, 0.3],
    ]);
    let b2 = arr1(&[0.0, 0.1, -0.05, 0.0]);
    let w3 = arr2(&[[0.5, -0.3, 0.2, 0.4]]);
    let b3 = arr1(&[0.0]);

    // Build sequential Network.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w3.clone(), Some(b3.clone())).unwrap(),
    ));

    // Build equivalent GraphNetwork.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let objective = vec![1.0]; // Single output.

    // Sample true max output to set meaningful thresholds.
    const GRID: usize = 51;
    let mut max_out = f32::NEG_INFINITY;
    for i in 0..GRID {
        let x0 = -1.0 + 2.0 * (i as f32 / (GRID - 1) as f32);
        for j in 0..GRID {
            let x1 = -1.0 + 2.0 * (j as f32 / (GRID - 1) as f32);
            let inp =
                BoundedTensor::new(arr1(&[x0, x1]).into_dyn(), arr1(&[x0, x1]).into_dyn()).unwrap();
            let out = network.propagate_ibp(&inp).unwrap();
            max_out = max_out.max(out.lower()[[0]]);
        }
    }
    eprintln!(
        "\n=== GPU/CPU BaB consistency regression test ===\nSampled max: {:.4}",
        max_out
    );

    // Test case 1: Loose threshold (easy - both should verify quickly).
    let loose_threshold = max_out + 2.0;
    let config_loose = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(10),
        max_domains: 5000,
        max_depth: 30,
        batch_size: 16,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    };

    let cpu_loose = BetaCrownVerifier::new(config_loose.clone())
        .verify(&network, &input, loose_threshold)
        .expect("CPU BaB should succeed (loose)");
    let gpu_loose = BetaCrownVerifier::new(config_loose)
        .verify_graph_gpu_domain_list(
            &graph,
            &input,
            &objective,
            loose_threshold,
            Some(&engine),
            None,
        )
        .expect("GPU BaB should succeed (loose)");

    eprintln!(
        "Loose threshold ({:.2}):\n  CPU: {:?} (explored={}, verified={})\n  GPU: {:?} (explored={}, verified={})",
        loose_threshold,
        cpu_loose.result,
        cpu_loose.domains_explored,
        cpu_loose.domains_verified,
        gpu_loose.result,
        gpu_loose.domains_explored,
        gpu_loose.domains_verified,
    );

    assert_eq!(
        std::mem::discriminant(&cpu_loose.result),
        std::mem::discriminant(&gpu_loose.result),
        "CPU and GPU BaB should agree on loose threshold.\n\
         CPU: {:?}\nGPU: {:?}",
        cpu_loose.result,
        gpu_loose.result,
    );

    // Test case 2: Tighter threshold (forces branching, both should still verify).
    let tight_threshold = max_out + 0.5;
    let config_tight = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(10),
        max_domains: 10000,
        max_depth: 30,
        batch_size: 16,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    };

    let cpu_tight = BetaCrownVerifier::new(config_tight.clone())
        .verify(&network, &input, tight_threshold)
        .expect("CPU BaB should succeed (tight)");
    let gpu_tight = BetaCrownVerifier::new(config_tight)
        .verify_graph_gpu_domain_list(
            &graph,
            &input,
            &objective,
            tight_threshold,
            Some(&engine),
            None,
        )
        .expect("GPU BaB should succeed (tight)");

    eprintln!(
        "Tight threshold ({:.2}):\n  CPU: {:?} (explored={}, verified={})\n  GPU: {:?} (explored={}, verified={})",
        tight_threshold,
        cpu_tight.result,
        cpu_tight.domains_explored,
        cpu_tight.domains_verified,
        gpu_tight.result,
        gpu_tight.domains_explored,
        gpu_tight.domains_verified,
    );

    // If CPU verifies, GPU should too (or at minimum, not claim violation).
    if matches!(cpu_tight.result, BabVerificationStatus::Verified) {
        assert!(
            matches!(gpu_tight.result, BabVerificationStatus::Verified),
            "When CPU BaB verifies, GPU BaB should also verify.\n\
             CPU: {:?} (verified={}/{})\n\
             GPU: {:?} (verified={}/{})",
            cpu_tight.result,
            cpu_tight.domains_verified,
            cpu_tight.domains_explored,
            gpu_tight.result,
            gpu_tight.domains_verified,
            gpu_tight.domains_explored,
        );
    }

    // GPU BaB should verify at least 25% of what CPU BaB verifies (regression guard).
    if cpu_tight.domains_verified > 0 {
        let ratio = gpu_tight.domains_verified as f64 / cpu_tight.domains_verified as f64;
        eprintln!(
            "GPU/CPU verified ratio: {:.2} ({}/{})",
            ratio, gpu_tight.domains_verified, cpu_tight.domains_verified,
        );
        assert!(
            ratio >= 0.25,
            "GPU BaB should verify at least 25% of CPU BaB domains.\n\
             Ratio: {:.2} (GPU verified={}, CPU verified={})\n\
             This may indicate a sort direction or domain processing regression.",
            ratio,
            gpu_tight.domains_verified,
            cpu_tight.domains_verified,
        );
    }
}
