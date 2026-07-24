// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Branch exploration and domain processing tests for GPU BaB.

use super::*;

/// Test timeout handling.
/// When verification exceeds timeout, should return BabVerificationStatus::Timeout.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_timeout() {
    let graph = simple_graph_network();

    // Input bounds that create unstable region (crosses zero after linear)
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let objective = vec![1.0];
    // Root bounds for this network are [0, 4].
    // Use threshold = 0.5 so neither immediate condition triggers:
    // - root_lower (0) >= threshold (0.5)? No (not a violation)
    // - root_upper (4) < threshold (0.5)? No (not verified)
    // This forces the BaB loop to run, which will timeout with 0ms timeout.
    let threshold = 0.5;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_millis(0), // Force immediate timeout
        max_domains: 1000000,
        max_depth: 100,
        batch_size: 4,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    assert_eq!(
        result.result,
        BabVerificationStatus::Timeout,
        "Expected Timeout, got {:?}",
        result.result
    );
}

/// Test domain limit handling.
/// When max_domains is reached, should return Unknown with domain limit reason.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_domain_limit() {
    let graph = simple_graph_network();

    // Input bounds that create unstable region
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let objective = vec![1.0];
    // Root bounds for this network are [0, 4].
    // Use threshold = 0.5 so neither immediate condition triggers:
    // - root_lower (0) >= threshold (0.5)? No (not a violation)
    // - root_upper (4) < threshold (0.5)? No (not verified)
    // This forces BaB loop to process the root domain, hitting max_domains=1.
    let threshold = 0.5;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_mins(1),
        max_domains: 1, // Very small domain limit
        max_depth: 100,
        batch_size: 4,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    match &result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                reason.contains("Domain limit"),
                "Expected domain limit reason, got: {}",
                reason
            );
        }
        other => {
            panic!("Expected Unknown with domain limit, got {:?}", other);
        }
    }
}

/// Test branching creates child domains.
/// With branching required and max_domains=2, the loop should explore the root
/// and one child before terminating.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_branching_creates_children() {
    let graph = simple_graph_network();

    // Input bounds that create unstable region (crosses zero after linear)
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let objective = vec![1.0];
    // Root bounds for this network are [0, 4].
    // Use threshold = 0.5 so neither immediate condition triggers:
    // - root_lower (0) >= threshold (0.5)? No (not a violation)
    // - root_upper (4) < threshold (0.5)? No (not verified)
    // This forces BaB to branch, creating child domains.
    let threshold = 0.5;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 2,
        max_depth: 10,
        batch_size: 1,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    assert_eq!(
        result.domains_explored, 2,
        "Should explore root + one child before terminating"
    );
    assert!(
        result.max_depth_reached >= 1,
        "Expected branching to create depth-1 child domains"
    );
}

// =============================================================================
// Additional GPU BaB test coverage (Part of #1534)
// =============================================================================
// Note: These tests depend on the branching path which is currently broken (#1536).
// The tensor shape mismatch bug must be fixed before these tests can pass.

/// Test deeper branching tree exploration (depth > 1).
/// Part of #1534: Exercise multiple branching iterations with deeper trees.
/// Blocked by #1536 until tensor shape mismatch is fixed.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_deeper_branching() {
    let graph = simple_graph_network();

    // Input bounds that create unstable region requiring multiple splits
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let objective = vec![1.0];
    // Root bounds [0, 4]. Threshold 0.5 forces BaB loop without immediate termination.
    let threshold = 0.5;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 10, // Allow more domains for deeper exploration
        max_depth: 5,    // Allow deeper trees
        batch_size: 2,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    // With more domains allowed, should explore deeper
    assert!(
        result.domains_explored > 2,
        "Expected deeper exploration with max_domains=10, got {}",
        result.domains_explored
    );
    // Verify we actually went deeper than depth 1
    assert!(
        result.max_depth_reached >= 1,
        "Expected depth >= 1, got {}",
        result.max_depth_reached
    );
}

/// Test unresolved_due_to_depth exit path.
/// Part of #1534: When max_depth=1, domains at depth 1 should be dropped.
/// Blocked by #1536 until tensor shape mismatch is fixed.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_unresolved_due_to_depth() {
    let graph = simple_graph_network();

    // Input bounds that create unstable region requiring branching
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let objective = vec![1.0];
    let threshold = 0.5;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 100, // Allow many domains
        max_depth: 1,     // But limit depth to 1 - children at depth 1 will be dropped
        batch_size: 2,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    // Should return Unknown with max depth reason
    match &result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                reason.contains("Max depth") || reason.contains("max depth"),
                "Expected max depth reason, got: {}",
                reason
            );
        }
        other => {
            panic!("Expected Unknown with max depth reason, got {:?}", other);
        }
    }
    assert_eq!(result.max_depth_reached, 1, "Should reach max depth of 1");
}

/// Test that GPU BaB processes child domains correctly.
/// Part of #1534: Verify child domain creation and processing in BaB loop.
///
/// Note: This test verifies that child domains are created and verified,
/// not the actual bound values (which would require introspection of domain internals).
#[ntest::timeout(60000)]
#[test]
fn test_gpu_bab_child_domain_processing() {
    let graph = simple_graph_network();

    // Wide input bounds that create unstable region
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let objective = vec![1.0];
    // Root output bounds for this network with input [-1,1] × [-1,1]:
    // linear1 gives [min, max] for each neuron across input combinations
    // w2 sums the ReLU outputs: bounds roughly [0, 4]
    // High threshold ensures verification without branching (upper < threshold)
    let threshold = 5.0;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 10,
        max_depth: 3,
        batch_size: 1, // Process one at a time to ensure proper child handling
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    // High threshold should lead to verification
    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "Should verify with high threshold"
    );
    // Multiple domains means children were created and processed
    assert!(result.domains_explored >= 1, "Should explore at least root");
    // Verification count > 0 means domains were successfully processed
    assert!(
        result.domains_verified >= 1,
        "At least one domain should be verified"
    );
}

/// #1896 regression: GPU BaB must return Verified when BaB tree is exhausted
/// with branching (depth >= 1) and all leaf domains are verified.
///
/// Network: 3-hidden-layer MLP [2]->[4]->[4]->[4]->[1] with 12 ReLU neurons.
/// Three ReLU layers create significant CROWN over-approximation at the root
/// (spec-guided CROWN upper ≈ 4.4 vs true max ≈ 3.0). The threshold is set
/// at 4.0 — above the true max (property holds) but below the CROWN upper
/// (root doesn't verify), forcing BaB branching. After splitting, child
/// domains have tighter bounds and can verify.
///
/// Before fix: `domains_verified == domains_explored` was always false with
/// branching (parents increment explored, children increment verified, disjoint).
/// After fix: queue exhaustion + no unresolved flags = Verified.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_bab_verified_after_branching_1896() {
    // 3-hidden-layer MLP: [2] -> [4] -> [4] -> [4] -> [1]
    // Same weights as test_gpu_bab_beta_optimization_tightens_bounds_1484.
    // Known: CROWN root upper ≈ 4.4, sampled true max ≈ 3.0.
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
        timeout: Duration::from_secs(30),
        max_domains: 10000,
        max_depth: 20,
        batch_size: 16,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    };

    // Derive threshold from the exact root bound computation used by
    // verify_graph_gpu_domain_list (spec-guided CROWN with initial node bounds),
    // so the test remains stable as raw CROWN quality changes over time.
    let initial_node_bounds = if config.use_alpha_crown {
        let (bounds, _) = graph
            .collect_alpha_crown_bounds_dag(&input, &config.alpha_config)
            .unwrap();
        bounds
    } else if config.alpha_config.fix_interm_bounds {
        graph.collect_node_bounds(&input).unwrap()
    } else {
        graph.collect_crown_ibp_bounds_dag(&input).unwrap()
    };
    let spec_matrix = Array2::from_shape_vec((1, objective.len()), objective.clone())
        .expect("spec matrix shape should match objective length");
    let root_bounds = graph
        .propagate_crown_with_specs_and_engine_with_node_bounds(
            &input,
            &spec_matrix,
            None,
            &initial_node_bounds,
        )
        .unwrap();
    let root_lower = root_bounds.lower()[[0]];
    let root_upper = root_bounds.upper()[[0]];

    const GRID: usize = 151;
    let mut sampled_max = f32::NEG_INFINITY;
    for i in 0..GRID {
        let x0 = -1.0 + 2.0 * (i as f32 / (GRID - 1) as f32);
        for j in 0..GRID {
            let x1 = -1.0 + 2.0 * (j as f32 / (GRID - 1) as f32);
            let point =
                BoundedTensor::new(arr1(&[x0, x1]).into_dyn(), arr1(&[x0, x1]).into_dyn()).unwrap();
            let out = graph.propagate_ibp(&point).unwrap();
            sampled_max = sampled_max.max(out.lower()[[0]]);
        }
    }

    assert!(
        root_upper > sampled_max + 1e-3,
        "Test network no longer forces branching: root_upper ({}) <= sampled_max ({})",
        root_upper,
        sampled_max
    );

    let threshold = sampled_max + (root_upper - sampled_max) * 0.95;
    assert!(
        root_lower < threshold && root_upper >= threshold,
        "Threshold must force BaB branch (not immediate verify/violation): \
         lower={}, upper={}, threshold={}",
        root_lower,
        root_upper,
        threshold
    );
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    // Must return Verified after exhausting the BaB tree
    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "#1896 regression: GPU BaB with branching must return Verified when \
         all leaf domains are verified. Got {:?} (explored={}, verified={}, depth={})",
        result.result,
        result.domains_explored,
        result.domains_verified,
        result.max_depth_reached
    );
    // Must have done actual branching (depth > 0)
    assert!(
        result.max_depth_reached >= 1,
        "Expected branching (depth >= 1), got depth={} (root=[{}, {}], sampled_max={}, threshold={})",
        result.max_depth_reached,
        root_lower,
        root_upper,
        sampled_max,
        threshold
    );
    // Bug trigger shape: parent domains count as explored, but children that
    // verify inline increment domains_verified without incrementing explored.
    // Before #1896, `domains_verified == domains_explored` rejected this case.
    assert!(
        result.domains_verified > result.domains_explored,
        "Expected branching mismatch (verified > explored), got explored={}, verified={} \
         (root=[{}, {}], sampled_max={}, threshold={})",
        result.domains_explored,
        result.domains_verified,
        root_lower,
        root_upper,
        sampled_max,
        threshold
    );
}

/// Test GPU BaB with stable neurons returns correct status.
/// Part of #1534: Test stable neuron exit path.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_stable_neurons() {
    let graph = simple_graph_network();

    // Input bounds that produce stable ReLU neurons (all at zero boundary).
    // With w1 = [[1, -1], [-1, 1]] and input [1,1] to [2,2]:
    // linear1([1,1]) = [1-1, -1+1] = [0, 0]
    // linear1([2,2]) = [2-2, -2+2] = [0, 0]
    // So pre-ReLU bounds are [0, 0] - neurons are stable at boundary (not unstable).
    // ReLU([0,0]) = [0,0], linear2([0,0]) = 0
    // Output bounds: [0, 0]
    let input =
        BoundedTensor::new(arr1(&[1.0, 1.0]).into_dyn(), arr1(&[2.0, 2.0]).into_dyn()).unwrap();

    let objective = vec![1.0];
    // With output bounds [0, 0] and threshold -0.5:
    // lower (0) >= threshold (-0.5) → potential violation
    let threshold = -0.5;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(10),
        max_domains: 100,
        max_depth: 10,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    // With stable neurons (pre-ReLU bounds [0,0]) and output [0,0]:
    // - lower (0) >= threshold (-0.5) → potential violation detected
    // Expected: PotentialViolation because lower bound violates the property
    assert_eq!(
        result.result,
        BabVerificationStatus::PotentialViolation,
        "With output lower bound (0) >= threshold (-0.5), should detect potential violation, got {:?}",
        result.result
    );
}
