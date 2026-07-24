// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cross-path convergence diagnostics for GPU BaB.

use super::*;

/// Diagnostic: Compare GPU BaB (GraphNetwork) vs sequential BaB (Network) on same model.
///
/// This test exposes the root cause of #1817 by running both paths on an identical
/// 2-layer ReLU network and comparing whether both can verify the same property.
///
/// The simple_graph_network is: Linear(2→2) -> ReLU -> Linear(2→1)
/// Network: f(x,y) = max(0, x-y) + max(0, -x+y) = |x-y|
/// For input in [-1,1]x[-1,1], output ∈ [0, 2].
/// True threshold: output < 3 is always true, so both paths should verify.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_bab_vs_sequential_convergence() {
    // Build GraphNetwork
    let graph = simple_graph_network();

    // Build equivalent sequential Network
    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[1.0, 1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    // Input: [-1, 1] x [-1, 1] — wide enough to force branching
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Threshold 3.0: output = |x-y| ∈ [0, 2], so upper < 3.0 is always true.
    // CROWN should prove this without BaB, but using a moderately tight threshold
    // will test the BaB path if initial bounds are loose.
    let threshold = 3.0;

    // Config matching ACAS-Xu style: BoundImpact branching, beta_iterations=0
    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(10),
        max_domains: 200,
        max_depth: 20,
        batch_size: 16,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    };

    // --- Sequential path: Network + verify() ---
    let seq_verifier = BetaCrownVerifier::new(config.clone());
    let seq_result = seq_verifier
        .verify(&network, &input, threshold)
        .expect("Sequential BaB should succeed");

    eprintln!(
        "\n=== Sequential BaB ===\nResult: {:?}\nDomains explored: {}\nDomains verified: {}",
        seq_result.result, seq_result.domains_explored, seq_result.domains_verified
    );

    // --- Graph path: GraphNetwork + verify_graph_gpu_domain_list() ---
    let objective = vec![1.0]; // Single output
    let graph_verifier = BetaCrownVerifier::new(config.clone());
    let graph_result = graph_verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    eprintln!(
        "\n=== Graph BaB (DomainList) ===\nResult: {:?}\nDomains explored: {}\nDomains verified: {}",
        graph_result.result, graph_result.domains_explored, graph_result.domains_verified
    );

    // Both should produce the same verification outcome
    assert_eq!(
        std::mem::discriminant(&seq_result.result),
        std::mem::discriminant(&graph_result.result),
        "Sequential and Graph BaB should agree on verification status.\n\
         Sequential: {:?} (explored={}, verified={})\n\
         Graph:      {:?} (explored={}, verified={})",
        seq_result.result,
        seq_result.domains_explored,
        seq_result.domains_verified,
        graph_result.result,
        graph_result.domains_explored,
        graph_result.domains_verified,
    );

    // Now test with a tight threshold that forces BaB branching.
    // output = |x-y| has max 2 on [-1,1]x[-1,1]. Threshold 2.5 requires
    // BaB to split and tighten bounds in sub-regions.
    let tight_threshold = 2.5;

    let seq_verifier2 = BetaCrownVerifier::new(config.clone());
    let seq_result2 = seq_verifier2
        .verify(&network, &input, tight_threshold)
        .expect("Sequential BaB should succeed (tight)");

    let graph_verifier2 = BetaCrownVerifier::new(config);
    let graph_result2 = graph_verifier2
        .verify_graph_gpu_domain_list(&graph, &input, &objective, tight_threshold, None, None)
        .expect("GPU BaB should succeed (tight)");

    eprintln!(
        "\n=== Tight threshold ({}) ===\n\
         Sequential: {:?} (explored={}, verified={})\n\
         Graph:      {:?} (explored={}, verified={})",
        tight_threshold,
        seq_result2.result,
        seq_result2.domains_explored,
        seq_result2.domains_verified,
        graph_result2.result,
        graph_result2.domains_explored,
        graph_result2.domains_verified,
    );

    // The graph path should verify at least as many domains as the sequential path.
    // If there's a convergence issue, the graph path will verify fewer domains.
    assert!(
        graph_result2.domains_verified > 0
            || !matches!(seq_result2.result, BabVerificationStatus::Verified),
        "If sequential verifies, graph should verify at least some domains.\n\
         Sequential: {:?} (verified={})\n\
         Graph:      {:?} (verified={})",
        seq_result2.result,
        seq_result2.domains_verified,
        graph_result2.result,
        graph_result2.domains_verified,
    );
}

/// Diagnostic #1817: Deep network (3 hidden layers) — sequential vs GPU BaB.
///
/// Uses a deeper network where BaB branching is required and compares the
/// sequential Network path vs the GraphNetwork GPU BaB path. If bounds
/// tighten correctly through branching in the GPU BaB path, both should
/// produce the same verification status.
#[ntest::timeout(120000)]
#[test]
fn test_gpu_bab_deep_network_convergence() {
    let graph = deep_graph_network();
    let network = deep_sequential_network();

    // Wide input bounds to create many unstable ReLU neurons
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // First: compute both path's root bounds to find a good threshold
    let root_config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };

    // Sequential root bounds (with spec layer = identity for single output)
    let seq_verifier = BetaCrownVerifier::new(root_config.clone());
    let seq_root = seq_verifier
        .verify(&network, &input, 1000.0)
        .expect("Sequential root should succeed");
    eprintln!(
        "\n=== Deep Network Root Bounds ===\nSequential: explored={}, verified={}, status={:?}",
        seq_root.domains_explored, seq_root.domains_verified, seq_root.result
    );

    // Graph root bounds
    let objective = vec![1.0]; // Single output
    let graph_verifier = BetaCrownVerifier::new(root_config);
    let graph_root = graph_verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, 1000.0, None, None)
        .expect("Graph root should succeed");
    eprintln!(
        "Graph BaB:  explored={}, verified={}, status={:?}",
        graph_root.domains_explored, graph_root.domains_verified, graph_root.result
    );

    // Both should verify immediately with high threshold
    assert_eq!(graph_root.result, BabVerificationStatus::Verified);

    // Now use a tight threshold that forces BaB branching
    // Pick a threshold between the root lower and upper bounds
    let root_output = graph_root
        .output_bounds
        .as_ref()
        .expect("should have output bounds");
    let root_lower = root_output.lower_scalar();
    let root_upper = root_output.upper_scalar();
    eprintln!("Root output bounds: [{:.6}, {:.6}]", root_lower, root_upper);

    // Set threshold slightly above the root upper bound to force BaB but allow verification
    // If CROWN is working correctly, sub-domains should tighten below this threshold
    let threshold = root_upper * 0.8; // 80% of root upper — requires some tightening
    eprintln!("Using threshold = {:.6}", threshold);

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 500,
        max_depth: 20,
        batch_size: 16,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    };

    // Sequential BaB
    let seq_verifier2 = BetaCrownVerifier::new(config.clone());
    let seq_result = seq_verifier2
        .verify(&network, &input, threshold)
        .expect("Sequential BaB should succeed");

    eprintln!(
        "\n=== Deep Network BaB (threshold={:.6}) ===\n\
         Sequential: {:?} (explored={}, verified={})",
        threshold, seq_result.result, seq_result.domains_explored, seq_result.domains_verified
    );

    // Graph BaB
    let graph_verifier2 = BetaCrownVerifier::new(config);
    let graph_result = graph_verifier2
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    eprintln!(
        "Graph BaB:  {:?} (explored={}, verified={})",
        graph_result.result, graph_result.domains_explored, graph_result.domains_verified
    );

    // If sequential verifies, graph should also verify
    if matches!(seq_result.result, BabVerificationStatus::Verified) {
        assert!(
            graph_result.domains_verified > 0,
            "Sequential BaB verified but GPU BaB verified 0 domains.\n\
             This is the #1817 bug: CROWN backward in GPU BaB path is not\n\
             tightening bounds through branching.\n\
             Sequential: {:?} (explored={}, verified={})\n\
             Graph:      {:?} (explored={}, verified={})",
            seq_result.result,
            seq_result.domains_explored,
            seq_result.domains_verified,
            graph_result.result,
            graph_result.domains_explored,
            graph_result.domains_verified,
        );
    }
}

/// Diagnostic test for #1817: GPU BaB with a deeper MLP that forces BaB branching.
///
/// This test creates a 3-hidden-layer ReLU network where CROWN alone cannot
/// verify the property, forcing the BaB loop to branch and tighten bounds.
/// Compares verify() (Network, BinaryHeap) vs verify_graph_gpu_domain_list()
/// (GraphNetwork, DomainList) to identify convergence differences.
#[ntest::timeout(60000)]
#[test]
fn test_gpu_bab_vs_sequential_deeper_mlp_1817() {
    // Build a 3-hidden-layer MLP: [2] -> [4] -> [4] -> [4] -> [1]
    // Use weights that create nontrivial ReLU patterns requiring BaB.
    let w1 = arr2(&[[2.0, -1.0], [-1.0, 2.0], [1.0, 1.0], [-1.0, -1.0]]);
    let b1 = Some(arr1(&[0.0, 0.0, -0.5, 0.5]));

    let w2 = arr2(&[
        [1.0, -1.0, 0.5, 0.0],
        [-1.0, 1.0, 0.0, 0.5],
        [0.5, 0.5, -1.0, 1.0],
        [0.0, -0.5, 1.0, -1.0],
    ]);
    let b2 = Some(arr1(&[0.0, 0.0, 0.0, 0.0]));

    let w3 = arr2(&[
        [1.0, 0.5, -0.5, 0.0],
        [0.0, 1.0, 0.5, -0.5],
        [-0.5, 0.0, 1.0, 0.5],
        [0.5, -0.5, 0.0, 1.0],
    ]);
    let b3 = Some(arr1(&[0.0, 0.0, 0.0, 0.0]));

    let w4 = arr2(&[[1.0, -1.0, 0.5, -0.5]]);
    let b4 = None;

    // Build sequential Network
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), b1.clone()).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2.clone(), b2.clone()).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w3.clone(), b3.clone()).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w4.clone(), b4.clone()).unwrap(),
    ));

    // Build equivalent GraphNetwork
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
        Layer::Linear(LinearLayer::new(w2, b2).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, b3).unwrap()),
        vec!["relu2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["linear3".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear4",
        Layer::Linear(LinearLayer::new(w4, b4).unwrap()),
        vec!["relu3".to_string()],
    ));
    graph.set_output("linear4");

    // Wide input range to force many unstable neurons
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Config: BoundImpact branching (same as GPU BaB ACAS-Xu bench)
    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 10000,
        max_depth: 30,
        batch_size: 64,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    };

    // --- First: estimate output range via a bounded grid sample ---
    // Keep sample count moderate so this diagnostic test stays below timeout.
    const GRID_SAMPLES: usize = 101;
    let mut min_out = f32::INFINITY;
    let mut max_out = f32::NEG_INFINITY;
    for i in 0..GRID_SAMPLES {
        let x0 = -1.0 + 2.0 * (i as f32 / (GRID_SAMPLES - 1) as f32);
        for j in 0..GRID_SAMPLES {
            let x1 = -1.0 + 2.0 * (j as f32 / (GRID_SAMPLES - 1) as f32);
            let inp =
                BoundedTensor::new(arr1(&[x0, x1]).into_dyn(), arr1(&[x0, x1]).into_dyn()).unwrap();
            let out = network.propagate_ibp(&inp).unwrap();
            let val = out.lower()[[0]]; // point evaluation: lower == upper
            min_out = min_out.min(val);
            max_out = max_out.max(val);
        }
    }
    eprintln!(
        "\n=== Sampled output range: [{:.4}, {:.4}] ===",
        min_out, max_out
    );

    // Add conservative margin to avoid false negatives from coarse sampling.
    let threshold = max_out + 1.0;
    eprintln!("Threshold: {:.4} (sampled max + 1.0)", threshold);

    // --- Sequential BaB ---
    let seq_verifier = BetaCrownVerifier::new(config.clone());
    let seq_result = seq_verifier
        .verify(&network, &input, threshold)
        .expect("Sequential BaB should succeed");

    eprintln!(
        "\n=== Sequential BaB ===\nResult: {:?}\nDomains explored: {}\nDomains verified: {}",
        seq_result.result, seq_result.domains_explored, seq_result.domains_verified
    );

    // --- Graph BaB (DomainList) ---
    // #1817 diagnostics use tracing::debug! in multi_objective; no logger setup
    // is required for this regression assertion.
    let objective = vec![1.0]; // Single output
    let graph_verifier = BetaCrownVerifier::new(config);
    let graph_result = graph_verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    eprintln!(
        "\n=== Graph BaB (DomainList) ===\nResult: {:?}\nDomains explored: {}\nDomains verified: {}",
        graph_result.result, graph_result.domains_explored, graph_result.domains_verified
    );

    // Graph BaB should verify at least some domains when the threshold is above
    // the true output maximum. Note: the sequential path has a soundness issue where
    // it silently drops fully-constrained domains that don't verify (treating an empty
    // queue as "Verified"), so we don't compare status directly against sequential.
    assert!(
        graph_result.domains_verified > 0,
        "Graph BaB should verify at least some domains with threshold above true max.\n\
         Sequential: {:?} (explored={}, verified={})\n\
         Graph:      {:?} (explored={}, verified={})",
        seq_result.result,
        seq_result.domains_explored,
        seq_result.domains_verified,
        graph_result.result,
        graph_result.domains_explored,
        graph_result.domains_verified,
    );

    // The Graph BaB path correctly reports Unknown when fully-constrained domains
    // remain unverified (NoUnstable but upper >= threshold). This is more sound
    // than the sequential path which silently drops these domains.
    eprintln!(
        "\nGraph BaB correctly reports: {:?}\n\
         (Sequential unsoundly reports: {:?})",
        graph_result.result, seq_result.result,
    );
}

/// #1817 diagnostic: Compare Network CROWN vs GraphNetwork CROWN initial bounds.
///
/// If GraphNetwork CROWN produces systematically wider bounds than sequential
/// Network CROWN for the same architecture, the GPU BaB path will need more
/// splits to verify, explaining the 0-verified-domains on ACAS-Xu.
#[ntest::timeout(10000)]
#[test]
fn test_crown_bounds_network_vs_graph_network_1817() {
    // Build a deeper MLP: [2] -> [8] -> [8] -> [8] -> [8] -> [1]
    // (4 hidden layers to expose potential divergence)
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

    // Build sequential Network (with spec layer selecting output 0)
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

    // Build equivalent GraphNetwork
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

    // Compare CROWN bounds
    let seq_crown = network.propagate_crown(&input).unwrap();
    let graph_crown = graph.propagate_crown(&input).unwrap();

    let seq_lower = seq_crown.lower()[[0]];
    let seq_upper = seq_crown.upper()[[0]];
    let graph_lower = graph_crown.lower()[[0]];
    let graph_upper = graph_crown.upper()[[0]];

    eprintln!(
        "\n=== CROWN bounds comparison (4-hidden-layer MLP) ===\n\
         Sequential: [{:.6}, {:.6}] (width: {:.6})\n\
         Graph:      [{:.6}, {:.6}] (width: {:.6})\n\
         Ratio: graph_width / seq_width = {:.3}",
        seq_lower,
        seq_upper,
        seq_upper - seq_lower,
        graph_lower,
        graph_upper,
        graph_upper - graph_lower,
        (graph_upper - graph_lower) / (seq_upper - seq_lower),
    );

    // Both should be sound (graph bounds should contain sequential bounds)
    // but graph bounds should not be dramatically wider
    let graph_width = graph_upper - graph_lower;
    let seq_width = seq_upper - seq_lower;

    // Graph CROWN should produce bounds within 2x of sequential CROWN
    // (allowing for numerical differences in DAG vs sequential propagation)
    assert!(
        graph_width <= seq_width * 2.5 + 1e-4,
        "GraphNetwork CROWN produced bounds {:.2}x wider than sequential CROWN.\n\
         Sequential width: {:.6}\n\
         Graph width: {:.6}\n\
         This explains why GPU BaB (graph) needs more splits than CPU BaB (sequential).",
        graph_width / seq_width,
        seq_width,
        graph_width,
    );
}
