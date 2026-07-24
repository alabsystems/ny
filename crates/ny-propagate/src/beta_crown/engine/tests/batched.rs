// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_batched_processing_same_result() {
    // Test that batched processing gives the same result as sequential
    let network = simple_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Sequential (batch_size=1, no parallel children)
    let config_seq = BetaCrownConfig {
        max_domains: 50,
        timeout: Duration::from_secs(10),
        batch_size: 1,
        parallel_children: false,
        ..Default::default()
    };
    let verifier_seq = BetaCrownVerifier::new(config_seq);
    let result_seq = verifier_seq.verify(&network, &input, -5.0).unwrap();

    // Batched parallel (batch_size=4, parallel children)
    let config_par = BetaCrownConfig {
        max_domains: 50,
        timeout: Duration::from_secs(10),
        batch_size: 4,
        parallel_children: true,
        ..Default::default()
    };
    let verifier_par = BetaCrownVerifier::new(config_par);
    let result_par = verifier_par.verify(&network, &input, -5.0).unwrap();

    // Both should give the same verification result
    assert_eq!(result_seq.result, result_par.result);
    println!("Sequential: {:?}", result_seq);
    println!("Parallel: {:?}", result_par);
}

#[ntest::timeout(10000)]
#[test]
fn test_parallel_children_enabled() {
    // Test that parallel_children flag works
    let network = simple_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 50,
        timeout: Duration::from_secs(10),
        batch_size: 1,
        parallel_children: true, // Use parallel child creation
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -5.0).unwrap();

    assert_eq!(result.result, BabVerificationStatus::Verified);
}

#[ntest::timeout(10000)]
#[test]
fn test_large_batch_processing() {
    // Test with large batch size on a deeper network
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7], [-0.2, 0.8]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = arr2(&[[0.5, -0.3, 0.7, 0.1], [-0.4, 0.6, -0.2, 0.5]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let w3 = arr2(&[[1.0, -0.5]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(10),
        batch_size: 8, // Large batch
        parallel_children: true,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -10.0).unwrap();

    println!("Large batch result: {:?}", result);
    assert!(result.domains_explored > 0);
}

#[ntest::timeout(5000)]
#[test]
fn test_propagate_crown_with_batched_domains_success() {
    // Success-path test that exercises batched CROWN propagation without fallback.
    use crate::batched_domain::BatchedDomains;

    // Create simple graph network: Linear -> ReLU (scalar output)
    let w1 = arr2(&[[1.0, -1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.set_output("relu1");

    let input_bounds = Arc::new(
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap(),
    );

    // Collect initial bounds
    let initial_bounds = graph.collect_node_bounds(&input_bounds).unwrap();

    // Create root domain
    let root = GraphBabDomain::root(
        initial_bounds,
        -10.0, // placeholder lower
        10.0,  // placeholder upper
        &input_bounds,
        false,
    )
    .unwrap();

    // Create BatchedDomains from single domain
    let domains = vec![&root];
    let layer_names = vec!["relu1".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    // Verify BatchedDomains was created correctly
    assert_eq!(batched.len(), 1);
    assert!(batched.layer_lowers().contains_key("relu1"));

    // Create verifier
    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    // Objective: sum of outputs (coefficient = 1.0 for each)
    let output_node = graph.output_name();
    let output_node = if output_node.is_empty() {
        "relu1"
    } else {
        output_node
    };
    let output_dim = root
        .node_bounds
        .get(output_node)
        .expect("output bounds should exist")
        .lower()
        .len();
    let objective = vec![1.0; output_dim];

    let engine = NaiveCpuGemmEngine;

    // The method should process and return updates.
    let result = verifier
        .propagate_crown_with_batched_domains(&graph, &domains, &batched, &objective, &engine);

    let updates = result.expect("batched CROWN propagation should succeed");
    assert_eq!(updates.len(), 1, "Should have one update for one domain");
    assert_eq!(updates[0].domain_idx, 0);
    assert!(updates[0].new_lower_bound.is_finite());
    assert!(updates[0].new_upper_bound.is_finite());
    assert!(
        updates[0].new_layer_bounds.is_empty(),
        "Layer bounds are not extracted in this path yet"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_propagate_crown_with_batched_domains_multi_domain_success() {
    // Multi-domain success-path test: exercises batched CROWN propagation
    // with multiple domains, ensuring the batch processing path works correctly
    // without falling back to single-domain serial processing.
    //
    // Redesigned for #2992: uses `propagate_crown_with_batched_domains_full`
    // which carries per-domain alpha states. Constrained domains have fewer
    // unstable neurons (the constrained neuron uses a fixed slope 0 or 1
    // instead of an optimizable alpha), producing genuinely different CROWN
    // backward bounds. The deprecated `propagate_crown_with_batched_domains`
    // hardcoded None for alpha states, making all domains produce identical bounds.
    use crate::batched_domain::BatchedDomains;

    // 3-layer network: Linear1(2->2) -> ReLU1 -> Linear2(2->2) -> ReLU2 -> Linear3(2->1)
    //
    // Same architecture as the alpha-state test: two ReLU layers with mixed-sign
    // output weights ensure CROWN beats IBP. The asymmetric input bounds make
    // heuristic alpha=1, so constraining a neuron to inactive (slope=0) changes
    // the backward computation differently from the unconstrained heuristic.
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[1.0, 1.0], [1.0, -1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    let w3 = arr2(&[[1.0, -0.5]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

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
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    // Asymmetric input bounds: heuristic gives alpha=1 for relu1 neurons.
    // Constraining neuron 0 to inactive (slope=0) differs from heuristic (alpha=1),
    // changing the CROWN backward computation.
    let input_bounds = Arc::new(
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[1.5, 1.5]).into_dyn()).unwrap(),
    );

    let initial_bounds = graph.collect_node_bounds(&input_bounds).unwrap();

    let root = GraphBabDomain::root(initial_bounds, -10.0, 10.0, &input_bounds, false).unwrap();

    // Child 1: ReLU neuron 0 is active (x >= 0).
    // This makes neuron 0 use fixed slope 1 instead of heuristic alpha.
    let constraint1 = GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    };
    let child1 = root
        .with_constraint(&graph, constraint1, false)
        .unwrap()
        .expect("child1 should be feasible");

    // Child 2: ReLU neuron 0 is inactive (x <= 0).
    // This makes neuron 0 use fixed slope 0 instead of heuristic alpha.
    let constraint2 = GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: false,
        score: 0.0,
    };
    let child2 = root
        .with_constraint(&graph, constraint2, false)
        .unwrap()
        .expect("child2 should be feasible");

    let domains: Vec<&GraphBabDomain> = vec![&root, &child1, &child2];
    let layer_names = vec!["relu1".to_string(), "relu2".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    assert_eq!(batched.len(), 3, "Should have 3 domains in batch");

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    let output_node = graph.output_name();
    let output_dim = root
        .node_bounds
        .get(output_node)
        .expect("output bounds should exist")
        .lower()
        .len();
    let objective = vec![1.0; output_dim];

    // Use the _full API which carries per-domain alpha states via
    // BatchedBackwardContext. The deprecated API hardcoded None for alpha.
    let results = verifier
        .propagate_crown_with_batched_domains_full(
            &graph,
            &domains,
            &batched,
            &objective,
            &NaiveCpuGemmEngine,
        )
        .expect("batched CROWN propagation should succeed for multiple domains");

    assert_eq!(results.len(), 3, "Should have results for all 3 domains");

    // Collect lower bounds from each domain for comparison.
    let mut lower_bounds = Vec::new();
    for (idx, result) in results.iter().enumerate() {
        let (bounds, _) = result.as_ref().unwrap_or_else(|| {
            panic!("domain {} should produce Some result", idx);
        });
        let lb = bounds.lower_scalar();
        let ub = bounds.upper_scalar();
        assert!(
            lb.is_finite(),
            "Domain {} lower bound should be finite",
            idx
        );
        assert!(
            ub.is_finite(),
            "Domain {} upper bound should be finite",
            idx
        );
        assert!(
            lb <= ub,
            "Domain {} lower bound should not exceed upper bound",
            idx
        );
        lower_bounds.push(lb);
    }

    // With per-domain alpha states, constrained domains use fixed slopes
    // (0 for inactive, 1 for active) for the constrained neuron while the
    // root uses heuristic alpha. This produces different CROWN relaxation
    // slopes in the backward pass → different output bounds.
    // At minimum, child1 (active) and child2 (inactive) should differ since
    // their constrained neuron has opposite fixed slopes.
    assert!(
        (lower_bounds[1] - lower_bounds[2]).abs() > 1e-6,
        "child1 (active) and child2 (inactive) should produce different lower bounds \
         (got {} and {})",
        lower_bounds[1],
        lower_bounds[2]
    );
}

/// Regression for #1840: batched graph backward wildcard dispatch must use per-layer
/// CROWN backward (Sigmoid here), not identity/pass-through fallback.
#[ntest::timeout(5000)]
#[test]
fn test_batched_graph_dispatches_sigmoid_backward_1840() {
    use crate::batched_domain::BatchedDomains;

    let w1 = arr2(&[[1.2, -0.7]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[2.0]]);
    let linear2 = LinearLayer::new(w2, Some(arr1(&[-0.4]))).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "sigmoid1",
        Layer::Sigmoid(crate::SigmoidLayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["sigmoid1".to_string()],
    ));
    graph.set_output("linear2");

    let input_bounds = Arc::new(
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap(),
    );

    let expected = graph
        .propagate_crown(input_bounds.as_ref())
        .expect("graph_crown should succeed on sigmoid network");

    let initial_bounds = graph
        .collect_node_bounds(&input_bounds)
        .expect("should collect node bounds");
    let root = GraphBabDomain::root(initial_bounds, -10.0, 10.0, &input_bounds, false).unwrap();

    let domains: Vec<&GraphBabDomain> = vec![&root];
    let layer_names: Vec<String> = vec![];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let objective = vec![1.0_f32];

    let updates = verifier
        .propagate_crown_with_batched_domains(
            &graph,
            &domains,
            &batched,
            &objective,
            &NaiveCpuGemmEngine,
        )
        .expect("batched propagation should succeed");
    assert_eq!(
        updates.len(),
        1,
        "single domain should produce single update"
    );

    let update = &updates[0];
    let expected_lower = expected.lower_scalar();
    let expected_upper = expected.upper_scalar();

    assert!(
        (update.new_lower_bound - expected_lower).abs() < 1e-4,
        "batched lower bound mismatch: got {}, expected {}",
        update.new_lower_bound,
        expected_lower
    );
    assert!(
        (update.new_upper_bound - expected_upper).abs() < 1e-4,
        "batched upper bound mismatch: got {}, expected {}",
        update.new_upper_bound,
        expected_upper
    );
}

/// Regression for #1845: batched graph backward must consume per-domain alpha
/// state (propagate_linear_with_alpha), not always the fixed heuristic path.
///
/// Uses the BatchedBackwardContext API (the real GPU BaB path) rather than the
/// deprecated tuple-based propagate_crown_batched_backward which hardcodes
/// empty alpha states.
///
/// Redesigned for #2992: uses a 3-layer network (2 ReLU layers) where CROWN
/// provides tighter bounds than IBP. Alpha=0 at relu1 zeroes the CROWN
/// coefficients for relu1's crossing neurons, giving a tight lower bound
/// (-0.476) that survives IBP tightening (-1.125). The heuristic alpha=1
/// path gives a loose CROWN lower (-1.46) that gets clamped to IBP.
/// Single-layer topologies have exact IBP bounds clamping all alpha effects.
#[ntest::timeout(5000)]
#[test]
fn test_batched_graph_uses_alpha_state_for_relu_backward_1845() {
    use crate::batched_domain::BatchedDomains;
    use crate::beta_crown::state::GraphDomainAlphaState;

    // 3-layer graph: Linear(2->2) -> ReLU1 -> Linear(2->2) -> ReLU2 -> Linear(2->1).
    //
    // Key design properties for alpha-sensitivity after IBP tightening:
    // 1. Two ReLU layers: IBP compounds conservatism across both, making CROWN
    //    significantly tighter at the output. Single-hidden-layer networks have
    //    IBP bounds as tight as CROWN, clamping all alpha effects.
    // 2. Mixed positive/negative weights in the output layer (w3 = [1, -0.5]):
    //    the negative coefficient creates IBP pessimism that CROWN can avoid.
    // 3. Asymmetric input bounds: ensures heuristic alpha=1 (not alpha=0).
    //
    // Math trace (see #2992 for detailed derivation):
    //   alpha=heuristic(=1) at relu1: CROWN lower ≈ -1.46, tightened to IBP ≈ -1.13
    //   alpha=0 at relu1:             CROWN lower ≈ -0.48, tightened stays -0.48
    //   Difference ≈ 0.65, well above numerical noise.
    //
    // NOTE: alpha is only modified at relu1 (genuinely crossing neurons).
    // relu2 neuron 0 has pre-activation lower ≈ -1e-45 from directed rounding;
    // setting alpha=0 there would zero the main CROWN pathway.
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[1.0, 1.0], [1.0, -1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    let w3 = arr2(&[[1.0, -0.5]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

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
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    // Asymmetric input [-0.5, 1.5]^2:
    //   relu1 n0: pre-act [-0.75, 2.25] → u > -l → heuristic alpha=1
    //   relu1 n1: pre-act [-0.95, 1.65] → u > -l → heuristic alpha=1
    // Forcing alpha=0 at relu1 zeroes the lower relaxation slopes, producing
    // a CROWN bound dominated by the ReLU upper relaxation bias terms.
    let input_bounds = Arc::new(
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[1.5, 1.5]).into_dyn()).unwrap(),
    );

    let initial_bounds = graph
        .collect_node_bounds(&input_bounds)
        .expect("should collect node bounds");
    let mut heuristic_domain =
        GraphBabDomain::root(initial_bounds, -10.0, 10.0, &input_bounds, false).unwrap();

    // Initialize alpha state for BOTH domains so both go through
    // propagate_linear_with_alpha (not the heuristic fallback).
    // The root domain has empty alpha state by default; populate it.
    let heuristic_alpha = GraphDomainAlphaState::from_graph_bounds(
        &graph,
        &heuristic_domain.node_bounds,
        &heuristic_domain.history,
        heuristic_domain.input_bounds.as_ref(),
    );
    assert!(
        !heuristic_alpha.is_empty(),
        "expected unstable ReLU neurons for alpha optimization"
    );
    heuristic_domain.alpha_state = heuristic_alpha;

    // Clone domain and set alpha=0 for relu1 neurons only.
    // relu2 neuron 0 has pre-activation lower ≈ -1e-45 (denormalized float from
    // directed rounding) — technically crossing but effectively always-active.
    // Setting alpha=0 for that neuron zeroes the main CROWN pathway, producing
    // bounds ≈ IBP. We only perturb relu1 where both neurons are genuinely
    // crossing (l << 0).
    let mut alpha_domain = heuristic_domain.clone();
    if let Some(relu1_neurons) = alpha_domain.alpha_state.neurons_mut().get_mut("relu1") {
        for neuron_state in relu1_neurons.values_mut() {
            neuron_state.set_alpha(0.0);
        }
    }

    let domains: Vec<&GraphBabDomain> = vec![&heuristic_domain, &alpha_domain];
    let layer_names = vec!["relu1".to_string(), "relu2".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names)
        .expect("batched domains should build");

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let objective = vec![1.0_f32];

    // Use propagate_crown_with_batched_domains_full which internally creates
    // BatchedBackwardContext (the real GPU BaB path) carrying alpha states,
    // NOT the deprecated tuple-based API which hardcodes None.
    let results = verifier
        .propagate_crown_with_batched_domains_full(
            &graph,
            &domains,
            &batched,
            &objective,
            &NaiveCpuGemmEngine,
        )
        .expect("batched propagation should succeed");
    assert_eq!(results.len(), 2, "expected one result per domain");

    // Both domains must produce valid, finite, ordered bounds.
    let mut lower_bounds = Vec::new();
    for (idx, result) in results.iter().enumerate() {
        let (bounds, _) = result.as_ref().unwrap_or_else(|| {
            panic!("domain {} should produce Some result", idx);
        });
        let lb = bounds.lower_scalar();
        let ub = bounds.upper_scalar();
        assert!(
            lb.is_finite(),
            "domain {} lower bound should be finite, got {}",
            idx,
            lb
        );
        assert!(
            ub.is_finite(),
            "domain {} upper bound should be finite, got {}",
            idx,
            ub
        );
        assert!(
            lb <= ub,
            "domain {} bounds should be ordered: lb={} > ub={}",
            idx,
            lb,
            ub
        );
        lower_bounds.push(lb);
    }

    // With a multi-layer network, CROWN is tighter than IBP at the output.
    // Different alpha values (heuristic vs zero) produce different CROWN
    // relaxation slopes, yielding different lower bounds that survive IBP
    // tightening. This is the #2992 regression: the original single-ReLU
    // topology had exact IBP bounds that clamped away all alpha effects.
    assert!(
        (lower_bounds[0] - lower_bounds[1]).abs() > 1e-6,
        "alpha=heuristic and alpha=0 should produce different lower bounds \
         for multi-layer network (got {} and {})",
        lower_bounds[0],
        lower_bounds[1]
    );
}

/// Regression for #3782 Slice 2: batched graph backward must thread the
/// upper-path alpha state into ReLU backward instead of hardcoding `None`.
#[ntest::timeout(5000)]
#[test]
fn test_batched_graph_uses_upper_alpha_state_for_relu_backward_3782() {
    use crate::batched_domain::BatchedDomains;
    use crate::beta_crown::state::GraphDomainAlphaState;

    let w1 = arr2(&[[1.0, 0.5], [-0.3, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[1.0, 1.0], [1.0, -1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    let w3 = arr2(&[[1.0, -0.5]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

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
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    let input_bounds = Arc::new(
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[1.5, 1.5]).into_dyn()).unwrap(),
    );

    let initial_bounds = graph
        .collect_node_bounds(&input_bounds)
        .expect("should collect node bounds");
    let mut dual_alpha_domain =
        GraphBabDomain::root(initial_bounds, -10.0, 10.0, &input_bounds, false).unwrap();
    dual_alpha_domain.alpha_state = GraphDomainAlphaState::from_graph_bounds(
        &graph,
        &dual_alpha_domain.node_bounds,
        &dual_alpha_domain.history,
        dual_alpha_domain.input_bounds.as_ref(),
    );

    let mut lower_only_domain = dual_alpha_domain.clone();
    for neuron_map in lower_only_domain
        .alpha_state
        .upper_neurons_mut()
        .values_mut()
    {
        for neuron_state in neuron_map.values_mut() {
            neuron_state.set_alpha(0.0);
        }
    }

    let domains: Vec<&GraphBabDomain> = vec![&dual_alpha_domain, &lower_only_domain];
    let layer_names = vec!["relu1".to_string(), "relu2".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names)
        .expect("batched domains should build");

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let objective = vec![1.0_f32];

    let results = verifier
        .propagate_crown_with_batched_domains_full(
            &graph,
            &domains,
            &batched,
            &objective,
            &NaiveCpuGemmEngine,
        )
        .expect("batched propagation should succeed");
    assert_eq!(results.len(), 2, "expected one result per domain");

    let mut lower_bounds = Vec::new();
    let mut upper_bounds = Vec::new();
    for (idx, result) in results.iter().enumerate() {
        let (bounds, _) = result.as_ref().unwrap_or_else(|| {
            panic!("domain {} should produce Some result", idx);
        });
        let lb = bounds.lower_scalar();
        let ub = bounds.upper_scalar();
        assert!(
            lb.is_finite(),
            "domain {} lower bound should be finite",
            idx
        );
        assert!(
            ub.is_finite(),
            "domain {} upper bound should be finite",
            idx
        );
        assert!(lb <= ub, "domain {} bounds should be ordered", idx);
        lower_bounds.push(lb);
        upper_bounds.push(ub);
    }

    assert!(
        (lower_bounds[0] - lower_bounds[1]).abs() < 1e-5,
        "changing only alpha_upper should leave lower bounds unchanged \
         (got {} and {})",
        lower_bounds[0],
        lower_bounds[1]
    );
    assert!(
        (upper_bounds[0] - upper_bounds[1]).abs() > 1e-6,
        "changing only alpha_upper should change the propagated upper bound \
         (got {} and {})",
        upper_bounds[0],
        upper_bounds[1]
    );
}

/// Regression test for #3008: sequential BaB prefilter now checks domain_is_violation.
///
/// The simple_network computes |x1-x2| for input in [-1,1]². With threshold=1.5,
/// the property |x1-x2| > 1.5 is false for inputs near x1≈x2 (output ≈ 0).
/// Initial CROWN bounds are ambiguous (lower < 1.5, upper ≈ 2.0), so BaB enters
/// the branching loop. After splitting, some child domains have upper < 1.5,
/// which should trigger PotentialViolation via the prefilter.
#[ntest::timeout(10000)]
#[test]
fn test_sequential_bab_prefilter_detects_violation_3008() {
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(5),
        batch_size: 1,
        verify_upper_bound: false,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, 1.5).unwrap();

    // The property |x1-x2| > 1.5 is false (x1=x2=0 gives output 0).
    // With the #3008 fix, the prefilter detects violation in a child domain.
    // Without the fix, BaB would still terminate correctly but waste more work.
    assert!(
        matches!(
            result.result,
            BabVerificationStatus::PotentialViolation | BabVerificationStatus::Unknown { .. }
        ),
        "Expected PotentialViolation or Unknown for false property, got {:?}",
        result.result
    );
    // Verify it's not incorrectly reported as Verified
    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "Must not return Verified for a false property"
    );
}
