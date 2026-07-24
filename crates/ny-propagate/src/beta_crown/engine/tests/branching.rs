// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::prelude::*;
use crate::{AdaIN1dLayer, GroupNormLayer, InstanceNorm1dLayer, LayerNormLayer};

fn assert_sequential_preact_bias(layer: Layer, target_shape: &[usize], expected: ArrayD<f32>) {
    let mut network = Network::new();
    network.add_layer(layer);
    network.add_layer(Layer::ReLU(ReLULayer));

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let recovered = verifier
        .sequential_preact_bias(&network, 1, target_shape)
        .expect("producer bias should be recoverable");

    assert_eq!(recovered, expected);
}

#[ntest::timeout(5000)]
#[test]
fn test_domain_ordering() {
    // Test that BabDomain ordering works correctly (higher lb = higher priority)
    let bounds1 =
        vec![BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap()];
    let bounds2 = bounds1.clone();

    let d1 = BabDomain::root(bounds1, 1.0, 2.0).unwrap();
    let d2 = BabDomain::root(bounds2, 0.5, 2.0).unwrap();

    // d1 has higher lower bound, should be "greater"
    assert!(d1 > d2);

    // Test with heap
    let mut heap = BinaryHeap::new();
    heap.push(d2);
    heap.push(d1);

    // Should pop d1 first (higher lower bound)
    assert_eq!(heap.pop().unwrap().lower_bound, 1.0);
    assert_eq!(heap.pop().unwrap().lower_bound, 0.5);
}

#[ntest::timeout(10000)]
#[test]
fn test_babsr_branching_heuristic() {
    // Test BaBSR branching with BoundImpact heuristic actually branches.
    //
    // simple_network: y = relu(x1-x2) + relu(-x1+x2) = |x1-x2|
    //   W1 = [[1,-1],[-1,1]], W2 = [[1,1]], no bias
    //   Output range for input [-1,1]^2: [0, 2]
    //   CROWN lower bound: 0 (exact for this symmetric network)
    //
    // Threshold 0.5: true minimum is 0 (at x1=x2), so property is falsifiable.
    // CROWN lower bound 0 < 0.5 forces BaB to enter the branching loop.
    // Previous threshold of -5.0 was trivially verified without any branching.
    let network = simple_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(10),
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, 0.5).unwrap();

    // Branching must have occurred (initial CROWN LB=0 < threshold=0.5)
    assert!(
        result.domains_explored >= 2,
        "BaBSR must branch when threshold exceeds CROWN lower bound; got domains_explored={}",
        result.domains_explored,
    );

    // Property is falsifiable (min output = 0 at x1=x2), cannot be verified
    assert_ne!(
        result.result,
        BabVerificationStatus::Verified,
        "output = |x1-x2| has min 0; threshold 0.5 is unreachable",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_fsb_branching_heuristic() {
    // Test FSB branching (FilteredSmartBranching) actually branches.
    // Same network and reasoning as test_babsr_branching_heuristic:
    // threshold 0.5 exceeds CROWN LB, forcing the BaB loop to branch.
    // Previous threshold of -5.0 was trivially verified without any branching.
    let network = simple_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(10),
        branching_heuristic: BranchingHeuristic::FilteredSmartBranching,
        fsb_candidates: 4,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, 0.5).unwrap();

    // FSB must branch when initial bounds are insufficient
    assert!(
        result.domains_explored >= 2,
        "FSB must branch when threshold exceeds CROWN lower bound; got domains_explored={}",
        result.domains_explored,
    );

    // Property is falsifiable (min output = 0 at x1=x2)
    assert_ne!(
        result.result,
        BabVerificationStatus::Verified,
        "output = |x1-x2| has min 0; threshold 0.5 is unreachable",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_babsr_selects_high_impact_neuron() {
    // Test that BaBSR (BoundImpact) selects the highest-scoring neuron when
    // the signed lA column carries negative coefficients, matching the
    // reference score kernel's lower-bound setting.
    //
    // Network: Linear(2->3) -> ReLU -> Linear(3->1)
    //   W1 = [[1,0],[0,1],[1,-1]], no bias
    //   W2 = [[-0.1, -0.2, -1.5]], no bias
    //   Input: [-0.5, 0.5]^2
    //
    // Pre-ReLU bounds (from Layer 0 output = W1 @ x):
    //   h0 = x1,         range [-0.5, 0.5]  (unstable)
    //   h1 = x2,         range [-0.5, 0.5]  (unstable)
    //   h2 = x1 - x2,    range [-1.0, 1.0]  (unstable)
    //
    // CROWN backward from output identity [1] through W2:
    //   signed lA columns = [-0.1, -0.2, -1.5]
    //
    // Intercepts for symmetric [-a, a]: (-(-a)*a) / (a-(-a)) = a/2
    //   h0: 0.25,  h1: 0.25,  h2: 0.5
    //
    // With zero producer bias, the reference kernel reduces to
    // abs(min(lA, 0) * intercept):
    //   neuron 0: |-0.1 * 0.25| = 0.025
    //   neuron 1: |-0.2 * 0.25| = 0.050
    //   neuron 2: |-1.5 * 0.50| = 0.750  <- highest
    //
    // BaBSR must select neuron 2 (score 30x higher than neuron 0).
    // A trivial always-pick-neuron-0 heuristic fails this test.
    //
    // Ref: alpha-beta-CROWN heuristics/babsr.py:babsr_score
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = arr2(&[[-0.1, -0.2, -1.5]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Construct domain with hand-computed layer bounds
    let pre_relu_bounds = BoundedTensor::new(
        arr1(&[-0.5, -0.5, -1.0]).into_dyn(),
        arr1(&[0.5, 0.5, 1.0]).into_dyn(),
    )
    .unwrap();
    let post_relu_bounds = BoundedTensor::new(
        arr1(&[0.0, 0.0, 0.0]).into_dyn(),
        arr1(&[0.5, 0.5, 1.0]).into_dyn(),
    )
    .unwrap();
    // Output = W2 @ relu(h): range [-1.65, 0]
    let output_bounds =
        BoundedTensor::new(arr1(&[-1.65]).into_dyn(), arr1(&[0.0]).into_dyn()).unwrap();

    let domain = BabDomain::root(
        vec![pre_relu_bounds, post_relu_bounds, output_bounds],
        -1.65,
        0.0,
    )
    .unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let choice = verifier
        .select_split_neuron(&network, &input, &domain)
        .unwrap()
        .expect("BaBSR should find an unstable neuron");

    assert_eq!(
        choice.0, 1,
        "selected layer should be the ReLU layer (layer 1)"
    );
    assert_eq!(
        choice.1, 2,
        "BaBSR should select neuron 2 (score 0.75) over neuron 0 (0.025) and neuron 1 (0.05)"
    );
    assert!(
        choice.2 > 0.0,
        "BaBSR score should be positive; got {}",
        choice.2,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_babsr_prefers_larger_affine_bias_2513() {
    let linear1 =
        LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[10.0, 1.0]))).unwrap();
    let linear2 = LinearLayer::new(arr2(&[[-1.0, -1.0]]), None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    let input = BoundedTensor::new(
        arr1(&[-11.0, -2.0]).into_dyn(),
        arr1(&[-9.0, 0.0]).into_dyn(),
    )
    .unwrap();

    let pre_relu_bounds =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let post_relu_bounds =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let output_bounds =
        BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[0.0]).into_dyn()).unwrap();

    let domain = BabDomain::root(
        vec![pre_relu_bounds, post_relu_bounds, output_bounds],
        -2.0,
        0.0,
    )
    .unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let choice = verifier
        .select_split_neuron(&network, &input, &domain)
        .unwrap()
        .expect("BaBSR should find an unstable neuron");

    assert_eq!(choice.0, 1, "selection should target the ReLU layer");
    assert_eq!(
        choice.1, 0,
        "bias-aware BaBSR should prefer neuron 0: both intervals tie, but bias 10.0 beats bias 1.0"
    );
    assert!(
        choice.2 > 1.0,
        "bias-aware BaBSR score should reflect the larger affine bias, got {}",
        choice.2
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_sequential_preact_bias_recovers_affine_norm_biases_2513() {
    let layer_norm = LayerNormLayer::new(arr1(&[1.0, 1.0]), arr1(&[10.0, 1.0]), 1e-5).unwrap();
    assert_sequential_preact_bias(
        Layer::LayerNorm(layer_norm),
        &[2],
        arr1(&[10.0, 1.0]).into_dyn(),
    );

    let group_norm = GroupNormLayer::new(arr1(&[1.0, 1.0]), arr1(&[3.0, 5.0]), 1, 1e-5).unwrap();
    assert_sequential_preact_bias(
        Layer::GroupNorm(group_norm),
        &[2, 3],
        arr2(&[[3.0, 3.0, 3.0], [5.0, 5.0, 5.0]]).into_dyn(),
    );

    let instance_norm =
        InstanceNorm1dLayer::new(arr1(&[1.0, 1.0]), arr1(&[7.0, 11.0]), 1e-5).unwrap();
    assert_sequential_preact_bias(
        Layer::InstanceNorm1d(instance_norm),
        &[2, 2],
        arr2(&[[7.0, 7.0], [11.0, 11.0]]).into_dyn(),
    );

    let adain = AdaIN1dLayer::new(
        InstanceNorm1dLayer::new(arr1(&[2.0, 4.0]), arr1(&[1.0, 2.0]), 1e-5).unwrap(),
        arr1(&[3.0, 5.0]),
        arr1(&[7.0, 11.0]),
    )
    .unwrap();
    // Fixed-style AdaIN collapses to InstanceNorm with beta = style_gamma * beta + style_beta.
    assert_sequential_preact_bias(
        Layer::AdaIN1d(adain),
        &[2, 3],
        arr2(&[[10.0, 10.0, 10.0], [21.0, 21.0, 21.0]]).into_dyn(),
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_babsr_fanout_matches_sequential_path() {
    // Symmetric bounds yield equal intercepts, so selection depends on the
    // signed fan-out sum. Use negative merged weights so the reference score
    // kernel stays informative in lower-bound mode.
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    // Sequential reference: merged fan-out is equivalent to a single linear [-1, -2].
    let mut seq_network = Network::new();
    seq_network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), None).unwrap(),
    ));
    seq_network.add_layer(Layer::ReLU(ReLULayer));
    seq_network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0, -2.0]]), None).unwrap(),
    ));

    let seq_pre_bounds =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let seq_output_bounds =
        BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[0.0]).into_dyn()).unwrap();
    let seq_domain = BabDomain::root(vec![seq_pre_bounds, seq_output_bounds], -3.0, 0.0).unwrap();

    let seq_choice = verifier
        .select_split_neuron(&seq_network, &input, &seq_domain)
        .unwrap()
        .expect("sequential BaBSR should find an unstable neuron");
    assert_eq!(seq_choice.0, 1, "sequential choice should be on ReLU layer");
    assert_eq!(seq_choice.1, 1, "sequential BaBSR should pick neuron 1");

    // Graph with fan-out + merge:
    // branch_a weights [-5, -1], branch_b weights [4, -1] => merged [-1, -2].
    // A buggy fan-out implementation that sums |branch| instead of signed sums
    // would prefer neuron 0 (|-5|+|4| = 9) instead of neuron 1 (|-1|+|-1| = 2).
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "branch_a",
        Layer::Linear(LinearLayer::new(arr2(&[[-5.0, -1.0]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "branch_b",
        Layer::Linear(LinearLayer::new(arr2(&[[4.0, -1.0]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sum",
        Layer::Add(AddLayer),
        vec!["branch_a".to_string(), "branch_b".to_string()],
    ));
    graph.set_output("sum");

    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let graph_domain = GraphBabDomain::root(node_bounds, -1.0, 1.0, &input, false).unwrap();
    let relu_nodes = vec!["relu".to_string()];
    let unstable = verifier.find_unstable_graph_neurons(&graph, &graph_domain, &relu_nodes);
    assert_eq!(unstable.len(), 2, "both ReLU neurons should be unstable");

    let graph_choice = verifier
        .select_graph_branch(&graph, &graph_domain, &unstable)
        .unwrap();
    assert_eq!(graph_choice.0, "relu");
    assert_eq!(
        graph_choice.1, seq_choice.1,
        "graph BaBSR should match sequential choice"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_babsr_prefers_larger_affine_bias_2513() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[10.0, 1.0]))).unwrap(),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(arr2(&[[-1.0, -1.0]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        arr1(&[-11.0, -2.0]).into_dyn(),
        arr1(&[-9.0, 0.0]).into_dyn(),
    )
    .unwrap();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let domain = GraphBabDomain::root(node_bounds, -2.0, 0.0, &input, false).unwrap();
    let relu_nodes = vec!["relu".to_string()];
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);
    let choice = verifier
        .select_graph_branch(&graph, &domain, &unstable)
        .unwrap();

    assert_eq!(choice.0, "relu");
    assert_eq!(
        choice.1, 0,
        "graph BaBSR should prefer the larger recovered producer bias when intervals and lA tie"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_babsr_missing_coefficients_use_zero_fallback_1864() {
    // Regression for #1864 parity: when CROWN coefficient extraction cannot produce
    // all ReLU sensitivities (here due to unsupported ReduceSum backward in the
    // lightweight branching scorer), missing coefficients must be treated as 0.0
    // so those neurons are deprioritized.
    let mut network = Network::new();
    network.add_layer(Layer::AddConstant(AddConstantLayer::new(
        arr1(&[0.0, 0.0]).into_dyn(),
    )));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::ReduceSum(ReduceSumLayer::new(vec![-1], false)));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Two unstable neurons with different intercepts:
    // neuron 0 intercept = 0.05, neuron 1 intercept = 0.5.
    // With old fallback=1.0, BaBSR picked neuron 1. With fallback=0.0 for missing
    // coeffs, neuron 1 is deprioritized and neuron 0 is selected.
    let pre_relu_bounds =
        BoundedTensor::new(arr1(&[-0.1, -1.0]).into_dyn(), arr1(&[0.1, 1.0]).into_dyn()).unwrap();
    let post_relu_bounds =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[0.1, 1.0]).into_dyn()).unwrap();
    let reduce_bounds =
        BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.1]).into_dyn()).unwrap();

    let domain = BabDomain::root(
        vec![pre_relu_bounds, post_relu_bounds, reduce_bounds],
        -1.0,
        1.0,
    )
    .unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let choice = verifier
        .select_split_neuron(&network, &input, &domain)
        .unwrap()
        .expect("BaBSR should find an unstable ReLU neuron");

    assert_eq!(choice.0, 1, "selection should target the ReLU layer");
    assert_eq!(
        choice.1, 0,
        "missing CROWN coefficients must use 0.0 fallback and be deprioritized"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_network_bab_no_branch_returns_unknown_1865() {
    // Regression test for #1865: Network BaB must return Unknown (not Verified)
    // when child domains have no unstable neurons left to branch on but bounds
    // are too loose to verify. Before #1865 fix: the unresolved child domains
    // disappeared from the search tree and core.rs returned false Verified.
    //
    // Network: y = relu(x), single neuron.
    //   W1 = [[1]], W2 = [[1]], no bias
    //   Input: [-1, 1]
    //
    // After the root domain's single ReLU neuron is split:
    //   Active child (x >= 0): y = x, bounds [0, 1], not verified against threshold 0.5
    //   Inactive child (x <= 0): y = 0, bounds [0, 0], not verified against threshold 0.5
    // Both children have 0 unstable neurons → select_split_neuron returns Ok(None)
    // → had_no_branch = true → must return Unknown.
    let w1 = arr2(&[[1.0]]);
    let w2 = arr2(&[[1.0]]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 100,
        timeout: Duration::from_secs(5),
        ..Default::default()
    });

    // Threshold 0.5: relu(x) for x in [-1, 1] has output in [0, 1].
    // Initial lower bound 0 < 0.5, not verified. After splitting the single
    // neuron, both children still can't verify (active: lb=0, inactive: lb=0).
    // No more neurons to branch → must be Unknown.
    let result = verifier.verify(&network, &input, 0.5).unwrap();
    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "Bug #1865: Network BaB returned Verified despite no-branch unresolved domains. \
         After splitting the only ReLU neuron, child domains had no neurons to branch on \
         but bounds were too loose. Got {:?}",
        result.result,
    );
    // Verify it returns Unknown with a reason mentioning no unstable neurons
    if let BabVerificationStatus::Unknown { ref reason } = result.result {
        assert!(
            reason.contains("unstable")
                || reason.contains("no branch")
                || reason.contains("No unstable"),
            "Expected Unknown reason to mention no unstable neurons, got: {reason}",
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_bab_depth_limit_returns_unknown_1865() {
    // Regression test for #1865: Network BaB (core.rs) was silently dropping
    // depth-limited domains and returning Verified when the queue emptied.
    //
    // Network: simple_network() = W2 * relu(W1 * x)
    //   W1 = [[1, -1], [-1, 1]], W2 = [[1, 1]]
    //   Input: [-1, 1]^2
    //
    // With max_depth=0 (no splitting allowed), the initial CROWN bounds are
    // too loose to verify a tight threshold. Before #1865 fix: the domain
    // hit max depth, was silently dropped via bare `continue`, the queue
    // emptied, and core.rs returned Verified. After fix: returns Unknown.
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 1, // Depth 1 — root domain enters loop at depth 0, children at depth 1 hit limit
        timeout: Duration::from_secs(5),
        ..Default::default()
    });

    // Threshold 0.5: The network's true output range includes values near 0
    // (when both relu inputs are zero), so initial CROWN bounds can't verify
    // output > 0.5. With max_depth=1, child domains hit the depth limit.
    let result = verifier.verify(&network, &input, 0.5).unwrap();
    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "Bug #1865: Network BaB returned Verified despite depth-limited domains. \
         The max_depth limit was reached but core.rs silently dropped the domain. Got {:?}",
        result.result,
    );
    // Verify it returns Unknown with a reason mentioning depth
    if let BabVerificationStatus::Unknown { ref reason } = result.result {
        assert!(
            reason.contains("depth") || reason.contains("Depth"),
            "Expected Unknown reason to mention depth, got: {reason}",
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_babsr_matches_single_objective_1857() {
    // Regression test for #1857: multi-objective graph BaB must use BaBSR
    // (BoundImpact) branching when configured, not intercept-only.
    //
    // Graph: _input -> relu -> branch_a (W=[-5,-1]) \
    //                       -> branch_b (W=[4,-1]) -> sum (Add)
    //
    // With symmetric input [-1,1]^2, both neurons have identical intercepts
    // (= 0.5). Intercept-only scoring picks neuron 0 (arbitrary tie-break).
    // BaBSR scoring uses signed CROWN coefficients: merged output weights are [-1, -2],
    // so neuron 1 has higher sensitivity and BaBSR should pick neuron 1.
    //
    // Before #1857 fix: multi-objective path always used intercept-only,
    // diverging from the single-objective BaBSR result.
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    // Build the same graph used in test_graph_babsr_fanout_matches_sequential_path
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "branch_a",
        Layer::Linear(LinearLayer::new(arr2(&[[-5.0, -1.0]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "branch_b",
        Layer::Linear(LinearLayer::new(arr2(&[[4.0, -1.0]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sum",
        Layer::Add(AddLayer),
        vec!["branch_a".to_string(), "branch_b".to_string()],
    ));
    graph.set_output("sum");

    // Single-objective reference: GraphBabDomain with BaBSR
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let single_domain =
        GraphBabDomain::root(node_bounds.clone(), -1.0, 1.0, &input, false).unwrap();
    let relu_nodes = vec!["relu".to_string()];
    let unstable = verifier.find_unstable_graph_neurons(&graph, &single_domain, &relu_nodes);
    assert_eq!(unstable.len(), 2, "both ReLU neurons should be unstable");

    let single_choice = verifier
        .select_graph_branch(&graph, &single_domain, &unstable)
        .unwrap();
    assert_eq!(single_choice.0, "relu");
    assert_eq!(
        single_choice.1, 1,
        "single-objective BaBSR should pick neuron 1 (higher CROWN coefficient)"
    );

    // Multi-objective: same graph, same bounds, BoundImpact configured.
    // Two objectives with loose bounds so both are unverified.
    let multi_domain = MultiObjectiveGraphBabDomain::root(
        node_bounds,
        vec![(-1.0, 1.0), (-2.0, 2.0)],
        &input,
        &[0.5, 0.5],
        false,
    )
    .unwrap();
    let multi_unstable =
        verifier.find_unstable_graph_neurons_multi(&graph, &multi_domain, &relu_nodes);
    assert_eq!(
        multi_unstable.len(),
        2,
        "multi-objective: both ReLU neurons should be unstable"
    );

    let multi_choice = verifier
        .select_graph_branch_multi(&graph, &multi_domain, &multi_unstable, &[], None)
        .unwrap();
    assert_eq!(multi_choice.0, "relu");
    assert_eq!(
        multi_choice.1, single_choice.1,
        "Bug #1857: multi-objective BaBSR should match single-objective BaBSR choice. \
         Multi-objective picked neuron {} but single-objective picked neuron {}. \
         This indicates multi-objective path is ignoring BranchingHeuristic::BoundImpact.",
        multi_choice.1, single_choice.1,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_graph_babsr_matches_single_objective_bias_case_2513() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[10.0, 1.0]))).unwrap(),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(arr2(&[[-1.0, -1.0]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        arr1(&[-11.0, -2.0]).into_dyn(),
        arr1(&[-9.0, 0.0]).into_dyn(),
    )
    .unwrap();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let relu_nodes = vec!["relu".to_string()];
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let single_domain =
        GraphBabDomain::root(node_bounds.clone(), -2.0, 0.0, &input, false).unwrap();
    let single_unstable = verifier.find_unstable_graph_neurons(&graph, &single_domain, &relu_nodes);
    let single_choice = verifier
        .select_graph_branch(&graph, &single_domain, &single_unstable)
        .unwrap();

    let multi_domain = MultiObjectiveGraphBabDomain::root(
        node_bounds,
        vec![(-2.0, 0.0), (-1.0, 1.0)],
        &input,
        &[0.0, 0.0],
        false,
    )
    .unwrap();
    let multi_unstable =
        verifier.find_unstable_graph_neurons_multi(&graph, &multi_domain, &relu_nodes);
    let multi_choice = verifier
        .select_graph_branch_multi(&graph, &multi_domain, &multi_unstable, &[], None)
        .unwrap();

    assert_eq!(single_choice.0, "relu");
    assert_eq!(
        single_choice.1, 0,
        "single-objective graph BaBSR should prefer the larger recovered producer bias"
    );
    assert_eq!(
        multi_choice.1, single_choice.1,
        "multi-objective graph BaBSR should match the single-objective bias-aware choice"
    );
}

/// Algorithm audit: verify graph CROWN coefficient backward pass against
/// hand-computed expected values for a simple linear -> ReLU -> linear network.
///
/// Network: _input(2) -> relu(2) -> linear(1) with W = [[-3.0, -0.5]]
///
/// CROWN backward from output identity [1.0]:
///   At relu node: lA = [1.0] @ W = [-3.0, -0.5] (1x2)
///     coeff[0] = sum(|lA[:, 0]|) = |3.0| = 3.0
///     coeff[1] = sum(|lA[:, 1]|) = |0.5| = 0.5
///
/// With symmetric input bounds [-1, 1]^2 (all unstable), both neurons have
/// identical intercepts, so BaBSR should prefer neuron 0 (coeff 3.0 > 0.5).
///
/// Ref: alpha-beta-CROWN heuristics/babsr.py:babsr_score, heuristics/utils.py:compute_ratio
#[ntest::timeout(10000)]
#[test]
fn test_graph_babsr_crown_coefficients_hand_computed() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(LinearLayer::new(arr2(&[[-3.0, -0.5]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear");

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let domain = GraphBabDomain::root(node_bounds, -1.0, 1.0, &input, false).unwrap();
    let relu_nodes = vec!["relu".to_string()];
    let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);
    assert_eq!(unstable.len(), 2, "both neurons should be unstable");

    // Verify BaBSR picks neuron 0 (higher coefficient: 3.0 vs 0.5).
    // Both neurons have identical symmetric bounds so intercepts are equal.
    let choice = verifier
        .select_graph_branch(&graph, &domain, &unstable)
        .unwrap();
    assert_eq!(choice.0, "relu", "branching should target the relu node");
    assert_eq!(
        choice.1, 0,
        "hand-computed CROWN: neuron 0 has coeff 3.0 vs neuron 1 coeff 0.5, \
         so BaBSR should select neuron 0"
    );

    // With zero producer bias and negative lA, the reference kernel reduces to
    // abs(min(lA, 0) * intercept). Symmetric bounds [-1, 1] give intercept 0.5,
    // so neuron 0 scores |-3.0 * 0.5| = 1.5.
    let expected_score = 0.5 * 3.0;
    assert!(
        (choice.2 - expected_score).abs() < 1e-6,
        "BaBSR score should be |-3.0 * 0.5| = {expected_score}, got {}",
        choice.2
    );
}

/// Algorithm audit: verify CROWN coefficient backward through a deeper graph
/// with two ReLU layers. Hand-compute expected coefficients layer by layer.
///
/// Graph: _input(2) -> relu0(2) -> linear1 W=[[2,-1],[0,3]] -> relu1(2) -> linear2 W=[[-1,-4]] -> output
///
/// CROWN backward from output [1.0]:
///   At linear2: lA = [1.0] @ [[-1,-4]] = [-1, -4]  (1x2 matrix)
///   At relu1: coeff[relu1,0] = |1| = 1.0, coeff[relu1,1] = |4| = 4.0
///     Apply relu1 slope (all unstable with symmetric bounds): slope = u/(u-l) = 0.5
///     lA_after_relu1 = [-1*0.5, -4*0.5] = [-0.5, -2.0]
///   At linear1: lA = [-0.5, -2.0] @ [[2,-1],[0,3]] = [-1.0, -5.5]
///   At relu0: coeff[relu0,0] = |1.0| = 1.0, coeff[relu0,1] = |5.5| = 5.5
///
/// With symmetric bounds, BaBSR should prefer relu1 neuron 1 (coeff=4.0) over relu1 neuron 0 (coeff=1.0),
/// and relu0 neuron 1 (coeff=5.5) over relu0 neuron 0 (coeff=1.0).
/// Overall best: relu0 neuron 1 (5.5) beats relu1 neuron 1 (4.0).
#[ntest::timeout(10000)]
#[test]
fn test_graph_babsr_two_relu_layers_hand_computed() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(arr2(&[[2.0, -1.0], [0.0, 3.0]]), None).unwrap()),
        vec!["relu0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(arr2(&[[-1.0, -4.0]]), None).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // relu1 pre-activation bounds: both neurons unstable with symmetric bounds.
    let relu1_pre_bounds =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let mut node_bounds_plain = std::collections::HashMap::new();
    node_bounds_plain.insert("linear1".to_string(), relu1_pre_bounds);
    // Output node bounds needed for CROWN backward initialization (#2159).
    // linear2 W=[[-1,-4]] produces 1-dim output; actual values don't affect output_dim.
    node_bounds_plain.insert(
        "linear2".to_string(),
        BoundedTensor::new(arr1(&[-10.0]).into_dyn(), arr1(&[10.0]).into_dyn()).unwrap(),
    );

    let domain = GraphBabDomain::root(node_bounds_plain, -1.0, 1.0, &input, false).unwrap();

    let relu_nodes = vec!["relu0".to_string(), "relu1".to_string()];
    let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);
    assert_eq!(unstable.len(), 4, "all 4 neurons should be unstable");

    let choice = verifier
        .select_graph_branch(&graph, &domain, &unstable)
        .unwrap();

    // Hand-computed expected: relu0 neuron 1 has coeff 5.5 (highest).
    // intercept for symmetric [-1,1] is 0.5, so score = 5.5 * 0.5 = 2.75
    assert_eq!(
        choice.0, "relu0",
        "deepest ReLU neuron with highest sensitivity should be selected"
    );
    assert_eq!(
        choice.1, 1,
        "hand-computed: relu0 neuron 1 has coeff 5.5 (highest of all 4 neurons)"
    );
    let expected_score = 5.5 * 0.5;
    assert!(
        (choice.2 - expected_score).abs() < 1e-5,
        "BaBSR score should be |-5.5 * 0.5| = {expected_score}, got {}",
        choice.2
    );
}

/// Verify BaBSR branching correctly excludes constrained neurons in a two-ReLU
/// graph network with fan-out. After constraining relu0 neuron 1 (the global best),
/// BaBSR should select the next-best neuron: relu1 neuron 1 (|coeff|=4.0).
///
/// This validates that `find_unstable_graph_neurons` respects `GraphSplitHistory`
/// constraints and that BaBSR coefficient computation is independent of constraint
/// state (coefficients are computed for the full graph backward, but scoring only
/// considers unconstrained unstable neurons).
///
/// Re: #1817, verifies W1 BaBSR commit (4095006) with constrained-neuron branches.
#[ntest::timeout(10000)]
#[test]
fn test_graph_babsr_with_constrained_neuron_selects_next_best() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    // Same graph as test_graph_babsr_two_relu_layers_hand_computed:
    // _input(2) -> relu0(2) -> linear1 W=[[2,-1],[0,3]] -> relu1(2) -> linear2 W=[[-1,-4]]
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(arr2(&[[2.0, -1.0], [0.0, 3.0]]), None).unwrap()),
        vec!["relu0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(arr2(&[[-1.0, -4.0]]), None).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let relu1_pre_bounds =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let mut node_bounds = std::collections::HashMap::new();
    node_bounds.insert("linear1".to_string(), relu1_pre_bounds);
    // Output node bounds needed for CROWN backward initialization (#2159).
    node_bounds.insert(
        "linear2".to_string(),
        BoundedTensor::new(arr1(&[-10.0]).into_dyn(), arr1(&[10.0]).into_dyn()).unwrap(),
    );

    // Create a child domain where relu0 neuron 1 is constrained (active).
    // In the unconstrained case, relu0 neuron 1 had the highest coeff (5.5).
    let root_domain = GraphBabDomain::root(node_bounds, -1.0, 1.0, &input, false).unwrap();
    let constraint = GraphNeuronConstraint {
        node_name: "relu0".to_string(),
        neuron_idx: 1,
        is_active: true,
        score: 2.75,
    };
    let child_domain = root_domain
        .with_constraint(&graph, constraint, false)
        .unwrap()
        .expect("constraint should produce a valid child domain");

    // relu0 neuron 1 is now constrained; 3 neurons remain unstable
    let relu_nodes = vec!["relu0".to_string(), "relu1".to_string()];
    let unstable = verifier.find_unstable_graph_neurons(&graph, &child_domain, &relu_nodes);
    assert_eq!(
        unstable.len(),
        3,
        "after constraining relu0[1], 3 neurons should remain unstable: relu0[0], relu1[0], relu1[1]"
    );
    assert!(
        !unstable.contains(&("relu0".to_string(), 1)),
        "relu0 neuron 1 should be excluded from unstable set (it is constrained)"
    );

    // BaBSR should now select relu1 neuron 1 (|coeff|=4.0, score=2.0)
    // instead of relu0 neuron 1 (which was the global best at 5.5*0.5=2.75).
    let choice = verifier
        .select_graph_branch(&graph, &child_domain, &unstable)
        .unwrap();
    assert_eq!(
        choice.0, "relu1",
        "with relu0[1] constrained, relu1 neuron 1 should be the next-best BaBSR target"
    );
    assert_eq!(
        choice.1, 1,
        "relu1 neuron 1 (coeff=4.0) should beat relu0 neuron 0 (coeff=1.0) and relu1 neuron 0 (coeff=1.0)"
    );
    let expected_score = 4.0 * 0.5;
    assert!(
        (choice.2 - expected_score).abs() < 1e-5,
        "BaBSR score should be |-4.0 * 0.5| = {expected_score}, got {}. \
         Note: CROWN coefficients are computed for the full graph (including constrained neurons), \
         but scoring only considers unconstrained unstable neurons.",
        choice.2
    );
}

/// Regression test for #2098 + #2481: constructing a ReLU node with empty inputs
/// panics in debug builds (debug_assert in GraphNode::new, added by #2481).
/// This replaces the original #2098 test that relied on constructing malformed
/// nodes — constructor-level validation now prevents the scenario that the
/// downstream defensive handling was designed for.
#[test]
#[should_panic(expected = "requires at least 1 input(s) but got 0")]
fn test_graph_node_rejects_relu_with_empty_inputs_2098() {
    // GraphNode::new debug_assert fires on arity violation (#2481).
    let _node = GraphNode::new("relu_bad", Layer::ReLU(ReLULayer), vec![]);
}

/// Regression test for #1915: select_graph_branch must return Err on empty unstable list.
#[ntest::timeout(5000)]
#[test]
fn test_select_graph_branch_empty_unstable_returns_error_1915() {
    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    // Build a minimal graph with a ReLU node (reuse pattern from other tests)
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0, 1.0]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear1");

    let input = BoundedTensor::new(
        arr1(&[-1.0f32, -1.0]).into_dyn(),
        arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let domain =
        GraphBabDomain::root(std::collections::HashMap::new(), -1.0, 1.0, &input, false).unwrap();

    // Empty unstable list — must return Err, not panic.
    let empty_unstable: Vec<(String, usize)> = vec![];
    let result = verifier.select_graph_branch(&graph, &domain, &empty_unstable);
    assert!(
        result.is_err(),
        "empty unstable list must return Err (#1915)"
    );
}

/// Regression test for #1915: select_graph_branch_multi must return Err on empty unstable list.
#[ntest::timeout(5000)]
#[test]
fn test_select_graph_branch_multi_empty_unstable_returns_error_1915() {
    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    // Build a minimal graph with a ReLU node (reuse pattern from other tests)
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0, 1.0]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear1");

    let input = BoundedTensor::new(
        arr1(&[-1.0f32, -1.0]).into_dyn(),
        arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let domain = MultiObjectiveGraphBabDomain::root(
        std::collections::HashMap::new(),
        vec![(-1.0, 1.0), (-1.0, 1.0)],
        &input,
        &[0.0, 0.0],
        false,
    )
    .unwrap();

    // Empty unstable list — must return Err, not panic.
    let empty_unstable: Vec<(String, usize)> = vec![];
    let result = verifier.select_graph_branch_multi(&graph, &domain, &empty_unstable, &[], None);
    assert!(
        result.is_err(),
        "empty unstable list must return Err for multi-objective branch selection (#1915)"
    );
}

/// Regression test for #2095: empty sequential layer bounds must return Err
/// instead of fabricating output_dim=1 for CROWN coefficient initialization.
#[ntest::timeout(5000)]
#[test]
fn test_compute_crown_coefficients_empty_layer_bounds_returns_error_2095() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });
    let network = simple_network();
    let domain = BabDomain::root(vec![], -1.0, 1.0).unwrap();

    let err = verifier
        .compute_crown_coefficients(&network, &domain)
        .expect_err("empty layer_bounds must error");
    assert!(
        matches!(err, ny_core::NyError::InternalError(ref msg) if msg.contains("layer_bounds empty")),
        "expected InternalError mentioning empty layer_bounds, got {err:?}"
    );
}

/// Regression test for #2095: missing graph output bounds must return Err
/// instead of fabricating output_dim=1 for graph BaBSR coefficients.
#[ntest::timeout(5000)]
#[test]
fn test_select_graph_branch_missing_output_bounds_returns_error_2095() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-1.0f32, -1.0]).into_dyn(),
        arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .unwrap();
    let domain =
        GraphBabDomain::root(std::collections::HashMap::new(), -1.0, 1.0, &input, false).unwrap();
    let relu_nodes = vec!["relu".to_string()];
    let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);
    assert!(
        !unstable.is_empty(),
        "input interval should yield unstable ReLU candidates"
    );

    let err = verifier
        .select_graph_branch(&graph, &domain, &unstable)
        .expect_err("missing output bounds must error");
    assert!(
        matches!(err, ny_core::NyError::InternalError(ref msg) if msg.contains("missing output bounds")),
        "expected InternalError mentioning missing output bounds, got {err:?}"
    );
}

/// Regression test for #2271: unsupported layers in sequential CROWN coefficient
/// backward pass must stop propagation instead of silently passing through.
#[ntest::timeout(10000)]
#[test]
fn test_compute_crown_coefficients_stops_at_unsupported_layer_2271() {
    use crate::ExpLayer;

    // Network: Linear(2→2) → Exp → ReLU → Linear(2→1)
    // The Exp layer is unsupported in compute_crown_coefficients.
    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[1.0, 1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::Exp(ExpLayer));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    // Layer bounds: one per layer (4 layers).
    // Bounds chosen so the ReLU at layer 2 has unstable neurons.
    let layer_bounds = vec![
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap(),
        BoundedTensor::new(arr1(&[0.1, 0.1]).into_dyn(), arr1(&[2.0, 2.0]).into_dyn()).unwrap(),
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[1.5, 1.5]).into_dyn()).unwrap(),
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap(),
    ];
    let domain = BabDomain::root(layer_bounds, -1.0, 2.0).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let coeffs = verifier
        .compute_crown_coefficients(&network, &domain)
        .expect("should succeed even with unsupported layers");

    // The backward pass starts from the output (layer 3: Linear) and goes:
    //   layer 3 (Linear) → coeffs updated
    //   layer 2 (ReLU) → coefficients recorded, slopes applied
    //   layer 1 (Exp) → unsupported, BREAK
    //   layer 0 (Linear) → never reached
    //
    // So we should have coefficients for the ReLU layer (layer_idx=2) but NOT
    // for any layer below the Exp. The key invariant is that the function doesn't
    // crash and returns valid partial results.
    assert!(
        !coeffs.is_empty(),
        "should have coefficients for the ReLU layer above the unsupported Exp"
    );

    // Verify coefficients exist for layer 2 (the ReLU) neurons
    assert!(
        coeffs.contains_key(&(2, 0)),
        "should have coefficient for ReLU neuron (2, 0)"
    );
    assert!(
        coeffs.contains_key(&(2, 1)),
        "should have coefficient for ReLU neuron (2, 1)"
    );

    // Verify propagation stopped at the Exp layer: no coefficients for layers 0 or 1
    for &(layer_idx, neuron_idx) in coeffs.keys() {
        assert!(
            layer_idx >= 2,
            "propagation should stop at Exp (layer 1): found coefficient for ({layer_idx}, {neuron_idx})"
        );
    }

    // Verify exactly 2 coefficients (one per ReLU neuron)
    assert_eq!(
        coeffs.len(),
        2,
        "should have exactly 2 coefficients (ReLU neurons only)"
    );

    // Verify coefficient values are finite (NaN safety)
    for (&(layer_idx, neuron_idx), value) in &coeffs {
        assert!(
            value.is_finite(),
            "coefficient ({layer_idx}, {neuron_idx}) should be finite, got {value}"
        );
    }
}

/// Regression test for #2539: kFSB intercept ranking must use ascending sort
/// (smallest intercept first), matching alpha-beta-CROWN kfsb.py:92 `largest=False`.
///
/// Strategy: construct a network with 4 unstable neurons having distinct intercept
/// scores. Use `KfsbInterceptOnly` with `fsb_candidates = 1` so the main ranking
/// picks only the 1 largest-intercept neuron, and the intercept backup ranking
/// picks only the 1 smallest-intercept neuron. With correct ascending sort, the
/// smallest-intercept neuron enters the evaluation set. With the old descending
/// bug, both rankings would pick the same neuron (largest intercept), leaving the
/// smallest-intercept neuron unevaluated.
///
/// The test verifies that kFSB produces a valid result (not None) and completes
/// without error, exercising both the main and intercept ranking paths.
///
/// Ref: alpha-beta-CROWN complete_verifier/heuristics/kfsb.py:92
///   `itb_idx = torch.topk(all_itb, topk, largest=False)  # k-smallest elements.`
#[ntest::timeout(10000)]
#[test]
fn test_kfsb_intercept_ranking_ascending_sort_2539() {
    // Network: Linear(2→4) → ReLU → Linear(4→1)
    // Identity-like first linear so pre-activation bounds approximately match input.
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.3, 0.7]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = arr2(&[[1.0, 1.0, 1.0, 1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    // Craft asymmetric pre-activation bounds to get distinct intercept scores.
    // intercept = (-lower * upper) / (upper - lower)
    //
    // Neuron 0: [-0.1, 0.2] → intercept = 0.02/0.3 ≈ 0.0667  (smallest)
    // Neuron 1: [-1.0, 2.0] → intercept = 2.0/3.0  ≈ 0.6667
    // Neuron 2: [-2.0, 1.0] → intercept = 2.0/3.0  ≈ 0.6667
    // Neuron 3: [-3.0, 3.0] → intercept = 9.0/6.0  = 1.5000  (largest)
    let pre_relu_bounds = BoundedTensor::new(
        arr1(&[-0.1, -1.0, -2.0, -3.0]).into_dyn(),
        arr1(&[0.2, 2.0, 1.0, 3.0]).into_dyn(),
    )
    .unwrap();

    // Post-ReLU bounds: [max(0,l), u] for each unstable neuron
    let post_relu_bounds = BoundedTensor::new(
        arr1(&[0.0, 0.0, 0.0, 0.0]).into_dyn(),
        arr1(&[0.2, 2.0, 1.0, 3.0]).into_dyn(),
    )
    .unwrap();

    // Output bounds (scalar output)
    let output_bounds =
        BoundedTensor::new(arr1(&[-5.0]).into_dyn(), arr1(&[6.0]).into_dyn()).unwrap();

    let domain = BabDomain::root(
        vec![pre_relu_bounds, post_relu_bounds, output_bounds],
        -5.0,
        6.0,
    )
    .unwrap();

    let input =
        BoundedTensor::new(arr1(&[-3.0, -3.0]).into_dyn(), arr1(&[3.0, 3.0]).into_dyn()).unwrap();

    // KfsbInterceptOnly with fsb_candidates=1:
    //   main_ranked (descending by intercept) → top-1 = neuron 3 (intercept=1.5)
    //   intercept_ranked (ascending by intercept) → top-1 = neuron 0 (intercept=0.067)
    //
    // With the old descending bug, intercept_ranked would ALSO pick neuron 3,
    // so only 1 candidate would be evaluated. With the fix, 2 candidates are evaluated.
    let config = BetaCrownConfig {
        max_domains: 10,
        timeout: Duration::from_secs(5),
        branching_heuristic: BranchingHeuristic::KfsbInterceptOnly,
        fsb_candidates: 1,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier
        .select_split_neuron(&network, &input, &domain)
        .expect("kFSB should not error");

    // kFSB must find a neuron to split (there are 4 unstable neurons).
    let (layer_idx, neuron_idx, score) =
        result.expect("kFSB should select a neuron from 4 unstable candidates");

    assert_eq!(layer_idx, 1, "selected neuron must be on the ReLU layer");
    assert!(score.is_finite(), "kFSB score must be finite, got {score}");

    // With fsb_candidates=1 in intercept-only mode:
    //   main_ranked (descending by intercept) → top-1 = neuron 3 (intercept=1.5)
    //   intercept_ranked (ascending by intercept) → top-1 = neuron 0 (intercept=0.067)
    //
    // Neuron 0 can ONLY enter the eval set via the intercept backup ranking.
    // With the old descending bug, intercept_ranked would also pick neuron 3
    // (dedup to 1 candidate), and neuron 0 would never be considered.
    //
    // The child-bound evaluation for this network favors neuron 0 (score=-6.0)
    // over neuron 3 — so neuron 0 is selected. This selection proves the
    // ascending sort is working: neuron 0 could not have been selected if it
    // never entered eval_candidates.
    assert_eq!(
        neuron_idx, 0,
        "Bug #2539 regression: kFSB must select neuron 0 (smallest intercept = 0.067), \
         which only enters the eval set via ascending intercept backup ranking. \
         If neuron {neuron_idx} was selected instead, the intercept sort may be wrong \
         (descending instead of ascending per kfsb.py:92 largest=False)."
    );

    println!(
        "kFSB intercept-only selected: layer={layer_idx}, neuron={neuron_idx}, score={score:.6}"
    );
}

/// Regression test for #2539: kFSB (non-intercept-only) branching smoke test.
/// Exercises the `Kfsb` heuristic through `verify()` to confirm end-to-end
/// correctness after the sort direction fix.
#[ntest::timeout(10000)]
#[test]
fn test_kfsb_branching_heuristic_smoke_2539() {
    let network = simple_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(10),
        branching_heuristic: BranchingHeuristic::Kfsb,
        fsb_candidates: 4,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -5.0).unwrap();

    println!("kFSB Result: {:?}", result);
    assert_eq!(result.result, BabVerificationStatus::Verified);
}

/// Regression test for #2539: KfsbInterceptOnly branching smoke test.
/// Exercises the `KfsbInterceptOnly` heuristic through `verify()`.
#[ntest::timeout(10000)]
#[test]
fn test_kfsb_intercept_only_branching_heuristic_smoke_2539() {
    let network = simple_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(10),
        branching_heuristic: BranchingHeuristic::KfsbInterceptOnly,
        fsb_candidates: 4,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -5.0).unwrap();

    println!("kFSB InterceptOnly Result: {:?}", result);
    assert_eq!(result.result, BabVerificationStatus::Verified);
}

/// Regression test for #2461: sequential BaBSR CROWN coefficient computation must
/// produce finite slopes when ReLU pre-activation bounds have near-zero width.
/// Without the RELU_RELAX_MIN_WIDTH guard at sequential.rs:697, `u / (u - l)`
/// would overflow to Inf for pathologically narrow crossing intervals.
#[ntest::timeout(10000)]
#[test]
fn test_sequential_crown_coefficients_near_zero_width_2461() {
    // Network: Linear(2→2) → ReLU → Linear(2→1)
    // Identity first linear so pre-activation bounds == input bounds.
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[1.0, 1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    // Layer bounds: neuron 0 has near-zero width crossing (u - l = 2e-20 < 1e-8 guard),
    // neuron 1 is normal crossing.
    let layer_bounds = vec![
        BoundedTensor::new(
            arr1(&[-1e-20_f32, -1.0]).into_dyn(),
            arr1(&[1e-20_f32, 1.0]).into_dyn(),
        )
        .unwrap(),
        // ReLU output bounds (post-activation): [0, u] for crossing neurons
        BoundedTensor::new(
            arr1(&[0.0, 0.0]).into_dyn(),
            arr1(&[1e-20_f32, 1.0]).into_dyn(),
        )
        .unwrap(),
        // Output bounds
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap(),
    ];
    let domain = BabDomain::root(layer_bounds, -1.0, 2.0).unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let coeffs = verifier
        .compute_crown_coefficients(&network, &domain)
        .expect("near-zero-width bounds must not cause errors");

    // All coefficients must be finite (not Inf or NaN).
    for (&(layer_idx, neuron_idx), &score) in &coeffs {
        assert!(
            score.is_finite(),
            "coefficient for ({layer_idx}, {neuron_idx}) is {score}, expected finite. \
             Without RELU_RELAX_MIN_WIDTH guard, near-zero-width crossing neurons \
             produce Inf slopes."
        );
    }
}

/// Regression test for #2461: graph BaBSR CROWN coefficient computation must
/// produce finite slopes when ReLU pre-activation bounds have near-zero width.
/// Without the RELU_RELAX_MIN_WIDTH guard at graph.rs:440, `u / (u - l)`
/// would overflow to Inf for pathologically narrow crossing intervals.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_coefficients_near_zero_width_2461() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(LinearLayer::new(arr2(&[[3.0, 0.5]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear");

    // Input with near-zero-width crossing: neuron 0 has width 2e-20 < 1e-8 guard.
    let input = BoundedTensor::new(
        arr1(&[-1e-20_f32, -1.0]).into_dyn(),
        arr1(&[1e-20_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let domain = GraphBabDomain::root(node_bounds, -1.0, 1.0, &input, false).unwrap();
    let relu_nodes = vec!["relu".to_string()];
    let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);

    // At least one neuron should be unstable (the normal-width one).
    assert!(
        !unstable.is_empty(),
        "expected at least one unstable neuron"
    );

    let choice = verifier
        .select_graph_branch(&graph, &domain, &unstable)
        .expect("near-zero-width bounds must not cause branching errors");

    assert!(
        choice.2.is_finite(),
        "branching score is {}, expected finite. Without RELU_RELAX_MIN_WIDTH guard, \
         near-zero-width crossing neurons produce Inf slopes that corrupt all scores.",
        choice.2
    );
}

// --- Sign BaB branching tests (Part of #3769) ---

/// Create a simple Sign-only network for BaB branching tests.
///
/// Network: Linear(2->2) -> Sign -> Linear(2->1)
///   W1 = [[1, -1], [-1, 1]], W2 = [[1, 1]], no bias
///
/// Pre-Sign bounds for input [-1, 1]^2:
///   h0 = x1 - x2, range [-2, 2] (unstable: crosses zero)
///   h1 = -x1 + x2, range [-2, 2] (unstable: crosses zero)
///
/// Post-Sign: sign(x1-x2) + sign(-x1+x2) = 0 for all x (exact output always 0).
/// IBP bounds: [-2, 2]. CROWN bounds: [-2, 2] (Sign spans-zero relaxation).
///
/// With threshold 0.5: CROWN lower bound -2 < 0.5, so BaB must branch.
fn simple_sign_network() -> Network {
    use crate::layers::misc::SignLayer;

    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = arr2(&[[1.0, 1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::Sign(SignLayer));
    network.add_layer(Layer::Linear(linear2));
    network
}

#[ntest::timeout(10000)]
#[test]
fn test_sign_babsr_branches_3769() {
    // Regression for #3769: BaB must branch on Sign neurons.
    // Before the fix, BaBSR only considered ReLU neurons as branching
    // candidates, returning None for Sign-only models and reporting
    // "No unstable ReLU neurons left in some domains."
    //
    // This test verifies that BaB finds unstable Sign neurons (both cross
    // zero) and explores child domains via zero-threshold branching.
    let network = simple_sign_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(10),
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, 0.5).unwrap();

    // BaB must explore >1 domains (root + at least one child from Sign branching).
    // Before #3769, this was 0 or 1 because Sign neurons were invisible to branching.
    assert!(
        result.domains_explored >= 2,
        "BaBSR must branch on Sign neurons; got domains_explored={} (reason: {:?})",
        result.domains_explored,
        result.result,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sign_kfsb_branches_3769() {
    // Regression for #3769: KFSB branching also handles Sign neurons.
    // KFSB uses forward/backward coefficient scoring. The Sign fixed-CROWN
    // proxy slope (`sign_fixed_crown_proxy_slope`) enables coefficient-based
    // scoring for Sign neurons: boundary cases [0,u] or [l,0] get non-zero
    // slopes, while fully-unstable [l,u] gets slope 0.
    let network = simple_sign_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(10),
        branching_heuristic: BranchingHeuristic::Kfsb,
        fsb_candidates: 4,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, 0.5).unwrap();

    assert!(
        result.domains_explored >= 2,
        "KFSB must branch on Sign neurons; got domains_explored={} (reason: {:?})",
        result.domains_explored,
        result.result,
    );
}

/// #mo-scorer-fix regression: kfsb / fsb / width must produce DIFFERENT
/// scores (and not all the same pick) in `select_graph_branch_multi`.
///
/// Before the fix, all three heuristics silently fell to the intercept-only
/// fallback (`want_babsr` only covered BoundImpact), so they produced
/// IDENTICAL branch picks with IDENTICAL scores — the measured metaroom
/// scorer degeneracy. Crafted case: neuron 0 has a WIDE pre-activation but a
/// tiny output coefficient; neuron 1 is narrow with a large coefficient, so
/// width must aim at neuron 0 while the coefficient-aware kFSB family aims at
/// neuron 1, and every heuristic's score is its own kernel's value.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_scorer_degeneracy_fixed_kfsb_fsb_width_differ() {
    let build_graph = || {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(LinearLayer::new(arr2(&[[-0.1, -5.0]]), None).unwrap()),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear2");
        graph
    };
    let graph = build_graph();
    // Pre-activations: n0 in [-3, 1] (width 4, coeff 0.1), n1 in [-0.5, 0.5]
    // (width 1, coeff 5.0).
    let input =
        BoundedTensor::new(arr1(&[-3.0, -0.5]).into_dyn(), arr1(&[1.0, 0.5]).into_dyn()).unwrap();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let relu_nodes = vec!["relu".to_string()];
    let objectives = vec![vec![1.0f32]];

    // The per-heuristic scorers are gated behind `NY_MO_SCORER_FIX=1`
    // (#scorer-fix-default-off): without it, width/kfsb/fsb all collapse onto
    // the shared intercept fallback — the exact degeneracy this test pins as
    // fixed. Enable the gate so the test exercises the fixed scorers it names.
    let picks = crate::tests::with_serialized_env_vars(&[("NY_MO_SCORER_FIX", "1")], || {
        let mut picks = Vec::new();
        for heuristic in [
            BranchingHeuristic::LargestBoundWidth,
            BranchingHeuristic::Kfsb,
            BranchingHeuristic::FilteredSmartBranching,
        ] {
            let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                branching_heuristic: heuristic.clone(),
                fsb_candidates: 2,
                kfsb_reduce_op: crate::beta_crown::config::KfsbReduceOp::Max,
                beta_iterations: 0,
                ..Default::default()
            });
            let domain = MultiObjectiveGraphBabDomain::root(
                node_bounds.clone(),
                vec![(-5.0, 5.0)],
                &input,
                &[0.0],
                false,
            )
            .unwrap();
            let unstable = verifier.find_unstable_graph_neurons_multi(&graph, &domain, &relu_nodes);
            assert_eq!(unstable.len(), 2, "both neurons unstable ({heuristic:?})");
            let (node, neuron, score) = verifier
                .select_graph_branch_multi(&graph, &domain, &unstable, &objectives, None)
                .unwrap();
            assert_eq!(node, "relu");
            assert!(
                score.is_finite(),
                "{heuristic:?} must produce a finite real score, got {score}"
            );
            picks.push((heuristic, neuron, score));
        }
        picks
    });

    // Width aims at the wide neuron 0 with its REAL width score (u - l = 4).
    assert_eq!(picks[0].1, 0, "width must pick the widest neuron");
    assert!(
        (picks[0].2 - 4.0).abs() < 1e-5,
        "width score must be the real pre-activation width, got {}",
        picks[0].2
    );
    // The coefficient-aware kFSB family must aim at the high-impact neuron 1
    // (the degeneracy had ALL heuristics returning one shared fallback pick).
    assert_eq!(picks[1].1, 1, "kfsb must pick the high-coefficient neuron");
    assert_eq!(picks[2].1, 1, "fsb must pick the high-coefficient neuron");
    // And no two heuristics may return the SAME score value.
    assert_ne!(picks[0].2, picks[1].2, "width vs kfsb scores must differ");
    assert_ne!(picks[0].2, picks[2].2, "width vs fsb scores must differ");
    assert_ne!(
        picks[1].2, picks[2].2,
        "kfsb vs fsb scores must differ (different prescore reduce + candidate sets)"
    );
}
