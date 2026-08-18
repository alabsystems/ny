// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Runtime contract tests for the default-off named-node stem branch treatment.

use super::prelude::*;

fn stem_branch_fixture() -> (
    BetaCrownVerifier,
    GraphNetwork,
    MultiObjectiveGraphBabDomain,
    Vec<(String, usize)>,
) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "stem_pre",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
        ),
    ));
    graph.add_node(GraphNode::new(
        "Relu_6",
        Layer::ReLU(ReLULayer),
        vec!["stem_pre".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "tail_pre",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0, 0.0]]), Some(arr1(&[0.0]))).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "Relu_15",
        Layer::ReLU(ReLULayer),
        vec!["tail_pre".to_string()],
    ));
    graph.set_output("Relu_15");

    let input = BoundedTensor::new(
        arr1(&[-10.0, -10.0]).into_dyn(),
        arr1(&[10.0, 10.0]).into_dyn(),
    )
    .unwrap();
    let mut node_bounds = std::collections::HashMap::new();
    // Relu_6[0]: width 10.1, intercept ~0.099.
    // Relu_6[1]: width 2.0, intercept 0.5.
    node_bounds.insert(
        "stem_pre".to_string(),
        BoundedTensor::new(
            arr1(&[-0.1, -1.0]).into_dyn(),
            arr1(&[10.0, 1.0]).into_dyn(),
        )
        .unwrap(),
    );
    // Relu_15[0]: width 4.0, intercept 1.0 (global legacy winner).
    node_bounds.insert(
        "tail_pre".to_string(),
        BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap(),
    );
    let domain =
        MultiObjectiveGraphBabDomain::root(node_bounds, vec![(-1.0, 1.0)], &input, &[0.0], false)
            .unwrap();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::LargestBoundWidth,
        ..Default::default()
    });
    let unstable = verifier.find_unstable_graph_neurons_multi(
        &graph,
        &domain,
        &["Relu_6".to_string(), "Relu_15".to_string()],
    );
    assert_eq!(
        unstable,
        vec![
            ("Relu_6".to_string(), 0),
            ("Relu_6".to_string(), 1),
            ("Relu_15".to_string(), 0),
        ]
    );
    (verifier, graph, domain, unstable)
}

fn clear_branch_experiment_env(env: &mut ny_test_utils::env::EnvEditor) {
    for key in [
        "NY_BETA_GPU_PROBE",
        "NY_BRANCH_LA",
        "NY_BRANCH_LA_PROBE",
        "NY_BRANCH_STEM",
        "NY_BRANCH_STEM_K",
        "NY_BRANCH_STEM_LAYERS",
        "NY_BRANCH_STEM_NODES",
        "NY_BRANCH_STEM_PROBE",
        "NY_BRANCH_TRACE",
        "NY_MO_GATHER_SCORE",
        "NY_MO_SCORER_FIX",
    ] {
        env.remove(key);
    }
}

fn arm_sealed_stem_env(env: &mut ny_test_utils::env::EnvEditor, nodes: &str, k: &str) {
    env.set("NY_BRANCH_STEM", "1");
    env.set("NY_BRANCH_STEM_K", k);
    env.set("NY_BRANCH_STEM_NODES", nodes);
    // Mandatory engagement evidence for the sealed treatment.
    env.set("NY_BRANCH_STEM_PROBE", "1");
    env.set("NY_BRANCH_TRACE", "1");
}

#[test]
fn explicit_stem_nodes_engage_legacy_intercept_and_bypass_mo_scorer_fix() {
    let (verifier, graph, domain, unstable) = stem_branch_fixture();

    crate::tests::with_env_edits(|env| {
        clear_branch_experiment_env(env);
        env.set("NY_MO_SCORER_FIX", "1");

        let fixed_width_pick = verifier
            .select_graph_branch_multi(&graph, &domain, &unstable, &[], &[0.0], None)
            .unwrap();
        assert_eq!(
            (fixed_width_pick.0.as_str(), fixed_width_pick.1),
            ("Relu_6", 0),
            "without stem, the enabled scorer fix must use real width"
        );
        assert!((fixed_width_pick.2 - 10.1).abs() < 1.0e-5);

        arm_sealed_stem_env(env, "Relu_6,Relu_9,Relu_12", "8");
        let stem_pick = verifier
            .select_graph_branch_multi(&graph, &domain, &unstable, &[], &[0.0], None)
            .unwrap();
        assert_eq!(
            (stem_pick.0.as_str(), stem_pick.1),
            ("Relu_6", 1),
            "stem must engage the explicit-node set and retain legacy intercept ranking"
        );
        assert!((stem_pick.2 - 0.5).abs() < 1.0e-6);
    });
}

#[test]
fn explicit_stem_nodes_fail_open_when_unscorable_or_depth_window_exhausted() {
    let (verifier, graph, mut domain, unstable) = stem_branch_fixture();

    crate::tests::with_env_edits(|env| {
        clear_branch_experiment_env(env);
        env.set("NY_MO_SCORER_FIX", "1");
        arm_sealed_stem_env(env, "Relu_9,Relu_12", "8");

        let no_named_candidate = verifier
            .select_graph_branch_multi(&graph, &domain, &unstable, &[], &[0.0], None)
            .unwrap();
        assert_eq!(
            (no_named_candidate.0.as_str(), no_named_candidate.1),
            ("Relu_15", 0),
            "an explicit set with no unstable/scorable member must fail open"
        );
        assert!((no_named_candidate.2 - 1.0).abs() < 1.0e-6);

        domain.history.add_constraint(GraphNeuronConstraint {
            node_name: "prior_relu".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 0.0,
        });
        arm_sealed_stem_env(env, "Relu_6,Relu_9,Relu_12", "1");
        let depth_exhausted = verifier
            .select_graph_branch_multi(&graph, &domain, &unstable, &[], &[0.0], None)
            .unwrap();
        assert_eq!(
            (depth_exhausted.0.as_str(), depth_exhausted.1),
            ("Relu_15", 0),
            "history depth at K must disable the named-node restriction"
        );
        assert!((depth_exhausted.2 - 1.0).abs() < 1.0e-6);
    });
}
