// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Input-split regressions for GPU BaB DomainList parity.

use super::*;

fn single_input_identity_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "id",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
    ));
    graph.set_output("id");
    graph
}

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

fn gpu_input_split_verifier(
    adv_check: i32,
    max_domains: usize,
    max_depth: usize,
    timeout: Duration,
) -> BetaCrownVerifier {
    BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains,
        max_depth,
        timeout,
        adv_check,
        ..Default::default()
    })
}

/// Regression for #3870: GPU DomainList input split must run adv_check before
/// counting the root domain, matching the CPU single-objective path.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_input_split_adv_check_finds_root_counterexample_3870() {
    let graph = single_input_identity_graph();
    let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite bounds");
    let threshold = 0.5_f32;

    let without_adv_check = gpu_input_split_verifier(-1, 2048, 12, Duration::from_secs(2));
    let with_adv_check = gpu_input_split_verifier(0, 2048, 12, Duration::from_secs(2));

    let without_adv_result = without_adv_check
        .verify_graph_gpu_domain_list(&graph, &input, &[1.0], threshold, None, None)
        .expect("adv_check=-1 should not cause errors");
    let with_adv_result = with_adv_check
        .verify_graph_gpu_domain_list(&graph, &input, &[1.0], threshold, None, None)
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
        "adv_check should return before GPU BaB counts the root domain"
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

/// Regression for #3870: enabling adv_check on the GPU DomainList path must
/// not change the final verification status.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_input_split_adv_check_disabled_preserves_status_3870() {
    let graph = splittable_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite bounds");
    let threshold = -0.5_f32;

    let without_adv_check = gpu_input_split_verifier(-1, 64, 8, Duration::from_secs(5));
    let with_adv_check = gpu_input_split_verifier(0, 64, 8, Duration::from_secs(5));

    let result_disabled = without_adv_check
        .verify_graph_gpu_domain_list(&graph, &input, &[1.0], threshold, None, None)
        .expect("adv_check=-1 should not error");
    let result_enabled = with_adv_check
        .verify_graph_gpu_domain_list(&graph, &input, &[1.0], threshold, None, None)
        .expect("adv_check=0 should not error");

    assert_eq!(
        std::mem::discriminant(&result_disabled.result),
        std::mem::discriminant(&result_enabled.result),
        "adv_check should not change GPU input-split verification outcome: disabled={:?}, enabled={:?}",
        result_disabled.result,
        result_enabled.result,
    );
}
