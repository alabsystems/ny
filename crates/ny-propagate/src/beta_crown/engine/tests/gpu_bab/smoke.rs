// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Smoke tests for GPU BaB: immediate verification/violation, result structure.

use super::*;

/// Test immediate verification when root bounds prove the property.
/// When verify_upper_bound=true and upper < threshold, should return Verified immediately.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_immediate_verification() {
    let graph = simple_graph_network();

    // Tight input bounds that lead to tight output bounds
    let input =
        BoundedTensor::new(arr1(&[0.1, 0.1]).into_dyn(), arr1(&[0.2, 0.2]).into_dyn()).unwrap();

    // Objective: just take the first (only) output
    let objective = vec![1.0];

    // Set a high threshold that output bounds are definitely below
    let threshold = 100.0;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(10),
        max_domains: 100,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "Should verify immediately when upper bound < threshold"
    );
    assert_eq!(
        result.domains_explored, 1,
        "Should only explore root domain"
    );
    assert_eq!(result.domains_verified, 1);
}

/// Test immediate potential violation when root lower bound is already at/above threshold.
/// When verify_upper_bound=true and lower >= threshold, should return PotentialViolation.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_immediate_potential_violation_upper() {
    let graph = simple_graph_network();

    // Input bounds that create output with lower bound above a low threshold
    let input =
        BoundedTensor::new(arr1(&[1.0, 1.0]).into_dyn(), arr1(&[2.0, 2.0]).into_dyn()).unwrap();

    let objective = vec![1.0];

    // Very low threshold - output lower bound should be above this
    let threshold = -100.0;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(10),
        max_domains: 100,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    assert_eq!(
        result.result,
        BabVerificationStatus::potential_violation(),
        "Should return PotentialViolation when lower >= threshold"
    );
    assert_eq!(result.domains_explored, 1);
    assert_eq!(result.domains_verified, 0);
}

/// Test verify_upper_bound=false mode: want lower > threshold.
/// When lower > threshold, should return Verified immediately.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_lower_bound_verification() {
    let graph = simple_graph_network();

    // Input bounds that create positive output
    let input =
        BoundedTensor::new(arr1(&[1.0, 1.0]).into_dyn(), arr1(&[2.0, 2.0]).into_dyn()).unwrap();

    let objective = vec![1.0];

    // Low threshold - output lower bound should be above this
    let threshold = -100.0;

    let config = BetaCrownConfig {
        verify_upper_bound: false, // Want lower > threshold
        timeout: Duration::from_secs(10),
        max_domains: 100,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "Should verify when lower > threshold in lower-bound mode"
    );
    assert_eq!(result.domains_verified, 1);
}

/// Test potential violation in lower bound mode.
/// When verify_upper_bound=false and upper < threshold, should return PotentialViolation.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_lower_bound_potential_violation() {
    let graph = simple_graph_network();

    // Input bounds
    let input =
        BoundedTensor::new(arr1(&[0.1, 0.1]).into_dyn(), arr1(&[0.2, 0.2]).into_dyn()).unwrap();

    let objective = vec![1.0];

    // Very high threshold - output upper should be below this
    let threshold = 1000.0;

    let config = BetaCrownConfig {
        verify_upper_bound: false,
        timeout: Duration::from_secs(10),
        max_domains: 100,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    assert_eq!(
        result.result,
        BabVerificationStatus::potential_violation(),
        "Should return PotentialViolation when upper < threshold in lower-bound mode"
    );
}

/// Test BaB loop processes domains correctly.
/// Verifies that domains are extracted, processed, and the loop terminates.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_loop_processing() {
    let graph = simple_graph_network();

    // Input bounds that should lead to verification without branching
    // (tight bounds that produce verified root)
    let input =
        BoundedTensor::new(arr1(&[0.5, 0.5]).into_dyn(), arr1(&[0.6, 0.6]).into_dyn()).unwrap();

    let objective = vec![1.0];
    let threshold = 10.0; // High threshold to ensure verification

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 100,
        max_depth: 10,
        batch_size: 4,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("GPU BaB should succeed");

    // With tight bounds and high threshold, should verify immediately
    assert_eq!(result.result, BabVerificationStatus::Verified);
    assert!(result.time_elapsed > Duration::ZERO);
}

/// Test result structure fields are populated correctly.
#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_result_structure() {
    let graph = simple_graph_network();

    let input =
        BoundedTensor::new(arr1(&[0.1, 0.1]).into_dyn(), arr1(&[0.2, 0.2]).into_dyn()).unwrap();

    let objective = vec![1.0];
    let threshold = 100.0;

    // Note: Default config has verify_upper_bound=false (lower-bound mode).
    // With threshold=100 and small positive output bounds, this will verify
    // because lower bound > -100 is trivially true (we don't verify against 100).
    // The high threshold ensures the test exercises the result structure,
    // not the verification logic which is tested elsewhere.
    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("Should succeed");

    // Verify result structure fields are populated
    assert!(result.domains_explored >= 1, "Should explore at least root");
    assert!(
        result.time_elapsed > Duration::ZERO,
        "Should record elapsed time"
    );
    // max_depth_reached should be at least 0 (root domain is depth 0)
    assert!(
        result.max_depth_reached <= result.domains_explored,
        "Max depth {} should not exceed domains explored {}",
        result.max_depth_reached,
        result.domains_explored
    );
    // cuts_generated should be 0 (no cuts in this path)
    assert_eq!(result.cuts_generated, 0);
}

/// Test objective_bounds helper with positive coefficients.
#[ntest::timeout(10000)]
#[test]
fn test_objective_bounds_positive_coefficients() {
    // Test via public API - create a simple network
    let w = arr2(&[[1.0]]); // 1x1 identity
    let linear = LinearLayer::new(w, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear)));
    graph.set_output("linear1");

    // Input with bounds [1, 2]
    let input = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();

    // Output should also be [1, 2]
    // With objective=[1.0], bounds should be [1*1, 1*2] = [1, 2]
    let objective = vec![1.0];
    let threshold = 100.0;

    let config = BetaCrownConfig {
        verify_upper_bound: true, // Explicitly set: want upper < threshold
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("Should succeed");

    // Should verify since upper bound (2) < threshold (100)
    assert_eq!(result.result, BabVerificationStatus::Verified);
}

/// Test objective_bounds helper with negative coefficients.
/// Negative coefficient flips the bound direction.
#[ntest::timeout(10000)]
#[test]
fn test_objective_bounds_negative_coefficients() {
    let w = arr2(&[[1.0]]);
    let linear = LinearLayer::new(w, None).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear)));
    graph.set_output("linear1");

    // Input with bounds [1, 2]
    let input = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();

    // With objective=[-1.0], bounds should be [-1*2, -1*1] = [-2, -1]
    let objective = vec![-1.0];
    let threshold = 0.0; // Upper bound (-1) < threshold (0)

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("Should succeed");

    // Should verify since upper bound (-1) < threshold (0)
    assert_eq!(result.result, BabVerificationStatus::Verified);
}
