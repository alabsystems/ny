// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::result::BabVerificationStatus;
use crate::beta_crown::BetaCrownVerifier;
use crate::layers::LinearLayer;
use crate::network::GraphNode;
use crate::{GraphNetwork, Layer, ReLULayer};
use ndarray::{arr1, arr2};
use std::time::Duration;

use super::super::domain_filter::PreFilterOutcome;
use super::test_domain;

fn adaptive_route_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".into()],
    ));
    graph.add_node(GraphNode::new(
        "output",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
        vec!["relu".into()],
    ));
    graph.set_output("output");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn graph_relu_split_adaptive_route_preserves_legacy_result_and_accounting() {
    let _env_lock = ny_test_utils::env::lock_env();
    let gate_name =
        crate::beta_crown::engine::graph::adaptive_microbatch::ADAPTIVE_MICROBATCH_GATE_ENV;
    let graph = adaptive_route_graph();
    let input =
        ny_tensor::BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .unwrap();
    let base = BetaCrownConfig {
        batch_size: 1,
        max_domains: 16,
        max_depth: 4,
        timeout: Duration::from_secs(5),
        enable_cuts: false,
        use_alpha_crown: false,
        ..Default::default()
    };
    let gate_dark = ny_test_utils::env::ScopedEnvVar::set(gate_name, "0");
    let preset_legacy = BetaCrownVerifier::new(BetaCrownConfig {
        auto_enlarge_batch_size: true,
        ..base.clone()
    })
    .verify_graph_relu_split(&graph, &input, &[1.0], 0.5)
    .unwrap();
    drop(gate_dark);

    let _gate_on = ny_test_utils::env::ScopedEnvVar::set(gate_name, "1");
    let legacy = BetaCrownVerifier::new(base.clone())
        .verify_graph_relu_split(&graph, &input, &[1.0], 0.5)
        .unwrap();
    let adaptive = BetaCrownVerifier::new(BetaCrownConfig {
        auto_enlarge_batch_size: true,
        ..base
    })
    .verify_graph_relu_split(&graph, &input, &[1.0], 0.5)
    .unwrap();

    assert_eq!(adaptive.result, legacy.result);
    assert_eq!(adaptive.domains_explored, legacy.domains_explored);
    assert_eq!(adaptive.domains_verified, legacy.domains_verified);
    assert_eq!(adaptive.max_depth_reached, legacy.max_depth_reached);
    assert_eq!(preset_legacy.result, legacy.result);
    assert_eq!(preset_legacy.domains_explored, legacy.domains_explored);
    assert_eq!(preset_legacy.domains_verified, legacy.domains_verified);
    assert_eq!(preset_legacy.max_depth_reached, legacy.max_depth_reached);
}

/// Root lower bound above threshold -> Verified (default verify_upper_bound=false).
#[ntest::timeout(5000)]
#[test]
fn test_check_root_early_exit_verified_lower_above_threshold() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());

    let result = verifier
        .check_root_early_exit(5.0, 10.0, 3.0, &mut lifecycle)
        .expect("should not error");

    let r = result.expect("should return Some for verified root");
    assert!(
        matches!(r.result, BabVerificationStatus::Verified),
        "lower(5.0) > threshold(3.0) should verify, got {:?}",
        r.result
    );
    assert_eq!(lifecycle.domains_explored, 1);
    assert_eq!(lifecycle.domains_verified, 1);
}

/// Root upper bound below threshold -> PotentialViolation (default mode).
#[ntest::timeout(5000)]
#[test]
fn test_check_root_early_exit_violation_upper_below_threshold() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());

    let result = verifier
        .check_root_early_exit(-5.0, -1.0, 3.0, &mut lifecycle)
        .expect("should not error");

    let r = result.expect("should return Some for violation");
    assert!(
        matches!(r.result, BabVerificationStatus::PotentialViolation),
        "upper(-1.0) < threshold(3.0) should be violation, got {:?}",
        r.result
    );
    assert_eq!(lifecycle.domains_explored, 1);
    assert_eq!(lifecycle.domains_verified, 0);
}

/// Bounds that are neither verified nor violated -> None (continue to BaB).
#[ntest::timeout(5000)]
#[test]
fn test_check_root_early_exit_undecided_returns_none() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());

    let result = verifier
        .check_root_early_exit(1.0, 5.0, 3.0, &mut lifecycle)
        .expect("should not error");

    assert!(result.is_none(), "undecided root should return None");
    assert_eq!(lifecycle.domains_explored, 0);
    assert_eq!(lifecycle.domains_verified, 0);
}

/// NaN lower bound -> not verified and not violated -> None.
#[ntest::timeout(5000)]
#[test]
fn test_check_root_early_exit_nan_lower_returns_none() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());

    let result = verifier
        .check_root_early_exit(f32::NAN, 5.0, 3.0, &mut lifecycle)
        .expect("should not error");

    assert!(
        result.is_none(),
        "NaN lower bound must not verify or violate"
    );
}

/// NaN upper bound -> not verified and not violated -> None.
#[ntest::timeout(5000)]
#[test]
fn test_check_root_early_exit_nan_upper_returns_none() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());

    let result = verifier
        .check_root_early_exit(5.0, f32::NAN, 3.0, &mut lifecycle)
        .expect("should not error");

    assert!(
        result.is_none(),
        "NaN upper bound must not verify or violate"
    );
}

/// verify_upper_bound=true: upper < threshold -> Verified.
#[ntest::timeout(5000)]
#[test]
fn test_check_root_early_exit_verify_upper_mode_verified() {
    let config = BetaCrownConfig {
        verify_upper_bound: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());

    let result = verifier
        .check_root_early_exit(-1.0, 2.0, 3.0, &mut lifecycle)
        .expect("should not error");

    let r = result.expect("should return Some for verified (upper mode)");
    assert!(
        matches!(r.result, BabVerificationStatus::Verified),
        "upper(2.0) < threshold(3.0) with verify_upper=true should verify, got {:?}",
        r.result
    );
}

/// verify_upper_bound=true: lower >= threshold -> PotentialViolation.
#[ntest::timeout(5000)]
#[test]
fn test_check_root_early_exit_verify_upper_mode_violation() {
    let config = BetaCrownConfig {
        verify_upper_bound: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());

    let result = verifier
        .check_root_early_exit(5.0, 10.0, 3.0, &mut lifecycle)
        .expect("should not error");

    let r = result.expect("should return Some for violation (upper mode)");
    assert!(
        matches!(r.result, BabVerificationStatus::PotentialViolation),
        "lower(5.0) >= threshold(3.0) with verify_upper=true should be violation, got {:?}",
        r.result
    );
}

/// NaN lower bound -> domain dropped, unresolved_due_to_propagation_failure set.
#[ntest::timeout(5000)]
#[test]
fn test_pre_filter_nan_lower_domain_dropped() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut cut_pool = GraphCutPool::default();

    let mut domain = test_domain(-1.0, 1.0);
    domain.lower_bound = f32::NAN;

    let outcome = verifier
        .pre_filter_batch(vec![domain], 0.0, &mut lifecycle, &mut cut_pool)
        .expect("should not error");

    match outcome {
        PreFilterOutcome::Process(domains) => {
            assert!(domains.is_empty(), "NaN domain must be filtered out");
        }
        PreFilterOutcome::Violation => panic!("NaN domain must not trigger violation"),
    }
    assert!(
        lifecycle.unresolved_due_to_propagation_failure,
        "NaN domain must set unresolved_due_to_propagation_failure"
    );
    assert_eq!(lifecycle.domains_explored, 1);
}

/// +Inf upper bound -> domain dropped, unresolved_due_to_propagation_failure set.
#[ntest::timeout(5000)]
#[test]
fn test_pre_filter_inf_upper_domain_dropped() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut cut_pool = GraphCutPool::default();

    let mut domain = test_domain(-1.0, 1.0);
    domain.upper_bound = f32::INFINITY;

    let outcome = verifier
        .pre_filter_batch(vec![domain], 0.0, &mut lifecycle, &mut cut_pool)
        .expect("should not error");

    match outcome {
        PreFilterOutcome::Process(domains) => {
            assert!(domains.is_empty(), "Inf domain must be filtered out");
        }
        PreFilterOutcome::Violation => panic!("Inf domain must not trigger violation"),
    }
    assert!(lifecycle.unresolved_due_to_propagation_failure);
}

/// Domain verified (lower > threshold) -> counted, not passed to processing.
#[ntest::timeout(5000)]
#[test]
fn test_pre_filter_verified_domain_counted() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut cut_pool = GraphCutPool::default();
    let domain = test_domain(5.0, 10.0);

    let outcome = verifier
        .pre_filter_batch(vec![domain], 3.0, &mut lifecycle, &mut cut_pool)
        .expect("should not error");

    match outcome {
        PreFilterOutcome::Process(domains) => {
            assert!(
                domains.is_empty(),
                "verified domain must not be passed to processing"
            );
        }
        PreFilterOutcome::Violation => panic!("verified domain must not trigger violation"),
    }
    assert_eq!(lifecycle.domains_verified, 1);
    assert_eq!(lifecycle.domains_explored, 1);
}

/// Domain is a violation (upper < threshold) -> returns Violation.
#[ntest::timeout(5000)]
#[test]
fn test_pre_filter_violation_returns_violation() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut cut_pool = GraphCutPool::default();
    let domain = test_domain(-5.0, -1.0);

    let outcome = verifier
        .pre_filter_batch(vec![domain], 3.0, &mut lifecycle, &mut cut_pool)
        .expect("should not error");

    assert!(
        matches!(outcome, PreFilterOutcome::Violation),
        "upper(-1.0) < threshold(3.0) must be Violation"
    );
}

/// Domain at depth >= max_depth -> dropped, unresolved_due_to_depth set.
#[ntest::timeout(5000)]
#[test]
fn test_pre_filter_depth_limit_exceeded() {
    let config = BetaCrownConfig {
        max_depth: 5,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut cut_pool = GraphCutPool::default();

    let mut domain = test_domain(1.0, 5.0);
    domain.depth = 5;

    let outcome = verifier
        .pre_filter_batch(vec![domain], 3.0, &mut lifecycle, &mut cut_pool)
        .expect("should not error");

    match outcome {
        PreFilterOutcome::Process(domains) => {
            assert!(
                domains.is_empty(),
                "domain at max_depth must be filtered out"
            );
        }
        PreFilterOutcome::Violation => panic!("depth-limited domain must not trigger violation"),
    }
    assert!(
        lifecycle.unresolved_due_to_depth,
        "must set unresolved_due_to_depth"
    );
}

/// Undecided domain below depth limit -> passed through for processing.
#[ntest::timeout(5000)]
#[test]
fn test_pre_filter_undecided_domain_passes_through() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut cut_pool = GraphCutPool::default();
    let domain = test_domain(1.0, 5.0);

    let outcome = verifier
        .pre_filter_batch(vec![domain], 3.0, &mut lifecycle, &mut cut_pool)
        .expect("should not error");

    match outcome {
        PreFilterOutcome::Process(domains) => {
            assert_eq!(domains.len(), 1, "undecided domain must pass through");
        }
        PreFilterOutcome::Violation => panic!("undecided domain must not trigger violation"),
    }
    assert_eq!(lifecycle.domains_explored, 1);
    assert_eq!(lifecycle.domains_verified, 0);
    assert!(!lifecycle.unresolved_due_to_depth);
    assert!(!lifecycle.unresolved_due_to_propagation_failure);
}

/// Mixed batch: NaN + verified + undecided -> NaN dropped, verified counted,
/// undecided passes through.
#[ntest::timeout(5000)]
#[test]
fn test_pre_filter_mixed_batch() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut cut_pool = GraphCutPool::default();

    let mut nan_domain = test_domain(-1.0, 1.0);
    nan_domain.lower_bound = f32::NAN;
    let verified_domain = test_domain(5.0, 10.0);
    let undecided_domain = test_domain(1.0, 5.0);
    let batch = vec![nan_domain, verified_domain, undecided_domain];

    let outcome = verifier
        .pre_filter_batch(batch, 3.0, &mut lifecycle, &mut cut_pool)
        .expect("should not error");

    match outcome {
        PreFilterOutcome::Process(domains) => {
            assert_eq!(
                domains.len(),
                1,
                "only the undecided domain should pass through"
            );
        }
        PreFilterOutcome::Violation => panic!("no violation domain in batch"),
    }
    assert_eq!(lifecycle.domains_explored, 3);
    assert_eq!(lifecycle.domains_verified, 1);
    assert!(lifecycle.unresolved_due_to_propagation_failure);
}
