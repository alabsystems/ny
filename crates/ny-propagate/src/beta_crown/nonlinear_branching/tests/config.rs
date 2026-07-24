// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::beta_crown::branching::LayerRef;

use super::super::{
    BranchingDecision, BranchingPointMethod, NonlinearBranchingConfig, NonlinearHeuristicMethod,
};

#[ntest::timeout(5000)]
#[test]
fn test_config_defaults() {
    let config = NonlinearBranchingConfig::default();
    assert_eq!(config.num_branches, 2);
    assert_eq!(config.num_candidates, 1);
    assert!(!config.filter);
    assert!(!config.relu_only);
    assert_eq!(config.point_method, BranchingPointMethod::Uniform);
    assert_eq!(config.method, NonlinearHeuristicMethod::Bbps);
}

#[ntest::timeout(5000)]
#[test]
fn test_config_serialization() {
    let config = NonlinearBranchingConfig {
        point_method: BranchingPointMethod::Uniform,
        num_branches: 3,
        num_candidates: 5,
        filter: true,
        relu_only: false,
        method: NonlinearHeuristicMethod::BoundWidth,
        min_branch_width: 0.001,
    };

    let json = serde_json::to_string(&config).unwrap();
    let parsed: NonlinearBranchingConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.num_branches, 3);
    assert_eq!(parsed.num_candidates, 5);
    assert!(parsed.filter);
    assert_eq!(parsed.method, NonlinearHeuristicMethod::BoundWidth);
}

#[ntest::timeout(5000)]
#[test]
fn test_branching_decision_to_splits() {
    let decision = BranchingDecision {
        layer: LayerRef::Name("gelu_0".to_string()),
        neuron_idx: 5,
        points: vec![0.5],
        score: 1.0,
        original_bounds: (0.0, 1.0),
        input_index: None,
        norm_inv_rms: None,
    };

    let splits = decision.to_splits().expect("valid decision");
    assert_eq!(splits.len(), 2);
    assert!(splits[0].lower_bound().is_none());
    assert_eq!(splits[0].upper_bound(), Some(0.5));
    assert_eq!(splits[1].lower_bound(), Some(0.5));
    assert!(splits[1].upper_bound().is_none());
}
