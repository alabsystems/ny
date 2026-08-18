// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_fixtures::require_test_model;

fn assert_profile_result_matches(alias: &ProfileResult, legacy: &ProfileResult) {
    assert_eq!(alias.layers.len(), legacy.layers.len());
    assert_eq!(alias.initial_width, legacy.initial_width);
    assert_eq!(alias.final_width, legacy.final_width);
    assert_eq!(alias.total_expansion, legacy.total_expansion);
    assert_eq!(alias.max_growth_layer, legacy.max_growth_layer);
}

#[ntest::timeout(10000)]
#[test]
fn test_profile_bounds_simple_mlp() {
    let model_path = require_test_model("simple_mlp.onnx");

    let config = ProfileConfig::default();
    let result = profile_bounds(&model_path, &config).expect("Failed to profile bounds");

    // Should have layers
    assert!(!result.layers.is_empty());

    // All growth ratios should be positive
    for layer in &result.layers {
        assert!(
            layer.growth_ratio > 0.0 || layer.growth_ratio.is_nan(),
            "Growth ratio should be positive for layer {}",
            layer.name
        );
    }

    // Print summary for debugging
    eprintln!("{}", result.summary());
}

#[ntest::timeout(10000)]
#[test]
fn test_analyze_profile_matches_profile_bounds_simple_mlp() {
    let model_path = require_test_model("simple_mlp.onnx");
    let config = ProfileConfig::default();

    let alias = analyze_profile(&model_path, &config).expect("Failed to analyze profile");
    let legacy = profile_bounds(&model_path, &config).expect("Failed to profile bounds");

    assert_profile_result_matches(&alias, &legacy);
}

#[ntest::timeout(10000)]
#[test]
fn test_analyze_profile_exports_match_existing_signatures() {
    let _model_api: fn(&crate::OnnxModel, &ProfileConfig) -> Result<ProfileResult, ProfileError> =
        analyze_profile_model;
    let _graph_api: fn(
        &ny_propagate::GraphNetwork,
        &ProfileConfig,
        &[usize],
    ) -> Result<ProfileResult, ProfileError> = analyze_profile_graph;
}

#[ntest::timeout(10000)]
#[test]
fn test_profile_error_round_trips_ny_error_variant() {
    let err = ProfileError::propagation(
        "profile",
        ny_core::NyError::UnsupportedConfiguration("typed propagation failure".into()),
    );
    let ny_error: ny_core::NyError = err.into();

    match ny_error {
        ny_core::NyError::UnsupportedConfiguration(msg) => {
            assert_eq!(msg, "typed propagation failure");
        }
        other => panic!(
            "expected UnsupportedConfiguration after ProfileError round-trip, got: {other:?}"
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_profile_graph_rejects_invalid_epsilon_with_custom_input() {
    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("bounded input");
    let config = ProfileConfig {
        epsilon: f32::NAN,
        continue_after_overflow: true,
        input: Some(input),
    };

    let err = profile_bounds_graph(&ny_propagate::GraphNetwork::new(), &config, &[1])
        .expect_err("invalid epsilon must fail before graph analysis");
    assert!(err.to_string().contains("epsilon"), "err = {err}");
}

#[ntest::timeout(10000)]
#[test]
fn test_profile_graph_marks_fallback_after_propagation_failure_as_overflow() {
    use ndarray::{arr1, arr2};
    use ny_propagate::layers::{Layer, LinearLayer, ReLULayer};
    use ny_propagate::{GraphNetwork, GraphNode};
    use ny_tensor::BoundedTensor;

    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[1.0_f32, 0.0]]), None).expect("valid linear fixture");
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    // The layer expects two inputs, while this deliberately malformed
    // analysis input has one. Diagnostic continuation may retain the input
    // bounds, but the resulting layer must never be reported as tight/safe.
    let input = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("bounded input");
    let config = ProfileConfig {
        epsilon: 0.5,
        continue_after_overflow: true,
        input: Some(input),
    };

    let result =
        profile_bounds_graph(&graph, &config, &[1]).expect("diagnostic continuation succeeds");
    assert_eq!(result.overflow_at_layer, Some(0));
    assert_eq!(result.layers[0].status, BoundStatus::Overflow);
    assert_eq!(
        result.layers[1].status,
        BoundStatus::Overflow,
        "a descendant computed from substituted fallback bounds must remain failed"
    );
    assert_eq!(result.difficulty_score, 100.0);
}
