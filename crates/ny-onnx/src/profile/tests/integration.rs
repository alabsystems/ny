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
