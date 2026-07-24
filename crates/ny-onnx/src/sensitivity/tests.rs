// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_fixtures::{
    require_test_model, require_test_model_with_hint, TRANSFORMER_TEST_MODEL_HINT,
};

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name() {
    assert_eq!(truncate_name("short", 10), "short");
    assert_eq!(truncate_name("very_long_layer_name", 10), "...er_name");
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_simple_mlp() {
    let model_path = require_test_model("simple_mlp.onnx");

    let config = SensitivityConfig {
        epsilon: 0.01,
        continue_after_overflow: true,
        input: None,
    };

    let result = analyze_sensitivity(&model_path, &config).expect("Failed to analyze sensitivity");

    // Should have multiple layers
    assert!(
        !result.layers.is_empty(),
        "Expected at least one layer in sensitivity analysis"
    );

    // All sensitivities should be positive
    for layer in &result.layers {
        assert!(
            layer.sensitivity >= 0.0,
            "Sensitivity should be non-negative for layer {}",
            layer.name
        );
    }

    // Print summary for debugging
    eprintln!("{}", result.summary());
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_transformer_block_dag() {
    // This model contains residual connections (binary Add), requiring DAG propagation.
    let model_path =
        require_test_model_with_hint("transformer_block.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let config = SensitivityConfig {
        epsilon: 0.01,
        continue_after_overflow: true,
        input: None,
    };

    let result = analyze_sensitivity(&model_path, &config).expect("Failed to analyze sensitivity");
    assert!(
        !result.layers.is_empty(),
        "Expected at least one node in DAG sensitivity analysis"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_config_default() {
    let config = SensitivityConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert!(!config.continue_after_overflow);
    assert!(config.input.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_sensitivity_is_high_sensitivity() {
    let layer = LayerSensitivity {
        name: "test_layer".to_string(),
        layer_type: "Linear".to_string(),
        input_width: 0.02,
        output_width: 0.2,
        sensitivity: 10.0,
        mean_output_width: 0.15,
        output_shape: vec![10],
        has_overflow: false,
        propagation_failed: false,
    };

    // sensitivity=10 is above threshold=5
    assert!(layer.is_high_sensitivity(5.0));
    // sensitivity=10 is not above threshold=15
    assert!(!layer.is_high_sensitivity(15.0));
    // exact threshold
    assert!(!layer.is_high_sensitivity(10.0));
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_sensitivity_is_contractive() {
    let contractive = LayerSensitivity {
        name: "relu".to_string(),
        layer_type: "ReLU".to_string(),
        input_width: 1.0,
        output_width: 0.5,
        sensitivity: 0.5,
        mean_output_width: 0.3,
        output_shape: vec![10],
        has_overflow: false,
        propagation_failed: false,
    };
    assert!(contractive.is_contractive());

    let expanding = LayerSensitivity {
        name: "linear".to_string(),
        layer_type: "Linear".to_string(),
        input_width: 0.5,
        output_width: 1.0,
        sensitivity: 2.0,
        mean_output_width: 0.8,
        output_shape: vec![10],
        has_overflow: false,
        propagation_failed: false,
    };
    assert!(!expanding.is_contractive());

    let neutral = LayerSensitivity {
        name: "identity".to_string(),
        layer_type: "Identity".to_string(),
        input_width: 1.0,
        output_width: 1.0,
        sensitivity: 1.0,
        mean_output_width: 1.0,
        output_shape: vec![10],
        has_overflow: false,
        propagation_failed: false,
    };
    assert!(!neutral.is_contractive());
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_result_layers_by_sensitivity() {
    let result = SensitivityResult {
        layers: vec![
            LayerSensitivity {
                name: "low".to_string(),
                layer_type: "ReLU".to_string(),
                input_width: 1.0,
                output_width: 0.5,
                sensitivity: 0.5,
                mean_output_width: 0.3,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: false,
            },
            LayerSensitivity {
                name: "high".to_string(),
                layer_type: "Linear".to_string(),
                input_width: 1.0,
                output_width: 10.0,
                sensitivity: 10.0,
                mean_output_width: 8.0,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: false,
            },
            LayerSensitivity {
                name: "medium".to_string(),
                layer_type: "Softmax".to_string(),
                input_width: 1.0,
                output_width: 3.0,
                sensitivity: 3.0,
                mean_output_width: 2.0,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: false,
            },
        ],
        total_sensitivity: 15.0,
        max_sensitivity: 10.0,
        max_sensitivity_layer: Some(1),
        input_epsilon: 0.01,
        final_width: 5.0,
        overflow_at_layer: None,
    };

    let sorted = result.layers_by_sensitivity();
    assert_eq!(sorted.len(), 3);
    assert_eq!(sorted[0].name, "high"); // sensitivity=10
    assert_eq!(sorted[1].name, "medium"); // sensitivity=3
    assert_eq!(sorted[2].name, "low"); // sensitivity=0.5
}

/// Regression test for #2601: NaN sensitivities must sort last, not corrupt rankings.
#[ntest::timeout(10000)]
#[test]
fn test_layers_by_sensitivity_nan_last_2601() {
    let result = SensitivityResult {
        layers: vec![
            LayerSensitivity {
                name: "nan_layer".to_string(),
                layer_type: "Linear".to_string(),
                input_width: 0.0,
                output_width: 0.0,
                sensitivity: f32::NAN,
                mean_output_width: 0.0,
                output_shape: vec![10],
                has_overflow: true,
                propagation_failed: false,
            },
            LayerSensitivity {
                name: "finite_high".to_string(),
                layer_type: "ReLU".to_string(),
                input_width: 1.0,
                output_width: 5.0,
                sensitivity: 5.0,
                mean_output_width: 4.0,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: false,
            },
            LayerSensitivity {
                name: "finite_low".to_string(),
                layer_type: "Softmax".to_string(),
                input_width: 1.0,
                output_width: 1.0,
                sensitivity: 1.0,
                mean_output_width: 0.8,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: false,
            },
        ],
        total_sensitivity: 6.0,
        max_sensitivity: 5.0,
        max_sensitivity_layer: Some(1),
        input_epsilon: 0.01,
        final_width: 5.0,
        overflow_at_layer: None,
    };

    let sorted = result.layers_by_sensitivity();
    assert_eq!(sorted.len(), 3);
    // Finite values descending, NaN last
    assert_eq!(sorted[0].name, "finite_high");
    assert_eq!(sorted[1].name, "finite_low");
    assert_eq!(sorted[2].name, "nan_layer");
    assert!(sorted[2].sensitivity.is_nan());
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_result_hot_spots() {
    let result = SensitivityResult {
        layers: vec![
            LayerSensitivity {
                name: "low".to_string(),
                layer_type: "ReLU".to_string(),
                input_width: 1.0,
                output_width: 0.5,
                sensitivity: 0.5,
                mean_output_width: 0.3,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: false,
            },
            LayerSensitivity {
                name: "high".to_string(),
                layer_type: "Linear".to_string(),
                input_width: 1.0,
                output_width: 10.0,
                sensitivity: 10.0,
                mean_output_width: 8.0,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: false,
            },
            LayerSensitivity {
                name: "very_high".to_string(),
                layer_type: "Softmax".to_string(),
                input_width: 1.0,
                output_width: 100.0,
                sensitivity: 100.0,
                mean_output_width: 80.0,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: false,
            },
        ],
        total_sensitivity: 500.0,
        max_sensitivity: 100.0,
        max_sensitivity_layer: Some(2),
        input_epsilon: 0.01,
        final_width: 100.0,
        overflow_at_layer: None,
    };

    let hot_spots_5 = result.hot_spots(5.0);
    assert_eq!(hot_spots_5.len(), 2);
    assert!(hot_spots_5.iter().any(|l| l.name == "high"));
    assert!(hot_spots_5.iter().any(|l| l.name == "very_high"));

    let hot_spots_50 = result.hot_spots(50.0);
    assert_eq!(hot_spots_50.len(), 1);
    assert_eq!(hot_spots_50[0].name, "very_high");

    let hot_spots_1000 = result.hot_spots(1000.0);
    assert!(hot_spots_1000.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_result_summary_basic() {
    let result = SensitivityResult {
        layers: vec![LayerSensitivity {
            name: "linear_1".to_string(),
            layer_type: "Linear".to_string(),
            input_width: 0.02,
            output_width: 0.1,
            sensitivity: 5.0,
            mean_output_width: 0.08,
            output_shape: vec![10],
            has_overflow: false,
            propagation_failed: false,
        }],
        total_sensitivity: 5.0,
        max_sensitivity: 5.0,
        max_sensitivity_layer: Some(0),
        input_epsilon: 0.01,
        final_width: 0.1,
        overflow_at_layer: None,
    };

    let summary = result.summary();
    assert!(summary.contains("Sensitivity Analysis"));
    assert!(summary.contains("linear_1"));
    assert!(summary.contains("5.00"));
    assert!(summary.contains("MODERATE"));
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_result_summary_with_overflow() {
    let result = SensitivityResult {
        layers: vec![
            LayerSensitivity {
                name: "layer_1".to_string(),
                layer_type: "Linear".to_string(),
                input_width: 0.02,
                output_width: 0.1,
                sensitivity: 5.0,
                mean_output_width: 0.08,
                output_shape: vec![10],
                has_overflow: false,
                propagation_failed: false,
            },
            LayerSensitivity {
                name: "layer_2".to_string(),
                layer_type: "Softmax".to_string(),
                input_width: 0.1,
                output_width: f32::INFINITY,
                sensitivity: f32::INFINITY,
                mean_output_width: f32::INFINITY,
                output_shape: vec![10],
                has_overflow: true,
                propagation_failed: false,
            },
        ],
        total_sensitivity: f32::INFINITY,
        max_sensitivity: 5.0,
        max_sensitivity_layer: Some(0),
        input_epsilon: 0.01,
        final_width: f32::INFINITY,
        overflow_at_layer: Some(1),
    };

    let summary = result.summary();
    assert!(summary.contains("OVERFLOW"));
    assert!(summary.contains("WARNING:"));
    assert!(summary.contains("layer_2"));
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_result_summary_with_propagation_failure() {
    let result = SensitivityResult {
        layers: vec![LayerSensitivity {
            name: "broken_layer".to_string(),
            layer_type: "Unknown".to_string(),
            input_width: 0.02,
            output_width: 0.02,
            sensitivity: 1.0,
            mean_output_width: 0.02,
            output_shape: vec![10],
            has_overflow: false,
            propagation_failed: true,
        }],
        total_sensitivity: 1.0,
        max_sensitivity: 0.0,
        max_sensitivity_layer: None,
        input_epsilon: 0.01,
        final_width: 0.02,
        overflow_at_layer: None,
    };

    let summary = result.summary();
    assert!(summary.contains("SKIPPED"));
    assert!(summary.contains("propagation failure"));
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_result_summary_status_thresholds() {
    // Test that different sensitivity values get correct status
    let make_layer = |name: &str, sens: f32| LayerSensitivity {
        name: name.to_string(),
        layer_type: "Linear".to_string(),
        input_width: 1.0,
        output_width: sens,
        sensitivity: sens,
        mean_output_width: sens * 0.8,
        output_shape: vec![10],
        has_overflow: false,
        propagation_failed: false,
    };

    let result = SensitivityResult {
        layers: vec![
            make_layer("stable", 0.5),   // sensitivity < 1.0 -> STABLE
            make_layer("ok", 1.5),       // 1.0 <= sensitivity <= 2.0 -> OK
            make_layer("moderate", 5.0), // 2.0 < sensitivity <= 10.0 -> MODERATE
            make_layer("high", 15.0),    // sensitivity > 10.0 -> HIGH
        ],
        total_sensitivity: 56.25,
        max_sensitivity: 15.0,
        max_sensitivity_layer: Some(3),
        input_epsilon: 0.01,
        final_width: 15.0,
        overflow_at_layer: None,
    };

    let summary = result.summary();
    assert!(summary.contains("STABLE"));
    assert!(summary.contains("OK"));
    assert!(summary.contains("MODERATE"));
    assert!(summary.contains("HIGH"));
}

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name_various_lengths() {
    // Exact fit
    assert_eq!(truncate_name("exactly_10", 10), "exactly_10");
    // Under limit
    assert_eq!(truncate_name("short", 10), "short");
    // Over limit: "this_is_way_too_long_name" (25 chars) -> keep last 7 chars = "ng_name"
    assert_eq!(truncate_name("this_is_way_too_long_name", 10), "...ng_name");
    // Exactly one over: "abcdefghijk" (11 chars) -> keep last 7 = "efghijk"
    assert_eq!(truncate_name("abcdefghijk", 10), "...efghijk");
    // Very short width: "longname" (8 chars), width 5 -> keep last 2 = "me"
    assert_eq!(truncate_name("longname", 5), "...me");
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_error_display() {
    let load_err = SensitivityError::load(
        "sensitivity",
        ny_core::NyError::ModelLoad("file not found".into()),
    );
    assert!(load_err.to_string().contains("load failed"));

    let prop_err = SensitivityError::propagation_msg("sensitivity", "shape mismatch");
    assert!(prop_err.to_string().contains("propagation failed"));

    let no_layers = SensitivityError::no_layers("sensitivity");
    assert!(no_layers.to_string().contains("no layers"));

    let invalid_shape = SensitivityError::invalid_input_shape("sensitivity", "bad shape");
    assert!(invalid_shape.to_string().contains("bad shape"));
}

#[ntest::timeout(10000)]
#[test]
fn test_sensitivity_error_round_trips_ny_error_variant() {
    let err = SensitivityError::propagation(
        "sensitivity",
        ny_core::NyError::ShapeMismatch {
            expected: vec![1, 2],
            got: vec![1, 3],
        },
    );
    let ny_error: ny_core::NyError = err.into();

    match ny_error {
        ny_core::NyError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![1, 2]);
            assert_eq!(got, vec![1, 3]);
        }
        other => {
            panic!("expected ShapeMismatch after SensitivityError round-trip, got: {other:?}")
        }
    }
}
