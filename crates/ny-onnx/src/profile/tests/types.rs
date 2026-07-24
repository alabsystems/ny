// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_bound_status() {
    let epsilon = 0.01;
    // Initial width is 2 * epsilon = 0.02

    // Tight: width < 10 * initial = 0.2
    assert!(matches!(
        BoundStatus::from_width(0.1, epsilon),
        BoundStatus::Tight
    ));

    // Moderate: width < 100 * initial = 2.0
    assert!(matches!(
        BoundStatus::from_width(1.0, epsilon),
        BoundStatus::Moderate
    ));

    // Wide: width < 10000 * initial = 200
    assert!(matches!(
        BoundStatus::from_width(50.0, epsilon),
        BoundStatus::Wide
    ));

    // Very wide
    assert!(matches!(
        BoundStatus::from_width(1000.0, epsilon),
        BoundStatus::VeryWide
    ));

    // Overflow
    assert!(matches!(
        BoundStatus::from_width(f32::INFINITY, epsilon),
        BoundStatus::Overflow
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_profile_config_default() {
    let config = ProfileConfig::default();
    assert_eq!(config.epsilon, 0.01);
    assert!(config.continue_after_overflow);
    assert!(config.input.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_bound_status_display() {
    assert_eq!(format!("{}", BoundStatus::Tight), "TIGHT");
    assert_eq!(format!("{}", BoundStatus::Moderate), "MODERATE");
    assert_eq!(format!("{}", BoundStatus::Wide), "WIDE");
    assert_eq!(format!("{}", BoundStatus::VeryWide), "VERY WIDE");
    assert_eq!(format!("{}", BoundStatus::Overflow), "OVERFLOW");
}

#[ntest::timeout(10000)]
#[test]
fn test_bound_status_from_width_boundary_values() {
    let epsilon = 0.01;
    // Initial width is 2 * epsilon = 0.02

    // Tight/Moderate boundary: ratio = 10 -> width = 10 * 0.02 = 0.2
    assert!(matches!(
        BoundStatus::from_width(0.19, epsilon),
        BoundStatus::Tight
    ));
    assert!(matches!(
        BoundStatus::from_width(0.21, epsilon),
        BoundStatus::Moderate
    ));

    // Moderate/Wide boundary: ratio = 100 -> width = 100 * 0.02 = 2.0
    assert!(matches!(
        BoundStatus::from_width(1.99, epsilon),
        BoundStatus::Moderate
    ));
    assert!(matches!(
        BoundStatus::from_width(2.01, epsilon),
        BoundStatus::Wide
    ));

    // Wide/VeryWide boundary: ratio = 10000 -> width = 10000 * 0.02 = 200
    assert!(matches!(
        BoundStatus::from_width(199.0, epsilon),
        BoundStatus::Wide
    ));
    assert!(matches!(
        BoundStatus::from_width(201.0, epsilon),
        BoundStatus::VeryWide
    ));

    // NaN should also be overflow
    assert!(matches!(
        BoundStatus::from_width(f32::NAN, epsilon),
        BoundStatus::Overflow
    ));

    // Negative infinity
    assert!(matches!(
        BoundStatus::from_width(f32::NEG_INFINITY, epsilon),
        BoundStatus::Overflow
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_profile_is_choke_point() {
    let layer = LayerProfile {
        name: "test".to_string(),
        layer_type: "Linear".to_string(),
        input_width: 0.1,
        output_width: 1.0,
        mean_output_width: 0.8,
        median_output_width: 0.7,
        growth_ratio: 10.0,
        cumulative_expansion: 50.0,
        output_shape: vec![10],
        num_elements: 10,
        status: BoundStatus::Moderate,
    };

    assert!(layer.is_choke_point(5.0));
    assert!(!layer.is_choke_point(15.0));
    assert!(!layer.is_choke_point(10.0)); // exact threshold
}

#[ntest::timeout(10000)]
#[test]
fn test_profile_result_layers_by_growth() {
    let result = ProfileResult {
        layers: vec![
            LayerProfile {
                name: "low".to_string(),
                layer_type: "ReLU".to_string(),
                input_width: 1.0,
                output_width: 0.5,
                mean_output_width: 0.4,
                median_output_width: 0.4,
                growth_ratio: 0.5,
                cumulative_expansion: 0.5,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Tight,
            },
            LayerProfile {
                name: "high".to_string(),
                layer_type: "Linear".to_string(),
                input_width: 0.5,
                output_width: 5.0,
                mean_output_width: 4.0,
                median_output_width: 4.0,
                growth_ratio: 10.0,
                cumulative_expansion: 5.0,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Moderate,
            },
            LayerProfile {
                name: "medium".to_string(),
                layer_type: "Softmax".to_string(),
                input_width: 5.0,
                output_width: 15.0,
                mean_output_width: 12.0,
                median_output_width: 12.0,
                growth_ratio: 3.0,
                cumulative_expansion: 15.0,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Wide,
            },
        ],
        input_epsilon: 0.01,
        initial_width: 1.0,
        final_width: 15.0,
        total_expansion: 15.0,
        max_growth_layer: Some(1),
        max_growth_ratio: 10.0,
        overflow_at_layer: None,
        difficulty_score: 30.0,
    };

    let sorted = result.layers_by_growth();
    assert_eq!(sorted.len(), 3);
    assert_eq!(sorted[0].name, "high"); // growth=10
    assert_eq!(sorted[1].name, "medium"); // growth=3
    assert_eq!(sorted[2].name, "low"); // growth=0.5
}

#[ntest::timeout(10000)]
#[test]
fn test_profile_result_choke_points() {
    let result = ProfileResult {
        layers: vec![
            LayerProfile {
                name: "low".to_string(),
                layer_type: "ReLU".to_string(),
                input_width: 1.0,
                output_width: 0.5,
                mean_output_width: 0.4,
                median_output_width: 0.4,
                growth_ratio: 0.5,
                cumulative_expansion: 0.5,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Tight,
            },
            LayerProfile {
                name: "high".to_string(),
                layer_type: "Linear".to_string(),
                input_width: 0.5,
                output_width: 50.0,
                mean_output_width: 40.0,
                median_output_width: 40.0,
                growth_ratio: 100.0,
                cumulative_expansion: 50.0,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Wide,
            },
        ],
        input_epsilon: 0.01,
        initial_width: 1.0,
        final_width: 50.0,
        total_expansion: 50.0,
        max_growth_layer: Some(1),
        max_growth_ratio: 100.0,
        overflow_at_layer: None,
        difficulty_score: 50.0,
    };

    let chokes = result.choke_points(10.0);
    assert_eq!(chokes.len(), 1);
    assert_eq!(chokes[0].name, "high");

    let no_chokes = result.choke_points(1000.0);
    assert!(no_chokes.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_profile_result_problematic_layers() {
    let result = ProfileResult {
        layers: vec![
            LayerProfile {
                name: "tight".to_string(),
                layer_type: "ReLU".to_string(),
                input_width: 0.01,
                output_width: 0.01,
                mean_output_width: 0.01,
                median_output_width: 0.01,
                growth_ratio: 1.0,
                cumulative_expansion: 1.0,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Tight,
            },
            LayerProfile {
                name: "moderate".to_string(),
                layer_type: "Linear".to_string(),
                input_width: 0.01,
                output_width: 0.5,
                mean_output_width: 0.4,
                median_output_width: 0.4,
                growth_ratio: 50.0,
                cumulative_expansion: 50.0,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Moderate,
            },
            LayerProfile {
                name: "wide".to_string(),
                layer_type: "Softmax".to_string(),
                input_width: 0.5,
                output_width: 100.0,
                mean_output_width: 80.0,
                median_output_width: 80.0,
                growth_ratio: 200.0,
                cumulative_expansion: 10000.0,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Wide,
            },
            LayerProfile {
                name: "overflow".to_string(),
                layer_type: "Exp".to_string(),
                input_width: 100.0,
                output_width: f32::INFINITY,
                mean_output_width: f32::INFINITY,
                median_output_width: f32::INFINITY,
                growth_ratio: f32::INFINITY,
                cumulative_expansion: f32::INFINITY,
                output_shape: vec![10],
                num_elements: 10,
                status: BoundStatus::Overflow,
            },
        ],
        input_epsilon: 0.01,
        initial_width: 0.02,
        final_width: f32::INFINITY,
        total_expansion: f32::INFINITY,
        max_growth_layer: Some(2),
        max_growth_ratio: 200.0,
        overflow_at_layer: Some(3),
        difficulty_score: 100.0,
    };

    let problems = result.problematic_layers();
    assert_eq!(problems.len(), 2); // Wide and Overflow
    assert!(problems.iter().any(|l| l.name == "wide"));
    assert!(problems.iter().any(|l| l.name == "overflow"));
}

#[ntest::timeout(10000)]
#[test]
fn test_profile_result_summary_basic() {
    let result = ProfileResult {
        layers: vec![LayerProfile {
            name: "linear_1".to_string(),
            layer_type: "Linear".to_string(),
            input_width: 0.02,
            output_width: 0.1,
            mean_output_width: 0.08,
            median_output_width: 0.07,
            growth_ratio: 5.0,
            cumulative_expansion: 5.0,
            output_shape: vec![10],
            num_elements: 10,
            status: BoundStatus::Tight,
        }],
        input_epsilon: 0.01,
        initial_width: 0.02,
        final_width: 0.1,
        total_expansion: 5.0,
        max_growth_layer: Some(0),
        max_growth_ratio: 5.0,
        overflow_at_layer: None,
        difficulty_score: 17.0,
    };

    let summary = result.summary();
    assert!(summary.contains("Bound Width Profile"));
    assert!(summary.contains("linear_1"));
    assert!(summary.contains("5.00x"));
    assert!(summary.contains("TIGHT"));
    assert!(summary.contains("Verification difficulty"));
}

#[ntest::timeout(10000)]
#[test]
fn test_profile_result_summary_with_overflow() {
    let result = ProfileResult {
        layers: vec![LayerProfile {
            name: "exploding_layer".to_string(),
            layer_type: "Exp".to_string(),
            input_width: 100.0,
            output_width: f32::INFINITY,
            mean_output_width: f32::INFINITY,
            median_output_width: f32::INFINITY,
            growth_ratio: f32::INFINITY,
            cumulative_expansion: f32::INFINITY,
            output_shape: vec![10],
            num_elements: 10,
            status: BoundStatus::Overflow,
        }],
        input_epsilon: 0.01,
        initial_width: 0.02,
        final_width: f32::INFINITY,
        total_expansion: f32::INFINITY,
        max_growth_layer: Some(0),
        max_growth_ratio: f32::INFINITY,
        overflow_at_layer: Some(0),
        difficulty_score: 100.0,
    };

    let summary = result.summary();
    assert!(summary.contains("WARNING"));
    assert!(summary.contains("Overflow"));
    assert!(summary.contains("exploding_layer"));
}
