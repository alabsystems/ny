// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ===== truncate_name tests =====

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name_shorter_than_width() {
    assert_eq!(truncate_name("short", 10), "short");
}

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name_exact_width() {
    assert_eq!(truncate_name("exactly_10", 10), "exactly_10");
}

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name_longer_than_width() {
    assert_eq!(truncate_name("very_long_layer_name", 10), "...er_name");
}

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name_empty_string() {
    assert_eq!(truncate_name("", 10), "");
}

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name_width_equals_length() {
    assert_eq!(truncate_name("abcde", 5), "abcde");
}

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name_single_char_over() {
    // "abcdef" length 6, width 5 -> "...ef"
    assert_eq!(truncate_name("abcdef", 5), "...ef");
}

#[ntest::timeout(10000)]
#[test]
fn test_truncate_name_minimum_width() {
    // With width=4, we get "..." + 1 char
    assert_eq!(truncate_name("abcdefgh", 4), "...h");
}

// ===== assess_float16 tests =====

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_safe_small_values() {
    assert!(matches!(assess_float16(-100.0, 100.0), QuantSafety::Safe));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_safe_at_boundary() {
    assert!(matches!(
        assess_float16(-65504.0, 65504.0),
        QuantSafety::Safe
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_overflow() {
    assert!(matches!(
        assess_float16(-70000.0, 70000.0),
        QuantSafety::Overflow
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_denormal() {
    assert!(matches!(assess_float16(-1e-6, 1e-6), QuantSafety::Denormal));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_unknown_infinity() {
    assert!(matches!(
        assess_float16(f32::NEG_INFINITY, f32::INFINITY),
        QuantSafety::Unknown
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_unknown_nan() {
    assert!(matches!(
        assess_float16(f32::NAN, 1.0),
        QuantSafety::Unknown
    ));
    assert!(matches!(
        assess_float16(1.0, f32::NAN),
        QuantSafety::Unknown
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_unknown_neg_infinity() {
    assert!(matches!(
        assess_float16(f32::NEG_INFINITY, 0.0),
        QuantSafety::Unknown
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_zero() {
    assert!(matches!(assess_float16(0.0, 0.0), QuantSafety::Safe));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_just_above_max() {
    // Exactly at the boundary + epsilon
    assert!(matches!(
        assess_float16(-65505.0, 65505.0),
        QuantSafety::Overflow
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_asymmetric_bounds() {
    // Only one side exceeds
    assert!(matches!(
        assess_float16(-100.0, 70000.0),
        QuantSafety::Overflow
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_negative_only_overflow() {
    assert!(matches!(
        assess_float16(-70000.0, 0.0),
        QuantSafety::Overflow
    ));
}

#[ntest::timeout(10000)]
#[test]
fn test_float16_assessment_small_positive_denormal() {
    // Very small positive value in denormal range
    assert!(matches!(assess_float16(0.0, 1e-6), QuantSafety::Denormal));
}

// ===== assess_int8 tests =====

#[ntest::timeout(10000)]
#[test]
fn test_int8_assessment_safe() {
    let (safety, scale) = assess_int8(-100.0, 100.0);
    assert!(matches!(safety, QuantSafety::Safe));
    assert!(scale.is_some());
    // Scale should be 127.0 / 100.0 = 1.27
    assert!((scale.unwrap() - 1.27).abs() < 0.01);
}

#[ntest::timeout(10000)]
#[test]
fn test_int8_assessment_zero() {
    let (safety, scale) = assess_int8(0.0, 0.0);
    assert!(matches!(safety, QuantSafety::Safe));
    assert_eq!(scale, Some(1.0));
}

#[ntest::timeout(10000)]
#[test]
fn test_int8_assessment_scaling_required() {
    let (safety, scale) = assess_int8(-1000.0, 1000.0);
    assert!(matches!(safety, QuantSafety::ScalingRequired));
    assert!(scale.is_some());
    // Scale should be 127.0 / 1000.0 = 0.127
    assert!((scale.unwrap() - 0.127).abs() < 0.001);
}

#[ntest::timeout(10000)]
#[test]
fn test_int8_assessment_unknown_infinity() {
    let (safety, scale) = assess_int8(f32::NEG_INFINITY, f32::INFINITY);
    assert!(matches!(safety, QuantSafety::Unknown));
    assert!(scale.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_int8_assessment_unknown_nan() {
    let (safety, scale) = assess_int8(f32::NAN, 0.0);
    assert!(matches!(safety, QuantSafety::Unknown));
    assert!(scale.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_int8_assessment_very_small_values() {
    // Very small values that need large scale (precision issues)
    let (safety, scale) = assess_int8(-1e-8, 1e-8);
    assert!(matches!(safety, QuantSafety::ScalingRequired));
    assert!(scale.is_some());
    // Scale should be 127.0 / 1e-8 = 1.27e10 (very large, hence ScalingRequired)
    let s = scale.expect("invariant: scale is Some when ScalingRequired");
    assert!(
        (s - 127.0 / 1e-8).abs() / s < 0.01,
        "Expected scale ~1.27e10, got {}",
        s
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_int8_assessment_asymmetric_bounds() {
    // Asymmetric bounds - max_abs should use the larger absolute value
    let (safety, scale) = assess_int8(-50.0, 100.0);
    assert!(matches!(safety, QuantSafety::Safe));
    // Scale based on max_abs = 100
    assert!((scale.unwrap() - 1.27).abs() < 0.01);
}

#[ntest::timeout(10000)]
#[test]
fn test_int8_assessment_negative_only() {
    let (safety, scale) = assess_int8(-100.0, -50.0);
    assert!(matches!(safety, QuantSafety::Safe));
    // Scale based on max_abs = 100
    assert!((scale.unwrap() - 1.27).abs() < 0.01);
}

#[ntest::timeout(10000)]
#[test]
fn test_int8_assessment_overflow_very_large() {
    // If scale would be too small (< 1e-10), it's overflow
    let (safety, scale) = assess_int8(-1e15, 1e15);
    assert!(matches!(safety, QuantSafety::Overflow));
    assert!(scale.is_none());
}

// ===== QuantFormat Display tests =====

#[ntest::timeout(10000)]
#[test]
fn test_quant_format_display_float16() {
    assert_eq!(format!("{}", QuantFormat::Float16), "float16");
}

#[ntest::timeout(10000)]
#[test]
fn test_quant_format_display_int8() {
    assert_eq!(format!("{}", QuantFormat::Int8), "int8");
}

// ===== QuantSafety Display tests =====

#[ntest::timeout(10000)]
#[test]
fn test_quant_safety_display_safe() {
    assert_eq!(format!("{}", QuantSafety::Safe), "SAFE");
}

#[ntest::timeout(10000)]
#[test]
fn test_quant_safety_display_denormal() {
    assert_eq!(format!("{}", QuantSafety::Denormal), "DENORMAL");
}

#[ntest::timeout(10000)]
#[test]
fn test_quant_safety_display_scaling_required() {
    assert_eq!(format!("{}", QuantSafety::ScalingRequired), "SCALE");
}

#[ntest::timeout(10000)]
#[test]
fn test_quant_safety_display_overflow() {
    assert_eq!(format!("{}", QuantSafety::Overflow), "OVERFLOW");
}

#[ntest::timeout(10000)]
#[test]
fn test_quant_safety_display_unknown() {
    assert_eq!(format!("{}", QuantSafety::Unknown), "UNKNOWN");
}

// ===== LayerQuantization::is_safe_for tests =====

#[ntest::timeout(10000)]
#[test]
fn test_layer_quantization_is_safe_for_float16_safe() {
    let layer = LayerQuantization {
        name: "test".to_string(),
        layer_type: "Linear".to_string(),
        min_bound: -100.0,
        max_bound: 100.0,
        max_abs: 100.0,
        output_shape: vec![1, 10],
        float16_safety: QuantSafety::Safe,
        int8_safety: QuantSafety::Safe,
        int8_scale: Some(1.27),
        has_overflow: false,
        propagation_failed: false,
    };
    assert!(layer.is_safe_for(QuantFormat::Float16));
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_quantization_is_safe_for_float16_overflow() {
    let layer = LayerQuantization {
        name: "test".to_string(),
        layer_type: "Linear".to_string(),
        min_bound: -70000.0,
        max_bound: 70000.0,
        max_abs: 70000.0,
        output_shape: vec![1, 10],
        float16_safety: QuantSafety::Overflow,
        int8_safety: QuantSafety::Overflow,
        int8_scale: None,
        has_overflow: false,
        propagation_failed: false,
    };
    assert!(!layer.is_safe_for(QuantFormat::Float16));
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_quantization_is_safe_for_float16_denormal() {
    let layer = LayerQuantization {
        name: "test".to_string(),
        layer_type: "Linear".to_string(),
        min_bound: -1e-6,
        max_bound: 1e-6,
        max_abs: 1e-6,
        output_shape: vec![1, 10],
        float16_safety: QuantSafety::Denormal,
        int8_safety: QuantSafety::Safe,
        int8_scale: Some(1.0),
        has_overflow: false,
        propagation_failed: false,
    };
    // Denormal is not considered "safe" for float16
    assert!(!layer.is_safe_for(QuantFormat::Float16));
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_quantization_is_safe_for_int8_safe() {
    let layer = LayerQuantization {
        name: "test".to_string(),
        layer_type: "Linear".to_string(),
        min_bound: -100.0,
        max_bound: 100.0,
        max_abs: 100.0,
        output_shape: vec![1, 10],
        float16_safety: QuantSafety::Safe,
        int8_safety: QuantSafety::Safe,
        int8_scale: Some(1.27),
        has_overflow: false,
        propagation_failed: false,
    };
    assert!(layer.is_safe_for(QuantFormat::Int8));
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_quantization_is_safe_for_int8_scaling_required() {
    let layer = LayerQuantization {
        name: "test".to_string(),
        layer_type: "Linear".to_string(),
        min_bound: -1000.0,
        max_bound: 1000.0,
        max_abs: 1000.0,
        output_shape: vec![1, 10],
        float16_safety: QuantSafety::Safe,
        int8_safety: QuantSafety::ScalingRequired,
        int8_scale: Some(0.127),
        has_overflow: false,
        propagation_failed: false,
    };
    // ScalingRequired is considered safe for int8
    assert!(layer.is_safe_for(QuantFormat::Int8));
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_quantization_is_safe_for_int8_overflow() {
    let layer = LayerQuantization {
        name: "test".to_string(),
        layer_type: "Linear".to_string(),
        min_bound: -1e15,
        max_bound: 1e15,
        max_abs: 1e15,
        output_shape: vec![1, 10],
        float16_safety: QuantSafety::Overflow,
        int8_safety: QuantSafety::Overflow,
        int8_scale: None,
        has_overflow: false,
        propagation_failed: false,
    };
    assert!(!layer.is_safe_for(QuantFormat::Int8));
}

// ===== propagation_failed safety gate tests =====

#[ntest::timeout(10000)]
#[test]
fn test_layer_quantization_propagation_failed_not_safe_for_float16() {
    let layer = LayerQuantization {
        name: "test".to_string(),
        layer_type: "Linear".to_string(),
        min_bound: -100.0,
        max_bound: 100.0,
        max_abs: 100.0,
        output_shape: vec![1, 10],
        float16_safety: QuantSafety::Safe, // Would be safe if propagation succeeded
        int8_safety: QuantSafety::Safe,
        int8_scale: Some(1.27),
        has_overflow: false,
        propagation_failed: true, // But propagation failed — assessment is unreliable
    };
    assert!(
        !layer.is_safe_for(QuantFormat::Float16),
        "propagation_failed should make is_safe_for return false"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_quantization_propagation_failed_not_safe_for_int8() {
    let layer = LayerQuantization {
        name: "test".to_string(),
        layer_type: "Linear".to_string(),
        min_bound: -100.0,
        max_bound: 100.0,
        max_abs: 100.0,
        output_shape: vec![1, 10],
        float16_safety: QuantSafety::Safe,
        int8_safety: QuantSafety::Safe, // Would be safe if propagation succeeded
        int8_scale: Some(1.27),
        has_overflow: false,
        propagation_failed: true, // But propagation failed — assessment is unreliable
    };
    assert!(
        !layer.is_safe_for(QuantFormat::Int8),
        "propagation_failed should make is_safe_for return false"
    );
}

// ===== QuantizeConfig tests =====

#[ntest::timeout(10000)]
#[test]
fn test_quantize_config_default() {
    let config = QuantizeConfig::default();
    assert!((config.epsilon - 0.01).abs() < 1e-6);
    assert!(config.continue_after_overflow);
    assert!(config.input.is_none());
}

// ===== QuantizeResult tests =====

fn make_layer_quantization(
    name: &str,
    layer_type: &str,
    min_bound: f32,
    max_bound: f32,
    float16_safety: QuantSafety,
    int8_safety: QuantSafety,
    int8_scale: Option<f32>,
) -> LayerQuantization {
    LayerQuantization {
        name: name.to_string(),
        layer_type: layer_type.to_string(),
        min_bound,
        max_bound,
        max_abs: min_bound.abs().max(max_bound.abs()),
        output_shape: vec![1, 10],
        float16_safety,
        int8_safety,
        int8_scale,
        has_overflow: false,
        propagation_failed: false,
    }
}

fn create_quantize_result() -> QuantizeResult {
    QuantizeResult {
        layers: vec![
            make_layer_quantization(
                "layer1",
                "Linear",
                -100.0,
                100.0,
                QuantSafety::Safe,
                QuantSafety::Safe,
                Some(1.27),
            ),
            make_layer_quantization(
                "layer2",
                "ReLU",
                -70000.0,
                70000.0,
                QuantSafety::Overflow,
                QuantSafety::ScalingRequired,
                Some(0.00181),
            ),
            make_layer_quantization(
                "layer3",
                "Softmax",
                -1e-6,
                1e-6,
                QuantSafety::Denormal,
                QuantSafety::Safe,
                Some(1.0),
            ),
        ],
        float16_safe: false,
        int8_safe: true,
        float16_overflow_count: 1,
        int8_overflow_count: 0,
        denormal_count: 1,
        input_epsilon: 0.01,
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_result_summary_contains_header() {
    let result = create_quantize_result();
    let summary = result.summary();
    assert!(summary.contains("Quantization Safety Analysis"));
    assert!(summary.contains("Layer"));
    assert!(summary.contains("Min"));
    assert!(summary.contains("Max"));
    assert!(summary.contains("F16"));
    assert!(summary.contains("I8"));
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_result_summary_contains_layers() {
    let result = create_quantize_result();
    let summary = result.summary();
    assert!(summary.contains("layer1"));
    assert!(summary.contains("layer2"));
    assert!(summary.contains("layer3"));
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_result_summary_contains_safety_status() {
    let result = create_quantize_result();
    let summary = result.summary();
    assert!(summary.contains("UNSAFE")); // float16
    assert!(summary.contains("SAFE")); // int8
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_result_float16_unsafe_layers() {
    let result = create_quantize_result();
    let unsafe_layers = result.float16_unsafe_layers();
    assert_eq!(unsafe_layers.len(), 1);
    assert_eq!(unsafe_layers[0].name, "layer2");
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_result_int8_unsafe_layers() {
    let result = create_quantize_result();
    let unsafe_layers = result.int8_unsafe_layers();
    assert!(unsafe_layers.is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_result_denormal_layers() {
    let result = create_quantize_result();
    let denormal_layers = result.denormal_layers();
    assert_eq!(denormal_layers.len(), 1);
    assert_eq!(denormal_layers[0].name, "layer3");
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_result_empty_layers() {
    let result = QuantizeResult {
        layers: vec![],
        float16_safe: true,
        int8_safe: true,
        float16_overflow_count: 0,
        int8_overflow_count: 0,
        denormal_count: 0,
        input_epsilon: 0.01,
    };
    assert!(result.float16_unsafe_layers().is_empty());
    assert!(result.int8_unsafe_layers().is_empty());
    assert!(result.denormal_layers().is_empty());
}

#[allow(deprecated)]
#[ntest::timeout(10000)]
#[test]
fn test_quantization_result_alias_compatibility() {
    let canonical = create_quantize_result();
    // The alias is a plain `type` alias; assigning through it proves compatibility.
    let result: QuantizationResult = canonical;
    assert_eq!(result.float16_unsafe_layers().len(), 1);
    assert!(result.summary().contains("Quantization Safety Analysis"));
}

// ===== QuantizeError tests =====

#[ntest::timeout(10000)]
#[test]
fn test_quantize_error_display_load_error() {
    let err = QuantizeError::load("quantize", ny_core::NyError::ModelLoad("test error".into()));
    let msg = format!("{}", err);
    assert!(
        msg.contains("quantize") && msg.contains("load failed"),
        "Expected context+load message, got: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_error_round_trips_ny_error_variant() {
    let err = QuantizeError::load(
        "quantize",
        ny_core::NyError::ModelLoad("typed load failure".into()),
    );
    let ny_error: ny_core::NyError = err.into();

    match ny_error {
        ny_core::NyError::ModelLoad(msg) => {
            assert_eq!(msg, "typed load failure");
        }
        other => panic!("expected ModelLoad after QuantizeError round-trip, got: {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_error_display_propagation_error() {
    let err = QuantizeError::propagation_msg("quantize", "prop error");
    let msg = format!("{}", err);
    assert!(
        msg.contains("quantize") && msg.contains("propagation failed"),
        "Expected context+propagation message, got: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_error_display_no_layers() {
    let err = QuantizeError::no_layers("quantize");
    let msg = format!("{}", err);
    assert!(
        msg.contains("no layers"),
        "Expected no-layers message, got: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_quantize_error_display_invalid_input_shape() {
    let err = QuantizeError::invalid_input_shape("quantize", "bad shape");
    let msg = format!("{}", err);
    assert!(
        msg.contains("invalid input shape") && msg.contains("bad shape"),
        "Expected input shape message, got: {msg}"
    );
}
