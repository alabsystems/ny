// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::utils::truncate_name;
use super::*;
use ny_core::{HeuristicUsed as RustHeuristicUsed, SoundnessProvenance as RustSoundnessProvenance};

// =========================================================================
// Helper Function Tests
// =========================================================================

#[test]
fn test_truncate_name_short() {
    assert_eq!(truncate_name("short", 10), "short");
}

#[test]
fn test_truncate_name_exact() {
    assert_eq!(truncate_name("exactly10!", 10), "exactly10!");
}

#[test]
fn test_truncate_name_long() {
    let result = truncate_name("this_is_a_very_long_layer_name", 15);
    assert!(result.starts_with("..."));
    assert_eq!(result.len(), 15);
    assert!(result.ends_with("layer_name"));
}

// =========================================================================
// DiffStatus Tests
// =========================================================================

#[test]
fn test_diff_status_repr() {
    assert_eq!(DiffStatus::Ok.__repr__(), "DiffStatus.Ok");
    assert_eq!(DiffStatus::DriftStarts.__repr__(), "DiffStatus.DriftStarts");
    assert_eq!(
        DiffStatus::ExceedsTolerance.__repr__(),
        "DiffStatus.ExceedsTolerance"
    );
    assert_eq!(
        DiffStatus::ShapeMismatch.__repr__(),
        "DiffStatus.ShapeMismatch"
    );
}

// =========================================================================
// LayerComparison Tests
// =========================================================================

#[test]
fn test_layer_comparison_creation() {
    let lc = LayerComparison {
        name: "layer1".to_string(),
        name_b: Some("layer1_b".to_string()),
        max_diff: 0.001,
        mean_diff: 0.0005,
        exceeds_tolerance: false,
        shape_a: vec![1, 64, 128],
        shape_b: vec![1, 64, 128],
    };
    assert_eq!(lc.name, "layer1");
    assert_eq!(lc.name_b, Some("layer1_b".to_string()));
    assert!(!lc.exceeds_tolerance);
}

#[test]
fn test_layer_comparison_repr() {
    let lc = LayerComparison {
        name: "test_layer".to_string(),
        name_b: None,
        max_diff: 1e-4,
        mean_diff: 5e-5,
        exceeds_tolerance: true,
        shape_a: vec![1, 64],
        shape_b: vec![1, 64],
    };
    let repr = lc.__repr__();
    assert!(repr.contains("test_layer"));
    assert!(repr.contains("exceeds=true"));
}

// =========================================================================
// DiffResult Tests
// =========================================================================

fn make_diff_result(first_bad_layer: Option<usize>) -> DiffResult {
    let layers = vec![
        LayerComparison {
            name: "layer0".to_string(),
            name_b: None,
            max_diff: 1e-6,
            mean_diff: 5e-7,
            exceeds_tolerance: false,
            shape_a: vec![1, 64],
            shape_b: vec![1, 64],
        },
        LayerComparison {
            name: "layer1".to_string(),
            name_b: None,
            max_diff: 1e-3,
            mean_diff: 5e-4,
            exceeds_tolerance: first_bad_layer == Some(1),
            shape_a: vec![1, 64],
            shape_b: vec![1, 64],
        },
    ];
    DiffResult {
        layers,
        first_bad_layer,
        drift_start_layer: None,
        max_divergence: 1e-3,
        tolerance: 1e-4,
        suggestion: None,
    }
}

#[test]
fn test_diff_result_is_equivalent_true() {
    let result = make_diff_result(None);
    assert!(result.is_equivalent());
}

#[test]
fn test_diff_result_is_equivalent_false() {
    let result = make_diff_result(Some(1));
    assert!(!result.is_equivalent());
}

#[test]
fn test_diff_result_first_bad_layer_name() {
    let result = make_diff_result(Some(1));
    assert_eq!(result.first_bad_layer_name(), Some("layer1".to_string()));

    let result_none = make_diff_result(None);
    assert_eq!(result_none.first_bad_layer_name(), None);
}

#[test]
fn test_diff_result_statuses() {
    let mut result = make_diff_result(Some(1));
    result.layers[1].exceeds_tolerance = true;

    let statuses = result.statuses();
    assert_eq!(statuses.len(), 2);
    assert!(matches!(statuses[0], DiffStatus::Ok));
    assert!(matches!(statuses[1], DiffStatus::ExceedsTolerance));
}

#[test]
fn test_diff_result_statuses_with_drift() {
    let mut result = make_diff_result(None);
    result.drift_start_layer = Some(0);

    let statuses = result.statuses();
    assert!(matches!(statuses[0], DiffStatus::DriftStarts));
}

#[test]
fn test_diff_result_statuses_shape_mismatch() {
    let mut result = make_diff_result(None);
    result.layers[0].shape_b = vec![1, 32]; // Different shape

    let statuses = result.statuses();
    assert!(matches!(statuses[0], DiffStatus::ShapeMismatch));
}

#[test]
fn test_diff_result_repr() {
    let result = make_diff_result(None);
    let repr = result.__repr__();
    assert!(repr.contains("DiffResult"));
    assert!(repr.contains("layers=2"));
    assert!(repr.contains("is_equivalent=true"));
}

#[test]
fn test_diff_result_summary() {
    let result = make_diff_result(Some(1));
    let summary = result.summary();
    assert!(summary.contains("Layer-by-Layer Comparison"));
    assert!(summary.contains("layer0"));
    assert!(summary.contains("layer1"));
}

// =========================================================================
// LayerSensitivity Tests
// =========================================================================

fn make_layer_sensitivity(sensitivity: f32) -> LayerSensitivity {
    LayerSensitivity {
        name: "test_layer".to_string(),
        layer_type: "Linear".to_string(),
        input_width: 0.1,
        output_width: sensitivity * 0.1,
        sensitivity,
        mean_output_width: sensitivity * 0.1,
        output_shape: vec![1, 64],
        has_overflow: false,
        propagation_failed: false,
    }
}

#[test]
fn test_layer_sensitivity_is_high_sensitivity() {
    let layer = make_layer_sensitivity(15.0);
    assert!(layer.is_high_sensitivity(10.0));
    assert!(!layer.is_high_sensitivity(20.0));
}

#[test]
fn test_layer_sensitivity_is_contractive() {
    let contractive = make_layer_sensitivity(0.5);
    assert!(contractive.is_contractive());

    let expansive = make_layer_sensitivity(2.0);
    assert!(!expansive.is_contractive());
}

#[test]
fn test_layer_sensitivity_repr() {
    let layer = make_layer_sensitivity(5.0);
    let repr = layer.__repr__();
    assert!(repr.contains("test_layer"));
    assert!(repr.contains("5.00"));
}

// =========================================================================
// SensitivityResult Tests
// =========================================================================

fn make_sensitivity_result() -> SensitivityResult {
    SensitivityResult {
        layers: vec![
            make_layer_sensitivity(2.0),
            make_layer_sensitivity(15.0),
            make_layer_sensitivity(0.8),
        ],
        total_sensitivity: 24.0,
        max_sensitivity: 15.0,
        max_sensitivity_layer: Some(1),
        input_epsilon: 0.01,
        final_width: 0.24,
        overflow_at_layer: None,
    }
}

#[test]
fn test_sensitivity_result_hot_spots() {
    let result = make_sensitivity_result();
    let hot_spots = result.hot_spots(10.0);
    assert_eq!(hot_spots.len(), 1);
    assert_eq!(hot_spots[0].sensitivity, 15.0);
}

#[test]
fn test_sensitivity_result_has_overflow() {
    let result = make_sensitivity_result();
    assert!(!result.has_overflow());

    let mut overflow_result = make_sensitivity_result();
    overflow_result.overflow_at_layer = Some(1);
    assert!(overflow_result.has_overflow());
}

#[test]
fn test_sensitivity_result_max_layer_name() {
    let result = make_sensitivity_result();
    // max_sensitivity_layer is Some(1), but all layers have same name in test
    assert!(result.max_sensitivity_layer_name().is_some());
}

#[test]
fn test_sensitivity_result_repr() {
    let result = make_sensitivity_result();
    let repr = result.__repr__();
    assert!(repr.contains("SensitivityResult"));
    assert!(repr.contains("max_sensitivity=15.00"));
}

// =========================================================================
// SensitivityResult::summary() Tests
// =========================================================================

#[test]
fn test_sensitivity_result_summary_propagation_failed_shows_failed() {
    let result = SensitivityResult {
        layers: vec![
            LayerSensitivity {
                name: "good_layer".to_string(),
                layer_type: "Linear".to_string(),
                input_width: 0.1,
                output_width: 0.5,
                sensitivity: 5.0,
                mean_output_width: 0.4,
                output_shape: vec![1, 64],
                has_overflow: false,
                propagation_failed: false,
            },
            LayerSensitivity {
                name: "broken_layer".to_string(),
                layer_type: "Unknown".to_string(),
                input_width: 0.5,
                output_width: 0.5,
                sensitivity: 1.0,
                mean_output_width: 0.5,
                output_shape: vec![1, 64],
                has_overflow: false,
                propagation_failed: true,
            },
        ],
        total_sensitivity: 5.0,
        max_sensitivity: 5.0,
        max_sensitivity_layer: Some(0),
        input_epsilon: 0.01,
        final_width: 0.5,
        overflow_at_layer: None,
    };

    let summary = result.summary();
    // The propagation_failed layer must show "FAILED" status
    assert!(
        summary.contains("FAILED"),
        "Expected 'FAILED' in summary for propagation_failed layer, got:\n{}",
        summary
    );
    assert!(
        summary.contains("broken_layer"),
        "Expected broken_layer name in summary"
    );
    // The good layer should show "MODERATE" (sensitivity=5.0 > 2.0)
    assert!(
        summary.contains("MODERATE"),
        "Expected 'MODERATE' status for good_layer (sensitivity=5.0)"
    );
}

// =========================================================================
// QuantSafety Tests
// =========================================================================

#[test]
fn test_quant_safety_repr() {
    assert_eq!(QuantSafety::Safe.__repr__(), "QuantSafety.Safe");
    assert_eq!(QuantSafety::Denormal.__repr__(), "QuantSafety.Denormal");
    assert_eq!(
        QuantSafety::ScalingRequired.__repr__(),
        "QuantSafety.ScalingRequired"
    );
    assert_eq!(QuantSafety::Overflow.__repr__(), "QuantSafety.Overflow");
    assert_eq!(QuantSafety::Unknown.__repr__(), "QuantSafety.Unknown");
}

#[test]
fn test_quant_safety_str() {
    assert_eq!(QuantSafety::Safe.__str__(), "SAFE");
    assert_eq!(QuantSafety::Denormal.__str__(), "DENORMAL");
    assert_eq!(QuantSafety::ScalingRequired.__str__(), "SCALE");
    assert_eq!(QuantSafety::Overflow.__str__(), "OVERFLOW");
    assert_eq!(QuantSafety::Unknown.__str__(), "UNKNOWN");
}

// =========================================================================
// LayerQuantization Tests
// =========================================================================

fn make_layer_quantization(f16: QuantSafety, i8: QuantSafety) -> LayerQuantization {
    LayerQuantization {
        name: "quant_layer".to_string(),
        layer_type: "Linear".to_string(),
        min_bound: -10.0,
        max_bound: 10.0,
        max_abs: 10.0,
        output_shape: vec![1, 64],
        float16_safety: f16,
        int8_safety: i8,
        int8_scale: Some(0.1),
        has_overflow: false,
        propagation_failed: false,
    }
}

#[test]
fn test_layer_quantization_is_float16_safe() {
    let safe = make_layer_quantization(QuantSafety::Safe, QuantSafety::Safe);
    assert!(safe.is_float16_safe());

    let unsafe_layer = make_layer_quantization(QuantSafety::Overflow, QuantSafety::Safe);
    assert!(!unsafe_layer.is_float16_safe());
}

#[test]
fn test_layer_quantization_is_int8_safe() {
    let safe = make_layer_quantization(QuantSafety::Safe, QuantSafety::Safe);
    assert!(safe.is_int8_safe());

    let scaling = make_layer_quantization(QuantSafety::Safe, QuantSafety::ScalingRequired);
    assert!(scaling.is_int8_safe());

    let unsafe_layer = make_layer_quantization(QuantSafety::Safe, QuantSafety::Overflow);
    assert!(!unsafe_layer.is_int8_safe());
}

#[test]
fn test_layer_quantization_repr() {
    let layer = make_layer_quantization(QuantSafety::Safe, QuantSafety::ScalingRequired);
    let repr = layer.__repr__();
    assert!(repr.contains("quant_layer"));
    assert!(repr.contains("SAFE"));
    assert!(repr.contains("SCALE"));
}

// =========================================================================
// QuantizationResult Tests
// =========================================================================

fn make_quantization_result() -> QuantizationResult {
    QuantizationResult {
        layers: vec![
            make_layer_quantization(QuantSafety::Safe, QuantSafety::Safe),
            make_layer_quantization(QuantSafety::Overflow, QuantSafety::Safe),
            make_layer_quantization(QuantSafety::Safe, QuantSafety::Overflow),
        ],
        float16_safe: false,
        int8_safe: false,
        float16_overflow_count: 1,
        int8_overflow_count: 1,
        denormal_count: 0,
        input_epsilon: 0.01,
    }
}

#[test]
fn test_quantization_result_float16_unsafe_layers() {
    let result = make_quantization_result();
    let unsafe_layers = result.float16_unsafe_layers();
    assert_eq!(unsafe_layers.len(), 1);
}

#[test]
fn test_quantization_result_int8_unsafe_layers() {
    let result = make_quantization_result();
    let unsafe_layers = result.int8_unsafe_layers();
    assert_eq!(unsafe_layers.len(), 1);
}

#[test]
fn test_quantization_result_repr() {
    let result = make_quantization_result();
    let repr = result.__repr__();
    assert!(repr.contains("QuantizationResult"));
    assert!(repr.contains("layers=3"));
}

// =========================================================================
// BoundStatus Tests
// =========================================================================

#[test]
fn test_bound_status_repr() {
    assert_eq!(BoundStatus::Tight.__repr__(), "BoundStatus.Tight");
    assert_eq!(BoundStatus::Moderate.__repr__(), "BoundStatus.Moderate");
    assert_eq!(BoundStatus::Wide.__repr__(), "BoundStatus.Wide");
    assert_eq!(BoundStatus::VeryWide.__repr__(), "BoundStatus.VeryWide");
    assert_eq!(BoundStatus::Overflow.__repr__(), "BoundStatus.Overflow");
}

#[test]
fn test_bound_status_str() {
    assert_eq!(BoundStatus::Tight.__str__(), "TIGHT");
    assert_eq!(BoundStatus::Moderate.__str__(), "MODERATE");
    assert_eq!(BoundStatus::Wide.__str__(), "WIDE");
    assert_eq!(BoundStatus::VeryWide.__str__(), "VERY WIDE");
    assert_eq!(BoundStatus::Overflow.__str__(), "OVERFLOW");
}

// =========================================================================
// LayerProfile Tests
// =========================================================================

fn make_layer_profile(growth: f32, status: BoundStatus) -> LayerProfile {
    LayerProfile {
        name: "profile_layer".to_string(),
        layer_type: "Linear".to_string(),
        input_width: 0.1,
        output_width: growth * 0.1,
        mean_output_width: growth * 0.1,
        median_output_width: growth * 0.1,
        growth_ratio: growth,
        cumulative_expansion: growth,
        output_shape: vec![1, 64],
        num_elements: 64,
        status,
    }
}

#[test]
fn test_layer_profile_is_choke_point() {
    let layer = make_layer_profile(10.0, BoundStatus::Wide);
    assert!(layer.is_choke_point(5.0));
    assert!(!layer.is_choke_point(15.0));
}

#[test]
fn test_layer_profile_repr() {
    let layer = make_layer_profile(5.0, BoundStatus::Moderate);
    let repr = layer.__repr__();
    assert!(repr.contains("profile_layer"));
    assert!(repr.contains("growth=5.00x"));
    assert!(repr.contains("MODERATE"));
}

// =========================================================================
// ProfileResult Tests
// =========================================================================

fn make_profile_result() -> ProfileResult {
    ProfileResult {
        layers: vec![
            make_layer_profile(2.0, BoundStatus::Tight),
            make_layer_profile(10.0, BoundStatus::Wide),
            make_layer_profile(1.5, BoundStatus::Moderate),
        ],
        input_epsilon: 0.01,
        initial_width: 0.02,
        final_width: 0.6,
        total_expansion: 30.0,
        max_growth_layer: Some(1),
        max_growth_ratio: 10.0,
        overflow_at_layer: None,
        difficulty_score: 75.0,
    }
}

#[test]
fn test_profile_result_choke_points() {
    let result = make_profile_result();
    let choke_points = result.choke_points(5.0);
    assert_eq!(choke_points.len(), 1);
    assert_eq!(choke_points[0].growth_ratio, 10.0);
}

#[test]
fn test_profile_result_problematic_layers() {
    let result = make_profile_result();
    let problematic = result.problematic_layers();
    assert_eq!(problematic.len(), 1);
}

#[test]
fn test_profile_result_has_overflow() {
    let result = make_profile_result();
    assert!(!result.has_overflow());

    let mut overflow_result = make_profile_result();
    overflow_result.overflow_at_layer = Some(2);
    assert!(overflow_result.has_overflow());
}

#[test]
fn test_profile_result_max_growth_layer_name() {
    let result = make_profile_result();
    assert!(result.max_growth_layer_name().is_some());
}

#[test]
fn test_profile_result_repr() {
    let result = make_profile_result();
    let repr = result.__repr__();
    assert!(repr.contains("ProfileResult"));
    assert!(repr.contains("expansion=30.00x"));
    assert!(repr.contains("difficulty=75/100"));
}

// =========================================================================
// OutputBound Tests
// =========================================================================

#[test]
fn test_output_bound_width() {
    let bound = OutputBound {
        lower: -1.0,
        upper: 2.0,
    };
    assert!((bound.width() - 3.0).abs() < 1e-6);
}

#[test]
fn test_output_bound_midpoint() {
    let bound = OutputBound {
        lower: -1.0,
        upper: 3.0,
    };
    assert!((bound.midpoint() - 1.0).abs() < 1e-6);
}

#[test]
fn test_output_bound_repr() {
    let bound = OutputBound {
        lower: 0.5,
        upper: 1.5,
    };
    let repr = bound.__repr__();
    assert!(repr.contains("OutputBound"));
    assert!(repr.contains("0.5"));
    assert!(repr.contains("1.5"));
}

// =========================================================================
// VerifyStatus Tests
// =========================================================================

#[test]
fn test_verify_status_repr() {
    assert_eq!(VerifyStatus::Verified.__repr__(), "VerifyStatus.Verified");
    assert_eq!(VerifyStatus::Violated.__repr__(), "VerifyStatus.Violated");
    assert_eq!(VerifyStatus::Unknown.__repr__(), "VerifyStatus.Unknown");
    assert_eq!(VerifyStatus::Timeout.__repr__(), "VerifyStatus.Timeout");
}

#[test]
fn test_verify_status_str() {
    assert_eq!(VerifyStatus::Verified.__str__(), "VERIFIED");
    assert_eq!(VerifyStatus::Violated.__str__(), "VIOLATED");
    assert_eq!(VerifyStatus::Unknown.__str__(), "UNKNOWN");
    assert_eq!(VerifyStatus::Timeout.__str__(), "TIMEOUT");
}

// =========================================================================
// VerifyResult Tests
// =========================================================================

fn make_verify_result(status: VerifyStatus) -> VerifyResult {
    let output_bounds = Some(vec![
        OutputBound {
            lower: 0.0,
            upper: 1.0,
        },
        OutputBound {
            lower: -0.5,
            upper: 0.5,
        },
    ]);

    let soundness = SoundnessProvenance {
        mode: "sound".to_string(),
        heuristics_used: vec![],
    };

    VerifyResult {
        status,
        soundness,
        output_bounds,
        counterexample: None,
        counterexample_output: None,
        reason: None,
        method: "IBP".to_string(),
        actual_method: None,
        epsilon: 0.01,
    }
}

#[test]
fn test_soundness_provenance_from_rust() {
    let rust = RustSoundnessProvenance::from_heuristics(vec![
        RustHeuristicUsed::LogSoftmaxCrownSampling { num_nodes: 2 },
    ]);
    let py: SoundnessProvenance = rust.into();
    assert_eq!(py.mode, "heuristic");
    assert_eq!(py.heuristics_used.len(), 1);
    assert_eq!(py.heuristics_used[0].type_, "logsoftmax_crown_sampling");
    assert_eq!(py.heuristics_used[0].num_nodes, Some(2));
}

#[test]
fn test_soundness_provenance_normalization_forward_modes_from_rust() {
    let rust = RustSoundnessProvenance::from_heuristics(vec![
        RustHeuristicUsed::LayerNormForwardMode { num_nodes: 1 },
        RustHeuristicUsed::RmsNormForwardMode { num_nodes: 2 },
        RustHeuristicUsed::GroupNormForwardMode { num_nodes: 3 },
        RustHeuristicUsed::InstanceNormForwardMode { num_nodes: 4 },
        RustHeuristicUsed::AdaInForwardMode { num_nodes: 5 },
    ]);
    let py: SoundnessProvenance = rust.into();
    assert_eq!(py.mode, "heuristic");
    assert_eq!(py.heuristics_used.len(), 5);
    assert_eq!(py.heuristics_used[0].type_, "layernorm_forward_mode");
    assert_eq!(py.heuristics_used[0].num_nodes, Some(1));
    assert_eq!(py.heuristics_used[1].type_, "rmsnorm_forward_mode");
    assert_eq!(py.heuristics_used[1].num_nodes, Some(2));
    assert_eq!(py.heuristics_used[2].type_, "groupnorm_forward_mode");
    assert_eq!(py.heuristics_used[2].num_nodes, Some(3));
    assert_eq!(py.heuristics_used[3].type_, "instancenorm_forward_mode");
    assert_eq!(py.heuristics_used[3].num_nodes, Some(4));
    assert_eq!(py.heuristics_used[4].type_, "adain_forward_mode");
    assert_eq!(py.heuristics_used[4].num_nodes, Some(5));
}

#[test]
fn test_soundness_provenance_softmax_sampling_from_rust() {
    let rust = RustSoundnessProvenance::from_heuristics(vec![
        RustHeuristicUsed::SoftmaxCrownSampling { num_nodes: 1 },
        RustHeuristicUsed::CausalSoftmaxCrownSampling { num_nodes: 3 },
    ]);
    let py: SoundnessProvenance = rust.into();
    assert_eq!(py.mode, "heuristic");
    assert_eq!(py.heuristics_used.len(), 2);
    assert_eq!(py.heuristics_used[0].type_, "softmax_crown_sampling");
    assert_eq!(py.heuristics_used[0].num_nodes, Some(1));
    assert_eq!(py.heuristics_used[1].type_, "causal_softmax_crown_sampling");
    assert_eq!(py.heuristics_used[1].num_nodes, Some(3));
}

#[test]
fn test_soundness_provenance_reduce_extremum_fixed_index_from_rust() {
    let rust = RustSoundnessProvenance::from_heuristics(vec![
        RustHeuristicUsed::ReduceExtremumFixedIndex { num_nodes: 2 },
    ]);
    let py: SoundnessProvenance = rust.into();
    assert_eq!(py.mode, "heuristic");
    assert_eq!(py.heuristics_used.len(), 1);
    assert_eq!(py.heuristics_used[0].type_, "reduce_extremum_fixed_index");
    assert_eq!(py.heuristics_used[0].num_nodes, Some(2));
}

#[test]
fn test_verify_result_is_verified() {
    let verified = make_verify_result(VerifyStatus::Verified);
    assert!(verified.is_verified());
    assert!(!verified.is_violated());

    let unknown = make_verify_result(VerifyStatus::Unknown);
    assert!(!unknown.is_verified());
}

#[test]
fn test_verify_result_is_violated() {
    let violated = make_verify_result(VerifyStatus::Violated);
    assert!(violated.is_violated());
    assert!(!violated.is_verified());
}

#[test]
fn test_verify_result_max_output_width() {
    let result = make_verify_result(VerifyStatus::Verified);
    let max_width = result.max_output_width();
    assert!(max_width.is_some());
    assert!((max_width.unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn test_verify_result_max_output_width_none() {
    let mut result = make_verify_result(VerifyStatus::Verified);
    result.output_bounds = None;
    assert!(result.max_output_width().is_none());
}

#[test]
fn test_verify_result_repr() {
    let result = make_verify_result(VerifyStatus::Verified);
    let repr = result.__repr__();
    assert!(repr.contains("VerifyResult"));
    assert!(repr.contains("VERIFIED"));
    assert!(repr.contains("IBP"));
}

#[test]
fn test_verify_runtime_signatures_include_backend_kwarg() {
    use pyo3::types::PyModule;

    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "ny").expect("create module");
        ny(&module).expect("init ny");
        let inspect = PyModule::import(py, "inspect").expect("import inspect");

        for function_name in ["verify", "verify_bytes", "verify_torch"] {
            let function = module
                .getattr(function_name)
                .unwrap_or_else(|_| panic!("missing runtime function: {function_name}"));
            let signature = inspect
                .call_method1("signature", (function,))
                .expect("inspect.signature should work")
                .to_string();

            assert!(
                signature.contains("backend='auto'"),
                "{function_name} signature should expose backend='auto', got {signature}"
            );
        }
    });
}

// =========================================================================
// Python Verify Backend Helper Tests (#3627)
// =========================================================================

use crate::verify::{
    build_beta_crown_verifier, build_standard_verifier, resolve_verify_backend,
    resolve_verify_backend_with_factory,
};
use ndarray::{Array1, Array2};
use ny_core::{Bound, VerificationSpec};
use ny_gpu::Backend;
use ny_propagate::{
    layers::{Layer, LinearLayer, ReLULayer},
    PropagationConfig, PropagationMethod,
};
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Instant;

struct CountingGpuCrownEngine {
    expected_lower: Vec<f32>,
    expected_upper: Vec<f32>,
    gpu_calls: AtomicUsize,
    crown_backward_deadline: Mutex<Option<Instant>>,
}

impl CountingGpuCrownEngine {
    fn from_expected(expected: &ny_tensor::BoundedTensor) -> Self {
        Self {
            expected_lower: expected.lower().iter().copied().collect(),
            expected_upper: expected.upper().iter().copied().collect(),
            gpu_calls: AtomicUsize::new(0),
            crown_backward_deadline: Mutex::new(None),
        }
    }

    fn gpu_calls(&self) -> usize {
        self.gpu_calls.load(Ordering::SeqCst)
    }
}

impl ny_core::GemmEngine for CountingGpuCrownEngine {
    fn gemm_f32(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> ny_core::Result<Vec<f32>> {
        let engine = ny_core::NaiveCpuGemmEngine;
        ny_core::GemmEngine::gemm_f32(&engine, m, k, n, a, b)
    }

    fn as_gpu_crown_backward(&self) -> Option<&dyn ny_core::GpuCrownBackward> {
        Some(self)
    }
}

impl ny_core::GpuCrownBackward for CountingGpuCrownEngine {
    fn crown_backward_gpu(
        &self,
        _layers: &[ny_core::GpuCrownLayer],
        _spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> ny_core::Result<ny_core::GpuCrownResult> {
        if self
            .crown_backward_deadline
            .lock()
            .expect("mock GPU deadline mutex should not be poisoned")
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(ny_core::NyError::DeadlineExceeded(
                "mock Python GPU CROWN deadline exceeded before launch".to_string(),
            ));
        }
        self.gpu_calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(input_lower.len(), input_upper.len());
        assert_eq!(num_specs, self.expected_lower.len());
        Ok(ny_core::GpuCrownResult {
            lower_bounds: self.expected_lower.clone(),
            upper_bounds: self.expected_upper.clone(),
        })
    }

    fn crown_backward_gpu_sound(
        &self,
        layers: &[ny_core::GpuCrownLayer],
        spec: &[f32],
        num_specs: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> ny_core::Result<ny_core::GpuCrownResult> {
        // The fixture returns bounds precomputed by the proven CPU CROWN path,
        // so it legitimately models the sound backend contract.
        self.crown_backward_gpu(layers, spec, num_specs, input_lower, input_upper)
    }

    fn provides_sound_gpu_crown(&self) -> bool {
        true
    }

    fn set_crown_backward_deadline(&self, deadline: Option<Instant>) {
        *self
            .crown_backward_deadline
            .lock()
            .expect("mock GPU deadline mutex should not be poisoned") = deadline;
    }

    fn honors_crown_backward_deadline(&self) -> bool {
        true
    }
}

#[test]
fn test_resolve_verify_backend_cpu_crown_returns_no_engine() {
    let backend = resolve_verify_backend("cpu", PropagationMethod::Crown).unwrap();
    assert!(!backend.use_gpu);
    assert!(backend.engine.is_none());
}

#[test]
fn test_resolve_verify_backend_non_crown_method_skips_device_init() {
    let backend = resolve_verify_backend("wgpu", PropagationMethod::Ibp).unwrap();
    assert!(!backend.use_gpu);
    assert!(backend.engine.is_none());
}

#[test]
fn test_resolve_verify_backend_auto_prefers_wgpu() {
    let attempted = RefCell::new(Vec::new());
    let backend =
        resolve_verify_backend_with_factory("auto", PropagationMethod::Crown, |candidate| {
            attempted.borrow_mut().push(candidate);
            match candidate {
                Backend::Wgpu => Ok(Arc::new(ny_core::NaiveCpuGemmEngine)),
                other => panic!("unexpected auto backend candidate: {other}"),
            }
        })
        .unwrap();

    assert!(backend.use_gpu);
    assert!(backend.engine.is_some());
    assert_eq!(attempted.into_inner(), vec![Backend::Wgpu]);
}

#[test]
fn test_resolve_verify_backend_auto_falls_back_to_cpu_when_gpu_unavailable() {
    let attempted = RefCell::new(Vec::new());
    let backend =
        resolve_verify_backend_with_factory("auto", PropagationMethod::AlphaCrown, |candidate| {
            attempted.borrow_mut().push(candidate);
            Err(ny_core::NyError::InvalidSpec(format!(
                "{candidate} unavailable"
            )))
        })
        .unwrap();

    assert!(!backend.use_gpu);
    assert!(backend.engine.is_none());
    assert_eq!(attempted.into_inner(), vec![Backend::Wgpu]);
}

#[test]
fn test_resolve_verify_backend_rejects_unknown_backend() {
    match resolve_verify_backend("invalid", PropagationMethod::Crown) {
        Ok(_) => panic!("invalid backend should return an error"),
        Err(error) => assert!(error.to_string().contains("Unknown backend: invalid")),
    }
}

#[test]
fn test_resolve_verify_backend_non_crown_still_validates_backend_name() {
    match resolve_verify_backend("invalid", PropagationMethod::Ibp) {
        Ok(_) => panic!("invalid backend should still return an error for non-CROWN methods"),
        Err(error) => assert!(error.to_string().contains("Unknown backend: invalid")),
    }
}

#[test]
fn test_build_standard_verifier_uses_stored_engine_without_deadline() {
    let weight1 =
        Array2::from_shape_vec((4, 2), vec![1.0, 0.5, -0.5, 1.0, 0.3, -0.7, -0.2, 0.8]).unwrap();
    let weight2 = Array2::from_shape_vec((1, 4), vec![1.0, -0.5, 0.3, 0.2]).unwrap();
    let mut network = ny_propagate::Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    let input = ny_tensor::BoundedTensor::new(
        Array1::from_vec(vec![-1.0, -1.0]).into_dyn(),
        Array1::from_vec(vec![1.0, 1.0]).into_dyn(),
    )
    .unwrap();
    let expected = network
        .propagate_crown_with_engine(&input, Some(&ny_core::NaiveCpuGemmEngine))
        .unwrap();
    let mock_gpu = Arc::new(CountingGpuCrownEngine::from_expected(&expected));
    let verifier = build_standard_verifier(
        PropagationConfig {
            method: PropagationMethod::Crown,
            ..Default::default()
        },
        Some(mock_gpu.clone()),
    );
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        None,
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();
    assert!(matches!(
        result,
        ny_core::VerificationResult::Verified { .. }
    ));
    assert!(
        mock_gpu.gpu_calls() > 0,
        "build_standard_verifier should preserve its stored engine for an unbounded verify()"
    );
}

#[test]
fn test_build_beta_crown_verifier_stores_engine() {
    let expected = ny_tensor::BoundedTensor::new(
        Array1::from_vec(vec![-1.0]).into_dyn(),
        Array1::from_vec(vec![1.0]).into_dyn(),
    )
    .unwrap();
    let mock_gpu = Arc::new(CountingGpuCrownEngine::from_expected(&expected));
    let verifier =
        build_beta_crown_verifier(ny_propagate::BetaCrownConfig::default(), Some(mock_gpu));
    assert!(verifier.engine_arc().is_some());
}

// =========================================================================
// Clone Tests
// =========================================================================

#[test]
fn test_layer_comparison_clone() {
    let original = LayerComparison {
        name: "layer".to_string(),
        name_b: None,
        max_diff: 0.001,
        mean_diff: 0.0005,
        exceeds_tolerance: false,
        shape_a: vec![1, 64],
        shape_b: vec![1, 64],
    };
    let cloned = original.clone();
    assert_eq!(cloned.name, original.name);
    assert_eq!(cloned.max_diff, original.max_diff);
}

#[test]
fn test_diff_result_clone() {
    let original = make_diff_result(Some(1));
    let cloned = original.clone();
    assert_eq!(cloned.layers.len(), original.layers.len());
    assert_eq!(cloned.first_bad_layer, original.first_bad_layer);
}

#[test]
fn test_sensitivity_result_clone() {
    let original = make_sensitivity_result();
    let cloned = original.clone();
    assert_eq!(cloned.layers.len(), original.layers.len());
    assert_eq!(cloned.max_sensitivity, original.max_sensitivity);
}

#[test]
fn test_quantization_result_clone() {
    let original = make_quantization_result();
    let cloned = original.clone();
    assert_eq!(cloned.layers.len(), original.layers.len());
    assert_eq!(cloned.float16_safe, original.float16_safe);
}

#[test]
fn test_profile_result_clone() {
    let original = make_profile_result();
    let cloned = original.clone();
    assert_eq!(cloned.layers.len(), original.layers.len());
    assert_eq!(cloned.difficulty_score, original.difficulty_score);
}

#[test]
fn test_verify_result_clone() {
    let original = make_verify_result(VerifyStatus::Verified);
    let cloned = original.clone();
    assert_eq!(cloned.method, original.method);
    assert_eq!(cloned.epsilon, original.epsilon);
}

// =========================================================================
// Boundary Validation Tests (#2797)
// =========================================================================
// Validates that NaN, negative, and Inf inputs produce Err at the
// Python-Rust boundary instead of propagating to downstream panics.

use super::utils::{validate_epsilon, validate_input_finite, validate_tolerance};

#[test]
fn test_validate_epsilon_valid_zero() {
    assert!(validate_epsilon(0.0).is_ok());
}

#[test]
fn test_validate_epsilon_valid_positive() {
    assert!(validate_epsilon(0.01).is_ok());
}

#[test]
fn test_validate_epsilon_valid_small() {
    assert!(validate_epsilon(1e-7).is_ok());
}

#[test]
fn test_validate_epsilon_rejects_nan() {
    assert!(validate_epsilon(f32::NAN).is_err());
}

#[test]
fn test_validate_epsilon_rejects_negative() {
    assert!(validate_epsilon(-1.0).is_err());
}

#[test]
fn test_validate_epsilon_rejects_neg_small() {
    assert!(validate_epsilon(-1e-7).is_err());
}

#[test]
fn test_validate_epsilon_rejects_positive_inf() {
    assert!(validate_epsilon(f32::INFINITY).is_err());
}

#[test]
fn test_validate_epsilon_rejects_negative_inf() {
    assert!(validate_epsilon(f32::NEG_INFINITY).is_err());
}

#[test]
fn test_validate_tolerance_valid_zero() {
    assert!(validate_tolerance(0.0).is_ok());
}

#[test]
fn test_validate_tolerance_valid_positive() {
    assert!(validate_tolerance(1e-4).is_ok());
}

#[test]
fn test_validate_tolerance_rejects_nan() {
    assert!(validate_tolerance(f32::NAN).is_err());
}

#[test]
fn test_validate_tolerance_rejects_negative() {
    assert!(validate_tolerance(-0.5).is_err());
}

#[test]
fn test_validate_tolerance_rejects_positive_inf() {
    assert!(validate_tolerance(f32::INFINITY).is_err());
}

#[test]
fn test_validate_tolerance_rejects_negative_inf() {
    assert!(validate_tolerance(f32::NEG_INFINITY).is_err());
}

// =========================================================================
// Input Array NaN/Inf Validation Tests (#2898)
// =========================================================================
// Tests validate_input_finite(), the shared validator used by diff(),
// diff_bytes(), run_with_intermediates(), and compare() to reject
// NaN/Inf numpy input arrays at the Python-Rust boundary.

use ndarray::ArrayD;

#[test]
fn test_validate_input_finite_accepts_valid() {
    let arr = ArrayD::from_elem(vec![2, 3].as_slice(), 1.0_f32);
    assert!(validate_input_finite(&arr).is_ok());
}

#[test]
fn test_validate_input_finite_accepts_zeros() {
    let arr = ArrayD::from_elem(vec![1, 4].as_slice(), 0.0_f32);
    assert!(validate_input_finite(&arr).is_ok());
}

#[test]
fn test_validate_input_finite_accepts_negatives() {
    let arr = ArrayD::from_elem(vec![3].as_slice(), -42.0_f32);
    assert!(validate_input_finite(&arr).is_ok());
}

#[test]
fn test_validate_input_finite_accepts_empty() {
    let arr = ArrayD::from_elem(vec![0].as_slice(), 0.0_f32);
    assert!(validate_input_finite(&arr).is_ok());
}

#[test]
fn test_validate_input_finite_rejects_all_nan() {
    let arr = ArrayD::from_elem(vec![2, 3].as_slice(), f32::NAN);
    assert!(validate_input_finite(&arr).is_err());
}

#[test]
fn test_validate_input_finite_rejects_single_nan() {
    let mut arr = ArrayD::from_elem(vec![4].as_slice(), 1.0_f32);
    arr[[2]] = f32::NAN;
    assert!(validate_input_finite(&arr).is_err());
}

#[test]
fn test_validate_input_finite_rejects_positive_inf() {
    let arr = ArrayD::from_elem(vec![2].as_slice(), f32::INFINITY);
    assert!(validate_input_finite(&arr).is_err());
}

#[test]
fn test_validate_input_finite_rejects_negative_inf() {
    let arr = ArrayD::from_elem(vec![2].as_slice(), f32::NEG_INFINITY);
    assert!(validate_input_finite(&arr).is_err());
}

#[test]
fn test_validate_input_finite_rejects_mixed_inf() {
    let mut arr = ArrayD::from_elem(vec![3].as_slice(), 0.5_f32);
    arr[[1]] = f32::INFINITY;
    assert!(validate_input_finite(&arr).is_err());
}

// =========================================================================
// Beta-CROWN Output Bounds Regression Tests (#2802)
// =========================================================================
// Validates that build_verify_result returns actual computed bounds
// instead of fabricated [-inf, +inf] when real bounds are available.

use crate::verify::build_verify_result;
use ny_core::{Bound as RustBound, MethodUsed, VerificationResult as RustVerificationResult};

/// Helper: build a RustVerificationResult::Verified with given output bounds.
fn make_rust_verified(output_bounds: Vec<RustBound>) -> RustVerificationResult {
    RustVerificationResult::Verified {
        provenance: ny_core::SoundnessProvenance::sound(),
        output_bounds,
        proof: None,
        actual_method: Some(MethodUsed::BetaCrown),
    }
}

/// Regression test #2802: Verified status with real bounds passes them through
/// to the Python VerifyResult (not vacuous [-inf, +inf]).
#[test]
fn test_build_verify_result_verified_real_bounds_2802() {
    let bounds = vec![RustBound::new(-0.5, 1.5), RustBound::new(0.1, 0.9)];
    let rust_result = make_rust_verified(bounds);
    let py_result = build_verify_result(rust_result, "beta".to_string(), 0.01);

    assert!(matches!(py_result.status, VerifyStatus::Verified));
    let output_bounds = py_result.output_bounds.expect("should have output bounds");
    assert_eq!(output_bounds.len(), 2);
    // Bounds must be the actual computed values, not [-inf, +inf]
    assert!((output_bounds[0].lower - (-0.5)).abs() < 1e-6);
    assert!((output_bounds[0].upper - 1.5).abs() < 1e-6);
    assert!((output_bounds[1].lower - 0.1).abs() < 1e-6);
    assert!((output_bounds[1].upper - 0.9).abs() < 1e-6);
}

/// Regression test #2802: Verified status with empty bounds (BaB loop path
/// where output_bounds is None) returns empty vec, not fabricated bounds.
#[test]
fn test_build_verify_result_verified_empty_bounds_2802() {
    let rust_result = make_rust_verified(vec![]);
    let py_result = build_verify_result(rust_result, "beta".to_string(), 0.01);

    assert!(matches!(py_result.status, VerifyStatus::Verified));
    let output_bounds = py_result.output_bounds.expect("should have output bounds");
    assert!(
        output_bounds.is_empty(),
        "empty bounds should not be inflated to [-inf, +inf]"
    );
}

/// Regression test #2802: Unknown status with real bounds passes them through.
#[test]
fn test_build_verify_result_unknown_real_bounds_2802() {
    let bounds = vec![RustBound::new(-1.0, 2.0)];
    let rust_result = RustVerificationResult::Unknown {
        provenance: ny_core::SoundnessProvenance::sound(),
        bounds,
        reason: ny_core::UnknownReason::PotentialViolation,
        actual_method: Some(MethodUsed::BetaCrown),
    };
    let py_result = build_verify_result(rust_result, "beta".to_string(), 0.01);

    assert!(matches!(py_result.status, VerifyStatus::Unknown));
    let output_bounds = py_result.output_bounds.expect("should have output bounds");
    assert_eq!(output_bounds.len(), 1);
    assert!((output_bounds[0].lower - (-1.0)).abs() < 1e-6);
    assert!((output_bounds[0].upper - 2.0).abs() < 1e-6);
}

/// Regression test #2802: Timeout status with None partial_bounds returns None
/// output_bounds (not fabricated [-inf, +inf]).
#[test]
fn test_build_verify_result_timeout_none_bounds_2802() {
    let rust_result = RustVerificationResult::Timeout {
        provenance: ny_core::SoundnessProvenance::sound(),
        partial_bounds: None,
        actual_method: Some(MethodUsed::BetaCrown),
    };
    let py_result = build_verify_result(rust_result, "beta".to_string(), 0.01);

    assert!(matches!(py_result.status, VerifyStatus::Timeout));
    assert!(
        py_result.output_bounds.is_none(),
        "Timeout with no bounds should be None, not fabricated"
    );
}

/// Regression test #2802: Timeout status with real partial_bounds passes them through.
#[test]
fn test_build_verify_result_timeout_real_bounds_2802() {
    let bounds = vec![RustBound::new_allow_infinite(-2.0, 3.0)];
    let rust_result = RustVerificationResult::Timeout {
        provenance: ny_core::SoundnessProvenance::sound(),
        partial_bounds: Some(bounds),
        actual_method: Some(MethodUsed::BetaCrown),
    };
    let py_result = build_verify_result(rust_result, "beta".to_string(), 0.01);

    assert!(matches!(py_result.status, VerifyStatus::Timeout));
    let output_bounds = py_result.output_bounds.expect("should have partial bounds");
    assert_eq!(output_bounds.len(), 1);
    assert!((output_bounds[0].lower - (-2.0)).abs() < 1e-6);
    assert!((output_bounds[0].upper - 3.0).abs() < 1e-6);
}

// =========================================================================
// BaB Threshold Derivation Tests (#3229)
// =========================================================================
// Validates that derive_bab_threshold correctly computes the BaB threshold
// from spec output bounds, matching the pattern in network.rs:183-195.
// The previous hardcoded `f32::NEG_INFINITY` caused BaB to trivially verify
// every property since any finite lower bound satisfies `> NEG_INFINITY`.

use crate::verify::derive_bab_threshold;

/// Regression test #3229: Trivial output bounds (-inf, +inf) produce NEG_INFINITY
/// threshold (no finite lower constraints → BaB has no meaningful target).
#[test]
fn test_derive_bab_threshold_trivial_bounds_3229() {
    let bounds = vec![
        RustBound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY),
        RustBound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY),
    ];
    assert_eq!(derive_bab_threshold(&bounds), f32::NEG_INFINITY);
}

/// Regression test #3229: Finite lower bounds produce meaningful threshold.
/// BaB must check `lower_bound > threshold` against this value, not NEG_INFINITY.
#[test]
fn test_derive_bab_threshold_finite_lower_bounds_3229() {
    let bounds = vec![RustBound::new(-0.5, 1.0), RustBound::new(0.3, 2.0)];
    // Threshold = min(finite lower bounds) = min(-0.5, 0.3) = -0.5
    assert!((derive_bab_threshold(&bounds) - (-0.5)).abs() < 1e-6);
}

/// Regression test #3229: Mixed finite and infinite lower bounds — only
/// finite bounds contribute to threshold.
#[test]
fn test_derive_bab_threshold_mixed_finite_infinite_3229() {
    let bounds = vec![
        RustBound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY),
        RustBound::new(0.1, 0.9),
        RustBound::new_allow_infinite(f32::NEG_INFINITY, 5.0),
    ];
    // Only bound[1] has finite lower (0.1)
    assert!((derive_bab_threshold(&bounds) - 0.1).abs() < 1e-6);
}

/// Regression test #3229: Single output bound with finite lower.
#[test]
fn test_derive_bab_threshold_single_finite_bound_3229() {
    let bounds = vec![RustBound::new(-1.0, 1.0)];
    assert!((derive_bab_threshold(&bounds) - (-1.0)).abs() < 1e-6);
}

/// Regression test #3229: Empty bounds produce NEG_INFINITY (defensive).
#[test]
fn test_derive_bab_threshold_empty_bounds_3229() {
    assert_eq!(derive_bab_threshold(&[]), f32::NEG_INFINITY);
}

/// Regression test #3229: Multiple finite bounds — threshold is the minimum.
#[test]
fn test_derive_bab_threshold_multiple_finite_minimum_3229() {
    // With multiple finite lower bounds, threshold = min of all
    let bounds = vec![
        RustBound::new(0.5, 1.0),
        RustBound::new(-0.2, 0.8),
        RustBound::new(0.1, 0.9),
    ];
    // min(0.5, -0.2, 0.1) = -0.2
    assert!((derive_bab_threshold(&bounds) - (-0.2)).abs() < 1e-6);
}

// =========================================================================
// Output Specification Tests
// =========================================================================
// The (-inf, +inf) requirement is satisfied by every network, so a run
// without a caller-supplied output specification must never report a
// property verdict (Verified/Violated); it folds to Unknown with the
// computed bounds attached.

use crate::verify::{
    bab_verified_spec_gap, build_output_spec_bounds, fold_unspecified_property,
    NO_OUTPUT_SPEC_REASON,
};

#[test]
fn test_build_output_spec_bounds_valid_mixed() {
    let bounds =
        build_output_spec_bounds(&[(0.0, 1.0), (f32::NEG_INFINITY, 0.5)], 2).expect("valid spec");
    assert_eq!(bounds.len(), 2);
    assert!((bounds[0].lower() - 0.0).abs() < 1e-6);
    assert!((bounds[0].upper() - 1.0).abs() < 1e-6);
    assert_eq!(bounds[1].lower(), f32::NEG_INFINITY);
    assert!((bounds[1].upper() - 0.5).abs() < 1e-6);
}

#[test]
fn test_build_output_spec_bounds_rejects_length_mismatch() {
    let err = build_output_spec_bounds(&[(0.0, 1.0)], 2).expect_err("length mismatch");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(ref message)
            if message.contains("1 entries") && message.contains("2 outputs")),
        "unexpected error: {err}"
    );
}

#[test]
fn test_build_output_spec_bounds_rejects_nan() {
    let err = build_output_spec_bounds(&[(f32::NAN, 1.0)], 1).expect_err("NaN bound");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(ref message) if message.contains("NaN")),
        "unexpected error: {err}"
    );
}

#[test]
fn test_build_output_spec_bounds_rejects_inverted() {
    let err = build_output_spec_bounds(&[(1.0, 0.0)], 1).expect_err("inverted bound");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(ref message) if message.contains("malformed")),
        "unexpected error: {err}"
    );
}

/// An all-(-inf, +inf) requirement is satisfied by every network, so it must
/// be rejected rather than trivially reported as Verified.
#[test]
fn test_build_output_spec_bounds_rejects_unconstrained() {
    let err = build_output_spec_bounds(
        &[
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, f32::INFINITY),
        ],
        2,
    )
    .expect_err("unconstrained spec");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(ref message)
            if message.contains("nothing to verify")),
        "unexpected error: {err}"
    );
}

#[test]
fn test_fold_unspecified_property_verified_becomes_unknown() {
    let result = RustVerificationResult::Verified {
        provenance: ny_core::SoundnessProvenance::sound(),
        output_bounds: vec![RustBound::new(-0.5, 1.5)],
        proof: None,
        actual_method: Some(MethodUsed::Crown),
    };
    match fold_unspecified_property(result) {
        RustVerificationResult::Unknown {
            bounds,
            reason,
            actual_method,
            ..
        } => {
            assert_eq!(bounds.len(), 1, "computed bounds must be preserved");
            assert!((bounds[0].lower() - (-0.5)).abs() < 1e-6);
            assert_eq!(
                reason,
                ny_core::UnknownReason::Other {
                    message: NO_OUTPUT_SPEC_REASON.to_string()
                }
            );
            assert_eq!(actual_method, Some(MethodUsed::Crown));
        }
        other => panic!("Verified without a spec must fold to Unknown, got {other:?}"),
    }
}

#[test]
fn test_fold_unspecified_property_violated_becomes_unknown() {
    let result = RustVerificationResult::Violated {
        provenance: ny_core::SoundnessProvenance::sound(),
        counterexample: vec![0.0],
        output: vec![0.0],
        details: None,
        actual_method: Some(MethodUsed::BetaCrown),
    };
    assert!(
        matches!(
            fold_unspecified_property(result),
            RustVerificationResult::Unknown { .. }
        ),
        "Violated without a spec must fold to Unknown"
    );
}

#[test]
fn test_fold_unspecified_property_passes_through_timeout_and_unknown() {
    let timeout = RustVerificationResult::Timeout {
        provenance: ny_core::SoundnessProvenance::sound(),
        partial_bounds: None,
        actual_method: None,
    };
    assert!(matches!(
        fold_unspecified_property(timeout),
        RustVerificationResult::Timeout { .. }
    ));

    let unknown = RustVerificationResult::Unknown {
        provenance: ny_core::SoundnessProvenance::sound(),
        bounds: vec![],
        reason: ny_core::UnknownReason::PotentialViolation,
        actual_method: None,
    };
    assert!(matches!(
        fold_unspecified_property(unknown),
        RustVerificationResult::Unknown {
            reason: ny_core::UnknownReason::PotentialViolation,
            ..
        }
    ));
}

// =========================================================================
// BaB Per-Output Spec Gap Tests
// =========================================================================
// BaB Verified only proves min(all outputs) >= threshold (the global minimum
// of the finite required lowers), so each output must still be checked
// against its own requirement before reporting Verified.

#[test]
fn test_bab_verified_spec_gap_uniform_lower_only_spec_passes() {
    // Every output requires >= 0.0 with no upper constraint; the BaB verdict
    // itself justifies [threshold, +inf) per output, so no gap remains.
    let required = vec![
        RustBound::new_allow_infinite(0.0, f32::INFINITY),
        RustBound::new_allow_infinite(0.0, f32::INFINITY),
    ];
    assert_eq!(bab_verified_spec_gap(&required, None, 0.0), None);
}

#[test]
fn test_bab_verified_spec_gap_tighter_individual_lower_downgrades() {
    // Output 1 requires >= 5.0 but BaB only proved >= 0.0 globally.
    let required = vec![
        RustBound::new_allow_infinite(0.0, f32::INFINITY),
        RustBound::new_allow_infinite(5.0, f32::INFINITY),
    ];
    let gap = bab_verified_spec_gap(&required, None, 0.0).expect("gap expected");
    assert!((gap - 5.0).abs() < 1e-6);
}

#[test]
fn test_bab_verified_spec_gap_finite_upper_without_bounds_downgrades() {
    // The BaB threshold says nothing about upper bounds, so a finite upper
    // requirement cannot be confirmed without per-output tensor bounds.
    let required = vec![RustBound::new(0.0, 1.0)];
    let gap = bab_verified_spec_gap(&required, None, 0.0).expect("gap expected");
    assert_eq!(gap, f32::INFINITY);
}

#[test]
fn test_bab_verified_spec_gap_computed_bounds_satisfy_spec() {
    let required = vec![RustBound::new(0.0, 1.0)];
    let computed = vec![RustBound::new(0.2, 0.8)];
    assert_eq!(
        bab_verified_spec_gap(&required, Some(computed.as_slice()), 0.0),
        None
    );
}

#[test]
fn test_bab_verified_spec_gap_computed_bounds_violate_upper() {
    let required = vec![RustBound::new(0.0, 1.0)];
    let computed = vec![RustBound::new(0.2, 1.5)];
    let gap =
        bab_verified_spec_gap(&required, Some(computed.as_slice()), 0.0).expect("gap expected");
    assert!((gap - 0.5).abs() < 1e-6);
}

/// Infinite endpoints on both sides must compare, not produce NaN via
/// inf - inf arithmetic.
#[test]
fn test_bab_verified_spec_gap_matching_infinite_endpoints_no_nan() {
    let required = vec![RustBound::new_allow_infinite(f32::NEG_INFINITY, 1.0)];
    let computed = vec![RustBound::new_allow_infinite(f32::NEG_INFINITY, 0.5)];
    assert_eq!(
        bab_verified_spec_gap(&required, Some(computed.as_slice()), f32::NEG_INFINITY),
        None
    );
}
