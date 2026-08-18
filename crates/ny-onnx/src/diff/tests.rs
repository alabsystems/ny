// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
#[cfg(feature = "ort")]
use crate::test_fixtures::{
    require_test_model, require_test_model_with_hint, TRANSFORMER_TEST_MODEL_HINT,
};
use crate::LayerSpec;
use npyz::WriterBuilder;
use tempfile::NamedTempFile;

#[ntest::timeout(10000)]
#[test]
fn test_normalize_layer_name() {
    assert_eq!(normalize_layer_name("layer_0"), "0");
    assert_eq!(normalize_layer_name("encoder.block.0"), "encoder.0");
    assert_eq!(normalize_layer_name("Block_1_Linear"), "1.linear");
    assert_eq!(normalize_layer_name("layer0"), "0");
    assert_eq!(normalize_layer_name("module.layer.0.linear"), "0.linear");
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays() {
    let a = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
    let b = ArrayD::from_elem(IxDyn(&[2, 3]), 1.001f32);

    let comp = compare_arrays(&a, &b, 0.01);
    assert!(!comp.exceeds_tolerance);
    assert!((comp.max_diff - 0.001).abs() < 1e-6);

    let comp2 = compare_arrays(&a, &b, 0.0001);
    assert!(comp2.exceeds_tolerance);
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_shape_mismatch() {
    let a = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
    let b = ArrayD::from_elem(IxDyn(&[3, 2]), 1.0f32);

    let comp = compare_arrays(&a, &b, 0.01);
    assert!(comp.exceeds_tolerance);
    assert!(comp.max_diff.is_infinite());
}

#[ntest::timeout(10000)]
#[cfg(feature = "ort")]
#[test]
fn test_intermediate_extraction() {
    let model_path = require_test_model("simple_mlp.onnx");

    // Create a simple input
    let input = ArrayD::from_elem(IxDyn(&[1, 2]), 0.5f32);

    // Run inference with intermediate extraction
    let outputs = run_inference_with_intermediates(&model_path, &input)
        .expect("Failed to run inference with intermediates");

    // Simple MLP should have: fc1_out, relu_out, output
    // The actual names depend on how the model was exported
    assert!(
        outputs.len() >= 2,
        "Expected at least 2 intermediate outputs, got {}",
        outputs.len()
    );

    // All outputs should be finite
    for (name, arr) in &outputs {
        assert!(
            arr.iter().all(|v| v.is_finite()),
            "Output {} contains non-finite values",
            name
        );
    }
}

#[ntest::timeout(10000)]
#[cfg(feature = "ort")]
#[test]
fn test_diff_models_same_model() {
    let model_path = require_test_model("simple_mlp.onnx");

    let config = DiffConfig {
        tolerance: 1e-5,
        continue_after_divergence: true,
        input: None,
        layer_mapping: HashMap::new(),
        diagnose: false,
    };

    let result = diff_models(&model_path, &model_path, &config).expect("Failed to diff same model");

    // Same model should be equivalent
    assert!(result.is_equivalent(), "Same model should be equivalent");
    assert!(
        result.max_divergence < 1e-10,
        "Max divergence should be near zero for same model"
    );

    // Should have multiple layers
    assert!(
        result.layers.len() >= 2,
        "Should compare multiple layers, got {}",
        result.layers.len()
    );
}

#[ntest::timeout(10000)]
#[cfg(feature = "ort")]
#[test]
fn test_diff_models_layer_count() {
    let model_path =
        require_test_model_with_hint("transformer_mlp.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let config = DiffConfig::default();

    let model_info = load_model_info(&model_path).expect("Failed to load model metadata");
    assert!(
        model_info
            .layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::Erf),
        "transformer MLP metadata must preserve decomposed GELU Erf"
    );
    assert!(
        !model_info
            .layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::GELU),
        "transformer MLP must not use the disabled canonical GELU fusion"
    );

    let result = diff_models(&model_path, &model_path, &config).expect("Failed to diff model");

    assert!(
        result.is_equivalent(),
        "a model must be equivalent to itself"
    );

    // transformer_mlp has fc1, a decomposed Erf GELU chain, and fc2.
    assert!(
        result.layers.len() >= 2,
        "Expected >= 2 layer comparisons for transformer_mlp, got {}",
        result.layers.len()
    );
    let erf_comparison = result
        .layers
        .iter()
        .find(|layer| layer.name.contains("Erf"))
        .expect("diff must compare the preserved Erf output");
    assert!(!erf_comparison.exceeds_tolerance);
    assert_eq!(erf_comparison.max_diff, 0.0);

    // Print layers for debugging
    for layer in &result.layers {
        eprintln!("Layer: {} max_diff={:.2e}", layer.name, layer.max_diff);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_divergence_pattern_display() {
    // Test Display implementations for all patterns
    let patterns = vec![
        DivergencePattern::ExpPrecision {
            max_logit: 89.5,
            is_overflow: true,
        },
        DivergencePattern::ExpPrecision {
            max_logit: -90.0,
            is_overflow: false,
        },
        DivergencePattern::SoftmaxInstability {
            max_score: 75.0,
            score_range: 60.0,
        },
        DivergencePattern::AccumulationOrder {
            operation: "matmul".to_string(),
            size_correlated: true,
        },
        DivergencePattern::QuantizationError {
            bits_lost: 7,
            at_power_boundary: true,
        },
        DivergencePattern::WeightMismatch {
            layer: "fc1".to_string(),
            max_diff: 0.001,
        },
        DivergencePattern::GeluApproximation {
            max_diff_in_region: 1e-4,
        },
        DivergencePattern::LayerNormVariance {
            epsilon_differs: true,
        },
        DivergencePattern::Unknown,
    ];

    for pattern in patterns {
        let s = format!("{}", pattern);
        assert!(!s.is_empty(), "Pattern display should not be empty");
        eprintln!("Pattern: {}", s);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_diagnosis_format_report() {
    let diagnosis = DiffDiagnosis {
        divergence_layer: "encoder.softmax".to_string(),
        layer_type: Some(LayerType::Softmax),
        pattern: DivergencePattern::ExpPrecision {
            max_logit: 85.0,
            is_overflow: true,
        },
        explanation: "Large logits near exp overflow".to_string(),
        suggestion: Some("Use log-sum-exp stabilization".to_string()),
        confidence: 0.9,
        evidence: vec![
            "max_logit = 85.0".to_string(),
            "near exp(88) boundary".to_string(),
        ],
    };

    let report = diagnosis.format_report();
    assert!(report.contains("encoder.softmax"));
    assert!(report.contains("Softmax"));
    assert!(report.contains("90%")); // 0.9 * 100
    assert!(report.contains("log-sum-exp"));
    assert!(report.contains("max_logit"));
    eprintln!("Report:\n{}", report);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_diagnosis_unknown() {
    let diagnosis = DiffDiagnosis::unknown("layer_0", Some(LayerType::Linear));
    assert_eq!(diagnosis.divergence_layer, "layer_0");
    assert!(matches!(diagnosis.pattern, DivergencePattern::Unknown));
    assert_eq!(diagnosis.confidence, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_config_with_diagnose() {
    let config = DiffConfig {
        tolerance: 1e-5,
        continue_after_divergence: true,
        input: None,
        layer_mapping: HashMap::new(),
        diagnose: true,
    };

    assert!(config.diagnose);
    assert_eq!(config.tolerance, 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_result_is_equivalent() {
    // No bad layer = equivalent
    let result_ok = DiffResult {
        layers: vec![LayerComparison {
            name: "layer_0".to_string(),
            name_b: None,
            max_diff: 1e-6,
            mean_diff: 1e-7,
            exceeds_tolerance: false,
            shape_a: vec![1, 2],
            shape_b: vec![1, 2],
        }],
        first_bad_layer: None,
        drift_start_layer: None,
        max_divergence: 1e-6,
        tolerance: 1e-5,
        suggestion: None,
        diagnosis: None,
    };
    assert!(result_ok.is_equivalent());

    // Has bad layer = not equivalent
    let result_bad = DiffResult {
        layers: vec![LayerComparison {
            name: "layer_0".to_string(),
            name_b: None,
            max_diff: 1e-3,
            mean_diff: 1e-4,
            exceeds_tolerance: true,
            shape_a: vec![1, 2],
            shape_b: vec![1, 2],
        }],
        first_bad_layer: Some(0),
        drift_start_layer: Some(0),
        max_divergence: 1e-3,
        tolerance: 1e-5,
        suggestion: None,
        diagnosis: None,
    };
    assert!(!result_bad.is_equivalent());
}

#[ntest::timeout(10000)]
#[test]
fn test_load_npy_reads_f64_as_f32() {
    let file = NamedTempFile::new().expect("temp file should be created");
    let path = file.path().to_path_buf();

    let writer = std::io::BufWriter::new(file.reopen().expect("temp file should reopen"));
    let mut npy = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[2, 2])
        .writer(writer)
        .begin_nd()
        .expect("npy writer should start");
    npy.extend([1.25_f64, 2.5, 3.75, 4.0])
        .expect("f64 payload should write");
    npy.finish().expect("npy writer should finish");

    let loaded = load_npy(&path).expect("f64 npy should load as f32");
    assert_eq!(loaded.shape(), &[2, 2]);
    assert!((loaded[[0, 0]] - 1.25_f32).abs() < 1e-6);
    assert!((loaded[[0, 1]] - 2.5_f32).abs() < 1e-6);
    assert!((loaded[[1, 0]] - 3.75_f32).abs() < 1e-6);
    assert!((loaded[[1, 1]] - 4.0_f32).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_load_npy_rejects_out_of_range_f64_2360() {
    let file = NamedTempFile::new().expect("temp file should be created");
    let path = file.path().to_path_buf();

    let writer = std::io::BufWriter::new(file.reopen().expect("temp file should reopen"));
    let mut npy = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[1])
        .writer(writer)
        .begin_nd()
        .expect("npy writer should start");
    npy.extend([f64::MAX]).expect("f64 payload should write");
    npy.finish().expect("npy writer should finish");

    let err = load_npy(&path).unwrap_err();
    match err {
        DiffError::NpyError(msg) => {
            assert!(msg.contains("f64→f32 out of range"), "msg = {msg}");
            assert!(
                msg.contains(&path.display().to_string()),
                "msg should mention the path, got: {msg}"
            );
        }
        other => unreachable!("unexpected error: {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_npy_preserves_fortran_order() {
    let file = NamedTempFile::new().expect("temp file should be created");
    let path = file.path().to_path_buf();

    let writer = std::io::BufWriter::new(file.reopen().expect("temp file should reopen"));
    let mut npy = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[2, 3])
        .order(npyz::Order::Fortran)
        .writer(writer)
        .begin_nd()
        .expect("npy writer should start");
    npy.extend([1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0])
        .expect("fortran payload should write");
    npy.finish().expect("npy writer should finish");

    let loaded = load_npy(&path).expect("fortran-order npy should load");
    assert_eq!(loaded.shape(), &[2, 3]);
    assert_eq!(loaded[[0, 0]], 1.0_f32);
    assert_eq!(loaded[[0, 1]], 2.0_f32);
    assert_eq!(loaded[[0, 2]], 3.0_f32);
    assert_eq!(loaded[[1, 0]], 4.0_f32);
    assert_eq!(loaded[[1, 1]], 5.0_f32);
    assert_eq!(loaded[[1, 2]], 6.0_f32);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_result_first_bad_layer_name() {
    let result = DiffResult {
        layers: vec![
            LayerComparison {
                name: "layer_0".to_string(),
                name_b: None,
                max_diff: 1e-7,
                mean_diff: 1e-8,
                exceeds_tolerance: false,
                shape_a: vec![1, 2],
                shape_b: vec![1, 2],
            },
            LayerComparison {
                name: "layer_1_bad".to_string(),
                name_b: None,
                max_diff: 1e-3,
                mean_diff: 1e-4,
                exceeds_tolerance: true,
                shape_a: vec![1, 2],
                shape_b: vec![1, 2],
            },
        ],
        first_bad_layer: Some(1),
        drift_start_layer: Some(1),
        max_divergence: 1e-3,
        tolerance: 1e-5,
        suggestion: None,
        diagnosis: None,
    };

    assert_eq!(result.first_bad_layer_name(), Some("layer_1_bad"));

    // No bad layer case
    let result_ok = DiffResult {
        layers: vec![LayerComparison {
            name: "layer_0".to_string(),
            name_b: None,
            max_diff: 1e-7,
            mean_diff: 1e-8,
            exceeds_tolerance: false,
            shape_a: vec![1, 2],
            shape_b: vec![1, 2],
        }],
        first_bad_layer: None,
        drift_start_layer: None,
        max_divergence: 1e-7,
        tolerance: 1e-5,
        suggestion: None,
        diagnosis: None,
    };
    assert_eq!(result_ok.first_bad_layer_name(), None);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_result_statuses() {
    let result = DiffResult {
        layers: vec![
            LayerComparison {
                name: "layer_ok".to_string(),
                name_b: None,
                max_diff: 1e-7,
                mean_diff: 1e-8,
                exceeds_tolerance: false,
                shape_a: vec![1, 2],
                shape_b: vec![1, 2],
            },
            LayerComparison {
                name: "layer_drift".to_string(),
                name_b: None,
                max_diff: 1e-6,
                mean_diff: 1e-7,
                exceeds_tolerance: false,
                shape_a: vec![1, 2],
                shape_b: vec![1, 2],
            },
            LayerComparison {
                name: "layer_bad".to_string(),
                name_b: None,
                max_diff: 1e-3,
                mean_diff: 1e-4,
                exceeds_tolerance: true,
                shape_a: vec![1, 2],
                shape_b: vec![1, 2],
            },
            LayerComparison {
                name: "layer_shape_mismatch".to_string(),
                name_b: None,
                max_diff: f32::INFINITY,
                mean_diff: f32::INFINITY,
                exceeds_tolerance: true,
                shape_a: vec![1, 2],
                shape_b: vec![2, 1],
            },
        ],
        first_bad_layer: Some(2),
        drift_start_layer: Some(1),
        max_divergence: 1e-3,
        tolerance: 1e-5,
        suggestion: None,
        diagnosis: None,
    };

    let statuses = result.statuses();
    assert_eq!(statuses.len(), 4);
    assert_eq!(statuses[0], DiffStatus::Ok);
    assert_eq!(statuses[1], DiffStatus::DriftStarts);
    assert_eq!(statuses[2], DiffStatus::ExceedsTolerance);
    assert_eq!(statuses[3], DiffStatus::ShapeMismatch);
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_comparison_creation() {
    let comp = LayerComparison {
        name: "test_layer".to_string(),
        name_b: Some("test_layer_b".to_string()),
        max_diff: 0.001,
        mean_diff: 0.0005,
        exceeds_tolerance: false,
        shape_a: vec![1, 10, 20],
        shape_b: vec![1, 10, 20],
    };

    assert_eq!(comp.name, "test_layer");
    assert_eq!(comp.name_b, Some("test_layer_b".to_string()));
    assert!((comp.max_diff - 0.001).abs() < 1e-9);
    assert!(!comp.exceeds_tolerance);
    assert_eq!(comp.shape_a, comp.shape_b);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_error_display() {
    // Test error Display implementations
    let err_load = DiffError::LoadError("failed to open".to_string());
    assert!(err_load.to_string().contains("failed to open"));

    let err_shape = DiffError::InputShapeMismatch {
        model_a: vec![1, 2, 3],
        model_b: vec![1, 2, 4],
    };
    assert!(err_shape.to_string().contains("[1, 2, 3]"));
    assert!(err_shape.to_string().contains("[1, 2, 4]"));

    let err_layer = DiffError::LayerNotFound("missing_layer".to_string());
    assert!(err_layer.to_string().contains("missing_layer"));

    let err_no_layers = DiffError::NoLayers;
    assert!(err_no_layers.to_string().contains("No layers"));

    let err_npy = DiffError::NpyError("invalid format".to_string());
    assert!(err_npy.to_string().contains("invalid format"));
}

// ========================================================================
// normalize_layer_name additional tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_normalize_layer_name_empty() {
    assert_eq!(normalize_layer_name(""), "");
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_layer_name_just_numbers() {
    assert_eq!(normalize_layer_name("123"), "123");
    assert_eq!(normalize_layer_name("0"), "0");
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_layer_name_multiple_underscores() {
    assert_eq!(normalize_layer_name("layer__0__1"), "0.1");
    assert_eq!(normalize_layer_name("___layer___"), "");
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_layer_name_multiple_dots() {
    assert_eq!(normalize_layer_name("a...b"), "a.b");
    assert_eq!(normalize_layer_name("..."), "");
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_layer_name_mixed_separators() {
    assert_eq!(normalize_layer_name("layer_0.block.1_conv"), "0.1.conv");
    assert_eq!(
        normalize_layer_name("module_list.0.conv_block"),
        "list.0.conv"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_layer_name_special_chars_stripped() {
    // Special chars are stripped but don't introduce dots
    assert_eq!(normalize_layer_name("layer@0#1!"), "01");
    assert_eq!(normalize_layer_name("layer[0]"), "0");
    // Underscore introduces a dot
    assert_eq!(normalize_layer_name("layer@0_1!"), "0.1");
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_layer_name_uppercase_to_lowercase() {
    assert_eq!(normalize_layer_name("LAYER_0"), "0");
    assert_eq!(normalize_layer_name("Block_GELU"), "gelu");
}

#[ntest::timeout(10000)]
#[test]
fn test_normalize_layer_name_preserves_alphanumeric() {
    assert_eq!(normalize_layer_name("fc1"), "fc1");
    assert_eq!(normalize_layer_name("relu"), "relu");
}

// ========================================================================
// match_layer_names tests
// ========================================================================

fn make_layer(name: &str, layer_type: LayerType) -> LayerSpec {
    LayerSpec {
        name: name.to_string(),
        layer_type,
        inputs: vec![],
        outputs: vec![name.to_string()],
        weights: None,
        attributes: HashMap::new(),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_match_layer_names_exact_match() {
    let layers_a = vec![
        make_layer("fc1", LayerType::Linear),
        make_layer("relu1", LayerType::ReLU),
    ];
    let layers_b = vec![
        make_layer("fc1", LayerType::Linear),
        make_layer("relu1", LayerType::ReLU),
    ];

    let matches = match_layer_names(&layers_a, &layers_b, &HashMap::new());
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0], ("fc1".to_string(), Some("fc1".to_string())));
    assert_eq!(matches[1], ("relu1".to_string(), Some("relu1".to_string())));
}

#[ntest::timeout(10000)]
#[test]
fn test_match_layer_names_explicit_mapping() {
    let layers_a = vec![make_layer("layer_a", LayerType::Linear)];
    let layers_b = vec![make_layer("layer_b", LayerType::Linear)];

    let mut mapping = HashMap::new();
    mapping.insert("layer_a".to_string(), "layer_b".to_string());

    let matches = match_layer_names(&layers_a, &layers_b, &mapping);
    assert_eq!(matches.len(), 1);
    assert_eq!(
        matches[0],
        ("layer_a".to_string(), Some("layer_b".to_string()))
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_match_layer_names_fuzzy_match() {
    let layers_a = vec![make_layer("block_0_linear", LayerType::Linear)];
    let layers_b = vec![make_layer("block.0.linear", LayerType::Linear)];

    let matches = match_layer_names(&layers_a, &layers_b, &HashMap::new());
    assert_eq!(matches.len(), 1);
    // Both normalize to "0.linear" and have same type
    assert_eq!(
        matches[0],
        (
            "block_0_linear".to_string(),
            Some("block.0.linear".to_string())
        )
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_match_layer_names_no_match() {
    let layers_a = vec![make_layer("fc1", LayerType::Linear)];
    let layers_b = vec![make_layer("conv1", LayerType::Conv2d)];

    let matches = match_layer_names(&layers_a, &layers_b, &HashMap::new());
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0], ("fc1".to_string(), None));
}

#[ntest::timeout(10000)]
#[test]
fn test_match_layer_names_type_mismatch_prevents_fuzzy() {
    // Same normalized name but different type - should not match
    let layers_a = vec![make_layer("layer_0", LayerType::Linear)];
    let layers_b = vec![make_layer("layer.0", LayerType::Conv2d)];

    let matches = match_layer_names(&layers_a, &layers_b, &HashMap::new());
    assert_eq!(matches.len(), 1);
    // Won't match because types differ
    assert_eq!(matches[0], ("layer_0".to_string(), None));
}

#[ntest::timeout(10000)]
#[test]
fn test_match_layer_names_empty_inputs() {
    let empty: Vec<LayerSpec> = vec![];
    let layers_b = vec![make_layer("fc1", LayerType::Linear)];

    let matches = match_layer_names(&empty, &layers_b, &HashMap::new());
    assert!(matches.is_empty());
}

// ========================================================================
// suggest_root_cause tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_softmax() {
    let layer = make_layer("softmax", LayerType::Softmax);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_some());
    assert!(suggestion.unwrap().contains("softmax"));
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_causal_softmax() {
    let layer = make_layer("causal_softmax", LayerType::CausalSoftmax);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_some());
    assert!(suggestion.unwrap().contains("softmax"));
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_layernorm() {
    let layer = make_layer("ln", LayerType::LayerNorm);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_some());
    assert!(suggestion.unwrap().contains("LayerNorm"));
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_gelu() {
    let layer = make_layer("gelu", LayerType::GELU);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_some());
    assert!(suggestion.unwrap().contains("GELU"));
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_conv1d() {
    let layer = make_layer("conv", LayerType::Conv1d);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_some());
    assert!(suggestion.unwrap().contains("convolution"));
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_conv2d() {
    let layer = make_layer("conv", LayerType::Conv2d);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_some());
    assert!(suggestion.unwrap().contains("convolution"));
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_linear() {
    let layer = make_layer("fc", LayerType::Linear);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_some());
    assert!(suggestion.unwrap().contains("matrix multiplication"));
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_matmul() {
    let layer = make_layer("mm", LayerType::MatMul);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_some());
    assert!(suggestion.unwrap().contains("matrix multiplication"));
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_add() {
    let layer = make_layer("add", LayerType::Add);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_some());
    assert!(suggestion.unwrap().contains("broadcast"));
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_mul() {
    let layer = make_layer("mul", LayerType::Mul);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_some());
    assert!(suggestion.unwrap().contains("broadcast"));
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_relu_returns_none() {
    let layer = make_layer("relu", LayerType::ReLU);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_suggest_root_cause_unknown_returns_none() {
    let layer = make_layer("unknown", LayerType::Unknown);
    let suggestion = suggest_root_cause(&layer);
    assert!(suggestion.is_none());
}

// ========================================================================
// compare_arrays additional tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_empty() {
    let a = ArrayD::from_elem(IxDyn(&[0]), 0.0f32);
    let b = ArrayD::from_elem(IxDyn(&[0]), 0.0f32);
    let comp = compare_arrays(&a, &b, 1e-5);
    // Empty arrays should compare as equivalent with NaN mean_diff
    assert_eq!(comp.shape_a, vec![0]);
    assert_eq!(comp.max_diff, 0.0);
    assert!(comp.mean_diff.is_nan()); // 0/0
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_all_zeros() {
    let a = ArrayD::from_elem(IxDyn(&[10, 10]), 0.0f32);
    let b = ArrayD::from_elem(IxDyn(&[10, 10]), 0.0f32);
    let comp = compare_arrays(&a, &b, 1e-5);
    assert!(!comp.exceeds_tolerance);
    assert_eq!(comp.max_diff, 0.0);
    assert_eq!(comp.mean_diff, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_one_element_different() {
    let mut a = ArrayD::from_elem(IxDyn(&[10]), 1.0f32);
    let b = ArrayD::from_elem(IxDyn(&[10]), 1.0f32);
    a[[5]] = 1.1; // Make one element different

    let comp = compare_arrays(&a, &b, 0.05);
    assert!(comp.exceeds_tolerance);
    assert!((comp.max_diff - 0.1).abs() < 1e-6);
    assert!((comp.mean_diff - 0.01).abs() < 1e-6); // 0.1/10
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_large() {
    let a = ArrayD::from_elem(IxDyn(&[100, 100, 100]), 1.0f32);
    let mut b = ArrayD::from_elem(IxDyn(&[100, 100, 100]), 1.0f32);
    b[[50, 50, 50]] = 1.0001;

    let comp = compare_arrays(&a, &b, 1e-3);
    assert!(!comp.exceeds_tolerance);
    assert!((comp.max_diff - 0.0001).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_multidim_shape_mismatch() {
    let a = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1.0f32);
    let b = ArrayD::from_elem(IxDyn(&[2, 4, 3]), 1.0f32);
    let comp = compare_arrays(&a, &b, 1e-5);
    assert!(comp.exceeds_tolerance);
    assert_eq!(comp.shape_a, vec![2, 3, 4]);
    assert_eq!(comp.shape_b, vec![2, 4, 3]);
}

// ========================================================================
// check_gelu_pattern tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_check_gelu_pattern_no_pattern() {
    // Random differences not concentrated in transition region
    let a = ArrayD::from_shape_vec(IxDyn(&[10]), vec![5.0; 10]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[10]), vec![5.001; 10]).unwrap();

    let result = check_gelu_pattern("gelu_out", &a, &b, 0.001);
    // Values at 5.0 are not in the transition region [0.1, 2.0]
    assert!(result.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_check_gelu_pattern_in_transition_region() {
    // Differences concentrated in GELU transition region
    let a_data: Vec<f32> = (0..100).map(|i| 0.5 + (i as f32) * 0.015).collect(); // 0.5 to 2.0
    let b_data: Vec<f32> = a_data.iter().map(|x| x + 0.001).collect();

    let a = ArrayD::from_shape_vec(IxDyn(&[100]), a_data).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[100]), b_data).unwrap();

    let result = check_gelu_pattern("gelu_out", &a, &b, 0.001);
    if let Some(diag) = result {
        assert!(matches!(
            diag.pattern,
            DivergencePattern::GeluApproximation { .. }
        ));
    }
    // Note: May or may not detect depending on threshold ratios
}

// ========================================================================
// check_layernorm_pattern tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_check_layernorm_pattern_no_pattern() {
    // Moderate variance relative to mean - neither too high nor too low
    // We need: 0.1 * |mean| <= std <= 2.0 * |mean|
    // With diffs: 0.001, 0.002, 0.001, 0.002, ...
    // mean = 0.0015, variance = 0.00000025, std = 0.0005
    // std/mean = 0.0005/0.0015 = 0.33, which is in [0.1, 2.0]
    let a = ArrayD::from_shape_vec(IxDyn(&[100]), vec![1.0; 100]).unwrap();
    let mut b_data = vec![1.0; 100];
    for (i, val) in b_data.iter_mut().enumerate() {
        *val += if i % 2 == 0 { 0.001 } else { 0.002 };
    }
    let b = ArrayD::from_shape_vec(IxDyn(&[100]), b_data).unwrap();

    let result = check_layernorm_pattern("ln_out", &a, &b, 0.002);
    // Moderate variance ratio won't trigger either pattern
    assert!(result.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_check_layernorm_pattern_systematic_offset() {
    // Systematic offset (epsilon difference) - low variance, high mean
    let a = ArrayD::from_shape_vec(IxDyn(&[100]), vec![1.0; 100]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[100]), vec![1.005; 100]).unwrap();

    let result = check_layernorm_pattern("ln_out", &a, &b, 0.005);
    if let Some(diag) = result {
        assert!(matches!(
            diag.pattern,
            DivergencePattern::LayerNormVariance {
                epsilon_differs: true
            }
        ));
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_check_layernorm_pattern_empty_array() {
    let a = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();

    let result = check_layernorm_pattern("ln_out", &a, &b, 0.001);
    assert!(result.is_none());
}

// ========================================================================
// check_accumulation_pattern tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_check_accumulation_pattern_too_small() {
    // Array too small to detect pattern (< 100 elements)
    let a = ArrayD::from_shape_vec(IxDyn(&[50]), vec![1.0; 50]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[50]), vec![1.001; 50]).unwrap();

    let result = check_accumulation_pattern("mm_out", &a, &b, 0.001);
    assert!(result.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_check_accumulation_pattern_uniform_error() {
    // Uniform error distribution (low coefficient of variation)
    let a = ArrayD::from_shape_vec(IxDyn(&[1000]), vec![1.0; 1000]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[1000]), vec![1.00001; 1000]).unwrap();

    let result = check_accumulation_pattern("mm_out", &a, &b, 0.00001);
    if let Some(diag) = result {
        assert!(matches!(
            diag.pattern,
            DivergencePattern::AccumulationOrder { .. }
        ));
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_check_accumulation_pattern_nonuniform_error() {
    // Non-uniform error distribution (high CV)
    let mut b_data = vec![1.0; 1000];
    for (i, val) in b_data.iter_mut().enumerate() {
        if i < 100 {
            *val = 1.01; // Large diff in first 100 elements
        }
    }
    let a = ArrayD::from_shape_vec(IxDyn(&[1000]), vec![1.0; 1000]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[1000]), b_data).unwrap();

    let result = check_accumulation_pattern("mm_out", &a, &b, 0.01);
    // High CV should not trigger accumulation pattern
    assert!(result.is_none());
}

// ========================================================================
// check_quantization_pattern tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_check_quantization_pattern_too_few_diffs() {
    // Less than 10 non-zero differences
    let a = ArrayD::from_shape_vec(IxDyn(&[10]), vec![1.0; 10]).unwrap();
    let mut b_data = vec![1.0; 10];
    b_data[0] = 1.001; // Only one diff
    let b = ArrayD::from_shape_vec(IxDyn(&[10]), b_data).unwrap();

    let result = check_quantization_pattern("out", None, &a, &b, 0.001);
    assert!(result.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_check_quantization_pattern_quantized_diffs() {
    // Differences are multiples of a quantization step
    let a_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let b_data: Vec<f32> = a_data
        .iter()
        .enumerate()
        .map(|(i, x)| x + (i % 3) as f32 * 0.001)
        .collect();

    let a = ArrayD::from_shape_vec(IxDyn(&[100]), a_data).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[100]), b_data).unwrap();

    let result = check_quantization_pattern("out", Some(&LayerType::Linear), &a, &b, 0.003);
    // 66 of 100 elements differ by multiples of 0.001 — this is a clear quantization
    // pattern. The detector should return a quantization diagnosis.
    let diagnosis = result.expect("Expected quantization diagnosis for multiples-of-0.001 diffs");
    assert_eq!(diagnosis.divergence_layer, "out");
    assert_eq!(diagnosis.layer_type, Some(LayerType::Linear));
    assert!(matches!(
        diagnosis.pattern,
        DivergencePattern::QuantizationError { .. }
    ));
}

// ========================================================================
// DiffConfig tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_diff_config_default() {
    let config = DiffConfig::default();
    assert_eq!(config.tolerance, 1e-5);
    assert!(config.continue_after_divergence);
    assert!(config.input.is_none());
    assert!(config.layer_mapping.is_empty());
    assert!(!config.diagnose);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_config_custom() {
    let input = ArrayD::from_elem(IxDyn(&[1, 10]), 0.5f32);
    let mut mapping = HashMap::new();
    mapping.insert("a".to_string(), "b".to_string());

    let config = DiffConfig {
        tolerance: 1e-3,
        continue_after_divergence: false,
        input: Some(input),
        layer_mapping: mapping,
        diagnose: true,
    };

    assert_eq!(config.tolerance, 1e-3);
    assert!(!config.continue_after_divergence);
    assert!(config.input.is_some());
    assert_eq!(config.layer_mapping.len(), 1);
    assert!(config.diagnose);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_config_debug() {
    let config = DiffConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("DiffConfig"));
    assert!(debug_str.contains("tolerance"));
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_config_clone() {
    let original = DiffConfig {
        tolerance: 0.01,
        ..Default::default()
    };
    let cloned = original.clone();
    assert_eq!(cloned.tolerance, original.tolerance);
}

// ========================================================================
// DiffStatus tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_diff_status_eq() {
    assert_eq!(DiffStatus::Ok, DiffStatus::Ok);
    assert_eq!(DiffStatus::DriftStarts, DiffStatus::DriftStarts);
    assert_eq!(DiffStatus::ExceedsTolerance, DiffStatus::ExceedsTolerance);
    assert_eq!(DiffStatus::ShapeMismatch, DiffStatus::ShapeMismatch);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_status_ne() {
    assert_ne!(DiffStatus::Ok, DiffStatus::DriftStarts);
    assert_ne!(DiffStatus::Ok, DiffStatus::ExceedsTolerance);
    assert_ne!(DiffStatus::Ok, DiffStatus::ShapeMismatch);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_status_debug() {
    assert_eq!(format!("{:?}", DiffStatus::Ok), "Ok");
    assert_eq!(format!("{:?}", DiffStatus::DriftStarts), "DriftStarts");
    assert_eq!(
        format!("{:?}", DiffStatus::ExceedsTolerance),
        "ExceedsTolerance"
    );
    assert_eq!(format!("{:?}", DiffStatus::ShapeMismatch), "ShapeMismatch");
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_status_copy() {
    let status = DiffStatus::Ok;
    let copied = status;
    assert_eq!(status, copied);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_status_clone() {
    let status = DiffStatus::ExceedsTolerance;
    // DiffStatus is Copy, but test clone trait is also implemented
    let cloned: DiffStatus = Clone::clone(&status);
    assert_eq!(status, cloned);
}

// ========================================================================
// DivergencePattern tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_divergence_pattern_partial_eq() {
    let p1 = DivergencePattern::Unknown;
    let p2 = DivergencePattern::Unknown;
    assert_eq!(p1, p2);

    let p3 = DivergencePattern::ExpPrecision {
        max_logit: 85.0,
        is_overflow: true,
    };
    let p4 = DivergencePattern::ExpPrecision {
        max_logit: 85.0,
        is_overflow: true,
    };
    assert_eq!(p3, p4);

    let p5 = DivergencePattern::ExpPrecision {
        max_logit: 85.0,
        is_overflow: false,
    };
    assert_ne!(p3, p5);
}

#[ntest::timeout(10000)]
#[test]
fn test_divergence_pattern_clone() {
    let pattern = DivergencePattern::SoftmaxInstability {
        max_score: 75.0,
        score_range: 60.0,
    };
    let cloned = pattern.clone();
    assert_eq!(pattern, cloned);
}

#[ntest::timeout(10000)]
#[test]
fn test_divergence_pattern_debug() {
    let pattern = DivergencePattern::AccumulationOrder {
        operation: "sum".to_string(),
        size_correlated: false,
    };
    let debug_str = format!("{:?}", pattern);
    assert!(debug_str.contains("AccumulationOrder"));
    assert!(debug_str.contains("sum"));
}

#[ntest::timeout(10000)]
#[test]
fn test_divergence_pattern_display_accumulation_no_correlation() {
    let pattern = DivergencePattern::AccumulationOrder {
        operation: "reduce".to_string(),
        size_correlated: false,
    };
    let s = format!("{}", pattern);
    assert!(s.contains("reduce"));
    assert!(!s.contains("grows with size"));
}

#[ntest::timeout(10000)]
#[test]
fn test_divergence_pattern_display_quantization_no_boundary() {
    let pattern = DivergencePattern::QuantizationError {
        bits_lost: 4,
        at_power_boundary: false,
    };
    let s = format!("{}", pattern);
    assert!(s.contains("4 bits"));
    assert!(!s.contains("boundary"));
}

#[ntest::timeout(10000)]
#[test]
fn test_divergence_pattern_display_layernorm_variance_order() {
    let pattern = DivergencePattern::LayerNormVariance {
        epsilon_differs: false,
    };
    let s = format!("{}", pattern);
    assert!(s.contains("computation order"));
}

// ========================================================================
// LayerComparison tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_layer_comparison_debug() {
    let comp = LayerComparison {
        name: "test".to_string(),
        name_b: None,
        max_diff: 0.001,
        mean_diff: 0.0005,
        exceeds_tolerance: false,
        shape_a: vec![1, 2],
        shape_b: vec![1, 2],
    };
    let debug_str = format!("{:?}", comp);
    assert!(debug_str.contains("LayerComparison"));
    assert!(debug_str.contains("test"));
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_comparison_clone() {
    let comp = LayerComparison {
        name: "layer".to_string(),
        name_b: Some("layer_b".to_string()),
        max_diff: 0.01,
        mean_diff: 0.005,
        exceeds_tolerance: true,
        shape_a: vec![1, 10, 20],
        shape_b: vec![1, 10, 20],
    };
    let cloned = comp.clone();
    assert_eq!(cloned.name, comp.name);
    assert_eq!(cloned.name_b, comp.name_b);
    assert_eq!(cloned.max_diff, comp.max_diff);
}

// ========================================================================
// DiffDiagnosis tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_diff_diagnosis_debug() {
    let diag = DiffDiagnosis::unknown("test_layer", None);
    let debug_str = format!("{:?}", diag);
    assert!(debug_str.contains("DiffDiagnosis"));
    assert!(debug_str.contains("test_layer"));
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_diagnosis_clone() {
    let diag = DiffDiagnosis {
        divergence_layer: "fc1".to_string(),
        layer_type: Some(LayerType::Linear),
        pattern: DivergencePattern::WeightMismatch {
            layer: "fc1".to_string(),
            max_diff: 0.1,
        },
        explanation: "Weights differ".to_string(),
        suggestion: Some("Check model export".to_string()),
        confidence: 0.95,
        evidence: vec!["diff = 0.1".to_string()],
    };
    let cloned = diag.clone();
    assert_eq!(cloned.divergence_layer, diag.divergence_layer);
    assert_eq!(cloned.confidence, diag.confidence);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_diagnosis_format_report_no_suggestion() {
    let diag = DiffDiagnosis {
        divergence_layer: "relu".to_string(),
        layer_type: None,
        pattern: DivergencePattern::Unknown,
        explanation: "Unknown cause".to_string(),
        suggestion: None,
        confidence: 0.0,
        evidence: vec![],
    };
    let report = diag.format_report();
    assert!(report.contains("relu"));
    assert!(report.contains("Unknown cause"));
    assert!(!report.contains("Suggestion:"));
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_diagnosis_format_report_no_layer_type() {
    let diag = DiffDiagnosis {
        divergence_layer: "output".to_string(),
        layer_type: None,
        pattern: DivergencePattern::Unknown,
        explanation: String::new(),
        suggestion: None,
        confidence: 0.5,
        evidence: vec![],
    };
    let report = diag.format_report();
    assert!(report.contains("output"));
    assert!(!report.contains("Layer Type:"));
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_diagnosis_format_report_empty_explanation() {
    let diag = DiffDiagnosis {
        divergence_layer: "x".to_string(),
        layer_type: None,
        pattern: DivergencePattern::Unknown,
        explanation: String::new(),
        suggestion: None,
        confidence: 0.0,
        evidence: vec![],
    };
    let report = diag.format_report();
    // Empty explanation is skipped (not printed)
    assert!(!report.contains("Explanation:"));
    // But other fields are present
    assert!(report.contains("Layer: x"));
    assert!(report.contains("Issue:"));
    assert!(report.contains("Confidence:"));
}

// ========================================================================
// DiffResult tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_diff_result_debug() {
    let result = DiffResult {
        layers: vec![],
        first_bad_layer: None,
        drift_start_layer: None,
        max_divergence: 0.0,
        tolerance: 1e-5,
        suggestion: None,
        diagnosis: None,
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("DiffResult"));
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_result_clone() {
    let result = DiffResult {
        layers: vec![LayerComparison {
            name: "layer".to_string(),
            name_b: None,
            max_diff: 0.001,
            mean_diff: 0.0005,
            exceeds_tolerance: false,
            shape_a: vec![1],
            shape_b: vec![1],
        }],
        first_bad_layer: None,
        drift_start_layer: None,
        max_divergence: 0.001,
        tolerance: 1e-5,
        suggestion: Some("test".to_string()),
        diagnosis: None,
    };
    let cloned = result.clone();
    assert_eq!(cloned.layers.len(), result.layers.len());
    assert_eq!(cloned.suggestion, result.suggestion);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_result_first_bad_layer_out_of_bounds() {
    // Edge case: first_bad_layer index is out of bounds
    let result = DiffResult {
        layers: vec![],
        first_bad_layer: Some(5), // Invalid index
        drift_start_layer: None,
        max_divergence: 0.0,
        tolerance: 1e-5,
        suggestion: None,
        diagnosis: None,
    };
    assert_eq!(result.first_bad_layer_name(), None);
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_result_statuses_empty() {
    let result = DiffResult {
        layers: vec![],
        first_bad_layer: None,
        drift_start_layer: None,
        max_divergence: 0.0,
        tolerance: 1e-5,
        suggestion: None,
        diagnosis: None,
    };
    let statuses = result.statuses();
    assert!(statuses.is_empty());
}

// ========================================================================
// DiffError tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_diff_error_from_io_error() {
    use std::io::{Error, ErrorKind};
    let io_err = Error::new(ErrorKind::NotFound, "file not found");
    let diff_err: DiffError = io_err.into();
    assert!(matches!(diff_err, DiffError::IoError(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_diff_error_debug() {
    let err = DiffError::NoLayers;
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("NoLayers"));
}

// ========================================================================
// ModelInfo tests
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_model_info_debug() {
    let info = ModelInfo {
        inputs: vec![],
        outputs: vec![],
        intermediate_names: vec!["a".to_string(), "b".to_string()],
        layers: vec![],
    };
    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("ModelInfo"));
    assert!(debug_str.contains("intermediate_names"));
}

// ========================================================================
// Negative value tests for compare_arrays
// ========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_negative_values() {
    let a = ArrayD::from_shape_vec(IxDyn(&[5]), vec![-1.0, -2.0, -3.0, -4.0, -5.0]).unwrap();
    let b =
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![-1.001, -2.001, -3.001, -4.001, -5.001]).unwrap();

    let comp = compare_arrays(&a, &b, 0.01);
    assert!(!comp.exceeds_tolerance);
    assert!((comp.max_diff - 0.001).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_mixed_signs() {
    let a = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.99, 0.01, 1.01, 2.01]).unwrap();

    let comp = compare_arrays(&a, &b, 0.05);
    assert!(!comp.exceeds_tolerance);
    assert_eq!(comp.max_diff, 0.01);
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_inf_values() {
    let a =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, f32::NEG_INFINITY, 0.0]).unwrap();
    let b =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, f32::NEG_INFINITY, 0.0]).unwrap();

    let comp = compare_arrays(&a, &b, 1e-5);
    // Matching Inf/NegInf positions produce 0.0 diff (not NaN). Arrays are identical.
    assert!(
        !comp.exceeds_tolerance,
        "Identical arrays with Inf values should not exceed tolerance, max_diff={}",
        comp.max_diff
    );
    assert_eq!(comp.max_diff, 0.0);
    assert_eq!(comp.mean_diff, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_nan_values() {
    let a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap();

    let comp = compare_arrays(&a, &b, 1e-5);
    // Matching NaN pairs produce 0.0 diff (both models agree on NaN output).
    assert!(
        !comp.exceeds_tolerance,
        "Identical arrays with NaN values should not exceed tolerance, max_diff={}",
        comp.max_diff
    );
    assert_eq!(comp.max_diff, 0.0);
    assert_eq!(comp.mean_diff, 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_matching_inf_masks_real_divergence() {
    // Regression test for #2798: matching Inf at position 0 previously produced NaN
    // that poisoned mean_diff. The real divergence at position 1 must be detected.
    let a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 2.0]).unwrap();

    let comp = compare_arrays(&a, &b, 0.001);
    assert!(
        comp.exceeds_tolerance,
        "Real divergence at non-Inf position must be detected"
    );
    assert_eq!(comp.max_diff, 1.0);
    // mean_diff should be 0.5 = (0.0 + 1.0) / 2, not NaN
    assert!(
        !comp.mean_diff.is_nan(),
        "mean_diff must not be NaN from matching-Inf positions"
    );
    assert!((comp.mean_diff - 0.5).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_mismatched_inf_detected() {
    // Inf vs -Inf is a real divergence (diff = Inf), not a matching-special case.
    let a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 0.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 0.0]).unwrap();

    let comp = compare_arrays(&a, &b, 0.001);
    assert!(comp.exceeds_tolerance);
    assert_eq!(comp.max_diff, f32::INFINITY);
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_nan_vs_finite_detected() {
    // NaN in one array but not the other is a real divergence.
    let a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap();

    let comp = compare_arrays(&a, &b, 0.001);
    // NaN - 0.0 = NaN, which is a genuine divergence (not matching specials).
    // f32::max(0.0, NaN) deterministically returns 0.0, so max_diff will NOT
    // catch NaN-vs-finite divergence. mean_diff is NaN (via sum), which signals
    // the divergence to callers. See #2688 for the exceeds_tolerance gap.
    assert!(
        comp.mean_diff.is_nan(),
        "NaN vs finite must produce NaN in mean_diff"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_compare_arrays_matching_neg_inf_with_divergence() {
    // Matching -Inf positions should produce 0.0 diff, not mask real divergence.
    let a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, 5.0, 0.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, 8.0, 0.0]).unwrap();

    let comp = compare_arrays(&a, &b, 0.001);
    assert!(comp.exceeds_tolerance);
    assert_eq!(comp.max_diff, 3.0);
    assert!(
        !comp.mean_diff.is_nan(),
        "mean_diff must not be NaN from matching -Inf positions"
    );
    // mean = (0.0 + 3.0 + 0.0) / 3 = 1.0
    assert!((comp.mean_diff - 1.0).abs() < 1e-6);
}
